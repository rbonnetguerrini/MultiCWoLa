"""Experiment orchestration."""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from multiclass_cwola.baselines.demix_alg4 import run_demix_alg4_input, run_demix_alg4_posterior
from multiclass_cwola.baselines.known_prior import run_known_prior_oracle
from multiclass_cwola.baselines.oracle import run_oracle_supervised
from multiclass_cwola.baselines.pairwise import run_source_ovr_baseline
from multiclass_cwola.baselines.wei import run_wei_ccm
from multiclass_cwola.calibration.posthoc import calibrate_logits, fit_calibrator
from multiclass_cwola.data.real import build_real_dataset
from multiclass_cwola.data.synthetic import build_synthetic_dataset
from multiclass_cwola.evaluation.matching import align_probabilities, match_label_permutation
from multiclass_cwola.evaluation.metrics import (
    basic_classification_metrics,
    mixing_matrix_recovery_error,
    posterior_l1_error,
    runtime_memory_summary,
    simplex_geometric_distance,
    vertex_recovery_error,
)
from multiclass_cwola.models.backbones import build_classifier
from multiclass_cwola.simplex.fitters import fit_simplex, transform_simplex_points
from multiclass_cwola.simplex.projection import batch_simplex_least_squares
from multiclass_cwola.training.trainer import infer_classifier, train_classifier
from multiclass_cwola.utils.io import save_config, save_dataframe
from multiclass_cwola.utils.repro import (
    choose_device,
    ensure_dir,
    environment_summary,
    save_json,
    save_text,
    seed_everything,
    setup_logging,
)
from multiclass_cwola.visualization.plots import (
    plot_kspace_scatter,
    plot_mspace_simplex,
    plot_metric_curve,
    plot_pi_heatmap,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class RunArtifacts:
    """Outputs from one experiment run."""

    metrics: dict[str, Any]
    metrics_frame: pd.DataFrame
    output_dir: Path


def _config_to_dict(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return deepcopy(config)


def _resolve_output_dir(config: dict[str, Any] | None = None) -> Path:
    """Use an explicit override when present, else Hydra's runtime output directory."""
    if config is not None and config.get("output_dir_override"):
        return ensure_dir(Path(str(config["output_dir_override"])))
    try:
        return ensure_dir(HydraConfig.get().runtime.output_dir)
    except Exception:
        return ensure_dir(Path.cwd())


def _apply_named_experiment_overrides(config: dict[str, Any]) -> dict[str, Any]:
    name = str(config["experiment"]["name"])
    data_cfg = config["data"]
    if name == "nonidentifiable_rank_deficient":
        data_cfg["pi_mode"] = "near_rank_deficient"
        data_cfg["condition_strength"] = 0.02
        data_cfg["pi_similarity"] = 0.99
    elif name == "nonidentifiable_M_lt_K":
        data_cfg["M"] = min(int(data_cfg["M"]), max(2, int(data_cfg["K"]) - 1))
        data_cfg["pi_mode"] = "dirichlet"
        data_cfg["condition_strength"] = 0.1
    elif name == "shift_sweep":
        data_cfg["shift"]["enabled"] = True
        if data_cfg["shift"]["type"] == "none":
            data_cfg["shift"]["type"] = "spurious"
    return config


def _build_bundle(config: dict[str, Any]):
    if config["data"]["kind"] == "synthetic":
        return build_synthetic_dataset(config["data"])
    return build_real_dataset(config["data"])


def _adapt_model_config_for_bundle(config: dict[str, Any], bundle) -> None:
    """Apply data-dependent model defaults before persisting the resolved config."""
    if bundle.task_type == "image" and str(config["model"]["backbone"]) == "mlp":
        config["model"]["backbone"] = "cnn"


def _aligned_metrics(
    val_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    num_bins: int,
) -> tuple[dict[str, float], dict[int, int], np.ndarray]:
    val_predictions = val_probabilities.argmax(axis=1)
    _, mapping = match_label_permutation(val_labels, val_predictions, num_classes=num_classes)
    aligned_test = align_probabilities(test_probabilities, mapping, num_classes=num_classes)
    return (
        basic_classification_metrics(test_labels, aligned_test, num_bins=num_bins),
        mapping,
        aligned_test,
    )


def _safe_aligned_metrics(
    val_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    num_bins: int,
) -> tuple[dict[str, float], dict[int, int], np.ndarray]:
    if not np.all(np.isfinite(val_probabilities)) or not np.all(np.isfinite(test_probabilities)):
        metrics = {"accuracy": np.nan, "macro_f1": np.nan, "log_loss": np.nan, "ece": np.nan}
        return metrics, {}, test_probabilities
    return _aligned_metrics(
        val_probabilities,
        test_probabilities,
        val_labels,
        test_labels,
        num_classes=num_classes,
        num_bins=num_bins,
    )


def _selected_method_name(cfg: dict[str, Any]) -> str | None:
    method_cfg = cfg.get("method")
    if method_cfg is None:
        return None
    if isinstance(method_cfg, str):
        return method_cfg
    if isinstance(method_cfg, dict):
        value = method_cfg.get("method")
        return None if value is None else str(value)
    return None


def _simplex_method_names(cfg: dict[str, Any]) -> tuple[str, str]:
    simplex_cfg = cfg.get("simplex") or {}
    fitter_name = simplex_cfg.get("name") or simplex_cfg.get("method") or "simplex"
    method_code_name = simplex_cfg.get("method") or fitter_name
    return str(fitter_name), str(method_code_name)


def _class_alignment_array(mapping: dict[int, int], num_classes: int) -> np.ndarray:
    alignment = np.full(num_classes, -1, dtype=int)
    for fitted_k, true_k in mapping.items():
        if 0 <= int(fitted_k) < num_classes:
            alignment[int(fitted_k)] = int(true_k)
    return alignment


def _demix_method_config(cfg: dict[str, Any], method_name: str) -> dict[str, Any]:
    method_cfg = cfg.get("method")
    if isinstance(method_cfg, dict):
        demix_cfg = dict(method_cfg)
    else:
        demix_cfg = {"method": method_name}
    demix_cfg.setdefault("random_state", int(cfg["seed"]))
    return demix_cfg


def _load_simplex_config(name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[3] / "configs" / "simplex" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown simplex config: {name}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _low_rank_train_cfg(
    base_train_cfg: dict[str, Any],
    low_rank_cfg: dict[str, Any],
    *,
    num_classes: int,
) -> dict[str, Any]:
    train_cfg = deepcopy(base_train_cfg)
    centered = bool(low_rank_cfg.get("centered", True))
    target_rank = low_rank_cfg.get("target_rank", "auto")
    if target_rank in (None, "auto"):
        target_rank = num_classes - 1 if centered else num_classes
    train_cfg["low_rank_regularization"] = {
        "enabled": True,
        "weight": float(low_rank_cfg.get("weight", 0.0)),
        "centered": centered,
        "target_rank": int(target_rank),
        "num_classes": int(num_classes),
        "normalize_by": str(low_rank_cfg.get("normalize_by", "sqrt_batch")),
    }
    return train_cfg


def _save_minimal_run_outputs(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    bundle,
    metrics: dict[str, Any],
    run_start: float,
    summary_method_name: str,
) -> RunArtifacts:
    metrics |= bundle.metadata
    metrics |= runtime_memory_summary(run_start)
    metrics_frame = pd.DataFrame([metrics])
    save_json(output_dir / "metrics.json", metrics)
    save_dataframe(output_dir / "metrics.csv", metrics_frame)
    save_text(
        output_dir / "summary.txt",
        "\n".join(
            [
                f"Experiment: {cfg['experiment']['name']}",
                f"Task: {cfg['task']}",
                f"Seed: {cfg['seed']}",
                f"Method: {summary_method_name}",
                f"Method accuracy: {metrics.get('method_accuracy', 'N/A')}",
                f"Method macro F1: {metrics.get('method_macro_f1', 'N/A')}",
                f"Environment: {environment_summary()}",
            ]
        )
        + "\n",
    )
    LOGGER.info("finished run at %s", output_dir)
    return RunArtifacts(metrics=metrics, metrics_frame=metrics_frame, output_dir=output_dir)


def run_experiment(config: DictConfig | dict[str, Any]) -> RunArtifacts:
    """Run one configured experiment end to end."""
    setup_logging()
    run_start = time.perf_counter()
    cfg = _apply_named_experiment_overrides(_config_to_dict(config))
    output_dir = _resolve_output_dir(cfg)
    ensure_dir(output_dir / "plots")
    ensure_dir(output_dir / "checkpoints")
    seed_everything(int(cfg["seed"]))
    device = choose_device(str(cfg["train"].get("device", "auto")))
    bundle = _build_bundle(cfg)
    _adapt_model_config_for_bundle(cfg, bundle)
    reuse_checkpoints = bool(cfg.get("reuse_checkpoints", False))

    save_config(output_dir / "resolved_config.yaml", OmegaConf.create(cfg))
    save_json(output_dir / "mixture_metadata.json", bundle.metadata | {"pi": bundle.pi.tolist()})

    selected_method = _selected_method_name(cfg)
    if selected_method == "demix_alg4_input":
        demix_result = run_demix_alg4_input(
            train_x=bundle.train.x,
            train_source=bundle.train.source,
            val_x=bundle.val.x,
            test_x=bundle.test.x,
            num_sources=bundle.num_sources,
            num_classes=bundle.num_classes,
            config=_demix_method_config(cfg, selected_method),
        )
        demix_metrics, _, _ = _safe_aligned_metrics(
            demix_result.val_probabilities,
            demix_result.test_probabilities,
            bundle.val.y,
            bundle.test.y,
            num_classes=bundle.num_classes,
            num_bins=int(cfg["eval"]["num_bins"]),
        )
        metrics = {
            "experiment": cfg["experiment"]["name"],
            "seed": int(cfg["seed"]),
            "task": cfg["task"],
            "num_sources": bundle.num_sources,
            "num_classes": bundle.num_classes,
            "selected_method": selected_method,
            "method_label": "Demix Alg. 4 input",
            "method_information_level": "fair",
            "method_accuracy": demix_metrics["accuracy"],
            "method_macro_f1": demix_metrics["macro_f1"],
            "method_log_loss": demix_metrics["log_loss"],
            "method_ece": demix_metrics["ece"],
        }
        metrics |= {f"demix_alg4_input_{key}": value for key, value in demix_metrics.items()}
        save_json(output_dir / "demix_details.json", {"demix_alg4_input": demix_result.diagnostics})
        return _save_minimal_run_outputs(
            cfg=cfg,
            output_dir=output_dir,
            bundle=bundle,
            metrics=metrics,
            run_start=run_start,
            summary_method_name="demix_alg4_input",
        )

    source_model = build_classifier(
        cfg["model"],
        input_shape=bundle.input_shape,
        num_outputs=bundle.num_sources,
        num_classes=bundle.num_classes,
        enable_bottleneck=True,
    )
    init_ckpt = cfg.get("init_source_checkpoint")
    if init_ckpt:
        import torch as _torch
        state = _torch.load(str(init_ckpt), map_location="cpu")
        missing, unexpected = source_model.load_state_dict(state, strict=False)
        _log = logging.getLogger(__name__)
        if missing:
            _log.warning("init_source_checkpoint missing keys: %s", missing)
        if unexpected:
            _log.warning("init_source_checkpoint unexpected keys (dropped): %s", unexpected)
    source_history = train_classifier(
        source_model,
        train_x=bundle.train.x,
        train_labels=bundle.train.source,
        val_x=bundle.val.x,
        val_labels=bundle.val.source,
        train_cfg=cfg["train"],
        checkpoint_path=(
            output_dir / "checkpoints" / "source_model.pt" if cfg["save_checkpoints"] else None
        ),
        reuse_checkpoint=reuse_checkpoints,
        device=device,
        num_classes_K=int(bundle.num_classes),
    )
    train_out = infer_classifier(
        source_model, bundle.train.x, int(cfg["train"]["batch_size"]), device
    )
    val_out = infer_classifier(source_model, bundle.val.x, int(cfg["train"]["batch_size"]), device)
    test_out = infer_classifier(
        source_model, bundle.test.x, int(cfg["train"]["batch_size"]), device
    )
    source_val_before = basic_classification_metrics(
        bundle.val.source,
        val_out.probabilities,
        num_bins=int(cfg["eval"]["num_bins"]),
    )
    source_test_before = basic_classification_metrics(
        bundle.test.source,
        test_out.probabilities,
        num_bins=int(cfg["eval"]["num_bins"]),
    )

    calibration = fit_calibrator(
        logits=val_out.logits,
        labels=bundle.val.source,
        method=str(cfg["calibration"]["method"]),
        max_iter=int(cfg["calibration"]["max_iter"]),
        lr=float(cfg["calibration"]["lr"]),
    )
    train_g = calibrate_logits(train_out.logits, calibration)
    val_g = calibrate_logits(val_out.logits, calibration)
    test_g = calibrate_logits(test_out.logits, calibration)
    source_val_after = basic_classification_metrics(
        bundle.val.source,
        val_g,
        num_bins=int(cfg["eval"]["num_bins"]),
    )
    source_test_after = basic_classification_metrics(
        bundle.test.source,
        test_g,
        num_bins=int(cfg["eval"]["num_bins"]),
    )

    if selected_method == "demix_alg4_posterior":
        demix_result = run_demix_alg4_posterior(
            train_posteriors=train_g,
            train_source=bundle.train.source,
            val_posteriors=val_g,
            test_posteriors=test_g,
            num_sources=bundle.num_sources,
            num_classes=bundle.num_classes,
            config=_demix_method_config(cfg, selected_method),
        )
        demix_metrics, _, _ = _safe_aligned_metrics(
            demix_result.val_probabilities,
            demix_result.test_probabilities,
            bundle.val.y,
            bundle.test.y,
            num_classes=bundle.num_classes,
            num_bins=int(cfg["eval"]["num_bins"]),
        )
        metrics = {
            "experiment": cfg["experiment"]["name"],
            "seed": int(cfg["seed"]),
            "task": cfg["task"],
            "num_sources": bundle.num_sources,
            "num_classes": bundle.num_classes,
            "selected_method": selected_method,
            "method_label": "Demix Alg. 4 posterior",
            "method_information_level": "fair",
            "source_nll_before": calibration.nll_before,
            "source_nll_after": calibration.nll_after,
            "source_val_accuracy_before": source_val_before["accuracy"],
            "source_val_log_loss_before": source_val_before["log_loss"],
            "source_val_ece_before": source_val_before["ece"],
            "source_val_accuracy_after": source_val_after["accuracy"],
            "source_val_log_loss_after": source_val_after["log_loss"],
            "source_val_ece_after": source_val_after["ece"],
            "source_test_accuracy_before": source_test_before["accuracy"],
            "source_test_log_loss_before": source_test_before["log_loss"],
            "source_test_ece_before": source_test_before["ece"],
            "source_test_accuracy_after": source_test_after["accuracy"],
            "source_test_log_loss_after": source_test_after["log_loss"],
            "source_test_ece_after": source_test_after["ece"],
            "method_accuracy": demix_metrics["accuracy"],
            "method_macro_f1": demix_metrics["macro_f1"],
            "method_log_loss": demix_metrics["log_loss"],
            "method_ece": demix_metrics["ece"],
            "source_train_runtime_seconds": source_history.runtime_seconds,
        }
        metrics |= {f"demix_alg4_posterior_{key}": value for key, value in demix_metrics.items()}
        save_json(output_dir / "demix_details.json", {"demix_alg4_posterior": demix_result.diagnostics})
        return _save_minimal_run_outputs(
            cfg=cfg,
            output_dir=output_dir,
            bundle=bundle,
            metrics=metrics,
            run_start=run_start,
            summary_method_name="demix_alg4_posterior",
        )

    simplex_fit = fit_simplex(
        train_g,
        num_vertices=bundle.num_classes,
        config=cfg["simplex"],
        selection_points=val_g,
        seed=int(cfg["seed"]),
    )
    alpha_train = simplex_fit.barycentric_coordinates
    alpha_val = batch_simplex_least_squares(
        transform_simplex_points(val_g, simplex_fit),
        simplex_fit.vertices,
    )
    alpha_test = batch_simplex_least_squares(
        transform_simplex_points(test_g, simplex_fit),
        simplex_fit.vertices,
    )

    simplex_metrics, simplex_mapping, aligned_alpha_test = _aligned_metrics(
        alpha_val,
        alpha_test,
        bundle.val.y,
        bundle.test.y,
        num_classes=bundle.num_classes,
        num_bins=int(cfg["eval"]["num_bins"]),
    )

    metrics: dict[str, Any] = {
        "experiment": cfg["experiment"]["name"],
        "seed": int(cfg["seed"]),
        "task": cfg["task"],
        "num_sources": bundle.num_sources,
        "num_classes": bundle.num_classes,
        "source_nll_before": calibration.nll_before,
        "source_nll_after": calibration.nll_after,
        "source_val_accuracy_before": source_val_before["accuracy"],
        "source_val_log_loss_before": source_val_before["log_loss"],
        "source_val_ece_before": source_val_before["ece"],
        "source_val_accuracy_after": source_val_after["accuracy"],
        "source_val_log_loss_after": source_val_after["log_loss"],
        "source_val_ece_after": source_val_after["ece"],
        "source_test_accuracy_before": source_test_before["accuracy"],
        "source_test_log_loss_before": source_test_before["log_loss"],
        "source_test_ece_before": source_test_before["ece"],
        "source_test_accuracy_after": source_test_after["accuracy"],
        "source_test_log_loss_after": source_test_after["log_loss"],
        "source_test_ece_after": source_test_after["ece"],
        "simplex_reconstruction_error": simplex_fit.diagnostics.reconstruction_error,
        "simplex_raw_reconstruction_error": simplex_fit.diagnostics.raw_reconstruction_error,
        "simplex_vertex_condition": simplex_fit.diagnostics.vertex_condition_number,
        "simplex_min_vertex_distance": simplex_fit.diagnostics.min_vertex_distance,
        "simplex_raw_negative_fraction": simplex_fit.diagnostics.raw_barycentric_negative_fraction,
        "simplex_raw_negative_mass": simplex_fit.diagnostics.raw_barycentric_negative_mass,
        "simplex_raw_sum_error": simplex_fit.diagnostics.raw_barycentric_sum_error,
        "simplex_projection_gap": simplex_fit.diagnostics.barycentric_projection_gap,
        "simplex_mean_barycentric_entropy": simplex_fit.diagnostics.mean_barycentric_entropy,
        "simplex_barycentric_collapse_fraction": simplex_fit.diagnostics.barycentric_collapse_fraction,
        "simplex_selection_score": simplex_fit.diagnostics.selection_score,
        "simplex_selected_candidate": simplex_fit.diagnostics.selected_candidate,
        "simplex_selector": str(cfg["simplex"].get("selector", "restart_only")),
        "simplex_candidate_count": (
            len(simplex_fit.candidate_rows) if simplex_fit.candidate_rows is not None else 0
        ),
        "simplex_cluster_support_fraction": (
            simplex_fit.selection_summary.get("cluster_support_fraction")
            if simplex_fit.selection_summary is not None
            else None
        ),
        "simplex_preprocess": str(cfg["simplex"].get("preprocess", "none")),
        "simplex_degenerate": simplex_fit.diagnostics.degenerate,
        "posterior_l1_error": posterior_l1_error(aligned_alpha_test, bundle.test.true_posteriors),
        "vertex_recovery_error": vertex_recovery_error(
            simplex_fit.vertices, bundle.source_given_class
        ),
        "simplex_oracle_geometric_distance": simplex_geometric_distance(
            simplex_fit.vertices,
            bundle.source_given_class,
        ),
        "simplex_oracle_relative_geometric_distance": simplex_geometric_distance(
            simplex_fit.vertices,
            bundle.source_given_class,
            normalize=True,
        ),
        "mixing_matrix_recovery_error": mixing_matrix_recovery_error(simplex_fit.pi_hat, bundle.pi),
    }
    _is_bottleneck = bool(getattr(source_model, "returns_log_probs", False))
    _method_key = "ours_bottleneck" if _is_bottleneck else "ours_linear"
    metrics |= {
        f"{_method_key}_accuracy": simplex_metrics["accuracy"],
        f"{_method_key}_macro_f1": simplex_metrics["macro_f1"],
        f"{_method_key}_log_loss": simplex_metrics["log_loss"],
        f"{_method_key}_ece": simplex_metrics["ece"],
    }
    metrics["source_train_runtime_seconds"] = source_history.runtime_seconds

    extra_methods = list(cfg.get("extra_simplex_methods", []) or [])
    extra_simplex_results: dict[str, Any] = {}
    for extra_name in extra_methods:
        try:
            extra_cfg = _load_simplex_config(str(extra_name))
        except Exception as exc:  # noqa: BLE001
            metrics[f"extra_{extra_name}_error"] = f"config_load_failed: {exc}"
            continue
        try:
            extra_fit = fit_simplex(
                train_g,
                num_vertices=bundle.num_classes,
                config=extra_cfg,
                selection_points=val_g,
                seed=int(cfg["seed"]),
            )
            extra_alpha_val = batch_simplex_least_squares(
                transform_simplex_points(val_g, extra_fit), extra_fit.vertices
            )
            extra_alpha_test = batch_simplex_least_squares(
                transform_simplex_points(test_g, extra_fit), extra_fit.vertices
            )
            extra_metrics_dict, _, _ = _aligned_metrics(
                extra_alpha_val,
                extra_alpha_test,
                bundle.val.y,
                bundle.test.y,
                num_classes=bundle.num_classes,
                num_bins=int(cfg["eval"]["num_bins"]),
            )
            metrics[f"method_{extra_name}_accuracy"] = extra_metrics_dict["accuracy"]
            metrics[f"method_{extra_name}_macro_f1"] = extra_metrics_dict["macro_f1"]
            metrics[f"method_{extra_name}_log_loss"] = extra_metrics_dict["log_loss"]
            metrics[f"method_{extra_name}_ece"] = extra_metrics_dict["ece"]
            metrics[f"method_{extra_name}_vertex_recovery_error"] = vertex_recovery_error(
                extra_fit.vertices, bundle.source_given_class
            )
            metrics[f"method_{extra_name}_reconstruction_error"] = (
                extra_fit.diagnostics.reconstruction_error
            )
            metrics[f"method_{extra_name}_degenerate"] = extra_fit.diagnostics.degenerate
            extra_simplex_results[extra_name] = extra_fit
        except Exception as exc:  # noqa: BLE001
            metrics[f"method_{extra_name}_error"] = f"fit_failed: {exc}"

    low_rank_cfg = cfg.get("low_rank_source") or {}
    if bool(low_rank_cfg.get("enabled", False)):
        low_rank_name = str(low_rank_cfg.get("method_name", "ours_simplex_lowrank"))
        low_rank_simplex_name = str(low_rank_cfg.get("simplex_config", "archetypal_spread"))
        low_rank_simplex_cfg = _load_simplex_config(low_rank_simplex_name)
        low_rank_model = build_classifier(
            cfg["model"],
            input_shape=bundle.input_shape,
            num_outputs=bundle.num_sources,
            num_classes=bundle.num_classes,
            enable_bottleneck=True,
        )
        low_rank_history = train_classifier(
            low_rank_model,
            train_x=bundle.train.x,
            train_labels=bundle.train.source,
            val_x=bundle.val.x,
            val_labels=bundle.val.source,
            train_cfg=_low_rank_train_cfg(
                cfg["train"],
                low_rank_cfg,
                num_classes=bundle.num_classes,
            ),
            checkpoint_path=(
                output_dir / "checkpoints" / "source_model_lowrank.pt"
                if cfg["save_checkpoints"]
                else None
            ),
            reuse_checkpoint=reuse_checkpoints,
            device=device,
        )
        low_rank_train_out = infer_classifier(
            low_rank_model, bundle.train.x, int(cfg["train"]["batch_size"]), device
        )
        low_rank_val_out = infer_classifier(
            low_rank_model, bundle.val.x, int(cfg["train"]["batch_size"]), device
        )
        low_rank_test_out = infer_classifier(
            low_rank_model, bundle.test.x, int(cfg["train"]["batch_size"]), device
        )
        low_rank_calibration = fit_calibrator(
            logits=low_rank_val_out.logits,
            labels=bundle.val.source,
            method=str(cfg["calibration"]["method"]),
            max_iter=int(cfg["calibration"]["max_iter"]),
            lr=float(cfg["calibration"]["lr"]),
        )
        low_rank_train_g = calibrate_logits(low_rank_train_out.logits, low_rank_calibration)
        low_rank_val_g = calibrate_logits(low_rank_val_out.logits, low_rank_calibration)
        low_rank_test_g = calibrate_logits(low_rank_test_out.logits, low_rank_calibration)
        low_rank_fit = fit_simplex(
            low_rank_train_g,
            num_vertices=bundle.num_classes,
            config=low_rank_simplex_cfg,
            selection_points=low_rank_val_g,
            seed=int(cfg["seed"]),
        )
        low_rank_alpha_val = batch_simplex_least_squares(
            transform_simplex_points(low_rank_val_g, low_rank_fit),
            low_rank_fit.vertices,
        )
        low_rank_alpha_test = batch_simplex_least_squares(
            transform_simplex_points(low_rank_test_g, low_rank_fit),
            low_rank_fit.vertices,
        )
        low_rank_metrics, _, aligned_low_rank_test = _aligned_metrics(
            low_rank_alpha_val,
            low_rank_alpha_test,
            bundle.val.y,
            bundle.test.y,
            num_classes=bundle.num_classes,
            num_bins=int(cfg["eval"]["num_bins"]),
        )
        low_rank_train_reg_cfg = _low_rank_train_cfg(
            cfg["train"],
            low_rank_cfg,
            num_classes=bundle.num_classes,
        )["low_rank_regularization"]
        metrics[f"{low_rank_name}_accuracy"] = low_rank_metrics["accuracy"]
        metrics[f"{low_rank_name}_macro_f1"] = low_rank_metrics["macro_f1"]
        metrics[f"{low_rank_name}_log_loss"] = low_rank_metrics["log_loss"]
        metrics[f"{low_rank_name}_ece"] = low_rank_metrics["ece"]
        metrics[f"{low_rank_name}_posterior_l1_error"] = posterior_l1_error(
            aligned_low_rank_test,
            bundle.test.true_posteriors,
        )
        metrics[f"{low_rank_name}_vertex_recovery_error"] = vertex_recovery_error(
            low_rank_fit.vertices,
            bundle.source_given_class,
        )
        metrics[f"{low_rank_name}_simplex_oracle_geometric_distance"] = (
            simplex_geometric_distance(low_rank_fit.vertices, bundle.source_given_class)
        )
        metrics[f"{low_rank_name}_simplex_oracle_relative_geometric_distance"] = (
            simplex_geometric_distance(
                low_rank_fit.vertices,
                bundle.source_given_class,
                normalize=True,
            )
        )
        metrics[f"{low_rank_name}_simplex_reconstruction_error"] = (
            low_rank_fit.diagnostics.reconstruction_error
        )
        metrics[f"{low_rank_name}_simplex_degenerate"] = low_rank_fit.diagnostics.degenerate
        metrics[f"{low_rank_name}_source_nll_before"] = low_rank_calibration.nll_before
        metrics[f"{low_rank_name}_source_nll_after"] = low_rank_calibration.nll_after
        metrics[f"{low_rank_name}_source_train_runtime_seconds"] = (
            low_rank_history.runtime_seconds
        )
        metrics[f"{low_rank_name}_regularization_weight"] = float(
            low_rank_train_reg_cfg["weight"]
        )
        metrics[f"{low_rank_name}_regularization_target_rank"] = int(
            low_rank_train_reg_cfg["target_rank"]
        )
        metrics[f"{low_rank_name}_regularization_centered"] = bool(
            low_rank_train_reg_cfg["centered"]
        )
        metrics[f"{low_rank_name}_regularization_normalize_by"] = str(
            low_rank_train_reg_cfg["normalize_by"]
        )
        metrics[f"{low_rank_name}_simplex_config"] = low_rank_simplex_name
        save_json(
            output_dir / "low_rank_source_details.json",
            {
                "method_name": low_rank_name,
                "simplex_config": low_rank_simplex_name,
                "regularization": low_rank_train_reg_cfg,
                "calibration": {
                    "method": low_rank_calibration.method,
                    "nll_before": low_rank_calibration.nll_before,
                    "nll_after": low_rank_calibration.nll_after,
                },
                "vertices": low_rank_fit.vertices.tolist(),
                "oracle_vertices": bundle.source_given_class.tolist(),
                "diagnostics": {
                    "reconstruction_error": low_rank_fit.diagnostics.reconstruction_error,
                    "raw_reconstruction_error": low_rank_fit.diagnostics.raw_reconstruction_error,
                    "vertex_condition_number": (
                        low_rank_fit.diagnostics.vertex_condition_number
                    ),
                    "min_vertex_distance": low_rank_fit.diagnostics.min_vertex_distance,
                    "barycentric_projection_gap": (
                        low_rank_fit.diagnostics.barycentric_projection_gap
                    ),
                    "mean_barycentric_entropy": (
                        low_rank_fit.diagnostics.mean_barycentric_entropy
                    ),
                    "barycentric_collapse_fraction": (
                        low_rank_fit.diagnostics.barycentric_collapse_fraction
                    ),
                    "selection_score": low_rank_fit.diagnostics.selection_score,
                    "selected_candidate": low_rank_fit.diagnostics.selected_candidate,
                    "degenerate": low_rank_fit.diagnostics.degenerate,
                },
            },
        )

    if cfg["stage2"]["enabled"]:
        stage2_model_cfg = {
            "backbone": "mlp",
            "hidden_dims": cfg["stage2"]["hidden_dims"],
            "embedding_dim": 32,
            "dropout": 0.1,
        }
        if bundle.task_type == "image":
            stage2_model_cfg = {
                "backbone": "cnn",
                "channels": [16, 32],
                "embedding_dim": 64,
                "dropout": 0.1,
            }
        stage2_model = build_classifier(
            stage2_model_cfg, input_shape=bundle.input_shape, num_outputs=bundle.num_classes
        )
        stage2_train_cfg = dict(cfg["train"])
        stage2_train_cfg["epochs"] = int(cfg["stage2"]["epochs"])
        stage2_train_cfg["lr"] = float(cfg["stage2"]["lr"])
        stage2_train_cfg["weight_decay"] = float(cfg["stage2"]["weight_decay"])
        train_classifier(
            stage2_model,
            train_x=bundle.train.x,
            train_labels=alpha_train,
            val_x=bundle.val.x,
            val_labels=alpha_val,
            train_cfg=stage2_train_cfg,
            checkpoint_path=(
                output_dir / "checkpoints" / "stage2_model.pt" if cfg["save_checkpoints"] else None
            ),
            reuse_checkpoint=reuse_checkpoints,
            soft_labels=True,
            device=device,
        )
        stage2_val = infer_classifier(
            stage2_model, bundle.val.x, int(cfg["train"]["batch_size"]), device
        )
        stage2_test = infer_classifier(
            stage2_model, bundle.test.x, int(cfg["train"]["batch_size"]), device
        )
        stage2_metrics, _, _ = _aligned_metrics(
            stage2_val.probabilities,
            stage2_test.probabilities,
            bundle.val.y,
            bundle.test.y,
            num_classes=bundle.num_classes,
            num_bins=int(cfg["eval"]["num_bins"]),
        )
        metrics |= {f"stage2_{key}": value for key, value in stage2_metrics.items()}

    demix_details: dict[str, Any] = {}
    if cfg["run_baselines"]:
        baselines_cfg = cfg.get("baselines") or {}
        run_oracle_baseline = bool(baselines_cfg.get("oracle", True))
        run_ovr_baseline = bool(baselines_cfg.get("ovr", True))

        if run_oracle_baseline:
            oracle_outputs, oracle_history = run_oracle_supervised(
                model_cfg=cfg["model"],
                train_cfg=cfg["train"],
                input_shape=bundle.input_shape,
                train_x=bundle.train.x,
                train_y=bundle.train.y,
                val_x=bundle.val.x,
                val_y=bundle.val.y,
                test_x=bundle.test.x,
                device=device,
                checkpoint_path=(
                    str(output_dir / "checkpoints" / "oracle_model.pt")
                    if cfg["save_checkpoints"]
                    else None
                ),
                reuse_checkpoint=reuse_checkpoints,
            )
            metrics |= {
                f"oracle_supervised_{key}": value
                for key, value in basic_classification_metrics(
                    bundle.test.y, oracle_outputs.probabilities, int(cfg["eval"]["num_bins"])
                ).items()
            }
            metrics["oracle_supervised_train_runtime_seconds"] = oracle_history["runtime_seconds"]

        demix_input_cfg = {
            "method": "demix_alg4_input",
            "label": "KSBS-Demix",
            "information_level": "fair",
            "representation": "input_quantized_histogram",
            "quantizer": "kmeans",
            "num_bins": 128,
            "random_projection_dim": 32,
            "eps": 1e-8,
            "face_eps": 1e-4,
            "max_face_iter": 100,
            "num_face_restarts": 20,
            "num_nonsquare_restarts": 20,
            "prior_mode": "reconstruction",
            "failure_policy": "report_nan",
            "random_state": int(cfg["seed"]),
        }
        demix_input = run_demix_alg4_input(
            train_x=bundle.train.x,
            train_source=bundle.train.source,
            val_x=bundle.val.x,
            test_x=bundle.test.x,
            num_sources=bundle.num_sources,
            num_classes=bundle.num_classes,
            config=demix_input_cfg,
        )
        demix_input_metrics, _, _ = _safe_aligned_metrics(
            demix_input.val_probabilities,
            demix_input.test_probabilities,
            bundle.val.y,
            bundle.test.y,
            num_classes=bundle.num_classes,
            num_bins=int(cfg["eval"]["num_bins"]),
        )
        metrics |= {f"demix_alg4_input_{key}": value for key, value in demix_input_metrics.items()}
        demix_details["demix_alg4_input"] = demix_input.diagnostics

        if run_ovr_baseline:
            ovr_val_soft, ovr_test_soft = run_source_ovr_baseline(
                model_cfg=cfg["model"],
                train_cfg=cfg["train"],
                input_shape=bundle.input_shape,
                train_x=bundle.train.x,
                train_source=bundle.train.source,
                val_x=bundle.val.x,
                val_source=bundle.val.source,
                test_x=bundle.test.x,
                num_sources=bundle.num_sources,
                num_classes=bundle.num_classes,
                device=device,
                seed=int(cfg["seed"]),
                checkpoint_dir=(output_dir / "checkpoints" if cfg["save_checkpoints"] else None),
                reuse_checkpoints=reuse_checkpoints,
            )
            ovr_metrics, _, _ = _aligned_metrics(
                ovr_val_soft,
                ovr_test_soft,
                bundle.val.y,
                bundle.test.y,
                num_classes=bundle.num_classes,
                num_bins=int(cfg["eval"]["num_bins"]),
            )
            metrics |= {f"ovr_{key}": value for key, value in ovr_metrics.items()}

        kp_val = run_known_prior_oracle(val_g, bundle.source_given_class)
        kp_test = run_known_prior_oracle(test_g, bundle.source_given_class)
        kp_metrics, _, _ = _aligned_metrics(
            kp_val,
            kp_test,
            bundle.val.y,
            bundle.test.y,
            num_classes=bundle.num_classes,
            num_bins=int(cfg["eval"]["num_bins"]),
        )
        metrics |= {f"known_prior_demix_{key}": value for key, value in kp_metrics.items()}

        if cfg.get("run_wei_baselines", True):
            wei_cfg = cfg.get("wei") or {}
            wei_train_cfg = dict(cfg["train"])
            if isinstance(wei_cfg, dict):
                train_overrides = wei_cfg.get("train_overrides") or {}
                if isinstance(train_overrides, dict):
                    wei_train_cfg.update(train_overrides)
            for wei_name, wei_runner in (("wei_ccm", run_wei_ccm),):
                runner_kwargs: dict[str, Any] = {}
                if cfg["save_checkpoints"]:
                    runner_kwargs["checkpoint_path"] = output_dir / "checkpoints" / f"{wei_name}_model.pt"
                runner_kwargs["reuse_checkpoint"] = reuse_checkpoints
                wei_result = wei_runner(
                    model_cfg=cfg["model"],
                    train_cfg=wei_train_cfg,
                    input_shape=bundle.input_shape,
                    train_x=bundle.train.x,
                    train_mixture=bundle.train.source,
                    val_x=bundle.val.x,
                    val_mixture=bundle.val.source,
                    test_x=bundle.test.x,
                    theta=bundle.pi,
                    num_classes=bundle.num_classes,
                    device=device,
                    **runner_kwargs,
                )
                wei_metrics, _, _ = _aligned_metrics(
                    wei_result.val_probabilities,
                    wei_result.test_probabilities,
                    bundle.val.y,
                    bundle.test.y,
                    num_classes=bundle.num_classes,
                    num_bins=int(cfg["eval"]["num_bins"]),
                )
                metrics |= {f"{wei_name}_{key}": value for key, value in wei_metrics.items()}
                metrics[f"{wei_name}_runtime_seconds"] = wei_result.runtime_seconds
                metrics[f"{wei_name}_best_epoch"] = int(wei_result.best_epoch)
                metrics[f"{wei_name}_epochs_ran"] = len(wei_result.train_loss)
                metrics[f"{wei_name}_best_val_loss"] = (
                    float(min(wei_result.val_loss)) if wei_result.val_loss else None
                )
                metrics[f"{wei_name}_final_val_loss"] = (
                    float(wei_result.val_loss[-1]) if wei_result.val_loss else None
                )
                metrics[f"{wei_name}_final_train_loss"] = (
                    float(wei_result.train_loss[-1]) if wei_result.train_loss else None
                )

    # K-space plotting uses recovered latent coordinates. Any class-index alignment
    # below is evaluation-only and is not fed back into fitting or decoding.
    is_bottleneck = bool(getattr(source_model, "returns_log_probs", False))
    bn_mapping: dict[int, int] = {}
    bottleneck_alpha_train: np.ndarray | None = None
    bottleneck_alpha_val: np.ndarray | None = None
    bottleneck_alpha_test: np.ndarray | None = None
    bottleneck_vertices_M: np.ndarray | None = None
    bottleneck_pi_hat_MK: np.ndarray | None = None
    bottleneck_alpha_source = "not_bottleneck"
    if is_bottleneck:
        if hasattr(source_model.head, "alpha"):
            with torch.no_grad():
                _emb_train = torch.as_tensor(train_out.embeddings, dtype=torch.float32).to(device)
                _emb_val = torch.as_tensor(val_out.embeddings, dtype=torch.float32).to(device)
                _emb_test = torch.as_tensor(test_out.embeddings, dtype=torch.float32).to(device)
                bottleneck_alpha_train = source_model.head.alpha(_emb_train).cpu().numpy()
                bottleneck_alpha_val = source_model.head.alpha(_emb_val).cpu().numpy()
                bottleneck_alpha_test = source_model.head.alpha(_emb_test).cpu().numpy()
            bottleneck_alpha_source = "head.alpha"
        else:
            with torch.no_grad():
                _pi_col = source_model.head.pi().detach().cpu().numpy()
            from multiclass_cwola.data.common import source_given_class_from_pi
            _verts_k = source_given_class_from_pi(_pi_col)
            bottleneck_alpha_train = batch_simplex_least_squares(train_g, _verts_k)
            bottleneck_alpha_val = batch_simplex_least_squares(val_g, _verts_k)
            bottleneck_alpha_test = batch_simplex_least_squares(test_g, _verts_k)
            bottleneck_alpha_source = "least_squares_fallback"
        if hasattr(source_model.head, "pi"):
            with torch.no_grad():
                bottleneck_pi_hat_MK = source_model.head.pi().detach().cpu().numpy()
            bottleneck_vertices_M = bottleneck_pi_hat_MK.T
        _, bn_mapping = match_label_permutation(
            bundle.val.y,
            bottleneck_alpha_val.argmax(axis=1),
            num_classes=bundle.num_classes,
        )
        alpha_test_k = align_probabilities(
            bottleneck_alpha_test,
            bn_mapping,
            num_classes=bundle.num_classes,
        )
    else:
        alpha_test_k = aligned_alpha_test
    np.save(output_dir / "alpha_test.npy", alpha_test_k)
    if bundle.test.y is not None:
        np.save(output_dir / "test_true_labels.npy", bundle.test.y)

    if cfg["save_plots"]:
        kspace_title = (
            f"Decoded latent posterior --bottleneck (K={bundle.num_classes})"
            if is_bottleneck
            else f"Decoded latent posterior --post-hoc (K={bundle.num_classes})"
        )
        plot_kspace_scatter(
            alpha_test_k,
            bundle.test.y,
            out_path=output_dir / "plots" / "kspace_posterior.png",
            max_points=int(cfg["eval"]["plot_max_points"]),
            title=kspace_title,
        )
        _max_pts = int(cfg["eval"]["plot_max_points"])
        if is_bottleneck and bottleneck_alpha_test is not None and bottleneck_vertices_M is not None:
            bn_reconstructed_g = bottleneck_alpha_test @ bottleneck_vertices_M  # (N, M)
            plot_mspace_simplex(
                bn_reconstructed_g,
                bottleneck_vertices_M,
                None,
                bundle.test.y,
                out_path=output_dir / "plots" / "mspace_posterior.png",
                max_points=_max_pts,
                title=f"Source posterior geometry --bottleneck (M={bundle.num_sources}, K={bundle.num_classes})",
            )
        else:
            plot_mspace_simplex(
                test_g,
                simplex_fit.vertices,
                bundle.source_given_class,
                bundle.test.y,
                out_path=output_dir / "plots" / "mspace_posterior.png",
                max_points=_max_pts,
                title=f"Source posterior geometry --post-hoc (M={bundle.num_sources}, K={bundle.num_classes})",
            )
        curves = pd.DataFrame(
            {
                "epoch": np.arange(1, len(source_history.train_loss) + 1),
                "train_loss": source_history.train_loss,
                "val_loss": source_history.val_loss,
            }
        )
        plot_metric_curve(
            curves,
            x_col="epoch",
            y_col="val_loss",
            out_path=output_dir / "plots" / "source_training_curve.png",
            title="Source Classifier Validation Loss",
        )
        # Pi heatmaps: estimated Pi_hat (row-stochastic, aligned) vs oracle.
        # Bottleneck head.pi() is column-stochastic; convert to row-stochastic via
        # _estimate_pi_from_vertices so both heatmaps use the same convention as bundle.pi.
        if is_bottleneck and hasattr(source_model.head, "pi"):
            with torch.no_grad():
                _pi_col = source_model.head.pi().detach().cpu().numpy()  # (M, K) col-stochastic
            from multiclass_cwola.simplex.fitters import _estimate_pi_from_vertices
            pi_hat_np = _estimate_pi_from_vertices(_pi_col.T)  # vertices (K, M) -> row-stoch (M, K)
            _pi_align_map = bn_mapping
        else:
            pi_hat_np = simplex_fit.pi_hat  # (M, K) row-stochastic from _estimate_pi_from_vertices
            _pi_align_map = simplex_mapping
        # Permute columns to match true class ordering (same alignment applied to alpha).
        if pi_hat_np is not None and _pi_align_map:
            pi_hat_aligned = np.zeros_like(pi_hat_np)
            for pred_k, true_k in _pi_align_map.items():
                if pred_k < pi_hat_np.shape[1] and true_k < pi_hat_np.shape[1]:
                    pi_hat_aligned[:, true_k] = pi_hat_np[:, pred_k]
            pi_hat_np = pi_hat_aligned
        if pi_hat_np is not None:
            plot_pi_heatmap(
                pi_hat_np,
                output_dir / "plots" / "pi_hat_heatmap.png",
                title=f"Pi_hat  (M={bundle.num_sources}, K={bundle.num_classes})",
            )
        plot_pi_heatmap(
            np.asarray(bundle.pi),
            output_dir / "plots" / "pi_oracle_heatmap.png",
            title=f"Pi oracle  (M={bundle.num_sources}, K={bundle.num_classes})",
        )

    metrics |= bundle.metadata
    metrics |= runtime_memory_summary(run_start)
    metrics_frame = pd.DataFrame([metrics])
    if simplex_fit.candidate_rows is not None:
        save_dataframe(output_dir / "simplex_candidates.csv", pd.DataFrame(simplex_fit.candidate_rows))
    if simplex_fit.selection_summary is not None:
        save_json(output_dir / "simplex_selection_summary.json", simplex_fit.selection_summary)
    if demix_details:
        save_json(output_dir / "demix_details.json", demix_details)
    fitter_name, method_code_name = _simplex_method_names(cfg)
    class_alignment = _class_alignment_array(simplex_mapping, bundle.num_classes)
    bottleneck_class_alignment = _class_alignment_array(bn_mapping, bundle.num_classes)
    artifact_arrays: dict[str, np.ndarray] = {
        "fitted_vertices_M": simplex_fit.vertices,
        "decoded_alpha_train": alpha_train,
        "decoded_alpha_val": alpha_val,
        "decoded_alpha_test": alpha_test,
        "decoded_alpha_test_aligned_eval": aligned_alpha_test,
        "fitted_vertex_order": class_alignment,
        "class_alignment": class_alignment,
        "oracle_vertices_M_eval": bundle.source_given_class,
        "pi_hat_MK": np.empty((0, 0)) if simplex_fit.pi_hat is None else simplex_fit.pi_hat,
        "simplex_mapping_pairs_eval": np.array(sorted(simplex_mapping.items()), dtype=int),
        # Backward-compatible aliases from the first artifact export.
        "vertices": simplex_fit.vertices,
        "pi_hat": np.empty((0, 0)) if simplex_fit.pi_hat is None else simplex_fit.pi_hat,
        "alpha_train": alpha_train,
        "alpha_val": alpha_val,
        "alpha_test": alpha_test,
        "aligned_alpha_test": aligned_alpha_test,
        "simplex_mapping": np.array(sorted(simplex_mapping.items()), dtype=int),
    }
    if bottleneck_alpha_train is not None:
        artifact_arrays |= {
            "bottleneck_alpha_train": bottleneck_alpha_train,
            "bottleneck_alpha_val": bottleneck_alpha_val,
            "bottleneck_alpha_test": bottleneck_alpha_test,
            "bottleneck_alpha_test_aligned_eval": alpha_test_k,
            "bottleneck_class_alignment": bottleneck_class_alignment,
        }
    if bottleneck_vertices_M is not None:
        artifact_arrays["bottleneck_pi_hat"] = bottleneck_vertices_M
        artifact_arrays["bottleneck_vertices_M"] = bottleneck_vertices_M
    if bottleneck_pi_hat_MK is not None:
        artifact_arrays["bottleneck_pi_hat_MK"] = bottleneck_pi_hat_MK
    np.savez_compressed(output_dir / "simplex_fit_arrays.npz", **artifact_arrays)
    save_json(
        output_dir / "simplex_fit_arrays.json",
        {
            "artifact_schema_version": 2,
            "fitter_name": fitter_name,
            "method_code_name": method_code_name,
            "run_geometry_mode": "bottleneck" if is_bottleneck else "post_hoc_simplex",
            "npz_file": "simplex_fit_arrays.npz",
            "orientation": {
                "fitted_vertices_M": (
                    "K_by_M: fitted_vertices_M[k, m] = estimated P(source=m | fitted vertex k); "
                    "decoding solves min ||alpha @ fitted_vertices_M - g_theta(x)||_2^2"
                ),
                "oracle_vertices_M_eval": (
                    "K_by_M: oracle_vertices_M_eval[k, m] = true P(source=m | class=k); "
                    "evaluation-only"
                ),
                "pi_hat_MK": "M_by_K row-stochastic estimate derived from fitted vertices; evaluation/diagnostic",
                "bottleneck_pi_hat": (
                    "K_by_M, matching fitted_vertices_M; transpose gives the M_by_K matrix "
                    "used in g_theta(x) = Pi @ alpha_theta(x)"
                ),
            },
            "arrays": {key: list(value.shape) for key, value in artifact_arrays.items()},
            "class_alignment": class_alignment.tolist(),
            "class_alignment_meaning": (
                "evaluation-only map from fitted coordinate index to latent class index; "
                "-1 means unavailable"
            ),
            "bottleneck_class_alignment": (
                bottleneck_class_alignment.tolist() if is_bottleneck else None
            ),
            "bottleneck_alpha_source": bottleneck_alpha_source,
            "oracle_vertices_eval_only": True,
            "prior_free_fit_uses_oracle_vertices": False,
            "kspace_plot_source": (
                "bottleneck_alpha_test_aligned_eval"
                if is_bottleneck
                else "decoded_alpha_test_aligned_eval"
            ),
        },
    )
    save_json(
        output_dir / "simplex_details.json",
        {
            "vertices": simplex_fit.vertices.tolist(),
            "oracle_vertices": bundle.source_given_class.tolist(),
            "oracle_geometric_distance": metrics["simplex_oracle_geometric_distance"],
            "oracle_relative_geometric_distance": metrics[
                "simplex_oracle_relative_geometric_distance"
            ],
            "pi_hat": None if simplex_fit.pi_hat is None else simplex_fit.pi_hat.tolist(),
            "diagnostics": {
                "reconstruction_error": simplex_fit.diagnostics.reconstruction_error,
                "raw_reconstruction_error": simplex_fit.diagnostics.raw_reconstruction_error,
                "vertex_condition_number": simplex_fit.diagnostics.vertex_condition_number,
                "min_vertex_distance": simplex_fit.diagnostics.min_vertex_distance,
                "raw_barycentric_negative_fraction": simplex_fit.diagnostics.raw_barycentric_negative_fraction,
                "raw_barycentric_negative_mass": simplex_fit.diagnostics.raw_barycentric_negative_mass,
                "raw_barycentric_sum_error": simplex_fit.diagnostics.raw_barycentric_sum_error,
                "barycentric_projection_gap": simplex_fit.diagnostics.barycentric_projection_gap,
                "mean_barycentric_entropy": simplex_fit.diagnostics.mean_barycentric_entropy,
                "barycentric_collapse_fraction": simplex_fit.diagnostics.barycentric_collapse_fraction,
                "selection_score": simplex_fit.diagnostics.selection_score,
                "selected_candidate": simplex_fit.diagnostics.selected_candidate,
                "degenerate": simplex_fit.diagnostics.degenerate,
            },
            "preprocess": str(cfg["simplex"].get("preprocess", "none")),
            "selector": str(cfg["simplex"].get("selector", "restart_only")),
            "selection_summary": simplex_fit.selection_summary,
        },
    )
    save_json(output_dir / "metrics.json", metrics)
    save_dataframe(output_dir / "metrics.csv", metrics_frame)
    summary = [
        f"Experiment: {cfg['experiment']['name']}",
        f"Task: {cfg['task']}",
        f"Seed: {cfg['seed']}",
        f"Method accuracy: {metrics[f'{_method_key}_accuracy']:.4f}",
        f"Method macro F1: {metrics[f'{_method_key}_macro_f1']:.4f}",
        f"Posterior L1 error: {metrics['posterior_l1_error']}",
        f"Simplex oracle relative distance: {metrics['simplex_oracle_relative_geometric_distance']}",
        f"Simplex degenerate: {metrics['simplex_degenerate']}",
        f"Simplex candidate: {metrics['simplex_selected_candidate']}",
        f"Simplex projection gap: {metrics['simplex_projection_gap']:.4f}",
        f"Environment: {environment_summary()}",
    ]
    save_text(output_dir / "summary.txt", "\n".join(summary) + "\n")
    LOGGER.info("finished run at %s", output_dir)
    return RunArtifacts(metrics=metrics, metrics_frame=metrics_frame, output_dir=output_dir)


def run_sweep(
    config: DictConfig | dict[str, Any],
    parameter_path: str,
    values: list[Any],
    metric_key: str = "ours_linear_accuracy",
    extra_plot_name: str = "sweep_metric.png",
) -> RunArtifacts:
    """Run a simple one-parameter sweep and save aggregate metrics."""
    base_cfg = _apply_named_experiment_overrides(_config_to_dict(config))
    output_dir = _resolve_output_dir(base_cfg)
    ensure_dir(output_dir / "plots")
    rows: list[dict[str, Any]] = []
    for value in values:
        trial_cfg = deepcopy(base_cfg)
        cursor = trial_cfg
        parts = parameter_path.split(".")
        for key in parts[:-1]:
            cursor = cursor[key]
        cursor[parts[-1]] = value
        value_label = str(value).replace("/", "_")
        trial_dir = ensure_dir(output_dir / f"{parameter_path.replace('.', '_')}_{value_label}")
        trial_cfg["output_dir_override"] = str(trial_dir)
        run = run_experiment(trial_cfg)
        row = dict(run.metrics)
        row[parameter_path] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    save_dataframe(output_dir / "sweep_metrics.csv", frame)
    plot_metric_curve(
        frame,
        x_col=parameter_path,
        y_col=metric_key,
        out_path=output_dir / "plots" / extra_plot_name,
        title=f"{metric_key} vs {parameter_path}",
    )
    return RunArtifacts(metrics={"rows": len(rows)}, metrics_frame=frame, output_dir=output_dir)
