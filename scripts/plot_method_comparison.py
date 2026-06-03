"""Bar-chart comparison of per-method accuracy across datasets.

Usage:
    python scripts/plot_method_comparison.py \
        --input-root /path/to/outputs/realdata_hidden_priors \
        --output-dir docs/figures

Reads all metrics.json files under --input-root, pivots by method, and
writes one grouped bar chart per dataset (+ an aggregate figure).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Method display config -----------------------------------------------
# Each entry: (metrics_key_prefix, display_label, information_level)
# information_level: "fair" | "oracle-prior" | "oracle"
METHODS: list[tuple[str, str, str]] = [
    ("ovr",                  "OvR",                "fair"),
    ("demix_alg4_input",     "KSBS-Demix",         "fair"),
    ("ours_linear",          "Simplex",            "fair"),
    ("ours_bottleneck",      "Bottleneck",         "fair"),
    ("wei_ccm",              "Wei-CCM",            "oracle-prior"),
    ("known_prior_demix",    "Oracle simplex",     "oracle-prior"),
    ("oracle_supervised",    "Oracle",             "oracle"),
]

LEVEL_COLORS = {
    "fair":         "#4c72b0",
    "oracle-prior": "#dd8452",
    "oracle":       "#55a868",
}

LEVEL_HATCHES = {
    "fair":         "",
    "oracle-prior": "//",
    "oracle":       "xx",
}


def load_runs(input_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(input_root.rglob("metrics.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        row: dict = {
            "run_path": str(p.parent),
            "dataset": d.get("dataset_name", "unknown"),
            "seed": d.get("seed"),
            "num_classes": d.get("num_classes"),
            "num_sources": d.get("num_sources"),
        }
        for prefix, label, level in METHODS:
            key = f"{prefix}_accuracy"
            if key in d and d[key] is not None:
                row[label] = float(d[key])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_dataset(ax: plt.Axes, subset: pd.DataFrame, dataset: str) -> None:
    present_methods = [
        (prefix, label, level)
        for prefix, label, level in METHODS
        if label in subset.columns and subset[label].notna().any()
    ]
    labels = [lbl for _, lbl, _ in present_methods]
    means = [subset[lbl].mean() for _, lbl, _ in present_methods]
    stds  = [subset[lbl].std(ddof=0) if len(subset) > 1 else 0.0
             for _, lbl, _ in present_methods]
    colors  = [LEVEL_COLORS[lvl]   for _, _, lvl in present_methods]
    hatches = [LEVEL_HATCHES[lvl]  for _, _, lvl in present_methods]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, width=0.65, error_kw={"linewidth": 1})
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    n_seeds = subset["seed"].nunique()
    ax.set_title(f"{dataset}  (n={len(subset)} run{'s' if len(subset)>1 else ''}, {n_seeds} seed{'s' if n_seeds>1 else ''})")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def make_legend(fig: plt.Figure) -> None:
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=LEVEL_COLORS[lvl], hatch=LEVEL_HATCHES[lvl],
              label=lvl, edgecolor="black", linewidth=0.5)
        for lvl in ("fair", "oracle-prior", "oracle")
    ]
    fig.legend(handles=handles, title="Information level",
               loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True,
                        help="Root dir containing run output directories.")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(input_root)
    if df.empty:
        print("No runs found under", input_root)
        return

    datasets = sorted(df["dataset"].dropna().unique())
    print(f"Found {len(df)} runs across datasets: {datasets}")

    # One figure per dataset
    for dataset in datasets:
        sub = df[df["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(9, 4))
        plot_dataset(ax, sub, dataset)
        make_legend(fig)
        fig.tight_layout()
        out = output_dir / f"method_comparison_{dataset}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # Aggregate figure: one subplot per dataset
    if len(datasets) > 1:
        fig, axes = plt.subplots(1, len(datasets), figsize=(9 * len(datasets) // 2, 5), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            plot_dataset(ax, df[df["dataset"] == dataset], dataset)
        make_legend(fig)
        fig.suptitle("Method comparison across datasets", fontsize=11)
        fig.tight_layout()
        out = output_dir / "method_comparison_all.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # Also save a summary CSV
    summary_cols = ["dataset", "seed"] + [lbl for _, lbl, _ in METHODS if lbl in df.columns]
    df[summary_cols].to_csv(output_dir / "method_comparison.csv", index=False)
    print(f"Saved {output_dir}/method_comparison.csv")


if __name__ == "__main__":
    main()
