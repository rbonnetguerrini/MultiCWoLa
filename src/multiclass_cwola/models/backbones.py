"""Backbone networks for source and latent classifiers."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class MLPBackbone(nn.Module):
    """Simple MLP with a named embedding layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.embedding = nn.Linear(current_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = self.feature_extractor(x)
        return torch.relu(self.embedding(x))


class SmallCNNBackbone(nn.Module):
    """A small CNN suitable for MNIST-sized experiments."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        channels: list[int],
        embedding_dim: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        in_channels = input_shape[0]
        conv_layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in channels:
            conv_layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
            )
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.extend([nn.ReLU(), nn.MaxPool2d(kernel_size=2)])
            current_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = int(self.conv(dummy).reshape(1, -1).shape[1])
        self.embedding = nn.Linear(flat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return torch.relu(self.embedding(x))


class BottleneckSimplexHead(nn.Module):
    """Architectural bottleneck that constrains outputs to a K-vertex simplex in R^M.

    Outputs ``log q(x)`` where ``q(x) = (1 - eps(x)) * Pi @ alpha(x) + eps(x) * c``,
    with ``Pi`` column-stochastic, ``alpha(x)`` on the (K-1)-simplex, and an
    optional slack residual ``c`` lifting points off the strict K-simplex hull.
    Pair with ``nn.NLLLoss`` for source classification (the trainer picks this
    up via ``returns_log_probs=True`` on the parent model).

    Knobs (all opt-in via the ``model.bottleneck`` config block):
      - ``pi_normalization``: ``"softmax"`` (default, bounded sharpness) or
        ``"softplus"`` (vertices can approach one-hot).
      - ``slack_mode``: ``"none"`` (default) | ``"uniform"`` (c = 1/M) |
        ``"learnable"`` (c is a learned column-stochastic vector). Lets points
        sit *inside* Delta^{M-1} rather than only on the K-simplex faces.
      - ``slack_max``: upper bound on per-example slack eps(x)  in  [0, slack_max].
      - ``vertex_spread_lambda``: weight for ``-log det(Pi_tilde^T Pi_tilde + epsI)`` reg,
        which penalises Pi columns that drift toward each other.
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        num_classes: int,
        pi_init: str = "identity_like",
        pi_temperature: float = 1.0,
        pi_normalization: str = "softmax",
        slack_mode: str = "none",
        slack_max: float = 0.2,
        vertex_spread_lambda: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if pi_normalization not in {"softmax", "softplus"}:
            raise ValueError(f"Unknown pi_normalization: {pi_normalization}")
        if slack_mode not in {"none", "uniform", "learnable"}:
            raise ValueError(f"Unknown slack_mode: {slack_mode}")
        self.num_sources = num_sources
        self.num_classes = num_classes
        self.pi_temperature = float(pi_temperature)
        self.pi_normalization = pi_normalization
        self.slack_mode = slack_mode
        self.slack_max = float(slack_max)
        self.vertex_spread_lambda = float(vertex_spread_lambda)
        self.eps = float(eps)
        self.latent_linear = nn.Linear(input_dim, num_classes)
        self.pi_logits = nn.Parameter(
            self._init_pi_logits(num_sources, num_classes, pi_init)
        )
        if slack_mode != "none":
            # Per-example slack scalar eps(x)  in  [0, slack_max] via sigmoid.
            self.slack_head = nn.Linear(input_dim, 1)
            nn.init.zeros_(self.slack_head.weight)
            nn.init.constant_(self.slack_head.bias, -2.0)  # start with small eps
        if slack_mode == "learnable":
            self.slack_residual = nn.Parameter(torch.zeros(num_sources))
        # Cache for the regulariser to read alpha without an extra forward.
        self._last_alpha: torch.Tensor | None = None

    @staticmethod
    def _init_pi_logits(M: int, K: int, mode: str) -> torch.Tensor:
        if mode == "random":
            return torch.randn(M, K) * 0.01
        if mode == "uniform":
            return torch.zeros(M, K)
        if mode == "identity_like":
            base = torch.zeros(M, K)
            for k in range(K):
                base[k % M, k] = 3.0
            base = base + 0.01 * torch.randn(M, K)
            return base
        raise ValueError(f"Unknown pi_init mode: {mode}")

    def pi(self) -> torch.Tensor:
        """Return the column-stochastic mixing matrix Pi of shape (M, K)."""
        scaled = self.pi_logits / self.pi_temperature
        if self.pi_normalization == "softmax":
            return torch.softmax(scaled, dim=0)
        raw = torch.nn.functional.softplus(scaled)
        return raw / (raw.sum(dim=0, keepdim=True) + self.eps)

    def alpha(self, features: torch.Tensor) -> torch.Tensor:
        """Return latent posteriors alpha(x) of shape (B, K)."""
        return torch.softmax(self.latent_linear(features), dim=1)

    def slack_residual_distribution(self) -> torch.Tensor:
        """Return the column-stochastic slack residual c of shape (M,)."""
        if self.slack_mode == "uniform":
            pi = self.pi_logits  # for device/dtype only
            return torch.full(
                (self.num_sources,), 1.0 / self.num_sources,
                device=pi.device, dtype=pi.dtype,
            )
        if self.slack_mode == "learnable":
            return torch.softmax(self.slack_residual, dim=0)
        raise RuntimeError("slack_residual_distribution called with slack_mode='none'")

    def set_pi_frozen(self, frozen: bool) -> None:
        self.pi_logits.requires_grad_(not frozen)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        a = self.alpha(features)
        if self.training:
            self._last_alpha = a
        q = a @ self.pi().t()  # (B, M)
        if self.slack_mode != "none":
            eps_x = self.slack_max * torch.sigmoid(self.slack_head(features))  # (B, 1)
            c = self.slack_residual_distribution().unsqueeze(0)  # (1, M)
            q = (1.0 - eps_x) * q + eps_x * c
        return torch.log(q + self.eps)

    def regularization_loss(self) -> torch.Tensor:
        """Vertex-spread penalty on the current Pi. Zero if disabled or near-zero lambda."""
        if self.vertex_spread_lambda <= 0.0:
            return self.pi_logits.new_zeros(())
        pi = self.pi()  # (M, K)
        pi_centered = pi - pi.mean(dim=0, keepdim=True)
        gram = pi_centered.t() @ pi_centered  # (K, K)
        gram = gram + 1e-4 * torch.eye(self.num_classes, device=gram.device, dtype=gram.dtype)
        _, logabsdet = torch.linalg.slogdet(gram)
        return -self.vertex_spread_lambda * logabsdet


class ClassifierModel(nn.Module):
    """Backbone plus a classification head (linear by default, bottleneck-optional)."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        num_outputs: int,
        head: nn.Module | None = None,
        returns_log_probs: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_outputs = int(num_outputs)
        self.head = head if head is not None else nn.Linear(embedding_dim, num_outputs)
        self.returns_log_probs = returns_log_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        return self.head(embedding)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _maybe_build_bottleneck(
    model_cfg: dict,
    embedding_dim: int,
    num_outputs: int,
    num_classes: int | None,
    enable_bottleneck: bool,
) -> tuple[nn.Module | None, bool]:
    """Return (head, returns_log_probs) for a bottleneck head, or (None, False)."""
    if not enable_bottleneck:
        return None, False
    bottleneck_cfg = model_cfg.get("bottleneck") or {}
    if not bool(bottleneck_cfg.get("enabled", False)):
        return None, False
    if num_classes is None:
        raise ValueError(
            "build_classifier requires num_classes when model.bottleneck.enabled=true"
        )
    head = BottleneckSimplexHead(
        input_dim=embedding_dim,
        num_sources=int(num_outputs),
        num_classes=int(num_classes),
        pi_init=str(bottleneck_cfg.get("pi_init", "identity_like")),
        pi_temperature=float(bottleneck_cfg.get("pi_temperature", 1.0)),
        pi_normalization=str(bottleneck_cfg.get("pi_normalization", "softmax")),
        slack_mode=str(bottleneck_cfg.get("slack_mode", "none")),
        slack_max=float(bottleneck_cfg.get("slack_max", 0.2)),
        vertex_spread_lambda=float(bottleneck_cfg.get("vertex_spread_lambda", 0.0)),
    )
    pi_init_path = bottleneck_cfg.get("pi_init_path")
    if pi_init_path:
        import json as _j

        with open(str(pi_init_path)) as _f:
            _details = _j.load(_f)
        _pi_hat = torch.tensor(_details["pi_hat"], dtype=torch.float32)
        if _pi_hat.shape != (head.num_sources, head.num_classes):
            raise ValueError(
                f"pi_init_path simplex shape {tuple(_pi_hat.shape)} does not match "
                f"(M={head.num_sources}, K={head.num_classes})"
            )
        with torch.no_grad():
            head.pi_logits.copy_(torch.log(_pi_hat.clamp_min(1e-4)))
    return head, True


def build_classifier(
    model_cfg: dict,
    input_shape: tuple[int, ...],
    num_outputs: int,
    num_classes: int | None = None,
    enable_bottleneck: bool = False,
) -> ClassifierModel:
    """Build a classifier for tabular or image inputs.

    The bottleneck simplex head is only meaningful for the M-way source
    classifier; opt-in via ``enable_bottleneck=True`` (and a non-None
    ``num_classes``). K-way baselines (oracle, stage-2, etc.) leave it off.
    """
    backbone_name = str(model_cfg["backbone"])
    if backbone_name == "mlp":
        if len(input_shape) > 1:
            input_dim = int(np.prod(input_shape))
        else:
            input_dim = int(input_shape[0])
        embedding_dim = int(model_cfg.get("embedding_dim", 32))
        backbone = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=list(model_cfg.get("hidden_dims", [64, 64])),
            embedding_dim=embedding_dim,
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    elif backbone_name == "cnn":
        embedding_dim = int(model_cfg.get("embedding_dim", 64))
        backbone = SmallCNNBackbone(
            input_shape=input_shape,
            channels=list(model_cfg.get("channels", [16, 32])),
            embedding_dim=embedding_dim,
            use_batchnorm=bool(model_cfg.get("use_batchnorm", True)),
        )
    elif backbone_name == "resnet50_pretrained":
        from multiclass_cwola.models.galaxy_model import ResNet50Backbone

        embedding_dim = int(model_cfg.get("embedding_dim", 512))
        backbone = ResNet50Backbone(
            embedding_dim=embedding_dim,
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            imagenet_normalize=bool(model_cfg.get("imagenet_normalize", True)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            weights=str(model_cfg.get("weights", "DEFAULT")),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    head, returns_log_probs = _maybe_build_bottleneck(
        model_cfg, embedding_dim, num_outputs, num_classes, enable_bottleneck
    )
    return ClassifierModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        num_outputs=num_outputs,
        head=head,
        returns_log_probs=returns_log_probs,
    )
