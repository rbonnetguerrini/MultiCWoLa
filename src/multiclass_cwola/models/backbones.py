"""The bring-your-own-backbone bottleneck head (architectural recovery, R2)."""

from __future__ import annotations

import torch
from torch import nn


class BottleneckSimplexHead(nn.Module):
    """Architectural bottleneck that constrains outputs to a K-vertex simplex in R^M.

    Outputs ``log q(x)`` where ``q(x) = (1 - eps(x)) * Pi @ alpha(x) + eps(x) * c``,
    with ``Pi`` column-stochastic, ``alpha(x)`` on the (K-1)-simplex, and an
    optional slack residual ``c`` lifting points off the strict K-simplex hull.
    Pair with ``nn.NLLLoss`` for source classification (the trainer picks this
    up via ``returns_log_probs=True`` on the parent model).

    Bring-your-own-backbone usage: drop this in as the final head of any model
    that produces ``(B, input_dim)`` embeddings, train with cross-entropy / NLL
    on mixture labels, then extract the recovered structure:

      - ``predict_latent(features)`` -> latent posteriors alpha(x) of shape (B, K);
        pass your backbone's embeddings (the same tensor this head receives).
      - ``pi()`` -> recovered column-stochastic mixing matrix Pi_hat of shape (M, K);
        its columns are the simplex vertices in source-posterior space.

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

    @torch.no_grad()
    def predict_latent(self, features: torch.Tensor) -> torch.Tensor:
        """Inference-time latent posteriors alpha(x) of shape (B, K).

        Documented extraction entry point for the bring-your-own-backbone path:
        pass the embeddings your backbone feeds this head. Same as :meth:`alpha`
        but under ``no_grad`` and with the module in eval semantics for callers.
        """
        return self.alpha(features)

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
