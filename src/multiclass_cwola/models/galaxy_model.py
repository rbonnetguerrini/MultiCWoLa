"""Pretrained ResNet50 backbone for Galaxy10 (and other natural-image datasets).

Drop-in replacement for `SmallCNNBackbone` in `build_classifier`. Inputs are
expected as CHW float32 tensors in [0, 1]; ImageNet mean/std normalization is
applied in-graph so the bundle's data layer stays unchanged.

The same head abstraction (linear or `BottleneckSimplexHead`) plugs in
unmodified -- `enable_bottleneck` toggles between them via `build_classifier`.
"""

from __future__ import annotations

import torch
from torch import nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet50Backbone(nn.Module):
    """ImageNet-pretrained ResNet50 with a learnable projection to `embedding_dim`.

    The conv stack is frozen by default (`freeze_backbone=True`); only the
    projection head trains. Call `unfreeze()` to release the backbone for
    fine-tuning at a smaller LR.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        freeze_backbone: bool = True,
        imagenet_normalize: bool = True,
        dropout: float = 0.1,
        weights: str = "DEFAULT",
    ) -> None:
        super().__init__()
        from torchvision import models

        weights_arg = None if str(weights).lower() == "none" else weights
        resnet = models.resnet50(weights=weights_arg)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # -> (B,2048,1,1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self._frozen = bool(freeze_backbone)
        if self._frozen:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.imagenet_normalize = bool(imagenet_normalize)
        if self.imagenet_normalize:
            self.register_buffer(
                "_in_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
            )
            self.register_buffer(
                "_in_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
            )

    def train(self, mode: bool = True) -> "ResNet50Backbone":
        super().train(mode)
        if self._frozen:
            # Keep BN/Dropout in eval mode so frozen-backbone activations are stable.
            self.feature_extractor.eval()
        return self

    def unfreeze(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
        self._frozen = False

    def is_frozen(self) -> bool:
        return self._frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if self.imagenet_normalize:
            x = (x - self._in_mean) / self._in_std
        feats = self.feature_extractor(x)
        return self.projection(feats)

    def optimizer_param_groups(
        self, head_lr: float, backbone_lr: float
    ) -> list[dict]:
        """Return AdamW-ready param groups (head + projection vs backbone)."""
        head_params = list(self.projection.parameters())
        backbone_params = [p for p in self.feature_extractor.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": float(head_lr)}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": float(backbone_lr)})
        return groups
