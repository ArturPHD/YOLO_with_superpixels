"""Fusion modules for injecting superpixel tokens into YOLO feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn

from .tokens import SPiTTokenProvider, TokenBundle, tokens_to_dense_map


def _gate_value(init: str) -> float:
    if init == "zero":
        return 0.0
    if init == "tiny":
        return 1e-3
    if init == "one":
        return 1.0
    raise ValueError("gate_init must be one of: zero, tiny, one.")


def _choose_heads(channels: int, requested: int) -> int:
    requested = max(1, int(requested))
    for heads in range(min(requested, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class SuperpixelNeckFusion(nn.Module):
    """Fuse SPiT token features with one YOLO neck feature map.

    Modes:
        ``add``: project dense token map to YOLO channels and add with a gate.
        ``concat``: concatenate YOLO and token maps, then use a 1x1 conv.
        ``cross_attention``: use YOLO locations as queries and SPiT tokens as
        keys/values, then add attention output with a gate.
    """

    def __init__(
        self,
        token_provider: SPiTTokenProvider | None = None,
        mode: str = "add",
        gate_init: str = "tiny",
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"add", "concat", "cross_attention"}:
            raise ValueError("mode must be one of: add, concat, cross_attention.")
        self.token_provider = token_provider or SPiTTokenProvider()
        self.mode = mode
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)
        self.gamma = nn.Parameter(torch.tensor(_gate_value(gate_init), dtype=torch.float32))

        self.dense_projector: nn.Conv2d | None = None
        self.concat_projector: nn.Conv2d | None = None
        self.token_projector: nn.Linear | None = None
        self.cross_attention: nn.MultiheadAttention | None = None
        self.initialized_channels: int | None = None
        self.initialized_token_dim: int | None = None

    def initialize(
        self,
        yolo_channels: int,
        token_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create all fusion parameters before the training optimizer is built."""

        yolo_channels = int(yolo_channels)
        token_dim = int(token_dim)
        if yolo_channels <= 0 or token_dim <= 0:
            raise ValueError("yolo_channels and token_dim must be positive.")

        if self.mode in {"add", "concat"}:
            self.dense_projector = nn.Conv2d(token_dim, yolo_channels, kernel_size=1, bias=True)
            nn.init.normal_(self.dense_projector.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.dense_projector.bias)

        if self.mode == "concat":
            self.concat_projector = nn.Conv2d(yolo_channels * 2, yolo_channels, kernel_size=1, bias=True)
            with torch.no_grad():
                self.concat_projector.weight.zero_()
                self.concat_projector.bias.zero_()
                diag = torch.arange(yolo_channels)
                self.concat_projector.weight[diag, diag, 0, 0] = 1.0
                self.concat_projector.weight[:, yolo_channels:, 0, 0].normal_(mean=0.0, std=1e-3)

        if self.mode == "cross_attention":
            self.token_projector = nn.Linear(token_dim, yolo_channels, bias=True)
            nn.init.normal_(self.token_projector.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.token_projector.bias)
            heads = _choose_heads(yolo_channels, self.attention_heads)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=yolo_channels,
                num_heads=heads,
                dropout=self.attention_dropout,
                batch_first=True,
            )

        self.initialized_channels = yolo_channels
        self.initialized_token_dim = token_dim
        if device is not None:
            if dtype is None:
                self.to(device=device)
            else:
                self.to(device=device, dtype=dtype)

    def forward(self, yolo_feature: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        if yolo_feature.ndim != 4:
            raise ValueError(f"Expected BCHW YOLO feature, got shape {tuple(yolo_feature.shape)}.")

        bundle = self.token_provider(rgb)
        if self.mode == "cross_attention":
            return self._forward_cross_attention(yolo_feature, bundle)

        dense_tokens = tokens_to_dense_map(bundle, output_size=tuple(yolo_feature.shape[-2:]))
        dense_tokens = dense_tokens.to(device=yolo_feature.device, dtype=yolo_feature.dtype)
        projected = self._project_dense(dense_tokens, out_channels=int(yolo_feature.shape[1]))

        if self.mode == "add":
            return yolo_feature + self.gamma.to(dtype=yolo_feature.dtype) * projected

        return self._forward_concat(yolo_feature, projected)

    def _project_dense(self, dense_tokens: torch.Tensor, out_channels: int) -> torch.Tensor:
        in_channels = int(dense_tokens.shape[1])
        if (
            self.dense_projector is None
            or self.dense_projector.in_channels != in_channels
            or self.dense_projector.out_channels != out_channels
        ):
            raise RuntimeError(
                "SuperpixelNeckFusion dense projector was not initialized for "
                f"token_dim={in_channels}, yolo_channels={out_channels}. "
                "patch_neck_injection must initialize fusion modules before optimizer creation."
            )
        return self.dense_projector(dense_tokens)

    def _forward_concat(self, yolo_feature: torch.Tensor, projected_tokens: torch.Tensor) -> torch.Tensor:
        channels = int(yolo_feature.shape[1])
        if self.concat_projector is None or self.concat_projector.in_channels != channels * 2:
            raise RuntimeError(
                "SuperpixelNeckFusion concat projector was not initialized for "
                f"yolo_channels={channels}. patch_neck_injection must initialize fusion modules "
                "before optimizer creation."
            )

        fused = torch.cat([yolo_feature, projected_tokens], dim=1)
        return self.concat_projector(fused)

    def _forward_cross_attention(self, yolo_feature: torch.Tensor, bundle: TokenBundle) -> torch.Tensor:
        batch, channels, height, width = yolo_feature.shape
        tokens = bundle.tokens[:, 1:].to(device=yolo_feature.device, dtype=yolo_feature.dtype)
        valid = bundle.valid_mask[:, 1:].to(device=yolo_feature.device)
        token_dim = int(tokens.shape[-1])

        if self.token_projector is None or self.token_projector.in_features != token_dim:
            raise RuntimeError(
                "SuperpixelNeckFusion token projector was not initialized for "
                f"token_dim={token_dim}. patch_neck_injection must initialize fusion modules "
                "before optimizer creation."
            )

        if self.cross_attention is None or self.cross_attention.embed_dim != channels:
            raise RuntimeError(
                "SuperpixelNeckFusion cross-attention was not initialized for "
                f"yolo_channels={channels}. patch_neck_injection must initialize fusion modules "
                "before optimizer creation."
            )

        query = yolo_feature.flatten(2).transpose(1, 2)
        key_value = self.token_projector(tokens)
        attention_out, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attention_map = attention_out.transpose(1, 2).reshape(batch, channels, height, width)
        return yolo_feature + self.gamma.to(dtype=yolo_feature.dtype) * attention_map
