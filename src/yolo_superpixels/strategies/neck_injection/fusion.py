"""Fusion modules for injecting superpixel tokens into YOLO feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn

from .tokens import SPiTTokenProvider, TokenBundle, tokens_to_dense_map


TOKEN_ABLATION_MODES = {
    "normal",
    "zero",
    "constant",
    "batch_shuffle",
    "spatial_shuffle",
    "random_matched",
}


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
        self.token_ablation = "normal"
        self.token_ablation_seed = 0
        self._token_ablation_calls = 0
        self._diagnostics_enabled = False
        self._diagnostics_max_queries = 256
        self._diagnostics_max_forwards = 0
        self._collect_diagnostics_this_forward = False
        self._diagnostic_records: list[dict[str, float]] = []
        self._last_attention_statistics: dict[str, float] = {}

    def _ensure_runtime_state(self) -> None:
        """Hydrate non-parameter fields absent from checkpoints saved by older code."""

        defaults: dict[str, object] = {
            "token_ablation": "normal",
            "token_ablation_seed": 0,
            "_token_ablation_calls": 0,
            "_diagnostics_enabled": False,
            "_diagnostics_max_queries": 256,
            "_diagnostics_max_forwards": 0,
            "_collect_diagnostics_this_forward": False,
            "_diagnostic_records": [],
            "_last_attention_statistics": {},
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value)

    def set_token_ablation(self, mode: str = "normal", seed: int = 0) -> None:
        """Set a validation-time token perturbation without altering fusion weights."""

        self._ensure_runtime_state()
        if mode not in TOKEN_ABLATION_MODES:
            raise ValueError(f"token ablation must be one of {sorted(TOKEN_ABLATION_MODES)}.")
        self.token_ablation = mode
        self.token_ablation_seed = int(seed)
        self._token_ablation_calls = 0

    def set_feature_diagnostics(
        self,
        enabled: bool,
        max_queries: int = 256,
        max_forwards: int = 0,
    ) -> None:
        """Enable bounded diagnostic summaries for validation-only analysis."""

        self._ensure_runtime_state()
        self._diagnostics_enabled = bool(enabled)
        self._diagnostics_max_queries = max(1, int(max_queries))
        self._diagnostics_max_forwards = max(0, int(max_forwards))
        self._collect_diagnostics_this_forward = False
        self._diagnostic_records.clear()

    def diagnostic_summary(self) -> dict[str, float]:
        """Return per-forward mean diagnostics collected while enabled."""

        self._ensure_runtime_state()
        if not self._diagnostic_records:
            return {}
        keys = sorted({key for record in self._diagnostic_records for key in record})
        return {
            key: sum(record[key] for record in self._diagnostic_records if key in record)
            / sum(1 for record in self._diagnostic_records if key in record)
            for key in keys
        }

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

        self._ensure_runtime_state()
        self._collect_diagnostics_this_forward = self._diagnostics_enabled and (
            self._diagnostics_max_forwards == 0
            or len(self._diagnostic_records) < self._diagnostics_max_forwards
        )
        bundle = self.token_provider(rgb)
        raw_token_rms = self._valid_token_rms(bundle) if self._collect_diagnostics_this_forward else None
        bundle = self._apply_token_ablation(bundle)
        if self.mode == "cross_attention":
            output = self._forward_cross_attention(yolo_feature, bundle)
            self._record_diagnostics(yolo_feature, bundle, output, raw_token_rms)
            return output

        dense_tokens = tokens_to_dense_map(bundle, output_size=tuple(yolo_feature.shape[-2:]))
        dense_tokens = dense_tokens.to(device=yolo_feature.device, dtype=yolo_feature.dtype)
        projected = self._project_dense(dense_tokens, out_channels=int(yolo_feature.shape[1]))

        if self.mode == "add":
            output = yolo_feature + self.gamma.to(dtype=yolo_feature.dtype) * projected
            self._record_diagnostics(yolo_feature, bundle, output, raw_token_rms)
            return output

        output = self._forward_concat(yolo_feature, projected)
        self._record_diagnostics(yolo_feature, bundle, output, raw_token_rms)
        return output

    def _apply_token_ablation(self, bundle: TokenBundle) -> TokenBundle:
        """Perturb non-CLS token values while keeping tensor device and padding valid."""

        if self.token_ablation == "normal":
            return bundle

        if self.token_ablation == "batch_shuffle" and self.mode != "cross_attention":
            raise ValueError("batch_shuffle is defined only for cross_attention fusion.")

        tokens = bundle.tokens.clone()
        valid = bundle.valid_mask
        region_valid = valid.clone()
        region_valid[:, 0] = False
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(self.token_ablation_seed + self._token_ablation_calls)
        self._token_ablation_calls += 1

        if self.token_ablation == "zero":
            tokens = tokens.masked_fill(region_valid.unsqueeze(-1), 0.0)
        elif self.token_ablation == "constant":
            for batch_index in range(tokens.shape[0]):
                indices = region_valid[batch_index].nonzero(as_tuple=False).flatten()
                if indices.numel() > 0:
                    mean = tokens[batch_index, indices].mean(dim=0, keepdim=True)
                    tokens[batch_index, indices] = mean
        elif self.token_ablation == "batch_shuffle":
            if tokens.shape[0] > 1:
                # A deterministic cyclic shift guarantees that every image receives
                # another image's tokens without a device-to-host identity check.
                permutation = torch.roll(torch.arange(tokens.shape[0], device=tokens.device), shifts=1)
                tokens = tokens[permutation]
                valid = valid[permutation]
            else:
                tokens = self._spatially_shuffle_tokens(tokens, region_valid, generator)
        elif self.token_ablation == "spatial_shuffle":
            tokens = self._spatially_shuffle_tokens(tokens, region_valid, generator)
        elif self.token_ablation == "random_matched":
            for batch_index in range(tokens.shape[0]):
                indices = region_valid[batch_index].nonzero(as_tuple=False).flatten()
                if indices.numel() > 0:
                    values = tokens[batch_index, indices]
                    mean = values.mean(dim=0, keepdim=True)
                    std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
                    tokens[batch_index, indices] = torch.randn(
                        values.shape,
                        device=tokens.device,
                        dtype=tokens.dtype,
                        generator=generator,
                    ) * std + mean

        return TokenBundle(tokens=tokens, valid_mask=valid, labels=bundle.labels)

    @staticmethod
    def _spatially_shuffle_tokens(
        tokens: torch.Tensor,
        region_valid: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        output = tokens.clone()
        for batch_index in range(tokens.shape[0]):
            indices = region_valid[batch_index].nonzero(as_tuple=False).flatten()
            if indices.numel() > 1:
                permutation = torch.randperm(indices.numel(), device=tokens.device, generator=generator)
                output[batch_index, indices] = tokens[batch_index, indices[permutation]]
        return output

    @staticmethod
    def _valid_token_rms(bundle: TokenBundle) -> float:
        valid = bundle.valid_mask.clone()
        valid[:, 0] = False
        values = bundle.tokens[valid]
        if values.numel() == 0:
            return 0.0
        return float(values.detach().float().square().mean().sqrt().cpu())

    def _record_diagnostics(
        self,
        yolo_feature: torch.Tensor,
        bundle: TokenBundle,
        output: torch.Tensor,
        raw_token_rms: float | None,
    ) -> None:
        if not self._collect_diagnostics_this_forward:
            return
        feature_rms = float(yolo_feature.detach().float().square().mean().sqrt().cpu())
        delta_rms = float((output - yolo_feature).detach().float().square().mean().sqrt().cpu())
        record = {
            "yolo_feature_rms": feature_rms,
            "token_rms_raw": raw_token_rms if raw_token_rms is not None else self._valid_token_rms(bundle),
            "token_rms_effective": self._valid_token_rms(bundle),
            "fusion_delta_rms": delta_rms,
            "delta_to_yolo_rms": delta_rms / max(feature_rms, 1e-12),
            "fusion_gamma": float(self.gamma.detach().float().cpu()),
        }
        record.update(self._last_attention_statistics)
        self._diagnostic_records.append(record)

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
        self._last_attention_statistics = {}
        if self._collect_diagnostics_this_forward:
            query_step = max(1, query.shape[1] // self._diagnostics_max_queries)
            sampled_query = query[:, ::query_step]
            _, weights = self.cross_attention(
                query=sampled_query,
                key=key_value,
                value=key_value,
                key_padding_mask=~valid,
                need_weights=True,
                average_attn_weights=False,
            )
            probabilities = weights.detach().float().clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
            self._last_attention_statistics = {
                "attention_query_rms": float(sampled_query.detach().float().square().mean().sqrt().cpu()),
                "attention_key_rms": float(key_value.detach().float().square().mean().sqrt().cpu()),
                "attention_entropy": float(entropy.cpu()),
                "attention_effective_tokens": float(entropy.exp().cpu()),
            }
        attention_map = attention_out.transpose(1, 2).reshape(batch, channels, height, width)
        return yolo_feature + self.gamma.to(dtype=yolo_feature.dtype) * attention_map
