"""Sparse overlap-aware fusion between CNN cells and superpixel tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..neck_injection.tokens import TokenBundle


def _gate_value(init: str) -> float:
    values = {"zero": 0.0, "tiny": 1e-3, "one": 1.0}
    try:
        return values[init]
    except KeyError as exc:
        raise ValueError("gate_init must be one of: zero, tiny, one.") from exc


def _choose_heads(channels: int, requested: int) -> int:
    for heads in range(min(max(1, int(requested)), channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


@dataclass(frozen=True)
class OverlapEdges:
    """Sparse area correspondence from cells to local superpixel-token IDs.

    ``cell_index`` addresses a flattened ``B * H * W`` CNN grid. ``region_index``
    addresses the corresponding per-image token table, including its unused CLS
    position at index zero. ``area`` sums to one for every populated CNN cell.
    """

    cell_index: torch.Tensor
    batch_index: torch.Tensor
    region_index: torch.Tensor
    area: torch.Tensor
    num_cells: int


def build_overlap_edges(labels: torch.Tensor, output_size: tuple[int, int], num_tokens: int) -> OverlapEdges:
    """Compute exact raster overlap fractions on the labels' current device."""

    if labels.ndim != 3:
        raise ValueError(f"Expected BxHxW labels, got {tuple(labels.shape)}.")
    batch, label_h, label_w = labels.shape
    target_h, target_w = (int(output_size[0]), int(output_size[1]))
    if target_h <= 0 or target_w <= 0 or num_tokens <= 1:
        raise ValueError("output_size must be positive and num_tokens must include a CLS token.")

    device = labels.device
    y_cells = torch.div(torch.arange(label_h, device=device) * target_h, label_h, rounding_mode="floor")
    x_cells = torch.div(torch.arange(label_w, device=device) * target_w, label_w, rounding_mode="floor")
    cell_ids = (y_cells[:, None] * target_w + x_cells[None, :]).reshape(1, -1)
    num_cells = target_h * target_w
    flat_labels = labels.reshape(batch, -1).long()
    if torch.any(flat_labels <= 0) or torch.any(flat_labels >= num_tokens):
        raise ValueError("Superpixel labels must be contiguous token IDs in 1..num_tokens-1.")

    batch_ids = torch.arange(batch, device=device, dtype=torch.long)[:, None]
    global_cells = batch_ids * num_cells + cell_ids
    packed = global_cells * int(num_tokens) + flat_labels
    unique, counts = torch.unique(packed.reshape(-1), return_counts=True, sorted=True)
    edge_cells = torch.div(unique, int(num_tokens), rounding_mode="floor")
    region_index = unique.remainder(int(num_tokens))
    cell_counts = torch.bincount(global_cells.reshape(-1), minlength=batch * num_cells)
    area = counts.to(dtype=torch.float32) / cell_counts[edge_cells].to(dtype=torch.float32)
    return OverlapEdges(
        cell_index=edge_cells,
        batch_index=torch.div(edge_cells, num_cells, rounding_mode="floor"),
        region_index=region_index,
        area=area,
        num_cells=num_cells,
    )


class BackboneCorrespondenceFusion(nn.Module):
    """Fuse local superpixel tokens into a dense CNN feature map.

    ``overlap_pool`` is the simple correspondence control. In
    ``overlap_attention`` each CNN cell attends only to tokens that overlap it.
    """

    def __init__(
        self,
        token_provider: nn.Module,
        mode: str = "overlap_attention",
        gate_init: str = "tiny",
        attention_heads: int = 4,
        overlap_logit_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"overlap_pool", "overlap_attention"}:
            raise ValueError("mode must be 'overlap_pool' or 'overlap_attention'.")
        self.token_provider = token_provider
        self.mode = mode
        self.attention_heads = int(attention_heads)
        self.overlap_logit_bias = float(overlap_logit_bias)
        self.gamma = nn.Parameter(torch.tensor(_gate_value(gate_init), dtype=torch.float32))
        self.query: nn.Linear | None = None
        self.key: nn.Linear | None = None
        self.value: nn.Linear | None = None
        self.output: nn.Linear | None = None
        self.initialized_channels: int | None = None
        self.initialized_token_dim: int | None = None

    def initialize(
        self,
        yolo_channels: int,
        token_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        yolo_channels, token_dim = int(yolo_channels), int(token_dim)
        if yolo_channels <= 0 or token_dim <= 0:
            raise ValueError("yolo_channels and token_dim must be positive.")
        self.value = nn.Linear(token_dim, yolo_channels, bias=True)
        self.output = nn.Linear(yolo_channels, yolo_channels, bias=True)
        nn.init.normal_(self.value.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.value.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)
        if self.mode == "overlap_attention":
            self.query = nn.Linear(yolo_channels, yolo_channels, bias=True)
            self.key = nn.Linear(token_dim, yolo_channels, bias=True)
            nn.init.normal_(self.query.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.query.bias)
            nn.init.normal_(self.key.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.key.bias)
        self.initialized_channels = yolo_channels
        self.initialized_token_dim = token_dim
        if device is not None:
            self.to(device=device, dtype=dtype)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, segment: torch.Tensor, count: int) -> torch.Tensor:
        """Softmax over the variable-length edge list for every CNN cell."""

        expanded = segment[:, None].expand_as(scores)
        maxima = torch.full((count, scores.shape[1]), -torch.inf, device=scores.device, dtype=scores.dtype)
        maxima.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - maxima[segment])
        denominator = torch.zeros_like(maxima)
        denominator.scatter_add_(0, expanded, exp_scores)
        return exp_scores / denominator[segment].clamp_min(torch.finfo(scores.dtype).tiny)

    def _require_initialized(self, channels: int, token_dim: int) -> tuple[nn.Linear, nn.Linear]:
        if self.value is None or self.output is None:
            raise RuntimeError("BackboneCorrespondenceFusion must be initialized before optimizer creation.")
        if self.value.in_features != token_dim or self.value.out_features != channels:
            raise RuntimeError("BackboneCorrespondenceFusion was initialized for incompatible dimensions.")
        return self.value, self.output

    def forward(self, yolo_feature: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        if yolo_feature.ndim != 4:
            raise ValueError(f"Expected BCHW YOLO feature, got {tuple(yolo_feature.shape)}.")
        bundle = self.token_provider(rgb)
        if not isinstance(bundle, TokenBundle):
            raise TypeError("token_provider must return a TokenBundle.")
        batch, channels, height, width = yolo_feature.shape
        if bundle.tokens.shape[0] != batch or bundle.labels.shape[0] != batch:
            raise ValueError("TokenBundle batch size does not match the YOLO feature batch.")

        token_dim = int(bundle.tokens.shape[-1])
        value_layer, output_layer = self._require_initialized(channels, token_dim)
        edges = build_overlap_edges(bundle.labels, (height, width), int(bundle.tokens.shape[1]))
        feature_dtype = yolo_feature.dtype
        flat_feature = yolo_feature.flatten(2).transpose(1, 2).reshape(batch * edges.num_cells, channels)
        tokens = bundle.tokens.to(device=yolo_feature.device, dtype=feature_dtype)
        edge_tokens = tokens[edges.batch_index, edges.region_index]
        edge_values = value_layer(edge_tokens).to(dtype=feature_dtype)

        if self.mode == "overlap_pool":
            weights = edges.area.to(device=yolo_feature.device, dtype=feature_dtype)
            context = torch.zeros_like(flat_feature)
            context.index_add_(0, edges.cell_index, edge_values * weights[:, None])
        else:
            if self.query is None or self.key is None:
                raise RuntimeError("overlap_attention projections were not initialized.")
            heads = _choose_heads(channels, self.attention_heads)
            head_dim = channels // heads
            query = self.query(flat_feature)[edges.cell_index].to(dtype=feature_dtype).reshape(-1, heads, head_dim)
            key = self.key(edge_tokens).to(dtype=feature_dtype).reshape(-1, heads, head_dim)
            scores = (query * key).sum(dim=-1) * (head_dim**-0.5)
            if self.overlap_logit_bias:
                scores = scores + self.overlap_logit_bias * edges.area.to(scores.dtype).clamp_min(1e-8).log()[:, None]
            weights = self._segment_softmax(scores, edges.cell_index, batch * edges.num_cells).to(dtype=feature_dtype)
            weighted_values = edge_values.reshape(-1, heads, head_dim) * weights[:, :, None]
            context_heads = torch.zeros(
                (batch * edges.num_cells, heads, head_dim),
                device=yolo_feature.device,
                dtype=feature_dtype,
            )
            context_heads.index_add_(0, edges.cell_index, weighted_values)
            context = context_heads.reshape(batch * edges.num_cells, channels)

        injected = (
            output_layer(context)
            .to(dtype=feature_dtype)
            .reshape(batch, height * width, channels)
            .transpose(1, 2)
            .reshape_as(yolo_feature)
        )
        return yolo_feature + self.gamma.to(dtype=feature_dtype) * injected
