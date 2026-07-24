"""SPiT token extraction helpers for YOLO neck injection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...superpixels.algorithms import (
    SPiTDenseEdgeAlgorithm,
    TorchGridSuperpixelAlgorithm,
    TorchSLICSuperpixelAlgorithm,
    TorchVoronoiSuperpixelAlgorithm,
)


@dataclass
class TokenBundle:
    """Variable-length superpixel tokens plus labels that map pixels to tokens.

    Attributes:
        tokens: ``B x T x D`` token tensor. Index 0 is the CLS token; superpixel
            labels are expected to start from 1.
        valid_mask: ``B x T`` boolean mask. ``valid_mask[:, 0]`` is the CLS token.
        labels: ``B x H x W`` local label map with values ``1..N`` per image.
    """

    tokens: torch.Tensor
    valid_mask: torch.Tensor
    labels: torch.Tensor


def _parse_size(size: int | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError("image_size must be an int, a 2-tuple, or None.")
    return (int(size[0]), int(size[1]))


def _local_relabel(labels: torch.Tensor) -> torch.Tensor:
    """Convert arbitrary per-image labels to contiguous ``1..N`` labels."""

    if labels.ndim != 3:
        raise ValueError(f"Expected BxHxW labels, got shape {tuple(labels.shape)}.")

    output = torch.empty_like(labels, dtype=torch.long)
    for batch_index in range(labels.shape[0]):
        _, inverse = torch.unique(
            labels[batch_index].reshape(-1),
            sorted=True,
            return_inverse=True,
        )
        output[batch_index] = inverse.view_as(labels[batch_index]).long() + 1
    return output


def tokens_to_dense_map(bundle: TokenBundle, output_size: tuple[int, int]) -> torch.Tensor:
    """Scatter token embeddings back to a dense feature map.

    ``bundle.labels`` tells which token owns each pixel. The label map is resized
    with nearest-neighbor interpolation to the YOLO feature-map size, then each
    feature location gathers the embedding of its owning superpixel token.
    """

    tokens = bundle.tokens
    labels = bundle.labels
    if tokens.ndim != 3:
        raise ValueError(f"Expected BxTxD tokens, got shape {tuple(tokens.shape)}.")
    if labels.ndim != 3:
        raise ValueError(f"Expected BxHxW labels, got shape {tuple(labels.shape)}.")
    if tokens.shape[0] != labels.shape[0]:
        raise ValueError("Token batch size and label batch size do not match.")

    batch, max_tokens, token_dim = tokens.shape
    target_h, target_w = output_size
    resized_labels = F.interpolate(
        labels[:, None].float(),
        size=(target_h, target_w),
        mode="nearest",
    ).long()[:, 0]
    gather_index = resized_labels.clamp(0, max_tokens - 1).reshape(batch, -1)
    dense = tokens.gather(
        1,
        gather_index.unsqueeze(-1).expand(-1, -1, token_dim),
    )
    return dense.transpose(1, 2).reshape(batch, token_dim, target_h, target_w).contiguous()


class SPiTTokenProvider(nn.Module):
    """Run the SPiT token path and return tokens suitable for neck fusion.

    This uses SPiT's tokenizer/extractor/embedder/MaskedViT blocks, but bypasses
    its classification head. The returned token features are normalized
    embeddings after the selected ViT blocks.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        num_bins: int = 5,
        heads: int = 4,
        depth: int = 1,
        keep_k: int = 0,
        extractor: str = "interpolate",
        tokenizer: str = "superpixel",
        in_channels: int = 3,
        image_size: int | tuple[int, int] | list[int] | None = 160,
        classes: int = 1,
        spit_weights: str | Path | None = None,
        freeze: bool = False,
        drop_delta: bool = False,
        bbox_reg: bool = True,
        mode: str = "bilinear",
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if emb_dim <= 0:
            raise ValueError("emb_dim must be positive.")
        if heads <= 0 or emb_dim % heads != 0:
            raise ValueError("heads must be positive and divide emb_dim.")
        if depth < 0:
            raise ValueError("depth must be non-negative.")

        SPiTDenseEdgeAlgorithm._ensure_local_spit_on_path()
        from spit.nn import SPiT

        kwargs = dict(extra_kwargs or {})
        kwargs.setdefault("drop_delta", drop_delta)
        kwargs.setdefault("bbox_reg", bbox_reg)
        kwargs.setdefault("mode", mode)

        self.image_size = _parse_size(image_size)
        self.spit = SPiT(
            emb_dim=emb_dim,
            num_bins=num_bins,
            heads=heads,
            depth=depth,
            classes=classes,
            keep_k=keep_k,
            extractor=extractor,
            tokenizer=tokenizer,
            in_channels=in_channels,
            **kwargs,
        )
        if spit_weights:
            checkpoint = torch.load(spit_weights, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.spit.load_state_dict(state, strict=False)
        if freeze:
            for parameter in self.spit.parameters():
                parameter.requires_grad_(False)
            self.spit.eval()

    @property
    def emb_dim(self) -> int:
        return int(self.spit.embedder.emb_dim)

    def forward(self, rgb: torch.Tensor) -> TokenBundle:
        if rgb.ndim != 4 or rgb.shape[1] < 3:
            raise ValueError(f"Expected Bx3xHxW RGB input, got shape {tuple(rgb.shape)}.")

        rgb = rgb[:, :3].detach().float().clamp(0.0, 1.0)
        if self.image_size is not None and tuple(rgb.shape[-2:]) != self.image_size:
            rgb = F.interpolate(rgb, size=self.image_size, mode="bilinear", align_corners=False)

        features, labels, coords = self.spit.tokenizer(rgb, None)
        features = self.spit.extractor(features, labels, coords)
        embed_dtype = next(self.spit.embedder.parameters()).dtype
        features = features.to(dtype=embed_dtype)
        features, attention_mask, _global_idx, batch_idx = self.spit.embedder(features, labels, coords)
        features = self.spit.doi(features)

        for block in self.spit.blocks:
            features = block(features, attention_mask, batch_idx)

        batch, max_tokens = attention_mask.shape
        features = self.spit.norm(features).view(batch, max_tokens, -1)
        local_labels = _local_relabel(labels)
        return TokenBundle(
            tokens=features,
            valid_mask=attention_mask,
            labels=local_labels,
        )


class RegionFeatureTokenProvider(nn.Module):
    """Build lightweight GPU tokens from torch-native superpixel labels.

    This is a practical baseline for neck-injection experiments when the full
    SPiT interpolation/histogram token path is too expensive or unavailable. It
    keeps everything on the input tensor device and represents each superpixel by
    mean RGB and mean normalized position, then learns an embedding projection.
    """

    def __init__(
        self,
        algorithm: str = "torch_slic",
        emb_dim: int = 64,
        image_size: int | tuple[int, int] | list[int] | None = 160,
        n_segments: int = 300,
        compactness: float = 1.0,
        num_iters: int = 7,
        jitter: float = 0.35,
        seed: int = 0,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_size = _parse_size(image_size)
        self.algorithm_name = algorithm
        self.algorithm = self._build_algorithm(
            algorithm=algorithm,
            n_segments=n_segments,
            compactness=compactness,
            num_iters=num_iters,
            jitter=jitter,
            seed=seed,
        )
        hidden_dim = hidden_dim or max(emb_dim, 32)
        self.embedder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    @staticmethod
    def _build_algorithm(
        algorithm: str,
        n_segments: int,
        compactness: float,
        num_iters: int,
        jitter: float,
        seed: int,
    ):
        if algorithm in {"torch_slic", "gpu_slic"}:
            return TorchSLICSuperpixelAlgorithm(
                n_segments=n_segments,
                compactness=compactness,
                num_iters=num_iters,
            )
        if algorithm in {"torch_grid", "gpu_grid"}:
            return TorchGridSuperpixelAlgorithm(n_segments=n_segments)
        if algorithm in {"torch_voronoi", "gpu_voronoi"}:
            return TorchVoronoiSuperpixelAlgorithm(
                n_segments=n_segments,
                jitter=jitter,
                seed=seed,
            )
        raise ValueError(
            "RegionFeatureTokenProvider supports torch_slic, torch_grid, and "
            f"torch_voronoi. Got {algorithm!r}."
        )

    @property
    def emb_dim(self) -> int:
        last = self.embedder[-1]
        assert isinstance(last, nn.Linear)
        return int(last.out_features)

    def forward(self, rgb: torch.Tensor) -> TokenBundle:
        if rgb.ndim != 4 or rgb.shape[1] < 3:
            raise ValueError(f"Expected Bx3xHxW RGB input, got shape {tuple(rgb.shape)}.")

        rgb = rgb[:, :3].detach().float().clamp(0.0, 1.0)
        if self.image_size is not None and tuple(rgb.shape[-2:]) != self.image_size:
            rgb = F.interpolate(rgb, size=self.image_size, mode="bilinear", align_corners=False)

        with torch.no_grad():
            labels = self.algorithm._segment_batch(rgb).long()
            local_labels = _local_relabel(labels)

        raw_tokens, valid_mask = self._region_statistics(rgb, local_labels)
        embed_parameter = next(self.embedder.parameters())
        raw_tokens = raw_tokens.to(device=embed_parameter.device, dtype=embed_parameter.dtype)
        tokens = self.embedder(raw_tokens)
        return TokenBundle(tokens=tokens, valid_mask=valid_mask, labels=local_labels)

    @staticmethod
    def _region_statistics(rgb: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = rgb.shape
        device = rgb.device
        dtype = rgb.dtype
        max_regions = int(labels.max().item())

        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy, xx], dim=0).expand(batch, -1, -1, -1)
        ones = torch.ones((batch, 1, height, width), device=device, dtype=dtype)
        features = torch.cat([rgb, coords, ones], dim=1)
        feature_dim = int(features.shape[1])

        raw = torch.zeros((batch, max_regions + 1, feature_dim), device=device, dtype=dtype)
        valid = torch.zeros((batch, max_regions + 1), device=device, dtype=torch.bool)
        valid[:, 0] = True

        for batch_index in range(batch):
            flat_labels = labels[batch_index].reshape(-1)
            flat_features = features[batch_index].permute(1, 2, 0).reshape(-1, feature_dim)
            sums = torch.zeros((max_regions + 1, feature_dim), device=device, dtype=dtype)
            sums.scatter_add_(0, flat_labels[:, None].expand(-1, feature_dim), flat_features)
            counts = sums[:, -1:].clamp_min(1.0)
            means = sums / counts
            raw[batch_index] = means
            valid[batch_index, : int(flat_labels.max().item()) + 1] = True
            raw[batch_index, 0] = flat_features.mean(dim=0)

        return raw, valid


class SPiTLabelTokenProvider(nn.Module):
    """Build lightweight tokens from Marius/SPiT superpixel labels.

    This uses SPiT's content-aware hierarchical superpixel tokenizer, then
    represents each produced region with mean RGB and mean normalized position.
    It is much lighter than the full SPiT interpolation/histogram/masked-ViT
    token path, but still tests whether SPiT superpixel boundaries help YOLO
    neck features.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        image_size: int | tuple[int, int] | list[int] | None = 160,
        lambda_grad: float = 27.8,
        lambda_col: float = 10.0,
        maxlvl: int = 4,
        drop_delta: bool = False,
        bbox_reg: bool = True,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        SPiTDenseEdgeAlgorithm._ensure_local_spit_on_path()
        from spit.tokenizer.tokenizer import superpixel_tokenizer

        self.image_size = _parse_size(image_size)
        self.lambda_grad = float(lambda_grad)
        self.lambda_col = float(lambda_col)
        self.maxlvl = int(maxlvl)
        self.drop_delta = bool(drop_delta)
        self.bbox_reg = bool(bbox_reg)
        self.superpixel_tokenizer = superpixel_tokenizer

        hidden_dim = hidden_dim or max(emb_dim, 32)
        self.embedder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    @property
    def emb_dim(self) -> int:
        last = self.embedder[-1]
        assert isinstance(last, nn.Linear)
        return int(last.out_features)

    def forward(self, rgb: torch.Tensor) -> TokenBundle:
        if rgb.ndim != 4 or rgb.shape[1] < 3:
            raise ValueError(f"Expected Bx3xHxW RGB input, got shape {tuple(rgb.shape)}.")

        rgb = rgb[:, :3].detach().float().clamp(0.0, 1.0)
        if self.image_size is not None and tuple(rgb.shape[-2:]) != self.image_size:
            rgb = F.interpolate(rgb, size=self.image_size, mode="bilinear", align_corners=False)

        with torch.no_grad():
            _, labels, _ = self.superpixel_tokenizer(
                rgb,
                self.lambda_grad,
                self.lambda_col,
                self.drop_delta,
                self.bbox_reg,
                False,
                self.maxlvl,
            )
            local_labels = _local_relabel(labels.long())

        raw_tokens, valid_mask = RegionFeatureTokenProvider._region_statistics(rgb, local_labels)
        embed_parameter = next(self.embedder.parameters())
        raw_tokens = raw_tokens.to(device=embed_parameter.device, dtype=embed_parameter.dtype)
        tokens = self.embedder(raw_tokens)
        return TokenBundle(tokens=tokens, valid_mask=valid_mask, labels=local_labels)
