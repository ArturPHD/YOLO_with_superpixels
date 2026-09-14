"""SP-11 train-only foreground structural supervision for Ultralytics YOLO.

This deliberately does not alter the historical ``auxiliary_loss`` strategy.
SP-11 predicts foregroundness from P3 and uses superpixel labels only to make
predictions locally consistent inside pure foreground/background regions.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...superpixels.algorithms import SuperpixelAlgorithm
from ..auxiliary_loss.patch import find_auxiliary_loss_target
from ..base import StrategyInfo


@dataclass
class StructuralSupervisionConfig:
    """Configuration for SP-11 foreground and region-consistency supervision."""

    target_layer: int | None = None
    enabled: bool = True
    fail_without_image: bool = True
    lambda_fg: float = 0.05
    lambda_sp: float = 0.05
    foreground_loss_type: str = "balanced_bce"
    scale_by_batch: bool = True
    head_hidden_channels: int | None = None
    head_init: str = "tiny"
    auxiliary_init_seed: int | None = 0
    region_algorithm: str = "torch_slic"
    region_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"n_segments": 300, "compactness": 1.0, "num_iters": 7}
    )
    purity_threshold: float = 0.9
    min_region_size: int = 4
    region_balance_policy: str = "balanced_fg_bg"
    diagnostic_batches: int = 0
    diagnostic_output_path: str | None = None
    diagnostic_gradient_norms: bool = False
    runtime_profile: bool = False
    base_model_path: str | None = None
    base_model_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.lambda_fg < 0 or self.lambda_sp < 0:
            raise ValueError("lambda_fg and lambda_sp must be non-negative.")
        if self.foreground_loss_type not in {"bce", "balanced_bce"}:
            raise ValueError("foreground_loss_type must be 'bce' or 'balanced_bce'.")
        if self.head_init not in {"zero", "tiny", "default"}:
            raise ValueError("head_init must be 'zero', 'tiny', or 'default'.")
        if self.region_algorithm not in {"torch_slic", "torch_grid"}:
            raise ValueError("SP-11 region_algorithm must be 'torch_slic' or 'torch_grid'.")
        if not 0.5 <= self.purity_threshold <= 1.0:
            raise ValueError("purity_threshold must be between 0.5 and 1.0.")
        if self.min_region_size < 1:
            raise ValueError("min_region_size must be positive.")
        if self.region_balance_policy not in {"balanced_fg_bg", "all_regions_equal"}:
            raise ValueError("region_balance_policy must be 'balanced_fg_bg' or 'all_regions_equal'.")
        if self.diagnostic_batches < 0:
            raise ValueError("diagnostic_batches must be non-negative.")
        self.region_kwargs = dict(self.region_kwargs)

    @property
    def requested_n_segments(self) -> int | None:
        value = self.region_kwargs.get("n_segments")
        return int(value) if value is not None else None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, values: dict[str, Any]) -> "StructuralSupervisionConfig":
        return cls(**dict(values))

    def build_region_algorithm(self) -> SuperpixelAlgorithm:
        from ...integrations.ultralytics import _build_algorithm_from_name

        return _build_algorithm_from_name(self.region_algorithm, self.region_kwargs)


@dataclass
class StructuralSupervisionPatchResult:
    target_layer: int
    target_module: str
    detect_layer: int | None
    detect_sources: list[int]
    foreground_channels: int = 1
    already_patched: bool = False


class StructuralSupervisionContext:
    """Per-forward mailbox shared by capture hook, P3 wrapper, and criterion."""

    def __init__(self) -> None:
        self.image: torch.Tensor | None = None
        self.foreground_logits: torch.Tensor | None = None
        self.feature_shape: tuple[int, int] | None = None
        self.last_fg_loss: float | None = None
        self.last_sp_loss: float | None = None
        self.last_region_stats: dict[str, float] = _empty_region_stats()
        self.diagnostic_count = 0
        self.diagnostic_gradient_count = 0
        self.diagnostic_gradient_norms = False
        self.diagnostic_output_path: str | None = None
        self.structural_preprocess_seconds = 0.0
        self.structural_preprocess_calls = 0

    def set_image(self, image: torch.Tensor) -> None:
        self.image = image[:, :3]

    def clear_runtime_tensors(self) -> None:
        self.image = None
        self.foreground_logits = None
        self.feature_shape = None

    def configure_diagnostics(self, config: StructuralSupervisionConfig) -> None:
        self.diagnostic_gradient_norms = config.diagnostic_gradient_norms
        self.diagnostic_output_path = config.diagnostic_output_path

    def record_logit_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Append an optional P3 foreground-logit gradient norm after backward."""

        if not self.diagnostic_gradient_norms or not self.diagnostic_output_path:
            return gradient
        output = Path(self.diagnostic_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "gradient",
            "batch": self.diagnostic_gradient_count,
            "p3_foreground_logits_gradient_l2": float(gradient.detach().float().norm().cpu()),
        }
        self.diagnostic_gradient_count += 1
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        return gradient


class ForegroundnessHead(nn.Module):
    """Small P3 head that emits one class-agnostic foreground logit channel."""

    def __init__(self, in_channels: int, hidden_channels: int | None = None, init: str = "tiny") -> None:
        super().__init__()
        hidden = hidden_channels or min(128, max(16, in_channels // 2))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        first, last = self.net[0], self.net[-1]
        assert isinstance(first, nn.Conv2d) and isinstance(last, nn.Conv2d)
        nn.init.kaiming_normal_(first.weight, nonlinearity="relu")
        nn.init.zeros_(first.bias)
        if init == "zero":
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif init == "tiny":
            nn.init.normal_(last.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(last.bias)
        elif init == "default":
            nn.init.kaiming_normal_(last.weight, nonlinearity="linear")
            nn.init.zeros_(last.bias)
        else:
            raise ValueError("init must be 'zero', 'tiny', or 'default'.")

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class StructuralSupervisionWrapper(nn.Module):
    """Wrap P3 and queue logits only; the criterion owns all SP-11 losses."""

    def __init__(
        self,
        module: nn.Module,
        head: ForegroundnessHead,
        context: StructuralSupervisionContext,
        enabled: bool = True,
        fail_without_image: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.head = head
        self.context = context
        self.enabled = enabled
        self.fail_without_image = fail_without_image
        self.f = getattr(module, "f", -1)
        self.i = getattr(module, "i", None)
        self.type = getattr(module, "type", module.__class__.__name__)
        self.np = getattr(module, "np", sum(parameter.numel() for parameter in self.parameters()))

    def forward(self, x):
        output = self.module(x)
        if not self.training or not self.enabled or not isinstance(output, torch.Tensor):
            if not self.training:
                self.context.clear_runtime_tensors()
            return output
        if self.context.image is None and self.fail_without_image:
            raise RuntimeError("SP-11 did not capture the augmented RGB input before P3.")
        self.context.foreground_logits = self.head(output)
        if self.context.diagnostic_gradient_norms:
            self.context.foreground_logits.register_hook(self.context.record_logit_gradient)
        self.context.feature_shape = tuple(int(value) for value in output.shape[-2:])
        return output


class StructuralInputCaptureHook:
    """Capture the actual augmented RGB input only for train-time forwards."""

    def __init__(self, context: StructuralSupervisionContext) -> None:
        self.context = context

    def __call__(self, module: nn.Module, args) -> None:
        if not module.training:
            self.context.clear_runtime_tensors()
        elif args and isinstance(args[0], torch.Tensor):
            self.context.set_image(args[0])


def foreground_target_from_boxes(
    batch: dict[str, Any],
    size: tuple[int, int],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize union-of-boxes foreground targets at feature-map resolution.

    Standard Ultralytics detection batches use normalized ``xywh`` boxes after
    augmentations. Absolute xywh inputs are also accepted for direct callers.
    """

    image = batch.get("img")
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("SP-11 requires batch['img'] as a BxCxHxW tensor.")
    batch_size, _, image_height, image_width = image.shape
    height, width = int(size[0]), int(size[1])
    target_device = device or image.device
    target = torch.zeros((batch_size, 1, height, width), device=target_device, dtype=dtype)
    boxes, indices = batch.get("bboxes"), batch.get("batch_idx")
    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        return target
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or not isinstance(indices, torch.Tensor):
        raise ValueError("SP-11 expects bboxes=Nx4 xywh and matching batch_idx.")
    boxes = boxes.detach().to(device=target_device, dtype=torch.float32)
    indices = indices.detach().to(device=target_device, dtype=torch.long).reshape(-1)
    if len(indices) != len(boxes):
        raise ValueError("batch_idx and bboxes must contain the same number of entries.")
    # Keep the normalized/absolute compatibility rule, but do all box work on
    # the target device. In particular, no per-object .item()/.tolist() causes
    # CUDA synchronisation in a dense COCO batch.
    normalized = boxes.abs().amax() <= 1.5
    normalized_scale = boxes.new_tensor((width, height, width, height))
    absolute_scale = boxes.new_tensor((width / image_width, height / image_height) * 2)
    xywh = boxes * torch.where(normalized, normalized_scale, absolute_scale)
    left = torch.floor(xywh[:, 0] - xywh[:, 2] / 2).long().clamp(0, width)
    right = torch.ceil(xywh[:, 0] + xywh[:, 2] / 2).long().clamp(0, width)
    top = torch.floor(xywh[:, 1] - xywh[:, 3] / 2).long().clamp(0, height)
    bottom = torch.ceil(xywh[:, 1] + xywh[:, 3] / 2).long().clamp(0, height)
    valid = (indices >= 0) & (indices < batch_size) & (left < right) & (top < bottom)
    y = torch.arange(height, device=target_device).view(1, height, 1)
    x = torch.arange(width, device=target_device).view(1, 1, width)
    masks = (
        valid[:, None, None]
        & (x >= left[:, None, None])
        & (x < right[:, None, None])
        & (y >= top[:, None, None])
        & (y < bottom[:, None, None])
    )
    union = torch.zeros((batch_size, height, width), device=target_device, dtype=dtype)
    safe_indices = indices.clamp(0, max(batch_size - 1, 0)).view(-1, 1, 1).expand_as(masks)
    union.scatter_reduce_(0, safe_indices, masks.to(dtype), reduce="amax", include_self=True)
    return union[:, None]


def foreground_bce_loss(logits: torch.Tensor, target: torch.Tensor, loss_type: str = "balanced_bce") -> torch.Tensor:
    """Foreground BCE, optionally rebalanced per batch for sparse foreground."""

    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, target)
    if loss_type != "balanced_bce":
        raise ValueError("loss_type must be 'bce' or 'balanced_bce'.")
    with torch.no_grad():
        positive, negative = target, 1.0 - target
        pos_fraction = positive.mean().clamp_min(1e-4)
        neg_fraction = negative.mean().clamp_min(1e-4)
        weights = positive * (0.5 / pos_fraction) + negative * (0.5 / neg_fraction)
        weights = weights / weights.mean().clamp_min(1e-6)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()


def labels_on_feature_grid(algorithm: SuperpixelAlgorithm, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Create labels from final augmented RGB, then nearest-resize labels to P3."""

    labels_for_batch = getattr(algorithm, "labels_for_batch", None)
    if labels_for_batch is None:
        raise ValueError(f"{algorithm.__class__.__name__} does not expose labels_for_batch().")
    with torch.no_grad():
        labels = labels_for_batch(image[:, :3].detach())
        if labels.ndim != 3:
            raise ValueError(f"labels_for_batch must return BxHxW, got {tuple(labels.shape)}.")
        if tuple(labels.shape[-2:]) != tuple(size):
            labels = F.interpolate(labels[:, None].float(), size=size, mode="nearest")[:, 0]
        return labels.long()


def region_consistency_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    foreground_target: torch.Tensor,
    purity_threshold: float = 0.9,
    min_region_size: int = 4,
    balance_policy: str = "balanced_fg_bg",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Variance inside pure regions with explicit foreground/background balancing."""

    if probabilities.ndim != 4 or probabilities.shape[1] != 1:
        raise ValueError("probabilities must be Bx1xHxW.")
    if labels.shape != probabilities.shape[:1] + probabilities.shape[-2:]:
        raise ValueError("labels must be BxHxW matching probabilities.")
    if foreground_target.shape != probabilities.shape:
        raise ValueError("foreground_target must match probabilities.")
    if balance_policy not in {"balanced_fg_bg", "all_regions_equal"}:
        raise ValueError("balance_policy must be 'balanced_fg_bg' or 'all_regions_equal'.")
    foreground_variances: list[torch.Tensor] = []
    background_variances: list[torch.Tensor] = []
    stats: dict[str, Any] = _empty_region_stats()
    for index in range(probabilities.shape[0]):
        _, inverse = torch.unique(labels[index].reshape(-1), sorted=True, return_inverse=True)
        count = int(inverse.max().item()) + 1 if inverse.numel() else 0
        values, target = probabilities[index, 0].reshape(-1), foreground_target[index, 0].reshape(-1)
        counts = torch.bincount(inverse, minlength=count).to(dtype=values.dtype)
        sums = torch.zeros(count, device=values.device, dtype=values.dtype).scatter_add_(0, inverse, values)
        squares = torch.zeros(count, device=values.device, dtype=values.dtype).scatter_add_(0, inverse, values.square())
        targets = torch.zeros(count, device=values.device, dtype=values.dtype).scatter_add_(0, inverse, target)
        means = sums / counts.clamp_min(1.0)
        variance = (squares / counts.clamp_min(1.0) - means.square()).clamp_min(0.0)
        purity = targets / counts.clamp_min(1.0)
        pure_foreground = purity >= purity_threshold
        pure_background = purity <= 1.0 - purity_threshold
        pure = pure_foreground | pure_background
        large_enough = counts >= float(min_region_size)
        active_foreground = pure_foreground & large_enough
        active_background = pure_background & large_enough
        if active_foreground.any():
            foreground_variances.append(variance[active_foreground])
        if active_background.any():
            background_variances.append(variance[active_background])
        stats["actual_regions"] += float(count)
        stats["active_foreground_regions"] += float(active_foreground.sum().item())
        stats["active_background_regions"] += float(active_background.sum().item())
        stats["ignored_ambiguous_regions"] += float((~pure).sum().item())
        stats["ignored_small_regions"] += float((pure & ~large_enough).sum().item())
        stats["region_area_sum"] += float(counts.sum().item())
        stats["region_area_count"] += float(count)
        stats["region_area_values"].append(counts.detach())
    zero = probabilities.sum() * 0.0
    foreground_loss = torch.cat(foreground_variances).mean() if foreground_variances else zero
    background_loss = torch.cat(background_variances).mean() if background_variances else zero
    if balance_policy == "all_regions_equal":
        all_variances = foreground_variances + background_variances
        loss = torch.cat(all_variances).mean() if all_variances else zero
    elif foreground_variances and background_variances:
        loss = 0.5 * (foreground_loss + background_loss)
    else:
        loss = foreground_loss if foreground_variances else background_loss
    stats["foreground_region_variance_raw"] = float(foreground_loss.detach().float().cpu())
    stats["background_region_variance_raw"] = float(background_loss.detach().float().cpu())
    return loss, _finalize_region_stats(stats)


class StructuralSupervisionCriterion:
    """YOLO criterion plus separate SP-11 foreground and consistency terms."""

    def __init__(self, base_criterion: Any, context: StructuralSupervisionContext, config: StructuralSupervisionConfig):
        self.base_criterion, self.context, self.config = base_criterion, context, config
        self.region_algorithm = config.build_region_algorithm() if config.lambda_sp > 0 else None

    def __call__(self, preds, batch):
        try:
            loss, loss_items = self.base_criterion(preds, batch)
            zero = _zero_scalar(loss_items)
            fg_raw, sp_raw = zero, zero
            fg_weighted, sp_weighted = zero, zero
            stats: dict[str, Any] = _empty_region_stats()
            logits, image = self.context.foreground_logits, self.context.image
            base_loss = loss
            scale = _batch_scale(batch, self.config.scale_by_batch)
            if self.config.enabled and logits is not None:
                target = foreground_target_from_boxes(batch, tuple(logits.shape[-2:]), logits.device, torch.float32)
                fg_loss = foreground_bce_loss(logits.float(), target, self.config.foreground_loss_type)
                fg_raw = fg_loss.detach()
                self.context.last_fg_loss = float(fg_loss.detach().float().cpu())
                if self.config.lambda_fg > 0:
                    fg_weighted = fg_loss.detach() * (scale * self.config.lambda_fg)
                    loss = loss + scale * self.config.lambda_fg * fg_loss
                if self.config.lambda_sp > 0:
                    if image is None:
                        if self.config.fail_without_image:
                            raise RuntimeError("SP-11 needs captured augmented RGB to produce region labels.")
                    else:
                        assert self.region_algorithm is not None
                        labels = self._labels_on_feature_grid(image, logits)
                        sp_loss, stats = region_consistency_loss(
                            torch.sigmoid(logits.float()), labels, target,
                            purity_threshold=self.config.purity_threshold,
                            min_region_size=self.config.min_region_size,
                            balance_policy=self.config.region_balance_policy,
                        )
                        stats["requested_regions"] = float(
                            (self.config.requested_n_segments or 0) * logits.shape[0]
                        )
                        self.context.last_sp_loss = float(sp_loss.detach().float().cpu())
                        self.context.last_region_stats = stats
                        sp_raw = sp_loss.detach()
                        sp_weighted = sp_loss.detach() * (scale * self.config.lambda_sp)
                        loss = loss + scale * self.config.lambda_sp * sp_loss
            else:
                self.context.last_fg_loss, self.context.last_sp_loss = None, None
                self.context.last_region_stats = _finalize_region_stats(stats)
            final_stats = _finalize_region_stats(stats)
            self.context.last_region_stats = final_stats
            aux_total = fg_weighted + sp_weighted
            self._record_diagnostic(base_loss, fg_raw, fg_weighted, sp_raw, sp_weighted, aux_total)
            return loss, _append_items(loss_items, fg_raw, fg_weighted, sp_raw, sp_weighted, aux_total, final_stats)
        finally:
            self.context.clear_runtime_tensors()

    def __getattr__(self, name: str):
        base = self.__dict__.get("base_criterion")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _labels_on_feature_grid(self, image: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        assert self.region_algorithm is not None
        if not self.config.runtime_profile:
            return labels_on_feature_grid(self.region_algorithm, image, tuple(logits.shape[-2:]))
        if image.device.type == "cuda":
            torch.cuda.synchronize(image.device)
        started = time.perf_counter()
        labels = labels_on_feature_grid(self.region_algorithm, image, tuple(logits.shape[-2:]))
        if image.device.type == "cuda":
            torch.cuda.synchronize(image.device)
        self.context.structural_preprocess_seconds += time.perf_counter() - started
        self.context.structural_preprocess_calls += 1
        return labels

    def _record_diagnostic(
        self,
        base_loss: torch.Tensor,
        fg_raw: torch.Tensor,
        fg_weighted: torch.Tensor,
        sp_raw: torch.Tensor,
        sp_weighted: torch.Tensor,
        aux_total: torch.Tensor,
    ) -> None:
        if self.context.diagnostic_count >= self.config.diagnostic_batches:
            return
        record = {
            "batch": self.context.diagnostic_count,
            "yolo_loss_raw": float(base_loss.detach().float().cpu()),
            "fg_aux_loss_raw": float(fg_raw.detach().float().cpu()),
            "fg_aux_loss_weighted": float(fg_weighted.detach().float().cpu()),
            "sp_consistency_loss_raw": float(sp_raw.detach().float().cpu()),
            "sp_consistency_loss_weighted": float(sp_weighted.detach().float().cpu()),
            "aux_total_weighted": float(aux_total.detach().float().cpu()),
            "gradient_norms": None,
        }
        self.context.diagnostic_count += 1
        if self.config.diagnostic_output_path:
            output = Path(self.config.diagnostic_output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")


def _empty_region_stats() -> dict[str, Any]:
    return {
        "requested_regions": 0.0, "actual_regions": 0.0,
        "active_foreground_regions": 0.0, "active_background_regions": 0.0,
        "ignored_ambiguous_regions": 0.0, "ignored_small_regions": 0.0,
        "region_area_sum": 0.0, "region_area_count": 0.0, "region_area_values": [],
        "mean_region_area": 0.0, "median_region_area": 0.0,
        "foreground_region_variance_raw": 0.0, "background_region_variance_raw": 0.0,
    }


def _finalize_region_stats(stats: dict[str, Any]) -> dict[str, float]:
    if "region_area_values" not in stats:
        return {name: float(value) for name, value in stats.items()}
    area_tensors = stats.pop("region_area_values")
    count, area_sum = max(float(stats.pop("region_area_count")), 1.0), float(stats.pop("region_area_sum"))
    stats["mean_region_area"] = area_sum / count
    areas = torch.cat(area_tensors).float() if area_tensors else None
    stats["median_region_area"] = float(areas.median().detach().cpu()) if areas is not None else 0.0
    return {name: float(value) for name, value in stats.items()}


def _zero_scalar(loss_items: Any) -> torch.Tensor:
    if isinstance(loss_items, torch.Tensor):
        return loss_items.new_zeros(())
    if isinstance(loss_items, dict):
        for value in loss_items.values():
            if isinstance(value, torch.Tensor):
                return value.new_zeros(())
    return torch.tensor(0.0)


def _append_items(
    loss_items: Any,
    fg_raw: torch.Tensor,
    fg_weighted: torch.Tensor,
    sp_raw: torch.Tensor,
    sp_weighted: torch.Tensor,
    aux_total: torch.Tensor,
    stats: dict[str, float],
):
    values = [fg_raw, fg_weighted, sp_raw, sp_weighted, aux_total] + [
        fg_raw.new_tensor(stats[name]) for name in (
            "requested_regions", "actual_regions", "active_foreground_regions", "active_background_regions",
            "ignored_ambiguous_regions", "ignored_small_regions", "mean_region_area", "median_region_area",
            "foreground_region_variance_raw", "background_region_variance_raw",
        )
    ]
    if isinstance(loss_items, torch.Tensor):
        return torch.cat([loss_items, *(value.to(loss_items).reshape(1) for value in values)])
    if isinstance(loss_items, list):
        return [*loss_items, *values]
    if isinstance(loss_items, tuple):
        return (*loss_items, *values)
    if isinstance(loss_items, dict):
        names = (
            "fg_aux_loss_raw", "fg_aux_loss_weighted", "sp_consistency_loss_raw",
            "sp_consistency_loss_weighted", "aux_total_weighted", "sp_requested_regions",
            "sp_actual_regions", "sp_active_foreground_regions", "sp_active_background_regions",
            "sp_ignored_ambiguous_regions", "sp_ignored_small_regions", "sp_mean_region_area",
            "sp_median_region_area", "sp_foreground_region_variance_raw",
            "sp_background_region_variance_raw",
        )
        return {**loss_items, **dict(zip(names, values))}
    return loss_items


def _batch_scale(batch: dict[str, Any], enabled: bool) -> int:
    image = batch.get("img")
    return int(image.shape[0]) if enabled and isinstance(image, torch.Tensor) else 1


def _module_list(model: nn.Module) -> nn.ModuleList | nn.Sequential:
    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)):
        raise ValueError("Expected an Ultralytics-style model with layers at model.model.")
    return layers


def _infer_module_out_channels(module: nn.Module) -> int:
    direct = _direct_out_channels(module)
    if direct is not None:
        return direct
    for attr_name in ("cv3", "cv2", "cv", "conv"):
        direct = _direct_out_channels(getattr(module, attr_name, None))
        if direct is not None:
            return direct
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return int(child.out_channels)
    raise ValueError(f"Could not infer output channels for {module.__class__.__name__}.")


def _direct_out_channels(module: Any) -> int | None:
    if isinstance(module, nn.Conv2d):
        return int(module.out_channels)
    conv = getattr(module, "conv", None)
    if isinstance(conv, nn.Conv2d):
        return int(conv.out_channels)
    value = getattr(module, "out_channels", None)
    return int(value) if isinstance(value, int) else None


def _module_device_dtype(module: nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    for parameter in module.parameters(recurse=True):
        return parameter.device, parameter.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.device, buffer.dtype
    return None, None


def _ensure_namespace_args(model: nn.Module) -> None:
    args = getattr(model, "args", None)
    try:
        from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
        values, namespace_type = vars(DEFAULT_CFG).copy(), IterableSimpleNamespace
    except ImportError:
        values, namespace_type = {}, SimpleNamespace
    if isinstance(args, dict):
        values.update(args)
    elif args is not None and hasattr(args, "__dict__"):
        values.update(vars(args))
    model.args = namespace_type(**values)


def _install_input_capture_hook(model: nn.Module, context: StructuralSupervisionContext) -> None:
    old_handle = getattr(model, "_yolo_superpixels_structural_capture_handle", None)
    if old_handle is not None:
        old_handle.remove()
    handle = _module_list(model)[0].register_forward_pre_hook(StructuralInputCaptureHook(context))
    setattr(model, "_yolo_superpixels_structural_capture_handle", handle)


def _make_head(config: StructuralSupervisionConfig, in_channels: int, device: torch.device | None, dtype: torch.dtype | None):
    if config.auxiliary_init_seed is None:
        head = ForegroundnessHead(in_channels, config.head_hidden_channels, config.head_init)
    else:
        devices = [device.index] if device is not None and device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(config.auxiliary_init_seed))
            if device is not None and device.type == "cuda":
                torch.cuda.manual_seed_all(int(config.auxiliary_init_seed))
            head = ForegroundnessHead(in_channels, config.head_hidden_channels, config.head_init)
    return head.to(device=device, dtype=dtype) if device is not None else head


def _install_structural_criterion(model: nn.Module, context: StructuralSupervisionContext, config: StructuralSupervisionConfig) -> None:
    context.configure_diagnostics(config)
    criterion = getattr(model, "criterion", None)
    if isinstance(criterion, StructuralSupervisionCriterion):
        criterion.context, criterion.config = context, config
        criterion.region_algorithm = config.build_region_algorithm() if config.lambda_sp > 0 else None
        return
    _ensure_namespace_args(model)
    model.criterion = StructuralSupervisionCriterion(criterion or model.init_criterion(), context, config)


def patch_structural_supervision(
    model: nn.Module,
    config: StructuralSupervisionConfig | None = None,
    install_criterion: bool = True,
) -> StructuralSupervisionPatchResult:
    """Install the SP-11 P3 foregroundness head and criterion wrapper."""

    config = config or StructuralSupervisionConfig()
    layers = _module_list(model)
    target_layer, detect_layer, detect_sources = find_auxiliary_loss_target(model, config.target_layer)
    target_module = layers[target_layer]
    if isinstance(target_module, StructuralSupervisionWrapper):
        target_module.enabled, target_module.fail_without_image = config.enabled, config.fail_without_image
        target_module.context.configure_diagnostics(config)
        _install_input_capture_hook(model, target_module.context)
        setattr(model, "_yolo_superpixels_structural_context", target_module.context)
        if install_criterion:
            install_structural_criterion(model, config)
        return StructuralSupervisionPatchResult(
            target_layer, target_module.module.__class__.__name__, detect_layer, detect_sources, already_patched=True
        )
    device, dtype = _module_device_dtype(target_module)
    context = StructuralSupervisionContext()
    context.configure_diagnostics(config)
    wrapper = StructuralSupervisionWrapper(
        target_module,
        _make_head(config, _infer_module_out_channels(target_module), device, dtype),
        context,
        config.enabled,
        config.fail_without_image,
    )
    if device is not None:
        wrapper = wrapper.to(device=device, dtype=dtype)
    layers[target_layer] = wrapper
    _install_input_capture_hook(model, context)
    setattr(model, "_yolo_superpixels_structural_context", context)
    if install_criterion:
        install_structural_criterion(model, config)
    return StructuralSupervisionPatchResult(target_layer, target_module.__class__.__name__, detect_layer, detect_sources)


def install_structural_criterion(model: nn.Module, config: StructuralSupervisionConfig | None = None) -> None:
    context = getattr(model, "_yolo_superpixels_structural_context", None)
    if not isinstance(context, StructuralSupervisionContext):
        raise ValueError("patch_structural_supervision must run before install_structural_criterion.")
    _install_structural_criterion(model, context, config or StructuralSupervisionConfig())


def strip_structural_supervision(model: nn.Module, remove_criterion_wrapper: bool = True) -> int:
    """Remove SP-11 train-only modules, preserving ordinary RGB detector output."""

    removed, layers = 0, _module_list(model)
    for index, module in enumerate(list(layers)):
        if isinstance(module, StructuralSupervisionWrapper):
            layers[index], removed = module.module, removed + 1
    handle = getattr(model, "_yolo_superpixels_structural_capture_handle", None)
    if handle is not None:
        handle.remove()
        setattr(model, "_yolo_superpixels_structural_capture_handle", None)
    if remove_criterion_wrapper and isinstance(getattr(model, "criterion", None), StructuralSupervisionCriterion):
        model.criterion = model.criterion.base_criterion
    context = getattr(model, "_yolo_superpixels_structural_context", None)
    if isinstance(context, StructuralSupervisionContext):
        context.clear_runtime_tensors()
    return removed


@dataclass
class StructuralSupervisionStrategy:
    """SP-11: train-only foregroundness plus pure-region consistency."""

    config: StructuralSupervisionConfig = field(default_factory=StructuralSupervisionConfig)
    info: StrategyInfo = StrategyInfo(
        name="structural_supervision",
        package="yolo_superpixels.strategies.structural_supervision",
        description="Train-only foreground supervision with superpixel consistency.",
        status="experimental",
    )

    def patch_model(self, model: nn.Module) -> StructuralSupervisionPatchResult:
        return patch_structural_supervision(model, self.config)

    def prepare_training(self) -> None:
        return None
