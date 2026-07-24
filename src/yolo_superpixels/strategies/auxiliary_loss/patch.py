"""Auxiliary superpixel-structure loss for Ultralytics-style YOLO models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...superpixels.adjacency import AdjacencyChannelBuilder, Direction, tensor_labels_to_adjacency
from ...superpixels.algorithms import SuperpixelAlgorithm
from ..base import StrategyInfo


@dataclass
class AuxiliaryLossConfig:
    """Serializable configuration for training-only superpixel supervision."""

    algorithm_names: list[str] = field(default_factory=lambda: ["torch_slic"])
    algorithm_kwargs: dict[str, Any] | list[dict[str, Any]] = field(default_factory=dict)
    channel_order: tuple[Direction, Direction] = ("right", "down")
    target_layer: int | None = None
    enabled: bool = True
    fail_without_image: bool = False
    fail_open: bool = False
    loss_weight: float = 0.05
    edge_value_scale: float = 1.0
    target_mode: str = "auto"
    target_resize: str = "nearest"
    clamp_target: bool = True
    loss_type: str = "balanced_bce"
    head_hidden_channels: int | None = None
    head_init: str = "tiny"
    scale_by_batch: bool = True

    def __post_init__(self) -> None:
        self.algorithm_names = [name for name in self.algorithm_names if name.lower() not in {"none", "rgb", ""}]
        self.channel_order = tuple(self.channel_order)  # type: ignore[assignment]
        if self.target_mode not in {"auto", "labels", "edges"}:
            raise ValueError("target_mode must be 'auto', 'labels', or 'edges'.")
        if self.target_resize not in {"nearest", "area", "bilinear"}:
            raise ValueError("target_resize must be 'nearest', 'area', or 'bilinear'.")
        if self.loss_type not in {"bce", "balanced_bce"}:
            raise ValueError("loss_type must be 'bce' or 'balanced_bce'.")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be non-negative.")

    @property
    def target_channels(self) -> int:
        return 2 * len(self.algorithm_names)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, values: dict[str, Any]) -> "AuxiliaryLossConfig":
        copied = dict(values)
        if "channel_order" in copied:
            copied["channel_order"] = tuple(copied["channel_order"])
        return cls(**copied)

    def _normalize_algorithm_kwargs(self) -> list[dict[str, Any]]:
        count = len(self.algorithm_names)
        if count == 0:
            return []
        if isinstance(self.algorithm_kwargs, list):
            values = [dict(value) for value in self.algorithm_kwargs]
            if len(values) != count:
                raise ValueError(f"Expected {count} algorithm kwargs entries, got {len(values)}.")
            return values
        if count == 1:
            return [dict(self.algorithm_kwargs)]
        if self.algorithm_kwargs:
            raise ValueError("Pass algorithm_kwargs as a list when using multiple algorithms.")
        return [{} for _ in range(count)]

    def build_algorithms(self) -> tuple[SuperpixelAlgorithm, ...]:
        from ...integrations.ultralytics import _build_algorithm_from_name

        return tuple(
            _build_algorithm_from_name(name, kwargs)
            for name, kwargs in zip(self.algorithm_names, self._normalize_algorithm_kwargs())
        )

    def build_target_builder(self) -> "AuxiliarySuperpixelTargetBuilder":
        return AuxiliarySuperpixelTargetBuilder(
            channel_builder=AdjacencyChannelBuilder(
                algorithm=self.build_algorithms(),
                channel_order=self.channel_order,
                edge_representation="same",
                edge_value_scale=self.edge_value_scale,
                fail_open=self.fail_open,
                enabled=True,
            ),
            target_mode=self.target_mode,
            resize=self.target_resize,
            clamp_target=self.clamp_target,
        )


@dataclass
class AuxiliaryLossPatchResult:
    """Information about an installed auxiliary-loss patch."""

    target_layer: int
    target_module: str
    detect_layer: int | None
    detect_sources: list[int]
    target_channels: int
    already_patched: bool = False


class AuxiliaryLossContext:
    """Small pickle-safe runtime mailbox shared by hooks, wrapper, and criterion."""

    def __init__(self) -> None:
        self.image: torch.Tensor | None = None
        self.loss: torch.Tensor | None = None
        self.last_loss: float | None = None
        self.last_target_shape: tuple[int, ...] | None = None

    def set_image(self, image: torch.Tensor) -> None:
        self.image = image[:, :3]

    def clear_image(self) -> None:
        self.image = None

    def clear_loss(self) -> None:
        self.loss = None

    def clear_runtime_tensors(self) -> None:
        self.image = None
        self.loss = None


class AuxiliarySuperpixelTargetBuilder:
    """Build low-resolution right/down superpixel adjacency targets."""

    def __init__(
        self,
        channel_builder: AdjacencyChannelBuilder,
        target_mode: str = "auto",
        resize: str = "nearest",
        clamp_target: bool = True,
    ) -> None:
        self.channel_builder = channel_builder
        self.target_mode = target_mode
        self.resize = resize
        self.clamp_target = clamp_target

    @property
    def target_channels(self) -> int:
        return self.channel_builder.extra_channels

    def __call__(
        self,
        image: torch.Tensor,
        size: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        with torch.no_grad():
            edges = self._edges_for_grid(image[:, :3].detach(), size=size)
            edges = edges.to(device=image.device, dtype=torch.float32, non_blocking=True)
            edges = self.channel_builder.transform_edges(edges)
            if self.clamp_target:
                edges = edges.clamp(0.0, 1.0)
            return edges.to(dtype=dtype)

    def _edges_for_grid(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        outputs = []
        for algorithm in self.channel_builder.algorithms:
            labels_builder = getattr(algorithm, "labels_for_batch", None)
            if labels_builder is not None and self.target_mode in {"auto", "labels"}:
                labels = labels_builder(image)
                if labels.ndim != 3:
                    raise ValueError(
                        f"labels_for_batch must return BxHxW labels, got shape {tuple(labels.shape)}."
                    )
                if tuple(labels.shape[-2:]) != tuple(size):
                    labels = F.interpolate(labels[:, None].float(), size=size, mode="nearest")[:, 0].long()
                outputs.append(tensor_labels_to_adjacency(labels, channel_order=self.channel_builder.channel_order))
                continue
            if self.target_mode == "labels":
                raise ValueError(
                    f"{algorithm.__class__.__name__} does not expose labels_for_batch(), "
                    "so it cannot be used with target_mode='labels'."
                )
            single_builder = AdjacencyChannelBuilder(
                algorithm=algorithm,
                channel_order=self.channel_builder.channel_order,
                edge_representation="same",
                edge_value_scale=1.0,
                fail_open=self.channel_builder.fail_open,
                enabled=True,
            )
            edges = single_builder.edges_for_batch(image)
            if tuple(edges.shape[-2:]) != tuple(size):
                edges = _resize_edges(edges, size=size, mode=self.resize)
            outputs.append(edges)
        return torch.cat(outputs, dim=1)


class AuxiliarySuperpixelHead(nn.Module):
    """Tiny prediction head that asks a YOLO feature map to reproduce superpixel adjacency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        init: str = "tiny",
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("AuxiliarySuperpixelHead needs at least one target channel.")
        if hidden_channels is None:
            hidden_channels = min(128, max(16, min(in_channels, out_channels * 8)))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )
        self._init_weights(init)

    def _init_weights(self, init: str) -> None:
        first = self.net[0]
        last = self.net[2]
        if isinstance(first, nn.Conv2d):
            nn.init.kaiming_normal_(first.weight, nonlinearity="relu")
            nn.init.zeros_(first.bias)
        if not isinstance(last, nn.Conv2d):
            return
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
            raise ValueError("head_init must be 'zero', 'tiny', or 'default'.")

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class AuxiliarySuperpixelWrapper(nn.Module):
    """Wrap one YOLO feature module and compute a training-only auxiliary loss."""

    def __init__(
        self,
        module: nn.Module,
        head: AuxiliarySuperpixelHead,
        target_builder: AuxiliarySuperpixelTargetBuilder,
        context: AuxiliaryLossContext,
        enabled: bool = True,
        fail_without_image: bool = False,
        loss_type: str = "balanced_bce",
    ) -> None:
        super().__init__()
        self.module = module
        self.head = head
        self.target_builder = target_builder
        self.context = context
        self.enabled = enabled
        self.fail_without_image = fail_without_image
        self.loss_type = loss_type
        self.f = getattr(module, "f", -1)
        self.i = getattr(module, "i", None)
        self.type = getattr(module, "type", module.__class__.__name__)
        self.np = getattr(module, "np", sum(parameter.numel() for parameter in self.parameters()))

    def forward(self, x):
        output = self.module(x)
        if not self.enabled or not self.training or not isinstance(output, torch.Tensor):
            return output

        image = self.context.image
        if image is None:
            if self.fail_without_image:
                raise RuntimeError(
                    "AuxiliarySuperpixelWrapper did not receive the original RGB image. "
                    "Make sure patch_auxiliary_loss installed its input capture hook."
                )
            return output

        image = image.to(device=output.device, non_blocking=True)
        target = self.target_builder(image, size=tuple(output.shape[-2:]), dtype=output.dtype)
        prediction = self.head(output)
        self.context.loss = _superpixel_auxiliary_loss(prediction.float(), target.float(), self.loss_type)
        self.context.last_target_shape = tuple(target.shape)
        self.context.clear_image()
        return output


class InputCaptureHook:
    """Pickle-safe forward pre-hook that stores the original RGB image."""

    def __init__(self, context: AuxiliaryLossContext) -> None:
        self.context = context

    def __call__(self, _module, args) -> None:
        if args and isinstance(args[0], torch.Tensor):
            self.context.set_image(args[0])


class AuxiliaryDetectionCriterion:
    """Wrap an Ultralytics detection criterion and add the queued auxiliary loss."""

    def __init__(
        self,
        base_criterion: Any,
        context: AuxiliaryLossContext,
        loss_weight: float,
        enabled: bool = True,
        scale_by_batch: bool = True,
    ) -> None:
        self.base_criterion = base_criterion
        self.context = context
        self.loss_weight = float(loss_weight)
        self.enabled = enabled
        self.scale_by_batch = scale_by_batch

    def __call__(self, preds, batch):
        loss, loss_items = self.base_criterion(preds, batch)
        aux_loss = self.context.loss
        aux_item = _zero_loss_item(loss_items)
        if self.enabled and self.loss_weight > 0 and aux_loss is not None:
            scale = _batch_size_from_batch(batch) if self.scale_by_batch else 1
            loss = loss + aux_loss * self.loss_weight * scale
            self.context.last_loss = float(aux_loss.detach().float().cpu())
            aux_item = aux_loss.detach() * self.loss_weight
        self.context.clear_loss()
        return loss, _append_loss_item(loss_items, aux_item)

    def __getattr__(self, name: str):
        base_criterion = self.__dict__.get("base_criterion")
        if base_criterion is None:
            raise AttributeError(name)
        return getattr(base_criterion, name)


def _resize_edges(edges: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "bilinear":
        return F.interpolate(edges, size=size, mode=mode, align_corners=False)
    return F.interpolate(edges, size=size, mode=mode)


def _superpixel_auxiliary_loss(prediction: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(prediction, target, reduction="mean")
    if loss_type != "balanced_bce":
        raise ValueError("loss_type must be 'bce' or 'balanced_bce'.")

    with torch.no_grad():
        positive = target
        negative = 1.0 - target
        positive_fraction = positive.mean(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
        negative_fraction = negative.mean(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
        weights = positive * (0.5 / positive_fraction) + negative * (0.5 / negative_fraction)
        weights = weights / weights.mean().clamp_min(1e-6)
    raw = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    return (raw * weights).mean()


def _zero_loss_item(loss_items):
    if isinstance(loss_items, torch.Tensor):
        return loss_items.new_zeros(())
    return torch.tensor(0.0)


def _append_loss_item(loss_items, item: torch.Tensor):
    item = item.detach()
    if isinstance(loss_items, torch.Tensor):
        return torch.cat([loss_items, item.to(device=loss_items.device, dtype=loss_items.dtype).reshape(1)])
    if isinstance(loss_items, list):
        return [*loss_items, item]
    if isinstance(loss_items, tuple):
        return (*loss_items, item)
    return loss_items


def _batch_size_from_batch(batch: dict[str, Any]) -> int:
    image = batch.get("img") if isinstance(batch, dict) else None
    if isinstance(image, torch.Tensor) and image.ndim >= 1:
        return int(image.shape[0])
    return 1


def _module_list(model: nn.Module) -> nn.ModuleList | nn.Sequential:
    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)):
        raise ValueError("Expected an Ultralytics-style model with layers at model.model.")
    return layers


def _as_source_list(sources: Any, detect_index: int) -> list[int]:
    if isinstance(sources, int):
        values = [sources]
    elif isinstance(sources, (list, tuple)):
        values = [int(value) for value in sources]
    else:
        raise ValueError(f"Unsupported Detect.f value: {sources!r}.")
    return [detect_index + value if value < 0 else value for value in values]


def find_auxiliary_loss_target(model: nn.Module, target_layer: int | None = None) -> tuple[int, int | None, list[int]]:
    """Find the default P3 feature feeding Detect, or validate an explicit layer."""

    layers = _module_list(model)
    if target_layer is not None:
        target = int(target_layer)
        if target < 0:
            target = len(layers) + target
        if target < 0 or target >= len(layers):
            raise ValueError(f"target_layer={target_layer} is outside model.model length {len(layers)}.")
        return target, None, []

    for detect_index in range(len(layers) - 1, -1, -1):
        module = layers[detect_index]
        if module.__class__.__name__.lower().endswith("detect") and hasattr(module, "f"):
            sources = _as_source_list(getattr(module, "f"), detect_index)
            if sources:
                return sources[0], detect_index, sources
    raise ValueError("Could not find a Detect module with source indices. Pass target_layer explicitly.")


def _install_input_capture_hook(model: nn.Module, context: AuxiliaryLossContext) -> None:
    old_handle = getattr(model, "_yolo_superpixels_aux_capture_handle", None)
    if old_handle is not None:
        old_handle.remove()
    first_layer = _module_list(model)[0]
    handle = first_layer.register_forward_pre_hook(InputCaptureHook(context))
    setattr(model, "_yolo_superpixels_aux_capture_handle", handle)


def _module_device_dtype(module: nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    for parameter in module.parameters(recurse=True):
        return parameter.device, parameter.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.device, buffer.dtype
    return None, None


def _infer_module_out_channels(module: nn.Module) -> int:
    direct = _direct_out_channels(module)
    if direct is not None:
        return direct
    for attr_name in ("cv3", "cv2", "cv", "conv"):
        child = getattr(module, attr_name, None)
        direct = _direct_out_channels(child)
        if direct is not None:
            return direct
    if isinstance(module, (nn.Sequential, nn.ModuleList)) and len(module) > 0:
        direct = _direct_out_channels(module[-1])
        if direct is not None:
            return direct
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return int(child.out_channels)
    raise ValueError(
        f"Could not infer output channels for {module.__class__.__name__}. "
        "Pass a target_layer whose module contains a Conv2d."
    )


def _direct_out_channels(module: Any) -> int | None:
    if isinstance(module, nn.Conv2d):
        return int(module.out_channels)
    conv = getattr(module, "conv", None)
    if isinstance(conv, nn.Conv2d):
        return int(conv.out_channels)
    value = getattr(module, "out_channels", None)
    if isinstance(value, int):
        return value
    return None


def _install_auxiliary_criterion(
    model: nn.Module,
    context: AuxiliaryLossContext,
    config: AuxiliaryLossConfig,
) -> None:
    criterion = getattr(model, "criterion", None)
    if isinstance(criterion, AuxiliaryDetectionCriterion):
        criterion.context = context
        criterion.loss_weight = float(config.loss_weight)
        criterion.enabled = config.enabled
        criterion.scale_by_batch = config.scale_by_batch
        return

    _ensure_namespace_args(model)
    base = criterion if criterion is not None else model.init_criterion()
    model.criterion = AuxiliaryDetectionCriterion(
        base_criterion=base,
        context=context,
        loss_weight=config.loss_weight,
        enabled=config.enabled,
        scale_by_batch=config.scale_by_batch,
    )


def _ensure_namespace_args(model: nn.Module) -> None:
    args = getattr(model, "args", None)
    try:
        from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

        values = vars(DEFAULT_CFG).copy()
        namespace_type = IterableSimpleNamespace
    except ImportError:
        values = {}
        namespace_type = SimpleNamespace
    if isinstance(args, dict):
        values.update(args)
    elif args is not None and hasattr(args, "__dict__"):
        values.update(vars(args))
    model.args = namespace_type(**values)


def patch_auxiliary_loss(
    model: nn.Module,
    config: AuxiliaryLossConfig | None = None,
    target_builder: AuxiliarySuperpixelTargetBuilder | None = None,
    install_criterion: bool = True,
) -> AuxiliaryLossPatchResult:
    """Patch an Ultralytics YOLO model in-place with a training-only auxiliary loss."""

    config = config or AuxiliaryLossConfig()
    if config.target_channels <= 0 and target_builder is None:
        raise ValueError("Auxiliary loss needs at least one algorithm name.")

    layers = _module_list(model)
    target_layer, detect_layer, detect_sources = find_auxiliary_loss_target(model, config.target_layer)
    target_module = layers[target_layer]

    if isinstance(target_module, AuxiliarySuperpixelWrapper):
        builder = target_builder or config.build_target_builder()
        if builder.target_channels != target_module.target_builder.target_channels:
            target_module = target_module.module
            layers[target_layer] = target_module
        else:
            target_module.target_builder = builder
            target_module.enabled = config.enabled
            target_module.fail_without_image = config.fail_without_image
            target_module.loss_type = config.loss_type
            _install_input_capture_hook(model, target_module.context)
            if install_criterion:
                install_auxiliary_criterion(model, config)
            setattr(model, "_yolo_superpixels_aux_context", target_module.context)
            return AuxiliaryLossPatchResult(
                target_layer=target_layer,
                target_module=target_module.module.__class__.__name__,
                detect_layer=detect_layer,
                detect_sources=detect_sources,
                target_channels=target_module.target_builder.target_channels,
                already_patched=True,
            )

    builder = target_builder or config.build_target_builder()
    in_channels = _infer_module_out_channels(target_module)
    head = AuxiliarySuperpixelHead(
        in_channels=in_channels,
        out_channels=builder.target_channels,
        hidden_channels=config.head_hidden_channels,
        init=config.head_init,
    )
    context = AuxiliaryLossContext()
    wrapper = AuxiliarySuperpixelWrapper(
        module=target_module,
        head=head,
        target_builder=builder,
        context=context,
        enabled=config.enabled,
        fail_without_image=config.fail_without_image,
        loss_type=config.loss_type,
    )
    device, dtype = _module_device_dtype(target_module)
    if device is not None:
        wrapper = wrapper.to(device=device, dtype=dtype)
    layers[target_layer] = wrapper
    _install_input_capture_hook(model, context)
    setattr(model, "_yolo_superpixels_aux_context", context)
    if install_criterion:
        install_auxiliary_criterion(model, config)
    return AuxiliaryLossPatchResult(
        target_layer=target_layer,
        target_module=target_module.__class__.__name__,
        detect_layer=detect_layer,
        detect_sources=detect_sources,
        target_channels=builder.target_channels,
    )


def install_auxiliary_criterion(model: nn.Module, config: AuxiliaryLossConfig | None = None) -> None:
    """Install or refresh the criterion wrapper after the training model is fully configured."""

    config = config or AuxiliaryLossConfig()
    context = getattr(model, "_yolo_superpixels_aux_context", None)
    if not isinstance(context, AuxiliaryLossContext):
        raise ValueError("patch_auxiliary_loss must run before install_auxiliary_criterion.")
    _install_auxiliary_criterion(model, context, config)


def strip_auxiliary_loss(model: nn.Module, remove_criterion_wrapper: bool = True) -> int:
    """Remove auxiliary-loss wrappers from a model in-place and return the count."""

    removed = 0
    layers = _module_list(model)
    for index, module in enumerate(list(layers)):
        if isinstance(module, AuxiliarySuperpixelWrapper):
            layers[index] = module.module
            removed += 1

    handle = getattr(model, "_yolo_superpixels_aux_capture_handle", None)
    if handle is not None:
        handle.remove()
        setattr(model, "_yolo_superpixels_aux_capture_handle", None)

    if remove_criterion_wrapper and isinstance(getattr(model, "criterion", None), AuxiliaryDetectionCriterion):
        model.criterion = model.criterion.base_criterion
    context = getattr(model, "_yolo_superpixels_aux_context", None)
    if isinstance(context, AuxiliaryLossContext):
        context.clear_runtime_tensors()
    return removed


@dataclass
class BackboneAuxLossStrategy:
    """Strategy object for superpixel-structure supervision after the backbone/neck."""

    config: AuxiliaryLossConfig = field(default_factory=AuxiliaryLossConfig)
    info: StrategyInfo = StrategyInfo(
        name="backbone_auxiliary_loss",
        package="yolo_superpixels.strategies.auxiliary_loss",
        description="Add a training-only auxiliary loss for superpixel structure prediction.",
        status="experimental",
    )

    def patch_model(self, model: nn.Module) -> AuxiliaryLossPatchResult:
        return patch_auxiliary_loss(model, self.config)

    def prepare_training(self) -> None:
        return None
