"""Install local superpixel correspondence at a YOLO backbone P3 branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from ..input_channels.patch import SUPERPIXEL_DISABLE_FUSE_ATTR, install_ultralytics_fuse_guard
from ..neck_injection.patch import NeckInjectionConfig
from .fusion import BackboneCorrespondenceFusion

_PATCH_ATTR = "_yolo_superpixels_backbone_correspondence"
_CAPTURE_ATTR = "_yolo_superpixels_backbone_capture_handle"


@dataclass
class BackboneCorrespondenceConfig(NeckInjectionConfig):
    """Configuration for sparse P3-cell to superpixel-token correspondence."""

    placement: str = "native_backbone"
    fusion: str = "overlap_attention"
    p3_layer: int | None = None
    overlap_logit_bias: float = 1.0
    validate_target: bool = True
    validation_imgsz: int = 64

    def __post_init__(self) -> None:
        if self.placement not in {"native_backbone", "isolated_p3"}:
            raise ValueError("placement must be 'native_backbone' or 'isolated_p3'.")
        if self.fusion not in {"overlap_pool", "overlap_attention"}:
            raise ValueError("fusion must be 'overlap_pool' or 'overlap_attention'.")

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, values: dict[str, Any]) -> "BackboneCorrespondenceConfig":
        return cls(**values)


@dataclass
class BackboneCorrespondencePatchResult:
    """Resolved graph locations for one installed correspondence intervention."""

    placement: str
    p3_layer: int
    p3_consumer_layer: int
    detect_layer: int
    detect_sources: list[int]
    p3_shape: tuple[int, int] | None
    p3_stride: int | None
    already_patched: bool = False


@dataclass
class _PatchState:
    placement: str
    p3_layer: int
    p3_consumer_layer: int
    detect_layer: int
    detect_sources: list[int]
    p3_shape: tuple[int, int] | None
    p3_stride: int | None
    wrapper: "_CorrespondenceWrapper"


def _layers(model: nn.Module) -> nn.ModuleList | nn.Sequential:
    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)):
        raise ValueError("Expected an Ultralytics-style model with model.model layers.")
    return layers


def _sources(module: nn.Module, index: int) -> list[int]:
    source = getattr(module, "f", -1)
    values = [source] if isinstance(source, int) else list(source)
    return [index + int(value) if int(value) < 0 else int(value) for value in values]


def _find_detect(layers: nn.ModuleList | nn.Sequential) -> tuple[int, list[int]]:
    for index in range(len(layers) - 1, -1, -1):
        module = layers[index]
        if module.__class__.__name__.lower().endswith("detect") and hasattr(module, "f"):
            sources = _sources(module, index)
            if sources:
                return index, sources
    raise ValueError("Could not find an Ultralytics Detect module with feature sources.")


def find_backbone_p3_target(model: nn.Module, p3_layer: int | None = None) -> tuple[int, int, int, list[int]]:
    """Resolve the original backbone P3 tensor and the neck concat that consumes it.

    In YOLO12n the result is ``(4, 13, 21, [14, 17, 20])``; in YOLO26n it is
    ``(4, 15, 23, [16, 19, 22])``. The resolver follows the first Detect/P3
    branch backwards until it reaches the first multi-input fusion node.
    """

    layers = _layers(model)
    detect_layer, detect_sources = _find_detect(layers)
    if p3_layer is not None:
        candidate = int(p3_layer)
        if candidate < 0:
            candidate += len(layers)
        if not 0 <= candidate < len(layers):
            raise ValueError(f"p3_layer={p3_layer} is outside the model graph.")
        consumers = [index for index, module in enumerate(layers) if candidate in _sources(module, index)]
        if not consumers:
            raise ValueError(f"No graph node consumes explicit p3_layer={candidate}.")
        return candidate, min(consumers), detect_layer, detect_sources

    current = detect_sources[0]
    visited: set[int] = set()
    while current not in visited:
        visited.add(current)
        sources = _sources(layers[current], current)
        if len(sources) >= 2:
            older = [source for source in sources if source < current]
            if not older:
                break
            return min(older), current, detect_layer, detect_sources
        if len(sources) != 1 or sources[0] < 0:
            break
        current = sources[0]
    raise ValueError(
        "Could not trace the highest-resolution Detect branch back to a backbone skip tensor. "
        "Pass p3_layer explicitly for this architecture."
    )


def _module_device_dtype(module: nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    for parameter in module.parameters(recurse=True):
        return parameter.device, parameter.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.device, buffer.dtype
    return None, None


def _infer_channels(module: nn.Module) -> int:
    for candidate in (module, getattr(module, "cv3", None), getattr(module, "cv2", None), getattr(module, "conv", None)):
        conv = getattr(candidate, "conv", candidate)
        if isinstance(conv, nn.Conv2d):
            return int(conv.out_channels)
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return int(child.out_channels)
    raise ValueError(f"Could not infer output channels for {module.__class__.__name__}.")


def _token_dim(provider: nn.Module) -> int:
    value = getattr(provider, "emb_dim", None)
    value = value() if callable(value) else value
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{provider.__class__.__name__} must expose a positive emb_dim.")
    return int(value)


class _InputCapture:
    def __init__(self, wrapper: "_CorrespondenceWrapper") -> None:
        self.wrapper = wrapper

    def __call__(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            self.wrapper.set_current_image(args[0])


class _CorrespondenceWrapper(nn.Module):
    def __init__(self, fusion: BackboneCorrespondenceFusion, enabled: bool, fail_without_image: bool) -> None:
        super().__init__()
        self.fusion = fusion
        self.enabled = enabled
        self.fail_without_image = fail_without_image
        self._current_image: torch.Tensor | None = None

    def set_current_image(self, image: torch.Tensor) -> None:
        self._current_image = image[:, :3]

    def _fuse(self, feature: torch.Tensor) -> torch.Tensor:
        image, self._current_image = self._current_image, None
        if not self.enabled:
            return feature
        if image is None:
            if self.fail_without_image:
                raise RuntimeError("Backbone correspondence did not capture the augmented RGB input.")
            return feature
        return self.fusion(feature, image.to(device=feature.device, non_blocking=True))


class BackboneP3Wrapper(_CorrespondenceWrapper):
    """Native branch: P3 is fused before it feeds both deeper backbone and neck."""

    def __init__(self, module: nn.Module, fusion: BackboneCorrespondenceFusion, enabled: bool, fail_without_image: bool) -> None:
        super().__init__(fusion, enabled, fail_without_image)
        self.module = module
        self.f = getattr(module, "f", -1)
        self.i = getattr(module, "i", None)
        self.type = getattr(module, "type", module.__class__.__name__)
        self.np = getattr(module, "np", sum(parameter.numel() for parameter in module.parameters()))

    def forward(self, x):
        return self._fuse(self.module(x))


class IsolatedP3ConsumerWrapper(_CorrespondenceWrapper):
    """Route-specific branch: replace P3 only in its one neck consumer input list."""

    def __init__(
        self,
        module: nn.Module,
        p3_position: int,
        fusion: BackboneCorrespondenceFusion,
        enabled: bool,
        fail_without_image: bool,
    ) -> None:
        super().__init__(fusion, enabled, fail_without_image)
        self.module = module
        self.p3_position = int(p3_position)
        self.f = getattr(module, "f", -1)
        self.i = getattr(module, "i", None)
        self.type = getattr(module, "type", module.__class__.__name__)
        self.np = getattr(module, "np", sum(parameter.numel() for parameter in module.parameters()))

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError("The isolated P3 consumer must receive a multi-input feature list.")
        if not 0 <= self.p3_position < len(x) or not isinstance(x[self.p3_position], torch.Tensor):
            raise RuntimeError("The resolved P3 source is absent from the isolated neck consumer inputs.")
        inputs = list(x)
        inputs[self.p3_position] = self._fuse(inputs[self.p3_position])
        return self.module(inputs)


def _install_capture(model: nn.Module, wrapper: _CorrespondenceWrapper) -> None:
    old = getattr(model, _CAPTURE_ATTR, None)
    if old is not None:
        old.remove()
    handle = _layers(model)[0].register_forward_pre_hook(_InputCapture(wrapper))
    setattr(model, _CAPTURE_ATTR, handle)


def _validate_target(model: nn.Module, p3_layer: int, imgsz: int) -> tuple[int | None, tuple[int, int] | None]:
    layers = _layers(model)
    captured: dict[int, torch.Tensor] = {}
    handle = layers[p3_layer].register_forward_hook(lambda _m, _a, output: captured.setdefault(p3_layer, output.detach()))
    was_training = model.training
    device, dtype = _module_device_dtype(layers[p3_layer])
    device = device or torch.device("cpu")
    dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float32
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, 3, int(imgsz), int(imgsz)), device=device, dtype=dtype))
    finally:
        handle.remove()
        model.train(was_training)
    output = captured.get(p3_layer)
    if output is None or output.ndim != 4:
        raise ValueError("Resolved backbone P3 layer did not emit a BCHW tensor.")
    shape = tuple(int(value) for value in output.shape[-2:])
    stride = int(imgsz) // shape[0] if imgsz % shape[0] == 0 else None
    if stride is not None and stride != int(getattr(model, "stride", [stride])[0]):
        raise ValueError(f"Resolved P3 has stride {stride}; expected highest-resolution Detect stride.")
    return stride, shape


def patch_backbone_correspondence(model: nn.Module, config: BackboneCorrespondenceConfig | None = None) -> BackboneCorrespondencePatchResult:
    """Install native-backbone or isolated-P3 correspondence before optimizer creation."""

    config = config or BackboneCorrespondenceConfig()
    layers = _layers(model)
    previous = getattr(model, _PATCH_ATTR, None)
    if previous is not None:
        if previous.placement != config.placement:
            raise RuntimeError("This model is already patched with a different backbone correspondence placement.")
        previous.wrapper.enabled = config.enabled
        previous.wrapper.fail_without_image = config.fail_without_image
        _install_capture(model, previous.wrapper)
        return BackboneCorrespondencePatchResult(
            placement=previous.placement,
            p3_layer=previous.p3_layer,
            p3_consumer_layer=previous.p3_consumer_layer,
            detect_layer=previous.detect_layer,
            detect_sources=previous.detect_sources,
            p3_shape=previous.p3_shape,
            p3_stride=previous.p3_stride,
            already_patched=True,
        )

    p3_layer, consumer_layer, detect_layer, detect_sources = find_backbone_p3_target(model, config.p3_layer)
    p3_stride, p3_shape = _validate_target(model, p3_layer, config.validation_imgsz) if config.validate_target else (None, None)
    provider = config.build_token_provider()
    target_module = layers[p3_layer]
    channels = _infer_channels(target_module)
    device, dtype = _module_device_dtype(target_module)
    fusion = BackboneCorrespondenceFusion(
        token_provider=provider,
        mode=config.fusion,
        gate_init=config.gate_init,
        attention_heads=config.attention_heads,
        overlap_logit_bias=config.overlap_logit_bias,
    )
    fusion.initialize(channels, _token_dim(provider), device=device, dtype=dtype)

    if config.placement == "native_backbone":
        wrapper: _CorrespondenceWrapper = BackboneP3Wrapper(target_module, fusion, config.enabled, config.fail_without_image)
        layers[p3_layer] = wrapper
    else:
        consumer = layers[consumer_layer]
        sources = _sources(consumer, consumer_layer)
        if p3_layer not in sources:
            raise RuntimeError("Resolved isolated consumer does not reference the selected P3 layer.")
        wrapper = IsolatedP3ConsumerWrapper(
            consumer,
            sources.index(p3_layer),
            fusion,
            config.enabled,
            config.fail_without_image,
        )
        layers[consumer_layer] = wrapper
    if device is not None:
        wrapper.to(device=device, dtype=dtype)
    _install_capture(model, wrapper)
    setattr(model, SUPERPIXEL_DISABLE_FUSE_ATTR, True)
    install_ultralytics_fuse_guard()
    result = BackboneCorrespondencePatchResult(
        placement=config.placement,
        p3_layer=p3_layer,
        p3_consumer_layer=consumer_layer,
        detect_layer=detect_layer,
        detect_sources=detect_sources,
        p3_shape=p3_shape,
        p3_stride=p3_stride,
    )
    setattr(
        model,
        _PATCH_ATTR,
        _PatchState(
            placement=result.placement,
            p3_layer=result.p3_layer,
            p3_consumer_layer=result.p3_consumer_layer,
            detect_layer=result.detect_layer,
            detect_sources=result.detect_sources,
            p3_shape=result.p3_shape,
            p3_stride=result.p3_stride,
            wrapper=wrapper,
        ),
    )
    return result
