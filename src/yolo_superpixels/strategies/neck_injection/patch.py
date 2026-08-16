"""Patch YOLO models with a neck-level superpixel token injector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..base import StrategyInfo
from ..input_channels.patch import SUPERPIXEL_DISABLE_FUSE_ATTR, install_ultralytics_fuse_guard
from .fusion import SuperpixelNeckFusion
from .tokens import RegionFeatureTokenProvider, SPiTLabelTokenProvider, SPiTTokenProvider


@dataclass
class NeckInjectionConfig:
    """Serializable configuration for the SPiT neck-injection strategy."""

    fusion: str = "add"
    target_layer: int | None = None
    enabled: bool = True
    fail_without_image: bool = False
    gate_init: str = "tiny"
    attention_heads: int = 4
    attention_dropout: float = 0.0
    token_provider: str = "region"
    emb_dim: int = 64
    num_bins: int = 5
    heads: int = 4
    depth: int = 1
    keep_k: int = 0
    extractor: str = "interpolate"
    tokenizer: str = "superpixel"
    image_size: int | None = 160
    spit_weights: str | None = None
    freeze_spit: bool = False
    spit_train_mode: str = "joint"
    spit_lr_scale: float = 1.0
    spit_freeze_epochs: int = 0
    drop_delta: bool = False
    bbox_reg: bool = True
    mode: str = "bilinear"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    region_algorithm: str = "torch_slic"
    region_n_segments: int = 300
    region_compactness: float = 1.0
    region_num_iters: int = 7
    region_jitter: float = 0.35
    region_seed: int = 0
    spit_label_lambda_grad: float = 27.8
    spit_label_lambda_col: float = 10.0
    spit_label_maxlvl: int = 4
    validate_target: bool = True
    validation_imgsz: int = 64

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, values: dict[str, Any]) -> "NeckInjectionConfig":
        return cls(**values)

    def build_token_provider(self) -> nn.Module:
        if self.token_provider == "region":
            return RegionFeatureTokenProvider(
                algorithm=self.region_algorithm,
                emb_dim=self.emb_dim,
                image_size=self.image_size,
                n_segments=self.region_n_segments,
                compactness=self.region_compactness,
                num_iters=self.region_num_iters,
                jitter=self.region_jitter,
                seed=self.region_seed,
            )
        if self.token_provider == "spit_labels":
            return SPiTLabelTokenProvider(
                emb_dim=self.emb_dim,
                image_size=self.image_size,
                lambda_grad=self.spit_label_lambda_grad,
                lambda_col=self.spit_label_lambda_col,
                maxlvl=self.spit_label_maxlvl,
                drop_delta=self.drop_delta,
                bbox_reg=self.bbox_reg,
            )
        if self.token_provider != "spit":
            raise ValueError("token_provider must be 'region', 'spit_labels', or 'spit'.")
        valid_train_modes = {"joint", "frozen", "delayed", "scratch"}
        if self.spit_train_mode not in valid_train_modes:
            raise ValueError(
                f"spit_train_mode must be one of {sorted(valid_train_modes)}; "
                f"got {self.spit_train_mode!r}."
            )
        if self.spit_lr_scale <= 0:
            raise ValueError("spit_lr_scale must be positive.")
        if self.spit_freeze_epochs < 0:
            raise ValueError("spit_freeze_epochs must be non-negative.")
        if self.spit_train_mode == "delayed" and self.spit_freeze_epochs <= 0:
            raise ValueError("delayed SPiT training requires spit_freeze_epochs > 0.")
        if self.spit_train_mode in {"frozen", "delayed"} and not self.spit_weights:
            raise ValueError(f"SPiT train mode {self.spit_train_mode!r} requires pretrained weights.")
        if self.spit_train_mode == "scratch" and self.spit_weights:
            raise ValueError("scratch SPiT training must not receive pretrained spit_weights.")
        if not self.spit_weights and not self.extra_kwargs.get("allow_random_spit", False):
            raise ValueError(
                "Full token_provider='spit' requires spit_weights by default. "
                "Pass extra_kwargs={'allow_random_spit': True} only for a deliberate random-SPiT ablation."
            )
        return SPiTTokenProvider(
            emb_dim=self.emb_dim,
            num_bins=self.num_bins,
            heads=self.heads,
            depth=self.depth,
            keep_k=self.keep_k,
            extractor=self.extractor,
            tokenizer=self.tokenizer,
            image_size=self.image_size,
            spit_weights=self.spit_weights,
            freeze=self.freeze_spit,
            drop_delta=self.drop_delta,
            bbox_reg=self.bbox_reg,
            mode=self.mode,
            extra_kwargs=self.extra_kwargs,
        )


@dataclass
class NeckInjectionPatchResult:
    """Information about an installed neck-injection patch."""

    target_layer: int
    target_module: str
    detect_layer: int | None
    detect_sources: list[int]
    target_stride: int | None = None
    target_shape: tuple[int, int] | None = None
    already_patched: bool = False


class NeckInjectionWrapper(nn.Module):
    """Wrap one YOLO neck module and fuse SPiT token features into its output."""

    def __init__(
        self,
        module: nn.Module,
        fusion: SuperpixelNeckFusion,
        enabled: bool = True,
        fail_without_image: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.fusion = fusion
        self.enabled = enabled
        self.fail_without_image = fail_without_image
        self._current_image: torch.Tensor | None = None
        self.f = getattr(module, "f", -1)
        self.i = getattr(module, "i", None)
        self.type = getattr(module, "type", module.__class__.__name__)
        self.np = getattr(module, "np", sum(parameter.numel() for parameter in module.parameters()))

    def set_current_image(self, image: torch.Tensor) -> None:
        self._current_image = image[:, :3]

    def forward(self, x):
        output = self.module(x)
        if not self.enabled:
            return output
        if self._current_image is None:
            if self.fail_without_image:
                raise RuntimeError(
                    "NeckInjectionWrapper did not receive the original RGB image. "
                    "Make sure patch_neck_injection installed its input capture hook."
                )
            return output

        image = self._current_image.to(device=output.device, non_blocking=True)
        self._current_image = None
        return self.fusion(output, image)


class InputCaptureHook:
    """Pickle-safe forward pre-hook that sends the original RGB to the wrapper."""

    def __init__(self, wrapper: NeckInjectionWrapper) -> None:
        self.wrapper = wrapper

    def __call__(self, _module, args) -> None:
        if args and isinstance(args[0], torch.Tensor):
            self.wrapper.set_current_image(args[0])


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

    normalized = []
    for value in values:
        normalized.append(detect_index + value if value < 0 else value)
    return normalized


def find_neck_injection_target(model: nn.Module, target_layer: int | None = None) -> tuple[int, int | None, list[int]]:
    """Find the default P3 neck feature feeding the Detect module."""

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
            if not sources:
                break
            return sources[0], detect_index, sources

    raise ValueError(
        "Could not find a Detect module with source indices. "
        "Pass target_layer explicitly."
    )


def _install_input_capture_hook(model: nn.Module, wrapper: NeckInjectionWrapper) -> None:
    old_handle = getattr(model, "_yolo_superpixels_neck_capture_handle", None)
    if old_handle is not None:
        old_handle.remove()

    first_layer = _module_list(model)[0]

    handle = first_layer.register_forward_pre_hook(InputCaptureHook(wrapper))
    setattr(model, "_yolo_superpixels_neck_capture_handle", handle)


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
        "Pass target_layer explicitly or use a module with an identifiable output Conv2d."
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


def _token_dim(token_provider: nn.Module) -> int:
    for attr_name in ("emb_dim", "token_dim"):
        value = getattr(token_provider, attr_name, None)
        if isinstance(value, int):
            return int(value)
        if callable(value):
            resolved = value()
            if isinstance(resolved, int):
                return int(resolved)
    raise ValueError(
        f"Could not infer token dimension from {token_provider.__class__.__name__}. "
        "Expose an integer .emb_dim or .token_dim attribute."
    )


def validate_neck_injection_target(
    model: nn.Module,
    target_layer: int,
    detect_sources: list[int],
    imgsz: int = 64,
) -> tuple[int | None, tuple[int, int] | None]:
    """Validate that the selected layer is the high-resolution Detect input."""

    if not detect_sources:
        return None, None

    layers = _module_list(model)
    capture_layers = sorted(set(detect_sources + [target_layer]))
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook(index: int):
        def _capture(_module, _args, output):
            if isinstance(output, torch.Tensor):
                captured[index] = output.detach()

        return _capture

    for index in capture_layers:
        handles.append(layers[index].register_forward_hook(hook(index)))

    was_training = model.training
    device, dtype = _module_device_dtype(model)
    device = device or torch.device("cpu")
    if dtype is None or not dtype.is_floating_point:
        dtype = torch.float32
    dummy = torch.zeros((1, 3, int(imgsz), int(imgsz)), device=device, dtype=dtype)

    try:
        model.eval()
        with torch.no_grad():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    if target_layer not in captured:
        raise ValueError(f"Target layer {target_layer} did not produce a tensor during validation forward.")

    target_shape = tuple(int(value) for value in captured[target_layer].shape[-2:])
    source_shapes = {
        source: tuple(int(value) for value in captured[source].shape[-2:])
        for source in detect_sources
        if source in captured
    }
    if source_shapes:
        max_area = max(height * width for height, width in source_shapes.values())
        target_area = target_shape[0] * target_shape[1]
        if target_area != max_area:
            raise ValueError(
                f"Selected target_layer={target_layer} has feature size {target_shape}, "
                f"but Detect sources have sizes {source_shapes}; expected the highest-resolution Detect source."
            )

    target_stride = int(imgsz) // target_shape[0] if target_shape[0] and int(imgsz) % target_shape[0] == 0 else None
    model_stride = getattr(model, "stride", None)
    if model_stride is not None and len(model_stride) > 0 and target_stride is not None:
        expected_stride = int(model_stride[0])
        if target_stride != expected_stride:
            raise ValueError(
                f"Selected target_layer={target_layer} has stride {target_stride}, "
                f"but the first Detect stride is {expected_stride}."
            )
    return target_stride, target_shape


def patch_neck_injection(
    model: nn.Module,
    config: NeckInjectionConfig | None = None,
    token_provider: SPiTTokenProvider | None = None,
) -> NeckInjectionPatchResult:
    """Patch an Ultralytics YOLO model in-place with a neck injector."""

    config = config or NeckInjectionConfig()
    layers = _module_list(model)
    target_layer, detect_layer, detect_sources = find_neck_injection_target(model, config.target_layer)
    target_module = layers[target_layer]
    target_stride, target_shape = (
        validate_neck_injection_target(model, target_layer, detect_sources, config.validation_imgsz)
        if config.validate_target
        else (None, None)
    )

    if isinstance(target_module, NeckInjectionWrapper):
        target_module.enabled = config.enabled
        target_module.fail_without_image = config.fail_without_image
        _install_input_capture_hook(model, target_module)
        return NeckInjectionPatchResult(
            target_layer=target_layer,
            target_module=target_module.module.__class__.__name__,
            detect_layer=detect_layer,
            detect_sources=detect_sources,
            target_stride=target_stride,
            target_shape=target_shape,
            already_patched=True,
        )

    token_provider = token_provider or config.build_token_provider()
    fusion = SuperpixelNeckFusion(
        token_provider=token_provider,
        mode=config.fusion,
        gate_init=config.gate_init,
        attention_heads=config.attention_heads,
        attention_dropout=config.attention_dropout,
    )
    yolo_channels = _infer_module_out_channels(target_module)
    token_dim = _token_dim(token_provider)
    device, dtype = _module_device_dtype(target_module)
    fusion.initialize(yolo_channels=yolo_channels, token_dim=token_dim, device=device, dtype=dtype)
    wrapper = NeckInjectionWrapper(
        module=target_module,
        fusion=fusion,
        enabled=config.enabled,
        fail_without_image=config.fail_without_image,
    )
    if device is not None:
        wrapper = wrapper.to(device=device, dtype=dtype)
    layers[target_layer] = wrapper
    _install_input_capture_hook(model, wrapper)
    setattr(model, SUPERPIXEL_DISABLE_FUSE_ATTR, True)
    install_ultralytics_fuse_guard()
    return NeckInjectionPatchResult(
        target_layer=target_layer,
        target_module=target_module.__class__.__name__,
        detect_layer=detect_layer,
        detect_sources=detect_sources,
        target_stride=target_stride,
        target_shape=target_shape,
    )


@dataclass
class NeckInjectionStrategy:
    """Strategy object for experiments that inject SPiT token features into the neck."""

    config: NeckInjectionConfig = field(default_factory=NeckInjectionConfig)
    info: StrategyInfo = StrategyInfo(
        name="neck_injection",
        package="yolo_superpixels.strategies.neck_injection",
        description="Inject SPiT/superpixel token features into the YOLO neck.",
        status="experimental",
    )

    def patch_model(self, model: nn.Module) -> NeckInjectionPatchResult:
        return patch_neck_injection(model, self.config)

    def prepare_training(self) -> None:
        return None
