"""Patch PyTorch models so their first Conv2d consumes superpixel channels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPERPIXEL_DISABLE_FUSE_ATTR = "_yolo_superpixels_disable_fuse"


def install_ultralytics_fuse_guard() -> bool:
    """Make Ultralytics ``model.fuse()`` preserve ``SuperpixelConv2d``.

    Ultralytics fuses Conv+BN before standalone validation/prediction. Its fuse
    helper returns a plain ``nn.Conv2d``, which would remove our RGB-to-5-channel
    adapter and reintroduce warmup channel mismatches. The guard is attribute
    based, so it only affects models patched by this package.
    """

    try:
        from ultralytics.nn.tasks import BaseModel
    except ImportError:
        return False

    original_fuse = getattr(BaseModel, "fuse", None)
    if original_fuse is None or getattr(original_fuse, "_yolo_superpixels_guard", False):
        return False

    def fuse_guard(self, *args, **kwargs):
        if getattr(self, SUPERPIXEL_DISABLE_FUSE_ATTR, False):
            return self
        return original_fuse(self, *args, **kwargs)

    fuse_guard._yolo_superpixels_guard = True  # type: ignore[attr-defined]
    fuse_guard._yolo_superpixels_original = original_fuse  # type: ignore[attr-defined]
    BaseModel.fuse = fuse_guard
    return True


class SuperpixelConv2d(nn.Conv2d):
    """A Conv2d that accepts RGB tensors and internally appends superpixel channels."""

    def __init__(
        self,
        *args,
        channel_builder: Any,
        base_channels: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.channel_builder = channel_builder
        self.base_channels = base_channels

    @classmethod
    def from_conv2d(
        cls,
        conv: nn.Conv2d,
        channel_builder: Any,
        target_channels: int = 5,
        base_channels: int = 3,
        init_extra: str = "tiny",
    ) -> "SuperpixelConv2d":
        """Create a superpixel-aware copy of ``conv``.

        Existing RGB weights are preserved. Extra-channel weights are initialized
        with tiny random values by default, so the patched model starts very close
        to the original RGB behavior.
        """

        if conv.groups != 1:
            raise ValueError(
                "Cannot patch grouped/depthwise first convolutions automatically. "
                f"Found groups={conv.groups}."
            )
        if target_channels < base_channels:
            raise ValueError(f"target_channels must be >= {base_channels}, got {target_channels}.")
        if conv.in_channels > target_channels:
            raise ValueError(
                f"Cannot reduce first Conv2d from {conv.in_channels} to {target_channels} channels."
            )
        if conv.in_channels < base_channels:
            raise ValueError(
                f"Expected first Conv2d to have at least {base_channels} input channels, "
                f"got {conv.in_channels}."
            )

        patched = cls(
            in_channels=target_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
            channel_builder=channel_builder,
            base_channels=base_channels,
        )

        with torch.no_grad():
            patched.weight.zero_()
            copy_channels = min(conv.in_channels, target_channels)
            patched.weight[:, :copy_channels].copy_(conv.weight[:, :copy_channels])

            extra_channels = target_channels - conv.in_channels
            if extra_channels > 0:
                extra_slice = patched.weight[:, conv.in_channels : target_channels]
                if init_extra == "zero":
                    extra_slice.zero_()
                elif init_extra == "mean":
                    extra_slice.copy_(conv.weight.mean(dim=1, keepdim=True).repeat(1, extra_channels, 1, 1))
                elif init_extra == "tiny":
                    extra_slice.normal_(mean=0.0, std=1e-5)
                elif init_extra == "kaiming":
                    nn.init.kaiming_normal_(extra_slice, mode="fan_out", nonlinearity="relu")
                else:
                    raise ValueError("init_extra must be one of: 'zero', 'tiny', 'mean', 'kaiming'.")

            if conv.bias is not None and patched.bias is not None:
                patched.bias.copy_(conv.bias)

        return patched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}.")
        if x.shape[1] == self.base_channels:
            x = self.channel_builder.append_to_batch(x)
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"SuperpixelConv2d expected {self.base_channels} RGB channels or "
                f"{self.in_channels} prebuilt channels, got {x.shape[1]}."
            )
        if x.dtype != self.weight.dtype:
            x = x.to(dtype=self.weight.dtype)
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


@dataclass
class PatchResult:
    """Information about a first-convolution patch."""

    path: str
    old_in_channels: int
    new_in_channels: int
    already_patched: bool = False


def _named_children_recursive(module: nn.Module, prefix: str = ""):
    for child_name, child in module.named_children():
        path = f"{prefix}.{child_name}" if prefix else child_name
        yield module, child_name, path, child
        yield from _named_children_recursive(child, path)


def _set_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and name.isdigit():
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def _first_conv2d_module(model: nn.Module, allowed_channels: Iterable[int] | None = None):
    allowed = set(allowed_channels or [])
    for parent, child_name, path, child in _named_children_recursive(model):
        if isinstance(child, nn.Conv2d):
            if not allowed or child.in_channels in allowed:
                return parent, child_name, path, child
    return None


def patch_first_conv2d(
    model: nn.Module,
    channel_builder: Any,
    target_channels: int = 5,
    base_channels: int = 3,
    init_extra: str = "tiny",
    allow_repatch: bool = True,
) -> PatchResult:
    """Patch the first Conv2d found in ``model`` in-place.

    The patched model can still be called with normal ``B x 3 x H x W`` tensors.
    The first convolution receives ``B x 5 x H x W`` internally.
    """

    allowed_channels = set(range(base_channels, target_channels + 1))
    found = _first_conv2d_module(model, allowed_channels=allowed_channels)
    if found is None:
        raise ValueError(
            f"No Conv2d with one of {sorted(allowed_channels)} input channels was found."
        )

    parent, child_name, path, conv = found
    if isinstance(conv, SuperpixelConv2d):
        if not allow_repatch:
            raise ValueError(f"Model is already patched at {path}.")
        if conv.in_channels != target_channels:
            patched = SuperpixelConv2d.from_conv2d(
                conv,
                channel_builder=channel_builder,
                target_channels=target_channels,
                base_channels=base_channels,
                init_extra=init_extra,
            )
            _set_child(parent, child_name, patched)
            setattr(model, SUPERPIXEL_DISABLE_FUSE_ATTR, True)
            install_ultralytics_fuse_guard()
            return PatchResult(
                path=path,
                old_in_channels=conv.in_channels,
                new_in_channels=target_channels,
                already_patched=True,
            )
        conv.channel_builder = channel_builder
        setattr(model, SUPERPIXEL_DISABLE_FUSE_ATTR, True)
        install_ultralytics_fuse_guard()
        return PatchResult(
            path=path,
            old_in_channels=conv.in_channels,
            new_in_channels=conv.in_channels,
            already_patched=True,
        )

    patched = SuperpixelConv2d.from_conv2d(
        conv,
        channel_builder=channel_builder,
        target_channels=target_channels,
        base_channels=base_channels,
        init_extra=init_extra,
    )
    _set_child(parent, child_name, patched)
    setattr(model, SUPERPIXEL_DISABLE_FUSE_ATTR, True)
    install_ultralytics_fuse_guard()
    return PatchResult(path=path, old_in_channels=conv.in_channels, new_in_channels=target_channels)


def infer_first_conv_in_channels(weights_or_model) -> int | None:
    """Return input channels of the first Conv2d in a checkpoint/model, if available."""

    if weights_or_model is None:
        return None

    candidate = weights_or_model
    if isinstance(weights_or_model, (str, Path)):
        path = Path(weights_or_model)
        if not path.exists():
            return None
        candidate = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(candidate, dict):
        candidate = (
            candidate.get("ema")
            or candidate.get("model")
            or candidate.get("state_dict")
        )

    if isinstance(candidate, nn.Module):
        found = _first_conv2d_module(candidate)
        return int(found[3].in_channels) if found else None

    if isinstance(candidate, dict):
        for value in candidate.values():
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                return int(value.shape[1])

    return None


install_ultralytics_fuse_guard()
