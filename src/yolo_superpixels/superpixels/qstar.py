"""GPU-native adapter around the vendored Q* hierarchical superpixel pipeline.

Q* is kept in ``third_party/qstar_superpixels`` under its upstream MIT licence.
This module only handles import discovery, image-size compatibility and the
small public surface needed by YOLO strategies.  No NumPy/CuPy conversion is
performed on the tensor path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_local_qstar_on_path() -> Path:
    """Make the repository-vendored Q* package importable."""

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "qstar_superpixels"
        if (candidate / "qstar_superpixels" / "__init__.py").is_file():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return candidate
    raise ImportError(
        "Vendored Q* source was not found. Expected "
        "third_party/qstar_superpixels/qstar_superpixels in this repository."
    )


@dataclass(frozen=True)
class QStarConfig:
    """Stable subset of Q* parameters exposed by the YOLO integration."""

    target_k: int = 512
    max_level: int = 5
    min_child: int = 4
    relax_iters: int = 5
    compactness: float = 0.5
    margin: float = 0.1
    min_area: int = 9
    min_diam: int = 1
    merge_rounds: tuple[int, ...] = (5, 5)
    merge_top_k: tuple[int, ...] = (50, 75)
    balance_power: tuple[float, ...] = (0.0, 0.6)
    cmp_mix: float = 0.25

    def pipeline_config(self):
        _ensure_local_qstar_on_path()
        from qstar_superpixels import PipelineConfig

        return PipelineConfig(
            target_k=self.target_k,
            max_level=self.max_level,
            min_child=self.min_child,
            relax_iters=self.relax_iters,
            compactness=self.compactness,
            margin=self.margin,
            min_area=self.min_area,
            min_diam=self.min_diam,
            merge_rounds=self.merge_rounds,
            merge_top_k=self.merge_top_k,
            balance_power=self.balance_power,
            cmp_mix=self.cmp_mix,
        )


@dataclass
class QStarResult:
    """One selected Q* hierarchy level, retaining global region indexing."""

    labels: torch.Tensor
    features: torch.Tensor
    batch_indices: torch.Tensor


class QStarRunner(nn.Module):
    """Execute Q* on the same device as an RGB tensor.

    Q*'s upstream quadtree is most reliable on a square canvas whose side is
    divisible by ``2 ** max_level``. The adapter replicate-pads only the
    lower/right edge to that canvas, runs the untouched implementation, then
    crops labels back to the input geometry. Standard YOLO ``imgsz=640`` and
    the default Q* tokenizer resolution of ``160`` are already square, so this
    is a no-op for the normal training path.
    """

    def __init__(
        self,
        config: QStarConfig | None = None,
        *,
        level: int = 0,
        learned: bool = False,
    ) -> None:
        super().__init__()
        if level < 0:
            raise ValueError("qstar level must be non-negative.")
        self.config = config or QStarConfig()
        self.level = int(level)
        self.learned = bool(learned)
        _ensure_local_qstar_on_path()
        from qstar_superpixels import SuperpixelPipeline

        self.pipeline = SuperpixelPipeline(self.config.pipeline_config(), learned=self.learned)
        self._frozen = False

    @property
    def feature_dim(self) -> int:
        return 6 if self.learned else 3

    @property
    def frozen(self) -> bool:
        return self._frozen

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)
        for parameter in self.pipeline.parameters():
            parameter.requires_grad_(not self._frozen)
        if self._frozen:
            self.pipeline.eval()
        else:
            self.pipeline.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.pipeline.eval()
        return self

    def _prepare_input(self, rgb: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if rgb.ndim != 4 or rgb.shape[1] < 3:
            raise ValueError(f"Expected Bx3xHxW RGB input, got shape {tuple(rgb.shape)}.")
        # Ultralytics' final-validation warmup may use torch.empty(), whose
        # uninitialised memory can contain NaN/Inf. Real dataloader images are
        # finite, but the auxiliary Q* branch must not make this framework
        # lifecycle step fail.
        image = torch.nan_to_num(
            rgb[:, :3].detach().float(), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        height, width = image.shape[-2:]
        divisor = 2**self.config.max_level
        minimum_side = self.config.min_child * divisor
        side = max(height, width, minimum_side)
        working_side = ((side + divisor - 1) // divisor) * divisor
        pad_h = working_side - height
        pad_w = working_side - width
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        if self.learned:
            mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            image = (image - mean) / std
        return image, (height, width)

    def forward(self, rgb: torch.Tensor) -> QStarResult:
        image, original_size = self._prepare_input(rgb)
        # Q*'s hierarchy is numerically more stable in fp32 than AMP fp16.
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            result = self.pipeline(image)
        all_segs = result.all_segs()
        if self.level >= len(all_segs):
            raise ValueError(
                f"qstar level {self.level} is unavailable; this configuration returns "
                f"levels 0..{len(all_segs) - 1}."
            )
        height, width = original_size
        labels = all_segs[self.level][:, :height, :width].long()
        features = result.features[self.level]
        batch_indices = result.batch_indices[self.level].long()
        if labels.device != rgb.device or features.device != rgb.device:
            raise RuntimeError("Q* moved tensors off the RGB input device.")
        return QStarResult(labels=labels, features=features, batch_indices=batch_indices)
