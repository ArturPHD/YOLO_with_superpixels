"""Common interfaces for superpixel integration strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StrategyInfo:
    """Human-readable metadata for an experiment strategy."""

    name: str
    package: str
    description: str
    status: str = "experimental"


class SuperpixelYOLOStrategy(Protocol):
    """Protocol implemented by strategy adapters."""

    info: StrategyInfo

    def patch_model(self, model):
        """Patch or wrap a YOLO/PyTorch model in-place and return patch metadata."""

    def prepare_training(self) -> None:
        """Prepare process-global state before training starts."""
