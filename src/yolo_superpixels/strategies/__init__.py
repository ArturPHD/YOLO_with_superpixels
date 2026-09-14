"""Experiment strategies for injecting superpixel information into YOLO."""

from .base import StrategyInfo, SuperpixelYOLOStrategy
from .auxiliary_loss import BackboneAuxLossStrategy
from .neck_injection import NeckInjectionStrategy
from .structural_supervision import StructuralSupervisionStrategy

__all__ = [
    "BackboneAuxLossStrategy",
    "NeckInjectionStrategy",
    "StructuralSupervisionStrategy",
    "StrategyInfo",
    "SuperpixelYOLOStrategy",
]
