"""Experiment strategies for injecting superpixel information into YOLO."""

from .base import StrategyInfo, SuperpixelYOLOStrategy
from .auxiliary_loss import BackboneAuxLossStrategy
from .neck_injection import NeckInjectionStrategy

__all__ = [
    "BackboneAuxLossStrategy",
    "NeckInjectionStrategy",
    "StrategyInfo",
    "SuperpixelYOLOStrategy",
]
