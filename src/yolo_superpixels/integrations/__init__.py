"""Framework integrations."""

from .ultralytics import SuperpixelDetectionTrainer, SuperpixelYOLO, SuperpixelYOLOAdapter
from .ultralytics_structural import (
    StructuralSupervisionDetectionTrainer,
    StructuralSupervisionYOLOAdapter,
    SuperpixelStructuralYOLO,
)

__all__ = [
    "StructuralSupervisionDetectionTrainer",
    "StructuralSupervisionYOLOAdapter",
    "SuperpixelDetectionTrainer",
    "SuperpixelStructuralYOLO",
    "SuperpixelYOLO",
    "SuperpixelYOLOAdapter",
]
