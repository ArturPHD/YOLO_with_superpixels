"""Framework integrations."""

from .ultralytics import SuperpixelDetectionTrainer, SuperpixelYOLO, SuperpixelYOLOAdapter
from .ultralytics_structural import (
    StructuralSupervisionDetectionTrainer,
    StructuralSupervisionYOLOAdapter,
    SuperpixelStructuralYOLO,
)
from .ultralytics_backbone import (
    BackboneCorrespondenceDetectionTrainer,
    BackboneCorrespondenceYOLOAdapter,
    SuperpixelBackboneYOLO,
)

__all__ = [
    "StructuralSupervisionDetectionTrainer",
    "StructuralSupervisionYOLOAdapter",
    "BackboneCorrespondenceDetectionTrainer",
    "BackboneCorrespondenceYOLOAdapter",
    "SuperpixelDetectionTrainer",
    "SuperpixelStructuralYOLO",
    "SuperpixelBackboneYOLO",
    "SuperpixelYOLO",
    "SuperpixelYOLOAdapter",
]
