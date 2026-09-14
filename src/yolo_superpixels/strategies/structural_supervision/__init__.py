"""SP-11 foreground structural supervision strategy."""

from .patch import (
    ForegroundnessHead,
    StructuralSupervisionConfig,
    StructuralSupervisionContext,
    StructuralSupervisionCriterion,
    StructuralSupervisionPatchResult,
    StructuralSupervisionStrategy,
    StructuralSupervisionWrapper,
    foreground_bce_loss,
    foreground_target_from_boxes,
    install_structural_criterion,
    labels_on_feature_grid,
    patch_structural_supervision,
    region_consistency_loss,
    strip_structural_supervision,
)

__all__ = [
    "ForegroundnessHead", "StructuralSupervisionConfig", "StructuralSupervisionContext",
    "StructuralSupervisionCriterion", "StructuralSupervisionPatchResult", "StructuralSupervisionStrategy",
    "StructuralSupervisionWrapper", "foreground_bce_loss", "foreground_target_from_boxes",
    "install_structural_criterion", "labels_on_feature_grid", "patch_structural_supervision",
    "region_consistency_loss", "strip_structural_supervision",
]
