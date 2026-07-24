"""Training-only auxiliary losses for superpixel structure."""

from .patch import (
    AuxiliaryDetectionCriterion,
    AuxiliaryLossConfig,
    AuxiliaryLossContext,
    AuxiliaryLossPatchResult,
    AuxiliarySuperpixelHead,
    AuxiliarySuperpixelTargetBuilder,
    AuxiliarySuperpixelWrapper,
    BackboneAuxLossStrategy,
    find_auxiliary_loss_target,
    install_auxiliary_criterion,
    patch_auxiliary_loss,
    strip_auxiliary_loss,
)

__all__ = [
    "AuxiliaryDetectionCriterion",
    "AuxiliaryLossConfig",
    "AuxiliaryLossContext",
    "AuxiliaryLossPatchResult",
    "AuxiliarySuperpixelHead",
    "AuxiliarySuperpixelTargetBuilder",
    "AuxiliarySuperpixelWrapper",
    "BackboneAuxLossStrategy",
    "find_auxiliary_loss_target",
    "install_auxiliary_criterion",
    "patch_auxiliary_loss",
    "strip_auxiliary_loss",
]
