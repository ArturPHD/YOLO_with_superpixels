"""Inject SPiT/superpixel token features into a YOLO neck."""

from .fusion import SuperpixelNeckFusion
from .patch import (
    NeckInjectionConfig,
    NeckInjectionPatchResult,
    NeckInjectionStrategy,
    NeckInjectionWrapper,
    find_neck_injection_target,
    patch_neck_injection,
)
from .tokens import (
    QStarTokenProvider,
    RegionFeatureTokenProvider,
    SPiTLabelTokenProvider,
    SPiTTokenProvider,
    TokenBundle,
    tokens_to_dense_map,
)

__all__ = [
    "NeckInjectionConfig",
    "NeckInjectionPatchResult",
    "NeckInjectionStrategy",
    "NeckInjectionWrapper",
    "RegionFeatureTokenProvider",
    "QStarTokenProvider",
    "SPiTLabelTokenProvider",
    "SPiTTokenProvider",
    "SuperpixelNeckFusion",
    "TokenBundle",
    "find_neck_injection_target",
    "patch_neck_injection",
    "tokens_to_dense_map",
]
