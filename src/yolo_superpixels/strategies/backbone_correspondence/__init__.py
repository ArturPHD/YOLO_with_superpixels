"""Geometry-aware superpixel-to-CNN correspondence inside YOLO."""

from .fusion import BackboneCorrespondenceFusion, OverlapEdges, build_overlap_edges
from .patch import (
    BackboneCorrespondenceConfig,
    BackboneCorrespondencePatchResult,
    patch_backbone_correspondence,
)

__all__ = [
    "BackboneCorrespondenceConfig",
    "BackboneCorrespondenceFusion",
    "BackboneCorrespondencePatchResult",
    "OverlapEdges",
    "build_overlap_edges",
    "patch_backbone_correspondence",
]
