"""Reusable superpixel algorithms and channel builders."""

from .adjacency import (
    EDGE_REPRESENTATIONS,
    AdjacencyChannelBuilder,
    EdgeRepresentation,
    algorithm_output_to_edges,
    labels_to_adjacency,
)
from .algorithms import (
    CallableSuperpixelAlgorithm,
    OpenCVSuperpixelAlgorithm,
    SLICSuperpixelAlgorithm,
    SPiTDenseAdjacencyAlgorithm,
    SPiTDenseEdgeAlgorithm,
    SPiTPseudoVoronoiAlgorithm,
    TensorSuperpixelAlgorithm,
    TorchColorAffinityAlgorithm,
    TorchGradientAffinityAlgorithm,
    TorchGridSuperpixelAlgorithm,
    TorchSLICSuperpixelAlgorithm,
    TorchVoronoiSuperpixelAlgorithm,
)

__all__ = [
    "AdjacencyChannelBuilder",
    "EDGE_REPRESENTATIONS",
    "EdgeRepresentation",
    "CallableSuperpixelAlgorithm",
    "OpenCVSuperpixelAlgorithm",
    "SLICSuperpixelAlgorithm",
    "SPiTDenseAdjacencyAlgorithm",
    "SPiTDenseEdgeAlgorithm",
    "SPiTPseudoVoronoiAlgorithm",
    "TensorSuperpixelAlgorithm",
    "TorchColorAffinityAlgorithm",
    "TorchGradientAffinityAlgorithm",
    "TorchGridSuperpixelAlgorithm",
    "TorchSLICSuperpixelAlgorithm",
    "TorchVoronoiSuperpixelAlgorithm",
    "algorithm_output_to_edges",
    "labels_to_adjacency",
]
