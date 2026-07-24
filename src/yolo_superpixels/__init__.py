"""Superpixel-aware YOLO experiment framework."""

from importlib import import_module

from .superpixels import (
    EDGE_REPRESENTATIONS,
    AdjacencyChannelBuilder,
    CallableSuperpixelAlgorithm,
    EdgeRepresentation,
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
    labels_to_adjacency,
)

__all__ = [
    "AdjacencyChannelBuilder",
    "EDGE_REPRESENTATIONS",
    "EdgeRepresentation",
    "CallableSuperpixelAlgorithm",
    "OpenCVSuperpixelAlgorithm",
    "PatchResult",
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
    "NeckInjectionConfig",
    "NeckInjectionDetectionTrainer",
    "NeckInjectionPatchResult",
    "NeckInjectionStrategy",
    "NeckInjectionYOLOAdapter",
    "AuxiliaryLossConfig",
    "AuxiliaryLossDetectionTrainer",
    "AuxiliaryLossPatchResult",
    "AuxiliaryLossYOLOAdapter",
    "BackboneAuxLossStrategy",
    "SuperpixelConv2d",
    "SuperpixelAuxiliaryYOLO",
    "SuperpixelNeckYOLO",
    "SuperpixelDetectionTrainer",
    "SuperpixelYOLO",
    "SuperpixelYOLOAdapter",
    "infer_first_conv_in_channels",
    "labels_to_adjacency",
    "patch_first_conv2d",
    "patch_auxiliary_loss",
    "patch_neck_injection",
    "strip_auxiliary_loss",
]

_LAZY_EXPORTS = {
    "PatchResult": (".strategies.input_channels", "PatchResult"),
    "SuperpixelConv2d": (".strategies.input_channels", "SuperpixelConv2d"),
    "infer_first_conv_in_channels": (".strategies.input_channels", "infer_first_conv_in_channels"),
    "patch_first_conv2d": (".strategies.input_channels", "patch_first_conv2d"),
    "SuperpixelDetectionTrainer": (".integrations.ultralytics", "SuperpixelDetectionTrainer"),
    "SuperpixelYOLO": (".integrations.ultralytics", "SuperpixelYOLO"),
    "SuperpixelYOLOAdapter": (".integrations.ultralytics", "SuperpixelYOLOAdapter"),
    "NeckInjectionConfig": (".strategies.neck_injection", "NeckInjectionConfig"),
    "NeckInjectionPatchResult": (".strategies.neck_injection", "NeckInjectionPatchResult"),
    "NeckInjectionStrategy": (".strategies.neck_injection", "NeckInjectionStrategy"),
    "patch_neck_injection": (".strategies.neck_injection", "patch_neck_injection"),
    "NeckInjectionDetectionTrainer": (".integrations.ultralytics_neck", "NeckInjectionDetectionTrainer"),
    "NeckInjectionYOLOAdapter": (".integrations.ultralytics_neck", "NeckInjectionYOLOAdapter"),
    "SuperpixelNeckYOLO": (".integrations.ultralytics_neck", "SuperpixelNeckYOLO"),
    "AuxiliaryLossConfig": (".strategies.auxiliary_loss", "AuxiliaryLossConfig"),
    "AuxiliaryLossPatchResult": (".strategies.auxiliary_loss", "AuxiliaryLossPatchResult"),
    "BackboneAuxLossStrategy": (".strategies.auxiliary_loss", "BackboneAuxLossStrategy"),
    "patch_auxiliary_loss": (".strategies.auxiliary_loss", "patch_auxiliary_loss"),
    "strip_auxiliary_loss": (".strategies.auxiliary_loss", "strip_auxiliary_loss"),
    "AuxiliaryLossDetectionTrainer": (".integrations.ultralytics_auxiliary", "AuxiliaryLossDetectionTrainer"),
    "AuxiliaryLossYOLOAdapter": (".integrations.ultralytics_auxiliary", "AuxiliaryLossYOLOAdapter"),
    "SuperpixelAuxiliaryYOLO": (".integrations.ultralytics_auxiliary", "SuperpixelAuxiliaryYOLO"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
