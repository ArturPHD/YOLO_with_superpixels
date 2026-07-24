"""Input-channel strategy: RGB plus right/down superpixel adjacency channels."""

from .patch import PatchResult, SuperpixelConv2d, infer_first_conv_in_channels, patch_first_conv2d

__all__ = [
    "PatchResult",
    "SuperpixelConv2d",
    "infer_first_conv_in_channels",
    "patch_first_conv2d",
]
