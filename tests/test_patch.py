import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from yolo_superpixels import (
    AdjacencyChannelBuilder,
    DistanceChannelBuilder,
    SuperpixelYOLOAdapter,
    SuperpixelConv2d,
    patch_first_conv2d,
)


class ConstantSplitAlgorithm:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        labels = np.zeros(image.shape[:2], dtype=np.int32)
        labels[:, image.shape[1] // 2 :] = 1
        return labels


def test_patch_first_conv_accepts_rgb_input():
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), nn.ReLU())
    builder = AdjacencyChannelBuilder(ConstantSplitAlgorithm(), fail_open=False)

    result = patch_first_conv2d(model, builder)

    assert result.path == "0"
    assert result.old_in_channels == 3
    assert result.new_in_channels == 5
    assert isinstance(model[0], SuperpixelConv2d)

    output = model(torch.rand(2, 3, 8, 8))
    assert output.shape == (2, 4, 8, 8)


def test_patch_first_conv_expands_to_multiple_algorithms():
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), nn.ReLU())
    builder = AdjacencyChannelBuilder(
        [ConstantSplitAlgorithm(), ConstantSplitAlgorithm()],
        fail_open=False,
    )

    result = patch_first_conv2d(model, builder, target_channels=7)

    assert result.old_in_channels == 3
    assert result.new_in_channels == 7
    assert isinstance(model[0], SuperpixelConv2d)
    assert model[0].in_channels == 7
    output = model(torch.rand(2, 3, 8, 8))
    assert output.shape == (2, 4, 8, 8)


def test_patch_first_conv_supports_four_channel_distance_builder():
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), nn.ReLU())
    builder = DistanceChannelBuilder(ConstantSplitAlgorithm(), mode="soft_interior", fail_open=False)

    result = patch_first_conv2d(model, builder, target_channels=4)

    assert result.old_in_channels == 3
    assert result.new_in_channels == 4
    assert isinstance(model[0], SuperpixelConv2d)
    assert model[0].in_channels == 4
    output = model(torch.rand(2, 3, 8, 8))
    assert output.shape == (2, 4, 8, 8)


def test_adapter_infers_target_channels_for_soft_modes():
    interior = SuperpixelYOLOAdapter(
        algorithm=ConstantSplitAlgorithm(),
        channel_mode="soft_interior",
    )
    both = SuperpixelYOLOAdapter(
        algorithm=ConstantSplitAlgorithm(),
        channel_mode="soft_boundary_interior",
    )

    assert interior.target_channels == 4
    assert both.target_channels == 5


def test_superpixel_conv_casts_rgb_input_to_weight_dtype_for_amp_validation():
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), nn.ReLU())
    builder = AdjacencyChannelBuilder(ConstantSplitAlgorithm(), fail_open=False)
    patch_first_conv2d(model, builder)
    model.half()

    output = model(torch.rand(2, 3, 8, 8, dtype=torch.float32))

    assert output.dtype == torch.float16
    assert output.shape == (2, 4, 8, 8)
