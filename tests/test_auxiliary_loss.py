from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from yolo_superpixels.superpixels.adjacency import AdjacencyChannelBuilder
from yolo_superpixels.strategies.auxiliary_loss import (
    AuxiliaryDetectionCriterion,
    AuxiliaryLossConfig,
    AuxiliarySuperpixelTargetBuilder,
    AuxiliarySuperpixelWrapper,
    patch_auxiliary_loss,
    strip_auxiliary_loss,
)


class FakeTensorAlgorithm:
    def edges_for_batch(self, batch_tensor, channel_order=("right", "down")):
        batch, _, height, width = batch_tensor.shape
        right = torch.ones((batch, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)
        down = torch.zeros((batch, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)
        return torch.stack((right, down), dim=1)


class FakeLabelAlgorithm:
    def labels_for_batch(self, batch_tensor):
        batch = batch_tensor.shape[0]
        labels = torch.tensor(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            device=batch_tensor.device,
        )
        return labels.unsqueeze(0).expand(batch, -1, -1)


class FakeDetect(nn.Module):
    def __init__(self, sources):
        super().__init__()
        self.f = sources

    def forward(self, x):
        return x


class TinyYOLOLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Identity(),
                nn.Conv2d(3, 4, kernel_size=1),
                nn.Conv2d(4, 4, kernel_size=1),
                FakeDetect([1, 2]),
            ]
        )

    def forward(self, x):
        y0 = self.model[0](x)
        y1 = self.model[1](y0)
        y2 = self.model[2](y1)
        return self.model[3](y2)


def make_target_builder():
    return AuxiliarySuperpixelTargetBuilder(
        AdjacencyChannelBuilder(algorithm=FakeTensorAlgorithm(), edge_representation="same"),
        resize="nearest",
    )


def test_auxiliary_target_resizes_labels_before_adjacency():
    builder = AuxiliarySuperpixelTargetBuilder(
        AdjacencyChannelBuilder(algorithm=FakeLabelAlgorithm(), edge_representation="same"),
        target_mode="labels",
        resize="nearest",
    )

    target = builder(torch.rand(1, 3, 4, 4), size=(2, 2), dtype=torch.float32)

    expected_right = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    expected_down = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(target[:, 0], expected_right)
    torch.testing.assert_close(target[:, 1], expected_down)


def test_patch_auxiliary_loss_computes_loss_only_in_train_mode():
    model = TinyYOLOLike()
    result = patch_auxiliary_loss(
        model,
        AuxiliaryLossConfig(algorithm_names=["torch_slic"]),
        target_builder=make_target_builder(),
        install_criterion=False,
    )

    assert result.target_layer == 1
    assert result.detect_layer == 3
    assert result.detect_sources == [1, 2]
    assert isinstance(model.model[1], AuxiliarySuperpixelWrapper)

    model.train()
    output = model(torch.rand(2, 3, 8, 8))
    assert output.shape == (2, 4, 8, 8)
    assert model.model[1].context.loss is not None
    assert torch.isfinite(model.model[1].context.loss)

    model.model[1].context.clear_loss()
    model.eval()
    model(torch.rand(2, 3, 8, 8))
    assert model.model[1].context.loss is None


def test_auxiliary_detection_criterion_adds_scaled_loss():
    class BaseCriterion:
        def __call__(self, preds, batch):
            return torch.tensor(2.0), torch.tensor([1.0, 2.0, 3.0])

    context = type("Context", (), {})()
    context.loss = torch.tensor(0.5)
    context.last_loss = None
    context.clear_loss = lambda: setattr(context, "loss", None)
    criterion = AuxiliaryDetectionCriterion(
        BaseCriterion(),
        context=context,
        loss_weight=0.1,
        enabled=True,
        scale_by_batch=True,
    )

    loss, loss_items = criterion(None, {"img": torch.zeros(4, 3, 8, 8)})

    torch.testing.assert_close(loss, torch.tensor(2.2))
    torch.testing.assert_close(loss_items, torch.tensor([1.0, 2.0, 3.0, 0.05]))
    assert context.loss is None
    assert context.last_loss == 0.5


def test_auxiliary_detection_criterion_is_deepcopy_safe():
    class BaseCriterion:
        some_value = 123

        def __call__(self, preds, batch):
            return torch.tensor(2.0), torch.tensor([1.0, 2.0, 3.0])

    context = type("Context", (), {})()
    context.loss = None
    context.clear_loss = lambda: None
    criterion = AuxiliaryDetectionCriterion(
        BaseCriterion(),
        context=context,
        loss_weight=0.1,
        enabled=True,
        scale_by_batch=True,
    )

    copied = deepcopy(criterion)

    assert copied.some_value == 123


def test_strip_auxiliary_loss_restores_original_module():
    model = TinyYOLOLike()
    patch_auxiliary_loss(
        model,
        AuxiliaryLossConfig(algorithm_names=["torch_slic"]),
        target_builder=make_target_builder(),
        install_criterion=False,
    )

    removed = strip_auxiliary_loss(model)

    assert removed == 1
    assert isinstance(model.model[1], nn.Conv2d)
