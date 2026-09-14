from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from yolo_superpixels.strategies.structural_supervision import (
    StructuralSupervisionConfig,
    StructuralSupervisionContext,
    StructuralSupervisionCriterion,
    StructuralSupervisionWrapper,
    foreground_bce_loss,
    foreground_target_from_boxes,
    labels_on_feature_grid,
    patch_structural_supervision,
    region_consistency_loss,
    strip_structural_supervision,
)
from yolo_superpixels.superpixels.algorithms import TorchGridSuperpixelAlgorithm, TorchSLICSuperpixelAlgorithm


class FakeDetect(nn.Module):
    def __init__(self, sources):
        super().__init__()
        self.f = sources

    def forward(self, value):
        return value


class TinyYOLOLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [nn.Identity(), nn.Conv2d(3, 8, 1), nn.Conv2d(8, 8, 1), FakeDetect([1, 2])]
        )

    def init_criterion(self):
        return BaseCriterion()

    def forward(self, image):
        value = image
        for module in self.model:
            value = module(value)
        return value


class BaseCriterion:
    def __call__(self, _predictions, _batch):
        return torch.tensor(2.0), torch.tensor([1.0, 2.0, 3.0])


class TrackingLabels:
    def __init__(self, labels: torch.Tensor | None = None):
        self.calls = 0
        self.labels = labels

    def labels_for_batch(self, image):
        self.calls += 1
        if self.labels is not None:
            return self.labels.to(device=image.device).expand(image.shape[0], -1, -1)
        height, width = image.shape[-2:]
        return torch.zeros((image.shape[0], height, width), dtype=torch.long, device=image.device)


def make_batch(boxes=None, batch_idx=None, batch_size=1, size=8):
    if boxes is None:
        boxes = torch.empty((0, 4))
    if batch_idx is None:
        batch_idx = torch.empty((0,), dtype=torch.long)
    return {"img": torch.rand(batch_size, 3, size, size), "bboxes": boxes, "batch_idx": batch_idx}


def test_box_to_foreground_target_zero_boxes():
    target = foreground_target_from_boxes(make_batch(), size=(4, 4))
    assert target.shape == (1, 1, 4, 4)
    assert target.sum() == 0


def test_box_to_foreground_target_unions_overlapping_boxes():
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.625, 0.5, 0.5, 0.5]])
    target = foreground_target_from_boxes(make_batch(boxes, torch.tensor([0, 0])), size=(4, 4))
    assert target[0, 0, 1:3, 1:4].all()


def test_box_to_foreground_target_clips_border_boxes():
    boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5], [1.0, 1.0, 0.5, 0.5]])
    target = foreground_target_from_boxes(make_batch(boxes, torch.tensor([0, 0])), size=(4, 4))
    assert target[0, 0, 0, 0] == 1
    assert target[0, 0, -1, -1] == 1


def test_target_uses_final_augmented_box_coordinates():
    # This is a resized/flipped final batch: target must follow the final box,
    # not an earlier source-image coordinate system.
    boxes = torch.tensor([[0.75, 0.25, 0.25, 0.25]])
    target = foreground_target_from_boxes(make_batch(boxes, torch.tensor([0])), size=(8, 8))
    assert target[0, 0, 1:3, 5:7].all()
    assert target[0, 0, 5:7, 1:3].sum() == 0


def test_foreground_bce_has_gradients():
    logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 1:3, 1:3] = 1
    foreground_bce_loss(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_foreground_head_amp_half_path():
    from yolo_superpixels.strategies.structural_supervision import ForegroundnessHead

    head = ForegroundnessHead(8).half()
    output = head(torch.rand(1, 8, 4, 4).half())
    assert output.dtype == torch.float16


@pytest.mark.parametrize("algorithm", [TorchSLICSuperpixelAlgorithm(n_segments=8, num_iters=1), TorchGridSuperpixelAlgorithm(n_segments=8)])
def test_label_algorithms_generate_feature_grid_labels(algorithm):
    labels = labels_on_feature_grid(algorithm, torch.rand(1, 3, 16, 16), size=(4, 4))
    assert labels.shape == (1, 4, 4)
    assert labels.dtype == torch.long


def test_region_loss_is_label_permutation_invariant():
    probability = torch.tensor([[[[0.1, 0.3], [0.7, 0.9]]]], requires_grad=True)
    target = torch.zeros_like(probability)
    labels_a = torch.tensor([[[0, 0], [1, 1]]])
    labels_b = torch.tensor([[[9, 9], [3, 3]]])
    loss_a, _ = region_consistency_loss(probability, labels_a, target)
    loss_b, _ = region_consistency_loss(probability, labels_b, target)
    torch.testing.assert_close(loss_a, loss_b)


def test_pure_foreground_and_background_regions_are_active():
    probability = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    target = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    labels = torch.tensor([[[1, 1], [2, 2]]])
    _, stats = region_consistency_loss(probability, labels, target, purity_threshold=0.9, min_region_size=2)
    assert stats["active_pure_regions"] == 2
    assert stats["ignored_ambiguous_regions"] == 0


def test_ambiguous_regions_are_ignored():
    probability = torch.tensor([[[[0.1, 0.9], [0.5, 0.5]]]])
    target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    labels = torch.zeros((1, 2, 2), dtype=torch.long)
    loss, stats = region_consistency_loss(probability, labels, target, purity_threshold=0.9, min_region_size=1)
    torch.testing.assert_close(loss, probability.sum() * 0.0)
    assert stats["active_pure_regions"] == 0
    assert stats["ignored_ambiguous_regions"] == 1


def test_minimum_region_size_excludes_tiny_regions():
    probability = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    target = torch.zeros_like(probability)
    labels = torch.tensor([[[1, 1], [2, 3]]])
    _, stats = region_consistency_loss(probability, labels, target, min_region_size=2)
    assert stats["active_pure_regions"] == 1
    assert stats["ignored_small_regions"] == 2


def test_constant_predictions_have_zero_region_variance_and_gradients():
    logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
    probabilities = torch.sigmoid(logits)
    labels = torch.tensor([[[1, 1], [2, 2]]])
    target = torch.zeros_like(probabilities)
    loss, _ = region_consistency_loss(probabilities, labels, target)
    assert loss.abs() < 1e-7
    loss.backward()
    assert logits.grad is not None


def test_region_consistency_has_gradients():
    logits = torch.tensor([[[[-1.0, 1.0], [0.0, 0.5]]]], requires_grad=True)
    target = torch.zeros_like(logits)
    labels = torch.zeros((1, 2, 2), dtype=torch.long)
    loss, _ = region_consistency_loss(torch.sigmoid(logits), labels, target, min_region_size=1)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_lambda_sp_zero_never_calls_region_algorithm():
    context = StructuralSupervisionContext()
    context.image = torch.rand(1, 3, 8, 8)
    context.foreground_logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    config = StructuralSupervisionConfig(lambda_fg=0.1, lambda_sp=0.0)
    criterion = StructuralSupervisionCriterion(BaseCriterion(), context, config)
    assert criterion.region_algorithm is None
    loss, items = criterion(None, make_batch(torch.tensor([[0.5, 0.5, 0.5, 0.5]]), torch.tensor([0])))
    assert torch.isfinite(loss)
    assert items.shape[0] == 12


def test_patch_a_does_not_install_sp11_and_b_installs_head():
    arm_a = TinyYOLOLike()
    assert not any(isinstance(module, StructuralSupervisionWrapper) for module in arm_a.modules())
    arm_b = TinyYOLOLike()
    result = patch_structural_supervision(arm_b, StructuralSupervisionConfig(lambda_sp=0.0), install_criterion=True)
    assert result.target_layer == 1
    assert isinstance(arm_b.model[1], StructuralSupervisionWrapper)


def test_arm_c_and_d_select_expected_label_algorithms():
    assert isinstance(StructuralSupervisionConfig(region_algorithm="torch_slic").build_region_algorithm(), TorchSLICSuperpixelAlgorithm)
    assert isinstance(StructuralSupervisionConfig(region_algorithm="torch_grid", region_kwargs={"n_segments": 300}).build_region_algorithm(), TorchGridSuperpixelAlgorithm)


def test_eval_does_not_run_head_or_generate_regions():
    model = TinyYOLOLike()
    patch_structural_supervision(model, StructuralSupervisionConfig(lambda_sp=0.05), install_criterion=False)
    wrapper = model.model[1]
    assert isinstance(wrapper, StructuralSupervisionWrapper)
    model.eval()
    with torch.no_grad():
        model(torch.rand(1, 3, 8, 8))
    assert wrapper.context.foreground_logits is None
    assert wrapper.context.image is None


def test_strip_restores_detection_module_and_output_shape():
    model = TinyYOLOLike()
    patch_structural_supervision(model, StructuralSupervisionConfig(lambda_sp=0.0), install_criterion=False)
    image = torch.rand(1, 3, 8, 8)
    model.eval()
    before = model(image)
    assert strip_structural_supervision(model) == 1
    assert isinstance(model.model[1], nn.Conv2d)
    after = model(image)
    torch.testing.assert_close(before, after)


def test_criterion_clears_context_and_is_deepcopy_safe():
    context = StructuralSupervisionContext()
    context.image = torch.rand(1, 3, 8, 8)
    context.foreground_logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    criterion = StructuralSupervisionCriterion(BaseCriterion(), context, StructuralSupervisionConfig(lambda_sp=0.0))
    copied = deepcopy(criterion)
    assert copied.base_criterion is not None
    criterion(None, make_batch())
    assert context.image is None
    assert context.foreground_logits is None
