"""Regression coverage for the vendored Q* GPU-native adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_superpixels import QStarSuperpixelAlgorithm
from yolo_superpixels.superpixels.qstar import QStarConfig, QStarRunner
from yolo_superpixels.strategies.neck_injection import NeckInjectionConfig
from yolo_superpixels.strategies.neck_injection.tokens import QStarTokenProvider


def test_qstar_token_provider_returns_local_tokens_and_projection_gradients():
    provider = QStarTokenProvider(emb_dim=12, image_size=160, target_k=64)
    bundle = provider(torch.rand(1, 3, 128, 160))

    assert bundle.tokens.shape[0] == 1
    assert bundle.tokens.shape[-1] == 12
    assert bundle.labels.shape == (1, 128, 160)
    assert bundle.valid_mask.dtype is torch.bool
    assert torch.all(bundle.valid_mask[:, 0])
    assert int(bundle.labels.min()) == 1
    for index in range(1):
        assert int(bundle.labels[index].max()) + 1 == int(bundle.valid_mask[index].sum())

    bundle.tokens.square().mean().backward()
    assert next(provider.embedder.parameters()).grad is not None


def test_qstar_labels_and_edges_are_tensor_native_on_cpu():
    algorithm = QStarSuperpixelAlgorithm(target_k=64)
    image = torch.rand(1, 3, 128, 160)
    labels = algorithm.labels_for_batch(image)
    edges = algorithm.edges_for_batch(image)

    assert labels.shape == (1, 128, 160)
    assert labels.device == image.device
    assert edges.shape == (1, 2, 128, 160)
    assert edges.device == image.device
    assert torch.isfinite(edges).all()


def test_qstar_config_builds_provider_and_preserves_requested_hierarchy_level():
    provider = NeckInjectionConfig(
        token_provider="qstar",
        emb_dim=10,
        image_size=160,
        qstar_target_k=64,
        qstar_level=1,
    ).build_token_provider()

    assert isinstance(provider, QStarTokenProvider)
    assert provider.qstar.level == 1
    bundle = provider(torch.rand(1, 3, 128, 160))
    assert bundle.tokens.shape[-1] == 10


def test_qstar_pads_a_tiny_rectangular_image_and_crops_labels_back():
    provider = QStarTokenProvider(emb_dim=8, image_size=None, target_k=32)
    bundle = provider(torch.rand(1, 3, 64, 96))
    assert bundle.labels.shape == (1, 64, 96)
    assert torch.isfinite(bundle.tokens).all()


def test_qstar_sanitizes_nonfinite_framework_warmup_input():
    provider = QStarTokenProvider(emb_dim=8, image_size=None, target_k=32)
    image = torch.rand(1, 3, 64, 96)
    image[:, :, 0, 0] = float("nan")
    image[:, :, 0, 1] = float("inf")
    image[:, :, 0, 2] = float("-inf")

    bundle = provider(image)

    assert bundle.labels.shape == (1, 64, 96)
    assert torch.isfinite(bundle.tokens).all()


def test_qstar_adaptive_quadtree_handles_invalid_candidates_before_masking():
    # This fixed seed exercises candidates from small boxes that would extend
    # beyond the integral image before Q* applies its valid-candidate mask.
    torch.manual_seed(41)
    runner = QStarRunner(QStarConfig(target_k=64))
    results = [runner(torch.rand(1, 3, 160, 160)) for _ in range(4)]

    assert all(result.labels.shape == (1, 160, 160) for result in results)
    assert all(torch.isfinite(result.features).all() for result in results)


def test_qstar_token_provider_preserves_rectangular_geometry_inside_its_budget():
    provider = QStarTokenProvider(emb_dim=8, image_size=160, target_k=32)
    bundle = provider(torch.rand(1, 128, 192).unsqueeze(1).repeat(1, 3, 1, 1))

    # 128x192 fits a 160x160 budget as 107x160; the Q* working canvas is
    # padded internally and labels are cropped back before token fusion.
    assert bundle.labels.shape == (1, 107, 160)
    assert torch.isfinite(bundle.tokens).all()


def test_qstar_import_shim_exposes_the_vendored_checkpoint_module():
    import qstar_superpixels

    assert qstar_superpixels.SuperpixelPipeline.__module__ == "qstar_superpixels.pipeline"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this device-residency smoke.")
def test_qstar_cuda_provider_keeps_tokens_masks_and_labels_on_cuda():
    provider = QStarTokenProvider(emb_dim=8, image_size=160, target_k=64).cuda()
    image = torch.rand(1, 3, 128, 160, device="cuda")
    bundle = provider(image)

    assert bundle.tokens.device == image.device
    assert bundle.valid_mask.device == image.device
    assert bundle.labels.device == image.device
    bundle.tokens.sum().backward()
    assert next(provider.embedder.parameters()).grad is not None
