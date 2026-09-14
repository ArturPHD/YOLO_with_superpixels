import pytest

torch = pytest.importorskip("torch")

from yolo_superpixels.superpixels.algorithms import TorchSLICSuperpixelAlgorithm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA repeatability smoke requires a CUDA device.")
def test_torch_slic_cuda_repeatability_under_deterministic_setting():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        image = torch.linspace(0, 1, 3 * 48 * 48, device="cuda", dtype=torch.float32).reshape(1, 3, 48, 48)
        algorithm = TorchSLICSuperpixelAlgorithm(n_segments=32, compactness=1.0, num_iters=2)
        first = algorithm.labels_for_batch(image)
        second = algorithm.labels_for_batch(image)
        assert torch.equal(first, second), "torch_slic labels were not bit-identical across repeated CUDA calls."
    finally:
        torch.use_deterministic_algorithms(previous)
