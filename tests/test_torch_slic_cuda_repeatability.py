import pytest

torch = pytest.importorskip("torch")

from yolo_superpixels.strategies.structural_supervision.protocol import torch_slic_repeatability_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA repeatability smoke requires a CUDA device.")
def test_torch_slic_cuda_repeatability_under_deterministic_setting():
    record = torch_slic_repeatability_metadata()
    assert record["bit_identical"], "torch_slic labels were not bit-identical across repeated CUDA calls."
    assert len(set(record["label_sha256"])) == 1
    assert record["gpu"] and record["torch"] and record["cuda"]
