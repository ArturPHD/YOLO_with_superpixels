from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from yolo_superpixels.strategies.backbone_correspondence import (
    BackboneCorrespondenceConfig,
    BackboneCorrespondenceFusion,
    build_overlap_edges,
    patch_backbone_correspondence,
)
from yolo_superpixels.strategies.backbone_correspondence.patch import (
    BackboneP3Wrapper,
    IsolatedP3ConsumerWrapper,
)
from yolo_superpixels.strategies.neck_injection import TokenBundle

ROOT = Path(__file__).resolve().parents[1]


class FixedTokenProvider(nn.Module):
    emb_dim = 4

    def forward(self, rgb):
        batch = rgb.shape[0]
        tokens = torch.zeros((batch, 5, self.emb_dim), device=rgb.device, dtype=rgb.dtype)
        tokens[:, 1] = 1.0
        tokens[:, 2] = 2.0
        tokens[:, 3] = 3.0
        tokens[:, 4] = 4.0
        labels = torch.tensor(
            [[[1, 1, 2, 2], [1, 3, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]],
            device=rgb.device,
            dtype=torch.long,
        ).expand(batch, -1, -1)
        valid = torch.ones((batch, 5), device=rgb.device, dtype=torch.bool)
        return TokenBundle(tokens=tokens, valid_mask=valid, labels=labels)


def test_overlap_edges_match_exact_raster_area():
    labels = FixedTokenProvider()(torch.zeros(1, 3, 4, 4)).labels
    edges = build_overlap_edges(labels, (2, 2), num_tokens=5)

    pairs = {
        (int(cell), int(region)): float(area)
        for cell, region, area in zip(edges.cell_index.cpu(), edges.region_index.cpu(), edges.area.cpu())
    }
    assert pairs[(0, 1)] == pytest.approx(0.75)
    assert pairs[(0, 3)] == pytest.approx(0.25)
    assert pairs[(1, 2)] == pytest.approx(1.0)
    assert pairs[(2, 3)] == pytest.approx(1.0)
    assert pairs[(3, 4)] == pytest.approx(1.0)
    totals = torch.zeros(4).index_add_(0, edges.cell_index.cpu(), edges.area.cpu())
    torch.testing.assert_close(totals, torch.ones(4))


@pytest.mark.parametrize("mode", ["overlap_pool", "overlap_attention"])
def test_local_correspondence_fusion_is_finite_and_differentiable(mode):
    fusion = BackboneCorrespondenceFusion(FixedTokenProvider(), mode=mode, gate_init="one", attention_heads=2)
    fusion.initialize(yolo_channels=4, token_dim=4)
    feature = torch.randn(2, 4, 2, 2, requires_grad=True)
    output = fusion(feature, torch.rand(2, 3, 4, 4))
    output.square().mean().backward()

    assert output.shape == feature.shape
    assert torch.isfinite(output).all()
    assert feature.grad is not None
    assert any(parameter.grad is not None for parameter in fusion.parameters() if parameter.requires_grad)


def _load_real_yolo(weights_name: str):
    weights = ROOT / weights_name
    if not weights.exists():
        pytest.skip(f"{weights_name} is not available locally; real YOLO integration test skipped.")
    ultralytics = pytest.importorskip("ultralytics")
    return ultralytics.YOLO(str(weights)).model


@pytest.mark.parametrize(
    ("weights_name", "consumer_layer"),
    [("yolo12n.pt", 13), ("yolo26n.pt", 15)],
)
@pytest.mark.parametrize("placement", ["native_backbone", "isolated_p3"])
def test_real_yolo_p3_routes_and_optimizer_include_fusion(weights_name, consumer_layer, placement):
    model = _load_real_yolo(weights_name)
    config = BackboneCorrespondenceConfig(
        placement=placement,
        token_provider="region",
        region_n_segments=8,
        region_num_iters=1,
        emb_dim=16,
        image_size=64,
        attention_heads=2,
        validation_imgsz=64,
    )
    result = patch_backbone_correspondence(model, config)

    assert result.p3_layer == 4
    assert result.p3_consumer_layer == consumer_layer
    assert result.p3_stride == 8
    assert result.p3_shape == (8, 8)
    if placement == "native_backbone":
        assert isinstance(model.model[4], BackboneP3Wrapper)
    else:
        assert isinstance(model.model[consumer_layer], IsolatedP3ConsumerWrapper)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    wrapper = model.model[4] if placement == "native_backbone" else model.model[consumer_layer]
    fusion_parameters = [parameter for parameter in wrapper.fusion.parameters() if parameter.requires_grad]
    assert fusion_parameters
    assert all(id(parameter) in optimizer_ids for parameter in fusion_parameters)

    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 64))
    assert output is not None
