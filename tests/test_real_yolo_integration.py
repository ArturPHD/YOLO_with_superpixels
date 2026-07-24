from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from yolo_superpixels.strategies.auxiliary_loss import AuxiliaryLossConfig, AuxiliarySuperpixelWrapper, patch_auxiliary_loss
from yolo_superpixels.strategies.neck_injection import NeckInjectionConfig, NeckInjectionWrapper, patch_neck_injection

ROOT = Path(__file__).resolve().parents[1]


def _load_real_yolo(weights_name: str):
    weights = ROOT / weights_name
    if not weights.exists():
        pytest.skip(f"{weights_name} is not available locally; real YOLO integration test skipped.")
    ultralytics = pytest.importorskip("ultralytics")
    return ultralytics.YOLO(str(weights)).model


@pytest.mark.parametrize(
    ("weights_name", "expected_target_layer"),
    [("yolo12n.pt", 14), ("yolo26n.pt", 16)],
)
def test_real_yolo_neck_target_is_stride8_p3(weights_name, expected_target_layer):
    model = _load_real_yolo(weights_name)
    config = NeckInjectionConfig(
        fusion="add",
        token_provider="region",
        region_algorithm="torch_slic",
        region_n_segments=8,
        region_num_iters=1,
        emb_dim=16,
        image_size=64,
        validate_target=True,
        validation_imgsz=64,
    )

    result = patch_neck_injection(model, config=config)

    assert result.target_layer == expected_target_layer
    assert result.target_stride == 8
    assert result.target_shape == (8, 8)


@pytest.mark.parametrize("weights_name", ["yolo12n.pt", "yolo26n.pt"])
@pytest.mark.parametrize("fusion", ["add", "concat", "cross_attention"])
def test_real_yolo_neck_fusion_parameters_are_in_optimizer_before_forward(weights_name, fusion):
    model = _load_real_yolo(weights_name)
    config = NeckInjectionConfig(
        fusion=fusion,
        token_provider="region",
        region_algorithm="torch_slic",
        region_n_segments=8,
        region_num_iters=1,
        emb_dim=16,
        attention_heads=2,
        image_size=64,
        validate_target=True,
        validation_imgsz=64,
    )

    result = patch_neck_injection(model, config=config)
    wrapper = model.model[result.target_layer]
    assert isinstance(wrapper, NeckInjectionWrapper)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer_param_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    fusion_params = [parameter for parameter in wrapper.fusion.parameters() if parameter.requires_grad]

    assert fusion_params
    assert all(id(parameter) in optimizer_param_ids for parameter in fusion_params)


@pytest.mark.parametrize("weights_name", ["yolo12n.pt", "yolo26n.pt"])
def test_real_yolo_neck_forward_after_eager_initialization(weights_name):
    model = _load_real_yolo(weights_name)
    config = NeckInjectionConfig(
        fusion="add",
        token_provider="region",
        region_algorithm="torch_slic",
        region_n_segments=8,
        region_num_iters=1,
        emb_dim=16,
        image_size=64,
        validate_target=True,
        validation_imgsz=64,
    )
    patch_neck_injection(model, config=config)

    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 64))

    assert output is not None


@pytest.mark.parametrize(
    ("weights_name", "expected_target_layer"),
    [("yolo12n.pt", 14), ("yolo26n.pt", 16)],
)
def test_real_yolo_auxiliary_loss_returns_logged_sp_loss(weights_name, expected_target_layer):
    model = _load_real_yolo(weights_name)
    config = AuxiliaryLossConfig(
        algorithm_names=["torch_slic"],
        algorithm_kwargs={"n_segments": 8, "compactness": 1.0, "num_iters": 1},
        loss_weight=0.05,
        target_mode="labels",
        loss_type="balanced_bce",
    )

    result = patch_auxiliary_loss(model, config=config, install_criterion=True)
    assert result.target_layer == expected_target_layer
    assert isinstance(model.model[result.target_layer], AuxiliarySuperpixelWrapper)

    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.empty((0,), dtype=torch.long),
        "cls": torch.empty((0, 1)),
        "bboxes": torch.empty((0, 4)),
    }
    model.train()
    loss, loss_items = model(batch)

    assert torch.isfinite(loss).all()
    assert loss_items.shape[0] == 4
    assert loss_items[-1] > 0
