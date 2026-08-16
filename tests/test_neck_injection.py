import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from yolo_superpixels.strategies.neck_injection import (
    NeckInjectionConfig,
    TokenBundle,
    patch_neck_injection,
    tokens_to_dense_map,
)
from yolo_superpixels.strategies.neck_injection.tokens import _compatible_state_dict


class FakeTokenProvider(nn.Module):
    def __init__(self, token_dim=6):
        super().__init__()
        self.token_dim = token_dim

    def forward(self, rgb):
        batch, _, height, width = rgb.shape
        tokens = torch.zeros((batch, 3, self.token_dim), dtype=rgb.dtype, device=rgb.device)
        tokens[:, 1] = 1.0
        tokens[:, 2] = 2.0
        labels = torch.ones((batch, height, width), dtype=torch.long, device=rgb.device)
        labels[:, :, width // 2 :] = 2
        valid = torch.ones((batch, 3), dtype=torch.bool, device=rgb.device)
        return TokenBundle(tokens=tokens, valid_mask=valid, labels=labels)


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


def test_tokens_to_dense_map_scatters_by_labels():
    tokens = torch.tensor([[[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    labels = torch.tensor([[[1, 2], [2, 1]]])
    dense = tokens_to_dense_map(TokenBundle(tokens=tokens, valid_mask=valid, labels=labels), (2, 2))

    assert dense.shape == (1, 2, 2, 2)
    torch.testing.assert_close(dense[0, :, 0, 0], torch.tensor([1.0, 10.0]))
    torch.testing.assert_close(dense[0, :, 0, 1], torch.tensor([2.0, 20.0]))


def test_compatible_state_dict_skips_mismatched_checkpoint_head():
    module = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    state = {
        "module.0.weight": torch.ones_like(module[0].weight),
        "module.0.bias": torch.ones_like(module[0].bias),
        "module.1.weight": torch.ones((1000, 3)),
        "module.1.bias": torch.ones(1000),
        "module.extra.weight": torch.ones(1),
    }

    compatible, skipped = _compatible_state_dict(module, state)

    assert set(compatible) == {"0.weight", "0.bias"}
    assert skipped["1.weight"].startswith("shape")
    assert skipped["1.bias"].startswith("shape")
    assert skipped["extra.weight"] == "missing"


@pytest.mark.parametrize("fusion", ["add", "concat", "cross_attention"])
def test_patch_neck_injection_supports_all_fusion_modes(fusion):
    model = TinyYOLOLike()
    config = NeckInjectionConfig(
        fusion=fusion,
        gate_init="tiny",
        attention_heads=2,
        tokenizer="default",
    )

    result = patch_neck_injection(model, config=config, token_provider=FakeTokenProvider())

    assert result.target_layer == 1
    assert result.detect_layer == 3
    assert result.detect_sources == [1, 2]

    output = model(torch.rand(2, 3, 8, 8))
    assert output.shape == (2, 4, 8, 8)
    assert torch.isfinite(output).all()


def test_patch_neck_injection_can_disable_injection():
    model = TinyYOLOLike()
    config = NeckInjectionConfig(fusion="add", enabled=False, tokenizer="default")
    patch_neck_injection(model, config=config, token_provider=FakeTokenProvider())

    output = model(torch.rand(1, 3, 6, 6))

    assert output.shape == (1, 4, 6, 6)


def _fake_spit_provider():
    from yolo_superpixels.strategies.neck_injection.tokens import SPiTTokenProvider

    provider = SPiTTokenProvider.__new__(SPiTTokenProvider)
    nn.Module.__init__(provider)
    provider.spit = nn.Sequential(nn.Linear(3, 4), nn.Dropout(p=0.5), nn.Linear(4, 2))
    provider._spit_frozen = False
    provider.loaded_spit_weights = None
    provider.loaded_spit_weight_keys = 0
    provider.skipped_spit_weight_keys = {}
    provider.missing_spit_weight_keys = []
    return provider


def test_spit_freeze_survives_parent_train_mode():
    provider = _fake_spit_provider()

    provider.set_spit_frozen(True)
    provider.train(True)

    assert provider.training is True
    assert provider.spit_frozen is True
    assert provider.spit.training is False
    assert all(not parameter.requires_grad for parameter in provider.spit.parameters())
    assert provider.spit_parameter_counts()[0] == 0

    provider.set_spit_frozen(False)
    provider.train(True)

    assert provider.spit_frozen is False
    assert provider.spit.training is True
    assert all(parameter.requires_grad for parameter in provider.spit.parameters())
    trainable, total = provider.spit_parameter_counts()
    assert trainable == total


def test_delayed_spit_policy_freezes_first_21_complete_epochs():
    from yolo_superpixels.integrations.ultralytics_neck import _spit_should_be_frozen

    config = NeckInjectionConfig(spit_train_mode="delayed", spit_freeze_epochs=21)

    assert _spit_should_be_frozen(config, 0) is True
    assert _spit_should_be_frozen(config, 20) is True
    assert _spit_should_be_frozen(config, 21) is False


def test_spit_optimizer_group_split_scales_only_spit_lr():
    from yolo_superpixels.integrations.ultralytics_neck import _split_spit_optimizer_groups

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.yolo = nn.Linear(3, 3)
            self.provider = _fake_spit_provider()

    model = ToyModel()
    optimizer = torch.optim.SGD(
        [
            {
                "params": list(model.parameters()),
                "lr": 0.01,
                "weight_decay": 0.001,
                "param_group": "weight",
            }
        ],
        lr=0.01,
        momentum=0.9,
    )

    indices = _split_spit_optimizer_groups(optimizer, model, lr_scale=0.01)

    assert len(indices) == 1
    spit_group = optimizer.param_groups[indices[0]]
    assert spit_group["lr"] == pytest.approx(0.0001)
    assert spit_group["weight_decay"] == pytest.approx(0.001)
    assert spit_group["spit_group"] is True
    assert spit_group["spit_lr_scale"] == pytest.approx(0.01)

    spit_ids = {id(parameter) for parameter in model.provider.spit.parameters()}
    regular_ids = {
        id(parameter)
        for index, group in enumerate(optimizer.param_groups)
        if index not in indices
        for parameter in group["params"]
    }
    scaled_ids = {id(parameter) for parameter in spit_group["params"]}
    assert scaled_ids == spit_ids
    assert regular_ids.isdisjoint(spit_ids)
