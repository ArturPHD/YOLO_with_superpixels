import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from yolo_superpixels.strategies.neck_injection import (
    NeckInjectionConfig,
    TokenBundle,
    patch_neck_injection,
    tokens_to_dense_map,
)


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
