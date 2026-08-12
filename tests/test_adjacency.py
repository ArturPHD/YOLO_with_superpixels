import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yolo_superpixels import (
    AdjacencyChannelBuilder,
    DistanceChannelBuilder,
    SPiTDenseAdjacencyAlgorithm,
    TorchColorAffinityAlgorithm,
    TorchGradientAffinityAlgorithm,
    TorchGridSuperpixelAlgorithm,
    TorchSLICSuperpixelAlgorithm,
    TorchVoronoiSuperpixelAlgorithm,
    tensor_edges_to_soft_distance_channels,
    tensor_labels_to_soft_distance_channels,
)
from yolo_superpixels.adjacency import algorithm_output_to_edges, labels_to_adjacency


def test_labels_to_right_down_adjacency():
    labels = np.array(
        [
            [1, 1, 2],
            [1, 3, 3],
        ]
    )

    edges = labels_to_adjacency(labels)

    expected_right = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    expected_down = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(edges[0], expected_right)
    np.testing.assert_array_equal(edges[1], expected_down)


def test_algorithm_output_accepts_hwc_edges():
    edge_hwc = np.zeros((4, 5, 2), dtype=np.float32)
    edge_hwc[..., 0] = 1.0

    edges = algorithm_output_to_edges(edge_hwc, height=4, width=5)

    assert edges.shape == (2, 4, 5)
    np.testing.assert_array_equal(edges[0], np.ones((4, 5), dtype=np.float32))
    np.testing.assert_array_equal(edges[1], np.zeros((4, 5), dtype=np.float32))


def test_algorithm_output_reorders_prebuilt_cpu_edges():
    edge_chw = np.zeros((2, 4, 5), dtype=np.float32)
    edge_chw[0] = 1.0

    edges = algorithm_output_to_edges(edge_chw, height=4, width=5, channel_order=("down", "right"))

    np.testing.assert_array_equal(edges[0], np.zeros((4, 5), dtype=np.float32))
    np.testing.assert_array_equal(edges[1], np.ones((4, 5), dtype=np.float32))


class TensorOnlyAlgorithm:
    def __init__(self) -> None:
        self.seen_device = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise AssertionError("CPU numpy path should not be used for tensor algorithms.")

    def edges_for_batch(self, batch_tensor, channel_order=("right", "down")):
        self.seen_device = batch_tensor.device
        batch, _, height, width = batch_tensor.shape
        right = torch.ones((batch, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)
        down = torch.zeros((batch, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)
        edges = torch.stack((right, down), dim=1)
        if tuple(channel_order) == ("down", "right"):
            edges = edges[:, [1, 0]]
        return edges


class TensorValidNeighborAlgorithm(TensorOnlyAlgorithm):
    def edges_for_batch(self, batch_tensor, channel_order=("right", "down")):
        self.seen_device = batch_tensor.device
        batch, _, height, width = batch_tensor.shape
        right = torch.zeros((batch, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)
        down = torch.zeros((batch, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)
        if width > 1:
            right[:, :, :-1] = 1.0
        edges = torch.stack((right, down), dim=1)
        if tuple(channel_order) == ("down", "right"):
            edges = edges[:, [1, 0]]
        return edges


class CpuLabelAlgorithm:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [1, 1, 2],
                [1, 3, 3],
            ],
            dtype=np.int32,
        )


class TensorLabelAlgorithm:
    def __init__(self) -> None:
        self.seen_device = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise AssertionError("CPU numpy path should not be used when labels_for_batch exists.")

    def labels_for_batch(self, batch_tensor):
        self.seen_device = batch_tensor.device
        batch, _, height, width = batch_tensor.shape
        labels = torch.zeros((batch, height, width), dtype=torch.long, device=batch_tensor.device)
        labels[:, :, width // 2 :] = 1
        return labels


def test_adjacency_builder_uses_tensor_batch_path():
    algorithm = TensorOnlyAlgorithm()
    builder = AdjacencyChannelBuilder(algorithm, edge_representation="same")
    images = torch.rand(2, 3, 4, 5)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 5, 4, 5)
    assert output.device == images.device
    assert algorithm.seen_device == images.device
    torch.testing.assert_close(output[:, :3], images)
    torch.testing.assert_close(output[:, 3], torch.ones((2, 4, 5)))
    torch.testing.assert_close(output[:, 4], torch.zeros((2, 4, 5)))


@pytest.mark.parametrize(
    ("representation", "expected_right_value", "expected_down_value"),
    [
        ("same", 1.0, 0.0),
        ("boundary", 0.0, 1.0),
        ("zero_centered", 1.0, -1.0),
        ("boundary_zero_centered", -1.0, 1.0),
    ],
)
def test_adjacency_builder_supports_edge_representations(
    representation,
    expected_right_value,
    expected_down_value,
):
    builder = AdjacencyChannelBuilder(TensorValidNeighborAlgorithm(), edge_representation=representation)
    images = torch.rand(1, 3, 4, 5)

    output = builder.append_to_batch(images)

    torch.testing.assert_close(output[:, 3, :, :-1], torch.full((1, 4, 4), expected_right_value))
    torch.testing.assert_close(output[:, 4, :-1, :], torch.full((1, 3, 5), expected_down_value))
    torch.testing.assert_close(output[:, 3, :, -1], torch.zeros((1, 4)))
    torch.testing.assert_close(output[:, 4, -1, :], torch.zeros((1, 5)))


def test_cpu_label_algorithm_zero_centered_is_not_transformed_twice():
    builder = AdjacencyChannelBuilder(CpuLabelAlgorithm(), edge_representation="zero_centered")
    images = torch.rand(1, 3, 2, 3)

    output = builder.append_to_batch(images)

    expected_right = torch.tensor([[[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]])
    expected_down = torch.tensor([[[1.0, -1.0, -1.0], [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(output[:, 3], expected_right)
    torch.testing.assert_close(output[:, 4], expected_down)
    assert set(output[:, 3:].unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_tensor_labels_to_soft_interior_map_peaks_in_region_center():
    labels = torch.zeros(1, 5, 5, dtype=torch.long)

    channels = tensor_labels_to_soft_distance_channels(
        labels,
        mode="soft_interior",
        max_distance=4,
        boundary_width=2.0,
    )

    assert channels.shape == (1, 1, 5, 5)
    assert channels[0, 0, 2, 2] == 1.0
    assert channels[0, 0, 1, 1] == 0.5
    assert channels[0, 0, 0, 0] == 0.0


def test_tensor_labels_to_soft_boundary_map_fades_from_boundary():
    labels = torch.zeros(1, 5, 5, dtype=torch.long)

    channels = tensor_labels_to_soft_distance_channels(
        labels,
        mode="soft_boundary",
        max_distance=4,
        boundary_width=2.0,
    )

    assert channels.shape == (1, 1, 5, 5)
    assert channels[0, 0, 0, 0] == 1.0
    assert channels[0, 0, 1, 1] == 0.5
    assert channels[0, 0, 2, 2] == 0.0


def test_distance_builder_supports_four_channel_soft_interior_gpu_path():
    algorithm = TensorLabelAlgorithm()
    builder = DistanceChannelBuilder(algorithm, mode="soft_interior", max_distance=8)
    images = torch.rand(2, 3, 8, 8)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 4, 8, 8)
    assert algorithm.seen_device == images.device
    torch.testing.assert_close(output[:, :3], images)
    assert torch.isfinite(output[:, 3]).all()
    assert float(output[:, 3].min()) >= 0.0
    assert float(output[:, 3].max()) <= 1.0


def test_distance_builder_supports_five_channel_boundary_interior_mode():
    builder = DistanceChannelBuilder(TensorLabelAlgorithm(), mode="soft_boundary_interior")
    images = torch.rand(2, 3, 8, 8)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 5, 8, 8)
    torch.testing.assert_close(output[:, :3], images)
    assert torch.isfinite(output[:, 3:]).all()


def test_edges_to_soft_distance_channels_support_affinity_without_labels():
    edges = torch.ones(1, 2, 5, 5)
    edges[:, 0, :, 2] = 0.0

    channels = tensor_edges_to_soft_distance_channels(
        edges,
        mode="soft_boundary_interior",
        max_distance=2,
        boundary_width=2.0,
        threshold=0.5,
    )

    assert channels.shape == (1, 2, 5, 5)
    assert channels[0, 0, 2, 2] == 1.0
    assert channels[0, 1, 2, 2] == 0.0
    assert channels[0, 1, 2, 1] > 0.0


def test_distance_builder_supports_color_affinity_edge_fallback():
    builder = DistanceChannelBuilder(
        TorchColorAffinityAlgorithm(temperature=0.1),
        mode="soft_interior",
        max_distance=2,
    )
    images = torch.zeros(1, 3, 5, 5)

    output = builder.append_to_batch(images)

    assert output.shape == (1, 4, 5, 5)
    torch.testing.assert_close(output[:, :3], images)
    assert output[0, 3, 2, 2] == 1.0
    assert output[0, 3, 0, 0] == 0.0


def test_distance_builder_can_disable_prebuilt_soft_channel():
    builder = DistanceChannelBuilder(TensorLabelAlgorithm(), mode="soft_interior", enabled=False)
    images = torch.rand(2, 4, 8, 8)

    output = builder.append_to_batch(images)

    torch.testing.assert_close(output[:, :3], images[:, :3])
    torch.testing.assert_close(output[:, 3:], torch.zeros(2, 1, 8, 8))


def test_adjacency_builder_default_representation_is_zero_centered():
    builder = AdjacencyChannelBuilder(TensorValidNeighborAlgorithm())
    images = torch.rand(1, 3, 4, 5)

    output = builder.append_to_batch(images)

    torch.testing.assert_close(output[:, 3, :, :-1], torch.ones((1, 4, 4)))
    torch.testing.assert_close(output[:, 4, :-1, :], -torch.ones((1, 3, 5)))
    torch.testing.assert_close(output[:, 3, :, -1], torch.zeros((1, 4)))
    torch.testing.assert_close(output[:, 4, -1, :], torch.zeros((1, 5)))


def test_adjacency_builder_can_disable_extra_channels_without_algorithm_call():
    builder = AdjacencyChannelBuilder(TensorOnlyAlgorithm(), enabled=False)
    images = torch.rand(2, 3, 4, 5)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 5, 4, 5)
    torch.testing.assert_close(output[:, :3], images)
    torch.testing.assert_close(output[:, 3:], torch.zeros((2, 2, 4, 5)))


def test_adjacency_builder_supports_rgb_only_no_algorithm():
    builder = AdjacencyChannelBuilder(None)
    images = torch.rand(2, 3, 4, 5)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 3, 4, 5)
    torch.testing.assert_close(output, images)


def test_adjacency_builder_supports_multiple_algorithms():
    builder = AdjacencyChannelBuilder(
        [TensorOnlyAlgorithm(), TensorOnlyAlgorithm()],
        edge_representation="same",
    )
    images = torch.rand(2, 3, 4, 5)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 7, 4, 5)
    torch.testing.assert_close(output[:, :3], images)
    torch.testing.assert_close(output[:, 3], torch.ones((2, 4, 5)))
    torch.testing.assert_close(output[:, 4], torch.zeros((2, 4, 5)))
    torch.testing.assert_close(output[:, 5], torch.ones((2, 4, 5)))
    torch.testing.assert_close(output[:, 6], torch.zeros((2, 4, 5)))


def test_adjacency_builder_zeros_prebuilt_channels_when_disabled():
    builder = AdjacencyChannelBuilder(TensorOnlyAlgorithm(), enabled=False)
    images = torch.rand(2, 5, 4, 5)

    output = builder.append_to_batch(images)

    torch.testing.assert_close(output[:, :3], images[:, :3])
    torch.testing.assert_close(output[:, 3:], torch.zeros((2, 2, 4, 5)))
    assert not torch.equal(output[:, 3:], images[:, 3:])


def test_color_affinity_algorithm_is_tensor_native():
    builder = AdjacencyChannelBuilder(
        TorchColorAffinityAlgorithm(temperature=0.1),
        edge_representation="same",
    )
    images = torch.zeros(1, 3, 3, 4)

    output = builder.append_to_batch(images)

    assert output.shape == (1, 5, 3, 4)
    torch.testing.assert_close(output[:, 3, :, :-1], torch.ones((1, 3, 3)))
    torch.testing.assert_close(output[:, 3, :, -1], torch.zeros((1, 3)))
    torch.testing.assert_close(output[:, 4, :-1, :], torch.ones((1, 2, 4)))
    torch.testing.assert_close(output[:, 4, -1, :], torch.zeros((1, 4)))


def test_torch_slic_algorithm_is_tensor_native():
    builder = AdjacencyChannelBuilder(
        TorchSLICSuperpixelAlgorithm(n_segments=4, compactness=10.0, num_iters=1),
        edge_representation="same",
    )
    images = torch.rand(2, 3, 8, 8)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 5, 8, 8)
    assert output.device == images.device
    torch.testing.assert_close(output[:, :3], images)
    assert torch.all((output[:, 3:] == 0) | (output[:, 3:] == 1))


class FakeSPiTDenseEmbedder:
    def eval(self):
        return self

    def __call__(self, batch):
        batch_size, _, height, width = batch.shape
        raw = torch.zeros((batch_size, 5, height, width), dtype=batch.dtype, device=batch.device)
        labels = torch.tensor(
            [
                [1, 1, 2],
                [1, 3, 3],
            ],
            device=batch.device,
        )
        return raw, labels.unsqueeze(0).expand(batch_size, -1, -1)


def test_spit_dense_adjacency_converts_labels_to_right_down_channels():
    builder = AdjacencyChannelBuilder(
        SPiTDenseAdjacencyAlgorithm(embedder=FakeSPiTDenseEmbedder()),
        edge_representation="same",
    )
    images = torch.rand(2, 3, 2, 3)

    output = builder.append_to_batch(images)

    expected_right = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=images.dtype)
    expected_down = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=images.dtype)
    assert output.shape == (2, 5, 2, 3)
    assert output.device == images.device
    torch.testing.assert_close(output[:, 3], expected_right.expand(2, -1, -1))
    torch.testing.assert_close(output[:, 4], expected_down.expand(2, -1, -1))


@pytest.mark.parametrize(
    "algorithm",
    [
        TorchGradientAffinityAlgorithm(temperature=0.1),
        TorchGridSuperpixelAlgorithm(n_segments=4),
        TorchVoronoiSuperpixelAlgorithm(n_segments=4, jitter=0.25, chunk_size=16),
    ],
)
def test_new_gpu_candidate_algorithms_are_tensor_native(algorithm):
    builder = AdjacencyChannelBuilder(algorithm)
    images = torch.rand(2, 3, 8, 8)

    output = builder.append_to_batch(images)

    assert output.shape == (2, 5, 8, 8)
    assert output.device == images.device
    torch.testing.assert_close(output[:, :3], images)
    assert torch.isfinite(output[:, 3:]).all()
