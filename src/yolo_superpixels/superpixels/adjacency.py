"""Build right/down adjacency channels from superpixel labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .algorithms import SuperpixelAlgorithm

Direction = Literal["right", "down"]
EdgeRepresentation = Literal["same", "boundary", "zero_centered", "boundary_zero_centered"]
EDGE_REPRESENTATIONS: tuple[EdgeRepresentation, ...] = (
    "same",
    "boundary",
    "zero_centered",
    "boundary_zero_centered",
)


def labels_to_adjacency(
    labels: np.ndarray,
    channel_order: Sequence[Direction] = ("right", "down"),
) -> np.ndarray:
    """Convert an ``H x W`` superpixel label map to two adjacency channels.

    ``right`` is 1 where pixel ``(y, x)`` and ``(y, x + 1)`` have the same label.
    ``down`` is 1 where pixel ``(y, x)`` and ``(y + 1, x)`` have the same label.
    Boundary pixels without a right/down neighbor are 0.
    """

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label map, got shape {labels.shape}.")

    height, width = labels.shape
    right = np.zeros((height, width), dtype=np.float32)
    down = np.zeros((height, width), dtype=np.float32)

    if width > 1:
        right[:, :-1] = labels[:, :-1] == labels[:, 1:]
    if height > 1:
        down[:-1, :] = labels[:-1, :] == labels[1:, :]

    channels = {"right": right, "down": down}
    try:
        return np.stack([channels[name] for name in channel_order], axis=0)
    except KeyError as exc:
        raise ValueError("channel_order must contain only 'right' and 'down'.") from exc


def algorithm_output_to_edges(
    output: np.ndarray,
    height: int,
    width: int,
    channel_order: Sequence[Direction] = ("right", "down"),
) -> np.ndarray:
    """Normalize algorithm output to ``2 x H x W`` float32 edge channels."""

    output = np.asarray(output)

    if output.shape == (height, width):
        return labels_to_adjacency(output, channel_order=channel_order)

    if output.ndim == 3 and output.shape[1:] == (height, width):
        if output.shape[0] == 2:
            return _order_numpy_edges(output.astype(np.float32, copy=False), channel_order)
        if output.shape[0] >= 5:
            return _order_numpy_edges(output[3:5].astype(np.float32, copy=False), channel_order)

    if output.ndim == 3 and output.shape[:2] == (height, width):
        if output.shape[-1] == 2:
            return _order_numpy_edges(output.transpose(2, 0, 1).astype(np.float32, copy=False), channel_order)
        if output.shape[-1] >= 5:
            return _order_numpy_edges(output[..., 3:5].transpose(2, 0, 1).astype(np.float32, copy=False), channel_order)

    raise ValueError(
        "Superpixel algorithm must return labels HxW, edges 2xHxW/HxWx2, "
        f"or five channels. Got shape {output.shape}; expected H={height}, W={width}."
    )


def _order_numpy_edges(edges: np.ndarray, channel_order: Sequence[Direction]) -> np.ndarray:
    order = tuple(channel_order)
    if order == ("right", "down"):
        return edges
    if order == ("down", "right"):
        return edges[[1, 0]]
    raise ValueError("channel_order must be ('right', 'down') or ('down', 'right').")


def tensor_to_uint8_hwc(image_tensor) -> np.ndarray:
    """Convert a ``CxHxW`` torch image tensor to uint8 RGB ``HxWx3``."""

    import torch

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(image_tensor)!r}.")
    if image_tensor.ndim != 3 or image_tensor.shape[0] < 3:
        raise ValueError(f"Expected a CxHxW tensor with at least 3 channels, got {image_tensor.shape}.")

    rgb = image_tensor[:3].detach().float().cpu()
    if rgb.numel() == 0:
        raise ValueError("Cannot convert an empty image tensor.")

    min_value = float(rgb.min())
    max_value = float(rgb.max())
    if min_value >= -0.5 and max_value <= 1.5:
        rgb = rgb.clamp(0.0, 1.0) * 255.0
    else:
        rgb = rgb.clamp(0.0, 255.0)

    return rgb.byte().permute(1, 2, 0).contiguous().numpy()


def tensor_labels_to_adjacency(
    labels,
    channel_order: Sequence[Direction] = ("right", "down"),
):
    """Convert ``B x H x W`` tensor labels to ``B x 2 x H x W`` adjacency channels."""

    import torch

    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(labels)!r}.")
    if labels.ndim != 3:
        raise ValueError(f"Expected BxHxW labels, got shape {tuple(labels.shape)}.")

    batch, height, width = labels.shape
    right = torch.zeros((batch, height, width), dtype=torch.float32, device=labels.device)
    down = torch.zeros((batch, height, width), dtype=torch.float32, device=labels.device)

    if width > 1:
        right[:, :, :-1] = labels[:, :, :-1] == labels[:, :, 1:]
    if height > 1:
        down[:, :-1, :] = labels[:, :-1, :] == labels[:, 1:, :]

    channels = {"right": right, "down": down}
    try:
        return torch.stack([channels[name] for name in channel_order], dim=1)
    except KeyError as exc:
        raise ValueError("channel_order must contain only 'right' and 'down'.") from exc


def tensor_algorithm_output_to_edges(
    output,
    batch_size: int,
    height: int,
    width: int,
    channel_order: Sequence[Direction] = ("right", "down"),
):
    """Normalize tensor algorithm output to ``B x 2 x H x W`` float edge channels."""

    import torch

    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)

    if output.ndim == 2 and batch_size == 1 and output.shape == (height, width):
        output = output.unsqueeze(0)

    if output.ndim == 3:
        if output.shape == (batch_size, height, width):
            return tensor_labels_to_adjacency(output, channel_order=channel_order)
        if batch_size == 1 and output.shape in {(2, height, width), (5, height, width)}:
            output = output.unsqueeze(0)
        elif batch_size == 1 and output.shape[:2] == (height, width) and output.shape[-1] in {2, 5}:
            output = output.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported tensor output shape {tuple(output.shape)}.")

    if output.ndim != 4:
        raise ValueError(f"Expected tensor output with 2, 3, or 4 dims, got {tuple(output.shape)}.")

    if output.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {output.shape[0]}.")

    if output.shape[-2:] == (height, width):
        if output.shape[1] == 2:
            edges = output
        elif output.shape[1] >= 5:
            edges = output[:, 3:5]
        else:
            raise ValueError(f"Expected 2 or at least 5 channels, got {output.shape[1]}.")
    elif output.shape[1:3] == (height, width):
        if output.shape[-1] == 2:
            edges = output.permute(0, 3, 1, 2).contiguous()
        elif output.shape[-1] >= 5:
            edges = output[..., 3:5].permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Expected 2 or at least 5 channels, got {output.shape[-1]}.")
    else:
        raise ValueError(
            "Tensor algorithm must return labels BxHxW, edges Bx2xHxW/BxHxWx2, "
            f"or five channels. Got shape {tuple(output.shape)}; expected B={batch_size}, "
            f"H={height}, W={width}."
        )

    if tuple(channel_order) == ("right", "down"):
        return edges.float()
    if tuple(channel_order) == ("down", "right"):
        return edges[:, [1, 0]].float()
    raise ValueError("channel_order must be ('right', 'down') or ('down', 'right').")


def _validate_representation(representation: str) -> EdgeRepresentation:
    if representation not in EDGE_REPRESENTATIONS:
        allowed = ", ".join(EDGE_REPRESENTATIONS)
        raise ValueError(f"edge_representation must be one of: {allowed}.")
    return representation  # type: ignore[return-value]


def numpy_valid_edge_mask(
    height: int,
    width: int,
    channel_order: Sequence[Direction] = ("right", "down"),
) -> np.ndarray:
    """Return ``2 x H x W`` validity mask for right/down neighbor comparisons."""

    right = np.zeros((height, width), dtype=np.float32)
    down = np.zeros((height, width), dtype=np.float32)
    if width > 1:
        right[:, :-1] = 1.0
    if height > 1:
        down[:-1, :] = 1.0
    channels = {"right": right, "down": down}
    try:
        return np.stack([channels[name] for name in channel_order], axis=0)
    except KeyError as exc:
        raise ValueError("channel_order must contain only 'right' and 'down'.") from exc


def tensor_valid_edge_mask(
    batch_size: int,
    height: int,
    width: int,
    device,
    dtype,
    channel_order: Sequence[Direction] = ("right", "down"),
):
    """Return ``B x 2 x H x W`` validity mask for right/down neighbor comparisons."""

    import torch

    right = torch.zeros((batch_size, height, width), dtype=dtype, device=device)
    down = torch.zeros((batch_size, height, width), dtype=dtype, device=device)
    if width > 1:
        right[:, :, :-1] = 1.0
    if height > 1:
        down[:, :-1, :] = 1.0
    channels = {"right": right, "down": down}
    try:
        return torch.stack([channels[name] for name in channel_order], dim=1)
    except KeyError as exc:
        raise ValueError("channel_order must contain only 'right' and 'down'.") from exc


def represent_numpy_edges(
    edges: np.ndarray,
    representation: EdgeRepresentation = "same",
    channel_order: Sequence[Direction] = ("right", "down"),
) -> np.ndarray:
    """Convert same-neighbor edges to the requested representation."""

    representation = _validate_representation(representation)
    edges = edges.astype(np.float32, copy=False)
    if representation == "same":
        return edges

    _, height, width = edges.shape
    mask = numpy_valid_edge_mask(height, width, channel_order=channel_order)
    if representation == "boundary":
        return (1.0 - edges) * mask
    if representation == "zero_centered":
        return (2.0 * edges - 1.0) * mask
    if representation == "boundary_zero_centered":
        return (1.0 - 2.0 * edges) * mask
    raise AssertionError("unreachable")


def represent_tensor_edges(
    edges,
    representation: EdgeRepresentation = "same",
    channel_order: Sequence[Direction] = ("right", "down"),
):
    """Convert same-neighbor tensor edges to the requested representation."""

    representation = _validate_representation(representation)
    edges = edges.float()
    if representation == "same":
        return edges

    batch_size, _, height, width = edges.shape
    mask = tensor_valid_edge_mask(
        batch_size=batch_size,
        height=height,
        width=width,
        device=edges.device,
        dtype=edges.dtype,
        channel_order=channel_order,
    )
    if representation == "boundary":
        return (1.0 - edges) * mask
    if representation == "zero_centered":
        return (2.0 * edges - 1.0) * mask
    if representation == "boundary_zero_centered":
        return (1.0 - 2.0 * edges) * mask
    raise AssertionError("unreachable")


@dataclass
class AdjacencyChannelBuilder:
    """Append superpixel adjacency channels to a batch of RGB tensors.

    One algorithm contributes two channels: right and down adjacency/affinity.
    Passing multiple algorithms builds ``RGB + 2 * N`` channels.
    """

    algorithm: SuperpixelAlgorithm | Sequence[SuperpixelAlgorithm] | None
    channel_order: tuple[Direction, Direction] = ("right", "down")
    edge_representation: EdgeRepresentation = "zero_centered"
    edge_value_scale: float = 1.0
    fail_open: bool = False
    enabled: bool = True
    disabled_value: float = 0.0

    def __post_init__(self) -> None:
        self.edge_representation = _validate_representation(self.edge_representation)

    @property
    def algorithms(self) -> tuple[SuperpixelAlgorithm, ...]:
        if self.algorithm is None:
            return ()
        if isinstance(self.algorithm, (list, tuple)):
            return tuple(self.algorithm)
        return (self.algorithm,)

    @property
    def extra_channels(self) -> int:
        return 2 * len(self.algorithms)

    def disabled_edges_for_batch(self, batch_tensor):
        """Return disabled extra channels without running the superpixel algorithm."""

        import torch

        if batch_tensor.ndim != 4:
            raise ValueError(f"Expected a BCHW tensor, got shape {batch_tensor.shape}.")

        batch_size, _, height, width = batch_tensor.shape
        return torch.full(
            (batch_size, self.extra_channels, height, width),
            fill_value=self.disabled_value,
            dtype=batch_tensor.dtype,
            device=batch_tensor.device,
        )

    def _edges_for_tensor_single(self, algorithm: SuperpixelAlgorithm, image_tensor):
        """Build a ``2 x H x W`` torch tensor for one ``3 x H x W`` image."""

        import torch

        height, width = image_tensor.shape[-2:]
        if not torch.any(image_tensor[:3].detach()):
            return torch.zeros((2, height, width), dtype=image_tensor.dtype, device=image_tensor.device)

        try:
            rgb_image = tensor_to_uint8_hwc(image_tensor)
            output = algorithm(rgb_image)
            edge_array = algorithm_output_to_edges(
                output,
                height=height,
                width=width,
                channel_order=self.channel_order,
            )
        except Exception:
            if not self.fail_open:
                raise
            return torch.zeros((2, height, width), dtype=image_tensor.dtype, device=image_tensor.device)

        edges = torch.as_tensor(edge_array, dtype=image_tensor.dtype, device=image_tensor.device)
        edges = self.transform_edges(edges.unsqueeze(0))[0].to(dtype=image_tensor.dtype)
        return edges

    def edges_for_tensor(self, image_tensor):
        """Build a ``2N x H x W`` torch tensor for one ``3 x H x W`` image."""

        import torch

        height, width = image_tensor.shape[-2:]
        if not self.enabled:
            return torch.full(
                (self.extra_channels, height, width),
                fill_value=self.disabled_value,
                dtype=image_tensor.dtype,
                device=image_tensor.device,
            )
        if not self.algorithms:
            return torch.empty((0, height, width), dtype=image_tensor.dtype, device=image_tensor.device)
        return torch.cat(
            [self._edges_for_tensor_single(algorithm, image_tensor) for algorithm in self.algorithms],
            dim=0,
        )

    def _edges_for_batch_single(self, algorithm: SuperpixelAlgorithm, batch_tensor):
        """Build ``B x 2 x H x W`` edge channels for one algorithm."""

        import torch

        batch_size, _, height, width = batch_tensor.shape
        tensor_builder = getattr(algorithm, "edges_for_batch", None)
        if tensor_builder is not None:
            try:
                output = tensor_builder(batch_tensor[:, :3], channel_order=self.channel_order)
                edges = tensor_algorithm_output_to_edges(
                    output,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    channel_order=("right", "down"),
                )
            except Exception:
                if not self.fail_open:
                    raise
                return torch.zeros(
                    (batch_size, 2, height, width),
                    dtype=batch_tensor.dtype,
                    device=batch_tensor.device,
                )

            edges = edges.to(dtype=batch_tensor.dtype, device=batch_tensor.device, non_blocking=True)
            return self.transform_edges(edges)
        else:
            return torch.stack(
                [self._edges_for_tensor_single(algorithm, image) for image in batch_tensor],
                dim=0,
            )

    def transform_edges(self, edges):
        """Apply representation and scale to ``B x 2 x H x W`` same-neighbor channels."""

        transformed = represent_tensor_edges(
            edges,
            representation=self.edge_representation,
            channel_order=self.channel_order,
        )
        if self.edge_value_scale != 1.0:
            transformed = transformed * self.edge_value_scale
        return transformed

    def edges_for_batch(self, batch_tensor):
        """Build ``B x 2N x H x W`` edge channels, using GPU paths when available."""

        import torch

        if batch_tensor.ndim != 4:
            raise ValueError(f"Expected a BCHW tensor, got shape {batch_tensor.shape}.")

        if not self.enabled:
            return self.disabled_edges_for_batch(batch_tensor)

        batch_size, _, height, width = batch_tensor.shape
        if not self.algorithms:
            return torch.empty((batch_size, 0, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)

        return torch.cat(
            [self._edges_for_batch_single(algorithm, batch_tensor) for algorithm in self.algorithms],
            dim=1,
        )

    def append_to_batch(self, batch_tensor):
        """Return ``B x (3 + 2N) x H x W`` from a ``B x 3 x H x W`` tensor.

        If the tensor already has the expected channel count, it is returned
        unchanged. This makes repeated patching and manual preprocessing safe.
        """

        import torch

        if batch_tensor.ndim != 4:
            raise ValueError(f"Expected a BCHW tensor, got shape {batch_tensor.shape}.")
        expected_channels = 3 + self.extra_channels
        if batch_tensor.shape[1] == expected_channels:
            if not self.enabled:
                output = batch_tensor.clone()
                output[:, 3:expected_channels] = self.disabled_value
                return output
            return batch_tensor
        if batch_tensor.shape[1] != 3:
            raise ValueError(
                "AdjacencyChannelBuilder expects 3-channel RGB input or already built "
                f"{expected_channels}-channel input, got {batch_tensor.shape[1]} channels."
            )

        edge_batch = self.edges_for_batch(batch_tensor)
        return torch.cat([batch_tensor, edge_batch], dim=1)
