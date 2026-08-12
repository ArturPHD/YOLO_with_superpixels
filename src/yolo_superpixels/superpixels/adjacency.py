"""Build right/down adjacency channels from superpixel labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

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
DistanceChannelMode = Literal["soft_interior", "soft_boundary", "soft_boundary_interior"]
DISTANCE_CHANNEL_MODES: tuple[DistanceChannelMode, ...] = (
    "soft_interior",
    "soft_boundary",
    "soft_boundary_interior",
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


def algorithm_output_to_labels(output: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize a CPU algorithm output to an ``H x W`` label map.

    Distance-to-boundary maps need actual superpixel labels. Edge/affinity-only
    outputs cannot be converted losslessly, so they are rejected here.
    """

    output = np.asarray(output)
    if output.shape == (height, width):
        return output.astype(np.int64, copy=False)
    if output.ndim == 3 and output.shape == (height, width, 1):
        return output[..., 0].astype(np.int64, copy=False)
    if output.ndim == 3 and output.shape == (1, height, width):
        return output[0].astype(np.int64, copy=False)
    raise ValueError(
        "Soft distance channels require a superpixel label map shaped HxW. "
        f"Got shape {output.shape}; expected H={height}, W={width}."
    )


def tensor_algorithm_output_to_labels(output, batch_size: int, height: int, width: int):
    """Normalize a tensor algorithm output to ``B x H x W`` labels."""

    import torch

    candidates: list[Any]
    if isinstance(output, dict):
        candidates = [
            output.get(key)
            for key in ("seg", "segments", "labels", "label", "superpixels")
        ]
    elif isinstance(output, (list, tuple)):
        candidates = list(output)
    else:
        candidates = [output]

    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate)
        labels = candidate
        if labels.ndim == 2 and batch_size == 1 and labels.shape == (height, width):
            return labels.unsqueeze(0).long()
        if labels.ndim == 3 and labels.shape == (batch_size, height, width):
            return labels.long()
        if labels.ndim == 4 and labels.shape == (batch_size, 1, height, width):
            return labels[:, 0].long()
        if labels.ndim == 4 and labels.shape == (batch_size, height, width, 1):
            return labels[..., 0].long()

    shapes = [
        tuple(candidate.shape)
        for candidate in candidates
        if hasattr(candidate, "shape")
    ]
    raise ValueError(
        "Soft distance channels require superpixel labels shaped BxHxW. "
        f"Got candidate shapes: {shapes}; expected B={batch_size}, H={height}, W={width}."
    )


def _validate_representation(representation: str) -> EdgeRepresentation:
    if representation not in EDGE_REPRESENTATIONS:
        allowed = ", ".join(EDGE_REPRESENTATIONS)
        raise ValueError(f"edge_representation must be one of: {allowed}.")
    return representation  # type: ignore[return-value]


def _validate_distance_mode(mode: str) -> DistanceChannelMode:
    if mode not in DISTANCE_CHANNEL_MODES:
        allowed = ", ".join(DISTANCE_CHANNEL_MODES)
        raise ValueError(f"distance_channel_mode must be one of: {allowed}.")
    return mode  # type: ignore[return-value]


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


def distance_mode_extra_channels(mode: DistanceChannelMode) -> int:
    """Return how many channels one algorithm contributes for a distance mode."""

    mode = _validate_distance_mode(mode)
    return 2 if mode == "soft_boundary_interior" else 1


def tensor_labels_to_boundary_mask(labels):
    """Return a boolean mask of pixels touching a superpixel/image boundary."""

    import torch

    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(labels)!r}.")
    if labels.ndim != 3:
        raise ValueError(f"Expected BxHxW labels, got shape {tuple(labels.shape)}.")

    batch, height, width = labels.shape
    boundary = torch.zeros((batch, height, width), dtype=torch.bool, device=labels.device)

    if width > 1:
        diff = labels[:, :, 1:] != labels[:, :, :-1]
        boundary[:, :, 1:] |= diff
        boundary[:, :, :-1] |= diff
    if height > 1:
        diff = labels[:, 1:, :] != labels[:, :-1, :]
        boundary[:, 1:, :] |= diff
        boundary[:, :-1, :] |= diff

    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    return boundary


def tensor_boundary_mask_to_distance(boundary_mask, max_distance: int = 32):
    """Approximate distance to the nearest boundary with iterative GPU erosion."""

    import torch
    import torch.nn.functional as F

    if max_distance <= 0:
        raise ValueError("max_distance must be positive.")
    if boundary_mask.ndim != 3:
        raise ValueError(f"Expected BxHxW boundary mask, got shape {tuple(boundary_mask.shape)}.")

    active = (~boundary_mask.bool()).to(dtype=torch.float32).unsqueeze(1)
    distance = torch.zeros_like(active)
    for _ in range(int(max_distance)):
        distance = distance + active
        active = -F.max_pool2d(-active, kernel_size=3, stride=1, padding=1)
        active = (active > 0.5).to(dtype=distance.dtype)
    return distance[:, 0]


def tensor_labels_to_boundary_distance(labels, max_distance: int = 32):
    """Approximate distance to the nearest label boundary with iterative GPU erosion.

    The distance is measured in Chebyshev-like pixel steps. Boundary pixels are
    0, pixels one erosion away are 1, and central pixels survive for more steps.
    """

    import torch

    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(labels)!r}.")
    if labels.ndim != 3:
        raise ValueError(f"Expected BxHxW labels, got shape {tuple(labels.shape)}.")

    return tensor_boundary_mask_to_distance(
        tensor_labels_to_boundary_mask(labels),
        max_distance=max_distance,
    )


def tensor_edges_to_boundary_mask(edges, threshold: float = 0.5):
    """Return a boundary mask from right/down same-neighbor affinity channels."""

    import torch

    if not isinstance(edges, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(edges)!r}.")
    if edges.ndim != 4 or edges.shape[1] != 2:
        raise ValueError(f"Expected Bx2xHxW edges, got shape {tuple(edges.shape)}.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    batch, _, height, width = edges.shape
    boundary = torch.zeros((batch, height, width), dtype=torch.bool, device=edges.device)
    right_same = edges[:, 0].float()
    down_same = edges[:, 1].float()

    if width > 1:
        diff = right_same[:, :, :-1] <= float(threshold)
        boundary[:, :, :-1] |= diff
        boundary[:, :, 1:] |= diff
    if height > 1:
        diff = down_same[:, :-1, :] <= float(threshold)
        boundary[:, :-1, :] |= diff
        boundary[:, 1:, :] |= diff

    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    return boundary


def _normalize_distance_per_label(distance, labels):
    """Normalize distances so each superpixel interior peaks at 1."""

    import torch

    normalized = torch.zeros_like(distance)
    for batch_index in range(int(labels.shape[0])):
        labels_flat = labels[batch_index].reshape(-1).long()
        distance_flat = distance[batch_index].reshape(-1)
        unique, inverse = torch.unique(labels_flat, sorted=False, return_inverse=True)
        max_per_label = torch.zeros(
            int(unique.numel()),
            dtype=distance_flat.dtype,
            device=distance_flat.device,
        )
        max_per_label.scatter_reduce_(
            0,
            inverse,
            distance_flat,
            reduce="amax",
            include_self=True,
        )
        denom = max_per_label.gather(0, inverse).clamp_min(1.0)
        normalized[batch_index] = (distance_flat / denom).reshape_as(distance[batch_index])
    return normalized


def tensor_labels_to_soft_distance_channels(
    labels,
    mode: DistanceChannelMode = "soft_interior",
    max_distance: int = 32,
    boundary_width: float = 6.0,
    dtype=None,
):
    """Convert labels to one or two smooth boundary/interior channels.

    ``soft_interior`` is normalized per superpixel: centers approach 1 and
    boundaries are 0. ``soft_boundary`` is a fixed-width boundary band: boundary
    pixels are 1 and the signal fades to 0 after ``boundary_width`` pixels.
    """

    import torch

    mode = _validate_distance_mode(mode)
    labels = labels.long()
    distance = tensor_labels_to_boundary_distance(labels, max_distance=max_distance)
    target_dtype = dtype or torch.float32
    interior = _normalize_distance_per_label(distance, labels).to(dtype=target_dtype)
    boundary = (1.0 - distance / max(float(boundary_width), 1e-6)).clamp(0.0, 1.0)
    boundary = boundary.to(dtype=target_dtype)

    if mode == "soft_interior":
        return interior.unsqueeze(1)
    if mode == "soft_boundary":
        return boundary.unsqueeze(1)
    if mode == "soft_boundary_interior":
        return torch.stack((boundary, interior), dim=1)
    raise AssertionError("unreachable")


def tensor_edges_to_soft_distance_channels(
    edges,
    mode: DistanceChannelMode = "soft_interior",
    max_distance: int = 32,
    boundary_width: float = 6.0,
    threshold: float = 0.5,
    dtype=None,
):
    """Convert right/down affinity channels to soft boundary/interior maps.

    This path is for algorithms such as ``color_affinity`` that provide local
    neighbor similarity but not a discrete superpixel label map. Interior is
    normalized by ``max_distance`` rather than per superpixel, because no stable
    component labels are available.
    """

    import torch

    mode = _validate_distance_mode(mode)
    boundary_mask = tensor_edges_to_boundary_mask(edges, threshold=threshold)
    distance = tensor_boundary_mask_to_distance(boundary_mask, max_distance=max_distance)
    target_dtype = dtype or torch.float32
    interior = (distance / float(max_distance)).clamp(0.0, 1.0).to(dtype=target_dtype)
    boundary = (1.0 - distance / max(float(boundary_width), 1e-6)).clamp(0.0, 1.0)
    boundary = boundary.to(dtype=target_dtype)

    if mode == "soft_interior":
        return interior.unsqueeze(1)
    if mode == "soft_boundary":
        return boundary.unsqueeze(1)
    if mode == "soft_boundary_interior":
        return torch.stack((boundary, interior), dim=1)
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


@dataclass
class DistanceChannelBuilder:
    """Append soft distance-to-boundary channels to RGB tensors.

    One algorithm contributes either one channel (``soft_interior`` or
    ``soft_boundary``) or two channels (``soft_boundary_interior``). Label
    algorithms use true superpixel regions. Affinity algorithms can also be used:
    their low right/down affinity values are treated as a soft boundary source.
    """

    algorithm: SuperpixelAlgorithm | Sequence[SuperpixelAlgorithm] | None
    mode: DistanceChannelMode = "soft_interior"
    max_distance: int = 32
    boundary_width: float = 6.0
    affinity_threshold: float = 0.5
    fail_open: bool = False
    enabled: bool = True
    disabled_value: float = 0.0

    def __post_init__(self) -> None:
        self.mode = _validate_distance_mode(self.mode)
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        if self.boundary_width <= 0:
            raise ValueError("boundary_width must be positive.")
        if not 0.0 <= self.affinity_threshold <= 1.0:
            raise ValueError("affinity_threshold must be in [0, 1].")

    @property
    def algorithms(self) -> tuple[SuperpixelAlgorithm, ...]:
        if self.algorithm is None:
            return ()
        if isinstance(self.algorithm, (list, tuple)):
            return tuple(self.algorithm)
        return (self.algorithm,)

    @property
    def channels_per_algorithm(self) -> int:
        return distance_mode_extra_channels(self.mode)

    @property
    def extra_channels(self) -> int:
        return self.channels_per_algorithm * len(self.algorithms)

    def disabled_channels_for_batch(self, batch_tensor):
        """Return disabled soft channels without running the superpixel algorithm."""

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

    def _channels_for_tensor_single(self, algorithm: SuperpixelAlgorithm, image_tensor):
        """Build soft channels for one ``3 x H x W`` image."""

        import torch

        height, width = image_tensor.shape[-2:]
        if not torch.any(image_tensor[:3].detach()):
            return torch.zeros(
                (self.channels_per_algorithm, height, width),
                dtype=image_tensor.dtype,
                device=image_tensor.device,
            )

        try:
            rgb_image = tensor_to_uint8_hwc(image_tensor)
            output = algorithm(rgb_image)
            try:
                label_array = algorithm_output_to_labels(output, height=height, width=width)
                labels = torch.as_tensor(label_array, dtype=torch.long, device=image_tensor.device)
                return self._channels_from_labels(labels.unsqueeze(0), dtype=image_tensor.dtype)[0]
            except ValueError:
                edge_array = algorithm_output_to_edges(output, height=height, width=width)
                edges = torch.as_tensor(edge_array, dtype=image_tensor.dtype, device=image_tensor.device)
                return self._channels_from_edges(edges.unsqueeze(0), dtype=image_tensor.dtype)[0]
        except Exception:
            if not self.fail_open:
                raise
            return torch.zeros(
                (self.channels_per_algorithm, height, width),
                dtype=image_tensor.dtype,
                device=image_tensor.device,
            )

    def _channels_from_labels(self, labels, dtype):
        return tensor_labels_to_soft_distance_channels(
            labels,
            mode=self.mode,
            max_distance=self.max_distance,
            boundary_width=self.boundary_width,
            dtype=dtype,
        )

    def _channels_from_edges(self, edges, dtype):
        return tensor_edges_to_soft_distance_channels(
            edges,
            mode=self.mode,
            max_distance=self.max_distance,
            boundary_width=self.boundary_width,
            threshold=self.affinity_threshold,
            dtype=dtype,
        )

    def channels_for_tensor(self, image_tensor):
        """Build a ``CN x H x W`` tensor for one ``3 x H x W`` image."""

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
            [
                self._channels_for_tensor_single(algorithm, image_tensor)
                for algorithm in self.algorithms
            ],
            dim=0,
        )

    def _channels_for_batch_single(self, algorithm: SuperpixelAlgorithm, batch_tensor):
        """Build ``B x CN x H x W`` soft channels for one algorithm."""

        import torch

        batch_size, _, height, width = batch_tensor.shape
        label_builder = getattr(algorithm, "labels_for_batch", None)
        if label_builder is not None:
            try:
                output = label_builder(batch_tensor[:, :3])
                labels = tensor_algorithm_output_to_labels(
                    output,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                )
            except Exception:
                if not self.fail_open:
                    raise
                return torch.zeros(
                    (batch_size, self.channels_per_algorithm, height, width),
                    dtype=batch_tensor.dtype,
                    device=batch_tensor.device,
                )
            labels = labels.to(device=batch_tensor.device, non_blocking=True).long()
            return self._channels_from_labels(labels, dtype=batch_tensor.dtype)

        tensor_builder = getattr(algorithm, "edges_for_batch", None)
        if tensor_builder is not None:
            try:
                output = tensor_builder(batch_tensor[:, :3], channel_order=("right", "down"))
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
                    (batch_size, self.channels_per_algorithm, height, width),
                    dtype=batch_tensor.dtype,
                    device=batch_tensor.device,
                )
            edges = edges.to(dtype=batch_tensor.dtype, device=batch_tensor.device, non_blocking=True)
            return self._channels_from_edges(edges, dtype=batch_tensor.dtype)

        return torch.stack(
            [self._channels_for_tensor_single(algorithm, image) for image in batch_tensor],
            dim=0,
        )

    def channels_for_batch(self, batch_tensor):
        """Build ``B x CN x H x W`` soft channels, using GPU labels when available."""

        import torch

        if batch_tensor.ndim != 4:
            raise ValueError(f"Expected a BCHW tensor, got shape {batch_tensor.shape}.")

        if not self.enabled:
            return self.disabled_channels_for_batch(batch_tensor)

        batch_size, _, height, width = batch_tensor.shape
        if not self.algorithms:
            return torch.empty((batch_size, 0, height, width), dtype=batch_tensor.dtype, device=batch_tensor.device)

        return torch.cat(
            [
                self._channels_for_batch_single(algorithm, batch_tensor)
                for algorithm in self.algorithms
            ],
            dim=1,
        )

    def append_to_batch(self, batch_tensor):
        """Return ``B x (3 + CN) x H x W`` from a ``B x 3 x H x W`` tensor."""

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
                "DistanceChannelBuilder expects 3-channel RGB input or already built "
                f"{expected_channels}-channel input, got {batch_tensor.shape[1]} channels."
            )

        channel_batch = self.channels_for_batch(batch_tensor)
        return torch.cat([batch_tensor, channel_batch], dim=1)
