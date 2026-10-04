import torch
import torch.nn.functional as F
from torch import Tensor


_SAFE_MASK = torch.zeros(16, dtype=torch.bool)
for m in range(16):
    adj = {(0,2),(0,3),(1,2),(1,3),(2,0),(2,1),(3,0),(3,1)}
    bits = [i for i in range(4) if m & (1 << i)]
    if len(bits) <= 1:
        _SAFE_MASK[m] = True
        continue
    visited = {bits[0]}
    stack = [bits[0]]
    while stack:
        cur = stack.pop()
        for nb in bits:
            if nb not in visited and (cur, nb) in adj:
                visited.add(nb)
                stack.append(nb)
    _SAFE_MASK[m] = len(visited) == len(bits)

def _flat_image_coords(
    B: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )
    return torch.stack([yy, xx], dim=-1).reshape(1, H * W, 2).expand(B, -1, -1).reshape(B * H * W, 2)


def region_stats(
    fimg: Tensor,
    seg: Tensor,
    nV: int,
    coords: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    C = fimg.shape[-1]
    flat_seg = seg.reshape(-1)

    sums = fimg.new_zeros(nV, C)
    xy_sums = fimg.new_zeros(nV, 2)

    sums.scatter_add_(0, flat_seg[:, None].expand(-1, C), fimg)
    xy_sums.scatter_add_(0, flat_seg[:, None].expand(-1, 2), coords.to(fimg.dtype))
    counts = flat_seg.bincount(minlength=nV)

    denom = counts.clamp_min(1).to(fimg.dtype)
    return sums / denom[:, None], xy_sums / denom[:, None], counts


def cheap_boundary_relax(
    fimg: Tensor,
    seg: Tensor,
    means: Tensor,
    centers: Tensor,
    sizes: Tensor,
    coords: Tensor,
    safe_mask: Tensor,
    compactness: float = 0.1,
    margin: float = 0.0,
) -> Tensor:
    _, H, W = seg.shape
    dtype = fimg.dtype

    seg_pad = F.pad(seg, (1, 1, 1, 1), mode='replicate')
    n_up = seg_pad[:, 0:H,     1:W + 1]
    n_dn = seg_pad[:, 2:H + 2, 1:W + 1]
    n_lt = seg_pad[:, 1:H + 1, 0:W]
    n_rt = seg_pad[:, 1:H + 1, 2:W + 2]

    boundary = (seg != n_up) | (seg != n_dn) | (seg != n_lt) | (seg != n_rt)
    if not boundary.any():
        return seg

    b_idx, y_idx, x_idx = boundary.nonzero(as_tuple=True)
    p_idx = b_idx * H * W + y_idx * W + x_idx

    cand = torch.stack([
        seg[b_idx, y_idx, x_idx],
        n_up[b_idx, y_idx, x_idx],
        n_dn[b_idx, y_idx, x_idx],
        n_lt[b_idx, y_idx, x_idx],
        n_rt[b_idx, y_idx, x_idx],
    ], dim=0)  # (5, M) — global labels

    cand_means   = means[cand]                           # (5, M, C)
    cand_centers = centers[cand]                         # (5, M, 2)
    cand_sizes   = sizes[cand].clamp_min(1).to(dtype)   # (5, M)

    pix_feat   = fimg[p_idx].unsqueeze(0)                # (1, M, C)
    pix_coords = coords[p_idx].unsqueeze(0).to(dtype)   # (1, M, 2)

    color_dist   = (pix_feat - cand_means).pow(2).sum(-1)
    spatial_dist = (pix_coords - cand_centers).pow(2).sum(-1) / cand_sizes
    dist = color_dist + compactness * spatial_dist

    best_dist, best_idx = dist.min(0)
    accept = (dist[0] - best_dist) > margin

    neighbor_mask = (
          (cand[1] == cand[0]).long()
        | ((cand[2] == cand[0]).long() << 1)
        | ((cand[3] == cand[0]).long() << 2)
        | ((cand[4] == cand[0]).long() << 3)
    )
    accept = accept & safe_mask[neighbor_mask]

    if not accept.any():
        return seg

    accept_idx = accept.nonzero(as_tuple=False).flatten()
    new_seg = seg.clone()
    new_seg[
        b_idx[accept_idx],
        y_idx[accept_idx],
        x_idx[accept_idx],
    ] = cand[best_idx[accept_idx], accept_idx]

    return new_seg


def cheap_relax_loop(
    fimg: Tensor,
    seg: Tensor,
    nV: int,
    num_iters: int = 5,
    compactness: float = 0.5,
    margin: float = 0.1,
) -> Tensor:
    if seg.ndim != 3:
        raise ValueError(f'Expected seg with shape (B, H, W), got {tuple(seg.shape)}.')

    B, H, W = seg.shape

    if nV is None:
        nV = int(seg.max().item()) + 1

    coords = _flat_image_coords(B, H, W, fimg.device, fimg.dtype)
    safe_mask = _SAFE_MASK.to(seg.device)

    for _ in range(num_iters):
        means, centers, sizes = region_stats(fimg, seg, nV, coords)
        seg = cheap_boundary_relax(
            fimg=fimg,
            seg=seg,
            means=means,
            centers=centers,
            sizes=sizes,
            coords=coords,
            safe_mask=safe_mask,
            compactness=compactness,
            margin=margin,
        )

    return seg
