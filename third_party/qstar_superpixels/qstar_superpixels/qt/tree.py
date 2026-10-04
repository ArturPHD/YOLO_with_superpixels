import torch


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from typing import NamedTuple


from .solvers import tree_budget_solve_bisect


from .tools import bbox_to_segmap


def get_integral_image(imgs: Tensor) -> tuple[Tensor, Tensor]:
    pimgs = F.pad(imgs, (1,0,1,0))
    ii_sm = pimgs.cumsum(-1).cumsum(-2)
    ii_sq = pimgs.pow(2).cumsum(-1).cumsum(-2)
    return ii_sm, ii_sq


def _box_sse_per_image(
    ii_sm: Tensor,
    ii_sq: Tensor,
    b_idx: Tensor,
    ymin: Tensor,
    xmin: Tensor,
    ymax: Tensor,
    xmax: Tensor,
) -> Tensor:
    _, C, H1, W1 = ii_sm.shape
    stride = H1 * W1

    # _find_adaptive_splits scores every candidate before applying its valid
    # candidate mask. Candidates for smaller boxes may temporarily extend past
    # the integral-image boundary; clamp those temporary queries so they can
    # be safely discarded by that mask instead of raising an IndexError.
    ymin = ymin.clamp(0, H1 - 1)
    ymax = ymax.clamp(0, H1 - 1)
    xmin = xmin.clamp(0, W1 - 1)
    xmax = xmax.clamp(0, W1 - 1)

    def query(ii):
        # Flatten indices.
        ii_flat = ii.permute(0, 2, 3, 1).reshape(-1, C)  # contiguous copy, (B*H1*W1, C)
        def _idx(b, y, x):
            return b * stride + y * W1 + x
        return (
            ii_flat[_idx(b_idx, ymax, xmax)]
            - ii_flat[_idx(b_idx, ymin, xmax)]
            - ii_flat[_idx(b_idx, ymax, xmin)]
            + ii_flat[_idx(b_idx, ymin, xmin)]
        )  # (M, C)

    s  = query(ii_sm)
    sq = query(ii_sq)
    n  = ((ymax - ymin) * (xmax - xmin)).clamp(min=1).to(s.dtype).unsqueeze(-1)  # (M, 1)
    return (sq - s.pow(2) / n).clamp(min=0).sum(dim=-1)  # (M,)


def _find_adaptive_splits(
    ii_sm: Tensor,
    ii_sq: Tensor,
    ymin: Tensor,
    xmin: Tensor,
    ymax: Tensor,
    xmax: Tensor,
    margin: int,
    min_child: int,
    alpha: float = 0.15,
    lambda_balance: float = 0.05,
) -> tuple[Tensor, Tensor]:
    B, N = ymin.shape
    device = ymin.device

    max_h: int = int((ymax - ymin).max().item())
    max_w: int = int((xmax - xmin).max().item())

    b_base = torch.arange(B, device=device).view(B, 1)

    Ky = max_h - 2 * margin + 1
    H1 = ii_sm.shape[2]

    if Ky <= 0:
        y_split = (ymin + ymax) // 2
    else:
        dy = torch.arange(margin, max_h - margin + 1, device=device)
        ys = ymin.unsqueeze(-1) + dy.view(1, 1, Ky)

        h = ymax - ymin
        mid_y = (ymin + ymax) // 2

        rad_y = (alpha * h.float()).ceil().long()
        rad_y = rad_y.clamp_min(min_child)

        lo_y = torch.maximum(ymin + min_child, mid_y - rad_y)
        hi_y = torch.minimum(ymax - min_child, mid_y + rad_y)

        valy = (ys >= lo_y.unsqueeze(-1)) & (ys <= hi_y.unsqueeze(-1))

        b_y = b_base.unsqueeze(-1).expand(B, N, Ky).reshape(-1)

        ys_f = ys.clamp(0, H1 - 1).reshape(-1)
        ym_f = ymin.unsqueeze(-1).expand(B, N, Ky).reshape(-1)
        yM_f = ymax.unsqueeze(-1).expand(B, N, Ky).reshape(-1)
        xm_f = xmin.unsqueeze(-1).expand(B, N, Ky).reshape(-1)
        xM_f = xmax.unsqueeze(-1).expand(B, N, Ky).reshape(-1)

        sse_y_top = _box_sse_per_image(
            ii_sm, ii_sq,
            b_y,
            ym_f, xm_f,
            ys_f, xM_f,
        )
        sse_y_bot = _box_sse_per_image(
            ii_sm, ii_sq,
            b_y,
            ys_f, xm_f,
            yM_f, xM_f,
        )

        total_y = (sse_y_top + sse_y_bot).reshape(B, N, Ky)

        h_f = h.float().clamp_min(1)
        pen_y = lambda_balance * (
            (ys.float() - mid_y.float().unsqueeze(-1)) / h_f.unsqueeze(-1)
        ).pow(2)

        total_y = total_y + pen_y
        total_y = total_y.masked_fill(~valy, float("inf"))

        has_valid_y = valy.any(dim=-1)
        best_y = ymin + dy[total_y.argmin(dim=-1)]

        y_split = torch.where(has_valid_y, best_y, mid_y)
        y_split = torch.maximum(
            ymin + min_child,
            torch.minimum(ymax - min_child, y_split),
        )

    Kx = max_w - 2 * margin + 1
    W1 = ii_sm.shape[3]

    if Kx <= 0:
        x_split = (xmin + xmax) // 2
    else:
        dx = torch.arange(margin, max_w - margin + 1, device=device)
        xs = xmin.unsqueeze(-1) + dx.view(1, 1, Kx)

        w = xmax - xmin
        mid_x = (xmin + xmax) // 2

        rad_x = (alpha * w.float()).ceil().long()
        rad_x = rad_x.clamp_min(min_child)

        lo_x = torch.maximum(xmin + min_child, mid_x - rad_x)
        hi_x = torch.minimum(xmax - min_child, mid_x + rad_x)

        valx = (xs >= lo_x.unsqueeze(-1)) & (xs <= hi_x.unsqueeze(-1))

        b_x = b_base.unsqueeze(-1).expand(B, N, Kx).reshape(-1)

        xs_f = xs.clamp(0, W1 - 1).reshape(-1)
        ym_f = ymin.unsqueeze(-1).expand(B, N, Kx).reshape(-1)
        yM_f = ymax.unsqueeze(-1).expand(B, N, Kx).reshape(-1)
        xm_f = xmin.unsqueeze(-1).expand(B, N, Kx).reshape(-1)
        xM_f = xmax.unsqueeze(-1).expand(B, N, Kx).reshape(-1)

        sse_x_lft = _box_sse_per_image(
            ii_sm, ii_sq,
            b_x,
            ym_f, xm_f,
            yM_f, xs_f,
        )
        sse_x_rgt = _box_sse_per_image(
            ii_sm, ii_sq,
            b_x,
            ym_f, xs_f,
            yM_f, xM_f,
        )

        total_x = (sse_x_lft + sse_x_rgt).reshape(B, N, Kx)

        w_f = w.float().clamp_min(1)
        pen_x = lambda_balance * (
            (xs.float() - mid_x.float().unsqueeze(-1)) / w_f.unsqueeze(-1)
        ).pow(2)

        total_x = total_x + pen_x
        total_x = total_x.masked_fill(~valx, float("inf"))

        has_valid_x = valx.any(dim=-1)
        best_x = xmin + dx[total_x.argmin(dim=-1)]

        x_split = torch.where(has_valid_x, best_x, mid_x)
        x_split = torch.maximum(
            xmin + min_child,
            torch.minimum(xmax - min_child, x_split),
        )

    return y_split, x_split


def adaptive_quadtree_corner_indices(
    ii_sm: Tensor,
    ii_sq: Tensor,
    max_level: int | None = None,
    min_child: int = 4,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    B = ii_sm.shape[0]
    device = ii_sm.device

    # NOTE: ii_sm is padded with one zero row/col
    H = ii_sm.shape[2] - 1
    W = ii_sm.shape[3] - 1

    v2_h = (H & -H).bit_length() - 1
    v2_w = (W & -W).bit_length() - 1
    feasible = min(v2_h, v2_w)

    if max_level is None:
        max_level = feasible
    elif max_level > feasible:
        raise ValueError(
            f"max_level={max_level} exceeds feasible depth {feasible} "
            f"(H={H} divisible by 2^{v2_h}, W={W} divisible by 2^{v2_w})."
        )

    # Root box, same for all images
    ymin = torch.zeros(B, 1, dtype=torch.long, device=device)
    xmin = torch.zeros(B, 1, dtype=torch.long, device=device)
    ymax = torch.full((B, 1), H, dtype=torch.long, device=device)
    xmax = torch.full((B, 1), W, dtype=torch.long, device=device)

    all_ymin, all_xmin = [ymin], [xmin]
    all_ymax, all_xmax = [ymax], [xmax]
    all_lvls    = [torch.zeros(1, dtype=torch.long, device=device)]
    all_parents = [torch.tensor([-1], dtype=torch.long, device=device)]

    node_offset = 1

    for l in range(max_level):
        N = ymin.shape[1]
        # margin = max(min_child, feasibility requirement for further splits)
        margin = max(min_child, 1 << (max_level - l - 1))

        y_split, x_split = _find_adaptive_splits(
            ii_sm, ii_sq, ymin, xmin, ymax, xmax,
            margin=margin,
            min_child=min_child,
            alpha=0.15,
            lambda_balance=0.05,
        )  # (B, N) each

        # 4 children per parent: TL, TR, BL, BR
        c_ymin = torch.stack([ymin,    ymin,    y_split, y_split], dim=-1).reshape(B, 4 * N)
        c_xmin = torch.stack([xmin,    x_split, xmin,    x_split], dim=-1).reshape(B, 4 * N)
        c_ymax = torch.stack([y_split, y_split, ymax,    ymax   ], dim=-1).reshape(B, 4 * N)
        c_xmax = torch.stack([x_split, xmax,    x_split, xmax   ], dim=-1).reshape(B, 4 * N)

        c_lvls = torch.full((4 * N,), l + 1, dtype=torch.long, device=device)
        c_parents = (
            torch.arange(N, dtype=torch.long, device=device)
            .repeat_interleave(4) + (node_offset - N)
        )

        all_ymin.append(c_ymin)
        all_xmin.append(c_xmin)
        all_ymax.append(c_ymax)
        all_xmax.append(c_xmax)
        all_lvls.append(c_lvls)
        all_parents.append(c_parents)

        node_offset += 4 * N
        ymin, xmin, ymax, xmax = c_ymin, c_xmin, c_ymax, c_xmax

    bbox = torch.stack([
        torch.cat(all_ymin, dim=1),
        torch.cat(all_xmin, dim=1),
        torch.cat(all_ymax, dim=1),
        torch.cat(all_xmax, dim=1),
    ], dim=1)

    sizes = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    return (
        bbox,
        torch.cat(all_lvls),
        sizes,
        torch.cat(all_parents),
    )


def adaptive_quadtree_distortion(
    ii_sm: Tensor,
    ii_sq: Tensor,
    bbox: Tensor,
    sizes: Tensor,
) -> Tensor:
    B, _, N = bbox.shape
    device = bbox.device

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(-1)
    ymin = bbox[:, 0].reshape(-1)
    xmin = bbox[:, 1].reshape(-1)
    ymax = bbox[:, 2].reshape(-1)
    xmax = bbox[:, 3].reshape(-1)

    sse = _box_sse_per_image(ii_sm, ii_sq, b_idx, ymin, xmin, ymax, xmax)  # (B*N,)
    return sse.reshape(B, N)


class QTResult(NamedTuple):

    fimg: Tensor
    seg: Tensor
    b_idx: Tensor


class AdaptiveQuadtree(nn.Module):

    def __init__(self, max_level:int=5, min_child:int=4, target_k:int=384):
        super().__init__()
        self.target_k = target_k
        self.max_level = max_level
        self.min_child = min_child

    def forward(self, images: Tensor) -> QTResult:
        B, C, H, W = images.shape
        ii_sm, ii_sq = get_integral_image(images)
        bbox, levels, sizes, parents = adaptive_quadtree_corner_indices(
            ii_sm, ii_sq, self.max_level, self.min_child
        )
        distortion = adaptive_quadtree_distortion(ii_sm, ii_sq, bbox, sizes)
        active = tree_budget_solve_bisect(
            distortion, levels, parents, self.max_level, self.target_k
        )
        b_idx = torch.arange(B, device=bbox.device).unsqueeze(1).expand_as(active)[active]
        labels = torch.arange(active.sum().item(), device=bbox.device)
        bbox = bbox.permute(0, 2, 1)[active].T
        seg = bbox_to_segmap(bbox, b_idx, labels, B, H, W)

        return QTResult(
            fimg=images.permute(0,2,3,1).reshape(-1, C),
            seg=seg,
            b_idx=b_idx,
        )
