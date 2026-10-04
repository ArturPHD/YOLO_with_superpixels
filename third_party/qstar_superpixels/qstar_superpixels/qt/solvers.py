import torch

from torch import Tensor


def _normalize_lam(lam, batch_size, device, dtype):
    lam = torch.as_tensor(lam, device=device, dtype=dtype)

    if lam.ndim == 0:
        return lam.expand(batch_size)
    if lam.shape == (batch_size, 1):
        return lam[:, 0]
    if lam.shape == (batch_size,):
        return lam

    raise ValueError(
        f"lam must be a scalar, shape ({batch_size},), or shape ({batch_size}, 1), got {tuple(lam.shape)}."
    )


def _normalize_rate(rate, levels, dtype):
    if rate is None:
        return torch.ones_like(levels, dtype=dtype)

    rate = torch.as_tensor(rate, device=levels.device, dtype=dtype)
    if rate.ndim == 0:
        return rate.expand_as(levels)
    if rate.shape == levels.shape:
        return rate

    raise ValueError(
        f"rate must be a scalar or shape {tuple(levels.shape)}, got {tuple(rate.shape)}."
    )


def _level_nodes(levels, max_level):
    return [torch.nonzero(levels == level, as_tuple=False).squeeze(1) for level in range(max_level + 1)]


def _has_children(parent_idx, num_nodes):
    has_children = torch.zeros(num_nodes, dtype=torch.bool, device=parent_idx.device)
    child_mask = parent_idx >= 0
    has_children[parent_idx[child_mask]] = True
    return has_children


def tree_threshold_solve(score, levels, parent_idx, max_level, threshold):
    has_children = _has_children(parent_idx, levels.numel())
    accepted = (score < threshold) | (~has_children).unsqueeze(0)
    ancestor_accepted = torch.zeros_like(accepted)
    level_nodes = _level_nodes(levels, max_level)

    for level in range(1, max_level + 1):
        nodes = level_nodes[level]
        if nodes.numel() == 0:
            continue

        pidx = parent_idx[nodes]
        ancestor_accepted[:, nodes] = accepted[:, pidx] | ancestor_accepted[:, pidx]

    return accepted & ~ancestor_accepted


def tree_rd_solve(distortion, levels, parent_idx, max_level, lam, rate=None):
    batch_size, num_nodes = distortion.shape
    lam = _normalize_lam(lam, batch_size, distortion.device, distortion.dtype)
    rate = _normalize_rate(rate, levels, distortion.dtype)

    keep_cost = distortion + lam.view(-1, 1) * rate.view(1, -1)
    total_cost = keep_cost.clone()
    keep_node = torch.ones_like(distortion, dtype=torch.bool)
    has_children = _has_children(parent_idx, num_nodes)
    level_nodes = _level_nodes(levels, max_level)
    parent_lookup = torch.full((num_nodes,), -1, dtype=torch.long, device=levels.device)

    for level in range(max_level - 1, -1, -1):
        parents = level_nodes[level]
        children = level_nodes[level + 1]
        if parents.numel() == 0:
            continue

        split_cost = torch.zeros(
            batch_size,
            parents.numel(),
            device=distortion.device,
            dtype=distortion.dtype,
        )
        split_valid = torch.zeros(parents.numel(), dtype=torch.bool, device=levels.device)

        if children.numel() > 0:
            parent_lookup.fill_(-1)
            parent_lookup[parents] = torch.arange(parents.numel(), device=levels.device)
            rel_parent = parent_lookup[parent_idx[children]]

            if (rel_parent < 0).any():
                raise ValueError("Each node must have its parent on the previous level.")

            split_cost.scatter_add_(
                1,
                rel_parent.view(1, -1).expand(batch_size, -1),
                total_cost[:, children],
            )
            split_valid[rel_parent] = True

        can_split = has_children[parents] & split_valid
        keep_here = ~can_split.unsqueeze(0) | (keep_cost[:, parents] <= split_cost)
        keep_node[:, parents] = keep_here
        total_cost[:, parents] = torch.where(keep_here, keep_cost[:, parents], split_cost)

    ancestor_kept = torch.zeros_like(keep_node)

    for level in range(1, max_level + 1):
        nodes = level_nodes[level]
        if nodes.numel() == 0:
            continue

        pidx = parent_idx[nodes]
        ancestor_kept[:, nodes] = keep_node[:, pidx] | ancestor_kept[:, pidx]

    leaf_mask = keep_node & ~ancestor_kept
    return leaf_mask, total_cost[:, 0]


def tree_budget_solve_bisect(
    distortion, levels, parent_idx, max_level, target_k, rate=None, tol=1, max_iter=32
):
    batch_size = distortion.shape[0]
    target_k = torch.as_tensor(target_k, device=distortion.device, dtype=distortion.dtype)

    if target_k.ndim == 0:
        target_k = target_k.expand(batch_size)
    elif target_k.shape != (batch_size,):
        raise ValueError(
            f"target_k must be a scalar or shape ({batch_size},), got {tuple(target_k.shape)}."
        )

    lo = torch.zeros(batch_size, device=distortion.device, dtype=distortion.dtype)
    hi = distortion.max(dim=1).values
    leaf_mask = torch.zeros_like(distortion, dtype=torch.bool)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        leaf_mask, _ = tree_rd_solve(
            distortion, levels, parent_idx, max_level, mid, rate
        )
        k = leaf_mask.sum(-1).to(distortion.dtype)

        lo = torch.where(k > target_k + tol, mid, lo)
        hi = torch.where(k < target_k - tol, mid, hi)

        if ((k - target_k).abs() <= tol).all():
            break

    return leaf_mask
