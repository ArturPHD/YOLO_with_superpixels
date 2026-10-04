import torch
from torch import Tensor
from warnings import warn

def _dpcc_roots(labels: Tensor):
    device = labels.device
    changed = torch.ones(labels.size(0), dtype=torch.bool, device=device)

    while changed.any():
        idx = torch.nonzero(changed, as_tuple=False).flatten()
        current_labels = labels[idx]
        new_labels = labels[current_labels]
        labels[idx] = new_labels
        changed[idx] = (new_labels != current_labels)
    return labels


def _dpcc_iterative(n:int, u:Tensor, v:Tensor, labels: Tensor, maxit:int = 64):
    m = len(u)
    it = 0

    while m > 0 and it <= maxit:
        l2h = (u < v).sum()
        h2l = m - l2h

        if l2h >= h2l:
            mask = u < v
        else:
            mask = u > v

        # Multiple edges may propose different parents for the same vertex.
        # Indexed assignment with duplicate indices is order-dependent;
        # reduce proposals consistently in the chosen hooking direction.
        reduction = 'amax' if l2h >= h2l else 'amin'
        labels.scatter_reduce_(0, u[mask], labels[v[mask]], reduction, include_self=True)

        # Compute roots
        _dpcc_roots(labels)

        # Compute new edges
        mask = labels[u] != labels[v]
        u = labels[u[mask]]
        v = labels[v[mask]]
        m = len(u)
        it += 1

    if it > maxit:
        msg = f'DPCC iteration limit - current iteration: {it} > max iterations: {maxit}.'
        warn(msg, RuntimeWarning)


def concom(tgt:Tensor, n:int, maxit=64) -> tuple[Tensor, int]:
    assert tgt.ndim == 1
    device = tgt.device
    labels = torch.arange(n, device=device)
    src = labels.clone()
    _dpcc_iterative(n, src, tgt, labels, maxit=maxit)
    is_root = (labels == torch.arange(n, device=device))
    root_idx = is_root.nonzero(as_tuple=False).flatten()
    k = root_idx.numel()
    lut = torch.empty(n, device=device, dtype=torch.long)
    lut[root_idx] = torch.arange(k, device=device)
    return lut[labels], k


def concom_src(src:Tensor, tgt:Tensor, n:int, maxit=64) -> tuple[Tensor, int]:
    assert tgt.ndim == 1
    device = tgt.device
    labels = torch.arange(n, device=device)
    _dpcc_iterative(n, src, tgt, labels, maxit=maxit)
    is_root = (labels == torch.arange(n, device=device))
    root_idx = is_root.nonzero(as_tuple=False).flatten()
    k = root_idx.numel()
    lut = torch.empty(n, device=device, dtype=torch.long)
    lut[root_idx] = torch.arange(k, device=device)
    return lut[labels], k
