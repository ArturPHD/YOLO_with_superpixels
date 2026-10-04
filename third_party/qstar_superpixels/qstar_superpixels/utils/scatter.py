import torch


from torch import Tensor


def _scatter_reduce_2d(src:Tensor, idx:Tensor, red:str, nnz:int|None=None) -> Tensor:
    add_dim = False
    if src.ndim == 1:
        add_dim = True
        src = src.unsqueeze(-1)
    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)
    if nnz is None:
        nnz = int(idx.max().item()) + 1
    out = src.new_empty(nnz, src.shape[1])
    out.scatter_reduce_(0, idx, src, red, include_self=False)
    if not add_dim: return out
    return out.squeeze(-1)


def scatter_sum_2d(src:Tensor, idx:Tensor, nnz:int|None=None):
    return _scatter_reduce_2d(src, idx, 'sum', nnz=nnz)


def scatter_max_2d(src:Tensor, idx:Tensor, nnz:int|None=None):
    return _scatter_reduce_2d(src, idx, 'amax', nnz=nnz)


def scatter_neighbor_argmax(
    val: Tensor, u: Tensor, v: Tensor, n_vtx:int, valbit:int=16
) -> Tensor:
    '''Scatter-based neighbor argmax over edges.

    Parameters
    ----------
    val : Tensor
        Edge values of shape (m,).
    u : Tensor
        Source vertex indices of shape (m,).
    v : Tensor
        Target vertex indices of shape (m,).
    n_vtx : int
        Number of vertices.
    valbit : int, optional
        Number of bits to quantize values. Default is 16.

    Returns
    -------
    Tensor
        A tensor of shape (n_vtx,) with indices of maximum neighbor edges.
    '''
    if n_vtx is None:
        n_vtx = max(int(u.max()) + 1, int(v.max()) + 1)
    idxbit = 63 - valbit
    q = val.clamp(0, 1).mul(2**valbit - 1).long().bitwise_left_shift(idxbit)
    packed = torch.cat([q | u, q | v], 0)
    tgt = torch.cat([v, u], 0)
    out = v.new_full((n_vtx,), -1, dtype=torch.long)
    out.scatter_reduce_(0, tgt, packed, 'amax', include_self=False)
    nbr = out.bitwise_and(2**idxbit - 1)
    return nbr
