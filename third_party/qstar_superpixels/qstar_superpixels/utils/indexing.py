import torch
from torch import Tensor

def fast_uidx_1d(src:Tensor) -> Tensor:
    perm = src.argsort()
    aux = src[perm]
    mask = src.new_zeros(aux.shape[0], dtype=torch.bool)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    return perm[mask]


def fast_uidx_long2d(src:Tensor) -> Tensor:
    m = src.max() + 1
    r, c = src
    cons = r*m + c
    return fast_uidx_1d(cons)
