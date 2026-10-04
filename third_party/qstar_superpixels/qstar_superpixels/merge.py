import torch


import torch.nn.functional as F


from torch import Tensor


from .utils.indexing import fast_uidx_long2d


from .utils.concom import concom, concom_src


from .utils.scatter import scatter_max_2d, scatter_sum_2d, scatter_neighbor_argmax


def get_seg_edges(seg:Tensor, unique:bool=False) -> Tensor:
    lr = torch.stack((seg[..., :-1].flatten(), seg[..., 1:].flatten()))
    ud = torch.stack((seg[:, :-1].flatten(), seg[:, 1:].flatten()))
    # Every region needs a survival edge, including one-pixel regions.
    labels = seg.unique()
    edges = torch.cat([lr, ud, torch.stack((labels, labels))], -1)
    if unique:
        uidx = fast_uidx_long2d(edges)
        return edges[:,uidx]
    return edges


def relabel_concom(seg:Tensor) -> tuple[Tensor, int]:
    device = seg.device
    B, H, W = seg.shape
    E = get_seg_edges(torch.arange(B*H*W, device=device).view(B,H,W))
    u, v = E[:,seg.view(-1)[E[0]] == seg.view(-1)[E[1]]]
    cc, nV = concom_src(u, v, B*H*W)
    return cc.view(B,H,W), nV


def image_coords(b, h, w, device=torch.device('cpu')):
    flat = torch.arange(b*h*w, device=device)
    bs = flat // (h*w)
    ys = (flat % (h*w)) // w
    xs = (flat % (h*w)) % w
    return torch.stack([bs, ys, xs], dim=0)


def bboxes(seg, n):
    B,H,W = seg.shape
    b,y,x = image_coords(B,H,W,device=seg.device)
    return (
        scatter_max_2d(torch.stack([-y,-x,y,x], dim=1), seg.view(-1), n) *
        torch.tensor([ -1, -1, 1, 1], device=seg.device)
    ).mT


def filter_regions(fmu, seg, siz, E, bbox, min_diam=1, min_area=25):
    nV = fmu.shape[0]
    device = fmu.device

    pivot = torch.tensor([-1, -1, 1, 1], device=bbox.device)

    src, dst = E
    eq = src == dst

    tiny = (
        ((bbox[2] - bbox[0]) < min_diam) |
        ((bbox[3] - bbox[1]) < min_diam) |
        (siz < min_area)
    )

    t0 = tiny[src]
    t1 = tiny[dst]

    tiny_big  = t0 & ~t1
    tiny_tiny = t0 & t1 & ~eq
    big_self  = ~t0 & eq
    tiny_self = t0 & eq

    sim = fmu[E].diff(1, 0).pow(2).mean(-1).neg().exp()[0]

    fE = torch.zeros_like(sim)

    # Non-tiny regions stay fixed.
    fE[big_self] = 1.0

    # Tiny regions strongly prefer existing non-tiny neighbours.
    fE[tiny_big] = 0.5 + 0.5 * sim[tiny_big]

    # Tiny-tiny fallback: useful cleanup, but weaker than any tiny-big edge.
    fE[tiny_tiny] = 0.25 * sim[tiny_tiny]

    # Last-resort survival, assuming self-loops exist.
    fE[tiny_self] = 1e-6

    amax = scatter_neighbor_argmax(fE, src, dst, nV)
    cc, nV = concom(amax, nV)

    seg = cc[seg]
    nsiz = seg.reshape(-1).bincount(minlength=nV)

    fmu = scatter_sum_2d(siz.unsqueeze(1) * fmu, cc, nV) / nsiz.unsqueeze(1)

    E = cc[E]
    E = E[:, fast_uidx_long2d(E)]

    bbox = (scatter_max_2d(bbox.mT * pivot, cc) * pivot).mT

    return fmu, seg, nsiz, E, bbox


def batch_per_region(seg, n):
    batch_idx = torch.arange(
        seg.shape[0],
        device=seg.device
    )[:, None, None].expand_as(seg)
    lookup = torch.full((n,), -1, dtype=torch.long, device=seg.device)
    lookup[seg.flatten()] = batch_idx.flatten()
    return lookup


def merge_top_k(
    fmu,
    seg,
    siz,
    E,
    bbox,
    b_idx,
    top_k=50,
    cmp_mix=0.25,
    eps=1e-5,
    balance_regions=False,
    target_regions=None,
    balance_power=0.0,
    min_top_k=1,
    max_top_k=None,
    return_cc=True,
):
    """
    Merge neighbouring regions using feature similarity and compactness.

    When balance_regions=True, the number of candidate edges selected for
    each image is adjusted according to its current number of regions:

        k_b = top_k * (n_b / target_regions) ** balance_power

    If target_regions is None, the median region count in the batch is used.

    Args:
        fmu:
            Region features, shaped [N, C].
        seg:
            Region-index segmentation maps, shaped [B, H, W].
        siz:
            Region sizes, shaped [N].
        E:
            Directed region edges, shaped [2, M].
        bbox:
            Region bounding boxes, shaped [4, N].
        b_idx:
            Image index associated with each region, shaped [N].
        top_k:
            Baseline number of candidate edges selected per image.
        cmp_mix:
            Geometric interpolation between feature similarity and compactness.
        eps:
            Numerical stability constant.
        balance_regions:
            Enable image-dependent candidate quotas.
        target_regions:
            Reference region count. If None, use the batch median.
        balance_power:
            Strength of the balancing pressure. Zero recovers fixed top-k;
            values around 0.5 give mild balancing, while 1.0 is proportional.
        min_top_k:
            Minimum candidate quota per image.
        max_top_k:
            Optional maximum candidate quota per image.

    Returns:
        Updated fmu, seg, siz, E, bbox, and b_idx.
    """
    B, H, W = seg.shape
    num_regions = fmu.shape[0]

    # Associate each edge with the image containing its source region.
    edge_regions = b_idx[E[0]]

    # Feature similarity.
    fE = (
        fmu[E]
        .diff(dim=0)
        .pow(2)
        .mean(dim=-1)
        .neg()
        .exp()[0]
    )

    # Compactness of the joint bounding box.
    pivot = bbox.new_tensor([-1, -1, 1, 1])

    jbox = (
        bbox.mT[E]
        .mul(pivot)
        .min(dim=0)
        .values
        .mul(pivot)
        .mT
    )

    bh = jbox[2] - jbox[0] + 1
    bw = jbox[3] - jbox[1] + 1
    cmp = 4 * bh * bw / (bh + bw + eps).pow(2)

    fE = cmp.pow(cmp_mix) * fE.pow(1.0 - cmp_mix)

    # Self-edges should not participate in candidate ranking.
    eq = torch.eq(*E)
    fE[eq] = 0

    # Stable lexicographic sorting prevents exact score ties at image
    # boundaries from mixing edges belonging to different batch elements.
    by_score = fE.argsort(descending=True, stable=True)
    asrt = by_score[edge_regions[by_score].argsort(stable=True)]

    # Number and starting position of edges belonging to each image.
    bc = edge_regions.bincount(minlength=B)
    st = F.pad(bc.cumsum(dim=0)[:-1], (1, 0))

    if balance_regions:
        regions_per_image = b_idx.bincount(minlength=B).to(fE.dtype)

        if target_regions is None:
            positive_counts = regions_per_image[regions_per_image > 0]

            if positive_counts.numel() == 0:
                target = regions_per_image.new_tensor(1.0)
            else:
                target = positive_counts.median()
        else:
            target = torch.as_tensor(
                target_regions,
                device=fE.device,
                dtype=fE.dtype,
            )

        target = target.clamp_min(1.0)

        pressure = (
            regions_per_image.clamp_min(1.0) / target
        ).pow(balance_power)

        k_per_image = (
            top_k * pressure
        ).round().to(torch.long)

        k_per_image.clamp_(min=min_top_k)

        if max_top_k is not None:
            k_per_image.clamp_(max=max_top_k)
    else:
        k_per_image = torch.full(
            (B,),
            top_k,
            device=bc.device,
            dtype=torch.long,
        )

    # An image cannot select more candidates than it has edges.
    k_per_image = torch.minimum(k_per_image, bc)

    max_k = int(k_per_image.max().item())

    if max_k > 0:
        offsets = torch.arange(max_k, device=st.device)

        valid_mask = (
            offsets.unsqueeze(0)
            < k_per_image.unsqueeze(1)
        )

        select = (
            st.unsqueeze(1) + offsets.unsqueeze(0)
        )[valid_mask]

        fEs = torch.zeros_like(fE)
        fEs[asrt[select]] = fE[asrt[select]]
    else:
        fEs = torch.zeros_like(fE)

    # Self-edges remain available so that regions need not merge.
    fEs[eq] = (
        2 * siz[E[0, eq]].sqrt() / (H * W) ** 0.5
    ).clip(0, 1)

    amax = scatter_neighbor_argmax(
        fEs,
        E[0],
        E[1],
        num_regions,
    )

    cc, nV = concom(amax, num_regions)

    seg = cc[seg]
    nsiz = seg.reshape(-1).bincount(minlength=nV)

    fmu = (
        scatter_sum_2d(
            siz.unsqueeze(1) * fmu,
            cc,
            nV,
        )
        / nsiz.unsqueeze(1)
    )

    E = cc[E]
    uidx = fast_uidx_long2d(E)
    E = E[:, uidx]

    b_idx = batch_per_region(seg, nV)

    bbox = (
        scatter_max_2d(
            bbox.mT * pivot,
            cc,
        )
        * pivot
    ).mT

    if not return_cc:
        return fmu, seg, nsiz, E, bbox, b_idx
    return fmu, seg, nsiz, E, bbox, b_idx, cc
