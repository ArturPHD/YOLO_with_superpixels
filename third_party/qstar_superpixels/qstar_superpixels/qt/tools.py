import torch

def bbox_to_segmap(boxes, b_idx, labels, B, H, W):
    """
    Scatter axis-aligned bounding boxes into a batched segmentation map.

    Parameters
    ----------
    boxes : Tensor, shape (4, M)
        (row_tl, col_tl, row_br, col_br) for M leaf boxes.
    b_idx : Tensor, shape (M,)
        Batch index per box.
    labels : Tensor, shape (M,)
        Label per box.
    H : int
        Image height.
    W : int
        Image width.
    B : int
        Batch size.

    Returns
    -------
    seg : Tensor, shape (B, H, W)
        Per-pixel label.
    """
    row_tl, col_tl, row_br, col_br = boxes

    # Allocate padded canvas to handle col_br == W or row_br == H
    diff = torch.zeros((B, H + 1, W + 1), dtype=labels.dtype, device=boxes.device)

    # Scatter values to the four corners of each bounding box
    diff.index_put_((b_idx, row_tl, col_tl), labels, accumulate=True)
    diff.index_put_((b_idx, row_br, col_br), labels, accumulate=True)
    diff.index_put_((b_idx, row_tl, col_br), -labels, accumulate=True)
    diff.index_put_((b_idx, row_br, col_tl), -labels, accumulate=True)

    # Integrate to reconstruct the piece-wise constant regions
    seg = diff.cumsum(dim=1).cumsum(dim=2)

    # Crop the padding
    return seg[:, :H, :W]
