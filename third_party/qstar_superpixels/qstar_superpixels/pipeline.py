"""The notebook superpixel stages, shared by inference and fitting."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .features import SPCN
from .merge import batch_per_region, bboxes, filter_regions, get_seg_edges, merge_top_k, relabel_concom
from .qt.tree import AdaptiveQuadtree
from .relax import cheap_relax_loop


@dataclass(frozen=True)
class PipelineConfig:
    # Defaults from qstarfit_2test.ipynb.
    target_k: int = 512
    max_level: int = 5
    min_child: int = 4
    relax_iters: int = 5
    compactness: float = 0.5
    margin: float = 0.1
    min_area: int = 9
    min_diam: int = 1
    merge_rounds: tuple[int, ...] = (5, 5)
    merge_top_k: tuple[int, ...] = (50, 75)
    balance_power: tuple[float, ...] = (0.0, 0.6)
    cmp_mix: float = 0.25

    def __post_init__(self):
        if not (len(self.merge_rounds) == len(self.merge_top_k) == len(self.balance_power)):
            raise ValueError("Merge stage settings must have matching lengths.")
        if self.target_k < 1 or self.max_level < 0 or self.min_child < 1:
            raise ValueError("Invalid quadtree settings.")
        if self.relax_iters < 0 or any(n < 0 for n in self.merge_rounds + self.merge_top_k):
            raise ValueError("Iteration counts and candidate quotas must be nonnegative.")
        if self.min_area < 1 or self.min_diam < 0 or not 0 <= self.cmp_mix <= 1:
            raise ValueError("Invalid filtering or compactness settings.")


@dataclass
class SuperpixelResult:
    """Global region IDs across the batch, with mappings between saved levels.

    seg0 is the relaxed, connected, filtered partition. Each mapping maps
    region IDs at one saved level to IDs at the next. Features are region
    means at each level; batch_indices identify the image owning each region.
    """

    seg0: Tensor
    mappings: tuple[Tensor, ...]
    features: tuple[Tensor, ...]
    batch_indices: tuple[Tensor, ...]

    def all_segs(self) -> tuple[Tensor, ...]:
        levels = [self.seg0]
        for mapping in self.mappings:
            levels.append(mapping[levels[-1]])
        return tuple(levels)

    @property
    def seg(self) -> Tensor:
        return self.all_segs()[-1]

    @property
    def seg1(self) -> Tensor:
        return self.all_segs()[1]

    @property
    def seg2(self) -> Tensor:
        return self.all_segs()[2]

    def region_counts(self) -> tuple[Tensor, ...]:
        return tuple(b.bincount(minlength=self.seg0.shape[0]) for b in self.batch_indices)


class SuperpixelPipeline(nn.Module):
    """Partition RGB, supplied features, or learned six-channel CNN features.

    RGB mode reproduces the test notebook's inference path. Learned mode
    uses the same CNN in both fitting and inference. Inputs for the supplied
    notebook checkpoints must use ImageNet normalization.
    """

    def __init__(self, config: PipelineConfig | None = None, *, learned: bool = False):
        super().__init__()
        self.config = config or PipelineConfig()
        c = self.config
        self.aqt = AdaptiveQuadtree(c.max_level, c.min_child, c.target_k)
        self.learned = learned
        if learned:
            # Keep notebook parameter names for state_dict compatibility.
            self.cn = SPCN(3, 6, 8.0)
            self.pr = nn.Linear(6, 3)

    def forward(self, x: Tensor) -> SuperpixelResult:
        if x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point():
            raise ValueError("Expected a nonempty floating tensor [B, C, H, W].")
        if not torch.isfinite(x).all():
            raise ValueError("Inputs must contain only finite values.")
        if self.learned and (x.shape[1] != 3 or x.shape[2] % 4 or x.shape[3] % 4):
            raise ValueError("Learned mode expects RGB with height and width divisible by four.")
        c = self.config
        res = self.aqt(self.cn(x) if self.learned else x)
        seg = cheap_relax_loop(res.fimg, res.seg.contiguous(), res.b_idx.numel(),
                               c.relax_iters, c.compactness, c.margin)
        seg, n = relabel_concom(seg)
        sizes = seg.flatten().bincount(minlength=n)
        means = res.fimg.new_zeros(n, res.fimg.shape[-1])
        means = means.index_add(0, seg.flatten(), res.fimg) / sizes[:, None]
        edges = get_seg_edges(seg, True)
        boxes = bboxes(seg, n)
        means, seg, sizes, edges, boxes = filter_regions(
            means, seg, sizes, edges, boxes, c.min_diam, c.min_area)
        # Filtering changes IDs, so metadata must be rebuilt before merging.
        batch = batch_per_region(seg, means.shape[0])
        seg0 = seg
        mappings, features, batches = [], [means], [batch]
        for rounds, quota, power in zip(c.merge_rounds, c.merge_top_k, c.balance_power):
            mapping = torch.arange(means.shape[0], device=x.device)
            for _ in range(rounds):
                means, seg, sizes, edges, boxes, batch, cc = merge_top_k(
                    means, seg, sizes, edges, boxes, batch, quota, c.cmp_mix,
                    balance_regions=power != 0, balance_power=power, return_cc=True)
                mapping = cc[mapping]
            mappings.append(mapping)
            features.append(means)
            batches.append(batch)
        return SuperpixelResult(seg0, tuple(mappings), tuple(features), tuple(batches))

    def reconstruction_loss(self, x: Tensor, weights=(0.15, 0.5, 0.35)) -> Tensor:
        """Notebook weighted log-cosh objective on region-mean reconstructions.

        Partition decisions are discrete; gradients flow through the CNN,
        region means, and RGB projection for the selected partitions.
        """
        if not self.learned:
            raise ValueError("Reconstruction fitting requires learned=True.")
        result = self(x)
        if len(weights) != len(result.features) or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Provide a nonnegative loss weight for each saved level.")
        target = x.permute(0, 2, 3, 1)
        losses = []
        for means, seg in zip(result.features, result.all_segs()):
            delta = self.pr(means)[seg] - target
            losses.append((delta + (F.softplus(-8 * delta) - math.log(2)) / 4).mean())
        return sum(w * loss for w, loss in zip(weights, losses))
