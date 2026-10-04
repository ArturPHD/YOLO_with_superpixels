# Extraction provenance

This folder was extracted from the QSTAR research workspace. It is independent
of the original notebooks and `qstar` package.

| Component | Original source |
| --- | --- |
| Adaptive quadtree and RD budget solver | `qstar/qt/tree.py`, `solvers.py`, `tools.py` |
| Boundary relaxation | `qstar/car/relax.py` |
| Region adjacency, filtering, bounding boxes | `qstar/merge/utils.py` |
| Balanced hierarchical merging | cell 7 of `qstarfit_2test.ipynb`, also used in `qstarfit_2.ipynb` |
| Three saved hierarchy levels | cell 8 of the tiered fitting/test notebooks |
| Learned feature model | `qstar/cnn/cnn.py`, SPCN only |
| Reconstruction fitting objective | `TokenizerFitter` in the tiered notebooks |
| Cautious AdamW | `qstar/optim/cadam.py`; original attribution to C-Optim retained |
| Tiered checkpoint | exact copy of `tierfit_0_0.5ep.pth` from the workspace root |
| Bundled demo images | first 16 samples from `qstartest.ipynb` cell 6's batch, local IN1k indices 4321–4336 |

The source folder had no Git history. Sources are identified by filename and
notebook cell, rather than a commit. The checkpoint's filename is retained,
but its training duration is not independently verified.

## Deliberate corrections

- Recompute region-to-image IDs after tiny-region filtering. The notebooks
  passed stale IDs to the first merge after region IDs had changed.
- Always include a self-edge for every region, so single-pixel regions can
  survive without an invalid neighbor index. The old graph only had self-edges
  when adjacent pixels already belonged to the same region.
- Sort merge candidates lexicographically by image and descending score.
  Subtracting a floating score from the image ID could mix neighboring images
  at exact score boundaries.
- Resolve duplicate connected-component parent proposals with an integer
  scatter reduction. The original indexed assignment could select different
  parents depending on execution order; this showed up as differing outputs
  after loading identical model parameters.
- Compose hierarchy maps directly, including identity maps for zero-round
  stages, instead of relying on the ambiguous `get_hierarchy_sublevel` helper.
- Share one pipeline between fitting and inference. Explicit RGB mode preserves
  RGB partitioning; explicit learned mode applies SPCN in both paths.
- Return a consistent result object. The training notebook declared an `fimg`
  field but constructed its result without that argument.
- Clamp temporary integral-image queries used to score invalid adaptive-split
  candidates. The validity mask still excludes those candidates; the guard
  prevents an out-of-range tensor index before that mask is applied.

These changes can alter partitions compared with saved notebook outputs.
Quadtree search, relaxation energy, filter scores, merge scores and quotas,
learned feature architecture, and the log-cosh objective were retained.
Codec, curve projection, retired algorithms, datasets, and notebook outputs
were not carried into this project.
