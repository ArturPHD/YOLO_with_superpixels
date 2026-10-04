# Q* superpixels

A standalone extraction of the superpixel experiments in `qstartest.ipynb`,
`qstarfit_2.ipynb`, and `qstarfit_2test.ipynb`. It contains the adaptive
quadtree → boundary relaxation → connected-region cleanup → hierarchical
merging pipeline. The package requires only PyTorch. Image loading and the
example notebook have optional dependencies.

`demo.ipynb` runs immediately with a random selection from 16 bundled images
in `imgs/`, exported from the example batch in `qstartest.ipynb`. Launch it
with the project folder or its parent as the working directory. Install the
`notebook` extra if the notebook dependencies are not already available.

This is an exploratory continuation of Differentiable Hierarchical Visual
Tokenization (NeurIPS 2025). Codec and curve reconstruction experiments are
outside this extraction.

## Run

This folder can be copied elsewhere and used independently. From this folder:

```bash
python -m pip install -e '.[images]'
python -m qstar_superpixels image.jpg artifacts/regions.pt
python -m qstar_superpixels image.jpg artifacts/learned.pt \
  --checkpoint checkpoints/tierfit_0_0.5ep.pth --device cuda:0
python -m unittest discover -s tests -v
```

On the original UiO machine, invoke Python through
`~/pyinterpreter_uvtorch2_7.sh` in every command above. For example:

```bash
~/pyinterpreter_uvtorch2_7.sh -m unittest discover -s tests -v
```

The wrapper sources `.bash_profile`, loads the CUDA and Python modules, activates
`/work2/venvs/uvtorch2_7`, and executes Python with the supplied arguments.
The environment currently reports PyTorch 2.9.1+cu126 despite the wrapper's name.
CUDA runs require a process with access to the host GPUs.

Image inputs are loaded at their original resolution and ImageNet-normalized.
There is no implicit resizing or cropping. The existing quadtree implementation
requires height and width divisible by `2 ** max_level` (32 by default).
To match the notebook preprocessing, resize the shorter side to 256 and take a
224×224 center crop; the example notebook demonstrates this. `.pt` inputs must
contain a floating tensor `[B, C, H, W]`, already normalized as appropriate.
Learned mode requires three channels and dimensions divisible by four.

The CLI saves a dictionary of CPU tensors: `segmentations`, `mappings`,
`features`, and `batch_indices`. Region IDs are contiguous and global across
the batch; distinct images never share region IDs.

## Python API

```python
import torch
from qstar_superpixels import PipelineConfig, SuperpixelPipeline

x = torch.randn(2, 3, 224, 224)  # replace with normalized RGB images
model = SuperpixelPipeline()    # direct RGB features, as in the test notebook
with torch.no_grad():
    result = model(x)
fine, middle, coarse = result.all_segs()
print(result.region_counts())
```

`seg0` is the partition after relaxation, connected-component relabeling, and
tiny-region filtering. `mappings[i]` maps region IDs at saved level `i` to IDs
at level `i + 1`. `features[i]` contains pixel-weighted feature means for that
level; `batch_indices[i]` contains each region's image index. `seg` returns the
final level. `seg1` and `seg2` provide convenience access to the default tiers.

The default settings follow `qstarfit_2test.ipynb`: initial quadtree target 512,
five relaxation iterations, compactness 0.5, margin 0.1, minimum area 9,
then five merges with top-k 50 and five with top-k 75 and balancing power 0.6.
The initial budget is approximate; relaxation and connected-component splitting
can increase region counts, and merging reduces them. Top-k selects candidate
edges per image, not a requested final number of superpixels.

For the earlier single-stage schedule, configure:

```python
config = PipelineConfig(target_k=384, merge_rounds=(6,),
                        merge_top_k=(50,), balance_power=(0.0,))
model = SuperpixelPipeline(config)
```

## Learned features and fitting

```python
from qstar_superpixels.optim import AdamW

model = SuperpixelPipeline(learned=True)
model.load_state_dict(torch.load("checkpoints/tierfit_0_0.5ep.pth",
                                map_location="cpu", weights_only=True))
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.2,
                  no_deprecation_warning=True)
optimizer.zero_grad(set_to_none=True)
loss = model.reconstruction_loss(x, weights=(0.15, 0.5, 0.35))
loss.backward()
optimizer.step()
```

The learned model preserves the notebook's `cn.*` and `pr.*` state-dict names:
SPCN maps RGB to six channels, and a linear layer projects region means back
to RGB. Fitting uses the same three-level log-cosh reconstruction objective.
`optim.AdamW` preserves the notebook's cautious optimizer; substituting the
standard PyTorch AdamW changes the training algorithm.

Learned mode runs the CNN for both fitting and inference. RGB mode reproduces
the test notebook's RGB-only inference path, which bypassed the learned CNN.
Calling learned mode without loading weights gives an untrained model.
The original training notebook used initial target 384 and loss weights
`(0.1, 0.6, 0.3)`; those choices remain configurable. The checkpoint itself
does not encode these settings or establish quality metrics.

The partition choices are discrete. Gradients flow through CNN features,
pixel-weighted aggregation, and the RGB projection conditional on the chosen
partition; there is no soft assignment or gradient through region IDs.

## Extraction notes

See [PROVENANCE.md](PROVENANCE.md) for the source of each component and the
small correctness changes made during extraction. The original files remain
untouched. No ImageNet dataset, `quix`, or original project imports are needed.

Tests exercise batch isolation, contiguous labels, four-neighbor connectivity,
hierarchy composition, feature aggregation, checkpoint loading, finite fitting
gradients, and singleton images. A CUDA test runs when CUDA is visible.
These are correctness checks, not a segmentation-quality benchmark.
