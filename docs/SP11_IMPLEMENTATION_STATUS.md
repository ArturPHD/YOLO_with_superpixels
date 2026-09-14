# SP-11 Implementation Status

## Scope

SP-11 is a new, train-only strategy. It is intentionally separate from the
historical `auxiliary_loss` strategy, which predicts right/down adjacency with
BCE and remains available unchanged for historical experiments and checkpoints.

SP-11 asks a different question: can foreground predictions become more useful
when they are supervised by detection boxes and regularized to be locally
consistent inside image-adaptive regions?

## Architecture

The default target is the highest-resolution source feeding `Detect`, namely
`Detect.f[0]` / P3. A lightweight head is attached there:

`P3 (B x C x Hf x Wf) -> foregroundness head -> B x 1 x Hf x Wf logits`

The P3 wrapper only stores the captured final augmented RGB image, foreground
logits, and feature shape in a per-forward context. It does not calculate an
SP-11 loss and it does not generate superpixels.

The criterion wrapper receives both the model output and the actual Ultralytics
training batch. It builds the target from final `batch["bboxes"]` and
`batch["batch_idx"]`, then calculates all SP-11 terms and clears the context in
a `finally` block. This prevents stale image/logit tensors across batches.

## Objective

`L_total = L_YOLO + lambda_fg * L_foreground + lambda_sp * L_region`

`L_foreground` is BCEWithLogits between foreground logits and the union of all
GT boxes at P3 resolution. The default `balanced_bce` gives foreground and
background equal aggregate weight; plain BCE remains available.

`L_region` is not superpixel-ID reconstruction. Let `p=sigmoid(logits)`. For a
region `r`, it is the mean within-region variance:

`mean_r(mean_i_in_r((p_i - mean_r(p))^2))`

Region IDs are only grouping labels, so the loss is invariant to a permutation
of those identifiers.

## Ambiguous-region protection

For each feature-grid region, SP-11 computes target purity `q_r`, the mean
box-derived foreground target in that region. Consistency is applied only when:

`q_r >= purity_threshold` or `q_r <= 1 - purity_threshold`

Default `purity_threshold=0.9`. Regions crossing a foreground/background box
edge are ignored. `min_region_size` excludes tiny regions as well.

## Four arms

| Arm | Configuration | Intervention |
|---|---|---|
| A | `configs/sp11/A_rgb_baseline.yaml` | ordinary matched RGB YOLO |
| B | `configs/sp11/B_foreground_aux.yaml` | foregroundness BCE only; `lambda_sp=0` |
| C | `configs/sp11/C_foreground_aux_slic.yaml` | B plus `torch_slic` consistency |
| D | `configs/sp11/D_foreground_aux_grid.yaml` | B plus deterministic `torch_grid` consistency control |

C and D request the same `n_segments=300`. The central scientific comparison is
`C vs B and D`; `C > A` alone is not evidence that superpixels help.

## Runtime and logging

Training logs retain box, cls, and DFL losses and add:

- `fg_aux_loss`
- `sp_consistency_loss`
- requested/actual region counts
- active pure and ignored ambiguous/small region counts
- mean and median region area

During eval, validation, prediction, export, and after explicit stripping, the
P3 wrapper returns the ordinary YOLO feature and does not invoke the
foregroundness head, `torch_slic`, or `torch_grid`. Validation receives zero
SP-11 fields so the loss schema stays stable.

## Reproducibility

`auxiliary_init_seed` makes B/C/D foreground-head initialization deterministic
without changing global RNG state. `base_model_path`, its automatic file hash,
and optional externally supplied `base_model_sha256` are persisted to the
existing `superpixel_config.json` sidecar. The sidecar also fingerprints the
source tree and package versions.

For paired A/B/C/D runs, use the exact same base checkpoint, seed, training
arguments, and base checkpoint hash.

## Tests and smoke status

Unit coverage includes box rasterization, zero/overlapping/border boxes,
augmentation-coordinate target alignment, BCE and consistency gradients,
SLIC/grid label paths, permutation invariance, pure/ambiguous/tiny regions,
lambda-sp-zero behavior, train/eval separation, stripping, and criterion
cleanup. `scripts/run_e2e_smoke.py --strategy structural` adds the same tiny
train-save-reload-val-predict lifecycle route used by existing strategies.

Current status: focused unit and real-model integration coverage passed on
2026-09-14 (`92 passed, 1 skipped` after the base-checkpoint metadata and
lifecycle-path regression tests). The real YOLO12n CPU lifecycle smoke also
passed on 2026-09-14: one tiny synthetic-data epoch, checkpoint save, then
fresh-process validation and prediction all completed successfully in 14.7 s.
No long COCO SP-11 experiment has been launched.

## Known limitations

- `torch_slic` is a GPU-friendly SLIC-like implementation, not a bit-exact
  `skimage.slic` replacement.
- Region counts at P3 may differ from requested image-grid counts after nearest
  label downsampling; both requested and actual counts are logged.
- The implementation is experimental infrastructure, not evidence of a
  scientific benefit. A small A/B/C/D lifecycle smoke must pass before any
  long COCO series.
