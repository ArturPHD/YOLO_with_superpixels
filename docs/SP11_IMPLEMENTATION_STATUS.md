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
region `r`, its raw term is the within-region variance:

`mean_r(mean_i_in_r((p_i - mean_r(p))^2))`

Region IDs are only grouping labels, so the loss is invariant to a permutation
of those identifiers.

The default `balanced_fg_bg` policy computes separate means over active pure
foreground regions and active pure background regions:

`L_region = 0.5 * mean(V_fg) + 0.5 * mean(V_bg)`

when both sets exist. If one set is empty, the available set is used. The
diagnostic-only `all_regions_equal` mode averages every active region together.
This policy is explicit in the resolved protocol and is not inferred from the
data.

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

- `fg_aux_loss_raw`, `fg_aux_loss_weighted`
- `sp_consistency_loss_raw`, `sp_consistency_loss_weighted`
- `aux_total_weighted`
- requested/actual region counts
- active foreground/background and ignored ambiguous/small region counts
- mean and median region area
- raw foreground/background region-variance means

Ultralytics writes these training columns as its epoch-level means of batch
items, not sums over an epoch. Validation bypasses SP-11 and therefore records
zeros for the SP-11 fields while retaining a stable CSV schema.

During eval, validation, prediction, export, and after explicit stripping, the
P3 wrapper returns the ordinary YOLO feature and does not invoke the
foregroundness head, `torch_slic`, or `torch_grid`. Validation receives zero
SP-11 fields so the loss schema stays stable.

## Reproducibility

`configs/sp11/pilot_v1.yaml` is the single shared, versioned source of training
settings. The four arm YAML files contain intervention-only overrides. Run
`scripts/run_sp11_pilot.py --dry-run` before every pilot: it resolves the model
and data-YAML files, calculates checkpoint/data identities, verifies any
supplied expected checkpoint SHA-256, checks arm drift, and writes a
machine-readable manifest with protocol SHA, shared-protocol SHA, arm-config
SHA, resolved-config SHA, checkpoint SHA, and data-YAML SHA. It fails closed if
an arm attempts to override a shared training setting.

`auxiliary_init_seed` makes B/C/D foreground-head initialization deterministic
without changing global RNG state. `base_model_path`, its automatic file hash,
and optional externally supplied `base_model_sha256` are persisted to the
existing `superpixel_config.json` sidecar. The sidecar also fingerprints the
source tree and package versions.

For paired A/B/C/D runs, use the exact same base checkpoint, seed, training
arguments, and base checkpoint hash.

`--diagnostic-batches N` writes raw YOLO, raw/weighted foreground, raw/weighted
region, and total weighted auxiliary scales for the first `N` training batches.
It never alters lambda. `--diagnostic-gradient-norms` additionally records the
P3 foreground-logit gradient L2 norm after backward. `--runtime-profile` is for
a short smoke only: it synchronizes C/D label generation to measure structural
preprocessing time. The normal launcher always writes wall time per epoch, peak
VRAM when CUDA is used, and images/sec when Ultralytics exposes image counts.

## Tests and smoke status

Unit coverage includes box rasterization, zero/overlapping/border boxes,
augmentation-coordinate target alignment, BCE and consistency gradients,
SLIC/grid label paths, permutation invariance, pure/ambiguous/tiny regions,
lambda-sp-zero behavior, train/eval separation, stripping, and criterion
cleanup. `scripts/run_e2e_smoke.py --strategy structural` adds the same tiny
train-save-reload-val-predict lifecycle route used by existing strategies.

Current status: the protocol/diagnostic hardening pass passed on 2026-09-14:
`python -m compileall` and `ruff check` passed; the full suite reported
`107 passed, 1 skipped`; and the CUDA `torch_slic` repeatability smoke was
bit-identical under PyTorch deterministic mode. The no-training A/B/C/D dry run
resolved one checkpoint SHA and one data-YAML SHA across all arms. No long COCO
SP-11 experiment has been launched.

## Known limitations

- `torch_slic` is a GPU-friendly SLIC-like implementation, not a bit-exact
  `skimage.slic` replacement.
- Region counts at P3 may differ from requested image-grid counts after nearest
  label downsampling; both requested and actual counts are logged.
- The implementation is experimental infrastructure, not evidence of a
  scientific benefit. A small A/B/C/D lifecycle smoke must pass before any
  long COCO series.
