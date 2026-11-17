# BatchnormInference Test Plan

> **Status: draft for review (ALMIOPEN-2903).** Not reviewed yet. It follows
> [RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md) and the structure of
> [`sdpa-bwd.md`](sdpa-bwd.md). Counts in §6 are **measured on `develop` `5e23123` (2026-11-17)**;
> counts for the proposed sweep come from the generator's `--dry-run --report` once it exists
> (§3), and are not stated here until then.

This plan lists what a complete BatchnormInference test sweep must exercise and how the sweep is
spread across levels. It covers three op dirs under `integration-test-bundles/`:

| OpDir | Topology | Fusion |
|---|---|---|
| `BatchnormInference` | `Default` | none |
| `BatchnormInferencePointwise` | `Default` | `RELU_FWD` after the batchnorm |
| `BatchnormFwdInference` | `Inference` (sweep) and 6 old-style single-bundle dirs | none |

It does not depend on an engine: the sweep describes the problem space and each engine's
`support.json` claims decide which cases run. **Levels:** quick (`quick/`), PR (`standard/`),
nightly (`comprehensive/`), full (`full/`); quick is a subset of standard; standard, nightly and full are disjoint.

**Out of scope** (leave untouched): `VarianceExt`, `VarianceExtPointwise`,
`PointwiseBatchnormBackward`, `Batchnorm`, `BatchnormBackward`, `BatchnormPointwise`. Filling the
gaps in §2.6 is also out of scope; they are logged with a reason.

## Contents
- [1. The operation](#1-the-operation)
  - [1.1 Tensors and attributes](#11-tensors-and-attributes)
  - [1.2 Semantics sources and status](#12-semantics-sources-and-status)
- [2. Feature inventory](#2-feature-inventory)
  - [2.1 Layouts and ranks](#21-layouts-and-ranks)
  - [2.2 Data types and upcast](#22-data-types-and-upcast)
  - [2.3 Stats and affine tensors](#23-stats-and-affine-tensors)
  - [2.4 Fused ReLU](#24-fused-relu)
  - [2.5 Shapes](#25-shapes)
  - [2.6 Known gaps](#26-known-gaps)
- [3. Building the sweep](#3-building-the-sweep)
  - [3.1 Level contents](#31-level-contents)
  - [3.2 Products deliberately not run](#32-products-deliberately-not-run)
  - [3.3 Engine claims](#33-engine-claims)
- [4. Golden reference](#4-golden-reference)
- [5. Pass criteria](#5-pass-criteria)
- [6. Current state and migration](#6-current-state-and-migration)
- [7. Open decisions](#7-open-decisions)

## 1. The operation

Inference batchnorm normalizes `x` per channel `c` with **given** statistics (no batch statistics
are computed, which is what separates it from training):

```text
y = scale_c * (x - mean_c) * inv_variance_c + bias_c
```

`inv_variance` is the graph's input (the tensor uid is `inv_var`), not a variance. Because every
input is supplied, the op is elementwise per channel: cost grows with element count, and the
interesting risks are layout/stride handling, dtype handling and channel-broadcast, not numerics of
a reduction.

### 1.1 Tensors and attributes

| OpDir | uid map |
|---|---|
| `BatchnormInference` | Y0 X1 bias2 inv_var3 mean4 scale5 |
| `BatchnormInferencePointwise` | Y0 (virtual) OUT1 bias2 inv_var3 mean4 scale5 x6 |
| `BatchnormFwdInference` | x0 mean1 inv_var2 scale3 bias4 y5 |

Case record: `values.io_data_type`, `values.tensors[]` (`uid`, `dims`, `strides`, `data_type`) and,
for Pointwise only, `values.attributes` with `relu_lower_clip`, `relu_lower_clip_slope`,
`relu_upper_clip`. Statistics tensors have dims `[1, C, 1, ...]` and strides `[C, 1, 1, ...]`.

### 1.2 Semantics sources and status

Engine behavior is recorded by the per-engine `support.json` claims and
`OperationSupport-GpuReferenceImpl.md`. No separate semantics register for batchnorm was found (not exhaustively searched).
`epsilon` is not a case axis: inference consumes `inv_variance` directly.

## 2. Feature inventory

### 2.1 Layouts and ranks

Six layouts across three ranks: NCL, NLC (rank 3); NCHW, NHWC (rank 4); NCDHW, NDHWC (rank 5).
Channel-last strides are explicit, e.g. NHWC `[200704, 1, 1792, 16]` for `1x16x112x112`, NLC
`[672, 1, 3]` and NCL `[672, 224, 1]` for `1x3x224`. A layout is a stride pattern, so `C = 1` is a
shape edge (§2.5), not a layout.

### 2.2 Data types and upcast

`io_data_type` is fp32, fp16 or bf16. **Upcast** means the output is fp32 while X is fp16 or bf16 and
the statistics are fp32; its id suffix is `_upcast`. Upcast cases exist in the current sweeps; only
`HIP_MLOPS_ENGINE` claims them, MIOpen does not (§3.3).

### 2.3 Stats and affine tensors

Every existing case uses full-rank statistics `[1, C, 1, ...]`. The bare `[C]` form is accepted by
the HIP engine but has no case (§2.6).

### 2.4 Fused ReLU

Pointwise adds `RELU_FWD` in five variants, plain plus four clip forms (id suffixes):

| Variant | Suffix | lower | slope | upper |
|---|---|---|---|---|
| plain | none | null | null | null |
| lo0 | `_rlo0_norslope_norhi` | 0 | null | null |
| hi6 | `_norlo_norslope_rhi6` | null | null | 6 |
| clamp | `_rlo0p1_norslope_rhi0p5` | 0.1 | null | 0.5 |
| leaky | `_norlo_rslope0p01_norhi` | null | 0.01 | null |

Leaky is present in `full` only and claimed by no engine: it is a `notrun` register entry with the
reason "both engines reject `relu_lower_clip_slope`", not a hidden omission.

### 2.5 Shapes

Id form: `<dims_>_<fp32|fp16|bfp16>_<layout>[_<clip suffix>][_upcast]`. The catalog keeps the
shapes already on `develop` and adds nothing new:

| Class | Shapes | Why |
|---|---|---|
| Small | `1_3_224`, `1_3_14_14`, `2_3_3_1_1` | one per rank, cheapest |
| Edge | `4_3_1`, `1_256_1_1`, `32_3_1_14`, `2_3_1_1`, `32_1_14_14`, `32_3_14_1` | degenerate spatial dims, `C = 1`, odd dims |
| Medium | `2_16_512`, `1_64_1024`, `16_3_8_14_14` | larger C, rank 5 |
| Large | `8_32_2048`, `16_128_512`, `1_16_112_112`, `5_256_14_14`, `1_16_8_112_112`, `5_256_4_14_14` | full level only |

### 2.6 Known gaps

| Gap | Reason it stays open |
|---|---|
| Upcast on the MIOpen engine | engine does not claim it; log, do not skip silently |
| Reduced-rank `[C]` statistics | HIP accepts it; no bundle case; out of scope for this ticket |
| `IDENTITY` pointwise (no activation) as a distinct case | covered by `plain`; no separate case |
| Leaky ReLU | engines reject it (§2.4) |

## 3. Building the sweep

Bundles are **generated** from this plan and nobody edits them by hand. The generator
(`generate_batchnorm_inference_bundles.py`, with ids and values from `batchnorm_inference_plan.py`)
holds the tier rules as code. It borrows the plan-as-code idea of `generate_sdpa_bwd_bundles.py` but
none of its TOML or calibration parts. Unit tests are in `reference-data-scripts/tests/`.

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_batchnorm_inference_bundles.py \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles --dry-run --report
```

`--dry-run` lists the files that differ and exits 1 if any do; without it the files are rewritten.
The 3-file bundle format is kept: `graph.template.json` (reused as-is), `sweep.json` (indent 2) and
`support.json` (indent 2, sorted keys). Plan-as-code replaces hand-appended cases.

### 3.1 Level contents

Each case id is `<dims>_<dtype>_<layout>[<clip>][_upcast]`. Shape classes (dims are N, C, spatial...):

| Class | Shapes |
|---|---|
| Small | `4x3x1`, `1x3x224`, `2x3x1x1`, `1x256x1x1`, `1x3x14x14`, `2x3x3x1x1` |
| Edge | `32x3x1x14`, `32x3x14x1`, `32x1x14x14` |
| Medium | `2x16x512`, `1x64x1024`, `16x3x8x14x14` |
| Large | `8x32x2048`, `16x128x512`, `1x16x112x112`, `5x256x14x14`, `5x256x4x14x14`, `1x16x8x112x112` |

"Base combos" for a shape are its rank's layouts (ncl/nlc, nchw/nhwc, ncdhw/ndhwc) x dtype, plus
an `_upcast` variant for fp16 and bfp16. Quick is a subset of standard; standard, comprehensive and
full are disjoint, and a run at a level includes the levels below it.

| Level | Rule | BatchnormInference | Pointwise |
|---|---|---|---|
| quick | 6 cases: every layout once, every dtype twice, one upcast. Pointwise adds the 3 ReLU clips on `2x3x1x1` fp32 nchw. | 6 | 9 |
| PR (`standard`) | Base combos on Small + Edge. Pointwise adds the 3 ReLU clips x layouts, fp32, on Small. | 90 | 126 |
| nightly (`comprehensive`) | Base combos on Medium. Pointwise rotates the clip across each shape's base combos instead of crossing them. | 30 | 30 |
| full | Base combos on Large. Pointwise adds the 3 ReLU clips on `1x16x112x112` and the leaky clip on 3 small shapes. | 60 | 72 |

Leaky ReLU appears only in `full` and is unclaimed by both engines (§3.3), so those cases are
recorded as not run. The 200,704-element `1x16x112x112` clip cases are in `full`; per-case
overhead, not element count, probably dominates wall-clock, so the placement is re-costed once a
measured run exists.

### 3.2 Products deliberately not run

- Leaky ReLU on every shape (§2.4).
- Upcast with the MIOpen engine (§2.6).
- Value duplicates: `BatchnormFwdInference/Inference` `nchw_{bfp16,fp16,fp32}` equal
  `BatchnormInference` `Default` and are dropped. The 3 golden FwdInference bundles and the odd-dim
  shapes they carry are kept; they carry the `.dvc` golden pointers (not yet checked whether any other bundle does).

### 3.3 Engine claims

| Engine | Claims |
|---|---|
| `HIP_MLOPS_ENGINE` | 4D/5D and 3D (NCL/NLC), upcast, no leaky |
| `MIOPEN_ENGINE` | all layouts and ranks with fewer cases, no upcast, no leaky |

Architectures (from `support.json`): HIP_MLOPS gfx1100 linux; gfx1101 windows; gfx1150 linux;
gfx1151 linux and windows; gfx1200 linux; gfx90a, gfx942, gfx950 linux. MIOPEN gfx1030 windows;
gfx1101 linux; gfx1150 linux; gfx1151 linux and windows; gfx1201 linux; gfx90a, gfx942, gfx950
linux. The generator emits `support.json` from the claim rules above, so a claim change is a plan
change.

## 4. Golden reference

Baseline only, not exhaustive: keep the 3 existing `BatchnormFwdInference` golden cases
(nchw, fp32/fp16/bf16). Confirmation by a third party means checking the reference against
`torch.nn.functional.batch_norm(..., training=False)` for the same inputs, which needs a conversion
from `inv_variance` to `running_var`. **Not done yet:** `torch` cannot be imported on the authoring
box, so this plan claims no third-party confirmation. There is also no
`generate_batchnorm_fwd_golden.py`; `generate_batchnorm_reference.py` is the only existing script.

## 5. Pass criteria

A case passes when the engine output matches the reference within the tolerance row of the claiming
engine TOML for that dtype. No TOML changes are proposed by this plan. Upcast cases are compared in
fp32.

## 6. Current state and migration

Measured on `develop` `5e23123` (2026-11-17):

| OpDir | quick | standard | comprehensive | full | total |
|---|---|---|---|---|---|
| `BatchnormInference` | 10 | 82 | 28 | 60 | 180 |
| `BatchnormInferencePointwise` | 15 | 113 | 17 | 135 | 280 |
| `BatchnormFwdInference` (sweep) | 5 | 2 | 4 | 0 | 11 |

Plus 6 old-style single-bundle dirs and 6 golden `.dvc` pointers under `BatchnormFwdInference`.
The ticket text quoted 108 (BatchnormInference) and 168 (Pointwise); the table above is what the dirs contain. Migration:
generate the proposed set, diff it against the table, and for every removed bundle record which rule
removed it.

## 7. Open decisions

- Quick-tier breadth: 6-case minimum (proposed) versus the existing pairwise 18 per op.
- The reviewer's tier plan (`BatchnormInference.md` from ALMIOPEN-2851) has not been seen; it is an
  input, not the plan.
- Whether the `BatchnormFwdInference` single-bundle dirs are deleted or kept next to the `Inference`
  sweep.
- Which gfx box runs the GPU `ctest` validation.
