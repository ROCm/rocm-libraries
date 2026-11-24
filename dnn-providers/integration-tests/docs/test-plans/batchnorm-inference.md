# BatchnormInference Test Plan

> **Status: draft for review (ALMIOPEN-2903).** This plan has not been reviewed yet. No
> generator, golden case, claim harvest or timing run exists for it; the case counts in §4.1
> are estimates computed from the level rules, and the generator's `--report` replaces them.
> Audited on `develop` at `1207c1ecfbc`.

This plan lists every feature of batchnorm inference (`BatchnormInferenceAttributes`), plain and
fused with a `relu_fwd` pointwise node, that a complete test sweep must exercise, and how the sweep
is spread across levels. The sweep describes the problem space and does not derive it from any
engine; each engine's support claims decide which cases it runs. Out of scope: training
(`Batchnorm/`, `BatchnormPointwise/`), the backward chains (`BatchnormBackward/`,
`BatchnormInferencePointwiseBatchnormBackward/`), the VarianceExt node
(`BatchnormInferenceAttributesVarianceExt*`: raw variance plus an epsilon tensor, its own plan),
layer and RMS norm, and data types other than fp32, fp16 and bf16, for which no reference
registers a signature.

The plan follows [RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md) (Per-Op
Test Plans); [§8](#8-how-the-sweep-was-built) states the method this plan used. See
[the test-plans index](README.md) for the plans and the tier definitions.

The bundles will be **generated** from this plan, as the
[convolution forward plan](convolution-fwd.md) does it. Nobody edits them by hand:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_batchnorm_inference_bundles.py \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles --report
```

| Plan item | Where it lives (to be written) |
|---|---|
| Axes, legality, shape catalog, ceilings, activations, value recipes, trimmed products | `reference-data-scripts/batchnorm_inference_plan.py` |
| Level rules, graph templates, sweep writer, purpose tags | `reference-data-scripts/generate_batchnorm_inference_bundles.py` (deterministic, idempotent, `--dry-run` and `--report`) |
| Golden data (§5) | `reference-data-scripts/generate_batchnorm_inference_golden.py` |
| Bundles | `integration-test-bundles/<level>/BatchnormInference/Default` and `BatchnormInferencePointwise/Default` in every level |

Every sweep case records the plan items it implements (`metadata.plan_items`), its `purpose`
(§4.2) and its size class, as the convolution bundles do. That is the traceability in both
directions: a plan item names its cases with a filter on `plan_items`, and a case names its items.
When two candidates produce the same graph, the generator keeps the one from the lower level (or
the earlier rule) and writes the union of their plan items. `--report` fails on an orphan either
way.

The generator replaces every captured case in the two directories; the captured cases have no
purpose or plan items, use fp32 statistics only, and have golden data that cannot be reproduced.
`BatchnormFwdInference/` holds the same graph under other uids and is retired once the golden cases
of §5 pass (T8, §4.3).

The features come from the graph API:

- [batchnorm_inference_attributes.fbs](../../../../projects/hipdnn/flatbuffers_sdk/schemas/batchnorm_inference_attributes.fbs), plus `tensor_attributes.fbs`, `pointwise_attributes.fbs`, `data_types.fbs`, `graph.fbs`
- [BatchnormInferenceNode.hpp](../../../../projects/hipdnn/frontend/include/hipdnn_frontend/node/BatchnormInferenceNode.hpp) and [Utilities.hpp](../../../../projects/hipdnn/frontend/include/hipdnn_frontend/node/detail/Utilities.hpp) (`validateChannelOnlyTensorShape`)
- [GpuBatchnormFwdInfSignatureKey.hpp](../../src/harness/gpu-graph-executor/detail/GpuBatchnormFwdInfSignatureKey.hpp) and the CPU `BatchnormFwdInferenceSignatureKey.hpp`

## 1. The operation

```text
y[n, c, s...] = scale[c] · (x[n, c, s...] − mean[c]) · inv_variance[c] + bias[c]
```

`inv_variance` is an input, already inverted: `1 / sqrt(variance + epsilon)` is computed outside
the op, so epsilon is not an axis of this plan. Every statistic is supplied, so the op is
elementwise per channel: no reduction, no accumulation order. Its risks are layout and stride
handling, channel broadcast, data types and the fused epilogue.

| Tensor | Dims | Rule |
|---|---|---|
| x | N, C, spatial… | at least 2 dims; the API sets no upper limit |
| y | same as x | shape must equal x when set; dims and strides are inferred from x when not set |
| mean, inv_variance, scale, bias | 1, C, 1… | at least 2 dims; `dims[0] == 1`, `dims[1] == C`, every later dim 1. The frontend does not require the rank to equal x's rank (F12). |

The node has no attributes besides the six tensor uids. The node `compute_data_type` and the graph's
io, intermediate and compute data types are set separately. Unlike training, inference does not
require `N × spatial > 1` (comment in `BatchnormInferenceNode::pre_validate_node`). The fused graph
feeds a virtual y into a `relu_fwd` pointwise node with `relu_lower_clip`, `relu_upper_clip` and
`relu_lower_clip_slope`.

| Topology | Bundle | Nodes | uid map |
|---|---|---|---|
| `plain` | `BatchnormInference/Default` | batchnorm inference | Y 0, X 1, bias 2, inv_variance 3, mean 4, scale 5 |
| `relu` | `BatchnormInferencePointwise/Default` | batchnorm inference → `relu_fwd` | Y 0 (virtual), OUT 1, bias 2, inv_variance 3, mean 4, scale 5, X 6 |

**Legality** is declared once, in `batchnorm_inference_plan.legal()`, and copies
`pre_validate_node`: all six tensors set; x at least 2 dims; y shape equal to x; each statistic and
affine tensor at least 2 dims, `dims[0] == 1`, `dims[1] == C`, every later dim 1. The generator
never writes an illegal graph.

## 2. Feature inventory

Each feature lists the values a complete sweep must hit. The **Oracle** column says whether the
verification references can judge the feature today (§6). The **Earliest level** column is the
lowest level that holds a case of the feature (`Q` quick, `PR`, `N` nightly, `F` full; `—` means
the sweep cannot express it yet).

### 2.1 Geometry

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F1 | Rank | 3, 4, 5 (1D, 2D, 3D spatial). Also rank 2 (`[N, C]`, no spatial) and rank 6 (4D spatial): the API accepts both, so either support or a clean rejection must be shown. | ready (both references are stride-based; the CPU reference document lists 4D and 5D only, §6) | Q (3–5), PR (2, 6) |
| F2 | Batch N | 1, 2 (default), odd (3), 8, 32, 64 | ready | Q |
| F3 | Channels C | 1, 3, odd prime (13), 15, 17, 31, 33, 63, 65, 96, 256, 2048 | ready | Q |
| F4 | Spatial extent | 1 per axis; one axis 1 (16×1, 1×16, 8×1×8); odd (7, 15, 17, 55, 2047); model sizes (7, 28, 56, 112); long 1D (2048, 1,048,576) | ready | Q |
| F5 | One element per channel | `N × spatial == 1` (`1 × 256 × 1 × 1`): legal for inference, not for training | ready | PR |
| F6 | Output shape | y dims and strides explicit vs inferred. **Inferred outputs are not a bundle case:** the graph JSON loader requires `dims` on every tensor (`hipdnn_flatbuffers_sdk/utilities/json/TensorAttributes.hpp`). They stay a frontend C++ test. | explicit: ready | Q (explicit) |

### 2.2 Memory layout and tensor kind

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F7 | Layout | channels-first and channels-last for each rank: NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | ready | Q |
| F8 | Mixed layouts | x channels-first with y channels-last (the statistic tensors have no layout) | ready | PR |
| F9 | Non-packed strides | padded rows (`np_row`) and a padded channel stride (`np_chan`) on x and y | ready (stride-based) | PR |
| F10 | Alignment | declared alignment below 16 tested by placement, as in the convolution plan (F17 there) | **not run**: I3 | — |
| F11 | 64-bit indexing | x with more than 2³¹ elements (X3) | to confirm (§6) | F |
| F12 | Statistic rank below x rank | `[1, C]` statistics and affine tensors with a rank-4 x. Legal: the frontend checks `dims[0]`, `dims[1]` and the later dims, not the rank. | to confirm (§6) | PR |

**Tensor kind.** All six tensors are device tensors in every case. A constant or runtime
pass-by-value operand is not covered: the GPU reference declines any pass-by-value operand
(`GpuBatchnormFwdInfPlan.hpp`), and a per-channel tensor is a scalar only when C is 1.

### 2.3 Data types

The reference signatures are (x, scale and bias, mean and inv_variance, y, compute):

| Signature | GPU reference | CPU reference |
|---|---|---|
| fp32, fp32, fp32, fp32, fp32 | yes | yes |
| fp16, fp32, fp32, fp16, fp32 | yes | yes |
| bf16, fp32, fp32, bf16, fp32 | yes | yes |
| fp16, fp32, fp32, fp32, fp32 (upcast) | yes | yes |
| bf16, fp32, fp32, fp32, fp32 (upcast) | yes | yes |
| fp16, fp16, fp16, fp16, fp32 | yes | yes |
| bf16, bf16, bf16, bf16, fp32 | yes | yes |
| fp16, fp16, fp16, fp16, fp16 | — | yes |
| bf16, bf16, bf16, bf16, bf16 | — | yes |

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F13 | IO data type | fp32, fp16, bf16 | ready | Q |
| F14 | Upcast | y fp32 with x fp16 or bf16, fp32 statistics | ready | Q |
| F15 | Statistic data type equal to io | fp16 or bf16 scale, bias, mean and inv_variance | ready | Q (probe) |
| F16 | Node compute type | fp16 or bf16 compute, with statistics in the same type | CPU reference only | PR |

### 2.4 Values and numerics

The fill recipes of a bundle are `free` (uniform in [lo, hi]) and `fixed` (constant), set per
tensor in `metadata.inputs`. The harness defaults (`FillInputs.cpp`) are x, scale and bias uniform
in [−1, 1], mean in [−0.1, 0.1] and inv_variance in [0.5, 1.5]. The generator writes the fill of
every case explicitly.

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F17 | Value regimes | default; constant (every input fixed, so y is exact); zero scale (y = bias); tiny variance (inv_variance uniform in [100, 1000]); large magnitude (x uniform in [−a, a], with `a` chosen so that \|y\| reaches about a quarter of the io type's maximum); cancellation (x uniform in [0.99, 1.01], mean fixed at 1) | ready | PR |
| F18 | Special values | NaN and ±Inf in x or a statistic must propagate to the right outputs | needs I4 and the `allclose_matching_infinities` validator | — |
| F19 | Determinism | repeated execution gives bitwise-identical output | **C++ test** (I5) | — |

### 2.5 Fusion

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F20 | ReLU variant | plain (all three fields unset); lower clip 0 written explicitly (`relu_lo0`); ReLU6 (upper 6); clamp (lower 0.1, upper 0.5); a non-zero lower clip with no upper clip (`relu_lo0p1`); leaky (slope 0.01); slope 0 written explicitly (`relu_slope0`), which must run as plain ReLU | ready: both references pass all three fields to the ReLU functor | Q |
| F21 | Fused intermediate type | virtual y in fp32 (template default) vs in the io type | ready | PR |
| F22 | Other epilogues | `identity`, sigmoid, tanh, a bias or residual add after the batchnorm | ready | — (T9) |

### 2.6 Execution

Workspace limits, plan reuse with new pointers, and serialise → deserialise → execute cannot be
expressed as a bundle case. They stay C++ tests until the harness can repeat and compare (I5).
Inferring y dims and strides from x (F6) is frontend behaviour: the harness loads a serialized graph
and never runs the frontend's `validate()` or property inference.

### 2.7 Invalid problems

These are frontend unit tests, not sweep cases. Each must be rejected with `INVALID_VALUE` or
`ATTRIBUTE_NOT_SET`:

- R1 a missing tensor (any of the six)
- R2 x with fewer than 2 dims
- R3 a statistic or affine tensor with fewer than 2 dims, including the bare `[C]` form
- R4 a statistic or affine tensor with `dims[0] != 1`
- R5 a statistic or affine tensor with `dims[1] != C`
- R6 a statistic or affine tensor with a later dim other than 1
- R7 y dims different from x

The bare `[C]` form is the "reduced-rank statistics" case the ticket names. The frontend rejects it
(R3), so it is not a legal graph and not a sweep case, even though `HIP_MLOPS_ENGINE` accepts `[C]`
scale and bias. See D1 in §4.3.

## 3. Shape catalog

Batchnorm inference does about five arithmetic operations per element, so the RFC's FLOP ceilings
never bind. The ceilings bound the elements of x; bytes are x plus y in fp32. A timing run replaces
them.

| Level | Per-case ceiling | Shape sets of the level's own cases |
|---|---|---|
| quick | ≤ 64K elements (≤ 0.5 MB) | S |
| PR | ≤ 1M elements (≤ 8 MB) | S, H, M7 |
| nightly | ≤ 8M elements (≤ 64 MB) | S, L, M, H |
| full | none | L, M, X |

The classes are C in {1, 3, 13, 15–65, 96, 256, 2048}, spatial in {1, 7, 15, 17, 55, 56, 112,
2048} and N in {1, 2, 3, 8, 32, 64}. The boundaries are N = 1, one element per channel, tiny
variance, rank 2 and rank 6; the hazards are C either side of 16, 32 and 64, odd spatial extents
and non-packed strides.

**Defaults are non-degenerate.** Every S and L default has N, C and every spatial dim above 1. N is
2 rather than 1, so that no default dimension collapses to 1 (RFC 0022 §7); S2 and S9 keep the
batch-1 cases.

### 3.1 Small (S)

Defaults: N 2, C 16; spatial 64 (1D), 16×16 (2D), 8×8×8 (3D).

| Id | Features | Change from the defaults | Ranks |
|---|---|---|---|
| S1 | core | none | 1D 2D 3D |
| S2 | F2 | N 1 | 1D 2D 3D |
| S3 | F3 | C 1 | 1D 2D 3D |
| S4 | F3 | C 3 | 1D 2D 3D |
| S5 | F3 | C 13 | 1D 2D 3D |
| S6 | F4 odd | spatial 7 per dimension | 1D 2D 3D |
| S7a, S7b | F4 one axis 1 | S7a: 16×1 (2D), 8×1×8 (3D). S7b: 1×16. | S7a 2D 3D; S7b 2D |
| S8 | F4 | spatial 1 per dimension | 1D 2D 3D |
| S9 | F5 | N 1, C 256, spatial 1 per dimension | 1D 2D 3D |

### 3.2 Medium (L)

| Rank | N | C | Spatial |
|---|---|---|---|
| 1D | 8 | 64 | 2048 |
| 2D | 8 | 64 | 56² |
| 3D | 4 | 32 | 8×28² |

L1 has the base size. L4 and L5 set C to 3 and 13; L6 sets odd spatial (2047, 55², 7×27²). Size:
about 0.1–1.6M elements per case.

### 3.3 Model (M)

| Id | Source | N, C, spatial | Elements |
|---|---|---|---|
| M1 | ResNet-50 stem batchnorm | 8, 64, 112² | 6.4M |
| M2 | ResNet-50 stage-1 batchnorm | 8, 256, 56² | 6.4M |
| M3 | ResNet-50 stage-4 batchnorm | 32, 2048, 7² | 3.2M |
| M4 | MobileNetV2 depthwise-block batchnorm | 8, 96, 56² | 2.4M |
| M5 | 3D U-Net batchnorm | 2, 32, 32³ | 2.1M |
| M6 | I3D batchnorm | 2, 64, 8×56² | 3.2M |
| M7 | Conformer convolution-module batchnorm (1D) | 8, 256, 128 | 262K |

These are published layer sizes chosen by hand. Whether the
[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking) corpus records batchnorm geometries
that could replace them with a pinned R set, as the convolution plan does, is to be confirmed in
review.

### 3.4 Hazard (H)

One change from the S defaults. Channel counts either side of the common vector and workgroup
widths, odd spatial extents, and batch sizes that real models use.

| Id | Change |
|---|---|
| H1–H6 | C 15, 17, 31, 33, 63, 65 |
| H7, H8 | spatial 15, 17 per dimension |
| H9 | N 3 |
| H10 | N 64 |
| H11 | C 2048, spatial 1 per dimension (wide channel, nothing else to index) |

### 3.5 Stress (X)

| Id | Shape | Features |
|---|---|---|
| X1 | M1 at N 32 (25.7M elements) | F2 |
| X2 | M2 at N 32 (25.7M elements), channels-last | F2, F7 |
| X3 | N 1, C 1, 46341² (2,147,488,281 elements > 2³¹); fp16 and bf16 | F11 |
| X4 | 1D, N 1, C 16, 1,048,576 (16.8M elements) | F4 |

## 4. Building the sweep

### 4.1 Level contents

- **Core axes:** rank (3) × layout (2) × io data type (3) = 18 cells, fp32 statistics.
- **Feature axes:** everything in §2 that is not core.

| Rule | Meaning |
|---|---|
| Core cover | Quick holds every core cell on S1 for `plain` and for `relu` with plain ReLU (36 cells). At least one engine must execute each cell in quick (RFC 0022 §8.1). |
| One feature at a time | Each S and L shape changes one feature from the defaults, so a failure points at that feature. |
| Size grows with level | Each level has a size ceiling (§3). PR proves a feature works; nightly proves it still works at medium size. |
| Pair cover | The level rules loop over rank, layout and data type and rotate the other values by position (`PAIRS[(i + offset) % 6]` over the six layout × data type pairs, `UPAIRS[…]` over the four upcast pairs, `ACTS[k % 4]`). By the end of nightly every S shape has every pair of its rank, every upcast pair, and every claimed ReLU variant. |
| Full | Complete products only where two features plausibly interact: ReLU variant × L × every pair, M × every pair. §4.3 lists what is deliberately not run. |
| Probes | Every unsupported feature and awkward combination has a probe (§4.2). Every level from quick to nightly holds rejection probes; full holds X3. |
| Regressions | None pinned yet. |
| Concrete graphs | Each level runs checked-in graphs only. The generator is deterministic; random sampling belongs to the exhaustive tool (§7). |
| Unsupported stays in | A feature that no engine supports today is still in the sweep. It is unclaimed and skipped, and it shows up as `unclaimed_support` once an engine adds it. |

Each level answers one question (RFC 0022 §8.2). A case sits in the lowest level that holds its
graph; the levels cascade through each provider's tier YAML.

| Feature group | quick — what catches gross breakage? | PR — what must every PR test? | nightly — wider per engine | full — wider sweeps |
|---|---|---|---|---|
| Shape size | ≤ 64K elements | ≤ 1M | ≤ 8M | unbounded |
| Plain core (18) | S1 × every cell | every other S shape × rank once, pair rotated | every S shape × every pair of its rank | — |
| F14 upcast | S1 × rank (fp16 channels-first, bf16 channels-last) | S4, S6, S9 × rank, upcast pair rotated | every S shape × every upcast pair | L × every upcast pair |
| L | — | — | L1, L4–L6 × every pair | (upcast pairs above) |
| H hazards | — | each H once (2D), pair rotated | each H × every pair (2D); 1D and 3D × 2 pairs | — |
| Model (M) | — | M7 × 2 pairs | M1–M6 × 2 pairs; M7 × every pair | M1–M7 × every pair |
| Stress (X) | — | — | — | X1, X2, X4 × 3 data types; X3 fp16 and bf16 (probe) |
| F17 values | default | constant × rank; zero scale (2D) | tiny variance and large magnitude × rank × data type; cancellation (2D) × data type | — |
| Rejection probes (`plain`) | F15 (2D fp16); F9 `np_row` (2D) | F15 × rank × 2; F16 fp16 and bf16 (2D); F9 `np_row` and `np_chan` × rank; F8 × rank; F1 rank 2 × 3 data types and rank 6; F12 | F15 and F16 × rank × layout × (fp16, bf16); F9 and F8 × rank × data type | X3 |
| Fused core (18) | S1 × every cell, plain ReLU | — | — | — |
| F20 claimed clips (`relu_lo0`, ReLU6, clamp) | each clip once per rank, pair rotated (9) | each clip × every core cell; `relu_slope0` (2D, fp32 and fp16) | every S shape × every pair, ReLU variant rotated | L × every pair × every claimed variant; X1 fp16 × every claimed variant |
| Fused shapes and upcast | — | S1 upcast × rank; S4, S6, S9 × rank, variant rotated | L1 × every pair; L4–L6 once; M1, M2, M5 × 2 pairs | — |
| Fused probes | leaky; `relu_lo0p1` (2D) | leaky and `relu_lo0p1` × rank; F21 fp16 and bf16; F15 fused | leaky × every core cell; `relu_lo0p1` × rank × data type | — |
| Golden (§5) | G1–G7, G11 | G8–G10, G12, G15 | G13, G14 | — |

Cases generated (own level / cumulative), estimated from the rules above:

| Level | Plain | Fused | Own | Cumulative | Claim | Probe | Regression |
|---|---|---|---|---|---|---|---|
| quick | 26 | 29 | **55** | 55 | 51 | 4 | 0 |
| PR | 70 | 69 | **139** | 194 | 112 | 27 | 0 |
| nightly | 455 | 198 | **653** | 847 | 598 | 55 | 0 |
| full | 83 | 265 | **348** | 1,195 | 346 | 2 | 0 |

By size class: quick S 55. PR S 126, H 11, M 2. Nightly S 433, L 99, H 99, M 22. Full L 309, M 24,
X 15. The captured sweeps hold 460 cases in the same two directories today. Quick grows from 25
to 55 cases so that it executes every core cell; the timing run confirms the budget. Budgets from
RFC 0022 §8.2 apply to the suite, not this op alone.

**Files.**

- **Topology:** `{level}/BatchnormInference/Default/graph.template.json` and `{level}/BatchnormInferencePointwise/Default/graph.template.json`, written by the generator.
- **Cases, purpose, plan items:** `sweep.json` per level, written by the generator.
- **Golden:** `golden/{case-id}/tensors.dvc`, written by the golden generator.
- **Claims:** `support.json`, harvested with `--write-support-claims`, never hand-written.
- **Tolerances and skips:** [MIOPEN_ENGINE.toml](../../../miopen-provider/config/MIOPEN_ENGINE.toml) and [HIP_MLOPS_ENGINE.toml](../../../hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml), only from observed error. No entry names a `BatchnormInference` case id today.
- **Lanes:** [test_categories_integration.yaml](../../../miopen-provider/test_categories_integration.yaml) and [HIP_MLOPS_ENGINE_test_categories_integration.yaml](../../../hip-kernel-provider/HIP_MLOPS_ENGINE_test_categories_integration.yaml). No batchnorm inference pins today.
- **CMake:** `dnn-providers/miopen-provider/CMakeLists.txt` sets `BENCHMARK_REFERENCE_SMOKE_FILTER "quick_BatchnormFwdInference_Inference.*"`; it moves to the new quick ids when `BatchnormFwdInference/` is retired.

**Case ids.** The scheme is
`<plan items>_<rank>d_i<spatial>_n<N>c<C>_<dtype>[_p<dtype>][_y<dtype>]_<layout>[_<activation>][_<value>]`.
`_p` names a statistic type different from fp32 (F15), `_y` a y type different from x (upcast), and
the layout is `cf` or `cl` (`xcfycl` for F8). Ids are lowercase, as the bundle verifier requires,
and an id that collides inside one sweep gets a `_<hash6>` suffix. Input seeds come from the id, so
an id's data does not change between runs. Every captured case id is retired; for each removed
case the PR records which plan item now holds its behaviour, or which row of §4.3 drops it. A
generator change that renames ids updates the TOML globs, the tier YAML and the CMake filter in the
same pull request.

### 4.2 Probes and regressions

| Purpose | Expected outcome | Claims |
|---|---|---|
| `claim` | A claiming engine accepts and is correct. | `--write-support-claims` records them. |
| `probe` | A typed rejection, or a correct accept. | Excluded from claims; the harvest filter is not yet in the repository (I2, as for the convolution plan). |
| `regression` | Same as the original bug case. | Same as the original case. |
| `expected_rejection` | — | None: every rejection for this op is a frontend rule (§2.7, D1). |

A case is a **probe** when it has at least one of these features. The generator writes the reasons
into `metadata.notes`:

| Probe class | Items |
|---|---|
| Unsupported feature | F1 rank 2 and rank 6; F15 statistics in the io type; F16 fp16 or bf16 compute; F20 leaky and `relu_lo0p1` |
| Awkward combination | F8 mixed layouts; F9 non-packed strides; F12 statistic rank below x rank; F21 io-type intermediate |
| Size | X3 (F11) |

- **claim:** ranks 3–5, both layouts, fp32, fp16 and bf16, upcast, the S, L, M and H shapes, X1, X2 and X4, every value regime of F17, plain ReLU, `relu_lo0`, ReLU6, clamp and `relu_slope0`, and the golden cases. Hazard shapes are claims here, not probes as in the convolution plan (D2).
- **`relu_slope0`** is a claim: an explicit slope of 0 means plain ReLU. The engines disagree on it today (below); the case gives the accept side of that check a bundle.
- **regression:** none yet. The exhaustive tool or an escaped bug adds them as concrete cases in the lowest affordable level, written after the rotated cases so they move no other case id.
- When an engine adds a probe feature, the run reports `unclaimed_support`; the plan then moves those cases to `claim` and the generator is edited, not the bundles.

**From plan to engine.** No engine is named in the plan. What the engines accept, read from
source and not yet measured, is an input to the claim harvest only:

- Both engines require packed tensors with one layout, fp32 statistics and affine tensors, and fp32 intermediates, and both decline leaky and `relu_lo0p1`.
- MIOpen also requires x and y in the same type (no upcast), and statistic and affine tensors of the same rank as x (`getDerivedShape`), so it declines F12. Its rank ≥ 3 and `N × spatial > 1` checks (`validateSpatialDimensions`) run for training only, so the rank-2 probes (F1) decide what it does with `[N, C]`.
- On a slope of exactly 0, MIOpen treats it as no leaky ramp (`MiopenUtils.hpp`); `HIP_MLOPS_ENGINE` declines any graph whose slope is set (`BatchnormApplicabilityChecks.cpp`).

Those limits show up as skips and harvested claims, not as plan edits. In the captured sweeps,
`MIOPEN_ENGINE` claims exactly the cases that are neither upcast nor leaky, and `HIP_MLOPS_ENGINE`
every case that is not leaky, including the 1D layouts and `1 × 256 × 1 × 1`.

### 4.3 Products deliberately not run

| Id | Product not run | Cases it would add | Why it adds little | Covered instead by | Add it back when |
|---|---|---|---|---|---|
| T1 | Every claimed ReLU variant × every S and M shape × every pair | ~700 | The epilogue is applied element by element to y. It depends on the data type, the layout and the fusion path, not on the shape. | Every variant × every core cell (PR); variant rotated over every S shape × every pair (nightly); L × every pair × every variant (full). | A defect needs a variant together with a specific shape. |
| T2 | Upcast × every S shape before nightly | ~50 | Upcast changes the output type, which every rank and layout shows. | S1 × rank (quick), S4, S6, S9 (PR), every S shape × every upcast pair (nightly). | An upcast defect depends on a shape class. |
| T3 | F15 and F16 × every S shape | ~200 | No engine accepts them; each case repeats one decline. | S1 × rank × layout × (fp16, bf16) in nightly. | An engine claims the type: then run it × core × S shapes. |
| T4 | F8 and F9 × every S shape | ~225 | Layout and stride handling is per tensor, not per shape. No engine accepts them. | S1 × rank × data type in nightly. | An engine accepts them, or a stride defect depends on a shape. |
| T5 | Leaky × every S, L and M shape | ~260 | Both engines decline any non-zero slope; each case repeats one decline. | Leaky × every core cell (nightly) and once per rank in PR. | An engine executes leaky: the references already model the slope. |
| T6 | H × every pair in 1D and 3D | 88 | Hazards act on N, C and the spatial extent; layout and data type multiply runs, not hazard coverage. | 2D × every pair; 1D and 3D × 2 pairs (nightly). | A hazard is found that depends on rank. |
| T7 | Any 3-way product beyond variant × layout × data type | not counted | Pairwise cover catches most interaction faults at a fraction of the cases. | Pairwise (§4.1). | A 3-way fault is found. Add that triple. |
| T8 | A separate `BatchnormFwdInference` sweep | 11 sweep cases and 6 single-graph bundles today | Same graph as `BatchnormInference/Default` under different uids. | The golden cases of §5, on the generated sweep. | Reviewers want a twin directory. |
| T9 | F22 epilogues other than `relu_fwd` | not counted | The graph template holds a `relu_fwd` node; both engines fuse only the ReLU family after batchnorm. | `plain` covers the batchnorm itself. | An engine fuses another epilogue: add a topology and its template. |
| T10 | Today's full-tier clip products in `full/BatchnormInferencePointwise` (135 cases) | retired | They mostly repeat one check. | Pair cover by nightly and variant × L in full. | A clip regression escapes nightly. |

Trims run after case ids are assigned, so a trim renames no case, and no trim removes the last
probe of a feature from any level.

**Deviations from RFC 0022.**

| Id | RFC rule | This plan | Why |
|---|---|---|---|
| D1 | §6: each rejection an engine can see is an `expected_rejection` bundle | The invalid problems of §2.7 are frontend unit tests. The harness never calls the frontend's `validate()`, so an engine that accepts an illegal graph (for example `HIP_MLOPS_ENGINE` with bare `[C]` scale and bias) is not caught by a bundle. | Same gap as the convolution plan's D1 (ALMIOPEN-3232). |
| D2 | §9 (as applied by the convolution plan): hazard shapes are probes | H shapes, S8, S9 and X1, X2 and X4 are claims. | The op is elementwise: an engine that accepts the op is expected to run any legal shape, and today's `support.json` already claims `1 × 256 × 1 × 1` for both engines. |

### 4.4 Cost

| Estimate | Value |
|---|---|
| Elements per case | quick ≤ 16K; PR ≤ 262K (M7); nightly ≤ 6.4M (M1, M2); full ≤ 25.7M (X1, X2), plus X3 at 2.1G (probe) |
| Oracle | GPU reference first. The CPU reference is parallel but costs seconds on M and X shapes, so X shapes must find a GPU reference. |
| Compile | Paid once per unique shape on a cold cache if the engine compiles per shape. Not measured for batchnorm; the timing run (I9) decides whether the counts fit the budgets. |
| Device memory | X3 needs about 8.6 GB in fp16. The generator sets `minimum_vram_mb` on every case whose tensors exceed 1 GiB. |

### 4.5 Measured run

None yet for this sweep. The cold-cache timing and the claim harvest run on each claimed
architecture once the bundles are generated (I9); the claims of the captured sweeps span gfx90a,
gfx942, gfx950, gfx1030, gfx1100, gfx1101, gfx1150, gfx1151, gfx1200 and gfx1201.

The captured sweeps this plan replaces are green: TheRock's nightly test run on gfx94X of
2026-11-19 ([rockrel run 44450853090](https://github.com/ROCm/rockrel/actions/runs/44450853090),
quick through comprehensive) ran their 276 cases below full. `HIP_MLOPS_ENGINE` passed 276;
`MIOPEN_ENGINE` passed 168 and skipped 108, each a case it does not claim. No case failed, no claim
broke, and no case reported `unclaimed_support`. Their 195 full cases have no measured run.

## 5. Golden reference

Policy: baseline, not exhaustive. Golden cases are chosen by the criteria in
[RFC 0022 §12.2](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#122-policy-and-selection-criteria).
Each one is generated by PyTorch and fits the 2 MiB per-case budget with finite values. **G-A
covers every rank × layout family once with a rotating data type, so each data type appears twice,
and each fused form once.** Each golden case is also an ordinary claim case, so no level that runs
without `dvc pull` loses coverage.

| Id | Graph | Bundle | Case | Criterion | Level |
|---|---|---|---|---|---|
| G1 | 1D NCL fp32, S1 | `BatchnormInference/Default` | `s1_1d_i64_n2c16_fp32_cf` | G-A baseline | quick |
| G2 | 1D NLC fp16, S1 | `BatchnormInference/Default` | `s1_1d_i64_n2c16_fp16_cl` | G-A baseline | quick |
| G3 | 2D NCHW bf16, S1 | `BatchnormInference/Default` | `s1_2d_i16x16_n2c16_bf16_cf` | G-A baseline | quick |
| G4 | 2D NHWC fp32, S1 | `BatchnormInference/Default` | `s1_2d_i16x16_n2c16_fp32_cl` | G-A baseline | quick |
| G5 | 3D NCDHW fp16, S1 | `BatchnormInference/Default` | `s1_3d_i8x8x8_n2c16_fp16_cf` | G-A baseline | quick |
| G6 | 3D NDHWC bf16, S1 | `BatchnormInference/Default` | `s1_3d_i8x8x8_n2c16_bf16_cl` | G-A baseline | quick |
| G7 | → ReLU, 2D NCHW fp32, S1 | `BatchnormInferencePointwise/Default` | `s1_2d_i16x16_n2c16_fp32_cf_relu` | G-A fused baseline | quick |
| G8 | → ReLU6, 2D NHWC fp16, S1 | `BatchnormInferencePointwise/Default` | `s1_2d_i16x16_n2c16_fp16_cl_relu6` | G-C upper-clip semantics | PR |
| G9 | → clamp, 1D NCL bf16, S1 | `BatchnormInferencePointwise/Default` | `s1_1d_i64_n2c16_bf16_cf_clamp` | G-C lower and upper clip semantics | PR |
| G10 | → ReLU with slope 0, 2D NCHW fp32, S1 | `BatchnormInferencePointwise/Default` | `s1_2d_i16x16_n2c16_fp32_cf_relu_slope0` | G-C: an explicit zero slope is plain ReLU | PR |
| G11 | upcast fp16 → fp32, 2D NCHW, S1 | `BatchnormInference/Default` | `s1_2d_i16x16_n2c16_fp16_yfp32_cf` | G-B: the upcast signatures have no independent check | quick |
| G12 | 2D NHWC bf16, S9 (`1 × 256 × 1 × 1`) | `BatchnormInference/Default` | `s9_2d_i1x1_n1c256_bf16_cl` | G-C: one element per channel | PR |
| G13 | 2D NCHW fp16, S1, tiny variance | `BatchnormInference/Default` | `s1_2d_i16x16_n2c16_fp16_cf_tinyvar` | G-D: large inv_variance | nightly |
| G14 | 2D NHWC bf16, S1, large magnitude | `BatchnormInference/Default` | `s1_2d_i16x16_n2c16_bf16_cl_large` | G-D: outputs near the type's range | nightly |
| G15 | 1D NCL fp16, M7 (Conformer; 1.0 MiB) | `BatchnormInference/Default` | `m7_1d_i128_n8c256_fp16_cf` | G-F production anchor | PR |

Fifteen cases, inside the 6–20 range of RFC 0022 §12.2.

- **Generator.** `generate_batchnorm_inference_golden.py` computes the bundle inputs with `torch.nn.functional.batch_norm(x, mean, running_var, scale, bias, training=False, eps=0)` on CPU, with `running_var = inv_variance ** -2`, then applies the ReLU variant for the fused cases. It does not reuse the conversion in [utilities/batchnorm_inference.py](../../reference-data-scripts/utilities/batchnorm_inference.py), which computes `1 / inv_variance² − 1e-5` in the statistics' own type and relies on `batch_norm`'s default `eps = 1e-5` to add it back. With fp32 statistics that round trip is accurate (relative error about 1e-7, so G13 is safe either way). With fp16 statistics `inv_variance²` overflows from about 256, `variance + eps` becomes 0 and the recovered inv_variance is infinite; with bf16 statistics the error reaches about 0.7 %. Converting in fp64 with `eps = 0` avoids both, which matters once F15 cases get golden data.
- **Precision strategy.** Precision uplift (fp64, rounded once to the output type), the RFC 0022 §12.3 default. With no accumulation, matched precision would differ by at most one rounding, so the plan does not deviate. Pin the PyTorch version and seed in `meta.json`.
- **Confirmation.** `hipdnn_golden_data_tests --reference both` passes for every case, and an independent NumPy fp64 evaluation of the formula in §1 matches the PyTorch output before commit. Never regenerate from a hipDNN reference.
- **Storage.** `golden/<case-id>/tensors.dvc` beside the sweep, on the `golden-data` DVC remote.
- **Existing golden data.** `BatchnormFwdInference/Inference` holds six golden sweep cases and the single-graph bundles hold six more, all marked `reference_source: PyTorch 2.12.0+rocm7.2`. Their notes say "Dummy metadata (placeholder)" and their generator (`generate_batchnorm_fwd_golden.py`) is not in the tree, so they cannot be reproduced or counted as third-party confirmation. They are retired with T8 once G1–G15 pass.

## 6. Pass criteria and reference prerequisites

`auto` order: golden, GPU reference, CPU reference. See
[Verification modes](../running-tests.md#verification-modes).

- **GPU reference:** BatchNorm Forward Inference, stride-based, seven signatures with fp32 compute (§2.3); Pointwise `RELU_FWD` takes the lower clip, upper clip and slope ([OperationSupport-GpuReferenceImpl.md](../OperationSupport-GpuReferenceImpl.md), `GpuPointwisePlan.hpp`).
- **CPU reference:** BatchNorm Forward Inference, nine signatures including fp16 and bf16 compute; the plan builder checks rank ≥ 2 only, while [OperationSupport-ReferenceImpl.md](../../../../projects/hipdnn/docs/OperationSupport-ReferenceImpl.md) lists NCHW, NHWC, NCDHW and NDHWC. Pointwise `RELU_FWD` takes all three fields (`ReferencePointwiseBase.hpp`).

| Item | Rule |
|---|---|
| Verification | `auto`; a case with no oracle fails as unverifiable. |
| Tolerance | `ToleranceResolver` uses `tol::batchnorm::getToleranceInference<T>()` in `hipdnn_test_sdk/utilities/TestTolerances.hpp`: fp32 2e-4, fp16 5e-4, bf16 8e-3. For the fused graph the maximum across nodes applies. A dynamic inference tolerance exists (`DynamicTolerancesBatchNorm.hpp`, `calculateBatchnormInferenceTolerance`) but the harness does not use it; [Tolerances](../tolerances.md) describes the harness's dynamic tolerance, which covers convolution forward only. The large-magnitude cases of F17 may exceed the static fp16 value; calibrate from observed error on the golden cases, put overrides in the engine TOML, and never loosen a tolerance to make a case pass. Whether to use the dynamic inference tolerance instead is to be decided in review. |
| Support claims | Enforced. See [support-claim-enforcement.md](../support-claim-enforcement.md). Claims harvested per engine and arch; probes never claimed. |
| Quick gate | All 36 core cells executed and matched on at least one engine. |
| Traceability | Generator `--report` enforces two-way traceability. |

Gaps and fixes. The generator emits these cases now; they are unclaimed until the work lands. The
ids are those of RFC 0022 §15.

| Feature | Gap | Needed | RFC 0022 §15 |
|---|---|---|---|
| F1 rank 3 and rank 6 on the CPU reference | The document lists 4D and 5D layouts only; the plan builder accepts any rank ≥ 2 and indexes the statistics by `(0, c)`. | Confirm on a 1D bundle, then fix the document or the builder. | I6 |
| F11 64-bit indexing | Not checked whether the GPU reference kernel indexes with 64 bits. | Read `GpuFpReferenceBatchnorm::fwdInference`; fix if needed. | I6 |
| F12 statistic rank below x rank | Both references index the statistics through their own dims and strides; not checked with `[1, C]` against a rank-4 x. | Confirm both references before the probe can pass. | I6 |
| F16 fp16 and bf16 compute | Only the CPU reference registers these signatures. | Add them to the GPU registry, or accept the CPU cost. | I6 |
| F17 values | The fill recipes `free` and `fixed` are enough for every regime listed. | — | I4 |
| F18 NaN / Inf | The fill recipes cannot place special values, and the default comparison treats NaN as a mismatch. | I4, plus `allclose_matching_infinities` on those cases. | I4 |
| F10 alignment | The harness passes allocator pointers aligned to at least 256 bytes. | Exact-alignment and offset placement. | I3 |
| F19, §2.6 | Bundles run one execution. | Repeat-and-compare on bundles. Until then, C++ tests. | I5 |
| Generator | None for batchnorm. | `generate_batchnorm_inference_bundles.py`, following `generate_conv_fwd_bundles.py`. The templates already expose the three ReLU fields, and per-tensor data types allow F15, F16 and F21. | I1 |
| Golden | No golden generator for the sweep; `generate_batchnorm_reference.py` writes one golden graph, and the existing helper's variance conversion is lossy for fp16 and bf16 statistics with large inv_variance (§5). | `generate_batchnorm_inference_golden.py`. | I7 |
| `purpose` | The harness ignores the tag, and `--write-support-claims` records every accepted case, probes included. | A harvester filter in the repository. | I2 |
| Timing | Not run. | Cold-cache timing on the claimed architectures before the counts are locked. | I9 |

The gaps this plan logs instead of covering (F6 inferred outputs, F10, F18, F19, F22) are linked
from [KNOWN_TESTING_GAPS.md](../../../../projects/hipdnn/docs/testing/KNOWN_TESTING_GAPS.md) in the
same PR as the bundles.

## 7. Exhaustive mode

Batchnorm inference has an exhaustive mode. It is a separate tool (RFC 0022 §14); this plan only
declares what the tool reads. The declarations live in
`reference-data-scripts/batchnorm_inference_plan.py`, the same module the bundle generator reads,
so there is one copy of the legality rules.

| Declaration | Where | Content |
|---|---|---|
| Axes | `RANKS`, `LAYOUTS`, `CORE_DTYPES`, `SIGNATURES`, `ACTS`, `TOPOLOGIES`, and the case fields of `DEFAULTS` | every feature of §2 |
| Legality | `legal()` | the frontend validation of §1 and §2.7 |
| Shape ranges | — | rank 2–6; N 1–256; C 1–4096; spatial 1–2048 per dimension (1D up to 2²⁰); size budget x + y ≤ 1 GiB |
| Value domains | `value_recipes()` | uniform; constant; zero scale; tiny variance; large magnitude; cancellation; NaN, Inf and subnormal once I4 lands |
| Return path | — | minimized reproducers become `regression` cases written by the generator; reference disagreements become G-B golden candidates |

## 8. How the sweep was built

This section states the method of this plan. The general rules (the authoring process, sweep
rules, levels, purpose, trade-off register, oracles, golden tests and review checklist) are in
[RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md).

- **Inventory.** §2 comes from the graph API sources listed at the top of this plan: each field of the batchnorm and pointwise attributes and each frontend validation rule is a feature (RFC 0022 §6). The reference signature registries set the data type axes. Existing bundles and engine support do not decide what is listed.
- **Defaults.** The S and L defaults make every dimension that can exceed 1 exceed 1 (§3).
- **Construction.** The level rules build each level from loops over rank, layout and data type and from positional rotation of the other values (§4.1). There is no all-pairs construction or selection seed. Levels are built in order, and a graph that a lower level or an earlier rule already holds is not written again.
- **Sizing.** Element ceilings bound each shape; the op's cost is per element, and compile time per unique shape is not yet measured (§4.4).
- **Trims.** T3, T4 and T5 rest on what the engines decline today; the others follow from the op being elementwise (§4.3).
- **Oracles.** Each feature was checked against both reference registries before it was listed (§6).
- **Golden.** §5 selects cases by the criteria of RFC 0022 §12.2; none exists yet.

Review checklist (RFC 0022 §13.3):

- [x] Each feature traces to a schema field or validation rule.
- [x] Each tensor has dtype, layout, stride, kind and alignment axes; legality declared.
- [x] No degenerate default.
- [ ] Each case has a purpose; quick executes each core cell; nightly retests each feature at L. Purposes are declared; execution needs a run.
- [x] Each unrun product is in the register; each level has rejection probes.
- [x] Each oracle gap listed with its fix.
- [ ] Counts fit budgets after timing.
- [ ] Golden follows §5 and is confirmed.
- [x] Exhaustive section present.
- [ ] Bundles and plan items map both ways (`--report`). Needs the generator.
