# Convolution Backward-Data Test Plan

This plan lists every feature of convolution backward data (dgrad,
`ConvolutionBwdAttributes`) and its pointwise fusions that a complete test sweep
must exercise, and how the sweep is spread across levels. It does not depend on
any engine: the sweep describes the problem space, and each engine's support
claims decide which cases it runs. Forward convolution and weight gradient have
their own plans ([convolution-fwd.md](convolution-fwd.md), `convolution-wrw.md`).

The plan follows RFC 0021 (Per-Op Test Plans) and the [methodology](methodology.md).
See [the test-plans index](README.md) for the level definitions.

Dgrad shares its parameters with the forward convolution, but not its
problem. A dgrad kernel reduces over K, writes C, and has to work out which
filter taps reach each dx element. Those taps depend on how stride and dilation
split the filter into stride phases. The engines pick their kernels on exactly
these properties: per-group C and K, stride phases, padding seen as a transposed
convolution, and the extent of dy. This plan's catalog is built from them (§2.1, §3)
and from the corpus of dgrad graphs real models run (§3.3). It does not reuse the
forward plan's shapes.

The bundles are **generated** from this plan. Nobody edits them by hand:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_conv_bwd_bundles.py \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles --report
```

| Plan item | Where it lives |
|---|---|
| Case model, legality, behaviour classes, shape catalog, ceilings, data types, activations, value domains, purpose rule, trimmed products (`TRIMS`) | `reference-data-scripts/conv_bwd_plan.py` |
| R set: corpus pick, selection rule, provenance | `reference-data-scripts/conv_bwd_corpus.json` |
| Level rules, graph templates, sweep writer, purpose tags, trims | `reference-data-scripts/generate_conv_bwd_bundles.py` |
| Bundles | `integration-test-bundles/{quick,standard,comprehensive,full}/ConvolutionBwd/Default` and `ConvolutionBwdPointwise/{Default,Variant2..Variant4}` |

Every sweep case records the plan items it implements (`metadata.plan_items`),
its `purpose` (§4.2) and its size class. A plan item names its cases with a
filter on `plan_items`, and a case names its items.

The features come from the graph API:

- `flatbuffers_sdk/schemas/convolution_bwd_attributes.fbs` and `convolution_common.fbs`
- `tensor_attributes.fbs`, `pointwise_attributes.fbs`, `data_types.fbs`, `graph.fbs`
- the frontend rules in `hipdnn_frontend/node/ConvolutionDgradNode.hpp`

---

## 1. The operation

Dgrad is the gradient of `y = conv(x, w)` with respect to x. For each dx element
it sums over the K/G dy channels of its group and over the filter taps whose
window covers it:

```
dx[n, c, i...] = Σ_{k in group(c)} Σ_{r...} dy[n, k, (i + pre − r·dil) / stride] · w[k, c', r...]
                 (only the terms where the division is exact and the index lies inside dy)
```

| Tensor | Dims | Rule |
|---|---|---|
| dy | N, K, out… | input; at least 3 dims; `dy[1] == w[0]` |
| w | K, C/G, filter… | input; same rank as dy; `dx[1] % w[1] == 0`, so G = C / w[1]; `K % G == 0` |
| dx | N, C, in… | output; `dx[0] == dy[0]`; for every spatial dim `out == floor((in + pre + post − (dil·(f − 1) + 1)) / stride) + 1`. **dx dims are never inferred**: the frontend rejects a graph without them (`ATTRIBUTE_NOT_SET`). Only dx strides are inferred, from dy's stride order. |

Parameters: `pre_padding`, `post_padding`, `stride` and `dilation`, one value
per spatial dimension, and `conv_mode` (`CROSS_CORRELATION` or `CONVOLUTION`).
The node `compute_data_type` and the graph's io, intermediate and compute data
types are set separately.

Three quantities follow from the formula and drive both the kernels and the
plan:

- **Reduction length.** A dx element sums `A = K/G × T` products, where T is the
  number of taps that reach it: along a dimension with stride s and dilation d a
  dx position meets one tap in every `s / gcd(s, d)`, so
  `T = Π ceil(f / (s / gcd(s, d)))`. The reduction runs over K, not C.
- **Stride phases.** With stride s > 1 the dx positions fall into s phases per
  dimension, and each phase uses its own subset of the filter taps. When
  `f % s ≠ 0` the phases have different tap counts; when `f < s`, or when s and d
  share a factor, some phases have no taps at all and their dx elements are 0.
- **Transposed padding.** Read as a transposed convolution over dy, the padding
  is `d·(f − 1) − pre`. At 0 ("full") the dx border sums fewer taps than the
  interior; when padding exceeds the dilated filter it is negative, and dy's
  border rows reach no dx element.

---

## 2. Feature inventory

Each feature lists the values a complete sweep must hit. The **Oracle** column
says whether the verification references can judge the feature today (§6); the
**Earliest level** column is the lowest level that holds a case of it (`Q`
quick, `PR`, `N` nightly, `F` full; `—` means the sweep cannot express it yet).

**Oracle.** There is **no GPU reference for dgrad**
(`GpuReferenceGraphExecutor.cpp` declines `ConvolutionBwdAttributes`). Every
dgrad case is judged by the CPU reference (`CpuFpReferenceConvolution::dgrad`),
which registers dy, w, dx = fp32/fp32/fp32, fp16/fp16/fp16 and bf16/bf16/bf16
with fp32 compute, addresses every tensor through its own strides, and accepts
only `CROSS_CORRELATION`. A case the engine accepts and no oracle can judge
**fails** ("no oracle can verify this bundle"), so oracle gaps show up as
failures, not skips.

**Legality** is declared once, in `conv_bwd_plan.legal()`, and copies
`ConvolutionDgradNode::pre_validate_node`: equal ranks, `C % G == 0`,
`K % G == 0`, one parameter per spatial dimension, stride and dilation ≥ 1,
padding ≥ 0, and a dilated filter no larger than the padded dx. dy's dims are
computed from the output formula, so its check holds by construction. The
generator never writes an illegal graph.

### 2.1 Geometry and data flow

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F1 | Spatial rank | 1D, 2D, 3D; 4D (6-dim tensors) as a probe | CPU ref | Q (1D–3D), PR (4D) |
| F2 | Batch N | 1, 2, 3, 8, 32, 64, 128, 256 | CPU ref | Q |
| F3 | Output channels C/G (what dgrad writes) | 1, 3, 4, odd below 16 (5, 7, 13), 8–32, 40–256, > 256, not a multiple of 8 | CPU ref | Q |
| F4 | Reduction channels K/G (what dgrad sums) | 1, 2, 3, 4, 7, odd (17, 21), 8–32, 40–256, > 256 (320, 2048) | CPU ref | Q |
| F5 | Channel ratio | K/G = C/G; K/G = 2·C/G and 4·C/G (reduction-heavy: the backward of a 1×1 expansion); C/G = 4·K/G (output-heavy); small C/G with large K/G (4 / 320: latent `conv_in`); large C/G with tiny K/G (128 / 3, 2, 4, 21: output heads) | CPU ref | Q |
| F6 | Groups | dense; grouped with C/G = K/G = 4, 8, 16, 24, 32 (ResNeXt); grouped with K/G ≠ C/G (8 / 16, 4 / 320, 8 / 1); group counts 2, 3, 4, 5, 8, 16, 32; depthwise multiplier 1, 2, 4 | CPU ref | Q |
| F7 | Filter | 1, 2, 3, 4, 5, 7, 10, 11, 16, 31; non-square 1×3, 3×1, 3×5, 1×3×3, 3×1×1; filter equal to dx; filter larger than dx | CPU ref | Q |
| F8 | Stride phases | s = 1; `f % s = 0` (f = s, f = 2s); `f % s ≠ 0` (3/2, 7/2, 5/2, 16/8); `f < s` (1/2, 3/4); gcd(s, d) > 1 (s 2, d 2); coprime s, d (s 2, d 3); per-dimension stride | CPU ref | Q |
| F9 | Dilation | 1, 2, 3, 5, 12; per dimension | CPU ref | Q |
| F10 | Transposed padding | full (pad 0 on a filter > 1); positive (the usual "same" padding); negative (padding larger than the dilated filter) | CPU ref | Q |
| F11 | Asymmetric padding | pre 0 / post 1 and pre 1 / post 0 with a stride-2 even filter | CPU ref takes pre and post separately; no CPU test covers asymmetric dgrad (I6) | Q (probe) |
| F12 | Output extent | dy spatial 1 (filter equal to dx); dy larger than dx (even filter, pad 1); dx width at most 8 and 4; dx 16×48 with a 5×5 filter; odd wide dx (32×109); 1×1 dx; extreme aspect 1×256. **Inferred dx dims are illegal** (frontend rejects them): a frontend unit test, not a sweep case. | CPU ref | Q |
| F13 | `conv_mode` | `CROSS_CORRELATION`, `CONVOLUTION` | **none**: the CPU ref rejects `CONVOLUTION` for dgrad, there is no GPU ref (I6) | Q (probe) |
| F37 | dx elements no tap reaches | stride phases with no taps (`f < s`, gcd(s, d) > 1); a floor remainder: dx's last rows lie past the last window. The engine must write 0 there; the harness prefills outputs with a NaN sentinel, so an unwritten element fails. | CPU ref | Q |

### 2.2 Memory layout and tensor kind

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F14 | Layout | channels-first and channels-last for each rank (NCL, NLC, NCHW, NHWC, NCDHW, NDHWC), for dy, w and dx together | CPU ref | Q |
| F15 | Mixed layouts | dy, w and dx in different orders: all six non-uniform combinations | CPU ref | PR |
| F16 | Non-packed strides | padded rows (`fast`), padded channel stride (`chan`), padded plane (`plane`) on dy and dx | CPU ref | PR |
| F17 | Alignment | a base pointer at exactly the declared alignment with an offset (the methodology's rule) | **not run**: I3 | — |
| F18 | 64-bit indexing | dx and dy with more than 2³¹ elements (X3) | CPU ref | F |
| F35 | Override shape | `is_override_shape_enabled` on an explicitly shaped graph | CPU ref | PR |

Every tensor is a device tensor: a dgrad input that is a constant or a runtime
scalar is not a valid graph.

### 2.3 Data types

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F19 | IO data type | fp32, fp16, bf16; fp8 (E4M3, E5M2 and the FNUZ forms) with fp32 dx; int8 with int32 dx; double | **none** beyond the core three (I6) | Q (core; fp8 E4M3 and double as probes) |
| F20 | Mixed precision | fp16 or bf16 dy and w with fp32 dx; dy fp16 with w fp32; dy fp32 with w fp16; fp16 compute | **none** (only uniform core triples are registered) | PR |

### 2.4 Values and numerics

Fill recipes are `free` (uniform in [lo, hi]) and `fixed`, per tensor in
`metadata.inputs`, each with its own seed. Every generated case sets its fills:
dy and w are uniform in [−1, 1] by default, so sums cancel.

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F21 | Input distribution | uniform [−1, 1]; zero dy (dx must be exactly 0); constant dy = w = 1 (dx counts the taps, so a wrong phase split is visible as an integer error); large (dy, w uniform in [0, a], `a = √(½ · max / A)`); tiny (fp32/bf16 ±1e-20, fp16 ±3e-3) | CPU ref | PR |
| F22 | Special values | NaN and ±Inf in dy or w | needs I4 and a matching-infinities validator | — |
| F23 | Accumulation length | A from 1 (depthwise, stride 2) to 186,624 (filter larger than dx, 3D); long reductions come from a large K (K 512, A 4,608; K 2048, A 18,432) or a large filter | CPU ref | N |
| F24 | Determinism | bitwise-identical repeats when a deterministic plan is requested | C++ test (I5) | — |

### 2.5 Fusions

The API lets pointwise nodes follow dgrad (an epilogue on dx) or precede it (a
prologue on dy). The backward pointwise modes take the incoming gradient as
`in_0` and the forward activation's input as `in_1`. There is no bias or scale
epilogue: the bias gradient is a reduction (dbias), not a pointwise op on dx.

| Topology | Bundle | Nodes | Backward of |
|---|---|---|---|
| `dact` | `ConvolutionBwdPointwise/Default` | dgrad → act_bwd(·, x) | `act → conv`: the previous layer's activation |
| `pre_dact` | `Variant2` | act_bwd(dy, y) → dgrad | `conv → act`: this layer's activation |
| `add` | `Variant3` | dgrad → add(dx_residual) | a skip connection summed into dx |
| `add_drelu` | `Variant4` | dgrad → add(dx_residual) → relu_bwd(·, x) | a ResNet block |

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F26 | Backward activation | `relu_bwd`: plain, upper clip (ReLU6), leaky slope. Also `sigmoid_bwd`, `tanh_bwd`, `elu_bwd` (α 1), `gelu_bwd`, `gelu_approx_tanh_bwd`, `swish_bwd` (β 1), `softplus_bwd` (β 1) | CPU ref: relu, sigmoid, tanh; **none** for the others (I6) | Q |
| F28 | Residual-gradient add | a second full-size tensor added to dx | CPU ref | Q |
| F29 | Chains | the four topologies | CPU ref (ReLU family) | Q |
| F30 | Materialised intermediate | dgrad's dx (or the prologue's dy) as a second output instead of virtual | CPU ref | PR |
| F31 | Fused intermediate type | virtual intermediate in fp32 vs in the io type | CPU ref | PR |
| F36 | Prologue fusion | a pointwise node feeding dgrad's dy (`pre_dact`) | CPU ref | Q |

### 2.6 Execution

These cannot be expressed as a bundle case and stay C++ tests until the harness
can repeat and compare (I5): workspace limits (F32), plan reuse with new data
and pointers (F33), and serialisation round trips (F34).

### 2.7 Invalid problems

Frontend unit tests, not sweep cases. Each must be rejected:

- dx dims not set (dgrad never infers them);
- `dy[1] ≠ w[0]`, `dx[0] ≠ dy[0]`, `dx[1] % w[1] ≠ 0`, `w[0] % G ≠ 0`;
- stride or dilation < 1, negative padding;
- rank mismatch between dy, w and dx; parameter count ≠ spatial rank;
- dy spatial dims inconsistent with the output formula;
- a dilated filter larger than the padded dx.

---

## 3. Shape catalog

The sweep draws its shapes from seven sets: **S** (small, one feature changed
from the defaults), **L** (the same classes at medium size), **M** (named model
layers), **R** (real dgrad graphs from the corpus), **H** (hazard breakpoints),
**A** (long accumulation) and **X** (stress). Each level raises the size
ceiling:

| Level | Per-case ceiling (fp32 dy + w + dx) | Shape sets |
|---|---|---|
| quick | ≤ 20 MFLOP, ≤ 1 MB | S (3D at 16×8×8 or smaller) |
| PR | ≤ 200 MFLOP, ≤ 16 MB | S, H, R, plus M1, M8, M11, which exceed the ceiling by design |
| nightly | ≤ 8 GFLOP, ≤ 64 MB | S, L, M, R, H, A |
| full | none | S, L, R, X |

FLOP counts 2 × multiply-accumulates. When a 3D S shape does not fit a level at
16³, the generator tries 16×8×8, 16×8×4 and 8×8×4.

### 3.1 Small (S)

Defaults: N 2, C 16, K 32, G 1, spatial 16 per dimension, filter 3, stride 1,
dilation 1, pad 1. N is 2 so that no default dimension is 1; C and K differ so
that the reduction and the output are never the same size by accident. Each
class changes what its row says; ranks are 1D, 2D, 3D unless the row names
fewer. The classes are grouped by the dgrad property they exercise.

**Basics and transposed padding.**

| Id | Change | Ranks | Why for dgrad |
|---|---|---|---|
| S1 | filter 1, pad 0 | 1–3 | a GEMM over K |
| S2 | pad 0 | 1–3 | full transposed padding: the dx border sums fewer taps |
| S3 | — | 1–3 | the default "same" case; feature probes sit on it |
| S4 | filter 5, pad 2 | 1–3 | |
| S5 | filter 7, pad 3 | 1, 2 | |
| S6 | dilation 2, pad 2 | 1–3 | dilated taps at stride 1 |
| S7 | dx 16×8 (2D), 16×8×4 (3D) | 2, 3 | rectangular dx |
| S8 | N 8, filter 1, pad 0 | 1–3 | batch |
| S9 | pad 4 | 1–3 | negative transposed padding: dy border rows reach no dx element |
| S10 | filter 2, stride 2, pre 0, post 1 | 1–3 | F11 asymmetric (probe) |
| S11 | filter 2, stride 2, pre 1, post 0 | 1–3 | F11 pre-only (probe) |
| S12 | filter 2, pad 1 | 1–3 | even filter: dy is larger than dx |
| S13 | filter equal to dx, pad 0 | 1–3 | dy spatial 1: one dy element spreads over all of dx |
| S14 | filter = dx + 2, pad 1 | 1–3 | filter larger than dx |
| S15 | stride (1, 2, 1), dilation (2, 1, 2), pad (1, 0, 1) | 2, 3 | per-dimension parameters |
| S16a, S16b, S16c | filter 1×3 / 1×3×3; 3×1 / 3×1×1; 3×5 | 2, 3; 2, 3; 2 | non-square filters |

**Stride phases.**

| Id | Change | Ranks | Why for dgrad |
|---|---|---|---|
| S17 | stride 2 | 1–3 | `f % s ≠ 0` (phases of 2 and 1 taps) and a floor remainder on an even dx |
| S18 | stride 2, dx 15 | 1–3 | the same phases without a remainder |
| S19 | filter 2, stride 2, pad 0 | 1–3 | `f = s`: every dx element meets exactly one tap |
| S20 | filter 4, stride 2, pad 1 | 1, 2 | `f = 2s`: equal phases of 2 taps |
| S21 | filter 4, stride 4, pad 0 | 2 | patchify 4 |
| S22 | C 3, filter 16, stride 16, pad 0, dx 32² | 2 | ViT patch embedding: dgrad scatters each dy element over a 16×16 block |
| S23 | filter 1, stride 2, pad 0 | 1–3 | `f < s`: 3 of 4 dx elements (2D) meet no tap |
| S24 | stride 4 | 1–3 | `f < s` with filter 3 |
| S25 | stride 3, pad 0 | 1–3 | `f = s` with a floor remainder: dx's last row meets no window |
| S26 | stride 2, dilation 2, pad 2 | 1–3 | gcd(s, d) = 2: odd dx positions meet no tap |
| S27 | stride 2, dilation 3, pad 3 | 1, 2 | coprime stride and dilation |
| S28 | C 3, filter 7, stride 2, pad 3 | 1–3 | strided stem, `f % s ≠ 0`, 3 output channels |
| S29 | stride 2, pad 0 | 1–3 | strided without padding |
| S30 | filter 7, stride 2, pad 3 | 2 | `f % s ≠ 0` with a large filter |
| S31 | filter 1×3×3, stride (1, 2, 2), pad (0, 1, 1) | 3 | (2+1)D video: strided in two of three dimensions |

**Channel classes** (C is dgrad's output, K its reduction). 2D unless noted.

| Id | C, K (small / medium) | Ranks | Why for dgrad |
|---|---|---|---|
| S32 | C 1 | 1–3 | one dx channel |
| S33 | C 3, K 64 | 2, 3 | input-image gradient |
| S34 | C 4, K 320 | 2 | small output, long reduction (latent `conv_in`; MFMA kernels with 4 channels per group) |
| S35 | K 1 | 1, 2 | the reduction is over taps only |
| S36 | C 128, K 3 | 2, 3 | wide output, 3-channel reduction (image `conv_out`) |
| S37 | C 64 / 256, K 2, filter 1 | 2 | binary head |
| S38 | C 13 / 61, K 17 / 67, dx 15 / 55 | 1–3 | odd channels and odd dx |
| S39 | C = K = 24 | 2 | a multiple of 8 that is not a power of two |
| S40 | C = K = 4 | 2 | both below the vector width |
| S41 | C 16 / 64, K 64 / 256, filter 1 | 2 | K = 4C: backward of a 1×1 expansion |
| S42 | C 64 / 256, K 16 / 64, filter 1 | 2 | C = 4K: backward of a 1×1 reduction |
| S77 | C 128 / 256, K 3, filter 1 | 2, 3 | RGB output head |
| S78 | C 128 / 256, K 21, filter 1 | 2 | class head: a reduction that is not a multiple of 8 |
| S79 | C 128, K 4, filter 3 | 2 | latent `conv_out` |
| S80 | C = K = 4, filter 1 | 2 | latent quant conv |

**Grouped convolution** (2D unless noted; per-group sizes are the class, the
medium form raises the group count).

| Id | G, C/G, K/G (small / medium) | Ranks | Why for dgrad |
|---|---|---|---|
| S43 | G 4 / 32, C/G = K/G = 4 | 2 | ResNeXt 32×4d |
| S44 | G 8 / 32, C/G = K/G = 8 | 2 | ResNeXt 32×8d |
| S45 | G 4 / 8, C/G = K/G = 16 | 2 | |
| S46 | G 2 / 4, C/G = K/G = 24 | 2 | |
| S47 | G 2 / 4, C/G = K/G = 32 | 2 | |
| S48 | G 2, C/G 4, K/G 320 | 2 | grouped small-output, long-reduction (MFMA kernels) |
| S49 | G 2 / 4, C/G 8 / 16, K/G 16 / 32 | 1–3 | K/G = 2·C/G |
| S50 | G 4 / 32, C/G = K/G = 8 / 4, stride 2 | 1–3 | grouped and strided |
| S51 | G 2 / 4, C/G = K/G = 8 / 16, filter 5, dilation 2, pad 4, dx 15 / 55 | 2 | grouped, dilated |
| S52 | G 4 / 16, C/G 8, K/G 1 | 2 | one dy channel per group |

**Depthwise** (G = C; 2D unless noted).

| Id | C (small / medium), change | Ranks | Why for dgrad |
|---|---|---|---|
| S53 | 16 / 64 | 1–3 | multiplier 1 |
| S54 | 16 / 64, K = 2C | 1–3 | multiplier 2 |
| S55 | 16 / 32, K = 4C | 1, 2 | multiplier 4 |
| S56 | 16 / 64, stride 2 | 1–3 | strided depthwise |
| S57 | 16 / 96, filter 5, stride 2, pad 2 | 2 | MobileNetV3, EfficientNet |
| S58 | 16 / 64, filter 7, pad 3 | 2 | ConvNeXt |
| S59 | 64, filter 7, pad 0, dx 16² / 28² | 2 | windowed depthwise: full transposed padding |
| S60 | 5, filter 7, pad 3, dx 9² / 56² | 2 | odd C below 16 |
| S61 | 64, dx 70×40 / 112×56 | 2 | tall dx |
| S62 | 32 / 64, dilation 2, pad 2, dx 15² / 55² | 2 | dilated depthwise |
| S63 | 32 / 64, stride (1, 2) | 2 | unequal strides |
| S64 | 64 / 256, filter 31, pad 15, dx 128 / 512 | 1 | Conformer |

**Image size and tiling** (2D).

| Id | Change (small / medium) | Why for dgrad |
|---|---|---|
| S65 | C = K = 32 / 128, dx 8² | dy width 8: kernels that keep the dy halo in registers |
| S66 | C = K = 32 / 256, dx 4² | dy width 4 |
| S67 | filter 5, pad 2, dx 16×48 / 48² | 5×5 with dy width 48 |
| S68 | N 1 / 2, C = K = 32 / 64, dx 32×109 / 64×109 | odd wide dx: row tiles with a remainder |
| S69 | dx 17×19 / 55×57 | odd non-square dx |
| S70 | N 8 / 32, filter 1, dx 1×1 | a pure GEMM (N × K → N × C) |
| S71 | filter 1×3, dx 1×256 / 1×4096 | extreme aspect (1D carried as 2D) |
| S72 | stride 2, dx 17×19 / 55×57 | strided on an odd non-square dx |

**Rank-specific.**

| Id | Change | Ranks | Why |
|---|---|---|---|
| S73 | filter 7, dilation 3, pad 9 | 1 | vocoder (HiFi-GAN) |
| S74 | filter 11, dilation 5, pad 25 | 1 | vocoder |
| S75 | C 1, K 32 / 512, filter 10, stride 5, dx 160 / 16000 | 1 | wav2vec feature extractor |
| S76 | 6-dim tensors, N 1, dx 6×6×4×4 | 4 | F1 4D (probe) |

### 3.2 Medium (L)

L*n* has the classes of S*n* on a medium base:

| Rank | N | C | K | Spatial |
|---|---|---|---|---|
| 1D | 4 | 64 | 128 | 1024 |
| 2D | 4 | 64 | 128 | 56² |
| 3D | 2 | 32 | 32 | 32³ |

A class that names its own medium values (the "/ medium" figures above) takes
them; otherwise it keeps its small channel, group and batch values and takes the
medium image. If a case exceeds the nightly ceiling, the generator halves N;
then, for a dense class whose channels are not its feature, C and K; for a
grouped class, the group count (per-group sizes kept); for a depthwise class, C
(multiplier kept). The 4D class has no L form.

### 3.3 Model (M) and corpus (R)

**M** is twelve dgrad layers of models trained today:

| Id | Source | N, C, spatial; K | Filter, stride, pad, groups |
|---|---|---|---|
| M1 | ResNet-50 3×3 | 8, 64, 56²; 64 | 3×3, p1 |
| M2 | ResNet-50 1×1 expansion | 8, 64, 56²; 256 | 1×1 |
| M3 | ResNet-50 downsample | 8, 512, 28²; 1024 | 1×1, s2 |
| M4 | ResNet-50 stem | 8, 3, 224²; 64 | 7×7, s2, p3 |
| M5 | ResNeXt grouped | 8, 128, 56²; 128 | 3×3, p1, G 32 |
| M6 | MobileNetV2 depthwise | 8, 96, 56²; 96 | 3×3, p1, G 96 |
| M7 | ConvNeXt depthwise | 8, 128, 56²; 128 | 7×7, p3, G 128 |
| M8 | Stable Diffusion UNet `conv_in` | 2, 4, 64²; 320 | 3×3, p1 |
| M9 | Stable Diffusion VAE `conv_out` | 1, 128, 256²; 3 | 3×3, p1 |
| M10 | UNet3D | 2, 32, 32³; 32 | 3×3×3, p1 |
| M11 | Conformer depthwise | 8, 256, 128; 256 | 31, p15, G 256 |
| M12 | ResNet-50 stage 3 | 8, 256, 14²; 256 | 3×3, p1 |

**R** comes from the backward-data graphs of the convolution corpus of
[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking)
(`Workloads/corpus/Convolution`, directory `conv/dgrad`): 2,481 unique dgrad
geometries recorded from live training runs of 31 models in 8 families
(classification 906, detection 481, diffusion 405, audio 188, video 177,
segmentation 175, 3-D medical 114, super-resolution 48). What they use:

| Property | Corpus share |
|---|---|
| Stride phases | s 1: 80%; `f % s ≠ 0`: 13%; `f % s = 0`: 4%; `f < s`: 2% |
| Transposed padding | positive 59%; full (pad 0) 41%; negative never |
| Floor remainder (dx rows no window reaches) | 15% |
| K/G ≤ 4 (tiny reduction) / C/G ≤ 4 (tiny output) | 9% / 9% |
| K/G > C/G / K/G < C/G | 29% / 33% |
| dy width ≤ 8 | 15% |
| Dense / depthwise / grouped | 90% / 6% / 5% |

- **Signature.** A dgrad behaviour signature is `conv_bwd_plan.signature()`:
  rank, group class, C/G class, K/G class, stride-phase class,
  transposed-padding class and floor remainder. The corpus has 111 signatures.
- **Selection.** Per signature R takes the median-FLOP graph inside each level
  ceiling (PR, nightly, and full up to 120 GFLOP and 1 GiB); every model gets at
  least one graph in PR or nightly; a graph picked for two levels sits in the
  lower.
- **Counts.** PR 45, nightly 86, full 76: 207 cases covering all 111 signatures
  and all 31 models.
- **Cases keep the model's data type and layout** (132 fp16, 42 fp32, 33 bf16;
  171 channels-first). Audio models keep the corpus form of a 1D conv: 2D with
  H = 1.
- **Purpose and provenance.** R cases are `claim` cases; `plan_items` holds the
  R id, `metadata.notes` the corpus shape id, models and family. The pick and
  its pin (repository commit and archive md5) are data in
  `conv_bwd_corpus.json`; a corpus update is a deliberate regeneration.
- **Synthetic cover.** Read with channel counts as "1, 2, 3, 4, multiple of 8,
  or other", the corpus has 47 behaviour classes; the S, L, M, H, A and X
  shapes reach 42 of them. The other five pair a 3-channel or
  non-multiple-of-8 channel count with a strided 3D or 7×7 stem; only R reaches
  them (18 corpus graphs). The exact signatures differ mostly by channel size
  (8–32 vs 40–256 vs > 256), which R covers at the sizes models use.
- **Limits of the corpus.** It was captured at small batch on one GPU and holds
  no fused graphs and no fp8 or int8. S, H and X keep the large batches and the
  awkward cases.

### 3.4 Hazard (H)

One change from the S defaults. C hazards hit dgrad's output vector width, K
hazards its reduction.

| Id | Change |
|---|---|
| H1–H6 | C / K: 15 / 32, 17 / 33, 31 / 31, 33 / 65, 63 / 64, 65 / 63 |
| H7 | C 7 (output below the vector width) |
| H8 | K 7 (reduction below the vector width) |
| H9–H12 | N 3, 1, 32, 64 |
| H13–H15 | spatial 31, 33, 17 |
| H16, H17 | G 3 (C 12, K 12); G 5 (C 15, K 20) |

### 3.5 Accumulation (A) and stress (X)

| Id | Shape | Feature |
|---|---|---|
| A1 | 2D, N 1, C 64, K 512, 14², 3×3 | A 4,608 (F23) |
| A2 | 3D, N 1, C 32, K 128, 8³, 3×3×3 | A 3,456 (F23) |
| X1 | 2D, N 256, C = K 64, 56², 3×3 | F2 |
| X2 | 3D, N 2, C = K 64, 64³, 3×3×3 | size |
| X3 | 2D, N 1, C = K 1, 46341², 1×1 (more than 2³¹ elements); fp16 and bf16 | F18 |
| X4 | 2D, N 1, C 64, K 2048, 14², 3×3 | A 18,432 (F23) |

---

## 4. Building the sweep

| Rule | Meaning |
|---|---|
| Core cover | Every level holds every rank × layout × data type cell (18) for plain dgrad and for each fused topology. In quick, at least one engine executes each cell of plain dgrad. |
| One feature at a time | Each S and L class changes one dgrad property from the defaults, so a failure points at it. |
| Size grows with level | PR proves a class works small; nightly at medium size; full adds production sizes and stress. |
| All-pairs | By the end of nightly every S class meets every layout × data type pair of its rank, and every feature probe meets every core value. |
| Probes | Every unsupported feature, awkward combination and hazard has a probe in every level (directly or through the cascade). |
| Concrete graphs | Each level runs checked-in graphs only; the generator is deterministic. |
| Unsupported stays in | A feature no engine supports stays in the sweep, unclaimed; it reports `unclaimed_support` once an engine adds it. |

### 4.1 Level contents

| Feature group | quick | PR | nightly | full |
|---|---|---|---|---|
| Shape size | ≤ 20 MFLOP | ≤ 200 MFLOP (+3 M) | ≤ 8 GFLOP | unbounded |
| Plain core | S1, S3, S6, S8, S17, S19, S23, S44, S53, pair rotated per rank | every S class × rank once, pair rotated | every S class × every pair of its rank; L × 2 pairs | L × every pair |
| Rejection probes | `CONVOLUTION`, asymmetric pad (S10), fp8, double, sigmoid fusion | `CONVOLUTION` per rank; 6 extra types; 5 mixed pairs | × core | asymmetric padding, mixed layouts, strides, mixed types and values × every S class of S1–S31; `CONVOLUTION`, extra types and override shape stay at nightly's × core (T11) |
| F15, F16 | — | 1 case each | rank × dtype, 2 mixes and 3 stride kinds | paired with every S class of S1–S31 |
| F19, F20 | fp8 E4M3, double | each extra type and mixed pair once | × rank × layout | F20 paired with every S class of S1–S31; F19 — (T11) |
| F21 values | uniform | + zero, const, large | + tiny, × rank × dtype; A1, A2 | × core |
| F35 | — | 1 case | × rank × dtype | — (T11) |
| H hazards | — | each H once (2D) | each H × every pair (2D); 1D, 3D × 2 pairs | — |
| Model (M) | — | M1, M8, M11 | M1–M12 × every pair of its rank | — |
| Corpus (R) | — | 45 graphs | 86 graphs | 76 graphs |
| Stress | — | — | — | X1, X2, X4 × every pair; X3 fp16 and bf16 |
| Fusion core (4 topologies × 18) | 1 per cell; ReLU family rotated on `dact`, `pre_dact` | `dact`, `pre_dact`: 2× core, all 10 activations over the cells; `add`, `add_drelu`: 1× core | from quick and PR (T9) | from quick and PR (T9) |
| Fusion features F30, F31 | — | each once per prologue / epilogue | × rank | — |

The level rules generate more fused cases than the table shows (nightly: `dact`
and `pre_dact` over every S class of S1–S31 × every pair, the L classes and
M1, M2, M6, M8; full: every activation × core and × every S and L class of
S1–S31, and `add`, `add_drelu` over every S class); T9 (§4.3) removes them
until an engine executes a dgrad fusion.

Cases generated (own level / cumulative):

| Level | Plain dgrad | Fused | Own | Cumulative | Claim | Probe |
|---|---|---|---|---|---|---|
| quick | 29 | 73 | **102** | 102 | 43 | 59 |
| PR | 231 | 131 | **362** | 464 | 206 | 156 |
| nightly | 1,506 | 12 | **1,518** | 1,982 | 1,151 | 367 |
| full | 1,150 | 0 | **1,150** | 3,132 | 662 | 488 |

By size class: quick S 102. PR S 297, H 17, M 3, R 45. Nightly S 903, L 298,
M 66, H 153, A 12, R 86. Full S 462, L 592, R 76, X 20. 656 distinct dgrad
geometries. The trims of §4.3 remove 3,411 of the 6,543 generated cases.
`generate_conv_bwd_bundles.py --report` prints the counts again.

### 4.2 Probes and regressions

| Purpose | Expected outcome | Claims |
|---|---|---|
| `claim` | A claiming engine accepts and is correct. | Recorded. |
| `probe` | A typed rejection, or a correct accept. | Not recorded; support matrices exclude probes. |
| `regression` | Same as the original bug case. | Same as the original case. |

A case is a **probe** when it has at least one of:

| Probe class | Items |
|---|---|
| Unsupported feature | F13 `CONVOLUTION`; F11 asymmetric padding; F19 fp8, int8, double; F1 4D spatial; F26 non-ReLU backward activation; F28, F29, F36: the `pre_dact`, `add` and `add_drelu` topologies |
| Awkward combination | F15 mixed layouts; F16 non-packed strides; F20 mixed precision and compute type; F30 materialised intermediate; F31 io-type intermediate; F35 override shape |
| Value domain | F21 zero, const, large, tiny |
| Hazard | every H shape; X3 (F18) |

Everything else is a `claim`: ranks 1–3, both layouts, fp32, fp16 and bf16, the
S, L, M, R and A shapes, `dact` with the ReLU family, and X1, X2, X4. There are
no `regression` cases yet; each escaped dgrad bug's reproducer will land in the
lowest level that can afford it.

### 4.3 Products deliberately not run

The complete product of every feature with every other is far larger than any
level can run. These are the products this plan leaves out on purpose, what
covers the same risk instead, and what brings them back.

**Design decisions (built into the level rules).**

| Id | Product not run | Why it adds little | Covered instead by | Add it back when |
|---|---|---|---|---|
| D1 | Every backward activation × every S and L class × every layout and data type | An activation's backward is element-wise on dx (or dy); it depends on the data type, layout and fusion path, not on stride phases or channel classes. | Every activation on every core cell (PR) and over every S class of S1–S31 once (full, before T9). | A defect needs an activation and a geometry together, or an engine picks fused kernels by geometry class. |
| D2 | Every extra data type, mixed pair, `CONVOLUTION` mode, mixed layout and non-packed stride × every S class × every layout and dtype | A data type, mode or layout changes kernel selection per tensor, visible on any geometry. | Each × rank × layout or dtype in nightly; each mixed pair, layout mix and stride kind paired with every S class of S1–S31 once in full. | An engine claims the type or mode, or a layout bug needs a geometry feature. |
| D3 | The kernel-selecting classes S32–S80 (channels, groups, depthwise, tiling, rank-specific, heads) crossed with the fused topologies or the feature probes | These classes select the dgrad kernel itself; an epilogue or a per-tensor feature does not depend on which one runs. | Each class in plain dgrad from PR up, over every pair in nightly, and at L size in nightly and full. | A defect needs a fusion or a feature together with one of these classes. |
| D4 | Hazard shapes × every layout and dtype in 1D and 3D | Hazards act on C, K, N, G and extent; layout and dtype multiply compile work, not hazard cover. | 2D × every pair; 1D and 3D × two pairs (nightly). | A hazard depends on rank. |
| D5 | Three-way products | Pairwise cover catches most interaction faults; the corpus R set adds the real three-way combinations. | All-pairs (§4); R. | A three-way fault is found: add that triple. |

**Measured trims (applied after case ids are assigned, `conv_bwd_plan.TRIMS`).**
The cases that remain keep their ids, seeds and claims, and undoing a trim
restores exactly what it removed. No trim removes the last probe of a feature
from any level, and every core cell of every topology stays in quick and PR.
F11, F15, F16, F20 and the F30, F31 fused variants are never trimmed: F11 exists
only in full, and the others hold the evidence for the defects of §4.5.

| Id | Product not run | Cases | Why it adds little | Covered instead by | Add it back when |
|---|---|---|---|---|---|
| T9 | Plain repeats of the four fused topologies in nightly and full | 3,171 (nightly 1,218, full 1,953) | No engine executed a fused dgrad graph on gfx90a or gfx942 (§4.5); each case repeats the same decline. | Each topology's 18 core cells in quick and PR, all 10 activations over the PR cells, and the F30, F31 variants in nightly. | An engine executes a dgrad fusion: remove its topology from `UNEXECUTED_TOPOLOGIES`. |
| T11 | F13, F19 and F35 paired with every S class in full | 240 | No engine executed them, and no reference can judge F13 or F19 (§6). | Each × core in nightly (F13 18, F19 108, F35 9). | An engine executes the feature: remove it from `UNEXECUTED_FEATURES`; for F13 and F19 the reference fixes of §6 must land too. |

### 4.4 Cost

| Estimate | Value |
|---|---|
| FLOP per case | quick ≤ 14 M, PR ≤ 1.9 G (the three M cases), nightly ≤ 7.3 G, full 116 G (X2) |
| Oracle | CPU reference only, multi-threaded (`hardware_concurrency`). It sets the cost of nightly and full: X and the largest R cases take seconds each. |
| Compile and solver search | paid once per unique shape on a cold cache |
| Device memory | X1 ≈ 0.4 GB, X3 ≈ 13 GB (fp16), so `minimum_vram_mb` is set on those cases |

### 4.5 Measured run

**Which cases execute.** T9 and T11 come from serial runs (one test at a time,
chunks of 100 tests per process) of all 6,543 generated cases with the MIOpen
plugin, the only engine that runs dgrad today, at the default tolerances:

| Architecture | OK | Skipped | Failed | Crashed | Wall time |
|---|---|---|---|---|---|
| gfx90a (MI250) | 1,872 | 3,912 | 745 | 14 | 26 min |
| gfx942 (MI325X) | 2,160 | 3,914 | 457 | 12 | 25 min |

- No fused graph executed on either architecture (3,387 cases, all four
  topologies), and F11, F13, F19, F30, F31 and F35 never executed.
- Every plain-dgrad class of §3 executed on both, except the asymmetric-padding
  classes S10 and S11 (and their L forms) and the 4D class S76, which the plugin
  declines.
- No plain-dgrad case failed beyond the numerical floor: the 538 (gfx90a) and 272
  (gfx942) claim cases that failed did so only because the default tolerance is
  below the floor of their reduction length (§6). The engine TOML's classes by
  accumulation length cover them.

**The current set.** All 3,132 cases, serial, with the MIOpen plugin and the
engine TOML:

| Architecture | OK | Skipped | Failed | Crashed | Wall time |
|---|---|---|---|---|---|
| gfx90a (MI250) | 2,474 | 501 | 142 | 15 | 31 min |
| gfx942 (MI325X) | 2,460 | 503 | 156 | 13 | 26 min |

No claim case fails on either architecture; every failure is a probe, and each
is a graph the MIOpen plugin accepts and then fails on:

| Graph | gfx90a | gfx942 | What happens |
|---|---|---|---|
| F20 mixed dy, w, dx types | 85 of 109 accepted: 13 crash, 53 fail at execute, 19 run with no oracle | 83 of 109: 13, 51, 19 | The plugin accepts whenever MIOpen offers a solution. Execute returns `miopenStatusBadParm` or `miopenStatusUnknownError`; the 3D cases then fault the device (error 700). dy fp32 with w fp16 runs, and no reference registers that type triple. Only fp16 with an fp16 compute type is declined. |
| F15 mixed layouts | 61 of 65 executed are wrong | 62 of 65 | Output disagrees with the CPU reference by the full output magnitude; the same graphs in one layout pass. |
| F16 non-packed strides | 9 of 103 executed are wrong | 22 of 103 | Wrong output, up to 1e35 in some channel-stride cases. |
| X3, 64-bit indexing | 2 of 2 fail | 2 of 2 | `miopenStatusUnknownError`; on gfx90a then an illegal memory access. |

They are left failing, not skipped, so each shows in every run until the plugin
declines the graph or MIOpen computes it correctly. Claims are harvested from
these runs: 2,028 claim cases on each of gfx90a and gfx942 (linux), every claim
case that executed; the other 34 claim cases are `dact` with the ReLU family,
which no engine executes.

---

## 5. Golden reference

The selection follows [methodology §4.2](methodology.md#42-which-cases-should-be-golden).
Dgrad needs golden data more than most ops: with no GPU reference, every case has
one live oracle. Cases are generated by PyTorch on CPU
(`torch.nn.grad.conv{1,2,3}d_input`), with the backward pointwise ops applied in
PyTorch for the fused cases, and record `forward_source`.

| Id | Graph | Case | Criterion | Level |
|---|---|---|---|---|
| G1 | dgrad | 2D NCHW fp32, S3 | G-A baseline | quick |
| G2 | dgrad | 1D NLC fp16, S17 (stride 2) | G-A baseline | PR |
| G3 | dgrad | 3D NCDHW bf16, S3 | G-A baseline | PR |
| G4 | dgrad | 2D NHWC fp16, S1 (1×1) | G-A baseline | PR |
| G5 | dgrad | 3D NDHWC fp32, S17 | G-A baseline | PR |
| G6 | dgrad | 1D NCL bf16, S49 (grouped) | G-A baseline | PR |
| G7 | dgrad → relu_bwd(·, x) | 2D NHWC fp16, S3 | G-A baseline; G-C clip semantics of `relu_bwd` | quick |
| G8 | relu_bwd(dy, y) → dgrad | 2D NCHW bf16, S3 | G-A baseline (prologue) | PR |
| G9 | dgrad → add(dx_residual) | 2D NCHW fp32, S3 | G-A baseline | PR |
| G10 | dgrad → add → relu_bwd | 2D NHWC bf16, S3 | G-A baseline | PR |
| G11 | dgrad, `CONVOLUTION` mode | 2D NCHW fp32, S3 | G-B oracle gap (F13) | PR |
| G12 | dgrad | 2D NCHW fp32, S10 (pre 0, post 1) | G-C asymmetric padding convention | PR |
| G13 | dgrad | 2D NCHW fp32, S26 (stride 2, dilation 2) | G-C: dx elements no tap reaches are 0 (F37) | PR |
| G14 | dgrad → gelu_bwd(·, x) | 2D NCHW fp32, S3 | G-B oracle gap (F26) | PR |
| G15 | dgrad | 2D NCHW fp32, S13 (filter equal to dx, A 8,192) | G-D: long accumulation | PR |

Fifteen cases, each also a live sweep case. **The generator
(`generate_conv_bwd_golden.py`) does not exist yet (I7); no dgrad case has
golden data.** Precision: uplift (fp64, cast) for fp32; matched precision (fp32
accumulation, rounded to the io type) for fp16 and bf16; recorded per case in
`meta.json`. Confirmation is `hipdnn_golden_data_tests --reference both`, except
G11 and G14, which are confirmed by a second independent computation (NumPy)
until their reference fixes land. Storage is `golden/<case-id>/tensors.dvc`
beside the sweep.

---

## 6. Pass criteria and reference prerequisites

| Item | Rule |
|---|---|
| Verification | `auto`: golden → GPU reference → CPU reference. For dgrad the GPU step always declines, so `auto` is the CPU reference until golden data exists. |
| Tolerance | `TestTolerances.hpp` `conv::getToleranceBwd`: fp32 8.5e-6, fp16 and bf16 2e-2, used as atol and rtol; no `double` value. The absolute error between two correct implementations grows with the accumulation length A (§1), so an engine that runs long reductions needs tolerance classes by A in its TOML, calibrated from observed error and never loosened to make a case pass. |
| Support claims | Enforced. Only `claim` cases are recorded (§4.2). |

These features need reference or harness work before their cases can be judged
or run. The generator emits their cases now; they are unclaimed until the work
lands.

| Feature | Gap | Needed | RFC 0021 |
|---|---|---|---|
| All dgrad cases | No GPU reference for `ConvolutionBwdAttributes`: every case has a single live oracle, and large cases cost seconds on the CPU. | A GPU dgrad reference. | I6 |
| F13 `CONVOLUTION` | The CPU reference rejects it; no GPU reference. An accepting engine fails "no oracle". | Flip the filter in the reference; until then golden G11. | I6 |
| F19, F20 | The CPU reference registers only the three uniform core triples. | Register the types, or golden data. | I6, I7 |
| F26 elu, gelu, gelu-tanh, swish, softplus backward | The CPU pointwise reference implements only relu, sigmoid and tanh backward. | Implement them; until then golden G14. | I6 |
| F11 | No CPU test covers asymmetric dgrad. | Add one; golden G12. | I6 |
| F21 subnormal, zero-excluded, sparse; F22 | Fill recipes are `free` and `fixed` only. | Value-domain control; a matching-infinities validator. | I4 |
| F17; F24, F32–F34 | Allocator pointers are 256-byte aligned; bundles run one execution. | Exact placement; repeat-and-compare. | I3, I5 |
| `purpose` | The harness ignores the tag. | The harvester skips probes. | I2 |

Engine-specific tolerances and skips go in the engine TOML, keyed to the case
ids: `<plan items>_<rank>d_i<dx spatial>_n<N>c<C>k<K>[g<G>]_f<filter>_<dtype>[_w<dtype>_dx<dtype>]_<layout | dy..w..dx..>[_<activation>]`.
A change that renames ids updates the TOML globs in the same pull request.

---

## 7. Exhaustive mode

Dgrad has an exhaustive mode: a separate tool (RFC 0021 §14) that reads the
declarations in `conv_bwd_plan.py`.

| Declaration | Where | Content |
|---|---|---|
| Axes | `RANKS`, `LAYOUTS`, `CORE_DTYPES`, `EXTRA_TYPES`, `MIXED_TYPES`, `ACTS`, `TOPOLOGIES`, the case fields of `DEFAULTS` | every feature of §2 |
| Legality | `legal()` | §1 and §2.7 |
| Behaviour classes | `signature()`, `stride_class()`, `tpad_class()`, `remainder()`, `channel_class()`, `group_class()` | the properties of §2.1 that select kernels |
| Shape ranges | rank 1–4; N 1–256; C, K 1–2048; G dividing C and K; spatial 1–16000 (1D), 1–224 (2D), 1–64 (3D); filter 1–31; stride 1–16; dilation 1–12; padding 0 – dilated filter; x + w + y ≤ 1 GiB |
| Value domains | `value_recipes()`, `VALUE_CLASSES` | §2.4 |
| Return path | minimised reproducers become `regression` cases; oracle bugs become G-B golden candidates |
