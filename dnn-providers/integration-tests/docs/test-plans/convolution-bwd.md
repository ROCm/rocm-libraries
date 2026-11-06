# Convolution Backward-Data Test Plan

This plan lists every feature of convolution backward data (dgrad,
`ConvolutionBwdAttributes`) and its pointwise fusions that a complete test sweep
must exercise, and how the sweep is spread across levels. It does not depend on
any engine: the sweep describes the problem space, and each engine's support
claims decide which cases it runs. Convolution forward has its own plan
([convolution-fwd.md](convolution-fwd.md)); weight gradient (`ConvolutionWrw`)
is out of scope and gets `convolution-wrw.md`.

The plan follows RFC 0021 (Per-Op Test Plans) and the [methodology](methodology.md).
See [the test-plans index](README.md) for the level definitions. It is built the
same way as the forward plan and shares its geometry: where a section is the
same, this plan says so and points there instead of repeating it.

The bundles are **generated** from this plan. Nobody edits them by hand:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_conv_bwd_bundles.py \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles --report
```

| Plan item | Where it lives |
|---|---|
| Data types, activations, topologies, value domains, purpose rule, trimmed products (`TRIMS`) | `reference-data-scripts/conv_bwd_plan.py` |
| Geometry: legality, shape catalog S, L, M, H, A, X, ceilings | `reference-data-scripts/conv_fwd_plan.py` (imported; dgrad has the forward geometry) |
| R set: corpus pick, selection rule, provenance | `reference-data-scripts/conv_bwd_corpus.json` |
| Level rules, graph templates, sweep writer, purpose tags, trims | `reference-data-scripts/generate_conv_bwd_bundles.py` |
| Bundles | `integration-test-bundles/{quick,standard,comprehensive,full}/ConvolutionBwd/Default` and `ConvolutionBwdPointwise/{Default,Variant2..Variant4}` |

Every sweep case records the plan items it implements (`metadata.plan_items`),
its `purpose` (§4.2) and its size class, as in the forward plan.

The features come from the graph API:

- `flatbuffers_sdk/schemas/convolution_bwd_attributes.fbs` and `convolution_common.fbs`
- `tensor_attributes.fbs`, `pointwise_attributes.fbs`, `data_types.fbs`, `graph.fbs`
- the frontend rules in `hipdnn_frontend/node/ConvolutionDgradNode.hpp`

---

## 1. The operation

Dgrad is the gradient of `y = conv(x, w)` with respect to x:

```
dx[n, c, i...] = Σ_{k in group(c)} Σ_{r...} dy[n, k, (i + pre_pad − r·dilation) / stride] · w[k, c', r...]
                 (only terms where the division is exact and the index is inside dy)
```

| Tensor | Dims | Rule |
|---|---|---|
| dy | N, K, out… | input; at least 3 dims; `dy[1] == w[0]` |
| w | K, C/G, filter… | input; same rank as dy; `dx[1] % w[1] == 0`, so G = C / w[1]; `K % G == 0` |
| dx | N, C, in… | output; `dx[0] == dy[0]`; `out = floor((in + pre + post − (dil·(f−1)+1)) / stride) + 1` must hold for every spatial dim. **dx dims are never inferred**: the frontend rejects a graph without them (`ATTRIBUTE_NOT_SET`). Only dx strides are inferred, from dy's stride order. |

Parameters are those of the forward convolution: `pre_padding`,
`post_padding`, `stride`, `dilation` (one value per spatial dimension) and
`conv_mode`. The node `compute_data_type` and the graph's io, intermediate and
compute data types are set separately.

The geometry (N, C, K, G, spatial, filter, stride, dilation, padding) is the
forward convolution's, with x read as dx and y as dy. The FLOP count is the
same. What changes is the data flow: a dx element sums over the K/G output
channels of its group and over the filter taps that land on it. With stride s
and dilation d a dx position meets one tap in every s / gcd(s, d), so the
accumulation length is `A = K/G × Π ceil(f / (s / gcd(s, d)))`, and some dx
elements may meet no tap at all (F37).

---

## 2. Feature inventory

Ids follow the forward plan, so one id means one feature in both plans. The
forward features that have no dgrad form are F25 (bias add) and F27 (scale):
the bias gradient is a reduction (dbias), not a pointwise epilogue of dgrad, and
a scaled gradient has no fused form in the API's backward fusion patterns. Two
features are dgrad's own: F36 (prologue fusion) and F37 (dx elements that no tap
reaches).

**Oracle.** There is **no GPU reference for dgrad**
(`GpuReferenceGraphExecutor.cpp` declines `ConvolutionBwdAttributes`). Every
dgrad case is judged by the CPU reference (`CpuFpReferenceConvolution::dgrad`),
which registers dy, w, dx = fp32/fp32/fp32, fp16/fp16/fp16 and
bf16/bf16/bf16 with fp32 compute, and accepts only `CROSS_CORRELATION`. A case
the engine accepts and no oracle can judge **fails** ("no oracle can verify
this bundle"), so the oracle gaps in §6 show up as failures, not skips.

**Legality** is declared once, in `conv_fwd_plan.legal()` (the forward rules
are the same rules: `C % G == 0`, `K % G == 0`, stride and dilation ≥ 1,
padding ≥ 0, one parameter per spatial dimension, a filter no larger than the
padded input). The dgrad-only rule, that dy's spatial dims equal the output
formula, holds by construction: the generator computes dy from dx.

### 2.1 Geometry

F1–F11 are the forward plan's (§2.1 there) with the same values to cover and the
same earliest levels. Differences:

| Id | Feature | Dgrad difference | Oracle | Earliest level |
|---|---|---|---|---|
| F1–F10 | rank, N, C, K, G, spatial, filter, stride, dilation, symmetric padding | none | CPU ref | Q |
| F11 | Asymmetric padding | none | CPU ref takes pre and post separately; no CPU test covers asymmetric dgrad (I6) | Q (probe) |
| F12 | Output edge cases | dy spatial size 1 (S19: one dy element per channel spreads over the whole dx), dx larger than dy (stride, dilation). **Inferred dx dims are illegal** (the frontend rejects them), so the forward plan's "inferred output" case is a frontend unit test here. | CPU ref | Q |
| F13 | `conv_mode` | `CONVOLUTION` | **none**: the CPU ref rejects it for dgrad, there is no GPU ref (I6) | Q (probe) |
| F37 | dx elements no tap reaches | stride larger than the dilated filter (S46: stride 4, filter 3; S34: 1×1, stride 2); stride and dilation sharing a factor (S8: stride 2, dilation 2, so every other dx element meets no tap); a floor remainder (S20: the last dx rows lie past the last window). The engine must write 0 there; the harness prefills outputs with a NaN sentinel, so an unwritten element fails. | CPU ref | Q (S8); S20, S34, S46 from PR |

### 2.2 Memory layout and tensor kind

As the forward plan's §2.2, applied to dy, w and dx. F15 mixes the three in all
six non-uniform combinations; F16 pads dy and dx; F17 (alignment) is not run
(I3); F18 is X3; F35 is the override-shape flag. Every tensor is a device
tensor: a dgrad input that is a constant or a runtime scalar is not a valid
graph.

### 2.3 Data types

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F19 | IO data type | fp32, fp16, bf16; fp8 (E4M3, E5M2 and their FNUZ forms) with fp32 dx; int8 with int32 dx; double | **none** beyond the core three (I6) | Q (core; fp8 e4m3 and double as probes) |
| F20 | Mixed precision | dx type ≠ dy type (fp16 / bf16 in, fp32 out); dy type ≠ w type; node compute type fp16 | **none** (only uniform core triples are registered) | PR |

### 2.4 Values and numerics

As the forward plan's §2.4 with dy in place of x: F21 (uniform, zero dy,
constant dy = w = 1, large, tiny), F22 (special values, not run), F23
(accumulation length, now `A` of §1), F24 (determinism: a C++ test). The large
class uses `a = √(½ · max / A)` with dgrad's A. The forward plan's long
accumulations come from a large C, which is short for dgrad (A1 and X4 have
K 64, so A = 576). Dgrad's long accumulations come from a large K or a large
filter: S19 and L19 (filter equal to the input, A up to 131,072), S45, M10, and R
cases up to A = 10,368.

### 2.5 Fusions

The API lets pointwise nodes sit after dgrad (an epilogue on dx) or before it
(a prologue on dy). The backward pointwise modes take the incoming gradient as
`in_0` and the forward activation's input as `in_1`. The topologies are the
bundle directories:

| Topology | Bundle | Nodes | Where it comes from |
|---|---|---|---|
| `dact` | `ConvolutionBwdPointwise/Default` | dgrad → act_bwd(·, x) | backward of `act → conv`: the previous layer's activation |
| `pre_dact` | `Variant2` | act_bwd(dy, y) → dgrad | backward of `conv → act`: this layer's activation, applied to dy before the dgrad |
| `add` | `Variant3` | dgrad → add(dx_residual) | gradient of a skip connection summed into dx |
| `add_drelu` | `Variant4` | dgrad → add(dx_residual) → relu_bwd(·, x) | ResNet block backward |

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F26 | Backward activation | `relu_bwd`: plain, upper clip (ReLU6), leaky (`relu_lower_clip_slope`). Also `sigmoid_bwd`, `tanh_bwd`, `elu_bwd` (α = 1), `gelu_bwd`, `gelu_approx_tanh_bwd`, `swish_bwd` (β = 1), `softplus_bwd` (β = 1) | CPU ref: relu, sigmoid, tanh only; **none** for elu, gelu, gelu-tanh, swish, softplus (I6) | Q |
| F28 | Residual-gradient add | add a second full-size tensor to dx | CPU ref | Q |
| F29 | Chains | the four topologies above | CPU ref (for the ReLU family) | Q |
| F30 | Materialised intermediate | the fused intermediate (dgrad's dx, or the prologue's dy) as a second output instead of virtual | CPU ref | PR |
| F31 | Fused intermediate type | virtual intermediate in fp32 vs in the io type | CPU ref | PR |
| F36 | Prologue fusion | a pointwise node feeding dgrad's dy (`pre_dact`) | CPU ref | Q |

### 2.6 Execution and 2.7 invalid problems

As the forward plan's §2.6 (workspace, reuse, serialisation: C++ tests) and
§2.7, plus the dgrad-only frontend rejections:

- dx dims not set (dgrad never infers them);
- `dy[1] ≠ w[0]`;
- `dx[0] ≠ dy[0]`;
- dy spatial dims inconsistent with the output formula.

---

## 3. Shape catalog

The S, L, M, H, A and X sets, the level ceilings, the S ladder and the L
scaling rule are the forward plan's (§3, §3.1, §3.2, §3.4, §3.5 there),
declared once in `conv_fwd_plan.py` and read by both generators. A shape is a
conv geometry, and dgrad runs the same geometry backwards: S4 (stride 2) here is
the backward of the forward S4. Sharing the catalog keeps the two plans
comparable case for case. A change to the catalog regenerates both plans'
bundles.

### 3.3 Corpus (R)

**R** is drawn from the backward-data graphs of the same
[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking) corpus the forward
plan uses (`Workloads/corpus/Convolution`, directory `conv/dgrad`): 2,481 unique
dgrad geometries from 31 models in 8 families. The selection rule is the forward
plan's: a behaviour signature is (rank, group class, filter, stride, dilation,
padding class); for each signature R takes the median-FLOP graph inside each
level ceiling (PR, nightly, and full up to 120 GFLOP and 1 GiB), every model gets
at least one graph in PR or nightly, and a graph picked for two levels sits in
the lower.

- **Counts.** PR 19, nightly 51, full 43: 113 cases covering 56 of the 60
  signatures and all 31 models. The other four signatures have no graph under
  the full ceiling.
- **Cases keep the model's data type and layout** (80 fp16, 19 bf16, 14 fp32;
  93 channels-first). Audio models keep the corpus form of a 1D conv: 2D with
  H = 1.
- **Purpose and provenance.** R cases are `claim` cases; `plan_items` holds the
  R id, `metadata.notes` the corpus shape id, models and family. The pick and the
  pin are data in `conv_bwd_corpus.json`.

---

## 4. Building the sweep

The rules are the forward plan's (§4 there): core axes rank × layout × data type
(18 cells), core cover in every level for plain dgrad and for each fused
topology, one feature at a time, size grows with level, all-pairs by the end of
nightly, complete products only where two features plausibly interact, probes in
every level, concrete checked-in graphs, and unsupported features stay in.

### 4.1 Level contents

| Feature group | quick | PR | nightly | full |
|---|---|---|---|---|
| Shape size | ≤ 20 MFLOP | ≤ 200 MFLOP (+3 M) | ≤ 8 GFLOP | unbounded |
| Plain core (18) | S1–S9, layout × dtype pair rotated per rank | every S shape × rank once, pair rotated | every S shape × every pair of its rank | L × every pair |
| Geometry classes S24–S62 | — | each class × rank once | × every pair; L24–L62 × 2 pairs | L24–L62 × every pair; no fused or feature crossing (T12) |
| Rejection probes | `CONVOLUTION`, asymmetric pad (S18), fp8, double, sigmoid fusion | `CONVOLUTION` per rank; 6 extra types; 5 mixed pairs | × core | asymmetric padding, mixed layouts, strides and mixed types × every S shape; `CONVOLUTION`, extra types and override shape stay at nightly's × core (T11) |
| F15–F16 layout, strides | — | 1 case each | rank × dtype, 2 mixes and 3 stride kinds | paired with every S shape |
| F19–F20 types | fp8 e4m3, double | each extra type and mixed pair once | × rank × layout | F20 paired with every S shape; F19 — (T11) |
| F21 values | uniform | + zero, const, large | + tiny, × rank × dtype; A1, A2 | × core; paired with every S shape |
| F35 | — | 1 case | × rank × dtype | — (T11) |
| H hazards | — | each H once (2D) | each H × every pair (2D); 1D, 3D × 2 pairs | — |
| Model (M) | — | M2, M9, M11 | M1–M12 × every pair of its rank | — |
| Corpus (R) | — | 19 graphs | 51 graphs | 43 graphs |
| Stress | — | — | — | X1, X2, X4 × every pair; X3 fp16 and bf16 |
| Fusion core (4 topologies × 18) | 1 per cell; ReLU family rotated on `dact`, `pre_dact` | `dact`, `pre_dact`: 2× core, all 10 activations over the cells; `add`, `add_drelu`: 1× core | from quick and PR (T9) | from quick and PR (T9) |
| Fusion features F30, F31 | — | each once per prologue / epilogue | × rank | — |

The level rules generate more fused cases than the table shows: in nightly
`dact` and `pre_dact` over every S shape × every pair, every L shape once and
M2, M3, M6, M9 × every pair; in full `dact` with every activation × core and ×
every S and L shape, `pre_dact` with every activation × core, and `add`,
`add_drelu` over every S shape. T9 (§4.3) removes all of them until an engine
executes a dgrad fusion.

Cases generated (own level / cumulative):

| Level | Plain dgrad | Fused | Own | Cumulative | Claim | Probe |
|---|---|---|---|---|---|---|
| quick | 30 | 73 | **103** | 103 | 44 | 59 |
| PR | 183 | 130 | **313** | 416 | 159 | 154 |
| nightly | 1,329 | 12 | **1,341** | 1,757 | 964 | 377 |
| full | 961 | 0 | **961** | 2,718 | 537 | 424 |

By size class: quick S 103. PR S 273, H 18, M 3, R 19. Nightly S 791, L 251,
M 74, H 162, A 12, R 51. Full S 398, L 500, R 43, X 20. 517 distinct conv
geometries in all. The trims of §4.3 remove 4,687 of the 7,405 generated cases.
`generate_conv_bwd_bundles.py --report` prints the counts again, with the
trimmed count per level and trim.

### 4.2 Probes and regressions

The purpose rule is the forward plan's (§4.2 there). A dgrad case is a
**probe** when it has at least one of:

| Probe class | Items |
|---|---|
| Unsupported feature | F13 `CONVOLUTION`; F11 asymmetric padding; F19 fp8, int8, double; F1 4D spatial; F26 non-ReLU backward activation; F28, F29, F36: the `pre_dact`, `add` and `add_drelu` topologies |
| Awkward combination | F15 mixed layouts; F16 non-packed strides; F20 mixed precision and compute type; F30 materialised intermediate; F31 io-type intermediate; F35 override shape |
| Value domain | F21 zero, const, large, tiny |
| Hazard | every H shape; X3 (F18) |

Everything else is a `claim`: ranks 1–3, both layouts, fp32, fp16 and bf16, the
S, L, M, R and A shapes, `dact` with the ReLU family, and X1, X2, X4. There are no
`regression` cases yet.

### 4.3 Products deliberately not run

T1–T7 of the forward plan apply unchanged (activation × geometry, extra types ×
geometry, `CONVOLUTION` × geometry, layout features × geometry, non-activation
topologies × L, hazards × layout in 1D and 3D, 3-way products); the level rules
above already leave them out. The trims below are applied after case ids are
assigned (`conv_bwd_plan.TRIMS`), so the cases that remain keep their ids, seeds
and claims, and undoing a trim restores exactly the cases it removed. No trim
removes the last probe of a feature from any level, and every core cell of every
topology stays in quick and PR.

| Id | Product not run | Cases it would add | Why it adds little | Covered instead by | Add it back when |
|---|---|---|---|---|---|
| T9 | Plain repeats of the four fused topologies (`dact`, `pre_dact`, `add`, `add_drelu`) in nightly and full | 4,151 (nightly 1,617, full 2,534) | No engine executed a fused dgrad graph on gfx90a or gfx942 (§4.5): the MIOpen plugin has no dgrad fusion and declined all 2,855 of them on both. Each case repeats the same decline. | Each topology's 18 core cells in quick and PR (run by every level through the cascade), all 10 activations over the PR cells, and the F30, F31 variants in nightly. | An engine executes a dgrad fusion: remove its topology from `UNEXECUTED_TOPOLOGIES` in `conv_bwd_plan.py`. |
| T11 | F13 `CONVOLUTION`, F19 extra types and F35 override shape, each paired with every S shape in full | 198 | No engine executed these features on either architecture, and no reference can judge F13 or F19 (§6); each pairing repeats one decline. | Each feature × core in nightly (F13 18 cases, F19 108, F35 9). | An engine executes the feature: remove it from `UNEXECUTED_FEATURES`; for F13 and F19 the reference fixes of §6 must land too. |
| T12 | The geometry classes S24–S62 (and L24–L62) crossed with the fused topologies or the feature probes, in nightly and full | 338 (after T9) | As the forward plan's T12: a geometry class selects the convolution's own solver path; an epilogue or a per-tensor feature does not depend on it. | Each class in plain dgrad from PR up, over every pair in nightly and at L size in nightly and full. | As the forward plan's T12. |

The forward plan's T8 and T10 have no dgrad form: T8 trims unexecuted
activations of an executed topology, and T10 a data type of an executed
topology; no dgrad topology executes, so T9 covers both. F11 (asymmetric
padding), F15, F16, F20 and the F30, F31 fused variants are never trimmed,
whether or not they executed: F11 exists only in full, and the others hold the
evidence for the defects of §4.5.

### 4.4 Cost

| Estimate | Value |
|---|---|
| FLOP per case | as the forward plan: quick ≤ 14 M, PR ≤ 3.6 G (the three M cases), nightly ≤ 7.2 G, full 116 G (X2) |
| Oracle | **CPU reference only.** It is multi-threaded (`hardware_concurrency`). Measured with the engine: an executed case costs 0.4–0.7 s on average, the largest (X2, 116 GFLOP) about 10 s. |
| Compile and solver search | paid once per unique shape on a cold cache |
| Device memory | X1 ≈ 0.4 GB, X3 ≈ 13 GB (fp16), so `minimum_vram_mb` is set on those cases |

### 4.5 Measured run

**Which cases execute.** The trims T9 and T11 come from serial runs (one test at
a time, chunks of 100 tests per process) of the 5,556 cases the level rules
generate before any measured trim, with the MIOpen plugin, the only engine that
runs dgrad today:

| Architecture | OK | Skipped | Failed | Crashed | Wall time |
|---|---|---|---|---|---|
| gfx90a (MI250) | 1,624 | 3,325 | 592 | 15 | 29 min |
| gfx942 (MI325X) | 1,854 | 3,327 | 362 | 13 | 26 min |

On both architectures no fused graph executed (2,855 cases, all four
topologies), and F11, F13, F19, F30, F31 and F35 never executed. 401 (gfx90a) and
202 (gfx942) claim cases failed only because the default tolerance is below the
numerical floor (§6); the engine TOML's classes by accumulation length cover
them.

**The current set.** All 2,718 cases, serial, at this plan's tip, with the
MIOpen plugin and the engine TOML:

| Architecture | OK | Skipped | Failed | Crashed | Wall time |
|---|---|---|---|---|---|
| gfx90a (MI250) | 2,091 | 487 | 125 | 15 | 28 min |
| gfx942 (MI325X) | 2,083 | 489 | 133 | 13 | 16 min |

No claim case fails on either architecture; every failure is a probe, and each
is a graph the MIOpen plugin accepts and then fails on:

| Graph | gfx90a | gfx942 | What happens |
|---|---|---|---|
| F20 mixed dy, w, dx types | 74 of 95 accepted: 13 crash, 44 fail at execute, 17 run with no oracle | 72 of 95 accepted: 13 crash, 42 fail at execute, 17 run with no oracle | The plugin accepts whenever MIOpen offers a solution. Execute returns `miopenStatusBadParm` or `miopenStatusUnknownError`; the 3D cases then fault the device (error 700). dy fp32 with w fp16 runs, and no reference registers that type triple. Only fp16 with an fp16 compute type is declined. |
| F15 mixed layouts | 46 of 49 executed are wrong | 46 of 49 | Output disagrees with the CPU reference by the full output magnitude; the same graphs in one layout pass. |
| F16 non-packed strides | 18 of 91 executed fail (1 at execute) | 26 of 91 (1 at execute) | Wrong output, up to 1e35 in some channel-stride cases. |
| X3, 64-bit indexing | 2 of 2 fail | 2 of 2 | `miopenStatusUnknownError`; on gfx90a then an illegal memory access. |

These match the forward plan's mixed-precision, mixed-layout and
non-packed-stride findings. They are left failing, not skipped, so each shows in
every run until the plugin declines the graph or MIOpen computes it correctly.
Claims are harvested from these runs: 1,669 claim cases on each of gfx90a and
gfx942 (linux), every claim case that executed. The other 35 claim cases are
`dact` with the ReLU family, which no engine executes.

---

## 5. Golden reference

The selection follows [methodology §4.2](methodology.md#42-which-cases-should-be-golden).
Dgrad needs golden data more than forward does: with no GPU reference, every case
has exactly one live oracle. The cases are generated by PyTorch
(`torch.nn.grad.conv{1,2,3}d_input`, which is the transposed convolution) on
CPU, with the backward pointwise ops applied in PyTorch for the fused cases, and
record `forward_source` so forward and backward golden data come from the same
computation.

| Id | Graph | Case | Criterion | Level |
|---|---|---|---|---|
| G1 | dgrad | 2D NCHW fp32, S3 | G-A baseline | quick |
| G2 | dgrad | 1D NLC fp16, S4 (stride 2) | G-A baseline | PR |
| G3 | dgrad | 3D NCDHW bf16, S3 | G-A baseline | PR |
| G4 | dgrad | 2D NHWC fp16, S1 (1×1) | G-A baseline | PR |
| G5 | dgrad | 3D NDHWC fp32, S4 | G-A baseline | PR |
| G6 | dgrad | 1D NCL bf16, S7 (grouped) | G-A baseline | PR |
| G7 | dgrad → relu_bwd(·, x) | 2D NHWC fp16, S3 | G-A baseline; G-C clip semantics of `relu_bwd` | quick |
| G8 | relu_bwd(dy, y) → dgrad | 2D NCHW bf16, S3 | G-A baseline (prologue) | PR |
| G9 | dgrad → add(dx_residual) | 2D NCHW fp32, S3 | G-A baseline | PR |
| G10 | dgrad → add → relu_bwd | 2D NHWC bf16, S3 | G-A baseline | PR |
| G11 | dgrad, `CONVOLUTION` mode | 2D NCHW fp32, S3 | G-B oracle gap (F13): no hipDNN reference judges it | PR |
| G12 | dgrad | 2D NCHW fp32, S18 (pre 0, post 1) | G-C asymmetric padding convention | PR |
| G13 | dgrad | 2D NCHW fp32, S46 (stride 4 > filter 3) | G-C: dx elements no tap reaches are 0 (F37) | PR |
| G14 | dgrad → gelu_bwd(·, x) | 2D NCHW fp32, S3 | G-B oracle gap (F26: no CPU gelu_bwd) | PR |
| G15 | dgrad | 2D NCHW fp32, S19 (filter equal to the input, A = 32 × 256 = 8,192) | G-D: long accumulation | PR |

Fifteen cases. Each golden case is also a live sweep case. **The generator
(`generate_conv_bwd_golden.py`) does not exist yet (I7); no dgrad case has
golden data.** Precision strategy, confirmation and storage are the forward
plan's (§5 there); G11 and G14 are confirmed by a second independent
computation until the reference fixes land.

---

## 6. Pass criteria and reference prerequisites

| Item | Rule |
|---|---|
| Verification | `auto`: golden → GPU reference → CPU reference. For dgrad the GPU step always declines, so `auto` is the CPU reference until golden data exists. |
| Tolerance | `TestTolerances.hpp` `conv::getToleranceBwd`: fp32 8.5e-6, fp16 and bf16 2e-2, used as atol and rtol; no `double` value exists. As in the forward plan, the fp32 value is below the numerical floor of a long reduction; engine TOMLs carry tolerance classes by accumulation length A (§1), calibrated from observed error. |
| Support claims | Enforced. Only `claim` cases are recorded (§4.2). |

These features need reference or harness work before their cases can be judged
or run. The generator emits their cases now; they are unclaimed until the work
lands.

| Feature | Gap | Needed | RFC 0021 |
|---|---|---|---|
| All dgrad cases | No GPU reference for `ConvolutionBwdAttributes`: X shapes and long L cases cost minutes on the CPU reference, and every case has a single live oracle. | A GPU dgrad reference. | I6 |
| F13 `CONVOLUTION` | The CPU reference rejects it; no GPU reference. An accepting engine fails "no oracle". | Flip the filter in the reference; until then golden data (G11). | I6 |
| F19 fp8, int8, double; F20 mixed types | The CPU reference registers only the three uniform core triples. | Register the types, or golden data. | I6, I7 |
| F26 elu, gelu, gelu-tanh, swish, softplus backward | The CPU pointwise reference implements only relu, sigmoid and tanh backward. | Implement them; until then golden data (G14). | I6 |
| F11 asymmetric padding | The CPU reference takes pre and post separately but has no asymmetric dgrad test. | Add one; golden G12. | I6 |
| F21 subnormal, zero-excluded, sparse; F22 NaN / Inf | As the forward plan. | I4. | I4 |
| F17 alignment; F24, F32–F34 | As the forward plan. | I3, I5. | I3, I5 |
| `purpose` | The harness ignores the tag. | As the forward plan. | I2 |

Engine-specific tolerances and skips go in the engine TOML, keyed to the
generated case ids:
`<plan items>_<rank>d_i<dx spatial>_n<N>c<C>k<K>[g<G>]_f<filter>_<dtype>[_w<dtype>_dx<dtype>]_<layout | dy..w..dx..>[_<activation>]`.
A change to the generator that renames ids updates the TOML globs in the same
pull request.

---

## 7. Exhaustive mode

As the forward plan's §7. The declarations live in `conv_fwd_plan.py`
(geometry, legality, shape ranges) and `conv_bwd_plan.py` (`EXTRA_TYPES`,
`MIXED_TYPES`, `ACTS`, `TOPOLOGIES`, `DEFAULTS`, `value_recipes()`,
`VALUE_CLASSES`).
