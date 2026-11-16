# Convolution Forward Test Plan

This plan lists every feature of convolution forward (`ConvolutionFwdAttributes`)
and its pointwise fusions that a complete test sweep must exercise, and how the
sweep is spread across levels. The sweep describes the problem space and does
not derive it from any engine; each engine's support claims decide which cases
it runs. The one place the plan rests on an engine is the trims T8, T9 and T11
(§4.3), which come from measured MIOpen runs. Backward-data (`ConvolutionBwd`)
and weight gradient (`ConvolutionWrw`) are out of scope; each gets its own plan
in this folder (`convolution-bwd.md`, `convolution-wrw.md`).

The plan follows [RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md) (Per-Op Test Plans); [§8](#8-how-the-sweep-was-built) states the method this plan used.
See [the test-plans index](README.md) for the plans and the tier definitions.

The bundles are **generated** from this plan. Nobody edits them by hand:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_conv_fwd_bundles.py \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles --report
```

The [reference data scripts README](../../reference-data-scripts/README.md#generate_conv_fwd_bundlespy)
describes the files, the directories the generator rewrites and the claim rule.

| Plan item | Where it lives |
|---|---|
| Axes, legality, shape catalog, ceilings, activations, value domains, trimmed products (`TRIMS`) | `reference-data-scripts/conv_fwd_plan.py` |
| R set: corpus pick, selection rule, provenance | `reference-data-scripts/conv_fwd_corpus.json` |
| Level rules, graph templates, sweep writer, purpose tags, trims | `reference-data-scripts/generate_conv_fwd_bundles.py` |
| Bundles | `integration-test-bundles/<level>/ConvolutionFwd/Default` and `ConvolutionFwdPointwise/Default` and `Variant2` in every level; `Variant3` in quick, PR and nightly; `Variant4` to `Variant6` in quick and PR |

Every sweep case records the plan items it implements (`metadata.plan_items`),
its `purpose` (§4.2) and its size class. That is the traceability in both
directions: a plan item names its cases with a filter on `plan_items`, and a
case names its items. When two candidates produce the same graph, the generator
keeps the one from the lower level (or the earlier rule) and writes the sorted
union of both candidates' plan items into its `plan_items`; its id and seed stay
those of the kept candidate.

The features come from the graph API:

- `flatbuffers_sdk/schemas/convolution_fwd_attributes.fbs`
- `tensor_attributes.fbs`
- `pointwise_attributes.fbs`
- `data_types.fbs`
- `graph.fbs`
- the frontend rules in `hipdnn_frontend/node/ConvolutionFpropNode.hpp`

---

## 1. The operation

```
y[n, k, o...] = Σ_{c in group(k)} Σ_{r...} x[n, c, o·stride − pre_pad + r·dilation] · w[k, c', r...]
```

| Tensor | Dims | Rule |
|---|---|---|
| x | N, C, spatial… | at least 3 dims; the API sets no upper limit |
| w | K, C/G, filter… | same rank as x; `C % w[1] == 0`, so G = C / w[1]; `K % G == 0` |
| y | N, K, out… | `out = floor((in + pre + post − (dil·(f−1)+1)) / stride) + 1`. Dims and strides are inferred when not set; inferred strides follow x's stride order. |

Parameters: `pre_padding`, `post_padding`, `stride` and `dilation` each give one
value per spatial dimension. `conv_mode` is `CROSS_CORRELATION` or `CONVOLUTION`.
The node `compute_data_type` and the graph's io, intermediate and compute data
types are set separately.

---

## 2. Feature inventory

Each feature lists the values a complete sweep must hit. The **Oracle** column
says whether the verification references can judge the feature today (§6). The
**Earliest level** column is the lowest level that holds a case of the feature
(`Q` quick, `PR`, `N` nightly, `F` full; `—` means the sweep cannot express it
yet).

**Legality** is declared once, in `conv_fwd_plan.legal()`, and copies
`ConvolutionFpropNode::pre_validate_node`: `C % G == 0`, `K % G == 0`, stride and
dilation ≥ 1, padding ≥ 0, one parameter per spatial dimension, and a filter no
larger than the padded input. The generator never writes an illegal graph.

### 2.1 Geometry

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F1 | Spatial rank | 1D, 2D, 3D. Also 4D spatial (6-dim tensors): the API accepts it, so either support or a clean rejection must be shown. | ready | Q (1D–3D), PR (4D) |
| F2 | Batch N | 1, 2, odd (3), 8, 32, 64, 256. N 7 is not covered (ALMIOPEN-3228). | ready | Q |
| F3 | Input channels C | 1, 3, 4, odd prime (13), 16, 32, 64, 128, 256, 512, 2048; values that are not a multiple of the vector width (8) | ready | Q |
| F4 | Output channels K | 1, odd (17), K < C (reduction), K == C, K > C (expansion), 1024–2048 | ready | Q |
| F5 | Groups G = C / w[1] | 1 (dense); 2, 4, 32 with K/G > 1; depthwise G = C with multiplier K/G = 1, 2, 4; C/G = 1 with K/G > 1 | ready | Q |
| F6 | Input spatial size | 1, odd (7, 15), 16, 32, model sizes (14, 28, 56, 112, 224), rectangular H ≠ W, long 1D (16000), large 3D (64³). Extent 2 and an extreme aspect input (1 × 256) are not covered (ALMIOPEN-3228). | ready | Q |
| F7 | Filter size | 1, 2 (even), 3, 5, 7, 11, 31 (1D); non-square 1×3, 3×1, 1×7, 7×1, 3×5; 3D 1×3×3, 3×1×1; filter equal to input (output 1); filter larger than input with padding | ready | Q |
| F8 | Stride | 1, 2, 3, 4; stride that does not divide `in + pads − dilated filter` (floor remainder); stride larger than filter (input skipped); different per dimension | ready | Q |
| F9 | Dilation | 1, 2, 3, 4; different per dimension. A dilated filter equal to the padded input (output 1) is not covered (ALMIOPEN-3228). | ready | Q |
| F10 | Symmetric padding | 0, 1, 2, 3; "same" (f // 2); "full" (f − 1); padding ≥ filter (border outputs read only padding); different per dimension | ready | Q |
| F11 | Asymmetric padding | pre ≠ post: TF-style "same" with an even filter or stride 2 (pre 0, post 1), pre only, post only, different per dimension | ready: the GPU reference tests fp32, fp16 and bf16; the CPU reference tests fp32 in 2D and 3D (I6) | Q (probe) |
| F12 | Output size edge cases | output spatial 1; output larger than input (heavy padding); y dims and strides set explicitly vs inferred. **Inferred outputs are not a bundle case:** the graph JSON loader requires `dims` on every tensor (`hipdnn_flatbuffers_sdk/utilities/json/TensorAttributes.hpp`), and a sweep case that removes them is dropped at registration with no failure. They stay a C++ graph test. | explicit: ready; inferred: C++ test | Q (explicit) |
| F13 | `conv_mode` | `CROSS_CORRELATION`, `CONVOLUTION` (filter flipped) | **neither reference** (I6): the GPU reference declines the mode and the CPU reference runs cross-correlation | Q (probe) |

### 2.2 Memory layout and tensor kind

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F14 | Layout | channels-first and channels-last for each rank: NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | ready | Q |
| F15 | Mixed layouts | x, w and y in different orders (all six non-uniform combinations). Inferring y strides from x is a frontend test (§2.6). | ready | PR |
| F16 | Non-packed strides | padded rows (`fast`), channel or plane stride padded (`chan`, `plane`), on x and y; a view sliced out of a larger tensor. The weight tensor is never non-packed. | ready | PR |
| F17 | Alignment | `TensorAttributes.alignment` below the default 16 (2, 4, 8). **Tested by placement:** the base pointer sits at exactly the declared alignment (N, not 2N) with a non-zero offset, crossed with a leading stride that is not a multiple of the vector width. One case uses a pointer *less* aligned than declared and expects `BAD_PARAM`. A case that only declares a small alignment tests nothing, because allocator pointers are 256-byte aligned. | **not run**: I3 | — |
| F18 | 64-bit indexing | a tensor with more than 2³¹ elements (X3) | ready (GPU ref) | F |
| F35 | Override shape | `is_override_shape_enabled` true on an explicitly shaped graph (RFC 0008) | ready | PR |

**Tensor kind.** x, w, the conv output, bias and residual are device tensors in
every case. The scale operand of F27 is the only one that varies: compile-time
constant, runtime pass-by-value scalar ([RFC 0016](../../../../projects/hipdnn/docs/rfcs/0016_RuntimePassByValueTensors.md)),
or per-channel device tensor. The plan does not cover a conv weight or input that
is a constant or a runtime scalar, so those kinds are not axes of this op.

### 2.3 Data types

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F19 | IO data type | fp32, fp16, bf16; fp8 (E4M3, E5M2 and their FNUZ forms); int8 with int32 output; double | fp32, fp16, bf16 ready; fp8, int8 and double have no reference (§6) | Q (fp32, fp16, bf16; fp8 e4m3 and double as probes) |
| F20 | Mixed precision | y data type ≠ x data type (fp16 in, fp32 out); x data type ≠ w data type; node compute type fp32 vs fp16. Graph-level io, intermediate and compute types that fill per-node types are a frontend test (§2.6). | ready for the two fp32-output pairs (fp16 in and bf16 in); no reference for mixed x and w types or fp16 compute (§6) | PR |

Out of scope: the `DataType` values UINT8, INT4, INT64, BOOLEAN, FP8_E8M0, FP4_E2M1, FP6_E2M3 and FP6_E3M2. The plan has no case for them.

### 2.4 Values and numerics

The fill recipes of a bundle are `free` (uniform in [lo, hi]) and `fixed`
(constant), set per tensor in `metadata.inputs`, with one derived seed for each
tensor. The generator writes the fill of every case explicitly: x and w are
uniform in [−1, 1] unless the value class below says otherwise.

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F21 | Input distribution | uniform [−1, 1] (default); all zero x; constant (x = w = 1, exact result); large magnitude (x, w uniform in [0, a] with a = √(½·max/accumulation), so outputs approach the dtype maximum and cannot cancel to near zero, where a rounding error of the large intermediates would exceed any absolute tolerance); tiny (fp32/bf16 ±1e-20, fp16 ±3e-3: products are subnormal or underflow). A tiny case catches NaN, Inf and faults only: every output is below atol, so a wrong value passes. Not expressible yet: subnormal inputs, zero-excluded, sparse, cancelling by construction (I4). | ready for the five above | PR |
| F22 | Special values | NaN and ±Inf in x or w must propagate to the right outputs | needs I4 and the `allclose_matching_infinities` validator | — |
| F23 | Accumulation length | reduction length C/G × filter volume: 1, ~100, ≥ 4096 (A1: C 512 × 3×3 = 4,608), 3,456 (A2, 3D), 18,432 (X4) | ready | N |
| F24 | Determinism | repeated execution gives bitwise-identical output when a deterministic plan is requested | **C++ test** (I5) | — |

### 2.5 Fusions

Fused graphs put pointwise nodes after the conv. The conv output may be virtual
or materialised. The topologies are the bundle directories:

| Topology | Bundle | Nodes |
|---|---|---|
| `act` | `ConvolutionFwdPointwise/Default` | conv → activation |
| `bias_act` | `Variant2` | conv → bias add → activation |
| `scale_bias_act` | `Variant3` | conv → scale (`mul`) → bias add → relu |
| `add_relu` | `Variant4` | conv → add(residual) → relu |
| `bias_add_relu` | `Variant5` | conv → bias add → add(residual) → relu |
| `act_add` | `Variant6` | conv → relu → add(residual) (activation before the add) |

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F25 | Bias add | per-channel `[1, K, 1…]`; full-tensor; scalar; bias data type ≠ io; bias on `in_0` vs `in_1` of the add | ready | Q (per-channel) |
| F26 | Activation | `relu_fwd`: plain, lower clip, upper clip (ReLU6), clamp (lower and upper), leaky (`relu_lower_clip_slope`). Also `sigmoid_fwd`, `tanh_fwd`, `elu_fwd` (α = 1), `gelu_fwd`, `gelu_approx_tanh_fwd`, `swish_fwd` (β = 1), `softplus_fwd` (β = 1), `identity` | ready | Q |
| F27 | Scale | multiply by a scalar (compile-time constant 0.5 and runtime pass-by-value); per-channel scale (folded batch norm) | ready | PR |
| F28 | Residual add | add a second full-size tensor (ResNet shortcut) | ready | Q |
| F29 | Chains | the six topologies above | ready | Q |
| F30 | Outputs | conv y virtual vs materialised as a second output | ready | PR |
| F31 | Fused intermediate type | virtual conv y in fp32 vs in the io type | ready | PR |

The values of `elu_alpha`, `swish_beta` and `softplus_beta` are always 1.0, so an
engine that ignores them still passes (ALMIOPEN-3228). Zero lower clip
(`relu_lower_clip` 0) and zero slope (`relu_lower_clip_slope` 0) are written
explicitly; both must run as plain ReLU.

### 2.6 Execution

These cannot be expressed as a bundle case. They stay C++ tests
(`src/integration-tests/conv/`; for example `IntegrationConvFwdSerializeRoundTripFp32`
in `IntegrationConvForwardSerializeRoundTrip.cpp`) until the harness can repeat
and compare (I5).

| Id | Feature | Values to cover |
|---|---|---|
| F32 | Workspace | zero workspace limit; unlimited |
| F33 | Reuse | execute the same plan twice with new data and new buffer pointers |
| F34 | Serialisation | graph and plan serialise → deserialise → execute (3D, fp16, bf16, fused are missing) |

Two inventory items are frontend behaviour, not bundle cases: the harness loads a
serialized graph and never runs the frontend's `validate()` or property
inference. They stay frontend C++ tests:

- F15: y strides inferred from x;
- F20: graph-level io, intermediate and compute types filling per-node types.

### 2.7 Invalid problems

These are frontend unit tests, not sweep cases. Each must be rejected with
`INVALID_VALUE`. This deviates from RFC 0022 §6, which asks for an
`expected_rejection` bundle for each rejection an engine can see; see the
deviations after §4.3 (D1).

- `C % w[1] ≠ 0`
- `K % G ≠ 0`
- stride or dilation < 1
- negative padding
- rank mismatch between x, w and y
- parameter count ≠ spatial rank
- y dims inconsistent with the output formula
- filter larger than the padded input

---

## 3. Shape catalog

The sweep draws its shapes from six sets:

- **S** shapes are small and isolate one feature.
- **L** shapes are the same features at medium size.
- **M** shapes are named model layers.
- **R** shapes are real model geometries from the dnn-benchmarking corpus.
- **H** shapes sit on hazard breakpoints.
- **X** shapes are stress cases.

A feature that only ever runs at toy size misses the bugs that appear when the
engine switches to a different solver or tiling at realistic channel counts and
spatial sizes. Each level therefore raises the size ceiling: PR retests every
feature small, nightly retests it at medium size, and full adds stress.

| Level | Per-case ceiling (fp32) | Shape sets of the level's own cases |
|---|---|---|
| quick | ≤ 20 MFLOP, ≤ 1 MB | S; 3D at 16×8×4, 16×8×8, 8×8×4 and 8×4×4 |
| PR | ≤ 200 MFLOP, ≤ 16 MB | S, H, R, plus three M cases (M2, M9, M11); M2 and M9 exceed the ceiling by design |
| nightly | ≤ 8 GFLOP, ≤ 64 MB | S, L, M, R, H, A |
| full | none | S, L, R, X |

FLOP counts 2 × multiply-accumulates; MB is the sum of x, w and y in fp32. A 3D S
shape takes the first of four spatial sizes on which it fits the level: 16³,
16×8×8, 16×8×4, 8×8×4. The rectangular S6 shape is 16×8×4 on the first three
and 8×4×4 on the last. A shape that fits none is left out of that level.

**Defaults are non-degenerate.** Every S and L shape has K > G unless it is marked
depthwise. A depthwise shape (S13) has C = G and K = G or 2G; that is a
multiplier of 1 or 2, not the K = G collapse of a dense conv.

### 3.1 Small (S)

Defaults: N 2, C 16, K 32, G 1, spatial 16 per dimension, stride 1, dilation 1,
filter 3, pad 1 unless the row says otherwise. N is 2 rather than 1 so that no
default dimension collapses to 1 (RFC 0022 §7); H15 keeps a batch-1 case.

| Id | Features | Parameters | Ranks |
|---|---|---|---|
| S1 | F7 1×1 | filter 1, pad 0 | 1D 2D 3D |
| S2 | F10 no pad | filter 3, pad 0 | 1D 2D 3D |
| S3 | F10 same | filter 3, pad 1 | 1D 2D 3D |
| S4 | F8 | filter 3, stride 2, pad 1 | 1D 2D 3D |
| S5 | F9 | filter 3, dilation 2, pad 2 | 1D 2D 3D |
| S6 | F6 rectangular | spatial 16×8 / 16×8×4, filter 3, pad 1 | 2D 3D |
| S7 | F5 grouped | G 2, filter 3, pad 1 | 1D 2D 3D |
| S8 | F2 F5 F8 F9 | N 2, C 32, G 4, K 64, filter 3, stride 2, pad 1, dilation 2 | 1D 2D 3D |
| S9 | F2 | N 8, filter 1 | 1D 2D 3D |
| S10 | F7 | filter 5, pad 2 | 1D 2D 3D |
| S11 | F3 F7 stem | C 3, K 16, filter 7, stride 2, pad 3 | 1D 2D 3D |
| S12a, S12b | F7 non-square | S12a: 1×3, pad (0, 1); 3D 1×3×3. S12b: 3×1, pad (1, 0). | S12a 2D 3D; S12b 2D |
| S13a, S13b | F5 depthwise | G = C = 16; K 16 (S13a) and 32 (S13b); filter 3, pad 1 | 1D 2D 3D |
| S14 | F3 F4 F6 odd | C 13, K 17, spatial 15, filter 3, pad 1 | 1D 2D 3D |
| S15 | F8 F9 F10 per-dimension | stride (1, 2, 1), dilation (2, 1, 2), pad (1, 0, 1) | 2D 3D |
| S16 | F8 F9 F10 large | filter 3, stride 3, dilation 3, pad 3, spatial 32 | 1D 2D 3D |
| S17 | F3 | C 1, K 16, filter 3, pad 1 | 1D 2D 3D |
| S18 | F11 | filter 2, stride 2, pre 0, post 1 | 1D 2D 3D |
| S19 | F12 output 1 | filter equal to input, pad 0 | 1D 2D 3D |
| S20 | F8 remainder | spatial 16, filter 3, stride 3, pad 0 ((16 − 3) mod 3 = 1) | 1D 2D 3D |
| S21 | F7 even | filter 2, pad 1 | 1D 2D 3D |
| S22 | F10 pad ≥ filter | filter 3, pad 4 | 1D 2D 3D |
| S23 | F1 4D spatial | 6-dim tensors, N 1, spatial 6×6×4×4, filter 3, pad 1 | 4D |

S24–S46 are the convolution classes that real models use and S1–S23 missed; the
corpus analysis of §3.3 found them. Each still changes one thing from the
defaults, so each class is tested small in PR and at medium size (L24–L46) in
nightly. In nightly and full they are crossed with layout and data type only,
not with the fused topologies or the feature probes (T12, §4.3).

| Id | Class | Parameters | Ranks | Source |
|---|---|---|---|---|
| S24 | grouped, strided | G 4, stride 2 | 1D 2D 3D | ResNeXt, RegNet |
| S25 | depthwise, strided | G = C = K, stride 2 | 1D 2D 3D | MobileNetV3, EfficientNet, ir-CSN |
| S26 | depthwise 5×5, strided | G = C = K, filter 5, stride 2, pad 2 | 2D | MobileNetV3 |
| S27 | depthwise 7×7 | G = C = K, filter 7, pad 3 | 2D | ConvNeXt |
| S28 | depthwise multiplier 4 | G = C, K = 4C | 1D 2D 3D | F5 |
| S29 | patchify 2 | filter 2, stride 2, pad 0 | 1D 2D 3D | ConvNeXt, UNet, nnU-Net |
| S30 | patchify 4 | filter 4, stride 4, pad 0 | 2D | ConvNeXt stem |
| S31 | patchify 16 | C 3, filter 16, stride 16, pad 0, spatial 32 | 2D | ViT patch embedding |
| S32 | temporal 3×1×1 | filter (3, 1, 1), pad (1, 0, 0) | 3D | R(2+1)D, WAN VAE |
| S33 | spatial 1×3×3, strided | filter (1, 3, 3), stride (1, 2, 2), pad (0, 1, 1) | 3D | R(2+1)D |
| S34 | pointwise downsample | filter 1, stride 2, pad 0 | 1D 2D 3D | ResNet, ir-CSN |
| S35 | vocoder dilation | filter 7, dilation 3, pad 9 | 1D | HiFi-GAN |
| S36 | vocoder dilation | filter 11, dilation 5, pad 25 | 1D | HiFi-GAN |
| S37 | dilation 4 | dilation 4, pad 4 | 1D 2D 3D | DeepLabV3 |
| S38 | atrous | dilation 12, pad 12 | 2D | DeepLabV3 ASPP |
| S39 | strided, no pad | stride 2, pad 0 | 1D 2D 3D | Inception |
| S40 | 5×5, no pad | filter 5, pad 0 | 2D | Inception |
| S41 | long strided filter | filter 16, stride 8, pad 4 | 1D | HiFi-GAN |
| S42 | strided filter 4 | filter 4, stride 2, pad 1 | 1D | HiFi-GAN |
| S43 | F7 non-square | filter 3×5, pad (1, 2) | 2D | F7 |
| S44 | F11 pre-only padding | filter 2, stride 2, pre 1, post 0 | 1D 2D 3D | F11 |
| S45 | F7 filter larger than the input | filter = spatial + 2, pad 1 (output 1) | 1D 2D 3D | F7 |
| S46 | F8 stride 4 | stride 4 | 1D 2D 3D | F8 |

S47–S62 are 2D classes that S1–S46 and the corpus pick both miss: depthwise
with odd or small channel counts, depthwise on non-square and odd images,
depthwise with dilation, unequal strides or padding other than "same", grouped
with dilation at stride 1, and dense convolution with strides or dilation on
non-square images. They came from comparing the plan with what a direct-kernel
engine serves; each is a generic API hazard, not an engine rule. Before them,
every depthwise case had an even C ≥ 16, and every non-square image ran at
stride 1 without dilation. Values are the S size; the L form keeps the defining
channel count, stride, dilation and padding and grows the image (and C = K = 128
where C is not the feature).

| Id | Class | Parameters (2D) |
|---|---|---|
| S47 | depthwise, odd C < 16, 7×7 | G = C = K 5, 9², filter 7, pad 3 |
| S48 | depthwise, odd C, odd image | G = C = K 7, 13², filter 3, pad 1 |
| S49 | depthwise stride 2, odd C, odd image | G = C = K 5, 17², stride 2 |
| S50 | depthwise C ≥ 64, tall non-square image | G = C = K 64, 70×40 |
| S51 | depthwise stride 2, non-square, even height | G = C = K 96, 16×18, stride 2 |
| S52 | depthwise stride 2, odd image, C ≥ 64 | G = C = K 96, 17², stride 2 |
| S53 | depthwise, dilated | G = C = K 32, 15², dilation 2, pad 2 |
| S54 | depthwise, unequal strides | G = C = K 32, stride (1, 2) |
| S55 | depthwise, no padding | G = C = K 32, pad 0 |
| S56 | depthwise, padding larger than "same" | G = C = K 32, pad 2 |
| S57 | depthwise filter > 7, C < 64 | G = C = K 5, 56×24, filter 17, pad 8 |
| S58 | grouped (not depthwise), dilated, stride 1 | G 2, C = K 16, 15², filter 5, dilation 2, pad 4 |
| S59 | non-square filter, unequal stride, non-square image | C 32, K 64, 11×17, filter 3×5, stride (1, 2), pad (1, 2) |
| S60 | dense strided, non-square odd image | C 32, K 64, 17×19, stride 2 |
| S61 | dense dilated, non-square image | C = K 32, 15×17, dilation 2, pad 2 |
| S62 | depthwise multiplier 2, odd C | N 3, G = C 5, K 10, 12² |

S8 already crosses groups with dilation, but only at stride 2; S58 is the
stride-1 form. S1–S62 make 127 shape × rank combinations. Each is generated per
layout and data type as the level rules (§4.1) require.

### 3.2 Medium (L)

L*n* has the same features as S*n* (S1–S22 and S24–S62; there is no L23). What changes is the
base size:

| Rank | N | C = K | Spatial |
|---|---|---|---|
| 1D | 4 | 128 | 1024 |
| 2D | 4 | 128 | 56² |
| 3D | 2 | 32 | 32³ |

- **Features that set a value replace the default:**
  - C 3 (L11), C 1 (L17);
  - C 61, K 67 and spatial 55 / 31³ / 1023 (L14);
  - G = C (L13), G 4 (L8);
  - spatial 56×28 / 32×32×16 (L6).
- **Scaling rule:** if a shape exceeds the nightly ceiling, the generator halves N,
  then C and K, until it fits. Examples: L10 2D (filter 5) runs at N 2; L19
  (filter = input) runs at C = K = 64 in 1D and 2D.
- **Size range:** about 0.3 MFLOP–6.6 GFLOP and 0.3–56 MB per case.

### 3.3 Model (M) and corpus (R)

**M** is twelve named layers, chosen by hand:

| Id | Source | N, C, spatial → K | Filter, stride, pad, groups |
|---|---|---|---|
| M1 | ResNet-50 stem | 8, 3, 224² → 64 | 7×7, s2, p3 |
| M2 | ResNet-50 3×3 | 8, 64, 56² → 64 | 3×3, p1 |
| M3 | ResNet-50 1×1 | 8, 256, 56² → 64 | 1×1 |
| M4 | ResNet-50 downsample | 8, 512, 28² → 1024 | 1×1, s2 |
| M5 | ResNeXt grouped | 8, 128, 56² → 128 | 3×3, p1, G 32 |
| M6 | MobileNetV2 depthwise | 8, 96, 56² → 96 | 3×3, p1, G 96 |
| M7a, M7b | Inception non-square | 8, 192, 17² → 192 | 1×7 p(0, 3); 7×1 p(3, 0) |
| M8 | EfficientNet 5×5 depthwise | 8, 240, 28² → 240 | 5×5, p2, G 240 |
| M9 | UNet3D | 2, 32, 32³ → 32 | 3×3×3, p1 |
| M10 | I3D stem | 2, 3, 16×112² → 64 | 7×7×7, s2, p3 |
| M11 | Conformer depthwise | 8, 256, 128 → 256 | 31, p15, G 256 |
| M12 | wav2vec feature extractor | 4, 1, 16000 → 512 | 10, s5 |

**R** is drawn from the convolution corpus of
[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking)
(`Workloads/corpus/Convolution`): 4,546 unique forward geometries recorded from
live `aten::convolution` calls of 31 models in 8 families (classification,
detection, diffusion, segmentation, super-resolution, video, 3-D medical,
audio). The S, L and M sets alone missed 35 of the corpus's 60 behaviour
signatures (strided grouped and depthwise convs, ConvNeXt 7×7 depthwise,
patchify convs, (2+1)D video convs, vocoder dilations, atrous dilations), and
where they matched a signature it was often at a small fraction of real size.

- **Selection.** A behaviour signature is (effective rank, group class, filter,
  stride, dilation, padding class): what selects a solver path, not the exact
  size. Each signature's graphs are split into exclusive ceiling bins (PR: fits
  the PR ceiling; nightly: fits the nightly ceiling but not PR, or every graph
  that fits nightly if none does; full: up to 120 GFLOP and 1 GiB but not
  nightly), and R takes the median-FLOP graph of each bin. Every model gets at
  least one graph in PR or nightly. A graph picked for two levels sits in the
  lower. The exact rule, with its tie and rank details, is the `selection`
  string in `conv_fwd_corpus.json`.
- **Counts.** PR 30, nightly 60, full 47: 137 cases covering all 60 signatures
  and all 31 models.
- **Cases keep the model's data type and layout** (mostly fp16 and NCHW). The
  pairwise dtype × layout cover comes from S and L, so R adds realism and solver
  paths, not crossing. Audio models keep the corpus form of a 1D conv: 2D with
  H = 1.
- **Purpose and provenance.** R cases are `claim` cases. `plan_items` holds the
  R id (R001…R137); `metadata.notes` names the corpus shape id, the models and the
  family. The pick, the selection rule and the pin (repository commit and archive
  md5) are data in `conv_fwd_corpus.json`. Each case's `why` records how the rule
  chose it: its signature, the ceiling bin, how many candidate graphs the rule
  ranked (the signature's graphs in that bin, or for a model-coverage pick the
  model's graphs that fit nightly), the pick's position in FLOP order (the
  median), and the reason (signature median, or the one graph that gives a model
  its first case).
  A corpus update is a deliberate regeneration.
- **Limits of the corpus.** It was captured at small batch on one GPU, with
  `MIOPEN_FIND_MODE=fast`, and holds no fused graphs and no int8 or fp8. S, H
  and X keep large batch and the awkward cases; the fused topologies keep their
  synthetic cover.

### 3.4 Hazard (H)

One change from the S defaults (N 2, C 16, K 32, 16 per dimension, filter 3,
pad 1). Powers of 2 ± 1, dims that are not multiples of 4 or 8, and batch sizes
that real models use.

| Id | Change |
|---|---|
| H1–H6 | C and K: 15/32, 17/33, 31/31, 33/65, 63/64, 65/63 |
| H7, H8 | K 7 (not a multiple of the vector width); K 1 (single output channel) |
| H9 | N 3 |
| H10, H11 | G 3 (C 12, K 12); G 5 (C 15, K 20) |
| H12–H14 | spatial 31, 33, 17 |
| H15 | N 1 (the defaults use N 2, so H15 holds the batch-1 case) |
| H16, H17 | N 32, N 64 (training batches; 9% of corpus geometries each run at N 64 and 128) |
| H18 | C 4 (F3, not a multiple of the vector width) |

Non-dense strides are F16. The extreme int32-index case is X3.

### 3.5 Accumulation (A) and stress (X)

| Id | Shape | Features |
|---|---|---|
| A1 | 2D, N 1, C 512 → K 64, 14², 3×3 (accumulation 4,608) | F23 |
| A2 | 3D, N 1, C 128 → K 32, 8³, 3×3×3 (accumulation 3,456) | F23 |
| X1 | M2 at N 256 | F2 |
| X2 | 3×3×3, N 2, C = K 64, 64³ | F6 |
| X3 | N 1, C = K 1, 46341², filter 1 (y has 2,147,488,281 elements > 2³¹); fp16 and bf16 | F18 |
| X4 | C 2048 → K 64, 14², 3×3 (accumulation 18,432) | F23 |

---

## 4. Building the sweep

The sweep is a set of cases over three core axes plus feature axes.

- **Core axes:** rank × layout × data type. 1D, 2D and 3D each have two
  layouts, and there are three float types: 18 combinations.
- **Feature axes:** everything in §2 that is not core.

| Rule | Meaning |
|---|---|
| Core cover | Every level, together with the levels below it, holds every core combination at least once, for plain conv and for **each** fused topology (seven topologies × 18). In quick, at least one engine executes each 2-D and 3-D core cell of plain conv, `act` and `bias_act` (RFC 0022 §8.1): the core matrix crosses them with every 2-D and 3-D rank × layout × data type cell (36 cells) on S3, with plain `relu`, which MIOpen executes on gfx942; gfx90a declines every fused graph. The 1-D cells of `act` and `bias_act` rotate the ReLU family and the shapes, and no engine executes them in quick. Of the other four topologies, MIOpen declines every graph except fp32 `add_relu`, which it accepts and computes wrongly (ALMIOPEN-3009). ALMIOPEN-3008 tracks the defects that cause the declines that are workarounds. |
| One feature at a time | Each S and L shape changes one feature from the defaults, so a failure points at that feature. |
| Size grows with level | Each level has a size ceiling (§3). PR proves a feature works; nightly proves it still works at medium size. |
| Pair cover | The level rules loop over rank, layout and data type, and rotate the other values by position (`PAIRS[(i + offset) % 6]`, `acts[k % 13]`, `MIXES[…]`). The generator does not construct an all-pairs set, check pair coverage or use a selection seed. By the end of nightly, loops give every S shape every layout × data type pair of its rank, and give `conv_mode`, the extra data types and the value classes every rank × data type combination; mixed layouts (two mixes) and non-packed strides (three kinds) are placed once per rank × data type. Full gives every S shape one value of each feature, by rotation. |
| Full | Complete products only where two features plausibly interact (activation × core, L × core). §4.3 lists the larger products that are deliberately not run. |
| Probes | Every unsupported feature, awkward combination and hazard has a probe (§4.2). Every level holds rejection probes. |
| Concrete graphs | Each level runs checked-in graphs only. The generator is deterministic; random sampling belongs to the exhaustive tool (§7). |
| Unsupported stays in | A feature that no engine supports today is still in the sweep. It is unclaimed and skipped, and it shows up as `unclaimed_support` once an engine adds it. The trims T8, T9 and T11 (§4.3) are the one place the plan uses a measured engine result. |

### 4.1 Level contents

Each level answers one question (RFC 0022 §8.2). A case sits in the lowest level
that holds its graph; the levels cascade.

| Feature group | quick — what catches gross breakage? | PR — what must every PR test? | nightly — wider per engine | full — wider sweeps |
|---|---|---|---|---|
| Shape size | ≤ 20 MFLOP | ≤ 200 MFLOP (+3 M) | ≤ 8 GFLOP | unbounded |
| Plain core (18) | S1–S9, layout × dtype pair rotated per rank; core matrix: S3 × every 2-D and 3-D layout × dtype cell | every S shape × rank once, pair rotated (≥ 2× core) | every S shape × every pair of its rank | L × every pair |
| Geometry classes S24–S62 | — | each class × rank once | × every pair of its rank; L24–L62 × 2 pairs | L24–L62 × every pair; no fused or feature crossing (T12 for S24–S46; S47–S62 are never put in the fused and feature rotations) |
| Rejection probes | `CONVOLUTION`, asymmetric pad (S18), fp8, double | `CONVOLUTION` per rank; 6 extra types; 5 mixed pairs | × core | asymmetric padding, mixed layouts, strides and mixed types × every S shape; `CONVOLUTION`, extra types and override shape stay at nightly's × core (T11) |
| F13 `conv_mode` | S3 | 1 per rank | × core (rank × layout × dtype) | — (T11) |
| F15–F16 layout, strides | — | 1 case each | rank × dtype, 2 mixes and 3 stride kinds | paired with every S shape |
| F19–F20 types | fp8 e4m3, double | each extra type and mixed pair once | × rank × layout | F20 paired with every S shape; F19 — (T11) |
| F21 values | uniform | + zero, const, large | + tiny, × rank × dtype; A1, A2 (long accumulation) | × core; paired with every S shape; X4 |
| F35 | — | 1 case | × rank × dtype | — (T11) |
| H hazards | — | each H once (2D) | each H × every pair (2D); 1D, 3D × 2 pairs (H15–H18: batch 1, 32, 64 and C 4) | — |
| Model (M) | — | M2, M9, M11 | M1–M12 × every pair of its rank | — |
| Corpus (R) | — | 30 graphs (one per signature that fits) | 60 graphs (realistic size, every model) | 47 graphs (production size) |
| Stress | — | — | — | X1–X4 (X3 fp16 and bf16) |
| Fusion core (6 topologies × 18) | 1 per cell; ReLU family rotated; core matrix: `act` and `bias_act` with plain `relu` on S3 × every 2-D and 3-D layout × dtype cell; sigmoid on `act` (S3, 2-D, fp16 channels-first, the form MIOpen executes) | `act`, `bias_act`: 2× core; the others 1×; all 13 activations over the cells | `act` and `bias_act`: every S shape × every pair, activation rotated; the other four topologies: from quick and PR (T9) | `act`: the executed activations × core and × every S and L shape (T8); `bias_act`: activation × core; the other four topologies: from quick and PR (T9) |
| F26 zero lower clip, zero slope | — | `relu_clip0` and `relu_slope0` × `act`, `bias_act` × S2 2-D × (channels-first, fp32 and fp16) | — | — |
| Fusion features F25–F31 | — | each variant once | each variant × rank (`bias_act`, `scale_bias_act`, `act`) | — |
| Fusion at size | — | — | every L shape once per `act` and `bias_act`; M2, M3, M6, M9 × every pair | — |
| Golden (§5, deferred) | G1–G3, G9–G14 | — | G4–G6, G8, G15–G17 | G7 |

Cases generated (own level / cumulative):

| Level | Plain conv | Fused | Own | Cumulative | Claim | Probe | Regression |
|---|---|---|---|---|---|---|---|
| quick | 40 | 133 | **173** | 173 | 83 | 90 | 0 |
| PR | 193 | 172 | **365** | 538 | 179 | 186 | 0 |
| nightly | 1,334 | 1,020 | **2,354** | 2,892 | 1,260 | 1,092 | 2 |
| full | 965 | 909 | **1,874** | 4,766 | 1,254 | 620 | 0 |

By size class: quick S 173. PR S 314, H 18, M 3, R 30. Nightly S 1,613, L 385,
M 122, H 162, A 12, R 60. Full S 976, L 831, R 47, X 20. 541 distinct conv
geometries in all. The trims T8, T9, T11 and T12 (§4.3) remove 3,798 of the
generated synthetic cases, 879 duplicate graphs are merged into the case that
holds the same graph, and the R set adds 137. Counts are those of the generator;
`generate_conv_fwd_bundles.py --report` prints them again, with the trimmed and
merged count per level and the trim table.

### 4.2 Probes and regressions

Each case carries a `purpose` in its metadata (RFC 0022 §9):

| Purpose | Expected outcome | Claims |
|---|---|---|
| `claim` | A claiming engine accepts and is correct. | `--write-support-claims` records them. |
| `probe` | A typed rejection, or a correct accept. | Excluded from claims by the harvest used for this PR, which is outside the repository (ALMIOPEN-3231). |
| `regression` | Same as the original bug case. | Same as the original case. |

A case is a **probe** when it has at least one of these features. The generator
writes the reasons into `metadata.notes`:

| Probe class | Items |
|---|---|
| Unsupported feature | F13 `CONVOLUTION`; F11 asymmetric padding; F19 fp8, int8, double; F1 4D spatial; F26 non-ReLU activation except `sigmoid` and `tanh` in `act`; F26 ReLU with a nonzero lower clip or a slope (`relu_lo`, `relu_leaky`); F27–F29 scale, residual and chains beyond conv → bias → activation |
| Awkward combination | F15 mixed layouts; F16 non-packed strides; F20 mixed precision and compute type; F25 bias variants; F30 materialised conv output; F31 io-type intermediate; F35 override shape |
| Value domain | F21 zero, const, large, tiny |
| Hazard | every H shape; X3 (F18) |

Everything else is a `claim`: ranks 1–3, both layouts, fp32, fp16 and bf16, the
S, L, M, R and A shapes, per-channel bias, plain ReLU, ReLU6, clamp, zero lower
clip and zero slope, `sigmoid` and `tanh` in `act`, and X1, X2, X4.
When an engine adds a probe feature, the run reports `unclaimed_support`; the
plan then moves those cases to `claim` and the generator is edited, not the
bundles.

The regression cases so far are two, in nightly: conv → ReLU and conv → bias →
ReLU with `relu_lower_clip_slope` 0 (S3, 2D, fp32, channels-first), which must
run as standard ReLU (#14363). They are written by the generator after every
rotated case, so adding one moves no other case id; the activation encoding
lives in `ACT_ENCODINGS`, outside the activation rotation. Each new regression
lands in the lowest level that can afford it, with its source in `notes`.

`relu_clip0` (`relu_lower_clip` 0 written explicitly) and `relu_slope0` are also
`claim` cases in PR: S2, 2-D, channels-first, fp32 and fp16, in `act` and
`bias_act`. They give the accept side of an engine's lower-clip and slope checks
a bundle.

### 4.3 Products deliberately not run

The complete product of every feature with every other runs to about 14,000
cases, nearly all of them in `full`. Most of those cases exercise an interaction
that the conv cannot have, or one that a smaller set already covers. The table
lists the products this plan leaves out on purpose, what covers the same risk
instead, and what would justify adding them back. Removing a product here is a
decision, not an omission: a reviewer can see what was traded and why.

| Id | Product not run | Cases it would add | Why it adds little | Covered instead by | Add it back when |
|---|---|---|---|---|---|
| T1 | Every activation × every S / L / M shape × every layout and dtype × both fused topologies | ~10,000 | The activation is applied element by element to the conv output. It depends on the data type, the layout and the fusion path, not on stride, dilation, padding or groups. Repeating 13 activations over ~400 geometries re-runs the same epilogue ~30 times. | Activation × core, complete for both topologies (every activation on every rank, layout and dtype). Activation × every S and L shape once on the `act` topology for the activations an engine executes (T8). Plain conv covers geometry on its own. | A defect is found whose trigger needs a specific activation together with a specific geometry, or an engine picks fused kernels by geometry class. |
| T2 | Every extra data type (fp8 ×4, int8, double) and mixed-precision pair × every S shape × every layout | ~1,660 | A data type changes kernel selection and rounding. That is visible on every rank and layout, not on each geometry. No engine claims these types for conv today, so the cases would only skip. | Each type × rank × layout on S3 (nightly); each mixed pair also paired with every S shape once (full). | An engine claims the type: then run that type × core × S shapes. |
| T3 | `CONVOLUTION` mode × every S shape × every layout and dtype | ~380 | The mode only flips the filter. Index mistakes show up with any non-symmetric filter, on any layout. No engine supports it today. | Mode × core on S3 (nightly). | An engine adds support. |
| T4 | Mixed layouts, non-packed strides and alignment × every S shape × every dtype | ~580 | Layout handling is per tensor, not per geometry feature. | Mixed layouts and strides × rank × dtype (nightly), plus each paired with every S shape once (full). Alignment is not run (I3). | A layout bug depends on a geometry feature (for example strides combined with dilation). |
| T5 | The four non-activation fused topologies × every L shape and every pair | ~1,000 | Scale, residual and chain nodes are element-wise and see only the output tensor. | Each × core (quick, PR); the F27 scale variants (nightly). | An engine fuses these topologies. |
| T6 | Hazard shapes × every layout and dtype in 1D and 3D | ~500 | Hazards act on C, K, N, G and spatial extent. The layout and dtype multiply compile work, not hazard coverage. | 2D × every pair; 1D and 3D × two pairs (nightly). | A hazard is found that depends on rank. |
| T7 | Any 3-way product beyond activation × layout × dtype | not counted | Pairwise coverage catches the large majority of interaction faults at a fraction of the cases. | Pairwise (§4 rules). | A 3-way fault is found. Add that triple, not the whole product. |
| T8 | Activation × every S and L shape on `act` in full, for the activations no engine executes: ReLU lower clip, leaky ReLU, elu, gelu, gelu-tanh, swish, softplus, identity | 1,776 | No engine executed one of these graphs on gfx942 or gfx90a (§4.5), so each case repeats the same decline. | Each activation keeps all 18 core cells on `act` and on `bias_act`, and about 70 cases over about 65 distinct tensor shapes in quick, PR and nightly. | An engine executes the activation: add it to `EXECUTED_ACTS` in `conv_fwd_plan.py`, which restores its full product. |
| T9 | Plain repeats of `scale_bias_act`, `add_relu`, `bias_add_relu` and `act_add` in nightly and full | 476 | No engine executed a graph of these topologies on gfx942 or gfx90a (§4.5) except fp32 `add_relu`, which MIOpen accepts and computes wrongly (ALMIOPEN-3009); the quick and PR cases reach that defect too. | Each topology's 18 core cells in quick and PR (run by every level through the cascade); the F27 scale variants. | An engine executes the topology: remove it from `UNEXECUTED_TOPOLOGIES`. |
| T11 | F13 `CONVOLUTION`, F19 extra types and F35 override shape, each paired with every S shape in full | 198 | No engine executes these features, and no reference can judge F13 or F19 (§6); each pairing repeats one decline. | Each feature × core in nightly (F13 18 cases, F19 36, F35 9). | An engine claims the feature; for F13 and F19 the reference fixes of §6 must land too. |
| T12 | The geometry classes S24–S46 (and L24–L46) crossed with the fused topologies or with the feature probes, in nightly and full | 1,348 | They are geometry classes: they select solver paths of the convolution itself. An epilogue is element-wise (T1) and a layout or type feature is per tensor (T2, T4), so neither depends on which geometry class runs underneath. | Each class in plain conv in every level from PR up, over every layout × dtype pair in nightly and at medium size (L) in nightly and full; the fused topologies and features keep their cover on S1–S23. | A defect is found whose trigger needs one of these classes together with a fusion or a feature, or an engine picks fused kernels by geometry class. |

T8, T9 and T11 rest on measured execution, not on the API; T12 is a design trade-off
like T1–T7. All four are applied after the fact: the level rules build the
complete products, case ids are assigned, and the generator then removes the
trimmed cases (`conv_fwd_plan.TRIMS`). The cases that remain keep their ids,
seeds and claims, and undoing a trim brings back exactly the cases it removed.
The trims remove cases from nightly and full only, so every quick and PR case,
including the core matrix of §4, stays. No trim removes the last probe of a
feature from any level: every feature keeps its probes, directly or through the
cascade. Strict `expected_rejection` bundles are never trimmed.

T10 is retired; trim ids are not reused.

**Deviations from RFC 0022.**

| Id | RFC rule | This plan | Why | Tracked |
|---|---|---|---|---|
| D1 | §6: each rejection an engine can see is an `expected_rejection` bundle | The invalid problems of §2.7 are frontend unit tests; no `ConvolutionFwd` sweep holds an `expected_rejection` case. The harness never calls the frontend's `validate()`, so an engine that accepts one of these graphs is not caught by a bundle. | The bundle flag landed after the plan's sweeps; the engine-visible rules need choosing and confirming first. | ALMIOPEN-3232 |
| D2 | §12.3: precision uplift is the default for low-precision golden data | fp16 and bf16 golden data use matched precision (fp32 accumulation, rounded to the io type) (§5) | It matches the engines' accumulation contract and keeps the calibrated tolerance tighter than the op default of 0.1. This is the conv answer to the RFC's open question on golden precision. | ALMIOPEN-3230 |

### 4.4 Cost

| Estimate | Value |
|---|---|
| FLOP per case | 0.002 M – 116 G; quick ≤ 14 M, PR ≤ 3.6 G (the M cases), nightly ≤ 7.2 G, full 116 G (X2) |
| Oracle | GPU reference first. The CPU reference fallback costs seconds per M case and minutes per X case, so X cases must find a GPU reference. |
| Compile and solver search | Paid once per unique shape on a cold cache; it dominates execution. Measured in §4.5. |
| Device memory | X1 ≈ 0.4 GB, X3 ≈ 13 GB (fp16). The generator sets `minimum_vram_mb` on every case whose tensors exceed 1 GiB, so on the X3 cases. |

### 4.5 Measured run

One cold run of every level of the 5,469-case set on gfx942 (MI300X, 64
CPUs, one GPU) with the MIOpen plugin, in 60-test chunks across 10 processes. The
serial times of the current set are in the last table of this section.

| Level | Cases | Wall time |
|---|---|---|
| quick | 139 | 38 s |
| PR | 265 | 31 s |
| nightly | 1,900 | 42 s |
| full | 3,165 | 42 s |

The whole tree runs in under three minutes, against budgets of 5, 30, 60 min and
none. Two facts drive the numbers. A case that no engine accepts skips in about
1 ms, and most cases are of that kind. A case that executes costs 0.3–2 s of
compile and solver search; GPU execution is milliseconds even for X2. The
budgets therefore leave room to grow nightly by an order of magnitude, and the
estimate in §4.4 holds: compile and solver search, not execution, set the cost.
Run in chunks like this: one GPU fault (`hipErrorIllegalAddress`) makes every
later test in the same process fail with error 700, so a single faulting case
can hide thousands of results.

**Which cases execute.** The trims T8, T9 and T11 come from serial runs (one test at a
time) of the 5,469 cases the level rules generated before the R set, S24–S46,
H15–H18 and the N 2 default existed, with the MIOpen plugin, the only engine
that runs convolution today:

| Architecture | OK | Skipped | Failed | Crashed |
|---|---|---|---|---|
| gfx942 (MI300X; repeated on MI325X at the merged tip, same counts) | 1,875 | 3,594 | 0 | 0 |
| gfx90a (MI210) | 1,273 | 4,105 | 88 | 3 |

gfx942 runs with the engine TOML's skips; every gfx90a failure is a probe. On
gfx942 the `act` topology executed only ReLU, ReLU6, clamp, sigmoid and tanh;
`bias_act` executed only in fp32, because the generator typed the bias node
`float` while the bias tensor had the io type and MIOpen requires the two to
match (the generator now types the bias node by its bias tensor);
`scale_bias_act`, `bias_add_relu` and `act_add` never executed. gfx90a declined
every fused graph. On both architectures F13, F19 and F35 never executed.

**The current set.** All 4,766 cases at `e84e21ad0f`, serial, with the MIOpen
plugin:

| Architecture | OK | Skipped | Failed | Crashed | Wall time |
|---|---|---|---|---|---|
| gfx942 (MI308X) | 2,885 | 1,881 | 0 | 0 | 15.1 min |
| gfx90a (MI250) | 2,124 | 2,642 | 0 | 0 | 14.7 min |

The engine TOML's skips apply on every architecture, so neither run shows a
known defect class as a failing probe. In quick, gfx942 executes all 36 core
matrix cells; gfx90a executes the 12 plain-conv cells and declines the fused
ones. The skips cover three fp16 F25 bias variants MIOpen accepts: full-size and
scalar biases are read as per-channel (ALMIOPEN-3009 / 3014), and a bias as the
add's first operand runs but no reference implements that order (ALMIOPEN-3227).
X4 fp16 (accumulation length 18,432) varies run to run by up to 0.17 against the
GPU reference, so it has an interim fp16 class at atol 0.2 until the dynamic
tolerance of ALMIOPEN-3187. No claim case fails on either architecture. Claims are
harvested from these runs: 2,448 on gfx942 and 1,698 on gfx90a.

---

## 5. Golden reference

**Deferred.** None of the golden cases below exists yet, and the golden
generator (`generate_conv_fwd_golden.py`, RFC 0022 §15 I7) is not written. The
table is the selection the generator will implement; until then every case it
names runs live against the hipDNN references, like the rest of the sweep.

Golden cases are chosen by the criteria in
[RFC 0022 §12.2](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#122-policy-and-selection-criteria). Each one is to
be generated by PyTorch and must fit the 2 MiB per-case budget with finite values.
**G-A covers every rank × layout family and every data type for the plain op,
and every fused topology once.** The baseline is a cover, not the complete 18-cell
product: each of the six rank × layout families appears once with a rotating
data type, so each data type appears twice.

Every row is a target: it names an emitted sweep case by bundle directory and
id, and its level is the level that holds that case.

| Id | Graph | Bundle | Case | Criterion | Level |
|---|---|---|---|---|---|
| G1 | conv, 2D NCHW fp32, S3 | `ConvolutionFwd/Default` | `s3_2d_i16x16_n2c16k32_f3x3_fp32_cf` | G-A baseline | quick |
| G2 | conv, 1D NLC fp16, S4 | `ConvolutionFwd/Default` | `s4_1d_i16_n2c16k32_f3_fp16_cl` | G-A baseline | quick |
| G3 | conv, 3D NCDHW bf16, S3 | `ConvolutionFwd/Default` | `s3_3d_i8x8x4_n2c16k32_f3x3x3_bf16_cf` | G-A baseline | quick |
| G4 | conv, 2D NHWC fp16, S1 (1×1) | `ConvolutionFwd/Default` | `s1_2d_i16x16_n2c16k32_f1x1_fp16_cl` | G-A baseline | nightly |
| G5 | conv, 3D NDHWC fp32, S4 | `ConvolutionFwd/Default` | `s4_3d_i16x16x16_n2c16k32_f3x3x3_fp32_cl` | G-A baseline | nightly |
| G6 | conv, 1D NCL bf16, S7 | `ConvolutionFwd/Default` | `s7_1d_i16_n2c16k32g2_f3_bf16_cf` | G-A baseline | nightly |
| G7 | conv → bias → ReLU6, 2D NHWC fp16, S3 | `ConvolutionFwdPointwise/Variant2` | `s3_2d_i16x16_n2c16k32_f3x3_fp16_cl_relu6` | G-A baseline; G-C upper-clip semantics | full |
| G8 | conv → clamp, 2D NCHW bf16, S8 | `ConvolutionFwdPointwise/Default` | `s8_2d_i16x16_n2c32k64g4_f3x3_bf16_cf_relu_clamp` | G-A baseline; G-C lower and upper clip semantics | nightly |
| G9 | conv → scale → bias → ReLU, 2D NCHW fp32, S5 | `ConvolutionFwdPointwise/Variant3` | `s5_2d_i16x16_n2c16k32_f3x3_fp32_cf` | G-A baseline | quick |
| G10 | conv → add(residual) → ReLU, 2D NHWC bf16, S2 | `ConvolutionFwdPointwise/Variant4` | `s2_2d_i16x16_n2c16k32_f3x3_bf16_cl` | G-A baseline | quick |
| G11 | conv → bias → add(residual) → ReLU, 2D NCHW fp16, S9 | `ConvolutionFwdPointwise/Variant5` | `s9_2d_i16x16_n8c16k32_f1x1_fp16_cf` | G-A baseline | quick |
| G12 | conv → ReLU → add(residual), 2D NCHW fp32, S8 | `ConvolutionFwdPointwise/Variant6` | `s8_2d_i16x16_n2c32k64g4_f3x3_fp32_cf` | G-A baseline | quick |
| G13 | conv, `CONVOLUTION` mode, 2D NCHW fp32, S3 | `ConvolutionFwd/Default` | `s3_f13_2d_i16x16_n2c16k32_f3x3_fp32_cf` | G-B oracle gap (F13): no hipDNN reference judges this mode correctly | quick |
| G14 | conv, 2D NCHW fp32, S18 (pre 0, post 1) | `ConvolutionFwd/Default` | `s18_2d_i16x16_n2c16k32_f2x2_fp32_cf` | G-C: asymmetric padding convention | quick |
| G15 | conv, 2D NCHW fp16, S19 (output 1) | `ConvolutionFwd/Default` | `s19_2d_i16x16_n2c16k32_f16x16_fp16_cf` | G-D: tiny output, relative error dominates | nightly |
| G16 | conv, 2D NHWC fp32, A1: N 1, C 512 → K 64, 14², 3×3, pad 1 (accumulation length 4,608; 1.56 MiB) | `ConvolutionFwd/Default` | `a1_2d_i14x14_n1c512k64_f3x3_fp32_cl` | G-D: long accumulation | nightly |
| G17 | conv, 1D NLC fp16, M11 (Conformer depthwise; N 8; 1.02 MiB) | `ConvolutionFwd/Default` | `m11_1d_i128_n8c256k256g256_f31_fp16_cl` | G-F production anchor | nightly |

Seventeen cases, inside the 6–20 range of RFC 0022 §12.2. Each golden case is also a
live sweep case, so no level that runs without `dvc pull` will lose coverage.

- **Not golden:** NaN / Inf cases (F22) stay live with a matching-infinities
  validator, because golden tensors must be finite. All X shapes and the other
  M shapes stay live because they are above the size budget.
- **Generator:** `reference-data-scripts/generate_conv_fwd_golden.py` computes
  `torch.nn.functional.conv{1,2,3}d` on CPU from the bundle inputs, and applies
  scale, bias, residual and activation for the fused cases. For G13 it flips the
  filter in every spatial dimension before calling PyTorch, which only
  implements cross-correlation. **It does not exist yet (I7); no conv case has
  golden data.**
- **Precision strategy:** precision uplift (fp64, cast to the io type) for fp32
  cases. Matched precision (fp32 accumulation, rounded to the io type) for fp16
  and bf16 cases, which matches the engines' accumulation contract and keeps
  the calibrated tolerance tighter than the op default of 0.1. This deviates
  from the RFC 0022 §12.3 default (D2 after §4.3). Recorded per case
  in `meta.json`.
- **Confirmation:** `hipdnn_golden_data_tests --reference both` must pass for every
  golden case except G13. G13 cannot be confirmed by a hipDNN reference until F13's
  reference fix lands; until then it is confirmed by a second independent
  computation (the same convolution done directly with NumPy in fp64).
- **Storage:** `golden/<case-id>/tensors.dvc` beside the sweep, on the
  `golden-data` DVC remote, following
  `standard/BatchnormFwdInference/Inference/golden/`.

---

## 6. Pass criteria and reference prerequisites

| Item | Rule |
|---|---|
| Verification | `auto`: golden → GPU reference → CPU reference. See [Verification modes](../running-tests.md#verification-modes). |
| Tolerance | `hipdnn_test_sdk/utilities/TestTolerances.hpp` `conv::getToleranceFwd`. The maximum across graph nodes is used as atol and as rtol. Overrides go in the engine TOML. The fp32 default (1e-5, 1e-5) is below the numerical floor of a long reduction: the absolute error between two correct fp32 implementations grows with the accumulation length A = C/G × filter volume (observed on gfx942: 5e-5 at A ≤ 1,152; 1.9e-4 at A ≈ 4,600; 8e-3 at A ≈ 440,000). Every engine that runs L, M, H, A and X shapes in fp32 needs a tolerance class per A. Calibrate it from observed error; never loosen it to make a case pass. The MIOpen engine TOML carries such fp32 classes; they stay until the dynamic tolerance of #15481 merges, and are removed then. |
| Support claims | Enforced. See [support-claim-enforcement.md](../support-claim-enforcement.md). The sidecars hold `claim` and `regression` cases; the harvest that keeps probes out is outside the repository (§4.2). |

These features need reference or harness work before their cases can be judged
or run. The generator emits their cases now; they are unclaimed until the work
lands.

| Feature | Gap | Needed | RFC 0022 §15 |
|---|---|---|---|
| F13 `CONVOLUTION` | The GPU graph reference declines the mode (`GpuConvolutionFwdPlan.hpp`). The CPU graph reference (`cpu_graph_executor/detail/ConvolutionFwdPlan.hpp`) does not check `conv_mode` and runs cross-correlation, so under `auto` it judges a correct `CONVOLUTION` result wrong. `CpuFpReferenceConvolution::isApplicable` rejects the mode, but the graph executor does not call it. | Make the CPU plan builder decline the mode (or flip the filter), then add the flip to the GPU kernels; until then, golden data only. | I6 |
| F11 asymmetric padding | The GPU reference tests it in fp32, fp16 and bf16 (`TestGpuConvFwdRefAsymPad*`, against the CPU reference). The CPU reference tests it in fp32 only (2D and 3D). | Add fp16 and bf16 CPU reference tests. | I6 |
| F19 fp8, int8, double | The CPU and GPU conv-forward registries each hold five signatures, as (x, w, y, compute): (fp32, fp32, fp32, fp32), (fp16, fp16, fp16, fp32), (bf16, bf16, bf16, fp32), (fp16, fp16, fp32, fp32) and (bf16, bf16, fp32, fp32). None has an fp8, int8 or double type. | Add the signatures to both registries (ALMIOPEN-3227), or generate golden data. | I6, I7 |
| F20 mixed x and w types, fp16 compute | The same five signatures are the only mixed-precision ones: fp16 in and bf16 in with fp32 output. A mixed x and w type (`x_half_w_float`, `x_float_w_half`) and fp16 compute (`half_compute_half`) have no signature. | Add the signatures (ALMIOPEN-3227). | I6 |
| F21 subnormal, zero-excluded, sparse | The fill recipes are `free` and `fixed` only. | Value-domain control per tensor. | I4 |
| F22 NaN / Inf | Same as above, and the default comparison treats NaN as a mismatch. | I4, plus `[[validator_overrides]]` `allclose_matching_infinities` on those cases. | I4 |
| F17 alignment | The harness passes allocator pointers aligned to at least 256 bytes. | Exact-alignment and offset placement. | I3 |
| F24, F32–F34 | Bundles run one execution. | Repeat-and-compare on bundles. Until then, C++ tests. | I5 |
| `purpose` | The harness ignores the tag, and `--write-support-claims` records every accepted case, probes included. The harvest used for this PR filters probes out; it is outside the repository (ALMIOPEN-3231). | A harvester filter in the repository; support-matrix generation that excludes probes. | I2 |
| All | The conv sweeps came from C++ tests or by hand. | `generate_conv_fwd_bundles.py` (this plan). | I1 |

Engine-specific tolerances and skips go in the engine TOML and are keyed to the
generated case ids, and the claim sidecars name cases by id too; the generator
keeps a claim only while its id exists. The id scheme is
`<plan items>_<rank>d_i<spatial>_n<N>c<C>k<K>[g<G>]_f<filter>_<dtype>[_w<dtype>_y<dtype>]_<layout>[_<activation>]`,
where a mixed x, w and y layout replaces `<layout>` with `x<l>w<l>y<l>` and a
mixed type adds the `w` and `y` types. Ids are written in lowercase as the bundle
verifier requires (`s13a_f16chan_2d_…` for plan items S13a and F16chan), and an
id that collides inside one sweep gets a `_<hash6>` suffix. Input seeds come from
the id as the plan items spell it, so the case of an id does not change its data.
A change to the generator that renames ids updates the TOML globs and exact ids
in the same pull request.

---

## 7. Exhaustive mode

Convolution forward has an exhaustive mode. It is a separate tool (RFC 0022 §14);
this plan only declares what the tool reads. The declarations live in
`reference-data-scripts/conv_fwd_plan.py`, the same module the bundle generator
reads, so there is one copy of the legality rules.

| Declaration | Where | Content |
|---|---|---|
| Axes | `RANKS`, `LAYOUTS`, `CORE_DTYPES`, `EXTRA_TYPES`, `MIXED_TYPES`, `ACTS`, `TOPOLOGIES`, and the case fields of `DEFAULTS` | every feature of §2 |
| Legality | `legal()` | the frontend validation of §2 and §2.7 |
| Shape ranges | rank 1–4; N 1–256; C, K 1–2048; G dividing C and K; spatial 1–16000 (1D), 1–224 (2D), 1–64 (3D); filter 1–31; stride and dilation 1–4; padding 0 – filter; size budget: x + w + y ≤ 1 GiB in the exhaustive run |
| Value domains | `value_recipes()` | uniform [−1, 1], zero, constant, large (non-negative, a = √(½ · max / accumulation)), tiny; NaN, Inf, subnormal and sparse once I4 lands |
| Return path | — | minimized reproducers become `regression` cases written by the generator; oracle bugs become G-B golden candidates |

---

## 8. How the sweep was built

This section states the method of this plan. The general rules (the authoring
process, sweep rules, levels, purpose, trade-off register, oracles, golden tests
and review checklist) are in
[RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md). Other
plans use other methods.

- **Inventory.** §2 comes from the graph API sources listed at the top of this
  plan: each field and enum of the conv attributes and each frontend validation
  rule is a feature (RFC 0022 §6). Existing bundles, C++ tests and engine support
  do not decide what is listed.
- **Defaults.** The S and L defaults make every dimension that can exceed 1
  exceed 1 and keep K > G, so channel and group indexing are exercised (§3).
- **Construction.** `rules()` in `generate_conv_fwd_bundles.py` builds each level
  from loops over rank, layout and data type, and from positional rotation of the
  remaining values (§4, pair cover). It has no all-pairs construction, pair check
  or selection seed. The generator is deterministic: ids derive from the plan
  items and the shape, and seeds from the ids. Levels are built in order, and a
  graph that a lower level or an earlier rule already holds is not written again.
- **Sizing.** The ceilings of §3 bound each shape, and the scaling rule shrinks
  an L shape to fit. Compile and solver search per unique shape, not execution,
  set the cost of a run; the oracle's cost matters for M and X shapes (§4.4).
- **Trims.** The complete product of every feature with every other is about
  14,000 cases (§4.3), and the register records each product left out. T1–T7 and
  T12 follow from the structure of the op. T8, T9 and T11 follow from measured
  MIOpen runs (§4.5), the one place the plan uses an engine. A trim runs after
  ids are assigned, so it renames no case, and it never removes the last probe of
  a feature.
- **Oracles.** Each feature is checked against both references before its cases
  are written (§6). A reference can silently compute the wrong thing: the CPU
  graph reference runs cross-correlation for a `CONVOLUTION` graph, and the GPU
  reference declines it.
- **Golden.** §5 selects cases by the criteria of RFC 0022 §12.2; none exists yet.
- **Engine fitness.** The plan does not decide whether an engine is fit to
  integrate. An engine integration gate that would consume these graphs is planned
  ([RFC 0022 §19](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#19-open-questions-and-future-work),
  item 5); its reproducers return as `regression` cases (§7).
