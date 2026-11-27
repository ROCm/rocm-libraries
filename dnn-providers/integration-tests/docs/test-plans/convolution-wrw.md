# Convolution Backward-Weights Test Plan

This plan lists every feature of convolution backward weights (wrw, weight
gradient, `ConvolutionWrwAttributes`) that a complete test sweep must exercise,
and how the sweep is spread across levels. It does not depend on any engine: the
sweep describes the problem space, and each engine's support claims decide which
cases it runs. Forward convolution and data gradient have their own plans
([convolution-fwd.md](convolution-fwd.md), `convolution-bwd.md`).

The plan follows [RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md) (Per-Op Test Plans).
See [the test-plans index](README.md) for the level definitions.

Wrw shares its parameters with the forward convolution, but not its problem.
Every dw element is a sum over the whole batch and over every dy position, so
the reduction is as long as `N × dy volume` (millions in a training step) while
the output, `K × C/G × filter`, can be a single element or a million. The
engines pick their kernels on exactly these properties: how long the reduction
is against how large the output is (split-K and atomics), the filter shape
(Winograd, 1×1 and depthwise kernels), the stride against the filter (x elements
no window reads), padding that reaches past the filter (dw taps that read only
padding), and the channel counts. This plan's catalog is built from them (§2.1,
§3) and from the corpus of wrw graphs real models run (§3.3). It does not reuse
the data-gradient plan's shapes: a data-gradient case reduces over K and writes
C, so its small-batch layers are short reductions here.

The bundles are **generated** from this plan. Nobody edits them by hand:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_conv_wrw_bundles.py \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles --report
```

| Plan item | Where it lives |
|---|---|
| Case model, legality, behaviour classes, shape catalog, ceilings, data types, value domains, purpose rule, trimmed products (`TRIMS`) | `reference-data-scripts/conv_wrw_plan.py` |
| R set: corpus pick, selection rule, provenance | `reference-data-scripts/conv_wrw_corpus.json` |
| Level rules, graph template, sweep writer, purpose tags, trims | `reference-data-scripts/generate_conv_wrw_bundles.py` |
| Generator tests (bundles match the generator, recipes follow the operand) | `reference-data-scripts/tests/test_generate_conv_wrw_bundles.py` |
| Bundles | `integration-test-bundles/{quick,standard,comprehensive,full}/ConvolutionWrw/Default` |

Every sweep case records the plan items it implements (`metadata.plan_items`),
its `purpose` (§4.2) and its size class. A plan item names its cases with a
filter on `plan_items`, and a case names its items.

The features come from the graph API:

- `flatbuffers_sdk/schemas/convolution_wrw_attributes.fbs` and `convolution_common.fbs`
- `tensor_attributes.fbs`, `data_types.fbs`, `graph.fbs`
- the frontend rules in `hipdnn_frontend/node/ConvolutionWgradNode.hpp`

---

## 1. The operation

Wrw is the gradient of `y = conv(x, w)` with respect to w. For each dw element
it sums, over the whole batch and every dy position, the product of the dy value
and the x value its window reads at that tap:

```
dw[k, c', r...] = Σ_n Σ_{o...} dy[n, k, o...] · x[n, group(k)·C/G + c', o·stride − pre + r·dil]
                  (only the terms where the x index lies inside x; the rest read padding and add 0)
```

| Tensor | Dims | Rule |
|---|---|---|
| x | N, C, in… | input; at least 3 dims |
| dy | N, K, out… | input; same rank as x; `dy[0] == x[0]`; for every spatial dim `out == floor((in + pre + post − (dil·(f − 1) + 1)) / stride) + 1` |
| dw | K, C/G, filter… | output; same rank as dy; `dy[1] == dw[0]`; `x[1] % dw[1] == 0`, so G = C / dw[1]; `K % G == 0`. **dw dims are never inferred**: the frontend rejects a graph without them (`ATTRIBUTE_NOT_SET`). Only dw strides are inferred, from x's stride order. |

Parameters: `pre_padding`, `post_padding`, `stride` and `dilation`, one value per
spatial dimension, and `conv_mode` (`CROSS_CORRELATION` or `CONVOLUTION`). The
node `compute_data_type` and the graph's io, intermediate and compute data types
are set separately. The wrw graph's tensor uids are dw 0, dy 1 and x 2.

Four quantities follow from the formula and drive both the kernels and the plan:

- **Reduction length.** One dw element sums `A = N × Π(dy spatial)` products.
  Products that read padding are zero, so A is an upper bound for taps at the
  border. A reaches 4 million on one ResNet-class layer at a training batch and
  2 billion on the largest probe (X3).
- **Reduction against output.** The output holds `K × C/G × Πf` elements. A long
  reduction into a small output (A 4 million, one element) leaves no
  parallelism over the output, so engines split the reduction across
  workgroups and add the parts with atomics or a second pass; a short
  reduction into a large output (A 1, a million elements) is an outer product.
  The error of the first grows with A and its sum can overflow a 16-bit
  partial; the second exposes the output indexing.
- **Skipped x elements.** With stride s > filter f, or with a remainder
  `(in + pre + post − dilated f) % s ≠ 0`, some x elements lie in no window and
  never contribute. With a filter larger than x, or padding wider than the
  dilated filter, a tap can read only padding for every dy position: that dw
  element is exactly 0 and the engine must write it.
- **Padding reads.** At the border a tap sums fewer products than the
  interior; a case with no padding (`pad 0`) has none that read padding, so it
  is the one case where every product is real.

---

## 2. Feature inventory

Each feature lists the values a complete sweep must hit. The **Oracle** column
says whether the verification references can judge the feature today (§6); the
**Earliest level** column is the lowest level that holds a case of it (`Q`
quick, `PR`, `N` nightly, `F` full; `—` means the sweep cannot express it yet).

**Oracle.** Two references judge wrw, in the harness's `auto` order (golden
data, then the GPU reference, then the CPU reference). The GPU reference
(`GpuConvolutionWrwPlan.hpp`) and the CPU reference
(`CpuFpReferenceConvolution::wgrad`) both register x, dy, dw = fp32/fp32/fp32,
fp16/fp16/fp16 and bf16/bf16/bf16 with fp32 compute, address every tensor
through its own strides, take pre and post padding separately, and accept only
`CROSS_CORRELATION`. A case the engine accepts and no oracle can judge
**fails** ("no oracle can verify this bundle"), so oracle gaps show up as
failures, not skips. In the tables, "GPU, CPU ref" means either reference can
judge the feature, and the GPU reference is the one that runs.

**Legality** is declared once, in `conv_wrw_plan.legal()`, and copies
`ConvolutionWgradNode::pre_validate_node`: equal ranks, `C % G == 0`,
`K % G == 0`, one parameter per spatial dimension, stride and dilation ≥ 1,
padding ≥ 0, and a dilated filter no larger than the padded x. dy's dims are
computed from the output formula, so its check holds by construction. The
generator never writes an illegal graph.

### 2.1 Geometry and data flow

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F1 | Spatial rank | 1D, 2D, 3D; 4D (6-dim tensors) as a probe | GPU, CPU ref | Q (1D–3D), PR (4D) |
| F2 | Batch N (the reduction axis) | 1, 2, 3, 5, 8, 32, 64, 128, 256 | GPU, CPU ref | Q |
| F3 | x channels C/G (dw columns) | 1, 3, 4, odd below 16 (5, 7, 13), 8–32, 40–256, > 256, not a multiple of 8 | GPU, CPU ref | Q |
| F4 | dy channels K/G (dw rows) | 1, 2, 3, 4, 7, odd (17, 21), 8–32, 40–256, > 256 (320, 2048) | GPU, CPU ref | Q |
| F5 | Channel ratio | K/G = C/G; K/G = 4·C/G and C/G = 4·K/G (the two 1×1 backward passes); small C/G with large K/G (4 / 320: latent `conv_in`); large C/G with tiny K/G (128 / 3, 4, 21: output heads) | GPU, CPU ref | Q |
| F6 | Groups | dense; grouped with C/G = K/G = 4, 8, 16, 24, 32 (ResNeXt); grouped with K/G ≠ C/G (8 / 16, 4 / 320, 8 / 1); group counts 2, 3, 4, 5, 8, 16, 32; depthwise multiplier 1, 2, 4 | GPU, CPU ref | Q |
| F7 | Filter | 1, 2, 3, 4, 5, 7, 10, 11, 16, 31; non-square 1×3, 3×1, 3×5, 1×7, 7×1, 21×11, 1×3×3, 3×1×1; filter equal to x; filter larger than x | GPU, CPU ref | Q |
| F8 | Stride against the filter | s = 1; f = s (windows tile x), f > s (windows overlap), f < s (x elements between windows are read by no tap); gcd(s, d) > 1; coprime s, d; per-dimension stride | GPU, CPU ref | Q |
| F9 | Dilation | 1, 2, 3, 5, 12; per dimension | GPU, CPU ref | Q |
| F10 | Padding reads | none (every product real); "same" (border taps sum fewer products); wider than the dilated filter (pad 4 and 8 on a 3-wide filter: border dy positions read only padding) | GPU, CPU ref | Q |
| F11 | Asymmetric padding | pre 0 / post 1 and pre 1 / post 0 with a stride-2 even filter | GPU, CPU ref take pre and post separately; no reference test covers asymmetric wrw (I6) | PR (probe) |
| F12 | Output extent | dy spatial 1 (filter equal to x: A = N); dy larger than x (even filter, pad 1); x width at most 8 and 4; x 16×48 with a 5×5 filter; odd wide x (32×109); 1×1 x; extreme aspect 1×256. **Inferred dw dims are illegal** (frontend rejects them): a frontend unit test, not a sweep case. | GPU, CPU ref | PR |
| F13 | `conv_mode` | `CROSS_CORRELATION`, `CONVOLUTION` | **none**: both references reject `CONVOLUTION` for wrw (I6) | PR (probe) |
| F37 | dw taps no x reaches | a filter larger than x; padding wider than the dilated filter on an image smaller than the filter. The engine must write 0 there; the harness prefills outputs with a NaN sentinel, so an unwritten element fails. | GPU, CPU ref | PR |
| F38 | Reduction length against output size | A from 1 to 2³¹ (§3.5): the dw output from one element (A 4.2 million) to 1.05 million (A 1); every class of §3.1 under a long reduction (§3.2) | GPU, CPU ref | Q (A 4,096), N (A to 4.2 million), F (A to 16.8 million; 2³¹ as a probe) |

### 2.2 Memory layout and tensor kind

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F14 | Layout | channels-first and channels-last for each rank (NCL, NLC, NCHW, NHWC, NCDHW, NDHWC), for x, dy and dw together | GPU, CPU ref | Q |
| F15 | Mixed layouts | x, dy and dw in different orders: all six non-uniform combinations | GPU, CPU ref | PR |
| F16 | Non-packed strides | padded rows (`fast`), padded channel stride (`chan`), padded plane (`plane`) on x, dy and dw | GPU, CPU ref | PR |
| F17 | Alignment | a base pointer at exactly the declared alignment with an offset (RFC 0022's rule) | **not run**: I3 | — |
| F18 | 64-bit indexing | x and dy with more than 2³¹ elements, and A above 2³¹ into one dw element (X3) | GPU, CPU ref | F |
| F35 | Override shape | `is_override_shape_enabled` on an explicitly shaped graph | GPU, CPU ref | PR |

Every tensor is a device tensor: a wrw input that is a constant or a runtime
scalar is not a valid graph.

### 2.3 Data types

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F19 | IO data type | fp32, fp16, bf16; fp8 (E4M3, E5M2 and the FNUZ forms) with fp32 dw; int8 with int32 dw; double | **none** beyond the core three (I6) | Q (core), PR (fp8 E4M3 and double as probes) |
| F20 | Mixed precision | fp16 or bf16 x and dy with fp32 dw; x fp16 with dy fp32; x fp32 with dy fp16; fp16 compute | **none** (only uniform core triples are registered) | PR |

### 2.4 Values and numerics

Fill recipes are `free` (uniform in [lo, hi]) and `fixed`, per tensor in
`metadata.inputs`, each with its own seed. **A recipe belongs to an operand and
is keyed by that operand's uid** (dy 1, x 2): `conv_wrw_plan.value_recipes`
states them by role and the generator maps the roles to the wrw graph's uids.
The data-gradient graph numbers its operands differently (uid 2 is w there), so
a recipe table is never copied between the two. Every generated case sets its
fills: dy and x are uniform in [−1, 1] by default, so sums cancel.

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| F21 | Input distribution | uniform [−1, 1]; **zero** (dy 0, x uniform: dw must be exactly 0); **const** (dy = x = 1: dw counts the products that read x, so a wrong tap or a dropped part of a split reduction is an integer error; only where A stays inside the data type); **positive** (dy and x uniform in [0, a], a = min(1, √(0.4 · max / A)): the sum cannot cancel and grows with A, which exposes accumulation error and overflow of partial sums; only up to A 2²⁰, see §3.5); **large** (uniform in [0, a], a = √(½ · max / A): outputs sit near the data type's maximum); **tiny** (fp32 and bf16 ±1e-15, fp16 ±0.3: small, and far enough above the type's smallest normal number that almost no product falls below it; a kernel that flushes sub-normal products to zero differs from the reference by a few percent per element, which no tolerance can grade) | GPU, CPU ref | PR |
| F22 | Special values | NaN and ±Inf in dy or x | needs I4 and a matching-infinities validator | — |
| F23 | Accumulation length | A from 1 (outer product) to 16.8 million in a normal case (§3.5); with positive fills at every A of the ladder | GPU, CPU ref | N |
| F24 | Determinism | bitwise-identical repeats when a deterministic plan is requested; wrw kernels that split the reduction with atomics are not deterministic otherwise | C++ test (I5) | — |

### 2.5 Fusions

No pointwise fusion is in this plan. The MIOpen plugin builds `ConvolutionWrw`
as a single node (`MiopenConvPlanBuilder`, `ConvWrwPlan`), no other engine
registers a wrw plan, and the references only judge the single node.
A fused wrw sweep would be all `unclaimed` probes that no engine executes. The
forward and data-gradient plans carry fused topologies for engines that fuse
them; when an engine fuses wrw (for example a gradient scale or an accumulate
`dw + dw_prev` after the wrw node, or an activation backward before it), add a
topology to `conv_wrw_plan.py` and a bundle directory `ConvolutionWrwPointwise`
as those plans do.

### 2.6 Execution

These cannot be expressed as a bundle case and stay C++ tests until the harness
can repeat and compare (I5): workspace limits (F32), plan reuse with new data and
pointers (F33), and serialisation round trips (F34).

### 2.7 Invalid problems

Frontend unit tests, not sweep cases. Each must be rejected:

- dw dims not set (wrw never infers them);
- `dy[1] ≠ dw[0]`, `dy[0] ≠ x[0]`, `x[1] % dw[1] ≠ 0`, `dw[0] % G ≠ 0`;
- stride or dilation < 1, negative padding;
- rank mismatch between x, dy and dw; parameter count ≠ spatial rank;
- dy spatial dims inconsistent with the output formula;
- a dilated filter larger than the padded x.

---

## 3. Shape catalog

The sweep draws its shapes from nine sets: **S** (small, one feature changed
from the defaults), **L** (the same classes at medium size), **LA** (the
kernel-selecting classes under a long reduction), **M** (named model layers),
**R** (real wrw graphs from the corpus), **H** (hazard breakpoints), **A**
(the reduction ladder), **B** (reproducers of escaped defects) and **X** (stress). Each level raises the size ceiling:

| Level | Per-case ceiling (fp32 x + dy + dw) | Shape sets |
|---|---|---|
| quick | ≤ 20 MFLOP, ≤ 1 MB | S (3D at 16×8×8 or smaller), A1 |
| PR | ≤ 200 MFLOP, ≤ 16 MB | S, H, R, B, A2, A10, A11, plus M1, M4, M11, which exceed the ceiling by design |
| nightly | ≤ 8 GFLOP, ≤ 64 MB | S, L, LA, M, R, H, A |
| full | none | S, L, LA, R, A, X |

FLOP counts 2 × multiply-accumulates. When a 3D S shape does not fit a level at
16³, the generator tries 16×8×8, 16×8×4 and 8×8×4.

### 3.1 Small (S)

Defaults: N 2, C 16, K 32, G 1, spatial 16 per dimension, filter 3, stride 1,
dilation 1, pad 1. N is 2 so that no default dimension is 1; C and K differ so
that dw's rows and columns are never the same size by accident. Each class
changes what its row says; ranks are 1D, 2D, 3D unless the row names fewer. The
classes are grouped by the wrw property they exercise.

**Basics, padding reads and filter shapes.**

| Id | Change | Ranks | Why for wrw |
|---|---|---|---|
| S1 | filter 1, pad 0 | 1–3 | dw is a GEMM over N × dy |
| S2 | pad 0 | 1–3 | no padding reads: every product is real |
| S3 | — | 1–3 | the default "same" case; feature probes sit on it |
| S4 | filter 5, pad 2 | 1–3 | |
| S5 | filter 7, pad 3 | 1, 2 | |
| S6 | dilation 2, pad 2 | 1–3 | dilated taps at stride 1 |
| S7 | x 16×8 (2D), 16×8×4 (3D) | 2, 3 | rectangular x |
| S8 | N 8, filter 1, pad 0 | 1–3 | batch |
| S9 | pad 4 | 1–3 | pad > d(f − 1): border dy positions read only padding |
| S10 | filter 2, stride 2, pre 0, post 1 | 1–3 | F11 asymmetric (probe) |
| S11 | filter 2, stride 2, pre 1, post 0 | 1–3 | F11 pre-only (probe) |
| S12 | filter 2, pad 1 | 1–3 | even filter: dy is larger than x |
| S13 | filter equal to x, pad 0 | 1–3 | dy spatial 1: A = N, one dy element per batch entry |
| S14 | filter = x + 2, pad 1 | 1–3 | filter larger than x: dw taps no x reaches are 0 (F37) |
| S15 | stride (1, 2, 1), dilation (2, 1, 2), pad (1, 0, 1) | 2, 3 | per-dimension parameters |
| S16a, S16b, S16c | filter 1×3 / 1×3×3; 3×1 / 3×1×1; 3×5 | 2, 3; 2, 3; 2 | non-square filters |
| S84 | dilation 12, pad 12 | 1, 2 | the largest dilation: the dilated filter is 25 wide on a 16-wide x |
| S85 | pad 8 | 1–3 | most dy positions read only padding |
| S86 | filter 1×7, pad (0, 3) | 2 | a Winograd multipass wrw shape |
| S87 | filter 7×1, pad (3, 0) | 2 | the same, transposed |

**Stride against the filter.**

| Id | Change | Ranks | Why for wrw |
|---|---|---|---|
| S17 | stride 2 | 1–3 | f > s (overlapping windows) and a remainder on an even x |
| S18 | stride 2, x 15 | 1–3 | the same without a remainder |
| S19 | filter 2, stride 2, pad 0 | 1–3 | f = s: windows tile x, every x element is read once |
| S20 | filter 4, stride 2, pad 1 | 1, 2 | f = 2s |
| S21 | filter 4, stride 4, pad 0 | 2 | patchify 4 |
| S22 | C 3, filter 16, stride 16, pad 0, x 32² | 2 | ViT patch embedding: wrw reduces N × 4 over a 16×16 filter |
| S23 | filter 1, stride 2, pad 0 | 1–3 | f < s: 3 of 4 x elements (2D) are read by no tap |
| S24 | stride 4 | 1–3 | f < s with filter 3 |
| S25 | stride 3, pad 0 | 1–3 | f = s with a remainder: x's last row is read by no window |
| S26 | stride 2, dilation 2, pad 2 | 1–3 | gcd(s, d) = 2: every other x element is skipped |
| S27 | stride 2, dilation 3, pad 3 | 1, 2 | coprime stride and dilation |
| S28 | C 3, filter 7, stride 2, pad 3 | 1–3 | strided stem, 3 input channels |
| S29 | stride 2, pad 0 | 1–3 | strided without padding |
| S30 | filter 7, stride 2, pad 3 | 2 | large filter, stride 2 |
| S31 | filter 1×3×3, stride (1, 2, 2), pad (0, 1, 1) | 3 | (2+1)D video: strided in two of three dimensions |
| S88 | C 8 / 1, K 16 / 32, filter 21×11, stride (2, 1), pad (10, 5), x 48×32 / 161×700 | 2 | DeepSpeech2: tall non-square filter, per-dimension stride |
| S89 | filter 1, pad 0, x 4×4×4 | 3 | 1×1×1 on a small volume: A 128 |
| S90 | filter 1×3×3, stride (1, 2, 2), pad (0, 1, 1), x 4×8×8 | 3 | (2+1)D video on a small volume: A 128 |

**Channel classes** (K is dw's rows, C its columns). 2D unless noted.

| Id | C, K (small / medium) | Ranks | Why for wrw |
|---|---|---|---|
| S32 | C 1 | 1–3 | one x channel |
| S33 | C 3, K 64 | 2, 3 | RGB input |
| S34 | C 4, K 320 | 2 | small columns, many rows (latent `conv_in`; MFMA kernels with 4 channels per group) |
| S35 | K 1 | 1–3 | one dy channel: dw is one row |
| S36 | C 128, K 3 | 2, 3 | wide columns, 3 rows (image `conv_out`) |
| S37 | C 64 / 256, K 2, filter 1 | 2 | binary head |
| S38 | C 13 / 61, K 17 / 67, x 15 / 55 | 1–3 | odd channels and odd x |
| S39 | C = K = 24 | 2 | a multiple of 8 that is not a power of two |
| S40 | C = K = 4 | 2 | both below the vector width |
| S41 | C 16 / 64, K 64 / 256, filter 1 | 2 | K = 4C: backward of a 1×1 expansion |
| S42 | C 64 / 256, K 16 / 64, filter 1 | 2 | C = 4K: backward of a 1×1 reduction |
| S77 | C 128 / 256, K 3, filter 1 | 2, 3 | RGB output head |
| S78 | C 128 / 256, K 21, filter 1 | 2 | class head: rows that are not a multiple of 8 |
| S79 | C 128, K 4, filter 3 | 2 | latent `conv_out` |
| S80 | C = K = 4, filter 1 | 2 | latent quant conv |
| S81 | K 1, pad 2 (f − 1) | 1–3 | one dy channel, pad f − 1 |
| S82 | K 1, stride 2 | 1–3 | one dy channel with f > s and a remainder |
| S83 | G 4, C/G 8, K/G 1, stride 2, dilation 2, pad 2 | 1–3 | one dy channel per group with gcd(s, d) > 1 |

**Grouped convolution** (2D unless noted; per-group sizes are the class, the
medium form raises the group count).

| Id | G, C/G, K/G (small / medium) | Ranks | Why for wrw |
|---|---|---|---|
| S43 | G 4 / 32, C/G = K/G = 4 | 2 | ResNeXt 32×4d |
| S44 | G 8 / 32, C/G = K/G = 8 | 2 | ResNeXt 32×8d |
| S45 | G 4 / 8, C/G = K/G = 16 | 2 | |
| S46 | G 2 / 4, C/G = K/G = 24 | 2 | |
| S47 | G 2 / 4, C/G = K/G = 32 | 2 | |
| S48 | G 2, C/G 4, K/G 320 | 2 | grouped small columns, many rows (MFMA kernels) |
| S49 | G 2 / 4, C/G 8 / 16, K/G 16 / 32 | 1–3 | K/G = 2·C/G |
| S50 | G 4 / 16, C/G = K/G = 8, stride 2 | 1–3 | grouped and strided |
| S51 | G 2 / 4, C/G = K/G = 8 / 16, filter 5, dilation 2, pad 4, x 15 / 55 | 2 | grouped, dilated |
| S52 | G 4 / 16, C/G 8, K/G 1 | 1–3 | one dy channel per group |

**Depthwise** (G = C; 2D unless noted).

| Id | C (small / medium), change | Ranks | Why for wrw |
|---|---|---|---|
| S53 | 16 / 64 | 1–3 | multiplier 1 |
| S54 | 16 / 64, K = 2C | 1–3 | multiplier 2 |
| S55 | 16 / 32, K = 4C | 1, 2 | multiplier 4 |
| S56 | 16 / 64, stride 2 | 1–3 | strided depthwise |
| S57 | 16 / 96, filter 5, stride 2, pad 2 | 2 | MobileNetV3, EfficientNet |
| S58 | 16 / 64, filter 7, pad 3 | 2 | ConvNeXt |
| S59 | 64, filter 7, pad 0, x 16² / 28² | 2 | windowed depthwise: no padding reads |
| S60 | 5, filter 7, pad 3, x 9² / 56² | 2 | odd C below 16 |
| S61 | 64, x 70×40 / 112×56 | 2 | tall x |
| S62 | 32 / 64, dilation 2, pad 2, x 15² / 55² | 2 | dilated depthwise |
| S63 | 32 / 64, stride (1, 2) | 2 | unequal strides |
| S64 | 64 / 256, filter 31, pad 15, x 128 / 512 | 1 | Conformer |

**Image size and tiling** (2D).

| Id | Change (small / medium) | Why for wrw |
|---|---|---|
| S65 | C = K = 32 / 128, x 8² | dy width 8: kernels that keep the dy window in registers |
| S66 | C = K = 32 / 256, x 4² | dy width 4 |
| S67 | filter 5, pad 2, x 16×48 / 48² | 5×5 with dy width 48 |
| S68 | N 1 / 2, C = K = 32 / 64, x 32×109 / 64×109 | odd wide x: row tiles with a remainder |
| S69 | x 17×19 / 55×57 | odd non-square x |
| S70 | N 8 / 32, filter 1, x 1×1 | a pure GEMM: dw = dyᵀ x over the batch, A = N |
| S71 | filter 1×3, x 1×256 / 1×4096 | extreme aspect (1D carried as 2D) |
| S72 | stride 2, x 17×19 / 55×57 | strided on an odd non-square x |

**Rank-specific.**

| Id | Change | Ranks | Why |
|---|---|---|---|
| S73 | filter 7, dilation 3, pad 9 | 1 | vocoder (HiFi-GAN) |
| S74 | filter 11, dilation 5, pad 25 | 1 | vocoder |
| S75 | C 1, K 32 / 512, filter 10, stride 5, x 160 / 16000 | 1 | wav2vec feature extractor |
| S76 | 6-dim tensors, N 1, x 6×6×4×4 | 4 | F1 4D (probe) |

### 3.2 Medium (L) and long reduction (LA)

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

LA*n* puts the long reduction under the classes that select a kernel:

| Rank | N | C | K | Spatial | A at stride 1 |
|---|---|---|---|---|---|
| 1D | 8 | 16 | 16 | 16,384 | 131,072 |
| 2D | 16 | 16 | 16 | 224² | 802,816 |
| 3D | 4 | 8 | 8 | 32³ | 131,072 |

An LA class keeps the filter, stride, dilation, padding, channel and group values
of S*n* and takes this base's batch and image. A class that fixes the batch or
the extent itself (S7, S13, S14, S18, S65–S72, S74, S75, S76) has no LA form.
The generator halves N until the case fits the nightly ceiling and keeps the case
only when A is still at least 65,537, because a long reduction is the point:
`conv_wrw_plan.LA_MIN`. A stride-2 class reaches A 100,352 after one halving;
a class with many channels (S34, S48) halves to a shorter reduction and drops.
66 classes have an LA form in some rank; 97 cases at nightly.

### 3.3 Model (M) and corpus (R)

**M** is twelve wrw layers of models trained today, each at the batch of a
training step (the reduction is over it):

| Id | Source | N, C, spatial; K | Filter, stride, pad, groups |
|---|---|---|---|
| M1 | ResNet-50 3×3 | 8, 64, 56²; 64 | 3×3, p1 |
| M2 | ResNet-50 1×1 expansion | 8, 64, 56²; 256 | 1×1 |
| M3 | ResNet-50 downsample | 8, 512, 28²; 1024 | 1×1, s2 |
| M4 | ResNet-50 stem | 8, 3, 224²; 64 | 7×7, s2, p3 (A 100,352, dw 9,408) |
| M5 | ResNeXt grouped | 8, 128, 56²; 128 | 3×3, p1, G 32 |
| M6 | MobileNetV2 depthwise | 8, 96, 56²; 96 | 3×3, p1, G 96 |
| M7 | ConvNeXt depthwise | 8, 128, 56²; 128 | 7×7, p3, G 128 |
| M8 | Stable Diffusion UNet `conv_in` | 2, 4, 64²; 320 | 3×3, p1 |
| M9 | Stable Diffusion VAE `conv_out` | 1, 128, 256²; 3 | 3×3, p1 (A 65,536) |
| M10 | UNet3D | 2, 32, 32³; 32 | 3×3×3, p1 |
| M11 | Conformer depthwise | 8, 256, 128; 256 | 31, p15, G 256 |
| M12 | ResNet-50 stage 3 | 8, 256, 14²; 256 | 3×3, p1 |

**R** comes from the weight-gradient graphs of the convolution corpus of
[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking)
(graph store `s3://therock-dvc/dnn-benchmarking-graphs/`, tags
`workload=corpus`, `operation=convolution`, `direction=wgrad`): 2,444 unique wrw
geometries recorded from live training runs of 31 models in 8 families
(classification 906, detection 478, diffusion 393, audio 180, video 177,
segmentation 159, 3-D medical 114, super-resolution 48). What they use:

| Property | Corpus share |
|---|---|
| Rank | 2D 89%; 3D 11%; none 1D (audio models carry a 1D conv as 2D with H = 1) |
| Filter | 3×3 44%; 1×1 41%; non-square 10%; 5×5 and 5-wide 3%; larger 1% |
| Stride against the filter | s 1: 81%; f > s: 13%; f = s: 3%; f < s: 2% |
| Padding | "same" 54%; none 46%; never wider than the dilated filter, never asymmetric |
| Reduction A | ≤ 1,024: 15%; ≤ 65,536: 55%; above: 30% (4.8% above 2²⁰, largest 8.2 million) |
| Taps no x reaches (F37) / remainder | 4 graphs / 15% |
| tiny dy channels K/G ≤ 4 / tiny x channels C/G ≤ 4 | 9% / 9% |
| dy width ≤ 8 | 25% |
| Dense / depthwise / grouped | 90% / 6% / 5% |

- **Signature.** A wrw behaviour signature is `conv_wrw_plan.signature()`:
  rank, group class, C/G and K/G classes (1, 2, 3, 4 or above), filter class
  (1×1, 3×3, non-square, 5-wide, larger), stride class, padding class and A class
  (≤ 1k, ≤ 64k, above). The corpus has 114 signatures.
- **Selection.** Per signature R takes the median-FLOP graph inside each level
  ceiling (PR, nightly, and full up to 120 GFLOP and 1 GiB); every model gets at
  least one graph in PR or nightly; a graph picked for two levels sits in the
  lower.
- **Counts.** PR 41, nightly 72, full 69: 182 cases covering all 114 signatures
  and all 31 models.
- **Cases keep the model's data type and layout** (122 fp16, 27 fp32, 33 bf16;
  149 channels-first).
- **Purpose and provenance.** R cases are `claim` cases; `plan_items` holds the
  R id, `metadata.notes` the corpus shape id, models and family. The pick and
  its pin (the store key of each graph, the sha256 of its geometry, so a store
  update cannot change a pick) are data in `conv_wrw_corpus.json`; a corpus
  update is a deliberate regeneration.
- **Synthetic cover.** The S, L, LA, M, H, A and X shapes reach 90 of the 114
  signatures and 95% of the corpus graphs. The other 24 pair a 3-channel or
  4-channel side, a 5-wide or larger filter, or a stride with a long reduction on
  a 3D volume; only R reaches them (116 corpus graphs).
- **Limits of the corpus.** It was captured at small batch on one GPU, holds
  no fp8 or int8 and no graph whose padding is wider than the filter. S, H, A and
  X keep the large batches and the awkward cases.

### 3.4 Hazard (H)

One change from the S defaults. K hazards hit dw's row tiles, C hazards its
columns, and N and the extent the reduction `N × dy volume`.

| Id | Change |
|---|---|
| H1–H6 | C / K: 15 / 32, 17 / 33, 31 / 31, 33 / 65, 63 / 64, 65 / 63 |
| H7 | C 7 (columns below the vector width) |
| H8 | K 7 (rows below the vector width) |
| H9–H12 | N 3, 1, 32, 64 |
| H13–H15 | spatial 31, 33, 17 |
| H16, H17 | G 3 (C 12, K 12); G 5 (C 15, K 20) |
| H18 | spatial 7: A = 98, not a multiple of 8 |
| H19 | N 5, spatial 9: A = 405, odd |

### 3.5 Reduction ladder (A), regressions (B) and stress (X)

The ladder keeps the channels small so that the cost is the reduction and the
output is small against it (the split-K regime), and has the two opposite
corners.

| Id | Shape | A | dw elements |
|---|---|---|---|
| A1 | 2D, N 4, C = K 16, 32², 3×3 | 4,096 | 2,304 |
| A2 | 2D, N 8, C = K 16, 64², 3×3 | 32,768 | 2,304 |
| A3 | 2D, N 16, C = K 16, 112², 3×3 | 200,704 | 2,304 |
| A4 | 2D, N 32, C = K 8, 128², 3×3 | 524,288 | 576 |
| A5 | 2D, N 1, C = K 1, 2048², 1×1 | 4,194,304 | 1 |
| A6 | 1D, N 4, C = K 16, 65,536, 3 | 262,144 | 768 |
| A7 | 3D, N 2, C = K 8, 48³, 3×3×3 | 221,184 | 1,728 |
| A8 | 2D, N 1, C = K 1, 4096², 3×3 | 16,777,216 | 9 |
| A9 | 2D, N 64, C = K 8, 224², 3×3 | 3,211,264 | 576 |
| A10 | 2D, N 1, C = K 256, 4², 3×3 | 16 | 589,824 |
| A11 | 2D, N 1, C = K 1024, 1², 1×1 | 1 | 1,048,576 (an outer product) |

The positive fills (F21) run on the ladder up to A 2²⁰ (A1–A4, A6, A7; A5 and above
are left out): the GPU reference sums a dw element in one thread, sequentially in
fp32, and for 4.2 million products in [0, 1] that sum stagnates, 1.9 × 10⁻³ below the
exact sum (§6). The error is the oracle's, not the engine's.

**B** holds the shapes of cases that failed in the past, as `regression` cases:

| Id | Shape | Why |
|---|---|---|
| B1 | 3D, N 8, C 16, K 1, 16³, 1×1×1 | A 32,768 into 16 dw elements: the split-K cases of ROCm/rocm-libraries#8029 and #8030, which differed from run to run in one data type |
| B2 | 2D, N 8, C 16, K 1, 16², 1×1 | the same, in 2D |

| Id | Shape | Feature |
|---|---|---|
| X1 | 2D, N 256, C = K 64, 56², 3×3 | F2: A 802,816, 59 GFLOP |
| X2 | 3D, N 2, C = K 64, 64³, 3×3×3 | size: 116 GFLOP |
| X3 | 2D, N 1, C = K 1, 46341², 1×1 (more than 2³¹ elements; A above 2³¹ into one dw element); fp16 and bf16 | F18 |
| X4 | 2D, N 1, C 64, K 2048, 14², 3×3 | A 196, dw 1.2 million: a short reduction into a large output |
| X5 | 2D, N 128, C = K 32, 112², 1×1 | A 1.6 million, dw 1,024 |

---

## 4. Building the sweep

| Rule | Meaning |
|---|---|
| Core cover | PR and every level above hold every rank × layout × data type cell (18). Quick is a smoke set of 8 cases that catches gross breakage in a minute; it does not cover the cells. |
| One feature at a time | Each S and L class changes one wrw property from the defaults, so a failure points at it. |
| Size grows with level | PR proves a class works small; nightly at medium size and under a long reduction; full adds production sizes and stress. |
| All-pairs | By the end of nightly every S class meets every layout × data type pair of its rank, and every feature probe meets every core value. |
| Reduction against class | Every kernel-selecting class has a long-reduction form (LA) in nightly and full, because a long reduction changes which kernel runs. |
| Probes | Every unsupported feature, awkward combination and hazard has a probe in PR and in every level above (directly or through the cascade). Quick holds none of them. |
| Concrete graphs | Each level runs checked-in graphs only; the generator is deterministic. |
| Unsupported stays in | A feature no engine supports stays in the sweep, unclaimed; it reports `unclaimed_support` once an engine adds it. |

### 4.1 Level contents

| Feature group | quick | PR | nightly | full |
|---|---|---|---|---|
| Shape size | ≤ 20 MFLOP | ≤ 200 MFLOP (+3 M) | ≤ 8 GFLOP | unbounded |
| Plain core | S3 on 2D (fp32 channels-first, fp16 channels-last), S17 on 2D (bf16), S3 on 3D (fp32 channels-last), S1 (bf16 channels-last), S53 (fp16), S44 (fp32) on 2D | the quick set on every rank (S1, S3, S6, S8, S17, S19, S23, S44, S53, pair rotated) and every S class × rank once, pair rotated | every S class × every pair of its rank; L × 2 pairs; LA × 1 pair | L × every pair; LA × 2 more pairs |
| Rejection probes | — | `CONVOLUTION` per rank; asymmetric pad (S10); fp8 and double; 6 extra types; 5 mixed pairs | × core | asymmetric padding, mixed layouts, strides, mixed types and values × every S class of S1–S31; `CONVOLUTION`, extra types and override shape stay at nightly's × core (T11) |
| F15, F16 | — | 1 case each | rank × dtype, 2 mixes and 3 stride kinds | paired with every S class of S1–S31 |
| F19, F20 | — | each extra type and mixed pair once | × rank × layout | F20 paired with every S class of S1–S31; F19 — (T11) |
| F21 values | uniform | + zero, const, positive, large, tiny | × rank × dtype; positive × the A ladder | × core; × every S class of S1–S31 |
| F35 | — | 1 case | × rank × dtype | — (T11) |
| H hazards | — | each H once (2D) | each H × every pair (2D); 1D, 3D × 2 pairs | — |
| Reduction ladder (A) | A1 | A2, A10, A11 | A1–A7, A10, A11 × every pair; positive fills on A1–A4, A6, A7 | A8, A9 × every pair |
| Regression (B) | — | B1, B2 × every pair | — | — |
| Model (M) | — | M1, M4, M11 | M1–M12 × every pair of its rank | — |
| Corpus (R) | — | 41 graphs | 72 graphs | 69 graphs |
| Stress | — | — | — | X1, X2, X4, X5 × every pair; X3 fp16 and bf16 |

Cases generated (own level / cumulative):

| Level | Cases | Claim | Probe | Regression | Cumulative |
|---|---|---|---|---|---|
| quick | **8** | 8 | 0 | 0 | 8 |
| PR | **294** | 233 | 49 | 12 | 302 |
| nightly | **1,825** | 1,419 | 406 | 0 | 2,127 |
| full | **1,451** | 941 | 510 | 0 | 3,578 |

By size class: quick S 7, A 1. PR S 216, H 19, M 3, R 41, A 3, B 12. Nightly S 1,005,
L 340, LA 97, M 66, H 171, A 74, R 72. Full S 480, L 670, LA 194, R 69, A 12,
X 26. 788 distinct wrw geometries and 342 behaviour signatures. The trim of §4.3
removes 240 of the 1,691 cases the level rules generate for full.
`generate_conv_wrw_bundles.py --report` prints the counts again.

### 4.2 Probes and regressions

| Purpose | Expected outcome | Claims |
|---|---|---|
| `claim` | A claiming engine accepts and is correct. | Recorded. |
| `probe` | A typed rejection, or a correct accept. | Not recorded; support matrices exclude probes. |
| `regression` | Same as the original bug case. | Same as the original case. |

A case is a **regression** when it is a B shape (§3.5); it is a **probe** when it has at least one of:

| Probe class | Items |
|---|---|
| Unsupported feature | F13 `CONVOLUTION`; F11 asymmetric padding; F19 fp8, int8, double; F1 4D spatial |
| Awkward combination | F15 mixed layouts; F16 non-packed strides; F20 mixed precision and compute type; F35 override shape |
| Value domain | F21 zero, const, positive, large, tiny |
| Hazard | every H shape; X3 (F18) |

Everything else is a `claim`: ranks 1–3, both layouts, fp32, fp16 and bf16, and
the S, L, LA, M, R, A and X1, X2, X4, X5 shapes. There are no `regression` cases
yet; each escaped wrw bug's reproducer will land in the lowest level that can
afford it.

### 4.3 Products deliberately not run

The complete product of every feature with every other is far larger than any
level can run. These are the products this plan leaves out on purpose, what
covers the same risk instead, and what brings them back.

**Design decisions (built into the level rules).**

| Id | Product not run | Why it adds little | Covered instead by | Add it back when |
|---|---|---|---|---|
| D2 | Every extra data type, mixed pair, `CONVOLUTION` mode, mixed layout and non-packed stride × every S class × every layout and dtype | A data type, mode or layout changes kernel selection per tensor, visible on any geometry. | Each × rank × layout or dtype in nightly; each mixed pair, layout mix and stride kind paired with every S class of S1–S31 once in full. | An engine claims the type or mode, or a layout bug needs a geometry feature. |
| D3 | The kernel-selecting classes S32–S90 crossed with the feature probes | These classes select the wrw kernel itself; a per-tensor feature does not depend on which one runs. | Each class in plain wrw from PR up, over every pair in nightly, at L size and under a long reduction (LA) in nightly and full. | A defect needs a feature together with one of these classes. |
| D4 | Hazard shapes × every layout and dtype in 1D and 3D | Hazards act on C, K, N, G and extent; layout and dtype multiply compile work, not hazard cover. | 2D × every pair; 1D and 3D × two pairs (nightly). | A hazard depends on rank. |
| D5 | Three-way products | Pairwise cover catches most interaction faults; the corpus R set adds the real three-way combinations. | All-pairs (§4); R. | A three-way fault is found: add that triple. |
| D6 | The reduction ladder × every S class | The ladder fixes the channels and filter and varies A, so a failure points at A. The classes under a long reduction are the LA forms. | A × every pair, positive fills over the ladder (nightly), LA (nightly, full). | A defect needs a long reduction with a class that has no LA form. |

**Measured trims (applied after case ids are assigned, `conv_wrw_plan.TRIMS`).**
The cases that remain keep their ids, seeds and claims, and undoing a trim
restores exactly what it removed. No trim removes the last probe of a feature
from any level. F11, F15, F16 and F20 are never trimmed.

| Id | Product not run | Cases | Why it adds little | Covered instead by | Add it back when |
|---|---|---|---|---|---|
| T11 | F13, F19 and F35 paired with every S class in full | 240 | No engine executed them, and no reference can judge F13 or F19 (§6). | Each × core in nightly (F13 18, F19 108, F35 9). | An engine executes the feature: remove it from `UNEXECUTED_FEATURES`; for F13 and F19 the reference fixes of §6 must land too. |

### 4.4 Cost

| Estimate | Value |
|---|---|
| FLOP per case | quick ≤ 19 M, PR ≤ 1.9 G (M1), nightly ≤ 7.4 G, full 116 G (X2) |
| Reduction length A | quick ≤ 4,096, PR ≤ 100,352, nightly ≤ 4.2 million, full 16.8 million, and 2.1 billion (X3) |
| Oracle | GPU reference; no case of the measurement runs below used the CPU reference. The GPU reference sums one dw element in one thread, so a long reduction into a small output is its slowest case: the two X3 cases take about 7 minutes each. |
| Compile and solver search | paid once per unique shape on a cold cache |
| Device memory | X3 ≈ 8.6 GB of inputs (`minimum_vram_mb` is set on those cases); X1 and X5 ≈ 0.4 GB |
| Run time, serial, trimmed set | gfx90a (MI250): quick 37 s, PR 2.8 min, nightly 14 min, full 27 min (half of it X3); gfx942 (MI325X): quick 24 s, PR 1.9 min, nightly 10 min, full 18 min |

### 4.5 Measured run

**Which cases execute.** T11 comes from a serial run (one test process at a time,
chunks of 100 tests per process) of all 3,821 cases the level rules generate
(trimmed products included), with the MIOpen plugin, the only engine that runs wrw
today, with the engine TOML's wrw tolerance tables (§6):

| Architecture | OK | Skipped | Failed | Wall time |
|---|---|---|---|---|
| gfx90a (MI250) | 3,081 | 731 | 9 | 45 min |
| gfx942 (MI325X) | 3,089 | 731 | 1 | 30 min |

- Every skip is a probe the plugin declines ("does not support this graph"). F11
  (asymmetric padding), F13, F15 (mixed layouts), F19, F20 and F35 never executed on
  either architecture. F16 executes for the channels-first cases (53 of 109) and is
  declined for every channels-last one. Among the plain cases the plugin declines the
  asymmetric classes S10, S11, L10, L11, LA10 and LA11 and the 4D class S76.
  Everything else of §3 executed on both: every plain wrw class, B1 and B2, the whole
  ladder, the corpus, the L and LA forms, and X1 to X5.
- Both architectures passed X3 (more than 2³¹ elements, 7 minutes each) in that run
  only because its positive fills left a sum smaller than the band. With uniform fills
  fp16 channels-first fails on both: MIOpen's naive wrw kernel is the only solver
  offered and returns 0 (ALMIOPEN-3571).
- The one gfx942 failure and one of the gfx90a failures is the positive-fill case at
  A 4.2 million, where the reference is wrong (§6); the ladder's positive fills now
  stop at A 2²⁰.
- The other eight gfx90a failures are one defect class, ALMIOPEN-3564: bf16
  channels-first 2D 3×3 outputs that are wrong on some runs (H3, H4, H5, M9), and
  fp32 2D 3×3 outputs about 1,000 float roundings off in multipass Winograd and
  assembly implicit-GEMM solvers (L36, L38, L79, R065). They vary with the input
  seed: four runs with different seeds failed 9, 10, 10 and 10 cases in total, 15
  different cases of this class among them.
- Every executed case was judged by the GPU reference (`[ VERIFIER ] gpu_ref`).

**The current set.** The 3,578 cases of the bundles, serial, with the engine TOML. The
`[[test_skips]]` entries are 37 gfx90a cases of ALMIOPEN-3564 (23 shapes: 15 seen
failing and 8 more the ticket names, and the value variants of the S2 and S3 shape; 16
are claim cases) and the X3 case of ALMIOPEN-3571 (fp16 channels-first, on both
architectures):

| Architecture | OK | Skipped | Failed | Wall time |
|---|---|---|---|---|
| gfx90a (MI250) | 3,051 | 527 | 0 | 38 min |
| gfx942 (MI325X) | 3,086 | 492 | 0 | 26 min |

Claims are harvested from two earlier runs of the same cases on other input seeds: a
case is claimed for an architecture when it is a `claim` or `regression` case and
passed on both seeds there. That gives 2,597 claims on gfx90a (linux) and 2,613 on
gfx942 (linux). The 16 gfx90a claim cases the TOML skips are not claimed there.

### 4.6 The deterministic engine

The MIOpen plugin has a second engine, `MIOPEN_ENGINE_DETERMINISTIC`, that avoids
split-K kernels that add partial results with atomics. No lane runs it and it has no
engine TOML, so it is graded at the test SDK's allowance, which has no entry for the
kernels that engine picks. The same 3,578 cases ran on it (serial, one seed):

| Architecture | OK | Skipped | Failed |
|---|---|---|---|
| gfx90a (MI250) | 2,651 | 491 | 436 |
| gfx942 (MI325X) | 3,076 | 491 | 11 |

The failures are accuracy: fp16 outputs of long reductions on gfx90a (its fp16
backward kernels round the operands to bf16), and on both architectures a few
elements of fp16 reductions of A above 100,000 and fp32 outputs at the band's edge
(a difference of 4.8e-7 against 4.77e-7). On gfx942 X3 (the plan's one probe of 2³¹ elements) fails for fp16 and
for bf16 channels-last. Claims for this engine are therefore
not recorded: the old `ConvolutionWrw/Default` sidecars held some, and they are dropped
with the cases they named. Recording them needs an engine TOML with its own tolerance
tables, calibrated like the `MIOPEN_ENGINE` ones.

---

## 5. Golden reference

The selection follows [RFC 0022 §12.2](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#122-policy-and-selection-criteria).
Wrw needs golden data more than most ops: the GPU and CPU references are the
only oracles, and no wrw case has golden data. Cases would be generated by
PyTorch on CPU (`torch.nn.grad.conv{1,2,3}d_weight`) and record `forward_source`.

| Id | Graph | Case | Criterion | Level |
|---|---|---|---|---|
| G1 | wrw | 2D NCHW fp32, S3 | G-A baseline | quick |
| G2 | wrw | 1D NLC fp16, S17 (stride 2) | G-A baseline | PR |
| G3 | wrw | 3D NCDHW bf16, S3 | G-A baseline | PR |
| G4 | wrw | 2D NHWC fp16, S1 (1×1) | G-A baseline | PR |
| G5 | wrw | 3D NDHWC fp32, S17 | G-A baseline | PR |
| G6 | wrw | 1D NCL bf16, S49 (grouped) | G-A baseline | PR |
| G7 | wrw, `CONVOLUTION` mode | 2D NCHW fp32, S3 | G-B oracle gap (F13) | PR |
| G8 | wrw | 2D NCHW fp32, S10 (pre 0, post 1) | G-C asymmetric padding convention | PR |
| G9 | wrw | 2D NCHW fp32, S14 (filter larger than x) | G-C: dw taps no x reaches are 0 (F37) | PR |
| G10 | wrw | 2D NCHW fp32, S13 (filter equal to x, A = N) | G-D: shortest reduction | PR |
| G11 | wrw | 2D NCHW fp32, A5 (A 4.2 million into one element) | G-D: long accumulation | nightly |

Eleven cases, each also a live sweep case. **The generator
(`generate_conv_wrw_golden.py`) does not exist yet (I7); no wrw case has golden
data.** Precision: uplift (fp64, cast) for fp32; matched precision (fp32
accumulation, rounded to the io type) for fp16 and bf16; recorded per case in
`meta.json`. Confirmation is `hipdnn_golden_data_tests --reference both`, except
G7, which is confirmed by a second independent computation (NumPy) until its
reference fix lands. Storage is `golden/<case-id>/tensors.dvc` beside the sweep.

---

## 6. Pass criteria and reference prerequisites

| Item | Rule |
|---|---|
| Verification | `auto`: golden → GPU reference → CPU reference. For wrw the GPU reference judges every executed case (§4.5); the CPU reference is the fallback where the GPU one declines, and golden data takes precedence once it exists. |
| Tolerance | The convolution dynamic tolerance ([Tolerances](../tolerances.md), "Backward data and backward weights"): a statistical band whose reduction length is `K = N × dy volume` (this plan's A), with the test SDK's wrw allowance and the engine's own in `[conv_wrw_tolerance.<type>]` of its TOML. The per-op default (`TestTolerances.hpp` `conv::getToleranceWrw`, used as atol and rtol) does not apply to a long reduction: the absolute error between two correct implementations grows with A, so a static value passes the short reductions and fails the long ones. An engine's allowance is calibrated from observed error and never loosened to make a case pass. |
| Support claims | Enforced. Only `claim` and `regression` cases are recorded (§4.2). |

These features need reference or harness work before their cases can be judged
or run. The generator emits their cases now; they are unclaimed until the work
lands.

| Feature | Gap | Needed | RFC 0022 |
|---|---|---|---|
| F13 `CONVOLUTION` | Both references reject it. An accepting engine fails "no oracle". | Flip the filter in the references; until then golden G7. | I6 |
| F19, F20 | The references register only the three uniform core triples. | Register the types, or golden data. | I6, I7 |
| F11 | No reference test covers asymmetric wrw. | Add one; golden G8. | I6 |
| F21 positive and large fills at A above 2²⁰ | The GPU reference sums one dw element in one thread, sequentially in fp32; 4.2 million products in [0, 1] stagnate to 1.9 × 10⁻³ below the exact sum, outside any band an engine can meet. | A pairwise or compensated sum in the reference; until then the ladder's positive fills stop at A 2²⁰. | I6 |
| F21 subnormal, zero-excluded, sparse; F22 | Fill recipes are `free` and `fixed` only. | Value-domain control; a matching-infinities validator. | I4 |
| F17; F24, F32–F34 | Allocator pointers are 256-byte aligned; bundles run one execution. | Exact placement; repeat-and-compare. | I3, I5 |
| `purpose` | The harness ignores the tag. | The harvester skips probes. | I2 |

Engine-specific tolerances and skips go in the engine TOML, keyed to the case
ids. A case id is lowercase_snake_case, as the bundle verifier requires; the plan
item is its first token (`S3` is `s3`, `R074` is `r074`):
`<plan items>_<rank>d_i<x spatial>_n<N>c<C>k<K>[g<G>]_f<filter>_<dtype>[_dy<dtype>_dw<dtype>]_<layout | x..dy..dw..>`.
A change that renames ids updates the TOML globs in the same pull request.


---

## 7. Exhaustive mode

Wrw has an exhaustive mode: a separate tool (RFC 0022 §14) that reads the
declarations in `conv_wrw_plan.py`.

| Declaration | Where | Content |
|---|---|---|
| Axes | `RANKS`, `LAYOUTS`, `CORE_DTYPES`, `EXTRA_TYPES`, `MIXED_TYPES`, the case fields of `DEFAULTS` | every feature of §2 |
| Legality | `legal()` | §1 and §2.7 |
| Behaviour classes | `signature()`, `stride_class()`, `pad_class()`, `empty_taps()`, `remainder()`, `a_class()`, `filter_class()`, `group_class()` | the properties of §2.1 that select kernels |
| Shape ranges | rank 1–4; N 1–256; C, K 1–2048; G dividing C and K; spatial 1–16000 (1D), 1–224 (2D), 1–64 (3D); filter 1–31; stride 1–16; dilation 1–12; padding 0 – dilated filter and beyond; x + dy + dw ≤ 1 GiB; A up to 2³¹ | `ACCUM_SHAPES`, `X_SHAPES` |
| Value domains | `value_recipes()`, `VALUE_CLASSES` | §2.4 |
| Return path | minimised reproducers become `regression` cases; oracle bugs become G-B golden candidates |
