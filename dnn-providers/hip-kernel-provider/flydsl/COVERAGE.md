<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# What the FlyDSL packs cover

Built once for each of two LLVM generic targets -- `gfx11-generic` (RDNA3 /
RDNA3.5: gfx1100–gfx1103, gfx1150–gfx1153) and `gfx12-generic` (RDNA4: gfx1200,
gfx1201) -- and shipped to every member, and once for each of the CDNA targets
gfx90a, gfx942 and gfx950: **RMSNorm forward**, inference and training (90 kernel
objects per target, §1–§2), and **RMSNorm backward** (63 per target, §2a); and on the two
RDNA families **SDPA forward**, prefill and decode engines (156 per family, §5); on
`gfx12-generic` also **AOTriton's gfx1201 kernel** as an engine of its own (72, §5.6).
**SDPA forward on CDNA** (§7) is three engines -- the generic kernel's prefill, gfx950's
dual-wave kernel and the split-KV decode -- built per arch for gfx950, gfx942 and gfx90a
from FlyDSL's own CDNA kernels, and on gfx950 **SDPA backward** (§9, 2,729 objects). Every graph a pack accepts is
computed by one of its objects; everything else is **declined**, so another
engine gets the plan.

That last point is the design rule this whole file exists to document. The
kernels are pre-built HSACO — [`FlydslRmsNormNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp)
cannot change what they compute, only decide whether they are the right thing to
run. So every condition below is enforced by *refusing*, never by adapting. A
matcher that stretched to accept a near-miss would compute the wrong answer
silently; one that declines it costs a fallback.

---

## 1. The 90 forward instances

Generated from [`generators/_instances.py`](generators/_instances.py), which is
the single source of truth — the compiler, the packer and the descriptor emitter
all read it, so adding an instance is a row there rather than an edit in three
places that can disagree. The rows are the product of four axes:

| Axis | Values | Knob |
|---|---|---|
| x/y dtype | bf16, f16, f32 | `dtype` |
| gamma | x's dtype, or f32 beside bf16/f16 | `scale_f32` 0 / 1 |
| phase | inference, or training that also writes inv_rms | `store_rstd` 0 / 1 |
| width | generic (`N` read at run time), or baked at 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 12288, 16384 | `N` |

Per phase: 3 + 2 generic objects (one per dtype and gamma form) and 2 dtypes × 2 gamma
forms × 10 baked widths, so 45; doubled for training, 90. Names gain `_f32scale1` and
`_rstd1` only when those knobs are set, so the inference objects keep their names. The
f32-gamma form reads gamma as f32 beside bf16/f16 x and y (the kernel's
`weight_dtype_str="f32"`): the mixed form cuDNN and PyTorch commonly send. f32 has no
baked tier because baking was measured for bf16 only (below) and an f32 baked tier
never was. An f32 object with `N` baked in would not take the 128-bit fast path (16-bit
elements only), but it would run the unrolled compile-time-`N` loop rather than the
generic one, which is where five of the ten 16-bit baked widths get their gain.
`block_threads` is single-valued
today: it is launch geometry, not correctness, so a second value only earns its
6 extra objects once benchmarking says it wins somewhere. It stays in the knob
dict so the descriptors and the matcher already carry the axis when it grows.

### Why these widths

By measurement. Each candidate width was timed baked against the generic tier (kernel
time from the profiler, bf16, M = 4096 and 16384) on gfx90a, gfx942, gfx950, gfx1201 and
gfx1100, and kept where baking gained at least 15% on at least two of them. Every baked
width gained on all five, by 1.2–4.2×. 3584 (Qwen2-7B) predates the measurement.

**Widening it is cheap, and is not a coverage fix** — every other `N` is already
served, by the generic tier. Adding a width buys speed on that width, nothing
else.

### Why the two tiers exist

FlyDSL's vectorized 128-bit path caches tiles in a Python list of registers,
which only exists while the tile loop is unrolled, and unrolling needs a
compile-time count. So the fast path **genuinely cannot take a runtime `N`** —
the tier split is forced by the kernel, not chosen for convenience. The fast path
only runs when `N` is a multiple of its 2048-element tile, though, so of the baked
widths only 4096, 6144, 8192, 12288 and 16384 use it; 2560, 3072, 3584, 5120 and 7168
run the scalar loop unrolled at their compile-time `N`, and were still measured
≥ 1.2× faster than the generic tier. The baked list is chosen by that measurement,
not by which widths reach the fast path.

The generic tier is the vendored runtime-`N` variant (modification 1 in
[`kernels_src/kernels/norm/rmsnorm_kernel.py`](kernels_src/kernels/norm/rmsnorm_kernel.py)):
it reads `N` from the input tensor descriptor and predicates the tail, so one
binary covers every width at the cost of the fast path.

Both tiers stay in the candidate set when both apply — the matcher admits both
for a baked width, and the tie is broken by ranking, not by refusal.

Which one wins is expressed purely as **data**. `IKernelHeuristic::rank` orders
`score desc → priority desc → kernelId asc`, and `flydslRmsNormScore` returns
`kernel.priority` unchanged rather than inventing a second opinion — so each row's
`priority` in `_instances.py` (`PRIORITY_SPECIALIZED` 100, `PRIORITY_GENERIC` 10,
carried into each object's descriptor) *is* the ranking, decided at the first
comparison (100 > 10, baked before generic). The later tiebreaks never come into play here,
which is the point: selection is authored in the descriptors and refined by the
autotuner, not computed from the shape. A score derived from the shape would
also make the winner cache's key insufficient.

### The gap that is deliberate: N ≤ 2048

`RMSNORM_SMALL_N_THRESHOLD = 2048`, and 2048 is **absent** from the specialized
list on purpose. At or below that width `build_rmsnorm_module` does not
specialize the kernel above — it returns `_build_rmsnorm_large_m_small_n_module`,
a *different* kernel with a different ABI (an extra `m_in` kernarg) and a
different launch geometry (grid divided by `BLOCK_M` rather than one block per
row). Asking for `N=2048` here would not produce a specialization of this kernel
at all.

That path is worth having — it is upstream's large-M/small-N specialization —
but it is a second launch record, so it lands as its own instance family rather
than hiding inside this one. **Until then small `N` is served by the generic
tier**, correctly but without the specialization. The generator asserts the
Python threshold and the table's constant still agree, because drift there is
the kind that produces a correct-looking object dispatched with the wrong grid.

---

## 2. What a graph must be

Two matchers run, in this order.

### Graph-scoped — `flydslRmsNormGraphMatches`

Evaluated once per graph. Every one of these is a hard refusal.

| Condition | Why |
|---|---|
| Exactly **one node**, and its attributes are `RMSNormAttributes` | Each kernel serves one complete graph. `RMSNormAttributes` is its own union arm, so matching it is the whole node-type test — there is no mode enum to disambiguate it from layer norm. |
| No `bias_tensor_uid` | Every instance was compiled with no bias term and adds nothing after scaling. |
| `inv_rms_tensor_uid` present exactly when `forward_phase == TRAINING`; otherwise NOT_SET or INFERENCE | A training graph wants the statistic even when it forgot to name the tensor, and one that names it outside training asks for an output cuDNN's semantics do not produce: both decline. inv_rms must be FLOAT, x's rank, x's extent on every row axis and 1 on every normalised one, packed, 4-byte aligned: the `store_rstd` objects write one f32 per row. The kernel matcher pairs `store_rstd` with it. |
| Node compute type `UNSET`, `FLOAT`, `HALF` or `BFLOAT16` | The kernels accumulate in f32, which is at least the precision any of these asks for; any other (e.g. `DOUBLE`, `INT32`) asks for one they do not provide. |
| x, scale, y all present and all **device operands** | Not virtual, not pass-by-value, not ragged (no `ragged_offset_tensor_uid`): the kernel reads each through a pointer as a dense tensor. |
| x, scale, y each declare an alignment that is a multiple of 16 bytes (the schema default) | The fast path copies 128-bit vectors from each operand's base, and the backend enforces only the alignment a graph declares. |
| x and y one dtype, in {`BFLOAT16`, `HALF`, `FLOAT`}; gamma in that dtype, or `FLOAT` beside `BFLOAT16`/`HALF` | Each object reads x and writes y in one element type, and reads gamma in that type or, for the `scale_f32` objects, as f32. |
| x, y rank ≥ 2; scale rank ≥ 1; `dims` and `strides` both present and equal length | Rank ≥ 2 so there is a row to reduce over. The matcher runs on an *unvalidated* graph, so it must be total against null or mismatched dims. |
| `sameShape(x, y)` | |
| x, y, scale all **packed row-major**, ignoring the stride of any axis of extent 1 | The innermost stride is a compile-time literal 1 in the kernel and has no descriptor slot, and the launch passes the width as the row stride. The check is stronger than the descriptor strictly needs — it could carry a padded row stride — but it is what makes collapsing rank > 2 to (M, N) sound, since one row stride has to describe all the leading axes at once. An axis of extent 1 is never stepped along, so its stride is not checked: hipdnn_torch describes an `F.rms_norm` weight with stride 0 on its unit axes, and the data SDK's packed test exempts them the same way. |
| rows > 0 and columns > 0, and every element of x and y within 2³¹ − 1 bytes | Each operand is one buffer view with 32-bit byte offsets; the bound counts bytes, so f32 reaches it at half the elements. |
| gamma **encodes the normalised shape** as the frontend reads it: the normalised axes are the longest trailing run where gamma's extent equals x's (never axis 0), and gamma is 1 on every axis before it | Any other shape would be a broadcast the kernel does not perform. The rule of `validateScaleNormalizedShape` in the frontend, but for rank: a gamma of lower rank than x, which the frontend rejects, is aligned at the innermost axis with its missing leading axes read as 1, so the matcher stays total. |
| epsilon is one scalar: present, with dims, and extent 1 on every axis | The kernel takes one f32; a tensor of more elements would be read as its first. |
| epsilon is runtime-user-supplied, **or** a finite, non-negative f32 | See below. |
| The graph is not override-shape | Rows and width are bound from the graph's shapes at prepare, which an override-shape graph may replace at execute. |

Rank > 2 is accepted, not rejected: the axes before the normalised run collapse to
a row count and the run itself to the width, both sound because x is packed. So a
rank-4 graph with a `(1,1,1,N)` scale is a match, with packed strides or with stride
0 on its unit axes; so is cuDNN's layout, `x {B*S, D, 1, 1}` with
`scale {1, D, 1, 1}` (D rows wide, not 1), and a multi-axis normalised shape such
as `x [8, 4, 1024]` with `scale [1, 4, 1024]` (8 rows of 4096, served by the
baked N = 4096 instance).

Before the matcher runs, the ingestor compares the graph's schema floor with the
engine's: the UED declares `sdk_version` 1.2.0, so a runtime pass-by-value
epsilon reaches the matcher. A graph that declares a non-default tensor
alignment needs 1.3.0 and is declined there.

On success it binds `flydsl_rmsnorm.{x,scale,epsilon,y}.uid`, a training graph's
`flydsl_rmsnorm.inv_rms.uid`, plus `.rows` and `.columns`, so nothing downstream
re-derives them from the graph.

### Kernel-scoped — `flydslRmsNormKernelMatches`

Evaluated once per candidate kernel, against the KMD metadata:

- `scale_f32` must be 1 exactly when gamma's dtype differs from x's (f32 beside
  bf16/f16), so an object reads gamma in the type it has.
- `dtype` metadata must equal the graph dtype's kernel spelling. This function
  is the one place the two vocabularies meet — descriptors say `bf16`/`f16`/`f32`,
  the flatbuffer enum says `BFLOAT16`/`HALF`/`FLOAT`.
- The baked `N`: `GENERIC_N_SENTINEL = 0` admits any width; any other value is
  admitted only when it equals the bound `columns`. A real graph never has a
  zero-width normalised axis, so the value is free to carry the sentinel.

### Epsilon, a kernel argument

Epsilon is the kernel's last argument, an f32 (vendored modification 4), so one
object serves every value: PyTorch's 1e-5, vLLM's and HF's 1e-6, and `eps=None`
(f32's machine epsilon) alike.

- A graph that **bakes** its epsilon into the op-graph is matched for any finite,
  non-negative f32, and declined otherwise (negative, NaN, infinite) — falling
  through to another engine rather than failing the plan build.
- A graph that **defers** epsilon to execute cannot be inspected then, so
  `launch()` checks the resolved value the same way and throws
  `HIPDNN_PLUGIN_STATUS_INVALID_VALUE` on one that is not usable.

---

## 2a. The backward pack — `hipkernel:flydsl_rmsnorm_bwd`

[`FlydslRmsNormBwdNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormBwdNative.cpp),
63 objects per target from the vendored two-stage builder
([`kernels_src/kernels/norm/rmsnorm_bwd_kernel.py`](kernels_src/kernels/norm/rmsnorm_bwd_kernel.py),
two recorded modifications). Each object holds two kernels: the partial kernel walks
the rows with `programs` blocks of 512 threads, writes dx, and leaves one f32 dgamma row
per program in the workspace; the reduce kernel, named by `reduce_symbol`, sums those
rows into dgamma in gamma's dtype. **dgamma is deterministic**: no atomics, and the same
program count and row order on every launch of a plan.

| Axis | Values |
|---|---|
| x/dy/dx dtype; gamma | as the forward: bf16, f16, f32; gamma in x's dtype or f32 beside bf16/f16 |
| width | baked at 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 12288, 16384 (16-bit), or a **bounded** generic tier: `n_max` 1024, 8192, 16384 |

The generic tiers bound the width instead of taking any: the partial kernel holds each
thread's dgamma columns in registers across rows, so `n_max` sizes them. 1024 → ~22
VGPRs, 8192 → ~115, 16384 → ~240, none spilling VGPRs; 32768 spills, so wider graphs
are declined. (No shipped object spills VGPRs or uses scratch; the `n_max` 16384 partial
kernels on the CDNA targets spill a few SGPRs — 15–16 on gfx90a, 3 on gfx942/gfx950 —
into VGPR lanes.) The narrower tier ranks first. Baked widths were chosen by the same
measurement as the forward's and gained 1.2–8.4× on all five archs; 128, 512 and 1024
stay generic (within ~25%).

**Program count** follows upstream's selector: one per CU below 2048 rows on the vec8
path (1024 otherwise), 1.5 per CU above it on the vec8 path, 2 per CU otherwise, never
more than the rows. The workspace is programs × N × 4 bytes. Measured against a flat 2
per CU on gfx90a/gfx942/gfx950: within 2% almost everywhere, up to 10% faster at
4096 × 8192. An **atomic** dgamma (upstream's other builder) was measured and not
shipped: 2.6–90× slower at large M from contention, and its small-M edge was inside
host launch overhead.

**The graph matcher** takes one RMSNormBackward node: no dbias; compute type `UNSET`,
`FLOAT`, `HALF` or `BFLOAT16` (as the forward); every operand a device operand (not
virtual, pass-by-value or ragged); x, dy, dx one dtype; gamma and dgamma one dtype, x's
or f32 beside 16-bit x; dx shaped as x and dgamma as gamma; inv_rms one f32 per row (the
forward's rule); every row operand and gamma/dgamma packed and 16-byte aligned; gamma
encoding the normalised shape; every element of x, dy and dx within 2³¹ − 1 bytes (the
forward's bound); not override-shape. **The kernel matcher** pairs dtype and gamma
form, and serves a baked width exactly or a generic one up to its `n_max`.

---

## 3. Out of scope

Not gaps to be filled before this ships — each is something the shipped objects
do not compute, declined rather than approximated.

| | Why, and what handles it |
|---|---|
| **Bias, dbias** | No instance has the term; neither upstream backward builder computes dbias. |
| **Backward wider than 16384** | The bounded generic tiers stop there (§2a). |
| **dtypes other than bf16 / f16 / f32** | fp8 and the rest are not built. Declined on dtype in both matchers. |
| **Other mixed dtypes** (y unlike x; a 16-bit gamma unlike x; a 16-bit gamma beside f32 x) | Only an f32 gamma beside bf16/f16 x and y has objects. |
| **Non-packed / padded / transposed operands** | Innermost stride is a literal 1 with no descriptor slot, and the launch passes the width as the row stride. Unit axes are exempt (§2). |
| **Rank < 2 on x or y** | No row to reduce over. |
| **Gamma that does not encode a normalised shape** (e.g. `[1,2,1,3]` against `[1,2,3,3]`) | Not a broadcast the kernel performs; the frontend rejects it too. |
| **Multi-node graphs, fused add+RMSNorm** | One node per kernel. Fusion would be its own pack. |
| **Architectures outside gfx11-generic, gfx12-generic, gfx90a, gfx942 and gfx950** | Objects ship for the gfx11 generic family (gfx1100–gfx1103, gfx1150–gfx1153), the gfx12 one (gfx1200, gfx1201), and the CDNA targets gfx90a, gfx942 and gfx950. gfx1170/gfx1171 belong to `gfx11-7-generic`. With `GPU_TARGETS` naming no arch we ship kernels for, the integration reports dormant at configure time and stages nothing — it does not fail, and it does not silently produce an empty shard. Adding a family is a regeneration ([REGEN.md](REGEN.md) §3), not a code change. |
| **Ops other than RMSNorm** | Not these packs': `OPS` in `_instances.py` holds RMSNorm forward and backward, the SDPA forward families (§5, §7) and the SDPA backward ops (§9), each engine its own. |

---

## 4. Where each claim is checked

| Claim | Checked by |
|---|---|
| The 153 objects per target (90 forward, 63 backward) are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 — SHA256 against `manifest.json` |
| Descriptors agree with the objects they describe | `gen_descriptors.py --check`, run by the build before staging |
| Each object is built for the target its directory names | `gen_descriptors.py --check` reads `amdhsa.target` and, for a generic target, the ELF machine |
| Every member arch's shard holds exactly the checked-in objects | `tools/check_shards.py`, run by the build after the product pack |
| The vendored sources match upstream but for recorded modifications | `tools/diff_upstream.py` ([REGEN.md](REGEN.md) §5) |
| The pack ships the generic tier for every dtype and gamma form and the specialized tier for the 16-bit ones, every kernel declares the 9-slot signature (epsilon last), and the engine's schema admits runtime epsilons | `TestFlydslRmsNormPacks` — and the shard census that runs it, which fails when the suite is green against an empty shard |
| The pack ships the backward tiers per form, each object names its reduce kernel, and every kernel declares the 15-slot partial signature | `TestFlydslRmsNormBwdPacks` and its shard census |
| The accept/refuse rules above hold | The matcher suites in `TestFlydslRmsNormEngine.cpp` and `TestFlydslRmsNormBwdEngine.cpp`. `TestFlydslRmsNormGraphMatcherRefusal` and `TestFlydslRmsNormBwdGraphMatcherRefusal` hold one host refusal case per gated field — forward: y dtype, shape and strides, pass-by-value operands, a second node, an alignment that is not a multiple of 16, strides of the wrong length, an inv_rms alignment that is not a multiple of 4; backward: dy and dgamma shape, per-operand packing, alignment and virtual-ness, two nodes. `TestFlydslRmsNormKernelMatcher` and `TestFlydslRmsNormBwdKernelMatcher` test each width rule in both directions (served and refused). The GPU suites there check y, inv_rms, dx and dgamma against host references and dgamma's determinism |
| Every staged object computes what its name says | `TestGpuFlydslRmsNormObjects` (`TestFlydslRmsNormObjects.cpp`): launches each object by name through its pack's dispatch handler against an f64 host reference, into NaN-prefilled outputs, with a different scale per row, the backward twice and compared bit for bit. `OnePerForm` runs one object per (dtype, gamma form, inv_rms, tier) in every test tier, pre-merge included; `Full/...Every` runs all 153 objects of the device's target, in the full (nightly) tier only |
| The baked tier is faster than the generic one where both apply | `Full/TestGpuFlydslRmsNormTiers` (wall-clock comparison), full tier only |
| A device case cannot pass by skipping on an arch FlyDSL ships for | Each GPU case that finds its objects unstaged **fails** on an arch the build ships FlyDSL for, and skips only on other arches |
| `gen_descriptors.py --check` and `tools/check_shards.py` still refuse a tampered tree | CTest `hip-kernel-provider-flydsl-drift-guards` (pytest, `flydsl/tests/test_drift_guards.py`): copies the content tree and one packed shard, breaks one thing at a time, and requires each check to refuse it. Host-only, quick tier |

Tiers: GTest names prefixed `Full/` are excluded from the quick, standard and
comprehensive tiers by `HIP_MLOPS_ENGINE_test_categories.yaml` and run only in the full
(nightly) tier; every other case above runs in every tier.

---

## 5. SDPA forward — `hipkernel:flydsl_sdpa_rdna` and `hipkernel:flydsl_sdpa_rdna_decode`

Two engines, one kernel family each: the prefill kernel below, and the split-KV decode
kernel (`flash_attn_decode_rdna.py`; the GQA group packed into 16-row tiles, the keys
split across workgroups, a merge kernel in the same object), with its own argument list,
launch and workspace. Both apply the same graph matcher (`FlydslSdpaRdnaCommon.hpp`), so
they accept the same graphs; each kernel matcher declines only what its kernel cannot
compute. Which engine runs a graph both serve is the engine-level heuristic's (UHD) to
decide, or autotune's — no rule here prefers one.

The prefill kernel is FlyDSL's RDNA4 flash-attention forward from AITER, ported to the
gfx11 WMMA ABI and widened so a handful of objects cover many shapes
([`kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py`](kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py),
whose header lists every modification). The native halves are
[`FlydslSdpaRdnaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaNative.cpp)
and
[`FlydslSdpaRdnaDecodeNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaDecodeNative.cpp).

### 5.1 What is baked, and what is not

| Baked per object | Runtime kernel argument |
|---|---|
| dtype (`bf16`, `f16`) | batch |
| `head_dim` | `seq_len_q`, `seq_len_kv` — independently, any value |
| causal variant: a right bound is applied, or not | query heads, and the GQA/MQA group size |
| schedule: tile (`block_m` 64 or 128, `block_n` 32), output-column split (`dv_split`), head-dim pairs (`hd_pair`) | every operand's (batch, sequence, head) stride |
| whether an additive bias is applied (`has_bias`) | the bias's (batch, head, query, key) strides — 0 on a broadcast axis |
| | softmax scale |
| | `right_bound`, `left_bound`, diagonal alignment |
| | whether to write the LSE output, and its strides |

So the instance key is `dtype × head_dim × causal × has_bias × bias_q`, and every axis
it enumerates is **model-determined**, never request-determined. That is what
keeps the specialized table at 48 rows — each cell below is one object without a bias,
one with an f32 bias and one with a bias in the query's dtype (the MLA, sink and generic
objects below are beside it):

| | `head_dim` 64 | `head_dim` 96 | `head_dim` 128 | `head_dim` 256 |
|---|---|---|---|---|
| bf16 | causal, non-causal | causal, non-causal | causal, non-causal | causal, non-causal |
| f16 | causal, non-causal | causal, non-causal | causal, non-causal | causal, non-causal |

The schedule is launch geometry, chosen per class by measurement; the dispatcher
reads it from the metadata. Each schedule below was measured against the one it
replaced on the same inputs, correct to the same error, and kept only where it
won:

| Class | `block_m` | `dv_split` | `hd_pair` | Why |
|---|---|---|---|---|
| `head_dim` 64, no bias | 64 | 1 | 0 | twice the workgroups of a 128-row tile, with Q still in registers |
| `head_dim` 128, non-causal, no bias | 64 | 1 | 0 | the same; most visible on short and BHSD-layout prefill |
| `head_dim` 96; 128 causal or with a bias; 64 with a bias | 128 | 1 | 0 | a 64-row object here has to re-read Q per K-step to fit, which costs more than the tile saves |
| `head_dim` 256 | 64 | 1 | 1 | head-dim pairs, below |

`head_dim` 256 is a schedule of its own (kernel modifications 13 and 18). Up to
128 each wave holds its Q rows and every O accumulator in registers; at 256 those
two alone would fill the 256-VGPR file. So two waves share each 16 query rows:
each computes QK^T over half of the head dimension, re-reading its half of Q one
K-step ahead, the two halves are summed through LDS, and each accumulates half of
O's columns. QK^T is computed once per row, at 256 threads per workgroup. (The
earlier schedule split the output columns across two workgroups that each
recomputed the whole of QK^T.)

**Every other head_dim** that is a multiple of 8, up to 256, is served by a
generic tier (kernel modification 15): twenty-four more objects, bf16 and f16 ×
causal × largest head (64, 96, 128, 160, 224 or 256), without a bias. The actual
head_dim is a runtime argument; Q and K columns past it are zeroed as they load
and O columns past it are not stored, so the tensors are read exactly as wide as
they are. A generic object computes at its largest width, so the tiers are cut
where the odd head dims models use fall: 40, 48 and 56 run at 64; 72, 80 and 88
at 96; 136 to 160 at 160 and 168 to 224 at 224. The tier up to 64 uses a 64-row
tile, as the specialized `head_dim` 64 does (at 96 and 128 a 64-row tile measured
slower or mixed, so those keep 128); the tiers above 128 use head-dim pairs, as
the 256 object does; the 256 tier also splits its columns across two
workgroups (`dv_split` 2), because its masked loads on top of the pair schedule do
not fit the register file in one. They rank below the specialized objects, each
narrower tier above the wider (`priority` 40, 30, 20, 15, 12, 10), so under the static
order 64/96/128/256 run their own object and the narrowest generic tier takes the rest
(autotune, timing every candidate, may find otherwise). A bias
on a head no specialized object serves declines.

The causal variant is the one that applies a right bound — top-left or
bottom-right causal, and any band to the right of the diagonal. It is two
objects rather than a runtime flag because it changes the loop: it bounds the
KV loop at the diagonal and skips fully masked tiles, and the non-causal one
carries a V prefetch across iterations that the causal one drops for register
pressure. Everything else about the mask is a runtime argument, served by both.

**Decode** has its own seventy-two objects (`hipkernel:flydsl_sdpa_rdna_decode`; bf16 and f16 × causal × `head_dim`
64, 96, 128, 256 without a bias, with an f32 one and with one in the query's dtype, plus
two generic tiers, the wide 576/512 and 512/512 objects and the sink objects at 64 and
128), from a kernel
written for it
([`flash_attn_decode_rdna.py`](kernels_src/kernels/attention_rdna/flash_attn_decode_rdna.py)).
The prefill objects give a decode step 128 query rows of which one is used, and read
each KV head once per query head. The decode kernel packs the query heads of one KV
head (the GQA group) and the query positions into 16-row tiles, one workgroup per tile,
so a graph reads each KV head once per tile rather than once per query head. Its objects
without a bias take as many 16-row tiles as the grid's second dimension holds (65,535),
so they also claim graphs with far more than a decode step's `(Hq / Hk) × Sq` rows, where
the prefill objects, which share each K/V tile across 128 rows, are usually faster; which
engine runs such a graph is autotune's (UHD's) to decide. The bias objects take one tile
(16 rows): at `head_dim` 128 and up they already fill the register file, and a tiled
launch's row index spills. It splits the keys a row can see --
from the window's edge to the diagonal -- across workgroups, which the dispatcher
sizes to the device: two workgroups per unit the device reports, because RDNA reports
workgroup processors of two CUs each; and merges the splits by their LSEs with a
second kernel compiled into the same object. Their `priority` (150) ranks them only
within the decode engine; it does not compare them with the prefill objects, which are
another engine's. At `head_dim` 256 two workgroups share
each split's rows, one per half of the output columns (`dv_split` 2), to fit the
register file. The bias objects read the same broadcast bias
as the prefill ones, each lane its own (head, position) row of the GQA group's slice.
Every other head_dim that is a multiple of 8 runs on a generic decode object -- up to
96, or above it up to 256 with the column split -- that reads it at runtime and never
fetches the K and V columns past it, so a step moves only the bytes its head has (an
up-to-128 tier does not fit the register file). Within the decode engine they rank
below the exact objects, the narrower tier above the wider (`priority` 145, 140).
K and V with different head counts stay on the prefill objects. A split launch takes
workspace (a partial O, the row max and the row sum per split, so the merge weighs each
split by its key count even when a bias near the dtype's lowest value puts every max far
below zero); one split writes O directly.

### 5.2 What a graph must be

One SDPA-forward node, and:

| Condition | Why |
|---|---|
| Q, K, V, O present, rank 4, device operands, **not ragged**, each declaring 16-byte alignment; no two elements of O at one address | The kernel reads each through a pointer as a dense batch; a ragged (THD) operand would be read as one. Rows move as 16-byte vectors from each base, and many workgroups write O at once. |
| One dtype for all four, `BFLOAT16` or `HALF` | One element type throughout; no fp32 WMMA operand form exists. |
| Head-dim stride 1 on all four; other strides positive | Each lane reads a row's head-dim values as one contiguous vector. Every other stride is a kernel argument, so BSHD, BHSD and packed-QKV views are all served. |
| K and V share batch and sequence; K has Q's head dim; O is Q's shape with V's head dim; V's head dim at most Q/K's, and one an object bakes: equal to Q/K's, 128 with Q/K 192 (MLA prefill), or 512 with Q/K 576 (absorbed-MLA decode) | No broadcast. The graph matcher admits any V no wider than Q/K; each object's kernel matcher then requires its baked `head_dim_v`. The MLA objects bake both head dims (kernel modification 20); every other object has one. |
| `Hq % Hk == 0` and `Hq % Hv == 0`; K and V may differ in head count | The K head and the V head are each the query head divided by its own group size. |
| Any `left_bound`/`right_bound` ≥ −1 at either named `diagonal_alignment`, or the deprecated causal booleans (not both) | The kernel applies the reference's two-sided rule (`CpuFpReferenceSdpa` `isMasked`) directly. A boolean sets `right_bound` 0 unless the graph sets one: `causal_mask` at the stored `diagonal_alignment` (bottom-right when it says so), `causal_mask_bottom_right` always bottom-right. An alignment the schema does not name declines, boolean or not. |
| Scale: `attn_scale_value` (> 0 and ≤ 2²⁰), a pass-by-value scalar tensor, or absent | Absent is 1.0, cuDNN's default, which hipDNN's references apply too. Every FlyDSL SDPA family serves the same range (`FlydslSdpaShared.hpp` `MAX_SOFTMAX_SCALE`): positive, because the CDNA kernels take the running max over unscaled scores, and at most 2²⁰, beyond which each family overflows f32 at a different point, so all decline alike. A pass-by-value scale is read from host memory at launch, and one outside the range is refused there. |
| Stats, if requested: a named f32 `[B, H, Sq, 1]` device tensor, 4-byte aligned, positive strides on its axes longer than 1, no two elements at one address | The kernel writes the natural-log LSE of the scaled scores there, through its strides. |
| Bias (`attn_mask`), if present: an f32 device tensor, or one in the query's dtype, declaring 16-byte alignment, of rank 1–4 whose dims, right-aligned to `[B, H, Sq, Skv]`, are each that extent or 1; positive strides on real axes; one (batch, head) slice under 2³⁰ elements | Added to the scaled scores before the mask, as the reference does; a size-1 or absent axis broadcasts. The kernel reads it through a 32-bit buffer descriptor with 32-bit offsets. |
| `Sq`, `Skv` < 2²⁹; one (batch, kv head) slice of K or V under 2 GiB; grid under 2³¹ | int32 mask arithmetic, and 32-bit bounds-checked buffer descriptors. |

Any sequence length is correct: K and V reads past `seq_len_kv` return zero and
every such score is masked, so nothing is padded and nothing pads the softmax.
A row no key reaches — a narrow window, or bottom-right causal with more
queries than keys — gets O = 0 and LSE = −inf, as the reference defines it.

### 5.3 Against common SDPA feature requirements

| # | Requirement | Status |
|---|---|---|
| 1 | fp16, bf16; fp32 accumulate | **served** |
| 2 | BSHD, BHSD, general B/S/H strides, packed QKV | **served**; packed QKV checked on the GPU as strided views into one buffer |
| 3 | MHA, MQA, GQA (`Hq % Hk == 0 && Hq % Hv == 0`) | **served**, including K and V with different head counts |
| 4 | attention scale | **served** (positive, at most 2²⁰) |
| 5 | causal, top-left | **served** |
| 6 | causal, bottom-right, `Sq ≠ Skv` | **served** |
| 7 | softmax stats / LSE | **served** |
| 8 | arbitrary sequence lengths | **served** |
| 9 | head dims 64, 128, 256; `d % 8 == 0` | **served**: 64/96/128/256 specialized, every other multiple of 8 up to 256 by the generic tier (not with a bias) |
| 10 | padding mask via `SEQ_LEN_Q`/`SEQ_LEN_KV` | missing |
| 11 | sliding window (`left_bound`) | **served** (and right-side bands) |
| 12 | varlen / THD | missing (declined) |
| 13 | decode (`Sq == 1`) | **served by a decode path**: GQA-packed rows in 16-row tiles (one tile, 16 rows, on the bias objects) and split-KV, K and V with one head count (MQA, speculative steps) at `head_dim` 64/96/128/256 and every other multiple of 8 up to 256 |
| — | `Dv ≠ Dqk` (MLA) | **served** for un-absorbed prefill (Q/K 192, V 128) and absorbed MLA (Q/K 576, V 512) on the decode objects, which tile any number of rows, so absorbed chunked prefill runs there too; every mask, LSE, no bias. The 576/512 objects are built for absorbed MLA's one KV head, but the matcher only requires K's and V's head counts to match |
| 14 | additive bias | **served**: f32 or the query's dtype, broadcast over any of B, H, Sq, Skv; with every mask, the LSE output and every specialized head dim, on the prefill and the decode objects |
| — | attention sink | **served** at `head_dim` 64 and 128 (gpt-oss), prefill and decode, with every mask and the LSE output, no bias: one logit per query head in the query's dtype, rank 1 `[Hq]` or rank 4 `[1, Hq, 1, 1]`, a key with no value in each row's softmax (kernel modification 21). A row no key reaches gets O = 0 and LSE = the sink. hipDNN's CPU reference declines sinks; the GPU tests check them against their own reference |

### 5.4 What is still missing, and what closing it takes

Each is declined today, so the graph plans on another engine or fails cleanly
at `check_support()`, never at `execute()`.

| | Effort | What it takes |
|---|---|---|
| **Padding mask** | medium | Two per-batch length pointers read once per workgroup. The diagonal offset is already computed in-kernel from the lengths for exactly this. hipDNN's CPU reference does not implement padding yet, so its semantics must be defined there first. |
| **Varlen / THD** | medium, after padding | Ragged offsets replace the batch stride; the grid stays over the longest sequence and tiles past a sequence's end exit early. |
| Bias on a generic head_dim | small | A `has_bias` variant of the twenty-four generic prefill objects (+24). |
| A prefill schedule for absorbed MLA; other `Dv ≠ Dqk` pairs | medium; small per pair | 192/128 prefill (modification 20) and the 576/512 and 512/512 decode objects ship. Chunked prefill at 576/512 runs on the decode objects' 16-row tiles; a prefill schedule at that width would share each K/V tile across more rows. Other pairs are a row each. |
| fp32 I/O, FP8, dropout, paged KV, block masks, ALiBi, softcap | out of scope | Not planned for these engines; ALiBi and softcap have no reference semantics. |
| **Backward** | out of scope here | These engines are inference-only. LSE is emitted so a backward can be added without an ABI break. |
| **Other architectures** | see below | The objects are `gfx11-generic` and `gfx12-generic` builds, so every RDNA3 / RDNA3.5 / RDNA4 part is served. |

**Every gfx11 generic member** (gfx1100–gfx1103, gfx1150–gfx1153) is served by
the one object set; the GPU suites (`TestGpuFlydsl*`) pass on gfx1100, gfx1101
and gfx1151, and the other five members rest on the loader's generic-target rule,
the ELF machine check and the shard check.
FlyDSL 0.3.4 needs the `_generic_targets.py` shim to lower for a generic target
(REGEN.md §3), and the SDPA objects are built with `amdgpu-use-amdgpu-trackers`,
without which d128 causal spills a few VGPRs: the generic ISA lacks gfx115x's
scalar-float instructions. **gfx1170/gfx1171** need `gfx11-7-generic` objects.
**gfx12** (gfx1200, gfx1201) has its own `gfx12-generic` object set from the same
sources: its WMMA operand layout differs, so one object cannot span both families,
but each kernel branches on the layout at the few sites that depend on it
(modification 17). The gfx12 set compiles with no spill, its argument layouts and
ELF target are checked, and the GPU suites pass on gfx1201; gfx1200 runs the same
objects.

### 5.5 Where each claim is checked

| Claim | Checked by |
|---|---|
| The objects are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 — every object byte-identical, per target |
| Each decode object carries its merge kernel, with both argument layouts as declared | `gen_sdpa_rdna_decode.py`, before the object is written; the 40-slot main layout by `TestFlydslSdpaRdnaDecodePacks` |
| Each prefill object has the declared 38-argument layout and spills no registers | `gen_sdpa_rdna.py`, before the object is written |
| The vendored kernel matches AITER but for its recorded modifications | `tools/diff_upstream.py --aiter` ([REGEN.md](REGEN.md) §5) |
| Each engine's shard ships exactly one object per `dtype × head_dim × causal × has_bias × bias_q × head_dim_max × has_sink` class, each with its engine's one signature | `TestFlydslSdpaRdnaPacks` and `TestFlydslSdpaRdnaDecodePacks`, and the shard census that runs them |
| The accept/decline rules above, the bindings, and candidate selection | `TestFlydslSdpaRdnaGraphAccepts`, `TestFlydslSdpaRdnaGraphDeclines`, `TestFlydslSdpaRdnaBinding`, `TestFlydslSdpaRdnaKernelMatch` |
| The staged objects compute attention and LSE — GQA, ragged lengths, cross-attention, both causal corners, windows, bands, keyless rows, decode, `head_dim` 96, runtime and default scale — against a double-precision reference | `TestGpuFlydslSdpaRdnaDispatch` |

---

### 5.6 AOTriton's gfx1201 kernel — `hipkernel:flydsl_sdpa_aiw`

AOTriton's FlyDSL forward for gfx12 (the kernel PyTorch's flash path runs there), vendored
under [`kernels_src/kernels/attention_rdna4/`](kernels_src/kernels/attention_rdna4/README.md)
with modifications for hipDNN's feature surface, as an engine of its own: its argument
list (50 slots), its 3-D grid (query heads, query tiles, batch) and its tuning knobs share
nothing with §5's kernels. `gfx12-generic` only; the native half is
[`FlydslSdpaAiwNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaAiwNative.cpp),
which applies §5's graph matcher and declines only what this kernel cannot compute.

- **Served:** bf16 and f16; `head_dim` 64, 128 and 256 exactly, any multiple of 8 up to
  a padded tier's width (64, 96, 128, 160, 224, 256), Q/K 192 with V 128; causal at
  either alignment, sliding windows and two-sided bands (hipDNN's bottom-right window is
  the kernel's shifted by `Skv - Sq`); an additive mask in Q's dtype or f32, broadcast
  over B, H and Sq, its key axis contiguous; attention sinks at 64 and 128; a V head
  count of its own; the LSE through its own strides; a scale baked or read at launch.
- **Edge cases, as §5's prefill:** a mask at the lowest finite value (bf16 or f32
  `finfo.min`, the HF padding convention) on every key of a row gives O = mean(V) and an
  LSE near the mask, as PyTorch does; a mask or sink above 2.36e38 no longer overflows
  (README modification 9: the scores, max and sink live in the natural-log domain, the
  RDNA kernel's modification 22). The LSE of a row whose softmax sum is NaN is NaN.
- **Declined:** a mask whose key axis is strided; Q, K, V or O, or a mask, with a batch,
  head or sequence (query) stride that is not a multiple of 8 elements; an f32 mask whose
  declared alignment is under 32 bytes; more than 65,535 query tiles or batches (the
  grid's y and z); a mask or sink on a padded tier (those ship plain); head dims that are
  not multiples of 8; anything past 32-bit offsets. The kernel also takes packed (varlen)
  batches and dropout, which this engine does not serve yet.
- **Which engine runs a graph** both this and §5's prefill engine serve is autotune's
  (UHD's) to decide: measured on gfx1201, this one is faster on MLA prefill and long
  sequences, §5's on decode and sinks.

### 5.7 Edge cases: the same answer from every family

Every FlyDSL SDPA engine -- RDNA prefill and decode, AOTriton's gfx1201 kernel, and the
CDNA generic, dual-wave and decode kernels (§7) -- gives the same result for the same
graph, matching PyTorch's `scaled_dot_product_attention` and hipDNN's CPU reference, so
swapping one family for another changes nothing. The GPU suites check these cases against
the reference on the families that serve them (a decode bias, for instance, exists only on
RDNA).

| Case | Result |
|---|---|
| A row whose every key a bias masks with −inf | O = 0, LSE = −inf |
| A row whose every key a bias masks with the dtype's lowest finite value (a padding mask) | Uniform over the keys the causal or window mask leaves: O = their mean of V |
| Some keys masked with either value | Softmax over the rest |
| A row no key reaches (causal with more queries than keys, a window) | O = 0, LSE = −inf; with a sink, LSE = the sink |
| A sink far above every score | Folded into the running max: LSE = the sink, O ≈ 0 |
| A split-KV decode | Splits weighed by their key counts, whatever the bias |
| A NaN score in a row | O and LSE NaN for that row |
| Scale | Positive and at most 2²⁰ in every family (§5.2) |

What differs, and why:

- **The LSE of a row no key reaches is −inf**, hipDNN's definition (its CPU reference and
  every kernel). PyTorch's own backends disagree (its CPU flash path writes 0), and cuDNN
  documents none, so there is no industry value to match.
- **NaN propagation is the hardware's, not a guarantee.** The CDNA kernels and AOTriton's
  softmax compile with no-NaN fast math; their "no key" tests are bit tests that keep a NaN
  row NaN (CDNA modification 29), but a NaN in the inputs is outside the contract, as it
  is for every engine. Which rows a NaN or infinite V element at a *masked* key reaches
  depends on each kernel's tile size; PyTorch makes every row NaN. Matching that would cost
  a select per P·V element, so it is not done.
- **RMSNorm** (§1, §2a): y and inv_rms differ by about 1 ulp between wave32 (RDNA) and
  wave64 (CDNA) parts, from the reduction tree's shape. dgamma is bitwise repeatable on a
  device, but its summation grouping follows the device's compute-unit count, so its last
  bits can differ between GPUs. Both are hardware differences.

## 6. The instance budget: what is baked, what is runtime, and why

Every feature is either a **runtime argument** (one object serves it, at whatever
cost it adds when unused) or a **compile-time axis** (more objects, each paying
only for what it computes). The count of objects is a cost too — every one is
built, reviewed, packed into every member shard and verified — so neither side
wins by default. The rule this provider follows:

1. **Default classification.** A per-launch scalar that changes only loop
   bounds, addresses or the epilogue is a runtime argument: lengths, head
   counts, strides, scale, mask offsets and bounds, `lse_on`. Anything that adds
   work per score in the inner loop or keeps extra state live in registers (an
   additive bias tile, softcap, ALiBi), or that changes the schedule (a
   `head_dim` 256 layout, a decode tile), is a compile-time axis.
2. **The gate.** A runtime feature stays runtime only if, with the feature
   unused, the shipped object matches a build with the feature compiled out: no
   spill, no lost occupancy, and time within the measurement noise on a fixed
   shape set (repeated, interleaved runs; a difference counts only when its sign
   repeats). One that fails becomes a compile-time axis or is restructured.
3. **Axes are model-determined, never request-determined.** `dtype`,
   `head_dim`, `causal`, a future `has_bias` — properties of the model a graph
   comes from. Never a sequence length, head count or window width: those stay
   runtime, so the count does not grow with the shapes a model is run at. Nor a
   feature's value -- epsilon, softmax scale, a bias, a sink, a mask bound or a
   dropout probability: a feature's presence may be an axis (`has_bias`,
   `has_sink`, `causal`), its value is always a runtime argument. The one
   request-determined exception is a fast path beside a generic object that
   serves every request, admitted by the kernel matcher only where it applies
   (CDNA `kv_align` 64, `any_diagonal` 0, §7.1); removing it changes speed,
   never coverage.
4. **An axis splits only what needs it.** A bias variant adds bias objects; it
   does not double the table. Specialized fast objects are added beside a
   generic one only where they measurably win, ranked by `priority` (the RMSNorm
   pattern, §1).

The measurements behind each decision are kept outside the repository; what is
recorded here is the decision and the gate it passed.

### 6.1 Runtime features audited against compiled-out builds

| Feature (SDPA) | Where its cost would be | Result | Decision |
|---|---|---|---|
| Bottom-right alignment, right-side bands, sliding windows | scalar setup, the KV loop's start; the per-score compare exists without them | ≤ 9 VGPRs, ~10 SGPRs; time within noise on every shape | runtime |
| Keyless-row guard (O = 0, LSE = −inf) | one select per KV step, one in the epilogue | 0 VGPRs; time within noise | runtime |
| LSE output (`lse_on`) | epilogue only | 0 VGPRs; time within noise | runtime |
| Separate V head group (`Hk ≠ Hv`) | one scalar divide for V's slice | 0 VGPRs, up to ~15 SGPRs; time within noise on every shape | runtime |
| Runtime head_dim | masked loads and stores in every K step | compile-time by the rule (a generic tier beside the specialized objects); the specialized objects' instructions are unchanged | baked axis, +8 |
| Additive bias | per-score loads and an add in the inner loop | compile-time by the rule (`has_bias`); its arguments are appended to every object's list, and the plain objects' instructions are unchanged | baked axis, +16 |
| Additive bias on the decode objects | per-score loads and an add in the inner loop | tried as a runtime argument (a uniform branch skipped with no bias bound): `head_dim` 64 slower with no bias, beyond noise with a repeating sign; testing it once and running one of two copies of the KV loop instead spills at 128 and 256, the register allocation being the larger copy's | baked axis, +16 |
| Row tiles on the decode objects (groups over 16 rows) | the row tile as the grid's second dimension; no work per score | 0 VGPRs; time within noise on every decode shape (computing the tile from the block index instead cost a few percent on short d64 decode, sign repeating). The bias objects are left one tile: theirs spill 2 VGPRs at `head_dim` 128 and up with it | runtime, not on the bias objects |
| Head-dim pairs (`hd_pair`) | a schedule, not a feature: the partial-score exchange through LDS, behind the barrier that already publishes V | compile-time by the rule; objects without it compile to identical code | baked, replaces `dv_split` above 128 |
| Idle-wave skip (rows past `seq_len_q`) | one branch per KV tile | removing it gains a few percent on small f16 non-causal prefill (`head_dim` 64) and nothing elsewhere, and roughly doubles every partly empty tile the decode objects do not take — K and V with different head counts, chunked prefill of fewer than 128 rows, a bias on a generic head_dim | kept, re-measured with the decode family in place |

### 6.2 Instance ledger

Objects per generic family; each family's set is packed into every member arch's
shard, so the shipped count per arch is the same as the authored count.

| Op | Axes (compile-time) | Objects | Status |
|---|---|---|---|
| RMSNorm forward | phase (2) × [gamma form (2) × dtype (2) × {generic N, ten baked widths}, plus generic-N f32] (§1) | 90 | shipped, `gfx11-generic` (and gfx90a, gfx942, gfx950) |
| RMSNorm backward | gamma form (2) × dtype × {three bounded generic tiers, twelve baked widths for 16-bit} (§2a) | 63 | shipped, `gfx11-generic` (and gfx90a, gfx942, gfx950) |
| SDPA forward | dtype (2) × head_dim {64, 96, 128} × causal (2) | 12 | shipped, `gfx11-generic` |
| SDPA forward, `head_dim` 256 | dtype (2) × causal (2); own schedule (Q re-read, `dv_split` 2) | 4 | shipped, `gfx11-generic`; the 12 above rebuilt byte-identical with it in the source |
| SDPA forward, additive f32 bias | one bias object beside each of the 16 above (`has_bias`) | 16 | shipped, `gfx11-generic`; the 16 plain objects compile to identical code with the bias arguments appended |
| SDPA forward, generic head_dim | dtype (2) × causal (2) × largest head {128, 256}; head_dim a runtime argument; no bias | 8 | shipped, `gfx11-generic`; the 32 above compile to identical code with the head_dim argument appended |
| SDPA forward, narrower generic tiers | dtype (2) × causal (2) × largest head {64, 96, 160}, for the odd head dims models use; exact 160/192 objects measured within a few percent of these tiers and not added | 12 | shipped, `gfx11-generic`; the 72 above rebuilt byte-identical with kernel modification 16 in the source |
| SDPA forward, generic tier up to 224 | dtype (2) × causal (2); two output-column tiles, where the 256 tier needs four; a separate up-to-192 tier measured only a few percent faster and was not added | 4 | shipped, `gfx11-generic`; the 84 above unchanged |
| SDPA decode | dtype (2) × head_dim {64, 96, 128} × causal (2); its own kernel plus merge kernel per object | 12 | shipped, `gfx11-generic`; the 40 prefill objects unchanged |
| SDPA decode, `head_dim` 256 | dtype (2) × causal (2); output columns split across two workgroups (`dv_split` 2) | 4 | shipped, `gfx11-generic`; the 52 above rebuilt byte-identical with it in the source |
| SDPA decode, additive f32 bias | one bias object beside each of the 16 above (`has_bias`) | 16 | shipped, `gfx11-generic`; the 16 plain decode objects compile to identical code with the bias arguments appended, the 40 prefill objects are unchanged |
| SDPA decode, generic head_dim | dtype (2) × causal (2) × largest head {96, 256}; head_dim a runtime argument (appended to both kernels), K/V columns past it never fetched; no bias | 8 | shipped, `gfx11-generic`; the 32 decode objects above compile to identical code with the head_dim argument appended, the prefill objects are unchanged |
| SDPA forward and decode, bias in the query's dtype | one object beside each of the 32 f32-bias objects above (`bias_q`), the bias read as bf16/f16 and widened to f32 (kernel modification 19) | 32 | shipped, `gfx11-generic`; the 108 above rebuilt byte-identical |
| SDPA forward, MLA (V narrower) | dtype (2) × causal (2); Q/K 192, V 128 (`head_dim_v`, kernel modification 20); the 192 schedule (head-dim pairs); no bias | 4 | shipped, `gfx11-generic`; the 140 above rebuilt byte-identical |
| SDPA decode, wide heads with one KV head | dtype (2) × causal (2) × {Q/K 576 with V 512 (absorbed MLA), 512/512}; the output columns split four ways (`dv_split` 4, 128 each, the d256 objects' register footprint); Q re-read per K-step; no bias | 8 | shipped, `gfx11-generic`; the 144 above rebuilt byte-identical |
| SDPA forward and decode, attention sink | dtype (2) × `head_dim` {64, 128} × causal (2) × {prefill, decode}; a per-head logit in the query's dtype (`has_sink`, kernel modification 21 and the decode kernel's own); no bias | 16 | shipped, `gfx11-generic`; the 152 above compile to identical instructions with the sink arguments appended |
| **Total, gfx11-generic** | | **309** | |
| SDPA prefill schedules (§5.1) | not an axis: `block_m` 64 for `head_dim` 64, non-causal 128 without a bias and the generic tier up to 64, and head-dim pairs (`hd_pair`) for 256 and the generic tiers above 128, replace the schedules of the same classes | ±0 | shipped; the 66 objects whose schedule did not change rebuilt byte-identical with kernel modification 18 in the source |
| **gfx12-generic** | the same table, built from the same sources for RDNA4 | **309** | shipped; GPU suites pass on gfx1201 |
| SDPA forward, AOTriton's gfx1201 kernel (§5.6) | dtype (2) × causal (2) × {64, 128, 256 each plain, with a mask in Q's dtype, with an f32 mask; sinks at 64 and 128; MLA 192/128; padded tiers 64, 96, 128, 160, 224, 256} | 72 | shipped, `gfx12-generic` only; GPU suites pass on gfx1201 |
| **Total, gfx12-generic** | | **381** | |
| **Total, both families** | | **690** | |
| SDPA forward, CDNA (§7) | per arch: dtype (2) × `head_dim` × causal (2), the kernel family fixed by arch and `head_dim`; the generic non-causal class also by `kv_align`, the dual-wave causal class by `any_diagonal`, the dual-wave classes by `has_bias`; companion tiers (§7.6); the MLA objects by `head_dim_v` and `qk_mask` | 100 on gfx950, 102 on gfx942, 102 on gfx90a | shipped; GPU suites pass on all three |

Adding a row here, with its gate result in §6.1 where it introduces a runtime
feature, is part of adding the feature.

---

## 7. SDPA forward on CDNA — `hipkernel:flydsl_sdpa_cdna`, `..._cdna_dualwave`, `..._cdna_decode`

Three engines, separate from §5's, one kernel family each: the generic MFMA kernel's
prefill objects (`hipkernel:flydsl_sdpa_cdna`, gfx90a, gfx942 and gfx950), gfx950's dual-wave
kernel (`hipkernel:flydsl_sdpa_cdna_dualwave`), and the generic kernel in its split-KV
mode with a merge kernel (`hipkernel:flydsl_sdpa_cdna_decode`). They share the graph
matcher, the checks and the dispatch (`FlydslSdpaCdnaCommon.hpp`), each for its own
family; an object's metadata carries only its family's knobs. Where this section speaks of
a class's "default" and "companions", ranking applies within one engine: the generic
kernel's tiles against each other, by descriptor `priority` plus a bonus where a tier's
measured rule prefers it (the engine's interim native ranker). That static order is
provisional: autotune times every object the engine admits for a graph and caches the
measured order, and a trained UHD ranker replaces the static one. Which engine runs a
graph several serve is the engine-level heuristic's (UHD), or autotune's; `priority`
never compares objects of different engines. The kernels are FlyDSL's own MFMA flash-attention forwards,
vendored whole under
[`kernels_src/kernels/attention_cdna/`](kernels_src/kernels/attention_cdna/README.md),
whose README lists the modifications (an import path, a runtime row guard in
the cooperative K/V loads that FlyDSL 0.3.4 cannot trace in a helper, a runtime
softmax scale, runtime head counts and token strides, a runtime LSE switch, a
runtime causal diagonal, runtime operand strides, and a per-(batch, head) bias view).
The native halves are
[`FlydslSdpaCdnaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaNative.cpp),
[`FlydslSdpaCdnaDualwaveNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaDualwaveNative.cpp)
and
[`FlydslSdpaCdnaDecodeNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaDecodeNative.cpp).

The objects are built **per arch**, not per generic family: CDNA has no generic
target that spans gfx942 and gfx950, and the two get different kernels. gfx90a (CDNA2)
takes the generic kernel's gfx942 path -- the same 64 KiB of LDS, DMA of at most four
bytes and the 32x32x8 MFMA (vendored-kernel modification 24) -- with tiles of its own
(§7.7).

| Kernel (`kernel`) | Workgroup | Grid |
|---|---|---|
| `dualwave` — `flash_attn_gfx950.py`, software-pipelined, two waves per 32 query rows, gfx950 only | 512 threads, `block_m` 256 | (query heads, query blocks, batch) |
| `generic` — `flash_attn_generic.py` | `block_m` × 2 threads (`block_m` 64, 128 or 256) | batch × query tiles × query heads, flattened |

Which objects serve each class was chosen by measurement (§7.6). Per arch, in bf16
and f16, the default tier first:

| Arch | `head_dim` | Non-causal | Causal |
|---|---|---|---|
| gfx950 | 64 | `dualwave` (setprio off); `generic` 64 | `dualwave`; `generic` 64, preferred for small batch × heads × query rows |
| gfx950 | 128 | `dualwave` (setprio off); `generic` 64 | `dualwave` (+ offset-0 build); `generic` 64, preferred below 256 query rows |
| gfx950 | 256 | `generic` 128; `generic` 64 (each + `kv_align` 64), one wave per SIMD | `generic` 128; `generic` 64, preferred below 256 query rows |
| gfx942 | 64 | `generic` 128 (three waves per EU); `generic` 64, preferred for small problems (each + `kv_align` 64) | the same, without `kv_align` |
| gfx942 | 128 | `generic` 128; `generic` 64 for small problems, 256 for large ones (each + `kv_align` 64) | the same, the 128 tile without KV global prefetch |
| gfx942 | 256 | `generic` 128; `generic` 64 for small problems (each + `kv_align` 64), LDS tiles unpadded | the same, without `kv_align` |
| gfx90a | 64 | `generic` 128 (+ `kv_align` 64); `generic` 128, one wave per SIMD | `generic` 128, one wave per SIMD |
| gfx90a | 128 | `generic` 256, three waves per SIMD (+ `kv_align` 64) | `generic` 256, three waves per SIMD, without KV global prefetch; `generic` 128, one wave per SIMD |
| gfx90a | 256 | `generic` 128, three waves per SIMD, LDS tiles unpadded | `generic` 128, one wave per SIMD, LDS tiles unpadded |
| gfx950 bias | 64, 128 | `dualwave`; `generic` 64 for what it declines (a window, a strided LSE) | `dualwave` (+ offset-0 build at 128); `generic` 64 (128 at 128, one wave per SIMD) for what it declines |
| gfx950 bias | 256 | `generic` 128, one wave per SIMD | the same |
| gfx942, gfx90a bias | 64, 128, 256 | `generic` 128 | the same (at 128 without KV global prefetch) |
| gfx950 MLA | Q/K up to 256 (multiples of 8), V 128 | `generic` 128 with a Q/K mask (`qk_mask`, `head_dim_v` 128), one wave per SIMD | the same |
| gfx942, gfx90a MLA | Q/K up to 256 (multiples of 8), V 128 | `generic` 128 with a Q/K mask | the same |
| gfx950 absorbed MLA, decode | Q/K up to 1024 (576 in practice; multiples of 8), V 512; built for absorbed MLA's one KV head, but any GQA group is admitted | `generic` 64 with a Q/K mask, O split into four column slices, the keys split and the group packed | the same |
| gfx942, gfx90a absorbed MLA, decode | Q/K up to 768 (multiples of 8), V 512, as above | the same, K staged in three 256-column chunks per KV tile (`k_chunk`), without the KV global prefetch | the same |
| every arch, any other multiple of 8 | up to 64, 128, 256 (Q, K and V the graph's width) | `generic` with Q/K and V masked to the runtime width (`qk_mask`, `head_dim_v` = the bucket), `priority` below every exact tier (76, 74, 70), the narrower bucket above the wider | the same |
| gfx90a, the widths that are not powers of two | up to 96, 160, 192, 224 (modification 26) | `generic` buckets at those widths, tiles from an MI210 sweep (96: 256 rows; 160: 256 rows; 192, 224: 128 rows, the 256-row tiles spilling 22-46 VGPRs); causal 96 and 160 with measured companions | the same |
| every arch, attention sink | 64, 128 | `generic` (gfx950 64 rows; gfx942 and gfx90a gfx942's default 128-row tier) with the sink (`has_sink`), and the split, packed decode tier with it | the same |
| every arch, decode | 64, 128, 256 | `generic` 64 (128 at 256) with the keys split across workgroups and the GQA group packed into its rows (`split_kv`, `pack_gqa`), an engine of its own (`hipkernel:flydsl_sdpa_cdna_decode`) that serves any query length; which engine runs a graph is autotune's (UHD's) | the same |

The dual-wave kernel takes `head_dim` 64 and 128 only, so gfx950's 256 comes from the
generic kernel, which at that width holds its tile without spilling only with the
whole register file per wave. The generic kernel's exact objects take powers of two; its bucket objects serve the other head dims. At 256 its
padded K and transposed-V tiles need 67,072 bytes of LDS, past gfx942's 64 KiB, so gfx942's
objects are built with both unpadded -- exactly 64 KiB (vendored-kernel modification 9);
K keeps its XOR swizzle against bank conflicts.

The un-absorbed MLA objects (a 256 bucket with `head_dim_v` 128, `priority` 100) admit
any Q/K up to 256 with V 128, a plain `head_dim` 128 graph included. Under the static
order they tie the exact `head_dim` 128 default tiers (100, the tie falling to the kernel
id) and outrank the companions (80), so on gfx950, whose generic d128 tiers are all
companions, they come first in `hipkernel:flydsl_sdpa_cdna` for most d128 graphs without
a bias or sink (those outside the 64-row companion's preferred range). They
compute those graphs correctly, at the 256 bucket's cost (§7.5, item 10); the measured
order autotune caches, and UHD after it, is what picks among the candidates.

### 7.1 What is baked, and what is not

The set started as the kernels exactly as upstream builds them, and widens one
baked field at a time (§7.3).

| Baked per object | Runtime kernel argument |
|---|---|
| dtype (`bf16`, `f16`) | batch |
| `head_dim` (64, 128, 256), or a bucket (64, 128, 256; on gfx90a also 96, 160, 192, 224; 1024 or 768 for absorbed MLA) whose Q/K width is runtime (`qk_mask`) | query and key lengths, independently; the Q/K head dim under `qk_mask` |
| V's head dim (`head_dim_v`: `head_dim`'s, 128 or 512 on the MLA objects, or following Q/K's on a bucket object) | V's strides, apart from K's (read through them on the `qk_mask` and MLA objects) |
| causal, or none | the causal diagonal: top-left, bottom-right, a band to its right; a sliding window's lower edge (generic kernel) |
| | query and KV head counts (MHA, GQA, MQA) |
| | every operand's (batch, sequence, head) strides: BSHD, BHSD, packed QKV |
| | softmax scale |
| | whether to write the LSE output, and its (batch, head, query) strides (generic kernel) |
| the schedule: kernel, `block_m`, waves per EU, setprio, KV global prefetch -- a class's tiers (§7.6) | which tier serves a graph, by its size (the score) or by autotune |
| generic non-causal: whether the KV tail is masked (`kv_align`) | |
| gfx950 d128 causal: whether any diagonal is served (`any_diagonal`) | |
| whether an additive bias is added (`has_bias`) | the bias's (batch, head, query) strides |
| whether an attention sink is added (`has_sink`) | the sink's head stride |
| decode: keys split, the GQA group packed into rows (`split_kv`, `pack_gqa`); O's columns split (`dv_split`); K staged in column chunks (`k_chunk`) | how many splits |

One hundred objects on gfx950, one hundred and two on gfx942 and on gfx90a. A class -- (dtype, `head_dim`, causal,
bias) -- ships a default tier that serves every shape of it, and measured companions
for the shapes where another schedule wins (§7.6). Two variants specialise a tier
further and rank above it where they apply. The generic kernel can skip masking the
last KV tile when the key length is a multiple of its 64-row KV tile, which is
measurably faster; so a non-causal generic tier may ship twice -- `kv_align` 0,
masked, for any length, and `kv_align` 64, unmasked, admitted by the kernel matcher
only for such lengths. On gfx950 the dual-wave causal kernel serves an arbitrary
diagonal only when built with upstream's `cross_seqlen`, which at `head_dim` 128
costs a few percent; so that tier ships twice too -- `any_diagonal` 1 for every
diagonal, and `any_diagonal` 0, the plain build, admitted only at offset 0: causal
self-attention, and top-left causal with fewer queries than keys. At `head_dim` 64
the two measured the same, and only `any_diagonal` 1 ships.

Each tensor is passed as FlyDSL's
`fx.Tensor`: an 8-byte pointer followed by a 40-byte by-value layout (four i32
extents, then the i64 strides of the three non-unit-stride dimensions);
`gen_sdpa_cdna.py` asserts every object's argument list against its family's
before writing it, and the dispatcher marshals by that position list.

### 7.2 What a graph must be

One SDPA-forward node, and:

| Condition | Why |
|---|---|
| Q, K, V, O rank 4, device operands, **not ragged**, each declaring 16-byte alignment; head-dim stride 1; every other stride on an axis longer than 1 positive and a multiple of 8 elements; one (batch, head) slice of each within 32-bit addressing | The batch, sequence and head strides are kernel arguments, so BSHD, BHSD and packed-QKV views are served; each lane moves 16-byte vectors; each slice is one buffer view, whose record count is 32-bit. V with strides other than K's is served by the `qk_mask` and MLA objects, whose masked load path reads V through its own; the generic kernel's direct-to-LDS path and the dual-wave kernel read V through K's descriptor, so those objects decline it. |
| One dtype for all four, `bf16` or `f16` | One object per dtype. |
| `Hq % Hk == 0`; V shaped as K but for a head dim at most K's, O as Q with V's head dim | The kernels do not broadcast; V's head dim is its own only on the MLA objects (modification 13). |
| `head_dim` equal to an object's baked value, or, on a `qk_mask` object, a multiple of 8 up to its bucket; V's head dim equal to the object's `head_dim_v`, or Q/K's on a bucket object | Checked by the kernel matcher against each candidate's metadata. A masked K or V load zeroes whole 8-column halves. |
| No mask, or `right_bound` ≥ 0 at either named `diagonal_alignment` (or one of the deprecated causal booleans: right 0 unless the graph sets one, `causal_mask` at the stored alignment, `causal_mask_bottom_right` always bottom-right; an unnamed alignment declines either way), with a diagonal offset `align + right_bound` of at most `Skv − Sq`; a `left_bound` only with a right one | The kernels apply the reference's rule, key `j` visible from query row `i` iff `i + align − left ≤ j ≤ i + offset`, with `align` 0 top-left and `Skv − Sq` bottom-right. The causal objects mask the last KV tile's padding with the same compare, which holds only while every row's diagonal stays inside the keys: so bottom-right is served at any lengths, top-left when `Sq ≤ Skv`, and a band only as far as `Skv − Sq`. A sliding window (`left_bound`) runs on the generic kernel only (modification 11) -- on gfx950 its causal tiers beside the dual-wave objects; a window no row reaches is no window. |
| Scale: `attn_scale_value` (> 0 and ≤ 2²⁰), a pass-by-value scalar tensor, or absent | Absent is 1.0, cuDNN's default, which hipDNN's references apply too. The same range as the RDNA engines (`FlydslSdpaShared.hpp` `MAX_SOFTMAX_SCALE`): positive, because the running max is over unscaled scores, and at most 2²⁰, below where the causal objects' running-max floor overflows f32. A pass-by-value scale is read from host memory at launch, and one outside the range is refused there. |
| Key length a multiple of 64, for a `kv_align` 64 object | That object does not mask the last KV tile. |
| Stats, if requested: a named f32 `[B, H, Sq, 1]` device tensor, each axis longer than 1 at a positive stride, every element within 32-bit offsets | The kernels write the natural-log LSE of the scaled scores. The generic kernel and its merge take its strides (modification 20); the dual-wave kernel writes one packed `[B, H, Sq]`, so a strided LSE runs on gfx950's generic companions. A row no key reaches gets −inf (or the sink); a row whose softmax sum is NaN (a NaN or +inf score) gets NaN, as PyTorch's does, and a decode split left NaN carries NaN into the merged O and LSE rather than being skipped (modification 29). |
| Bias (`attn_mask`), if present: in the query's dtype, declaring 16-byte alignment, rank 1–4 right-aligned to `[B, H, Sq, Skv]`, any of B, H and Sq broadcast, the key axis contiguous; strides multiples of 8 elements, a query stride of at least `Skv` (or 0); a key length that is a multiple of 4; one (batch, head) slice within 32-bit addressing | Added to the scaled scores, as PyTorch adds a float `attn_mask`, which it requires in the query's dtype. Both kernels add it (the generic one by modification 12); rows move as 16-byte vectors; each slice is one buffer view, and the kernels read keys in runs of 4, so a run straddling the view's end could read as zero. A bias in another dtype, broadcast along the keys, or misaligned (a key length not a multiple of 8 in a packed bias, or not a multiple of 4 at all) declines. A bias build softmaxes over `q·k · scale + bias`, as PyTorch, hipDNN's CPU reference and the RDNA kernels do (modification 28): a bias that masks whole KV tiles with −inf is served, a row it masks with −inf on every key gets O = 0 and LSE −inf, and a row masked on every key with the dtype's lowest *finite* value averages V uniformly over the keys the causal mask leaves. |
| Attention sink, if present: one logit per query head in the query's dtype, rank 1 `[Hq]` or rank 4 `[1, Hq, 1, 1]`, a device operand, its head stride positive (unless one head) with `(Hq − 1) × stride` within int32, at `head_dim` 64 or 128, without a bias | Served by the generic kernel's `has_sink` objects (modification 19), which are built without a bias, so a graph with both declines; the dual-wave kernel builds none. The sink joins against the max over the row's scores and itself, prefill and decode merge alike, so a sink far above the scores gives O ≈ 0 and LSE = the sink, as PyTorch and the RDNA kernels do (modification 29). |
| Declined outright: padding / varlen, dropout, paged KV, block masks, FP8 descales, ALiBi, the split max / sum-exp stats, a non-`AUTO` implementation | The objects are built without them. |
| Batch pitch and grid within int32 | 32-bit addressing. |

### 7.3 What comes next, and the rule it follows

Each feature that widens this set is measured by §6's gate against the object it
replaces. While the set is being widened, coverage comes first: a small measured
cost may be accepted to serve a feature, but it is recorded in §7.5 as deferred
performance work rather than passed silently, and a significant one is weighed
against the coverage it buys. In order:

| | What it takes |
|---|---|
| ~~Runtime scale~~ | Done: a trailing f32 argument on both kernels (modification 2). |
| ~~Runtime head counts~~ | Done: trailing i32 arguments on both kernels, and the generic kernel's token pitches with them (modification 3). |
| ~~Full strides: BHSD, packed QKV~~ | Done: per-(batch, head) buffer views, the head's offset in each view's base (modification 6). V with its own strides runs on the `qk_mask` and MLA objects (§7.2). |
| ~~LSE output~~ | Done: compiled in, switched at launch (modification 4); any strides on the generic kernel (modification 20). |
| ~~`Sq ≠ Skv`, both causal alignments, right-side bands~~ | Done: the diagonal offset is a kernel argument (modification 5). Top-left with more queries than keys and wider bands need the last KV tile's padding masked apart from the diagonal. |
| ~~Sliding windows~~ | Done on the generic kernel: a runtime lower edge in the mask and the KV loop's start (modification 11). The dual-wave kernel's pipeline starts at a fixed tile; gfx950 windows run on its generic tiers. |
| ~~`head_dim` 64 and 256, f16~~ | Done: rows in the instance table. Other head dims need the generic kernel's row loads generalised past powers of two. 256 on gfx942 followed with unpadded LDS tiles (modification 9). |
| Varlen, paged KV | Both kernels already take cumulative sequence offsets and a block table, passed unused today. |
| ~~Additive bias~~ | Done: the dual-wave kernel's at `head_dim` 64 and 128 on gfx950 (modification 8), and the generic kernel's (modification 12) at 64, 128 and 256 on gfx90a, gfx942 and gfx950; a bias that masks whole tiles or rows gives what PyTorch gives (modification 28). f32 bias and misaligned rows need a conversion or unaligned path. |
| ~~Attention sinks~~ | Done on the generic kernel at head dims 64 and 128 (modification 19), a baked axis like the bias: a per-query-head logit in the query's dtype, rank 1 `[Hq]` or rank 4 `[1, Hq, 1, 1]`, in the softmax's denominator and the LSE, taken against the max over the scores and the sink (modification 29). hipDNN's CPU reference declines sinks; the GPU tests check them against their own reference. |
| ~~Other head dims~~ | Done for every multiple of 8 up to 256, as on RDNA (modification 17): bucket objects at 64, 128 and 256 (and on gfx90a also 96, 160, 192 and 224, modification 26) whose Q, K and V follow the graph's head dim, the K and V loads bounded in 8-column halves. |
| ~~Decode: KV split and GQA packing~~ | Done on the generic kernel (modifications 14, 15): one workgroup per (batch, row tile, KV head) with the group's query heads as rows, the keys split over the grid's y by a count the engine picks from the shape and the device's compute units, and a merge kernel in the same object. Without a bias. |
| ~~V head dim ≠ Q/K (MLA prefill)~~ | Done on the generic kernel (modification 13): V's own strides, a baked `head_dim_v`, and a runtime Q/K width masked to zero past it within a 256 bucket. Absorbed MLA (576/512, one KV head) and 512/512 ship on gfx950 (modification 16: O split by columns, Q streamed). gfx942 takes them with K staged in 256-column chunks, three passes per KV tile (modification 18). |

The instance set is then chosen by the provider's AOT selection process (build
every candidate, tune, cover, prune) rather than by hand -- done for the features
above (§7.6), and repeated as features land.

### 7.4 Where each claim is checked

| Claim | Checked by |
|---|---|
| The objects are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 |
| Each object has its family's argument layout, workgroup size and arch, fits the arch's LDS, and spills no registers but where a class's spill is explicitly allowed and recorded (`vgpr_spills`) | `_sdpa_cdna_build.py` (all three generators), before the object is written |
| The vendored kernels match FlyDSL but for their recorded modifications | `tools/diff_upstream.py` ([REGEN.md](REGEN.md) §5) |
| Each engine's shard ships no two objects of one build, one top-ranked serve-all object per graph class, and each object with its family's signature | `TestFlydslSdpaCdnaPacks`, `TestFlydslSdpaCdnaDualwavePacks`, `TestFlydslSdpaCdnaDecodePacks`, and the shard census that runs them |
| A generic-kernel companion outranks its class's default exactly where its rule prefers it | `TestFlydslSdpaCdnaKernelMatch` |
| The accept/decline rules above, the causal and scale bindings, and candidate selection, including `kv_align` | `TestFlydslSdpaCdnaGraphAccepts`, `TestFlydslSdpaCdnaGraphDeclines`, `TestFlydslSdpaCdnaBinding`, `TestFlydslSdpaCdnaKernelMatch` |
| Every staged object the matcher admits computes attention — non-causal and causal, ragged tails, a longer sequence, a scale value, a runtime scale tensor, MHA, MQA and an odd group size, cross-attention, chunked prefill, decode, rows no key reaches, a top-left band, `head_dim` 64 and 256, f16, BHSD and packed QKV, a per-head bias and a padding mask — and the LSE output, against a reference | `TestGpuFlydslSdpaCdnaDispatch` (and `Full/TestGpuFlydslSdpaCdnaFullTier`, below), on gfx950, gfx942 and gfx90a |

Every candidate a shape admits runs against one set of inputs and one f64 reference, computed
once per case. The sweeps and large shapes are `Full/TestGpuFlydslSdpaCdnaFullTier`, full tier
only (§4's tier note): every width inside the non-power-of-two buckets, DeepSeek-V4's head dim
512 at prefill lengths, the lowest-finite bias rows in every combination, the combinations no
other case reaches, and the 1024-long sequence. The quick tier keeps one configuration of each
behavior, so the unit binary fits its 300 s quick-tier budget on a CDNA runner.

### 7.5 Measured costs, and deferred performance work

Each change measured against the object it replaced on the same inputs: 15 to 21
interleaved rounds per shape (bf16, 32/8 heads, `head_dim` 128, batch 1 to 8,
lengths 1024 to 8192), with an identical copy of the baseline as a control for
the harness's own bias. The measurements are kept outside the repository.

| Change | Result | Decision |
|---|---|---|
| gfx942 non-causal: mask the KV tail (a correctness fix) | the masked object is measurably slower at lengths a multiple of 64, on every shape | kept, plus the unmasked object beside it for those lengths (`kv_align` 64) |
| Runtime softmax scale, gfx950 | same registers and instruction count; slower by a fraction of a percent at mid-size shapes, sign repeating | runtime, for coverage — **deferred** |
| Runtime softmax scale, gfx942 | 2–4 more registers (the scale is splatted into a VGPR pair for packed multiplies; the `kv_align` 64 object moves 4 values into AGPRs), a few more instructions; slower by a fraction of a percent, sign repeating on the long shapes | runtime, for coverage — **deferred** |
| Runtime head counts, both kernels (and the generic kernel's token pitches) | same VGPRs; a 32-bit divide or two per workgroup and, on gfx942, runtime token pitches in the KV loads. Slower by under a percent on most shapes and up to a couple of percent on the shortest; the gfx950 causal object also pays on long shapes, so part of the cost is in how the loop is scheduled, not only the divides. Done in 64-bit arithmetic it cost more | runtime, for coverage — **deferred** |
| LSE output, switched at runtime | 2–4 more VGPRs, no occupancy step; with LSE off, no measurable cost but on the gfx942 causal object, which is slower by a fraction of a percent on every shape | runtime, for coverage — **deferred** (gfx942 causal) |
| Causal diagonal as an argument, and `cross_seqlen` on gfx950 | neutral on gfx942. On gfx950 the `cross_seqlen` build is a few percent slower on every causal shape; the plain build at offset 0 (`any_diagonal` 0) recovers most of it, still slower by under a percent than computing the offset in-kernel | two gfx950 causal objects; the residual cost **deferred** |
| Runtime operand strides (per-(batch, head) views) | neutral on the gfx950 dual-wave objects. On gfx942 the d64 objects grew past the three-waves-per-SIMD register step, measurably slower; LLVM's register-pressure trackers bring them back under it and recover the time, but on the d128 objects the same option schedules the loop much worse, so it applies at d64 only. The remaining cost -- gfx942 d128, gfx950 d256 -- is under a few percent | runtime, for coverage; trackers at gfx942 d64; the rest **deferred** |
| Sliding window on the generic kernel (`window_lo`, modification 11) | gfx942, against the objects without it, no window bound: a few more VGPRs on d64; the causal d128 object slower by about half a percent with a repeating sign (two compares on each diagonal tile's mask), d64 within noise. With a window, time scales with the window | runtime, for coverage -- **deferred** (gfx942 causal d128) |
| Additive bias (`has_bias`, a compile-time axis) | the plain dual-wave objects carry two unused bias-stride arguments, neutral but for a fraction of a percent on two d64 shapes. A bias object costs what reading its bias costs -- a full `[B, H, S, S]` bias is as many bytes as the scores. At `head_dim` 128 the bias objects spill a few VGPRs (upstream's own bias addressing spills too), with no cost visible beyond the bias's own | two objects per class on gfx950; the d128 spill allowed and recorded |
| LSE strides on the generic kernel and its merge (modification 20) | against the objects before it, LSE on and off: within noise on gfx942 and on gfx950 at `head_dim` 64 and 128, prefill and decode; the merge's strided LSE write costs the split decode objects a fraction of a percent with LSE on; the gfx950 `head_dim` 256 non-causal object is slower by about half a percent on every shape, LSE on or off, from how the changed prologue reschedules its loop | runtime, for coverage -- **deferred** (gfx950 d256 non-causal) |
| Scaled-domain bias and NaN-safe row handling (modifications 28 and 29; the same edge-case results as the RDNA kernels, §5.7) | objects without a bias within about a percent. Bias objects: the causal ones on gfx950 and gfx942 slower by 2.5-5%, the gfx950 `head_dim` 128 non-causal one by about 2% (its exponent no longer fuses), the gfx90a `head_dim` 64 causal 128-row object by 9-13% on three of four shapes; on gfx950 the `head_dim` 64 bias objects by up to about 6% (generic causal at a decode shape, dual-wave causal), 3% (dual-wave non-causal); the gfx950 dual-wave `head_dim` 128 causal bias objects got faster (8-17%). The aiw `head_dim` 128 float-mask objects (aiw modification 9) slower by 3-4% | kept, for cross-family parity -- **deferred** |

**Deferred performance work** -- costs accepted for coverage, to revisit:

1. *Runtime scale, both kernels.* Options: a baked-`1/√head_dim` object beside
   each runtime-scale one (the common scale at full speed), or a formulation that
   keeps the scale out of the packed multiplies on gfx942.
2. *Runtime head counts.* Divide by the head counts with host-computed
   multiply-and-shift constants instead of a divide sequence; find what moves
   in the gfx950 causal object's loop; or ship baked-head objects for the most
   common head configurations beside the runtime ones.
3. *LSE on the gfx942 causal object.* A second, LSE-free causal object, or
   moving the LSE math under the store's own predicate.
4. *The diagonal argument on the gfx950 offset-0 object.* It reads an offset the
   kernel used to derive; an offset-0 build could fold it to a constant.
5. *Runtime strides on the generic kernel.* gfx942 d128 and gfx950 d256 pay a
   little for per-head addressing; a contiguous-BSHD build beside the strided one,
   or a schedule that recovers it.
6. *gfx942 `head_dim` 256.* Correct on every corpus shape, but its throughput is well
   below gfx942's `head_dim` 128 -- further below than gfx950's 256 is below its 128.
   Its tile runs one wave per SIMD with all 64 KiB of LDS and unpadded K/V rows, and
   the unpadded transposed-V tile is most of it: measured at `head_dim` 128, where both
   fit, dropping V's padding alone makes the object roughly half as fast (bank
   conflicts on the V^T reads), dropping K's a few percent. An XOR swizzle on V^T in
   place of the padding (as K has), a smaller KV tile, or a head-dim split as the RDNA
   kernel does at 256 are the candidates. Nothing else in hipDNN serves this class on
   gfx942, so it ships as it is.
7. *The generic kernel's bias loads.* They are issued in the score path with no
   prefetch, so their latency is exposed: a bias object pays well beyond the bias's own
   bytes, even for a broadcast padding mask. Prefetching the next tile's bias with its
   K tile, or staging it through LDS as the dual-wave kernel does, would hide it.
8. *The window compares on the generic causal objects.* Every diagonal tile's mask does
   a lower-edge compare too, and `window_lo` is read whether or not a window is bound;
   a window-free causal build beside the windowed one would remove it. Also: gfx950
   windows run on the generic kernel at about half the dual-wave kernel's speed --
   a window start in the dual-wave pipeline would recover it.
9. *V's own strides on the generic objects* (modification 13). Every generic object
   now addresses V through its own batch, head and token strides. Measured against the
   objects before it: neutral on the gfx942 tiers but the `head_dim` 128 causal one
   without KV global prefetch (a fraction of a percent slower, consistently); on gfx950
   the `head_dim` 256 non-causal default got slightly faster and the 64-row companions
   slightly slower (up to about a percent). A shared-stride build for the graphs whose
   K and V match (all but MLA) would remove it.
10. *The MLA objects stage V at the Q/K bucket's width.* V is loaded and transposed
    256 columns wide where 128 are used, and Q/K at 192 run a 256-wide QK product. A
    V-width load mapping and a 192-wide build (a new K swizzle and row mapping, since
    192 is not a power of two) would recover both, once a model's prefill time
    justifies it.
11. *The absorbed-MLA objects (gfx950).* Built for a 1024-wide Q/K bucket where 576 is
    used: the bucket is the narrowest the cooperative K load maps onto the workgroup, so
    the QK product and the K tile's LDS traffic run 1024 wide. Q is re-read from global
    at each K-step with no prefetch. A 576-exact load mapping (rows that do not divide
    the threads, as the RDNA kernel's modification 16) and a one-step Q prefetch would
    recover both. gfx942's objects (modification 18) run a 768 bucket in three K passes, the third mostly masked.
12. *The softmax scale on the dual-wave kernel's f32 scores* (modification 21). Upstream
    rounded `Q * sm_scale * log2e` to bf16/f16 before the QK product, which put the LSE
    off by up to about 5e-3 and O off by a few ulps at larger scales. Applying the scale
    inside the softmax's max-subtraction FMA instead fixes both, but the backend packs
    only about half those FMAs into `v_pk_fma_f32` and needs 4 to 16 more VGPRs. Measured
    against upstream's objects: `head_dim` 128 within 0-2%, 64 within 1-4% (5.8% at
    one long non-causal shape); with a bias, `head_dim` 128 non-causal within 1% and
    causal 7-62% slower, its spill growing from 4 to 13 VGPRs. The generic kernel's bias
    objects are slower still on every one of those shapes, so the dual-wave ones stay.
    Getting every FMA packed, and the causal bias build back under the register file,
    would recover it.
13. *The scaled-domain bias* (modifications 28 and 29). Adding the bias after the scale,
    and masking with the lowest finite value so fast math cannot fold the mask away, costs
    the generic bias objects an unfused exponent and a few register moves per tile: a few
    percent on the causal objects, most on gfx90a `head_dim` 64 causal (9-13%). A form that
    keeps the bias inside one FMA without changing the results, or a gfx90a tile retuned
    for the bias build, are the candidates.
14. *gfx950 `head_dim` 128 causal prefill objects at a single query row.* The plain and sink
    64-row objects are about 12% slower at one query row against 2048 keys than when the
    RDNA and CDNA matchers each resolved the mask themselves, before `FlydslSdpaShared.hpp`;
    the same objects run that much slower under the current host code, so the cause is in
    the arguments or launch the host gives them, and is not yet found.
    The split decode object serves that shape several times faster, so it is not the one
    autotune picks there.

### 7.6 How the instance set was chosen

By the provider's AOT selection process, with the measurements kept outside the
repository:

1. **Reduce the knobs.** Everything the graph fixes is runtime (§7.1). What is
   left is the schedule: the kernel; the generic kernel's `block_m` (64, 128, 256),
   path (N32 / N128), waves per EU and its build-time switches (DMA, three-buffer
   prefetch, KV global prefetch, QK prefetch depth, K padding, 16-byte LDS vectors,
   the reduction); the dual-wave kernel's lazy rescale, setprio, stagger and waves
   per EU.
2. **Build every candidate** through `gen_sdpa_cdna.py`'s own build and checks,
   one factor at a time around each class's previous default, bf16.
3. **Time every candidate on every shape** of its class, on gfx950 (MI355X) and
   gfx942 (MI325X): the forward SDPA graphs of the dnn-benchmarking workloads
   (model configurations from its attention, cuDNN-attention, PyTorch, rocKE and
   HipKittens sets; AITER's and AOTriton's sweeps weighted down as synthetic) that
   the matcher admits -- about nine hundred distinct shapes on gfx950 and eight
   hundred on gfx942 -- each output checked against a reference.
4. **Cover and prune.** The knobs that won anywhere were crossed and timed again,
   with the `kv_align` and `any_diagonal` variants; per class, a greedy cover
   counting "best" within noise, then sets of one to three tiers compared by how
   far their best object, and the object a one-feature size rule picks, fall
   behind the best candidate on each shape.
5. **Ship the set**, its rules as descriptor metadata (`prefer_feature`,
   `prefer_min`, `prefer_max`), read by the score.

What came out of it:

- The dual-wave kernel's 256-row tile is mostly idle on short queries -- decode,
  small prefill -- where the generic kernel's 64-row tile is several times
  faster at the extreme. It is each gfx950 class's companion; a size rule picks
  it for causal graphs, and autotune finds it for non-causal decode, where no
  one-feature rule separated the two well.
- On gfx942 the 128-row tile alone left large problems well behind the 256-row
  one (upstream's launcher switches between them by size) and small ones behind
  the 64-row one: the head_dim 128 classes ship all three.
- Small, free gains kept: setprio off on the non-causal dual-wave objects, three
  waves per EU at gfx942 head_dim 64, and KV global prefetch off on gfx942's causal
  head_dim 128 tile.
- Kept specialisations: `kv_align` 64 is worth a separate object wherever the
  generic kernel serves non-causal graphs at a 128-row tile or larger; the
  `any_diagonal` 0 build only at gfx950 `head_dim` 128.
- Rejected: the generic kernel's `ds_bpermute` reduction and, on gfx942 at
  `head_dim` 64, KV global prefetch off (neither builds); three-buffer prefetch
  (over the LDS limit at the larger head dims, slower elsewhere); and its DMA paths
  -- N128 on gfx950, which turns DMA on, and gfx942's DMA switch -- which produce
  wrong results, so no object is built with either.
- Accepted spills, each the winner on the shapes its tier is for: the generic
  64-row tile at gfx950 `head_dim` 128 (causal) and the 256-row tile at gfx942
  `head_dim` 128; recorded per object and allowed by name in `gen_sdpa_cdna.py`.

The rules were fitted on bf16 and the f16 objects share them. Bias classes are
rare in the corpus and ship their default tier only. The process is repeated as
features land; the rules are coarse by design, and a trained heuristic replaces
them when the ingestor gains one.

### 7.7 gfx90a (MI210, MI250)

gfx90a runs the generic kernel on gfx942's code path (vendored-kernel modification 24)
and ships what gfx942 ships but the dual-wave engine: `hipkernel:flydsl_sdpa_cdna` (78
objects, with the 96/160/192/224 buckets below) and `hipkernel:flydsl_sdpa_cdna_decode` (24).

- **The plain prefill classes** (bf16 and f16; `head_dim` 64, 128, 256; causal or not)
  were chosen by a sweep on an MI210 (104 CUs): `block_m` 64/128/256 × waves per SIMD
  1/2/3 × KV global prefetch on/off, plus `kv_align` 64 and K padding off, each checked
  against an fp32 reference at a ragged length and timed on six shapes from 16 × 512 to
  32/8-head GQA at 8192. Per class, a greedy cover over both dtypes' shapes keeps the
  fewest tiles within 3% of the best measured on each -- one, or two at `head_dim` 64
  non-causal and 128 causal. No tier carries a `prefer` rule: until UHD's ranker exists,
  autotune picks among them.
- **Bias, sink, MLA, decode and absorbed-MLA classes, and the 64/128/256 buckets** take
  gfx942's tiers, which fit the same LDS, unmeasured on gfx90a beyond the GPU suites.
- **The 96, 160, 192 and 224 buckets** (vendored-kernel modification 26) were chosen by the
  same sweep at their own widths (0 wrong of 336 configs), over tiles that spill no VGPR
  in either dtype. Against PyTorch's flash path on MHA shapes: 96 0.64-0.77x, 160
  0.70-0.81x, 192 0.60-0.75x, 224 0.82-0.85x, up from 0.4-0.5x (13-18 TFLOP/s) on the 256
  bucket; 192 and 224 reach 64-70 TFLOP/s on long sequences. The 256-row tiles there
  were ~20% faster still but spill 22-46 VGPRs, past every spill allowance shipped.
- **Measured** (fp16/bf16, long MHA 2 × 16 heads × 8192): `head_dim` 128 non-causal
  114 TFLOP/s -- 63% of the 181 TFLOP/s peak, 86% of the best bf16 GEMM measured on
  the same GPU (133) and 1.10x PyTorch's flash path (AOTriton); `head_dim` 64 102
  TFLOP/s (1.10x); causal 82-96 TFLOP/s (0.92-1.08x).
- **Through hipDNN, autotuned** (the dnn-benchmarking attention graphs FlyDSL serves on
  gfx942, on an MI210): of the 2,243 measured, 2,242 served and none wrong (the rest
  were the largest decode graphs, past the test node's memory); against
  PyTorch's flash path on equal-head graphs 1.02x overall and fastest on 62%, `head_dim`
  64 1.38x, 128 1.15x, Q/K 192 with V 128 2.11x; on the large prefill graphs `head_dim`
  128 runs at 117 TFLOP/s median (125 best: 69% of peak, 94% of the measured GEMM).
- **Behind AOTriton, deferred performance work:** `head_dim` 256 (39-44 TFLOP/s, about
  0.6x: the tiles that fit 64 KiB of LDS are poor there); the odd head dims, now on
  buckets of their own width at 0.6-0.85x (above); and short or narrow problems (16 × 512, 16 heads × 1024: 0.35-0.6x, the
  128- and 256-row tiles leave CUs idle).
- **Found on the way:** without the KV global prefetch, bf16 at `head_dim` 64 with 128
  rows and 128 with 256 rows did not trace (an upstream loader mismatch, fixed by
  vendored-kernel modification 25, on gfx942 too).

---

## 8. Known deviations from the descriptor RFCs

Approved for this integration.

| Deviation | What FlyDSL does instead | Status |
|---|---|---|
| **Two-launch objects name their second kernel in metadata, not in RFC 0017 §15.1 `sources`.** RMSNorm backward (partial, then reduce) and both split-KV decode engines (split, then merge) launch two kernels compiled into one `.hsaco`. | Each UKD's `kernel_source` names the first kernel; the KMD field `reduce_symbol` (§2a) or `merge_symbol` (§5, §7) names the second, in the same packed object. The native `prepare()` loads both and `launch()` runs them in order on the plan stream. The generators verify both symbols exist in the object, with their recorded argument layouts, before writing it; the packer and the state manager's launch-set check see only the first symbol. | This is the form RFC 0021 §3 (kernel slots, draft) names as an accepted alternative to `sources` for dependent kernels: one UKD whose code object holds them, launched by its native handler through symbol names its KMD records, with a split-KV kernel and its reduction as the example. Optional later move: `hsaco` entries in `sources`, once the packer accepts them (proposed in #14215), which would bring both kernels under the packer's and loader's launch-set checks. The objects would not change. |

---

## 9. SDPA backward on CDNA — `hipkernel:flydsl_sdpa_bwd_merged` and `hipkernel:flydsl_sdpa_bwd_split`

gfx950 only. Both engines share one graph matcher
([`FlydslSdpaBwdCommon.hpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaBwdCommon.hpp)),
a pure function of the graph that either computes the problem or says why it declines; what
belongs to one engine (tile routing, bias staging, workspace bounds) is that engine's.

| Engine | Launches | Ops (objects) |
|---|---|---|
| `hipkernel:flydsl_sdpa_bwd_merged` | one: dK, dV, dQ and, on a bias graph, dBias in one grid, with no prep kernel | `sdpa_bwd_merged` (309) |
| `hipkernel:flydsl_sdpa_bwd_split` | three, in order, one per slot: prep (Delta and the statistics), dK/dV, dQ (and dBias) | `sdpa_bwd_prep` (86), `sdpa_bwd_dkdv` (1,319), `sdpa_bwd_dq` (1,010); `sdpa_bwd_split` carries the descriptors only |

The split engine is slotted (RFC 0021), so a plan that uses it cannot be serialized. A
broadcast dBias is summed by the dBias reduce kernel, a second symbol of the code object
that launches it (`sdpa_bwd_dbias_reduce`, 3 objects; the fp32 family has its own); the
dSink reduce is `sdpa_bwd_dsink_reduce` (2). The 16-bit families cover bf16 and f16;
fp32 is a native f32 MFMA family. A graph none of the shipped rows can compute is
declined with a named cause (data type, operand layout, mask, workspace overflow), so
another engine gets the plan. The kernels and how the shipped objects are pinned are in
[`kernels_src/kernels/attention_bwd_cdna/`](kernels_src/kernels/attention_bwd_cdna/README.md);
the split ops also carry the gfx11-generic set (432 objects: prep 33, dK/dV 198, dQ 198 and the 3 standalone dBias reduce objects, which are companions of the dQ op), built from [`kernels_src/kernels/attention_bwd_rdna/`](kernels_src/kernels/attention_bwd_rdna/README.md).
