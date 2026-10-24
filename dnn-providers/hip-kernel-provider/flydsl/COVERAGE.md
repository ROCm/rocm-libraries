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
from FlyDSL's own CDNA kernels. Every graph a pack accepts is
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
| **Ops other than RMSNorm** | Not these packs': `OPS` in `_instances.py` holds RMSNorm forward and backward and the SDPA forward families (§5, §7), each an engine of its own. |

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
generic tier (kernel modification 15): seventy-two more objects, bf16 and f16 ×
causal × largest head (64, 96, 128, 160, 224 or 256) × without a bias, with an f32
one and with one in the query's dtype. The actual
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
on such a head runs on its tier's bias objects, so every head_dim the generic tiers
serve takes a bias as the specialized ones do, with the same results (§5.7). Each bias
object keeps its plain twin's schedule, but for the tier up to 64, which takes the
128-row tile the specialized `head_dim` 64 bias object was measured to need.

The causal variant is the one that applies a right bound — top-left or
bottom-right causal, and any band to the right of the diagonal. It is two
objects rather than a runtime flag because it changes the loop: it bounds the
KV loop at the diagonal and skips fully masked tiles, and the non-causal one
carries a V prefetch across iterations that the causal one drops for register
pressure. Everything else about the mask is a runtime argument, served by both.

**Decode** has its own eighty-eight objects (`hipkernel:flydsl_sdpa_rdna_decode`; bf16 and f16 × causal × `head_dim`
64, 96, 128, 256 and two generic tiers, each without a bias, with an f32 one and with one
in the query's dtype, plus the wide 576/512 and 512/512 objects and the sink objects at 64 and
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
96, above it up to 256 with the column split, and past 256 up to 512 with the columns split
four ways (H4, kernel modification 30; no bias, no sink) -- that reads it at runtime and never
fetches the K and V columns past it, so a step moves only the bytes its head has (an
up-to-128 tier does not fit the register file), with or without a bias (a bias object
taking one tile, as at a specialized head_dim). Within the decode engine they rank
below the exact objects, the narrower tier above the wider (`priority` 145, 140, 135).
The tier up to 512 is the only RDNA object for 264-504 on gfx11 (on gfx12 §5.6's padded tiers
serve them too); it serves any query length at decode-tile speed. A K or V column past the
runtime head_dim reads at its slice's own end, past the buffer's range, so the hardware
returns zeros with no select after the load (modification 30): with the selects the tiers
below use, the gfx11 build spilled 5 VGPRs. It is slower than the exact 512/512 objects at
the same width (§5.8, deferred item 9).
On gfx11 the f16 objects still spill 1 VGPR, accepted for that class only
(`_SPILL_ALLOWANCE` in `gen_sdpa_rdna_decode.py`); the bf16 ones and gfx12's spill none.
K and V with different head counts stay on the prefill objects. A split launch takes
workspace (a partial O, the row max and the row sum per split, so the merge weighs each
split by its key count even when a bias near the dtype's lowest value puts every max far
below zero); one split writes O directly.

### 5.2 What a graph must be

One SDPA-forward node, and:

| Condition | Why |
|---|---|
| Q, K, V, O present, rank 4, device operands, **not ragged** but in compact THD on §5.6's objects (§5.9), each declaring 16-byte alignment; no two elements of O at one address | The RDNA kernels read each through a pointer as a dense batch; a ragged (THD) operand would be read as one. Rows move as 16-byte vectors from each base, and many workgroups write O at once. |
| One dtype for all four, `BFLOAT16` or `HALF` | One element type throughout; no fp32 WMMA operand form exists. |
| Head-dim stride 1 on all four; other strides positive | Each lane reads a row's head-dim values as one contiguous vector. Every other stride is a kernel argument, so BSHD, BHSD and packed-QKV views are all served. |
| K and V share batch and sequence; K has Q's head dim; O is Q's shape with V's head dim; V's head dim one an object bakes: equal to Q/K's, 128 with Q/K 192 (MLA prefill), or 512 with Q/K 576 (absorbed-MLA decode) | No broadcast. The graph matcher admits any V head dim, narrower or wider than Q/K's (§5.6's padded tiers take both at runtime, H1); each RDNA object's kernel matcher then requires its baked `head_dim_v`, and a generic tier's V is its runtime `head_dim`. The MLA objects bake both head dims (kernel modification 20); every other object has one. |
| `Hq % Hk == 0` and `Hq % Hv == 0`; K and V may differ in head count | The K head and the V head are each the query head divided by its own group size. |
| Any `right_bound` ≥ −1 and `left_bound` ≥ 1 or −1 at either named `diagonal_alignment`, or the deprecated causal booleans (not both) | The kernel applies the reference's two-sided rule (`CpuFpReferenceSdpa` `isMasked`) directly. `left_bound` follows cuDNN: `L` keeps the `L` keys ending at the diagonal, so the kernel is handed `L − 1`; `left_bound` 0 keeps none and declines. A boolean sets `right_bound` 0 unless the graph sets one: `causal_mask` at the stored `diagonal_alignment` (bottom-right when it says so), `causal_mask_bottom_right` always bottom-right. An alignment the schema does not name declines, boolean or not. |
| Scale: `attn_scale_value` (> 0 and ≤ 2²⁰), a pass-by-value scalar tensor, or absent | Absent is 1.0, cuDNN's default, which hipDNN's references apply too. Every FlyDSL SDPA family serves the same range (`FlydslSdpaShared.hpp` `MAX_SOFTMAX_SCALE`): positive, because the CDNA kernels take the running max over unscaled scores, and at most 2²⁰, beyond which each family overflows f32 at a different point, so all decline alike. A pass-by-value scale is read from host memory at launch, and one outside the range is refused there. |
| Stats, if requested: a named f32 `[B, H, Sq, 1]` device tensor, 4-byte aligned, positive strides on its axes longer than 1, no two elements at one address | The kernel writes the natural-log LSE of the scaled scores there, through its strides. |
| Bias (`attn_mask`), if present: an f32 device tensor, or one in the query's dtype, declaring 16-byte alignment, of rank 1–4 whose dims, right-aligned to `[B, H, Sq, Skv]`, are each that extent or 1; positive strides on real axes; one (batch, head) slice under 2³⁰ elements | Added to the scaled scores before the mask, as the reference does; a size-1 or absent axis broadcasts. The kernel reads it through a 32-bit buffer descriptor with 32-bit offsets. |
| `Sq`, `Skv` < 2²⁹; one (batch, kv head) slice of K or V under 2 GiB; grid under 2³¹ | int32 mask arithmetic, and 32-bit bounds-checked buffer descriptors. |

Any sequence length is correct: K and V reads past `seq_len_kv` return zero and
every such score is masked, so nothing is padded and nothing pads the softmax.
A row no key reaches — a narrow window, or bottom-right causal with more
queries than keys — gets O = 0 and LSE = −inf, as the reference defines it.

**Graph schema.** Before a matcher runs, the ingestor declines a graph whose schema floor
(hipDNN's `computeMinimumEnginePluginApiVersion`) is above the engine's UED `sdk_version`.
Every FlyDSL SDPA engine (RDNA, aiw and CDNA alike) claims **1.4.0**, the newest version
the SDK defines (`K_MAX_SUPPORTED_API_VERSION`), so ragged graphs reach the matchers; each
field the schema added up to it is gated:

| Schema | Field | Gate |
|---|---|---|
| 1.1.0 | graph `is_override_shape_enabled` | declined (`usesUnreadSchemaFields`): no matcher sees the execute-time shapes |
| 1.2.0 | tensor `is_runtime_pass_by_value` | the scale tensor must be one (`isServableScaleTensor`); a Philox seed or offset may be (`isServablePhiloxTensor`); every other tensor declines one where it is read (operands, LSE, bias, sink, lengths, offset tables, RNG dump) |
| 1.3.0 | tensor `ragged_offset_tensor_uid` | Q/K/V/O: compact THD or cuDNN's packed form, else declined (§5.9, §7.10); the LSE: packed through Q's table or dense; every other tensor declines it |
| 1.3.0 | tensor `alignment` | `declaresAlignment` wherever a device tensor is read: Q/K/V/O and the bias 16 bytes, LSE, lengths, offset tables and RNG dump 4, sink 2, seed and offset on the device 8 |
| 1.4.0 | tensor `ragged_offset_multiplier` | a ragged tensor's must be its own sequence stride (`isPackedThrough`); on a tensor that is not ragged anything but 1 declines (`usesUnreadSchemaFields`) |

`TestFlydslSdpaGraphSchema` pins the claim to the SDK's newest version: an SDK that adds one
fails it until the new fields are reviewed here. RMSNorm stays at 1.2.0 (§2). **What still
stands between a ragged graph and these engines:** the provider itself declares engine
plugin API 1.2.0 (`src/core/PluginPublic.cpp`), and hipDNN's backend skips a plugin below a
graph's floor, so through the backend no hip-kernel-provider engine receives a ragged graph
yet. Raising that is a provider-wide change: the non-ingestor engines (`asm_sdpa`,
`hip_mlops`, `hip_flash2`) would then receive ragged and aligned graphs with no version gate
of their own.

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
| 9 | head dims 64, 128, 256; `d % 8 == 0` | **served**: 64/96/128/256 specialized, every other multiple of 8 up to 256 by the generic tier, with or without a bias; multiples of 8 from 264 to 512 by the decode engine's generic tier up to 512 at any query length (no bias; H4), and on gfx12 by §5.6's padded tiers up to 384 and 512, with or without a mask |
| 10 | padding mask via `SEQ_LEN_Q`/`SEQ_LEN_KV` | **served** (§5.9) by every RDNA prefill and decode object (kernel modifications 26 and 28) and every object of §5.6, at every mask; padded rows O = 0 and LSE = −inf, lengths clamped to [0, S], both lengths without `padding_mask` read as padding (hipDNN's SEM-008) |
| 11 | sliding window (`left_bound`) | **served** (and right-side bands) |
| 12 | varlen / THD | **served** (§5.9): compact THD, ragged Q/O and K/V through cumulative token tables, the LSE packed alike, by §5.6's objects (gfx12) and the prefill objects (gfx11 and gfx12, kernel modification 29); packed with both lengths (cuDNN's form) by the prefill objects; the decode objects decline both |
| 13 | decode (`Sq == 1`) | **served by a decode path**: GQA-packed rows in 16-row tiles (one tile, 16 rows, on the bias objects) and split-KV, K and V with one head count (MQA, speculative steps) at `head_dim` 64/96/128/256 and every other multiple of 8 up to 512 (past 256 without a bias) |
| — | `Dv ≠ Dqk` | **served** for un-absorbed MLA prefill (Q/K 192, V 128) and absorbed MLA (Q/K 576, V 512) on the decode objects, which tile any number of rows, so absorbed chunked prefill runs there too; every mask, LSE; a bias on the 192/128 prefill objects (N2), none on the decode ones. The 576/512 objects are built for absorbed MLA's one KV head, but the matcher only requires K's and V's head counts to match. On gfx12, **any pair** of multiples of 8 up to 512, V narrower or wider, on §5.6's padded tiers (H1); the RDNA objects serve only the pairs above |
| 14 | additive bias | **served**: f32 or the query's dtype, broadcast over any of B, H, Sq, Skv; with every mask, the LSE output and every head dim the plain objects serve (specialized and generic tiers), on the prefill and the decode objects; with a sink at `head_dim` 64 and 128 (prefill and decode, both bias dtypes); on the MLA 192/128 prefill objects; not on the wide decode objects |
| — | dropout | **served** (§5.10): on gfx11 by the RDNA prefill objects' dropout twins (`head_dim` 64/96/128/256), on gfx12 by §5.6's objects (`head_dim` 64/128/256); bf16/f16, causal or not, with or without an additive bias (f32 or the query's dtype; R1, R2), no sink, no padding; the decode objects decline it |
| — | attention sink | **served** at `head_dim` 64 and 128 (gpt-oss), prefill and decode, with every mask and the LSE output, with or without an additive bias (f32 or the query's dtype): one logit per query head in the query's dtype, rank 1 `[Hq]` or rank 4 `[1, Hq, 1, 1]`, a key with no value in each row's softmax (kernel modification 21). A row no key reaches gets O = 0 and LSE = the sink; with a bias, the sink and the bias share the natural-log domain, so a sink far above a row the bias holds at its lowest finite value takes the whole softmax. On gfx12, §5.6's sink objects also take either additive mask. hipDNN's CPU reference declines sinks; the GPU tests check them against their own reference |

### 5.4 What is still missing, and what closing it takes

Each is declined today, so the graph plans on another engine or fails cleanly
at `check_support()`, never at `execute()`.

| | Effort | What it takes |
|---|---|---|
| THD on the decode objects; packed with lengths on aiw | medium each | Decode: the merge kernel indexes O by batch, so the row-offset table would join its arguments (a decode ABI change, every decode object rebuilt and A/B'd); a packed decode step runs on the prefill objects meanwhile. aiw packed with lengths: the fill kernel needs each sequence's base (a row-offset argument) and the main kernel a length clamp to the slot (`POSITION ARRAY` gives starts only), an aiw kernel change with every object rebuilt; the RDNA prefill objects serve it on gfx12 meanwhile. |
| Bias on the wide decode (576/512, 512/512) objects | small per class | A `has_bias` variant of each class, built and checked against the reference as the generic tiers' were; not built until a model needs them. |
| A prefill schedule for absorbed MLA; other `Dv ≠ Dqk` pairs on gfx11 | medium; small per pair | 192/128 prefill (modification 20) and the 576/512 and 512/512 decode objects ship; on gfx12 §5.6's padded tiers serve any pair (H1). Chunked prefill at 576/512 runs on the decode objects' 16-row tiles; a prefill schedule at that width would share each K/V tile across more rows. Other pairs on the RDNA objects: a runtime V width on the generic tiers (a second appended argument, masked V loads and O stores; every generic tier rebuilt and A/B'd), not built until a model needs it. |
| fp32 I/O, FP8, paged KV, block masks, ALiBi, softcap | out of scope | Not planned for these engines; ALiBi and softcap have no reference semantics. |
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
| Each decode object carries its merge kernel, with both argument layouts as declared | `gen_sdpa_rdna_decode.py`, before the object is written; the 44-slot main layout by `TestFlydslSdpaRdnaDecodePacks` |
| Each prefill object has the declared 47-argument layout and spills no registers | `gen_sdpa_rdna.py`, before the object is written; the layout again by `TestFlydslSdpaRdnaPacks` |
| The vendored kernel matches AITER but for its recorded modifications | `tools/diff_upstream.py --aiter` ([REGEN.md](REGEN.md) §5) |
| Each engine's shard ships exactly one object per `dtype × head_dim × causal × has_bias × bias_q × head_dim_max × has_sink` class, each with its engine's one signature | `TestFlydslSdpaRdnaPacks` and `TestFlydslSdpaRdnaDecodePacks`, and the shard census that runs them |
| The accept/decline rules above, the bindings, and candidate selection | `TestFlydslSdpaRdnaGraphAccepts`, `TestFlydslSdpaRdnaGraphDeclines`, `TestFlydslSdpaRdnaBinding`, `TestFlydslSdpaRdnaKernelMatch`; padding (§5.9) `TestFlydslSdpaVarlen*` and `Padding/TestFlydslSdpaRdnaFieldDeclines` |
| Which shipped aiw object a graph selects for the classes added after the engine landed (head dims 264-512 on the tiers up to 384 and 512; a sink with either mask at 64 and 128; a V head dim of its own on the tier of the wider, H1), and that the top-ranked one matches the reference on gfx12 (§5.7's mask and sink rows included), beside §5's decode objects at 512/512 | `TestFlydslSdpaAiwClasses`, `TestGpuFlydslSdpaAiwClasses` |
| The staged objects compute attention and LSE — GQA, ragged lengths, cross-attention, both causal corners, windows, bands, keyless rows, decode, `head_dim` 96, runtime and default scale, a bias at every generic tier, bias rows and tiles at −inf and at the lowest finite value (§5.7) — against a double-precision reference | `TestGpuFlydslSdpaRdnaDispatch`, and every staged object by `TestGpuFlydslSdpaObjects` |

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
  a padded tier's width (64, 96, 128, 160, 224, 256, 384, 512), Q/K 192 with V 128 exactly;
  on a padded tier, any pair of Q/K and V head dims, V narrower or wider, each a multiple of 8
  and the wider within the tile (the kernel bounds Q/K's columns by `hdim_qk` and V/O's by
  `hdim_vo`, two runtime arguments; H1, no object change); causal at
  either alignment, sliding windows and two-sided bands (hipDNN's bottom-right window is
  the kernel's shifted by `Skv - Sq`); an additive mask in Q's dtype or f32, broadcast
  over B, H and Sq, its key axis contiguous, on the exact head dims and every padded tier
  alike; attention sinks at 64 and 128, alone or with either mask (one natural-log domain,
  README modification 9; the sink joins the row's max, so a sink far above a row the mask
  holds at its lowest finite value takes the whole softmax, as PyTorch's does); a V head
  count of its own; the LSE through its own strides; a scale baked or read at launch;
  padding (per-batch lengths, §5.9) and packed (THD) sequences (§5.9) on every object, read
  at runtime; dropout (§5.10) on its `has_dropout` objects.
- **Edge cases, as §5's prefill:** a mask at the lowest finite value (bf16 or f32
  `finfo.min`, the HF padding convention) on every key of a row gives O = mean(V) and an
  LSE near the mask, as PyTorch does; a mask or sink above 2.36e38 no longer overflows
  (README modification 9: the scores, max and sink live in the natural-log domain, the
  RDNA kernel's modification 22). The LSE of a row whose softmax sum is NaN is NaN.
- **Declined:** a mask whose key axis is strided; Q, K, V or O, or a mask, with a batch,
  head or sequence (query) stride that is not a multiple of 8 elements; an f32 mask whose
  declared alignment is under 32 bytes; more than 65,535 query tiles or batches (the
  grid's y and z); a sink on a padded tier or at 256 (with or without a mask); head dims that are
  not multiples of 8, or past 512, on either side (absorbed MLA's 576/512 runs on §5's decode
  objects); anything past 32-bit offsets; padded or packed,
  a bottom-right mask other than plain causal (§5.9); packed with a mask, or with a dense LSE
  (§5.9); dropout with a mask, a sink, padding, packing, at a padded head dim or MLA (§5.10).
- **Which engine runs a graph** both this and §5's prefill engine serve is autotune's
  (UHD's) to decide: measured on gfx1201, this one is faster on MLA prefill and long
  sequences, §5's on decode and sinks.
- **Head dims past 256** (264-512, AOTriton's gfx1201 FlyDSL build ships 384 and 512) run on
  the padded tiers up to 384 and 512, the tuning table's own 384 and 512 schedules, plain or
  with either mask. They are the only RDNA objects for 264-504 at any tile; at 512/512 with one
  KV head §5's decode objects serve too, and autotune picks. Both tiers spill heavily (as an
  exact 384 or 512 build of this kernel does; allowances per class in `gen_sdpa_aiw.py`, §5.8).

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
| Padding: a row past its batch's query length | Not written: O and the LSE keep what the buffer held (§5.9) |
| Padding: a batch whose key length is 0 | Every row keyless: O = 0, LSE = −inf |
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

### 5.8 Measured costs, and deferred performance work (RDNA, aiw)

The RDNA counterpart of §7.5. Measured on gfx1201 against the tree before the change: 7
interleaved rounds per shape, bf16, 32 query and 8 KV heads, median of the round medians.
The measurements are kept outside the repository.

| Change | Result | Decision |
|---|---|---|
| A bias on the generic prefill and decode tiers, a mask on the aiw padded tiers (new objects only; §6.2) | every object shipped before regenerates byte-identical and times as before (63 object × shape rows: median +0.06%, all within ±0.6% but one decode row, faster by 3.7%) | shipped |
| What the new bias objects cost over their plain twins | prefill, the tier up to 96 at 4096 tokens: +11–13%, as the specialized `head_dim` 128 bias objects (+10–13%); the tier up to 224: +4–7%; the tier up to 128 at `head_dim` 120: +22–29%, 4–7% slower than the specialized `head_dim` 128 bias objects running their full width. Decode: see the next row | kept, for coverage -- **deferred** (item 1) |
| Decode bias loads issued ahead of the QK product (modification 25) | the bias objects below `head_dim` 128 issued their sixteen per-tile bias loads after the QK product and waited on them at once; ahead, they overlap it, as at 128 and up already. Measured in steady state (5 rounds of 6,000 launches per timing, against the objects before it): `head_dim` 64 at B4, 8192 keys −20% (the bias cost over the plain object falls from +32% to +6%), short caches −1 to −4%; the generic tier up to 96 at `head_dim` 80 on long caches −2 to −5% (bias cost +5–6% → about +1%), short caches within noise (+0.1–0.7%, sign not repeating). Timed in 300-launch batches instead, on gfx1201 and gfx1200 alike, `head_dim` 64 gains 4–13% on long caches and the generic tier's short-cache rows read +0.7–1.9% (the accepted cost). The specialized `head_dim` 96 objects measured +0.8–1.0% on short caches and −0.2% on long ones with it, so keep the late loads. Objects without a bias and every bias object that already loaded early are byte-identical | shipped, `head_dim` 64 and the generic tier up to 96 |
| Per-batch lengths in every RDNA prefill and decode object (modification 26; all 440 objects change) | 6 interleaved rounds against the objects before it, 63 object × shape rows, dense graphs: gfx1201 median +0.06% (all rows within ±1.6% but one), gfx1101 median +0.17% (max +2.6%, a node shared with other jobs). The exception: the `head_dim` 64 causal sink prefill object on gfx1201, +6.5% at 10 µs and +1.3% at 133 µs, with identical registers and the length path not taken (−1.1% on gfx1101): the object's code layout. Two earlier forms cost every short launch 2–8% and were replaced: a branch on the length pointers (it waited on them before the other argument loads) and a branchless read (a cold line every dense launch) | shipped -- **deferred** (item 5) |
| Padding's own cost on the RDNA prefill objects (gfx1201, top-left causal) | padded at full lengths against the same graph dense: 1.000–1.016×; at half lengths 0.27–0.43× the dense time | -- |
| The aiw padded mask objects | faster than the RDNA generic bias objects at `head_dim` 120 (by about 20%), 200 (about 40%) and at 80 on 1024 tokens; at 80 on 4096 tokens the bf16 mask is faster, the f32 one 50% slower than its plain twin and slower than the RDNA object. VGPR spills at 128, 224 and 256 (up to 86, allowed per class in `gen_sdpa_aiw.py`) | shipped; which engine runs a graph is autotune's (UHD's) choice |
| The aiw padded tiers up to 384 and 512 (new objects only; 5 rounds, bf16, two gfx1201 nodes within 3%) | the tier up to 384: 47–51 TFLOP/s at `head_dim` 384 and 40–43 at 320 (B1, H8, 4096 tokens; 39–44 at B2, H16, 1024), either mask within ±8% of plain. The tier up to 512: 34–42 TFLOP/s plain at 512 and 34–35 at 448; a mask costs 1.3–2.1× (bf16 25–33, f32 20–27 TFLOP/s). Against §5's decode objects at 512/512 with one KV head: 1.8–2.0× faster non-causal and 2.0–3.9× faster bottom-right causal (the more the diagonal masks, the more) from 64 query rows up (4096 and 16384 keys); at one query row the decode objects are 8× faster (48 against 397 µs, B4 H8, 4096 keys). AOTriton's tuning notes put exact 384/512 builds at 54–61 and 52–55 TFLOP/s (f16, upstream) | shipped; autotune picks between the aiw and decode objects at 512/512 -- **deferred** (item 6) |
| An aiw sink with a mask (new objects only) | the sink adds nothing measurable to a mask object: within −1.6% to +0.9% of the same mask without a sink at `head_dim` 64 and 128 (1024 and 4096 tokens, both masks, causal and not). Over a sink alone the mask costs what it costs without a sink (bf16 +4–28%, f32 +6–26%) | shipped |
| The RDNA gfx11 dropout twins against their plain twins (kernel modification 27; gfx1101, bf16, p = 0.1, 5 rounds, new objects only, the plain ones byte-identical) | 1.29–2.04×, median 1.51×: `head_dim` 64 1.48–1.61×, 96 1.38–1.53×, 128 1.29–1.41× (the 128 twins re-read Q per K-step, their plain twins hold it), 256 1.78–2.04× (the twin splits its columns two ways, so each half repeats the QK product and the Philox draw) | shipped, for coverage -- **deferred** (item 7) |
| Packed (THD) sequences on the aiw objects (no object changed; gfx1201, bf16, top-left causal, the LSE packed, 5 rounds) | packed at full lengths against the same problem dense: 0.998–1.003× (padded: 0.998–1.020×); at half the lengths 0.25–0.64× the dense time | -- |
| Padded rows filled and lengths clamped on every RDNA prefill and decode object (modification 28; all 456 change, registers unchanged but one decode object's +1 SGPR) | 5 interleaved rounds (alternating order) against the objects before it: dense graphs on gfx1201 prefill median +0.15%, decode +0.27%; on gfx1101 +0.35% and +0.12%, within noise but for a single 43 µs row. Padded graphs at half lengths write the padded rows they used to skip: up to +12% on a short launch, the feature's own cost | shipped |
| Packed (THD) sequences on every RDNA prefill object (modification 29; all 360 change, VGPRs unchanged but +1 on eight gfx11 generic-tier bias objects, within the same allocation granule; the decode objects byte-identical) | 7 interleaved rounds (alternating order) against the objects before it, 51 object × shape rows, dense graphs: gfx1201 median +0.04%, every row within −1.5% to +0.7% (a 5-round chunk round had flagged the bf16 `head_dim` 96 causal object at +6.7% on a 43 µs launch; it did not repeat); gfx1101 median −0.05% (a node shared with other jobs, where byte-identical decode objects swung by up to 13%). The packed path itself: packed at full lengths against the same problem dense 0.995–1.011× on gfx1201 (0.98–1.06× on gfx1101), at half the lengths 0.24–0.43×, and packed slots at full length with per-batch lengths at half (P3) 0.25–0.45× | shipped |
| The RDNA generic decode tier up to 512 (H4; new objects only, the 568 others byte-identical) against the exact 512/512 objects at nearly the same width (bf16, bottom-right causal, one or two KV heads, 1 and 16 query rows, 4096 and 16384 keys; 5 rounds) | at `head_dim` 504: gfx1201 1.23–1.29×, gfx1101 2.6–3.2×; at 384: gfx1201 0.95–1.02×, gfx1101 1.24–1.73×. The first build (a select after each K and V load) spilled 5 VGPRs on gfx11 and took 3.0–3.6× there; reading out-of-range columns at the slice's end removed the spill and a tenth of the cost, so the spill was not the main cost. On gfx11 nothing else serves 264–504 | shipped, for coverage -- **deferred** (item 9) |
| Padded rows filled on the aiw objects (modification 10: a fill kernel beside each main kernel, which keeps its registers; the length clamp) | 5 interleaved rounds on gfx1201 against the objects before it: dense graphs median +0.01% (37 rows, −2.7% to +0.5%). A padded graph launches the fill kernel too: +2.8% to +4% at full lengths (133–417 µs), and at half lengths, which now write their padded rows, up to +24% on a 56 µs launch. The first form, the same stores inside the main kernel, cost one VGPR and on six objects a wave of occupancy (`head_dim` 128 non-causal 7–19% slower dense) and was replaced | shipped -- **deferred** (item 8) |

**Deferred performance work** -- costs accepted for coverage, to revisit:

1. *The generic prefill tier up to 128 with a bias.* Like the specialized `head_dim` 128 bias
   object it re-reads Q per K-step (kernel modification 14), and it also masks the columns
   past the runtime head_dim. A 64-row tile or head-dim pairs for this one tier are the
   candidates.
2. *The decode bias objects' per-element bias loads.* Each lane loads its sixteen bias values
   one element at a time, a full tile's worth of address arithmetic and sixteen loads per KV
   step; issued ahead (modification 25) their latency is hidden, but the bias still costs
   about 6% at `head_dim` 64 on long caches. Vector loads of a lane's contiguous keys (the
   gfx12 layout reads eight adjacent keys per lane) are the candidate. An earlier note here
   put the generic tier's cost at 12–14% on a short cache: that was its plain twin timed in
   a warmer GPU state than its bias objects; measured alike, the gap is about 2%.
3. *The bias schedule of the generic tier up to 64.* It takes the specialized `head_dim` 64
   bias object's 128-row tile; the 64-row tile its plain twin uses was not measured with a
   bias.
4. *The aiw f32-mask padded objects.* The f32 mask's v8 loads on top of the column masks
   spill at 128, 224 and 256 and cost most at 96; the kernel's tuning table plans these
   tiers as it plans the plain ones. A plan of their own (fewer row sub-tiles, a shallower
   prefetch) is the candidate.
5. *The `head_dim` 64 causal sink prefill object on gfx12-generic* is 6.5% slower on a
   10 µs launch since modification 26 (1.3% at 133 µs), with the same registers and the new
   path not taken: a code-layout effect on that one object. Candidates: an alignment hint on
   its KV loop, or a schedule of its own.
6. *The aiw tiers up to 384 and 512.* They run the tuning table's own 384 and 512 schedules,
   which spill at any build of this kernel (an exact 384 or 512 object spills 40-129); the
   padded column masks add about 40 spills at 384 and 20 at 512, and a mask at 512 another
   30-80, which is where the 1.3-2.1x mask cost comes from. Candidates: exact 384 and 512
   objects beside the tiers (16, AOTriton's own set; their speed against the tiers is
   unmeasured), and a 512 mask schedule of its own (fewer QK shards or a shallower V
   prefetch).
7. *The RDNA `head_dim` 256 dropout twins* take 1.8–2.0× their plain twins on gfx1101, where
   the other head dims take 1.3–1.6×: with the Philox state the plain 256 schedule spills, so
   the twin splits its output columns two ways and each half repeats the QK product and the
   mask draw. Candidates: draw the mask once per tile into LDS for both halves, or a 256
   schedule whose Philox state fits without the split.
8. *The aiw fill kernel's launch on padded graphs.* A padded graph pays a second launch
   (+3-4% at 130-420 µs when no row is padded), since the host cannot see the device
   lengths. Candidates: fold the fill back into the main kernel once a form leaves its
   registers alone (an occupancy floor the allocator honours), or skip the launch where the
   caller states the lengths are full.
9. *The RDNA generic decode tier up to 512 on gfx11* takes 2.6–3.2× the exact 512/512
   objects at `head_dim` 504 (gfx12 1.2–1.3×), with no spill in its bf16 build: every K and V
   load computes its own offset (a select between the column and the slice's end), where
   the exact objects fold the column into the load's immediate offset and keep more loads in
   flight at the same register count. Candidates: branch once per 16-column K-step and per
   8-column V chunk on the runtime head_dim (uniform), loading with immediate offsets inside
   and skipping whole steps past it, which also stops a narrower head paying the tier's
   full width; or tiers at 384 and 448.

### 5.9 Varlen and padding

hipDNN states padding cuDNN's way: `padding_mask` with `seq_len_q` and `seq_len_kv`, one
length per batch, over dense Q/K/V/O whose sequence extent is the longest sequence
(`FlydslSdpaVarlen.hpp`, shared by every family's graph matcher). Packed sequences are RFC
0014's ragged tensors: each operand's sequences stored end to end, located through an offset
table, the sequence extent still the longest sequence.

| | aiw (gfx12) | RDNA prefill, decode | CDNA |
|---|---|---|---|
| Padding (both lengths, with or without `padding_mask`) | **served**, every object (the kernel's runtime varlen path, `varlen_bits` 0x0404; padded rows filled by modification 10) | **served**, every object (kernel modifications 26 and 28) | **served**, §7.8 |
| Packed THD, compact (ragged Q/O and K/V, no lengths) | **served**, every object (the kernel's stacked varlen path, `varlen_bits` 0x0B0B) | **served** on the prefill objects (kernel modification 29); the decode objects decline | **served** on the generic prefill objects, §7.10 |
| Packed with both lengths (cuDNN's form; each table entry a slot) | declined (below) | **served** on the prefill objects (modification 29, P3) | **served** on the generic prefill objects (P3, §7.10) |

- **The lengths:** INT32 `[B, 1, 1, 1]`, batch stride 1, a device tensor declaring 4-byte
  alignment; not virtual, not ragged, not pass-by-value. INT64 lengths, either length alone,
  and `padding_mask` without both lengths decline.
- **Both lengths without `padding_mask` are padding too** (`servableVarlenAsReferenceReads`):
  hipDNN's references read them so (their SEM-008, which keeps hipDNN's golden bundles
  valid); cuDNN would refuse such a graph.
- **Lengths outside `[0, S]` are clamped to it** in the kernels, as hipDNN's GPU reference
  clamps them (RDNA kernel modification 28, aiw modification 10): a length past `S` would
  otherwise read past the batch's slice.
- **Padded rows -- past a batch's query length, inside `S` -- are written as O = 0 and LSE =
  −inf**, hipDNN's reference contract (cuDNN ≥ 9.14), sink objects included: a padded row
  is not a keyless row. RDNA (kernel modification 28): a store of constants beside the live
  rows' store, under bit 2 of `flags`; on the decode objects with several splits the merge
  kernel fills them. aiw (modification 10): a second kernel in each object,
  `flash_attn_aiw_padfill_kernel`, launched by the native beside the main kernel for padded
  graphs only -- inside the main kernel the same stores cost a VGPR in every form tried, and
  at `head_dim` 128 that cut occupancy (19% slower); the main kernels keep their VGPR and
  spill counts. Rows can instead be left as the buffer held them (AOTriton's behaviour, and
  free); the RDNA and aiw natives choose with one switch, `FILL_PADDED_ROWS`
  (`FlydslSdpaRdnaCommon.hpp`), on for every padded graph. CDNA has no such switch: a padded
  launch always fills (§7.8). Whole tiles past the length still run no KV iteration.
- **Keys past a batch's key length are masked** with the dense tail's own compare, so a
  batch with no key gives every row O = 0 and LSE = −inf.
- **The mask, per batch:** top-left masks and windows are the same literal bounds for every
  batch. A bottom-right diagonal sits at each batch's own `Skv_b − Sq_b`. The RDNA kernels
  compute that offset in the kernel for every bound. The aiw kernel resolves bottom-right
  causal per sequence (its `WINDOW_BOTRIGHT` sentinel) but takes any other bound as one
  literal per launch, so the aiw objects decline a bottom-right window or band under padding
  and the RDNA objects serve it.
- **What a dense launch pays** (kernel modification 26): the RDNA kernels read the lengths
  only when bit 1 of their `flags` argument (the LSE switch's word) says they are passed, so a
  dense launch loads nothing more. Measured against the objects before it (§5.8): within
  noise on gfx1101 and, but for one object, on gfx1201.
- **Tested** against a test-local per-batch reference (`TestGpuFlydslSdpaAiwDispatch.Pads*`,
  `TestGpuFlydslSdpaRdnaDispatch.Pads*`, `...PutsEachBatchsBottomRightMaskAtItsOwnCorner`,
  `...DecodesEachBatchAgainstItsOwnCache`, until hipDNN's reference serves padding): lengths 0,
  1, odd and full, keyless batches, bottom-right with more queries than keys, a bottom-right
  window (RDNA), GQA, a window, a mask, a sink, generic and padded head-dim tiers, batched
  decode with caches of 0, 1, short and full length, split several ways, lengths below 0 and
  past `S`, both lengths without `padding_mask`; O and the LSE filled with NaN beforehand and
  padded rows required to read exactly O = 0 and LSE = −inf. Each RDNA kernel change was
  mutation-checked against these on gfx1151 (modification 28: every step killed but the
  lower clamp, an equivalent mutant -- a negative length already acts as 0), and the
  natives' fill switches on gfx1151 and gfx1201.
- **Packed (THD), compact** (`servablePackedVarlen`), the form hipDNN's ragged reference runs:
  Q and O ragged through one INT32 `[B + 1, 1, 1, 1]` device table, K and V through one (the
  same one for self-attention, or another), each tensor's `ragged_offset_multiplier` its own
  sequence stride, so the stored offsets count tokens (AOTriton's and AITER's `cu_seqlens`),
  and each token's heads within one sequence stride (token-major; a head-major packing
  declines). The kernel reads sequence b's rows from `offset[b]` to `offset[b + 1]` and
  ignores the batch stride. The offsets are the caller's to keep non-decreasing from 0 with no
  sequence past the extent, as the reference requires; the kernel does not check them.
- **The LSE under packing** must be packed through Q's table too (`[T, H]`, any token and
  head stride): a dense `[B, H, S, 1]` LSE needs each sequence's batch base, which the stacked
  launch does not have, so it declines. No LSE is fine.
- **Not served packed:** a mask (laid out by batch) and dropout, on every family. On aiw, a
  bottom-right mask other than plain causal (resolved per sequence only for causal, as under
  padding); the RDNA prefill objects move every bottom-right bound with each sequence's own
  lengths and serve them all. A sink is served. The RDNA decode objects decline packed graphs
  (their merge kernel indexes O by batch), so a packed one-query-row graph runs on the prefill
  objects. CDNA serves compact packing on its generic prefill objects under the same rules,
  and packing with both lengths too (P3, §7.10).
- **RDNA prefill, packed (kernel modification 29):** bit 3 of `flags` makes the kernel read
  sequence b's token range from the two tables (the `ROW_OFF_Q` / `ROW_OFF_KV` pointers
  modification 26 appended, unread until then, so no argument moved). A batch's base is its
  first token's row times the batch stride, which the native passes as each operand's
  sequence stride (the LSE's too); its query and key counts are the tables' differences,
  clamped to `[0, S]`. Q's loads stop at the rows the sequence owns, so none reaches the next
  sequence or past the last; the K/V slices already stop at its key count. The read sits in
  the lengths' one branch on the flags word, so a dense launch still loads nothing.
- **Packed with both lengths (P3, `servablePackedVarlenWithLengths`), cuDNN's form**: ragged
  operands as above plus both length tensors, with or without `padding_mask` (as over dense
  operands). Each table entry is then a slot: sequence b's first `seq_len[b]` tokens are its
  rows and keys, clamped to `[0, slot]`; the slot's rows past the length are padded rows
  (O = 0, LSE = −inf, as under padding) and its keys past the length are never read. Served
  on the RDNA prefill objects (both bits 1 and 3). One length alone, `padding_mask` without
  both, or a length tensor the kernels cannot read declines, and so does every packing the
  compact form declines. **aiw declines it:** its kernel could read the form (`varlen_bits`
  STACKED, LENGTH INDIVIDUAL, POSITION ARRAY: `0x1515`), but its fill kernel addresses a
  padded row by batch, which is another sequence's under packing, and its main kernel clamps
  a length to `Max_seqlen_q`, not to the slot; both are aiw kernel changes (§5.4). On gfx12
  the RDNA prefill objects serve it.
- **Tested** on gfx1201 (`TestGpuFlydslSdpaAiwDispatch.Packs*`, `...PutsEachPacked*`): the
  harness packs the dense values token-major and unpacks O and the LSE, so the per-sequence
  reference above applies unchanged; lengths 0, 1, odd and full, keyless sequences,
  bottom-right with more queries than keys, GQA, a window, a sink, a padded tier, one table
  for every operand. Every matcher gate has a decline case (`Packed/...FieldDeclines`) and was
  mutated; so were the native's packed paths (the VarlenBits, K's table, the per-sequence
  window, the bound K table), each failing these device cases on gfx1201.
- **Tested, RDNA prefill** (`TestGpuFlydslSdpaRdnaDispatch.Packs*`, `...PutsEachPacked*`,
  on gfx1151, gfx1101 and gfx1201): the same harness and reference; lengths 0, 1, odd and
  full, keyless sequences, bottom-right causal and a bottom-right window with GQA, a generic
  tier, a sink, one table for every operand; packed with lengths: lengths 0, below and past
  the slot, keys cut short, with and without `padding_mask`, a sink (padded rows −inf, not
  the sink). The packed buffers hold exactly the sequences' tokens, so a load past the last
  sequence would leave the allocation.
- **What packing costs:** nothing measurable on gfx1201 -- packed at full lengths times as the
  same problem dense (0.998–1.003×, §5.8).

### 5.10 Dropout

hipDNN states dropout cuDNN's way: `dropout_probability` with a seed and an offset tensor
(`FlydslSdpaDropout.hpp`, shared by every family's graph matcher).

| | aiw (gfx12) | RDNA prefill, decode | CDNA |
|---|---|---|---|
| Probability in (0, 1) with seed and offset | **served** by 36 `has_dropout` objects: bf16/f16 × causal × `head_dim` 64, 128, 256 × {no mask, a mask in the query's dtype, an f32 mask (R1)}, no sink | prefill on gfx11: **served** by 48 `has_dropout` objects (kernel modification 27): bf16/f16 × causal × `head_dim` 64, 96, 128, 256 × {no bias, an f32 bias, one in the query's dtype (R2)}, no sink; prefill on gfx12 (the aiw objects serve it, decision Q3) and decode: declined | **served** by 24 `has_dropout` objects per arch on the generic prefill tiers (12 without a bias, 12 with one the vectorized objects read), §7.9 |
| Probability 0 | the plain objects (the undropped result) | the plain objects | the plain objects |

- **The mask is AOTriton's**, bit for bit from AOTriton 0.15 on: Philox4x32-10 keyed by the seed,
  one counter per 4 keys of a (batch, query head, row) at
  `offset1 + offset2 + ((b·Hq + h)·Sq + i)·cdiv(Skv, 4) + j/4`, key `j` taking word `j % 4`,
  kept iff that word as int32 exceeds `int32(f32(p − 0.5)·2³²)`. Checked against AOTriton
  b5e8cfb's `fast_dropout_mask` and the host oracle the tests use. **PyTorch 2.10 bundles
  AOTriton 0.11.1, whose words 1 and 2 sit the other way round** (columns `4g + {0, 2, 1, 3}`),
  so its masks differ from these; from AOTriton 0.15 on they are the same. cuDNN's mapping is
  undocumented: the promise there is the same semantics (keep with probability `1 − p`, kept
  probabilities scaled by `1/(1 − p)`, the softmax normaliser and the LSE undropped), not the
  same bits.
- **The threshold is computed in f32**, as AOTriton's kernel does it; PyTorch-ROCm applies
  exactly that one (measured at probabilities where the f32 and double forms disagree on an
  element). `philox.py`'s host helper computes it in double and is not used.
- **One graph, one mask:** the mask depends only on the seed, the offset, B, Hq, Sq, Skv and
  each element's position, not on the tile schedule; each query head of a GQA group draws its
  own.
- **Seed and offset:** `[1, 1, 1, 1]`, INT64 on the device (8-byte aligned; the kernel reads
  it there, as PyTorch passes it), or INT64/INT32 by value (widened on the host; a seed given by
  value is written to 8 bytes of workspace with two memsets on the launch stream, so the
  launch stays capturable). INT32 on the device, a seed or offset without a probability, a
  probability outside [0, 1), the mask-and-scale form, and the RNG dump decline.
- **Declined in this first wave**, each a row of objects when a model needs it: dropout with a
  sink, under padding, on the padded head-dim tiers and the generic tiers, MLA,
  on the RDNA gfx12 prefill objects (the aiw objects serve gfx12), and on the decode objects
  (CDNA's own first wave and its declines: §7.9). Decode objects never serve dropout
  (inference); a dropping graph with one query row runs on the prefill objects.
- **Tested:** the oracle against AOTriton's own masks (`TestFlydslSdpaPhiloxOracle`), every
  matcher gate (`Dropout/TestFlydslSdpaRdnaFieldDeclines`, `TestFlydslSdpaDropout*`), and on
  gfx1201 the mask read off O bit for bit (Q zero, V the identity), the full output and the
  undropped LSE against the reference with the oracle's mask, a seed and offset by value, and
  probability 0 (`TestGpuFlydslSdpaAiwDispatch.Drops*`); on gfx1151 the same for the RDNA
  twins: the mask bit for bit at `head_dim` 64, 96 (causal), 128 and 256 (causal), the
  undropped LSE with GQA and a bottom-right mask, seed and offset by value
  (`TestGpuFlydslSdpaRdnaDispatch.Drops*`). Each of the RDNA kernel's dropout steps (the peer
  bit trade, the signed compare, offset1, the scale, the query-head plane, the bit order) was
  mutated and the mutant failed these tests.
- **The RDNA twins (kernel modification 27)** draw the mask after the row sum, so `l` and the
  LSE stay undropped, and fold `1/(1 − p)` into `1/l`. gfx11's WMMA layout gives a lane the
  even or the odd keys of 16; each lane draws the 8 consecutive keys of its half (two Philox
  counters) and trades the packed keep bits with its xor-16 peer, one shuffle per 16 keys. At
  `head_dim` 128 a twin re-reads Q per K-step and at 256 splits its columns two ways, where
  its plain twin does not: with the Philox state the plain schedules spill (29 and 2 VGPRs).
  The twins take 169–244 VGPRs, no spill.

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
| Per-batch lengths (padding, modification 26) | a flag test and two scalar loads per workgroup, under the flag only | tried as a branch on the length pointers and as a branchless read: both cost short launches 2–8%; behind a bit of the LSE flag word, within noise but for one object (§5.8) | runtime |
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
| SDPA forward and decode, a bias on the generic tiers | an f32-bias and a query-dtype-bias object (`has_bias`, `bias_q`) beside each of the 24 generic prefill and 8 generic decode objects; each keeps its plain twin's schedule, but the prefill tier up to 64 takes the 128-row tile of the specialized `head_dim` 64 bias object | 64 | shipped, `gfx11-generic`; the 309 above byte-identical |
| SDPA forward, dropout (§5.10) | a dropout twin (`has_dropout`, kernel modification 27, Philox per score) of dtype (2) × causal (2) × `head_dim` {64, 96, 128, 256}, no bias, no sink; Q re-read at 128, columns split two ways at 256 | 16 | shipped, `gfx11-generic` only (gfx12 takes dropout from the aiw objects); the 373 above byte-identical |
| SDPA forward and decode, a bias with a sink (N1) | an f32-bias and a query-dtype-bias object beside each of the 16 sink objects (`has_sink` with `has_bias`, `bias_q`), each on its bias schedule; the kernels compose the two (the sink takes the bias's natural-log domain), no spill | 16 | shipped, both RDNA targets; the 389 above byte-identical |
| SDPA forward, MLA with a bias (N2) | an f32-bias and a query-dtype-bias object beside each of the 4 MLA objects (`has_bias`, `bias_q`), on the 192 schedule; no spill | 8 | shipped, both RDNA targets; the 421 above byte-identical |
| SDPA forward, dropout with a bias (R2) | an f32-bias and a query-dtype-bias object beside each of the 16 dropout twins (`has_dropout` with `has_bias`, `bias_q`), on the bias schedule (at 256 the twins' column split); no spill | 32 | shipped, `gfx11-generic` only; the 429 above byte-identical |
| SDPA decode, a generic tier up to 512 (H4) | dtype (2) × causal (2); every multiple of 8 up to 512 (kernel modification 30 lifts the generic decode cap from 256), the output columns split four ways (`dv_split` 4), K and V columns past the runtime head_dim read as zeros past the slice; no bias, no sink; the gfx11 f16 objects spill 1 VGPR, allowed for this class (§5.1) | 4 | shipped, both RDNA targets; the 461 above byte-identical |
| **Total, gfx11-generic** | | **465** | |
| SDPA prefill schedules (§5.1) | not an axis: `block_m` 64 for `head_dim` 64, non-causal 128 without a bias and the generic tier up to 64, and head-dim pairs (`hd_pair`) for 256 and the generic tiers above 128, replace the schedules of the same classes | ±0 | shipped; the 66 objects whose schedule did not change rebuilt byte-identical with kernel modification 18 in the source |
| **gfx12-generic** | the same table but its dropout twins, built from the same sources for RDNA4 | **417** | shipped; GPU suites pass on gfx1201 |
| SDPA forward, AOTriton's gfx1201 kernel (§5.6) | dtype (2) × causal (2) × {64, 128, 256 each plain, with a mask in Q's dtype, with an f32 mask; sinks at 64 and 128; MLA 192/128; padded tiers 64, 96, 128, 160, 224, 256} | 72 | shipped, `gfx12-generic` only; GPU suites pass on gfx1201 |
| SDPA forward, AOTriton's kernel, a mask on the padded tiers | a mask in Q's dtype and an f32 one beside each of the 24 padded-tier objects (`has_bias`, `bias_f32`); their VGPR spills are allowed per class and recorded (§5.8) | 48 | shipped, `gfx12-generic` only; the 72 above byte-identical |
| SDPA forward, AOTriton's kernel, dropout (§5.10) | a dropout twin (`has_dropout`, AOTriton's `ENABLE_DROPOUT`, Philox per score) of dtype (2) × causal (2) × `head_dim` {64, 128, 256}, no mask, no sink; 256 spills 4–6 VGPRs over its plain twin, allowed per class | 12 | shipped, `gfx12-generic` only; the 120 above byte-identical (the kernel's argument list already carried dropout) |
| SDPA forward, AOTriton's kernel, head dims past 256 (§5.6) | padded tiers up to 384 and 512 (the tuning table's 384 and 512 schedules) × dtype (2) × causal (2) × {plain, a mask in Q's dtype, an f32 one}; they spill 62-141 VGPRs, allowed per class (§5.8) | 24 | shipped, `gfx12-generic` only; the 132 above byte-identical |
| SDPA forward, AOTriton's kernel, a sink with a mask (§5.6) | a mask in Q's dtype and an f32 one beside each of the 8 sink objects (`has_sink` with `has_bias`, `bias_f32`); `head_dim` 128 spills 2-4 VGPRs, within the mask objects' allowances | 16 | shipped, `gfx12-generic` only; the 156 above byte-identical |
| SDPA forward, AOTriton's kernel, dropout with a mask (R1) | a mask in Q's dtype and an f32 one beside each of the 12 dropout twins (`has_dropout` with `has_bias`, `bias_f32`); 256 spills 34-52 VGPRs, allowed per class (`_DROPOUT_BIAS_SPILL_ALLOWANCE`), 64 and 128 none | 24 | shipped, `gfx12-generic` only; the 172 above byte-identical |
| **Total, gfx12-generic** | | **613** | |
| **Total, both families** | | **1078** | |
| SDPA forward, CDNA (§7) | per arch: dtype (2) × `head_dim` × causal (2), the kernel family fixed by arch and `head_dim`; the generic non-causal class also by `kv_align`, the dual-wave causal class by `any_diagonal`, the dual-wave classes by `has_bias`; companion tiers (§7.6); the MLA objects by `head_dim_v` and `qk_mask` | 100 on gfx950, 102 on gfx942, 102 on gfx90a | shipped; GPU suites pass on all three |
| SDPA forward, CDNA, a bias on the bucket objects | one bias object beside each bucket class's default tier (`has_bias`) | 12 on gfx950, 12 on gfx942, 28 on gfx90a | shipped; the objects without it compile to identical instructions, but the generic kernel's causal prefill ones (modification 30, below) |
| SDPA forward, CDNA, a bias read one key at a time | dtype (2) × bucket {64, 128, 256} × causal (2), `bias_kstrided` (modification 31; f32 or the query's dtype at runtime) | 12 per arch | shipped; the generic kernel takes two more arguments, appended to every object's list, and the objects without it compile to identical instructions |
| SDPA forward, CDNA, a sink with a bias | one bias object beside each prefill sink object (`has_bias` × `has_sink`) | 8 per arch | shipped |
| SDPA forward, CDNA, diagonals past the keys | not an axis: the generic kernel's causal prefill objects clamp each row's diagonal (modification 30) | ±0 | shipped; every generic causal prefill object changes by the clamp (§7.5); the decode objects compile to identical instructions |
| SDPA forward and decode, CDNA, padding (§7.8) | not an axis: the generic kernel reads per-batch lengths behind a flag (modification 32), and its argument list grows once for packed sequences and dropout | ±0 | shipped; every generic prefill and decode object changes (§7.5); the dual-wave objects byte-identical |
| SDPA forward, CDNA, dropout (§7.9) | dtype (2) × causal (2) × `head_dim` {64, 128, 256}, `has_dropout` (modification 33), no bias, no sink, a generic prefill tier | 12 per arch | shipped; the objects without it byte-identical (modification 32 already carried the arguments) |
| SDPA forward, CDNA, exact f32 (§7.11) | `f32` × {buckets 64 and 128; 256 exact, or a bucket on gfx950} × causal (2), generic prefill objects (modification 35), no bias, sink or dropout | 6 per arch | shipped; every 16-bit object byte-identical |
| SDPA forward, CDNA, packed THD (§7.10) | not an axis: the generic prefill kernel reads the row-offset tables behind a flag (modification 34), through arguments modification 32 appended | ±0 | shipped; every generic prefill and decode object changes (§7.5; the decode ones only by the keyless-view fix, modification 34); the dual-wave objects byte-identical |
| SDPA forward, CDNA, a sink with a strided bias | dtype (2) × `head_dim` {64, 128} × causal (2): a strided build (`bias_kstrided`) of each prefill sink-and-bias tier, ranked below it | 8 per arch | shipped; every existing object byte-identical |
| SDPA forward, CDNA, dropout with a bias (§7.9) | dtype (2) × causal (2) × `head_dim` {64, 128, 256}, `has_dropout` × `has_bias` (vectorized), each dropout object's tier | 12 per arch | shipped; every existing object byte-identical |
| SDPA forward, CDNA, exact 512/512 prefill | dtype (2) × causal (2), `head_dim` = `head_dim_v` 512, O split four ways (`dv_split` 4, modification 16), Q streamed past 256; K chunked by 256 on gfx942 and gfx90a (modification 18) | 4 per arch | shipped; every existing object byte-identical; the decode engine's wide objects serve 512/512 too (both candidates, autotune picks) |
| **Total, CDNA** | | **174 on gfx950, 176 on gfx942, 192 on gfx90a** | |

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
| every arch, a bias at any other multiple of 8 | up to each bucket's width | the bucket's default tier with the bias (`has_bias`); on gfx942 and gfx90a at 64 without three waves per SIMD, which spill with it | the same |
| every arch, a bias the vectorized objects do not read (f32, broadcast along the keys, rows off the 16-byte vectors) | up to 64, 128, 256 (Q, K and V the graph's width) | the bucket's bias tier reading the bias one key at a time (`bias_kstrided`; gfx950 128 at 128 rows, one wave per SIMD), `priority` 20 below the bucket's; it serves every bias, ranked below the vectorized objects | the same |
| every arch, attention sink | 64, 128 | `generic` (gfx950 64 rows; gfx942 and gfx90a gfx942's default 128-row tier) with the sink (`has_sink`), and the split, packed decode tier with it | the same |
| every arch, attention sink with a bias | 64, 128 | the prefill sink tier with the bias (gfx942 and gfx90a at 64 without three waves per SIMD; gfx950 causal 128 at 128 rows, one wave per SIMD) | the same |
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
| dtype (`bf16`, `f16`; `f32` on the generic prefill objects, §7.11) | batch |
| `head_dim` (64, 128, 256), or a bucket (64, 128, 256; on gfx90a also 96, 160, 192, 224; 1024 or 768 for absorbed MLA) whose Q/K width is runtime (`qk_mask`) | query and key lengths, independently; the Q/K head dim under `qk_mask` |
| V's head dim (`head_dim_v`: `head_dim`'s, 128 or 512 on the MLA objects, or following Q/K's on a bucket object) | V's strides, apart from K's (read through them on the `qk_mask` and MLA objects) |
| causal, or none | the causal diagonal: top-left, bottom-right, a band to its right; a sliding window's lower edge (generic kernel) |
| | query and KV head counts (MHA, GQA, MQA) |
| | each batch's own query and key lengths under padding (generic kernel, §7.8) |
| | every operand's (batch, sequence, head) strides: BSHD, BHSD, packed QKV |
| | softmax scale |
| | whether to write the LSE output, and its (batch, head, query) strides (generic kernel) |
| the schedule: kernel, `block_m`, waves per EU, setprio, KV global prefetch -- a class's tiers (§7.6) | which tier serves a graph, by its size (the score) or by autotune |
| generic non-causal: whether the KV tail is masked (`kv_align`) | |
| gfx950 d128 causal: whether any diagonal is served (`any_diagonal`) | |
| whether an additive bias is added (`has_bias`) | the bias's (batch, head, query) strides |
| whether the bias is read one key at a time (`bias_kstrided`, generic bucket objects) | its key stride, and whether it is f32 or in the query's dtype |
| whether an attention sink is added (`has_sink`) | the sink's head stride |
| whether attention probabilities are dropped (`has_dropout`, generic prefill objects) | the probability (as the keep threshold and the scale), the Philox seed and offset |
| decode: keys split, the GQA group packed into rows (`split_kv`, `pack_gqa`); O's columns split (`dv_split`); K staged in column chunks (`k_chunk`) | how many splits |

174 objects on gfx950, 176 on gfx942 and 192 on gfx90a. A class -- (dtype, `head_dim`, causal,
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
| Q, K, V, O rank 4, device operands, dense or packed THD (§7.10), each declaring 16-byte alignment; head-dim stride 1; every other stride on an axis longer than 1 positive and a multiple of 8 elements; one (batch, head) slice of each within 32-bit addressing | The batch, sequence and head strides are kernel arguments, so BSHD, BHSD and packed-QKV views are served; each lane moves 16-byte vectors; each slice is one buffer view, whose record count is 32-bit. V with strides other than K's is served by the `qk_mask` and MLA objects, whose masked load path reads V through its own; the generic kernel's direct-to-LDS path and the dual-wave kernel read V through K's descriptor, so those objects decline it. |
| One dtype for all four, `bf16` or `f16`, or `f32` (§7.11) | One object per dtype. f32 runs on the generic prefill objects only, with `mma_core_mode` unset or `FLOAT`, and without a bias, a sink or dropout; a 16-bit graph asking for a `FLOAT` MMA declines. |
| `Hq % Hk == 0`; V shaped as K but for a head dim at most K's, O as Q with V's head dim | The kernels do not broadcast; V's head dim is its own only on the MLA objects (modification 13). |
| `head_dim` equal to an object's baked value, or, on a `qk_mask` object, a multiple of 8 up to its bucket; V's head dim equal to the object's `head_dim_v`, or Q/K's on a bucket object | Checked by the kernel matcher against each candidate's metadata. A masked K or V load zeroes whole 8-column halves. |
| No mask, or `right_bound` ≥ 0 at either named `diagonal_alignment` (or one of the deprecated causal booleans: right 0 unless the graph sets one, `causal_mask` at the stored alignment, `causal_mask_bottom_right` always bottom-right; an unnamed alignment declines either way); a `left_bound` (−1, or L ≥ 1) only with a right one | The kernels apply the reference's rule, key `j` visible from query row `i` iff `i + align − L < j ≤ i + offset`, with `align` 0 top-left and `Skv − Sq` bottom-right (a `left_bound` L keeps L keys up to and including the diagonal, cuDNN and SEM-001, so the kernel's window edge is `align − (L − 1)`, and 0 declines), so both alignments are served at any lengths and a band of any width. A row whose diagonal passes the last key -- top-left with more queries than keys, a band wider than `Skv − Sq` -- sees every key: the generic kernel's prefill objects clamp each row's diagonal to the last key, so the last KV tile's padding stays masked (modification 30); an offset past `Skv − 1` is stored as `Skv − 1`, which shows every row every key. The dual-wave kernel and the split decode objects mask that padding with their diagonal compare alone, so they serve only diagonals inside the keys (offset at most `Skv − Sq`), and the generic prefill tiers take the rest, at any query length. A diagonal at `Skv − 1` or past it with no window shows every row every key, so it binds as no mask. With more than one query row, every engine's non-causal objects then serve it, dual-wave and decode included; at one query row such a bottom-right diagonal is inside the keys and stays causal. Under padding or packing this holds per batch. A sliding window (`left_bound`) runs on the generic kernel only (modification 11) -- on gfx950 its causal tiers beside the dual-wave objects; a window no row reaches is no window. |
| Scale: `attn_scale_value` (> 0 and ≤ 2²⁰), a pass-by-value scalar tensor, or absent | Absent is 1.0, cuDNN's default, which hipDNN's references apply too. The same range as the RDNA engines (`FlydslSdpaShared.hpp` `MAX_SOFTMAX_SCALE`): positive, because the running max is over unscaled scores, and at most 2²⁰, below where the causal objects' running-max floor overflows f32. A pass-by-value scale is read from host memory at launch, and one outside the range is refused there. |
| Key length a multiple of 64, for a `kv_align` 64 object | That object does not mask the last KV tile. |
| Stats, if requested: a named f32 `[B, H, Sq, 1]` device tensor, each axis longer than 1 at a positive stride, every element within 32-bit offsets | The kernels write the natural-log LSE of the scaled scores. The generic kernel and its merge take its strides (modification 20); the dual-wave kernel writes one packed `[B, H, Sq]`, so a strided LSE runs on gfx950's generic companions. A row no key reaches gets −inf (or the sink); a row whose softmax sum is NaN (a NaN or +inf score) gets NaN, as PyTorch's does, and a decode split left NaN carries NaN into the merged O and LSE rather than being skipped (modification 29). |
| Bias (`attn_mask`), if present: in the query's dtype or f32, declaring 16-byte alignment, rank 1–4 right-aligned to `[B, H, Sq, Skv]`, any axis broadcast (extent 1, absent, or stride 0), every stride a non-negative int32; one (batch, head) slice, `(Sq − 1) · stride_q + (Skv − 1) · stride_k + 1` elements, within 32-bit byte offsets | Added to the scaled scores, as PyTorch adds a float `attn_mask`. The vectorized bias objects -- both kernels' (the generic one's by modification 12) -- read a bias in the query's dtype whose keys are contiguous, every stride a multiple of 8 elements, the query stride at least `Skv` (or 0) and the key length a multiple of 4: rows move as 16-byte vectors and keys in runs of 4, and a run straddling the view's end could read as zero. Every other bias -- f32, broadcast along the keys, a key length or row pitch off those vectors, rows that overlap -- runs on the generic kernel's strided bias objects (`bias_kstrided`, modification 31), bucket objects at 64, 128 and 256 that read it one key at a time at a runtime key stride and element type; they read the others too, ranked below. A bias build softmaxes over `q·k · scale + bias`, as PyTorch, hipDNN's CPU reference and the RDNA kernels do (modification 28): a bias that masks whole KV tiles with −inf is served, a row it masks with −inf on every key gets O = 0 and LSE −inf, and a row masked on every key with the dtype's lowest *finite* value (f32's −FLT_MAX included: the strided objects keep their scores halved, modification 31) averages V uniformly over the keys the causal mask leaves. |
| Attention sink, if present: one logit per query head in the query's dtype, rank 1 `[Hq]` or rank 4 `[1, Hq, 1, 1]`, a device operand, its head stride positive (unless one head) with `(Hq − 1) × stride` within int32, at `head_dim` 64 or 128; with any bias | Served by the generic kernel's `has_sink` objects (modification 19), with a bias by their bias builds (prefill tiers only: the decode tier packs the GQA group into rows, which addresses no per-head bias); the dual-wave kernel builds none. A sink with a bias only the strided objects read (f32, broadcast along the keys, off the 16-byte rows) runs on strided builds of those sink-and-bias tiers (`bias_kstrided`, modification 31), ranked below the vectorized ones. The sink joins against the max over the row's scores and itself, prefill and decode merge alike, so a sink far above the scores gives O ≈ 0 and LSE = the sink, as PyTorch and the RDNA kernels do (modification 29). |
| Padding, if present: both per-batch lengths, with or without `padding_mask` (§7.8) | Served on the generic prefill and decode objects (modification 32); the dual-wave and `kv_align` 64 objects decline it, and so do the decode objects under a top-left causal mask. |
| Dropout, if present: a probability in [0, 1) with an INT64 device or INT64/INT32 by-value seed and offset (§7.9); above 0 without a sink, padded or not, with or without a bias the vectorized objects read | Above 0, served by the generic `has_dropout` objects (modification 33) at `head_dim` 64, 128, 256, and their bias builds; 0 is the undropped result every object computes. |
| Packed THD, if present: compact, Q/O and K/V ragged through INT32 `[B + 1, 1, 1, 1]` token tables, the LSE packed through Q's, optionally with both per-batch lengths (P3); without a bias or dropout (§7.10) | Served on the generic prefill objects (modification 34); the dual-wave, decode and `kv_align` 64 objects decline it. |
| Declined outright: the dropout mask-and-scale form and the RNG dump, paged KV, block masks, FP8 descales, ALiBi, the split max / sum-exp stats, a non-`AUTO` implementation | The objects are built without them. |
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
| ~~`Sq ≠ Skv`, both causal alignments, right-side bands~~ | Done: the diagonal offset is a kernel argument (modification 5). Top-left with more queries than keys and bands of any width: the generic prefill objects clamp each row's diagonal to the last key (modification 30); the dual-wave kernel's pipeline and the split decode objects mask their padding with the diagonal, so the generic prefill tiers serve those. |
| ~~Sliding windows~~ | Done on the generic kernel: a runtime lower edge in the mask and the KV loop's start (modification 11). The dual-wave kernel's pipeline starts at a fixed tile; gfx950 windows run on its generic tiers. |
| ~~`head_dim` 64 and 256, f16~~ | Done: rows in the instance table. Other head dims need the generic kernel's row loads generalised past powers of two. 256 on gfx942 followed with unpadded LDS tiles (modification 9). |
| ~~Padding~~ | Done on the generic kernel (modification 32, §7.8): per-batch lengths in upstream's cu_seqlens slots, read behind a flag; padded rows O = 0, LSE = −inf. |
| ~~Packed THD~~ | Done on the generic prefill objects (modification 34, §7.10): the row-offset tables behind a flag bit, each sequence's views from its first token. The decode objects and the merge kernel follow when a packed decode graph needs them. |
| ~~fp32 I/O~~ | Done on the generic prefill objects (modification 35, §7.11): exact f32 MFMA on every arch, the 64 and 128 buckets and 256 (a bucket on gfx950). Bias, sink, dropout and split decode in f32 follow when a model needs them. |
| Paged KV | The generic kernel takes a block table, passed unused today. |
| ~~Additive bias~~ | Done: the dual-wave kernel's at `head_dim` 64 and 128 on gfx950 (modification 8), and the generic kernel's (modification 12) at 64, 128 and 256 on gfx90a, gfx942 and gfx950; a bias that masks whole tiles or rows gives what PyTorch gives (modification 28); on the bucket objects, so at every multiple of 8 up to 256; and an f32 bias, one broadcast along the keys and rows off the 16-byte vectors on strided bucket objects at 64, 128 and 256 that read it one key at a time (modification 31). |
| ~~Attention sinks~~ | Done on the generic kernel at head dims 64 and 128 (modification 19), a baked axis like the bias: a per-query-head logit in the query's dtype, rank 1 `[Hq]` or rank 4 `[1, Hq, 1, 1]`, in the softmax's denominator and the LSE, taken against the max over the scores and the sink (modification 29); with a bias the vectorized objects read, on the sink tiers' bias builds. hipDNN's CPU reference declines sinks; the GPU tests check them against their own reference. |
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
| Every staged object the matcher admits computes attention — non-causal and causal, ragged tails, a longer sequence, a scale value, a runtime scale tensor, MHA, MQA and an odd group size, cross-attention, chunked prefill, decode, rows no key reaches, a top-left band, diagonals that pass the keys, `head_dim` 64 and 256, f16, BHSD and packed QKV, a per-head bias and a padding mask, a bias at the bucket head dims, an f32 bias, one broadcast along the keys or laid out off the 16-byte rows, a bias with a sink, padding (§7.8), dropout (§7.9), packed THD (§7.10), exact f32 (§7.11) — and the LSE output, against a reference | `TestGpuFlydslSdpaCdnaDispatch` (and `Full/TestGpuFlydslSdpaCdnaFullTier`, below), on gfx950, gfx942 and gfx90a |

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
| Exact f32 objects (modification 35, new objects only; 5 rounds, every 16-bit object byte-identical so no A/B) | against the fastest 16-bit object on the same shape (B1 H32/8 at 4096 and B4 H16/4 at 1024 tokens; `head_dim` 64, 96, 128, 256; causal and not): gfx942 (an MI325X) 2.2-6.1x the time, f32 at 36-111 TFLOP/s by the useful FLOPs (at 256 the split objects run the QK product twice); gfx950 6.0-9.0x, 42-114 TFLOP/s, against its dual-wave bf16 objects. The f32 MFMA's peak is an eighth (gfx942) and a sixteenth (gfx950) of bf16's, so the f32 objects run nearer their peak than the 16-bit ones do. gfx950's tiles from timing the alternatives that hold: causal 64 and 128 on 128 rows (+14-21%), the unsplit 256 bucket against `dv_split` 2 (+12-19%) | shipped; f32 tile selection **deferred** (item 19) |
| Diagonals that pass the keys (modification 30): a key cap in the masked tiles of every generic causal prefill object | against the objects before it, 13 interleaved rounds per shape (116 shape rows on gfx950 and gfx942, 156 on gfx90a): gfx950 median 0.0%, none slower by more than 1.5%, the `head_dim` 128 causal 64-row tile 7-8% faster at one query row; gfx942 median 0.0%, every row within 1.2%; gfx90a median 0.0%, the 160 and 64 buckets' causal tiles 2-3% slower, the 224 bucket's 2-3% faster, the `head_dim` 64 causal bias tile 5% slower at the long prefill shape and 5-11% faster at the decode shapes (158 to 192 VGPRs). gfx950's `head_dim` 128 causal 64-row tile spills 4 VGPRs where it spilled 2. The decode objects keep the plain compare and compile to the same instructions | kept, for cross-family parity -- the gfx90a costs **deferred** |
| A bias on the bucket objects | new objects; a full bias at `head_dim` 128 runs 11-15% slower on the 128 bucket than on the exact bias objects (B1 H32 S4096: gfx950 755 against 669 µs, gfx942 1452 against 1260, gfx90a 5414 against 4890) -- the bucket's masked loads | shipped |
| The strided bias objects (`bias_kstrided`, modification 31) | new objects. An f32 full bias at B1 H32 S4096 against the vectorized objects' bf16 one at the same head dim: `head_dim` 128 1.7x on gfx950 (1147 against 669 µs), 1.9x on gfx942 and gfx90a; `head_dim` 64 2.5x, 2.6x and 4.4x -- one load per score. The two arguments every generic object takes leave the others' instructions unchanged | shipped, for coverage -- **deferred** |
| A bias with a sink | new objects, the sink tiers' bias builds | shipped |
| Padding, dropout and packed THD (modifications 32-34), dense graphs | 9 interleaved rounds per arch, the objects before modification 32 against these. gfx942, 249 dense rows (prefill, decode, bias, sink): median +0.3%, geomean +0.5%, range −1.1% to +3.6%. gfx950, 248 rows: median +0.3%, geomean +0.6%, range −1.9% to +7.0%. gfx90a, 362 rows: median +0.2%, geomean +0.4%, range −4.7% to +4.8%. Every KV loop is instruction-identical between modifications 32 and 34; the outliers are scheduling and prologue. gfx942's `head_dim` 128 non-causal 128-row bias object is +3.2-3.6% on its three shapes: its KV loop now packs 22 bias adds and multiplies into `v_pk_*`, with 33 more hazard-wait cycles per iteration. The old object spilled 4 SGPRs inside that loop and the new one does not. gfx950's `head_dim` 128 causal 64-row tiles are +6.7% at one query row (item 14's objects; the decode objects serve that shape several times faster). gfx950's `head_dim` 128 non-causal 64-row bias object is +3.7% at S4096; its loop has the same size and waits as before, but with registers reassigned. Split decode objects at the shortest launches (12-40 µs) are +2-4% on every arch. The dual-wave objects are byte-identical and moved −2% to +3%, which is the noise floor on short launches | runtime, for coverage -- **deferred** (item 18) |
| Padding, packed THD, dropout: the feature's own cost (the same rounds) | Padded at full lengths against the same object dense: median +0.4% (gfx90a), +0.5% (gfx942), +1.2% (gfx950); at most +5.6-8.9%, on the shortest decode launches (two length loads per workgroup). Packed at full lengths against dense: median +0.3%, +0.4%, +1.0%; at most +3.4-7.2%, on B1 S256 launches of 10-25 µs. Dropout objects at p = 0.1 against the fastest plain object at the same shape: gfx942 1.43-2.11x (median 1.75x), gfx90a 1.20-1.96x (1.57x), gfx950 1.62-3.54x (2.76x; there the fastest plain object is the dual-wave kernel's). The cost is one Philox4x32-10 per 4 scores | shipped |

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
    autotune picks there. With modification 30's key cap in place they measure 7-8% faster
    at that shape than before it, so about half of it is back; the cause is still not
    found.
15. *The strided bias objects* (modification 31). Each score loads its own bias element,
    so an f32 bias, or one broadcast along the keys, runs well below the vectorized
    objects' speed. Measured against the vectorized bucket bias objects at the same
    shapes (13 interleaved rounds, bf16, prefill S1024-S4096 and decode): gfx950 about
    1.2-1.5x slower at `head_dim` 96/128; gfx942 1.5-2.6x at `head_dim` 56/64 and 4-5.4x at
    96/128; gfx90a 1.8-3.5x at 56/64 and 2.8-3.9x at 96/128. Decode shapes cost least
    (1.1-2x, except gfx942 `head_dim` 128 at about 4x). An f32 bias costs more than a
    query-dtype one at `head_dim` 64 on every arch. An f32 bias with contiguous keys could take 16-byte loads as the
    bf16 objects do (an f32 vectorized bias object, or a unit-stride fast path inside the
    strided one for rows that stay aligned); a bias broadcast along the keys needs one load
    per row, not per score; and the bias could be staged through LDS with its K tile
    (item 7).
16. ~~*A sink with a bias the vectorized objects do not read*~~ Done: a strided build of
    each prefill sink-and-bias tier, 8 objects per arch (§6.2).
17. *gfx90a under modification 30.* The 160 and 64 buckets' causal tiles are 2-3% slower,
    and the `head_dim` 64 causal bias tile went from 158 to 192 VGPRs (5% slower at the long
    prefill shape, faster at decode shapes): a gfx90a tile retuned for the key cap, or a
    cheaper cap, are the candidates.
18. *Modifications 32-34 on dense graphs.* Two things to look at. First, gfx942's
    `head_dim` 128 non-causal 128-row bias object (+3.2-3.6%): the packed bias arithmetic's
    hazard waits are new in its loop. A schedule that keeps the scalar form would fix it, or
    a different SLP decision for that loop. gfx950's `head_dim` 128 non-causal 64-row bias
    object (+3.7%) has its loop's registers reassigned the same way. Second, the split decode
    objects' prologue on the shortest launches (+2-4%): the per-batch length flag and the
    keyless-view test, read behind uniform branches. Folding the flag tests into one branch, or a varlen-free decode build for
    dense graphs, would remove most of it. Dropout's own cost (1.2-3.5x at p = 0.1) is the
    Philox draws; amortising them over the 4 scores each draw covers is already done, so a
    cheaper generator (fewer rounds, as some frameworks allow) is the remaining lever, and
    it would break bit parity with AOTriton.
19. *Exact f32 (modification 35).* The first wave's tiles were picked by occupancy and a
    short timing (gfx942 and gfx950, a few tiles each), not the full AOT selection the 16-bit
    set went through (§7.6). On gfx942 and gfx90a the 256 objects split O's columns, so each
    half recomputes the QK product: gfx950's unsplit 256 measured 12-19% faster than its split
    one, and a gfx942 tile that holds 256 whole would need a smaller K/V staging than 64 KiB
    allows today. The f32 kernel reuses the 16-bit K-step's pack layout (four 32x32x2 MFMAs
    per 4-element pack); a layout built for f32 (fewer, wider LDS reads) and the bank-conflict
    padding for 4-byte elements were not explored.

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
and ships what gfx942 ships but the dual-wave engine: `hipkernel:flydsl_sdpa_cdna` (138
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

### 7.8 Varlen and padding (CDNA)

Padding as §5.9 states it (`padding_mask` with `seq_len_q` and `seq_len_kv`, INT32 `[B, 1, 1, 1]`,
one length per batch, over dense Q/K/V/O sized by the longest sequence) is served on the generic
kernel's prefill and decode objects, which read each batch's lengths at runtime (vendored-kernel
modification 32): 0 new objects.

| | Generic prefill | Split decode | Dual-wave (gfx950) |
|---|---|---|---|
| Padding, any mask | **served**: causal at either corner, bands, windows, bias (vectorized or strided), sink, LSE, every head-dim tier | **served**: no mask, or bottom-right causal (its diagonal stays inside every batch's keys); top-left causal declines | declines (reads no lengths); gfx950's generic tiers serve it |
| `kv_align` 64 objects | decline (no KV tail mask); the class's masked default tier serves | | |
| Packed THD (ragged offsets) | **served** (§7.10) | declined | declined |

- **Rows past a batch's query length: O = 0 and LSE = −inf**, hipDNN's reference contract (cuDNN ≥
  9.14), on sink objects too: a padded row is padding, not a row no key reached. The fill is
  what a padded launch does; there is no "skip" switch on CDNA.
- **Keys past a batch's key length are never read**, and masked by the dense tail's own compare;
  a batch with no key gives every row O = 0 and LSE = −inf (or the sink).
- **The mask, per batch:** a bottom-right diagonal (and a window's edge) sits at each batch's
  own `Skv_b − Sq_b`; top-left bounds are the same literal for every batch.
- **Lengths** are clamped in the kernel to `[0, S]`, as hipDNN's GPU reference clamps them.
  **Lengths without `padding_mask`** (both of them) are padding too, as hipDNN's references read
  them (their SEM-008); one alone declines. INT64, a batch stride other than 1, pass-by-value,
  virtual or ragged lengths, and a base not declared 4-byte aligned decline.
- **With dropout** (§7.9): served; each batch draws its mask at the graph's declared extents, so a
  padded batch's real rows get the dense graph's bits.
- **A bias under padding runs on the strided objects** (`bias_kstrided`, modification 31), sink
  included. A batch's key length need not be a multiple of the vectorized objects' 4-key runs. The
  last real row's final run would then straddle the batch's bias view and read as zero; found on
  the device at `Skv_b` 63. Dropout with a bias under padding declines, because no dropout object
  reads a bias one key at a time. A kernel fix, giving the last row's view the launch's key width,
  would restore the vectorized objects here.
- **Cost:** a padded launch loads two lengths per workgroup behind a flag; a dense launch loads
  nothing. Measured in §7.5.
- **Tested** against a test-local per-batch reference until hipDNN's serves padding
  (`TestGpuFlydslSdpaCdnaDispatch.Pads*`, `DecodesPaddedBatchesWithSplitKeys`,
  `ClampsLengthsOutsideTheSequence`, `GivesABatchWithNoKeysRowsNoKeyReaches`): lengths 0, 1, odd,
  full, past `S` and negative; batches with more queries than keys; both corners, a band and a
  window; GQA, BHSD, f16, a bucket head dim, a bias in the query's dtype and f32, a sink; decode
  over caches of different lengths with splits and the merge; O and the LSE filled with NaN
  beforehand. Every gate: `Padding/TestFlydslSdpaCdnaFieldDeclines`,
  `TestFlydslSdpaCdnaKernelMatch.RunsPaddingOnlyOnObjectsThatReadEachBatchsLengths`,
  `RanksAPaddedGraphOnTheMaskedDefaultTier`.

### 7.9 Dropout (CDNA)

Dropout as §5.10 states it (`dropout_probability` with a seed and an offset tensor) is served on 12
`has_dropout` objects per CDNA arch (vendored-kernel modification 33): bf16/f16 × causal or not ×
`head_dim` 64, 128, 256, no bias, no sink, generic prefill tiers.

| | Generic prefill | Split decode | Dual-wave (gfx950) |
|---|---|---|---|
| Probability in (0, 1) with seed and offset | **served** by the `has_dropout` objects: either causal corner, bands and windows, GQA, the LSE | declines (inference; a dropping graph with one query row runs on the prefill objects) | declines; gfx950's generic dropout objects serve it |
| Probability 0 | the plain objects (the undropped result) | the plain objects | the plain objects |

- **The same mask as the aiw objects and AOTriton (≥ 0.15), bit for bit:** Philox4x32-10 (the aiw
  kernel's round, restated in the CDNA kernel so its objects digest only their own sources), the
  same counter layout, word order and f32 threshold (§5.10). The provider's tests run the aiw
  objects' own mask cases on the CDNA objects (`DropsTheOraclesMaskBitForBit`, the same graphs, seed
  and offset), both against one host oracle.
- **Applied after the softmax's row sum**, so `l` and the LSE are undropped; `1 / (1 − p)` folds into
  O's normaliser.
- **Seed and offset:** as §5.10. The CDNA kernel reads both through pointers, so a seed given by
  value is written to the engine's workspace (two memsets on the launch stream) and an offset given
  by value rides `offset2` beside 8 zeroed workspace bytes read as `offset1`: 16 bytes of
  workspace when either is a host value.
- **Under padding** (§7.8) the dropout objects draw each batch's mask at the graph's declared
  extents (`Sq`, `Skv` the padded maxima, `i`, `j` positions within the sequence), as AOTriton's
  varlen does: a padded batch's real rows get the dense graph's bits, and its padded rows O = 0,
  LSE = −inf.
- **With a bias** (the second wave): a bias build of each dropout object, 12 per arch. The bias is
  in the query's dtype, read as the vectorized objects read it. It is added to the scores, and the
  mask is drawn after the softmax's sum, as AOTriton composes them. Every dropout tier holds the
  bias without a spill. A bias only the strided objects read (f32, broadcast along the keys) finds
  no dropout object and declines at kernel match.
- **Declined**, each a row of objects when a model needs it: dropout with a sink, with a strided
  bias, with a bias under padding (§7.8), at bucket head dims, MLA, on the dual-wave and decode objects.
- **Cost:** measured in §7.5.
- **Tested** against the test-local oracle until hipDNN's reference serves dropout
  (`TestGpuFlydslSdpaCdnaDispatch.Drops*`, `ComputesTheUndroppedResultAtProbabilityZero`): the mask
  read off O bit for bit (Q zero, V the identity) on the aiw cases' graphs, the full output and the
  undropped LSE with GQA, BHSD, both dtypes, a key length no multiple of 4 or of the KV tile, a seed
  and offset by value (INT64 and INT32) and on the device; under padding, the dense mask bit for bit
  and the full output with a batch of no keys and one of no queries. Every gate:
  `Dropout/TestFlydslSdpaCdnaFieldDeclines`, `TestFlydslSdpaCdnaKernelMatch.RunsADroppingGraphOnlyOnTheDropoutObjects`,
  `RunsAPaddedDroppingGraphOnTheDropoutObjects`.

### 7.10 Packed THD (CDNA)

Compact THD is served on the generic prefill objects, which read the row-offset tables at runtime
(vendored-kernel modification 34). This is hipDNN's ragged reference's form and the one the aiw
objects serve (§5.9, `servablePackedVarlen`): 0 new objects. Q and O are ragged through one INT32
`[B + 1, 1, 1, 1]` device table, and K and V through one (the same one or another). Each table holds
cumulative token counts from 0. Each multiplier is the tensor's own sequence stride, so the offsets
count tokens. The layout is token-major.

| | Generic prefill | Split decode | Dual-wave (gfx950) |
|---|---|---|---|
| Packed THD | **served**: any mask (a bottom-right diagonal per sequence), GQA, a sink, the LSE packed through Q's table, every head-dim tier, `bf16`/`f16` | declines (indexes by batch; a packed graph with one query row runs on the prefill objects) | declines; gfx950's generic tiers serve it |
| `kv_align` 64 objects | decline (no KV tail mask) | | |

- **Each sequence is its own problem:** its views start at its first token and span its own token
  count, clamped in the kernel to `[0, S]`. Nothing is read or written past the sequence's last
  token, because the next sequence's tokens begin there. A bottom-right diagonal sits at each
  sequence's own `Skv_b − Sq_b`, as under padding.
- **The LSE** must be packed through Q's own table, `[T, H]` in any stride. A dense LSE would need
  each sequence's batch base, which a packed launch does not index, so it declines. So does a
  ragged LSE beside dense operands.
- **Declined, as on the aiw objects:** an operand left dense beside ragged ones; Q and O, or K and
  V, through different tables; offsets that do not count tokens; head-major packing; INT64,
  virtual, by-value or misaligned tables; one length alone, or `padding_mask` alone, with
  packing; packing with a bias; packing with dropout.
- **With per-batch lengths (P3, cuDNN's ragged operands with `seq_len`; CDNA-local,
  `cdnaServablePackedVarlen`):** served. Both INT32 `[B, 1, 1, 1]` lengths, with or without
  `padding_mask`, beside the same packing. Each table gives a sequence's slot. The length gives
  the real rows within it, clamped in the kernel to the slot (`_load_varlen`, no kernel change).
  The slot's rows past the length are padded rows (O = 0, LSE = −inf). Its keys past the length
  are never read, and a bottom-right diagonal sits at the lengths. The aiw and RDNA engines
  decline this form for now.
- **Cost:** a dense launch takes one more scalar branch in the prologue and loads nothing. The KV
  loop is instruction-identical to modification 33's. Measured in §7.5.
- **A fix with it:** a keyless view at the tensor's start (padded batch 0, head 0; the first packed
  sequence) no longer has the gfx942 objects' pre-loop K prefetch read the row before K's base
  (modification 34). This is on the decode objects too.
- **Tested** against the test-local per-sequence reference (`TestGpuFlydslSdpaCdnaDispatch.Packs*`).
  The harness packs the dense values token-major, unpacks O and the LSE, and checks that a guard of
  `S` rows past the last sequence stays unwritten. Cases: lengths 0, 1, odd and full; a sequence
  with no keys (with and without a sink); more queries than keys and fewer, at both corners, with a
  window and a band; GQA; f16; one table for every operand; no LSE; a bucket head dim; and
  single-token sequences. Every gate: `Packed/TestFlydslSdpaCdnaFieldDeclines`,
  `TestFlydslSdpaCdnaPackedGraph`,
  `TestFlydslSdpaCdnaKernelMatch.RunsPackedSequencesOnlyOnTheGenericPrefillObjects`. P3 on the device:
  `PacksPaddedSequencesInTheirSlots` (lengths 0, short, full, past the slot; with and without
  `padding_mask`; both corners; a sink).

### 7.11 Exact f32 (CDNA)

AOTriton serves fp32 attention on every arch it ships, as exact fp32 (`TRITON_F32_DEFAULT=ieee`).
The generic CDNA kernel does too (vendored-kernel modification 35): f32 products and f32
accumulation through `v_mfma_f32_32x32x2f32`, four per 16-bit K-step, with P kept in f32 and f32
denormals kept (no DAZ). It runs the gfx942-class path on every arch: the hardware transpose, K16
MFMA, permlane O store and DMA are 16-bit only. 6 objects per arch, no bias, sink or dropout:

| | gfx942, gfx90a | gfx950 |
|---|---|---|
| bucket to 64 (`qk_mask`) | non-causal 256 rows, causal 128 | non-causal 256 rows, causal 128 |
| bucket to 128 | K in 64-column chunks, 128 rows (256 rows spill 37-38 VGPRs with the bucket's masks) | non-causal: K chunked, 256 rows; causal: unchunked, 128 rows |
| 256: Q re-read | exact 256, K chunked, O's columns split in two (`dv_split` 2), 256 rows | a bucket to 256, unsplit, 128 rows, one wave per SIMD |

- **Occupancy decides the tiles.** On gfx942 and gfx90a every f32 tile fills 34-51 KiB of the 64
  KiB LDS, one workgroup per CU, so a 128-row tile puts one wave on each SIMD and a 256-row tile
  two; registers are not the limit there. On gfx950 the 160 KiB holds two to four workgroups,
  or one whole 256 tile, and the tiles were timed against the others that hold. No object
  spills. A bucket cannot split O's columns (modification 17), so on gfx942 and gfx90a f32 head
  dims from 136 to 248 have no object yet; on gfx950 the 256 bucket serves them.
- **Precision:** within 1e-5 of the double reference on unit-scale inputs, the LSE within 2e-3
  (the suites' bound; measured about 1e-6). f32 denormal inputs give denormal outputs
  (`KeepsF32DenormalInputs`).
- **The other engines:** the dual-wave kernel and the split decode objects take no f32 (the kernel
  matcher declines it on their tiles); the RDNA and aiw engines decline f32 at graph scope (RDNA has
  no f32 WMMA; the user deferred RDNA fp32). That is a coverage gap, not a behaviour difference.
- **Declined in f32 until objects serve them:** a bias, a sink, dropout, a `HALF` or `BFLOAT16`
  `mma_core_mode` (a reduced-precision MMA these objects do not run). Padding, packed THD, packed
  slots with per-batch lengths (P3) and a diagonal past the keys bound as no mask (§7.2) are
  served (the generic kernel reads them in f32 as in 16 bits).
- **Speed:** §7.5.
- **Tested:** `TestFlydslSdpaCdnaF32Graph` (accepts and declines), `TestFlydslSdpaCdnaF32KernelMatch`,
  `TestGpuFlydslSdpaCdnaDispatch.ComputesF32*` (`ComputesF32InPaddedSlotsAndAtAnEveryKeyDiagonal`
  for P3 and the rebound diagonal) and `KeepsF32DenormalInputs`, the per-object suite
  and the census (`ShipsTheF32FirstWave`); every new gate mutation-checked.

---

## 8. Known deviations from the descriptor RFCs

Approved for this integration.

| Deviation | What FlyDSL does instead | Status |
|---|---|---|
| **Two-launch objects name their second kernel in metadata, not in RFC 0017 §15.1 `sources`.** RMSNorm backward (partial, then reduce) and both split-KV decode engines (split, then merge) launch two kernels compiled into one `.hsaco`. | Each UKD's `kernel_source` names the first kernel; the KMD field `reduce_symbol` (§2a) or `merge_symbol` (§5, §7) names the second, in the same packed object. The native `prepare()` loads both and `launch()` runs them in order on the plan stream. The generators verify both symbols exist in the object, with their recorded argument layouts, before writing it; the packer and the state manager's launch-set check see only the first symbol. | This is the form RFC 0021 §3 (kernel slots, draft) names as an accepted alternative to `sources` for dependent kernels: one UKD whose code object holds them, launched by its native handler through symbol names its KMD records, with a split-KV kernel and its reduction as the example. Optional later move: `hsaco` entries in `sources`, once the packer accepts them (proposed in #14215), which would bring both kernels under the packer's and loader's launch-set checks. The objects would not change. |
