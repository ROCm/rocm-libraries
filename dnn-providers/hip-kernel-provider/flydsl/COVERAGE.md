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
Every graph a pack accepts is
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
| gamma **encodes the normalised shape** as the frontend reads it: the normalised axes are the longest trailing run where gamma's extent equals x's (never axis 0), and gamma is 1 on every axis before it | Any other shape would be a broadcast the kernel does not perform. The same rule as `validateScaleNormalizedShape` in the frontend. |
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

On success it binds `flydsl_rmsnorm.{x,scale,epsilon,y}.uid` plus `.rows` and
`.columns`, so nothing downstream re-derives them from the graph.

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
| **Architectures outside gfx11-generic, gfx12-generic, gfx90a, gfx942 and gfx950** | Objects are checked in for the gfx11 generic family (gfx1100–gfx1103, gfx1150–gfx1153), the gfx12 one (gfx1200, gfx1201), and the CDNA targets gfx90a, gfx942 and gfx950. gfx1170/gfx1171 belong to `gfx11-7-generic`. With `GPU_TARGETS` naming no arch we ship kernels for, the integration reports dormant at configure time and stages nothing — it does not fail, and it does not silently produce an empty shard. Adding a family is a regeneration ([REGEN.md](REGEN.md) §3), not a code change. |
| **Ops other than RMSNorm** | `OPS` in `_instances.py` has RMSNorm forward and backward only. |

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

So the instance key is `dtype × head_dim × causal × has_bias`, and every axis
it enumerates is **model-determined**, never request-determined. That is what
keeps the table at 32 rows — each cell below is one object without a bias and
one with:

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
narrower tier above the wider (`priority` 40, 30, 20, 15, 12, 10), so 64/96/128/256
always run their own object and the narrowest generic tier takes the rest. A bias
on a head no specialized object serves declines.

The causal variant is the one that applies a right bound — top-left or
bottom-right causal, and any band to the right of the diagonal. It is two
objects rather than a runtime flag because it changes the loop: it bounds the
KV loop at the diagonal and skips fully masked tiles, and the non-causal one
carries a V prefetch across iterations that the causal one drops for register
pressure. Everything else about the mask is a runtime argument, served by both.

**Decode** has its own fifty-six objects (`decode` 1; bf16 and f16 × causal × `head_dim`
64, 96, 128, 256 without a bias, with an f32 one and with one in the query's dtype, plus
two generic tiers), from a kernel
written for it
([`flash_attn_decode_rdna.py`](kernels_src/kernels/attention_rdna/flash_attn_decode_rdna.py)).
The prefill objects give a decode step 128 query rows of which one is used, and read
each KV head once per query head. The decode kernel packs the query heads of one KV
head (the GQA group) and the query positions into 16-row tiles, one workgroup per tile,
so a graph reads each KV head once per tile rather than once per query head. It serves
`(Hq / Hk) × Sq ≤ 128`; past that, the prefill objects, which share each K/V tile
across 128 rows, are faster. The bias objects take one tile (16 rows): at `head_dim`
128 and up they already fill the register file, and a tiled launch's row index spills. It splits the keys a row can see --
from the window's edge to the diagonal -- across workgroups, which the dispatcher
sizes to the device: two workgroups per unit the device reports, because RDNA reports
workgroup processors of two CUs each; and merges the splits by their LSEs with a
second kernel compiled into the same object. They rank above the prefill objects
(`priority` 150), so such a graph runs here. At `head_dim` 256 two workgroups share
each split's rows, one per half of the output columns (`dv_split` 2), to fit the
register file. The bias objects read the same broadcast bias
as the prefill ones, each lane its own (head, position) row of the GQA group's slice.
Every other head_dim that is a multiple of 8 runs on a generic decode object -- up to
96, or above it up to 256 with the column split -- that reads it at runtime and never
fetches the K and V columns past it, so a step moves only the bytes its head has (an
up-to-128 tier does not fit the register file). They rank above every prefill object
and below the exact ones, the narrower tier above the wider (`priority` 145, 140).
K and V with different head counts stay on the prefill objects. A split launch takes
workspace (a partial O and LSE per split); one split writes O directly.

### 5.2 What a graph must be

One SDPA-forward node, and:

| Condition | Why |
|---|---|
| Q, K, V, O present, rank 4, device operands, **not ragged** | The kernel reads each through a pointer as a dense batch; a ragged (THD) operand would be read as one. |
| One dtype for all four, `BFLOAT16` or `HALF` | One element type throughout; no fp32 WMMA operand form exists. |
| Head-dim stride 1 on all four; other strides positive | Each lane reads a row's head-dim values as one contiguous vector. Every other stride is a kernel argument, so BSHD, BHSD and packed-QKV views are all served. |
| K and V share batch and sequence; K has Q's head dim; O is Q's shape with V's head dim; V's head dim equals Q/K's, or is 128 with Q/K 192 (MLA) | No broadcast. The MLA objects bake both head dims (kernel modification 20); every other object has one. |
| `Hq % Hk == 0` and `Hq % Hv == 0`; K and V may differ in head count | The K head and the V head are each the query head divided by its own group size. |
| Any `left_bound`/`right_bound` ≥ −1, either alignment, or the deprecated causal booleans (not both) | The kernel applies the reference's two-sided rule (`CpuFpReferenceSdpa` `isMasked`) directly. |
| Scale: `attn_scale_value` (> 0), a pass-by-value scalar tensor, or absent | Absent is 1.0, cuDNN's default, which hipDNN's references apply too. The running max is over unscaled scores, which orders them correctly only for a positive scale. A pass-by-value scale is read from host memory at launch, and a non-positive or non-finite one is refused there. |
| Stats, if requested: a named f32 `[B, H, Sq, 1]` device tensor | The kernel writes the natural-log LSE of the scaled scores there. |
| Bias (`attn_mask`), if present: an f32 device tensor, or one in the query's dtype, of rank 1–4 whose dims, right-aligned to `[B, H, Sq, Skv]`, are each that extent or 1; positive strides on real axes; one (batch, head) slice under 2³⁰ elements | Added to the scaled scores before the mask, as the reference does; a size-1 or absent axis broadcasts. The kernel reads it through a 32-bit buffer descriptor with 32-bit offsets. |
| `Sq`, `Skv` < 2²⁹; one (batch, kv head) slice of K or V under 2 GiB; grid under 2³¹ | int32 mask arithmetic, and 32-bit bounds-checked buffer descriptors. |

Any sequence length is correct: K and V reads past `seq_len_kv` return zero and
every such score is masked, so nothing is padded and nothing pads the softmax.
A row no key reaches — a narrow window, or bottom-right causal with more
queries than keys — gets O = 0 and LSE = −inf, as the reference defines it.

### 5.3 Against hipDNN's attention Tier 0 / Tier 1

| # | Requirement | Status |
|---|---|---|
| 1 | fp16, bf16; fp32 accumulate | **served** |
| 2 | BSHD, BHSD, general B/S/H strides, packed QKV | **served**; packed QKV checked on the GPU as strided views into one buffer |
| 3 | MHA, MQA, GQA (`Hq % Hk == 0 && Hq % Hv == 0`) | **served**, including K and V with different head counts |
| 4 | attention scale | **served** (positive) |
| 5 | causal, top-left | **served** |
| 6 | causal, bottom-right, `Sq ≠ Skv` | **served** |
| 7 | softmax stats / LSE | **served** |
| 8 | arbitrary sequence lengths | **served** |
| 9 | head dims 64, 128, 256; `d % 8 == 0` | **served**: 64/96/128/256 specialized, every other multiple of 8 up to 256 by the generic tier (not with a bias) |
| 10 | padding mask via `SEQ_LEN_Q`/`SEQ_LEN_KV` | missing |
| 11 | sliding window (`left_bound`) | **served** (and right-side bands) |
| 12 | varlen / THD | missing (declined) |
| 13 | decode (`Sq == 1`) | **served by a decode path** (ATT-13): GQA-packed rows and split-KV, for `(Hq / Hk) × Sq ≤ 128` (MQA, speculative steps) at `head_dim` 64/96/128/256 and every other multiple of 8 up to 256 |
| — | `Dv ≠ Dqk` (MLA) | **served** for un-absorbed prefill (Q/K 192, V 128) and absorbed decode (Q/K 576, V 512, one KV head, `(Hq / Hk) × Sq ≤ 128`); every mask, LSE, no bias. Absorbed chunked prefill (more rows) declines |
| 14 | additive bias | **served**: f32 or the query's dtype, broadcast over any of B, H, Sq, Skv; with every mask, the LSE output and every specialized head dim, on the prefill and the decode objects |
| — | attention sink | **served** at `head_dim` 64 and 128 (gpt-oss), prefill and decode, with every mask and the LSE output, no bias: one logit per query head in the query's dtype, rank 1 `[Hq]` or rank 4 `[1, Hq, 1, 1]`, a key with no value in each row's softmax (kernel modification 21). A row no key reaches gets O = 0 and LSE = the sink. hipDNN's CPU reference declines sinks; the GPU tests check them against their own reference |

### 5.4 What is still missing, and what closing it takes

Each is declined today, so the graph plans on another engine or fails cleanly
at `check_support()`, never at `execute()`.

| | Effort | What it takes |
|---|---|---|
| **Padding mask** (Tier 1) | medium | Two per-batch length pointers read once per workgroup. The diagonal offset is already computed in-kernel from the lengths for exactly this. hipDNN's CPU reference does not implement padding yet, so its semantics must be defined there first. |
| **Varlen / THD** (Tier 1) | medium, after padding | Ragged offsets replace the batch stride; the grid stays over the longest sequence and tiles past a sequence's end exit early. |
| Bias on a generic head_dim | small | A `has_bias` variant of the eight generic objects (+8). |
| Absorbed MLA chunked prefill; other `Dv ≠ Dqk` pairs | medium; small per pair | 192/128 prefill (modification 20) and the 576/512 and 512/512 one-KV-head decode objects ship. Chunked prefill at 576/512 (Hq × Sq past 128 rows) needs a prefill schedule at that width, or the decode kernel's row tiles lifted past 128 for single-KV-head graphs. Other pairs are a row each. |
| fp32 I/O, FP8, dropout, paged KV, block masks, ALiBi, softcap | out of scope | As hipDNN's requirements state; ALiBi and softcap have no reference semantics. |
| **Backward** | out of scope here | Inference-only, as for hipDNN's Tier 0/1. LSE is emitted so a backward can be added without an ABI break. |
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
| Each object has the declared 36-argument layout and spills no registers | `gen_sdpa_rdna.py`, before the object is written |
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
- **Declined:** a mask whose key axis is strided; head dims that are not multiples of 8;
  anything past 32-bit offsets. The kernel also takes packed (varlen) batches and
  dropout, which this engine does not serve yet.
- **Which engine runs a graph** both this and §5's prefill engine serve is autotune's
  (UHD's) to decide: measured on gfx1201, this one is faster on MLA prefill and long
  sequences, §5's on decode and sinks.

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
   runtime, so the count does not grow with the shapes a model is run at.
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

Adding a row here, with its gate result in §6.1 where it introduces a runtime
feature, is part of adding the feature.
