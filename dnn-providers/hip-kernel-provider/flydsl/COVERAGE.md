<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# What the FlyDSL packs cover

Two packs, built once for each of two LLVM generic targets -- `gfx11-generic`
(RDNA3 / RDNA3.5: gfx1100–gfx1103, gfx1150–gfx1153) and `gfx12-generic` (RDNA4:
gfx1200, gfx1201) -- and shipped to every member: **RMSNorm forward** (12 kernel
objects per family, §1–§4) and **SDPA forward** (128 per family, §5). A third,
**SDPA forward on CDNA** (§7), is a separate engine built per arch for gfx950 and
gfx942 from FlyDSL's own CDNA kernels; it is a first, deliberately narrow object
set that proves the path end to end. Every graph a pack accepts is
computed by one of its objects; everything else is **declined**, so another
engine gets the plan.

That last point is the design rule this whole file exists to document. The
kernels are pre-built HSACO — [`FlydslRmsNormNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp)
cannot change what they compute, only decide whether they are the right thing to
run. So every condition below is enforced by *refusing*, never by adapting. A
matcher that stretched to accept a near-miss would compute the wrong answer
silently; one that declines it costs a fallback.

---

## 1. The 12 instances

Generated from [`generators/_instances.py`](generators/_instances.py), which is
the single source of truth — the compiler, the packer and the descriptor emitter
all read it, so adding an instance is a row there rather than an edit in three
places that can disagree.

| Instance | Tier | dtype | `N` | `block_threads` | priority |
|---|---|---|---|---|---|
| `rmsnorm_bf16_ngeneric_bt256` | generic | bf16 | runtime | 256 | 10 |
| `rmsnorm_f16_ngeneric_bt256` | generic | f16 | runtime | 256 | 10 |
| `rmsnorm_bf16_n3072_bt256` | specialized | bf16 | 3072 | 256 | 100 |
| `rmsnorm_bf16_n3584_bt256` | specialized | bf16 | 3584 | 256 | 100 |
| `rmsnorm_bf16_n4096_bt256` | specialized | bf16 | 4096 | 256 | 100 |
| `rmsnorm_bf16_n5120_bt256` | specialized | bf16 | 5120 | 256 | 100 |
| `rmsnorm_bf16_n8192_bt256` | specialized | bf16 | 8192 | 256 | 100 |
| `rmsnorm_f16_n3072_bt256` | specialized | f16 | 3072 | 256 | 100 |
| `rmsnorm_f16_n3584_bt256` | specialized | f16 | 3584 | 256 | 100 |
| `rmsnorm_f16_n4096_bt256` | specialized | f16 | 4096 | 256 | 100 |
| `rmsnorm_f16_n5120_bt256` | specialized | f16 | 5120 | 256 | 100 |
| `rmsnorm_f16_n8192_bt256` | specialized | f16 | 8192 | 256 | 100 |

2 dtypes × (1 generic + 5 specialized widths). `block_threads` is single-valued
today: it is launch geometry, not correctness, so a second value only earns its
6 extra objects once benchmarking says it wins somewhere. It stays in the knob
dict so the descriptors and the matcher already carry the axis when it grows.

### Why these five widths

3072 / 4096 / 5120 / 8192 are Llama-family hidden sizes; 3584 is Qwen2-7B. This
is a list of what the targeted models actually run, not a sweep.

**Widening it is cheap, and is not a coverage fix** — every other `N` is already
served, by the generic tier. Adding a width buys speed on that width, nothing
else.

### Why the two tiers exist

FlyDSL's vectorized 128-bit path caches tiles in a Python list of registers,
which only exists while the tile loop is unrolled, and unrolling needs a
compile-time count. So the fast path **genuinely cannot take a runtime `N`** —
the tier split is forced by the kernel, not chosen for convenience.

The generic tier is the vendored runtime-`N` variant (modification 1 in
[`kernels_src/kernels/norm/rmsnorm_kernel.py`](kernels_src/kernels/norm/rmsnorm_kernel.py)):
it reads `N` from the input tensor descriptor and predicates the tail, so one
binary covers every width at the cost of the fast path.

Both tiers stay in the candidate set when both apply — the matcher admits both
for a baked width, and the tie is broken by ranking, not by refusal.

Which one wins is expressed purely as **data**. `IKernelHeuristic::rank` orders
`score desc → priority desc → kernelId asc`, and `flydslRmsNormScore` returns
`kernel.priority` unchanged rather than inventing a second opinion — so the
`priority` column above *is* the ranking, decided at the first comparison
(100 > 10, baked before generic). The later tiebreaks never come into play here,
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
| No `inv_rms_tensor_uid`, and `forward_phase != TRAINING` | All 12 were built `store_rstd=False`. A TRAINING graph wants the statistic even when it forgot to name the tensor, so the phase is refused too. |
| x, scale, y all present and all **device operands** | Not virtual, not pass-by-value: the kernel reads each through a pointer. |
| One dtype for all three, in {`BFLOAT16`, `HALF`} | The kernel loads gamma with the same element type it loads x with, and writes y in that type. |
| x, y rank ≥ 2; scale rank ≥ 1; `dims` and `strides` both present and equal length | Rank ≥ 2 so there is a row to reduce over. The matcher runs on an *unvalidated* graph, so it must be total against null or mismatched dims. |
| `sameShape(x, y)` | |
| x, y, scale all **packed row-major** | The innermost stride is a compile-time literal 1 in the kernel and has no descriptor slot. The check is stronger than the descriptor strictly needs — it could carry a padded row stride — but it is what makes collapsing rank > 2 to (M, N) sound, since one row stride has to describe all the leading axes at once. |
| rows > 0 and columns > 0 | |
| gamma is **exactly** the normalised axis: `rowCount(scale) == 1 && innermostExtent(scale) == columns` | Any other extent would be a broadcast the kernel does not perform. |
| epsilon is runtime-user-supplied, **or** within 1% of `1e-5` | See below. |

Rank > 2 is accepted, not rejected: leading axes collapse to a row count
(`rowCount` = product of every extent but the innermost). A rank-4 NCHW graph
with a `(1,1,1,N)` broadcast scale is a match.

On success it binds `flydsl_rmsnorm.{x,scale,epsilon,y}.uid` plus `.rows` and
`.columns`, so nothing downstream re-derives them from the graph.

### Kernel-scoped — `flydslRmsNormKernelMatches`

Evaluated once per candidate kernel, against the KMD metadata:

- `dtype` metadata must equal the graph dtype's kernel spelling. This function
  is the one place the two vocabularies meet — descriptors say `bf16`/`f16`, the
  flatbuffer enum says `BFLOAT16`/`HALF`.
- The baked `N`: `GENERIC_N_SENTINEL = 0` admits any width; any other value is
  admitted only when it equals the bound `columns`. A real graph never has a
  zero-width normalised axis, so the value is free to carry the sentinel.

### Epsilon, which is checked twice

Epsilon is a **literal in the kernel body** (`EPS = 1e-5`), not a kernarg. So:

- A graph that **bakes** its epsilon into the op-graph is compared at match time
  against `BAKED_EPSILON = 1e-5` with `EPSILON_RELATIVE_TOLERANCE = 1e-2`, and
  declined when it disagrees — falling through to an engine that can honour it,
  rather than failing the whole plan build.
- A graph that **defers** epsilon to execute cannot be inspected then, so
  `launch()` re-checks the resolved value and throws
  `HIPDNN_PLUGIN_STATUS_INVALID_VALUE` on a mismatch.

The 1% tolerance is wide enough that `1e-5` surviving a half-precision
round-trip (~1.4e-3 relative error) still matches, and far too narrow to admit a
caller who genuinely wanted a different epsilon.

---

## 3. Out of scope

Not gaps to be filled before this ships — each is something the shipped objects
do not compute, declined rather than approximated.

| | Why, and what handles it |
|---|---|
| **RMSNorm backward** | Forward-only. The vendored copy deliberately drops the `rmsnorm_bwd_kernel` re-export (modification 2). |
| **Training phase / saved `inv_rms`** | `store_rstd=False` in all 12. |
| **Bias** | No instance has the term. |
| **dtypes other than bf16 / f16** | fp32, fp8 and the rest are not built. Declined on dtype in both matchers. |
| **Mixed dtypes across x / scale / y** | The kernel uses one element type throughout. |
| **Non-packed / padded / transposed operands** | Innermost stride is a literal 1 with no descriptor slot. |
| **Rank < 2 on x or y** | No row to reduce over. |
| **Broadcast gamma that is not the normalised axis** | Not a broadcast the kernel performs. |
| **Multi-node graphs, fused add+RMSNorm** | One node per kernel. Fusion would be its own pack. |
| **Architectures outside gfx11-generic and gfx12-generic** | Objects are checked in for the gfx11 generic family (gfx1100–gfx1103, gfx1150–gfx1153) and the gfx12 one (gfx1200, gfx1201). gfx1170/gfx1171 belong to `gfx11-7-generic`, and CDNA is another family. With `GPU_TARGETS` naming no arch we ship kernels for, the integration reports dormant at configure time and stages nothing — it does not fail, and it does not silently produce an empty shard. Adding a family is a regeneration ([REGEN.md](REGEN.md) §3), not a code change. |
| **Ops other than RMSNorm** | `OPS` in `_instances.py` has one entry. |

---

## 4. Where each claim is checked

| Claim | Checked by |
|---|---|
| The 12 objects are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 — SHA256 against `manifest.json`, currently 12/12 |
| Descriptors agree with the objects they describe | `gen_descriptors.py --check`, run by the build before staging |
| Each object is built for the target its directory names | `gen_descriptors.py --check` reads `amdhsa.target` and, for a generic target, the ELF machine |
| Every member arch's shard holds exactly the checked-in objects | `tools/check_shards.py`, run by the build after the product pack |
| The vendored sources match upstream but for recorded modifications | `tools/diff_upstream.py` ([REGEN.md](REGEN.md) §5) |
| The pack ships both tiers for both dtypes, and every kernel declares the 8-slot signature | `TestFlydslRmsNormPacks` — and the shard census that runs it, which fails when the suite is green against an empty shard |
| The accept/refuse rules above hold | The matcher-acceptance suites in `TestFlydslRmsNormEngine.cpp` |

---

## 5. SDPA forward — `hipkernel:flydsl_sdpa_rdna`

The kernel is FlyDSL's RDNA4 flash-attention forward from AITER, ported to the
gfx11 WMMA ABI and widened so a handful of objects cover many shapes
([`kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py`](kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py),
whose header lists every modification). The native half is
[`FlydslSdpaRdnaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaNative.cpp).

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
| K and V share every extent; O has Q's; V's head dim equals Q/K's | The kernel has one `head_dim` and does not broadcast. |
| `Hq % Hk == 0` and `Hq % Hv == 0`; K and V may differ in head count | The K head and the V head are each the query head divided by its own group size. |
| Any `left_bound`/`right_bound` ≥ −1, either alignment, or the deprecated causal booleans (not both) | The kernel applies the reference's two-sided rule (`CpuFpReferenceSdpa` `isMasked`) directly. |
| Scale: `attn_scale_value` (> 0), a pass-by-value scalar tensor, or absent | Absent is `1/√head_dim`, the reference's default. The running max is over unscaled scores, which orders them correctly only for a positive scale. |
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
| 14 | additive bias | **served**: f32 or the query's dtype, broadcast over any of B, H, Sq, Skv; with every mask, the LSE output and every specialized head dim, on the prefill and the decode objects |

### 5.4 What is still missing, and what closing it takes

Each is declined today, so the graph plans on another engine or fails cleanly
at `check_support()`, never at `execute()`.

| | Effort | What it takes |
|---|---|---|
| **Padding mask** (Tier 1) | medium | Two per-batch length pointers read once per workgroup. The diagonal offset is already computed in-kernel from the lengths for exactly this. hipDNN's CPU reference does not implement padding yet, so its semantics must be defined there first. |
| **Varlen / THD** (Tier 1) | medium, after padding | Ragged offsets replace the batch stride; the grid stays over the longest sequence and tiles past a sequence's end exit early. |
| Bias on a generic head_dim | small | A `has_bias` variant of the eight generic objects (+8). |
| `Dv ≠ Dqk` (MLA) | medium–large | A second head dim for V's LDS tile, the O accumulators and the store. |
| fp32 I/O, FP8, dropout, paged KV, block masks, sinks, ALiBi, softcap | out of scope | As hipDNN's requirements state; ALiBi and softcap have no reference semantics. |
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
| The 128 objects are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 — 128/128 byte-identical |
| Each decode object carries its merge kernel, with both argument layouts as declared | `gen_sdpa_rdna.py`, before the object is written; the 38-slot main layout by `TestFlydslSdpaRdnaPacks` |
| Each object has the declared 36-argument layout and spills no registers | `gen_sdpa_rdna.py`, before the object is written |
| The vendored kernel matches AITER but for its recorded modifications | `tools/diff_upstream.py --aiter` ([REGEN.md](REGEN.md) §5) |
| The shard ships exactly one object per `dtype × head_dim × causal × has_bias × head_dim_max` class, each with the 36-slot signature | `TestFlydslSdpaRdnaPacks`, and the shard census that runs it |
| The accept/decline rules above, the bindings, and candidate selection | `TestFlydslSdpaRdnaGraphAccepts`, `TestFlydslSdpaRdnaGraphDeclines`, `TestFlydslSdpaRdnaBinding`, `TestFlydslSdpaRdnaKernelMatch` |
| The staged objects compute attention and LSE — GQA, ragged lengths, cross-attention, both causal corners, windows, bands, keyless rows, decode, `head_dim` 96, runtime and default scale — against a double-precision reference | `TestGpuFlydslSdpaRdnaDispatch` |

---

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
| RMSNorm forward | dtype (2) × {generic N, N ∈ 3072/3584/4096/5120/8192} | 12 | shipped, `gfx11-generic` |
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
| **Total, gfx11-generic** | | **140** | |
| SDPA prefill schedules (§5.1) | not an axis: `block_m` 64 for `head_dim` 64, non-causal 128 without a bias and the generic tier up to 64, and head-dim pairs (`hd_pair`) for 256 and the generic tiers above 128, replace the schedules of the same classes | ±0 | shipped; the 66 objects whose schedule did not change rebuilt byte-identical with kernel modification 18 in the source |
| **gfx12-generic** | the same table, built from the same sources for RDNA4 | **140** | shipped; GPU suites pass on gfx1201 |
| **Total, both families** | | **280** | |
| SDPA forward, CDNA (§7) | per arch: dtype (2) × `head_dim` × causal (2), the kernel family fixed by arch and `head_dim`; the generic non-causal class also by `kv_align`, the dual-wave causal class by `any_diagonal`, the dual-wave classes by `has_bias` | 30 on gfx950, 12 on gfx942 | shipped; GPU suites pass on both |

Adding a row here, with its gate result in §6.1 where it introduces a runtime
feature, is part of adding the feature.

---

## 7. SDPA forward on CDNA — `hipkernel:flydsl_sdpa_cdna`

A separate engine from §5, with its own descriptors, native symbols and dispatch
handler, so the RDNA kernels' knobs, argument list and launch geometry never
meet the CDNA ones. The kernels are FlyDSL's own MFMA flash-attention forwards,
vendored whole under
[`kernels_src/kernels/attention_cdna/`](kernels_src/kernels/attention_cdna/README.md),
whose README lists the modifications (an import path, a runtime row guard in
the cooperative K/V loads that FlyDSL 0.3.4 cannot trace in a helper, a runtime
softmax scale, runtime head counts and token strides, a runtime LSE switch, a
runtime causal diagonal, runtime operand strides, and a per-(batch, head) bias view).
The native half is
[`FlydslSdpaCdnaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaNative.cpp).

The objects are built **per arch**, not per generic family: CDNA has no generic
target that spans gfx942 and gfx950, and the two get different kernels.

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
| gfx950 bias | 64, 128 | `dualwave` | `dualwave` (+ offset-0 build at 128) |
| gfx950 bias | 256 | `generic` 128, one wave per SIMD | the same |
| gfx942 bias | 64, 128, 256 | `generic` 128 | the same (at 128 without KV global prefetch) |

The dual-wave kernel takes `head_dim` 64 and 128 only, so gfx950's 256 comes from the
generic kernel, which at that width holds its tile without spilling only with the
whole register file per wave. The generic kernel takes powers of two only. At 256 its
padded K and transposed-V tiles need 67,072 bytes of LDS, past gfx942's 64 KiB, so gfx942's
objects are built with both unpadded -- exactly 64 KiB (vendored-kernel modification 9);
K keeps its XOR swizzle against bank conflicts.

### 7.1 What is baked, and what is not

The set started as the kernels exactly as upstream builds them, and widens one
baked field at a time (§7.3).

| Baked per object | Runtime kernel argument |
|---|---|
| dtype (`bf16`, `f16`) | batch |
| `head_dim` (64, 128, 256) | query and key lengths, independently |
| causal, or none | the causal diagonal: top-left, bottom-right, a band to its right; a sliding window's lower edge (generic kernel) |
| | query and KV head counts (MHA, GQA, MQA) |
| | every operand's (batch, sequence, head) strides: BSHD, BHSD, packed QKV |
| | softmax scale |
| | whether to write the LSE output |
| the schedule: kernel, `block_m`, waves per EU, setprio, KV global prefetch -- a class's tiers (§7.6) | which tier serves a graph, by its size (the score) or by autotune |
| generic non-causal: whether the KV tail is masked (`kv_align`) | |
| gfx950 d128 causal: whether any diagonal is served (`any_diagonal`) | |
| whether an additive bias is added (`has_bias`) | the bias's (batch, head, query) strides |

Forty-four objects on gfx950, fifty-four on gfx942. A class -- (dtype, `head_dim`, causal,
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
| Q, K, V, O rank 4, device operands, **not ragged**; head-dim stride 1; every other stride on an axis longer than 1 positive and a multiple of 8 elements; K and V with the same strides; one (batch, head) slice of each within 32-bit addressing | The batch, sequence and head strides are kernel arguments, so BSHD, BHSD and packed-QKV views are served; each lane moves 16-byte vectors; the kernels take one stride set for K and V; each slice is one buffer view, whose record count is 32-bit. |
| One dtype for all four, `bf16` or `f16` | One object per dtype. |
| `Hq % Hk == 0`; V shaped as K, O as Q | The kernels have one head dimension and do not broadcast. |
| `head_dim` equal to an object's baked value | Checked by the kernel matcher against each candidate's metadata. |
| No mask, or `right_bound` ≥ 0 at either alignment (or the deprecated causal booleans), with a diagonal offset `align + right_bound` of at most `Skv − Sq`; a `left_bound` only with a right one | The kernels apply the reference's rule, key `j` visible from query row `i` iff `i + align − left ≤ j ≤ i + offset`, with `align` 0 top-left and `Skv − Sq` bottom-right. The causal objects mask the last KV tile's padding with the same compare, which holds only while every row's diagonal stays inside the keys: so bottom-right is served at any lengths, top-left when `Sq ≤ Skv`, and a band only as far as `Skv − Sq`. A sliding window (`left_bound`) runs on the generic kernel only (modification 11) -- on gfx950 its causal tiers beside the dual-wave objects; a window no row reaches is no window. |
| Scale: `attn_scale_value` (> 0), a pass-by-value scalar tensor, or absent | Absent is `1/√head_dim`, the reference's default. The running max is over unscaled scores, which orders them correctly only for a positive scale. |
| Key length a multiple of 64, for a `kv_align` 64 object | That object does not mask the last KV tile. |
| Stats, if requested: a named f32 `[B, H, Sq, 1]` device tensor in packed (batch, head, query) order | The kernels write the natural-log LSE of the scaled scores as one contiguous `[B, H, Sq]`. |
| Bias (`attn_mask`), if present: in the query's dtype, rank 1–4 right-aligned to `[B, H, Sq, Skv]`, any of B, H and Sq broadcast, the key axis contiguous; strides multiples of 8 elements, a query stride of at least `Skv` (or 0); one (batch, head) slice within 32-bit addressing | Added to the scaled scores, as PyTorch adds a float `attn_mask`, which it requires in the query's dtype. Both kernels add it (the generic one by modification 12); rows move as 16-byte vectors; each slice is one buffer view. A bias in another dtype, broadcast along the keys, or misaligned (a key length not a multiple of 8 in a packed bias) declines. |
| Declined outright: padding / varlen, dropout, paged KV, block masks, sinks, FP8 descales, ALiBi, the split max / sum-exp stats, a non-`AUTO` implementation | The objects are built without them. |
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
| ~~Full strides: BHSD, packed QKV~~ | Done: per-(batch, head) buffer views, the head's offset in each view's base (modification 6). K and V with different strides, and stats strides, are next. |
| ~~LSE output~~ | Done: compiled in, switched at launch (modification 4). Any stats strides are next, with the full strides. |
| ~~`Sq ≠ Skv`, both causal alignments, right-side bands~~ | Done: the diagonal offset is a kernel argument (modification 5). Top-left with more queries than keys and wider bands need the last KV tile's padding masked apart from the diagonal. |
| ~~Sliding windows~~ | Done on the generic kernel: a runtime lower edge in the mask and the KV loop's start (modification 11). The dual-wave kernel's pipeline starts at a fixed tile; gfx950 windows run on its generic tiers. |
| ~~`head_dim` 64 and 256, f16~~ | Done: rows in the instance table. Other head dims need the generic kernel's row loads generalised past powers of two. 256 on gfx942 followed with unpadded LDS tiles (modification 9). |
| Varlen, paged KV | Both kernels already take cumulative sequence offsets and a block table, passed unused today. |
| ~~Additive bias~~ | Done: the dual-wave kernel's (modification 8), and the generic kernel's (modification 12) for gfx942 and gfx950's `head_dim` 256. f32 bias and misaligned rows need a conversion or unaligned path. |

The instance set is then chosen by the provider's AOT selection process (build
every candidate, tune, cover, prune) rather than by hand -- done for the features
above (§7.6), and repeated as features land.

### 7.4 Where each claim is checked

| Claim | Checked by |
|---|---|
| The objects are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 |
| Each object has its family's argument layout, workgroup size and arch, fits the arch's LDS, and spills no registers but where a class's spill is explicitly allowed and recorded (`vgpr_spills`) | `gen_sdpa_cdna.py`, before the object is written |
| The vendored kernels match FlyDSL but for their recorded modifications | `tools/diff_upstream.py` ([REGEN.md](REGEN.md) §5) |
| Each arch's shard ships no two objects of one build, exactly one serve-all default per graph class, and each object with its family's signature | `TestFlydslSdpaCdnaPacks`, and the shard census that runs it for gfx950 and gfx942 |
| A companion outranks its class's default exactly where its rule prefers it | `TestFlydslSdpaCdnaKernelMatch` |
| The accept/decline rules above, the causal and scale bindings, and candidate selection, including `kv_align` | `TestFlydslSdpaCdnaGraphAccepts`, `TestFlydslSdpaCdnaGraphDeclines`, `TestFlydslSdpaCdnaBinding`, `TestFlydslSdpaCdnaKernelMatch` |
| Every staged object the matcher admits computes attention — non-causal and causal, ragged tails, a longer sequence, a scale value, a runtime scale tensor, MHA, MQA and an odd group size, cross-attention, chunked prefill, decode, rows no key reaches, a top-left band, `head_dim` 64 and 256, f16, BHSD and packed QKV, a per-head bias and a padding mask — and the LSE output, against a reference | `TestGpuFlydslSdpaCdnaDispatch`, on gfx950 and gfx942 |

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
