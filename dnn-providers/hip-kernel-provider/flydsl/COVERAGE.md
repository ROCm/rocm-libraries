<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# What the FlyDSL packs cover

Built once for each of two LLVM generic targets -- `gfx11-generic` (RDNA3 /
RDNA3.5: gfx1100–gfx1103, gfx1150–gfx1153) and `gfx12-generic` (RDNA4: gfx1200,
gfx1201) -- and shipped to every member: **RMSNorm forward** (12 kernel objects per
family, §1–§4). Every graph a pack accepts is
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
