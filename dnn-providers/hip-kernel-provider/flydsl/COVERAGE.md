<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# What the FlyDSL packs cover

Built once for each of two LLVM generic targets -- `gfx11-generic` (RDNA3 /
RDNA3.5: gfx1100–gfx1103, gfx1150–gfx1153) and `gfx12-generic` (RDNA4: gfx1200,
gfx1201) -- and shipped to every member, and once for each of the CDNA targets
gfx90a, gfx942 and gfx950: **RMSNorm forward** (25 kernel objects per target,
§1–§4). Every graph a pack accepts is
computed by one of its objects; everything else is **declined**, so another
engine gets the plan.

That last point is the design rule this whole file exists to document. The
kernels are pre-built HSACO — [`FlydslRmsNormNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp)
cannot change what they compute, only decide whether they are the right thing to
run. So every condition below is enforced by *refusing*, never by adapting. A
matcher that stretched to accept a near-miss would compute the wrong answer
silently; one that declines it costs a fallback.

---

## 1. The 25 instances

Generated from [`generators/_instances.py`](generators/_instances.py), which is
the single source of truth — the compiler, the packer and the descriptor emitter
all read it, so adding an instance is a row there rather than an edit in three
places that can disagree.

| Instance | Tier | x/y dtype | gamma dtype | `N` | `block_threads` | priority |
|---|---|---|---|---|---|---|
| `rmsnorm_bf16_ngeneric_bt256` | generic | bf16 | bf16 | runtime | 256 | 10 |
| `rmsnorm_f16_ngeneric_bt256` | generic | f16 | f16 | runtime | 256 | 10 |
| `rmsnorm_f32_ngeneric_bt256` | generic | f32 | f32 | runtime | 256 | 10 |
| `rmsnorm_bf16_n3072_bt256` | specialized | bf16 | bf16 | 3072 | 256 | 100 |
| `rmsnorm_bf16_n3584_bt256` | specialized | bf16 | bf16 | 3584 | 256 | 100 |
| `rmsnorm_bf16_n4096_bt256` | specialized | bf16 | bf16 | 4096 | 256 | 100 |
| `rmsnorm_bf16_n5120_bt256` | specialized | bf16 | bf16 | 5120 | 256 | 100 |
| `rmsnorm_bf16_n8192_bt256` | specialized | bf16 | bf16 | 8192 | 256 | 100 |
| `rmsnorm_f16_n3072_bt256` | specialized | f16 | f16 | 3072 | 256 | 100 |
| `rmsnorm_f16_n3584_bt256` | specialized | f16 | f16 | 3584 | 256 | 100 |
| `rmsnorm_f16_n4096_bt256` | specialized | f16 | f16 | 4096 | 256 | 100 |
| `rmsnorm_f16_n5120_bt256` | specialized | f16 | f16 | 5120 | 256 | 100 |
| `rmsnorm_f16_n8192_bt256` | specialized | f16 | f16 | 8192 | 256 | 100 |
| `rmsnorm_bf16_ngeneric_bt256_f32scale1` | generic | bf16 | f32 | runtime | 256 | 10 |
| `rmsnorm_f16_ngeneric_bt256_f32scale1` | generic | f16 | f32 | runtime | 256 | 10 |
| `rmsnorm_bf16_n3072_bt256_f32scale1` | specialized | bf16 | f32 | 3072 | 256 | 100 |
| `rmsnorm_bf16_n3584_bt256_f32scale1` | specialized | bf16 | f32 | 3584 | 256 | 100 |
| `rmsnorm_bf16_n4096_bt256_f32scale1` | specialized | bf16 | f32 | 4096 | 256 | 100 |
| `rmsnorm_bf16_n5120_bt256_f32scale1` | specialized | bf16 | f32 | 5120 | 256 | 100 |
| `rmsnorm_bf16_n8192_bt256_f32scale1` | specialized | bf16 | f32 | 8192 | 256 | 100 |
| `rmsnorm_f16_n3072_bt256_f32scale1` | specialized | f16 | f32 | 3072 | 256 | 100 |
| `rmsnorm_f16_n3584_bt256_f32scale1` | specialized | f16 | f32 | 3584 | 256 | 100 |
| `rmsnorm_f16_n4096_bt256_f32scale1` | specialized | f16 | f32 | 4096 | 256 | 100 |
| `rmsnorm_f16_n5120_bt256_f32scale1` | specialized | f16 | f32 | 5120 | 256 | 100 |
| `rmsnorm_f16_n8192_bt256_f32scale1` | specialized | f16 | f32 | 8192 | 256 | 100 |

Two gamma forms × 2 dtypes × (1 generic + 5 specialized widths), plus a generic f32
object. The second form reads gamma as f32 beside bf16/f16 x and y (`scale_f32: 1`,
the kernel's `weight_dtype_str="f32"`): the mixed form cuDNN and PyTorch commonly
send. f32 has no specialized tier because the kernel's 128-bit fast path (below) is
for 16-bit elements only: an f32 object with `N` baked in runs the same scalar loop
the generic one does. `block_threads` is single-valued
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
| No `inv_rms_tensor_uid`, and `forward_phase != TRAINING` | All 25 were built `store_rstd=False`. A TRAINING graph wants the statistic even when it forgot to name the tensor, so the phase is refused too. |
| x, scale, y all present and all **device operands** | Not virtual, not pass-by-value: the kernel reads each through a pointer. |
| x and y one dtype, in {`BFLOAT16`, `HALF`, `FLOAT`}; gamma in that dtype, or `FLOAT` beside `BFLOAT16`/`HALF` | Each object reads x and writes y in one element type, and reads gamma in that type or, for the `scale_f32` objects, as f32. |
| x, y rank ≥ 2; scale rank ≥ 1; `dims` and `strides` both present and equal length | Rank ≥ 2 so there is a row to reduce over. The matcher runs on an *unvalidated* graph, so it must be total against null or mismatched dims. |
| `sameShape(x, y)` | |
| x, y, scale all **packed row-major**, ignoring the stride of any axis of extent 1 | The innermost stride is a compile-time literal 1 in the kernel and has no descriptor slot, and the launch passes the width as the row stride. The check is stronger than the descriptor strictly needs — it could carry a padded row stride — but it is what makes collapsing rank > 2 to (M, N) sound, since one row stride has to describe all the leading axes at once. An axis of extent 1 is never stepped along, so its stride is not checked: hipdnn_torch describes an `F.rms_norm` weight with stride 0 on its unit axes, and the data SDK's packed test exempts them the same way. |
| rows > 0 and columns > 0, and every element of x and y within 2³¹ − 1 bytes | Each operand is one buffer view with 32-bit byte offsets; the bound counts bytes, so f32 reaches it at half the elements. |
| gamma **encodes the normalised shape** as the frontend reads it: the normalised axes are the longest trailing run where gamma's extent equals x's (never axis 0), and gamma is 1 on every axis before it | Any other shape would be a broadcast the kernel does not perform. The same rule as `validateScaleNormalizedShape` in the frontend. |
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

## 3. Out of scope

Not gaps to be filled before this ships — each is something the shipped objects
do not compute, declined rather than approximated.

| | Why, and what handles it |
|---|---|
| **RMSNorm backward** | Forward-only. The vendored copy deliberately drops the `rmsnorm_bwd_kernel` re-export (modification 2). |
| **Training phase / saved `inv_rms`** | `store_rstd=False` in all 25. |
| **Bias** | No instance has the term. |
| **dtypes other than bf16 / f16 / f32** | fp8 and the rest are not built. Declined on dtype in both matchers. |
| **Other mixed dtypes** (y unlike x; a 16-bit gamma unlike x; a 16-bit gamma beside f32 x) | Only an f32 gamma beside bf16/f16 x and y has objects. |
| **Non-packed / padded / transposed operands** | Innermost stride is a literal 1 with no descriptor slot, and the launch passes the width as the row stride. Unit axes are exempt (§2). |
| **Rank < 2 on x or y** | No row to reduce over. |
| **Gamma that does not encode a normalised shape** (e.g. `[1,2,1,3]` against `[1,2,3,3]`) | Not a broadcast the kernel performs; the frontend rejects it too. |
| **Multi-node graphs, fused add+RMSNorm** | One node per kernel. Fusion would be its own pack. |
| **Architectures outside gfx11-generic, gfx12-generic, gfx90a, gfx942 and gfx950** | Objects are checked in for the gfx11 generic family (gfx1100–gfx1103, gfx1150–gfx1153), the gfx12 one (gfx1200, gfx1201), and the CDNA targets gfx90a, gfx942 and gfx950. gfx1170/gfx1171 belong to `gfx11-7-generic`. With `GPU_TARGETS` naming no arch we ship kernels for, the integration reports dormant at configure time and stages nothing — it does not fail, and it does not silently produce an empty shard. Adding a family is a regeneration ([REGEN.md](REGEN.md) §3), not a code change. |
| **Ops other than RMSNorm** | `OPS` in `_instances.py` has one entry. |

---

## 4. Where each claim is checked

| Claim | Checked by |
|---|---|
| The 25 objects per target are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 — SHA256 against `manifest.json`, currently 25/25 |
| Descriptors agree with the objects they describe | `gen_descriptors.py --check`, run by the build before staging |
| Each object is built for the target its directory names | `gen_descriptors.py --check` reads `amdhsa.target` and, for a generic target, the ELF machine |
| Every member arch's shard holds exactly the checked-in objects | `tools/check_shards.py`, run by the build after the product pack |
| The vendored sources match upstream but for recorded modifications | `tools/diff_upstream.py` ([REGEN.md](REGEN.md) §5) |
| The pack ships the generic tier for every dtype and gamma form and the specialized tier for the 16-bit ones, every kernel declares the 9-slot signature (epsilon last), and the engine's schema admits runtime epsilons | `TestFlydslRmsNormPacks` — and the shard census that runs it, which fails when the suite is green against an empty shard |
| The accept/refuse rules above hold | The matcher-acceptance suites in `TestFlydslRmsNormEngine.cpp` |
