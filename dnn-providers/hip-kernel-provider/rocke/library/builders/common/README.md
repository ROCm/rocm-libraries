# `builders/common` — cross-arch attention utilities

This package holds builder-layer modules that are shared across all
architecture targets (gfx942, gfx950, gfx1250, …).

## `attention_spec_builder.py`

`attention_spec_builder.py` is the single seam between an
`UnifiedAttentionProblem` descriptor and the arch-specific tiled-spec
dataclasses (`UnifiedAttention2DTiledSpec`, `UnifiedAttention3DTiledSpec`)
that drive kernel compilation.

`kernels/common/attention_unified.py` imports `_tiled_spec_from_problem`
and `_tiled_3d_spec_from_problem` from here; callers that already import
from that module do not need to change.

### Tiled-2D dispatch cases

`_tiled_spec_from_problem` evaluates branches in the order below; the first
matching branch wins.

#### Branch 1 — gfx1250

When `arch == "gfx1250"`: single-warp wave32 WMMA path.  Fixed
`num_warps=1`, `block_m_per_warp=16`; `tile_size` from `_select_2d_tile_size`.

#### Branch 2 — gfx942 bf16 wide-K flash (`_enable_gfx942_bf16_flash`)

Uses `mfma_f32_32x32x8_bf16` (the CDNA3-legal wide-K atom) with a
transposed V layout.  Two sub-variants:

* **Sliced-K ring** (`_enable_gfx942_flash_k_sliced_ring`): `nw=4`,
  `tile_size=64`, `use_conflict_free_v_store=True`,
  `use_k_single_buffer=False` (ring uses a 3-slot staging scheme).
* **Non-ring**: geometry from `_gfx942_bf16_wide_geometry` — `nw=4` /
  `tile_size=64` for D64, `nw=2` / `use_k_single_buffer=True` for D128;
  cfvst from `_gfx942_bf16_wide_use_cfvst`.

Both sub-variants set `use_mfma_32x32x8=True`, `use_transposed_qk_32x32=True`,
and the `use_transposed_mask_*` flags when `_enable_gfx942_flash_mask_limit`
is true.

#### Branch 3 — gfx942 fp16 flash (`_enable_gfx942_fp16_flash`)

Same 32x32x8 / transposed orientation as branch 2 but for fp16 shapes.
`num_warps` from `_select_gfx942_flash_num_warps`; cfvst and single-buffer
from `_gfx942_flash_use_cfvst` / `_gfx942_flash_use_single_buffer`.

#### Branch 4 — gfx950 / gfx942 generic (default)

All remaining shapes.  Key knob decisions:

| Knob | Condition |
|------|-----------|
| `use_mfma_32x32` | `_enable_mfma_32x32` |
| `use_transposed_qk_32x32` | `_enable_transposed_qk_32x32` |
| `use_transposed_half_local_pv` | same as `use_transposed_qk_32x32` — always stacked on the transposed path |
| `use_transposed_scalar_state` | `_enable_combo_2d OR _enable_transposed_subflags` |
| `use_mfma32_skip_legacy_qreg` | same as `use_transposed_scalar_state` |
| `use_transposed_mask_once` + `use_transposed_mask_limit` | `(combo AND no-SW) OR _enable_transposed_subflags` |
| `use_fast_paged_kv_desc` | combo AND no-SW AND bf16 AND `num_query_heads==64` AND `num_kv_heads==8` |
| `use_register_pv` | `_enable_register_pv`: bf16, no sinks, no SW, no softcap, no alibi, no qq_bias, no additive bias, no fp8 KV, no 32x32 path |
| `use_early_v_schedule` | `_enable_early_v_schedule` |
| `use_v_double_buffer` | `_enable_v_double_buffer` (gfx950 spec only; injected via field-presence guard) |
| `use_sched_barrier` | `_enable_sched_barrier` (gfx950 spec only; field-presence guard) |
| `use_k_single_buffer` | `_enable_k_single_buffer`: d128 small-tile cohort + env opt-in (gfx950 spec only) |
| `use_fp8_mfma_qk` | `_enable_fp8_mfma_qk`: decode fp8 path only |
| `use_i64_kv_addr` | cache size > 2 GiB (see gfx950 README "Paged-cache size" section) |
| `use_additive_bias` | `problem.use_additive_bias` (gfx950 spec only; passed directly by `_spec_gfx950_generic`) |

### Tiled-3D dispatch cases

`_tiled_3d_spec_from_problem` has two branches:

* **gfx1250**: calls `_resolve_gfx1250_tiled3d` to get a `_ResolvedTiled3D`
  struct, then builds the spec from its fields (`num_waves`, `use_wide_lds_reads`,
  `use_dtla_prefetch`, `use_ds_tr_reads`, `use_fused_reduce`, `use_dpp_softmax`, …).
* **gfx942 / gfx950 (default)**: `_num_segments`, `_select_3d_waves_per_eu`,
  `_kv_storage_dtype`, `_gfx942_3d_tile_size_override`,
  `_gfx950_3d_tile_size_override`, `_enable_gfx942_3d_wide_kv_load`,
  `_enable_i64_kv_addr`, `_use_decode_grid`, `_kv_cache_policy_3d`,
  `_kv_lds_swizzle_3d`, `_wave_page_id_load_3d`; on gfx950 only,
  `use_additive_bias` from the problem.

### Knob reference

Every knob a selector resolves must also reach the launcher cache key
(`_tiled_cache_key` / `_tiled_3d_cache_key` in
[`attention_unified.py`](../../kernels/common/attention_unified.py)): add the
selector's resolved output there when you wire it. The key is computed per launch
without building the spec, so a missing entry lets two shapes that build different
kernels share one cached binary.
[`test_attention_tiled_cache_key.py`](../../tests/test_attention_tiled_cache_key.py)
fails when key equality and built-spec equality disagree.

Tiled-3D knobs (gfx942 and gfx950 split-KV segment kernel) come first; the
Knob Catalog in `platform/dsl_docs/optimization/optimization_runbook.md`
(§12.1.K2) lists them with the other 3D fields.

**`kv_cache_policy`** (3D) — Cache policy of the K/V copies, by the same names
as the 2D field. `_kv_cache_policy_3d` picks `"stream"` (non-temporal) when every
sequence decodes: each K/V byte is read once, so caching it only evicts other
data. The fp8 K/V loader keeps `"all"`; the gfx942 wide loader takes `"all"` or
`"stream"`.

**`use_kv_lds_swizzle`** (3D) — Stores each K/V LDS row's 16-byte chunks
XOR-permuted by the row, which spreads the QK and PV reads over the LDS banks.
The copies fetch permuted source chunks, so the tiles need no padding. fp16/bf16
K/V, paged or strided. Selected by `_kv_lds_swizzle_3d`.

**`use_wave_page_id_load`** (3D) — The wave loads the segment's page-table
entries 64 at a time, one per lane, and each tile reads its page ids from those
registers, so no page-table load sits on the KV loop's critical path. Paged
fp16/bf16 K/V. Selected by `_wave_page_id_load_3d`; the decode split rule in
`_num_segments` assumes it where both apply.

**`use_decode_grid`** (3D) — One CTA per sequence along grid x when every
sequence has one query token, instead of the prefill upper bound and a search
over `query_start_len`. Derived from the problem by `_use_decode_grid`, not a
tuning knob; it stays a spec field because the binary and the launch grid differ.

**`tile_size_override`** (gfx950 3D) — A multiple of the page size: each loop
iteration gathers several pages. `_gfx950_3d_tile_size_override` uses it for
decode with too few CTAs to fill the device, together with the K/V LDS swizzle.

**`use_transposed_half_local_pv`** — Rewrites the PV phase of the transposed
32x32 kernel so each 32-lane warp-half reads only the P rows it owns,
eliminating the cross-half `lane^32` shuffle fetch.  Bit-identical to the
plain transposed path; measured +1.24x on Qwen3-30B-A3B prefill (bf16/d64,
single-seq, q=2048).  Enabled whenever `use_transposed_qk_32x32` is set.

**`use_mfma32_skip_legacy_qreg`** — Drops the dead 16x16 Q gather that the
original 32x32 code-path inherited from the 16x16 body.  After the 32x32
switch the Q registers are gathered in a separate 32x32-aligned loop, so the
legacy gather is a no-op VALU block; removing it shrinks the inner loop without
any correctness impact.  Enabled in the same cohort as `use_transposed_scalar_state`
(combo AND `_enable_transposed_subflags`).

**`use_fast_paged_kv_desc`** — Activates a specialised fast paged-KV descriptor
for the no-SW bf16 combo geometry (`nw=4`, `T=64`).  The descriptor pre-computes
a multi-block pointer stride so the inner loop skips the per-block
`make_buffer_rsrc`; only valid for the exact 64-query / 8-kv-head combination
that the spec's `__post_init__` validator enforces.  (A tensor-parallel-sharded
16q/2kv GQA-8 model also passes `num_queries_per_kv==8` but fails the absolute
head-count check — intentional.)

**`use_register_pv`** — Stores the P matrix in VGPRs rather than publishing
to `P_lds` before the PV multiply.  Requires the 16x16x32 MFMA path
(incompatible with `use_mfma_32x32`; the 32x32 path has its own in-register
P pipeline).  Enabled for bf16 shapes not on the sinks / sliding-window /
softcap / alibi / qq-bias / fp8-KV path.

**`use_agpr_alloc_zero`** — Requests zero AGPR allocation so the LLVM backend
uses VGPR-form MFMA for the PV accumulators, avoiding the AGPR↔VGPR copy
traffic that arises when the online-softmax alpha scaling touches the
accumulator.  Not wired into the production selector (opt-in spec flag, default
OFF); used in `attention_tiled_2d_fastkv_regp` and the ck83 Fix-A residency
experiments on gfx942.

**`use_v_double_buffer`** — Pre-fetches the next V tile while the current PV
MFMA is in flight.  Gfx950 spec only (gfx942 spec does not declare this field).
Enabled for the single-batch d128 short-prefill cohort by `_enable_v_double_buffer`.

**`use_sched_barrier`** — CK-Tile-derived `sched_barrier` steering: fences the
QK MFMA cluster from the post-QK async-prefetch VMEM stream so the LLVM
scheduler keeps QK MFMAs packed.  Gfx950 only.  Enabled for the single-batch
d128 short-prefill cohort where one resident wave cannot otherwise hide the
prefetch-in-MFMA-window cost (`num_warps==1 AND use_v_double_buffer`).

**`use_k_single_buffer`** — Reduces K LDS from two slots to one so a larger
`T=64` tile fits the 32 KB / 2-WG/CU budget at HD=128.  The next-K prefetch
is issued *after* the PV-wait barrier to avoid the WAR race that single-buffer
naively introduces.  Gfx950 only; see the gfx950 attention
[README](../gfx950/attention/README.md) "Single-batch d128" section for measured
results.

**`use_additive_bias`** — Feature flag, not a perf lever.  Adds a float32
additive bias to the scaled QK scores before the online softmax, on both the
2D tiled and 3D split-KV kernels:
`S[q, k] += bias[seq*batch_stride + q_head*head_stride + q_local*sq_stride + k]`,
where `q_head` is the *query* head (GQA-correct), `q_local` is the query
position within its sequence, and `k` is the absolute key position (Sk stride
is 1).  Like ALiBi/QQ bias it is added after the softcap, so the score is
`softcap(QK * scale) + bias`.  A stride of 0 broadcasts that dimension, so `[B,H,Sq,Sk]`,
`[1,H,Sq,Sk]`, `[B,1,Sq,Sk]` and `[1,1,Sq,Sk]` layouts are all supported, and
`sq_stride` may exceed `max_seqlen_k` for a padded row.  The bias load is
predicated on the same row/causal mask as the score: masked lanes (padding rows,
tail columns) load element 0 instead of an out-of-range index, and the value is
discarded.  Gfx950 spec only (field-presence guard);
`supports_native_unified_attention_tiled` / `_3d_tiled` reject it on other
arches.  Default OFF: the bias pointer and its three strides are appended to the
end of the kernel ABI only when ON (kernel name gains an `addb` token), so the
non-bias IR is unchanged.  On 2D the bias is emitted only on the 16x16 path and
the plain transposed 32x32 path: `__post_init__` rejects it together with
`use_transposed_mask_limit`, `use_transposed_invariant_hoist`,
`use_transposed_mask_once`, the non-transposed 32x32 path,
`use_grouped_kv2_softmax` and `use_register_pv`, and the selector keeps those
knobs off when `problem.use_additive_bias` is set.
Enabled by `UnifiedAttentionProblem.use_additive_bias`; the tensor and strides
are passed to `run_unified_attention_torch(additive_bias=...,
additive_bias_batch_stride=..., additive_bias_head_stride=...,
additive_bias_sq_stride=...)`.  The entry point validates these before launch:
the tensor must be present iff the flag is on, float32 and contiguous along Sk;
`additive_bias_sq_stride` has no default when bias is on and must be `0`
(broadcast over Sq, e.g. a `[B,1,1,Sk]` key-padding bias) or `>= max_seqlen_k`.
The kernels address the bias with i32 element indices and i32 strides, so each
stride must be in `[0, 2**31)` and the largest index a launch can read
(`(num_seqs-1)*batch_stride + (H_q-1)*head_stride + (max_seqlen_q-1)*sq_stride
+ max_seqlen_k-1`) must be below `2**31` (8 GiB of f32); larger layouts raise
`NotImplementedError` instead of wrapping into an out-of-bounds read.
Broadcasting a dimension (stride 0) shrinks the index; i64 bias addressing is a
follow-up.  That same index must also lie inside the tensor's storage (past its
storage offset), so strides that do not match the tensor raise `ValueError`.  Python engine only for now (no C engine mirror yet).  Numeric coverage:
[test_attention_additive_bias_gfx950_numeric.py](../../tests/test_attention_additive_bias_gfx950_numeric.py).
