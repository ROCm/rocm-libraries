# Plan: porting sliding-window, batching, and bias to the dense prefill kernel

Scope: extend the productized gfx950 dense flash-attention prefill kernel
(`kernels/gfx950/attention_dense.py`, host builder `attention_dense_prefill.py`)
with **sliding-window attention (SWA)**, richer **batching** (varlen), and
**bias addition** (ALiBi / additive bias / logit softcap), informed by a
structured read of the flyDSL implementations in `~/flydsl-main`.

This document records (1) what flyDSL actually does for each feature — tile
layouts, MFMA atom sizes, layout transforms, wave/register/LDS allocation, and
tile/wave/K-loop scheduling — and (2) the concrete change plan for the rocke
dense kernel, with the always-on 877-TFLOPS pipeline preserved as the no-feature
fast path.

---

## 0. Baseline: the rocke dense kernel we are extending

- `BLOCK_M = 256` query rows/CTA, `block_n = 64` KV tile, `NBUF = 2` double
  buffer, `num_waves = 8` (each wave owns 32 query rows), `waves_per_eu = 2`,
  256 VGPR / 0 spill.
- MFMA `32x32x16` for both QK and PV (`mfma_32x32x16_for_dtype`).
- CK-1 transposed PV (P fed to PV MMA in native QK-output layout via half-local
  V load `pv32_v_load_paired`; no cross-half P relayout).
- LDS K padded `[NBUF, BN, D+8]` (bank-conflict fix); V unpadded `[NBUF, BN, D]`.
- Diagonal-only causal masking: mask-free body loop + masked diagonal tail.
- Depth-1 cluster fusing `exp2` into the PV MFMA loop; `sched_group_barrier`
  template (DS_READ/MFMA/VALU/TRANS).
- Default grid `(Sq//256, Hq, B)`; persistent grid-stride variant loops one 1-D
  grid of `num_persistent` CTAs over `W = (Sq//256)*Hq*B`.
- Inputs: dense contiguous `[B, S, H, D]`, uniform seqlen; params
  `q,k,v,o,scale`. No varlen/paged, no bias, no sliding window.

**Key finding:** flyDSL's production gfx950 prefill kernel
(`flash_attn_gfx950.py`, "DUALWAVE_SWP") is the *same design family* as this
kernel — same `BLOCK_M=256`/`BLOCK_N=64`, same `32x32x16` atoms, 8 waves × 32
rows, P-in-registers, V transpose-read, double-buffered `[K0,V0,K1,V1]`,
`sched_group_barrier` + `s_setprio` + staggered wave barriers, and a lazy online
rescale. So flyDSL is a high-fidelity reference for the pipeline mechanics; the
deltas we need are exactly the three requested features.

---

## 1. flyDSL findings by feature

### 1a. Core pipeline + batching (`flash_attn_gfx950.py`, `flash_attn_generic.py`)

- **Tile layout / atoms:** `BLOCK_M=256`, `BLOCK_N=64`, `K_SUB_N=32`, 8 waves ×
  32 rows; MFMA `f32_32x32x16_{bf16,f16}` (K=16, v8 packs, v16f32 acc) for QK
  and PV; fp8 path uses `mfma_f32_32x32x16_fp8_fp8` with optional wide
  `32x32x64` PV. Q register-resident, pre-scaled by `1/sqrt(D)*log2(e)`.
- **Transforms:** K uses either XOR swizzle (`col ^ ((row&mask)<<4)`) or a padded
  `u_rk` layout (`+8 bf16`); V is transpose-read with `ds_read_b64_tr_b16`
  (padded `+32 bf16`) — matches our `pv32_v_load_paired`. P stays in VGPRs (no
  LDS roundtrip); half-wave partners combined via `permlane32_swap`. O store
  fuses partners with `permlane32_swap + cvt_pk_bf16_f32` → `buffer_store_dwordx4`.
- **Wave/register/LDS:** WG=512 threads (8 waves), `waves_per_eu=2`, interleaved
  double buffer `[K0,V0,K1,V1]` (~68 KiB @ D=128). DUALWAVE staggers two
  4-wave groups with extra barriers to time-multiplex the SW pipeline.
- **K-loop scheduling:** hand-built 8-cluster software pipeline, `j += 2` (2 KV
  tiles/iter), PV lags QK by ~1 tile; async `buffer_load_lds` (128B) with split
  `s_waitcnt` (lgkmcnt vs vmcnt); `sched_group_barrier` (MFMA/VALU/EXP masks),
  `s_setprio`, `s_nop`. **Lazy online rescale** skips O/P rescale when
  `max(m_tile - m_row) <= 8.0` (ballot-vs-exec test) — a lever we have not baked.
- **Batching:** 3D grid `(NUM_HEADS_Q, num_q_blocks, grid_z)` with
  `grid_z = batch` (or `batch*num_kv_splits` for split-K); GQA via
  `kv_head = q_head // group_size`; **varlen** via `cu_seqlens_q/kv` rebased
  buffer descriptors (`q_tok_base = cu[b]`, `seqlen_q_b = cu[b+1]-cu[b]`), grid
  sized by `max_seqlen_q` with early-exit of empty q-tiles; **paged** via a
  block table staged in LDS; **split-K** with an fp32 workspace + combine kernel.
  **No persistent/grid-stride** (our persistent mode is a rocke-specific lever).

### 1b. Sliding window (`pa_decode_swa.py` — decode/paged/fp8 only)

**flyDSL has SWA only on the paged-decode path; flash-attn prefill is causal-only.**
The technique is directly portable:

- **Masking math (compile-time `W`):** per query, `causal_bound = C + 1 - L + qi`
  (exclusive upper), `seq_start = C - L - W + qi` (inclusive lower); allowed iff
  `seq_start <= k < causal_bound`, i.e. `k ∈ [q_abs - W, q_abs]` (**W+1** keys
  incl. diagonal; note the `+1` convention). `q_abs = C - L + qi`.
- **Two-level pruning:** (a) tile-level — `seq_start_global = max(0, C-L-W)`,
  `tail_start_tile = seq_start_global >> 8` (tiles are 256 tokens), only visit
  `[tail_start_tile, num_tiles)`; host bounds partitions with
  `ceil((W+L-1)/256)+1`. (b) per-lane range mask on the MFMA `f32x4` output:
  `in_range = (tok < causal_bound) & (tok >= seq_start)`, `select(in_range, s, -inf)`.
- **Mask placement / layout:** applied **after QK MFMA, before row-max/exp2**, in
  the *native MFMA accumulator layout* — **no relayout/transpose**. Token indices
  built inline (`kv_tok_base + td*16 + i`); bounds are SGPR scalars broadcast
  across the 4 lanes.
- **No mask-free interior fast path in SWA** (every visited tile is masked). But
  flash-attn *causal* has `tile_needs_mask` (skip mask when
  `max_kv_col <= q_start`) — the optimization to replicate for the SWA interior.
- **Wave cooperation:** all waves iterate the same tile range; per-wave rows
  differ only in the per-lane bound; cooperative K/V loads unchanged.
- **Cost:** +1 compare vs causal-only (two bounds instead of one) per `f32x4`.

### 1c. Bias / ALiBi / softcap — **not implemented anywhere in flyDSL**

Full-repo grep confirms: no ALiBi, no additive bias tensor, no QQ-bias, no
softcap in any kernel; the only `alibi_slopes`/`bias` refs are test harnesses
passing `None` to the external aiter reference. flyDSL applies only causal / KV-pad
/ SWA masks + QK softmax scaling (+ fp8 quant scales). So for bias we have **no
flyDSL reference to port** — the rocke `unified_attention` tiled path is the
in-repo reference (it *does* implement ALiBi + QQ-bias + softcap). flyDSL still
tells us the *insertion mechanics*:

- **Insertion point:** after QK MFMA, **before mask/row-max/exp2**, on the raw
  accumulator layout (same place SWA masks) — no relayout.
- **Domain:** gfx950 pre-scales Q by `sm_scale*log2(e)`, so scores are already in
  log2 domain; any bias must be added **in that domain** (multiply linear biases
  by `log2(e)` = `RCP_LN2`), matching how the rocke tiled path applies ALiBi
  (`slope*(key_pos-ctx)*RCP_LN2`) and QQ-bias (`qq_bias[...]*RCP_LN2`).
- **ALiBi:** `S += slope_h * (k_pos - q_pos)` using the same per-lane
  `k_pos`/`q_pos` maps SWA/causal already compute (`_mfma_32x32_c_row/_col`).
- **Softcap:** nonlinear `S = cap * tanh(S/cap)` applied **after scale, before
  mask** (must precede max). Careful: our exp2 fast path assumes softmax arg
  `<= 0` and no overflow guard — softcap changes the score range, revisit.
- **Composition:** per-row constant biases cancel in softmax; **relative** biases
  (ALiBi, elementwise, softcap) do not and must be applied before max so online
  rescale stays consistent.

---

## 2. Porting plan

Ordering by value/effort: **(A) sliding window** (cheap, high value, byte-safe
default) → **(B) varlen batching** → **(C) bias/ALiBi/softcap**. Each is gated so
the no-feature path emits today's exact 877-TFLOPS kernel.

### A. Sliding window

Design already scoped in prior analysis; flyDSL confirms the mechanics.

- **Spec:** add `sliding_window: int = 0` to `AttentionDenseSpec` (0 = disabled →
  bit-identical to today). Bake into `kernel_name()`; require `W % block_n == 0`
  for the clean path (`__post_init__`). Mirror in the persistent builder.
- **Geometry:** both causal upper (`k<=q`) and window lower (`k>=q-W+1`) are
  slope-1 diagonals; the valid region per 256-row block is a **parallelogram
  band**. With `n_per = BLOCK_M//BN = 4` and `Wt = W//BN`:
  `lo_tile = max(0, qb*n_per - Wt)`, `hi_tile = min(n_ktiles, (qb+1)*n_per)`.
  Each boundary band is exactly `n_per = 4` tiles wide.
- **Three-phase K-loop:** (L) window-edge tiles `[lo_tile, lo_tile+n_per)` lower-
  masked; (M) interior `[lo_tile+n_per, diag_start)` **mask-free** (identical hot
  loop → same peak); (R) causal diagonal `[diag_start, hi_tile)` upper-masked
  (existing tail). If `W < BLOCK_M` bands overlap → single merged both-bounds
  phase. Reuse the existing `select`-guarded bound idiom for empty phases.
- **Masking:** extend `do_mask` with `mask_lower`/`mask_upper` flags; reuse
  `_mfma_32x32_c_row/_col` exactly (no new layout). Lower predicate
  `cmp_gt(ktok, query_tok - W)` (W compile-time → immediate).
- **Register pinning:** hoist `query_tok` (lane-only, tile-invariant) and
  `query_tok-(W-1)` to once-per-block VGPRs; masking stays outside the loop-
  carried set (`m,l,o,pk`) → no VGPR growth, preserve 0-spill.
- **Pipeline:** prologue primes `lo_tile`/`lo_tile+1` (not 0/1) and left-masks the
  prologue tile; init `m=-inf,l=0,o=0` unchanged (tiles `<lo_tile` never visited,
  correct because they'd be all `-inf`). Epilogue PV unchanged.
- **Wave cooperation:** loop bounds are block-level → all 8 waves iterate the same
  tile range and cooperatively DMA the same K/V tiles; only per-lane masks differ.
- **Perf:** tiles/block drop from `O(qb)` (up to 128 @ Sq=8192) to `O(Wt+n_per)`
  → SWA is *faster* than full causal; interior phase runs at current peak.
- **Optional lever (from flyDSL):** bake the **lazy online rescale**
  (skip O/P rescale when `max(m_tile-m_row) <= 8`) — orthogonal win, evaluate
  separately.

### B. Varlen batching

Today batch is dense/uniform (contiguous `[B,S,H,D]`, `Sq%256==0`, grid
`(Sq//256, Hq, B)`). flyDSL's varlen recipe is directly applicable:

- **ABI:** add optional `cu_seqlens_q`, `cu_seqlens_kv` (int32 `[B+1]`) params;
  Q/K/V become packed `[total_tokens, H, D]`.
- **Per-batch rebase:** `q_tok_base = cu_q[b]`, `seqlen_q_b = cu_q[b+1]-cu_q[b]`
  (same for KV); fold into the existing `q_base`/`k_base` offset math.
- **Grid:** size by `max_seqlen_q` (grid `(max_Sq//256, Hq, B)`); **early-exit**
  q-blocks past `seqlen_q_b`. `n_ktiles`/`n_upper` become per-batch runtime
  (currently compile-time) — the main change, since causal `n_upper` and SWA
  `lo_tile/hi_tile` now derive from runtime `seqlen_kv_b`.
- **Cross-seqlen causal:** support bottom-right alignment `delta = seqlen_kv_b -
  seqlen_q_b` in the mask (flyDSL `_causal_mask_inplace` convention).
- **Scope note:** paged KV is a larger effort (block table + LDS staging); defer.
  Varlen (non-paged) is the near-term batching win and composes with SWA.
- **Persistent variant:** `W = sum(ceil(seqlen_q_b/256))*Hq` becomes ragged; keep
  persistent for the uniform path first, extend later.

### C. Bias / ALiBi / softcap

No flyDSL reference; use the in-repo rocke tiled path semantics + flyDSL insertion
mechanics.

- **ALiBi:** add `use_alibi` + `alibi_slopes_ptr` (`[Hq]` f32). Preload the head
  slope to a register once per CTA; in the masked/boundary and interior phases add
  `slope * (ktok - query_tok) * RCP_LN2` to `s_reg[nsub][i]` before `softmax_max`.
  Reuses the exact `ktok`/`query_tok` lane maps → no relayout. Applies to *all*
  tiles (not just boundary), so interior phase gains +1 FMA/score.
- **Additive bias:** add `qq_bias_ptr` + stride; per-lane global load of
  `bias[q_pos, k_pos - ctx]`, add (`*RCP_LN2`) before max. Higher cost (a global
  load per score); keep behind its own flag.
- **Softcap:** add `softcap: float`; apply `S = cap*tanh(S/cap)` after QK scale,
  before mask/max. **Caveat:** revisit the `exp2_fast` no-overflow-guard
  assumption (softmax arg `<= 0`) — softcap bounds scores, likely fine but verify.
- **Register/perf:** slopes pinned in registers (compile-time `use_alibi`,
  runtime slope value); bias/softcap flags default-off so the hot path is
  unchanged. All additions live before the fused PV/exp2 loop → no change to the
  loop-carried live set.
- **Composition:** apply bias → mask (`-inf`) → row-max → exp2, so masked
  positions stay `-inf`; matches flyDSL ordering and keeps online rescale valid.

---

## 3. Validation

- **Parity:** extend `attention_dense_prefill.py::run` references:
  - SWA: SDPA with a banded mask (causal ∧ `q-k <= W`); reuse `max_abs < 2e-2`.
  - Varlen: per-sequence SDPA over `cu_seqlens` slices.
  - ALiBi/softcap: SDPA + explicit bias / `cap*tanh` on scores.
- **Golden safety:** with `sliding_window=0`, `use_alibi=False`, `softcap=0`,
  uniform batch → assert byte-identity vs today's kernel (same `kernel_name`
  minus new suffixes; compare HSACO or output bitwise).
- **Perf:** confirm SWA tile-prune (e.g. Sq=8192, W=1024 → ~20 tiles/block vs up
  to 128) and that the interior phase matches current TFLOPS; confirm no VGPR
  spill regression (stay at 256 VGPR / 0 spill).

## 4. Risks / open questions

- **VGPR budget:** kernel is at the 256-VGPR / 0-spill ceiling. Boundary masking
  and ALiBi FMAs must stay out of the loop-carried set; measure `waves_per_eu`
  sensitivity (wpe=3 is a known trap here).
- **`W % block_n != 0`:** window edge mid-tile → `n_per+1`-tile band; constrain W
  to a multiple of `block_n` initially, or add the extra boundary tile.
- **exp2_fast + softcap:** verify the no-overflow-guard assumption still holds
  once scores are softcapped/biased.
- **Varlen runtime bounds:** converting compile-time `n_ktiles/n_upper` to runtime
  may cost registers / block some constant folding; measure vs the uniform path
  and keep the uniform build as a specialization.
- **Persistent + varlen:** ragged work decode is non-trivial; land uniform-persistent
  and default-varlen first.
- **ALiBi interior cost:** unlike masking (boundary-only), ALiBi touches every
  tile (+1 FMA/score); quantify the interior-phase throughput hit.

## References (flyDSL, `~/flydsl-main`)

- Core + batching: `kernels/attention/flash_attn_gfx950.py`,
  `flash_attn_generic.py`, `dualwave_swp_common.py`, `flash_attn_interface.py`.
- Sliding window: `kernels/attention/pa_decode_swa.py` (masking math + tile
  prune), `flash_attn_generic.py` (`tile_needs_mask` causal fast path).
- Bias: none (confirmed absent); in-repo reference is the rocke
  `kernels/gfx950/attention_tiled_2d.py` / `attention_tiled_3d.py` ALiBi/QQ-bias/
  softcap paths.
