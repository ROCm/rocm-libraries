# HSTU Attention Backward — Kernel Architecture (gfx950)

How the three HSTU backward gradient kernels (`dv`, `dk`, `dq`) are tiled,
staged, and scheduled on gfx950 (CDNA4, wave64). Read [`ALGORITHM.md`](ALGORITHM.md)
first for the math and [`README.md`](README.md) for the optimization case study
and measured numbers.

Source: `library/kernels/common/hstu_attention_bwd.py`
(`_build_hstu_attention_bwd_tiled`, selected when `HstuBwdSpec.block_m > 0`). A
simpler one-16×16-tile-per-CTA MFMA path (`_build_hstu_attention_bwd_mfma`) and a
scalar reference path remain as fallbacks.

## 1. Unified tiling

All three kernels share one body, parameterized by roles. A CTA:

- **owns** `block_m` rows of the output-row dim (`kv` for dv/dk, `q` for dq) and
  the **full `out_dim`** (`hidden_dim` for dv, `head_dim` for dk/dq),
- splits `block_m` across `num_waves` wave64 warps
  (`ROWS_PER_WAVE = block_m / num_waves`, `OWN_SUB = ROWS_PER_WAVE / 16`),
- **streams** the reduction dim (`q` for dv/dk, `kv` for dq) in `block_n` tiles.

Grid: `(ceil(max_seq_len / block_m), num_heads, batch)`; block: `num_waves * 64`
threads. Compile-time factors: `D_CHUNKS = out_dim/16`,
`STREAM_SUB = block_n/16`, `K_STEPS = head_dim/16`, `DA_STEPS = hidden_dim/16`,
`HEAD_DIM_K = round_up(head_dim, 64)`.

The MMA atom is `mfma_f32_16x16x16_{f16,bf16}`. Per streamed tile the kernel
issues 32 (dv) / 48 (dk, dq) MFMAs — matching the FlyDSL ISA exactly.

## 2. Streamed-operand LDS staging (once per tile)

Two streamed operands are staged into LDS once per `block_n` tile and reused
across all own-subtiles, waves, and `d`-chunks:

- **`lds_head`** `[block_n, HEAD_DIM_K]` — the streamed head-dim tensor (`Q` for
  dv/dk, `K` for dq). **XOR-swizzled** `col ^ ((row & 7) << 3)` (the gfx950
  `(swz_rows, swz_shift) = (8, 3)` period), stride padded to a multiple of 64 so
  the swizzle stays in-row. Feeds GEMM1's A-operand (and GEMM2's B-operand for
  dk/dq).
- **`lds_hidden`** `[block_n, hidden_dim]` — the streamed hidden-dim tensor (`dO`
  for dv/dk, `V` for dq), row-major (no swizzle). Feeds GEMM2's B-operand (dv) or
  the `dA` GEMM's A-operand (dk/dq).

Staging is a vectorized global→LDS pass (`global_load_vN` + `smem_store_vN`,
widest power-of-two vector ≤ 8 that divides the tile). OOB stream rows are clamped
to a valid token (finite data), and the gate zeroes their contribution — so no
explicit vector zeroing is needed.

## 3. The three GEMMs per streamed tile

Operand roles are chosen so the score fragment can be **reused directly** as the
next GEMM's A-operand (see §4). The resident (owned-tile) operands are hoisted
out of the streamed loop into VGPRs.

```
GEMM1 (score, contract head_dim):   C[m=stream, n=own] = Σ_d A_stream[stream,d] · B_own[own,d]
  A = lds_head (streamed),  B = resident own (K for dv/dk, Q for dq)

dA   (dk/dq, contract hidden_dim):  C[m=stream, n=own] = Σ_hd A_stream[stream,hd] · B_own[own,hd]
  A = lds_hidden (streamed dO/V),  B = resident own (V for dk, dO for dq)

gate:  own = lane%16 (N), stream = (lane//16)*4 + r (M);  apply HSTU mask + SiLU/SiLU'
       dv  -> silu(sc);   dk/dq -> (1/N) * silu'(sc) * dA        (kept in registers)

GEMM2 (output, contract stream):    C[m=own, n=out] += Σ_stream A_gate[own,stream] · B_out[stream,out]
  A = gate fragment (reused),  B = lds_head (dk/dq: Q/K) or lds_hidden (dv: dO)
```

`GEMM1`/`dA` are emitted as `C[m=stream, n=own]` (A=streamed, B=resident own).

## 4. Fragment reuse (no LDS round-trip for the score)

For the 16×16×16 atom, feeding an accumulator `C` directly back as an A-operand
computes the **transpose**: `A[i, k] = C[k, i]`. Emitting GEMM1 as
`C[m=stream, n=own]` therefore makes the (cast-to-native) gated fragment usable
*directly* as GEMM2's `A[own, stream]` — exactly the orientation the output GEMM
needs, with **no LDS round-trip and no extra barrier** for the score. This mirrors
FlyDSL's fragment reuse (the earlier Rocke revision round-tripped the gate
through a `[block_m, block_n]` `lds_gate` tile; this version removes it).

Consequence: exactly **2 barriers per streamed iteration** — one after the LDS
fill, one after GEMM2 (the WAR guard before the next tile overwrites LDS) —
matching the FlyDSL ISA.

## 5. Causal range limiting

For pure causal (no window/contextual/targets) the streamed range is clipped at
`block_n` granularity so fully-masked tiles are skipped:

- dv/dk (own kv, stream q, keep `q >= kv`): stream `[floor(own_base/block_n)*block_n, seq_len)`.
- dq (own q, stream kv, keep `kv <= q`): stream `[0, min(own_base + block_m, seq_len))`.

Variants (window/contextual/targets) keep the full `[0, seq_len)` sweep and rely
on the per-cell mask.

## 6. Epilogue

The `OWN_SUB × D_CHUNKS` f32 accumulators are scaled (`1/N` for dv, `alpha` for
dk/dq), cast to the native dtype, and stored with per-row `seq_len` guards. Each
output row is single-writer, so stores are plain (no atomics), deferred to after
the streamed loop.

## 7. Occupancy

- `waves_per_eu` is plumbed through the spec → `amdgpu-waves-per-eu`. On gfx950
  dk/dq are VGPR-bound (benefit from a `wpe=2` cap → ~2 waves/SIMD); dv is
  LDS-bound and wants max occupancy (~4 waves/SIMD).
- LDS budget: `block_n * HEAD_DIM_K * 2 + block_n * hidden_dim * 2` bytes
  (validated ≤ 64 KB).

## 8. Future optimizations

Ordered by expected impact toward closing the remaining ~1.3–1.6× vs FlyDSL (see
[`README.md`](README.md) for current numbers and the ISA report the gap analysis
is based on):

1. **Async `buffer_load_lds` DMA for the streamed head operand.** FlyDSL DMAs
   global→LDS directly (`buffer_load_dwordx4 … lds`, 16 B/lane), freeing the
   staging VGPRs and cutting instruction count. Rocke exposes
   `async_buffer_load_lds_addr` (used by `attention_tiled_2d`); wiring it with the
   XOR swizzle would replace the current `global_load_vN + smem_store_vN`
   register-staged path. **Highest-leverage remaining item.**
2. **Register-prefetch overlap of the hidden operand behind GEMM1.** Prefetch
   `dO`/`V` to VGPRs at the loop top, run GEMM1 (which only needs `lds_head`),
   then `ds_write` the hidden operand to LDS — hiding its global-load latency
   behind GEMM1's MFMAs (FlyDSL's `s_waitcnt vmcnt(N)` staircase). Needs care to
   stay at 2 barriers (safe for dv; dk/dq read `lds_head` in GEMM2 so ordering is
   tighter).
3. **gfx950-specific `waves_per_eu` tuning.** The current tuned CSV is
   gfx942-derived; `dv`'s `wpe=1` hurts on gfx950 (dv wants high occupancy). A
   gfx950 sweep + tuned CSV (dv → no cap / high, dk/dq → 2) is a pure-tuning win.
4. **Partial `s_waitcnt` overlap.** Replace some full `sync()` drains with
   partial `vmcnt`/`lgkmcnt` waits so LDS reads overlap MFMA (FlyDSL's per-`ds_read`
   `lgkmcnt` staircase). Depends on Rocke exposing finer waitcnt control in this
   body.
5. **Wider tiles / 32×32 atoms.** Larger `block_m`/`block_n` and the
   `mfma_f32_32x32x16` hero atom raise arithmetic intensity where VGPRs allow;
   pair with an occupancy sweep.
6. **C++ engine mirror + byte-identity gate.** The kernel is Python-only today.
   Porting `_build_hstu_attention_bwd_tiled` to the C++ engine and blessing the
   byte-identity golden is required before it is production-complete per
   `platform/AGENTS.md`.
7. **fp8 K/V and bf16 packed atomics** for the memory-bound shapes, following the
   FMHA fp8 dequant path (`helpers/mfma_attention._load_kv_dequant_packed`).
8. **Dispatch + heuristic wiring** (tuned `block_m/block_n/num_waves/waves_per_eu`
   CSV like the forward's `hstu_attention_bwd_tuned.csv`) and registry/parity
   coverage before enabling in production dispatch.
