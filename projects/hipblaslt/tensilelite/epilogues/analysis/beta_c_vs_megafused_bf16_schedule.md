# Is the residual-add schedule structurally close to beta*C? (gfx950)

Question: the native `beta*C` GlobalWriteBatch (GWB) epilogue (`D = alpha*A*B + beta*C`) is
believed near-optimal. The MegaFused BF16 epilogue
(`Tensile/Components/Subtile/SubtileMegaFusedEmit.py`) adds a residual mechanism that is
mechanically the same shape — *load a second global operand, combine with the MFMA
accumulator, store bf16* (`H = acc + residual`, store `ResidualOut = bf16(H)`). We want to
know whether the **structure of the residual schedule tracks the beta*C schedule**, and where
it stops tracking it.

Short answer: **at the macro level, yes — the residual path is built on the same GWB
template** (all operand loads issued in a prolog, then a per-tile *decreasing-vmcnt* drain
that interleaves combine and store, with exec-masked stores). **The divergence is entirely
inside the per-tile store body**, which is heavier than beta*C's. Everything below is from the
actual generated ISA for the winner tile MT384x256, MI16x16x32, DepthU64, PGR2, StreamK3,
gfx950 wave64.

## Kernels analysed (both freshly built on `users/fabianmcg/epilogues`)

| Path | Kernel | File |
|------|--------|------|
| beta*C | `Cijk_Alik_Bljk_BBS_BH_UserArgs` (MFE0, UseBeta=1) | `/tmp/benchmark-analysis/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_UserArgs_00/00_Final/caches/b22fe6d671c7/source/build_tmp/SOURCE/assembly/Cijk_Alik_Bljk_BBS_BH_UserArgs_MT384x256x64_MI16iBcZlTWXIvm1nuEJOjADqd2oQR40bkIl6qZ6gFHOpEE=.s` |
| MegaFused BF16 | `Cijk_Alik_Bljk_BBS_H_PRMS_RA_MFE_UserArgs` (MFE1) | `/tmp/mega-analysis/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_H_PRMS_RA_MFE_UserArgs_00/00_Final/caches/909e803aa86a/source/build_tmp/SOURCE/assembly/Cijk_Alik_Bljk_BBS_H_PRMS_RA_MFE_UserArgs_MT384xfVYUnVh8K705iHVAc27rUxfY_C721D_WY2hBX-vU4uU=.s` |

Build-provenance note (verified): the MegaFused emitter is reached only when
`MegaFusedEpilogue: True`. `benchmark.yaml` never sets it (build log: 19×`MFE0`, 0×`MFE1`),
so its BF16 kernel is the *split* emitters, not `SubtileMegaFusedEmit.py`. The MFE1 kernel
above comes from `benchmark_fused_bf16_partialrms_residualadd.yaml` (the YAML `perf_harness.sh`
uses). Both kernels were rebuilt fresh. Register footprint is identical, so there is no
register-pressure divergence: both `.vgpr_count 252`, `.sgpr_count 89`,
`.group_segment_fixed_size 163840`, `.vgpr_spill_count 0`.

## Epilogue boundaries (line ranges)

- **beta*C hot path: 18244–25424.** Runtime takes Beta!=0 → the NonEdge paired store,
  deferred to `label_GW_B1_FD0_VW4_NonEdge_Deferred:` (18244), holding
  `Global Write Beta Batch #0..#7`. (M=8192=21·384+128, 128%8=0 → NonEdge path.)
- **MegaFused fused residual/RMS/gamma pass: 2937–13112.** Opens at `Global Write Elements`
  (2937) with three drain waits (`lgkmcnt(0)` 2939, `lgkmcnt(0)` 2940, `vmcnt(0)` 2941),
  closes with `vmcnt(0)` 13111 + `vmcnt(0)` 13112. `H*gamma` is written back into the
  accumulator; D is then stored (13243–19015) through the **same** GWB subtile dwordx4 paired
  path as beta*C, with beta=0 (no C load).

## Phase-by-phase schedule alignment

Both paths process a tile group as: **[prolog: issue every operand load] → [drain loop:
per-tile decreasing-vmcnt wait → combine → store]**. This is the GWB template; the source even
labels it "GWB decreasing-vlcnt schedule" / "GWB-style split". The alignment:

| Phase | beta*C (batch #0, 18286–…) | Residual (N-group, 3002–…) | Structurally close? |
|-------|----------------------------|----------------------------|:---:|
| **Operand-load prolog** | 12× `buffer_load_dwordx2` C issued back-to-back, one shared addr VGPR `v10`, immediate SRD offsets 0,32,…,352 (18286–18330) | 8× `buffer_load_dwordx2` "R wide" residual issued back-to-back in `_prologResidualLoads` (3002–3039) | **YES** — same prolog-batches-all-loads shape |
| **Drain wait** | `s_waitcnt vmcnt(11)` then `(10),(9),…` decreasing (18429, 18439, …) | `s_waitcnt vmcnt(7)` then `(6),…` decreasing (3046, …) | **YES** — same decreasing-vmcnt drain; loads stay in flight |
| **Convert operand** | `v_cvt_f32_bf16` C→f32 (18430) | `v_cvt_f32_bf16` residual→f32 (3047–3050) | **YES** |
| **Combine** | `v_fmac_f32 acc, C, beta` (18431) | `v_pk_add_f32 acc, residual` + `v_fma_f32 rmsSum,H,H` (3062–3067) | **partly** — same slot; residual also squares H for RMS here |
| **Pack to bf16** | 4× `v_cvt_pk_bf16_f32` (18456–18459) | 2× `v_cvt_pk_bf16_f32` (3071–3072) | narrower (see below) |
| **Exec-masked store** | `s_mov_b64 exec, s[56:57]` → `buffer_store_dwordx4` → `s_mov_b64 exec, -1` (18491–18494) | `s_and_saveexec_b64` → `buffer_store_dwordx2` → straddle guard → `s_mov_b64 exec, saved` (3076–3127) | **NO** — this is where it stops tracking beta*C |
| **Extra (no beta analog)** | — | `v_pk_mul_f32` gamma + `v_accvgpr_write` writeback (3128–3137); later RMS reduction 13117–13242 | n/a |

So the residual schedule *is* close to beta*C for load / wait / convert / combine. It diverges
only in the store body — and note that **beta*C also toggles exec around its stores**
(`s_mov_b64 exec, s[56:57]` / `exec, -1`, 18491/18494), so exec-masking per se is shared; the
difference is *how heavy* the residual store body is.

## Where the store body stops tracking beta*C

### beta*C store body (light): precomputed address + scalar mask + cooperative dwordx4

```
18456 v_cvt_pk_bf16_f32 v140, v[vgprValuC+144], v[vgprValuC+145]  // 4 packs: 8 f32 -> 4 bf16 dwords
18462 ds_bpermute_b32   v140, v144, v140                          // gather partner lane-group (2 waves cooperate)
18466 v_add_u32         v146, v9, v145                            // D addr = colVgpr(v9) + lane_group*8   (NO multiply)
18472 s_lshr_b64        s[56:57], -1, s58                         // OOB mask: scalar, computed once per block
18489 v_permlane32_swap_b32 v140, v142
18491 s_mov_b64 exec, s[56:57]                                    // apply reused scalar mask
18494 buffer_store_dwordx4 v[140:143], v146, s[sgprSrdD...]       // ONE store writes 8 bf16
18494 s_mov_b64 exec, -1
```
Per tile: **0 data-dependent multiplies, 0 `v_cmp`/`v_cndmask`, 1 wide (8×bf16) store, exec set
by a plain `s_mov` from a mask built once per block.**

### Residual store body (heavy): per-tile address recompute + per-element mask + dwordx2 + straddle guard

```
3068 v_mul_lo_u32 v237, s[sgprSizesFree+0], v168   // base0 = token_n * N_hidden   (data-dependent 16-cycle mul, every tile)
3069 v_add_u32    v237, v237, v150                  // + nhBase
3070 v_lshlrev_b32 v237, 0x1, v237                  // *2 (bf16)
3071 v_cvt_pk_bf16_f32 v238, v172, v173             // 2 packs -> 4 bf16 (half beta's width)
3074 v_cmp_lt_u32 s[76:77], v241, s[sgprSizesFree+0]// b64Safe = nhBase+3 < N_hidden
3075 s_and_b64  s[76:77], s[74:75], s[76:77]        // narrow = tokMask AND b64Safe
3076 s_and_saveexec_b64 s[78:79], s[76:77]          // save + narrow exec (heavier than s_mov)
3077 buffer_store_dwordx2 v[238:239], v237, ...     // store writes only 4 bf16
3078 s_andn2_b32 s76,s74,s76 / s_andn2_b32 s77,... / s_and_b64 / s_mov_b64 exec
3082 s_cbranch_scc0 label_mf_roStraddleEnd_m0n0     // per-tile straddle guard (skipped at runtime for N%4==0)
3083..3125 4× { v_cmp, v_mul_lo_u32, v_lshl, v_add, v_cmp, v_lshl, v_add, v_cndmask,
                v_cndmask, v_cvt_pk, buffer_store_short }   // per-element fallback (always assembled)
3127 s_mov_b64 exec, s[78:79]                       // restore
```
Per tile: **1–5 data-dependent `v_mul_lo_u32`, per-element `v_cmp`/`v_cndmask`, 1 narrow
(4×bf16) store, `s_and_saveexec` + a straddle `s_cbranch`, and an always-emitted per-element
fallback.**

## Quantified store-body divergence (grep over the two ranges, reviewer-verified exact)

| Op (store-body cost) | beta*C 18244–25424 | Residual 2937–13112 | Meaning |
|----------------------|------:|------:|---------|
| `v_mul_lo_u32` | 192 (coord setup only) | 532 | data-dependent store-addr multiply per tile/element |
| `v_add_u32` | 240 | 1772 | address arithmetic |
| `v_lshlrev_b32` | 64 | 1021 | byte-scale shifts |
| `v_cmp*` | 0 | 1296 | per-element OOB tests |
| `v_cndmask_b32` | 8 | 1152 | per-element OOB clamps |
| `s_and_saveexec_b64` | 0 | 96 | save+narrow vs beta's `s_mov` from reused mask |
| `s_cbranch_scc0` | 200 (batch guards) | 96 | per-tile straddle guard |
| `buffer_store` width | dwordx4 (8×bf16) | dwordx2 (4×bf16) | residual is half-width |
| `buffer_store` count | 96 | 96 dwordx2 + 384 short (skipped) | + always-emitted fallback |
| `ds_bpermute` / `v_permlane*_swap` | 248 / 112 | 0 / 0 | residual does not cooperatively pack |

Total VALU over each pass: beta*C 2283, residual pass 8346. The delta is dominated by the
store-body address/mask ops above (~4750 of the extra VALU), plus the genuinely-extra residual
work that has no beta analog: `v_pk_add_f32` residual add (192), `v_fma_f32` ΣH² (384),
`v_pk_mul_f32` gamma (192), and the cross-wave RMS reduction (13117–13242).

## Verdict on closeness, and what to change

- **Macro schedule structure: close to beta*C.** Prolog-issues-all-loads, decreasing-vmcnt
  drain, interleaved combine+store, exec-masked stores — all match. The load pipeline is not
  the problem (residual `vmcnt(7)` decreasing == C `vmcnt(11)` decreasing).
- **Store body: not close.** beta*C reaches near-optimal by (1) a **precomputed column-address
  VGPR + SRD/immediate row offsets** (no per-tile multiply), (2) a **scalar OOB mask built once
  per block** (no per-element `v_cmp`/`v_cndmask`), and (3) a **cooperative `ds_bpermute`
  dwordx4** store (8×bf16, exec set by a cheap reused `s_mov`). The residual store instead
  recomputes `token_n*N_hidden` per tile, masks per element, stores dwordx2, and carries a
  per-tile `s_and_saveexec` + straddle branch + always-emitted per-element fallback.

**Highest-leverage change (makes the store body track beta*C):** in `_storeResidualOutRow` /
`_issueResidualOutWide` / `_issueResidualOutStraddle` / `_computeBf16Addr`, adopt the subtile
GWB store idiom the beta path already uses — one precomputed column-address VGPR advanced by
SRD/immediate offsets instead of `token_n*N_hidden` multiplies, the scalar
`SubtileMGuard`/`SubtileNGuard` block masks instead of per-element `v_cmp`/`v_cndmask`, and a
`ds_bpermute`-packed `buffer_store_dwordx4` (8×bf16) with a plain `s_mov` exec instead of
`dwordx2` + `s_and_saveexec` + straddle fallback. That removes ~4750 addressing/masking VALU,
the 96 per-tile saveexec/branch pairs, and the 384 static fallback stores, leaving only the
irreducible extra work (the ResidualOut write stream and the RMS reduction), which is what
actually distinguishes the residual feature from beta*C.
