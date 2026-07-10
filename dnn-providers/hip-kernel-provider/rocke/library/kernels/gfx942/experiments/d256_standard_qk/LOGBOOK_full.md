# D256 gfx942 bf16 prefill optimization — LOGBOOK (AICK-1495)

**Goal:** beat AOTriton (torch-flash / SDPA FLASH on MI300X) for **D256 bf16 prefill, bs16, Sq4096 & Sq8192** on gfx942.
**Mandate (per user + [[design-lessons]] #7):** follow the optimization runbook — Step 0 exhaustive raw-flag sweep, then **rocprof HARDWARE-COUNTER bottleneck classification** (NOT static-only), correctness-gate every change (fp32 ref, tol 2e-2), log every step.

## Ground truth loaded
- gfx942 D256 is the **un-optimized** path: `_d256_gfx950_fast` is gfx950-only, so gfx942 D256 uses the default 16x16 register_pv small-tile. bs16→T=32 fits 64 KB LDS; bs64→T=128 is LDS-infeasible on gfx942 (falls to scalar) → **bs16 is the target**.
- gfx942 facts (arch/gfx942.md): LDS 64 KB / 32 banks; NO transpose-read (`ds_read_tr` gfx950-only); NO wide load-to-LDS (1-dword on CDNA3); dense atoms 16x16x16 & **32x32x8 (K=8)**; no wide-K dense bf16.
- Bench recipe (d256-bench-both-arches): arch-agnostic gfx950 harness `parity_unified_attention.py` on a gfx942 node → builds gfx942 kernel. `--torch-compile`/`--aotriton` = baseline.
- Profiling: `probe_rocprof_single.py` (kernel window under rocprofv3) → `stage4_analyze/analyze_lds_conflicts.py`, `stage5_compare/compare_rocprof_stats.py`. Counters: SQ_LDS_BANK_CONFLICT, VALU/LDS busy, mem latency, occupancy.
- Prior art: gfx942 attention case study (D128) has measured per-opcode LDS conflict periods + rocprof evidence — read for bottleneck priors.
- Colleague's gfx950 win (PR #9260): slab-pad K_lds swizzle broke LDS bank conflicts (897→660), beat torch. Swizzle is in the *gfx950* kernel; would need port if gfx942 is also bank-conflict-bound.

## Step-0 knob space to sweep (gfx942 D256 spec)
tile_size, num_warps, block_m_per_warp, use_mfma_32x32 (→32x32x8), use_transposed_qk_32x32 + transposed subflags, use_q_direct_reg/global, use_register_pv, use_k_single_buffer, use_v_double_buffer, use_early_v_schedule, use_sched_barrier, kv_ring_depth, use_wide_kv_load, use_conflict_free_v_store (cfvst), use_k_sliced_ring, use_k_sliced_ldsseq, kv_cache_policy, use_i64_kv_addr, waves_per_eu, (bank-pad swizzle port). Plus fp8-KV (accuracy-gated).

---
## Journey log
"""

### [setup] Prior-art bottleneck (gfx942 case study, Batch 2.1 rocprof, D128)
gfx942 attention is **LDS-bank-conflict bound**, NOT compute/occupancy: MFMA pipe 4.63% busy (D128),
11.34 conflicts/LDS-inst, ~57% of busy waiting on LDS, inst mix VALU54/LDS20/MFMA7. Hot conflict =
the **strided-V B-operand LDS read** (gfx942 emulates ds_read_tr16 with strided loads → conflicts).
Shipped de-conflict wins: V_lds +8 pad (+2-5% every D128). Killed: blanket P/Acc XOR swizzle (wrong
tile + VGPR blowup), register-PV (1→2 CTA/CU only +2.6% → NOT occupancy-bound), async-DMA width
(CDNA3 1-dword only). Best gfx942 D128 = 63% of flash (wide4). => hypothesis for D256: same
LDS-bank-conflict bound on the strided V (and maybe K) read; confirm with fresh rocprof on D256 bs16.

### [step0] Enumerated gfx942 D256 spec knobs (from kernels/gfx942/attention_tiled_2d.py)
Geometry: num_warps, block_m_per_warp, tile_size, waves_per_eu.
MFMA/transpose: use_mfma_32x32(?), use_transposed_qk_32x32(?), transposed subflags (scalar_state,
  invariant_hoist, mask_once, half_local_pv, mask_limit).
Register/K-buffer: use_register_pv, use_k_single_buffer, use_k_sliced_ring, use_k_sliced_ldsseq.
**V de-conflict (PRIORITY — hits the case-study bottleneck):** use_conflict_free_v,
  use_conflict_free_v_store(+_split,+_ck_vlds), use_early_v_schedule. (V_lds +8 pad already baked/shipped.)
Scheduling: use_iglp_opt, use_qk_pv_sched_group_barrier, use_agpr_alloc_zero.
Load/grid: use_q_direct_global, use_global_load_lds_k, kv_cache_policy(stream/...), use_q_major_grid,
  use_fast_paged_kv_desc.
=> Step-0 sweep will prioritize {V-deconflict} x {geometry: num_warps/tile/block_m} x {register_pv,
   k_single} once baseline+rocprof confirm the bottleneck (expected: strided-V LDS bank conflicts).

### [baseline] RED baseline — gfx942 MI300X (ctr-rack31-mi300x-3), ck-auto (rocKE), fp32-ref PASS max_abs=0.01562
- D256 bs16 Sq4096: **22.00 ms** (ck-auto)
- D256 bs16 Sq8192: **87.52 ms** (ck-auto)
(triton lanes err on a version kwarg mismatch — ignore; aotriton is the target baseline. Next: get aotriton/torch-compile numbers.)

### [baseline] AOTriton (SDPBackend.FLASH) target — same node, dense causal SDPA
- D256 bs16 Sq4096: aotriton **0.935 ms**, torch.compile 0.939 ms  (ck 22.00 ms → **23.5× behind**)
- D256 bs16 Sq8192: aotriton **3.201 ms**, torch.compile 3.202 ms  (ck 87.52 ms → **27.3× behind**)
HONEST ASSESSMENT: ~24-27× gap. gfx942 D256 is the un-optimized 16x16 small-tile path; prior gfx942
effort (case study) topped at 63% of flash on the EASIER D128. Beating AOTriton here almost certainly
needs a structural redesign, not knobs. Plan: rocprof→bottleneck→sweep high-value levers (32x32x8 atom,
bigger tile, V-deconflict, register_pv)→quantify the achievable gain honestly; document the redesign path.
NOTE: forced ck_2d lane errored ("no kernel image") — must confirm via rocprof WHICH kernel ck-auto dispatches.
[discovery] rocprof-compute needs /opt/rocm/.info/version (missing in kreb image) -> ValueError. Fix: write it in-container before profiling.
[discovery] rocprof-compute rocm_version detection fails in kreb image (not fixed by writing .info/version); PIVOT to rocprofv3 raw --pmc counters + manual derived ratios (matches lesson #7 approach).

### [profile] rocprofv3 DYNAMIC counters on the dispatched kernel — BOTTLENECK CONFIRMED
Kernel: `rocke_uattn2d_tiled_d256_b16_t32_h16kv2_bf16_wpe2_regpv` (ck-auto dispatches the 2D tiled kernel).
Occupancy: **num_warps=1** (WG=64), LDS 57344 B (56 KB) → 1 CTA/CU. VGPR 104, AGPR 160.
Counters (avg/dispatch, Sq8192): GRBM_GUI_ACTIVE 1.46e9; SQ_LDS_BANK_CONFLICT **2.30e9**; SQ_LDS_IDX_ACTIVE 2.86e9;
  SQ_INSTS_VALU 7.18e8; SQ_INSTS_LDS 2.20e8; SQ_INSTS_MFMA 6.74e7; SQ_VALU_MFMA_BUSY_CYCLES 1.08e9.
DERIVED: **MFMA util 0.06%** (matrix cores idle); **~4.1 bank-conflict cycles per LDS-idx-active**;
  inst mix VALU 71% / LDS 22% / MFMA 7%.
=> **LDS-bank-conflict bound + latency-exposed (num_warps=1, 1 CTA/CU).** Matches the case-study prior
   (strided-V read conflicts, MFMA-starved). NOTE: a STATIC histogram would read "VALU-bound" (71%) —
   the DYNAMIC counter shows MFMA 0.06% busy => stalled on LDS, not compute. (Lesson #7 vindicated.)
HYPOTHESES (ranked): (1) num_warps 1→2/4 = more resident waves hide LDS-conflict latency (case-study
  wide4 = +19.7% on D128); (2) V-read de-conflict (use_conflict_free_v_store / bank-pad); (3) tile_size
  32→64 amortize; (4) 32x32x8 atom (secondary — MFMA not the bound). Sweep these, correctness-gate.

### [optimize] BLOCKER — num_warps>1 hangs the gfx942 D256 lowerer (the original AICK-1495 failure)
Sweep combo `nw2` built for 378s+ with no output => the documented gfx942 D256 **Python-lowerer hang**
(unrolled head_dim=256 loop → combinatorial IR explosion; ">9 min, killed" per the ticket). The shipped
nw1 t32 works; nw>=2 (and likely bigger tiles) hang. => the #1 hypothesis (raise num_warps to hide
LDS-conflict latency, the case-study wide4 win) is BLOCKED on gfx942 D256 by the lowerer hang.
CONSTRAINT: stuck near nw1/t32/1-CTA-per-CU. Occupancy can't rise (D256 LDS too big for 2 CTA/CU;
nw>1 hangs). Only non-hang lever left = **V-read de-conflict at fixed geometry** (reduce the 4.1
conflict-cycles/access directly). Case-study says V_lds+8 pad already shipped (+2-5%), cfv-store parked.
=> Pivoting to a V-deconflict sweep at nw1 t32 (direct build, unique cache-key per spec).

### [optimize] Lever space EXHAUSTED on gfx942 D256 — verdict: structural work required
Direct-build sweep (nw1 t32 bs16, correct arch via `compile_kernel(...,arch="gfx942")`):
- Base spec: **COMPILE_FAIL** — `AttributeError: no attribute 'use_conflict_free_v_store_split'`.
  Root: the V-deconflict fields are **arch-derived by the dispatcher's spec override**
  (`_tiled_spec_from_problem`), NOT dataclass fields; a directly-constructed spec lacks them.
- `+conflict_free_*` kwargs: **SPEC_INVALID** (unexpected keyword) — same reason (not ctor args).

Authoritative shipped gfx942 D256 spec (dumped from the real dispatcher):
  num_warps=1, tile_size=32, block_m_per_warp=16, use_register_pv=True, waves_per_eu=2,
  use_mfma_32x32=False, **use_conflict_free_v_store_split=True**, **use_conflict_free_v_ck_vlds=True**,
  use_conflict_free_v=False, use_conflict_free_v_store=False, use_k_single_buffer=False. LDS=57344 (56 KB).

=> **The shipped kernel ALREADY applies V-read de-conflict** (store_split + ck_vlds). register_pv already on.
=> num_warps>1 (the case-study wide4 latency-hiding win) is BLOCKED by the lowerer hang.
=> Occupancy can't rise: 56 KB LDS > 32 KB, so 1 CTA/CU; nw>1 hangs. MFMA 0.06% busy.
=> **All available flag-tuning levers are exhausted.** The kernel is structurally LDS-bank-conflict-bound
   at 1 CTA/CU and stays 24-27x behind AOTriton.

### [wrap] FINAL — gfx942 (MI300X) D256 bf16 prefill vs AOTriton (torch.compile FLASH)
| config              | ck-auto (shipped) |     AOTriton |  ratio |
|---------------------|-------------------|--------------|--------|
| bs16 Sq4096         |  22.0 ms (22 TF/s)| 0.93 ms (147 TF/s) | 0.042x (~24x behind) |
| bs16 Sq8192         |  87.5 ms (22 TF/s)| 3.23 ms (170 TF/s) | 0.037x (~27x behind) |
| bs64 (any Sq)       |  LDS-INFEASIBLE (T=128 > 64 KB cap) -> scalar fallback; not benchable as tiled |

Bottleneck (rocprofv3, DYNAMIC counters — lesson #7 applied): LDS-bank-conflict bound,
~4.1 conflict-cycles/LDS-access, MFMA 0.06% busy, VALU71/LDS22/MFMA7, nw=1 -> 1 CTA/CU.

Next levers (all STRUCTURAL — no flag win remains):
1. Fix the head-dim=256 lowerer hang (roll the unrolled loop) -> unlocks num_warps>1 / wider tiles
   -> the case-study wide4 latency-hiding win becomes buildable. (biggest enabler)
2. Port PR #9260 slab-pad swizzle (K/V_lds bank-conflict pad) to gfx942 — orthogonal to occupancy,
   no nw>1 needed, gave +25% (897->660 conflicts/1000) on gfx950. NOT in this tree (no `use_kq_lds_pad`).
3. FA3-style warp specialization (AVD article) — the ceiling-breaker; not in Triton-on-AMD either.

### [BREAKTHROUGH 2026-07-11] 9.2x uplift + profile of the winner
Fix: relax conflict-free-V HD gate (HD in (64,128)->(64,128,256), attention_tiled_2d.py:1720) + route D256
to the K=8 use_mfma_32x32x8 + use_transposed_qk_32x32 + use_conflict_free_v_store path (runtime-rolled =>
sidesteps the nw>1 lowerer hang) + use_k_single_buffer (fit 64KB) + use_iglp_opt (+60%).
Perf @ Sq8192: shipped 87.5ms(6.3TF) -> x8 nw2 T64 ks+iglp 9.5ms(57.8TF) = 9.2x; AOTriton 3.2ms(170TF) -> ~3x behind.
Sq4096: 21.9ms -> 2.68ms (8.2x); AOTriton 0.93ms.
Winner: rocke_uattn2d_tiled_d256_b16_t64_h16kv2_bf16_w2_wpe2_mw32_mfma32x8_stqk_cfvst_iglp1_k1buf
Profile (rocprofv3 + llvm-objdump static):
 - NOW VALU-BOUND: dyn VALU80/LDS14/MFMA6; static 5803 instr, VALU 63%, MFMA 2.2% (128). Top VALU: v_cndmask 649 (mask),
   v_perm+writelane/readlane 672 (transpose shuffles), s_nop 498 (bubbles), v_exp 66.
 - LDS conflicts reduced ~5x (4.12->0.83 conflict/access, LDS insts 220M->77M) but NOT gone: 256 ds_read_u16 remain.
 - Interleave: no explicit gfx950 softmax<->MFMA field on gfx942; iglp_opt (LLVM whole-loop) is the +60% driver;
   explicit qk_pv_sched_group_barrier gave NO gain (35.9 vs 36.1 plain).
Next: port mask-phase-split (cut v_cndmask), widen residual ds_read_u16->b64, raise occupancy (fp8-KV-LDS/smaller tile).

### [2026-07-11 cont'd] flag ceiling reached + approach assessment
Additional levers tested on the winning x8 transposed-cfv path (all correctness-gated PASS):
 - use_causal_mask_phase_split (cmps): +3% (57.7->59.5 TF/s @Sq8192). Correct. Modest — I-cache cost of
   emitting the body twice nearly cancels the mask-VALU saving (matches the case-study's parked note).
   Fix needed: phase-1 iter-args uniquely named (LLVM phi collision in the single flat function).
 - waves_per_eu {2,3,4,auto}: neutral (all ~59 TF/s).
 - use_k_sliced_ring (HD-gate relaxed to 256, VALIDATED correct at HD=256): NET-NEGATIVE (49 vs 59) —
   ring-slice LDS traffic > prefetch benefit; didn't unlock 2 CTA/CU. Gate relax REVERTED.
 - use_agpr_alloc_zero: negative (31 vs 36 earlier). use_qk_pv_sched_group_barrier: no gain vs iglp.
 - nw4/nw8/bigger tiles: LDS overflow (CODEGEN_BC_TO_RELOCATABLE), the 64KB wall.
FINAL winner: nw2 T64 k_single iglp +cmps = 59.5 TF/s / 9.24ms @Sq8192 (9.5x over shipped 87.5ms),
  52 TF/s / 2.64ms @Sq4096. AOTriton 173 TF/s / 3.17ms -> we are ~2.9x behind (was ~27x).

APPROACH ASSESSMENT (why ~2.9x remains; can we improve?):
 Root: VALU-bound at 1 CTA/CU. Three interlocking walls, none flag-fixable:
  1. LDS wall (64KB): D256 tile fills 64KB -> 1 CTA/CU (no 2nd wave) AND no K/V double-buffer (no
     prefetch overlap -> s_nop 8.6% bubbles). fp8-KV-in-LDS (would halve) is validator-blocked on gfx942
     (ds_read_tr_b8 gfx950-only). K-slice fits but net-negative.
  2. Transpose-VALU wall: gfx942 has no ds_read_tr, so conflict-free-V needs an in-register 2x2 transpose
     (v_perm+writelane+readlane = 672 instr ~12%). Price of dodging LDS conflicts without the gfx950 read.
  3. Masking VALU (v_cndmask 649): partly cut by cmps (+3%), I-cache-capped.
 IMPROVABLE ONLY VIA MAJOR REWRITES (not flags): (a) persistent work-stealing grid (AOTriton's causal
 load-balance lever, absent in the DSL kernel); (b) a transpose-free V feed (pre-transposed V in HBM, or a
 K-contiguous tiling); (c) fp8-KV-in-LDS gfx942 path (non-tr PV) -> 2 CTA/CU. Each is multi-day.
 Flag/gate levers are EXHAUSTED; ceiling of the 32x32x8-transposed structure is ~59 TF/s.
 Production follow-up: route D256 gfx942 -> x8 cfv+iglp+cmps in the dispatcher (currently direct-build only).

### [2026-07-11] fp8-KV-in-LDS wall-breaker — PROTOTYPED, PROVEN-NEGATIVE (dead end on gfx942 D256)
Goal: halve K/V LDS via fp8 -> 2 CTA/CU or K double-buffer -> hide the 48% mem-latency stall + LDS waits.
Evidence (built on real gfx942, validator temporarily relaxed in fresh tree; reverted):
 1. use_fp8_mfma_pv COMPILE_FAIL on the emitter's OWN assertion: "32x32x16 PV needs bf16 V in LDS;
    use_fp8_mfma_pv is broken/slower anyway" -> V MUST stay bf16 -> V_lds (36KB) is irreducible.
 2. use_fp8_mfma_qk-only COMPILE_FAIL (COMPILE_SOURCE_TO_BC status=1) on gfx942; and even if it built:
    K_fp8(16KB)+V_bf16(36KB)=52KB -> still 1 CTA/CU; K_fp8_double+V_bf16=68KB -> overflow. No win.
 3. All fp8 paths require kv_storage_dtype='fp8e4m3' (fp8-INPUT model) -> changes the workload; bf16->fp8
    e4m3 quantize (~12%/elem) would blow the max_abs=1.56e-2 bf16 gate.
VERDICT: the binding LDS item is V_lds=36KB (transposed [HD256,T64+8]). fp8 cannot shrink it (PV broken).
D256 T64 is STRUCTURALLY pinned at 1 CTA/CU; the occupancy/prefetch wall is unbreakable via fp8.
=> 2 CTA/CU is closed for D256 gfx942. Remaining structural lever = a transpose-free V feed (shrink the
63% VALU) or AOTriton's different-tiling approach. 9.5x stands as at/near the architecture's D256 ceiling.

### [2026-07-11] transpose-free V feed — STUDIED (AOTriton) + PROTOTYPED (direct-HBM-V) — NET-ZERO, ceiling confirmed
AOTriton V-feed (read modules/flash/kernel/fwd_kernel_inner.py): standard-QK (S=Q@K^T), K loaded
TRANSPOSED, V loaded NATURAL (composed_dot_rhs), and NEITHER K nor V goes through LDS -- streamed
HBM->register per KV-block, L2-cached. That's why AOTriton isn't LDS-capped. Our gfx942 wide path is
hard-locked to transposed-QK (use_mfma_32x32x8 requires use_transposed_qk_32x32), which needs V as
[HD,T] -> V_lds is the cross-lane transpose medium (load coalesced-by-tid, MFMA reads B-distribution).
Scoped 3 transpose-free variants: register-hold (64 VGPR -> can't 2-CTA, case-study R), cross-lane
bpermute (LDS-port serialize, case-study L2 reverted on D128), per-lane HBM (uncoalesced V^T).
PROTOTYPED the per-lane direct-HBM-V feed (use_v_hbm_direct, fresh tree; 6 edits): BUILDS + CORRECT
(max_abs=1.56e-02) but **NET-ZERO** (59.4->59.4 TF/s @Sq8192, byte-identical). Root: freeing V_lds did
NOT drop LDS -- still 64KB, still 1 CTA/CU. Because the 64KB peak = K_lds(32KB) + Acc_lds(32KB,
OUT_STRIPE_COLS=HD=256), with V_lds aliased into that region.
=> **2 CTA/CU is ABSOLUTELY impossible for D256 T64 gfx942**: K_lds alone (32KB) = the entire 2-CTA
budget (<=32KB), leaving zero room for the accumulator. K can't shrink (fp8-K dead, K-slice negative,
T<64 invalid for nw2). The occupancy wall is K_lds itself, not V_lds.
FINAL: 9.5x (9.24ms @Sq8192) is the gfx942 D256 ceiling for this LDS-staged transposed-QK architecture.
The remaining ~2.9x to AOTriton requires AOTriton's architecture (standard-QK, NO K/V LDS, register/L2
streamed) -- a ground-up rewrite, not a modification of this kernel.

### [2026-07-11 pm] AOTriton-match run — architectural walls mapped (retrofit exhausted)
Goal: match AOTriton (173 TF/s) from our 59 TF/s (2.9x gap). Librarian nailed AOTriton's D256
recipe: no-LDS register-streamed K/V/Acc, narrow 16x16x16 atom, BLOCK_M=64/BLOCK_N=32, num_warps=2,
DYNAMIC PERSISTENT grid + atomic work-stealing (causal balance), num_stages=1, VGPR-bound occupancy.

Experiments (all on real MI300X, gfx942):
1. **Register math**: VGPR=128/AGPR=248 -> registers allow 2-4 CTA/CU; LDS(64KB) caps at 1 CTA/CU.
   So LDS is the binding wall; freeing it *should* raise occupancy 2-4x.
2. **Tile-shrink to free LDS** (nw1 T32, 32KB->2CTA): 21.7 TF/s = **2.7x SLOWER**. Freeing LDS by
   shrinking the tile costs more MFMA throughput than occupancy recovers. Wrong way to free LDS.
3. **Q_DIRECT_GLOBAL** (Q->reg, breaks K_lds alias): builds+correct, 56.5 TF/s (enabler, slight overhead).
4. **direct-HBM-V stream** (`use_v_hbm_direct`): the V *read* value is CORRECT (proven with store-on,
   max_abs=1.56e-2). BUT: (a) skipping the V store to free LDS breaks correctness (load-bearing
   coupling — store's LDS writes / pipeline schedule needed even though reads bypass V_lds); (b)
   stubbing V_lds alloc to free it -> comgr `invalid getelementptr` (residual V_lds refs: smem_addr_of,
   passed-by-name, store/slot fns). The V_lds transpose medium is too entangled to remove by retrofit.
5. **Narrow register-PV path** (AOTriton-like architecture, already in kernel): 6.3-20 TF/s = **10x
   SLOWER** than the wide winner; T64 variants won't compile (register spill). AOTriton hits 173 on the
   *same narrow atom* -> the gap is IMPLEMENTATION QUALITY (streaming+scheduling+occupancy), not the atom.

**Definitive conclusion**: Neither existing path can match AOTriton. The fast wide 32x32x8 path (59) is
LDS-walled and the wall CANNOT be removed by retrofit (V_lds transpose medium is load-bearing via
aliasing + pipeline barriers). The narrow register-streamed path (architecturally AOTriton-like) is 10x
too slow (poor implementation). Matching AOTriton requires a **ground-up register-streamed kernel**
(standard-QK or transposed, K/V HBM->reg, no LDS staging, small BLOCK_N=32, persistent atomic grid) —
DSL primitives confirmed available (global_load, mfma atoms, global_atomic_add, scf_for). This is a
multi-day from-scratch kernel, not a modification of the existing one.

### [2026-07-11 pm-2] K-STREAM WIN: +35% (59->80 TF/s), gap to AOTriton 2.9x->2.16x
After mapping the walls, found the real lever: **memory-path, not occupancy**.
- **`use_k_hbm_direct`** (+ `use_q_direct_global`): K read DIRECT from HBM->registers (L2-cached)
  in the QK MFMA operand, SKIPPING the wasteful async DMA HBM->K_lds->reg round-trip. This is
  AOTriton's core insight. Committed `193efa4b` on `users/avirgoel/ck/d256-gfx942-kstream` (local, no push).
  MI300X D256 bf16 causal GQA16/2 (direct-launch vs torch.compile-flash / AOTriton):
    Sq8192: 59.5 -> **80.2 TF/s (+35%)**   Sq4096: 51.6 -> **66.4 (+29%)**   max_abs=1.56e-2 PASS.
- **Occupancy is NOT the lever** (re-confirmed): stubbing K_lds to free 32KB (->2 CTA/CU) gave 0 extra
  perf over K-stream-with-K_lds-allocated. The +35% is 100% from killing the LDS round-trip latency.
- **V-stream is net-NEGATIVE** at long seqlen: `use_v_hbm_direct` is correct (once the naive-path V read
  is gated — TRANSPOSED_V=False under qdir routes V through smem_load_vN(V_lds), not _v_t_load) but the
  V^T A-operand read is uncoalesced (4 strided tokens/lane, stride HD), so V_lds's transpose is
  BENEFICIAL. Sq8192 K+V=75.8 < K-only=80.1. Keep V in LDS. (flag added, off by default.)
- Geometry on K-stream base: T64 nw2 optimal (T32=66.5, T128 unstable). Same sweet spot.

**Remaining ~2.16x to AOTriton** (levers, ranked):
  1. Coalesced V (needs standard-QK S=Q@K^T so V is the natural PV B-operand) — bigger rewrite.
  2. Persistent grid + atomic work-stealing (causal load balance) — global_atomic_add + scf_for exist.
  3. K-read caching across the 2 M-atoms (halve K L2 traffic) — QK-loop hoist.

### [2026-07-11 pm-3] K-stream -> buffer_load (+37%, 81.6 TF/s) + VALU-bound diagnosis
- buffer_load_vN for K (HW OOB->0, no software bounds-select; causal mask already zeros past-seq
  keys): 80.2 -> **81.6 TF/s** @Sq8192 (+37% vs 59.5 ship), 67.3 @Sq4096. Correct. Commit 5944b871 (amended).
- **rocprofv3 on K-stream: VALU-BOUND.** VALU:MFMA = **13.6:1**; LDS bank-conflict = **0**; L2 hit 80.7%;
  VMEM low. Memory bottleneck is GONE (K-stream fixed it) -> now pure VALU throughput wall.
- Implication: occupancy CANNOT help (hides memory, not VALU). Coalesced-V rewrite would help only via
  transpose-VALU removal (~12% static), NOT memory. Masking ~11%. No single lever = 2.16x.
- **Verdict on the remaining 2.16x to AOTriton**: both kernels are VALU-heavy (softmax); AOTriton's VALU
  is ~2x more efficient + better MFMA-overlapped (narrow atom, small tiles, num_stages, no transposes).
  Closing it = an AOTriton-quality register-streamed standard-QK kernel (VALU reduction across masking +
  transposes + conversions + overlap) = multi-session ground-up work, not incremental flags.
- **Delivered this run: +37% (memory-path K-stream), VALU-bound diagnosis, precise remaining path.**

### [2026-07-11 pm-4] standard-QK rewrite BEGUN — design + scaffold (body is multi-session)
- **Design spec complete**: wiki/SDPA/d256-gfx942-standard-qk-rewrite-design.md. Key derived result:
  standard-QK (S=Q@K^T) needs **NO operand transpose** on the 32x32x8 atom — both A(Q) and B(K) read
  contiguous dims (dim is the MFMA K axis, stored contiguous) -> both coalesced. PV: A=P, B=V natural
  (coalesced) -> acc[q,dim] natural -> natural epilogue. Eliminates V_lds transpose store + V^T strided
  reads + O^T epilogue transpose (the bulk of the transpose VALU). Only the P reshape (C->A) remains
  (~= today's P^T reshape). 7-step impl plan in the spec.
- **Scaffold done**: use_standard_qk flag + validation (requires 32x32x8, mutually-excl transposed_qk,
  bf16) + honest NotImplementedError guard in the builder (path not wired yet). In /scratch working tree.
- **Body NOT implemented** (honest): the QK/softmax/PV/epilogue natural-orientation body is a large
  reorientation of the intricate transposed softmax(1000+ lines: hoisted masking, ALiBi, lane^32
  exchanges) + PV + epilogue. Genuinely multi-session; cannot be partially correctness-gated. Did NOT
  churn out untested code (no half-built work).
- **K-stream +37% deliverable regression-verified intact** (81.6/67.1 TF/s, max_abs=1.56e-2). Commit 5944b871.
- Fresh 6h node held (67529762) for the next-session body implementation against the design blueprint.

### [2026-07-11 pm-5] standard-QK REWRITE — QK lane math VALIDATED on GPU
- Built standalone stdqk_qk.py: S[q,kv]=Q@K^T via 32x32x8 (f16, gfx942-legal), A=Q + B=K BOTH loaded
  load_a_row_major_contiguous (square atom -> m==n dist, contiguous d). **max_abs_err=1.5e-05 -> QK CORRECT.**
- Proves the core design: standard-QK needs NO QK transpose (both operands row-major contiguous).
- Reused mfma_gemm_inner helpers (decode_mfma_lanes, load_a_row_major_contiguous, mfma_k_loop,
  store_acc_to_global) — the GEMM scaffold makes the attention kernel tractable.
- Note: mfma_gemm_inner bf16 dispatch picks gfx950-only 32x32x16; used f16 for lane-math validation
  (identical C-dist per the atom docstring). Real kernel uses raw mfma_f32_32x32x8_bf16 (.1k).
- Next: online softmax on S[q,kv] (row-wise), P reshape C[q,kv]->A[q,kv], PV acc=P@V (V load_b
  col-strided = wave-coalesced), natural epilogue.

### [2026-07-11 pm-6] standard-QK REWRITE — FULL PATH VALIDATED end-to-end on GPU
- stdqk_attn.py (saved as stdqk_attn_poc_validated.py): minimal end-to-end standard-QK attention
  (1 q-tile[32] x 1 kv-tile[32], D=256, f16, non-causal) = **ATTN CORRECT, max_abs_err=1.86e-04** vs torch SDPA.
- PROVES the entire standard-QK architecture works:
  * QK: S=Q@K^T natural (A=Q, B=K both row-major contiguous, NO transpose).
  * softmax: row-wise on S[q,kv] via LDS bridge (write C-layout -> LDS, softmax rows, read back).
  * P reshape C[q,kv]->A[q,kv]: read P from LDS as A-operand (smem load_a). Correct.
  * PV: acc[q,d]=P@V, V via load_b_col_strided (wave-coalesced, NO V_lds, NO V^T transpose).
  * epilogue: acc/l stored natural [q,d] (NO O^T transpose).
- Built on mfma_gemm_inner helpers. Gotchas fixed: smem_load_vN indices positional (store uses list);
  fmax not cmp_gt for f32; unique iv_name/acc_name per mfma_k_loop (SSA collision).
- **The rewrite is de-risked: design proven correct end-to-end.** This is a minimal UNOPTIMIZED PoC
  (row-serial LDS softmax, single tile, f16) — NOT yet a production kernel and NOT yet benchmarked/faster.
- Remaining for production: multi-tile online softmax, causal, GQA, bf16, multi-q-block grid, and
  optimization (efficient cross-lane softmax, K/V streaming) to beat 81.6 / approach AOTriton 173.

### [2026-07-11 pm-7] standard-QK — SCALABLE runtime kv-loop, causal, grid (correct); perf baseline measured
- stdqk_attn_mt.py: multi-tile online softmax (running m/l + acc rescale) — CORRECT SK=64 (1.33e-04), SK=128 (1.09e-04).
- stdqk_attn_grid.py: +causal +multi-q-block grid (Python-unrolled) — CORRECT SQ=128 (2.8e-04). BUT SQ>=2048 compile
  EXPLODES (unrolled 64 kv-tiles x 8 d-tiles -> huge IR, >300s comgr). => unroll doesn't scale.
- stdqk_attn_loop.py (saved stdqk_attn_loop_validated.py): **runtime kv-loop via scf_for + acc in LDS** ->
  constant kernel size, scales to any Sq, causal work-skip (upper=qblk+1). CORRECT at SQ=512/2048 (2.8e-04).
  Uses scf_for as context manager (all state in LDS; no carried values).
- **PERF BASELINE (naive, unoptimized):** SQ=512 -> 0.72 TF/s; SQ=2048 -> 2.99 TF/s (causal-counted).
  ~20-100x off the 81.6 TF/s K-stream. Dominant costs (measure-first):
  1. **scalar-LDS softmax**: ~4096 scalar ds_read/tile (64 lanes x 64 loads; both lane-groups redundantly
     recompute the same 32 rows). Needs **cross-lane register reduction** (DPP/permute on C-layout), no LDS round-trip.
  2. **acc in LDS RMW**: 256 LDS ops/tile. Needs **register-carried acc via scf_for_iter** (loop-carried iter_args).
  3. **occupancy**: 1 wave64/CTA, 32-query blocks -> few CUs busy at small Sq. Needs BLOCK_M>=64, multi-warp.
  4. No K/V double-buffering / streaming / prefetch.
- STATUS: standard-QK ALGORITHM fully proven (QK, online softmax, causal, grid) AND scalable. The perf gap is the
  OPTIMIZATION phase = essentially AOTriton's structure (register-acc, cross-lane softmax, big blocks, streaming) +
  GQA + bf16. Genuinely multi-session. K-stream +37% (81.6, 5944b871) remains the banked deliverable.

### [2026-07-11 pm-8] standard-QK — 2 optimizations measured; bottleneck = 1 wave/CU (no latency hiding)
- Perf progression (SQ=2048 causal, f16, single head, MI300X):
  | version                         | TF/s | note |
  |---------------------------------|------|------|
  | runtime-loop, acc in LDS        | 2.99 | naive scalable baseline |
  | + vectorized softmax (ds_b128)  | 3.24 | +8%  (softmax LDS not dominant) |
  | + register-acc (scf_for_iter)   | 3.67 | +15% (frees 32KB acc_lds) |
- All CORRECT (2.8e-04). Saved stdqk_attn_regacc_validated.py (register-carried 128 acc iter_args + vectorized softmax).
- **Bottleneck diagnosis:** still ~22x off 81.6 K-stream. Root = **1 wave64/CTA + 3 sync()/kv-iter -> zero latency
  hiding** (the project's core lesson again). Register-acc freed LDS but raised VGPR (128 acc regs) so occupancy
  didn't jump. Softmax-LDS was NOT dominant (+8% only).
- **Remaining levers (all structural, multi-session):** (1) multi-wave CTAs (BLOCK_M>=64, 128+ threads = 2+ waves
  to hide latency across the syncs); (2) async K/V double-buffer/stream (prefetch next tile during compute);
  (3) cross-lane register softmax (kill the LDS bridge entirely); (4) higher occupancy (balance VGPR vs CTAs/CU);
  plus GQA + bf16. This = AOTriton's full structure.
- HONEST STANDING: standard-QK ALGORITHM fully proven + scalable + 2 opts (2.99->3.67 TF/s). Reaching 81.6 (let alone
  173) needs the multi-wave + async-pipeline production build. K-stream +37% (81.6, 5944b871) remains the banked win.

### [2026-07-11 pm-9] standard-QK BREAKTHROUGH — bottleneck was GRID-STARVATION, not the kernel
- Added head dim to grid (grid=(NQB,H,1)) = real workload. TF/s JUMPED (causal, f16, MI300X):
  | shape                  | single-head | +heads(H=16) |
  |------------------------|-------------|--------------|
  | SQ=2048                | 3.67        | **43.9**     |
  | SQ=4096                | (n/a)       | **55.8**     |
  - CORRECT (4.2e-04). blocks=NQB*H (1024 / 2048) >> 304 CUs -> occupancy filled.
- The single-head benchmark was PESSIMISTIC (64 blocks / 304 CUs = 1 CTA/CU). The naive standard-QK kernel is
  **already ~0.68x the 81.6 K-stream** at SQ=4096 with the real multi-head grid, unoptimized.
- Saved stdqk_attn_mh_validated.py.
- Re-diagnosed remaining levers (occupancy now filled by grid): per-iter overhead = LDS-bridge softmax
  (write S_lds -> reduce in LDS -> P-reshape via LDS) + 3 sync()/iter, and no async K/V prefetch.
- STRUCTURAL REWRITE (starting): (1) cross-lane REGISTER softmax on the C-layout -> kills S_lds write, LDS reduce,
  P-reshape-via-LDS, and 2 of 3 syncs; (2) async K/V double-buffer (num_stages pipeline, AOTriton's lever).

### [2026-07-11 pm-10] standard-QK OPTIMIZATION SWEEP — 73.2 TF/s (0.90x K-stream), profiler-guided
- Full sweep (SQ=4096 causal, H=16, MI300X; TF/s causal-counted):
  | step                          | TF/s  | delta | keep |
  |-------------------------------|-------|-------|------|
  | single-head (grid-starved)    | 3.67  |  --   |  -   |
  | + head dim in grid            | 55.8  | +14x  |  Y   | occupancy: blocks>>304 CUs
  | + S_lds pad [32,33]           | 57.7  | +3%   |  Y   | kills LDS bank conflict (29.5->0/MFMA)
  | + softmax guard lane<32       | 73.2  | +27%  |  Y   | drop redundant 2x softmax (both lane-groups)
  | + iglp_opt(1)                 | 58.2  | -20%  |  N   | reshuffles poorly (unlike K-stream)
  | + mask-once at raw-S write    | 59.2  | -19%  |  N   | moves VALU off guarded path -> all-lane crit path
  | cross-lane reg softmax (alt)  | 40.7  | -30%  |  N   | butterfly 160 swizzles > vectorized LDS; +VGPR
  | Q-in-LDS (alt)                | 51.6  | -8%   |  N   | +16KB LDS hurts occ; L2 already absorbs Q reuse
- **BEST = stdqk_attn_BEST.py (grid+pad+guard) = 73.2 TF/s @ SQ=4096, ~0.90x the 81.6 K-stream.** CORRECT (4.4e-04).
- **rocprofv3 (BEST):** VALU:MFMA = **11.8:1** (VALU-bound, same class as transposed K-stream), LDS bank conflict **0**.
- KEY FINDINGS: (1) the whole earlier single-head perf story was GRID-STARVATION, not the kernel; real multi-head
  grid -> 14x. (2) standard-QK is ALSO VALU-bound (softmax/rescale/exp dominate; transpose was only ~12%), so it
  reaches near-parity with the optimized transposed K-stream but doesn't blow past it. (3) remaining VALU
  (acc-rescale 128 fmuladd/iter + exp) is largely FUNDAMENTAL to D256 online-softmax attention (AOTriton pays it too).
- Path to >81.6 / ->173: cut fundamental VALU (fewer exp via 2^x tricks already used; reduce rescale) or overlap
  VALU/MFMA better (iglp lost here; needs manual sched_group_barrier tuning) + bf16 + GQA + multi-wave BLOCK_M>=64.

### [2026-07-11 pm-11] standard-QK REAL WORKLOAD (bf16 + GQA 16/2) + direct AOTriton bench
- Switched to REAL dtype+config: bf16 (MfmaAtom.bf16_32x32x8 — CDNA3-legal on gfx942, the helper wrongly picked
  gfx950-only 32x32x16), GQA 16/2 (kv_head=q_head//8), D256, causal = the exact Qwen3-Next gated_attn prefill.
- All CORRECT (bf16 max_abs ~3.5e-3 vs fp32 ref, excellent).
- **Direct bench vs AOTriton (torch SDPA flash backend on ROCm, enable_gqa, same shapes):**
  | Sq   | ck-DSL std-QK | AOTriton  | ratio  |
  |------|---------------|-----------|--------|
  | 4096 | 80.9 TF/s     | 151.4 TF/s| 0.53x  |
  | 8192 | 96.2 TF/s     | 174.8 TF/s| 0.55x  |
- GQA gave a further bump over plain multi-head (71.9->80.9 @4096, ->96.2 @8192) because K/V shared across 8 q-heads
  -> heavy L2 reuse (2 kv-heads vs 16).
- **HONEST STANDING:** the from-scratch standard-QK kernel MATCHES/EXCEEDS the 81.6 K-stream (the shipped deliverable)
  at the real workload, but is **~0.53x AOTriton (151-175 TF/s)**. The 2x gap is real and confirmed measured.
- Saved stdqk_attn_REALWORKLOAD.py. AOTriton's edge = num_stages SW pipeline + persistent grid + tuned Triton
  codegen (per librarian study) — the genuine multi-day levers to close the 2x.
