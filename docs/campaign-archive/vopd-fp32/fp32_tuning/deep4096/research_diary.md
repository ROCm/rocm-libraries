# Deep 4096x4096x4096 TN Optimization — Research Diary

**Target shape:** 4096x4096x4096, **TN orientation** (transA=T transB=N, library Cijk_Alik_Bljk).
**Goal:** beat the current best. Iterative — small experiments, think between, log insights here.
**HW:** RX 7900 XTX gfx1100, 96 CU, FP32 VOPD dual-issue peak ~61.4 TFLOPS.

## How I work here (process)
- ONE small Tensile experiment at a time (focused fork grid), then COLD-CACHE confirm the top
  candidate(s) via hipblaslt-bench best-of-3 (transA=T transB=N). Cache-warm Tensile numbers
  mislead (MT128x256 looked great cache-warm but lost cold-cache — see TN refine history).
- Write hypothesis + result + insight to this diary EVERY iteration. Read it back before each new
  iteration. Diary is the durable memory (survives context loss).
- GPU: one job at a time. Always pkill hipblaslt-bench before timing. Warm GPU first. best-of-3.
- NEVER deploy/commit a winner without user OK. Trials merge onto a COPY; deployed TN reverted after.

## Baseline (2026-06-04 ~00:55)
- **TN 4096^3 current deployed heuristic = ~28,900 GFlops (28.9 TFLOPS)**, reproducible (28852-28915).
- Winning config: **MT128x128x16, TT8x16, WG16x8, DU16, GRVWA/B2, VWA/B2, LRVW2, LPA2 LPB2, PGR1, PLR1, SIA1**.
- Context: NT on the same 4096^3 does ~33,800 GF. **TN is ~15% behind NT** → orientation asymmetry
  (TN = A^T·B). This gap suggests TN has recoverable headroom. 28.9 TF = ~47% of VOPD peak.

## Key knowledge carried in (prior TN work)
- TN winners cluster: MT128x128/MT128x64, DU16, GSU1, WGM4/8, VW2.
- MT128x256/256x128: cache-warm strong but COLD-CACHE ~5% WORSE on TN (confirmed, reverted). Avoid.
- TransposeLDS=1 rejected/lost on TN. DU=24 valid but never wins. WGM 1/16/32 minor.
- Obvious big-tile + TLDS levers exhausted. Need finer / less-obvious levers.

## Hypotheses (ordered)
1. DU sweep at MT128x128: TN uses DU16; does DU8/32 help?
2. WG shape: 16x8 vs 8x16 vs 16x16 vs 32x8.
3. PrefetchLocalRead (PLR) + ScheduleIterAlg (SIA): latency hiding on A^T load path.
4. LdsPad A/B fine {0,2,4,8}: TN transposed-A read bank conflicts.
5. GRVW / WaveSeparateGlobalRead: TN global load pattern.
6. 1LDSBuffer / DirectToLds on non-TLU side (TN has TLUA=false → A is transpose-load side).
7. StaggerU {0,32,64,128} DRAM bank diversity.

## Iterations
(append each below: hypothesis -> config -> cache-warm -> cold-cache -> insight)

### Iter 1 (00:54) — DU sweep at MT128x128 (DU 4/8/16/32/64), WGM1, SU32, LPA/LPB {0,2}
- Cache-warm ranking: **DU16 best** (21820) > DU8 (21487). DU32/64 didn't make top-8 → worse.
  LPA2+LPB2 clearly best (21820 vs LPA0/LPB0 19306). So DU16 + LPA2 + LPB2 confirmed optimal pair.
- **COLD-CACHE = 29604 GF vs baseline 28900 → +2.4%.** Winner MT128x128x16 LPA2 LPB2, same as
  baseline EXCEPT this trial pinned **WGM=1 and StaggerU=32**; baseline entry was WGM?/SU?.
- INSIGHT: DU8/32 do NOT help (hypothesis 1 rejected — DU16 is right for TN 4096^3). BUT a small
  real gain appeared, and the only deltas vs baseline are WGM=1 + SU=32. → Hypothesis: **WGM=1
  and/or StaggerU is the lever**, not DU. Next iter: isolate WGM and StaggerU.
- Note: cache-warm 21820 is far below cold-cache 29604 here because this single-shape cache-warm
  run had low EnqueuesPerSync warmup; trust cold-cache number (29604).

### Iter 2 (00:58) — isolate WGM {1,2,4,8} x StaggerU {0,16,32,64} at MT128x128x16/LPA2/LPB2
(first attempt aborted: StaggerU=128 invalid — valid SU = [0,2,4,8,16,32,64]. Fixed.)
- Cache-warm top: **WGM1+SU0 (24269)** > WGM1+SU32 (23869) > WGM2+SU16 (23804). Clear pattern:
  **WGM=1 dominates** (top 2 and 5 of top 8 are WGM1); higher WGM (4,8) ranks lower.
  StaggerU: SU0 slightly best, SU16/32/64 all close — StaggerU barely matters here.
- **COLD-CACHE = 30195 GF vs baseline 28900 → +4.5%.** Best so far.
- INSIGHT: **WGM=1 is the lever** (confirmed — hypothesis 2 partial win). Why: 4096^3 is large and
  square; WGM (workgroup re-mapping for L2 locality) helps L2 reuse on rectangular/skinny shapes,
  but on a big square it adds scheduling overhead with little locality benefit → WGM=1 (no remap)
  wins. StaggerU near-neutral (square shape, DRAM banks already well-spread). The deployed baseline
  likely used WGM4/8.
- STATE: best config = MT128x128x16, TT8x16, WG16x8, DU16, LPA2, LPB2, **WGM1, SU0**, GSU1 → 30195 GF.
- NEXT: with WGM1 locked, explore the latency-hiding axis (PLR depth, SIA, PGR) — hypothesis 3.

### Iter 3 (01:00) — latency hiding: PGR{1,2} PLR{0,1,2} SIA{1,2,3} 1LDS{0,1} at WGM1/SU0
- Cache-warm top: **PGR1+PLR1+SIA1 (23982)** = the default. PGR2/PLR2 slightly worse (23524),
  PLR0 worse (22802). SIA2/3 and 1LDSBuffer=1 didn't make top-6 → no help or rejected.
- COLD-CACHE = 29930 (≈ iter2's 30195, same WGM1/SU0 winner, within noise).
- INSIGHT: latency-hiding params ALREADY optimal (PGR1/PLR1/SIA1). Hypothesis 3 rejected. The A^T
  load latency is not fixable via prefetch depth here. So TN's gap vs NT is NOT prefetch.
- STATE unchanged: best = MT128x128x16/WGM1/SU0/LPA2/LPB2 ~30200 GF (+4.5% vs baseline).
- NEXT (hypothesis 4/5): the LDS/read path. TN has TLUA=false (A is transpose-loaded). Try finer
  LdsPadA {0,2,4,6,8} (bank-conflict tuning on the transposed A read) and GlobalReadVectorWidth /
  WaveSeparateGlobalRead. This is the orientation-specific axis most likely to close the NT gap.

### Iter 4 (01:04) — LDS padding fine: LdsPadA{0,1,2,3,4,8} LdsPadB{0,2,4} WSGRA{0,1} at WGM1/SU0
(first attempt aborted: LdsPadA=6 invalid — valid LPA = [-1,0,1,2,3,4,8,16,32,48,64]. Fixed.)
- Cache-warm top: **LPA2+LPB4 (20962)** > LPA2+LPB2 (20896) > LPA4+LPB2 (20792). WaveSeparate
  GlobalReadA=1 did NOT make top-8 → no help. Odd pads (1,3) didn't win. LPB4 is a mild but real
  improvement over LPB2 on the B-side read.
- **COLD-CACHE = 30400 GF vs baseline 28900 → +5.2%. NEW BEST.** (was 30195 at LPB2.)
- INSIGHT: LdsPadB=4 (not the deployed LPB2) shaves a bit more — a B-side LDS bank-conflict win.
  LdsPadA stays 2. WaveSeparateGlobalRead irrelevant. So the read-path win is small (~0.7% over
  iter2) and on the B side, not the transposed-A side as I guessed.
- STATE: best = MT128x128x16, TT8x16, WG16x8, DU16, **LPA2, LPB4, WGM1, SU0** → 30400 GF (+5.2%).
- Cumulative picture: the gains are WGM1 (big, ~+4%) + LPB4 (small, ~+1%). Tile/DU/prefetch all
  already optimal. TN's residual gap vs NT (33.8 TF) is likely intrinsic to the A^T data path, not
  closable by these knobs. Diminishing returns appearing.
- NEXT: tool flaw noted — iterate.sh benches a STALE lib if Tensile aborts (NO CSV). Add a guard.
  Then iter5: test a couple genuinely different angles (GlobalReadVectorWidth 4, AssertFree/summation
  multiples, MT128x128 vs MT128x64 at WGM1) to see if anything beats 30400; else declare converged.
- (guard added to iterate.sh: aborts cleanly on missing CSV instead of benching stale lib.)

### Iter 5 (01:07) — TT{8x16,16x8,8x8} x VWA{2,4} x VWB{2,4} at WGM1/SU0/DU16/LPA2/LPB4
- Cache-warm: **TT8x16+VW2 (24096)** dominates. VWA=4/VWB=4 MUCH worse (GRVWA4 → 15752, ~35% down)
  — wider global reads hurt (likely alignment/occupancy). MT256x64 (from TT16x8) = 22669, MT128x64
  (TT8x8) = 20139, both lose to MT128x128.
- COLD-CACHE = 29560 (noise band of the 30400 best; same winning kernel family).
- INSIGHT: VW2 confirmed optimal (VW4 bad). TT8x16 / MT128x128 confirmed best tile. Nothing beats
  iter4. **Search has CONVERGED.**

## CONVERGENCE SUMMARY (after 5 iterations)
- **Best found: MT128x128x16, TT8x16, WG16x8, DU16, LPA2, LPB4, WGM1, StaggerU0, GSU1, VW2, PGR1, PLR1.**
- **Cold-cache 4096^3 TN: ~30400 GF (30.4 TFLOPS) vs deployed baseline 28900 → +5.2%.**
- The win decomposes as: **WGM=1** (the big lever, ~+4%; deployed used higher WGM) + **LdsPadB=4**
  (small, ~+1%; deployed used LPB2). Everything else (tile, DU16, prefetch, VW2, StaggerU) was
  already optimal in the deployed kernel.
- Rejected levers (confirmed not helpful for TN 4096^3): DU 8/32/64, bigger tiles (MT128x256/256x128
  — also lost cold-cache earlier), TransposeLDS, VW4, WaveSeparateGlobalRead, SIA2/3, PGR2, PLR2,
  1LDSBuffer, odd LdsPad, GSU2.
- TN's residual gap vs NT (33.8 TF) appears intrinsic to the A^T data path; not closable by the
  Tensile knobs swept. ~30.4 TF is the practical TN ceiling for this config family.
- DELIVERABLE: a +5.2% improvement for TN 4096^3 is available by setting WGM=1 + LPB=4. Candidate to
  fold into TN logic's 4096^3 entry — PENDING USER OK (no deploy/commit yet). Best logic saved at
  iter4_merged/. Could also re-test whether WGM=1 helps OTHER large square TN shapes (likely yes).

### Iter 6 (01:10) — does WGM=1 GENERALIZE? WGM{1,4} x LPB{2,4} across 5 large shapes (cache-warm)
Per-shape WGM1 vs WGM4 (best LPB each):
- 4096x4096x4096:  WGM1 24036 vs WGM4 23260 → **WGM1 +3.3%**
- 2048x8192x8192:  WGM1 27972 vs WGM4 26625 → **WGM1 +5.1%**
- 2048x2048x2048:  WGM1 11770 vs WGM4 11962 → ~tie (WGM4 +1.6%)
- 4096x8192x8192:  WGM1 29509 vs WGM4 30665 → **WGM4 +3.9%**
- 8192x8192x8192:  WGM1 30302 vs WGM4 32360 → **WGM4 +6.8%**
- **KEY INSIGHT: WGM=1 does NOT generalize — it's SHAPE-DEPENDENT.** WGM1 wins the mid-large squares
  (4096^3, 2048x8192^2); WGM4 wins the BIGGEST (8192^3, 4096x8192^2). Rationale: bigger problems
  launch many more workgroups, so L2-locality remapping (WGM>1) pays off; mid shapes lack enough
  workgroups to benefit and WGM1's lower overhead wins.
- CONSEQUENCE: a blanket "set WGM=1" would REGRESS 8192^3 (which we just fixed!) and 4096x8192^2.
  The right move is PER-SHAPE: WGM=1 only for the shapes where it wins. The grid-tuned logic already
  picks per-shape winners — so the real improvement is to ensure 4096^3's entry uses WGM1+LPB4.
- This is exactly why per-shape tuning (not global params) matters, and why cold-cache + multi-shape
  checks are essential before deploying any "insight".
- REVISED DELIVERABLE: WGM1+LPB4 is a confirmed +5.2% specifically for 4096^3 (and likely 2048x8192^2).
  Do NOT apply globally. To capture it: re-tune just the WGM1-favoring shapes and merge their entries.
  PENDING USER OK.

## RETUNE + MERGE ATTEMPT (later 2026-06-04) — ABORTED, result was NOISE

User asked to retune the WGM1-favoring shapes and merge. Did it carefully; outcome below.

### What happened
1. Retuned 10 large shapes with WGM{1,4,8}xLPB{2,4} fork → Tensile picked per-shape winners
   (4096^3→WGM1, 8192^3→WGM4, etc. — confirmed shape-dependence again).
2. Merged all 10 onto deployed TN → A/B showed 4096^3 +4.8% BUT 3072x4096x4096 -6.8%. Reverted.
3. Root-caused: 3072 was ALREADY an exact entry with a good kernel; the broad merge OVERWROTE it
   with a worse one. 4096^3 was NOT an entry (resolved via nearest-K) → adding it helped.
4. Surgical retry: merged ONLY 4096^3 (1 size,1 sol added). A/B then showed DIFFERENT shapes moving
   (4096x8192^2 -6%, 2048^3 +7%) — shapes the merge never touched. RED FLAG.

### Root cause: MEASUREMENT NOISE > EFFECT SIZE
Benched the SAME lib (lib_before) on 4096^3 six times: **27641, 28255, 28118, 27893, 28528, 28084
→ ~3.2% spread on identical binary.** Paired interleaved A/B (before/after back-to-back to cancel
drift) then showed 4096^3 at -3.3% — OPPOSITE of the +4.8% from the first (non-interleaved) test.
Also absolute levels drifted DOWN over the ~10h session (baseline was 28900-30467, now ~28000),
indicating GPU/system thermal/clock state changed.

### CONCLUSION
The ~3-5% "improvements" are **within the measurement noise floor** on this machine/session and
CANNOT be reliably confirmed. Earlier single-shot before/after deltas (iters 1-6) were misleading
because before/after were measured ~minutes-to-30min apart, so GPU drift contaminated them.
- WGM=1's shape-dependence (iter6) is still a real qualitative insight (Tensile picks it for some
  shapes when given the choice).
- But there is NO deployable, noise-robust win for TN 4096^3 from these knobs. NOT deploying.
- Reverted everything. Repo CLEAN at d6c855b6e91. TN/NT/NN unchanged from committed.

### METHODOLOGY LESSON (for future tuning on this box)
- Noise floor here is ~3% on large shapes. Only trust improvements >~5% AND confirmed via PAIRED
  INTERLEAVED bench (before/after back-to-back per shape), not sequential phases.
- Build both lib versions up front, keep them, and swap HIPBLASLT_TENSILE_LIBPATH between them in a
  tight loop (see paired_ab.sh). Sequential "bench all-before then all-after" is invalid — GPU
  drifts between the halves.
- Consider locking clocks / longer cooldowns if chasing sub-5% effects.
