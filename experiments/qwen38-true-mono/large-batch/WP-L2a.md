# WP-L2a: the productized large-batch GEMM core (DESIGN-LB E1), MXFP4 W4A8, rows 5-128, TP1 and TP2, gfx950

W7, 2026-11-01. `../DESIGN-LB.md` 15.6 (WP-L2a), 15.4 (at-risk small GEMMs), 4.3 (unit plans), 6 (E1); `P-L3.md` (the
probe this productizes, with its fix round); FlyTape's `blocks/tiled.py` and `models/qwen4/bodies_tiled.py` (the plans
and restructures studied). Ratios only here. Absolute times, job logs, WaveScope captures and raw JSON:
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2a.md` and `.../large-batch/wpl2a-raw/`. Per-cell tables,
the climb log and the lever ledger: `WP-L2a-tables.md`.

This is the second edition. The first froze the plan table after 6 climb rounds (its tables: `WP-L2a-tables-r6.md`). This
edition adds:
- the 32x32x64 MFMA lever (built, proven, loses);
- 10 more climb rounds, 8 of them with every pair of levers;
- a frozen table re-confirmed on both nodes and in the container, against the round-6 table in the same jobs.

## Gate and verdict

**Gate: met** (DESIGN-LB 15.6). The gate requires:
- P-L3's numerics suite, including the op_sel mutants;
- mutants for every new lever;
- a stale-read test through the grid barrier.

The in-situ ratios of DESIGN-LB 15.2 held, and every one improved.

- **The at-risk TP2 small GEMMs are closed in phase form**, at every row count, on both nodes. The metric is phase time
  over FlyTape's own traced phase per layer, with FlyTape's step traced in the same job before and after the LB timing
  (container job C3). The out-proj is compared with the faster of FlyTape's `gdn_out` and `qsa_out` phases, the HC
  mixers with the faster of its TP1 and TP2 phases:

  | cell | P-L3 | round-6 table | final table |
  |---|---|---|---|
  | out-proj TP2 | 1.39-2.03 | 0.83-0.93 | **0.68-0.85** |
  | HC down | 1.22-1.30 | 0.73-0.77 | **0.69-0.78** |
  | HC up | 1.06-1.34 | 0.52-0.61 | **0.50-0.57** |

  Against P-L3's FlyTape trace (another job) the same cells give 0.63-0.84, 0.68-0.77 and 0.52-0.59. The same-job bracket
  ran 51 cells (FlyTape has no TP1 B128 and no rows-5 configuration); all 51 are below FlyTape's phase, at 0.22-0.85.
  FlyTape's phases moved 0.973-1.078 between its before and after traces (median 1.008).
  - HC down's upper end went from 0.77 to 0.78. That cell's pick did not change (its FlyTape phase did).
- **All 66 cells win.** The cells are 11 projection x TP shapes x rows 5, 8, 16, 32, 64, 128. The frozen table's pick
  WINs 66 of 66 in all four confirm runs (two per rocke node, phase metric):
  - against P-L3's verdict arm of the cell, at the same MXFP4 W4A8 precision: 0.35-0.86 of its phase time;
  - against the fastest of four FlyTape-plan replicas on rocKE: 0.22-0.72.
  The container run, with the image's COMGR, wins 51 of 51 against both.
- **The 10 extra climb rounds paid.** 25 of the 66 picks changed after round 6. Same-job comparisons in the four confirm
  runs, the round-6 pick timed with its own controls:
  - 17 WIN in every run, 6 WIN in some runs and TIE in others, 2 TIE, no LOSS;
  - 0.80-0.98 of the round-6 pick's phase time, largest on out-proj TP2 and QSA in-proj TP2 at 128 rows;
  - the container run: 13 WIN and 6 TIE over the 19 changed cells it covers.
- **Numerics.** The second edition's jobs (climb rounds 7-16, four confirm runs, the container run):
  - 0 failures in 109,736 per-cell checks of timed binaries;
  - 2,632 of 2,632 suite mutant runs caught, 308 of them on `mfma=32` arms;
  - 96 of 96 stale-read twins caught in the barrier-dependency chain test;
  - worst error 0.148 eps; 0 barrier timeouts in any timed launch.
  The first edition's jobs gave 43,630 checks, 996 of 996 mutants and 144 of 144 twins, also with no failure.
- **Compile gate.** Every candidate is compiled with the job's COMGR and scanned for spills and the inline-asm VMEM
  hazard before it is timed. No build anywhere had an asm-hazard hit. The 66 final picks are spill-free in every job,
  container included.
- **Not met: the DESIGN-LB 15.6 comparison "`act=fp4` against aiter `gemm_a4w4` on gdn_in / qsa_in / head
  r64-128".** `act=fp4` is built and numerics-proven. Per launch it loses to aiter's W4A4 GEMM in 12 of 12 cells
  (1.09-1.65). The W4A8 screen finalists lose to it in 10 of 12 (section 7.2, job C2). This is reported as measured; it
  is not part of the numerics / in-situ gate.

## 1. Summary

1. **E1 is a product now.** The pieces:
   - `large_batch/e1.py`, the kernel. It is a fork of `gemm.py`. With the new knobs off and `hz=False` it lowers to
     the same LLVM IR as `gemm.py`, checked for P-L3's 13 arm shapes.
   - `large_batch/e1_plans.py`, the planner: grid fill first, the cost model, the candidate generator and the measured
     plan table `PLANS` for every projection x TP x row bucket, behind `plan(proj, tp, rows)`. `PLANS` holds only
     changes confirmed on both nodes.
   - The one-time host repacks (`pack_weights_tile_sop`, `pack_act_scales_sop`).
   - A producer epilogue (`yq`) that writes the next GEMM's activations in the op_sel layout.

   `gemm.py` is unchanged: sha `39e27905...`, as P-L3 measured it; `insitu.py` subclasses it.
2. **What closed the small GEMMs.** Evidence: WaveScope before / after (section 5) and the climb's ledger (section 6).
   - **The phase barrier.** P-L3's phase barrier was most of a small GEMM's phase. It uses 8 counters, a returning
     arrival and `s_sleep 1` polls. On the traced CU of P-L3's out-proj TP2 arm, 97.8 % of wave time is WAIT and 87.7 %
     of stall cycles sit at the grid barrier.
     `bar=persist` is W5's striped barrier: 64 words on their own lines, one non-returning arrival per CTA, a 64-lane
     ballot poll with `s_nop`, reset by the launch's last CTA. On FlyTape's own plans it is 0.50-0.88 of the old
     barrier's phase time. It is in every final pick.
   - **K split over the waves of a CTA (`kw`).** The groups' accumulators are added in LDS in a fixed order. Out-proj,
     router and in-proj picks fold all or most of K inside one CTA. The cross-CTA last-arriver fold costs partial
     stores, a drain, an atomic and partial reads. FlyTape has no in-CTA form: its `OUT_SPLITS` / `MIX_SPLITS` plans
     fold across CTAs.
   - **op_sel scales for any per-wave chunk count.** P-L3 needed a multiple of 4 chunks per unit. Ranges of 1-2 chunks
     now shift the group dword once per unit; other ranges funnel-shift two group dwords once per unit. This put
     FlyTape's own split counts (1, 2 or 3 chunks per unit) and the `kw` ranges on the fast scale path.
   - **`xsrc=global` with op_sel scales.** Each wave reads its own B operands: no LDS staging and no per-chunk barrier.
   - **Branch-free activation staging (`xjunk`).** On the cluster's COMGR (not on the laptop's), every build that
     predicates its staging store inside the unrolled chunk loop spills: 256 VGPRs plus scratch. That includes P-L3's
     own bucket-16 arms. With `xjunk`, idle threads write an LDS junk row instead.
   - **Work rebalances that only win as pairs.** The every-pair rounds (9-16) found structural pairs in which each part
     loses alone and the pair wins:
     - `kw/2 + tb/2` and `tb/2 + mt*2`: halve the token tiles per unit and double the weight tiles or halve the K
       groups. They won 0.89-0.92 on out-proj TP2 / TP1 and QSA in-proj TP2 at 128 rows. Alone, the parts lose by
       1.10-1.25.
     - `w4 + kw/2` on in-proj at 32 rows.
     WaveScope (section 5): on out-proj TP2 at 128 rows the rebalanced pick's stall cycles at vector-memory loads are
     1/15 of the round-6 pick's (share of stall 32.5 % to 6.4 %), and its total stall is 0.34 of the round-6 pick's.
   - **Interactions found by the two-node climb** (PLAN 13 / 15). 74 changes were confirmed on both nodes over 16 rounds
     (section 6).
3. **What lost.** These stay default-off knobs, with each verdict recorded against the base it was measured on:
   - **`mfma=32`, the 32x32x64 scaled MFMA.** It is built for every op_sel form: short / funnel ranges, `kw`, split-K,
     `fold=spread`, `act=fp4`, global B operands. It is numerics-proven: 8 suite arms, 308 of 308 mutant runs caught. On
     the picks, solo, it costs 1.02-1.55 (median 1.17, 195 measurements over 10 rounds). It is in no WIN combination,
     although rounds 9-16 timed it paired with every other lever.
     - On gfx950 this shape has the MAC rate per CU of 16x16x128 (the spec tables give both shapes the same dense FP8 /
       FP4 throughput). It reads the same activation and weight bytes per wave.
     - It halves a wave's independent accumulator tiles and doubles the token tile, so the picks lose MFMA / load
       overlap and waste half a tile at 16-row buckets.
   - `fold=spread`, the split-K fold distributed over the group: each unit waits for its group and folds 1 / S. It is
     FlyTape's reduce phase merged into the GEMM. It costs 1.19-2.45 of the last-arriver fold, because the group wait and
     the departure round trip cost more than the parallel fold saves at these shapes.
   - P-L3's barrier on every pick (1.09-2.76); splits x 2; token block x 2; 8 waves where the pick has 4.
4. **The library arms** (job C2, launch metric). The LB W4A8 screen finalists are 0.066-0.146 of ATOM's own W4A8 dense
   linear (`native_quant_linear`), WIN in 12 of 12 large cells. aiter's W4A4 `gemm_a4w4` (lower precision) still wins
   per launch on in-proj and head at 64-128 rows, as it did in P-L3 (section 7.2).
5. **The win rule's false positives** (found in this edition). A lever that leaves the pick unchanged (`fg16` on a pick
   that already has `fg=16`, `xpad` on one that already pads) made the lookback time the pick against a second compile
   of itself: 18 such finalists over rounds 1-9. They came out 14 TIE, 3 WIN and 1 LOSS. So a single-node WIN on a
   0.99-ish ratio is not evidence by itself. The two-node rule caught all 3 WINs: the other node reverted each of them,
   and none reached the table. The bench now drops such no-op arms.
   - With about 12 trials a round, the two-node rule passes a few noise changes a round. That matches rounds 13-16
     (1-2 confirmations each, median ratio 0.97-0.99). The climb was stopped there (section 6).

## 2. What was built

| piece | path |
|---|---|
| E1 kernel: `E1Knobs`, `build_e1_kernel`, `e1_params`, `gs_bytes`, `yq_bytes`, host repacks. New knobs: `kw`, `act`, `bar`, `fg`, `kseg`, `yq`, `chain`, `hz`, `xjunk`, `fold=spread`, `mfma=32`, short / funnel op_sel ranges, `xsrc=global` with op_sel. Mutants `kwred`, `seg`, `bardrop`, `barnone`, `noacq`, `nowait`, plus P-L3's `kskip`, `scale`, `fold`, `sopsel` | `dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/qwen38_mono/large_batch/e1.py` |
| planner and measured plan table (`SHAPES`, `model_cost`, `candidates`, `PLANS` (frozen after round 16), `plan`, `row_key`) | `.../large_batch/e1_plans.py` |
| CPU tests (72): IR identity with `gemm.py` for P-L3's 13 arms; every lever and mutant lowers (`mfma=32` included); `hz` pads every asm VMEM block that has an SGPR operand; refusals; the repacks equal P-L3's bench layouts; every `PLANS` entry builds; `plan()` is valid for every projection x TP x rows 1-128 | `.../rocke/library/tests/test_qwen38_lb_e1.py` (P-L3's `test_qwen38_lb_gemm.py`: its 27 tests pass unchanged) |
| bench: numerics suite, chain stale-read test, screen + lookback + confirm. Options `--combo top6 / all2 / all2x` (every pair; and the 6 best measured pairs x every lever) and `--picks-prev` (time an earlier table's pick in the confirm). Also: compile gate, ISA dump, WaveScope launcher and job, climb bookkeeping and driver (`climb.py`, `round.sh`), confirm drivers (`picks_run.sh`, `c3_run.sh`), reports (`picks_report.py`, `climb_report.py`, `l2a_report.py`), job body, staging | `hackathon-prep/qwen38/truemono-lb/wp-l2a/` |
| alola facts | `.claude/skills/alola-cluster/SKILL.md`, rules 197 and 202 |

No existing file changed. Python engine only; kernel names `q38lb_e1_*`; nothing served imports the new modules. W5's
`persist/` is imported (`sync.GridSync`, `poll_counter`, `pad_lds`) and never edited; `insitu.pad_asm_vmem` is imported.

**What WP-L2a did with each FlyTape `tiled.py` piece on the task's list:**

| FlyTape | what WP-L2a did with it |
|---|---|
| `TwoStage` source normalisation (`mix_down` multiplies the stream by `rrms (1 + g)` while staging) | `kseg`, the per-stream re-association (DESIGN-LB S1 / E2): the producer writes the plain residual, `(1 + g)` is folded into W once, and the GEMM scales each K segment's partial by `rrms_s` before any sum. Built and numerics-proven (mutant `seg`). The producer side of the HC re-association and its faithful-order twin belong to WP-L2b |
| last-arriver fold (`_fold`, `_out_fold`, `FOLD_LOADS`) | kept as `fold=last`, the default in every pick. `fg` (partial loads in flight, FlyTape's `FOLD_LOADS`) is a lever. The distributed form `fold=spread` is built and measured, and loses. Where it applies, the in-CTA alternative `kw` wins |
| XCD round-robin slot (`xcd_worker`) | `map=rr` is that slot for our dispatch (CTA c runs on XCD c mod 8, per P-L3's spans). `map=xcd`, the XCD-contiguous `worker` order of FlyTape's GEMM units, is a lever. It loses solo on most bases (median 1.006 over 1,005 measurements), yet it is in 90 winning combinations |
| `UNROLL` / `DEPTH` | our units are fully unrolled. Depth 2 / 3 / 4 / 6 and all chunks in flight (`dall`, as FlyTape's `mix_up` does) are levers |

## 3. Method

- **Cells.** gdn_in, out-proj (gdn_out, which has qsa_out's shape), qsa_in and head at the TP1 and TP2 rank shapes; HC
  down, HC up and router (TP-invariant). Rows 5, 8, 16, 32, 64, 128 (buckets 16-128; live rows is a runtime argument):
  66 cells. Every LB arm is MXFP4 W4A8 (ATOM's numerics). FlyTape's replica arms run its bf16 plan, FlyTape's dense
  precision.
- **Metrics** (P-L3's), with L rotating over at least 4x the MALL of weights (cold weights, hot activations):
  - `phase`: one launch walks L layers with a grid barrier between layers. This is the verdict metric (FlyTape's
    phase semantics).
  - `stream`: the same walk with no barrier (the throughput bound).
  - `launch`: one launch per layer, in a HIP graph.
- **Search** (never a grid).
  - Screen 1 (6 rounds): the planner's structural plans and P-L3's arm.
  - Screen 2, the lookback: every lever on the 3 best bases (losers included), plus pairs of the 6 best levers.
  - Confirm: the best 2 with second-graph and recompiled controls, 24 rounds, judged by the win rule (DESIGN 9.1).
  - Four plan-of-record runs (two per node) gave the first plan table.
- **The two-node lookback climb.** 16 rounds alternating n01-09 and n07-17. Each round runs on every cell's pick:
  - every lever solo, losers included;
  - combinations:
    - rounds 1-8: every pair and triple of the 6 best solo levers;
    - round 9: every pair of every lever;
    - rounds 10-16: every pair, plus the 6 best measured pairs extended by every other lever.
    That is 1,750-1,900 timed combinations per round in rounds 2-8, 7,900 in round 9 and 11,500-11,700 in rounds
    10-16;
  - the previous pick as an arm.
  A change becomes the pick only after it WINs on both nodes. A rejected change is "rejected on base X" and is re-tested
  in every later round (PLAN 13 / 15). The climb stopped after round 16: rounds 13-16 confirmed 2, 2, 2 and 1 changes,
  as many as the two-node rule's false-positive rate predicts (section 1.5).
- **Freeze and confirm.** `PLANS` = the confirmed picks only. The 10 one-node trials pending after round 16 stay in the
  ledger. Then:
  - four `picks` confirm runs, two per node, each also timing the round-6 table's pick with its own controls;
  - one container job: FlyTape traced before and after the LB timing (the same-job bracket).
- **Discipline.**
  - Numerics run on every timed binary, in the same process, before its timing.
  - Arm order and metric order are shuffled every round; every side has controls.
  - Node and GPU bus are recorded per run.
  - Every GPU is health-checked in its own process. A GPU fails the check if it errors or returns a non-finite
    fill. n07-17's bus-e5 GPU never appeared.
  - The asm-hazard pad (`hz`) is on in every build, and the hazard scan is a compile gate.
- **Jobs** (MI355X, q38 venv unless stated):

  | step | jobs |
  |---|---|
  | triage screens | 68236624-68236787 |
  | screens, projections A | 68237128, 68237254 (n01-09); 68237207, 68237264 (n07-17) |
  | screens, projections B | 68237210, 68237343 (n01-09); 68237305, 68237432 (n07-17) |
  | climb rounds 1-6 | 68237318, 68237514, 68237667 (n01-09); 68237403, 68237594, 68237734 (n07-17) |
  | round-6 confirm | 68237806, 68237917 (n01-09); 68237808, 68237933 (n07-17); container 68238017 (C2, n01-09, 2 GPUs) |
  | climb rounds 7-16 | 68238317, 68240316, 68241017, 68241205, 68241299 (n01-09); 68240237, 68240746, 68241126, 68241255, 68241344 (n07-17) |
  | final confirm | 68241387, 68241391 (n01-09); 68241388, 68241405 (n07-17); container 68241430 (C3, n01-09, 2 GPUs) |
  | WaveScope (Markham MI350X) | 68236788, `w7l2a-att2`, `w7l2a-att3`, 68241481 (`w7l2a-att5`) |
  | cluster-COMGR ISA bisection (CPU-only) | 68236659, 68236671 |

  - Round 8's job (68240237) was submitted by a poller that survived a workstation suspend (alola rule 202). Its stages
    were the round-7 state, as the climb required, so it was used.
  - 68240499 (round 10 with 2 GPUs) was cancelled while pending. Another user's array was backfilling n07-17 one GPU at
    a time; rounds 10-16 used 1 GPU behind the health check.

## 4. Numerics

- **Reference.** float64 torch of the dequantized operands: MXFP4 x E8M0 weights, MXFP8 or MXFP4 x E8M0 activations.
  P-L3's metric: `max(|y - ref| - 2^-8 |ref|) / sum|x w| <= 2^-11`. `kseg` references scale each K segment exactly.
- **Suite, in every job that times a lookback round or a confirm.** 83 arms over 7 shapes (248 records a job), covering
  every new lever alone and in combination:
  - kw 2 / 4 / 8 with splits 1-24, short / funnel op_sel ranges, `xsrc=global`, `act=fp4`, `bar=persist`, `fg`,
    `kseg`, `fold=spread`;
  - `mfma=32` with `kw`, splits 3 / 5 / 12, `act=fp4` and `fold=spread`.
  - Checks: 2-layer walks at partial rows (5, 37, 64, 100, 128); 3-run determinism; the grid-synced walk bitwise equal
    to the free one, with no barrier timeout; counters back at zero after every launch; N2, every arm bitwise equal to
    the others with the same (splits, kw, kseg).
  - Mutants per arm: `kskip`, `sopsel`, `fold`, `kwred`, `seg`, `nowait`, all caught (188 per job). `nowait` applies
    only where `fold=spread` takes its spread path (U <= grid).
- **Chain test: the barrier carries a real dependency.**
  - A 2560 x 2560 GEMM walks 4 layers. Layer l > 0 reads layer l - 1's `yq` output: MXFP8 of bf16 Y, in the op_sel
    layout. The yq buffers are refilled with fresh garbage before every launch.
  - Each layer is checked against the float64 reference of the X the kernel actually read. Every `yq` output is
    checked bitwise against the host MXFP8 rule applied to the layer's bf16 Y.
  - Stale-read twins: `bardrop` (one arrival short) and `barnone` (no wait), both with CTA 0 starting every unit
    60 us late. Caught 240 of 240 over both editions; base arms 120 of 120 PASS.
  - `noacq` (no acquire after the barrier) is informational: caught 0 of 120 times. A miss proves nothing (alola rule
    182).
- **Per cell, before timing.** Every candidate, both controls and every replica, on the cell's own data: 153,366 checks
  over both editions, 0 failures, worst 0.148 eps. `act=fp4` arms match the float64 reference to 0.000 eps.
- **Compile gate.** It uses the job's COMGR, `llvm-objdump` and P-L4's hazard scanner. A spill or a hazard drops a
  candidate before it is timed. Reference arms (P-L3's base, FlyTape's replicas) are timed as built, with their gate
  result recorded.

## 5. Where the time went (WaveScope ATT, Markham MI350X, one traced CU; stall cycles as ratios)

| cell | arm | wave states (EXEC / WAIT / STALL) | stall cycles / reference arm's | grid-barrier share of stall | vmem_load share |
|---|---|---|---|---|---|
| out-proj TP2, 8 rows | P-L3's arm (6 splits, P-L3 barrier) | 1.8 / 97.8 / 0.2 % | 1 | 87.7 % | - |
| out-proj TP2, 8 rows | round-6 pick (4 waves, `kw=4`, global B, persist barrier) | 20.2 / 72.6 / 6.4 % | 0.18 | 51.0 % | - |
| HC down, 64 rows | P-L3's arm (20 splits, 2 token blocks) | 3.7 / 94.0 / 1.9 % | 1 | 84.1 % | - |
| HC down, 64 rows | round-6 pick (10 splits, 1 token tile, 4 waves) | 15.0 / 79.8 / 4.7 % | 0.28 | 60.5 % | - |
| out-proj TP2, 128 rows | round-6 pick (`tb=4`, `kw=4`, 8 waves) | 15.5 / 47.0 / 37.1 % | 1 | 20.5 % | 32.5 % |
| out-proj TP2, 128 rows | final pick (4 waves, `mt=2`, `tb=2`, `kw=2`, `xa=2`) | 23.4 / 63.1 / 13.0 % | 0.34 | 50.0 % | 6.4 % |
| QSA in TP2, 128 rows | round-6 pick (`tb=4`, `kw=2`, 8 waves) | 19.1 / 53.3 / 27.3 % | 1 | 20.2 % | 19.4 % |
| QSA in TP2, 128 rows | final pick (4 waves, `mt=2`, `tb=2`) | 24.6 / 62.2 / 12.8 % | 0.38 | 25.2 % | 4.4 % |
| HC down, 5 rows | round-6 pick (20 splits, 4 waves) | 6.2 / 92.2 / 0.8 % | 1 | 87.9 % | 0.5 % |
| HC down, 5 rows | final pick (10 splits, `mt=2`, `kw=2`, global B, nt) | 4.9 / 93.8 / 0.3 % | 0.87 | 98.1 % | 0.3 % |

- **Small GEMMs.** The traced CU waits at the grid barrier for the slowest CTA. That CTA is the split-K fold's last
  arriver: it reads every split's partials alone.
  - HC down at 64 rows: the round-6 combination halves the splits, the token block and the rows per unit, which cuts
    that fold 8x.
  - HC down at 5 rows: halving the splits again and folding half of K inside the CTA (`kw=2`) removes 13 % of the stall.
    The traced CU is still 98 % at the barrier: this cell stays latency-bound.
- **Large rows.** At 128 rows the round-6 picks' stall sits on vector-memory loads, mostly the activation and
  activation-scale loads (`e1.py` lines 825 and 924), next to the grid barrier.
  - The rebalanced picks stage half the token tiles per unit and cover twice the weight rows per wave.
  - Stall cycles at vector-memory loads fall 15x (out-proj) and 12x (QSA in-proj).
  - The grid barrier becomes the larger share of a smaller total.

## 6. The lookback climb (PLAN 13 / 15)

- **Rounds.** 16 rounds over all 66 cells, alternating n01-09 and n07-17. 255 trial changes:
  - **74 confirmed**: a WIN over the previous pick on the second node (42 on n01, 32 on n07).
  - 171 reverted: TIE or LOSS on the second node. Each is recorded as "rejected on base X" and re-tested in every
    later round.
  - 10 pending at the freeze: a WIN on one node, not yet on the other. They stay in the ledger, not in the table.
- **Per round** (rounds 1-7: 37 confirmed, as in the first edition plus round 7's 3):

  | round | combinations | confirmed | reverted | new trials | median confirmed ratio |
  |---|---|---|---|---|---|
  | 8 (n07) | top6 | 0 | 14 | 16 | - |
  | 9 (n01) | all2 | 1 | 15 | 17 | 0.935 |
  | 10 (n07) | all2x | 6 | 11 | 24 | 0.922 |
  | 11 (n01) | all2x | 13 | 11 | 20 | 0.962 |
  | 12 (n07) | all2x | 10 | 10 | 14 | 0.968 |
  | 13 (n01) | all2x | 2 | 12 | 17 | 0.988 |
  | 14 (n07) | all2x | 2 | 15 | 10 | 0.967 |
  | 15 (n01) | all2x | 2 | 8 | 12 | 0.984 |
  | 16 (n07) | all2x | 1 | 11 | 10 | 0.945 |

  Round 8, with combinations of the six best solo levers only, confirmed nothing. Round 9, with every pair, restarted
  the climb.
- **Solo losers win in combinations** (the reason for PLAN 13). WIN finalists whose every lever lost solo on that base:
  - 11 in rounds 1-8 (top-6 combinations: near-1.00 losers);
  - 55 in rounds 9-16, 16 of them below 0.95. Examples: `kw/2 + tb/2` 0.888 (solo 1.10 and 1.20), `tb/2 + mt*2` 0.915
    (1.17, 1.19), `mt*2 + w4` 0.939 (1.21, 1.22).
  `WP-L2a-tables.md` 4.4 lists all 66.
- **Per-lever solo medians on the picks, all 16 rounds** (full ledger in `WP-L2a-tables.md` 4.2):

  | lever | median solo ratio | in WIN combinations |
  |---|---|---|
  | `xpad`, `xjunk`, `bpre`, `fg32`, `fg16`, `xa2` | 0.998-1.003 | 3-51 |
  | `map`, `nt`, depth 3 / 2 | 1.006-1.012 | 28-90 |
  | depth 6, `w4`, no `sched_barrier` fences, no cross-unit prefetch, all chunks in flight | 1.032-1.051 | 13-74 |
  | `w8`, `s/2`, `xsrc` swap | 1.090-1.133 | 0-19 |
  | `mfma=32` | 1.169 | 0 |
  | `mt` x 2 / 2, `kw` x 2 / 2, `tb` / 2 | 1.170-1.260 | 7-37 |
  | `tb` x 2 | 1.292 | 0 |
  | `fold=spread` | 1.695 | 0 |
  | P-L3's barrier | 1.746 | 0 |
  | splits x 2 | 2.210 | 0 |

## 7. Results

### 7.1 Final picks per projection

Four confirm runs, two per node; ranges over rows 5-128 of the per-cell medians; phase metric unless stated. Columns:
- `base`: P-L3's verdict arm of the cell, at the same MXFP4 W4A8 precision.
- `fastest replica`: the fastest of FlyTape's bf16 plan on rocKE with P-L3's barrier or the persist barrier, with or
  without branch-free staging. These ratios mix precision, as P-L3's mx column did.
- `floor`: the fastest of 6 read streamers over the same bytes (default and nt loads), stream metric.
- `FT in-situ`: FlyTape's own traced phase per layer. The same-job column is the container run C3, so its LB arms use the
  image's COMGR; the other column is the venv runs against P-L3's FlyTape trace.
- `round-6`: the first edition's pick, same jobs; changed cells only.

| projection | pick/base phase | pick/base stream | pick/base launch | pick/fastest replica phase | floor/pick (stream) | pick / FT in-situ, same job | pick / FT in-situ, P-L3's trace | pick/round-6 pick |
|---|---|---|---|---|---|---|---|---|
| out-proj TP2 | 0.39-0.42 | 0.34-0.47 | 0.51-0.71 | 0.43-0.50 | 0.17-0.37 | **0.68-0.85** | 0.63-0.84 | 0.80-0.97 |
| out-proj TP1 | 0.44-0.58 | 0.49-0.88 | 0.60-0.93 | 0.44-0.63 | 0.18-0.48 | 0.38-0.57 | 0.38-0.57 | 0.86-0.94 |
| HC down | 0.54-0.60 | 0.41-0.93 | 0.69-0.96 | 0.47-0.68 | 0.05-0.10 | **0.69-0.78** | 0.68-0.77 | 0.86-0.93 |
| HC up | 0.44-0.50 | 0.52-0.69 | 0.57-0.68 | 0.67-0.72 | 0.11-0.38 | **0.50-0.57** | 0.52-0.59 | 0.97 |
| router | 0.36-0.42 | 0.39-0.62 | 0.57-0.79 | 0.32-0.53 | 0.05-0.08 | 0.21-0.25 | 0.21-0.24 | 0.92 |
| gdn_in TP1 | 0.52-0.66 | 0.59-0.88 | 0.74-0.95 | 0.22-0.27 | 0.28-0.71 | 0.23 | 0.22-0.23 | 0.89-0.96 |
| gdn_in TP2 | 0.39-0.55 | 0.56-0.71 | 0.56-0.80 | 0.26-0.29 | 0.23-0.53 | 0.33-0.37 | 0.32-0.38 | 0.83-0.96 |
| qsa_in TP1 | 0.43-0.62 | 0.44-0.84 | 0.66-0.84 | 0.25-0.32 | 0.37-0.82 | 0.25-0.31 | 0.25-0.30 | 0.96 |
| qsa_in TP2 | 0.35-0.44 | 0.38-0.51 | 0.46-0.71 | 0.29-0.35 | 0.29-0.70 | 0.33-0.36 | 0.32-0.36 | 0.83-0.96 |
| head TP1 | 0.63-0.86 | 0.67-0.94 | 0.64-0.94 | 0.22-0.29 | 0.54-1.01 | 0.26-0.33 | 0.26-0.33 | 0.95-0.97 |
| head TP2 | 0.59-0.78 | 0.66-0.93 | 0.66-0.91 | 0.22-0.29 | 0.55-1.03 | 0.26-0.36 | 0.24-0.35 | 0.95-0.97 |

- **Verdicts.** Pick vs base: WIN 66 of 66. Pick vs fastest replica: WIN 66 of 66. Both hold in every run on both nodes
  (phase), and in the container run (51 of 51). Pick vs round-6 pick: section Gate.
- **Launch metric (one launch per layer).** The picks are phase-form picks.
  - Venv runs: their launch ratio to P-L3's arm is 0.46-0.96 in every cell. In the first edition two in-proj cells were
    above 1.
  - Container run (the image's COMGR): in-proj TP1 at 8 and 16 rows give 1.06 and 1.06.
  A launch-form table would pick differently; WP-L3 decides the launch structure.
- **Bytes floor.**
  - The head streams at the floor at 5-8 rows: 1.01-1.03 means the GEMM matches the fastest streamer.
  - The small GEMMs stay latency-bound (P-L3 7): HC down 0.05-0.10 of the floor, router 0.05-0.08.

### 7.2 Library arms (container C2, launch metric, 12 cells: gdn_in / qsa_in / head x TP1 / TP2 x 64 / 128 rows)

| LB arm vs library | ratio | verdicts |
|---|---|---|
| LB W4A8 / ATOM W4A8 (`native_quant_linear`, ATOM's dense path, matched precision) | 0.066-0.146 | WIN 12 |
| LB W4A4 (`act=fp4`) / ATOM W4A8 | 0.060-0.154 | WIN 12 |
| LB W4A8 / aiter `gemm_a4w4` (W4A4: half the activation bytes, lower precision) | 0.94-1.61 | WIN 1 (head TP2 64), TIE 1 (head TP1 64), LOSS 10 |
| LB W4A4 / aiter `gemm_a4w4` (precision matched) | 1.09-1.65 | LOSS 12 |
| LB W4A4 / LB W4A8, same cell | 0.92-1.34 (launch), 0.91-1.28 (phase) | - |

- The LB arms here are the cell's fastest screen finalist by launch time, so they do not depend on the plan table and
  were not re-run for this edition.
- Every library arm passed the float64 check on its own operands before it was timed. The wrong-layer must-fail
  control of the check caught 24 of 24.
- **`act=fp4` does not pay here.** The MFMA is not the binding limit of these phase-form plans, and the fp4 candidates
  only went through screen 1 (3 structural plans, no climb).
- A per-launch comparison does not apply to the LB product form, which runs these GEMMs inside persistent launches.

## 8. What this means for DESIGN-LB 15

- **15.2's GEMM rows move.** In-situ phase over FlyTape's same-job phase, MXFP4 W4A8:

  | row | DESIGN-LB 15.2 (P-L3) | round-6 table | final table |
  |---|---|---|---|
  | out-proj TP2 | 1.39-2.03 | 0.83-0.93 | 0.68-0.85 |
  | out-proj TP1 | 0.78-0.97 | 0.38-0.56 | 0.38-0.57 |
  | HC down | 1.22-1.30 | 0.73-0.77 | 0.69-0.78 |
  | HC up | 1.06-1.34 | 0.52-0.61 | 0.50-0.57 |
  | in-proj TP1 | 0.34-0.43 | 0.22-0.25 | 0.23 |
  | in-proj TP2 | 0.59-0.86 | 0.34-0.44 | 0.33-0.37 |
  | QSA in-proj | 0.48-0.60 (TP1), 0.79-0.95 (TP2) | 0.25-0.31 (TP1), 0.34-0.47 (TP2) | 0.25-0.31 (TP1), 0.33-0.36 (TP2) |
  | router | 0.53-0.60 | 0.21-0.25 | 0.21-0.25 |
  | head | 0.32-0.45 | 0.25-0.36 | 0.26-0.36 |

  The cells whose pick did not change move with FlyTape's phase in the bracket job (for example in-proj TP1 0.22 to
  0.23).
  - **The "at risk" marks of 15.4 for out-proj at TP2 and for the HC down / up GEMMs are cleared in phase form.**
  - 15.3's HC and GDN family ratios should be re-derived from these (the W6 synthesis owns that).
- **Where the gains sit.**
  - The phase-form gains are largest where the grid barrier dominated. The real LB kernel replaces most barriers with
    polled edges (`sync=mix`, WP-L1), so the barrier lever's share moves into WP-L1's per-edge sync.
  - The structural levers (`kw`, short op_sel ranges, global B operands, the work rebalances) win in the stream metric
    too: pick / P-L3 arm is 0.34-0.94 there.
- **DESIGN-LB 15.6's WP-L2a row.** Done, all of it:
  - the planner with grid fill first (`e1_plans.model_cost`, `candidates`);
  - the op_sel layout for producers (`yq`) and the one-time weight-scale repack;
  - `fold=last` as the default (no pick uses another fold);
  - `act=fp4`, measured; `wdtype=bf16` (`w="bf16"`);
  - the 32x32x64 MFMA lever: built, proven and measured; it loses (section 1.3).

## 9. Caveats and open items

- **Precision.** The replica columns compare MXFP4 W4A8 LB arms with FlyTape's bf16 dense plan (3.8x the weight bytes),
  as P-L3's mx column did. The matched-precision comparisons are against P-L3's arm and against ATOM's W4A8 linear.
- **Exchange accounting at TP2.** FlyTape's TP2 out-proj phase also issues its exchange pushes (`_out_fold`). Its
  consume / wait half sits in the next phase (`attn_mix_combine`). The LB out-proj phase contains no exchange.
  - This report uses the accounting of P-L3 and DESIGN-LB 15.2 / 15.3: GEMM phases against GEMM phases, with the
    exchanges budgeted as their own family. WP-C5 F5 measured that family for LB at 0.048-0.061 of FlyTape's TP2 step,
    below FlyTape's own 0.060-0.082.
  - Alternative: charge LB's whole measured in-situ exchange (WP-C5's default `lam`, push plus consume, another job) to
    the LB out-proj. Against FlyTape's `gdn_out` and `qsa_out` phases (job C3) that gives 0.92-1.15 at 8-16 rows and
    1.03-1.78 at 32-128 rows. The first edition gave 0.94-1.17 and 1.16-1.91.
  - That over-charges LB, but it shows where TP2's remaining out-proj cost is: the exchange, which is link-bound at
    64-128 rows (WP-C5 7). Overlapping it with the next layer's units is WP-L5's DESIGN-CCL item.
- **Barrier.** The phase metric's barrier is ours: the picks' phase ratios include `bar=persist`, and FlyTape's phases
  include its own barrier.
- **Noise in the climb.** About 3 in 18 single-node A-vs-A pairs pass the win rule (section 1.5). The two-node rule cuts
  that to a few percent per trial. A few of the 74 confirmed changes are probably noise-level. None of the 25 changed
  cells lost to its round-6 pick in any of the four confirm runs.
- **Compilers.** The venv numbers use the cluster venv's COMGR (torch 2.12, llvm22 flavor). The container job uses
  the ROCm 10 image's COMGR, with its own gate run. The laptop COMGR allocates registers differently (alola rule 197),
  so local compile checks are not authoritative for spills.
- **Open items.**
  - The 10 pending climb trials (one-node WINs).
  - Triples beyond the 6 best measured pairs, and an interaction-aware surrogate over the full lever set (PLAN 13).
  - `fold=spread`, `kseg` and `mfma=32` stay default-off.
  - `act=fp4` through the climb, and against aiter in the persistent form.
  - `part=bf16` (halves the split-K partial bytes, tolerance-gated) for the HC down fold, which is still that cell's
    critical path (98 % of the traced CU's stall at 5 rows).
  - A launch-form plan table (WP-L3).
  - The quick end-to-end tok/s of PLAN 14 does not apply yet: no served path uses the LB GEMM (WP-L6).

## 10. Rerun

From `hackathon-prep/qwen38/truemono-lb/wp-l2a` (Git Bash):

```bash
# one climb round (PLANS <- out/plans_climb.txt, CPU tests, stage, submit, wait, fetch, climb.py round), a = n01-09, b = n07-17
bash round.sh 17 a 7797 all2x
# freeze: write the confirmed picks (climb_state C) as a PLANS literal, then python put_plans.py <it>
# frozen-table confirm runs (numerics, chain, picks + the round-6 pick as 'prev'); stage suffix f16 = the frozen table
bash sync_l2a.sh both f16
bash picks_run.sh rF1a a 7801; bash picks_run.sh rF1b b 7802; bash picks_run.sh rF2a a 7803; bash picks_run.sh rF2b b 7804
bash c3_run.sh C3 7805                                       # container: FlyTape traced before and after, 51 cells
python picks_report.py --runs out/rF1a/run/l2a.json out/rF2a/run/l2a.json out/rF1b/run/l2a.json out/rF2b/run/l2a.json \
  --ft-x <P-L3 FlyTape traces> --ratio F4_ratio.md --abs F4_abs.md
python picks_report.py --runs out/C3/run/l2a.json --ft-job out/C3 --ratio C3_ratio.md --abs C3_abs.md
python climb_report.py --runs out/rL*/run/l2a.json --out climb_tables.md
# WaveScope: cluster_sync.sh the rocKE tree to w7l2a-rk and out/stage_att/w7l2a-h to w7l2a-h, then
#   cluster_job.sh --arch gfx950 --tag <t> --stage w7l2a-rk --cpus 16 --mem 96 --env ATT_TAG=<t> --env "ATT_ITEMS=$(cat out/att_items_f16.txt)" \
#     -- bash -c 'bash $HOME/rocke-sdpa-claude/w7l2a-h/att_job.sh'
```

## Verification (independent verifier, W7 vWP-L2a, 2026-11-01)

**Verdict: confirmed.** The gate (DESIGN-LB 15.6: P-L3's numerics suite with the op_sel mutants, mutants for the new
levers, the stale-read test, the in-situ ratios of 15.2 held) was exercised on MI355X and holds. The verifier re-ran the
same-job FlyTape bracket on the node the builder's bracket did not use; it reproduces the at-risk closure.

**Re-derived from the builder's raw data** (`wp-l2a/out/*/run/l2a.json`, the job logs, `climb_state.json`; independent
scripts, not `picks_report.py`):
- Numerics, rounds 7-16 + rF1a/rF1b/rF2a/rF2b + C3: 109,736 per-cell checks, 0 failures; 2,632 of 2,632 suite mutants
  caught (308 on `mfma=32` arms); 96 of 96 stale-read twins caught, 48 of 48 chain base arms PASS; worst 0.148 eps;
  0 barrier timeouts. All match the report.
- Confirm runs: pick vs base and pick vs fastest replica WIN 66 of 66 in each of the four venv runs (two per node, buses
  05 / 15 on n01-09, 75 / 95 on n07-17) and 51 of 51 in C3.
- Pick vs round-6 pick: 25 changed cells; 17 WIN in all four runs, 6 WIN/TIE, 2 TIE, 0 LOSS. C3: 13 WIN, 6 TIE.
- C3 bracket: out-proj TP2 0.675-0.846, HC down 0.688-0.780, HC up 0.502-0.575, all 51 cells 0.215-0.846; FlyTape
  drift 0.973-1.078 (median 1.008).
- Climb ledger: 255 trials, 74 confirmed (42 on n01-09, 32 on n07-17), 171 reverted, 10 pending. `climb.py` confirms a
  trial only on the node that did not find it.
- `mfma=32` solo, rounds 7-16: 195 measurements, 1.021-1.554, median 1.169; in no WIN finalist.
- The 315 picks timed in rF* and C3 equal `e1_plans.plan()` of the local tree. `e1_plans.py` is byte-equal to
  `out/e1_plans.frozen16.py`. `gemm.py` sha `39e27905...`, as in every job. `e1.py` sha `debca5f9...`, as in every job.
- 99 CPU tests pass locally (`test_qwen38_lb_e1.py` + `test_qwen38_lb_gemm.py`).
- Bench code read:
  - `win_rule`: the median must clear both sides' second-graph and recompiled controls by 0.002, with 75 % of the paired
    rounds below 1.
  - `time_arms` shuffles the arm order and the metric order every round.
  - The per-build gate drops any product build that spills or hits the asm hazard.
  - Numerics run on every timed E1 binary before timing.
  - The lookback drops arms whose kernel name equals the pick's (the no-op fix).

**Re-run: job 68241659** (`w7-vWP-L2a-V1`, container, n07-17, 2 GPUs, buses 65 / 15, both healthy; stages
`w7-vWP-L2a-rk` = the current local rocKE tree, `w7-vWP-L2a-h` = the builder's harness, unchanged; seed 9311). It is C3's
recipe on the other node. It adds the numerics suite and the chain test on the container COMGR; C3 ran picks only.
- Compile gate 299 builds, 0 failing. Suite: 60 of 60 arms PASS, 188 of 188 mutants caught. Chain: 12 of 12 base arms
  PASS, 24 of 24 stale-read twins caught. Per cell: 650 of 650 checks good, worst 0.148 eps, 0 timeouts.
- Pick vs base: WIN 51 of 51 (0.365-0.837). Pick vs fastest replica: WIN 51 of 51 (0.222-0.705). Pick vs round-6
  pick: 14 WIN, 5 TIE, 0 LOSS over the 19 changed cells.
- **Same-job FlyTape bracket on n07-17** (pick phase / FlyTape phase, mean of the before and after traces):

  | cell | builder C3 (n01-09) | verifier V1 (n07-17) | V1, stricter of the two traces |
  |---|---|---|---|
  | out-proj TP2 | 0.68-0.85 | 0.63-0.85 | 0.63-0.88 |
  | HC down | 0.69-0.78 | 0.68-0.78 | 0.69-0.79 |
  | HC up | 0.50-0.57 | 0.50-0.58 | 0.50-0.59 |
  | all 51 cells | 0.22-0.85 | 0.22-0.85 | at most 0.88 |

  No cell is above 1 in either job, even against the faster of FlyTape's two traces.

**Issues found (none reverses the verdict):**
1. **"On both nodes" in the Gate section was same-job on n01-09 only.** C3 was the only bracket job. On n07-17 the
   in-situ ratio rested on P-L3's FlyTape trace from another job. V1 closes this: the closure holds same-job on n07-17.
2. **C3 ran without the numerics suite and the chain test**, so the mutants and twins had not run on the image's
   COMGR. V1 ran both on that COMGR; all were caught.
3. **FlyTape's phases drift more on n07-17.** V1's post / pre trace ratios are 0.81-1.105, against C3's 0.973-1.078.
   The worst is qsa_out TP2 at 128 rows (0.81), so the bracket's spread is wider than section 1 states. All 51 cells
   stay below 1 even against the stricter trace.
4. **The out-proj TP2 accounting favours LB per phase.** Section 9 discloses this. FlyTape's `gdn_out` / `qsa_out` TP2
   phases include its exchange pushes; the LB phase has none. The closure therefore holds only under DESIGN-LB 15's
   per-family accounting. The report's upper bound, which charges LB's whole exchange, gives 0.92-1.78. The fair middle
   number has not been measured: LB's push alone added to the out-proj phase. WP-L3 / WP-L5 should time it in situ.
5. **The range "0.80-0.98 of the round-6 pick's phase time" is the range of per-cell medians.** Single runs reach 0.793
   (out-proj TP2 r128, rF1a) and 0.995 (head TP2 r5, rF2a, TIE).
6. **The `base` arm (P-L3's, timed as `e1ref`) is timed as built even if it spills.** The picks runs do not store its
   gate record. At bucket 16, where P-L3's predicated staging spills on the cluster COMGR, pick/base may be measured
   against a spilling baseline. The gate does not rest on pick/base. The FlyTape comparisons include the branch-free
   (`j`) replicas.
7. **No must-fail mutants run on the picks themselves.** The structural levers in the final picks (`mt`, `tb`, `xa`,
   `w4`, `map`, `nt`) have no dedicated mutant. They are covered by per-cell checks with the same tolerance function
   whose sensitivity the suite mutants prove, and the suite has `mt` / `tb` combinations. This is acceptable, but a
   mutant pass on the 66 picks would close it.

Files: harness `hackathon-prep/qwen38/truemono-lb/vwp-l2a/` (`stage.sh`, `c3n07.sh`, `fetch.sh`, `out/V1/`); log
`hackathon-prep/logs/q38/w7-vWP-L2a-V1-68241659.out`.
