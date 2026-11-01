# W1 review: performance and feasibility skeptic (DESIGN.md)

Reviewer role: try to show the true mono will NOT beat the hybrid (`ALL` of e2ecomp2). Ratios, shares and byte
counts only. Below 1 is faster for time. "hybrid units" = the design's convention (hybrid round or step = 1).

Sources checked against the design's numbers, not taken from it: `RP/GAP_TABLE.md` (sections 2, 3, 6),
`RP/BUBBLES.md`, `RP/E2E_COMPOSED.md`, `RP/ledger/{schedule,layerkernel,head,fused-moe}.md`, the mono sources
(`A/kernels/stages/hc.py`, `out_proj.py`, `layout.py`, `mailbox.py`, `A/draft_runner.py`), the private T1 / T3
absolute files (converted to ratios here), the FlyTape baseline, and the offline K=3 kernel trace of the plain mono
(`logs/q38/results/harness-fix-ro1/ro_k3_roctx`, 86 clean rounds, aggregated per kernel name for this review).

## Verdict

The design's direction (persistent pass/round, mailboxes over grid syncs, probes first) is sound, and two of its
budget lines survive the check (G3, the sync-primitive ratios). The headline does not: **"0.73-0.94 of the hybrid
at K=3 (central 0.82-0.86)" and "0.73-0.95 at K=0" are not supported by the measured data.** After the corrections
below, my re-budget is:

| line | design K=3 | skeptic K=3 | design K=0 | skeptic K=0 | main reason (finding) |
|---|---|---|---|---|---|
| G1 boundaries + ramp | 0.00-0.02 | -0.05 to +0.015 | 0.00-0.02 | -0.05 to +0.015 | B1: the only real-layer measurement lost, and lost more on `best` |
| G2 HC spread | 0.03-0.09 | 0.015-0.05 | 0.05-0.14 | 0.02-0.075 | B2: bandwidth-floor bound; mechanism partly misidentified; A2 fan-in cost |
| G3 non-layer in kernel | 0.044-0.070 | 0.03-0.065 | 0.02-0.04 | 0.015-0.035 | holds up bottom-up (M-ok below); round/pass mode only |
| G4 head | 0.003-0.009 | 0.002-0.006 | 0.001-0.004 | 0.001-0.003 | m1: fold already fused in the hybrid, and double counted with G3 |
| G5 QSA | 0.01-0.03 | 0-0.005 | 0.01-0.03 | 0-0.005 | M2: default-off tolerance levers counted; Q4 trivial at served contexts |
| G6 router window | 0.01-0.025 | 0-0.015 | 0.005-0.015 | 0-0.01 | M8: no mechanism evidence; ledger router knobs did not move it |
| G7 post-poll tails | 0-0.01 | 0 (default) | - | 0 | M2: 8-wave split-K changes the summation order: not bitwise |
| G8 cross-layer overlap | -0.01 to +0.02 | -0.02 to +0.005 | -0.01 to +0.02 | -0.02 to +0.005 | m3: only the toy supports a gain; every real-layer early stream lost |
| T parity tax | -0.03 to 0 | -0.05 to 0 | -0.03 to 0 | -0.05 to 0 | M3 |
| composition | none (sum) | x0.8 on the positive sum | none | x0.8 | M1 |
| **true mono / hybrid** | **0.73-0.94** | **0.87-1.07 (central ~0.94)** | **0.73-0.95** | **0.88-1.08 (central ~0.96)** | |

If P1 fails and the plan falls back to `layers=1` + a tail kernel (the design's own fallback), K=3 becomes about
0.90-1.01 and K=0 about 0.95-1.03: **K=0 is then a likely loss.** The grouped/wide (0.70-0.90) and long-ISL
(0.85-0.98) cells have no measured anchor at all (M7). The plan can still win at K>=1 CONC 1 in `round` mode, but
"beats every cell" needs evidence the design does not have yet. Every number in this table is an estimate from
existing measurements; nothing was run for this review.

---

## Blockers

### B1. The only real-layer cross-layer measurement lost, and the design's fix does not cover the whole loss

**Claim (3.2, G1, G8):** the 2-layer stack lost because of fences (about 1.4 %), the fenced XL hand-off, and run-ahead
contention. The `xl` mailbox plus `xhold` fixes all three, so G1 is 0.00-0.02 and G8 is -0.01 to +0.02.

**Evidence against:**
- `ledger/schedule.md` #13: per layer vs mono 1.024-1.040 on the mono profile, **1.053-1.073 on `numa`, 1.034-1.090
  on `best`**. The loss grows by 2-5 points once NUMA and holds are on. The hybrid serves `best`, so that is the
  profile that counts. The design's three-cause attribution comes from a mono-profile diagnostic (1.023 vs 1.038,
  delta r4s only) and does not explain the extra loss on `best`.
- The no-fence diagnostic (1.023) still loses, even though it also removes one launch boundary per two layers (the
  boundary is 0.019-0.029 of a launch, GAP_TABLE 6). So the in-kernel transition costs about 0.03 per layer beyond
  the fences. The design gives no measured decomposition of that 0.03 into "XL hand-off" and "run-ahead"; both are
  hypotheses.
- **The default `xhold` is undefined in the stages this design reuses.** It waits on "a striped per-XCD *M4 done*
  counter". In the hybrid and in this design, M3+M4+M5 are the fused MoE (`moe_fused`; E2E_COMPOSED NF arm: "m3
  fused, m4 / m5 empty"). M4 is an empty stage, so the counter either fires at the fused body's start or has no
  producer. Even with the FlyDSL M4, "M4 done" is earlier than "layer done": the next layer's B1/Q1 streams
  (22.5 MB at every row count, GAP_TABLE 6) would overlap M5, which is 1.5-3.1x its own floor and on the critical
  path. That is the run-ahead mechanism that lost (#13), and that `dyn` lost to (#14: +10-17 % on `best`). The
  design states that this default "reproduces the graph boundary's no-overlap contention profile". It does not: the
  boundary releases no layer-l+1 stream before every layer-l CTA has exited.
- The holds that make up part of N (r1: best/numa 0.965-0.969) are tuned against a kernel entry (`head_sleep=128`,
  `hold.b1=32` after entry). A persistent pass has no per-layer entry, and the design does not say what these holds
  key on. The fused MoE already showed that FlyDSL-tuned holds can erase a whole lever: NF/N at K=0 was 0.999, not
  0.975 (E2E_COMPOSED).

**Why it is a blocker:** G1 and G8 assume a mechanism that has lost on real layers. G3 and G4 exist only in
`pass`/`round` mode, so they depend on it too. The critical path (P1 -> WP1 -> ... -> WP4) commits about 30 GPU-hours
of ports before P1's answer would change the plan.

**Fix:**
1. P1 runs on the `best` profile (and `numa`), not only mono: delta r1, delta r4s and qsa r4s, 2 and 4 layers, both
   controls. Arms:
   - (a) the old fenced stack (must reproduce 1.03-1.09);
   - (b) no fences (timing only, the 1.023 diagnostic);
   - (c) `xl` mailbox, `xhold=layer` (the next layer's A1/A2 and every stream released only when the CTA has seen
     the `xl` pairs it consumes, i.e. layer-complete);
   - (d) (c) plus A1-only `xpre`;
   - (e) (c) with `xhold=none`.
   Arms (b)-(e) decompose the loss.
2. Run a Coz entry sweep on the stacked build: delays at layer B's stage entries, plus a new delay point at
   "CTA leaves layer A". This measures where run-ahead costs, instead of assuming it.
3. Change the default to `xhold=layer` (it reproduces the boundary). "M4 done" (or a fused-MoE phase counter such as
   "down issued") becomes a non-default knob value.
4. Define a hold anchor for the pass: "the CTA observed layer l's `xl` arrival". Re-derive `head_sleep` and the
   entry holds against it in P1. Do not port the per-launch values.
5. Budget G1 at -0.05 to +0.015 until P1 reports. Write a fallback budget (`layers=1` + tail kernel) next to the main
   one, and make P1's WIN a hard go/no-go for WP4, not a soft "defer".

### B2. G2 (the "biggest new lever", and the only K=0 carrier) is about 2x its physical bound, and its mechanism is partly misidentified

**Claim (1.3, G2):** A1/M1 run on 14/56 CTAs, so they are bound by per-CTA bandwidth (0.11 of peak at 14 CTAs, 0.58
at 80). The spread saves 0.03-0.09 at K=3 and 0.05-0.14 at K=0.

**Evidence against:**
- **A1 is not BW(k)-bound at its own CTA count.** Its incr is 2.64-3.40x its floor computed at that same CTA count
  (GAP_TABLE 6; the design quotes this itself). Spreading lowers the floor, and the floor is only a third of A1's
  time. The bound on what the spread can remove is the floor delta:
  - r1: BW(80)/BW(14) = 5.0 (private BW table), so the floor drops by 0.80 of itself. That is 0.80/2.87 = 0.28 of
    A1's incr, or about 0.044 of a delta r1 launch (A1 = 0.16 of it).
  - r4s: dedupe (1.99 of 7.64 MB) and BW(80)/BW(56) = 1.36 bring the floor to 0.19 of itself. That is 0.31 of A1's
    incr, about 0.031 of a delta r4s launch.
  The rest of A1 (norm of all 4 streams per task, pack, barrier, chain, in-CTA reduce, publish) is per-task latency
  that the spread does not shorten.
- **M1 already hides its weight stream.** `hc.py:531-536`: the polled HC "prefetch[es] everything, then poll[s]".
  M1's wait/span is 0.48-0.67, so its bytes land while it waits on C1. What a spread can still remove is the
  post-poll tail: tail/incr is about 0.40 (r1) and 0.39 (r4s). Even halving it saves only about 0.026 (r1) / 0.018
  (r4s) of a launch.
- **The bound in hybrid units** (served hybrid layer about 0.84 of the mono layer):
  - K=3: (0.031 + 0.018) / 0.84 x 0.822 = **about 0.048**;
  - K=0: (0.044 + 0.026) / 0.84 x 0.894 = **about 0.075**.
  These assume every floor microsecond is critical-path time and that the consumer side is free. The design's top
  ends (0.09, 0.14) are about 1.9x these bounds. Blame work segments say the same: work a1 + work m1 is 0.103 of a
  delta r4s launch and 0.128 of delta r1 (BUBBLES 2), against the 0.158-0.29 incr shares the design uses (incr
  includes waits and the entry ramp).
- **Double count with G1.** GAP_TABLE 3: "a1 ... carries the kernel entry ramp". G1 books "the entry ramp: A1
  weights in flight before the hop", and G2 is sized off A1's incr, which contains that ramp. In a pass, A1 becomes a
  polled stage like M1. Its weights can then be issued during the previous layer's M5 wait (the design's own A1
  `xpre`). Once they are, A1's bandwidth term hides behind the hop just as M1's does, and the spread's
  bandwidth benefit mostly evaporates. G1, G2 and G8 claim the same layer-entry window three times.
- **The consumer side is not free.** Today A2 polls A1's packed low-rank result: 40 + 10 pairs per row (MXFP8 words
  and exponents, `hc.py:671-678`), about 0.7 KB of pairs per row. With 80 units summed "redundantly in A2", every
  one of the 80 A2 CTAs polls 320 columns x 4 partials of f32 pairs per row: **about 14x the poll volume** (10 KB per
  row; 160 KB per A2 CTA at 16 rows). It also redoes the 4-partial sum, `silu(y/4)` and the MXFP8 pack of all 320
  values on its critical path. A2's fan-in also grows from 10 to 80 producers per row. The ledger shows that poll
  volume costs: `pipe` (twice the poll loads) lost 0.2-1.7 %. M2 inherits the same cost from M1.

**Fix:**
1. **Make P3 in situ, in the FlyDSL mono inside `ATOM-truemono`** (edit `layout.py` TASKS/BASE plus the hc.py
   unit map), not a standalone rocKE microbench. Standalone results have mispredicted before: the toy stack gained
   0.75 and the real stack lost 1.04; T5 B1/Q1 standalone was 0.84-0.90 of the mono stage. Gate: bitwise vs mono;
   layer A/B on `best` at r1 and r4s, both controls. This costs about 1 G and answers G2 before any rocKE port.
2. Probe three variants:
   - (i) 80 units with A2-side partial sums (the design);
   - (ii) **20 CTAs, one n-block per CTA, 4 waves on its 4 partials, in-CTA ordered reduce.** This is bitwise, adds
     no fan-in growth and no consumer work, and doubles the CTA count of the down chain;
   - (iii) (ii) with r4 dedupe.
3. Re-budget G2 at 0.015-0.05 (K=3) and 0.02-0.075 (K=0), and remove the entry-ramp share from G1 or G2 (book it once).
4. State the K=0 kill criterion now. If P3 + P1 together project less than 0.03 at K=0, the K=0 cell stays on the
   hybrid and the plan says so (see M9).

---

## Majors

### M1. Additive sum with no composition discount

The design sums G1..G8. The hybrid's own record says levers compose sub-additively:
- first composition: 0.80 of the summed gains at K=3;
- second composition: ALL vs the product of standalone ratios 1.020-1.025 (K=3) and 1.032 (K=0);
- hold lookbacks were sub-additive (delta r4s: m3 + head_sleep + a2 + b1 was worse than m3 + head_sleep);
- the fused MoE lost its whole K=0 gain on FlyDSL-tuned holds.

Several G-lines share a mechanism: G1/G2/G8 (layer entry), and G2/G6 together with the holds (bandwidth contention
at stage entry). **Fix:** compose multiplicatively and apply the measured 0.8 factor to the positive sum. Add an
explicit "hold re-tune" line, since every hold in `sched.BEST` was tuned on FlyDSL bodies. Report a central value,
not only a range.

### M2. Default-off and non-bitwise levers counted in the default budget

- **G5 (0.01-0.03).** Q5-Q7 hop fusion and dense-under-budget are tolerance class and default off (the design says
  so). MFMA Q3 is a long-ISL tolerance knob. The only exact lever, the Q4 chunk spread, does nothing at served
  contexts: Q4 "selects all blocks at L 639" and 512 of 2048 blocks at L 8191 (GAP_TABLE 5), so at L up to about
  2048 there is nothing to rank. The served gsm8k contexts are that short. The design's evidence (hop q4>q5 0.147 of
  the qsa r4s path) is from the L 8192 bench; in-server, q4 is 0.04 of the qsa launch against 0.08 in the bench.
- **G7 (0-0.01).** An "8-wave split-K of the post-poll MFMA chain" changes the f32 summation order. Mono's C1, for
  example, chains `w, w+4, ...` on 4 waves and sums `((w0 + w1) + w2) + w3` (`out_proj.py:3-6`). Moving that to 8
  partials is not bitwise, which contradicts section 5 ("default is bitwise the mono stage").

**Fix:** budget G5 at 0-0.005 and G7 at 0 for the default configuration. Book tolerance levers in a separate
"opt-in, after Tier A / gsm8k / long-context" column. Either reclassify `8w` as tolerance class or redefine it as an
N split.

### M3. Parity tax T is too narrow

The design keeps T at -0.03 to 0 while re-porting about 0.4 of the round (A1, A2, B1, C1, M1, M2, Q1-Q7, P1, P2). The
measured analogues:
- rocKE framework vs mono idioms: 1.05-1.07 at d1 with weights;
- MoE split port: 0.98-1.01, with losing cells;
- T5 GEMV family standalone: B1/Q1 at 0.84-0.90 of the mono in-layer stage's bandwidth fraction (head ledger,
  "Gate ... not at B1 / Q1");
- the rocKE-vs-FlyDSL ping-pong: rk_asm 1.03-1.10 of fly at several distances in mailbox-c4.

0.4 x 1.05 alone is 0.02 of the round, before any hold re-tune or the toolchain's code-object noise (a recompiled
mono moves 0.1-1 %).

**Fix:** T = -0.05 to 0. Gate every WP2 emitter in situ (the T11 rig at the served variant, `best`) at <= 1.00 before
WP3 starts, not at WP3.

### M4. The SGPR budget is already exhausted on a served variant

qsa r4s with `numa`/`best` "reports 68 bytes of private (scratch) memory ... LLVM reserved a frame (SGPR count 104,
at the limit)" (`ledger/schedule.md`). The design adds three things to that:
- a runtime layer loop with loop-carried state (layer index, parity, tags, table pointer);
- per-stage scalar argument loads from a device table;
- `args=sgpr_prefetch`, which holds the next layer's row while the current stage's args are live. The 41 args are
  mostly 64-bit pointers.

That is SGPR spilling (`v_writelane`, which raises VGPR pressure on a body already at 228-246 VGPRs) or scratch.
P6 tests B3 + fused MoE + Q2 + stand-ins, not the variant at the limit.

**Fix:**
- P6 must build the real qsa r4s and delta_ple r4s bodies (FlyDSL port shapes, NUMA on) inside the dynamic loop.
- Make `args=lds` the default.
- Prefetch the next row into LDS, not SGPRs.
- Fail on any `sgpr_spill_count` > 0 or `private_segment_size` > 0.

### M5. Compile time of one inlined whole-round function is unbudgeted

`alwaysinline` + `convergent` puts about 5 layer-kernel bodies, the head and the tail stages into one LLVM function.
One build of that function is serial: CPU fan-out parallelises across builds, not within one, and "per-body caching"
is impossible for inlined bodies. The RSI loop (WP9) needs hundreds of builds, iglp-sensitive ones compiled fresh.
The 9.2x call cost was for a stage-sized callee. A `noinline` call per **layer body** is amortised over a whole layer,
and it would allow separate compilation.

**Fix:**
- P6 reports the compile time of the full K=3 round module (ratio vs one layer kernel) and of a `noinline`-per-layer-
  body variant.
- Time the `noinline` variant's per-layer call cost on hardware (one ratio per layer).
- Decide inline vs call before WP1 fixes the framework shape.

### M6. Pass-mode consumers of the layer input are incomplete

The design routes the layer output to A1/A2 as the `xl` mailbox, and writes plain `h_out` only at the pass end. But
the layer input also feeds:
- C1's residual: "prefetched from `h_in` before any poll" (`out_proj.py:9-10`);
- P2 on the PLE layer (`ple.py:420-438`).

Both then need `xl` polls (or the bulk rule with `inv`) and must keep the parity slot live until layer l+1's C1/P2
finishes. The `xl` payload is bf16 pairs, which doubles the bytes of the 4 x 2560 x rows streams, and NUMA writes
two copies. At 16 rows that is about 1.3 MB of `sc1` stores per layer from M5, plus polls by A1 units, A2, C1 and P2.
None of this is in the budget or in P4 (P4 measures a single-pair all-to-all).

**Fix:** list every consumer of the layer input per kind and count the `xl` bytes per variant. P4 adds a
payload-carrying hop (one row and 16 rows of streams) next to the single-pair all-to-all. Consider plain `h_out` +
bulk-rule acquire for the late consumers (C1, P2), whose data has long since landed.

### M7. The grouped/wide and long-ISL cells have no measured anchor

- **Grouped/wide (0.70-0.90).** There is no round decomposition at CONC 2-4 (GAP_TABLE covers CONC 1 only). The only
  served CONC 4 data put the hybrid levers within noise of plain mono: sched-fx10 k=3 0.977 (spread 0.020); head G6
  neutral within +-2 %. A lower end of 0.70 is better than anything measured at CONC 1 (0.856). At 16 rows the MoE
  union grows toward the independence bound, the layer moves toward its bytes floor, and the hop-latency levers
  (this design's strength) shrink in share.
- **Long ISL.** In-server QSA at 16K-256K was never run (GAP_TABLE 7). The Q3 f64 scores grow with L.

**Fix:** add two Phase 0 probes before WP6/WP7 are budgeted:
- P7: a round decomposition (offline driver) at CONC 2 and 4, K=1 and 3;
- P8: qsa layer stage shares at L 32K / 128K / 256K (the bench already supports `--context`).

Until they report, quote these cells as "unknown".

### M8. G6 (router window) has no supporting mechanism

The fused-MoE ledger shows the window is fixed against everything tried:
- "router window unchanged by any throttle (0.165-0.17 med)";
- `route_gate=pre_poll` lost 1.013;
- `router_knobs=fused` lost in 4 of 6 cells (CF);
- `spec=prev` lost 1.20-1.26.

M3 already polls M2's mixed row with its weights issued, so "overlapped with M2's tail" has little left to overlap.
Spreading the router rows over more CTAs raises the `lg` hop's fan-in, and every routing CTA needs all logits. The
fused body also sits under `hold.m3` in `best`.

**Fix:** budget G6 at 0-0.015. Measure it first with the existing null-bytes build (`w_null`, phase stamps) at a
wider router before it gets a WP.

### M9. The goal says beat the full matrix with rocKE owning every kernel. The routing table allows neither.

`ATOM_Q38_TRUE_MONO_CELLS` hands every non-winning cell back to the hybrid (FlyDSL). With the re-budget, K=0 CONC 1
and the fallback path are near parity or losses, and the wide/long cells are unknown.

**Fix:**
- Name the cells expected to lose (K=0 at least) and the lever that would close each one.
- For K=0, budget `loop`-style multi-step decode per launch: the outside-graph share at K=0 is 0.030 + 0.006 idle of
  a step, larger than at K=3.
- Report the matrix as "wins / parity / loses" per cell instead of one range.

---

## What holds up (checked, not just accepted)

- **G3 (non-layer work in kernel).** Bottom-up from the mono K=3 kernel trace:
  - per round, 59 non-layer, non-head kernels take 0.076 of the mono round (0.089 hybrid units, matching the design's
    0.088);
  - of that, 14 `copyBuffer` 0.014, 3 `_draft_front_kernel` 0.014, `_hc_pre`/`_hc_mix` x4 0.012, and about 38 tiny
    meta / fold / index / fill / sampling kernels 0.036.
  Copies, fills and index selects vanish in kernel. The meta/fold chain becomes a few hops. Front and HC mix are real
  work (front is 7.86e7 B per round) and stay at about half. That gives 0.03-0.065 hybrid units: the design's range
  is plausible. It needs `round` mode, though, so it inherits B1, M4 and M5.
- **Sync primitives.** NUMA-calibrated 160-producer all-to-all is 0.58 of plain and 0.97x FlyTape's idle striped
  sync. The graph boundary gap is 0.24 of the stream gap. The all-to-all is 0.80 (r1) / 0.71 (r4s) of the in-graph
  boundary gap. All of these match the private T3/T1 files.
- **FlyTape ratios** (1.70x / 1.59x) match `flytape-baseline.md`. The conclusion "its in-step design is not the
  lever" is right.
- **The measurement discipline** (section 9) matches the ledgers' hard-won rules.

## Minors

- **m1. G4 double counts the fold.** G3 lists "accept and folds" and G4 "no separate fold launch". The hybrid head
  already runs the fused one-launch fold (head ledger L3/G3, `fsync=counter64`), so the fold is not available to
  either. G4 is about 0.002-0.006 (ramp only).
- **m2. Tags in `round` mode.** The drafter uses layer id 48 with its own step counter (`draft_runner.py:15`). Three
  draft steps plus the verify in one launch need distinct tags per forward. "Tags stay `step * 64 + layer + 1`" with
  one step load per launch would collide across draft steps, and the Tier C "tag crossing 2^25" case arrives 4x
  sooner at K=3. Specify tag = f(round, forward index, layer) and add a gate that forces a collision.
- **m3. G8's positive end is toy-only.** The +0.02 comes from `prefetch=l2_touch` on the toy (d8 -5 %; at d2-d4 it
  LOST 2-7 %). On real layers every extra early stream lost: `b3pf` +0.2-1.0 %, `spec=prev` 1.20-1.26, `dyn` +5-17 %,
  stack run-ahead. Default G8 to <= 0 until P1 arm (d) shows a gain.
- **m4. P4 measures idle-arrival primitives.** FlyTape's striped-sync figure and the all-to-all probe are both idle-arrival.
  Under the pass's skewed arrivals, the striped counter and the `xhold` counter cost device-scope atomics from 160-205
  CTAs per layer. Add a skewed-arrival arm.
- **m5. The competitor can move.** The A1 spread (as a FlyDSL edit) and T16's graph-capturable draft step are
  adoptable by the hybrid in the ATOM copy. If the user's bar is "the best hybrid", the margin is the true mono's
  structure-only levers (G3 in round mode, NUMA on wide), not the G2 port.
- **m6. Integration hazard with ATOM's async scheduling.** ATOM stays a step ahead (`_fill_deferred_decode_ids_kernel`,
  staging H2D). It also allocates KV blocks on the host. Kernel-written next-round meta must not race host writes,
  and block-boundary crossings need pre-reserved blocks. If the host must rewrite meta every round anyway, part of G3
  returns as a host-to-device dependency. Specify the ownership protocol before WP5a.
- **m7. Stamp-ring sampling.** "Every Nth layer" stamps in a pass still bias the layers they do not stamp: the stamped
  build is a different code object (1.013-1.024 of plain). Keep the rule "time only unstamped builds", and do not
  read per-layer timing from sampled stamps either.

## Missing de-risking probes (proposed Phase 0 changes, in priority order)

| probe | replaces / extends | answers | cost |
|---|---|---|---|
| P3' HC spread **in situ** in the FlyDSL mono copy, variants (i)-(iii) of B2 | P3 (standalone) | G2 real value and A2 fan-in cost; K=0 go/no-go | 1 G |
| P1' stacked real layers on `best`/`numa`, arms (a)-(e) + Coz at the layer transition | P1 | G1/G8 sign; the `xhold` default; the hold anchor | 3-4 G |
| P6' full-round module compile time + qsa r4s/delta_ple r4s bodies in the dynamic loop (SGPR, spills), inline vs `noinline` per layer body | P6 | M4, M5 | 40 C, 0.5 G |
| P4' payload-carrying `xl` hop (1 and 16 rows), skewed arrivals | P4 | M6, m4 | 0.5 G |
| P7 round decomposition at CONC 2 / 4, K=1 / 3 | new | anchors the grouped/wide cells | 1 G |
| P8 qsa stage shares at L 32K / 128K / 256K | new | anchors the long-ISL cell | 1 G |
| P9 in-kernel round-tail chain with real work (front GEMV, HC mix, head, accept, meta) vs today's kernel sequence | extends P2 | G3 bottom-up | 1-2 G |

Hard gates to write into section 8: WP1 does not start until P3' and P1' report. If P1' is not at or below the
boundary per layer on `best`, `pass`/`round` are dropped from the critical path and the fallback budget becomes the
plan of record.
