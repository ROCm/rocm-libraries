# DESIGN: a true rocKE monokernel for Qwen3.8-Flash-Next decode (gfx950, TP1, MXFP4)

Status: design, revision 3 **+ D0 outcome (W2, 2026-10-31; `D0.md`): Path B is the plan of record, inline bodies with
`args=const`, HC spread per D0.3, K=0 kill criterion tripped (WP9 mandatory), lever order N8 -> N1 -> N4 -> N3b -> N2;
per-cell budgets re-issued in `D0.md` section 4 (they supersede the centrals of sections 0, 2.2-2.4 and 12.5, which are
kept as history).** Revision 3 (W2, 2026-10-30: re-budgeted for the user decisions of `PLAN.md` "User decisions
(2026-10-30)": accuracy-gated non-bitwise restructuring allowed, Rust direct compiler as a pilot track, RULER and
non-greedy sampling / logprobs in scope; section 12). Revision 2 (W1, after reviews `reviews/w1-perf.md` and
`reviews/w1-complete.md`; resolution of every finding in section 11) is kept as the bitwise-only plan of record and
fallback. No rocKE or ATOM source is edited by this document. Ratios, shares and byte counts only;
absolute times live in `hackathon-prep/private-perf/qwen38-true-mono/`. Below 1 is faster for time, above 1 for tok/s.

Abbreviations:
- `R` = `copy-rocm-lib/dnn-providers/hip-kernel-provider/rocke`, `P` = `R/platform/python/rocke`, `L` = `R/library`,
  `Q` = `L/kernels/gfx950/qwen38_mono`.
- `A` = `dev/ATOM/atom/models/qwen4_exp_mono`, `AT` = `dev/ATOM/atom` (both read-only references; edits go to
  `dev/ATOM-truemono`, created by WP0a, never to `dev/ATOM`).
- `F` = `dev/flytape-main/flytape-main/flytape`.
- `RP` = `hackathon-prep/reports/qwen38-rocke`.
- "mono" = ATOM's FlyDSL per-layer monokernel; "hybrid" = mono + `ATOM_Q38_SCHED=best` + rocKE fused MoE + rocKE
  B3 + rocKE lm_head (the `ALL` arm of `RP/E2E_COMPOSED.md`, run e2ecomp2); "true mono" = this design.
- "hybrid units": the hybrid round (K>=1) or step (K=0) = 1.
- Path A = `pass`/`round` persistent launches. Path B = the fallback: `layers=1` rocKE layer kernels in ATOM's graph
  plus one persistent "tail" launch for the non-layer work.

---

## 0. Recommendation in one page

1. **One code base, a launch-granularity knob.** Every layer kind is ONE traced rocKE body (delta, delta_ple, qsa;
   per row variant), run inside a runtime loop over a device-side forward schedule, as FlyTape traces each run of
   one layer type once (`F/runtime.py:1539-1560`). Granularities: `layers=1` (T11: the per-layer bitwise gate vs
   mono and Path B's layer kernels), `tail` (Path B: non-layer work of a round in one persistent launch), `pass`
   (48 target layers + final HC mix + head + argmax), `round` (pass + accept + k draft forwards). The on-device
   request loop (`loop`) is an optional last step worth at most about 0.01 of a round (2.5).
2. **Path A vs Path B is decided by a hard go/no-go (D0, section 8), not assumed.** The only real-layer cross-layer
   test so far (the 2-layer stack) LOST, more so on `best`. P1 rebuilds it on `best`/`numa` with five arms and a
   Coz sweep at the layer transition. If the in-kernel hand-off is not at or below the graph boundary per layer,
   Path B becomes the plan of record.
3. **Inside a layer: keep the mono's tagged-mailbox dataflow** with NUMA dual copies, holds and the ir poll from day
   one. FlyTape's striped grid sync is about as fast for a full all-to-all at idle arrival (section 4), but every
   CTA must arrive, so it cannot express partial edges (B1->B3, B3 siblings) or late stages. Grid sync is used only
   at the residency ticket, the round start in `loop` mode, and as a knob.
4. **Across layers (Path A): the layer output is a mailbox.** M5's combine epilogue publishes the 4 HC streams and
   their per-tile square sums as tagged NUMA pairs to EVERY reader of the layer input: A1, A2, C1 (residual) and P2
   (PLE layer), as the PLE layer's P2 already feeds A1/A2/C1 (`A/kernels/layout.py:102-106`). Next-layer weight
   streams are held until the CTA has consumed its `xl` pairs (`xhold=layer`, the graph boundary's no-overlap
   profile); earlier release values are knobs that Coz must justify.
5. **The HC down-projection spread is a probed lever, not the headline.** A1 is tail-bound, M1 wait-bound
   (`RP/GAP_TABLE.md` section 6); a spread only removes the bandwidth-floor share. Its bound is about 0.048 hybrid
   units at K=3 and 0.075 at K=0. P3 measures it IN SITU in the FlyDSL mono copy, including a no-fan-in 20-CTA
   variant, before any rocKE port.
6. **Honest budget (section 2), multiplicative with a 0.8 composition factor, every lever bitwise:**

   | cell | Path A, gated (D0 pass + WP2 parity gates + hold re-climb) | Path A ungated | Path B (fallback) |
   |---|---|---|---|
   | K=3 CONC 1 | **0.87-1.00, central about 0.94** | 0.87-1.10, central about 0.98 | 0.91-1.01, central about 0.95 |
   | K=0 CONC 1 | **0.88-1.01, central about 0.95** | 0.88-1.11, central about 0.99 | 0.91-1.01, central about 0.96 |
   | K=1/2 CONC 1, CONC 2-4 grouped/wide, long ISL | unknown until P7, P8 and WP0b anchor them | | |

   **Expected losing or at-risk cells** (named with their closing lever, section 2.4): K=0 CONC 1 and K=0 CONC 2-4
   (multi-step decode per launch, WP9); long ISL above 32K (Q3/Q4 spread, WP7); wide CONC 4 at K=3 (fused MoE
   and union dedupe at 16 rows, WP2d/WP6).

   **Revision 3 (section 12): accuracy-gated ceiling.** With non-bitwise restructuring allowed behind the tolerance
   ladder T0-T5 (12.4), every stage and hand-off gets a lever with a bitwise twin (N1-N11, 12.3); each is booked at
   its expected value (pass probability x value, falling back to its bitwise twin), composed at 0.75. Central values,
   hybrid units (full per-cell table in 12.5):

   | cell | bitwise only (rev 2, Path A gated) | accuracy-gated, Path A | accuracy-gated, Path B |
   |---|---|---|---|
   | K=3 CONC 1 | 0.94 (0.87-1.00) | **0.86 (0.67-1.00)** | 0.87 (0.70-1.01) |
   | K=0 CONC 1 | 0.95 (0.88-1.01) | **0.89 (0.71-1.01)** | 0.90 (0.74-1.01) |
   | wide K=2/3 CONC 2-4 (hybrid at mono parity there; derived, P7) | 0.84-0.86 (0.72-1.00) | **0.80-0.815 (0.62-1.00)** | same as A |

   The narrow cells move from "win by about 0.06" to "win by about 0.11-0.14"; the wide cells, where the hybrid has no
   rocKE MoE / B3 / levers, are the largest margin. The largest expected-value levers: N8 draft vocab-pruned head
   (output-exact, acceptance-gated), N4 QSA hop fusion, N1 HC chain, N3 router, N2 out_proj split-K into its
   producers. Each new WP (WP13-WP20, WP-R1/R2) is ordered by expected value over risk (12.7).
6b. **D0 outcome (2026-10-31, `D0.md`).** The probes moved G2 at K=3 (0.004), G4 and G6 to about zero, put G3 at
   0.018-0.028 (P9, the low end of Path B's band) and found Path A's loop body costing more than its cheaper hop (P1).
   **Path B is the plan of record**; Path A returns only through D0.7's switch test on the rocKE bodies (WP3a). Re-issued
   centrals (Path B): K=3 CONC 1 **0.98 bitwise / 0.90 accuracy-gated (0.75-1.01)**; K=0 CONC 1 **0.97 / 0.91, 0.89 with
   the now-mandatory WP9 (0.74-1.00)**; wide K=2/3 CONC 2-4 0.95-0.96 / 0.91-0.92 (L unmeasured until WP2d). The
   headline targets (K=3 <= 0.875, K=0 <= 0.78) need the optimistic end of N1-N4 plus WP9 and the unbooked Rust / Path A
   upside. The rev-3 Path A centrals of the table above (0.86 / 0.89) are withdrawn.
7. **Definition of done** (the user's goal): every cell of the matrix routed to the true mono AND passing the e2e
   WIN rule (9.1) against a pinned hybrid, with every decode kernel a rocKE kernel. The routing table
   (`ATOM_Q38_TRUE_MONO_CELLS`) is a transitional serving device during the work, not the finish line. Non-greedy
   sampling and logprobs are in the matrix through WP8 (section 7.7).
8. **Order of work (section 8):** WP0a (ATOM copy) and WP0b (baselines for every cell) | rocKE-only probes, then the
   ATOM-copy probes P1/P3/P9, then decision D0, then platform, emitters in parallel (each gated in situ at <= 1.00),
   layer kernels, then pass or tail, then round, then grouped/wide, long ISL, sampler, K=0 closure, accuracy, RSI.
   Revision 3 adds probes P11, P13-P15 and P3x (P11 and P3x report before D0), the accuracy-gated emitter variants
   WP13-WP20 (Group 3b, built beside their bitwise twins and gated in `layers=1` kernels, so Path A and Path B share
   them), the Rust pilot WP-R1 (parallel, never on the critical path) and RULER / non-greedy gates in WP8/WP10 (12.7).

---

## 1. Where the time goes today

### 1.1 Mono round and step (measured; `RP/GAP_TABLE.md` sections 2.1-2.2)

| part (mono K=3 round, device stamps rs1) | share |
|---|---|
| target layer kernels (48) | 0.789 |
| target in-graph launch boundaries (47) | 0.023 |
| last target layer -> first draft layer (lm_head, fold, sampling, draft front + meta, host) | 0.040 |
| draft layer kernels (3) | 0.048 |
| between draft layers (draft head, front, meta, host-bound eager launches) | 0.062 |
| last draft layer -> next target pass (last draft head, verify meta, prologue, graph launch) | 0.038 |

- lm_head over the round: 0.065 (target 0.016 + three draft heads 0.049); draft front 0.014; copies 0.013 (`RP/
  GAP_TABLE.md` 2.1).
- Bottom-up from the offline K=3 kernel trace (reviewer check, `logs/q38/results/harness-fix-ro1/ro_k3_roctx`): 59
  non-layer, non-head kernels per round take 0.076 of the mono round: 14 copies 0.014, 3 draft fronts 0.014, 8 HC
  pre/mix 0.012, about 38 tiny meta/fold/index/fill/sampling kernels 0.036.
- K=0 step: layer kernels incl. boundaries 0.928 (boundaries 0.026 of the r1 pass in the 48-layer stack bench),
  in-graph non-layer 0.036 (head 0.025), outside the graph 0.030, idle 0.006.
- The target pass runs at 2.31x its issued-bytes floor and 3.37x its unique-bytes floor. CTAs poll 0.60-0.75 of
  their time. Hand-off hops are 0.26-0.55 of the layer critical path (`RP/BUBBLES.md` section 2). Host-bound idle is
  0.003 of a K=3 round.

### 1.2 The hybrid on the same scale (derived from e2ecomp2 increments)

The hybrid is 0.856 of the mono K=3 round and 0.868 of the K=0 step (`RP/E2E_COMPOSED.md`). Its levers act on
layer kernels (N 0.930, F 0.943, B 0.988 per round) and on the head (0.987). Its head is 0.78-0.83 of the mono's
(`ledger/head.md`), already with the fused one-launch fold (`fsync=counter64`).

| hybrid part | K=3 CONC 1 round | K=0 CONC 1 step |
|---|---|---|
| layer kernels (target + draft) | 0.822 (target 0.775, draft 0.047) | 0.894 |
| in-graph launch boundaries | 0.027 | 0.028 |
| lm_head (all passes) | 0.063 | 0.025 |
| other non-layer work (prologue, meta, commit_ple, HC final mix, front, sampling, copies, eager-drafter launch overheads, idle) | 0.088 | 0.054 |

The hybrid was measured only at CONC 1, K=3 and K=0. K=1/2, CONC 2-4, long ISL in-server and long-context accuracy
have no hybrid baseline (`RP/E2E_COMPOSED.md`, `RP/GAP_TABLE.md` section 7). The only CONC 4 data put the hybrid
levers within noise of plain mono (sched-fx10 k=3 0.977, spread 0.020; head G6 neutral within 2%). WP0b fixes this.

### 1.3 Inside the layer kernels (mono stage shares of the K=3 round, `RP/GAP_TABLE.md` section 3)

m4 0.171, m5 0.096, a1 0.085, b3 0.076, m1 0.073, c1 0.054, b1 0.048, m3 0.045, q7 0.031, a2 0.029, q5 0.017,
q1 0.015, m2 0.014, q4 0.009, q6 0.007, q3 0.006, p 0.005, q2 0.004. The hybrid has already taken m3-m5 (fused MoE
0.78-0.89 of the mono MoE), b3 (0.963 of the best mono layer) and the hop latency (NUMA, holds). Left on FlyDSL code
in the hybrid: a1, a2, b1, c1, m1, m2, q1-q7, p1, p2, about 0.4 of the round (`research/q38-learnings.md` pitfall 10).

**What A1 and M1 are bound by** (corrected in revision 2):
- A1 has 14 tasks at r1 and 56 at r4 (`A/kernels/layout.py:37-41`). Each streams one 32-column group of `w_down` over
  K=10240: 8 waves run 2 n-blocks x 4 K-partials of 20 k_iters, wave 0 sums the 4 partials in order (`A/kernels/
  stages/hc.py:7-12, 86-89`). A1 wait/span is 0.00 and tail/span 0.80-0.94: it is **tail-bound** (norm of all 4
  streams, pack, chain, in-CTA reduce, publish), and its incr is 2.64-3.40x the bytes floor at its own CTA count.
  A spread lowers that floor (BW(80)/BW(14) = 5.0 at r1; at r4s dedupe 1.99 of 7.64 MB issued plus BW(80)/BW(56)
  = 1.36), but the floor is only about a third of A1's time.
- M1 issues all its weights before polling C1 (`hc.py:531-536`); wait/span 0.48-0.86. Its bytes land during the wait.
  Only the post-poll tail (about 0.40 of M1's incr) can shrink.
- Blame work segments: work a1 + work m1 = 0.103 of a delta r4s launch and 0.128 of delta r1 (`RP/BUBBLES.md` 2).
- A1's incr also carries the kernel entry ramp (GAP_TABLE 3). It is booked once, in G1 (2.1).

### 1.4 What FlyTape teaches (and what it does not)

- Measured on the rocke partition (private baseline): FlyTape's TP1 MXFP4 B1 device step is 1.70x the mono's e2e
  K=0 per-token time, and its B4 step is 1.59x the mono's whole K=3 round. **Its in-step design is not the lever:**
  about 486 grid syncs per step (`research/flytape.md` 2a), about 0.12 of its step at idle sync cost; MXFP4 decoded
  to bf16 for VALU dots (`F/blocks/gemv.py:115-133`); bf16 HC mixers, 6.6 MB per mix_down phase against the mono's
  1.93 MB MXFP8 A1; only 324 of 1024 waves busy in mix_down.
- **What transfers:** whole-run persistence (no launches and no host in the loop; head, sampling, stop on device,
  `F/runtime.py:1707-1739`); one trace per layer type with runtime layer loops (`F/runtime.py:1539-1560`); the
  residency ticket with a timeout (`F/runtime.py:1046-1114`); the 8-stripe-per-XCD ballot grid sync (`F/runtime.py:
  1117-1183`); prefetch between arrive and wait; write-through `sc1` data for device-scope buffers; the expert gap and
  NT policy on expert weights (`F/models/qwen4/weights.py:37-40`, `EXPERT_GAP`); ISA lint for exec-join spills
  (`flytape-main/scripts/lint_isa.py`).
- The A1 spread over many waves is FlyTape's idea; section 1.3 bounds what it is worth on MXFP8 A1.

---

## 2. Quantified budget

### 2.1 Rules

- Hybrid units. Every default-on line is bitwise vs the mono stage; tolerance-class levers (Q5-Q7 hop fusion,
  dense-under-budget, `q3=mfma`, 8-partial split-K) are in a separate opt-in column at 0 in the default.
- **Composition:** levers compose sub-additively in the hybrid's own record (first composition 0.80 of summed gains at
  K=3; ALL vs the product of standalone ratios 1.020-1.025 at K=3 and 1.032 at K=0; hold lookbacks sub-additive; the
  fused MoE's K=0 gain erased by FlyDSL-tuned holds). So: true mono / hybrid = 1 - 0.8 x (sum of positive parts) +
  (sum of negative parts). Lines sharing a mechanism (G1/G8 at layer entry; G2/G6 and holds at stage-entry bandwidth)
  are additionally booked once (the A1 entry ramp is in G1 only).
- A **hold re-tune** line H covers every `sched.BEST` hold having been tuned on FlyDSL bodies at kernel entry. The
  pass has no per-layer entry; holds are re-derived against the `xl`-arrival anchor (3.2).
- Each range has a gated value (after the hard gates that reject a failing port instead of paying for it: D0 for
  G1/G8, the WP2 in-situ <= 1.00 gate for T, the WP3 hold re-climb for H) and an ungated value.
- Sections 2.2-2.4 are the **bitwise-only** budget (revision 2, unchanged). The accuracy-gated budget of revision 3
  (non-bitwise levers N1-N11 on top, each with a bitwise twin as its fallback) is section 12.5.

### 2.2 K=3 CONC 1 (verify r4s + draft r4s/r1/r1)

| id | share attacked | mechanism | ungated | gated | evidence |
|---|---|---|---|---|---|
| G1 | boundaries 0.027 + A1 entry ramp | in-kernel `xl` mailbox replaces the graph boundary (Path A only) | -0.05..+0.015 | 0..+0.015 | the stack lost 1.024-1.040 (mono), 1.053-1.073 (`numa`), 1.034-1.090 (`best`) per layer; the no-fence diagnostic still 1.023 with one boundary fewer per 2 layers; the NUMA-calibrated one-pair all-to-all is 0.80 (r1) / 0.71 (r4s) of the boundary gap but the `xl` hop carries payload (P4) |
| G2 | A1 + M1 bytes-floor share | spread (P3 variant ii: 20 CTAs, one n-block each, in-CTA ordered reduce, bitwise, no fan-in growth; or variant i/iii) | 0.015..0.05 | same | bound about 0.048: (0.031 + 0.018) / 0.84 x 0.822 (1.3) |
| G3 | other non-layer 0.088 | prologue/meta at launch start, accept, final mix, front, folds, copies, fills, the eager drafter's launches become in-kernel stages | 0.03..0.065 | same | bottom-up from the K=3 trace (1.1): copies/fills vanish, meta/fold become a few hops, front and HC mix stay about half |
| G4 | heads 0.063 | head panels issued before the poll; head CTAs start while the last layer drains (the fold is already fused in the hybrid) | 0.002..0.006 | same | ramp only (`ledger/head.md`: 0.87 streaming) |
| G5 | QSA chain | exact levers only: Q4 chunk spread (nothing to rank at served L up to about 2048) | 0..0.005 | same | GAP_TABLE 5: Q4 selects all blocks at L 639; in-server q4 0.04 of the qsa launch |
| G6 | fused-MoE router window | wider router rows; measured first with the null-bytes build (P10) | 0..0.015 | same | the window did not move under any knob tried (`ledger/fused-moe.md`) |
| G7 | B1/C1/Q1 post-poll tails | 8-wave split-K changes the sum order: tolerance class, off | 0 | 0 | `out_proj.py:3-6`: `((w0+w1)+w2)+w3` |
| G8 | cross-layer overlap | earlier next-layer stream release than `xhold=layer` (P1 arm d, Coz) | -0.02..+0.005 | -0.01..+0.005 | every real-layer early stream lost (`b3pf`, `spec=prev`, `dyn`, stack run-ahead); the gain is toy-only |
| H | hold re-tune | holds re-anchored and re-climbed on the rocKE bodies | -0.02..0 | -0.01..0 | NF/N at K=0 0.999 instead of 0.975 |
| T | parity tax on about 0.4 of the round re-ported | rocKE re-implementation at equal structure | -0.05..0 | -0.02..0 | toy d1 1.05-1.07; T5 B1/Q1 standalone 0.84-0.90 of the in-layer bandwidth fraction; MoE split port 0.98-1.01; a recompiled mono moves 0.1-1% |

Path A, gated: positive sum 0.047-0.161, x0.8 = 0.038-0.129; negatives -0.04..0 -> **0.87-1.00, central about 0.94.**
Ungated: **0.87-1.10, central about 0.98.**

Path B (G1 = G8 = 0; the tail launch holds the non-layer work and the draft forwards as `layers=1`-style bodies in one
persistent launch, so G3 shrinks to 0.015-0.045 and G4 to 0-0.003), gated: **0.91-1.01, central about 0.95.**

### 2.3 K=0 CONC 1 (decode r1)

G1 -0.05..+0.015 (gated 0..+0.015); G2 0.02-0.075 (bound 0.075: (0.044 + 0.026) / 0.84 x 0.894); G3 0.015-0.035
(sampling, input prep, prologue, idle); G4 0.001-0.003; G5 0-0.005; G6 0-0.01; G7 0; G8 -0.02..+0.005 (gated
-0.01..+0.005); H -0.02..0 (gated -0.01..0); T -0.05..0 (gated -0.02..0).
- Path A gated: **0.88-1.01, central about 0.95**; ungated 0.88-1.11, central about 0.99.
- Path B gated: **0.91-1.01, central about 0.96.**
- K=0 is the at-risk cell. **Kill criterion (D0):** if P1 + P3 project less than 0.03 hybrid units at K=0, K=0 is
  declared an expected-loss cell and WP9 (multi-step decode per launch, 2.4) becomes mandatory for it.

### 2.4 Other cells: expected outcome and the lever for each

Per cell: "wins / parity / loses / unknown", replaced by P7/P8/WP0b anchors as they report.

| cell | variant(s) (`A/dispatch.py`, `A/groups.py`, `A/kernels/sched.py`) | hybrid there | expected | lever that closes it |
|---|---|---|---|---|
| K=3 CONC 1 | target r4s, draft r4s/r1/r1 | measured | wins (Path A), parity-to-win (Path B) | G2, G3 |
| K=0 CONC 1 | r1 | measured | **at risk**: parity to small win | G2, then WP9 multi-step decode per launch: the outside-graph share at K=0 is 0.030 + 0.006 idle of a mono step (about 0.041 hybrid units), larger than at K=3 |
| K=0 CONC 2-4 | r2..r4 | none | **at risk**, unknown | as K=0 CONC 1, plus HC row dedupe |
| K=1/2 CONC 1 | r2s, r3s (+ r1 drafts) | none | unknown (expect between K=0 and K=3) | G3 shrinks with k |
| K=1 CONC 2 | r4g2 (narrow grouped) | NUMA, B3 r4g2, FlyDSL MoE | unknown (P7) | fused MoE grouped (new, WP2d) |
| K=2/3 CONC 2, K>=1 CONC 3-4 | r6g2, r8g2, r6g3, **r9g3**, r12g3, **r8g4** (K=3 CONC 2 lands here), r12g4, r16g4 (wide; `layout_wide.WIDE_ROWS` 16) | FlyDSL MoE and B3, no NUMA (wide builds drop `numa`/`place`/`dyn`), rocKE head | unknown (P7); **wide CONC 4 at K=3 at risk**: at 16 rows the MoE union grows toward the independence bound and the layer moves toward its bytes floor, shrinking hop-latency levers | NUMA on wide (never run: NUMA-area sizing item in WP2d), fused MoE + union dedupe to 16 rows, HC dedupe (w_down issued 16x today) |
| long ISL decode (8K, 32K, 128K, `longest_context` max) | qsa variants at L | none in-server | unknown (P8); **above 32K at risk** | Q4 chunk spread (exact), Q3 parity per L; `q3=mfma` opt-in |
| accuracy suites | all | gsm8k measured at CONC 1 | must hold | bitwise gates; tolerance only on opt-in levers |

### 2.5 The on-device request loop (`loop`)

What `loop` adds over `round` is host-bound idle (0.003), device launch gaps (0.005) and end-of-round idle (0.002) of a
mono K=3 round: about 0.01. It takes over ATOM's scheduling and needs on-device stop (EOS, max_tokens, stop strings)
and cross-round re-reads of tables the host rewrites. It stays optional (WP12). WP9's multi-step K=0 decode is its
smaller sibling: N decode steps per launch for a stable batch, host-visible tokens per step through a mapped cursor.

### 2.6 What the hybrid could adopt (the competitor can move)

The A1 spread (a FlyDSL edit) and a graph-capturable draft step (T16) are adoptable by the hybrid in the ATOM copy.
The baseline for every A/B is the pinned hybrid of WP0b (the recipe's commits). If the bar becomes "the best hybrid",
the margin is the true mono's structure-only levers: G3 in `round` mode, NUMA and fused MoE on wide, HC dedupe. P3's
in-situ FlyDSL variant is recorded as "hybrid-adoptable" in the ledger, and WP2d's wide functions are not shipped into
the hybrid while it is the comparison baseline (or the baseline is re-measured when they ship).

---

## 3. Launch structure

### 3.1 Options

| option | removes | keeps | verdict |
|---|---|---|---|
| per-layer rocKE layer kernels (`layers=1`) | FlyDSL stages (G2, G5) | 48 boundaries, non-layer launches | required step (per-layer bitwise gate), Path B's body |
| `tail` launch | non-layer launches, eager drafter launches | 48 target boundaries | Path B |
| N-layer stacks | N-1 of N boundaries | non-layer launches | probe only (P1 arms) |
| whole target pass | 47 boundaries, head launch, final mix launch, per-layer step/kernarg waits | prologue/meta launches at K>=1, the eager drafter | Path A at K=0 |
| whole MTP round | everything in the round but the host boundary | host scheduling per round | Path A at K>=1 |
| on-device request loop | per-round host boundary | nothing | optional (2.5) |

All are the same code with a different forward schedule and end-of-launch stage (6.1).

### 3.2 The 2-layer stack loss and what Path A changes (hypotheses until P1)

Measured (`ledger/schedule.md` lever 13; `A/kernels/stack.py:1-27`): per layer 1.024-1.040 vs mono, 1.053-1.073 on
`numa`, 1.034-1.090 on `best`. Without fences (timing only, mono profile, delta r4s) 1.023 vs 1.038. Decomposition is
NOT measured; about 0.03 per layer is unexplained beyond the fences, and the extra loss on `best` is unexplained. The
candidate causes, each mapped to a P1 arm:

| candidate cause | design response | P1 arm that tests it |
|---|---|---|
| fenced XL hand-off (`vmcnt(0)`, barrier, `buffer_wbl2 sc1`, one pair; consumer `buffer_inv sc1`, `stack.py:99, 113, 130`) | tagged `sc1` pairs as their own ready flags (`A/kernels/mailbox.py:1-22`), NUMA dual copies, no `wbl2`/`inv` | (a) vs (b) vs (c) |
| run-ahead contention (CTAs without A1/A2 tasks start layer B's streams, `stack.py:24-26`) | `xhold=layer` default: a CTA issues next-layer weights only after it has consumed the `xl` pairs its first next-layer stage needs, which no CTA can do before every M5 producer of layer l has published | (c) vs (e) |
| holds tuned against kernel entry (`head_sleep=128`, `hold.b1=32`) with no anchor in a pass | hold anchor = "this CTA observed layer l's `xl` arrival"; holds re-derived against it in P1 and re-climbed in WP3 | (c) with re-derived holds; Coz delay point "CTA leaves layer l" |
| missing A1 entry overlap | A1-only `xpre`: A1's own next-layer weights (1.9 MB/layer) issued after the CTA's last own M5 load | (d) |

**Layer-input consumers** (revision 2): A1 (all 4 streams + sums), A2 (sums + its 128 values, `hc.py:19-24`), C1 (the
residual, `out_proj.py:9-10, 87-96`), P2 on delta_ple (`ple.py:415-440`). C1 task t and M5 task t own the same 16
hidden columns (`moe.py:27-31`, `out_proj.py:3`), so the C1 edge is one-to-one: C1 polls one producer's pairs (or is
co-placed). P2 polls its tiles' producers. The parity slot of layer l stays live until layer l+1's C1 and P2 finish;
`validate_stack` checks this. Alternative knob `xl_late=bulk`: C1/P2 read plain `h_out` written with the bulk rule and
one `buffer_inv sc1` (their data landed long before they run); P1 times both.

**`xl` bytes** (bf16 pairs = twice the stream bytes; NUMA writes two copies): per layer 4 x 2560 x rows x 2 B of
payload, doubled by the pair format and again by NUMA: about 0.16 MB at rows 1, 0.66 MB at rows 4, 1.3 MB at rows 16
of `sc1` stores from M5, plus the sums. Against 22.5 MB of B1/Q1 weights per layer this is small in bytes but it is on
the critical path; P4 measures a payload-carrying hop at 1 and 16 rows with skewed arrivals.

### 3.3 Recommended structure, concretely (Path A)

```
launch(round):  ticket -> prologue + verify meta stages (read ATOM's per-step buffers; 7.3)
  for f, l in schedule:                  # f = forward index (0 target, 1..k draft), l = layer slot
      body[kind][variant](f, l)          # mailbox dataflow; xl hand-off to l+1; tags per 6.2
      if last layer of forward f:
          final_mix(f) -> head(f, fused argmax) -> (f == 0 ? accept : draft token)
          -> draft_front(f+1) [registers: accepted count a, positions pos0 + a + j]
  publish outputs into ATOM's buffers (7.6)
```

- K=0: `pass` with head + argmax; the next step's prologue runs at the start of the next launch.
- CONC>1: grouped/wide variants; the target verify is padded to the capture size and padded groups run on null state
  (`A/group_meta.py:12-18`, `A/groups.py:5-9`); draft step 0 is never padded (n x S rows, `A/groups.py:9-13`). The
  round kernel carries both group counts and skips padded groups in accept, head and meta.

---

## 4. Synchronization per hand-off

Measured primitives (ratios; `private-perf/.../mailbox-hop-latency.md`, `research/flytape.md` section 3):
- 160-producer all-to-all, plain mailbox -> NUMA dual copies -> NUMA calibrated: 1 : 0.71 : 0.58.
- NUMA-calibrated all-to-all is about 0.97x FlyTape's 8-stripe idle grid sync (per-worker flags 1.52x its stripes).
- Graph launch gap: 0.24 of the stream launch gap.
- All at idle arrival with one pair per producer; P4 adds skewed arrivals and payload.

| hand-off | default | alternative knob | reason |
|---|---|---|---|
| intra-layer stage edges (A1->A2->B1/B3/C1, C1->M1/M2, M2->M3, Q chain, P1->P2) | tagged pair mailbox, NUMA dual copy, IOD-local poll, `ir` poll | `hop=<edge>:stripe` | partial edges; NUMA is the largest measured lever |
| B1->B3, B3 siblings | mailbox (`xchg=numa2`) | - | partial dependency |
| KV rows, index lists, compressed keys (bulk) | mono bulk rule: plain stores, `vmcnt(0)`, one wave `buffer_wbl2 sc1` + ready pair; consumer `buffer_inv sc1` (`A/kernels/mailbox.py:16-20`) | `bulk=sc1` | bitwise; `sc1` mandatory in `loop` mode |
| M5(l) -> A1/A2/C1/P2(l+1) | `xl` mailbox, NUMA, parity slot (3.2) | `xl=fence` (control), `xl_late=bulk` for C1/P2 | P1 decides |
| next-layer weight streams | `xhold=layer` | `xhold=moe_down_issued`, `xhold=sleep:N`, `xhold=none` | contention; Coz climbs |
| last layer -> final mix -> head | mailbox; head CTAs issue first panels before polling | - | B1's issue-before-poll rule |
| head -> argmax -> accept | `mx_decode_gemv` fused epilogue (last-arriver tagged slots + epoch counter); accept on the last arriver's CTA | - | bitwise vs ATOM's sampler (7.6) |
| accept -> draft front / everyone | one broadcast pair carrying `a` (accepted count) and the bonus token, NUMA copies + one per XCD | - | 256 pollers on one line otherwise |
| **values produced in-launch that the mono reads from tables** (draft positions, cache length `L`, draft slots) | carried in registers derived from the broadcast `a` (pos0 + a + j), or in a tagged mailbox; **never scalar or plain loads of a table written in the same launch** | bulk rule + scalar-cache invalidate (`s_dcache_inv`) if a table must be rewritten | `common.py:79-87` scalar loads, `qsa.py:182-194` plain `pos`; pitfall `research/q38-learnings.md:251` |
| launch start | residency ticket (per-XCD rank, start count, `s_memrealtime` timeout -> error word) | ticket off | non-resident grid times out instead of hanging |
| round start in `loop` mode | striped grid sync + acquire | - | cross-round bulk data (7.5) |

Losing levers stay off: `pause`, `pipe`, `prio`, `place`, `dyn`, last-arriver hops, `reduce_task` (with NUMA),
`step_load=smem`.

---

## 5. Stage emitters

Every emitter is a rocKE inline emitter `emit(b, ctx, p, t)` on `Q/framework.py`'s `StageSpec`.
- Default: **bitwise the mono stage** of the same variant (same MFMA ops and order, same MXFP8 pack rule `P/helpers/
  mx_quant.py`). Narrow references are the FlyDSL narrow stages; **wide references** are `hc_wide.py`,
  `delta_group.py`, `delta_group_wide.py`, `moe_wide.py`, `out_proj_wide.py`, `ple_wide.py`, `qsa_wide.py`,
  `qsa_fast_wide.py`, `common_wide.py` (`A/kernels/stages/`).
- Knobs that change summation order are tolerance class and default off (`8w` split-K included; an N-split variant
  `8w=n` that keeps each column's chain is bitwise and may be default if it wins).
- Variants: `r1..r4`, `r2s..r4s`, `r4g2`, and wide `r6g2, r8g2, r6g3, r9g3, r12g3, r8g4, r12g4, r16g4`.
- **Gate per emitter (WP2):** bitwise in the T11 rig AND in situ (the rig at the served variant, `best`) at <= 1.00 of
  the hybrid's stage under the WIN rule, before WP3 starts.

Day-one levers (each won in situ somewhere): NUMA, ir, DPP, hold (re-anchored), pre (weights before the poll), nt,
dedupe, kp (launch entry only; per-layer args via the table, 6.2).

| emitter | source (bitwise reference) | variants | new knobs |
|---|---|---|---|
| A1/M1 HC norm + down + inject | `hc.py` a1/m1; `hc_wide.py` | all | `spread=` (P3 variants i/ii/iii), `xpre` |
| A2/M2 HC up + mix + MXFP8 pack | `hc.py` a2/m2; `hc_wide.py`; `INV_ROWS` from 12 rows | all | `psum=a2` (only with spread i) |
| GEMV-tile family: B1 (+conv, verify window, b3pre host), C1 (+HC write `w`/`whs`, `xl` residual), Q1, P1 | `delta.py`, `out_proj.py`, `qsa.py` q1, `ple.py`; wide twins | all | `kiter_group`, `8w=n` |
| B3 gated delta rule | `Q/stage_b3.py` (`TUNED_SPECS`, `tuned_numa`) | narrow + r4g2; wide is new | wide variants |
| M3+M4+M5 fused MoE | `Q/moe_fused.py`; reachable today only via `stage_adapt` as self-placed m3/m4/m5 (`Q/stage_adapt.py:30-33, 62`) | rows <= 4 today, `NUMA_HDR` tied to the narrow scratch (`moe_fused.py:68-72, 549-550`); grouped/wide to 16 rows is **new design** (WP2d) | `xl` combine epilogue, `router=wide` (after P10) |
| Q2 norm + RoPE + cache append | `Q/stage_q2.py` | rows <= 4; add 16 | - |
| Q3 index scores, Q4 top-512 | `qsa_fast.py`, `qsa_fast_wide.py` (f64 dot, radix passes, bitonic rank; ties to the lower block, `qsa_fast.py:18-22`) | all; L to the cell's max | `q4_chunks` (exact, planted-tie gate at chunk boundaries), `q3=mfma` (tolerance) |
| Q5 scores, Q6 softmax, Q7 PV | `qsa.py`, `qsa_group.py`, `qsa_wide.py` | all | `fuse=split_kv`/`single_cta` (tolerance) |
| P2 PLE gate / conv / residual | `ple.py`, `ple_group.py`, `ple_wide.py` | all | `xl` input |
| `xl` producer / consumers (WP2g) | new: M5 combine epilogue + A1/A2/C1/P2 poll prologues | all | `xl=fence`, `xl_late=bulk` |
| final HC mix | rows <= 8: bitwise vs `hc_rows` (`AT/model_ops/qwen4_exp/hyperconnection.py`, `HC_FINAL_ROWS_MAX = 8`); rows > 8: ATOM uses the six-op `mix` GEMM path (`:123-140`), whose order a per-row kernel cannot reproduce | rows 1..16 | rows > 8 are **tolerance class** (Tier A drift, acceptance, gsm8k); the round bitwise gate at > 8 rows uses an ATOM-truemono reference with `HC_FINAL_ROWS_MAX` raised to 16 |
| head + fused argmax | `L/kernels/gfx950/mx_decode_gemv.py` (bf16 act, `korder=mono`, `epilogue=fused`, `ctas=243`, rows_max 4/16) | rows 1..16 | `logits=out` (full logits to memory for logprobs and non-greedy, 7.7) |
| accept | ATOM's `[3,B]` verdict, bitwise vs `rejection_sampler` greedy on top of the sampler's argmax (`topk_select(..., tie="low")`, `AT/model_ops/sampler.py:42-50, 137-150`) | K=1..3, groups 1..4, padded groups skipped | - |
| draft front | `A/front.py` (its Triton order) | rows 1..16 | - |
| prologue + meta (launch START) | `A/prologue.py`, `A/step_meta.py:312, 362, 401` (integer/copy ops: bit-identical by construction) | all | - |

---

## 6. Code size, tags, compile time, registers

### 6.1 One body per (kind, variant), runtime forward loop

- `layers="dynamic"` in `Q/framework.py` (today's multi-layer launch is unrolled, `framework.py:556`): one body per
  (kind, variant) in an `scf` loop over a device-side schedule (kind, layer slot, forward index, parity, arg row).
- **Inline vs call is decided by P6 before WP1.** `alwaysinline` + `convergent` stage bodies stay (a stage-sized real
  call cost 9.2x the inlined cycles; a non-convergent call hung a stamped build). The open question is a `noinline`
  call per **layer body**: amortised over a layer, separately compilable and cacheable. P6 times the full K=3 round
  module compile (ratio vs one layer kernel), the widest round (r16g4 target x 3 kinds + draft r16g4 + r4 + heads at 16
  and 4 rows), and the per-layer call cost on hardware.
- Cold once-per-launch stages (prologue, meta, accept) are `noinline` candidates if inlining raises the register max.

### 6.2 Tags, regions and per-layer arguments

**Tag plan** (`A/kernels/mailbox.py:39, 48-51`: `tag = step * LAYER_SLOTS + slot + 1`, `LAYER_SLOTS = 64`; `check_layer`
keeps slot < 63). One round counter `step` is loaded once per launch and advanced once per round.

| instance | slot | regions |
|---|---|---|
| target layers 0..47 | 0..47 | layer scratch, parity = slot & 1 |
| draft forward j (j = 0..k-1, all run layer 48's weights) | 48 + j (48..50) | draft scratch, parity = slot & 1 |
| final mix / head / accept-or-draft-token of forward f (f = 0 target, 1..k drafts) | 52 + f (52..55) | per-f tail regions |
| draft front of forward f | 56 + f (56..59) | per-f front region |
| prologue + meta | 60 | meta region |

- The fused MoE's private `tag ^ 2^31` scheme takes the instance's tag, so it stays unique per instance.
- Today the drafter has its own counter advanced per draft step (`A/draft_runner.py:134, 214-220`, `A/step_meta.py:
  392`). In `round` mode the drafter's distinct slots replace that counter; tag crossing 2^25 then arrives at the
  verify rate, not 4x sooner.
- `validate_stack` runs over the full round schedule (every instance, both parities, k and G); a mutant reusing one tag
  must fail it and must fail the GPU gate.

**Per-layer arguments.** ATOM's 41 `LAYER_ARGS` per layer (`A/kernels/kernel.py:89-131`) go into a read-only device table
(51 rows incl. draft slots, about 17 KB) written by the host before the launch. Scalar loads are legal on it because no
launch writes it. **Default `args=const` (D0.2, from P6): each stage reloads its args from the table with scalar loads;**
`args=lds` (each stage reloads its args from an LDS copy of the current row, the next row prefetched into LDS during the
current layer's last stage) is a knob: it never won on hardware (P6 `Lil/Li` noise, `O1l/O1` 1.006-1.008). `args=sgpr_prefetch` is a knob, never default (qsa r4s with
`numa` already sits at SGPR 104 with a 68 B scratch frame, `ledger/schedule.md`). Every value produced in the launch
(positions, `L`, slots of draft steps) is NOT in the table (section 4).

### 6.3 Register and spill budget

- 512 threads per CU: <= 256 VGPR+AGPR per lane; the mono runs at 228-246. SGPR at the limit on qsa r4s `numa`.
- P6 builds the REAL qsa r4s and delta_ple r4s bodies (port shapes, NUMA on) inside the dynamic loop, plus the widest
  round. Every build fails CI on `.sgpr_spill_count` > 0, `.vgpr_spill_count` > 0, `private_segment_size` > 0, or an
  exec-join spill (port FlyTape's `scripts/lint_isa.py`). `fresh` ids per stage (`A/kernels/kernel.py:222-238`),
  per-stage arg reloads, loop-carried state limited to (schedule index, step).

### 6.4 Compile time and backends

- Builds fan out as parallel alola CPU jobs, job-local comgr cache, iglp-sensitive builds compiled fresh.
- Server start: every whole-round kernel per (K, G) variant compiles at install, never during graph capture (as
  `rocke_head` does); the cost is in each e2e WP.
- The Rust direct code-object compiler (`users/vanantha/rocke/rust-compiler`, a working-tree overlay) is a third
  engine outside the byte-identity gate: bitwise outputs + ISA lint per body. Its analysis note is `research/
  rust-direct.md` (written 2026-10-29). Per the user decision of 2026-10-30 it is a parallel **pilot track** (12.6,
  WP-R1): minimal overlay of the rust branch's own commits, IR-only direct builds, the head GEMV and the toy-chain
  mailbox first, behind a default-off `code_object_backend` knob. The LLVM path stays default.

### 6.5 Repo rules (cpp-parity exemption)

`rocke/AGENTS.md:21-34` (byte identity) and `platform/AGENTS.md:448-473` (no helper without a C++ port path) apply;
the user's rule is Python-only, no C++ work. Therefore:
- grid sync, residency ticket, `xhold` counters and the layer table live in `Q/` (library family, like `mailbox_ir`),
  **not** `P/helpers/`; nothing Python-only enters `rocke.helpers`;
- every true-mono build uses the `Q/README.md` exemption: a distinct name suffix, explicit `backend="python"`, tests that
  assert the backend that ran;
- `layers="dynamic"` and the round stages are new framework paths behind the exemption; the framework's mirrored path
  stays byte-identical (`test_toy_cpp_engine_byte_identical` keeps passing in every WP's gate).

---

## 7. ATOM integration contract (in `dev/ATOM-truemono`, WP0a; never `dev/ATOM`)

### 7.1 Weights and layouts
- Reuse ATOM's bound per-layer args (`A/weights.py`); no repacking. Head: ATOM's packed `Mxfp4LMHead` (`A/head.py`).
- FlyTape's expert gap needs a repack: deferred probe only.

### 7.2 State pools, KV, MTP rollback
- Unchanged contract: GDN snapshots per row in slots `s[r]`, row 0 reading `s[a-1]`; one conv window `[3+k, QKV]` in
  `s[0]`; PLE window `[QKV, 9+k]`; paged QSA caches (block 16, index compression 4) (`A/step_meta.py:24-60`,
  `A/state.py`).
- Inherited refusals (`A/config.py:78-114`): ReplaySSM off (`ATOM_ENABLE_REPLAYSSM=0` on the served line), bf16 KV with
  block 16, f32 mamba state, text only.
- `snapshot=last` + replay records needs ATOM's state contract D2 (`RP/SPEC_DECODE_ROCKE.md` 2.3): later, separate.

### 7.3 Per-step words: one owner each
- ATOM's scheduler owns block tables, slots, `spec_state_indices_tensor`, positions of the verify input, KV block
  allocation and the deferred input ids (`A/step_meta.py:278-280`, `A/runner.py:313-321`). It writes them before the
  launch as today. Blocks for k+1 speculative positions are reserved by ATOM before the launch (as for the eager path).
- The kernel runs **prologue and verify meta as stages at the START of the launch**, reading those buffers, so each word
  has one writer and the result equals ATOM's by construction. The n-gram advance (`A/prologue.py:16-21, 276-291`) runs
  exactly once per launch, in kernel; the host never runs `StepPrologue` for a true-mono launch.
- Draft-step meta (positions, slots, `L`) is computed in kernel from the in-kernel accept, in registers (section 4).
- The kernel's END writes only GPU-produced outputs into ATOM's buffers (7.6). This also removes the race with ATOM's
  async scheduling (`_fill_deferred_decode_ids_kernel`, staged H2D): ATOM reads our outputs exactly as it reads its own.

### 7.4 Prefill, transitions, refusals
- Prefill and mixed steps stay ATOM's (`A/dispatch.py:7-13, 53`); drafts for the first verify come from ATOM's proposer
  over the prefill chunk (`AT/spec_decode/eagle_proposer.py:697-705`). The first true-mono round reads the state pools,
  KV and draft tokens that path wrote; its gate is a Tier G chain that starts from a prefill.
- Transitions round <-> hybrid <-> ATOM eager (CONC change, a prefill joining, a request needing a path the true mono
  refuses): both paths read and write the same pools and per-step buffers; the round counter `step` is shared with the
  hybrid's counters through one device word so tags stay disjoint across paths; Tier G chains cross each transition.
- A true-mono refusal of a width falls back to the hybrid runner **for that width**, logged, instead of `LazyRunner`'s
  "mono off for good" (`A/groups.py:15-17`, `AT/mono/runtime/route.py`).
- Graph capture: padded capture sizes, compile at install; a capture/replay validation item in WP4h.

### 7.5 Coherence rules
- Same-launch data: tagged pairs or the bulk rule. Values the mono reads from tables but this design produces in-launch:
  registers or mailbox (section 4). Data a later launch reads is coherent through the kernel boundary.
- Per-CTA-owned state (B3 `(head, split)`, B1 conv tiles, P2 tiles) stays on a fixed CTA per variant.
- `loop` mode: every cross-round, cross-CTA read uses `sc1`, plus an acquire at the round start; scalar-loaded tables the
  host rewrites across rounds are re-read after `s_dcache_inv`. P5 has negative controls on both paths.

### 7.6 Call-path contract (where the hook sits)
The mono runner never runs the head: it returns mixed hidden states (K=0) or unmixed HC streams (K>=1) (`A/runner.py:
359-367`). The head, sampler, rejection sampler, logprobs and drafting live in `AT/model_engine/model_runner.py`
(`compute_logits` at 3136-3182, `postprocess` 3193-3360: `self.sampler` 3217/3248, `prepare_sampled_ids` 3295,
`send_mtp_status_to_cpu_async` 3307, `propose_draft_token_ids` 3314/3342/3562) and `AT/spec_decode/eagle_proposer.py:
612-830`. In ATOM-truemono:
- `pass` (K=0) replaces the model forward + `compute_logits` + `self.sampler` for greedy rows; it writes
  `sampled_tokens [bs]` into the tensor `postprocess` would have produced.
- `round` (K>=1) replaces the target forward, `compute_logits`, the sampler, the rejection sampler and
  `propose_draft_token_ids`; it writes `sampled_tokens [bs, k+1]`, the verdict `(num_bonus, num_reject, anchors)` in
  the `[3,B]` format, `draft_token_ids [k, bs]` and `next_token_ids`. `postprocess` then runs its unchanged
  bookkeeping: `prepare_sampled_ids`, the deferred-output path and `send_mtp_status_to_cpu_async` are called by ATOM
  as today, on our tensors.
- Requests that need full logits or logprobs use `logits=out` (7.7) until WP8 lands; before WP8 they are routed to the
  hybrid and logged.
- `ATOM_Q38_TRUE_MONO=off|layer|tail|pass|round|loop` (default `off`). Unset, nothing is imported and the build keys are
  unchanged (the hybrid's ISA-identity gate).

### 7.7 Non-greedy sampling and logprobs
- Phase 1 (`logits=out`): the in-kernel head writes the logits rows to memory; ATOM's sampler and rejection sampler run
  on them (bitwise the hybrid's inputs). This keeps the matrix complete at a measured cost.
- Phase 2 (WP8): an on-device sampler stage using ATOM's RNG contract (same seeds/offsets) with a
  distribution-equivalence gate (tolerance class); bitwise where ATOM's op sequence can be reproduced.
- **Decision (user, 2026-10-30): non-greedy sampling and logprobs are IN scope** and in the performance matrix
  (two new cells, 12.5). Phase 2 details, the speculative rejection-sampling residual and their gates: 12.8.

### 7.8 Numerics gates, in order
1. Each stage bitwise vs the mono stage in the T11 rig: the hybrid recipe's gate set (mailboxes on the logical view,
   NUMA copy 1 == copy 0, every written state pool, determinism, 12 launches without zeroing, forced U 10/30/40, tag
   crossing 2^25, a mutant that must differ).
2. Each layer bitwise vs the mono layer inside the `pass` kernel, on the debug build (per-layer `h_out` dump; a
   different binary from the timed one).
3. The pass on the **served binary**: logits and argmax bitwise vs 48 mono launches + head + ATOM's sampler path
   (`tie="low"`), with a planted tie and NaN/inf rows.
4. The round bitwise vs the hybrid round (Tier G, never built before, `RP/GAP_TABLE.md` 7): verified tokens, accepted
   count, draft tokens and every state pool, over multi-step chains with random acceptance, rollbacks straddling KV
   page (16) and index-compression (4) boundaries, padded groups, prefill entry and every transition (7.4). At > 8 rows
   against the `HC_FINAL_ROWS_MAX = 16` reference (section 5).
5. **Race stress** for `pass`/`round`: delay injection (random per-CTA `s_sleep` at stage entries, `xl` and `xhold`
   points) over thousands of replays (the toy LDS race showed in 3 of 3000 replays, only under stress).
6. **Mutants that must fail** (above stage level): parity slot swapped; `xl` tag off by one layer; row 0 reading `s[a]`
   instead of `s[a-1]`; accept count off by one; planted tie at a Q4 chunk boundary; draft position off by one; one
   reused tag; scalar load of an in-launch-written word. WP1 negative controls: an oversize grid must time out; a
   dropped arrival must not release the grid sync.
7. Tolerance-class knobs: Tier A drift budget, then gsm8k, acceptance length and the long-context suite (9.3). Off
   until all pass. Revision 3 replaces this line by the tolerance ladder T0-T5 of 12.4 (per-stage tolerance,
   layer drift, teacher-forced logit drift calibrated by P11, acceptance, GSM8K / HumanEval / RULER, re-gate of the
   composed config). Gates 1-6 still apply to every lever's bitwise twin, and gates 5-6 (race stress, mutants) apply
   to the tolerance variants unchanged: non-bitwise never means non-deterministic (12.1).

---

## 8. Work packages (final, dependency-ordered)

### 8.1 Cluster rules for every package

- Login `ctr2-alola-login-01` (login-03 DNS is dead). Stages packed by `tools/q38/q38_sync.sh`; jobs through
  `tools/q38/q38_job.sh` on the rocke partition, `--mode container` for anything running ATOM or FlyDSL.
- The rocKE stage comes from copy-rocm-lib (dev/rocm-libraries lacks this work), with the CRLF rewrite (`tools/crlf.py`).
- `q38_job.sh` mounts the first `atom*` stage at `/atom`: a hybrid-vs-copy job stages the copy as `atom-truemono` and
  the pinned hybrid under a non-`atom` name (e.g. `hb-atom`) with `Q38_ATOM_DIR` + `PYTHONPATH` (`qwen38/README.md`).
- Heavy I/O in node-local `/tmp`; at most 6 of our GPUs busy on the rocke partition, `squeue` checked first, never
  `--exclusive`. **Timed A/B and e2e jobs** run with at most one busy neighbour of ours on the node (the only measured
  neutral case) until WP0b validates more; builds and numerics take the other GPUs.
- CPU builds fan out as alola CPU jobs. Every new cluster fact goes into the alola-cluster skill immediately.

### 8.2 Package list

Cost: G = GPU-hours, C = CPU core-hours; e2e costs include server-start compiles and A-vs-A arms. `||` = may run in
parallel within its group.

**Group 0 (no dependencies; all parallel)**

| WP | title | gate | cost |
|---|---|---|---|
| WP0a | create `dev/ATOM-truemono` (copy of the pinned hybrid ATOM), `ATOM_Q38_TRUE_MONO` skeleton, `layer` hook, stage naming `atom-truemono` / `hb-atom` | ATOM CPU tests pass; hook unset = ISA-identical build keys | 0 G, 4 C |
| WP0b | baselines: pinned hybrid ALL and mono on every cell (K 0..3 x CONC 1..4; long ISL decode-only at 8K/32K/128K/max per CONC with `max_model_len` pinned per CONC; gsm8k, acceptance, long-context suite), A-vs-A spread per cell; multi-neighbour timing hygiene check. Rev 3: the matrix of `baseline/MATRIX.md` covers the speed cells and GSM8K/HumanEval at K=3 CONC 4; still to add: RULER 8K/32K/128K/max A-vs-A (12.8), the two non-greedy cells, K=0/K=1 accuracy | each cell's A-vs-A spread recorded; private absolutes | 24 G |
| P2 | head in kernel (rocKE): producer stage + head panels before poll + fused argmax, rows 1/4/8/16 (rows-16 bimodality) | argmax/logits bitwise vs `mx_decode_gemv` + ATOM sampler tie rule | 1 G, 4 C |
| P4 | sync primitives: striped grid sync, ticket, per-XCD counter vs NUMA all-to-all at 160/256 producers; idle AND skewed arrivals; payload-carrying `xl` hop at 1 and 16 rows | negative controls (oversize grid times out, dropped arrival blocks) | 1 G |
| P5 | coherence: cross-XCD rewrite-after-read of a KV row in one persistent launch; scalar-path read of an in-launch-written word | both negative controls fail, positives pass | 0.5 G |
| P6 | code size/compile: real qsa r4s + delta_ple r4s bodies (NUMA on) and the widest round in the dynamic loop; `args=lds`; inline vs `noinline` per layer body; full round compile time; per-layer call cost | no spill/scratch; decision recorded | 0.5 G, 60 C |
| P7 | round decomposition (offline driver) at CONC 2 and 4, K=1 and 3, mono and hybrid | anchors the grouped/wide budget | 1 G |
| P8 | qsa stage shares at L 32K / 128K / max (bench `--context`) | anchors the long-ISL budget | 1 G |
| P10 | router window at a wider router with the null-bytes build (`w_null`, phase stamps) | G6 sign; WP only if it moves | 0.5 G |
| P11 (rev 3, **before D0**) | accuracy-gate calibration: teacher-forced logit-drift harness (48 layers + head on real text, golden fp32 chain as anchor), the pinned hybrid's own drift and A-vs-A, planted-noise sensitivity curve per stage (which drift moves GSM8K / RULER / acceptance), drift budget split per N lever | thresholds T2 recorded; a planted 2x drift must fail T2, a 1-ulp perturbation must pass | 1.5 G |
| P13 (rev 3) | QSA under budget: is Q4's under-budget list rank-ordered (`qsa_fast.py:19-23` "takes them all ... ranking them") so a position-order dense path is tolerance class, or order-free in Q5-Q7 (then bitwise); in-situ value of `q4=skip_under_budget` in the FlyDSL copy at served L | decides N4's class and share | 0.5 G |
| P14 (rev 3) | **done by the quick track (`quick/README.md`: 96K subset, coverage, served audit, e2e round 0.970-0.975).** Draft vocab pruning, offline: acceptance length vs draft-head vocab subset size (8K..64K rows by corpus frequency) from dumped draft hidden states, GSM8K + HumanEval + MT-Bench prompts, K=1..3 | N8 subset size; acceptance within the A-vs-A spread | 0.5 G, 8 C |
| P15 (rev 3) | router restructure inside the hybrid's rocKE fused MoE (`moe_fused` router knobs, hybrid function path): MFMA K-split router over all CTAs + redundant per-CTA top-10 vs the mono M3 code, with the null-bytes build; router-flip count vs golden | N3 share and flip rate | 0.5 G, 10 C |

**Group 1 (needs WP0a; parallel)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| P1 | real-layer stack on `best` and `numa`, delta r1, delta r4s, qsa r4s, 2 and 4 layers, arms (a) fenced (must reproduce 1.03-1.09 on `best`), (b) no fences, (c) `xl` + `xhold=layer` + re-derived holds, (d) (c) + A1 `xpre`, (e) (c) + `xhold=none`; `xl` to A1/A2/C1/P2 incl. `xl_late=bulk`; Coz sweep incl. "CTA leaves layer l" | WP0a | bitwise vs single launches; WIN rule per layer vs graph-launched single layers on `best` | 4 G |
| P3 | HC spread IN SITU in the FlyDSL mono copy (`layout.py` TASKS/BASE + hc.py unit map): (i) 80 units + A2-side sums (incl. A2/M2 poll volume), (ii) 20 CTAs one n-block each, in-CTA ordered reduce, (iii) (ii) + r4 dedupe; r1 and r4s on `best` | WP0a | bitwise vs mono; layer A/B both controls; K=0 kill criterion (2.3) | 1.5 G |
| P9 | in-kernel round-tail chain with real work (front GEMV, HC mix, head, accept, meta) vs today's kernel sequence | WP0a, P2 | bitwise vs ATOM ops; G3 bottom-up | 1.5 G |
| P3x (rev 3, **before D0**) | P3 extended with tolerance variants in the FlyDSL copy: (iv) free-order K-split A1/M1 over 80-160 CTAs, f32 partials, fixed-order A2/M2-side sums of <= 40 partials; (v) (iv) + norm-after-GEMV re-association (per-stream `rrms` applied to the 4 per-stream partial dots); (vi) partial-down in the M5 / C1 epilogue (A1/M1 become 20-CTA reducers) | WP0a, P11 | T0-T2 (12.4) vs golden and vs the hybrid's own drift; layer A/B both controls; K=0 kill criterion re-run with the higher ceiling | 1 G |

**D0 decision (after P1, P3, P4, P6, P9; no cost):** Path A or Path B; inline vs `noinline` per layer body; spread
variant; K=0 kill criterion; budget table re-issued with measured G1/G2/G3. WP1 does not start before D0.
Revision 3 adds to D0: P11's thresholds (T2) and per-lever drift budget; P3x's tolerance variant for N1 (or none);
the K=0 kill criterion evaluated on the accuracy-gated ceiling (WP9 mandatory if P3 + P3x + P1 project less than
0.04 hybrid units at K=0); and the order of WP13-WP20 re-ranked by measured value (P13-P15 may report after D0;
their WPs wait for them, nothing else does).

**D0 outcome (W2, 2026-10-31; full record, per-cell budget and wave table in `D0.md`).** Path B (D0.1); inline +
`args=const` (D0.2); spread per D0.3; K=0 kill criterion tripped, WP9 mandatory (D0.4); lever order N8, N1, N4, N3b,
N2, then N6, N5, N7a (D0.5); Rust direct committed by PLAN decision 6, entering as per-family default backend where it
won on both nodes (D0.6, WP-R3); Path A switch test WP3a (D0.7). P11 and P3x did not report before D0 (deviation from
12.7); they are the first accuracy launches. Work packages now unblocked, in launch order:

| wave | parallel group | packages | depends |
|---|---|---|---|
| 1 | 1a (now) | WP1, WP4h, P11, P13, P15 | D0 / WP0a / none |
| 1 | 1b (as GPUs free) | WP0b-acc (RULER, HumanEval A-vs-A, non-greedy cells; with the quick-stack RULER 8K ablation), WP-R3 (Rust direct backend into the working tree, default `comgr`) | none |
| 2 | 2a | P3x | P11 |
| 2 | 2b (<= 5 at a time) | WP2a, WP2b, WP2c, WP2e, WP2f; WP2d at the first free slot | WP1 (+ P3 / P9 / P7) |
| 3 | 3a | WP3, then WP3a (Path A switch test: P1 loop rig on the rocKE bodies) | WP2a/b/c/f, WP2d narrow |
| 3 | 3b (in order) | WP13 (N8), WP15 (N1), WP14 (N4), WP16 (N3b), WP17 (N2), then WP18, WP19, WP20 | 12.7 deps + P11 / P3x / P13 / P15 |
| 4 | - | WP4B -> WP5a || WP5b -> WP5c -> WP10a | WP3, WP2e, WP4h |
| 5 | 5a | WP6, WP7, WP8, WP9 (mandatory) | WP5c |
| 6 | - | WP10; WP11 continuous; WP12 optional | |

Done or superseded: P14 (quick track), WP-R1 / WP-R2 (pilot done; superseded by PLAN decision 6 and WP-R3).
Conditional: WP2g (full), WP4 (Path A) only if WP3a passes.

**Group 2 (needs D0)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP1 | **UNBLOCKED by D0 (wave 1).** Scope additions from D0: inline + P6 R1-R5, D0.8 hand-off / sync defaults, per-CTA arrivals, poll timeout + error word, `tagged_poll` reload fix as a new op variant, static in-launch-KV acquire check (P5), scratch / hand-off overlap rule (P8), WP0a's ISA gate, per-op Rust lever scoping, per-edge protocol knobs (`XCD-SCHEDULING.md`), `search/INVENTORY.md` M1-M12. Platform in `Q/`: grid sync, ticket, `xhold` counters, layer table (`args=const` default, `args=lds` knob), `layers="dynamic"` + parity + `validate_stack` over the full round tag plan, ISA lint, delay-injection stress harness, mutant harness, cpp-parity exemption naming (`research/rust-direct.md` exists; the Rust pilot is WP-R1) | D0 | CPU tests; GPU microbench equal to P4; negative controls; byte-identity test unchanged | 1.5 G, 40 C |

**Group 3 (needs WP1; parallel, at most 5 at a time; each gated bitwise AND in situ <= 1.00 on `best`)**

| WP | title | depends | cost |
|---|---|---|---|
| WP2a | HC A1/M1/A2/M2, narrow + wide, spread per D0.3 (A1 `spread.a1=10` rows 1-3; A1 (ii) rows 4 on qsa only; M1 none except (ii) at qsa r4s; no dedupe), `xpre` | WP1, P3 | 3 G, 40 C |
| WP2b | GEMV family B1/C1/Q1/P1, narrow + wide, `8w=n` | WP1 | 3 G, 40 C |
| WP2c | QSA Q3/Q4/Q5-Q7, grouped + wide, `q4_chunks` + tie gate (+ tie-margin mutant, P8); L 639/8191/16383/max-K; parity timed at 32K/128K/max (P8) | WP1 | 5 G, 60 C |
| WP2d | fused MoE grouped/wide to 16 rows (new design, union dedupe, NUMA-area sizing and calibration for wide scratch and parity slots), B3 wide, Q2 to 16 rows | WP1, P7 | 8 G, 80 C |
| WP2e | (D0: start from `p9_tail.best_spec()`; first task a K=0 tail trace) round stages: final mix (<= 8 bitwise, > 8 tolerance + raised-reference), head adapter (+ `logits=out`), accept (ATOM sampler tie rule, padded groups), front, prologue/meta at launch start | WP1, P9 | 2 G, 20 C |
| WP2f | P2 PLE narrow + wide | WP1 | 1 G, 10 C |
| WP2g | **conditional (D0.7): only a minimal `xl` producer / consumer for WP3a; the full package only if WP3a meets the Path A switch rule.** `xl` producer (M5 combine epilogue) and consumers (A1/A2/C1/P2), `xl_late=pairs` default (`bulk` loses, P1) | WP1, P1 | 1.5 G, 15 C |
| WP4h | **UNBLOCKED by D0 (wave 1).** ATOM call-path contract in ATOM-truemono (7.6): `postprocess` hooks, outputs, deferred-output path, prefill entry, transitions, per-width refusal fallback, graph capture/replay validation, routing table | WP0a, D0 (host side; runs alongside WP2) | 2 G, 10 C |

**Group 3b (revision 3; accuracy-gated emitter variants WP13-WP20, and the Rust pilot WP-R1/R2): section 12.7.**
Each tolerance variant is a knob on the WP2 emitter it extends, built beside its bitwise twin, gated T0-T2 and then
in situ (`layers=1`, WIN rule vs the bitwise twin) before it may enter WP3/WP4/WP5c builds; T3-T5 run per cell in
WP10 (with a WP10a checkpoint after the first three levers land).

**Group 4**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP3 | layer kernels `layers=1`, every narrow variant, ATOM ABI, ATOM-truemono `layer` mode; holds re-anchored and re-climbed with Coz and the combination lookback (D0.9: no `sched.BEST` hold inherited unmeasured) | WP2a, b, c, f; WP2d narrow part | bitwise layer (recipe's `sched-*`/`moe-*`/`b3-*` set); in-situ WIN at r1, r4s, qsa r1/r4s, no LOSS | 8 G |

**Group 5 (one of the two, per D0)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP4 (Path A) | **conditional on WP3a (D0.7).** `pass` kernel: dynamic loop, `xl`, `xhold`, final mix, head + argmax; K=0 serving | WP3, WP2e, WP2g, WP4h | gates 2, 3, 5, 6 (7.8); pass vs hybrid pass (cold); e2e K=0 CONC 1 vs pinned hybrid | 10 G |
| WP4B (Path B) | **plan of record (D0.1).** `tail` kernel: final mix, head, accept, draft forwards, front; target layers stay graph-launched `layers=1` | WP3, WP2e, WP4h | same gates; e2e K=0/K=3 CONC 1 | 7 G |

**Group 6 (parallel a || b, then c)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP5a | in-kernel accept + draft meta in registers + launch-start prologue/meta ownership | WP4 or WP4B | Tier G chains incl. prefill entry and transitions | 2 G |
| WP5b | draft forward chain in kernel (slots 48+j, per-f tail regions): front -> draft QSA body -> mix -> head | WP4 or WP4B | draft tokens bitwise vs the hybrid drafter | 3 G |
| WP5c | `round` kernel (Path A) or full `tail` round (Path B) + proposer hook | WP5a, WP5b | gate 4 (Tier G); e2e K=1/2/3 CONC 1 vs pinned hybrid | 10 G |

**Group 7 (needs WP5c; parallel, within the 6-GPU cap)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP6 | grouped/wide: every variant of 2.4 incl. r9g3 and padding, K=0 r2..r4 | WP5c, WP2d, P7 | Tier C grouped + Tier G padded; e2e CONC 2-4 per K | 16 G |
| WP7 | long ISL: Q chain at 8K/32K/128K/max; `q4_chunks`; `q3=mfma` opt-in | WP5c, WP2c, P8 | bitwise at default; decode-only metrics; long-context suite for knobs | 10 G |
| WP8 | non-greedy sampling and logprobs (in scope, user 2026-10-30): `logits=out` serving, then on-device sampler, fused logsumexp / top-N logprobs, speculative rejection-sampling residual (12.8) | WP5c | phase 1 bitwise inputs to ATOM's sampler; phase 2 distribution-equivalence + logprob error bound + non-greedy acceptance and suites | 6 G |
| WP9 | **mandatory (D0.4: kill criterion tripped).** K=0 closure: multi-step decode per launch (N steps, mapped cursor, on-device stop for those N); HC row dedupe at r2..r4 is an off knob only (P3: dedupe loses 1.02-1.14) | WP4 or WP4B, D0 kill result | Tier G K=0 chains; e2e K=0 CONC 1-4 | 6 G |

**Group 8**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP10 | accuracy on every cell's final config: the T0-T5 ladder (12.4): GSM8K, HumanEval, RULER 8K/32K/128K/max, acceptance length, non-greedy cells; WP10a checkpoint after WP13-WP15 | WP6, WP7, WP8, WP9, WP13-WP20 | within the cell's A-vs-A spread | 22 G |
| WP11 | RSI (9.2) on each kind, pass/tail and round | WP3 onward, continuous | bitwise per lever; WIN rule | ongoing, <= 6 GPUs |
| WP12 | `loop` (optional): on-device request loop, `sc1` bulk rule, round-start grid sync, on-device stop | WP5c, P5 | P5 controls; Tier G; e2e vs `round` | 6 G |

**Critical path:** WP0a -> P1 -> D0 -> WP1 -> WP2a/b/c/f (parallel) -> WP3 -> WP4|WP4B -> WP5a||WP5b -> WP5c -> WP6
-> WP10. WP0b, P7, P8 run alongside and must report before WP6/WP7 budgets are quoted.
**After D0 (Path B):** D0 -> WP1 -> WP2a/b/c/f || WP2d narrow -> WP3 -> WP4B -> WP5a || WP5b -> WP5c -> WP10a -> WP6 ->
WP10, with the accuracy path P11 -> P3x -> WP15 -> WP10a beside it; about 182 G to WP10 (`D0.md` 5).

**Total to WP10** (excluding WP11/WP12): about 137 G and 385 C on Path A (Path B about 134 G). Phase 0 (Groups 0-1)
is about 31 G and 68 C (WP0b alone 18 G), which is the cost of the go/no-go and of the baselines.
Revision 3 adds about 48 G and 290 C (P11, P3x, P13-P15 4 G and 18 C; WP13-WP20 22.5 G and 210 C; WP8 +2 G;
WP10 +12 G for RULER and the non-greedy cells; WP0b +6 G for RULER and HumanEval A-vs-A; WP-R1 1 G and 60 C), plus
WP-R2's 3 G and 100 C only if WP-R1 wins: about 185 G to WP10 on Path A. Phase 0 becomes about 41 G.

---

## 9. Measurement and RSI

### 9.1 A/B discipline

- **Numerics before timing, always,** in the same job.
- **Layer and pass A/B:** in-situ HIP-graph replays with 16 cold weight copies (>= 4x MALL) and 96 launches (layers) or
  8 cold passes; 12-30 rounds with the arm order shuffled every round; per-round ratio; two controls (a second graph of
  the baseline and a separately compiled copy).
- **Win rule:** median below both controls' minima by 0.002 AND at least 3/4 of rounds below 1; a loss is mirrored;
  confirm on a fresh seed and allocation and on the other rocke node. Decide a lever within one job.
- **E2E win rule:** one server at a time; a seeded permutation of the arms rotated per pair, 6 pairs; per-cell A-vs-A
  (pinned hybrid twice; spread 0.3-1.5% at CONC 1). WIN = the median ratio below the A-vs-A minimum AND at least 5 of
  6 pairs below 1. Metrics: ms per verify round, per-token time, per-request tok/s, acceptance length, accuracy count.
  Long ISL: decode-only (per-token time at depth L, ms per verify round), since prefill is ATOM's in both stacks.
- **Stamps bias timing** (a stamped build is a different code object, 1.013-1.024 of plain). Time only unstamped builds;
  per-layer timing is never read from sampled stamps either.
- **Clocks:** sample `amd-smi` SCLK/power during timed jobs, drop throttled rounds; neighbour rule in 8.1.
- **Ratios only** in this tree and `RP/`; absolutes in `private-perf/qwen38-true-mono/`.

### 9.2 RSI loop

1. Find the bubble first: stamps timeline decoded per layer (stamp rings, debug build); blame graph of the persistent
   kernel incl. `xl` and round edges; Coz entry/post delays incl. the `xhold` and "CTA leaves layer l" points; poll-spin
   heat maps; WaveScope ATT before/after each lever (`ROCKE_DEBUG_LOC=1`, `llvm-dwarfdump` on PATH); PMC FETCH_SIZE,
   TCC_HIT, TCC_MISS in separate passes. The simulator never ranks levers (it is used only as a bound).
2. Hill climb one lever at a time on the current best body.
3. Lookback after each structural change (spread, `xl`, pass/tail, round, wide): signed lookback (`climb_b3.py
   --lookback signed`, about 1200 cells, exclude values that lost by more than 1.5% everywhere, re-time the top 6 for
   30 rounds).
4. Losers stay registered knobs with byte-identical off state; a knob shows in the kernel name only when set.
5. Intelligent sweeps only (`P/dispatch/tuning/search/`: GP-EI, TPE, SHADE/JDE, Thompson portfolio, racing), warm-started
   from the nearest variant. Build every value of every knob.
6. Per-cell BEST tables; a value enters only after confirmation.

### 9.3 Long-context suite

No long-context accuracy suite exists in the harness. Proposed (WP0b sets it up; user to confirm): RULER needle
retrieval (single and multi-key) at 8K/32K/128K and the cell max, greedy, threshold = within the pinned hybrid's
A-vs-A spread per length. **Confirmed by the user (2026-10-30)** as the long-context suite; scope, haystack choice,
sample counts and cost in 12.8.

---

## 10. Risks

| risk | effect | mitigation |
|---|---|---|
| P1: in-kernel hand-off slower than a graph boundary on `best` | Path A lost; K=3 central about 0.95, K=0 about 0.96 | Path B is pre-planned (WP4B); D0 is a hard gate before any port |
| HC spread below its bound | K=0 margin gone | P3 in situ before ports; kill criterion; WP9 |
| a port misses parity | T eats the margin | per-emitter in-situ <= 1.00 gate in WP2 (a failing port blocks WP3, it is not paid for) |
| SGPR/VGPR spills in the dynamic loop | lost cycles or scratch | P6 on the real at-limit bodies; `args=lds`; spill CI gate |
| round-module compile time | RSI speed | P6 decides inline vs `noinline` per layer body; CPU fan-out |
| wide builds (fused MoE, NUMA never run) | grouped/wide cells | WP2d as new design; P7 anchor; NUMA-area sizing |
| long-L QSA parity | long-ISL cells | WP2c gates every L; Q4 spread exact with tie gate |
| ATOM drift breaks the bit-equality of prologue/meta | state corruption | launch-start ownership (one writer); Tier G in the copy's CI |
| hybrid adopts A1 spread or graph-captured draft step | margin shrinks | pinned baseline; structure-only levers (2.6) |
| timing noise from busy neighbours | false wins | at most one busy neighbour for timed jobs; WP0b validates |
| non-greedy / logprobs | matrix completeness | `logits=out` then WP8 |
| (rev 3) accuracy suites too coarse to see a tolerance lever (GSM8K per-run stderr about 0.0056, HumanEval 0.033: `baseline/MATRIX.md` cell 14) | drift compounds unseen across N levers | T2 teacher-forced logit drift calibrated by P11 is the sensitive gate; suites confirm; per-lever drift budget; T5 re-gate of the composed config |
| (rev 3) tolerance levers compose numerically | a set of individually passing levers fails as a whole | drift budget split by expected value (12.4); T5 on every cell's final config; bitwise twin of every lever stays buildable |
| (rev 3) a non-bitwise reduction becomes non-deterministic | run-to-run output drift, broken race and mutant gates | fixed-order trees only, no float atomics (12.1); determinism gate (12 launches, two tags) unchanged |
| (rev 3) draft vocab pruning loses acceptance on some domain | lower tok/s on code / chat | P14 over GSM8K + HumanEval + MT-Bench prompts; subset size knob; acceptance gate per K |
| (rev 3) long-context levers (N4 split-KV, N10 `q3=mfma`, N5 chunkwise B3, N11 state precision) | RULER loss at 128K/max | RULER at every length per lever (T4); N11 last and off by default |
| (rev 3) the hybrid adopts N1 / N4 / N8 (FlyDSL- or head-side edits) | narrow-cell margin shrinks | pinned baseline (2.6); the structure-only margin (round, wide MoE/B3/NUMA, N2/N3 cross-stage fusion inside one kernel) is listed separately in 12.5 |
| (rev 3) Rust pilot consumes effort without a win | none on serving (default-off) | WP-R1 stop rule (12.6); WP-R2 only after a WIN |

---

## 11. Review resolution

Every finding of `reviews/w1-perf.md` (P-) and `reviews/w1-complete.md` (C-) and where it is resolved. "Fixed" =
the design changed; "rebutted" = kept with evidence.

### Blockers

| id | finding | resolution |
|---|---|---|
| P-B1 | only real-layer stack test lost; fix does not cover `best`; `xhold` "M4 done" undefined with fused MoE; holds lack anchor | Fixed. P1 runs on `best`/`numa` with arms (a)-(e) and a Coz point at "CTA leaves layer l" (8.2 Group 1); default `xhold=layer` (3.2, 4); hold anchor = `xl` arrival, holds re-derived (3.2) and re-climbed (WP3); G1 at -0.05..+0.015 ungated (2.2); D0 is a hard go/no-go before WP1, and Path B has its own budget and WPs (WP4B) |
| P-B2 | G2 about 2x its bound; A1 tail-bound, M1 hides weights; double count with G1; A2 fan-in cost | Fixed. 1.3 rewritten; G2 0.015-0.05 (K=3) and 0.02-0.075 (K=0) with the derivation; entry ramp booked in G1 only; P3 in situ in the FlyDSL copy with variants (i)-(iii), (ii) adds no fan-in; K=0 kill criterion (2.3) |
| C-B1 | `xl` misses C1 and P2 | Fixed. Consumer set {A1, A2, C1, P2}; C1 edge one-to-one; parity slot lifetime; `xl_late=bulk` knob; in P1, `validate_stack` and the gates (3.2, 4, WP2g) |
| C-B2 | ATOM hook at the wrong level | Fixed. 7.6 call-path contract (functions replaced, outputs, deferred-output path) and WP4h |
| C-B3 | two owners of per-step words; n-gram advance not idempotent; WP6a missing | Fixed. Prologue/meta at launch START reading ATOM's buffers, one writer per word, host never runs `StepPrologue` for a true-mono launch (7.3); tail writes only GPU outputs; "WP6a" removed (gate lives in WP2e/WP5a) |
| C-B4 | no tag plan for a whole round | Fixed. Tag/slot/region table (6.2), `validate_stack` over the full round, reused-tag mutant |
| C-B5 | in-launch-written words read by scalar/plain loads | Fixed. Draft positions, `L`, slots in registers from the broadcast `a` or mailbox; the table holds only host-written values; P5 scalar-path negative control; section 4 row |

### Majors

| id | finding | resolution |
|---|---|---|
| P-M1 | no composition discount | Fixed. Multiplicative 0.8 factor, H line, central values (2.1-2.3) |
| P-M2 | tolerance / non-bitwise levers in default | Fixed. G5 0-0.005, G7 0; `8w` split-K tolerance, `8w=n` bitwise alternative (5) |
| P-M3 | T too narrow | Fixed. T -0.05..0 ungated; per-emitter in-situ <= 1.00 gate in WP2 before WP3 (gated T -0.02..0) |
| P-M4 | SGPRs exhausted | Fixed. P6 on real qsa r4s and delta_ple r4s; `args=lds` default; spill/scratch CI (6.2, 6.3) |
| P-M5 | compile time unbudgeted | Fixed. P6 times the full round module and `noinline`-per-layer-body variant and its call cost; decided at D0 (6.1) |
| P-M6 | missing layer-input consumers; `xl` bytes | Fixed. See C-B1; `xl` bytes per row count (3.2); P4 payload hop |
| P-M7 | grouped/wide and long ISL unanchored | Fixed. P7, P8; cells quoted "unknown" (0, 2.4) |
| P-M8 | G6 has no mechanism | Fixed. G6 0-0.015; P10 null-bytes probe before any WP |
| P-M9 | goal mismatch | Fixed. Definition of done (0.7); losing/at-risk cells named with levers (2.4); WP9 for K=0; WP8 for sampling |
| C-M1 | final-mix reference wrong above 8 rows | Fixed. <= 8 bitwise vs `hc_rows`; > 8 tolerance class vs the served hybrid and bitwise vs a raised-`HC_FINAL_ROWS_MAX` reference (5, 7.8) |
| C-M2 | G2 justified by bandwidth; A2 poll volume; K=0 low end a loss | Fixed. See P-B2; K=0 ranges now include values above 1 explicitly |
| C-M3 | no hybrid baseline for most cells | Fixed. WP0b (every cell, pinned hybrid, A-vs-A per cell); WP2d not shipped into the baseline (2.6) |
| C-M4 | grouped/wide matrix incomplete | Fixed. Variant list per cell incl. r9g3, r6g3, r8g4 (2.4); padding (3.3, 7.8); wide sources (5); fused MoE wide re-costed as new design 8 G; NUMA-area sizing (WP2d) |
| C-M5 | `xhold` default does not reproduce the boundary | Fixed. `xhold=layer` |
| C-M6 | weak race gates, no high-level mutants, P4 ungated | Fixed. Gates 5 and 6 (7.8); P4 negative controls |
| C-M7 | cpp-parity exemption missing | Fixed. 6.5: everything in `Q/`, exemption suffix and backend assertions, byte-identity test unchanged; Rust as third engine outside the gate |
| C-M8 | prefill and transitions not designed | Fixed. 7.4 and Tier G chains |
| C-M9 | routing vs goal; no sampler WP | Fixed. 0.7, 7.7, WP8 |
| C-M10 | long ISL under-specified | Fixed. `longest_context` = max_model_len + K bounded by MAX_CONTEXT; `max_model_len` pinned per CONC (WP0b); decode-only metrics; suite proposed (9.3); Tier G rollbacks across page and compression boundaries |
| C-M11 | 6 busy GPUs vs timing hygiene | Fixed. At most one busy neighbour for timed jobs; WP0b validates more (8.1) |

### Minors

| id | resolution |
|---|---|
| P-m1 | Fixed: G4 ramp only, 0.002-0.006; fold counted nowhere (hybrid already fused) |
| P-m2 | Fixed: 6.2 tag plan |
| P-m3 | Fixed: G8 gated -0.01..+0.005, default `xhold=layer` |
| P-m4 | Fixed: P4 skewed arrivals |
| P-m5 | Fixed: 2.6, pinned baseline |
| P-m6 | Fixed: 7.3 launch-start ownership; ATOM reserves k+1 blocks before the launch |
| P-m7 | Fixed: 9.1, no per-layer timing from sampled stamps |
| C-m1 | Fixed: P1 control 1.03-1.09 on `best` |
| C-m2 | Fixed: `F/models/qwen4/weights.py:37-40` |
| C-m3 | Fixed: WP0a, routing table in WP4h, graph capture in WP4h, `xl` emitters WP2g |
| C-m4 | Fixed: WP1 writes `research/rust-direct.md`; PLAN.md updated |
| C-m5 | Fixed: gate 3 vs ATOM's sampler path `tie="low"` with NaN/inf rows |
| C-m6 | Fixed: P2 rows 8 and 16 |
| C-m7 | Fixed: Q4 planted-tie gate (5, 7.8) |
| C-m8 | Fixed: G1 evidence row says one-pair microbench; P4 payload hop |
| C-m9 | Fixed: PLAN.md now says `loop` is optional |
| C-m10 | Fixed: 2.5 / WP12 on-device stop and table re-reads |
| C-m11 | Fixed: server-start compiles and e2e counts in costs (8.2) |
| C-m12 | Fixed: section 0 and 4 now consistent (`xhold` counter, grid sync only at ticket / `loop` start / knob) |
| C-m13 | Fixed: P6 builds the widest round |
| C-m14 | Fixed: simulator used only as a bound (9.2) |
| C-m15 | Fixed: 8.1 cluster logistics |
| C-m16 | Fixed: 7.2 inherited refusals |
| C-m17 | Fixed: gate 2 on the debug build, gate 3 on the served binary |

Rebuttals (kept with evidence): none of the findings was rejected. Two were narrowed:
- P-B1's "make P1 a hard go/no-go before WP4": made stricter, a go/no-go before WP1 (D0), because Path A and Path B
  share every WP2 emitter but differ in WP1's framework shape (dynamic loop, `xl`, tag plan).
- P-M9's "budget loop-style multi-step decode for K=0": kept out of the default budget (it changes ATOM's per-step
  contract) and made the named closing lever WP9, triggered by the D0 kill criterion.

---

## 12. Rev3: accuracy-gated ceiling

Revision 3 (W2, 2026-10-30) re-budgets the design for the user decisions in `PLAN.md`: (1) non-bitwise restructuring
is allowed behind per-stage tolerance tests and the accuracy suites (GSM8K, HumanEval, RULER 8K/32K/128K/max) within
the pinned hybrid's A-vs-A spread, bitwise staying the default where it costs nothing; (2) the Rust direct compiler is
a pilot track; (3) non-greedy sampling and logprobs are in scope. Sections 0-11 stay the bitwise-only plan of record
and the fallback of every lever below. All shares are hybrid units (section 0 abbreviations), ratios only.

### 12.1 Principles

1. **Every non-bitwise lever has a bitwise twin** (a knob value of the same emitter). The tolerance variant becomes
   default for a cell only if it passes T0-T5 (12.4) AND wins against its twin under the 9.1 WIN rule. On a tie the
   bitwise twin stays default ("bitwise where it costs nothing"). Losers stay default-off knobs.
2. **Non-bitwise never means non-deterministic.** Every reduction is a fixed-order tree in f32 (no float atomics, no
   arrival-order sums, no last-arriver accumulation). Gates 5 and 6 of 7.8 (race stress, mutants) and the determinism
   gate (12 launches without zeroing, two tags bitwise equal) apply to tolerance variants unchanged.
3. **Draft-side levers are output-exact.** Anything that only changes the drafter (its head, its layer) changes the
   proposals, never the verified tokens: greedy verify keeps the target's argmax, and rejection sampling keeps the
   target distribution for any proposal distribution (12.8). These levers are gated by acceptance length (T3), not by
   the accuracy suites.
4. **Restructure, do not re-port.** The hybrid's record (`research/q38-learnings.md` pitfall 2): ports of the same
   structure land at parity; wins came from restructuring (expert-major fused MoE) and from schedule levers. The
   accuracy-gated levers are restructurings that the bitwise contract forbade.
5. **Booking rule.** A lever is booked at its expected value: p(pass) x (value if it passes) + (1 - p) x (its bitwise
   twin's value), minus the twin's value already in the bitwise budget. Increments are composed at **0.75** (the
   hybrid's own record was 0.80 with four levers on partly shared mechanisms; these levers all act on the same serial
   chain, so the discount is larger). Sensitivity at 0.6 and 0.9 is in 12.5.

### 12.2 Grounding units

- **Stage shares of the K=3 round in hybrid units** (mono shares of 1.3 x 0.93 NUMA / 0.856, i.e. x1.086 for non-MoE
  stages; the hybrid's MoE from `ledger/fused-moe.md`): a1 0.092, b3 0.081, m1 0.079, c1 0.059, b1 0.052, router window
  (m3 + routing, 0.26-0.28 of the mono MoE, `MONO_IMPROVEMENTS_SHEET.md` 2) about 0.07, rest of the MoE about 0.21,
  q1-q7 0.097, a2 0.031, m2 0.015, p 0.005; lm_head 0.063 (draft heads about 0.047, target 0.016: the 0.049 / 0.016
  split of `RP/GAP_TABLE.md` 2.1 at the hybrid head's 0.78-0.83). K=0 (delta r1 in-server launch shares x 0.894): a1
  about 0.13, m4 0.13, b3 0.11, m1 0.10, b1 0.08, m5 0.08, m3 0.07, c1 0.05, a2 0.04.
- **One hop.** One blamed hop is 0.043 / 0.061 / 0.021 of a delta r4s / delta r1 / qsa r4s launch (`RP/BUBBLES.md`
  2), x0.71 with NUMA dual copies (section 4). Removing one serial hop from every delta layer (36 of 48, about 0.75 of
  target layer time) is worth about **0.018 hybrid units at K=3 and 0.029 at K=0**; from every qsa layer (12 target +
  3 draft) about 0.004 / 0.003. The q4 -> q5 hop is the exception: 0.147 of a qsa r4s launch at L 8192.
- **Work segments** on the blame critical path (`RP/BUBBLES.md` 2): work a1 + work m1 = 0.103 (delta r4s) / 0.128
  (delta r1) of a launch; work b3 0.111 / 0.153; work m3 0.051 / 0.056; work c1 0.054 (qsa r4s); hop b3 > c1 0.096 /
  0.082; hop q4 > q5 0.147, hop q6 > q7 0.043, hop q3 > q4 0.038 (qsa r4s).
- **Serial hops per layer today:** delta 11 (M5(l-1)->A1, A1->A2, A2->B1, B1->B3, B3 siblings, B3->C1, C1->M1,
  M1->M2, M2->M3, M3->M4 routing, M4->M5), qsa 14 (`RP/ROADMAP.md` QSA row). A global reduce + broadcast costs two
  hops unless the consumers reduce redundantly, which is affordable for at most about 40 partials of 324 or 513 f32
  per consumer (26-82 KB of L2 reads), not for 160 (P3x variant vi measures where the line is).
- **What the floors leave:** the pass runs at 2.31x its issued-bytes floor and CTAs poll 0.60-0.75 of the time, so a
  round of about 0.86 hybrid units is still about 2x its bytes floor (`RP/GAP_TABLE.md` 3). The ceiling is set by
  hops and post-poll tails, not bandwidth: levers that only cut bytes (MXFP4 activations, `spec=prev`) do not pay.

### 12.3 Non-bitwise levers per stage and hand-off

"If pass" = range and central value saved, hybrid units (K=3 CONC 1 / K=0 CONC 1, served short context). "Incr" = the
booked expected increment over the bitwise budget (12.1 rule 5). Fallback = the bitwise twin and its budget line.

| id | stage / hand-off | mechanism | share attacked (12.2) | if pass K=3 / K=0 | p(pass) | incr K=3 / K=0 | accuracy risk | gate (12.4) | bitwise twin (fallback) |
|---|---|---|---|---|---|---|---|---|---|
| N1 | HC chain A1/A2, M1/M2 and the M5(l)->A1(l+1), C1->M1 hand-offs | (a) free-order K-split of the HC down-projection over 80-160 CTAs, f32 partials, A2/M2-side fixed-order sums of <= 40 partials (FlyTape's many-wave mix_down, on the mono's MXFP8 weights); (b) **algebraic re-association, norm after GEMV**: the down product of the normalized streams is the sum over the 4 streams of `rrms_s * W_s (x_s * (1+g_s))`, so the 4 per-stream partial dots start as soon as stream tiles land and the per-stream `rrms_s` (from the per-tile square sums M5 / C1 already publish) scales them at the end, taking the 4-stream norm out of A1's serial tail; (c) **partial-down in the producer epilogue**: M5 task t and C1 task t (which own the same 16 hidden columns x 4 streams) multiply their 64 inputs by the next stage's 324 x 64 weight slice (about 11 KB, issued after the task's own last load) and publish 324 f32 partials; A1 / M1 become 20-CTA reducers. (c) works across a launch boundary, so Path B keeps it. MXFP8 pack spread over the reducers | a1 + m1 incr 0.17 / 0.23; work a1 + m1 0.103 / 0.128 of a delta launch; A1 tail/span 0.80-0.94; M1 post-poll tail 0.40 of its incr | 0.035-0.065, 0.05 / 0.05-0.09, 0.07 | 0.8 | 0.016 / 0.020 | low-medium: f32 sum order; MXFP8 codes of un-normalized vs normalized streams differ where `rrms` is not a power of two (same relative error distribution) | T0 (A1/M1/A2/M2 <= 1.25x the mono's own 0.002 max HC drift), T1, T2 | G2 spread variant ii/iii (P3): 0.03 / 0.045 |
| N2 | B3 -> C1 and Q7 -> C1 (out_proj) | **out_proj K-split into its producers**: each B3 (head, split) task and each Q7 task multiplies its 64-128 columns of `o` by its `W_o` slice (2560 x 64-128 MXFP4, one scaled-MFMA K step, issued before the B3/Q7 poll) and publishes 2560 f32 partials; C1 becomes a fixed-order reduce of its 16 columns' partials + the HC write (+ N1c's M1 partial-down). The C1 GEMV (incr 2.0-3.9x its floor at 160 CTAs) leaves the chain; the B3 -> C1 hop stays but carries partials | c1 0.059 / 0.05; hop b3 > c1 0.096 / 0.082 of a launch | 0.01-0.035, 0.02 / 0.01-0.035, 0.02 | 0.7 | 0.014 / 0.014 | medium: out_proj sum order; the bf16 rounding point moves into C1 | T0-T2 | C1 bitwise port (`8w=n`): 0 |
| N3 | M2 -> router -> experts (fused MoE window) | **MFMA router + on-device top-k**; P15 picks between (a) the router K-split fused into the M2 epilogue (each M2 task's 513 x 32 bf16 slice issued before its poll, 513 f32 partials; 8 per-XCD reducers sum the 80 partials in fixed order, run top-10 + renormalized softmax + shared gate and NUMA-broadcast the route) and (b) an MFMA N-split router over all 256 CTAs with the logits all-to-all (2 KB) and a redundant per-CTA top-10 at fused-MoE entry (FlyTape's per-workgroup routing), removing the route-table build and its barrier. The mono M3 runs 65/130 tasks at 8.5x its floor; the fused MoE's fixed overhead is 0.625 (r4s) / 0.845 (r1) of the mono MoE, gated by this window | router window about 0.07 / 0.06 | 0.015-0.04, 0.025 / 0.015-0.04, 0.025 | 0.7 | 0.014 / 0.014 | medium: a near-tie at the top-10 boundary flips an expert (Tier A today: no flips) | T0 router-flip count <= the hybrid's own vs golden; T2 | G6 (P10, wider router rows): 0.005 |
| N4 | QSA chain Q3-Q7 (served L <= 2048) | (a) **dense under budget**: at L <= 2048 Q4 takes all blocks but ranks them (`qsa_fast.py:19-23`), so Q5-Q7 run in rank order; a position-order dense path skips the Q3 score consumer and Q4 (Q3 still writes the completing block's compressed key, off the critical path): -2 hops including the 0.147 q4 > q5 segment; bitwise if P13 finds Q5-Q7 order-free; (b) **split-KV online softmax**: Q5 + Q6 + Q7 as one flash-decode stage (exp2 running max / sum, PV per 64-token stage, f32 probabilities instead of Q6's bf16 publish, merges in fixed groups of 8 as in FlyTape's row gang): -1 to -2 hops. A last-arriver Q4 fold (ROADMAP) is NOT taken: last-arriver hops lost 1.6-2.3x (`ledger/schedule.md`) | in-server qsa launch q3 0.02, q4 0.04, q5 0.07, q6 0.03, q7 0.13 = 0.29 (r4s) / 0.32 (r1); qsa launches about 0.24 / 0.22 of the round | 0.02-0.045, 0.03 / 0.02-0.045, 0.03 | 0.75 | 0.021 / 0.021 | medium: softmax order; above budget (8K+) the sparse path keeps Q4 | T0 (Q7 pack vs golden), T2, T4 RULER at every length | G5 Q4 chunk spread (exact): 0.002 |
| N5 | B3 at verify rows (r2s-r4s, grouped, wide) | **chunkwise (WY / UT) form of the gated delta rule** for the k+1 verify rows: the rows' rank-1 updates become one small MFMA-able block; each row's snapshot `S_r` (ATOM's rollback contract, 7.2) is formed from the shared WY factors in parallel instead of four dependent steps | b3 0.081 at K=3 (0.14 of an r4s launch at 2.4x the r1 bytes: the serial row chain) | 0.01-0.03, 0.018 / 0 | 0.6 | 0.011 / 0 | medium-high: f32 re-association in a state carried over the whole sequence | T0 state after every row <= 1e-4 rel (ROADMAP tolerance), T2, T4 RULER 128K / max | B3 tuned sets: 0 |
| N6 | dense GEMVs B1, Q1, P1 (C1 is N2) | **re-blocked full-grid streaming**: FlyTape's wave-owned rows (a wave owns whole rows, every 16 B load of its rows in flight, DPP + `readlane` reduce) or the 8-wave K-split of the post-poll MFMA chain (20 -> 10 k_iters + one LDS reduce); weights still issued before the poll | b1 0.052 + q1 0.016 / b1 0.08 + q1 0.016; B1 incr 1.4-1.7x its floor | 0.005-0.02, 0.01 / same | 0.7 | 0.007 / 0.007 | low: per-column sum order | T0-T2 | `8w=n` N-split (bitwise): 0 |
| N7 | activation format at hand-offs (A2/M2, B3 out, Q7 pack) | (a) **bf16 hand-off activations**: the producer skips the MXFP8 amax / E8M0 / convert pack on its critical tail; consumers decode MXFP4 weights to bf16 as the head does (`cvt_scalef32_pk_bf16_fp4`; the head streams 0.87 of its time with bf16 activations); payload doubles but stays under 1% of a layer's bytes. (b) **MXFP4 activations (W4A4)** are NOT booked: activations are under 1% of the bytes, MFMA rate is not the bound, and the accuracy risk is the largest of any lever; off knob for the wide MoE only | pack latency in 4 producer tails | 0-0.015, 0.005 / same | 0.5 | 0.0025 / 0.0025 | (a) low (more precise than the served MXFP8 reference, still a deviation); (b) high | T0-T4 | MXFP8 packs: 0 |
| N8 | draft heads (K >= 1) | **vocab-pruned draft head**: the draft heads read a corpus-frequency subset (16K-32K of 248320 rows, a re-packed copy of the MXFP4 head, 1/8-1/16 of its bytes) with the same fused argmax; the target head is unchanged, so verified tokens are unchanged (12.1 rule 3). Drafter-layer approximations (reduced top-k in the draft MoE) stay off knobs | draft heads 0.047 (K=3), 0.031 (K=2), 0.016 (K=1) | 0.02-0.04, 0.03 net of acceptance (K=3) / 0 | 0.85 | 0.021 (K=3), 0.014 (K=2), 0.007 (K=1) / 0 | none on outputs; acceptance only | T3 per K over GSM8K + HumanEval + MT-Bench prompts (P14 offline first); non-greedy residual rule (12.8) | full-vocab draft head: 0 |
| N9 | cross-layer `xl` hand-off and round tail | booked inside N1 (the `xl` payload carries N1c's partials for A1 instead of the 4 streams) and G3 (final HC mix above 8 rows is already tolerance class, section 5) | - | - | - | - | - | - | G1 / G3 |
| N10 | long context: Q3 index scores | **`q3=mfma`**: index scores by bf16 / fp8 MFMA instead of the f64 dot; the exact Q4 chunk spread (rev 2) stays | Q3 above L 2048 (f64 dot over L/4 compressed blocks; 26x its floor at L 8192) | 8K: 0.005-0.02, 0.01; 64K-248K: 0.02-0.06, 0.035 (with N4b) | 0.6 | long-ISL cells only | medium-high: selection flips at the 512th score's tie margin, which is what RULER tests | T0 (selection differs only at the tie margin), T4 RULER every length | f64 scores: 0 |
| N11 | GDN state precision | bf16 storage of the recurrent state snapshots (f32 math) | state bytes, wide and long cells only (r16: 16 snapshots per head) | wide: 0-0.02 | 0.3 | not booked | high: recurrent error over 256K tokens | T0, T4 RULER max | f32 state: 0 |

**Hop count after revision 3** (12.2 units): qsa 14 -> 11-12 at L <= 2048 (N4a -2, N4b -1), 13 above budget. Delta
stays at 11 in the central case: N1-N3 take the WORK out of A1, M1, C1 and M3 (their post-poll tails) but keep the
reduce / broadcast hops. A delta hop is cut only where P3x variant vi shows that consumer-side redundant reduction of
<= 40 partials beats a reducer hop (then about 0.018 / 0.029 per hop; not booked).

### 12.4 Tolerance ladder (replaces 7.8 gate 7 for every N lever)

| step | scope | test | pass rule | must-fail mutant |
|---|---|---|---|---|
| T0 | stage (Tier B extended, the T11 rig) | stage output vs the golden fp32 chain from the stage's own upstream mailboxes; router-flip count (Tier A/F fixtures, real layers 0 and 3); Q4 selection diff; B3 state after every row; determinism | max and RMS drift <= max(1.25x the hybrid stage's own drift, the MXFP8 re-quantization floor); flips <= the hybrid's (0 today); Q4 diffs only at the 512th-score tie margin; state <= 1e-4 rel | a dropped partial, a doubled partial, a wrong per-stream `rrms` |
| T1 | layer (Tier A) | layer output drift vs golden from the layer input alone, rows 1 / 4 seq / 16, QSA at L 639 / 8191 | <= 1.25x the hybrid's own drift in the same cell (today the mono is 0.96-1.10x the self-calibrated bound) | a sign-flipped stream mix |
| T2 | end to end, teacher-forced (P11) | 48 layers + head on real text (GSM8K, HumanEval, long-prompt prefixes); per-token KL(golden, arm) and top-1 agreement with the golden | mean KL and top-1 disagreement within the lever's share of the drift budget B; B = half the largest planted perturbation that kept GSM8K, HumanEval and RULER inside the hybrid's A-vs-A in P11 | planted noise at 2x the budget at one stage |
| T3 | acceptance | acceptance length per K (1..3), CONC 1 and 4 | >= the pinned hybrid's A-vs-A minimum | the N8 subset cut to 1K rows |
| T4 | suites | lm_eval GSM8K 1,319 5-shot (flexible + strict), HumanEval 164 pass@1, RULER 8K / 32K / 128K / max (12.8) | inside the pinned hybrid's A-vs-A spread per suite and length (`baseline/MATRIX.md` cell 14: GSM8K per-run stderr about 0.0056, HumanEval 0.033; the suites confirm, T2 is the sensitive gate) | - |
| T5 | composed | T2-T4 on each cell's final config | as T2-T4 with the whole budget B | - |

**Drift budget split** (T2 shares of B, by expected value): N1 0.2, N3 0.2, N4 0.2, N2 0.15, N5 0.15, N6 0.05, N7
0.05; N8 0 (output-exact); N10 / N11 take theirs from N4's long-context allowance. If T5 fails, levers are dropped in
ascending order of (increment / drift used) until it passes.

### 12.5 Revised per-cell budgets (hybrid units; central and range; Path A unless stated)

Ranges: the best case is the bitwise best case minus 0.75x the sum of (if-pass high - fallback low); the worst case is
the bitwise worst case (a lever that fails its gate is not paid for). Cells without a measured decomposition (P7
grouped / wide, P8 long ISL, WP0b non-greedy) are derived and marked; they are re-issued when the probes report.

| # (MATRIX.md) | cell | hybrid / mono there | bitwise only | accuracy-gated | booked increments (before x0.75) | anchor |
|---|---|---|---|---|---|---|
| 1 | K=3 CONC 1 GSM8K | round 0.854-0.865 | 0.94 (0.87-1.00) | **0.86 (0.67-1.00)**; Path B 0.87 (0.70-1.01) | N1 .016, N2 .014, N3 .014, N4 .021, N5 .011, N6 .007, N7 .0025, N8 .021 = 0.107 | measured shares |
| 6 | K=0 CONC 1 GSM8K | per token 0.874 | 0.95 (0.88-1.01) | **0.89 (0.71-1.01)**; Path B 0.90 (0.74-1.01) | N1 .020, N2 .014, N3 .014, N4 .021, N6 .007, N7 .0025 = 0.079 | measured shares |
| 9 | K=1 CONC 1 (`r2s`) | round 0.874 | 0.945 (0.87-1.01) | 0.88 (0.70-1.01) | as cell 1 with N5 .006, N8 .007 = 0.088 | derived |
| 11 | K=2 CONC 1 (`r3s`) | round 0.880 | 0.94 (0.87-1.01) | 0.87 (0.69-1.01) | as cell 1 with N5 .009, N8 .014 = 0.098 | derived |
| 4 | K=3 CONC 1, 8K / 1K | rnd 0.877 | 0.945 (0.87-1.02) | 0.87 (0.69-1.02) | N4 above budget (N4b only) .013, N10 .006, rest as cell 1 = 0.104 | derived; P8 |
| 13 | K=3 CONC 1, 64K / 248K | rnd 0.866 / 0.844 | 0.97 (0.88-1.05), at risk | 0.88 (0.66-1.05) | N4b + N10 at long L .030, rest as cell 1 = 0.116 | unanchored: P8 |
| 8 | K=0 CONC 1, 8K / 1K | TPOT 0.881 | 0.955 (0.88-1.02) | 0.90 (0.72-1.02) | as cell 6 with N4 .013, N10 .006 = 0.076 | derived; P8 |
| 7, 7b | K=0 CONC 4 (`r1..r4`) | TPOT 0.880 | 0.96 (0.88-1.02), at risk | 0.90 (0.72-1.02) | as cell 6 = 0.079 | derived; WP0b |
| 10, 10b | K=1 CONC 2 (`r4g2`; FlyDSL MoE in the hybrid) | round 0.921-0.930 | 0.90 (0.80-1.00) | 0.85 (0.66-1.00) | about 0.07 (hop share lower than narrow) | derived; P7 |
| 2 | K=3 CONC 2 (`r8g4`, `r4s`) | round 0.984 | 0.86 (0.75-1.00) | 0.815 (0.66-1.00) | about 0.06 | derived; P7 |
| 3, 3b | K=3 CONC 4 (`r8g4 r12g4 r16g4`) | about 1.0 | 0.84 (0.72-1.00) | **0.80 (0.62-1.00)** | about 0.055 (N11 not booked) | derived; P7 |
| 5 | K=3 CONC 4, 8K / 1K | rnd 0.993 | 0.85 (0.73-1.00) | 0.81 (0.63-1.00) | about 0.055 | derived; P7, P8 |
| 12, 12b | K=2 CONC 4 (`r6g3 r9g3 r12g3`) | 0.990-0.999 | 0.85 (0.73-1.00) | 0.81 (0.63-1.00) | about 0.055 | derived; P7 |
| new | K=3 CONC 1, non-greedy (T 0.7, top-p 0.95) | none | phase 1 0.96 (0.89-1.03); phase 2 0.945 | phase 1 0.88 (0.70-1.03); phase 2 0.865 | as cell 1; phase 1 pays the logits write and ATOM's sampler / rejection-sampler launches | WP0b, WP8 |
| new | K=0 CONC 1, logprobs = 5 | none | phase 1 0.97; phase 2 0.955 | phase 1 0.91; phase 2 0.895 | as cell 6 | WP0b, WP8 |
| 14 | accuracy suite (K=3 CONC 4) + RULER | A-vs-A | holds by construction (bitwise) | must hold: T4 / T5 | - | WP10 |

- **Wide cells (derived, unanchored until P7).** The hybrid is at mono parity there (wide builds keep the FlyDSL MoE and
  B3 and drop every lever, `baseline/MATRIX.md`). The bitwise true mono brings: NUMA (0.92-0.95 per narrow layer,
  assumed 0.95-0.98 at 8-16 rows where hops are a smaller share); the fused MoE with union dedupe (expert loads are
  issued per row: about 176 at r16 against a union of about 106 for 4 independent groups of U 28.5 over 512 experts,
  0.60 of the expert bytes; 2.62 MB per expert-row from `RP/GAP_TABLE.md` 3, 107.6 MB / 41 loads); HC dedupe (A1/M1
  issue `w_down` per row: 7.6 MB each at r4 = 4x the 1.9 MB unique, about 30 MB each at r16); B3 wide; the round
  structure. The accuracy-gated levers add less there (hops are a smaller share at 16 rows).
- **Composition sensitivity** (cell 1 / cell 6): factor 0.6 -> 0.876 / 0.903; 0.75 -> 0.86 / 0.89; 0.9 -> 0.844 /
  0.879.
- **Beyond the central case, not booked:** WP9 multi-step decode at K=0 (outside-graph share about 0.041 hybrid units
  of a K=0 step, 2.3); the Rust direct levers (0-0.02, 12.6); a delta hop cut by consumer-side reduction (about 0.018 /
  0.029 per hop, P3x vi); N11 on wide / long cells.
- **What "crush" means here:** cell 1 at 0.86 hybrid units is about 0.74 of the mono round, about 1.16x the hybrid's
  per-request tok/s; the wide cells, where the hybrid has no margin over mono, reach 1.2-1.25x. The narrow cells reach
  0.80 or below only at the optimistic end of N1-N5; that is why WP13-WP17 come first and P11 / P3x report before D0.
- **Hybrid-adoptable vs structure-only margin (2.6).** N1, N4, N6, N7 are FlyDSL stage edits and N8 a head-side edit:
  the hybrid could adopt them. N2 / N3 cross stage boundaries inside one layer kernel (the mono could follow at FlyDSL
  effort). Structure-only to the true mono: the round launch (G3), rocKE fused MoE / B3 / NUMA on the wide builds, HC
  dedupe, the drafter in kernel. If the bar becomes "the best hybrid", cells 2, 3, 5, 12 keep most of their margin and
  cells 1 / 6 keep about G3 + 0.75x (N2 + N3).

### 12.6 Rust direct compiler: pilot track (user decision 2)

Source: `research/rust-direct.md`. The direct compiler (rocKE IR -> HSACO in Rust, no LLVM; gfx950 + gfx1151) compiles a
median 27-31x faster than COMGR at code-speed parity on the census (conv 0.986, attention forward 1.003, GEMM 0.992) and
is byte-deterministic. It compiles no Q38 kernel today: unknown op names (`tagged_poll`, `buffer_load_cp` /
`buffer_store_cp`, `int_to_ptr`, `mfma_scale_mx` 16x16x128, `cvt_scalef32_pk_*`, `permlane16_swap`, `math.exp`, f64
`sqrt`) raise instead of falling back, and about 250 `inline_asm` sites in 8 representative builds are refused.

**Levers it unlocks for this design** (none booked in 12.5 until WP-R1 wins):

| lever | why LLVM / COMGR cannot | what it buys here |
|---|---|---|
| exact waitcnt (a per-register scoreboard we own) | `SIInsertWaitcnts` is opaque; the asm poll forces `vmcnt(0)` every retry | counted `vmcnt` "poll first, then weights": the N1c / N2 / N3a producers issue their weight slices and poll without draining them; a scalar-memory poll on `lgkmcnt` (needs P5-style validation; `step_load=smem` lost on the toy) |
| unhoistable polls (a leaf op expanded after isel, like `flag_wait_eq`; LICM moves only pure ops) | IR has no while loop; correctness rests on side-effecting asm + `convergent` | every poll, including the partial-pair polls of N1-N3, as an IR op without asm; rule: never mark mailbox / scratch arguments `noalias readonly` |
| cache-policy bits per access (sc0 / sc1 / nt on MUBUF and global) | asm, or Python-only `buffer_*_cp` aux bits | `nt` on every single-use stream and partial buffer (B3's `state_nt`, 0.9695, was the largest B3 lever); `sc1` write-through partial pairs without `wbl2` |
| register pinning (precolored SGPR ranges) | not expressible; `fresh` only prevents merging | (step, layer slot, parity, mailbox bases, NUMA copy offsets) in fixed SGPRs for the whole round; kernarg preload (0.44 of stage-entry stalls on the toy); relief for qsa r4s `numa` at SGPR 104 with a 68 B scratch frame |
| compile time | the COMGR tail is 1-3 orders above its median and history-dependent (`iglp_opt(1)`) | a several-MiB whole-round module: about 10x more candidates per hour for hill climbs with lookback and exhaustive combination sweeps (9.2) |
| no exec-join spills (spill stores under the definition's EXEC, reloads never across an EXEC write, `verifyspill`) | possible; needs an ISA lint | safer register use at 512 threads (the mono runs at 228-246 VGPRs; FlyTape's full-shape builds spill 146-159) |

**Pilot (WP-R1), now, never on the critical path:** the minimal overlay of the rust branch's own commits (about 160
files, manifest `overlay/rust-direct.manifest`, no git state change; `lower_llvm.py`, C++ / CMake and GDN files not
taken); g950p's crate from `copy-rocm-lib-agents` only if it is a superset passing `cargo test`, `isa_tables check`
(both ISAs), `hazard_corpus` and `bench950` identity; fallback hygiene (every Q38 op name and asm arity becomes
UNSUPPORTED with a COMGR fallback; a silent fallback is a test failure); IR-only direct builds (`mb_impl=ir`,
`cvt=intrinsic`, `int_to_ptr`); first targets `mx_decode_gemv` (`buffer_*_cp`, `cvt_scalef32_pk_bf16_fp4`,
`permlane16_swap`, `tagged_poll`, scoped atomic; about 6-8 d) and the toy-chain mailbox (`tagged_poll`,
`buffer_store_cp`, `int_to_ptr`, `s_getreg`, `s_sleep`; about 5 d), behind `FrameworkKnobs.code_object_backend`
(default `comgr`, off state byte-identical). Cargo builds on alola CPUONLY jobs with the private rustup (alola skill
rule 100). Gates: bitwise vs the COMGR object for the same IR and vs the mono head; `target_compile --max-vgprs`,
`verifyspill`, `ROCKE_DIRECT_THREADS=1` identity; exec-join ISA lint; shuffled paired A/B vs COMGR with a byte-copy
null. **Stop rule:** if direct is not below 1.00 (nulls tight) on the toy chain and the head after the waitcnt / cpol
levers, it stays an off knob and WP-R2 does not start.

**WP-R2 (only after a WP-R1 WIN):** B3 and fused-MoE ops (`math.exp` bitwise to LLVM's expansion, DPP, `ds_or`, `ff1`,
`v_maximum3`, MXFP8 converts, `mfma_scale_mx` 16x16x128 with opsel), branch relaxation (a hard blocker past about
+-128 KiB, needed for a whole-round kernel), kernarg preload, the PC -> IR line map for WaveScope / Coz blame, SGPR
pinning; then the levers above as default-off RSI knobs (WP11). About 4-6 engineer-weeks in all.

### 12.7 Work-package deltas

**New probes** (8.2): **P11** (accuracy-gate calibration) and **P3x** (tolerance HC variants) report **before D0**;
P13 (QSA under-budget order), P14 (draft vocab pruning, offline) and P15 (router restructure in the hybrid's rocKE fused
MoE) run in Groups 0/1 and gate only their own WPs.

**New WPs, ordered by expected value over risk** (Group 3b; each extends a WP2 emitter with a tolerance knob, gated
T0-T2 and then in situ in `layers=1` kernels vs its bitwise twin, so Path A and Path B share them; T3-T5 in WP10 /
WP10a):

| order | WP | lever | extends / depends | incr K=3 / K=0 (before x0.75) | risk | cost |
|---|---|---|---|---|---|---|
| 1 | WP13 | N8 vocab-pruned draft head (+ drafter off knobs) | WP2e head adapter, P2, P14 | 0.021 / 0 | low (output-exact; acceptance) | 2 G, 10 C |
| 2 | WP14 | N4 QSA dense under budget + split-KV online softmax | WP2c, P13 | 0.021 / 0.021 | medium (RULER) | 4 G, 40 C |
| 3 | WP15 | N1 HC chain: K-split, norm after GEMV, partial-down in the M5 / C1 epilogues, `xl` carrying partials (with WP2g) | WP2a, WP2g, P3x, P11 | 0.016 / 0.020 | low-medium | 3 G, 30 C |
| 4 | WP16 | N3 MFMA router + on-device top-k in the fused MoE (narrow and wide) | WP2d, P10, P15 | 0.014 / 0.014 | medium (router flips) | 3 G, 30 C |
| 5 | WP17 | N2 out_proj K-split into the B3 / Q7 epilogues; C1 = reduce + HC write | WP2b, WP2c, B3 emitter, WP15 | 0.014 / 0.014 | medium | 3 G, 30 C |
| 6 | WP18 | N6 dense GEMV re-block (wave-owned rows / 8-wave K-split) for B1, Q1, P1 | WP2b, WP2f | 0.007 / 0.007 | low | 2 G, 20 C |
| 7 | WP19 | N5 chunkwise B3 for verify rows (narrow, grouped, wide) | WP2d (B3 wide), WP3 | 0.011 / 0 | medium-high (RULER) | 4 G, 40 C |
| 8 | WP20 | N7a bf16 hand-offs; N11 bf16 state (off by default); N10 `q3=mfma` promoted to an accuracy-gated default candidate inside WP7 | WP2a/b/c, WP7 | 0.0025 / 0.0025 (+ long-ISL N10) | low (N7a) / high (N11) | 1.5 G, 10 C |
| parallel | WP-R1 | Rust direct pilot (12.6). **Done (R1, R2); superseded by PLAN decision 6 and D0.6: WP-R3 ports the direct backend into the working tree (default `comgr`, per-family direct default where it won on both nodes)** | none | not booked (D0 upside about 0.003 / 0.01) | none (default off) | 1 G, 60 C |
| conditional | WP-R2 | Rust direct for B3 / MoE, branch relaxation, preload, line map, pinning | WP-R1 WIN | not booked | none (default off) | 3 G, 100 C |

**Changed WPs:** WP0b adds RULER 8K/32K/128K/max A-vs-A, HumanEval A-vs-A and the two non-greedy cells (24 G);
WP2a-WP2d emitters reserve the tolerance-knob hooks (off state byte-identical, no cost change); WP7 takes N10; WP8
grows to on-device sampling, fused logsumexp / top-N logprobs and the speculative residual (12.8; 6 G); WP9's trigger
becomes "P1 + P3 + P3x project less than 0.04 at K=0"; WP10 runs T3-T5 per cell including RULER and the non-greedy
cells, with a WP10a checkpoint after WP13-WP15 (22 G); WP11 (RSI) climbs tolerance variants only beside their bitwise
twins and takes the Rust levers after WP-R1 wins. **The critical path is unchanged** to WP3; Group 3b runs alongside
WP3-WP5c inside the 6-GPU cap; WP10a sits between WP5c and WP6. Totals in 8.2 (about 185 G to WP10 on Path A).

### 12.8 Scope: RULER, non-greedy sampling, logprobs (user decisions 1 and 3)

**RULER (the long-context suite, confirmed):**
- Tasks: needle retrieval single-key and multi-key (multi-value / multi-query if the generator offers them offline),
  greedy, at 8K / 32K / 128K / max (max = the cell-13 length class, `--max-model-len 262141`), at K=3 (served) and K=0.
- Haystack generated offline on the cluster (noise or repeated-text haystack); if an essay haystack is used, its data
  is staged first. Which one ran is recorded with the results.
- Samples: about 50 per task per length (binomial stderr about 0.03 at accuracy 0.95, 0.07 at 0.5); the threshold is
  the pinned hybrid's A-vs-A spread per length (two hybrid runs), measured in WP0b. Prefill dominates the cost at
  128K / max (cell 13 is prefill-bound), so WP0b measures the cost first and sets the sample counts from it.
- Runs: WP0b (baseline A-vs-A), T4 of every long-context lever (N4, N5, N10, N11), WP10 / T5.

**Non-greedy sampling and logprobs (in scope, WP8):**
- Phase 1, `logits=out` (7.7): the in-kernel head writes f32 logits rows (248320 x 4 B, about 1 MB per row) and ATOM's
  sampler and rejection sampler run on them: bitwise ATOM's inputs, at the cost of the logits write plus ATOM's
  sampler launches (the non-greedy cells of 12.5).
- Phase 2, on device: temperature, top-k, top-p and min-p in a second pass of the head epilogue; ATOM's RNG contract
  (same generator, seeds and offsets): bitwise when ATOM's op sequence is reproduced (for example an exponential-race
  argmax), tolerance class otherwise. Logprobs: per-CTA (max, sum exp) partials folded in fixed order (online
  logsumexp) plus per-CTA top-N lists merged; logprob = logit - lse (tolerance class, f32 order).
- Speculative non-greedy: accept draft token x when u < p(x) / q(x), p from the target head and q from the draft head;
  at the first rejection, sample the residual normalize(max(0, p - q)) in kernel (a second pass over that row's
  logits). With N8 the proposal q has no mass outside the subset; rejection sampling stays exact (the residual equals
  p there).
- Gates: phase 1 bitwise inputs; phase 2 bitwise vs ATOM's sampler where reproduced, else a chi-square goodness of fit
  over >= 1e5 samples per fixed logits-row family (p > 0.01; a planted wrong temperature must fail); a logprob error
  bound set by P11 against ATOM's `log_softmax` in f32; acceptance length at T 0.7 / 1.0 within the A-vs-A spread
  (T3); GSM8K and HumanEval at a non-zero temperature over 3 seeds within the A-vs-A spread (T4).
