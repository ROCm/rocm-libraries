# DESIGN-LB: the large-batch true rocKE monokernel (rows 5-128, TP1 and TP2, gfx950, MXFP4)

Status: design, revision 1 (W3, 2026-10-30, user decision 4 of `PLAN.md`: CONC 8-128, TP1 and TP2, beat FlyTape's tiled
tape at the same batch / TP / precision and ATOM's native path at equal accuracy). Builds on `DESIGN.md` rev3 (the
small-batch true mono: stage emitters on `framework.py`, tagged NUMA mailboxes, accuracy-gated levers with bitwise
twins, the T0-T5 tolerance ladder) and on `DESIGN-CCL.md` (IPC peer mailboxes, in-kernel collectives). No source is
edited by this document. Ratios, shares and byte counts only; absolute times are in
`hackathon-prep/private-perf/qwen38-true-mono/` (`flytape-baseline.md`; new results in `large-batch/`).

**Revision 1.1 (W6 synthesis, 2026-11-01): section 15 "Measured budget (W4/W6)" re-issues the budget of sections 0.4
and 10 from the verified probes (BASELINE, L0c-DP2, CCL-LM, P-L1, P-L2, P-L3, P-L4, WP-C5), gives the go / no-go
and the dependency-ordered build plan. Where section 15 and an earlier section disagree, section 15 wins.**

Abbreviations as in `DESIGN.md` (`R`, `P`, `L`, `Q`, `F`, `AT` = `dev/ATOM-truemono/atom`, read-only for W3) plus
`FT` = FlyTape's measured device step at the same config (FT = 1), `LB` = this design, `B` = rows per step (= CONC at
K=0), `U` = active experts per layer, "floor" = per-rank bytes that must cross HBM per step.

---

## 0. Summary

1. **Large batch is a bandwidth problem with a synchronisation tax.** Per rank at TP2 B128, 0.97 of the bytes are
   routed-expert weights (about 470 of 512 experts active per layer at random routing) and the f32 GDN state (read and
   written every step); dense weights are 0.03 (MXFP4) to 0.10 (FlyTape's bf16). FlyTape runs at about 0.64 of its own
   roofline at TP2 B128 and about 0.43 at TP2 B16 (its measured steps against its per-rank bytes at the bandwidth its
   quoted per-rank roofline implies). The LB mono wins by (a) running the same bytes at a higher fraction of bandwidth and
   (b) fewer bytes where the checkpoint allows; it cannot win by MFMA throughput (compute is under 0.05 of the step).
2. **Structure: one stage list per layer kind, tiled emitters with the batch as the MFMA N dimension, tile-granular
   dataflow instead of phases.** 10 stages per GDN layer (FlyTape: about 16 phases), each edge polled at the
   granularity the consumer actually needs (token tile x column group, per head, per expert), every unit issuing its
   weights before it polls, exchanges pushed from producer epilogues (`DESIGN-CCL.md` 4.9). FlyTape's striped grid sync
   is a knob (`sync=grid`), not the default.
3. **Two comparison regimes, both reported.** *Precision-matched to FlyTape* (bf16 dense / mixers / router / head,
   MXFP4 W4A4 experts, f32 state, random weights, zero state, prompt 128 and 8K, its bench semantics): the floor ratio is
   1.0, so the whole win is efficiency. *Checkpoint precision* (ATOM's MXFP4 dense W4A8, MXFP4 W4A8 experts): fewer
   bytes, the precision ATOM serves, compared with ATOM's native path and, as a secondary line, with FlyTape.
4. **Budget (section 10; derived, anchored by WP-L0a / P-L1..P-L4 before any number is quoted as a result):**

   | cell | precision-matched / FT | checkpoint precision / FT | + bf16 GDN state (accuracy-gated, N11) |
   |---|---|---|---|
   | TP2 B128 p128 | **0.80 (0.73-0.89)** | **0.74 (0.67-0.83)** | 0.62 (0.57-0.70) |
   | TP2 B16 p128 | **0.72 (0.61-0.86)** | **0.56 (0.48-0.67)** | 0.51 (0.44-0.62) |
   | TP1 B128 (FlyTape unsupported) | - | step about 2.0x the TP2 LB step: per-GPU throughput about 1.35x FlyTape TP2 B128's | - |
   | ATOM native, CONC 8-128, TP1 / TP2 | unknown until WP-L0b | target <= 0.80 of its TPOT at equal accuracy | - |

   **Superseded by section 15.3** (measured): checkpoint precision, K0, TP2 B128 0.92 (0.83-1.03) of FlyTape's step,
   TP2 B16 0.83 (0.71-0.95); the only family above FlyTape is the MoE (P-L4: 1.05-1.29 of its expert + combine
   phases). The precision-matched (bf16) column is no longer a budget line (P-L3: the bf16 GEMM is slower than
   FlyTape's in-situ phase in 50 of 72 cells).

5. **The GDN recurrence is already near its bytes floor in FlyTape** (its recurrence moves the state at a bandwidth
   above its own roofline figure). MFMA or chunkwise forms do not cut bytes at K=0 (each row has its own state; a
   matrix-vector per row is all there is), so the LB recurrence is a pipelined VALU streamer in ATOM's [v, k] f32
   layout (gain: efficiency only, about 0.76 of FlyTape's GDN family at checkpoint precision). The real byte levers are
   bf16 state storage (N11, accuracy-gated on RULER) and, with MTP, one state read per request and `snapshot=last` (ATOM
   state contract D2). The chunkwise WY form is used where it pays: MTP verify rows (WP-L9).
6. **Work packages (section 12):** L0 anchors and probes (FlyTape traces, ATOM native baselines, bytes/bandwidth model,
   recurrence / sync / GEMM / MoE probes) -> L1 LB framework -> L2 emitters (GEMM core, HC, GDN, MoE, QSA, head + PLE) ->
   L3/L4 TP1 layers and pass -> L5 TP2 (needs CCL WP-C2/C3) -> L6/L7 ATOM-side serving and e2e + accuracy -> L8 levers,
   L9 MTP at LB, L10 RSI, L11 TP4.

---

## 1. Scope and cells

| cell | rows | TP | compared with | metric |
|---|---|---|---|---|
| LB-F (FlyTape-matched) | B in {8, 16, 32, 64, 128} (B5-7 run the 8 bucket), prompt 128 and 8192, 64 new | 1 (B <= 64; FlyTape refuses TP1 B128), 2 | FlyTape tiled tape, precision-matched and checkpoint arms | median device step (FlyTape's semantics: `step_ticks` gaps, first step excluded) |
| LB-A (ATOM serving) | CONC 8, 16, 32, 64, 128 at K=0; random 1k/1k and 8k/1k; GSM8K speed set | 1, 2 | ATOM's native path (what serves CONC > 4 today, `baseline/MATRIX.md`) | TPOT, output tok/s, per-request tok/s; accuracy (GSM8K, HumanEval, RULER) |
| LB-M (MTP at large batch, phase 2) | CONC c at K=1..3 with c(K+1) <= 128 | 1, 2 | ATOM native MTP at the same CONC | ms per verify round, acceptance, tok/s |

Rows 1-4 stay on the small-batch path (`DESIGN.md`); rows 5-16 are covered by both paths and the routing table picks the
winner per (rows, K, TP) (the narrow path's wide builds `r8..r16` vs the LB bucket 8 / 16).

---

## 2. What the baselines say

- **FlyTape measured on rocke** (`private-perf/.../flytape-baseline.md`, M0): TP2 mxfp4 B16 and B128 steps; TP1 B1-B4.
  No TP2 B128 phase trace yet (WP-L0a). Its README's family split at B128 TP2 (an older, slower build than ours):
  MoE 0.425, HC mixers 0.26, GDN 0.245, QSA 0.06 of the step (`F/models/qwen4/README.md`). TP1 B4 trace (ours): mixers
  0.34, MoE 0.31, GDN 0.20, QSA 0.10, head 0.03.
- **FlyTape structure** (`research/flytape.md` 2b): about 16 grid-synced phases per layer on the tiled tape; split-K
  GEMMs with last-arriver folds; expert-major MoE with per-CTA LDS counting sort, W4A4 scaled-fp4 MFMA, fused
  `moe_expert` only at TP2 for B 12-32 and 96-128; GDN recurrence one workgroup per (row, value head), f32 VALU, 3
  barriers per unit; row-gang QSA; tiled head with argmax slots; TP via 1 KB lines exchanged in the `gdn_out` /
  `qsa_out` / `moe_combine` epilogues and polled in the next `mix_combine`.
- **FlyTape's own remaining levers at B128** (README): MFMA QSA at long context, an accelerated replay recurrence (its
  16-token checkpoint-and-replay was exact but 2.5x slower), the mixer / exchange synchronisation floor.
- **Its precision is not ATOM's:** "mxfp4" = MXFP4 W4A4 experts only; dense projections, mixers, router, head bf16.
- **ATOM native at CONC > 4:** no baseline yet (WP-L0b). It uses aiter's custom all-reduce for TP
  (`AT/model_ops/communication_op.py`) and keeps the engram (n-gram) table in host memory with device hashing and UVA
  gathers (`AT/model_ops/engram/`).

---

## 3. Bytes per rank per decode step (p128; random routing, U = 512 (1 - (1 - 10/512)^B))

MXFP4 = 0.53125 B per weight (E8M0 scale per 32); router bf16 in both; KV is small at p128; partials and activations
excluded (they mostly live in MALL). "Dense" = HC mixers, GDN / QSA projections, router, shared expert, head, PLE proj.

| TP | B | U | experts (GB) | GDN state r+w (GB) | dense MXFP4 / bf16 (GB) | LB checkpoint total | FlyTape total | floor ratio | with bf16 state |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 74.7 | 9.37 | 1.81 | 2.37 / 8.23 | 13.6 | 19.4 | 0.70 | 0.65 |
| 1 | 16 | 138.6 | 17.4 | 3.62 | 2.37 / 8.23 | 23.4 | 29.3 | 0.80 | 0.74 |
| 1 | 32 | 239.6 | 30.0 | 7.25 | 2.37 / 8.23 | 39.8 | 45.6 | 0.87 | 0.79 |
| 1 | 64 | 367.1 | 46.0 | 14.5 | 2.37 / 8.23 | 63.1 | 69.0 | 0.92 | 0.81 |
| 1 | 128 | 471.0 | 59.0 | 29.0 | 2.37 / 8.23 | 90.9 | 96.8 | 0.94 | 0.79 |
| 2 | 8 | 74.7 | 4.68 | 0.91 | 1.43 / 4.86 | 7.0 | 10.5 | 0.67 | 0.63 |
| 2 | 16 | 138.6 | 8.68 | 1.81 | 1.43 / 4.86 | 12.0 | 15.4 | 0.78 | 0.72 |
| 2 | 32 | 239.6 | 15.0 | 3.62 | 1.43 / 4.86 | 20.1 | 23.6 | 0.85 | 0.78 |
| 2 | 64 | 367.1 | 23.0 | 7.25 | 1.43 / 4.86 | 31.8 | 35.2 | 0.90 | 0.80 |
| 2 | 128 | 471.0 | 29.5 | 14.5 | 1.43 / 4.86 | 45.7 | 49.1 | 0.93 | 0.78 |

- FlyTape's own estimate for TP2 B128 is about 48 GB per rank; this model gives 49.1. Real (checkpoint) routing is
  correlated, so U is lower for both stacks at equal rows; the matched workload uses random routing like FlyTape.
- At p8192 add about 12 x B x 2048 x 1 KB (TP2) of sparse-attended KV plus the indexer key scan (about 0.08 of the
  total at B128).
- The floor ratio is the checkpoint-precision advantage only. In the precision-matched regime it is 1.0.
- `WP-L0c` turns this table into a script (`hackathon-prep/qwen38/truemono-lb/bytes_model.py`) that every budget and
  every A/B report uses, with measured U from the router at each workload.

---

## 4. Architecture

### 4.1 Launch structure

- **Bring-up: `layers=1` LB layer kernels** graph-launched by the harness / ATOM-side driver (one kernel per layer +
  a head kernel). At B >= 32 a step is long against launch boundaries (48 boundaries are under 0.02 of the step,
  derived from `DESIGN.md` 1.1's boundary share scaled by the step length), so this is a usable product, the per-layer
  numerics gate, and the A/B unit.
- **Then `pass`** (48 layers + final mix + head + argmax + TP gather in one persistent launch, the dynamic layer loop of
  `DESIGN.md` 6.1): adds cross-layer overlap (layer l+1's HC down units start on the token tiles whose exchange lines
  have landed while layer l's combine is still finishing other tiles). Expected 0.02-0.05 of the step over
  `layers=1`; P-L2 and WP-L4 measure it.
- No on-device request loop (the step is long; `DESIGN.md` 2.5's `loop` value shrinks with step length).
- Grid: 256 CTAs x 512 threads, one CTA per CU (LDS padding), residency ticket with timeout (`DESIGN.md` WP1).
- **Rows buckets**: bodies are compiled per (layer kind, bucket in {8, 16, 32, 64, 128}, TP), with the live row count a
  runtime argument (token tiles = ceil(rows / 16); padded rows are masked: they route to no expert, skip the recurrence
  and the head, and their pushes go to the junk line). 3 kinds x 5 buckets x 2 TP = 30 layer bodies + heads; built as
  parallel alola CPU jobs.

### 4.2 Stage list (GDN layer; the QSA and PLE layers replace or add stages as noted)

Units are sized by the plan model of E1 (4.3); counts are for TP2 B128.

| # | stage | inputs (edge granularity) | work | output (published as) |
|---|---|---|---|---|
| S1 | HC-A combine + down | previous layer's `moe_y` exchange lines for this unit's stream columns and token tile (all ranks; rank-ordered sum); residual stream s | `r_s += s_s y` (redundant per row group; row group 0 writes `r_s`), stage `r_s (1+g_s)` bf16 / MXFP8 in LDS, partial down-projection of stream s (64 rows of 324) with the per-stream sum of squares; weights issued before the poll | f32 partials per (row group, stream, token tile); last arriver of the 4 streams folds: `sum_s rrms_s p_s`, inject gate, silu -> `down` bf16 tile + flag |
| S2 | HC-A up | `down` flags of its token tile (all 6 row groups) | up GEMM (10240 x 320) for 64 output rows x token tile; epilogue `x = sum_i sigmoid(acc_i) r_i rrms_i (1+g_i) / 4`, MXFP8 pack | `x` tile (bf16 + MXFP8) + flag per (column group, token tile) |
| S3 | GDN in-proj + conv + gates | `x` flags for its token tiles (full K) | GEMM 8240 x 2560 (TP2) with a fold if K-split; epilogue: depthwise conv (4 taps, conv state update), SiLU, `beta = sigmoid(b)`, `g = -exp(A_log) softplus(a + dt_bias)` | q / k / v / z / b / a rows + flags per (row group, token tile) |
| S4 | GDN recurrence | flags of the row groups holding head h's q, k (key head h / 3), v, z, b, a for its rows (partial edge) | per (row, value head): L2 norms, decay, `kv = S^T k`, `delta = beta (v - kv)`, `S += k delta^T`, `o = S^T q`, gated RMSNorm `o rrms w sigmoid(z)`; state read + write (ATOM's per-row state pointers, [v, k] f32) | `o` (bf16 / MXFP8) + flag per (head, token tile) |
| S5 | GDN out-proj | flags of the heads in its K slice for its token tile (partial edge) | GEMM 2560 x 3072 (TP2), K split by head groups, fixed-order last-arriver fold | TP1: `attn_y` tile + flag. TP2: the fold pushes bf16 lines to every rank (all-reduce, `DESIGN-CCL.md` 4.5) |
| S6 | HC-M combine + down | `attn_y` lines (all ranks) | as S1 with `gr_down[2 pos + 1]` | as S1 |
| S7 | HC-M up | as S2 | as S2; epilogue also writes the MoE input (MXFP8 for W4A8, MXFP4 for the W4A4 knob) | `x` tile + flag |
| S8 | router | `x` flags (full K) | GEMM 513 x 2560 (bf16 or the checkpoint's format), fold; the last arriver of a token tile's 9 row groups runs top-10, renormalised softmax, shared-expert gate | route (ids, weights) + flag per token tile |
| S9 | experts (expert-major, on-device sort) | all route flags (one all-to-all edge) | every CTA counting-sorts the B x 10 assignments in LDS (redundant, no hop); units = (expert, byte-line part) on a balanced byte line per IOD (`moe_fused` `map=line`), each unit loops over its expert's 16-token tiles so the weights are read once; gate_up -> silu x up -> requantize -> down inside the unit (TP2, I = 320: one CTA can hold the whole expert, FlyTape's `moe_expert`) or through an IOD-local intermediate mailbox (TP1, I = 640); shared expert as dense units over all rows | f32 partial per (row, slot) for its columns + a done flag per (expert, column group) |
| S10 | combine | done flags of the (<= 11 per row) experts of its token block and column group (partial edge) | `y = sum over slots in slot order w_j part_j` (+ shared) | TP1: `moe_y` + flag. TP2: bf16 lines pushed to every rank (all-reduce) |

- **QSA layer:** S3-S5 become Q-proj (13952 rows; TP2: 6144 q-gate + 256 KV + 640 indexer rows, the indexer
  replicated), Q-core (q/k norm, RoPE, paged KV append in ATOM's block-16 layout, indexer key compression, top-512 block
  select past the 2048-token budget, sparse attention: QK and PV on bf16 MFMA with the GQA heads as N, 12 real of 16 at
  TP2; row-gang over `gang` CTAs per row, gang by context as FlyTape's `gang_size`), Q-out (as S5).
- **PLE layer (zero-based layer 1):** adds the engram gather (ATOM's contract: device hashing + UVA gather from the
  host shard, `ngram=uva`; or the table in VRAM, `ngram=device`, for the FlyTape-matched arm), `ple_proj` (E1), stats,
  gate / dilated conv / residual (`DESIGN.md` P2 emitter at LB width).
- **Pass end:** final HC mix (S1/S2 shape with `gr_down[96]`), head (E10) with fused argmax, TP all-gather of (value,
  global id) pairs, ATOM's tie rule.
- Stage count per GDN layer 10 (FlyTape about 16), and every edge but S8 -> S9 is partial.

### 4.3 Unit plans (shared by every GEMM-shaped stage)

`Q/large_batch/plans.py`: a GEMM with M weight rows, K, and N = rows is cut into units (row group R in {16, 32, 64},
token tiles per unit T in {1..8}, K splits S) by a cost model: weight bytes from HBM (each row group read once per
unit set sharing an XCD; re-reads from L2 when token tiles split), partial bytes and fold cost, units vs 256 CTAs
(rounds and tail), LDS (activation chunks double-buffered at 128 K x 16 tokens per tile, FlyTape's padded stride), and
the consumer's edge granularity. Folds are fixed-order (split 0..S-1) by the last arriver of a counter (agent scope,
local) or by a reducer unit; never arrival-order sums. Example: the HC down-projection at B128 is 6 row groups x 4
streams (K split = streams, which makes the per-stream norm an exact re-association) x 8 token tiles = 192 units with a
4-partial fold; at B16 the streams split further so the grid fills. The planner is unit-tested against FlyTape's plans
(it must reproduce `MIX_SPLITS`, `OUT_SPLITS`, `GDN_SPLITS`, `QSA_SPLITS` when constrained to FlyTape's shapes) and
tuned by the intelligent search (`P/dispatch/tuning/search/`), never a grid.

### 4.4 Per-stage rules carried from the narrow mono

- Weights before the poll (every unit), NT on single-use streams (expert weights, state snapshot stores: `state_nt` was
  B3's largest lever), down weights after the gate_up epilogue (the fused MoE's spill lesson), no `b.sync()` (no vmcnt
  drain), `alwaysinline` + `convergent` stage bodies, fresh ids, ISA lint for spills and exec-join spills, scalar loads
  only of memory nothing writes in the launch.
- Holds become **issue-depth knobs per stage** (how many units ahead a CTA issues weights): at large batch contention
  is between streams of the same bandwidth-bound stage, which the Coz entry slopes find as they did at narrow width.

---

## 5. Sync model at large batch

| mechanism | where | why |
|---|---|---|
| **tile flags** (default): payload stored write-through (`sc1`), then a tagged flag per (stage instance, column group, token tile); consumers poll only the flags their unit needs (NUMA dual-copy flags, IOD-local poll) | every intra-layer edge and the cross-layer edge | partial edges (a token tile, a head, a K slice) start as soon as their inputs exist; the phase tails of FlyTape's grid syncs overlap instead of adding |
| publish variants (`publish=unit_end / comm_wave / counted`, `DESIGN-CCL.md` 4.2.1) | every flag | gfx950's shared `vmcnt` would otherwise drain the next unit's weight loads at each flag |
| **expert done flags** | S9 -> S10 | 470 x 40 flags per layer, no atomics, no reset; the counter form (`combine=count`) is a knob |
| last-arriver folds (local counters, fixed-order sums) | split-K folds, router top-k, HC fold | cheaper than a reducer stage when the fold reads few partials (FlyTape's choice); lost at narrow width as a hop mechanism (1.6-2.3x), but at LB it replaces a phase, not a hop. `fold=reducer` is the knob |
| exchanges (TP) | S5, S10 -> S1/S6 of the next layer; head gather | `DESIGN-CCL.md` 4.9: pushed from the fold / combine epilogue, polled with the tile flags, no extra phase |
| **striped grid sync** (`sync=grid`, FlyTape's 8 stripes per XCD, ballot poll) | knob for every stage boundary | A/B baseline and fallback; P-L2 decides per edge |
| residency ticket | launch start | non-resident grid times out instead of hanging |

**Deadlock freedom:** every CTA runs its units in stage order; edges point to strictly earlier stages; all CTAs
resident (on every rank). **WAR:** parity-double-buffered scratch per layer (`TaskTable.validate_stack`-style checker
over the LB task graph, extended with the CCL slot rule of `DESIGN-CCL.md` 4.9).

**Decision procedure (P-L2 then WP-L1):** a synthetic three-stage chain at LB shapes (GEMM -> fold -> GEMM, B 16 and
128) with idle and skewed arrivals: tile flags (each publish variant) vs striped grid sync vs per-expert flags; then the
same arms in situ in the first LB layer kernel (WP-L3). Default per edge = the in-situ winner under the WIN rule.

---

## 6. Stage emitters (`Q/large_batch/`)

| id | emitter | key design points | shares code with |
|---|---|---|---|
| E1 | tiled MX GEMM core | W as the A operand of `mfma_scale_f32_16x16x128_f8f6f4` (MXFP4 weights x MXFP8 activations; `act=fp4` knob for W4A4), batch tiles of 16 as N, 128-K activation chunks in LDS double-buffered, weights loaded straight to registers with depth `DEPTH` (knob), fixed-order folds, epilogue hooks (conv/gates, HC combine, MXFP8 pack, TP push, flags); `wdtype=bf16` mode (bf16 MFMA 16x16x32, FlyTape's) for the precision-matched arm | `L/kernels/gfx950/mx_mfma.py`, `P/core/arch/mfma_scale_mx.py`, `P/helpers/mx_quant.py` (MXFP8 rule), `mx_decode_gemv` (scale handling, korder) |
| E2 | HC LB (A1/A2/M1/M2 at LB) | S1/S2 above; per-stream K split makes `norm after GEMV` an exact re-association of the per-stream rrms; faithful-order twin = rrms first (FlyTape's TwoStage) | `DESIGN.md` WP2a (HC emitters, N1 lever), `hc_wide.py` semantics |
| E3 | GDN in-proj epilogue | conv state update in ATOM's `[3+k, QKV]` window, gates in f32, per-head flags | `stage_b3.py` gate math (`beta`, `decay`, f32 order) |
| E4 | GDN recurrence LB | section 6.1 | `stage_b3.py` (norms, gated RMSNorm, snapshot table, `state_nt`) |
| E5 | out-proj with head-sliced K | partial edges on S4 flags, fold, TP push | E1 |
| E6 | router LB | E1 + per-token-tile top-10 (wave per token, iterative wave argmax, renormalised softmax, shared gate, ATOM's tie order), bitwise-reproducible order | `moe_fused` router knobs, `topk_softmax.py` |
| E7 | MoE LB | section 6.2 | `moe_fused.py` (expert-major plan, `map=line`, NUMA, exchange in IOD), `moe.py` (union dedupe) |
| E8 | combine LB | slot-order sums, TP push; `combine=fused_next` knob (S1 reads the 11 slot partials directly) | `moe_fused` combine (rank-ordered, no atomics) |
| E9 | QSA LB | proj (E1), core row-gang, MFMA attention, `q3=mfma` indexer scores (tolerance, N10), radix top-512 with ties to the lower block | `DESIGN.md` WP2c (Q3-Q7, `qsa_wide.py` / `qsa_fast_wide.py` semantics), `stage_q2.py` (norm, RoPE, append) |
| E10 | head LB | E1 over vocab rows (TP: the rank's half) with a fused argmax epilogue (per-unit (value, global id) per token, LDS slots, wave-per-token merge, last-arriver merge per token tile), TP gather of pairs, `logits=out` and the logprob / sampling pieces of `DESIGN.md` WP8 | `mx_decode_gemv` (epilogues, tie rule, epoch counter), `DESIGN.md` P2 / WP2e / WP8 |
| E11 | PLE LB | engram gather (`ngram=uva` ATOM contract / `ngram=device`), `ple_proj` (E1), stats, gate / conv / apply | `DESIGN.md` WP2f |
| E12 | final mix, embed, prologue | final HC mix at LB rows (tolerance class above 8 rows, as `DESIGN.md` 5) | `DESIGN.md` WP2e |
| E13 | LB framework | buckets, runtime token tiles, masking, tile flags + publish variants, `sync=grid`, stamps, error words, dynamic layer loop, LB validate | `framework.py`, `DESIGN.md` WP1 (ticket, grid sync, `layers="dynamic"`, args table, ISA lint, stress and mutant harnesses) |

### 6.1 The GDN recurrence at large batch

Facts: per (row, value head) the state is 128 x 128 f32 = 64 KiB read and written per step; at TP2 B128 that is 14.5
GB per rank per step, 0.32 of the LB checkpoint bytes. The arithmetic (about 4 x 16K FMA per (row, head)) is under 0.01
of the step on the VALU. FlyTape's recurrence moves its state at a bandwidth above its own roofline figure, so the
recurrence is already close to the floor; FlyTape's GDN family (in-proj + recurrence + out-proj) runs at about 0.75 of
its bytes floor (its README family time vs its bytes at its roofline bandwidth).

Design (default, bitwise-deterministic, tolerance class vs ATOM native only through the summation order):
- **Thread map in ATOM's layout.** ATOM's state is [v, k] f32 per (row, head) behind a per-row pointer table (`conv_in,
  conv_out, rec_in, rec_out` i64, `stage_b3.py`). A wave owns 16 v-rows; 4 adjacent lanes split a v-row's 128 k
  values into interleaved 16-byte quads (each group of 4 lanes reads 64 contiguous bytes, a wave instruction 16 x 64
  B), so `kv[v] = sum_k S[v,k] k[k]` and `o[v] = sum_k S'[v,k] q[k]` are in-lane sums plus a 2-step DPP reduce over the
  4 lanes; no LDS block reductions and no barriers inside a unit (FlyTape needs 3 barriers and LDS sums per unit). 8
  waves cover 128 v-rows: one (row, head) per CTA per round, 32 state VGPRs per lane.
- **Pipelined across units:** the next unit's state (and q, k, v, gates) is issued before the current unit's math;
  depth 2-3 units in flight per CTA (64-96 VGPRs of state), so each CU keeps its share of the bytes in flight
  continuously (FlyTape's 4-wave CTA runs one unit at a time).
- **Stores:** the new state with `nt` (streaming; B3's `state_nt` lesson), snapshot slots per ATOM's MTP contract.
- **Partial edge:** a unit starts when its head's in-proj row groups for its row are flagged (S3 -> S4); the out-proj
  K slice for a head group starts when its heads are done (S4 -> S5).
- Knobs: `lanes_per_vrow` (2 / 4 / 8), `depth` (1-4), `state_nt`, `unit_order` (row-major / head-major / XCD-local),
  `state=bf16` (N11: bf16 storage, f32 math; halves the GDN bytes; accuracy-gated on RULER max and the long-context
  suite, `DESIGN.md` 12.3 N11), `fuse_out` (out-proj folded into the recurrence unit: each (head, token tile) unit
  multiplies its o by its W_o slice and publishes f32 partials; trades a hop for partial traffic; off by default at LB).
- **What MFMA buys:** nothing at K=0. With MTP verify rows (WP-L9) the k+1 rows of a request share one state: the
  chunkwise WY / UT form (`DESIGN.md` N5) turns their rank-1 updates into one small MFMA block, the state is read once
  per request, and with `snapshot=last` (ATOM state contract D2, a change W2 owns in ATOM-truemono) written once plus
  replay records: GDN bytes per request drop from (1 + (k+1)) to about 2 state sizes.
- P-L1 measures: FlyTape's `gdn_recur` at the TP2 B128 shape vs the LB recurrence variants vs a pure read+write
  streamer of the same bytes (the floor), in one job.

### 6.2 The MoE at large batch

- **Route:** S8's per-token-tile routers publish (ids, weights); every CTA reads all B x 10 assignments and builds the
  per-expert token lists by an LDS counting sort (FlyTape's `_build_lists`, deterministic by token id within an
  expert), so the only global edge of the MoE is S8 -> S9.
- **Units:** the experts' total bytes are dealt on a balanced byte line per IOD (`moe_fused` `map=line`): a unit is
  (expert, gate_up row block) or (expert, down column block) sized so that per-CTA bytes are equal within one unit;
  at TP2 (I = 320) a whole expert fits one CTA and the intermediate stays in LDS (requantized to MXFP8, or MXFP4 for
  W4A4); at TP1 (I = 640) the gate_up -> down hand-off goes through an IOD-local mailbox as in `moe_fused`. A unit loops
  over the expert's 16-token tiles so its weights are read once whatever the token count (at B128, U = 471 and 2.7
  tokens per expert on average: most experts are one partially filled tile; popular experts take several).
- **Shared expert:** dense units over all rows (E1), started during the router window (it does not depend on the route).
- **Partials:** f32 per (row, slot, column) (B x 11 x 2560 x 4 B = 14 MB per layer at B128, MALL-resident); the combine
  reads them in slot order. Knob `part=bf16` (tolerance) halves it.
- **Precision:** W4A8 default (ATOM's MXFP8 activations; same weight bytes, compute is not the limit), `act=fp4` (W4A4,
  FlyTape's) for the matched arm and as a knob.
- **Measure fixed overhead first** (`research/q38-learnings.md` 1.2: the narrow fused MoE's router window was 3x the
  design model): P-L4 builds a null-bytes MoE (all the routing, sort, flags and combine, no expert weights) at B16-128
  before any byte tuning.
- Levers (knobs): expert gap repack (FlyTape's `EXPERT_GAP`, needs a weight repack), union-aware placement for
  correlated routing, `combine=fused_next`, `part=bf16`.

---

## 7. Tensor parallelism

### 7.1 Sharding (TP2; one process per GPU, `DESIGN-CCL.md` 3.1)

| block | sharding | exchange |
|---|---|---|
| HC mixers, norms, residual streams | replicated | none (they consume the all-reduced `attn_y` / `moe_y`) |
| GDN | heads: 8 of 16 key heads, 24 of 48 value heads per rank; conv channels and the recurrent state follow the heads; in-proj column-parallel, out-proj row-parallel (K = the rank's heads) | all-reduce #1 from S5's fold |
| QSA | 12 of 24 q heads, 1 of 2 KV heads (KV cache per rank), indexer replicated (computed on every rank) | all-reduce #1 from Q-out |
| router | replicated | none |
| routed + shared experts | intermediate split (MXFP4: 320 + 320 exactly, no padding), gate/up column-parallel, down row-parallel | all-reduce #2 from S10 |
| PLE | replicated (`ple_proj`), engram table replicated in host memory (`ngram=uva`) or VRAM (`ngram=device`; `ngram=shard` splits the 16 hash heads across ranks and all-gathers the gathered rows) | PLE reduce of `y_mlp` (all-reduce) |
| head | vocab-parallel (124160 rows per rank) | all-gather of (value, global id) per row (+ cancel word); LSE / top-K' merges for logprobs and sampling |

Two all-reduces per layer (96 per step at 48 layers), each B x 2560 bf16 partials per rank (0.66 MB at B128): pushed
from the producing fold as tiles complete, so the link time (the exchange's bytes over one xGMI link) overlaps the
GEMM, and only the last tile's transfer plus one hop is exposed (`DESIGN-CCL.md` 6.3 measures it).

### 7.2 Bitwise and numerics under TP

Every rank computes bit-identical residuals (rank-ordered f32 sums), so routing, tokens and stop decisions agree
without extra exchanges. TP2 vs TP1 differ by the K split of S5 / S10 and the bf16 partial rounding: a drift budget item
of the tolerance ladder (section 9), with `ar_dtype=f32` as the twin used to measure it.

### 7.3 TP4 and above (stretch, WP-L11)

KV heads (2) do not divide 4: each KV head is replicated on two ranks (KV memory x2, q heads 6 per rank). MoE at TP8:
80 intermediate columns per rank do not divide MXFP4's 32-column groups; options are padding to 96 (wasted bytes) or a
TP2 x EP4 layout (experts split across 4 groups, which then needs an all-to-all or replicated activations plus a
masked all-reduce). Two-shot all-reduce and `hc=sp` (token-sharded HC residual: reduce-scatter instead of all-reduce,
the HC mixers run on the rank's rows only, all-gather of `x` before the column-parallel GEMMs) are the TP4+ levers.

---

## 8. Memory and index widths

### 8.1 Per-rank memory at B128, 8K context

| item | TP1 | TP2 | note |
|---|---|---|---|
| routed + shared experts (MXFP4) | about 64 GB | about 32 GB | 48 x 513 x 2.61 MB (TP1) |
| dense weights (MXFP4) | about 2.4 GB | about 1.4 GB | precision-matched arm: bf16, about 3.5x |
| engram table | 0 (`ngram=uva`, host) / about 51 GB (`ngram=device`) | same, or about 25.5 GB with `ngram=shard` | ATOM keeps it in host memory |
| GDN state (f32) | 14.5 GB | 7.25 GB | 36 x 128 rows x heads x 64 KiB; with MTP the pool is per request x (k+1) slots, the same total at equal rows; `state=bf16` halves |
| conv state | about 0.6 GB | about 0.3 GB | f32 window `[3+k, conv_dim]` |
| QSA KV (bf16, paged, block 16) | about 25 GB | about 13 GB | 2 (TP1) / 1 (TP2) KV heads x 256 x 2 x 2 B x 12 layers per token; at 32K x 128 rows TP1 needs about 100 GB, so ATOM's KV pool bounds concurrency at long context |
| indexer compressed keys | about 0.4 GB | about 0.4 GB | replicated |
| LB scratch (parity-double-buffered activations, partials, flags) + CCL arena | about 0.2 GB | about 0.2 GB | MoE partials 2 x 14 MB per layer parity |
| total (`ngram=uva`) | about 107 GB | about 55 GB | of 288 GB HBM |

### 8.2 Index widths (why FlyTape cannot run TP1 B128 and how LB does)

FlyTape indexes `gdn_state` with i32 element offsets (3.6e9 elements at TP1 B128). LB never forms a global element
index: the GDN state is reached through ATOM's per-row i64 state pointers (one SGPR pair per unit) plus an i32 offset
inside one (row, layer) state (3 MiB); expert weights through per-layer i64 bases plus i32 offsets inside a layer
(1.3 GB at TP1 < 4 GiB, so buffer resources with per-layer bases are valid); KV through ATOM's block table (i64 block
bases); the engram gather in 64-bit global addressing (320M rows x 160 B); scratch and mailboxes below 4 GiB. A CPU
test builds every TP1 B128 body and checks that no address computation overflows i32 (an IR lint over `mul`/`add`
chains feeding address operands, plus planted tests at the last row / last layer / last expert).

---

## 9. Numerics and accuracy

- References: the fp32 golden chain of `DESIGN.md` P11 (48 layers + head on real text), ATOM's native forward at the
  same rows and TP (the served reference), and FlyTape's oracle (`F/models/qwen4/reference.py`) for the matched arm.
- LB is tolerance class by construction (ATOM's native kernels at CONC > 4 are not the mono's; no bitwise reference
  exists there). Determinism is still mandatory: fixed-order folds, rank-ordered sums, no float atomics, 12 launches
  bitwise equal, race stress with delay injection, mutants that must fail (`DESIGN.md` 7.8 gates 5-6).
- Ladder (`DESIGN.md` 12.4, references swapped): T0 per stage vs golden (<= 1.25x ATOM native's own stage drift); T1
  layer drift at rows 8 / 32 / 128 and QSA at L 639 / 8191; T2 teacher-forced KL / top-1 agreement within the P11
  budget; T4 GSM8K (1,319, 5-shot), HumanEval (164), RULER 8K / 32K / 128K at CONC 32 and 128 inside ATOM native's
  A-vs-A spread; T5 per final config. TP2: ranks bitwise equal, TP2-vs-TP1 drift inside the budget.
- Drift budget items specific to LB: bf16 all-reduce partials, per-stream norm re-association (exact in real numbers,
  different rounding), W4A4 (knob), `part=bf16` (knob), `state=bf16` (N11, the riskiest: recurrent error over long
  contexts; gated on RULER max), `q3=mfma`.

---

## 10. Quantified budget vs FlyTape

(Pre-measurement budget, kept for the record. The measured budget is section 15.)

### 10.1 Method

`LB step / FT step = (LB floor / FT floor) x (eta_FT / eta_LB)`, where eta = floor / (step x beta) and beta is the
per-rank bandwidth FlyTape's own roofline claim implies (48 GB per rank in its quoted roofline time). With the M0 measurements
(private), eta_FT = 0.64 at TP2 B128 and 0.43 at TP2 B16 (the two configs measured). Using beta from FlyTape's claim is
conservative: its own recurrence exceeds it, so eta_LB values near 0.9 are reachable for the streaming families.

### 10.2 Family view at TP2 B128 (FlyTape's README split, older build; WP-L0a re-anchors on the current build)

| family | FlyTape share | FlyTape vs its floor | LB mechanism | LB / FlyTape family (checkpoint) | (precision-matched) |
|---|---|---|---|---|---|
| MoE (router, sort, experts, shared, combine) | 0.425 | about 0.75 of floor | same expert bytes; balanced byte lines, weights before poll, partial combine edges, no phase between up / down / combine | 0.85 (0.78-0.92) | 0.85 |
| HC mixers (2 per layer, incl. combine and exchange waits) | 0.26 | about 0.05 of floor (sync / latency bound) | 2 stages per mixer instead of 4 phases, per-token-tile edges, exact re-association removes the norm hop, MXFP4 weights | 0.35 (0.25-0.5) | 0.4 |
| GDN (in-proj, recurrence, out-proj) | 0.245 | about 0.75 of floor | pipelined barrier-free recurrence, MXFP4 projections, head-granular edges | 0.76 (0.68-0.85); 0.40 with `state=bf16` | 0.82 |
| QSA (p128) | 0.06 | about 0.1 of floor | MXFP4 proj, partial edges, MFMA core | 0.5 (0.35-0.7) | 0.6 |
| head, PLE, rest | 0.01 | - | in-kernel head + gather | 1.0 | 1.0 |
| **step (sum)** | 1 | 0.54 (0.64 for the current build) | | **0.68** of the README build | **0.71** |

FlyTape's step has since fallen to about 0.84 of the README build (token blocking, XCD slots, up-weight warming and
row-gang QSA, all on mixers and QSA). If those gains came from the families LB already cuts most, LB / FT rises to
about 0.81 (checkpoint) and 0.85 (matched); if they were proportional, it stays at 0.68 / 0.71. The headline numbers of
section 0 (0.74 and 0.80 central) sit inside both brackets and match 10.1's eta model (eta_LB = 0.80 central, 0.72-0.88).

### 10.3 Other cells (derived; WP-L0a anchors B32 / B64 and TP1)

- **TP2 B16:** floor ratio 0.78 (matched 1.0); FlyTape at eta 0.43 is sync-bound here; LB eta 0.60 (0.50-0.70) ->
  0.56 (0.48-0.67) checkpoint, 0.72 (0.61-0.86) matched. The narrow mono ran at 2.3x its issued-bytes floor at rows
  1-4, so eta 0.6 at rows 16 assumes the per-stage bytes grow (U = 139 experts) while the hop count stays; P-L2 and
  WP-L3 test it first.
- **TP2 B32 / B64:** floor ratios 0.85 / 0.90; expected between the B16 and B128 lines.
- **TP1 B8-B64 vs FlyTape TP1:** floor ratios 0.70-0.92; FlyTape TP1 traces (WP-L0a) give eta_FT.
- **TP1 B128:** floor 90.9 GB (1.99x the TP2 per-rank floor); at equal eta the step is about 2.0x the LB TP2 step, about
  1.47 FT(TP2 B128), i.e. per-GPU throughput about 1.36x FlyTape TP2's. Against ATOM native TP1: unknown (WP-L0b).
- **p8192:** FlyTape's QSA attention grows to a several-fold larger share at 8K (README); LB's MFMA row-gang core and
  `q3=mfma` take the same share; budget re-issued after WP-L0a's 8K trace.
- **ATOM native (LB-A):** no data. Target <= 0.80 of its TPOT at equal accuracy at CONC 32-128 (it pays per-layer
  launches, separate all-reduce kernels and per-op activations at these sizes); WP-L0b sets the real bar.

### 10.4 What would make LB lose

MoE efficiency not above FlyTape's (equal bytes, so the MoE family alone caps the gain at about 0.6 of the step);
the recurrence not pipelining beyond FlyTape's (GDN family parity); tile-flag publish bubbles larger than the phase tails
they replace (P-L2); TP exchange exposed per tile rather than per layer (WP-C5). Each has a probe before the first
real-layer port, and `sync=grid` keeps FlyTape's structure available inside rocKE as the fallback arm.

---

## 11. Code shared with the small-batch path

| shared piece | small-batch owner (`DESIGN.md`) | LB use |
|---|---|---|
| `framework.py` StageSpec / TaskTable, stamps, LDS arena, phase barriers without vmcnt drain | existing | E13 builds on it (task graph with partial edges) |
| residency ticket, striped grid sync, `layers="dynamic"`, args table, ISA lint, delay-injection and mutant harnesses | WP1 | E13; **whichever workflow needs it first lands it once in `Q/` (coordination point with W2; no duplicate module)** |
| tagged NUMA mailboxes (`rocke.helpers.mailbox`, `mailbox_ir`), IOD calibration | existing | tile flags, CCL flags |
| scaled MFMA, MXFP8 pack rule, MXFP4 scale handling | `mx_mfma.py`, `mx_quant.py`, `mx_decode_gemv.py` | E1 |
| B3 gate math, norms, gated RMSNorm, snapshot table, `state_nt` | `stage_b3.py` | E3, E4 |
| expert-major fused MoE plan (`map=line`, NUMA, IOD-local exchange, rank-ordered combine) | `moe_fused.py` | E7, E8 |
| Q2 norm / RoPE / append, Q3-Q7 (indexer, radix top-512, sparse attention) | `stage_q2.py`, WP2c | E9 |
| head with fused argmax, `logits=out`, sampling / logprobs | `mx_decode_gemv.py`, P2, WP2e, WP8 | E10 |
| PLE, final mix, prologue | WP2f, WP2e | E11, E12 |
| CCL (peer mailboxes, collectives) | `DESIGN-CCL.md` | TP2 everywhere; the narrow path's TP2 uses the same `put` with peer bases |
| tolerance ladder, P11 golden chain and drift harness | rev3 12.4, P11 | section 9 |

LB-only: the tiled GEMM core and planner (E1), the LB recurrence (E4), LB MoE units with token-tile loops (E7), row-gang
QSA at LB, the LB head over N = rows, the rows buckets.

---

## 12. Work packages

(Revision 1 plan. Group L0 is done; the build order from here is section 15.6, which re-orders L2 around the MoE
rebuild and promotes WP-L9 (MTP) onto the critical path.)

Cost: G = GPU-hours (multi-GPU jobs count every GPU), C = CPU core-hours. Cluster rules as `DESIGN.md` 8.1 and the W3
rules: `q38_job.sh` (`--gpus` <= 7, `Q38_MAX_GPUS=10`), stages `w3-*`, at most 4 jobs queued, idle-GPU check for any
persistent or multi-GPU job (FlyTape and persistent kernels hang on a shared GPU), container mode for FlyTape / ATOM /
FlyDSL, venv mode for pure rocKE + torch; heavy I/O in node-local `/tmp`; CPU builds fan out as alola CPU jobs; job logs
read under `~/rocke-sdpa-claude/q38/logs/` when `sacct` is unreachable.

**Group L0 (now; parallel; probes are rocKE-only microkernels unless noted)**

| WP | title | gate | cost |
|---|---|---|---|
| WP-L0a | FlyTape anchors: `ft_bench_tp1.sh` with `TP=2`, `CONFIGS` mxfp4 B8/16/32/64/128 x p128 / p8192 and `TRACE_CFG` at B16 and B128 (phase families of the current build); TP1 B8-B64 + traces; reps rotated, idle GPU pairs; the matched-workload spec written down (weights seed, router, zero state, step semantics) | A-vs-A spread per config; node + GPU ids per row | 8 G |
| WP-L0b | ATOM native baselines (container, `dev/ATOM` staged read-only): TP1 and TP2 at CONC 8 / 16 / 32 / 64 / 128, K=0 (and K=3 for LB-M), random 1k/1k and 8k/1k, GSM8K speed set; accuracy A-vs-A (GSM8K, HumanEval at CONC 32 / 128, RULER 8K / 32K); the all-reduce it uses | A-vs-A spread per cell | 24 G |
| WP-L0c | bytes model script (section 3) + bandwidth curve on gfx950 at 256 x 512 (read stream, 1:1 read+write stream like the state, MXFP4 tile pattern), measured U per workload | model reproduces FlyTape's per-rank 48 GB claim within 3% | 1 G, 2 C |
| P-L1 | recurrence probe (6.1): FlyTape `gdn_recur` shape vs LB variants vs the pure streamer, TP2 B128 and TP1 B64 shapes, `state=bf16` arm | T0 vs f32 CPU reference; every arm deterministic | 2 G, 10 C |
| P-L2 | sync probe (5): tile flags x publish variants vs striped grid sync vs expert flags, B16 / B128, idle and skewed arrivals; uses W2's P4 primitives where they exist | negative controls (dropped arrival blocks, oversize grid times out) | 1.5 G, 10 C |
| P-L3 | GEMM core probe: E1 at the gdn_in, out-proj, HC down / up and head shapes, B 8-128, `wdtype=mx` and `bf16`, vs bytes floor and vs hipBLASLt / aiter at the same shapes | T0 vs fp32 reference | 2 G, 20 C |
| P-L4 | MoE probe: null-bytes LB MoE (fixed overhead first), then W4A8 / W4A4 units, B16 / B64 / B128, random and correlated routing, vs FlyTape's MoE phases from WP-L0a | T0; slot-order combine deterministic | 2 G, 20 C |

**Group L1 (needs P-L2; CCL WP-C1/C2 run alongside per `DESIGN-CCL.md` 8)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP-L1 | LB framework (E13): buckets, runtime tiles, masking, task graph with partial edges, tile flags + publish variants, `sync=grid`, ticket, dynamic loop (shared with `DESIGN.md` WP1), error words, stamps, LB validate (WAR, tags, CCL slots), ISA lint, index-width lint | P-L2 | CPU tests; GPU microbench equals P-L2; negative controls; byte-identity tests unchanged | 2 G, 40 C |

**Group L2 (needs WP-L1; at most 5 in parallel; each gated T0 + determinism + race stress, then isolated A/B vs
FlyTape's phase equivalent (matched precision) and vs the bytes floor)**

| WP | title | depends | cost |
|---|---|---|---|
| WP-L2a | E1 GEMM core + planner + folds + `wdtype=bf16` + `act=fp4` | P-L3 | 3 G, 40 C |
| WP-L2b | E2 HC LB (faithful-order twin + re-association) | WP-L2a | 2 G, 20 C |
| WP-L2c | E3 / E4 / E5 GDN (recurrence per P-L1) | WP-L2a, P-L1 | 3 G, 30 C |
| WP-L2d | E6 / E7 / E8 MoE LB | WP-L2a, P-L4 | 5 G, 50 C |
| WP-L2e | E9 QSA LB (p128 and 8K, row-gang, MFMA core, select) | WP-L2a | 4 G, 40 C |
| WP-L2f | E10 / E11 / E12 head, PLE, final mix, embed | WP-L2a | 2 G, 20 C |

**Group L3 (TP1)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP-L3 | `layers=1` LB layer kernels (gdn, gdn_ple, qsa) per bucket; in-situ sync decision per edge (P-L2 arms) | Group L2 | T1; layer A/B vs FlyTape TP1 phase sums (matched) at B8-B64; WIN rule | 6 G |
| WP-L4 | TP1 `pass` (48 layers + head) and `layers=1` + head graph | WP-L3 | T2 (teacher-forced); matched A/B vs FlyTape TP1 B8-B64; TP1 B128 runs (FlyTape cannot) | 8 G |

**Group L4 (TP2 and serving)**

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP-L5 | TP2: rank-local sharded weights (host, from ATOM's checkpoint layout and from the matched random weights), CCL all-reduce in S5 / S10 folds and Q-out, head all-gather, KV head split, PLE reduce | WP-L4, CCL WP-C2 + WP-C3 (+ WP-C5 before budgets are quoted) | ranks bitwise equal; TP2-vs-TP1 drift in budget; **headline A/B vs FlyTape TP2 B8-B128 (matched and checkpoint), p128 and p8192** | 12 G |
| WP-L6 | ATOM-side serving experiment in `hackathon-prep/qwen38/truemono-lb/` (a runner overlay on a staged copy of ATOM-truemono; W2 owns and merges the real hook): CONC 8-128 per-step launch, bucket changes as requests join / leave, prefill entry, engram UVA contract, refusal fallback to ATOM native, graph capture | WP-L4 (TP1), WP-L5 (TP2), CCL WP-C7 | Tier-G-style chains vs ATOM native outputs within tolerance; transitions | 6 G |
| WP-L7 | e2e LB-A cells vs ATOM native (TP1, TP2) and accuracy T3-T5 | WP-L6 | WIN rule per cell (6 pairs, shuffled, A-vs-A); suites inside ATOM native's spread | 24 G |

**Group L5 (continuous after the gates they depend on)**

| WP | title | depends | cost |
|---|---|---|---|
| WP-L8 | levers, each a default-off knob with its own gate: `state=bf16` (N11), W4A4 default question, expert gap repack, union-aware placement, `ngram=shard`, `part=bf16`, `combine=fused_next`, `fuse_out`, `publish=counted` (+ Rust direct pilot if WP-R1 wins), two-shot all-reduce, `hc=sp` | WP-L5 | 10 G, 60 C |
| WP-L9 | MTP at large batch (LB-M): rows c(K+1) <= 128, chunkwise WY recurrence for verify rows, `snapshot=last` (needs ATOM contract D2 from W2), round structure from `DESIGN.md` WP5/WP6 at LB width, vs ATOM native MTP | WP-L7, `DESIGN.md` WP5c | 12 G |
| WP-L10 | RSI: WaveScope per stage, stamps timeline, Coz entry/post slopes per stage and per issue depth, hill climbing with lookback and exhaustive combination sweeps of retained winners, intelligent search for plans | WP-L3 onward | ongoing |
| WP-L11 | TP4 (KV-head replication, two-shot, `hc=sp`) when 4 idle GPUs exist | WP-L5, CCL WP-C4 | 8 G |

**Critical path:** WP-L0a + P-L1..P-L4 -> WP-L1 -> WP-L2a -> WP-L2b/c/d/e/f -> WP-L3 -> WP-L4 -> WP-L5 (needs CCL WP-C2/C3)
-> WP-L6 -> WP-L7. WP-L0b runs alongside and must report before any LB-A budget is quoted.
**Total to WP-L7:** about 120 G and 300 C (WP-L8..L11 about 30 G more, RSI ongoing). CCL adds about 27 G (`DESIGN-CCL.md` 8).

**CCL pieces each WP needs:** WP-L5 needs one-shot all-reduce (lines, bf16 partials, rank-ordered sum), the argmax
all-gather, epoch / header / timeouts, the WAR checker over the LB task graph (WP-C2, WP-C3) and the in-situ defaults
(WP-C5); WP-L2f's TP head needs the merges (WP-C2); WP-L6 needs the host runtime inside ATOM's process group and the
error reporting (WP-C1, WP-C7); WP-L8's two-shot and `hc=sp` need reduce-scatter (WP-C2) and the crossover table
(WP-C4); WP-L11 needs WP-C4 at TP4.

---

## 13. Measurement protocol (LB specifics on top of `DESIGN.md` 9.1)

- **Matched-workload A/B vs FlyTape:** one container job per (TP, config set) on an idle GPU pair: FlyTape arm
  (`bench_qwen4_full.py`, its semantics) and LB arms (a driver with the same seeds, router spread, zero state, prompt,
  64 new tokens, device step stamps, first step excluded) alternate in a seeded shuffled order per round (fresh
  process per arm-run, as FlyTape's `bench_ab.py`), 8+ rounds (FlyTape documents process means varying about 0.6% and
  GPU pairs about 1%). Controls: a second LB arm built from the same source but recompiled, and FlyTape A-vs-A.
- **Numerics before timing in the same job:** the LB arm's T0 / T1 checks on the matched weights against the CPU / fp32
  reference, ranks bitwise equal at TP2; FlyTape's tokens deterministic across reps.
- **Every row records:** node, GPU bus ids and xGMI pair, SCLK samples, measured U, rows bucket, knobs.
- **Ratios here; absolutes** in `private-perf/qwen38-true-mono/large-batch/`.
- E2E (LB-A) follows `baseline/MATRIX.md`'s harness (`matrix.sh` arms, `PRIME=1`, warmup per cell), with the LB arm as
  one more entry, and the e2e WIN rule of `DESIGN.md` 9.1.

---

## 14. Risks

| risk | effect | mitigation / probe |
|---|---|---|
| MoE efficiency gain small (equal bytes; FlyTape near 0.75 of floor) | the largest family barely moves; LB/FT about 0.85 at B128 matched | P-L4 null-bytes first; byte-line balance; weights before poll; union-aware placement for real routing; honest reporting of the matched ratio |
| recurrence pipelining does not beat FlyTape's | GDN family at parity | P-L1 before WP-L2c; `state=bf16` behind the accuracy gate; MTP `snapshot=last` |
| tile-flag publish bubbles | sync tax not reduced | publish variants; `sync=grid` per edge where it wins |
| eta at B16 lower than assumed (hop-bound like the narrow mono) | B16 ratio near FlyTape's | narrow path's wide builds (`r16`) compete for rows 9-16 in the routing table |
| precision-matched arm needs bf16-weight GEMMs (`wdtype=bf16`) | extra emitter work | E1 mode, not a separate emitter; built and gated once |
| ATOM-side integration owned by W2 | serving cells late | experiments in `hackathon-prep/qwen38/truemono-lb/`; hand-off notes to W2 |
| memory at TP1 B128 long context | concurrency capped by KV | ATOM's pool decides; LB refuses buckets that do not fit and falls back to ATOM native, logged |
| idle GPU pairs scarce on the rocke partition | TP2 timing waits | jobs check idleness, use the most idle pair, never `--exclusive`; TP1 work proceeds meanwhile |
| compile time of 30 LB bodies x knob sweeps | slow RSI | alola CPU fan-out, job-local comgr cache, fresh process for iglp builds; Rust pilot if it wins |

---

## 15. Measured budget (W4/W6)

W6 synthesis, 2026-11-01. Inputs: every verified large-batch result in `large-batch/` with its Verification and Fix
rounds: `BASELINE.md` and `L0c-DP2.md` (the bars, bytes model, floors), `CCL-LM.md` (standalone all-reduce),
`P-L1.md` (recurrence), `P-L2.md` (sync model), `P-L3.md` (dense GEMM core, fold-matched fix round), `P-L4.md` (MoE,
fix round), `WP-C5.md` (in-situ exchange, fix round); `wave1/WP1.md` for the platform. Ratios only. "Measured" = a
probe number under the DESIGN 9.1 rule on both rocke nodes; "derived" = arithmetic on measured numbers with the
assumptions stated next to it. No new hardware run was made for this section.

### 15.1 The bar per cell (PLAN decisions 4 and 5)

Each value is the per-cell reference over FlyTape's step at the same TP and B (FT = 1), so an LB step below it beats
the reference. TP1: the reference is ATOM native K3 (BASELINE: faster than FlyTape TP1 in every cell). Two GPUs: the
better of FlyTape TP2 and ATOM DP2 K3 (L0c-DP2 fix round; FlyTape DP2 and ATOM TP2 are never the bar). FlyTape cells
move by up to 0.028 between jobs: read these as +-0.015.

| TP | B | bar, 1k (ref / FT step) | bar, 8k | reference arm |
|---|---|---|---|---|
| 1 | 8 | 0.912 | 0.852 | ATOM K3 |
| 1 | 16 | 0.914 | 0.923 | ATOM K3 |
| 1 | 32 | 0.955 | 0.951 | ATOM K3 |
| 1 | 64 | 0.987 | 0.979 | ATOM K3 (FlyTape within 2.1 %) |
| 1 | 128 | no FlyTape; ATOM K3 = 0.754 of ATOM K0 | 0.765 (K1; K3 within 0.7 %) | ATOM K3 / K1 |
| 2 | 8 | 0.917 | 0.890 | ATOM DP2 K3 |
| 2 | 16 | 0.974 | 0.962 | ATOM DP2 K3 (1k: within 3.1 %) |
| 2 | 32 | 1.000 | 1.000 | FlyTape TP2 (ATOM DP2 K3 within 3.2 % / 0.7 %) |
| 2 | 64 | 0.993 | 0.997 | ATOM DP2 K3 (within 1.5 % / 0.8 %) |
| 2 | 128 | 1.000 | 1.000 | FlyTape TP2 (ATOM DP2 K3 1.107-1.158x it) |

- These bars are cross-precision and cross-metric (BASELINE caveats): FlyTape is a K0 device step on random weights
  with W4A4 experts and bf16 dense; ATOM is a served MTP per-token time on real weights at W4A8. LB runs ATOM's
  numerics (W4A8 checkpoint), so the LB step below is the "checkpoint precision" line.
- **TP1 means beating an MTP arm.** Every TP1 bar is below 1: a K0 LB step must be 0.85-0.99 of FlyTape's to beat
  ATOM K3 (and 0.75-0.77 of ATOM's own K0 step at B128). Section 15.3 shows K0 alone gets there only with a thin
  margin; MTP in the LB path (WP-L9) is what gives TP1 a robust margin.

### 15.2 What each stage measured (LB / FlyTape unless stated; checkpoint precision, mx = MXFP4 W4A8)

| stage (DESIGN-LB 4.2) | FlyTape share of step (TP2 B128 / TP2 B32 / TP1 B32) | measured | source | state |
|---|---|---|---|---|
| S9 + S10 experts + combine | about 0.39 / 0.37 / 0.46 (derived: FT MoE family + `moe_combine`, minus router) | best arm, same-job bracket: **1.05-1.29 (gs) / 1.02-1.22 (launch)**; W4A8 TP1 1.06-1.14, TP2 1.07-1.29 (worst TP2 r32, where FlyTape fuses `moe_expert`); 1.51-2.30x the bytes floor; null-bytes build alone 0.47-0.77 of the full kernel; correlated routing widens the floor gap (2.08-3.17x at strong skew) | P-L4 F4, 5.1-5.2, F3 | **at risk: the only family above FlyTape** |
| S8 router GEMM | about 0.05 / 0.09 / 0.06 | mx 0.53-0.60 of FlyTape's in-situ phase (stream 0.18-0.30); top-k not yet fused | P-L3 6 | on track |
| S4 GDN recurrence | 0.178 / 0.100 / 0.120 | f32: in situ 0.84-0.94 (best launch), 0.50-0.90 (stream); steady state at the floor for both stacks (LB 1.025-1.046, FlyTape algorithm 1.033-1.085): nothing left at f32. `state=bf16`: 0.51-0.65 in situ (launch), at its own floor, accuracy-gated (N11) | P-L1 4.1, 4.3 | on track |
| S3 GDN in-proj | about 0.07 / 0.07 / 0.10 | mx in situ: TP1 0.34-0.43 phase (0.20-0.25 stream), TP2 0.59-0.86 (0.35-0.44); conv / gates epilogue not built yet | P-L3 6 | on track |
| S5 / Q-out out-proj | about 0.03 / 0.04 / 0.05 | mx in situ: TP1 0.78-0.97 phase (0.41-0.53 stream), **TP2 1.39-2.03 phase (0.69-1.36 stream)**; 0.06-0.22 of its bytes floor (latency-bound); bf16 at parity with FlyTape's split-6 plan | P-L3 6 | at risk at TP2 |
| S1 / S2 / S6 / S7 HC down / up | 0.109 / 0.136 / 0.086 (+ reduce / combine phases) | mx in situ: down 1.22-1.30 phase (0.53-1.03 stream), up 1.06-1.34 (0.30-0.51); 0.03-0.19 of the floor (0.7-7 MB per layer, latency-bound) | P-L3 6 | at risk (phase form) |
| QSA (proj, core, out) | 0.041 / 0.055 / 0.064 | `qsa_in` mx 0.48-0.60 (TP1) / 0.79-0.95 (TP2) phase; **core not measured** | P-L3 6 | unknown |
| E10 head | 0.009 / 0.015 / 0.019 | mx 0.32-0.45 phase | P-L3 6 | on track |
| TP exchanges (2 per layer, S5 and S10) | FlyTape's own: 0.060-0.082 (skew floor) to 0.078-0.107 (old proxy) of its TP2 step | in situ, both measured: **96 exchanges = 0.048-0.061 of FlyTape's TP2 step**; out-proj exchange 0.22-0.39 of FlyTape's per-exchange cost at B8-B64, 0.58-0.77 at B128 (link-bound); MoE-down exchange 0.84-4.3x the out-proj one | WP-C5 F2, F4, F5 | on track for the budget; DESIGN-CCL's 0.25 per-exchange gate not met (DESIGN-CCL 10) |
| stage-to-stage sync | non-work share of FlyTape's layer 0.13-0.21 | `sync=mix` (grid on >= 0.4 edges, per-wave lazy flags) 0.979-0.983 of grid at idle, 0.927 with one late unit; recovers only 0.007-0.034 of a FlyTape layer; the rest is phase-tail imbalance. Sync share of the layer: TP1 LB plan 0.04-0.11, FlyTape plan 0.10-0.22, **TP2 rank shape about 0.2 whatever the plan** | P-L2 3-4 | at risk at TP2 |
| persistent walk vs one launch per layer | - | MoE: the persistent walk costs 1.04-1.08 of per-layer launches (P-L4 gs vs launch; WP-C5 F4); recurrence: one launch per layer costs 1.01-1.40x the floor vs 1.025-1.046 streamed | P-L4, WP-C5 F4, P-L1 | open (decided in WP-L4) |

Precision-matched regime (bf16 dense, FlyTape's): the LB bf16 GEMM wins 39, ties 13, mixed 3 of 55 cells against
FlyTape's plan in the same harness (fold-matched), but is slower than FlyTape's measured in-situ phase in 50 of 72
cells (phase metric) and trails hipBLASLt per launch in 36 of 55 (P-L3 summary items 3, 4 and 8; section 6). **The precision-matched line is
dropped from the budget**: it stays a reported arm (`wdtype=bf16`), not a target. The large-batch case rests on the
mx path at ATOM's numerics (55 of 55 over FlyTape's plan; 0.07-0.44 of ATOM's own W4A8 linear).

### 15.3 Derived step budget per cell (K0, checkpoint precision, p128, random routing)

Method: LB step / FT step = sum over families of (FlyTape share) x (LB / FlyTape ratio of 15.2).
- Shares: BASELINE's traces at TP2 B128, TP2 B32 and TP1 B32; P-L1's measured recurrence share per cell; TP2 B16
  uses the B32 shares, TP2 B64 their mean with B128, TP1 B16 / B64 the TP1 B32 shares. `moe_combine` is taken as
  0.035 (TP2) / 0.020 (TP1) of the step. The router / experts split per cell comes from P-L4 F4 (best / FT
  expert + combine over best / FT whole MoE).
- Ratios: the low end uses the launch metric for the MoE, the stream metric for the recurrence and GEMMs, and the
  high end the gs / phase metrics. Projections are split 2 : 1 between in-proj and out-proj. The QSA family is
  taken at 0.6-1.2, because its core is unmeasured. Split-K folds sit inside the complete-output GEMM times (P-L3 fix
  round), so FlyTape's reduce phases count as 0 for LB plus 0-0.02 at TP1. At TP2, WP-C5 F5's measured 96 exchanges
  are added per cell.
- Not included: cross-layer overlap of a `pass` (DESIGN-LB 4.1: 0.02-0.05 expected), WP-L8 levers, and the conv /
  gates and top-k epilogues (inside the ranges' slack only).

Decomposition at TP2 B128 (shares of FlyTape's step):

| family | FlyTape | LB (range) |
|---|---|---|
| MoE: router + experts + combine | 0.444 | 0.45-0.49 (router 0.01-0.03, experts + combine 0.44-0.46) |
| GDN: in-proj + recurrence + out-proj | 0.276 | 0.22-0.27 |
| HC mixers + reduce / combine phases (incl. FlyTape's exchange waits) | 0.225 | 0.12-0.20 (HC GEMMs 0.06-0.13, exchanges 0.057-0.061) |
| QSA | 0.041 | 0.025-0.049 |
| head, other | 0.014 | 0.008-0.009 |
| **step** | **1** | **0.82-1.01, central 0.91** |

Per cell (central (range); "fix MoE" = WP-L2d's rebuilt MoE at 0.85 of FlyTape's expert + combine, the P-L4 section 7
estimate of its removable share; "+ bf16 state" adds N11 on top, accuracy-gated):

| TP | B | bar (1k / 8k) | LB K0 / FT, measured families | MoE share of the LB step | state K0 | + fix MoE | + fix MoE + bf16 state |
|---|---|---|---|---|---|---|---|
| 2 | 8 | 0.917 / 0.890 | not derivable: no MoE at rows 8 (P-L4 ran 16-128) | - | **open, at risk** | - | - |
| 2 | 16 | 0.974 / 0.962 | 0.82 (0.69-0.94) | 0.51 | on track | 0.74 (0.62-0.86) | 0.72 |
| 2 | 32 | 1.000 / 1.000 | 0.91 (0.79-1.03) | 0.55 | **at risk** | 0.76 (0.65-0.87) | 0.73 |
| 2 | 64 | 0.993 / 0.997 | 0.89 (0.78-0.99) | 0.52 | **at risk** | 0.78 (0.68-0.88) | 0.73 |
| 2 | 128 | 1.000 / 1.000 | 0.91 (0.82-1.01) | 0.51 | **at risk** | 0.80 (0.71-0.88) | 0.72 |
| 1 | 8 | 0.912 / 0.852 | not derivable (no MoE at rows 8) | - | **open, at risk** | - | - |
| 1 | 16 | 0.914 / 0.923 | 0.79 (0.68-0.90) | 0.62 | on track (thin) | 0.70 (0.60-0.80) | 0.68 |
| 1 | 32 | 0.955 / 0.951 | 0.85 (0.76-0.95) | 0.63 | on track (thin) | 0.73 (0.64-0.82) | 0.69 |
| 1 | 64 | 0.987 / 0.979 | 0.89 (0.80-0.98) | 0.61 | **at risk** | 0.76 (0.68-0.85) | 0.70 |
| 1 | 128 | ATOM K3 | LB / ATOM K3 0.86 (0.77-0.94), derived | - | on track (thin) | about 0.73 | - |

- TP1 B128 (FlyTape cannot run it): ATOM K0 runs at 0.390 of the streamed floor (L0c-DP2 4), so its K3 per-token
  time corresponds to 0.518 of the floor; LB / ATOM K3 = 0.518 / eta_LB, with eta_LB carried over from the TP1 B64
  line (0.538 / 0.89 = 0.60, range 0.55-0.67). It assumes ATOM's read bytes equal the checkpoint floor's (the bytes
  model's `atom` and `checkpoint` regimes differ by under 1 % in their DP2 / TP2 ratios) and that eta does not fall
  from B64 to B128 (it rises for every stack in L0c-DP2 4).
- **Against the revision-1 budget** (section 0.4: 0.74 at TP2 B128, 0.56 at TP2 B16): the measured families give
  0.91 and 0.82. The difference is almost all MoE: section 10.2 assumed 0.85 of FlyTape's MoE and P-L4 measured
  1.05-1.29 on the expert + combine phases. The GDN, router, in-proj and head families came in at or below their
  revision-1 values; HC and TP2 out-proj came in worse in phase form.
- **The MoE is 0.51-0.63 of the LB step in every cell.** Bringing it to 0.85 of FlyTape's expert + combine moves every
  measured cell to 0.70-0.80, under its bar with a margin of 0.15 or more, and takes every range below its bar.
  Without it, 5 of the 8 derivable cells have ranges that touch or cross the bar.
- **MTP (WP-L9) at TP1.** If LB gains from K3 what ATOM gains (P-L1's one-state-read chain supports at least that:
  0.36-0.39 of the K0 state bytes with `snap=last`, 0.65-0.71 with ATOM's `snap=all`), LB K3 / ATOM K3 ~ (LB K0 / FT)
  x (FT / ATOM K0) = 0.54-0.63 at 1k B16-B64 and 0.56-0.66 at 8k. That is the robust TP1 margin; K0 alone is thin.
  The same holds for the TP2 B8-B16 cells, where ATOM DP2 K3 is the bar.
- **Two GPUs: LB TP2 vs LB DP2.** One DP2 GPU reads 1.14-1.40x a TP2 rank's bytes (L0c-DP2 4); LB's exchanges cost
  0.048-0.061 of the step; but the TP2 rank shape pays about 0.2 of a layer in sync against 0.04-0.11 at TP1 (P-L2
  4.1), and P-L4 F4 has LB TP2 / DP2 at 0.71-1.00 on the MoE phases. The 2-GPU LB entry is the better of LB TP2 and
  LB DP2 (two TP1 replicas, no CCL), measured per cell in WP-L4 / WP-L5.
- Correlated (real) routing cuts the bytes of both stacks; P-L4 F3 shows the LB MoE's gap to its floor grows under it
  (1.53-2.28x random, 2.08-3.17x strong skew), so the real-routing budget is likely worse for LB than this
  random-routing one until union-aware placement lands. FlyTape takes no routing input; the ATOM bars are measured on
  real routing.

### 15.4 Stages on track / at risk, and the lever that closes each

| stage | state | lever (owner WP) | what closing means |
|---|---|---|---|
| MoE experts + combine (S9 / S10) | at risk, decisive | build DESIGN-LB 6.2 as designed, which P-L4 did not: TP2 whole expert per CTA with `h` in LDS, TP1 IOD-local `h` mailbox; MoE input staged once per CTA in LDS (0.09-0.11); K split over waves with an LDS reduction; router top-k fused into the router epilogue (0.05-0.1); `combine=fused_next` with the consumer's read timed; per-layer walk no slower than one launch per layer (WP-C5 F4); union-aware placement for correlated routing (WP-L2d) | <= 1.00 of FlyTape's expert + combine in a same-job bracket at every cell (rows 8-128, TP1 and TP2, W4A8), target 0.85; null-bytes share well below today's 0.47-0.77 |
| MoE rows 8 bucket | open | build and measure r8 in the same rebuild (WP-L2d) | TP1 / TP2 B8 cells become derivable |
| out-proj at TP2, HC down / up (latency-bound small GEMMs) | at risk in phase form | tile-flag dataflow with weights issued before the poll (`sync=mix`), the HC per-stream re-association (removes the norm hop and the reduce phase), the LB plan's fewer boundaries (10-11 stages against 14-16), `fuse_out`; `fold=none` only if the in-situ consumer read is timed and wins (P-L3 7) (WP-L2a, WP-L2b, WP-L3) | in-situ phase <= FlyTape's per stage; the P-L3 stream column (0.30-1.03) is the reachable bound |
| TP2 sync share (about 0.2 of a layer) | at risk | planner balance (P-L2 5.2: the non-work share is phase-tail imbalance, not the barrier); cross-layer overlap in `pass`; partial edges as skew insurance (0.04-0.12 of a layer with one late unit) (WP-L1, WP-L3, WP-L4) | TP2 layer sync share at or below the TP1 LB plan's 0.04-0.11 |
| TP exchanges | on track for the step budget | defaults fixed at TP2 (lamport one-shot, `cons=pass`, `lay=unit` for the MoE); overlap with the next layer's units for the per-exchange 0.25 target; `ar_dtype=fp8` (0.53x bytes, tolerance) (WP-L5, WP-L8) | per-step share stays below FlyTape's skew floor (0.060-0.082) with both exchanges in situ |
| GDN recurrence | on track (f32 at the floor) | `lri_ldnt` (`best_knobs`); `state=bf16` (N11, RULER-gated); MTP chain + `snap=last` (WP-L2c, WP-L8, WP-L9) | no further f32 gain exists; bytes are the only lever |
| GDN in-proj, router GEMM, head | on track | conv / gates and top-k / argmax epilogues on E1; op_sel scale layout in every producer (P-L3 7) (WP-L2a, WP-L2c, WP-L2f) | keep the measured ratios with the epilogues in |
| QSA core | unknown | new probe P-L5 (below) before WP-L2e | a measured QSA family ratio at p128 and 8K |
| precision-matched bf16 arm | dropped from the budget | kept as `wdtype=bf16` knob; a faster grid barrier (FlyTape's striped sync, cross-phase prefetch) if the arm is ever needed | - |
| correctness infrastructure | at risk | inline-asm VMEM hazard scan as a compile gate on every LB / CCL build (`hz` on); a deterministic stale-read test (P-L4 F2: the `noacq` twin misses 6 (cell, arm) pairs); host check of the launch-fatal word after every launch / replay (D0.8 amendment) (WP-L1) | every gate run uses them |

### 15.5 Go / no-go

**GO for the large-batch build, conditional, MoE first.** The measured families say the LB K0 step sits at 0.79-0.91
of FlyTape's (central) in the derivable cells: under every bar at the centre, but with ranges that touch the bar in 5
of 8 cells, all because of one stage. Everything else (recurrence, in-proj, router, head, exchanges) came in at or
below FlyTape, and the exchange is already cheaper than FlyTape's own. Conditions:

1. **Gate G-MoE (before WP-L3 integrates the MoE):** the rebuilt MoE (15.4 row 1) at <= 1.00 of FlyTape's expert +
   combine phases in a same-job bracket in every cell, both nodes, W4A8; target 0.85. **Kill / re-plan criterion:** if
   the 6.2 shape with its levers still measures above 1.05 in any TP2 cell, the 2-GPU large-batch entry becomes
   LB DP2 + MTP (no CCL on the critical path) and the TP2 budget is re-issued before more TP2 work is spent.
2. **MTP is on the critical path for TP1** (bars 0.85-0.99 of FlyTape, met by K0 only with a thin margin) and for the
   TP2 B8-B16 cells (ATOM DP2 K3 bars). WP-L9 moves from group L5 to directly after WP-L4.
3. **No-go items (removed or demoted, all kept as default-off knobs):** the precision-matched bf16 budget line;
   `fold=none` as a default; the mailbox all-to-all and counted publish as sync defaults (P-L2); the `line` form,
   publish variants, two-shot and FlyTape's `cons=phase` as TP2 exchange defaults (WP-C5 F2); P-L4's wave-unit /
   global-`h` MoE as the base for WP-L2d (it stays as `moe_lb.py` for comparisons).
4. Revision-1 numbers (0.74 / 0.56) are withdrawn as targets; 15.3's per-cell values and bars replace them.

### 15.6 Build plan (dependency-ordered)

W5's WP1 platform has landed and passed its gate after its fix round (`wave1/WP1.md`): `qwen38_mono/persist/`
(`PersistKnobs` / `EdgePlan`, `GridSync` with 8 stripes per XCD, the residency ticket with the launch-fatal word,
`check_launch`, `kv` protocols, lints, the ISA gate, the runtime layer-loop driver) and the `FrameworkKnobs`
`persist` / `edges` / `err_arg` fields. LB code imports these and adds LB pieces under `large_batch/` (and `ccl/`);
it never edits W5's files (DESIGN-LB 11's "land once in `Q/`" rule). What WP1 does not provide (its section 9:
emitter-level edge knobs, a framework delay-injection stress harness, per-use-site R2) is built LB-side where LB needs
it, or waits for W5's WP2 / WP3.

**Wave LB-1: can start now (needs only WP1, already landed)**

| WP | what | depends | gate |
|---|---|---|---|
| WP-L2d (rebuild, first) | DESIGN-LB 6.2's MoE as designed (15.4 row 1), rows 8-128 incl. the r8 bucket, TP1 and TP2, W4A8 + `act=fp4`; in-situ MoE-down exchange on it (`insitu_moe.py` `lay=unit`); new module next to `moe_lb.py` (which stays) | P-L4, WP-C5 F4 | G-MoE (15.5); T0, determinism, deterministic stale-read test, mutants on MI355X in the timing jobs; random + correlated routing |
| WP-L1 | LB framework on `persist/`: rows buckets, runtime token tiles, masking, task graph with partial edges, `sync=mix` per edge (`gmask_full(plan, 0.4)`, per-wave lazy flags, `s_nop` pause; other arms as knobs), LB validate (WAR + CCL slots via `ccl/war.py`), index-width lint, asm-hazard scan as a compile gate, deterministic stale-read harness, launch-fatal check per launch | WP1 (landed), P-L2 | CPU tests; GPU microbench reproduces P-L2's verdicts; negative controls; existing builds byte-identical |
| WP-L2a | E1 productized from `gemm.py`: planner with grid fill first, op_sel scale layout for producers and a one-time weight-scale repack, `fold=last` default, `act=fp4` (against aiter `gemm_a4w4` on gdn_in / qsa_in / head r64-128), 32x32x64 MFMA lever, `wdtype=bf16` as a knob | P-L3 | P-L3's numerics suite incl. op_sel mutants; in-situ ratios of 15.2 held |
| WP-L2c | E3 (conv / gates epilogue on E1) / E4 (`lri_ldnt`, `best_knobs`) / E5; `state=bf16` knob with its RULER plan | WP-L2a (E3, E5), P-L1 | T0 vs fp64 + bitwise vs the exact-order emulation |
| P-L5 (new probe) | QSA core at LB (row-gang, MFMA QK / PV, indexer, select) at p128 and 8K vs FlyTape's QSA phases; the last unmeasured family | - | T0; WIN rule; same-job FlyTape bracket |
| WP-L0c remainder | measured U and routing correlation from the real checkpoint (router dump from an ATOM native run on a staged copy; `dev/ATOM` read-only) | - | budget 15.3 re-issued under real routing |

**Wave LB-2: after LB-1 (and the named W5 pieces)**

| WP | what | depends |
|---|---|---|
| WP-L2b | E2 HC LB with the per-stream re-association and the faithful-order twin | WP-L2a; reads W5 WP2a's HC semantics (no edits) |
| WP-L2e | E9 QSA LB | P-L5; W5 WP2c's Q3-Q7 semantics |
| WP-L2f | E10 head (E1 over vocab rows, fused argmax) can start on WP-L2a; E11 PLE / E12 final mix reuse W5's WP2e / WP2f emitters | WP-L2a; **waits for W5 WP2e / WP2f** |
| WP-L3 | TP1 `layers=1` layer kernels per bucket; in-situ decisions: sync per edge, `fold=none` with the consumer read timed, `fuse_out`, `combine=fused_next` | WP-L1, WP-L2a-f, G-MoE |
| WP-L4 | TP1 `pass` on WP1's loop driver and the `layers=1` + head graph; the LB DP2 arm (two TP1 replicas) against ATOM DP2 K3 | WP-L3 |

**Wave LB-3: TP2, MTP, serving**

| WP | what | depends |
|---|---|---|
| WP-L9 (promoted) | MTP at LB: chain recurrence (P-L1 `chain=4`), verify rows c(K+1) <= 128, round structure at LB width | WP-L4; `snap=last` needs the ATOM D2 contract and DESIGN WP5c from **W5** (`snap=all` works today at 0.65-0.71) |
| WP-L5 | TP2: CCL in situ with WP-C5's defaults, head all-gather, KV head split, PLE reduce; exchange overlap with the next layer's units (the DESIGN-CCL per-exchange target); LB TP2 vs LB DP2 per cell | WP-L4, CCL WP-C7, G-MoE |
| WP-L6 / WP-L7 | ATOM-side serving experiment and e2e cells + accuracy suites | **W5's ATOM runner hook**; WP-L4 / WP-L5 / WP-L9 |
| WP-L8 / WP-L10 | levers (`state=bf16`, `ar_dtype=fp8`, `part=bf16`, union-aware placement, ...) and RSI with combination lookback over every retained knob, losers included (PLAN 13, 15) | continuous after WP-L3 |

**Next WPs to launch (in this order, at most two workflows at a time):** WP-L2d rebuild and WP-L1 in parallel; then
WP-L2a + WP-L2c + P-L5; WP-L0c's real-routing measurement alongside whenever a GPU is free.
