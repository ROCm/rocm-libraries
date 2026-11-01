# P-L4: the large-batch expert-major MXFP4 MoE (DESIGN-LB S9 / S10, E7 / E8), rows 16-128, TP1 / TP2, W4A8 and W4A4

Probe P-L4 of `../DESIGN-LB.md` (section 12; design 6.2). One persistent rocKE launch (256 CTAs x 512 threads, one
CTA per CU) runs a whole Qwen3.8-Flash-Next MoE layer for up to 128 rows: top-10 routing from the router logits,
per-CTA LDS counting sort into per-expert token lists, expert-major gate_up units (MXFP4 weights x MXFP8 activations
= ATOM's W4A8, or x MXFP4 = FlyTape's W4A4), SiLU + requantization, down units, a slot-ordered combine. First the
null-bytes build (every expert weight load against a zero-length buffer: the fixed overhead), then the full build at
rows 16 / 32 / 64 / 128, TP1 (I = 640) and the TP2 rank shape (I = 320). Ratios only here; absolute times are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L4.md`.

Status (after the Fix round at the end of this file): **done. The numerics gate is met on the timed SKU: tolerance vs
float64 torch and FlyTape's oracle formula, bitwise knob invariance, the 2-layer walk, and every non-stale must-fail
mutant caught in every cell on MI355X, under random and correlated routing. The stale-read mutant `noacq` is NOT a
reliable must-fail check (Fix round F2): 6 (cell, arm) pairs are never or almost never caught on MI355X whatever the
twin placement, and the catch rate of others depends on the job (the container job missed 8 of 16 `hcgs` cells).
Where it is missed, the twin method cannot show that the acquire is needed; the acquire stays (P5) and every build
passes with it. The performance target is NOT met:** in a same-job FlyTape
bracket the best LB MoE costs 1.05-1.29x (gs) / 1.02-1.22x (launch) FlyTape's MoE expert + combine phases, 1.53-2.28x
the bytes floor (gs; same job), and its null-bytes build alone costs 0.47-0.77 of the full kernel (per run; 0.51-0.76 as the mean of
the two lookback runs in table 5.1). Correlated routing (Fix round F3) makes the floor gap larger, not smaller.
W4 started this probe (`moe_lb.py` v1 / v2, jobs `rk-q38-w4-pl4-*`); W6 (this file) found and fixed a correctness bug
in v2, added the levers below, and measured. Jobs: MI355X 68233445 / 68233493 / 68233754 / 68233948 / 68234000
(`rocke`, n01-09 and n07-17); MI350X numerics / mutant / screen / WaveScope jobs 68232819-68233795 (`GFX950&MARKHAM`).
Fix round: MI355X 68234596 / 68234721 / 68234874; MI350X / compile-only 68234446-68235065.

## 1. Summary

1. **Fixed overhead is the dominant risk, as DESIGN-LB 6.2 feared.** With zero weight bytes (`null=oob`: same
   instruction stream, routing, sort, polls, h / partial traffic and combine) the default kernel costs **0.47-0.77 of
   its own full time** (per run, all MI355X runs; TP1 0.47-0.66, TP2 0.58-0.77; larger at fewer rows; table 5.1 shows
   the mean of the two lookback runs, 0.51-0.76), i.e. **0.83-1.87x the bytes floor**
   (the fastest streamer of the same active-expert bytes) and **0.61-0.97 of FlyTape's measured MoE expert + combine
   phases** (cross-job, FlyTape's traced 48-layer step). The structure alone, before a byte of weights moves, is as
   slow as reading all the weights.
2. **The full build:** best arm per cell (combination lookback, both rocke nodes agree within 0.02) 0.89-0.98 of
   the default, **1.51-2.30x the bytes floor** (default 1.60-2.50x). Against FlyTape (phase metric, W4A4 matched precision /
   W4A8): **TP1 1.06-1.12 / 1.10-1.18, TP2 1.08-1.31 / 1.09-1.32 of FlyTape's experts + combine** (worst at TP2
   32 rows, where FlyTape runs its fused `moe_expert`); 0.84-1.08 of FlyTape's whole MoE family, which also contains
   its router GEMM + top-k (0.11-0.25 of it; this probe starts from the logits). Launch metric (one launch per
   layer): 1.02-1.27 / 0.81-1.05. Not a win anywhere on the experts. FlyTape has no TP1 128-row point. **Fix round
   (F4): these FlyTape ratios are cross-job; the same-job bracket gives 1.05-1.29 (gs) / 1.02-1.22 (launch) on the
   experts + combine; the closest cell, TP1 fp4 r16, is 1.05 gs / 1.015 launch: slower, by little on the launch
   metric.**
3. **A correctness bug in the v2 kernel W4 left (never run on hardware) was found and fixed: inline-asm VMEM after a
   VALU write of its SGPR base.** `device_ops.atomic_add_scoped` / `ccl.wire.poll_count_ge` asm read their `saddr`
   right after the compiler restored the spilled base pair with `v_readlane`; gfx9 needs 5 wait states there and
   LLVM does not pad VMEM inside inline asm. Grid-barrier arrivals landed at a wrong address (every CTA timed out,
   the barrier counters stayed 0) or the launch faulted (`MEMORY_APERTURE_VIOLATION`), depending on register
   allocation (a heisenbug: a debug store hid it, an unrelated knob exposed it). Knob `hz` (default on) starts every
   asm atomic / poll with `s_nop 4`; `hz=0` reproduces the v2 IR exactly and faults in the 2-layer walk. The same
   helpers back P-L3's GEMM grid barrier and the CCL code: **Fix round F1 checked them with an ISA scanner over
   every build (941 P-L3 GEMM, 2,535 CCL): no hazard in any of them with the current compiler**; the scanner flags
   all 224 `hz=0` MoE builds and none of the 368 `hz=1` ones. Recorded in
   `rocke/platform/dsl_docs/optimization/arch/gfx950.md` and the alola skill (rule 176).
4. **What W4's v2 changes bought over its measured v1** (now arm `v1`: last-arriver combine, `sc1` MoE-input
   loads): v1 / default 1.24-1.62 (the combine phase with no-return arrivals + one acquire instead of `sc1` x loads).
5. **Levers (all bitwise equal to the default, kept as knobs):** op_sel scale bytes `sop` (P-L3's lever; here only
   0.94-0.99, the MoE is not MFMA-bound), per-wave byte balance `wbal` (0.93-0.98 at TP1 and TP2 r32; LOSS 1.14-1.16
   at TP2 r16), grid-barrier hand-off + combine `hand=gbar,combine=gbar` (wins at TP1 r64-128 and with `dt=8` at
   TP2 r16-64, loses elsewhere), `dt=8` (TP2), `gcb=2` (TP1), plain (non-`nt`) weight loads `wcp=default` (wins at
   r128, loses at r16), packed down scales `dsop` (tie); losers: cross-unit prefetch `xu` (1.00-1.15), expert gap
   repack `gap` (1.00-1.02), `depth` 2 / 4 / 5 (tie), `map=split` (tie), `combine=last` / `phase` vs `gbar` alone.
6. **Where the time goes (WaveScope ATT on one CU + diagnosis arms):** waves are EXEC 10 %, WAIT 47 %, STALL 42 %;
   42 % of stall cycles are vector-memory instructions that cannot issue (gate_up weights 19 %, MoE-input gathers
   11 %), 19 % are dependency polls (flow hand-off / combine), 10 % CTA barriers (route poll, layer end). Diagnosis
   arms (MI350X, ratios): route + sort alone 0.09-0.11 of the default; MoE-input (x) loads 0.09-0.11; h loads in the
   down units 0.04-0.08; the counted combine 0.08; the remaining null structure (unit loops, MFMA chains, epilogues)
   0.24-0.29. Removing x and h loads entirely would leave the full kernel at 0.87.
7. **Verdict for DESIGN-LB (restated in the Fix round, F5):** the built variant (a wave owns (tile, 32 columns) over
   all K, the gate_up -> down hand-off through global `h` by per-(expert, tile) counters or a grid barrier, a counted
   or grid-barrier combine) cannot reach FlyTape's MoE; the gap is structural (latency chains, gathers and syncs),
   not bytes or MFMA. This kernel did **not** build DESIGN-LB 6.2's TP2 shape (a whole expert per CTA with the
   intermediate kept in LDS) nor its TP1 IOD-local mailbox, so the measurement condemns the wave-unit / global
   hand-off variant, not E7 as designed. WP-L2d should build 6.2's CTA-level fused TP2 shape and the TP1 IOD-local
   exchange, plus what section 7 adds: the MoE input staged once in LDS, K split across waves with an LDS
   reduction, the router top-k fused into the router GEMM epilogue, and `combine=fused_next`.

## 2. What was built

| file | content |
|---|---|
| `rocke/library/kernels/gfx950/qwen38_mono/large_batch/moe_lb.py` | the LB MoE (`MoeKnobs`, `build_moe_kernel`, `build_stream_kernel` = bytes floor, `unit_plan`, layout helpers `sop_layout_gus` / `sop_layout_xs`); Python engine only, names `q38lb_moe*` / `q38lb_mst*`; nothing served imports it. W4 wrote v1 / v2 (route, sort, units, counters, combine, null build, mutants); W6 added `hz` (default on), `combine=gbar`, `hand=gbar` (+ cross-barrier first-unit prefetch), `gcb`, `sop`, `dsop`, `xu`, `wbal`, `gap`, `drot` + `pretouch` (stale-read twins), mutants `noacq`, `sopsel`, diagnosis-only `diag` (`nogst`, `nodst`, `nox`, `noh` added), `dbgerr`, `ustamp` |
| `rocke/platform/dsl_docs/optimization/arch/gfx950.md` | the inline-asm VMEM / VALU-SGPR hazard (doc only) |
| `hackathon-prep/qwen38/truemono-lb/p-l4/` | `pl4_bench.py` (numerics suite, mutants, stale-read twins, interleaved timing, probe / lookback arms `--karms`, stamps), `job.sh` (per-GPU health check: a 2-GPU allocation skips the bad n07-17 bus e5), `dbg_arms.py` (per-spec fault triage), `att_moe.py` + `att_job.sh` (WaveScope capture), `pl4_report.py`, `pl4_tables.py`, `pl4_final.py`, `w6/` (the patch scripts, applied, and job submitters) |

Kernel structure (default build): route (token r on wave 0 of CTA r: 10 iterative wave argmaxes, renormalised softmax,
shared-expert sigmoid gate; one agent-scope arrival) -> every CTA polls, acquires (`buffer_inv sc1`) and sorts all
B x 10 assignments in LDS (bitmaps, packed block scan, tile table, token lists; deterministic) -> gate_up units
(tile, 32 columns) dealt on a balanced byte line per CTA, round robin over the 8 waves; `depth` K chunks of weights
in flight; epilogue SiLU x up, MXFP8 (ATOM's rule) or MXFP4 (FlyTape's RCEIL) requantization, `sc1` stores, one
arrival per (expert, tile) -> down units (tile, 64 rows): weights issued, poll the tile's gate_up counter, `sc1` h
loads, MFMAs, `w_slot x part` f32 partials -> per-(token, column block, lane group) no-return arrivals -> combine
units poll 64 counters and sum the 11 slot partials in slot order (bf16 y). Counters live in two epoch-parity sets
(no reset atomics); one launch can walk L layers behind a striped grid barrier (the `gs` metric).

## 3. Method

- Cells: TP1 (I = 640) and TP2 (I = 320 per rank: one rank's intermediate split; the cross-rank y reduce is the CCL's
  job and is not in this probe) x W4A8 (`fp8`) and W4A4 (`fp4`) x rows 16, 32, 64, 128 (bucket = rows) = 16 cells.
  513 experts (512 + shared) of random MXFP4 weights per layer, random router logits (near-uniform routing as
  FlyTape's bench: U = 134-143 active experts at 16 rows, 231-252 at 32, 357-383 at 64, 461-478 at 128).
- Metrics (us per layer; 8 layers rotating, about 5 / 10 GB of weights, cold): **gs** (one launch walks the 8
  layers with a grid barrier between them: phase semantics, the headline) and **launch** (one launch per layer in a
  HIP graph).
- Arms per cell, interleaved in shuffled rounds in one process: `lb` (default), `lb2` (second graph) and `lb_rc`
  (recompiled twin) as controls; `null` (default, zero-byte weights), `nsop` / `nhcgs` (null of two lever builds);
  knob arms; streamers `st` (16 B / lane, depth 4), `st_def`, `st4u4`, `st8u2` over the same active-expert bytes:
  **floor = the fastest streamer**. Win rule of DESIGN.md 9.1 (median beyond both controls by 0.002 and 75 % of
  rounds). 12-16 rounds.
- Final runs: base arm sets on n07-17 (bus 85, 68233445; bus 15, 68233754) and n01-09 (bus f5, 68233493);
  combination lookback (19 pairs / triples of the retained knobs, decision 13) on both nodes: n01-09 bus 05
  (68233948) and n07-17 bus 95 (68234000). Lever screens and diagnosis on an MI350X host (ratios for triage only).
- FlyTape reference: its traced 48-layer step (`bench_qwen4_full.py --trace`, MXFP4, prompt 128), per layer = the
  sum of every `*.moe_*` phase / 48: `moe_router` (router GEMM + top-k), `moe_up` + `moe_down` or the fused
  `moe_expert`, `moe_combine` (at TP2 including its cross-rank push). Median of the traced runs on the same node
  type: W3 68217648 / 68217675 (n01-09; B32 and B128 only), P-L3 68231605 (n01-09), P-L1 68230603 (n07-17, before
  and after its rocKE run). **Corrected in the Fix round:** that is 4 runs at B32 / B128 but only 3 at B16 / B64, and
  the per-cell spread (max / min of experts + combine) is 1.003-1.053, not "within 3 %" (TP2 B64 1.053, TP2 B128
  1.045, TP2 B32 1.035, TP1 B64 1.034). TP1 B16-B64, TP2 B16-B128 (FlyTape TP1 B128 refuses: i32 index limit).
  The first same-job FlyTape bracket waited 2 h behind the q38 GPU budget and was withdrawn; the Fix round ran one
  (68234721, F4), which supersedes the cross-job FlyTape ratios. FlyTape is W4A4 only: `fp4` cells are the matched-precision comparison, `fp8`
  cells carry ATOM's (more expensive) numerics.

## 4. Numerics gate (hardware; every timed binary on its own data before timing)

- Checks per launch: route ids equal torch's top-k (order, ties), weights within 2^-20; gate_up h within one
  quantum of float64 `silu(g) u` of the decoded operands plus the MFMA allowance, E8M0 scales equal to the host
  rule; down + combine within `2^-8 |ref| + 2^-11 sum|terms|` of float64 with the kernel's own h; end to end
  against FlyTape's oracle formula (`reference.moe`: f32 gate_up, fake-quantized h, bf16 down, bf16 slot-order
  accumulation): mean error <= 1.02x the oracle's and max <= the oracle's (observed: 0.34-0.39 of the oracle's mean,
  0.29-0.58 of its max); 3-launch determinism; every knob arm bitwise equal to the default; the 2-layer grid-synced
  walk bitwise equal to two single launches; error word 0; the counter area left exactly as one launch must leave it.
- **Records:** 16 of 16 cells PASS in every MI355X timing run (t1 + t2, f7, both lookback runs; those runs had
  `--no-mutants`) and in the final MI350X flow-build mutant runs (68233792-68233795). **Corrected in the Fix round:**
  the MI350X stale-read runs (68234266 v1, 68234268 v2, 68234292 v3, 68234293 v4) print `numerics FAIL` for TP1 fp8
  r16 and r32, only because `noacq` on `hcg` was not caught there; no mutant ran on MI355X before the Fix round, which
  ran every mutant on MI355X (F2). Every knob arm (sop, wbal, dsop, gcb, dt, depth, xu, gap, map, wcp, hand / combine gbar and
  their 19 lookback combinations) bitwise equal to the default in every cell it ran.
- **Mutants (must fail; h, hs, partials and y poisoned before each launch):** `slot` (combine drops slot 9), `wscale`
  (non-op_sel scale path unshifted), `sort` (token ranks off by one), `cnt` (combine before the last arrival /
  combine barrier dropped), `sopsel` (op_sel picks the next chunk's scale byte): all caught in every cell on the
  default and on `ds`, `dsw`, `hcg`, `hcgs` (final MI350X runs 68233792-68233795: 240 checks on the flow builds; 68234266 / 68234268 / 68234292 / 68234293:
  272 on the gbar builds).
  `race` (hand-off poll / barrier dropped) is reported, not required (timing-dependent); caught in every run.
- **Stale-read mutant `noacq` (no acquire after a grid barrier) on its twin** (`pretouch`: every wave loads the h /
  partial lines it will read, right before its CTA arrives; `drot=128`: the down deal rotated by half the grid, so a
  CTA reads h produced by far CTAs; no cross-barrier weight prefetch, so nothing evicts the pretouched lines): with
  the acquire kept the twin is bitwise equal to the default in 32 of 32 (cell, arm); without it the mutant is caught
  in **30 of 32** (27 on the first launch, 3 within 5), on `hcg` and `hcgs` in every TP2 cell and every TP1 cell
  except **`hcg` (the non-op_sel gbar build) at TP1 fp8 r16 / r32: never in 5 launches, in four jobs** (its op_sel
  twin `hcgs` is caught there on the first launch). The partial-line half alone (`combine=gbar`) is caught at once
  everywhere it ran. Two twin placements lost bite and were dropped: pretouch at layer start (evicted by the gate_up
  stream) and inside the arrival after the CTA barrier (TP1 fp4 never caught). The acquire stays regardless (P5: no
  safe-by-eviction).

## 5. Results

### 5.1 Fixed overhead (null-bytes build), gs metric, both nodes (lookback runs)

`null / lb` = the null build over the full default; `null / floor` = over the bytes floor; `null / FT e+c` = over
FlyTape's MoE experts + combine per layer (cross-job). From the same table as 5.2:

| rows | TP1 fp8 | TP1 fp4 | TP2 fp8 | TP2 fp4 |
|---|---|---|---|---|
| 16 | 0.67 / 1.22 (FT 0.76) | 0.61 / 1.07 (FT 0.66) | 0.76 / 1.87 (FT 0.89) | 0.72 / 1.73 (FT 0.82) |
| 32 | 0.60 / 1.04 (FT 0.75) | 0.54 / 0.88 (FT 0.63) | 0.68 / 1.44 (FT 0.97) | 0.63 / 1.26 (FT 0.85) |
| 64 | 0.59 / 1.01 (FT 0.75) | 0.51 / 0.84 (FT 0.61) | 0.66 / 1.29 (FT 0.80) | 0.60 / 1.13 (FT 0.70) |
| 128 | 0.56 / 1.00 | 0.51 / 0.83 | 0.62 / 1.21 (FT 0.80) | 0.57 / 1.07 (FT 0.70) |

(cells: `null / lb` / `null / floor`, FlyTape ratio where traced.)

### 5.2 Full build, best combination per cell (both rocke nodes), gs metric

| cell | best arm (both nodes) | best / lb (n01-09) | best / lb (n07-17) | best / floor | null / lb | null / floor | best / FT e+c | best / FT MoE | null / FT e+c |
|---|---|---|---|---|---|---|---|---|---|
| tp1 fp4 r16 | sop | 0.972 W | 0.979 T | 1.71 | 0.61 | 1.07 | 1.06 | 0.89 | 0.66 |
| tp1 fp4 r32 | sw | 0.944 W | 0.955 W | 1.56 | 0.54 | 0.88 | 1.11 | 0.98 | 0.63 |
| tp1 fp4 r64 | sw | 0.933 W | 0.940 W | 1.53 | 0.51 | 0.84 | 1.12 | 1.02 | 0.61 |
| tp1 fp4 r128 | g2 | 0.909 W | 0.918 W | 1.51 | 0.51 | 0.83 | - | - | - |
| tp1 fp8 r16 | sw | 0.978 W | 0.978 T | 1.78 | 0.67 | 1.22 | 1.10 | 0.92 | 0.76 |
| tp1 fp8 r32 | g2 | 0.941 W | 0.948 W | 1.64 | 0.60 | 1.04 | 1.17 | 1.03 | 0.75 |
| tp1 fp8 r64 | g2 | 0.941 W | 0.938 W | 1.60 | 0.59 | 1.01 | 1.18 | 1.08 | 0.75 |
| tp1 fp8 r128 | g2W | 0.888 W | 0.896 W | 1.58 | 0.56 | 1.00 | - | - | - |
| tp2 fp4 r16 | h8d | 0.940 W | 0.936 W | 2.25 | 0.72 | 1.73 | 1.08 | 0.84 | 0.82 |
| tp2 fp4 r32 | h8 | 0.972 W | 0.955 W | 1.93 | 0.63 | 1.26 | 1.31 | 1.06 | 0.85 |
| tp2 fp4 r64 | h8dW | 0.951 W | 0.950 W | 1.79 | 0.60 | 1.13 | 1.11 | 0.96 | 0.70 |
| tp2 fp4 r128 | sW | 0.950 W | 0.947 W | 1.77 | 0.57 | 1.07 | 1.16 | 1.02 | 0.70 |
| tp2 fp8 r16 | h8d | 0.941 W | 0.933 W | 2.30 | 0.76 | 1.87 | 1.09 | 0.85 | 0.89 |
| tp2 fp8 r32 | sw | 0.934 W | 0.932 W | 1.96 | 0.68 | 1.44 | 1.32 | 1.07 | 0.97 |
| tp2 fp8 r64 | h8W | 0.934 W | 0.927 W | 1.82 | 0.66 | 1.29 | 1.13 | 0.97 | 0.80 |
| tp2 fp8 r128 | sW | 0.919 W | 0.921 W | 1.80 | 0.62 | 1.21 | 1.19 | 1.04 | 0.80 |

Arm names (all with `sop`): `sw` = `wbal`; `sW` = `wcp=default`; `g2` = `gcb=2,depth=2`; `g2W` = `g2` + `wcp=default`;
`h8` = `hand=gbar,combine=gbar,dt=8`; `h8d` = `h8` + `dsop`; `h8W`, `h8dW` = + `wcp=default`; `swW` = `wbal` +
`wcp=default`. W / T = the win rule's verdict in that run. FlyTape ratios: per layer, cross-job (median of the four traced FlyTape runs, section 3).

Launch metric (one launch per layer): best / lb 0.895-0.986, best / floor 1.43-2.04, best / FT e+c 1.02 (TP1 fp4
r16) - 1.27 (TP2 fp8 r32), best / FT MoE 0.81-1.05. Per-cell tables of every run (base arm sets t1 / t2 / f7 and the
two lookback runs, both metrics): `P-L4-tables.md`.

### 5.3 Run to run

The default's controls (`lb2`, `lb_rc`) sat within 0.98-1.01 in every cell and run. The best-arm ratios of the two
lookback runs (different nodes, GPU buses 05 and 95) differ by <= 0.017; the base-arm runs t1 / t2 / f7 (buses 85,
f5, 15) agree with them within 0.02. The floor ratios differ by up to 0.07 between nodes (P-L1's caveat).

## 6. Diagnosis

1. **ATT (WaveScope, MI350X, TP2 fp8 r128, `sop`, one CU, 32 traced waves):** wave states IDLE 0.2 %, EXEC 10.5 %,
   WAIT 47.0 %, STALL 42.2 %. Stall by class: vector-memory load issue 41.9 % (gate_up weight loads 19.0 %, MoE-input
   gathers 10.7 %, gate_up scales 4.5 %, down weights 4.9 %), `s_waitcnt vmcnt(0)` 30.5 % (the hand-off / combine polls
   alone 18.6 %, down h loads 6.0 %, down / gate_up store drains 5.8 %), `s_barrier` 10.5 % (the route poll's CTA
   barrier 6.8 %, the layer-end barrier 3.3 %), MFMA 2 %. The null build: EXEC 19.8 %, the same polls and barriers,
   vector-memory issue stalls still 27.6 % with no weight bytes.
2. **Diagnosis arms (MI350X, ratios to the default; `diag` builds, wrong numerics by design):** null 0.62;
   null without MoE-input loads 0.51-0.54, without h loads 0.58-0.60, without both 0.47-0.52, plus no combine
   0.40-0.44; route + sort + launch only 0.09-0.11. Full build without x loads 0.90-0.91, without h loads 0.92,
   without both 0.87; without the partial stores 0.87-0.92.
3. **What did not explain it:** MFMA (a scaled MFMA per 16 rows of mostly 1-3 live tokens; ATT: 2 % of the stall cycles);
   in-flight bytes (`depth` 4 / 5: tie); cross-unit prefetch (`xu`: no better: the waits are not at unit
   boundaries); HBM channel camping (`gap`: no gain); vector-memory instruction count in the down units (`dsop`
   removes 22-24 % of them: tie); per-wave imbalance (`wbal`: 2-7 %).
4. **In-kernel timers are not usable here:** summing a few `s_memtime` / `s_memrealtime` stamps per unit into LDS
   doubled the launch time (`ustamp`); only phase-boundary stamps (`stamp`, within 0.01) and ATT are trustworthy.

## 7. What it means for DESIGN-LB (WP-L2d)

- **Scope of the verdict (Fix round F5):** what was measured is the wave-unit variant with the gate_up -> down
  hand-off through global `h` (`hand=flow|gbar`). DESIGN-LB 6.2 already specifies, for TP2, a whole expert per CTA
  with the intermediate kept in LDS, and for TP1 an IOD-local mailbox; neither was built here. The numbers condemn
  this variant, not E7 as designed, and item 2 below is 6.2's own TP2 plan rather than a change of plan.
- **Budget:** DESIGN-LB section 10 assumed the MoE at about 0.75 of its floor and LB / FT about 0.85. Measured: the
  built variant reaches 0.44-0.65 of the floor (1 / best-over-floor) and, in the same-job bracket, 1.05-1.29x
  FlyTape's experts + combine (gs; launch 1.02-1.22); the LB step budget must carry the MoE at FlyTape's time or
  worse until WP-L2d builds 6.2's shape.
- **Plan shape that the measurements point to** (each item with the share it can remove, from section 6):
  1. MoE input staged once per CTA and tile in LDS (gate_up B operand from LDS): up to 0.09-0.11.
  2. Fused gate_up -> down at TP2 (I = 320: h stays in LDS, requantized there; FlyTape's `moe_expert`), an IOD-local
     h exchange at TP1: up to 0.04-0.08 (h loads) plus the hand-off polls / barrier (0.1-0.19 of stall).
  3. CTA-level units with K split over waves and an LDS reduction (FlyTape's `_Up` "K quarters"): every wave of a CTA
     streams the same unit, so no per-wave unit-count imbalance and more independent load streams per unit.
  4. Router top-k fused into the router GEMM epilogue (S8) and the per-token publication folded into the sort's
     input (no separate route counter + CTA-wide poll barrier): 0.05-0.1.
  5. `combine=fused_next` (the next stage reads the 11 slot partials): the combine (0.08) leaves the MoE.
  Sum of the removable shares: about 0.3-0.4 of the current default, i.e. the region where it would match or beat
  FlyTape's experts + combine (best / FT e+c 1.05-1.29 now, same job). Not measured: it is the WP-L2d build.
  6. Under correlated routing (F3) the hot experts' many tiles become long single-wave units: the union-aware
     placement lever (DESIGN-LB 6.2 / WP-L8) belongs to the same rebuild.
- Keep: op_sel scales (`sop`), the epoch-parity counter sets, `hz`, slot-order deterministic combine, the per-GPU
  health pick in job scripts. The stale-read twin method stays as a partial check only (F2): WP-L2d needs a stronger,
  deterministic stale-read test before it relies on the twin.
- **Shared asm helpers** (`device_ops.atomic_add_scoped`, `ccl.wire.poll_count_ge`, every asm `global_*` with an SGPR
  base): checked in the Fix round (F1) with an ISA scanner; no hazard in any P-L3 GEMM or CCL build today. The risk
  is allocation-dependent, so the scanner (`p-l4/fix/asm_hazard.py`) should run as a compile gate on every new build of
  those kernels; the helpers themselves were not changed (that would change existing builds' bytes).

## 8. Caveats

- The section 5 FlyTape ratios are cross-job (3-4 traced runs per cell, spread up to 1.053, section 3); the Fix
  round's same-job bracket (F4) supersedes them. FlyTape has no TP1 128-row point. FlyTape's `moe_combine` at TP2 includes its
  cross-rank push, ours has no cross-rank reduce; its router phase includes the router GEMM, ours starts from
  logits.
- Random weights. Sections 5-6 use near-uniform routing (as FlyTape's bench); the Fix round (F3) adds two synthetic
  correlated routings (no measured checkpoint routing exists yet: DESIGN-LB WP-L0c "measured U" is still open). The
  union-aware placement lever was not built (it is a WP-L8 lever in DESIGN-LB 12).
- PLAN decision 5 (2-GPU cells: the better of TP2 and DP2) is applied at phase level in the Fix round (F4): both
  stacks prefer TP2 for the MoE phase at B32-B128, so the verdict does not change. DP2 at B16 needs TP1 at 8 rows,
  which this probe does not build.
- `fp8` (W4A8) is ATOM's numerics class; FlyTape's W4A4 is cheaper (half the MoE-input bytes and gathers): compare
  `fp4` cells for the kernel-to-kernel ratio.
- The MI350X runs (screens, ATT, diagnosis) are another SKU (lower power cap): ratios for triage only; every
  recorded verdict is from MI355X.
- The stale-read mutant is not a reliable must-fail check (F2): 6 (cell, arm) pairs are not caught by any of 9 twin
  placements, and its catch rate elsewhere depends on the job.

## 9. Rerun

```bash
H=hackathon-prep/qwen38/truemono-lb/p-l4
bash $H/sync_pl4.sh both                                  # stages w6-pl4-h, w6-pl4-rocke
bash $H/w6/submitL.sh <tag> [node]                        # 16 cells x 19 combinations + floor / null, MI355X
bash $H/w6/submitF.sh <tag> venv|container [node]         # base arm set (+ FlyTape bracket in container mode)
bash $H/w6/numM.sh <tag> <cells> [knobs]                  # numerics + mutants + stale-read twins (MARKHAM gfx950)
python $H/pl4_final.py --runs A=<dir>,B=<dir> --ft <flytape ft dirs>   # the 5.2 table (--abs: private)
# Fix round (stages w6-fpl4-h / w6-fpl4-rocke on Austin, w6fpl4-h / w6fpl4-rk on Markham)
bash $H/fix/sync_fix.sh both both
bash $H/fix/submitM.sh <tag>            # MI355X: numerics + mutants (16 cells), correlated m / s + random timing
bash $H/fix/submitFT.sh <tag>           # MI355X container, 2 GPUs: FlyTape traced before / after the LB timing
bash $H/fix/submitN.sh <tag>            # MI355X: numerics + mutants at two twin placements, 3 routings
bash $H/fix/hazF.sh <tag> moe|gemm|ccl [i/n]   # compile-only ISA hazard sweep (Markham gfx950 host)
python $H/fix/corr_report.py <m1 dir>; python $H/fix/bracket_report.py <ft1 dir>; python $H/fix/mut_tally.py <pl4.json...>
```

## Verification (W6 independent verifier, 2026-10-31)

**Verdict: PARTIAL.** The probe's own gate (DESIGN-LB P-L4 row: T0 + deterministic slot-order combine) was exercised
on hardware with must-fail mutants that fail, and the negative performance verdict (no win against FlyTape's expert
+ combine phases) stands. But one item of the probe's scope was not run, two report statements are inaccurate, and
section 7's verdict blames the DESIGN-LB plan for a kernel that did not build that plan's TP2 shape.

What I checked (logs and raw JSON, not only this file):

- **Numerics gate on hardware: confirmed.** All 16 cells PASS (tolerance, 3-launch determinism, every knob arm
  bitwise equal, the 2-layer walk equal to single launches) in the MI355X runs t1 / t2 / f7 / l1 / l2
  (`p-l4-raw/*/job.log`, `pl4.json`). Default controls `lb2` / `lb_rc` are within 0.980-1.012 of `lb` in every l2
  cell. Arms are interleaved in shuffled rounds. Node and GPU bus ids are recorded, and each GPU is health-checked.
- **Mutants: confirmed on hardware, with one disclosure gap.** `slot`, `wscale`, `sort`, `cnt` and `sopsel` are caught in every cell
  of n11-n14 (68233752-68233795) and v1-v4 (68234266-68234293); every log is `MI350X`. **None of the MI355X
  timing jobs ran a mutant.** Every one ran with `--no-mutants`, so the rule "numerics before timing with must-fail
  mutants" holds only across jobs on another SKU, with another venv (`ckdsl-venv`). Section 4 says "MI350X", but
  section 1 and the builder summary present it as one hardware gate. v1 prints `numerics FAIL` for TP1 fp8 r16 /
  r32 (the uncaught `noacq` on `hcg`). That matches the disclosed gap. But section 4 also says "16 of 16 cells PASS ... in
  the final MI350X mutant runs" without saying that the stale-read runs report two FAIL cells.
- **My own re-run (stage `w6-vP-L4-n1`, MI350X job 68234359, bg-1w300-k2-3a, bus c3).** The current on-disk
  `moe_lb.py` and harness were freshly synced as stages `w6vpl4-rk` / `w6vpl4-h`, with no reuse of the builder's
  stage. The run covered cells TP1 fp8 r16 (the gap cell) and TP2 fp4 r32 (the worst FlyTape ratio), knobs
  `sop,hcg,hcgs`, mutant arms `hcg,hcgs`. Results:
  - The default passes, and all knob arms and stale-read twins (acquire kept) are bitwise equal.
  - 33 of 33 required mutants are caught.
  - `noacq` on `hcg` at TP1 fp8 r16 is again NOT caught in 5 launches. It is caught on `hcgs` and at TP2 fp4 r32.
    So the gap reproduces.
  - In the same job, `dbg_arms --layers 2 - hz=0` (2:fp8:16): the default walk is clean. `hz=0` passes its single
    launch, then `Memory Fault Error` in the 2-layer walk. **The `hz` fix and its "hz=0 still faults" claim
    reproduce.** The hz=0 IR equals the pre-fix v3 IR (`w6/ir_v3.txt` vs `ir_v6.txt`).
  - Log: `hackathon-prep/qwen38/truemono-lb/p-l4/w6v/w6-vP-L4-n1.log`.
  - No MI355X re-run: both rocke nodes were fully allocated (16 of 16 GPUs, our rk-q38 budget 8 of 8).
- **FlyTape reference: the comparison is fair, but some statements are inaccurate.**
  - The match itself is fair: rows, W4A4 for `fp4`, the rank shape at TP2, and the sum of the `*.moe_*` phases.
    FlyTape has no separate shared-expert phase; its `moe_up` / `moe_down` carry it, as LB's 513 experts do. I
    checked `pl4_report.ft_moe` against the raw `phases` blocks.
  - The comparison is cross-job and has no same-job bracket. This is disclosed.
  - "Median of four traced runs ... agree within 3 % per cell" is wrong. B16 / B64 have 3 runs (W3 traced only
    B32 / B128). The per-cell spread of expert + combine is 1.003-1.053: TP2 B64 is 1.053, TP2 B128 1.045,
    TP2 B32 1.035, TP1 B64 1.034.
  - LB depends more on the node than FlyTape does: `lb` on n07-17 over n01-09 reaches 1.075 (TP1 fp4 r128), while
    the FlyTape runs on the two nodes are within about 3 %.
  - So the closest cell is a tie, not "1.06x slower". At TP1 fp4 r16 the n01-09 best arm is 0.98 of the same-node
    FlyTape run (P-L3 68231605) on the launch metric, and 0.996 of the median. On gs it is 1.02 / 1.03.
  - The verdict "not a win anywhere on the experts" holds: no cell beats FlyTape beyond noise, and every other cell
    is >= 1.08 on both metrics. The lower end of the "1.06-1.32x" headline is within node and cross-job noise.
  - PLAN decision 5 (better of TP2 and DP2 for 2-GPU cells) is not applied. This is acceptable for a phase probe,
    since ATOM DP2 has no MoE phase equivalent, but it should be stated.
- **The probe scope was narrowed.** DESIGN-LB's P-L4 row specifies "random and correlated routing". Only near-uniform
  random routing was run. The design lever for correlated routing (union-aware placement) was not built. Section 8
  mentions this only as a caveat, not as a deferred item.
- **The section 7 verdict does not follow as worded.**
  - DESIGN-LB 6.2 already specifies this for TP2: "a whole expert fits one CTA and the intermediate stays in LDS",
    plus an IOD-local mailbox at TP1.
  - The built kernel has neither: `MoeKnobs` has only `hand=flow|gbar` through global `h`.
  - What was measured is a wave-unit variant with a global-memory hand-off, not the E7 plan as designed.
  - The recommendation "fused gate_up -> down at TP2" is the design's own TP2 plan, not a change of plan.
  - The diagnosis (structure-bound: polls, gathers, barriers) is well supported by the ATT data and the diag arms.
  - Read section 7 as "the wave-unit / global hand-off variant cannot reach FlyTape; build 6.2's CTA-level fused TP2
    shape (WP-L2d)". It does not show that "E7 cannot".
- **Minor inconsistencies.**
  - null / lb is "0.47-0.77" in section 1 and the builder summary, but "0.51-0.76" in the status line and the 5.1
    table. The 0.47 comes from the base-arm runs (f7: TP1 fp4 r128), not the lookback table.
  - "Best arm" is the maximum over about 20 arms, so the selection is biased in LB's favour. That only strengthens
    the negative verdict.
- **Not re-checked here, owners' follow-up (as section 7 says):** `device_ops.atomic_add_scoped` /
  `wire.poll_count_ge` users in P-L3 `gemm.py` and the CCL kernels.

Required fixes before P-L4 counts as complete for WP-L2d:

1. Run, or explicitly defer, the correlated-routing cells.
2. Correct the FlyTape spread statement and the run count.
3. Restate section 7 as above.
4. Note that the mutants never ran on the timed SKU or in the timing jobs. Add `--mutarms` / mutants to one MI355X
   numerics pass when rocke has capacity.

## Fix round (W6, 2026-10-31)

Every issue in the Verification above was addressed. Where a statement needed data, it was re-measured on hardware.

Jobs:

- MI355X (`rocke`, venv unless noted):
  - 68234596 `w6-fpl4-m1` (n07-17 bus 05). Run 1: numerics + every mutant, all 16 cells. Runs 2-4: correlated /
    correlated / random routing, numerics + mutants, then timing.
  - 68234721 `w6-fpl4-ft1` (n07-17, q38 container, 2 GPUs). The LB MoE ran on bus 75. FlyTape TP1 ran on bus 75 and
    TP2 on buses 75 + 65, traced before and after the LB run.
  - 68234874 `w6-fpl4-n1` (n01-09 bus 05): numerics + every mutant at two twin placements, for random and both
    correlated routings.
- MI350X (`GFX950&MARKHAM`, ratios and triage only): 68234446, 68234532, 68234550, 68234671, 68234676, 68234678,
  68235065.
- Compile-only ISA sweep on Markham gfx950 hosts: 68234526-68234530, 68234605-68234609.

Run checks:

- Every GPU was health-checked.
- In every new timing run the controls `lb2` / `lb_rc` stayed within 0.986-1.015 of `lb`, timing error words were
  0, and each run used 16 shuffled rounds.

Where things are:

- Absolute numbers: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L4.md`, section "Fix round".
- Raw data: `p-l4-raw/w6-fpl4-{m1,ft1,n1}/`.
- Harness: `hackathon-prep/qwen38/truemono-lb/p-l4/fix/` (`submitM.sh`, `submitFT.sh`, `submitN.sh`, `numF.sh`,
  `hazF.sh`, `asm_hazard.py`, `corr_report.py`, `bracket_report.py`, `mut_tally.py`, `ft_spread.py`, `sync_fix.sh`)
  and `pl4_bench.py` (`--routing`, `--twins`).

| verifier issue | resolution |
|---|---|
| correlated routing not run | Run (F3): two synthetic correlated routings, all 16 cells, numerics + mutants + timing on MI355X. Union-aware placement is deferred to WP-L8, where DESIGN-LB 12 lists it. |
| mutants never ran on MI355X or in the timing jobs | Run (F2): every mutant, all 16 cells, on MI355X. Three jobs on both rocke nodes, including both timing jobs (m1 runs 2-4, ft1). |
| section 4 "16 of 16 PASS" vs the FAIL lines | Corrected in section 4. The stale-read check is now characterised (F2). |
| FlyTape "four runs within 3 %" | Corrected in section 3: 3 runs at B16 / B64, 4 at B32 / B128, spread 1.003-1.053. Superseded by the same-job bracket (F4). |
| TP1 fp4 r16 is a tie, not 1.06 | Same-job bracket (F4): 1.053 gs, 1.015 launch. LB is slower on both metrics, by little on launch. |
| section 7 wording | Restated in section 1 item 7 and the first bullet of section 7 (F5). |
| null / lb 0.47-0.77 vs 0.51-0.76 | Both ranges are now stated with their definitions (per run vs the lookback mean), in the status line and section 1. |
| PLAN decision 5 not applied | Applied at phase level (F4). Both stacks prefer TP2 for the MoE at B32-B128, so the verdict is unchanged. |
| asm hazard in P-L3 / CCL unchecked | Checked (F1): ISA scan of 941 P-L3 GEMM builds and 2,535 CCL builds. No hazard. |

### F1. The inline-asm SGPR hazard in P-L3's GEMM and the CCL kernels

**How the scanner works.** `p-l4/fix/asm_hazard.py` disassembles each build. From every VMEM instruction that reads
an SGPR, it walks the control flow backwards:

- Branch targets come from the SOPP offsets.
- Each instruction counts one wait state; `s_nop N` counts N + 1.
- It reports any VALU write of one of those SGPRs fewer than 5 wait states earlier: `v_readlane`,
  `v_readfirstlane`, and VOP3 / VOPC SGPR destinations.

LLVM pads its own VMEM instructions, so every hit is an unpadded inline-asm VMEM.

**Positive control.**

- The scanner flags all 224 `hz=0` MoE builds and all 28 kept ISA dumps from before `hz` (`p-l4/w6/isa1`-`isa4`).
  It flags none of the 23 kept dumps built with `hz` (`isa5`-`isa12`).
- It flags none of the 368 `hz=1` MoE builds. Another 16 enumerated builds (the `xu=3` twin) are invalid knob
  combinations and were refused.
- A first version mis-read branch targets that named an asm label and reported false hits. This was fixed before
  the counted run.

**Builds scanned.**

- P-L3: every planner candidate of `pl3_bench.py` (all projections, TP, rows 16-128, `mx` and `bf16`), 941 builds.
- CCL: every build from `ccl-lm-fix/ir_snapshot.py` and `ccl-lm/compile_lm.py`, plus the link-bandwidth probes,
  2,535 builds.
- Toolchain: comgr from torch 2.12.0+rocm7.2 (the same torch build as the rocke venv), llvm22, gfx950.

**Result: no hazard in any P-L3 GEMM or CCL build.**

- Whether the hazard appears depends on register allocation, so any later change can expose it.
- The scanner should run as a compile gate on those kernels. A 600-build shard takes under a minute on 16 cores.
- The helpers were not changed, because changing them would change existing builds' bytes.

### F2. Mutants on the timed SKU (MI355X)

**Non-stale must-fail mutants.** These are `slot`, `wscale`, `sort`, `cnt` and `sopsel`, on the default and on the
arms `hcg`, `hcgs` and `h8d` (flow and gbar builds), all 16 cells:

| job, run | node, bus | routing | caught |
|---|---|---|---|
| m1 run 1 | n07-17, 05 | random | 224 / 224 |
| m1 runs 2 / 3 / 4 (timing job) | n07-17, 05 | corr-m / corr-s / random | 127 / 128, 127 / 128, 128 / 128 |
| ft1 (timing job, container) | n07-17, 75 | random | 128 / 128 |
| n1 runs 1 / 2 / 3 | n01-09, 05 | random / corr-m / corr-s | 224 / 224 each |

- Both misses are the `cnt` race (combine before the last arrival), at TP2 r16 under correlated routing, on its
  single launch. With few active experts, the combine happened to run late.
- `cnt` now gets up to 5 launches, like the other race. In n1 it was caught in every cell, once at launch 2.
- In every run, every knob arm stayed bitwise equal to the default, and every 2-layer walk matched single launches.

**Stale-read mutant `noacq`: not a reliable must-fail check on MI355X.**

Catch rates per (cell, arm) for the arms `hcg`, `hcgs` and `h8d` (the default build has no grid barrier, so no
`noacq`), caught by any twin placement:

| job, run | caught |
|---|---|
| m1 run 1 | 35 / 40 |
| n1 (random / corr-m / corr-s) | 34 / 40, 33 / 40, 36 / 40 |
| m1 runs 2-4 (`hcgs` only) | 15-16 / 16 |
| ft1 (container, bus 75, `hcgs` only) | 8 / 16 |

Six pairs are never or almost never caught:

- `hcg` at TP1 fp8 r16 and r32.
- `h8d` at TP2 fp8 r32 and fp4 r32 / r64 / r128. `h8d` is the TP2 arm `hand=gbar,combine=gbar,sop,dt=8,dsop`, a
  winning arm at TP2 r16.

Twin placements tried (all on MI350X; `ptall` also on MI355X in n1):

- The down-deal rotation `drot` at 1, 64, 128 and 129 (129 crosses XCDs: CTA t and t + 128 share an XCD).
- `tdelay` 8 and 32 (new, twin-only): odd CTAs sleep before producing, so consumers pretouch earlier.
- `ptall` (new, twin-only): the 32 CTAs of each XCD pretouch the whole h / hs region.
- Combinations of these.

None of the 9 placements catches those six pairs. `tdelay` also lost bite elsewhere (TP2 fp4 r32 `hcgs`).

What stays:

- The acquire stays in every build (P5: no safe-by-eviction). The builds pass with it.
- `ptall` and `tdelay` stay as default-off, twin-only knobs. The IR of 16 representative existing builds is
  byte-identical with them added.
- Before WP-L2d relies on the twin, it needs a deterministic stale-read test. One option: a producer that cannot
  store until every consumer has pretouched, without evicting the pretouched lines.

### F3. Correlated routing

`--routing corr:<pop>:<rho>:<groups>` works per layer:

- Each of the 512 routed experts gets a popularity bias drawn from N(0, pop^2), added to its logit.
- Rows are drawn into 8 topic groups that share a fraction rho of their logit variance.

Two strengths: corr-m = `corr:0.5:0.3:8` and corr-s = `corr:1.0:0.5:8`. Both are synthetic, because no measured
checkpoint routing exists yet (WP-L0c's measured U is still open). They bracket a moderate and a strongly skewed case.

All runs are in the same job (m1, n07-17 bus 05) with the same arms as the lookback: `sop` + the 9 lookback winners,
`null`, and the floor streamers over the same active-expert bytes.

| gs metric | random | corr-m | corr-s |
|---|---|---|---|
| active experts U / U random | 1 | 0.66-0.81 | 0.35-0.54 |
| most tokens on one expert | 3-12 | 7-60 | 11-127 |
| default / default random | 1 | 0.71-0.91 | 0.48-0.76 |
| best / floor | 1.53-2.28 | 1.68-2.67 | 2.08-3.17 |
| null / default | 0.49-0.76 | 0.52-0.77 | 0.60-0.83 |
| best arm / default | 0.89-0.98 | 0.93-0.98 | 0.91-0.99 |

On the launch metric, best / floor is 1.45-2.04 (random), 1.58-2.28 (corr-m) and 1.84-2.64 (corr-s).

- Correlated routing cuts the bytes (U) faster than it cuts the LB time, so the gap to the floor grows.
- The fixed overhead stays.
- A hot expert's 16-token tiles run as one long single-wave unit, up to 8 tiles at B128.
- Numerics pass in every cell at both strengths (F2).
- The per-cell table is in `P-L4-tables.md`, "Fix round".
- FlyTape's bench takes no routing input, so there is no FlyTape number under correlated routing. DESIGN-LB's matched
  comparison stays the random-routing one.

### F4. Same-job FlyTape bracket and PLAN decision 5

ft1 ran FlyTape's traced 48-layer step (TP1 B16-B64, TP2 B16-B128, prompt 128) before and after the LB run, on the
same node and GPUs.

The table gives the LB best arm (per cell, the lowest median of up to 11 arms; the `dt=8` arms exist only at TP2) over FlyTape's experts + combine. The
FlyTape value is the mean of the before and after runs; the range in brackets uses each run alone.

| cell | gs | launch | default / FT e+c (gs) | best / FT whole MoE (gs) |
|---|---|---|---|---|
| tp1 fp4 r16 | 1.053 (1.048-1.058) | 1.015 (1.010-1.020) | 1.067 | 0.890 |
| tp1 fp4 r32 | 1.080 | 1.037 | 1.122 | 0.960 |
| tp1 fp4 r64 | 1.068 | 1.042 | 1.123 | 0.981 |
| tp1 fp8 r16 | 1.061 | 1.024 | 1.127 | 0.897 |
| tp1 fp8 r32 | 1.136 | 1.101 | 1.169 | 1.010 |
| tp1 fp8 r64 | 1.139 | 1.104 | 1.214 | 1.047 |
| tp2 fp4 r16 | 1.053 | 1.027 | 1.119 | 0.827 |
| tp2 fp4 r32 | 1.256 | 1.194 | 1.300 | 1.018 |
| tp2 fp4 r64 | 1.102 | 1.073 | 1.167 | 0.953 |
| tp2 fp4 r128 | 1.147 | 1.100 | 1.208 | 1.009 |
| tp2 fp8 r16 | 1.070 | 1.041 | 1.142 | 0.840 |
| tp2 fp8 r32 | 1.289 | 1.221 | 1.354 | 1.045 |
| tp2 fp8 r64 | 1.140 | 1.106 | 1.232 | 0.986 |
| tp2 fp8 r128 | 1.169 | 1.120 | 1.255 | 1.028 |

**Not a win in any cell on the expert + combine phases:** 1.05-1.29 on gs, 1.02-1.22 on launch.

- The closest cell, TP1 fp4 r16, is 1.05 on gs and 1.015 on launch. Picking the best of several arms already
  favours LB.
- Against FlyTape's whole MoE family, the best arm is 0.83-1.05. That family includes FlyTape's router GEMM + top-k,
  which this probe does not run.

**FlyTape run to run** (`fix/ft_spread.py`, every traced run including ft1 before / after):

- 5-6 runs per cell, max / min 1.007-1.078.
- The ft1 runs sit 0-4.7 % above the median of all runs (n07-17, bus 75).
- So the cross-job ratios in section 5 are 0-5 % optimistic for LB in some cells. The same-job table replaces them.

**PLAN decision 5 (2 GPUs: the better of TP2 and DP2), at phase level.** DP2 means two TP1 replicas at B / 2 rows
each.

- For both stacks, TP2 is the faster MoE phase at B32-B128: LB TP2 / DP2 is 0.71-1.00 (gs), FlyTape 0.70-0.86.
- So the better-of ratio equals the TP2 one: LB / FlyTape 1.10-1.29 (gs) / 1.07-1.22 (launch) at B32-B128.
- B16 would need LB at TP1 with 8 rows, which no run measured.
- This is a per-step phase view, not decision 5's per-request latency match.

### F5. Section 7

Restated in place, in section 1 item 7 and the first two bullets of section 7:

- The measurement condemns the variant that was built: wave units with the `h` hand-off through global memory.
- DESIGN-LB 6.2's TP2 shape (a whole expert per CTA, `h` kept in LDS) and its TP1 IOD-local mailbox were not built.
- WP-L2d is where they get built and measured.

### F6. Code changes in this round

- `moe_lb.py`:
  - New twin-only knobs `ptall` and `tdelay`, default off. Both are refused without `pretouch`; `ptall` also needs
    `hand=gbar`.
  - The LLVM IR of 16 representative existing builds is byte-identical before and after. The set covers: default,
    flow / gbar, twins and `noacq`, `sop` / `dsop` / `dt=8`, `gcb`, `wbal`, `null`, `hz=0`, `v1`, `xu=3`, mutants.
- `library/tests/test_qwen38_lb_moe.py`:
  - 4 tests were failing before this round. `test_lowers` and `test_mutants_differ` built `noacq` / `sopsel` on the
    default build, which refuses them. They now use the base builds those mutants need.
  - Added lowering arms for the W6 levers and two tests for the new knobs.
  - Result: 83 passed (CPU, lowering only).
- `pl4_bench.py`:
  - New `--routing` and `--twins` options (twin placements written `d<drot>[t<tdelay>][a]`).
  - The `cnt` race gets 5 launches.
  - Any twin placement that catches `noacq` meets the stale-read requirement.
  - Max tokens per expert is recorded per cell.
