# W7 status: large-batch wave LB-1 (2026-11-02)

Wave report for the large-batch true rocKE monokernel (`DESIGN-LB.md` section 15.6, wave LB-1). Ratios, shares and
counts only; absolute numbers stay in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/`. Below 1 is faster.
Inputs: `WP-L1.md`, `WP-L2a.md`, `WP-L2c.md`, `P-L5.md` (each with its Verification section; P-L5 also with its Fix
round) and the MoE rebuild's `P-L4b.md` (WP-L2d). Nothing was run for this file. `DESIGN-LB.md` section 15 carries the
matching `W7:` markers and the re-derived budget (15.7).

## 1. Verdict

Wave LB-1 is done except WP-L0c's real-routing remainder (not started). All five built packages met their gates on
MI355X on both rocke nodes, and none changed the bytes of an existing build.
- Independently CONFIRMED: WP-L1, WP-L2a, WP-L2c.
- P-L5: TP1 confirmed. TP2 was PARTIAL (its TP2 table rested on container-compiled binaries that failed numerics), then
  closed by a fix round (prebuilt venv binaries, per-cell checks of both twins, both nodes). The fix round has not
  been re-verified.
- WP-L2d (`P-L4b.md`): G-MoE (DESIGN-LB 15.5 condition 1) met on both nodes; no independent verification yet.

**Budget consequence (DESIGN-LB 15.7, derived):** the K0 step is 0.71-0.81 of FlyTape's (central) in every 2-GPU
cell and 0.73-0.78 at TP1, with every range under its bar. The B8 cells are now derivable. The thin cells are TP1 B8
(8k bar, margin 0.04) and TP1 B16 (margin 0.05); MTP (WP-L9) stays on the TP1 critical path.

## 2. Package by package

| WP | gate | verification | what WP-L3 / later gets | still open |
|---|---|---|---|---|
| **WP-L1** framework (`large_batch/lbf*.py`) | met | CONFIRMED (n01-09 rerun of the final source, every gate held; n07-17 rerun lost to a node outage, covered by the kernel-identical builder runs) | rows buckets 8-128 with runtime tiles and masking; per-edge `EdgeSync` (default `gmask_full(0.4)` = `sync=mix`, every arm a knob); runtime layer loop; LB validate (deadlock, RAW, WAR incl. TP exchange slots, `ccl/war.py` hook); index-width lint; asm-hazard compile gate (it refused real `hz=0` hazards); deterministic stale-read harness (cache twin stale in 100 %); launch-fatal check per launch | 31 host-module CPU mutation survivors unclassified (LB validate / bounds / masking predicates: `lbf_plan.py` 406, 367, 443, 170 / 186) before WP-L2 relies on LB validate for slot reuse; one unexplained single-word mismatch in 1 of 7,280+ steps (likely harness race) |
| **WP-L2a** E1 GEMM core (`e1.py`, `e1_plans.py`) | met | CONFIRMED (container rerun on n07-17, same-job FlyTape; numerics suite there too) | 66-cell plan table (74 changes climbed, two-node rule); out-proj TP2 0.68-0.85, HC down 0.69-0.78, HC up 0.50-0.57, in-proj 0.23 / 0.33-0.37, `qsa_in` 0.25-0.31 / 0.33-0.36, router 0.21-0.25, head 0.26-0.36 of FlyTape's in-situ phase; op_sel producer layout; `fold=last`; `mfma=32` and `wdtype=bf16` default-off | `act=fp4` loses to aiter `gemm_a4w4` per launch 12 / 12; 10 pending climb trials; triples / surrogate lookback; `part=bf16` for the HC down fold; the fair TP2 out-proj figure with LB's own push (exchange accounting); no mutants on the 66 final picks themselves; launch-form plan table (WP-L3) |
| **WP-L2c** GDN family (`gdn_lb.py`) | met | CONFIRMED (n07-17 same-job FlyTape bracket, agrees within 0.03) | S3 + S4 + S5 in one persistent launch per step: TP1 0.44-0.69 (0.51-0.69 rows-matched), TP2 0.73-0.82 of FlyTape's GDN family, f32 state; per stage S3 0.32-0.37 / 0.51-0.71, S4 0.51-0.86 / 0.75-0.87, S5 0.37-0.50 / 0.54-0.85; one launch per step beats per-stage (1.06-1.43x) and per-layer (1.04-1.39x) launches; state warm (`pre=s4`); `state=bf16` 0.68-0.97 of f32 (default off) | `state=bf16` RULER T2 / T4 (drift: state error 0.003 -> 0.15 by 2,048 steps); partial edges S3 -> S4 / S4 -> S5 and per-edge flags (WP-L3); TP2 out-proj exchange (WP-L5); container COMGR fault on the `kskip` mutant (venv objects used); report wording fixes V2-V5 |
| **P-L5** QSA core (`qsa_lb.py`) | met (TP1 confirmed; TP2 via fix round) | PARTIAL, then Fix round closing all 8 issues; not re-verified | faithful port at parity at p128 (0.97-1.02 of FlyTape's `qsa_core`); per-cell picks 0.72-0.91 (TP1) / 0.72-0.88 (TP2) of the port (conservative), 0.60-0.84 of FlyTape in situ; levers `mpad`, `qloc`, `sel=v1/v2`, `score=mfma` + `kw16`, `waves=8` + `qlds`, `gang=4`; strict prebuilt-object mode (`PL5_HSACO_STRICT`) | port faster than FlyTape at 8K / 32K (0.76-0.94) unexplained until the DESIGN 16 IR diff; paged KV (`kvl=page16`) not built; register-lean 512-thread variant; climbs from `pick3`; RULER gate for `score=mfma` |
| **WP-L2d** MoE rebuild (`moe_lb2.py`, `P-L4b.md`) | G-MoE met | none yet | B5 base (`deal=rr, dt=4, doff=imm, pstg=4, pc=0, xb=1, depth=6, pcal=1, cmb=flow, dsq=1`) + per-cell dispatcher: TP1 0.84-0.98 (r8-r64), TP2 0.81-0.98 (r8-r128) of FlyTape's expert + combine; 0.58-0.84 of its whole MoE; 0.73-0.93 of P-L4; correlated routing 0.58-0.78 of the line; `entry=logits` / `entry=router` and `fnext` (`combine=fused_next`, chain 0.917-0.96) built; `pdt=bf16` 0.97-0.98 | independent verification; 0.85 target missing at TP1 r8 / r16 and TP2 r32 / r128; timed path takes route pairs (top-k in kernel 1.03-1.17x); no reliable r8 stale-read mutant (`flowrel` added as must-fail; `flowsc` report-only); no FlyTape correlated-routing trace; P-L4b section 6 still lists `fnext` / `entry=router` as not built (section 7.2 built them) |
| WP-L0c remainder (real routing) | - | - | - | not started; the budget is random-routing only |

## 3. Updated per-cell budget (derived; `DESIGN-LB.md` 15.7)

Method of DESIGN-LB 15.3: LB step / FT step = sum over families of FlyTape's share x the LB / FlyTape ratio. Unlike
15.3, every ratio is a phase or in-situ figure (no stream-metric low end), so the ranges are the measured spreads.

| family | FlyTape share (TP2 B128 / TP2 B32 / TP1 B32) | LB / FlyTape ratio used | source |
|---|---|---|---|
| MoE expert + combine | 0.394 / 0.370 / 0.460 | dispatcher arm per cell (W4A8, both nodes, pre / post brackets) x 1.03-1.17 (in-kernel top-k, `entry=logits`) | P-L4b 3, 7.2 |
| router GEMM | 0.050 / 0.090 / 0.060 | 0.21-0.25 | WP-L2a 8 |
| GDN in-proj + recurrence + out-proj | 0.276 / 0.210 / 0.270 | chain per cell, f32 state (TP1 0.514-0.688, TP2 0.732-0.813), + 0.03 on the high end for the node spread | WP-L2c 4.1, Verification |
| HC GEMMs | 0.109 / 0.136 / 0.086 | 0.50-0.78 | WP-L2a 8 |
| QSA family | 0.041 / 0.055 / 0.064 | 0.60-0.85 | P-L5 6 (core, conservative) + WP-L2a (`qsa_in`) |
| head | 0.009 / 0.015 / 0.019 | 0.26-0.36 | WP-L2a 8 |
| other (embed, PLE, final mix) | 0.005 | 0.8-1.2 (unmeasured) | - |
| FlyTape's reduce / combine / exchange-wait phases (the rest: 0.116-0.119 TP2, 0.036 TP1) | rest | TP2: LB's 96 in-situ exchanges, 0.048-0.061 of FT's step; TP1: 0-0.02 | WP-C5 F5 |

Shares as 15.3 (TP2 B16 uses B32, TP2 B64 the mean of B32 and B128, TP1 B16 / B64 use TP1 B32). The B8 cells borrow
the B32 shares (no B8 share trace), so they are the least certain.

| TP | B | bar (1k / 8k) | W6 15.3 K0 (range) | **W7 K0 central (range)** | MoE share of LB step | + `fused_next` | + bf16 state | state |
|---|---|---|---|---|---|---|---|---|
| 2 | 8 | 0.917 / 0.890 | not derivable | **0.71 (0.65-0.78)** | 0.51 | 0.69 | 0.70 | on track (shares borrowed) |
| 2 | 16 | 0.974 / 0.962 | 0.82 (0.69-0.94) | **0.72 (0.66-0.79)** | 0.51 | 0.70 | 0.70 | on track |
| 2 | 32 | 1.000 / 1.000 | 0.91 (0.79-1.03) | **0.76 (0.69-0.83)** | 0.54 | 0.74 | 0.74 | on track |
| 2 | 64 | 0.993 / 0.997 | 0.89 (0.78-0.99) | **0.74 (0.67-0.80)** | 0.50 | 0.71 | 0.69 | on track |
| 2 | 128 | 1.000 / 1.000 | 0.91 (0.82-1.01) | **0.81 (0.74-0.87)** | 0.52 | 0.78 | 0.74 | on track |
| 1 | 8 | 0.912 / 0.852 | not derivable | **0.75 (0.68-0.81)** | 0.64 | 0.72 | 0.73 | thin at 8k (margin 0.04) |
| 1 | 16 | 0.914 / 0.923 | 0.79 (0.68-0.90) | **0.78 (0.71-0.86)** | 0.65 | 0.75 | 0.76 | thin (margin 0.05) |
| 1 | 32 | 0.955 / 0.951 | 0.85 (0.76-0.95) | **0.73 (0.65-0.81)** | 0.61 | 0.70 | 0.69 | on track |
| 1 | 64 | 0.987 / 0.979 | 0.89 (0.80-0.98) | **0.77 (0.68-0.85)** | 0.59 | 0.74 | 0.71 | on track |
| 1 | 128 | ATOM K3 | 0.86 (0.77-0.94) vs ATOM K3 | **0.74 (0.66-0.82) vs ATOM K3** | - | - | - | on track (eta carried from B64) |

Reading:
- The W6 budget's decisive gap is closed. W6 had 5 of 8 derivable ranges touching or crossing the bar, all because of
  the MoE (1.05-1.29 of FlyTape's e+c). With the rebuild at 0.81-0.98 and WP-L2a's GEMM rows, every range sits under
  its bar, and the 2-GPU cells have margins of 0.11-0.19 on the high end.
- The MoE is still 0.50-0.65 of the LB step, so it is still the largest lever: the 0.85 target cells (TP1 r8 / r16,
  TP2 r32 / r128), `combine=fused_next` (in the column above, to be measured in situ), and `pdt=bf16` (0.97-0.98,
  accuracy-gated, not included).
- TP1 B8 / B16 margins are thin, as in W6: MTP (WP-L9) is what makes TP1 robust (15.5 condition 2 unchanged).
- **Not in this budget:**
  - a composed layer: WP-L3 measures the families together, with real edges between them;
  - the TP2 sync tail beyond what each family's own phase form carries (P-L2: about 0.2 of a layer at the TP2 rank
    shape; WP-L1: the barrier wins at idle);
  - E2's HC re-association, E10's argmax, E11 PLE and E12 final mix;
  - real (correlated) routing (WP-L0c remainder);
  - FlyTape's job-to-job drift (+-0.015).
- The 2-GPU entry is still the better of LB TP2 and LB DP2 per cell (WP-L4 / WP-L5). WP-L2c measured a DP2 replica's
  GDN chain at 1.14 / 1.13 / 1.02 / 1.00x the TP2 rank's at B16 / B32 / B64 / B128, before the TP2 exchange.

## 4. Next wave (LB-2): what can start

Concurrency: at most two workflows at a time (PLAN; memory cap), so the order below matters.

| WP | can start? | inputs ready | entry conditions / carry-overs |
|---|---|---|---|
| **WP-L2b** E2 HC LB (per-stream re-association + faithful-order twin) | **yes, first** | WP-L2a's E1 at the HC shapes (down 0.69-0.78, up 0.50-0.57); W5 WP2a's HC semantics (`wave2/WP2a.md`, gate met, read-only) | build on the WP-L2a plan table (best kernel base, not P-L3's arm); HC down fold is the cell's critical path (98 % of the traced CU's stall at 5 rows): `part=bf16` and the re-association remove the norm hop and the reduce phase; gate as WP-L2a (numerics suite, stale twins, same-job FlyTape, both nodes) |
| **WP-L2e** E9 QSA LB | **yes, first (in parallel with WP-L2b)** | P-L5's `qsa_lb.py` picks (port-relative 0.72-0.91); WP-L2a's `qsa_in`; W5 WP2c's Q3-Q7 semantics (`wave2/WP2c.md`, gate met) | paged KV (`kvl=page16`, ATOM block-16) is the first build item; time only venv-built objects in strict mode (`PL5_HSACO_STRICT=1`), check both twins per cell on random data; RULER gate before `score=mfma` can be a default; the DESIGN 16 IR diff of `qsa_core` (port faster than FlyTape at 8K / 32K, unexplained) belongs here; ideally after an independent check of P-L5's fix round |
| **WP-L2f** E10 head / E11 PLE / E12 final mix + embed | **E10 yes** (third); E11 / E12 **no** | E1 head pick 0.26-0.36 | E10: E1 over vocab rows with the fused argmax (and the TP2 head all-gather merge, CCL WP-C2); E11 / E12 wait for W5's WP2e / WP2f emitters (launched by W8, no report yet) |
| **WP-L3** TP1 `layers=1` layer kernels | **not for the gate**; integration prep can start once WP-L2b lands | WP-L1 framework; WP-L2a / L2c / L2d bodies | composes `gdn_lb` + E1 + `moe_lb2` + E2 / E9 on the WP-L1 framework; decides per edge in situ: `sync=mix` vs grid (WP-L1: grid wins at idle, `mix` wins with a late unit), partial edges S3 -> S4 / S4 -> S5, `fold=none` with the consumer read timed, `fuse_out`, `combine=fused_next`, walk and slots; carries every losing arm as a knob into the combination lookback (PLAN 13 / 15); needs the 31 WP-L1 host mutation survivors classified first and P-L4b verified before any layer number is quoted |
| **WP-L4** TP1 `pass` + head graph; LB DP2 arm | **no** | - | after WP-L3; measures the LB DP2 arm against ATOM DP2 K3 and decides the persistent walk vs per-layer launches with the whole layer (framework loop wins 1.018-1.023, GDN chain per step wins, MoE `gs` ~ `launch`); TP1 B128 runs (FlyTape cannot) |

Alongside, whenever a GPU is free: independent verification of P-L4b and of P-L5's fix round; WP-L0c's real-routing
measurement (router dump from ATOM native on a staged copy; re-issues the budget under real routing); the `state=bf16`
RULER T2 / T4 runs (N11, GDN chain).

## 5. Rules, tools and records from this wave

- Container COMGR is not trusted for timed builds: it produced wrong `waves=8` QSA builds with and without spills
  (P-L5), and a faulting `kskip` mutant build (WP-L2c). Load venv-built code objects by name (strict mode) in container
  jobs. Laptop COMGR allocates registers differently (alola rule 197): spill verdicts come from the cluster ISA.
- The asm hazard gate stays on (`hz`): it fired for real in WP-L1 (`v_readlane` -> asm store) and P-L4b found a new
  case (ballot `v_cmp` -> asm `v_mbcnt`).
- Two-node rule: single-node A-vs-A copies passed the win rule 3 of 18 times in WP-L2a's climb; only the two-node rule
  accepts a change, and no-op arms are dropped from the bench.
- Alola skill rules added this wave: 195 (WP-L1), 198 (a running job on a node that goes down is requeued and leaves
  only its log header), 202 (orphaned submit poller after a laptop suspend; 1-GPU backfill starving 2-GPU jobs on
  n07-17), 205 (WP-L2c), 208 (check a foreign stage still exists before reusing it), 209 (corrected), 217 (Markham
  20-minute cap, hsaco staging path, submit-task poller), 219 (strict prebuilt mode), and 185 (P-L4b findings).
- Absolute numbers and raw data: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/` (`WP-L1.md`,
  `WP-L2a.md`, `WP-L2c.md`, `P-L5.md`, `P-L4b.md`, and the `*-raw/` folders).
