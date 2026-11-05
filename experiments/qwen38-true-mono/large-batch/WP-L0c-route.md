# WP-L0c-route: real-checkpoint routing at large batch (W10)

WP-L0c's remainder (`../DESIGN-LB.md` 12 / 15.6, `W7-STATUS.md` 2 and 4): capture the served router's expert choices from
the real Qwen3.8-Flash-Next MXFP4 checkpoint on the rocke partition, measure U per layer and the correlation across rows,
re-time the WP-L2d MoE (`moe_lb2`, B5 and the per-cell dispatcher picks, read-only use) under those routes, and re-issue
the per-cell K0 budget (DESIGN-LB 15.7) under real routing. Ratios, shares and counts only; absolute times are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L0c-route.md`. Below 1 is faster. Measured 2026-11-02 on
MI355X (both rocke nodes).

## 0. Summary

1. **Real routing is strongly correlated.** Over 212,736 captured decode router calls (K0, steady state: the first 16
   decode steps dropped), the unique routed experts per layer and step are U / U_rand = 0.81-0.90 at 8 rows,
   0.67-0.85 at 16, 0.61-0.77 at 32, 0.56-0.71 at 64 and 0.57-0.73 at 128 rows (U_rand = 512 (1 - (1 - 10/512)^R), the
   random routing every LB and FlyTape figure so far used). Random-token prompts (the benchmark_serving cells the bars
   are measured on) and gsm8k are the most skewed, the mixed chat text (gsm8k / MATH / MBPP / MT-Bench) the least. Two rows share
   0.5-1.2 of their 10 experts (random: 0.195); the busiest expert of a step-layer takes 91-127 of 128 rows at worst (30-56 on average); a row keeps
   3.1-3.8 of its 10 experts from one step to the next. At K=3 (MTP, 4 rows per request) U / U_rand is 0.52-0.64 at
   32-128 rows. Middle and late layers are the most skewed (layers 15 / 16 / 31 / 32 at 0.41-0.45 of U_rand, rand1k r128).
2. **The capture is ATOM's own routing.** The stored ids / weights are the output of ATOM's `FusedMoE.select_experts`
   in its native path (`ATOM_MONO_ENABLE=0`, TP1, eager), hooked from outside ATOM (no edit). Gate: against an independent
   f32 top-10 of the captured logits, 0 non-tie differences in 56.2 M (K0) and 5.25 M (K3) rows; 1.6 % / 3.4 % of rows
   differ only by exact bf16 ties at the 10th place. Both must-fail capture mutants (ids + 1, rows rolled) were caught.
3. **The MoE under real routing** (B5 and the dispatcher picks, strict venv-built code objects, three jobs on both nodes,
   real and random routing paired in one shuffled loop with two random draws as the control): the cell's best arm under
   real routes / the dispatcher pick under random routes (rho) is
   - 0.93-0.99 at r8 (next to the random-draw band 0.96-1.03: not claimed),
   - 0.78-0.93 at r16, 0.74-0.90 at r32, 0.65-0.82 at r64, 0.68-0.82 at r128 (TP1) / 0.79-0.93 (TP2), every captured case
     a WIN against its own random reference (204 of 204 per job at r16+, three jobs),
   - with random-token or gsm8k prompts the lowest and the text mix the highest in every cell. K=3 routes at the same rows give
     0.62-0.83.
   Against FlyTape's experts + combine (same job, FlyTape traced before and after, random weights): 0.51-0.82 under real
   routes, 0.74-0.89 under random routes in the same jobs.
4. **The random-routing picks do not all transfer.** Under real routes the dispatcher's pick falls back to B5 or
   behind it at TP1 r16 (`t116b`: 0.99-1.19x B5 over the captured cases, three jobs; 0.91-0.94 on random routes) and
   TP2 r128 (`t2128a`: 0.99-1.10x; 0.91-0.96 random); `t232b` reaches 1.11x at TP2 r32 where `t232a` holds; `c132`
   reaches 1.015x on one case. Phase stamps show why: hot experts concentrate the work on a few CTAs (`t116b`: the median
   CTA's expert phase drops to 0.004 of the span while the max stays at 0.91; B5's deal spreads it). The other picks
   hold (0.81-0.97 of B5 on captured routes).
5. **Budget (DESIGN-LB 15.7) re-issued under real routing** (derived from the measured rho, no composed layer yet):
   the K0 LB / FlyTape step central is 0.70 (TP2 B8), 0.65-0.69 (B16), 0.68-0.71 (B32), 0.63-0.67 (B64),
   0.73-0.77 (B128); TP1 0.72-0.73 (B8), 0.68-0.74 (B16), 0.64-0.66 (B32), 0.63-0.69 (B64), 0.61-0.66 (B128 vs ATOM K3,
   derived as in 15.3). The B16-B128 cells gain 0.03-0.14 on W7's random-routing central, the B8 cells 0.01-0.03 (inside
   the routing-draw noise). TP1 B8 / B16, W7's thin cells, move from high-end margins 0.04 / 0.05 to 0.06-0.07 / 0.08-0.15.
   - FlyTape stays at its random-routing step (its bench has random weights); its own gain from real routing is not
     measurable. The bar's ATOM DP2 K3 arm is served with real weights, so against ATOM the real-routing figure is the
     matched one.
   - PLAN 20's TP2 B128 target (10,000+ tok/s total) needs LB / FT <= 0.774 of FlyTape TP2's step. W7's random-routing
     central (0.81) did not meet it; the real-routing central (0.73-0.77, range 0.66-0.84) does for random-token and
     gsm8k prompts and is at the line for the text mix. Derived, not measured: WP-L5's full TP2 pass measures it.

## 1. What was built (harness only; no rocKE or ATOM file changed)

`hackathon-prep/qwen38/truemono-lb/wp-l0c-route/`:

| file | what |
|---|---|
| `site/sitecustomize.py` | capture hook: a `sys.meta_path` finder patches `Qwen4ExpSparseMoeBlock.forward` (a forward hook on `gate`: step context, shared-gate logit, an independent f32 top-10) and `FusedMoE.select_experts` (the STORED ids / weights) right after ATOM's modules execute; tie-aware gate on the GPU; mutants `RTCAP_MUTANT=shift / row` |
| `capture.py`, `cap_job.sh` | ATOM offline engine, native path, TP1, `--enforce-eager --level 0`; closed batches of exactly CONC prompts x 256 greedy decode tokens (ignore_eos), workloads `rand1k` / `rand8k` (uniform random token ids, ISL 1024 / 8192), `gsm8k`, `mix` (gsm8k / MATH-500 / MBPP / MT-Bench turn 1, chat template); K0 and K3 (MTP) engines |
| `analyze.py` | U per layer per step, U / U_rand, pair overlap, 16-token items, max tokens per expert, top-16 share, temporal overlap, 4-step union; the gate verdict; sampled routes for the bench (8 steady steps x 48 layers per run) |
| `rt_bench.py`, `bench_job.sh` | the timing / numerics driver over WP-L2d's frozen harness (`pl4b_bench` / `pl4_ref`, unchanged): per cell the cases rnd1 / rnd2 (random, two seeds) and `<workload>:a` / `:b` (captured routes: bench layers from capture layers 0, 6, .., 42 of sampled step 2 / layers 3, 9, .., 45 of step 6), injected as router logits whose torch top-10 and softmax reproduce the captured ids and weights; injection gate + must-fail mutant (`RT_INJ_MUTANT=1`); WP-L2d's numerics and mutants per case; one shuffled loop over every (case, arm) graph; `RT_SAVE` / `RT_LOAD` + `RT_STRICT` prebuilt code objects; `--stamps`, `--route-div` (K3 routes) |
| `summarize.py`, `ft_compare.py`, `budget.py` | tables: ratios per case, same-job FlyTape e+c, the re-issued budget |

Stages (frozen copies, read-only): `w10-rt-lb2-rocke` / `w10-rt-lb2-h` = WP-L2d's `lb2-rocke` / `lb2-h` of 2026-11-02
(`moe_lb2.py` sha256 `155601151a23f699`, `moe_lb.py` `febf48f145d6a6c8`, `pl4b_bench.py` `56b85a1de410bbec`,
`pl4_ref.py` `612122f896083924`); `atom-rkrepro` (`qwen4_exp.py` `f00493e937a65d83`, `moe.py` `4d391c060b5405d2`, identical
to `dev/ATOM`'s); `w10-rt-routes` (k0 `routes.npz` `5dbd80387f1697a0`); `w10-rt-hsaco` (164 venv-built code objects,
tar `d0c8b318a9b71e64`). Harness at the timed runs: `rt_bench.py` `4a189666c3be0a0b`, `bench_job.sh` `ad899952546554fa`;
capture `sitecustomize.py` `c5281ca5f618b73e`, `capture.py` `a9a81cd200ab6e96`, `analyze.py` `48304981adb5423d` (re-analysis).

## 2. Jobs

| job | what | node, GPU bus | gates |
|---|---|---|---|
| 68251612 `w10-rt-cap2` | capture: 3 short gate runs (plain, `shift`, `row`), K0 (rand1k / gsm8k / mix at CONC 8-128, rand8k at 16 / 64), K3 (rand1k / mix at CONC 8-32) | n01-09, 95 | capture gate PASS (K0, K3, plain); both mutants CAUGHT |
| 68251769 `w10-rt-an2` | steady-state re-analysis (first 16 K0 / 4 K3 steps dropped) | n01-09 (CPU work) | - |
| 68251768 `w10-rt-sv1` | injection mutant; 10 cells (TP1 / TP2, r8-r128, fp8) x 8 cases, numerics + mutants + 16 rounds, venv COMGR, every code object saved | n01-09, 95 | injection mutant CAUGHT at every case; numerics: section 5.1 |
| 68251823 `w10-rt-ld-n01`, 68251824 `w10-rt-ld-n07` | the same on the saved objects (container, strict: 164 / 164 LOAD, 0 compiled), 24 rounds, FlyTape TP1 B8-64 / TP2 B8-128 traced before and after | n01-09 e5 (+95 FlyTape); n07-17 75 (+15) | as above |
| 68251972 `w10-rt-sv2` | stamps (TP1 r16, TP2 r64 / r128; B5 + pick); K3 routes at r32-r128 (16 rounds, numerics + mutants) | n01-09 | injection PASS; numerics PASS |

Timing discipline (DESIGN 9.1): every ratio is same-process, shuffled over all (case, arm) graphs per round, two metrics
(`gs` 8-layer walk, `launch` one launch per layer); controls: second graph `lb2b`, recompiled `lb2rc` and the second
random draw `rnd2`; win rule per case against the rnd2 / rnd1 band; three jobs on two nodes. SCLK sampled from sysfs
every 0.5 s: at most 3 samples per bench window below 0.97 of the 2400 MHz busy median (n07-17: one inside the TP1
r128 rounds). **Deviation:** 2-4 other jobs of ours ran on the node during each timed window (DESIGN 8.1 allows one;
`sacct` audit), plus CI jobs; the cross-node / cross-job agreement of rho (section 5) is the evidence that it did not
move the paired ratios.

## 3. Routing statistics (K0, steady state; random routing = 1 for the ratios, 0.195 for overlaps)

| workload | rows | U / U_rand (p10..p90) | per-layer range | 16-token items / random | max tokens on one expert (mean, max) | pair overlap | temporal overlap |
|---|---|---|---|---|---|---|---|
| rand1k | 8 | 0.838 (0.70..0.96) | 0.64..0.98 | 0.834 | 3.7, 8 | 0.90 | 3.69 |
| rand1k | 16 | 0.669 (0.53..0.81) | 0.43..0.87 | 0.668 | 7.7, 16 | 1.20 | 3.73 |
| rand1k | 32 | 0.614 (0.50..0.73) | 0.43..0.81 | 0.615 | 12.9, 31 | 1.01 | 3.72 |
| rand1k | 64 | 0.559 (0.44..0.68) | 0.38..0.78 | 0.569 | 25.8, 58 | 1.08 | 3.84 |
| rand1k | 128 | 0.569 (0.45..0.69) | 0.41..0.79 | 0.604 | 49.8, 113 | 1.04 | 3.76 |
| gsm8k | 8 | 0.813 | 0.68..0.93 | 0.809 | 4.4, 8 | 1.09 | 3.09 |
| gsm8k | 16 | 0.750 | 0.58..0.89 | 0.749 | 7.5, 16 | 0.94 | 3.20 |
| gsm8k | 32 | 0.659 | 0.49..0.82 | 0.661 | 13.9, 32 | 0.95 | 3.25 |
| gsm8k | 64 | 0.602 | 0.44..0.76 | 0.611 | 27.4, 64 | 0.97 | 3.18 |
| gsm8k | 128 | 0.605 | 0.44..0.76 | 0.638 | 55.5, 127 | 0.99 | 3.23 |
| mix | 8 | 0.903 | 0.86..0.96 | 0.898 | 3.1, 8 | 0.55 | 3.48 |
| mix | 16 | 0.848 | 0.78..0.93 | 0.846 | 4.8, 12 | 0.52 | 3.32 |
| mix | 32 | 0.768 | 0.66..0.88 | 0.767 | 8.7, 25 | 0.56 | 3.35 |
| mix | 64 | 0.714 | 0.60..0.85 | 0.715 | 16.2, 45 | 0.57 | 3.38 |
| mix | 128 | 0.732 | 0.63..0.84 | 0.746 | 30.6, 91 | 0.55 | 3.38 |
| rand8k | 16 / 64 | 0.698 / 0.569 | 0.47..0.90 / 0.38..0.78 | 0.697 / 0.578 | 7.2 / 25.1 | 1.07 / 1.03 | 3.79 / 3.84 |

- K=3 (verify rows = 4 per request, steady): rand1k 0.550 / 0.527 / 0.521 and mix 0.629 / 0.637 / 0.644 of U_rand at
  32 / 64 / 128 rows; the union over 4 consecutive K0 steps of a closed batch is 0.45-0.84 of U_rand(4R).
- Including the first decode steps (the chat template's shared first tokens) lowers U further at small rows (gsm8k r8:
  0.78 over all steps, 0.51 over 32 steps).
- The skewed layers repeat across workloads (per-layer U ratio correlation rand1k vs mix 0.62): layers 15, 16, 31, 32,
  39 are the most skewed for random-token prompts, 31 and 43-47 for text; layer 0 is the least skewed everywhere.
- The capture is at TP1; the router is replicated, so TP2 routes the same tokens to the same experts (up to bf16 ties).

## 4. Bytes floors with the measured U (`large_batch/bytes_model.py`, `step_bytes(..., u=U)`, checkpoint precision, p128)

| rows | MoE bytes, real / random (rand1k / gsm8k / mix) | step bytes, real / random, TP1 | TP2 |
|---|---|---|---|
| 8 | 0.84 / 0.82 / 0.90 | 0.89 / 0.87 / 0.93 | 0.89 / 0.88 / 0.94 |
| 16 | 0.67 / 0.75 / 0.85 | 0.76 / 0.82 / 0.89 | 0.76 / 0.82 / 0.89 |
| 32 | 0.62 / 0.66 / 0.77 | 0.71 / 0.74 / 0.83 | 0.71 / 0.75 / 0.83 |
| 64 | 0.56 / 0.60 / 0.71 | 0.68 / 0.71 / 0.79 | 0.68 / 0.71 / 0.79 |
| 128 | 0.57 / 0.61 / 0.73 | 0.72 / 0.75 / 0.83 | 0.72 / 0.75 / 0.83 |

The streamed floor of a TP2 B128 step falls to 0.72-0.83 of the random-routing floor of DESIGN-LB section 3.

## 5. The MoE re-timed

### 5.1 Numerics (every timed code object, random data, MI355X)

- Injection gate: the bench's host routes equal the captured ids (as sets per row) and weights (max error 5e-8) in every
  (cell, case) of every job; the must-fail mutant (one captured id moved) failed the gate in every case.
- WP-L2d's suite per case (T0 against float64, slot-order determinism over 3 launches, hdump / dirty-h / stale-read twins,
  2-layer walk, null build; must-fail mutants on rnd1 and the first captured case of every cell): PASS in 79 of 80
  (cell, case) checks in `sv1` and `ld-n07` and 78 of 80 in `ld-n01`; the exceptions, both reported, neither a kernel
  defect:
  - **TP2 r64 `mix:a`, every build, every job:** the E2E rule (kernel max error <= the bf16-per-slot oracle's max error)
    is exceeded, k/o 1.21-1.24, while the element-level bounds of the hdump twin pass (h codes 0 / 225,280 off their
    quantization neighbours, scale codes 0 / 7,040, N1b 1.2e-5 against 2^-11) and the host fold of the kernel's partials
    equals y bit for bit. The case has hot experts and peaked weights; the max-error heuristic is routing-sensitive.
  - **`flow0` (listed as the deterministic must-fail combine mutant) escaped once:** TP2 r8 `rand1k:a` in `w10-rt-ld-n01`
    (1 launch); caught in all other 59 runs across the three jobs, `flowrel` caught in all 60. At 8 rows with real
    routes the combine can find its partials complete without waiting; `flow0` is not deterministic there.
- The K3-routes job (`sv2`): every check PASS.

### 5.2 rho: the cell's best arm under real routes / the pick under random routes

Geometric mean of cases a and b, random reference = geometric mean of rnd1 and rnd2; median (min..max) over the three
jobs (sv1 venv n01-09, ld-n01, ld-n07 container strict), `launch` metric (`gs` within 0.03 of it).

| TP | rows | pick | random-draw control rnd2 / rnd1 | rand1k | gsm8k | mix |
|---|---|---|---|---|---|---|
| 1 | 8 | `t18b` | 0.960..1.027 | 0.936 (0.935..0.941) | 0.933 (0.928..0.938) | 0.955 (0.952..0.960) |
| 1 | 16 | `t116b` -> B5 | 0.974..1.025 | 0.779 (0.776..0.784) | 0.837 (0.830..0.843) | 0.915 (0.896..0.929) |
| 1 | 32 | `c132` | 0.989..1.002 | 0.768 (0.742..0.769) | 0.775 (0.764..0.785) | 0.837 (0.816..0.838) |
| 1 | 64 | `c164` | 1.002..1.007 | 0.656 (0.648..0.656) | 0.681 (0.678..0.683) | 0.798 (0.787..0.800) |
| 1 | 128 | `c164` | 1.004..1.009 | 0.702 (0.698..0.715) | 0.682 (0.681..0.713) | 0.806 (0.802..0.820) |
| 2 | 8 | `b28` | 0.979..0.988 | 0.958 (0.952..0.959) | 0.965 (0.956..0.985) | 0.963 (0.963..0.964) |
| 2 | 16 | `k216` | 0.958..0.993 | 0.801 (0.799..0.814) | 0.863 (0.854..0.879) | 0.897 (0.897..0.908) |
| 2 | 32 | `t232a` / `t232b` | 0.985..0.993 | 0.777 (0.770..0.791) | 0.828 (0.807..0.852) | 0.874 (0.857..0.900) |
| 2 | 64 | `b264` | 1.008..1.060 | 0.686 (0.668..0.702) | 0.708 (0.706..0.733) | 0.803 (0.799..0.818) |
| 2 | 128 | `t2128a` -> B5 | 0.962..1.043 | 0.816 (0.788..0.831) | 0.791 (0.790..0.829) | 0.889 (0.870..0.927) |

- Every captured case at r16 and above is a WIN against its own random reference under the win rule (rnd2 / rnd1 band),
  same arm, every job. At r8 the gain (0.93-0.97) sits next to the random-draw band and is not claimed.
- The routing draw matters even for random routing (rnd2 / rnd1 up to 0.95-1.06): single-draw random-routing figures
  (W7 and before) carry about +-0.05 from the draw alone.
- **K=3 routes** (sv2, venv, n01-09; rows = 4 x CONC): rho 0.744 / 0.618 / 0.679 (TP1 r32 / r64 / r128, rand1k), 0.777 /
  0.736 / 0.751 (mix); TP2 0.787 / 0.671 / 0.772 (rand1k), 0.824 / 0.768 / 0.833 (mix). Close to K0 at the same rows.
- Against the bytes-floor streamer of the same routed union, B5 is 1.27-1.32x (TP1 r32-r128) / 1.44-1.48x (TP2 r32-r128)
  under random routes and 1.39-1.96x under real routes (gs metric, `sv1`): the kernel loses efficiency on skewed routes,
  so there is more than the bytes in it (section 6).

### 5.3 Against FlyTape's experts + combine (same job, random weights; best arm, median (min..max) over cases a / b, both
nodes and both FlyTape brackets)

| cell | random routes (rnd1, rnd2) | rand1k | gsm8k | mix |
|---|---|---|---|---|
| TP1 r8 | 0.81-0.85 | 0.77 (0.76-0.79) | 0.77 (0.76-0.78) | 0.79 (0.76-0.82) |
| TP1 r16 | 0.86-0.89 | 0.68 (0.66-0.71) | 0.74 (0.72-0.75) | 0.80 (0.77-0.81) |
| TP1 r32 | 0.82-0.85 | 0.63 (0.60-0.67) | 0.64 (0.64-0.65) | 0.69 (0.67-0.71) |
| TP1 r64 | 0.82-0.84 | 0.54 (0.51-0.57) | 0.56 (0.55-0.57) | 0.66 (0.64-0.68) |
| TP2 r8 | 0.74-0.77 | 0.72 (0.70-0.74) | 0.73 (0.72-0.74) | 0.73 (0.71-0.75) |
| TP2 r16 | 0.79-0.81 | 0.64 (0.63-0.66) | 0.69 (0.67-0.70) | 0.72 (0.70-0.74) |
| TP2 r32 | 0.85-0.87 | 0.67 (0.66-0.70) | 0.71 (0.69-0.74) | 0.76 (0.72-0.79) |
| TP2 r64 | 0.76-0.82 | 0.54 (0.52-0.55) | 0.56 (0.55-0.58) | 0.63 (0.61-0.68) |
| TP2 r128 | 0.77-0.83 | 0.65 (0.61-0.69) | 0.65 (0.63-0.66) | 0.72 (0.70-0.74) |

The random-routing column is 0.03-0.08 below P-L4b 7.6 in most cells (venv-built objects here, container-built there;
FlyTape's own time moves up to 12 % between jobs). The real-routing columns compare our real-routes MoE with FlyTape's
random-routes MoE: a statement about the serving workload, not a matched kernel comparison.

## 6. Why some picks lose under real routes (stamps, `sv2`, venv; stamped builds can allocate registers differently)

| cell | arm | case | span / same arm on rnd1 | per-CTA expert phase, median / max (of the span) |
|---|---|---|---|---|
| TP1 r16 | B5 | rnd1 / rand1k:b / mix:b | 1 / 0.709 / 0.804 | 0.868 / 0.943; 0.632 / 0.907; 0.652 / 0.917 |
| TP1 r16 | `t116b` | rnd1 / rand1k:b / mix:b | 1 / 0.908 / 0.944 | 0.842 / 0.909; 0.004 / 0.907; 0.004 / 0.907 |
| TP2 r128 | B5 | rnd1 / rand1k:b | 1 / 0.724 | 0.854 / 0.939; 0.759 / 0.916 |
| TP2 r128 | `t2128a` | rnd1 / rand1k:b | 1 / 0.884 | 0.886 / 0.937; 0.554 / 0.936 |
| TP2 r64 | B5 / `b264` | rnd1 / rand1k:b | 1 / 0.749; 1 / 0.719 | 0.834 / 0.943, 0.767 / 0.913; 0.847 / 0.937, 0.799 / 0.906 |

`t116b` deals whole wave-owned experts (`gu=wt`): with fewer, hotter experts most CTAs get no expert work and a few
carry the hot ones (median 0.004, max 0.91). Under `t2128a` (`dt=5`) the median CTA's expert phase ends at 0.55 of the
span on the most skewed case while the last CTA runs to 0.94. B5's round-robin deal with split last-round pieces spreads the hot experts better. The span is set
by the CTA with the hot expert in every skewed case: a hot expert of 50-127 tokens is 4-8 passes of 16 tokens that one
CTA streams while its weights are read once; splitting such an expert over CTAs by passes is the missing lever.

## 7. Budget (DESIGN-LB 15.7) under real routing

Method: 15.7's LB / FT step per cell, with the MoE term scaled by the measured rho. The W7 MoE term is the pick's e+c
x f (f = 1.03-1.17 for the in-kernel top-k); the top-k part does not depend on the routing, so the term scales by
(rho + f - 1) / f: LB/FT(real) = LB/FT(W7) x (1 - s (1 - (rho + f - 1) / f)), s = 15.7's MoE share of the LB step.
Central: W7 central, median rho, f = 1.10; range: W7 low with min rho and f = 1.03 .. W7 high with max rho and f = 1.17.
`budget.py` over the three jobs; `launch` metric.

| TP | B | bar (1k / 8k) | W7 K0 central (range), random routing | real routing: rand1k | gsm8k | mix |
|---|---|---|---|---|---|---|
| 2 | 8 | 0.917 / 0.890 | 0.71 (0.65-0.78) | 0.70 (0.63-0.77) | 0.70 (0.64-0.77) | 0.70 (0.64-0.77) |
| 2 | 16 | 0.974 / 0.962 | 0.72 (0.66-0.79) | **0.65** (0.59-0.73) | 0.67 (0.61-0.75) | 0.69 (0.63-0.76) |
| 2 | 32 | 1.000 / 1.000 | 0.76 (0.69-0.83) | **0.68** (0.61-0.75) | 0.70 (0.62-0.77) | 0.71 (0.64-0.79) |
| 2 | 64 | 0.993 / 0.997 | 0.74 (0.67-0.80) | **0.63** (0.56-0.70) | 0.64 (0.57-0.71) | 0.67 (0.60-0.74) |
| 2 | 128 | 1.000 / 1.000 | 0.81 (0.74-0.87) | **0.74** (0.66-0.80) | 0.73 (0.66-0.80) | 0.77 (0.69-0.84) |
| 1 | 8 | 0.912 / 0.852 | 0.75 (0.68-0.81) | 0.72 (0.65-0.78) | 0.72 (0.65-0.78) | 0.73 (0.66-0.79) |
| 1 | 16 | 0.914 / 0.923 | 0.78 (0.71-0.86) | **0.68** (0.61-0.76) | 0.70 (0.63-0.79) | 0.74 (0.66-0.83) |
| 1 | 32 | 0.955 / 0.951 | 0.73 (0.65-0.81) | **0.64** (0.55-0.71) | 0.64 (0.56-0.72) | 0.66 (0.58-0.74) |
| 1 | 64 | 0.987 / 0.979 | 0.77 (0.68-0.85) | **0.63** (0.54-0.70) | 0.64 (0.55-0.71) | 0.69 (0.60-0.76) |
| 1 | 128 | ATOM K3 | 0.74 (0.66-0.82) vs ATOM K3 | **0.62** (0.55-0.70) | 0.61 (0.54-0.70) | 0.66 (0.59-0.75) |

- rand1k is the bars' own workload (benchmark_serving random prompts): bold. Every range is under its bar, the B8 cells
  move by only 0.01-0.03 (the routing gain at 8 rows is within the random-draw band).
- The MoE share of the LB step falls from W7's 0.50-0.65 to about 0.42-0.63 (0.42-0.48 at TP2 B16-B128 rand1k, 0.63 at
  TP1 B8): the GDN family and the TP2 exchange now carry more of the step.
- **TP2 B128 and PLAN 20 (10,000+ tok/s on 2 GPUs):** that target is LB / FT <= 0.774 of FlyTape TP2's step. Random
  routing (W7) put the central at 0.81; real routing puts it at 0.73-0.77 (range 0.66-0.84). With `combine=fused_next`
  (0.917-0.96 of the MoE chain) the central drops by about another 0.02-0.03. Still a derived number: WP-L5's measured
  TP2 pass is the claim, and it should be timed on captured routes (this package's stage) as well as random ones.
- Not in this budget: a composed layer; the TP2 sync tail; real routing's effect on FlyTape (unmeasurable with its
  random-weight bench); per-layer routing variation inside a step (the bench walks 8 of the 48 layers per case: 16 of 48
  per cell over cases a and b).

## 8. What this changes for the other packages

1. **WP-L2d / the MoE agent (read-only here):** the dispatcher must be tuned on real routes. The TP1 r16 and TP2 r128
   picks lose 3-15 % to B5 on every captured case; a re-tune (`lb2_tune` over `rt_bench`'s cases, the routes stage
   `w10-rt-routes` with both K0 and K3 routes) and a hot-expert split (passes of one expert over several CTAs; DESIGN-LB
   6.2's union-aware placement) are the levers. In these jobs the best arm is 0.51-0.82 of FlyTape's random-routes e+c
   under real routes in every cell (under the 0.85 target everywhere) and 0.74-0.89 under random routes (TP1 r16 and
   TP2 r32 above 0.85). `flow0` is not a deterministic must-fail mutant at r8 on real routes (section 5.1).
2. **WP-L3 / WP-L4 / WP-L5:** time composed layers and the TP2 pass on captured routes next to random ones; the GDN chain
   and the exchanges are now a larger share of the step.
3. **WP-L9 (MTP, TP1):** K=3 verify rows are correlated within a request (temporal overlap 3.1-3.8 of 10 per step);
   rho at the K3 rows is 0.62-0.83, so MTP's MoE cost per token is lower than random-routing estimates.
4. **DESIGN-LB section 3 note and `bytes_model.py`:** measured U per workload is now available (section 3 table and
   `analysis.json` per run); floors with real U in section 4.

## 9. Not done / open

- WaveScope: no kernel was changed, so there is no before / after; phase stamps (section 6) were used for the
  attribution instead. An ATT capture of B5 vs `t2128a` on `rand1k:b` would pin the idle CTAs' stall classes.
- DESIGN 8.1: 2-4 of our jobs beside every timed window (section 2).
- fp4 (W4A4) cells were not re-timed (fp8 / W4A8 only, the checkpoint precision).
- Greedy decoding only; sampled decoding may spread routes further.
- FlyTape under real routing (would need its bench on the real checkpoint).

## 10. Reproduce

```
H=hackathon-prep/qwen38/truemono-lb/wp-l0c-route
# capture (container, 1 GPU; ~30 min)
q38_job.sh --tag <t> --mode container --stage atom-rkrepro --stage <harness stage> --env RT_H=<harness stage> \
  --env RT_STEPS=mut,k0,k3 --env 'RT_RUNS_K0=rand1k:8,16,32,64,128;gsm8k:8,16,32,64,128;mix:8,16,32,64,128;rand8k:16,64' \
  --env 'RT_RUNS_K3=rand1k:8,16,32;mix:8,16,32' -- 'bash /job/stages/<harness stage>/cap_job.sh'
# routes stage: tar k0/routes.npz k3/routes.npz from the results dir into q38/stages/w10-rt-routes.tar (+ .sha256)
# timing: venv save (RT_STEPS=injmut,save), then container strict load with FlyTape brackets (RT_STEPS=load);
#   RT_ARGS = $H/logs/rtargs1.txt / rtargs2.txt; K3 routes: RT_STEPS=plain2 RT_RFILE2=k3/routes.npz (rtargs_k3.txt)
python $H/budget.py <rt_bench.json ...> --launch
```

## Verification (independent, W10, 2026-11-02)

**Verdict: confirmed, with the caveats below.** I tried to refute the package and could not. Every table matches the raw
data. An independent re-time on route samples the builder never timed reproduces rho, the budget and the two pick losses.
This is still not a measured 10,000+ tok/s. The TP2 B128 figure stays a derived budget until WP-L5 measures it.

What I checked (harness `hackathon-prep/qwen38/truemono-lb/vwp-l0c-route/`, raw data in the private-perf
`WP-L0c-route-raw/verify-n01/`):

1. **Raw data to report.** `budget.py --launch` over the three timed `rt_bench.json` files reproduces section 5.2's rho
   and section 7's table to every printed digit. The steady-state analysis output (`an2`) matches section 3. The capture
   job log (68251612) shows the K0 and K3 gates PASS (0 non-tie rows), `shift` / `row` CAUGHT, and every decode record
   sourced from ATOM's `select_experts`. Both strict container ledgers show 164 / 164 LOAD with sha256 values equal to
   the venv SAVE ledger. The harness hashes in the job logs (`rt_bench.py` `4a189666` for sv1 and the ld jobs, `c3741294`
   for sv2) match section 1. The pick specs are byte-equal to P-L4b's `f10` / `f14` arms. W7's constants in `budget.py`
   equal DESIGN-LB 15.7. 0.774 is 7,742 / 10,000.
2. **Selection bias in "best arm".** rho takes, per round, the minimum of B5 and the pick. Using a fixed arm chosen by
   its median instead gives the same rho to 3 decimals in every cell and job. The arms are separated by much more than
   the control noise (lb2b / lb2 sd 0.003-0.012).
3. **Are the timed route samples representative?** For each case I compared its U / U_rand with the steady-state
   population mean over the same 8 capture layers. The difference is -0.05 to +0.11. On average the timed samples are
   slightly less skewed than the population, so rho is if anything slightly pessimistic.
4. **Re-timing on new routes (job 68252199, n01-09, bus 95, venv).** All 61 code objects were loaded strict from
   `w10-rt-hsaco`, with 0 compiled. The route samples are new: sampled steps 1 / 4 instead of 2 / 6, and layer offsets
   1 / 4 instead of 0 / 3, so none of these layers were in the builder's timing. Seed 4242 gives new weights and new
   random draws. There were 24 shuffled rounds over TP1 r16, TP2 r64 and TP2 r128, using the frozen `rt_bench.py`
   unchanged through a wrapper.
   - Gates: injection PASS 18 / 18; WP-L2d numerics PASS 24 / 24, including TP2 r64 `mix` on the new sample (consistent
     with the builder's reading that the `mix:a` max-error exceedance is sample-specific); every required mutant caught,
     `flow0` included; every captured case a WIN against its random reference (both metrics). SCLK: busy median 2400 MHz,
     4 of 157 busy samples below 0.97 of it.
   - rho (launch; rand1k / gsm8k / mix): TP1 r16 0.767 / 0.829 / 0.880 (report 0.779 / 0.837 / 0.915); TP2 r64
     0.648 / 0.706 / 0.813 (report 0.686 / 0.708 / 0.803); TP2 r128 0.752 / 0.823 / 0.871 (report 0.816 / 0.791 / 0.889).
   - Re-derived K0 central: TP2 B128 0.71 / 0.74 / 0.76 (report 0.74 / 0.73 / 0.77); TP2 B64 0.62 / 0.64 / 0.68; TP1 B16
     0.67 / 0.70 / 0.72. The `gs` metric gives TP2 B128 0.72 / 0.75 / 0.77.
   - Pick transfer, reproduced:
     - `t116b` / B5 is 1.05-1.18 on all 6 captured cases (LOSS) and 0.92-0.93 on random routes.
     - `t2128a` / B5 is 1.03-1.10 on all 6 (LOSS) and 0.96 on random routes.
     - `b264` holds on all 6 (0.94-0.95, WIN).
5. **The 10k line without re-dispatch.** Section 7 uses the cell's best arm, which at TP2 r128 is B5, not the current
   dispatcher pick. With the shipped pick `t2128a` under real routes (median over the three jobs), the TP2 B128 central
   becomes about 0.76 (rand1k), 0.76 (gsm8k) and 0.775 (mix). So the "meets 0.774 for random-token prompts, at the line for
   the text mix" conclusion holds either way. Re-dispatching TP2 r128 and TP1 r16 to B5 is still worth 0.01-0.02 of the
   step.

Caveats (none refutes the package):
- **Not measured.** PLAN decision 20 asks for a measured 10,000+ tok/s. The real-routing budget only says the target is
  reachable on paper: central 0.71-0.77 against a 0.774 line, with a range of 0.65-0.84 that straddles it.
- **Numerics.** The builder's two exceptions stand as reported. On TP2 r64 `mix:a`, the kernel's mean error is 0.41x the
  oracle's while its max is 1.21x. `flow0` escaped once, at r8. Neither reproduced on the new samples. Both are
  exceptions to the written rule, accepted on evidence, not passes of it.
- **DESIGN 8.1.** My run shared n01-09 with 2 of our jobs and 4 others (the same deviation as the builder's). n07-17 was
  `down*` during my window, so the re-time is single-node. The builder's three jobs cover both nodes, and my ratios fall
  inside or next to their spread.
- **Limited coverage.** K3 rho (section 5.2) comes from one venv job on one node. fp4 cells were not re-timed, and only
  greedy decoding was captured.
