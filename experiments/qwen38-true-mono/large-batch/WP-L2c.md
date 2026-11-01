# WP-L2c: the large-batch GDN family (in-proj + conv / gates, recurrence + out-proj hand-off, out-proj) in one persistent chain, rows 5-128, TP1 and TP2, gfx950

W7, 2026-11-01. `../DESIGN-LB.md` 15.6 (WP-L2c), 4.2 (S3-S5), 6 (E3 / E4 / E5), 6.1 (the recurrence); `P-L1.md` (the
recurrence this productizes), `WP-L2a.md` (the E1 GEMM core it builds on), `WP-L1.md` (the framework and its sync
verdicts), `P-L2.md`; `../DESIGN.md` 12.3 N11 and 12.4 (the T ladder). Ratios only here. Absolute times, job logs,
WaveScope captures and raw JSON: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2c.md` and
`.../large-batch/wp-l2c-raw/`. Per-cell tables and the lever ledger: `WP-L2c-tables.md`.

## Gate and verdict

**Gate: met** (DESIGN-LB 15.6: T0 against fp64 plus bitwise against the exact-order emulation; mutants, stale-read twins
and determinism on MI355X before any timing; the `state=bf16` knob with its accuracy plan).

- **The whole GDN family runs in one persistent launch per step** (`q38lbg_*`, 256 CTAs x 512 threads, a runtime layer
  loop on W5's `persist/` platform): S3 (the qkvz in-projection with the causal-conv + SiLU epilogue and the conv-state
  shift; the ba projection with the gates epilogue), S4 (P-L1's recurrence, which also writes the out-projection's MXFP8
  input in E1's op_sel layout), S5 (the out-projection), at rows 5-128, TP1 (48 / 16 heads) and the TP2 rank shape
  (24 / 8 heads), f32 or bf16 recurrent state.
- **Against FlyTape's GDN phases in the same job** (its traced `gdn_proj` + `gdn_recur` + `gdn_out` (+ `gdn_out_reduce`
  at TP1) per layer, before and after the LB timing; f32 state on both sides): see section 4.1 for the per-cell table.
  - **f32 state: TP1 0.44-0.69 of FlyTape's GDN family (rows 5-64), TP2 0.73-0.82 (rows 5-128).** FlyTape's two
    brackets agree within 0.025 (and with an earlier job's within 0.02).
  - Per stage, in situ: in-proj with the conv / gates epilogue 0.32-0.37 (TP1) / 0.51-0.71 (TP2) of `gdn_proj`;
    recurrence 0.51-0.86 / 0.75-0.87 of `gdn_recur`; out-proj 0.37-0.50 / 0.54-0.85 of `gdn_out`.
  - With `state=bf16` (N11, default off, accuracy-gated): TP1 0.41-0.48, TP2 0.57-0.77.
- **The lookback climb** (6 rounds, both rocke nodes, PLAN 13 / 15): the final builds run at **0.69-0.92 of the first
  gated build** of each cell (same job, both nodes; the first build is 1.09-1.45x the final). 25 changes were accepted,
  every one a WIN on both nodes; the last round accepted nothing.
- **Numerics.** 314 cell checks over 24 jobs on both nodes (12 cells x f32 / bf16 state in every final run), 0 failures;
  2,826 of 2,826 must-fail mutant runs caught; 314 of 314 deterministic stale-read twins fail; the composed kernel
  equals the standalone stage kernels bit for bit in 942 of 942 stage-isolation checks; 12,063 lever builds checked
  against their base (bitwise, or stage checks + determinism + both walks for order-changing levers), 1 unexplained
  failure (section 2.4); determinism stress: 0 differing runs in 12,812 reruns of 180 builds.
- **`state=bf16` (N11)** passes T0 everywhere (state within the bf16 storage bound, output bitwise to its own
  emulation) and runs at **0.68-0.97 of the f32 build** (same job, interleaved, both nodes and the bracket job). It stays **default off**: the drift run
  (2,048 steps in place through the whole chain) shows the long-memory error growing with the step count (state error
  0.003 -> 0.15 of the largest state, out-projection error 0.002 -> 0.05), so T2 / T4 (RULER at 128K / max) decide it;
  they were not run here (section 4.5 gives the plan).
- **Not met / not done**: no quick end-to-end number (PLAN 14; no served path runs the LB chain before WP-L6); the
  partial edges of DESIGN-LB 4.2 (S3 -> S4 per head, S4 -> S5 per head group) are not built (WP-L1 measured the grid
  edge as the idle winner; the per-edge decision is WP-L3's); the TP2 out-projection exchange is not in this chain
  (WP-C5 / WP-L5); T2 / T4 for `state=bf16`.

## 1. What was built

| piece | file |
|---|---|
| the GDN family: `GdnShape`; E3 (`E3Knobs`, qkvz in-proj on E1 + conv / SiLU epilogue + conv-state shift; epilogue thread maps `cepi` = `lane` (MFMA lane: 4 channels x 1 token), `lds` (bf16 Y through LDS, 8 consecutive channels per thread), `batch` (the lane map with all of a unit's table / conv-weight / conv-state loads issued before any math, `cgrp` per group); `cpre` (the unit's table entries and conv weights at unit start)); E3g (`E3gKnobs`, the bf16 ba projection + gates epilogue, ATOM's `sigmoid_fast` / `softplus_fast` order); E4 (`E4Knobs`, P-L1's recurrence whose unit finisher also writes the bf16-rounded output as MXFP8 codes + E8M0 scales in E1's op_sel layout: `hand="mx8"`); E5 (an E1 build reading the hand-off with no repack); the composed chain (`ChainKnobs`: `stages`, `acq` = `load` / `wave` / `cta`, `pre` = `none` / `s5` / `s4` / `s45` + `warm`, `lpad` = `entry` / `fit`, `delay`, mutants `early` / `noacq`, `hz`) | `dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/qwen38_mono/large_batch/gdn_lb.py` |
| the deterministic stale-read harness for the chain's acquire forms (positives `load`, `wave`, `cta`; the XCD-leader form measured unsound and refused; twins `xcd_nol1`, `xcd_nol2`, `plain`) | `.../large_batch/gdn_lb_stale.py` |
| CPU tests (60): the parents (`e1.py`, `gdn_recur.py`) lower to the same IR before and after composing; E4 with the hand-off off is P-L1's kernel; every stage / chain variant lowers (TP1 / TP2, every bucket, dbg twins, all mutants, ablations, every lever); refusals; LDS pool of `lpad="fit"`; the S4 prefetch off leaves the base IR unchanged | `.../rocke/library/tests/test_qwen38_lb_gdn_lb.py` (P-L1's 123 and WP-L2a's tests pass unchanged: 255 in the three files) |
| bench (numerics suite, stale-read harness, drift, timing with the win rule, determinism stress, levers / combinations / bases from files, code-object dump / load), compile gate, job bodies, staging, climb bookkeeping, reports, WaveScope launcher | `hackathon-prep/qwen38/truemono-lb/wp-l2c/`: `l2c_bench.py`, `gate.py`, `job.sh`, `sub.sh`, `sub_ct.sh`, `sync_l2c.sh`, `mk_job.sh`, `climb.py`, `levtab.py`, `ledger_sum.py`, `num_sum.py`, `final_report.py`, `ft_gdn.py`, `att_l2c.py`, `att_job.sh`, `att_sum.py`, `cfg/` (bases `base_r0` .. `base_final`, combinations, the ledger) |
| alola facts (COMGR's silent LDS-limit error, the packer and the exclusive pad, compile-crash isolation, Markham 20-min cap, budget accounting of pending jobs) | `.claude/skills/alola-cluster/SKILL.md`, rules 201 and 205 |

No existing file's builds changed: `gdn_recur.py` (sha `d74fee15`, as P-L1 timed it), `gdn_recur_ref.py`, `gemm.py`
(`39e27905`, as P-L3 timed it) and `insitu*.py` are untouched; `e1.py` / `e1_plans.py` are WP-L2a's and are imported.
Every knob added to `gdn_lb.py` after the first measured builds is default-off and was checked IR-identical (names
included: the name hash skips late fields at their defaults) on 112 chain builds per change. Python engine only; names
`q38lb_e3*`, `q38lb_e4*`, `q38lbg_*`; nothing served imports the module.

**How the stages compose.** The stage classes are subclasses of E1 and of P-L1's recurrence, bound to one builder
through a scope that maps the parent's kernel parameters to per-layer values, suffixes loop names and drops `ret`.
Each stage instance is built inside the layer loop with opaque per-instance thread ids, so no stage's entry geometry is
held (and spilled) across the others (alola 201). Edges are W5's striped grid sync (8 stripes per XCD, abort and
launch-fatal on timeout), with the residency ticket and the launch-fatal word checked after every launch; consumers of
in-launch data read with `sc1` loads (`acq="load"`), so no edge invalidates a cache.

**In situ, and the WP-L1 framework.** The chain is the in-situ form of the three stages on W5's `persist/` platform
(the same `GridSync`, ticket and launch-fatal word WP-L1 imports). It does not run inside `lbf.py`: that kernel's stage
interface carries 512-word synthetic payload units and host unit tables, not E1 / recurrence stage bodies, and WP-L1
measured its per-edge flags (`sync=mix`) at 1.010-1.011 of its grid barrier at idle arrivals, so the grid edge is the
measured default for a uniform chain like this one; the per-edge choice (and the partial edges) is WP-L3's in-situ
decision. The edge model is the grid edge, and its cost is measured separately (`edges` arm: the chain with every stage
ablated).

## 2. Numerics

### 2.1 References and rules (every timed binary, in the same process, before its timing)

Synthetic inputs with ATOM's layouts: MXFP4 weights (random codes, E8M0 scales 119-121), MXFP8 activations by the host
`mx_quant` rule, bf16 ba weights and x, conv weights, `A_log` and `dt_bias` per head, per-row table
`(conv_in, conv_out, rec_in, rec_out)`, conv state `[3][qkv]` bf16, recurrent state `[hv][v][k]` f32 or bf16; 2 layers.
Each stage is checked from the kernel's own upstream outputs (T0):

| stage | check | rule |
|---|---|---|
| S3 qkvz + conv | z rows and the conv state's new tap vs float64 of the dequantized operands; the conv pre-activation bitwise vs its f32 emulation (dbg twin); SiLU output vs float64 of the kernel's own conv | WP-L2a's metric <= 1 eps (2^-11 of sum\|x w\|); conv bitwise; SiLU <= 1 bf16 ulp; state shift exact |
| S3g ba + gates | bf16(b), bf16(a) vs float64 (dbg twin); beta and g vs their f32-order emulation with correctly rounded transcendentals | ba <= 1 eps (2^-14); beta <= 8 f32 ulp; g <= 1 unit of (8 ulp + A 2^-22) |
| S4 recurrence | P-L1's N1 (output per head vs fp64, state vs fp64) and N2 (bitwise vs the exact-order emulation of the build's order; the dbg twin's dumped transcendentals within 4 ulp) | as P-L1 (f32 state: 4x the float32 torch reference's error; bf16 state: 2^-8) |
| hand-off | MXFP8 codes and op_sel scales vs the host rule applied to the kernel's bf16 output | bitwise |
| S5 out-proj | vs float64 of the kernel's own hand-off | <= 1 eps |

Cell-level checks: the timed binary equals its dbg twin; 3 runs bitwise equal; one launch per layer equals the layer
loop; a recompiled copy equals; the standalone stage kernels (`layers=1` bring-up form) give the chain's bits; and each
chain stage run alone over the standalone stages' upstream reproduces them bit for bit (stage isolation). Levers that
keep the order must equal the base bit for bit in both walks; order-changing levers (other plans, combinations) pass
the stage checks, determinism and both walks.

### 2.2 Must-fail mutants and the stale-read twin

| mutant | what it breaks | caught by |
|---|---|---|
| E3 `tap` | conv tap 0 dropped | S3 |
| E3 `state` | conv state not shifted | S3 |
| E3 `kskip` (E1's) | last K chunk skipped | S3 |
| E3g `dtb` | `dt_bias` dropped | S3g |
| E3g `swap` | beta stored in g's slot | S3g, S4, hand-off, S5 |
| E4 `nodecay` (P-L1's) | decay skipped | S4 |
| E4 `hscale` | hand-off scale one binade high | hand-off |
| E4 `hpair` | code bytes of a lane pair swapped | hand-off |
| E5 `kskip` | last K chunk skipped | S5 |
| chain `early` (gating) | every grid edge released one arrival early while CTA 0 starts every stage late | outputs differ (the twin's positive control `delay` keeps the bits) |
| chain `noacq` (informational) | plain loads of in-launch data | not caught (a miss proves nothing, alola 182) |

2,826 of 2,826 mutant runs caught and 314 of 314 `early` twins failing, in every job on both nodes; `delay` kept the bits
in all 314.

**The acquire forms** (`gdn_lb_stale.py`, consumers pretouch before the producer may store): `load`, `wave` and `cta`
read 0 stale words in every launch at XCD distance 0, 1 and 3; the twins `xcd_nol1` / `xcd_nol2` / `plain` read stale
in 0.94-1.00 of (trial, consumer) pairs; an XCD-leader form (one L2 invalidate per XCD, followers `buffer_inv sc0`) reads
stale in 0.94 and is refused by the chain's knob check.

### 2.3 Results

| quantity | result |
|---|---|
| cell checks (tp x rows x state; 24 jobs, both nodes) | 314 / 314 PASS |
| worst S3 / S5 error | 0.0022 / 0.006 eps (rule 1) |
| SiLU | <= 0.5 bf16 ulp from float64 of the kernel's own conv; conv pre-activation bitwise |
| beta / g (dbg twin) | 2 f32 ulp / 0.52 (rule 1) |
| bf16 state (rule 2^-8 = 3.9e-3) | 3.6e-3 worst |
| S4 output / bound; state (f32) | <= 0.5 of the bound; <= 0.22 of the bound (1e-7 relative, the float32 reference's own) |
| S4 bitwise vs emulation, timed == dbg twin, hand-off bitwise | every cell |
| stage isolation (chain stage == standalone stage) | 942 / 942 |
| lever builds checked | 12,063; 1 failure (2.4) |
| determinism stress (section 3) | 0 of 12,812 reruns differ (180 builds) |

### 2.4 One unexplained lever failure

In job r2b (n07-17, GPU bus 05) the combination `e5f12 + pre_s45` of TP1 rows 128 failed its lever check once (the
harness then printed only FAIL); it was not timed. It did not recur: the same job's seed on n07-17 (x3b, bus 75), and 255
reruns of it and of its parts (`e5f12`, `pre_s45`, `pre_s5` and three neighbouring combinations) under the stress
protocol (x2b, bus 85: alternating walks, every fourth run with CTA 0 late) gave 0 differing runs and passing stage
checks. No final build contains `e5f12`. The harness now prints which check fails.

## 3. Method

- **Cells.** TP1 and the TP2 rank shape x rows 5, 8, 16, 32, 64, 128 (buckets 16-128; the live row count is a kernel
  argument) x f32 / bf16 recurrent state. FlyTape has no TP1 B128 and no rows-5 configuration: rows 5 is compared with
  its B8, TP1 rows 128 only within LB.
- **Arms** (one HIP graph each, timed by events, per layer): `chain` (one launch walks L layers; L rotates at least 4x
  the MALL of weights and state), its second graph and a recompiled copy (the controls), `s3` / `s4` / `s5` (the chain
  with one stage's work, every edge kept), `edges` (no stage work), `chain:launch` (one launch per layer),
  `split:launch` (the four standalone stage kernels per layer: the `layers=1` bring-up), the streamed standalone kernels
  (`e3`, plain E1 on the same plan, `e4`, P-L1's kernel, `e5`), every lever / combination of the round, the first base
  (`combo_r0base`) and the other state's final build on its own data (`xstate`).
- **Discipline** (DESIGN 9.1): numerics of every timed binary first, in the same process; arm order shuffled every
  round (16-24 rounds); the win rule against both controls (WIN when the median ratio is below both controls' minima by
  0.002 and at least 3/4 of rounds are below 1); node and GPU bus recorded; every GPU health-checked (n07-17 bus e5
  skipped); the asm-hazard pad on and the hazard scan + spill check as a compile gate (a refused build is never timed);
  launch-fatal word after every launch and replay. Determinism stress: 24-256 reruns per build, alternating walks, every
  fourth run with CTA 0 late, bitwise against run 0.
- **The climb** (PLAN 13 / 15). Round 1 screened, on both nodes, every lever solo (12 planner E3 plans and 8 E5 plans
  from WP-L2a's `e1_plans.candidates` at the chain's 8 waves (4-wave plans converted), 21 / 33 grid-fill plans with
  16-64-row units, the conv epilogue forms, the S4 / S5 prefetches, the LDS pad, the recurrence depth, `acq`) and the
  best category combinations (losers inside the top 3 of a category included). A change was accepted only when it
  WON on both nodes; every rejected lever was re-tested on every new base, with removals (`cepi_lane`, `pre_none`,
  `lpad_entry`) and new combinations in different orders. Rounds 3-6 ran on the accepted bases; round 6 accepted
  nothing.
- **Jobs** (MI355X; q38 venv unless stated): first bases 68237886-68238366; screens 68240682 (s4a), 68240789 (r1a),
  68240866 (r1f), 68240913 (r1b); climb rounds 68240971 / 68241028, 68241129 / 68241130, 68241230 / 68241316,
  68241512 / 68241513, 68241562 / 68241563; bf16 + drift 68240790; stress 68241362, 68241401; final confirmation
  68241619 (n01-09) / 68241620 (n07-17), the code-object run 68241748 and the same-job FlyTape bracket 68241852 (q38
  container); WaveScope on Markham MI350X 68238273 and 68241125-68241140.
- **Compilers.** The venv's COMGR (torch 2.12, llvm22) compiled every timed build. The container's image COMGR compiled
  the E3 `kskip` mutant of TP1 rows 8 into a kernel that faults on the GPU (job 68241618: a memory fault that ended the
  process; the same mutant built by the venv COMGR runs and is caught in every venv job). The FlyTape-bracket job
  therefore loads the venv-compiled code objects of the final run (`--hsaco-dir`): the bracketed LB binaries are byte
  for byte the measured ones. The image-COMGR fault is an open item (section 8).
- **A harness bug, fixed.** The first final runs (68241585 / 68241586) timed the other state's build as a lever of the
  cell, i.e. an f32-state kernel on bf16 state buffers: out-of-bounds, a GPU memory fault that ended both processes.
  The bench now refuses any lever whose state differs from the cell's and times the other state's build on its own data
  (`xstate`, a second cell with the same seed); the final runs were repeated (68241619 / 68241620).

## 4. Results

### 4.1 Against FlyTape's GDN phases (same job)

Same-job bracket (job 68241852, q38 container, n01-09, GPU bus 05; the LB binaries are the venv-built code objects of
job 68241748, section 3). Ratios: LB time per layer over FlyTape's per-layer phase time before (pre) and after (post)
the LB timing. Stage rows use the in-situ stage time (the chain with only that stage's work minus the edges-only chain);
FlyTape's phases include their own grid-sync wait. The verdicts (first-base and other-state columns) are the win rule
against the chain's controls (LOSS = the arm is slower than the final chain); the last two columns are medians.

| TP | rows | state | chain / FT GDN family (pre; post) | S3 / FT gdn_proj | S4 / FT gdn_recur | S5 / FT gdn_out (+ reduce at TP1) | edges / chain | first base / final (same job) | other state / this | per-stage launches / chain | one launch per layer / chain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 (FT B8) | f32 | 0.437; 0.435 | 0.318; 0.322 | 0.514; 0.508 | 0.439; 0.425 | 0.160 | 1.190 LOSS | 0.949 WIN | 1.380 | 1.279 |
| 1 | 8 | f32 | 0.515; 0.514 | 0.344; 0.348 | 0.759; 0.750 | 0.503; 0.486 | 0.139 | 1.165 LOSS | 0.916 WIN | 1.368 | 1.243 |
| 1 | 16 | f32 | 0.559; 0.548 | 0.353; 0.344 | 0.767; 0.770 | 0.481; 0.462 | 0.130 | 1.170 LOSS | 0.889 WIN | 1.247 | 1.197 |
| 1 | 32 | f32 | 0.583; 0.575 | 0.323; 0.316 | 0.768; 0.770 | 0.378; 0.365 | 0.114 | 1.215 LOSS | 0.819 WIN | 1.184 | 1.119 |
| 1 | 64 | f32 | 0.688; 0.677 | 0.366; 0.355 | 0.859; 0.855 | 0.484; 0.468 | 0.075 | 1.234 LOSS | 0.725 WIN | 1.111 | 1.072 |
| 1 | 128 (no FT) | f32 | - | - | - | - | 0.040 | 1.123 LOSS | 0.686 WIN | 1.075 | 1.040 |
| 2 | 5 (FT B8) | f32 | 0.798; 0.816 | 0.687; 0.712 | 0.753; 0.760 | 0.844; 0.851 | 0.178 | 1.239 LOSS | 0.954 WIN | 1.371 | 1.343 |
| 2 | 8 | f32 | 0.732; 0.748 | 0.514; 0.532 | 0.856; 0.863 | 0.688; 0.694 | 0.185 | 1.404 LOSS | 0.957 WIN | 1.402 | 1.363 |
| 2 | 16 | f32 | 0.769; 0.774 | 0.663; 0.677 | 0.775; 0.774 | 0.731; 0.722 | 0.146 | 1.318 LOSS | 0.872 WIN | 1.274 | 1.290 |
| 2 | 32 | f32 | 0.732; 0.742 | 0.614; 0.632 | 0.749; 0.745 | 0.542; 0.560 | 0.135 | 1.293 LOSS | 0.890 WIN | 1.255 | 1.209 |
| 2 | 64 | f32 | 0.776; 0.770 | 0.534; 0.536 | 0.784; 0.777 | 0.805; 0.783 | 0.113 | 1.249 LOSS | 0.782 WIN | 1.163 | 1.118 |
| 2 | 128 | f32 | 0.809; 0.813 | 0.675; 0.682 | 0.870; 0.869 | 0.621; 0.638 | 0.068 | 1.099 LOSS | 0.765 WIN | 1.107 | 1.074 |
| 1 | 5 (FT B8) | bf16 | 0.409; 0.408 | 0.315; 0.319 | 0.397; 0.392 | 0.436; 0.422 | 0.166 | 1.205 LOSS | 1.061 LOSS | 1.375 | 1.297 |
| 1 | 8 | bf16 | 0.462; 0.461 | 0.342; 0.347 | 0.570; 0.563 | 0.502; 0.485 | 0.140 | 1.177 LOSS | 1.112 LOSS | 1.378 | 1.264 |
| 1 | 16 | bf16 | 0.483; 0.474 | 0.367; 0.357 | 0.507; 0.510 | 0.489; 0.470 | 0.134 | 1.176 LOSS | 1.168 LOSS | 1.298 | 1.219 |
| 1 | 32 | bf16 | 0.438; 0.432 | 0.323; 0.316 | 0.451; 0.452 | 0.362; 0.349 | 0.125 | 1.282 LOSS | 1.359 LOSS | 1.205 | 1.176 |
| 1 | 64 | bf16 | 0.480; 0.472 | 0.362; 0.352 | 0.467; 0.465 | 0.499; 0.483 | 0.098 | 1.318 LOSS | 1.469 LOSS | 1.138 | 1.097 |
| 1 | 128 (no FT) | bf16 | - | - | - | - | 0.062 | 1.176 LOSS | 1.590 LOSS | 1.088 | 1.061 |
| 2 | 5 (FT B8) | bf16 | 0.756; 0.772 | 0.707; 0.732 | 0.587; 0.592 | 0.894; 0.902 | 0.166 | 1.265 LOSS | 1.041 LOSS | 1.405 | 1.357 |
| 2 | 8 | bf16 | 0.692; 0.707 | 0.514; 0.532 | 0.648; 0.653 | 0.746; 0.752 | 0.194 | 1.452 LOSS | 1.053 LOSS | 1.408 | 1.374 |
| 2 | 16 | bf16 | 0.643; 0.648 | 0.518; 0.529 | 0.559; 0.558 | 0.736; 0.727 | 0.172 | 1.433 LOSS | 1.207 LOSS | 1.308 | 1.328 |
| 2 | 32 | bf16 | 0.622; 0.630 | 0.617; 0.636 | 0.490; 0.487 | 0.533; 0.551 | 0.148 | 1.333 LOSS | 1.203 LOSS | 1.247 | 1.255 |
| 2 | 64 | bf16 | 0.582; 0.578 | 0.529; 0.531 | 0.456; 0.452 | 0.794; 0.773 | 0.122 | 1.290 LOSS | 1.369 LOSS | 1.200 | 1.190 |
| 2 | 128 | bf16 | 0.570; 0.572 | 0.674; 0.682 | 0.472; 0.471 | 0.609; 0.626 | 0.093 | 1.116 LOSS | 1.486 LOSS | 1.124 | 1.101 |

### 4.2 The climb, both nodes

The final build of each cell against the first gated build (the climb's starting point: the first planner pair that
passes the compile gate, P-L1's `best_knobs`, `acq="load"`) and the other state's final build, in the final runs on
both nodes: the final build's time over the first base's (both builds in the same job, interleaved; every value a WIN
for the final build), and the bf16 build's over the f32 build's:

| TP | rows | f32: final / first base (n01-09; n07-17) | bf16: final / first base | bf16 build / f32 build |
|---|---|---|---|---|
| 1 | 5 | 0.831; 0.838 | 0.833; 0.837 | 0.947; 0.966 |
| 1 | 8 | 0.865; 0.869 | 0.844; 0.854 | 0.905; 0.911 |
| 1 | 16 | 0.863; 0.858 | 0.855; 0.853 | 0.895; 0.900 |
| 1 | 32 | 0.823; 0.826 | 0.783; 0.776 | 0.819; 0.809 |
| 1 | 64 | 0.814; 0.819 | 0.763; 0.759 | 0.720; 0.712 |
| 1 | 128 | 0.896; 0.895 | 0.851; 0.859 | 0.682; 0.681 |
| 2 | 5 | 0.810; 0.809 | 0.786; 0.790 | 0.951; 0.953 |
| 2 | 8 | 0.712; 0.714 | 0.690; 0.696 | 0.958; 0.950 |
| 2 | 16 | 0.767; 0.766 | 0.699; 0.702 | 0.861; 0.857 |
| 2 | 32 | 0.780; 0.780 | 0.757; 0.750 | 0.890; 0.886 |
| 2 | 64 | 0.798; 0.805 | 0.781; 0.774 | 0.789; 0.789 |
| 2 | 128 | 0.914; 0.915 | 0.901; 0.905 | 0.763; 0.755 |

Accepted changes (median ratio n01-09 / n07-17 against the previous base; every one a WIN on both nodes):

| round | cells | change |
|---|---|---|
| 1 + 2 | all 12 f32 cells | another E3 and / or E5 plan (planner or grid-fill), the conv epilogue `lds` / `batch` (or `cpre` at TP2 rows 128), the S4 state warm (`pre=s4`, at large rows two units: `warm=2`), the exact LDS pad: 0.77-0.94 |
| 3 | TP1 64, 128; TP2 8, 64 | another E3 plan: 0.93-0.98 |
| 4 | TP1 5; TP2 8, 32 (f32); 5 bf16 cells | the entry LDS pad back at TP2 rows 8-32 (0.90-0.92), plan / epilogue / prefetch combinations (0.95-0.98) |
| 5 | TP2 16 bf16 | the entry LDS pad: 0.92 |
| 6 | - | none (TP2 rows 16 f32: E3 planner plan 0 at 0.97, a WIN on one node only) |

What the final builds use: no split-K E3 (every cell folds K inside the CTA); the S4 state warm in every cell; the
batched or LDS conv epilogue in 22 of 24; E5 plans with token-tile splits (1-2 tiles per unit) at 32-128 rows.

### 4.3 Where the chain's time goes (final builds, both nodes, ratios of the chain)

Means of the two nodes' medians; "work" = the chain with only that stage's work minus the edges-only chain; the
parts sum to 1.00-1.10 of the chain.

| state | TP | rows | S3 work | S4 work | S5 work | edges | sum of the parts | conv epilogue price | hand-off price |
|---|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 5 | 0.40 | 0.28 | 0.22 | 0.17 | 1.06 | 1.08 | 1.021 |
| f32 | 1 | 8 | 0.36 | 0.34 | 0.21 | 0.14 | 1.06 | 1.13 | 1.012 |
| f32 | 1 | 16 | 0.31 | 0.45 | 0.18 | 0.13 | 1.07 | 1.19 | 1.004 |
| f32 | 1 | 32 | 0.22 | 0.58 | 0.09 | 0.12 | 1.02 | 1.11 | 1.002 |
| f32 | 1 | 64 | 0.18 | 0.66 | 0.08 | 0.08 | 1.01 | 1.55 | 1.010 |
| f32 | 1 | 128 | 0.16 | 0.74 | 0.06 | 0.05 | 1.01 | 1.74 | 1.004 |
| f32 | 2 | 5 | 0.45 | 0.26 | 0.21 | 0.17 | 1.09 | 1.17 | 1.031 |
| f32 | 2 | 8 | 0.35 | 0.31 | 0.19 | 0.20 | 1.05 | 1.23 | 1.014 |
| f32 | 2 | 16 | 0.37 | 0.38 | 0.15 | 0.16 | 1.05 | 1.15 | 1.005 |
| f32 | 2 | 32 | 0.30 | 0.49 | 0.11 | 0.15 | 1.05 | 1.18 | 0.999 |
| f32 | 2 | 64 | 0.21 | 0.59 | 0.10 | 0.13 | 1.03 | 1.42 | 1.010 |
| f32 | 2 | 128 | 0.22 | 0.67 | 0.06 | 0.08 | 1.03 | 1.40 | 1.003 |
| bf16 | 1 | 5 | 0.42 | 0.23 | 0.23 | 0.16 | 1.05 | 1.08 | 1.016 |
| bf16 | 1 | 8 | 0.40 | 0.29 | 0.24 | 0.15 | 1.07 | 1.14 | 1.016 |
| bf16 | 1 | 16 | 0.37 | 0.34 | 0.19 | 0.14 | 1.04 | 1.20 | 1.005 |
| bf16 | 1 | 32 | 0.30 | 0.45 | 0.13 | 0.13 | 1.01 | 1.39 | 1.004 |
| bf16 | 1 | 64 | 0.26 | 0.52 | 0.12 | 0.11 | 1.01 | 1.60 | 1.006 |
| bf16 | 1 | 128 | 0.24 | 0.59 | 0.10 | 0.07 | 1.01 | 1.73 | 1.010 |
| bf16 | 2 | 5 | 0.48 | 0.22 | 0.23 | 0.17 | 1.10 | 1.17 | 1.015 |
| bf16 | 2 | 8 | 0.37 | 0.26 | 0.21 | 0.20 | 1.04 | 1.26 | 1.017 |
| bf16 | 2 | 16 | 0.36 | 0.34 | 0.20 | 0.17 | 1.06 | 1.20 | 1.013 |
| bf16 | 2 | 32 | 0.36 | 0.38 | 0.13 | 0.16 | 1.02 | 1.17 | 1.002 |
| bf16 | 2 | 64 | 0.28 | 0.46 | 0.13 | 0.13 | 1.00 | 1.50 | 1.015 |
| bf16 | 2 | 128 | 0.32 | 0.53 | 0.08 | 0.10 | 1.03 | 1.42 | 1.010 |

- **The conv epilogue** costs 1.08-1.23x the plain E1 at rows 5-32 and 1.40-1.74x at rows 64-128 (f32 builds; the
  standalone E3 over E1 on the same plan, streamed). At large rows that is bytes, not latency: the conv state read + write and the qkvz
  output are about 0.9x the in-proj's MXFP4 weight bytes at 128 rows (TP2: 3.9 + 3.9 + 2.1 MB vs 11.2 MB), and FlyTape's
  `gdn_proj` moves the same state.
- **The hand-off** (MXFP8 codes + op_sel scales from the recurrence's finisher) costs 0.99-1.04x P-L1's kernel: free.
- **The edges** (3 grid edges per layer) are 0.04-0.20 of the chain, largest at small rows and TP2.

### 4.4 One persistent launch vs launches per stage or per layer (final builds, both nodes)

- The four standalone stage kernels per layer (`layers=1` bring-up) take **1.06-1.43x** the chain (largest at small rows
  and TP2).
- The chain launched once per layer takes **1.04-1.39x** the chain walking all layers in one launch.
- Both are LOSS for the per-layer forms in every cell. This agrees with WP-L1 (1.018-1.021 for its synthetic bodies)
  and reverses P-L4's MoE sign for the GDN family: the per-layer prologue of three stages (table loads, the recurrence's
  pipeline fill, E1 setup) is what a launch boundary repeats.

### 4.5 `state=bf16` (N11, default off)

- **Speed** (same job, interleaved, the bf16 build on its own data: both nodes and the bracket job): the bf16 build runs
  at **0.68-0.97 of the f32 build**, WIN in all 36 node x cell results: 0.68-0.89 at rows 32-128, 0.86-0.97 at rows 5-16
  (where the recurrence is a smaller share). Against FlyTape: 4.1.
- **T0**: the state stays within the bf16 storage bound (2^-8 = 3.9e-3 of the largest state; measured 2.5e-3 to 3.6e-3), the output is
  bitwise equal to the emulation of its own order, all mutants caught; drift through the whole chain (2,048 steps in
  place, fresh inputs every step, f32 chain as reference, TP2 rows 16):

  | regime | step 1 | 16 | 64 | 256 | 1,024 | 2,048 |
  |---|---|---|---|---|---|---|
  | mixed decay (0.07-0.98): state / out-proj | 3.2e-3 / 0 | 4.4e-3 / 1.2e-2 | 2.8e-3 / 1.9e-2 | 3.0e-3 / 1.0e-2 | 2.6e-3 / 1.1e-2 | 2.7e-3 / 8.8e-3 |
  | long memory (0.95-0.999): state / out-proj | 2.9e-3 / 2.0e-3 | 1.4e-2 / 2.7e-2 | 2.5e-2 / 3.7e-2 | 4.8e-2 / 3.9e-2 | 1.1e-1 / 4.6e-2 | 1.5e-1 / 5.3e-2 |

  With mixed decay the error settles at the bf16 quantum. With long memory it grows with the step count (about as
  steps^0.4 in the state, a random walk of storage roundings that a slow decay does not forget). P-L1's 512-step run
  stopped before the growth was visible.
- **The accuracy gate is not run here** and the knob stays off. The plan (DESIGN 12.4, P11's protocol): T2 teacher-forced
  drift with a served-path emulation of N11 (round the GDN recurrent state cache `mamba_v_cache` to bf16 after each
  decode step's target pass, inside the captured graph; exact for the storage semantics, since the kernel computes in
  f32 on bf16-stored values), against B = 7.81e-3 nats with N11's share of the long-context allowance; then T4 RULER at
  8K / 32K / 128K / max, which the long-memory drift above makes the deciding test. The emulation hook needs a runtime
  overlay on a staged copy of ATOM-truemono (WP-L6 / WP-L8); nothing here edits ATOM. Variants worth building if the
  plain form fails (default-off): bf16 storage only for heads whose decay class forgets (per-head precision from
  `A_log` / `dt_bias`), or stochastic rounding of the stored state.

## 5. WaveScope (Markham MI350X, one traced CU, before / after the climb)

| cell | first base: EXEC / WAIT / STALL | final: EXEC / WAIT / STALL | grid-wait share of stall (first -> final) | total latency of the CU, final / first |
|---|---|---|---|---|
| TP1 rows 32 | 19.2 / 53.9 / 26.7 % | 23.2 / 50.5 / 26.1 % | 45.4 -> 36.9 % | 0.815 |
| TP2 rows 128 | 21.1 / 55.3 / 23.5 % | 22.3 / 50.3 / 27.2 % | 49.0 -> 39.5 % | 0.959 |
| TP2 rows 8 | 16.3 / 67.6 / 15.8 % | 16.1 / 72.0 / 11.6 % | 57.5 -> 62.9 % | 0.813 |

- The traced CU spends most of its stall at the edges' grid wait: the slowest CTA of each stage sets the pace (P-L2's
  phase-tail imbalance). The climb cut vector-memory load stall at the start of S4 (the state warm puts the first
  unit's state in L2 while the grid finishes S3: `vmem_load` stall 12.0 -> 6.9 % at TP2 rows 8) and the grid-wait share
  at rows 32-128 (fewer, better-filled units).
- At TP2 rows 8 the remaining time is the edges and the 1-unit-per-CTA tails of every stage (192 recurrence units on
  256 CTAs): the per-edge flags and partial edges of WP-L3 are the lever left.
- The acquire change of the first bases (`acq="wave"` -> `"load"`: no L2 invalidate at the edges) was the first major
  change: 0.72 of the traced CU's latency at TP1 rows 32 (capture `att1`).

## 6. Lever ledger (full table in `WP-L2c-tables.md`)

| lever (family) | solo, round 1 on the first bases: median (best) | part of accepted changes |
|---|---|---|
| E3 planner plan (12) | 1.022 (0.858) | 13 |
| E3 grid-fill plan (21) | 1.080 (0.844) | 5 |
| E5 planner plan (8) | 1.001 (0.879) | 10 |
| E5 grid-fill plan (33) | 0.989 (0.884) | 5 |
| conv epilogue `lds` (`cepi`, `cepi_fit`) | 0.947 / 0.957 | 7 |
| conv epilogue `batch` (`cbatch`, `_g4`, `_cpre`) | 0.972-0.984 | 8 |
| `cpre` | 0.989 | 1 |
| S4 state warm (`pre_s4`, `pre_s4w2`, `pre_s45`) | 0.984-0.986 | 15 |
| out-proj weights before the S4 -> S5 wait (`pre_s5`) | 1.003 | 0 (in `pre_s45` only) |
| exact LDS pad (`lpad=fit`) / entry pad back (`lpad_entry`) | 0.998 / 1.001 (best 0.902) | 1 / 5 |
| recurrence depth of the streamed form (`rec_stream`) | 1.062 | 0 |
| `acq="cta"` / `"wave"` (vs `load`) | 1.10 / 1.21-2.31 (job l2) | 0 |
| combinations | 0.864 (0.773) | 16 of 25 accepted changes |

Solo losers that won inside combinations (PLAN 13): grid-fill E3 plans (solo median 1.08) are in 5 accepted changes;
`lpad_entry` (solo median 1.00) is a 0.90-0.92 WIN on the bases where the grid-fill E5 plans replaced the planner's.
Verdicts are "on base X": the ledger records the base hash of every measurement.

## 7. What this means for DESIGN-LB 15

- **15.2's GDN rows** (in situ, same job, f32 state; TP1 / TP2):
  - S4 recurrence (state warm and hand-off included): 0.51-0.86 / 0.75-0.87 of FlyTape's `gdn_recur` (15.2: 0.84-0.94
    per launch, 0.50-0.90 streamed, for the recurrence alone).
  - S3 in-proj with the conv / gates epilogue in: 0.32-0.37 / 0.51-0.71 of `gdn_proj` (15.2: 0.34-0.43 / 0.59-0.86 for the
    GEMM alone, epilogue not built; WP-L2a's E1 picks 0.22-0.25 / 0.34-0.44).
  - S5 out-proj from the hand-off: 0.37-0.50 / 0.54-0.85 of `gdn_out` (15.2: TP2 1.39-2.03; WP-L2a 0.83-0.93).
  - The GDN family (in-proj + recurrence + out-proj, grid edges included): 0.44-0.69 of FlyTape's family at TP1 and
    0.73-0.82 at TP2; 0.41-0.48 / 0.57-0.77 with `state=bf16`.
- **15.3.** The GDN family share at TP2 B128 (0.276 of FlyTape's step, LB 0.22-0.27) becomes 0.22 (0.81 x 0.276) at f32 and
  0.16 (0.570 x 0.276) with `state=bf16` (accuracy-gated).
- **15.4 "GDN recurrence: no further f32 gain"** holds for the recurrence's own bytes; the state warm (an L2 prefetch
  of the first unit's state during the S3 -> S4 edge) is the in-situ gain P-L1 could not see.
- **One launch per step** for the GDN family (4.4) is a measured default for WP-L3 / WP-L4.
- **LB TP2 vs LB DP2 for this family** (PLAN 5; per GPU, same job, exchange excluded): a DP2 replica's GDN chain (TP1 at
  n / 2 rows) takes 1.14 / 1.13 / 1.02 / 1.00x the TP2 rank's at B16 / B32 / B64 / B128 (f32; bf16 1.22 / 1.15-1.17 /
  1.02-1.03 / 0.98-0.99). The TP2 rank still has to add its out-proj exchange
  (WP-C5), which this ratio leaves out; WP-L4 / WP-L5 decide per cell with the whole layer.
- **Open for WP-L3**: per-edge flags and the partial edges S3 -> S4 (per head) and S4 -> S5 (per head group); the
  1-unit tails at TP2 rows 5-16.

## 8. Caveats and open items

- **Precision of the FlyTape comparison.** FlyTape's dense projections are bf16 (3.8x the weight bytes of MXFP4); the LB
  stages run ATOM's checkpoint numerics (MXFP4 W4A8 projections, bf16 ba). The recurrence is f32 state on both sides;
  the `state=bf16` rows are the N11 lever, not a matched comparison (P-L1 verification issue 4).
- **TP2 exchange.** FlyTape's TP2 `gdn_out` phase also pushes its exchange lines; LB's S5 has no exchange (WP-C5 F5 /
  WP-L5 own it). As in WP-L2a, GEMM phases are compared with GEMM phases and the exchange is budgeted separately.
- **FlyTape's phases include the next grid sync's wait**, LB's chain includes its three edges: both are in-situ per-layer
  forms; LB's L layers rotate cold weights and state (at least 4x the MALL).
- **Synthetic data** for both stacks (random weights / inputs / state); real decays and routing are not modelled.
- **Unexplained single lever failure** (2.4) and the **image-COMGR fault** of the E3 `kskip` mutant (3, Compilers): not
  reproduced with the venv COMGR in any job; the container bracket uses the venv-built code objects. Root-causing the
  image-COMGR fault (a different register allocation exposing a latent issue, or a COMGR defect) is open.
- WaveScope ran on MI350X, not MI355X; the timing verdicts are MI355X.
- PLAN 14 quick end-to-end tok/s: not applicable until a served path runs the LB chain (WP-L6).

## 9. Rerun

From `hackathon-prep/qwen38/truemono-lb/wp-l2c` (Git Bash), `Q38_MAX_GPUS=8`:

```bash
bash sync_l2c.sh both 30                                  # stages w7-l2c-rocke30 (the copy-rocm-lib rocKE tree), w7-l2c-h30
# final confirmation (both nodes), numerics + stale + timing + determinism stress on the final bases
RKSFX=30 bash sub.sh fa n01-09 1 30 01:30:00 "--phase+numerics,stale,timing,stress+--states+f32,bf16+--rounds+24+--workers+16+--seed+7701+--stress-n+128+--base+cfg/base_final.json+--combos+cfg/combos_final.json+--extra-arms+xstate"
RKSFX=30 bash sub.sh fb n07-17 2 30 01:30:00 "...same with --seed+7702"
# a climb round: solo levers + combinations on a base, both nodes, then
python climb.py <runs node A> <runs node B> --combos-in cfg/combos_rK.json --base-in cfg/base_rK.json \
  --base-out cfg/base_rK+1.json --combos-out cfg/combos_rK+1.json --ledger cfg/ledger.jsonl --round rK
# FlyTape bracket: dump the code objects in a venv run (--dump-hsaco hsaco), stage them as <harness>/hsaco, then
RKSFX=30 bash sub_ct.sh ct n01-09 31 03:00:00 "--phase+numerics,stale,drift,timing,stress+...+--hsaco-dir+hsaco"
python final_report.py --ct <results>/w7-l2c-ct --ratio r.md --abs a.md
# WaveScope (Markham): bash tmp/att_run.sh (items in tmp/att_items.tsv), python att_sum.py logs/att-*.log
```

## Verification

Independent verifier (W7), 2026-11-01. **Verdict: confirmed.** The WP-L2c gate (DESIGN-LB 15.6: T0 against fp64 plus
bitwise against the exact-order emulation, with mutants, stale-read twins and determinism on MI355X before timing) was
exercised in every final job on both rocke nodes and reproduces in a fresh job on the node that had no same-job FlyTape
bracket. The numbers stand. Items V2-V5 below correct the report's wording; none of them changes the gate.

**Re-run (the cheapest decisive check): job 68242045, tag `w7-vWP-L2c-ct`.** It ran on n07-17, GPU bus 65 (bus 15 also
healthy; e5 not allocated) in the q38 container. FlyTape was traced before and after the LB timing, as in 68241852, but
on the other node and with a fresh seed (90211). The job used a freshly staged copy of the local rocKE tree
(`w7-vWP-L2c-rk`; `gdn_lb.py` d7ea22ae, `gdn_recur.py` d74fee15 and `e1.py` debca5f9, the same as the final runs) and the
builder's harness stage `w7-l2c-h31` (`l2c_bench.py` 0e9ba587, the same as the bracket job). The LB kernels were the
venv-built code objects (`--hsaco-dir`), as in the bracket job.

- **Numerics:** 24 of 24 cells PASS (TP1 / TP2 x rows 5-128 x f32 / bf16). All 216 mutant runs were caught. All 24
  `early` stale-read twins fail, and `delay` keeps the bits in all 24. Stage isolation is bitwise, and the timed binary
  equals its dbg twin, its recompiled copy and both walks. The acquire stale-read harness: `load` / `wave` / `cta` read
  0 stale words at XCD distances 0 / 1 / 3. The twins `xcd_nol1`, `xcd_nol2` and `plain` read stale words (0.94-1.00),
  and the XCD-leader form is again unsound (0.94).
- **FlyTape bracket on n07-17 (f32 state):**

  | | rows 5 (vs FT B8) | rows 8-64 (TP1) / 8-128 (TP2) | report (n01-09) |
  |---|---|---|---|
  | TP1 | 0.43-0.45 | 0.51-0.69 | 0.44-0.69 |
  | TP2 | 0.80-0.81 | 0.76-0.82 | 0.73-0.82 |

  With `state=bf16`: TP1 0.40-0.47 and TP2 0.57-0.77, against the reported 0.41-0.48 and 0.57-0.77. The per-stage
  ratios (S3 0.31-0.35 / 0.51-0.73, S4 0.50-0.88, S5 0.34-0.91) are within 0.06 of section 4.1, cell for cell. FlyTape's
  pre and post traces agree within 0.02 on the family.
- **Controls and verdicts:** chain#2 and the recompiled chain sit at 0.985-1.007. The first base / final ratio is
  1.10-1.45 (LOSS for the first base in all 24 cells). The other-state arm is a WIN for bf16 in all 12 cells
  (0.68-0.97). Per-stage launches / chain is 1.07-1.42, and one launch per layer / chain is 1.03-1.38 (LOSS in all 24
  cells in this job).

**Code and log review.**

- The win rule (`win_rule`: the median below both controls' per-round minimum by 0.002, and 3/4 of rounds below 1) is
  applied as described. Arms are shuffled per round and run as separate graphs for chain#2 and the recompiled copy.
- `climb.py` accepts a change only if it is a WIN in both node runs on the same base hash. Ledger: rounds r1+r2 and r3
  re-tested all 12 f32 cells, and r4 all 24. r5 re-tested the 10 cells whose base changed in r4, and r6 the 4 cells
  around the one r5 change.
- The FlyTape extraction is `(gdn.<p> + gdn_ple.<p>) / 36`. Reduce phases are counted at TP1 only (`family`) and
  excluded at TP2 (`family_noreduce`), so the comparison is fold-matched. At TP2 the exchange is left out on both sides.
- Every final build is complete-output (E3 / E5 `splits=1`; E3g folds in-kernel), so no fold is missing on the LB side.
- The compile gate (`gate.scan_hsaco`: any scratch, spill or asm-hazard hit refuses the build) ran on every build. The
  13 `lever ...: FAIL` lines outside the r2b case are gate refusals (scratch 12-188) or the `altstate` harness bug (fa /
  fb), not numerics failures.
- The numerics totals reproduce from the jobs' `l2c.json` with `num_sum.py`: 314 cells / 2,826 mutants / 942 isolation
  checks over the 24 non-bring-up jobs. The 4 failing cells are bring-up jobs n2-n4, before the first gated base. The
  `.out` logs of r1b / r3a / r3b are truncated (6 of 12 cells shown), so count from `l2c.json`, not from the logs.

**Issues (wording; no gate impact):**

- **V1 (closed by this re-run).** The same-job FlyTape bracket ran on one node only (n01-09). Job 68242045 adds n07-17
  and agrees.
- **V2.** The headline "TP1 0.44-0.69" takes its low end from rows 5 against FlyTape B8. LB then processes 5 rows of
  recurrent state against FlyTape's 8 (S4 0.51 at rows 5 vs 0.76 at rows 8). Quote the row-matched range, **TP1
  0.51-0.69**, with rows 5 as a separate line.
- **V3.** Section 4.4 says one launch per layer is a LOSS "in every cell". It is a TIE at TP1 rows 128 in 6 builder
  results: fa2 bf16, fb2 f32 and bf16, ct4 f32, fh f32 and bf16 (1.036-1.057). In 68242045 it is a LOSS (1.031 / 1.055).
  At TP1 rows 128 it is at most a narrow loss.
- **V4.** In section 2.4, "No final build contains `e5f12`" leaves out that the other part of the failing combination,
  `pre_s45`, is in 4 final builds (TP1 rows 5 f32 / bf16, TP1 rows 32 bf16, TP2 rows 32 f32).
  - Mitigation: those cells pass numerics, stress and stale twins in every final job and in 68242045.
  - Code check: the prefetch issues only read-only out-proj weight loads (`prefetch`) and state loads by the CTA that
    later owns that unit (`warm_state`). Both feed a dead sink, so no stale-read path is visible.
  - Still, the single failure should be attributed to the combination, not dismissed through `e5f12`.
- **V5.** Section 2.3's "beta 2 f32 ulp" is the dbg-twin check. The timed-binary layer check uses a loose rule
  (reference b / a = f64 projection rounded to bf16), so `num_sum.py`'s WORST shows beta up to 2.3e4 ulp. That is
  expected, since the timed binary equals its dbg twin bit for bit, but the table should say which check it reports.

Raw data: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/vWP-L2c-raw/` (log, `l2c.json`, FlyTape traces,
`ratio.md`, `abs.md`).
