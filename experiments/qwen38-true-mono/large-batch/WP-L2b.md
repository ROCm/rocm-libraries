# WP-L2b: the large-batch HC mixers (combine + TwoStage norm + split-K down + up with the stream mix) in one persistent chain, rows 5-128, TP1 and TP2, gfx950

W10, 2026-11-02. `../DESIGN-LB.md` 15.6 / 15.7 (WP-L2b, E2), 4.2 (S1 / S2 / S6 / S7); `W7-STATUS.md` section 4 (entry
conditions); `WP-L2a.md` (the E1 GEMM core this builds on); `WP-L2c.md` (the chain / edge machinery it reuses);
`../wave2/WP2a.md` (HC semantics, read only); `../reverse/` (toolkit). Ratios only here. Absolute times, job logs,
WaveScope captures and raw JSON: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2b.md` and
`.../large-batch/wp-l2b-raw/`. Harness: `hackathon-prep/qwen38/truemono-lb/wp-l2b/`. Below 1 is faster.

## Gate and verdict

**Gate: met for the build, the numerics and the measurement discipline. Against FlyTape's whole mixer family: faster
in every cell, by less than the W7 budget assumed (TP2 B128 exchange-free: near parity, 0.94-0.99 estimated). Not
met: DESIGN-LB 15.4's per-stage criterion for C (1.08-1.32x FlyTape's combine at TP1). Corrected in the Fix round
(end of file); the bullets below carry the corrections.**

- **One persistent launch walks any number of HC instances** (`q38lbh_*`, 256 CTAs, runtime layer loop, W5's striped
  grid edges, residency ticket, launch-fatal word): C (combine + norm + MXFP8 pack), D (the 10240 -> 320 down GEMM
  on WP-L2a's E1 core with the activation epilogue), U (the 320 -> 10240 up GEMM with the 4-stream mix in registers and
  the next GEMM's MXFP8 pack). ATOM's numerics (MXFP4 weights, MXFP8 activations, ATOM's bf16 rounding points).
- **Against FlyTape's mixer phases in the same job** (its traced `*_mix_combine` + `*_mix_down` + `*_mix_reduce` +
  `*_mix_up`, before and after the LB timing; strict prebuilt code objects in the container; both rocke nodes):
  - **TP1 0.82-0.92 of FlyTape's mixer family** (rows 5-64; FlyTape cannot run TP1 B128). The thinnest cell is B8
    (0.917 / 0.905). Rows 5 are against FlyTape **B8** (FlyTape has no B5 configuration; `l2b_report.py` keys
    `max(rows, 8)`): read 1:5 / 2:5 as the 1:8 / 2:8 cells (LB's own B5 is 0.957-0.969 of its B8).
  - **TP2, fair (exchange-free) reading: 0.84-0.92 at rows 5-64** (FlyTape's TP2 family with its combine replaced by
    its measured TP1 combine at the same batch), and **about 0.94-0.99 at B128** (FlyTape's TP1 combine is not
    measurable at B128; its B64 combine gives 0.960 / 0.993, a B32 -> B64 growth extrapolation 0.942 / 0.989). Against
    FlyTape's down + reduce + up alone LB's whole chain is 1.11 / 1.15x at B128. The exchange-inclusive figures (0.60-0.70,
    B128 0.605 / 0.632) count FlyTape's peer wait (36-46 % of its family) with no matching cost on LB's side (no TP2
    exchange before WP-L5) and are **not** a mixer comparison.
  - FlyTape's before / after traces agree within 0.025 in every cell.
- **Per phase (critical path from stamps x time):** LB's C + its edge is **1.08-1.32x FlyTape's `mix_combine`** at
  TP1 (it does the norm and the MXFP8 pack that FlyTape does in its down staging); D + edge is 0.46-0.91 of FlyTape's
  `mix_down` + `mix_reduce`; U + edge 0.73-1.23 of `mix_up` (1.12-1.23 where U folds D's partials, `ufold`).
- **The lookback climb** (10 rounds, both nodes, PLAN 13 / 15): the final builds run at **0.70-0.85 of the first
  gated build** of each cell (same job, both nodes; the first build is 1.17-1.43x the final). 82 changes accepted,
  every one a WIN on both nodes in the same round; 14,926 arm measurements (each on both nodes) in the ledger. The last
  round accepted 2 changes of about 1 %: the climb has converged on this structure.
- **New structure this lane (all default off, every existing build IR-identical):** split-stream C units
  (`CKnobs.cw` 2 / 4 at 4 waves) 0.94-0.98 in 10 cells; the D fold moved into U's staging (`ufold`, FlyTape's
  `mix_reduce` folded into `mix_up`'s source) with 16-row D units 0.93-0.98 at rows 16-32 (bit-identical to its base);
  split-C injection-dot partials finished off C (`gplace="s"`) 0.95-0.97 at B128; bit-exact packed C arithmetic
  (`cvec="pkx"`, bitwise to its base in 24 / 24 checks, a tie in time).
- **Numerics** (census corrected in the Fix round; all 38 jobs now fetched). 862 recorded cell checks over 38 jobs on
  both nodes, 832 PASS; the 30 FAIL are job `n2`'s 22 cells (its `c:eps` mutant was not caught on unit-variance data;
  the checker was fixed, no miss since) and 8 `cfuse` B128 builds the gate refused (scratch). Job `n1` recorded no cell:
  its 3 cells (first-generation bases) passed the dbg twin, determinism, walk and recompiled-copy checks, then the
  stage checker stopped with `ref_quantize_mxfp8 needs finite input` before a verdict (no verdict = not passed; nothing
  timed; those bases are not the final ones). 10,826 of 10,848 must-fail mutant runs caught (the 22
  misses are `n2`'s cells); 854 / 854 deterministic stale-read twins changed the bits and 854 / 854 delay controls kept
  them; determinism, the per-layer walk = the persistent loop, a recompiled copy = the timed object, the dbg twin = the
  timed object, and the checker's own must-fail controls all 854 / 854; 72 / 72 bitwise-lever checks equal.
- **Not met / not done**: the W7 budget's HC row (0.50-0.78, GEMM phases only) does not hold for the family (section
  6: TP1 B8 reaches its 8k bar; **TP2 B128 moves from 0.81 to about 0.88 central**, which matters for the 10,000 tok/s
  target); no quick end-to-end number (no served path runs the LB chain before WP-L6); the TP2 exchange is WP-L5's; the
  final bases are **tolerance class** (ATOM's `sigmoid_fast` everywhere, `v_rsq` inverse RMS in 7 cells, the split
  square-sum order in the split-C cells) and need the accuracy suites before serving (PLAN 1); the bit-exact ATOM-order
  twin is the first build (1.17-1.43x).

## 1. What was built

| piece | file |
|---|---|
| the chain: `HcKnobs` (`norm` faithful / reassoc, `waves`, the C map `CKnobs` (`cw` 1 / 2 / 4, `inv` f64 / rsq), the D / U E1 plans, `epre`, `pre` (L2 warm `d` / `u` / `n`, register prefetch `D` / `U`), `acq`, `ticket`, `lpad`, `sig` acc / fast, `cstat` cta / fold / dfold / cfold, `cfuse`, `cvec` scalar / pk / dot2 / pkx, `uvec`, `gplace` c / d / u / s, `eload`, `ufold`, `stamp`, `dbg`, mutants); C (`_Comb`), gates off C (`_Gates`, `_GSum`), D (`_Dn`, fused `_DnF`, per-stream fold `_DnS`), U (`_Up`, U with the D fold in its staging `_UpF`), `to_ufold` | `dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/qwen38_mono/large_batch/hc_lb.py` |
| CPU tests (56): the parents (E1, the WP-L2c chain) lower to the same IR before and after HC builds; every cell / norm / wave count / mutant / lever / ablation lowers; refusals; the up-weight permutation; `ufold`, split C, `gplace="s"`, `cvec="pkx"` lower, leave the base IR unchanged and are refused where not built | `.../rocke/library/tests/test_qwen38_lb_hc_lb.py` |
| bench (numerics suite, stale-read twins, timing with the win rule, stamps, levers / combinations / bases from files, code-object dump and strict load), compile gate, job bodies, staging, climb bookkeeping, final-arm builder, FlyTape phase parser, reports, reverse step, WaveScope launcher, numerics census | `hackathon-prep/qwen38/truemono-lb/wp-l2b/`: `l2b_bench.py`, `gate.py`, `job.sh`, `sub.sh`, `sub_ct.sh`, `sync.sh`, `fetch.sh`, `climb.py`, `mk_final.py`, `ft_mix.py`, `l2b_report.py`, `rv_hc.py`, `att_l2b.py`, `att_job.sh`, `num_sum.py`, `cfg/` (bases `base_r1` .. `base_r11`, combinations, `ledger.jsonl`) |

`e1.py`, `e1_plans.py` (WP-L2a) and `gdn_lb.py` (WP-L2c) are imported, not edited. Every knob added after the first
measured builds is default off and was checked IR-identical (names included) on 40 builds per change (the 12 cell bases
x plain / dbg / stamp, plus defaults). Python engine only; nothing served imports the module.

**How a step's HC runs.** C: one unit per (token, stream) wave, or with `cw` 2 / 4 at 4 waves a CTA per (token, two
streams) / (token, stream); `r' = bf16(r + bf16(y g))` (TP2: the ranks' `y` summed in rank order), the stream's square
sum, `inv = 1 / sqrt(ss / 2560 + eps)` (f64 or `v_rsq`), `n = bf16(bf16(r' inv) w')`, MXFP8 per 32-K block in E1's
op_sel layout; the next combine's gates from 4 injection dots over `n` (in C; on the top CTAs during D / U; or, with
split C, as per-stream partials finished after the edge). D: an E1 build (`k` 10240, 320 rows) with split-K; the last
arriver folds the partials in split order and packs `bf16(silu(bf16(bf16(y) / 4)))` as MXFP8; or (`ufold`) D stores
its partials only and every U unit folds them in its own staging. U: an E1 build over the up weights with rows permuted
to `4 col + stream`, so one MFMA lane holds the 4 streams of a column and the mix stays in registers
(`bf16((((p0 + p1) + p2) + p3) / 4)`); x bf16 + MXFP8 for the next GEMM.

## 2. Reverse step: FlyTape's mixer against the rocKE chain

FlyTape's source (`flytape/models/qwen4/bodies_tiled.py`) and its marked tape's ISA (`logs/q38/rv-ft1`, the reverse
toolkit's split, copy 0 of each phase) against the LB builds of `base_r9` at 1:16 and 1:64 (job `rv3`, debug == plain
`.text`; the 1:16 build runs `ufold`; rounds 9-10 changed only plan fields of these two cells).

**Structure.** FlyTape runs four grid-synced phases per HC; LB three stages and three edges:

| | FlyTape | LB (final) |
|---|---|---|
| combine | `mix_combine`: one wave per (row, stream) spread over all workers; `r' = r + g y` in f32, bf16 store, square sum, `rsqrt`; no norm, no pack | C: the combine, the norm and the MXFP8 pack (ATOM's rounding points), the gates' dots (or their partials) |
| down | `mix_down`: bf16 activations normalised in the staging (`TwoStage`: `r' rrms (1 + gain)`), 324 rows (320 + 4 gate rows), `MIX_SPLITS` 40 (B128: 20 x 2 token blocks), partials to memory | D: MXFP4 x MXFP8 E1, split-K 2-10 x K-waves 1-4, the fold by the last arriver (or `ufold`) |
| reduce | `mix_reduce`: a separate phase; every thread folds the splits of a (token, row), silu, the gates | inside D (last arriver) or U (`ufold`) |
| up | `mix_up`: one stream per wave, K = 320 in flight in one round trip, the 4 streams summed through LDS | U: the 4 streams in one lane's accumulators (no LDS mix), epilogue operands issued at unit start |

**Static ISA per stage** (instructions in the stage's code, not time; rocKE / FlyTape):

| stage | 1:16 instr | 1:16 VALU | 1:64 instr | 1:64 VALU | reading |
|---|---|---|---|---|---|
| C vs `mix_combine` | 2.89 | 3.35 | 3.29 | 3.86 | rocKE's C carries the norm + MXFP8 pack (FlyTape's sits in its down staging) and ATOM's bf16 rounding steps: `_bf16r` / `_lo` / `_hi` alone are 491 instructions at 1:16. This is why C is the one phase slower than FlyTape's, and why splitting C over more waves (`cw`) paid |
| D vs `mix_down` + `mix_reduce` | 0.48 | 0.28 | 0.65 | 0.50 | E1's `sop` loop: no scale rewrites, `nt` weights, 12 loads in flight at 1:16 (FlyTape 0 at the phase's start); FlyTape's reduce alone is 41 loads with 48 waits |
| U vs `mix_up` | 3.23 | 4.36 | 1.14 | 1.48 | 1:16 runs `ufold` (the fold, silu and MXFP8 pack in U's staging: 58 loads); 1:64 is the plain U |
| edges / frame (rocKE only) | 336 / 747 instructions at 1:16 | | 416 / 629 | | FlyTape's grid sync lives in its runtime, outside the marked phases |

Resources: rocKE 204 / 180 VGPRs, 101 / 153 SGPR spills, no VGPR spills; FlyTape's whole tape 310 / 409 VGPRs and
1,829 / 1,725 SGPR spills.

**Where each side leaves time.** FlyTape: four grid syncs per HC instead of three, a whole reduce phase (41 partial
loads per thread, then a sync), bf16 activations staged through LDS (twice LB's MXFP8 bytes). LB: the C stage's VALU
chain on one wave per (token, stream) at small rows (C was 0.26-0.31 of the chain before split C), the D fold's
round trip (section 4.2), and the edges (0.15-0.30 of the chain together).

## 3. Numerics

Every timed code object, in every job, before any timing (DESIGN.md 9.1, DESIGN-LB 13):
- stage-wise T0 on random data at the cell's shape from the kernel's own upstream: `r'` bitwise to the host emulation,
  `inv` within 4e-6 relative of fp64 (2e-5 with `v_rsq`), the C MXFP8 codes and scales bitwise to the host rule on the
  kernel's own `inv`, the gates within 2 bf16 ulp, D's `y` within one MXFP8 eps of the fp64 GEMM over the kernel's own
  activations, the activation MXFP8 bitwise (block-scale slack 1 per 500 blocks), U's accumulators within one eps
  (dbg twin), x within its bf16 tolerance band, the next GEMM's MXFP8 bitwise;
- the dbg twin equal to the timed object; determinism (3 runs); the per-layer launch walk equal to the persistent loop;
  a recompiled copy (`tag="rc"`) equal to the timed object;
- 12-13 must-fail mutants per cell (C `ygate`, `eps`, `xblk`, `inj`, `rskip`; D `silu`, `kskip`, `fold` or `ufold`,
  `kwred`; U `mix`, `sig`, `xs`, `kskip`, `kwred`) and the checker's own must-fail controls (no launch; one row x 1.05);
- the deterministic stale-read twin (`early`: every edge released one arrival early while CTA 0 starts every stage
  late) must change the bits, its positive control (`delay` alone) must not;
- every lever / combination arm: T0 on 2-layer random data and determinism before it is timed (an arm that fails is not
  timed: `cvec=dot2` (square sums 0.13 low, a recorded loser); stage ablations without U have no output to check);
- bitwise levers checked against their base on the same data: `ufold` 48 / 48, `cvec="pkx"` 24 / 24;
- the compile gate refuses spilling builds (872 refused over the lane) and the asm hazard pads stay on (`hz`); COMGR
  refused 577 B128 builds over the LDS limit (`HcKnobs.check`'s LDS estimate undercounts those 4-row-group plans).

Census (`num_sum.py`, 38 jobs incl. `n1`, both nodes): see the verdict. 49,827 code objects built or loaded over the
lane (`n4a` / `n4b` died on an argument error before any GPU work and `f2d` was cancelled; none has data). The container
jobs loaded all 333 / 340 venv-built objects strictly (alola 219) and every timed object passed on both nodes.

## 4. Results

### 4.1 Final builds against FlyTape (same job, both nodes) and against their own earlier builds (same job)

Container jobs `ct2a` (n01-09) and `ct2b` (n07-17): FlyTape traced before and after, LB's final bases (`cfg/base_r11`)
plus the arms below, 12 shuffled rounds over 128 HC instances; venv jobs `f2e` / `f2v` repeat the LB part on both
nodes (the arm ratios span all four jobs). FlyTape's family per HC = its four mixer phases over the step / 96.

| TP | rows | LB / FT family (n01-09 / n07-17) | TP2: FT's TP1 combine as proxy | FT family share of its step | first / final | r5 / final | r9 / final | cw1 / final | noufold / final | reassoc twin / final | per-layer launches / final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 0.882 / 0.876 | - | 0.173-0.175 | 1.323-1.331 | 1.064-1.074 | 1 (same) | 1.050-1.063 | - | 0.993-0.999 | 1.420-1.431 |
| 1 | 8 | 0.917 / 0.905 | - | 0.173-0.175 | 1.330-1.342 | 1.056-1.062 | 1.021-1.029 | 1.053-1.064 | - | 1.023-1.034 | 1.565-1.592 |
| 1 | 16 | 0.904 / 0.860 | - | 0.150 | 1.363-1.386 | 1.122-1.140 | 1.015-1.025 | 1.049-1.065 | 1.134-1.139 | 1.168-1.183 | 1.542-1.574 |
| 1 | 32 | 0.841 / 0.823 | - | 0.125-0.127 | 1.410-1.428 | 1.141-1.153 | 1 (same) | 1.022-1.037 | 1.123-1.148 | 1.200-1.215 | 1.378-1.388 |
| 1 | 64 | 0.884 / 0.864 | - | 0.108-0.109 | 1.246-1.264 | 1.028-1.044 | 1.010-1.022 | 1.014-1.033 | - | 1.013-1.026 | 1.323-1.332 |
| 1 | 128 | n/a (FlyTape cannot run TP1 B128) | - | - | 1.175-1.189 | 1.037-1.048 | 1.041-1.048 | - | - | - | 1.344-1.359 |
| 2 | 5 | 0.663 / 0.668 | 0.855 / 0.877 | 0.300-0.304 | 1.345-1.360 | 1.072-1.077 | 0.998-1.014 | 1.053-1.078 | - | 1.010-1.015 | 1.435-1.442 |
| 2 | 8 | 0.693 / 0.697 | 0.894 / 0.915 | 0.300-0.304 | 1.341-1.351 | 1.071-1.085 | 1 (same) | 1.074-1.085 | - | 1.005-1.016 | 1.564-1.585 |
| 2 | 16 | 0.665 / 0.674 | 0.861 / 0.874 | 0.265-0.269 | 1.365-1.390 | 1.148-1.161 | 1.011-1.023 | 1.067-1.077 | 1.135-1.145 | 1.173-1.187 | 1.573-1.596 |
| 2 | 32 | 0.630 / 0.637 | 0.835 / 0.838 | 0.255-0.264 | 1.380-1.396 | 1.121-1.139 | 1.006-1.029 | 1.042-1.067 | 1.088-1.102 | 1.174-1.193 | 1.513-1.560 |
| 2 | 64 | 0.604 / 0.657 | 0.870 / 0.894 | 0.219-0.240 | 1.212-1.225 | 1.037-1.051 | 1 (same) | 1.019-1.032 | - | 1.065-1.075 | 1.422-1.436 |
| 2 | 128 | 0.605 / 0.632 | - | 0.222-0.231 | 1.199-1.215 | 1.055-1.067 | 1.050-1.066 | - | - | - | 1.255-1.268 |

`first` = the cell's first gated build (ATOM order and rounding, bit-exact to the faithful emulation); `r5` = the base
before this lane's restructures (no split C, no `ufold`, no `gplace="s"`); `r9` = the base of the first FlyTape bracket
(below); `cw1` = the final with one wave per (token, stream) in C; `noufold` = the final with r5's D / U plans (fold by
the last arriver); `reassoc twin` = the per-stream re-association (`inv` applied after the down GEMM per K segment)
with its own default plans; per-layer launches = one launch per HC instance. Every arm is a LOSS against the final in
every job except `reassoc` at 1:5, 2:5, 2:8 (and 1:64 in the venv jobs) and `r9` where rounds 9-10 changed the cell by
1-2 % (1:16, 1:64, 2:5, 2:16, 2:32): TIEs in some of the four jobs. The B128 cells (`gplace="s"`) have no `cw1` / `reassoc` arm (both refused
with split-C partial gates).

The first bracket (`ct1a` / `ct1b`, jobs 68250389 / 68250398, bases `base_r9`) gave TP1 0.83-0.94, TP2 0.58-0.70
(proxy 0.85-0.91); rounds 9-10 then took TP2 B128 from 0.643 / 0.660 to 0.605 / 0.632.

Reading:
- **TP1**: LB's whole chain costs 0.82-0.92 of FlyTape's four phases. B8 is the thinnest cell.
- **TP2**: FlyTape's `mix_combine` waits for the peer's partial `y` (37-46 % of its family); LB's TP2 chain reads both
  ranks' `y` from memory (no exchange in this WP). With FlyTape's TP1 combine in place of its TP2 one, LB is 0.84-0.92
  at rows 5-64 and about 0.94-0.99 at B128 (estimated, Fix round F1); against FlyTape's down + reduce + up alone LB's
  chain is 0.99-1.15x. The "LB / FT family" column at TP2 is exchange-inclusive on FlyTape's side only.
- **Rows 5** (1:5, 2:5) are compared with FlyTape B8 (FlyTape has no B5); read them as the B8 cells.
- **The re-association (DESIGN-LB S1 / E2) does not pay** on these bases: moving `inv` after the GEMM puts it into D's
  K segments (`kseg`); the twin ties at 1:5 / 2:5 / 2:8 and loses 1.01-1.21 elsewhere. Every variant built on it (`cstat` fold /
  dfold / cfold, `cfuse`) lost on the base it was measured on (section 5).
- **One persistent launch per step wins 1.26-1.60x** over per-layer launches.

### 4.2 Where the time goes (stamps, critical path per HC, both nodes agree within 0.01)

Share of the chain: C / edge 1 / D / edge 2 / U / edge 3.

| cell | round-3 base (the first stamped base) | final |
|---|---|---|
| TP1 B5 | 0.306 / 0.054 / 0.326 / 0.050 / 0.203 / 0.052 | 0.161 / 0.072 / 0.406 / 0.058 / 0.236 / 0.065 |
| TP1 B32 (`ufold` final) | 0.263 / 0.053 / 0.354 / 0.048 / 0.203 / 0.077 | 0.138 / 0.096 / 0.145 / 0.109 / 0.419 / 0.087 |
| TP1 B64 | 0.260 / 0.047 / 0.340 / 0.046 / 0.236 / 0.069 | 0.149 / 0.081 / 0.383 / 0.050 / 0.256 / 0.078 |
| TP2 B128 | 0.234 / 0.055 / 0.311 / 0.038 / 0.300 / 0.062 | 0.189 / 0.045 / 0.342 / 0.039 / 0.317 / 0.064 |

Inside D (final, plain fold): the last GEMM ends at 0.11-0.16 of the chain, the fold starts at 0.21-0.26 (after the
partial stores, `vmcnt(0)` and the arrival atomic) and ends at 0.31-0.36 (the partial loads), the activation pack ends
at 0.39: **the fold's round trip is 0.20-0.25 of the chain, the largest single item left**. `ufold` removes it from D
(D ends at 0.15-0.18) but U's staging then waits about 0.13 for the partials (every U unit reads all splits of its
tokens through the MALL), so it wins only at rows 16-32 and only with 16-row D units (5 splits x 4 K-waves).

Per phase against FlyTape (same job; LB stage + the edge after it, FlyTape phase including its sync):

| TP | rows | LB C / FT combine | LB D / FT down + reduce | LB U / FT up | LB chain in units of FT's step |
|---|---|---|---|---|---|
| 1 | 5 | 1.12 | 0.87-0.89 | 0.75 | 0.151-0.155 |
| 1 | 8 | 1.12-1.14 | 0.90-0.91 | 0.79 | 0.156-0.161 |
| 1 | 16 | 1.08-1.11 | 0.51-0.56 | 1.18-1.23 | 0.129-0.135 |
| 1 | 32 | 1.10-1.17 | 0.46-0.48 | 1.12 | 0.104-0.105 |
| 1 | 64 | 1.28-1.32 | 0.82-0.86 | 0.73-0.75 | 0.095 |
| 2 | 5 | 0.43-0.44 | 0.81-0.85 | 0.75-0.76 | 0.199-0.204 |
| 2 | 8 | 0.41-0.43 | 0.86-0.90 | 0.81-0.82 | 0.208-0.212 |
| 2 | 16 | 0.43 | 0.50-0.51 | 1.18-1.21 | 0.179 |
| 2 | 32 | 0.40-0.41 | 0.60-0.61 | 0.95-0.96 | 0.162-0.166 |
| 2 | 64 | 0.32-0.38 | 0.81-0.85 | 0.78-0.79 | 0.144-0.145 |
| 2 | 128 | 0.31-0.33 | 0.80-0.82 | 0.91-0.93 | 0.140-0.141 |

### 4.3 WaveScope before / after

ATT on Markham MI350X (`w10l2b-att2`, one capture job: the first gated build and the `base_r9` build at 1:8 and 1:64,
the traced CU 1, 4-instance launches; the earlier lane's `att1` captured the first builds alone):
- 1:8: total stall cycles -34 %; the traced CU's waves EXEC 12.8 % -> 25.3 %, WAIT 85.4 % -> 68.8 %; the grid edge's
  `s_barrier` 72 % -> 55 % of the stalls (the split C puts this CU to work in C).
- 1:64: total stall cycles -19 %; the edge barrier 33 % -> 37 % of a smaller total; `vmem_load` stalls 17 % -> 11 %.
- In both, the edge (`gdn_lb.py:1288-1290`, the grid sync's poll and barrier) is still the largest stall line, then
  the partial-fold / staging waits.

## 5. The climb (lookback, PLAN 13 / 15)

Per cell and round: every solo lever of the current base (chain levers and every D / U plan field, losers included)
plus 24 pairs / triples of the previous round's best solo levers (losers kept when they rank), each a numerics-gated
arm timed against the base in 10 shuffled rounds with two controls (a second copy and a recompiled copy). An arm is
accepted only if it WINs on both nodes in the same round with the same recipe; every other arm is "rejected on base
<hash>" in `cfg/ledger.jsonl` and re-tested in the next round's arms on the new base. Structural levers enter the
pool as soon as they are built (`ufold` in round 5, split C in 6, `gplace="s"` in 9, `pkx` in 10).

| round | arms (each on both nodes) | WIN on both | accepted cells | what was accepted (examples) |
|---|---|---|---|---|
| 1 | 696 | 55 | 12 | `sig=fast` (8 cells), `u:depth`, `u:tb=1` |
| 2 | 1,521 | 200 | 12 | register prefetch of D / U (`pre=DU`), `map=xcd`, `d:kw=2` |
| 3 | 1,446 | 55 | 12 | gates off C (`gplace` d / u: 0.874-0.911 in 10 cells), `uvec=pk` |
| 4 | 1,427 | 85 | 11 | plan depth / `xu` / `ticket` / `lpad` combinations, `cvec=pk` at B128 |
| 5 | 1,777 | 28 | 6 | **`ufold` with 16-row D units** (0.930-0.979 at rows 16-32) |
| 6 | 1,619 | 73 | 10 | **split-stream C `cw=4`** (0.943-0.983 in 9 cells) |
| 7 | 1,609 | 63 | 6 | `cw=2` at B64, prefetch / plan combinations |
| 8 | 1,600 | 17 | 4 | `inv=rsq` combinations, `ticket` |
| 9 | 1,618 | 39 | 7 | **`cw=2, gplace="s"` at B128** (0.948 TP2, 0.965 TP1) |
| 10 | 1,613 | 9 | 2 | U plan combinations (about 0.99); `cvec="pkx"` bitwise and a tie |

Retained as default-off knobs (losers on the base named in the ledger, re-tested every round): `norm=reassoc` and its
`cstat` fold / dfold / cfold forms; `cfuse` (C fused into D's staging, 1.18-1.45: every row group redoes the combine of
its K range); `waves=8` / `w8cw2`; `acq` wave / cta; `cvec=dot2` (numerics refused) and `pk` below B128; `ufold` with
the stock D plans (1.06-1.12); `gplace="s"` below B128 (1.05-1.09: the dot partials lengthen C); the 1-split D plans
(`kw` 2-4: 1.07-1.59); E1's fold / staging / prefetch toggles; epilogue and LDS-padding toggles.

## 6. Consequence for the W7 budget (DESIGN-LB 15.7; derived, no new end-to-end run)

W7 priced the HC by its GEMM phases only (FlyTape share 0.086 / 0.136 / 0.109 at TP1 B32 / TP2 B32 / TP2 B128, x
0.50-0.78 from WP-L2a, central 0.64) and put FlyTape's combine / reduce phases in "the rest" (TP1: LB 0-0.02 of the
step; TP2: LB's 96 exchanges 0.048-0.061, with no C or fold cost). Measured, LB's whole chain costs (LB / FT family) x
(FT family share), in units of FlyTape's step (section 4.2, last column):

| TP | B | W7 HC term (central) | measured LB chain | delta | W7 K0 central -> with the measured chain | bar (1k / 8k) |
|---|---|---|---|---|---|---|
| 1 | 8 | about 0.065 (B32 shares borrowed) | 0.151-0.161 | +0.09-0.10 | 0.75 -> about 0.84-0.85 | 0.912 / 0.852 |
| 1 | 16 | about 0.065 | 0.129-0.135 | +0.06-0.07 | 0.78 -> about 0.85 | 0.914 / 0.923 |
| 1 | 32 | about 0.065 | 0.104-0.105 | +0.04 | 0.73 -> about 0.77 | 0.955 / 0.951 |
| 1 | 64 | about 0.065 | 0.095 | +0.03 | 0.77 -> about 0.80 | 0.987 / 0.979 |
| 2 | 8 | 0.087 (B32 shares borrowed; + exchanges) | 0.199-0.212 (no exchange) | +0.11-0.13 | 0.71 -> about 0.83 | 0.917 / 0.890 |
| 2 | 16 | 0.087 (+ exchanges) | 0.179 | +0.09 | 0.72 -> about 0.81 | 0.974 / 0.962 |
| 2 | 32 | 0.087 (+ exchanges) | 0.162-0.166 | +0.08 | 0.76 -> about 0.84 | 1.000 / 1.000 |
| 2 | 64 | 0.078 (+ exchanges) | 0.144-0.145 | +0.07 | 0.74 -> about 0.81 | 0.993 / 0.997 |
| 2 | 128 | 0.070 (+ exchanges) | 0.140-0.141 | +0.07 | 0.81 -> about 0.88 | 1.000 / 1.000 |

- **TP1 B8 at 8k loses its margin** (about 0.84-0.85 against 0.852). W7 already flagged TP1 B8 / B16 as thin and MTP
  (WP-L9) as their remedy; TP1 B8's other families also borrow B32 shares, so this cell needs its own share trace.
- **TP2 B128 moves to about 0.88 central.** The 10,000 tok/s target on 2 GPUs at B128 (PLAN decision 20) needs the
  LB step at or below 0.774 of FlyTape's. W7's central (0.81) already missed it; the measured HC chain widens the gap
  by about 0.07. **This chain is not the solved part** (corrected in the Fix round): exchange-free it is near parity
  with FlyTape's mixers at TP2 B128 (about 0.94-0.99), it costs 0.140 of FlyTape's step against W7's 0.070, its C
  stage is 1.65-1.77x FlyTape's TP1 B64 combine there, and it still has to absorb the TP2 exchange (WP-L5). The room
  for 10k is in this chain (C, the D fold's round trip, the edges), in the MoE (0.52 of the LB step at TP2 B128) and in
  the layer composition (WP-L3: C fused into the producer of `y`, `combine=fused_next`).
- Every 2-GPU cell stays under its bar; so do TP1 B16-B64.

## 7. Open items

- **C is the one phase slower than FlyTape's** (1.08-1.32x its combine at TP1): C fused into the producer of `y`
  (out-projection / MoE combine epilogue: no C -> D edge; WP-L3 `fuse_out` / `combine=fused_next`).
- **D's fold round trip** (0.20-0.25 of the chain): data-carrying (tagged) partials polled by a designated folder
  instead of store + `vmcnt(0)` + atomic + reload (the mono's tagged mailboxes); `part=bf16` (tolerance class).
- **Edges** (0.15-0.30 of the chain): per-edge flags instead of grid syncs, decided per edge in situ (WP-L3).
- The TP2 exchange (WP-L5); the accuracy suites for the tolerance-class bases (`sigmoid_fast`, `v_rsq`, the split
  square-sum order); B8 FlyTape share traces for the B8 budget cells; `HcKnobs.check`'s LDS estimate (577 COMGR
  refusals at B128).
- Rejected on their base and re-tested every round: section 5.

Alola facts recorded by this lane: `.claude/skills/alola-cluster/SKILL.md` rules 230 and 235.

## Verification

Independent verifier (W10), 2026-11-02. Job `w10-vWP-L2b-v1` (68251295, container, n01-09, bench GPU bus 05, not the
bus 75 of `ct2a` / `ct2b`): the same strict prebuilt stage `w10-l2b-hs23` (all objects loaded, 0 refused, no COMGR
build), the same staged harness / rocKE tree (`h23` / `rk21`; the job's sha256 of `hc_lb.py`, `e1.py`, `e1_plans.py`,
`gdn_lb.py`, `l2b_bench.py`, `job.sh`, `gate.py` equal the local files), `cfg/base_r11`, a different seed (7331), 96
instances per launch (FlyTape's step count) instead of 128, 16 shuffled rounds, FlyTape traced TP1 B8 and TP2 B128 before
and after in the same job. Absolute numbers: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2b.md`
(Verification) and `wp-l2b-raw/vWP-L2b-v1/`.

**Reproduced.**
- Numerics on the timed objects (both cells): dbg twin PASS, timed object PASS, determinism / walk / recompiled copy
  equal, checker controls fail as required, mutants 12 / 12 and 13 / 13 caught, the `early` stale twin changes the bits
  and `delay` keeps them. Compile / hazard gate on (no `--no-gate`), GPU health precheck run.
- LB / FlyTape mixer family: **TP1 B8 0.934** (claimed 0.917 / 0.905; LB 1.015x the `ct2a` time on another GPU and
  L = 96), **TP2 B128 0.579** (claimed 0.605 / 0.632; FlyTape's TP2 combine was 1.18x larger here). FlyTape's
  before / after traces within 0.015; its traced step within 0.4 % of its untraced step. Controls `hc#2` 0.998 / 0.999,
  `hc_rc` 0.996 / 0.993. Per-layer launches 1.554 / 1.243 (claimed 1.565-1.592 / 1.255-1.268). Stamp shares match 4.2
  (TP2 B128 C 0.188 / D 0.340 / U 0.317 against 0.189 / 0.342 / 0.317).
- `l2b_report.py` re-run on the fetched `ct2a` / `ct2b` raw data gives every ratio of the 4.1 table exactly.
- Ledger: all 82 accepted changes have a WIN / WIN record for the same round, cell, base and arm (0 exceptions).
- CPU tests: `test_qwen38_lb_hc_lb.py` 56 passed locally (parents' IR unchanged by HC builds, knobs refused where not
  built).

**Issues found (none overturns the gate; the first changes how the TP2 numbers must be read).**
1. **The TP2 headline (0.60-0.70, B128 0.605 / 0.632) is not a like-for-like mixer comparison and the B128 cell has no
   fair figure.** FlyTape's TP2 `mix_combine` carries its exchange wait (45-50 % of its family at B128 here); LB's TP2
   chain has no exchange. WP-L2a compared the HC mixers against the faster of FlyTape's TP1 and TP2 phases; on that
   convention TP2 B128 has no reference (FlyTape has no TP1 B128), and LB's whole chain is **1.11 / 1.15 / 1.16x
   FlyTape's down + reduce + up alone** (`ct2a` / `ct2b` / this job). With FlyTape's TP1 combine extrapolated from
   B32 / B64 to B128 (an estimate, not measured), the exchange-free TP2 B128 ratio is about **0.93-0.98**, not 0.61-0.63.
   So the summary's "the HC chain is already at 0.61-0.63 of FlyTape's mixers there, so the remaining room for 10k is
   the MoE and layer composition, not this chain" is not supported: the chain is near parity with FlyTape's
   exchange-free mixers at TP2 B128, costs 0.140 of FlyTape's step against W7's 0.070, and still has to absorb the TP2
   exchange (WP-L5). The section 6 budget arithmetic itself (0.81 -> about 0.88) uses the measured chain cost and is
   correct.
2. **Rows 5 are compared with FlyTape B8** (`l2b_report.py` uses `max(rows, 8)`; FlyTape was only run at B8); not
   stated in 4.1. LB's own B5 is 0.96-0.97 of its B8, so the 1:5 / 2:5 ratios flatter LB by about that much.
3. The per-stage criterion of DESIGN-LB 15.4 ("in-situ phase <= FlyTape's per stage") is not met for C (1.08-1.32x
   FlyTape's combine at TP1); disclosed in 4.2 / 7, but the gate line reads as met without it.
4. Census: the fetched raw data hold 37 jobs (`n1`, 68245388, was never fetched) and give 862 / 832 cells, 10,826 /
   10,848 mutants, 854 / 854 twins and checks; the report's 864 / 834, 10,852 / 10,874 and 856 / 856 include the
   unfetched job. The failing cells are as stated (22 in `n2`, 8 `cfuse` B128).
5. The 10,000 tok/s goal is not reached or measured here (as the report says): FlyTape's TP2 B128 step in this job
   implies about 0.78 of the step is needed; nothing served runs this chain.

**Verdict: PARTIAL.** The WP-L2b build gate (numerics suite, stale twins, strict prebuilt objects, same-job FlyTape,
both nodes, WIN-on-both-nodes climb) is exercised and reproduces on an independent GPU; TP1 rows 8-64 beat FlyTape's
mixer family with margin (0.82-0.93). The TP2 "0.60-0.70" and the B128 "0.605 / 0.632" are exchange-inclusive on
FlyTape's side only; the fair TP2 B128 reading is near parity (about 0.93-0.98, estimated), and the conclusion that no
room for 10k remains in this chain does not follow.

## Fix round

W10, 2026-11-02, against the Verification issues above. No new GPU timing was needed: every correction below is
computed from measured same-job data already in `wp-l2b-raw/` (`ct2a` 68251074, `ct2b` 68251078, `ct1a` / `ct1b`,
`vWP-L2b-v1` 68251295) plus the one job that was never fetched (`n1`, 68245388, fetched now). Script:
`hackathon-prep/qwen38/truemono-lb/wp-l2b-fix/tp2_fair.py` (ratios; `--abs` only into private-perf). The body above
(verdict, 3, 4.1 Reading, 6) is corrected in place.

**F1. TP2 framing (issues 1, 2).** Every TP2 figure now names what FlyTape's side contains. Per TP2 cell, LB's chain
over: (a) FlyTape's family as it runs (its combine = peer wait + peer sum + combine); (b) its down + reduce + up only
(a hard upper bound on the fair ratio); (c) its family with the combine replaced by its measured TP1 combine at the
same batch; at B128, where FlyTape has no TP1 run, (d) its TP1 B64 combine (an upper bound on the ratio if FlyTape's
combine does not get cheaper with more rows) and (e) its B64 combine grown by its own B32 -> B64 factor (estimate).

| job | rows | (a) as run | (b) no combine | (c) TP1 combine, same batch | (d) TP1 B64 combine | (e) growth extrapolation |
|---|---|---|---|---|---|---|
| ct2a | 5 / 8 / 16 / 32 / 64 | 0.663 / 0.693 / 0.665 / 0.630 / 0.604 | 1.042 / 1.089 / 1.051 / 0.995 / 1.027 | 0.855 / 0.894 / 0.861 / 0.835 / 0.870 | - | - |
| ct2b | 5 / 8 / 16 / 32 / 64 | 0.668 / 0.697 / 0.674 / 0.637 / 0.657 | 1.072 / 1.118 / 1.072 / 1.010 / 1.050 | 0.877 / 0.915 / 0.874 / 0.838 / 0.894 | - | - |
| ct2a | 128 | 0.605 | 1.111 | n/a | 0.960 | 0.942 |
| ct2b | 128 | 0.632 | 1.146 | n/a | 0.993 | 0.989 |
| ct1a / ct1b (`base_r9`) | 128 | 0.643 / 0.660 | 1.226 / 1.207 | n/a | 1.063 / 1.042 | 1.059 / 1.027 |
| vWP-L2b-v1 (verifier, L 96) | 128 | 0.579 | 1.159 | n/a | n/a (no TP1 trace) | n/a |

Stage split at TP2 (stamps x time, LB stage + the edge after it): LB's C / FlyTape's TP1 combine 1.12-1.30 at rows
5-64 and 1.65-1.77 at B128 (against TP1 B64); LB's D + U + edges / FlyTape's down + reduce + up 0.76-0.87 in every
cell (0.85 / 0.87 at B128). (c)-(e) favour LB slightly: FlyTape's TP1 combine loads one `y`, LB's TP2 C sums both
ranks'.

Reading: **exchange-free, the chain is near parity with FlyTape's mixers at TP2 B128 (about 0.94-0.99, estimated;
0.96 / 0.99 bounded under the stated assumption), and was a loss (1.03-1.06) on the round-9 bases.** The 0.60-0.70 /
0.605-0.632 headline was FlyTape's exchange wait, not a mixer win. The earlier claim "the remaining room for 10k is not
in this chain" is withdrawn: the chain costs 0.140 of FlyTape's TP2 B128 step (W7 assumed 0.070), its C stage is the
largest per-stage loss (1.65-1.77x), and it has not yet absorbed the TP2 exchange (WP-L5). The budget arithmetic in 6
(0.81 -> about 0.88 central against a 0.774 need) is unchanged.

**F2. Rows 5 (issue 3).** `l2b_report.py` keys FlyTape by `max(rows, 8)`; FlyTape has no B5 configuration (as in
WP-L2a / WP-L2c). Now stated in the verdict and 4.1. LB's own B5 is 0.962 / 0.969 (TP1) and 0.957 / 0.959 (TP2) of its
B8, so if FlyTape's B5 is cheaper than its B8 by the same factor, 1:5 / 2:5 read as the 1:8 / 2:8 cells (TP1 0.917 /
0.905, TP2 exchange-free 0.894 / 0.915).

**F3. Per-stage criterion (issue 4).** The gate line now states that DESIGN-LB 15.4's per-stage criterion is not met
for C (1.08-1.32x FlyTape's combine at TP1; 1.12-1.30 at TP2 rows 5-64 and 1.65-1.77 at B128 against its TP1 combine).
D + edge (0.46-0.91) and U + edge (0.73-1.23; above 1 only in the `ufold` cells, where U carries D's fold) are as in
4.2.

**F4. Census (issue 5).** `n1` (68245388) fetched into `wp-l2b-raw/n1/`; `num_sum.py` over all 38 jobs: 862 cells /
832 PASS, mutants 10,826 / 10,848, stale twins, delay controls, determinism, walk, recompiled copy, dbg twin and
checker controls 854 / 854 each, 72 / 72 bitwise-lever checks. `n1` contributes no recorded cell: its 3 cells passed the
dbg-twin / determinism / walk / recompiled checks and then the stage checker stopped (`ref_quantize_mxfp8 needs finite
input`) before a verdict; nothing was timed. The earlier 864 / 834, 10,852 / 10,874 and 856 / 856 had no data behind
them and are replaced. `n4a` / `n4b` (argument error before GPU work) and `f2d` (cancelled) have no data.

**F5. 10,000 tok/s (issue 6).** Not measured, and not reachable from this WP: nothing served runs this chain before
WP-L6, and the derived TP2 B128 step (about 0.88 of FlyTape's central) is above the 0.774 that 10k needs. What this WP
contributes toward 10k, by measured size at TP2 B128: C (0.19 of the chain plus its edge 0.045; 1.65-1.77x FlyTape's
TP1 combine), the D fold's round trip (0.20-0.25), the edges (0.15-0.30). A served 10k number needs WP-L3 / L5 / L6.

**Verdict after the fix round: PARTIAL for the program's 10k goal; WP-L2b's build gate holds with the corrected
reading** (TP1 0.82-0.92, TP2 exchange-free 0.84-0.92 at rows 5-64 and about 0.94-0.99 at B128; C misses its
per-stage target).
