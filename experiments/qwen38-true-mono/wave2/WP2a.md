# WP2a: the HC stages A1, A2, M1, M2 as rocKE stage functions (wave 2, W8)

D0 section 5 wave 2 (WP2a: "HC per D0.3"; every spread value, the dedupe and the M1 placements as knobs; N1 hooks
reserved), PLAN decision 16 (dump FlyDSL's code, reach parity by construction, then crush it with the parametric search),
decisions 13 / 15 (lookback with the losers kept, rejections "on base X", re-tests on the new base). Ratios, shares and
counts only. Absolute times: `hackathon-prep/private-perf/qwen38-true-mono/wave2/WP2a.md`. Below 1 is faster.

## 0. Summary

**Built, bitwise at every served width, and faster than FlyDSL's HC stages in every served cell: the pick's layer median
is below the FlyDSL-HC stack's in 46 of 48 confirmation runs (12 cells x 2 per node x 2 nodes), a DESIGN 9.1 WIN in all
four runs in 9 of the 12 cells, and no run loses.**

- The four HC stages (A1, A2, M1, M2) are rocKE stage functions (`stages2/hc.py`) for all three kinds at every narrow
  width the K=0..3 CONC 1 path serves (`r1`, `r2s`, `r3s`, `r4s`) plus `r2-r4`, with every P3 spread value (D0.3's
  `spread.a1=10` at rows 1-3, variant (ii) `spread=2`, the dedupe values, the M1 placements) and the N1 hook
  (`acc=2`, tolerance class) reserved. They link into ATOM-truemono's layer through the extern-stage ABI (one additive,
  default-off ATOM hook passes `a1_preload`'s registers; identity gate: every existing build's `.text` unchanged) and
  build as Path B framework kernels (bitwise vs FlyDSL's stages there too).
- **Parity by construction** (section 3; the `.ll` half in section 11.2): the same MFMA count, chain order and polls as
  FlyDSL in the ISA, the same arithmetic in the `.ll` (all 60 stage pairs, and all 108 grouped / wide ones of the fix
  round); A1's VALU at 0.972-1.008 of FlyDSL's over the 15 rows (fix round correction: this line said 1.003-1.004,
  which holds for delta / qsa r1 only) after three bitwise codegen fixes (`umax` amax, the row-shift exchange, SGPR-anchored LDS
  offsets); one redundant barrier removed. The catalogue's findings (A1 VALU-bound; 19-20 MFMAs after `lgkmcnt(0)`, 18
  chained on one accumulator; scale registers rewritten inside the chain) are reproduced and each is a knob (`sgb` + `ra`:
  MFMAs after `lgkmcnt(0)` 20 -> 1, scale rewrites 9 -> 1; `scale=opsel`; `acc=2` for the chain, tolerance class).
- **Numerics**: every bitwise knob value in every served cell, and the port at all 21 narrow cells and every spread
  value, is bitwise (581 in-situ checks on the final code; `acc=2` is the tolerance class), and 178 of 178 must-fail
  mutants are caught.
- **Timing** (tuned stack's layer, the picks confirmed twice per node on both nodes, DESIGN 9.1): the layer runs at
  0.973-0.994 of the FlyDSL-HC stack at delta r1, delta r3s / r4s, delta_ple r2s-r4s, qsa r3s / r4s (WIN in 4 of 4
  runs), 0.988-0.997 at qsa r1 (4 of 4), 0.973-1.001 at qsa r2s and 0.992-1.002 at delta_ple r1 (3 of 4 WIN, the fourth
  neutral), and 0.994-0.996 at delta r2s (2 WIN, 2 neutral below 1). No cell loses. The 1:1 port alone already wins at
  delta r1, qsa r1 / r4s.
- **Rust direct backend**: bitwise, but 1.03-1.11x of COMGR on these kernels; no lever or combination of the 24 wins:
  COMGR stays. **WaveScope / stamps**: ATT per-stage attribution is unreliable on this schedule (CTA roles); the stamp
  timelines show the picks finishing the HCs 3-6 % earlier (delta_ple r2s, qsa r3s).

## 1. What was built

| piece | where |
|---|---|
| the emitter: A1 / M1 (stream norm, MXFP8 pack, 20-step scaled-MFMA down chain, injection, the P3 spread variants 1-11 incl. the quarter pack and the dedupe rows) and A2 / M2 (up chain, mix, stream update), every narrow width `r1-r4`, `r2s-r4s`, the three kinds; `HcKnobs` (section 4); `HcSpec`; `write_hc_functions`; framework form (`framework_factory`, `stage_fn` registry); CLI | `copy-rocm-lib/.../qwen38_mono/stages2/hc.py` (new module in the new package `stages2/`) |
| CPU tests (79): LDS arena equal to ATOM's `for_rows` at every width, every variant and kind lowers, every knob changes the IR or is refused, inert knobs, spec round trip, the `pre_args` ABI and sidecars, `llvm-as` validity of every knob set, framework layer kernels, the registry | `copy-rocm-lib/.../library/tests/test_qwen38_stages2_hc.py` |
| ATOM-truemono hook (additive, default off): an extern A1 may take `a1_preload`'s 24 registers as extra arguments (`pre_args`, named in the sidecar), and a function's `hc.spread` must equal the build's spread lever of its HC (refused otherwise) | `ATOM-truemono/.../kernels/extern_stages.py`, `kernels/kernel.py` (`_run_extern`) |
| in-situ harness: bitwise gate (output, every state, the base's whole logical mailbox part, NUMA copies, determinism), must-fail mutants, DESIGN 9.1 timing with two controls, stamps timelines | `hackathon-prep/qwen38/truemono/WP2a/insitu_hc.py`, `stamps_hc.py` |
| tuner backend + driver (the HC knobs per stage + the HC schedule levers hold / prio / head_sleep / spread; signed lookback, order restarts, TPE; per-config bitwise gate; external hang watcher) | `WP2a/hc_tune.py`, `WP2a/job_tune.sh` |
| reverse jobs (layer dumps with / without the functions, per-stage split and diff, parity table), ATOM identity gate, framework + Rust direct job, WaveScope job | `WP2a/job_rv.sh`, `rv_summary.py`, `job_ident.sh`, `job_rt.sh` / `rt_hc.py`, `job_att.sh` / `att_hc.py` / `att_hc_cmp.py` |

No existing rocKE file was edited: `stages2/hc.py` and the test are new and nothing outside `stages2/` imports them, so
every existing rocKE build is byte-identical by construction. The ATOM-truemono edits are additive and default off; the
identity gate (`w8-wp2a-ident1`) rebuilt 13 layer builds with and without the edit (plain, the best stack, with the
existing extern stages) and every `.text` is identical (keys-only compare), while a build with a mutant HC function
differs as required. (Fix-round correction: "ATOM's CPU tests pass (104)" was wrong. Those 104 were rocKE tests (three
files, 18 skipped) run at 12:19 on the code before `hc.py` and its test reached their final versions; ATOM's own
extern-stage tests had not been run. On the final code: section 11.6.)

## 2. Method as run (PLAN decision 16)

1. **Dump FlyDSL's code** of the mono layer at every served width, with the FlyDSL HC stages and with the rocKE functions
   linked through `ATOM_Q38_EXTERN_STAGES` (reverse toolkit; debug builds with stage markers; per-stage split and diff;
   jobs `w8-wp2a-rv1` (first port) and `rv2` (final defaults: 3 kinds x `r1 r2s r3s r4s r4`, 4 sets, 60 code objects)).
2. **Port, then prove parity by construction and by bits.** Same operations and order as FlyDSL's `stages/hc.py`
   (`_a1_task`, `a2`, `_spread_a1` / `_spread_down` / `_spread_pack` / `_quarter_pack`): same MFMA intrinsic and operand
   formats, the same 20-step K order on one accumulator, the same bf16 / f32 rounding points, the same lane exchanges, the
   same polls; per stage in the ISA (section 3) and bit for bit in situ (section 5).
3. **Every hard-coded choice and every catalogued inefficiency is a knob** (section 4); losers stay in the pools.
4. **Climb** with the mono tuner per served cell, on the tuned stack (section 6).
5. **Time in situ** on the tuned stack's layer (`truemono-quick/lookback/hyb_layer`: the stack's FlyDSL pick with the
   rocKE MoE / B3 / GEMV functions it already links; 16 cold weight copies, shuffled rounds, a second graph and a
   recompiled build as controls; DESIGN 9.1 win rule), both rocke nodes.
6. **Rust direct backend** (all 24 levers, single and combined) and **WaveScope / stamps** before and after (sections 7, 8).

## 3. Parity by construction: what the FlyDSL code and the port compile to

Per stage at the served widths (job `w8-wp2a-rv2`, final defaults, plain schedule). Columns: instructions / VALU /
MFMAs right after `s_waitcnt lgkmcnt(0)` / MFMAs accumulating into the previous MFMA's destination / scaled MFMAs with a
scale VGPR written since the previous MFMA / barriers. `port` = the defaults (FlyDSL's choices), `sgb` = `sgb=1,ra=1`,
`opsel` = `sgb=1,ra=1,scale=opsel`.

| stage, kind, width | FlyDSL | port | sgb | opsel |
|---|---|---|---|---|
| A1 delta r1 | 2134 / 1637 / 20 / 18 / 9 / 6 | 2142 / 1644 / 19 / 18 / 9 / 6 | 2133 / 1644 / 1 / 18 / 1 / 6 | 2166 / 1664 / 1 / 18 / 3 / 6 |
| A2 delta r1 | 529 / 287 / 2 / 2 / 0 / 4 | 500 / 267 / 3 / 2 / 1 / 4 | 500 / 267 / 1 / 2 / 0 / 4 | 504 / 270 / 1 / 2 / 0 / 4 |
| M1 delta r1 | 2195 / 1649 / 20 / 18 / 9 / 6 | 2207 / 1655 / 19 / 18 / 9 / 6 | 2194 / 1655 / 1 / 18 / 1 / 6 | 2228 / 1675 / 1 / 18 / 3 / 6 |
| M2 delta r1 | 486 / 220 / 2 / 2 / 0 / 5 | 455 / 206 / 3 / 2 / 1 / 5 | 454 / 206 / 1 / 2 / 0 / 5 | 460 / 209 / 1 / 2 / 0 / 5 |
| A1 delta r4s | 2422 / 1638 / 20 / 18 / 9 / 6 | 2170 / 1643 / 19 / 18 / 9 / 6 | 2160 / 1643 / 1 / 18 / 1 / 6 | 2191 / 1663 / 1 / 18 / 3 / 6 |
| A2 delta r4s | 1144 / 732 / 2 / 2 / 0 / 5 | 1057 / 676 / 3 / 2 / 1 / 5 | 1059 / 676 / 1 / 2 / 0 / 5 | 1063 / 679 / 1 / 2 / 0 / 5 |
| M1 delta r4s | 2257 / 1685 / 20 / 18 / 9 / 7 | 2229 / 1658 / 20 / 18 / 9 / 7 | 2219 / 1658 / 1 / 18 / 1 / 7 | 2251 / 1678 / 1 / 18 / 3 / 7 |
| M2 delta r4s | 892 / 469 / 2 / 2 / 0 / 5 | 819 / 433 / 3 / 2 / 1 / 5 | 821 / 433 / 1 / 2 / 0 / 5 | 825 / 436 / 1 / 2 / 0 / 5 |
| A1 qsa r1 | 2135 / 1638 / 20 / 18 / 9 / 6 | 2150 / 1647 / 19 / 18 / 9 / 6 | 2141 / 1647 / 1 / 18 / 1 / 6 | 2172 / 1667 / 1 / 18 / 3 / 6 |
| A2 qsa r1 | 519 / 287 / 2 / 2 / 0 / 4 | 490 / 267 / 3 / 2 / 1 / 4 | 491 / 267 / 1 / 2 / 0 / 4 | 495 / 270 / 1 / 2 / 0 / 4 |
| M1 qsa r1 | 2197 / 1650 / 20 / 18 / 9 / 6 | 2204 / 1656 / 19 / 18 / 9 / 6 | 2195 / 1656 / 1 / 18 / 1 / 6 | 2227 / 1676 / 1 / 18 / 3 / 6 |
| M2 qsa r1 | 489 / 223 / 2 / 2 / 0 / 5 | 460 / 208 / 3 / 2 / 1 / 5 | 458 / 208 / 1 / 2 / 0 / 5 | 464 / 211 / 1 / 2 / 0 / 5 |
| A1 qsa r4s | 2152 / 1639 / 20 / 18 / 9 / 6 | 2173 / 1646 / 19 / 18 / 9 / 6 | 2163 / 1646 / 1 / 18 / 1 / 6 | 2194 / 1666 / 1 / 18 / 3 / 6 |
| A2 qsa r4s | 1136 / 732 / 2 / 2 / 0 / 4 | 1046 / 676 / 3 / 2 / 1 / 4 | 1048 / 676 / 1 / 2 / 0 / 4 | 1052 / 679 / 1 / 2 / 0 / 4 |
| M1 qsa r4s | 2256 / 1685 / 20 / 18 / 9 / 7 | 2227 / 1657 / 20 / 18 / 9 / 7 | 2216 / 1657 / 1 / 18 / 1 / 7 | 2247 / 1677 / 1 / 18 / 3 / 7 |
| M2 qsa r4s | 896 / 470 / 2 / 2 / 0 / 5 | 820 / 433 / 3 / 2 / 1 / 5 | 822 / 433 / 1 / 2 / 0 / 5 | 826 / 436 / 1 / 2 / 0 / 5 |
| A1 delta_ple r1 | 2197 / 1652 / 20 / 18 / 9 / 6 | 2201 / 1659 / 19 / 18 / 9 / 6 | 2192 / 1659 / 1 / 18 / 1 / 6 | 2224 / 1679 / 1 / 18 / 3 / 6 |
| A2 delta_ple r1 | 485 / 220 / 2 / 2 / 0 / 5 | 461 / 207 / 3 / 2 / 1 / 5 | 460 / 207 / 1 / 2 / 0 / 5 | 465 / 210 / 1 / 2 / 0 / 5 |
| M1 delta_ple r1 | 2204 / 1650 / 20 / 18 / 9 / 6 | 2205 / 1657 / 19 / 18 / 9 / 6 | 2200 / 1657 / 1 / 18 / 1 / 6 | 2232 / 1677 / 1 / 18 / 3 / 6 |
| M2 delta_ple r1 | 479 / 217 / 2 / 2 / 0 / 5 | 454 / 205 / 3 / 2 / 1 / 5 | 453 / 205 / 1 / 2 / 0 / 5 | 458 / 208 / 1 / 2 / 0 / 5 |
| A1 delta_ple r4s | 2256 / 1689 / 20 / 18 / 9 / 6 | 2225 / 1657 / 19 / 18 / 9 / 6 | 2215 / 1657 / 1 / 18 / 1 / 6 | 2249 / 1677 / 1 / 18 / 3 / 6 |
| A2 delta_ple r4s | 901 / 466 / 2 / 2 / 0 / 6 | 825 / 429 / 3 / 2 / 1 / 6 | 827 / 429 / 1 / 2 / 0 / 6 | 828 / 432 / 1 / 2 / 0 / 6 |
| M1 delta_ple r4s | 2261 / 1686 / 20 / 18 / 9 / 7 | 2230 / 1658 / 19 / 18 / 9 / 7 | 2219 / 1658 / 1 / 18 / 1 / 7 | 2251 / 1678 / 1 / 18 / 3 / 7 |
| M2 delta_ple r4s | 886 / 463 / 2 / 2 / 0 / 5 | 807 / 427 / 3 / 2 / 1 / 5 | 809 / 427 / 1 / 2 / 0 / 5 | 813 / 430 / 1 / 2 / 0 / 5 |

(Every kind and width: `hackathon-prep/logs/q38/w8-wp2a/rv2/parity_rv2.md`; the first port's table: `.../rv1/parity_rv1.md`.)

What the diff showed, and what each finding became:

1. **The catalogue's MFMA findings are reproduced by the port** (19-20 of 20 MFMAs after `lgkmcnt(0)`, 18 chained on
   one accumulator, 9 scale rewrites in A1 / M1, as FlyDSL's 20 / 18 / 9), and the knobs remove them: `sgb=1` (a
   `sched_group_barrier` interleave) with `ra=1` (operand read-ahead) takes the MFMAs after `lgkmcnt(0)` from 20 to 1 and the scale rewrites from 9 to 1;
   `scale=opsel` (the 20 scale words hoisted and packed 4 per dword, picked with op_sel) leaves no scale register written
   in the chain (the 3 counted are the packed words' first loads). `ra` alone and `scale=hoist` alone are undone by
   LLVM's scheduler (the ISA is unchanged), which is why `sgb` exists.
2. **The 18-deep chain is the bitwise order itself**: one accumulator, K steps in FlyDSL's order. Two accumulators
   (`acc=2`) shorten the dependency chain but change the summation order: a tolerance-class knob (the N1 hook),
   validated, never a default and never in a bitwise climb (section 5 and 6.4).
3. **A1 is VALU-bound in FlyDSL too** (VALU = 77 % of A1's instructions). The port first had 6 % more VALU; three
   bitwise codegen fixes brought it to parity (A1 VALU 0.972-1.008 of FlyDSL over the 15 rows: 0.995-1.008 for the
   memory-read A1 of delta / qsa, 0.972-1.004 for the PLE layer's polled A1; M1 0.975-1.004; A2 / M2 0.91-0.95. Fix-round
   correction: this line said 1.003-1.004):
   - FlyDSL's block amax is `llvm.maximum` on non-negative values; `amax=umax` (default) takes the unsigned max of the
     bit patterns (the same bits, NaN propagated, one VALU per step) instead of `llvm.maxnum`'s canonicalizing compares
     (`amax=fmax` kept as a knob);
   - the xor-4 lane exchange is FlyDSL's two bank-masked row shifts (`shfl4=rowshift`, default; `select` kept);
   - LDS entries past the DS 16-bit offset were addressed with a `v_mov` + `v_mad` per access: `anchor=1` (default)
     addresses them from an SGPR copy of their base, as FlyDSL's pointer arithmetic does (79 instructions fewer in A1).
4. **A redundant barrier.** rocKE's fenced barriers do not merge, so the port's A2 had one barrier more than FlyDSL's right
   after the wait barrier; it is skipped when the previous op is a barrier (barrier counts now equal in every stage).
5. **Loads.** At `r2s+` the port issues fewer, wider weight loads (the `vmem_load` column differs, e.g. 34 vs 40); waits
   are the FlyDSL pattern (`vmcnt(0)` counts equal or one fewer).
6. **Polls** are the compiler-visible `tagged_poll` (`mailbox impl=ir`, FlyDSL's form) by default; `asm` / `irf` and every
   pause (`s_nop`, `s_sleep N`, none) are knobs. `wait=prepoll` (wave 0 spins on one hint pair per producer before the
   CTA barrier) follows the spread layout's pairs.

## 4. Knobs (`HcKnobs`; defaults = FlyDSL's choices; all keep the bits except `acc`)

| knob | values (default first) | what |
|---|---|---|
| `mailbox` | impl ir / asm / irf; pause s_nop / s_sleep1 / none | poll / put implementation |
| `pause_sleep` | 0, 1, 2, 4, 8 | the poll pause as `s_sleep N` (ATOM's `pause=N` form) |
| `numa`, `numa_hdr` | the build's (fixed per stack) | ATOM's numa=1 mailbox layout |
| `spread` | the build's: 0, 1-11 | P3 spread values (1 = 80 units, 2 = 20 n-block tasks, 3 / 4 = dedupe, 10 = quarter pack; 5-9 / 11 = mutants); must equal ATOM's `spread[.a1/.m1]` lever (refused otherwise) |
| `pre` | args (A1 attention, memory-read builds), none | `a1_preload`'s registers as 24 extra arguments (FlyDSL's form) or the loads inside the stage |
| `wait` | barrier, prepoll | wait ahead of the stream / C1-side polls |
| `ra` | 0, 1, 2 | LDS operand read-ahead of the scaled-MFMA chains |
| `sgb` | 0, 1 | `sched_group_barrier` interleave of the chain |
| `scale` | ldsstep, hoist, opsel | the down chain's A scales: per step / hoisted / hoisted and op_sel-packed |
| `w_cpol`, `wi_cpol`, `nw_cpol` | 0, 1, 2, 3, 16, 17, 18, 19 | cache bits of the weight / scale, injection and norm (`h`, `w'`) loads |
| `a2order` | mono, h_first | A2 load order (norm rows and stream words before the up weights) |
| `stats` | streams, halves | polled HC's chunk square sums from the streams (VALU) or the producer's 16-column sums |
| `inv` | lds, regs | inverse RMS through LDS + barrier, or every wave computes all four |
| `amax` | umax, fmax | block amax form (section 3) |
| `shfl4` | rowshift, select | xor-4 exchange |
| `anchor` | 1, 0 | LDS high-offset addressing (section 3) |
| `hbar` | 1, 0 | the CTA barrier between the norm loads and the down weights |
| `acc` | 1, 2 | **tolerance class (N1 hook)**: K-split accumulators per chain |
| HC schedule (in the climb) | hold / prio per HC stage, head_sleep, spread.a1 / spread.m1 (all ATOM values incl. dedupe 3 / 4) | ATOM's sched levers of the four stages |
| `mutate` | test only | `eps`, `down_part`, `pack_scale`, `inj_lane`, `up_kit`, `mix_avg`, `quarter` |

## 5. Numerics (DESIGN 7.8 gate 1, in situ)

Per arm, after one launch from a restored snapshot: the layer output and every state the kind writes equal to the plain
FlyDSL mono, twice (deterministic), nonzero; the base's whole logical mailbox part (every stage's mailboxes) equal to the
FlyDSL-HC base's; both NUMA copies equal.

| job (code) | what | result |
|---|---|---|
| `w8-wp2a-num6a` (final code, n01-09) | all 21 narrow cells (`r1-r4`, `r2s-r4s` x delta / qsa / delta_ple) on the plain schedule; every spread value FlyDSL builds at each width (`spread[.a1/.m1]` = 1, 2, 3, 4 (dedupe), 10 (quarter pack) and the reduce mutants 6, 9) on a FlyDSL base with that schedule; the 12 served cells on the tuned stack; qsa at 64K and 128K context | 173 arm checks: every arm equals its FlyDSL base bit for bit; the 6 arms on the spread-mutant bases (`spread.a1=6` at r1 and `9` at r4s, one per kind; fix-round correction: this said 24, which counted the `_base` lines and two steps per base) equal the mutant base and differ from the plain mono, as designed (they are the gate's own must-fail cases); 178 of 178 must-fail mutants caught |
| `w8-wp2a-num6b` (final code, n07-17) | every knob value (33 arms: `ra`, `sgb`, `scale`, every cache-bit value, `a2order`, `wait`, `stats`, `hbar`, `anchor`, `amax`, `shfl4`, `inv`, `pause` forms, `mailbox impl`, `pre=none`, combinations, `acc=2`) on all 12 served stack cells | 408 checks: every bitwise knob value bitwise in every cell; `acc=2` (tolerance class) deterministic, output and states equal, and 1 / 10 bytes of the internal stream copies `isc` differ in delta_ple r1 / qsa r2s (it is bitwise elsewhere by luck of the data) |
| `num1`, `num2`, `num4`, `num5a/b` (earlier code) | the same matrices on the first port and the intermediate fixes | all pass; kept for the history |
| every tuner evaluation | each config, before timing; finalists again after the confirmation | gate failures 0 in all climbs |

The must-fail mutants (`down_part`, `pack_scale`, `inj_lane`, `up_kit`, `mix_avg`, `quarter`, and `eps` = the norm
epsilon off by 2^20 ulp) each change one value the bitwise gate must see; every one was caught in every cell it applies
to. The order is proven by construction (section 3: the same MFMA sequence on one accumulator).

## 6. Timing in situ (DESIGN 9.1; layer time over the tuned stack's FlyDSL-HC build, same job)

### 6.1 The port (FlyDSL's choices) and first arms (job `w8-wp2a-t1`, n07-17, 30 rounds)

| cell | port | port, `pre=none` | findings set (`ra=1,sgb=1,scale=opsel`) | `ra=2,sgb=1,scale=hoist` | `w_cpol=2` (nt) | controls (min-max) |
|---|---|---|---|---|---|---|
| delta r1 | **0.9881 WIN** | 0.9926 WIN | 0.9901 WIN | 0.9870 WIN | 0.9879 WIN | 0.9979-1.0020 |
| qsa r1 | **0.9878 WIN** | 0.9874 WIN | 0.9876 WIN | 0.9839 WIN | 0.9866 WIN | 0.9974-1.0014 |
| delta_ple r1 | 0.9984 | 0.9985 | 1.0060 LOSS | 1.0050 LOSS | 0.9954 | 0.9957-1.0016 |
| delta r2s | 1.0039 | 1.0088 LOSS | 1.0102 LOSS | 1.0060 LOSS | 1.0129 LOSS | 0.9991-1.0026 |
| qsa r2s | 1.0110 LOSS | 1.0160 LOSS | 1.0064 LOSS | 1.0119 LOSS | 1.0111 LOSS | 0.9977-1.0018 |
| delta r3s | 1.0034 | 1.0085 LOSS | 1.0044 | 1.0029 | 1.0099 LOSS | 0.9987-1.0060 |
| qsa r3s | 0.9985 | 1.0015 | **0.9898 WIN** | 0.9929 WIN | 0.9964 WIN | 0.9984-1.0020 |
| delta r4s | 1.0024 | 1.0064 LOSS | 1.0085 LOSS | 1.0056 LOSS | 1.0408 LOSS | 0.9988-1.0022 |
| qsa r4s | **0.9856 WIN** | 0.9954 WIN | 0.9867 WIN | 0.9883 WIN | 1.0073 LOSS | 0.9985-1.0025 |
| delta_ple r4s | 1.0039 | 1.0047 LOSS | 1.0068 LOSS | 1.0058 LOSS | 1.0273 LOSS | 0.9961-1.0021 |

The 1:1 port already wins at r1 (delta, qsa) and qsa r4s and is within the controls' reach elsewhere except qsa r2s
(1.011). Where it wins, the stamps say why (section 8): the A2 / M2 side (the skipped barrier, the A1-side waits)
finishes earlier than FlyDSL's.

### 6.2 Climbs (mono tuner, stack base; budgets 96-512; the confirmation of the finalists in the same job, 16-30 rounds)

Per cell, the climbs that ran (job, node, budget, the pick's median in its own confirmation): delta r1 (tuneA2 n07, 96:
0.9876), qsa r1 (tuneA2 n07, 96: 0.9916), delta_ple r1 (tuneC2 n07, 256: 0.9919; tuneZ n01, 384: 0.9952), delta r2s
(tuneB n01, 96: 0.9979; tuneD2 n01, 384: 0.9941), qsa r2s (tuneA2 n07, 96: 0.9927; tuneD2 n01, 256: 0.9875), delta_ple
r2s (tuneC2 n07, 256: 0.9724), delta r3s (tuneB n01, 96: 0.9977; tuneD2 n01, 384: 0.9965; xconfX n07, 512: 0.9892),
qsa r3s (tuneC2 n07, 256: 0.9822), delta_ple r3s (tuneC2 n07, 256: 0.9941), delta r4s (tuneA2 n07, 96: 1.0001; tuneD2
n01, 384: 0.9960), qsa r4s (tuneB n01, 96: 0.9840), delta_ple r4s (tuneB n01, 96: 0.9994; tuneD2 n01, 384: 0.9987;
xconfX n07, 512: 0.9965). Larger budgets found the delta r2s / r3s and delta_ple r2s / r3s wins; the 96-evaluation climbs
had stopped short there. Then, from the confirmed picks of the five cells that were not yet a WIN in all four final runs
(`--start`, decision 15; 384 evaluations each): delta r2s (G1 n01: 0.9948), delta r4s (G1 n01: 0.9816), delta_ple r1
(G1 n01: 0.9945), qsa r2s (G2 n07: 0.9803, in a job whose recompiled control ran 0.987: the four-run confirmation below
is the number), delta_ple r4s (G2 n07: 0.9928).

### 6.3 The picks on both nodes (confirmation-only jobs; each cell twice per node in fresh processes; 30 rounds)

Ratios vs the tuned stack's FlyDSL-HC layer. Jobs `w8-wp2a-F1` (n01-09) / `F2` (n07-17) confirmed the climbs' two best
picks per cell; for the five cells that were not a WIN in all four runs there, a second climb started from the F pick
(section 6.2, `G1` / `G2`), and its pick was confirmed the same way (`F3a/b`, `F4a/b`). DESIGN 9.1: WIN = median below
the controls' minimum with at least 3/4 of the rounds below 1.

| cell | port n01-09 | port n07-17 | pick n01-09 | pick n07-17 | pick verdicts (4 runs) | controls (min-max) | confirm jobs |
|---|---|---|---|---|---|---|---|
| delta r1 | 0.9880/0.9874 | 0.9883/0.9875 | 0.9879/0.9866 | 0.9881/0.9874 | 4 WIN | 0.9957-1.0033 | F1 / F2 |
| qsa r1 | 0.9940/0.9965 | 0.9928/0.9921 | 0.9914/0.9948 | 0.9880/0.9966 | 4 WIN | 0.9977-1.0099 | F1 / F2 |
| delta_ple r1 | 0.9973/1.0013 | 1.0013/1.0009 | 0.9917/1.0018 | 0.9938/0.9946 | 3 WIN, 1 NEUTRAL | 0.9961-1.0026 | F3a / F3b |
| delta r2s | 1.0032/1.0031 | 1.0026/1.0024 | 0.9943/0.9964 | 0.9945/0.9948 | 2 WIN, 2 NEUTRAL | 0.9950-1.0031 | F3a / F3b |
| qsa r2s | 0.9949/0.9985 | 0.9922/0.9917 | 0.9787/1.0008 | 0.9909/0.9734 | 3 WIN, 1 NEUTRAL | 0.9956-1.0029 | F4a / F4b |
| delta_ple r2s | 1.0024/1.0026 | 1.0021/1.0040 | 0.9775/0.9757 | 0.9726/0.9766 | 4 WIN | 0.9943-1.0048 | F1 / F2 |
| delta r3s | 1.0022/1.0032 | 1.0036/1.0025 | 0.9916/0.9933 | 0.9904/0.9901 | 4 WIN | 0.9973-1.0025 | F1 / F2 |
| qsa r3s | 0.9986/0.9981 | 0.9983/0.9964 | 0.9840/0.9855 | 0.9818/0.9799 | 4 WIN | 0.9928-1.0063 | F1 / F2 |
| delta_ple r3s | 1.0034/1.0040 | 1.0037/1.0057 | 0.9899/0.9899 | 0.9925/0.9937 | 4 WIN | 0.9960-1.0034 | F1 / F2 |
| delta r4s | 1.0042/1.0017 | 1.0026/1.0041 | 0.9870/0.9872 | 0.9903/0.9855 | 4 WIN | 0.9976-1.0036 | F3a / F3b |
| qsa r4s | 0.9883/0.9882 | 0.9861/0.9870 | 0.9867/0.9848 | 0.9874/0.9850 | 4 WIN | 0.9981-1.0040 | F1 / F2 |
| delta_ple r4s | 1.0018/1.0023 | 0.9997/1.0006 | 0.9919/0.9922 | 0.9903/0.9918 | 4 WIN | 0.9949-1.0030 | F4a / F4b |

The picks (per stage, `HcKnobs` + HC schedule levers; full specs with the resolved per-stage `HcKnobs` strings and the ATOM schedule: `hackathon-prep/qwen38/truemono/WP2a/picks_final.json`):

| cell | pick (`stage[knobs]`, `sched[...]`) |
|---|---|
| delta r1 | `m1[amax=fmax]` |
| qsa r1 | `a1[sgb=1] a2[w_cpol=19]` |
| delta_ple r1 | `a1[amax=fmax,anchor=0,pause_sleep=2,ra=2,scale=hoist,shfl4=select,w_cpol=2,wi_cpol=18] a2[anchor=0,scale=opsel,w_cpol=19] m1[amax=fmax,mailbox.impl=irf,mailbox.pause=none,sgb=1,shfl4=select,w_cpol=17] m2[mailbox.pause=s_sleep1] sched[head_sleep=192,hold_m2=96,prio_m1=1,prio_m2=2,spread_a1=3]` |
| delta r2s | `a1[amax=fmax,hbar=0,mailbox.pause=none,w_cpol=16] a2[mailbox.impl=asm,nw_cpol=2,pause_sleep=8,scale=opsel] m2[shfl4=select,w_cpol=1]` |
| qsa r2s | `a1[anchor=0,mailbox.impl=asm,mailbox.pause=s_sleep1,nw_cpol=18,ra=2]` |
| delta_ple r2s | `a1[amax=fmax,pause_sleep=2,scale=opsel,sgb=1,wi_cpol=18] a2[mailbox.impl=asm,mailbox.pause=s_sleep1,pause_sleep=8,ra=2] m1[wi_cpol=16] m2[anchor=0,inv=regs,scale=hoist,sgb=1,shfl4=select] sched[spread_a1=1]` |
| delta r3s | `a1[anchor=0,hbar=0,mailbox.impl=asm,nw_cpol=16] a2[a2order=h_first,amax=fmax,inv=regs,mailbox.pause=none,ra=2,scale=hoist,shfl4=select] m1[amax=fmax,anchor=0,mailbox.impl=irf,nw_cpol=16,pause_sleep=4,ra=1,scale=hoist,sgb=1,w_cpol=16] m2[mailbox.impl=irf,nw_cpol=18,ra=2,shfl4=select] sched[hold_a2=96,hold_m2=0,prio_m1=3,spread_m1=2]` |
| qsa r3s | `a1[amax=fmax,anchor=0,mailbox.pause=s_sleep1,ra=1,sgb=1] a2[mailbox.pause=s_sleep1,shfl4=select] m1[mailbox.pause=s_sleep1,pause_sleep=8,scale=hoist,sgb=1,wi_cpol=16] m2[inv=regs,nw_cpol=2,sgb=1,shfl4=select] sched[hold_m2=16]` |
| delta_ple r3s | `a1[amax=fmax,anchor=0,inv=regs,mailbox.impl=irf,sgb=1,shfl4=select] a2[inv=regs,nw_cpol=16,pause_sleep=8,shfl4=select] m1[amax=fmax,mailbox.pause=none,sgb=1,stats=halves] m2[pause_sleep=4,scale=opsel,shfl4=select,w_cpol=19] sched[head_sleep=0,hold_a2=16,hold_m1=192,prio_m1=3,spread_a1=2]` |
| delta r4s | `a1[hbar=0,mailbox.impl=asm,pause_sleep=1,ra=2,shfl4=select] a2[a2order=h_first,anchor=0,scale=hoist,w_cpol=2] m1[amax=fmax,anchor=0] m2[anchor=0,scale=hoist] sched[head_sleep=64,hold_a2=16,hold_m2=64,prio_a1=3,prio_a2=2,spread_a1=2]` |
| qsa r4s | `a1[nw_cpol=2,ra=1,scale=opsel]` |
| delta_ple r4s | `a1[nw_cpol=2,ra=1,scale=opsel,w_cpol=3] a2[inv=regs,mailbox.pause=s_sleep1,nw_cpol=16,pause_sleep=4,ra=1,scale=hoist] m1[amax=fmax,mailbox.pause=none,pause_sleep=1,scale=hoist,w_cpol=17] m2[anchor=0,inv=regs,nw_cpol=16,shfl4=select,w_cpol=3] sched[hold_a2=32,hold_m2=64,prio_a1=1,prio_a2=1,spread_a1=2]` |

### 6.4 Tolerance class: two accumulators (`acc=2`, the N1 hook; job `w8-wp2a-t2`, n01-09, 30 rounds)

| cell | port | `acc=2` | `acc=2,sgb=1,ra=1` | `acc=2,sgb=1,ra=1,scale=hoist` |
|---|---|---|---|---|
| delta r1 | 0.9882 | 0.9894 | 0.9895 | 0.9882 |
| qsa r1 | 0.9976 | 1.0049 | 0.9982 | 1.0000 |
| delta_ple r1 | 0.9997 | 1.0010 | 1.0032 | 1.0074 LOSS |
| qsa r3s | 0.9962 | 0.9975 | **0.9861 WIN** | 0.9929 |
| delta r4s | 1.0036 | 1.0033 | 1.0059 LOSS | 1.0071 LOSS |
| qsa r4s | 0.9883 | 0.9879 | 0.9902 | 0.9882 |

Splitting the chain does not pay on its own: the chain's 18 dependent MFMAs are not what bounds A1 (A1 is VALU-bound,
section 3; the MFMAs are 20 of about 2150 instructions). The one gain (qsa r3s) is the `sgb` interleave, which the bitwise
picks already use there. `acc=2` stays an off-by-default tolerance knob for the N1 ladder (in these runs it happened to be
bitwise in 4 of 6 cells; in delta_ple r1 1 byte of the internal stream copy differs).

## 7. Rust direct backend

The HC runs as framework layer kernels (`stages2.hc.framework_factory`, Path B `layers=1` form: the `attn` run A1+A2,
the `mlp` run M1+M2) are bitwise against ATOM's own FlyDSL HC stages at every kind and width tried (`w8-wp2a-rt1`:
delta / qsa `attn` r1, r4s, r2s; delta / delta_ple `mlp` r1, r4s), and the Rust direct code-object compiler
(`wpr3-bin-v1`) compiles them bitwise too (every arm passes numerics against COMGR and against FlyDSL's stages). It is
slower than COMGR on these kernels. Ratios vs the COMGR build of the same kernel (24 rounds; controls C2 = second
graph, R = fresh-process COMGR compile, both within 0.987-1.004):

| kernel | D (direct defaults) | best single lever | B3 set (WP-R3) | best combination |
|---|---|---|---|---|
| delta attn r1 | 1.106-1.113 LOSS | `wgfence` 1.045 LOSS | 1.024-1.026 LOSS | B3 + `loadfirst` 1.012 (neutral, 1 of 24 rounds below 1) |
| qsa attn r4s | 1.106-1.109 LOSS | `wgfence` 1.041 LOSS | 1.021-1.022 LOSS | B3 + `loadfirst` 1.019 LOSS |
| delta mlp r1 | 1.031-1.042 LOSS | `wgfence` 1.019 neutral | 0.998-1.006 neutral | B3 + `loadfirst` 0.995 neutral (17 of 24) |
| delta mlp r4s | 1.056-1.061 LOSS | `wgfence` 1.037 LOSS | 1.011-1.016 | B3 1.011 neutral |
| delta_ple mlp r1 | 1.038-1.040 LOSS | - | 0.997 neutral | `wgfence+loadfirst+loadorder` 0.993 neutral (21 of 24) |

All 24 levers were screened one at a time (`w8-wp2a-rt2`): 14 compile to the direct defaults' bytes on these kernels
(`storewait dppcombine ldsquads dppcombinef minmax xorcmp dsrebase bitop3 selxor subadd bufoffany acqfence wpecap`, and
`ldspairs` at r1), the other 10 each lose; combinations (`rt3`: the B3 set, the MoE M1 set, each with `loadfirst`, the
fence / load-order trio, all levers at once (same bytes as B3 + `loadfirst`)) reach neutral on the `mlp` runs and stay a
loss on `attn`. **COMGR stays the HC backend**; the direct path's gap is in its own codegen of these stages (D alone
loses 4-11 %), not in a lever. Kept as a measured option for WP3's `layers=1` kernels, re-tested there (decision 15).

## 8. WaveScope and stage timelines

**WaveScope (ATT).** Captures of the stamped layer (`w8-wp2a-att1`: stack schedule, CUs 0 and 5; `att2b`: plain schedule,
6 CUs, the port and the findings set `ra=1,sgb=1,scale=opsel` against the FlyDSL-HC base and against the port), summed
over the captured CUs:

| capture | layer latency, arm / base |
|---|---|
| delta r1 stack, port / FlyDSL | 0.976 |
| delta r4s stack, port / FlyDSL | 0.979 |
| delta r1 plain, port / FlyDSL | 0.990 |
| delta r1 plain, findings set / port | 0.999 |
| qsa r4s plain, port / FlyDSL | 1.008 |
| qsa r4s plain, findings set / port | 0.991 |

Per-stage ATT attribution does not work for this kernel: a captured CU runs whatever CTA role the schedule gives it
(spread, NUMA placement), so stages without an HC change swing 0.6-1.4x between two builds of the same schedule, and in
the plain schedule the captured CUs hold almost no A1 / M1 tasks. Two harness fixes came out of it: stage brackets that
share a stamp (empty m4 / m5 bodies on the captured CU) are paired without reuse (`WP2a/att_hc_cmp.py`, the stock
`att_compare.py` aborts), and the mailbox-share column reads ~0 for the rocKE functions only because they carry no
FlyDSL debug line info (not a saving). Both are alola rule 213.

**Stage timelines (stamps; `w8-wp2a-stamps1`, `WP2a/stamps_hc.py`).** The stage stamps time every task of every stage
in situ (64 launches per arm, interleaved; stamped builds are not the timed builds, so these are diagnostics). Plain
schedule, the port over the FlyDSL HC (all arms bitwise in the same job):

| cell | layer total | A1 critical-path increment (share of layer) | A2 increment | A2 end | M1 increment (share) | M2 end |
|---|---|---|---|---|---|---|
| delta r1 | 0.996 | 1.043 (0.18) | 0.94 | 1.018 | 1.084 (0.13) | 0.985 |
| qsa r1 | 0.997 | 1.010 (0.14) | 1.00 | 1.006 | 1.085 (0.08) | 0.995 |
| delta r4s | 0.985 | 1.045 (0.11) | 0.57 | 0.937 | 0.999 (0.09) | 0.976 |
| qsa r4s | 0.998 | 1.032 (0.09) | 0.57 | 0.952 | 1.002 (0.07) | 0.997 |
| delta_ple r1 | 0.997 | 1.064 (0.11) | 0.67 | 0.997 | 1.056 (0.11) | 0.994 |

Read: the A1 / M1 tasks end 1-8 % later than FlyDSL's while A2 / M2 finish earlier, so the HC as a whole ends at
0.94-1.02 of FlyDSL's and the layer at 0.985-0.998. The A1 / A2 split moved (the skipped redundant A2 barrier, the
A1-side waits), so per-stage increments are not the port's cost; the HC end and the layer total are. The A1 slowest
task's work (`tail_max`) is 1.02-1.07 of FlyDSL's in the delta / qsa cells (A1 reads memory there): the remaining A1 gap the climbs work on (`hbar`,
`anchor=0`, `ra`, `sgb`, the cache bits). `anchor=0` ends A1 later still (1.02-1.10 of FlyDSL's), as its ISA says. The findings set
does not shorten A1's end in the plain schedule (1.005-1.061); the climbs pick it where the stack's schedule lets the
shorter chain matter (qsa r4s / r3s, qsa r2s on n01-09). (The per-task `tail_med` of polled stages is not comparable:
the rocKE polls do not write the per-wave ready stamps FlyDSL's do.)

On the tuned stack's schedule with the picks (`w8-wp2a-stamps2`, 64 launches per arm; all arms bitwise), the stage ends
show where the picks win (ratios vs the FlyDSL-HC stack layer):

| cell | arm | layer total | A1 end | A2 end | M1 end | M2 end |
|---|---|---|---|---|---|---|
| delta_ple r2s | port | 0.993 | 1.012 | 1.024 | 1.013 | 0.992 |
| delta_ple r2s | pick | 0.967 | 1.012 | 0.941 | 0.970 | 0.952 |
| qsa r3s | port | 0.985 | 1.069 | 0.947 | 0.981 | 0.978 |
| qsa r3s | pick | 0.977 | 1.069 | 0.917 | 0.979 | 0.972 |
| qsa r4s | port | 0.986 | 1.038 | 0.979 | 0.992 | 0.979 |
| qsa r4s | pick | 0.988 | 1.041 | 0.979 | 0.991 | 0.982 |
| delta r1 | port | 0.984 | 1.060 | 1.079 | 1.014 | 0.980 |
| delta r1 | pick | 0.977 | 1.025 | 1.076 | 1.012 | 0.981 |
| delta r3s | port | 1.015 | 1.044 | 0.997 | 1.048 | 1.015 |
| delta r3s | pick | 1.013 | 1.068 | 1.109 | 1.029 | 1.014 |

The delta_ple r2s pick (`spread.a1=1` with the packed-scale A1 and the asm-poll A2) moves the attention HC's end 6 %
earlier and the MLP HC's 5 %; qsa r3s gains in A2. delta r3s is the counter-example: its stamped build is slower than
FlyDSL's in every stage while the unstamped pick wins 0.990-0.993 on both nodes: the stamps' stores and clock reads change
that build's schedule (the WP2b note on stamped builds applies), so the unstamped rounds decide.

## 9. Lookback, losers, and "rejected on base X"

All verdicts are on the base "tuned stack (`picksI`; qsa with `qdense=3`) layer with its rocKE MoE / B3 / GEMV functions
and FlyDSL's HC stages", the stack's own HC schedule levers (`spread.a1=10` at r1, D0.3) as the climb's start.

- **Pools keep the losers** (PLAN 13): every climb's pool is the full space (67 HC knobs + the HC schedule levers, 168-235
  changed levels per cell; `pruned: 0`). 7008 configs were measured in 26 climbs (budgets 96-512), every one gated
  bitwise first, with 0 gate failures and 0 hangs. (Before them, a numerics job hung on `wait=prepoll` A2 under spread, which polled a hint pair
  the spread layout never writes, now fixed; and a first climb, `tuneA`, hung in a gate check that the in-process
  watchdog could not interrupt: it was cancelled and rerun as `tuneA2` under the external hang watcher, alola rule 213.)
- **Decision 15 paid twice**: re-climbing from a confirmed pick (`hc_tune.py --start`: every lever re-tested on the pick
  as the new base) turned the four cells that were neutral from the port into wins: delta r4s 0.996 -> 0.986-0.990 (adds
  `spread.a1=2` and the asm poll on A1), qsa r2s -> 0.973-0.991 (A1 `ra=2`, asm poll with `s_sleep 1`, `nw_cpol=18` on
  top of `anchor=0`), delta_ple r4s 1.000 -> 0.990-0.992, delta_ple r1 -> 0.992-0.995 on n07-17 (n01-09: one of two runs
  neutral).
- **Solo losers that are in winning picks**: `anchor=0` (more instructions; loses alone in most cells) is in 8 of the 12
  picks (delta_ple r1-r4s, qsa r2s / r3s, delta r3s / r4s); `inv=regs` (+276 instructions, -2 barriers; 1.031 on every
  stage at delta_ple r1) is in 5 picks on single stages (delta_ple r2s-r4s, delta r3s, qsa r3s); the cache bits
  (`w_cpol` / `nw_cpol` / `wi_cpol`) are the most frequent solo losers (59-79 losing levels per stage across the cells)
  and appear in 11 of the 12 picks; `hbar=0` (no barrier between the norm and the weight loads) is in every delta
  r2s-r4s pick.
- **The findings set** (`ra=1,sgb=1,scale=opsel` on all four stages, "cat" in t1) against the port, same job (t1,
  30 rounds): 0.991 at qsa r3s, 0.995 at qsa r2s, 1.000-1.001 at qsa r1 / r4s, 1.001-1.008 at delta and delta_ple. Its
  pieces (`ra`, `sgb`, `scale=opsel` / `hoist`) are in 11 of the 12 picks, on the stages where they pay: where A1's
  chain is on the critical path (qsa) the shorter chain pays; at delta the extra VALU of the packed scales costs more
  than the chain saves.
- **Rejected on this base**: `w_cpol=2` (nt) on every stage (r4s: 1.027-1.041 LOSS; r1: 0.988-0.995), `pre=none`
  (0.5-1 % slower than `pre=args` except qsa r1), `stats=halves` on M1 (neutral to 1.002), `acc=2` (section 6.4), the
  Rust direct backend (section 7). Each is a knob value in the pools, to be re-tested on WP3's base (decision 15).
- **Spread** (every value per HC searched, the dedupe 3 / 4 and the M1 placements included). The picks' schedules: r1
  delta / qsa and qsa r2s keep `spread.a1=10` (D0.3); delta_ple r1 takes the dedupe `spread.a1=3`; delta r2s / r3s and
  delta_ple r3s keep `spread.a1=2`, delta r3s adds `spread.m1=2`; delta_ple r2s takes `spread.a1=1`; delta r4s and
  delta_ple r4s take `spread.a1=2` (D0.3 had "delta r4s: none": with the rocKE HC it pays); qsa r3s keeps the stack's
  `spread.a1=2,spread.m1=2`; qsa r4s runs without spread (D0.3's (ii) there was not supported in combination, D0 m2,
  and the climbs agree). D0.3's values were the start; the lookback's picks are the decision.

## 10. Open items

- **Grouped / wide widths** (`r4g2 .. r16g4`): not in WP2a's scope (the K=0..3 CONC 1 path serves `r1-r4s`). Rows > 4
  run ATOM's separate `hc_wide` and are refused by `supported()`; grouped narrow builds (`r4g2`) were not built or
  tested. ATOM builds the spread only for narrow ungrouped layers.
- **delta r2s** is the one cell whose pick is not a WIN in at least 3 of 4 runs (0.994-0.996, all below 1, 2 inside the
  controls); a third climb from the G1 pick is the next step there, and in WP3's joint lookback (HC knobs with the
  layer's MoE / B3 / GEMV / schedule knobs in one pool).
- **Quick end-to-end** (PLAN 14) with the picks on the full stack: left to WP3, which builds the `layers=1` kernels from
  these functions (`framework_factory`), re-climbs the HC knobs jointly (decision 15) and is the end-to-end measurement.
- **N1**: `acc=2` is built and validated (deterministic, tolerance class) and timed (section 6.4: no gain alone); the N1
  ladder (WP15) decides it.
- **WaveScope per stage**: needs captures that pin the CU running a given HC task (or stamped builds that do not change
  the schedule); the stamp timelines are the per-stage evidence here.
- Picks for WP3: `hackathon-prep/qwen38/truemono/WP2a/picks_final.json` (per cell: canonical knobs, resolved per-stage
  `HcKnobs` strings, the ATOM schedule, the four confirmation medians).

## Verification (independent verifier, W8, 2026-11-01)

Verdict: **PARTIAL.** The parity, bitwise-numerics, byte-identity and layer-timing claims hold, and a fresh-seed
re-run on both nodes reproduces them. Two gates PLAN asks of every port were not run (the quick end-to-end number,
decision 14, and the `.ll` half of decision 16), and some summary figures overstate the evidence. `gate_met` should
stay open until the end-to-end number exists.

**What the verifier re-ran.** Fresh stages of the current trees were used (`w8-vWP2a-h-a`, `rocke-w8-vWP2a-a`,
`atom-truemono-w8-vWP2a-a`; `hc.py` sha `f6f352861ca8d8ff`, the ATOM hook hashes `8c4cdf3caba10612` /
`32be269f05a73434`, all the same as in the builder's jobs). Two jobs ran on one GPU each: `w8-vWP2a-c1` on n01-09
with seed 7 and `w8-vWP2a-c2` on n07-17 with seed 11. Cell order was reversed between the two jobs. Each cell ran
`hc_tune.py --budget 0 --extra` with only the final pick from `picks_final.json`, 30 rounds, on the tuned-stack
base. Scripts: `hackathon-prep/qwen38/truemono/WP2a-verify/` (`vsync.sh`, `job_v.sh`, `vpicks/`). Logs:
`hackathon-prep/logs/q38/w8-vWP2a-c1.submit.log` and `w8-vWP2a-c2.submit.log`.

- **CPU tests on the final code.** The ident1 job's CPU tests ran at 12:19, before `hc.py` and the test file
  reached their final versions (12:57). The re-run gives: rocKE `test_qwen38_stages2_hc.py` **79 passed**, with
  llvm-as present, so the 8 IR-validity tests ran (locally they skip with "no llvm-as"). ATOM-truemono's
  `test_qwen4_exp_mono_extern_stages.py` + `extern_moe_pairing.py` **23 passed**.
- **Bitwise gate.** Every one of the 24 runs passed the gate self-test: the port was bitwise and the `down_part`
  mutant was caught. Every pick was gated bitwise before and after timing, both jobs ended rc 0, and there were 0
  gate failures.
- **Timing (DESIGN 9.1 on the tuned stack's FlyDSL-HC layer; pick median, verdict):**

| cell | n01-09 (seed 7) | n07-17 (seed 11) | builder's 4 runs |
|---|---|---|---|
| delta r1 | 0.9878 WIN | 0.9878 WIN | 4 WIN |
| qsa r1 | **0.9986 NEUTRAL** (29/30 below 1; the port arm 0.996 WIN in the same run) | 0.9893 WIN | 4 WIN |
| delta_ple r1 | 0.9945 WIN | 0.9959 NEUTRAL | 3 WIN 1 NEUTRAL |
| delta r2s | 0.9964 NEUTRAL | 0.9972 NEUTRAL | 2 WIN 2 NEUTRAL |
| qsa r2s | 0.9776 WIN | 0.9938 WIN | 3 WIN 1 NEUTRAL |
| delta_ple r2s | 0.9784 WIN | 0.9726 WIN | 4 WIN |
| delta r3s | 0.9912 WIN | 0.9896 WIN | 4 WIN |
| qsa r3s | 0.9820 WIN | 0.9818 WIN | 4 WIN |
| delta_ple r3s | 0.9914 WIN | 0.9915 WIN | 4 WIN |
| delta r4s | 0.9878 WIN | 0.9878 WIN | 4 WIN |
| qsa r4s | 0.9844 WIN | 0.9869 WIN | 4 WIN |
| delta_ple r4s | 0.9891 WIN | 0.9943 WIN | 4 WIN |

  All 24 medians are below 1 and none is a LOSS: "no cell loses" is confirmed. 20 of 24 runs are WIN. Of the 9
  cells claimed WIN in all 4 runs, 8 are WIN on both nodes again. **qsa r1 is not robust:** on n01-09 its pick is
  NEUTRAL and does not beat the port arm. delta r2s stays NEUTRAL on both nodes, as the builder reported.

**Checked from the artifacts:**

- **Parity.** The ISA side is real. `rv2/parity_rv2.md` gives per-stage class counts for 60 code objects, and
  `rv2/hc_delta_r1.md` gives the per-stage diffs and the catalogue findings. Bitwise numerics on MI355X are real:
  `num6a` had 173 CHECK lines and 178 MUTANT lines, all with `differs: true`. Its rc 1 comes only from the designed
  spread-mutant bases 6 and 9, where the arm equals the base. The "24 arms" counted there include the job's retries
  and the `_base` lines (6 steps x 2 x 2). `num6b` had 408 checks; only `acc2` differs, in the internal stream copy.
- **Byte identity.** `ident1` ran 13 builds x {plain, best, ext}. Every `.text` and resource set is equal
  (key-only PASS), and the HC mutant fails the gate as it must. On the rocKE side, nothing outside `stages2/`
  imports it (only the two test files reference it), so identity holds by construction.
- **Base and selection.** The base is picksI + `qdense=3` with the hybrid's rocKE MoE / B3 externs. Each final
  pick is one fixed config across its 4 confirmation runs. For the G cells, `g` was chosen over `start` on the same
  F3/F4 data, a mild selection effect; the independent re-run above removes it.

**Issues:**

1. **PLAN decision 14 not exercised.** No quick end-to-end number was measured in the same job as the hybrid,
   though the picks link through `ATOM_Q38_EXTERN_STAGES` like the stack's MoE / B3 externs. It was deferred to
   WP3.
2. **Decision 16 is half done.** The diff is ISA only (`rv_isa.py split` / `rv_diff.py`, and `rv1/opdiff.py`
   compares ISA opcode histograms). The `.ll` files were dumped (`dumps_ll.tgz`), but no `.ll` diff against
   FlyDSL's final IR was produced or reported.
3. **Overstated A1 VALU parity.** The summary's "A1 VALU 1.003-1.004 of FlyDSL" is narrower than the table: A1
   VALU runs 0.972-1.008 over the 15 hc rows (e.g. qsa r4 1.008, qsa r3s 1.007, delta r3s 1.006).
4. **Unexplained instruction-count gap.** At delta A1 r2s-r4 the port has about 250 fewer instructions than FlyDSL
   (n 0.89-0.90), one more `vmem_store` (3 vs 2) and fewer VGPRs. The table flags these as outside 10 % but the
   text does not explain them. They are bitwise, so this is a documentation gap, not a defect.
5. **Wrong test attribution.** The summary says "ATOM's 104 CPU tests pass". Those 104 were rocKE tests (three
   files, 18 skipped) on the pre-final code. ATOM's own extern tests were never run in WP2a. The verifier ran them:
   23 pass.
6. **Scope narrower than DESIGN.** DESIGN's WP2a row says "narrow + wide ... `xpre`". Grouped / wide widths
   (r4g2, r8g4, r16g4, which the consolidated stack serves at CONC 2-4) and `xpre` were not built. D0's
   plan-of-record row is narrow-only, so this is a recorded scope cut, not a hidden failure.
7. **qsa r1** is a WIN in 5 of 6 runs over both jobs, not "WIN in all runs". Its pick (`a1[sgb=1] a2[w_cpol=19]`)
   did not beat the bare port on n01-09.
