# WP-L2d: the large-batch MoE rebuild (`moe_lb2`) against FlyTape's MoE

Ratios only. Absolute times are in the private copy
(`hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L4b.md`).

## 0. Summary

- **Kernel.** `large_batch/moe_lb2.py` (names `q38lb2_moe*`), sha256 `2e676242...` (CRLF stripped) at the final runs.
  P-L4's `moe_lb.py` is not edited; `moe_lb2` imports five helpers from it.
- **Harness.** `hackathon-prep/qwen38/truemono-lb/p-l4b/` (`pl4b_bench.py`, `pl4b_report.py`, `isa350.sh`,
  `isa_metrics.py`, patch scripts `patch_v2*.py`).
- **Final configuration** (the `lb2` base of jobs `lb2-f3` / `lb2-f4`):
  `deal=rr, dt=4, doff=imm, pstg=4, pc=0, xb=1, depth=6`.
  - Whole experts dealt round robin, XCD blocks per round.
  - FlyTape-form down: 4 output tiles share an h fragment, staged 256 B partial-row stores.
  - The last round's experts split in up to 4 pieces.
- **Against FlyTape's experts + combine, same job, MI355X, both nodes** (best `moe2` arm per cell; fp8 and fp4
  regimes; tables in section 3).

  | rows | TP1 | TP2 |
  |---|---|---|
  | r8 | 1.12-1.19 L | 1.13-1.15 L |
  | r16 | 0.97-1.02 (tie) | 0.95-0.96 W |
  | r32 | 0.92-0.97 W | 1.03-1.07 L |
  | r64 | 0.93-0.98 W | 0.90-1.01 |
  | r128 | no FlyTape TP1 r128 trace | 0.93-1.00 |

  - Against P-L4's best arm (`sop=1, wcp=default`), the rebuild is 0.83-0.94 at r16 and above. P-L4 stays ahead at r8
    (P-L4 / FlyTape 1.00-1.03).
  - Against the WP-L2d starting point (the atom line), the rebuild is 0.75-0.90.
  - Against FlyTape's whole MoE (router + experts + combine), the rebuild is 0.76-0.94 in every cell. Our kernel covers
    the top-k from logits, the sort, the experts and the combine, but not the router GEMM.
- **Gate.** DESIGN-LB section 15 asks for 1.00 or less of FlyTape's experts + combine in every cell on both nodes. That
  is not met at:
  - r8 (both TP);
  - TP1 fp4 r16;
  - TP2 r32;
  - TP2 fp4 r64 on n07-17.

  The 0.85 target is met in no cell.
- **Numerics.** Every build passed on MI355X: T0 against the float64 reference, the deterministic slot-order combine,
  bitwise knob invariance, the 2-layer walk, the dirty-h and stale-read twins, and the must-fail mutants (`slot`,
  `sort`, `sopsel`, `seg`, `half`, `hsc`, `desc`, `mask`, `tick`). There is one exception: the stale-read mutant
  `noacq` (acquire dropped) was not caught at the r8 cells within its 5 launches. It was caught at r16 and above. The
  race window is too short at 8 rows (rule 182 already flags this twin as unreliable). On n01-09 it was also missed
  once at TP2 fp8 r16.

## 1. What the rebuild is

The rebuild is one persistent kernel per layer walk (`gs`) or per layer (`launch`; the two metrics agree within 2 %
here). Per layer:

1. **Sort.** The route pairs give per-expert bitmaps, counts, items (expert, pass of 16 tokens) and token lists. Every
   CTA rebuilds them in LDS.
2. **Deal (`deal=rr`).** Item k goes to CTA k mod 256, as in FlyTape's `moe_expert`.
   - `xb`: per round, XCD x takes a block of consecutive units. One expert's passes (the shared expert's) then share an
     L2, and every XCD stays busy in a partial round. `xh` uses blocks of 32 and leaves XCDs idle when a round is
     partial.
   - `pc=0`: the full rounds stay whole; the last round's items are split into up to 4 atom pieces (partial sums in
     slot segments, `desc` = pieces).
3. **Stream (`pipe=stream`).** One CTA streams its units. Gate_up runs in atom loops, K split over the waves' two
   halves. An LDS reduction plus the silu-requant epilogue feed h in LDS. The down head and the next unit's gate_up head
   are issued as a lookahead across units (`lka=tail`).
4. **Down.** `dt=4` output tiles per h fragment, each with its own accumulator. `doff=imm`: one SGPR tile base, the
   chunk as an immediate. `pstg=4`: partials go through a per-wave LDS stage and are stored as 256 B rows. Partials are
   f32, per (token, slot, segment).
5. **Combine.** Grid barrier, then y = sum over slots (order) of w_s x sum over segments (order) of the partials.

**Alternative builds kept as knobs:**

- `gu=wt`: FlyTape's fused gate_up. A wave owns whole gate / up row-tile pairs over the full K (TP2: K halves, `wk=2`).
  silu(g) u goes to an f32 panel, all threads requantize after one barrier. Options: `wg` sweeps, `word`, `wpin`.
- `deal=dyn`: whole units from a global ticket. `uord=heavy`: a stable counting sort by token count.
- `deal=line`: the WP-L2d starting point.
- `shs`: shared-expert items of 4 / 8 tokens.
- `lpf`: x B-operand prefetch distance.
- `pst`: partial-store cache bits.

## 2. Lever ledger (every verdict with its base)

Ratios are the arm's time / the base's time in the same job. **W** is under the base by more than the controls'
spread, **L** is over it, **T** is a tie (the win rule: shuffled interleaved arms, `lb2b` A-vs-A and `lb2rc`
recompiled controls in every job).

Rows marked MI350X are Markham triage. Verdicts come from the MI355X jobs `lb2-f1` (n07-17), `lb2-f2` (n01-09),
`lb2-f3` (n07-17) and `lb2-f4` (n01-09).

Bases:
- **B0**: the atom line (`deal=line`, the WP-L2d start).
- **B1**: B0 + `deal=rr`.
- **B2**: B1 + `dt=4, doff=imm, pstg=4`.
- **B3**: B2 + `pc=0, xb=1, depth=6` (final).

| lever | base | result | ATT / stamps reading |
|---|---|---|---|
| `pipe=unit` (v1: unit loops) vs `stream` | B0 | v1 1.01-1.08 L | v2 removes the per-unit drain |
| `lka=early` | B0 | TP1 r64 1.05 L, else T | |
| `depth` 2 / 6 / 8 / 12, `ddepth` 4 / 12, `ab` 1 / 4, `prio`, `xv`, `xsync=cnt`, `tlbp` | B0 | T (`ov` 1 / 2: TP2 r32 0.975 W only) | WAIT 74-80 %: barriers and MFMAs waiting for loads |
| `xh` | B0 | TP2 r128 0.96-0.975 W | |
| `wl` (weighted line), `snap` | B0 | TP2 r128 0.955-0.966 W (snap 0.889), snap TP2 r16 / r32 1.29 / 1.13 L | line-only knobs: not applicable from B1 on |
| **`deal=rr`** (whole items, FlyTape's schedule) | B0 | **TP2 0.85-0.89 W**, TP1 r16 0.96 W, TP1 r64 T | per-segment overhead and partial chunks gone |
| `deal=hy` (whole rounds + atom-line rest) | B0 | T | |
| `deal=dyn` (tickets) | B1, B2 | T | equal-size units: nothing to rebalance |
| `uord=heavy` | B1 | 1.02-1.04 L | |
| `pc=2` (every item) | B1 | TP1 r16 1.17 L, TP2 1.02-1.10 L (uniform) | |
| `gu=wt` (wave-owned pairs, full K) | B1 | TP1 0.89-0.96 W, TP2 1.00-1.04 T/L | first TP2 build 2.3-3.6 L: cluster-compiler spills (section 5) |
| `wt word=grp`, `wpin` | B1 | T | |
| `dt=4` + `doff=imm` | B1 | T | serial MFMA chains 69 -> 9 per kernel, `s_nop` before stores 661 -> 137 cycles |
| **+ `pstg=4`** (staged 256 B row stores) | B1 | **TP2 0.95-0.977 W**, TP1 0.98 T | |
| `line` on B2 | B2 | 1.07-1.37 L (MI355X: 1.03-1.48) | |
| `xh` | B2 | TP2 r32 / r128 0.965-0.975 W, r16 1.02-1.10 L; TP1 r128 / fp4 r32 0.97-0.98 W, r16 1.03-1.07 L | |
| `depth` 6 | B2 | TP2 r16 0.97 W, TP1 0.97-0.98 W, else T | |
| `depth` 2 / 8, `ddepth` 4 / 12, `ab` 1 / 4, `prio` dn / alt, `dyn`, `lka=early`, `pstg=2` | B2 | `depth=2` and `ab=1` L, the rest T | |
| `gu=wt` | B2 | TP1 r16 0.956 W, r64 0.985 W, r128 T, TP2 1.02-1.04 L | ATT: issue-stalled VMEM 27 %, i.e. the CU's memory path is saturated |
| `shs` 4 / 8 | B2 | T / L | the shared-expert CTAs stay the slowest with 4-token items: the cost is the shared weights read on several XCDs, not the 16 tokens |
| **`xb`** | B2 + `pc=0` | r16 T (no `xh` loss), TP2 r128 0.97-0.98 W; with `depth=6`: 0.967-0.986 everywhere | the shared-expert passes share an L2 |
| `lpf` 2 / 3 | B2 + `pc=0` | T; with `depth=6` / 8: 0.97-0.99 W | |
| **`pc=0`** (pieces for the last round only) | B2 + `xb` + `depth=6` | **r8 0.86-0.96 W, TP1 / TP2 r64 0.94-0.97 W**, r16 / r32 T | TP1 r64 stamps: a CTA with a whole unit and a half piece still runs about 1.75x a one-unit CTA (a piece costs about 0.75 of a unit) |
| `pc=0` corr routing r16 (first form, every item) | B2 | TP1 0.81 W, TP2 0.94 W; r64 / r128 1.15-1.21 L | the last-round form avoids the losses |
| `xh`, `wt`, `d8+lpf2`, `pc=1` | B3, MI355X | per-cell best arm (section 3) | |
| P-L4 best (`sop=1, wcp=default`) | B3, MI355X | r16-r128 1.04-1.24 L; r8 0.87-0.94 W | P-L4's line deal is better at 73-77 items |

**Diagnostic arms** (B0 / B2, MI350X):
- **Bytes and in-flight depth.**
  - `null` (no weight bytes): 0.41-0.53.
  - Streamer floor of the same bytes: 0.42-0.73. Streamer arms with 32 / 64 / 128 KiB in flight per CU: 1.02-1.24 /
    0.66-0.81 / 0.58-0.72 of B1. The MoE behaves like the 32 KiB streamer.
- **Stamps (B3, TP2 r32).** The experts phase is 0.70 of the span (median wave). The combine-barrier wait is 0.15
  (median): the slowest waves end 26 % after the median. The store drain is 0.04.
- **Not a store cost.** The no-stores probe (`diag=nost`) measured 0.66-0.70, but LLVM deletes the down MFMAs and their
  loads when nothing is stored, so it is not a store cost (section 5).

**Decision 15 re-tests.** On B2 / B3 every earlier reject that applies to whole-unit deals was re-run: depth,
ddepth, ab, prio, lka, xh, dyn, uord, pc, tpp-free levers, wt.

- The line-only levers (`wl`, `snap`, `ov`, `hy`) do not apply to unit deals.
- P-L4's `xu` (cross-unit prefetch) is `lka=tail` here (on in every base).
- P-L4's `dsop` is the per-tile down-scale register here (0.2 scale loads per MFMA).
- P-L4's `gap` / `wbal` analogues are `wl` (line only).
- The order of restructures was varied: rr before / after wt, dt / pstg before / after wt, xb before / after pc.
  Verdicts did not change with the order, except that `xh` loses at r16 on every base and `xb` fixes that.

## 3. Final ratios against FlyTape (MI355X, same job)

Each job's FlyTape traced steps are taken before and after our runs (`ft-pre*` / `ft-post*`, mxfp4, P128). FlyTape per
layer = the step's `*.moe_*` phases / 48. `best` = the fastest `moe2` arm of the cell. Its ratio is against
FlyTape's experts + combine, and the range is over the pre / post brackets.

### n07-17 (`lb2-f3`, job 68241845, bus 65)

| cell | best arm | best / FT e+c | B3 / FT e+c | B0 (line) / FT | P-L4 / FT | best / FT MoE | floor / FT |
|---|---|---|---|---|---|---|---|
| tp1 fp8 r8 | x | 1.116 | 1.121 | 1.315 | 1.004 | 0.880 | 0.560 |
| tp1 fp8 r16 | w | 1.009 | 1.042 | 1.166 | 1.106 | 0.856 | 0.651 |
| tp1 fp8 r32 | wx | 0.918 | 0.929 | 1.093 | 1.108 | 0.818 | 0.717 |
| tp1 fp8 r64 | base | 0.925 | 0.925 | 1.029 | 1.150 | 0.848 | 0.733 |
| tp1 fp4 r8 | base | 1.191 | 1.191 | 1.334 | 1.023 | 0.938 | 0.567 |
| tp1 fp4 r16 | w | 1.014 | 1.052 | 1.157 | 1.070 | 0.860 | 0.645 |
| tp1 fp4 r32 | wx | 0.950 | 0.989 | 1.099 | 1.078 | 0.846 | 0.725 |
| tp1 fp4 r64 | d8 | 0.979 | 0.979 | 1.026 | 1.077 | 0.897 | 0.726 |
| tp2 fp8 r8 | x | 1.125 | 1.133 | 1.450 | 1.027 | 0.809 | 0.427 |
| tp2 fp8 r16 | p1 | 0.963 | 0.968 | 1.209 | 1.078 | 0.757 | 0.507 |
| tp2 fp8 r32 | x | 1.029 | 1.050 | 1.318 | 1.242 | 0.829 | 0.687 |
| tp2 fp8 r64 | d8 | 0.965 | 0.966 | 1.117 | 1.070 | 0.838 | 0.631 |
| tp2 fp8 r128 | d8 | 0.925 | 0.946 | 1.164 | 1.062 | 0.816 | 0.642 |
| tp2 fp4 r8 | base | 1.148 | 1.148 | 1.475 | 1.010 | 0.825 | 0.439 |
| tp2 fp4 r16 | base | 0.963 | 0.963 | 1.176 | 1.072 | 0.758 | 0.504 |
| tp2 fp4 r32 | wx | 1.046 | 1.109 | 1.304 | 1.230 | 0.842 | 0.693 |
| tp2 fp4 r64 | x | 1.010 | 1.013 | 1.114 | 1.047 | 0.878 | 0.627 |
| tp2 fp4 r128 | d8 | 0.936 | 0.951 | 1.157 | 1.047 | 0.826 | 0.645 |

- TP1 r128 has no FlyTape trace (FlyTape's TP1 traces stop at B64). There, B3 runs 0.88-0.89 of the line and
  0.85-0.89 of P-L4.
- The two r8 cells with a failed `noacq` mutant (tp1 fp8, tp2 fp4) were timed with `--continue-on-fail`.

### n01-09 (`lb2-f4`, job 68241985, bus 05)

| cell | best arm | best / FT e+c (pre..post) | B3 / FT e+c | B0 (line) / FT | P-L4 / FT | best / FT MoE | floor / FT |
|---|---|---|---|---|---|---|---|
| tp1 fp8 r8 | base | 1.154 (1.142..1.167) | 1.154 | 1.363 | 0.993 | 0.905 | 0.549 |
| tp1 fp8 r16 | w | 0.967 (0.959..0.976) | 0.994 | 1.121 | 1.102 | 0.815 | 0.637 |
| tp1 fp8 r32 | wx | 0.949 (0.943..0.954) | 0.962 | 1.134 | 1.141 | 0.838 | 0.730 |
| tp1 fp8 r64 | x | 0.944 (0.941..0.947) | 0.946 | 1.045 | 1.182 | 0.862 | 0.746 |
| tp1 fp4 r8 | base | 1.173 (1.160..1.186) | 1.173 | 1.338 | 1.002 | 0.920 | 0.552 |
| tp1 fp4 r16 | w | 1.024 (1.015..1.033) | 1.047 | 1.148 | 1.074 | 0.863 | 0.645 |
| tp1 fp4 r32 | wx | 0.965 (0.960..0.971) | 0.986 | 1.140 | 1.110 | 0.852 | 0.735 |
| tp1 fp4 r64 | base | 0.949 (0.946..0.952) | 0.949 | 1.018 | 1.122 | 0.867 | 0.744 |
| tp2 fp8 r8 | x | 1.132 (1.131..1.132) | 1.145 | 1.478 | 1.043 | 0.819 | 0.435 |
| tp2 fp8 r16 | p1 | 0.950 (0.949..0.951) | 0.955 | 1.173 | 1.069 | 0.745 | 0.502 |
| tp2 fp8 r32 | x | 1.050 (1.048..1.051) | 1.064 | 1.310 | 1.273 | 0.850 | 0.702 |
| tp2 fp8 r64 | base | 0.898 (0.898..0.898) | 0.898 | 1.027 | 1.072 | 0.777 | 0.630 |
| tp2 fp8 r128 | d8 | 0.996 (0.993..0.999) | 1.002 | 1.245 | 1.143 | 0.870 | 0.694 |
| tp2 fp4 r8 | base | 1.127 (1.126..1.127) | 1.127 | 1.482 | 1.019 | 0.816 | 0.451 |
| tp2 fp4 r16 | p1 | 0.962 (0.961..0.963) | 0.966 | 1.210 | 1.055 | 0.755 | 0.502 |
| tp2 fp4 r32 | wx | 1.071 (1.070..1.072) | 1.075 | 1.317 | 1.242 | 0.867 | 0.707 |
| tp2 fp4 r64 | x | 0.960 (0.960..0.960) | 0.964 | 1.097 | 1.046 | 0.830 | 0.625 |
| tp2 fp4 r128 | d8 | 0.979 (0.977..0.982) | 0.991 | 1.196 | 1.137 | 0.856 | 0.690 |

- The A-vs-A and recompiled controls stayed within 0.4 % of the base on this node. On n07-17 they moved by up to 4.5 %
  in one cell (tp2 fp4 r32), so verdicts within 3 % are ties there.
- The earlier B2 runs (`lb2-f1` n07-17, `lb2-f2` n01-09) agree with these within 2-5 % per cell.

## 4. Correlated routing (MI350X triage, `corr:1.0:0.5:8`)

Ratios to B2, same job.

| cell | line | xh | wt | dyn | pieces (first form) |
|---|---|---|---|---|---|
| tp1 fp8 r16 | 1.01 | 1.07 L | 0.94 W | T | 0.81 W |
| tp1 fp8 r64 | 1.13-1.15 L | 0.98-1.02 | 0.97 W | T | 1.21 L |
| tp2 fp8 r16 | 1.26-1.28 L | 1.05 L | 1.03 L | T | 0.93-0.94 W |
| tp2 fp8 r32 | 1.36-1.37 L | 1.03 | 1.01-1.02 | T | 0.99-1.01 |
| tp2 fp8 r128 | 1.25-1.27 L | 0.88 W | 1.01-1.04 | T | 1.15-1.17 L |

Under correlated routing the item counts drop and a few experts carry many passes. The rebuild is 0.73-0.88 of the
line there (the line's per-segment and partial-chunk overheads grow with the skew). Correlated-routing MI355X runs and
a FlyTape comparison under its own routing were not done.

## 5. Findings (also in the alola skill, rule 185)

- **Compilers differ.** The cluster's torch comgr and the local Windows comgr compile the same IR differently. A
  wave-owned gate_up build with no scratch locally spilled 2.3 KiB on the cluster: the cluster LLVM hoisted a whole
  unrolled K loop's loads and LDS reads above a uniform branch. Every timing verdict must use the cluster comgr's ISA
  (`isa350.sh`). The final builds have no VGPR scratch on the cluster compiler.
  - The reverse toolkit's report (`reverse/MOE-LB.md`, sha `580f8526`) found serial MFMA chains, a `s_nop 6` behind
    every store and SGPR spills. `dt` / `doff` / `pstg` remove the first two.
  - SGPR spill traffic remains: about 330 `v_writelane` and 600-900 `v_readlane` static per kernel. Resources built at
    every use (`lrs`) did not reduce it.
- **Hazard pad.** Inline-asm VALU reads of an SGPR pair written by the preceding VALU compare (`v_cmp` -> asm
  `v_mbcnt`) gave wrong ranks on MI350X, and the failures moved with unrelated code. `s_nop 4` in the asm (under `hz`)
  fixed them. Answer to the `hz` question:
  - P-L3's `gemm.py` grid barrier and the CCL helpers were scanned by W6's `asm_hazard.py`: all clean (rule 182).
  - The pad is still needed wherever an asm instruction consumes a VALU-written SGPR. The new case here is the
    ballot -> `mbcnt` pair.
- **Deterministic sorts.** A per-CTA LDS sort that every CTA recomputes must be deterministic. The atom-order counting
  sort gave each CTA a different order (units dealt twice or never). The stable ballot-rank sort fixed it.
- **DCE trap.** A probe that drops the down stores lets LLVM delete the down MFMAs and their loads.
- **Stamps perturb tight builds.** Per-segment stamps changed a tight build's register allocation (wt ran 2x slower
  stamped): use ATT there.
- **Combine barrier.** At r32 and above the combine barrier wait is the largest non-streaming item. It is imbalance:
  the slowest waves end about 26 % after the median, and the store drain is small. At TP1 r64 the CTAs holding a second
  (piece) unit set the span, because a half-item piece costs about 0.75 of a whole unit (per-segment overhead).

## 6. What remains

- **r8** (1.12-1.19 against FlyTape; P-L4 1.00-1.03). Options:
  - A dispatcher pick of P-L4's line kernel at r8.
  - Finer pieces than 4 per item.
  - A wave-granular two-phase form (FlyTape's TP2 `moe_up` / `moe_down`, a grid barrier between).
- **TP2 r32 (1.03-1.05) and TP1 r16 (tie).** The combine-barrier imbalance (about 15 % of the span) and the
  non-overlapped per-unit compute (`null` 0.41-0.53) are the levers. The CU's memory path is already saturated during
  the stream (ATT).
  - The likely structural answer is the wave-granular two-phase design: perfect balance at 16-column units, h through
    L2 instead of LDS. It is not built.
  - Deeper in-flight depth without stalls in the atom loop also helps: the 64 KiB streamer is 0.66-0.81 of today's time.
- **Not built:**
  - `combine=fused_next` (the combine folded into the next stage).
  - `entry=router` (the router GEMM + top-k in the kernel; `entry=logits` exists, numerics only).
  - The Rust direct-compiler track.
  - A tuner lookback beyond the per-base sweeps above.
- **Not done:**
  - Correlated-routing MI355X verdicts.
  - The 8-row bucket's `noacq` mutant at 8 rows (probabilistic).
- **SGPR spill traffic** (section 5) and the route's DPP argmax chains (reverse toolkit) are untouched.
