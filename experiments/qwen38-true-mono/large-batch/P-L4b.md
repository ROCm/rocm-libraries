# WP-L2d: the large-batch MoE rebuild (`moe_lb2`) against FlyTape's MoE

Ratios only. Absolute times are in the private copy
(`hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L4b.md`).

## 0. Summary

- **Kernel.** `large_batch/moe_lb2.py` (names `q38lb2_moe*`), sha256 `7a610aea4cba6b26...` (CRLF stripped) at the
  round-2 final runs. P-L4's `moe_lb.py` is not edited; `moe_lb2` imports
  five helpers from it.
- **Harness.** `hackathon-prep/qwen38/truemono-lb/p-l4b/`:
  - the bench and reports: `pl4b_bench.py`, `pl4b_report.py`, `final_report.py`, `whole_report.py`;
  - the in-job tuner: `lb2_tune.py`, `tune355.sh` (venv) / `tune355c.sh` (container);
  - jobs and picks: `submitf.sh`, `picks.sh`, `mkpicks.py`;
  - ISA and tooling: `isa350.sh`, `isa_metrics.py`, `rustbuild.sh`, and the patch scripts.

  The tuner's space is `tune/catalog.moe_lb2`.
- **Base B5** (round 1, sections 1-6): `deal=rr, dt=4, doff=imm, pstg=4, pc=0, xb=1, depth=6, pcal=1, cmb=flow,
  dsq=1`.
  - Whole experts are dealt round robin, with XCD blocks per round.
  - The down takes FlyTape's form: 4 output tiles share an h fragment, with staged 256 B partial-row stores and the
    quad-tile down-scale layout.
  - The last round's experts are split into pieces aligned to 4-atom down chunks.
  - The combine has no grid barrier: per-token slot counters.
- **Round 2** (section 7): a per-cell dispatcher over B5. The tuner's combination lookback ran over every knob
  (losers included), on both nodes, in both compilers. New levers:
  - bf16 partials (`pdt`, accuracy-gated);
  - longer combine-poll pauses;
  - the router GEMM + top-k in the kernel (`entry=router`);
  - `combine=fused_next`.

  The picks are 0.91-0.98 of B5 and are cross-validated between the nodes.
- **Against FlyTape's experts + combine** (same job, MI355X, container compiler, fp8 and fp4; `lb2-f14` n01-09 /
  `lb2-f15` n07-17, section 7.6):

  | rows | TP1 | TP2 |
  |---|---|---|
  | r8 | 0.865-0.907 | 0.797-0.807 |
  | r16 | 0.906-0.910 | 0.823-0.829 |
  | r32 | 0.831-0.854 | 0.891-0.953 |
  | r64 | 0.827-0.864 | 0.813-0.820 |
  | r128 | no FlyTape TP1 r128 trace | 0.794-0.855 |

  - DESIGN-LB section 15's gate (1.00 or less everywhere, both nodes) holds with margin.
  - The 0.85 target:
    - met on both nodes at TP2 r8 / r16 / r64;
    - met on one node and 0.851-0.864 on the other at TP1 r32 / r64 and TP2 r128;
    - not met at TP1 r8 (0.87-0.91), TP1 r16 (0.91) or TP2 r32 (0.89-0.95).
  - FlyTape's own time moves by up to 12 % between jobs, so the ratios move with it.
- **The whole MoE** (router GEMM + top-10 + experts + combine in one kernel; section 7.5) is 0.75-0.92 of FlyTape's
  router + experts + combine.
  - With `combine=fused_next` (the next stage sums the slot partials), the chain is 0.79-0.87 of FlyTape's MoE plus
    the same consumer read of y.
  - The fused chain is 0.92-0.97 of our unfused chain.
- **Numerics.** Every build and arm passed on MI355X in every job:
  - T0 against the float64 reference (bf16 partials: the error stays at 0.52-0.59 of a bf16-per-slot combine's);
  - the deterministic slot-order combine, bitwise knob invariance, the 2-layer walk;
  - the dirty-h and stale-read twins;
  - the router's exact-logit equality;
  - the consumer chains;
  - the must-fail mutants, including the new deterministic stale-read mutant `flowrel` (caught at every cell, r8
    included) and `rsum` / `rsc0` for the router.

  The tuner's numerics found and fixed 4 bugs in default-off levers (section 7.1).
- **Round 1 results** (sections 2-4): B5 at 0.81-0.98 of FlyTape's experts + combine (`lb2-f8` / `f9`); correlated
  routing 0.58-0.78 of the line.

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
5. **Combine (`cmb=flow`).** No grid barrier. Each segment, once its partial stores have completed (vmcnt 0 in every
   wave, a CTA barrier), adds 12 / nseg to a per-token counter (two parities, so the next layer's counters are zeroed
   by this layer's combine). A combine unit (token, 256 columns) polls its token's counter for 132 (11 slots x 12),
   then reads the desc and partials with sc1 loads. y = sum over slots (order) of w_s x sum over segments (order) of
   the partials. `cmb=gbar` (grid barrier) stays as a knob.

**Alternative builds kept as knobs:**

- `gu=wt`: FlyTape's fused gate_up. A wave owns whole gate / up row-tile pairs over the full K (TP2: K halves, `wk=2`).
  silu(g) u goes to an f32 panel, all threads requantize after one barrier. Options: `wg` sweeps, `word`, `wpin`.
- `deal=dyn`: whole units from a global ticket. `uord=heavy`: a stable counting sort by token count.
- `deal=line`: the WP-L2d starting point.
- `shs`: shared-expert items of 4 / 8 tokens.
- `lpf`: x B-operand prefetch distance.
- `pst`: partial-store cache bits.
- `pipe=w2`: FlyTape's wave-granular two-phase form (gate_up units, a grid barrier, down units; h through L2).
- `shd` (shared-expert down split over two CTAs), `dsk` / `sdl` (down-scale layouts and loads), `cspread`, `gpc` /
  `gpause` (barrier polling), `cob=direct` / `dlev` (the Rust direct compiler and its lever scopes).
- `sskip` is refused in `check()`: it fails numerics.

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
- **B3**: B2 + `pc=0, xb=1, depth=6`.
- **B4**: B3 + `pcal=1, cmb=flow`.
- **B5**: B4 + `dsq=1` (final).

Later verdicts come from the MI355X jobs `lb2-f5` (n07-17, B4), `lb2-f8` (n01-09, B5), `lb2-f9` (n07-17, B5) and
`lb2-c1` (n01-09, B5, correlated routing).

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
| `div_small` (no runtime integer divide in the unit decode) | B3 | removes the `pc=0` decode overhead | |
| **`pcal`** (pieces aligned to 4-atom down chunks, cap min(4, ceil(NA / 4))) | B3 | **r8 0.87-0.91 W**, else T | a piece no longer ends mid down chunk |
| **`cmb=flow`** (per-token counters, no combine barrier) | B3 + `pcal` | **r8 0.86-0.89, r16 0.92-0.95, r32 0.92-0.94, r64 0.97-0.99, r128 0.95-0.97 W** | the slow CTAs' tails no longer hold every token's combine |
| P-L4 at r8 (the dispatcher pick) | B5, MI355X | 1.07-1.25 L | not needed once `pcal` + `flow` are in |
| **`dsq`** (quad-tile down-scale layout) | B4 | **TP2 about 0.98 W**, TP1 T | one scale load per 4 tiles |
| `pipe=w2` (FlyTape's two-phase form) | B4 | 1.04-1.16 L, r8 0.97 W (still above B5 at r8) | the grid barrier between phases costs more than the balance gains |
| `shd`, `shs` 4 / 8, `dsk` 2 / 4 / 5, `sdl`, `cspread`, `gpc` / `gpause` variants, `ddepth` 12 / 16, `lpf` 2 / 3, `xh`, `dyn`, `uord=heavy` | B4 / B5 | T or L | |
| `sskip` (skip the shared-expert sort) | B4 | numerics fail: refused | |
| `pst=def` / `nt` (+ an L2 release before the counter bump) | B4 | 1.04-1.20 L | the `buffer_wbl2` release costs more than sc1 stores |
| `cob=direct` (Rust direct code-object compiler), `dlev` lever scopes | B4 | 1.01-1.08 L, levers T | bitwise equal to the COMGR build |
| `gu=wt, wg=2, pc=1` (`w`) | B5, MI355X | TP1 r16 0.96-0.97 W (the dispatcher pick), elsewhere T / L | |
| `pc=1` (`p1`), `dsq=0` (`b4`), `depth=8, lpf=2` (`d8`), `xh` (`x`) | B5, MI355X | best arm within 2.5 % of B5 (TP2 r16 `p1` 0.97-0.98); `p1` / `w` r8 1.25-1.35 L, `x` r16 1.04-1.16 L, `b4` TP2 1.02-1.05 L | the per-cell pick matters only at TP1 r16 |

**Diagnostic arms** (B0 / B2, MI350X):
- **Bytes and in-flight depth.**
  - `null` (no weight bytes): 0.41-0.53.
  - Streamer floor of the same bytes: 0.42-0.73. Streamer arms with 32 / 64 / 128 KiB in flight per CU: 1.02-1.24 /
    0.66-0.81 / 0.58-0.72 of B1. The MoE behaves like the 32 KiB streamer.
- **Stamps (B3, TP2 r32).** The experts phase is 0.70 of the span (median wave). The combine-barrier wait is 0.15
  (median): the slowest waves end 26 % after the median. The store drain is 0.04.
- **Not a store cost.** The no-stores probe (`diag=nost`) measured 0.66-0.70, but LLVM deletes the down MFMAs and their
  loads when nothing is stored, so it is not a store cost (section 5).

**Decision 15 re-tests.** On B2 / B3, and again on B4 / B5, every earlier reject that applies to whole-unit deals
was re-run: depth,
ddepth, ab, prio, lka, xh, dyn, uord, pc, tpp-free levers, wt.

- The line-only levers (`wl`, `snap`, `ov`, `hy`) do not apply to unit deals.
- P-L4's `xu` (cross-unit prefetch) is `lka=tail` here (on in every base).
- P-L4's `dsop` is the per-tile down-scale register here (0.2 scale loads per MFMA).
- P-L4's `gap` / `wbal` analogues are `wl` (line only).
- The order of restructures was varied: rr before / after wt, dt / pstg before / after wt, xb before / after pc.
  Verdicts did not change with the order, except that `xh` loses at r16 on every base and `xb` fixes that.

## 3. Final ratios against FlyTape (MI355X, same job)

Each job's FlyTape traced steps are taken before and after our runs (`ft-pre*` / `ft-post*`, mxfp4, P128). FlyTape per
layer = the step's `*.moe_*` phases / 48 (`pl4b_report.py`). `best` = the fastest `moe2` arm of the cell (`B5` = the
base). Its ratio is against FlyTape's experts + combine, and the range is over the pre / post brackets. `floor` is the
streamer over the same weight bytes. Arms: `b4` = `dsq=0`, `w` = `gu=wt, wg=2, pc=1`, `p1` = `pc=1`, `d8` =
`depth=8, lpf=2`, `x` = `xb=0, xh=1`.

### n01-09 (`lb2-f8`, job 68244019, bus 75)

Neighbours during the run (sacct): our 1-GPU jobs `w9-n4a`, `w8-wp2a` / `wp2b`, `rb-k12`, and one other user's build.

| cell | best arm | best / FT e+c (pre..post) | B5 / FT e+c | B0 (line) / FT | P-L4 / FT | best / FT MoE | floor / FT |
|---|---|---|---|---|---|---|---|
| tp1 fp8 r8 | b4 | 0.922 (0.913..0.932) | 0.944 | 1.321 | 1.006 | 0.723 | 0.556 |
| tp1 fp8 r16 | w | 0.968 (0.966..0.971) | 1.009 | 1.167 | 1.116 | 0.816 | 0.643 |
| tp1 fp8 r32 | p1 | 0.837 (0.835..0.840) | 0.844 | 1.041 | 1.114 | 0.743 | 0.707 |
| tp1 fp8 r64 | b4 | 0.838 (0.838..0.839) | 0.844 | 0.967 | 1.146 | 0.769 | 0.708 |
| tp1 fp4 r8 | b4 | 0.937 (0.927..0.946) | 0.942 | 1.352 | 1.007 | 0.735 | 0.557 |
| tp1 fp4 r16 | w | 0.967 (0.965..0.970) | 0.995 | 1.139 | 1.066 | 0.815 | 0.641 |
| tp1 fp4 r32 | p1 | 0.838 (0.835..0.840) | 0.843 | 1.044 | 1.087 | 0.743 | 0.707 |
| tp1 fp4 r64 | B5 | 0.848 (0.847..0.848) | 0.848 | 0.965 | 1.084 | 0.778 | 0.706 |
| tp2 fp8 r8 | B5 | 0.844 (0.835..0.854) | 0.844 | 1.504 | 1.033 | 0.608 | 0.431 |
| tp2 fp8 r16 | p1 | 0.853 (0.851..0.855) | 0.874 | 1.221 | 1.078 | 0.669 | 0.505 |
| tp2 fp8 r32 | p1 | 0.981 (0.980..0.982) | 0.987 | 1.343 | 1.298 | 0.788 | 0.710 |
| tp2 fp8 r64 | B5 | 0.816 (0.813..0.818) | 0.816 | 1.019 | 1.067 | 0.705 | 0.633 |
| tp2 fp8 r128 | B5 | 0.918 (0.918..0.919) | 0.918 | 1.280 | 1.154 | 0.800 | 0.706 |
| tp2 fp4 r8 | x | 0.810 (0.801..0.820) | 0.816 | 1.436 | 1.010 | 0.583 | 0.443 |
| tp2 fp4 r16 | p1 | 0.859 (0.857..0.861) | 0.869 | 1.200 | 1.055 | 0.673 | 0.503 |
| tp2 fp4 r32 | p1 | 0.963 (0.962..0.964) | 0.974 | 1.349 | 1.260 | 0.774 | 0.712 |
| tp2 fp4 r64 | B5 | 0.820 (0.818..0.823) | 0.820 | 1.022 | 1.035 | 0.710 | 0.630 |
| tp2 fp4 r128 | B5 | 0.913 (0.913..0.914) | 0.913 | 1.282 | 1.154 | 0.795 | 0.703 |

### n07-17 (`lb2-f9`, job 68244192, bus 75)

Neighbours during the run (sacct): our 1-GPU `w9` / `w8` jobs.

| cell | best arm | best / FT e+c (pre..post) | B5 / FT e+c | B0 (line) / FT | P-L4 / FT | best / FT MoE | floor / FT |
|---|---|---|---|---|---|---|---|
| tp1 fp8 r8 | b4 | 0.923 (0.912..0.935) | 0.933 | 1.327 | 1.005 | 0.725 | 0.553 |
| tp1 fp8 r16 | w | 0.979 (0.972..0.987) | 1.013 | 1.173 | 1.133 | 0.825 | 0.647 |
| tp1 fp8 r32 | b4 | 0.876 (0.871..0.882) | 0.882 | 1.089 | 1.111 | 0.779 | 0.721 |
| tp1 fp8 r64 | b4 | 0.910 (0.908..0.912) | 0.918 | 1.040 | 1.203 | 0.833 | 0.761 |
| tp1 fp4 r8 | b4 | 0.937 (0.925..0.949) | 0.939 | 1.350 | 1.010 | 0.735 | 0.562 |
| tp1 fp4 r16 | w | 0.977 (0.969..0.984) | 1.004 | 1.148 | 1.085 | 0.823 | 0.661 |
| tp1 fp4 r32 | p1 | 0.877 (0.872..0.883) | 0.883 | 1.089 | 1.079 | 0.779 | 0.722 |
| tp1 fp4 r64 | B5 | 0.916 (0.913..0.918) | 0.916 | 1.028 | 1.144 | 0.838 | 0.756 |
| tp2 fp8 r8 | B5 | 0.837 (0.826..0.849) | 0.837 | 1.483 | 1.020 | 0.606 | 0.425 |
| tp2 fp8 r16 | p1 | 0.854 (0.852..0.855) | 0.870 | 1.205 | 1.077 | 0.671 | 0.501 |
| tp2 fp8 r32 | p1 | 0.943 (0.941..0.945) | 0.948 | 1.302 | 1.229 | 0.767 | 0.682 |
| tp2 fp8 r64 | d8 | 0.853 (0.853..0.853) | 0.857 | 1.099 | 1.078 | 0.741 | 0.631 |
| tp2 fp8 r128 | p1 | 0.946 (0.943..0.949) | 0.948 | 1.294 | 1.195 | 0.824 | 0.723 |
| tp2 fp4 r8 | B5 | 0.828 (0.817..0.840) | 0.828 | 1.461 | 1.000 | 0.599 | 0.436 |
| tp2 fp4 r16 | p1 | 0.855 (0.854..0.856) | 0.865 | 1.204 | 1.065 | 0.672 | 0.509 |
| tp2 fp4 r32 | p1 | 0.932 (0.930..0.935) | 0.935 | 1.296 | 1.209 | 0.758 | 0.685 |
| tp2 fp4 r64 | B5 | 0.847 (0.847..0.847) | 0.847 | 1.092 | 1.040 | 0.736 | 0.623 |
| tp2 fp4 r128 | d8 | 0.938 (0.935..0.942) | 0.940 | 1.302 | 1.175 | 0.817 | 0.728 |

- TP1 r128 has no FlyTape trace (FlyTape's TP1 traces stop at B64). There, B5 runs 0.82-0.84 of the line and
  0.76-0.82 of P-L4.
- The A-vs-A (`lb2b`) and recompiled (`lb2rc`) controls stayed within 1.5 % of the base in every cell on both nodes.
- B5 alone is above 1.00 only at TP1 r16 (1.004-1.013); the dispatcher's `w` arm there is 0.97-0.98.
- History: B3 (`lb2-f3` n07-17, `lb2-f4` n01-09) was 1.12-1.19 at r8 and 1.03-1.07 at TP2 r32. B4 (`lb2-f5`,
  job 68242669, n07-17, bus 05) was at or under 1.00 everywhere except TP2 r32 (1.004 / 1.011).

## 4. Correlated routing (`corr:1.0:0.5:8`)

### MI355X (`lb2-c1`, job 68244268, n01-09, bus 65, B5)

Ratios to B5, same job. Numerics passed in every cell (same checks and mutants as section 0).

| cell | line / B5 | P-L4 / B5 | best arm / B5 | B5 (corr) / FT e+c (uniform trace) |
|---|---|---|---|---|
| tp1 fp8 r16 | 1.44 | 1.15 | B5 | 0.66 |
| tp1 fp8 r32 | 1.44 | 1.16 | B5 | 0.54 |
| tp1 fp8 r64 | 1.29 | 1.19 | `w` 0.978 | 0.46 |
| tp1 fp4 r64 | 1.29 | 1.13 | `w` 0.997 | 0.45 |
| tp2 fp8 r16 | 1.72 | 1.30 | B5 | 0.64 |
| tp2 fp8 r32 | 1.70 | 1.29 | B5 | 0.66 |
| tp2 fp8 r64 | 1.47 | 1.29 | `p1` 0.978 | 0.49 |
| tp2 fp8 r128 | 1.60 | 1.41 | `p1` 0.989 | 0.48 |
| tp2 fp4 r128 | 1.62 | 1.34 | `p1` 0.986 | 0.49 |

FlyTape has no correlated-routing trace: the last column compares with its uniform-routing experts + combine (from
`lb2-f8`) and only shows that correlated routing (fewer, larger items) is cheaper for B5.

### MI350X triage (ratios to B2, same job)

| cell | line | xh | wt | dyn | pieces (first form) |
|---|---|---|---|---|---|
| tp1 fp8 r16 | 1.01 | 1.07 L | 0.94 W | T | 0.81 W |
| tp1 fp8 r64 | 1.13-1.15 L | 0.98-1.02 | 0.97 W | T | 1.21 L |
| tp2 fp8 r16 | 1.26-1.28 L | 1.05 L | 1.03 L | T | 0.93-0.94 W |
| tp2 fp8 r32 | 1.36-1.37 L | 1.03 | 1.01-1.02 | T | 0.99-1.01 |
| tp2 fp8 r128 | 1.25-1.27 L | 0.88 W | 1.01-1.04 | T | 1.15-1.17 L |

Under correlated routing the item counts drop and a few experts carry many passes. The rebuild is 0.73-0.88 of the
line there (the line's per-segment and partial-chunk overheads grow with the skew). The MI355X run above confirms it
on B5. A FlyTape comparison under its own correlated routing is not possible without a FlyTape trace at that routing.

## 5. Findings (also in the alola skill, rule 185)

- **Compilers differ.** The cluster's torch comgr and the local Windows comgr compile the same IR differently. A
  wave-owned gate_up build with no scratch locally spilled 2.3 KiB on the cluster: the cluster LLVM hoisted a whole
  unrolled K loop's loads and LDS reads above a uniform branch. Every timing verdict must use the cluster comgr's ISA
  (`isa350.sh`). The final builds have no VGPR scratch on the cluster compiler.
  - The reverse toolkit's report (`reverse/MOE-LB.md`, sha `580f8526`) found serial MFMA chains, a `s_nop 6` behind
    every store and SGPR spills. `dt` / `doff` / `pstg` remove the first two.
  - SGPR spill traffic remains: about 330 `v_writelane` and 600-900 `v_readlane` static per kernel. Resources built at
    every use (`lrs`) did not reduce it. `isa_metrics.py` places them: on B5 only 11 of 898 `v_readlane` sit inside the
    hot atom / down loops, so removing the spills is not a timing lever.
  - The route's DPP argmax chains are only in `entry=logits`; the timed path takes route pairs.
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
  `cmb=flow` removes the barrier: each token's combine waits only for its own 11 slots.
- **Stale-read mutant under flow.** `noacq` (acquire dropped at the combine barrier) has no barrier to act on under
  `cmb=flow`, and its layer-start pretouch form was never caught. The flow hand-off is covered instead by `flow0` (a
  counter that never counts: must fail, deterministic) and `flow` (an early threshold: reported, caught only where a
  race shows). A reliable r8 stale-read mutant was not found.
- **L2 release cost.** Plain or `nt` partial stores plus a `buffer_wbl2` release before the counter bump cost 4-20 %
  against sc1 stores.
- **Two cluster compilers.** The venv jobs (`q38_job.sh --mode venv`: torch 2.12, ROCm 7.2 comgr) and the container
  jobs (the FlyTape image: torch 2.13, ROCm 10.0 comgr) compile the same IR differently. Picks tuned under the venv
  compiler did not all hold under the container compiler, where the FlyTape verdicts run: a TP1 r32 pick at 0.98 of
  B5 measured 1.05-1.09, a TP1 r8 pick at 0.95 measured 1.00. Tune in the verdicts' mode (`tune355c.sh`).
- **Rust direct compiler.** `ROCKE_CODEOBJ_BACKEND=direct` (built from `platform/rust/rocke-codeobj`) produced
  bitwise-equal results, ran 1-8 % slower than the COMGR build, and its lever scopes (`dlev`) had no measurable
  effect. It refuses i16 vector extracts (the two-byte x path uses byte extracts under `cob=direct`).

## 6. What remains

- **The 0.85 target at TP1 r8, TP1 r16 and TP2 r32.** The stamps and ATT (section 7.4) show:
  - **TP1 r8 and r16.** The cost is imbalance at item / piece granularity: 8- and 4-atom pieces at r8, and 141 whole
    items on 256 CTAs at r16. A piece's fixed part, above all its 16 x H partial rows, makes finer pieces lose.
    Candidates:
    - pieces whose partial rows cover only the valid tokens;
    - sub-item splits without per-piece partial rows (a cross-CTA reduction of the gate_up K halves);
    - a balanced line with cheap segments.
  - **TP2 r32.** One whole item per CTA streams at about 0.73 of the streamer's rate. VMEM issue is full; the reduction
    barrier and the per-item fixed phases make up the rest.
- **The whole MoE.** The router + top-k adds 11-23 % to B5 with routes given. The tag-polled pair hand-off and the
  one-wave top-10 per token are the next targets (`rsync=cnt` lost).
- **`combine=fused_next`** is modelled with a consumer kernel. The real consumer (the HC mix stage) has to adopt the
  partials' read in the monokernel (WP-L3).
- **Built and losing (default-off knobs, in the tuner's space):** `pipe=w2`, `cob=direct` (fp8 only; fp4 refused),
  `rsync=cnt`, `rmode=ks`, and `shd` / `dsk` / `cspread` alone (some of them win in combinations: section 7.1).
- **Verdict hygiene.** Tune in the verdicts' compiler (container). Picks for TP2 r8 / r16 / r64 / r128 and TP1 r8 /
  r16 / TP2 r32 came from the venv tuner and held under the container compiler. TP1 r32 / r64 needed the container
  re-tune.

## 7. Round 2: toward 0.85 (after the gate)

### 7.1 The tuner's combination lookback (`qwen38_mono/tune/`, every knob, losers included)

- **Space.** `tune/catalog.moe_lb2(tp, act, bucket, base)`: 48 knobs over `Moe2Knobs` with every built value,
  the losers included:
  - `pipe=w2`;
  - the Rust direct compiler with kernel and per-stage lever scopes (`dlev` / `dls` / `dlx` / `dlc`);
  - `shd`, `shs`, `dsk`, `sdl`, `cspread`, the `pst` variants;
  - `gpc` / `gpause`, including `s_sleep32/64/127`;
  - `pdt`;
  - every deal, piece, order and depth knob.

  A base with `entry=router` adds `rmode`, `rks` and `rsync`.
  - Legality is `Moe2Knobs.check()`. A level that is legal only with a companion level is reached by the engine's
    repair.
  - A non-B5 base re-levels every categorical knob: its own value becomes level 0.
  - The catalog contract test covers the space (`test_qwen38_tune.py`).
- **Search.** `SignedLookback` per cell (tp, rows). The objective is the geometric mean over fp8 and fp4 of the arm's
  median time / B5's, measured in the same shuffled interleaved rounds with the A-vs-A and recompiled controls.
  1. One lever at a time, with every reject re-tested after each accepted move (decision 15).
  2. The climb repeated in 2 other lever orders (order diversity).
  3. The model-guided combination lookback over every knob and level.
  4. A revisit of the rejects.

  Every arm passes numerics before timing: T0, bitwise equal to the base when order-preserving, 2 launches bitwise
  equal, and the 2-layer walk. The base's must-fail mutants run once per session.
- **In-job.** The search runs on the GPU node (`p-l4b/lb2_tune.py`): about 1.5 s per evaluation, 300-500
  evaluations per cell in 4-9 minutes.
- **Passes.**
  1. From B5, on one node.
  2. From the pass-1 pick, on the other node. All ties: the picks are local optima on both nodes.
  3. From the picks, with the bug fixes below and the new levers (`pdt`, longer poll pauses), on both nodes with new
     seeds.

  The whole-MoE bases (`entry=router`) get their own search.
- **Common finding.** Every pick includes a combine-poll pause of `s_sleep16` or longer (`gpause`). Under the flow
  combine, the waves spin on their tokens' counters while the slow CTAs still stream; the pause takes that traffic
  off the memory path.
- **What the passes found** (ratios to the run's own base, the confirm step: 16 rounds, both regimes, WIN unless
  marked):

  | cell | pass 1 (venv, from B5) | pass 3 (venv, from the pass-1 pick; n01-09 / n07-17) | container | what the picks change |
  |---|---|---|---|---|
  | TP1 r8 | 0.954-0.958 | 0.952-0.969 / 0.960-0.965 | tie from pass 3 | longer poll pause, valid rows only (`xv`), 4-token shared items, bf16 partials, `dt=5` |
  | TP1 r16 | 0.945-0.948 | 0.973-0.990 / 0.967-0.972 | tie from pass 3 | wave-owned gate_up (`gu=wt, wg=1, word=grp, wk=1`), `prio`, `sdl`, `shs=8`, bf16 partials |
  | TP1 r32 | 0.982-0.983 | - | 0.960-0.964 from B5 | `prio=dn, pc=1, pcal=0, sdl, gpc=sc1, s_sleep64, lpf=3, pstg=2, ab=4`, bf16 partials |
  | TP1 r64 | 0.986-0.988 | - | 0.962 from B5 | `xv, dt=1, pcal=0, s_sleep64`, bf16 partials |
  | TP2 r8 | 0.960-0.962 | - | - | `ddepth=12, prio=alt, xv, pc=3, gpc=sc1`, 4-token shared items |
  | TP2 r16 | 0.950-0.957 | - | - | `depth=10, ddepth=16, prio=dn, xv, doff=mul, gpc=sc1`, 8-token shared items |
  | TP2 r32 | 0.980 / 0.984 (fp4 T) | 0.968-0.974 / 0.961-0.971 | tie from pass 3 | `dt=1, doff=mul, dsk=4, sdl, gpc=sc1, s_sleep32, pstg=4, ab=1`, bf16 partials |
  | TP2 r64 | 0.98 | - | - | `depth=10, xv, lds1=0` |
  | TP2 r128 | 0.983-0.985 | 0.962 / 0.96-0.97 | - | `dt=5, sdl, s_sleep32`, bf16 partials |

- **Bugs the tuner found.** These are numerics failures of default-off levers, in combinations never built by hand.
  Each was fixed or refused in `check()`, then re-verified on MI355X (`lb2-n60`).
  - **`pst=nt` / `def` under `cmb=flow`.** There was no L2 release before the per-token counter bump; the release
    existed only at the grid-barrier arrival. Fixed: a `buffer_wbl2` before the bump.
  - **`pc=4` with `pcal` at TP2.** An item has 3 down chunks, so one of 4 aligned pieces was empty and its partial row
    was never written. Refused: under `pcal`, `pc` is at most the item's down chunks.
  - **`gu=wt` with one sweep and the panel aliasing the x rows** (`pal`, TP1 fp8). The first wave to finish wrote the
    panel over x rows that other waves were still reading. Fixed: a CTA barrier before the panel stores when the
    panel is aliased.
  - **`uord != id` with `pc=0`.** The harness's host model of the descriptors ignored the order; the kernel was
    correct. Fixed in `pl4b_bench.item_nseg`.
  - **`cob=direct` at fp4.** The direct compiler refuses the fp4 requant's inline asm (`v_cvt_scalef32_pk*`). Refused
    in `check()`; the tuner measures a direct arm with COMGR at fp4.

### 7.2 New levers

- **`pdt=bf16`** (bf16 partials; accuracy-gated). Halves the partial bytes written by the down and read by the
  combine.
  - Timing: 0.967-0.987 of B5 alone, and about 0.97-0.98 on top of every pick.
  - T0 holds: the mean error against the float64 reference is 0.52-0.59 of a bf16-per-slot combine's error (f32
    partials: 0.34-0.38).
- **Longer poll pauses** (`gpause` = `s_sleep` 32 / 64 / 127). `s_sleep32` / `s_sleep64` alone: 0.978-0.981 at
  TP1 r8.
- **`entry=router`** (the router GEMM in the kernel). logits = xr . wr^T (bf16 MFMA 16x16x32), then P-L4's top-10
  (`entry=logits`), the tagged pairs and the sort.
  - **Modes.**
    - `rmode=ks`: units of (16 rows, 16 experts, K / `rks`) over the grid's waves; the route sums the `rks` partial
      slots in order.
    - `rmode=cta`: one 16 x 16 logits tile per CTA. The K split runs over its 8 waves and is summed in LDS in wave
      order (deterministic). `rmode=cta` beats `ks` by 1-8 %.
    - `rsync=cnt` (one routed-token counter instead of the per-pair tags) is kept as a knob; it lost 1.5-4 %.
  - **Numerics.** The router inputs are exact: entries in {-1, 0, 1} and {-1, 0, 1} / 16, plus a per-expert
    tie-break, so every partial sum is exact in f32. Then:
    - the in-kernel logits equal the host's bit for bit, and the routes equal the host's;
    - the whole output is bit for bit the `entry=logits` build's;
    - the must-fail mutants `rsum` (a K split left out) and `rsc0` (the pairs read without waiting) are caught at
      every cell.
  - **Cost against `entry=route`** (route pairs given, B5): `lg` (top-k + tags) 1.03-1.17; with the router GEMM on
    top (`rmode=cta`), 1.08-1.32 in total.
- **`fnext`** (`combine=fused_next`, DESIGN-LB E8). The MoE kernel ends after the experts (counters only), and the
  next stage reads the 11 slot partials directly.
  - **Model.** A consumer kernel is launched after each MoE launch:
    - `stage=cons_y`: out = res + y (the unfused chain);
    - `stage=cons_part`: out = res + the slot / segment sums, after the token's counter (the fused chain).
  - **Numerics.** cons_y is bit for bit bf16(res + y). The fused output is within ulp(y) + ulp(out) of res + y
    everywhere (it skips y's rounding).
  - **Timing.** The fused chain is 0.917-0.96 of the unfused chain (`lb2-r2`, `lb2-r3`). It removes the combine's
    y round trip and its separate pass.

### 7.3 Stale-read mutant at r8 under `cmb=flow`

- **`flowrel`** (the per-token counter released at the segment start, combine units spread over every CTA): caught
  in one launch at every cell, r8 included. It is now a must-fail mutant of every flow build.
- **`flowsc`** (combine loads without sc1 after a plain pretouch of the same lines just before the poll): caught in 1
  of 20 cells. Cross-XCD L2 staleness exists, but the weight stream usually evicts the pretouched lines first.
  - It is report-only.
  - The pretouch-before-poll twin (`pretouch=3`, sc1 kept) passes bit for bit everywhere.

### 7.4 WaveScope and stamps on the cells above 0.85

- **ATT** (`lb2-att7`, MI350X, B5; TP1 r8, TP1 r16 `w`, TP2 r32, TP2 r128). The stall is:
  - VMEM issue, 37-48 % (the CU's load path is full);
  - the gate_up K-half reduction barrier, 11-13 %;
  - the waits on the weight loads, 10-11 %;
  - the staged partial stores, 3-8 %.
- **Stamps** (`lb2-s21`, MI355X).
  - TP1 r8: the pieces are 8 or 4 atoms (4-atom alignment). 146 CTAs carry 8 atoms (0.80 of the span), 73 carry 4
    (0.48), and 37 are idle.
  - TP1 r16: 141 CTAs each stream one whole item and 115 are idle (no split fits in one round).
  - TP2 r32: one whole item per CTA; experts are 0.81 of the span (median wave).
- **Reading.** A piece costs its weights plus a fixed part: x staging, the reductions and requant, and above all its
  partial rows (16 tokens x H f32, the bytes of a whole down chunk).
  - That is why finer pieces lose and why `pdt=bf16` helps.
  - Balancing r16 needs splits at sub-item granularity without the per-piece partial rows.

### 7.5 Round-2 verdicts (MI355X, container, same-job FlyTape brackets)

**Jobs.**
- n01-09: `lb2-f10` (job 68245207) and `lb2-f12` (68245391).
- n07-17: `lb2-f11` (68245217) and `lb2-f13` (68245412).
- All four ran in container mode (the FlyTape image's compiler, section 5), with FlyTape traced before and after.
- Run 1 is experts + combine under random routing. Run 2 is the whole MoE (`entry=router`, exact router inputs);
  its TP2 cells ran again in `lb2-f12` / `f13` after a harness crash in `f10` / `f11`.
- Neighbours (sacct): other users' long-running 1-GPU jobs on both nodes, plus our other lanes' 1-GPU jobs.

**Dispatcher picks are cross-validated.** A cell's pick on one node is the candidate with the lowest geometric mean
(fp8, fp4) of time / B5 measured on the other node. No pick is reported on the data that selected it.
- Run 1 candidates: B5, `w`, the pass-1 pick `k*`, its `pdt=bf16` twin `b*`, and the pass-3 picks `t*a` / `t*b`.
- Run 2 candidates: B5 + router, `r*` (pick + router), `rb*` (+ bf16 partials), `ru*` (the whole-MoE tuner's picks).

**Experts + combine against FlyTape's experts + combine:**

n01-09 (`lb2-f10`):

| cell | pick | pick / FT e+c (pre..post) | B5 / FT e+c | pick / B5 |
|---|---|---|---|---|
| tp1 fp8 r8 | `t18b` | 0.886 (0.885..0.887) | 0.933 | 0.950 |
| tp1 fp4 r8 | `t18b` | 0.877 (0.876..0.878) | 0.939 | 0.933 |
| tp1 fp8 r16 | `t116b` | 0.913 (0.911..0.914) | 0.999 | 0.914 |
| tp1 fp4 r16 | `t116b` | 0.912 (0.910..0.913) | 0.987 | 0.924 |
| tp1 fp8 r32 | `lb2` | 0.889 (0.889..0.889) | 0.889 | 1.000 |
| tp1 fp4 r32 | `lb2` | 0.897 (0.897..0.897) | 0.897 | 1.000 |
| tp1 fp8 r64 | `lb2` | 0.924 (0.920..0.927) | 0.924 | 1.000 |
| tp1 fp4 r64 | `lb2` | 0.921 (0.918..0.925) | 0.921 | 1.000 |
| tp2 fp8 r8 | `b28` | 0.787 (0.775..0.800) | 0.832 | 0.947 |
| tp2 fp4 r8 | `b28` | 0.788 (0.776..0.801) | 0.823 | 0.958 |
| tp2 fp8 r16 | `k216` | 0.824 (0.823..0.825) | 0.854 | 0.965 |
| tp2 fp4 r16 | `k216` | 0.825 (0.824..0.826) | 0.857 | 0.963 |
| tp2 fp8 r32 | `t232b` | 0.907 (0.905..0.910) | 0.945 | 0.960 |
| tp2 fp4 r32 | `t232b` | 0.912 (0.910..0.915) | 0.932 | 0.979 |
| tp2 fp8 r64 | `b264` | 0.815 (0.812..0.817) | 0.853 | 0.955 |
| tp2 fp4 r64 | `b264` | 0.837 (0.834..0.840) | 0.857 | 0.976 |
| tp2 fp8 r128 | `t2128a` | 0.815 (0.813..0.816) | 0.853 | 0.955 |
| tp2 fp4 r128 | `t2128a` | 0.811 (0.809..0.813) | 0.848 | 0.956 |

n07-17 (`lb2-f11`):

| cell | pick | pick / FT e+c (pre..post) | B5 / FT e+c | pick / B5 |
|---|---|---|---|---|
| tp1 fp8 r8 | `t18b` | 0.903 (0.902..0.903) | 0.961 | 0.940 |
| tp1 fp4 r8 | `t18b` | 0.885 (0.884..0.885) | 0.955 | 0.926 |
| tp1 fp8 r16 | `t116a` | 0.931 (0.931..0.931) | 1.016 | 0.916 |
| tp1 fp4 r16 | `t116a` | 0.934 (0.934..0.935) | 1.014 | 0.921 |
| tp1 fp8 r32 | `lb2` | 0.914 (0.913..0.914) | 0.914 | 1.000 |
| tp1 fp4 r32 | `lb2` | 0.916 (0.916..0.917) | 0.916 | 1.000 |
| tp1 fp8 r64 | `b164` | 0.937 (0.936..0.938) | 0.901 | 1.040 |
| tp1 fp4 r64 | `b164` | 0.877 (0.877..0.878) | 0.899 | 0.976 |
| tp2 fp8 r8 | `b28` | 0.785 (0.783..0.787) | 0.829 | 0.947 |
| tp2 fp4 r8 | `b28` | 0.788 (0.787..0.790) | 0.833 | 0.946 |
| tp2 fp8 r16 | `k216` | 0.828 (0.826..0.830) | 0.866 | 0.957 |
| tp2 fp4 r16 | `k216` | 0.822 (0.820..0.824) | 0.867 | 0.948 |
| tp2 fp8 r32 | `t232b` | 0.909 (0.907..0.911) | 0.955 | 0.952 |
| tp2 fp4 r32 | `t232b` | 0.915 (0.913..0.918) | 0.943 | 0.971 |
| tp2 fp8 r64 | `b264` | 0.778 (0.777..0.779) | 0.808 | 0.962 |
| tp2 fp4 r64 | `b264` | 0.771 (0.771..0.772) | 0.795 | 0.970 |
| tp2 fp8 r128 | `t2128a` | 0.802 (0.802..0.803) | 0.846 | 0.948 |
| tp2 fp4 r128 | `t2128a` | 0.796 (0.796..0.797) | 0.838 | 0.951 |

- 0.85 or better on both nodes: TP2 r8 (0.785-0.788), r16 (0.822-0.828), r64 (0.771-0.837), r128 (0.796-0.815).
- Above 0.85: TP1 r8 (0.877-0.903), r16 (0.912-0.934), r32 (0.889-0.916), r64 (0.877-0.937), and TP2 r32
  (0.907-0.915).
- FlyTape's own times moved between jobs. TP1 r32 / r64 FlyTape e+c was 4-6 % faster than in `lb2-f8`, and B5 / FT
  there is 0.89-0.92 against 0.84 then. This is why the brackets are same-job only.
- The venv-tuned picks for TP1 r32 / r64 did not hold under the container compiler; the cross-validation picked B5
  there. The container re-tune (`lb2-c2` / `c3`) found 0.96 of B5 at both cells. `lb2-f14` / `f15` verify those
  picks (section 7.6).

**The whole MoE** (router GEMM + top-10 + experts + combine in one kernel) **against FlyTape's router + experts +
combine** (`*.moe_router` + `moe_up` / `moe_down` + `moe_combine`):

n01-09 (`lb2-f10` run 2, `lb2-f12`):

| cell | pick | whole MoE / FT MoE (pre..post) | B5 with routes given / FT MoE | fused_next chain / (FT MoE + consumer) | fused / unfused chain |
|---|---|---|---|---|---|
| tp1 fp8 r8 | `ru18` | 0.849 (0.848..0.850) | 0.732 | 0.840 | 0.948 |
| tp1 fp4 r8 | `ru18` | 0.825 (0.824..0.826) | 0.722 | 0.789 | 0.931 |
| tp1 fp8 r16 | `ru116` | 0.867 (0.866..0.869) | 0.821 | 0.843 | 0.955 |
| tp1 fp4 r16 | `ru116` | 0.850 (0.849..0.851) | 0.816 | 0.848 | 0.974 |
| tp1 fp8 r32 | `rb132` | 0.890 (0.890..0.890) | 0.790 | - | - |
| tp1 fp4 r32 | `rb132` | 0.857 (0.857..0.857) | 0.790 | - | - |
| tp1 fp8 r64 | `rb164` | 0.868 (0.866..0.871) | 0.833 | - | - |
| tp1 fp4 r64 | `rb164` | 0.871 (0.868..0.873) | 0.840 | - | - |
| tp2 fp8 r8 | `rb28` | 0.763 (0.734..0.767) | 0.608 | - | - |
| tp2 fp4 r8 | `rb28` | 0.754 (0.725..0.758) | 0.598 | - | - |
| tp2 fp8 r16 | `lb2` | 0.813 (0.811..0.817) | 0.660 | - | - |
| tp2 fp4 r16 | `lb2` | 0.813 (0.811..0.818) | 0.657 | - | - |
| tp2 fp8 r32 | `rb232` | 0.859 (0.850..0.867) | 0.742 | 0.820 | 0.951 |
| tp2 fp4 r32 | `rb232` | 0.857 (0.848..0.866) | 0.739 | 0.822 | 0.955 |
| tp2 fp8 r64 | `rb264` | 0.819 (0.815..0.820) | 0.748 | - | - |
| tp2 fp4 r64 | `rb264` | 0.806 (0.802..0.807) | 0.744 | - | - |
| tp2 fp8 r128 | `rb2128` | 0.839 (0.836..0.845) | 0.747 | 0.818 | 0.971 |
| tp2 fp4 r128 | `rb2128` | 0.841 (0.838..0.846) | 0.744 | 0.812 | 0.962 |

n07-17 (`lb2-f11` run 2, `lb2-f13`):

| cell | pick | whole MoE / FT MoE (pre..post) | B5 with routes given / FT MoE | fused_next chain / (FT MoE + consumer) | fused / unfused chain |
|---|---|---|---|---|---|
| tp1 fp8 r8 | `ru18` | 0.861 (0.861..0.861) | 0.748 | 0.855 | 0.947 |
| tp1 fp4 r8 | `ru18` | 0.860 (0.860..0.860) | 0.739 | 0.805 | 0.915 |
| tp1 fp8 r16 | `ru116` | 0.888 (0.888..0.889) | 0.857 | 0.870 | 0.957 |
| tp1 fp4 r16 | `ru116` | 0.876 (0.876..0.877) | 0.851 | 0.871 | 0.974 |
| tp1 fp8 r32 | `rb132` | 0.913 (0.912..0.914) | 0.816 | - | - |
| tp1 fp4 r32 | `rb132` | 0.888 (0.888..0.889) | 0.814 | - | - |
| tp1 fp8 r64 | `rb164` | 0.921 (0.920..0.922) | 0.865 | - | - |
| tp1 fp4 r64 | `rb164` | 0.898 (0.897..0.898) | 0.860 | - | - |
| tp2 fp8 r8 | `rb28` | 0.769 (0.752..0.783) | 0.613 | - | - |
| tp2 fp4 r8 | `rb28` | 0.759 (0.742..0.772) | 0.608 | - | - |
| tp2 fp8 r16 | `rb216` | 0.856 (0.849..0.857) | 0.688 | - | - |
| tp2 fp4 r16 | `rb216` | 0.889 (0.881..0.889) | 0.684 | - | - |
| tp2 fp8 r32 | `rb232` | 0.900 (0.877..0.921) | 0.796 | 0.849 | 0.941 |
| tp2 fp4 r32 | `rb232` | 0.895 (0.873..0.916) | 0.787 | 0.841 | 0.937 |
| tp2 fp8 r64 | `rb264` | 0.823 (0.802..0.851) | 0.751 | - | - |
| tp2 fp4 r64 | `rb264` | 0.813 (0.792..0.841) | 0.746 | - | - |
| tp2 fp8 r128 | `rb2128` | 0.851 (0.831..0.878) | 0.766 | 0.819 | 0.961 |
| tp2 fp4 r128 | `rb2128` | 0.847 (0.827..0.874) | 0.757 | 0.816 | 0.960 |

- The whole MoE is 0.75-0.92 of FlyTape's. The router + top-k inside the kernel adds 11-23 % to B5 with routes given
  (`lb2-r4`); FlyTape's own router phase is 8-28 % of its MoE.
- **`combine=fused_next`.** The fused chain (MoE without a combine, plus the consumer that sums the partials) is
  0.915-0.974 of the unfused chain (MoE with combine, plus the consumer that reads y). Against FlyTape's MoE plus
  the same consumer read, it is 0.79-0.87.
- **Numerics.** Every arm passed. The chains' check compares against the same build's own y: the consumer of y is
  bit for bit, and the fused sums are within ulp(y) + ulp(out). Verified separately in `lb2-n61` (container).

### 7.6 Round-2 final: experts + combine (`lb2-f14` n01-09, `lb2-f15` n07-17)

- **Setup.** Both jobs ran in container mode with FlyTape traced before and after, and the same arms.
  - The candidates are those of section 7.5, plus the container-tuned picks `c132` / `c164` for TP1 r32 / r64.
  - Picks are cross-validated between the two jobs.
- **Numerics.** Every cell passed on both nodes, and no required mutant was missed.
- **Neighbours.** Other users' long-running 1-GPU jobs and our other lanes' 1-GPU jobs.

n01-09 (`lb2-f14`, job 68245913):

| cell | pick | pick / FT e+c (pre..post) | B5 / FT e+c | pick / B5 | B0 (line) / FT | P-L4 / FT | floor / FT |
|---|---|---|---|---|---|---|---|
| tp1 fp8 r8 | `t18b` | 0.881 (0.871..0.892) | 0.943 | 0.934 | 1.334 | 0.998 | 0.550 |
| tp1 fp4 r8 | `t18b` | 0.865 (0.854..0.875) | 0.933 | 0.927 | 1.335 | 0.995 | 0.555 |
| tp1 fp8 r16 | `t116b` | 0.909 (0.901..0.917) | 0.997 | 0.911 | 1.148 | 1.116 | 0.649 |
| tp1 fp4 r16 | `t116b` | 0.910 (0.901..0.918) | 0.997 | 0.912 | 1.143 | 1.065 | 0.641 |
| tp1 fp8 r32 | `c132` | 0.831 (0.830..0.833) | 0.856 | 0.971 | 1.060 | 1.110 | 0.714 |
| tp1 fp4 r32 | `c132` | 0.832 (0.830..0.833) | 0.850 | 0.978 | 1.056 | 1.084 | 0.720 |
| tp1 fp8 r64 | `c164` | 0.827 (0.825..0.829) | 0.852 | 0.971 | 0.983 | 1.139 | 0.717 |
| tp1 fp4 r64 | `c164` | 0.830 (0.828..0.832) | 0.855 | 0.971 | 0.967 | 1.068 | 0.709 |
| tp2 fp8 r8 | `b28` | 0.797 (0.786..0.808) | 0.842 | 0.946 | 1.473 | 1.031 | 0.433 |
| tp2 fp4 r8 | `b28` | 0.807 (0.796..0.818) | 0.850 | 0.949 | 1.472 | 1.013 | 0.449 |
| tp2 fp8 r16 | `k216` | 0.828 (0.827..0.829) | 0.866 | 0.957 | 1.187 | 1.073 | 0.506 |
| tp2 fp4 r16 | `k216` | 0.829 (0.827..0.830) | 0.871 | 0.951 | 1.188 | 1.063 | 0.508 |
| tp2 fp8 r32 | `t232b` | 0.953 (0.951..0.954) | 0.991 | 0.961 | 1.344 | 1.285 | 0.718 |
| tp2 fp4 r32 | `t232b` | 0.952 (0.951..0.954) | 0.988 | 0.964 | 1.353 | 1.277 | 0.717 |
| tp2 fp8 r64 | `b264` | 0.820 (0.817..0.823) | 0.846 | 0.969 | 1.061 | 1.097 | 0.644 |
| tp2 fp4 r64 | `b264` | 0.814 (0.811..0.817) | 0.837 | 0.972 | 1.052 | 1.049 | 0.640 |
| tp2 fp8 r128 | `t2128a` | 0.855 (0.851..0.858) | 0.923 | 0.926 | 1.221 | 1.159 | 0.714 |
| tp2 fp4 r128 | `t2128a` | 0.852 (0.848..0.856) | 0.913 | 0.934 | 1.223 | 1.153 | 0.711 |

n07-17 (`lb2-f15`, job 68246039):

| cell | pick | pick / FT e+c (pre..post) | B5 / FT e+c | pick / B5 | B0 (line) / FT | P-L4 / FT | floor / FT |
|---|---|---|---|---|---|---|---|
| tp1 fp8 r8 | `t18b` | 0.907 (0.907..0.908) | 0.957 | 0.948 | 1.351 | 1.018 | 0.563 |
| tp1 fp4 r8 | `t18b` | 0.878 (0.877..0.879) | 0.949 | 0.925 | 1.356 | 1.012 | 0.565 |
| tp1 fp8 r16 | `t116b` | 0.906 (0.904..0.909) | 0.996 | 0.910 | 1.148 | 1.124 | 0.646 |
| tp1 fp4 r16 | `t116b` | 0.910 (0.908..0.912) | 0.992 | 0.917 | 1.148 | 1.082 | 0.649 |
| tp1 fp8 r32 | `c132` | 0.851 (0.846..0.857) | 0.888 | 0.959 | 1.111 | 1.125 | 0.718 |
| tp1 fp4 r32 | `c132` | 0.854 (0.849..0.859) | 0.885 | 0.965 | 1.105 | 1.094 | 0.718 |
| tp1 fp8 r64 | `c164` | 0.862 (0.851..0.874) | 0.901 | 0.957 | 1.043 | 1.162 | 0.729 |
| tp1 fp4 r64 | `c164` | 0.864 (0.852..0.876) | 0.899 | 0.961 | 1.026 | 1.086 | 0.719 |
| tp2 fp8 r8 | `b28` | 0.803 (0.802..0.803) | 0.841 | 0.954 | 1.475 | 1.036 | 0.434 |
| tp2 fp4 r8 | `b28` | 0.805 (0.804..0.806) | 0.844 | 0.954 | 1.458 | 1.006 | 0.440 |
| tp2 fp8 r16 | `k216` | 0.823 (0.822..0.824) | 0.862 | 0.955 | 1.189 | 1.078 | 0.502 |
| tp2 fp4 r16 | `k216` | 0.828 (0.827..0.829) | 0.870 | 0.952 | 1.197 | 1.061 | 0.499 |
| tp2 fp8 r32 | `t232a` | 0.891 (0.889..0.894) | 0.927 | 0.961 | 1.262 | 1.213 | 0.674 |
| tp2 fp4 r32 | `t232a` | 0.894 (0.891..0.896) | 0.925 | 0.966 | 1.281 | 1.204 | 0.675 |
| tp2 fp8 r64 | `b264` | 0.813 (0.812..0.814) | 0.847 | 0.960 | 1.090 | 1.064 | 0.625 |
| tp2 fp4 r64 | `b264` | 0.813 (0.812..0.813) | 0.840 | 0.967 | 1.093 | 1.037 | 0.617 |
| tp2 fp8 r128 | `t2128a` | 0.797 (0.797..0.798) | 0.840 | 0.950 | 1.155 | 1.089 | 0.648 |
| tp2 fp4 r128 | `t2128a` | 0.794 (0.794..0.795) | 0.834 | 0.953 | 1.166 | 1.059 | 0.646 |

- The dispatcher picks are 0.91-0.98 of B5 in every cell on both nodes. The ratios to B5 reproduce between the two
  nodes and between `f10` / `f11` and `f14` / `f15`.
- **0.85 or better** (pick / FT e+c):
  - on both nodes: TP2 r8 (0.797-0.807), r16 (0.823-0.829) and r64 (0.813-0.820);
  - on n01-09 only: TP1 r32 (0.831-0.832) and r64 (0.827-0.830); on n07-17 they are 0.851-0.854 and 0.862-0.864;
  - TP2 r128: 0.794-0.797 on n07-17; 0.852-0.855 on n01-09, where FlyTape's experts + combine ran 12 % faster than in
    the n07-17 job (ours 3 %).
- **Not met:**
  - TP1 r8: 0.865-0.907.
  - TP1 r16: 0.906-0.910.
  - TP2 r32: 0.891-0.894 on n07-17, and 0.952-0.953 on n01-09, where FlyTape ran 8 % faster than on n07-17
    (B5 / FT 0.99 against 0.93-0.95 in the other jobs).
- **FlyTape's spread.** FlyTape's own experts + combine time moves by up to 12 % between jobs, which moves
  every ratio by as much. Verdicts stay same-job only.

### 7.7 Round 3 (in progress): the cells above 0.85

All runs below use the container compiler on MI355X. Every numerics check passed and every required mutant was
caught.

- **`rsync=local`: every CTA routes every token itself** (no route hand-off; wave w takes tokens w, w + 8, ...).
  - It lost everywhere (`lb2-f16`, n07-17, whole MoE): the tag hand-off runs at 0.73-0.84 of `local` on B5 +
    router, and at 0.69-0.88 on the picks (each tag arm against its local twin).
  - Recomputing the top-k in all 256 CTAs (logit reads and top-10 loops) costs more than the hand-off it removes.
  - Its must-fail mutant `rskip` (the CTA's own pairs never stored, so the sort reads stale pairs) is caught at every
    cell.
  - Kept as a default-off knob in the tuner's space.
- **`pfh`: CTAs without a unit prefetch a busy unit's weights into L2 / MALL** (TP1 r8 / r16 leave 37 / 115 CTAs
  idle).
  - `pfh=1` (the down block) lost: 1.21-1.22 of B5 at TP1 r8, 1.16-1.30 of B5 on the r8 pick (the pick alone:
    0.93-0.99), tie at r16 / r32 / r64
    (`lb2-p3`, n01-09).
  - The extra loads compete with the streams; the busy CTAs gain nothing from MALL hits.
  - `pfh=2` is fixed to compile (duplicate loop names) but not yet measured.
- **`pfh=2`** (the down block plus the back half of the gate_up block): 1.18-2.36 of B5 and of the picks (`lb2-p4`,
  n07-17). Prefetching another CTA's weights loses badly; `pfh` stays a default-off loser.
- **Where the remaining gap is** (`lb2-p4` depth sweep and stamps on the TP2 r32 pick, container). This corrects the
  earlier round-3 diagnosis, which proposed LDS-DMA streaming and a CTA-pair K split.
  - **TP2 r32 is not bound by bytes in flight.**
    - Gate_up time per atom is flat over `depth` 4 / 8 / 10 (0.0518 / 0.0489 / 0.0521 of the span), and the arms are
      within 3 % (d4 0.97-0.99, d6 0.955-0.96, d8 0.957-0.959, d10 0.975-0.99 of B5).
    - The per-CTA stream rate times the 243 busy CTAs is about the streamer floor's aggregate rate. The gap is the
      per-item fixed phases and the slowest-CTA tail: the experts phase ends at 0.82-0.84 of the span on the median
      wave and 0.92 on the slowest.
    - LDS-DMA weight streaming would not add in-flight capacity anyway: a CU's VGPR file (512 KiB) is larger than its
      LDS (160 KiB), and LDS also holds x, h, the reductions and the store stage. **Not built: evidence `lb2-p4`**
      (revisit on a future base, decision 15).
  - **TP1 r8 / r16: a CTA-pair K split cannot shorten the critical path.**
    - A paired half item is 1.2 MB per CTA at TP1. Today's r8 pieces are at most 8 atoms (about 0.96 MB).
    - At r16, 2 x 141 = 282 half items exceed the 256 CTAs, so 26 CTAs would carry two halves.
    - It keeps each CTA's fixed part (x staging, first-load latency, the h-complete barrier, the store drain: about
      0.16 of the span at r8) and adds an L2 exchange. **Not built: evidence `lb2-p4`** (revisit on a future base,
    decision 15).
  - **What is left on these cells** is per-CTA fixed latency and tail variance, not bytes. The candidates:
    - overlap the first weight loads with the sort (a unit's expert is known before the full item table);
    - shorten the h-complete -> down hand-off;
    - in the layer, `combine=fused_next`, which takes the combine tail off the MoE (0.94-0.97 of B5 at TP2 r32,
      consumer included).
- **Per-segment phases** (`lb2-p4` stamps, wave 0, median CTA, fractions of the span):

  | cell | first atom reduced | atoms | h-complete hand-off | down operands | down | after the slowest segment |
  |---|---|---|---|---|---|---|
  | TP1 r8 | 0.038-0.039 | 0.50-0.53 | 0.020-0.021 | 0.003-0.004 | 0.25-0.26 | about 0.07-0.11 |
  | TP1 r16 | 0.033-0.035 | 0.51-0.53 | 0.017-0.018 | 0.003 | 0.26-0.27 | about 0.08 |
  | TP2 r32 | 0.057-0.058 | 0.42-0.45 | 0.012-0.013 | 0.004 | 0.26 | about 0.08-0.10 |

  The last column runs from the slowest CTA's segment end to the span's end (the sort's 0.02-0.03 before the first
  segment taken off): the flow drain, the counter bump, the
  combine poll and the combine's reads.
- **Built on that breakdown and measured** (container; numerics pass, every required mutant caught with each new
  knob as the base):
  - **`ehead`** issues the first unit's weight head as soon as the item table exists, before the sort's token lists
    and the x staging. It ties on B5 (0.995-1.011) and is 0.3-1.5 % slower on the picks (`lb2-p5`, n01-09). The
    first-atom phase is load latency, not issue order.
  - **`cspec`** has the combine load segments 0 and 1 of every slot with the descriptor (one memory round trip
    less). It ties at r8-r32 and loses at r64 / r128 (1.02-1.06) on B5; on the picks it loses 3-25 % (`lb2-p6`,
    n07-17). The 22 preloaded rows per lane cost more than the round trip.
  - **Not built: the h-complete hand-off** (a per-down-chunk h readiness instead of the CTA barrier). It is
    1.2-2.1 % of the span; the bound is below the remaining gap.
- **Tuner lookback** on the new knobs (decision 15): `lb2-c6` / `c7` (container) from the picks of the three cells,
  with `ehead`, `cspec`, `pfh` and `rsync` in the space.
  - TP1 r16 and TP2 r32: ties. The picks stay.
  - TP1 r8: `pc=0, lpf=2, ab=2` on the pick, 0.978 fp8 (WIN), 0.999 fp4 (TIE). It is being confirmed on both nodes
    (`lb2-f17` / `f18`).
  - None of the new knobs enters a pick.
- **`combine=fused_next` hand-off.** For WP-L2b / WP-L3: `COORD-moe-fused-next.md` (what the consumer reads, the
  layout, the acquire / reset / ordering rules, and the `flowrel` / `flow0` mutants).
