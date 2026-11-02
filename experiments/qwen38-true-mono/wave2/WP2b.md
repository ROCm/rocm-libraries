# WP2b: the GEMV-family stages B1, C1, Q1, P1 as rocKE stage functions (wave 2, W8)

D0 section 5 wave 2 (WP2b: "GEMV family B1/C1/Q1/P1 per DESIGN 5, all served widths"), PLAN decision 16 (reverse-engineer
FlyDSL's code, reach parity by construction, then crush it with the parametric search), decisions 13 / 15 (lookback with
the losers kept, rejections "on base X", re-tests on the new base). Ratios, shares and counts only. Absolute times:
`hackathon-prep/private-perf/qwen38-true-mono/wave2/WP2b.md`. Below 1 is faster.

Status: **done. Built at every served width (3 kinds x 16 widths = 48 cells), bitwise everywhere, and faster than the
FlyDSL stages on the tuned stack in all 48 cells: 95 of 96 (cell, node) verdicts are DESIGN 9.1 WINs, the 96th a TIE at
0.9886 caused by its own control (section 6.4); no loss.** The 1:1 port alone (the FlyDSL instruction choices) loses
0.2-2.7 % of the layer in every cell; the reasons are in section 3. The adopted ports take 0.4-4.2 % off the whole delta
layer, 0.4-4.8 % off the qsa layer and 0.9-4.8 % off the delta_ple layer (section 6.5), cut the GEMV stages' MFMA-chain
time to 0.46-0.74 of FlyDSL's (P1: 0.86-0.90; WaveScope, section 6.6), and move the quick end-to-end by +1.4 % req tok/s
at K=3 and +0.9 % at K=0 (ms per token 0.991) (section 7).

## 1. What was built

| piece | where |
|---|---|
| the emitter: B1 (QKVZ + conv + SiLU + conv state, verify window, grouped history), C1 (out_proj + 4-wave reduce + HC write `w` / `whs`, PLE residual), Q1 (QSA projection), P1 (embedding MXFP8 pack, key chain, 4 value chains), narrow `r1-r4`, `r2s-r4s` and grouped / wide `r4g2 .. r16g4`; `GemvKnobs` (section 4); `GemvShape`; `write_gemv_functions`; CLI | `copy-rocm-lib/.../qwen38_mono/stages2/gemv.py` (new package `stages2/`) |
| CPU tests (96): knob validation, every shape lowers, the parity build's MFMA operand formats and op_sel bytes equal FlyDSL's, the asm build ties every accumulator, every mutant changes the IR, refusals, NUMA sidecar, the B3 prologue host | `copy-rocm-lib/.../library/tests/test_qwen38_wp2b_gemv.py` |
| in-situ harness: `check` (bitwise vs the plain mono + whole scratch vs the FlyDSL pick, must-fail mutants), `time` (DESIGN 9.1), `stamps` (stage timelines), `launch` (WaveScope) | `hackathon-prep/qwen38/truemono/WP2b/wp2b_insitu.py` |
| tuner backend + driver (signed lookback, order restarts, TPE; GEMV-only or joint with the layer's MoE / B3 / schedule knobs) | `WP2b/wp2b_tune.py` |
| served-stack generator (functions next to a stack's own; its numa lever, its pause, the B3 prologue host) and quick-e2e arms | `WP2b/wp2b_gen.py`, `WP2b/arms_wp2b.sh`, `WP2b/stage_quick_wp2b.sh` |
| reverse toolkit jobs: layer dumps with / without the functions, per-stage split, diff, MFMA-chain and hazard audit; container codegen repro | `WP2b/wp2b_rv_job.sh`, `WP2b/wp2b_isa.py`, `WP2b/wp2b_llc_job.sh` |
| WaveScope analysis: per-region MFMA counts and shares, per-visit MFMA-chain spans from the per-wave timelines | `WP2b/wp2b_att_regions.py`, `WP2b/wp2b_att_chain.py` |
| picks: the six tuned cells, the 47-cell served set, transfer / tuned grouped arms, confirmation arms | `WP2b/picks_wp2b.json`, `WP2b/picks_all.json`, `WP2b/xfer_picks.json`, `WP2b/tuned_grouped.json`, `WP2b/confirm*.json` |
| job driver and staging (no local deletion: every pack goes to a new `stage_build/<timestamp>` dir) | `WP2b/wp2b_job.sh`, `WP2b/sync_wp2b.sh` |

No existing rocKE file was edited: `stages2/` is new and nothing outside it imports it, so every existing build is
byte-identical by construction (WP2b's files: `stages2/__init__.py`, `stages2/gemv.py`, the new test file; the CPU test
also checks that the `stage_fn` registry is unchanged). No ATOM-truemono file was edited: the functions link through the
existing `ATOM_Q38_EXTERN_STAGES` hook (any stage name; `{kind}` path templates select per-kind files).

## 2. Method as run (PLAN decision 16)

1. **Dump FlyDSL's code at every served width.** The reverse toolkit (`reverse/FLYDSL-STAGES.md`) dumped the served stack
   (picksI + qdense; rocKE MoE / B3 linked) with and without the WP2b functions, debug and plain, per kind, at `r1 r2 r3 r4
   r2s r3s r4s r4g2 r8g4 r16g4` (CPU jobs `w8-wp2b-rv2 .. rv6b`, 210 code objects in the last one; 86 of 90 debug builds
   have the plain build's `.text`; the four that differ are the known non-reproducible builds, alola rule 137).
2. **Port, then prove parity by construction and by bits.** The emitter mirrors FlyDSL's operations and order (same MFMA
   intrinsic, operand formats, op_sel bytes, chain order, the same bf16 / f32 rounding points, the same DPP lane maps,
   `v_maximum3_f32` for FlyDSL's `llvm.maximum`), checked statically (CPU tests) and per stage in the ISA (section 3), and
   bit for bit in situ (section 5).
3. **Every hard-coded choice and every catalogued inefficiency is a knob** (section 4).
4. **Climb** with the mono tuner (`SignedLookback` + order restarts + TPE; losers kept in the pools) at the r1 / r4s cells,
   transfer those picks to every other width, and climb again where a transfer did not win or the width is the widest
   served (12 of 48 cells climbed; section 6).
5. **Time in situ** on the tuned stack's layer (`truemono-quick/lookback/hyb_layer`: 16 cold weight copies, 96
   launches per graph, shuffled rounds, a second graph and a recompiled build as controls; DESIGN 9.1 win rule), both
   rocke nodes.
6. **Stamps / WaveScope** before and after (sections 6.1, 6.6); quick end-to-end (section 7).

## 3. Parity by construction: what the FlyDSL code and the port compile to

Per stage, at the served widths (job `w8-wp2b-rv6b`). Columns: instructions / MFMAs whose destination equals their
srcC (tied accumulator) / MFMAs right after `s_waitcnt lgkmcnt(0)` / MFMAs with a scale VGPR written since the previous
MFMA / `s_nop` wait states inside the MFMA chain / static hazard-audit violations. `stk` = FlyDSL stage in the served
stack, `parity` = the port at the FlyDSL instruction choices, `tuned` = the port with `mfma=asm,asm_lead=-1`
(+ `mb_impl=ir,w_cpol=2`).

| stage, kind, width | FlyDSL (stk) | parity port | tuned port |
|---|---|---|---|
| B1 delta r1 | 520 / 19/20 / 20 / 10 / 0 / 0 | 524 / 1/20 / 20 / 10 / 102 / 0 | 528 / 20/20 / 4 / 2 / 0 / 0 |
| B1 delta r4s | 585 / 19/20 / 20 / 10 / 0 / 0 | 572 / 12/20 / 20 / 10 / 36 / 0 | 581 / 20/20 / 4 / 2 / 0 / 0 |
| B1 delta r4g2 | 545 / 19/20 / 20 / 10 / 0 / 0 | 531 / 12/20 / 20 / 10 / 44 / 0 | 537 / 20/20 / 4 / 2 / 0 / 0 |
| B1 delta r8g4 | 635 / 19/20 / 20 / 10 / 0 / 0 | 604 / 12/20 / 20 / 10 / 36 / 0 | 601 / 20/20 / 4 / 2 / 0 / 0 |
| B1 delta r16g4 | 1151 / 19/20 / 20 / 10 / 0 / 0 | 996 / 12/20 / 20 / 10 / 44 / 0 | 992 / 20/20 / 4 / 2 / 0 / 0 |
| C1 delta r1 | 414 / 11/12 / 12 / 12 / 5 / 0 | 395 / 11/12 / 12 / 8 / 3 / 0 | 434 / 12/12 / 3 / 2 / 22 / 0 |
| C1 delta r4s | 542 / 11/12 / 12 / 12 / 7 / 0 | 524 / 11/12 / 12 / 12 / 9 / 0 | 536 / 12/12 / 2 / 2 / 22 / 0 |
| C1 delta r16g4 | 884 / 11/12 / 12 / 12 / 4 / 0 | 783 / 11/12 / 12 / 12 / 10 / 0 | 809 / 12/12 / 3 / 2 / 22 / 0 |
| Q1 qsa r1 | 359 / 19/20 / 20 / 10 / 0 / 0 | 355 / 5/20 / 20 / 10 / 91 / 0 | 370 / 20/20 / 4 / 2 / 0 / 0 |
| Q1 qsa r4s | 433 / 19/20 / 20 / 10 / 0 / 0 | 417 / 12/20 / 20 / 10 / 35 / 0 | 439 / 20/20 / 4 / 2 / 0 / 0 |
| Q1 qsa r8g4 | 472 / 19/20 / 20 / 10 / 0 / 0 | 446 / 7/20 / 20 / 10 / 73 / 0 | 454 / 20/20 / 4 / 2 / 0 / 0 |
| P1 delta_ple r1 | 450 / 23/40 / 23 / 11 / 37 / 0 | 435 / 33/40 / 23 / 13 / 37 / 0 | 466 / 40/40 / 6 / 3 / 25 / 0 |
| P1 delta_ple r4s | 765 / 33/40 / 23 / 11 / 30 / 0 | 713 / 33/40 / 24 / 13 / 31 / 0 | 756 / 40/40 / 7 / 4 / 33 / 0 |
| P1 delta_ple r16g4 | 1896 / 35/40 / 25 / 12 / 46 / 0 | 1636 / 35/40 / 25 / 13 / 31 / 0 | 1758 / 40/40 / 7 / 4 / 33 / 0 |

(Every width and kind: `WP2b/local/parity_table.md`; raw: `logs/q38/w8-wp2b-rv6b/`.)

The adopted set as linked (`WP2b/picks_all.json`, all 48 cells, 112 stage functions; job `w8-wp2b-rv7`, raw
`logs/q38/w8-wp2b-rv7/`, metrics `WP2b/local/rv7_isa.txt`): 0 static hazard violations in all 112; every B1 and Q1 chain
tied (48 of 48; MFMAs after `lgkmcnt(0)` 1-4 of 20, scale rewrites 1-2), C1 tied in the 41 cells whose pick has
`mfma=asm` (the other 7 keep the intrinsic form, the knob the climbs chose there), P1 at its parity form (`oob=split` and
the pause only). 46 of 48 debug builds have the plain build's `.text` (the 2 others: alola rule 137).

What the diff showed, and what each finding became:

1. **The intrinsic chain is not tied in the linked layer.** Same IR, same intrinsic as FlyDSL; inside the mono layer the
   register allocator gave B1 / Q1's MFMA destination the dying weight register (3-address `_vgprcd_e64` form; the
   two-address pass never formed the tied variant, confirmed with the container's own LLVM in `wp2b_llc_job.sh`), so
   each dependent MFMA waited 6-7 wait states (91-114 per 20-step chain at r1). FlyDSL's chain is tied by the same
   allocator's luck. Knob `mfma=asm`: the same instruction as inline asm with the accumulator tied (`"=v,v,v,0,v,v"`);
   it chains back to back with no padding inside the chain. The asm is invisible to LLVM's hazard recognizer, so the
   emitter adds `asm_lead` wait states before an MFMA whose source a VALU just wrote (2; `asm_lead=-1` = only where the
   emitter knows of such a source: the zero accumulator, a packed scale at its first use) and `asm_tail` after the
   chain (>= 11), and `wp2b_isa.py` audits every build's ISA: 0 violations in all 210 dumps. Lower values are refused by
   the knob's validation, never searched.
2. **The NUMA header load drained every load in flight at each stage's entry** (the B3 / MoE ports have the same
   pattern): a vector load + `readfirstlane` costs an `s_waitcnt vmcnt(0)` before the stage's first instruction. Knob
   `hdr`: `smem` (a constant-space scalar load, the default) / `vmem`.
3. **The catalogue's MFMA findings are gone in the tuned port**: MFMAs after `lgkmcnt(0)` 20 -> 4 (B1 / Q1) and 12 -> 2-3
   (C1); scale-VGPR rewrites 10 -> 2 (B1 / Q1) and 12 -> 2 (C1, `bsop`: the weight scale byte of four steps in one dword,
   picked by op_sel, instead of FlyDSL's shift per step) and 11 -> 3-4 (P1).
4. **P1's `fx.max` is `llvm.maximum.f32`**, which gfx950 lowers to `v_maximum3_f32` (24 per CTA at r4s). rocKE has no
   such op; the first port emulated it (152 VALU). Knob `pmax=max3` (default) emits FlyDSL's own instruction.
5. **Polls.** FlyDSL's polls are compiler-visible loops (first iteration peeled, counted waits); rocKE's default `asm` poll
   is an opaque loop with `vmcnt(0)` per retry. Knob `mb_impl`: `asm` / `ir` / `irf` (the compiler-visible
   `tagged_poll` loop, with the reload fix), with every `pause` the ATOM lever has (`s_sleep N` too).

## 4. Knobs (`GemvKnobs`, defaults = FlyDSL's choices; every one keeps the bits)

| knob | values (default first) | what |
|---|---|---|
| `mb_impl` | asm, ir, irf | poll / put implementation |
| `pause` | the cell's ATOM `pause` lever, -1, 0, 1, 2, 4, 8, 16 | poll retry: none / `s_nop 0` / `s_sleep N` |
| `numa`, `numa_hdr` | the stack's (fixed) | ATOM's numa=1 mailbox area |
| `w_cpol`, `s_cpol` | 0, 2 (nt), 16 (sc1), 18, 1, 3, 19 | weight / weight-scale cache bits (FlyDSL: plain) |
| `oob` | load, split | waves 4-7: out-of-range loads (FlyDSL) / no loads (the body twice behind a wave-uniform branch) |
| `asop` | 0, 1 | A-operand E8M0 scales of four steps packed in one dword, op_sel |
| `bsop` (C1) | 0, 1 | the weight scales likewise (FlyDSL: one shifted dword per MFMA) |
| `ra` | 0..4 | LDS operand read-ahead (the IR order; LLVM reorders the op form, the asm form follows it) |
| `sgb` | none, mfma1, mfma2 | `sched_group_barrier` interleave of the chain |
| `mfma` | op, asm | intrinsic / inline asm with the accumulator tied |
| `asm_lead`, `asm_tail` | 2 (-1 auto, 3, 4); 16 (12, 20) | hazard wait states of the asm form (lower values refused) |
| `barrier` | wg, lds | FlyDSL's release / acquire fences around `s_barrier`, or `lgkmcnt(0)` + `s_barrier` |
| `hdr` | smem, vmem | the NUMA header load |
| `pmax` (P1) | max3, emu | FlyDSL's `v_maximum3_f32`, or the select emulation |
| `epi` (B1, Q1 narrow) | lds, reg | the epilogue from LDS after a CTA barrier (FlyDSL), or from the MFMA registers by the MFMA waves (no LDS, no barrier; each wave owns 16 channels of every row) |
| `cprio` (not C1) | 0..3 | `s_setprio` around the MFMA chain |
| `host_b3pre` (B1) | the stack's B3 spec | the B3 `pre=dma` prologue the FlyDSL B1 calls, emitted by the rocKE B1 |
| `conv_wait` (B1) | vm0 | the verify window's `vmcnt(0)` (a correctness wait, not searched) |
| `mutate` | test only | `put_bit`, `chain_drop`, `a_kit`, `opsel`, `conv_tap`, `resid`, `vsum`, `row`, `chain_swap` |

## 5. Numerics (DESIGN 7.8 gate 1, in situ on the stack's layer)

Per arm, after one launch from a restored snapshot: the layer output and every state the kind writes equal to the plain
FlyDSL mono, twice, nonzero; and the whole scratch (every mailbox of every stage, both NUMA copies, the stack's private
regions) equal to the FlyDSL pick's.

| job | cells | arms bitwise | required mutants caught |
|---|---|---|---|
| chk5 (n01-09) | all 21 narrow cells (`r1-r4`, `r2s-r4s` x delta / qsa / delta_ple) | 154 of 154 (parity + 4 knob sets) | 273 of 273 |
| chk4 (n01-09) | 6 served cells | 75 of 75 (10 knob sets incl. asm, asop / bsop, split, ir, cpol, barrier, sgb) | all |
| chkG1 / chkG2 | all 27 grouped / wide cells (`r4g2, r6g2, r8g2, r6g3, r9g3, r12g3, r8g4, r12g4, r16g4` x 3 kinds) | all (chkG1's three `r4g2` failures were the harness's, below; chkG2 re-ran them: bitwise) | all |
| every timed arm (sections 6.2-6.5) and every tuner evaluation | all 48 cells | gated before timing and post-checked after (`POST_CHECK True` in every job) | |

The order-only mutants (`chain_swap`, `vsum`: a fp32 reassociation) are reported, not required: an output moves only where
a 1-ulp fp32 change crosses a bf16 rounding boundary (all 56 were invisible in chk5, and all in chkG2). The order itself
is proven by construction (section 3: the same MFMA sequence on one accumulator).

Harness findings on the way (both fixed in `wp2b_insitu.py`, recorded as alola rule 212):

- The rigs leave the PLE embedding row at zero, so P1's outputs did not depend on its weights: every P1 mutant passed the
  first gate. The harness fills the row with seeded random data and recomputes the reference.
- The grouped rig at `r4g2` keeps a scratch too small for ATOM's NUMA area: arms built right after a mutant read its
  stale same-tag pairs and failed; with the narrow rig's scratch and calibration every `r4g2` arm is bitwise.

## 6. Timing in situ (DESIGN 9.1; layer time over the tuned stack's FlyDSL pick, same job)

Every number is the layer time of an arm over the FlyDSL pick's in the same job (`truemono-quick/lookback/hyb_layer`: 16
cold weight copies, 96 launches per graph, shuffled rounds, 24 rounds unless stated; a second graph and a recompiled
build of the pick as controls). WIN = median below both controls' minimum by 0.002 and below 1 in at least 3/4 of the
rounds (DESIGN 9.1). Every arm passed the bitwise gate before it was timed and the post-check after.

### 6.1 The 1:1 port and the first fixes (jobs `w8-wp2b-t1` n07-17, `t2` n01-09)

| cell | parity port (all stages) | B1 asm tied | C1 parity | C1 irf + asm | controls (min-max) |
|---|---|---|---|---|---|
| delta r4s | 1.0111 LOSS | 0.9921 WIN (irf) / 1.0023 (asm poll) | 1.0100 LOSS | 1.0077 LOSS | 0.9979-1.0018 |
| delta r1 | 1.0203 LOSS | 0.9958 (24/24, inside the margin) | 1.0073 LOSS | 1.0947 LOSS | 0.9973-1.0048 |
| qsa r4s | 1.0112 LOSS | Q1: 1.0016 | 1.0081 LOSS | 1.0004 | 0.9985-1.0026 |
| delta_ple r4s | 1.0234 LOSS | 0.9950 WIN | 1.0093 LOSS | 1.0133 LOSS | 0.9981-1.0037 |

The C1 port alone loses about 1 % in every cell; with stamps (`w8-wp2b-s1`) its increment grows by 13 % and M1 / M2 wait
longer at r4s with the asm poll, and not with `irf` (stamped builds bias time: diagnostics only). The `irf` C1 at delta r1
was a 9.5 % outlier in one unstamped build (not in the stamped one, not elsewhere): treated as a code-layout effect of that
build, not a property of the poll; it is a loser kept in the pools. With stamps, B1's critical-path increment fell to
0.973 of FlyDSL's at r1 and r4s with the first B1 asm port (`w8-wp2b-s1`; FlyDSL shares of the layer: delta r1 B1 0.13,
C1 0.06; delta r4s B1 0.10, C1 0.05; qsa r4s Q1 0.05, C1 0.03).

### 6.2 The six tuned cells (r1 / r4s; tuner picks)

Climbs: `w8-wp2b-tuA .. tuF` (GEMV-only signed lookback from the parity port, then from the round-1 picks; up to 150
evaluations per cell per round; confirmation 16 rounds in the tuning job). Adopted per-stage knobs: `WP2b/picks_wp2b.json`.

| cell | adopted pick (per-stage knobs) | tuning job (n01-09, 16 rounds) |
|---|---|---|
| delta r4s | B1 `mfma=asm,asm_lead=-1,asm_tail=20,mb_impl=ir,w_cpol=2,s_cpol=2,barrier=lds,cprio=3`; C1 `mfma=asm,asm_lead=3,asm_tail=20,mb_impl=irf,bsop=1,oob=split,ra=4,w_cpol=2,s_cpol=2,barrier=lds` | 0.9728 WIN |
| delta r1 | B1 `mfma=asm,asm_lead=-1,asm_tail=12,mb_impl=irf,oob=split,ra=3,w_cpol=3,s_cpol=2,barrier=lds,hdr=vmem`; C1 `mb_impl=ir,asop=1,ra=3,w_cpol=19,barrier=lds,pause=1` | 0.9563 WIN (with `asm_lead=0`, see below) |
| delta_ple r1 | B1 `mfma=asm,asm_lead=-1,asm_tail=12,cprio=3,oob=split,w_cpol=2,s_cpol=2,barrier=lds`; C1 as delta r4s with `asm_lead=-1`; P1 parity | 0.9565 WIN |
| delta_ple r4s | B1 `mfma=asm,asm_tail=20,asop=1,mb_impl=ir,oob=split,pause=8,ra=1,w_cpol=2,s_cpol=18,barrier=lds,cprio=3`; C1 as delta r4s with `pause=16`; P1 `oob=split,pause=4` | 0.9863 WIN (round 2) |
| qsa r4s | C1 `mfma=asm,mb_impl=ir,bsop=1,oob=split,pause=2,w_cpol=19`; Q1 `mfma=asm,asm_lead=-1,asm_tail=12,epi=reg,mb_impl=irf,asop=1,oob=split,pause=8,ra=3,w_cpol=2,s_cpol=18,sgb=mfma2` | 0.9908 WIN (round 2) |
| qsa r1 | C1 `mfma=asm,mb_impl=ir,oob=split,pause=0,ra=4,w_cpol=19`; Q1 `mfma=asm,mb_impl=irf,oob=split,pause=0,w_cpol=19,s_cpol=18` | 0.9618 WIN (round 2; round 1: 0.9789) |

The tuner's delta r1 pick had `asm_lead=0` on B1 (no lead even after the zero accumulator). It passed the bitwise gate, but
the static rule says a VALU write needs 2 wait states before an XDL read; values below 2 are now refused, and the adopted
delta r1 arm is the same pick with `asm_lead=-1`. Its measurements in 6.5 are of the adopted arm.

### 6.3 The other narrow widths (r2, r3, r4, r2s, r3s): the six picks transferred

No climb per width: the r1 and the r4s pick of the kind, each timed as an arm at every other narrow width (`w8-wp2b-nar1`,
n07-17), the faster one adopted (`WP2b/xfer_picks.json`), then confirmed on the other node (`w8-wp2b-n2n01`, n01-09).
Both transfers WIN in 27 of 30 (cell, pick) pairs on n07-17 (the other 3: TIE); the parity port loses in all 15 cells.

### 6.4 Grouped / wide widths (r4g2 .. r16g4): transfer, then climbs at the widest

The same transfer at the 27 grouped cells (`w8-wp2b-grp1`, `grp2`, n01-09; confirmed on n07-17 by `w8-wp2b-g2n07`), then
GEMV-only climbs from the transfer at the four widest served cells (`w8-wp2b-tuW`, 96 evaluations each, n01-09;
confirmed on n07-17 by `w8-wp2b-c2n07` and on n01-09 by `c3n01`):

| cell | transfer arm (n01-09) | climb pick (tuning job, 16 rounds, n01-09) | n07-17: climb / transfer | adopted |
|---|---|---|---|---|
| delta r16g4 | r1 pick: 0.9913 | 0.9885 WIN | 0.9854 / 0.9876 | climb |
| delta r8g4 | r1 pick: 0.9890 | 0.9872 WIN | 0.9874 / 0.9899 | climb |
| delta_ple r16g4 | r4s pick: 0.9878 | 0.9813 WIN | 0.9884 / 0.9856 | climb (mean of both nodes) |
| qsa r16g4 | r4s pick: 0.9958 | 0.9939 TIE | 0.9963 / 0.9970 | climb (mean of both nodes) |

Three cells did not win with a transfer on both nodes at the first try. `qsa r9g3`: both transfers TIE on n01-09 (0.9986 /
0.9988). `qsa r2s`: WIN on n07-17 (0.9844), TIE on n01-09 (0.9992). GEMV-only climbs from the r4s pick (`w8-wp2b-tuQ`, 80
evaluations each, n01-09): `qsa r9g3` 0.9924 WIN, re-timed 0.9953 (n01-09, `c5n01`) / 0.9941 (n07-17, `c5n07`), both WIN;
`qsa r2s` 0.9852 WIN, re-timed 0.9956 WIN on n07-17 and 0.9886 on n01-09, a TIE there only because that job's
recompiled-pick control itself ran at 0.9861 (a FlyDSL code-layout effect of 1.4 %; alola rule 221). `delta r12g4`'s
transfer tied once on n07-17 (0.9991, `g2n07`) and won when re-timed on both nodes (0.9899 / 0.9901, `c4n07` /
`c4n01`); the r8g4 / r16g4 climb picks transferred to r12g4 did not beat it on both nodes (0.9891-0.9952).

### 6.5 Every served cell, adopted arm, both nodes

`parity port` = the 1:1 port (every ported stage of the kind at the default knobs), range over the jobs that timed it.
`adopted` = what the served set (`WP2b/picks_all.json`) links: `tuned` (6.2 / 6.4) or `xfer r1` / `xfer r4s` (the
kind's r1 / r4s pick, 6.3 / 6.4). One 24-round job per node and cell.

| cell | parity port | adopted (source) | n01-09 | n07-17 |
|---|---|---|---|---|
| delta r1 | 1.0121-1.0203 | tuned (6.2) | 0.9658 WIN | 0.9580 WIN |
| delta r2 | 1.0130 | xfer r1 | 0.9813 WIN | 0.9833 WIN |
| delta r3 | 1.0227 | xfer r4s | 0.9826 WIN | 0.9884 WIN |
| delta r4 | 1.0170 | xfer r4s | 0.9934 WIN | 0.9958 WIN |
| delta r2s | 1.0116 | xfer r1 | 0.9868 WIN | 0.9806 WIN |
| delta r3s | 1.0133 | xfer r4s | 0.9804 WIN | 0.9757 WIN |
| delta r4s | 1.0093-1.0111 | tuned (6.2) | 0.9739 WIN | 0.9788 WIN |
| delta r4g2 | 1.0151 | xfer r1 | 0.9846 WIN | 0.9876 WIN |
| delta r6g2 | 1.0105 | xfer r4s | 0.9854 WIN | 0.9861 WIN |
| delta r8g2 | 1.0133 | xfer r4s | 0.9833 WIN | 0.9867 WIN |
| delta r6g3 | 1.0107 | xfer r4s | 0.9818 WIN | 0.9820 WIN |
| delta r9g3 | 1.0088 | xfer r1 | 0.9814 WIN | 0.9849 WIN |
| delta r12g3 | 1.0133 | xfer r4s | 0.9875 WIN | 0.9793 WIN |
| delta r8g4 | 1.0106-1.0117 | tuned | 0.9875 WIN | 0.9874 WIN |
| delta r12g4 | 1.0116 | xfer r4s | 0.9901 WIN | 0.9899 WIN |
| delta r16g4 | 1.0074-1.0142 | tuned | 0.9886 WIN | 0.9854 WIN |
| qsa r1 | 1.0084-1.0183 | tuned (6.2) | 0.9524 WIN | 0.9618 WIN |
| qsa r2 | 1.0086 | xfer r1 | 0.9803 WIN | 0.9763 WIN |
| qsa r3 | 1.0103 | xfer r1 | 0.9822 WIN | 0.9812 WIN |
| qsa r4 | 1.0095 | xfer r4s | 0.9959 WIN | 0.9936 WIN |
| qsa r2s | 1.0271 | tuned | 0.9886 TIE (recompiled-pick control at 0.9861; tuning job 0.9852 WIN) | 0.9956 WIN |
| qsa r3s | 1.0088 | xfer r4s | 0.9849 WIN | 0.9864 WIN |
| qsa r4s | 1.0087-1.0112 | tuned (6.2) | 0.9886 WIN | 0.9895 WIN |
| qsa r4g2 | 1.0087 | xfer r1 | 0.9879 WIN | 0.9891 WIN |
| qsa r6g2 | 1.0089 | xfer r1 | 0.9893 WIN | 0.9914 WIN |
| qsa r8g2 | 1.0074 | xfer r1 | 0.9856 WIN | 0.9903 WIN |
| qsa r6g3 | 1.0085 | xfer r4s | 0.9873 WIN | 0.9904 WIN |
| qsa r9g3 | 1.0097-1.0126 | tuned | 0.9953 WIN | 0.9941 WIN |
| qsa r12g3 | 1.0099 | xfer r1 | 0.9947 WIN | 0.9965 WIN |
| qsa r8g4 | 1.0065 | xfer r1 | 0.9904 WIN | 0.9901 WIN |
| qsa r12g4 | 1.0157 | xfer r4s | 0.9957 WIN | 0.9905 WIN |
| qsa r16g4 | 1.0098-1.0114 | tuned | 0.9940 WIN | 0.9963 WIN |
| delta_ple r1 | 1.0064-1.0153 | tuned (6.2) | 0.9693 WIN | 0.9591 WIN |
| delta_ple r2 | 1.0146 | xfer r4s | 0.9778 WIN | 0.9776 WIN |
| delta_ple r3 | 1.0089 | xfer r4s | 0.9848 WIN | 0.9812 WIN |
| delta_ple r4 | 1.0125 | xfer r4s | 0.9865 WIN | 0.9907 WIN |
| delta_ple r2s | 1.0067 | xfer r4s | 0.9657 WIN | 0.9520 WIN |
| delta_ple r3s | 1.0087 | xfer r4s | 0.9747 WIN | 0.9713 WIN |
| delta_ple r4s | 1.0053-1.0234 | tuned (6.2) | 0.9869 WIN | 0.9891 WIN |
| delta_ple r4g2 | 1.0052 | xfer r4s | 0.9806 WIN | 0.9833 WIN |
| delta_ple r6g2 | 1.0093 | xfer r4s | 0.9800 WIN | 0.9797 WIN |
| delta_ple r8g2 | 1.0082 | xfer r4s | 0.9771 WIN | 0.9817 WIN |
| delta_ple r6g3 | 1.0021 | xfer r4s | 0.9747 WIN | 0.9779 WIN |
| delta_ple r9g3 | 1.0146 | xfer r4s | 0.9867 WIN | 0.9838 WIN |
| delta_ple r12g3 | 1.0041 | xfer r4s | 0.9845 WIN | 0.9811 WIN |
| delta_ple r8g4 | 1.0091 | xfer r4s | 0.9841 WIN | 0.9869 WIN |
| delta_ple r12g4 | 1.0182 | xfer r4s | 0.9902 WIN | 0.9826 WIN |
| delta_ple r16g4 | 1.0112-1.0126 | tuned | 0.9870 WIN | 0.9884 WIN |

### 6.6 WaveScope before / after (ATT, stamped builds, `w8-wp2b-att1` n07-17, `att2` n01-09; CUs 0 and 5)

Per stage, the span from the first to the last MFMA issue of each wave's visit (median over visits), the adopted arm over
the FlyDSL pick, same trace settings (`WP2b/wp2b_att_chain.py` over the per-wave timelines):

| stage | cell | MFMA-chain span, adopted / FlyDSL (CU 0, CU 5) |
|---|---|---|
| B1 | delta r4s | 0.56, 0.59 |
| B1 | delta r1 | 0.46, 0.47 |
| C1 | delta r4s | 0.56, 0.58 (FlyDSL visit from CU 5) |
| Q1 | qsa r4s | 0.49, 0.51 |
| C1 | qsa r4s | 0.68, 0.60 |
| B1 | delta_ple r4s | 0.74, 0.64 |
| C1 | delta_ple r4s | 0.73, 0.63 |
| P1 | delta_ple r4s | 0.90, 0.86 |
| control: an unported 20-MFMA stage in the same traces | delta_ple r4s | 1.00, 0.98 |

What the traces show: the FlyDSL chains issue one MFMA after each `s_waitcnt lgkmcnt(0)` (the LDS operand of every
step, and a scale VGPR rewritten between steps); the tuned chains issue back to back after the read-ahead loads. The tail
(last MFMA to the stage's exit stamp) is within 10 % of FlyDSL's at every stage. The traces' per-stage latency shares
are not comparable between the two builds: the FlyDSL stages stamp a "ready" mark after their poll and the rocKE
functions do not, so the stamped regions differ by the poll.

## 7. Quick end-to-end (PLAN decision 14; `truemono-quick`, 3 passes, 32 gsm8k prompts, n01-09)

The consolidated full stack (`full3` K=3 / `full0` K=0) with and without the WP2b functions (`arms_wp2b.sh`; functions
generated per served build by `wp2b_gen.py` over the stack's own functions). Ratios with / without of req tok/s (mean of 3
passes; higher is better), ms per round (K=3) and ms per token (K=0) (lower is better):

| job | functions linked | K=3 req tok/s | K=3 ms / round | K=0 req tok/s | K=0 ms / token |
|---|---|---|---|---|---|
| `qk-w8wp2b-e1` | the six r1 / r4s cells (14 functions) | 1.0044 | 0.9962 | 1.0059 | 0.9985 |
| `qk-w8wp2b-e2` | 47 cells, transfer picks at the grouped widths, FlyDSL at `qsa r9g3` | 1.0066 | 0.9976 | 1.0051 | 0.9950 |
| `qk-w8wp2b-e3` | **the adopted set, all 48 cells** (`WP2b/picks_all.json`, 112 functions) | **1.0141** | 0.9968 | **1.0088** | **0.9908** |

In e2 and e3 every pass with the functions is above every pass without them, at both K. Correct answers and acceptance
stay within the run-to-run spread (e3 K=3: 88 vs 92 of 96 correct, acceptance 3.045 vs 3.068; e1: 90 vs 89, e2: 90 vs
91; the hybrid's passes range 27-31 of 32); the functions are bitwise equal to the FlyDSL stages in situ (section 5). The
end-to-end gain is smaller than the layer ratios suggest (about 0.016-0.03 of a round from 6.5 and the layer counts): the
in-situ rig times one layer's 96 back-to-back launches over cold weights, while in serving the GEMV stages overlap the
neighbouring stages more and host / attention time is a larger share. The quick e2e is the measurement.

## 8. Lookback, losers, and "rejected on base X"

All verdicts are on the base "tuned stack (`picksI`) layer + the other GEMV stages at the stated knobs". Solo losers kept in
every pool (PLAN 13): `asop` / `bsop` lost alone at the first base (1.0121-1.024) and are in the delta r4s pick (`bsop=1`),
the delta r1 pick (`asop=1` on C1) and the qsa picks; `oob=split`, `barrier=lds`, `ra=3-4` were neutral alone and are in
the picks; `epi=reg` is in the qsa r4s round-2 pick together with `w_cpol=2` (the pair the lookback's GP flagged as
interacting). Decision-15 re-test of the layer's earlier levers on the new base: the joint climb (`w8-wp2b-tuG`, MoE / B3 /
schedule knobs in one pool with the GEMV knobs, started at the adopted picks) found `moe.map=groups, moe.iod_split=0,
sched.hold.m1=64` (and 4 more) at delta r4s: 0.9695 on n01-09, but 0.9782 vs the GEMV-only pick's 0.9788 on n07-17
(`c1n07`, a TIE), and the layer knobs alone lose there (1.0054). Not adopted; recorded as "no gain on base WP2b picks,
n07-17".

## 9. What the numbers mean for D0 / the targets

The delta layer is 35 of 48 target layers; its adopted GEMV ports take 0.4-4.2 % off its layer time on the tuned stack,
at every served width, on both nodes (section 6.5). The rocKE GEMV family is no longer a parity risk for the true mono:
every FlyDSL GEMV stage has a bitwise rocKE replacement that is faster in situ, at every served width. End to end the
served stack gains +1.4 % req tok/s at K=3 and +0.9 % at K=0 (ms per token 0.991) (section 7).

## 10. Not done / follow-ups

- Climbs at every width: 12 of 48 cells were climbed (the six r1 / r4s cells, the four widest grouped cells, `qsa r2s`,
  `qsa r9g3`); the other 36 run a transferred pick, which wins on both nodes. Per-width climbs would add a little: the four
  grouped climbs moved the layer by +0.7 % to -0.3 % against the transfer.
- Tolerance-class lever not built: several accumulators per chain (split-K or interleaved chains) shortens the chain's
  latency but changes the summation order: an N lever behind the T0-T5 ladder (DESIGN 12.4), not WP2b's bitwise scope.
- WaveScope covered the r4s / r1 cells; the grouped widths were not traced.

## Verification (independent verifier, W8; 2026-11-02)

Verdict: **partial.** The numerics and in-situ timing claims hold, and I reproduced them on both nodes. Three things in
the report are inaccurate: the chk4 row of section 5, how sections 7 / 9 attribute the end-to-end gain, and the
statement that the committed emitter is the code that ran on hardware. What follows is what I checked and how.

**Re-run (stage `w8-vWP2b-h`; the builder's `rocke-w8-wp2b` / `atom-truemono-w8-wp2b`; same `picksI.json` and
`hyb_layer.py`, LF hashes equal to the builder's stage).** Jobs `w8-vWP2b-n01` (68246667, n01-09) and `w8-vWP2b-n07`
(68246669, n07-17). Both ran with fresh seed 9137, the check then the time mode, 24 rounds, both controls, and the
adopted arms read from `picks_all.json` (`WP2b/vWP2b_arms.json`).

- Numerics: delta r4s and qsa r2s gave CHECK_PASS on both nodes. Every arm was bitwise vs the plain mono and its scratch
  equal to the pick's. All 23 required mutants were caught per node (b1 / c1 / q1, including `c1@row` at r4s), and the
  only uncaught ones were the 4 informational order-only `chain_swap` mutants. All 12 timed arms passed the gate before
  timing and POST_CHECK after.
- Timing, adopted arm over the FlyDSL pick (DESIGN 9.1). It was a WIN in 12 of 12 (cell, node) pairs, 24 of 24 rounds
  below 1 each time. The 1:1 parity port was a LOSS in all 12 (1.004-1.012):

| cell | n01-09 | n07-17 | report (n01 / n07) |
|---|---|---|---|
| delta r4s | 0.9755 WIN | 0.9759 WIN | 0.9739 / 0.9788 |
| delta r1 | 0.9612 WIN | 0.9596 WIN | 0.9658 / 0.9580 |
| qsa r4 | 0.9933 WIN | 0.9904 WIN | 0.9959 / 0.9936 |
| qsa r12g3 | 0.9938 WIN | 0.9952 WIN | 0.9947 / 0.9965 |
| qsa r2s | 0.9812 WIN | 0.9740 WIN | 0.9886 TIE / 0.9956 |
| delta_ple r4s | 0.9824 WIN | 0.9833 WIN | 0.9869 / 0.9891 |

  At qsa r2s the recompiled-pick control again ran fast (0.986 / 0.984). Across the six jobs that timed this cell it
  ranged 0.984-1.011, and the adopted arm moved with it (0.974-0.996). The cell's verdict needs more than one job per
  node (alola rule 228); in these two jobs it wins on both nodes.
- The 96 CPU tests pass locally (`tests/test_qwen38_wp2b_gemv.py`).

**Audit of the builder's data.**

- Section 6.5 regenerates exactly from the raw results (`wp2b_table.py` over `logs/q38/wp2b-private`; all 96 entries
  match). I re-applied the DESIGN 9.1 win rule to every arm in all 16 timing jobs: 0 verdict mismatches, 0 failed gates,
  `post_check_ok` everywhere, both controls present everywhere, and every job at 24 rounds. Each of the six tuned cells'
  confirmation arms (c1n07 / c2n07 / c3n01) is the adopted pick (qsa r1 differs only in the default `pause=0`).
- chk5 (154 / 154 arms, 273 / 273 required mutants), chkG2 (all 21 grouped cells PASS) and the rv7 audit (112
  functions, 0 hazards, every B1 / Q1 chain tied, 7 C1 functions on the intrinsic form) match the report.
- The parity evidence is a per-stage ISA diff with its catalogue findings (rv6b `diff/*.md`), `.ll` profiles, CPU
  checks of MFMA formats and op_sel bytes, and the bitwise gates. This is parity of operations and rounding, not of
  instruction streams; the report says so.
- Byte identity of existing builds holds by construction. No module outside `stages2/` or the test imports
  `stages2.gemv`, the module has no import-time side effects (no monkeypatching or registry writes; the CPU test checks
  the registry), and the MoE / B3 `.ll` hashes in the check logs do not change between arms. No byte-diff build of an
  existing stage was run.
- The end-to-end runs (e1-e3) include the pinned hybrid in the same job, in rotated order, with 3 passes.

**Findings.**

1. **chk4 is misreported.** Section 5 gives it as "75 of 75, all [mutants]". In the log, delta r4s and qsa r4s end in
   CHECK_FAIL: the required mutant `c1@mutate=row` was not caught. delta_ple r4s stopped with a `ValueError` (a knob
   refusal) after 9 arms and ran no mutants. Coverage was restored by chk5 and by this re-run, where `c1@row` is caught
   at r4s. The section-5 row should say so.
2. **chkG1 is understated.** In chkG1, delta r4g2's single-stage arms `b1` and `c1` failed as the first arms, right
   after the gate self-test's mutants. delta_ple r4g2 was a GPU memory fault (illegal address in the grouped layer
   kernel), not a bitwise miss. The scratch-size explanation (alola rule 212) is plausible, and chkG2 passes with the
   narrow rig's scratch. The report should still name the fault.
3. **The end-to-end gain is overstated and misattributed.** A CONC 1 quick e2e runs only the `r1` (K=0) and `r4s`
   (K=3) cells. e2 and e3 linked identical functions for those six cells; I diffed the staged `picks_all.json`
   snapshots `quick-20261101-175728` and `-212013`, and only grouped and qsa r2s / r9g3 picks differ. So e2 and e3
   are replicates of the same served code, and "all 48 cells linked" adds nothing at CONC 1. Their K=3 req tok/s
   ratios were 1.0066 and 1.0141, and K=0 1.0051 and 1.0088. The robust signal is the round time: ms per round
   0.996-0.998 (mean) and 0.994-0.997 (median) across e1-e3. In e3, K=3 ms per token was 1.0045, worse, because
   acceptance was lower (run-to-run noise). The defensible statement is about +1 % req tok/s at K=3 and +0.7 % at K=0
   (e2 and e3 pooled), from the r1 / r4s cells. The 42 other cells (K=1 / K=2 and CONC 2-4) have in-situ evidence only;
   none of them was served end to end.
4. **The committed emitter is not byte-for-byte the code that ran on hardware.** The staged `stages2/gemv.py` used by
   every GPU job (manifest sha `673456c5...`, synced 14:20 local) differs from the copy in the tree (`ef9da99e...`,
   15:13). The only difference is an appended `framework_factory`, a Path-B factory. The ATOM extern path is unchanged,
   so the WP2b results still apply to it, but the factory itself has no hardware evidence. The report does not
   mention it.
5. **Minor points.**
   - The tuner's lookback ran 0 rounds in tuA, tuB, tuC, tuG and tuQ. tuG is the decision-15 joint re-test. The
     adopted six-cell picks come from tuD-tuF (3-8 lookback rounds).
   - Section 6.2's numbers come from the tuning jobs, and the delta r1 number is for a knob value that is now refused.
   - For the 36 transfer cells, one node's 6.5 value comes from the job that chose between the two transfer arms, a
     mild selection bias. The other node's value is an independent confirmation.

**What still stands:** bitwise rocKE replacements for B1 / C1 / Q1 / P1 at all 48 served widths, and in situ they are
faster than the FlyDSL stages of the tuned stack. This was confirmed on both nodes for the six re-timed cells,
including the reported TIE. The end-to-end effect at CONC 1 is about 1 % at K=3, with the per-round time 0.994-0.998
of the stack's.
