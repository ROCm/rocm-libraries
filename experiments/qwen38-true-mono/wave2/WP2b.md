# WP2b: the GEMV-family stages B1, C1, Q1, P1 as rocKE stage functions (wave 2, W8)

D0 section 5 wave 2 (WP2b: "GEMV family B1/C1/Q1/P1 per DESIGN 5, all served widths"), PLAN decision 16 (reverse-engineer
FlyDSL's code, reach parity by construction, then crush it with the parametric search), decisions 13 / 15 (lookback with
the losers kept, rejections "on base X", re-tests on the new base). Ratios, shares and counts only. Absolute times:
`hackathon-prep/private-perf/qwen38-true-mono/wave2/WP2b.md`. Below 1 is faster.

Status: **built, bitwise at every served width, and faster than the FlyDSL stages on the tuned stack in every narrow
served cell measured so far, on both rocke nodes** (details and open items below). The 1:1 port alone (the FlyDSL
instruction choices) loses 0.8-2.3 % of the layer; the reasons are in section 3, and the tuned ports win 2.4-4.4 % of the
whole layer (delta) and 0.5-2.1 % (qsa).

## 1. What was built

| piece | where |
|---|---|
| the emitter: B1 (QKVZ + conv + SiLU + conv state, verify window, grouped history), C1 (out_proj + 4-wave reduce + HC write `w` / `whs`, PLE residual), Q1 (QSA projection), P1 (embedding MXFP8 pack, key chain, 4 value chains), narrow `r1-r4`, `r2s-r4s` and grouped / wide `r4g2 .. r16g4`; `GemvKnobs` (section 4); `GemvShape`; `write_gemv_functions`; CLI | `copy-rocm-lib/.../qwen38_mono/stages2/gemv.py` (new package `stages2/`) |
| CPU tests (92): knob validation, every shape lowers, the parity build's MFMA operand formats and op_sel bytes equal FlyDSL's, the asm build ties every accumulator, every mutant changes the IR, refusals, NUMA sidecar, the B3 prologue host | `copy-rocm-lib/.../library/tests/test_qwen38_wp2b_gemv.py` |
| in-situ harness: `check` (bitwise vs the plain mono + whole scratch vs the FlyDSL pick, must-fail mutants), `time` (DESIGN 9.1), `stamps` (stage timelines), `launch` (WaveScope) | `hackathon-prep/qwen38/truemono/WP2b/wp2b_insitu.py` |
| tuner backend + driver (signed lookback, order restarts, TPE; GEMV-only or joint with the layer's MoE / B3 / schedule knobs) | `WP2b/wp2b_tune.py` |
| served-stack generator (functions next to a stack's own; its numa lever, its pause, the B3 prologue host) and quick-e2e arms | `WP2b/wp2b_gen.py`, `WP2b/arms_wp2b.sh`, `WP2b/stage_quick_wp2b.sh` |
| reverse toolkit jobs: layer dumps with / without the functions, per-stage split, diff, MFMA-chain and hazard audit; container codegen repro | `WP2b/wp2b_rv_job.sh`, `WP2b/wp2b_isa.py`, `WP2b/wp2b_llc_job.sh` |
| job driver and staging (no local deletion: every pack goes to a new `stage_build/<timestamp>` dir) | `WP2b/wp2b_job.sh`, `WP2b/sync_wp2b.sh` |

No existing rocKE file was edited: `stages2/` is new and nothing outside it imports it, so every existing build is
byte-identical by construction (`git status`: only `stages2/` and the new test file are WP2b's; the CPU test also checks
that the `stage_fn` registry is unchanged). No ATOM-truemono file was edited: the functions link through the existing
`ATOM_Q38_EXTERN_STAGES` hook (any stage name; `{kind}` path templates select per-kind files).

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
4. **Climb** with the mono tuner (`SignedLookback` + order restarts + TPE; losers kept in the pools) per served cell.
5. **Time in situ** on the tuned stack's layer (`truemono-quick/lookback/hyb_layer`: 16 cold weight copies, 96
   launches per graph, shuffled rounds, a second graph and a recompiled build as controls; DESIGN 9.1 win rule), both
   rocke nodes.
6. **Stamps / WaveScope** before and after (section 6.4).

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
| chkG1 / chkG2 | grouped / wide `r4g2, r6g2, r8g2, r6g3, r9g3, r12g3, r8g4, r12g4, r16g4` x 3 kinds | see section 9 | all |
| every tuner evaluation | the 6 served cells | gated before timing (and after, in the confirmations) | |

The order-only mutants (`chain_swap`, `vsum`: a fp32 reassociation) are reported, not required: an output moves only where
a 1-ulp fp32 change crosses a bf16 rounding boundary (all 56 were invisible in chk5, and all in chkG2). The order itself
is proven by construction (section 3: the same MFMA sequence on one accumulator).

Harness findings on the way (both fixed in `wp2b_insitu.py`, recorded as alola rule 212):

- The rigs leave the PLE embedding row at zero, so P1's outputs did not depend on its weights: every P1 mutant passed the
  first gate. The harness fills the row with seeded random data and recomputes the reference.
- The grouped rig at `r4g2` keeps a scratch too small for ATOM's NUMA area: arms built right after a mutant read its
  stale same-tag pairs and failed; with the narrow rig's scratch and calibration every `r4g2` arm is bitwise.

## 6. Timing in situ (DESIGN 9.1; layer time over the tuned stack's FlyDSL pick, same job)

### 6.1 The 1:1 port and the first fixes (jobs `w8-wp2b-t1` n07-17, `t2` n01-09; 24 rounds)

| cell | parity port (all stages) | B1 asm tied | C1 parity | C1 irf + asm | controls (min-max) |
|---|---|---|---|---|---|
| delta r4s | 1.0111 LOSS | 0.9921 WIN (irf) / 1.0023 (asm poll) | 1.0100 LOSS | 1.0077 LOSS | 0.9979-1.0018 |
| delta r1 | 1.0203 LOSS | 0.9958 (24/24, inside the margin) | 1.0073 LOSS | 1.0947 LOSS | 0.9973-1.0048 |
| qsa r4s | 1.0112 LOSS | Q1: 1.0016 | 1.0081 LOSS | 1.0004 | 0.9985-1.0026 |
| delta_ple r4s | 1.0234 LOSS | 0.9950 WIN | 1.0093 LOSS | 1.0133 LOSS | 0.9981-1.0037 |

The C1 port alone loses about 1 % in every cell; with stamps (`w8-wp2b-s1`) its increment grows by 13 % and M1 / M2 wait
longer at r4s with the asm poll, and not with `irf` (stamped builds bias time: diagnostics only). The `irf` C1 at delta r1
was a 9.5 % outlier in one unstamped build (not in the stamped one, not elsewhere): treated as a code-layout effect of that
build, not a property of the poll; it is a loser kept in the pools.

### 6.2 Tuned ports (tuner picks, confirmed on 16 rounds in the tuning job, n01-09)

| cell | pick (per-stage knobs) | vs FlyDSL pick | vs parity port |
|---|---|---|---|
| delta r4s | B1 `mfma=asm,asm_lead=-1,asm_tail=20,mb_impl=ir,w_cpol=2,s_cpol=2,barrier=lds,cprio=3`; C1 `mfma=asm,asm_lead=3,asm_tail=20,mb_impl=irf,bsop=1,oob=split,ra=4,w_cpol=2,s_cpol=2,barrier=lds` | **0.9728** WIN (16/16) | 0.9642 |
| delta r1 | B1 `mfma=asm,asm_tail=12,mb_impl=irf,oob=split,ra=3,w_cpol=3,s_cpol=2,barrier=lds,hdr=vmem` (lead: see 6.3); C1 `mb_impl=ir,asop=1,ra=3,w_cpol=19,barrier=lds,pause=1` | **0.9563** WIN | 0.9454 |
| delta_ple r1 | B1 `mfma=asm,asm_lead=-1,asm_tail=12,cprio=3,oob=split,w_cpol=2,s_cpol=2,barrier=lds`; C1 as delta r4s with `asm_lead=-1`; P1 parity | **0.9565** WIN | 0.9551 |
| qsa r4s | C1 `mb_impl=irf,oob=split,w_cpol=19`; Q1 `mb_impl=ir,w_cpol=19` (round 1) | **0.9952** WIN | 0.9861 |
| qsa r1 | C1 `mb_impl=ir,w_cpol=19`; Q1 `mb_impl=irf,s_cpol=18,w_cpol=19` (round 1) | **0.9789** WIN | 0.9704 |

### 6.3 Second node, fresh seed (job `w8-wp2b-t3n07`, n07-17, seed 3, 24 rounds)

| arm | delta r4s | delta r1 |
|---|---|---|
| round-2 delta r4s pick | **0.9764** WIN (24/24) | 0.9788 WIN |
| round-2 delta r1 pick, with `asm_lead=-1` | 0.9818 WIN | **0.9590** WIN (24/24) |
| round-1 picks | 0.9829 / 0.9939 WIN | 0.9767 / 0.9676 WIN |
| parity port | 1.0084 LOSS | 1.0121 LOSS |
| controls | 0.998-1.0019 | 0.997-1.0034 |

The tuner's delta r1 pick had `asm_lead=0` on B1 (no lead even after the zero accumulator). It passed the bitwise gate, but
the static rule says a VALU write needs 2 wait states before an XDL read; values below 2 are now refused, and the
confirmed delta r1 arm is the same pick with `asm_lead=-1` (0.9590 on n07-17 vs 0.9563 with 0 on n01-09).

### 6.4 Stage timelines (stamps, diagnostics)

Critical-path increments of the FlyDSL pick (shares of the layer): delta r1 B1 0.13, C1 0.06; delta r4s B1 0.10, C1 0.05;
qsa r4s Q1 0.05, C1 0.03. With the B1 asm port (irf) B1's increment fell to 0.973 of FlyDSL's at both r1 and r4s
(`w8-wp2b-s1`).

## 7. Lookback, losers, and "rejected on base X"

All verdicts above are on the base "tuned stack (`picksI`) layer + the other GEMV stages at the stated knobs". Solo losers
kept in every pool (PLAN 13): `asop` / `bsop` lost alone at the first base (1.0121-1.024) and are in the delta r4s pick
(`bsop=1`) and the delta r1 pick (`asop=1` on C1); `oob=split`, `barrier=lds`, `ra=3-4` were neutral alone and are in the
picks. The joint climb (the layer's MoE / B3 / schedule knobs with the GEMV knobs in one pool) from the round-2 picks is
the decision-15 re-test of every earlier layer lever on the new base (section 9).

## 8. What the numbers mean for D0 / the targets

The delta layer is 35 of 48 target layers; its tuned GEMV ports take 2.4 % (K=3 verify, r4s) and 4.1-4.4 % (K=0, r1) off
its layer time on the tuned stack, both nodes. With layer kernels about 0.79 of a mono round (PLAN), the delta + qsa +
PLE ports together are roughly 0.016-0.02 of a K=3 round and 0.03 of a K=0 step (estimate from the layer ratios and
layer counts; the quick end-to-end number, section 9, is the measurement).

## 9. Open items / still running at the time of writing

- qsa round 2 and delta_ple r4s climbs (`w8-wp2b-tuF`), the joint decision-15 re-test (`w8-wp2b-tuG`).
- Grouped / wide timing (r8g4, r16g4) and checks of the remaining wide variants (`chkG2`).
- Quick end-to-end (PLAN 14) with the WP2b functions on the full stack (`arms_wp2b.sh`).
- WaveScope (ATT) before / after per stage (`wp2b_job.sh att`).
- Tolerance-class lever not built: several accumulators per chain (split-K or interleaved chains) shortens the chain's
  latency but changes the summation order: an N lever behind the T0-T5 ladder (DESIGN 12.4), not WP2b's bitwise scope.
