# W15 CLIMB: hill climbing with lookback over the K2 knob surface (ratios only)

DeepSeek-V4.1-Flash decode, TP2, DSpark 5, MI355X (gfx950), ATOM-dsv41 `df86c482` + the W16 I1 hooks.

- Absolute times: `hackathon-prep/private-perf/dsv41-mono/w15/CLIMB-abs.md`.
- Harness: `hackathon-prep/dsv41/w15/climb/`.
- Raw outputs: `hackathon-prep/dsv41/w15/climb/results/<job>/` and `/scratch/users/vanantha/w15c/results/`.

Each round appends a section. Ledger columns:

- **move**: the config change against the round's base;
- **base**: the round's base;
- **ratio**: move / base, one value per rig width, median over batches with the per-batch range;
- **verdict**: WIN / TIE / LOSS, with the node(s);
- **WaveScope delta**: the ATT ledger change, or why none was taken.

---

## Round 0 (2026-11-05)

**Verdict: IMPROVED** (at the rig rung). The base moved from the K2-moe port to the port with
`ug_sop=1`: the ug GEMV picks its MXFP4 weight-scale bytes by the scaled MFMA's `op_sel` (bitwise).

- **6 rows.** 0.995 of the base on both nodes:
  - n14-05: 6 of 6 batches 0.992-0.995, every one below all 24 base-copy values (0.995-1.005);
  - n07-17: 5 of 5 batches 0.990-0.998 against base copies at 0.993-1.006, rank-sum p ~ 0.01.
- **24 / 48 rows.** A tie on both nodes.
- **Bitwise.** Every launch, both nodes (n07-17's 12-byte noise class aside).

No new schedule lever moved the base: holds, `prio`, `pprio`, placement shifts and `rbar` all tie. Every lookback
combination's 6-row gain decomposes to `ug_sop` alone (0.6). The served / graph-mode rung has not run yet: it is
round 1's first item.

### 0.1 What was built this round

All of the following are default-off knobs, additive, and bitwise by construction (ordering, placement or priority
only).

**ATOM side** (`ATOM_DSV41_RK_LEV`; `mono/kernels/rk_lev.py`, `rk_lev_dev.py`, `moe.py`, `layer_post.py`,
`attn_pre.py`). These carry PLAN decision 10's schedule surface:

| lever | what |
|---|---|
| `prio.<stage>=P` (P 1-3; any of the 20 hold stages) | `s_setprio P` where the stage's CTAs take a task. With any `prio.*` on, every hold site sets its own stage's priority (0 unless given), so one stage's priority never leaks into the next. |
| `pprio=P` (new) | Every ATOM-side mailbox poll runs at priority 0, and the rest of the wave's code at `P`. Spinning waves give their issue slots to working ones. |
| `pl.<stage>=D` (router, route, shared, ug, down, red; D 1-255) | K2b stage placement: the stage's first task runs on CTA `base + D` instead of ATOM's `base` (`first_task`). The task -> work map does not change, only which CTA runs a task. Dependencies only flow from earlier stages to later ones, so every shift is deadlock-free. |

**rocKE side** (`dsv41_mono/stage_moe.py`):

| knob | what |
|---|---|
| `rbar` (route; 1 = ATOM) | `rbar=0` drops route's trailing barrier. It targets the PORT-K2-moe WaveScope ledger item "7 of 8 waves idle at route's trailing barrier, 0.23 at 24 rows". Route keeps no LDS state, and every wave skips the same barrier, so the CTA's barrier count stays matched. |

**Rig:** `ATOM_DSV41_RK_RIG_ROWS` (`rk/rig.py`) times only the listed launch widths.

**Harness:** `climb_driver.py` is the climb itself.

- **Engine and rig.** It loads one eager engine and uses W16's in-situ rig as the score.
- **Search.** The strategy is `qwen38_mono/tune` `SignedLookback`: a climb one lever at a time; a re-test of every
  reject after each accepted move; a covering screen plus a GP-EI combination lookback over every level, losers
  included; then a revisit. It runs through the rocKE search engine (`run_search` + W16's `PrefetchEvaluator` /
  `LadderBackend`), one rig batch per round of the search.
- **Score.** Per rig round, the config's K2 time over the base config's copy in the same batch, combined over the
  widths by a geometric mean.
- **Controls in every batch.** Each batch also times:
  - the base config (`ref_base`);
  - a second build of the batch's first candidate (`ctl_dup`);
  - two must-fail mutants, a down `put_bit` and a `put_bit` across the xq / route / reduce / shm set.

**Space** (`climb_space.json`): 48 params, 75 non-base levels.

- the new `rbar`, `pl.*` (x3 values), `pprio` (x2) and `prio.*` (13 K2 stages);
- `hold.<stage>` = 4 / 16 for the 13 K2 stages (the Coz-style slope sweep, playbook rank 1);
- the port's down / ug knobs (`r_max`, `pslot`, `smap`, `sop`, `sh_first`, `ug_sop`, `ug_ra`), losers included;
- the WP4 K2 levers `gsop` / `gsop2` / `gra` / `askip` / `pskip` / `poll1` / `pfx` / `xcd`.

The base is the K2-moe port (every K2b stage as a rocKE function, PORT-K2-moe.md). The engine's own arm is ATOM
mono, which is the rig's bitwise reference.

### 0.2 Gates

**Knob-off identity.** With every new knob off, the traced K1 / K2 IR equals a pristine W16 `i1e` tree byte for
byte. This was checked at K1 s6 and at K2 s6 and s24 (the full FlyDSL pass-pipeline text, `local_trace_ccl2.py`).
The generated route functions equal the pre-edit `stage_moe.py`'s `.ll` at 5 / 6 / 24 / 48 rows, for the target
(384 / 6) and the draft (128 / 3). The knobs on do change the IR:

- `prio` / `pprio` emit `s_setprio`;
- `pl.*` changes the K2 s24 IR;
- `rbar=0` changes the route `.ll` at every width.

**Bitwise.** Every candidate launch is compared byte for byte with ATOM's build from the same snapshot, with LDS
poisoned in between. In the climb, all 282 timed non-mutant variants on n14-05 (160 candidates) were bitwise.
On n07-17, 34 of 282 had 6-30 differing bytes in some launches, and the base config's own copies and twins show the
same (ATOM's own K2a-seam nondeterminism, as PORT-K2-moe's ATOM-vs-ATOM control found). n07-17 DIFF verdicts are
therefore read against the same batch's twin, and bitwise status is taken from n14-05.

**Mutants.** Every batch carried both mutants:

| jobs | caught |
|---|---|
| climb jobs r0h / r0i | 114 / 114 and 114 / 114 |
| confirmation 1 | 8 / 8 per node |

### 0.3 The climb (jobs r0h on n14-05 and r0i on n07-17)

- **Setup.** Each job ran 160 evaluations in 57 rig batches at 6 and 48 rows (C1 and C8 steps of the eager engine),
  with 2 code-object copies per variant built copy-major.
- **Climb order.** r0h climbed in space order, r0i in reverse order (order is a knob).
- **Accepted moves: none, on either node.**
- **Singles.** Every new single lever and level sat in 0.989-1.009 of the base (per width, per node). That is
  inside the per-batch A-vs-A band: `ctl_dup` (a second build of a candidate, 110 per node) read 0.985-1.048 on
  n14-05 and 0.990-1.036 on n07-17, mostly within +-0.6 %. Two older WP4 levers lose at 6 rows on both nodes. The
  table below lists the singles that stand out.

| move | n14-05 6 / 48 rows | n07-17 6 / 48 rows | verdict |
|---|---|---|---|
| `ug_ra=1` (the known loser) | 1.001 / **1.039** | 1.005 / **1.038** | LOSS at 48 rows, both nodes |
| `xcd` (WP4 ug XCD homing) | **1.013** / 1.001 | **1.016** / 1.003 | LOSS at 6 rows, both nodes |
| `pfx` (WP4 wo_b weights first) | **1.008** / 0.996 | **1.010** / 0.997 | LOSS at 6 rows, both nodes (WP4: 1.017-1.022) |
| `pl.down=128` | 0.994 / 0.995 | - | TIE (confirmation 1 below) |
| `hold.*` = 4 / 16, all 13 K2 stages (26 levels) | 0.989-1.009 | 0.996-1.004 | TIE: no negative Coz slope beyond the twin band |
| `prio.*=3`, all 13 K2 stages | 0.994-1.007 | 0.991-1.003 | TIE |
| `pprio` = 1 / 3 | 0.995-0.998 | 0.996-1.002 | TIE |
| `pl.*` (18 levels) | 0.994-1.005 | 0.993-1.007 | TIE |
| `rbar=0` (new) | 1.001 / 0.999 | 1.002 / 0.999 | TIE |
| `ug_sop=1` | 1.000 / 0.999 | 0.996 / 0.999 | TIE (r0g, an earlier job: 0.984 at 6 rows) |

**What the lookback chose.**

- r0h's lookback best was a 10-lever combination at 0.993 / 0.992.
- r0i's "best", 0.989, came from a batch whose `ref_base` copy was itself slow: every candidate in that batch read
  0.977-0.988 at 48 rows.
- **Lesson.** One base copy per batch is not a stable normalizer. In confirmation 1, the copy loaded first
  (`ref_base`) ran 1.003-1.005 of the other two base copies at 48 rows on both nodes. From now on, normalize by the
  median of several base copies at shuffled build positions (`--shuffle`).

### 0.4 Confirmation 1 (jobs r0j on n14-05 and r0k on n07-17)

- **Setup.** 12 candidates (the lookback picks of both nodes plus the best singles) and 3 base copies, in 4 batches
  per node.
- **Ratio.** Each value is the candidate over the median of the batch's base copies: the median over batches, with
  the batch range in brackets.

| candidate | n14-05 6 rows | n14-05 48 rows | n07-17 6 rows | n07-17 48 rows |
|---|---|---|---|---|
| base copies (twin band) | 0.999-1.003 [0.994-1.004] | 0.997-1.006 | 0.996-1.003 [0.992-1.012] | 0.998-1.003 |
| `F` (r0i lookback: `pl.shared=8, pl.red=8`, prio slice2 / route / ug / down / red, hold woa / slice2 = 4, down `r_max=1, pslot`, `ug_sop`, `gsop2`) | **0.993** [0.990-0.997] | 0.999 | **0.990** [0.990-0.994] | 0.999 |
| `G` (r0i lookback: `pl.red=1`, prio woa / gate2 / shared / red, hold wob / shared = 4, `ug_sop`, `askip`, `pskip`) | **0.993** [0.989-0.998] | 0.998 | **0.990** [0.989-0.996] | 1.000 |
| `askip` | 0.999 | 0.995 [0.994-0.997] | 0.996 | 0.996 [0.994-0.998] |
| `C` (r0h lookback) | 1.000 | 0.995 | 0.997 | 0.999 |
| `A` (r0h lookback best) | 0.998 | 0.999 | 1.000 | 1.001 |
| `pl.down=128` | 1.000 | 0.999 | 0.997 | 1.002 |
| `pprio=3`, `prio.score=3`, `hold.route=4`, `hold.gate2=16` | 0.997-1.001 | 0.998-1.000 | 0.997-1.001 | 0.998-1.002 |

`F` and `G` are the only candidates whose median falls below every base copy's median at 6 rows on both nodes.
Their ranges still touch the twin band's lower edge. Their shared elements are `ug_sop=1`, `prio.red=3`, a
`pl.red` shift and several `prio.*`. Confirmation 2 (0.6) decomposes them.

### 0.5 Rust direct compiler

- **In place: not applicable this round.** The hybrid links every rocKE stage into ATOM's FlyDSL kernel as LLVM IR,
  and the direct compiler cannot emit a linkable device function (PORT-K2-moe / PORT-K2-attn-gemv). So the
  per-stage `ROCKE_CODEOBJ_DIRECT_LIB` knob has no in-place arm.
- **Standalone status.** Carried from the ports, not re-measured:
  - xq wins 0.957-0.961 on both nodes;
  - route loses 1.01-1.06 (missing `v_pk_add_f32` pairing; uniform index arithmetic not folded);
  - reduce ties;
  - down still miscompiles.
- **What makes it a lever.** A device-function emission mode in `rocke-codeobj` (an LLVM-linkable relocatable
  object with the extern ABI), or the S1 rocKE-owned K2 layer kernel (W16), where whole-kernel direct applies.
  Both are compiler / S1 work, not climb moves. They are recorded as the next Rust step.

### 0.6 Confirmation 2 (jobs r0l on n14-05 and r0m on n07-17)

- **Setup.** `F` / `G` decomposed. Rows 6 / 24 / 48 (C1 / C4 / C8), 4 base copies, the build order shuffled per
  batch, 6 batches on n14-05 and 5 on n07-17.
- **Ratio.** Each value is the candidate over the median of the batch's 4 base copies (leave-one-out for the base
  copies themselves): the median over batches, with the batch range in brackets.

| candidate | n14-05 6 rows | n14-05 24 | n14-05 48 | n07-17 6 rows | n07-17 24 | n07-17 48 |
|---|---|---|---|---|---|---|
| base copies (4) | 0.999-1.002 [0.995-1.005] | 0.999-1.001 | 1.000-1.001 | 0.998-1.001 [0.993-1.006] | 0.997-1.001 | 0.999-1.001 |
| **`ug_sop=1`** | **0.995** [0.992-0.995] | 1.001 | 0.999 | **0.995** [0.990-0.998] | 0.999 | 1.000 |
| `ug_sop` + `prio.red=3` | 0.995 [0.993-1.000] | 1.001 | 1.001 | 0.997 [0.994-1.006] | 1.000 | 1.001 |
| `F` | 0.994 [0.994-0.997] | 1.002 | 1.001 | 0.997 [0.994-0.998] | 1.001 | 1.002 |
| `G` | 0.993 [0.989-0.997] | 1.002 | 1.000 | 0.997 [0.996-1.001] | 1.000 | 0.999 |
| `F` without `ug_sop` | 1.000 | 1.001 | 1.002 | 1.002 | 1.000 | 1.000 |
| `G` without `ug_sop` | 0.999 | 0.998 | 1.000 | 1.000 | 0.997 | 1.001 |
| `F`'s `prio.*` only | 0.999 | 1.002 | 1.001 | 1.001 | 0.999 | 1.000 |
| `G`'s `prio.*` only | 0.999 | 1.000 | 0.999 | 1.001 | 1.000 | 1.000 |
| union of F / G `prio.*` + `ug_sop` | 0.995 | 1.001 | 0.999 | 0.998 | 1.000 | 1.000 |
| `askip` | 0.999 | 0.999 | 0.997 | 1.002 | 1.000 | 0.996 |
| `C` | 0.998 | 0.999 | 0.996 | 1.002 | 0.997 | 0.996 |

**Verdicts.**

- `ug_sop=1` is a **WIN at 6 rows on both nodes** (accepted: the new base) and a TIE at 24 / 48. Everything else
  in `F` / `G` is a TIE. The interaction screen (union, `+prio.red`) adds nothing on top of `ug_sop`.
- `askip` and `C` read 0.996-0.997 at 48 rows on both nodes, inside the base copies' 48-row spread at the batch level
  (0.994-1.005). They are retained as candidates for the next round's re-test on the new base.

**A hang (gate finding).** On n07-17, batch 6 of r0m hung in the rig's timing loop: candidate `C`, its second
code-object copy, 48 rows, layer 1, round 0, both ranks in the same launch. py-spy put both TP ranks in
`rig.py:167` (`Event.synchronize`). The same binary had run every earlier launch of 5 batches, 6 more on n14-05 and
4 in confirmation 1. So it is an intermittent deadlock, either of a lever in `C` (`pl.route=8`, `pl.red=8`, `poll1`,
`prio` / `hold`, down `r_max=1`, `smap`, `sop`, `gsop2`, `askip`) or of the node. The rig's restore is
stream-ordered and every launch is fenced by `torch.cuda.synchronize` plus a TP barrier, so the rig itself is not the
suspect. The job was cancelled (results of batches 1-5 kept). `C` is not adopted.

**Round 1 must bisect it:** `C`'s levers one by one, under repeated launches with deadline polls, on both nodes,
before any of them is combined again. `ug_sop` alone ran 34 batches without a hang.

### 0.7 WaveScope

**Before / after of the accepted move** (jobs w15c-att1 and w15c-split1 on n07-17).

- **Setup.** ATT (`rocprofv3 --att`, one CU per XCD) of the served-engine K2 at 6 rows, layers 120-121, both ranks:
  the port (`on6`) against the port with `ug_sop=1` (`ugs6`), each attributed to stages through its own
  stage-marked debug split (rv_isa).
- **Sample.** One of the eight `ugs6` dispatches waited about 100x longer than the others (a slow peer). The numbers
  below are from the other three `ugs6` dispatches and all four `on6` ones.

| ug stage (per traced CU) | port | port + `ug_sop` | ratio |
|---|---|---|---|
| static instructions | 1128 | 1094 | 0.970 |
| `s_nop` executed per dispatch | 1216-1456 | 464-624 | about 0.4 |
| ug-stage wall per CTA, per MFMA issued (median CTA) | 17.3-19.4 | 13.4-15.3 | about 0.78 |
| ug share of the K2 code latency | 0.374-0.415 | 0.335-0.409 | - |

**Mechanism.** Without `op_sel`, every scaled MFMA reads a scale VGPR that a VALU shift / mask has just written.
The hazard padding (`s_nop`) plus the VALU issue sit inside ug's MFMA chain: the P-L3 gfx950 fact, here at the
port's ug. `op_sel` picks the byte from the LDS word, so both go away. ug is 0.37-0.42 of the K2 code latency at
6 rows, so a sub-1 % K2 gain fits.

The per-MFMA wall figure mixes routing-dependent work across steps (the MFMA counts per dispatch differ). It is
indicative only; the rig's same-state A/B (0.6) is the measurement.

**Ledger items behind the built knobs:**

- **rbar.** `rbar=0` removes the barrier that the PORT-K2-moe ATT ledger counted as 0.23 idle waves at 24 rows, and
  it ties. That confirms the idle waves are off the critical path: route's wave 0 (poll, then top-k) is the latency,
  and the other 7 waves have nothing to do until ug's own wait.
- **Holds and priority.** Both tie. With one 8-wave CTA per CU, 2 waves share a SIMD, and they run the same stage,
  so `s_setprio` has nothing to arbitrate. The Qwen evidence agrees ("no effect with one CTA per CU").
- **ug_ra.** ug_ra's loss at 48 rows is the one recorded in PORT-K2-moe 4.2. Past one tile ug is issue-bound on its
  weight loads (VMEM issue stall 0.97), and read-ahead adds LDS / VGPR pressure in front of that stream. It stays a
  default-off knob, "rejected on base parity port" (two nodes, this round).
- **Ties are not losses**, so the restructure rule does not trigger for them. The three losses (`ug_ra` at 48 rows,
  `xcd` and `pfx` at 6 rows, both nodes) are older WP4 / port levers. `ug_ra` has its ATT diagnosis (above). `xcd`
  and `pfx` are recorded **LOSS, diagnosis pending** on this base, not "rejected", until round 1 has run their ATT
  and one structurally different restructure. The candidates for that restructure: `xcd` homing by token tile
  instead of by expert, and `pfx` issuing only the first K step of wo_b ahead.

### 0.8 Round 0 ledger

| move | base | ratio (6 / 24 / 48 rows; n14-05 \| n07-17) | verdict | WaveScope delta |
|---|---|---|---|---|
| ug `ug_sop=1` | K2-moe port | 0.995 / 1.001 / 0.999 \| 0.995 / 0.999 / 1.000 | **WIN at 6 rows, both nodes. Accepted: new base** | ug: static instructions 0.97, `s_nop` about 0.4, wall per MFMA about 0.78 |
| `rbar=0` (new) | port | 1.001 / - / 0.999 \| 1.002 / - / 0.999 | TIE | the 0.23 idle-wave barrier is off the critical path |
| `pprio` = 1 / 3 (new) | port | 0.995-0.998 \| 0.996-1.002 | TIE | not taken (tie) |
| `prio.<stage>=3` x13 (new) | port | 0.994-1.007 \| 0.991-1.003 | TIE | not taken (2 waves a SIMD run the same stage) |
| `pl.<stage>` x18 (new) | port | 0.994-1.005 \| 0.993-1.007 | TIE | not taken |
| `hold.<stage>` = 4 / 16 x26 | port | 0.989-1.009 \| 0.996-1.004 | TIE (no Coz slope) | not taken |
| `xcd` | port | 1.013 / - / 1.001 \| 1.016 / - / 1.003 | LOSS at 6 rows, both nodes; diagnosis pending | round 1 |
| `pfx` | port | 1.008 / - / 0.996 \| 1.010 / - / 0.997 | LOSS at 6 rows, both nodes; diagnosis pending | round 1 |
| `ug_ra=1` | port | 1.001 / - / 1.039 \| 1.005 / - / 1.038 | LOSS at 48 rows; rejected on base parity port (ATT: VMEM-issue-bound ug) | PORT-K2-moe 4.2 |
| `askip`; `C` | port | 0.999 / 0.999 / 0.997 \| 1.002 / 1.000 / 0.996; `C` 0.996 at 48 both | TIE (retained candidates); `C` HUNG once on n07-17 | round 1 bisect |
| 4 lookback combinations (`A`, `C`, `F`, `G`) and the interaction screens | port | `F` / `G` 0.993-0.997 at 6 rows = `ug_sop` | contained in the accepted move | - |
| Rust direct per stage | - | not applicable in place | open: needs device-function emission or S1 whole-K2 | - |

### 0.9 Next (round 1)

1. **Served / graph-mode rung for the new base.** Graph-mode K2 step critical path (HTL) and the C1 / C4 served
   window, with A-vs-A twins, both nodes. The rig is eager single-launch K2.
2. **Bisect `C`'s intermittent hang**, one lever at a time, with repeated launches and deadline polls.
3. **ATT diagnosis plus one restructure each for the `xcd` and `pfx` losses.** For `xcd`: tile-homed instead of
   expert-homed. For `pfx`: only the first wo_b K step issued ahead.
4. **Re-climb on the new base** (decision 15: every reject is re-tested after an accepted move). Order as a knob:
   ug / down knobs first, then the schedule surface, then a seeded random order. Score against the median of 4
   base copies with the build order shuffled. The climb's single `ref_base` normalizer was biased: the first-built
   copy ran 0.3-0.5 % slow at 48 rows.
5. **Widen the surface where V4.1 has headroom.**
   - The decision-10 items not built yet: `spread` (more CTAs on the serial seam stages), per-stage CTA counts, and
     the issue order in down / shared.
   - Playbook idea 2 (seam hop fusion) and idea 5 (slot-readiness down).
   - The rig showed the schedule surface flat to +-0.5 %. The remaining K2 time is in ug's weight stream and the
     seams' hops, not in contention.
