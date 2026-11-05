# W15 PORT K2-moe: ATOM's K2b routed MoE + shared expert as rocKE stage functions

W15, 2026-11-04. MI355X (gfx950), TP2 no-EP, DSpark 5, the B0 recipe (bf16 KV, fp8 index plane, `--level 0` + FULL
graphs). Ratios only. Absolute numbers, raw ATT / PMC / timeline files and job logs are in
`hackathon-prep/private-perf/dsv41-mono/w15/PORT-K2-moe-abs.md` and `.../w15/raw/k2moe/`.

This continues the port that the first W15 port agent left after the 2026-11-04 network drop (PLAN decision 14). Its
code (`stage_moe.py`, the ATOM hook `rk_extern_moe.py`, the rig) and its first gate job were kept; nothing was redone.

## 0. Verdict

- **Bitwise.** The whole K2b group runs as rocKE: `xq`, `route`, `ug` (one tile, past one tile, token parallel),
  `down` (one / two tasks, token parallel), `reduce`, the shared expert (one tile) and the shared unit past one tile
  (`shm`). In ATOM's served engine it is **bitwise** with ATOM's FlyDSL stages on every K2 launch: 3 DSpark seeds
  plus the native path, every served width (draft 5 / 10 / 15 / 20 / 25 / 30 / 35 / 40 rows, target 6 / 12 / 18 /
  24 / 36 / 42 / 48, native 1 / 4), **on both nodes** (§2). Every byte a launch writes is compared, scratch included,
  with LDS poisoned between reference and candidate.
- **Mutants.** Each stage has must-fail mutants, and all of them are caught. The first gate found that the `reduce`
  `col` mutant was equivalent (it rotated the read and the store column together). It was rewritten to move only the
  store. The new mutant is in the §2 table.
- **Knob-off identity.** With the environment unset, K2 `.pre.ll` and `.text` are byte-identical to pristine ATOM
  (`df86c482`) at s6 / s12 / s24 / s48 and draft s5 / s20 / s40, on both nodes. The ATOM tree checked is the
  current `dev/ATOM-dsv41`, which also holds the K2-attn-gemv agent's hooks.
- **In situ, the parity port is a tie at 6 rows and pulls ahead with width (§3).**
  - In-place eager A/B of the K2 launch from the same state, against a recompiled no-op control:
    - 6 rows: 1.000;
    - 12 rows: 0.995;
    - 24 rows: 0.995;
    - 48 rows: 0.987.
  - Graph-mode GPU decode step (HTL, one pass, n01-09): C1 1.004, C2 0.996, C4 0.995, C8 0.986. The A-vs-A twin
    is at 0.999-1.002 and the no-op control at 0.995-1.001.
  - Confirmation passes on the second node are §8 item 1.
- **The first structural levers already move the port below parity (§3.2).** Measured in the same rig against the
  no-op control:
  - `r_max=1` (one (task, slot) block per wave a round in `down`): about 0.989 of ATOM's K2 at 6 rows, 0.992 at 12,
    0.993 at 24 and 0.991 at 48.
  - `pslot` (poll each slot's ug flags right before its MFMAs), `sop` and `sh_first=0`: 0.993-0.997.
  - The combination screen (`c1b`) is in §3.2.
- **WaveScope ledger of OUR code (§4).** The K2b group is weight-stream bound where the decode step spends its time:
  - `ug` waits on its MXFP4 weight loads (the MFMA that consumes them carries 0.41 of the rocKE stages' traced
    latency at 6 rows);
  - `down`'s `ld` carries 0.16-0.18; its VMEM issue stalls are 0.43-0.46 of its issued cycles.
  - The bubbles on top of the stream: `down`'s ug-flag poll (0.08-0.09), barriers (route's trailing barrier 0.12 at
    24 rows; ug / down / shared 0.02-0.05 each) and, in ATOM's own build, the shared expert's and ug's barrier waits.
- **COMGR vs Rust direct, auto-select: COMGR on every K2b stage today (§5).** Three direct-compiler defects were found:
  - `fcmp uno` refused in `xq`: **fixed** in the crate, with a unit test; `xq` now compiles direct.
  - `tile.wave_ballot` not lowered in `route`: open.
  - A **miscompile** in `down`: direct-built `down` differs from COMGR's at every shape. A bisect job is queued (§5).

  `reduce` is bitwise direct; its timing is pending. Per decision 18 every one of these is a compiler defect to fix,
  not a rejection.
- **The knob surface (§6).** Every hard-coded choice of the port is a knob with default = parity, 13 of them.
  This round added `ug_sop` / `ug_ra`.

## 1. What is ported

| ATOM stage (`moe.py`) | rocKE function (`dsv41_mono/`) | extern stage name | builds served |
|---|---|---|---|
| `stage_xq` | `stage_moe.build_xq_kernel` | `x8q` (file `xq_c<cm>`) | every width |
| `stage_route` | `stage_moe.build_route_kernel` | `route` | every width, target 384 / top-6, draft 128 / top-3 |
| `stage_ug` (one tile, not token parallel) | `stage_ug.build_ug_kernel` (WP4 4.3) | `ug` (`rk_extern`) | 1-16 rows |
| `stage_ug_tile` / `_rk_ug_xq` (past one tile, token parallel) | `stage_ug` under `ug_n<S>g<G>` | `ugx` | 12-48 rows |
| `stage_down` (one / two tasks, token parallel) | `stage_moe.build_down_kernel` | `down` (`t1` / `t2`, `p`) | every width |
| `stage_reduce` | `stage_moe.build_reduce_kernel` | `reduce` | every width (own rows when token parallel) |
| shared expert, one tile | `stage_shared` (WP4 4.2) | `shared` / `shx` (token parallel) | 1-16 rows |
| shared unit past one tile | `stage_moe.build_shm_kernel` | `shm` | 17-48 rows |

- **Hook.** ATOM side: `atom/models/deepseek_v41/mono/kernels/rk_extern_moe.py` (new). It adds 7 extern stages to
  `rk_extern`, plus Python-level call sites in `moe.py` and one block in `layer_post.py`.
- **Turning it on.** `ATOM_DSV41_RK_EXTERN="x8q=D;route=D;down=D;reduce=D;ugx=D;shx=D;shm=D;ug=D;shared=D"`.
  Unset, nothing is traced: every build's IR, key and kernel name are ATOM's.
- **Generating the functions.** The CLI is
  `python -m kernels.gfx950.dsv41_mono.stage_moe --out D [--stages ..] [--knobs ..]`, together with `stage_shared`.
  A full set is 795 functions; generation takes about 1-2 min on the node.

## 2. Gates

The in-place gate is `ATOM_DSV41_RK_LEV_CHECK`, gate kind `ext`. On every eager K2 launch it does the following:
- run ATOM's build;
- restore the state, with LDS poisoned;
- run the extern-linked build;
- compare every written byte: peers, row buffers, `ffn_out` and the whole mailbox scratch.

Arms: DSpark GSM8K groups at C1 / C2 / C4 / C8, 48 tokens per request, plus native rows 1 / 4.

| node | tree | arm | K2 launches a rank | DIFF launches |
|---|---|---|---|---|
| n01-09 | a1 (no `shm`: it ran before `shm` was added) | seeds 0 / 1 / 2 | 12,382 / 12,216 / 12,382 | 0 / 0 / 0 |
| n01-09 | a1 | native r0 | 19,200 | 0 |
| n01-09 | a3 + rocKE a4 (all stages incl. `shm`) | seeds 0 / 1 / 2 | 12,136 / 12,652 / 12,652 | 0 / 0 / 0 |
| n01-09 | a3 + a4 | native r0 | 19,200 | 0 |
| n07-17 | a3 + a4 | seeds 0 / 1 / 2, native | PENDING_J1B | PENDING_J1B |

Widths covered by the seeds: K2 5 / 6 / 10 / 12 / 15 / 18 / 20 / 24 / 25 / 30 / 35 / 36 / 40 / 42 / 48. These are
all target and draft widths that reach K2 in these runs. The native path covers 1 and 4.

**Must-fail mutants.** Each mutant arm swaps one stage's function set for a mutant. A mutant is caught if it hangs or
if any launch differs.

| mutant | stage | what it breaks | n01-09 (a1) | n01-09 (a3) | n07-17 (a3) |
|---|---|---|---|---|---|
| `put_bit` | xq | one bit of a written word | caught (3,731 diff launches, hang) | PENDING | PENDING |
| `code` | xq | an MXFP8 code | caught (3,774, hang) | - | - |
| `put_bit` | route | the route weight bits | caught (3,857, hang) | - | - |
| `idx` | route | the expert id | caught (20, hang) | PENDING | PENDING |
| `put_bit` | down | the pushed pair | caught (3,943, hang) | PENDING | PENDING |
| `chain_drop` | down | one MFMA K step | caught (3,943, hang) | - | - |
| `scale_byte` | down | the E8M0 scale byte | caught (3,903) | - | - |
| `row` | down | a token row | caught (3,767, hang) | - | - |
| `shared_drop` | down | the shared expert's add | caught (4,069) | - | - |
| `topk_order` | down | top-k add order | caught (3,808, hang) | PENDING | PENDING |
| `put_bit` | reduce | the reduced word | caught (4,112, hang) | PENDING | PENDING |
| `col` (first form) | reduce | rotate read and store column | **NOT caught (0 / 8,058): equivalent mutant** | - | - |
| `col` (fixed: store column only) | reduce | the sum lands one column pair over | - | PENDING | PENDING |
| `put_bit` | ug (one tile) | the MXFP8 mid word | caught (1,635, hang) | - | - |
| `put_bit` | ugx (past one tile) | the same | caught (2,053, hang) | PENDING | PENDING |
| `put_bit` | shared (one tile) | the shared mid | caught (1,758, hang) | - | - |
| `put_bit` | shx (token parallel) | the same | caught (480, hang) | PENDING | PENDING |
| `put_bit` | shm (past one tile) | the shared mid | - | PENDING | PENDING |
| `row` | shm | a token row | - | PENDING | PENDING |

**Knob-off identity.** pristine `df86c482` vs the tree (`wp4_run.sh dump`). `.pre.ll` and `.text` are SAME at K2 s6 /
s12 / s24 / s48 and draft K2 s5 / s20 / s40, on:
- n01-09 with tree a1;
- n07-17 with tree a3;
- n01-09 with tree a3.

The extern-on builds link `rocke_dsv41_*` symbols, and the kernel names carry `_rk<digest>` suffixes.

## 3. In situ, against ATOM

### 3.1 The parity port

- **In-place eager A/B** (`ATOM_DSV41_RK_BENCH`, job t1, n01-09). Each K2 launch is run as ATOM's build and as each
  variant, from the same state, 3 shuffled rounds, every layer of 4 timed steps per width.
  - The number is the per-step sum over layers of variant / ATOM, median over steps and both ranks.
  - `nop` is a recompiled no-op lever build, the compile-lottery control (rule 260).

  | rows | `nop` | `all` (the full port) | `all` / `nop` | `down` only | `ug` only | `sh` (shared, shx, shm) | `xr` (xq + route) | `red` |
  |---|---|---|---|---|---|---|---|---|
  | 6 | 1.002 | 1.002 | 1.000 | 0.998 | 0.999 | 0.994 | 0.998 | 0.998 |
  | 12 | 1.004 | 0.999 | 0.995 | 1.002 | 1.006 | 0.998 | 1.003 | 0.999 |
  | 18 | 1.004 | 1.001 | 0.997 | 0.993 | 0.993 | 1.001 | 0.999 | 0.998 |
  | 24 | 1.000 | 0.996 | 0.995 | 0.993 | 0.994 | 0.996 | 0.999 | 0.997 |
  | 48 | 1.001 | 0.988 | 0.987 | 0.993 | 0.995 | 0.995 | 0.998 | 0.998 |

- **Graph-mode GPU decode step** (HTL, `ATOM_DSV41_RK_HOSTTL`, FULL graphs, t1 n01-09, one pass). This is the
  decode step against the mean of ATOM and its A-vs-A twin:

  | cell | twin / ATOM | `nop` | port (`m1`) |
  |---|---|---|---|
  | C1 | 0.999 | 0.999 | 1.004 |
  | C2 | 1.002 | 1.001 | 0.996 |
  | C4 | 1.000 | 0.995 | 0.995 |
  | C8 | 1.002 | 1.000 | 0.986 |

  The job's 6 h limit ended pass 2 after its first arm, so this is one pass on one node.
- **Graph-mode stage timeline** (`ATOM_MONO_TIMELINE` stamps, the stage fronts of one replay, one engine per arm, no
  twin). K2 span ratio port / ATOM:

  | rows | 6 | 12 | 18 | 24 | 42 | 48 |
  |---|---|---|---|---|---|---|
  | ratio | 1.000-1.002 | 0.953 | 0.966 | 1.049 | 1.006 | 0.970 |

  Without a twin these are indicative only. The span swings ±5 % between widths, while the in-place A/B above (same
  state, controls) agrees within 0.5 %. The graph-mode A/B with twins on both nodes is §8 item 1.

### 3.2 Structure levers (first screen)

Same rig and same controls (t1). Only the `down` function set is swapped in each variant, and the numbers are
relative to `nop`:

| variant | 6 | 12 | 18 | 24 | 48 |
|---|---|---|---|---|---|
| `dr1`: `r_max=1` | **0.989** | **0.992** | 0.991 | **0.993** | **0.991** |
| `dps`: `pslot=1` | 0.992 | 0.995 | 0.992 | 0.994 | 0.992 |
| `dpssm`: `pslot=1,smap=1` | 0.993 | 0.995 | 0.990 | 0.993 | 0.993 |
| `dsop`: `sop=1` | 0.993 | 0.994 | 0.991 | 0.994 | 0.991 |
| `dsh0`: `sh_first=0` | 0.994 | 0.995 | 0.992 | 0.996 | 0.993 |

- **Reading.** Every one of these structural changes to the down stage beats ATOM's down at every width. `r_max=1`
  is the strongest. It halves the (task, slot) blocks a wave holds per round, so each round's weights and ug-flag
  poll are on the critical path half as long, and the down stage has more rounds of shorter latency chains.
- **The combination screen** (job `c1b`):
  - `r_max=1` × {`pslot`, `sop`, `smap`, `sh_first=0`, `pslot+sop`};
  - `pslot+sop`, `wfirst=0` (the must-lose issue-order control), and the ug levers `ug_sop` / `ug_ra`;
  - whole-port combinations `c1` (port with down@`r_max=1`) and `c2` (+ ug@`ug_sop,ug_ra`).

  Results (job `c1b`, n01-09, DSpark seed 0, the same in-place rig). Each cell is variant / ATOM over `nop` /
  ATOM. This job's `nop` sat at 1.001-1.008, the compile lottery.

  | variant | 6 | 12 | 18 | 24 | 48 |
  |---|---|---|---|---|---|
  | `all` (parity port) | 0.998 | 0.984 | 0.997 | 0.990 | 0.987 |
  | `c1` = port, down@`r_max=1` | 0.992 | 0.985 | 0.992 | 0.988 | 0.987 |
  | `dr1pssop` (down only: `r_max=1,pslot=1,sop=1`) | 0.990 | **0.981** | 0.991 | 0.987 | 0.991 |
  | `dr1sm` (down only: `r_max=1,smap=1`) | 0.993 | **0.980** | 0.992 | 0.988 | 0.991 |
  | `dr1ps` | 0.992 | 0.989 | 0.992 | 0.987 | 0.992 |
  | `dr1sop` | 0.992 | 0.982 | 0.991 | 0.990 | 0.992 |
  | `dr1sh0` | 0.992 | 0.983 | 0.993 | 0.991 | 0.992 |
  | `dpssop` | 0.995 | 0.986 | 0.991 | 0.988 | 0.994 |
  | `ugs` (ug@`ug_sop=1`) | 0.991 | 0.989 | 0.990 | 0.989 | 0.993 |
  | `ugr` (ug@`ug_ra=1`) | 0.995 | **1.007** | **1.019** | **1.018** | **1.024** |
  | `c2` = `c1` + ug@`ug_sop,ug_ra` | 0.989 | 1.003 | 1.013 | 1.012 | 1.015 |
  | `dwf0` (down `wfirst=0`, expected to lose) | 0.994 | 0.988 | 0.994 | 0.990 | 0.995 |

  Reading of the screen:
  - `ug_ra` (LDS read-ahead of the activation operand) **loses** from 12 rows on, by up to 2.4 %, and it drags `c2`
    with it. It is kept as a default-off knob, recorded "rejected on base parity port, n01-09". The WaveScope ledger
    says why: past one tile, ug is issue-bound on its weight loads (`stage_ug.py:153`, VMEM issue stall 0.97), and
    read-ahead adds LDS / VGPR pressure in front of the same stream.
  - `wfirst=0` was planned as the must-lose issue-order control, but it does **not** lose. The down stage's ug-flag
    poll is already satisfied when a round starts at these widths, so issuing the weights before or after it makes
    no difference. This is why `pslot` gains only a little: the ug -> down latency chain is mostly hidden behind ug's
    own stream.
  - The best singles stack only partly. `dr1pssop` and `dr1sm` give about 0.98 of ATOM's K2 at 12 rows and about
    0.99 elsewhere. The c1b `all` / `c1` / `dr1*` spread across widths (0.98-0.99 at 12 / 24 / 48) is about 1 %, and
    this job's `nop` (1.001-1.008) moved by the same amount, so round 2 re-tests on both nodes with another seed.
  - **Lookback round 2** (jobs `r2-n01-09`, `r2-n07-17`, seed 1). The arms are `nop`, `all`, `c1`, `c3` (down@
    `r1pssop` + ug@`ug_sop`), `c4` (down@`r1sm` + ug@`ug_sop`), `c5` (down@`r1ps` + ug@`ug_sop`), `c6` (down@`r1pssop`),
    `ugs`, `dr1`, and `dwf0` re-tested. Result: PENDING_R2.

## 4. WaveScope: the bubble ledger of OUR code

### 4.1 Method (what had to be fixed first)

- **ATT in the served engine.** `rocprofv3 --att` runs on an eager offline engine (DSpark, GSM8K) with
  `--kernel-include-regex` set to the K2 build and `--kernel-iteration-range` set to 2 layers per rank. It targets
  CU 1 of SE 0 of each XCD (`--att-shader-engine-mask 0x11111111`: 8 CTAs per dispatch, one per XCD). It runs 4 arms:
  ATOM's build and ours, at 6 and 24 rows.
- **The decoder kept only the waves.** In the first ATT run (t1), every K2 dispatch kept its waves but had no
  `code.json` / `filenames.json`. The cause: rocprofv3 writes these last, in each TP rank's finalization, and
  `llm.close()` SIGTERMed the ranks after ATOM's `ATOM_SHUTDOWN_TIMEOUT_S` (default 5 s).
  - Fix 1: the rig now sets `ATOM_SHUTDOWN_TIMEOUT_S=1200` for profiled runs. The next run (j1b) got the decoder's
    own `code.json`.
  - Fix 2: `att_nocode.py` rebuilds `code.json` from the waves and the dumped code object, using row = instruction
    ordinal. It was checked against a decoded dispatch that has the decoder's `code.json`: 129 / 129 rows equal in
    hits, latency, stall and idle. Every rebuilt K2 dispatch passes its own type check: every ATT instruction type
    agrees with its row's mnemonic class, 1.0000 over 0.35-0.57 M instructions.
- **Labels.** Stages come from the P2 stage-marker debug build (`rv_isa split`). Its `.text` matches the dispatched
  code object at 0.99996-0.99997. Source lines of our rocKE functions come from `ROCKE_DEBUG_LOC=1` builds via
  `llvm-symbolizer` (`w15_att.py`).

### 4.2 Per-stage ledger

Rows are the K2b stages; dispatches are pooled. Column meanings:
- `ours/ATOM busy`: per traced CTA, the issued-instruction cycles of the stage's own code, waits and polls excluded;
- `exec / wait / stall`: wave-state shares of the stage's span;
- `poll / bar / vm / vld / valu / mfma`: shares of its issued cycles;
- `waves`: active waves of a CTA in the stage.

Each node's ledger is 4 dispatches × 8 CTAs per arm.

6 rows (j1b, n07-17):

| stage | ours: exec / wait / stall | ours: poll / bar / vm / vld / valu / mfma | ATOM: exec / wait / stall | ATOM: poll / bar / vm / vld / valu / mfma | ours/ATOM busy |
|---|---|---|---|---|---|
| ug | 0.03 / 0.97 / 0.01 | 0.11 / 0.82 / 0.04 / 0.00 / 0.00 / 0.00 | 0.03 / 0.95 / 0.01 | 0.11 / 0.80 / 0.04 / 0.01 / 0.01 / 0.00 | 0.730 |
| down | 0.12 / 0.41 / 0.47 | 0.05 / 0.12 / 0.25 / 0.45 / 0.04 / 0.01 | 0.15 / 0.41 / 0.44 | 0.00 / 0.12 / 0.27 / 0.43 / 0.09 / 0.01 | 0.909 |
| reduce | 0.30 / 0.66 / 0.04 | 0.83 / 0.00 / 0.00 / 0.00 / 0.02 / 0.00 | 0.18 / 0.80 / 0.02 | 0.88 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00 | (tiny stage) |
| route | 0.19 / 0.77 / 0.03 | 0.00 / 0.81 / 0.04 / 0.00 / 0.02 / 0.00 | 0.82 / 0.00 / 0.18 | (a different CTA role was traced) | (role-dependent) |

24 rows (j1b, n07-17):

| stage | ours: exec / wait / stall | ours: poll / bar / vm / vld | ATOM: exec / wait / stall | ATOM: poll / bar / vm / vld | ours/ATOM busy |
|---|---|---|---|---|---|
| ug | 0.12 / 0.45 / 0.43 | 0.01 / 0.15 / 0.22 / 0.42 | 0.17 / 0.66 / 0.18 | 0.02 / 0.10 / 0.42 / 0.15 | (role mix: see text) |
| down | 0.13 / 0.43 / 0.44 | 0.01 / 0.13 / 0.27 / 0.44 | 0.15 / 0.39 / 0.46 | 0.00 / 0.15 / 0.22 / 0.46 | 0.898 |
| route | 0.09 / 0.90 / 0.01 | 0.00 / 0.91 / 0.03 / 0.01 | 0.19 / 0.79 / 0.02 | 0.00 / 0.86 / 0.05 / 0.01 | |
| shared / shm | 0.16 / 0.70 / 0.14 | 0.01 / 0.29 / 0.25 / 0.11 | 0.15 / 0.78 / 0.07 | 0.06 / 0.53 / 0.21 / 0.04 | |
| reduce | 0.29 / 0.69 / 0.02 | 0.90 / 0.00 / 0.00 / 0.00 | 0.17 / 0.82 / 0.00 | 0.95 / 0.00 / 0.00 / 0.00 | |

**Ranked bubbles of our code.** Share of the rocKE functions' traced latency, by source line, summed over 4
dispatches per width:

| rank | 6 rows | 24 rows |
|---|---|---|
| 1 | `stage_ug.py:208` ug MFMA (waits on its MXFP4 weight loads, stall 0.98): **0.41** | `stage_ug.py:153` ug weight loads (VMEM issue stall 0.97): **0.22** |
| 2 | `stage_moe.py:342` `ld` (down weights / scales, stall 0.96): 0.18 | `stage_moe.py:342` `ld`: 0.16 |
| 3 | `stage_moe.py:365` `poll_pairs` (down's ug-flag poll): 0.09 | `stage_moe.py:527` route's trailing barrier: **0.12** |
| 4 | `stage_moe.py:775` down MFMA: 0.06 | `stage_ug.py:208` ug MFMA: 0.09 |
| 5 | `stage_ug.py:175` ug scale word LDS read: 0.06 | `stage_moe.py:365` down's ug-flag poll: 0.08 |
| 6 | down / ug / shared / route barriers (`stage_wqkv.py:163`): 0.03 / 0.02 / 0.02 / 0.01 | down MFMA 0.06, ug barrier 0.05, ug scale read 0.04, down barrier 0.03 |

- **Weight stream.** The K2b group is the weight stream. Ranks 1-2 (0.59 of the traced rocKE latency at 6 rows, 0.38
  at 24) are waits on, and issue stalls of, the expert weight loads. This is the byte floor that WP4 / P1 measured.
- **Bubbles on top of the stream:**
  - down's ug-flag poll (the latency chain ug -> down);
  - route's trailing barrier at 24 rows;
  - the per-stage `s_barrier`s.

  These are what the structure levers attack: `pslot` the poll, `r_max=1` the round length, `sh_first` the order.
- **Busy work.** Our `down` issues 0.90-0.91 of ATOM's busy cycles per CTA:
  - VALU 0.46-0.58 of ATOM's;
  - MFMA and loads equal;
  - VMEM wait 0.96-1.21.

  So the port is not slower at the instruction level. Its wins come from the same stream with less overhead.
- **Caveat on the per-CTA ledger.** ATT traces one CU per XCD, and the CTA role that CU gets (which ug / down / route
  tasks) varies between dispatches. The span-based columns of small stages (route, xq, reduce, the share of ug at
  24 rows) therefore swing with the role. The code-row columns (busy, class shares) are stable. The n01-09 ledger
  (j3, same arms) is pooled in when it lands (§8).

**PMC** (t1, `rocprofv3 --pmc`, every K2 dispatch of the arm, median per dispatch, ours / ATOM).
- Instruction counts: MFMA 1.002-1.018 (the same work), VALU 0.81-0.83, LDS 0.93-0.95, SALU 1.03-1.07.
- The wave-cycle counters are not usable here. Under `--pmc` the dispatches serialize and K2's TP waits dominate
  (6 rows: 0.48, 24 rows: 1.04, with no consistent sign).
- `FETCH_SIZE` / `WRITE_SIZE` hang the profiled engine (rc 124 / 137). The rig drops that group.

## 5. COMGR vs Rust direct, per stage

Standalone rig: `srig.py`. Each stage body is a standalone kernel on one GPU, with synthetic but well-formed
mailboxes and route tables. Arms:
- C: COMGR (torch 2.12's, llvm22);
- C2: C's bytes loaded again;
- R: COMGR recompiled;
- D: the Rust direct compiler;
- Dm / Db: direct with the booked lever sets.

Numerics compare every arm to C; then paired, shuffled graph timing.

| stage | direct status | numerics vs COMGR | timing vs C (median of 24 shuffled rounds; controls C2, R) | auto-select |
|---|---|---|---|---|
| `xq` | was refused (`fcmp uno`): **fixed** | **bitwise** (D, Dm, Db; 8 inputs, twice) | n01-09: D **1.127** (loss), Dm **0.958**, Db 0.959 (wins); C2 0.998, R 0.998. n07-17: PENDING_S5B | **direct + Dm** once n07-17 confirms (COMGR until then) |
| `route` | was refused (`tile.wave_ballot`, then f32 `math.sqrt`): **both fixed** | PENDING_S5 (6 / 48 rows, draft 5) | PENDING_S5 | COMGR until measured |
| `down` | compiles | **FAIL** at every shape, with or without spills (see below) | - | COMGR (defect open) |
| `reduce` | compiles | **bitwise** at 6 and 24 rows (D, Dm, Db) | D 0.996 / 0.998, Dm 1.000 / 0.999, Db 1.002 / 1.002 against R 0.995 / 1.000: **tie** | COMGR (direct only where it wins) |
| `ug`, `shared`, `shm` | not yet in the standalone rig | - | - | COMGR |

The direct-compiler changes of this round are in `platform/rust/rocke-codeobj`, the compiler core (allowed under
decision 18). Each one has a unit test. The crate's own suite passes on Linux (build `m3`, CPUONLY: 498 / 498) and
locally (494 + 4 Windows-skipped), and `isa_tables.py check --isa gfx950` is clean:
- **`fcmp uno` and the other unordered / ordered predicates.** `isel.rs` maps `ord`, `uno`, `ult`, `ueq`, `ule`, `ugt`
  and `uge` to the gfx9 VOPC f32 codes 0x17-0x1e. Test: `unordered_fcmp_predicates_select_vopc_codes`. `xq` uses
  `uno` for ATOM's NaN-propagating max / min.
- **`tile.wave_ballot`.** A new `OpKind::WaveBallot` sits next to `WaveAny` / `WaveAll` in every cross-lane list
  (`ast`, `analysis`, `opt`, `lane_cse`), with an i32 -> i64 validation rule.
  - Selection: the vote's lane mask, AND exec, copied to an SGPR pair (wave32: high half 0). The single-use
    `select(c, 1, 0)` operand is fused, so the compare's own mask is used, as in LLVM.
  - Test: `wave_ballot_is_one_compare_into_an_sgpr_pair`, on gfx1151 and gfx950.
- **f32 `math.sqrt` (gfx950).** This is LLVM's correctly rounded expansion with IEEE denormals (`lowerFSQRTF32`):
  - 2^32 pre-scale below 2^-96;
  - `v_sqrt_f32`;
  - the two neighbour tests `fma(-s', s, x)`;
  - 2^-16 post-scale;
  - `v_cmp_class` keeps +-0 / +inf.

  It needed a new gfx9 ISA row `v_sqrt_f32` (VOP1 0x27) in `isa/gfx950.isa`, with `gen_isa.rs` regenerated. The
  sequence matches COMGR's instruction for instruction in `route` (checked in the disassembly). `route` now compiles
  direct at 6 and 48 rows (72 VGPRs, the same as COMGR, no spills).

**The `down` miscompile (open, bisected).**
- The bisect (`s3`, n01-09) has the same result in every variant:
  - every payload word of every token and column differs from COMGR's;
  - the tag words are right;
  - it fails with the base build (direct spills 34 VGPRs), with `r_max=1` and `pslot=1` (no spills), and with
    `shared_drop` (no shared expert path).
- So it is neither the spill path nor the shared path: the routed fp4 x fp8 scaled-MFMA path is wrong.
- The COMGR reference itself contains +-inf words with these synthetic weights. `s4` (queued) poisons LDS between
  launches with two patterns, to check that the body writes every LDS word it reads in the standalone harness. If
  COMGR's output moves with the poison, the harness has to stage the inputs that ATOM's earlier stages leave in LDS.
- The two compiles' scaled-MFMA lines agree in shape (11 MFMAs, `cbsz:4` on the routed ones, same `op_sel_hi`).
  Next is an operand-level diff of the routed MFMA chain: scale registers, A-operand tuples, `vec_concat` of the
  mid words.
- Also seen: direct holds 1.5-2.1x COMGR's VGPRs on this body (`r_max=1`: 160 vs 76). That is a register-allocation
  defect of its own, and it would cost occupancy even once `down` is correct.

The Qwen MoE r1 won with direct (0.970-0.976). On V4.1 the direct compiler first has to compile these bodies
correctly. That work is in the compiler core and is allowed under decision 18.

## 6. Knob surface (every hard-coded choice of the port; default = ATOM / parity)

| knob | stage(s) | values | default | class | built | screened |
|---|---|---|---|---|---|---|
| `nan` | xq, route | prop / num | prop | bitwise except NaN inputs | yes | no |
| `pause` | all polls | 0 / 1 (`s_sleep 1`) | 1 | bitwise | yes | no |
| `poll_max` | all polls | 1-16 pairs a round trip | 12 | bitwise | yes | no |
| `w_cpol` | ug, down weight streams | cache bits (nt 2, plain 0, sc1 16) | nt | bitwise | yes | no |
| `red_bar` | down / reduce reduction barrier | wg / lds | wg | bitwise | yes | no |
| `sop` | down | 0 / 1 (scale byte by op_sel, ATOM's `dsop`) | 0 | bitwise | yes | §3.2 |
| `r_max` | down | 1 / 2 (task, slot) blocks a wave a round | 2 | bitwise | yes | §3.2 (best single) |
| `sh_first` | down | 1 / 0 (shared expert before / after the rounds) | 1 | bitwise | yes | §3.2 |
| `wfirst` | down | 1 / 0 (round weights before / after the ug-flag poll) | 1 | bitwise | yes | c1b (must-lose control) |
| `pslot` | down | 0 / 1 (each slot polls its own ug flags right before its MFMAs) | 0 | bitwise | yes | §3.2 |
| `smap` | down | 0 / 1 (a wave's slots contiguous vs ATOM's wave-strided) | 0 | bitwise | yes | §3.2 |
| `chains` | down | 1 / 2 / 3 MFMA accumulators | 2 | **tolerance** (reassociated K sum) | yes | no (needs a tolerance gate) |
| `ug_sop` / `ug_ra` | ug, ugx | `stage_wqkv` levers: op_sel-packed scales; LDS read-ahead | 0 / 0 | bitwise | yes (this round) | c1b |
| Rust direct (`ROCKE_CODEOBJ_DIRECT_LIB`) + lever set | per stage | COMGR / direct / direct + levers | COMGR | bitwise when correct | yes | §5 |

Choices that stay in ATOM's orchestration (`moe.py` `run_moe`, not in the stage functions) and belong to the
decision-10 schedule surface:
- CTA allocation of ug / down / shared / reduce tasks (`BLOCKS`, task strides, `ug_groups`, `tok_par_from`);
- `ROUTE_COPIES`, `ROUTER_EARLY_FROM`, `DOWN_TASKS`, `RED_COLS`;
- per-stage holds / priorities.

The port reads them from ATOM's tree (`stage_moe.check_atom` asserts every constant and every served route-table
offset), so a schedule knob on the ATOM side needs no port change.

## 7. The rig (reusable)

Location: `hackathon-prep/dsv41/w15/rig/moe/`.

| file | what |
|---|---|
| `w15moe_job.sh`, `w15moe_job_local.sh` | raw-sbatch bodies: `/scratch` stages, or node-local `/tmp` for a node without `/scratch` |
| `submit.sh`, `release.sh` | held submit, plus a login-node watcher that keeps the W15 GPU cap |
| `n07/push.sh`, `n07/pull.sh`, `n07/seed_w15m.sh` | n07-17 after its reboot: stage push and pull through `srun`, and a reseed that writes no home files (the Austin home is over quota) |

Contents of `h/` (the in-container harness):
- `w15moe_run.sh`, phases:
  - `gen`: function sets per knob spec and mutant;
  - `ident`: knob-off identity plus extern-on resources;
  - `split`: stage-marked debug dumps plus `rv_isa split`;
  - `att`: ATT plus analysis, with the `att_nocode` fallback;
  - `pmc`;
  - `eng`: the `wp4_run.sh` phases `gate` / `bench` / `gtl` / `htl`.
- `att_nocode.py`, `w15_att.py`, `moe_ledger.py`: the ledger.
- `srig.py`, `dprobe.py`: the standalone COMGR-vs-direct rig, and the local compile probe.
- `bench_sum.py`, `wp4_htl_cmp.py`, `../gtl_cmp.py`: the in-situ summaries.

A candidate screen in the rig costs seconds per candidate per width, inside one running engine (rk_bench). A full
gate is about 12 min per arm.

## 8. Pending (jobs queued or running at the time of writing)

1. The n07-17 gate (j1b) and its mutants, and the a3 mutant set on n01-09 (j3). Then the graph-mode HTL A/B with
   twins on **both nodes**, for the port and the best `c1b` combination.
2. `c1b`: the combination screen (§3.2), then a climb with signed lookback over the §6 surface plus the decision-10
   schedule surface. That is W15 Climb.
3. `s3`: the direct-compiler `down` bisect, `xq` / `reduce` timing, and the `wave_ballot` lowering.
4. The j3 ATT ledger (n01-09), pooled with j1b.
