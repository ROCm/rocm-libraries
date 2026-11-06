# W15 PORT K2-moe: ATOM's K2b routed MoE + shared expert as rocKE stage functions

W15, 2026-11-04. MI355X (gfx950), TP2 no-EP, DSpark 5, the B0 recipe (bf16 KV, fp8 index plane, `--level 0` + FULL
graphs). Ratios only. Absolute numbers, raw ATT / PMC / timeline files and job logs are in
`hackathon-prep/private-perf/dsv41-mono/w15/PORT-K2-moe-abs.md` and `.../w15/raw/k2moe/`.

This continues the port that the first W15 port agent left after the 2026-11-04 network drop (PLAN decision 14). Its
code (`stage_moe.py`, the ATOM hook `rk_extern_moe.py`, the rig) and its first gate job were kept; nothing was redone.

## 0. Verdict

- **Bitwise on both nodes.** The whole K2b group runs as rocKE stage functions:
  - `xq`, `route`;
  - `ug`: one tile, past one tile, token parallel;
  - `down`: one / two tasks, token parallel;
  - `reduce`;
  - the shared expert (one tile) and the shared unit past one tile (`shm`).

  In ATOM's served engine it is **bitwise** with ATOM's FlyDSL stages on every gated launch (§2):
  - 3 DSpark seeds plus the native path;
  - every width that reached K2: draft 5-40, target 6-48, native 1 / 4;
  - n01-09: 0 diffs;
  - n07-17: 0 diffs in every region the port writes.

  On n07-17, rank 0 shows a rare `res_ffn0` byte (the K2a seam's output, which the port does not write). The
  ATOM-vs-ATOM control on that node shows more of the same (19 / 9 / 5 launches against the port's 3 / 2 / 1), so
  it is ATOM's own nondeterminism. Every byte a launch writes is compared, scratch included, with LDS poisoned
  between reference and candidate.
- **Mutants.** Every must-fail mutant is caught on both nodes: 10 mutants over xq, route, down, reduce, ugx, shx and
  shm (§2). The first gate found that the `reduce` `col` mutant was equivalent (it rotated the read and the store
  column together, 0 of 8,058 launches caught). It now moves only the store, and both nodes catch it.
- **Knob-off identity.** With the environment unset, K2 `.pre.ll` and `.text` are byte-identical to pristine ATOM
  (`df86c482`) at s6 / s12 / s24 / s48 and draft s5 / s20 / s40, on both nodes. The tree checked is the current
  `dev/ATOM-dsv41` with both W15 port agents' hooks.
- **In situ, the parity port ties ATOM at 6-18 rows and is about 0.5-1 % faster at 24-48 rows (§3).**
  - In-place eager A/B of whole K2 launches from the same state, port / ATOM (raw), four job-runs (t1, c1b and r2 on
    n01-09; r2 on n07-17), two seeds:
    - 6 rows: 1.002-1.005;
    - 12 rows: 0.991-0.999;
    - 18 rows: 0.998-1.002;
    - 24 rows: 0.992-0.996;
    - 48 rows: **0.988-0.992**.
  - The recompiled ATOM control (`nop`) sits at 1.000-1.014, and once at 1.059 (n07-17, 6 rows): that is the size of
    the compile lottery these numbers live in.
  - Graph-mode GPU decode step (HTL, n01-09, one pass): C1 1.004, C2 0.996, C4 0.995, C8 **0.986**. The twin is at
    0.999-1.002 and `nop` at 0.995-1.001.
  - ATT, pooled over both nodes: the traced K2b stage spans are **0.954** (6 rows) and **0.945** (24 rows) of ATOM's,
    and down's own busy cycles 0.885-0.907.
  - The served confirmation is not done yet (§8).
- **Structure levers (§3.2).** All built, all bitwise, screened in the rig:
  - Single levers on the down stage beat ATOM's down at every width: `r_max=1` 0.989-0.993 of ATOM's K2, `pslot` /
    `sop` / `sh_first=0` / `smap` 0.990-0.996.
  - Combinations: the c1b round has `dr1pssop` / `dr1sm` at 0.980-0.981 of `nop` at 12 rows. The lookback re-test
    (r2, seed 1, **both nodes**) does not hold that up: every whole-port combination (`c1`, `c3`-`c6`) is at
    0.993-1.007 of the parity port itself, inside the compile-lottery noise. No combination is adopted. All of them
    stay as knobs.
  - `ug_ra` loses up to 2.4 % from 12 rows on. It is kept as a default-off knob, "rejected on base parity port".
  - Further structure needs the decision-15 search engine's Race / NoiseModel, and the two idle-wave barriers the
    ledger found (route, shm).
- **WaveScope ledger of OUR code (§4).** The K2b group is weight-stream bound:
  - ug's MFMA waits on its MXFP4 loads (0.40 of the rocKE stages' traced latency at 6 rows);
  - down's `ld` takes 0.10-0.18;
  - VMEM issue stalls are 0.42-0.46 of down's and ug's issued cycles.

  On top of the stream there are three kinds of bubbles:
  - down's ug-flag poll (0.07-0.09);
  - at 24 rows, **idle waves at two barriers**: route's (0.23) and shm's (0.17);
  - per-stage barriers (0.02-0.03 each).

  `ATOM_SHUTDOWN_TIMEOUT_S` and `att_nocode.py` make ATT of the served K2 work (§4.1).
- **COMGR vs Rust direct (§5).**
  - Three compiler-core defects fixed, each with a unit test (crate 498 / 498 tests on Linux):
    - `fcmp uno` (xq);
    - `tile.wave_ballot` (route);
    - f32 `math.sqrt` (route), which needed a new gfx950 ISA row `v_sqrt_f32`.
  - Standalone, both nodes, bitwise:
    - `xq`: direct + Dm **0.957-0.961** (win);
    - `route`: 1.010-1.056 (loss; diagnosed as missing `v_pk_add_f32` pairing and unfolded uniform address math);
    - `reduce`: tie.
  - `down` miscompiles direct. It is bisected to the composed body: not spills, the shared path, the scaled MFMA,
    constant division, the tagged poll, the round loop or uninitialized LDS. Open.
  - In the hybrid all extern functions are LLVM-linked, so the direct verdicts apply to standalone or whole-K2 rocKE
    builds.
- **Knob surface (§6, `dsv41_mono/k2moe_space.py`).**
  - 13 port knobs plus the per-stage compiler backend, as a decision-15 `SearchSpace`. Level 0 is parity, and
    `chains` (the tolerance class) is refused by default.
  - `function_sets` / `extern_spec` map a config to the rig's builds.

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

| node | tree | arm | gated launches a rank (K1 + K2) | DIFF launches |
|---|---|---|---|---|
| n01-09 | a1 (no `shm`: it ran before `shm` was added) | seeds 0 / 1 / 2 | 12,382 / 12,216 / 12,382 | 0 / 0 / 0 |
| n01-09 | a1 | native r0 | 19,200 | 0 |
| n01-09 | a3 + rocKE a4 (all stages incl. `shm`) | seeds 0 / 1 / 2 | 12,136 / 12,652 / 12,652 | 0 / 0 / 0 |
| n01-09 | a3 + a4 | native r0 | 19,200 | 0 |
| n07-17 | a3 + a4 | seeds 0 / 1 / 2 | 12,738 / 12,812 / 12,824 | rank 1: 0 / 0 / 0; rank 0: 3 / 2 / 1 launches, every one only 1-2 bytes of `res_ffn0`, which the port does not write. ATOM-vs-ATOM on the same node: 19 / 9 / 5 (see below) |
| n07-17 | a3 + a4 | native r0 | 19,200 | 0 |

Widths covered by the seeds: K2 5 / 6 / 10 / 12 / 15 / 18 / 20 / 24 / 25 / 30 / 35 / 36 / 40 / 42 / 48. These are
all target and draft widths that reach K2 in these runs. The native path covers 1 and 4.

**Must-fail mutants.** Each mutant arm swaps one stage's function set for a mutant. A mutant is caught if it hangs or
if any launch differs.

| mutant | stage | what it breaks | n01-09 (a1) | n01-09 (a3) | n07-17 (a3) |
|---|---|---|---|---|---|
| `put_bit` | xq | one bit of a written word | caught (3,731 diff launches, hang) | caught (3,731, hang) | caught (3,986) |
| `code` | xq | an MXFP8 code | caught (3,774, hang) | - | - |
| `put_bit` | route | the route weight bits | caught (3,857, hang) | - | - |
| `idx` | route | the expert id | caught (20, hang) | caught (23, hang) | caught (23, hang) |
| `put_bit` | down | the pushed pair | caught (3,943, hang) | caught (3,817, hang) | caught (3,811, hang) |
| `chain_drop` | down | one MFMA K step | caught (3,943, hang) | - | - |
| `scale_byte` | down | the E8M0 scale byte | caught (3,903) | - | - |
| `row` | down | a token row | caught (3,767, hang) | - | - |
| `shared_drop` | down | the shared expert's add | caught (4,069) | - | - |
| `topk_order` | down | top-k add order | caught (3,808, hang) | caught (3,768, hang) | caught (3,851, hang) |
| `put_bit` | reduce | the reduced word | caught (4,112, hang) | caught (3,774, hang) | caught (3,762, hang) |
| `col` (first form) | reduce | rotate read and store column | **NOT caught (0 / 8,058): equivalent mutant** | - | - |
| `col` (fixed: store column only) | reduce | the sum lands one column pair over | - | **caught** (3,857, hang) | **caught** (2,133, hang) |
| `put_bit` | ug (one tile) | the MXFP8 mid word | caught (1,635, hang) | - | - |
| `put_bit` | ugx (past one tile) | the same | caught (2,053, hang) | caught (2,050, hang) | caught (561, hang) |
| `put_bit` | shared (one tile) | the shared mid | caught (1,758, hang) | - | - |
| `put_bit` | shx (token parallel) | the same | caught (480, hang) | caught (680, hang) | caught (1,502) |
| `put_bit` | shm (past one tile) | the shared mid | - | caught (1,456, hang) | caught (1,450) |
| `row` | shm | a token row | - | caught (1,493, hang) | caught (hang before its first record) |

**The n07-17 `res_ffn0` bytes.** In the three DSpark seeds on n07-17, 6 of 38,374 gated launches a rank (K1 + K2) differ in 1-2
bytes of `res_ffn0`, on rank 0 only:
- the token-parallel widths: 24 rows twice, 48 rows three times, 12 rows once;
- nothing in `ffn_out`, `peers`, scratch or any region the K2b stages write;
- the same arms on n01-09: 0 differences (a1 and a3 trees).

`res_ffn` is the FFN residual that the K2a seam (`slice`) writes; the port does not write it.

**A-vs-A control** (job `j4a`, n07-17, `ATOM_DSV41_RK_LEV=nop=1`: ATOM's own K2 against a recompiled ATOM K2, same
gate, DSpark seeds 0 / 1 / 2): 19 / 9 / 5 launches on rank 0 differ, all in `res_ffn0` (one of them also in 1
scratch byte), and rank 1 has 0. That is 33 launches against the port's 6. So these bytes are ATOM's own nondeterminism on that node's rank 0 (a K2a seam read that can race), not
the port. **The n07-17 gate passes**: every region the port writes is bitwise. The planned per-stage bisect (`j4b`)
was cancelled as unnecessary.

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

- **Reading (first screen).** Against `nop`, every one of these down-stage variants is ahead of ATOM's K2 at every
  width, and `r_max=1` most of all. It halves the (task, slot) blocks a wave holds per round, so each round's weights
  and ug-flag poll sit on the critical path for half as long.
- **Raw ratios.** This job's `nop` sat at 1.001-1.004. The raw variant / ATOM ratios are therefore 0.990-0.999, and
  the second-seed, second-node re-test below (r2) does not reproduce a stacked gain. Read this table as "the
  levers are bitwise and do not hurt", not as an adopted win.
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
    `ugs`, `dr1`, and `dwf0` re-tested. Results, raw variant / ATOM (the parity port `all`, the `nop` control, the
    best whole-port combination over `all`):

    | rows | n01-09 `all` | n01-09 `nop` | n01-09 best combo / `all` | n07-17 `all` | n07-17 `nop` | n07-17 best combo / `all` |
    |---|---|---|---|---|---|---|
    | 6 | 1.002 | 1.007 | 0.993 (`c4`) | 1.004 | 1.059 | 1.007 |
    | 12 | 0.996 | 1.004 | 0.998 | 0.992 | 1.008 | 1.002 |
    | 18 | 1.002 | 1.007 | 0.995 | 0.998 | 1.008 | 1.002 |
    | 24 | 0.995 | 1.002 | 0.999 | 0.992 | 1.002 | 1.002 |
    | 42 | 0.996 | 1.006 | 0.998 | - | - | - |
    | 48 | 0.992 | 1.014 | 1.001 | 0.989 | 1.005 | 1.001 |

    Verdict: the down-stage levers that won as singles against `nop` (§3.2 first table) do not stack into a
    reproducible whole-port gain on a second seed and a second node. The parity port is the best measured
    configuration, and the levers stay default-off knobs for the climb. Its noise model has to absorb a ~1 %
    per-compile lottery (the `nop` column), which is decision 15's Race / NoiseModel.

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

**Ranked bubbles of our code.** Share of the rocKE functions' traced latency, by source line, pooled over 8 dispatches
per width: 2 ranks × 2 layers × both nodes (j1b n07-17, j3 n01-09):

| rank | 6 rows | 24 rows |
|---|---|---|
| 1 | `stage_ug.py:208` ug MFMA (waits on its MXFP4 weight loads, stall 0.98): **0.40** | `stage_moe.py:527` **route's trailing barrier** (7 waves wait while wave 0 runs the top-k): **0.23** |
| 2 | `stage_moe.py:342` `ld` (down weights / scales, stall 0.96): 0.18 | `stage_moe.py:946` **shm's barrier** (the shared unit past one tile waits for its x rows): **0.17** (+ its poll 0.02) |
| 3 | `stage_moe.py:365` `poll_pairs` (down's ug-flag poll): 0.09 | `stage_ug.py:153` ug weight loads (VMEM issue stall 0.97): 0.14 |
| 4 | `stage_moe.py:775` down MFMA: 0.06 | `stage_moe.py:342` down `ld`: 0.10 |
| 5 | `stage_ug.py:175` ug scale word LDS read: 0.06 | `stage_moe.py:365` down's ug-flag poll: 0.07 |
| 6 | barriers of down / shared / ug (`stage_wqkv.py:163`): 0.03 / 0.02 / 0.02 | ug MFMA 0.06, down MFMA 0.04, ug barrier 0.03 |

Pooled stage walls (traced CU spans), ours / ATOM:

| rows | K2b sum | ug | down | reduce |
|---|---|---|---|---|
| 6 | **0.954** | 1.000 | 0.988 | 0.963 |
| 24 | **0.945** | 0.958 | 0.941 | 1.044 |

Down's own busy cycles per CTA are 0.907 (6 rows) and 0.885 (24 rows) of ATOM's. These agree with the in-place A/B
(§3): the port is a little faster than ATOM inside the same K2, and the gap grows with width.

- **Weight stream.** The K2b group is the weight stream. Ranks 1-2 (0.59 of the traced rocKE latency at 6 rows, 0.38
  at 24) are waits on, and issue stalls of, the expert weight loads. This is the byte floor that WP4 / P1 measured.
- **Bubbles on top of the stream:**
  - down's ug-flag poll (the latency chain ug -> down);
  - at 24 rows, **idle waves at barriers**: route (0.23) and shm (0.17) keep 7 of 8 waves at a barrier while one
    wave works;
  - the per-stage `s_barrier`s.

  `pslot` attacks the poll, `r_max=1` the round length, and `sh_first` the order. The two idle-wave barriers are the
  next structural targets:
  - route: one wave per token over up to 8 tokens a call, instead of wave 0 for each token;
  - shm: let the waiting waves issue the next unit's weight loads.

  Both are knobs to build (§6, "next").
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
| `xq` | was refused (`fcmp uno`): **fixed** | **bitwise** (D, Dm, Db; 8 inputs, twice; both nodes) | D **1.122-1.128** (loss); Dm **0.957 / 0.961**, Db 0.957 / 0.960 (n01-09 / n07-17, two jobs on n01-09: 0.958, 0.957); C2 0.997-1.004, R 0.997-1.003 | **direct + Dm** (wins on both nodes) |
| `route` | was refused (`tile.wave_ballot`, then f32 `math.sqrt`): **both fixed** | **bitwise** at 6 / 48 rows and draft 5 rows, both nodes | D 1.050-1.056 (target), 1.026-1.027 (draft); Dm 1.022-1.024; Db 1.010-1.016: **loss** on both nodes | COMGR (loss = compiler defect, below) |
| `down` | compiles | **FAIL** at every shape, with or without spills (see below) | - | COMGR (defect open) |
| `reduce` | compiles | **bitwise** (D, Dm, Db): 6 rows on both nodes, 24 rows on n01-09 | D 0.993-0.999, Dm 0.997-1.000, Db 1.000-1.002 against R 0.995-1.000 and C2 0.998-1.001: **tie** | COMGR (direct only where it wins) |
| `ug`, `shared`, `shm` | not yet in the standalone rig | - | - | COMGR |

**Scope of "auto-select" in the hybrid.** The extern stage functions are LLVM device functions linked into ATOM's
FlyDSL K2 (`rk_extern`), and the direct compiler emits whole code objects (one kernel, no linked functions). So in
the hybrid every K2b stage runs COMGR-compiled code whatever this table says. The verdicts apply to the same bodies
compiled as standalone kernels: a whole-K2 rocKE build (the "rocKE mono" step), or a K2b split into its own launch.
There `xq` would take direct + Dm. The standalone rig is still where each direct-vs-COMGR difference is found and
fixed (decision 18).

**`route` loss, diagnosed from the two listings** (6 rows; local COMGR vs build `m3`; 1,461 vs 1,280 instructions):
- the 4 router-part sums are 85 scalar `v_add_f32` where COMGR pairs them into 18 `v_pk_add_f32`;
- about 200 extra SALU ops (`s_and` 69, `s_add` 49, `s_lshr` 48, `s_mul` 21, `s_sub` 24): uniform index arithmetic
  that COMGR folds into buffer offsets / immediates;
- 49 vs 43 `v_cmp_u_f32` NaN tests.

Next fixes: packed f32 add pairing, uniform address folding into `buffer_*` `soffset` / `offset:`. The `Db` lever
set (with `dppcombine`, `vmemfirst` and others) already recovers about 70 % of the gap.

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
- So it is neither the spill path nor the shared expert path.
- **Not a harness artifact.** `s4` poisons every CU's LDS before each launch, with two patterns. COMGR's output does
  not move (poison sensitivity 0 bytes at 6 and 24 rows), so the body writes every LDS word it reads. Direct's
  output is wrong identically under both patterns, so the miscompile is deterministic.
- **Not the scaled MFMA, not constant division.** `mfma_probe.py` (job `mp1` / `mp2`, n01-09, 2 seeds) builds
  minimal one-wave kernels. Every one is bitwise COMGR vs direct:
  - fp8 x fp8 and fp4 x fp8 16x16x128;
  - the A scale as `(word >> 8 byte) & 0xff` and as `op_sel` byte 2;
  - a 9-step chain on 2 accumulators;
  - `arith.div` / `arith.mod` by all of down's kinds of constants, per lane and uniform.
- **Not the poll, not the round loop either.** `mp3` adds two more probes, both bitwise, 2 seeds:
  - `poll5`: a 5-spec `tile.tagged_poll` with `reload=launder` and `s_sleep 1`, as down's ug-flag poll;
  - `loop`: a dynamic-trip `scf.for` with trip `(nu + 3) / 4` of a loaded uniform, LDS read-modify-write per round,
    as down's round loop into `contrib`.
- What is left untested is the composition. In the full body, LDS at kernel-argument word addresses (`lds="arg"`)
  meets the `v_mov` asm `fresh` copies, the per-slot route-table reads (`pick_at`, `expert_slot`) and
  `s_barrier_bare`. The tags are right while every payload is wrong, and direct's sums are smaller than COMGR's
  (COMGR overflows to inf on these synthetic weights, direct stays finite), which points at a wrong table or slot
  read rather than at arithmetic.
- Next: bisect the down body itself with `MoeKnobs`-level cut points: the shared part only; one round; one slot.
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
| next: `route_par` | route | one wave per token over up to 8 tokens a call (needs a token-range ABI) | off | bitwise | **no** (ledger target, §4.2) | - |
| next: `shm_pre` | shm | waves waiting at shm's barrier issue the next unit's weight loads | off | bitwise | **no** (ledger target) | - |

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
- `srig.py` (with `--poison` LDS-sensitivity checks), `dprobe.py`, `mfma_probe.py`: the standalone COMGR-vs-direct rig,
  the local compile / resource probe, and the minimal direct-compiler probes (scaled MFMA, constant div / mod, tagged
  poll, dynamic round loop).
- `../srig/srig_job2.sh`, `../srig/mprobe_job.sh`: their job bodies. Steps take knobs, and `W15M_SC` makes them run
  node-local.
- `../rustbuild_m.sh`: builds and tests a crate tarball on CPUONLY into `codeobj-<tag>`.
- `bench_sum.py`, `wp4_htl_cmp.py`, `../gtl_cmp.py`: the in-situ summaries.

A candidate screen in the rig costs seconds per candidate per width, inside one running engine (rk_bench). A full
gate is about 12 min per arm.

## 8. Pending

1. **Served confirmation.** HTL (graph-mode GPU decode step) with twins on both nodes for the parity port and for
   `c1` / `c3`, then a served cell. The fixed 256-token decode window, the decision-12 funnel.
2. **The climb (W15 Climb, decision 15).** Drive `k2moe_space.search_space()` with the search engine:
   - HillClimb / Portfolio, with Race / NoiseModel for the ~1 % compile lottery;
   - the rig as rung 0;
   - `chains` behind a tolerance gate;
   - together with the decision-10 schedule surface (CTA allocation per stage, issue order, holds).
3. **Build the two idle-wave levers** the ledger found (`route_par`, `shm_pre`), and screen them.
4. **Direct compiler.** The `down` miscompile: cut points inside the body. The `route` loss: `v_pk_add_f32` pairing
   and uniform offset folding. The register-allocation gap (1.5-2.1x COMGR's VGPRs on down). Then `ug` / `shared` /
   `shm` in the standalone rig.
