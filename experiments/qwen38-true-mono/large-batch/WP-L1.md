# WP-L1: the large-batch framework on W5's `persist/` platform (rows 5-128, TP1 and the TP2 rank shape, gfx950)

W7, 2026-11-01. `../DESIGN-LB.md` 15.6 (WP-L1), 15.4 ("correctness infrastructure"), 5 (sync model), 4.1 (rows
buckets, runtime token tiles, masking), 8.2 (index widths); `P-L2.md` 5 (the per-edge default WP-L1 starts from);
`../DESIGN-CCL.md` 10.6 (the asm hazard rule). Ratios only here. Absolute times, job logs and raw JSON are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L1.md` and `.../large-batch/wp-l1-raw/`. Full ratio
tables: `WP-L1-tables.md` (final runs) and `WP-L1-tables-v1.md` (first base).

## Gate and verdict

**Gate: met.** Every point of DESIGN-LB 15.6's WP-L1 row is built and checked on MI355X on both rocke nodes. Two
points did not come out the way the plan assumed, and both are reported as measured.

| gate item | result (final runs g / h: n07-17 bus 75, n01-09 bus 05; seeds 20261101 / 777003) |
|---|---|
| CPU tests (`library/tests/test_qwen38_lb_framework.py`) | 81 pass (26 of them added from the mutation runs, section 5); P-L2's 29 `test_qwen38_lb_sync_chain.py` tests pass unchanged |
| existing builds byte-identical | no existing file was edited (6 new files, below). The P-L2 sources the bench runs next to the framework print the same sha256 as P-L2's fix round (`sync_chain.py`, `sync_plan.py`, `persist/sync.py`, `ccl/wire.py`). The `stagecode` knob added after the v2 measurements leaves all 56 earlier framework builds (14 knob sets x 4 stage counts) identical in LLVM IR |
| ISA hazard compile gate (every build scanned; padded builds must be clean) | 153 / 153 padded builds clean per run (framework, P-L2 chain and stale-harness builds). The 4 `hz=0` builds of the final source **have real hazards** and are refused by the gate (next section) |
| index-width lint (IR interval analysis + host formulas) | 5 / 5 gate cells clean per run. Planted overflow (300 layers of per-layer slots) caught by both checks (CPU test) |
| LB validate (deadlock, read-after-write, write-after-read at layer parity) | 40 / 40 clean per run (5 cells x 4 sync plans x 2 walks). Planted violations are caught: a WAR toy graph, a dropped wait (RAW), exchange slots reused before the other rank read them, a grid barrier treated as ordering another rank's exchange lines (RAW), a cycle (CPU tests) |
| numerics on MI355X | 476 / 476 records PASS per run, 2 steps each (952 steps): 12 arms x 7 cells (incl. runtime rows 17 of bucket 128, 5 of bucket 8, 50 of bucket 64) x 2 skews x 2 walks, + layer-parity slots. Each step checks the payload of every retained layer of every live unit against the host model, the launch-fatal word after **every** launch, the residency ticket and the ctl words |
| must-fail controls | 52 / 52 gating per run (none, early twins x walks, stale-tag twins x walks x slot forms, deadline controls, oversize grids); 12 / 12 informational (`noacq`, random-skew early) |
| deterministic stale-read harness | positives 0 stale words in every launch; every twin stale in **100 %** of (trial, consumer) pairs in every launch, same and other XCD, both nodes, including the cache twin (plain loads) that P-L4 F2 could not catch reliably |
| timed binaries re-verified after timing (payload zeroed) | 2052 / 2052 per run; 228 / 228 must-fail sentinels timed as arms caught |
| P-L2's verdicts reproduced | **P-L2's own builds, timed in this harness: yes** (its default over its barrier 0.983 / 0.984, P-L2 0.979-0.983). **On the framework: partly.** The framework's barrier is 0.977 of P-L2's, so the per-edge default (`sync=mix`) no longer beats the barrier at idle arrivals (1.010-1.011, 7 TIE / 8 LOSS of 15). It still wins every cell with one late unit per stage (0.940-0.942). The other P-L2 verdicts hold (section 3) |
| runtime layer loop vs one launch per layer | the loop wins: one launch per layer costs 1.018-1.021 (idle and uniform skew, default and grid arms, both nodes; every byteful cell but one, section 3.1) |
| WaveScope before / after | done on the first base and after the read-set fix (section 4) |

## 1. What was built

All new files; nothing existing changed. Python engine only; names `q38lbf_*`; nothing served imports them.

| piece | file |
|---|---|
| host side: rows buckets `{8, 16, 32, 64, 128}`, per-unit token tiles and masking, the per-edge sync plan (`EdgeSync`, default = P-L2's `gmask_full(plan, 0.4)`), device tables, per-layer tags, the host payload model with masking, LB validate (deadlock, RAW, WAR at any slot count, TP exchange slots across ranks; `ccl_war` hook to `ccl/war.py` for the CCL arena schedule), index-width host bounds | `dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/qwen38_mono/large_batch/lbf_plan.py` |
| the framework kernel: static per-stage code with a `StageSpec` / emitter interface (`prefetch` before the wait, `body`, then the framework's publish), the runtime layer loop, per-stage grid barrier or polled read-set flags, masking, per-layer tags, slot forms, ticket, launch-fatal word, mutants, the P-L2 stream body as the first emitter (`StreamEmitter`); knob `stagecode=loop` (one body in a runtime stage loop) | `.../large_batch/lbf.py` |
| asm-hazard compile gate: `pad_asm_vmem` (`hz`), the ISA scanner (`p-l4/fix/asm_hazard.py`'s algorithm), `compile_gated` | `.../large_batch/lbf_hazard.py` |
| index-width lint: interval analysis over the rocKE IR from declared argument / table bounds to every address sink (buffer offsets and sizes, load / store indices, i32 `zext` into 64-bit addresses, the vaddr of asm VMEM) | `.../large_batch/lbf_lint.py` |
| deterministic stale-read harness kernel | `.../large_batch/lbf_stale.py` |
| CPU tests | `.../rocke/library/tests/test_qwen38_lb_framework.py` |
| GPU bench, report, job body, staging, WaveScope launcher + job, mutation testing | `hackathon-prep/qwen38/truemono-lb/wp-l1/` (`lbf_bench.py`, `lbf_report.py`, `job.sh`, `sync_wpl1.sh`, `att_lbf.py`, `att_job.sh`, `mut/`); `pl2_bench.py` is P-L2's bench copied unchanged (sha256 `dcc91dfc...`), imported for its device helpers and to time P-L2's own builds next to the framework |

**On the `persist/` platform.** The framework imports W5's primitives and edits none of them: `persist.sync`
(`GridSync`, 8 stripes per XCD, the per-XCD ticket with the launch-fatal word, `raise_fatal`, `check_launch` after every
launch or replay, `pad_lds`, `st_sc1`), `persist.knobs.PersistKnobs` (the D0 defaults; it applies `machine_licm`,
`arrivals`, `stripes`, `ticket`, `fatal_word` and refuses every other non-default field explicitly, WP1's
"nothing silently ignored" rule) and `ccl.wire.poll_flags` (the deadline-bounded `sc1` flag poll). W5's
`persist/driver.py` loop driver is built around the small-batch `framework.py` `StageSpec` / `TaskTable` (per-row
mailboxes); the LB task graph is host tables of units, so the LB layer loop is its own (same R3 rule: the loop carries
only the layer index and the running counters; no per-layer argument table is needed, because the layer index addresses
everything).

**Mechanisms as built** (DESIGN-LB 15.6 list):

- **Rows buckets, runtime token tiles, masking.** A graph is planned per (plan kind, TP, bucket); the live row count is a
  kernel argument (`ntiles = ceil(rows / 16)`). A unit whose first token tile is past the live tiles does no work and
  publishes nothing; every consumer skips it in its wait set (deps word high bits) and its read set. Measured at rows 17
  of bucket 128 (a third of the units masked), 5 of bucket 8 and 50 of bucket 64.
- **Task graph with partial edges.** P-L2's per-token-tile / per-head / per-expert read sets (`sync_plan`), 16 / 14
  (FlyTape's list) and 11 / 10 (DESIGN-LB's list) stages at TP1 / TP2.
- **`sync=mix` per edge** (`gmask_full(plan, 0.4)`, per-wave lazy flags, `s_nop` pause), with `grid`, `tile`, `a2a`,
  `none`, `pub=drain`, `lazy=0`, the 0.9 threshold and `s_sleep1` as knobs. `pub=counted` is not ported (it lost in
  every P-L2 cell; it stays a `sync_chain` knob).
- **Walk** (`loop`: one launch per step; `launch`: one launch per layer, the launch boundary as the first stage's edge),
  a host knob of the same binary. Tags are per layer (`E * ltot + l + 1`), so both walks write the same bits (both pass
  the same model in every numerics record).
- **Slots** per layer or per layer parity (a runtime argument); the previous occupant of a slot always carries
  `tag - slots`, which is what the stale-tag twin polls for.
- **LB validate**: an event graph of 3+ layers (tasks, per-CTA program order, one hub per grid barrier, launch
  boundaries, flag edges, cross-rank exchange edges) with ancestor bitsets: no cycle, every read after its producer,
  every slot reuse after every read of the previous occupant.
- **Launch-fatal check per launch**: the bench calls `persist.sync.check_launch` after every launch of every numerics
  step and control (the D0.8 amendment).

## 2. Correctness infrastructure

### 2.1 The asm hazard scan as a compile gate (it fired)

Every build of a run is compiled through `lbf_hazard.compile_gated`: disassemble (`/opt/rocm/llvm/bin/llvm-objdump` is
present in venv jobs on both rocke nodes), scan, refuse on any hit. The scanner is `p-l4/fix/asm_hazard.py`'s algorithm;
on P-L4's kept dumps it flags the same 28 of 28 `hz=0` dumps and none of the 5 `hz` dumps.

- First base (v1): every build clean, `hz=0` included.
- After the read-set fix (v2, section 4), the 4 `hz=0` builds of the default arm had 3 hits each: `v_readlane` restores
  a spilled SGPR base 0 wait states before an asm `global_store_dword ... sc1` (the payload / flag store). The padded
  builds stayed clean. The change that exposed the hazard did not touch any asm. This is the allocation-dependence
  DESIGN-CCL 10.6 warned about, now in a new kernel family. The gate refused the `hz=0` arm (the bench records it as
  refused and does not time it); `hz` stays on. On the v1 base, where the `hz=0` build was still clean, it timed
  0.994-0.997 of the padded build. That is a small cost of the pad, but the unpadded build is unsafe, so `hz=0` is not
  a usable lever.

### 2.2 The deterministic stale-read harness

P-L4 F2: the `noacq` twin (pretouch + plain loads) of a full kernel is not a reliable must-fail check on MI355X. The
harness forces the timing instead: **a producer may not store until its consumer has done what the twin needs**, and
nothing runs between that point and the twin's read that could evict the lines.

- One launch, 128 producer and 128 consumer CTAs. Consumer `c` reads producer `(c - 128 - xs) mod 128`: `xs = 0` puts
  the pair on the same XCD, `xs = 1` on neighbouring XCDs. 32 trials per launch, 4 launches, each trial on its own
  payload, flag and hold word.
- Consumer: pretouch the lines, then signal (`sig=pre`), poll the flag with the framework's poll, read with the
  protocol under test. `sig=post`: the signal comes after the read.
- Producer: hold until the signal (bounded; a timeout is counted, not fatal), store with the framework's payload store,
  then the drain publish.

| protocol, twin, signal | role | stale share of (trial, consumer) pairs per launch, both placements, both nodes |
|---|---|---|
| `sc1` loads (the framework's read), `pre` and `post` | positive | 0.000 |
| `buffer_inv sc1` + plain loads, `pre` and `post` | positive | 0.000 |
| plain loads, no acquire (`noacq`), `pre` | must-fail | **1.000** |
| plain loads, no acquire, `post` (producer released by the hold timeout) | must-fail | **1.000** |
| read before the poll (`early`), `post` | must-fail | **1.000** |
| poll for the previous tag (`tag`), `post` | must-fail | **1.000** |

The order twins are deterministic by construction: the producer cannot store before the early read has completed. The
cache twin is deterministic on hardware: a pretouched line is still in the consumer CU's vL1D when the plain read issues,
because the only memory traffic in between is the signal store and the flag polls (`sc1` loads). P5's 3.6 % and P-L4's
misses came from evictions and timing between the pretouch and the read, which the harness removes. In the full kernel
the stale-tag twins (`tile:tag`, `def:tag`, `defsl:tag`, both walks, both slot forms) are deterministic too: every launch
either reads the slot's previous payload or times out on the launch-fatal word, and both outcomes count as caught.

### 2.3 Index widths, LB validate

- **Index-width lint** (`lbf_lint.py`): every address sink of the framework kernel is bounded at the bucket's sizes.
  The lint and the host formulas agree on the first overflow: the payload byte offset at 300 layers with one slot per
  layer at TP1 B128. At the plan's 6-48 layers with layer parity there is ample margin.
  - The deps word (unit id plus masked tile) is clamped with `smin(d, N - 1)`, so the payload / flag index is provably
    in range.
  - No 64-bit index is formed in i32: the weight block base is zero-extended before its multiply.
- **LB validate.** DESIGN-LB's dense layer is WAR-safe at layer parity under every sync plan. That holds even for pure
  flags, because the full edges (HC up, router -> experts) order every unit of layer `l + 1` after every reader of
  layer `l`.
  - Slot reuse is unsafe only on sparser graphs, which the toy-graph tests cover.
  - For TP2, the exchange slot check shows that a grid barrier never orders another rank's exchange lines: they must be
    polled (DESIGN-CCL 4.9). The checker enforces it.

## 3. The sync model on the framework (final runs; ratios vs the framework's `grid` arm, loop walk, idle)

Geomean over the 15 byteful cells (P-L2's 13 plus the new rows-8 bucket at TP1 / TP2), n07-17 / n01-09; W / T / L under
the DESIGN 9.1 win rule (controls `grid#2`, `grid_rc`: 0.998-1.003):

| arm | idle | uniform skew | one late unit per stage | P-L2 (idle, its own barrier) |
|---|---|---|---|---|
| `def` = `sync=mix`, per-wave lazy flags, 0.4 (P-L2's default) | 1.011 / 1.010, 0/7/8 both | 1.004 / 1.004 | **0.940 / 0.942, 15 WIN** | 0.979-0.983 WIN |
| `tile` (read-set flags, drain) | 1.027 / 1.026 | 1.014 / 1.017 | **0.915 / 0.916** | 0.998-1.003 |
| `tile_w` | 1.072 / 1.074, 15 LOSS | 1.052 / 1.053 | 0.946 / 0.948 | 1.015 LOSS |
| `a2a` | 1.059 / 1.058, 15 LOSS | 1.043 / 1.046 | 1.007 | 1.016 LOSS |
| `mix90d` | 1.021 / 1.022 | 1.015 / 1.016 | 0.922 / 0.924 | 0.991-0.996 |
| `none` (floor) | 0.879 / 0.875 | 0.857 / 0.853 | 0.625 / 0.622 | 0.83 |
| P-L2's `sync_chain` default (`chain`) | 1.007 / 1.007 | 1.004 / 1.006 | 0.941 / 0.942 | - |
| P-L2's `sync_chain` barrier (`chain_grid`) | **1.023 / 1.023, 15 LOSS** | 1.029 / 1.032 | 1.023 | - |

- **P-L2 reproduces in this harness.** P-L2's own builds give `chain` / `chain_grid` 0.983 / 0.984 at idle, 0.976 /
  0.975 under uniform skew and 0.920 / 0.921 with one late unit, against P-L2's 0.979-0.983, 0.974-0.977 and 0.927.
  - The default arm itself carries over: framework `def` / P-L2 `chain` is 1.003-1.005.
  - So does the flag poll: `tile` / `chain_tile` is 0.996-0.997.
- **What changed is the barrier.** The framework's grid arm is 0.977 / 0.978 of P-L2's (byteful, every cell below 1 in
  every round; 0.93-0.96 in the null and rows-8 cells).
  - The flag arms gain nothing from that, so against the faster barrier the per-edge default is a TIE or LOSS at idle.
    The 0.4 threshold, per-wave flags, lazy publish and `s_nop` pause are unchanged.
  - The likely cause is P-L2's dummy tasks. A P-L2 CTA without a unit in a stage runs a full dummy task (table loads,
    an out-of-range weight ring, a read set of zero producers with its LDS barrier) between arrival and wait. The
    framework's CTA arrives and waits. This is not proven: no build isolates it.
- **Which arm wins depends on arrival skew.**
  - Idle: the barrier wins; the mix default is within 1.0-2.5 % in the byteful cells (worst at TP2) and 2-7 % in the
    null cells.
  - Uniform skew: a tie at TP1, a 0.7-1.5 % loss at TP2.
  - One late unit per stage: every flag arm except `a2a` wins by 5-9 %, `tile` most (`a2a` 1.007).
  - Real stage bodies are skewed (per-expert token counts, P-L4's correlated routing with up to 127 tokens on one
    expert). So `def` stays the framework default, provisionally. Every arm stays a knob, and WP-L3 re-decides per
    edge in situ (DESIGN-LB 5's procedure).
  - The idle flip is recorded so that WP-L3 does not start from P-L2's idle win.
- **TP1 vs TP2.** As in P-L2, flags pay at TP1 large rows (tp1:128:lb: `tile` 0.991-0.996 idle, `def` 0.994-0.997) and
  lose at the TP2 rank shape (1.010-1.025 idle for `def`). P-L2 5.2's TP2 sync share carries over.

**Combination lookback (PLAN 13 / 15; base = the final source).** 19 `sync=mix` combinations over publish x lazy x
threshold x machine LICM x stage code (all retained knobs, the solo losers included) plus `grid_sl`, each x walk x slot
form, on 7 cells, idle.
- None beats the framework barrier on the geomean. The best are `c.d.1.40.0.l` (drain, lazy, 0.4, LICM off, runtime
  stage loop) at 1.003 / 1.004 and `c.d.1.40.1` at 1.007 / 1.008, with 1-3 cell WINs at TP1 (tp1:16:ft, tp1:128:ft,
  tp1:64:lb).
- Every combination loses at TP2 and in the null cells.
- The per-wave flag combos lose more than the drain ones once the stage loop or the 0.9 threshold is added.
- One launch per layer adds about 1-5 % to every combination.
- Layer-parity slots do not change the ranking.
- The full table is in `WP-L1-tables.md` "Combination lookback". These verdicts are on this base; the losers stay knobs
  and are re-tested after every accepted change (PLAN 15).

### 3.1 Runtime layer loop vs one launch per layer (the knob the task asked for)

One launch per layer (same binary, `nl = 1`, a graph of 6 launches per step) over the runtime layer loop (one launch
per step), the same arm in the same rounds:

| arm | idle (n07 / n01) | uniform skew |
|---|---|---|
| `def` | 1.018 / 1.021 | 1.017 / 1.021 |
| `grid` | 1.021 / 1.020 | 1.020 / 1.022 |
| `def_sl` (runtime stage loop) | 1.019 / 1.019 | - |

- 0 % of rounds below 1 in every byteful cell but one (tp1:64:lb, `def`, n07: 0.998, 0.70 of rounds below 1). Per
  cell 0.998-1.044 (null cells up to 1.062). Largest at small rows and TP2, where a layer is short against a launch
  boundary plus a residency ticket.
- **This reverses P-L4's sign** (the persistent MoE walk at 1.04-1.08 of per-layer launches). Here the loop body has no
  per-layer setup beyond table loads. The P-L4 cost came from inside the MoE kernel's layer walk (its per-layer route
  and sort and its grid barrier), not from the loop as such.
- `walk=loop` is the default. Per DESIGN-LB 15.2 the question stays open until WP-L4 re-measures it on real bodies and
  re-tests it after every accepted change.

### 3.2 Other knobs on the final base

| knob (vs the default) | idle, n07 / n01 | verdict |
|---|---|---|
| layer-parity slots (`slots=2`) vs one per layer | 0.997 / 0.998 (byteful per cell 0.989-1.004; null cells 0.987-1.018) | kept as a knob; parity is what a 48-layer pass needs (memory), and it costs nothing |
| `stagecode=loop` (runtime stage loop, 1/7 of the code) | default arm 1.017 / 1.018 LOSS; grid arm 0.997 / 0.998 TIE | default-off knob. The smaller code does not pay: the static code lets LLVM specialise each stage's flag / barrier branches |
| machine LICM on (`ml`, R4 off) | 1.000 / 1.000 | TIE (D0 default kept) |
| poll pause `s_sleep 1` | 1.011-1.012 vs grid, = `def` | TIE |
| `hz=0` | refused by the compile gate on this base (2.1) | - |
| A-vs-A recompiled default / recompiled stage-loop default | 1.000-1.001 / 1.000 | controls in band |

## 4. WaveScope before / after, and the read-set fix

The first base (v1) ran 1.068 / 1.070 of P-L2's chain on the same arm (every byteful cell 1.03-1.13, no round below
1). P-L2's chain default ran 0.94 of the framework's own barrier arm on the same plan and bytes, so every framework arm
was slow, not only the flag arms.

WaveScope (ATT) on one CU, MI350X (`bg-1w300-e1-2a`, `w7-wpl1-att2`), tp1:64:lb, default arm:
- More `s_waitcnt vmcnt(0)` time than the chain (12.5 % of latency against 8.6 %), most of it at the read-set's deps
  loads (`lbf.py` read_set, 4.6 %).
- The ISA showed why: each deps load was followed by `s_waitcnt vmcnt(0)` before its payload load, so the 8 producers
  of a group went out one round trip at a time. In P-L2's chain all 8 deps loads issue back to back.
- The cause is the masking test. The payload load's offset `select` now depends on the loaded word's tile bits as well
  as its unit id, and LLVM interleaved and serialized the pairs.
- **Fix:** issue the group's 8 deps loads first, then compute and issue the payload loads.
  - Local ISA: 8 back-to-back deps loads, 26 instead of 41 `vmcnt(0)` in a 2-stage build.
  - Timed: v2 / v3 `def` / `chain` 1.003-1.005, the barrier faster than P-L2's (section 3).

After the fix (`w7-wpl1-att3`, the same CU and launcher), total latency per CU relative to P-L2's chain barrier on the
same cell:
- tp1:64:lb: framework grid 0.975, default 0.952, `stagecode=loop` 0.951, P-L2 default 0.965.
- tp2:16:lb: framework grid 0.978, default 0.989, stage loop 0.999.

In the tp1:64:lb capture the default arm's `s_barrier` is 26.0 % of latency (P-L2's default 17.9 %, P-L2's barrier
31.9 %) and its `vmcnt(0)` share is back to the chain's (8.8 % against 8.4 %). These are single-CU captures on MI350X; the
MI355X timing in section 3 decides.

## 5. Mutation testing

Method: WP1's `mutate_auto` (vendored in `hackathon-prep/qwen38/truemono-lb/wp-l1/mut/`). 200 mutants are sampled
(seed 1) over the five `lbf*.py` modules, and a mutant is killed when `test_qwen38_lb_framework.py` fails or times out.
CPU survivors in the GPU-path modules then go through the reduced hardware gate (`wpl1_hwmut.py`), as in WP1 F4.

| run | job | killed / mutants | by file (killed / sampled) |
|---|---|---|---|
| CPU 1 | 68236973 (CPUONLY) | 91 / 200 | `lbf.py` 13 / 59, `lbf_hazard.py` 17 / 22, `lbf_lint.py` 23 / 47, `lbf_plan.py` 36 / 57, `lbf_stale.py` 2 / 15 |
| CPU 2 (after 22 new tests; same seed) | 68237069 (CPUONLY) | 109 / 200 | `lbf_hazard.py` 19 / 22, `lbf_lint.py` 34 / 47, `lbf_plan.py` 42 / 57 (`lbf.py`, `lbf_stale.py` unchanged) |
| hardware: the 59 GPU-path survivors of CPU 1 | 68237090 (n07-17, buses 65 / 15, 2 streams) | 31 / 59 | `lbf.py` 28 / 46, `lbf_stale.py` 3 / 13 |

How the hardware gate killed mutants (numerics, controls and the stale harness on tp1:32:lb at 20 live rows and on
tp2:16:lb):
- wrong payload words in the numerics;
- a must-fail control that was not caught (`grid:early` once the bias test was flipped; a deadline control);
- the stale harness (positives that read stale words, twins that read fresh ones);
- compile failures, and two hangs killed by the timeout.

The unmutated tree passed before and after the run.

The 28 hardware survivors, classified:

| class | lines | action |
|---|---|---|
| timing-only: the synthetic stream body's weight reads, which feed only a sink, so numerics cannot see them | `lbf.py` 253, 327, 334, 336, 337, 345, 511 | none. The probe body moves bytes for timing; the WP-L2 emitters replace it with bodies whose outputs are checked |
| twin-only code (the informational `noacq` pretouch) | `lbf.py` 256, 518 | none |
| the diagnostic timeout counter (the launch-fatal word is the gate) | `lbf.py` 396 (x2) | none |
| knob defaults and bounds | `lbf.py` 86, 113 | CPU test added (`test_knob_defaults_and_bounds`) |
| `sync=a2a` under `stagecode=loop` (the runtime previous-stage row; no gate builds it) | `lbf.py` 464, 473 | the combination is now refused in `LBKnobs.check` instead of shipping untested |
| equivalent (masked-lane index, the shared emitter object, ticket-line zeroing past the used words, the stale kernel's thread constant, constant encoding, resource size, a `vmcnt` wait and a signal point that the hold timeout covers) | `lbf.py` 266, 569, 585; `lbf_stale.py` 51, 81 (x2), 101, 163, 164 | none |
| dead host helpers (`new_word`, `old_word`, never used by the kernel or the bench) | `lbf_stale.py` 56 (x2), 60 (x2) | removed |

The follow-ups (one refusal, the removed helpers, the new tests) change no kernel IR. The 56 framework builds are
identical to the measured v2 / v3 sources, and the stale kernels are not touched.

## 6. Caveats

- **Synthetic stage bodies** (P-L2's byte streams), so the absolute sync costs and the idle verdict apply to these
  bodies. The framework is the platform the WP-L2 emitters plug into; WP-L3 re-decides the per-edge plan, the walk and
  the slots in situ.
- **The TP2 cells are one rank's shape without the exchange**, as in P-L2. The exchange slot check exists in LB validate
  and the `ccl_war` hook; the in-situ exchange is WP-C5's (`insitu.py`) and is wired in WP-L5.
- **One numerics event in the first triage job is unexplained.** In job 68236610 one payload word of one unit
  (layer 3, `a2a`, one launch per layer, rows 17 of bucket 128) was wrong in 1 of 312 steps.
  - It did not recur in 1,008 steps of the same cases (68236633), nor in the 6,272 later numerics steps of 8 jobs.
  - The bench now syncs the device after zeroing the payload buffer and logs every bad word. A single wrong word cannot
    come from a read-set or ordering error (that changes all 512 words of a unit), so a harness race between torch's
    fill and the first launch is the likely cause. This is not proven.
- WaveScope ran on MI350X (Markham), not MI355X.
- Neighbours: n01-09 carried 2-3 of our other lanes' jobs and a foreign 2-GPU job during h. n07-17 carried 1-2 of
  ours. The paired ratios of the two nodes agree to about 0.003 per arm.
- No quick end-to-end number (PLAN 14): WP-L1 has no deployable lever. Its stage bodies are synthetic, and nothing it
  builds runs in a served path yet.

## 7. Rerun

```bash
H=/c/Users/vanantha/Documents/dev/hackathon-prep/qwen38/truemono-lb/wp-l1
# CPU tests (local)
cd .../rocke/library && ROCKE_LLVM_FLAVOR=llvm22 ROCKE_BACKEND=python python -m pytest -q tests/test_qwen38_lb_framework.py
# stage + run (MI355X; both nodes; suffix = stage version)
bash $H/sync_wpl1.sh both 3
Q38_MAX_GPUS=8 Q38_NODE=smci355-ccs-aus-n07-17 bash $H/../submit.sh log --tag w7-wpl1-x --mode venv --gpus 2 --cpus 32 \
  --mem 192 --time 01:30:00 --stage w7-wpl1-rocke3 --stage w7-wpl1-h3 --env LBF_RKS=w7-wpl1-rocke3 --env LBF_HS=w7-wpl1-h3 \
  -- 'bash $Q38_STAGES/w7-wpl1-h3/job.sh'
# report (ratio tables / absolute tables)
python $H/lbf_report.py --runs g/run/lbf.json,h/run/lbf.json --ratio r.md --abs a.md
# WaveScope (Markham gfx950): cluster_sync.sh the tree as w7wpl1-rk and att_lbf.py + att_job.sh as w7wpl1-h, then
cluster_job.sh --arch gfx950 --mem 48 --stage w7wpl1-rk --tag w7-wpl1-attN --env ATT_TAG=attN \
  --env "ATT_ITEMS=tp1:64:lb/chain_grid;tp1:64:lb/def" -- bash -c 'bash $HOME/rocke-sdpa-claude/w7wpl1-h/att_job.sh'
```

One run (gate + 476 numerics records + 64 controls + 16 stale-harness cases + 19 cells x 25 arms x 3 skews x 2 walks
x 20 rounds + the combination lookback) takes about 7.5 min of one MI355X.

## Verification

Independent verifier (W7 vWP-L1), 2026-11-01. **Verdict: confirmed, with the doc corrections and the open item
listed below.** Every gate item ran on MI355X with must-fail checks that do fail. The comparisons are on the same
plan and bytes, in shuffled interleaved rounds, with A-vs-A and recompiled controls, on both nodes. The conclusions
follow from the tables. The decisive check was re-run on the final source and it reproduces.

**What was checked from the builder's artefacts.**
- Raw JSON of g / h (`wp-l1-raw/{g,h}/run/lbf.json`):
  - numerics: 476 records = 12 arms x 7 cells x {idle, uni} x {loop, launch} + 5 arms x 7 x 2 x 2 at layer parity,
    all `ok`, 2 steps each. The console prints only the idle / loop / one-slot subset, so the full count has to be
    read from the JSON.
  - must-fail checks: 52 / 52 gating controls caught (26 checks x 2 cells), and 12 / 12 informational.
  - stale harness: every positive at 0.000 and every twin (cache twin included) at 1.000, per launch, at both
    placements.
  - timed binaries: 2052 / 2052 re-verified and 228 / 228 sentinels caught.
  - hazard gate: 153 PASS and the 4 `hz=0` builds refused.
  - `rc` is 0 on both runs.
- The bench code (`lbf_bench.py`):
  - the launch-fatal word is checked after every launch (`run_step`);
  - the payload is zeroed and the device synced before each checked step;
  - timing arms run in a fresh `random.shuffle` order every round;
  - `grid#2` (second graph) and `grid_rc` (recompiled) are the controls of the win rule;
  - the `chain*` arms are P-L2's builds on the same plan, weights and seed.
- The hazard scanner is not vacuous: it fires on real `v_readlane` -> asm `global_store ... sc1` pairs in the `hz=0`
  builds of this source.
- P-L2's sources the bench runs next to the framework have P-L2's sha256 (`sync_chain.py` 07ea4e.., `sync_plan.py`
  debb00.., `persist/sync.py`, `ccl/wire.py`).

**The final source had not run on hardware.** g / h ran `lbf.py` 0c7bc98.. and `lbf_lint.py` b63d1e... The tree now
has `lbf.py` 5948d1.., `lbf_lint.py` 05d3cf.. and `lbf_stale.py` 30d2cb.. (the mutation follow-ups), and W5 has since
changed `persist/knobs.py` (486955.. -> 9d762f..). I checked how far the kernel changed:
- Removing the one follow-up hunk (the `stagecode=loop` + `a2a` refusal in `LBKnobs.check`) from the current `lbf.py`
  gives back exactly sha 0c7bc98... So the kernel code of the measured source is unchanged.
- The `lbf_stale.py` change only removes two unused host helpers and edits the docstring.

**Re-run (stage `w7-vWP-L1-rocke` = the current copy-rocm-lib rocKE tree, `w7-vWP-L1-h` = the current harness).**
- CPU, local: `test_qwen38_lb_framework.py` + `test_qwen38_lb_sync_chain.py`, 110 / 110 pass (81 + 29) against W5's
  current `persist/knobs.py`.
- MI355X, full default bench with a new routing seed (9001). Job 68237441 `w7-vWP-L1-b`, n01-09 bus 05 (both GPUs
  passed the health check). Every gate holds on the final source:
  - hazard gate: 153 PASS, and the same 4 `hz=0` builds refused (same hazard addresses);
  - lint and LB validate clean;
  - numerics 476 / 476 (952 steps);
  - gating controls 52 / 52, informational controls 12 / 12;
  - stale harness: positives 0.000, every twin 1.000 (cache twin included), both placements;
  - timed binaries 2052 / 2052 re-verified, sentinels 228 / 228, combo re-verify 588 / 588;
  - rc 0.
- The headline ratios reproduce on the new seed (verifier run, against g / h):

| quantity (idle unless noted) | verifier (n01-09) | g / h |
|---|---|---|
| `def` / `grid` (geomean byteful; W/T/L) | 1.013 (0/6/9) | 1.011 / 1.010 (0/7/8) |
| `def` / `grid`, uni / late1 | 1.006 / 0.940 (15 WIN) | 1.004 / 0.940-0.942 (15 WIN) |
| `chain` / `chain_grid` (P-L2's own verdict) | 0.984 | 0.983 / 0.984 |
| framework `grid` / `chain_grid` | 0.975 | 0.977 / 0.978 |
| `def` / `chain` | 1.004 | 1.003-1.005 |
| `tile_w`, `a2a` vs `grid` | 1.076, 1.061 (15 LOSS each) | 1.072-1.074, 1.058-1.059 |
| one launch per layer / loop: `def`, `grid` | 1.019, 1.023 | 1.018-1.021, 1.020-1.021 |
| `def_sl` / `def` (stage loop) | 1.016 | 1.017 / 1.018 |
| layer-parity slots, `def` | 0.998 | 0.997 / 0.998 |
| A-vs-A recompiled `def`; `grid#2`, `grid_rc` | 1.000; 1.002, 1.001 | 1.000-1.001; 0.998-1.003 |
| best combination vs `grid` (`c.d.1.40.0.l`) | 1.002 (1/3/2); none below 1 on the geomean | 1.003 / 1.004 |

- **Second node not re-run.** The n07-17 twin (job 68237439 `w7-vWP-L1-a`, seed 4242) was running when n07-17 went
  `down*` (Not responding) at about 05:45 UTC. Slurm requeued it, and it then pended on `ReqNodeNotAvail`, so I
  cancelled it.
  - n07-17 coverage of this kernel therefore rests on the builder's g (68236882) and hm1 (68237090).
  - Both ran the same kernel code (the `lbf.py` diff above is host-only).
  - I have no evidence that the bench caused the outage: jobs from other lanes (`w7-l2a-*`, `qk-*`) were on the node, and the
    same bench had run on n07-17 buses 75, 65 and 15 before.

**Findings (none changes the verdict).**
1. **The "gate met" claim for "GPU microbench reproduces P-L2's verdicts" holds only in a narrow sense.**
   - What holds: P-L2's own builds reproduce in this harness (verifier 0.984), and the framework default runs at
     P-L2's default speed (`def` / `chain` 1.004).
   - What does not: P-L2's idle verdict (the default beats the barrier) does not carry over to the framework,
     because the framework's barrier is about 2.5 % faster. The doc says this openly, and WP-L3 must start from it.
2. **Section 3.1 / gate table wording.** It says "0 % of rounds below 1 in every byteful cell but one". That is true
   for the `def` arm only. The `grid` arm at idle on n07 has two more cells with rounds below 1: tp1:128:ft 0.998
   (0.65 of rounds) and tp1:128:lb 1.002 (0.25). The geomean verdict is unaffected.
3. **Section 5 table: the CPU 2 row says `lbf_stale.py` is unchanged.** In fact its kills went from 2 / 15 to 1 / 15
   (`mut2/mut.json`). The 109 total is right.
4. **Open item: 31 CPU mutation survivors in the host modules are not classified.** After CPU 2 they are
   `lbf_plan.py` 15, `lbf_lint.py` 13 and `lbf_hazard.py` 3. Section 5 classifies only the 28 hardware survivors of
   `lbf.py` / `lbf_stale.py`. The host survivors include LB-validate and bound logic:
   - `lbf_plan.py` 406, `rl >= layers or (nxt == 0 and rl == 0)` (cmp);
   - 367, the `layers = max(layers, slots + 1, ...)` horizon (const);
   - 443, the `IndexBounds.check` limit (const, arith);
   - 170 / 186, the masking predicates (cmp, arith).

   These need tests, or an equivalence argument, before WP-L2 emitters rely on LB validate for slot reuse.

Absolute times of the verifier run and its raw JSON: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/vwp-l1-raw/`
(`job-b-68237441.out`, `b/run/lbf.json`, `b_ratio.md`, `b_abs.md`). Harness copy: `hackathon-prep/qwen38/truemono-lb/vwp-l1/`.
