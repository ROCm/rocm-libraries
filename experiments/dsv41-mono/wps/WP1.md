# WP1: prefill off the decode critical path (runtime)

W12, DeepSeek-V4.1-Flash on MI355X (gfx950), TP2 no-EP, DSpark 5, the B0 recipe (bf16 KV, fp8 index, `--level 0` +
FULL graphs). Ratios only. Absolute numbers: `hackathon-prep/private-perf/dsv41-mono/WP1-abs.md` and `raw/wp1/`.

## Verdict

**GRAPH (exact-length CUDA graphs of ATOM's own eager prefill) meets D0's point expectation at C4, C8 and C16 on both rocke
nodes (mean statistic), and cuts TTFT to about 0.3-0.5x. At C2 it lands inside D0's range on the mean (1.081 / 1.059)
but below the point estimate on n07-17, and by B0's median statistic it is 0.991 / 1.036: no gain on n01-09 at C2. At C1
it is a tie with its comparator (0.99) and 0.96 of the best ATOM path on n07-17 (a small loss). At C32 / C64 (info cells,
round 18: 2 passes, A-vs-A twins of both the comparator and GRAPH, both nodes) GRAPH is 1.42 / 1.13-1.14.**

**Batched graphs (`bk=4`) on top of GRAPH: +4-6% at C16 on both nodes (exact permutation p 0.0002 on each node, every
batched arm-run faster than every GRAPH arm-run); at C8 the gain is not established (1.021 n07-17 p 0.01, 1.007 n01-09
p 0.19); at C32 / C64 +3-7% / +6-10%. `bk=4` stays default-off and is not a candidate default: its value gate is met at
the centre but not in the tail (section 2.2), and with `max=8` one C64 server died of an out-of-memory error (section 4).
`max=8` alone adds 7% at C32 on both nodes; at C64 1.14 (n07-17) but 1.05 with one losing pass (n01-09). BATCH (admission
hold) raises the per-request decode rate only by holding new requests (TTFT 1.3-3.2x), is dominated by GRAPH on every
metric that counts the wait, and stays a default-off lever. MIX was not built: for prompts of this length a measured bound
shows it cannot beat GRAPH (section 6).**

Per-request decode tok/s, WP1 arm / its comparator (same job, fresh server per arm-run, 2 rotated passes, per-request
pooled; "mean" is the statistic that sees C2's bimodal stall, "median" is B0's):

| cell | D0 expected | n01-09 mean | n07-17 mean | n01-09 median | n07-17 median | A-vs-A twin band (both nodes, mean) |
|---|---|---|---|---|---|---|
| C1 | none | 0.993 | 0.991 | 1.009 | 1.006 | 0.986-1.030 |
| C2 | 1.09 (1.05-1.12) | 1.081 | 1.059 | 0.991 | 1.036 | 1.015-1.027 |
| C4 | 1.15 (1.10-1.22) | 1.167 | 1.171 | 1.205 | 1.204 | 1.005-1.016 |
| C8 | 1.22 (1.15-1.30) | 1.233 | 1.246 | 1.270 | 1.337 | 1.013-1.017 |
| C16 | 1.26 (1.18-1.35) | 1.323 | 1.327 | 1.411 | 1.406 | 1.004-1.030 |
| C32 (info, round 18, 2 passes) | larger | 1.415 | 1.417 | 1.537 | 1.546 | 0.997-1.006 (comparator and GRAPH twins) |
| C64 (info, round 18, 2 passes) | larger | 1.130 | 1.138 | 1.222 | 1.215 | 0.964-1.010 |

C1-C16: jobs wp1-t1n01 / wp1-t2n07. C32 / C64: round 18 (wp1-s13 n01-09, wp1-s14 n07-17), GRAPH (`gm1` + its twin `gm1_b`)
over the comparator (`base` + its twin `base_b`), pooled; every GRAPH arm-run beats every comparator arm-run on both
nodes (4 vs 4 runs, exact permutation p 0.029, the floor of that design). The earlier single-pass C32 / C64 cells
(wp1-t2n07: 1.449 / 1.135) agree.

Reading the C1 / C2 rows honestly: C1 is a tie (as D0 expects); C2's mean gain (1.06-1.08) is inside D0's range but its
median is 0.991 on n01-09, i.e. the gain comes from fewer stalled requests, not from a faster typical request.

That is mono + GRAPH vs mono (`gm1/base`). Native + GRAPH vs native (`gm0/nat`) gives the same picture (C8 1.25-1.26,
C16 1.24-1.30 mean). Against ATOM's **best** served path per cell (the higher of mono, its twin and native), the best WP1
arm is (mean) C1 0.96-0.99 (0.962 on n07-17: a small loss), C2 1.04-1.05, C4 1.15-1.17, C8 1.22, C16 1.25-1.32; C32 1.41-1.42
and C64 1.11-1.13 for GRAPH (round 18, against the better of the comparator and its twin). The best round-18 arm
(batched graphs + `max=8`) reaches C32 1.56-1.57 and C64 1.33-1.35 on that basis, but that arm is not usable as is
(section 4: one out-of-memory crash).

Other effects of GRAPH (both nodes):
- TTFT falls to 0.30-0.48x of the comparator (the prefill is 0.24x as long and no longer waits behind host launches).
- Aggregate output tok/s rises 1.14-1.17x at C1, 1.24-1.30x at C2-C4, 1.30-1.35x at C8-C16 (C32 1.41, C64 1.09).
- The P1 stall metric (share of GPU time in prefill steps, HOSTTL, served C8 / C16) falls from 0.31-0.36 / 0.36-0.38
  to 0.11 / 0.14. The prefill step's GPU span is 0.24x the eager one (both nodes).
- The decode-only step does not regress: at matched batch sizes (HOSTTL, every bs with >= 10 steps in both arms) the
  GRAPH/comparator decode step ratio is 0.987-1.026, median 0.994.
- Server start: the 161 graph captures (lengths 40-200) add about 0.2-0.25 of the comparator's start-to-ready time,
  outside every timed window.

Relation to the 3x goal: WP1 alone moves the best served path 0.96-0.99x (C1), 1.04-1.05x (C2), 1.15-1.17x (C4), 1.22x
(C8) and 1.25-1.32x (C16); with batched graphs about 1.30-1.40x at C16 (GRAPH x the round-18 batched factor, not measured
in one job against the comparator). It is one factor of D0's composition; it does not reach 3x at any cell, far from it,
as D0 predicted. At C1 it cannot help by construction (a request's own prefill is outside its decode rate). WP1 does not
address TP1.

## 1. What was built (ATOM-dsv41, all additive, all default-off)

| knob | what | files |
|---|---|---|
| `ATOM_DSV41_RK_PREFILL_GRAPH=<lo>-<hi>[,cap=k][,max=m][,check=..][,mut=..]` | One CUDA graph per exact prompt length `lo..hi` of ATOM's **own eager prefill forward** (bounded replay, prefill FFN, every kernel at the eager shape), captured at server start inside ATOM's graph-capture window; a fresh one-request final-chunk prefill of a captured length replays it. Scheduler side: at most `cap` (1) new requests per prefill step when the first is graph-eligible, unless `m` or more wait (`max`, default off). | `atom/models/deepseek_v41/rk/prefill.py`, `rk/pfsched.py`; hooks in `model_runner.py` (`run_model` eager branch, `capture_cudagraph`), `scheduler.py` (Phase-2 admission) |
| `...GRAPH=<lo>-<hi>,cap=4,bk=4[,bt=16]` (batched graphs) | 2..`bk` fresh requests of `lo..hi` tokens admitted together run as ONE padded forward of `Tb` rows (total rounded up to `bt`), replayed from a graph keyed (requests, `Tb`): 91 graphs for `bk=4`, `bt=16`. Made split-independent instead of exact: fixed-capacity metadata (`_prepare(running_tokens=Tb, max_q_len=hi)`, compress plans at capacity with sentinel tails, padding rows batch id -1), the index tile table always spans `ceil(hi / block)` PAGE columns, and bounded replay off for the step (every layer on every row, as a padded or decode step). Tolerance-class: differs from ATOM's eager batched prefill in layer 39 running all rows and, for prompts over the 128-row window, in the decoder windows (full, not truncated); gated against an eager run of the same bucketed step (2.2) and end to end | `rk/prefill.py` (`_install_bucket`: patches `prepare_prefill`, `unit_tiles`, `runtime.replayable` only for bucketed steps) |
| `ATOM_DSV41_RK_PREFILL_BATCH=<n>,<ms>` | While decode runs, hold new-request admission until `n` wait or the oldest waited `ms`; then admit the whole queue into one eager prefill | `rk/pfsched.py`, `scheduler.py` hook |
| `ATOM_DSV41_RK_PREFILL_DETROUTER=1` | Gate diagnostic only: the MoE gate above 16 rows runs ATOM's fixed-order router 16 rows at a time instead of hipBLASLt | `rk/prefill.py` |
| `ATOM_DSV41_RK_PREFILL_MIX` | **not built** (section 6) | - |

Knob off: the scheduler's `_RK_PF_ADMIT` and the runner's `_RK_PF` are `None`; the only change is one `None` test in
each hook, so the served path is unchanged.

How the capture is made safe (all inside `rk/prefill.py`):
- Bounded replay (`bounded_replay.build_replay_plan`) uploads its plan from the host **inside** the forward, including
  the request's STATE slot. Under the knob that upload goes to fixed per-length device buffers. The capture records reads
  of them. Before each replay the runner builds the same plan on the host and copies it in, then hands it to the step
  (`step.replay`), as the eager forward would have, for the DSpark context-KV write that reads it afterwards.
- Everything else the prefill forward reads is already at fixed addresses (the same contract ATOM's decode capture
  relies on): positions, `cu_seqlens_q`, compress plans, visibility, indptrs, block tables, the STATE slot map, Engram's
  overlap staging.
- The output is copied inside the graph to a buffer outside the graph pool; the graphs share one private pool.
- At replay the step's context signature (scheduled / running bs and tokens, unified, DP / TBO state, width) must equal
  the captured one, or the step runs eagerly. Eligibility: one request, position 0, no prefix hit / replay, final chunk,
  no image, no scheduler draft anchor.

## 2. Gates

### 2.1 Bitwise vs eager is not attainable: ATOM's eager prefill is not run-to-run deterministic

The spec asked for "prefill outputs bitwise vs eager". The in-situ check (`check=twin3`: snapshot the request's STATE
slot, PAGE units and DSpark aux rows, run the eager forward, restore, re-prepare the step, run it again, three times)
shows that **two eager prefills of the same step from the same state never agree bitwise** (40 of 40 pairs differ, both
nodes, every region). A forward-hook bisect over the layers (`check=bisect`) located the sources:
1. the MoE gate at prefill widths (> 16 rows) is a hipBLASLt GEMM (Stream-K: its K split follows the CU count, ATOM's own
   `router.py` docstring says so); first differing module, relative error about 3e-3;
2. with that replaced by the fixed-order router (`DETROUTER`, diagnostic), the next one is the routed-expert `fused_moe`
   (relative error 5e-4 to 8e-4, atomic accumulation).

Routing flips amplify these into an output relative error of 0.11-0.22 (median 0.14) and a per-layer KV divergence from
layer 1 on. This is a property of ATOM's served prefill, independent of WP1. A bitwise gate therefore cannot separate a
correct graph from a broken one; the gate below does.

### 2.2 The gate used: write footprint + value distribution + must-fail mutants (served config, no diagnostics)

Per eligible prefill, in situ on the served server (`check=graph` / `twin3` / `batch`):
- **Write footprint**: which 4096-byte chunks of the request's STATE slot, PAGE units (main pages + index planes) and aux
  rows the forward changed, and which output rows are non-zero, must be identical between the two runs. Values may
  differ (2.1); the set of bytes written may not: a stale address, a wrong slot, a dropped or redirected page, a wrong
  length bucket all change it. 16- and 256-byte grains are reported too. At 16 bytes low-entropy index rows make even
  eager-vs-eager disagree on about 0.03% of chunks. At 256 bytes the first gate jobs were clean (180 of 180 checks,
  wp1-t2n07), but later jobs showed rare mismatches (5 of 228 correct-path checks in wp1-s10 / s11: four at a server's
  first prefill, one of 2 page chunks mid-run), so the gate became the 4096-byte grain, where every correct-path check of
  every job agreed and every mutant failed. **This is a weaker gate than first designed, and the choice was made after
  seeing correct-path data**: a wrong write that stays inside a 4096-byte chunk the eager forward also writes is invisible
  to it. Both mutants are still caught at 4096 B. Round 18 (wp1-a5) re-ran the 256-byte grain on 6 fresh servers: 576 of 576
  checks OK (both ranks), so the 256-B mismatches are rare, but the gate stays stated at 4096 B.
- **Values** (as specified): the graph-vs-eager output relative error must sit in the eager-vs-eager twin distribution.
  Verdict below the table: **met at the centre, not met in the tail**; for batched graphs the twin yardstick is small.

| comparison (both ranks) | job | checks | footprint 4096 B | footprint 256 B | out_rel median (range) |
|---|---|---|---|---|---|
| eager vs eager (twin3, 1v2 and 2v3) | t2n07 | 40 | 40 OK | 40 OK | 0.14 (0.116-0.167) |
| GRAPH vs eager (C1, C4, C8 traffic, 26 lengths) | t2n07 | 140 | 140 OK | 140 OK | 0.142 (0.113-0.255) |
| GRAPH vs eager, with batched graphs installed (C8 / C16, 24 lengths) | s11 | 78 | 78 OK | 76 OK | 0.135 (0.111-0.243) |
| batched graph vs an eager run of the same bucketed step (k = 2-4, 36 lengths) | s11 | 114 | 114 OK | 112 OK | 0.091 (0.012-0.433) [1] |
| batched graph, k = 2 only | s10 | 24 | 24 OK | 24 OK | - |
| mutant `bucket`, single (replay the next length's graph) | t2n07 / s11 | 16 / 2 | all DIFF | all DIFF | 1.0 |
| mutant `bucket`, batched (replay the next row bucket's graph) | s11 | - | runner fault (illegal memory access) at the first mutated step: fails loudly | | |
| mutant `page`, single (the request's last PAGE redirected) | t2n07 / s11 | 16 / 36 | all DIFF (every page chunk) | all DIFF | 0.139 / 0.143 |
| mutant `page`, batched (request 0's last PAGE redirected) | s11 | 60 (16 steps) | every mutated step caught (16 of 16; request 0 DIFF, every page chunk) | same | - |
| BATCH: batched eager prefill vs each request's own (k >= 2, C4 / C8) | t2n07 | 50 | 50 OK | 50 OK | 0.136 (0.113-0.19) |
| **round 18 (wp1-a5, n01-09, stage v12)**: eager vs eager, singles at the GRAPH gate's traffic (C1 / C4 / C8, `check=twin`) | a5 | 140 | 140 OK | 140 OK | 0.146 (0.110-0.186) |
| eager vs eager with batched graphs installed: singles (C8 / C16) | a5 | 88 | 88 OK | 88 OK | 0.134 (0.117-0.186) |
| eager vs eager of the same **bucketed** step (`btwin`, new in v12; k = 3-4 only) | a5 | 104 | 104 OK | 104 OK | 0.097 (0.006-0.284) |
| GRAPH vs eager, batched graphs installed, 6 fresh servers (s9's configuration) | a5 | 522 | 522 OK | 522 OK | 0.141 (0.104-0.209) |
| batched graph vs eager of the same bucketed step, same 6 servers (k = 2-4) | a5 | 630 | 630 OK | 630 OK | 0.099 (0.005-0.610) |

[1] A bucketed step runs every layer on every row (section 1, batched graphs), so its output has values on
all of a request's rows, not only the last; the relative error is over all of them.

**Value gate, honestly.** Each check is logged by both ranks with identical values; the statistics below count one value
per check (rank 0). The eager-vs-eager yardstick of the first round was small (20 values, 9 lengths, C1 traffic only:
0.116-0.167), and GRAPH-vs-eager went past it (0.255 in t2n07, 0.243 in s11), so the gate **as written was not met**.
Round 18 measured the yardsticks properly:
- Singles: a pooled eager-vs-eager twin of 134 values (t2n07 + a5, the GRAPH gate's traffic and lengths) spans
  0.110-0.186. GRAPH-vs-eager over 370 values (t2n07, s11, a5) has the same centre (Mann-Whitney p 0.31-0.93 per job,
  P(graph > twin) 0.45-0.50), but 10 of 370 values lie above the twin maximum where about 3 are expected if the two were
  one distribution (max 0.255). The centre matches; the tail is somewhat heavier than the twin sample shows.
- Bucketed steps: there was no bucketed eager-vs-eager twin before round 18 (the verifier was right). The new `btwin` mode
  (stage v12: `check=twin` with `bk` set runs a bucketed step eagerly twice) gives 52 values, k = 3-4 only (none at
  k = 2), 0.006-0.284. Batched-graph-vs-eager has the same centre (Mann-Whitney p 0.73-0.77), but 21 of 372 values lie
  above the twin maximum (about 7 expected; max 0.61), and at k = 4 12 of 200 against 4.9 expected. **The value gate is
  not met in the tail for batched graphs**, and the k = 2 case has no twin at all.

So: GRAPH (singles, exact graphs of the eager forward) passes the footprint gate and matches the eager twin's centre with
a slightly heavier tail; its end-to-end accuracy is indistinguishable from the comparator's (2.3). Batched graphs are a
tolerance-class path whose bucket-level tail is heavier than eager's own run-to-run spread; their end-to-end accuracy gate
(2.3) passes, but they stay default-off and are not a candidate default.

The `page` mutant is the case the value comparison alone misses (its output error sits inside the twin band); the
footprint catches it every time. Native (mono off) GRAPH vs eager was also checked (wp1-s7: 64 checks; 16-byte
mismatches at the twin's rate; 256 / 4096 grains were added after that job).

### 2.3 Accuracy (end to end)

GSM8K, all 1319 questions, same job, fresh servers, CONC 16, an A-vs-A twin of the comparator. Per-question paired
comparison (McNemar, exact two-sided) next to the score differences:

| run | GRAPH range | GRAPH-arm prefills replayed | flexible: GRAPH vs the twins | strict: GRAPH vs the twins | twin vs twin |
|---|---|---|---|---|---|
| 0-shot chat (wp1-s6a1, n01-09) | 40-200 | 1314 of 1316 (last 30-s log) | between the twins (-0.005 / +0.004); McNemar p 0.53 / 0.68 | 0 for every arm (0-shot format) | -0.009, p 0.24 |
| 3-shot multi-turn, B0's format (wp1-a2, n01-09) | 40-1024 | 93% (1038 of the 1111 counted at the last 30-s log) | -0.007 / -0.011 below; discordant 47 / 54 (twin pair: 47); p 0.24 / 0.076 | -0.009 / -0.005 below; p 0.29 / 0.63 | +0.004 flexible (p 0.56), -0.005 strict (p 0.64) |
| 3-shot replicate (wp1-a3, n01-09; 2 GRAPH + 2 comparator servers) | 40-1024 | - | GRAPH 0.000 / +0.004 vs the twins' range (lower / upper) | inside the twins' range (twins 0.021 apart) | +0.001 flexible, -0.021 strict |
| batched graphs `bk=4` (tolerance class), 0-shot chat (wp1-a4, n01-09; 2 + 2 servers) | 40-200, `cap=4,bk=4` | 159 batched + 808 single replays at the last log | +0.000 / +0.002 vs the higher twin (at or above both twins); discordant 66-81 (twin pair: 97), McNemar p >= 0.82 | 0 for every arm (0-shot format) | +0.001, p 1.0 |
| **round 18**: batched graphs `bk=4`, 0-shot chat **plus a system instruction asking for a final `#### n` line** (wp1-a5, n01-09; 2 + 2 servers, order gb4, base, gb4 twin, base twin) | 40-200, `cap=4,bk=4` | 632-641 single + 82-90 batched replays, 2 eager, by the 30-s log before the end (the logged counters reached 838-844 + 104-112) | 0.9409 / 0.9500 for the two batched servers vs twins 0.9492 / 0.9378: inside the twins' range; McNemar vs each twin p 0.088-1.0 | 0.9416 / 0.9500 vs twins 0.9500 / 0.9386: inside the twins' range; p 0.117-1.0 | 0.0114 flexible / strict, p 0.017 |

Read honestly: in 0-shot the GRAPH arm sits between the twins. In wp1-a2's 3-shot it was below both twins on both
metrics, outside the literal twin range, with no significant paired difference. The replicate (wp1-a3, n01-09, fresh
servers, order base, GRAPH, base twin, GRAPH twin) puts both GRAPH runs inside that job's twin range on strict-match and
at or above it on flexible-extract (one equal to the lower twin, one 0.004 above the higher).

Pooled over the seven 3-shot runs (3 GRAPH, 4 comparator, two jobs), per-question pairs:
- flexible-extract: every GRAPH-vs-comparator pair disagrees on 47-59 questions (comparator-vs-comparator: 47-63), net
  -14 to +6 questions (comparator pairs: -5 to +12); smallest McNemar p 0.076 (comparator pairs: 0.14). Means: GRAPH
  0.003 below the comparator (0.4 of a standard error).
- strict-match: 94-110 discordant (comparator pairs: 105-114), net -19 to +23 (comparator: -13 to +27); smallest p 0.031
  (comparator pairs: 0.012). Means: GRAPH 0.0007 below.

The GRAPH arm's accuracy is indistinguishable from the comparator's own run-to-run variation, which is itself as large
as the differences seen (ATOM's served path is not deterministic, 2.1).

Batched graphs (tolerance class): the first gate (wp1-a4) used plain 0-shot prompts, the only GSM8K form inside their
40-200 range, and there strict-match is 0 for every arm, so **its strict half was vacuous** (the verifier was right). Round
18 (wp1-a5) fixed that without leaving the graph range: 0-shot chat with a system instruction to end with `#### <answer>`
(`acc_si_arm` in harness h18), which makes strict-match score 0.94-0.95 for every arm. Both batched-graph servers sit inside
the comparator twins' range on both metrics, with no significant paired difference against either twin. The comparator's
own twin pair differs by 0.011 at McNemar p 0.017 (15 net questions), the same order as any arm-vs-arm difference: served
accuracy noise on this path is large (2.1, rule 258 of the cluster notes).

Long context:
- GRAPH (wp1-a2): passkey (P3's `niah.py`), 20 samples each at 8K and 24K tokens: 20/20 and 20/20 for the GRAPH arm and
  both twins. These prompts are past the graph range and prefill eagerly, chunked as ATOM chunks them: GRAPH leaves them
  unchanged, as designed.
- Batched graphs (round 18, wp1-a5): the same passkey set run **while a C16 stream of 512 short GSM8K prompts runs on the
  same server**, so the long chunked eager prefills interleave with batched / single graph prefills and the bucketed
  metadata patches (`_install_bucket`: `prepare_prefill`, `unit_tiles`, `runtime.replayable`) are live: 20/20 and 20/20 at
  8K / 24K for both batched-graph servers and both comparator twins; 512 of 512 short requests completed on every server.
  Each batched-graph server logged 42-45 batched and about 360 single graph replays next to 55-56 eager prefill steps
  (the long prompts' chunks) in that run.

### 2.4 CPU gate

`tests/test_deepseek_v41_rk_prefill_admission.py` (6 tests): knob off, parsing, BATCH hold / release / re-arm, no hold
without decode, GRAPH cap and `max`. Seven hand mutants of `pfsched.py` (comparison flips, release flag, cap and max
arithmetic, eligibility test) each fail at least one test.

## 3. Timing method

- `hackathon-prep/dsv41/wp1/` harness: a fresh server per arm-run, B0's client (`wp1_bench.py`, a copy of
  `bench/dsv41_bench.py` plus the timed window in CLOCK_MONOTONIC), cells C1:16 C2:16 C4:32 C8:32 C16:64, untimed warm-up
  at the cell's concurrency, 2 passes with the arm order rotated, an A-vs-A twin of the comparator (`base_b`).
- Arms: `base` (mono + DSpark), `base_b` (twin), `nat` (native + DSpark), `gm1` (mono + GRAPH 40-200), `gm0` (native +
  GRAPH 40-200).
- Statistic: per-request decode tok/s pooled over passes. Both the median (B0's) and the mean are reported: at C2 the
  per-request distribution is bimodal (requests that took a foreign prefill vs those that did not), and with 16 samples
  the median lands between the modes; the mean sees the stall.
- Both nodes: n01-09 (job wp1-t1n01) and n07-17 (wp1-t2n07). Stall metric: P1's HOSTTL (`ATOM_DSV41_RK_HOSTTL`) in
  separate arm-runs (C8 mono, C16 native).

## 4. Levers kept default-off ("rejected on base B0" unless stated)

| lever | result | status |
|---|---|---|
| BATCH `n,ms` | First runs (wp1-s1 `4,50`; wp1-t3 `2,30` / `4,100` / `8,250`) were **invalid**: the hold never fired (TTFT and prefill batch sizes identical to the comparator). Bug: the release re-armed only when `admit()` saw an empty queue, which the scheduler never passes (Phase 2 asks only while something waits). Fixed (a release covers the requests waiting when it fired); the CPU test now follows the scheduler's calling contract and a mutant of the old behaviour fails it. Fixed re-run, wp1-t4, n07-17, 2 passes (mean, C4 / C8 / C16): `2,30` 1.03 / 1.02 / 1.03; `4,100` 1.11 / 1.15 / 1.20; `8,250` 1.30 / 1.30 / 1.32 (1 pass: the second pass's server hung at its first batched warm-up prefill in an NCCL broadcast and was not timed; one occurrence, cause not found). **These decode-rate gains are mostly bought with TTFT**: 1.28x / 1.93x / 3.2x the comparator's TTFT at C8 for the three seeds (GRAPH: 0.46x), because a held request is not decoding, which lowers the effective concurrency the others share. Counting the wait, tokens per request-second (tokens / request e2e) is 0.93-1.15 for `4,100` and `8,250` against GRAPH's 1.33-1.40 in the same job, and aggregate output tok/s 1.02-1.15 (C4 `8,250` not measured) against GRAPH's 1.28-1.32. Dominated by GRAPH on every metric that includes the wait; single node, not confirmed on n01-09 | default-off; "rejected on base B0 (dominated by GRAPH)" |
| ATOM's own `--scheduler-delay-factor 0.5` (comparator lever, not WP1's) | wp1-t3 and wp1-t4 (n07-17, 2 passes each): 0.99-1.06 vs the comparator, inside the twin band; TTFT 1.0-2.0x. With GRAPH (`gm1df`): 0.98-1.02 of GRAPH alone | ATOM option, no gain |
| GRAPH `max=8` (eager-batch the queue when 8 or more wait) | info cells, n07-17, 1 pass, no twin (wp1-t3; superseded by round 18, next row): vs GRAPH `cap=1` 1.06 (C32) and 1.14-1.15 (C64), i.e. 1.54 / 1.28 (mean) and 1.75 / 1.37 (median) of the comparator: a burst of arrivals is cheaper as one eager prefill than as a queue of graph prefills. At C4-C16 (wp1-t4, 2 passes) 0.99 / 1.00 / 1.04 of GRAPH alone (bursts are rare there) | default-off; C >= 32 info gain confirmed with twins in round 18 (next row), not a default (C64 n01-09 not established; untested burst-memory margin, see the next row) |
| GRAPH batched graphs `cap=4,bk=4,bt=16` | **Round 18 (wp1-s13 n01-09, wp1-s14 n07-17; the same job holds GRAPH, a GRAPH twin `gm1_b`, batched graphs and a batched twin `gb4_b`; C8 / C16 4 rotated passes; per arm-run means, exact permutation test over the 8 + 8 arm-runs):** C8 1.021 (n07-17, p 0.012) and 1.007 (n01-09, p 0.19): **not established** (GRAPH twins 1.007 / 0.998; per-pass twin ratios 0.965-1.040, as large as the effect). C16 1.040 (n07-17) and 1.061 (n01-09), p 0.0002 on each node, every batched arm-run above every GRAPH arm-run on both nodes: **established**; the GRAPH twin is 0.990 / 1.036 (n01-09's twin itself drifted, p 0.03), so read the gain as about +4-6%. Batched twin 0.992-1.003. C32 / C64 (2 passes, twins of comparator and GRAPH): 1.070 / 1.105 (n07-17), 1.033 / 1.059 (n01-09); at C32 on n01-09 the arm-runs overlap GRAPH's. With 2 arm-runs against 4 the exact permutation p cannot go below 0.067, so the info cells are "consistent on both nodes", not significant. TTFT equal to or below GRAPH's at C8-C32, 0.80-0.85x at C64. Earlier rounds (s11, s12n01; 1.01-1.04 at C8 / C16 without a GRAPH twin) are superseded by these numbers. 30 ms admission hold on top (`BATCH=4,30`, s11 / s12n01 only): 1.02-1.05, no GRAPH twin, not re-tested. Gates: footprint at 4096 B (and 256 B in round 18) clean; value gate met at the centre, **not in the tail** (2.2); accuracy (strict + flexible) and long context passed (2.3) | default-off; **not a candidate default** (value-gate tail; C8 not established); the C16-C64 gain is real on both nodes |
| GRAPH `max=8` and `cap=4,bk=4,max=8` (C32 / C64 info cells) | Round 18, both nodes, 2 passes, against GRAPH + its twin (the earlier single-pass cells in t3 / s12n01 are superseded): `max=8` alone 1.074 / 1.136 (n07-17) and 1.074 / 1.050 (n01-09; one C64 pass at 0.93 of GRAPH). Batched graphs + `max=8`: 1.109 / 1.192 (n07-17) and 1.102 / 1.198 (n01-09), every arm-run above every GRAPH arm-run; against the comparator 1.57 / 1.36 and 1.56 / 1.33. **But one of its four C64 servers died** (s14 pass 2): `max=8` admitted the whole queue (53 requests, 4560 tokens) into one eager prefill, and aiter's fused-MoE stage-1 output allocation failed with HIP out-of-memory inside ATOM's memory budget (the batched graphs' private pool and 252 graphs leave less headroom than GRAPH alone; `max=8` alone had no runner death in 4 of 4 C64 servers). Cause attributed; not fixed: `max` needs a token cap on the eager burst (not built) | default-off; `max=8` alone: info gain, n01-09 C64 not established; `bk=4,max=8`: **rejected on base B0 until the burst is token-capped** (OOM) |
| DETROUTER | diagnostic only (changes the prefill routing's summation order) | never a served lever |

## 5. What WP1 leaves on the table

- After GRAPH the remaining prefill share is 0.11 (C8) / 0.14 (C16). A graph prefill of a ~90-token prompt is 1.6-2x a
  C8 decode step on the GPU: its routed MoE touches most of the 384 experts, so it is bytes-bound, not launch-bound.
  Batched graphs (built, section 1) share those expert reads when 2-4 requests arrive together; at C8 / C16 that is
  rare, so at C8 the gain is not established; at C16 it is +4-6% and at C32 / C64 +3-10% (round 18, both nodes).
- Large bursts (every cell's first wave, C64): one eager batched prefill (`max=8`) beats a queue of graph prefills;
  graphs for more than 4 requests were not captured (graph count and capture time grow with requests x row buckets).
- Reliability. Two hangs were seen in earlier rounds and **neither is attributed**:
  - wp1-s9: the first checked prefill of a `bk=4` server (`check=graph`) stuck on the GPU (one 88-token single prefill
    after 91 batched captures; nothing in the log after "Scheduled prefill batch").
  - wp1-t4: a batched warm-up prefill under BATCH `8,250` (no graphs installed) stuck in an NCCL broadcast until the
    watchdog's timeout. BATCH is dominated and was not re-run.
  Round 18 ran 35 fresh `bk=4` servers (12 per timing job on each node, 4 accuracy / long-context servers, the twin
  yardstick server, and **6 fresh servers in exactly wp1-s9's configuration**, `check=graph` at C8 / C16) with a stall
  watchdog that dumps the process list, GPU use and every Python stack after 300 s of server-log silence. No hang and no
  stall recurred. The one failure was the `bk=4,max=8` out-of-memory at C64 (section 4), which is attributed. So the
  s9 hang is 1 in about 58 `bk=4` server starts over all WP1 rounds, not reproduced; cells keep running under a time box
  (`CELL_TIMEOUT`), the dead-runner watchdog and now the stall dump.

## 6. MIX (chunked prefill rows on the decode step): not built, and why

ATOM's V4.1 step has one decode/prefill mode for the whole batch (`BatchStep.decode`); the decode graph's verify rows
are tentative and refuse a span at position 0 or longer than 6 rows (`PagedAttentionCache.begin_step`). So rows can
only ride a decode step as forced 6-row verify blocks after a head prefill. Measured bound, both nodes:
- the head prefill stays (any short one costs about a C1 decode step or more);
- each 6-row block adds the cost of one more request to the decode step: from P1's no-profiler step table, 6 more rows
  cost 0.29x of a C1 mono step at C1 -> C2, and more once mono's 48-row limit or a graph-ladder rung is crossed (C8 -> C9
  pads to 16, 1.23-1.27x);
- a 90-token prompt is 15 blocks, so MIX costs at least 15 x 0.29 = 4.4 C1-steps of decode GPU time, against 3.35
  C1-steps for GRAPH's whole prefill (its measured GPU span, 0.24x of the eager one), and it spreads the prompt's expert
  reads over 15 steps instead of sharing them in one.

MIX can only win for long chunked prompts, which the WP1 cells do not contain. It is recorded here as not built; the
spec's MIX gates (GSM8K + passkey 8K / 24K) were not run.

## 7. Pending and not done

- `max` burst token cap (needed before `bk=4,max=8` can be used): not built.
- A k = 2 bucketed eager-vs-eager twin, and a larger bucketed twin sample, to settle the batched-graph value tail: not run.
- BATCH `8,250` NCCL hang: not re-run (BATCH is dominated by GRAPH).
- MIX: not built (section 6).
- TP1: out of WP1's scope.

## 8. Reproduce

- ATOM stage: `hackathon-prep/dsv41/wp1/mkstage.py <name>` (packs `dev/ATOM-dsv41`); stages used: `atom-wp1-v5` (t1n01),
  `atom-wp1-v8` (t2n07, s7, t3, a2), `atom-wp1-v11` (a4, s12n01). Harness: `hackathon-prep/dsv41/wp1/h8/` (run_wp1.sh, plan files), job body
  `wp1/wp1_job.sh`, submitter `wp1/submit_when_free.sh` (holds the workflow's GPU cap).
- Summaries: `wp1/wp1_summ.py <results> --twin base_b:base --pairs gm1:base,gm0:nat --best-num gm1,gm0
  --best-den base,base_b,nat [--stat mean]`; gates: `wp1_checks.py <server log>`; stall: `wp1_htl.py <hosttl dir>
  <bench json>`.
- Jobs: gates wp1-s1..s7 (68256829, 68256939, 68257116, 68257171, 68257338, 68258312), timing wp1-t1n01 (68257253),
  wp1-t2n07 (68259019, gates + timing + stall + info cells).
- Round 18 (verifier follow-up): stage `atom-wp1-v12` = `atom-wp1-v11` with only `rk/prefill.py` replaced
  (`wp1/mkstage_overlay.py`; the change adds `check=twin` for bucketed steps, logged `btwin`, a diagnostic path that only
  runs under the knob with `check=twin`), harness `wp1/h18/` (stall dump, `acc_si_arm`, `niah_mix_arm`; plans
  `plan_s13.sh` for both timing jobs, `plan_a5.sh`), jobs wp1-s13 (68264824, n01-09), wp1-s14 (68264849, n07-17), wp1-a5
  (68265106, n01-09). Summaries: `wp1/wp1_r18.py <results> --twins .. --pairs ..` (per-pass ratios),
  `wp1/wp1_r18_perm.py <results> <conc> <armsA> <armsB>` (exact permutation over arm-run means); value-gate analysis
  from the `W12RKPF check` lines of the server logs.
