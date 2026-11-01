# Search-driven tuning of the true monokernel: guide for the tracks

How W2 (small batch), W4 (large batch + CCL) and the Rust direct-compiler track plug their knobs and objectives into
the tuner, the budget rules, and the RSI loop. Method: `dev/rocKE-Search-Driven-Kernel-Optimization.md` and DESIGN.md
9.1 / 9.2 / 12.4. Inventory of what came from rocm-libraries: `INVENTORY.md`. Ratios only in this tree; absolute
numbers in `hackathon-prep/private-perf/qwen38-true-mono/search/`.

## 1. What is where

| piece | path (rocKE root `dnn-providers/hip-kernel-provider/rocke/`) |
|---|---|
| search engine (ask/tell strategies, evaluator, DB, driver) | `platform/python/rocke/dispatch/tuning/search/` (copied from rocm-libraries) |
| the mono tuner | `library/kernels/gfx950/qwen38_mono/tune/` |
| knob spaces, legality, tuning ids | `tune/knobs.py` (`Knob`, `DataclassKnobs`, `LeverStringKnobs`, `CompositeKnobs`) |
| every registered space | `tune/catalog.py` (framework, toy chain, fused MoE, MoE port, B3, stage-fn ABI, MXFP4 GEMV, CCL, ATOM schedule levers, Rust direct levers, the N1-N11 registry) |
| strategies | `tune/strategies.py` (`SignedLookback`, `mono_portfolio`, `GainPrior`) + the engine's (`hill`, `de` SHADE, `tpe`, `gp`, `portfolio`, `sh`, `hyperband`) |
| measurement discipline | `tune/gate.py` (shuffled rounds, controls, 9.1 / e2e win rules, SCLK / power by PCI BDF) |
| objectives | `tune/objective.py` (stage / layer / round / e2e, gap share to the headline targets) |
| tuning DB, gain ledger, defaults, selector | `tune/tdb.py` |
| batch jobs | `tune/evaluate.py` (`JobEvaluator`, `LocalTransport`, `Q38Transport`, `run_batch` = `runner`) |
| a live backend (template) | `tune/live_toy.py` (`ToyChainBackend`) |
| replay of recorded A/B data | `tune/replay.py`, `tune/experiments.py` |
| CLI | `python -m kernels.gfx950.qwen38_mono.tune {catalog,replay-grid,replay-sched,live-toy,runner,export-defaults}` |
| tests (CPU) | `library/tests/test_qwen38_tune.py` |
| engine hooks for the platform owner | `tune/vendor/merge/0[1-4]-*.patch` (INVENTORY section 2) |

## 2. Plug in a knob (every track)

1. **Make the lever a default-off field** on its owner (a frozen dataclass such as `FusedKnobs`, `B3Knobs`,
   `FrameworkKnobs`, `CclKnobs`, a spec), validated in `__post_init__`; knob off = byte-identical build (DESIGN 9.2
   step 4). A losing lever stays a field.
2. **Register it in `tune/catalog.py`** in the owner's factory: `Knob(path, values, kind, gate=, numerics=,
   structural=)`.
   - `values`: every built value (losers included). For categorical / bool knobs the shipped default is first
     (level 0); the catalog test fails otherwise.
   - `gate`: the knob is read only while another knob is off its default (`numa_flip_arg` under
     `mailbox.numa_copies`); gated-off values drop out of the key.
   - `numerics=TOLERANCE` for anything that is not bitwise (an N lever, `exp="hw"`, a narrower `ar_dtype`): such
     configs must pass the T0-T5 ladder of DESIGN 12.4 instead of the bitwise gate.
   - `structural=True` for a base change (NUMA copies, a split, a hop mode, a launch structure): applying it re-opens
     the climb over every retained lever.
   - Measurement-only and test-only fields (stamps, `w_null`, `mutate`, hooks) go to `excluded` with a reason; the
     mutants are what the gate uses.
3. **Legality**: the owner's own validation is the first rule (illegal configs never compile; the engine repairs them
   to the nearest legal one). Add adapter `rules` for what the owner cannot see (LDS arena vs the mono's, known
   miscompiles, `is_valid_spec`). A level legal only together with another (`m4_interleave` needs `m4_guard=none`) is
   fine: the climber's companion move and the engine's repair find the pair. List it in the catalog test's
   `ALONE_REFUSED` with the rule's wording.
4. **Identity**: nothing to do. Keys are `rocke.dispatch.tuning.identity.config_key` over the ABI, arch, adapter,
   variant, the canonical knobs and a fingerprint of the base's defaults; a changed in-tree default re-keys every
   record (a stored pick can never silently mean another build). Tuning ids read `<adapter>.<variant>@<key>`.
5. **Accuracy-gated levers (N1-N11)**: add the field, then set `path` / `values` of its `catalog.N_LEVERS` entry;
   `attach_n_levers(adapter)` adds it to the owner's space as a `tolerance` knob as soon as the field exists.
6. **Orders are knobs too** (PLAN decision 15): a transform / pass / restructure ORDER is
   `Knob.sequence(path, items)`. Its levels are orderings encoded `"a>b>c"` (level 0 = the given order; every
   permutation up to 720, else the given order, every single-item move, the reversal and seeded samples). The owner's
   field receives a tuple (`knobs.seq_decode`). The climber's arms for an order knob are one ORDER move away
   (one item moved to another position, which covers adjacent swaps), not all orderings at once; the lookback treats
   orderings as levels like any other.
7. **Lever strings** (ATOM `ATOM_Q38_SCHED`, `ROCKE_DIRECT_LEVERS`): `LeverStringKnobs` renders the string;
   `catalog.sched(kind, variant)` and `catalog.direct()` are the templates; `catalog.direct_env(knobs)` gives the
   environment a direct-compiler build needs (the evaluator fingerprints it).
8. Run `python -m pytest library/tests/test_qwen38_tune.py` (CPU, seconds): the catalog contract covers the new knob.

## 3. Plug in an objective (a backend)

A backend implements the engine's protocol (`measure.Backend`): `compile(config, key)`, `check(config, key,
artifact)`, `measure_batch(items, fidelity)`, `environment()`. `tune/live_toy.py` is the template; copy its rules:

- **Numerics before timing, in the same process.** `check` compares every output the stage / layer / round writes
  (mailboxes, states, outputs) with the reference (bitwise, or the T0 tolerance for `tolerance` configs), at two
  steps (one straddling the tag wrap). The FIRST check of a session plants a must-fail mutant (a corrupted weight word,
  a `mutate=` build, a dropped partial for N1) and stops the session if the gate does not see it.
- **Timing**: `measure_batch` times every item together with the baseline arm in `fidelity.reps` rounds, a seeded
  shuffled order per round, graphs replayed after a READ cache flush (`live_toy.build_read_flush`; alola rule 50), and
  re-checks every arm's state after timing. Unstamped builds only (stamps bias timing).
- **Decision** (`ToyChainBackend.confirm` is the template): baseline, A-vs-A control (second instance, own buffers),
  recompiled control (separately loaded code object), finalists; 12+ rounds; `gate.win_rule` (median below both
  controls' minima by 0.002 AND >= 3/4 rounds below 1); throttled rounds (SCLK < 0.95 of the session median, sampled
  by PCI BDF) dropped. A WIN is recorded only after a confirmation on a fresh seed / allocation (and the other rocke
  node for a served default).
- **Environment**: return the toolchain switches the engine cannot see (`ROCKE_CODEOBJ_BACKEND`,
  `ROCKE_DIRECT_LEVERS`, the ATOM tree hash for served runs) from `environment()`; the evaluator's cache is keyed on it.

Objectives by level (`tune/objective.py`): a backend minimizes a time; `Objective(level, cell, share)` turns its
ratio into the round ratio and the share of the gap to the headline target it closes (K=3 CONC 1: verify round
<= 0.83 of the hybrid's; K=0 CONC 1: per-token time <= 0.78). `rank_by_gap` orders a lever ledger across stages and
levels by that share. Shares come from DESIGN's gap table, per cell.

| level | backend | win rule |
|---|---|---|
| stage | stage rig (W2 probes, `stage_b3` isolated kernel, fused-MoE rig) | 9.1 |
| layer | in-situ layer A/B (16 cold weight copies, graph of 96 launches) | 9.1 |
| round | the persistent round kernel (M3 / M4 of PLAN) | 9.1 |
| e2e | ATOM-truemono serve harness; `Objective.time_of(tok_s, tokens_per_s=True)` | `gate.e2e_rule` (median below the A-vs-A minimum, >= 5 / 6 pairs below 1) |

## 4. Per-track notes

- **W2 (small batch, rows 1..16, seq / grouped / wide)**: spaces `fused_moe(rows, seq)`, `b3(variant, numa=)`,
  `framework()`, `stage_fn()`, `sched(kind, variant)`, `mx_gemv(n, k, rows_max)`; join them for a layer or round
  search with `CompositeKnobs([("fw", ...), ("moe", ...), ("b3", ...), ("sched", ...)])` (one key over all parts).
  Variants are the DB's `problem` (`r1` ... `r16`, `r4s`, `r4g2`, `delta.r1` ...); `catalog.variant_features` makes
  them comparable for transfer. Export per-variant BEST tables with `export-defaults` (knob JSON per variant;
  `sched_string` is a ready `ATOM_Q38_SCHED` value).
- **W4 (large batch, TP1 / TP2, CCL)**: `ccl(op)` (forms, slots, homing, sync, publish; `ar_dtype=bf16` is a tolerance
  knob); variants `b8`..`b128` x `tp1` / `tp2` (add them to `variant_features` as they appear). The e2e objective is
  throughput at equal per-request latency; every TP2 cell also times the DP2 arm (PLAN decision 5), as a fixed arm in
  the confirmation rounds.
- **Rust direct compiler**: `direct()` (`backend` comgr / direct, levers gated on it). The backend is a knob like any
  other: the search can pick COMGR for one stage and direct for another. Non-bitwise transforms are allowed under the
  per-op tolerance tests (decision 6): mark such levers `numerics=TOLERANCE`. Compile time enters through the
  evaluator's `compile_s` (cost-aware portfolios can use it).

## 5. Budgets and cluster rules

- **Strategy by space and budget** (replay evidence, section 7): exhaustive when the legal space fits the budget;
  the signed-lookback climber when the budget is about 16-32 evaluations on a large space or a climb ledger exists
  (it uses the ledger as its prior); TPE / GP-EI past about 64 evaluations; a portfolio of the lookback climber with
  TPE as the default when the budget is unknown. Never a brute-force grid.
- **Evaluations**: a stage lever sweep is about 16-48 evaluations per variant; a lookback round about 2-4x the number
  of retained levers. Every search reserves its confirmation (finalists + baseline + 2 controls, 12+ rounds).
- **GPUs**: at most 2 GPUs for this lane at a time; at most 6 of ours busy on the rocke partition; timed jobs log every
  neighbour (`squeue -h -w $(hostname -s)`), all users. Each GPU step under `timeout -s KILL` and after a randn-fill
  health check (alola rule 146). Long searches: one job per search with the driver inside (cheapest), or
  `JobEvaluator` + `Q38Transport` (one job per round of candidates; `Q38_SUBMIT`, `Q38_LOGIN`, `Q38_AUS` in the
  environment).
- **CPU**: compiles run on the job's CPUs (16 per GPU) with a job-local `AMD_COMGR_CACHE_DIR`; replays and IR builds fan
  out as alola CPU jobs (`cluster_job.sh --arch cpu`). The laptop runs imports and the CPU unit tests only.
- **Stages**: q38 stage names starting with `rocke` get `PYTHONPATH` from `q38_job.sh`; otherwise export it (alola
  rule 153).

## 6. The RSI loop

1. **Find the bubble** (DESIGN 9.2.1): stamps timeline, blame graph, Coz entry / post delays, poll-spin heat maps,
   WaveScope ATT before / after (`ROCKE_DEBUG_LOC=1`), PMC. The simulator bounds, it never ranks.
2. **Hypothesis -> knob**: the mechanism becomes a default-off field (section 2), knob off byte-identical, numerics
   gated (bitwise or the tolerance ladder with its mutant).
3. **Search**: on the current best body (the DB's served record for the variant, not an old rung), warm-started from
   the nearest variant (`tdb.Selector.top_k`, `TuningDB.warm_starts`) and the gain ledger (`tdb.GainLedger.prior`
   and `prior_pairs`: they order the climb and set the surrogate's prior mean, they never remove a lever).
   `SignedLookback` climbs one lever at a time (APPLY only beyond the noise) and records each level's solo gain on
   the start, re-sweeps after a structural change, then runs the **combination lookback** (PLAN decision 13: ten
   levers can each lose alone while a combination of them wins by half):
   - the pool is EVERY knob and EVERY level, solo losers included; nothing is pruned from solo results, only measured
     combination data moves the model;
   - a pairwise covering screen over the whole pool (greedy random combinations covering the most uncovered pairs of
     changed levels; at most a quarter of the budget);
   - an interaction-aware GP on log time over every measured config (kernel = constant + main + every pairwise + every
     three-way interaction term, scales by marginal likelihood; prior mean = solo gains + recorded pair terms);
   - expected improvement over samples of the full product space densified around the best measured combinations
     (every 1-, 2-, 3-change move and crossovers), best EI plus the most uncertain points per round, until no
     combination is expected to beat the incumbent by the gain floor; then the climber climbs again from the pick.
   The ledger keeps solo gains, measured combinations and the fitted pairwise interaction terms
   (`SignedLookback.interactions()`, `GainLedger.add_pairs`), so the next lookback starts from them. The live
   confirmation always includes the solo-winners-only combination (the start plus every level that won alone), so
   "does the combination beat the solo winners" is answered in the same rounds.
4. **Lookback after each structural change**: the base changed, so every lever is re-swept, losers included
   (automatic in `SignedLookback`; across sessions, the ledger carries solo gains and pair terms).
5. **Order-dependent combinations** (PLAN decision 15: an idea that fails in isolation can win once another change is
   in, so a verdict is never final):
   - **base-relative verdicts**: every climb verdict is recorded with the base it was measured on (its changes from the
     start) and its signed gain (`report()["verdicts"]`, `GainLedger.add_verdicts`). A rejection holds for that base
     only; nothing is ever permanently rejected or filtered.
   - **re-test after every accepted move**: a climb step, a lookback pick that moves the base, or an externally supplied
     base (`SignedLookback.set_base(genome, reason)`, e.g. a restructure or another track's pick) re-queues every level
     that was rejected on another base, ranked by the interaction model (the GP's prediction of each level on the new
     base). The ranking decides WHEN, never WHETHER:
     - front of the queue: the levels the model predicts beat the new base by more than the gain floor and, while
       levers of the current pass are still untested, by more than such a lever is expected to bring (the mean
       first-test gain so far). Only those levels are re-tested now, not the lever's whole range;
     - behind the untested levers: every other rejected level of the lever;
     - once half of a known budget is spent (lookback on), the non-promising re-tests become the first points of the
       lookback screen instead of extra climb arms;
     - a level not re-tested yet stays open: it is re-queued at the next accepted move and re-tested in the revisit pass
       at the latest.
     `report()["moves"]` lists the accepted moves in order, with the re-tests each one put in front, behind or into the
     screen.
   - **revisit pass**: when the lookback ends, an explicit "revisit rejects" pass re-tests every level rejected on any
     other base on the final base.
   - **climb-order diversity**: `order_restarts` further climbs from the start in other lever orders (reversed, then
     shuffled); known configs cost nothing (history is shared), so a restart only pays for the new paths.
   - `lookback=False` gives a pure climb with the same re-tests (cheap; the order-dependence tests use it).
6. **Decide and record**: confirmation with both controls and the 9.1 rule, a fresh-seed repeat, then
   `tdb.make_record` into the DB (significant records only are served), signed gains into the ledger, `export-defaults`
   into the BEST tables; WaveScope after the change to see what moved.
7. Rank the next work by `objective.rank_by_gap` (share of the gap to the headline targets closed) and go to 1.

## 7. Validation (ratios; absolute samples in the private store)

### 7.1 Replay of recorded A/B data (alola CPUONLY jobs 68223301, 68223930, 68224548; 20 seeds per cell)

This subsection measured the FIRST lookback, which pruned consistent solo losers and capped its grid. That lookback
violates PLAN decision 13 and was replaced; 7.4 has the replay and live results of the current combination lookback
(same problems, same seeds). The rows of TPE, GP-EI, the engine portfolio and random are unaffected.

Problems are recorded data with a known optimum, searched through the real engine (noise included):

- **B3 signed-lookback grids, NUMA base** (`climb_b3.py` records of r1 / r2s / r4 / r4s): 767-1151 legal cells over 9
  levers each; every cell timed once against the climb's best, noise from the recorded re-timings (0.1-0.34 %).
  Start: the shipped B3 on the NUMA layout (1.032-1.050 of the recorded optimum).
- **B3 grids, mono base** (6 older climbs): 95-111 cells; start: the grid member nearest the shipped build.
- **Schedule rounds** (sched-l1 / l2 / l3b / l5: holds, head sleep, NUMA on delta r1 / r4s and qsa r4s): 70-74 recorded
  configs, 12 shuffled rounds each with the mono-vs-mono control; one recorded round per interleaved batch (pairing
  kept).

"hit": the pick's recorded value within 0.2 % (the 9.1 margin) of the recorded optimum, i.e. inside the noise; "gain":
mean share of the start -> optimum gain reached by the pick.

| problems | budget | phased (default) | signed lookback | TPE | GP-EI | engine portfolio | random |
|---|---|---|---|---|---|---|---|
| B3 NUMA grids (767-1151 cells) | 16 | 0.26 hit / 0.921 gain | **0.29 / 0.921** | 0.17 / 0.891 | 0.15 / 0.876 | 0.15 / 0.835 | 0.21 / 0.853 |
| | 32 | 0.35 / 0.934 | **0.45 / 0.940** | 0.43 / 0.937 | 0.41 / 0.942 | 0.34 / 0.917 | 0.29 / 0.912 |
| | 64 | 0.56 / 0.958 | 0.54 / 0.957 | **0.61 / 0.960** | 0.54 / 0.958 | 0.49 / 0.949 | 0.34 / 0.925 |
| | 96 | **0.59 / 0.962** | 0.54 / 0.960 | **0.59** / 0.958 | 0.55 / 0.960 | 0.50 / 0.953 | 0.38 / 0.929 |
| B3 mono-base grids (95-111 cells) | 16 | **0.62 / 0.921** | **0.62 / 0.931** | 0.48 / 0.850 | 0.55 / 0.888 | 0.16 / 0.676 | 0.21 / 0.697 |
| | 32 | 0.84 / 0.972 | 0.72 / 0.948 | **0.87 / 0.978** | **0.87** / 0.975 | 0.78 / 0.957 | 0.38 / 0.813 |
| | 64 | **0.86 / 0.975** | 0.72 / 0.948 | **0.86 / 0.975** | **0.86 / 0.975** | **0.86 / 0.975** | 0.62 / 0.912 |
| Schedule rounds (70-74 configs) | 16 | **0.63 / 0.982** | 0.02 / 0.943 | **0.67** / 0.974 | - | 0.33 / 0.953 | 0.42 / 0.964 |
| | 24 | **0.90 / 0.989** | 0.88 / 0.985 | 0.88 / **0.990** | - | 0.67 / 0.982 | 0.53 / 0.976 |
| | 32 | **0.95** / 0.991 | **0.95 / 0.993** | 0.93 / 0.992 | - | 0.92 / 0.992 | 0.62 / 0.979 |

Findings:

- **The default is `phased` (signed lookback, then TPE on everything the climber measured).** It is the best or within
  0.01 of the best gain in every row, never below the engine portfolio or random. The climber alone wins small budgets
  on large grids but stops early on small ones (it finishes its lookback and has nothing left to ask); TPE alone is
  weak at 16. A Thompson portfolio that splits a small budget from the start was worse than either member at 16-32
  (also tested with 2-3 members: lookback+TPE, lookback+GP, lookback+TPE+GP; job 68223930).
- **Decision step on the recorded rounds** (the 9.1 rule against the recorded mono-vs-mono control, phased at 32
  evaluations, seed 0): delta r1 -> `numa=1, hold.b1=32, hold.m3=32, hold.m4=64` and delta r4s -> `numa=1,
  hold.m3=128, head_sleep=128`, exactly ATOM's shipped `sched.BEST` for both (WIN, 12 / 12 rounds below 1); qsa r4s ->
  `numa=1, hold.m1=32, hold.m3=64, hold.q1=64`, 1.001 of the recorded optimum (shipped: `hold.m3=64, head_sleep=128,
  hold.m1=32`; inside the noise).
- **Two climber fixes came out of the replay**: (a) a lever with no legal arm on the current base used to be skipped;
  it now takes the nearest legal *companion* move that keeps its level (a combination-only level such as
  `m4_interleave` + `m4_guard=none`; in a recorded table, the nearest recorded config), and skipped levers are re-opened
  when the base moves; (b) when the lookback moves the base, the climber climbs again (bounded). Delta r1 at 24
  evaluations went from 0.60 to 1.00 hit.
- **Transfer starts across B3 variants did not help** (warm start from the nearest variant's recorded climb best: equal
  or worse than the shipped start at 8-32 evaluations): the variants' optima differ in structural levers (split, xchg,
  pre). Transfer is still the right start for a re-tune of the same variant on a new base (the DB's own record) and
  for shapes of one family (7.2).
- Recorded optimum vs noise: on the NUMA grids the top cells sit within 0.1 % of each other (single noisy batches), so
  "hit" is strict there; the gain column is the more stable measure.

### 7.2 Live on gfx950: the framework's toy chain vs its recorded optimum (job 68225392, 1 GPU, n01-09)

Target: `toy_chain(depth=8, work=8)`, 22 knobs (about 2e8 configurations before legality), the layer framework's
lever testbed with an exact int32 host model. Recorded optimum: `toy_chain.best_spec(8, 8)` (found earlier by hill
climbing with lookback, T9 gate). Every session first ran the gate's must-fail mutants (a corrupted weight word and a
corrupted activation word of the last launch: both caught; the clean run passed). Every candidate was checked before
timing and re-checked after it; timing: 4 shuffled rounds per batch against the baseline arm, a read cache flush
before every graph replay. Decision: 12 shuffled rounds with the A-vs-A and recompiled controls, the 9.1 rule. Ratios
are times over the shipped toy (lower is better); "vs recorded" is the pick over the recorded optimum in the same
rounds.

| run | strategy, start | evaluations | pick vs shipped | recorded optimum vs shipped | pick vs recorded (rounds below 1) |
|---|---|---|---|---|---|
| A | phased, shipped start | 96 | 0.742 (WIN) | 0.782 | **0.950 (12 / 12)** |
| A2 | phased, shipped start, seed 1 | 96 | 0.725 (WIN) | 0.766 | **0.945 (12 / 12)** |
| D | random search (baseline) | 96 | 0.902 (WIN) | 0.783 | 1.149 (0 / 12) |
| B | signed lookback, transfer start from `best_spec(4, 8)` | 32 | 0.761 (WIN) | 0.767 | **0.994 (12 / 12)** |
| B2 | as B, seed 3, another GPU (job 68225584) | 32 | 0.757 (WIN) | 0.785 | **0.962 (12 / 12)** |
| C | phased through the batch runner (one subprocess per round), shipped start | 32 | 0.824 (WIN) | 0.757 | 1.089 (0 / 12) |

- From the shipped build, 96 evaluations found configurations faster than the recorded optimum in two seeds (about 5 %
  in the same rounds, every round). Random search at the same budget stayed 15 % slower
  than the recorded optimum. Warm-started from the nearest shape's recorded best, 32 evaluations matched or beat the recorded
  optimum in two seeds (0.994 and 0.962, every round below 1). Cold at 32 evaluations is not enough (C), as the
  replay predicts.
- The picks differ from `best_spec`: they keep the DPP sums, prefetch and early weight issue, add `step_first` and
  `poll_max=16` (one pick also `kernarg_preload=8`), and drop the NUMA dual-copy mailboxes. On this node in this run the
  NUMA lever did not pay for the toy at this shape. This is a candidate update of `best_spec` for its owner, to be
  confirmed on a fresh allocation and on the other rocke node (DESIGN 9.1) before anything changes.
- The DB recorded the WIN (`toy_chain.d8w8l1@6f3d4848bcf2b104`, significant), and `export-defaults` served it.
- The batch-runner path (C) ran 21 batches end to end, each in its own process with its own gate self-test.
- Clocks: SCLK / power come from sysfs of the job's own card (by PCI address); job 68225392 sampled on a timer only (one
  sample per sub-second confirmation, so no throttle filtering), job 68225584 samples at every round boundary (85
  samples, SCLK constant across rounds, no round dropped).
- Not exercised yet: `Q38Transport` (a cluster job per round, submitted from the login side), which differs from the
  runner path only in submission and polling.

### 7.3 What the live runs caught (the gate earned its place)

The first live sessions (job 68224520) planted one mutant (a corrupted weight word of the chain's first launch). At
depth 8 it was not caught. The failed self-test surfaced only as a `run_error` on the first candidate (the shipped
start, so the climber lost its start), and the rest of the session was checked by a gate that had not proven itself.
Those sessions' results were discarded. On hardware (jobs 68225091, 68225242), d8w8 outputs ignored corrupted
first-launch seeds and weights entirely, at fresh addresses and with fresh tags, while d4w8 caught both. The cause is
arithmetic, not memory. The toy's int32 recurrence carries a level's `r_prev` into the next level multiplied by
`P (P + 1) / 2` = 12880 = 2^4 x 805 (P = 160), so after 8 levels the factor is 0 mod 2^32. A fault confined to an
earlier launch of a chain vanishes from that launch's output, and the check compares only the last launch's mailboxes
plus every launch's output. Fixes in the tuner: the mutants sit in the last launch's inputs; the host model must first
show that a mutant is observable; a failed self-test refuses the session (nothing is timed); every check uses fresh
step values. `test_toy_chain_gate_blind_spot_and_last_launch_mutants` pins the arithmetic on the host. The toy's own
`check_state` has the same blind spot for earlier launches at depth >= 8 (INVENTORY merge item M10).

### 7.4 Combination lookback (PLAN decision 13): replay and live

The lookback now keeps every knob and every level, solo losers included, and explores combinations model-guided
(section 6, step 3; default screen: as many points as knobs, 3 levers moved per point, at most a quarter of the
budget). Same recorded problems and seeds as 7.1 (alola CPUONLY jobs 68226121 and 68226676; screen variants A/B'd in
68226486, 10 seeds).

**The solo picture of the recorded data.** From each start, every recorded single change classifies a level as a solo
winner (faster alone) or a solo loser. "Solo-winners-only optimum" (SWO) = the best recorded config built only from
solo winners.

- In 7 of the 10 B3 grids, the recorded optimum uses levels that are not solo winners. On the NUMA r1 grid it contains
  five solo losers (`hoist=gates_z`, `state_issue=entry`, `state_nt=True`, `wait=barrier`, `xchg=numa2`): each is
  slower alone from the shipped start, together they beat SWO by 0.5 %. The recorded optimum over SWO: 0.995 (NUMA r1),
  0.998 (r2s), 0.999 (r4), 0.997 (r1_c2), 0.994 (r2s_c5), 0.993 (mono-base r4s).
- On the schedule rounds the recorded optimum is built from solo winners only (holds and NUMA each win alone), so
  there is nothing for a combination of losers to find there.

**Replay** ("hit" / "gain" as in 7.1; "loser combo beats SWO": share of runs whose pick contains a non-winner level and
beats SWO by more than the 0.2 % margin):

| problems | budget | phased (default) | climb + combination lookback | TPE | random |
|---|---|---|---|---|---|
| B3 NUMA grids | 16 | 0.08 hit / 0.881 gain | 0.05 / 0.777 | **0.17 / 0.891** | 0.21 / 0.853 |
| | 32 | 0.40 / 0.935 | **0.55 / 0.954** | 0.43 / 0.937 | 0.29 / 0.912 |
| | 64 | 0.51 / 0.955 | 0.49 / 0.955 | **0.61 / 0.960** | 0.34 / 0.925 |
| | 96 | 0.55 / **0.960** | 0.54 / 0.959 | **0.59** / 0.958 | 0.38 / 0.929 |
| B3 mono-base grids | 16 | **0.65** / 0.925 | 0.64 / **0.938** | 0.48 / 0.850 | 0.21 / 0.697 |
| | 32 | **0.87** / 0.973 | **0.87 / 0.976** | **0.87 / 0.978** | 0.38 / 0.813 |
| | 64 | **0.86 / 0.975** | **0.86** / 0.974 | **0.86 / 0.975** | 0.62 / 0.912 |
| Schedule rounds | 16 | 0.63 / **0.982** | 0.02 / 0.943 | **0.67** / 0.974 | 0.42 / 0.964 |
| | 24 | **0.92 / 0.991** | 0.60 / 0.983 | 0.88 / 0.990 | 0.53 / 0.976 |
| | 32 | **0.95** / **0.992** | **0.95 / 0.992** | 0.93 / **0.992** | 0.62 / 0.979 |

Loser combinations that beat SWO, per grid where the recorded data has one (climb + combination lookback / phased,
at 64 evaluations, 20 seeds): NUMA r1 **1.00 / 0.95**, mono-base r4s 0.95 / 0.95, r2s_c5 0.85 / 0.85, r1_c2 0.60 /
0.60, NUMA r2s 0.30 / 0.30 (a 0.2 % margin, at the noise). Random at 64: 0.19 (NUMA) and 0.29 (mono-base) over all
grids. The decision step on the schedule rounds (phased, 48 evaluations) re-derived ATOM's shipped `sched.BEST`
exactly for all three variants (delta r1, delta r4s and now also qsa r4s; 9.1 WIN, 12 / 12 rounds).

Against the first (pruning) lookback of 7.1: better at 32 evaluations on the B3 grids (NUMA hit 0.55 vs 0.45,
mono-base 0.87 vs 0.72), equal at 64-96, but worse at the smallest budgets (NUMA at 16: 0.05 vs 0.29; schedule
rounds at 24: 0.60 vs 0.88): the covering screen over every lever, losers included, costs evaluations that the
pruning version spent on fewer levers. That is the price of not excluding losers; `phased` (climb, lookback, then TPE
on all of it) keeps the small-budget rows close to TPE's.

**Live on gfx950** (toy chain d8w8, one GPU on each rocke node; the n07-17 job ran on bus 149, not the faulty e5;
gate mutants caught and clean pass in every session; 12-round confirmation with both controls; "SWO" here = the
climb's start plus every level that won alone in the first climb pass, timed in the same rounds):

| node (job) | run | evaluations | pick vs recorded optimum | pick vs SWO | solo losers in the pick |
|---|---|---|---|---|---|
| n07-17 (68226312) | phased, shipped start | 96 | 0.939 (12 / 12 below 1) | **0.975 (12 / 12)** | `lds_touch=True`, `mailbox.fence_impl=ir`, `phase_barriers=False` |
| n01-09 (68226268) | phased, shipped start, other seed | 96 | 0.934 (12 / 12) | 1.009 (1 / 12) | `kernarg_preload=8`, `mailbox.fence_impl=ir`, `mailbox.pause=none` |
| n07-17 (68226312) | lookback, transfer start from `best_spec(4, 8)`, ledger priors | 32 | 0.983 (12 / 12) | 1.005 (2 / 12) | none |
| n01-09 (68226268) | same, other seed | 32 | 0.996 (11 / 12) | 1.008 (1 / 12) | none |

- On n07-17 the combination search found a configuration with three solo losers that beats the best solo-winners-only
  combination by 2.5 % and the recorded optimum by 6 %, every round. On n01-09 the cold pick also carries three solo
  losers and beats the recorded optimum by 7 %, but it ties with that run's solo-winners-only combination. With a
  32-evaluation warm start, the picks match the recorded optimum and tie with SWO.
- An earlier n01-09 run (68226087, before "solo" was defined as the first climb pass) beat its SWO by 17 %. There only
  `reduce=dpp` counted as a solo winner, because the other levers were tried on the dpp base; that run is not used for
  the loser claim.
- The ledger now carries solo gains, measured combinations and pairwise interaction terms; the transfer runs (B)
  started their GP from run A's pair terms (`--gains`).

### 7.5 Order-dependent combinations (PLAN decision 15): tests, replay, live

**Unit tests** (`library/tests/test_qwen38_tune.py`, synthetic backends; each mechanism is mutation-checked, i.e. the
test fails with the mechanism switched off):

- B loses alone (+3 %) and wins big once A is in. Lever order B, C, A, D, so B is tested and rejected first. The
  climb applies A, re-queues B, and ends with A-then-B (`applied == ["A=True", "B=True"]`). B's verdicts read "rejected
  on the start" and "applied on the A=True base".
- Negative control: with `retest=False`, no revisit and no restarts, the same climb gives B up for good.
- The revisit pass alone and an order restart alone each recover B. An external `set_base` re-queues the rejects. An
  order knob (`Knob.sequence`) climbs by insertion moves.
- The re-test acquisition:
  - an untested lever that is due is not starved by re-tests the model only weakly believes in;
  - a promising re-test tests only its promising levels;
  - with the lookback on, the other rejects go to the lookback's expected-improvement pool and are re-tested by the
    revisit pass.

**Replay** (same problems and seeds as 7.1 / 7.4; alola CPUONLY; "hit / gain" as in 7.1). Four acquisition designs
were replayed in turn:

| design | job | NUMA grids, lookback, 32 | mono-base, lookback, 16 | schedule, phased, 16 |
|---|---|---|---|---|
| decision 13 only (7.4) | 68226676 | 0.55 / 0.954 | 0.64 / 0.938 | 0.63 / 0.982 |
| every reject to the front of the queue | r8 | regressed at 16-32 | | |
| model-promising rejects in front, the others behind the untested levers | r9 / r10 | 0.35 / 0.933 | 0.55 / 0.926 | 0.60 / **0.746** |
| + the front gate (expected first-test gain) and level-granular re-tests | 68235209 | 0.35 / 0.933 | 0.54 / 0.923 | 0.68 / 0.984 |
| + with the lookback on, the other rejects go to its expected-improvement pool (final) | 68235329 | **0.56 / 0.955** | **0.66 / 0.940** | **0.68 / 0.984** |

Every other cell of the final design is within seed noise of decision 13 or better, for example:

- NUMA phased at 96: 0.57 / 0.963;
- loser combinations beating SWO at 64: NUMA r1 1.00 / 0.95, r4s 0.95, r2s_c5 0.85, r1_c2 0.60, NUMA r2s 0.30.

What the re-tests still cost:

- schedule rounds, climb + lookback at 24: hit 0.55 vs 0.60 (gain 0.979 vs 0.983);
- schedule rounds, phased at 32-48: hit 0.92 vs 0.95, at equal gain (0.992).

Why the earlier designs lost:

- **r10 schedule rows.** The interaction model has little data early. It "predicted" that every other level of an
  8-level lever would beat the new base, because a level's solo gain is counted again on a base that already has
  another level of that lever. The climb re-swept the whole lever after every accepted move. On the schedule rounds the
  16-evaluation budget then ran out before the NUMA lever, which carries most of the gain.
- **r10 / r11 NUMA rows.** Forcing the re-tests, as climb arms or as screen points, took evaluations from the
  lookback's covering screen, which the GP needed.

The final design therefore splits the re-tests by the model:

- the model ranks the re-tests and only the promising levels go first;
- the rest are left to the lookback's acquisition. Every single change around the best combinations is in its pool.
- The revisit pass re-tests whatever is still open.

Nothing is dropped. A level that is not re-tested yet stays in `open_rejects` and is re-queued at every accepted move.

Retained knobs, off by default (`SignedLookback` arguments; the losing variants of this round):

- `handoff_at=0.5`: rejects go to the lookback only after half of the budget is spent (the r11 rows).
- `retest_screen_share`: forced screen points for re-tests (NUMA grids at 32: 0.33 / 0.935).
- `solo_combo_screen=True`: the main-effects pick (the start plus every lever's best solo level) as the first
  screen point (68235527). It helps at 16 (NUMA lookback gain 0.873 vs 0.782; mono-base 0.69 / 0.947 vs 0.66 /
  0.940) and costs hits at 32-48 (NUMA lookback at 32: 0.47 vs 0.56; schedule lookback at 48: 0.92 vs 0.97). The
  likely cause is a lucky noisy measurement of that one point becoming the incumbent.

The decision step on the schedule rounds (phased, 48 evaluations) still re-derives ATOM's shipped `sched.BEST` for
delta r1 and delta r4s. For qsa r4s it picks `hold.m1=32, hold.m3=64, hold.q1=64, numa=1` instead of the shipped
`head_sleep=128, hold.m1=32, hold.m3=64, numa=1`. Both are 9.1 WINs (12 / 12 rounds) and the new pick is 0.09 % slower
in median, inside the 0.2 % margin.

**Live on gfx950** (toy chain d8w8, phased 96 cold (A) + lookback 32 transfer (B) as in 7.4; one GPU per job; gate
mutants caught and clean pass in every session; 12-round confirmation with both controls). Column meanings:

- "pick levels rejected in the run": levels of the final pick that a climb verdict had rejected on some base;
- "climb: rejected, then applied": levels the climb itself applied after rejecting them on an earlier base.

| job (node) | design | run | pick vs recorded optimum | pick vs SWO | pick levels rejected in the run | climb: rejected, then applied |
|---|---|---|---|---|---|---|
| 68234903 (n01-09) | every reject to the front | A | 0.969 (12 / 12) | 1.058 (0 / 12) | 5 | 4 |
| | | B | 0.976 (12 / 12) | 1.010 (0 / 12) | - | 0 |
| 68235236 (n01-09) | front gate, level-granular | A | 0.922 (12 / 12) | 1.003 (3 / 12) | 2 | 2 |
| | | B | 0.971 (12 / 12) | - | - | 0 |
| 68235369 (n01-09) | final | A | 0.930 (12 / 12) | **0.946 (12 / 12)** | **9** (7 solo losers) | 0 |
| | | B | 0.984 (12 / 12) | 1.006 (0 / 12) | - | 0 |
| 68235651 (n07-17, bus 21) | final, other seed | A | 0.920 (12 / 12) | **0.932 (12 / 12)** | **3** (3 solo losers) | 0 |
| | | B | 0.981 (11 / 12) | 1.013 (1 / 12) | - | 0 |

What the live runs show:

- **With the final design both cold runs beat the solo-winners-only combination in every round**, by 5.4 % and
  6.8 %. Their picks carry 9 and 3 levels that a climb verdict had rejected on some base. Before decision 15, one of
  two cold runs did this (7.4: 0.975 and 1.009).
- In the final design the re-combination happens in the lookback, not in the climb: the re-tests that the model does
  not put in front go to the expected-improvement pool. The front-loaded design (68234903) re-applied four rejected
  levels inside the climb, yet its pick lost to SWO by 5.8 %. Its climb spent the budget on re-tests the model had no
  reason for.
- The 32-evaluation transfer runs (B) match the recorded optimum and tie with their SWO, as in 7.4: at that budget a
  warm start leaves little to combine.
- A third 1-GPU job pinned to n07-17 (68235238) failed the randn health precheck in 10 s (the known faulty GPU, exit
  125) and did not run. The n07-17 run above got a healthy GPU.
