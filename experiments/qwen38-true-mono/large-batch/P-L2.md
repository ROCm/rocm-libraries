# P-L2: the sync model at large batch (grid barrier vs polled edges vs mailbox all-to-all), gfx950

W6, 2026-10-31 / 11-01. `../DESIGN-LB.md` 5 (sync model) and 12 (P-L2). A persistent rocKE kernel runs L = 6 large-batch
decoder layers of the full GDN layer chain (HC mixers -> in-proj -> recurrence -> out-proj -> HC mixers -> router -> MoE
-> combine) with stage bodies that stream the stage's per-rank bytes from HBM, and switches only the synchronisation
between stages: FlyTape's striped grid barrier before every phase, per-token-tile / per-head / per-expert polled flags,
the mailbox all-to-all, per-edge mixes of them, and no sync at all (the timing floor). Rows 16-128, TP1 and the TP2 rank
shape, FlyTape's 14 / 16-phase plan and DESIGN-LB's 10 / 11-stage plan. Ratios only here; absolute times are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L2.md` (raw JSON `p-l2-raw/`). Full tables:
`P-L2-tables.md`.

## Gate and verdict

- **Numerics gate: met** on both rocke nodes, before every timing, on the timed binaries.
  - Every sync arm on 4 cells x idle and skewed arrivals, 2 launches each (fresh state, then a launch over the first
    one's flags and payload). The whole payload buffer (every word of every unit of every layer) matches the host model,
    the error words are 0, the residency ticket passes on all 256 CTAs and the epoch advances. That is 128 checks (256
    launches) per final run, 104 per base run. The fix-round reruns (see "Fix round") add the recompiled control
    `grid_rc`: 136 checks per run, all pass.
  - After its timing, every timed binary is re-verified (corrected in the fix round). The error words of every timed
    launch are 0. Then one launch per arm and skew runs over the arm's own post-timing state, with the shared payload
    buffer zeroed first, and its whole payload is compared with the host model. The original g / h re-verify did not
    zero the buffer, so after the first arm it could not see a broken payload (Verification, defect 1); only its
    error-word part was meaningful. The fix-round reruns use the zeroed form, all pass, and two must-fail sentinel
    mutants timed as arms fail it in every cell and skew.
- **Negative controls: all 38 gating checks caught in both final runs (34 / 34 in both base runs); 42 / 42 in the
  fix-round reruns, which add the early and stale-tag twins of the recommended default's own `s_nop` build
  (`mwn:early`, `mwn:tag`).**
  - No sync at all. Early release: the grid barrier misses one arrival; the flag arms skip the last flag of the wait set.
  - Stale-read twins: polling for the previous launch's tag, on every flag arm.
  - Deadline controls: a producer later than the poll deadline gives `poll_timeout`; for the grid barrier, the wait bound
    gives `sync_timeout` plus abort on every CTA. All are launch-fatal.
  - Oversize grid (257 and 512 CTAs): `ticket_timeout`, launch-fatal.
  - The informational `noacq` coherence twin (pretouch + plain loads) and the random-skew early twins are caught in 17 of
    18 per run. The misses are the expected ones (alola rule 182: a missed `noacq` proves nothing).
- **Sync share per layer** (section 4). Against FlyTape's traced GDN layer (its 14 / 16 phases incl. their grid-sync
  waits):
  - The striped barrier's own latency costs 0.05-0.11 of the layer.
  - Each phase's arrival drain costs 0.04-0.06.
  - Arrival imbalance (the phase tails) costs 0.11-0.26.
  - Everything that is not stage work (grid minus `none`) is 0.13-0.21 of the layer.
  - In the chain itself, CTAs wait 0.30-0.43 of their time under the grid barrier.
- **Decision (DESIGN-LB 5, per edge):** the measured default is `sync=mix`.
  - The grid barrier stays on every edge where a unit reads at least 0.4 of the previous stage: the HC
    combine / down / reduce / up edges, the in-proj's K halves, the router -> experts all-to-all, and some combine edges.
  - The narrow edges use per-wave flags with lazy publish (`pub=wave`, `lazy`): the recurrence per head, the out-proj
    per head slice, the router per K slice, the MoE down per expert, and the per-token-tile combine.
  - This combination beat the grid barrier on 10-12 of the 13 real-byte cells and lost none, on both nodes, at idle and
    random arrivals: geomean 0.979-0.983 idle, 0.974-0.980 random, 0.927 with one late unit per stage.
  - It also wins the pure hand-off (null) cells, 0.96-0.99.
  - Its parts each lost or tied alone (decision 13): per-wave flags on every edge 1.015; lazy publish with drained flags
    1.000 (threshold 0.9) and 0.995 (0.4); the 0.4 threshold with counted flags 1.040.
- **What the sync mechanism cannot recover:** at idle arrivals the best sync model recovers only 0.007-0.034 of
  FlyTape's layer. The gap to the no-sync floor is mostly dependency imbalance (phase tails on all-to-all edges), not the
  barrier mechanism.
  - Polled partial edges pay off when arrivals are skewed: with one late unit per stage, every flag arm recovers 0.04-0.12
    of the layer (0.13-0.21 in the null cells).
  - The mailbox all-to-all loses in 10-11 of the 13 real-byte cells at idle and ties the rest (geomean 1.016); with
    per-wave or counted publish it loses everywhere.

## 1. What was built

| piece | where |
|---|---|
| stage graph of one LB layer (FlyTape's phase list and DESIGN-LB's list; per-stage bytes from `bytes_model`, unit plans, read sets per unit, routing), device tables, host payload model, mix gmasks | `dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/qwen38_mono/large_batch/sync_plan.py` |
| the chain kernel (one persistent launch, all sync arms, publish variants, mutants, stamps) | `.../large_batch/sync_chain.py` |
| CPU tests (plan well-formedness, task tables, closed-form payload model vs a direct simulation, every variant lowers with its markers) | `.../library/tests/test_qwen38_lb_sync_chain.py` (29 pass) |
| GPU bench (numerics, controls, timing, stamps), reports, tables, job body, staging, local ISA check, MI350X triage wrapper | `hackathon-prep/qwen38/truemono-lb/p-l2/` (`pl2_bench.py`, `pl2_report.py`, `pl2_tables.py`, `job.sh`, `sync_pl2.sh`, `isa_local.py`, `mh.sh`) |

All code is new: Python engine only, `q38lbs_*` kernel names, imported by no served build. It imports W5's
`persist/sync.py` (striped barrier, ticket, error word) and W3's `ccl/wire.py` (bounded flag poll) and edits neither.
No existing file changed, so every existing build stays byte-identical.

**Kernel (one launch, 256 CTAs x 512 threads, one CTA per CU).** A CTA runs its task list in (layer, stage) order. Unit
`k` of a stage runs on CTA `k mod 256`, FlyTape's worker order. A CTA without a unit in a stage gets a dummy task, so the
barrier sees every CTA. A task does the following:
1. Wait by the arm, with the unit's first 8 weight loads issued before the wait. That is FlyTape's prefetch before the
   sync, and the narrow mono's weights before the poll.
2. Read its read set: 64 words of every producer, `sc1` loads (P5 `bulk=sc1`), all 512 lanes, 8 loads in flight per
   lane.
3. Stream its bytes. 8 KiB blocks of the layer's own weight region are read with 8 loads in flight a lane. `rw` units
   (the GDN state) write each block back.
4. Store a 2 KiB payload that depends on the checksum of everything it read, then publish.

Arms (compile-time `sync`, `pub`; host tables choose the per-edge plan):
- `grid`: P4's striped counter barrier.
  - 8 words per XCD, one agent-scope arrival per CTA after every wave's stores drain.
  - Wave 0 polls the 64 words lane-parallel.
  - The counters are cumulative, with abort-on-timeout and the launch-fatal word.
- `tile`: the flags of the read set, polled by all 512 lanes with `sc1` loads.
  - The poll has a 2 s `s_memrealtime` deadline and raises `poll_timeout` on expiry.
- `a2a`: the flags of every unit of the previous stage.
- `mix`: per stage, the barrier or the `tile` poll (task flag bits from `sync_plan.task_table(gmask)`).
- `none`: nothing.
- Publish variants:
  - `drain` (`unit_end`): every wave `vmcnt(0)`, CTA barrier, then one flag.
  - `wave` (`comm_wave`): every wave drains its own stores and stores its flag, with no barrier; the consumer polls 8 flags.
  - `counted`: the next task's first loads are issued before the publish, then `vmcnt(8)`, barrier, flag; the loads are
    carried into the next task.
  - `lazy` (mix only): only producers whose next stage polls publish a flag.
- Poll pause: `s_nop` or `s_sleep 1`.

Tags are the launch epoch (graph-replay safe): the last CTA to finish advances the epoch and re-zeroes the ticket. The
residency ticket is P4's per-XCD form and is launch-fatal on timeout. P4 / P5 rules carried over: 8 stripes, the
per-XCD ticket, abort-on-timeout for cumulative counters, `sc1` write-through payload and `sc1` loads,
deadline-bounded polls.

**Plans** (host data, one binary for all):
- `ft` is FlyTape's tiled tape.
  - TP1 has 16 phases: `attn_mix_combine / down / reduce / up`, `gdn_proj`, `gdn_recur`, `gdn_out`, `gdn_out_reduce`, the
    same four `mlp_mix_*` phases, `moe_router`, `moe_up`, `moe_down` and `moe_combine`.
  - TP2 has 14 phases: no out reduce, and a fused `moe_expert`.
  - Unit counts follow FlyTape's plans:
    - the mix down is 6 row groups x 40 splits (20 x 2 token blocks at B128);
    - the out-proj is 40 x 6 head splits;
    - the in-proj is 64-row groups x 2 splits plus the ragged one-chunk units;
    - the router is 9 x 20;
    - the reduces are element-strided folds;
    - experts are the active experts (uniform random top-10) plus the shared expert, split into parts until the grid fills.
  - Bytes per stage are the `matched` regime of `bytes_model`, so they compare with FlyTape.
- `lb` is DESIGN-LB's list: the combine is merged into the down projection, and the reduces into their producers as
  last-arriver folds. That gives 11 stages at TP1 and 10 at TP2.
- Read sets are the data dependencies (identical in every arm):
  - per token tile for the combine / down / router edges;
  - per head for proj -> recurrence -> out;
  - per expert for up -> down -> combine;
  - full for reduce -> up and route -> experts.
- The `0` cells keep the graph and stream no bytes (pure hand-off).
- The TP2 cells are one rank's shape without the exchange (WP-C5 measures the exchange).

## 2. Method

- Hardware: MI355X, `rocke` partition, q38 venv (torch 2.12 + ROCm 7.2 COMGR, llvm22), 1 GPU per job, every GPU
  health-checked first (never n07-17 bus e5). Final runs:
  - g = job 68235854 (n01-09, bus 65);
  - h = job 68235853 (n07-17, bus 05);
  - base runs d = 68235788 (n01-09, bus 15) and e = 68235789 (n07-17, bus 15);
  - the combination screen f = 68235804 (n01-09, bus 65);
  - fix-round reruns of g / h with the fixed bench: fg = 68236089 (n01-09, bus e5, healthy) and fh = 68236088
    (n07-17, bus 05); see "Fix round".
  - Job 68235749 (n07-17, bus 15) is a pre-run on the previous reduce plan; it is not used.
  - Neighbours are logged per job. On n01-09, 4-6 of our 1-GPU jobs and one foreign 2-GPU job ran alongside g / d / f.
    On n07-17, at most one other job of ours ran alongside h / e. Verdicts agree across the nodes to about 0.01.
- Triage only (never a verdict): MI350X, Markham, jobs 68235495, 68235524, 68235608, 68235652, 68235675, 68235713 and
  68235745.
- Timing (DESIGN 9.1):
  - Arms x skews are HIP graphs of 2 launches (12 layers per graph), 20 rounds, in a fresh shuffled order every round.
  - The per-layer time is the graph time / 12.
  - Ratios are taken against `grid` of the same skew per round.
  - Controls: `grid#2` (second graph of the same binary) and `grid_rc` (separately compiled copy). Both stayed within
    0.99-1.01 (geomean 0.998-1.003, extremes 0.989-1.014).
  - Win rule: the median below both controls' minimum by 0.002 and at least 3/4 of the rounds below 1; LOSS mirrored.
  - Timed builds are unstamped. Stamps come from separate stamped binaries, never timed.
- Skews (per unit, `s_sleep 1` units before its publish):
  - `idle`: none;
  - `uni`: U[0, 64] for every unit (`s_sleep 1` = 64 shader clocks each);
  - `late1`: one random unit per stage late by 400 units.

## 3. Results (final runs g | h; per-cell tables for every arm in `P-L2-tables.md`)

### 3.1 Per arm over the 13 real-byte cells (geomean of the per-cell median ratio vs `grid` [min, max]; WIN / TIE / LOSS)

| arm | idle (n01 / n07) | uni | late1 |
|---|---|---|---|
| `tile` (read-set flags, drain) | 0.998 / 1.002 [0.974, 1.036] 5/3/5, 3/6/4 | 0.992 / 0.994, no LOSS | 0.909 / 0.905, 13 WIN |
| `tile_w` (per-wave flags) | 1.015 / 1.016, 6-7 LOSS | 0.993 / 0.995 | 0.920 / 0.918 |
| `a2a` (mailbox all-to-all) | 1.016 / 1.016 [1.006, 1.033], 10-11 LOSS | 1.008 / 1.009 | 0.980 / 0.981 |
| `mix90d` (grid on full edges, drain flags elsewhere) | 0.991 / 0.996 [0.969, 1.026] 8/1/4, 5/6/2 | 0.991 / 0.990 | 0.911 / 0.908 |
| **`m.w.1.n.40`** (grid on >= 0.4 edges, per-wave lazy flags) | **0.979 / 0.983 [0.964, 1.000] 10/3/0, 11/2/0** | **0.974 / 0.976, 12/1/0 both** | 0.927 / 0.927 |
| `m.w.1.s.40` (same, `s_sleep 1` poll pause) | 0.979 / 0.982 [0.961, 0.999] 10/3/0 | 0.977 / 0.980 | 0.928 / 0.927 |
| `m.w.1.s.90` (grid on full edges only) | 0.990 / 0.992 | 0.979 / 0.981 | 0.912 / 0.912 |
| `none` (no sync: floor) | 0.831 / 0.836 [0.780, 0.939] | 0.805 / 0.808 | 0.609 / 0.611 |

Base runs d / e (other buses; kernel source one revision before the final one, which only added the mix + per-wave-flag
combination, so these arms compile from unchanged code paths), the publish and threshold variants:
- `tile_c` (counted publish): 1.018 / 1.019 at idle, 10-11 LOSS.
- `tile_p` (pause): 1.004 / 1.005, a tie.
- `a2a_w`: 1.102, LOSS everywhere. `a2a_c`: 1.043-1.045, LOSS everywhere.
- `mix90` / `mix40` (counted): 1.040-1.041, LOSS.
- `mix90l` (drain + lazy): 1.000.
- `mix40l`: 0.995-0.996.

The combination screen f (16 mix arms over publish {drain, wave} x lazy x pause x threshold {0.4, 0.9}, 7 cells) put the
`wave` + `lazy` + 0.4 arms first: 0.983-0.988 idle, never above 1.000. g / h confirm two of them on both nodes.

### 3.2 By cell (idle, n01 | n07; `W` / `L` = WIN / LOSS)

| cell | tile | a2a | mix90d | m.w.1.n.40 | none |
|---|---|---|---|---|---|
| tp1:16:ft | 0.995 \| 0.993 | 1.015 L \| 1.011 L | 0.985 W \| 0.984 W | 0.970 W \| 0.975 W | 0.816 \| 0.816 |
| tp1:32:ft | 1.001 \| 0.985 W | 1.022 L \| 1.018 L | 0.988 W \| 0.981 W | 0.978 W \| 0.967 W | 0.809 \| 0.803 |
| tp1:64:ft | 0.986 W \| 1.001 | 1.006 \| 1.011 L | 0.977 W \| 0.992 | 0.969 W \| 0.980 W | 0.852 \| 0.863 |
| tp1:128:ft | 0.974 W \| 0.997 | 1.015 L \| 1.010 | 0.969 W \| 0.981 W | 0.964 W \| 0.985 W | 0.881 \| 0.897 |
| tp2:16:ft | 1.031 L \| 1.036 L | 1.026 L \| 1.033 L | 1.017 L \| 1.026 L | 0.980 W \| 0.989 W | 0.806 \| 0.811 |
| tp2:32:ft | 1.012 L \| 1.012 L | 1.021 L \| 1.022 L | 1.007 L \| 1.004 | 0.990 \| 0.988 W | 0.780 \| 0.780 |
| tp2:64:ft | 1.023 L \| 1.013 L | 1.026 L \| 1.027 L | 1.006 L \| 1.009 | 0.994 \| 0.989 | 0.793 \| 0.789 |
| tp2:128:ft | 1.008 L \| 1.005 | 1.016 L \| 1.017 L | 0.993 W \| 1.000 | 1.000 \| 1.000 | 0.805 \| 0.808 |
| tp1:16:lb | 0.985 W \| 0.991 W | 1.012 L \| 1.014 L | 0.983 W \| 0.988 W | 0.968 W \| 0.971 W | 0.831 \| 0.832 |
| tp1:64:lb | 0.975 W \| 0.997 | 1.007 L \| 1.006 | 0.972 W \| 0.994 | 0.980 W \| 0.990 W | 0.891 \| 0.908 |
| tp1:128:lb | 0.985 W \| 0.981 W | 1.008 \| 1.011 L | 0.983 W \| 0.982 W | 0.983 W \| 0.984 W | 0.924 \| 0.939 |
| tp2:16:lb | 1.006 L \| 1.013 L | 1.017 L \| 1.020 L | 1.010 L \| 1.012 L | 0.964 W \| 0.976 W | 0.791 \| 0.794 |
| tp2:128:lb | 1.001 \| 1.000 | 1.010 \| 1.009 L | 1.000 \| 0.994 | 0.993 W \| 0.986 W | 0.839 \| 0.841 |
| tp1:16:ft0 (null) | 1.075 L \| 1.069 L | 1.039 L \| 1.038 L | 1.040 L \| 1.032 L | 0.970 W \| 0.963 W | 0.793 \| 0.790 |
| tp1:128:ft0 (null) | 1.110 L \| 1.099 L | 1.093 L \| 1.086 L | 1.040 L \| 1.028 L | 0.994 W \| 0.987 W | 0.810 \| 0.803 |
| tp2:16:ft0 (null) | 1.077 L \| 1.076 L | 1.039 L \| 1.037 L | 1.061 L \| 1.059 L | 0.972 W \| 0.968 W | 0.786 \| 0.779 |
| tp2:128:ft0 (null) | 1.093 L \| 1.102 L | 1.064 L \| 1.075 L | 1.024 L \| 1.044 L | 0.980 \| 0.984 W | 0.774 \| 0.787 |

- **TP1 vs TP2.** Read-set flags help at TP1, 0.97-1.00. At the TP2 rank shape they lose or tie, 1.00-1.04: half the
  heads and per-rank expert units leave fewer units per stage to overlap, while each unit's poll + publish costs the same.
- **The pure hand-off (null cells).** One flag poll + drain publish per unit costs more than one striped barrier per
  stage (1.07-1.11). Per-wave lazy publish on narrow edges only is the one flag form that beats the barrier there.
- **One late unit per stage (`late1`).** Every flag arm wins big: `tile` 0.88-0.95 with real bytes, 0.79-0.87 null;
  the combination 0.88-0.96. The barrier makes every CTA wait for the late unit; flags make only its consumers wait.

## 4. Sync share per layer, and FlyTape's phases

### 4.1 Sync share of the chain's own layer, 1 - T(none) / T(arm), idle (g | h)

| cell | grid | tile | mix90d | m.w.1.n.40 |
|---|---|---|---|---|
| tp1:16:ft | 0.185 \| 0.184 | 0.181 \| 0.178 | 0.172 \| 0.171 | 0.160 \| 0.163 |
| tp1:32:ft | 0.191 \| 0.197 | 0.191 \| 0.185 | 0.181 \| 0.181 | 0.173 \| 0.169 |
| tp1:64:ft | 0.148 \| 0.137 | 0.136 \| 0.138 | 0.128 \| 0.130 | 0.120 \| 0.120 |
| tp1:128:ft | 0.119 \| 0.103 | 0.095 \| 0.101 | 0.090 \| 0.086 | 0.086 \| 0.090 |
| tp2:16:ft | 0.194 \| 0.190 | 0.218 \| 0.218 | 0.208 \| 0.209 | 0.177 \| 0.180 |
| tp2:32:ft | 0.220 \| 0.220 | 0.230 \| 0.230 | 0.226 \| 0.223 | 0.213 \| 0.211 |
| tp2:64:ft | 0.207 \| 0.211 | 0.225 \| 0.221 | 0.212 \| 0.218 | 0.203 \| 0.202 |
| tp2:128:ft | 0.196 \| 0.193 | 0.202 \| 0.197 | 0.190 \| 0.192 | 0.195 \| 0.193 |
| tp1:64:lb | 0.108 \| 0.092 | 0.086 \| 0.089 | 0.082 \| 0.086 | 0.091 \| 0.082 |
| tp1:128:lb | 0.077 \| 0.061 | 0.064 \| 0.043 | 0.062 \| 0.044 | 0.061 \| 0.045 |
| tp2:128:lb | 0.161 \| 0.159 | 0.162 \| 0.158 | 0.160 \| 0.154 | 0.155 \| 0.147 |

`none` is a floor: no stage waits, so stages and layers overlap freely and every CTA runs its own task sum. The sync
share is everything a correct sync model adds on top of that floor: barrier or poll latency, publish cost, and the
dependency waits on the slowest producer.

- FlyTape's phase list with real bytes: 0.10-0.22 of the layer.
- DESIGN-LB's shorter list at TP1 B64-B128: 0.04-0.11, because three fewer boundaries come from merging the reduces and
  combines.
- The TP2 rank shape: about 0.2. Its stages are half as long, but the number of boundaries is the same.

### 4.2 Against FlyTape's traced GDN layer (`ft` plan, grid arm, idle; stamped binaries; shares of FlyTape's layer)

FlyTape's traces: the TP1 B16-B64 and TP2 B16-B128 phase tables of P-L4's fix-round job 68234721. Two traced runs per
config bracketed the LB run on n07-17. The GDN layer = the 35 `gdn.*` layers' phases / 35. That is a cross-job,
cross-node comparison; read it as model calibration, not as a race.

| cell | chain grid layer / FT layer (n01 \| n07) | barrier latency | arrival drain | arrival tail (imbalance) | grid - none | recovered by the best sync arm |
|---|---|---|---|---|---|---|
| tp1:32:ft | 0.975 \| 0.997 | 0.073 \| 0.076 | 0.056 \| 0.055 | 0.174 \| 0.186 | 0.191 \| 0.201 | 0.022 \| 0.034 |
| tp1:64:ft | 0.958 \| 0.985 | 0.053 \| 0.057 | 0.057 \| 0.056 | 0.256 \| 0.255 | 0.145 \| 0.135 | 0.033 \| 0.020 |
| tp2:32:ft | 0.912 \| 0.932 | 0.099 \| 0.103 | 0.059 \| 0.061 | 0.170 \| 0.177 | 0.209 \| 0.210 | 0.010 \| 0.011 |
| tp2:128:ft | 0.818 \| 0.844 | 0.057 \| 0.059 | 0.043 \| 0.047 | 0.112 \| 0.116 | 0.163 \| 0.163 | 0.007 \| 0.009 |

Definitions, per phase boundary, summed over the layer's 14 / 16 boundaries:
- barrier latency = last release - last arrival;
- arrival drain = the median CTA's own `vmcnt(0)` + barrier before it arrives;
- arrival tail = last arrival - median arrival.

The three terms overlap with each other and with work, so they do not add up to `grid - none`.

- **The chain reproduces FlyTape's layer within 0.82-1.00** with FlyTape's unit plan and its bytes. So a barrier
  structure like FlyTape's spends about 0.05-0.10 of the layer in the barrier mechanism itself and 0.11-0.26 in waiting
  for the last arrival of each phase. The chain's barrier is P4's 8-stripe counter with FlyTape's lane-parallel ballot
  poll (P4 measured it against FlyTape's design); FlyTape's own trace cannot separate its sync from its work, so these
  are estimates for its structure, not a reading of its trace.
- **Per phase** (`P-L2-tables.md` 4): the barrier latency is 0.02-0.05 of the long phases (`gdn_recur`, `moe_up` /
  `moe_down` / `moe_expert`), 0.03-0.12 of `gdn_proj`, and 0.4-0.57 of the short ones (the mix reduces, the TP1
  combines, `gdn_out_reduce`).
  - The short phases are where FlyTape's structure pays: 5-6 of its 14-16 phases per layer are reduce / combine phases
    of a few microseconds.
- **Model caveat:** the chain's short fold / combine phases take up to 2.5x FlyTape's time (the HC reduces 1.8-2.5x, the
  TP1 combines and `gdn_out_reduce` 1.8-2.4x; the TP2 combines 0.4-1.5x). A chain unit makes two dependent memory round
  trips: it reads its inputs, then streams. Its GEMM-like phases (proj, out, router) take 0.35-0.86x, because the stream
  body has no math. Layer totals stay within 0.82-1.00.

## 5. What it means for DESIGN-LB (WP-L1 / WP-L3)

1. **Default `sync=mix`, per edge.**
   - Grid barrier on edges where a unit reads at least 0.4 of the previous stage (`gmask_full(plan, 0.4)`).
   - Per-wave flags with lazy publish on the narrow edges.
   - Poll pause `s_nop` (`s_sleep 1` is equivalent).
   - WP-L3 re-decides it in situ with the same arms on the real stage bodies (DESIGN-LB 5 procedure). Every other arm
     stays a default-off knob: `tile`, `a2a`, the `drain` / `counted` publishes, the 0.9 threshold, and the pause.
2. **Do not expect the sync mechanism to close the large-batch gap at idle arrivals.** The best arm recovers about
   0.01-0.03 of a FlyTape layer. The rest of the 0.13-0.21 non-work share is phase-tail imbalance on all-to-all edges.
   That is the planner's job (balanced unit plans, fewer phases), and the `lb` plan's fewer boundaries already show it
   (0.04-0.11 at TP1 B64-B128).
3. **Partial edges are insurance against skew.** Real stage bodies have uneven units: expert token counts, recurrence
   rows, the TP exchange. One late unit per stage costs the barrier 0.04-0.12 of the layer more than the flag arms.
4. **The mailbox all-to-all (P4's winning microbench form) and the counted publish do not transfer to large batch.**
   Both lose at every cell. P4's pairs-beat-counters result came from a single all-to-all hop with no work.
5. **The CCL exchange (TP2) is not in this chain.** WP-C5's in-situ exchange is pushed and polled with the tile flags.
   It adds to the flag arms' publish cost per exchanged unit, not to the barrier's.

## 6. Caveats

- Synthetic stage bodies: byte streams (and a write-back for the state), not the P-L1 / P-L3 / P-L4 kernels. The sync
  costs, edges and unit plans are real; the per-unit latency structure of the real bodies is not (section 4.2). WP-L3
  repeats the decision in situ.
- One routing draw per plan (uniform random top-10, the same for every layer of a launch). Correlated routing
  concentrates the MoE edges; it is not measured here.
- TP2 cells are one rank's shape without the exchange.
- Neighbour load on n01-09 (4-6 of our 1-GPU jobs, one foreign 2-GPU job) during g / d / f. The paired, shuffled ratios
  agree with n07-17 (1 neighbour) to about 0.01 in every cell.
- The FlyTape comparison is cross-job (its traces from job 68234721 on n07-17, 4 runs bracketing another LB run).

## 7. Rerun

```bash
H=/c/Users/vanantha/Documents/dev/hackathon-prep/qwen38/truemono-lb/p-l2
# CPU tests (local)
cd .../rocke/library && ROCKE_LLVM_FLAVOR=llvm22 ROCKE_BACKEND=python python -m pytest -q tests/test_qwen38_lb_sync_chain.py
# local compile + ISA check (venv COMGR, no GPU)
ROCKE_COMGR_LIB=.../.venv-ckdsl/Lib/site-packages/_rocm_sdk_core/bin/amd_comgr0702.dll ROCKE_LLVM_FLAVOR=llvm22 \
  python $H/isa_local.py --out <dir> sync=mix,pub=wave,lazy=1
# stage + run (MI355X, q38 venv; '+' = space in PL2_ARGS)
bash $H/sync_pl2.sh both 3
Q38_NODE=smci355-ccs-aus-n07-17 bash $H/../submit.sh log --tag w6-pl2-x --mode venv --gpus 1 --cpus 32 --mem 192 \
  --time 01:30:00 --stage w6-pl2-rocke3 --stage w6-pl2-h3 --env PL2_RKS=w6-pl2-rocke3 --env PL2_HS=w6-pl2-h3 \
  --env PL2_ARGS=--seed+6262+--arms+grid,grid#2,grid_rc,tile,tile_w,a2a,mix90d,m.w.1.s.40,m.w.1.n.40,m.w.1.s.90,none,Xtile_early,Xmwn_early \
  -- 'bash $Q38_STAGES/w6-pl2-h3/job.sh'
# (Xtile_early / Xmwn_early: must-fail sentinels timed as arms; their pay-zeroed post-timing re-verify must FAIL.
#  The bench exits rc 4 if a timed arm fails its re-verify or a sentinel passes it.)
# fix-round gate counts + section 3.1 summary (ratios)
python $H/fix_summary.py fg/pl2.json fh/pl2.json
# reports (ratios / absolute) and tables
python $H/pl2_report.py --runs g/pl2.json,h/pl2.json --ft <FlyTape traced jsonl,...> --ratio r.md --abs a.md
python $H/pl2_tables.py --runs g/pl2.json,h/pl2.json --out t.md
```

One job of numerics + controls + 17 cells x 11-16 arms x 3 skews x 20 rounds + stamps takes about 3-5 min of one GPU.

## Verification (independent verifier, W6, 2026-11-01)

Verdict: **partial**. The gate (numerics + gating controls) really ran on MI355X on the timed binaries, every gating
mutant fails, and the headline (`sync=mix`, 0.4 threshold, per-wave lazy flags beats the striped barrier; `a2a` and
`tile` do not) reproduces on a new routing draw. One claimed check is refuted: the post-timing re-verification cannot
detect a broken arm (except the first one it checks), so "every timed binary is re-verified after its timing (payload
...)" is not true as a payload check. The verdicts do not depend on it, because the pre-timing numerics ran on the same
binaries.

**What was read.**
- Job logs of g / h (68235854 n01-09 bus 65, 68235853 n07-17 bus 05), d / e / f. The sha256 of `sync_chain.py`,
  `sync_plan.py`, `persist/sync.py`, `ccl/wire.py`, `pl2_bench.py` and `job.sh` printed in g / h equal the files on
  disk now. Both GPUs were health-checked; h is not bus e5.
- From the raw `pl2.json` of g / h: numerics 128 / 128 PASS, gating controls 38 / 38 caught, rc 0. The section 3.1
  geomeans, ranges and WIN / TIE / LOSS counts recompute exactly: `m.w.1.n.40` idle 0.979 / 0.983, 10/3/0 and 11/2/0;
  uni 0.974 / 0.976, 12/1/0; `a2a` idle 1.016, 10 / 11 LOSS; `tile_w` idle 1.015 / 1.016.
- Controls are genuine: early mutants use `lastlate` (the skipped producer is late), tag twins poll for tag - 1 (on a
  fresh launch that passes at once over zero flags, so a stale read is what is caught; the later `poll_timeout` is a
  side effect), deadline controls give the named fatal reason, oversize grids give `ticket_timeout`. The payload is
  tag-dependent (`H(tag, ...)`), so a stale read of the previous launch is a wrong word.
- Timing discipline holds: arms x skews are separate HIP graphs in a fresh shuffled order per round, `grid#2` (same
  binary) and `grid_rc` (separately compiled, `_rc` name) as A-vs-A controls, the win rule uses the extreme control
  samples, timed builds are unstamped (stamps are separate binaries). The grid arm publishes no flags, so the flag arms
  are not favoured by extra stores in the reference.
- Fair comparison: every arm runs the same plan, unit tables, read sets, bytes, rows and GPU. PLAN decision 5 (2-GPU
  reference) does not apply: P-L2 compares sync arms against each other, its TP2 cells are one rank's shape on one GPU
  without the exchange (stated), and section 4.2 is a cross-job calibration against FlyTape's trace, not a race.
- CPU tests: 29 pass locally.

**Defects found.**
1. **The post-timing re-verification is vacuous after the first arm.** `phase_time` zeroes each arm's `ctl` but runs
   every graph the same number of times, so every arm reaches the same epoch, and the re-verify launch
   (`run_checked(..., reset=False)`) runs every arm at the same tag over a shared `pay` buffer already holding the
   previous arm's correct payload at that tag. A unit that reads too early, or never writes, sees or leaves correct
   words. Shown on hardware (below): two must-fail mutants timed as arms pass the builder's re-verify and fail once
   `pay` is zeroed before the launch. Fix: zero `pay` (or give each arm's verify a distinct tag) before each re-verify
   launch. Only the error-word check (`err_timing`) of the timed launches is meaningful as is.
2. `grid_rc` (the recompiled control, a separate binary) was never in the numerics phase, so it had only the vacuous
   re-verify. Now checked: PASS.
3. The default's own pause variant had no mutant: `mw:early` / `mw:tag` are built with `s_sleep 1`, the recommended
   default `m.w.1.n.40` uses `s_nop`. Now checked: both twins of the `s_nop` build are caught.

**Re-run (job 68235992, n07-17 bus 05, health-checked, 1 foreign 1-GPU job on the node; stages `w6-vP-L2-rocke`
(same sha256 as g / h) and `w6-vP-L2-h` = the builder's `pl2_bench.py` unchanged + `vpl2.py`; new seed 777001, i.e.
a new routing draw; harness `hackathon-prep/qwen38/truemono-lb/vp-l2/`, raw JSON
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/vp-l2-raw/`).**
- A. Numerics on two cells the builder never checked (tp1:64:ft, tp2:16:ft) x idle / uni, 2 launches each, fresh
  state: `grid`, `grid_rc`, `m.w.1.n.40`, `m.w.1.s.40` 16 / 16 PASS (every payload word, error words 0, ticket 256).
  Must-fail: `none` caught; `mwn:early` (`s_nop` build, `lastlate`) caught on both cells; `mwn:tag` caught on both
  cells (bad payload words plus `poll_timeout`).
- B. Re-verify demo (tp1:32:ft, `lastlate`, 4 rounds): `Xtile_early` and `Xmwn_early` (the early mutants timed as
  arms) report **PASS** in the builder's post-timing re-verify (bad units 0) and **FAIL** in the `pay`-zeroed
  re-verify (19807 and 13091 bad units). `grid`, `grid_rc`, `m.w.1.n.40` PASS in both.
- C. Timing replication, idle / uni, 20 interleaved shuffled rounds, `pay`-zeroed re-verify of every timed arm after
  timing: all PASS. Ratios vs `grid`:

| cell | grid#2 / grid_rc | tile | a2a | m.w.1.n.40 | none |
|---|---|---|---|---|---|
| tp1:32:ft idle | 1.012 / 1.009 | 0.992 W | 1.025 L | 0.976 W | 0.807 |
| tp1:128:ft idle | 0.999 / 0.997 | 0.992 | 1.011 L | 0.981 W | 0.886 |
| tp2:16:ft idle | 1.000 / 0.999 | 1.030 L | 1.028 L | 0.987 W | 0.810 |
| tp2:128:lb idle | 0.989 / 0.997 | 1.004 | 1.008 | 0.995 | 0.838 |
| tp1:16:ft0 idle (null) | 0.993 / 0.986 | 1.065 L | 1.029 L | 0.962 W | 0.787 |
| byteful geomean idle | 1.000 / 1.000 | 1.004 (1/2/1) | 1.018 (0/1/3) | **0.984 (3/1/0)** | 0.835 |
| byteful geomean uni | 0.999 / 0.997 | 0.990 (2/2/0) | 1.005 (0/2/2) | **0.975 (3/1/0)** | 0.804 |

  Matches section 3 within about 0.01 per cell (the builder's tp2:128:lb idle WIN is a TIE here because `grid#2`
  drifted to 0.989 in this run, which widened the control band).

**Conclusions check.** Section 5's decision follows from the data: the combination wins or ties every cell and loses
none on three runs (two nodes, three routing draws), and every arm outside the per-wave lazy combination (`tile`, `tile_w`, `a2a`, `mix90d`, the counted and drain variants) loses somewhere at idle. The "sync cannot
close the gap" claim (best arm recovers 0.007-0.034 of FlyTape's layer, `none` floor 0.80-0.84) is consistent with
these runs. It is a statement about synthetic stream bodies, as section 6 says; WP-L3 must re-decide in situ.
Required edits before this is cited as a full pass: drop or fix the "every timed binary re-verified (payload)" claim
in the gate section and fix `phase_time`'s re-verify (defect 1).

## Fix round (W6, 2026-11-01)

Every Verification issue is fixed in the bench, and both final runs were rerun with the fixed bench on both rocke nodes.
The kernel sources did not change: `sync_chain.py`, `sync_plan.py`, `persist/sync.py` and `ccl/wire.py` have the same
sha256 as in g / h, printed in both job logs. Only the harness changed (`hackathon-prep/qwen38/truemono-lb/p-l2/`).

**What changed in `pl2_bench.py`.**
1. **Post-timing re-verify (defect 1).** Before each re-verify launch (one per timed arm and skew), `phase_time` now
   zeroes the shared payload buffer. The launch still runs over the arm's own post-timing flag and epoch state, so it
   checks the timed binary in the state the timing left it in. Each record carries `pay_zeroed` and `tag`.
   - The re-verify is now a gate: the bench exits with rc 4 if a timed arm fails it or has nonzero error words from
     its timed launches.
   - **Must-fail sentinels.** `X<mutant>` arms (`Xtile_early` = `tile:early`, `Xmwn_early` = `mwn:early`) are timed
     exactly like arms, in the same shuffled rounds, so they reach the same epoch as every other arm. Their re-verify
     runs under `lastlate` and must FAIL. That is a standing, in-run proof that the re-verify can catch a broken
     binary at the shared tag, which is the condition that made the old check vacuous. Sentinels get no win-rule
     verdict.
   - `pl2_report.py` labels the re-verify form ("payload zeroed first" / "payload NOT zeroed: vacuous after the first
     arm, error words only") and lists the sentinel catches. The old g / h reports therefore read correctly when
     regenerated.
2. **`grid_rc` in numerics (defect 2).** The numerics arm list is now `grid`, `grid_rc`, then the other sync arms (8 more
   checks per run).
3. **Mutants of the recommended default's own build (defect 3).** `mwn:early` and `mwn:tag` use `pause=s_nop`, the
   build of `m.w.1.n.40`, with the 0.4 mix plan. They run next to the `s_sleep 1` twins `mw:*`: `early` under
   `lastlate` (gating) and `big` (info), `tag` at idle (gating). That adds 4 gating and 2 info checks per run.

**Reruns.** Same seeds, arms, cells, skews, rounds and stamp cells as g / h, plus the two sentinels:
- fg = job 68236089, n01-09, bus e5. That GPU passed the health check; the known bad e5 is n07-17's. Neighbours: 4 of
  our 1-GPU jobs, a foreign 2-GPU job and a foreign 1-GPU job.
- fh = job 68236088, n07-17, bus 05. Neighbours: one foreign 1-GPU job.

Stages: `w6-pl2-h3` (`pl2_bench.py` sha256 `dcc91dfc...`, `job.sh` unchanged) and `w6-pl2-rocke2` (the g / h tree).
Raw data, absolute reports and these tables are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/p-l2-raw/{w6-pl2-fg,w6-pl2-fh,fgh_*}`.

| check (per run) | fg (n01-09) | fh (n07-17) |
|---|---|---|
| numerics (every payload word, error words 0, ticket 256, epoch advanced; 2 launches each) | 136 / 136 PASS | 136 / 136 PASS |
| of which `grid_rc` (4 cells x idle / uni) | 8 / 8 | 8 / 8 |
| gating controls caught | 42 / 42 | 42 / 42 |
| of which `mwn:early` (`lastlate`) and `mwn:tag` (idle), 2 cells | 4 / 4 | 4 / 4 |
| informational controls caught | 19 / 20 (`a2a:early` under `big` on tp2:128:lb; its gating `lastlate` run is caught) | 20 / 20 |
| post-timing re-verify, payload zeroed (17 cells x 3 skews x 10 arms) | 510 / 510 PASS | 510 / 510 PASS |
| sentinels caught by that re-verify (17 x 3 x 2) | 102 / 102 (5404-29011 bad units each) | 102 / 102 (5356-28807) |
| bench rc | 0 | 0 |

**Timing replication** (the 13 real-byte cells; geomean of the per-cell median ratio vs `grid` [min, max]; WIN / TIE /
LOSS). The sentinels add two arms to each round, and the results reproduce section 3.1 within 0.005 per arm:

| arm | idle fg / fh | uni fg / fh | late1 fg / fh |
|---|---|---|---|
| `grid#2`, `grid_rc` (controls) | 0.999-1.001 [0.990, 1.007] | 1.001 [0.993, 1.011] | 0.999-1.002 [0.994, 1.011] |
| `tile` | 1.001 / 1.003, 6/4/3 and 2/7/4 | 0.992 / 0.995, no LOSS | 0.908 / 0.905, 13 WIN |
| `tile_w` | 1.017 / 1.017, 7 LOSS each | 0.994 / 0.996 | 0.920 / 0.918 |
| `a2a` | 1.016 / 1.016, 12 and 11 LOSS | 1.009 / 1.010 | 0.981 / 0.981 |
| `mix90d` | 0.993 / 0.996 | 0.991 / 0.991 | 0.911 / 0.908 |
| **`m.w.1.n.40`** | **0.983 / 0.983 [0.965, 1.004], 9/4/0 and 11/2/0** | **0.974 / 0.977, 12/1/0 and 11/2/0** | 0.927 / 0.927, 13 WIN |
| `m.w.1.s.40` | 0.981 / 0.984, 9/4/0 both | 0.977 / 0.981 | 0.928 / 0.927 |
| `m.w.1.s.90` | 0.991 / 0.992 | 0.980 / 0.982 | 0.912 / 0.911 |
| `none` (floor) | 0.832 / 0.834 | 0.805 / 0.808 | 0.608 / 0.611 |

- The recommended default loses no cell in either rerun at any skew. The null cells give 0.96-0.995 at idle (3 WIN and
  1 TIE per run) and 0.945-0.98 under `uni` (4 WIN per run).
- Its TIEs are the TP2 B32-B128 `ft` cells and tp1:64:lb, each at 0.987-1.004. g / h had the same pattern (section
  3.2).

**Issue 4 (minor: tp2:128:lb idle came out a TIE in the verifier's run).** This is not a defect of the arm. The win
rule compares against the extreme samples of both controls, so when one control drifts (`grid#2` at 0.989 there), the
band widens and a borderline cell ties. In the reruns `grid#2` / `grid_rc` stay within 0.996-1.006 on that cell, and
`m.w.1.n.40` is a WIN on both nodes (0.987 / 0.993; g / h had 0.993 / 0.986). The cell sits about 0.01 below the
barrier, so a TIE in one run out of five is within the rule's intended conservatism. The doc keeps calling it a
borderline WIN. The win rule itself is unchanged: widening the band on control drift is the intended behaviour.

**Issue 5 (scope caveat).** No code change is possible inside P-L2's scope, so the caveat stays as stated in
section 6:
- The stage bodies are synthetic byte streams.
- The TP2 cells are one rank's shape without the exchange; WP-C5 measures the exchange.
- Section 4.2 is a cross-job calibration against FlyTape, not a race.

The section 5 decision is explicitly provisional: WP-L3 re-decides it in situ on the real bodies and with the exchange.

**Status after the fix round.** All three defects are fixed and re-measured on hardware on both nodes. The gate
claims in "Gate and verdict" are true as now worded: numerics 136 / 136, gating controls 42 / 42, and the zeroed
re-verify 510 / 510 with 102 / 102 sentinels caught, per rerun. The headline and the section 5 decision are unchanged.
CPU tests: 29 pass. Kernel sources unchanged, so existing builds are unaffected. No git operations. No new cluster
fact: n01-09's healthy bus-e5 GPU is already in the alola-cluster skill.
