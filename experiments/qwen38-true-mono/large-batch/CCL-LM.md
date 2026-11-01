# CCL-LM: the large-message in-kernel all-reduce (TP2, layer-shaped), gfx950

W4, 2026-10-30/31. Follow-up of `../ccl/BUILD.md` ("Fix round": the grid-synced all-reduce lost to the aiter custom
all-reduce from 160 KiB per rank, 1.13-1.35). Ratios only; absolute device times, link rates and phase stamps are in
`hackathon-prep/private-perf/qwen38-true-mono/large-batch/CCL-LM.md` (raw JSON under `ccl-lm-raw/`).
Abbreviations as `../DESIGN-CCL.md`: `P` = `rocke/platform/python/rocke`, `Q` = `rocke/library/kernels/gfx950/qwen38_mono`.

## Gate and verdict

Gate: the bitwise suite still passes at TP2 / TP4, and the layer-shaped (grid-synced) all-reduce beats the aiter graph
at 160 KiB - 5 MiB per rank at TP2.

**Superseded in part by the "Fix round" section at the end.** That section has the final selection, the strict
per-round-best comparison and both rocke nodes. The gate holds from 80 KiB to 2.5 MiB. At 5 MiB the selection is at
link-bound parity with aiter, so that point is not met. The tables directly below are the first round's, measured on
earlier code with the weaker numerics gate.

**Met.** The CCL-LM selection (`Q/ccl/layout.py: lm_select`, declared before its confirmation run) beats aiter's
captured graph at every measured size from 80 KiB to 5 MiB per rank, bf16 and f32, under the DESIGN 9.1 win rule
(median per-round ratio below both controls' per-round minima by 0.002 and at least 3/4 of the rounds below 1):

| per-rank payload | shapes (rows x hidden, dtype) | selected build | CCL-LM / aiter graph | W3 grid-synced best / aiter graph |
|---|---|---|---:|---:|
| 80 KiB | 16 x 2560 bf16 | lamport one-shot | 0.725 | 0.998 |
| 160 KiB | 32 x 2560, 8 x 10240 bf16; 16 x 2560 f32 | lamport one-shot (f32: 2 KiB units) | 0.687-0.814 | 1.010-1.162 |
| 320 KiB | 64 x 2560, 16 x 10240 bf16; 32 x 2560 f32 | lamport one-shot (f32: 2 KiB units) | 0.776-0.890 | 1.120-1.216 |
| 640 KiB | 128 x 2560, 32 x 10240 bf16 | pull, 2 KiB units, 32 CTAs | 0.901 | 1.168-1.169 |
| 640 KiB | 64 x 2560 f32 | lamport, 2 KiB units | 0.881 | 1.131 |
| 1.25 MiB | 64 x 10240 bf16 | pull, 2 KiB units, 64 CTAs | 0.939 | 1.182 |
| 1.25 MiB | 128 x 2560, 32 x 10240 f32 | pull, 4 KiB units, 32 CTAs | 0.939-0.947 | 1.175-1.184 |
| 2.5 MiB | 128 x 10240 bf16; 64 x 10240 f32 | pull, 4 KiB units, 48 CTAs | 0.976 | 1.165-1.175 |
| 5 MiB | 128 x 10240 f32 | pull, 4 KiB units, 48 CTAs | 0.990 | 1.138 |

Every row is a WIN (job 68225555, GPU pair `f5` / `85`; the 5 MiB row: job 68225793, pair `75` / `85`, which also
re-confirmed 160 KiB, 640 KiB and 2.5 MiB: 0.813, 0.906, 0.982, all WIN, on the final code). The same configurations
won on two other GPU pairs in the lever sweep (68224788, `e5` / `95`) and the refinement (68225304, `f5` / `e5`); the margin shrinks
with the payload because both implementations become bound by the one xGMI link between the two GPUs (below). The
W3 forms reproduce the Fix round's loss (1.12-1.22 from 160 KiB). Counting the persistent launch itself (event time of
the 96-collective launch / 96, launch overhead included) the selection is still 0.72-0.98 of aiter's graph per call
(job 68225555, ratio of medians).

Suite (`bench_ccl test`, the 17 W3 tests + 3 CCL-LM tests) on the final code (`5df9b59bd8c0`): **20/20 at TP4**
(job 68227153, buses `05` / `15` / `e5` / `95`), **20/20 at TP2 and on a permuted TP2** (job 68226756). The 17 W3
tests' kernels are byte-identical throughout.

Scope of the comparison (as in `../ccl/BUILD.md`): rocKE = device-stamp latency per collective inside one persistent
launch of 96 chained collectives with a grid-wide barrier after each (the consumer needs the whole vector, as a model
layer does); aiter = a captured HIP graph of 96 `custom_all_reduce` calls (registered-input capture, ATOM's path), time
per call; both on the same two GPUs, in the same processes, rounds interleaved in a new shuffled order (20 rounds),
numerics first (every rocKE arm bitwise against the CPU rank-order reference on every rank, aiter and RCCL against a
float64 sum). Controls per size: two more separately captured aiter graphs (A/A of the baseline), the W3 line one-shot
and the selected build each launched twice from the same binary and once from a recompiled copy. Relative results at
these configurations only (one 8-GPU gfx950 server, `smci355-ccs-aus-n01-09`, uncached arena, q38 container: torch
2.13 + ROCm 10 user space, aiter as installed there).

## What aiter does at these sizes (installed source, `csrc/include/custom_all_reduce.cuh`, sha256 `08c73190...`)

`allreduce()` at world size 2 (the container's thresholds, read from the installed file):
- bf16 / fp16 up to 32K elements (64 KiB): **LL** (`ar_ll_gfx9`, 256-thread blocks): every 16-byte line carries 8 bytes
  of data and two 4-byte epoch flags, pushed with nontemporal stores into the peer's scratch; the reader spins on the
  flags; 50 % payload efficiency, no barrier.
- up to 128K elements (256 KiB): **LL128** (`ar_ll128_unroll2`, 256 threads): 128-byte lines of 15 x 8 bytes of data
  plus an 8-byte flag, pushed with nontemporal 16-byte stores (it relies on a 128-byte line arriving whole); up to 192
  blocks. Up to 448K elements (896 KiB): the same with 512-thread blocks.
- above, and every f32 size: **1-stage** (`cross_device_reduce_1stage`, 80 blocks x 512 threads): a start barrier (one
  flag per block to the peer), then every block **reads** both ranks' inputs through the IPC-registered pointers (16-byte
  loads, double-buffered through LDS), warp 0 sums in rank order, an end barrier. A pull scheme.
- (world size 4: LL / LL128 for small sizes, 1-stage below 160 KiB, 2-stage pull reduce-scatter + all-gather above.)

So against our gate: 160-640 KiB bf16 meets aiter's LL128 push with in-line flags; 1.25 MiB and up, and all f32
sizes, its 1-stage pull. aiter's graph pays a kernel boundary per call; the rocKE layer shape pays a grid sync.

## Diagnosis (phase stamps and link probes, before the levers)

Tools (all kept): `diag` builds (`CclKnobs(diag=True)`: thread 0 of every CTA stamps 8 phases of collectives 8-11:
start, pushes issued, drained, published, consumed, grid-sync arrival, release, end; `P/examples/gfx950/ccl/bench_ccl_lm
diag`, per-CTA analysis `hackathon-prep/qwen38/truemono-lb/ccl-lm/diag_lm.py`) and an xGMI probe
(`Q/ccl/lm_probe.py`, `bench_ccl_lm linkbw`: push or pull of N bytes between the two ranks, one or both directions,
any subset of XCDs). WaveScope ATT was not captured (two-process torchrun capture was not attempted); the stamps carry
the diagnosis. Findings (ratios of the W3 line one-shot's per-collective span unless said otherwise):

1. **The line form waits a full write round trip before it can signal.** A wave pushes its line to both ranks, waits
   `vmcnt(0)` (the remote write acknowledgement), then stores the flag, which flies one more hop. At 160 KiB the push
   issue + drain + flag take 0.45 of the span before the consumer can even start polling; aiter's LL / LL128 put the
   flag inside the line. -> **lamport form** (payload-only lines, no drain, no flag).
2. **At 640 KiB and up the slowest CTA, not the link, sets the end.** With every CTA pushing at once the units complete
   in an XCD-dependent order (on one pair XCD pairs 4/5 first and 0/1 last; on another XCD 6 first), and the grid sync
   waits for the last CTA: at 2.5 MiB the median CTA had consumed its unit at 0.78 of the span (W3 line) / 0.76 (lamport)
   and the release came at 0.99; the lamport one-shot's slowest CTA finished about 0.24 of a collective after the
   median one.
3. **One XCD alone nearly drives the link** (probe, 1 MiB per direction moved by the 32 CTAs of one XCD or of one XCD
   pair, against the all-XCD rate of a 2.5 MiB transfer): 0.91-0.97 of the all-XCD rate for one XCD and 0.92-0.995 for
   one pair, push and pull alike (the 1 MiB runs also carry more fixed start-up cost), so the per-XCD fabric path is
   not the limit and per-XCD weighting cannot add bandwidth (the `place="table"` lever below confirms it).
4. **Pull moves more bytes per second than push**: the same transfer takes 0.955-0.962 of the push time when both GPUs
   send at once and 0.89-0.93 one-way (2.5 MiB and 32 MiB); `nt` stores change nothing (0.998-1.001). aiter's 1-stage
   (a pull) is near this link limit from about 1 MiB, which is why the margin shrinks at the largest sizes.
5. With pulls on a capped grid (32-48 CTAs, 2-4 KiB units, every wave looping over several units) the completion is
   XCD-uniform and tight: at 2.5 MiB the slowest CTA consumed 0.05-0.10 of a collective after the median one (lamport on
   the full grid: 0.24, same job and GPUs).

## What was built (all default-off knobs; every earlier build is byte-identical)

| knob (`CclKnobs`) | values | what |
|---|---|---|
| `form="lamport"` | push | payload-only lines: the producer stores the bit complement of its canonicalised wire data (the one pattern that would read as empty, bf16 `0xFFFF` / f32 `0xFFFFFFFF`, becomes the quiet NaN first); the consumer polls the payload itself (every element checked: no reliance on 8/16/128-byte store atomicity), zeroes what it read, and every wave drains its stores before its next push (WAR safe at S = 2; `Q/ccl/war.py` models the reset events and its no-drain mutant fails). One- and two-shot. |
| `form="pull"` | pull | every rank stages its partial in its own arena, publishes a flag per unit, consumers read every rank's unit through the peer bases with system-scope loads (aiter's 1-stage scheme). One-shot. |
| `unit_kib` | 1 / 2 / 4 | wider units (each lane moves 1-4 16-byte vectors per push / poll / read). |
| `ctas` | 0 / 1-256 | cap of the grid (fewer, longer-running CTAs). |
| `publish` | `unit_end` / `counted` / `comm_wave` | the two not-built publish variants of DESIGN-CCL 4.2.1: `counted` (the next unit's input load in flight while the stores drain, `vmcnt(n)` instead of `vmcnt(0)`; for pull: unit-by-unit staging) and `comm_wave` (waves stage their units in LDS, one wave per CTA pushes and drains). |
| `self_local` | bool | lamport one-shot keeps its own partial in registers (pushes to the peers only). |
| `ar_dtype` | `wire` / `bf16` | bf16 partials of f32 data (tolerance class). |
| `gsync` | `word` / `stripe` | grid sync of the layer shape: one system-scope arena word (W3) or 64 agent-scope stripes in device memory polled lane-parallel by wave 0 (P4's `s8`). |
| `place` | `rr` / `table` | unit -> wave placement from a host table weighted per XCD (calibrated in-job from diag stamps). |
| `pause` | + `s_sleep4`, `s_sleep16` | longer poll pauses. |
| `diag` | bool | the phase stamps above (diagnostic builds only, never timed for ratios). |

Plus: `layout.lm_select` / `auto_allreduce_spec(..., lm=True)` (the selection above; other rank counts and payloads
under 80 KiB keep W3's `select_knobs`), the link probe `Q/ccl/lm_probe.py`, the peer lint's rule that the DATA returned
by a load-only asm block is not an address (pull reads), and `P/examples/gfx950/ccl/bench_ccl_lm.py` (bench / diag /
linkbw) with the aiter plugin `hackathon-prep/qwen38/truemono-lb/ccl-lm/aiter_arm_lm.py`.

Byte identity: the LLVM IR of all 2284 earlier CCL builds (every bench and suite build at TP 1-4) is unchanged
(`ccl-lm/ir_snapshot.py`, sha256 of the IR text before and after); no file outside `Q/ccl/`, the CCL example, the two
CCL test files and the contract test's spec list was touched.

## Levers (lever / the arm it changes; per-round ratio, DESIGN 9.1 verdict counts WIN / LOSS / noise)

Job 68224788 (`e5` / `95`, 12 rounds, 16 sizes; S = up to 320 KiB, M = 640 KiB-1.25 MiB, L = 2.5-5 MiB). Controls:
the line one-shot's or the lamport one-shot's same-binary and recompiled twins (0.99-1.01).

| lever | base | S | M | L | verdict |
|---|---|---|---|---|---|
| lamport one-shot | line one-shot | 0.66-0.79 (7/0/0) | 0.82-0.88 (6/0/0) | 0.89-0.91 (3/0/0) | **win everywhere** |
| lamport two-shot | line two-shot | 0.67-0.78 (7/0/0) | 0.87-0.92 (6/0/0) | 0.92-0.97 (3/0/0) | win |
| pull 4 KiB | lamport 4 KiB | 1.00-1.05 (0/5/2) | 0.96-0.99 (6/0/0) | 0.95-0.97 (3/0/0) | **win from 640 KiB** |
| pull 4 KiB on 32 CTAs | pull 4 KiB, default grid | - | 0.98-0.99 (3/0/0, H1) | 0.97-0.98 (3/0/0) | win (refined to 48 CTAs at L) |
| 2 KiB units | 1 KiB (lamport) | 0.95-1.06 (4/3/0) | 0.89-0.94 (6/0/0) | 0.92-1.00 (2/0/1) | win from 640 KiB (f32 from 160 KiB) |
| 4 KiB units | 1 KiB (lamport) | 1.04-1.58 (0/7/0) | 0.88-0.99 (6/0/0) | 0.88-1.01 (2/1/0) | win at M |
| 2 / 4 KiB units | 1 KiB (line) | 1.01-1.26 (0/13/1) | 0.86-0.96 (12/0/0) | 0.87-0.95 (6/0/0) | win from 640 KiB |
| 16 waves per CTA | 8 (lamport) | 0.97-1.17 (1/4/2) | 0.89-0.94 (6/0/0) | 0.92-1.03 (2/1/0) | mixed |
| 4 waves per CTA | 8 (lamport) | 0.97-1.12 (1/6/0) | 1.12-1.17 (0/6/0) | 0.97-1.04 (2/1/0) | loss |
| `comm_wave` publish | line one-shot | 0.71-0.83 (7/0/0) | 0.86-0.90 (6/0/0) | - (one unit per wave only) | win over line, but lamport is faster |
| `counted` publish | line one-shot (unit_end) | 0.84 (1/0/0) | 0.90 (1/0/0) | 0.98-0.99 (2/0/0) | win over unit_end (job 68225793), still behind lamport; on 32-CTA multi-unit waves 0.96 / noise; for pull: noise (0.98-1.00) |
| two-shot | one-shot (lamport) | 1.00-1.24 (0/6/1) | 0.97-1.00 (3/0/3) | 0.97-0.98 (3/0/0) | win at L, below pull |
| dual flag homing under the grid sync | line one-shot (word and stripe sync) | 0.82-0.93 (14/0/0) | 0.97-1.01 (6/2/4) | 1.04-1.06 (0/6/0) | win at S only (line form) |
| striped agent-scope grid sync | one system-scope word | 0.96-1.03 (8/1/5) | 0.97-1.06 (5/6/1) | 1.01-1.08 (0/6/0) | win at S, loss at L |
| `self_local` | lamport one-shot | 0.97-1.03 (1/2/4) | 1.00-1.03 (0/4/2) | 0.96-1.00 (2/0/1) | noise |
| 3 slots | 2 | 1.00 (0/0/7) | 1.00 (0/0/6) | 1.00 (0/0/3) | noise |
| poll pause `s_sleep 1` / none / `s_sleep 4` | `s_nop` | 0.995-1.004 (0/1/20) | 0.995-1.004 (0/0/18) | 0.995-1.001 (0/0/9) | noise |
| poll pause `s_sleep 16` | `s_nop` | 1.00-1.03 (0/5/2) | 0.98-1.01 (3/3/0) | 0.98-1.00 (2/0/1) | loss at S |
| `nt` pushes | system scope only | 1.02-1.05 (0/14/0) | 1.00-1.01 (0/5/7) | 1.00-1.01 (0/2/4) | loss at S |
| XCD-weighted placement | round robin (lamport) | 1.01-1.11 (0/7/0) | 1.00-1.02 (0/4/2) | 1.02-1.03 (0/3/0) | loss (link-bound, finding 3); 4 KiB lamport / pull: mixed at S, loss or noise at M / L |
| bf16 partials of f32 data | f32 wire (lamport) | 0.64-0.75 (2/0/0) | 0.53-0.57 (3/0/0) | 0.54-0.60 (2/0/0) | win, tolerance class: against aiter's f32 graph 0.60-0.64 at 2.5-5 MiB, not a matched-precision result |

The first counted builds lost 1.5-2.5x (job 68224788): the lanes that publish nothing stored their dummy flag to one
shared word, and those same-address stores from every wave serialised; spreading them over 2048 words of the unused
arena gap fixed it (the row above is the fixed build). Losers stay registered default-off knobs. The stripe sync is used by the selection everywhere (it won at S; at L the
pull arms tie with and without it, H1).

## Numerics (on hardware, before every timing)

- Suite `bench_ccl test`: the 17 W3 tests plus `lm_bitwise` (every new knob and the `lm_select` builds bitwise against
  the CPU reference on every rank, chained, 2-3 launches without a reset, rows 1-128, bf16 / f32, 32-CTA grids, random
  per-rank placement tables with one XCD idle), `lm_canon` (planted `0xFFFF` / `0xFFFFFFFF` inputs: lamport one- and
  two-shot correct, NaN-aware), `lm_mutants`. Must-fail controls, each with exactly one allowed outcome and each seen to
  fail: `nocanon` (PEER_TIMEOUT on every rank), `noreset` (wrong results on every rank, one- and two-shot), pull `slot`
  (wrong results on every rank), `gsearly` (stripe sync releasing one arrival early: the release check fired at every
  grid sync, 30 / 30 per rank, while the real stripe build under the same delays stayed clean and bitwise), plus a
  dropped lamport use (PEER_TIMEOUT at that use) and 600-launch lamport delay-injection stress (one- and two-shot).
- Final code (`5df9b59bd8c0`): 20/20 at TP4 (68227153), at TP2 and on a permuted TP2 (68226756). Earlier runs of the lamport / unit / stripe / table / pull code: 20/20 at TP2 and TP4
  (68225064, `n07-17`), and the 17 W3 tests on TP4 coarse memory and a permuted TP2 (68222638).
- Every timed arm of every benchmark job passed its numeric gate (rocKE bitwise, aiter / RCCL within 2e-2 bf16 /
  1e-5 f32 of a float64 sum).

## Caveats and what is not done

- One node (`n01-09`): `n07-17`'s GPU on bus `e5` faults on any large allocation (alola skill rules 146, 150, 151), so
  the confirmation on the other node (DESIGN 9.1) could not be run in the container; three GPU pairs of `n01-09`
  instead.
- TP2 only for the timing (the gate); TP4 timing against aiter was not re-run (W3: 1.13-1.35 at 160 KiB+). The
  selection keeps W3's choice at TP != 2.
- The comparison is the layer shape of the microbenchmark, not WP-C5's in-situ GEMM + all-reduce (still not run).
- The `ctas` cap and the pull form decide how many CTAs a fused layer kernel would dedicate to the exchange; in a
  persistent model kernel that is a work-assignment choice, measured here only in isolation.
- 5 MiB f32 was confirmed in a separate short job (68225793) after the 16-size confirmation job lost aiter's last graph
  capture (`hipIpcOpenMemHandle` invalid argument inside aiter's registration at the 16th size).
- A WaveScope ATT capture of the two-process benchmark was not taken.

## Reproduce

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 -m rocke.examples.gfx950.ccl.bench_ccl_lm bench \
    --out DIR --rounds 20 --auto --arms none --no-ctas-default --extra-arms aiter_arm_lm:make   # the declared selection
python -m torch.distributed.run --standalone --nproc_per_node 2 -m rocke.examples.gfx950.ccl.bench_ccl_lm bench --out DIR  # every lever
python -m torch.distributed.run --standalone --nproc_per_node 2 -m rocke.examples.gfx950.ccl.bench_ccl_lm diag --out DIR
python -m torch.distributed.run --standalone --nproc_per_node 2 -m rocke.examples.gfx950.ccl.bench_ccl_lm linkbw --out DIR
python -m torch.distributed.run --standalone --nproc_per_node 4 -m rocke.examples.gfx950.ccl.bench_ccl test --out DIR
```

Harness and jobs: `hackathon-prep/qwen38/truemono-lb/ccl-lm/` (`job.sh` venv, `job_ctr.sh` container, `ratios_lm.py`,
`levers_lm.py`, `diag_lm.py`, `ir_snapshot.py`, `compile_lm.py`). Jobs: suites 68221864, 68222453, 68222638,
68223420, 68225064, 68226756, 68227153 (final code); diag + link probes 68221864, 68222453, 68225555; benchmarks 68224788 (lever sweep),
68225304 (refinement), 68225555 + 68225793 (declared selection; 68225793 also re-measured the counted publish).

## Verification (independent verifier, W4, 2026-10-31)

Verdict: **partial**. The gate holds: I re-ran it on hardware, on the final code, on two GPU pairs not used for the
declared-selection timing, with a fresh interleaving seed. But two things were not shown by the builder's own
evidence, and one was not done:
- The numerics gate in the benchmark cannot detect stale-data reads. I closed this for the 16 timed builds (below).
- The 5 MiB row is a near-tie against the best aiter capture.
- The other-node confirmation that DESIGN 9.1 requires was not run.

Jobs:
- 68228402 `w4-vCCL-LM-a`: container, buses `e5` / `95`.
- 68228541 `w4-vCCL-LM-b`: venv, buses `f5` / `e5`.
- 68228642 `w4-vCCL-LM-c`: venv, buses `15` / `95`.

All three ran on `smci355-ccs-aus-n01-09`, code `5df9b59bd8c0`, with GPUs idle per `amd-smi`. The harness is in
`hackathon-prep/qwen38/truemono-lb/vccl-lm/` and is a wrapper only: no builder file was edited. Raw data is under
`private-perf/qwen38-true-mono/large-batch/ccl-lm-raw/` (`68228402-vA`, `68228541-vB`, `68228642-vC`).

**What I checked against the logs and code (confirmed):**
- **Suite and mutants ran on hardware with failing controls.**
  - The `.out` files and test JSON of 68227153 (TP4 20/20) and 68226756 (TP2 and permuted TP2 20/20, on n07-17
    buses `75` / `15`) show every must-fail control failing:
    - `lm_mutants`: pull `slot` detected on every rank; `noreset` one- and two-shot detected on every rank;
      `gsearly` 30/30 release-check hits against 0/30 for the real stripe build; `drop_use` gives PEER_TIMEOUT / PEER_ABORT.
    - `lm_canon`: `nocanon` gives PEER_TIMEOUT.
  - Re-run in 68228402: 20/20 at TP2 on `e5` / `95`, final code.
- **Byte identity.** `ccl-lm/ir_snapshot.py` on the final code against `ir_base.json`: 2284/2284 builds identical. The
  builder's last snapshot predates `5df9b59bd8c0`, so this re-run is the first check on the final code.
- **Recomputed ratios.** The ratio tables reproduce exactly from the raw JSON (`ratios_lm.py`).
  - The launch-included ratio (event time of the 96-collective launch / 96, against aiter's graph time per call) is
    0.716-0.993 across all 19 measured rows of 68225555 and 68225793.
  - Against the per-round best of the three aiter captures, every round of every size is below 1 on device time
    (maximum 0.994).
- **Fairness.**
  - Same two GPUs in the same processes; the same rows x hidden and dtype on both sides (bf16 against bf16, f32
    against f32).
  - aiter is used on ATOM's path: a registered-input graph capture. Its installed dispatch was read from source.
  - The rocKE side does more work, not less: a grid sync after every collective plus the chain scaling.
  - aiter is the stronger baseline: aiter / RCCL best graph is 0.56-0.92.
  - PLAN decision 5's DP2 arm does not apply to a collective microbenchmark, since DP2 has no all-reduce. The report
    should say so, and it does not establish anything about TP2 against DP2 per serving cell.
- **Interleaving and controls.**
  - The arm order is shuffled every round. The shuffle seed is a fixed function of (round, size), so equal arm lists
    get identical orders in every job; 68228402 used a fresh seed (`VCCL_SEED`).
  - aiter has two A/A captures. `lm_auto` has same-binary and recompiled twins (0.998-1.002 everywhere).

**Re-run of the decisive timing (68228402):** declared selection against aiter, 20 rounds, fresh seed, final code.

| size | CCL-LM / aiter graph (9.1 verdict) | / best aiter capture per round (device; launch-included) | rounds < 1 |
|---|---:|---:|---:|
| 32 x 2560 bf16 (160 KiB) | 0.815 WIN | 0.819; 0.847 | 20/20 |
| 128 x 2560 bf16 (640 KiB) | 0.906 WIN | 0.907; 0.920 | 20/20 |
| 128 x 10240 bf16 (2.5 MiB) | 0.984 WIN | 0.984; 0.988 | 20/20 |
| 128 x 10240 f32 (5 MiB) | 0.979 WIN | 0.997; 0.999 (round max 1.005) | 20/20 |

These match the report (0.813-0.817, 0.901-0.906, 0.976-0.982, 0.990).

**At 5 MiB the margin is inside aiter's own capture-to-capture spread.** In this job one aiter capture was 0.981-0.983
of the other, so "0.979" means about 0.3 % against the best capture: about 0.1 % launch-included, with one round above
1. 68225793 gives the same picture: 0.993 device / 0.996 launch-included against the best capture. The 9.1 rule is
formally met, but read 5 MiB as a tie, not as 0.990.

GPU 0's clock sampler stayed at 0.98-1.0 of its maximum during the timed step, so no throttled rounds. The builder's timing jobs
did not sample clocks (DESIGN 9.1).

**New finding: the benchmark's per-arm numerics gate is blind to stale-partial reads.**
- **Cause.** The chain scales (TP2: 1/3, 2/3) make the 96-collective chain reach a bitwise fixed point after 2-3
  collectives: `ref.allreduce_chain` output is unchanged from iteration 3 (bf16) / 4 (f32). A build that reads the
  previous collective's partial therefore matches the reference. That covers a wrong ring slot, and also a
  read-before-publish race that sees the stale slot, which is exactly the bug that would make an arm look faster.
- **Shown on hardware.**
  - 68228402: the pull `slot` mutant of the selected 2.5 MiB build passed the bench gate at 160 KiB and 2.5 MiB on
    both ranks. Only lamport `noreset` was dropped.
  - 68228541: 3-collective chains (the suite's length) run right after the real build with the same seed also let all
    7 pull `slot` twins through, on both rank orders.
- **Implications.**
  - The report's "every timed arm of every benchmark job passed its numeric gate" is true but weak.
  - The suite's `lm_bitwise` reuses one seed across forms of a row count and runs 3-4 collective chains without a reset,
    so it is only partly discriminating for this bug class too. Its pull `slot` control is detected because that test
    resets first.
- **Closed for the selection (68228642).** I ran every one of the 16 exact timed `lm_auto` builds (`lm_select` +
  `header=False` + `sync="grid"` + `gsync="stripe"`) at the timed length (96 chained collectives, 2 launches) with
  non-contracting per-rank scales 0.75 / 0.5 (the chain grows 1.25x per collective and never reaches a fixed point).
  - All 16 are bitwise on every rank, on both rank orders.
  - All 16 must-fail twins (7 pull `slot`, 9 lamport `noreset`) fail.
  - So the timed selection is correct under a discriminating check.
- **Not closed for the other ~70 lever arms** of the sweep (68224788, 68225304). Their lever verdicts rest on the
  weak gate.
- **Fix.** Make the bench's chain scales non-contracting (as `vccl_exact2.py` does) and add a must-fail stale-slot arm
  to every bench job.

**Not done or caveats (beyond the report's own):**
1. The DESIGN 9.1 confirmation on the other rocke node was not run. The verifier's runs are again n01-09 only, because
   n07-17's bus-`e5` GPU still faults.
2. The 16-size confirmation job 68225555 ran code `8584df4ec488`, not the final `5df9b59bd8c0`. 68225793 and 68228402
   re-measured 5 of the sizes on the final code; the rest rest on the byte identity of all earlier builds plus
   unchanged `lm_auto` spec names.
3. A small missed lever: with pull on the default grid, the one-word grid sync is 0.3-0.6 % faster than the stripe sync
   at 640 KiB-2.5 MiB (68225304), but the selection uses the stripe everywhere. Capped-grid pull with the word sync was
   not measured.
4. `tests/test_builder_signature_contract.py` fails three tests. The cause is non-CCL `qwen38_mono` specs (probes,
   framework, moe, mailbox), not CCL-LM. `job.sh cpu` ignores that test's exit status.

## Fix round (W4, 2026-10-31)

All nine verifier issues are addressed below. One part of the gate is not met: at 5 MiB per rank the selection is at
parity with aiter, not ahead of it. Everything here is on the final code (`90e9e010478e`) unless a row says otherwise.
Ratios only; absolutes are in the private record ("Fix round"); raw data is in `ccl-lm-raw/<job>-f*`. Harness:
`hackathon-prep/qwen38/truemono-lb/ccl-lm-fix/` (`job_ctr.sh`, `job.sh`, `pick_gpus.py`, `ratios_fix.py`,
`final_table.py`, `levers_lm.py`, `clocks.py`, `ir_levers.py`, `ir_snapshot.py`, `contract_ccl.py`, `pair_stress.py`).

### Verdict after the fix round

The selection beats aiter's captured graph at every measured size from 80 KiB to 2.5 MiB per rank, on both rocke
nodes:
- under DESIGN 9.1 against the first capture;
- under the strict comparison against the best of aiter's three captures in the same round;
- with the persistent launch counted.

At 5 MiB (f32) it is at parity:
- strict WIN in 3 of 5 runs: n07-17 buses `05` / `65` twice, n01-09 buses `15` / `95` once;
- tie or slight loss in 2 runs on n01-09 buses `05` / `15`.

Every lever that could move this size stays within 0.993-1.02 of aiter there: both grid syncs, caps of 32-160 CTAs,
2 or 4 KiB units, 16 waves, the counted publish, and the arena memory type. Both implementations are bound by the one
xGMI link at this size. So the gate is met for 160 KiB-2.5 MiB and **not met at 5 MiB**: that point is a link-bound tie.

Final selection (declared in `layout.lm_select(..., layer=True)` before the final confirmation jobs).
- n01-09: jobs 68229464 (12 sizes) and 68229530 (the 4 f32 sizes that aiter lost in 68229464), both on buses `05` / `15`.
- n07-17: job 68229473, buses `05` / `65`.
- Each cell: `r1` = CCL-LM / aiter graph (first capture, 9.1 verdict); `rbest` = against the best capture of the same
  round; `lbest` = `rbest` with the launch counted (event time of the 96-collective launch / 96). Strict rule: WIN if
  `r1` is a 9.1 WIN, median <= 0.998 and at least 3/4 of rounds below 1; TIE otherwise unless a mirrored loss.
- Rounds overlapping a low-clock sample are dropped (clock filter, below).

| size | KiB / rank | build (`q38ccl_ar_n2_r*_c*_<dtype>_` + ...) | n01-09: r1 / rbest / lbest | n07-17: r1 / rbest / lbest | W3 best / aiter |
|---|---:|---|---|---|---:|
| 16 x 2560 bf16 | 80 | `lamport_nohdr_w4_sygrid_gsstripe` | 0.713 WIN / 0.717 WIN / 0.759 WIN | 0.710 WIN / 0.715 WIN / 0.762 WIN | 0.995-1.005 |
| 32 x 2560 bf16 | 160 | `lamport_nohdr_sygrid_gsstripe` | 0.817 WIN / 0.818 WIN / 0.850 WIN | 0.813 WIN / 0.814 WIN / 0.848 WIN | 1.153-1.321 |
| 8 x 10240 bf16 | 160 | `lamport_nohdr_sygrid_gsstripe` | 0.816 WIN / 0.821 WIN / 0.854 WIN | 0.814 WIN / 0.817 WIN / 0.849 WIN | 1.153-1.323 |
| 16 x 2560 f32 | 160 | `lamport_nohdr_sygrid_u2_gsstripe` | 0.690 WIN / 0.691 WIN / 0.718 WIN | 0.655 WIN / 0.657 WIN / 0.683 WIN | 1.002-1.024 |
| 64 x 2560 bf16 | 320 | `lamport_nohdr_sygrid_gsstripe` | 0.878 WIN / 0.879 WIN / 0.903 WIN | 0.869 WIN / 0.874 WIN / 0.902 WIN | 1.179-1.283 |
| 16 x 10240 bf16 | 320 | `lamport_nohdr_sygrid_gsstripe` | 0.875 WIN / 0.877 WIN / 0.905 WIN | 0.868 WIN / 0.872 WIN / 0.897 WIN | 1.184-1.280 |
| 32 x 2560 f32 | 320 | `lamport_nohdr_sygrid_u2_gsstripe` | 0.788 WIN / 0.789 WIN / 0.807 WIN | 0.763 WIN / 0.764 WIN / 0.782 WIN | 1.057-1.140 |
| 128 x 2560 bf16 | 640 | `pull_nohdr_sygrid_u2_gsstripe_g32` | 0.931 WIN / 0.932 WIN / 0.951 WIN | 0.900 WIN / 0.901 WIN / 0.918 WIN | 1.144-1.195 |
| 32 x 10240 bf16 | 640 | `pull_nohdr_sygrid_u2_gsstripe_g32` | 0.934 WIN / 0.935 WIN / 0.948 WIN | 0.899 WIN / 0.901 WIN / 0.918 WIN | 1.144-1.198 |
| 64 x 2560 f32 | 640 | `lamport_nohdr_sygrid_u2_gsstripe` | 0.879 WIN / 0.885 WIN / 0.897 WIN | 0.876 WIN / 0.877 WIN / 0.889 WIN | 1.109-1.136 |
| 64 x 10240 bf16 | 1280 | `pull_nohdr_sygrid_u2_gsstripe_g64` | 0.961 WIN / 0.967 WIN / 0.976 WIN | 0.937 WIN / 0.948 WIN / 0.955 WIN | 1.124-1.171 |
| 128 x 2560 f32 | 1280 | `pull_nohdr_sygrid_u4_pubcnt_gsstripe_g32` | 0.941 WIN / 0.945 WIN / 0.953 WIN | 0.929 WIN / 0.937 WIN / 0.944 WIN | 1.128-1.141 |
| 32 x 10240 f32 | 1280 | `pull_nohdr_sygrid_u4_pubcnt_gsstripe_g32` | 0.939 WIN / 0.942 WIN / 0.950 WIN | 0.935 WIN / 0.939 WIN / 0.946 WIN | 1.134-1.136 |
| 128 x 10240 bf16 | 2560 | `pull_nohdr_sygrid_u4_g64` | 0.990 noise / 0.996 TIE / 1.000 TIE | 0.977 WIN / 0.977 WIN / 0.981 WIN | 1.160-1.177 |
| 64 x 10240 f32 | 2560 | `pull_nohdr_sygrid_u4_g64` | 0.972 WIN / 0.978 WIN / 0.981 WIN | 0.974 WIN / 0.974 WIN / 0.978 WIN | 1.150-1.159 |
| 128 x 10240 f32 | 5120 | `pull_nohdr_sygrid_u4_g64` | 0.993 noise / 0.996 TIE / 0.998 TIE | 0.983 WIN / 0.993 WIN / 0.995 WIN | 1.138-1.143 |

Other runs of the same final binaries (byte-identical IR; the intermediate code `4894401c108a` differs from the final
code only in the 1.25 MiB f32 band and in host code):

| size | run (node, buses) | r1 / rbest / lbest |
|---|---|---|
| 2.5 MiB bf16 | 68229224 (n01-09 `e5` / `95`) | 0.982 WIN / 0.982 WIN / 0.986 WIN |
| 2.5 MiB bf16 | 68229249 (n07-17 `05` / `65`) | 0.976 WIN / 0.976 WIN / 0.980 WIN |
| 2.5 MiB bf16 | 68229576 (n01-09 `05` / `15`) | 0.975 WIN / 0.976 WIN / 0.980 WIN |
| 2.5 MiB bf16 | 68229681 (n01-09 `05` / `15`) | 0.977 WIN / 0.978 WIN / 0.982 WIN |
| 5 MiB f32 | 68229249 (n07-17 `05` / `65`) | 0.993 WIN / 0.993 WIN / 0.995 WIN |
| 5 MiB f32 | 68229365 (n01-09 `15` / `95`) | 0.988 WIN / 0.992 WIN / 0.994 WIN |
| 5 MiB f32 | 68229681 (n01-09 `05` / `15`) | 1.002 noise / 1.004 LOSS / 1.006 LOSS (3 of 20 rounds below 1) |

- **2.5 MiB bf16:** a WIN in 5 of 6 runs. The miss is job 68229464.
- **Job 68229464 was slow as a whole.** At 640 KiB-2.5 MiB every rocKE form was 1.5-3 % slower than in job 68229576 on
  the same GPU pair, while aiter and RCCL moved by under 1 %. That is job-level spread, not a pair property:
  68229576 and 68229681 on the same pair give WINs at 2.5 MiB.
- **5 MiB f32:** buses `05` / `15` gave a tie and a slight loss in two separate jobs, so at that size it is a property of
  the pair or the size. It is parity.

### What changed in the selection (fix-round levers, all confirmed on both nodes)

- **80 KiB: 4 waves per CTA** (lamport one-shot `waves=4`).
  - WIN against the old 8-wave build in the sweep and refinement (0.968-0.978).
  - Old / new: 1.016 (n01-09, a LOSS for the old build) and 1.012 (n07-17, noise).
  - At 160 KiB 4 waves lose (1.05), so the band is 80 KiB only.
- **Above 1.25 MiB: pull with 4 KiB units on 64 CTAs and the one-word grid sync** (was 48 CTAs with the stripe sync).
  - Against the old selection, sweep and refinement: 0.989-0.993 WIN at 2.5 MiB, noise at 5 MiB.
  - Old / new in the confirmations: 1.006-1.011 LOSS for the old build on both nodes at 2.5 MiB; 1.002 at 5 MiB.
- **1.25 MiB f32: counted publish** (pull, 4 KiB units, 32 CTAs, `publish="counted"`).
  - WIN 0.992-0.995 in the refinement and both intermediate confirmations.
  - Old / new in the final jobs: 1.007-1.008 LOSS for the old build on both nodes.
  - The knob-capped build is the same binary as the measured runtime-capped one: the IR is equal apart from the
    kernel name.
- **Interface:** `lm_select(..., layer=True)` now returns the layer shape with each band's measured grid sync. Before,
  the bench forced the stripe sync for every band. The non-layer `lm=True` builds only change in `waves` / `ctas` /
  `publish`.

### Issue 1: the bench's numerics gate was blind to stale-partial reads

- **Fix.** `ref.chain_scales(ranks, growth)` with `ref.LM_GROWTH = 1.25`. `bench_ccl_lm` runs its per-arm gate at the
  timed length (96 chained collectives) with these non-contracting scales by default (`--chain-growth`). The chain's sum
  grows 1.25x per collective, so it never reaches a fixed point. Growth 1 is W3's chain; its kernarg values are
  unchanged.
- **Must-fail twins at every size.** Every size also runs must-fail stale-read twins right after the real arms, never
  timed:
  - `mut_pull_slot`: the 2.5 MiB-class pull build reading the other slot;
  - `mut_lam_noreset`: lamport without the consumer reset;
  - `mut_lm_auto`: the selection's own twin, `slot` for pull and `noreset` for lamport.
  A twin the gate accepts prints `[lm] MUSTFAIL MISSED` and the bench exits with rc 4.
- **CPU proof.** `test_contracting_chain_hides_stale_reads_growth_does_not` (TP2 and TP4, bf16 and f32): a lag-1 or
  lag-2 stale read matches the reference bit for bit under W3's scales and never under growth.
- **Hardware, all fix-round benches.**
  - 3120 / 3120 rocKE arm-size gates bitwise. That covers 124 distinct arms: every lever arm of the sweep and the
    refinement, which closes the verifier's "~70 lever arms".
  - 330 / 330 must-fail twins rejected.
- **The lever sweep and refinement were re-run under this gate** (job 68229067, 83 + 21 arms, 16 sizes, buses `f5` /
  `e5`). The verdicts of the 22-row lever table above reproduce: same verdict per band, ranges within about 0.03. The
  differences:
  - 4 KiB lamport units at M: 4 WIN / 2 LOSS (was 6 / 0).
  - The fixed counted publish (line) now wins in all three bands: 0.816-0.877 / 0.895-0.953 / 0.987-0.988, still
    behind lamport.
  - 4 waves win at 80 KiB (adopted, above).
- **IR identity.** The sweep ran code `4eacb0fd6eb6`. Its 1339 lever / base / twin / selection builds are
  IR-identical on the final code (`ir_levers.py`: 1339 common builds, 0 differ). The final selection's builds are among
  them, under the same names.

### Issue 2: the suite's `lm_bitwise` was only partly discriminating

- **Seeds and launches.** `lm_bitwise` and `lm_mutants` run the non-contracting chain. Every build gets its own seed
  (from its name), and every launch of a multi-launch run gets a fresh input checked against its own reference. So
  neither the previous build's nor the previous launch's leftovers can match.
- **New `lm_bitwise` case.** It now also runs the exact timed layer-shaped selection builds (`lm_select(layer=True)`,
  header off) at 96 collectives x 2 launches, 10 sizes, at TP2.
- **New `lm_mutants` cases: stale twins right after the real build, same seed, no reset.**
  - Cases: pull `slot` (selection build at 2.5 MiB and a 2 KiB-unit build) and lamport `noreset` (one- and two-shot),
    each at 3 and 96 collectives.
  - All 8 are rejected on every run.
  - The same runs with W3's contracting chain are logged as information: 3 of the 8 (pull `slot`) pass undetected
    there, the case the verifier found.

Final code:
- 20 / 20 at TP4 (68229490, buses `05` / `15` / `85` / `95`);
- 20 / 20 at TP2 and on a permuted TP2: 68229490 (n01-09 `05` / `15`), 68229464 (n01-09), 68229473 (n07-17
  `05` / `65`).

### Issue 3: 5 MiB is a tie in substance

- `ratios_fix.py` reports the strict comparison next to the 9.1 verdict:
  - against the per-round best of aiter's three captures;
  - with the launch counted;
  - with the count of rounds below 1 and the worst round.
- Result: the parity verdict above.
- The search for headroom at 5 MiB used every pull and lamport variant of the sweep and refinement, ranked against the
  per-round best capture:
  - 4 KiB pull on 48-64 CTAs, either sync: 0.993-1.000;
  - 40 or 128 CTAs: 0.998-1.003;
  - 32 CTAs: 1.005-1.008;
  - 2 KiB units on 48-64 CTAs (refinement): 1.013-1.019;
  - the default 160-CTA grid: 1.014-1.015;
  - 16 waves: 0.999-1.002.
- **Arena memory type** (68229576, same pair as the misses): fine-grained ties uncached for the selection (0.997-0.999).
  Coarse-grained loses at 2.5 MiB (1.029). Uncached stays.
- Nothing left in the matched-precision design space moves the link-bound 5 MiB point. bf16 partials of f32 data
  (0.54-0.60 of the f32 wire) are a different precision class and are not counted.

### Issue 4: confirmation on the other rocke node

The bad n07-17 GPU (bus `e5`) is avoided by allocating 3 GPUs and letting `pick_gpus.py` choose two healthy idle ones.
It skips `e5` on n07-17 and any GPU with memory in use. Runs on n07-17:
- **68229249**, intermediate selection, all 16 sizes: every size a WIN under 9.1 and under the strict rule.
- **68229473**, final selection, all 16 sizes: the table above. Every size a WIN under both rules, 5 MiB included.

Both jobs also ran the TP2 and permuted TP2 suites (20 / 20).

### Issue 5: the 16-size confirmation ran earlier code

All 16 sizes were re-measured on the final code: 68229464 + 68229530 (n01-09) and 68229473 (n07-17). The selection
was declared in `layout.py` before those jobs.

### Issue 6: clocks and seeds

- **Clocks.**
  - Every job runs a clock / power sampler of every allocated GPU (`clocks.log`).
  - `bench_ccl_lm` records each round's UTC span (`round_t`). `ratios_fix.py --clocks` drops the rounds within 1 s of a
    sample whose gfx clock is below 0.97 of that GPU's median active clock. When a size's rounds all fall in one sample
    window, the size is kept unfiltered and flagged.
  - The active gfx clock medians are 0.997-1.002 of the maximum.
  - Dips to 0.90-0.97 occur in light-load phases; the final tables are after the filter.
  - Unfiltered, every verdict is the same except one cell: 32 x 10240 f32 on n01-09 is noise under 9.1 there, because
    a low-clock round widened aiter's own A/A band. Its median is 0.939.
- **Seeds.** The arm-order shuffle seed is fresh per run (`os.urandom`, the same on every rank). It is recorded in
  every row (`seed`) and mixed into each round's order.

### Issue 7: PLAN decision 5 (DP2 arm)

- **Why there is no DP2 arm.** This is a collective microbenchmark. A DP2 arm (two independent TP1 replicas) has no
  all-reduce, so there is nothing to compare at this level.
- **What the result does and does not cover.** The result says how fast the TP2 all-reduce of a layer is against
  aiter on the same two GPUs. It says nothing about TP2 against DP2 per serving cell.
- **Where that comparison belongs.** The per-cell TP2-vs-DP2 decision belongs to the end-to-end cells (L0c-DP2 lane,
  DESIGN-LB), where this all-reduce is one input to the TP2 arm.

### Issue 8: the missed word-sync lever

Capped-grid pull with the one-word grid sync is now measured: arms `pull_os_u{2,4}_g{32..128}`, job 68229067, word
against stripe at the same cap.

| comparison | S (up to 320 KiB) | M (640 KiB-1.25 MiB) | L (2.5-5 MiB) |
|---|---|---|---|
| 2 KiB units, 32 CTAs | 0.974-0.999 (5/0/2) | 0.992-1.008 (1/1/4) | 0.998-1.005 (0/0/3) |
| 2 KiB units, 64 CTAs | 0.974-1.002 (5/0/2) | 0.992-1.003 (0/0/6) | 0.998-1.003 (0/0/3) |
| 4 KiB units, 48 CTAs | 0.989-1.012 (2/1/4) | 0.990-0.999 (1/0/5) | 0.998-1.002 (0/0/3) |

- **Same cap.** The sync choice is noise at M / L. It wins at S, where lamport is selected anyway.
- **Different caps.** The word sync with 64 CTAs beats the stripe sync with 48 CTAs at 2.5 MiB (above), which is
  adopted.
- **Default grid.** On the default grid the word sync wins at L, as the verifier saw: 0.969-0.997, WIN in 5 of 6 L
  cells across the two runs.
- **Caps (word, 4 KiB units).** 40 CTAs lose to 48 at L, and 96 / 128 lose to 64. 56 ties 48.

### Issue 9: the contract test

- **New gate.** `contract_ccl.py` applies the contract test's three checks to everything under `Q/ccl`:
  - builders take `(spec, arch)` only;
  - the spec is the first parameter;
  - every spec is listed with its frozen required fields.
- **Results.**
  - 0 CCL violations; 4 builders, 5 specs.
  - Its must-fail mode (dropping the `AllReduceSpec` entry) exits 1.
- **Job wiring.** `job.sh cpu` now counts the gate's exit status and checks that the mutant fails (68229256,
  68229490).
- **Still failing (3 tests).** The full test still fails because of code outside CCL, which this lane does not own:
  - 23 builders off the `(spec, arch)` contract: `qwen38_mono` framework, stage_moe*, stage_b3, stage_fn, moe*,
    toy_chain, tune, probes p2 / p4 / p9, `mx_decode_gemv`;
  - 16 unlisted specs.
  The list is printed by the gate in every cpu step. Fixing them means editing other lanes' builders.

### New findings in the fix round

- **Rare `PEER_ABORT` of W3's pair form at 2.5 MiB in the interleaved bench.**
  - Seen twice: `pair_os_st` in 68224788 and `pair_os` in 68229224. Both at 128 x 10240 bf16 on n01-09 buses `e5` /
    `95`.
  - The containment worked: the arm was dropped, the group reset, and every later arm was bitwise.
  - Only rank 0's record (`PEER_ABORT`) was kept, so the origin is unknown.
  - `pair_stress.py` (68229365, buses `15` / `95`) did not reproduce it: 3600 launches in 4 phases, 0 errors, 0
    numerics failures. The phases:
    - shuffled pair / line / lamport / pull builds;
    - the same with the exchange area zeroed at every layout change;
    - pair alone;
    - mixed plus an aiter graph.
  - The final jobs' 30-round all-arm exposure steps at that size (106 arms, buses `05` / `15` and `05` / `65`) also
    saw none.
  - `bench_ccl_lm` now keeps every rank's error record and the preceding arm for a dropped arm (`drop_records`), so the
    next occurrence is classified.
  - Open. It does not touch the selection: it never selects the pair form at these sizes.
- **aiter's `hipIpcOpenMemHandle ... invalid argument` inside its registration is not tied to the capture count or to
  5 MiB.** It hit at the 7th size (68229224) and at the 4th size (68229464). The lost sizes were re-run in new
  processes (68229365, 68229530).
- **Job-level spread of 1.5-3 %** for the in-kernel exchanges on one GPU pair: above, and alola rule 167.

### Byte identity and tests (final code)

- The LLVM IR of all 2284 earlier CCL builds is unchanged (`ir_snapshot.py` against `ir_base.json`, 2284 / 2284).
- Every fix-round device-code change is a selection of existing knobs. No device code file changed.
- The host changes:
  - `ref.chain_scales(growth)` and `ref.LM_GROWTH`;
  - `Rank.chain_growth` (default 1.0: W3's chain unchanged);
  - `bench_ccl_lm`: growth, must-fail twins, seed, round times, drop records, the word-sync and cap arms;
  - the `bench_ccl` CCL-LM tests;
  - `layout.lm_select(layer=True)` and the three band changes;
  - the CPU tests.
- CPU tests: `test_qwen38_ccl.py` + `test_qwen38_ccl_lm.py` 245 passed locally, and the job.sh cpu step passed in
  68229490.

### Fix-round jobs

All container jobs are 2 GPUs (n07-17: 3 allocated, 2 used); the venv jobs are 4 GPUs. Everything was run at
20 rounds unless noted.

| job | mode, node, buses | what |
|---|---|---|
| 68228965 | container, n01-09 | lever sweep; cancelled to add round timestamps |
| 68229067 | container, n01-09 `f5` / `e5` | lever sweep (12 rounds) + refinement (20 rounds), strong gate; TP2 suite |
| 68229224 | container, n01-09 `e5` / `95` | intermediate selection, 15 sizes (aiter lost the 16th); TP2 + permuted TP2 suites |
| 68229249 | container, n07-17 `05` / `65` | intermediate selection, 16 sizes; TP2 + permuted TP2 suites |
| 68229256 | venv, n01-09 | cpu step (contract gate); TP4, TP2 and permuted TP2 suites, intermediate code |
| 68229365 | container, n01-09 `15` / `95` | 5 MiB and 2.5 MiB f32 redo; `pair_stress.py` |
| 68229464 | container, n01-09 `05` / `15` | final selection, 12 sizes (aiter lost 4); exposure step (30 rounds); TP2 + permuted suites |
| 68229473 | container, n07-17 `05` / `65` | final selection, 16 sizes; exposure step; TP2 + permuted suites |
| 68229490 | venv, n01-09 | final cpu step; TP4, TP2 and permuted TP2 suites |
| 68229530 | container, n01-09 `05` / `15` | final selection, the 4 f32 sizes lost in 68229464 |
| 68229576 | container, n01-09 `05` / `15` | arena memory types (uncached / fine / coarse), 12 rounds |
| 68229681 | container, n01-09 `05` / `15` | final selection at 2.5 MiB bf16 and 5 MiB f32, separate processes |
