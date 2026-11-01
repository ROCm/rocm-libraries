# gfx950 run-side measurement after the client and harness changes (P3)

Run-side measurement at the end of P2 of the implementation plan (`timing_runs/DECISIONS.md`,
P3), compared with the P0.2 baseline (`baseline.md`). It covers the client changes P1.1–P1.29 and
the Python/YAML/harness changes P2.1–P2.7.

## What was measured

| Item | Value |
|---|---|
| Commit | `3b4ec8e125d` (end of P2; detached worktree) |
| Client | `tensilelite-client` built from that commit, CMake cache identical to the P0.2 client's (paths normalised): Release, `GPU_TARGETS=gfx950`, `/usr/bin/amdclang++` (ROCm 7.2.1), msgpack only, `TENSILELITE_BUILD_TESTING=OFF`. sha256 `15f20fa8a2eb…`. |
| Baseline client | P0.2's (`2083e6254bf`, sha256 `4b2769f7f9a2…`), its worktree and artifacts, for the same-session control runs |
| Artifacts | Split build phase at `3b4ec8e125d` into a new directory, separate caches |
| Python | venv Python 3.12.3; TensileLite Python sources from each worktree; venv `rocisa` (unchanged) |
| Host | `smci355-1b112-b13-15` (as P0.2): 2× EPYC 9575F, 8× MI355X (gfx950), ROCm 7.2.1 |
| Cores / threads | `taskset -c 0-24`, `OMP_NUM_THREADS=25`, `OPENBLAS_NUM_THREADS=25` (as P0.2) |
| GPU | Per run, see below: the P0.2 GPU (`ROCR_VISIBLE_DEVICES=0`, PCI `75:00.0`) when idle, otherwise another idle MI355X (`ROCR` 5 = PCI `85:00.0` or 6 = `e5:00.0`, NUMA node 1) |
| Config list | The 185 configs that passed the split run phase: P0.2's 184 plus the new `gemm/gfx950/prediction_bf16` (P2.5) |

Committed results:

- `after_run/AGGREGATE_analysis.txt`: per-config run aggregate, P3 client, 185 logs.
- `after_run/common_test_outcomes.txt`: outcome of every test in the split run phase.

Raw logs (not committed, as for P0.2) are in `/home/alvasile/timing_runs_raw/P3/`, gzipped:
the per-config logs of both clients, the split logs, the P0.2-client control aggregate and the
monitoring logs.

## Conditions: why the comparison is same-session

The host was shared throughout (2026-10-31 18:40 to 2026-11-01 14:30 UTC):

- Another user's corpus run again pinned jobs to cores 0–8, 10–47 and 9 (the job described in
  `baseline.md`), from 19:25 to about 03:00. It was gone during the per-config pass and the two
  clean split pairs (no CPU growth of it in any of those runs).
- A root vLLM server held the measurement GPU (and rocm-smi GPUs 0–3, i.e. `ROCR` 0–3) at 100% /
  279 GB for hours at a time; other root jobs used the remaining GPUs in turns.
- `/` filled up twice (0 bytes free), so pytest temp dirs, caches and output were moved to
  `/dev/shm` (RAM); this applied to both clients alike.

An uncontended 1-hour window for a P0.2-style serial pass did not occur: a clean pass started at
03:43 got through 36 of 185 configs before the measurement GPU was taken again. So, as decided
during the run, the per-config pass was measured **interleaved**: for each config, the P0.2
client (with P0.2's worktree and artifacts) and the P3 client ran back to back on the same idle
GPU, alternating which went first, with the same script and settings as P0.2
(`run_per_config.sh`, serial, `TimingInstrumentation=True`, 25 pinned cores). A GPU was idle when
it showed < 5% use and < 2 GB VRAM twice; if another user's process took more than 1 GB on it
during a pair, the pair was re-run (up to three tries). So each config's two numbers share their
conditions, and the P0.2-client sum is a same-session control. Absolute numbers are not
comparable with the P0.2 pass on their own (see "Control").

- GPUs used per pair: `ROCR` 5 for 130 pairs, 0 for 46, 6 for 8.
- Other users' CPU on 0–24: no CPU growth of pinned corpus jobs during any run; unpinned root
  processes (vLLM workers, downloads) were parked on idle cores of 0–24 at times, as in P0.2.
- Two pairs, `gemm/gfx950/subtile_mxfp4` and `subtile_mxfp8`, had another process start on the
  GPU during every one of three tries; they are kept (both clients were exposed) and shown
  separately below.
- The run script clears the caches at the start of each call. One call per config (to re-run single
  pairs) therefore clears the helper-kernel cache before every config, while P0.2 cleared it once.
  That costs both clients the same ~21 s of helper-kernel compiles (see "Control").

## Correctness: common-test pass/fail set

Split run phase (`--use-cache`, `-n 4`) with the P3 client and artifacts: 568 tests, **195
passed, 2 failed, 371 skipped**. Compared with `baseline_run/common_test_outcomes.txt`, the only
difference is the new config:

- `test_config_run[gemm/gfx950/prediction_bf16.yaml]` PASSED (new in P2.5).

Three P3 split runs (2026-10-31 23:44 on `ROCR` 0, 2026-11-01 12:20 and 13:59 on `ROCR` 5) gave
identical outcome lists. The split build phase also differs from P0.2's only by that config:
196 passed, 2 failed (the known `custom_rr_f6`/`custom_rr_mx` build failures), 370 skipped.

Changes that were intended and do not change an outcome:

- P2.6 removed the `LibraryLogic:`/`LibraryClient:` sections of the 14 `ductile/*`, the 10 gfx950
  mxfp8 configs and the 12 `custom/*` configs. Of those, the ones that run on gfx950 all pass:
  the 8 `ductile/gfx950/*`, the 10 mxfp8 configs and 7 `custom/*` (`aiter_bf16`, `aiter_f4`,
  `rocblas_gemv_bf16_tn_m1/m2/m4`, `triton_f4`, `wave_bf16`); their run phase no longer writes
  logic files or runs the library client. The 6 `ductile/gfx1250/*` and `custom_ck`,
  `custom_rr`, `custom_tensile_sk` are skipped on gfx950, and `custom_rr_f6`/`custom_rr_mx` are
  the 2 known build-phase failures, all as in P0.2.
- P1.22 checks the Synchronizer after every run iteration. `streamk/gfx950/sk_sgemm_debug_modes`
  omits Stream-K reduction steps on purpose, so it sets `CheckSynchronizer: False`; it passes.
- P2.3 runs the client at `ClientLogLevel=0` and P2.4 changed `gemm/gfx950/xfp32`'s warmups:
  every gfx950 config passes as before. P2.7 set `SyncsPerBenchmark: 0` in `custom/*`: the same
  7 gfx950 custom configs pass; the other 5 are skipped or fail as in P0.2 (above).

The P0.2 client's split runs in this session: identical to the baseline on a free GPU
(2026-11-01 12:36 and 13:17). On 2026-10-31 22:31, while vLLM shared the measurement GPU,
`streamk/gfx950/sk_sgemm_hybrid_pap` **failed validation** with the P0.2 client (1.39 M of
8.4 M values wrong, 67 s instead of 2–4 s). It passed in every other run of either client. This
is a pre-existing sensitivity of that Stream-K config to sharing the GPU with another process,
not a P3 change; it is listed under "Findings".

## Uninstrumented split run phase (`-n 4`)

Same command as `split/run_split.sh`'s run phase, both clients back to back on one GPU:

| Pair | GPU | Conditions | P0.2 client | P3 client | Δ |
|---|---|---|---:|---:|---:|
| A–B, 2026-10-31 22:31 / 23:44 | `ROCR` 0 | corpus job on 0–24 throughout; vLLM on the GPU for the last ~28 min of the P0.2 run; during the P3 run one 30-s monitor sample showed another process with 13.8 GB on the GPU | 3,115 s | 966 s | not comparable |
| B–A, 2026-11-01 12:20 / 12:36 | `ROCR` 5 | clean (no corpus CPU, GPU free) | 2,445 s | 941 s | **−61.5%** |
| A–B, 2026-11-01 13:17 / 13:59 | `ROCR` 5 | clean | 2,499 s | 943 s | **−62.3%** |

The P0.2 split run (2,950 s, contended) in `baseline.md` is an upper bound; the two clean
same-session pairs (ABBA order) are the comparison: P0.2 client 2,445–2,499 s, P3 client
941–943 s, **about −62%**. The run-phase critical path changed (B–A pair): the slowest P0.2
tests were `subtile_mxfp4` (561 s), `spmm_f16_sb` (377 s), `sk_dynamic_mxfp4` (336 s), `spmm_bf16`
(332 s) and `sk_I8gemm_quick` (320 s); with P3 they are `subtile_mxfp4` (163 s),
`custom_mainloop_scheduling` (159 s), `dtl` (128 s), `plr_zero` (105 s), `bufferLoadStore0`
(101 s). The call-phase per-test sum fell from 8,929 s to 3,389 s (−62%). (With `-n 4`, four
25-thread OpenMP clients share 25 cores, so per-test times are larger than in the serial pass.)

## Run side (per-config, instrumented)

Seconds from "TIMING BY PURPOSE", summed over the 184 configs common to both (P3's
`prediction_bf16` adds 3.9 s and is in the committed aggregate only). "P0.2" is the committed
baseline pass, re-aggregated with the analyzer at `3b4ec8e125d` (byte-identical to the committed
`baseline_run/AGGREGATE_analysis.txt`, so analyzer changes contribute nothing). "Control" is the
P0.2 client in the interleaved pass.

| Category | P0.2 | Control | P3 | Δ vs P0.2 | Δ vs control |
|---|---:|---:|---:|---:|---:|
| TOTAL RUNTIME | 2,401.1 | 2,655.0 | 1,024.5 | −57.3% | −61.4% |
| RUN: benchmark client | 2,284.9 | 2,519.0 | 887.7 | −61.1% | −64.8% |
| Reference GEMM (CPU) | 1,044.4 | 1,138.2 | 227.6 | −78.2% | −80.0% |
| … `solve_cpu_slow` | 689.0 | 740.1 | 0.6 | −99.9% | −99.9% |
| … `solve_cpu_fast` | 198.9 | 229.6 | 226.9 | +14.1% | −1.2% |
| … per-solution reference re-runs | 156.3 | 168.3 | – | −100% | −100% |
| Input data preparation | 500.7 | 582.7 | 122.5 | −75.5% | −79.0% |
| … `gpu_input_reset` | 298.8 | 328.7 | 46.1 | −84.6% | −86.0% |
| … `cpu_data_init` | 105.0 | 151.1 | 52.4 | −50.1% | −65.3% |
| … MX per-solution input (`DataInitialization`) | 54.9 | 61.4 | 3.1 | −94.3% | −94.9% |
| … `gpu_input_preparation` | 41.3 | 40.7 | 20.4 | −50.6% | −49.9% |
| Validation | 329.1 | 341.6 | 112.5 | −65.8% | −67.1% |
| … Synchronizer check | 165.4 | 169.9 | 11.8 | −92.9% | −93.1% |
| … `ReferenceValidator` | 163.3 | 171.3 | 100.3 | −38.5% | −41.4% |
| Client launch (startup, setup & teardown) | 289.8 | 316.3 | 284.4 | −1.8% | −10.1% |
| Per-solution bookkeeping & reporting | 54.8 | 57.4 | 13.7 | −75.0% | −76.2% |
| Kernel launch & execution | 35.7 | 39.3 | 126.0 | +252.7% | +220.2% |
| … `warmup_first_sync` (new) | – | – | 93.9 | | |
| Library client (Python) | 24.6 | 37.3 | – | −100% | −100% |
| Library logic (Python) | 5.8 | 6.3 | 1.0 | −81.9% | −83.4% |
| BUILD: solution generation & compile | 63.1 | 85.4 | 85.6 | +35.8% | +0.3% |
| MEASUREMENT OVERHEAD (instrumentation) | 52.8 | 50.2 | 50.1 | −5.1% | −0.2% |
| Sum of per-config pytest wall times | 2,942 | 3,212 | 1,571 | −46.6% | −51.1% |

Over the 182 pairs without GPU interference the pytest sums are 2,917 s (control) and 1,394 s
(P3), −52.2%. The two GPU-shared pairs: `subtile_mxfp8` 203 → 109 s, `subtile_mxfp4` 92 → 68 s.

Per-config pytest seconds (P0.2 / control / P3) of the largest configs: `sk_I8gemm_quick`
311 / 323 / 6, `sk_f8gemm_quick` 262 / 295 / 12, `spmm_f16_sb` 124 / 133 / 44, `plr_zero`
94 / 102 / 52, `dtl` 82 / 90 / 59, `fp16_tn` 87 / 87 / 43, `bufferLoadStore0` 81 / 86 / 60,
`mxfp4_mxfp4_bf16_tn_act` 31 / 38 / 4, `xfp32` 62 / 66 / 24. No config got slower by more
than 2 s against its pair (ten configs are 1–2 s slower, all with 3–16 s runs).

What the numbers show:

- **The work is the same.** The main call counts are identical to P0.2 (28,596
  `cpu_reference_gemm`, 2,355,016 solutions, 2,106,835 `gpu_input_reset`, 855 client starts).
  Counts changed only where an item changed them on purpose: 10 library-client invocations fewer
  (P2.6), 592 instead of 1,947 slow-path reference calls (the rest moved to the fast path,
  P1.2–P1.11), and the renamed/new timers of P1.19–P1.23.
- **CPU reference (P1.1–P1.11):** the slow path went from 740 s (control; 689 s in P0.2) to
  0.6 s. The fast path took over those problems at about no net cost: 1,355 more calls (26,656 →
  28,011) in 226.9 s against the control's 229.6 s (−1.2%; +28 s against the P0.2 pass, which ran
  faster than the control). The per-solution re-runs of the reference (168 s in the control,
  156 s in P0.2) are gone: they were caused by re-uploaded inputs, which P1.16–P1.19 skip.
- **Input preparation (P1.13–P1.19):** `gpu_input_reset_copy_valid_to_gpu` 175 s → 0.1 s,
  batched-pointer uploads 93 → 19 s, MX regeneration 50 → 1.5 s, `cpu_data_init` −50%.
- **Validation (P1.20–P1.23):** the Synchronizer check is a GPU OR-reduction now (165 → 12 s);
  `validate_gpu_readback` 94 → 36 s.
- **Bookkeeping (P1.24–P1.29, P2.3):** −75%.
- **Client launch is unchanged** (`hip_initialization` 128 s, `process_exit` 68 s for 855
  client starts): no item of the plan targeted it. It is now the largest category (28% of the
  P3 total).

## Moved as expected, or for a known reason

- **Kernel launch & execution +87 s vs the control (+90 s vs P0.2)** is attribution, as the
  plan expected: P1.20 times the wait for the first warmup's completion in its own timer,
  `warmup_first_sync` (93.9 s). Before,
  that GPU time was hidden in whichever blocking call came next (`validate_gpu_readback`, the
  Synchronizer readback, input uploads), which is part of why those dropped by more than their
  own speedups.
- **BUILD +22 s** against P0.2, but equal for both clients (85.4 / 85.6 s): the interleaved pass
  clears the helper-kernel cache before every config (see "Conditions"); P0.2 cleared it once.
  `python_kernel_compilation` 43.5 → 65 s in both.
- **Library client −24.6 s, library logic −4.8 s** (P2.6), and `flush_timing_buffer` −10 calls.

## Control: P0.2 client in this session versus the P0.2 pass

The same client and artifacts took 10.6% longer in total than in the P0.2 pass (2,655 vs
2,401 s; pytest sums 3,212 vs 2,942 s). P0.2's clean-repeat spread was about 1%, so the session
was slower. The difference is:

- BUILD +22 s: the cache clearing above.
- Library client +12.7 s (`python_library_client_create_library`, Python).
- CPU reference +94 s (+9%), `cpu_data_init` +46 s, `process_exit` +18 s, input reset +30 s.
  Most of `cpu_data_init` is one config, `sk_bgemm_div` (one 2.35·10⁹-element tensor): 32.7 s
  instead of 6.1 s (P3 client: 3.7 s). These are host-memory-heavy phases. During the pass
  `/dev/shm` held 1.2–1.3 TB of other users' data in RAM, and 130 pairs ran on a GPU on the
  other NUMA node. Either may explain it, but neither was isolated.

So the deltas against the control are the like-for-like ones. The deltas against P0.2 are
smaller, because the control ran slower than P0.2 did.

## P1.12 replay (topic 1, slow-path reasons)

After P1.2–P1.11 (`44cd1a416fe`), the 59 slow-path YAMLs of the investigation were replayed with
an uncommitted client that logs each reference call's path and reason (rule 8). **All 59 pass.**
The reference calls are the same as in the investigation's replay (2,517). Slow-path calls went
from 1,966 (546.6 s) to 593 (0.53 s), and fast-path calls from 551 to 1,924. The remaining slow
calls are the expected ones: `NumElementsToValidate: 128` (sparse validation, 569 calls in 20
YAMLs) and `scaleCD` (24 calls, `gemm/gfx950/f8f16mix_f8s`). No other reason occurred. The
per-config pass here is consistent with it: 592 slow-path calls, 0.6 s.

## Findings that bear on the run side (not changed by the plan)

Decisions needed:

- **P2.5, option A (user decision).** With `--best-solution`, the library client silently falls back
  to library solution 0 when Origami selection returns nothing, and exits 0. A client option that
  fails on an empty selection (written only for the `LibraryClient` step) would let the new
  `prediction_bf16` config detect selection regressions. Not implemented; the plan said to stop
  and ask.
- **P1.28, `sync-after-warmups` default (user decision).** The client defaults
  `sync-after-warmups` to true, and `ClientWriter.py` never writes it. So P1.28's warmup-event
  drop never triggers in Tensile-generated runs. Measured on `dtl` with `sync-after-warmups=False`:
  −6.7 s (−13%) uninstrumented wall per pass, and 18,872 → 0 `hipEvent*` calls on one ini. Options:
  `ClientWriter` writes `sync-after-warmups=False` for correctness-only runs, or the client stops
  treating that sync as an event reader when nothing is timed.

Pre-existing issues (candidates for a deferred list):

- `sk_sgemm_hybrid_pap` failed validation once with the P0.2 client while another process
  used the GPU (above).
- `cpu-gemm-driver --bias` with `--batchCount > 1` fails on both reference paths: the driver
  registers the bias with batch stride m but allocates m elements, so batches > 0 read past the
  buffer (test infrastructure only; no CTest combines them; P1.9).
- Client launch (`hip_initialization`, `process_exit`, `factory_hip_get_device_properties`) is
  now 28% of the instrumented run total and was not in scope.
- GuardPageAll configs (`bufferLoadStore0`) re-upload batched-pointer arrays every run, because
  the Front/Back start pointers differ; one cached array per bounds-check mode would recover
  most of the remaining `gpu_input_reset_batched_inputs` (P1.18).
- `f8f16mix_f8s` and 3 calls of `use_beta_false` stay on the slow path (`scaleCD`, P1.7/P1.12),
  and the slow path applies ScaleAB differently from the fast path when a Half operand is
  stored wider than its F8 compute type (P1.7).
- The slow path's MX K tail is wrong for K not a multiple of the MX block; P1.11 moved those
  problems to the fast path, and the slow path is unchanged (P1.11).
- Slow-path bias-gradient error handling (missing `throw`, double `free`) for batched source
  shapes (P1.10).
- `generateMXInput`: the K tail in the re-arranged orientations is not deterministic, and neither
  is `norm_dist` between calls (P1.13). `rocm_random` shares one `std::mt19937` across OpenMP
  threads (data race, P1.15). Trig init takes an `int` index, which wraps above 2³¹ elements
  (`sk_bgemm_div`, P1.14).
- The MX GFX950 scale swizzle with a padded leading dimension (P1.19). MX configs crash in NaN
  and GuardPage modes (P1.19). Grouped GEMM with bounds checking is broken (P1.17). None of these
  is used by a common config.
- Client: `selection-only=True` segfaults (P1.25); `max-enqueues-per-sync=0` hangs (P1.27); the
  device-lost path exits via SIGABRT (P1.20).
- `ductile/gfx950/f8_tn_mx_subtile` sets `NumElementsToValidate: 0` but still validates 128
  elements (P1.12).

## Reproducing

```bash
git worktree add --detach /tmp/tl_plan/P3/wt 3b4ec8e125d
source /home/alvasile/venv/bin/activate
cmake --preset tensilelite -S /tmp/tl_plan/P3/wt/projects/hipblaslt -B /tmp/tl_plan/P3/build_client \
  -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx950 \
  -DCMAKE_CXX_COMPILER=/usr/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DHIPBLASLT_ENABLE_YAML=OFF -DTENSILELITE_ENABLE_CLIENT=ON -DTENSILELITE_BUILD_TESTING=OFF \
  -DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
taskset -c 128-191 cmake --build /tmp/tl_plan/P3/build_client -j64 --target tensilelite-client
export TL=/tmp/tl_plan/P3/wt/projects/hipblaslt/tensilelite
export CLIENT=/tmp/tl_plan/P3/build_client/tensilelite/client/tensilelite-client
export ART=/tmp/tl_plan/P3/artifacts TENSILE_CACHE=<cache>/tensile TENSILELITE_CACHE=<cache>/tensilelite
OUT=<dir> bash $TL/timing_runs/full_runs_gfx950/split/run_split.sh            # artifacts + split run
OUT=<dir> KEEP_CACHE=1 bash $TL/timing_runs/full_runs_gfx950/per_config/run_per_config.sh <configs>
```

The interleaved pass ran `run_per_config.sh` once per config and client, with
`ROCR_VISIBLE_DEVICES` set to the pair's GPU and `TMPDIR`, caches and `OUT` under `/dev/shm`
(driver scripts and the per-config script copy are kept with the raw logs in
`/home/alvasile/timing_runs_raw/P3/`, see its `README.txt`).
