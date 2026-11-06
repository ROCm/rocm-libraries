# gfx950 timing baseline at the starting commit (P0.2)

Re-measurement of the gfx950 common tests at the starting commit of the implementation plan
(`timing_runs/DECISIONS.md`, P0.2). The earlier committed baselines (`per_config_rerun_2`,
`per_config_build`) were measured with a client built at `4c6426cea6`. Every later measurement
(P3 `after_run/`, P6 `after_build/`) is compared with the numbers here, not with those.

## What was measured

| Item | Value |
|---|---|
| Commit | `2083e6254bf` (detached worktree). Code-identical to the starting commit `55e29436e9a`: the only difference is P0.0's measurement-script parametrization. Client, library and Python sources are also identical to `4c6426cea6`; only `scripts/{analyze,aggregate}_timing.py` and their tests changed since then. |
| Client | `tensilelite-client` built from that commit, CMake cache identical to `/tmp/tl_client_msgpack` (the `4c6426cea6` baseline client): Release, `GPU_TARGETS=gfx950`, `/usr/bin/amdclang++` (AMD clang 22.0.0git, ROCm 7.2.1), msgpack only (`HIPBLASLT_ENABLE_YAML=OFF`), `TENSILELITE_BUILD_TESTING=OFF`. sha256 `4b2769f7f9a2…`. |
| Python | venv Python 3.12.3; TensileLite Python sources from the worktree; `rocisa` is the venv's installed build (2026-10-19, unchanged). |
| Host | `smci355-1b112-b13-15`: 2× AMD EPYC 9575F (64 cores each, SMT 2), 8× AMD Instinct MI355X (gfx950), Linux 6.8.0-38, ROCm 7.2.1, amdgpu 6.16.13. Shared by ~45 users. |
| Cores / threads | `taskset -c 0-24` (25 physical cores, NUMA node 0), `OMP_NUM_THREADS=25`, `OPENBLAS_NUM_THREADS=25` |
| GPU | `ROCR_VISIBLE_DEVICES=0` = KFD node 2 = PCI `0000:75:00.0`, NUMA node 0. This is the same physical GPU the earlier baselines used. Note that `rocm-smi`/`amd-smi` call it GPU **3**, because they order by PCI bus and ROCR orders by KFD node. |
| Scripts | `split/run_split.sh`, `per_config/run_per_config.sh` (`KEEP_CACHE=1`, as `per_config_rerun_2`), `per_config/run_per_config_build.sh` (cold cache per config), all with caches and artifacts redirected through `measure_env.sh` |
| Config list | The 184 configs that passed the split run phase. Identical to `per_config_rerun_2`'s list. |

Committed results:

- `baseline_run/AGGREGATE_analysis.txt`: per-config run aggregate (184 logs).
- `baseline_build/AGGREGATE_analysis.txt`: per-config build aggregate (184 logs).
- `baseline_run/common_test_outcomes.txt`: correctness baseline (outcome of every test in the split
  run phase).

The raw per-config logs are not committed, as for the earlier baselines. That is
`<config>/<config>.log.gz` and `<config>_build.log.gz`, plus each config's `_analysis.txt` and
`times.txt`. A durable copy (757 MB, checked identical by md5) is in
`/home/alvasile/timing_runs_raw/P0.2/`:

- `baseline_run/`, `baseline_build/`: the committed passes.
- `noise_run_1/`, `noise_run_2_contended/`: the repeats used for the noise band.
- `split/`: split phase logs, junit XML and outcomes.
- The config lists, plus `snapshots.txt`, `contention.log`, `corpus.log` and `gpu.log`.

The working copy is in `/tmp/tl_plan/P0.2/`. **P3/P6 must re-aggregate these raw logs with the
`analyze_timing.py` of the commit they compare at**, not compare against the committed
`AGGREGATE_analysis.txt` as is: P1/P4 add timers and change the hierarchy. The command is
`python scripts/aggregate_timing.py <dir>/*/*.log.gz`, or `<dir>/*/*_build.log.gz` for the build
pass. The earlier baselines' raw logs re-aggregate byte-identically with the analyzer at this
commit, so analyzer changes do not contribute to any difference below.

`/tmp` is cleaned of files older than 30 days on this host
(`/usr/lib/tmpfiles.d/tmp.conf`: `D /tmp 1777 root root 30d`). The P0.2 worktree
(`/tmp/tl_plan/P0.2/wt`), client (`/tmp/tl_plan/P0.2/build_client`) and artifacts
(`/tmp/tl_plan/P0.2/artifacts`) may therefore disappear before P3/P6. If they are gone, rebuild
them from `2083e6254bf` with the configure and commands under "Reproducing" (client ~1.5 min,
artifacts = split build phase ~11 min).

## Correctness baseline

Split run phase (`--use-cache`, `-n 4`): 567 tests, **194 passed, 2 failed, 371 skipped**. The
outcome of every test is identical to the 2026-10-20 split run. The split build phase also
matches it: 195 passed, 2 failed, 370 skipped.

The two failures are pre-existing. `custom/custom_rr_f6.yaml` and `custom/custom_rr_mx.yaml` fail
to build (`RuntimeError: No valid solutions found`), so their run test finds no artifact tarball.
Every config in the per-config run and build passes returned 0.

## Run side (per-config, instrumented)

`run_per_config.sh`, serial, `TimingInstrumentation=True`, `KEEP_CACHE=1` (caches cleared once,
as for `per_config_rerun_2`). Measured 2026-10-31 01:40–02:34 UTC with no other users' work
pinned to cores 0–24. In seconds, from `AGGREGATE_analysis.txt` ("TIMING BY PURPOSE"):

| Category | Baseline (this) | `per_config_rerun_2` (`4c6426cea6`) | Δ |
|---|---:|---:|---:|
| TOTAL RUNTIME | 2,401.1 | 2,568.3 | −6.5% |
| RUN: benchmark client | 2,284.9 | 2,330.8 | −2.0% |
| Reference GEMM (CPU) | 1,044.4 | 1,080.5 | −3.3% |
| … `solve_cpu_slow` | 689.0 | 727.8 | −5.3% |
| … `solve_cpu_fast` | 198.9 | 200.9 | −1.0% |
| Input data preparation | 500.7 | 502.9 | −0.4% |
| Validation | 329.1 | 323.9 | +1.6% |
| Client launch (startup, setup & teardown) | 289.8 | 284.7 | +1.8% |
| Per-solution bookkeeping & reporting | 54.8 | 68.3 | −19.9% |
| Kernel launch & execution | 35.7 | 40.9 | −12.7% |
| BUILD: solution generation & compile | 63.1 | 64.0 | −1.5% |
| MEASUREMENT OVERHEAD (instrumentation) | 52.8 | 172.8 | −69.5% |
| … `flush_timing_buffer` | 33.6 | 152.9 | −78.1% |
| Sum of per-config pytest wall times | 2,942 | 3,123 | −5.8% |

- Every timer's call count is identical (for example 28,596 `cpu_reference_gemm`, 2,355,016
  solutions, 2,106,835 `gpu_input_reset`). The work is the same, as expected from identical
  sources.
- Most of the −167 s difference is `flush_timing_buffer`: the instrumentation writing its buffer
  to the log, −119 s. The rest is CPU reference −36 s, bookkeeping −14 s and kernel launch −5 s.
- The CPU reference difference (−3.3%; `solve_cpu_slow` −5.3%) is **unexplained and larger than
  the clean-repeat spread** of that category (0.6%, see "Noise band"). The old pass differs in
  client build (same sources), day and host load. Treat it as a difference between passes, not as
  a code effect.
- Use this baseline for P3, not `per_config_rerun_2`.

## Build side (per-config, cold cache per config)

`run_per_config_build.sh`, measured 2026-10-31 00:26–01:23 UTC:

| Category | Baseline (this) | `per_config_build` (`4c6426cea6`) | Δ |
|---|---:|---:|---:|
| TOTAL RUNTIME | 2,654.8 | 2,577.8 | +3.0% |
| BUILD: solution generation & compile | 2,605.6 | 2,530.1 | +3.0% |
| … `python_kernel_compilation` | 2,260.4 | 2,203.8 | +2.6% |
| … … `python_kernel_build_src_co` | 1,013.4 | 1,007.8 | +0.6% |
| … … `python_kernel_codegen` | 615.5 | 592.6 | +3.9% |
| … … `python_kernel_write_assemble` | 253.8 | 227.9 | +11.4% |
| … `python_solution_generation` | 321.6 | 306.6 | +4.9% |
| Ductile GA search (Python) | 23.7 | 19.7 | +20.2% |
| SHARED: cache & solution files | 23.2 | 23.2 | +0.1% |
| RUN: benchmark client | 14.3 | 13.0 | +10.3% |
| Sum of per-config pytest wall times | 3,361 | 3,273 | +2.7% |

No config differs by more than 15 s from `per_config_build`. Over the 10 most expensive build
configs, the two passes differ by 2.5% in total and by at most 9.4% per config (`plr_zero`).

**The whole build pass (00:26–01:23 UTC) overlapped another user's corpus run.** That run pins
short-lived compile jobs to core 9. `contention.log` shows them on core 9 in 51 of the 55
minutes; pidstat only catches part of them, so the low average it reports understates the load.
The effect on the build is not quantified. It is probably small, since build workers are
independent processes rather than OpenMP threads. Still, the +3.0% against `per_config_build`
(a uniform drift: +88 s summed over 184 configs, at most 7 s per config) cannot be separated from
it. The baseline may be inflated by a few percent. Re-measuring was not possible: no 1-hour clean
window occurred before this was recorded. **P6 must therefore re-run this P0.2 build pass**
(P0.2 worktree/client, `run_per_config_build.sh`, same settings) **back to back with its
`after_build/` pass, under identical conditions, and compare against that re-run.** It should
also re-run the split build phase, for P6's `--build-only` comparison, since the 634 s below was
contended too.

## Split phases (`-n 4`, uninstrumented)

| Phase | Wall time | Exit | Conditions |
|---|---:|---:|---|
| Build (`--build-only`) | 634 s | 1 (the 2 known failures) | core 9 contended throughout |
| Run (`--use-cache`) | 2,950 s | 1 (the 2 known failures) | core 9 contended throughout |

The 2026-10-20 split, measured at an older tree, took 1,183 s (build) and 2,804 s (run). The
run phase here (22:18–23:07 UTC) overlapped another user's job pinned to core 9 for its whole
duration. That job slows the OpenMP CPU reference about 2.5× (see "Noise band"), so **2,950 s is
an upper bound, not a clean uninstrumented baseline**. A clean repeat could not be obtained
(see "Conditions"). **P3 must re-measure this phase with the P0.2 worktree, client and artifacts
back to back with its own, under identical conditions, and compare against that**, not against
2,950 s. Likewise, the split build phase (634 s, also contended throughout) is not a clean
baseline: P6 re-measures it (see "Build side").

## Noise band

The 10 most expensive run configs (by pytest wall time in this baseline) were repeated
(`noise_run_<k>`, same script and settings, `KEEP_CACHE=1`). Samples whose run overlapped other
users' work pinned inside cores 0–24 were **excluded**: see "Conditions" for how this was
detected. Three repeats were run, but only part of them was uncontended. **Clean samples per
config: 3 for `sk_I8gemm_quick` and `sk_f8gemm_quick`**: the baseline pass, repeat 1, and
repeat 2's first two configs. **2 for the other eight configs**: the baseline pass and repeat 1.
All other repeat samples (repeat 2 configs 3–10, all of repeat 3, a restarted repeat 2) were
excluded. A further 2.7 h of waiting for an uncontended window (06:33–09:16 UTC) found none.

Per-config pytest wall time (s), clean samples only. Spread = (max − min) / mean.

| Config | Baseline | Repeat 1 | Repeat 2 | Spread | `per_config_rerun_2` (reference) |
|---|---:|---:|---:|---:|---:|
| streamk/sk_I8gemm_quick | 311 | 313 | 304 | 2.9% | 316 |
| streamk/sk_f8gemm_quick | 262 | 278 | 266 | 6.0% | 305 |
| gemm/gfx950/subtile_mxfp8 | 186 | 180 | – | 3.3% | 182 |
| sparse/gfx950/spmm_f16_sb | 124 | 126 | – | 1.6% | 144 |
| gemm/gfx950/subtile_mxfp4 | 95 | 89 | – | 6.5% | 82 |
| gemm/gfx950/plr_zero | 94 | 96 | – | 2.1% | 107 |
| gemm/fp16_tn | 87 | 87 | – | 0.0% | 103 |
| gemm/gfx950/dtl | 82 | 85 | – | 3.6% | 103 |
| gemm/bufferLoadStore0 | 81 | 83 | – | 2.4% | 99 |
| gemm/gfx950/subtile_mxfp8_correctness_regression_mt320x256_plr1_wgm4 | 71 | 67 | – | 5.8% | 69 |
| Sum | 1,393 | 1,404 | | 0.8% | 1,510 |

Categories summed over the same 10 configs (s), baseline vs repeat 1:

| Category | Baseline | Repeat 1 | Spread |
|---|---:|---:|---:|
| TOTAL RUNTIME | 1,359.6 | 1,369.7 | 0.7% |
| Reference GEMM (CPU) | 787.8 | 792.7 | 0.6% |
| Input data preparation | 229.0 | 226.7 | 1.0% |
| Validation | 179.9 | 181.7 | 1.0% |
| Client launch (startup, setup & teardown) | 82.5 | 82.2 | 0.4% |
| Per-solution bookkeeping & reporting | 28.6 | 30.2 | 5.4% |
| Kernel launch & execution | 21.6 | 24.2 | 11.4% |
| MEASUREMENT OVERHEAD (instrumentation) | 29.0 | 29.4 | 1.4% |

So, on an uncontended host, totals and the large categories repeat to within about 1%. Single
configs repeat to within about 6%. The small categories (bookkeeping, kernel launch) vary by
5–11%. With 2–3 samples this is an indication, not a confidence interval.

Contended samples, for scale (all excluded): with the core-9 job active, `sk_I8gemm_quick` took
1,169 s in the discarded first per-config run pass (23:07–00:26 UTC) and 749 s in repeat 3,
instead of 304–313 s. Over the 10 configs, Reference GEMM (CPU) in that discarded first pass was
2,008 s instead of 788 s, while GPU-side categories barely moved. CPU-reference changes in P3
can only be judged on uncontended runs.

Build side: no repeat could be measured uncontended. The only second sample is
`per_config_build`: same sources, different client build, measured 2026-10-21. It differs by
+3.0% in total, and per config by 0–9.4% over the 10 most expensive build configs.

## Conditions and interference

The host is shared, and the measurement cores and GPU are not reserved. Recorded with the
measurements in `/tmp/tl_plan/P0.2/`:

- `snapshots.txt`: `uptime`, per-core idle on 0–24, and other processes on 0–24 at the start and end
  of every pass. Load averages at the clean run pass: 8.8 (start), 7.9 (end). At the build pass:
  8.6 (start), 21.1 (end).
- `contention.log` (from 00:29): every 30 s, other users' processes on cores 0–24 with ≥1% CPU
  (pidstat).
- `corpus.log` (from 05:26): cumulative CPU time of other users' pinned corpus runs.
- `gpu.log` (from 06:16): use, sclk, VRAM and power of the measurement GPU. Before 06:16 the
  snapshots read `rocm-smi -d 0`, which is a different card (`0000:05:00.0`, ROCR device 1).
  So there is no GPU-side record for the committed passes. Indirect evidence that they were not
  GPU-contended: GPU-side categories are at or below `per_config_rerun_2`, and the clean
  repeat matches the baseline to within 1.4% on every category except bookkeeping (5.4%) and
  kernel launch (11.4%).

Interference seen:

- Another user's corpus run pins short-lived compile/timing jobs to core 9 (`--timing-cpus 9`)
  and to cores 0–8, 10–47 (`--cpus`). It ran about 20:20–01:25, 03:33–05:51 and from 06:26 UTC
  (still running at 09:16).
  - The first per-config run pass (23:07–00:26) overlapped it and was discarded: TOTAL 3,911 s,
    and the Int8 4111³ slow-path problem took 201.7 s instead of 72–75 s.
  - The committed run pass was re-measured afterwards in a clean window (01:40–02:34; no jobs
    from it on cores 0–24 after 01:30).
  - The committed build pass (00:26–01:23) and both split phases (22:07–23:07) overlapped it and
    were kept, with the caveats above.
- Unpinned processes (for example a root `python3` at 100% of one core) are placed on idle cores
  in 0–24 by the scheduler and move away once the measurement threads load those cores. They are
  not counted as contention.
- A root vLLM server (TP=4) occupied rocm-smi GPUs 0–3, including the measurement GPU, at up to
  ~279 GB and 100% use. It was there in short bursts from 06:35 UTC and continuously
  07:20–08:05 and 08:17–09:12. This was after all the committed passes.

Recommendation for P3/P6: monitor the measurement GPU as `rocm-smi -d 3`. Before and during each
pass, check for other users' jobs pinned into 0–24 (by CPU-time accounting, since pidstat misses
short-lived processes). Start passes only in a clean window (as `/tmp/tl_plan/P0.2/driver4.sh`
does), and keep or repeat only uncontended samples.

## Reproducing

```bash
git worktree add --detach /tmp/tl_plan/P0.2/wt 2083e6254bf
# client (CMake cache identical to /tmp/tl_client_msgpack); venv provides cmake 4.4.4
source /home/alvasile/venv/bin/activate
cmake --preset tensilelite -S /tmp/tl_plan/P0.2/wt/projects/hipblaslt -B /tmp/tl_plan/P0.2/build_client \
  -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx950 \
  -DCMAKE_CXX_COMPILER=/usr/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DHIPBLASLT_ENABLE_YAML=OFF -DTENSILELITE_ENABLE_CLIENT=ON -DTENSILELITE_BUILD_TESTING=OFF \
  -DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
taskset -c 128-191 cmake --build /tmp/tl_plan/P0.2/build_client -j64 --target tensilelite-client

export TL=/tmp/tl_plan/P0.2/wt/projects/hipblaslt/tensilelite
export CLIENT=/tmp/tl_plan/P0.2/build_client/tensilelite/client/tensilelite-client
export ART=/tmp/tl_plan/P0.2/artifacts
export TENSILE_CACHE=/tmp/tl_plan/P0.2/cache/tensile TENSILELITE_CACHE=/tmp/tl_plan/P0.2/cache/tensilelite
S=$TL/timing_runs/full_runs_gfx950
OUT=/tmp/tl_plan/P0.2/split bash $S/split/run_split.sh
OUT=/tmp/tl_plan/P0.2/baseline_run KEEP_CACHE=1 bash $S/per_config/run_per_config.sh /tmp/tl_plan/P0.2/configs.txt
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OUT=/tmp/tl_plan/P0.2/baseline_build \
  SCRATCH=/tmp/tl_plan/P0.2/build_scratch bash $S/per_config/run_per_config_build.sh /tmp/tl_plan/P0.2/configs.txt
```

`configs.txt` holds the `test_config_run` IDs that passed the split run phase (184 lines, the
same set as `per_config_rerun_2/configs.txt`). A durable copy is
`/home/alvasile/timing_runs_raw/P0.2/configs.txt`; use it if the `/tmp` copy is gone. The noise
lists are next to it (`noise_run_configs.txt`, `noise_build_configs.txt`).
