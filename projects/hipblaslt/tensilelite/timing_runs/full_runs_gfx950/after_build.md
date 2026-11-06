# gfx950 build-side measurement after the build-side changes (P6)

Build-side measurement at the end of P4 of the implementation plan (`timing_runs/DECISIONS.md`,
P6), compared with the P0.2 baseline (`baseline.md`). It covers the build-side changes
P4.1–P4.15 and the fix of a regression this measurement found (see "Regression found").

## What was measured

| Item | Value |
|---|---|
| Commit | `9f493d7a6c1` (end of P4 plus the custom-kernel fix; detached worktree). The first 25 per-config pairs and one supplementary split triple ran at `38c29661a7d` (end of P4), see "Regression found". |
| Client | P3's `tensilelite-client` (built at `3b4ec8e125d`, sha256 `15f20fa8a2eb…`). No client, library or CMake source changed between `3b4ec8e125d` and `9f493d7a6c1` (P4 and the fix are Python, YAML and `measure_env.sh` only). The client only runs during the build phase in ductile configs. |
| Baseline | P0.2's worktree (`2083e6254bf`) and client (sha256 `4b2769f7f9a2…`), for the same-session control runs |
| Python | venv Python 3.12.3; TensileLite Python sources from each worktree; venv `rocisa` (unchanged) |
| Host | `smci355-1b112-b13-15` (as P0.2): 2× EPYC 9575F, 8× MI355X (gfx950), ROCm 7.2.1, `/usr/bin/amdclang++` |
| Cores / threads | `taskset -c 0-24`, `OMP_NUM_THREADS=25`, `OPENBLAS_NUM_THREADS=25` (as P0.2) |
| GPU | An idle MI355X per pair or split run: `ROCR_VISIBLE_DEVICES=5` (PCI `85:00.0`) or `0` (PCI `75:00.0`, the P0.2 GPU). The build phase uses it only for the ductile configs' client runs. |
| Config list | Per-config pass: the 185 configs of P3 (P0.2's 184 plus `gemm/gfx950/prediction_bf16`, new in P2.5). Split: all of `Tests/common` with the gfx950 markers, as `split/run_split.sh`. |
| Date | 2026-11-01, 14:25–17:49 UTC |

Committed results:

- `after_build/AGGREGATE_analysis.txt`: per-config build aggregate (cold cache per config) at
  `9f493d7a6c1`/`38c29661a7d` (see above), 185 logs.
- `after_build/split_build_outcomes.txt`: outcome of every test in the split build phase at
  `9f493d7a6c1`.

Raw logs (not committed, as for P0.2/P3) are in `/home/alvasile/timing_runs_raw/P6/`, gzipped:
per-config logs of both sides, every split build log (with captured Tensile output), the
same-session P0.2 control aggregate, monitoring logs and the driver/analysis scripts (`scripts/`).

## How it was measured

`baseline.md` asks P6 to re-run the P0.2 build pass and split build phase back to back with its
own, under identical conditions, because both P0.2 build numbers overlapped another user's job.
As in P3, every comparison here is same-session:

- **Per-config pass (interleaved).** For each config, P0.2 (its worktree and client) and P6 ran
  back to back on the same idle GPU, alternating which went first, with
  `run_per_config_build.sh`'s settings (serial, `TimingInstrumentation=True`, caches cleared
  before every config, 25 pinned cores). The script copy used for both sides differs only in
  setting `ROCR_VISIBLE_DEVICES` to the pair's GPU and in skipping the per-call aggregate. A pair
  was re-run (up to three tries) if another user's process took more than 1 GB on the GPU or
  more than 10 CPU-seconds of pinned work on cores 0–24 during either run.
- **Split build phase** (`--build-only`, `-n 4`, cache emptied at the start and kept across
  configs, the CI-like build): the same command as `split/run_split.sh`'s build phase, plus `-rP`
  so that the captured Tensile output of passed tests (the cache lines counted below) is in the
  log. Three runs per triple, back to back: P0.2, P6, and P6 with
  `TENSILE_DISABLE_HELPER_CHUNK_CACHE=1`. At `9f493d7a6c1` there are two triples, in opposite
  order (C–B–A at 16:51, A–B–C at 17:20).
- Pytest temp dirs (`TMPDIR`), caches, scratch artifacts and output were on `/dev/shm` for both
  sides, because `/` was nearly full.

Conditions: no CPU from other users' jobs pinned into cores 0–24 in any run (no corpus job
there today; sampled every 30 s). The 1-minute load average was 17–40, which includes the
measurement's own 25 cores. GPUs: the pairs ran on `ROCR` 5 (141 pairs) and 0 (43 pairs). One
pair, `gemm/zgemm`, had another process on the GPU in all three tries and is kept (it builds no
client run, so the GPU is idle in it). In the last four split runs, on `ROCR` 0, idle root
processes held 0.5 GB of VRAM and briefly up to 16 GB, which the monitor flagged; the build does
not use the GPU outside the ductile configs, and those runs match their `ROCR` 5 counterparts.

## Regression found: custom-kernel configs failed to build at `38c29661a7d`

The first split build at `38c29661a7d` had **8 new build failures**:
`gemm/gfx950/custom_xfp32` and `custom/{custom_aiter_bf16, custom_aiter_f4,
custom_rocblas_gemv_bf16_tn_m1, _m2, _m4, custom_triton_f4, custom_wave_bf16}`. They all failed
with `KeyError: 'LdsNumBytes'` in `TensileCreateLibrary/Run.py` `processKernelSource`. P4.7
(`168da253f63`) and P4.8 (`66281bc369e`) read `LdsNumBytes`, `LoopIters` and `LdsOffsetBias`
from every kernel after codegen, and custom kernels (assembly read from a file) never have those
keys. Neither step's verification built a custom-kernel config. The fix, `9f493d7a6c1`
(`fix(tensilelite): do not require LdsNumBytes / LoopIters / LdsOffsetBias on custom kernels`),
reads them with `.get()` and copies only keys that exist.

At `9f493d7a6c1` every split build phase gives 196 passed, 2 failed, 370 skipped (4 runs,
identical lists). That differs from P0.2's build phase only by the new `prediction_bf16`
(PASSED). The 2 failures are the known `custom_rr_f6`/`custom_rr_mx` ("No valid solutions
found").

The worktree was moved to `9f493d7a6c1` between per-config pairs 25 and 26 (15:12:01 UTC).
Pairs 1–25 were measured at `38c29661a7d` and kept: they include none of the 8 configs (those are
pairs 31 and 178–184, all measured at `9f493d7a6c1`). For kernels that have the three keys, which
is every kernel of a config that built at `38c29661a7d`, the fix runs the same code. The split
triple at `38c29661a7d` is reported only as a supplement, since 8 of its configs stopped early.

## Split build phase (`-n 4`, cache kept across configs, uninstrumented)

| Run (UTC start, GPU) | Commit | Wall | Δ vs P0.2 (same triple) | Sum of test call times |
|---|---|---:|---:|---:|
| P6, chunk cache off (16:51, `ROCR` 5) | `9f493d7a6c1` | 539 s | −12.8% | 1,922 s |
| P6, chunk cache on (17:00, `ROCR` 5) | `9f493d7a6c1` | 546 s | −11.7% | 1,959 s |
| P0.2 (17:09, `ROCR` 0) | `2083e6254bf` | 618 s | | 2,428 s |
| P0.2 (17:20, `ROCR` 0) | `2083e6254bf` | 615 s | | 2,407 s |
| P6, chunk cache on (17:31, `ROCR` 0) | `9f493d7a6c1` | 525 s | −14.6% | 1,924 s |
| P6, chunk cache off (17:40, `ROCR` 0) | `9f493d7a6c1` | 511 s | −16.9% | 1,862 s |
| *Supplement:* P0.2 / P6 on / P6 off (14:25, `ROCR` 5) | `38c29661a7d` | 620 / 502 / 484 s | | 2,424 / 1,754 / 1,722 s |

- **Wall time: P0.2 616.5 s (mean of 2), P6 535.5 s with the chunk cache (−13.1%), 525 s
  without it (−14.8%).** P0.2's two runs agree within 0.5% and with the 620 s at 14:25, so
  this session's P0.2 build phase is 615–620 s. The 634 s in `baseline.md` was measured under
  contention, about 3% higher.
- The summed test call times fall more (−20%) than the wall time. With `-n 4` the wall time is
  bounded by how the 4 workers' queues end, and P6 sizes each worker's pools to the P4.3 budget
  (2 × 25 / 4 = 12 processes: 10 codegen workers plus 2 helper-compile chunks). P0.2 ran up to 25
  processes per worker. In the same-session runs at `9f493d7a6c1`, `dtl` takes 271–301 s under
  `-n 4` with P6 against 190–191 s with P0.2. With P6 it still ends at 63–71% of the run's wall
  time (P0.2: about 82%), so it is not on P6's critical path. Per-worker busy time ranges from
  about 430 to 542 s with P6, against 603–612 s with P0.2 (`split_p0_2` and the P6 runs, from
  `build.xml` durations in worker order).
- **Chunk cache on versus off: no measurable benefit.** In both orders the run without the
  chunk cache was faster, by 7 s and 14 s (1–3%). That is within the session's drift (the same
  configuration differed by 21–28 s between the triples).
- The two known failing configs now get further before failing (P4.10 prepares every step
  first), at a cost of 1–2 s of test time.

### Cache hit rates (CI-like run, empty cache at start)

Counted from the Tensile output in the split logs (`scripts/split_stats.py`), P6 at
`9f493d7a6c1`, identical in all four runs except where noted:

| Cache | Lookups | Hits | Rate | Notes |
|---|---:|---:|---:|---|
| Whole-file helper cache (`HelperKernelCache`) | 927 | 820 | 88.5% | P0.2: 753 of 859 (87.7%). P6 does 67 more lookups, all hits, in the two failing `custom_rr_*` configs, which get further now (see above), and 1 (a miss) in the new `prediction_bf16`. That miss is also why there are 107 PCH lookups instead of 106. |
| HIP-header PCH (P4.12) | 107 (one per whole-file miss) | 103–104 | 96–97% | 3–4 PCH builds: up to 3 xdist workers built the shared key at the same time at the start (each stores it once, by rename), and `custom_rr_mx` uses its own key. 2 entries, 26 MB. |
| Helper chunk cache (P4.15) | 182 chunks in 91 chunked compiles | 21 | **11.5%** | 159 entries, 24 MB. 0 key-computation failures, so FOLLOWUPS P4.15's caveat (a failed key computation still prints "0 of n") did not affect the count. 0 fallbacks to the one-piece compile. |

The chunk cache only gets lookups that the whole-file cache missed, and those are mostly helper
sets no earlier config had. The 21 hits are spread over 18 configs, 1–3 chunks each.
`subtile_bf16` hits 3 of 8. Under `-n 4` each helper compile uses 2 chunks, while a serial
build uses 4. Chunk contents depend on the budget (FOLLOWUPS P4.14), so a cache filled by one
of these builds does not serve the other.

In the per-config pass (cache cleared before every config), P6 built the PCH once per config
(184 builds, 79 hits by later helper compiles of the same config). 14 of 719 chunks hit (1.9%).

## Per-config pass (cold cache per config, instrumented)

Pytest wall seconds summed over the 184 configs common to both sides. "P0.2" is the committed
`baseline_build` pass. "Control" is P0.2 in the interleaved pass.

| Set | Configs | P0.2 | Control | P6 | Δ vs control |
|---|---:|---:|---:|---:|---:|
| All | 184 | 3,361 | 3,359 | 1,970 | **−41.4%** |
| … `LibraryLogic:`/`LibraryClient:` removed or added (P2.5/P2.6) | 25 | 259 | 257 | 178 | −30.7% |
| … all others | 159 | 3,102 | 3,102 | 1,792 | −42.2% |
| … `CpuThreads: 1` removed (P4.8) | 19 | 526 | 537 | 195 | −63.7% |
| … the 8 configs fixed by `9f493d7a6c1` | 8 | 57 | 57 | 40 | −29.8% |
| … pairs 1–25 (measured at `38c29661a7d`) | 25 | 600 | 595 | 430 | −27.7% |

`prediction_bf16` (P6 only) adds 5 s and is in the committed aggregate only.

**The control reproduces the committed P0.2 build pass**: pytest sums 3,359 against 3,361 s
(−0.1%), TOTAL RUNTIME 2,662.4 against 2,654.8 s (+0.3%). The core-9 contention during the
P0.2 pass therefore did not inflate it measurably, and comparisons against the committed
baseline are equally valid.

Seconds from "TIMING BY PURPOSE" over the same 184 configs. P0.2 is re-aggregated with the
analyzer at `9f493d7a6c1`. This only moves the helper compile, see the note below the table.

| Category | P0.2 | Control | P6 | Δ vs control |
|---|---:|---:|---:|---:|
| TOTAL RUNTIME | 2,654.8 | 2,662.4 | 1,284.1 | −51.8% |
| BUILD: solution generation & compile | 2,605.6 | 2,613.0 | 1,238.9 | −52.6% |
| … `python_kernel_compilation` | 2,260.4 | 2,273.7 | 903.2 | −60.3% |
| … … helper compile in the build's path (P0.2: serial; P6: `python_kernel_build_src_co_wait`) | 1,013.4 | 1,023.6 | 138.1 | −86.5% |
| … … `python_kernel_codegen` (calls) | 615.5 (855) | 607.4 (855) | 301.2 (213) | −50.4% |
| … … `python_kernel_write_assemble` | 253.8 | 264.1 | 236.0 | −10.6% |
| … … `python_kernel_bench_setup` | 132.2 | 132.6 | 4.5 | −96.6% |
| … … `python_kernel_bench_postprocess` | 105.2 | 103.9 | 106.0 | +2.0% |
| … … `python_kernel_validate` | 82.6 | 80.9 | 58.3 | −27.9% |
| … … `python_kernel_build_co` | 22.1 | 25.9 | 25.7 | −0.8% |
| … `python_solution_generation` | 321.6 | 319.0 | 314.2 | −1.5% |
| … Ductile GA search (Python) | 23.7 | 20.2 | 21.5 | +6.4% |
| SHARED: cache & solution files | 23.2 | 23.2 | 23.4 | +0.9% |
| RUN: benchmark client | 14.3 | 14.9 | 10.0 | −32.9% |
| Helper compile itself (`python_kernel_build_src_co`, overlapping total) | 1,013.4 | 1,023.6 | 393.0 | −61.6% |
| … `.compile` | 1,010.5 | 1,020.9 | 286.9 | −71.9% |
| … `.pch` (new; PCH built once per config here) | – | – | 98.7 | |
| Sum of per-config pytest wall times | 3,361 | 3,359 | 1,970 | −41.4% |

Note on the helper compile: since P4.9 it runs in a background thread during codegen. The tree
therefore shows only the time the build then waits for it (`python_kernel_build_src_co_wait`),
and the compile itself is listed under "Overlapping Totals". In P0.2's logs it ran serially and
appears as untracked time in `python_kernel_compilation` (1,017.2 s in P0.2, 1,027.1 s in the
control).

What the numbers show, by plan item:

- **Helper compile (topic 6, P4.9 and P4.11–P4.15): 1,024 s → 138 s in the build's path.** Its
  own time falls to 393 s: the bare-arch compile (P4.11), the PCH (P4.12), and chunks compiled in
  parallel (P4.14). Most of what remains overlaps codegen (P4.9/P4.10). In this cold-cache pass,
  building the PCH once per config costs 98.7 s of that, in the background.
- **Codegen (topic 7, P4.3–P4.8, P4.10): 607 → 301 s.** One pool per config (213 instead of 855
  `python_kernel_codegen` calls). The static-VGPR and overflow checks stop rejected kernels early
  (P4.5/P4.6). The 19 configs that had `CpuThreads: 1` now generate in parallel (P4.8): −64% for
  that set. `python_kernel_validate` −28% and `write_assemble` −11% probably follow from the
  same changes (not measured separately).
- **Benchmark setup (topic 8, P4.1/P4.2): 133 → 4.5 s.**
- **Unchanged as expected:** solution generation (−1.5%), bench postprocess, solution/cache
  files, Ductile GA search.
- **RUN: benchmark client −4.9 s** is the ductile configs' client runs during the build. P1's
  client changes act there too (input preparation −3.8 s, as in P3).
- The untimed part per config (pytest start-up, imports, artifact packing) is the same on both
  sides (697 against 686 s over 184 configs). So the wall-time gain (−41%) is smaller than the
  timed gain (−52%).
- Largest configs (pytest s, control → P6): `subtile_mxfp4` 211 → 45, `dtl` 168 → 137,
  `subtile_bf16` 121 → 19, `cgemm` 97 → 79, `bufferLoadStore0` 82 → 68,
  `custom_mainloop_scheduling` 76 → 30, `plr_zero` 72 → 60, `spmm_f16_sb` 72 → 43, `zgemm`
  67 → 61, `fp16_tn` 60 → 53. No config is slower than its pair.

### Configs with `LibraryLogic:`/`LibraryClient:` changes (P2.5/P2.6)

`--build-only` skips both steps (`Tensile.py`: "Build-only mode: skipping LibraryLogic and
LibraryClient"), so these sections do not change the build-side work. Their YAML changes do
not explain any of the gain. The 25 configs whose sections changed between `2083e6254bf` and
`9f493d7a6c1` are reported separately above. All 25 lost their sections (P2.6): the 8
`ductile/gfx950/*`, the 2 mxfp8 repro configs, the 8 `subtile_mxfp8_*` configs and the 7 gfx950
`custom/*` configs. The 8 `subtile_mxfp8_*` configs also lost `CpuThreads: 1` (P4.8).
`prediction_bf16` gained both sections (P2.5); it is P6-only, so it is not in the 25. The set's
gain (−30.7%) is smaller than the rest's (−42.2%) because all 25 are small configs (6–19 s each
in the control), where the untimed per-config overhead (about 3.7 s) dominates.

## Findings that bear on the build side

- **P4.7/P4.8 regression** (above), fixed in `9f493d7a6c1`. The verifications of those steps
  built no custom-kernel config. A unit-level case for a custom kernel now covers it (in the fix
  commit).
- **The chunk cache (P4.15) gives no measurable wall-time gain** in either build: 11.5% chunk
  hits in the CI-like build, 1.9% within cold single configs. The run without it was 1–3% faster
  in both orders (within drift). Its benefit depends on configs that share many helpers but
  miss the whole-file cache, which the common set rarely has. Also, chunk contents depend on
  the pool budget (FOLLOWUPS P4.14), so a cache filled by a `-n 4` build (2 chunks per compile)
  does not serve a serial build (4 chunks). Keeping it on is harmless. Whether to keep the
  feature is a decision for the plan owner.
- **FOLLOWUPS P4.15** (a failed chunk-key computation still prints "0 of n chunks hit"): no key
  computation failed in any P6 run, so the hit counts above are exact.
- **FOLLOWUPS P4.12** (`.tmp_*` entries of killed processes are never evicted; per-process temp
  name in `HelperKernelCache`): no cache-related build failure occurred in the 4 split runs at
  `9f493d7a6c1`, and the last run's cache had no `.tmp_*` left. (The earlier runs' caches were
  cleared before they could be checked.) 3 xdist workers built the same PCH key at once, which is
  the race the rename guards against.
- **FOLLOWUPS P4.14:** TensileCreateLibrary's (shipped library) helper compile is still in one
  piece. P6 measures only the benchmark/test build path.
- **FOLLOWUPS P4.7/P4.9:** `DisableAsmComments` reaches the helper `Kernels.h` inconsistently in
  serial codegen. Since P4.8 only pool-size-1 runs are affected, and no P6 run has pool size 1.
- The per-worker budget (P4.3) leaves cores idle at the end of a `-n 4` run once fewer than 4
  tests remain (see the split section). Not a regression: P0.2 oversubscribed instead.

## Reproducing

```bash
git worktree add --detach <wt> 9f493d7a6c1
source /home/alvasile/venv/bin/activate
export TL=<wt>/projects/hipblaslt/tensilelite
export CLIENT=<client built as in after_run.md, at 3b4ec8e125d or later>
export TENSILE_CACHE=<cache>/tensile TENSILELITE_CACHE=<cache>/tensilelite ART=<new dir>
# split phases: run_split.sh runs the build phase, then the --use-cache run phase. The P6 number
# is the "END build" line of $OUT/times.txt; the run phase that follows is not part of P6. P6
# also added -rP to the build command (to log the cache lines);
# TENSILE_DISABLE_HELPER_CHUNK_CACHE=1 for the chunk-cache-off runs
OUT=<dir> bash $TL/timing_runs/full_runs_gfx950/split/run_split.sh
# per-config cold build
ROCR_VISIBLE_DEVICES=<idle GPU> OUT=<dir> SCRATCH=<dir> \
  bash $TL/timing_runs/full_runs_gfx950/per_config/run_per_config_build.sh <configs>
```

The P0.2 side used P0.2's worktree and client (`baseline.md`, "Reproducing"). The interleaved
driver, the per-config script copy and the analysis scripts are in
`/home/alvasile/timing_runs_raw/P6/scripts/` (`driver.sh`, `split_stats.py`, `cmp_pairs.py`,
`cmp_agg.py`).
