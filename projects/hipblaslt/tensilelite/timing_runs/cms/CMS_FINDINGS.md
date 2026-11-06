# CMS (UseCustomMainLoopSchedule) build-only analysis, gfx950 target

Run 2026-10-16 on a gfx942 host (build-only, `--gpu-targets gfx950`, so no gfx950 GPU needed),
cold caches, `taskset -c 0-24` (25 cores), `TimingInstrumentation=True`, `/usr/bin/time -v`.
`gfx952` does not exist in this tree; treated as gfx950 (the gfx95x family).

## Results

| Config | Wall | User+Sys CPU | CPU used (of 25 cores) | Groups | Kernels/group |
|---|---|---|---|---|---|
| `gemm/gfx950/custom_mainloop_scheduling` | 112 s | 166 s | **148%** (about 1.5 cores) | 66 | 1 (53 groups) or 2 (13) |
| `gemm/gfx950/custom_mainloop_scheduling_tf32` | 45 s | 72 s | 160% | 22 | 1 |
| `ductile/gfx950/f8bf16_tn_cms` | 23 s, failed | | 392% | n/a | n/a |

`f8bf16_tn_cms` fails by design in build-only: the ductile backend warns that it does not support
build-only and forces a benchmark run, which needs a gfx950 GPU
(`Expected results file does not exist`). Its logs are kept; it is not part of the analysis.

## Is the pathology still there? Yes.

The 25-core pool is idle almost the whole run: about 1.5 cores busy on average.

1. **Groups run strictly one after another.** `BenchmarkProblems.main` loops serially over
   `for problemType: for sizeGroup: _benchmarkProblemType(...)`. The big file has 9 problem types and
   66 `BenchmarkProblemSizeGroup`s. Each group does its own fork / codegen / write / compile / library
   write before the next starts.
2. **Each group holds almost no parallel work.** Each group has `UseCustomMainLoopSchedule` as a
   single value (`[1]` in 51 groups) or `[0, 1]` (14 groups), and every other ForkParameter is fixed
   (58 unique fork sets across 66 groups, so the groups are different hand-picked tile configs rather
   than duplicates). Result: `Actual Solutions: 1 / 1` in 53 groups and `2 / 2` in 13.
   The log shows `Writing assembly kernels: Launching 25 threads for 1 tasks` (or 2 tasks) in all
   66 groups, i.e. 25 workers for 1-2 tasks.
3. **A group costs about 1 s no matter what** (median 0.98 s for 1-solution groups, 1.13 s for
   2-solution groups; `python_kernel_codegen` 0.76 s/group average). Total about 66 x 1.4 s plus the
   compiles.
4. **The assembler/compiler step is also serial.** `build_src_co.compile` ran 3 times (9.3 + 4.7 + 4.9 s
   = 18.8 s, 18% of the build) on unique helper kernels, one after the other, inside three of the
   groups.
5. Per-stage totals (66 groups): kernel codegen 50.4 s (49%), compile 18.8 s (18%), solution
   generation 13.8 s (13%), write/assemble 13.4 s (13%).

Not the cause: process-pool creation. A bare `Pool(25)` plus one task costs about 22 ms here, which
is small compared with the roughly 1 s per group.

## Experiment: does group-level parallelism recover the time?

`split_experiment.py` splits the same YAML round-robin by size group into N shard configs and
builds them concurrently (25 cores divided evenly, cold caches, same flags).

| Shards | Cores/shard | Wall | Speedup vs 112 s |
|---|---|---|---|
| 1 (original) | 25 | 112 s (pytest), 103.6 s inside the benchmark step | 1.0x |
| 5 | 5 | 36.8 s | 3.0x |
| 25 | 1 | 21.9 s | 5.1x |

The speedup is real but is capped by the compile step: each shard compiles its own copy of the
helper kernels because the shared helper-kernel cache starts cold (29 `cache MISS` and 265 s of total
compile CPU at 25 shards, against 3 misses and 18.8 s in the single run). Each shard at 25 ways takes
12-19 s, which is mostly its own helper compile.

## Where the speedup is

1. **Run size groups concurrently** (in `BenchmarkProblems.main`, or by building the problem
   types / groups with a process pool). This is the main win: 3x to 5x measured on a split of the
   same input. It needs per-group outputs to be independent; the experiment splits them without any
   issue, so for build-only they appear to be.
2. **Compile helper kernels once, up front**, instead of inside whichever group first misses the
   cache. This removes the serial 18.8 s and the duplicated compiles seen in the split runs
   (e.g. warm the helper cache, or key the cache so concurrent builders share it).
3. **Batch the YAML** only if a config author can: groups that share the same ProblemType and are
   only different tile configs can sit in one `ForkParameters` sweep, which Tensile already
   parallelizes. The CMS file cannot do that as written: each group is a specific hand-picked tile
   config, not a cross product, so there is no way to express them as one sweep without generating
   extra kernels.
4. Lower priority: Python codegen is 49% of CPU time at 0.76 s per kernel; a faster per-kernel codegen
   helps every config, not just these.

## Caveats

- Build-only for gfx950 on a gfx942 host: no GPU run, so kernel correctness was not checked.
- Single run per measurement, on a shared machine. The split experiments use hand-written shards,
  not a production change; they show the potential, not a verified implementation.
- The experiment ran under `Tensile.Tensile([...])` directly (not pytest), so exact overhead differs
  slightly from the pytest runs.

## Files

- `custom_mainloop_scheduling_build.log`, `..._time.txt` (`/usr/bin/time -v`), `..._analysis.txt`
- the same three for `custom_mainloop_scheduling_tf32` and `f8bf16_tn_cms`
- `split_experiment.py`, `run_split.sh`, `split_5/`, `split_25/`, `split_*_result.txt`
## What the helper-kernel cache saves, single shard and 25 shards

Same input and conditions (25 cores total), caches kept (warm) versus deleted (cold):

| Run | Cold | Warm | Saved by cache | Helper-kernel compiles |
|---|---|---|---|---|
| 1 shard (the 112 s case) | 111 s (pytest) | 101 s | about 10 s (9%) | cold 3 compiles = 18.8 s; warm still 1 miss = 9.2 s |
| 25 shards (1 core each) | 21.9 s | 10.1 s | 11.8 s (54%) | cold 29 misses / 265 s compile CPU; warm 0 misses, 66 hits |

- In the single shard, the warm run still had one helper-cache miss (9.2 s, `build_src_co.compile`
  called once), so it saved 9.6 s of the 18.8 s compile, not all of it. The compile is only
  9-17% of the single-shard run, because Python codegen (50 s), solution generation (14 s) and
  write/assemble (13 s) are untouched by the cache.
- With 25 shards the cache matters much more: every shard's wall time is dominated by the one
  serial helper compile it would otherwise do. Warm, each shard takes 1.6-7.6 s (benchmark step) and
  the whole job finishes in 10.1 s, including about 2-3 s of process start/import per shard.
- Net effect of both changes on the CMS file: 111 s cold single shard to 10.1 s warm 25 shards,
  about 11x. Parallelism alone gives 5x (cold), the warm cache adds another 2.2x on top at 25 shards.
- Files: `custom_mainloop_scheduling_warm_*` (single warm run), `split_25_warm/`,
  `split_25_warm_result.txt`, `run_split_warm.sh`.
