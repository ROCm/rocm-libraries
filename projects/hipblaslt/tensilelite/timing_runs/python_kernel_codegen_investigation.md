# `python_kernel_codegen` investigation (gfx950 per-config build)

`full_runs_gfx950/per_config_build/AGGREGATE_analysis.txt` reports `python_kernel_codegen` at
592,584.11 ms (23.0% of build wall time) over 855 calls, mean 693 ms. This note explains where
that time goes and what can be saved. Timing claims come from measurements: in-run probes of the
real builds, offline replays of the exact kernels, py-spy with native frames, cProfile (ranking
only) and in-run A/B runs with artifact comparison. Code reading was used to decide what to
measure. Inferences and estimates are marked as such.

## Summary

The timer wraps one call (`TensileCreateLibrary/Run.py:666`): `ParallelMap2(processKernelSource,
...)`, which generates the assembly text of every kernel of one benchmark step in a joblib/loky
pool (25 processes in these runs) and returns the text to the parent. Pool start-up, dispatch and
result return are **not** the cost: 2–12 ms before the first kernel starts and 2–85 ms after the
last one ends, per step (135 ms on a config's first step, when the pool starts). The time is the
generation of each kernel, and how much of it can run in parallel.

1. **Generating one kernel costs 0.04–3 s of single-threaded Python + rocisa C++, proportional to
   the assembly it produces (~90–280 ms per MB; kernels are 0.4–7 MB of text).** Of that, about 65%
   is Python building the instruction tree (the store/epilogue code alone is 40–64%), ~21% is the
   C++ `rocIsaPass`, ~8% freeing the tree and ~6% `str()` of the tree.
2. **19 configs set `CpuThreads: 1`, so codegen runs in one process.** They are 189.3 s of the
   592.6 s (32%), including all 13 `subtile_*` configs (169.4 s). `subtile_mxfp4`: 133.5 s → 22.6 s
   in-run with 25 processes; `.co` files and libraries identical, but see the `LdsNumBytes` caveat.
3. **Kernels that will be rejected are generated in full first.** A kernel whose static VGPR
   allocation already exceeds 256 is guaranteed to be rejected, but this is only acted on after
   the whole kernel is generated. These are often the slowest kernels: 8–89% of codegen CPU in four
   of the probed configs. Failing at `_initKernel` was exact on 798 kernels and cut `f8_gate_r`
   codegen 12.6 s → 3.5 s in-run (whole build 28 s → 11 s) with identical artifacts.
4. **Most steps are too small to use the pool.** 479 steps write ≤25 kernels (159.5 s); a step's
   wall time is then its slowest kernel while the other workers idle (`custom_mainloop_scheduling`:
   36 s of wall for 40.7 s of CPU on 25 cores). Steps are independent, but are generated one after
   another.

| Option | Measured / estimated effect | Output | Effort |
| --- | --- | --- | --- |
| **A. Remove `CpuThreads: 1` from the 19 test configs** (plus return `LdsNumBytes` from workers) | `subtile_mxfp4` codegen 133.5 → 22.6 s (5.9×), `write_assemble` 62.9 → 11.6 s, build 244 → 63 s, measured. Aggregate estimate: ~189 s → ~35 s codegen | identical `.co`/library; `Data/*.yaml` `LdsNumBytes` differs unless passed back | config + small code fix |
| **B. Fail fast on rejected kernels** (initial VGPR allocation > `MaxVgpr`, plus abort as soon as any rejection code is set) | VGPR check alone: `f8_gate_r` codegen −72% (12.6 → 3.5 s) in-run, `plr_zero` worker CPU −20%; both checks: CPU on rejected kernels 374.5 → 10.8 s (−97%) over 1,242 rejected kernels | identical (VGPR check: 798/798 kernels, artifacts byte-identical; both: 4,313/4,313 kernels same accept/reject) | ~10 lines |
| **C. Generate all steps of a config in one pool** | small-step configs: `custom_mainloop_scheduling` 36.0 → ≤2.1 s, `f8_gate_r` 12.4 → 3.3 s, `spmm_i8_sb` 17.6 → 9.1 s (simulated from measured per-kernel times); ≤25-kernel steps aggregate 142.4 s → ≤75 s (conservative bound) | identical (same kernels) | restructure `BenchmarkProblems` loop |
| D. Longest-first dispatch within a step | 0–6% on most configs, 24% on `spmm_i8_sb` (simulated, perfect cost knowledge) | identical | needs a cost predictor |
| E. Make the generator cheaper | epilogue 40–64%, `rocIsaPass` ~21%, free ~8%, `str` ~6% of per-kernel time; quadratic `hasAnyDependency` up to 67% on some sparse kernels | depends | larger refactors |

A, B and C are independent and additive (inference): A applies to the 19 single-process configs,
B to rejected kernels everywhere, C to the many-small-steps configs. None of these change the
generated code.

## Method

1. **Distribution.** Each step's `TIMING:python_kernel_codegen` line in the build logs
   (`/home/alvasile/timing_runs/full_runs_gfx950/per_config_build/<cfg>/<cfg>_build.log.gz`) was
   paired with the step's `Writing assembly kernels: Launching N threads for K tasks` line (K =
   kernels that passed validation; generated kernels are more, e.g. 4,093 generated vs 3,420 written
   for `plr_zero`). The 169 steps of the `CpuThreads: 1` `subtile_*` configs print no such line
   (single-process path) and are kept separately. Output: `/tmp/codegen/steps.txt`,
   `/tmp/codegen/cfg_totals.txt`.
2. **In-run probe.** pytest plugin `/tmp/codegen/cgprobe.py` (`-p cgprobe`, run by
   `/tmp/codegen/run_cg.sh`, the same build-only `test_config_build` invocation as the aggregate,
   25 cores via `taskset`). It replaces `Run.processKernelSource` with a module-level wrapper that the
   loky workers import by name, so every worker logs per-kernel start/end wall time, thread CPU
   time, output size and name (`w_<pid>.txt`). The parent logs the block and `ParallelMap2` window
   per step, the pickled task size, and (optionally) pickles the step's kernels, kernel writer and
   rocisa data for replay. Switches: `CG_FORCE_THREADS=N` (override `CpuThreads`),
   `CG_FAILFAST=1` (option B in the workers). `/tmp/codegen/analyze.py` splits each step into
   pool start → first kernel, kernel span, last kernel → return, worker busy/CPU, slowest kernel.
3. **Replay.** `/tmp/codegen/replay_cg.py` (time / cProfile one kernel), `phases.py` (wraps
   `rocIsaPass`, `kernelBody`, `checkResources`, `_initKernel`; times `str(module)` and freeing the
   module separately; `RIPO=` toggles `rocIsaPass` options, `NOCOMMENT=1` sets
   `outputNoComment`), `shares.py` (cProfile section shares), `ff.py` (baseline vs fail-fast over
   every kernel of a step, 16 processes), `sched.py` (actual vs longest-first vs lower bound).
   Replays reproduce in-run per-kernel times (1,931 vs 2,037 ms, 1,577 vs 1,646 ms, 122 vs 125 ms).
4. **Native profile.** `perf` is blocked (`perf_event_paranoid` = 4); py-spy 0.4.2 (installed to
   `/tmp/codegen/pyspy_inst`) with `--native` works, but `_rocisa.so` is stripped, so C++ frames are
   addresses only. C++ costs were therefore measured by timing the native calls directly.
5. Probed configs: `custom_mainloop_scheduling`, `fp16_tn`, `f8_gate_r`, `spmm_i8_sb`,
   `gfx950/dtl`, `plr_zero`, `subtile_mxfp4`.

## Distribution in the aggregate

| Step (kernels written) | Steps | Kernels | Codegen wall |
| --- | ---: | ---: | ---: |
| ≤25 | 479 | 2,974 | 159.5 s |
| 26–200 | 143 | 10,051 | 127.8 s |
| >200 | 64 | 35,101 | 135.9 s |
| `subtile_*` (`CpuThreads: 1`, no count logged) | 169 | — | 169.4 s |

Top configs: `subtile_mxfp4` 97.4 s (130 steps), `subtile_bf16` 61.8 s, `gfx950/dtl` 47.5 s,
`custom_mainloop_scheduling` 34.2 s (66 steps, 79 kernels), `plr_zero` 29.0 s, `spmm_f16_sb`
19.1 s, `spmm_{i8,bf16,f8,b8}_sb` 12–16 s each, `fp16_tn` 11.7 s, `f8_gate_r` 11.4 s (8 kernels
written).

## Where the time goes

### Pool overhead is negligible

| Config | Codegen block | Before first kernel | After last kernel | Worker CPU | Cores busy (of 25) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `custom_mainloop_scheduling` (66 steps) | 36.8 s | 0.6 s total (135 ms on step 0, 2–12 ms after) | 0.4 s | 40.5 s | 1.1 |
| `f8_gate_r` (4 steps) | 12.6 s | 0.14 s | 0.11 s | 46.7 s | 3.7 |
| `spmm_i8_sb` (8 steps) | 17.8 s | 0.03 s | 0.12 s | 219.5 s | 12.4 |
| `fp16_tn` (3 steps) | 14.6 s | 0.01 s | 0.02 s | 330.7 s | 22.7 |
| `gfx950/dtl` (26 steps) | 61.1 s | 0.10 s | 0.19 s | 1,312 s | 21.5 |
| `plr_zero` (17 steps) | 36.6 s | 0.06 s | 0.10 s | 789 s | 21.6 |

A task pickles to ~29 KB (kernel 13 KB, `KernelWriterAssembly` 10 KB, `globalParameters` 3 KB) in
0.2–0.7 ms. The large configs keep the cores ~86–96% busy; the small-step configs barely use them.

### One kernel: cost scales with the assembly text

| Config | Kernels | Median CPU / kernel | Median output | CPU per MB of output |
| --- | ---: | ---: | ---: | ---: |
| `custom_mainloop_scheduling` | 79 | 294 ms | 1.24 MB | 281 ms |
| `fp16_tn` | 2,060 | 124 ms | 0.93 MB | 139 ms |
| `gfx950/dtl` | 8,650 | 110 ms | 0.72 MB | 160 ms |
| `plr_zero` | 3,420 | 136 ms | 0.85 MB | 164 ms |
| `spmm_i8_sb` | 546 | 167 ms | 1.28 MB | 127 ms |
| `subtile_mxfp4` | 1,414 | 67 ms | 0.68 MB | 91 ms |

(Accepted kernels only; `dtl` alone generated 7.5 GB of assembly text.) The largest kernel seen,
`custom_mainloop_scheduling` step 40, is 7.0 MB / 161,079 lines and takes 1.9–2.0 s.

**Phases** (`phases.py`, GC disabled, median of 3; freeing measured separately by holding a
reference — in the real run it happens inside the call when `kernelBody` returns):

| Kernel | Total | `rocIsaPass` | `str(module)` | free module | Python tree construction (rest) |
| --- | ---: | ---: | ---: | ---: | ---: |
| cms step 40 (7.0 MB) | 1,708 ms | 617 ms | 123 ms | +208 ms | ~966 ms |
| cms step 30 (0.57 MB) | 104 ms | 30 ms | 7 ms | +10 ms | ~67 ms |
| `fp16_tn` k700 (1.56 MB) | 182 ms | 36 ms | 17 ms | +25 ms | ~129 ms |
| `f8_gate_r` k0 (2.98 MB) | 359 ms | 57 ms | 26 ms | +41 ms | ~276 ms |
| 24 kernels sampled evenly from `dtl` s11, `plr_zero` s12, `spmm_i8_sb` s0 | 5,023 ms | 1,160 ms | 315 ms | +434 ms | ~3,548 ms |

Over the 24-kernel sample (free included): Python ~65%, `rocIsaPass` ~21%, free ~8%, `str` ~6%.
GC is not a factor (1,852 ms GC off vs 1,783 ms GC on for the largest kernel; noise).

**Inside the Python part** (cProfile, ranking only): the store/epilogue
`notLocalSplitUGlobalWrite` is 40–64% of profiled time in 22 of 28 sampled kernels (17–39% in the rest); the main loop
(`PersistentLoop.compute`) 4–38%. The epilogue is large because it is fully unrolled per element
for every variant (edge/non-edge, beta, bias type, activation): in `fp16_tn` k700, 56% of the
21,419 lines follow `label_ActivationSetPCAddrEnd_*`. On some sparse kernels
`KernelWriter.hasAnyDependency` (a local-read × instruction scan, `KernelWriter.py:1887`) dominates:
25–67% of profiled time in four `spmm_i8_sb` kernels (903 k calls in one), 0% in `dtl`.

**Inside `rocIsaPass`** (`rocisa/src/pass/pass.cpp`; toggling options on the 7.0 MB kernel, which
changes output so only for attribution): mandatory macro/composite/text-variable passes ~231 ms;
register graph + `removeDuplicateAssignment` ~330 ms (also yields `maxVgpr` for occupancy);
`getCycles` ~94 ms (`MathClocksUnrolledLoop`, serialized into the library). None of these can
simply be skipped.

**Comments** (`--disable-asm-comments` / `outputNoComment`): output 1.56 → 0.86 MB but time only
−5% (156 → 149 ms), because comments are still built in Python; a minor lever for codegen
(it would also shrink the text written and assembled in `python_kernel_write_assemble`, not
measured).

### Repeat generation is not the issue

Within each probed config almost every generated kernel is unique (`dtl`: 9,107 generations, 9,056
unique, 8 s of 1,312 s duplicated), and no kernel name appeared in more than one probed config.

## Options

### A. Run the `CpuThreads: 1` configs in parallel

`CpuThreads: 1` is set in 71 test yamls; 19 ran in this aggregate (all 13 `subtile_*` gfx950
configs, `spmm_vw_lg_one{,_sb}` gfx94x/gfx950, `use_sgpr_for_gro{,_sb}`), totalling 189.3 s of
codegen and 92.0 s of `python_kernel_write_assemble`. `ParallelMap2` then takes its serial path
(no "Launching" line). In the subtile configs the setting was present from the moment each file
was added, in three PRs, none of which explains it:

| Commit | PR | Configs |
| --- | --- | --- |
| `1fb7e7dcc69` (bnemanich, 2026-05-03) | #7017 "Add gfx950 MXFP4 and BF16 Subtile-based kernel implementation" | `subtile_mxfp4`, `subtile_bf16` |
| `42e06b9f3f2` (Prabhjot Sandhu, 2026-06-03) | #7656 "[SubtileImpl] Add support for MXFP8 MT128x128 with DU=256" | `subtile_fp8`, `subtile_mxfp8` |
| `e91ff88833c` (Matthew Emmett, 2026-06-24) | #7781 "TensileLite: Add multi-DU support." | the nine `subtile_mxfp8_*` configs |

In #7017 it sits next to debug-style settings (`KeepBuildTmp: True`, commented
`PrintSolutionRejectionReason` / `GenerateSourcesAndExit`); the later files appear to have
copied it. (The files moved to `tensilelite/` in `11d0433bd6d`, #12150.) The non-subtile
`CpuThreads: 1` configs were not traced.

In-run, `subtile_mxfp4` with `CpuThreads` overridden to 25 (same cores, back to back):

| | Serial | 25 processes |
| --- | ---: | ---: |
| `python_kernel_codegen` | 133.5 s | 22.6 s |
| `python_kernel_write_assemble` | 62.9 s | 11.6 s |
| whole build | 244 s | 63 s |

All 130 `.co` files and 130 libraries (`TensileLibrary.dat`, decompressed) are byte-identical.
**25 of 155 `Data/00_Final.yaml` files differ in `LdsNumBytes`** (e.g. 82,432 serial vs 69,632
parallel): the subtile generator sets `kernel["LdsNumBytes"]` during codegen
(`Components/Subtile/SubtileLdsLayout.py:183`). Serially the kernel object *is* the solution, so
the value lands in the YAML; in a worker it is lost, because `passPostKernelInfoToSolution`
(`Run.py:457`) only copies `CUOccupancy`, `PrefetchGlobalRead`, `MathClocksUnrolledLoop` and
`CustomKernel` back. This is not subtile-specific: forcing `custom_mainloop_scheduling` serial
changed one solution's `LoopIters` (4 vs 1; set at `KernelWriter.py:7691`), again with identical
`.co` and libraries. So the solution metadata already depends on `CpuThreads`; enabling
parallelism for subtile should add `LdsNumBytes` (and probably `LoopIters`) to the returned
`KernelCodeGenResult` so both modes agree.

Aggregate estimate (not measured end to end): applying the measured 5.9× to the 189.3 s gives
~32 s, saving ~157 s of codegen and ~75 s of `write_assemble`. The non-subtile ones have few
kernels per step, so their gain will be smaller.

### B. Fail fast on a guaranteed VGPR overflow

`_initKernel` computes the static VGPR allocation and only prints
`warning: total VGPRS (370) overflowed max VGPRS (256).` (`KernelWriter.py:9616`). The VGPR pool
is created with that size (`KernelWriter.py:10270`) and can only grow (`RegisterPool::remove`
marks registers unavailable, `resize` only extends), so the final check
`vgprPool.size() > MaxVgpr` (`KernelWriterAssembly.py:2343`) always rejects the kernel — after the
whole kernel, including the expensive epilogue, has been generated. The rejection surfaces as
`err=-2` and is not printed unless `PrintSolutionRejectionReason` is set.

Prototype: after `_initKernel`, raise if `states.totalVgprs > regCaps["MaxVgpr"]` and
`forceGenerateKernel` is off.

| Step (every kernel, 16 processes) | Kernels | Rejected | Accept/reject mismatches | Codegen CPU |
| --- | ---: | ---: | ---: | ---: |
| `f8_gate_r` step 0 | 8 | 6 | 0 | 11.9 → 1.4 s (−88%) |
| `f8_gate_r` step 1 | 8 | 6 | 0 | 13.2 → 1.3 s (−90%) |
| `spmm_i8_sb` step 0 | 80 | 16 | 0 | 29.5 → 14.1 s (−52%) |
| `plr_zero` step 12 | 546 | 14 | 0 | 86.3 → 85.3 s (−1%; its rejections are mostly other reasons) |
| `plr_zero` step 13 | 156 | 92 | 0 | 81.6 → 26.8 s (−67%) |

Accepted kernels' output size was also unchanged. In-run with the check applied in the workers:
`f8_gate_r` codegen 12.6 → 3.5 s, whole build 28 → 11 s, all 16 artifacts byte-identical;
`plr_zero` worker CPU 789 → 630 s (−20%), codegen 36.6 → 33.6 s (baseline ran on other cores;
wall is noisy), all 68 artifacts byte-identical. Share of codegen CPU spent on rejected kernels in
the probes: `f8_gate_r` 89%, `plr_zero` 29%, `spmm_i8_sb` 24%, `dtl` 8%, `fp16_tn` 1%,
`subtile_mxfp4` 0.4%, `custom_mainloop_scheduling` 0%. Across the aggregate 8,457 of 116,264
solutions were rejected by the KernelWriter; their CPU share outside the probes was not measured.

#### All rejection reasons, and when each is known

Every kernel the probes recorded as rejected (1,242 kernels from `f8_gate_r`, `spmm_i8_sb`,
`plr_zero`, `gfx950/dtl`, `fp16_tn`, `subtile_mxfp4`) was replayed with a hook on
`StateValues.__setattr__` recording the first non-zero `overflowedResources` code, the time it was
set and the setting function (`/tmp/codegen/rejreasons.py`, output `/tmp/codegen/rejreasons.tsv`):

| Code | Reason | Kernels | Share of rejected-kernel CPU | Known at (median fraction of codegen time) | Set by | Caught by init VGPR check |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 1 | too many VGPRs | 679 | 83.6% | 0.50 | `checkResources` (`KernelWriterAssembly.py:2343`) | 598 |
| 2 | too many SGPRs | 427 | 15.0% | 0.03 | `overflowListener` in `allocTmpSgpr` (`KernelWriterAssembly.py:789`) | 5 |
| 5 | local read/write LDS scheduling | 132 | 1.3% | 0.21 | `_makeSubIterSchedule` (`KernelWriter.py:2011`) | 0 |
| 8 | not enough LDS (subtile) | 4 | 0.1% | 0.01 | `applyLdsLayout` (`SubtileLdsLayout.py:185`) | 0 |

No rejection came from any other exception. `overflowedResources` is never reset and is read only at
the end (to build the error and print the reason: `KernelWriter.py:5664`, `:7265`,
`KernelWriterAssembly.py:2368`), so once it is non-zero the kernel is guaranteed to be rejected
(unless `forceGenerateKernel`), yet generation continues: SGPR overflow is known 3% of the way in;
the VGPR flag is set at the end of `kernelBody`, after which `rocIsaPass`, `str()` and freeing the
tree (~50% of the kernel's time) still run. The 81 VGPR rejections the init check misses overflow
through registers allocated during generation.

**Second check: abort as soon as `overflowedResources` becomes non-zero.** Estimated CPU on rejected
kernels, from the recorded set times:

| Config | Rejected | Today | Init VGPR check | Abort on set | Both |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f8_gate_r` | 24 | 41.4 s | 0.0 s | 25.5 s | 0.0 s |
| `spmm_i8_sb` | 54 | 48.1 s | 0.0 s | 31.4 s | 0.0 s |
| `plr_zero` | 673 | 191.0 s | 57.1 s | 73.9 s | 2.7 s |
| `gfx950/dtl` | 457 | 90.2 s | 19.2 s | 44.0 s | 8.1 s |
| `fp16_tn` | 30 | 3.4 s | 0.0 s | 2.1 s | 0.0 s |
| `subtile_mxfp4` | 4 | 0.4 s | 0.4 s | 0.0 s | 0.0 s |
| **Total** | **1,242** | **374.5 s** | **76.7 s** | **177.0 s** | **10.8 s (−97%)** |

Validated with both checks (`ff.py`, `FF_ABORT_ON_SET=1`) on 9 complete steps covering every reason,
4,313 kernels: 0 accept/reject mismatches, accepted kernels' output size unchanged, CPU on rejected
kernels 160.1 → 4.3 s.

| Step | Kernels | Rejected | Step codegen CPU |
| --- | ---: | --- | ---: |
| `plr_zero` step 0 | 874 | 218 (SGPR) | 104.2 → 74.4 s (−29%) |
| `gfx950/dtl` step 4 | 336 | 136 (VGPR) | 75.4 → 32.3 s (−57%) |
| `plr_zero` step 13 | 156 | 92 (VGPR + SGPR) | 69.5 → 22.1 s (−68%) |
| `gfx950/dtl` step 11 | 2,424 | 64 (LDS scheduling) | 338.5 → 354.7 s (noise; 1.6 s of rejected work) |
| `gfx950/dtl` step 24 | 307 | 43 (LDS scheduling + VGPR) | 14.4 → 11.9 s (−17%) |
| `fp16_tn` step 1 | 38 | 30 | 3.6 → 0.6 s (−85%) |
| `spmm_i8_sb` step 1 | 80 | 24 | 27.2 → 6.4 s (−77%) |
| `f8_gate_r` step 0 | 8 | 6 | 10.4 → 1.0 s (−90%) |
| `subtile_mxfp4` step 0 | 90 | 0 | unchanged |

For a real implementation: the prototype raised from the `overflowedResources` setter and ignored
`forceGenerateKernel`; the real change should check that flag and raise at the setting sites. The
reason printed under `PrintSolutionRejectionReason` comes from `checkResources` at the end, so the
early exit must print it too; the error code (`-2`) is already the same. Moving the VGPR/SGPR checks
into solution validation, so these kernels never reach codegen, is a possible further step (not
tested).

### C. Generate all steps of a config together

Each `(problem type, size group)` step is generated, compiled and post-processed before the next
starts (`BenchmarkProblems.py`, loop around `_benchmarkProblemType`); no step reads another's
results (inference from the loop; build-only skips benchmarking). Simulating one 25-worker pool
per config with the measured per-kernel times (longest-first):

| Config | Sum of per-step spans (measured) | One pool (simulated) |
| --- | ---: | ---: |
| `custom_mainloop_scheduling` | 36.0 s | 2.1 s |
| `f8_gate_r` | 12.4 s | 3.3 s |
| `spmm_i8_sb` | 17.6 s | 9.1 s |
| `fp16_tn` | 14.4 s | 13.4 s |
| `gfx950/dtl` | 60.3 s | 54.1 s |
| `plr_zero` | 36.2 s | 32.5 s |

Aggregate bound (estimate): for the 447 ≤25-kernel steps outside the `CpuThreads: 1` configs, CPU ≤
kernels × step wall, so per config the pooled time is ≤ CPU/25 + longest step: 142.4 s → ≤75.2 s.
The probes suggest the real gain is larger. This needs the step loop split into "generate
solutions for all steps → one codegen pool → per-step assemble/build/write" (or keeping one pool
fed across steps); not prototyped.

### D. Longest-first dispatch

Within a step, joblib dispatches in input order. With perfect knowledge of per-kernel cost,
longest-first would save 0–6% on most probed configs and 24% on `spmm_i8_sb` (17.6 → 13.4 s,
which is also the lower bound). Needs a cost predictor (output size is not known before
generation); upper bound only.

### E. A cheaper generator (larger work)

Per-kernel shares above: Python tree construction ~65% (epilogue 40–64% of profiled time),
`rocIsaPass` ~21%, freeing ~8%, `str` ~6%. The code itself notes the epilogue size problem
(`KernelWriter.py:7102`: "Kernels with epilog especially with activation is too long (50000~
lines). Need to refactor global write elements."). The quadratic `hasAnyDependency` scan is a
contained target for the sparse configs. Not prototyped.

## Removing `CpuThreads: 1` in CI: cores, oversubscription and memory

### How CI runs the common tests

| Job | Command | pytest workers | CPUs |
| --- | --- | ---: | --- |
| TheRock `tensilelite-common` (`TEST_CATEGORY=hw-common`; gfx942 on PRs, gfx950/gfx90a postsubmit, gfx120X nightly) | `pytest_runner.py --numprocesses=4` (`test_categories.yaml`: `parallel_workers: 4`) | 4 | Kubernetes pod on a 1-GPU slice of an 8-GPU node (`linux-gfx942-1gpu-ccs-*`, `linux-gfx950-1gpu-ccs-ossci-rocm`); `KUBE_CPU_REQUEST=25` (from a real job log, see `CI_SPLIT_TIMING.md`) |
| Math CI `preliminary` | `tox -e py3 -- tensilelite/Tests -m common` | `TENSILE_NUM_PYTEST_WORKERS`, default 4 | not in this repo (internal Jenkins) |
| `.github/workflows/gfx1250_hipblaslt_tensilelite_test.yml` | `tox ... -n 4` in a container on `blas-rocm-mi455-sandbox` | 4 | not stated |
| FFM/emulation categories (`ffm-quick`, `emu-fast`, `emu-full`) | `pytest_runner.py` | 16 | not stated |

TheRock's `test_component.yml` turns `KUBE_CPU_REQUEST` into `OMP_NUM_THREADS` /
`OPENBLAS_NUM_THREADS` only. Tensile sizes its pools from `len(os.sched_getaffinity(0))`, which
ignores the Kubernetes request and the cgroup CPU quota, so each pytest worker starts either 25
processes (if the pod's CPUs are pinned) or one per host core (if not). Which one applies could not
be checked: no CI log was reachable (no GitHub credentials, API rate-limited). Any CI log answers it:
the `Generating assembly kernels: Launching N threads` line gives N per pytest worker.

### CI is already oversubscribed, regardless of `CpuThreads: 1`

Of the 540 YAMLs under `Tests/common`, 406 use the default `CpuThreads: -1` (all visible cores), 67
set 32, 2 set 8 and 65 set 1. Nothing in the test harness divides cores between pytest workers, so
with 4 workers and 25 visible cores up to 100 codegen processes compete for 25 CPUs whenever two
tests generate code at once. Removing `CpuThreads: 1` makes those 65 configs join that existing
pattern; it does not raise the worst case, which the default configs can already reach.

Cost of oversubscription, measured by replaying the same 1,000 `dtl` kernels on 25 pinned cores
(`/tmp/codegen/oversub.py`): 25 / 50 / 100 worker processes took 7.2 / 8.2 / 7.6 s (worker CPU
174.9 / 197.3 / 186.4 s). Throughput is unchanged within noise on this shared host; CPU rises
7–13%. Memory is the real cost.

### Per-worker memory

The pool is joblib's loky backend. Every worker is a fresh interpreter
(`python3 -m joblib.externals.loky.backend.popen_loky_posix`), not a fork of the parent, so each one
imports everything itself (537 modules); only the file-backed shared libraries (~38 MB) are shared.
Loky reuses the same workers across `ParallelMap2` calls in a build, so start-up is paid once per
build, not per step.

Single fresh process (`/tmp/codegen/memsteps.py`), resident memory:

| Stage | RSS |
| --- | ---: |
| bare Python | 10 MB |
| + joblib / numpy | 36 MB |
| + rocisa | 64 MB |
| + Tensile modules | 80 MB (38 MB shared libraries, ~42 MB private) |
| after one typical kernel (1.6–2.1 MB of assembly) | +1–2 MB |
| after the 7.0 MB `custom_mainloop_scheduling` kernel | peak 495 MB; still 355 MB after the result is dropped |

Peak memory scales with kernel size, about 60 MB of instruction tree per MB of assembly, and the
allocator does not return it to the OS, so a worker stays near the peak of its biggest kernel.
(An earlier 245 MB figure came from a harness that loaded the whole pickled step into every worker,
which real workers never do; it is superseded by the numbers below.)

Real loky workers, sampled once per second during real builds (`/tmp/codegen/memsample.py`):

| Build | Workers | Peak RSS median / max | Peak PSS median / max | Shared libs |
| --- | ---: | ---: | ---: | ---: |
| `fp16_tn` | 25 | 288 / 315 MB | 251 / 276 MB | 38 MB |
| `custom_mainloop_scheduling` | 29 | 308 / 569 MB | 259 / 531 MB | 38–40 MB |

One `fp16_tn` worker over the build: 122 → 196 → 252 → 281 → 295 MB. So 4 pytest workers × 25
loky workers ≈ 100 × ~250 MB ≈ 25 GB, more if workers hit very large kernels; if a pod exposed a
whole 192-core node, 4 × 192 workers would need ~190 GB. That exposure exists today for the
default configs.

Loky also has memory-leak protection (`joblib/externals/loky/process_executor.py`,
`_MAX_MEMORY_LEAK_SIZE = int(3e8)`): a worker whose memory grew more than 300 MB above its starting
size exits and is replaced. Observed in a `pytest -n 4` run: `f8_gate_r`'s workers exited at
513–596 MB after 10–30 s and were respawned (20 distinct pids for a 12-worker pool), each
replacement briefly overlapping the worker it replaced.

### Proposal: a total pool budget of 2× the available CPUs

Size every pool so that all pytest workers together start at most 2× the CPUs available, and a
single process never more than 1×:

1. CPUs available = min(affinity mask, loky `cpu_count()` (also applies the cgroup quota),
   `KUBE_CPU_REQUEST` when set).
2. Parallel workers = `PYTEST_XDIST_WORKER_COUNT` (set by pytest-xdist in every worker), else 1.
3. Pool size = max(1, min(CPUs, 2 × CPUs // workers)).
4. `CpuThreads` stays a cap: `-1` = the budget, N = min(budget, N), `0` = no threading.

| Setup | Per process | Total |
| --- | ---: | ---: |
| standalone, 96 CPUs | 96 | 96 |
| `-n 4`, 25 CPUs (TheRock / Math CI) | 12 | 48 |
| `-n 4`, pod sees 192 cores, `KUBE_CPU_REQUEST=25` | 12 | 48 |
| `-n 2`, 25 CPUs | 25 | 50 |
| `-n 16`, 128 CPUs (FFM) | 16 | 256 |

Sketch, replacing the body of `CPUThreadCount()` in `Common/Parallel.py`:

```python
OVERSUBSCRIPTION_FACTOR = 2

def _kubeCpuRequest():
    value = os.environ.get("KUBE_CPU_REQUEST")    # may be fractional, e.g. "8.0"
    try:
        return max(1, int(float(value))) if value else None
    except ValueError:
        return None

def availableCPUCount():
    if os.name == "nt":
        count = min(os.cpu_count() or 1, 61)
    else:
        count = len(os.sched_getaffinity(0))
        try:
            from joblib.externals.loky import cpu_count as lokyCpuCount   # affinity + cgroup quota
            count = min(count, lokyCpuCount())
        except Exception:
            pass
    request = _kubeCpuRequest()
    return max(1, min(count, request) if request else count)

def concurrentWorkerCount():
    try:
        return max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    except ValueError:
        return 1

def poolSizeBudget():
    cpus = availableCPUCount()
    return max(1, min(cpus, (OVERSUBSCRIPTION_FACTOR * cpus) // concurrentWorkerCount()))

def CPUThreadCount(enable=True):
    if not enable:
        return 1
    budget = poolSizeBudget()
    cpuThreads = globalParameters["CpuThreads"]
    return budget if cpuThreads == -1 else min(budget, cpuThreads)
```

This was prototyped and validated, then reverted (not part of this change):

- 15 unit tests (mocked affinity / loky count / environment) covering the table above, cgroup
  limits, unparsable environment values, `CpuThreads` caps and `0`, and that per-process × workers
  ≤ 2 × CPUs for six CPU/worker combinations; together with the existing
  `Tests/unit/characterization/Parallel/test_parallel_char.py`, 29 passed.
- Outside pytest with `PYTEST_XDIST_WORKER_COUNT=4` on 25 cores: `ParallelMap2` printed
  `Launching 12 threads` and used exactly 12 worker pids.
- Real `pytest -n 4` build of `fp16_tn`, `cgemm`, `zgemm`, `f8_gate_r` on 25 pinned cores, process
  tree sampled every 0.2 s (`/tmp/codegen/poolturnover.py`): 12 loky workers per pytest worker (13
  for a moment during a loky memory-leak restart, see above), at most 50 in total; all 4 passed.
  (Short-lived "loky" children of loky workers are forks for `amdclang++` before `exec`, which keep
  the parent's command line; they are not pools.)

Open decisions for this proposal: `KUBE_CPU_REQUEST` is a Kubernetes *request*, not a limit, but
TheRock already treats it as the pod's budget for OpenMP threads, and without it a pod that sees the
whole node would size pools from the whole node. Only pytest-xdist parallelism is detected; other
ways of running several Tensile processes on one machine are not.

### `CPUThreadCount()` / `ParallelMap2` issues

- `CPUThreadCount()` sizes pools from `len(os.sched_getaffinity(0))` only. It ignores cgroup CPU
  quotas, `KUBE_CPU_REQUEST`, and how many pytest-xdist workers run concurrently, so under
  `pytest -n N` each worker starts a full-size pool (the oversubscription above).
- `ParallelMap2` (`Common/Parallel.py`) computes
  `threadCount = procs if procs else CPUThreadCount(enable)` and on the next line overwrites it with
  `threadCount = CPUThreadCount(enable)`, so an explicit `procs` is ignored. The only caller passing
  it is `TensileLogic/Run.py:321` (`procs=jobs`), whose job count therefore has no effect.
- TensileCreateLibrary's `_cpuCount()` (`TensileCreateLibrary/Run.py:1226`) duplicates the
  affinity-only count to split jobs between architecture groups, with the same blind spots.

## Not measured / open items

- End-to-end aggregate with A/B/C applied; the aggregate numbers above are estimates from the probes.
- Whether `LdsNumBytes`/`LoopIters` in `Data/*.yaml` are consumed downstream (libraries and code
  objects are identical either way); which value is intended.
- CPU share of rejected kernels across all configs, and early-exit for the other overflow reasons.
- Whether any `CpuThreads: 1` config needs it for a reason not visible in the commit.
- How many cores Tensile sees inside CI pods (pinned 25 vs the whole node); read the
  `Launching N threads` line from any `tensilelite-common` / Math CI log.
- Step independence for option C was inferred from the loop, not proven for every config feature.
- Native (C++) hotspots inside `rocIsaPass`/`str`/free: `_rocisa.so` is stripped and `perf` is
  unavailable; only whole-call times were measured.
- Probe A/B timings ran on a shared host; comparisons are back to back on the same 25 cores except
  the `plr_zero` fail-fast run.
