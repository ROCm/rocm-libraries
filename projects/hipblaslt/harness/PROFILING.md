# Profiling and GEMM benchmark guide

Read this before the tier 2 perf A/B, including runs for StinkyTofu changes that are tested through
TensileLite. Measurement rules for performance runs. They apply to GEMM benchmarks and profiling only.
They are not rules for every GPU job. End-to-end runs cannot be serialized this way.

## Required fields in a tier 2 result

```text
baseline_commit, candidate_commit, gpu, runtime/driver version, clocks (if controllable),
repeat_count, noise_sd, benchmark_set, pattern_check_result
```

A tier 2 result missing any of these is incomplete.

## Exclusive GPU

A second job on the same GPU distorts timing. For a benchmark or profiling run:

```text
- nothing else runs on the target GPU for the whole measurement
- check first that the GPU is idle (a busy-process check, not just utilization)
- leave a short idle gap between runs
- if the GPU is held by someone else, stop and ask the developer
```

## Comparable results

```text
- compare old and new only in the same time window, on the same GPU, at the same clocks
- rebuild the baseline from the commit you compare against, and record its hash
- repeat each measurement and compute the SD of the run-to-run noise for that benchmark on that machine.
  Measure it fresh. If the database has a noise entry for this benchmark and machine, use it only as a
  rough guide to how many repeats you need.
- a perf difference of at least one SD needs a human check. Below one SD it is noise.
- record the fields listed at the top of this file with every result
```

## Order

```text
1. tier 1 first (CPU). Run tier 2 only for configs where tier 1 found a difference
   and no confirmed bad pattern.
2. correctness of the kernel under test before its performance. Run the candidate kernel with the
   client's validation on (`NumElementsToValidate` in the YAML; the perf YAMLs already validate the
   winner with `NumElementsToValidateWinner`). This checks the kernel being tested, not the whole CI.
   A kernel that gives wrong results has no meaningful perf number. Keep validation off for the timed
   runs.
3. benchmark the set together, not one case: a change must not regress the cases that already
   work well while improving the target case.
4. print the plan or the reason behind a result before concluding from it.
```

## SQTT capture

```text
1. Use the profiler that ships with ROCm (rocprofv3 --att, with the rocprof-trace-decoder). See the
   ROCprofiler-SDK "Using thread trace" docs:
   https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html
   Capture one kernel, with the same shape as the benchmark case, narrowed with
   `--kernel-include-regex` and `--kernel-iteration-range`.
2. If the profiler or the decoder is missing, or does not support your target, get it from its
   source repository. The public source is ROCm/rocm-systems on GitHub
   (`projects/rocprof-trace-decoder`). If the developer or the
   database names a different source, use that one. If neither does, stop and ask.
3. View the result with ROCprof Compute Viewer.
4. Setup that depends on the target or the machine, and the analysis (what to look for and what
   it means), are in the database (`$HARNESS_DB`). If it has a profiling procedure for your target,
   follow that. It adds detail to steps 1 to 3.
```

## Reporting

Where results go (PR body, tracker) follows the component's PR rules (`../AGENTS.md`,
`../TESTING.md`) and the developer's choice. Show the developer the exact text before posting.
