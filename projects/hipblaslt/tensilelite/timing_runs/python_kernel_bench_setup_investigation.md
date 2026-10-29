# `python_kernel_bench_setup` investigation (gfx950 per-config build)

`full_runs_gfx950/per_config_build/AGGREGATE_analysis.txt` reports `python_kernel_bench_setup` at
131,638.45 ms (5.1% of build wall time) over 855 calls, mean 154 ms. This note explains what that
time is and how to remove almost all of it. Timing claims come from measurements (in-run timers,
offline replays of the exact inputs, and cProfile used only for attribution); code reading was
used to decide what to measure. Inferences are marked as such.

## Summary

The timer wraps one loop in `writeBenchmarkFiles` (`tensilelite/BenchmarkProblems.py:504`) that
builds the list of unique *helper kernels* (the HIP source kernels in `Kernels.cpp`: `PostGSU`
conversion, `BetaOnly`, activation, reduction). For every solution it calls
`initHelperKernelObjects(...)`, which constructs every helper object that solution would need,
then keeps only the ones whose name has not been seen yet.

- **Almost all constructed helpers are thrown away.** A benchmark step has one `ProblemType`, and
  the helpers depend only on it and on a handful of solution fields, so thousands of solutions
  produce the same 2–14 helpers. `fp16_tn` step 0: 4,096 solutions build **24,576** helper objects
  to keep **9**.
- **Each helper object costs two full `deepcopy(ProblemType)`s**: once in the `init*KernelObjects`
  function, and again in the `KernelWriter*` constructor. That is 3.4–37 deep copies per solution
  depending on the problem type (bias types × vector widths × helper kinds).
- **One `deepcopy(ProblemType)` costs ~150 µs in the live process**, ~90% of which is the 13–15
  `DataType` members each copying their private `properties` dict. `deepcopy` is ~90% of the block
  under cProfile.

| Fix | `fp16_tn` (in-run) | `fp16_HH_BHS_bf16mfma` (in-run) | Output |
| --- | --- | --- | --- |
| baseline | 8,821 ms | 11,485 ms | — |
| **F1** `DataType.__deepcopy__` returns `DataType(self.value)` | 4,996 ms (1.8×) | 8,049 ms (1.4×) | identical |
| **F2** call `initHelperKernelObjects` once per distinct input key | 107 ms (82×) | 135 ms (85×) | identical |
| **F1 + F2** | **96 ms (92×)** | **79 ms (146×)** | identical |

"Identical" means: the ordered list of helper objects passed to `writeSolutionsAndKernels` has the
same names, header text and source text (SHA-256 over all of them, every step), and every
non-`.hsaco` file in the build artifact (`TensileLibrary.dat` decompressed, `cache.yaml`, `.co`,
`Data/*.yaml`, and the cache directory names, which are content hashes) is byte-identical to the
baseline run. `.hsaco` helper objects differ between any two builds (random `__hip_cuid_*`); their
input source is covered by the helper-source hash.

**Projected effect on the aggregate (estimate, not measured end to end):** F2 leaves ~25–65 µs per
solution in-run (measured on the two configs above, including building the unique helpers);
117,270 solutions × that ≈ 3–8 s, plus the 314 steps with ≤4 solutions (1.9 s today, which F2
cannot shrink and F1 roughly halves). So ~131.6 s → roughly 5–10 s.

## Method

1. **Distribution.** The per-config build logs
   (`/home/alvasile/timing_runs/full_runs_gfx950/per_config_build/<cfg>/<cfg>_build.log.gz`) print
   one `TIMING:python_kernel_bench_setup` line per benchmark step at the end of each run, in step
   order, and one `# Actual Solutions: N / M after SolutionStructs` line per step; the *n*-th of
   each were paired (`N` is the list passed to `writeBenchmarkFiles`). All 855 timers paired; the
   8 `ductile` configs print no solution count and were left with an empty count (2.6 s total).
   Output: `/tmp/benchsetup/steps.txt`.
2. **Capture.** A pytest plugin (`/tmp/benchsetup/bsdump.py`, loaded with `-p bsdump` by
   `/tmp/benchsetup/run_dump.sh`, the same build-only `test_config_build` invocation as the
   aggregate runs) wraps `BenchmarkProblems.writeBenchmarkFiles` and `timing_context`. Per step it
   pickles the exact `solutions`, `isaInfoMap`, assembler path and `debugConfig`; optionally
   cProfiles the block; and records wall time, main-thread CPU time (`time.thread_time`), GC pause
   time (`gc.callbacks`), live threads and in-process `deepcopy(ProblemType)` cost. It also hashes
   every helper object's name + header + source as passed to `writeSolutionsAndKernels`.
   Configs captured: `fp16_tn`, `cgemm`, `zgemm`, `gfx950/dtl`, `fp16_HH_BHS_bf16mfma`,
   `gfx950/bf16_gradient_bias`, `gfx950/spmm_f16_sb` (45 steps).
3. **Replay.** `/tmp/benchsetup/replay.py` re-runs the exact loop on a pickled step, counts objects
   and `deepcopy` calls, and times candidate replacements with an exact output check (ordered list
   of name, header text, source text).
4. **In-run A/B.** The plugin can apply F1 and/or F2 by monkeypatching (no repo edits);
   `BSDUMP_NOPROF=1` disables the profiler. Each config ran baseline / F1 / F2 / F1+F2 back to back
   on the same idle cores (`fp16_tn` on 87–104, `fp16_HH_BHS_bf16mfma` on 160–177);
   `/tmp/benchsetup/compare.py` diffs timers, helper hashes and artifacts.

## Distribution in the aggregate

Cost scales with the number of solutions in a step: 117,270 solutions in 855 steps, **1.1 ms per
solution** overall.

| Solutions in step | Steps | Solutions | Time |
| --- | ---: | ---: | ---: |
| 1–4 | 314 | 571 | 1.9 s |
| 5–50 | 340 | 6,315 | 12.0 s |
| 51–500 | 148 | 24,494 | 30.1 s |
| >500 | 53 | 85,889 | 85.8 s |

Top configs: `cgemm` 17.4 s (22,880 solutions), `gfx950/dtl` 15.4 s, `bufferLoadStore0` 9.8 s,
`zgemm` 9.3 s, `fp16_HH_BHS_bf16mfma` 8.9 s (2,048 solutions, 4.3 ms each), `spmm_f16_sb` 8.0 s,
`fp16_tn` 7.8 s. Per-solution cost ranges from ~0.5 ms (`zgemm`) to ~4.3 ms
(`fp16_HH_BHS_bf16mfma`); steps with 1–2 solutions in the gradient/bias configs still cost
~9–33 ms.

## Where the time goes

### What the loop builds versus what it keeps (replay, exact counts)

| Step | Solutions | Helper objects built | Unique kept | Top-level `deepcopy`/solution | Distinct `ProblemType`s | Distinct F2 keys |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16_tn` step 0 | 4,096 | 24,576 | 9 | 16.4 | 1 | 4 |
| `fp16_HH_BHS_bf16mfma` step 0 | 2,048 | 25,600 | 12 | 35.5 | 1 | 2 |
| `spmm_f16_sb` step 0 | 2,608 | 15,648 | 9 | 24.2 | 1 | 4 |
| `cgemm` step 0 | 2,480 | 6,820 | 3 | 5.4 | 1 | 4 |
| `zgemm` step 0 | 2,112 | 3,696 | 2 | 3.4 | 1 | 4 |
| `gfx950/dtl` step 3 | 1,984 | 6,944 | 4 | 6.0 | 1 | 2 |
| `bf16_gradient_bias` step 0 | 2 | 28 | 14 | 37.0 | 1 | 2 |

Every one of the 45 captured steps had exactly one distinct `ProblemType` and 1–4 distinct F2 keys.

### The profile

cProfile of `fp16_tn` step 0 in-run (26.6 s profiled versus 8.6 s unprofiled; used only for
attribution): `initHelperKernelObjects` 26.4 s cumulative, of which `copy.deepcopy` 24.3 s
(65,536 top-level calls recursing into 17.2 M). Split by helper kind: conversion (`PostGSU`)
12.7 s, `BetaOnly` 4.2 s, activation enum header 4.0 s, activation function 4.0 s. The per-call
`sorted(result, key=lambda x: ("Enum" in x.getKernelName(), result.index(x)))` is 1.5 s.
`KernelWriterAssembly(...)` and `cmdLineArchsFor` do not appear (negligible).

### Why one `deepcopy(ProblemType)` is expensive

`ProblemType.state` has 90 keys; 13–15 values are `DataType` objects. A `DataType` is an int plus
`properties`, a pointer to a row of the class-level table `DataType.properties`
(`Common/DataType.py:63-77`). `deepcopy` cannot know that row is shared and immutable, so every
copy duplicates the dict; after the first copy each `DataType` owns a private dict, and later
copies duplicate those too.

| Measurement | Value |
| --- | ---: |
| `deepcopy(ProblemType)` in the live build process (`fp16_tn`, two runs) | 151.5–155.5 µs |
| same object after a pickle round trip (replay) | 72–80 µs (fp16/zgemm), 112–130 µs (cgemm/dtl) |
| `deepcopy(DataType)` (replay) | 7.3 µs, of which the `properties` dict 2.45 µs |
| replay `deepcopy(ProblemType)` with `DataType.__deepcopy__ = DataType(value)` | 41.6 µs (from 77.5) |
| replay, `__deepcopy__` returning `self` | 36.3 µs |

The replay understates the in-run cost about 2×. In the live process `dt.properties is
DataType.properties[dt.value]` is `False` for the `ProblemType`'s members (verified in-run): they
were deep-copied earlier, so 13 `DataType`s carry 13 private dicts. Pickling a solution list
re-shares them (3 distinct dicts for 13 `DataType`s in the replayed object), and `deepcopy`'s memo
then copies each shared dict once. All speedups in the summary table are therefore from in-run
measurements; replay numbers are only used for counts and equivalence.

### Ruled out

- **GC:** in-run, GC pauses inside the block are 147–167 ms of 8.6–8.7 s (1,026–1,029 collections)
  for `fp16_tn`, 309 ms of 11.5 s for `fp16_HH_BHS_bf16mfma`.
- **GIL / descheduling:** main-thread CPU time equals wall time to within 0.1 ms on every large
  step. Live threads during the block are only `MainThread` plus the
  `concurrent.futures` `ExecutorManagerThread`/`QueueFeederThread`.
- **`KernelWriterAssembly` construction, `getKernels()`:** not visible in the profile; F2 keeps
  both and the remaining in-run time is 25–66 µs per solution.

## Fixes

### F2. Build helpers once per distinct input (the main fix)

`initHelperKernelObjects` and the functions it calls read only these solution fields:
`ProblemType`, `GlobalSplitU`, `GlobalSplitUAlgorithm`, `isStreamK(solution)`, `StreamKAtomic`,
`_GlobalAccumulation`, `AdaptiveGemmGSUA`, `ActivationFused`, `WavefrontSize`, `ISA` (plus the
global `internalParameters["GlobalSplitUPGR"]`). The prototype memoizes the result on

```python
key = (kind, index of an equal ProblemType in a small per-step list,
       s["GlobalSplitU"], s["GlobalSplitUAlgorithm"], isStreamK(s), s.get("StreamKAtomic"),
       s["_GlobalAccumulation"], s.get("AdaptiveGemmGSUA"), s["ActivationFused"],
       s["WavefrontSize"], tuple(s["ISA"]))
```

and returns the cached list for a repeated key; the existing name-based dedupe then skips them, so
the kept list and its order are unchanged. `ProblemType` equality (`__eq__` compares the state
dicts) is checked against a list that in practice holds one entry.

Measured: in-run 82× (`fp16_tn`) and 85× (`fp16_HH_BHS_bf16mfma`); replay 1–180× across all 45
captured steps (smaller ratios only on steps with very few solutions), identical output on all 45.

Implementation notes (inference): the key must be kept in sync with the fields the helper
writers read; a safer variant is to compute the key inside `KernelHelperNaming` next to the code
that reads the fields. The memo should be scoped to one `writeBenchmarkFiles` call.

### F1. Copy `DataType` by value

```python
def __deepcopy__(self, memo):
    return DataType(self.value)
```

A freshly constructed `DataType` is what the code makes everywhere else, and its `properties`
points at the shared class row again. Grep found no code that assigns `.value`, `.properties` or
`properties[...]` on a `DataType` instance outside `__init__`. Measured: in-run 1.4–1.8× on this
block alone, and it is the only fix that helps the 1–4-solution steps (where every helper built is
unique). Because it changes every `deepcopy` that touches a `DataType`, it also affects other
stages; in one run each, `python_benchpost_lib_construction` dropped 3,416 → 2,938 ms (`fp16_tn`)
and 2,647 → 2,327 ms (`fp16_HH_BHS_bf16mfma`), with byte-identical artifacts. Other timers moved
within run-to-run noise; those side effects were not measured with repeats.

### Smaller items (not needed once F2 is in)

- Each helper is deep-copied twice (`init*KernelObjects` copies, mutates, then the
  `KernelWriter*` constructor copies again). Removing one copy would halve the cost; not measured
  separately because F2 makes it irrelevant here.
- The per-solution `sorted(..., key=... result.index(x))` calls `getKernelName()` and an O(n)
  `index` for every element (1.5 s profiled in `fp16_tn` step 0); with F2 it runs once per key.

### Not recommended: reuse `TensileCreateLibrary.generateKernelHelperObjects`

TensileCreateLibrary already avoids most of this cost: `generateKernelHelperObjects`
(`TensileCreateLibrary/Run.py:897`) computes names with cheap `*KernelNames` functions and only
builds objects when a name is new. On the replayed steps it was 3–69× faster and produced the same
*set* of helpers, but in a **different order** (it sorts the enum header first globally), which
changes `Kernels.cpp` text and therefore the helper-cache key. More importantly, its name functions
do not compute the same names as the objects (verified on the replayed solutions):

| Solution | `conversionKernelNames` | names of built objects |
| --- | --- | --- |
| `fp16_tn` step 0 | `..._PostGSU1_VW1/2/4` | `..._PostGSU16_VW1/2/4` |
| `cgemm` step 0 (`ComplexFloat`) | `Cijk_SC_PostGSU1_VW1/2/4` (3) | `Cijk_SC_PostGSU16_VW1/2` (2) |
| `zgemm` step 0 (`ComplexDouble`) | 3 names | 1 object |

The names use the solution's `GlobalSplitU` rather than `GlobalSplitUPGR`, and the vector-width
lists disagree (`isDouble()` versus `numBytes()`); `MultipleBufferSingleKernel` also ignores
`AdaptiveGemmGSUA`. The function dedupes on these proxy names and aligns them with the objects
positionally (`itertools.compress`), which happened to give the right set on every captured step
but is not guaranteed to. This is also worth a separate look for the TensileCreateLibrary path
itself (not investigated further here).

## Not measured / open items

- End-to-end aggregate after the fixes (the projection above is an estimate from two configs).
- F1's effect on `python_kernel_codegen`, `python_solution_generation` and the other stages that
  deep-copy solutions, with repeats; single runs showed no change beyond noise except
  `python_benchpost_lib_construction`.
- Whether the TensileCreateLibrary name/object mismatch above can drop or misselect a helper for
  some configuration.
- Baseline in-run times in these A/B runs (8.8 s, 11.5 s) are higher than the same steps in the
  aggregate (7.7 s, 8.9 s): different cores and host load. Comparisons are only made within one
  core set and back-to-back runs.
