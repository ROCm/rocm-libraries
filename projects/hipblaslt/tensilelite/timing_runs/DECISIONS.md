# Timing investigation decisions

Record of the decisions taken on each investigation in this directory, one topic at a time.
It is the input to the implementation plan at the end of this file.

Status values: `pending` (not discussed yet), `discussing`, `decided`.

Deferred and "not for now" work is tracked separately in [DEFERRED.md](DEFERRED.md).

## Agenda

Ordered by measured cost, run side first (gfx950 `per_config_rerun_2`), then build side
(gfx950 `per_config_build`).

| # | Investigation | Cost in aggregate | Status |
| --- | --- | --- | --- |
| 1 | [CPU reference slow path](cpu_reference_slow_path_investigation.md) | 741 s slow path (of 1,080 s Reference GEMM, 42% of run wall) | decided |
| 2 | [Input data preparation](input_data_preparation_investigation.md) | 503 s (19.6%) | decided |
| 3 | [SynchronizerValidator](SynchronizerValidator.md) | 163 s (6.3%) | decided |
| 4 | [`cpu_data_init`](cpu_data_init_investigation.md) | 107 s (4.2%) | decided |
| 5 | [Per-solution bookkeeping](per_solution_bookkeeping_investigation.md) (incl. correctness-only runs) | 68 s (2.7%), plus correctness-mode items | decided |
| 6 | [`python_kernel_build_src_co`](python_kernel_build_src_co_investigation.md) | 1,008 s (39.1% of build wall) | decided |
| 7 | [`python_kernel_codegen`](python_kernel_codegen_investigation.md) | 593 s (23.0%) | decided |
| 8 | [`python_kernel_bench_setup`](python_kernel_bench_setup_investigation.md) | 132 s (5.1%) | decided |

## Cross-cutting items

These show up in more than one note. Each is decided once, in the topic listed.

| Item | Appears in | Decide in |
| --- | --- | --- |
| `KMP_BLOCKTIME=0` / idle OpenMP workers spinning | 1, 5 | 5 (decided: adopt, in the client) |
| GPU wait misattributed to the next blocking call (warmup sync placement, `SyncsPerBenchmark: 0`) | 2, 3, 5 | 3 (decided: syncs only under timing instrumentation) |
| `copyInputs` / `copyInputBuffers` sized by `maxElements` | 2, 4 | 2 (decided: adopt) |
| Redundant MX regeneration and the CPU reference recompute it triggers | 2, 4 | 2 (decided: adopt) |
| `copyInputs` restoring read-only inputs vs only outputs + Synchronizer | 2, 3 | 2 (decided: not now) |
| Pool sizing (`CPUThreadCount`, oversubscription under xdist) | 7 (and every build-side pool) | 7 (decided: 2× CPU budget) |

## Decisions

### 1. CPU reference slow path

Status: decided.

Adopted:

- **A.** In the slow path (`ReferenceSolution<…>::SolveCPU`), enter `#pragma omp critical` only
  when `problem.outputAmaxD()` is true, and drop the per-element `omp_set_num_threads`.
- **A+.** Replace the AMAX-D critical section with a per-thread maximum combined once
  (`reduction(max:…)` or equivalent). Replace the hard-coded `MAX_OMP_THREADS 64` cap with
  `omp_get_max_threads()`, so the reference follows `OMP_NUM_THREADS`.
- **B.** Fast-path (`solveCPUFast`) support for:
  - Int8 inputs, with outputs I8, I32, Half and BF16. Accumulation must be exact (a float
    accumulator is not exact above 2²⁴).
  - Float8 outputs (`F8 × F8 → F8`, BF8 variants, and `F8 × Half → F8` with the mixed-input
    shadow).
  - Write-back must match the slow path's `SaturateCast<DType>`.
- **C.** Fast-path support for the smaller rejection reasons: `gradient` (including the bias
  reduction), `useE`, MX K not a multiple of the MX block (partial last block), and complex types.
  `NumElementsToValidate: 128` stays on the slow path, as intended.

Constraint: the new fast-path support must follow the established fast-path patterns (as in the
FP8/FP4, f64 and tf32 additions): `isFastPathEligible`/`solveCPUFast` in `Reference.cpp`,
fast/slow `CPUGemm.*` CTest pairs through `cpu-gemm-driver` in `tests/CMakeLists.txt`, and
gtest cases in `tests/ReferenceFastPath_test.cpp` / `ReferenceMXFastPath_test.cpp`.

Not adopted / open:

- **D** (making the fast-path rejection reason observable) was not chosen.
- **Int8 accumulation (decided): integer accumulation, then the slow path's epilogue type.**
  Accumulate the K loop in `int32_t` (exact, as the GPU's int8 MFMA does), then apply alpha,
  beta, bias, activation and the saturating cast in the slow path's accumulator type for that
  type id: `int32_t` for `I8_I8_I32` / `I8_I32_I32` (int32 alpha), `float` for `I8_I32_S`,
  `I8_I8_S`, `I8_H_S`, `I8_B_S` (float alpha). This needs the fast path to separate its
  multiply-accumulate type from its epilogue type (a new template parameter, in the style of
  `OperandMathOpT`). Reusing the `double` instantiation was considered and not chosen.
- `KMP_BLOCKTIME` is decided under topic 5.

### 2. Input data preparation

Status: decided.

Adopted:

- **Fix 1. Skip unchanged per-solution uploads (`copy_valid_to_gpu`).** On the slow reset path,
  skip the H2D upload of host `valid` data to GPU `valid` buffers when the host data has not
  changed since the last upload (tracked with `m_inputEpoch`).
- **Fix 2. Cache the batched-pointer arrays (`batched_inputs`).** Build and upload them per
  problem; re-upload only when the epoch or a buffer base address changes, not per solution.
- **Fix 4. No MX regeneration per solution.**
  - Data and canonical scale are generated once at problem level and never regenerated per
    solution.
  - The GFX950 swizzled scale is produced by permuting the canonical scale
    (`preSwizzleScalesGFX950` in mxDataGenerator's `PreSwizzle.hpp`), once per problem, and
    cached, ideally in its own GPU buffer so solutions using the canonical and swizzled layouts do
    not overwrite each other.
  - `m_inputEpoch` is not bumped for it, so the second CPU reference computation goes away.
  - Required check: the permuted scale is byte-identical to today's
    `generateMXInput(…, GFX950)` output, including padding and the K-fast restride.
- **Shared fix. Size copies by the current problem.** `copyInputs` / `copyInputBuffers` (and the
  per-problem `copy_valid_to_gpu`) copy the current problem's allocated extent instead of
  `maxElements`.

Not adopted:

- **Fix 3 (restore only writable tensors in `copyInputs`): not doing it for now.** Too easy to
  get the writable set wrong.

Measurement: no per-topic measurement; covered by the per-phase aggregate reruns in the
implementation plan.

### 3. SynchronizerValidator

Status: decided.

Adopted:

- **Item 1. GPU check.** An OR-reduction kernel over the Synchronizer buffer, enqueued on the
  client's `stream`, followed by a 4-byte readback. Only when the result is nonzero: a full readback
  and the existing scan to build the diagnostic, mark the solution `FAILED`, and clear the buffer.
- **Item 1-alt. Vectorized host check, kept as the FFM-emulator path.** Full readback, then an
  OR-reduction first (auto-vectorizes), with the count/first-offset scan only when dirty. Used when
  running under the FFM emulator (`HSA_MODEL_MEMFILE`), where the GPU kernel's cost is unknown. To
  be tested and tuned later based on measured performance.
- **Diagnostic unchanged.** The dirty-buffer message (nonzero count, first offset) stays as it is,
  built from a full readback only on the dirty path.
- **Item 2. Move the check to `postSolution`**, so it sees every launch of the solution (all
  warmups and benchmark runs) on both reset paths, including the last solution's. Requires saving
  the `ProblemInputs` pointer (from `preSolution` / `validateWarmups`). Ordering correction (from
  the plan review, verified in `MetaRunListener.cpp`): `postSolution` runs listeners in reverse,
  so `SynchronizerValidator::postSolution` runs *after* `ReferenceValidator::postSolution`, which
  is what lets a dirty-buffer `FAILED` override the reference verdict. Keep the registration order
  unchanged and do the check inside `SynchronizerValidator::postSolution`. With guards:
  1. **Only check if the solution launched something.** Set a flag in `preWarmup` /
     `preEnqueues`; keep the existing `m_mayUseSynchronizer` gate. `postSolution` also runs for
     rejected and never-run solutions.
  2. **Exception path.** If the device is usable, check as normal. If the device is lost, catch
     the check's own HIP error and only mark the solution `FAILED`; never throw out of
     `postSolution`.
  3. **Explicit, separately timed sync before the check**, so queued warmups 1..N-1
     (`SyncsPerBenchmark: 0`) are not absorbed by the check. Only when timing instrumentation is
     on (see "Attribution syncs" below); otherwise the check's own readback does the waiting.
  - Keep clear-on-dirty, so every checked solution ends with a clean buffer.

Not adopted:

- Adding a Synchronizer reset to the fast reset path (`resetOutput`). Not needed with the
  `postSolution` check.

Known limitations (accepted):

- Solutions classified as unable to use the Synchronizer are not checked (as today).
- **Correction (plan review, verified):** not every solution has one run iteration.
  `BoundsCheck: 4` (GuardPageAll, e.g. `bufferLoadStore0`) sets `m_numRunsPerSolution = 2`
  (`DataInitialization.cpp`); the second iteration's `prepareGPUInputs` restores the
  Synchronizer and wipes the first iteration's residue before `postSolution`. Today's
  `validateWarmups` check runs in each iteration. **Decided:** run the check at the end of every
  run iteration (before the next iteration's input reset), and report the verdict in
  `SynchronizerValidator::postSolution`. Solutions with one iteration behave exactly as planned;
  GuardPageAll solutions get both iterations checked.
- Cache effects of the reduction kernel on timed runs: resolved. In `postSolution` it runs after
  the solution's timed runs, never between its warmups and timed runs.

- **Item 3. Wait attribution after warmup 0, only when timing instrumentation is on.** In
  `main.cpp`, after launching warmup 0 and before `listeners.validateWarmups`, wait for the stream
  (or warmup 0's stop event) under its own timer, **only if `g_timingInstrumentationEnabled`**
  (`--timing-instrumentation`). Its only purpose is attribution: without it,
  `ReferenceValidator`'s `validate_gpu_readback` absorbs the kernel's runtime once the Synchronizer
  check moves to `postSolution`. When instrumentation is off, no extra block is added, so the
  launch/host overlap is unchanged.
  - Accepted side effect: instrumented runs block where uninstrumented runs do not, so their
    schedule differs slightly from production runs.

- **Attribution syncs, only when timing instrumentation is on.** All explicit syncs added for
  attribution follow the same rule as item 3:
  - the wait after warmup 0 (item 3);
  - guard 3, the sync before the Synchronizer check;
  - **an end-of-iteration sync in `main.cpp` for every run iteration that launched** (one per
    solution except GuardPageAll, which has two), under its own timer, so warmups 1..N-1 of
    `SyncsPerBenchmark: 0` configs are no longer charged to the next solution's
    `gpu_input_reset` (the `batched_inputs` inflation from topic 2). For checked solutions this
    is the same sync as guard 3.

Noted, no action:

- `BenchmarkTimer::validateWarmups` (`--sync-after-warmups`) waits on the last warmup's stop
  event, which is not yet recorded when `NumWarmups > 1`, so it returns immediately (code
  reading). It does not affect benchmark results: timed runs either record their start event on
  the stream after the warmups (GPU timer) or call `hipDeviceSynchronize` first (host timer).

### 4. `cpu_data_init`

Status: decided.

Adopted:

- **Item 1. Skip the discarded MX reference floats.** Add an opt-out to `generateMXInput`
  (`clients/common/src/mxDataGen.cpp`) so it does not build (and free) the `getReferenceFloat()`
  vector. Only the TensileLite client opts out; hipBLASLt's own clients (`testing_matmul.hpp`,
  `fast_check_gtest.cpp`) use the returned reference, so the default stays unchanged.
- **Item 2a. Cheaper trig init loop, same values.** Hoist the per-element `std::vector`
  allocation, and use a flat index for packed strides instead of `CoordNumbered` +
  out-of-line `TensorDescriptor::index`. Generated values must stay identical.
- **Item 5. Dynamic OpenMP schedule for the init loops** (`schedule(dynamic, chunk)`). Values do
  not depend on the schedule: Trig/Serial values depend only on the element index, and Random init
  already uses unseeded per-thread generators.
- `copyInputs` sizing by `maxElements` (finding 3): covered by the shared fix in topic 2.

Not adopted:

- **Item 2b. `sinf` / argument reduction in trig init.** Keep double `sin` for now, so the generated
  data does not change.
- **Item 3. Removing the dead `m_cpuInit` branch.** Leave it as is.

Deferred:

- The MX generator's other costs (single-threaded `resize` zero-fill, 7–11× scaling on 25
  threads) in shared hipBLASLt client code.
- First-call fixed cost per client process (finding 4): the note has no fix.

### 5. Per-solution bookkeeping and correctness-only runs

Status: decided.

#### 5a. Per-solution bookkeeping

Adopted:

- **A. Stop idle OpenMP workers from spinning.** Call `kmp_set_blocktime(0)` in the client's
  `main.cpp` unless `KMP_BLOCKTIME` is already set in the environment. Covers every client run.
  Also expected to speed up the CPU reference (`cpu_reference_gemm` 2–5× faster on dtl with
  `KMP_BLOCKTIME=0`). This settles the cross-cutting `KMP_BLOCKTIME` item.
- **C. No predicate-tree dump by default.** Gate `debugEval` in `SolutionIterator::checkSolution`
  behind its own client option, default off, driven by a Tensile global parameter. Per-solution
  rows are unaffected.

- **D1. Stop the per-solution stdout CSV row in test runs.**
  - `LogReporter::postSolution` checks whether the row's level will be printed *before* copying
    the row (`readCurrentRow`), so a suppressed row costs almost nothing.
  - Test runs use `ClientLogLevel` Error (0): only FAILED rows and mismatch lines are printed.
    Manual benchmarking runs keep the current default and their per-solution table.
  - **How a run is identified as a test run:** the `tensile_args` fixture in
    `Tests/conftest.py` always appends `--global-parameters ClientLogLevel=0`, placed *before*
    any user-supplied `--global-parameters` so a developer can still override it (e.g.
    `ClientLogLevel=3`; Tensile applies repeated `--global-parameters` in order, last wins, and
    they override the YAML). No Tensile code change. Covers `common/` and `custom/` YAMLs.
    A dedicated `TestRun` global parameter was considered and not chosen: nothing else needs a
    "this is a test" signal.
  - **Test:** move the body of `tensile_args` into a plain helper and unit-test it:
    `ClientLogLevel=0` is present, and a user-supplied `--global-parameters ClientLogLevel=3`
    comes after it. (A `LogReporter` gtest and an end-to-end check were considered and not
    chosen.)
- **D2. Skip the granularity/memory values when nothing consumes them.**
  `BenchmarkTimer::preSolution` does not compute or report its 9 granularity/memory values unless
  the stdout row will be printed or `CSVExportExtraCols` is set.
- The results-file CSV (`ResultFileReporter`) is handled in 5b (write it only when consumed).

Not adopted:

- **B. Cheaper formatting** (`to_chars`, no row copy, `'\n'` + per-problem flush). Dropped in
  favour of not producing the output; only the D1 level check is kept.

Deferred:

- **D3. Typed reporter keys** in `MetaResultReporter`.

#### 5b. Correctness-only runs

Adopted:

- **E. Infer "nothing is timed" in the client.** Skip `projectedPerformance` and the per-solution
  performance calculation; report the NaN values directly, reproducing today's `-nan` text. Rule
  (static, from the client's own arguments): nothing is timed when `num-syncs-per-benchmark == 0`,
  or `max-enqueues-per-sync == 0`, or (`num-enqueues-per-sync == 0` **and**
  `min-flops-per-sync == 0`). Note: prototype `corr`'s `num-enqueues-per-sync == 0` alone is wrong.
  Per-solution run-time skips (restore-from-log, skip-slow) are unchanged. D2 falls under this.
- **F. No warmup timing events when nothing is timed.** Do not create the per-kernel start/stop
  `hipEvent`s; only `BenchmarkTimer` reads them. The topic 3 warmup-0 attribution sync uses
  `hipStreamSynchronize` so it does not depend on events.
- **G. `gemm/gfx950/xfp32.yaml`: drop `NumWarmups: 5` and `EnqueuesPerSync: 10`.** The two
  work-stealing configs keep their deliberate extra warmups (now checked by the `postSolution`
  Synchronizer check).
- **H. Copy the results CSV only if it exists** (`BenchmarkProblems.main`, same guard as the
  granularity CSV).
- **I. Write the results CSV only when consumed** (the YAML has a `LibraryLogic:` section or uses
  the Ductile backend). Otherwise Tensile passes "no results file" and the client has an explicit
  path that never constructs `ResultFileReporter`.
- **J. Extend the `SyncsPerBenchmark` guard test** (`Tests/unit/test_common_yaml_syncs_per_benchmark.py`)
  to `Tests/custom/`, and set `SyncsPerBenchmark: 0` in the 12 custom YAMLs.
- **K. Remove `LibraryLogic:` / `LibraryClient:` from common YAMLs**, with a whitelist test (same
  style as the guard test: mandatory reason, stale entries fail). Scope: `common/` and `custom/`.
  Whitelist the 6 files that test the logic/library path with real benchmark data
  (`gemm/fp8nfp16mix_hfp8ns`, `gemm/xfp32`, `groupedgemm/grouped_gemm`,
  `groupedgemm/gfx11/grouped_gemm_gfx11`, `groupedgemm/grouped_gemm_ck_gfx942`,
  `sparse/gfx94x/libray_logic`). Remove from the 14 `ductile/*`, 10 `gemm/gfx950/*mxfp8*` and 12
  `custom/*` files.
  - **`Prediction` coverage: replace the 10 mxfp8 sections with one new, minimal, non-MX
    `Prediction` config** (option c). The 10 gfx950 mxfp8 configs are today the only end-to-end
    coverage of: writing a `Prediction`-type logic file (LibraryLogic step) → building a library
    from it → the client selecting a solution through Origami (`ProblemPredictionLibrary::
    findTopSolutions` → `origami::rank_configs`). `Prediction` ships in 9 hipBLASLt logic files
    (gfx1170 ×6, gfx1201 ×3, all `Origami/`), all non-MX, while the mxfp8 tests pass
    `a/b_mx_block_size = 0` to Origami, so they do not exercise it as shipped.
    - New YAML: a non-MX type (e.g. BF16 or F16), as small as possible: very small problem sizes
      in both the benchmark step and `LibraryClient`, minimal solution set, numerical validation
      off (`NumElementsToValidate: 0`); it only has to exercise logic write → library build →
      Origami selection.
    - It is added to the K whitelist (reason: only end-to-end `Prediction` coverage); the 10 mxfp8
      configs lose their `LibraryLogic:` / `LibraryClient:` sections.
    - To verify when implementing: (1) with validation off and `SyncsPerBenchmark: 0`, the
      LibraryLogic step still keeps the solutions (`removeInvalidSolutions` on NaN results) and
      writes a non-empty library; (2) the LibraryClient step still performs Origami selection and
      fails the test if no solution is selected, even when no kernel is launched; (3) which archs
      it can run on: selection needs an Origami hardware model
      (`pAMDGPU->analyticalHardware`); without one `findTopSolutions` returns nothing.

### 6. `python_kernel_build_src_co`

Status: decided.

Adopted:

- **A. One xnack-agnostic helper object in benchmark/test builds.** The benchmark path
  (`BenchmarkProblems.cmdLineArchsFor` → `gfxToVariants` → `gfxVariantMap`) compiles the helper
  `Kernels.cpp` for the bare arch (`gfx950`) instead of `gfx950:xnack+` and `gfx950:xnack-`.
  This matches what already ships: shipped libraries build helpers from CMake `GPU_TARGETS`
  (bare arch, xnack-agnostic; `xnack+` only for ASAN), the installed library contains only
  `Kernels.so-000-gfx950.hsaco`, assembly kernels are already xnack-agnostic, and the runtime
  loader tries the bare-arch file first. Measured −47 to −53% wall and CPU; code identical to
  today's `xnack+` object.
- **C. Split `Kernels.cpp` into chunks, compile them in parallel, link once, and cache each
  compiled chunk.**
  - Kernels defined in `Kernels.h` (the 14 `Dijk_*_Reduction` kernels in gradient/bias configs)
    move into the `.cpp`, otherwise chunks fail to link with duplicate symbols.
  - Compile each chunk to a relocatable device object and link once with `lld -shared`, using the
    driver's own `cc1` / `lld` commands (as in the experiment), which kept every kernel identical.
    `-fgpu-rdc` is not used: it changed every kernel's code.
  - Chunk count: small (the measurement flattens beyond 4) and bounded by the pool budget from
    topic 7, since chunks run in parallel with each other and, with F, with codegen.
  - **Per-chunk cache:** each compiled chunk object is cached (same pattern and location as the
    helper-kernel cache and the PCH), keyed on the chunk's source, the headers it includes
    (including the generated `Kernels.h`), the exact compile arguments and the compiler build.
    Chunk assignment must be deterministic and content-stable (e.g. grouped by kernel, not by
    balancing line counts over the whole file), so that adding or removing one kernel does not
    change every chunk's content.
  - Evidence to weigh: within the 4 captured configs, a per-kernel cache would have skipped only
    6% of kernel compiles beyond today's whole-file cache; cross-config reuse was not measured
    (production builds were cold per config).
  - **Test requirement:** measure with a full `--build-only` run starting from an empty cache, to
    get the real chunk hit rate and wall time (and compare against a run without chunk caching).
  - The PCH (E) cuts the fixed header cost per chunk from ~400 ms to ~35 ms, which removes most of
    C's extra CPU.
- **E. Cached precompiled header of the HIP headers** for the helper-kernel compile. CI keeps
  `~/.tensile` (or `TENSILE_HELPER_CACHE_DIR`) across configs, so the PCH is built once per
  job and reused.
  - PCH contents: HIP headers only (implicit `__clang_hip_runtime_wrapper.h`,
    `<hip/hip_runtime.h>`, `<hip/hip_ext.h>`, `<hip/hip_fp8.h>`), built with
    `-fpch-instantiate-templates`; used with `-include-pch`. TensileLite static headers stay out
    (they are re-copied per build with new mtimes, which clang rejects).
  - Stored next to the helper-kernel cache, using `HelperKernelCache`'s pattern (SHA-256 key,
    write to `.tmp_<key>_<pid>` then `rename`, age-based eviction).
  - Key: the exact argument list used to build it (every `-D`, target and features, `-O`, `-std`,
    sanitizer flags), the full `amdclang++ --version` output (exact compiler build), and the
    target. A `-D` baked into a PCH otherwise leaks silently into compiles that do not define it.
  - On any PCH error (e.g. headers modified by a ROCm reinstall), rebuild the PCH or compile
    without it and retry, so a stale entry never fails a build.
  - Measured: −0.3 s per translation unit and target (≈170 s across the aggregate with both
    targets, about half with A), code identical.
- **F. Start the helper compile before codegen in the benchmark path**
  (`writeSolutionsAndKernels`): `writeHelpers` only needs `kernelHelperObjs`, available on entry,
  so the compile runs while kernel codegen, assembly and the asm link run. Same approach as
  PR #13671 on the TensileCreateLibrary path; no coordination needed. Upper bound on time hidden:
  ~24% of compile time today (estimate; codegen and the compile share cores).

Deferred:

- **D. Shrinking the `PostGSU16` reduction kernels.** Changes GSU-path code; needs a runtime
  benchmark of the reduction kernels first.

Not adopted:

- **B. `-parallel-jobs=2`.** Not needed with A (one target).

### 7. `python_kernel_codegen`

Status: decided.

Adopted:

- **A. Remove `CpuThreads: 1` from all 65 `common/` test YAMLs that set it** (19 of them ran in
  the gfx950 aggregate, 189 s of codegen; the setting was present when each file was added and no
  PR explains it). Measured on `subtile_mxfp4`: codegen 133.5 → 22.6 s, `write_assemble`
  62.9 → 11.6 s, build 244 → 63 s, `.co` and libraries identical.
  - **Prerequisite: serial and parallel codegen must produce the same metadata.** `LdsNumBytes`
    (set in `Components/Subtile/SubtileLdsLayout.py`) and `LoopIters` (set in `KernelWriter.py`)
    are written onto the kernel during codegen and are lost in worker processes, because
    `passPostKernelInfoToSolution` (`TensileCreateLibrary/Run.py`) only copies back
    `CUOccupancy`, `PrefetchGlobalRead`, `MathClocksUnrolledLoop` and `CustomKernel`. Add both to
    the returned `KernelCodeGenResult` so `Data/*.yaml` matches in both modes.
- **B. Fail fast on kernels that will be rejected.** Both checks:
  1. after `_initKernel`, stop if the static VGPR allocation (`states.totalVgprs`) exceeds
     `MaxVgpr` (the VGPR pool can only grow, so rejection is certain);
  2. stop as soon as any `overflowedResources` code is set (VGPR, SGPR, LDS scheduling, LDS), at
     the code sites that set it.
  - Respect `forceGenerateKernel` (no early exit when set); keep the same error code (`-2`); print
    the reason under `PrintSolutionRejectionReason`, as the end-of-codegen check does today.
  - Validated in the prototype on 4,313 kernels: 0 accept/reject mismatches; CPU on rejected
    kernels 374.5 → 10.8 s; `f8_gate_r` build 28 → 11 s, artifacts byte-identical.
- **C. Generate all steps of a config in one pool.** Restructure `BenchmarkProblems` into:
  solutions for all steps → one codegen pool → per-step assemble/build/write (or keep one pool fed
  across steps). Acceptance check: byte-identical artifacts per step against today (step
  independence was inferred from the loop, not proven). Simulated: `custom_mainloop_scheduling`
  36 → 2.1 s, `f8_gate_r` 12.4 → 3.3 s; ≤25-kernel steps across the aggregate 142 → ≤75 s.
- **Pool budget: 2× the available CPUs across all pytest-xdist workers** (replaces the body of
  `CPUThreadCount()` in `Common/Parallel.py`, as sketched in the note):
  - CPUs available = min(affinity mask, loky `cpu_count()` (applies the cgroup quota),
    `KUBE_CPU_REQUEST` when set and parseable).
  - Concurrent workers = `PYTEST_XDIST_WORKER_COUNT`, else 1.
  - Pool size = max(1, min(CPUs, 2 × CPUs // workers)); `CpuThreads` stays a cap (`-1` = budget,
    N = min(budget, N)).
  - Accepted limitations: `KUBE_CPU_REQUEST` is a request, not a limit (TheRock already treats it
    as the pod's budget); only pytest-xdist parallelism is detected.
  - Also fix `ParallelMap2` overwriting an explicit `procs` with `CPUThreadCount()` (so
    `TensileLogic/Run.py`'s `procs=jobs` takes effect), and have TensileCreateLibrary's
    `_cpuCount()` use the same helper.

Deferred (see [DEFERRED.md](DEFERRED.md)): D (longest-first dispatch), E (cheaper generator),
and moving the VGPR/SGPR checks into solution validation.

### 8. `python_kernel_bench_setup`

Status: decided.

Adopted:

- **F2. Build helper kernel objects once per distinct input, in-process.** In
  `writeBenchmarkFiles` (`BenchmarkProblems.py`), memoize `initHelperKernelObjects` on the
  solution fields it reads (`ProblemType`, `GlobalSplitU`, `GlobalSplitUAlgorithm`,
  `isStreamK`, `StreamKAtomic`, `_GlobalAccumulation`, `AdaptiveGemmGSUA`, `ActivationFused`,
  `WavefrontSize`, `ISA`, plus the global `internalParameters["GlobalSplitUPGR"]`). The key is
  computed inside `KernelHelperNaming`, next to the code that reads these fields, so they cannot
  drift apart. The memo is scoped to one `writeBenchmarkFiles` call; the existing name-based
  dedupe keeps the kept list and its order unchanged. Measured 82–85× on the block, output
  identical on all 45 captured steps.
- **F1. `DataType.__deepcopy__` returns `DataType(self.value)`**, so the class-level
  `properties` row is shared again instead of copied. Measured 1.4–1.8× on this block; also sped
  up `python_benchpost_lib_construction`; artifacts byte-identical. Affects every `deepcopy` that
  touches a `DataType` (nothing assigns `.value` / `.properties` outside `__init__`).
- **Investigation (findings only, no fix): TensileCreateLibrary helper name/object mismatch.**
  `generateKernelHelperObjects` (`TensileCreateLibrary/Run.py`), the path that builds shipped
  libraries, dedupes helper kernels on names predicted by the `*KernelNames` functions and aligns
  them with the built objects positionally (`itertools.compress`). The predicted names do not
  match the built objects: they use the solution's `GlobalSplitU` instead of `GlobalSplitUPGR`
  (`PostGSU1` vs `PostGSU16`), the vector-width lists disagree (`isDouble()` vs `numBytes()`; 3
  names vs 2 objects for `cgemm`, 3 vs 1 for `zgemm`), and `MultipleBufferSingleKernel` ignores
  `AdaptiveGemmGSUA`. On every captured benchmark step the final set happened to be right.
  - Task: determine whether any real configuration (shipped logic files, all archs) drops a
    helper, duplicates one, or pairs a name with the wrong object; write the findings to a
    markdown file in this directory. Do not change the code as part of this plan.

Not adopted:

- **Persisting the F2 memo to a file** across runs. After F1 + F2 only a few seconds remain
  across the aggregate, and a file cache keyed on solution fields would replay stale helper
  source after generator edits unless keyed on the generator code itself. Compiled helpers are
  already cached across runs by content hash (`HelperKernelCache`).
- **Reusing TensileCreateLibrary's `generateKernelHelperObjects`**: different helper order
  (changes `Kernels.cpp` and its cache key), and its name prediction is unreliable (see the
  investigation above).

## Implementation plan

This plan implements every adopted item above. Deferred work is in [DEFERRED.md](DEFERRED.md)
and is out of scope.

### Rules for every change

1. **One adopted item per commit.** Each item below (numbered `Pn.m`) is its own commit. An
   item that is split into steps (e.g. test infrastructure, then the change) gets one commit per
   step. Unrelated items are never combined, and commits are not squashed later.
2. **Commit messages say exactly why.** Conventional-commit subject (`perf(tensilelite): …`,
   `fix(tensilelite): …`, `test(tensilelite): …`, `docs(tensilelite): …`). The body states:
   - **What** changed, in concrete terms (files, functions, behaviour).
   - **Why**: the problem, with the measured numbers that motivated it, and a reference to the
     investigation note and section (e.g. `timing_runs/cpu_reference_slow_path_investigation.md`,
     "Why the slow path is slow") and the topic/item in this file.
   - **Expected effect**, marked as measured or estimated.
   - **How it was verified** (tests run, equivalence checks, measurements).
   - **Risks / behaviour changes**, or "none (output identical)".
3. **Every commit builds and passes its tests.** At minimum the tests listed for that item, plus
   the existing suites it touches (`CPUGemm.*` CTest for `cpu-gemm-driver`, `tensilelite-tests`,
   TensileLite Python unit tests). No commit relies on a later one to be correct.
4. **No behaviour change beyond what was decided.** Output-identical items are checked to be
   identical (byte-identical artifacts, identical client stdout/CSV/exit codes as stated per
   item).
5. **Timing results are committed.** Every aggregate measurement run (P0, P3, P6) commits its
   `AGGREGATE_analysis.txt` (as the existing `per_config_rerun_2` / `per_config_build` results
   are) plus a short comparison note against the baseline, in its own `docs(tensilelite): …`
   commit that names the measured commit hash, client binary, host, cores and GPU.
6. **Never touch the measured baseline binaries.** `/tmp/tl_client_msgpack` and
   `/tmp/tl_split_gfx950/artifacts` stay as they are; new builds go to separate directories.
7. **Supporting changes travel with the item that needs them.** A timer added by an item updates
   `scripts/analyze_timing.py`'s `TIMING_HIERARCHY` in the same commit; a disable switch needed for
   a measurement (PCH, chunk cache) lands with that feature. Infrastructure that no single item
   owns (script parametrization) is its own item (P0.0).
8. **Temporary investigation code is never committed.** Examples: rejection-reason logging used
   for the P1.12 replay, the `hipMemcpy` hash/readback shims, the corruption shim used for
   P1.27's injected-failure check. If a check is worth keeping, it becomes an explicit opt-in debug
   mode, stated in that item.
9. **Numbers in commit messages are labelled.** Numbers from the investigations were measured
   with the `4c6426cea6` client; per-commit "expected effect" is an estimate unless measured at
   that commit. Say which.

### Ordering and dependencies

```text
P0 baseline + build/test setup
 │
 ├─ P1 client (C++) run-side changes ── P2 Python/YAML/harness run-side changes ── P3 run-side measurement
 │      (P1.1 first: KMP_BLOCKTIME reduces variance for everything after it)
 │                                                                                  │
 │                                         P4 build-side changes (land after P3) ──┴── P6 build-side measurement
 │                                            (topic 8 → topic 7 → topic 6)
 │
 └─ P5 TensileCreateLibrary helper-name investigation (findings only; read-only, any time)
```

Hard dependencies inside phases are listed per item ("Needs: …").

P4 commits land after P3's measurement, not interleaved with P1–P3 (plan review): the
measurement scripts clear `~/.tensile` (where the PCH and chunk caches live), pin cores 0–24, and
P3 rebuilds artifacts from the branch head, so interleaved build-side changes would contaminate
the run-side measurement. P4 code may be developed earlier on a separate branch, but nothing is
built or measured on the measurement cores while a measurement runs. P2.5/P2.6 also change which
configs have LibraryLogic/LibraryClient steps, so the P6 note reports those configs separately.

### P0. Baseline and setup

- **P0.0 Parametrize the measurement scripts.** `split/run_split.sh`,
  `per_config/run_per_config.sh` and `per_config/run_per_config_build.sh` hardcode `CLIENT`,
  `ART`, `OUT` and clear `/home/alvasile/.tensile*`. Make `CLIENT`, `ART`, `OUT` and the cache
  directories (`TENSILE_HELPER_CACHE_DIR` etc.) overridable, with today's values as defaults, so
  P0/P3/P6 never overwrite the committed baseline artifacts and never share caches with
  development builds. Own commit.
- **P0.1 Build tree with tests.** Configure a separate build directory (not
  `/tmp/tl_client_msgpack`) from the starting commit with `TENSILELITE_BUILD_TESTING=ON` and
  `TENSILELITE_ENABLE_CLIENT=ON`, Release, `GPU_TARGETS=gfx950`, same compiler as the baseline.
  Targets used throughout: `tensilelite-client`, `cpu-gemm-driver`, `tensilelite-tests`.
- **P0.2 Baseline at the starting commit.** The committed baselines (`per_config_rerun_2`,
  `per_config_build`) were measured with a client built at `4c6426cea6`, older than the starting
  commit. Re-measure at the starting commit with the same scripts so every comparison is
  like-for-like:
  - Build artifacts: `full_runs_gfx950/split/run_split.sh` build phase into a new artifact dir.
  - Run side: `full_runs_gfx950/per_config/run_per_config.sh` (serial, `TimingInstrumentation=True`,
    25 pinned cores, `OMP_NUM_THREADS=25`) with `OUT` pointing at a new
    `full_runs_gfx950/baseline_run/` and the new client/artifacts.
  - Build side: `full_runs_gfx950/per_config/run_per_config_build.sh` (cold cache per config) with
    `OUT=full_runs_gfx950/baseline_build/`.
  - Record the full common-test pass/fail set (split run phase, `-n 4`) as the correctness
    baseline.
  - Also record the **uninstrumented** wall time of that split run phase (`-n 4`): P1.20 adds
    syncs under instrumentation only, so instrumented wall time understates production gains.
  - Keep the raw per-config logs (`*.log.gz`) so P0 can be re-aggregated with the final
    `analyze_timing.py` hierarchy before comparing.
  - Noise: record host load (`uptime`, other users' processes) and GPU clocks at the start and
    end of each run; repeat the ~10 most expensive configs 3× for a noise band.
  - Commit both `AGGREGATE_analysis.txt` files, the pass/fail list and a note with commit, host,
    cores, GPU (rule 5).

### P1. Client (C++) run-side changes

All in `projects/hipblaslt/tensilelite/client` unless stated. Order within P1 is the listed order.

**Topic 5a, A — OpenMP workers**

- **P1.1 `KMP_BLOCKTIME=0` by default.** In `main.cpp`, before the first OpenMP use: if
  `KMP_BLOCKTIME` is not set in the environment, make idle workers sleep immediately
  (`kmp_set_blocktime(0)`, or `setenv("KMP_BLOCKTIME","0",0)` before the runtime initializes;
  pick whichever is verified to apply to the worker pool used by later parallel regions).
  The decision names `kmp_set_blocktime(0)`; if `setenv` is chosen, the commit message says why.
  `kmp_set_blocktime` is libomp-only, so guard it for other OpenMP runtimes / Windows builds.
  `cpu-gemm-driver` does not get this; pin `KMP_BLOCKTIME` explicitly when quoting driver timings.
  - Verify: per-thread CPU sampled from `/proc/<pid>/task/*/stat` during a dtl solution loop shows
    no worker above 25% CPU (the note measured 11–24 spinning workers before); the >1 ms share of
    the bookkeeping timers drops on a dtl replay; an explicit `KMP_BLOCKTIME` in the environment
    still wins.

**Topic 1 — CPU reference** (`client/src/Reference.cpp`, `client/cpu_gemm_driver.cpp`,
`tests/CMakeLists.txt`, `tests/ReferenceFastPath_test.cpp`, `tests/ReferenceMXFastPath_test.cpp`)

Every fast-path extension follows the established pattern of the FP8/FP4 (#8106), f64/tf32
(#9203) additions: extend `isFastPathEligible` only for what `solveCPUFast` implements in the same
commit; add `cpu-gemm-driver` support for the new case; add fast/slow `CPUGemm.*` CTest pairs that
validate both paths against the driver's golden GEMM; add gtest cases for exactness edge cases.

- **P1.2 Slow path: lock only for AMAX-D (A).** In `ReferenceSolution<…>::SolveCPU`, enter the
  critical section only when `problem.outputAmaxD()`; drop the per-element
  `omp_set_num_threads`.
  - Verify: existing `CPUGemm.slow_*` tests; `cpu-gemm-driver` 1024³/2048³ fp32 slow path
    bit-identical to before (max diff 0) and faster (note: 1,727 → 191 ms; 45,237 → 1,507 ms).
- **P1.3 AMAX-D reduction and thread cap (A+).** Replace the AMAX-D critical section with a
  per-thread maximum combined once; delete the `omp_set_num_threads(MAX_OMP_THREADS)` calls in
  `SolveCPU` and `ReductionCPU` (setting `omp_get_max_threads()` would be a no-op). Behaviour
  change to state in the commit: today the first slow-path call permanently sets the main
  thread's team size to 64, so every later OpenMP region (fast path, data init) runs 64 threads
  on 25 pinned cores; after this, regions follow `OMP_NUM_THREADS`.
  - Verify: new driver support for AMAX-D output if absent, with fast/slow-agnostic tests that
    AMAX-D equals the max |D| computed by the golden GEMM; existing tests unchanged.
- **P1.4 Driver: output type separate from accumulator; Int8 inputs.** Test infrastructure only:
  `cpu-gemm-driver` gains an output (C/D) type option and alpha/beta type, Int8 A/B storage, and
  golden-GEMM support for them (integer accumulation for Int8, the slow path's epilogue type,
  `SaturateCast` semantics on the output). Slow-path CTest cases for the new combinations land
  here and must pass before any fast-path change.
  - The slow path accumulates float-alpha Int8 types (`I8_*_S`) in `float` per K element, so it
    is inexact once |sum| > 2²⁴. Slow-path CTest sizes must keep |sum| < 2²⁴ (the default sizes
    do: ~151·127² ≈ 2.4M).
- **P1.5 Fast path: separate MAC type from epilogue type.** Refactor `solveCPUFast` to take a
  multiply-accumulate type in addition to `AccumT` (epilogue type), defaulting to `AccumT`. No new
  types enabled. Verify: all existing `CPUGemm.fast_*` and gtests unchanged (bit-identical).
- **P1.6 Fast path: Int8 inputs.** Int8 A/B shadow buffers, `int32_t` accumulation, epilogue in
  `int32_t` (int32 alpha: `I8_I8_I32`, `I8_I32_I32`) or `float` (float alpha: `I8_I32_S`,
  `I8_I8_S`, `I8_H_S`, `I8_B_S`); Half/BF16 outputs reuse existing write-back. Enable these type
  combinations in `isFastPathEligible`. Output casts must reproduce the slow path's
  `SaturateCast` exactly (verified in `Reference.cpp`):
  - Int8 output: from a float epilogue, round-to-even then clamp to [−128, 127]; from an int32
    epilogue, clamp only.
  - Int32 output: the generic overload, i.e. a plain `static_cast` (truncation, no rounding, no
    clamp; out-of-range is undefined behaviour). Tests stay in range.
  - **Epilogue operation order (decided):** the Int8 fast-path epilogue applies alpha, the
    ScaleAB scales, `scaleAlphaVec`, beta·C, bias, activation and the output cast in exactly the
    slow path's order and types (`ReferenceSolution<…>::SolveCPU`), so Int8 outputs match the slow
    path exactly, including round-to-even ties. Tests compare fast vs slow exactly for integer
    outputs.
  - Tests: fast/slow CTest pairs for each output type (I8, I32, Half, BF16) × transposes, beta,
    bias, activation, ScaleAB; gtests for exact integer results above 2²⁴ for int32-epilogue
    types (for float-epilogue types the expected value is `float(exact sum)` then the epilogue),
    and Int8 saturation at ±127/−128.
  - Commit message notes that for large-K `I8_*_S` problems the reference changes (now exact in
    the dot product, matching the GPU's int32 accumulation better than the slow path did).
- **P1.7 Fast path: Float8 outputs.** `storeFrom` for Float8/BFloat8 (and `_fnuz`) outputs
  matching `SaturateCast` (including the BFloat16-accumulator path); enable `F8×F8→F8`, the BF8
  and hybrid variants, and `F8×Half→F8` (mixed-input shadow with compute-input quantization).
  - Tests: driver support for F8/BF8 outputs; fast/slow pairs per variant; gtest for rounding and
    saturation at the F8 range limits.
- **P1.8 Fast path: complex types (C).** `ComplexFloat` and `ComplexDouble` A/B/C/D, including
  conjugate ops (`TensorOp::ComplexConjugate`), matching the slow path's `Transform`/`multiply`.
  - Tests: driver complex support; fast/slow pairs incl. conjugate A/B; gtest for conjugation.
- **P1.9 Fast path: `useE` (C).** Store the pre-activation value to E in the epilogue
  (`SetValue` semantics as the slow path; not with gradient).
  - Tests: driver E output; fast/slow pairs with activation.
- **P1.10 Fast path: gradient (C).** Gradient GEMM: E-based activation derivative multiply, and
  the bias reduction (`ReductionCPU` for bias from D/A/B, with the D-source workspace).
  - Tests: driver gradient + bias-reduction support; fast/slow pairs for each bias source.
- **P1.11 Fast path: MX K not a multiple of the MX block (C).** Handle a partial last MX block
  (K = 333/255/63 with block 32 in `mxfp4_mxfp4_bf16_tn_act`).
  - Tests: MX gtests and CTest cases with K tails on both sides; existing MX tests unchanged.
- **P1.12 Topic 1 replay.** After P1.2–P1.11: replay the 59 slow-path YAMLs (as in the note's
  method, with uncommitted local rejection-reason logging, rule 8) and confirm every config still
  passes. Expected remaining slow-path reasons: `NumElementsToValidate: 128` (sparsity
  threshold) plus any of amaxD, scaleCD, mirror indices, multi-index, no batch indices,
  non-E8 / one-sided MX that occur; anything else is a finding. Results go into the P3 comparison
  note, not a separate commit.

**Topic 4 — `cpu_data_init`**

- **P1.13 MX generator: opt out of the reference floats.** Add an opt-out to `generateMXInput`
  (`clients/common/src/mxDataGen.cpp`, `clients/common/include/mxDataGen.hpp`) that skips
  `getReferenceFloat()`; default unchanged. The TensileLite client's two call sites opt out.
  - Verify: `tests/MXDataGen_test.cpp`; build the hipBLASLt clients that use the shared code
    (`clients/common`) and run their MX tests (`fast_check_gtest` MX cases), since P0.1 builds only
    TensileLite targets; generated data/scale bytes identical with and without the opt-out.
- **P1.14 Trig init without per-element overhead.** In `initArrayTrig` only (decision 2a):
  no per-element `std::vector`, flat index for packed strides; keep double `sin`/`cos`. Verify:
  generated arrays byte-identical to before for packed and strided descriptors (new gtest in
  `DataInit_test.cpp`).
- **P1.15 Dynamic schedule for init loops.** `schedule(dynamic, chunk)` on the init loops.
  Verify: Trig/Serial/Identity outputs byte-identical; timing under a contended core set.

**Topic 2 — input data preparation** (`client/include/DataInitialization.hpp`,
`client/src/DataInitialization.cpp`)

- **P1.16 Copy the current problem's extent, not `maxElements`.** `copyInputs` /
  `copyInputBuffers` / per-problem `copyValidToGPUBuffer` copy the current problem's allocated
  extent. In NaN / guard-page modes the guard regions already span the whole buffer
  (`padding = (maxElements − allocated)/2`), so the change matters in Disable mode.
  - Verify: Disable-mode multi-size configs where the size shrinks and then grows; bounds-check
    configs (`nan_bounds_check_odd_padding`, `bufferLoadStore0`); swizzled tensors
    (`getSwizzledTensorNumAllocatedElements`), padded MX scales, sparse metadata, grouped GEMM
    offsets and rotating-buffer mode.
- **P1.17 Skip unchanged per-solution uploads (fix 1).** The host epoch alone is not enough:
  GPU `valid` buffers have other writers (verified): the initial copy, the MX swizzled upload,
  every branch of `copySwizzledToGPUBuffer`, and the grouped path. Track, per (tensor, dtype), a
  tag of what the GPU `valid` buffer actually holds (host epoch plus layout/source); every writer
  sets or invalidates it; the initial state is "unknown". Skip the H2D only when the tag matches.
  - Verify: an opt-in debug mode (kept, rule 8) that on every skip reads `gpuInput.valid` back
    and compares it with the expected bytes, run once over the full common suite with zero
    mismatches; MX, sparse, swizzled, Trig and bounds-check configs pass.
- **P1.18 Cache the batched-pointer arrays (fix 2).** The pointers are `current + padding`, and
  padding depends on the bounds-check mode, which flips GuardPageFront → GuardPageBack between the
  two iterations of one GuardPageAll solution without any epoch or address change. Key the cache
  on the computed per-tensor start pointers plus the descriptors (equivalently: rebuild when the
  problem, `m_curBoundsCheck`, swizzle size or sparse metadata changes).
  - Verify: grouped/batched, GuardPageAll (`bufferLoadStore0`) and sparse configs pass.
- **P1.19 MX swizzled scale from the canonical scale (fix 4).** Two commits:
  1. Test first: gtest that permuting the generator's **compact** canonical scale (before the
     K-fast restride that `cpuInput.valid` receives; `generateMXInput(…, GFX950)` permutes the
     compact scale) with the existing `applyMXScaleLayoutInPlace` (`mxDataGen.cpp`) is
     byte-identical to `generateMXInput(…, GFX950)` output, for FP4/FP8, K-fast and K-slow
     layouts, a padded leading dimension, and batched problems. Also check that the data bytes
     from the two generator calls are identical (today's code pairs call 2's data with call 1's
     scale). Check the batched case carefully: today's swizzled per-batch size can exceed
     `scaleBatchStrideBytes`, so "identical to today" may encode overlapping batches; if so, stop
     and report rather than reproduce it.
  2. Change: per-solution MX handling never regenerates data or canonical scale; keep the compact
     canonical scale per problem (or un-restride), permute it once per problem, cache it in its
     own GPU buffer so canonical- and swizzled-layout solutions do not overwrite each other;
     `m_inputEpoch` is not bumped.
  - Verify: `pre_solution_mx_side_regenerate` and `cpu_reference_gemm_per_solution_inputs_changed`
    drop to zero on `subtile_mxfp4` / `subtile_mxfp8`; MX configs pass.

**Topic 3 — Synchronizer check and attribution syncs** (`client/src/SynchronizerValidator.cpp`,
`client/include/SynchronizerValidator.hpp`, `client/main.cpp`)

- **P1.20 Attribution syncs under timing instrumentation.** In `main.cpp`, only when
  `g_timingInstrumentationEnabled`: `hipStreamSynchronize` after warmup 0 (before
  `listeners.validateWarmups`) under its own timer, and at the end of each run iteration under its
  own timer, **before** the per-iteration Synchronizer check (P1.22), so that check does not absorb
  queued warmups. It goes **inside the `try`, inside the `needMoreRunsInSolution` loop**, never
  between the `catch` and `postSolution`: a sticky device error there would otherwise throw out of
  the solution loop. Uses the stream, not timing events
  (needed by P1.28). Verify: with instrumentation off, no extra HIP calls (count with a shim or
  trace); with it on, `validate_synchronizer_readback` / `validate_gpu_readback` no longer contain
  kernel time; a forced kernel fault still reaches the existing device-lost handling.
- **P1.21 Host check: OR-reduction first (1-alt).** Rewrite the clean-path scan as a vectorizable
  OR over the buffer; run the count/first-offset scan only when dirty. Diagnostic unchanged.
  Verify: `SynchronizerResidue_test.cpp` (clean, dirty, first offset, counts) and a speed check.
- **P1.22 Check in `postSolution` with guards.** Save the `ProblemInputs` pointer; do the check
  inside `SynchronizerValidator::postSolution` (it already runs after `ReferenceValidator`'s, so
  its `FAILED` overrides the reference verdict; keep registration order unchanged); guard 1 (only
  if the solution launched: flag from `preWarmup` / `preEnqueues`, plus `m_mayUseSynchronizer`);
  guard 2 (exception path: check if the device is usable, otherwise catch and mark `FAILED`, never
  throw); keep clear-on-dirty. GuardPageAll (two iterations per solution): the check itself runs
  at the end of every run iteration, before the next iteration's input reset (inside the
  `needMoreRunsInSolution` loop, after the iteration's launches; guards 1–2 apply per iteration),
  and `postSolution` reports the accumulated verdict.
  - Also verify: a GuardPageAll config with residue injected in the first iteration is reported.
  - Verify: `SynchronizerValidatorReporting_test.cpp` extended for: residue from a later launch
    attributed to its own solution; a dirty buffer gives `FAILED` even when the reference check
    passed; no check for non-launched solutions; residue after an `INVALID` launch with a usable
    device (state which verdict wins); device-lost path; residue reported exactly once.
- **P1.23 GPU OR-reduction check.** OR-reduction kernel on the client's stream + 4-byte readback;
  full readback and existing diagnostic only when nonzero; host check (P1.21) used under the FFM
  emulator (`HSA_MODEL_MEMFILE`) and as a fallback when the kernel cannot launch on the device
  (no code object for the arch: `hipErrorNoBinaryForGpu` / invalid device function).
  - Verify: residue injected into the buffer is detected with identical diagnostic; per-check
    time; Stream-K/MBSK/amaxD configs (`streamk/*`, work-stealing configs) pass.

**Topic 5 — bookkeeping and correctness-only runs (client side)**

- **P1.24 Predicate dump behind its own option (5a C).** New client option, default off, gating
  `debugEval` in `SolutionIterator::checkSolution`; wired to a new Tensile global parameter in
  `ClientWriter.py` / `Common/GlobalParameters.py` (Python part in the same commit, so the option
  is usable end to end). Verify: dump appears only with the option; rows unchanged.
- **P1.25 `LogReporter`: check the row level before copying (5b D1).** Verify: stdout identical at
  every log level on a dtl invocation (normalizing the `srand seed` line).
- **P1.26 Skip unused granularity/memory values (5a D2).** In `BenchmarkTimer::preSolution`, skip
  `projectedPerformance` and the 9 granularity/memory reports unless a `LogReporter` will print
  the row (stdout at its level, or the `--log-file` reporter `main.cpp` adds at
  `LogLevel::Normal`) or `CSVExportExtraCols` is set. Applies to every run, timed or not.
  - Verify: stdout, log file and results CSV byte-identical at every log level, with and without
    `--log-file` and `CSVExportExtraCols`.
- **P1.27 Infer "nothing is timed" (5b E).** Static rule from the client's arguments:
  `num-syncs-per-benchmark == 0`, or `max-enqueues-per-sync == 0`, or
  (`num-enqueues-per-sync == 0` and `min-flops-per-sync == 0`). When it holds: skip the
  performance calculation in `BenchmarkTimer::postSolution` and report the same NaN values (same
  `-nan` text).
  - Verify: results CSV and stdout byte-identical against the previous commit on dtl and xfp32
    invocations, clean and with injected failures (local corruption shim, not committed), exit
    codes identical; the four syncs/enqueues/min-flops combinations from the note behave as
    tabled.
- **P1.28 No warmup timing events when nothing reads them (5b F).** Needs: P1.20, P1.27.
  `BenchmarkTimer` reads warmup events in `postWarmup` when `skip-slow-solution-ratio` is set and
  in `validateWarmups` under `--sync-after-warmups` (verified). Create the events unless nothing is
  timed **and** the skip-slow ratio is 0 **and** `--sync-after-warmups` is off; otherwise keep
  them.
  - Verify: as P1.27; plus a nothing-timed run with skip-slow set; `BenchmarkTimer` unaffected
    when timing is on.
- **P1.29 Client "no results file" path (5b I, client side).** An explicit option that never
  constructs `ResultFileReporter`. Verify: no file written, exit code and stdout unchanged.

### P2. Python, YAML and test-harness run-side changes

- **P2.1 Copy the results CSV only if it exists (5b H).** `BenchmarkProblems.main`, same guard as
  the granularity CSV. Unit test.
- **P2.2 Write the results CSV only when consumed (5b I, Tensile side).** Needs: P1.29, P2.1.
  In `BenchmarkProblems` only (not in the shared `ClientWriter.writeClientConfig`, which
  `TensileRetuneLibrary.py` also uses and which copies `benchmark.csv`): pass "no results file"
  unless the config has a `LibraryLogic:` section or uses the Ductile backend. Behaviour change to
  state in the commit: the CSV's existence is also the "already benchmarked, skip" marker
  (`BenchmarkProblems.py`), so reruns into the same output directory re-run the client. Unit
  tests for both branches; a LibraryLogic config and a Ductile config still produce their
  outputs.
- **P2.3 Test runs use `ClientLogLevel=0` (5a D1).** `tensile_args` in `Tests/conftest.py`
  always appends `--global-parameters ClientLogLevel=0` before any user-supplied
  `--global-parameters`. The fixture body moves into a plain helper with a unit test: the value is
  present, and a user-supplied `ClientLogLevel=3` comes after it.
- **P2.4 `xfp32.yaml`: drop `NumWarmups: 5` and `EnqueuesPerSync: 10` (5b G).**
- **P2.5 New minimal `Prediction` config (5b K).** One small non-MX YAML (BF16 or F16): very small
  sizes in the benchmark step and `LibraryClient`, minimal solutions, `NumElementsToValidate: 0`.
  Must first confirm the three verification points in topic 5 (non-empty library with validation
  off; LibraryClient performs Origami selection and fails if none is selected; which archs have an
  Origami hardware model) and restrict the config to those archs. If LibraryClient turns out
  **not** to fail when nothing is selected, stop and ask before changing code (new scope).
- **P2.6 `LibraryLogic:` / `LibraryClient:` whitelist test and removals (5b K).** Needs: P2.5.
  Two commits: the whitelist test (initially listing every current file, with reasons); removal
  from the 14 `ductile/*`, 10 mxfp8 and 12 `custom/*` YAMLs, shrinking the whitelist to the 6
  kept files + the new `Prediction` config (verification: the stale-entry check fires on the
  removed entries).
- **P2.7 `SyncsPerBenchmark` guard covers `custom/` (5b J).** Needs: P2.6. Must come after the
  `custom/*` `LibraryLogic:` removal: all 12 custom YAMLs have `LibraryLogic:`, and with
  `SyncsPerBenchmark: 0` their logic step would get NaN results that `removeInvalidSolutions`
  (drops only `gflops == 0`) does not filter (verified). Two commits: set `SyncsPerBenchmark: 0`
  in the 12 custom YAMLs (run all 12 at that commit), then extend
  `Tests/unit/test_common_yaml_syncs_per_benchmark.py` to `Tests/custom/`.

### P3. Run-side measurement

- Rebuild the client at the end of P2; regenerate artifacts (split build phase) at that commit.
- Full common-test pass (split run phase, `-n 4`): the pass/fail set must equal the P0 baseline,
  except for intended changes (listed in the note).
- Per-config run aggregate (`run_per_config.sh`, same settings as P0) into
  `full_runs_gfx950/after_run/`.
- Commit `AGGREGATE_analysis.txt` and a comparison note (`docs(tensilelite): …`): total and
  per-category deltas vs P0, the P1.12 replay result, and anything that moved unexpectedly
  (attribution changes are expected: wait time moves to the new sync timers under instrumentation).

### P4. Build-side changes

Order: topic 8 (P4.1–P4.2), pool budget and fail-fast (P4.3–P4.6), parallel codegen metadata and `CpuThreads` (P4.7–P4.8), compile/codegen overlap (P4.9), one pool per config (P4.10), then the helper compile itself (P4.11–P4.15). P4 commits land after P3 (see "Ordering and dependencies").

**Topic 8 — `python_kernel_bench_setup`**

- **P4.1 F1: `DataType.__deepcopy__` returns `DataType(self.value)`.** Verify: TensileLite unit
  tests; artifacts of `fp16_tn` and `fp16_HH_BHS_bf16mfma` byte-identical (excluding `.hsaco`
  `__hip_cuid_*`).
- **P4.2 F2: memoize `initHelperKernelObjects` per step.** Key computed in `KernelHelperNaming`;
  memo scoped to one `writeBenchmarkFiles` call. Verify: ordered helper list (names, header,
  source) identical on the captured steps; artifacts byte-identical; a unit test that records the
  solution keys the helper writers actually read while running `initHelperKernelObjects` (e.g.
  through a recording dict proxy) and asserts that set is contained in the memo key.

**Topic 7 — `python_kernel_codegen` and pool sizing**

- **P4.3 Pool budget (2× CPUs across xdist workers).** `CPUThreadCount()` in `Common/Parallel.py`
  as decided, with unit tests (mocked affinity / loky count / environment, as in the prototype).
- **P4.4 `ParallelMap2` honours `procs`; TensileCreateLibrary `_cpuCount()` uses the shared
  helper.** Separate commits.
- **P4.5 Fail fast: static VGPR check after `_initKernel` (B.1).**
- **P4.6 Fail fast: stop when `overflowedResources` is set (B.2).**
  - Verify (both): accept/reject decision identical for every kernel of the probed steps (the
    note's `ff.py` method); `PrintSolutionRejectionReason` output still names the reason;
    `forceGenerateKernel` still generates; artifacts byte-identical.
- **P4.7 Return `LdsNumBytes` and `LoopIters` from codegen workers (A prerequisite).** Add them
  to what `passPostKernelInfoToSolution` copies back. Verify: a generic test comparing the full
  post-codegen kernel/solution state between serial and parallel codegen (not only these two
  keys), on `subtile_mxfp4` and `custom_mainloop_scheduling`; `Data/*.yaml` identical.
- **P4.8 Remove `CpuThreads: 1` from the 65 `common/` YAMLs that set it (A).** Needs: P4.3,
  P4.7. The 6 commented-out `CpuThreads: 1` lines (`common/sparse/*`) are removed in the same
  commit. Verify: `--build-only` serial vs parallel for all 65 YAMLs on their own archs (only 19
  run on gfx950), artifacts identical; the gfx950 ones pass.
- **P4.9 Start the helper compile before codegen in the benchmark path (topic 6, F).** Moved
  before the codegen restructure (plan review): in today's per-step structure, start
  `writeHelpers` + the helper compile on entry to `writeSolutionsAndKernels`, since it only needs
  `kernelHelperObjs`. Verify: artifacts byte-identical; total processes stay within the P4.3
  budget.
- **P4.10 One codegen pool per config (topic 7, C).** Restructure `BenchmarkProblems` to generate
  all steps' kernels in one pool, then assemble/build/write per step; keep F's overlap by
  launching every step's helper compile before the shared codegen pool, with the pool budget
  split between helper compiles/chunks and codegen workers. Verify: per-step artifacts
  byte-identical to before on configs with many small steps (`custom_mainloop_scheduling`,
  `f8_gate_r`, `spmm_i8_sb`) and large ones (`dtl`); per-step `--use-cache` hits still skip
  codegen; peak memory with all steps' kernels held at once.

**Topic 6 — helper-kernel compile**

(F, the compile/codegen overlap, is P4.9 above.)

- **P4.11 xnack-agnostic helper compile in benchmark/test builds (A).** `cmdLineArchsFor` gives
  the bare arch for the helper compile, except when sanitizer flags are present, where it keeps
  `xnack+` (matching shipped ASAN builds). `gfxVariantMap` also covers gfx906/908/90a/942.
  State in the commit: `cmdLineArchsFor` also feeds the benchmark cache key (`_Architectures`)
  and `HelperKernelCache`'s key, so existing `--use-cache` / helper-cache entries are invalidated
  once. Verify: helper kernels identical to today's `xnack+` object (disassembly with addresses
  stripped); the runtime loads `Kernels.so-000-gfx950.hsaco`; configs pass; a build-only check on
  gfx942 as well.
- **P4.12 Cached HIP-header PCH (E).** As decided. The key uses the full `amdclang++ --version`
  output (not `HelperKernelCache`'s `major.minor.patch`) and the resolved ROCm/HIP include path,
  plus the exact argument list and target. Includes a switch to disable the PCH. Verify: kernels
  identical with and without the PCH; stale-PCH fallback tested (touch a header's mtime); a cold
  and a warm compile timed.
- **P4.13 Move header-defined kernels into `Kernels.cpp` (C prerequisite).** The 14
  `Dijk_*_Reduction` kernels. Verify: kernels identical; gradient/bias configs pass.
- **P4.14 Chunked helper compile (C).** Needs: P4.11–P4.13. Deterministic, content-stable chunk
  assignment; each chunk includes a generated header with only its own kernels' declarations (plus
  the static headers and shared argument structs), so the per-chunk cache key does not depend on
  the whole `Kernels.h`; driver `cc1`/`lld` commands with `-include-pch` passed through; fall back
  to the unchunked compile on any failure of the chunked path; chunk count bounded by the P4.3
  budget. Verify: every kernel and its metadata identical to the unchunked build; the fallback
  path exercised by a test.
- **P4.15 Per-chunk compile cache (C).** Needs: P4.14. Keyed on the chunk source, its generated
  header, the static headers, exact compile arguments and the full compiler version. Includes a
  switch to disable the chunk cache (needed for P6). Verify: hit/miss behaviour under unit test;
  the empty-cache full `--build-only` run in P6.

### P5. Investigation: TensileCreateLibrary helper name/object mismatch (findings only)

Independent; any time. Determine whether `generateKernelHelperObjects` can drop, duplicate or
mispair a helper for any real configuration (shipped logic files, all archs). Write the findings
to a markdown file in this directory and commit it. No code change.

### P6. Build-side measurement

- Per-config build aggregate, cold cache per config (`run_per_config_build.sh`, same settings as
  P0) into `full_runs_gfx950/after_build/`: comparable with the P0 baseline.
- **Full `--build-only` run of all configs from an empty cache, cache kept across configs** (the
  split build phase, `-n 4`, CI-like): measures the PCH and per-chunk cache hit rates and wall
  time; repeat once with chunk caching disabled for comparison (required by topic 6, C).
- Commit both aggregates and a comparison note (rule 5).

### Plan review

The plan was critically reviewed by a separate agent (Claude Opus 5.5) against the decisions and
the code. Corrections incorporated above:

- Measurement isolation: P0.0 (parametrized scripts, separate caches), P4 lands after P3,
  uninstrumented wall time, raw logs kept, noise band, analyzer hierarchy kept current.
- Synchronizer: listener order corrected (verdict override relies on reverse `postSolution`
  order); end-of-iteration sync placed inside the `try`; device fallback for the GPU check.
- CPU reference: Int8/Int32 output-cast semantics corrected to match `SaturateCast`;
  float-accumulator slow-path limits for Int8 tests; `omp_set_num_threads` calls deleted, with
  the process-wide effect stated; expected remaining slow-path reasons listed for P1.12.
- Input preparation: upload skip keyed on what the GPU buffer actually holds (all writers), with
  a readback debug mode; batched-pointer cache keyed on computed pointers and descriptors;
  extent-sizing verification in Disable mode and for swizzled/MX/sparse/grouped/rotating tensors;
  MX swizzle permutes the compact canonical scale via `applyMXScaleLayoutInPlace`.
- Bookkeeping: D2 split from E and conditioned on every `LogReporter` (incl. `--log-file`);
  warmup events kept when skip-slow or `--sync-after-warmups` read them; results-CSV switch only
  in `BenchmarkProblems`, with the rerun-marker behaviour change stated.
- YAML ordering: `custom/` `LibraryLogic:` removal before `SyncsPerBenchmark: 0`.
- Build side: F moved before the one-pool restructure; xnack+ kept for sanitizer builds and the
  cache-key invalidation stated; PCH key uses the full compiler version and include path; chunk
  headers per chunk, fallback to the unchunked compile, cache/PCH disable switches; stronger
  equivalence tests for F2 and serial-vs-parallel codegen.
- Commit rules 7–9: supporting changes travel with their item, investigation code is never
  committed, and numbers in commit messages are labelled measured or estimated.

Decisions raised by the review (both decided):

1. **Synchronizer coverage for GuardPageAll:** check at the end of every run iteration, before the
   next input reset; report in `postSolution` (P1.22). Accepting the coverage loss was not chosen.
2. **Int8 epilogue matching:** the Int8 fast-path epilogue mirrors the slow path's operation order
   and types, for an exact match (P1.6). A shared epilogue function for all types and a ±1 test
   tolerance were not chosen.
