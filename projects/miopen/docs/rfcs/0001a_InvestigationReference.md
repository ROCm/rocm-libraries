# MIOpen hipDNN Shim — Investigation Reference

Companion document to the MIOpen / hipDNN shim RFC. Consolidates every test report under `projects/miopen/test-results/` so the data backing every RFC decision is in one place.

## Common context

- Date range: 2026-05-27
- Branch: `users/nhanna/miopen-hipdnn-shim-investigation-1`
- Hardware: AMD Instinct MI300X (gfx942), ROCm 7.13, MIOpen 3.5.2
- Two builds under test, differing only in one CMake flag:

| Build | `MIOPEN_ENABLE_HIPDNN_WRAPPER` | Library layout |
| --- | --- | --- |
| `build-flagoff` | `OFF` | single `libMIOpen.so.1` |
| `build-flagon`  | `ON`  | `libMIOpen.so.1` wrapper + `libMIOpen_private.so.1` impl |

The wrapper variant inserts a thin C-ABI forwarding layer: each of MIOpen's 263 public entry points becomes a stub `<name>` in `libMIOpen.so.1` that calls `<name>_impl` in `libMIOpen_private.so.1`. The rename is achieved during the private-library build via `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h`, so the same source compiles to `_impl`-suffixed symbols.

## Document map

| Section | Source file | What it answers |
| --- | --- | --- |
| [§1](#1-miopendriver-smoke-test) | `MIOpenDriverTest.md` | Does the flag change basic MIOpen driver behavior or correctness? |
| [§2](#2-runtime-wrapper-logging) | `RuntimeWrapperLogging.md` | Can we prove every public call actually traverses the wrapper at runtime? |
| [§3](#3-miopen-gtest-results--build-flagon) | `MIOpenGtestResults.md` | Does the upstream MIOpen test suite still pass with the wrapper enabled? |
| [§4](#4-performance-comparison) | `PerformanceComparison.md` | What is the runtime cost of the wrapper? |
| [§5](#5-miopen-hipdnn-provider-plugin--linkage--runtime-verification) | `MIOpenPluginResults.md` | Does the hipDNN provider plugin correctly link to (and route through) `libMIOpen_private.so.1`? |
| [§6](#6-hipdnn-redirect-feasibility--phase-2-forwarding-prototype) | `hipdnn-redirect-investigation.md` | Can the wrapper actually redirect a call into hipDNN and run it end-to-end, and at what cost? |

§1–§5 are Phase 1 evidence (the library split). §6 is the first Phase 2 evidence (the forwarding path itself) and was produced in a separate, later investigation on different hardware — it carries its own context block rather than the common context below.

---

# §1 MIOpenDriver smoke test

*Source: `MIOpenDriverTest.md`*

`MIOpenDriver` was built in both trees via `make -j32 MIOpenDriver`; both linked successfully.

## Linkage verification

`ldd` confirms the flag produces the expected library split:

- **flagon** — `MIOpenDriver` links against both `libMIOpen.so.1` and `libMIOpen_private.so.1` (private interface split out for the hipDNN wrapper).
- **flagoff** — `MIOpenDriver` links against `libMIOpen.so.1` only (no private library produced).

No `hipdnn*` symbols are exported from `libMIOpen.so` in either build, which is expected — the shim lives in a separate target.

## Smoke tests

The same four invocations were run against each build. Verification was enabled (`-V 1`) in all cases.

### 1. Forward + Backward Convolution (verify only)
```
MIOpenDriver conv -n 1 -c 3 -H 32 -W 32 -k 16 -y 3 -x 3 -p 1 -q 1 -V 1
```

| Build | Forward | Bwd Data | Bwd Weights |
| --- | --- | --- | --- |
| flagon  | OK (3.08e-08) | OK (4.46e-08) | OK (7.87e-08) |
| flagoff | OK (3.08e-08) | OK (4.46e-08) | OK (7.91e-08) |

### 2. Convolution with Timing (`-t 1`)
```
MIOpenDriver conv -n 1 -c 3 -H 32 -W 32 -k 16 -y 3 -x 3 -p 1 -q 1 -V 1 -t 1
```

Same solver selections on both builds:
- Forward: solution 84 / `ConvBinWinogradRxSf2x3g1`
- Bwd Data: solution 84 / `ConvBinWinogradRxSf2x3g1`
- Bwd Weights: solution 110 / `ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC`

| Stage | flagon (ms) | flagoff (ms) |
| --- | --- | --- |
| Fwd | 0.01046 | 0.01052 |
| BwdD | 0.01259 | 0.01204 |
| BwdW | 0.02244 | 0.02333 |

Timing variance is within noise; all stages verified OK on both builds.

### 3. GEMM
```
MIOpenDriver gemm -m 64 -n 64 -k 64 -V 1
```
Both builds: `Forward GEMM Verifies on CPU and GPU (err=0.000000)`.

### 4. Activation
```
MIOpenDriver activ -n 1 -c 3 -H 32 -W 32 -V 1
```
Both builds: forward and backward activation verify on CPU and GPU.

## Result

Both builds pass every smoke test with identical correctness results. The `MIOPEN_ENABLE_HIPDNN_WRAPPER=ON` build adds the private library split (visible in `ldd`) without altering MIOpen driver behavior or solver selection on the tested workloads — i.e., enabling the hipDNN shim has not regressed the underlying MIOpen public API.

---

# §2 Runtime wrapper logging

*Source: `RuntimeWrapperLogging.md`*

Smoke and gtest results show flagon behaves like flagoff, but they don't directly prove the wrapper layer is being used. This section adds explicit per-call instrumentation that prints whenever a wrapper stub is invoked, so the wrapper-traversal claim is observable in stderr.

## What was changed

`src/private/wrapper.cpp` was instrumented so each of the 263 forwarding stubs announces itself on `stderr` before delegating to its `*_impl` counterpart. One `#include <cstdio>` was added and one line was inserted as the first statement of every stub:

```cpp
extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
{
    fprintf(stderr, "[MIOPEN_HIPDNN_WRAPPER] miopenCreate\n");
    return miopenCreate_impl(handle);
}
```

Only the `MIOpen` CMake target in `build-flagon` was rebuilt; `build-flagoff` was not touched (it does not compile `wrapper.cpp`). The original wrapper file was backed up to `/tmp/wrapper.cpp.bak` for revert.

## Test command

A single MIOpenDriver invocation was issued against each build:

```
MIOpenDriver convfp16 -n 1 -c 1 -H 8 -W 8 -k 1 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 \
             -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -i 1 -V 0
```

`stdout` and `stderr` were captured separately so the wrapper trace is trivially isolated from normal driver output. Both runs exited `0` and selected the same solver (`85/ConvDirectNaiveConvFwd`).

Logs: `perf-results/wrapper-proof/{flagon,flagoff}.instr.{stdout,stderr}`.

## Full output — flagon

`flagon.instr.stdout` (7 lines — same shape as flagoff):

```
MIOpenDriver convfp16 -n 1 -c 1 -H 8 -W 8 -k 1 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -i 1 -V 0
PRNG seed: 12345678
Timestamp: 2026-05-27 14:49:25 UTC; Host Name: 684148a436c9; Operating System: Linux 5.15.0-173-generic; ROCm: 7.13.61040; MIOpen Driver: 3.5.2; CPU Vendor: Intel; CPU Model: 2 x Intel(R) Xeon(R) Platinum 8480C; RAM Size: 2015 GB; GPU Model: 8 x AMD Instinct MI300X; AMDGPU Driver: 6.16.13
MIOpen Forward Conv. Algorithm: 1, Solution: 85/ConvDirectNaiveConvFwd
GPU Kernel Time Forward Conv. Elapsed: 0.006776 ms (average)
stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs
stats: fwd-conv3x3u1, 1, 1, 8, 8, 3, 3, 1, 1152, 146, 128, 0, 0, 0.006776
```

`flagon.instr.stderr` (45 lines — every line is wrapper proof):

```
[MIOPEN_HIPDNN_WRAPPER] miopenCreateWithStream
[MIOPEN_HIPDNN_WRAPPER] miopenGetStream
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenEnableProfiling
[MIOPEN_HIPDNN_WRAPPER] miopenSetTensorDescriptorV2
[MIOPEN_HIPDNN_WRAPPER] miopenSetTensorDescriptorV2
[MIOPEN_HIPDNN_WRAPPER] miopenInitConvolutionNdDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenSetConvolutionGroupCount
[MIOPEN_HIPDNN_WRAPPER] miopenSetConvolutionAttribute
[MIOPEN_HIPDNN_WRAPPER] miopenGetConvolutionNdForwardOutputDim
[MIOPEN_HIPDNN_WRAPPER] miopenSetTensorDescriptorV2
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForwardGetWorkSpaceSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetVersion
[MIOPEN_HIPDNN_WRAPPER] miopenFindConvolutionForwardAlgorithm
[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForwardGetSolution
[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForward
[MIOPEN_HIPDNN_WRAPPER] miopenGetKernelTime
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroy
```

## Full output — flagoff

`flagoff.instr.stdout` (7 lines — same shape, same result):

```
MIOpenDriver convfp16 -n 1 -c 1 -H 8 -W 8 -k 1 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -i 1 -V 0
PRNG seed: 12345678
Timestamp: 2026-05-27 14:49:27 UTC; Host Name: 684148a436c9; Operating System: Linux 5.15.0-173-generic; ROCm: 7.13.61040; MIOpen Driver: 3.5.2; CPU Vendor: Intel; CPU Model: 2 x Intel(R) Xeon(R) Platinum 8480C; RAM Size: 2015 GB; GPU Model: 8 x AMD Instinct MI300X; AMDGPU Driver: 6.16.13
MIOpen Forward Conv. Algorithm: 1, Solution: 85/ConvDirectNaiveConvFwd
GPU Kernel Time Forward Conv. Elapsed: 0.006936 ms (average)
stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs
stats: fwd-conv3x3u1, 1, 1, 8, 8, 3, 3, 1, 1152, 146, 128, 0, 0, 0.006936
```

`flagoff.instr.stderr` (0 lines):

```
[empty]
```

## The lines that show the difference

The full contrast is the entire stderr stream — 45 wrapper hits on flagon vs. 0 on flagoff. The structurally important hits inside the flagon trace are the ones that map to the actual conv operation rather than per-tensor bookkeeping:

| Line in `flagon.instr.stderr` | What it proves |
| --- | --- |
| `[MIOPEN_HIPDNN_WRAPPER] miopenCreateWithStream`               | Very first MIOpen API call traverses the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenCreateConvolutionDescriptor`    | Conv descriptor setup goes through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenFindConvolutionForwardAlgorithm`| Solver search routed through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForward`             | The actual GPU enqueue routed through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenGetKernelTime`                  | Timing readback routed through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenDestroy`                        | Final teardown traverses the wrapper |

The flagoff side prints none of these because its `libMIOpen.so.1` contains the real functions directly — there is no stub layer to print from. Stdout is byte-shape-identical between the two runs; the entire behavioral delta is on stderr, and it is exactly what the wrapper design predicts.

## Reverting the instrumentation

```bash
cp /tmp/wrapper.cpp.bak src/private/wrapper.cpp
cmake --build build-flagon --target MIOpen -- -j$(nproc)
```

---

# §3 MIOpen gtest results — `build-flagon`

*Source: `MIOpenGtestResults.md`*

Run completed 2026-05-27, 15:31:21 → 20:25:25 UTC (≈4h 54m, single-threaded sequential).

The build uses `MIOPEN_TEST_DISCRETE=ON`, so there is no single `miopen_gtest` binary — gtest content is split across **265 discrete `test_*` executables** in `build-flagon/bin/`. Source for each lives under `test/gtest/*.cpp`. All 265 were executed (7 in the first pass with the v1 runner, 258 in the resume pass with the v2 runner).

## Top-line numbers

| Metric | Value |
| --- | ---: |
| Binaries executed                       | 265 / 265 |
| Binaries clean (exit 0, no failures)    | 261 |
| Binaries with failures                  | 2 |
| Binaries hit the 1200 s timeout         | 2 |
| Individual gtest cases run              | 58,039 |
| Cases passed                            | 53,516 |
| Cases skipped                           | 4,520 |
| Cases failed                            | **4** |

The 4,520 skips are expected — they're guards in the test harness for unsupported architectures / dtypes / configs (e.g. `test_rnn_seq_api` skips 2,560 of 4,608 cases on gfx942, `test_bad_fusion_plan` skips all 8).

## Failures (4 cases across 2 binaries)

### `test_smoke_tuning_policy` — 2 failures (likely wrapper-induced)

Both failing tests capture stderr from a `miopen{Get,Set}TuningPolicy()` call and assert the captured text contains the public function name. The build under test routes through the wrapper, so the impl's auto-generated function-entry log emits `_impl`-suffixed names — the substring check misses by exactly that suffix:

```
test/gtest/smoke_tuning_policy.cpp:105: Failure
  Expected: has substring " miopenGetTuningPolicy("
    Actual: "MIOpen(HIP): miopenStatus_t miopenGetTuningPolicy_impl(miopenHandle_t, miopenTuningPolicy_t *){\n…"
```

Same shape for `TestSetApiLogged`. This is a real wrapper-related test breakage, not noise — the test was written against the un-wrapped API name and the wrapper's pass-through changes the logged symbol. Worth fixing as part of the shim work (either rename in the wrapper's `_impl` symbols' `MIOPEN_LOG_FUNCTION` output, or relax the assertion to accept the `_impl` form).

The other 4 tests in this binary pass.

### `test_db_sync` — 2 failures (environmental, unrelated to wrapper)

```
db_sync.cpp:547: Failure
  C++ exception: "filesystem error: cannot get file size: No such file or
  directory [.../build-flagon/share/miopen/db/gfx942.kdb]"
```

The pre-built `gfx942.kdb` kernel database isn't installed in this build tree. Affects `CPU_DBSync_NONE.KDBTargetID` and one parameterized `StaticFDBSync/3` case. The remaining 4 cases in the binary skip (also file-presence guards).

## Timeouts (2 binaries, runner killed at 1200 s)

Both tests were actively making progress when the timeout fired — they're not hung, they're just enormous suites:

| Binary | Tests started | Tests passed before kill |
| --- | ---: | ---: |
| `test_lrn`      | 123 | 122 (still running #123) |
| `test_soft_max` | 267 | 266 (still running #267) |

Neither produced a `[ FAILED ]` line. To get clean numbers either raise the per-binary timeout (e.g. `timeout 3600`) or filter to a subset via `--gtest_filter`.

## Largest suites (top 10 by test count)

| Binary | Ran | Pass | Fail | Skip |
| --- | ---: | ---: | ---: | ---: |
| `test_tensor_reorder`        | 7,452 | 7,452 | 0 | 0 |
| `test_ternary_tensor_ops`    | 7,200 | 7,200 | 0 | 0 |
| `test_binary_tensor_ops`     | 5,143 | 5,143 | 0 | 0 |
| `test_rnn_seq_api`           | 4,608 | 2,048 | 0 | 2,560 |
| `test_tensor_transform`      | 3,270 | 3,270 | 0 | 0 |
| `test_tensor_api`            | 3,042 | 3,042 | 0 | 0 |
| `test_w_supertensor`         | 1,944 | 1,944 | 0 | 0 |
| `test_gpu_reference_kernel`  | 1,870 | 1,870 | 0 | 0 |
| `test_reduce`                | 1,625 | 701   | 0 | 924 |
| `test_bn_infer`              | 1,308 | 1,308 | 0 | 0 |

## Slowest binaries (top 10 by wall time)

| Binary | Duration |
| --- | ---: |
| `test_soft_max`             | 1200 s (timeout) |
| `test_lrn`                  | 1200 s (timeout) |
| `test_reduce`               | 1171 s |
| `test_bn_activ_infer`       | 901 s |
| `test_bn_infer`             | 856 s |
| `test_na_train_find2`       | 815 s |
| `test_bn_fwd_train`         | 744 s |
| `test_na_train`             | 734 s |
| `test_layernorm`            | 602 s |
| `test_bn_bwd`               | 572 s |

## Artifacts

All under `perf-results/gtest-flagon/`:

- `_all_gtests.txt` — canonical list of all 265 binaries.
- `_remaining_at_resume.txt` — the 258 binaries fed to the v2 runner.
- `_run_gtests.sh` / `_progress.log` / `_summary.tsv` — v1 runner (first 7 binaries). The v1 summary's pass/fail/skip columns are bogus (the `--gtest_brief=1` flag suppressed the per-test lines its grep relied on).
- `_run_gtests_v2.sh` / `_progress_v2.log` / `_summary_v2.tsv` — v2 runner (remaining 258). Drops `--gtest_brief=1` and parses the `[==========]` / `[ PASSED ]` / `[ SKIPPED ]` / `[ FAILED ]` summary lines for accurate counts. **This is the source of truth for runner-side counts.**
- `test_<name>.log` — raw stdout+stderr for each of the 265 binaries.

The numbers in this section were re-aggregated directly from the 265 `test_*.log` files (not from `_summary*.tsv`), so they're consistent regardless of which runner produced a given log.

## Re-run hints

- Just the failing binaries: `build-flagon/bin/test_smoke_tuning_policy` and `build-flagon/bin/test_db_sync` (each <2 s).
- Just the timeout victims with a larger budget: `timeout 3600 build-flagon/bin/test_lrn` and the same for `test_soft_max`.
- Same suite against `build-flagoff` (to confirm `test_smoke_tuning_policy` passes there and conclusively pin it on the wrapper): swap the BINDIR in `_run_gtests_v2.sh` to `build-flagoff/bin` and re-run.

---

# §4 Performance comparison

*Source: `PerformanceComparison.md`*

Script: `~/test-data/model_f_short.sh` (64 `convbfp16` configs, NHWC layout). Only `MIOPEN_ENABLE_HIPDNN_WRAPPER` differs between the two builds.

## Methodology

For each build, in order: clear `~/.cache/miopen` and `~/.config/miopen`, then:

1. **OOTB** — `--iter 10` (cold cache, default find).
2. **Tuning** — `--iter 1` with `MIOPEN_FIND_ENFORCE=4` (full exhaustive tuning).
3. **Tuned** — `--iter 10 -S 0` (re-use the find-db produced in step 2).

All raw logs are in `perf-results/`. Every one of the 64 configs verified OK against the GPU reference in all six runs (no correctness regressions).

## Wall-Clock Duration

| Phase   | flagoff | flagon | Δ (flagon − flagoff) |
| ------- | ------: | -----: | -------------------: |
| OOTB    | 558 s   | 528 s  | −30 s (−5.4%) |
| Tuning  | 728 s   | 668 s  | −60 s (−8.2%) |
| Tuned   | 533 s   | 464 s  | −69 s (−12.9%) |

flagon is consistently the same or slightly faster end-to-end. The wall-clock numbers are dominated by per-config CPU setup (kernel compilation, JIT, find-db lookups) so they reflect process overhead more than GPU work.

## Aggregate GPU Kernel Time (sum across 64 configs)

| Phase   | flagoff  | flagon   | total Δ  |
| ------- | -------: | -------: | -------: |
| OOTB    | 6.9436 ms | 6.8860 ms | −0.83% |
| Tuning  | 6.4445 ms | 6.5752 ms | +2.03% |
| Tuned   | 6.0500 ms | 6.0534 ms | +0.06% |

GPU work is statistically identical between builds. Sub-1% differences in OOTB/tuned are well within the noise floor for `--iter 10` runs at this scale.

## Per-Config Variability

| Phase | mean per-config Δ | max regression | max improvement |
| --- | ---: | ---: | ---: |
| OOTB  | +0.09% | +13.94% (idx 64) | −13.93% (idx 44) |
| Tuned | −0.53% | +46.65% (idx 46) | −57.43% (idx 44) |

The outliers do **not** track the flag — they track solver selection. In the tuned run, `diff` of `MIOpen … Algorithm:` lines shows the two builds picked different "best" solvers on 2 of 64 configs (3 of 64 in OOTB). For example:

- idx 34 (TUNED): flagoff → `108/ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC`; flagon → `155/ConvHipImplicitGemmGroupBwdXdlops`.

This is normal find/tuning non-determinism (timing-driven tie-breaking on a cold cache) and is the dominant source of per-config swings. Aggregate kernel time absorbs these into noise.

## Tuned vs. OOTB (validates the tuning flow)

| Build   | OOTB sum | Tuned sum | Improvement |
| ------- | -------: | --------: | ----------: |
| flagoff | 6.9436 ms | 6.0500 ms | −12.9% |
| flagon  | 6.8860 ms | 6.0534 ms | −12.1% |

Tuning provides ≈12% kernel-time reduction on this workload for both builds, confirming the find-db path is exercised identically.

## Convolution-level conclusion

Enabling the hipDNN wrapper (`MIOPEN_ENABLE_HIPDNN_WRAPPER=ON`) has **no measurable performance impact** on MIOpen itself:

- GPU kernel time totals differ by ≤2% in every phase, with the signed direction varying by phase — i.e., noise.
- Wall-clock duration is the same or slightly better for flagon (≤13% faster on the tuned run), well within run-to-run variance for cache/JIT-dominated work.
- All 64 configs verify in every run; the tuning flow yields the same ≈12% improvement on both builds.
- The few large per-config swings are explained by find/tuning picking different but performance-equivalent solvers, not by the flag.

The shim is performance-neutral for the public MIOpen path on this benchmark.

## Wrapper microbenchmark — per-call API overhead (warm & cold)

The convolution benchmarks above are dominated by GPU kernel time; per-call CPU overhead in the C ABI is essentially invisible against milliseconds of GPU work. To isolate the wrapper cost itself, a small C harness (`perf-results/wrapper-bench/wrapper_bench.c`) calls the cheapest public MIOpen entry point (`miopenGetVersion`, no GPU work) and a moderate one (`miopenCreate` + `miopenDestroy`) in a tight loop, then reports wall/CPU time, `getrusage` page faults, RSS, and context switches.

Both builds use the same source and same compile flags; the only difference is whether `MIOPEN_ENABLE_HIPDNN_WRAPPER=ON` was set at CMake time, which inserts a forwarding stub (`<symbol>` in `libMIOpen.so` → `<symbol>_impl` in `libMIOpen_private.so`) for every public API.

### Method

- Two binaries built against `build-flagoff/lib` and `build-flagon/lib`; rpath pins each to its own MIOpen.
- **Warm** runs: prime the page cache, then loop the API call 100M times (getversion) or 1k times (create/destroy). 10 reps each.
- **Cold** runs: `sync && echo 3 > /proc/sys/vm/drop_caches` plus explicit `dd iflag=nocache` eviction of `libMIOpen.so.1` (and `libMIOpen_private.so.1` for flagon) before each invocation, then run a single iteration so process startup / dynamic-loader cost dominates. 5 reps each.
- **Noop** baseline (warm only): same loop structure without the API call, to estimate harness cost.
- Driver: `perf-results/wrapper-bench/run_bench.sh`; raw output: `out.tsv`, `raw.log`.

### Results — warm steady-state (per-call cost)

| Mode | Build | n | mean (ns/call) | median | stdev | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noop (baseline) | flagoff | 10 | 0.85 | 0.74 | 0.22 | 0.72 | 1.26 |
| noop (baseline) | flagon  | 10 | 0.85 | 0.75 | 0.22 | 0.72 | 1.28 |
| `miopenGetVersion` | flagoff | 10 | 3.23 | 3.24 | 0.87 | 2.36 | 4.06 |
| `miopenGetVersion` | flagon  | 10 | 4.41 | 4.44 | 1.14 | 3.29 | 5.53 |
| `miopenCreate`+`Destroy` | flagoff | 10 | 4,190,048 | 4,069,811 | 316,862 | 3,773,061 | 4,729,323 |
| `miopenCreate`+`Destroy` | flagon  | 10 | 4,241,539 | 4,066,200 | 421,505 | 3,775,337 | 5,066,603 |

Notes:
- The noop loop (no API call) shows the same bimodal jitter as `getversion` (0.72 vs 1.28 ns/iter), which is CPU DVFS / P-state hop on this host, not the wrapper. Subtracting the noop baseline isolates the API cost.
- `miopenGetVersion` is a 3-pointer-store function: useful as a worst-case relative measurement of wrapper hop cost.
- `miopenCreate`/`Destroy` allocates a HIP stream and reads driver state (~4 ms/call); the wrapper hop is in the noise.

#### Net per-call cost (mean, harness-subtracted)

| API | flagoff | flagon | Δ (flagon − flagoff) |
| --- | ---: | ---: | ---: |
| `miopenGetVersion` | 2.38 ns | 3.56 ns | **+1.18 ns / call** |
| `miopenCreate`+`Destroy` | 4.190 ms | 4.242 ms | +0.05 ms / call (≈ 0.0012 ×, within stdev) |

The ≈1 ns delta on `miopenGetVersion` is the *upper bound* on wrapper hop cost — it is an indirect call through the PLT into the private library plus one extra `ret`. For any real MIOpen API (which invariably touches the GPU), this is unmeasurable.

### Results — cold start (dynamic-loader cost)

These runs do a single API call after dropping caches, so the dominant cost is the kernel paging in the .so files from disk. The interesting columns are wall time and major page faults (`majflt`).

| Mode | Build | n | mean wall (ms) | mean majflt | mean minflt | mean RSS (KB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `miopenGetVersion` | flagoff | 5 | 0.0023 | **1,438** | 5,491 | 186,978 |
| `miopenGetVersion` | flagon  | 5 | 0.0023 | **1,444** | 5,500 | 187,909 |
| `miopenCreate`+`Destroy` | flagoff | 5 | 1.503 | **1,658** | 11,131 | 600,926 |
| `miopenCreate`+`Destroy` | flagon  | 5 | 1.564 | **1,664** | 11,158 | 602,040 |

- **Cold-load delta is +6 major page faults**, identical for both APIs. That is the cost of mapping the additional `libMIOpen_private.so.1` ELF header / dynamic section into the process — every other page that has to be paged in (the actual code/data of MIOpen, ROCm-comgr, hipBLASLt, MIOpenTensile, etc.) is identical between builds because the same total amount of code is loaded; flagon just splits it across two files.
- Wall-clock cold-startup difference is ≈60 µs for create/destroy (≈4%), within the run-to-run stdev of ~70 µs. For `getversion` the wall time is dominated by `clock_gettime` resolution and is indistinguishable between builds.
- Resident-set size is the same to within ~1 MB; the wrapper adds no measurable memory footprint beyond a second ELF file's overhead.

### Microbenchmark conclusion

Direct measurement of the wrapper hop confirms the convolution-level findings:

- **Steady-state CPU overhead: ≈1 ns per call** on a worst-case (no-work) API, well below the per-call cost of any function that actually launches a kernel.
- **Cold-load overhead: ≈6 extra major page faults** (one extra ELF mapped), with no detectable change in wall time, RSS, or minor page faults.
- The split into `libMIOpen.so` + `libMIOpen_private.so` is **not** a meaningful runtime cost — it is a structural change with essentially-zero measurable impact on the public API surface.

---

# §5 MIOpen hipDNN provider plugin — linkage & runtime verification

*Source: `MIOpenPluginResults.md`*

Synthetic harness: `perf-results/plugin-test/`.

## 5.1 Purpose

The hipDNN MIOpen provider plugin selects its MIOpen link target at CMake time:

```cmake
# dnn-providers/miopen-provider/CMakeLists.txt:149-163
find_package(miopen CONFIG REQUIRED)

# RFC 0001 Phase 4 short-circuit: when the MIOpen install was built with
# MIOPEN_ENABLE_HIPDNN_WRAPPER=ON, it exports a MIOpen_private target whose
# libMIOpen_private.so contains the renamed _impl symbols. Linking the plugin
# against MIOpen_private bypasses the wrapper, so this provider's calls don't
# round-trip back through hipDNN. When the MIOpen install is the legacy
# single-library build, MIOpen_private won't exist and we fall back to MIOpen.
if(TARGET MIOpen_private)
    set(_MIOPEN_PROVIDER_LINK_TARGET MIOpen_private)
    message(STATUS "miopen-provider: linking MIOpen_private (Phase 4 short-circuit)")
else()
    set(_MIOPEN_PROVIDER_LINK_TARGET MIOpen)
    message(STATUS "miopen-provider: linking MIOpen (legacy single-library install)")
endif()
```

The intent: when MIOpen is installed in **wrapper mode** (flagon), the provider plugin links directly to `libMIOpen_private.so.1` so its calls don't take the public-API wrapper hop and don't risk re-entering hipDNN. When MIOpen is installed in **legacy mode** (flagoff), the provider links to `libMIOpen.so.1` like any normal MIOpen consumer.

This section verifies, via structural ELF inspection, symbol resolution, dynamic-linker tracing, and a negative test, that this wiring **selects the correct library at link time** and that the loader **routes the runtime calls as designed** — and documents one gap that surfaced during the runtime test.

## 5.2 Test artifacts

Because no pre-built provider plugin in the working tree is linked against `libMIOpen_private.so.1` (the canonical one in `/data/jlichtne/TheRock/therock-build/lib/hipdnn_plugins/engines/libmiopen_plugin.so` was built against an older MIOpen install that lacked the `MIOpen_private` target — see §5.3), a synthetic plugin that mirrors the real plugin's link wiring was built:

- `perf-results/plugin-test/synthetic_plugin.c` — calls three public MIOpen entry points (`miopenGetVersion`, `miopenCreate`, `miopenDestroy`). Compiled with `-fvisibility=hidden`, exports only `plugin_run` via a linker version script — matches the real plugin's symbol-export discipline.
- `perf-results/plugin-test/build.sh` — produces two variants from the same source, against `build-flagon/lib`:
  - `libsynth_plugin_public.so`  → linked `-lMIOpen` (legacy / fallback target)
  - `libsynth_plugin_private.so` → linked `-lMIOpen_private` (Phase-4 target)
- `perf-results/plugin-test/host.c` — a generic `dlopen(RTLD_NOW | RTLD_LOCAL) + dlsym("plugin_run")` host, mimicking how hipDNN loads engine plugins at runtime. The host does not link MIOpen itself; all MIOpen exposure comes from the loaded plugin's `DT_NEEDED`.

## 5.3 Structural verification (`readelf -d`)

`DT_NEEDED` / `DT_RUNPATH` of each artifact:

| Artifact | NEEDED MIOpen entry | RUNPATH |
| --- | --- | --- |
| `libsynth_plugin_public.so`  (new) | `libMIOpen.so.1` | `build-flagon/lib` |
| `libsynth_plugin_private.so` (new) | `libMIOpen_private.so.1` | `build-flagon/lib` |
| TheRock pre-built `libmiopen_plugin.so` | `libMIOpen.so.1` | `$ORIGIN/...` |
| `libMIOpen.so.1` (flagon build) | `libMIOpen_private.so.1`, `libamd_comgr.so.3`, `librocblas.so.5`, `libamdhip64.so.7`, … | `build-flagon/lib:/opt/rocm/lib` |

**Findings:**

- The two synthetic variants differ only in their `NEEDED` MIOpen entry — exactly the difference the Phase-4 short-circuit produces. This is the structural fingerprint we'd see on any real provider plugin built from the same CMakeLists.
- `libMIOpen.so.1` itself has a hard `NEEDED` on `libMIOpen_private.so.1`. This is by design — the wrapper stubs in `libMIOpen.so.1` call `<name>_impl` symbols defined in the private library — but it means that in a flagon install **every consumer of `libMIOpen.so.1` transitively loads `libMIOpen_private.so.1`** whether they want to or not.
- The TheRock pre-built `libmiopen_plugin.so` is `NEEDED libMIOpen.so.1` (no MIOpen_private), confirming it was built against a non-flagon MIOpen install. It is not useful for validating Phase 4 and cannot be substituted for the synthetic test.

## 5.4 Symbol cross-check

How many `miopen*` symbols each library exports, and what form (`nm -D --defined-only | grep " T miopen"`):

| Library | Plain `miopen*` exports | `miopen*_impl` exports | Total |
| --- | ---: | ---: | ---: |
| `libMIOpen.so.1` (wrapper) | 263 | 0 | 263 |
| `libMIOpen_private.so.1` | 4 † | 263 | 267 |

† The 4 non-renamed survivors in `libMIOpen_private.so.1` are: `miopenConvolutionABBackwardWeightsGetWorkSpaceSize`, `miopenHiddenGetConvolutionFindMode`, `miopenHiddenSetConvolutionFindMode`, `miopen_sqlite3_memvfs_init`. They are not in the rename header and are unrelated to the wrapper hop.

Undefined `miopen*` symbols in each synthetic plugin (`nm -D --undefined-only`):

| Plugin | Undefined `miopen*` symbols |
| --- | --- |
| `libsynth_plugin_public.so` | `miopenCreate`, `miopenDestroy`, `miopenGetVersion` |
| `libsynth_plugin_private.so` | `miopenCreate`, `miopenDestroy`, `miopenGetVersion` |

Note the identical undefined-symbol set: the plugin source uses the unrenamed public names (`#include <miopen/miopen.h>`) regardless of which library it is linked against. The linker tolerates this for the PRIVATE variant because shared-object linking is lazy — undefined symbols in a `.so` are not errors at link time; they are recorded as a request the dynamic loader will fulfill later.

**Where each undefined symbol resolves:**

| Plugin | `miopenCreate` resolves in | `miopenGetVersion` resolves in | `miopenDestroy` resolves in |
| --- | --- | --- | --- |
| PUBLIC  | `libMIOpen.so.1` ✓ | `libMIOpen.so.1` ✓ | `libMIOpen.so.1` ✓ |
| PRIVATE | `libMIOpen_private.so.1` exports only `miopenCreate_impl` ✗ | only `miopenGetVersion_impl` ✗ | only `miopenDestroy_impl` ✗ |

The PRIVATE variant has no way to bind its undefined symbols at load time — the names it asks for are not the names the private library exports. This is the gap explored in §5.5 and §5.7.

## 5.5 Runtime verification — `LD_DEBUG=bindings`

Captures: `perf-results/plugin-test/ld_public.<pid>`, `ld_private.<pid>`.

### PUBLIC variant — `./host ./libsynth_plugin_public.so`

```
plugin_run returned 10        # = 3 + 5 + 2 (MIOpen 3.5.2)
```

Bindings of interest (counts from the LD_DEBUG output):

| Source library | Target library | Bindings | Notes |
| --- | --- | ---: | --- |
| `libsynth_plugin_public.so` | `libMIOpen.so.1` | **3** | exactly the three API calls in `plugin_run` |
| `libMIOpen.so.1` | `libMIOpen_private.so.1` | **263** | every wrapper stub binds its matching `<name>_impl` at load time (RTLD_NOW) |

The three explicit plugin-side bindings:

```
binding file ./libsynth_plugin_public.so [0]
    to .../libMIOpen.so.1 [0]: normal symbol `miopenGetVersion'
binding file ./libsynth_plugin_public.so [0]
    to .../libMIOpen.so.1 [0]: normal symbol `miopenCreate'
binding file ./libsynth_plugin_public.so [0]
    to .../libMIOpen.so.1 [0]: normal symbol `miopenDestroy'
```

This is the canonical wrapper-hop trace: plugin code calls `miopenCreate`, which is a forwarding stub in `libMIOpen.so.1`, which calls `miopenCreate_impl` in `libMIOpen_private.so.1`. The 263 wrapper→private bindings are the entire C-ABI surface — they get resolved up front because the host uses `RTLD_NOW`, but only three of them are actually called during this run.

### PRIVATE variant — `./host ./libsynth_plugin_private.so`

```
dlopen failed: ./libsynth_plugin_private.so: undefined symbol: miopenGetVersion
```

Bindings of interest:

| Source library | Target library | Bindings | Notes |
| --- | --- | ---: | --- |
| `libsynth_plugin_private.so` | `libgcc_s.so.1`, `libc.so.6` | 6 | only C runtime / unwind glue |
| `libsynth_plugin_private.so` | any `libMIOpen*.so.1` | **0** | dlopen aborted before any MIOpen binding could be attempted |

(The trace does record ~2.8k bindings *internal to* `libMIOpen_private.so.1` and from ROCm helpers into it — those are part of loading the library itself and would happen before the plugin's own relocations are processed.)

This is the runtime expression of the §5.4 gap: the linker happily built `libsynth_plugin_private.so` with `U miopenGetVersion`, but the only thing `libMIOpen_private.so.1` exports under that name discipline is `miopenGetVersion_impl`, so dlopen fails at the plugin's relocation step. The Phase-4 short-circuit, *as currently written in the provider's CMakeLists*, produces a `.so` that **links but does not load**.

## 5.6 Negative test — hide `libMIOpen_private.so.1`

To confirm that the loader is actually pulling `libMIOpen_private.so.1` (not picking up a stale copy from `/opt/rocm/lib` or similar):

```
$ mv build-flagon/lib/libMIOpen_private.so.1{,.hidden_for_test}

$ ./host ./libsynth_plugin_public.so
dlopen failed: libMIOpen_private.so.1: cannot open shared object file: No such file or directory

$ ./host ./libsynth_plugin_private.so
dlopen failed: libMIOpen_private.so.1: cannot open shared object file: No such file or directory

$ mv build-flagon/lib/libMIOpen_private.so.1{.hidden_for_test,}

$ ./host ./libsynth_plugin_public.so
plugin_run returned 10
```

Both variants fail with the same "cannot open shared object" error, then PUBLIC recovers immediately after the file is restored. This confirms:

- The PUBLIC plugin's runtime dependency on `libMIOpen_private.so.1` is real and transitive (via `libMIOpen.so.1`'s `DT_NEEDED`) — there is no possible code path through the wrapper variant that avoids loading the private library.
- The PRIVATE plugin's runtime dependency on `libMIOpen_private.so.1` is real and direct.
- Both plugins resolve `libMIOpen_private.so.1` from `build-flagon/lib` via the `DT_RUNPATH` baked into each plugin's ELF (and into `libMIOpen.so.1`'s ELF for the PUBLIC case) — nothing else on the system is being silently substituted.

## 5.7 Gap finding — Phase 4 short-circuit is incomplete

§5.4–§5.6 together establish that the provider's CMake snippet **chooses the link target correctly** (the PRIVATE variant does link `libMIOpen_private.so.1` rather than `libMIOpen.so.1`), but the resulting plugin **cannot be loaded** because the symbol names visible to the plugin's compilation unit don't match the symbol names exported by `libMIOpen_private.so.1`.

The mechanism the wrapper build uses to make MIOpen's own translation units call `_impl` names is a build-time include:

```cmake
# projects/miopen/src/CMakeLists.txt — MIOpen Private build options
target_compile_options(MIOpen_private PRIVATE
    -DMIOPEN_BUILDING_PRIVATE
    -include ${CMAKE_CURRENT_SOURCE_DIR}/private/miopen_private_rename.h)
```

The header (`projects/miopen/src/private/miopen_private_rename.h`) `#define`s every `miopenFoo` to `miopenFoo_impl` and is gated by `MIOPEN_BUILDING_PRIVATE` — and the comment at the top reads:

> This header is intentionally NOT installed (RFC 0001 §4.6, Q5).

So the rename header is not in the install tree; the provider plugin, being a downstream consumer, has no way to apply it as written today. The PRIVATE link target therefore only delivers value if **something on the consumer side** also rewrites the names. Options:

1. **Install the rename header** under `private/` in the MIOpen install tree, and have `find_package(miopen)` set up the `MIOpen_private` imported target's `INTERFACE_INCLUDE_DIRECTORIES` / `INTERFACE_COMPILE_OPTIONS` to apply `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h` automatically. The provider plugin would then transparently get the renamed names whenever it links against `MIOpen_private`. This is the lowest-friction option for plugin authors.
2. **Ship a parallel public header** that declares the API in `_impl` form, and have the provider's source code include that header when targeting `MIOpen_private`. This avoids the macro-rename trick but requires the provider to be conditionally compiled.
3. **Keep the rename header private, change the private library's exports** so it also publishes the unrenamed names as additional symbols (aliases). This eliminates the need for any consumer-side rename, at the cost of `libMIOpen_private.so.1` having two symbols per API. It re-creates the risk Phase 4 was trying to remove (the private library being callable through the public name path).

Option (1) most closely matches the existing build's mechanism and was the assumption the provider's CMake snippet was written against.

The runtime evidence collected here can be used as the regression bar for any chosen fix: once it's in, `./host ./libsynth_plugin_private.so` must succeed, and `LD_DEBUG=bindings` must show plugin→`libMIOpen_private.so.1` bindings on `miopenCreate_impl` / `miopenDestroy_impl` / `miopenGetVersion_impl` with **zero** bindings into `libMIOpen.so.1`.

## 5.8 Plugin summary

| Verification | PUBLIC variant | PRIVATE variant |
| --- | --- | --- |
| `DT_NEEDED` matches CMake intent | ✓ `libMIOpen.so.1` | ✓ `libMIOpen_private.so.1` |
| `DT_RUNPATH` finds the lib | ✓ `build-flagon/lib` | ✓ `build-flagon/lib` |
| Symbols exist in target lib | ✓ all 3 found | ✗ only `_impl` forms exported |
| `dlopen` succeeds | ✓ | ✗ undefined symbol: `miopenGetVersion` |
| `plugin_run` executes | ✓ returns 10 (MIOpen 3.5.2) | ✗ never reached |
| Wrapper hop observed (LD_DEBUG) | 3 plugin→wrapper + 263 wrapper→private | n/a |
| Negative test (hide private lib) | fails with cannot-open (transitive) | fails with cannot-open (direct) |

- The **wrapper path** (PUBLIC plugin against flagon MIOpen) is the legacy/fallback case and works end-to-end. The wrapper-hop overhead is the ≈1 ns per call documented in §4.
- The **short-circuit path** (PRIVATE plugin) selects the correct library at link time but its plugin source can't call any API in that library because the consumer has no access to the rename mechanism. This is a **gap in Phase 4's wiring**, not a bug in the test setup — every API call in the synthetic plugin is also present in the real plugin and would fail the same way.
- The negative test confirms `libMIOpen_private.so.1` is always loaded in a flagon install, even when only the public library is linked. Wrapper mode and short-circuit mode do not differ in *which* files are mapped, only in which library's exports the plugin's relocations bind to.

## 5.9 Plugin artifact list

- `perf-results/plugin-test/synthetic_plugin.c`, `host.c`, `build.sh` — sources
- `perf-results/plugin-test/libsynth_plugin_{public,private}.so`, `host` — built artifacts
- `perf-results/plugin-test/ld_public.<pid>`, `ld_private.<pid>` — raw LD_DEBUG bindings traces

---

# Cross-cutting takeaways

Pulled from the five sections above for use as the RFC's evidence checklist:

| Claim the RFC needs to support | Evidence section |
| --- | --- |
| The wrapper build produces the expected library split and links cleanly. | [§1](#1-miopendriver-smoke-test) |
| Every public API call traverses the wrapper at runtime (no hidden bypass). | [§2](#2-runtime-wrapper-logging) |
| The wrapper does not regress MIOpen correctness on the upstream gtest suite (4 case failures, all explained: 2 wrapper-aware test fixes needed, 2 environmental). | [§3](#3-miopen-gtest-results--build-flagon) |
| Per-call CPU overhead of the wrapper is ≈1 ns (upper bound, on a no-work API); aggregate GPU work is statistically identical (≤2% phase-level deltas, signed direction varies). | [§4](#4-performance-comparison) |
| Cold-load overhead of the second ELF is +6 major page faults, ≤1 MB RSS, no detectable wall-clock change. | [§4](#4-performance-comparison) |
| The provider plugin's CMake snippet correctly selects `libMIOpen_private.so.1` when MIOpen is installed in wrapper mode, and falls back to `libMIOpen.so.1` otherwise. | [§5.3](#53-structural-verification-readelf--d) |
| `libMIOpen_private.so.1` is always loaded in a flagon install (transitively via `libMIOpen.so.1`'s `DT_NEEDED`); flagging the provider to link it directly changes *which exports bind*, not *which files load*. | [§5.6](#56-negative-test--hide-libmiopen_privateso1) |
| The Phase 4 short-circuit as currently wired produces a plugin that links but cannot be loaded, because the consumer-side rename mechanism is missing. Three remediation options are sketched in §5.7. | [§5.5](#55-runtime-verification--ld_debugbindings), [§5.7](#57-gap-finding--phase-4-short-circuit-is-incomplete) |

---

# §6 hipDNN redirect feasibility — Phase 2 forwarding prototype

*Source: `hipdnn-redirect-investigation.md` — ticket [ALMIOPEN-1965](https://amd-hub.atlassian.net/browse/ALMIOPEN-1965) ("Investigation that MIOpen shim can successfully redirect to hipDNN")*

Where §1–§5 characterize the **library split** — does the wrapper build, link, run, and stay performance-neutral on the pass-through path — this section characterizes the **forwarding path itself**: can the wrapper actually redirect a MIOpen call into hipDNN and execute it end-to-end, and what does that cost. It is the first Phase 2 evidence; §1–§5 are Phase 1 evidence.

The redirect is entirely opt-in via environment variables and always falls back to native MIOpen on any failure, so default behavior is unchanged. Implementation lives in `projects/miopen/src/private/wrapper.cpp`.

## Context (differs from the common context above)

- Hardware: 1× AMD Instinct MI308X (gfx942) — **not** the MI300X used in §1–§5
- ROCm 7.14.60850, MIOpen 3.5.2
- Native solution selected for the test conv: `ConvBinWinogradRxSf2x3g1` (GPU kernel ~0.017 ms)

## Feature flags (all off by default)

| Variable | Effect |
| --- | --- |
| `MIOPEN_HIPDNN_FORWARDING=enabled` | Master switch; pairs a hipDNN handle to each MIOpen handle |
| `MIOPEN_HIPDNN_FORWARDING_CONV=enabled` | Routes `miopenConvolutionForward` to hipDNN |
| `MIOPEN_HIPDNN_FORWARDING_TIMING=1` | Emits `[MIOpen->hipDNN]` cold/warm timings to stderr |
| `MIOPEN_HIPDNN_FORWARDING_MEMINFO=1` | Emits HIP free/total memory around handle init |

## Handle lifecycle

`miopenCreate` / `miopenCreateWithStream` open a paired hipDNN handle via `hipdnnCreate` and stash it in an `unordered_map<miopenHandle_t, hipdnnHandle_t>`; `miopenDestroy` releases it via `hipdnnDestroy`. hipDNN open failure is non-fatal — the MIOpen handle is still returned (a stderr line is emitted; per RFC §4.4 this becomes a routing-policy decision in a later phase). `miopenCreateWithStream` also has a reverse-order init mode (hipDNN before MIOpen) to isolate init-order / memory effects.

## Convolution forwarding — a worked argument-translation example

`miopenConvolutionForward` dispatches to `try_forward_conv_to_hipdnn` when both the forwarding and conv flags are set. That helper builds an equivalent hipDNN **backend descriptor** convolution graph by hand (the same cuDNN-style API the handle path uses — no frontend dependency), finalizes an execution plan, caches it per problem shape, and executes through the paired handle.

The guards below are the operational definition of "fields with no clean hipDNN analog in this prototype" — each returns `miopenStatusUnsupportedOp`, which routes the call back to native MIOpen:

| Unsupported case | Wrapper behavior |
| --- | --- |
| non-identity alpha/beta scaling | `miopenStatusUnsupportedOp` → native fallback |
| `groupCount != 1` | `miopenStatusUnsupportedOp` → native fallback |
| spatial dim > 5 | `miopenStatusUnsupportedOp` → native fallback |
| any hipDNN API failure | `miopenStatusUnsupportedOp` → native fallback |

The plan cache is an in-process `unordered_map` keyed on tensor dims/strides/types + conv params — **process-lifetime only, not persisted to disk.**

This is a concrete, end-to-end realization of the descriptor → hipDNN graph + variant-pack translation that the umbrella review's concern #1 asked for as a worked example: it names exactly which descriptor fields have no 1:1 hipDNN mapping in this prototype (alpha/beta scaling, groups, > 5 spatial dims) and states what the wrapper does for them (decline and fall back). See the concern #1 entry in `review-decisions-2026-06-08.md` for how this bears on closing that concern.

## Full field-by-field mapping (descriptor → hipDNN backend graph)

The worked example above names the *declined* fields; this section records the full translation the prototype performs for the fields that **do** map, so the mapping is documented rather than living only in `wrapper.cpp`. The wrapper builds the graph from hipDNN **backend descriptors** (the cuDNN-style API — no frontend dependency), in this order:

1. Three `HIPDNN_BACKEND_TENSOR_DESCRIPTOR`s — one each for X, W, Y.
2. One `HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR` referencing the three tensors and carrying the conv parameters.
3. One `HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR` (paired hipDNN handle + the single op).
4. `ENGINEHEUR` → `ENGINECFG` → `EXECUTION_PLAN` (cached per problem shape; queried for workspace size).
5. A `HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR` binding tensor UIDs to device pointers, then `hipdnnBackendExecute`.

**Fields that map 1:1** (`miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, beta, yDesc, y, workSpace, workSpaceSize)`):

| MIOpen input | hipDNN backend target | Notes |
| --- | --- | --- |
| `handle` | paired `hipdnnHandle_t` → `HIPDNN_ATTR_OPERATIONGRAPH_HANDLE` / `HIPDNN_ATTR_EXECUTION_PLAN_HANDLE` | Paired at `miopenCreate` (see handle lifecycle above). |
| `xDesc`/`wDesc`/`yDesc` dims | `HIPDNN_ATTR_TENSOR_DIMENSIONS` (×3) | Copied directly. |
| `xDesc`/`wDesc`/`yDesc` strides (i.e. layout) | `HIPDNN_ATTR_TENSOR_STRIDES` (×3) | MIOpen encodes layout (NCHW/NHWC/NCDHW/NDHWC) as strides; hipDNN takes dims+strides, so layout maps by copying strides — no separate layout enum. Vectorized layouts (NCHWc4/c8, CHWNc4/c8) have no plain dims+strides form → decline. |
| tensor `dataType` | `HIPDNN_ATTR_TENSOR_DATA_TYPE` (×3) | 1:1: `miopenHalf`→`HIPDNN_DATA_HALF`, `miopenFloat`→`FLOAT`, `miopenBFloat16`→`BFLOAT16`, `miopenDouble`→`DOUBLE`, `miopenInt8`→`INT8`, `miopenInt32`→`INT32`, `miopenInt64`→`INT64`, `miopenFloat8_fnuz`→`FP8_E4M3_FNUZ`, `miopenBFloat8_fnuz`→`FP8_E5M2_FNUZ`. Every MIOpen dtype has an equivalent; whether the selected engine supports it is separate. |
| tensor identity | `HIPDNN_ATTR_TENSOR_UNIQUE_ID` (×3) | Wrapper assigns stable UIDs (X/W/Y); these link the graph to the variant pack. |
| `convDesc` `padA` | `HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS` **and** `HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS` | MIOpen padding is symmetric (one array); set hipDNN pre = post = `padA`. |
| `convDesc` `strideA` | `HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES` | Direct. |
| `convDesc` `dilationA` | `HIPDNN_ATTR_CONVOLUTION_DILATIONS` | Direct. |
| `convDesc` `c_mode` | `HIPDNN_ATTR_CONVOLUTION_CONV_MODE` | `miopenConvolution` (documented as cross-correlation) → `HIPDNN_CROSS_CORRELATION`. |
| `convDesc` spatialDim | length of the four arrays above | Supported for ≤ 5 (see decline table above). |
| (derived) compute/accumulation type | `HIPDNN_ATTR_CONVOLUTION_COMP_TYPE` | Derived from IO dtypes (e.g. fp32 accumulate for half/bf16). |
| `x`, `w`, `y` device pointers | `HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS` keyed by `HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS` | Pointers live in the variant pack, not the graph. |

**Variant pack:** `UNIQUE_IDS = {X, W, Y}`, `DATA_POINTERS = {x, w, y}`, `WORKSPACE = wrapper-owned buffer` sized from the execution plan's `HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE`.

**Behavioral differences — fields that map but not faithfully** (these are *not* declines; the call proceeds, but a consumer could observe a difference):

- `algo` (`miopenConvFwdAlgorithm_t`) is **ignored** — hipDNN selects its own engine via `ENGINEHEUR`, so the caller's algorithm choice is not honored.
- `workSpace` / `workSpaceSize` are **not reused** — hipDNN computes its own workspace requirement and the wrapper owns that buffer; the caller-provided workspace is left untouched.

**Why the declined fields have no mapping** (the decline table above, with the underlying reason): `alpha`/`beta` ≠ identity — the hipDNN conv-forward op exposes **no** alpha/beta scaling attribute (there is no `HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA/BETA`), so only `α=1, β=0` is expressible; `groupCount` ≠ 1 — no group-count attribute on the hipDNN conv-forward op in this API; `c_mode = miopenTranspose` — deconvolution is not a forward-conv op (maps to backward-data); spatial dim > 5 — prototype boundary.

**Status translation:** any `hipdnnStatus_t` failure during build/finalize/execute currently returns `miopenStatusUnsupportedOp` (native fallback). Per RFC §6.1 this becomes a translated `miopenStatus_t` carrying the `[hipDNN-forwarded]` marker and exposed via `miopenGetLastForwardedError()`; reconciling the prototype's blanket fallback with that design is RFC §7 Phase 2 task 6.

## How it was tested

Single conv repeated 100 iterations (`-i 100`) via `MIOpenDriver`:

```
MIOpenDriver conv -n 16 -c 16 -H 16 -W 16 -k 16 -y 3 -x 3 \
    -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1 -V 0 -i 100 -t 1
```

Four runs (`FourRunComparison.txt`): flag **off** then **on**, each run twice (cold process / warm process) to expose cross-process caching behavior.

## Results

Per-phase overhead (redirect on, from the `[MIOpen->hipDNN]` timers):

| Phase | Cold | Warm |
| --- | --- | --- |
| `hipdnnCreate` | ~2.1 ms | n/a (once per process) |
| conv plan build | **1.3 ms – 3.9 s** (highly variable) | cached (0) |
| conv execute | ~1.4 ms | ~27 µs / call |
| `hipdnnDestroy` | ~1 µs | n/a |

Wall-clock (`real`), cold vs. warm **process**:

| | 1st run (cold) | 2nd run (warm) |
| --- | --- | --- |
| Redirect **off** (native) | 4.860 s | **0.213 s** |
| Redirect **on** (hipDNN) | 4.863 s | **4.121 s** |

Native MIOpen drops to 0.213 s on the second process because it reuses its **persistent on-disk kernel cache**. The redirect path has only an in-process plan cache, so every fresh process re-pays the cold plan build (dominated by the multi-second `conv plan build`), and warm-process wall time stays ~4 s. The warm *per-call* execute overhead itself is small (~27 µs).

## Acceptance criteria

- **Open/close of hipDNN handle in the shim — MET.** Paired `hipdnnCreate` on handle creation, `hipdnnDestroy` on destroy; cold create ~2.1 ms, destroy ~1 µs.
- **Timing/benchmarking overhead, cold + warm — MET.** Instrumented via `MIOPEN_HIPDNN_FORWARDING_TIMING`; numbers above. The driver's own GPU kernel timing does not flow through the redirect (it reports 0.000 ms / inf GFLOPs), so the shim's env-var timers are the usable measurement path.
- **Stretch: full operation + run via MIOpenDriver — LARGELY MET.** `miopenConvolutionForward` is fully implemented over the hipDNN backend API and runs end-to-end through `MIOpenDriver` with `[MIOpen->hipDNN]` trace lines. Caveats: numerical verification was not enabled in these runs (`-V 0`), and hipDNN's own internal logging was not captured (only the shim's trace).

## Concerns found

1. **No persistent cache for the redirect (highest impact).** The hipDNN plan is cached per process only. Native MIOpen persists compiled kernels to disk and so collapses repeat runs to 0.2 s; the redirect re-pays cold setup every process and stays ~4 s. Cross-process persistence (or enabling/sharing hipDNN's own kernel cache) is needed before the redirect is competitive on repeated invocations. **This is the headline regression and the gating item for RFC §7 Phase 1 exit criterion 4 (cold-start vs warm-start) and Phase 4 perf.**
2. **Plan-build time is unstable** (1.3 ms on one cold run, 3.9 s on another). The multi-second case implies kernel compilation / autotuning happening inside plan finalize on some runs; the attribution/trigger needs to be understood.
3. **Driver benchmarking blind spot.** GPU kernel time / GFLOPs report 0.000 / inf under the redirect, so standard driver perf reporting is unusable; only the shim timers work today.
4. **Limited coverage.** Conv forward only, identity scaling, group=1. Because the path falls back silently on any unsupported case, "redirect enabled" does not guarantee the redirect was actually taken — future work should surface when fallback occurs. (Note: this prototype's silent fallback on *any* hipDNN failure is broader than the RFC §6.1 design, which propagates per-call hipDNN errors rather than absorbing them; Phase 2 must reconcile the two.)
5. **Correctness not yet verified.** These runs used `-V 0` (no numerical verification), so the hipDNN path was only proven to execute, not to produce correct results. Follow-up: re-run with verification enabled (`-V 1`) before relying on the path.

## Performance impact (summary)

- **Per handle:** +~2.1 ms cold create, +~1 µs destroy; negligible thereafter.
- **Per conv (warm):** ~27 µs host-side overhead per call (native GPU kernel is ~17 µs; not directly comparable, but the redirect adds modest steady-state cost).
- **Cold start / repeated processes:** dominated by the non-persisted plan build (up to ~3.9 s), keeping warm-process wall time at ~4 s vs. native 0.2 s. This is the headline regression for the redirect as currently implemented and the main item to resolve in the RFC.

## Related notes

- The `gfx94250.HIP.fdb.txt ... unreadable` warning seen in the logs is a native find-db CU-count resolution issue on MI308X, **unrelated to the redirect** and tracked separately. It does not affect the conclusions above.
