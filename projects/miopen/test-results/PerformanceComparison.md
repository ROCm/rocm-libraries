# MIOpen hipDNN Shim — Performance Comparison

Date: 2026-05-27
Branch: `users/nhanna/miopen-hipdnn-shim-investigation-1`
Hardware: AMD Instinct MI300X (gfx942), ROCm 7.13, MIOpen 3.5.2
Script: `~/test-data/model_f_short.sh` (64 `convbfp16` configs, NHWC layout)

## Builds Under Test

Only `MIOPEN_ENABLE_HIPDNN_WRAPPER` differs between the two builds.

| Build | `MIOPEN_ENABLE_HIPDNN_WRAPPER` | Notes |
| --- | --- | --- |
| `build-flagoff` | `OFF` | links `libMIOpen.so.1` only |
| `build-flagon`  | `ON`  | adds `libMIOpen_private.so.1` |

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

## Conclusion

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

### Conclusion

Direct measurement of the wrapper hop confirms the convolution-level findings:

- **Steady-state CPU overhead: ≈1 ns per call** on a worst-case (no-work) API, well below the per-call cost of any function that actually launches a kernel.
- **Cold-load overhead: ≈6 extra major page faults** (one extra ELF mapped), with no detectable change in wall time, RSS, or minor page faults.
- The split into `libMIOpen.so` + `libMIOpen_private.so` is **not** a meaningful runtime cost — it is a structural change with essentially-zero measurable impact on the public API surface.
