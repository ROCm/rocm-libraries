# "Input data preparation" investigation (gfx950 per-config rerun 2)

`full_runs_gfx950/per_config_rerun_2/AGGREGATE_analysis.txt` reports **Input data preparation**
at 502,878.60 ms (19.6% of wall). This note explains why it is slow and which of its timers are
inflated or deflated by work that belongs elsewhere. Timing claims come from measurements; code
reading was only used to decide what to measure. Inferences are marked as such.

`cpu_data_init` (106.8 s) is covered separately in `cpu_data_init_investigation.md` and is not
repeated here.

| Timer | Count | Total (ms) | Mean (ms) |
| --- | --- | --- | --- |
| `gpu_input_reset` | 2,106,835 | 306,247.35 | 0.15 |
| ├─ `gpu_input_reset_copy_valid_to_gpu` | 719,180 | 177,839.34 | 0.25 |
| ├─ `gpu_input_reset_batched_inputs` | 1,473,329 | 95,916.03 | 0.07 |
| ├─ `gpu_input_reset_copy_inputs_d2d` | 1,473,329 | 26,309.71 | 0.02 |
| ├─ `gpu_input_reset_copy_swizzled` | 15,719 | 4,006.86 | 0.25 |
| └─ `gpu_input_reset_fast_reset_output` | 633,454 | 1,917.65 | 0.00 |
| `cpu_data_init` | 28,596 | 106,829.68 | 3.74 |
| `pre_solution_listener.DataInitialization` | 2,355,016 | 49,481.67 | 0.02 |
| └─ `pre_solution_mx_side_regenerate` | 2,298 | 44,069.34 | 19.18 |
| `gpu_input_preparation` | 28,641 | 39,596.08 | 1.38 |
| ├─ `gpu_input_prep_copy_valid_to_gpu` | 11,446 | 22,935.66 | 2.00 |
| └─ `gpu_input_prep_copy_swizzled` | 1,151 | 10,766.76 | 9.35 |
| `rotating_buffer_preparation` | 28,641 | 723.82 | 0.03 |

## Summary

1. **Per-solution re-upload of unchanged data (`copy_valid_to_gpu`, 177.8 s).** On the slow
   reset path, every solution re-uploads every input tensor from host memory. 99.45–99.92% of
   those uploads are byte-identical to the previous upload into the same buffer.
2. **`batched_inputs` (95.9 s) is inflated.** Its own work is 4–7 tiny (8–24 byte) uploads of batch
   pointer arrays per solution; 99.99% are byte-identical to the previous ones. Each costs
   ~12–15 µs of latency, and because they are the first blocking call after the D2D copies they
   also absorb those copies' GPU time and, on configs with `SyncsPerBenchmark: 0`, the previous
   solution's still-running warmup kernels.
3. **`copy_inputs_d2d` (26.3 s) is deflated.** `hipMemcpy` D2D returns in ~2 µs regardless of
   size; the copy itself is charged to whatever blocks next. Its real serialized cost is
   4.5–5.5× the timer.
4. **MX regeneration (`pre_solution_mx_side_regenerate`, 44.1 s) is redundant** and, through
   `m_inputEpoch`, triggers a redundant CPU reference recompute
   (`cpu_reference_gemm_per_solution_inputs_changed`, 151.7 s, booked under "Reference GEMM").

## Method

- **Binary.** The measured client, `/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client`
  (Release, `-O3`). Configs re-run with the cached split artifacts
  (`/tmp/tl_split_gfx950/artifacts`) exactly as `full_runs_gfx950/profiling/run_rerun.sh` does
  (`TimingInstrumentation=True`, `OMP_NUM_THREADS=25`, 25 pinned cores).
- **Environment.** The host was shared: during these runs cores 0–47 were ~99% busy with other
  users' work, so runs used idle NUMA-1 cores (64–88, 89–113) with NUMA-1 GPUs 4 and 5, two runs
  at a time. `perf` was unavailable (`perf_event_paranoid=4`).
- **`hipMemcpy` profiler.** An `LD_PRELOAD` shim (`memprof.cpp`) records per call site (return
  address, mapped to functions with `nm`; the client has no debug info) and kind: count, bytes and
  time inside `hipMemcpy`. Optional modes:
  - *hash*: hash each H2D source and count uploads byte-identical to the previous upload into the
    same device address;
  - *presync*: `hipDeviceSynchronize()` before each `hipMemcpy`/`hipMemset`, charging the wait to
    the **previous** GPU operation (memcpy call site, `hipExtModuleLaunchKernel`/`hipLaunchKernel`,
    memset, async copy). This serializes everything, so it gives each operation's full cost and
    shows where in-flight work is absorbed. An idle-GPU `hipDeviceSynchronize` costs 0.07 µs.
- **Shim overhead.** In passive mode the timers match production: `spmm_f16`
  `copy_valid_to_gpu` 21,126 ms vs 20,468 ms, `batched_inputs` 5,993 vs 5,622 ms,
  `copy_inputs_d2d` 1,366 vs 1,489 ms.
- **Microbenchmarks.** `copybench.cpp` (copy behavior) and `mxdeterm.cpp` (MX generator
  determinism, linked against the client's `libhipblaslt-mxdatagen.a`).
- Scratch sources, scripts and per-run outputs are in `/tmp/syncval_bench` (`memprof.cpp`,
  `memprof_sum.py`, `run_art.sh`, `copybench.cpp`, `mxdeterm.cpp`, `inprep/<config>_<mode>/`).

## Background: the two reset paths

Before each solution, `main.cpp` calls `DataInitialization::prepareGPUInputs`
(`DataInitialization.hpp`), timed as `gpu_input_reset`:

- **Fast path** (`fast_reset_output`, 633,454 calls, 1.9 s): GPU-generated data, bounds checking
  off, no problem-dependent data, no swizzle. Only output tensors are reset.
- **Slow path** (1,473,329 calls): taken when bounds checking is on, data is problem-dependent
  (`m_problemDependentData`: sparse, MX, Trig/Serial/Identity init modes, multiple bias types), or
  tensors are swizzled. Per solution it runs:

```text
copyValidToGPUBuffer     if problem-dependent: H2D every tensor, host valid → GPU valid   (copy_valid_to_gpu)
copySwizzledToGPUBuffer  if swizzled                                                       (copy_swizzled)
copyInputs               D2D every tensor, GPU valid → GPU current                         (copy_inputs_d2d)
initializeGPUBatchedInputs  H2D a batch-pointer array per A/B/C/D (+ more)                 (batched_inputs)
```

The host data only changes per problem (`cpu_data_init`), or when MX data is regenerated.

## Finding 1: per-solution re-upload of unchanged data (`copy_valid_to_gpu`, 177.8 s)

The biggest contributors are `spmm_f16_sb` (53.9 s), `xfp32` (27.7 s), `spmm_f16` (20.5 s),
`plr_zero` (15.2 s) and `spmm_bf16_sb` (11.3 s).

Hash mode, `copyInputBuffers` H2D (the `copy_valid_to_gpu` uploads):

| Config | Uploads | Byte-identical to previous upload | MB/call (min–max) | Upload time |
| --- | --- | --- | --- | --- |
| `spmm_f16` | 567,420 | 566,162 (99.78%) | 1.26 (0.001–3.94) | 21.3 s |
| `spmm_f16_sb` | 1,508,910 | 1,507,646 (99.92%) | 1.25 (0.001–3.94) | 55.7 s |
| `xfp32` | 462,000 | 459,436 (99.45%) | 2.22 (2.10–16.8) | 30.2 s |
| `subtile_mxfp4` | 167,071 | 163,921 (98.1%) | 0.59 (0–16.8) | 7.1 s |

- Nearly all of the bytes uploaded on every solution are already on the GPU. `spmm_f16` alone
  re-uploads ~715 GB at ~34 GB/s.
- The copies are sized by `p.maxElements` (the largest allocation in the run), not the current
  problem: in `xfp32`, every upload over 2,052 problems of varying size is at least 2.10 MB. The
  `cpu_data_init` investigation measured the same `maxElements` sizing directly.
- A pinned source would help only for small tensors (`copybench`, pageable vs pinned: 1 MB 41.8
  vs 26.7 µs; 4 MB 84.6 vs 81.7 µs; 16 MB and up ~56 GB/s both). Not uploading is the real fix.
- `DataInitialization` already tracks host rewrites with `m_inputEpoch` (used by
  `ReferenceValidator` to skip recomputing the CPU reference). Skipping the upload when the
  epoch is unchanged would remove ≥99% of this bucket, ~176 s (inferred from the identical-upload
  rates, not measured end-to-end).

The per-problem `gpu_input_prep_copy_valid_to_gpu` (22.9 s) is needed once per problem, but is
subject to the same `maxElements` sizing.

## Finding 2: `batched_inputs` is inflated (95.9 s)

`initializeGPUBatchedInputs` builds a pointer array per tensor on the host and uploads it with a
pageable `hipMemcpy`. In every profiled config each upload is 8–24 bytes (8 bytes per batch).

| Config | Uploads | Per solution | Byte-identical | Mean call |
| --- | --- | --- | --- | --- |
| `spmm_f16` | 449,568 | 7 | 449,511 (99.99%) | 13.3 µs |
| `spmm_f16_sb` | 1,190,378 | 7 | 1,190,321 (99.995%) | 12.4 µs |
| `xfp32` | 369,600 | 4 | 369,560 (99.99%) | 15.1 µs |
| `bufferLoadStore0` | 1,609,364 | 4 | not hashed | 14.9 µs |

The arrays only change when a problem or buffer address changes, yet they are rebuilt and
re-uploaded for every solution. The timer then holds three things:

1. **Latency of the tiny uploads.** An 8-byte pageable H2D costs 11.6 µs on an idle GPU
   (`copybench`). In presync mode each also leaves ~7–8 µs of work in flight after returning.
2. **The preceding D2D copies.** They return immediately (Finding 3), so the first blocking
   upload waits for them: in `copybench`, an 8-byte upload issued right after a 64 MB D2D takes
   28.5 µs instead of 11.6 µs.
3. **The previous solution's warmup kernels.** With `SyncsPerBenchmark: 0` (524 occurrences in
   the test YAMLs, including `bufferLoadStore0` and `spmm_f16`), warmups 1..N-1 are launched and
   nothing waits for them: `BenchmarkTimer::postWarmup` returns early unless
   `skip-slow-solution-ratio` is set. The next solution's reset absorbs them. Measured in presync
   mode on `bufferLoadStore0`: 2,345 ms of still-running-kernel time absorbed inside the reset,
   over 200,635 of 401,984 resets (11.7 µs each).

Passive-mode excess over the 11.6 µs floor: `bufferLoadStore0` 24.0 s measured vs 18.7 s floor
(5.3 s absorbed); `spmm_f16` 5.96 s vs 5.21 s (0.75 s absorbed). Caching the arrays per problem
(or re-uploading only when the epoch or a base address changes) would remove nearly all of the
bucket; the absorbed work would then surface in the next blocking call (inferred).

## Finding 3: `copy_inputs_d2d` is deflated (26.3 s)

`copybench`, GPU otherwise idle:

| D2D size | `hipMemcpy` returns after | Copy complete after |
| --- | --- | --- |
| 64 KB | 2.2 µs | 14.5 µs |
| 4 MB | 2.1 µs | 21.4 µs |
| 64 MB | 2.2 µs | 38.2 µs |
| 256 MB | 2.3 µs | 113.5 µs |

In the client the D2D calls average 2–3 µs. Presync mode gives their real serialized cost (time
in the call plus the wait charged to them):

| Config | `copy_inputs_d2d` timer (passive) | D2D call time + wait charged to D2D (presync) |
| --- | --- | --- |
| `spmm_f16` | 1,366 ms | 1,443 + 6,015 = 7,458 ms |
| `bufferLoadStore0` (NaN bounds check: `copyNaNInputBuffers` + `copyInputBuffers`) | 6,144 ms | 2,646 + 10,932 + 2,887 + 11,271 = 27,736 ms |

The remainder lands in `batched_inputs`, in the warmup kernel's start, and in the validators'
first blocking readback. `copyInputs` copies every tensor, including read-only inputs (A, B) that
kernels do not modify; only outputs (and the Synchronizer, see `SynchronizerValidator.md`) need
restoring between solutions. The split between necessary and avoidable D2D bytes was not
measured.

## Finding 4: redundant MX regeneration and the CPU reference recompute it triggers

On gfx950, `initializeMXData` caches MX data per side, keyed by the tensor descriptor plus a tag
saying `gpuInput.valid` holds a scale swizzled for the current solution's MI shape. The tag is
erased whenever a side is initialized without a swizzle, which includes the problem-level init.
So the first solution of each problem that needs the GFX950 swizzle misses the cache and
**regenerates** the side: one `generateMXInput` call for data and canonical scale into
`cpuInput.valid`, and a second one (with the swizzle layout) that rewrites the data buffer again
to produce the swizzled scale. It also increments `m_inputEpoch`.

- **The regenerated data is identical.** `generateMXInput` uses a fixed seed (`1713573849`,
  `mxDataGen.cpp`). Measured with `mxdeterm.cpp` (4096×4096, FP4 E2M1 and FP8 E4M3): two calls
  with identical arguments give byte-identical data and scales, and the data written by a
  `None`-layout call is byte-identical to that written by a `GFX950`-layout call. Each call costs
  77–160 ms at this size. The descriptor is unchanged at the point of regeneration, so the
  arguments are identical.
- **It happens once per problem.** In `subtile_mxfp4`: 1,370 regenerated sides = 685 problems × 2
  sides, and 685 `pre_solution_mx_copy_valid_to_gpu_after_regen` uploads. The hash run finds
  3,150 non-identical uploads in total, ~4.6 per problem. If the regenerated data differed from
  the problem-level data there would be at least 4 more per problem (A, B and both scales), so
  the uploads after regeneration are byte-identical too (inferred from the counts).
- **It triggers a second CPU reference.** `ReferenceValidator` recomputes the CPU reference when
  `inputEpoch(solution)` changes. Across the aggregate, in 35 of the 36 configs involved, the
  number of `cpu_reference_gemm_per_solution_inputs_changed` recomputes equals the number of
  after-regeneration uploads. In `subtile_mxfp4`, each of the 685 problems computes the reference
  twice (685 problem-level + 685 "inputs changed", 5,916 ms). Aggregate: 1,146 recomputes,
  151.7 s, booked under "Reference GEMM (CPU)".

| Config | Regenerate (ms) | Recompute (ms) | Count |
| --- | --- | --- | --- |
| `subtile_mxfp8` | 14,806 | 70,729 | 39 |
| `subtile_mxfp8_correctness_regression_mt256x320_plr0_wgm1` | 7,104 | 27,987 | 5 |
| `subtile_mxfp8_correctness_regression_mt320x256_plr1_wgm4` | 6,883 | 23,140 | 5 |
| `mxfp4_mxfp4_bf16_tn_act` | 1,241 | 13,744 | 78 |
| `subtile_mxfp4` | 5,908 | 4,126 | 685 |

Producing only the swizzled scale from the canonical one, without touching the data or the
epoch, would remove the 44.1 s here and the 151.7 s recompute (inferred).

## Inflated and deflated timers

| Timer | Effect | Why |
| --- | --- | --- |
| `gpu_input_reset_batched_inputs` | **inflated** | absorbs the preceding D2D copies and, with `SyncsPerBenchmark: 0`, the previous solution's warmup kernels |
| `gpu_input_reset_copy_inputs_d2d` | **deflated** | `hipMemcpy` D2D returns in ~2 µs; real cost 4.5–5.5× the timer, charged downstream |
| `gpu_input_reset` (whole) | **inflated** on `SyncsPerBenchmark: 0` configs | previous solution's kernel time (measured 2.3 s on `bufferLoadStore0`) |
| `gpu_input_reset_copy_valid_to_gpu` | accurate, but ≥99% redundant | re-uploads unchanged, allocation-sized data every solution |
| `pre_solution_mx_side_regenerate` | accurate, but redundant | regenerates byte-identical data; inflates "Reference GEMM" by 151.7 s via the epoch |

## Not measured / open items

- End-to-end effect of any fix (only per-call costs and redundancy were measured).
- How much of `copy_inputs_d2d` is necessary (outputs) vs avoidable (read-only inputs).
- Where the D2D and kernel time absorbed by `batched_inputs` would land once that bucket is
  removed; an explicit, separately timed sync after the warmups would make it visible.
- `gpu_input_prep_copy_swizzled` / `gpu_input_reset_copy_swizzled` (14.8 s combined) and the
  per-problem paths were not profiled beyond the counts above.
- Run-to-run variance on the shared host.
