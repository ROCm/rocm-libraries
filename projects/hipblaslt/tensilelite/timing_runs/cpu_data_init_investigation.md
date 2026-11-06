# `cpu_data_init` investigation (gfx950 per-config rerun 2)

`full_runs_gfx950/per_config_rerun_2/AGGREGATE_analysis.txt` reports `cpu_data_init` at
106,829.68 ms over 28,596 calls (mean 3.74 ms, 4.2% of wall). This note explains where that time
goes. Every number below is a timing measurement; reading the code was only used to decide what to
measure.

## Method

- **Raw logs.** Re-parsed the 188 per-config logs (`TIMING_CONTEXT` + `TIMING:cpu_data_init`).
  They reconcile exactly with the aggregate: 28,596 calls, 106,830 ms.
- **Instrumented client.** Rebuilt `tensilelite-client` from the exact sources the measured binary
  was built from (commit `4c6426cea6`, client and include trees), using the original
  `compile_commands.json` / `link.txt`, with extra `ScopedTimer`s in `prepareCPUInputs`,
  `initializeCPUInputs` (per tensor), `copyInputs` (per tensor) and around each
  `generateMXInput` call. It reproduces CI: `sk_bgemm_div` 8.4–15.0 s vs 15.1 s in CI, and the
  MX regression problems 0.4–2.4 s vs 0.3–2.2 s.
- **Microbenchmarks.** Built with the client's flags (`-O3 -DNDEBUG -mf16c -x hip -fopenmp=libomp`,
  no `-march`), using the real `TensorDescriptor`, `CoordNumbered`, `BFloat16` and
  `libhipblaslt-mxdatagen.a`. Their assembly was checked against the shipped binary with
  `objdump`.
- **Environment.** gfx950 host, 2× EPYC 9575F; runs pinned to cores 0–24 with
  `OMP_NUM_THREADS=25`, as in CI. `perf` was unavailable (`perf_event_paranoid=4`). The host was
  shared, with other users' processes using 30–90% of several cores in 0–24.

## What the timer covers

`ScopedTimer("cpu_data_init")` in `ReferenceValidator::preProblem` wraps
`DataInitialization::prepareCPUInputs(problem)`:

```text
cpu_data_init
└─ prepareCPUInputs
   ├─ initializeCPUInputs        (only if m_problemDependentData)
   │   ├─ initializeMXData → generateMXInput   (MX configs)
   │   └─ initArray per changed tensor          (initArrayTrig, Random C, Zero D, ...)
   ├─ copyInputs                 (hipMemcpy H2H of every tensor, maxElements each)
   ├─ initializeConstantInputs   (~0 ms)
   └─ ConvertToProblemInputs     (~0 ms)
```

The instrumented sub-timers add up to the parent:

| Run | initializeCPUInputs | copyInputs | cpu_data_init |
| --- | --- | --- | --- |
| 4096×4096×8192 TrigSin, first call | 246.5 ms | 65.3 ms | 311.8 ms |
| `sk_bgemm_div` | 8524.8 ms | 1213.2 ms | 9738.0 ms |
| MX regression, first problem | 570.0 ms | 574.0 ms | 1144.0 ms |

Not part of `cpu_data_init`: `data_init_setup` (one-time Random fills for configs whose data is
not problem-dependent), `cpu_reference_gemm`, and `pre_solution_mx_side_regenerate` (44.1 s, which
calls the same `generateMXInput` per solution but was not measured here).

## Where the 106.8 s goes

The distribution is heavy-tailed: the top 10 calls are 28.8% of the time, the top 200 are 59.6%,
and the top 1000 are 81.3%.

| Bucket (by A/B/C init mode and type) | Time | Share | Calls |
| --- | --- | --- | --- |
| MX configs (8/6/4-bit types, mxDataGenerator) | 44.6 s | 42% | 8,961 |
| Trig / Serial / Identity init modes | 43.1 s | 40% | 6,674 |
| Sparse configs | 12.5 s | 12% | 2,688 |
| Everything else (copy only) | 6.6 s | 6% | 10,273 |

Across all buckets, the first call in each client process (820 calls) is 54.0 s (50.5%). The
remaining 27,776 calls average 1.9 ms.

Largest single contributors: `streamk/sk_bgemm_div` 15.1 s (1 call),
`gfx950/subtile_mxfp8` 10.3 s (39 calls), `gfx950/subtile_mxfp4` 8.8 s (685 calls), and the two
`subtile_mxfp8_correctness_regression_*` configs 5.3 s and 5.2 s (5 calls each).

## Findings

### 1. MX data generation: about half the work is discarded

For MX configs, `initializeCPUInputs → initializeMXData` calls `generateMXInput`
(`clients/common/src/mxDataGen.cpp`) once per MX side. At problem level there is no current
solution, so no swizzled scale is generated (`swizzle=0` in every call observed).

Example: the 516M-element FP8 E4M3 A side (7168×71936) of the MX regression config. In-client,
`generateMXInput` took 2094 ms. A harness calling the same library reproduces 2130–2251 ms and
splits it:

| Step | Time (ms) | Share |
| --- | --- | --- |
| `generate()` (resize + RNG + FP8 convert) | 903–935 | ~42% |
| `getDataBytes()` copy | 128–132 | ~6% |
| `memcpy` into the client buffer | 18–19 | ~1% |
| **`getReferenceFloat()`, result discarded** | 938–1006 | **~45%** |
| **free of that 2 GB vector** | 102–116 | **~5%** |

- `generateMXInput` returns a full `std::vector<float>` reference (4 bytes per element). The
  TensileLite client ignores the return value, so that work is wasted.
- `generate()` includes a single-threaded `m_dataBytes.resize(n)` zero-fill: about 90 ms for
  516 MB.
- On 1 thread the call costs about 40 ns per element. It reaches only 7–11× speedup on 25 threads
  (20,935 ms → 2,094 ms for 516M elements).

### 2. Trig init: an expensive per-element loop

`initArrayTrig<T>(array, tensor)` runs for Trig init modes, which are problem-dependent and so
re-run whenever a tensor's descriptor changes. In the shipped binary's assembly, every element:

- allocates and frees a `std::vector<size_t>` (`operator new`, `memset`, `operator delete`)
- executes 2–4 64-bit `div` instructions (`CoordNumbered`) and calls `TensorDescriptor::index`
  out of line
- converts the index to `int`, then calls double-precision libm `sin`/`cos` (`sin@plt`), with no
  vectorization

Microbenchmark, 1 thread, 32M BF16 elements:

| Variant | ns/element |
| --- | --- |
| The client's loop | 22.1 |
| Without the vector allocation | 14.9 |
| Without the `sin` call | 12.1 |
| Just `sin` + store | 7.0 |
| `sinf` + store | 3.7 |

- **glibc `sin` slow path.** The argument is the element index. Single-thread cost is 6.9 ns
  below about 1.0e8 and 23.6–25 ns at 1.06e8 and above, glibc's large-argument reduction path.
  Large tensors pay the slow rate for most elements.
- **All changed tensors are re-initialized.** When `m_problemDependentData` is set,
  `initializeCPUInputs` re-runs `initArray` for every tensor whose descriptor changed, not only
  the Trig ones.
- **Worst case: `sk_bgemm_div`.** One 8192×57344×28672 BF16 problem with TrigSin A, TrigCos B and
  TrigSin C. It initializes about 2.35G trig elements and copies 5.6 GB, to validate
  `NumElementsToValidate: 128`; the CPU reference takes 13 ms. Instrumented breakdown of one run:
  A 843 ms, B 4,105 ms, C 3,551 ms, D 23 ms, `copyInputs` 1,213 ms.

Steady-state behaviour on a 4096×4096×8192 BF16→F32 problem (TrigSin A/B, 25 threads, 3 runs):

| Call | Trig init A + B | Init C + D | `copyInputs` | Total |
| --- | --- | --- | --- | --- |
| 1st (fresh process) | 160–263 ms | 15–22 ms | 65–90 ms | 273–352 ms |
| Same problem again | 0 (skipped) | 0 | 9.5–9.7 ms | 9.5–9.8 ms |
| K changes (4096) | 54–67 ms | 0 | 9.5–9.9 ms | 64–77 ms |
| K changes back (8192) | 106–134 ms | 0 | 9.6–9.8 ms | 116–143 ms |
| 256×256×256 | 0.5–0.9 ms | — | 9.6–10.0 ms | 10.2–10.6 ms |

### 3. `copyInputs` copies the largest size seen, every call

- `copyInputBuffers` copies `p.maxElements` (the run's largest allocation), not the current
  problem's size. In the table above, the 256×256×256 problem pays the same ~9.6 ms as the large
  one; the instrumented log shows A copying 33,554,432 elements for a problem whose A has 65,536.
- The cheap branch in `prepareCPUInputs` that would only reset the output never runs:
  `m_cpuInit` is initialized to `false` and only ever assigned `false`.
- A process's first copy is 4–9× slower (65–90 ms vs 9.6 ms) because the destination buffers are
  touched for the first time.

### 4. Fixed cost on each process's first call

- In the sparse `spmm_i8` run, the first call spent 53.3 ms copying a 1,024-element buffer and
  18.0 ms initializing 1,024 elements. Every later call took 0.6–3 ms.
- Standalone, the first `hipMemcpy(H2H)` after HIP init costs about 18 ms even for 1 KB (later
  calls ~0 ms), and the first OpenMP parallel region about 3 ms (0.2 ms later). The larger
  in-client values were not explained further without a profiler.
- Across CI, the 752 first calls on small problems (<4M elements) total 17.6 s, median 20.6 ms.
  This is why sparse configs show 50–70 ms for problems as small as 256×256×128.

### 5. Thread scaling breaks down on a shared host

The init loops use OpenMP's default static schedule, so one core shared with another process holds
up the whole loop.

- Pure `sin` scales linearly to 8 threads (462 → 60 ms), then flattens: 39.8 ms at 16 threads,
  40.2 ms at 25.
- With `schedule(runtime)` on the same cores, 25 threads take 58 ms with `OMP_SCHEDULE=static`
  and 21 ms with `dynamic,65536`.
- Identical `sk_bgemm_div` runs took 9.7, 15.0 and 8.4 s; B's init alone ranged from 3,665 to
  11,690 ms.

## Caveats

- HEAD's `DataInitialization` differs from the measured binary by about 300 lines (MX data
  generator on/off). The trig loop, `copyInputs`/`maxElements`, `m_cpuInit` and
  `getReferenceFloat` paths are the same shape in both, but absolute numbers on HEAD may differ.
- The ~50% MX waste figure comes from the standalone harness; `getReferenceFloat` was not timed
  inside the client.
- Sparse configs were only checked for first-call overhead; the pruning code was not profiled.
- Machine contention adds large run-to-run variance; ranges above are across repeated runs.

## Possible follow-ups (not implemented)

- Stop building the discarded MX reference floats when the caller ignores them.
- Make trig init cheaper: no per-element allocation, flat index for default strides, `sinf` or a
  bounded argument.
- Copy only the current problem's size in `copyInputs`, and fix or remove the dead `m_cpuInit`
  path.
- Use a dynamic schedule (or larger chunks) for the init loops.
