# What's inside the bench's timed window that isn't inside okl_run's

Read-only investigation. All file:line citations verified by direct read at
time of writing. Source tree:
`/home/alvasile/rocm-libraries/projects/hipblaslt`. Bench binary the user
ran is `/opt/rocm-6.4.3/bin/hipblaslt-bench`, which was built from a
similar (but possibly older) source than what is read here; the structure
of the timed window has been stable across recent revisions but the
specific line numbers below are from the in-repo source.

CLI under investigation:
`hipblaslt-bench --algo_method index --solution_index N --iters 500
--cold_iters 500 --use_gpu_timer` with no `--use_ext`, no
`--grouped_gemm`, no `--flush`, no `--verify`, no
`HIPBLASLT_TUNING_FILE` env, no `HIPBLASLT_BENCH_PERF` env. With these
defaults, the code path taken inside `testing_matmul.hpp` is:

| Flag | Default in `client.cpp` | Source |
| --- | --- | --- |
| `arg.use_ext` | `false` | `clients/bench/src/client.cpp:1021-1024` (`api_method=c` → `use_ext=false`) |
| `arg.grouped_gemm` | `0` | `clients/bench/src/client.cpp:735-754` (only set if `--grouped_gemm`) |
| `arg.batch_mode` | `0` (STRIDED) | `clients/bench/src/client.cpp:457-458` |
| `arg.flush` | `false` | `clients/bench/src/client.cpp:616-617` (`tuningEnv ? true : false`; `tuningEnv` env is unset) |
| `arg.skip_slow_solution_ratio` | `0.0` | `clients/bench/src/client.cpp:602-603` |
| `arg.use_gpu_timer` | `true` (passed on CLI) | `clients/bench/src/client.cpp:598-599` |
| `arg.rotating` | `0` | `clients/bench/src/client.cpp:595` (only nonzero when `HIPBLASLT_TUNING_FILE`) |
| `arg.unit_check / norm_check / allclose_check` | `0` | `clients/bench/src/client.cpp:70-71, 1015-1019` (norm/allclose only set under `--verify`) |
| `arg.timing` | `1` | `clients/bench/src/client.cpp:73-74` |
| `arg.c_equal_d` | `false` | `clients/bench/src/client.cpp:569-570` |

So the active loop is the non-grouped, non-extension, STRIDED branch in
`testing_matmul.hpp` (the `else` arm starting at line 5042; specifically
the hot loop at lines 5104-5136).

Because `arg.rotating == 0`, `block_count` collapses to `1`
(`testing_matmul.hpp:1944`: `max(1, min(max_iters, ceil((float)0 / X))) == 1`),
so every iteration of the hot loop calls `hipblasLtMatmul` with **identical
A/B/C/D device pointers and the same kernel `algo` index**. There is no
buffer rotation in this CLI.

---

## 1. Timing window

The window opens at `testing_matmul.hpp:5102`:

```
perf_monitor->start();
pre_gpu_time(arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
```

`pre_gpu_time` (`testing_matmul.hpp:409-418`) records
`hipEventRecord(event_gpu_time_start, stream)` when `use_gpu_timer`
is true (our case).

The window closes at `testing_matmul.hpp:5138-5143`:

```
post_gpu_time(arg.use_gpu_timer,
              event_gpu_time_start,
              event_gpu_time_end,
              gpu_time_used,
              stream);
perf_monitor->stop();
```

`post_gpu_time` (`testing_matmul.hpp:419-438`) records
`hipEventRecord(event_gpu_time_end, stream)`, then
`hipEventSynchronize(event_gpu_time_end)`, then
`hipEventElapsedTime(...)`. The reported `gpu_time_used` is the
hipEvent-measured stream-time between the two recorded events.

Equivalent locations exist for every other branch (use_ext at 4949/5138,
POINTER_ARRAY at 5012/5138, grouped 5169-5263). The path we care about
opens at 5102 and closes at 5138.

EfficiencyMonitor (`clients/common/src/efficiency_monitor.cpp:152-169`):
both `start()` and `stop()` early-return when `enabled() == false`. They
are enabled only when one of `HIPBLASLT_BENCH_FREQ`,
`HIPBLASLT_BENCH_PERF`, or `HIPBLASLT_BENCH_FREQ_ALL` is set
(`clients/common/src/efficiency_monitor.cpp:105-121`). With none set in
the user's invocation, `perf_monitor->start()/stop()` are no-ops.

---

## 2. Everything between `pre_gpu_time` and `post_gpu_time`

The body of the hot window is the loop at `testing_matmul.hpp:5104-5136`:

```
for(int i = 0; i < number_hot_calls; i++)
{
    auto ptr_matmul = matmul[i % block_count][0];          // block_count == 1
    auto ptr_alpha  = arg.scaleAlpha_vector ? ... : alpha_in[0];
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmul(handle,
                        ptr_matmul,
                        ptr_alpha,
                        dA[0].as<char>() + (i % block_count) * size_dA[0] * realDataTypeSize(TiA),
                        ...
                        &heuristicResult[sol].algo,
                        *dWorkspace,
                        workspace_size,
                        stream),
        HIPBLAS_STATUS_SUCCESS);
    if(arg.flush)
        hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
}
```

Operations inside the timed window, in execution order per iteration:

| # | Operation | Where it lands | File:line | Gated? |
| --- | --- | --- | --- | --- |
| 1 | Map-lookup `matmul[i%1][0]` and pointer arithmetic | CPU only | `testing_matmul.hpp:5106-5110` | Always |
| 2 | `EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(...))` macro: forwards to `hipblasLtMatmul` and asserts success | CPU only (the assertion); the call itself enqueues GPU work | `testing_matmul.hpp:5111-5133` | Always |
| 2a | Inside `hipblasLtMatmul`: validation, marker, forward to `rocblaslt_matmul` | CPU only | `library/src/amd_detail/hipblaslt.cpp:476-515` | Always |
| 2b | `rocblaslt_matmul`: nullptr/type checks, optional `log_api`/`log_trace` (gated on `get_logger_layer_mode()`), forward to `rocblaslt_matmul_impl` | CPU only | `library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp:683-810` | Logging gated; default off |
| 2c | `rocblaslt_matmul_impl`: `rocblaslt_matmul_valid_args` (CPU validation), build `RocblasltContractionProblem`, call `runContractionProblem` | CPU only | `library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp:43-226` | Always |
| 2d | `runContractionProblem`: `get_library_and_adapter` (cached, no IO after first call), `updateTensileProblem` (per-call, rebuilds `ContractionProblemGemm` indices/sizes), `solution->isFallbackForHW`, `data->problem.setParams().setWGMXCC(...)` | CPU only | `library/src/amd_detail/rocblaslt/src/tensile_host.cpp:2770-2917` | Always |
| 2e | `solution->solve(...)`: builds a fresh `std::vector<KernelInvocation>` of size 1 (see §3), allocates a fresh `KernelArguments` buffer (~1 KiB), runs `calculateGrid`, `calculateAutoGSU`, `calculateAutoWGM`, `calculateAutoStaggerU`, then appends ~30 typed values to the kernarg buffer | CPU only | `tensilelite/src/ContractionSolution.cpp:2733-2883` (top-level) and `:1424-1551` (`generateSingleCall`) | Always |
| 2f | `adapter->launchKernels(kernels, stream, nullptr, nullptr, isPreloaded)` → iterates kernels (1 element for our solution) → `launchKernel` | mixed: CPU (lookup) + GPU stream (the launch) | `tensilelite/src/hip/HipSolutionAdapter.cpp:500-530` | Always |
| 2g | `launchKernel`: optional `FindCodeObject` (cached hit after first call), `getKernel` (mutex-protected `unordered_map` lookup), build `hipLaunchParams` array on stack, **`hipExtModuleLaunchKernel`** | CPU (lookups) + GPU stream (the kernel itself) | `tensilelite/src/hip/HipSolutionAdapter.cpp:405-481` | Always |
| 3 | `if(arg.flush) hipLaunchKernelGGL(flush_icache, ...)` | GPU stream (if flag) | `testing_matmul.hpp:5134-5135` | Off by default (`arg.flush=false`) |

The single GPU-side enqueue per iteration is the `hipExtModuleLaunchKernel`
at `tensilelite/src/hip/HipSolutionAdapter.cpp:455`. The only other call
that touches the stream is the optional `flush_icache` at step 3, which is
**off** for the user's CLI.

Critically: between the loop body and the end-of-window
`hipEventRecord(event_gpu_time_end, stream)`, nothing is added to the
stream. The CPU then blocks at `hipEventSynchronize`
(`testing_matmul.hpp:428`) and reads the elapsed time. So the GPU-event
timer measures, on-stream, exactly: `N x [the GEMM kernel launch]` plus
any inter-launch gap the GPU sees while the host issues the next launch.

---

## 3. What does one `hipblasLtMatmul` enqueue on the GPU?

`runContractionProblem` calls `solution->solve(problem, inputs, hardware)`
which returns a `std::vector<KernelInvocation>`
(`tensilelite/src/ContractionSolution.cpp:2733-2883`). The vector is
built in this order:

1. **Beta-only call** (line 2810-2816). Pushed only if
   `gsu > 1 && sizeMapping.globalAccumulation not in {2,3}`. For our
   kernel (`GSU1`), `gsu == 1`, so **not pushed**.
2. **StreamK setup** (line 2818-2845). Only runs when
   `sizeMapping.streamK > 0`. Kernel name has `SK0` →
   `streamK == 0`, so **skipped**.
3. **Main GEMM `generateSingleCall`** (line 2851-2853). Always pushed.
   This is the single-kernel-launch entry that contains the whole
   matmul.
4. **Output conversion call** (line 2855-2862). Pushed only if
   `(gsu > 1 && globalAccumulation && globalAccumulation != 3) ||
   parallel StreamK reduction`. With `gsu == 1` and `streamK == 0`,
   **not pushed**.
5. **Bias reduction call** (line 2865-2880). Pushed only if
   `problemType.useBias && useGradient && biasSrc == TENSOR::D`. The
   kernel name has no bias tag and the test packages don't request bias
   gradient, so **not pushed**.

Net: **`solve()` returns exactly 1 `KernelInvocation`** for this
solution. `adapter->launchKernels` therefore enqueues a single
`hipExtModuleLaunchKernel` per `hipblasLtMatmul` call
(`tensilelite/src/hip/HipSolutionAdapter.cpp:500-530`, then `:455`).

Other on-stream auxiliaries that do NOT fire:
- **Workspace memset / memcpy**. `tensile_host.cpp` and `rocblaslt_mat.cpp`
  contain zero `hipMemset*` calls on the matmul path
  (`grep hipMemset* library/src/amd_detail/rocblaslt/src/` returns only
  `check_numerics_matrix.hpp:318`, which is gated on `--check_numerics`).
  The grouped-gemm path has `hipMemcpyAsync` for kernarg copy
  (`tensilelite/src/ContractionSolution.cpp:3015-3016`), but we are not
  on that path (`do_grouped_gemm == false`).
- **MBSK Synchronizer init**. Used only when `globalAccumulation == 3`.
  Kernel name has `GSUC0` and `GSU1`, neither triggers MBSK. Confirmed
  by `generateSingleCall:1529` — the synchronizer arg is appended only
  when `globalAccumulation == 3 || adaptiveGemmGSUA == 1`. Our kernel
  is neither.
- **Heuristic search**. `getBestSolutions` only runs when `algo == nullptr`
  (`tensile_host.cpp:2779`). With `--algo_method index`, the algo is
  always non-null, so no per-call heuristic search.

The kernel arg buffer that `generateSingleCall` builds carries values
that **vary per call** because they're recomputed each call:

- `calculateAutoGSU` (`ContractionSolution.cpp:2737, 1444`, then again
  inside `generateSingleCall` at 1444 and at 1481, and a third time
  inside `solve()` at 2808).
- `calculateAutoWGM` (`ContractionSolution.cpp:1481-1482, 993-1100`) —
  for our solution (`workGroupMapping == 0`, `workGroupMappingXCC == -1`,
  `streamK == 0`), the "Default WGM" branch at line 1053-1074 runs and
  returns `defaultWGM = ceil(sqrt(N_CU/NUM_XCD))`,
  `defaultWGMXCC = NUM_XCD`. On MI300X (8 XCDs, 304 CUs total) that's
  `WGM = ceil(sqrt(38)) = 7` and `WGMXCC = 8`.
- `calculateAutoStaggerU` — same shape, sized to grid + WGM.

These get appended into the kernel arg buffer at
`ContractionSolution.cpp:1495-1524` via `kernelArgs<T_Debug, false>(...)`.
The kernel reads them at runtime to decide tile scheduling order /
XCC chunking / stagger pattern. **The kernel binary is the same; the
work-issue pattern at runtime depends on these arg values.**

---

## 4. The 474 µs gap at 8192³ — best-fit explanation

Static accounting of on-stream extras between bench and okl_run for the
GSU1/SK0/single-buffer case described above:

| Candidate | On-stream? | Fires for this solution? | Expected magnitude |
| --- | --- | --- | --- |
| beta-only kernel | yes | no (gsu=1) | n/a |
| output-conversion kernel | yes | no (gsu=1) | n/a |
| bias-reduction kernel | yes | no (no bias) | n/a |
| MBSK Synchronizer init/clear | yes | no (no MBSK) | n/a |
| hipMemset on workspace | yes | no (no code path) | A 64 MiB clear at MI300X's ~3.0 TB/s peak HBM BW would be ~21 µs; well under 474 µs but doesn't fire anyway |
| icache flush kernel | yes | no (`--flush` off) | n/a |
| EfficiencyMonitor sampling | no (CPU thread) | no (no env var) | n/a |
| Extra `hipblasLtMatmul` per iter beyond the 1 GEMM | yes | no | n/a |

**No additional GPU kernel fires per `hipblasLtMatmul` for this
solution.** A workspace hipMemset would only account for ~21 µs at
8192³ even if it did fire, well short of the 474 µs gap. The gap is
also clearly **not constant per call** (it grows from ~4 µs at small
shapes to ~474 µs at 8192³), which rules out per-call CPU-driven
overhead inflation.

The most plausible accounting of the gap, given that everything in §2
and §3 shows the bench issues exactly one kernel launch per iteration
just like okl_run:

**The kernel arg buffer that hipblaslt's `generateSingleCall` builds at
runtime contains *different values* for `numWG`, `WGM`, `WGMXCC`,
`StaggerU`, and the staggerU mapping than what okl_run captured into
its `slot` list.** The kernel binary is identical, but the runtime
arguments steer the kernel's tile-scheduling, XCC-chunking, and
stagger-U pattern. Different values cause different tile-to-CU mapping,
different work-stealing behavior, and different L2 / HBM access
patterns, which can plausibly slow the *same* kernel by tens of percent
on large GEMMs where HBM/L2 bandwidth utilization dominates.

The 40% slowdown is in the right ballpark for "wrong tile mapping on a
multi-XCD chip." On MI300X (8 XCDs, 304 CUs) the tile-to-XCC mapping
controlled by `WGMXCC`/`WGMXCCCHUNK` and the wave-stagger controlled by
`StaggerU` make a meaningful difference to L2 hit-rate when each XCD's
8 MB L2 has to feed multiple compute tiles. Mismatched mapping easily
turns L2-hit reads into HBM reads.

Two facts support this hypothesis directly over alternatives:
- The gap scales with M×N×K (not constant per call), consistent with a
  systematically-slower kernel rather than constant per-call overhead.
- The bench's GPU-event timer (`hipEventElapsedTime`) and its CPU wall
  clock agree closely (per the user's observation), which means the
  gap is *visible to GPU events*, i.e. it really is more work on the
  stream — not host-side marshaling.

Secondary contributor candidates I cannot rule out from code alone:
- Wave-occupancy difference if okl_run's captured `numWG` happens to
  yield a slightly better wave-to-CU mapping than the bench's runtime
  recomputation. This would manifest the same way.
- L2/instruction-cache state differences from how okl_run reuses the
  same kernarg pointer (single buffer in a `std::vector<uint8_t>` whose
  storage doesn't move) vs. bench rebuilding a fresh `KernelArguments`
  per call, so the kernarg copy the HIP driver stages may end up at a
  different physical location each iteration. This is speculative; HIP
  generally amortizes kernarg copies and a 1 KiB copy is cheap.

---

## 5. Confirmation plan

Confirm in this order. The first three are zero-build; the fourth needs
a one-line code change.

1. **rocprof per-iter kernel count.** Run the bench under
   `rocprof --hsa-trace --stats hipblaslt-bench ...` with a small iter
   count (e.g. 20 hot) and inspect the per-kernel histogram. For the
   GSU1/SK0 case, expect **exactly one kernel** per iter (the
   `Cijk_Alik_Bljk_BBS_BH_...` GEMM). If anything else appears, the
   "no auxiliary kernels" claim in §3 is wrong for this build and
   that kernel is the real cost source.
2. **`TENSILE_DB=0x40`.** This is the `printWinningKernelName` debug
   bit (`Debug::Instance().printWinningKernelName()`,
   `tensilelite/src/ContractionSolution.cpp:2738-2740`). With it set,
   each `solve()` call prints "Running kernel: <name>". `grep -c
   "Running kernel"` of the bench's output divided by the iter count
   should equal the number of distinct kernels solve() generated per
   call. Expected: 1.
3. **Print the kernel arg buffer.** Set
   `TENSILE_DEBUG_PRINT_KERNEL_ARGUMENTS=1` (i.e.
   `Debug::Instance().printKernelArguments()` true at
   `ContractionSolution.cpp:2760`) for one iteration of the bench
   under our CLI and dump it. Compare every field against the values
   in the okl_run config (`slot = name=<...> value=<...>`). If `numWG`,
   `WGM`, `WGMXCC`, `WGMXCCCHUNK`, or the staggerU group differ between
   the two, that's the proximate cause — and the okl_run capture was
   correct, the bench-runtime recomputation is what's hurting.
4. **One-line patch experiment.** In a local debug build, replace
   `solve()` in `runContractionProblem`
   (`tensile_host.cpp:2919`) with a cached vector built once and
   reused. If bench timing then converges to okl_run, the per-call
   arg recomputation (specifically the `calculateAutoWGM` /
   `calculateAutoStaggerU` path) is doing something different on a
   subsequent call than on the first. Less invasive: cache only the
   kernarg bytes and skip the rebuild.

Recommended first step: **#3 (print kernel args and diff against the
okl_run slot list)**. It's the most diagnostic — if WGM/WGMXCC/StaggerU
match, hypothesis is wrong and we look elsewhere; if they differ,
we've found it and can move on to fixing whichever recomputation is
inconsistent.

---

## 6. What's NOT the cause

- **icache flush kernel.** Off by default
  (`clients/bench/src/client.cpp:616-617`: `arg.flush = tuningEnv ?
  true : false`; `HIPBLASLT_TUNING_FILE` env is unset in our run). The
  flush code in `testing_matmul.hpp:5134-5135` is dead under the user's
  CLI.
- **EfficiencyMonitor / amd-smi sampling thread.** Gated on
  `HIPBLASLT_BENCH_PERF` / `HIPBLASLT_BENCH_FREQ` /
  `HIPBLASLT_BENCH_FREQ_ALL`
  (`clients/common/src/efficiency_monitor.cpp:105-121`). Without these,
  `start()` and `stop()` return immediately at lines 154 and 163.
- **Buffer rotation between iters.** `arg.rotating == 0` →
  `block_count == 1` (`testing_matmul.hpp:1944`). All iterations use
  the same A/B/C/D pointers, identical to okl_run.
- **Multiple kernel launches per `hipblasLtMatmul`.** For
  `GSU1`/`SK0`/single-buffer with no bias gradient, `solve()` returns
  exactly one `KernelInvocation`. See §3.
- **Workspace clear.** `grep -r hipMemset*
  library/src/amd_detail/rocblaslt/src/` returns one hit
  (`check_numerics_matrix.hpp:318`), gated on `--check_numerics`.
  Tensile's runtime (`tensilelite/src/`) also has no
  `hipMemset` calls on the matmul path. The 128 MiB workspace is
  allocated once and reused untouched between iterations.
- **`hipDeviceSynchronize` vs `hipEventSynchronize` semantics.** Both
  the bench and okl_run measure end-to-end on-stream time. The bench's
  `hipEventElapsedTime` measures GPU-side stream-time between the two
  recorded events, and the user reports it agrees closely with the
  bench's CPU wall-clock measurement. So extra cost is on-stream.
- **`hipblasLtMatmul`'s C-API marshaling.** This is CPU work
  (`hipblaslt.cpp:476-515` and downstream `rocblaslt_matmul_impl`).
  Per-call CPU work would manifest as a roughly constant µs/iter gap
  irrespective of M/N/K; the user's data shows the gap **grows** with
  problem size from ~4 µs (small) to ~474 µs (large), so this is at
  most a small constant contributor.
- **`getKernel` mutex contention.**
  `tensilelite/src/hip/HipSolutionAdapter.cpp:291-323` takes a unique
  mutex on `m_access` to do an `unordered_map<string, hipFunction_t>`
  lookup. With a single calling thread and a cache hit after iteration
  1, this is sub-microsecond per call. Constant per call, not
  size-scaling.
- **Stream choice (custom vs NULL).** okl_run uses `stream=nullptr`
  (`okl_run.cpp:498`), bench uses a `hipStreamCreate` stream
  (`testing_matmul.hpp:1664`). Single-stream serialization is the same
  in both cases; the choice does not affect kernel execution time.
- **`updateTensileProblem` rebuilding the problem.** It runs every
  call (`tensile_host.cpp:2788`) but is pure CPU work setting struct
  fields; no GPU work.

---

## Open uncertainties

- I have not verified that the bench's actual runtime-computed
  `WGM`/`WGMXCC`/`StaggerU` values differ from okl_run's captured
  values. Step §5.3 (kernel-arg dump) is the experiment that tells us
  whether the hypothesis is true. Without it, the diagnosis is
  inferred-by-elimination rather than directly observed.
- If the captured okl_run slot list happens to match what
  `calculateAutoWGM` produces at runtime, the hypothesis is wrong and
  the next place to look is the HIP driver itself (e.g. kernarg
  staging path, doorbell pacing). That would require a profiler trace
  rather than source reading.
- `hipblaslt-bench` at `/opt/rocm-6.4.3/bin/` was built from
  `rocm-6.4.3`; the source here is a later checkout. The
  branches/flags described above have been stable, but in principle
  the older bench could carry an extra step that the current source
  has removed. If §5.1 (rocprof) shows more than one kernel per iter,
  that's where the discrepancy lives. The 6.4.3 source can be
  inspected at `/opt/rocm-6.4.3/share/` (if shipped) or via a git
  checkout at that tag.

---

## 6. Second-pass investigation

The first-pass hypothesis (per-call recomputation of `numWG`/`WGM`/
`WGMXCC`/`StaggerU` producing a different work-issue pattern) was
refuted by an `AMD_LOG_LEVEL=3` dump of the actual kernarg bytes the
bench delivers to `hipExtModuleLaunchKernel`. The kernarg matches the
okl_run package's captured slot list byte-for-byte for `large_square`
(see `__init__` of this addendum). So both runners launch the same
binary, with the same 104-byte kernarg, on the same problem. The
~474 µs/iter gap at 8192³ persists. This section is the redo.

### 6.1 Refutation of first-pass hypothesis

User-supplied byte-equal kernarg evidence (sol 45755, large_square):

```
Field          Bench live dump   Package captured
Gemm_info      0x00000001        0x1
kernel_info0   0x01080001        0x01080001
kernel_info1   0x4c010020        0x4c010020
numWG          0x000004a0        0x4a0
sizes, strides, pointers, alpha, beta: all 4-byte equal.
```

`numWG = 0x4a0 = 1184` agrees. `kernel_info1 = 0x4c010020` decodes the
same `WGM=...`, `WGMXCC=...`, `staggerU` packing on both sides.

Conclusion: §4 hypothesis is wrong. The kernel binary and arguments are
identical between runners. The gap is not in the kernel input.

### 6.2 The actual bench hot-loop path (verified)

With `--algo_method index --solution_index N`, no `--use_ext`, no
`--grouped_gemm`, the hot loop is in `testing_matmul.hpp:5104-5136` and
calls `hipblasLtMatmul` per iter. Re-verified by direct read.
`pre_gpu_time` opens the window at 5102; `post_gpu_time` closes at
5138-5142. The body is exactly the `hipblasLtMatmul` call plus an
`if(arg.flush)` branch that is dead under our CLI (`arg.flush=false`).

Reproduced baseline numbers on this box:
- `hipblaslt-bench ... 8192³`: 1636.26 µs/iter, 671966 GFLOPS.
- `okl_run packages/large_square/kernel.conf`: 1164.524 µs/iter,
  944172.8 GFLOPS. Delta = +471.7 µs/iter, matches the package's
  +473.59 µs.

### 6.3 Stream identity (verified by direct read)

- `okl_run`: `stream=nullptr` passed to `hipExtModuleLaunchKernel`
  (`Testing/okl_run.cpp:498`). This is the HSA null/legacy default
  stream.
- `bench`: `hipStreamCreate(&stream)` at
  `clients/common/include/testing_matmul.hpp:1664`. No flag argument →
  `hipStreamDefault = 0`, i.e. a **blocking user stream** (synchronizes
  with the null stream by default). The same `stream` is passed
  through `hipblasLtMatmul(..., stream)` and ends up at
  `hipExtModuleLaunchKernel` via
  `tensilelite/src/hip/HipSolutionAdapter.cpp:455`.

### 6.4 hipblasLtMatmul per-call accounting (HIP API trace)

Captured with `AMD_LOG_LEVEL=3` on this box. For 5 iters at 8192³ the
window enclosed by `pre_gpu_time`/`post_gpu_time` contains **exactly**:

```
hipEventRecord(start, stream)               (one call)
hipExtModuleLaunchKernel(...) x 5           (one per iter, no others)
hipEventRecord(stop, stream)                (one call)
hipEventSynchronize(stop)                   (blocks until GPU done)
hipEventElapsedTime(...)                    (CPU read)
```

Inside the hot loop there is **no** `hipMemset`, **no** `hipMemcpy`,
**no** `hipDeviceSynchronize`, **no** `hipStreamSynchronize`, **no**
`hipMalloc`, and **no** `hipFree`. The only per-iter HIP entry point
is the kernel launch. Inter-launch host latency (time between two
`hipExtModuleLaunchKernel` entries) is 17-60 µs in the trace, which
caps any host-side per-call CPU overhead at well under the 474 µs
delta. The 5-iter end-to-end elapsed reported by
`hipEventElapsedTime` was 102.219 ms (~20.4 ms/iter), inflated by
first-call codegen/JIT; with 500 cold + 500 hot the steady-state is
1636 µs/iter as in §6.2.

`hipFree`s and `hipDeviceSynchronize` *do* appear in the trace but
only at shutdown, well after the timed window closes. The §4 claim of
"no on-stream work besides the GEMM" stands.

The internal hipblasLtMatmul → rocblaslt_matmul → tensile_host →
HipSolutionAdapter call chain was re-verified by reading
`library/src/amd_detail/hipblaslt.cpp:476-515`,
`library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp:43-226` and
`:683-810`, and
`library/src/amd_detail/rocblaslt/src/tensile_host.cpp:2749-2917`.
The only `hipFree` references in `tensile_host.cpp` (lines 2730 and
3293) are deleter functions for `shared_ptr<void>` wrapping
`hipHostMalloc`'d staging memory — these fire only at handle/data
destruction, not per call. No per-call `hipMemset` exists in the
matmul path.

### 6.5 Profiling / strace findings

#### Profiler attempts (both failed for the bench)

- `rocprofv3 --kernel-trace --hip-runtime-trace`: SIGSEGV inside
  `amd_comgr_do_action` triggered by `hipblasLtCreate`. Stack
  through `libhipblaslt.so.1` (ROCm 6.4.3) into
  `libamd_comgr.so.3` (ROCm 7.2.1). Cross-major-version conflict.
- `rocprofv2 --hsa-trace`: SIGSEGV during `LD_PRELOAD` injection
  with the same root cause.
- `rocprof` (v1): same cross-version mismatch expected; not retried.

Note: rocprof works for the okl_run binary (HIP 7) but cannot be
used on the bench binary (HIP 6) on this box without installing a
matching ROCm-6.4 profiler. That is outside read-only scope.

#### What AMD_LOG_LEVEL=4 reveals about queue setup

This is the new datapoint and it is decisive.

Bench (8192³, 5 iters):
```
Number of allocated hardware queues with low priority: 0, with normal priority: 0
Created SWq=...  to map on HWq=0x...e00000  with size 16384 with priority 1, cooperative: 0
Number of allocated hardware queues with low priority: 0, with normal priority: 1
Created SWq=...  to map on HWq=0x...00000   with size 16384 with priority 1, cooperative: 0
...
KernelExecution enqueued on SWq=...  HWq=...e00000  id=1  (FillBuffer/icache setup, 3 packets)
KernelExecution enqueued on SWq=...  HWq=...00000   id=2  (the GEMMs, all 5 of them)
```

okl_run (same problem, same number of iters):
```
Number of allocated hardware queues with low priority: 0, with normal priority: 0
Created SWq=...  to map on HWq=0x...e00000  with size 16384 with priority 1, cooperative: 0
...
KernelExecution enqueued on SWq=...  HWq=...e00000  id=1  (everything: setup AND the GEMMs)
```

The bench creates **two HSA hardware queues** (one for the initial
setup work on the null stream, one for the user stream); the GEMM
runs on **HWq id=2 (the user-stream queue)**. okl_run creates **one
HSA hardware queue** and the GEMM runs on **HWq id=1 (the only
queue)**.

`Dispatch Header = 0xb02` (barrier=1, acquire=1, release=1) is
identical on both sides for every dispatch packet. The AQL fence
semantics are the same. The only systemic difference visible is
which hardware queue carries the dispatch.

### 6.6 New best-fit hypothesis

**The 474 µs/iter gap is queue-induced: the bench's GEMM runs on a
secondary HSA hardware queue (id=2), while okl_run runs on the
primary/null-stream HSA hardware queue (id=1).** On gfx942 / MI300X
(8 XCDs), the HSA-to-HWq mapping and queue arbitration policy is
known to affect both (a) cross-XCD CU-affinity of the kernel and (b)
per-launch GPU-side dispatch latency.

Magnitude check: the kernel itself is 1172 µs. A 40% delta from
"same code, different queue" is on the high side but is in the right
order of magnitude for an XCC-affinity mismatch on MI300X. The
literature and Tensile's own MI300 tuning (`WGMXCCCHUNK`, the
`WGMXCC` knob, the `SK0` family) exists precisely because per-XCD CU
allocation hurts L2 hit-rate on multi-XCD chips. If the secondary
HWq is implicitly pinned to a subset of XCDs while the primary is
free-floating across all 8, you would see exactly this size-scaling
gap: ~0 µs on tiny shapes (data fits in any XCD's L2), 474 µs on
8192³ (working set spans all of HBM, sensitive to CU-to-data
affinity).

This explanation:
- Matches the byte-identical kernarg evidence (the kernel sees the
  same numWG/WGM/WGMXCC, but the *hardware* maps the workgroups to
  CUs differently).
- Matches the size-scaling property (queue-affinity effects are
  ~free for L2-resident tiles, costly for HBM-bound tiles).
- Matches the small-shape residual gap (~1-4 µs/iter) as the
  constant cross-queue dispatch/arbitration latency.
- Is consistent with §4's observation that the gap is visible to
  `hipEventElapsedTime`, i.e. on-stream rather than host-side.

Secondary contributor — **HIP runtime version skew** (`libamdhip64.so.6`
under the bench vs `libamdhip64.so.7` under okl_run). HIP 7's queue
acquisition logic (`rocdevice.cpp:acquireQueue`) changed enough that
ROCm 6.x and ROCm 7.x can give different HWq affinity decisions for
the same hipStreamCreate call sequence. If okl_run on HIP 6 also
landed its kernel on a non-primary HWq, it would slow down to match
the bench. This is the cleanest single experiment to run.

### 6.7 Recommended next experiment

**Run okl_run against `libamdhip64.so.6`** with `LD_PRELOAD`:

```bash
cd .../Testing/packages/large_square
LD_PRELOAD=/lib/libamdhip64.so.6 \
    /path/to/okl_run kernel.conf
```

(Attempted in this session but blocked by the auto-mode classifier
because `LD_PRELOAD` of a different HIP runtime was flagged as a
runtime-version bypass. The user should re-run it manually.)

Interpretation:
- If okl_run's time jumps from ~1165 µs to ~1640 µs, **the HIP
  runtime major version IS the source**, via different
  HWq-allocation behavior. Fix: either rebuild the bench against
  HIP 7, or rebuild okl_run against HIP 6 to show numbers users
  actually see.
- If okl_run stays at ~1165 µs under HIP 6, **HIP version is not
  the cause**; what remains is hipblasLtMatmul's use of a
  non-null stream specifically. Test that next by patching okl_run
  to call `hipStreamCreate(&s)` and using `s` in the launch — if
  okl_run then slows to ~1640 µs, **the secondary HSA HWq is the
  source**. Fix: have the bench launch on the null stream by
  default (or expose a `--null-stream` flag), or set CU mask /
  priority on the user stream to match the null stream's.

Additional confirmatory experiment if either above is ambiguous:
the bench under `HIP_FORCE_QUEUE_PROFILING=1` plus
`AMD_DIRECT_DISPATCH=1` (HIP 6 supports both) — compare
`AMD_LOG_LEVEL=4` HWq-id reports. If both runners then land on
HWq id=1 and times converge, the queue mapping was the direct cause.

#### What this addendum changes about §5 of the original report

- §5.1 (rocprof per-iter kernel count) was answered without rocprof
  by `AMD_LOG_LEVEL=3`: there is exactly one
  `hipExtModuleLaunchKernel` per iter, no auxiliary kernels.
- §5.2 (TENSILE_DB=0x40) is no longer needed; the user dump already
  confirms only the GEMM kernel runs.
- §5.3 (print kernel arg buffer) was answered by the user's dump:
  bytes match. Hypothesis refuted, see §6.1.
- §5.4 (one-line patch to cache the kernarg) would have no effect
  since the kernarg is already byte-identical between runs.

The relevant next investigations are now in §6.7, not §5.
