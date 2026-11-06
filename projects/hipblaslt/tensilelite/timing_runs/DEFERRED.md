# Deferred timing work

Work that came out of the timing investigations in this directory and was deliberately **not**
included in the current implementation plan. Each entry says where it comes from, what it is,
what is known, and what it would take to pick it up. Decisions themselves are recorded in
[DECISIONS.md](DECISIONS.md).

Status values: `deferred` (worth doing later), `not for now` (rejected for the current plan, but
could be revisited), `needs data` (blocked on a measurement).

## Summary

| # | Item | Source | Status | Size of the prize (as measured or estimated) |
| --- | --- | --- | --- | --- |
| 1 | Restore only writable tensors in `copyInputs` | Input data preparation, fix 3 | not for now | `copy_inputs_d2d` 26.3 s timer, real cost 4.5–5.5× that |
| 2 | Tune the host-side Synchronizer check for the FFM emulator | SynchronizerValidator, item 1-alt | deferred | unknown (emulator cost not measured) |
| 3 | Cheaper trig argument (`sinf` / argument reduction) | `cpu_data_init`, item 2b | not for now | trig init 22.1 → ~3.7 ns/element; changes generated data |
| 4 | MX data generator: single-threaded zero-fill and poor thread scaling | `cpu_data_init`, finding 1 | deferred | ~90 ms per 516M-element side; 7–11× on 25 threads |
| 5 | First-call fixed cost per client process | `cpu_data_init`, finding 4 | needs data | 54 s of the 106.8 s `cpu_data_init` is first calls |
| 6 | Typed reporter keys in `MetaResultReporter` | Per-solution bookkeeping, D3 | deferred | not measured |
| 7 | Shrink the `PostGSU16` reduction kernels | `python_kernel_build_src_co`, D | deferred | 48–59% of each remaining helper compile |
| 8 | Longest-first dispatch within a codegen step | `python_kernel_codegen`, D | deferred | 0–6% on most configs, 24% on `spmm_i8_sb` (upper bound) |
| 9 | Cheaper kernel generator (epilogue, `hasAnyDependency`) | `python_kernel_codegen`, E | deferred | epilogue 40–64% of per-kernel time; `hasAnyDependency` 25–67% on some sparse kernels |
| 10 | Reject over-budget kernels in solution validation, before codegen | `python_kernel_codegen`, B (follow-up) | deferred | removes the remaining rejected-kernel codegen entirely |

## 1. Restore only writable tensors in `copyInputs`

- **Source:** [input_data_preparation_investigation.md](input_data_preparation_investigation.md),
  finding 3; topic 2 in DECISIONS.md.
- **What:** on the slow reset path, `copyInputs` restores *every* tensor (D2D, GPU valid → GPU
  current) before each solution, including read-only A and B. Only tensors kernels can write need
  restoring: D, E, AMAX-D, bias in gradient mode, the Synchronizer, and possibly others.
- **Known:** `hipMemcpy` D2D returns in ~2 µs, so `gpu_input_reset_copy_inputs_d2d` (26.3 s) is
  deflated; the real serialized cost was 4.5–5.5× the timer on `spmm_f16` and
  `bufferLoadStore0`. The split between necessary and avoidable bytes was not measured.
- **Why not now:** too easy to get the writable set wrong, and a full restore also masks a kernel
  that corrupts its inputs.
- **To pick up:** enumerate the writable set per problem type from the kernel arguments (not by
  hand), add a debug mode that verifies read-only inputs are unchanged after a solution, then
  measure the avoidable bytes.

## 2. Host-side Synchronizer check for the FFM emulator

- **Source:** [SynchronizerValidator.md](SynchronizerValidator.md), item 1-alt; topic 3.
- **What:** the adopted design keeps a host-side check (full readback, then a vectorizable
  OR-reduction, with the count/first-offset scan only when dirty) for runs under the FFM emulator
  (`HSA_MODEL_MEMFILE`), where a 409,600-element reduction kernel may be much slower than a copy.
- **Known:** on hardware, OR-reduction first measured 11.3 µs warm / 32.6 µs right after the D2H
  copy for 1.6 MB, against 165 µs for today's scalar scan. Nothing was measured under emulation.
- **To pick up:** measure both the GPU check and the host check under the emulator; choose per
  mode, and tune (e.g. explicit SIMD, chunked readback) based on the measurement.

## 3. Cheaper trig argument in `initArrayTrig`

- **Source:** [cpu_data_init_investigation.md](cpu_data_init_investigation.md), finding 2;
  topic 4, item 2b.
- **What:** trig init calls double-precision `sin`/`cos` of the element index. glibc takes its
  slow large-argument path above ~1.06e8 (23.6–25 ns vs 6.9 ns). `sinf` + store measured
  3.7 ns/element; reducing the argument first is another option.
- **Why not now:** it changes the generated input data, so runs before and after would not use
  identical inputs. The value-preserving part (no per-element allocation, flat index) is adopted.
- **To pick up:** decide whether input-data compatibility with earlier runs matters; if not,
  switch to `sinf` or a bounded argument and re-baseline.

## 4. MX data generator: zero-fill and thread scaling

- **Source:** [cpu_data_init_investigation.md](cpu_data_init_investigation.md), finding 1;
  topic 4.
- **What:** in `generateMXInput` (`clients/common/src/mxDataGen.cpp`, shared hipBLASLt client
  code), `generate()` starts with a single-threaded `m_dataBytes.resize(n)` zero-fill (~90 ms for
  516 MB), and the whole call reaches only 7–11× on 25 threads (20,935 → 2,094 ms for 516M
  elements). The discarded reference-float vector is handled separately (adopted).
- **To pick up:** profile `generate()` per stage on a large side; parallelize or avoid the
  zero-fill; coordinate with hipBLASLt client owners since the code is shared.

## 5. First-call fixed cost per client process

- **Source:** [cpu_data_init_investigation.md](cpu_data_init_investigation.md), finding 4.
- **What:** the first `cpu_data_init` call in each client process (820 calls) is 54.0 s of the
  106.8 s. Standalone, the first H2H `hipMemcpy` after HIP init costs ~18 ms even for 1 KB, and
  the first OpenMP parallel region ~3 ms; in-client first calls were larger (e.g. 53.3 ms to copy
  1,024 elements in `spmm_i8`) and were not explained without a profiler. Configs split into many
  small client processes pay it repeatedly.
- **To pick up:** profile a first call (perf was unavailable on the investigation host); options
  include warming HIP/OpenMP at client start-up off the timed path, or fewer client processes per
  config.

## 6. Typed reporter keys

- **Source:** [per_solution_bookkeeping_investigation.md](per_solution_bookkeeping_investigation.md),
  option D; topic 5a, D3.
- **What:** `MetaResultReporter` fans every `report()` out to each reporter with a `std::string`
  key (and copies a `shared_ptr` per reporter, `for(auto r : m_reporters)`); each reporter then
  compares the key against its list. Replace with typed keys.
- **Known:** not measured. Much of the fan-out goes away with the adopted items (no granularity
  reports when unused, no stdout rows in tests, no results CSV unless consumed).
- **To pick up:** re-measure the per-solution bookkeeping section after the adopted items land;
  only worth doing if `report()` still shows up.

## 7. Shrink the `PostGSU16` reduction kernels

- **Source:** [python_kernel_build_src_co_investigation.md](python_kernel_build_src_co_investigation.md),
  option D; topic 6.
- **What:** the "universal" `PostGSU16` helper kernels are 4,260–5,865 lines each (vs 100–145 for
  `PostGSU8/4/2/1`). `KernelWriterConversion.kernelBody` emits a software-pipelined reduction with
  a fully unrolled 15-case remainder `switch`, each case re-emitting all 16 accumulate steps, with
  every inline-asm add duplicated under `#if defined(__gfx950__)`. Size grows roughly with GSU².
- **Known:** removing these kernels entirely cut a one-target compile by 48–59% (bound, not a
  proposal). Options: generate the remainder as a loop, lower `GlobalSplitUPGR`, or share one
  remainder path.
- **Why deferred:** it changes the code of kernels on the GSU path.
- **To pick up:** write a runtime benchmark of the reduction kernels (all GSU values and vector
  widths), then compare the alternative generators against it.

## 8. Longest-first dispatch within a codegen step

- **Source:** [python_kernel_codegen_investigation.md](python_kernel_codegen_investigation.md),
  option D; topic 7.
- **What:** joblib dispatches kernels in input order; dispatching the most expensive kernels first
  shortens the step's tail.
- **Known:** with perfect knowledge of per-kernel cost (simulated): 0–6% on most probed configs,
  24% on `spmm_i8_sb` (17.6 → 13.4 s). Output size is not known before generation, so a cost
  predictor is needed. With topic 7's C (one pool per config), the tail effect shrinks further.
- **To pick up:** re-measure after C lands; if tails still matter, build a cheap cost predictor
  (e.g. from tile sizes, epilogue features, or a previous run's per-kernel times).

## 9. Cheaper kernel generator

- **Source:** [python_kernel_codegen_investigation.md](python_kernel_codegen_investigation.md),
  option E; topic 7.
- **What:** per-kernel time is ~65% Python instruction-tree construction, ~21% `rocIsaPass`, ~8%
  freeing the tree and ~6% `str()`. Two concrete targets:
  - **Epilogue:** `notLocalSplitUGlobalWrite` is 40–64% of profiled time in 22 of 28 sampled
    kernels; it is fully unrolled per element for every variant (edge/non-edge, beta, bias type,
    activation). The code itself notes it (`KernelWriter.py`: "Kernels with epilog especially with
    activation is too long (50000~ lines). Need to refactor global write elements.").
  - **`KernelWriter.hasAnyDependency`:** a local-read × instruction scan, quadratic; 25–67% of
    profiled time in four `spmm_i8_sb` kernels (903k calls in one), 0% in `dtl`. A contained
    target.
- **To pick up:** start with `hasAnyDependency` (index the dependencies instead of rescanning);
  the epilogue is a larger refactor that changes generated code and needs performance validation.

## 10. Reject over-budget kernels before codegen

- **Source:** [python_kernel_codegen_investigation.md](python_kernel_codegen_investigation.md),
  option B; topic 7.
- **What:** the adopted fail-fast stops codegen early for kernels that are certain to be rejected
  (static VGPR allocation over `MaxVgpr`, or any `overflowedResources` code set). Moving the
  VGPR/SGPR checks into solution validation would keep these kernels out of codegen altogether.
- **Known:** not tested. After the adopted checks, rejected kernels cost ~10.8 s of CPU across
  the probed configs (down from 374.5 s).
- **To pick up:** check whether the static allocation can be computed without `_initKernel`, and
  whether rejecting at validation changes which solutions are reported or counted.
