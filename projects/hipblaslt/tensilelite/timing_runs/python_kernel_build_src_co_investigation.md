# `python_kernel_build_src_co` investigation (gfx950 per-config build)

`full_runs_gfx950/per_config_build/AGGREGATE_analysis.txt` reports `python_kernel_build_src_co` at
1,007,817.04 ms (39.1% of build wall time) over 855 calls. Essentially all of it is
`python_kernel_build_src_co.compile`: 1,004,724.33 ms over the 263 calls that missed the helper
kernel cache (mean 3.8 s). The cache already removes duplicate compiles, and PR #13671 overlaps
the compile with other work. This note looks at the work that is left: why a single compile takes
3–10 s, and what would make it substantially cheaper. Timing claims come from measurements; code
reading was only used to decide what to measure. Inferences are marked as such.

## Summary

| Option | Wall time | CPU time | Code produced | Notes |
| --- | --- | --- | --- | --- |
| **A. Build one xnack-agnostic object instead of `xnack+` and `xnack-`** | **−47 to −53%** | **−47 to −53%** | identical to today's `xnack+` | loader already tries the bare name first; asm kernels already ship this way |
| B. `-parallel-jobs=2` (if both variants are kept) | −45 to −50% | unchanged | identical | one flag |
| C. Split `Kernels.cpp` into ~4 chunks, compile in parallel, link once | −71 to −82% alone; −72 to −83% with A | +34 to +93% alone; −7 to −38% with A | identical | needs header-defined kernels moved into one TU |
| D. Make the `PostGSU16` reduction kernels smaller | up to −48 to −59% of the remaining work | same | **different** | generator change; runtime perf must be validated |
| E. Cached precompiled header of the HIP headers | −0.3 s per target per compile (−7 to −19%) | same | identical | removes most of the per-file fixed cost; works with today's build layout; makes C cheap in CPU |

### All results

Wall / CPU time of one helper-kernel compile, cores 0–15 idle. "Chunks" means `Kernels.cpp`
split into that many translation units, compiled in parallel (2 jobs per chunk for two targets,
1 per chunk for one target, capped at 16), then linked once per target.

| Option | Targets | `gsuasb` | `bias_sb` | `spmm_bf16_sb` | `grad_bias` | Code |
| --- | --- | --- | --- | --- | --- | --- |
| Today | `xnack+`, `xnack-` | 3.07 / 3.11 s | 6.09 / 6.13 s | 6.96 / 7.01 s | 9.35 / 9.40 s | — |
| B: `-parallel-jobs=2` | `xnack+`, `xnack-` | 1.57 / 3.12 s | 3.33 / 6.69 s | 3.47 / 6.97 s | 4.77 / 9.47 s | identical |
| C: 4 chunks | `xnack+`, `xnack-` | 0.88 / 6.00 s | 1.27 / 9.60 s | 1.35 / 10.25 s | 1.70 / 12.59 s | identical |
| C: 8 chunks | `xnack+`, `xnack-` | 0.82 / 10.26 s | 1.30 / 14.09 s | 1.15 / 15.07 s | 1.64 / 17.44 s | identical |
| A: one xnack-agnostic object | `gfx950` | 1.62 / 1.65 s | 3.07 / 3.09 s | 3.26 / 3.29 s | 4.58 / 4.60 s | identical to `xnack+` |
| A + C: 2 chunks | `gfx950` | 1.06 / 2.05 s | 1.74 / 3.44 s | 1.90 / 3.70 s | 2.61 / 4.91 s | identical |
| **A + C: 4 chunks** | `gfx950` | **0.85 / 2.87 s** | **1.10 / 4.36 s** | **1.17 / 4.52 s** | **1.60 / 5.86 s** | identical |
| D bound: one target, `PostGSU16` kernels removed | one | 0.90 s | 1.40 s | 1.60 s | 2.02 s | n/a (kernels missing) |

The D row is not a proposal: it removes the `PostGSU16` kernels entirely, to bound what shrinking
them could save (compare one target with them: 1.72, 3.18, 3.43 and 4.98 s). It was measured
separately from the other rows (median of 3).

Not worth pursuing (measured): lowering `-O` (slower), `-nogpulib` (link fails, no gain), and a
per-kernel cache (would skip 6% of kernel compiles beyond today's whole-file cache), and keeping a
compiler process alive (process start-up is only ~7–15 ms; see E).

## Relation to PR #13671

Checked against `origin/users/talumbau/tcl-speed-1-helper-overlap` as fetched on 2026-10-29
(`27a83902aa7` "compile helper kernels while generating asm kernels", `d6301b0de2e` tests).

- **What it changes.** In `writeSolutionsAndKernelsTCL` (the TensileCreateLibrary path), it
  submits `buildSourceCodeObjectFiles` to a one-thread `ThreadPoolExecutor` before assembly
  codegen starts and joins it after `buildAssemblyCodeObjectFiles`. The compile itself is
  unchanged: still both xnack targets, one sequential clang job. The commit message cites
  ~175 s for one gfx942 `Kernels.cpp`.
- **It does not touch the timer measured here.** `python_kernel_build_src_co` is recorded in
  `writeSolutionsAndKernels`, the benchmark path called from `BenchmarkProblems.py`. On the PR
  branch that function still runs codegen, assembly, `writeHelpers`, the assembly link and only
  then the helper compile, serially.
- **The same overlap in the benchmark path would hide at most 24%.** `writeHelpers` there only
  needs `kernelHelperObjs`, which is available on entry, so the compile could also start before
  codegen. Per `python_kernel_compilation` call in the 186 production logs, the hidden time is at
  most min(helper compile, codegen + validate + write_assemble + build_co of the same call):

  | Compile cost | Total compile | Upper bound hidden by overlap | Still exposed |
  | --- | --- | --- | --- |
  | Today | 1,004.7 s | 237.7 s (24%) | 767.0 s |
  | With A (× 0.5) | 502.4 s | 194.6 s | 307.8 s |
  | With A + C, 4 chunks (× 0.22) | 221.0 s | 143.6 s | 77.4 s |

  Helper compiles happen in only 263 of the 855 calls (the rest hit the cache), while the 920.8 s
  of assembly work is spread over all 855; only 17 of the 263 compiles are shorter than their
  call's assembly work. These are upper bounds (inferred): codegen runs on a worker pool, so a
  concurrent compile competes with it for cores, especially a chunked one.
- **The options here apply to the PR's path too.** The TCL path compiles the same kind of
  `Kernels.cpp` with the same `Compiler`, so A and B halve that ~175 s job and C splits it
  across cores, independently of the overlap (inferred; the gfx942 library build was not
  measured here).

## Method

- **Captured compiles.** Four configs re-built with `--build-only` exactly as
  `run_per_config_build.sh` does, with `TENSILE_DISABLE_HELPER_CACHE=1` and
  `Tensile_CXX_COMPILER_LAUNCHER` pointing at a wrapper that saves `Kernels.cpp`, the headers in
  its include directory, and the argv, then runs the command unchanged. The captured times match
  production:

  | Config | Kernels in `Kernels.cpp` | Lines | Captured compile | Production mean |
  | --- | --- | --- | --- | --- |
  | `gemm/gsuasb` | 32 | 28,377 | 2.95–3.12 s | ~3.0 s |
  | `gemm/gemm_bias_strided_batched_gfx940` | 50 | 55,990 | 5.98–6.47 s | 5.25 s |
  | `sparse/gfx950/spmm_bf16_sb` | 50 | 56,806 | 6.19–6.48 s | 6.23 s |
  | `gradient/gfx950/bf16_gradient_bias` | 74 (+14 in `Kernels.h`) | 84,062 | 9.20–9.83 s | 9.33 s |

- **Experiments** re-run the captured command (`amdclang++ -D__HIP_HCC_COMPAT_MODE__=1
  --offload-device-only -x hip -O3 -Xoffload-linker --build-id=sha1 -std=c++17
  --offload-arch=gfx950:xnack+ --offload-arch=gfx950:xnack-`, AMD clang 22.0.0git, ROCm 7.2.1)
  with one change at a time, on cores 0–15, which were idle during these runs (the host is
  shared).
- **Code equivalence** is checked per kernel: disassembly with addresses and comments stripped
  and each function cut after its last `s_endpgm` (inter-function alignment padding moves with
  function order), plus register, LDS, scratch and spill counts from the code-object metadata.
  Whole files are never byte-identical, even between two identical compiles: each compile gets a
  random `__hip_cuid_*` symbol and its own build ID.
- Scratch scripts and captures: `/tmp/syncval_bench/kcap` (`launcher.sh`, `run_build.sh`,
  `cc.sh`, `chunkbuild.py`, `perkernel.py`, captures in `<config>/NNN/`).

## Distribution in the aggregate

From the 186 per-config build logs (184 configs compile at least once): 263 compiles, median
3.1 s, p10 2.8 s, p90 6.1 s, max 9.5 s. 183 of them take 2–4 s; a second group of 33+29 takes
5–7 s (bias, grouped, sparse), and 7 take 8–10 s (gradient + bias). No single config dominates:
the largest, `subtile_mxfp4`, is 29.9 s over 9 compiles.

The number of distinct (`Kernels.cpp`, `Kernels.h`) pairs in each captured config equals its
production compile count (2, 3, 2, 3), so the existing cache already removes every exact
duplicate.

## What `Kernels.cpp` contains

Generated HIP C++ helper kernels: GSU reduction/conversion kernels (`*_PostGSU{16,8,4,2,1}_VW{4,2,1}`,
each with a grouped `_GG_` twin), global-accumulation kernels (`*_GA`), and for bias gradients,
reduction kernels. `KernelHelperNaming.initConversionKernelObjects` creates one conversion object
per load vector width; each expands into the whole `PostGSU` family, which the runtime chooses
from by the actual GSU value. All kernels are `extern "C"`; macros are `#define`d and `#undef`d per
kernel. `Kernels.h` holds the argument structs and declarations, and for bias gradients also
defines `Dijk_*_Reduction` kernels.

## Where the time goes

### 1. Every kernel is compiled twice (50%)

`--gpu-targets gfx950` expands to `gfx950:xnack+` and `gfx950:xnack-`
(`Common/Architectures.py`), and the driver compiles the device code once per target,
sequentially, front end included:

| Config | Both variants | One variant |
| --- | --- | --- |
| `gsuasb` | 3.16 s | 1.56 s |
| `grad_bias` | 9.27 s | 4.64 s |

### 2. A fixed ~0.4–0.7 s per target, almost all HIP headers

Single target, `gsuasb` include directory:

| Translation unit | Time |
| --- | --- |
| empty | 0.23–0.24 s |
| `#include <hip/hip_runtime.h>` | 0.33 s |
| + `<hip/hip_fp8.h>` | 0.39 s |
| `#include <iostream> <vector> <string> <algorithm>` | 0.24 s |
| `#include "Kernels.h"` | 0.41–0.42 s |
| full `Kernels.cpp`, `-fsyntax-only` | 0.44 s |

The headers cost ~0.42 s (`gsuasb`) to ~0.67 s (`grad_bias`, larger `Kernels.h`); parsing the
kernels themselves is cheap. `-nogpulib` does not change the empty-TU time and makes the link
fail (the kernels need `ocml`/`ockl`).

The "empty" row is not start-up: it includes `__clang_hip_runtime_wrapper.h`, which the driver
adds to every HIP file. Process start-up, driver, compiler start-up, device libraries, link and
bundle together are ~21 ms. Of the ~400 ms, ~350 ms is HIP headers and ~25 ms TensileLite static
headers; the per-header breakdown and how to remove it are in option E.

### 3. The rest is optimization and machine code generation, proportional to code size

`-ftime-report`, `gsuasb`, one target (1.54 s): machine code generation 0.69 s (45%), front end
0.46 s (30%), optimizer 0.35 s (22%), IR generation 0.05 s (3%). The largest passes are AMDGPU
instruction selection (0.19 s), `InstCombine` (0.13 s) and register coalescing (0.07 s); no single
pass is pathological.

Lower optimization makes it slower, because unoptimized code is more expensive to lower:

| Config, one target | -O0 | -O1 | -O2 | -O3 |
| --- | --- | --- | --- | --- |
| `gsuasb` | 2.86 s | 2.51 s | 1.55 s | 1.56 s |
| `grad_bias` | 8.19 s | 7.39 s | 4.53 s | 4.65 s |

### 4. The `PostGSU16` kernels are about half of it

The "universal" `PostGSU16` kernels (non-`UnrollOnly`, `GlobalSplitUPGR` = 16) are 4,260–5,865
lines each, against 100–145 lines for `PostGSU8/4/2/1`. `KernelWriterConversion.kernelBody`
generates a software-pipelined reduction: 16 prefetched loads, a `while` loop over full groups of
16, then a `switch(gsuRemain)` with 15 fully unrolled cases, each re-emitting all 16 accumulate
steps plus its interleaved loads, with every inline-asm add duplicated under `#if
defined(__gfx950__)`/`#else`. One kernel contains 392 inline-asm blocks; code size grows roughly
with GSU². Compiled alone, a `PostGSU16_VW4` kernel costs 210–226 ms, `VW2` 94–117 ms, `VW1`
80–92 ms, while a `PostGSU8` kernel costs 3–37 ms.

Removing just these kernels from each file (one target, median of 3):

| Config | Full | Without `PostGSU16` | `PostGSU16` share | `PostGSU16` kernels |
| --- | --- | --- | --- | --- |
| `gsuasb` | 1.72 s | 0.90 s | 48% | 6 of 32 |
| `bias_sb` | 3.18 s | 1.40 s | 56% | 12 of 50 |
| `spmm_bf16_sb` | 3.43 s | 1.60 s | 53% | 12 of 50 |
| `grad_bias` | 4.98 s | 2.02 s | 59% | 18 of 74 |

## Options

### A. Build one xnack-agnostic helper object

Compiling with `--offload-arch=gfx950` (no xnack feature) produces one code object that loads in
either xnack mode.

- **Time.** `gsuasb` 1.62–1.65 s against 3.07–3.16 s today; `bias_sb` 3.07 s against 6.09 s;
  `spmm_bf16_sb` 3.26 s against 6.96 s; `grad_bias` 4.58 s against 9.35 s. CPU time halves too.
- **Code.** All 32 `gsuasb` kernels are identical to today's `xnack+` object. Against `xnack-`,
  17 kernels differ by `s_nop` padding: 720 against 714 `s_nop 0` in 26,771 instructions.
- **Loading.** `SolutionAdapter::initializeLazyLoading` already tries
  `Kernels.so-000-<arch>.hsaco` before the `-xnack-` and `-xnack+` names.
- **Consistency.** The main assembly kernels already ship this way: `TensileLibrary_gfx950.co`
  has ELF flags `gfx950, xnack, sramecc` (agnostic), while the helpers are
  `Kernels.so-000-gfx950-xnack+.hsaco` (`xnack+`) and `-xnack-.hsaco` (`xnack-`).
- **Risk.** On `xnack-` systems the helpers would run the `xnack+`-style code (the extra
  `s_nop`s). The runtime effect was not measured. If shipped libraries must keep both variants,
  this could apply to benchmark/test builds only.

### B. Compile the two variants in parallel

`-parallel-jobs=2` makes the driver run the two device compiles concurrently. `--offload-jobs=2`
is ignored by this driver path ("argument unused").

| Config | Today | `-parallel-jobs=2` |
| --- | --- | --- |
| `gsuasb` | 3.07 s wall, 3.1 s CPU | 1.57 s wall, 3.1 s CPU |
| `bias_sb` | 6.09 s, 6.1 s | 3.33 s, 6.7 s |
| `spmm_bf16_sb` | 6.96 s, 7.0 s | 3.47 s, 7.0 s |
| `grad_bias` | 9.35 s, 9.4 s | 4.77 s, 9.5 s |

The disassembly is identical to the sequential build. Not needed if A is adopted.

### C. Split the file and compile the pieces in parallel

The kernels are independent, so `Kernels.cpp` can be split into N translation units that each
include `Kernels.h`. Each (chunk, target) is compiled to a relocatable object, and one `lld
-shared` link per target produces the single `Kernels.so-000-*.hsaco` the runtime loads (the
driver already does exactly this compile-then-link for one TU). The experiment ran the driver's
own `cc1` and `lld` commands (from `-###`) in a process pool, with chunks balanced by line count.

- **Code.** Every kernel identical, with identical metadata, in every case tested: 32, 50, 50 and
  88 kernels, both targets.
- **Header-defined kernels.** `grad_bias`'s `Kernels.h` defines 14 `Dijk_*_Reduction` kernels;
  with more than one chunk, the link fails with duplicate symbols. The experiment guarded them
  so only chunk 0 defines them; a real change should move them into the `.cpp`.

Both targets (wall / CPU):

| Config | Today | 1 chunk, 2 jobs | 4 chunks, 8 jobs | 8 chunks, 16 jobs |
| --- | --- | --- | --- | --- |
| `gsuasb` | 3.07 / 3.1 s | 1.61 / 3.2 s | 0.88 / 6.0 s | 0.82 / 10.3 s |
| `bias_sb` | 6.09 / 6.1 s | 3.35 / 6.7 s | 1.27 / 9.6 s | 1.30 / 14.1 s |
| `spmm_bf16_sb` | 6.96 / 7.0 s | 3.27 s / — | 1.35 / 10.3 s | 1.15 / 15.1 s |
| `grad_bias` | 9.35 / 9.4 s | 4.75 / 9.4 s | 1.70 / 12.6 s | 1.64 / 17.4 s |

Combined with A (one xnack-agnostic target):

| Config | Today | A alone | A + 2 chunks | A + 4 chunks |
| --- | --- | --- | --- | --- |
| `gsuasb` | 3.07 / 3.1 s | 1.62 / 1.65 s | 1.06 / 2.05 s | 0.85 / 2.87 s |
| `bias_sb` | 6.09 / 6.1 s | 3.07 / 3.09 s | 1.74 / 3.44 s | 1.10 / 4.36 s |
| `spmm_bf16_sb` | 6.96 / 7.01 s | 3.26 / 3.29 s | 1.90 / 3.70 s | 1.17 / 4.52 s |
| `grad_bias` | 9.35 / 9.4 s | 4.58 / 4.60 s | 2.61 / 4.91 s | 1.60 / 5.86 s |

Each extra translation unit repays the fixed ~0.4 s, so CPU time grows with the chunk count and
gains flatten beyond 4 chunks. In CI (4 xdist workers on 25 cores) the compile competes with the
other workers, so a small chunk count (or one chosen from free cores) is the safer default. A
cached precompiled header (option E) cuts the fixed cost per chunk from ~400 ms to ~35 ms.

The driver has no supported switch for "compile to a relocatable device object without
linking": `--no-gpu-bundle-output` still links. The supported alternative is `-fgpu-rdc`
(bitcode per TU, machine codegen in `lld` LTO, which the driver already runs with
`--lto-partitions=8`). On `gsuasb`, one target: bitcode compile 0.98 s, then `lld` with 8 LTO
partitions 0.30 s wall (0.97 s with 1), so ~1.28 s against 1.62 s. **But none of the 32 kernels
came out identical** (26,761 against 26,771 instructions), so this route needs performance
validation; invoking `clang -cc1`/`lld` directly, or `-S` plus assembling, keeps today's code.

### D. Shrink the `PostGSU16` kernels

The `PostGSU16` kernels are 48–59% of each compile (table above). Their size comes from the
unrolled remainder `switch` in `KernelWriterConversion.kernelBody` (non-`UnrollOnly` path).
Generating the remainder as a loop, lowering `GlobalSplitUPGR`, or sharing one remainder path
would cut most of that, but it changes the code of a kernel on the GSU path. It is only worth doing
with a runtime benchmark of the reduction kernels showing no regression (not measured here).

### E. A cached precompiled header for the fixed cost

With hundreds of compiles (or chunks), the fixed per-file cost matters. It is not process
start-up: clang has no persistent server mode, and it would not help much anyway. Breakdown of a
compile of `#include "Kernels.h"` with no kernels (`gsuasb`), one target, median of 15, cores 0–7.
Headers were added one at a time in include order, so each row is that header's increment; the
rows sum to 400.3 ms against 405.1 ms measured for the whole driver run:

| Step | Owner | Time |
| --- | --- | --- |
| process start (`amdclang++ --version`) | driver | 3.4 ms |
| driver planning the jobs (`-###`) | driver | 0.9 ms |
| compiler (`cc1`) start-up: empty file, no headers, no device libraries | compiler | 5.9 ms |
| device-library bitcode (`ocml`, `ockl`, `oclc_*`) | ROCm | 7.3 ms |
| `__clang_hip_runtime_wrapper.h` (implicit, added by the driver) | HIP | 192.7 ms |
| `<hip/hip_runtime.h>` | HIP | 100.0 ms |
| `<hip/hip_ext.h>` | HIP | 0.7 ms |
| `<hip/hip_fp8.h>` | HIP | 57.2 ms |
| C++ standard headers (`<string>`, `<algorithm>`, `<iostream>`, `<vector>`, `<cstddef>`, `<math.h>`, `<stdio.h>`) | C++ std | 1.5 ms |
| `tensile_bfloat16.h` | TensileLite | 3.4 ms |
| `tensile_float8_bfloat8.h` | TensileLite | 8.3 ms |
| `TensileTypes.h` | TensileLite | 10.5 ms |
| `ReductionTemplate.h` | TensileLite | 1.6 ms |
| `memory_gfx.h` | TensileLite | 0.0 ms |
| `KernelHeader.h` | TensileLite | 1.1 ms |
| `Kernels.h` (generated: argument structs and declarations) | TensileLite, generated | 2.2 ms |
| `lld` (link) | linker | 3.0 ms |
| `clang-offload-bundler` | bundler | 0.8 ms |

Subtotals: HIP 350.6 ms (88%), TensileLite static headers 24.9 ms, generated `Kernels.h`
2.2 ms, C++ standard headers 1.5 ms, everything else (process, driver, compiler start-up, device
libraries, link, bundle) 21.3 ms.

The same breakdown with a PCH of the HIP and TensileLite static headers (`shared.pch`: implicit
wrapper, `hip_runtime.h`, `hip_ext.h`, `hip_fp8.h`, `KernelHeader.h` and everything it includes),
measured side by side in a second run (median of 15). With the PCH, its load cost shows up in the
first header row; increments within ±5 ms (including the small negative ones) are noise:

| Step | Owner | No PCH | HIP + TensileLite PCH |
| --- | --- | --- | --- |
| process start (`amdclang++ --version`) | driver | 3.3 ms | 3.3 ms |
| driver planning the jobs (`-###`) | driver | 0.9 ms | 0.9 ms |
| compiler (`cc1`) start-up: empty file, no headers, no device libraries | compiler | 5.7 ms | 5.7 ms |
| device-library bitcode (`ocml`, `ockl`, `oclc_*`) | ROCm | 7.3 ms | 7.3 ms |
| `__clang_hip_runtime_wrapper.h` (implicit) / **loading the PCH** | HIP / PCH | 197.0 ms | **65.6 ms** |
| `<hip/hip_runtime.h>` | HIP | 105.3 ms | −0.4 ms |
| `<hip/hip_ext.h>` | HIP | −3.2 ms | −0.1 ms |
| `<hip/hip_fp8.h>` | HIP | 60.9 ms | 0.1 ms |
| C++ standard headers | C++ std | −0.4 ms | −0.5 ms |
| `tensile_bfloat16.h` | TensileLite | 9.9 ms | 0.7 ms |
| `tensile_float8_bfloat8.h` | TensileLite | 0.6 ms | −0.6 ms |
| `TensileTypes.h` | TensileLite | 4.8 ms | 0.5 ms |
| `ReductionTemplate.h` | TensileLite | 4.5 ms | 0.2 ms |
| `memory_gfx.h` | TensileLite | −1.3 ms | −0.4 ms |
| `KernelHeader.h` | TensileLite | 1.6 ms | 0.7 ms |
| `Kernels.h` (generated; not in the PCH) | TensileLite, generated | −1.7 ms | 2.3 ms |
| `lld` (link) | linker | 3.7 ms | 3.7 ms |
| `clang-offload-bundler` | bundler | 0.8 ms | 0.8 ms |
| **Sum** | | **399.5 ms** | **89.7 ms** |

Measured whole driver runs: 412.9 ms without the PCH, 96.7 ms with it. Subtotals: HIP
360.0 → 65.1 ms (the PCH load), TensileLite static headers 20.0 → 1.1 ms, generated `Kernels.h`
about 2 ms either way. The per-header TensileLite numbers vary by ~5 ms between runs (compare
with the table above); their subtotal (20–25 ms) is stable.

**Why "loading the PCH" still costs ~66 ms, and how to remove most of it.** It is not reading the
file: in a syntax-only run of an empty file with the PCH (56 ms), `-ftime-trace` shows `ReadAST`
0.5 ms, `-fno-validate-pch` changes nothing, the PCH records no input files to re-check, and
system calls are negligible (`strace -c`). The empty file contributes no functions or globals to
the module. 46.8 ms is `PerformPendingInstantiations`: the headers' inline code uses function
templates whose implicit instantiation is deferred to the end of the translation unit, and a PCH
stores that list, so every file that uses the PCH redoes them (483 `InstantiateFunction` events,
mostly C++ standard library: `std::basic_string` 24 ms inclusive, `std::vector`, `__stoa` from
`std::to_string`, `uninitialized_copy`, and the `<cmath>` special functions such as
`__cyl_bessel_j`). `-fpch-instantiate-templates` performs them once while building the PCH:

| Translation unit, one target | No PCH | PCH | PCH built with `-fpch-instantiate-templates` |
| --- | --- | --- | --- |
| empty file, `-fsyntax-only` | 192.9 ms | 56.5 ms | 14.0 ms |
| empty file, full compile | 210.2 ms | 79.0 ms | 33.6 ms |
| `#include "Kernels.h"`, full compile | 397.5 ms | 81.3 ms | 34.8 ms |

Building that PCH costs 497 ms instead of 465 ms (13.5 MB instead of 12.9 MB), and afterwards
`PerformPendingInstantiations` is 0.0 ms per file. The HIP-only version through the normal driver
(`amdclang++ <same flags> -x hip -fpch-instantiate-templates -Xclang -emit-pch -fsyntax-only
hip.h > hip.pch`), used on today's build directories, one target, median of 5:

| Config | No PCH | HIP PCH | HIP PCH, `-fpch-instantiate-templates` | Code |
| --- | --- | --- | --- | --- |
| `gsuasb` | 1.57 s | 1.28 s | 1.24 s | 32/32 identical |
| `bias_sb` | 3.05 s | 2.76 s | 2.72 s | 50/50 identical |
| `spmm_bf16_sb` | 3.26 s | 2.96 s | 2.94 s | 50/50 identical |
| `grad_bias` | 4.61 s | 4.31 s | 4.28 s | 88/88 identical |

`-fpch-instantiate-templates` saves another ~30–40 ms per translation unit compiled. With today's
single `Kernels.cpp` that is one translation unit per compile and target; with C (chunks) it is
saved once per chunk.

**How the PCH is used, and when it pays off.** In all of these measurements the PCH was built
once and reused unchanged by every later compile (all four configs, every repetition); the
per-compile times above do not include building it. In practice it would be built once per
(compiler/ROCm version, flags, target) and cached, like `HelperKernelCache`, or built with the
TensileLite installation. One-time cost: ~0.5 s per target. Saving: ~0.33–0.36 s per translation
unit per target (no PCH vs instantiated HIP PCH above). Arithmetic, not measured end to end:

- **Cache kept across configs (e.g. a CI job reusing `~/.tensile`):** built once per target for
  the whole run; 263 compiles × 2 targets × ~0.33 s ≈ 170 s saved for ~1 s of PCH builds.
- **Cold cache per config, as in `run_per_config_build.sh`:** the PCH is rebuilt for every config.
  The 184 configs average 263 / 184 = 1.43 compiles, i.e. ~2.9 uses with both targets
  (~0.9 s saved for ~1.0 s of builds) or ~1.4 uses with option A (~0.47 s saved for ~0.5 s):
  roughly break-even. With A + C at 4 chunks it is ~5.7 uses per config (~1.9 s saved for 0.5 s).

**Caching it like the helper kernels.** The intended design: build the HIP PCH once per
(compiler build, ROCm install, flags, target), store it next to the helper-kernel cache, and have
every later TensileLite run read it. `HelperKernelCache` already has the pattern
(`~/.tensile/helper_cache` or `TENSILE_HELPER_CACHE_DIR`; SHA-256 key; write to
`.tmp_<key>_<pid>`, then `rename`, which is safe with concurrent xdist workers; age-based
eviction). What clang does with a cached PCH that does not match (measured, one target):

| Cached PCH used with... | Result |
| --- | --- |
| the same flags (control) | works |
| a different xnack feature (`gfx950:xnack-` vs `gfx950`) | error: target feature mismatch |
| a different arch (`gfx942`) | error: target CPU mismatch |
| `-O2` instead of `-O3` | error: `OptimizationLevel differs` |
| `-std=c++20` instead of `-std=c++17` | error: C++20 disabled in the PCH |
| a header's mtime changed (content unchanged), or its content changed | fatal error: file modified since the PCH was built |
| `-fsanitize=address` added | accepted, with a warning that ASan is ignored for this target |
| a `-D` macro defined at PCH build time but not in the compile | **accepted silently; the macro stays defined in the file** |
| a `-D` macro with a different value | error: definition of macro differs |
| a `-D` macro defined only in the compile | works (normal behavior) |

So:

- **Key on the full compile flags**, including every `-D`, the target and its features, `-O`,
  `-std`, and ASan, because a macro baked into the PCH would otherwise leak silently into files
  that do not define it. Simplest: hash the exact argument list used to build the PCH (the same
  `Compiler.default_args` plus the per-target flags) and use that same list for the compiles.
- **Key on the exact compiler build**, not only `major.minor.patch`: `HelperKernelCache` uses
  `compiler.version` (`22.0.0` for this `22.0.0git` build), while clang rejects a PCH from a
  different compiler build. Hash the full `amdclang++ --version` output, which includes the commit.
- **Keep the PCH to HIP headers.** Clang validates every header in a PCH by mtime and size, and an
  mtime-only change is fatal (`-fvalidate-ast-input-files-content` did not relax it here). The
  TensileLite static headers are re-copied into each build directory (`shutil.copy`, new mtime),
  so a PCH containing them would fail on every build unless built and used from one fixed path.
  The HIP headers sit in the ROCm install, unchanged until ROCm is reinstalled; this is also where
  ~350 of the ~375 ms of header cost is.
- **Fall back on errors.** A reinstall of the same ROCm version changes header mtimes; the
  compile then fails with "has been modified since the precompiled header was built". On any PCH
  error, rebuild the PCH (or compile without it) and retry, so a stale cache entry costs one
  rebuild, not a failed build.

That is header parsing, which a precompiled header (PCH) removes. Splitting it with two PCHs
(cc1 jobs, one target, median of 9; `hip.pch` = implicit wrapper + `hip_runtime.h` +
`hip_ext.h` + `hip_fp8.h`, 434 ms to build, 12.5 MB; `shared.pch` = that + `"KernelHeader.h"`):

| Translation unit | No PCH | `hip.pch` | `shared.pch` |
| --- | --- | --- | --- |
| empty file (only the implicit `__clang_hip_runtime_wrapper.h`) | 205 ms | 69 ms | 76 ms |
| `hip_runtime.h`, `hip_ext.h`, `hip_fp8.h` | 364 ms | 70 ms | 77 ms |
| `#include "Kernels.h"` (`gsuasb`) | 389 ms | 100 ms | 78 ms |

(Empty file with the implicit wrapper removed: 13 ms.) So the fixed cost is almost all HIP
headers: ~192 ms for the wrapper the driver adds to every HIP file and ~160 ms for the
`hip_runtime.h`/`hip_ext.h`/`hip_fp8.h` that Tensile's headers include, ~350 ms in total. The
Tensile static headers themselves add only ~25 ms. (`grad_bias`'s `Kernels.h` costs more, 626 ms,
because it contains kernel code.) With a PCH, ~70 ms remains: loading the PCH plus an empty
compile.

**The HIP-only PCH is the one to use.** It gets almost all of the saving, depends only on the
ROCm/clang version, flags and target (the headers live at fixed `/opt/rocm` paths), and works with
today's build directories unchanged. Built with the normal driver and used on the captured build
directories as they are (static headers copied next to `Kernels.h`), one target:

| Config | No PCH | HIP-only PCH | Code |
| --- | --- | --- | --- |
| `gsuasb` | 1.54 s | 1.25 s | 32/32 identical |
| `bias_sb` | 2.99 s | 2.71 s | 50/50 identical |
| `spmm_bf16_sb` | 3.18 s | 2.91 s | 50/50 identical |
| `grad_bias` | 4.53 s | 4.21 s | 88/88 identical |

Adding the Tensile static headers to the PCH saves another ~10–50 ms per compile (table below),
but requires changing where those headers are found (see "Header location").

Only `Kernels.h` is generated (`writeHelpers`; 10 distinct versions across the 27 captured
compiles). The other six headers (`TensileTypes.h`, `tensile_bfloat16.h`,
`tensile_float8_bfloat8.h`, `KernelHeader.h`, `ReductionTemplate.h`, `memory_gfx.h`) are static
files shipped in the package at `tensilelite/Source/`. `copyStaticFiles` →
`resources.copy_static_headers` copies them verbatim (`shutil.copy`) next to `Kernels.cpp` for
every build, so they are identical by construction; they only change when the TensileLite
checkout changes. The HIP headers come from the ROCm install. So one PCH of
`<hip/hip_runtime.h>`, `<hip/hip_ext.h>` and `"KernelHeader.h"` per target can serve every config
built with the same TensileLite and ROCm. Built once (463 ms, 12.9 MB) from a shared directory of
the static headers, then used for each config's whole `Kernels.cpp`, one target:

| Config | `#include "Kernels.h"` only | Whole `Kernels.cpp` | Code |
| --- | --- | --- | --- |
| `gsuasb` | 389 → 78 ms | 1.53 → 1.22 s | 32/32 identical |
| `bias_sb` | 392 → 82 ms | 2.97 → 2.66 s | 50/50 identical |
| `spmm_bf16_sb` | 392 → 81 ms | 3.18 → 2.89 s | 50/50 identical |
| `grad_bias` | 626 → 318 ms | 4.53 → 4.20 s | 88/88 identical |

- **With chunks (C).** Each chunk pays ~80 ms instead of ~390 ms of headers. Measured with a
  per-config PCH and 4 chunks, one target: total compile CPU 2.72 → 1.50 s (`gsuasb`) and
  4.26 → 2.98 s (`bias_sb`), all kernels identical. This removes most of C's extra CPU cost.
- **Through the normal driver.** Build: `amdclang++ <same flags> -x hip -Xclang -emit-pch
  -fsyntax-only hip.h > hip.pch`, where `hip.h` includes `<hip/hip_runtime.h>`,
  `<hip/hip_ext.h>` and `<hip/hip_fp8.h>` (`-x hip-header` is rejected, and with
  `-fsyntax-only` the PCH is written to stdout, not `-o`). Use: add `-include-pch hip.pch`.
- **Caching.** A PCH is only valid for the same compiler, flags (`-O3`, `-std`, `-D`), target and
  xnack feature, and header contents, so it would be cached per target keyed on those (e.g. the
  ROCm/clang version plus a hash of `tensilelite/Source/*.h`), like `HelperKernelCache`.
- **Header location (only for the Tensile part).** Today the copies sit next to `Kernels.h`, and `Kernels.h` includes
  `"KernelHeader.h"` with quotes, so the compile resolves to the per-build copy. For the PCH to be
  used, the compile must resolve the static headers to the same files the PCH was built from:
  build the PCH against one fixed directory (e.g. the package's `Source/`) and have the compile
  find the static headers there, not in per-build copies. `KernelHeader.h`, `TensileTypes.h` and
  `memory_gfx.h` use `#ifndef` guards, which would hide a duplicate copy, but
  `ReductionTemplate.h`, `tensile_bfloat16.h` and `tensile_float8_bfloat8.h` use `#pragma once`,
  which is per file, so a second copy at another path would be parsed again. In the experiment
  `Kernels.h` was in its own directory and the static headers only in the shared one.
- **Aggregate (inferred).** ~0.3 s per target per compile × 263 compiles × 2 targets ≈ 160 s of
  the 1,004.7 s, before any chunking; more with chunks, since every chunk saves it.

## Measured and not worth pursuing

- **Lower `-O`:** -O0/-O1 are 1.6–1.8× slower than -O3; -O2 is the same as -O3.
- **`-nogpulib`:** no change in the fixed cost, and the link fails (undefined `ocml`/`ockl`
  symbols).
- **Per-kernel cache:** across the distinct files compiled for each captured config, 474 of 506
  kernel blocks are unique; a per-kernel cache would skip 32 (6%) beyond today's whole-file cache.
  Cross-config reuse was not measured: these 4 configs share almost nothing, and production builds
  were cold per config.

## Not measured / open items

- Runtime performance of the xnack-agnostic helpers on an `xnack-` system (option A), and of
  `-fgpu-rdc` code (option C variant) or a smaller `PostGSU16` (option D).
- End-to-end build-time effect, and the real (contended) overlap if PR #13671's approach is
  extended to the benchmark path, under CI's xdist load.
- The gfx942 TensileCreateLibrary `Kernels.cpp` (~175 s per the PR) was not captured.
- A precompiled header combined with chunking end to end (measured: per-config PCH with 4
  chunks, and the instantiated HIP PCH on whole files, separately); the PCH cache itself is a
  design, not implemented.
- Only 4 of the 184 configs were captured; the per-config distribution above suggests the rest
  are the same kind of files (most compiles 2–4 s), but they were not inspected individually.
