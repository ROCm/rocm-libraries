# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

rocSOLVER is a LAPACK implementation on the ROCm platform (AMD GPUs). It lives inside the `rocm-libraries` monorepo at `projects/rocsolver`. It requires rocBLAS as a companion GPU BLAS library.

## Build Commands

First-time configure and build (release mode with clients, targeting gfx1100):
```bash
./install.sh -cna gfx1100 --cmake-arg="-DROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF"
```

Subsequent incremental builds (after CMake is configured):
```bash
cd build/release && make
```

Build mode flags for `install.sh`:
| Flag | Effect |
|---|---|
| *(none)* | Release build -> `build/release/` |
| `-g` | Debug build -> `build/debug/` |
| `-k` | Release-debug build -> `build/release_debug/` |
| `-c` | Include clients (tests + benchmarks) |
| `-n` | Skip specialized small-matrix kernels |
| `-a gfx1100` | Target GPU architecture |

Example -- release-debug with clients (targeting several architectures):
```bash
./install.sh -kcna "gfx950;gfx942;gfx90a;gfx1100;gfx1201;gfx1151" --cmake-arg="-DROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF"
```

## Running Tests and Benchmarks

**Important:** The test/bench binaries link against the system `/opt/rocm/lib/librocsolver.so.0` by default, not the local build. Always prepend `LD_LIBRARY_PATH` to use the locally built library:

```bash
# From build/release (or build/release_debug):
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-test <args>
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-bench <args>
```

Test case names are case-sensitive in gtest filters. LATRD tests call `rocsolver_latrd_template` (the public API path); SYTRD tests call `rocsolver_latrd_forsytrd_template` for `n > xxTRD_xxTD2_SWITCHSIZE` (256).

```bash
# Standard CI subsets (from build/release)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-test --gtest_filter='checkin*LATRD*float/*'
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-test --gtest_filter='checkin*LATRD*double/*'
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-test --gtest_filter='checkin*SYTRD*float/*'
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-test --gtest_filter='checkin*SYTRD*double/*'
```

## Benchmarking

```bash
# From build/release:
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./bench_latrd.sh ./clients/staging/rocsolver-bench
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./bench_sytrd.sh ./clients/staging/rocsolver-bench

# With numerical verification
VERIFY=1 LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./bench_latrd.sh ./clients/staging/rocsolver-bench
```

For detailed guidance on benchmarking LATRD/SYTRD across all execution paths (multi-kernel,
fused canonical, fused software sync, fused software barrier), including all environment
variables, block-count tuning, and result interpretation, see:

**`docs/latrd_sytrd_benchmarking_guide.md`**

## Code Formatting

Install the git hook for automatic formatting:
```bash
./scripts/install-hooks
```

Or format manually:
```bash
clang-format -i -style=file <file>
```

## Architecture

### Function Organization Pattern

Each LAPACK function follows a 3-file pattern:
- `roclapack_<func>.hpp` -- template implementation (the actual algorithm)
- `roclapack_<func>.cpp` -- instantiation for non-batched case
- `roclapack_<func>_batched.cpp` / `roclapack_<func>_strided_batched.cpp` -- batched variants

The same pattern applies to auxiliary functions under `library/src/auxiliary/`.

### Key Layers

**Public API** (`library/include/rocsolver/rocsolver.h`): C99-compatible interface, prefixed with `rocsolver_` / `ROCSOLVER_`.

**Template implementations** (`library/src/`):
- `auxiliary/` -- Building blocks: LARFG (Householder reflectors), LATRD, LACGV, etc.
- `lapack/` -- Higher-level: SYTRD/HETRD (symmetric tridiagonalization), GEBRD, GEQRF, etc.
- `specialized/` -- Optimized small-matrix kernels (enabled by default, skip with `-n`)
- `common/` -- Shared utilities
- `refact/` -- Sparse refactorization functions

**GPU kernels** are defined directly in `.hpp` files using `ROCSOLVER_KERNEL` (wraps `__global__`). Template parameters control thread count (`MAX_THDS`) and precision (`T`).

**`library/src/include/`** -- Internal headers:
- `lib_device_helpers.hpp` -- Device-side utilities, matrix indexing
- `lapack_device_functions.hpp` -- Device-side LAPACK primitives
- `rocblas.hpp` -- C++ wrappers for rocBLAS (GEMV, SYMV, SCAL, etc.)
- `rocsolver_device_workspace.hpp` -- GPU workspace management
- `ideal_sizes.hpp` -- Block size constants

### SYTRD/HETRD -> LATRD Relationship

`roclapack_sytrd_hetrd.hpp` drives the blocked tridiagonalization:
1. Calls `rocsolver_latrd_forsytrd_template` in a loop to reduce `nb`-column panels
2. Updates the trailing submatrix with a rank-2k update via rocBLAS SYMM/HEMM + GER
3. Finishes the last panel with SYTD2/HETD2 (unblocked)

`rocauxiliary_latrd.hpp` contains the LATRD kernel implementation. Execution paths are selectable via environment variables:
- `LATRD_MULTI_KERNEL=1` -- multi-kernel path (separate kernel launches per step)
- `COOP_LAUNCH=1` -- force cooperative kernel launch (single persistent kernel)
- `LATRD_SW_GRID_SYNC=1` -- fused kernel with software grid sync (full L2 fences)
- `LATRD_SW_RAW_SYNC=1` -- fused kernel with software barrier (sc1 raw stores, no L2 fences)
- `LATRD_COOP_GRID_X=N` -- override thread block count for the fused kernel
- `LATRD_COOP_SWITCH_SIZE=N` -- use fused kernel only when n < N (default 8192)
- `PRINT_DEBUG=1` -- verbose HIP call tracing via `HIP_TRACE` macro

### Memory Model

Each function has a `_getMemorySize` function that computes scratch space requirements, and a `_template` function that uses the pre-allocated workspace. Memory is managed through `rocsolver_device_workspace` passed via the rocBLAS handle.

### Batched Operations

All functions support three variants: single, batched (array of pointers), and strided batched. Matrix arguments use the `U` template parameter (either `T*` for non-batched or `T* const*` for batched) with `shift` + `stride` indexing.

## PR Targets

PRs go to the `develop` branch. All public identifiers must be prefixed with `rocsolver_` or `ROCSOLVER_`.
