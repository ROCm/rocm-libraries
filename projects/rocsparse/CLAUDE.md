# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository context

This directory (`projects/rocsparse`) is one project inside the larger `rocm-libraries` monorepo (the working
directory's parent-parent is the monorepo root, referenced in CMake as `ROCM_LIBRARIES_ROOT`). The current checkout
is on branch `gpuep-releases/therock-10.0`, a downstream release/integration branch used for AMD's "TheRock" build
system, which adds Windows/MSVC support on top of upstream ROCm. Recent history on this branch includes
Windows-specific build fixes (MSVC static runtime, Control Flow Guard, missing `amdhip64.lib` workarounds) across
several sibling projects (`rocblas`, `miopen`, `rocrand`, `composablekernel`). Keep this in mind when a change looks
unusual for a purely Linux/ROCm library — it may be there for Windows/TheRock compatibility.

rocSPARSE provides sparse BLAS-like routines (Level 1/2/3, conversion, preconditioners, reordering, generic SpMV/SpMM/SpGEMM/etc.)
implemented in HIP for AMD GPUs. hipSPARSE is the portable wrapper on top of this library; don't confuse the two.

## Build

Requires a HIP-clang toolchain (`amdclang++`); a plain Clang/GCC compiler is rejected by CMake with a `FATAL_ERROR`.

Typical Linux build:

```bash
mkdir -p build/release && cd build/release
CXX=/opt/rocm/bin/amdclang++ cmake -DBUILD_CLIENTS_TESTS=ON ../..
make
```

Or use the wrapper script from the repo root, which also handles dependency installation:

```bash
./install.sh -dci     # -d deps, -c clients, -i install after build
./install.sh -h       # see all options
```

Key CMake options (all defined in the top-level `CMakeLists.txt`):
- `BUILD_CLIENTS_TESTS` (OFF) — gtest-based unit tests, requires GoogleTest (auto-downloaded if missing, or set `GTEST_ROOT`)
- `BUILD_CLIENTS_BENCHMARKS` (OFF) — `rocsparse-bench`
- `BUILD_CLIENTS_SAMPLES` (OFF) — example programs
- `BUILD_WITH_ROCBLAS` (ON) — some routines depend on rocBLAS; `-n`/`--no-rocblas` in `install.sh` disables this
- `BUILD_ROCSPARSE_ILP64` — build with 64-bit `rocsparse_int`
- `BUILD_FORTRAN_CLIENTS` (ON) — Fortran API/bindings (`library/src/rocsparse.f90`), requires a Fortran compiler
- `BUILD_WITH_CSC_TRSV` / `BUILD_WITH_CSC_TRSM` — CSC format support in triangular solve routines
- `BUILD_WITH_ILDLT0` — incomplete LDL^T factorization support
- Windows-only: `USE_MSVC_STATIC_RUNTIME`, `USE_BINSKIM_COMPLIANT_COMPILE_FLAGS`, `USE_SPECTRE_MITIGATED_LIBRARIES`

`rmake.py` is an alternative Python-based build driver (parses similar flags: `-a/--architecture`, `-c/--clients`,
`-i/--install`, `--rocprim-path`, `--gtest-path`, `--matrices-dir`).

## Tests

Unit tests use GoogleTest and live under `clients/tests/` (test registration + `.yaml` parameter files) and
`clients/testings/` (the actual `testing_<routine>.cpp` implementations, shared between the test and benchmark binaries).

Build with `-DBUILD_CLIENTS_TESTS=ON`, then run the resulting binary directly:

```bash
cd build/release
./clients/staging/rocsparse-test                          # everything
./clients/staging/rocsparse-test --gtest_filter=*csrmv*    # a single routine
```

Test case parameters are NOT hardcoded in the `.cpp` files — each `test_<routine>.cpp` just registers a
GTest suite that is instantiated from the matching `test_<routine>.yaml`. `clients/common/rocsparse_gentest.py`
compiles all the per-routine YAML files (plus `clients/include/rocsparse_common.yaml` for shared defaults) into a
single binary data blob (`rocsparse_test.data`) at build time, which the test binary loads at runtime. To change
what parameter combinations a routine is tested with, edit the YAML, not the `.cpp`.

Curated test subsets are declared in `rtest.xml` and driven by `rtest.py`:

```bash
./rtest.py --emulation smoke        # fast sanity pass (rocsparse_smoke.yaml)
./rtest.py --emulation regression   # rocsparse_regression.yaml, needs --matrices-dir
./rtest.py --emulation extended     # rocsparse_extended.yaml, needs --matrices-dir
./rtest.py -t psdb                  # pre-checkin filter: *quick*:*pre_checkin*
./rtest.py -t osdb                  # nightly filter: *nightly*:*pre_checkin*
```

Benchmarks (`-DBUILD_CLIENTS_BENCHMARKS=ON`):

```bash
./clients/staging/rocsparse-bench -f csrmv --laplacian-dim 2000 -i 200
```

## Adding or modifying a routine

Per `.github/CONTRIBUTING.md`, a new routine requires, at minimum:
- `library/include/internal/<category>/rocsparse-<routine>.h` — public API declaration (categories: `level1`,
  `level2`, `level3`, `extra`, `precond`, `conversion`, `reordering`, `generic`, `utility`)
- `library/src/<category>/rocsparse_<routine>.cpp` — extern "C" wrapper per precision (s/d/c/z)
- `library/src/<category>/rocsparse_<routine>.hpp` — templated implementation
- `library/src/<category>/<routine>_device.h` — HIP device kernels (when applicable)
- `clients/testings/testing_<routine>.cpp` — GTest-driven correctness testing logic
- `clients/tests/test_<routine>.cpp` + `clients/tests/test_<routine>.yaml` — test registration + parameter matrix,
  and both files must be added to `ROCSPARSE_TEST_SOURCES` / analogous lists in `clients/tests/CMakeLists.txt`

At minimum support `float`, `double`, `rocsparse_float_complex`, `rocsparse_double_complex`. New routines are
expected to be performance-competitive (approximate GFLOPS/GB/s vs. hardware peak, compared against other sparse
libraries where relevant) — this project is selective about accepting new routines that don't clear that bar or
that meaningfully increase compile time / binary size / complexity.

## Architecture

rocSPARSE follows the "Hourglass API" pattern: a thin, C89-compatible public API (`library/include/`) backed by a
C++ implementation (`library/src/`), using opaque handle/descriptor types so users never see internal layout.

**`library/include/`** — everything exposed to consumers.
- `rocsparse.h` — umbrella header including everything else
- `rocsparse-auxiliary.h` — handle/descriptor management API
- `rocsparse-types.h` — public enums/typedefs
- `rocsparse-complex-types.h` — `rocsparse_float_complex` / `rocsparse_double_complex`
- `internal/<category>/` — per-category function declarations (level1/level2/level3/extra/precond/conversion/reordering/generic/utility)

**`library/src/`** — implementation, mirroring the same category subdirectories as `include/internal/`, plus:
- `include/` — shared internal infra: `handle.h`/`handle.cpp` (opaque handle struct), `status.h` (HIP→rocSPARSE
  status translation), `common.h`, `logging.h`, `utility.h`, and per-routine `*_info.hpp` structs that carry
  analysis/preprocessing state between an "analysis" call and the corresponding "solve"/compute call
- `auxiliary/` — handle creation/destruction, descriptor management implementation
- `hip/` — HIP-specific debug/utility glue

Each routine is generally split into three files following a fixed convention:
`rocsparse_<routine>.cpp` (extern "C" wrapper, one function per precision, dispatches to the templated impl),
`rocsparse_<routine>.hpp` (templated implementation parameterized on precision), and `<routine>_device.h`
(the actual HIP `__global__`/`__device__` kernels). Routines requiring a device-memory scratch buffer expose a
paired `..._buffer_size()` query — callers own allocation/deallocation and may reuse buffers across calls.

**`clients/`** mirrors this split for correctness testing:
- `testings/testing_<routine>.cpp` — the actual test logic (calls the API, compares against a reference/host
  implementation) — reused by both the test and benchmark binaries
- `tests/test_<routine>.cpp` + `.yaml` — GTest suite registration and the parameter matrix it's instantiated over
- `common/` — shared test infra: `rocsparse_host.cpp` (large reference/host-side implementations used as the
  correctness oracle), matrix importers/exporters (MatrixMarket, rocALUTION, rocsparseio formats), `rocsparse_gentest.py`
- `include/` — client-side test utilities and `rocsparse_common.yaml` (shared default test parameters)
- `benchmarks/` — `rocsparse-bench` sources
- `samples/` — standalone example programs

Fortran bindings live in `library/src/rocsparse.f90` / `rocsparse_enums.f90` (large, mostly-generated interface files).

## Code style

C/C++ formatted with `clang-format` (config in `.clang-format`) — use the ROCm-shipped version
(`/opt/rocm/llvm/bin/clang-format`), not a system-installed one, since results differ:

```bash
/opt/rocm/llvm/bin/clang-format -style=file -i <path>
# or, repo-wide:
git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format -style=file -i
```

`./.githooks/install` wires up a pre-commit hook that formats changed files automatically.

New files need the standard AMD MIT license header (see `.github/CONTRIBUTING.md` for the exact text).

## Docs

Sphinx docs live in `docs/`; `docs/conceptual/rocsparse-design.rst` is the authoritative design doc this file
summarizes. Build locally with:

```bash
cd docs && pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

or via CMake with `-DBUILD_DOCS=ON`.
