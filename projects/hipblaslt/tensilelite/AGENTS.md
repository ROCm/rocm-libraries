# AI Agent Guidance

This file provides guidance for AI coding agents when working with code in this repository.

## Overview

TensileLite is an auto-tuning framework for generating and selecting high-performance GPU kernels for tensor contractions (GEMM and related operations) on AMD GPUs. It is a component of hipBLASLt. The Python package (`Tensile/`) drives kernel generation and benchmarking; `rocisa/` provides a C++ (Nanobind-wrapped) assembly generation module; `include/` and `src/` form the C++ runtime library; and `client/` contains the benchmark executable.

## Commands

### Running Tests

```bash
# Full test suite (builds client + runs all common tests)
tox -e py3 -- Tensile/Tests -m common

# Python unit tests only (skips the long client build; requires a prior build)
tox -e unit -- Tensile/Tests/unit

# Run a specific test category
tox -e py3 -- Tensile/Tests -m gemm

# Run a single test directly (after a prior `invoke build-client`)
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out
```

### Building

```bash
# Build client to default location (build_tmp/)
invoke build-client

# Custom CMake build
cmake --preset tensilelite -S .. -B my-custom-build
cmake --build my-custom-build --parallel

# Run test with custom client path
./my-custom-build/Tensile.sh Tensile/Tests/common/<test>.yaml tensile-out \
    --prebuilt-client=my-custom-build/tensilelite-client/tensilelite-client

# Build with custom args (e.g., Debug + specific GPU)
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- Tensile/Tests -m common

# Detect local GPU architecture
invoke get-gpu-arch
```

### Linting and Formatting

```bash
tox -e lint          # flake8 (pyflakes errors only, E/W ignored)
tox -e format        # black (line-length=100) on Common/, TensileCreateLibrary/, Utilities/Decorators/
tox -e isort         # isort (black profile) on same directories
```

### Rebuilding Assembly Without Full Rerun

After a Tensile run creates `tensile-out/`, you can edit assembly and rebuild only object code:

```bash
make co TENSILE_OUT=tensile-out                          # auto-detect arch
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64   # gfx9 explicit
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32  # gfx11 explicit
```

## Architecture

### Three-Phase Workflow

1. **BenchmarkProblems** (`Tensile/BenchmarkProblems.py`): Generates kernel candidates from a YAML problem spec, builds them with rocisa, benchmarks on hardware. Output: `1_BenchmarkProblems/`, `2_BenchmarkData/`.

2. **LibraryLogic** (`Tensile/LibraryLogic.py`): Analyzes benchmark data to pick the best kernel per problem size, generating heuristic selection logic as YAML/MsgPack. Output: `3_LibraryLogic/`.

3. **ClientWriter** (`Tensile/ClientWriter.py`): Wraps the selected kernels in a C++ library and generates the benchmark client. Output: `4_LibraryClient/`.

Entry point: `Tensile/bin/Tensile` → `Tensile/Tensile.py:Tensile()` → `executeStepsInConfig()`.

### Key Python Modules

| Module | Role |
|--------|------|
| `Tensile/KernelWriter.py` | Emits GPU assembly via rocisa calls (largest module) |
| `Tensile/SolutionStructs/Solution.py` | Solution parameter validation and properties |
| `Tensile/SolutionStructs/Problem.py` | Problem definition and validation |
| `Tensile/Contractions.py` | Problem type taxonomy (GEMM, batched, grouped, sparse, stream-k) |
| `Tensile/LibraryIO.py` | YAML/MsgPack serialization |
| `Tensile/Common/` | Global parameters, architecture tables, utilities |
| `Tensile/Components/` | Modular kernel building blocks (MAC variants, local/global read/write, scheduling) |
| `Tensile/TensileCreateLibrary.py` | Standalone library-creation utility (no benchmarking) |

### rocisa

`rocisa/` is a C++ module (compiled with amdclang++, bound via Nanobind) that provides instruction-level assembly generation, optimization passes, and instruction counting for AMDGPU kernels. `KernelWriter.py` calls into it to emit actual assembly instructions.

Normal install (once after cloning, or after `rocisa/pyproject.toml` / `CMakeLists.txt` changes):

```bash
invoke rocisa            # editable pip install — picks up Python changes immediately
```

`rocisa/rocisa/__init__.py` runs a staleness check against a generated `_build_info.py`: if any `.cpp/.hpp/.h/.def/.inc` under the source roots is newer than the loaded `_rocisa.so`, import raises with a "rebuild" message. Pre-built wheels lack `_build_info.py` and skip the check.

Iterate on the C++ side without re-pip-installing:

```bash
cd rocisa && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ ..
make -j8
```

### C++ Runtime Library

`include/Tensile/` and `src/` implement the runtime that selects and dispatches kernels at hipBLASLt call time. Key headers: `Tensile.hpp`, `ContractionProblem.hpp`, `ContractionSolution.hpp`, `SolutionLibrary.hpp`. `ContractionSolution.cpp` implements kernel dispatch.

### Supported Targets

GPU architectures (see `Tensile/Common/Architectures.py`): gfx900, gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1010/1011/1012, gfx1030, gfx1100/1101/1102, gfx1200/1201, gfx1250 (each with optional `:xnack+/-`).

Test markers for architectures (see `pytest.ini`): `gfx11`, `gfx12`, `gfx94x`, `gfx950`, `gfx1250`, plus per-arch `xfail-gfxNNN` / `skip-gfxNNN`. Data type markers: `Float`, `Double`, `Half`, `BFloat16`, `Int8`, `Float8`/`BFloat8` (OCP and `_fnuz` NANOO variants), mixed `Float8BFloat8`, `Float4`, `Float6`, `BFloat6`.

## CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `TENSILELITE_ENABLE_HOST` | ON | Build C++ runtime library |
| `TENSILELITE_ENABLE_CLIENT` | ON | Build benchmark client |
| `TENSILELITE_ENABLE_AUTOBUILD` | OFF | Auto-rebuild rocisa wrapper scripts |
| `TENSILELITE_BUILD_TESTING` | OFF | Build C++ host library tests |
| `GPU_TARGETS` | (detected) | Semicolon-separated list of gfx targets |

## Gotchas

- `tox -e unit` skips the client build (hence "fast"); the env itself runs `pip install {toxinidir}/rocisa/` so it does **not** require a prior `invoke build-client` for rocisa to be importable. To run `pytest` directly outside tox, install rocisa once with `invoke rocisa`.
- `tox -e py3` (the full common-tests env) does invoke `build-client` itself inside its `commands` block — that's where the "long client build" happens. Override its CMake/client args via `TENSILELITE_CLIENT_ARGS`, and parallelism via `TENSILE_NUM_PYTEST_WORKERS` (default 4).
- Two test trees exist: `Tensile/Tests/` (YAML kernel tests, run via `tox`/`pytest`) vs `tests/` (C++ host-library gtest, gated by CMake `TENSILELITE_BUILD_TESTING=ON`).
- `invoke build-client` accepts `--clean`, `--build-dir`, `--build-type`, `--gpu-targets`, `--rocm-path`, `--export-compile-commands`, `--bundle-python-deps`, `--enable-rocprof`. See `tasks.py`.
- `rocisa.egg-info/` and `rocisa/build/` in the working tree are normal (left by editable install / cmake build); don't commit them.
