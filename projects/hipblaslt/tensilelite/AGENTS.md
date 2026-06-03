# AI Agent Guidance

This file provides guidance for AI coding agents when working with code in this repository.

## Overview

TensileLite is an auto-tuning framework for generating and selecting high-performance GPU kernels for tensor contractions (GEMM and related operations) on AMD GPUs. It is a component of hipBLASLt. The Python package (`Tensile/`) drives kernel generation and benchmarking; `rocisa/` provides a C++ (Nanobind-wrapped) assembly generation module; `include/` and `src/` form the C++ runtime library; and `client/` contains the benchmark executable.

## Working environment

```bash
# If you are outside the docker, and if you are asked to run using a docker. Ask the user for the container name.
docker exec <container> bash -ilc "command"

# If you are asked to run using a venv on Linux. Ask the user for the root of the venv
source <path-to-venv>/bin/activate && (the rest of the commands)
```

## Building

```bash
# Build client to default location (build_tmp/)
invoke build-client

# Detect local GPU architecture
invoke get-gpu-arch
```

For custom CMake builds, cmake presets, linting, running tests, rebuilding assembly, CMake options, and supported targets — see `AGENTS_reference.md`. Read that file automatically whenever the task involves any of those topics.

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

### C++ Runtime Library

`include/Tensile/` and `src/` implement the runtime that selects and dispatches kernels at hipBLASLt call time. Key headers: `Tensile.hpp`, `ContractionProblem.hpp`, `ContractionSolution.hpp`, `SolutionLibrary.hpp`. `ContractionSolution.cpp` implements kernel dispatch.

## Gotchas

- `tox -e unit` skips the client build (hence "fast"); the env itself runs `pip install {toxinidir}/rocisa/` so it does **not** require a prior `invoke build-client` for rocisa to be importable. To run `pytest` directly outside tox, install rocisa once with `invoke rocisa`.
- `tox -e py3` (the full common-tests env) does invoke `build-client` itself inside its `commands` block — that's where the "long client build" happens. Override its CMake/client args via `TENSILELITE_CLIENT_ARGS`, and parallelism via `TENSILE_NUM_PYTEST_WORKERS` (default 4).
- Two test trees exist: `Tensile/Tests/` (YAML kernel tests, run via `tox`/`pytest`) vs `tests/` (C++ host-library gtest, gated by CMake `TENSILELITE_BUILD_TESTING=ON`).
- `rocisa.egg-info/` and `rocisa/build/` in the working tree are normal (left by editable install / cmake build); don't commit them.
