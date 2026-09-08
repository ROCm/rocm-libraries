# AI Agent Guidance

This file provides guidance for AI coding agents when working with code in this repository.

## Overview

TensileLite is an auto-tuning framework for generating and selecting high-performance GPU kernels for tensor contractions (GEMM and related operations) on AMD GPUs. It is a component of hipBLASLt. The Python package (`tensilelite/`) drives kernel generation and benchmarking; `rocisa` is a separately packaged Python dependency whose in-tree source has its own developer workflow; `include/` and `src/` form the C++ runtime library; and `client/` contains the benchmark executable.

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

## License headers

New source files (Python under `tensilelite/`, C++/HIP under `include/`, `src/`, `client/`, `rocisa/`, plus CMake, shell, and YAML) MUST start with the short SPDX license header. Use the comment form that matches the file's syntax.

C / C++ / HIP files (`//` comments):

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
```

Python / shell / CMake / YAML files (`#` comments):

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

The header goes at the very top of the file, immediately after a `#!` shebang line if one is present.

Do NOT paste the legacy verbose multi-line MIT block ("Permission is hereby granted, free of charge, ... THE SOFTWARE." plus the warranty disclaimer) into new files; the short SPDX header replaces it.

Existing files that still carry the legacy verbose MIT block MAY be migrated to the SPDX header when you are already editing them, but only when it does not materially grow the PR. If swapping headers would substantially increase the diff's line footprint (e.g. many files touched solely to change the header), leave those headers unchanged and keep the SPDX requirement scoped to net-new files.

## Pull requests

Always write PR descriptions using the rocm-libraries PR template. Fill in every section (use "N/A" or "Docs only, no testing needed" where a section genuinely does not apply rather than deleting it).

PR titles **must** follow [Conventional Commits](https://www.conventionalcommits.org/) style:

```
type(optional-scope): short description
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

The PR body **must** include a JIRA or issue reference line (before or after the template sections):

```
JIRA ID : PROJECT-1234
```

See the [Libraries PR Bot Policy FAQ](https://github.com/ROCm/rocm-libraries/blob/develop/docs/LIBRARIES_PR_BOT_FAQ.md) for the full set of automated checks enforced on every PR.

### PR body template

```markdown
JIRA ID : <JIRA key or N/A>

## Motivation
<why this change is needed: the problem, bug, or feature being addressed>

## Technical Details
<what changed and how; key design decisions and trade-offs>

## Test Plan
<how the change was/should be validated: builds, unit/gtest, smoke, manual steps>

## Test Result
<outcome of the test plan: passing suites, benchmark numbers, before/after>

## Submission Checklist
- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.

## Risk level
<None/Low/Medium/High, with a short justification>
```

Use the `users/<github-username>/<branch-name>` branch convention and base PRs on `develop`.

## Architecture

### Three-Phase Workflow

1. **BenchmarkProblems** (`tensilelite/BenchmarkProblems.py`): Generates kernel candidates from a YAML problem spec, builds them with rocisa, benchmarks on hardware. Output: `1_BenchmarkProblems/`, `2_BenchmarkData/`.

2. **LibraryLogic** (`tensilelite/LibraryLogic.py`): Analyzes benchmark data to pick the best kernel per problem size, generating heuristic selection logic as YAML/MsgPack. Output: `3_LibraryLogic/`.

3. **ClientWriter** (`tensilelite/ClientWriter.py`): Wraps the selected kernels in a C++ library and generates the benchmark client. Output: `4_LibraryClient/`.

Entry point: `python -m tensilelite run` → `tensilelite/tensilelite.py:tensilelite()` → `executeStepsInConfig()`.

### Key Python Modules

| Module | Role |
|--------|------|
| `tensilelite/KernelWriter.py` | Emits GPU assembly via rocisa calls (largest module) |
| `tensilelite/SolutionStructs/Solution.py` | Solution parameter validation and properties |
| `tensilelite/SolutionStructs/Problem.py` | Problem definition and validation |
| `tensilelite/Contractions.py` | Problem type taxonomy (GEMM, batched, grouped, sparse, stream-k) |
| `tensilelite/LibraryIO.py` | YAML/MsgPack serialization |
| `tensilelite/Common/` | Global parameters, architecture tables, utilities |
| `tensilelite/Components/` | Modular kernel building blocks (MAC variants, local/global read/write, scheduling) |
| `tensilelite/tensilelite_create_library/` | Standalone library-creation implementation (no benchmarking) |

### rocisa

`rocisa/` is a C++ module (compiled with amdclang++, bound via Nanobind) that provides instruction-level assembly generation, optimization passes, and instruction counting for AMDGPU kernels. `KernelWriter.py` calls into it to emit actual assembly instructions.

TensileLite packaging treats rocisa as opaque and independently supplied. Do
not infer or change rocisa's wheel tags, Python ABI, native-library placement,
or release version while working on the TensileLite package boundary.

Normal install (once after cloning, or after `rocisa/pyproject.toml` / `CMakeLists.txt` changes):

```bash
invoke rocisa            # editable pip install — picks up Python changes immediately
```

`rocisa/rocisa/__init__.py` runs a staleness check against a generated `_build_info.py`: if any `.cpp/.hpp/.h/.def/.inc` under the source roots is newer than the loaded `_rocisa.so`, import raises with a "rebuild" message. Pre-built wheels lack `_build_info.py` and skip the check.

### C++ Runtime Library

`include/Tensile/` and `src/` implement the runtime that selects and dispatches kernels at hipBLASLt call time. Key headers: `Tensile.hpp`, `ContractionProblem.hpp`, `ContractionSolution.hpp`, `SolutionLibrary.hpp`. `ContractionSolution.cpp` implements kernel dispatch.

## Gotchas

- `tox -e unit` installs editable rocisa and TensileLite, builds the client, and configures the active installation binding before unit tests. To run `pytest` directly outside tox, use `invoke install` or perform the equivalent editable install and binding steps.
- `tox -e py3` (the full common-tests env) does invoke `build-client` itself inside its `commands` block — that's where the "long client build" happens. Override its CMake/client args via `TENSILELITE_CLIENT_ARGS`, and parallelism via `TENSILE_NUM_PYTEST_WORKERS` (default 4).
- `invoke install` is the Linux one-command source-development setup: it installs shared dev requirements, runs the editable rocisa workflow, builds the client, installs TensileLite editably, and then binds that installation with `python -m tensilelite_configure_client --client <absolute-executable>`. Use `--reset` to remove the current keyed binding.
- Two test trees exist: `tensilelite/Tests/` (YAML kernel tests, run via `tox`/`pytest`) vs `tests/` (C++ host-library gtest, gated by CMake `TENSILELITE_BUILD_TESTING=ON`).
- `rocisa.egg-info/` and `rocisa/build/` in the working tree are normal (left by editable install / cmake build); don't commit them.
