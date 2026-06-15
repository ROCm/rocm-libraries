# AI Agent Guidance

This file provides guidance for AI coding agents when working with code in this repository.

## Overview

hipBLASLt is a GEMM library for AMD GPUs built on HIP. The public API (`hipblasLtMatmul`) computes `D = Activation(alpha * op(A) * op(B) + beta * op(C) + bias)` and forwards work to an internal `rocblaslt` layer, which dispatches to GPU kernels generated at build time by **TensileLite** (the in-repo fork of Tensile under `tensilelite/`). The runtime selects a kernel per problem size from heuristic logic files and lazy-loads the matching code object from a *device library* directory.

This directory (`projects/hipblaslt`) is one component of the larger `rocm-libraries` superbuild but is designed to also build standalone — see `CONTRIBUTING.md` for the standalone setup, which is the recommended dev loop.

## Repository layout (high-level)

| Path | Purpose |
|------|---------|
| `library/include/hipblaslt/` | Public C/C++ headers (`hipblaslt.h`, `hipblaslt-ext.hpp`, etc.) |
| `library/src/amd_detail/` | hipBLASLt API implementation (thin layer over rocblaslt) |
| `library/src/amd_detail/rocblaslt/` | Internal GEMM dispatch, handle/aux, Tensile host integration (`tensile_host.cpp`), user-driven tuning, transform ops, legacy rocRoller kernels under `src/rocroller/` |
| `tensilelite/` | Kernel generator + runtime (Python + C++/Nanobind). |
| `clients/tests/` | gtest binary (`hipblaslt-test`) driven by YAML in `clients/tests/data/` (`matmul_gtest.yaml`, `auxiliary_gtest.yaml`, `smoke_gtest.yaml`, `rocroller_gtest.yaml`, …) |
| `clients/bench/` | Benchmark binary (`hipblaslt-bench`) |
| `clients/samples/` | Standalone usage examples (`01_hipblaslt_gemm`, …) |
| `library/include/hipblaslt/hipblaslt_{float8,bfloat6,e5m3,e8,float4,float6,xfloat32}.h` | Custom narrow types |
| `tasks.py` | Top-level invoke tasks — primary build entry point (`invoke build`) |

The flow at runtime is: user call → `hipblaslt.cpp` → `rocblaslt_mat.cpp` → `tensile_host.cpp` → TensileLite host (`tensilelite/src/`, `tensilelite/include/`) → loaded `.hsaco`/`.co` from device library. The Tensile path is what new work uses; a small set of pre-existing custom kernels under `library/src/amd_detail/rocblaslt/src/rocroller/` still ships alongside (gated by `HIPBLASLT_ENABLE_ROCROLLER`, ON by default; pass `--skip-rocroller` to `invoke build` to drop it).

## Build

Preferred dev loop (from `projects/hipblaslt`) — top-level `invoke build` driven by `tasks.py`. It manages the venv (under `build/venv`), installs Python deps, and configures + builds via cmake:

```bash
invoke build -ca gfx942        # build host + device libs + clients for gfx942
invoke build -ca gfx942 -d     # add --install-deps on first run
invoke --help build            # full flag list
```

Useful flags (selected): `-d` install deps, `-n` install package after build, `-c` clients, `-d/-r/-k` Debug/RelWithDebInfo/RelWithDebInfo (default Release), `--clean`, `-a/--architecture` GPU target(s), `--skip-rocroller`, `-y/--legacy-hipblas-direct` (older-ROCm direct-hipBLAS API path), `-t/--no-tensile` for client-only, `-z/--no-lazy-load`, `-f/--logic-filter` to scope TensileLite logic dirs (massively faster device-lib build).

`install.sh` is a deprecated compatibility wrapper that just shells out to `invoke build`; new instructions and tooling should call `invoke build` directly.

For raw cmake invocations, cmake presets, and running tests — see `AGENTS_reference.md`. Read that file automatically whenever the task involves any of those topics.

## License headers

New source files MUST begin with the short SPDX license header, not the legacy verbose MIT block. The header goes at the very top of the file (immediately after a `#!` shebang line, if one is present).

For C / C++ / HIP files (`//` comments):

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
```

For Python / shell / CMake / YAML files (`#` comments):

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

Do NOT paste the legacy verbose multi-line MIT block (the `Permission is hereby granted, free of charge, …` text through `… THE SOFTWARE.` plus the warranty disclaimer) into new files.

Existing files that still carry the legacy verbose MIT block MAY be migrated to the SPDX header when you are already editing them, but only when it does not materially grow the PR. If swapping headers would substantially increase the diff's line footprint (e.g. many files touched solely to change the header), leave those headers unchanged and keep the SPDX requirement scoped to net-new files.

## Pull requests

Always write PR descriptions using the rocm-libraries PR template. Fill in every section (use "N/A" or "Docs only, no testing needed" where a section genuinely does not apply rather than deleting it):

```markdown
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

**Associated ticket**: <JIRA/issue id, or N/A>
```

Use the `users/<github-username>/<branch-name>` branch convention and base PRs on `develop`.

## When working in `tensilelite/`

`tensilelite/` is a self-contained subproject with its own toolchain (tox, invoke, rocisa C++ module). It has its own guide file covering kernel-generation workflow, rocisa, and the three-phase BenchmarkProblems → LibraryLogic → ClientWriter pipeline. Read that file before editing kernel codegen or YAML test logic there — the commands and gotchas (e.g. `tox -e unit` requiring a prior `invoke build-client`) do not apply at the hipBLASLt top level.

## Gotchas specific to this top-level

- Configure/run `invoke build` from `projects/hipblaslt`, not the repo root, unless you actually need the superbuild.
- The TensileCreateLibrary CMake step uses whatever Python CMake found — if it's missing PyYAML/msgpack, the device-library build fails silently mid-way. `invoke build` handles this; if you bypass it with raw cmake, point `Python_EXECUTABLE` at the venv (`build/venv/bin/python`).
- Building only the host library (`-DHIPBLASLT_ENABLE_DEVICE=OFF` or preset `hipblaslt`, or `invoke build -t`) is fast and fine for compile checks, but the resulting library cannot run matmul without a separately built/installed device library.
- After editing YAML under `clients/tests/data/`, you must rebuild the `hipblaslt-test` (or `hipblaslt-test-data`) target to regenerate `hipblaslt_gtest.data`.
- A full device-lib build is slow. Use `invoke build -f 'gfx942/Equality/*'` (or similar `--logic-filter`) to scope it to a single arch/family while iterating.
