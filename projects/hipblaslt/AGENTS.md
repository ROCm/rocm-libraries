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

Raw cmake still works if you need it (e.g. for custom configures). After activating `build/venv`:

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx942 \
  -DPython_EXECUTABLE="$(pwd)/build/venv/bin/python" \
  -DPython3_EXECUTABLE="$(pwd)/build/venv/bin/python"
cmake --build build --parallel
```

CMake presets (`cmake --list-presets`): `default:release`, `hipblaslt` (host lib only, no device libs), `gemm-libs` (device libs only), `hipblaslt-clients`, `tensilelite`, `rocisa`, plus pinned configs `rocm-7.0.0` / `rocm-7.0.2`.

## Tests

Test data is **generated at build time** from `clients/tests/data/*.yaml` by `clients/tests/hipblaslt_gentest.py` into `build/clients/hipblaslt_gtest.data`. The binary looks for that file in its own directory, so:

```bash
cmake --build build --target tensilelite-device-libraries  # one-time per GPU config; slow
cmake --build build --target hipblaslt-test
cd build/clients && ./hipblaslt-test --gtest_filter='*quick*'
```

Runtime device-library lookup: `HIPBLASLT_TENSILE_LIBPATH` must point **directly** at the dir containing `TensileLibrary_lazy_<arch>.dat` and the code objects (e.g. `build/Tensile/library` or `/opt/rocm/lib/hipblaslt/library`). If unset, the loader resolves relative to the loaded shared library. Matmul tests will fail without a device library for the running GPU.

To skip a test on a known-broken platform, add an entry under `clients/tests/data/known_bugs.yaml` (matched by `function`/`initialization`/`known_bug_platforms`).

`rtest.py` at the repo root is a build-driven test runner — not the usual entry point for local dev.

## When working in `tensilelite/`

`tensilelite/` is a self-contained subproject with its own toolchain (tox, invoke, rocisa C++ module). It has its own guide file covering kernel-generation workflow, rocisa, and the three-phase BenchmarkProblems → LibraryLogic → ClientWriter pipeline. Read that file before editing kernel codegen or YAML test logic there — the commands and gotchas (e.g. `tox -e unit` requiring a prior `invoke build-client`) do not apply at the hipBLASLt top level.

## Gotchas specific to this top-level

- Configure/run `invoke build` from `projects/hipblaslt`, not the repo root, unless you actually need the superbuild.
- The TensileCreateLibrary CMake step uses whatever Python CMake found — if it's missing PyYAML/msgpack, the device-library build fails silently mid-way. `invoke build` handles this; if you bypass it with raw cmake, point `Python_EXECUTABLE` at the venv (`build/venv/bin/python`).
- Building only the host library (`-DHIPBLASLT_ENABLE_DEVICE=OFF` or preset `hipblaslt`, or `invoke build -t`) is fast and fine for compile checks, but the resulting library cannot run matmul without a separately built/installed device library.
- After editing YAML under `clients/tests/data/`, you must rebuild the `hipblaslt-test` (or `hipblaslt-test-data`) target to regenerate `hipblaslt_gtest.data`.
- A full device-lib build is slow. Use `invoke build -f 'gfx942/Equality/*'` (or similar `--logic-filter`) to scope it to a single arch/family while iterating.
