# Contributing to hipBLASLt

This document is for **standalone hipBLASLt development and testing**—building and running tests from the hipBLASLt project alone, without the full rocm-libraries superbuild. It focuses on contributor-specific setup and gotchas.

- **General contribution flow** (issues, pull requests): see the [Contribute](README.md#contribute) section in the main README.
- **Building, installing, and client usage** (install script, manual build, running `hipblaslt-test` and `hipblaslt-bench`): see the [official hipBLASLt documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html):
  - [Building and installing hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/install/building-installing-hipblaslt.html)
  - [hipBLASLt clients](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/conceptual/hipblaslt-clients.html)
- **Code layout** (where the library and client source live—e.g. `library/include`, `library/src/amd_detail`, `clients/`): see [hipBLASLt library organization](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/conceptual/hipblaslt-library-organization.html) in the official docs.

Below is **contributor-specific** information: standalone configure from `projects/hipblaslt`, Python environment for device libraries, why tests need device libs, where to run tests so the data file is found, and how test data is generated.

---

## Standalone build: use the hipBLASLt project directory

**Configure and build from `projects/hipblaslt`** for a focused standalone workflow.

The repo root can build hipblaslt via the superbuild (e.g. `ROCM_LIBS_ENABLE_COMPONENTS`), but that configures additional projects. For a faster dev loop, configure directly in `projects/hipblaslt`:

```bash
cd rocm-libraries/projects/hipblaslt   # or your path to the hipblaslt project

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx90a

cmake --build build --parallel
```

You need a ROCm install (e.g. `/opt/rocm`) and Python 3.8+. For `GPU_TARGETS`, see [GPU targets: one, several, or all](#gpu-targets-one-several-or-all) below.

---

## GPU targets: one, several, or all

**`GPU_TARGETS`** (CMake) controls which AMD GPU architectures are built (Tensile device libraries and any arch-specific code).

- **One GPU:** e.g. `-DGPU_TARGETS=gfx90a`
- **Multiple GPUs:** semicolon-separated, quoted: `-DGPU_TARGETS="gfx90a;gfx942;gfx950"`
- **All supported:** `-DGPU_TARGETS=all` (or leave unset). Takes longer.

Supported names are in `cmake/tensilelite_supported_architectures.cmake`.

---

## Device libraries (Tensile): required for matmul tests

hipBLASLt uses a **build-time** generator (Tensile, in-repo) to produce GPU kernel libraries (`.dat` + `.hsaco`/`.co`), and a **runtime** (TensileLite host) that loads those artifacts when you call the matmul API. Most `hipblaslt-test` cases are matmul tests, so at runtime the library needs that Tensile device library for your GPU. If it's missing, matmul tests fail. You must either:

1. **Build the device libraries** in this repo (target `tensilelite-device-libraries`), which populates `build/Tensile/library/`, or  
2. **Use an existing ROCm install** and set `HIPBLASLT_TENSILE_LIBPATH` to the directory that contains `TensileLibrary_lazy_<arch>.dat` and the code objects.

### Python environment for building device libraries

The `tensilelite-device-libraries` target runs a Python script (`Tensile.TensileCreateLibrary`) that needs PyYAML, msgpack, etc. The build does not install them; it uses whatever Python CMake found. **Use a venv and install the Tensile requirements** so both the device-library step and the test-data generator (which also uses Python) have the right deps:

```bash
cd projects/hipblaslt
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS; on Windows: .venv\Scripts\activate
pip install -r tensilelite/requirements.txt
```

Configure CMake to use this Python:

```bash
cmake -B build -S . \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python" \
  -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" \
  ... # other flags (CMAKE_BUILD_TYPE, compiler, GPU_TARGETS, etc.)
```

### How the runtime finds the library

- **`HIPBLASLT_TENSILE_LIBPATH`** must point **directly** to the directory that contains `TensileLibrary_lazy_<arch>.dat`/`.yaml` and the `.hsaco`/`.co` files (e.g. `build/Tensile/library` or, for an install, `/opt/rocm/lib/hipblaslt/library`).
- If unset, the library looks relative to the loaded shared library (e.g. when running from a build tree it typically resolves to `build/Tensile/library`).

### Minimal steps to get tests running (from `projects/hipblaslt`)

```bash
# 0. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r tensilelite/requirements.txt

# 1. Configure (use venv Python)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx90a \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python" -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python"

# 2. Build device libraries (slow, one-time per config)
cmake --build build --target tensilelite-device-libraries

# 3. Build test binary and generate hipblaslt_gtest.data
cmake --build build --target hipblaslt-test

# 4. Run from build/clients so the data file is found
cd build/clients && ./hipblaslt-test
```

---

## Running the client tests (hipblaslt-test)

### Where does `hipblaslt_gtest.data` come from?

It is **generated at build time**, not checked in. A CMake step runs `clients/tests/hipblaslt_gentest.py` (needs PyYAML) to turn YAML in `clients/tests/data/*.yaml` into `build/clients/hipblaslt_gtest.data`. Building `hipblaslt-test` (or `hipblaslt-test-data`) runs that step.

**Run from the directory that contains both the test binary and the data file.** The executable looks for `hipblaslt_gtest.data` in the same directory (e.g. `build/clients/`):

```bash
cd projects/hipblaslt/build/clients
./hipblaslt-test
./hipblaslt-test --gtest_filter=*quick*
```

If you run from elsewhere, the test may not find the data file. For more run options and client details, see the [hipBLASLt clients](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/conceptual/hipblaslt-clients.html) documentation.

---

## Test data and adding/changing tests

- **YAML data:** Test cases are driven by YAML under `clients/tests/data/` (e.g. `matmul_gtest.yaml`, `hipblaslt_common.yaml`). The build generates `hipblaslt_gtest.data` from these.
- **Test coverage:** Inspect the YAML under `clients/tests/data/` and the test sources under `clients/`.
- **Known bugs:** Add an entry in `clients/tests/data/known_bugs.yaml` to mark a test as a known failure on one or more GPU architectures (e.g. `function`, `initialization`, `known_bug_platforms`). At test time, any `Arguments`-driven test (matmul, rocroller_predicate, aux) whose parameters match an entry and whose current GPU `gcnArchName` matches `known_bug_platforms` is skipped via `GTEST_SKIP()` and counted in the `[ SKIPPED ]` summary; the test still appears in the test list. Entries with no `known_bug_platforms` are treated as known-bug on all platforms. Tests under `matrix_transform_gtest.cpp` and `hipblaslt_gtest_ext_op.cpp` use raw gtest fixtures and are not driven by this YAML.
- After editing YAML under `clients/tests/data/`, rebuild so `hipblaslt_gtest.data` is regenerated.

---

## Summary checklist for a working dev/test loop

1. Configure and build from **`projects/hipblaslt`** (not repo root).
2. **Build device libraries** so `build/Tensile/library/` has the `.dat` and code objects for your GPU, or set `HIPBLASLT_TENSILE_LIBPATH` to an existing install.
3. **Run tests from `build/clients/`** so `hipblaslt_gtest.data` is next to `hipblaslt-test`.
4. Use `--gtest_filter` to run specific tests during development.

---

## Troubleshooting

| Symptom | What to check |
|---------|---------------|
| CMake fails or wrong layout from repo root | Configure from `projects/hipblaslt` instead. |
| Missing `TensileLibrary_lazy_*.dat` or `.hsaco` | Build `tensilelite-device-libraries` or set `HIPBLASLT_TENSILE_LIBPATH` to a directory that contains them. |
| TensileCreateLibrary fails with `ModuleNotFoundError` (e.g. `yaml`, `msgpack`) | Use a Python venv, `pip install -r tensilelite/requirements.txt`, and configure with `-DPython_EXECUTABLE=/path/to/venv/bin/python`. |
| `hipblaslt_gtest.data` doesn't exist | Build `hipblaslt-test` or `hipblaslt-test-data`. If the generator failed (e.g. missing PyYAML), use the venv and `pip install -r tensilelite/requirements.txt`, then rebuild. |
| Test can't find test data | Run from `build/clients/` (same directory as the executable and `hipblaslt_gtest.data`). |
| Test skipped on your GPU | Check `clients/tests/data/known_bugs.yaml` for an entry that matches your test and `known_bug_platforms`. |

---

*This file is a living document. Add new findings here as you run into them.*
