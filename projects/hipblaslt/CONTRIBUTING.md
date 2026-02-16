# Contributing to hipBLASLt

This document captures how to do **standalone hipBLASLt development and testing**—building and running tests from the hipBLASLt project alone, without the full rocm-libraries superbuild. It is a running record of findings; we add to it as we hit new gotchas.

For general contribution flow (issues, pull requests), see the [Contribute](README.md#contribute) section in the main README.

---

## Architecture and dependencies: hipBLASLt, Tensile, TensileLite, and related components

This section summarizes how hipBLASLt is structured and how it relates to Tensile, TensileLite, and other dependencies. It is a quick reference for developers navigating the codebase.

### What hipBLASLt is

hipBLASLt is the **public HIP API** for flexible matrix-matrix operations (especially GEMM: *D = α·op(A)·op(B) + β·op(C) + bias*, with optional activations). It targets AMD GPUs via ROCm. The implementation lives in this repo; there is no separate “rocBLASLt” product—the internal implementation is named **rocblaslt** (in `library/src/amd_detail/rocblaslt/`) and implements the same conceptual API that hipBLASLt exposes to users.

### Layering (high level)

1. **hipBLASLt (public API)** — C API in `library/include/hipblaslt/`; thin wrappers in `library/src/amd_detail/hipblaslt*.cpp` that call into rocblaslt.
2. **rocblaslt (internal implementation)** — Core logic in `library/src/amd_detail/rocblaslt/`: handle/state, matrix layout, matmul descriptor, and **tensile_host.cpp**, which loads the Tensile device library and dispatches GEMM to the TensileLite host runtime.
3. **TensileLite host library** — C++ library in `tensilelite/` (target `tensilelite::tensilelite-host`). Linked into the hipblaslt library. At runtime it:
   - Loads the **Tensile library** (`.dat` / `.yaml` solution index and `.hsaco`/`.co` code objects) from disk (path from `HIPBLASLT_TENSILE_LIBPATH` or next to the shared library).
   - Selects solutions (kernels) for the given problem and launches HIP kernels.
4. **Tensile (build-time)** — The **device library generator**. In this repo it is the Python package under **`tensilelite/Tensile/`** (e.g. `TensileCreateLibrary`). It is *not* the separate [ROCm/Tensile](https://github.com/ROCm/Tensile) repo used by rocBLAS; hipBLASLt uses its own in-repo “Tensile” inside TensileLite. At build time, the **tensilelite-device-libraries** CMake target runs `python -m Tensile.TensileCreateLibrary` to:
   - Read library logic (YAML) from `library/` (or `HIPBLASLT_LIBLOGIC_PATH`),
   - Generate architecture-specific solution libraries and GPU code objects,
   - Write `TensileLibrary_lazy_<arch>.dat` and the `.hsaco`/`.co` files into `build/Tensile/library/` (or `HIPBLASLT_TENSILE_LIBPATH`).

So: **“Tensile”** in hipBLASLt = the in-repo Python-based generator and the **device** artifact it produces (`.dat` + code objects). **“TensileLite”** = the C++ **host** runtime (and the surrounding project: that runtime + the in-repo Tensile Python + rocisa + origami) that loads and runs those artifacts.

### Key dependencies (conceptual)

| Component | Role |
|----------|------|
| **Tensile (in tensilelite/Tensile/)** | Build-time: Python scripts that generate GEMM solution libraries and GPU code objects (`.dat` + `.hsaco`/`.co`) from YAML logic. Invoked by the `tensilelite-device-libraries` target. |
| **TensileLite host** | Runtime: C++ library that parses the Tensile library (msgpack/yaml), selects solutions, and runs HIP kernels. Linked into `libhipblaslt.so`. |
| **rocisa** | Python module (nanobind) under `tensilelite/rocisa/`. Used by the Tensile (TensileCreateLibrary) build-time pipeline for ISA/assembly handling. Built as a dependency of the device-library build. |
| **origami** | Shared C++ library (`shared/origami` in rocm-libraries). Analytical GEMM solution selection (tile sizes, mapping). Linked by the TensileLite host library. |
| **RocRoller** | Optional (`HIPBLASLT_ENABLE_ROCROLLER`). Alternative kernel path and custom kernels (e.g. in `rocblaslt/src/rocroller/`). Can add custom `.co` files into the same output directory as the Tensile library. |
| **hip / hipblas-common (or hipblas)** | HIP runtime; hipblas-common (or hipblas in legacy mode) for some API compatibility. Required by the hipBLASLt library. |
| **BLIS / LAPACK** | Used by the **clients** (tests/benchmarks) for reference comparisons, not by the core library. |
| **msgpack-cxx (or msgpackc-cxx)** | Used by the TensileLite host to deserialize the Tensile solution library (when not using YAML). |

### Data flow (GEMM path)

1. User calls `hipblasLtMatmul` (or similar) → hipBLASLt wrapper → rocblaslt.
2. rocblaslt (e.g. in **tensile_host.cpp**) ensures the Tensile library is loaded (lazy load: open `TensileLibrary_lazy_<arch>.dat` and related code objects from `HIPBLASLT_TENSILE_LIBPATH` or the default path).
3. TensileLite host uses the loaded library to find a solution for the problem, then runs the corresponding kernel (HIP launch).
4. If RocRoller is enabled, some paths may use rocroller’s custom kernels or selection instead.

### Why the naming is confusing

- **“Tensile”** elsewhere in ROCm (e.g. rocBLAS) often refers to the standalone [Tensile](https://github.com/ROCm/Tensile) repo. In hipBLASLt, “Tensile” is the **in-repo** generator under `tensilelite/Tensile/` and the **device** libraries it produces. Same idea (library generator + artifacts), different codebase.
- **“TensileLite”** is the name of the hipBLASLt subproject that contains both (a) the **host** C++ runtime and (b) the in-repo **Tensile** Python + logic. So “TensileLite” encompasses “Tensile” here.
- **rocblaslt** is the internal implementation name (no separate rocBLASLt package); the public name is hipBLASLt.

Keeping “build-time Tensile (generator + device libs)” vs “runtime TensileLite (host library loading those libs)” in mind helps when debugging missing `.dat`/`.hsaco` or build failures in the device-library step.

---

## Standalone build: use the hipBLASLt project directory

**Configure and build from `projects/hipblaslt`**, not from the rocm-libraries repo root.

- The repo root uses a superbuild that does not include the CMake modules and layout that hipBLASLt expects (e.g. `add_subdirectory_with_message`, `fetch_rocm_cmake`). Configuring from the root will fail or produce an incomplete build.
- Always run CMake with the hipBLASLt project as the source directory:

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

See [GPU targets: one, several, or all](#gpu-targets-one-several-or-all) below for how to set `GPU_TARGETS`. You need a ROCm install (e.g. under `/opt/rocm`) and Python 3.8+.

---

## GPU targets: one, several, or all

**`GPU_TARGETS`** (CMake cache string) controls which AMD GPU architectures are built. It affects the Tensile device libraries and any architecture-specific code.

- **Build for all supported GPUs:** set `GPU_TARGETS` to **`all`** (or leave it unset). The project will use its built-in list of base architectures (e.g. gfx908, gfx90a, gfx942, gfx950, gfx1100, gfx1101, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153, gfx1200, gfx1201). Building for “all” takes longer because the Tensile generator runs for each architecture.

  ```bash
  cmake -B build -S . ... -DGPU_TARGETS=all
  ```

- **Build for one GPU:** set a single target, e.g. `gfx90a`, `gfx942`, `gfx950`.

  ```bash
  cmake -B build -S . ... -DGPU_TARGETS=gfx90a
  ```

- **Build for multiple specific GPUs:** pass a **semicolon-separated** list. From the shell, quote the value so the semicolons are not interpreted by the shell.

  ```bash
  cmake -B build -S . ... -DGPU_TARGETS="gfx90a;gfx942;gfx950"
  ```

Supported names are defined in `cmake/tensilelite_supported_architectures.cmake` (e.g. gfx908, gfx90a, gfx942, gfx950, gfx1100–gfx1201, and some `:xnack+`/`:xnack-` variants). If you pass an unsupported name, CMake will error during configure.

---

## Device libraries (Tensile): required for matmul tests

### Why can't I run the tests?

**Yes: you need the Tensile kernels (device libraries) built and available.**  
Most hipblaslt-test cases are matmul tests. At runtime, the library loads GEMM kernels from a **Tensile device library**: a `.dat` index file plus architecture-specific `.hsaco` (or `.co`) code objects. If that library isn’t present for your GPU, matmul tests fail with errors about a missing Tensile library or missing `.dat`/`.hsaco` files. So to run the tests you must either:

1. **Build the device libraries** in this repo (target `tensilelite-device-libraries`), which runs the Tensile generator and fills `build/Tensile/library/` with the right files for your `GPU_TARGETS`, or  
2. **Use an existing ROCm install** that already has those files and set `HIPBLASLT_TENSILE_LIBPATH` to point at them.

Until one of those is done, matmul-based tests will not run successfully.

### Python environment for building device libraries

The **tensilelite-device-libraries** target runs a Python script (`Tensile.TensileCreateLibrary`) that needs several packages (PyYAML, msgpack, etc.). The build does **not** install them for you; it uses whatever Python CMake found and only sets `PYTHONPATH` so it can find the in-repo `Tensile` package and the built **rocisa** module. So you need a Python environment that has the dependencies installed.

**Recommended: use a venv and install the Tensile requirements.**

- **Requirements file:** `tensilelite/requirements.txt`  
  It lists the packages needed for the Tensile Python side (e.g. `packaging`, `pyyaml`, `msgpack`, `joblib`, `simplejson`, `ujson`, `orjson`, `yappi`). The **same** Python is used to run `hipblaslt_gentest.py`, which generates `hipblaslt_gtest.data` from the test YAML — so if PyYAML isn’t installed, that step can fail and the data file will never appear. Using a venv and this requirements file covers both the device-library build and the test-data generator. Optional dev deps are in `tensilelite/requirements-dev.txt` (includes `rocisa` for local dev; the build builds rocisa via CMake, so you don’t need to pip-install it for the device-library build).

**Setup:**

```bash
cd projects/hipblaslt
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS; on Windows: .venv\Scripts\activate
pip install -r tensilelite/requirements.txt
```

Then configure CMake so it uses this Python when running the device-library step:

```bash
cmake -B build -S . \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python" \
  -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" \
  ... # other flags (CMAKE_BUILD_TYPE, compiler, GPU_TARGETS, etc.)
```

If you don’t use a venv, the device-library build will still run with whatever `python3` CMake finds; if that environment is missing `pyyaml` or `msgpack`, the build will fail with import errors. Using a venv and `requirements.txt` avoids that.

**Minimal steps to get tests running (from `projects/hipblaslt`):**

```bash
# 0. Python env: create a venv and install Tensile Python dependencies (see below)
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r tensilelite/requirements.txt

# 1. Configure CMake to use that Python (so the device-library build sees the deps)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx90a \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python" -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python"

# 2. Build the Tensile device libraries for your GPU (slow, one-time per config)
cmake --build build --target tensilelite-device-libraries

# 3. Build the test binary and generate hipblaslt_gtest.data (runs hipblaslt_gentest.py)
cmake --build build --target hipblaslt-test

# 4. Run tests from the clients directory (so data file is found)
cd build/clients && ./hipblaslt-test
```

If step 2 fails (e.g. missing Python modules), see [Python environment for building device libraries](#python-environment-for-building-device-libraries) and [Building the device libraries](#building-the-device-libraries) below.

---

The runtime loads **Tensile device libraries** (GEMM kernel libraries) from a directory that must contain:

- `TensileLibrary_lazy_<arch>.dat` (e.g. `TensileLibrary_lazy_gfx90a.dat`)
- Corresponding `.hsaco` (or `.co`) code object files for that architecture

If these are missing, matmul tests will fail with errors about a missing Tensile library `.dat` file or missing `.hsaco`/`.co` files.

### How the runtime finds the library

- If **`HIPBLASLT_TENSILE_LIBPATH`** is set, that directory is used (and should contain a `library` subdir with the `.dat` and code objects, or the path is the one that directly contains the `.dat`—see code in `library/src/amd_detail/rocblaslt/src/tensile_host.cpp`).
- If unset, the library looks relative to `librocblaslt.so`: it checks `{lib_dir}/hipblaslt/library`, then `{lib_dir}/../Tensile/library`, then `{lib_dir}/library`. So when running from a **build tree**, it typically resolves to `build/Tensile/library`.

### Building the device libraries

The device libraries are produced by the **tensilelite-device-libraries** target, which runs the TensileCreateLibrary Python workflow. You must build this target so that `build/Tensile/library/` (or your configured output) is populated.

- **Full build:** `cmake --build build --parallel` should build the default target, which includes `tensilelite-device-libraries`. If you previously built without that target (e.g. only built `hipblaslt-test`), the Tensile directory may contain only a stray file or two.
- **Build device libraries explicitly:**

```bash
cd projects/hipblaslt/build
make tensilelite-device-libraries
# or: cmake --build build --target tensilelite-device-libraries
```

This can be slow (it runs TensileCreateLibrary for all configured `GPU_TARGETS`). When it completes, `build/Tensile/library/` should contain `TensileLibrary_lazy_<arch>.dat` and the corresponding code objects for each architecture you build for.

- If the target fails (e.g. Python/Tensile dependency issues), fix the environment and rebuild. Without a successful run, matmul tests will not have the required `.dat` and kernels.

### Using an existing ROCm installation

If you have a full ROCm installation that already includes hipBLASLt device libraries (e.g. under `/opt/rocm`), you can point the test run at that directory instead of building device libraries yourself:

```bash
export HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/hipblaslt
# Or the directory that directly contains TensileLibrary_lazy_<arch>.dat
./build/clients/hipblaslt-test --gtest_filter=*your_test*
```

The exact path may vary by ROCm version; check your install for `TensileLibrary_lazy_*.dat` and set `HIPBLASLT_TENSILE_LIBPATH` to that directory.

---

## Running the client tests (hipblaslt-test)

### Where does `hipblaslt_gtest.data` come from?

**The file is generated at build time** — it is not checked into the repo. A CMake custom command runs `clients/tests/hipblaslt_gentest.py` (Python, needs PyYAML) to expand the YAML in `clients/tests/data/*.yaml` into the binary data file `build/clients/hipblaslt_gtest.data`. That command is run when you build something that depends on **hipblaslt-test-data**, such as **hipblaslt-test**.

So if you only built `tensilelite-device-libraries` and never built the test executable, the data file will not exist. To create it:

```bash
cmake --build build --target hipblaslt-test
# or just the data file:
cmake --build build --target hipblaslt-test-data
```

The same Python used at configure time (e.g. your venv) is used for this step. **If PyYAML (or the venv) wasn’t installed, this command can fail** and the `.data` file will never be created — so you may see “gtest.data doesn’t exist” even though the build ran. Use the same venv and `pip install -r tensilelite/requirements.txt` (which includes `pyyaml`) so both the Tensile device-library step and the test-data generator have the deps they need.

- **Run from the directory that contains both the test binary and the test data file.**  
  The test executable looks for `hipblaslt_gtest.data` in the same directory as the executable (`hipblaslt_exepath()` + `"hipblaslt_gtest.data"`). The build generates `hipblaslt_gtest.data` under the clients build dir (e.g. `build/clients/hipblaslt_gtest.data`).

Typical usage:

```bash
cd projects/hipblaslt/build/clients
./hipblaslt-test
```

To run a subset of tests:

```bash
./hipblaslt-test --gtest_filter=*quick*
```

If you run from another directory, the test may not find `hipblaslt_gtest.data` and will fail or behave incorrectly.

---

## Test data and adding/changing tests

- **YAML data:** Test cases are driven by YAML under `clients/tests/data/` (e.g. `matmul_gtest.yaml`, `hipblaslt_common.yaml`). The build process generates `hipblaslt_gtest.data` from these.
- **Test coverage:** For a high-level overview of what the tests cover and notable gaps (e.g. special floating-point values, initialization types), see `docs/TEST_COVERAGE_OVERVIEW.md`.
- **Known bugs:** To skip tests on specific platforms (e.g. a known bug on one GPU architecture), add an entry in `clients/tests/data/known_bugs.yaml`. The top-level key is `Known bugs:` and each entry can specify `function`, `initialization`, `known_bug_platforms`, etc., so that the test is skipped on those platforms until the bug is fixed.
- After editing YAML under `clients/tests/data/`, reconfigure/build so that `hipblaslt_gtest.data` is regenerated.

---

## Summary checklist for a working dev/test loop

1. **Configure and build from `projects/hipblaslt`** (not repo root).
2. **Build device libraries** so `build/Tensile/library/` has `TensileLibrary_lazy_<arch>.dat` and code objects for your GPU, or set `HIPBLASLT_TENSILE_LIBPATH` to an existing install.
3. **Run tests from `build/clients/`** so `hipblaslt_gtest.data` is next to `hipblaslt-test`.
4. Use `--gtest_filter` to run specific tests during development.

---

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| CMake fails from repo root | Configure from `projects/hipblaslt` instead. |
| Missing `TensileLibrary_lazy_*.dat` or `.hsaco` | Build `tensilelite-device-libraries` or set `HIPBLASLT_TENSILE_LIBPATH` to a directory that has them. |
| TensileCreateLibrary fails with `ModuleNotFoundError` (e.g. `yaml`, `msgpack`) | Use a Python venv, `pip install -r tensilelite/requirements.txt`, and configure with `-DPython_EXECUTABLE=/path/to/venv/bin/python` (see [Python environment for building device libraries](#python-environment-for-building-device-libraries)). |
| `hipblaslt_gtest.data` doesn't exist | Build the test (or the data target): `cmake --build build --target hipblaslt-test` or `--target hipblaslt-test-data`. If the generator step failed earlier (e.g. missing PyYAML), the file was never created — use the same venv and `pip install -r tensilelite/requirements.txt` (includes `pyyaml`), then rebuild. See [Where does hipblaslt_gtest.data come from?](#where-does-hipblaslt_gtestdata-come-from). |
| Test can't find test data | Run from `build/clients/` (same directory as `hipblaslt-test` and `hipblaslt_gtest.data`). |
| Test skipped on a GPU you care about | Check `known_bugs.yaml` for an entry that matches your test and `known_bug_platforms`. |

---

*This file is a living document. Add new findings here as you run into them.*
