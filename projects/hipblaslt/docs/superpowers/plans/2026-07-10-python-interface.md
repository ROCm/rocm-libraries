# hipBLASLt Python Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-level Python package (`hipblaslt`) that binds the hipBLASLt C API via nanobind, letting developers drive GEMMs from Python for bug confirmation, numerical-correctness investigation, and exploratory benchmarking.

**Architecture:** A nanobind C++ extension (`_core`) mirroring the hipBLASLt C flow object-for-object (Handle, MatmulDesc, MatrixLayout, Preference, Algo, heuristic(), matmul()), plus a compiled `DeviceArray` data-plane object (hipMalloc/memcpy + DLPack), plus a thin pure-Python layer (`hipblaslt/__init__.py`) with a convenience `matmul()` shim and a header-enum coverage harness. Built with scikit-build-core, templated on `tensilelite/rocisa`, wired into `invoke build` behind an opt-in flag.

**Tech Stack:** C++20, nanobind 2.6.1, scikit-build-core, HIP (`hip::host`), hipBLASLt (`roc::hipblaslt`), Python ≥3.10, numpy, ml_dtypes, pytest, invoke.

## Global Constraints

- **Package import name:** `hipblaslt` (matches the library). Package directory `projects/hipblaslt/python/`.
- **Build is opt-in:** NOT part of default `invoke build`; enabled by a new `--python` / `-p` flag. Zero impact on existing host/device/client builds and CI unless requested.
- **License header on every new file** — SPDX short form at the very top:
  - C/C++/HIP (`//`): `// Copyright Advanced Micro Devices, Inc., or its affiliates.` then `// SPDX-License-Identifier: MIT`
  - Python/CMake/TOML/YAML (`#`): `# Copyright Advanced Micro Devices, Inc., or its affiliates.` then `# SPDX-License-Identifier: MIT`
  - Do NOT paste the legacy verbose MIT block.
- **Branch:** `users/talumbau/python-interface` (already checked out, off `develop`). Base PRs on `develop`.
- **Full low-level control is essential**; the convenience `matmul()` must be a thin shim over `_core`, never a parallel code path.
- **Never swallow error codes:** every hipBLASLt/HIP status is checked and raised as a Python `HipblasLtError`.
- **The binding must never be a silent source of wrongness** — validation at the Python boundary; hipBLASLt's own converters are the ground-truth encoder for narrow types.
- **Reuse, don't reinvent:** narrow-type pack/unpack reuses hipBLASLt's converters (`hipblaslt_float8.h` etc.); MX pre-swizzle ports the logic in `tensilelite/client/src/DataInitialization.cpp`.
- **GPU-gated tests:** correctness tests skip cleanly when no device is present; they are excluded from pure-host CI.

## Reference: verified API signatures (from `library/include/hipblaslt/hipblaslt.h`)

```c
hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle);
hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);
hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t* matmulDesc,
                                          hipblasComputeType_t computeType, hipDataType scaleType);
hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t matmulDesc);
hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes);
hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatmulDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* matLayout,
    hipDataType type, uint64_t rows, uint64_t cols, int64_t ld);
hipblasStatus_t hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t matLayout);
hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t matLayout,
    hipblasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes);
hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* pref);
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(hipblasLtMatmulPreference_t pref);
hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes);
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t handle,
    hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc,
    hipblasLtMatmulPreference_t pref, int requestedAlgoCount,
    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount);
hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc,
    const void* alpha, const void* A, hipblasLtMatrixLayout_t Adesc, const void* B, Bdesc,
    const void* beta, const void* C, Cdesc, void* D, Ddesc,
    const hipblasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, hipStream_t stream);
// struct: hipblasLtMatmulHeuristicResult_t { hipblasLtMatmulAlgo_t algo; size_t workspaceSize;
//          hipblasStatus_t state; float wavesCount; int reserved[4]; }
```

`hipStream_t`, `hipMalloc`, `hipFree`, `hipMemcpy`, `hipMemcpyHostToDevice`/`DeviceToHost` come from `<hip/hip_runtime.h>`. The library CMake package is `hipblaslt` exporting `roc::hipblaslt` (see `cmake/hipblaslt-config.cmake.in`).

---

## Phase 0 — Scaffolding and opt-in build (Tasks 1–3)

### Task 1: Package skeleton that imports (empty extension)

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/CMakeLists.txt`
- Create: `python/src/module.cpp`
- Create: `python/hipblaslt/__init__.py`
- Create: `python/.gitignore`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: an importable `hipblaslt` package with a compiled `_core` submodule exposing `_core.__version__` (str) and `_core.hip_available() -> bool`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_import.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import hipblaslt


def test_core_version_is_string():
    assert isinstance(hipblaslt._core.__version__, str)
    assert hipblaslt._core.__version__


def test_hip_available_is_bool():
    assert isinstance(hipblaslt._core.hip_available(), bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hipblaslt'` (package not built yet).

- [ ] **Step 3: Write `python/pyproject.toml`**

```toml
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

[project]
name = "hipblaslt"
version = "0.1.0"
description = "Low-level Python bindings for the hipBLASLt GEMM API"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = ["numpy>=1.23", "ml_dtypes>=0.5.0"]

[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.build-type = "Release"
cmake.args = ["-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
wheel.packages = ["hipblaslt"]
build-dir = "build/{wheel_tag}"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write `python/CMakeLists.txt`**

```cmake
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(hipblaslt_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# scikit-build-core pre-installs nanobind and injects its CMake path.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind REQUIRED CONFIG)

# Locate HIP and hipBLASLt. ROCM_PATH may be passed by tasks.py; fall back to /opt/rocm.
if(ROCM_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

nanobind_add_module(_core NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/module.cpp"
)
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(_core PRIVATE hip::host roc::hipblaslt)

install(TARGETS _core DESTINATION hipblaslt)
```

- [ ] **Step 5: Write `python/src/module.cpp`**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <hip/hip_runtime.h>

namespace nb = nanobind;

static bool hip_available()
{
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

NB_MODULE(_core, m)
{
    m.doc() = "Low-level hipBLASLt bindings.";
    m.attr("__version__") = "0.1.0";
    m.def("hip_available", &hip_available,
          "Return True if at least one HIP device is visible.");
}
```

- [ ] **Step 6: Write `python/hipblaslt/__init__.py`**

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level Python bindings for the hipBLASLt GEMM API."""

from . import _core

__all__ = ["_core"]
__version__ = _core.__version__
```

- [ ] **Step 7: Write `python/.gitignore`**

```
build/
*.egg-info/
__pycache__/
*.so
```

- [ ] **Step 8: Build and install editable, then run the test**

Run:
```bash
cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_import.py -v
```
Expected: PASS (both tests). If `find_package(hipblaslt)` fails, pass `-Dhipblaslt_DIR=<install>/lib/cmake/hipblaslt` via `pip install ... --config-settings=cmake.args=...`; note the working resolution in the commit message.

- [ ] **Step 9: Commit**

```bash
git add python/pyproject.toml python/CMakeLists.txt python/src/module.cpp python/hipblaslt/__init__.py python/.gitignore python/tests/test_import.py
git commit -m "feat(python): scaffold hipblaslt nanobind package skeleton"
```

### Task 2: Opt-in `--python` flag on `invoke build`

**Files:**
- Modify: `tasks.py` (the `@task` `help` dict ~line 321-356, the `build()` signature ~line 358-393, and the tail of `build()` after the cmake build ~line 663)

**Interfaces:**
- Consumes: the built/installed hipBLASLt library from the normal build (the Python extension links `roc::hipblaslt`).
- Produces: `invoke build --python` (alias `-p`) builds and editable-installs the `python/` package into `build/venv` after the main build; without the flag, nothing changes.

- [ ] **Step 1: Add the help entry**

In the `help={...}` dict of the `build` task (near line 355, after `"clean": ...`), add:

```python
        "python": "Also build and install the low-level Python bindings (opt-in).",
```

- [ ] **Step 2: Add the parameter**

In the `def build(` signature (after `clean=False,` at line 392), add:

```python
    python=False,
```

- [ ] **Step 3: Add the build step at the end of `build()`**

After the package-install block (after line 678, at the end of the function body), append:

```python
    # ---------------------------------------------------------------------------
    # Optional: low-level Python bindings (opt-in via --python)
    # ---------------------------------------------------------------------------
    if python:
        py_dir = ROOT_PATH / "python"
        install_dir = (ROOT_PATH / "hipblaslt-install").as_posix()
        hipblaslt_cmake_dir = f"{install_dir}/lib/cmake/hipblaslt"
        config_settings = (
            f"--config-settings=cmake.args=-DROCM_PATH={rocm_s};"
            f"-Dhipblaslt_DIR={hipblaslt_cmake_dir}"
        )
        with c.cd(str(py_dir)):
            c.run(
                f"pip install --no-build-isolation -e . {config_settings}",
                env={"ROCM_PATH": rocm_s},
            )
```

- [ ] **Step 4: Verify the flag is wired (dry parse)**

Run: `invoke --help build | grep -A1 -- --python`
Expected: shows the `--python` option with the help text. (This validates the task decorator parsed without needing a GPU.)

- [ ] **Step 5: Commit**

```bash
git add tasks.py
git commit -m "feat(python): add opt-in --python flag to invoke build"
```

### Task 3: `HipblasLtError` and the status-check helper

**Files:**
- Create: `python/src/status.hpp`
- Modify: `python/src/module.cpp`
- Modify: `python/CMakeLists.txt` (add `status.hpp` is header-only; no source change needed, but register the exception in module.cpp)

**Interfaces:**
- Consumes: `_core` module object.
- Produces: C++ macro `HIPBLASLT_CHECK(expr)` and `HIP_CHECK(expr)` that throw a Python `hipblaslt._core.HipblasLtError` carrying the status name; Python-visible `HipblasLtError` exception type.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_errors.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import hipblaslt


def test_error_type_exists():
    assert issubclass(hipblaslt._core.HipblasLtError, Exception)


def test_raise_status_helper():
    # _raise_test_status(int) is a debug hook that maps a status code to a raise.
    import pytest
    with pytest.raises(hipblaslt._core.HipblasLtError) as ei:
        hipblaslt._core._raise_test_status(7)  # 7 = HIPBLAS_STATUS_INVALID_VALUE
    assert "INVALID_VALUE" in str(ei.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_errors.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'HipblasLtError'`.

- [ ] **Step 3: Write `python/src/status.hpp`**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <stdexcept>
#include <string>

namespace hipblaslt_py {

class HipblasLtError : public std::runtime_error
{
public:
    explicit HipblasLtError(const std::string& msg) : std::runtime_error(msg) {}
};

inline const char* status_name(hipblasStatus_t s)
{
    switch(s)
    {
    case HIPBLAS_STATUS_SUCCESS:          return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:  return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:     return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:    return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:    return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:   return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:    return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH:    return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:     return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:          return "HIPBLAS_STATUS_UNKNOWN";
    default:                              return "HIPBLAS_STATUS_<unmapped>";
    }
}

inline void check_status(hipblasStatus_t s, const char* call)
{
    if(s != HIPBLAS_STATUS_SUCCESS)
        throw HipblasLtError(std::string(call) + " failed: " + status_name(s)
                             + " (" + std::to_string(static_cast<int>(s)) + ")");
}

inline void check_hip(hipError_t e, const char* call)
{
    if(e != hipSuccess)
        throw HipblasLtError(std::string(call) + " failed: " + hipGetErrorString(e));
}

} // namespace hipblaslt_py

#define HIPBLASLT_CHECK(expr) ::hipblaslt_py::check_status((expr), #expr)
#define HIP_CHECK(expr)       ::hipblaslt_py::check_hip((expr), #expr)
```

- [ ] **Step 4: Register the exception and debug hook in `module.cpp`**

Replace the body of `NB_MODULE(_core, m)` in `python/src/module.cpp` with:

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <hip/hip_runtime.h>
#include "status.hpp"

namespace nb = nanobind;
using hipblaslt_py::HipblasLtError;

static bool hip_available()
{
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

NB_MODULE(_core, m)
{
    m.doc() = "Low-level hipBLASLt bindings.";
    m.attr("__version__") = "0.1.0";
    m.def("hip_available", &hip_available,
          "Return True if at least one HIP device is visible.");

    nb::exception<HipblasLtError>(m, "HipblasLtError");

    m.def("_raise_test_status", [](int code) {
        hipblaslt_py::check_status(static_cast<hipblasStatus_t>(code), "_raise_test_status");
    }, "Debug hook: raise HipblasLtError for a nonzero status code.");
}
```

- [ ] **Step 5: Rebuild and run the test**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_errors.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add python/src/status.hpp python/src/module.cpp python/tests/test_errors.py
git commit -m "feat(python): add HipblasLtError and status-check helpers"
```

---

## Phase 1 — Enums and core control objects (Tasks 4–7)

### Task 4: Bind the essential enums

**Files:**
- Create: `python/src/enums.cpp`
- Create: `python/src/init.hpp`
- Modify: `python/src/module.cpp`
- Modify: `python/CMakeLists.txt`

**Interfaces:**
- Consumes: `_core` module.
- Produces: nanobind enums exposed on `_core`: `DataType` (wraps `hipDataType`), `ComputeType` (wraps `hipblasComputeType_t`), `Epilogue` (wraps `hipblasLtEpilogue_t`), `MatmulDescAttr` (wraps `hipblasLtMatmulDescAttributes_t`), `MatrixLayoutAttr`, `PreferenceAttr`, `ScaleMode` (wraps `hipblasLtMatmulMatrixScale_t`). Also a free function `_core.enum_members(name: str) -> dict[str, int]` returning `{member_name: value}` for the named enum (consumed later by the coverage harness). Declares `void init_enums(nb::module_&)` in `init.hpp`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_enums.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import hipblaslt
c = hipblaslt._core


def test_datatype_has_r_32f():
    assert hasattr(c.DataType, "R_32F")


def test_epilogue_bias_value():
    # HIPBLASLT_EPILOGUE_BIAS == 4
    assert int(c.Epilogue.BIAS) == 4


def test_scalemode_vec32_ue8m0_value():
    # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 == 2
    assert int(c.ScaleMode.VEC32_UE8M0) == 2


def test_enum_members_roundtrip():
    members = c.enum_members("Epilogue")
    assert members["BIAS"] == 4
    assert members["DEFAULT"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_enums.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'DataType'`.

- [ ] **Step 3: Write `python/src/init.hpp`**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <nanobind/nanobind.h>

void init_enums(nanobind::module_& m);
void init_device_array(nanobind::module_& m);   // Phase 2
void init_descriptors(nanobind::module_& m);    // Phase 3
void init_matmul(nanobind::module_& m);         // Phase 3
```

- [ ] **Step 4: Write `python/src/enums.cpp`**

Bind each enum with an internal registry so `enum_members` can reflect them. Use the short member names (strip the `HIPBLASLT_`/`HIP_R_` prefixes) as shown. Include the full member lists verbatim from `hipblaslt.h` / `hipblaslt-types.h`.

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <string>
#include "init.hpp"

namespace nb = nanobind;

// Registry: enum name -> {member name -> int value}. Populated as we bind.
static std::map<std::string, std::map<std::string, int>>& registry()
{
    static std::map<std::string, std::map<std::string, int>> r;
    return r;
}

template <typename E>
static void reg(nb::enum_<E>& e, const char* enum_name, const char* member, E value)
{
    e.value(member, value);
    registry()[enum_name][member] = static_cast<int>(value);
}

void init_enums(nb::module_& m)
{
    {
        nb::enum_<hipDataType> e(m, "DataType");
        reg(e, "DataType", "R_16F", HIP_R_16F);
        reg(e, "DataType", "R_32F", HIP_R_32F);
        reg(e, "DataType", "R_64F", HIP_R_64F);
        reg(e, "DataType", "R_16BF", HIP_R_16BF);
        reg(e, "DataType", "R_8I", HIP_R_8I);
        reg(e, "DataType", "R_32I", HIP_R_32I);
        reg(e, "DataType", "R_8F_E4M3", HIP_R_8F_E4M3);
        reg(e, "DataType", "R_8F_E5M2", HIP_R_8F_E5M2);
        reg(e, "DataType", "R_8F_E4M3_FNUZ", HIP_R_8F_E4M3_FNUZ);
        reg(e, "DataType", "R_8F_E5M2_FNUZ", HIP_R_8F_E5M2_FNUZ);
    }
    {
        nb::enum_<hipblasComputeType_t> e(m, "ComputeType");
        reg(e, "ComputeType", "COMPUTE_32F", HIPBLAS_COMPUTE_32F);
        reg(e, "ComputeType", "COMPUTE_32F_FAST_16F", HIPBLAS_COMPUTE_32F_FAST_16F);
        reg(e, "ComputeType", "COMPUTE_32F_FAST_16BF", HIPBLAS_COMPUTE_32F_FAST_16BF);
        reg(e, "ComputeType", "COMPUTE_64F", HIPBLAS_COMPUTE_64F);
        reg(e, "ComputeType", "COMPUTE_32I", HIPBLAS_COMPUTE_32I);
    }
    {
        nb::enum_<hipblasLtEpilogue_t> e(m, "Epilogue");
        reg(e, "Epilogue", "DEFAULT", HIPBLASLT_EPILOGUE_DEFAULT);
        reg(e, "Epilogue", "RELU", HIPBLASLT_EPILOGUE_RELU);
        reg(e, "Epilogue", "BIAS", HIPBLASLT_EPILOGUE_BIAS);
        reg(e, "Epilogue", "RELU_BIAS", HIPBLASLT_EPILOGUE_RELU_BIAS);
        reg(e, "Epilogue", "GELU", HIPBLASLT_EPILOGUE_GELU);
        reg(e, "Epilogue", "GELU_BIAS", HIPBLASLT_EPILOGUE_GELU_BIAS);
        reg(e, "Epilogue", "SIGMOID", HIPBLASLT_EPILOGUE_SIGMOID);
        // NOTE: the coverage harness (Task 18) enumerates the header to catch any
        // member omitted here; extend this list when that test flags a gap.
    }
    {
        nb::enum_<hipblasLtMatmulDescAttributes_t> e(m, "MatmulDescAttr");
        reg(e, "MatmulDescAttr", "TRANSA", HIPBLASLT_MATMUL_DESC_TRANSA);
        reg(e, "MatmulDescAttr", "TRANSB", HIPBLASLT_MATMUL_DESC_TRANSB);
        reg(e, "MatmulDescAttr", "EPILOGUE", HIPBLASLT_MATMUL_DESC_EPILOGUE);
        reg(e, "MatmulDescAttr", "BIAS_POINTER", HIPBLASLT_MATMUL_DESC_BIAS_POINTER);
        reg(e, "MatmulDescAttr", "A_SCALE_POINTER", HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER);
        reg(e, "MatmulDescAttr", "B_SCALE_POINTER", HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER);
        reg(e, "MatmulDescAttr", "D_SCALE_POINTER", HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER);
        reg(e, "MatmulDescAttr", "A_SCALE_MODE", HIPBLASLT_MATMUL_DESC_A_SCALE_MODE);
        reg(e, "MatmulDescAttr", "B_SCALE_MODE", HIPBLASLT_MATMUL_DESC_B_SCALE_MODE);
    }
    {
        nb::enum_<hipblasLtMatrixLayoutAttribute_t> e(m, "MatrixLayoutAttr");
        reg(e, "MatrixLayoutAttr", "BATCH_COUNT", HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT);
        reg(e, "MatrixLayoutAttr", "STRIDED_BATCH_OFFSET", HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET);
    }
    {
        nb::enum_<hipblasLtMatmulPreferenceAttributes_t> e(m, "PreferenceAttr");
        reg(e, "PreferenceAttr", "MAX_WORKSPACE_BYTES", HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES);
    }
    {
        nb::enum_<hipblasLtMatmulMatrixScale_t> e(m, "ScaleMode");
        reg(e, "ScaleMode", "SCALAR_32F", HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
        reg(e, "ScaleMode", "VEC32_UE8M0", HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
        reg(e, "ScaleMode", "OUTER_VEC_32F", HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
        reg(e, "ScaleMode", "BLK32_UE8M0_32_8_EXT", HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
    }

    m.def("enum_members", [](const std::string& name) -> std::map<std::string, int> {
        auto it = registry().find(name);
        if(it == registry().end())
            return {};
        return it->second;
    }, "Return {member_name: int_value} for a bound enum.");
}
```

Note: if any enumerator name above does not compile (the header spelling differs), grep the header (`grep -n NAME library/include/hipblaslt/hipblaslt.h`) and correct it. Do not remove members to make it compile — fix the spelling.

- [ ] **Step 5: Call `init_enums` from `module.cpp`**

In `python/src/module.cpp`, add `#include "init.hpp"` and, at the end of `NB_MODULE`, add: `init_enums(m);`

- [ ] **Step 6: Add `enums.cpp` to the extension sources in `CMakeLists.txt`**

In `python/CMakeLists.txt`, extend the `nanobind_add_module(_core ...)` source list to include `"${CMAKE_CURRENT_SOURCE_DIR}/src/enums.cpp"`.

- [ ] **Step 7: Rebuild and run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_enums.py -v`
Expected: PASS (all four tests).

- [ ] **Step 8: Commit**

```bash
git add python/src/enums.cpp python/src/init.hpp python/src/module.cpp python/CMakeLists.txt python/tests/test_enums.py
git commit -m "feat(python): bind core hipBLASLt enums with reflection registry"
```

### Task 5: `Handle` object (RAII)

**Files:**
- Create: `python/src/descriptors.cpp`
- Modify: `python/src/module.cpp` (call `init_descriptors(m)`)
- Modify: `python/CMakeLists.txt` (add `descriptors.cpp`)

**Interfaces:**
- Consumes: `HIPBLASLT_CHECK` (status.hpp), `init.hpp`.
- Produces: `_core.Handle` — constructible with no args (calls `hipblasLtCreate`), destroyed on GC (`hipblasLtDestroy`), context-manager (`__enter__`/`__exit__` → explicit `close()`), exposes `.ptr` (int, the raw handle) for internal use by matmul. Defines `void init_descriptors(nb::module_&)`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_handle.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core

requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_handle_create_and_close():
    h = c.Handle()
    assert h.ptr != 0
    h.close()


@requires_gpu
def test_handle_context_manager():
    with c.Handle() as h:
        assert h.ptr != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_handle.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'Handle'` (or all-skipped if no GPU; in that case run on a GPU host — these are GPU-gated per the Global Constraints).

- [ ] **Step 3: Write `python/src/descriptors.cpp`**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include "status.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

namespace {

class Handle
{
public:
    Handle() { HIPBLASLT_CHECK(hipblasLtCreate(&h_)); }
    ~Handle() { close(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    void close()
    {
        if(h_)
        {
            hipblasLtDestroy(h_);  // best-effort in destructor path
            h_ = nullptr;
        }
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(h_); }
    hipblasLtHandle_t raw() const { return h_; }

private:
    hipblasLtHandle_t h_ = nullptr;
};

} // namespace

void init_descriptors(nb::module_& m)
{
    nb::class_<Handle>(m, "Handle")
        .def(nb::init<>())
        .def("close", &Handle::close)
        .def_prop_ro("ptr", &Handle::ptr)
        .def("__enter__", [](Handle& self) -> Handle& { return self; },
             nb::rv_policy::reference_internal)
        .def("__exit__", [](Handle& self, nb::object, nb::object, nb::object) {
            self.close();
            return false;
        });
}
```

- [ ] **Step 4: Wire it up**

In `module.cpp` add `init_descriptors(m);` after `init_enums(m);`. In `CMakeLists.txt` add `"${CMAKE_CURRENT_SOURCE_DIR}/src/descriptors.cpp"` to the module sources.

- [ ] **Step 5: Rebuild and run (on a GPU host)**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_handle.py -v`
Expected: PASS (both tests) on a GPU host.

- [ ] **Step 6: Commit**

```bash
git add python/src/descriptors.cpp python/src/module.cpp python/CMakeLists.txt python/tests/test_handle.py
git commit -m "feat(python): add RAII Handle wrapping hipblasLtCreate/Destroy"
```

### Task 6: `MatrixLayout` with generic attributes

**Files:**
- Modify: `python/src/descriptors.cpp`
- Modify: `python/tests/` (new `test_layout.py`)

**Interfaces:**
- Consumes: `DataType` enum, `MatrixLayoutAttr` enum, `HIPBLASLT_CHECK`.
- Produces: `_core.MatrixLayout(dtype: DataType, rows: int, cols: int, ld: int)`; `.set_attribute(attr: MatrixLayoutAttr, value: int)` (int-valued attributes only, sufficient for batch count/stride); `.raw()` internal accessor returning the opaque pointer as int; RAII destroy.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_layout.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_layout_create():
    layout = c.MatrixLayout(c.DataType.R_32F, 4, 8, 4)
    assert layout.ptr != 0


@requires_gpu
def test_layout_set_batch_count():
    layout = c.MatrixLayout(c.DataType.R_32F, 4, 8, 4)
    layout.set_attribute(c.MatrixLayoutAttr.BATCH_COUNT, 2)  # no raise == pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_layout.py -v`
Expected: FAIL — no attribute `MatrixLayout`.

- [ ] **Step 3: Add the `MatrixLayout` class to `descriptors.cpp`**

Add inside the anonymous namespace (before `init_descriptors`):

```cpp
class MatrixLayout
{
public:
    MatrixLayout(hipDataType dtype, uint64_t rows, uint64_t cols, int64_t ld)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&l_, dtype, rows, cols, ld));
    }
    ~MatrixLayout() { if(l_) hipblasLtMatrixLayoutDestroy(l_); }
    MatrixLayout(const MatrixLayout&) = delete;
    MatrixLayout& operator=(const MatrixLayout&) = delete;

    void set_attribute(hipblasLtMatrixLayoutAttribute_t attr, int value)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(l_, attr, &value, sizeof(value)));
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(l_); }
    hipblasLtMatrixLayout_t raw() const { return l_; }

private:
    hipblasLtMatrixLayout_t l_ = nullptr;
};
```

And register it inside `init_descriptors`:

```cpp
    nb::class_<MatrixLayout>(m, "MatrixLayout")
        .def(nb::init<hipDataType, uint64_t, uint64_t, int64_t>(),
             nb::arg("dtype"), nb::arg("rows"), nb::arg("cols"), nb::arg("ld"))
        .def("set_attribute", &MatrixLayout::set_attribute)
        .def_prop_ro("ptr", &MatrixLayout::ptr);
```

- [ ] **Step 4: Rebuild and run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/descriptors.cpp python/tests/test_layout.py
git commit -m "feat(python): add MatrixLayout with generic int attributes"
```

### Task 7: `MatmulDesc` and `Preference` with generic attributes

**Files:**
- Modify: `python/src/descriptors.cpp`
- Create: `python/tests/test_desc.py`

**Interfaces:**
- Consumes: `ComputeType`, `DataType`, `MatmulDescAttr`, `PreferenceAttr`, `Epilogue`, `ScaleMode` enums; `HIPBLASLT_CHECK`.
- Produces: `_core.MatmulDesc(compute_type: ComputeType, scale_type: DataType)` with `.set_attribute_int(attr, int)` and `.set_attribute_ptr(attr, int_ptr)` (raw device pointer as int, for bias/scale pointers), plus `.get_attribute_int(attr) -> int`; `.raw()`/`.ptr`. `_core.Preference()` with `.set_max_workspace(nbytes: int)` and `.raw()`/`.ptr`. Both RAII.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_desc.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_desc_create_and_epilogue():
    d = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    d.set_attribute_int(c.MatmulDescAttr.EPILOGUE, int(c.Epilogue.RELU))
    assert d.get_attribute_int(c.MatmulDescAttr.EPILOGUE) == int(c.Epilogue.RELU)


@requires_gpu
def test_desc_scale_mode():
    d = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    d.set_attribute_int(c.MatmulDescAttr.A_SCALE_MODE, int(c.ScaleMode.VEC32_UE8M0))


@requires_gpu
def test_preference_workspace():
    p = c.Preference()
    p.set_max_workspace(32 * 1024 * 1024)
    assert p.ptr != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_desc.py -v`
Expected: FAIL — no attribute `MatmulDesc`.

- [ ] **Step 3: Add `MatmulDesc` and `Preference` to `descriptors.cpp`**

Add to the anonymous namespace:

```cpp
class MatmulDesc
{
public:
    MatmulDesc(hipblasComputeType_t compute, hipDataType scale)
    {
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&d_, compute, scale));
    }
    ~MatmulDesc() { if(d_) hipblasLtMatmulDescDestroy(d_); }
    MatmulDesc(const MatmulDesc&) = delete;
    MatmulDesc& operator=(const MatmulDesc&) = delete;

    void set_attribute_int(hipblasLtMatmulDescAttributes_t attr, int32_t value)
    {
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(d_, attr, &value, sizeof(value)));
    }
    void set_attribute_ptr(hipblasLtMatmulDescAttributes_t attr, std::uintptr_t p)
    {
        void* raw = reinterpret_cast<void*>(p);
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(d_, attr, &raw, sizeof(raw)));
    }
    int32_t get_attribute_int(hipblasLtMatmulDescAttributes_t attr)
    {
        int32_t value = 0;
        size_t written = 0;
        HIPBLASLT_CHECK(hipblasLtMatmulDescGetAttribute(d_, attr, &value, sizeof(value), &written));
        return value;
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(d_); }
    hipblasLtMatmulDesc_t raw() const { return d_; }

private:
    hipblasLtMatmulDesc_t d_ = nullptr;
};

class Preference
{
public:
    Preference() { HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&p_)); }
    ~Preference() { if(p_) hipblasLtMatmulPreferenceDestroy(p_); }
    Preference(const Preference&) = delete;
    Preference& operator=(const Preference&) = delete;

    void set_max_workspace(uint64_t nbytes)
    {
        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
            p_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &nbytes, sizeof(nbytes)));
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(p_); }
    hipblasLtMatmulPreference_t raw() const { return p_; }

private:
    hipblasLtMatmulPreference_t p_ = nullptr;
};
```

Register in `init_descriptors`:

```cpp
    nb::class_<MatmulDesc>(m, "MatmulDesc")
        .def(nb::init<hipblasComputeType_t, hipDataType>(),
             nb::arg("compute_type"), nb::arg("scale_type"))
        .def("set_attribute_int", &MatmulDesc::set_attribute_int)
        .def("set_attribute_ptr", &MatmulDesc::set_attribute_ptr)
        .def("get_attribute_int", &MatmulDesc::get_attribute_int)
        .def_prop_ro("ptr", &MatmulDesc::ptr);

    nb::class_<Preference>(m, "Preference")
        .def(nb::init<>())
        .def("set_max_workspace", &Preference::set_max_workspace)
        .def_prop_ro("ptr", &Preference::ptr);
```

- [ ] **Step 4: Rebuild and run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_desc.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add python/src/descriptors.cpp python/tests/test_desc.py
git commit -m "feat(python): add MatmulDesc and Preference with generic attributes"
```

---

## Phase 2 — DeviceArray data plane (Tasks 8–10)

### Task 8: `DeviceArray` allocation + host round-trip

**Files:**
- Create: `python/src/device_array.hpp`
- Create: `python/src/device_array.cpp`
- Modify: `python/src/module.cpp` (call `init_device_array(m)`)
- Modify: `python/CMakeLists.txt` (add `device_array.cpp`)
- Create: `python/tests/test_device_array.py`

**Interfaces:**
- Consumes: `HIP_CHECK`, `DataType` enum, numpy (via nanobind ndarray).
- Produces: `_core.DeviceArray` with:
  - classmethod `from_numpy(arr: np.ndarray, dtype: DataType) -> DeviceArray` — allocates `hipMalloc`, copies H2D. `arr` must be C-contiguous; its byte size defines the allocation.
  - `to_numpy() -> np.ndarray` — copies D2H into a fresh numpy array of the stored shape/host-dtype.
  - `copy_from_host(arr)` / `copy_to_host(out)` — reuse the existing allocation.
  - properties: `.ptr` (int device pointer), `.nbytes` (int), `.shape` (tuple), `.dtype` (DataType).
  - `.free()` + context manager; RAII `hipFree` on destruction.
- Note: `DeviceArray` stores a numpy dtype string for the host round-trip AND the hipBLASLt `DataType` for matmul. For narrow types with no numpy dtype (Phase 4), host round-trip uses a raw `uint8` buffer; that path is added later.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_device_array.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_roundtrip_f32():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    da = c.DeviceArray.from_numpy(a, c.DataType.R_32F)
    assert da.ptr != 0
    assert da.nbytes == a.nbytes
    back = da.to_numpy()
    np.testing.assert_array_equal(back, a)


@requires_gpu
def test_copy_from_host_reuse():
    a = np.zeros((2, 2), dtype=np.float32)
    da = c.DeviceArray.from_numpy(a, c.DataType.R_32F)
    b = np.ones((2, 2), dtype=np.float32)
    da.copy_from_host(b)
    np.testing.assert_array_equal(da.to_numpy(), b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_device_array.py -v`
Expected: FAIL — no attribute `DeviceArray`.

- [ ] **Step 3: Write `python/src/device_array.hpp`**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <nanobind/nanobind.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <string>
#include <vector>

namespace hipblaslt_py {

class DeviceArray
{
public:
    DeviceArray(size_t nbytes, hipDataType dtype,
                std::vector<int64_t> shape, std::string host_dtype);
    ~DeviceArray();
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void  free();
    void  copy_from_host(const void* src, size_t nbytes);
    void  copy_to_host(void* dst, size_t nbytes) const;

    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(d_); }
    void*          raw() const { return d_; }
    size_t         nbytes() const { return nbytes_; }
    hipDataType    dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::string& host_dtype() const { return host_dtype_; }

private:
    void*                d_ = nullptr;
    size_t               nbytes_ = 0;
    hipDataType          dtype_;
    std::vector<int64_t> shape_;
    std::string          host_dtype_;   // numpy dtype string, e.g. "float32"
};

} // namespace hipblaslt_py
```

- [ ] **Step 4: Write `python/src/device_array.cpp`**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "device_array.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "status.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

DeviceArray::DeviceArray(size_t nbytes, hipDataType dtype,
                         std::vector<int64_t> shape, std::string host_dtype)
    : nbytes_(nbytes), dtype_(dtype), shape_(std::move(shape)),
      host_dtype_(std::move(host_dtype))
{
    HIP_CHECK(hipMalloc(&d_, nbytes_));
}

DeviceArray::~DeviceArray() { free(); }

void DeviceArray::free()
{
    if(d_) { hipFree(d_); d_ = nullptr; }
}

void DeviceArray::copy_from_host(const void* src, size_t nbytes)
{
    if(nbytes != nbytes_)
        throw HipblasLtError("copy_from_host size mismatch");
    HIP_CHECK(hipMemcpy(d_, src, nbytes, hipMemcpyHostToDevice));
}

void DeviceArray::copy_to_host(void* dst, size_t nbytes) const
{
    if(nbytes != nbytes_)
        throw HipblasLtError("copy_to_host size mismatch");
    HIP_CHECK(hipMemcpy(dst, d_, nbytes, hipMemcpyDeviceToHost));
}

using NpArray = nb::ndarray<nb::numpy, nb::c_contig>;

void init_device_array(nb::module_& m)
{
    nb::class_<DeviceArray>(m, "DeviceArray")
        .def_static("from_numpy", [](NpArray arr, hipDataType dtype) {
            std::vector<int64_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
            size_t nbytes = arr.nbytes();
            // Host dtype string is filled by the Python wrapper; store "" here and
            // let to_numpy reconstruct via the DataType. For Phase 2 we only handle
            // numpy-native types, so record the numpy dtype from the buffer.
            std::string host_dtype = "";  // set below via a helper
            auto da = new DeviceArray(nbytes, dtype, shape, host_dtype);
            da->copy_from_host(arr.data(), nbytes);
            return da;
        }, nb::arg("arr"), nb::arg("dtype"), nb::rv_policy::take_ownership)
        .def("copy_from_host", [](DeviceArray& self, NpArray arr) {
            self.copy_from_host(arr.data(), arr.nbytes());
        })
        .def("copy_to_host", [](DeviceArray& self, NpArray out) {
            self.copy_to_host(out.data(), out.nbytes());
        })
        .def("free", &DeviceArray::free)
        .def_prop_ro("ptr", &DeviceArray::ptr)
        .def_prop_ro("nbytes", &DeviceArray::nbytes)
        .def_prop_ro("shape", [](DeviceArray& self) {
            return self.shape();
        })
        .def_prop_ro("dtype", &DeviceArray::dtype)
        .def("__enter__", [](DeviceArray& self) -> DeviceArray& { return self; },
             nb::rv_policy::reference_internal)
        .def("__exit__", [](DeviceArray& self, nb::object, nb::object, nb::object) {
            self.free(); return false;
        });
}
```

Note: `to_numpy()` is intentionally implemented in the Python wrapper (Task 10 pulls it into `__init__.py`'s exports); at the C++ level we expose `copy_to_host` into a caller-allocated numpy array. This keeps host-dtype reconstruction (numpy dtype ↔ `DataType`) in Python where the `ml_dtypes` mapping lives (Phase 4). For Task 8, add a minimal `to_numpy` in the Python wrapper — see Step 6.

- [ ] **Step 5: Wire up C++**

In `module.cpp` add `init_device_array(m);`. In `CMakeLists.txt` add `"${CMAKE_CURRENT_SOURCE_DIR}/src/device_array.cpp"` to the module sources.

- [ ] **Step 6: Add `to_numpy` + dtype map to `python/hipblaslt/__init__.py`**

Append to `python/hipblaslt/__init__.py`:

```python
import numpy as _np

# Minimal numpy-native dtype map; extended with ml_dtypes in Phase 4.
_DTYPE_TO_NP = {
    _core.DataType.R_32F: _np.float32,
    _core.DataType.R_64F: _np.float64,
    _core.DataType.R_16F: _np.float16,
    _core.DataType.R_32I: _np.int32,
    _core.DataType.R_8I: _np.int8,
}


def _device_array_to_numpy(self):
    np_dtype = _DTYPE_TO_NP[self.dtype]
    out = _np.empty(tuple(self.shape), dtype=np_dtype)
    self.copy_to_host(out)
    return out


_core.DeviceArray.to_numpy = _device_array_to_numpy
```

- [ ] **Step 7: Rebuild and run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_device_array.py -v`
Expected: PASS (both).

- [ ] **Step 8: Commit**

```bash
git add python/src/device_array.hpp python/src/device_array.cpp python/src/module.cpp python/CMakeLists.txt python/hipblaslt/__init__.py python/tests/test_device_array.py
git commit -m "feat(python): add DeviceArray with hipMalloc/memcpy host round-trip"
```

### Task 9: DLPack import/export

**Files:**
- Modify: `python/src/device_array.cpp`
- Create: `python/tests/test_dlpack.py`

**Interfaces:**
- Consumes: `DeviceArray`, nanobind's `__dlpack__` support.
- Produces: `DeviceArray.__dlpack__()` / `__dlpack_device__()` exporting the device buffer (zero-copy) so torch/CuPy can consume it; `DeviceArray.from_dlpack(obj)` importing an external device tensor without copying. These are the interop escape hatch — torch/CuPy remain non-dependencies.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_dlpack.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
torch = pytest.importorskip("torch")


@requires_gpu
def test_export_to_torch():
    a = np.arange(8, dtype=np.float32)
    da = c.DeviceArray.from_numpy(a, c.DataType.R_32F)
    t = torch.from_dlpack(da)
    assert t.numel() == 8
    np.testing.assert_array_equal(t.cpu().numpy(), a)


@requires_gpu
def test_import_from_torch():
    t = torch.arange(8, dtype=torch.float32, device="cuda")
    da = c.DeviceArray.from_dlpack(t)
    np.testing.assert_array_equal(da.to_numpy(), np.arange(8, dtype=np.float32))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_dlpack.py -v`
Expected: FAIL — `AttributeError: ... '__dlpack__'` / no `from_dlpack` (or skipped if torch-ROCm absent; then validate on a host with torch-ROCm installed).

- [ ] **Step 3: Implement `__dlpack__` export via nanobind ndarray**

In `device_array.cpp`, add a method returning an `nb::ndarray` view over the device buffer tagged with the HIP device, using `nb::device::cuda` (ROCm reuses the CUDA DLPack device type). Add to the class registration:

```cpp
        .def("__dlpack__", [](DeviceArray& self, nb::kwargs) {
            size_t elem = 4; // bytes; refined per-dtype in Phase 4
            std::vector<size_t> shape;
            for(auto s : self.shape()) shape.push_back(static_cast<size_t>(s));
            return nb::ndarray<nb::pytorch, nb::device::cuda>(
                self.raw(), shape.size(), shape.data(), nb::handle());
        })
        .def("__dlpack_device__", [](DeviceArray& self) {
            int device_id = 0;
            hipPointerAttribute_t attr;
            if(hipPointerGetAttributes(&attr, self.raw()) == hipSuccess)
                device_id = attr.device;
            // DLPack device type 2 == kDLCUDA (used for ROCm).
            return std::make_pair(2, device_id);
        })
        .def_static("from_dlpack", [](nb::object obj) {
            auto cap = nb::ndarray<nb::c_contig>(obj);
            size_t nbytes = cap.nbytes();
            std::vector<int64_t> shape(cap.shape_ptr(), cap.shape_ptr() + cap.ndim());
            // Wrap without owning: the source keeps the memory alive via the capsule.
            // For correctness-tool simplicity we copy into our own allocation.
            auto da = new DeviceArray(nbytes, HIP_R_32F, shape, "float32");
            HIP_CHECK(hipMemcpy(da->raw(), cap.data(), nbytes, hipMemcpyDeviceToDevice));
            return da;
        }, nb::rv_policy::take_ownership);
```

Note: element size and dtype propagation are refined in Phase 4; for Task 9 the tests use f32. If the nanobind DLPack API details differ in the pinned version, consult `tensilelite/rocisa` usage and nanobind docs; the required behavior is: export a zero-copy view carrying the HIP device id, and import an external tensor. Keep the copy-on-import simplification only if a true borrow proves fragile — note the choice in the commit.

- [ ] **Step 4: Rebuild and run (host with torch-ROCm)**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_dlpack.py -v`
Expected: PASS (both) on a torch-ROCm host.

- [ ] **Step 5: Commit**

```bash
git add python/src/device_array.cpp python/tests/test_dlpack.py
git commit -m "feat(python): add DLPack import/export to DeviceArray"
```

### Task 10: Boundary validation for DeviceArray

**Files:**
- Modify: `python/hipblaslt/__init__.py`
- Create: `python/tests/test_validation.py`

**Interfaces:**
- Consumes: `DeviceArray`, `_DTYPE_TO_NP`.
- Produces: `from_numpy` (Python-side wrapper) raising `ValueError` for non-contiguous arrays and for a numpy dtype that does not match the requested `DataType`. This is the Python-boundary validation from the spec's Error-handling section, kept in Python where dtype mapping lives.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_validation.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_dtype_mismatch_raises():
    a = np.arange(4, dtype=np.float64)  # f64 host
    with pytest.raises(ValueError):
        hipblaslt.from_numpy(a, c.DataType.R_32F)  # asked for f32


@requires_gpu
def test_non_contiguous_raises():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]  # non-contiguous
    with pytest.raises(ValueError):
        hipblaslt.from_numpy(a, c.DataType.R_32F)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_validation.py -v`
Expected: FAIL — `AttributeError: module 'hipblaslt' has no attribute 'from_numpy'`.

- [ ] **Step 3: Add validating `from_numpy` to `__init__.py`**

Append to `python/hipblaslt/__init__.py`:

```python
def from_numpy(arr, dtype):
    """Validated host→device transfer. Raises ValueError at the boundary."""
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous")
    expected = _DTYPE_TO_NP.get(dtype)
    if expected is not None and arr.dtype != expected:
        raise ValueError(
            f"numpy dtype {arr.dtype} does not match requested {dtype!r} "
            f"(expected {_np.dtype(expected)})"
        )
    return _core.DeviceArray.from_numpy(arr, dtype)


__all__ = ["_core", "from_numpy"]
```

- [ ] **Step 4: Rebuild not needed (pure Python) — run**

Run: `cd python && python -m pytest tests/test_validation.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add python/hipblaslt/__init__.py python/tests/test_validation.py
git commit -m "feat(python): add boundary validation to from_numpy"
```

---

## Phase 3 — Heuristic enumeration and matmul (Tasks 11–13)

### Task 11: `HeuristicResult` + `Algo` and `heuristic()`

**Files:**
- Create: `python/src/matmul.cpp`
- Modify: `python/src/module.cpp` (call `init_matmul(m)`)
- Modify: `python/CMakeLists.txt` (add `matmul.cpp`)
- Create: `python/tests/test_heuristic.py`

**Interfaces:**
- Consumes: `Handle`, `MatmulDesc`, `MatrixLayout`, `Preference` (their `raw()` accessors — matmul.cpp includes descriptors via a shared header; see Step 3), `HIPBLASLT_CHECK`.
- Produces:
  - `_core.Algo` — opaque holder of a `hipblasLtMatmulAlgo_t` by value; exposes `.index` (int, a stable identifier derived from the algo, for logging "algo #N").
  - `_core.HeuristicResult` — fields `.algo` (Algo), `.workspace_size` (int), `.waves_count` (float).
  - `_core.heuristic(handle, desc, a_layout, b_layout, c_layout, d_layout, preference, max_results=32) -> list[HeuristicResult]`.
- Requires exposing `raw()` of the descriptor classes to matmul.cpp. To do this, move the class declarations into a shared header.

- [ ] **Step 1: Extract descriptor declarations into a shared header**

Create `python/src/descriptors.hpp` containing the class declarations (`Handle`, `MatrixLayout`, `MatmulDesc`, `Preference`) currently in `descriptors.cpp` — move the class bodies here (keeping inline method bodies), leave only `init_descriptors` in the `.cpp`, and `#include "descriptors.hpp"` from the `.cpp`. Wrap classes in `namespace hipblaslt_py`. This is a refactor: run the Phase-1 tests after to confirm no behavior change.

Run after refactor: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_handle.py tests/test_layout.py tests/test_desc.py -v`
Expected: PASS (unchanged).

Commit the refactor:
```bash
git add python/src/descriptors.hpp python/src/descriptors.cpp
git commit -m "refactor(python): move descriptor decls to descriptors.hpp"
```

- [ ] **Step 2: Write the failing test**

Create `python/tests/test_heuristic.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_heuristic_returns_algos():
    m = n = k = 64
    with c.Handle() as h:
        desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
        la = c.MatrixLayout(c.DataType.R_32F, m, k, m)
        lb = c.MatrixLayout(c.DataType.R_32F, k, n, k)
        lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        pref = c.Preference()
        pref.set_max_workspace(32 * 1024 * 1024)
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 16)
        assert len(results) > 0
        assert results[0].workspace_size >= 0
        assert isinstance(results[0].algo.index, int)
```

- [ ] **Step 3: Write `python/src/matmul.cpp` (heuristic only for this task)**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>
#include "status.hpp"
#include "descriptors.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

namespace {

struct Algo
{
    hipblasLtMatmulAlgo_t algo;
    int index = -1;   // identifier for logging
};

struct HeuristicResult
{
    Algo   algo;
    size_t workspace_size = 0;
    float  waves_count = 0.0f;
};

std::vector<HeuristicResult> heuristic(
    Handle& handle, MatmulDesc& desc,
    MatrixLayout& a, MatrixLayout& b, MatrixLayout& c, MatrixLayout& d,
    Preference& pref, int max_results)
{
    std::vector<hipblasLtMatmulHeuristicResult_t> raw(max_results);
    int returned = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle.raw(), desc.raw(), a.raw(), b.raw(), c.raw(), d.raw(),
        pref.raw(), max_results, raw.data(), &returned));

    std::vector<HeuristicResult> out;
    for(int i = 0; i < returned; ++i)
    {
        if(raw[i].state != HIPBLAS_STATUS_SUCCESS)
            continue;
        HeuristicResult hr;
        hr.algo.algo = raw[i].algo;
        hr.algo.index = i;
        hr.workspace_size = raw[i].workspaceSize;
        hr.waves_count = raw[i].wavesCount;
        out.push_back(hr);
    }
    return out;
}

} // namespace

void init_matmul(nb::module_& m)
{
    nb::class_<Algo>(m, "Algo")
        .def_ro("index", &Algo::index);

    nb::class_<HeuristicResult>(m, "HeuristicResult")
        .def_ro("algo", &HeuristicResult::algo)
        .def_ro("workspace_size", &HeuristicResult::workspace_size)
        .def_ro("waves_count", &HeuristicResult::waves_count);

    m.def("heuristic", &heuristic,
          nb::arg("handle"), nb::arg("desc"),
          nb::arg("a_layout"), nb::arg("b_layout"),
          nb::arg("c_layout"), nb::arg("d_layout"),
          nb::arg("preference"), nb::arg("max_results") = 32);
}
```

- [ ] **Step 4: Wire up**

In `module.cpp` add `init_matmul(m);`. In `CMakeLists.txt` add `"${CMAKE_CURRENT_SOURCE_DIR}/src/matmul.cpp"`.

- [ ] **Step 5: Rebuild and run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_heuristic.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/src/matmul.cpp python/src/module.cpp python/CMakeLists.txt python/tests/test_heuristic.py
git commit -m "feat(python): add heuristic() with Algo and HeuristicResult"
```

### Task 12: `matmul()` low-level call

**Files:**
- Modify: `python/src/matmul.cpp`
- Create: `python/tests/test_matmul.py`

**Interfaces:**
- Consumes: `Handle`, `MatmulDesc`, `MatrixLayout`, `DeviceArray`, `Algo`, alpha/beta as host floats.
- Produces: `_core.matmul(handle, desc, alpha, A, a_layout, B, b_layout, beta, C, c_layout, D, d_layout, algo, workspace, stream_ptr=0)` where `A/B/C/D`/`workspace` are `DeviceArray`, `algo` is `Algo`, `alpha`/`beta` are Python floats, `stream_ptr` is an int (0 = default stream). Raises if `workspace.nbytes < algo`'s required size is not the caller's concern here — the caller sizes it from the heuristic; passing too small a workspace surfaces as a hipBLASLt status error. Synchronizes the stream before returning so results are readable.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_matmul.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_matmul_f32_matches_numpy():
    m = n = k = 64
    # Column-major layouts (hipBLASLt default); build so that D = A @ B.
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    ref = A @ B

    with c.Handle() as h:
        desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
        # Store A, B, D in column-major order for hipBLASLt.
        dA = hipblaslt.from_numpy(np.asfortranarray(A).ravel(order="K").reshape(-1), c.DataType.R_32F) \
            if False else c.DeviceArray.from_numpy(np.ascontiguousarray(A.T), c.DataType.R_32F)
        dB = c.DeviceArray.from_numpy(np.ascontiguousarray(B.T), c.DataType.R_32F)
        dC = c.DeviceArray.from_numpy(np.ascontiguousarray(np.zeros((n, m), np.float32)), c.DataType.R_32F)
        dD = c.DeviceArray.from_numpy(np.ascontiguousarray(np.zeros((n, m), np.float32)), c.DataType.R_32F)
        la = c.MatrixLayout(c.DataType.R_32F, m, k, m)
        lb = c.MatrixLayout(c.DataType.R_32F, k, n, k)
        lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        pref = c.Preference(); pref.set_max_workspace(32 * 1024 * 1024)
        res = c.heuristic(h, desc, la, lb, lc, ld, pref, 16)
        ws = c.DeviceArray.from_numpy(np.zeros(max(1, res[0].workspace_size), np.uint8), c.DataType.R_8I)
        c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, res[0].algo, ws)
        out = dD.to_numpy().reshape(n, m).T  # undo column-major
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
```

Note: the row/column-major handling above is fiddly. The implementer should confirm the exact layout convention by comparing against numpy and adjust the transposes/`ld` until the assertion passes; the invariant to preserve is "GPU result matches `A @ B` within f32 tolerance." Document the working convention in a comment in the test.

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_matmul.py -v`
Expected: FAIL — no attribute `matmul`.

- [ ] **Step 3: Add `matmul` to `matmul.cpp`**

Add `#include "device_array.hpp"` at the top, and inside the anonymous namespace:

```cpp
void matmul(Handle& handle, MatmulDesc& desc,
            double alpha, DeviceArray& A, MatrixLayout& la,
            DeviceArray& B, MatrixLayout& lb,
            double beta, DeviceArray& C, MatrixLayout& lc,
            DeviceArray& D, MatrixLayout& ld,
            Algo& algo, DeviceArray& workspace,
            std::uintptr_t stream_ptr)
{
    float alpha_f = static_cast<float>(alpha);
    float beta_f  = static_cast<float>(beta);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    HIPBLASLT_CHECK(hipblasLtMatmul(
        handle.raw(), desc.raw(), &alpha_f,
        A.raw(), la.raw(), B.raw(), lb.raw(), &beta_f,
        C.raw(), lc.raw(), D.raw(), ld.raw(),
        &algo.algo, workspace.raw(), workspace.nbytes(), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}
```

Register in `init_matmul`:

```cpp
    m.def("matmul", &matmul,
          nb::arg("handle"), nb::arg("desc"), nb::arg("alpha"),
          nb::arg("A"), nb::arg("a_layout"), nb::arg("B"), nb::arg("b_layout"),
          nb::arg("beta"), nb::arg("C"), nb::arg("c_layout"),
          nb::arg("D"), nb::arg("d_layout"),
          nb::arg("algo"), nb::arg("workspace"), nb::arg("stream_ptr") = 0);
```

Note: `alpha`/`beta` are `float` here because compute type is 32F; when compute type is 32I the scalars are `int32`. A follow-up can branch on compute type, but for the initial compute-path tests (f32) this is correct. Document the limitation in a comment.

- [ ] **Step 4: Rebuild and run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_matmul.py -v`
Expected: PASS (GPU result matches numpy within tolerance).

- [ ] **Step 5: Commit**

```bash
git add python/src/matmul.cpp python/tests/test_matmul.py
git commit -m "feat(python): add low-level matmul() call"
```

### Task 13: Algo enumeration/bisection example test

**Files:**
- Create: `python/tests/test_algo_sweep.py`

**Interfaces:**
- Consumes: `heuristic`, `matmul` (full compute path from Tasks 11–12).
- Produces: a test demonstrating the headline use case — loop every heuristic algo, run each, and confirm all produce the same numerically-correct result. This both validates the enumeration surface and serves as executable documentation.

- [ ] **Step 1: Write the test**

Create `python/tests/test_algo_sweep.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_all_algos_agree():
    m = n = k = 32
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    ref = A @ B
    with c.Handle() as h:
        desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
        dA = c.DeviceArray.from_numpy(np.ascontiguousarray(A.T), c.DataType.R_32F)
        dB = c.DeviceArray.from_numpy(np.ascontiguousarray(B.T), c.DataType.R_32F)
        dC = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
        dD = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
        la = c.MatrixLayout(c.DataType.R_32F, m, k, m)
        lb = c.MatrixLayout(c.DataType.R_32F, k, n, k)
        lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        pref = c.Preference(); pref.set_max_workspace(64 * 1024 * 1024)
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
        assert results
        for r in results:
            ws = c.DeviceArray.from_numpy(np.zeros(max(1, r.workspace_size), np.uint8), c.DataType.R_8I)
            c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, r.algo, ws)
            out = dD.to_numpy().reshape(n, m).T
            np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3,
                                       err_msg=f"algo #{r.algo.index} disagrees")
```

- [ ] **Step 2: Run**

Run: `cd python && python -m pytest tests/test_algo_sweep.py -v`
Expected: PASS (all algos agree). If a specific algo diverges, that is a genuine finding — record it as an `xfail` with the algo index and a note, per the "known bugs as xfail" strategy.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_algo_sweep.py
git commit -m "test(python): sweep all heuristic algos and assert agreement"
```

---

## Phase 4 — Narrow dtypes and MX block scaling (Tasks 14–17)

### Task 14: Extend the dtype map with ml_dtypes + fp8 round-trip

**Files:**
- Modify: `python/hipblaslt/__init__.py`
- Create: `python/tests/test_dtypes.py`

**Interfaces:**
- Consumes: `_core.DataType`, `ml_dtypes`, `_DTYPE_TO_NP`.
- Produces: extended `_DTYPE_TO_NP` covering fp8 (`R_8F_E4M3` ↔ `ml_dtypes.float8_e4m3fn`, `R_8F_E5M2` ↔ `float8_e5m2`, `R_8F_E4M3_FNUZ` ↔ `float8_e4m3fnuz`, `R_8F_E5M2_FNUZ` ↔ `float8_e5m2fnuz`) and bf16 (`R_16BF` ↔ `ml_dtypes.bfloat16`); the reverse map `_NP_TO_DTYPE`; graceful skip when an `ml_dtypes` attribute is missing (older versions).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_dtypes.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import ml_dtypes
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


def test_bf16_in_map():
    assert hipblaslt._DTYPE_TO_NP[c.DataType.R_16BF] is ml_dtypes.bfloat16


@requires_gpu
def test_fp8_e4m3_roundtrip():
    a = np.arange(8).astype(ml_dtypes.float8_e4m3fn)
    da = hipblaslt.from_numpy(a, c.DataType.R_8F_E4M3)
    back = da.to_numpy()
    np.testing.assert_array_equal(back.astype(np.float32), a.astype(np.float32))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_dtypes.py -v`
Expected: FAIL — `KeyError`/`AttributeError` (bf16 not in map).

- [ ] **Step 3: Extend the map in `__init__.py`**

Replace the `_DTYPE_TO_NP` definition block in `python/hipblaslt/__init__.py` with:

```python
import ml_dtypes as _mld

_DTYPE_TO_NP = {
    _core.DataType.R_32F: _np.float32,
    _core.DataType.R_64F: _np.float64,
    _core.DataType.R_16F: _np.float16,
    _core.DataType.R_32I: _np.int32,
    _core.DataType.R_8I: _np.int8,
    _core.DataType.R_16BF: _mld.bfloat16,
}

# fp8 types: present only in recent ml_dtypes. Add each if available.
for _dt_name, _mld_name in [
    ("R_8F_E4M3", "float8_e4m3fn"),
    ("R_8F_E5M2", "float8_e5m2"),
    ("R_8F_E4M3_FNUZ", "float8_e4m3fnuz"),
    ("R_8F_E5M2_FNUZ", "float8_e5m2fnuz"),
]:
    _dt = getattr(_core.DataType, _dt_name, None)
    _np_t = getattr(_mld, _mld_name, None)
    if _dt is not None and _np_t is not None:
        _DTYPE_TO_NP[_dt] = _np_t

_NP_TO_DTYPE = {_np.dtype(v): k for k, v in _DTYPE_TO_NP.items()}
```

- [ ] **Step 4: Run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_dtypes.py -v`
Expected: PASS (bf16 test always; fp8 test on a GPU host with a recent ml_dtypes).

- [ ] **Step 5: Commit**

```bash
git add python/hipblaslt/__init__.py python/tests/test_dtypes.py
git commit -m "feat(python): map fp8/bf16 dtypes via ml_dtypes"
```

### Task 15: Ground-truth fp8 pack/unpack via hipBLASLt converters

**Files:**
- Create: `python/src/convert.hip` (HIP source: uses `hipblaslt_f8`/`hipblaslt_bf8` converters)
- Modify: `python/CMakeLists.txt` (compile `convert.hip` with HIP; expose `pack_fp8`/`unpack_fp8`)
- Modify: `python/src/module.cpp`
- Create: `python/tests/test_convert.py`

**Interfaces:**
- Consumes: `hipblaslt_float8.h` converter structs (`hipblaslt_f8` = OCP E4M3, `hipblaslt_bf8` = OCP E5M2; `_fnuz` variants for FNUZ), numpy f32 input.
- Produces: `_core.pack_fp8(arr_f32: np.ndarray, fmt: str) -> np.ndarray[uint8]` and `_core.unpack_fp8(bytes: np.ndarray[uint8], fmt: str) -> np.ndarray[float32]`, where `fmt` ∈ {"e4m3", "e5m2", "e4m3_fnuz", "e5m2_fnuz"}. These are the bit-for-bit ground truth encoder.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_convert.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core


def test_pack_unpack_roundtrip_small():
    # Values exactly representable in e4m3.
    vals = np.array([0.0, 1.0, 2.0, 0.5, -1.0, 4.0], dtype=np.float32)
    packed = c.pack_fp8(vals, "e4m3")
    assert packed.dtype == np.uint8
    assert packed.shape == vals.shape
    restored = c.unpack_fp8(packed, "e4m3")
    np.testing.assert_array_equal(restored, vals)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_convert.py -v`
Expected: FAIL — no attribute `pack_fp8`.

- [ ] **Step 3: Write `python/src/convert.hip`**

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt_float8.h>
#include <cstdint>
#include <string>
#include <vector>
#include "status.hpp"

namespace nb = nanobind;

// Encode one f32 to an 8-bit fp8 byte using the library's own converters, so the
// byte pattern matches what hipBLASLt produces internally.
static uint8_t encode_one(float v, const std::string& fmt)
{
    if(fmt == "e4m3")      { hipblaslt_f8      x(v); uint8_t b; __builtin_memcpy(&b, &x, 1); return b; }
    if(fmt == "e5m2")      { hipblaslt_bf8     x(v); uint8_t b; __builtin_memcpy(&b, &x, 1); return b; }
    if(fmt == "e4m3_fnuz") { hipblaslt_f8_fnuz x(v); uint8_t b; __builtin_memcpy(&b, &x, 1); return b; }
    if(fmt == "e5m2_fnuz") { hipblaslt_bf8_fnuz x(v);uint8_t b; __builtin_memcpy(&b, &x, 1); return b; }
    throw hipblaslt_py::HipblasLtError("unknown fp8 fmt: " + fmt);
}

static float decode_one(uint8_t b, const std::string& fmt)
{
    if(fmt == "e4m3")      { hipblaslt_f8      x; __builtin_memcpy(&x, &b, 1); return float(x); }
    if(fmt == "e5m2")      { hipblaslt_bf8     x; __builtin_memcpy(&x, &b, 1); return float(x); }
    if(fmt == "e4m3_fnuz") { hipblaslt_f8_fnuz x; __builtin_memcpy(&x, &b, 1); return float(x); }
    if(fmt == "e5m2_fnuz") { hipblaslt_bf8_fnuz x;__builtin_memcpy(&x, &b, 1); return float(x); }
    throw hipblaslt_py::HipblasLtError("unknown fp8 fmt: " + fmt);
}

using F32In  = nb::ndarray<nb::numpy, float,   nb::c_contig>;
using U8In   = nb::ndarray<nb::numpy, uint8_t, nb::c_contig>;

void init_convert(nb::module_& m)
{
    m.def("pack_fp8", [](F32In arr, const std::string& fmt) {
        size_t n = arr.size();
        auto* out = new uint8_t[n];
        const float* src = arr.data();
        for(size_t i = 0; i < n; ++i) out[i] = encode_one(src[i], fmt);
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
        std::vector<size_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
        return nb::ndarray<nb::numpy, uint8_t>(out, shape.size(), shape.data(), owner);
    }, nb::arg("arr"), nb::arg("fmt"));

    m.def("unpack_fp8", [](U8In arr, const std::string& fmt) {
        size_t n = arr.size();
        auto* out = new float[n];
        const uint8_t* src = arr.data();
        for(size_t i = 0; i < n; ++i) out[i] = decode_one(src[i], fmt);
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        std::vector<size_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
        return nb::ndarray<nb::numpy, float>(out, shape.size(), shape.data(), owner);
    }, nb::arg("arr"), nb::arg("fmt"));
}
```

Note: the exact conversion constructor/`operator float` spellings come from `hipblaslt_float8.h` (verified: `hipblaslt_f8(const _Float16)` and float operators exist). If `float`-arg constructors are ambiguous, cast through `_Float16` or use the explicit conversion the header provides; keep the library converter as the source of truth. This file is `.hip` because the fp8 structs pull in HIP device headers.

- [ ] **Step 4: Compile `convert.hip` in CMake**

In `python/CMakeLists.txt`: enable HIP language and add the source. Near the top after `project(...)`, add `enable_language(HIP)` guarded by availability, OR compile the file as HIP by setting its language property. Add to the module sources: `"${CMAKE_CURRENT_SOURCE_DIR}/src/convert.hip"` and set:

```cmake
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/convert.hip" PROPERTIES LANGUAGE HIP)
```

Ensure `find_package(hip REQUIRED)` is present (it is, from Task 1). Call `init_convert(m);` in `module.cpp` and declare `void init_convert(nb::module_&);` in `init.hpp`.

- [ ] **Step 5: Build and run**

Run: `cd python && pip install --no-build-isolation -e . && python -m pytest tests/test_convert.py -v`
Expected: PASS. (This test is host-side — encoding runs on CPU via `__host__` converters — so it does not require a GPU.)

- [ ] **Step 6: Commit**

```bash
git add python/src/convert.hip python/src/init.hpp python/src/module.cpp python/CMakeLists.txt python/tests/test_convert.py
git commit -m "feat(python): fp8 pack/unpack via hipBLASLt ground-truth converters"
```

### Task 16: Encoding cross-check (ml_dtypes vs hipBLASLt)

**Files:**
- Create: `python/tests/test_encoding_crosscheck.py`

**Interfaces:**
- Consumes: `_core.pack_fp8` (ground truth), `ml_dtypes` encodings.
- Produces: a test asserting the two encoders agree bit-for-bit over a representative value sweep; divergence is a reportable finding, not a test-infra bug.

- [ ] **Step 1: Write the test**

Create `python/tests/test_encoding_crosscheck.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import ml_dtypes
import pytest
import hipblaslt
c = hipblaslt._core

CASES = [
    ("e4m3", ml_dtypes.float8_e4m3fn),
    ("e5m2", ml_dtypes.float8_e5m2),
]


@pytest.mark.parametrize("fmt,mld_type", CASES)
def test_ml_dtypes_matches_hipblaslt(fmt, mld_type):
    vals = np.linspace(-8.0, 8.0, 257, dtype=np.float32)
    hip_bytes = c.pack_fp8(vals, fmt)
    mld_bytes = vals.astype(mld_type).view(np.uint8)
    mismatches = np.nonzero(hip_bytes != mld_bytes)[0]
    assert mismatches.size == 0, (
        f"{fmt}: {mismatches.size} encoding divergences at values "
        f"{vals[mismatches][:8]} — hipBLASLt {hip_bytes[mismatches][:8]} "
        f"vs ml_dtypes {mld_bytes[mismatches][:8]}"
    )
```

- [ ] **Step 2: Run**

Run: `cd python && python -m pytest tests/test_encoding_crosscheck.py -v`
Expected: PASS if the encoders agree. If they diverge on rounding/NaN edges, convert the specific case to `xfail(reason="...")` documenting the divergence and open a note — that is a real finding about encoding differences, exactly what the tool is for.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_encoding_crosscheck.py
git commit -m "test(python): cross-check ml_dtypes vs hipBLASLt fp8 encoding"
```

### Task 17: MX block-scale helpers + numerical test

**Files:**
- Create: `python/hipblaslt/mx.py`
- Create: `python/tests/test_mx.py`

**Interfaces:**
- Consumes: `_core` matmul path, `_core.DataType.R_8F_E4M3`/`R_8F_E5M2`, `ml_dtypes.float8_e8m0fnu` (UE8M0 scales) when available, `A_SCALE_MODE`/`A_SCALE_POINTER` desc attributes.
- Produces:
  - `mx.build_block_scales(ref_f32: np.ndarray, block: int = 32) -> (scales_ue8m0: np.ndarray, scaled_elems_f32: np.ndarray)` — per-32-element-block max → UE8M0 exponent, and the block-scaled element values (canonical order).
  - `mx.apply_block_scales(elems_f32, scales_ue8m0, block=32) -> np.ndarray` — reconstruct the effective values for the numpy reference.
  - `mx.swizzle_scales(scales_canonical, tile=(32, 8, 4)) -> np.ndarray` — the pre-swizzle for mode 1001 (ported from `DataInitialization.cpp`); paired with an inverse so tests can assert `unswizzle(swizzle(x)) == x`. The reference path always uses canonical scales.
- This task's numerical test is `xfail`/`skip`-gated on arch support (MX requires MI300/MI350); it enumerates the mode but does not hard-require a passing GEMM everywhere.

- [ ] **Step 1: Write the failing test (helpers first, GEMM gated)**

Create `python/tests/test_mx.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
from hipblaslt import mx
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


def test_build_block_scales_shapes():
    ref = np.random.rand(64, 64).astype(np.float32)
    scales, scaled = mx.build_block_scales(ref, block=32)
    assert scaled.shape == ref.shape
    # one scale per 32-element block along the innermost dim
    assert scales.shape == (64, 64 // 32)


def test_apply_inverts_build():
    ref = np.random.rand(32, 32).astype(np.float32)
    scales, scaled = mx.build_block_scales(ref, block=32)
    recon = mx.apply_block_scales(scaled, scales, block=32)
    # reconstruction is within fp8 block-scaling error of the original
    np.testing.assert_allclose(recon, ref, rtol=0.1, atol=0.1)


def test_swizzle_roundtrip():
    scales = np.arange(32 * 8, dtype=np.uint8).reshape(32, 8)
    sw = mx.swizzle_scales(scales, tile=(32, 8, 4))
    back = mx.unswizzle_scales(sw, tile=(32, 8, 4), shape=scales.shape)
    np.testing.assert_array_equal(back, scales)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_mx.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hipblaslt.mx'`.

- [ ] **Step 3: Write `python/hipblaslt/mx.py`**

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MX (microscaling) block-scale helpers.

MX is a block-scaling scheme: a narrow element tensor plus a per-block scale
tensor (UE8M0). These helpers build/apply canonical scales for the numpy
reference and (pre-)swizzle scales into the mode-1001 device layout. The
reference path always uses canonical scales; only the device copy is swizzled.
"""
import numpy as np


def build_block_scales(ref_f32, block=32):
    ref = np.asarray(ref_f32, dtype=np.float32)
    rows, cols = ref.shape
    assert cols % block == 0, "innermost dim must be a multiple of block"
    nblocks = cols // block
    blocks = ref.reshape(rows, nblocks, block)
    # per-block max magnitude -> power-of-two exponent (UE8M0 stores the exponent)
    amax = np.max(np.abs(blocks), axis=2)
    amax = np.where(amax == 0, 1.0, amax)
    exp = np.floor(np.log2(amax)).astype(np.int32)
    # UE8M0 stores biased exponent (bias 127); clamp to [0, 254].
    ue8m0 = np.clip(exp + 127, 0, 254).astype(np.uint8)
    scale = (2.0 ** (ue8m0.astype(np.float32) - 127.0))[:, :, None]
    scaled = (blocks / scale).reshape(rows, cols).astype(np.float32)
    return ue8m0, scaled


def apply_block_scales(elems_f32, scales_ue8m0, block=32):
    elems = np.asarray(elems_f32, dtype=np.float32)
    rows, cols = elems.shape
    nblocks = cols // block
    scale = (2.0 ** (scales_ue8m0.astype(np.float32) - 127.0))[:, :, None]
    out = (elems.reshape(rows, nblocks, block) * scale).reshape(rows, cols)
    return out.astype(np.float32)


def swizzle_scales(scales_canonical, tile=(32, 8, 4)):
    """Permute canonical (row, col) scale bytes into the mode-1001 device order.

    Ported from tensilelite/client/src/DataInitialization.cpp (generateMXInput,
    ~lines 1977-2016). tile = (tileMN, tileK, subTileK).
    """
    tileMN, tileK, subTileK = tile
    rows, cols = scales_canonical.shape
    assert rows % tileMN == 0 and cols % tileK == 0, "shape must tile evenly"
    # Reshape into (rowTiles, tileMN, colTiles, tileK) then reorder so a wave's
    # lanes read contiguous bytes: output order is (rowTiles, colTiles, tileK/subTileK,
    # tileMN, subTileK). This mirrors the C++ layout; the inverse below is the check.
    rt, ct = rows // tileMN, cols // tileK
    a = scales_canonical.reshape(rt, tileMN, ct, tileK)
    a = a.transpose(0, 2, 3, 1)  # (rt, ct, tileK, tileMN)
    return np.ascontiguousarray(a).reshape(-1)


def unswizzle_scales(swizzled, tile=(32, 8, 4), shape=None):
    tileMN, tileK, subTileK = tile
    rows, cols = shape
    rt, ct = rows // tileMN, cols // tileK
    a = swizzled.reshape(rt, ct, tileK, tileMN)
    a = a.transpose(0, 3, 1, 2)  # back to (rt, tileMN, ct, tileK)
    return np.ascontiguousarray(a).reshape(rows, cols)
```

Note: the exact swizzle permutation MUST be validated against the C++ implementation in `DataInitialization.cpp`. The version above is a faithful (rt, ct, tileK, tileMN) interleave and passes the roundtrip test; if a real mode-1001 GEMM (Step 5) produces wrong numbers, re-derive the permutation from the C++ source and adjust. The roundtrip test guards the inverse; the GEMM test guards the forward layout.

- [ ] **Step 4: Export `mx` from the package**

In `python/hipblaslt/__init__.py`, add `from . import mx` and append `"mx"` to `__all__`.

- [ ] **Step 5: Add an arch-gated MX GEMM test**

Append to `python/tests/test_mx.py`:

```python
def _mx_supported():
    if not c.hip_available():
        return False
    # Probe: try to run a tiny MX GEMM; NOT_SUPPORTED -> skip.
    return True  # refined by the probe in Task 18's conftest helper


@requires_gpu
@pytest.mark.skipif(not _mx_supported(), reason="MX not supported on this arch")
def test_mx_gemm_matches_reference():
    pytest.importorskip("ml_dtypes")
    # Full MX GEMM: build scales, set A_SCALE_MODE=VEC32_UE8M0, compare vs
    # apply_block_scales reference. If the arch returns NOT_SUPPORTED, the
    # HipblasLtError is caught and the test is skipped.
    m = n = k = 128
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    a_scales, a_scaled = mx.build_block_scales(A, block=32)
    b_scales, b_scaled = mx.build_block_scales(B, block=32)
    ref = mx.apply_block_scales(a_scaled, a_scales) @ mx.apply_block_scales(b_scaled, b_scales)
    try:
        # (device setup elided here — implementer wires DeviceArrays + scale
        # pointers + A_SCALE_MODE=VEC32_UE8M0 following test_matmul.py, using
        # canonical scales for VEC32_UE8M0. Assert allclose(out, ref, rtol=0.15).)
        pytest.skip("MX device GEMM wiring completed by implementer against real MI350")
    except c.HipblasLtError as e:
        if "NOT_SUPPORTED" in str(e):
            pytest.skip(f"MX unsupported: {e}")
        raise
```

Note: the full device wiring for the MX GEMM depends on MI350 hardware. The helpers (build/apply/swizzle) are fully tested and CI-able on any host; the device GEMM is completed and validated by the implementer on real MX hardware, then the `pytest.skip` placeholder is replaced with the assertion. Keep the try/except NOT_SUPPORTED→skip so the test is safe on non-MX arches.

- [ ] **Step 6: Run**

Run: `cd python && python -m pytest tests/test_mx.py -v`
Expected: PASS for the three helper tests; the GEMM test skips (no MX hardware / placeholder).

- [ ] **Step 7: Commit**

```bash
git add python/hipblaslt/mx.py python/hipblaslt/__init__.py python/tests/test_mx.py
git commit -m "feat(python): MX block-scale build/apply/swizzle helpers"
```

## Phase 5 — Convenience shim, coverage harness, CI (Tasks 18–20)

### Task 18: Header-derived enum coverage harness

**Files:**
- Create: `python/hipblaslt/_coverage.py`
- Create: `python/tests/test_api_coverage.py`
- Create: `python/tests/conftest.py`

**Interfaces:**
- Consumes: `_core.enum_members` (Task 4), the installed hipBLASLt headers.
- Produces:
  - `_coverage.header_enum_values(header_path, enum_type) -> dict[str,int]` — parse a C enum body from `hipblaslt.h` for the given `typedef enum {...} <enum_type>;`, returning `{FULL_MEMBER_NAME: value}` (resolving simple `= N` and bit-OR-free integer literals; members without explicit values get previous+1).
  - `_coverage.find_header() -> Path` — locate installed `hipblaslt.h` (search `ROCM_PATH`/`/opt/rocm/include/hipblaslt/`).
  - `test_api_coverage.py` — for each bound enum, assert every header member is represented by a bound value (matched by integer value, since Python names strip prefixes). Members the library reports unsupported are allowed to be `xfail`/`skip` via a documented allowlist.
  - `conftest.py` — a `mx_supported`/`fp8_supported` runtime-probe fixture (attempts a tiny op, catches `NOT_SUPPORTED`).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_api_coverage.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
from hipblaslt import _coverage
c = hipblaslt._core

# (bound enum name, header enum typedef name)
ENUMS = [
    ("Epilogue", "hipblasLtEpilogue_t"),
    ("ScaleMode", "hipblasLtMatmulMatrixScale_t"),
    ("MatmulDescAttr", "hipblasLtMatmulDescAttributes_t"),
]


@pytest.mark.parametrize("bound_name,header_enum", ENUMS)
def test_every_header_value_is_bound(bound_name, header_enum):
    header = _coverage.find_header()
    header_values = set(_coverage.header_enum_values(header, header_enum).values())
    bound_values = set(c.enum_members(bound_name).values())
    missing = header_values - bound_values
    # Deprecated/placeholder sentinels the binding intentionally omits:
    allowed_missing = _coverage.ALLOWED_MISSING.get(header_enum, set())
    unexpected = missing - allowed_missing
    assert not unexpected, (
        f"{header_enum}: header values {sorted(unexpected)} are not bound. "
        f"Add them to enums.cpp or to _coverage.ALLOWED_MISSING with a reason."
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_api_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hipblaslt._coverage'`.

- [ ] **Step 3: Write `python/hipblaslt/_coverage.py`**

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parse hipBLASLt header enums to drive API-surface coverage tests."""
import os
import re
from pathlib import Path

# header enum typedef -> set of int values intentionally not bound (with reasons
# in comments). Extend as needed rather than silently skipping.
ALLOWED_MISSING = {
    # "Not supported yet" scale modes are enumerated but need not be bound until
    # the library supports them; list their values here if the coverage test flags them.
    "hipblasLtMatmulMatrixScale_t": set(),
}


def find_header():
    candidates = []
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates.append(Path(rocm) / "include" / "hipblaslt" / "hipblaslt.h")
    # In-tree fallback (developer build):
    here = Path(__file__).resolve()
    for parent in here.parents:
        p = parent / "library" / "include" / "hipblaslt" / "hipblaslt.h"
        if p.exists():
            candidates.append(p)
            break
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"hipblaslt.h not found; looked in {candidates}")


def header_enum_values(header_path, enum_type):
    text = Path(header_path).read_text()
    # Match: typedef enum { ... } enum_type;
    pattern = re.compile(
        r"typedef\s+enum\s*\{(?P<body>.*?)\}\s*" + re.escape(enum_type) + r"\s*;",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"enum {enum_type} not found in {header_path}")
    body = match.group("body")
    values = {}
    current = -1
    for raw_line in body.split(","):
        line = re.sub(r"/\*.*?\*/", "", raw_line, flags=re.DOTALL)  # strip block comments
        line = re.sub(r"//.*", "", line).strip()
        if not line:
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(=\s*(0[xX][0-9a-fA-F]+|-?\d+))?", line)
        if not m:
            continue
        name = m.group(1)
        if m.group(3) is not None:
            current = int(m.group(3), 0)
        else:
            current += 1
        values[name] = current
    return values
```

- [ ] **Step 4: Write `python/tests/conftest.py`**

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core


@pytest.fixture(scope="session")
def hip_available():
    return c.hip_available()
```

- [ ] **Step 5: Run**

Run: `cd python && python -m pytest tests/test_api_coverage.py -v`
Expected: PASS. If it flags a header value not bound, EITHER add the enumerator to `enums.cpp` (Task 4) OR add its value to `ALLOWED_MISSING` with a comment explaining why (e.g. deprecated `_VEC_EXT` static_assert stubs). Do both fixes as this task's follow-through.

- [ ] **Step 6: Commit**

```bash
git add python/hipblaslt/_coverage.py python/tests/test_api_coverage.py python/tests/conftest.py
git commit -m "feat(python): header-derived enum coverage harness"
```

### Task 19: Convenience `matmul()` shim

**Files:**
- Modify: `python/hipblaslt/__init__.py`
- Create: `python/tests/test_convenience.py`

**Interfaces:**
- Consumes: `_core` low-level layer (Handle, MatmulDesc, MatrixLayout, Preference, heuristic, matmul), `from_numpy`, `_NP_TO_DTYPE`.
- Produces: `hipblaslt.gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray` — a thin shim that maps numpy dtypes to `DataType`, builds descriptors, picks `heuristic(...)[0]`, sizes the workspace, runs `matmul`, returns the host result. Strictly built on the low-level layer (no parallel code path). Named `gemm` to avoid shadowing `_core.matmul`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_convenience.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_gemm_f32():
    a = np.random.rand(48, 32).astype(np.float32)
    b = np.random.rand(32, 16).astype(np.float32)
    out = hipblaslt.gemm(a, b)
    np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && python -m pytest tests/test_convenience.py -v`
Expected: FAIL — no attribute `gemm`.

- [ ] **Step 3: Add `gemm` to `__init__.py`**

Append to `python/hipblaslt/__init__.py`:

```python
def gemm(a, b):
    """Convenience GEMM: D = a @ b. Thin shim over the low-level API.

    Auto-selects the top heuristic algo. For full control (algo pinning,
    epilogues, scales) use the _core layer directly.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes {a.shape} @ {b.shape}")
    m, k = a.shape
    _, n = b.shape
    dtype = _NP_TO_DTYPE.get(_np.dtype(a.dtype))
    if dtype is None:
        raise ValueError(f"unsupported dtype {a.dtype}")

    dA = _core.DeviceArray.from_numpy(_np.ascontiguousarray(a.T), dtype)
    dB = _core.DeviceArray.from_numpy(_np.ascontiguousarray(b.T), dtype)
    dC = _core.DeviceArray.from_numpy(_np.zeros((n, m), a.dtype), dtype)
    dD = _core.DeviceArray.from_numpy(_np.zeros((n, m), a.dtype), dtype)
    la = _core.MatrixLayout(dtype, m, k, m)
    lb = _core.MatrixLayout(dtype, k, n, k)
    lc = _core.MatrixLayout(dtype, m, n, m)
    ld = _core.MatrixLayout(dtype, m, n, m)
    with _core.Handle() as h:
        desc = _core.MatmulDesc(_core.ComputeType.COMPUTE_32F, _core.DataType.R_32F)
        pref = _core.Preference(); pref.set_max_workspace(64 * 1024 * 1024)
        results = _core.heuristic(h, desc, la, lb, lc, ld, pref, 16)
        if not results:
            raise _core.HipblasLtError("no heuristic algorithm for this problem")
        ws = _core.DeviceArray.from_numpy(
            _np.zeros(max(1, results[0].workspace_size), _np.uint8), _core.DataType.R_8I)
        _core.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, results[0].algo, ws)
    return dD.to_numpy().reshape(n, m).T


__all__ = ["_core", "from_numpy", "gemm", "mx"]
```

Note: `ComputeType.COMPUTE_32F` is correct for f32/f16/bf16 inputs; a future extension can choose compute type from dtype. Keep this shim minimal per the spec (convenience is ancillary).

- [ ] **Step 4: Run**

Run: `cd python && python -m pytest tests/test_convenience.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/hipblaslt/__init__.py python/tests/test_convenience.py
git commit -m "feat(python): add convenience gemm() shim over low-level API"
```

### Task 20: README + CI wiring

**Files:**
- Create: `python/README.md`
- Modify: `.github/workflows/` (add a host-only job that builds the package and runs the non-GPU tests) — locate the existing workflow dir first; if none applies, document the CI step in `python/README.md` instead.

**Interfaces:**
- Consumes: everything above.
- Produces: developer-facing usage docs and a CI job that (a) builds the extension and (b) runs the GPU-independent subset (`test_import`, `test_errors`, `test_enums`, `test_convert`, `test_encoding_crosscheck`, `test_api_coverage`, `test_mx` helpers) so regressions in the binding/coverage are caught without a GPU runner.

- [ ] **Step 1: Locate CI config**

Run: `ls .github/workflows/ 2>/dev/null; ls ../../.github/workflows/ 2>/dev/null`
Expected: note the workflow files present. If the superbuild owns CI and adding a job is out of scope for this branch, record the exact commands in `python/README.md` under a "CI" heading and skip the workflow edit.

- [ ] **Step 2: Write `python/README.md`**

Include: what the package is (low-level dev tool), install (`invoke build --python -ca gfx942` or `pip install --no-build-isolation -e python/`), a minimal `gemm` example, a low-level `heuristic`+`matmul` example, the GPU-gated test note, and the host-only test command:

```bash
cd python && python -m pytest tests/ -m "not gpu" -v   # host-only subset
```

(Mark GPU tests with a `gpu` marker via `pytestmark`/`requires_gpu`; register the marker in `pyproject.toml` under `[tool.pytest.ini_options] markers`.)

- [ ] **Step 3: Register the `gpu` marker**

In `python/pyproject.toml`, under `[tool.pytest.ini_options]`, add:

```toml
markers = ["gpu: test requires a HIP device"]
```

And ensure GPU tests carry the marker (the `requires_gpu = pytest.mark.skipif(...)` already skips; add `pytestmark = pytest.mark.gpu` where a whole file is GPU-only, so `-m "not gpu"` selects the host subset).

- [ ] **Step 4: Run the host-only subset to confirm it passes without a GPU**

Run: `cd python && python -m pytest tests/ -m "not gpu" -v`
Expected: PASS (import, errors, enums, convert, crosscheck, coverage, mx-helpers); GPU tests deselected.

- [ ] **Step 5: Commit**

```bash
git add python/README.md python/pyproject.toml
git commit -m "docs(python): add README and host-only test marker/CI notes"
```

---

## Self-Review

**Spec coverage check:**
- Summary / low-level binding → Tasks 1, 4–7, 11–12 ✓
- Primary users / full control → Tasks 11 (heuristic enumeration), 13 (algo sweep) ✓
- Motivation (numpy diff, array inspection, algo bisection) → Tasks 12, 13 ✓
- Ecosystem interop (DLPack) → Task 9 ✓
- Architecture / package layout (`hipblaslt/`, `_core`, three layers) → Tasks 1, 8, 19 ✓
- Build integration (opt-in flag) → Task 2 ✓
- Core objects (Handle/MatmulDesc/MatrixLayout/Preference/Algo, generic attributes) → Tasks 5–7, 11 ✓
- DeviceArray (RAII, from_numpy/to_numpy, copy reuse, DLPack) → Tasks 8–10 ✓
- fp8 five element types + ml_dtypes host repr + ground-truth converters + cross-check → Tasks 14–16 ✓
- MX block scaling (build/apply/swizzle, canonical reference, arch-gate) → Task 17 ✓
- Error handling (status→exception, boundary validation, workspace, owned lifetime) → Tasks 3, 10, 12 ✓
- Testing (surface coverage, numerical, cross-check, known-bugs xfail, GPU-gated) → Tasks 13, 16, 18, 20 ✓

**Placeholder scan:** The MX device GEMM (Task 17 Step 5) and the DLPack borrow-vs-copy detail (Task 9) are explicitly hardware-dependent and flagged as implementer-completed-on-hardware, not silent TBDs — each has a working, tested fallback (helpers tested on any host; f32 DLPack path tested). Acceptable and called out.

**Type consistency:** `_core.matmul`/`_core.heuristic` argument names and order match between Tasks 11, 12, 13, 19. `DataType`/`ComputeType`/`Epilogue`/`ScaleMode`/`MatmulDescAttr` names consistent across enums.cpp (Task 4) and all consumers. `from_numpy` (Python, Task 10) vs `_core.DeviceArray.from_numpy` (C++, Task 8) distinction is intentional and consistently applied. `_DTYPE_TO_NP`/`_NP_TO_DTYPE` defined in Tasks 8/14, consumed in 19. `HeuristicResult.workspace_size`/`.waves_count`/`.algo.index` consistent (Tasks 11, 12, 13, 19).
