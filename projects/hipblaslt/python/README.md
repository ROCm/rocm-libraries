# hipBLASLt Python bindings

Low-level Python bindings for the hipBLASLt GEMM API, built with [nanobind](https://github.com/wjakob/nanobind).

The bindings are a development and testing tool for hipBLASLt and its consumers, not a ROCm deliverable. They're opt-in and source-build only: you build them from a rocm-libraries checkout against an installed ROCm, and hipBLASLt's own build never builds or installs them.

## Enums

`hipblaslt._core` binds every public enum as a Python `IntEnum`, with the C prefix dropped: `HIPBLASLT_EPILOGUE_RELU` is `Epilogue.RELU`, `HIP_R_16F` is `DataType.R_16F`, and `hipblaslt_ext::GemmType::HIPBLASLT_GEMM` is `GemmType.GEMM`. Values that exist only with an opt-in build feature (`HIPBLASLT_HAS_...`) are bound only when the hipBLASLt you build against turns that feature on. The few values left out on purpose, sentinels and values the header marks "Not supported yet", are listed with a reason in `src/hipblaslt/_coverage.py`.

## Building

You need CMake 3.25 or later and Ninja, plus a ROCm install with hipBLASLt's development files.

```bash
pip install cmake ninja
```

Point `ROCM_PATH` at the ROCm to build against, then install into the active Python environment:

```bash
# A ROCm install, for example /opt/rocm:
ROCM_PATH=/opt/rocm pip install projects/hipblaslt/python

# TheRock's rocm-sdk wheels (rocm[libraries,devel]):
ROCM_PATH="$(rocm-sdk path --root)" pip install projects/hipblaslt/python
```

On Windows, hipBLASLt's headers need clang, so build with ROCm's `amdclang++` and Ninja from a Visual Studio developer prompt:

```bat
for /f "delims=" %i in ('rocm-sdk path --root') do set ROCM_PATH=%i
set CXX=%ROCM_PATH%\lib\llvm\bin\amdclang++.exe
set CC=%ROCM_PATH%\lib\llvm\bin\amdclang.exe
set CMAKE_GENERATOR=Ninja
pip install projects\hipblaslt\python
```

To build hipBLASLt and the bindings together, use `invoke build --python` from `projects/hipblaslt`. It installs the bindings into the Python environment that runs `invoke`.

## Handles and descriptors

`Handle`, `MatmulDesc`, `MatrixLayout` and `Preference` wrap hipBLASLt's objects and release them when Python drops them. `Handle()` needs a visible HIP device and raises `HipblasLtError` without one; descriptors work without a GPU.

Every descriptor attribute goes through one call, `set_attribute(attr, value)`, and comes back through `get_attribute(attr)`:

```python
from hipblaslt import _core as hb

desc = hb.MatmulDesc(hb.ComputeType.COMPUTE_32F, hb.DataType.R_32F)
desc.set_attribute(hb.MatmulDescAttr.TRANSA, hb.Operation.T)
desc.set_attribute(hb.MatmulDescAttr.EPILOGUE, hb.Epilogue.RELU)
assert desc.get_attribute(hb.MatmulDescAttr.EPILOGUE) is hb.Epilogue.RELU
```

The bindings know each attribute's C storage type, and its enum if it has one, and write exactly that many bytes. A value that doesn't fit, or isn't a member of the attribute's enum, raises `ValueError` or `TypeError` before hipBLASLt sees it. That matters because hipBLASLt itself accepts most wrong values without complaint.

## Which libhipblaslt gets loaded

`import hipblaslt` picks one library and loads it by full path before anything else, in this order:

1. `HIPBLASLT_PY_LIBRARY`, a full path to `libhipblaslt.so.1` or `libhipblaslt.dll`;
2. TheRock's rocm-sdk wheels, if they're installed;
3. the hipBLASLt the bindings were built against;
4. `$ROCM_PATH`, then `$ROCM_HOME`, if that one is gone.

It warns (`HipblasLtLoadWarning`) when it doesn't end up on the hipBLASLt the bindings were built against and `HIPBLASLT_PY_LIBRARY` isn't set.

If torch (or anything else) already loaded hipBLASLt, the bindings use that copy instead. The import raises `HipblasLtLoadError` when `HIPBLASLT_PY_LIBRARY` can't take effect because a different hipBLASLt is already loaded, or when hipBLASLt's companion libraries come from another build. The first `Handle()` in a process warns (`HipblasLtLoadWarning`) when the loaded hipBLASLt isn't the build the bindings were compiled against. The check needs a handle, so it doesn't run at import.

## Testing

```bash
pip install pytest
pytest projects/hipblaslt/python/tests
```

Tests marked `gpu` skip when no device is visible. Set `HIPBLASLT_PY_REQUIRE_GPU=1` to make them fail instead, as the GPU CI lanes do.

Component CI runs these tests through [`scripts/ci_lane.sh`](scripts/ci_lane.sh), which installs a pinned TheRock nightly into a fresh venv, builds the bindings and runs pytest. CI runs those as separate steps; with no step named, the script runs them all, so it works the same way locally:

```bash
HIPBLASLT_PY_ROCM_VERSION=10.3.0a20261116 HIPBLASLT_PY_DEVICE=gfx942 \
  bash projects/hipblaslt/python/scripts/ci_lane.sh
```

The gfx950 lane runs when a PR has the `ci:run-gfx950` label, and on a manual Component CI run. See [TESTING.md](../TESTING.md#supported-configurations) for every lane.

## Design

The decisions behind the bindings, and why, are in [`DESIGN.md`](DESIGN.md).
