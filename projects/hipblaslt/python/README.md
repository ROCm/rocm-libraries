# hipblaslt Python bindings

Low-level Python bindings for the hipBLASLt GEMM API, targeting hipBLASLt
developers who need direct access to handle, descriptor, heuristic, and
matmul objects from Python — without any framework or high-level abstraction
in the way.

## Installation

### End users (uv — recommended)

```bash
uv pip install --no-build-isolation ./python/
```

Run from the `projects/hipblaslt` directory. The `--no-build-isolation` flag
is required because the build needs the ROCm headers already on the system.

### Contributors (conda dev environment — current convention)

```bash
conda activate pydev313          # Python 3.13 dev env for this project
pip install "scikit-build-core>=0.10" "nanobind>=2.0" pytest
cd python
pip install --no-build-isolation \
  --config-settings "cmake.args=-DROCM_PATH=/opt/rocm;-Dhipblaslt_DIR=/opt/rocm/lib/cmake/hipblaslt" \
  -e .
```

Adjust `-DROCM_PATH` and `-Dhipblaslt_DIR` to match your ROCm installation
(e.g. `/opt/rocm-7.2.4` for a versioned install).

### Via invoke build (host + device lib + Python in one step)

```bash
invoke build -ca gfx942 --python
```

Any PEP 517-compatible environment (plain venv, poetry, pixi) works the same
way — the package has no env-manager dependency.

## Usage

### Minimal GEMM (convenience shim)

```python
import numpy as np
import hipblaslt

a = np.random.rand(128, 64).astype(np.float32)
b = np.random.rand(64, 32).astype(np.float32)
d = hipblaslt.gemm(a, b)   # returns np.ndarray, shape (128, 32)
```

`gemm()` auto-selects the top heuristic algorithm and handles all device
transfers internally. For full control, use the low-level API below.

### Low-level: heuristic enumeration + matmul

This is the key feature of the binding — iterate all candidate algorithms
for a problem size and pin a specific one:

```python
import numpy as np
import hipblaslt
c = hipblaslt._core

m, n, k = 256, 128, 64
dtype = c.DataType.R_32F

A = np.random.rand(m, k).astype(np.float32)
B = np.random.rand(k, n).astype(np.float32)

# Allocate device arrays (column-major: pass the transpose)
dA = c.DeviceArray.from_numpy(np.ascontiguousarray(A.T), dtype)
dB = c.DeviceArray.from_numpy(np.ascontiguousarray(B.T), dtype)
dC = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), dtype)
dD = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), dtype)

la = c.MatrixLayout(dtype, m, k, m)
lb = c.MatrixLayout(dtype, k, n, k)
lc = c.MatrixLayout(dtype, m, n, m)
ld = c.MatrixLayout(dtype, m, n, m)

with c.Handle() as h:
    desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, dtype)
    pref = c.Preference()
    pref.set_max_workspace(64 * 1024 * 1024)

    # Enumerate up to 32 candidate algorithms
    results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
    print(f"{len(results)} candidate algorithms")
    for r in results:
        print(f"  algo #{r.algo.index}  workspace={r.workspace_size} bytes")

    # Run with a specific algorithm (results[0] is the top heuristic pick)
    ws = c.DeviceArray.from_numpy(
        np.zeros(max(1, results[0].workspace_size), np.uint8), c.DataType.R_8I
    )
    c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, results[0].algo, ws)
    out = dD.to_numpy().reshape(n, m).T   # back to row-major (m x n)

np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-3)
```

## Running tests

```bash
# Host-only (no GPU needed — import, enums, errors, convert, crosscheck, mx helpers):
cd python && conda run -n pydev313 python -m pytest tests/ -m "not gpu and not mi350" -v

# All GPU tests (needs a HIP device):
cd python && conda run -n pydev313 python -m pytest tests/ -m "gpu" -v

# MI350-deferred tests (needs gfx950 / MI350):
cd python && conda run -n pydev313 python -m pytest tests/ -m "mi350" -v
```

## CI

The superbuild CI is hosted at the `rocm-libraries` repository level
(`.github/workflows/component-ci.yml`). For local validation without a GPU:

```bash
cd python
conda run -n pydev313 python -m pytest tests/ -m "not gpu and not mi350" -v
```

Expected output: all host-only tests pass; GPU and MI350 tests are deselected
(not collected). To validate the build system (PEP 517 / uv):

```bash
# Install uv if needed:
conda run -n pydev313 python -m pip install uv

cd python && conda run -n pydev313 uv build --no-build-isolation
# Produces: dist/hipblaslt-0.1.0-cp313-cp313-linux_x86_64.whl
```
