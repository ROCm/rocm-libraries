# MX GEMM bridge

The bridge runs CK-Tile microscaling GEMM with FP8 (OCP E4M3) or packed FP4
(E2M1) inputs, one E8M0 scale per 32 K elements, FP32 accumulation, and FP16
output. Layout is RCR: A is stored as `[M, K]`, B as `[N, K]`, and C as `[M, N]`.
FP4 stores the even K element in the low nibble and the odd K element in the high
nibble, so its input storage is `[M, K/2]` and `[N, K/2]` bytes.

| Target | Pipeline | Epilogue | Warp tile |
| --- | --- | --- | --- |
| gfx950 | `comp_async` | CShuffle | 16 × 16 × 128 |
| gfx1250 | `comp_tdm` | TDM | 16 × 16 × 128 |

The gfx1250 path uses WMMA without cluster launch and supports revision 0.
M and N may include partial tiles; the host pads their scale buffers before
reshuffling. K must be divisible by both 128 and the selected block tile K.
Split-K, persistent execution, and K padding are not supported by this bridge's
gfx1250 path. The separate 32 × 32 packed-FP4 instruction is not selected.

The dispatcher generator reuses `MxGemmKernelBuilder` from Tile Engine. Both
entry points use the same generated kernel, while their host code chooses the
architecture's native scale reshuffling helper. Select one GPU architecture per
Tile Engine build.

## Run the bridge

From `projects/composablekernel`, with NumPy installed and a matching ROCm
compiler/runtime available:

```bash
export CK_TILE_HIPCC=/opt/rocm/bin/hipcc
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export CK_TILE_BENCH_WARMUP=1
export CK_TILE_BENCH_REPEAT=2
PYTHONPATH=dispatcher/python python3 - <<'PYCODE'
from pathlib import Path
import numpy as np
from mx_gemm_utils import (
    GpuMxGemmRunner, MxGemmProblem, default_fp4_config, default_fp8_config,
    setup_multiple_mx_gemm_dispatchers,
)

for make_config in (default_fp8_config, default_fp4_config):
    config = make_config("gfx1250")
    library = setup_multiple_mx_gemm_dispatchers(
        [config], output_dir=Path("build/mx_bridge"), gfx_arch="gfx1250",
        parallel=False,
    )[0]
    assert library is not None, "Kernel compilation failed"
    runner = GpuMxGemmRunner(library, dtype=config.datatype, arch="gfx1250")
    problem = MxGemmProblem(129, 257, 384)
    a_ref, b_ref, a, b, sa, sb = runner.make_inputs(problem, scale=1.0, seed=5)
    rng = np.random.default_rng(19)
    sa[:] = rng.integers(124, 130, size=sa.shape, dtype=np.uint8)
    sb[:] = rng.integers(124, 130, size=sb.shape, dtype=np.uint8)
    result = runner.run(problem, a, b, sa, sb)
    reference = runner.reference(a_ref, b_ref, sa, sb, problem).astype(np.float32)
    got = np.asarray(result.C, dtype=np.float32)
    denominator = np.abs(reference) + max(np.abs(reference).max() * 1e-2, 1e-6)
    error = np.max(np.abs(got - reference) / denominator)
    assert np.isfinite(got).all() and error <= 5e-2
    print(config.datatype, "PASS", "max_rel=", error, "time_ms=", result.time_ms)
PYCODE
```

## Regression tests

CPU tests cover architecture selection, invalid configurations, scale/packing
codecs, the CI configuration, and exact generated-header parity with Tile Engine:

```bash
python3 -m unittest discover -s dispatcher/tests -p test_mx_gemm_bridge.py -v
```

The GPU suite builds all eight gfx1250 CI configurations (two input types × two
M tiles × two N tiles), runs seven shapes with two seeds, varies scales across
rows and K blocks, and checks K-tail/split-K rejection. It also runs on gfx950
with that architecture's supported default configurations:

```bash
CK_TILE_BENCH_WARMUP=1 CK_TILE_BENCH_REPEAT=2 \
python3 -m unittest discover -s dispatcher/tests -p test_mx_gemm_gpu_correctness.py -v
```

The native C++ regression uses CK-Tile directly:

```bash
cmake -S . -B build/mx_native \
  -DBUILD_DEV=ON -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1250 \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/mx_native --target test_ck_tile_mx_gemm_e8m0_gfx1250 --parallel 2
ctest --test-dir build/mx_native -R test_ck_tile_mx_gemm_e8m0_gfx1250 --output-on-failure
```

For Tile Engine, enable `BUILD_CK_TILE_ENGINE`, use `GPU_TARGETS=gfx1250`, and
build `benchmark_mx_gemm_all`. The MX operation selects
`default_ci_config_gfx1250.json` automatically unless a custom MX config is
provided. Benchmark executables accept `-m=`, `-n=`, `-k=`, `-verify=1`, `-init=0`,
`-warmup=1`, and `-repeat=2` for a short correctness run with random inputs.
