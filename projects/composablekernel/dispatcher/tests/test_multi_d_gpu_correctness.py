#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU correctness test for the MULTI-D GEMM dispatcher bridge (PR #9308).

Multi-D GEMM fuses extra D operands into the epilogue:
``E = elementwise_op(A @ B, D0, D1, ...)``. This test builds a real multi_d
dispatcher .so (elementwise_op = MultiDAdd, 2 D tensors), runs it on-device via
``GpuMultiDGemmRunner``, and compares E to an independent fp32 numpy reference
``E_ref = A @ B + D0 + D1``.

Real numeric check -- random A/B/D, element-wise validation vs. fp32 reference,
plus a non-zero / finite guard so a mis-launched (all-zero) kernel FAILS.
The multi_d TE op is fp16-only, so only fp16 is exercised.

Runs green on gfx942 (MI300X). SKIPs cleanly with no GPU / hipcc / static lib.

Run:
  python3 -m pytest tests/test_multi_d_gpu_correctness.py -v
  python3 tests/test_multi_d_gpu_correctness.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    GpuMultiDGemmRunner,
    MultiDGemmProblem,
    setup_multiple_gemm_dispatchers,
)

_TOL = 1e-2  # fp16 GEMM + fp16 D-fuse precision band


def _detect_arch():
    import subprocess
    try:
        out = subprocess.check_output(["rocminfo"], text=True,
                                      stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        if "Name:" in line and "gfx" in line:
            return line.split()[-1].strip()
    return None


def _static_lib_present():
    try:
        import ctypes_utils as _cu
        return (_cu.get_build_dir() / "libck_tile_dispatcher.a").exists()
    except Exception:
        return False


def _max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    ref_max = float(np.abs(r).max())
    den = np.abs(r) + max(ref_max * 1e-2, 1e-6)
    return float(np.max(np.abs(g - r) / den))


class TestMultiDGemmGpu(unittest.TestCase):
    ARCH = _detect_arch()

    def setUp(self):
        if self.ARCH is None:
            self.skipTest("no GPU / rocminfo not available")
        if not _static_lib_present():
            self.skipTest(
                "dispatcher static lib (libck_tile_dispatcher.a) not built; "
                "multi_d is registry-routed and needs it"
            )

    def test_fp16_multid_add(self):
        num_d = 2
        cfg = GemmKernelConfig(
            dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32",
            layout_a="row", layout_b="col", layout_c="row",
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
            variant="multi_d", elementwise_op="MultiDAdd",
            num_d_tensors=num_d, d_layout="row",
            gfx_arch=self.ARCH,
        )
        so_paths = setup_multiple_gemm_dispatchers([cfg], verbose=False)
        so = so_paths[0]
        if so is None:
            self.fail("multi_d fp16 kernel failed to build")

        M, N, K = 512, 512, 512
        problem = MultiDGemmProblem(M=M, N=N, K=K, num_d=num_d)

        rng = np.random.default_rng(11)
        A = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
        B = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
        Ds = [rng.uniform(-1.0, 1.0, (M, N)).astype(np.float32)
              for _ in range(num_d)]

        runner = GpuMultiDGemmRunner(so)
        self.assertEqual(runner.num_d_tensors, num_d,
                         "kernel D-count mismatch")
        result = runner.run(A.astype(np.float16), B.astype(np.float16),
                            [d.astype(np.float16) for d in Ds], problem)
        self.assertEqual(result.status, 0,
                         f"multi_d run status={result.status}")
        self.assertGreater(result.time_ms, 0.0, "multi_d time_ms not positive")

        # Reference at fp16 input precision, accumulated + fused in fp32:
        #   E = A@B + D0 + D1  (MultiDAdd).
        A_q = A.astype(np.float16).astype(np.float32)
        B_q = B.astype(np.float16).astype(np.float32)
        E_ref = A_q @ B_q
        for d in Ds:
            E_ref = E_ref + d.astype(np.float16).astype(np.float32)

        E_got = np.asarray(result.output).astype(np.float32)
        self.assertFalse(np.all(E_got == 0.0), "multi_d output all-zero")
        self.assertTrue(np.all(np.isfinite(E_got)), "multi_d output NaN/Inf")
        mre = _max_rel_err(E_got, E_ref)
        self.assertLessEqual(
            mre, _TOL,
            f"multi_d fp16 max_rel={mre:.4f} > tol={_TOL} (M={M} N={N} K={K})",
        )
        print(f"[multi_d/fp16] max_rel={mre:.4e}, time_ms={result.time_ms:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
