# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Multi-GPU smoke tests for the FP16 RCR GEMM benchmark sweep harness."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import mkdtemp

from rocke.benchmark.gemm.fp16_rcr_sweep import (
    GemmSweepConfig,
    GemmSweepShape,
    compile_sweep_variants,
    expand_sweep,
    run_sweep_variants,
)


def _visible_device_count() -> int:
    """Visible HIP device count via the rocke runtime (no torch dependency)."""
    try:
        from rocke.runtime.hip_module import get_device_count

        return get_device_count()
    except Exception:
        return 0


def _device_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0)
    except Exception:
        return None


# The sweep builds+launches gfx950 code objects across lanes, so require both the
# exact arch and >=2 visible devices.
@unittest.skipUnless(
    _visible_device_count() >= 2 and _device_arch() == "gfx950",
    "requires at least two gfx950 GPUs",
)
class TestGemmFp16RcrMultiGpuSweep(unittest.TestCase):
    def test_compile_once_and_run_sweep_on_multiple_gpu_lanes(self):
        plan = expand_sweep(
            GemmSweepConfig(
                arch="gfx950",
                spec_id="auto",
                shapes=(GemmSweepShape(128, 128, 64, "multigpu", verify=True),),
                warmup_iters=1,
                timed_iters=3,
            )
        )
        self.assertGreaterEqual(len(plan.variants), 2)
        out_dir = Path(mkdtemp(prefix="rocke_gemm_multigpu_sweep_"))
        builds = compile_sweep_variants(plan, out_dir / "artifacts", parallel=2)
        self.assertEqual(len(builds), len(plan.variants))
        for build in builds:
            self.assertTrue(build.ok, build.error)
        runs = run_sweep_variants(
            plan,
            builds,
            parallel=min(_visible_device_count(), 4),
            timeout_s=120,
        )
        self.assertEqual(len(runs), len(builds))
        for run in runs:
            self.assertTrue(run.ok, run.error + run.stderr[-500:])
            self.assertIn("bad=0/", run.stdout)
            self.assertIsNotNone(run.tflops)


if __name__ == "__main__":
    unittest.main()
