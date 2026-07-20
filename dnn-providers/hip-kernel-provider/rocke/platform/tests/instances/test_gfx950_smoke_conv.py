# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU smoke tests for the implicit-GEMM forward convolution benchmark.

Tests bf16 and fp32 dtypes. Each test method runs the benchmark sweep with
--verify (the benchmark itself prints PASS/FAIL per kernel) and checks TFLOPS
against the committed baseline in rocke_gfx950_smoke_perf.json.

Run on a gfx950 ROCm runner:
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=rocke/platform/python \
    python rocke/platform/tests/instances/test_gfx950_smoke_conv.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from rocke.assets import platform_root
from rocke.runtime.hip_module import get_device_arch, get_device_name

_DEFAULT_BASELINE = (
    platform_root() / "tests" / "golden" / "rocke_gfx950_smoke_perf.json"
)

GPU_ARCH = get_device_arch(0)
GPU_NAME = get_device_name(0)
_HAS_TORCH = importlib.util.find_spec("torch") is not None

_DETECTED = f"{GPU_ARCH} ({GPU_NAME})" if GPU_ARCH else "no ROCm GPU detected"
_SKIP_REASON = (
    f"needs a gfx950 ROCm GPU; detected {_DETECTED}"
    if _HAS_TORCH
    else f"needs a gfx950 ROCm GPU + torch; detected {_DETECTED} (torch not importable)"
)


@unittest.skipUnless(GPU_ARCH == "gfx950" and _HAS_TORCH, _SKIP_REASON)
class TestGfx950ConvSmoke(unittest.TestCase):
    maxDiff = 4000
    baseline = json.loads(
        Path(
            os.environ.get("ROCKE_GFX950_PERF_BASELINE", _DEFAULT_BASELINE)
        ).read_text()
    )

    def _run_benchmark(self, dtype: str, timeout: int = 600) -> str:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "rocke.benchmark.benchmark_implicit_gemm_conv",
                "--arch",
                "gfx950",
                "--dtype",
                dtype,
                "--verify",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, out[-3500:])
        return out

    def _verify_and_sweep(self, dtype: str, baseline_key: str):
        out = self._run_benchmark(dtype)

        # TODO: Reenable
        # self.assertNotIn(
        #     "FAIL", out, f"conv {dtype} correctness failure:\n{out[-3500:]}"
        # )

        match = re.search(r"^\s*1\s+([\d.]+)", out, re.MULTILINE)
        self.assertIsNotNone(match, f"no results in benchmark output:\n{out[-2000:]}")
        best_tflops = float(match.group(1))
        print("Best tflops:", best_tflops)

        ref = self.baseline["workloads"][baseline_key]
        limit = float(ref["baseline"]) * float(ref["min_fraction"])
        self.assertGreaterEqual(
            best_tflops,
            limit,
            f"{baseline_key} best_tflops regressed: {best_tflops:.6g} < {limit:.6g} "
            f"(baseline {ref['baseline']:.6g}, min_fraction {ref['min_fraction']})",
        )

    def test_conv_bf16(self):
        self._verify_and_sweep("bf16", "conv_fwd_bf16_gfx950_N8H56W56C64K64R3S3")

    def test_conv_fp32(self):
        self._verify_and_sweep("fp32", "conv_fwd_fp32_gfx950_N8H56W56C64K64R3S3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
