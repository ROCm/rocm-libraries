# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU smoke tests for the implicit-GEMM convolution benchmark.

Tests bf16 and fp32 dtypes for both forward (fwd) and backward-weight (wgrad)
directions. Each test method runs the benchmark sweep with --verify (the
benchmark itself prints PASS/FAIL per kernel) and checks TFLOPS against the
committed baseline in rocke_gfx950_smoke_perf.json.

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

import importlib.resources

from rocke.runtime.hip_module import get_device_arch, get_device_name

_DEFAULT_BASELINE = (
    importlib.resources.files("rocke.golden") / "rocke_gfx950_smoke_perf.json"
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
        Path(os.environ["ROCKE_GFX950_PERF_BASELINE"]).read_text()
        if "ROCKE_GFX950_PERF_BASELINE" in os.environ
        else _DEFAULT_BASELINE.read_text()
    )

    def _run_benchmark(
        self,
        dtype: str,
        direction: str = "fwd",
        timeout: int = 600,
        extra_args: list | None = None,
    ) -> str:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [
            sys.executable,
            "-m",
            "rocke.benchmark.benchmark_implicit_gemm_conv",
            "--arch",
            "gfx950",
            "--dtype",
            dtype,
            "--direction",
            direction,
            "--sample",
            "0.05",
            "--warmup",
            "3",
            "--iters",
            "5",
            "--verify",
        ]
        if direction == "wgrad":
            cmd += ["--split-k", "-1"]
        if extra_args:
            cmd += extra_args
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, out[-3500:])
        return out

    _POINTWISE_ARGS = [
        "--Y",
        "1",
        "--X",
        "1",
        "--pH",
        "0",
        "--pW",
        "0",
        "--sH",
        "1",
        "--sW",
        "1",
    ]

    def _verify_and_sweep(
        self,
        dtype: str,
        baseline_key: str,
        direction: str = "fwd",
        extra_args: list | None = None,
    ):
        out = self._run_benchmark(dtype, direction=direction, extra_args=extra_args)

        self.assertNotIn(
            "FAIL", out, f"conv {dtype} correctness failure:\n{out[-3500:]}"
        )

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

    def test_conv_wgrad_bf16(self):
        self._verify_and_sweep(
            "bf16", "conv_wgrad_bf16_gfx950_N8H56W56C64K64R3S3", direction="wgrad"
        )

    def test_conv_wgrad_fp32(self):
        self._verify_and_sweep(
            "fp32", "conv_wgrad_fp32_gfx950_N8H56W56C64K64R3S3", direction="wgrad"
        )

    def test_conv_pointwise_bf16(self):
        self._verify_and_sweep(
            "bf16",
            "conv_fwd_bf16_gfx950_N8H56W56C64K64R1S1",
            extra_args=self._POINTWISE_ARGS,
        )

    # TODO: Reenable after threshold calculation implementation
    # def test_conv_pointwise_wgrad_bf16(self):
    #     self._verify_and_sweep(
    #         "bf16",
    #         "conv_wgrad_bf16_gfx950_N8H56W56C64K64R1S1",
    #         direction="wgrad",
    #         extra_args=self._POINTWISE_ARGS,
    #     )


if __name__ == "__main__":
    unittest.main(verbosity=2)
