# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU smoke tests for rocke performance-sensitive paths.

The CI contract is that GPU jobs invoke this file from ``python/test``. The
actual workloads remain the existing example/benchmark modules for attention,
GEMM, and fused-MoE.

Run on a gfx950 ROCm runner:
  HIP_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
    python tests/instances/test_rocke_gfx950_smoke.py
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rocke.runtime.hip_module import get_device_arch, get_device_name

_ROCKE = Path(__file__).resolve().parents[2]  # instances -> tests -> rocKE
_PY_ROOT = _ROCKE / "python"
_DEFAULT_BASELINE = (
    importlib.resources.files("rocke.golden") / "rocke_gfx950_smoke_perf.json"
)
_DEFAULT_REPORT = Path("/tmp/rocke_gfx950_smoke_perf_current.json")


# GPU/arch via the rocke HIP runtime (no torch); get_device_name is the rocminfo
# "Marketing Name". One smoke workload (fused_moe_e2e_perf) imports torch, so also gate
# on torch being importable (a dependency check, not a device probe) — a torch-free env
# then skips cleanly instead of hitting an ImportError.
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
class TestRockeGfx950Smoke(unittest.TestCase):
    maxDiff = 4000
    current_perf: dict[str, dict] = {}
    baseline = json.loads(
        Path(os.environ["ROCKE_GFX950_PERF_BASELINE"]).read_text()
        if "ROCKE_GFX950_PERF_BASELINE" in os.environ
        else _DEFAULT_BASELINE.read_text()
    )

    @classmethod
    def tearDownClass(cls):
        if not cls.current_perf:
            return
        report_path = Path(os.environ.get("ROCKE_GFX950_PERF_REPORT", _DEFAULT_REPORT))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "ck.dsl.gfx950_smoke_perf.current/v1",
            "baseline": os.environ.get(
                "ROCKE_GFX950_PERF_BASELINE", str(_DEFAULT_BASELINE)
            ),
            "device_arch": GPU_ARCH,
            "device": GPU_NAME,
            "workloads": cls.current_perf,
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote gfx950 perf report to {report_path}")

    def _run(self, *args: str, timeout: int = 600) -> str:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_PY_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=str(_ROCKE),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, out[-3500:])
        return out

    def _record_and_compare(self, name: str, metrics: dict[str, float | int | str]):
        self.current_perf[name] = metrics
        ref = self.baseline["workloads"][name]
        metric = ref["metric"]
        current = float(metrics[metric])
        baseline = float(ref["baseline"])

        if ref["direction"] == "lower_is_better":
            limit = baseline * float(ref["max_slowdown"])
            self.assertLessEqual(
                current,
                limit,
                f"{name} {metric} regressed: {current:.6g} > {limit:.6g} "
                f"(baseline {baseline:.6g}, slowdown limit {ref['max_slowdown']}x)",
            )
        elif ref["direction"] == "higher_is_better":
            limit = baseline * float(ref["min_fraction"])
            self.assertGreaterEqual(
                current,
                limit,
                f"{name} {metric} regressed: {current:.6g} < {limit:.6g} "
                f"(baseline {baseline:.6g}, minimum fraction {ref['min_fraction']})",
            )
        else:
            self.fail(f"unknown perf direction {ref['direction']!r} for {name}")

    def test_gemm_sweep_smoke(self):
        with tempfile.TemporaryDirectory(prefix="rocke_gemm_gfx950_") as tmp:
            report = Path(tmp) / "gemm.json"
            out = self._run(
                "-m",
                "rocke.benchmark.gemm.fp16_rcr_sweep",
                "--arch",
                "gfx950",
                "--shape",
                "2048,2048,2048:balanced-large-perf",
                "--output-dir",
                tmp,
                "--json",
                str(report),
                "--compile",
                "--run",
                "--warmup-iters",
                "5",
                "--timed-iters",
                "50",
                "--timeout-s",
                "180",
                timeout=900,
            )
            payload = json.loads(report.read_text())
        self.assertIn("runs", out)
        runs = [r for r in payload["runs"] if r["ok"]]
        self.assertTrue(runs, "GEMM sweep produced no successful runs")
        best = max(runs, key=lambda r: float(r["tflops"]))
        self._record_and_compare(
            "gemm_fp16_rcr_2048x2048x2048",
            {
                "shape": "2048x2048x2048",
                "best_tflops": float(best["tflops"]),
                "best_ms": float(best["ms"]),
                "best_cache_key": best["cache_key"],
                "successful_runs": len(runs),
            },
        )

    def test_fused_moe_smoke(self):
        with tempfile.TemporaryDirectory(prefix="rocke_moe_gfx950_") as tmp:
            report = Path(tmp) / "moe.json"
            out = self._run(
                "-m",
                "rocke.examples.gfx950.moe.fused_moe_e2e_perf",
                "--scenario",
                "small_T32_E4_K2_H128_I256",
                "--attempts",
                "2",
                "--warmup",
                "1",
                "--skip-aiter",
                "--skip-cktile",
                "--report",
                str(report),
                timeout=900,
            )
            rows = json.loads(report.read_text())
        self.assertIn("small_T32_E4_K2_H128_I256", out)
        self.assertNotIn("FAIL", out)
        rocke = rows[0]["results"]["rocke"]
        self._record_and_compare(
            "fused_moe_small_T32_E4_K2_H128_I256",
            {
                "scenario": "small_T32_E4_K2_H128_I256",
                "rocke_ms": float(rocke["ms"]),
                "max_abs": float(rocke["max_abs"]),
                "rel_max": float(rocke["rel_max"]),
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
