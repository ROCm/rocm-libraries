# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU smoke tests for rocke performance-sensitive paths.

The CI contract is that GPU jobs invoke this file from ``python/test``. The
actual workloads remain the existing example/benchmark modules for attention,
GEMM, and fused-MoE.

Run on a gfx950 ROCm runner:
  HIP_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Python \
    python tests/instances/test_rocke_gfx950_smoke.py
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROCKE = Path(__file__).resolve().parents[2]  # instances -> tests -> rocKE
_PY_ROOT = _ROCKE / "Python"
_DEFAULT_BASELINE = _ROCKE / "tests" / "golden" / "rocke_gfx950_smoke_perf.json"
_DEFAULT_REPORT = Path("/tmp/rocke_gfx950_smoke_perf_current.json")


def _detect_gpu_arch() -> tuple[bool, str | None, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None, "torch reports no ROCm GPU"
        props = torch.cuda.get_device_properties(0)
        arch = props.gcnArchName.split(":", 1)[0]
        return True, arch, torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, None, f"torch GPU detection failed: {exc!r}"


GPU_AVAILABLE, GPU_ARCH, GPU_REASON = _detect_gpu_arch()


@unittest.skipUnless(
    GPU_AVAILABLE and GPU_ARCH == "gfx950",
    f"needs a gfx950 ROCm GPU; detected {GPU_ARCH or GPU_REASON}",
)
class TestRockeGfx950Smoke(unittest.TestCase):
    maxDiff = 4000
    current_perf: dict[str, dict] = {}
    baseline = json.loads(
        Path(
            os.environ.get("ROCKE_GFX950_PERF_BASELINE", _DEFAULT_BASELINE)
        ).read_text()
    )

    @classmethod
    def tearDownClass(cls):
        if not cls.current_perf:
            return
        report_path = Path(os.environ.get("ROCKE_GFX950_PERF_REPORT", _DEFAULT_REPORT))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "ck.dsl.gfx950_smoke_perf.current/v1",
            "baseline": str(
                Path(os.environ.get("ROCKE_GFX950_PERF_BASELINE", _DEFAULT_BASELINE))
            ),
            "device_arch": GPU_ARCH,
            "device": GPU_REASON,
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

    def test_attention_decode_smoke(self):
        with tempfile.TemporaryDirectory(prefix="rocke_attn_gfx950_") as tmp:
            report = Path(tmp) / "attention.json"
            out = self._run(
                "-m",
                "rocke.examples.gfx950.attention.parity_unified_attention",
                "--scenario",
                "decode_d128_b16",
                "--attempts",
                "2",
                "--warmup",
                "1",
                "--skip-triton",
                "--paths",
                "auto",
                "--report",
                str(report),
                timeout=900,
            )
            rows = json.loads(report.read_text())
        self.assertIn("ck-auto", out)
        self.assertNotIn("FAIL", out)
        self._record_and_compare(
            "attention_decode_d128_b16",
            {
                "scenario": "decode_d128_b16",
                "ck_auto_ms": float(rows[0]["ck_auto_ms"]),
                "max_abs": float(rows[0]["ck_auto_vs_ref"]["max_abs"]),
            },
        )

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
