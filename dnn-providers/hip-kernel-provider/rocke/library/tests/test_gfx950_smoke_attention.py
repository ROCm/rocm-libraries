# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU smoke test for the library unified-attention decode path.

Moved out of platform/tests/instances/test_rocke_gfx950_smoke.py: this lane
invokes the library module ``builders.gfx950.attention.parity_unified_attention``,
so it belongs in the library test tree (platform must never reference the moved
library). It shares the committed gfx950 perf baseline, located via the
sanctioned ``rocke.assets`` accessor.

Run on a gfx950 ROCm runner:
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=rocke/platform/python:rocke/library \
    python rocke/library/tests/test_gfx950_smoke_attention.py
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rocke.assets import platform_root
from rocke.runtime.hip_module import get_device_arch, get_device_name

_LIBROOT = Path(__file__).resolve().parents[1]  # tests -> rocke/library
_PY_ROOT = platform_root() / "python"
_DEFAULT_BASELINE = (
    importlib.resources.files("rocke.golden") / "rocke_gfx950_smoke_perf.json"
)


# GPU/arch via the rocke HIP runtime (no torch); get_device_name is the rocminfo
# "Marketing Name". The attention parity subprocess imports torch for its numeric
# reference, so also gate on torch being importable (a dependency check, not a device
# probe) — a torch-free env then skips cleanly instead of hitting an ImportError.
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
class TestGfx950AttentionSmoke(unittest.TestCase):
    maxDiff = 4000
    baseline = json.loads(
        Path(
            os.environ.get("ROCKE_GFX950_PERF_BASELINE", _DEFAULT_BASELINE)
        ).read_text()
    )

    def _run(self, *args: str, timeout: int = 600) -> str:
        env = dict(os.environ)
        # The attention example lives in the library (`builders`) and imports the
        # platform SDK (`rocke.*`), so the subprocess needs BOTH roots on the path.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_PY_ROOT), str(_LIBROOT), env.get("PYTHONPATH", "")]
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=str(_LIBROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, out[-3500:])
        return out

    def _compare(self, name: str, metrics: dict[str, float | int | str]):
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

    def _run_scenario(self, scenario: str, timeout: int = 900) -> tuple[str, list]:
        with tempfile.TemporaryDirectory(prefix="rocke_attn_gfx950_") as tmp:
            report = Path(tmp) / "attention.json"
            out = self._run(
                "-m",
                "builders.gfx950.attention.parity_unified_attention",
                "--scenario",
                scenario,
                "--attempts",
                "2",
                "--warmup",
                "1",
                "--skip-triton",
                "--paths",
                "auto",
                "--report",
                str(report),
                timeout=timeout,
            )
            rows = json.loads(report.read_text())
        return out, rows

    def test_attention_decode_smoke(self):
        with tempfile.TemporaryDirectory(prefix="rocke_attn_gfx950_") as tmp:
            report = Path(tmp) / "attention.json"
            out = self._run(
                "-m",
                "builders.gfx950.attention.parity_unified_attention",
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
        self._compare(
            "attention_decode_d128_b16",
            {
                "scenario": "decode_d128_b16",
                "ck_auto_ms": float(rows[0]["ck_auto_ms"]),
                "max_abs": float(rows[0]["ck_auto_vs_ref"]["max_abs"]),
            },
        )

    def test_attention_decode_bias_smoke(self):
        """Correctness check for decode with softcap + ALiBi + QQ-bias (fp16 and bf16)."""
        for scenario in ("decode_bias_d128_b16", "decode_bias_bf16_d128_b16"):
            with self.subTest(scenario=scenario):
                out, rows = self._run_scenario(scenario)
                self.assertTrue(rows, f"{scenario}: empty report")
                self.assertIn("ck-auto", out, f"{scenario}: DSL path not exercised")
                self.assertNotIn("FAIL", out, f"{scenario}: correctness failure")
                for row in rows:
                    max_abs = float(row["ck_auto_vs_ref"]["max_abs"])
                    self.assertLess(
                        max_abs,
                        5e-2,
                        f"{scenario} seq={row.get('seq_lens')}: "
                        f"max_abs {max_abs:.3e} exceeds 5e-2",
                    )

    def test_attention_combo_bias_smoke(self):
        """Correctness check for 2D-combo prefill with softcap + ALiBi + QQ-bias."""
        scenario = "combo_bias_bf16_d64_b32_gqa8_64x8"
        out, rows = self._run_scenario(scenario)
        self.assertTrue(rows, f"{scenario}: empty report")
        self.assertIn("ck-auto", out, f"{scenario}: DSL path not exercised")
        self.assertNotIn("FAIL", out, f"{scenario}: correctness failure")
        for row in rows:
            max_abs = float(row["ck_auto_vs_ref"]["max_abs"])
            self.assertLess(
                max_abs,
                5e-2,
                f"{scenario} seq={row.get('seq_lens')}: "
                f"max_abs {max_abs:.3e} exceeds 5e-2",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
