# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU smoke test for the library unified-attention decode path.

Moved out of platform/tests/instances/test_rocke_gfx950_smoke.py: this lane
invokes the library module ``builders.gfx950.attention.parity_unified_attention``,
so it belongs in the library test tree (platform must never reference the moved
library). It shares the committed gfx950 perf baseline, located via the
sanctioned ``rocke.assets`` accessor.

Run on a gfx950 ROCm runner:
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=rocke/platform/Python:rocke/library \
    python rocke/library/tests/test_gfx950_smoke_attention.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rocke.assets import platform_root

_LIBROOT = Path(__file__).resolve().parents[1]  # tests -> rocke/library
_PY_ROOT = platform_root() / "Python"
_DEFAULT_BASELINE = (
    platform_root() / "tests" / "golden" / "rocke_gfx950_smoke_perf.json"
)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
