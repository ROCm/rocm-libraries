# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The on-device BF16x3 numeric gate, run by the suite (XF32 plan, step 9).

``rocke/examples/gfx950/xf32/bf16x3_parity.py`` is the only check that runs the
expansion on silicon: bitwise against a hand-written BF16x3 kernel, and for
accuracy against ``K * alpha * beta``. As an examples-only ``__main__`` script
neither pytest nor ``tests/run_all.py`` would ever invoke it, so the device
assertions would not run even on a gfx950 host. This test runs it there.

Elsewhere it skips: the emulated rows are gfx950-only. The harness allocates
through torch, so a host without torch skips too. Set ``ROCKE_REQUIRE_GFX950=1``
on a gfx950 validation node to turn either skip into a failure, so a node that
lost its device or its torch cannot report green.

The device probe and the harness each run in a subprocess, which keeps HIP
initialisation out of the pytest process and bounds a hung device.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

_ARCH = "gfx950"
_REQUIRE_ENV = "ROCKE_REQUIRE_GFX950"
_PYROOT = Path(__file__).resolve().parents[2] / "python"
_HARNESS = _PYROOT / "rocke" / "examples" / "gfx950" / "xf32" / "bf16x3_parity.py"
_PROBE = "import rocke.runtime.hip_module as hip; print(hip.get_device_arch(0))"


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_PYROOT), env.get("PYTHONPATH")) if p
    )
    return env


def _device_arch(env: dict) -> str:
    """HIP device 0's base arch, or a description of why there is none."""
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if probe.returncode != 0:
        tail = (probe.stderr or probe.stdout).strip().splitlines()[-1:]
        return f"<probe failed: {tail[0] if tail else probe.returncode}>"
    lines = probe.stdout.strip().splitlines()
    return lines[-1] if lines else "<no output>"


class TestBf16x3OnDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        arch = _device_arch(cls.env)
        if arch != _ARCH:
            reason = f"needs HIP device 0 to be {_ARCH}; found {arch}"
        elif importlib.util.find_spec("torch") is None:
            reason = "the device harness allocates through torch, which is absent"
        else:
            return
        if cls.env.get(_REQUIRE_ENV) == "1":
            raise AssertionError(f"{_REQUIRE_ENV}=1, but {reason}")
        raise unittest.SkipTest(reason)

    def test_the_expansion_is_faithful_and_accurate_on_device(self):
        # --require-device: the probe above saw a gfx950, so a harness that
        # cannot find one is broken and must not exit 0 as a skip.
        done = subprocess.run(
            [sys.executable, str(_HARNESS), "--require-device"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PASSED", done.stdout)


if __name__ == "__main__":
    unittest.main()
