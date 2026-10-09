# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The on-device BF16x3 numeric gate, run by the suite (XF32 plan, step 9).

``rocke/examples/gfx950/xf32/bf16x3_parity.py`` is the only check that runs the
expansion on silicon: bitwise against a hand-written BF16x3 kernel, and for
accuracy against ``K * alpha * beta``. As an examples-only ``__main__`` script
neither pytest nor ``tests/run_all.py`` would ever invoke it, so the device
assertions would not run even on a gfx950 host. This test runs it there.

**Every visible device is examined**, not just device 0, and the first gfx950
among them is handed to the harness through the visibility env var. A node
whose gfx950 sits at index 3 behind two other cards would otherwise skip the
only gate that exercises the emulation.

It skips only on a *confirmed* absence: the probe ran, and no visible device is
a gfx950 (the emulated rows being gfx950-only). A probe that fails -- a broken
install, a wedged driver, an import error -- is a failure, not a skip, because
it establishes nothing about what hardware is present. ``get_device_count()``
answers 0 rather than raising when HIP itself is missing, so a host with no
ROCm at all still skips rather than failing.

``ROCKE_REQUIRE_GFX950=1`` turns even the confirmed-absence skip into a
failure, so a validation node that lost its device cannot report green.

The device probe and the harness each run in a subprocess, which keeps HIP
initialisation out of the pytest process and bounds a hung device.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

_ARCH = "gfx950"
_REQUIRE_ENV = "ROCKE_REQUIRE_GFX950"
_PYROOT = Path(__file__).resolve().parents[2] / "python"
_HARNESS = _PYROOT / "rocke" / "examples" / "gfx950" / "xf32" / "bf16x3_parity.py"

# One "<index> <arch>" line per visible device; no output means none are
# visible. Both queries answer None/0 instead of raising when the runtime is
# absent, so a non-zero exit here is a real fault rather than a bare machine.
_PROBE = (
    "from rocke.runtime.hip_module import get_device_arch, get_device_count\n"
    "for i in range(get_device_count()):\n"
    "    print(i, get_device_arch(i))\n"
)

# HIP reads the first of these that is set; the chosen index must be resolved
# against that same list, since the indices a filtered process sees are
# positions within it rather than physical device numbers.
_VISIBILITY_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_PYROOT), env.get("PYTHONPATH")) if p
    )
    return env


def _visible_devices(env: dict):
    """``[(index, arch)]`` for every visible device.

    Raises :class:`AssertionError` if the probe itself fails -- see the module
    docstring on why that is not a skip.
    """
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if probe.returncode != 0:
        raise AssertionError(
            "the HIP device probe failed, so device presence is unknown and a "
            f"skip would be unsound (exit {probe.returncode}):\n"
            f"{(probe.stderr or probe.stdout).strip()}"
        )
    found = []
    for line in probe.stdout.strip().splitlines():
        index, _, arch = line.partition(" ")
        found.append((int(index), arch.strip()))
    return found


def _select(env: dict, index: int) -> dict:
    """``env`` with the chosen device made device 0 for the child process."""
    for var in _VISIBILITY_VARS:
        current = env.get(var)
        if current:
            # Already filtered: `index` is a position in this list, not a
            # device number, so hand back the entry rather than the index.
            entries = [e for e in current.split(",") if e.strip()]
            return dict(env, **{var: entries[index]})
    return dict(env, HIP_VISIBLE_DEVICES=str(index))


class TestBf16x3OnDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = _env()
        visible = _visible_devices(env)
        for index, arch in visible:
            if arch == _ARCH:
                cls.env = _select(env, index)
                return
        seen = ", ".join(f"{i}:{a}" for i, a in visible) or "none"
        reason = f"no visible {_ARCH} device (visible: {seen})"
        if env.get(_REQUIRE_ENV) == "1":
            raise AssertionError(f"{_REQUIRE_ENV}=1, but {reason}")
        raise unittest.SkipTest(reason)

    def test_the_expansion_is_faithful_and_accurate_on_device(self):
        # --require-device: the probe above saw a gfx950 and this process
        # selected it, so a harness that cannot find one is broken and must
        # not exit 0 as a skip.
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
