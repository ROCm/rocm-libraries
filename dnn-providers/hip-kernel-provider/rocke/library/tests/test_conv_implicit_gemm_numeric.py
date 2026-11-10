# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Implicit-GEMM convolution numeric gate on whatever gfx device this runs on.

Builds the conv bake-off with ``python -m builders.common.bake_off_implicit_gemm``
and verifies its first hsaco with ``python -m rocke.run_manifest --verify``
against the numpy fp32 reference. It moved here from
``TestNumericVerification.test_conv_numeric`` in
``platform/tests/instances/test_rocke_numeric.py`` because the builder is a
library module (the dependency rule is ``library -> platform`` only); the test
name and assertions are unchanged.

PYTHONPATH for the subprocesses is derived from this file's location so the
test is portable:
- ``library/`` (exposes ``builders.*``, ``kernels.*``, ``dispatch.*``)
- ``platform/python`` (exposes ``rocke.*``)

Requires a ROCm GPU and torch; skipped otherwise. Run it as a script, not under
pytest, in a process that has touched HIP:
    python3 rocke/library/tests/test_conv_implicit_gemm_numeric.py
"""

from __future__ import annotations

import glob
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import pytest

from rocke.runtime.hip_module import get_device_arch

pytestmark = pytest.mark.gpu

_LIBDIR = pathlib.Path(__file__).resolve().parents[1]  # rocke/library
_PYDIR = pathlib.Path(__file__).resolve().parents[2] / "platform" / "python"
_SUBPROC_PYTHONPATH = os.pathsep.join([str(_PYDIR), str(_LIBDIR)])


# Detect arch via the rocke HIP runtime (no torch). The verify subprocess imports
# torch for its numeric reference, so also gate on torch being importable (a
# dependency check, not a device probe) -- a torch-free env then skips cleanly.
ARCH = get_device_arch(0)
_CDNA = ARCH in ("gfx942", "gfx950")  # MFMA targets; gfx1151 is RDNA/WMMA
_HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(
    ARCH and _HAS_TORCH, "needs a ROCm GPU + torch (run under a torch venv)"
)
class TestNumericVerification(unittest.TestCase):
    """Launch + numeric-compare on whatever gfx device this runs on."""

    def _run(self, *cmd, timeout=300):
        env = dict(os.environ)
        env["PYTHONPATH"] = _SUBPROC_PYTHONPATH
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, *cmd],
            cwd=str(_LIBDIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr)

    def test_conv_numeric(self):
        """Implicit-GEMM convolution vs numpy fp32 reference."""
        if not _CDNA:
            self.skipTest(f"CDNA MFMA conv; running on {ARCH} (RDNA)")
        d = tempfile.mkdtemp(prefix=f"rocke_conv_{ARCH}_")
        rc, out = self._run(
            "-m",
            "builders.common.bake_off_implicit_gemm",
            "--arch",
            ARCH,
            "--output-dir",
            d,
        )
        self.assertEqual(rc, 0, f"conv build failed on {ARCH}:\n{out[-2500:]}")
        hsaco = glob.glob(os.path.join(d, "*.hsaco"))
        self.assertTrue(hsaco, f"no hsaco produced in {d}")
        rc2, out2 = self._run(
            "-m",
            "rocke.run_manifest",
            hsaco[0],
            os.path.join(d, "manifest.json"),
            "--verify",
        )
        self.assertEqual(rc2, 0, f"conv verify failed on {ARCH}:\n{out2[-2500:]}")
        self.assertIn("bad=0", out2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
