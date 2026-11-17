# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention parity gate for the library attention harness.

Spawns ``python -m builders.common.parity_fmha_extended --arch ARCH`` and
checks that all cases pass (or legitimately skip on gfx942). This mirrors
the structure of :class:`TestNumericVerification.test_extended_parity` in
``platform/tests/instances/test_rocke_numeric.py`` but targets the library
attention harness which is now the correct home for FMHA / Sage / Sparse
kernel parity checks (those kernels import from ``kernels`` — library layer).

PYTHONPATH is derived from this file's location so the test is portable:
- ``library/`` (exposes ``builders.*``, ``kernels.*``, ``dispatch.*``)
- ``platform/python`` (exposes ``rocke.*``)
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest

from rocke.runtime.hip_module import get_device_arch

_LIBDIR = pathlib.Path(__file__).resolve().parents[1]  # rocke/library
_PYDIR = pathlib.Path(__file__).resolve().parents[2] / "platform" / "python"
_SUBPROC_PYTHONPATH = os.pathsep.join([str(_PYDIR), str(_LIBDIR)])


# Detect arch via the rocke HIP runtime (no torch). The parity harness subprocess
# imports torch for its numeric reference, so also gate on torch being importable (a
# dependency check, not a device probe) — a torch-free env then skips cleanly instead of
# hitting an ImportError in the body.
ARCH = get_device_arch(0)
_CDNA = ARCH in ("gfx942", "gfx950")
_HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(ARCH and _HAS_TORCH, "needs a ROCm GPU + torch")
class TestAttentionParityLibrary(unittest.TestCase):
    """Launch the library attention parity harness and check all cases pass."""

    def _run(self, *cmd, timeout=420):
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

    def test_extended_parity_attention(self):
        """FMHA/sage/sparse/mfma-fwd/bwd attention cases vs torch reference.

        Runs on BOTH arches: on gfx950 every case runs and verifies
        numerically.  On gfx942 the builders raise a clean
        ValueError/NotImplementedError for any kernel that needs a
        gfx950-only atom; the harness reports those as SKIP (not FAIL)
        and keeps running. Gate: rc == 0 and no ``FAIL`` line anywhere.
        """

        if not _CDNA:
            self.skipTest(f"CDNA MFMA attention kernels; running on {ARCH} (RDNA)")
        rc, out = self._run(
            "-m",
            "builders.common.parity_fmha_extended",
            "--arch",
            ARCH,
        )
        self.assertEqual(rc, 0, f"attention parity failure on {ARCH}:\n{out[-3000:]}")
        # No FAIL line: every case either PASSed or legitimately SKIPped.
        self.assertNotIn(
            "FAIL", out, f"attention parity has FAIL on {ARCH}:\n{out[-3000:]}"
        )
        self.assertIn("pass", out)


if __name__ == "__main__":
    unittest.main()
