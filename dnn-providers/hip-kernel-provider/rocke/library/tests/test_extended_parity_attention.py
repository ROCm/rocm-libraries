# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention parity gate for the library attention harness.

Spawns ``python -m builders.common.parity_fmha_extended --arch ARCH`` and
checks that all cases pass (or legitimately skip on gfx942).  This mirrors
the structure of :class:`TestNumericVerification.test_extended_parity` in
``platform/tests/instances/test_rocke_numeric.py`` but targets the library
attention harness which is now the correct home for FMHA / Sage / Sparse
kernel parity checks (those kernels import from ``kernels`` — library layer).

PYTHONPATH is derived from this file's location so the test is portable:
- ``library/`` (exposes ``builders.*``, ``kernels.*``, ``dispatch.*``)
- ``platform/Python`` (exposes ``rocke.*``)
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest

_LIBDIR = pathlib.Path(__file__).resolve().parents[1]  # rocke/library
_PYDIR = pathlib.Path(__file__).resolve().parents[2] / "platform" / "Python"
_SUBPROC_PYTHONPATH = os.pathsep.join([str(_PYDIR), str(_LIBDIR)])


def _detect_gpu_arch():
    """(has_gpu, gfx) via torch — imported FIRST so rocke binds torch's HIP."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None
        name = torch.cuda.get_device_properties(0).gcnArchName
        return True, name.split(":", 1)[0]
    except Exception:
        return False, None


GPU, ARCH = _detect_gpu_arch()
_CDNA = ARCH in ("gfx942", "gfx950")


@unittest.skipUnless(ARCH and GPU, "needs a ROCm GPU + torch (run under a torch venv)")
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
