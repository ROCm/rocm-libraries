# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention parity gate for the library attention harness.

Spawns ``python -m builders.common.parity_fmha_extended --arch ARCH`` and
checks that all cases pass (or legitimately skip on gfx942). This mirrors
the structure of :class:`TestNumericVerification.test_extended_parity` in
``platform/tests/instances/test_rocke_numeric.py`` but targets the library
attention harness which is now the correct home for FMHA / Sage / Sparse
kernel parity checks (those kernels import from ``kernels`` — library layer).

The harness imports the ``rocke`` package this test imports, with ``library/``
(``builders.*``, ``kernels.*``, ``dispatch.*``) on its path
(``rocke.tests.subprocess_launcher``), in a source checkout and in the installed
test tree alike.
"""

from __future__ import annotations

import pathlib
import subprocess
import unittest

from rocke.runtime.hip_module import get_device_arch
from rocke.tests.subprocess_launcher import (
    rocke_python_command,
    rocke_subprocess_env,
)

_LIBDIR = pathlib.Path(__file__).resolve().parents[1]  # the library root


# Detect arch via the rocke HIP runtime. The parity harness subprocess computes
# numpy references.
ARCH = get_device_arch(0)
_CDNA = ARCH in ("gfx942", "gfx950")


@unittest.skipUnless(ARCH, "needs a ROCm GPU")
class TestAttentionParityLibrary(unittest.TestCase):
    """Launch the library attention parity harness and check all cases pass."""

    def _run(self, *cmd, timeout=420):
        env = rocke_subprocess_env(python_paths=[_LIBDIR])
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        r = subprocess.run(
            rocke_python_command(*cmd),
            check=False,
            cwd=str(_LIBDIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return r.returncode, (r.stdout + r.stderr)

    def test_extended_parity_attention(self):
        """FMHA/sage/sparse/mfma-fwd/bwd attention cases vs the numpy reference.

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
