# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU **numeric** verification suite, parametrized by the running device's arch.

Unlike ``test_rocke.py`` (build/IR/structure smoke tests) and
``test_rocke_multiarch.py`` (pure-Python arch metadata), this module actually
*launches* kernels on the GPU and compares results against numpy/torch
references — so it certifies numeric correctness on the gfx target it runs on.
Run it on a gfx942 node and on a gfx950 node to cover both architectures.

It drives the arch-aware parity harnesses (which exit non-zero on any tolerance
breach):
  - rocke.examples.common.ck_tile_parity         : elementwise(×12), reduce(×3),
                                             layernorm, rmsnorm, transpose2d,
                                             batched + grouped GEMM  (BOTH arches)
  - rocke.examples.common.universal_gemm_verify : Universal GEMM vs A@B.T (BOTH)
  - rocke.examples.common.bake_off_implicit_gemm + run_manifest --verify : conv (BOTH)
  - rocke.examples.common.parity_extended_kernels: fmha/attention/sage/sparse/streamk/
                                             mfma/block_scale/mx  (gfx950 today;
                                             see test_extended_parity for the
                                             gfx942 status)

Requires a ROCm GPU and torch (e.g. a dedicated ~/.venv); skipped otherwise.
Run:  PYTHONPATH=Python <torch-python> tests/instances/test_rocke_numeric.py
"""

from __future__ import annotations

import glob
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

_PYDIR = pathlib.Path(__file__).resolve().parents[2] / "Python"  # rocKE/Python


def _detect_gpu_arch():
    """(has_gpu, gfx) via torch — imported FIRST so rocke binds torch's HIP.

    Using torch (already required by the parity harnesses) avoids a HIP
    double-init: if rocke's HIP is loaded before torch, torch.cuda can fail
    to see the device. ``gcnArchName`` is like ``gfx942:sramecc+:xnack-``.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None
        name = torch.cuda.get_device_properties(0).gcnArchName
        return True, name.split(":", 1)[0]
    except Exception:
        return False, None


GPU, ARCH = _detect_gpu_arch()
_CDNA = ARCH in ("gfx942", "gfx950")  # MFMA targets; gfx1151 is RDNA/WMMA


@unittest.skipUnless(ARCH and GPU, "needs a ROCm GPU + torch (run under a torch venv)")
class TestNumericVerification(unittest.TestCase):
    """Launch + numeric-compare on whatever gfx device this runs on."""

    def _run(self, *cmd, timeout=300):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_PYDIR)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, *cmd],
            cwd=str(_PYDIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr)

    def test_core_parity(self):
        """Elementwise/reduce/norm/transpose/batched+grouped GEMM vs torch+numpy."""
        if not _CDNA:
            self.skipTest(f"CDNA/MFMA parity harness; running on {ARCH} (RDNA)")
        rc, out = self._run(
            "-m", "rocke.examples.common.ck_tile_parity", "--arch", ARCH
        )
        self.assertEqual(
            rc, 0, f"ck_tile_parity numeric failure on {ARCH}:\n{out[-2500:]}"
        )
        self.assertNotIn("FAIL", out)

    def test_gemm_numeric(self):
        """Universal GEMM result vs numpy A@B.T (RCR)."""
        if not _CDNA:
            self.skipTest(
                f"CDNA MFMA GEMM; running on {ARCH} (RDNA) — see test_wmma_gemm"
            )
        rc, out = self._run(
            "-m",
            "rocke.examples.common.universal_gemm_verify",
            "--arch",
            ARCH,
            "--m",
            "512",
            "--n",
            "512",
            "--k",
            "512",
        )
        self.assertEqual(rc, 0, f"gemm verify failed on {ARCH}:\n{out[-2500:]}")
        self.assertIn("bad=0", out)

    def test_conv_numeric(self):
        """Implicit-GEMM convolution vs numpy fp32 reference."""
        if not _CDNA:
            self.skipTest(f"CDNA MFMA conv; running on {ARCH} (RDNA)")
        d = tempfile.mkdtemp(prefix=f"rocke_conv_{ARCH}_")
        rc, out = self._run(
            "-m",
            "rocke.examples.common.bake_off_implicit_gemm",
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

    def test_extended_parity(self):
        """fmha/attention/sage/sparse/streamk/mfma/block_scale/mx vs torch.

        Runs on BOTH arches now that the extended ``build_*`` entrypoints
        thread ``arch`` into their MFMA-atom selection. On gfx950 every case
        runs and verifies numerically (24/24 PASS, 0 SKIP). On gfx942 the
        builders raise a clean ValueError/NotImplementedError for any kernel
        that genuinely needs a gfx950-only atom; the harness reports those as
        SKIP (not FAIL) and keeps running. Either way the gate is: rc == 0 and
        no ``FAIL`` line in the output (skips are allowed on gfx942).
        """
        if not _CDNA:
            self.skipTest(f"CDNA MFMA extended kernels; running on {ARCH} (RDNA)")
        rc, out = self._run(
            "-m",
            "rocke.examples.common.parity_extended_kernels",
            "--arch",
            ARCH,
            timeout=420,
        )
        self.assertEqual(rc, 0, f"extended parity failure on {ARCH}:\n{out[-3000:]}")
        # No FAIL line anywhere: every case either PASSed or legitimately
        # SKIPped (gfx950-only on gfx942). A FAIL means a numeric breach.
        self.assertNotIn(
            "FAIL", out, f"extended parity has FAIL on {ARCH}:\n{out[-3000:]}"
        )
        self.assertIn("pass", out)

    def test_wmma_gemm(self):
        """gfx1151 (RDNA3.5) WMMA GEMM vs numpy A@B.T, within fp tolerance.

        RDNA-only: WMMA accumulates in f32 in a different order than numpy, so
        the gate is max_abs_diff within tolerance (1e-2), not exact parity.
        """
        if ARCH != "gfx1151":
            self.skipTest(f"gfx1151/WMMA GEMM; running on {ARCH}")
        rc, out = self._run(
            "-m",
            "rocke.examples.gfx1151.wmma_gemm_verify",
            "--m",
            "128",
            "--n",
            "128",
            "--k",
            "128",
        )
        self.assertEqual(rc, 0, f"WMMA GEMM failed on {ARCH}:\n{out[-2500:]}")
        self.assertIn("PASS", out)

    def test_rdna_core_parity(self):
        """gfx1151 (RDNA3.5) non-MFMA parity vs torch+numpy.

        The ``ck_tile_parity`` harness runs the wave32-aware kernels and
        verifies them numerically: elementwise(×12), reduce(×3), layernorm2d,
        rmsnorm2d, transpose2d. The MFMA GEMM cases (batched/grouped) are
        SKIPped by the harness on a non-CDNA arch (RDNA uses WMMA, not MFMA),
        so they are out of scope here and covered by ``test_wmma_gemm``.

        The gate mirrors the CDNA ``test_core_parity``: rc == 0 and no ``FAIL``
        line in the output. ``SKIP`` is allowed (the MFMA-only cases).
        """
        if ARCH != "gfx1151":
            self.skipTest(f"gfx1151/RDNA non-MFMA parity; running on {ARCH}")
        rc, out = self._run(
            "-m", "rocke.examples.common.ck_tile_parity", "--arch", "gfx1151"
        )
        self.assertEqual(
            rc, 0, f"ck_tile_parity numeric failure on gfx1151:\n{out[-2500:]}"
        )
        self.assertNotIn("FAIL", out)
        # Sanity: the non-MFMA ops must actually have run (and PASSed), not all
        # been skipped — at minimum elementwise/reduce/norm/transpose are OK.
        self.assertIn("OK", out)

    def test_elementwise_hip_numeric(self):
        """HIP-C++ (hipcc) non-MMA elementwise-add vs numpy, every arch.

        Closes the non-MMA HIP-path numeric gate on all three arches. The
        elementwise builder has no MMA atom (just loads / vec pack / f32
        promote / store), so the same kernel runs on CDNA wave64
        (gfx942/gfx950) and RDNA wave32 (gfx1151). This is the wave32 HIP-path
        non-MMA coverage that complements the LLVM-path
        ``test_rdna_core_parity`` and the WMMA-path ``test_wmma_gemm`` on
        gfx1151.
        """
        rc, out = self._run(
            "-m",
            "rocke.examples.common.elementwise_verify_hip",
            "--arch",
            ARCH,
            timeout=300,
        )
        self.assertEqual(
            rc, 0, f"HIP-path elementwise failed on {ARCH}:\n{out[-2500:]}"
        )
        self.assertIn("PASS", out)
        self.assertIn("bad=0", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
