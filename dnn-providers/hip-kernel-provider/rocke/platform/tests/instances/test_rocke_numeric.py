# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU **numeric** verification suite, parametrized by the running device's arch.

Unlike ``test_rocke.py`` (build/IR/structure smoke tests) and
``test_rocke_multiarch.py`` (pure-Python arch metadata), this module actually
*launches* kernels on the GPU and compares results against numpy references —
so it certifies numeric correctness on the gfx target it runs on.
Run it on a gfx942 node and on a gfx950 node to cover both architectures.

It drives the arch-aware parity harnesses (which exit non-zero on any tolerance
breach):
  - rocke.examples.common.ck_tile_parity         : elementwise(×12), reduce(×3),
                                             layernorm, rmsnorm, transpose2d,
                                             batched + grouped GEMM  (BOTH arches)
  - rocke.examples.common.universal_gemm_verify : Universal GEMM vs A@B.T (BOTH)
  - rocke.examples.common.parity_extended_kernels: MoE/streamk/mfma/block_scale/mx
                                             (BOTH; see test_extended_parity
                                             for the gfx942 status), and the
                                             fused-MoE forward on its own
                                             (test_moe_e2e_forward, needs torch)

The conv numeric gate (bake_off_implicit_gemm + run_manifest --verify) is in
library/tests/test_conv_implicit_gemm_numeric.py.

Requires a ROCm GPU; skipped otherwise.
Run:  PYTHONPATH=python python tests/instances/test_rocke_numeric.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import unittest

from rocke.runtime.hip_module import get_device_arch
from rocke.tests.lanes import TORCH_IN_CI, skip_needing_arch, skip_until_reason
from rocke.tests.subprocess_launcher import (
    rocke_python_command,
    rocke_subprocess_env,
)

_HERE = pathlib.Path(__file__).resolve()
_LIBRARY_ROOTS = (
    _HERE.parents[3] / "library",  # source tree: rocke/library
    _HERE.parents[1] / "library",  # installed tree: <test-root>/tests/library
)


# Detect arch via the rocke HIP runtime. The parity harnesses run in subprocesses
# (see _run) with numpy references.
ARCH = get_device_arch(0)
_CDNA = ARCH in ("gfx942", "gfx950")  # MFMA targets; gfx1151 is RDNA/WMMA

# The extended-parity case that drives FusedMoeForward, which still takes torch
# tensors: it runs in its own test, so the other cases never wait for torch.
_MOE_E2E_CASE = "moe_e2e_forward"
# Checked, not imported: the harness subprocess imports torch itself.
_HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(ARCH, "needs a ROCm GPU")
class TestNumericVerification(unittest.TestCase):
    """Launch + numeric-compare on whatever gfx device this runs on."""

    def _run(self, *cmd, timeout=300):
        library_roots = [root for root in _LIBRARY_ROOTS if root.is_dir()]
        env = rocke_subprocess_env(python_paths=library_roots)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            rocke_python_command(*cmd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr)

    def test_core_parity(self):
        """Elementwise/reduce/norm/transpose/batched+grouped GEMM vs numpy."""
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

    def _run_extended(self, *cases):
        return self._run(
            "-m",
            "rocke.examples.common.parity_extended_kernels",
            "--arch",
            ARCH,
            "--op",
            ",".join(cases),
            timeout=420,
        )

    def test_extended_parity(self):
        """MoE streaming/streamk/mfma/block_scale/mx vs numpy: every case but
        the fused-MoE forward (``test_moe_e2e_forward``).

        On gfx950 every case runs and passes, so a SKIP line fails the test.
        On gfx942 the builders refuse the kernels that need a gfx950-only atom
        or fp8 format, and the harness reports those as SKIP. On both
        the gate is rc == 0 and no ``FAIL`` line.
        """
        if not _CDNA:
            self.skipTest(f"CDNA MFMA extended kernels; running on {ARCH} (RDNA)")
        from rocke.examples.common.parity_extended_kernels import ALL_CASES

        rc, out = self._run_extended(*(n for n in ALL_CASES if n != _MOE_E2E_CASE))
        self.assertEqual(rc, 0, f"extended parity failure on {ARCH}:\n{out[-3000:]}")
        self.assertNotIn(
            "FAIL", out, f"extended parity has FAIL on {ARCH}:\n{out[-3000:]}"
        )
        if ARCH == "gfx950":
            self.assertNotIn(
                "SKIP", out, f"extended parity skipped a case on gfx950:\n{out[-3000:]}"
            )
        self.assertIn("pass", out)

    @unittest.skipUnless(
        _HAVE_TORCH,
        skip_until_reason(TORCH_IN_CI, reason="FusedMoeForward takes torch tensors"),
    )
    def test_moe_e2e_forward(self):
        """The fused-MoE forward (router, sort, gather, grouped GEMMs, SwiGLU,
        weighted reduce) vs a torch eager reference, in the harness
        subprocess."""
        if not _CDNA:
            self.skipTest(f"CDNA MFMA extended kernels; running on {ARCH} (RDNA)")
        rc, out = self._run_extended(_MOE_E2E_CASE)
        self.assertEqual(rc, 0, f"fused-MoE forward failure on {ARCH}:\n{out[-3000:]}")
        self.assertNotIn("FAIL", out, out[-3000:])
        self.assertNotIn("SKIP", out, out[-3000:])
        self.assertIn("1/1 pass", out)

    def test_wmma_gemm(self):
        """gfx1151 (RDNA3.5) WMMA GEMM vs numpy A@B.T, within fp tolerance.

        RDNA-only: WMMA accumulates in f32 in a different order than numpy, so
        the gate is max_abs_diff within tolerance (1e-2), not exact parity.
        """
        if ARCH != "gfx1151":
            skip_needing_arch(
                ("gfx1151",), reason=f"gfx1151/WMMA GEMM; running on {ARCH}"
            )
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
        """gfx1151 (RDNA3.5) non-MFMA parity vs numpy.

        The ``ck_tile_parity`` harness runs the wave32-aware kernels and
        verifies them numerically: elementwise(×12), reduce(×3), layernorm2d,
        rmsnorm2d, transpose2d. The MFMA GEMM cases (batched/grouped) are
        SKIPped by the harness on a non-CDNA arch (RDNA uses WMMA, not MFMA),
        so they are out of scope here and covered by ``test_wmma_gemm``.

        The gate mirrors the CDNA ``test_core_parity``: rc == 0 and no ``FAIL``
        line in the output. ``SKIP`` is allowed (the MFMA-only cases).
        """
        if ARCH != "gfx1151":
            skip_needing_arch(
                ("gfx1151",), reason=f"gfx1151/RDNA non-MFMA parity; running on {ARCH}"
            )
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
