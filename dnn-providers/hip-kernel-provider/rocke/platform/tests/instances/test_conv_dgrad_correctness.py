# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numeric correctness tests for the conv backward-data (dgrad) implicit-GEMM kernel.

Builds one dgrad kernel per test case on the running GPU and compares the output
against a float32 torch reference (``torch.nn.grad.conv2d_input``).  Covers:

  - stride=1 (direct-store epilogue, no atomics)
  - stride=2 (tilde-decomposition, atomic epilogue)
  - split_k > 1 (atomic epilogue)
  - bf16 and fp32 data types
  - gfx1151 / gfx1201 via WMMA candidates
  - gfx1250 via WMMA wavelet pipeline (stride=1 mem + stride>1 / split_k wavelet)

Requires a ROCm GPU and torch (skip otherwise).

Run:
  PYTHONPATH=rocke/platform/python <torch-python> \
    rocke/platform/tests/instances/test_conv_dgrad_correctness.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest

from rocke.runtime.hip_module import get_device_arch

_PYDIR = os.path.join(os.path.dirname(__file__), "..", "..", "python")

ARCH = get_device_arch(0)
_HAS_TORCH = importlib.util.find_spec("torch") is not None

_MFMA_ARCHES = ("gfx90a", "gfx942", "gfx950")
_WMMA_ARCHES = ("gfx1151", "gfx1201")
_WMMA_WAVELET_ARCHES = ("gfx1250",)
_SUPPORTED_ARCHES = _MFMA_ARCHES + _WMMA_ARCHES + _WMMA_WAVELET_ARCHES

_SKIP_REASON = (
    f"needs a supported ROCm GPU ({', '.join(_SUPPORTED_ARCHES)}) + torch; "
    f"detected arch={ARCH!r}, torch={'ok' if _HAS_TORCH else 'missing'}"
)


def _run_benchmark(*extra_args, timeout=600):
    """Run benchmark_implicit_gemm_conv in a subprocess and return (rc, output)."""
    import io

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": _PYDIR}
    cmd = [
        sys.executable,
        "-m",
        "rocke.benchmark.benchmark_implicit_gemm_conv",
        "--arch",
        ARCH,
        "--direction",
        "dgrad",
        "--verify",
        "--sample",
        "0.05",
        "--warmup",
        "1",
        "--iters",
        "1",
        *extra_args,
    ]
    # Stream output to the terminal in real time and also collect it for
    # assertions.  Using Popen + readline avoids the buffering that hides
    # progress when capture_output=True is used with subprocess.run.
    buf = io.StringIO()
    with subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            buf.write(line)
        proc.wait(timeout=timeout)
    return proc.returncode, buf.getvalue()


@unittest.skipUnless(ARCH in _SUPPORTED_ARCHES and _HAS_TORCH, _SKIP_REASON)
class TestConvDgradCorrectness(unittest.TestCase):
    """Build and verify dgrad kernels numerically on the running GPU."""

    def _verify(self, *extra_args, label="", timeout=600):
        rc, out = _run_benchmark(*extra_args, timeout=timeout)
        self.assertEqual(
            rc,
            0,
            f"dgrad benchmark failed{' (' + label + ')' if label else ''} "
            f"on {ARCH}:\n{out[-3000:]}",
        )
        self.assertNotIn(
            "FAIL",
            out,
            f"dgrad numeric FAIL{' (' + label + ')' if label else ''} "
            f"on {ARCH}:\n{out[-3000:]}",
        )

    # ---- stride=1 (direct store, no atomics) ---------------------------------

    def test_fp16_stride1(self):
        """fp16 dgrad, stride=1 — single sub-GEMM, direct-store epilogue."""
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "4",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "1",
            label="fp16 stride=1",
        )

    def test_bf16_stride1(self):
        """bf16 dgrad, stride=1."""
        self._verify(
            "--dtype",
            "bf16",
            "--N",
            "4",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "1",
            label="bf16 stride=1",
        )

    def test_fp32_stride1(self):
        """fp32 dgrad, stride=1."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"fp32 dgrad candidates require MFMA; running on {ARCH}")
        self._verify(
            "--dtype",
            "fp32",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "1",
            label="fp32 stride=1",
        )

    # ---- stride=2 (tilde decomposition, atomic epilogue) ---------------------

    def test_fp16_stride2(self):
        """fp16 dgrad, stride=2 — tilde decomposition with atomic epilogue."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"stride>1 dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--sH",
            "2",
            "--sW",
            "2",
            "--split-k",
            "1",
            label="fp16 stride=2",
        )

    def test_bf16_stride2(self):
        """bf16 dgrad, stride=2."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"stride>1 dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "--dtype",
            "bf16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--sH",
            "2",
            "--sW",
            "2",
            "--split-k",
            "1",
            label="bf16 stride=2",
        )

    # ---- split_k > 1 (atomic epilogue) ---------------------------------------

    def test_fp16_split_k(self):
        """fp16 dgrad, split_k auto-selected — exercises atomic reduction path."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"split_k dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "4",
            "--Hi",
            "28",
            "--Wi",
            "28",
            "--C",
            "64",
            "--K",
            "128",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "-1",
            label="fp16 split_k=auto",
        )

    # ---- larger realistic shape ----------------------------------------------

    def test_fp16_resnet_shape(self):
        """fp16 dgrad, ResNet-style shape N8 H56 W56 C64 K64 R3 S3."""
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "8",
            "--Hi",
            "56",
            "--Wi",
            "56",
            "--C",
            "64",
            "--K",
            "64",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "-1",
            label="fp16 resnet N8H56W56C64K64",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
