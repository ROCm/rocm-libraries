################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""MX-scale M-major (UnrollMajorLDS==0) local-read rejection test.

Target code:
  - KernelWriterAssembly.py MXBlockA/B VGPR macro cold branch
    (`if not kernel["UnrollMajorLDS*"]:`), lines ~1145 (A) / ~1185 (B).
  - Tensile/SolutionStructs/Solution.py validation of the local-read width.

Background:
  A gfx1250 MX-F4 NN-layout config derives to a VALID solution with
  UnrollMajorLDS{A,B}==0 and MXBlock{A,B}>0. Unlike gfx950 (which rejects MX
  TLU=1 subtile geometry pre-emit), gfx1250 admits this solution and emission
  would otherwise reach the MX-scale VGPR-macro cold branch, then localReadMX.

  In that M-major layout a single 0.25-register local read spans fewer bytes
  than one MX scale unit (mxUnit = MatrixInstK // MXBlock = 128 // 32 = 4), so
  stridePerRead (=blockWidth*4 = 1.0) floors to tilePerRead == 0. Solution
  derivation must reject this unsupported layout before it reaches codegen.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os

import pytest

from config_harness import assert_config_rejects

pytestmark = pytest.mark.unit

_ARCH_1250 = "gfx1250"

_MX_F4_NN_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "mx_f4_nn_umlds0.yaml",
)


def test_mx_umlds0_is_rejected_during_derivation(monkeypatch, capsys):
    """The unsupported width is rejected before any kernel reaches codegen."""
    assert_config_rejects(
        _MX_F4_NN_CONFIG,
        _ARCH_1250,
        monkeypatch,
        capsys,
        {
            "reject: M-major MX-scale local read for A requires "
            "VectorWidthA >= MatrixInstK // MXBlockA (4), got 1": 2,
        },
    )
