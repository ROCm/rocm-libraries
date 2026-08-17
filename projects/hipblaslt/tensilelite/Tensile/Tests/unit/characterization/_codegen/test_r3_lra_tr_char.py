################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 — gfx950 LRA transposed-MFMA tile-assignment characterization.

Drives the designed LRA-transposed sweep config
(``data/test_data/_designed/gfx950/lra_tr.yaml``) through the config-driven
emit harness.

Target missing ranges in Tensile/Components/LraTileAssignment.py:
  144-249  : LraTileAssignmentTransposedMFMA.__call__
             (DataType=b BF16, HasLDSTrB128B16=True, enableLDSTr=True,
              tile01==1 -> isM=False path, wave-offset branch num1DWaves>1)
  285-409  : LraTileAssignmentTransposedMFMAB8.__call__
             (DataType=I8, HasLDSTrB64B8=True, enableLDSTr=True,
              both isM and not-isM branches)

Architecture note: HasLDSTrB128B16 and HasLDSTrB64B8 are gfx950-only caps.
gfx942 routes these types through LraTileAssignmentMFMA (non-transposed path)
when LDSTrInst=True is not available.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "lra_tr.yaml",
)


def test_r3_lra_tr_gfx950_emits():
    """LRA transposed-MFMA sweep config emits kernels on gfx950 with err==0.

    The BF16 NN + TransposeLDS=1 + LDSTrInst=True path dispatches to
    LraTileAssignmentTransposedMFMA (lines 144-249 of LraTileAssignment.py).
    The two MatrixInstruction shapes produce MIWaveGroup variations that
    exercise the num1DWaves>1 wave-offset branch (line 233).
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_"), f"unexpected basename: {base!r}"


def test_r3_lra_tr_gfx950_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the LRA-TR emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
