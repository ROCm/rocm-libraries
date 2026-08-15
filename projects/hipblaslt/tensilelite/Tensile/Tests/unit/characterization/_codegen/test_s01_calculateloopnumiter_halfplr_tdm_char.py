################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""S01 - KernelWriterAssembly calculateLoopNumIter HalfPLR + TDMSplit undo.

Drives the designed HalfPLR + TDMSplit single-wave config
(``data/test_data/_designed/gfx1250/s01_calculateloopnumiter_halfplr_tdm.yaml``)
through the config-driven emit harness. Targets the ``calculateLoopNumIter``
HalfPLR TDM re-enable / LDS-align and TDMSplit tail-undo cluster in
``Tensile/KernelWriterAssembly.py`` (lines 7558-7589):

  - the single-wave HalfPLR "re-enable TDM & align LDS buffer" arm
    (SMovB32 tdm{A,B}Group0 when NumWaves<=1), and
  - the TDMSplit tail-undo guard and single-wave undo
    (SkipUndoLabel + SCmpLeU32 + SCBranchSCC1, SSubU32 tdm{A,B}Group0).

``HalfPLR=1`` forces SuppressNoLoadLoop, MIWaveGroup (1,1) forces the
single-wave (NumWaves==1) arm, and ``TDMSplit=True`` with PGR>=2 (not Sparse)
enters the undo block, so emit fires all target lines during assignDerived +
emission.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s01_calculateloopnumiter_halfplr_tdm.yaml",
)


def test_s01_calculateloopnumiter_halfplr_tdm_emits():
    """HalfPLR + TDMSplit single-wave config emits kernels with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx1250" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s01_calculateloopnumiter_halfplr_tdm_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
