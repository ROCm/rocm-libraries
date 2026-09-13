################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R13 -- gfx942 UseScaleAB=Vector store-state addressing emit (AsmStoreState).

Drives a designed NN half (HHS_BH) GEMM with ``UseScaleAB: Vector`` on gfx942,
exercising the ScaleAVec column-address arms of ``Tensile/AsmStoreState.py`` that
no existing ``_codegen`` test reaches (the suite never emits a vector-scaled GEMM):

  StoreState.__init__ (optSingleColVgpr)   line 316       sharedColScaleAVecVgprs checkout
  StoreState.__init__ (numVgprsPerElement) line 380       ScaleAVec address register count
  StoreState.setupStoreElementsForBatch    line 1056-1057 addrScaleAVecVgpr per element
  StoreState.__del__                       line 1119-1120 sharedColScaleAVecVgprs checkIn

All gated by ``kernel["ProblemType"]["UseScaleAB"] == "Vector"`` AND isSingleKernel
(GSU==1). Single-buffer NN GEMM, so the store path takes the optSingleColVgpr
branch -- the only StoreState column-vgpr branch reachable in GEMM-only TensileLite
(optSharedColVgpr needs PackedC0IndicesX>1, which only OperationType:
TensorContraction produces, and TensileLite rejects that type; see work/char-findings.md F10).

Measured clean net-new vs the full ``-m unit`` baseline ``.coverage``:
AsmStoreState.py +6 (lines 316, 380, 1056, 1057, 1119, 1120). One kernel emitted,
err==0, basename ``Cijk_Ailk_Bljk_HHS_BH_SABV_...`` (SABV == ScaleAB Vector).
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "scaleab_vec_store.yaml",
)


def test_r13_scaleab_vec_store_gfx942_emits_assembly():
    """UseScaleAB=Vector NN GEMM emits >=1 gfx942 kernel with err==0 and ScaleA markers."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src and "gfx942" in src
        assert base.startswith("Cijk_")
        # SABV tag in the basename confirms UseScaleAB=Vector was selected.
        assert "SABV" in base, f"Kernel {base!r} missing SABV -- ScaleAB Vector not selected"
        # ScaleA addressing is emitted by the ScaleAVec store-state arms.
        assert "ScaleA" in src, f"Kernel {base!r} missing ScaleA -- ScaleAVec store arm not reached"


def test_r13_scaleab_vec_store_gfx942_golden(snapshot):
    """P4 golden: order-invariant {basename, err} digest of the ScaleAB=Vector emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
