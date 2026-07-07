################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 — gfx1250 TensorDataMover non-wave-separated path characterization.

Target: Tensile/Components/TensorDataMover.py (miss=158 before this test).

The existing gfx1250 emit suite (test_emit_gfx1250_char.py / F4_MX.yaml) only
exercises the *wave-separated* code path because that logic file has
MIWaveGroup=[2,2].  Lines 29-116 (``calculateStartAddr``, the non-wave-sep
variant) were completely unexecuted.

This test drives the designed ``datamover.yaml`` config (F4/BF16 MX-GEMM with
TDMInst=3) using two MatrixInstruction shapes that both have MIWaveGroup=[1,1]
(prod==1), which forces the emitter down the non-wave-separated
``calculateStartAddr`` branch.  The MX-F4 data type (MXBlock=32) additionally
hits the MXS arms inside ``setTensorDim0``, ``setTensorDim1``, and
``setTensorStride0``.

Golden: order-invariant {basename, err} digest snapshot (seeded once).
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
    "datamover.yaml",
)


def test_r3_datamover_gfx1250_emits_assembly():
    """TDM non-wave-separated (MIWaveGroup=[1,1]) config emits real gfx1250 assembly."""
    results = emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r3_datamover_gfx1250_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the TDM datamover emit."""
    results = emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
