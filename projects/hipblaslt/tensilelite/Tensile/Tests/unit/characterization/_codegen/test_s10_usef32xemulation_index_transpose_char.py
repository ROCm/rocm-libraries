################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S10 - LocalRead UseF32XEmulation index-transpose characterization.

Drives the designed gfx1250 UseF32XEmulation config
(``data/test_data/_designed/gfx1250/s10_usef32xemulation_index_transpose.yaml``)
through the config-driven emit harness. Targets the F32X pack / index-transpose
paths in ``Tensile/Components/LocalRead.py``.

``UseF32XEmulation`` (DataType=S + F32XdlMathOp=X + HPA on gfx1250 WMMA_V3)
forces ``needPack`` and the F32X pack loop; ``VectorWidthA/B=2`` gives
``lrvwTile>1`` (index-transpose candidate); the widened tile / DepthU grow
``numReadsPerUnroll`` / ``numVgpr`` so ``multiGroupXF32`` engages. ``emit`` runs
``assignDerivedParameters`` + emission, so the target LocalRead lines fire during
the emit call.

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
    "s10_usef32xemulation_index_transpose.yaml",
)


def test_s10_usef32xemulation_index_transpose_emits():
    """UseF32XEmulation index-transpose config emits kernels; all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx1250" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s10_usef32xemulation_index_transpose_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
