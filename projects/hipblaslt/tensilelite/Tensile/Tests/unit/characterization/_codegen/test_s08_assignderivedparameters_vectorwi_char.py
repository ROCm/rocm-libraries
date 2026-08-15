################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 - Solution.py assignDerivedParameters VectorWidth re-derivation.

Drives the designed gfx950 XF32-emulation NT config
(``data/test_data/_designed/gfx950/s08_assignderivedparameters_vectorwi.yaml``)
through the config-driven emit harness. The coverage target is
``Tensile/SolutionStructs/Solution.py`` lines 2494-2498, the
``UseF32XEmulation`` + TLUA + ``numSubTiles>1`` VectorWidth adjustment
(including the ``SourceSwap`` StoreVectorWidth clamp at 2498).

``DataType=S`` + ``F32XdlMathOp=X`` + HPA selects XF32 emulation; the NT layout
makes both tensors TLU; MI 16x16x32 with MIWaveTile=[4,4] and DepthU==32 keeps
sub-iteration enabled during the VW window, so the derived VectorWidth of 4 is
re-divided by numSubTiles. ``assignDerivedParameters`` runs as part of emission,
so the target lines fire during the emit call.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
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
    "s08_assignderivedparameters_vectorwi.yaml",
)


def test_s08_assignderivedparameters_vectorwi_emits():
    """XF32-emulation NT config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s08_assignderivedparameters_vectorwi_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
