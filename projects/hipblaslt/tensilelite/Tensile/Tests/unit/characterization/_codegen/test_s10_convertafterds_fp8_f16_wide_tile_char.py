################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S10 - LocalRead ConvertAfterDS fp8->f16 wide-tile characterization.

Drives the designed ConvertAfterDS config
(``data/test_data/_designed/gfx950/s10_convertafterds_fp8_f16_wide_tile.yaml``)
through the config-driven emit harness. Targets the wide-tile local-read
conversion arms in ``Tensile/Components/LocalRead.py``:

  - the ``lrvwTile==4`` arm (lines 1332/1338/1357) via the ``VectorWidthA=4``
    fork, and
  - the ``lrvwTile==8`` wide-tile arm (line 1363) via ``VectorWidthA=8``.

For fp8 A operand with ``ConvertAfterDS`` and NT-for-A (TransposeA=False ->
UnrollMajorLDSA=False), ``lrvwTileA`` follows ``VectorWidthA``, so the
``VectorWidthA`` fork of {4, 8} exercises both conversion arms. ``emit`` runs
``assignDerivedParameters`` and kernel emission, so the target lines fire
during the emit call.

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
    "s10_convertafterds_fp8_f16_wide_tile.yaml",
)


def test_s10_convertafterds_fp8_f16_wide_tile_emits():
    """ConvertAfterDS fp8->f16 wide-tile config emits kernels, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s10_convertafterds_fp8_f16_wide_tile_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
