################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S01 - KernelWriterAssembly tailLoopAllocDTVVgpr (DirectToVgprA) characterization.

Drives the designed DirectToVgprA config
(``data/test_data/_designed/gfx942/s01_tailloopallocdtvvgpr_directtovgp.yaml``)
through the config-driven emit harness. Targets ``tailLoopAllocDTVVgpr`` in
``Tensile/KernelWriterAssembly.py`` -- the packDTVA/convDTVA branch (A side) and
the ``vgprBaseA`` checkout, plus the DirectToVgpr emit arms those RegSets feed.

Reachability: ``TransposeA=False`` gives ``TLUA=True``; with ``TransposeLDS=1``
that forces ``UnrollMajorLDSA=False`` so ``lrvwTileA=VectorWidthA=2 (>1)``. bf16
(``numBytes<4``) + TLUA takes the DTV pack path, and ``DirectToVgprA=True`` with
``lrvwTileA>1`` and ``MIInputPerThread=4 (>1)`` makes ``packDTVA`` True, arming
the A-side alloc branch. ``emit`` runs assignDerivedParameters + emission, so the
target lines fire during the emit call.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
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
    "s01_tailloopallocdtvvgpr_directtovgp.yaml",
)


def test_s01_tailloopallocdtvvgpr_directtovgp_emits():
    """DirectToVgprA config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s01_tailloopallocdtvvgpr_directtovgp_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the DTV emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
