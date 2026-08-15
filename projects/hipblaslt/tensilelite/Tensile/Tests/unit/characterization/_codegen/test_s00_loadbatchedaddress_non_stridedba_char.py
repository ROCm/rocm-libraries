################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S00 - KernelWriterAssembly loadBatchedAddress non-StridedBatched characterization.

Drives the designed non-StridedBatched config
(``data/test_data/_designed/gfx942/s00_loadbatchedaddress_non_stridedba.yaml``)
through the config-driven emit harness. Targets ``loadBatchedAddress`` in
``Tensile/KernelWriterAssembly.py``, which is emitted only when
``not kernel["ProblemType"]["StridedBatched"]`` and dereferences an array of
buffer pointers for C/D, Beta-C (UseBeta), and A/B.

On gfx942 (MFMA, no ``RequiresXCntForVolatileVMEM``) the emit takes the else
arms (SLoadB64) for the D, C, and A/B pointer loads; the XCnt arms need a
gfx1250 twin. ``assignDerivedParameters`` + emission run during the emit call,
so the target lines fire without GPU, compile, or hardware.

CPU-only; pytestmark = pytest.mark.unit.
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
    "s00_loadbatchedaddress_non_stridedba.yaml",
)


def test_s00_loadbatchedaddress_non_stridedba_emits():
    """Non-StridedBatched config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s00_loadbatchedaddress_non_stridedba_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
