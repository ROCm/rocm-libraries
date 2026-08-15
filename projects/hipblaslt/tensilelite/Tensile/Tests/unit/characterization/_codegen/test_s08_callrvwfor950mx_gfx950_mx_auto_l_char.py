################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 - Solution calLRVWFor950MX auto-LRVW characterization.

Drives the designed gfx950 MXFP8 auto-LRVW config
(``data/test_data/_designed/gfx950/s08_callrvwfor950mx_gfx950_mx_auto_l.yaml``)
through the config-driven emit harness. Targets the AUTO branch of
``calLRVWFor950MX`` in ``Tensile/SolutionStructs/Solution.py`` (line 3659),
reached only when the ISA is gfx950 AND (MXBlockA or MXBlockB) AND
``LocalReadVectorWidth{A,B} == -1``. Existing gfx950 MX designed configs pin
``LocalReadVectorWidth=16`` and take the ``!= -1`` validation arm instead, so
they never enter the auto arm.

``assignDerivedParameters`` runs during the emit call, so the target line fires
while emitting the single forked kernel.

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
    "s08_callrvwfor950mx_gfx950_mx_auto_l.yaml",
)


def test_s08_callrvwfor950mx_gfx950_mx_auto_l_emits():
    """gfx950 MX auto-LRVW config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s08_callrvwfor950mx_gfx950_mx_auto_l_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
