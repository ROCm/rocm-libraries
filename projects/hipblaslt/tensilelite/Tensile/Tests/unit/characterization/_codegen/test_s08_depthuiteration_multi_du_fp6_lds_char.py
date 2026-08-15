################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 - Solution.py depthUIteration cluster characterization.

Drives the designed config
(``data/test_data/_designed/gfx1250/s08_depthuiteration_multi_du_fp6_lds.yaml``)
through the config-driven emit harness. The emit path runs
``assignDerivedParameters`` before emission, so the ``depthUIteration`` cluster
in ``Tensile/SolutionStructs/Solution.py`` (~3081-3093) fires during the emit
call. The config combines a TDM auto (-1) iterate-mode resolution group, a
TDM explicit-mask reject group, and an fp6 LdsPad clamp/reject group so the
derivation exercises the currently-missing arms of that block.

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
    "s08_depthuiteration_multi_du_fp6_lds.yaml",
)


def test_s08_depthuiteration_multi_du_fp6_lds_emits():
    """depthUIteration config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx1250" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s08_depthuiteration_multi_du_fp6_lds_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
