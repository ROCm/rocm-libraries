################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 LRA tile assignment characterization.

Drives the designed LRA sweep config
(``data/test_data/_designed/gfx942/lra.yaml``) through the config-driven emit
harness. Targets LraTileAssignmentVALU lines 78-91 via a VALU dot-path fork
sweep (fp16 HPA, WaveSplitK, WorkGroup[2] > 1 for the NumWaveSplitK > 1 arm
and WorkGroup[2] = 1 for the else arm).

Note: LraTileAssignmentTransposedMFMA paths (lines 144-249, 285-409, 473-592,
611-693) require asmCaps["HasLDSTrB128B16"] which gfx942 lacks; those ranges
cannot be reached on this architecture.
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
    "lra.yaml",
)


def test_r2_lra_gfx942_emits():
    """LRA sweep config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")


def test_r2_lra_gfx942_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the LRA sweep emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
