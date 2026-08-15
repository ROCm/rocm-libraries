################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S09 -- assignDerivedParameters MX-scale (MXBlock) non-subtile derivation.

Drives the designed MX-F4 NN non-subtile config
(``data/test_data/_designed/gfx1250/s09_assignderivedparameters_mx_scale.yaml``)
through the config-driven *derive* harness. Targets the non-subtile MX-scale
GRVW/GLT derivation cluster in ``Tensile/SolutionStructs/Solution.py``
``assignDerivedParameters`` (the ``MXBlockX and not UseSubtileImpl`` arms and the
``setGlobalLoadTileDimClassic`` MXSA/MXSB calls they feed).

These arms are unreachable on gfx950 (which rejects MX without UseSubtileImpl)
but reachable on gfx1250: it keeps UseSubtileImpl as user-set and has no
MX-subtile reject. The NN layout keeps TLUMXSA/TLUMXSB both True, exercising the
TLU MXSA/MXSB arms plus the classic global-load-tile-dim calls.

Probe-confirmed Solution.py lines that fire during derivation: 2375, 3234, 3402,
3829, 3836, 3941, 3975, 3976, 3977, 4007, 4051, 4052, 4053, 4994, 5003, 5004.

This is a *derive*-template test (not emit): ``assignDerivedParameters`` runs and
the target lines fire while ``solutions_from_config`` derives the Solution
objects. The full emit path is intentionally not driven here -- assembly
emission for this M-major MX-scale layout hits an unimplemented ``localReadMX``
guard, which is outside the Solution.py derivation cluster under test.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import solutions_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s09_assignderivedparameters_mx_scale.yaml",
)


def test_s09_assignderivedparameters_mx_scale_derives():
    """MX-scale non-subtile forks survive assignDerivedParameters (>=1 solution).

    Each fork reaches the non-subtile MX-scale GRVW/GLT derivation cluster in
    Solution.py ``assignDerivedParameters`` and derives a valid Solution, so at
    least one solution survives (the target lines fire during derivation).
    """
    sols = solutions_from_config(_CONFIG, arch=_ARCH)
    assert len(sols) >= 1, (
        f"Expected >=1 surviving solution (MX-scale non-subtile derivation), "
        f"got {len(sols)}"
    )


def test_s09_assignderivedparameters_mx_scale_golden(snapshot):
    """S09 golden: surviving-solution count pins the MX-scale derivation."""
    assert len(solutions_from_config(_CONFIG, arch=_ARCH)) == snapshot
