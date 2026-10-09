# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""S09 -- assignDerivedParameters DtlPlusLdsBuf / numLdsBlk reject (gfx942).

Reachable-invalid (category A): the bf16 TN MFMA config forks
PrefetchGlobalRead:[2,3] with DirectToLdsA/B enabled and large DepthU. During
Tensile/SolutionStructs/Solution.py:assignDerivedParameters the PGR>=3 fork
walks the DtlPlusLdsBuf LDS-budget path, then the feature combination is
rejected, so no valid solution survives.

Solution.py lines that fire during the rejected derivation (probe-confirmed):
  4939 : auto DtlPlusLdsBuf=1 when PGR>2 and DtlPlusLdsBuf==-1.
  4945 : disable DtlPlusLdsBuf when not(DirectToLdsA & DirectToLdsB).
  4964 : numLdsBlk=PGR for PGR>=3.

All forks reject during derivation, so ``len(solutions_from_config(...)) == 0``
pins the reachable-invalid reject (category A). CPU-only; no GPU, no compile.
"""

import os

import pytest

from config_harness import assert_config_rejects

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "s09_assignderivedparameters_dtlplusl.yaml",
)


def test_s09_assignderivedparameters_dtlplusl_rejects_with_reason(monkeypatch, capsys):
    """Every fork reports the expected DirectToLds, PGR, and LDS rejects."""
    assert_config_rejects(
        _CONFIG,
        _ARCH,
        monkeypatch,
        capsys,
        {
            "reject: b128 DirectToLds not supported": 8,
            "reject: DirectToLdsA not doable, but GNLCA enabled, rejecting": 4,
            "reject: DirectToLdsB not doable, but GNLCB enabled, rejecting": 4,
            "reject: PrefetchGlobalRead>=3 Supports only DirectToLdsA and DirectToLdsB": 1,
            "reject: Kernel Uses 67584 > 65536 bytes of LDS": 1,
            "reject: Kernel Uses 101376 > 65536 bytes of LDS": 1,
        },
    )
