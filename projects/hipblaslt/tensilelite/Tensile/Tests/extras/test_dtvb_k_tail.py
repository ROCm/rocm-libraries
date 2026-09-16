# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the DTVB K-tail BoundSizeMultiple predicate.

Lives under Tests/extras so collection does not pull rocisa (Tests/unit
conftest imports streamk5_test_helpers). The helper under test has no
toolchain imports.

Covers the regression in ROCm/rocm-libraries#11718: the old
isDirectToVgprDoable guard only raised AssertSummationElementMultiple
for NN. NT (Ailk_Bjlk) over-reads B the same way.
"""

import sys
from pathlib import Path

import pytest

# Import the helper by file path. Tensile.SolutionStructs.__init__ pulls
# Solution.py (rocisa); extras must stay collectable without a toolchain.
_HELPER = (
    Path(__file__).resolve().parents[2] / "SolutionStructs" / "DtvbKTail.py"
)
sys.path.insert(0, str(_HELPER.parent))
from DtvbKTail import applyDtvbKTailAssert  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.extras]


def _state(asem=1, depthu=64, transA=False, transB=True):
    return {
        "AssertSummationElementMultiple": asem,
        "DepthU": depthu,
        "ProblemType": {"TransposeA": transA, "TransposeB": transB},
    }


@pytest.mark.parametrize(
    "transA,transB,label",
    [
        (False, False, "NN"),
        (False, True, "NT"),
        (True, False, "TN"),
        (True, True, "TT"),
    ],
)
def test_dtvb_raises_asem_to_depthu_for_all_transposes(transA, transB, label):
    """DTVB must require K % DepthU == 0 for every transpose, not only NN."""
    state = _state(asem=1, depthu=64, transA=transA, transB=transB)

    applyDtvbKTailAssert(state, "B")

    assert state["AssertSummationElementMultiple"] == 64, label


def test_dtvb_nt_is_not_exempt():
    """The gfx1200 SFT crash was Ailk_Bjlk (NT) with K=94, DepthU=64."""
    state = _state(asem=1, depthu=64, transA=False, transB=True)

    applyDtvbKTailAssert(state, "B")

    assert 94 % state["AssertSummationElementMultiple"] != 0
    assert 64 % state["AssertSummationElementMultiple"] == 0


def test_dtva_does_not_change_asem():
    state = _state(asem=1, depthu=64)

    applyDtvbKTailAssert(state, "A")

    assert state["AssertSummationElementMultiple"] == 1


def test_dtvb_keeps_larger_existing_asem():
    state = _state(asem=128, depthu=64)

    applyDtvbKTailAssert(state, "B")

    assert state["AssertSummationElementMultiple"] == 128


def test_dtvb_depthu_32():
    state = _state(asem=1, depthu=32)

    applyDtvbKTailAssert(state, "B")

    assert state["AssertSummationElementMultiple"] == 32
