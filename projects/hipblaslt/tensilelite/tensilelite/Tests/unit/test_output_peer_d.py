# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only unit coverage of the OutputPeerD problem-type flag.

OutputPeerD makes a kernel store every D tile a second time into a peer D buffer (a split GEMM's
partitions then each hold the complete D). Only gfx1250 is validated for it. These tests cover
the Python side without a toolchain or a GPU: the limits that decide whether a solution may use
it, the problem-type naming and parsing, the selection predicate, and the kernel argument.
"""

import pytest

from tensilelite.Contractions import ProblemType as ContractionsProblemType
from tensilelite.SolutionStructs.Problem import ProblemType
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_GFX1250 = (12, 5, 0)


def _state(**over):
    """A plain-GEMM solution state that OutputPeerD accepts, with `over` applied."""
    problemType = {"OutputPeerD": True}
    problemType.update(over.pop("ProblemType", {}))
    state = {
        "ProblemType": problemType,
        "ISA": _GFX1250,
        "BufferStore": 1,
        "TemporalHintD": 0,
        "GlobalSplitU": 1,
        "TileProcessingStrategy": "None",
        "WorkAssignment": "StaticGrid",
        "StoreRemapVectorWidth": 0,
        "UseSubtileImpl": False,
        "WorkGroup": [16, 16, 1],
    }
    state.update(over)
    return state


# ---------------------------------------------------------------------------
# Solution limits
# ---------------------------------------------------------------------------
def test_plain_gfx1250_solution_is_allowed():
    assert Solution.outputPeerDUnmet(_state()) == []


@pytest.mark.parametrize("isa", [(9, 0, 10), (9, 4, 2), (9, 5, 0), (11, 0, 0), (12, 0, 0)])
def test_only_gfx1250_is_allowed(isa):
    assert Solution.outputPeerDUnmet(_state(ISA=isa)) == ["gfx1250"]


def test_missing_isa_is_not_allowed():
    state = _state()
    del state["ISA"]
    assert Solution.outputPeerDUnmet(state) == ["gfx1250"]


@pytest.mark.parametrize(
    "over, expected",
    [
        ({"BufferStore": 0}, "BufferStore=1"),
        ({"TemporalHintD": 3}, "TemporalHintD != 3"),
        ({"GlobalSplitU": 2}, "GlobalSplitU=1"),
        ({"TileProcessingStrategy": "StreamK"}, "no StreamK / persistent kernel"),
        ({"StoreRemapVectorWidth": 2}, "StoreRemapVectorWidth=0"),
        ({"UseSubtileImpl": True}, "UseSubtileImpl=False"),
        ({"WorkGroup": [16, 8, 2]}, "no LocalSplitU"),
        ({"ProblemType": {"SupportUserArgs": True}}, "SupportUserArgs=False"),
        ({"ProblemType": {"GroupedGemm": True}}, "no GroupedGemm"),
        ({"ProblemType": {"OutputAmaxD": True}}, "no OutputAmaxD"),
        ({"ProblemType": {"UseE": True}}, "no UseE"),
        ({"ProblemType": {"FusedA2A": "GemmA2A"}}, "FusedA2A=Disabled"),
    ],
)
def test_each_limit_is_reported_alone(over, expected):
    assert Solution.outputPeerDUnmet(_state(**over)) == [expected]


def test_every_unmet_limit_is_reported():
    unmet = Solution.outputPeerDUnmet(
        _state(ISA=(9, 5, 0), GlobalSplitU=2, ProblemType={"UseE": True})
    )
    assert unmet == ["gfx1250", "GlobalSplitU=1", "no UseE"]


def test_the_fused_post_loop_store_is_turned_off():
    # The fused post-loop store has no peer D store, so the gate switches it off.
    state = {"ProblemType": {"OutputPeerD": True}, "PostLoopStoreInNll": True}
    Solution.assignPostLoopStoreInNll(state)
    assert state["PostLoopStoreInNll"] is False


# ---------------------------------------------------------------------------
# Problem type: default, name, parsing and the selection predicate
# ---------------------------------------------------------------------------
def _problem_type(**overrides):
    cfg = {"DataType": 0}
    cfg.update(overrides)
    return ProblemType(cfg, False)


def test_output_peer_d_is_off_by_default_and_named_when_on():
    plain, peer = _problem_type(), _problem_type(OutputPeerD=True)
    assert plain["OutputPeerD"] is False
    assert "PeerD" not in str(plain)
    assert peer["OutputPeerD"] is True
    assert "PeerD" in str(peer)


def _contractions_problem_type(outputPeerD):
    config = {
        "OperationType": "GEMM",
        "DataType": "S",
        "DestDataType": "S",
        "ComputeDataType": "S",
        "TransposeA": 0,
        "TransposeB": 0,
        "Batched": True,
        "UseBeta": True,
        "TotalIndices": 4,
        "NumIndicesC": 3,
        "IndicesFree": [0, 1],
        "IndicesBatch": [2],
        "IndicesSummation": [3],
        "IndexAssignmentsA": [0, 3, 2],
        "IndexAssignmentsB": [3, 1, 2],
        "ComplexConjugateA": False,
        "ComplexConjugateB": False,
        "ActivationComputeDataType": "S",
        "HighPrecisionAccumulate": False,
    }
    if outputPeerD is not None:
        config["OutputPeerD"] = outputPeerD
    return ContractionsProblemType.FromOriginalState(config)


@pytest.mark.parametrize("given, expected", [(None, False), (False, False), (True, True)])
def test_contractions_problem_type_parses_the_flag(given, expected):
    assert _contractions_problem_type(given).outputPeerD is expected


@pytest.mark.parametrize("flag", [False, True])
def test_the_problem_selection_predicate_carries_the_flag(flag):
    predicates = _contractions_problem_type(flag).predicates(
        includeBatch=True, includeOperation=True, includeType=True
    )
    peer = [p for p in predicates if p.tag == "PeerDCheck"]
    assert len(peer) == 1
    assert peer[0].value is flag
