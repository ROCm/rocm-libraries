# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A2A-GEMM (FusedA2AMode=1, GatheredB) outer shard-loop codegen characterization.

The index structure is the stock GEMM one: K is a single summation index
holding k_local. The shard loop wraps the unroll loop; its trip count is a
runtime scalar, not a problem dimension.
"""

import pytest

pytestmark = pytest.mark.unit

_CONFIG = "Tensile/Tests/common/comm/gfx950/a2a_gemm_loop.yaml"


def _tn_problem_type(**overrides):
    from Tensile.SolutionStructs.Problem import ProblemType

    state = {
        "OperationType": "GEMM",
        "DataType": "h",
        "DestDataType": "h",
        "ComputeDataType": "s",
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": False,
    }
    state.update(overrides)
    return ProblemType(state, False)


class TestA2AGemmSummationIndices:
    def test_stock_gemm_has_one_summation_index(self):
        """Control arm: without the flag nothing changes."""
        pt = _tn_problem_type()
        assert pt["NumIndicesSummation"] == 1
        assert pt["IndicesSummation"] == [2]

    def test_a2a_gemm_leaves_the_index_structure_alone(self):
        pt = _tn_problem_type(FusedA2AMode=1)
        assert pt["NumIndicesSummation"] == 1
        assert pt["IndicesSummation"] == [2]
        assert pt["IndexAssignmentsA"] == [2, 0]
        assert pt["IndexAssignmentsB"] == [2, 1]
        assert pt["IndexUnroll"] == 2
        assert pt["NumIndicesC"] == 2


class TestA2AGemmConfigPath:
    """The yaml -> BenchmarkProcess -> Solution path."""

    def test_config_yields_a_solution(self):
        from config_harness import derive_states

        states = derive_states(_CONFIG, arch="gfx950", limit_solutions=1)
        assert len(states) == 1, "the config must derive exactly one solution"
        pt = states[0]["ProblemType"]
        assert pt["FusedA2AMode"] == 1
        assert pt["NumIndicesSummation"] == 1
