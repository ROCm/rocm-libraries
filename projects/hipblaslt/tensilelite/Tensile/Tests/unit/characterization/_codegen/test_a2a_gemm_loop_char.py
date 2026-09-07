# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A2A-GEMM (FusedA2AMode=1, GatheredB) outer shard-loop codegen characterization.

K is declared as two summation indices: index 2 = shard number (outer,
loopIdx 0), index 3 = k_local (unroll).
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

    def test_a2a_gemm_has_two_summation_indices(self):
        pt = _tn_problem_type(FusedA2AMode=1)
        assert pt["NumIndicesSummation"] == 2
        assert pt["IndicesSummation"] == [2, 3]

    def test_shard_index_is_outer_and_klocal_is_unroll(self):
        """Index 2 is the outer loop, index 3 the unroll."""
        pt = _tn_problem_type(FusedA2AMode=1)
        assert pt["IndicesSummation"][-1] == 3, "unroll must be k_local (index 3)"
        assert pt["IndicesSummation"][0] == 2, "outer must be the shard index (2)"

    def test_index_assignments_put_klocal_where_k_used_to_be(self):
        """TN: A = [k_local, nFeature, shard], B = [k_local, nToken, shard]."""
        pt = _tn_problem_type(FusedA2AMode=1)
        assert pt["IndexAssignmentsA"] == [3, 0, 2]
        assert pt["IndexAssignmentsB"] == [3, 1, 2]

    def test_batched_shifts_the_summation_indices_up_by_one(self):
        """Pins the remap ahead of initGEMM's Batched branch."""
        pt = _tn_problem_type(FusedA2AMode=1, Batched=True)
        assert pt["IndicesSummation"] == [3, 4]
        assert pt["IndicesBatch"] == [2]

    def test_metadata_indices_follow_the_remap(self):
        pt = _tn_problem_type(FusedA2AMode=1)
        assert pt["IndexUnroll"] in pt["IndexAssignmentsMetadata"]


class TestA2AGemmConfigPath:
    """The yaml -> BenchmarkProcess -> Solution path."""

    def test_config_yields_a_two_summation_solution(self):
        from config_harness import derive_states

        states = derive_states(_CONFIG, arch="gfx950", limit_solutions=1)
        assert len(states) == 1, "the config must derive exactly one solution"
        pt = states[0]["ProblemType"]
        assert pt["FusedA2AMode"] == 1
        assert pt["NumIndicesSummation"] == 2
        assert pt["IndicesSummation"] == [2, 3]
