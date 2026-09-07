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
        "Batched": True,
    }
    state.update(overrides)
    return ProblemType(state, False)


class TestA2AGemmSummationIndices:
    def test_stock_gemm_has_one_summation_index(self):
        """Control arm: without the flag nothing changes."""
        pt = _tn_problem_type()
        assert pt["NumIndicesSummation"] == 1
        assert pt["IndicesSummation"] == [3]

    def test_a2a_gemm_leaves_the_index_structure_alone(self):
        pt = _tn_problem_type(FusedA2AMode=1)
        assert pt["NumIndicesSummation"] == 1
        assert pt["IndicesSummation"] == [3]
        assert pt["IndexAssignmentsA"] == [3, 0, 2]
        assert pt["IndexAssignmentsB"] == [3, 1, 2]
        assert pt["IndexUnroll"] == 3
        assert pt["NumIndicesC"] == 3


class TestA2AGemmConfigPath:
    """The yaml -> BenchmarkProcess -> Solution path."""

    def test_config_yields_a_solution(self):
        from config_harness import derive_states

        states = derive_states(_CONFIG, arch="gfx950", limit_solutions=1)
        assert len(states) == 1, "the config must derive exactly one solution"
        pt = states[0]["ProblemType"]
        assert pt["FusedA2AMode"] == 1
        assert pt["NumIndicesSummation"] == 1

    def test_config_emits_a_kernel(self):
        from config_harness import emit_kernels_from_config

        kernels = emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")
        assert len(kernels) == 1, "the config must assemble to exactly one kernel"


class TestA2AGemmKernarg:
    """The fused-A2A kernarg segment carries the shard loop's trip count."""

    def _src(self):
        from config_harness import emit_kernels_from_config

        return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]

    def test_kernel_declares_the_shard_count(self):
        assert "FusedW" in self._src()


class TestA2AGemmShardLoop:
    """The shard loop wraps the unroll loop and closes before the store."""

    def _src(self):
        from config_harness import emit_kernels_from_config

        return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]

    def test_shard_loop_has_a_begin_label_and_a_back_edge(self):
        src = self._src()
        assert src.count("A2AShardLoopBegin") >= 2, "expected a label and a branch to it"

    def test_initc_precedes_the_shard_loop(self):
        src = self._src()
        assert src.index("initC") < src.index("A2AShardLoopBegin"), (
            "initC is emitted inside the shard loop"
        )

    def test_shard_loop_closes_before_the_summation_end(self):
        src = self._src()
        assert src.rindex("A2AShardLoopBegin") < src.index("Summation_End"), (
            "the back edge lands after endSummation"
        )


class TestA2AGemmTransitionPhase:
    """The transition phase sits between the tail loop and the back edge."""

    def _src(self):
        from config_harness import emit_kernels_from_config

        return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]

    def test_transition_phase_is_emitted(self):
        src = self._src()
        assert "A2A_TRANSITION begin" in src
        assert "A2A_TRANSITION end" in src

    def test_transition_phase_sits_inside_the_shard_loop(self):
        src = self._src()
        assert src.index("A2AShardLoopBegin") < src.index("A2A_TRANSITION begin"), (
            "the transition phase is emitted before the shard loop opens"
        )
        assert src.index("A2A_TRANSITION end") < src.rindex("A2AShardLoopBegin"), (
            "the transition phase is emitted after the back edge"
        )
