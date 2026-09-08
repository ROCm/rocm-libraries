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

# Same solution; K gives an odd unroll trip count K/DepthU.
_CONFIG_ODD_TRIP = "Tensile/Tests/common/comm/gfx950/a2a_gemm_loop_odd_trip.yaml"


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


class TestA2AGemmSolutionProblemType:
    """The mode reaches the serialized solution, where the host kernarg gate reads it."""

    def _contraction_problem_type(self, **overrides):
        from Tensile.Contractions import ProblemType

        return ProblemType.FromOriginalState(_tn_problem_type(**overrides))

    def test_state_keys_carry_the_mode(self):
        from Tensile.Contractions import ProblemType

        assert "fusedA2AMode" in ProblemType.StateKeys

    def test_mode_round_trips(self):
        assert self._contraction_problem_type(FusedA2AMode=1).fusedA2AMode == 1

    def test_stock_gemm_leaves_the_mode_at_zero(self):
        assert self._contraction_problem_type().fusedA2AMode == 0


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

    def _body(self):
        src = self._src()
        return src[src.index("A2A_TRANSITION begin"):src.index("A2A_TRANSITION end")]

    def test_transition_rebinds_both_srd_bases(self):
        body = self._body()
        assert "SrdA" in body, "the transition phase does not touch srdA"
        assert "SrdB" in body, "the transition phase does not touch srdB"

    def test_transition_restores_both_shadow_limits(self):
        """ShadowLimit is reset each round, not carried."""
        body = self._body()
        assert "ShadowLimitA" in body, "srdA's limit stays drained across rounds"
        assert "ShadowLimitB" in body, "srdB's limit stays drained across rounds"

    def test_transition_offsets_by_the_shard_index(self):
        body = self._body()
        assert "A2AShardIdx" in body, "the rebind ignores which shard comes next"


@pytest.mark.parametrize("config", [_CONFIG, _CONFIG_ODD_TRIP])
class TestA2AGemmPerRoundState:
    """State the shard loop body re-establishes on every round."""

    def _body(self, config):
        from config_harness import emit_kernels_from_config

        src = emit_kernels_from_config(config, limit=1, arch="gfx950")[0][1]
        return src[src.index("label_A2AShardLoopBegin:"):src.rindex("A2AShardLoopBegin")]

    def _before_the_tail_loop_branch(self, config):
        body = self._body(config)
        return body[:body.index("s_cbranch_scc1 label_SkipTailLoopL")]

    def _transition(self, config):
        body = self._body(config)
        return body[body.index("A2A_TRANSITION begin"):body.index("A2A_TRANSITION end")]

    def test_lds_write_address_is_reloaded_into_m0(self, config):
        assert "s_mov_b32 m0, s[sgprLocalWriteAddrA]" in self._body(config), (
            "the DirectToLds m0 update is hoisted out of the shard loop"
        )

    def test_inner_trip_count_is_recomputed(self, config):
        assert "s[sgprLoopCounterL], s[sgprSizesSum" in self._body(config), (
            "the unroll trip count is hoisted out of the shard loop"
        )

    def test_no_load_loop_is_inside_the_body(self, config):
        assert "NoLoadLoop" in self._body(config), (
            "the drain phase is hoisted out of the shard loop"
        )

    def test_local_write_address_is_reset_on_every_path(self, config):
        reachable = self._before_the_tail_loop_branch(config)
        assert reachable.count("Set LWA to first buffer offset") == 2, (
            "srdA/srdB local-write addresses keep the round's buffer parity"
        )

    def test_local_read_address_reset_sits_behind_the_tail_loop_branch(self, config):
        assert "Set LRA to first buffer offset" not in self._before_the_tail_loop_branch(config)

    def test_transition_phase_resets_both_local_read_addresses(self, config):
        assert self._transition(config).count("Set LRA to first buffer offset") == 2, (
            "local-read addresses keep the round's buffer parity while local-write "
            "addresses are reset"
        )

    def test_transition_phase_writes_gsu_sum_idx(self, config):
        assert "GSUSumIdx" in self._transition(config)
