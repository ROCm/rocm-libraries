# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A2A-GEMM (FusedA2AMode=1, GatheredB) outer shard-loop codegen characterization.

The index structure is the stock GEMM one: K is a single summation index
holding the whole W*k_local reduction. The shard loop wraps the unroll loop,
which covers K/W per round; the shard count is a runtime scalar, not a problem
dimension.
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


class TestA2AGemmSegmentAbi:
    """Arg order across the layout list, the addArg sequence and the C++ twin."""

    _CPP_KERNARG = "include/Tensile/FusedA2AKernArg.hpp"

    def _emitted_segment(self):
        import re

        from config_harness import emit_kernels_from_config

        src = emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]
        names = re.findall(r"-\s+\.name:\s+(\S+)", src)
        return names[names.index("peer_0_flagPtr"):]

    def _layout_args(self):
        from Tensile.Components.Signature import _FUSED_A2A_SEGMENT_ARGS

        return list(_FUSED_A2A_SEGMENT_ARGS)

    def _cpp_source(self):
        import pathlib

        hdr = pathlib.Path(self._CPP_KERNARG)
        if not hdr.is_file():
            pytest.skip("run from the tensilelite root to reach the C++ headers")
        return hdr.read_text()

    def test_segment_ends_with_the_cu_count(self):
        assert self._emitted_segment()[-2:] == ["FusedAM", "FusedNumCu"]

    def test_addarg_order_matches_the_layout_list(self):
        assert self._emitted_segment() == [n for n, _ in self._layout_args()]

    def test_cpp_twin_appends_the_scalars_in_the_emitted_order(self):
        import re

        appended = re.findall(r'append<[^>]+>\(\s*"([A-Za-z_]\w*)"', self._cpp_source())
        assert appended == [n for n in self._emitted_segment() if not n.startswith("peer_")]

    def test_cpp_twin_counts_every_four_byte_scalar(self):
        import re

        m = re.search(r"FUSED_A2A_SLOT_COUNT \+ 1\) \* 8 \+ (\d+) \* 4", self._cpp_source())
        assert m, "FUSED_A2A_SEGMENT_BYTES no longer has the scalar term"
        assert int(m.group(1)) == sum(1 for _, size in self._layout_args() if size == 4)


class TestA2AGemmBatchNumbering:
    """The prologue's batch span: the first block and the block count."""

    def _src(self):
        from config_harness import emit_kernels_from_config

        return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]

    def _fused_w_offset(self, src):
        import re

        m = re.search(
            r"s_load_dword s\[sgprA2AShardCounter\], "
            r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)",
            src,
        )
        assert m, "no FusedW kernarg load in the emitted kernel"
        return int(m.group(1), 16)

    def test_cu_count_is_read_from_the_kernarg(self):
        import re

        from Tensile.Components.Signature import fusedA2AKernArgLayout

        src = self._src()
        layout = fusedA2AKernArgLayout()
        want = self._fused_w_offset(src) + layout["FusedNumCu"] - layout["FusedW"]
        loaded = [
            int(h, 16)
            for h in re.findall(
                r"s_load_dword s(?:\[sgpr\w+\]|\d+), "
                r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)",
                src,
            )
        ]
        assert want in loaded, "FusedNumCu at %#x is never loaded" % want

    def test_block_span_is_clamped_to_the_token_block_count(self):
        import re

        assert re.search(r"s_min_u32[^\n]*sgprNumWorkGroups1", self._src())

    def test_span_lands_in_two_long_lived_scalars(self):
        src = self._src()
        assert ".set sgprA2ABlockLo," in src
        assert ".set sgprA2ABlockCount," in src

    def test_block_count_is_the_span_width(self):
        import re

        assert re.search(
            r"s_sub_u32 s\[sgprA2ABlockCount\][^\n]*sgprA2ABlockLo", self._src()
        )


class TestA2AGemmElection:
    """The batch's single enqueuer: the wave-0 gate, the SMEM atomic, its target."""

    def _src(self):
        from config_harness import emit_kernels_from_config

        return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]

    def _atomic(self, src):
        import re

        m = re.search(r"^s_atomic_inc [^\n]*$", src, re.M)
        assert m, "no s_atomic_inc in the emitted kernel"
        return m

    def test_election_atomic_carries_glc(self):
        assert " glc" in self._atomic(self._src()).group(0)

    def test_wave0_gate_precedes_the_election_atomic(self):
        import re

        src = self._src()
        head = src[: self._atomic(src).start()]
        assert re.search(
            r"v_readfirstlane_b32 s\d+, v\[vgprSerial\][^\n]*\n"
            r"s_cmp_eq_u32 s\d+, 0[^\n]*\n"
            r"s_cbranch_scc0 label_A2ASkipEnqueue",
            head,
        ), "no wave-0 gate ahead of the election atomic"

    def test_election_target_is_the_batch_work_group_count(self):
        import re

        src = self._src()
        m = re.search(
            r"s_mul_i32 s(\d+), s\[sgprA2ABlockCount\], s\[sgprNumWorkGroups0\][^\n]*\n"
            r"s_sub_u32 s\1, s\1, 1",
            src,
        )
        assert m, "the election DATA is not count * F - 1"
        assert self._atomic(src).group(0).startswith("s_atomic_inc s%s," % m.group(1))

    def test_election_counters_start_past_the_line_aligned_flag_block(self):
        import re

        from Tensile.Components.Signature import (
            FUSED_A2A_LINE_BYTES,
            FUSED_A2A_MODE1_FLAG_OFFSET,
        )

        src = self._src()
        mask = hex(0xFFFFFFFF & -FUSED_A2A_LINE_BYTES)
        assert re.search(
            r"s_and_b32 s\d+, s\d+, %s\b" % mask, src
        ), "the flag block width is not rounded up to a line"
        assert re.search(
            r"^s_atomic_inc s\d+, s\[\d+:\d+\], %s\b" % hex(FUSED_A2A_MODE1_FLAG_OFFSET),
            self._atomic(src).group(0),
        ), "the election atomic does not sit at the flag block offset"


class TestA2AGemmEnqueueLoops:
    """The enqueuer's packing pass: one reservation and one submit per queue."""

    def _src(self):
        from config_harness import emit_kernels_from_config

        return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]

    def _queue_loop_body(self, src):
        import re

        head = re.search(r"^label_a2a_queue_loop\w*:", src, re.M)
        tail = re.search(r"^s_cbranch_scc1 label_a2a_queue_loop\w*", src, re.M)
        assert head and tail, "no a2a queue loop in the emitted kernel"
        return src[head.end() : tail.start()]

    def test_each_queue_reserves_once_and_submits_once(self):
        body = self._queue_loop_body(self._src())
        assert body.count("s_atomic_cmpswap_x2") == 1
        assert body.count("s_store_dwordx2") == 3

    def test_cursors_are_raised_before_the_reserve_loop(self):
        import re

        body = self._queue_loop_body(self._src())
        rsv = re.search(r"^label_sdma_reserve_loop\w*:", body, re.M)
        assert rsv, "no reserve loop inside the queue loop"
        assert body[: rsv.start()].count("s_atomic_umax_x2") == 2

    def test_room_check_cache_is_seeded_before_the_reserve_loop(self):
        import re

        body = self._queue_loop_body(self._src())
        rsv = re.search(r"^label_sdma_reserve_loop\w*:", body, re.M)
        assert rsv, "no reserve loop inside the queue loop"
        assert "cachedHwReadIndex = hardware rptr" in body[: rsv.start()]

    def test_packet_loop_runs_once_per_block(self):
        import re

        body = self._queue_loop_body(self._src())
        assert re.search(
            r"s_mov_b32 s\d+, s\[sgprA2ABlockCount\][^\n]*\n"
            r"label_a2a_packet_loop\w*:",
            body,
        ), "the packet loop is not seeded from A2ABlockCount"

    def test_single_rank_skips_the_packing_pass(self):
        import re

        assert re.search(
            r"s_cmp_eq_u32 s\d+, 0[^\n]*// W == 1\?\n"
            r"s_cbranch_scc1 label_A2ASkipEnqueue",
            self._src(),
        ), "W == 1 does not skip the packing pass"


class TestA2AGemmPacketBody:
    """The COPY_SUBWIN + ATOMIC pair the enqueuer places for each token block."""

    # MacroTile1 of the config: one token block's row count, and rect_y once the
    # tail block is behind us.
    _MT_TOKEN = 256

    # srdShiftLeft["B"] (GlobalReadVectorWidthB, 8 here) * bpeGR (bf16).
    _PRE_PAD = 16

    def _src(self):
        from config_harness import emit_kernels_from_config

        return emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]

    def _segment_base(self, src):
        import re

        from Tensile.Components.Signature import fusedA2AKernArgLayout

        m = re.search(
            r"s_load_dword s\[sgprA2AShardCounter\], "
            r"s\[sgprKernArgAddress:sgprKernArgAddress\+1\], (0x[0-9a-f]+)",
            src,
        )
        assert m, "no FusedW kernarg load in the emitted kernel"
        return int(m.group(1), 16) - fusedA2AKernArgLayout()["FusedW"]

    def _queue_loop_body(self, src):
        import re

        head = re.search(r"^label_a2a_queue_loop\w*:", src, re.M)
        tail = re.search(r"^s_cbranch_scc1 label_a2a_queue_loop\w*", src, re.M)
        assert head and tail, "no a2a queue loop in the emitted kernel"
        return src[head.end() : tail.start()]

    def _packet_loop_body(self, src):
        import re

        head = re.search(r"^label_a2a_packet_loop\w*:", src, re.M)
        tail = re.search(r"^s_cbranch_scc1 label_a2a_packet_loop\w*", src, re.M)
        assert head and tail, "no a2a packet loop in the emitted kernel"
        return src[head.end() : tail.start()]

    def test_both_packets_share_one_register_block(self):
        import re

        body = self._packet_loop_body(self._src())
        copy = re.search(r"s_mov_b32 s(\d+), 0x\w+[^\n]*SUBWIN DW0", body)
        atom = re.search(r"s_mov_b32 s(\d+), 0x\w+[^\n]*ATOMIC DW0", body)
        assert copy and atom, "the COPY + ATOMIC pair is not emitted"
        assert copy.group(1) == atom.group(1), "the two packets use different blocks"

    def test_tail_block_row_count_is_clamped(self):
        import re

        body = self._queue_loop_body(self._src())
        pkt = re.search(r"^label_a2a_packet_loop\w*:", body, re.M)
        assert pkt, "no packet loop inside the queue loop"
        assert re.search(
            r"s_min_u32 s\d+, s\d+, %d\b" % self._MT_TOKEN, body[: pkt.start()]
        ), "rect_y is not clamped against the macro tile before the packet loop"

    def test_row_count_saturates_after_the_tail_block(self):
        import re

        assert re.search(
            r"v_mov_b32 v\d+, %d\b" % self._MT_TOKEN,
            self._packet_loop_body(self._src()),
        ), "rect_y never settles at the full macro tile inside the packet loop"

    def test_copy_source_is_the_peer_input_pointer(self):
        from Tensile.Components.SdmaRingEmitter import OFF_recvPtr

        src = self._src()
        want = self._segment_base(src) + OFF_recvPtr
        assert (
            "offset:%d " % want in self._queue_loop_body(src)
            or "offset:%d\n" % want in self._queue_loop_body(src)
        ), "the peer input pointer at offset %d is never loaded" % want

    def test_atomic_target_is_the_local_flag_block(self):
        import re

        from Tensile.Components.Signature import FUSED_A2A_MODE1_FLAG_OFFSET

        body = self._queue_loop_body(self._src())
        assert re.search(
            r"s_add_u32 s\d+, s\[sgprA2ACounterPtr\], s\d+[^\n]*\n"
            r"s_addc_u32 s\d+, s\[sgprA2ACounterPtr\+1\], 0",
            body,
        ), "the ATOMIC target is not derived from this device's counter block"
        assert re.search(
            r"s_add(?:c)?_u32 s\d+, s\d+, %d\b" % FUSED_A2A_MODE1_FLAG_OFFSET, body
        ), "the flag slot does not sit at the mode-1 flag block offset"

    def test_destination_undoes_the_address_pre_pad(self):
        import re

        # defineAndResources subtracts srdShiftLeft["B"] * bpeGR from AddressB;
        # the gather destination is the tensor base, so it adds that back.
        assert re.search(
            r"s_add_u32 s\d+, s\[sgprAddressB\], %d\b" % self._PRE_PAD,
            self._queue_loop_body(self._src()),
        ), "the destination base never re-adds the AddressB pre-pad"


class TestA2AGemmRegisterBudget:
    """The emitted kernel's SGPR high-water mark."""

    def test_kernel_stays_within_the_sgpr_ceiling(self):
        import re

        from config_harness import emit_kernels_from_config

        src = emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")[0][1]
        found = re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", src)
        assert found, "no .amdhsa_next_free_sgpr in the emitted kernel"
        assert int(found[0]) <= 102


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


class TestA2AGemmPredicates:
    """The runtime shard guards ride on the solution's compound predicate list.

    Each guard occupies four sites: the Python emitter here, the C++ struct, the
    Pair<> registration and the MappingTraits. The last arm checks the two C++
    registrations by name.
    """

    _GUARDS = ("A2AWorldNonZero", "A2AShardDivisible", "A2AGsuCoalescedOff")
    _CPP_PREDICATES = "include/Tensile/Serialization/ContractionPredicates.hpp"

    def _predicates(self, mode=None):
        from config_harness import derive_states

        from Tensile.Contractions import ProblemPredicate, ProblemType

        state = derive_states(_CONFIG, arch="gfx950", limit_solutions=1)[0]
        if mode is not None:
            state["ProblemType"]["FusedA2AMode"] = mode
        pt = ProblemType.FromOriginalState(state["ProblemType"])
        return {p.tag: p.value for p in ProblemPredicate.CompoundPredicates(state, pt)}

    def test_mode_one_emits_every_guard(self):
        assert set(self._GUARDS) <= set(self._predicates())

    def test_shard_divisible_carries_depth_u(self):
        assert self._predicates()["A2AShardDivisible"] == 64

    def test_mode_zero_emits_none_of_them(self):
        assert not set(self._GUARDS) & set(self._predicates(mode=0))

    def test_cpp_registers_every_guard(self):
        import pathlib

        hdr = pathlib.Path(self._CPP_PREDICATES)
        if not hdr.is_file():
            pytest.skip("run from the tensilelite root to reach the C++ headers")
        src = hdr.read_text()
        for tag in self._GUARDS:
            assert "Pair<Predicates::Contraction::%s>" % tag in src
            assert "MappingTraits<Predicates::Contraction::%s," % tag in src


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

    @pytest.mark.parametrize("mat", ["C", "D"])
    def test_store_srd_accumulation_is_outside_the_shard_loop(self, mat):
        src = self._src()
        accum = "s[sgprSrd%s+0], s[sgprSrd%s+0]" % (mat, mat)
        assert accum in src, "computeStoreSrdStart no longer accumulates into Srd%s" % mat
        body = src[src.index("label_A2AShardLoopBegin:"):src.rindex("A2AShardLoopBegin")]
        assert accum not in body, "Srd%s accumulation is inside the shard loop" % mat


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

    def test_transition_is_branched_over_on_the_final_round(self):
        body = self._body()
        skip = "s_cbranch_scc1 label_A2ATransitionEnd"
        assert "s_cmp_eq_i32 s[sgprA2AShardCounter], 1" in body, (
            "the transition phase runs on the final round too"
        )
        assert body.index(skip) < body.index("A2AShardIdx"), (
            "the skip branch sits after part of the next round's setup"
        )

    def test_the_skip_target_clears_the_whole_transition(self):
        body = self._body()
        assert body.index("label_A2ATransitionEnd:") > body.rindex("base += shard offset"), (
            "the skip branch lands mid-rebind, leaving one srd half-advanced"
        )


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

    def test_inner_trip_count_is_recomputed_from_k_local(self, config):
        assert "s[sgprLoopCounterL], s[sgprA2AKLocal]" in self._body(config), (
            "the unroll trip count is hoisted out of the shard loop, or divides "
            "the whole W*k_local reduction instead of one shard"
        )

    def test_shard_offsets_step_per_tensor_layout(self, config):
        lines = self._transition(config).splitlines()
        a = next(l for l in lines if "elements per A shard" in l)
        b = next(l for l in lines if "elements per B shard" in l)
        assert "sgprA2AKLocal" in a, "A steps a whole tensor instead of k_local along K"
        assert "sgprSizeJ" in b and "sgprStrideB1J" in b, (
            "B steps something other than one [nToken, k_local] segment"
        )

    def test_k_local_divide_sits_outside_the_shard_loop(self, config):
        from config_harness import emit_kernels_from_config

        src = emit_kernels_from_config(config, limit=1, arch="gfx950")[0][1]
        assert "k_local = K / W" in src
        assert "k_local = K / W" not in self._body(config)

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
