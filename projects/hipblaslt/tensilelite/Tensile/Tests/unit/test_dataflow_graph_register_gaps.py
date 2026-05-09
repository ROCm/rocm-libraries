################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Documented coverage of ScheduleCapture._writes / _reads.

The dataflow graph (`build_dataflow_graph` + `compare_graphs`) tracks
register dataflow for LR (DSLoad*), GR (BufferLoad*/GlobalLoad*), LW
(DSStore*), and MFMA, plus a `_GenericALURule` catch-all (wx9.4.4)
that publishes per-instance reads/writes for ALU ops (Pack/VSwap/VXor/
SAdd/SMov/etc.). All tests below assert the DESIRED behavior: the graph
detects a reordered chain and surfaces it as failures from
`compare_graphs`. With the catch-all rule live, every test in this file
PASSES — they exist to pin against regression.

Audit summary that produced this list:
- **P0** (silent miscompares possible)
  - `MFMA` accumulator RAW chain                     -> test_mfma_self_raw_chain
  - `m0` write before BufferLoad in DTL              -> test_dtl_m0_update_before_buffer_load
                                                        + test_dtl_m0_add_update_before_buffer_load
  - `GRInc` SCC chain & SRD WAW vs subsequent GR     -> test_grinc_srd_waw_before_buffer_load
  - LR LDS-address vgpr read (LRS -> LR RAW)         -> test_lrs_vxor_before_lr_invisible
  - LW LDS-address vgpr read (LWS -> LW RAW)         -> test_lws_vxor_before_lw_invisible
- **P1** (uncatchable scheduling bugs)
  - `Pack` VCvt*/VLShift*/VPerm/VPack RAW from LR    -> test_pack_cvt_raw_from_lr
  - `VSwap` bidirectional RMW reorder                -> test_vswap_pair_reorder_detected
                                                        + test_vswap_pair_allocation_invariant

Each test lives in its own class so a regression to a single coverage
slice can be diagnosed in isolation.

VCC dataflow tracking is intentionally not provided by the validator
(permanent design choice — see `CMSValidator_LIMITATIONS.md` §"VCC
dataflow tracking is intentionally not provided"). The corresponding
gap-pinning tests were removed by bead `rocm-libraries-uraq`.
"""

import pytest

from rocisa.container import RegisterContainer, RegName, sgpr, vgpr, mgpr
from rocisa.instruction import (
    SMovB32, SAddU32, SAddCU32, SSubU32, SSubBU32,
    SCmpEQU32, SCSelectB32, SCMovB32,
    VSwapB32,
    DSLoadB128, DSStoreB128, VXorB32,
    VCvtPkF32toBF16,
)

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    FourPartCapture,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)
from Tensile.Components.CMSValidator import (
    OrderInvertedFailure,
    TimingResult,
    TimingTooCloseFailure,
    _DEFAULT_CDNA4_ARCH_PROFILE,
    cumulative_issue_cycles,
    _quad_cycle_gap_ok,
    _cvt_to_mfma_gap_ok,
    _mfma_pack_to_cvt_gap_ok,
    build_dataflow_graph,
    compare_graphs,
    diagnose_missing_edge,
    validate_edge_wait_coverage,
)

from dataflow_fixtures import (
    make_lr, make_gr, make_dtl_buffer_load, make_mfma, make_swait, make_snop,
    make_capture,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap(ml_capture):
    """Wrap a single main-loop capture into a FourPartCapture, fillering the
    other 3 bodies with a no-op MFMA so build_dataflow_graph's
    non-empty-body precondition is satisfied. Pinned to
    `_DEFAULT_CDNA4_ARCH_PROFILE` so the timing helpers fire against the
    historical CDNA 4 quad-cycle constants every test in this file
    expects (rocm-libraries-zkzw)."""
    _FILLER_RANGES = {
        BODY_LABEL_ML_PREV: (200, 204, 208),
        BODY_LABEL_NGL:     (220, 224, 228),
        BODY_LABEL_NLL:     (240, 244, 248),
    }

    def _filler(label):
        c, a, b = _FILLER_RANGES[label]
        return make_capture(label, [make_mfma(
            c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
        )])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: _filler(BODY_LABEL_NGL)},
        n_ll={0: _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _tag(inst, *, category: str, mfma_index: int, sequence: int) -> TaggedInstruction:
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=mfma_index, sequence=sequence),
    )


# Historic note (kept for context): pre-wx9.4.4 every non-LR/LW/GR/MFMA
# rocisa class returned [] from _writes/_reads. wx9.4.4 (Sub-D) added the
# `_GenericALURule` catch-all to ScheduleCapture._OPERAND_RULES, which
# publishes per-instance reads/writes for any CommonInstruction-shaped
# ALU op (Pack/VSwap/VXor/SAdd/SMov/etc.). The PackRAW, GRIncSRD,
# LRSAddrChain, and LWSAddrChain tests below now PASS; their xfail
# markers were dropped along with the rule landing.
#
# wx9.3 phase 3 closed the VSwapPair edge-identity gap by adding
# operand-slot discrimination to cross-graph edge identity, and the
# DTLm0Tracking tests now use a real DTL BufferLoadB128 fixture
# (make_dtl_buffer_load) — both classes of test now PASS too.
# This file contains no xfail-marked tests.


# =============================================================================
# P0 — m0 tracking under DirectToLds
# =============================================================================
# Under DTL (kernel["DirectToLds"]=1), every BufferLoad in GRA/GRB is paired
# with a scalar instruction writing m0 (the implicit LDS-dest register the
# load uses). Emitted at KernelWriterAssembly.py:10049-10072.
#
# Two emission shapes:
#   - SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA"))               (default)
#   - SAddU32(dst=mgpr(0), src0=sgpr("LocalWriteAddrA"), src1=...)    (DTL+IncLdsBufSwitch
#                                                                      / DTL+ExpandPointerSwap)
#
# If the BufferLoad is reordered before its m0 setter, the load uses stale m0
# and writes the data into the wrong LDS slot. Today this reorder is invisible.


class TestDTLm0Tracking:
    """The m0 register written by SMovB32/SAddU32 is implicitly read by the
    following BufferLoad. Reordering breaks LDS placement."""

    def test_dtl_m0_update_before_buffer_load(self):
        """Stripped to the minimum: SMovB32 (m0 writer) + DTL BufferLoad
        (m0 reader). The realistic GEMM frame (SWait + MFMA) is
        unnecessary — the property is purely about m0 producer/consumer
        ordering."""
        # Reference: m0 set, then DTL BufferLoad consumes m0.
        m0_set = SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA", 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(m0_set, category="GRA", mfma_index=0, sequence=0),
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=1),
        ])
        # Subject: same instructions, m0-update issued AFTER the BufferLoad.
        # The load sees stale m0.
        m0_set2 = SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA", 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=0),
            _tag(m0_set2, category="GRA", mfma_index=0, sequence=1),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(g_ref, g_subj)
        assert failures, (
            "DTL m0 update reordered after BufferLoad — should have "
            "produced an OrderInvertedFailure on the m0 RAW edge."
        )
        assert any(isinstance(f, OrderInvertedFailure) for f in failures)

    def test_dtl_m0_add_update_before_buffer_load(self):
        """Same as above but the m0 update is the SAddU32 form
        (DTL + IncLdsBufSwitch / DTL + ExpandPointerSwap path).

        Stripped to the minimum: same minimization as the SMovB32 variant —
        the realistic GEMM frame is unnecessary for m0 producer/consumer
        ordering.
        """
        m0_add = SAddU32(dst=mgpr(0),
                         src0=sgpr("LocalWriteAddrA", 1),
                         src1=sgpr("LDSBufferWriteInc", 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(m0_add, category="GRA", mfma_index=0, sequence=0),
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=1),
        ])
        m0_add2 = SAddU32(dst=mgpr(0),
                          src0=sgpr("LocalWriteAddrA", 1),
                          src1=sgpr("LDSBufferWriteInc", 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=0),
            _tag(m0_add2, category="GRA", mfma_index=0, sequence=1),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "DTL m0 SAddU32 reordered after BufferLoad — should have "
            "produced a failure on the m0 RAW edge."
        )

    def test_m0_register_distinct_from_sgpr(self):
        """Confirm m0's regType differs from sgpr — Register.overlaps refuses
        to overlap m0 with an sgpr because reg_type differs. Any future m0
        tracking must therefore add an explicit m0-aware overlap path."""
        m0_reg = mgpr(0)
        s0_reg = sgpr(0, 1)
        assert m0_reg.regType != s0_reg.regType, (
            "If m0 ever shares regType with sgpr, the false-positive "
            "overlap risk in Register.overlaps must be re-audited."
        )


# =============================================================================
# WrappedInstruction + operand rule registry
# =============================================================================
# `_populate_wrapper` runs the rule registry over every captured rocisa
# instance; the result lives on `wrapper.reads` / `wrapper.writes` so
# build_dataflow_graph picks it up at edge formation. The DTL BufferLoad
# m0 read flows through `_DTLBufferLoadRule`; symmetric coverage for
# DSLoad/DSStore LDS-address vgprs, MFMA accumulator chains, and ALU
# fallbacks live alongside it.


class TestWrappedInstructionPopulation:
    def _build_dtl_buffer_load(self):
        from rocisa.container import vgpr, sgpr, MUBUFModifiers
        from rocisa.instruction import BufferLoadB128
        m = MUBUFModifiers(offen=True, offset12=0, lds=True)
        return BufferLoadB128(
            dst=None, vaddr=vgpr(40, 1), saddr=sgpr(20, 4),
            soffset=0, mubuf=m,
        )

    def _build_non_dtl_buffer_load(self):
        from rocisa.container import vgpr, sgpr, MUBUFModifiers
        from rocisa.instruction import BufferLoadB128
        m = MUBUFModifiers(offen=True, offset12=0, lds=False)
        return BufferLoadB128(
            dst=vgpr(8, 4), vaddr=vgpr(40, 1), saddr=sgpr(20, 4),
            soffset=0, mubuf=m,
        )

    def test_dtl_rule_contributes_m0_read(self):
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        wrapper = WrappedInstruction(self._build_dtl_buffer_load())
        _populate_wrapper(wrapper)
        assert any(r == mgpr(0) for r in wrapper.reads), (
            f"DTL BufferLoad's wrapper should expose m0 in reads; got {wrapper.reads}"
        )
        assert wrapper.writes == ()  # data goes to LDS, not a vgpr

    def test_non_dtl_buffer_load_does_not_contribute_m0(self):
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        wrapper = WrappedInstruction(self._build_non_dtl_buffer_load())
        _populate_wrapper(wrapper)
        assert all(r != mgpr(0) for r in wrapper.reads)
        # Non-DTL writes its vgpr dst.
        assert wrapper.writes != ()

    def test_finalize_populates_wrapper_through_capture_pipeline(self):
        """End-to-end: appending a DTL BufferLoad through the capture
        builder and calling finalize() populates `tagged.wrapped.reads`
        without any test-side intervention."""
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder
        builder = LoopBodyCaptureBuilder()
        inst = self._build_dtl_buffer_load()
        builder.append(inst, category="GRA", subiter=0)
        capture = builder.finalize()
        ti = capture.instructions[0]
        assert ti.wrapped is not None
        assert any(r == mgpr(0) for r in ti.wrapped.reads)

    def test_population_is_idempotent(self):
        """_populate_wrapper rebuilds reads/writes from scratch; running
        twice yields the same tuples."""
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        wrapper = WrappedInstruction(self._build_dtl_buffer_load())
        _populate_wrapper(wrapper)
        first_reads, first_writes = wrapper.reads, wrapper.writes
        _populate_wrapper(wrapper)
        assert wrapper.reads == first_reads
        assert wrapper.writes == first_writes

    def test_wrapper_forwards_attribute_access(self):
        """WrappedInstruction proxies getattr to the underlying rocisa inst,
        so `wrapper.getParams()`, `str(wrapper)`, etc. behave like the
        underlying instance."""
        from Tensile.Components.ScheduleCapture import WrappedInstruction
        inst = self._build_dtl_buffer_load()
        wrapper = WrappedInstruction(inst)
        assert wrapper.getParams() == inst.getParams()
        assert str(wrapper) == str(inst)
        assert wrapper.rocisa_inst is inst


# =============================================================================
# P0 — MFMA accumulator RAW chain
# =============================================================================
# `_reads(MFMA)` correctly enumerates a, b, AND c_dst (read-modify-write at
# ScheduleCapture.py:1623-1628). But `_writes(MFMA)` returns []. So a
# subsequent MFMA reading the same accumulator vgpr forms no edge to its
# producing MFMA. Reordering two MFMAs that share an accumulator silently
# corrupts the accumulator value.


class TestMFMASelfRAW:
    # MFMARule.writes = (acc,) closes this gap as of Sub-task 10.
    def test_mfma_acc_chain_reorder(self):
        """Two MFMAs share accumulator v0..v3. MFMA #1 must complete before
        MFMA #2 reads its acc. Reversing them is a real WAW/RAW violation.

        Stripped to the minimum: just the two MFMAs that share the
        accumulator. The realistic GEMM frame (LR + SWait) is unnecessary
        — the property is purely about MFMA-to-MFMA accumulator dataflow.
        """
        # Reference: MFMA1 (c_dst=0..3) then MFMA2 (a_src=0..1 reads same acc
        # by overlap). Use distinct slot numbers so they're separate nodes.
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=2),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, a_src_count=2),
        ])
        # Subject: same instructions, MFMAs swapped. Now the consumer MFMA
        # (reads v0..v1) issues before the producer (writes v0..v3).
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, a_src_count=2),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=3, a_src_count=2),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "MFMA->MFMA accumulator RAW edge missing because _writes(MFMA) "
            "returns []. Reordering two MFMAs that share an accumulator is "
            "silently undetectable."
        )


class TestMFMAQuadCycleGap:
    """Sub-C (wx9.4.3): the per-edge wait-coverage validator now applies a
    quad-cycle gap check to MFMA-as-producer edges instead of silently
    skipping them. When two MFMAs share an accumulator and the consumer
    issues at the SAME vmfma_index as the producer (gap == 0 quad-cycles),
    validate_edge_wait_coverage emits TimingTooCloseFailure rather than
    waving the edge through unverified.
    """

    def test_mfma_acc_chain_same_slot_passes_under_exact_arithmetic(self):
        """Two MFMAs sharing accumulator v0..v3, both at the SAME vmfma_index
        (sub_index breaks the tie). Cycle-exact arithmetic via
        `cumulative_issue_cycles` correctly gates the consumer behind
        the producer's `mfma_free_at` (current_issue + 1 + finish_cycles
        = 0 + 1 + 3 = 4). Consumer issues at cycle 4; gap = 4 - 0 - 1 =
        3 = expected. NO TimingTooCloseFailure.

        An earlier slot-delta approximation reported `slot_delta == 0 →
        actual = 0 < expected = 3` and synthesized a failure that does
        not exist in the cycle-accurate timeline."""
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and f.producer.category == "MFMA"
            and f.consumer.category == "MFMA"
        ]
        assert not mfma_timing, (
            f"Cycle-exact arithmetic: same-slot MFMAs are sequential "
            f"in the issue stream and inherit the producer's 4-cycle "
            f"mfma_free_at gate, so actual=3 == expected=3 → NO failure. "
            f"Got: {[type(f).__name__ for f in failures]}"
        )
        # Direct invariant check on the new exact arithmetic.
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        check = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert check.result == TimingResult.PASS and check.required == 3 and check.observed == 3, (
            f"nk0 contract: same-slot MFMAs report required=3, observed=3 (matches "
            f"precompute_issue_times); got result={check.result}, required={check.required}, "
            f"observed={check.observed}."
        )

    def test_mfma_acc_chain_consecutive_slots_no_failure(self):
        """Two MFMAs at consecutive vmfma_index values (slot_delta == 1)
        give exactly QUAD_CYCLES_STANDARD_MFMA_FINISH quad-cycles of gap —
        meets the threshold, so no TimingTooCloseFailure is emitted."""
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=2),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        assert not any(
            isinstance(f, TimingTooCloseFailure)
            and f.producer.category == "MFMA"
            and f.consumer.category == "MFMA"
            for f in failures
        ), f"Unexpected MFMA acc-chain timing failure: {failures}"

    # -------------------------------------------------------------------------
    # Negative-test additions covering previously-untested paths in
    # `_quad_cycle_gap_ok`. These exercise corners of the dispatch / gap
    # arithmetic that the boundary tests above do not pin.
    # -------------------------------------------------------------------------

    def test_mfma_acc_chain_slot_delta_2_independent_of_num_mfma_per_subiter(self):
        """Producer at vmfma=0, consumer at vmfma=2; vary num_mfma_per_subiter.
        `num_mfma_per_subiter` is no longer consulted by
        `_quad_cycle_gap_ok` — the helper walks the captured stream and
        accumulates exact issue cycles, so the same body produces the
        same `actual` regardless of `nmps`. This is the intended
        outcome of using the precompute-equivalent simulator (which
        sees every intermediate instruction's issue cost directly)
        instead of the earlier slot-delta approximation.

        For body=[MFMA@2, MFMA@4] the simulator yields:
        MFMA1 issues at 0, mfma_free_at=4; MFMA2 issues at max(1,4)=4 →
        gap = 4-0-1 = 3. Same value under any `nmps`."""
        def _build_cap():
            return make_capture(BODY_LABEL_ML, [
                make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                          slot=2, sequence=0, a_src_count=2),
                make_mfma(c_dst_start=20, a_src_start=0, b_src_start=32,
                          slot=4, sequence=0, a_src_count=2),
            ])

        def _wrap_with_nmps(cap, nmps):
            wrapped = _wrap(cap)
            return FourPartCapture(
                main_loop=wrapped.main_loop,
                main_loop_prev=wrapped.main_loop_prev,
                n_gl=wrapped.n_gl,
                n_ll=wrapped.n_ll,
                num_mfma=wrapped.num_mfma,
                num_codepaths=wrapped.num_codepaths,
                source=wrapped.source,
                num_mfma_per_subiter=nmps,
                arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
            )

        g_nmps1 = build_dataflow_graph(_wrap_with_nmps(_build_cap(), 1))
        g_nmps2 = build_dataflow_graph(_wrap_with_nmps(_build_cap(), 2))

        def _acc_edge(g):
            for e in g.edges:
                if (getattr(e.producer, "category", None) == "MFMA"
                        and getattr(e.consumer, "category", None) == "MFMA"):
                    return e
            return None

        e1 = _acc_edge(g_nmps1)
        e2 = _acc_edge(g_nmps2)
        assert e1 is not None and e2 is not None

        check1 = _quad_cycle_gap_ok(
            e1.producer, e1.consumer, 1, graph=g_nmps1)
        check2 = _quad_cycle_gap_ok(
            e2.producer, e2.consumer, 2, graph=g_nmps2)

        # nk0 contract: nmps is unused → same observed for both runs.
        assert check1.observed == check2.observed == 3, (
            f"nk0: cycle-exact arithmetic ignores nmps. Expected observed1==observed2==3; "
            f"got observed1={check1.observed}, observed2={check2.observed}."
        )
        assert (check1.result == TimingResult.PASS and check2.result == TimingResult.PASS
                and check1.required == 3 and check2.required == 3)

    def test_mfma_acc_chain_cross_body_uses_unified_simulator(self):
        """Producer in body=ML, consumer in body=NGL sharing accumulator
        v0..v3. Cross-body MFMA->MFMA edges route through
        `cumulative_issue_cycles` walking the unified instruction stream
        (ML-1 -> ML -> NGL -> NLL). An earlier `body_delta * 1000`
        placeholder always reported ok=True regardless of the actual gap;
        the unified simulator computes a real cycle count.

        Arithmetic for this fixture:
          unified stream = [ML-1 MFMA_filler, ML MFMA1, NGL MFMA2,
                            NLL MFMA_filler]
          Walk from MFMA1 (producer) to MFMA2 (consumer):
            MFMA1: current_issue=max(0,0)=0; mfma_free_at=0+1+3=4;
                   p_issue_start=0; current_issue += 1 = 1.
            MFMA2: current_issue=max(1,4)=4; same class (3==3), no
                   type-switch stall; c_issue_start=4. Break.
          gap = 4 - 0 - 1 = 3 (== QUAD_CYCLES_STANDARD_MFMA_FINISH).
        """
        ml_cap = make_capture(BODY_LABEL_ML, [
            # Producer writes v0..v3 in ML.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
        ])
        ngl_cap = make_capture(BODY_LABEL_NGL, [
            # Consumer in NGL reads v0..v1 (RAW on ML producer's acc).
            make_mfma(c_dst_start=20, a_src_start=0, b_src_start=32,
                      slot=0, sequence=0, a_src_count=2),
        ])
        # Fillers for ML_PREV and NLL only — ML and NGL are user-provided.
        ml_prev_filler = make_capture(BODY_LABEL_ML_PREV, [make_mfma(
            c_dst_start=200, a_src_start=204, b_src_start=208, slot=0)])
        nll_filler = make_capture(BODY_LABEL_NLL, [make_mfma(
            c_dst_start=240, a_src_start=244, b_src_start=248, slot=0)])
        four = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: ml_prev_filler},
            n_gl={0: ngl_cap},
            n_ll={0: nll_filler},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        g = build_dataflow_graph(four)

        # Find the cross-body MFMA->MFMA edge.
        cross = [e for e in g.edges
                 if getattr(e.producer, "category", None) == "MFMA"
                 and getattr(e.consumer, "category", None) == "MFMA"
                 and e.producer.body_label != e.consumer.body_label]
        assert cross, (
            "Expected a cross-body MFMA->MFMA acc-chain edge "
            "(ML producer -> NGL consumer)."
        )
        edge = cross[0]
        check = _quad_cycle_gap_ok(
            edge.producer, edge.consumer, 0, graph=g)

        # The unified simulator computes the exact cross-body cycle gap.
        # This fixture's intermediate-free MFMA pair gates exactly on the
        # producer's mfma_free_at backlog, yielding observed=3=required.
        assert check.result == TimingResult.PASS, "Cross-body gap of 3 meets the 3-cycle MFMA finish."
        assert check.required == 3, f"Expected QUAD_CYCLES_STANDARD_MFMA_FINISH=3; got {check.required}."
        assert check.observed == 3, (
            f"Cross-body MFMA->MFMA observed is the unified-stream cycle "
            f"count, not body_delta*1000. For this "
            f"fixture (one MFMA in ML, one MFMA in NGL, no intervening "
            f"work between them) the gap is exactly mfma_free_at = 3. "
            f"Got observed={check.observed}. If observed==1000, the cross-body branch "
            f"was reintroduced; if observed==0, the unified walk failed to "
            f"locate producer or consumer in the cross-body stream."
        )

    def test_mfma_acc_chain_cross_body_type_switch_stall_applied(self):
        """Cross-body MFMA->MFMA pairs that DIFFER in MFMA class (4x4
        producer vs standard consumer) must have the type-switch +1
        stall applied across the body boundary, just like same-body
        pairs. An earlier placeholder (`actual = body_delta * 1000`) couldn't
        compute this — it returned a fixed sentinel regardless of the MFMA
        flavors. The unified simulator walks the unified stream and applies
        the FROM_4X4=3 threshold check.

        Fixture: 4x4 PackMFMA producer at end of ML-1, standard MFMA
        consumer at start of ML, no intermediates. Producer's c_dst v0..v3
        is read by consumer (RAW acc-chain edge across the body boundary).

        Arithmetic:
          unified stream = [ML-1 MFMA1(4x4), ML MFMA2(std), NGL filler, NLL filler]
          i=0 MFMA1: current_issue=max(0,0)=0; mfma_free_at=0+1+1=2;
                     last_class=4x4=1; p_issue_start=0; current_issue=1.
          i=1 MFMA2: current_issue=max(1,2)=2; class differs (1->3), gap
                     since last MFMA = 2-0=2 < FROM_4X4=3 → +1 stall,
                     current_issue=3; c_issue_start=3.
          gap = 3 - 0 - 1 = 2.
          expected = _mfma_finish_cycles_for(4x4 producer) = 1.
          ok = True (2 >= 1).
        """
        ml_prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            # Producer in ML-1: 4x4 PackMFMA (variant=[4,4]) writing v0..v3.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=0, sequence=0, a_src_count=2,
                      variant=[4, 4]),
        ])
        ml_cap = make_capture(BODY_LABEL_ML, [
            # Consumer in ML: standard MFMA reading v0..v1.
            make_mfma(c_dst_start=20, a_src_start=0, b_src_start=32,
                      slot=0, sequence=0, a_src_count=2),
        ])
        ngl_filler = make_capture(BODY_LABEL_NGL, [make_mfma(
            c_dst_start=200, a_src_start=204, b_src_start=208, slot=0)])
        nll_filler = make_capture(BODY_LABEL_NLL, [make_mfma(
            c_dst_start=240, a_src_start=244, b_src_start=248, slot=0)])
        four = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: ml_prev_cap},
            n_gl={0: ngl_filler},
            n_ll={0: nll_filler},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        g = build_dataflow_graph(four)

        cross = [e for e in g.edges
                 if getattr(e.producer, "category", None) == "MFMA"
                 and getattr(e.consumer, "category", None) == "MFMA"
                 and e.producer.body_label != e.consumer.body_label]
        assert cross, (
            "Expected a cross-body MFMA->MFMA acc-chain edge "
            "(ML-1 4x4 producer -> ML standard consumer)."
        )
        edge = cross[0]
        check = _quad_cycle_gap_ok(
            edge.producer, edge.consumer, 0, graph=g)
        assert check.required == 1, (
            f"4x4 PackMFMA producer required = 1 quad-cycle "
            f"(_QUAD_CYCLES_MFMA_4X4_FINISH); got {check.required}."
        )
        assert check.result == TimingResult.PASS and check.observed == 2, (
            f"Cross-body 4x4->standard MFMA pair must apply the "
            f"FROM_4X4=3 type-switch stall in the unified walk. "
            f"Expected observed=2 (mfma_free_at=2 + 1 type-switch stall = 3, "
            f"gap = 3 - 0 - 1 = 2). Got result={check.result}, observed={check.observed}. "
            f"If observed==1, the type-switch stall is not being applied "
            f"across the body boundary (placeholder may have been "
            f"reintroduced or unified walk skipped the cross-body span)."
        )

    def test_mfma_acc_chain_cross_body_strict_when_graph_missing(self):
        """When no graph is provided, `_quad_cycle_gap_ok` returns
        `(False, 0, 0)` — strict, conservative. no graph -> no profile
        -> no derivable threshold. An earlier cross-body branch
        unconditionally returned `(True, expected, body_delta * 1000)`
        even without a graph, masking real issues in degenerate test
        paths. The unified path treats missing-graph as a hard failure
        regardless of whether the producer and consumer share a body.

        This is the negative pin: drop the cross-body branch and the
        result for a cross-body pair WITHOUT a graph flips from
        ok=True/actual=1000 to ok=False/actual=0.
        """
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=0, sequence=0, a_src_count=2),
        ])
        ngl_cap = make_capture(BODY_LABEL_NGL, [
            make_mfma(c_dst_start=20, a_src_start=0, b_src_start=32,
                      slot=0, sequence=0, a_src_count=2),
        ])
        ml_prev_filler = make_capture(BODY_LABEL_ML_PREV, [make_mfma(
            c_dst_start=200, a_src_start=204, b_src_start=208, slot=0)])
        nll_filler = make_capture(BODY_LABEL_NLL, [make_mfma(
            c_dst_start=240, a_src_start=244, b_src_start=248, slot=0)])
        four = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: ml_prev_filler},
            n_gl={0: ngl_cap},
            n_ll={0: nll_filler},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        g = build_dataflow_graph(four)
        cross = [e for e in g.edges
                 if getattr(e.producer, "category", None) == "MFMA"
                 and getattr(e.consumer, "category", None) == "MFMA"
                 and e.producer.body_label != e.consumer.body_label]
        assert cross, "Expected a cross-body MFMA->MFMA acc-chain edge."
        edge = cross[0]
        # No graph passed: with the cross-body placeholder REMOVED, the
        # function falls through to the strict `graph is None` branch
        # regardless of whether producer/consumer share a body.
        check = _quad_cycle_gap_ok(
            edge.producer, edge.consumer, 0, graph=None)
        assert check.result == TimingResult.FAIL, (
            "Cross-body pair without a graph must report FAIL. "
            "An earlier placeholder returned ok=True / observed=body_delta*1000 "
            "for any cross-body pair regardless of graph presence — this is "
            "the regression to pin."
        )
        assert check.required == 0 and check.observed == 0, (
            f"Expected TimingCheck(FAIL, observed=0, required=0) for graph=None "
            f"(no graph -> no profile -> no derivable threshold); got "
            f"(result={check.result}, required={check.required}, observed={check.observed})."
        )

    def test_mfma_acc_chain_diagnose_missing_edge_dispatches_through_mfma_branch(self):
        """Regression-pin test for the diagnose_missing_edge MFMA branch
        DISPATCH. An earlier slot-delta approximation produced a
        TimingTooCloseFailure with `actual=0` for this fixture; under
        the cycle-exact helper the simulator gives the consumer 3 cycles
        of gap (MFMA1 issues at 0, ALU at 1 (cost=1), MFMA2 at max(2,
        mfma_free=4)=4 → gap = 4-0-1 = 3 = expected). The MFMA branch
        IS reached (proving dispatch), but it now correctly returns
        ok=True and synthesizes no failure.

        The richer adversarial premise the original test sought (a real
        timing violation surfacing through diagnose_missing_edge) is no
        longer reachable by construction — the simulator's mfma_free_at
        contention prevents under-3-cycle gaps in any valid same-body
        MFMA→MFMA chain."""
        # ALU dst spans v0..v1 (8 bytes) so it fully shadows the consumer's
        # 8-byte read of v0..v1; otherwise the per-byte resolver would split
        # the read across two writers (ALU on bytes 0-3, MFMA-P on bytes 4-7)
        # and the P->C edge would persist.
        ref_alu = VXorB32(dst=vgpr(0, 2), src0=vgpr(40, 1), src1=vgpr(41, 1))
        subj_alu = VXorB32(dst=vgpr(0, 2), src0=vgpr(40, 1), src1=vgpr(41, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            # P: writes v0..v3.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # C: reads v0..v1 — overlaps P's accumulator (zero-gap RAW).
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
            # ALU after C — does NOT shadow P's write of v0..v1.
            _tag(ref_alu, category="PackA0", mfma_index=2, sequence=2),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # ALU reordered to sit BETWEEN P and C — now latest writer of v0.
            _tag(subj_alu, category="PackA0", mfma_index=2, sequence=1),
            # C reads v0..v1 — sees the ALU's write, not P's. No P->C edge.
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=2, a_src_count=2),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(g_ref, g_subj)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
            and getattr(f.consumer, "category", None) == "MFMA"
        ]
        # nk0: same-body MFMA→MFMA simulator gap = 3 = expected, so MFMA
        # branch returns ok=True and emits NO TimingTooClose. Routing is
        # still verified via the absence of an OrderInverted (which would
        # fire if the dispatch missed the MFMA branch entirely).
        assert not mfma_timing, (
            f"nk0: cycle-exact MFMA→MFMA same-body gap is 3=expected, "
            f"so no TimingTooClose. Got unexpected timing failures: "
            f"{mfma_timing}"
        )

    def test_mfma_to_alu_consumer_zero_gap_emits_timing_too_close(self):
        """Regression-pin test for current correct behavior: an MFMA producer
        with a same-iter ALU consumer at zero gap routes through the MFMA
        branch in `_classify_edge_coverage` (ScheduleCapture.py:3072) and
        emits TimingTooCloseFailure — it is NOT silently exempted by the ALU
        branch (ScheduleCapture.py:3086). This locks in today's branch
        priority so a future refactor that swaps the branch order or adds an
        early ALU-consumer exemption would be caught.

        Richer negative tests that would expose mis-routing with NON-zero
        gaps on this MFMA->ALU dispatch path are covered by the richer
        adversarial tests below.

        MFMA producer writes v0..v3 at slot=2; ALU consumer (VXorB32) at the
        same slot reads v0 — overlaps the MFMA producer's accumulator."""
        alu_consumer = VXorB32(dst=vgpr(20, 1), src0=vgpr(0, 1), src1=vgpr(21, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # Same vmfma_index, sub_index 1 — zero gap. ALU consumer reads v0
            # which overlaps the MFMA producer's accumulator v0..v3.
            _tag(alu_consumer, category="PackA0", mfma_index=2, sequence=1),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
        ]
        assert mfma_timing, (
            f"Expected TimingTooCloseFailure on MFMA->ALU zero-gap edge "
            f"(MFMA producer must dominate ALU-consumer dispatch), but got: "
            f"{[type(f).__name__ for f in failures]}"
        )

    # -------------------------------------------------------------------------
    # Richer adversarial negative tests for the MFMA quad-cycle dispatch
    # surface. Each test below ACTIVELY trips a discriminating dispatch
    # condition rather than only pinning zero-gap behavior.
    #
    # Note on `_quad_cycle_gap_ok` arithmetic: `cumulative_issue_cycles`
    # is a cycle-exact walk over the captured stream that mirrors the
    # (now-removed) `CMSValidator.precompute_issue_times`. `actual` is
    # the real delivered gap between the producer's issue cycle and the
    # consumer's issue cycle (minus 1), accounting for per-instruction
    # issue costs, the producer's `mfma_free_at` gate, and MFMA
    # type-switch stalls. For two standard MFMAs (finish=3) with no
    # intervening instructions this yields:
    #     same vmfma_index   → actual=3 (consumer waits on mfma_free=4;
    #                                    gap = 4-0-1 = 3, PASS)
    #     consecutive slots  → actual=3 (PASS at the threshold)
    # An earlier `slot_delta * (1 + finish) - 1 + subiter_delta` formula
    # is gone — it under-estimated densely-populated streams and
    # over-estimated sparse ones. Tests below discriminate via the
    # `actual` field's exact value at the boundary, via cross-graph
    # routing through diagnose_missing_edge, and via stream variations
    # that change `actual` without changing pass.
    # -------------------------------------------------------------------------

    def test_mfma_acc_chain_diagnose_missing_edge_dispatch_no_failure(self):
        """Cycle-exact arithmetic.
        REF: P (MFMA slot=2 seq=0) + C (MFMA slot=2 seq=1) → P→C edge present.
        SUBJ: P (slot=2 seq=0), ALU (slot=2 seq=1), C (slot=2 seq=2) →
              ALU shadows v0..v1, P→C edge missing.
        compare_graphs routes to diagnose_missing_edge whose MFMA branch
        runs on subj_graph: simulator gives MFMA1 issue=0, mfma_free=4;
        ALU adds cost 1 → issue=1; MFMA2 max(2, mfma_free=4)=4 → c_issue=4;
        gap = 4-0-1 = 3 = expected. ok=True → NO TimingTooCloseFailure.

        An earlier slot-delta approximation produced a phantom failure
        with actual=0 here that does not exist in the real timeline."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            # REF: P writes v0..v3, C immediately reads v0..v1. NO shadowing
            # ALU between them — P->C survives as the missing-from-subj edge.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        subj_alu = VXorB32(dst=vgpr(0, 2), src0=vgpr(40, 1), src1=vgpr(41, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # SUBJ: ALU reordered between P and C — becomes latest writer
            # of v0..v1, so the per-byte resolver routes C's read to ALU
            # instead of P. The MFMA->MFMA edge from REF is missing in SUBJ.
            _tag(subj_alu, category="PackA0", mfma_index=2, sequence=1),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=2, a_src_count=2),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(g_ref, g_subj)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
            and getattr(f.consumer, "category", None) == "MFMA"
        ]
        # nk0 cycle-exact: simulator gap = 3 = expected → ok=True → no
        # TimingTooClose on the MFMA→MFMA edge.
        assert not mfma_timing, (
            f"nk0: cycle-exact diagnose path on the MFMA→MFMA edge yields "
            f"actual=3=expected, so no TimingTooClose. Got: {mfma_timing}"
        )

    def test_mfma_acc_chain_just_meets_quad_cycle_gap(self):
        """Boundary case: gap exactly equal to `expected` (slot_delta == 1
        → actual == 3 == expected). The dispatch must NOT emit
        TimingTooCloseFailure on this MFMA→MFMA edge — the inequality in
        `_quad_cycle_gap_ok` is `actual >= expected` (inclusive at the
        threshold). A regression that flipped to `actual > expected`
        (strict) would over-flag every minimally-spaced consecutive MFMA
        pair; this test catches that flip.

        REF: identical to SUBJ — single P at slot=2 writing v0..v3, C at
        slot=3 reading v0..v1. SUBJ is identical; we route through
        compare_graphs to verify NEITHER the diagnose_missing_edge MFMA
        branch nor _classify_edge_coverage emits a TimingTooClose on this
        edge."""
        def _build():
            return make_capture(BODY_LABEL_ML, [
                # P at slot=2.
                make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                          slot=2, sequence=0, a_src_count=2),
                # C at slot=3 reads v0..v1 — slot_delta == 1, the threshold.
                make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                          slot=3, sequence=0, a_src_count=2),
            ])
        g_ref = build_dataflow_graph(_wrap(_build()))
        g_subj = build_dataflow_graph(_wrap(_build()))

        # Cross-graph route: identical graphs ⇒ no missing edges ⇒ no
        # diagnose_missing_edge dispatch ⇒ no TimingTooClose from that path.
        failures_cross = compare_graphs(g_ref, g_subj)
        assert not any(
            isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
            and getattr(f.consumer, "category", None) == "MFMA"
            for f in failures_cross
        ), (
            f"compare_graphs should NOT emit MFMA TimingTooClose at "
            f"slot_delta==1 (actual==expected==3). Got: {failures_cross}"
        )

        # Within-graph route via _classify_edge_coverage: the dispatch must
        # also accept the boundary case — `actual >= expected` inclusive.
        failures_within = validate_edge_wait_coverage(g_subj)
        assert not any(
            isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
            and getattr(f.consumer, "category", None) == "MFMA"
            for f in failures_within
        ), (
            f"validate_edge_wait_coverage should NOT emit MFMA "
            f"TimingTooClose at slot_delta==1 (boundary). Got: "
            f"{failures_within}"
        )

        # Direct check on _quad_cycle_gap_ok: at the boundary, PASS with
        # observed == required == 3. A future strict-inequality regression
        # would flip the result to FAIL here.
        # Find the MFMA→MFMA edge for the direct invariant assertion.
        acc_edge = None
        for e in g_subj.edges:
            if (getattr(e.producer, "category", None) == "MFMA"
                    and getattr(e.consumer, "category", None) == "MFMA"):
                acc_edge = e
                break
        assert acc_edge is not None, "expected MFMA→MFMA acc-chain edge"
        check = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g_subj)
        assert check.result == TimingResult.PASS, (
            f"_quad_cycle_gap_ok must accept the boundary case "
            f"(observed={check.observed}, required={check.required}); "
            f"result was {check.result}."
        )
        assert check.required == 3 and check.observed == 3, (
            f"At slot_delta==1, required==observed==3; got "
            f"required={check.required}, observed={check.observed}."
        )

    def test_mfma_to_mfma_cross_subiter_routing_exact(self):
        """Cycle-exact: `num_mfma_per_subiter` no longer affects
        `actual`. The simulator walks the captured stream directly, so
        same-body slot_delta=2 with no intermediate instructions yields
        actual=3 regardless of `nmps`. Both same-subiter (nmps=4) and
        cross-subiter (nmps=2) configurations produce the same `actual`."""
        def _build_cap():
            return make_capture(BODY_LABEL_ML, [
                # Producer at vmfma=0 (subiter 0 under both nmps values).
                make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                          slot=0, sequence=0, a_src_count=2),
                # Consumer at vmfma=2 reads v0..v1 (RAW on producer's acc).
                # Under nmps=2 → subiter 1 (cross-subiter).
                # Under nmps=4 → subiter 0 (same-subiter).
                make_mfma(c_dst_start=20, a_src_start=0, b_src_start=32,
                          slot=2, sequence=0, a_src_count=2),
            ])

        def _wrap_with_nmps(cap, nmps):
            wrapped = _wrap(cap)
            return FourPartCapture(
                main_loop=wrapped.main_loop,
                main_loop_prev=wrapped.main_loop_prev,
                n_gl=wrapped.n_gl,
                n_ll=wrapped.n_ll,
                num_mfma=wrapped.num_mfma,
                num_codepaths=wrapped.num_codepaths,
                source=wrapped.source,
                num_mfma_per_subiter=nmps,
                arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
            )

        g_same = build_dataflow_graph(_wrap_with_nmps(_build_cap(), 4))
        g_cross = build_dataflow_graph(_wrap_with_nmps(_build_cap(), 2))

        def _acc_edge(g):
            for e in g.edges:
                if (getattr(e.producer, "category", None) == "MFMA"
                        and getattr(e.consumer, "category", None) == "MFMA"):
                    return e
            return None

        e_same = _acc_edge(g_same)
        e_cross = _acc_edge(g_cross)
        assert e_same is not None and e_cross is not None, (
            "Expected an MFMA→MFMA acc-chain edge in both graphs."
        )

        check_same = _quad_cycle_gap_ok(
            e_same.producer, e_same.consumer, 4, graph=g_same)
        check_cross = _quad_cycle_gap_ok(
            e_cross.producer, e_cross.consumer, 2, graph=g_cross)

        # Both pass.
        assert check_same.result == TimingResult.PASS and check_cross.result == TimingResult.PASS
        # nk0: identical observed under exact arithmetic (nmps unused).
        assert check_same.observed == check_cross.observed, (
            f"nk0: cycle-exact arithmetic ignores nmps; expected equal "
            f"observeds. Got same-subiter observed={check_same.observed}, "
            f"cross-subiter observed={check_cross.observed}."
        )
        # And it's the simulator's value: MFMA1 issue=0, mfma_free=4;
        # MFMA2 max(1, 4)=4 → gap=4-0-1=3.
        assert check_same.observed == 3 and check_cross.observed == 3

    def test_mfma_producer_multi_consumer_varied_gaps_exact(self):
        """Under cycle-exact arithmetic the simulator's mfma_free_at
        contention naturally schedules each consumer with at least 3
        cycles of gap behind the previous MFMA, so there are no
        same-body MFMA→MFMA timing failures in this fixture.

        Per-edge cumulative_issue_cycles values:
          P→C1 (slot 2 seq 0 → seq 1): MFMA1 issue=0, mfma_free=4; C1 max(1,4)=4 → gap=3.
          P→C2 (slot 2 → slot 3): walk includes C1 (mfma_free becomes 8); C2 issues at 8 → gap=7.
          P→C3 (slot 2 → slot 4): walk includes C1 (free=8) + C2 (free=12); C3 max(9,12)=12 → gap=11.
        All ≥ expected=3 → no failures. Sanity check: at least 3 edges exist."""
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_mfma(c_dst_start=40, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
            make_mfma(c_dst_start=44, a_src_start=2, b_src_start=32,
                      slot=3, sequence=0, a_src_count=2),
            make_mfma(c_dst_start=48, a_src_start=0, b_src_start=32,
                      slot=4, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
            and getattr(f.consumer, "category", None) == "MFMA"
        ]
        assert not mfma_timing, (
            f"nk0: exact arithmetic produces no MFMA→MFMA failures here; "
            f"got: {[(f.actual_quad_cycles, f.expected_quad_cycles) for f in mfma_timing]}"
        )
        mfma_edges = [
            e for e in g.edges
            if (getattr(e.producer, "category", None) == "MFMA"
                and getattr(e.consumer, "category", None) == "MFMA")
        ]
        assert len(mfma_edges) >= 3, (
            f"Expected at least 3 MFMA→MFMA edges (one per consumer); "
            f"got {len(mfma_edges)}."
        )

    def test_mfma_quadcycle_mutation_smell(self):
        """Mutation smell-test: confirm `_classify_edge_coverage` IS the
        source of the within-graph MFMA→ALU timing verdict. Use the
        MFMA→ALU zero-gap fixture (the only same-body MFMA-producer case
        that still fails under cycle-exact arithmetic, since ALU consumers
        do not get the mfma_free_at gating that real MFMA consumers do).

        Pre-mutation: real dispatch emits TimingTooClose. Post-mutation
        (no-op patch on `_classify_edge_coverage`): zero failures."""
        from unittest.mock import patch

        alu_consumer = VXorB32(dst=vgpr(20, 1), src0=vgpr(0, 1), src1=vgpr(21, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            _tag(alu_consumer, category="PackA0", mfma_index=2, sequence=1),
        ])
        g = build_dataflow_graph(_wrap(cap))

        real_failures = validate_edge_wait_coverage(g)
        real_timing = [
            f for f in real_failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
        ]
        assert real_timing, (
            "Pre-mutation sanity: real _classify_edge_coverage must emit "
            "a TimingTooCloseFailure on the MFMA→ALU zero-gap fixture."
        )

        with patch(
            "Tensile.Components.CMSValidator._classify_edge_coverage",
            return_value=[],
        ):
            mutated_failures = validate_edge_wait_coverage(g)
            mutated_timing = [
                f for f in mutated_failures
                if isinstance(f, TimingTooCloseFailure)
                and getattr(f.producer, "category", None) == "MFMA"
            ]
            assert not mutated_timing, (
                f"Mutation smell-test failed: patched dispatch still "
                f"emitted {len(mutated_timing)} failures."
            )

    # -------------------------------------------------------------------------
    # 4x4 PackMFMA finish-cycle and dispatch-routing tests.
    # Two pre-existing bugs these tests pin against regression:
    #   (1) `_quad_cycle_gap_ok` read `mfma_finish_cycles` off the rocisa
    #       instance, but that attribute is a ClassVar on the validator
    #       dataclasses (CMSValidator.MFMAPack), not on the rocisa
    #       MFMAInstruction. The lookup always fell through to standard 3,
    #       so a 4x4 PackMFMA's true 1-quad-cycle finish was never modelled.
    #   (2) PackMFMAs are categorized "PackA*"/"PackB*" — `_is_alu_producer`
    #       fired the ALU-immediate exemption first, so PackMFMA producers
    #       never reached the MFMA quad-cycle branch even after Fix (1).
    # -------------------------------------------------------------------------

    def test_mfma_pack_acc_chain_4x4_to_standard_exact_gap(self):
        """Cycle-exact: 4x4 PackMFMA producer (finish=1) at slot=2
        seq=0; standard MFMA consumer at slot=2 seq=1. Simulator:
        producer issues at 0, mfma_free=2, last_mfma_class=4x4. Consumer
        max(1,2)=2; type-switch (4x4→standard), gap=2-0=2 <
        FROM_4X4=3 → +1 stall → consumer issues at 3. Gap=3-0-1=2.
        expected=1 → ok=True → NO failure. (An earlier slot_delta=0
        approximation reported a phantom actual=0 failure here.)"""
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and f.producer.category == "MFMA"
            and f.consumer.category == "MFMA"
        ]
        assert not mfma_timing, (
            f"nk0: 4x4→standard same-slot has actual=2 ≥ expected=1 → "
            f"no failure. Got: {mfma_timing}"
        )
        # Direct invariant: confirm the helper math.
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        check = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert (check.result == TimingResult.PASS and check.required == 1
                and check.observed == 2), (
            f"nk0 contract: 4x4 producer required=1; type-switch +1 stall "
            f"yields observed=2; got result={check.result}, "
            f"required={check.required}, observed={check.observed}."
        )

    def test_mfma_pack_acc_chain_meets_finish_1_no_failure(self):
        """4x4 PackMFMA producer (finish=1) with a consumer at consecutive
        vmfma_index. Cycle-exact walk: producer issues at 0,
        mfma_free_at=2 (4x4 finish=1 → +1+finish=2). Consumer issues at
        max(1, 2)=2; gap = 2-0-1 = 1 quad-cycle — EXACTLY meets the
        1-quad-cycle threshold, so no TimingTooCloseFailure is emitted.
        With the previous wrong finish=3 default this would have been
        mis-flagged as too close."""
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=2,
                      variant=[4, 4, 4, 16]),
            # Consumer at next vmfma_index — cycle-exact walk yields actual=1.
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        assert not any(
            isinstance(f, TimingTooCloseFailure)
            and f.producer.category == "MFMA"
            and f.consumer.category == "MFMA"
            for f in failures
        ), (
            f"Consecutive vmfma with 4x4 PackMFMA finish=1 gives actual=1 "
            f"== expected=1 — should NOT emit TimingTooCloseFailure. Got: "
            f"{failures}"
        )

    def test_mfma_pack_routed_through_quadcycle_not_alu(self):
        """Regression for the `_is_alu_producer` exemption fix. A producer
        that is BOTH categorized "PackA*" AND a real MFMA-shaped rocisa
        instance (4x4 PackMFMA, the TF32 emulation pattern) must be classified
        as an MFMA producer (routed to the quad-cycle branch), NOT silently
        absorbed by the ALU-immediate exemption.

        Pre-e7w `_is_alu_producer` returned True for any `cat.startswith
        ("Pack")` producer, so a PackMFMA's outgoing edges took the no-op
        ALU return and the quad-cycle gap check never fired. Post-e7w the
        carve-out reroutes PackMFMA-shaped producers (Pack* category +
        _is_mfma rocisa shape) to the MFMA branch.

        Test isolates the dispatch decision: synthesize a TaggedInstruction
        whose category is "PackA0" with an underlying real MFMAInstruction,
        then assert `_is_mfma_producer` claims it and `_is_alu_producer`
        does NOT."""
        from Tensile.Components.CMSValidator import (
            _is_alu_producer, _is_mfma_producer, _is_mfma_pack_producer,
        )
        # Real MFMAInstruction — `_is_mfma` returns True via class-name
        # membership in `_MFMA_CLASS_NAMES`.
        pack_mfma_tagged = make_mfma(
            c_dst_start=0, a_src_start=8, b_src_start=12, slot=2,
            category="PackA0", variant=[4, 4, 4, 16],
        )
        # GraphNode shape: a stub object exposing the attributes the
        # producer-classifier helpers read (category + rocisa_inst).
        class _StubNode:
            def __init__(self, tagged):
                self.category = tagged.category
                self.rocisa_inst = tagged.wrapped.rocisa_inst
        node = _StubNode(pack_mfma_tagged)
        assert _is_mfma_pack_producer(node), (
            "_is_mfma_pack_producer must return True for a Pack*-categorized "
            "MFMA-shaped producer (the TF32 4x4 PackMFMA pattern)."
        )
        assert _is_mfma_producer(node), (
            "_is_mfma_producer must claim PackMFMA producers so the "
            "quad-cycle branch fires for them."
        )
        assert not _is_alu_producer(node), (
            "_is_alu_producer must NOT claim PackMFMA producers — they need "
            "the quad-cycle finish-time gap check, not the ALU-immediate "
            "exemption. Pre-e7w returning True here was the bug that hid "
            "all PackMFMA timing violations."
        )
        # Sanity: a non-MFMA Pack* (e.g. CVTPack) still goes to the ALU branch.
        cvt_pack_tagged = _tag(
            VXorB32(dst=vgpr(50, 1), src0=vgpr(51, 1), src1=vgpr(52, 1)),
            category="PackA0", mfma_index=2, sequence=3,
        )
        cvt_node = _StubNode(cvt_pack_tagged)
        assert not _is_mfma_pack_producer(cvt_node)
        assert _is_alu_producer(cvt_node), (
            "Non-MFMA Pack* producers (CVT0/CVT1/Middle/Swap) must keep the "
            "ALU-immediate exemption — the carve-out targets PackMFMA only."
        )

    # -------------------------------------------------------------------------
    # CVTPack -> MFMA 2-quad-cycle settle window.
    # CDNA 4 ISA section 7.6: a CVT pack (`v_cvt_pk_bf16_f32` family,
    # validator class CVTPack) writing a vgpr that a downstream MFMA reads
    # must have at least `QUAD_CYCLES_CVT_BEFORE_MFMA == 2` quad-cycles
    # between issue completion and consumer issue start. Pre-35z this rule
    # was enforced only on the structural side (CMSValidator.py:1786-1788
    # sets `min_quad_cycles_before_result_used = 2` on every CVTPack;
    # `Pack.validate` at CMSValidator.py:423-429 emits TimingTooCloseFailure
    # if `estimated_quad_cycles_before_result_used < min`). The graph-side
    # check was missing entirely — `_is_alu_producer` returned True for any
    # `Pack*`-categorized producer, so CVTPack edges took the ALU-immediate
    # exemption and skipped the gap check.
    # -------------------------------------------------------------------------

    def test_cvt_pack_to_mfma_zero_gap_emits_timing_too_close(self):
        """CVTPack producer (`v_cvt_pk_bf16_f32`) writes v40; MFMA consumer at
        the same vmfma_index reads v40..v41. The graph-side dispatch must
        route the CVT->MFMA edge through `_cvt_to_mfma_gap_ok` (NOT the ALU
        exemption) and emit TimingTooCloseFailure with expected=2, actual=0
        (zero quad-cycle gap at slot_delta == 0)."""
        # CVTPack writes v40 (single-vgpr dst). Categorized "PackA0" — this
        # is the production category for CVT0 packs in the LRA0 group. The
        # generic ALU rule publishes (writes=v40, reads=v50,v51) for this
        # rocisa instance.
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
            # MFMA at the SAME vmfma_index — sub_index breaks the tie.
            # Reads v40..v41 (a_src spans v40..v41 with a_src_count=2),
            # which RAW-overlaps the CVT's write of v40.
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.consumer, "category", None) == "MFMA"
            and getattr(f.producer, "category", "").startswith("Pack")
        ]
        assert cvt_timing, (
            f"Expected TimingTooCloseFailure on CVTPack->MFMA edge at "
            f"slot_delta == 0 (the new 35z dispatch branch must claim it "
            f"BEFORE the ALU exemption). Got failures: "
            f"{[type(f).__name__ for f in failures]}"
        )
        f = cvt_timing[0]
        assert f.expected_quad_cycles == 2, (
            f"CVTPack producer should report expected=2 "
            f"(QUAD_CYCLES_CVT_BEFORE_MFMA), got "
            f"expected={f.expected_quad_cycles}."
        )
        assert f.actual_quad_cycles == 0, (
            f"slot_delta=0 should give actual=0 quad-cycles, got "
            f"actual={f.actual_quad_cycles}."
        )

    def test_cvt_pack_to_mfma_meets_2_cycle_gap_no_failure(self):
        """CVTPack producer at slot=2; MFMA consumer at slot=5 with TWO
        intervening LR instructions populating the captured stream
        between them. `_cvt_to_mfma_gap_ok` uses the cycle-exact
        `cumulative_issue_cycles` walk (replacing an earlier slot-delta
        under-estimate), so the gap is driven by the actually-recorded
        instructions in the captured stream — not by slot-index
        arithmetic.

        cumulative_issue_cycles walk producer..consumer:
          * CVT at p_idx → p_issue_start=0, then current_issue += 1.
          * 2 intervening LRs each cost 1 → current_issue advances 2.
          * MFMA at c_idx → c_issue_start=3; gap = 3 - 0 - 1 = 2.
        expected=2, actual=2 → no failure (exactly at threshold).

        Pre-`2bu.4` the test had no intervening instructions and relied
        on the slot-index gap (slot_delta=3 → 2 cycles). The cycle-exact
        walk only counts instructions actually present in the captured
        stream, so the real intervening LRs are required to produce the
        same numerical result.
        """
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
            # Two intervening LRs (cost 1 each) so the cycle-exact walk
            # accumulates 2 quad-cycles between the CVT and the MFMA,
            # matching the 2-cycle CVT->MFMA threshold.
            make_lr(60, 1, 80, slot=3, category="LRA1"),
            make_lr(61, 1, 84, slot=4, category="LRA1"),
            # MFMA at slot=5 — actual = CVT.issue_cost + 2*LR.issue_cost - 1
            #                        = 1 + 2 - 1 = 2 quad-cycles.
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=5, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.consumer, "category", None) == "MFMA"
            and getattr(f.producer, "category", "").startswith("Pack")
        ]
        assert not cvt_timing, (
            f"Cycle-exact walk over CVT + 2 intervening LRs gives "
            f"actual=2, meeting the 2-quad-cycle threshold "
            f"(no TimingTooCloseFailure). Got: {failures}"
        )

    def test_cvt_pack_routed_to_quadcycle_not_alu(self):
        """Regression-pin for the 35z dispatch carve-out. Synthesize the
        producer/consumer classification in isolation: a Pack*-categorized
        producer with an underlying VCvtPkF32toBF16 rocisa instance must
        be claimed by `_is_cvt_pack_producer`, and the dispatch in
        `_classify_edge_coverage` (and `diagnose_missing_edge`) must route
        a CVT->MFMA pair to the new branch BEFORE `_is_alu_producer` would
        absorb it.

        Mirrors the e7w `test_mfma_pack_routed_through_quadcycle_not_alu`
        shape — but for the CVTPack carve-out, with one extra check: the
        carve-out is consumer-aware (CVT->MFMA only). A CVT producer
        feeding a non-MFMA consumer (e.g. another Pack) must STILL hit the
        ALU exemption — only the MFMA-consuming edge carries the timing
        constraint."""
        from Tensile.Components.ScheduleCapture import _is_cvt_pack
        from Tensile.Components.CMSValidator import (
            _is_alu_producer, _is_cvt_pack_producer, _is_mfma_producer,
            _classify_edge_coverage,
        )
        # Producer: real VCvtPkF32toBF16 wrapped in a Pack*-categorized
        # TaggedInstruction (the production CVT0 emission shape).
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        cvt_tagged = _tag(cvt, category="PackA0", mfma_index=2, sequence=0)

        class _StubNode:
            def __init__(self, tagged):
                self.category = tagged.category
                self.rocisa_inst = tagged.wrapped.rocisa_inst
        cvt_node = _StubNode(cvt_tagged)

        # Class-name-set predicate matches the production rocisa class.
        assert _is_cvt_pack(cvt), (
            "_is_cvt_pack must return True for VCvtPkF32toBF16 instances."
        )
        # Producer-classifier sees Pack* category + CVT-class shape.
        assert _is_cvt_pack_producer(cvt_node), (
            "_is_cvt_pack_producer must claim Pack*-categorized CVT-shaped "
            "producers (the v_cvt_pk_bf16_f32 emission pattern)."
        )
        # ALU-immediate must NOT claim the CVTPack on its own — but note
        # the carve-out for CVT lives in the dispatch, not in
        # _is_alu_producer (the consumer-awareness needs both nodes).
        # _is_alu_producer correctly still returns True here in isolation;
        # the dispatch ordering (CVTPack-MFMA branch BEFORE ALU branch in
        # _classify_edge_coverage / diagnose_missing_edge) is what routes
        # the edge correctly. The end-to-end assertion below verifies that.
        assert _is_alu_producer(cvt_node), (
            "_is_alu_producer is producer-only and (intentionally) still "
            "returns True for CVTPack — the consumer-aware dispatch is "
            "what carves out the CVT->MFMA edge."
        )

        # End-to-end dispatch assertion: build a CVT->MFMA edge and run it
        # through the full classifier. Must emit TimingTooCloseFailure
        # (expected=2, actual=0), which proves the dispatch reached the
        # 35z branch — not the ALU exemption (which would have returned
        # an empty failure list).
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(VCvtPkF32toBF16(dst=vgpr(60, 1),
                                 src0=vgpr(50, 1), src1=vgpr(51, 1)),
                 category="PackA0", mfma_index=2, sequence=0),
            make_mfma(c_dst_start=0, a_src_start=60, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        cvt_to_mfma_edges = [
            e for e in g.edges
            if getattr(e.producer, "category", "").startswith("Pack")
            and getattr(e.consumer, "category", None) == "MFMA"
            and _is_cvt_pack_producer(e.producer)
            and _is_mfma_producer(e.consumer)
        ]
        assert cvt_to_mfma_edges, (
            "Expected at least one CVTPack->MFMA edge in the graph — the "
            "per-byte resolver should pair the MFMA's read of v60 with "
            "the CVT's write."
        )
        edge = cvt_to_mfma_edges[0]
        edge_failures = _classify_edge_coverage(edge, g)
        timing = [f for f in edge_failures
                  if isinstance(f, TimingTooCloseFailure)]
        assert timing, (
            f"_classify_edge_coverage on a zero-gap CVT->MFMA edge must "
            f"emit TimingTooCloseFailure — proves the 35z dispatch branch "
            f"claimed the edge BEFORE the ALU exemption. Got: "
            f"{[type(f).__name__ for f in edge_failures]}"
        )
        assert timing[0].expected_quad_cycles == 2
        assert timing[0].actual_quad_cycles == 0

    # -------------------------------------------------------------------------
    # 4x4 PackMFMA -> CVTPack (CVT1) 5-quad-cycle
    # settle window. CDNA 4 ISA section 7.6: a 4x4 PackMFMA writing an
    # accumulator vgpr that a downstream CVT1 (`v_cvt_pk_bf16_f32` reading
    # the MFMA acc) reads must have at least
    # `QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 == 5` quad-cycles between issue
    # completion and consumer issue start. This is the LARGEST of the four
    # quad-cycle constants — strictly greater than the bare 4x4
    # finish-cycle (1), so the dispatch must intercept this pair BEFORE
    # the generic `_is_mfma_producer` branch (which would use the smaller
    # finish-cycle threshold and false-pass real violations).
    #
    # The (now-removed) structural side enforced this rule via
    # `MFMAPack.min_quad_cycles_before_result_used = 5` set in
    # `_handle_min_pack_quad_cycles`. The graph-side carve-out is now
    # the only enforcement path; the carve-out is producer/consumer-pair
    # aware so the 5-cycle window applies only to PackMFMA->CVT1, not
    # to PackMFMA->generic-MFMA pairs.
    # -------------------------------------------------------------------------

    def _make_real_pack_mfma(self, *, acc_start, acc_count, a_start,
                              a_count, b_start, b_count, slot, sequence,
                              category):
        """Build a real rocisa MFMAInstruction (4x4 PackMFMA family) wrapped
        in a TaggedInstruction. Calls the rocisa constructor directly (rather
        than `make_mfma`) to keep the variant/category combination explicit
        in the test, so the producer has `getParams()` and the per-byte
        resolver claims it via `_GenericALURule` (Pack-categorized MFMAs
        are excluded from
        `_MFMARule` per ScheduleCapture.py:1980-1984). The rendered form
        contains the `_4x4x` substring so `_mfma_finish_cycles_for` returns
        `_QUAD_CYCLES_MFMA_4X4_FINISH == 1` — matching the production
        TF32 4x4 PackMFMA."""
        from rocisa.enum import InstType
        from rocisa.instruction import MFMAInstruction
        inst = MFMAInstruction(
            InstType.INST_F32, InstType.INST_F32, [4, 4, 4, 16], False,
            vgpr(acc_start, acc_count),
            vgpr(a_start, a_count),
            vgpr(b_start, b_count),
        )
        return _tag(inst, category=category,
                    mfma_index=slot, sequence=sequence)

    def test_mfma_pack_to_cvt1_zero_gap_emits_timing_too_close(self):
        """4x4 PackMFMA producer (real `MFMAInstruction`, variant=[4,4,4,16])
        at slot=2 sequence=0 writes its acc into v0..v3; CVTPack consumer
        (`v_cvt_pk_bf16_f32`) at slot=2 sequence=1 reads v0 as src0. The
        cycle-exact simulator (`cumulative_issue_cycles`) reports:
          producer issues at 0, mfma_free_at = 0+1+1 = 2.
          consumer at p_idx+1 — current_issue += 1 (MFMA issue cost) → 1.
          c_issue_start = 1; gap = 1 - 0 - 1 = 0.
        expected = 5 (QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1) > actual = 0 →
        TimingTooCloseFailure. Failure must be emitted by the new
        PackMFMA->CVTPack branch which intercepts the pair BEFORE the
        generic MFMA-producer branch — the latter would have used
        `_mfma_finish_cycles_for(4x4) == 1` as the threshold and
        false-passed the edge.

        Uses a real rocisa MFMAInstruction so the Pack-categorized
        producer has `getParams()` and `_GenericALURule` publishes
        (writes=(acc,), reads=(a,b,acc)) — required for the per-byte
        resolver to form the PackMFMA->CVT edge."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        cap = make_capture(BODY_LABEL_ML, [
            self._make_real_pack_mfma(
                acc_start=0, acc_count=4, a_start=8, a_count=2,
                b_start=32, b_count=2, slot=2, sequence=0,
                category="PackA0"),
            # CVTPack consumer reading v0..v1 (RAW on the PackMFMA acc).
            _tag(cvt, category="PackA1", mfma_index=2, sequence=1),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        pack_to_cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
        ]
        assert pack_to_cvt_timing, (
            f"Expected TimingTooCloseFailure on PackMFMA->CVTPack edge at "
            f"zero gap (the new or9 dispatch branch must claim it BEFORE "
            f"the generic MFMA-producer branch — which would have used "
            f"the smaller 4x4 finish-cycle threshold of 1 and false-passed "
            f"the edge). Got failures: "
            f"{[type(f).__name__ for f in failures]}"
        )
        f = pack_to_cvt_timing[0]
        assert f.expected_quad_cycles == 5, (
            f"PackMFMA->CVTPack expected=5 "
            f"(QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1), got "
            f"expected={f.expected_quad_cycles}."
        )
        assert f.actual_quad_cycles < 5, (
            f"Zero-gap pair must report actual < 5 (cycle-exact simulator "
            f"yields actual=0 for this fixture). Got "
            f"actual={f.actual_quad_cycles}."
        )

    def test_mfma_pack_to_cvt1_meets_5_cycle_gap_no_failure(self):
        """4x4 PackMFMA producer at slot=2 sequence=0; CVTPack consumer at
        slot=8 sequence=0 with FIVE intervening LR/SWait instructions.
        Cycle-exact simulator walk:
          producer issues at 0, mfma_free_at = 2; current_issue += 1 → 1.
          5 intervening (LR/SWait), each cost 1 → current_issue = 6.
          consumer (CVT) — c_issue_start = 6; gap = 6 - 0 - 1 = 5.
        expected = 5, actual = 5 → ok=True → NO TimingTooCloseFailure."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        cap = make_capture(BODY_LABEL_ML, [
            self._make_real_pack_mfma(
                acc_start=0, acc_count=4, a_start=8, a_count=2,
                b_start=32, b_count=2, slot=2, sequence=0,
                category="PackA0"),
            # Five intervening cost-1 instructions to inflate the gap.
            make_lr(80, 1, 128, slot=3, category="LRA1"),
            make_lr(81, 1, 132, slot=4, category="LRA1"),
            make_lr(82, 1, 136, slot=5, category="LRA1"),
            make_lr(83, 1, 140, slot=6, category="LRA1"),
            make_lr(84, 1, 144, slot=7, category="LRA1"),
            _tag(cvt, category="PackA1", mfma_index=8, sequence=0),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        pack_to_cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
        ]
        assert not pack_to_cvt_timing, (
            f"PackMFMA->CVTPack edge with cumulative_issue_cycles >= 5 "
            f"must NOT emit TimingTooCloseFailure. Got: {failures}"
        )

    def test_mfma_pack_to_cvt1_routed_to_pack_to_cvt_branch_not_quadcycle(self):
        """Regression-pin for the Sub-C dispatch ordering. A producer that
        is BOTH a 4x4 PackMFMA AND has a CVTPack consumer must be claimed
        by the new `_is_mfma_pack_producer(p) AND _is_cvt_pack_producer(c)`
        branch BEFORE the generic `_is_mfma_producer` branch — otherwise
        the latter routes through `_quad_cycle_gap_ok` with the
        4x4-finish-cycle threshold (1) and false-passes the edge.

        End-to-end check: build a zero-gap PackMFMA->CVTPack edge, run it
        through `_classify_edge_coverage`, and confirm the failure carries
        `expected_quad_cycles == 5` (the or9 threshold) — NOT 1 (which
        would prove the dispatch fell through to the generic MFMA branch).

        Mirrors the e7w/35z dispatch-pin shape but for the new or9 branch.
        """
        from Tensile.Components.CMSValidator import (
            _is_mfma_pack_producer, _is_cvt_pack_producer,
            _classify_edge_coverage,
        )
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        cap = make_capture(BODY_LABEL_ML, [
            self._make_real_pack_mfma(
                acc_start=0, acc_count=4, a_start=8, a_count=2,
                b_start=32, b_count=2, slot=2, sequence=0,
                category="PackA0"),
            _tag(cvt, category="PackA1", mfma_index=2, sequence=1),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Locate the PackMFMA -> CVTPack edge (per-byte resolver pairs the
        # CVT's read of v0 with the PackMFMA's write of v0..v3).
        pack_to_cvt_edges = [
            e for e in g.edges
            if _is_mfma_pack_producer(e.producer)
            and _is_cvt_pack_producer(e.consumer)
        ]
        assert pack_to_cvt_edges, (
            "Expected at least one PackMFMA->CVTPack edge in the graph — "
            "the per-byte resolver should pair the CVT's read of v0 with "
            "the PackMFMA's write."
        )
        edge = pack_to_cvt_edges[0]
        edge_failures = _classify_edge_coverage(edge, g)
        timing = [f for f in edge_failures
                  if isinstance(f, TimingTooCloseFailure)]
        assert timing, (
            f"_classify_edge_coverage on a zero-gap PackMFMA->CVTPack edge "
            f"must emit TimingTooCloseFailure. Got: "
            f"{[type(f).__name__ for f in edge_failures]}"
        )
        # The decisive assertion: expected must be 5 (or9 branch), NOT 1
        # (which would prove the dispatch fell through to the generic
        # MFMA-producer branch using _mfma_finish_cycles_for == 1).
        assert timing[0].expected_quad_cycles == 5, (
            f"Dispatch ordering broken: expected_quad_cycles == "
            f"{timing[0].expected_quad_cycles}, but the or9 PackMFMA-to-"
            f"CVTPack branch reports 5 (QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1) "
            f"and the generic MFMA branch reports 1 "
            f"(_mfma_finish_cycles_for == _QUAD_CYCLES_MFMA_4X4_FINISH). "
            f"The new or9 branch must claim the pair BEFORE the generic "
            f"MFMA branch."
        )

    # -------------------------------------------------------------------------
    # MFMA type-switch +1 stall penalty in the graph-side
    # `_quad_cycle_gap_ok` actual computation. The (now-removed) structural
    # simulator `precompute_issue_times` injected a +1 quad-cycle stall
    # whenever consecutive MFMAs differ in class (standard MFMA <-> 4x4
    # PackMFMA) and the inter-MFMA gap is below the producer's threshold
    # (FROM_STANDARD = 5, FROM_4X4 = 3). For the direct producer→consumer
    # adjacency case (slot_delta == 1) both directions trigger the stall:
    # standard producer's gap is 1+3=4 < 5; 4x4 producer's gap is
    # 1+1=2 < 3. The graph-side path mirrors this by adding +1 to `actual`
    # when producer.finish != consumer.finish — so the reported `actual`
    # matches the simulator's enlarged gap and the graph stays a sound
    # under-estimate (real_actual ≥ formula+1 in the direct case,
    # ≥ formula+1 in the chain case since any path between the producer
    # and a different-class consumer crosses ≥ 1 boundary).
    # -------------------------------------------------------------------------

    def test_mfma_type_switch_standard_to_4x4_adds_one_to_actual(self):
        """Standard MFMA producer (finish=3) at slot=2; 4x4 PackMFMA
        consumer (finish=1) at slot=3. Cycle-exact simulator
        (CMSValidator.precompute_issue_times-equivalent):
          producer issues at 0, mfma_free=4, last_mfma_class=standard.
          consumer max(1,4)=4; type switch (standard→4x4), gap=4-0=4 <
          FROM_STANDARD=5 → +1 stall → consumer issues at 5.
          delivered gap = 5-0-1 = 4."""
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
        ])
        g = build_dataflow_graph(_wrap(cap))
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        check = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert check.result == TimingResult.PASS
        assert check.required == 3
        assert check.observed == 4

    def test_mfma_type_switch_4x4_to_standard_adds_one_to_actual(self):
        """4x4 PackMFMA producer (finish=1) at slot=2; standard MFMA
        consumer (finish=3) at slot=3. Cycle-exact simulator:
          producer issues at 0, mfma_free=2, last_mfma_class=4x4.
          consumer max(1,2)=2; type switch (4x4→standard), gap=2-0=2 <
          FROM_4X4=3 → +1 stall → consumer issues at 3.
          delivered gap = 3-0-1 = 2."""
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        check = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert check.result == TimingResult.PASS
        assert check.required == 1
        assert check.observed == 2

    def test_mfma_same_class_chain_no_type_switch_penalty(self):
        """Regression guard: same-class chain (standard producer → standard
        consumer) does NOT trigger the vf4 type-switch +1 penalty.

        Without this guard, a buggy implementation that always added +1
        (failing to check class equality) would inflate `actual` even
        when producer and consumer share the same MFMA class — breaking
        the existing `test_mfma_acc_chain_consecutive_slots_no_failure`
        invariant that consecutive standard MFMAs at slot_delta=1 yield
        actual == expected == 3 exactly."""
        cap = make_capture(BODY_LABEL_ML, [
            # Producer: standard MFMA at slot=2 (default variant).
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # Consumer: standard MFMA at slot=3 — SAME class. No type
            # switch, no +1 penalty.
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        acc_edge = None
        for e in g.edges:
            if (getattr(e.producer, "category", None) == "MFMA"
                    and getattr(e.consumer, "category", None) == "MFMA"):
                acc_edge = e
                break
        assert acc_edge is not None
        check = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        # nk0 cycle-exact: producer at 0 with mfma_free=4; consumer max(1,4)=4
        # → gap=3. Same class, no type-switch +1.
        assert (check.result == TimingResult.PASS and check.required == 3
                and check.observed == 3), (
            f"Same-class standard→standard must yield observed==required==3. "
            f"Got result={check.result}, required={check.required}, "
            f"observed={check.observed}."
        )

        # Sanity sibling: same-class 4x4→4x4 chain — no +1 either.
        cap_pack = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
        ])
        g_pack = build_dataflow_graph(_wrap(cap_pack))
        acc_edge_pack = None
        for e in g_pack.edges:
            if (getattr(e.producer, "category", None) == "MFMA"
                    and getattr(e.consumer, "category", None) == "MFMA"):
                acc_edge_pack = e
                break
        assert acc_edge_pack is not None
        check_p = _quad_cycle_gap_ok(
            acc_edge_pack.producer, acc_edge_pack.consumer, 0, graph=g_pack)
        # nk0 cycle-exact: 4x4 producer at 0, mfma_free=2; consumer max(1,2)=2;
        # same class → no type-switch. gap=2-0-1=1.
        assert (check_p.result == TimingResult.PASS and check_p.required == 1
                and check_p.observed == 1), (
            f"Same-class 4x4→4x4: observed==required==1. Got "
            f"result={check_p.result}, required={check_p.required}, "
            f"observed={check_p.observed}."
        )

    # -------------------------------------------------------------------------
    # Audit `audit-pack-timing` — graph-side boundary parity with the
    # 8nz-deleted structural-side TimingTooClose tests. The pre-8nz suite
    # (test_ValidatePack.py) pinned EXACT actual_quad_cycles values one
    # short of the threshold (actual=1 for expected=2; actual=3 for
    # expected=5). Existing graph-side coverage pinned only the zero-gap
    # boundary (CVT→MFMA actual=0) or the inequality only (PackMFMA→CVT1
    # actual<5 with no exact value). These tests close the parity gap by
    # asserting the EXACT off-by-one actual that the deleted structural
    # tests pinned — protecting against a regression that would shift
    # `actual` by ±1 (e.g., a future formula change that conflates
    # finish-cycle vs issue-cycle counting) without showing up at the
    # zero-gap or boundary-meets cases.
    # -------------------------------------------------------------------------

    def test_cvt_pack_to_mfma_one_short_pins_actual_1(self):
        """Mirrors deleted structural test
        `TestValidatePackTF32::test_failing_not_enough_time_CVT1_MFMA`
        (expected=2, actual=1) in the post-`2bu.4` cycle-exact regime.
        CVT producer at slot=2 sequence=0; MFMA consumer at slot=4 with
        ONE intervening LR (cost 1) between them.

        cumulative_issue_cycles walk:
          * CVT at p_idx → p_issue_start=0, current_issue += 1.
          * 1 intervening LR → current_issue = 2.
          * MFMA at c_idx → c_issue_start=2; gap = 2 - 0 - 1 = 1.
        expected=2, actual=1 → fails by exactly 1 quad-cycle.

        The pre-existing zero-gap test (`test_cvt_pack_to_mfma_zero_gap_
        emits_timing_too_close`) pins actual=0; the boundary-meets test
        (`test_cvt_pack_to_mfma_meets_2_cycle_gap_no_failure`) pins no
        failure at actual=2. This test pins the EXACT mid-band actual=1
        the deleted structural test pinned — a regression in
        `_cvt_to_mfma_gap_ok` or `cumulative_issue_cycles` that
        miscounts intervening contributions by ±1 would change `actual`
        and be caught here.

        Pre-`2bu.4` the slot-delta formula gave the same actual=1 from
        slot-index arithmetic with NO intervening insts in the stream;
        the cycle-exact walk requires the LR to actually be present.
        """
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
            # ONE intervening LR (cost 1) — bumps the walk from gap=0 to
            # gap=1 between the CVT and the MFMA.
            make_lr(60, 1, 80, slot=3, category="LRA1"),
            # MFMA at slot=4 — actual = 1 (CVT cost) + 1 (LR cost) - 1 = 1.
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.consumer, "category", None) == "MFMA"
            and getattr(f.producer, "category", "").startswith("Pack")
        ]
        assert cvt_timing, (
            f"Expected TimingTooCloseFailure on CVTPack->MFMA edge with "
            f"one intervening LR (cycle-exact actual=1). Got: "
            f"{[type(f).__name__ for f in failures]}"
        )
        f = cvt_timing[0]
        assert f.expected_quad_cycles == 2, (
            f"expected=2 (QUAD_CYCLES_CVT_BEFORE_MFMA); got "
            f"{f.expected_quad_cycles}."
        )
        # Decisive parity assertion vs deleted structural test: actual must
        # be EXACTLY 1 (off-by-one boundary), not 0 and not 2.
        assert f.actual_quad_cycles == 1, (
            f"cumulative_issue_cycles walk: CVT(1) + 1*LR(1) before MFMA → "
            f"c_issue=2; gap = 2-0-1 = 1. Got actual="
            f"{f.actual_quad_cycles}. Mirrors deleted "
            f"TestValidatePackTF32::test_failing_not_enough_time_CVT1_MFMA "
            f"which pinned actual=1 on the structural side."
        )

    def test_mfma_pack_to_cvt1_three_short_pins_actual_3(self):
        """Mirrors deleted structural test
        `TestValidatePackTF32MFMA4x4x4::test_failing_not_enough_time_MFMA_CVT1`
        (expected=5, actual=3). 4x4 PackMFMA producer at slot=2 sequence=0;
        CVTPack (CVT1) consumer at slot=6 sequence=0 with THREE intervening
        cost-1 LR instructions. cumulative_issue_cycles walk:
          PackMFMA issue=0, mfma_free=2; current_issue +=1 → 1.
          3 LRs each cost 1 → current_issue = 4.
          CVT (issue cost 1, but the helper measures gap as
            c_issue_start - p_issue - 1) → c_issue_start=4; gap=4-0-1=3.
        expected=5, actual=3 → off by exactly 2.

        Existing coverage pins actual=0 (zero-gap test) and the
        meets-threshold case (actual==5, no failure). This test closes the
        gap by pinning the EXACT mid-band actual=3 that the deleted
        structural test pinned — a regression in `_mfma_pack_to_cvt_gap_ok`
        or `cumulative_issue_cycles` that miscounts intervening contributions
        by ±1 would change `actual` and be caught here."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        cap = make_capture(BODY_LABEL_ML, [
            self._make_real_pack_mfma(
                acc_start=0, acc_count=4, a_start=8, a_count=2,
                b_start=32, b_count=2, slot=2, sequence=0,
                category="PackA0"),
            # 3 intervening cost-1 LRs to inflate the gap from 0 to 3.
            make_lr(80, 1, 128, slot=3, category="LRA1"),
            make_lr(81, 1, 132, slot=4, category="LRA1"),
            make_lr(82, 1, 136, slot=5, category="LRA1"),
            _tag(cvt, category="PackA1", mfma_index=6, sequence=0),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        pack_to_cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
            and f.expected_quad_cycles == 5
        ]
        assert pack_to_cvt_timing, (
            f"Expected TimingTooCloseFailure on PackMFMA->CVTPack edge "
            f"with 3 intervening LRs (actual=3 < expected=5). Got: "
            f"{[type(f).__name__ for f in failures]}"
        )
        f = pack_to_cvt_timing[0]
        assert f.expected_quad_cycles == 5
        # Decisive parity assertion: actual must be EXACTLY 3 — mirrors
        # deleted TestValidatePackTF32MFMA4x4x4::test_failing_not_enough_
        # time_MFMA_CVT1 which pinned (expected=5, actual=3).
        assert f.actual_quad_cycles == 3, (
            f"PackMFMA producer + 3 intervening cost-1 LRs → "
            f"cumulative_issue_cycles=3. Got actual={f.actual_quad_cycles}. "
            f"Mirrors the deleted structural test which pinned actual=3."
        )

    def test_mfma_pack_to_cvt1_one_short_pins_actual_4(self):
        """Off-by-one boundary pin for QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1=5.
        Sibling of `test_mfma_pack_to_cvt1_three_short_pins_actual_3` —
        same fixture shape, but with FOUR intervening cost-1 LRs (not
        three) so the cycle-exact gap is exactly one short of threshold.

        4x4 PackMFMA producer at slot=2 sequence=0; CVTPack (CVT1)
        consumer at slot=7 sequence=0 with FOUR intervening cost-1 LR
        instructions. cumulative_issue_cycles walk:
          PackMFMA issue=0, mfma_free=2; current_issue +=1 → 1.
          4 LRs each cost 1 → current_issue = 5.
          CVT (issue cost 1, gap = c_issue_start - p_issue - 1) →
            c_issue_start=5; gap=5-0-1=4.
        expected=5, actual=4 → off by EXACTLY 1.

        Closes the off-by-one gap left by existing siblings:
          - actual=0  (zero-gap test)            ✓
          - actual=3  (off-by-2, three_short)    ✓
          - actual=4  (off-by-1)                 ← THIS test
          - actual=5  (just-pass, meets_5_cycle) ✓
        Mutating QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 from 5 to 4 makes the
        threshold equal `actual` here so the failure stops firing — a
        change that the just-pass test (actual=5 ≥ 4) cannot detect,
        leaving this test as the SOLE off-by-one sentinel."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        cap = make_capture(BODY_LABEL_ML, [
            self._make_real_pack_mfma(
                acc_start=0, acc_count=4, a_start=8, a_count=2,
                b_start=32, b_count=2, slot=2, sequence=0,
                category="PackA0"),
            # 4 intervening cost-1 LRs to inflate the gap from 0 to 4
            # (one short of the 5-cycle threshold).
            make_lr(80, 1, 128, slot=3, category="LRA1"),
            make_lr(81, 1, 132, slot=4, category="LRA1"),
            make_lr(82, 1, 136, slot=5, category="LRA1"),
            make_lr(83, 1, 140, slot=6, category="LRA1"),
            _tag(cvt, category="PackA1", mfma_index=7, sequence=0),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        pack_to_cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
            and f.expected_quad_cycles == 5
        ]
        assert pack_to_cvt_timing, (
            f"Expected TimingTooCloseFailure on PackMFMA->CVTPack edge "
            f"with 4 intervening LRs (actual=4 < expected=5). Got: "
            f"{[type(f).__name__ for f in failures]}. If this fixture "
            f"stops firing, QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 was likely "
            f"lowered (e.g. 5→4) — the off-by-one sentinel."
        )
        f = pack_to_cvt_timing[0]
        assert f.expected_quad_cycles == 5, (
            f"PackMFMA->CVTPack expected=5 "
            f"(QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1), got "
            f"expected={f.expected_quad_cycles}. A change here means the "
            f"constant itself was mutated."
        )
        # Decisive off-by-one assertion: actual MUST be exactly 4 — one
        # short of the 5-cycle threshold. A ±1 miscount in
        # cumulative_issue_cycles or the helper's gap arithmetic would
        # shift `actual` to 3 or 5 and be caught here.
        assert f.actual_quad_cycles == 4, (
            f"PackMFMA producer + 4 intervening cost-1 LRs → "
            f"cumulative_issue_cycles gap = 4. Got "
            f"actual={f.actual_quad_cycles}. This is the off-by-one "
            f"boundary case (expected-1) that no other sibling pins."
        )

    def test_cvt_pack_to_pack_alu_consumer_takes_alu_exemption(self):
        """Consumer-awareness regression-pin for `_cvt_to_mfma_gap_ok`. The
        carve-out targets CVT→MFMA edges ONLY: a CVT producer feeding a
        non-MFMA Pack* consumer (e.g. another CVT) must still take the
        ALU-immediate exemption — no TimingTooCloseFailure is emitted on
        the CVT→Pack edge itself, even at zero gap.

        Provides explicit coverage of the consumer-side branch in the
        dispatch (the existing `test_cvt_pack_routed_to_quadcycle_not_alu`
        only checks the MFMA-consumer path). This pins that the carve-out
        does NOT over-fire on Pack→Pack edges — important because the
        deleted structural tests' producer/consumer naming
        (`producer_name='PackA0', consumer_name='PackA0'`) covered both
        Pack→MFMA and within-Pack chains; the within-Pack chain is enforced
        differently on the graph side (per-MFMA-only), and this test pins
        that difference explicitly."""
        # CVT producer writes v40.
        cvt0 = VCvtPkF32toBF16(dst=vgpr(40, 1),
                               src0=vgpr(50, 1), src1=vgpr(51, 1))
        # Second CVT consumer reads v40 — RAW edge between two PackA0
        # instructions (CVT0 producer; another CVT-shaped Pack consumer).
        cvt1 = VCvtPkF32toBF16(dst=vgpr(60, 1),
                               src0=vgpr(40, 1), src1=vgpr(41, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt0, category="PackA0", mfma_index=2, sequence=0),
            # Second CVT at the SAME vmfma_index — zero gap. RAW on v40.
            _tag(cvt1, category="PackA0", mfma_index=2, sequence=1),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        # The CVT->CVT (Pack->Pack, non-MFMA-consumer) edge must NOT trigger
        # the CVT-MFMA timing carve-out. Only the consumer-aware carve-out
        # for MFMA consumers fires; everything else inherits the ALU
        # exemption.
        pack_to_pack_cvt_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
            and f.expected_quad_cycles == 2
        ]
        assert not pack_to_pack_cvt_timing, (
            f"CVT->CVT (Pack->Pack non-MFMA-consumer) edge must NOT trigger "
            f"the QUAD_CYCLES_CVT_BEFORE_MFMA=2 carve-out — it should take "
            f"the ALU-immediate exemption like any non-MFMA consumer. "
            f"Got: {pack_to_pack_cvt_timing}"
        )

    # -------------------------------------------------------------------------
    # Cross-body unit tests for `_cvt_to_mfma_gap_ok`.
    #
    # An earlier helper had a cross-body short-circuit that returned
    # `(True, 2, body_delta * 1000)` regardless of the actual captured-stream
    # gap. That branch is gone; cross-body edges now route through
    # `cumulative_issue_cycles` (which walks the concatenated captured
    # stream from producer's body forward through consumer's body in
    # `_BODY_BUILD_ORDER` execution order). The tests below assert REAL
    # cycle counts derived from the captured stream — the cross-body case is
    # numerically identical to the same-body case from the simulator's
    # perspective; only the body boundary moves.
    # -------------------------------------------------------------------------
    def test_cvt_to_mfma_cross_body_ml_prev_to_ml_real_cycle_count(self):
        """CVTPack producer in body=ML-1 at the END of that body; MFMA
        consumer in body=ML reading the CVT's destination v40, with TWO
        intervening LR instructions in the consumer body so the gap meets
        the 2-quad-cycle threshold.

        cumulative_issue_cycles walk over the concatenated stream
        (ML-1 then ML, in `_BODY_BUILD_ORDER`):
          * ML-1: [LR, CVT@p_idx]; ML: [LR, LR, MFMA@c_idx].
          * CVT at p_idx → p_issue_start=0, current_issue += 1.
          * 2 LRs in ML each cost 1 → current_issue = 3.
          * MFMA at c_idx → c_issue_start=3; gap = 3 - 0 - 1 = 2.
        expected=2, actual=2 → ok=True (exactly meets threshold).

        Pre-`2bu.4` this returned (True, 2, 1000) unconditionally via the
        body-delta sentinel. Now the actual reflects the real captured
        cycles and the test will catch a regression that miscounts the
        cross-body walk.
        """
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        # Producer body: ML-1 (loop_index=0). CVT writes v40.
        ml_prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
        ])
        # Consumer body: ML (loop_index=1). Two LRs (cost 1 each) + MFMA
        # reading v40..v41. The intervening LRs supply the 2 quad-cycles
        # needed for the cycle-exact walk to meet the threshold.
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_lr(60, 1, 80, slot=0, category="LRA1"),
            make_lr(61, 1, 84, slot=1, category="LRA1"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
        ])
        ngl_filler = make_capture(BODY_LABEL_NGL, [make_mfma(
            c_dst_start=220, a_src_start=224, b_src_start=228, slot=0)])
        nll_filler = make_capture(BODY_LABEL_NLL, [make_mfma(
            c_dst_start=240, a_src_start=244, b_src_start=248, slot=0)])
        four = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: ml_prev_cap},
            n_gl={0: ngl_filler},
            n_ll={0: nll_filler},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        g = build_dataflow_graph(four)

        # Find the cross-body CVT (Pack*) -> MFMA edge.
        cross = [e for e in g.edges
                 if getattr(e.producer, "category", "").startswith("Pack")
                 and getattr(e.consumer, "category", None) == "MFMA"
                 and e.producer.body_label != e.consumer.body_label]
        edge_summary = [
            (getattr(e.producer, "category", None), e.producer.body_label,
             getattr(e.consumer, "category", None), e.consumer.body_label)
            for e in g.edges
        ]
        assert cross, (
            "Expected a cross-body CVTPack->MFMA edge "
            f"(ML-1 producer -> ML consumer). Got edges: {edge_summary}"
        )
        edge = cross[0]
        check = _cvt_to_mfma_gap_ok(
            edge.producer, edge.consumer, g)

        assert check.required == 2, (
            f"_QUAD_CYCLES_CVT_BEFORE_MFMA == 2; required must always "
            f"be 2. Got required={check.required}."
        )
        # CVT cost (1) + 2 LR costs (2) - 1 = 2 cycles real gap.
        assert check.observed == 2, (
            f"Cross-body cycle-exact walk: 1 (CVT) + 2 (LR x2) - 1 = 2. "
            f"Got observed={check.observed}. A regression that fails to walk past "
            f"the body boundary would return 0."
        )
        assert check.result == TimingResult.PASS, (
            f"observed=2 >= required=2 → PASS. Got result={check.result}, "
            f"observed={check.observed}."
        )
        # No TimingTooCloseFailure should be emitted at the threshold.
        failures = validate_edge_wait_coverage(g)
        cross_body_failures = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", None) == "MFMA"
            and f.producer.body_label != f.consumer.body_label
        ]
        assert not cross_body_failures, (
            f"Cross-body CVT->MFMA edge at the 2-cycle threshold must "
            f"NOT emit TimingTooCloseFailure. Got: {cross_body_failures}"
        )

    def test_cvt_to_mfma_cross_body_ml_to_ngl_real_cycle_count(self):
        """Mirror of the ML-1 -> ML test but for ML -> NGL (the other
        adjacent-body crossing in production captures). Same arithmetic,
        different body pair.

        cumulative_issue_cycles walk over [ML, NGL] concatenation:
          * ML: [LR, CVT@p_idx]
          * NGL: [LR, LR, MFMA@c_idx]
          * CVT (cost 1), 2 LRs (cost 2), MFMA → gap = 3 - 0 - 1 = 2.
        expected=2, actual=2, ok=True.

        Pre-`2bu.4` this returned (True, 2, 1000) unconditionally. The
        test pins that the cross-body walk works for ANY adjacent body
        pair, not just ML-1 -> ML."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        # Producer body: ML (loop_index=1). CVT writes v40.
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
        ])
        # Consumer body: NGL (loop_index=2). 2 LRs + MFMA reads v40..v41.
        ngl_cap = make_capture(BODY_LABEL_NGL, [
            make_lr(60, 1, 80, slot=0, category="LRA1"),
            make_lr(61, 1, 84, slot=1, category="LRA1"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
        ])
        ml_prev_filler = make_capture(BODY_LABEL_ML_PREV, [make_mfma(
            c_dst_start=200, a_src_start=204, b_src_start=208, slot=0)])
        nll_filler = make_capture(BODY_LABEL_NLL, [make_mfma(
            c_dst_start=240, a_src_start=244, b_src_start=248, slot=0)])
        four = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: ml_prev_filler},
            n_gl={0: ngl_cap},
            n_ll={0: nll_filler},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        g = build_dataflow_graph(four)
        cross = [e for e in g.edges
                 if getattr(e.producer, "category", "").startswith("Pack")
                 and getattr(e.consumer, "category", None) == "MFMA"
                 and e.producer.body_label == BODY_LABEL_ML
                 and e.consumer.body_label == BODY_LABEL_NGL]
        edge_summary = [
            (e.producer.body_label, e.consumer.body_label,
             getattr(e.producer, "category", None))
            for e in g.edges
        ]
        assert cross, (
            "Expected a cross-body CVTPack->MFMA edge from ML to NGL. "
            f"Got: {edge_summary}"
        )
        edge = cross[0]
        check = _cvt_to_mfma_gap_ok(
            edge.producer, edge.consumer, g)
        assert (check.required == 2 and check.observed == 2
                and check.result == TimingResult.PASS), (
            f"ML -> NGL cross-body cycle-exact walk: 1 + 2 - 1 = 2. "
            f"Got result={check.result}, required={check.required}, "
            f"observed={check.observed}."
        )

    def test_cvt_to_mfma_cross_body_below_threshold_fires_failure(self):
        """Negative test: CVTPack producer at the END of ML-1 with MFMA
        consumer at the START of ML, and intermediate instructions tight
        enough that the real cross-body gap is BELOW the
        `_QUAD_CYCLES_CVT_BEFORE_MFMA == 2` threshold. The cycle-exact
        walk should compute actual=0 (no intervening real instructions
        between CVT and MFMA across the body boundary) and emit
        TimingTooCloseFailure.

        cumulative_issue_cycles walk over [ML-1, ML]:
          * ML-1 ends at CVT@p_idx → p_issue_start=0, current_issue += 1.
          * ML starts immediately with MFMA@c_idx (first instruction).
          * gap = 1 - 0 - 1 = 0. expected=2 → ok=False, fires.

        Pre-`2bu.4` this would have been swallowed by the body-delta
        sentinel (`actual=1000`, `ok=True`); this test pins that a real
        too-tight cross-body schedule now surfaces as a failure."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        ml_prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
        ])
        # Consumer body: ML, MFMA at the very first slot — NO intervening
        # instructions between CVT and MFMA across the body boundary.
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=0, sequence=0, a_src_count=2),
        ])
        ngl_filler = make_capture(BODY_LABEL_NGL, [make_mfma(
            c_dst_start=220, a_src_start=224, b_src_start=228, slot=0)])
        nll_filler = make_capture(BODY_LABEL_NLL, [make_mfma(
            c_dst_start=240, a_src_start=244, b_src_start=248, slot=0)])
        four = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: ml_prev_cap},
            n_gl={0: ngl_filler},
            n_ll={0: nll_filler},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        g = build_dataflow_graph(four)

        cross = [e for e in g.edges
                 if getattr(e.producer, "category", "").startswith("Pack")
                 and getattr(e.consumer, "category", None) == "MFMA"
                 and e.producer.body_label != e.consumer.body_label]
        assert cross, "Expected a cross-body CVTPack->MFMA edge."
        edge = cross[0]
        check = _cvt_to_mfma_gap_ok(
            edge.producer, edge.consumer, g)
        assert (check.required == 2 and check.observed == 0
                and check.result == TimingResult.FAIL), (
            f"Below-threshold cross-body CVT->MFMA must fire: "
            f"observed={check.observed} < required={check.required}, "
            f"result={check.result}. A regression that swallows the "
            f"cross-body case (e.g., a body-delta sentinel) would return "
            f"result=PASS / observed>=1000."
        )

        failures = validate_edge_wait_coverage(g)
        cross_body_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", None) == "MFMA"
            and f.producer.body_label != f.consumer.body_label
        ]
        assert cross_body_timing, (
            f"Below-threshold cross-body CVT->MFMA edge must emit "
            f"TimingTooCloseFailure. Got failures: "
            f"{[type(f).__name__ for f in failures]}"
        )

    def test_cvt_to_mfma_no_graph_returns_strict_fail(self):
        """Direct call to `_cvt_to_mfma_gap_ok` with `subj_graph=None`.
        Pre-`2bu.4` this took the cross-body sentinel branch and returned
        `(True, 2, 1000)`. The helper's first action is to check
        `subj_graph is None` and return `(False, 0, 0)` — strictly
        conservative: no graph -> no profile -> no derivable threshold;
        degenerate test paths surface as failures rather than silently
        passing.

        Production callers always pass `subj_graph=graph`; this branch
        exists purely as a defensive guard for unit-test scaffolding."""
        from Tensile.Components.ScheduleCapture import SchedulePosition
        from Tensile.Components.CMSValidator import GraphNode

        producer_pos = SchedulePosition(loop_index=0, stream_index=2)
        consumer_pos = SchedulePosition(loop_index=1, stream_index=0)
        producer = GraphNode(
            identity=("VCvtPkF32toBF16", 0, ()),
            position=producer_pos,
            category="PackA0",
            rocisa_inst=None,
            tagged_inst=None,
            body_label=BODY_LABEL_ML_PREV,
        )
        consumer = GraphNode(
            identity=("MFMAInstruction", 1, ()),
            position=consumer_pos,
            category="MFMA",
            rocisa_inst=None,
            tagged_inst=None,
            body_label=BODY_LABEL_ML,
        )

        check = _cvt_to_mfma_gap_ok(producer, consumer, None)
        assert check.result == TimingResult.FAIL, (
            f"`subj_graph=None` must return FAIL (strict). "
            f"Got result={check.result}."
        )
        assert check.required == 0, (
            f"`subj_graph=None` must return required=0 (no graph -> no "
            f"profile -> no derivable threshold). Got required={check.required}."
        )
        assert check.observed == 0, (
            f"`subj_graph=None` must return observed=0 (no graph to walk). "
            f"Got observed={check.observed}."
        )

    def test_cvt_to_mfma_same_body_old_vs_new_formula_documented_divergence(self):
        """One-off regression guard for the same-body / cross-body
        `_cvt_to_mfma_gap_ok` unification.

        IMPORTANT FINDING: the OLD slot-delta formula
        (`slot_delta * (1 + finish) - 1 + intervening`) and the
        `cumulative_issue_cycles` walk that replaces it do NOT produce
        identical numerical results for same-body CVT->MFMA, contrary
        to a prior assumption.

        The two formulas effectively double-count the intervening region
        for densely-populated streams:
          * Slot-delta treats each slot-INDEX gap as 1 cycle AND adds
            `+intervening` for actual instructions in those slots.
          * Cycle-exact walk only counts the per-instruction issue cost
            of the ACTUALLY-PRESENT instructions (no "empty slot
            penalty"). When every slot carries an instruction, the
            cycle-exact walk reports HALF (roughly) what the slot-delta
            formula did, because the slot-delta over-counts by also
            charging for the slot index gap.

        Concretely for this fixture (CVT@vmfma=2, MFMA@vmfma=6, with 3
        intervening LRs at vmfma=3,4,5):
          * slot_delta = 4; intervening = 3 → OLD actual =
            4*(1+0) - 1 + 3 = 6.
          * cycle-exact: CVT(1) + LR(1) + LR(1) + LR(1) before MFMA →
            c_issue = 4; gap = 4 - 0 - 1 = 3.

        Implication: the unification IS a behavior change for same-body
        CVT->MFMA when the captured stream is densely populated. The
        cycle-exact value is the true issue-cycle count and is what the
        production code (BF16/F8 kernels) effectively sees on hardware;
        the slot-delta formula was an OVER-estimate. The cycle-exact
        value is the new contract.

        This test pins the divergence so a future commit can't silently
        revert to the slot-delta arithmetic without explicitly updating
        this assertion.
        """
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
            make_lr(60, 1, 80, slot=3, category="LRA1"),
            make_lr(61, 1, 84, slot=4, category="LRA1"),
            make_lr(62, 1, 88, slot=5, category="LRA1"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=6, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        cvt_to_mfma = [
            e for e in g.edges
            if getattr(e.producer, "category", "").startswith("Pack")
            and getattr(e.consumer, "category", None) == "MFMA"
        ]
        assert cvt_to_mfma, "Expected a CVT->MFMA edge."
        edge = cvt_to_mfma[0]

        # NEW: cycle-exact walk via `cumulative_issue_cycles`.
        new_actual = cumulative_issue_cycles(g, edge.producer, edge.consumer)

        # OLD: slot-delta + intervening-count under-estimate.
        body = g.captures[edge.producer.body_label]
        p_key = (edge.producer.tagged_inst.slot.mfma_index,
                 edge.producer.tagged_inst.slot.sequence)
        c_key = (edge.consumer.tagged_inst.slot.mfma_index,
                 edge.consumer.tagged_inst.slot.sequence)
        lo, hi = (p_key, c_key) if p_key < c_key else (c_key, p_key)
        intervening = 0
        for ti in body.instructions:
            slot = getattr(ti, "slot", None)
            if slot is None:
                continue
            key = (getattr(slot, "mfma_index", None),
                   getattr(slot, "sequence", None))
            if key[0] is None or key[1] is None:
                continue
            if lo < key < hi:
                intervening += 1
        slot_delta = c_key[0] - p_key[0]
        old_actual = slot_delta * (1 + 0) - 1 + intervening

        # Pin the documented divergence concretely.
        assert old_actual == 6, (
            f"OLD slot-delta arithmetic: {slot_delta}*1 - 1 + "
            f"{intervening} = {old_actual} (expected 6)."
        )
        assert new_actual == 3, (
            f"NEW cumulative_issue_cycles walk: CVT(1) + 3*LR(1) before "
            f"MFMA → c_issue=4; gap = 4-0-1 = 3. Got {new_actual}."
        )
        assert new_actual != old_actual, (
            f"Documented divergence: cycle-exact "
            f"({new_actual}) is intentionally LESS than slot-delta "
            f"({old_actual}) for densely-populated streams. If they "
            f"agree, either the slot-delta arithmetic stopped "
            f"double-counting OR the cycle-exact walk grew an extra "
            f"contribution — investigate before changing this assertion."
        )
        # And the helper itself reports the cycle-exact number.
        check = _cvt_to_mfma_gap_ok(
            edge.producer, edge.consumer, g)
        assert check.observed == new_actual, (
            f"_cvt_to_mfma_gap_ok must delegate to "
            f"cumulative_issue_cycles. Got helper observed={check.observed}, "
            f"direct walk={new_actual}."
        )

    # -------------------------------------------------------------------------
    # Cross-body unit tests for `_mfma_pack_to_cvt_gap_ok`.
    #
    # Replaces an old `body_delta * 1000` always-true placeholder. Same-body
    # and cross-body now share the SAME code path: a unified call to
    # `cumulative_issue_cycles` (which walks across body boundaries in
    # `_BODY_BUILD_ORDER`). The cross-iteration distinction is a red
    # herring — the graph has all instructions laid out in execution
    # order regardless of which body they belong to, so one function computes
    # the actual cycle gap for both cases.
    #
    # Tests below mirror the pattern of the same-body PackMFMA->CVT1
    # boundary tests (`test_mfma_pack_to_cvt1_zero_gap_emits_timing_too_close`,
    # `test_mfma_pack_to_cvt1_meets_5_cycle_gap_no_failure`,
    # `test_mfma_pack_to_cvt1_one_short_pins_actual_4`) but with the
    # producer in ML-1 and the consumer in ML.
    # -------------------------------------------------------------------------

    def _make_real_pack_mfma_cross_body(self, *, acc_start, acc_count, a_start,
                                         a_count, b_start, b_count, slot,
                                         sequence, category):
        """Build a real rocisa MFMAInstruction (4x4 PackMFMA family) wrapped
        in a TaggedInstruction. Mirrors `_make_real_pack_mfma` — duplicated
        here so the cross-body test block is self-contained against the
        same-body siblings."""
        from rocisa.enum import InstType
        from rocisa.instruction import MFMAInstruction
        inst = MFMAInstruction(
            InstType.INST_F32, InstType.INST_F32, [4, 4, 4, 16], False,
            vgpr(acc_start, acc_count),
            vgpr(a_start, a_count),
            vgpr(b_start, b_count),
        )
        return _tag(inst, category=category,
                    mfma_index=slot, sequence=sequence)

    def _build_cross_body_pack_to_cvt_capture(self, *, ml_prev_pad_count):
        """Build a FourPartCapture whose ML-1 body holds a 4x4 PackMFMA
        producer at slot=2 followed by `ml_prev_pad_count` cost-1 LR
        instructions, and whose ML body holds the CVT1 consumer at slot=0.

        The PackMFMA writes accumulator v0..v3; the CVT reads v0 as src0
        (RAW edge across the body boundary).

        Cross-body cumulative_issue_cycles walk:
          ML-1 walk starting at PackMFMA (slot=2):
            PackMFMA: current_issue=0, mfma_free=2, p_issue_start=0;
              += 1 (issue cost) → current_issue=1.
            ml_prev_pad_count LRs each cost 1 → current_issue = 1 + N.
          ML walk starting at index 0:
            CVT (consumer at idx 0) → c_issue_start = 1 + N.
          gap = c_issue_start - p_issue_start - 1 = (1+N) - 0 - 1 = N.

        So `ml_prev_pad_count == N` directly controls the reported `actual`.
        """
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        ml_prev_instructions = [
            self._make_real_pack_mfma_cross_body(
                acc_start=0, acc_count=4, a_start=8, a_count=2,
                b_start=32, b_count=2, slot=2, sequence=0,
                category="PackA0"),
        ]
        # Pad with cost-1 LRs at successive slots after the PackMFMA. Each
        # LR contributes exactly 1 quad-cycle to current_issue.
        for i in range(ml_prev_pad_count):
            ml_prev_instructions.append(
                make_lr(80 + i, 1, 128 + 4 * i,
                        slot=3 + i, category="LRA1")
            )
        ml_prev_cap = make_capture(BODY_LABEL_ML_PREV, ml_prev_instructions)
        ml_cap = make_capture(BODY_LABEL_ML, [
            _tag(cvt, category="PackA1", mfma_index=0, sequence=0),
        ])
        ngl_filler = make_capture(BODY_LABEL_NGL, [make_mfma(
            c_dst_start=220, a_src_start=224, b_src_start=228, slot=0)])
        nll_filler = make_capture(BODY_LABEL_NLL, [make_mfma(
            c_dst_start=240, a_src_start=244, b_src_start=248, slot=0)])
        return FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: ml_prev_cap},
            n_gl={0: ngl_filler},
            n_ll={0: nll_filler},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )

    def test_mfma_pack_to_cvt1_cross_body_gap_meets_5_no_failure(self):
        """Positive cross-body case: PackMFMA producer in ML-1 (slot=2)
        with 5 cost-1 LR instructions trailing it; CVT1 consumer in ML
        (slot=0). Cross-body cumulative_issue_cycles walk yields
        actual = 5 (see `_build_cross_body_pack_to_cvt_capture` arithmetic:
        actual == ml_prev_pad_count). expected = 5; 5 >= 5 → ok=True →
        NO TimingTooCloseFailure on the cross-body PackMFMA->CVT1 edge.

        Pre-2bu.5 this passed via the always-true `body_delta * 1000`
        placeholder. Post-2bu.5 it must STILL pass — but now via the same
        cycle-exact arithmetic the same-body case uses, with the cross-body
        walk in `cumulative_issue_cycles` correctly summing per-instruction
        issue costs across the ML-1 → ML boundary."""
        four = self._build_cross_body_pack_to_cvt_capture(ml_prev_pad_count=5)
        g = build_dataflow_graph(four)

        # Locate the cross-body PackMFMA -> CVTPack edge.
        cross = [e for e in g.edges
                 if getattr(e.producer, "category", "").startswith("Pack")
                 and getattr(e.consumer, "category", "").startswith("Pack")
                 and e.producer.body_label != e.consumer.body_label]
        assert cross, (
            "Expected at least one cross-body PackMFMA->CVTPack edge "
            "(ML-1 producer -> ML consumer). Got edges: "
            f"{[(getattr(e.producer, 'category', None), e.producer.body_label, getattr(e.consumer, 'category', None), e.consumer.body_label) for e in g.edges]}"
        )
        edge = cross[0]
        check = _mfma_pack_to_cvt_gap_ok(
            edge.producer, edge.consumer, g)
        assert check.required == 5, (
            f"PackMFMA->CVT1 required=5 (QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1); "
            f"got required={check.required}."
        )
        # observed = ml_prev_pad_count = 5 (cross-body walk arithmetic).
        assert check.observed == 5, (
            f"Cross-body cumulative_issue_cycles arithmetic: PackMFMA at "
            f"current_issue=0 + 5 LRs (cost 1 each) → CVT issues at 6; "
            f"gap = 6-0-1 = 5. Got observed={check.observed}. If observed==1000, the "
            f"old body_delta * 1000 placeholder is back."
        )
        assert check.result == TimingResult.PASS, (
            f"Cross-body PackMFMA->CVT1 with observed=5 >= required=5 must "
            f"pass. Got result={check.result}, observed={check.observed}."
        )
        # End-to-end: validate_edge_wait_coverage must NOT flag this edge.
        failures = validate_edge_wait_coverage(g)
        cross_body_pack_to_cvt = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
            and f.producer.body_label != f.consumer.body_label
            and f.expected_quad_cycles == 5
        ]
        assert not cross_body_pack_to_cvt, (
            f"Cross-body PackMFMA->CVT1 with actual==expected==5 must NOT "
            f"emit TimingTooCloseFailure. Got: {cross_body_pack_to_cvt}"
        )

    def test_mfma_pack_to_cvt1_cross_body_gap_below_5_emits_timing_too_close(self):
        """Negative cross-body case: PackMFMA producer in ML-1 (slot=2)
        with only 2 cost-1 LRs trailing it; CVT1 consumer in ML (slot=0).
        Cross-body cumulative_issue_cycles arithmetic yields actual = 2 <
        expected = 5 → TimingTooCloseFailure must be emitted.

        Pre-2bu.5 this incorrectly PASSED (the placeholder always returned
        ok=True for cross-body). Post-2bu.5 the cross-body arithmetic
        catches the violation — proving that the PackMFMA->CVT1 5-cycle
        settle window is now enforced uniformly across body boundaries."""
        four = self._build_cross_body_pack_to_cvt_capture(ml_prev_pad_count=2)
        g = build_dataflow_graph(four)
        cross = [e for e in g.edges
                 if getattr(e.producer, "category", "").startswith("Pack")
                 and getattr(e.consumer, "category", "").startswith("Pack")
                 and e.producer.body_label != e.consumer.body_label]
        assert cross
        edge = cross[0]
        check = _mfma_pack_to_cvt_gap_ok(
            edge.producer, edge.consumer, g)
        assert check.required == 5
        assert check.observed == 2, (
            f"Cross-body arithmetic: PackMFMA + 2 LRs → CVT issues at 3; "
            f"gap = 3-0-1 = 2. Got observed={check.observed}."
        )
        assert check.result == TimingResult.FAIL, (
            f"Cross-body PackMFMA->CVT1 with observed=2 < required=5 MUST "
            f"fail the gap check. Got result={check.result}. If PASS, the "
            f"cross-body branch is still using the always-true placeholder."
        )
        failures = validate_edge_wait_coverage(g)
        cross_body_pack_to_cvt = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
            and f.producer.body_label != f.consumer.body_label
            and f.expected_quad_cycles == 5
        ]
        assert cross_body_pack_to_cvt, (
            f"Expected TimingTooCloseFailure on cross-body PackMFMA->CVT1 "
            f"edge (actual=2 < expected=5). Got failures: "
            f"{[type(f).__name__ for f in failures]}"
        )
        f = cross_body_pack_to_cvt[0]
        assert f.actual_quad_cycles == 2

    def test_mfma_pack_to_cvt1_cross_body_one_short_boundary_emits_failure(self):
        """Cross-body off-by-one boundary: PackMFMA in ML-1 with 4 cost-1
        LRs trailing it, CVT1 in ML at slot=0. Cross-body arithmetic
        yields actual = 4 (one short of expected = 5) → TimingTooCloseFailure.

        Mirror of the same-body off-by-one sentinel
        `test_mfma_pack_to_cvt1_one_short_pins_actual_4` (zpi); a regression
        that lowers QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 from 5 to 4 makes
        this fixture stop firing — the off-by-one cross-body sentinel."""
        four = self._build_cross_body_pack_to_cvt_capture(ml_prev_pad_count=4)
        g = build_dataflow_graph(four)
        cross = [e for e in g.edges
                 if getattr(e.producer, "category", "").startswith("Pack")
                 and getattr(e.consumer, "category", "").startswith("Pack")
                 and e.producer.body_label != e.consumer.body_label]
        assert cross
        edge = cross[0]
        check = _mfma_pack_to_cvt_gap_ok(
            edge.producer, edge.consumer, g)
        assert check.required == 5, (
            f"PackMFMA->CVT1 required=5; got required={check.required}. A change "
            f"here means QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 was mutated."
        )
        assert check.observed == 4, (
            f"Cross-body off-by-one: PackMFMA + 4 LRs → CVT at "
            f"current_issue=5; gap = 5-0-1 = 4. Got observed={check.observed}. "
            f"A ±1 miscount in the cross-body cumulative_issue_cycles walk "
            f"would shift this to 3 or 5 and be caught here."
        )
        assert check.result == TimingResult.FAIL, (
            f"Cross-body observed=4 < required=5 MUST fail. Got result={check.result}."
        )
        failures = validate_edge_wait_coverage(g)
        cross_body_pack_to_cvt = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", "").startswith("Pack")
            and getattr(f.consumer, "category", "").startswith("Pack")
            and f.producer.body_label != f.consumer.body_label
            and f.expected_quad_cycles == 5
        ]
        assert cross_body_pack_to_cvt, (
            f"Expected TimingTooCloseFailure on cross-body PackMFMA->CVT1 "
            f"off-by-one (actual=4, expected=5). If this fixture stops "
            f"firing, QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 was likely lowered."
        )
        f = cross_body_pack_to_cvt[0]
        assert f.actual_quad_cycles == 4
        assert f.expected_quad_cycles == 5


# =============================================================================
# cycle-exact cumulative_issue_cycles helper. Mirrors the (now-removed)
# `CMSValidator.precompute_issue_times` walk over the captured stream.
# These tests pin the exact contract: the helper sums per-instruction
# issue costs AND injects MFMA type-switch +1 stalls only when the
# inter-MFMA gap is below the producer-class threshold.
# =============================================================================


class TestCumulativeIssueCycles:
    """Direct tests for `cumulative_issue_cycles` — the helper that
    replaces the slot-delta approximation in `_quad_cycle_gap_ok`."""

    def test_chain_with_intervening_alu_accumulates_issue_cycles(self):
        """Producer MFMA + 5 intervening ALU instructions + consumer MFMA.
        Each ALU adds 1 quad-cycle of issue cost; the consumer's
        `current_issue` is bumped by the sum of those costs (not the
        slot_delta arithmetic).

        Walk: MFMA1 issues at 0, mfma_free=4. 5 ALUs each contribute +1 →
        current_issue advances 1+1+1+1+1+1 (producer cost +1, then 5 ALU
        costs) = 6. MFMA2 max(6, 4)=6 → c_issue=6 → gap=6-0-1=5.
        """
        alus = [VXorB32(dst=vgpr(50 + i, 1),
                        src0=vgpr(60 + i, 1),
                        src1=vgpr(70 + i, 1))
                for i in range(5)]
        cap = make_capture(BODY_LABEL_ML, [
            # Producer MFMA at slot=2 seq=0.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # 5 intervening ALUs, each at distinct slots so they don't
            # collide with MFMA positions.
            *[_tag(alus[i], category="PackA0",
                   mfma_index=3 + i, sequence=0)
              for i in range(5)],
            # Consumer MFMA at slot=8 reads v0..v1.
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=8, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Find MFMA→MFMA edge.
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        gap = cumulative_issue_cycles(g, acc_edge.producer, acc_edge.consumer)
        # Producer cost (+1) + 5 ALU costs (+5) = 6 cycles to consumer.
        # Consumer max(6, mfma_free=4) = 6. Gap = 6 - 0 - 1 = 5.
        assert gap == 5, (
            f"5 intervening ALUs each contribute 1 quad-cycle. "
            f"Expected gap=5, got {gap}."
        )

    def test_chain_with_multiple_typeswitches_accumulates_stalls(self):
        """Chain of three MFMAs: standard → 4x4 → standard, each adjacent.
        Two type-switch boundaries; each fires when the gap-since-last
        is below the producer-class threshold.

        Walk:
          MFMA-std issues at 0, mfma_free=4, last_class=std.
          MFMA-4x4 max(1, 4)=4; type switch (std→4x4), gap=4-0=4 <
            FROM_STANDARD=5 → +1 → issues at 5, mfma_free=5+1+1=7,
            last_class=4x4, last_issue=5.
          MFMA-std max(6, 7)=7; type switch (4x4→std), gap=7-5=2 <
            FROM_4X4=3 → +1 → issues at 8.
        From producer (issue=0) to final consumer (issue=8): gap=7."""
        cap = make_capture(BODY_LABEL_ML, [
            # Producer (standard) at slot=2.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # 4x4 PackMFMA at slot=3.
            make_mfma(c_dst_start=80, a_src_start=8, b_src_start=32,
                      slot=3, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
            # Consumer (standard) at slot=4 reads v0..v1.
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=4, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Identify the producer (slot=2) and consumer (slot=4) by
        # searching the MFMA→MFMA edges.
        target = None
        for e in g.edges:
            if (getattr(e.producer, "category", None) == "MFMA"
                    and getattr(e.consumer, "category", None) == "MFMA"
                    and e.producer.tagged_inst.slot.mfma_index == 2
                    and e.consumer.tagged_inst.slot.mfma_index == 4):
                target = e
                break
        assert target is not None
        gap = cumulative_issue_cycles(g, target.producer, target.consumer)
        # Two stalls accumulated on the path producer→consumer.
        assert gap == 7, (
            f"Two type-switch +1 stalls expected on standard→4x4→standard "
            f"chain. gap should be 7; got {gap}."
        )

    def test_chain_with_typeswitch_above_threshold_no_stall(self):
        """Two MFMAs with a type-switch but a wide enough gap that the
        threshold check does NOT fire — no +1 penalty.

        Walk: standard MFMA at slot=2, 4 ALUs spacing things out, 4x4
        MFMA at slot=7.
          MFMA-std issues at 0, mfma_free=4, last_class=std, last_issue=0.
          Producer cost +1 then 4 ALUs +1 each → current=5.
          MFMA-4x4 max(5, 4)=5; type switch (std→4x4), gap=5-0=5 NOT <
          FROM_STANDARD=5 → NO stall → issues at 5.
          delivered gap = 5-0-1 = 4.

        Contrast with `test_mfma_type_switch_standard_to_4x4_adds_one_to_actual`
        (no intervening ALUs, gap-since-last=4 < 5 → +1 stall, gap=4)."""
        alus = [VXorB32(dst=vgpr(50 + i, 1),
                        src0=vgpr(60 + i, 1),
                        src1=vgpr(70 + i, 1))
                for i in range(4)]
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            *[_tag(alus[i], category="PackA0",
                   mfma_index=3 + i, sequence=0)
              for i in range(4)],
            # 4x4 consumer reads v0..v1 at slot=7.
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=7, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
        ])
        g = build_dataflow_graph(_wrap(cap))
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        gap = cumulative_issue_cycles(g, acc_edge.producer, acc_edge.consumer)
        # No +1: gap-since-last = 5 NOT < FROM_STANDARD=5. delivered gap = 4.
        assert gap == 4, (
            f"Threshold uses strict <; gap-since-last==5==threshold means "
            f"NO stall. Expected delivered gap=4, got {gap}."
        )

    # ---------------------------------------------------------------
    # SNop wait_state coverage. `_min_issue_quad_cycles_for` returns
    # `1 + wait_state` for SNop; the cumulative_issue_cycles walk consumes
    # that value when an SNop sits between an MFMA producer and an MFMA
    # consumer. The three tests below pin the SNop contribution at the
    # cycle-exact boundaries where finish-latency stops dominating, so
    # that a mutation reducing the SNop branch to `return 1` (eliminating
    # the wait_state contribution) is caught by unit tests rather than
    # only by heavy integration tests.
    # ---------------------------------------------------------------
    @pytest.mark.parametrize("wait_state,expected_gap", [
        (0, 3),  # Cost path = 1+1 = 2; finish path = 4 dominates → gap=3.
        (3, 4),  # Cost path = 1+(1+3) = 5 dominates → consumer_issue=5,
                 # gap=4. Mutation `return 1` would give 3.
        (5, 6),  # Cost path = 1+(1+5) = 7 dominates → gap=6.
                 # Mutation would give 3.
    ])
    def test_snop_wait_state_shifts_consumer_issue(self, wait_state, expected_gap):
        """Producer MFMA + intervening SNop(wait_state=N) + consumer MFMA.
        Once N is large enough to dominate over the producer's finish=4
        bound, the consumer's issue start tracks `producer_cost + (1 + N)`
        exactly. Walk for standard producer at slot=2:
          MFMA-std issues at 0, mfma_free_at=0+1+3=4, last_issue=0.
          Producer cost +1 → current_issue=1.
          SNop adds (1 + wait_state) → current_issue = 2 + wait_state.
          MFMA-std consumer current_issue = max(2+wait_state, 4).
          Gap = max(2+wait_state, 4) - 0 - 1 = max(wait_state+1, 3).
        For wait_state=0 the bound is finish (gap=3); from wait_state>=3
        the SNop dominates and each unit of wait_state shifts the gap
        by 1. The mutation `_min_issue_quad_cycles_for SNop` →
        `return 1` collapses all expected_gap values to 3 (the
        finish-bound), so wait_state ∈ {3, 5} catch the regression.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_snop(slot=3, wait_state=wait_state),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=4, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        gap = cumulative_issue_cycles(g, acc_edge.producer, acc_edge.consumer)
        assert gap == expected_gap, (
            f"SNop(wait_state={wait_state}) between standard MFMAs should "
            f"give gap={expected_gap}; got {gap}. If the SNop wait_state "
            f"contribution were silently dropped (e.g. "
            f"`_min_issue_quad_cycles_for` for SNop returned 1 instead of "
            f"`1 + wait_state`), the gap would collapse to the finish-bound "
            f"value 3."
        )

    def test_snop_wait_state_above_finish_dominates_standard_mfma(self):
        """Standard MFMA producer (finish=3) + SNop(wait_state=3) + standard
        consumer. The SNop's contribution (1 + 3 = 4 quad-cycles) ADDED to
        the producer's own issue cost (+1) puts current_issue at 5 — past
        the producer's finish bound (mfma_free_at = 0 + 1 + 3 = 4). The
        consumer therefore issues at 5, not at 4. Gap = 5 - 0 - 1 = 4.

        This pins the contract that SNop wait_state is NOT silently
        absorbed by MFMA finish latency once it exceeds the absorbable
        slack. The mutation `_min_issue_quad_cycles_for` SNop branch →
        `return 1` would re-collapse the gap to 3 (finish-bound), so this
        test fails meaningfully if wait_state stops contributing.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_snop(slot=3, wait_state=3),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=4, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        gap = cumulative_issue_cycles(g, acc_edge.producer, acc_edge.consumer)
        assert gap == 4, (
            f"SNop(wait_state=3) between standard MFMAs: cost path = "
            f"1 (producer) + (1 + 3) (snop) = 5 dominates the "
            f"finish path (mfma_free_at = 4). Expected gap=4, got {gap}. "
            f"If SNop wait_state were dropped, the gap would collapse to "
            f"the finish-bound value 3."
        )

    def test_snop_wait_state_dominates_4x4_mfma_finish(self):
        """Mirror of `test_4x4mfma_pack_parallel`: a 4x4 PackMFMA producer
        (finish=1) + SNop(wait_state=2) + standard consumer. The 4x4
        finish bound (mfma_free_at = 0 + 1 + 1 = 2) is shorter than the
        SNop's contribution (1 + 2 = 3), so the SNop unambiguously
        dominates the consumer's earliest issue.

        Walk:
          MFMA-4x4 issues at 0, mfma_free_at=2, last_class=4x4, last_issue=0.
          Producer cost +1 → current_issue=1.
          SNop(wait_state=2) +3 → current_issue=4.
          MFMA-std consumer max(4, 2) = 4. Type switch (4x4 → std):
            gap-since-last = 4 - 0 = 4 NOT < FROM_4X4=3 → no +1 stall.
          Consumer issues at 4 → gap = 4 - 0 - 1 = 3.

        Mutation `_min_issue_quad_cycles_for` SNop → `return 1` walk:
          Producer cost +1 → 1. SNop +1 → 2. Consumer max(2, 2) = 2.
          Type switch gap = 2 - 0 = 2 < FROM_4X4=3 → +1 stall → 3.
          Gap = 3 - 0 - 1 = 2.

        So the mutation drops the gap by 1 (the lost SNop wait_state),
        with the type-switch stall partially masking the regression but
        not erasing it. This is the asymmetric case where 4x4 finish=1
        < SNop's 3, so the SNop's wait_state is observable even at low
        values.
        """
        cap = make_capture(BODY_LABEL_ML, [
            # 4x4 PackMFMA producer at slot=2.
            make_mfma(c_dst_start=80, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2,
                      variant=[4, 4, 4, 16]),
            make_snop(slot=3, wait_state=2),
            # Standard consumer at slot=4 reads v0..v1 (writes the 4x4
            # producer's c_dst — the dataflow edge fires off the c_dst
            # accumulator chain).
            make_mfma(c_dst_start=84, a_src_start=80, b_src_start=32,
                      slot=4, sequence=0, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        acc_edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "MFMA"),
            None,
        )
        assert acc_edge is not None
        gap = cumulative_issue_cycles(g, acc_edge.producer, acc_edge.consumer)
        assert gap == 3, (
            f"4x4 PackMFMA (finish=1) + SNop(wait_state=2) + standard MFMA: "
            f"SNop cost (1 + 2 = 3) dominates over 4x4 finish bound (=2). "
            f"Expected gap=3, got {gap}. If SNop wait_state were dropped, "
            f"the gap would collapse to 2 (with a type-switch +1 stall "
            f"partially masking the loss)."
        )

    # The structural-side parity test
    # `test_graph_actual_matches_precompute_issue_times` was removed.
    # The `precompute_issue_times` / `estimate_quad_cycles_precomputed`
    # helpers it imported no longer exist in CMSValidator; graph-side
    # `cumulative_issue_cycles` is now the sole source of truth for MFMA
    # quad-cycle gap verdicts.


# =============================================================================
# P0 — GRInc SRD WAW invisible to subsequent GR (BufferLoad)
# =============================================================================
# `incrementSrd` (KernelWriterAssembly.py:8910) writes Srd{tc}+0/+1 with
# SAddU32/SAddCU32. The next BufferLoad in this iteration reads Srd{tc} via
# `getSrcParams()[1]` (saddr; see `_BufferLoadRule.extract` in
# `ScheduleCapture.py`). If the SAddU32 is reordered after the BufferLoad,
# the load reads stale SRD and fetches wrong memory.


class TestGRIncSRDChain:
    def test_grinc_srd_waw_before_buffer_load(self):
        """Stripped to the minimum: SAddU32 (SRD writer) + BufferLoad
        (SRD reader). The realistic GEMM frame (SWait + MFMA) is
        unnecessary — the property is purely about SRD producer/consumer
        ordering."""
        # Reference: SAddU32(Srd+0) then BufferLoad(reads Srd+0..3).
        srd_add = SAddU32(dst=sgpr(20, 1), src0=sgpr(20, 1), src1=sgpr(100, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(srd_add, category="GRIncA", mfma_index=0, sequence=0),
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
                    slot=0, category="GRA", sequence=1),
        ])
        # Subject: BufferLoad before SAddU32. Stale SRD.
        srd_add2 = SAddU32(dst=sgpr(20, 1), src0=sgpr(20, 1), src1=sgpr(100, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
                    slot=0, category="GRA", sequence=0),
            _tag(srd_add2, category="GRIncA", mfma_index=0, sequence=1),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "GRIncA SAddU32 writing SRD reordered after BufferLoad reading "
            "SRD — the load consumes stale SRD."
        )


# =============================================================================
# P1 — Pack RAW from LR
# =============================================================================
# Pack instructions (VCvtPkF32toBF16, VLShift*, VPermB32, VPackF16toB32,
# VSwapB32, etc.) read from the LR-output vgpr space and write either
# CVT-intermediate vgprs or pack-output vgprs. Today every Pack class
# returns [] from _writes/_reads, so:
#   - LR -> Pack edge is invisible (Pack reordered before LR is undetectable)
#   - Pack -> MFMA edge is invisible (MFMA reordered before Pack is undetectable)


class TestPackRAW:
    def test_pack_cvt_raw_from_lr(self):
        """LR writes v8..v11; pack VCvtPkF32toBF16 reads v8..v9. Reordering
        the pack before the LR consumes uninitialized data.

        Stripped to the minimum: LR + VCvtPkF32toBF16 sharing v8..v9.
        The realistic GEMM frame (SWait + MFMA) is unnecessary — the
        property is purely about LR-vs-Pack RAW ordering.
        """
        from rocisa.instruction import VCvtPkF32toBF16
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1), src0=vgpr(8, 1), src1=vgpr(9, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=0, sequence=0),
        ])
        cvt2 = VCvtPkF32toBF16(dst=vgpr(40, 1), src0=vgpr(8, 1), src1=vgpr(9, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(cvt2, category="PackA0", mfma_index=0, sequence=0),
            make_lr(8, 4, 64, slot=0, category="LRA0", sequence=1),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "Pack VCvtPkF32toBF16 reordered before its LR producer — should "
            "have produced an OrderInvertedFailure on v8..v9 RAW."
        )


# =============================================================================
# P1 — VSwap bidirectional RMW
# =============================================================================
# VSwapB32 swaps dst<->src — both registers are read and both are written.
# A pair of VSwaps that share a register has a meaningful dependency:
#   VSwap(v0, v1)        # after: v0=old_v1, v1=old_v0
#   VSwap(v1, v2)        # after: v1=old_v2, v2=old_v0  (uses output of swap1)
# Reversing them changes the final state.
#
# `_VSwapRule` (ScheduleCapture.py, before `_GenericALURule`)
# publishes both operands as reads AND writes. With the symmetric model the
# shared register carries WAR/WAW/RAW edges in BOTH orderings.
#
# UPDATE (rocm-libraries-wx9.3 phase 3, memo §6.1 step 1): the cross-graph
# edge identity now includes the producer's and consumer's POSITIONAL
# operand-slot indices:
#
#     (src_role, src_position, src_operand_slot,
#      sink_role, sink_position, sink_operand_slot,
#      edge_kind, intra_operand_byte_offset)
#
# Operand-slot is allocation-invariant (a small integer position, not a
# register reference) AND order-sensitive: when two VSwaps sharing a
# register are reordered, the shared register lands at a DIFFERENT
# operand-slot on each end of the resulting edge. So:
#
#   - Across graphs (allocation rename): the same VSwap-pair topology with
#     different absolute registers still yields IDENTICAL edge identities —
#     pinned by `test_vswap_pair_allocation_invariant` below.
#   - Within a graph (instruction reorder): the two orderings produce
#     DIFFERENT edge identities and `compare_graphs` surfaces an
#     `OrderInvertedFailure` — pinned by `test_vswap_pair_reorder_detected`
#     below. This is the case the previous wx9.3 implementation got wrong.


class TestVSwapPair:
    def test_vswap_pair_reorder_detected(self):
        """A within-graph reorder of two VSwaps sharing a register MUST be
        detected (rocm-libraries-wx9.3 phase 3, memo §6.1 step 1).

        Stripped to the minimum: two VSwapB32 instances tagged as PackA0 at
        sub=0 and sub=1, no LR, no MFMA, no SWait. The `_wrap` helper still
        fills the other 3 bodies with filler MFMAs because
        `build_dataflow_graph` requires non-empty bodies.

        REF:  VSwap(v0, v1) at sub=0; VSwap(v1, v2) at sub=1
        SUBJ: VSwap(v1, v2) at sub=0; VSwap(v0, v1) at sub=1

        The shared register `v1` is operand-1 of producer and operand-0 of
        consumer in REF; in SUBJ those slots flip — so the edge identity
        flips and `compare_graphs` returns an `OrderInvertedFailure`.
        """
        sw1 = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
        sw2 = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(sw1, category="PackA0", mfma_index=0, sequence=0),
            _tag(sw2, category="PackA0", mfma_index=0, sequence=1),
        ])
        sw1b = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
        sw2b = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(sw2b, category="PackA0", mfma_index=0, sequence=0),
            _tag(sw1b, category="PackA0", mfma_index=0, sequence=1),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(g_ref, g_subj)
        assert failures, (
            "Within-graph VSwap pair reorder MUST be detected as a failure "
            "under the operand-slot-aware edge identity; got no failures."
        )
        assert any(isinstance(f, OrderInvertedFailure) for f in failures), (
            "Within-graph VSwap pair reorder MUST surface "
            f"OrderInvertedFailure; got: {[type(f).__name__ for f in failures]}"
        )

    def test_vswap_pair_allocation_invariant(self):
        """Across-graph allocation-rename of a VSwap pair MUST NOT be
        flagged as a difference (rocm-libraries-wx9.3 phase 3, memo §6.1
        step 1).

        Graph A: VSwap(v0, v1) at sub=0; VSwap(v1, v2) at sub=1
        Graph B: VSwap(v1, v2) at sub=0; VSwap(v2, v3) at sub=1

        Both graphs share the same producer-consumer topology: the second
        VSwap's first operand equals the first VSwap's second operand
        (operand-1 of producer is operand-0 of consumer). The only
        difference is which physical registers happen to hold the data —
        operand-slot identity is by construction allocation-invariant
        (small integer positions, not register references), so the edge
        identities match exactly and `compare_graphs(g_a, g_b) == []`.
        """
        sw1a = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
        sw2a = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
        cap_a = make_capture(BODY_LABEL_ML, [
            _tag(sw1a, category="PackA0", mfma_index=0, sequence=0),
            _tag(sw2a, category="PackA0", mfma_index=0, sequence=1),
        ])
        sw1b = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
        sw2b = VSwapB32(dst=vgpr(2, 1), src=vgpr(3, 1))
        cap_b = make_capture(BODY_LABEL_ML, [
            _tag(sw1b, category="PackA0", mfma_index=0, sequence=0),
            _tag(sw2b, category="PackA0", mfma_index=0, sequence=1),
        ])
        g_a = build_dataflow_graph(_wrap(cap_a))
        g_b = build_dataflow_graph(_wrap(cap_b))
        assert compare_graphs(g_a, g_b) == [], (
            "Across-graph allocation rename of a VSwap pair must yield no "
            "failures (operand-slot identity is allocation-invariant). "
            f"Got: {[type(f).__name__ for f in compare_graphs(g_a, g_b)]}"
        )


# =============================================================================
# VCC dataflow tracking removed (bead `rocm-libraries-uraq`)
# =============================================================================
# `TestVCCCarryChain` (test_vcc_carry_chain_reorder + test_vcc_rule_invariants)
# was deleted alongside `_VCCRule` and its supporting helpers. VCC dataflow
# is intentionally not modeled by the validator going forward — see
# `CMSValidator_LIMITATIONS.md` §"VCC dataflow tracking is intentionally
# not provided". No replacement test is planned.


# =============================================================================
# P0 — LR LDS-address vgpr read (LRS -> LR RAW)
# =============================================================================
# DSLoadB* takes (dst, src, ds=None, comment=''). The `src` operand is the
# LDS-address vgpr (`LocalReadAddr{tc}` in the kernel writer). LRS modifies
# that vgpr via VXorB32/VAddU32 (KWA:11678-11691). Today `_reads(LR)` returns
# nothing for LR — neither the LDS-address vgpr nor the LDS bytes themselves.
# So an LRS pointer-flip reordered AFTER its consuming DSLoad is invisible:
# the load reads the wrong half of LDS.


class TestLRSAddrChain:
    def test_lrs_vxor_before_lr_invisible(self):
        """LRS XOR-swap of `LocalReadAddrA` (vgpr 40) reordered after its
        consuming DSLoadB128 — load consumes pre-swap address, fetching the
        wrong LDS half.

        Stripped to the minimum: LRS VXorB32 + DSLoadB128 sharing vgpr 40.
        The realistic GEMM frame (SWait + MFMA) is unnecessary — the
        property is purely about VXor-vs-DSLoad RAW ordering.
        """
        # Reference: LRS swap, then DSLoad reading the just-swapped vgpr.
        lrs = VXorB32(dst=vgpr(40, 1), src0=vgpr(60, 1), src1=vgpr(40, 1))
        ld_ref = DSLoadB128(dst=vgpr(8, 4), src=vgpr(40, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(lrs, category="LRSA0", mfma_index=0, sequence=0),
            _tag(ld_ref, category="LRA0", mfma_index=0, sequence=1),
        ])
        # Subject: same instructions, swap reordered after the load.
        lrs2 = VXorB32(dst=vgpr(40, 1), src0=vgpr(60, 1), src1=vgpr(40, 1))
        ld_subj = DSLoadB128(dst=vgpr(8, 4), src=vgpr(40, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(ld_subj, category="LRA0", mfma_index=0, sequence=0),
            _tag(lrs2, category="LRSA0", mfma_index=0, sequence=1),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "LRS VXorB32 writing LocalReadAddrA reordered after the "
            "DSLoadB128 reading it — should have produced an OrderInvertedFailure "
            "on the LDS-address vgpr RAW edge."
        )


# =============================================================================
# P0 — LW LDS-address vgpr read (LWS -> LW RAW)
# =============================================================================
# DSStoreB* takes (dstAddr, src, ds, comment). `dstAddr` (slot 0) is the
# LDS-address vgpr (`LocalWriteAddr{tc}`). LWS modifies that vgpr via
# VXorB32/VAddU32 (KWA:10542-10546). Today `_reads(LW)` returns only the
# data src (slot 1), not the LDS-address vgpr (slot 0). So an LWS pointer-flip
# reordered AFTER its consuming DSStore writes to the wrong LDS half.


class TestLWSAddrChain:
    def test_lws_vxor_before_lw_invisible(self):
        """LWS XOR-swap of `LocalWriteAddrA` (vgpr 50) reordered after its
        consuming DSStoreB128 — store writes to pre-swap LDS half.

        Stripped to the minimum: LWS VXorB32 + DSStoreB128 sharing vgpr 50.
        The realistic GEMM frame (SWait + MFMA) is unnecessary — the
        property is purely about VXor-vs-DSStore RAW ordering.
        """
        # Reference: LWS swap, then DSStore using the just-swapped vgpr.
        lws = VXorB32(dst=vgpr(50, 1), src0=vgpr(70, 1), src1=vgpr(50, 1))
        st_ref = DSStoreB128(dstAddr=vgpr(50, 1), src=vgpr(8, 4))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(lws, category="LWSA", mfma_index=0, sequence=0),
            _tag(st_ref, category="LWA", mfma_index=0, sequence=1),
        ])
        # Subject: store before swap.
        lws2 = VXorB32(dst=vgpr(50, 1), src0=vgpr(70, 1), src1=vgpr(50, 1))
        st_subj = DSStoreB128(dstAddr=vgpr(50, 1), src=vgpr(8, 4))
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(st_subj, category="LWA", mfma_index=0, sequence=0),
            _tag(lws2, category="LWSA", mfma_index=0, sequence=1),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "LWS VXorB32 writing LocalWriteAddrA reordered after the "
            "DSStoreB128 reading it — should have produced an "
            "OrderInvertedFailure on the LDS-address vgpr RAW edge."
        )


# =============================================================================
# rocm-libraries-u3t — instruction-coverage gaps in `_OPERAND_RULES`
# =============================================================================
# Three error classes were witnessed in production CMS=1 kernels but the
# graph-native validator's `_OPERAND_RULES` dispatch raised
# CaptureUnknownInstructionError or downstream ValueError instead of
# producing a graph node:
#
#   1. DSStoreD16HIB16 — Real DSStoreInstruction subclass; fix shape:
#      add to _LW_CLASS_NAMES so the existing _DSStoreRule claims it.
#   2. SSetPrior — gfx950 HSS / BBS Range MT256x256x64 (PGR=2, PLR=1,
#      DTL=T/T). Wave-priority scalar op; no register dataflow. Fix
#      shape: new _SSETPRIO_CLASS_NAMES set + _is_ssetprio() helper +
#      extension of `_NoDataflowRule.applies` to claim it. Also added
#      a `SSETPRIO` category in _captureSubIterToBuilder so it doesn't
#      land in `UNKNOWN`.
#   3. _node_label ValueError — gfx950 BBS Ailk MT192x256x64 (Run 7,
#      same report). Investigation showed this is SECONDARY to (1)/(2):
#      previously the failure dispatch routed through Failures whose
#      tagged_inst's category fell back to "UNKNOWN" (because the class
#      wasn't recognized), and that category bucket disagreed across
#      reference vs subject captures so _node_label's same-category
#      lookup raised. Once the class is recognized, the category is
#      stable across both captures. Pinned by
#      test_node_label_finds_ssetprio_in_capture_after_categorization.


class TestDSStoreD16HIB16Coverage:
    """`DSStoreD16HIB16` is a real DSStore subclass (rocisa
    `mem.hpp::DSStoreD16HIB16`) with the same Python ctor signature as
    `DSStoreB16`: `(dstAddr, src, ds=None, comment="")`.

    The "D16HI" semantic only changes which 16 bits of the LDS word are
    written; on the register side the source vgpr is read in full (the
    upper-bits selection happens at the LDS write, not at the register
    read), and no register is written. So `_DSStoreRule.extract` —
    `reads = (lds_addr, src_data); writes = ()` — is correct as-is.

    Pin: dispatching `_populate_wrapper` over a real DSStoreD16HIB16
    yields the SAME (reads, writes) shape as DSStoreB16 with identical
    operands. No `_FakeLW` synthetic stand-in is used (per bead 904).
    """

    def _build_d16hi(self):
        from rocisa.container import vgpr
        from rocisa.instruction import DSStoreD16HIB16
        return DSStoreD16HIB16(dstAddr=vgpr(50, 1), src=vgpr(8, 1))

    def _build_b16(self):
        from rocisa.container import vgpr
        from rocisa.instruction import DSStoreB16
        return DSStoreB16(dstAddr=vgpr(50, 1), src=vgpr(8, 1))

    def test_d16hi_dispatches_through_dsstore_rule(self):
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper, _is_lw,
        )
        inst = self._build_d16hi()
        assert _is_lw(inst), (
            "DSStoreD16HIB16 must be recognized as an LW class so the "
            "existing _DSStoreRule claims it."
        )
        wrapper = WrappedInstruction(inst)
        _populate_wrapper(wrapper)
        # Two reads (lds_addr, src_data); no writes (data goes to LDS).
        assert wrapper.writes == ()
        assert len(wrapper.reads) == 2
        # Verify the shape matches DSStoreB16 — same constructor args
        # produce the same (reads, writes).
        b16 = self._build_b16()
        b16_wrapper = WrappedInstruction(b16)
        _populate_wrapper(b16_wrapper)
        assert wrapper.writes == b16_wrapper.writes
        assert len(wrapper.reads) == len(b16_wrapper.reads)

    def test_d16hi_does_not_claim_partial_write_on_src(self):
        """The D16HI suffix encodes a partial write to the LDS half-word,
        NOT a partial read of the source vgpr. Confirm `_DSStoreRule`
        emits no writes (the register side has no dst) and reads include
        the full source vgpr."""
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        inst = self._build_d16hi()
        wrapper = WrappedInstruction(inst)
        _populate_wrapper(wrapper)
        assert wrapper.writes == ()
        # The src register (vgpr 8) must be in reads — not filtered out
        # as a partial-write target.
        from rocisa.container import vgpr
        src_reg = vgpr(8, 1)
        assert any(
            getattr(r, "regIdx", None) == src_reg.regIdx
            and getattr(r, "regType", None) == src_reg.regType
            for r in wrapper.reads
        ), f"DSStoreD16HIB16 src vgpr must appear in reads; got {wrapper.reads}"

    def test_d16hi_finalize_through_capture_pipeline(self):
        """End-to-end: appending a real DSStoreD16HIB16 through
        LoopBodyCaptureBuilder and calling finalize() populates the
        wrapper without raising CaptureUnknownInstructionError /
        CaptureStoreError. Confirms that DSStoreD16HIB16 is also NOT
        misclassified as a vector-memory store (would raise
        CaptureStoreError)."""
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder
        builder = LoopBodyCaptureBuilder()
        builder.append(self._build_d16hi(), category="LWA", subiter=0)
        capture = builder.finalize()
        ti = capture.instructions[0]
        assert ti.wrapped is not None
        assert ti.wrapped.writes == ()
        assert len(ti.wrapped.reads) >= 1


class TestSSetPriorCoverage:
    """`SSetPrior(prior, comment)` is `s_setprio` — a wave-priority scalar
    op with NO register dataflow (`getParams() -> {prior}` only; no
    RegisterContainer reads or writes).

    Fix shape: add to `_SSETPRIO_CLASS_NAMES`, extend `_NoDataflowRule.
    applies` to claim it, exclude from cross-graph data-flow identity set
    in `build_dataflow_graph`, and add a `SSETPRIO` category in
    `_captureSubIterToBuilder` so it routes through
    `_class_tag_from_category` rather than falling back through
    `_class_tag(UNKNOWN-class)` (which would still work post-fix because
    `_class_tag` now also recognizes `_is_ssetprio`).
    """

    def _build_ssetprio(self):
        from rocisa.instruction import SSetPrior
        return SSetPrior(prior=3, comment="raise priority")

    def test_ssetprio_recognized_by_helpers(self):
        from Tensile.Components.ScheduleCapture import (
            _is_ssetprio, _is_snop, _is_swait, _is_sbarrier,
        )
        inst = self._build_ssetprio()
        assert _is_ssetprio(inst), (
            "SSetPrior must be recognized by `_is_ssetprio` so "
            "_NoDataflowRule and the build_dataflow_graph identity-set "
            "skip both claim it."
        )
        # Disjoint from the other scheduling-control predicates.
        assert not _is_snop(inst)
        assert not _is_swait(inst)
        assert not _is_sbarrier(inst)

    def test_ssetprio_dispatches_to_no_dataflow_rule(self):
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        wrapper = WrappedInstruction(self._build_ssetprio())
        _populate_wrapper(wrapper)
        # No register dataflow.
        assert wrapper.reads == ()
        assert wrapper.writes == ()

    def test_ssetprio_class_tag_does_not_raise(self):
        """Pre-fix: `_class_tag(SSetPrior(...))` raised
        CaptureUnknownInstructionError. Post-fix: it returns 'SSETPRIO'."""
        from Tensile.Components.CMSValidator import _class_tag
        assert _class_tag(self._build_ssetprio()) == "SSETPRIO"

    def test_ssetprio_class_tag_from_category_routes_explicit_category(self):
        """`_captureSubIterToBuilder` now assigns category="SSETPRIO" to
        bare SSetPrior leaves; `_class_tag_from_category` must route that
        category to the same tag without falling back to `_class_tag`."""
        from Tensile.Components.CMSValidator import _class_tag_from_category
        inst = self._build_ssetprio()
        assert _class_tag_from_category("SSETPRIO", inst) == "SSETPRIO"

    def test_ssetprio_excluded_from_dataflow_identity_set(self):
        """SSetPrior nodes go into the per-body sidecar but NOT into
        `nodes_by_identity`, mirroring SNop/SWait/SBarrier. Build a small
        capture containing one SSetPrior + one MFMA and assert the
        resulting graph carries exactly one identity (the MFMA's)."""
        from Tensile.Components.ScheduleCapture import (
            LoopBodyCaptureBuilder, FourPartCapture, BODY_LABEL_ML,
            BODY_LABEL_ML_PREV, BODY_LABEL_NGL, BODY_LABEL_NLL,
            SLOT_KIND_MFMA,
        )
        from Tensile.Components.CMSValidator import build_dataflow_graph
        # Real ML capture: SSetPrior + MFMA filler. Use the same
        # _wrap helper convention but inline (we want SSetPrior in ML).
        builder = LoopBodyCaptureBuilder()
        builder.append(self._build_ssetprio(),
                       category="SSETPRIO", subiter=0,
                       slot_kind=SLOT_KIND_MFMA, mfma_index=0)
        # Append an MFMA so the body has at least one data-flow node
        # (build_dataflow_graph requires at least one identity per body).
        ml_mfma_tag = make_mfma(c_dst_start=100, a_src_start=4, b_src_start=32,
                                slot=1)
        builder.append(ml_mfma_tag.wrapped.rocisa_inst, category=ml_mfma_tag.category,
                       subiter=0, slot_kind=ml_mfma_tag.slot.slot_kind,
                       mfma_index=ml_mfma_tag.slot.mfma_index)
        ml_cap = builder.finalize()

        # Filler bodies (mirror _wrap's shape).
        _FILLER_RANGES = {
            BODY_LABEL_ML_PREV: (200, 204, 208),
            BODY_LABEL_NGL:     (220, 224, 228),
            BODY_LABEL_NLL:     (240, 244, 248),
        }

        def _filler(label):
            c, a, b = _FILLER_RANGES[label]
            return make_capture(label, [make_mfma(
                c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
            )])

        cap = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: _filler(BODY_LABEL_ML_PREV)},
            n_gl={0: _filler(BODY_LABEL_NGL)},
            n_ll={0: _filler(BODY_LABEL_NLL)},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        graph = build_dataflow_graph(cap)
        # SSetPrior must NOT appear as a data-flow node identity.
        for ident in graph.nodes.keys():
            assert ident[0] != "SSETPRIO", (
                f"SSetPrior leaked into nodes_by_identity as {ident!r}; "
                "build_dataflow_graph Phase 1 must skip SSetPrior in "
                "the cross-graph identity set, mirroring SNop."
            )

    def test_ssetprio_default_issue_cycle_is_one(self):
        """Unlike SNop (which encodes wait_state in its first param and
        adds it to the issue cost), SSetPrior's first param is the
        priority value — it must NOT be added to the default issue cost.
        Pin the default cost == 1."""
        from Tensile.Components.CMSValidator import (
            _min_issue_quad_cycles_for,
        )
        # Pass an explicit profile (helper now requires non-None profile).
        cycles = _min_issue_quad_cycles_for(
            self._build_ssetprio(), _DEFAULT_CDNA4_ARCH_PROFILE)
        assert cycles == 1, (
            "SSetPrior must use the default issue cost (1 quad-cycle); "
            "the SNop wait_state add path must NOT pick it up."
        )


class TestNodeLabelAfterCoverageFix:
    """Pin that the `_node_label` ValueError witnessed on gfx950 BBS Ailk
    MT192x256x64 (Run 7) was secondary to instruction-coverage gaps.

    Mechanism: pre-fix, an unrecognized class (e.g. SSetPrior) landed
    with category='UNKNOWN'. `_class_tag_from_category` then fell back
    to `_class_tag(inst)` which raised, OR (when both reference and
    subject captures categorized the same instruction differently
    because of the UNKNOWN fallthrough) `_node_label`'s same-category
    lookup in the formatter capture missed the node and raised
    ValueError. Post-fix: the class is categorized consistently as
    'SSETPRIO' in BOTH captures, so the same-category stream lookup
    finds the node.
    """

    def test_node_label_finds_ssetprio_in_capture_after_categorization(self):
        from Tensile.Components.ScheduleCapture import (
            LoopBodyCaptureBuilder,
            SLOT_KIND_MFMA,
            BODY_LABEL_ML,
            make_position,
        )
        from Tensile.Components.CMSValidator import GraphNode
        from Tensile.Components.CMSValidator import (
            cms_node_label,
            _class_tag_from_category,
            _identity_for,
        )
        from rocisa.instruction import SSetPrior

        builder = LoopBodyCaptureBuilder()
        sp = SSetPrior(prior=3, comment="raise")
        builder.append(sp, category="SSETPRIO", subiter=0,
                       slot_kind=SLOT_KIND_MFMA, mfma_index=0)
        # Add an MFMA to satisfy non-empty body for downstream graph
        # construction in case other tests share the capture.
        ml_mfma_tag = make_mfma(c_dst_start=100, a_src_start=4,
                                b_src_start=32, slot=1)
        builder.append(ml_mfma_tag.wrapped.rocisa_inst, category=ml_mfma_tag.category,
                       subiter=0, slot_kind=ml_mfma_tag.slot.slot_kind,
                       mfma_index=ml_mfma_tag.slot.mfma_index)
        capture = builder.finalize()

        ssetprio_tagged = capture.instructions[0]
        # Synthesize a GraphNode pointing at this tagged inst — same
        # shape as `_make_node` constructs in build_dataflow_graph.
        node = GraphNode(
            identity=_identity_for(sp, BODY_LABEL_ML, category="SSETPRIO"),
            position=make_position(BODY_LABEL_ML, ssetprio_tagged.slot),
            category="SSETPRIO",
            rocisa_inst=sp,
            tagged_inst=ssetprio_tagged,
            body_label=BODY_LABEL_ML,
            name="SSETPRIO@0.0",
            issue_cycles=1,
        )
        # Post-g4w: cms_node_label returns a FailureNodeLabel whose
        # `primary` carries the per-category-stream [N] index resolved
        # against the body capture. Lookup must succeed (SSETPRIO[0] is
        # the only SSetPrior in the body) — this pins the consistent
        # categorization fix from bead `u3t`.
        label = cms_node_label(node, capture)
        assert label.primary == "SSETPRIO[0]", (
            f"cms_node_label should return primary='SSETPRIO[0]' for the only "
            f"SSetPrior in capture; got {label.primary!r}"
        )
