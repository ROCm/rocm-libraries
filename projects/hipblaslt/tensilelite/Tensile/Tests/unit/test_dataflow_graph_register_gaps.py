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
"""Documented coverage gaps in ScheduleCapture._writes / _reads.

The dataflow graph (`build_dataflow_graph` + `compare_graphs`) only tracks
register dataflow for LR (DSLoad*), GR (BufferLoad*/GlobalLoad*), LW
(DSStore*), and MFMA. Every other rocisa instruction class returns `[]`
from both `_writes(inst)` and `_reads(inst)` (ScheduleCapture.py:1607-1629),
so reordering instructions of those classes within a CMS body is silently
invisible to `compare_graphs`.

Each test below asserts the DESIRED behavior: the graph detects a
reordered chain and surfaces it as failures from `compare_graphs`. Today
those assertions all fail (graph returns `[]`), so the tests are marked
`xfail(strict=True)`. When `_writes`/`_reads` are extended to cover the
relevant class, the corresponding test will XPASS and force removal of
the xfail marker.

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
  - `VSwap` bidirectional RMW reorder                -> test_vswap_pair_reorder_invisible
  - `VAddCO` / `VAddCCO` VCC chain                   -> test_vcc_carry_chain_reorder

Each test lives in its own class so a future fix can drop the xfail on
exactly one gap at a time.
"""

import pytest

from rocisa.container import RegisterContainer, RegName, sgpr, vgpr, mgpr
from rocisa.instruction import (
    SMovB32, SAddU32, SAddCU32, SSubU32, SSubBU32,
    SCmpEQU32, SCSelectB32, SCMovB32,
    VAddCOU32, VAddCCOU32, VSwapB32,
    DSLoadB128, DSStoreB128, VXorB32,
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
    build_dataflow_graph,
    compare_graphs,
    diagnose_missing_edge,
    validate_edge_wait_coverage,
    OrderInvertedFailure,
    TimingTooCloseFailure,
    _quad_cycle_gap_ok,
)

from dataflow_fixtures import (
    make_lr, make_gr, make_dtl_buffer_load, make_mfma, make_swait, make_capture,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap(ml_capture):
    """Wrap a single main-loop capture into a FourPartCapture, fillering the
    other 3 bodies with a no-op MFMA so build_dataflow_graph's
    non-empty-body precondition is satisfied."""
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
    )


def _tag(inst, *, category: str, mfma_index: int, sequence: int) -> TaggedInstruction:
    return TaggedInstruction(
        inst=inst,
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
# The remaining xfails in this file are gated on test-fixture changes,
# not production rule changes — see _FIXTURE_GAP_XFAIL_REASON below.

_FIXTURE_GAP_XFAIL_REASON = (
    "Detection requires a test-fixture extension that wx9.4.4 did NOT "
    "deliver. The production `_GenericALURule` correctly publishes the "
    "writer's reads/writes (e.g. m0, vgpr dsts), but: "
    "(a) DTLm0Tracking uses dataflow_fixtures.make_gr / _FakeGR which "
    "synthesizes a NON-DTL BufferLoad (has a vgpr dst) — so no instruction "
    "in the fixture reads m0, and the m0 setter has no consumer to form "
    "an edge with. Needs a real (lds=True) BufferLoadB128 in the fixture. "
    "(b) VSwapPair / VCCCarryChain reorder two ALU instructions whose "
    "writes are independent in the REFERENCE order; the broken dependency "
    "only forms an edge in the SUBJECT order, but compare_graphs only "
    "flags edges present in REFERENCE but missing from SUBJECT. Detecting "
    "these requires either symmetric VSwap semantics (both regs are "
    "read-AND-written) or a bidirectional edge comparison."
)


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
        # Reference: m0 set, then DTL BufferLoad consumes m0.
        m0_set = SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA", 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(m0_set, category="GRA", mfma_index=0, sequence=0),
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject: same instructions, m0-update issued AFTER the BufferLoad.
        # The load sees stale m0.
        m0_set2 = SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA", 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=0),
            _tag(m0_set2, category="GRA", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
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
        (DTL + IncLdsBufSwitch / DTL + ExpandPointerSwap path)."""
        m0_add = SAddU32(dst=mgpr(0),
                         src0=sgpr("LocalWriteAddrA", 1),
                         src1=sgpr("LDSBufferWriteInc", 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(m0_add, category="GRA", mfma_index=0, sequence=0),
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        m0_add2 = SAddU32(dst=mgpr(0),
                          src0=sgpr("LocalWriteAddrA", 1),
                          src1=sgpr("LDSBufferWriteInc", 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_dtl_buffer_load(vaddr_vgpr_start=40, srd_sgpr_start=20,
                                 slot=0, category="GRA", sequence=0),
            _tag(m0_add2, category="GRA", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "DTL m0 SAddU32 reordered after BufferLoad — should have "
            "produced a failure on the m0 RAW edge."
        )

    def test_m0_register_distinct_from_sgpr(self):
        """Confirm m0's regType differs from sgpr — the existing _reg_overlaps
        will (correctly) refuse to overlap m0 with an sgpr. Any future m0
        tracking must therefore add an explicit m0-aware overlap path."""
        m0_reg = mgpr(0)
        s0_reg = sgpr(0, 1)
        assert m0_reg.regType != s0_reg.regType, (
            "If m0 ever shares regType with sgpr, the false-positive "
            "overlap risk in _reg_overlaps must be re-audited."
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
        MFMA #2 reads its acc. Reversing them is a real WAW/RAW violation."""
        # Reference: MFMA1 (c_dst=0..3) then MFMA2 (a_src=0..1 reads same acc
        # by overlap). Use distinct slot numbers so they're separate nodes.
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=2),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=3, a_src_count=2),
        ])
        # Subject: same instructions, MFMAs swapped. Now the consumer MFMA
        # (reads v0..v1) issues before the producer (writes v0..v3).
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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

    def test_mfma_acc_chain_zero_gap_emits_timing_too_close(self):
        """Two MFMAs sharing accumulator v0..v3, both at the SAME mfma_index
        (sub_index breaks the tie). The per-byte resolver builds the
        producer->consumer edge; the gap check sees slot_delta == 0, expected
        == QUAD_CYCLES_STANDARD_MFMA_FINISH (3), actual == 0 — emits
        TimingTooCloseFailure."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            # Producer: writes v0..v3.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # Consumer at the same mfma_index, ordered by sub_index. Reads
            # v0..v1 — overlaps producer's accumulator.
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        failures = validate_edge_wait_coverage(g)
        # Find the MFMA->MFMA timing failure.
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and f.producer.category == "MFMA"
            and f.consumer.category == "MFMA"
        ]
        assert mfma_timing, (
            f"Expected TimingTooCloseFailure on MFMA->MFMA acc-chain at "
            f"slot_delta==0 but got: {[type(f).__name__ for f in failures]}"
        )
        f = mfma_timing[0]
        assert f.expected_quad_cycles == 3
        assert f.actual_quad_cycles == 0

    def test_mfma_acc_chain_consecutive_slots_no_failure(self):
        """Two MFMAs at consecutive vmfma_index values (slot_delta == 1)
        give exactly QUAD_CYCLES_STANDARD_MFMA_FINISH quad-cycles of gap —
        meets the threshold, so no TimingTooCloseFailure is emitted."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
    # TDD-first negative-test additions for bead d6e — these expose untested
    # paths in `_quad_cycle_gap_ok`. They are XFAIL today; xpass under strict
    # will signal that the relevant fix bead (cmw / s7k / dispatch gap) has
    # landed and the xfail marker should be dropped.
    # -------------------------------------------------------------------------

    def test_mfma_acc_chain_slot_delta_2_uses_num_mfma_per_subiter(self):
        """Producer at vmfma=0, consumer at vmfma=2; vary num_mfma_per_subiter
        between two builds of the same instruction sequence. Post-cmw
        (sound under-estimate, approach (b)) the returned `actual` includes
        a +1-per-subiter-boundary correction so it differs between
        num_mfma_per_subiter=1 (subiter_delta=2 → actual=7+2=9) and
        num_mfma_per_subiter=2 (subiter_delta=1 → actual=7+1=8) for the
        same slot pair. The graph-side check remains a lower bound on the
        real cross-subiter gap; the structural-side `precompute_issue_times`
        retains ground truth for cycle-accurate verdicts."""
        # Same instruction layout in both fixtures; only num_mfma_per_subiter
        # differs at the FourPartCapture level.
        def _build_cap():
            return make_capture(BODY_LABEL_ML, [
                make_lr(8, 4, 64, slot=0, category="LRA0"),
                make_swait(slot=1, dscnt=0),
                # Producer writes v0..v3.
                make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                          slot=2, sequence=0, a_src_count=2),
                # Consumer at slot=4 (slot_delta == 2) reads v0..v1 (RAW on
                # producer's accumulator).
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
            )

        g_nmps1 = build_dataflow_graph(_wrap_with_nmps(_build_cap(), 1))
        g_nmps2 = build_dataflow_graph(_wrap_with_nmps(_build_cap(), 2))

        # Find the MFMA->MFMA acc-chain edge in each graph.
        def _acc_edge(g):
            for e in g.edges:
                if (getattr(e.producer, "category", None) == "MFMA"
                        and getattr(e.consumer, "category", None) == "MFMA"):
                    return e
            return None

        e1 = _acc_edge(g_nmps1)
        e2 = _acc_edge(g_nmps2)
        assert e1 is not None and e2 is not None, (
            "Expected an MFMA->MFMA acc-chain edge in both graphs."
        )

        ok1, exp1, act1 = _quad_cycle_gap_ok(e1.producer, e1.consumer, 1)
        ok2, exp2, act2 = _quad_cycle_gap_ok(e2.producer, e2.consumer, 2)

        # Today: the formula ignores num_mfma_per_subiter, so act1 == act2.
        # Post-fix: the actual should reflect subiter-aware timing and differ.
        assert act1 != act2, (
            f"_quad_cycle_gap_ok ignored num_mfma_per_subiter: "
            f"actual={act1} for nmps=1 and actual={act2} for nmps=2 "
            f"(slot_delta=2). Expected the parameter to influence the "
            f"computed actual gap (cmw fix)."
        )

    def test_mfma_acc_chain_cross_body_actual_reflects_real_gap(self):
        """Producer in body=ML, consumer in body=NGL sharing accumulator
        v0..v3. The cross-body early-return at ScheduleCapture.py:2922-2925
        reports `(True, expected, expected)`; the diagnostic is misleading
        because `actual` should be much larger than `expected` (a full body
        boundary's worth of issue cycles)."""
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        ok, expected, actual = _quad_cycle_gap_ok(edge.producer, edge.consumer, 0)

        # s7k: today actual == expected (misleading). Post-fix, actual should
        # reflect the real cross-body gap and exceed expected.
        assert ok, "Cross-body edge should always pass the gap check."
        assert actual > expected, (
            f"Cross-body _quad_cycle_gap_ok reported actual={actual} == "
            f"expected={expected}; this is the misleading early-return s7k "
            f"flags. Expected actual to reflect the real cross-body gap "
            f"(many issue cycles)."
        )

    def test_mfma_acc_chain_zero_gap_via_diagnose_missing_edge(self):
        """Regression-pin test for current correct behavior: the
        diagnose_missing_edge MFMA branch correctly fires the quad-cycle gap
        check on a missing zero-gap acc-chain edge and emits
        TimingTooCloseFailure. This locks in today's behavior so a future
        refactor cannot silently regress the cross-graph MFMA-producer route.

        Richer negative tests that exercise this dispatch path with NON-zero
        gaps (and would actually expose mis-handling rather than just pinning
        zero-gap behavior) are tracked in follow-up bead `rocm-libraries-cpe`.

        Reference: P (MFMA writes v0..v3) at slot=2, then C (MFMA reads
        v0..v1) at slot=2 sub=1 — edge P->C present, slot_delta == 0.
        An ALU op writing v0..v1 sits AFTER C (does not shadow the
        latest-writer for C's read; P->C edge survives).
        Subject: same {P, C, ALU} identities, but the ALU is reordered to
        sit BETWEEN P and C — now ALU is the latest writer of v0..v1, so
        the edge becomes ALU->C and P->C is missing.
        compare_graphs sees the missing P->C edge in subj -> routes through
        diagnose_missing_edge whose MFMA branch fires the quad-cycle check
        and emits TimingTooCloseFailure (slot_delta == 0)."""
        # ALU dst spans v0..v1 (8 bytes) so it fully shadows the consumer's
        # 8-byte read of v0..v1; otherwise the per-byte resolver would split
        # the read across two writers (ALU on bytes 0-3, MFMA-P on bytes 4-7)
        # and the P->C edge would persist.
        ref_alu = VXorB32(dst=vgpr(0, 2), src0=vgpr(40, 1), src1=vgpr(41, 1))
        subj_alu = VXorB32(dst=vgpr(0, 2), src0=vgpr(40, 1), src1=vgpr(41, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        # Soft mode so an unexplained miss doesn't crash before assertion.
        failures = compare_graphs(g_ref, g_subj, raise_on_unexplained=False)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
            and getattr(f.consumer, "category", None) == "MFMA"
        ]
        assert mfma_timing, (
            f"Expected diagnose_missing_edge MFMA branch to emit "
            f"TimingTooCloseFailure for the missing zero-gap acc-chain "
            f"edge, but got: {[type(f).__name__ for f in failures]}"
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
        gaps on this MFMA->ALU dispatch path are tracked in follow-up bead
        `rocm-libraries-cpe`.

        MFMA producer writes v0..v3 at slot=2; ALU consumer (VXorB32) at the
        same slot reads v0 — overlaps the MFMA producer's accumulator."""
        alu_consumer = VXorB32(dst=vgpr(20, 1), src0=vgpr(0, 1), src1=vgpr(21, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
    # Bead `rocm-libraries-cpe`: richer adversarial negative tests for the
    # MFMA quad-cycle dispatch surface. Each test below ACTIVELY trips a
    # discriminating dispatch condition rather than only pinning zero-gap
    # behavior — see bead "What's still missing" for the 5 scenarios.
    #
    # Note on cmw formula: post-cmw, `_quad_cycle_gap_ok` returns
    # `actual = slot_delta * (1 + finish) - 1 + subiter_delta` (sound
    # under-estimate). For finish=3 (standard MFMA) this gives:
    #     slot_delta=0 → actual=0   (FAIL: 0 < 3)
    #     slot_delta=1 → actual=3   (PASS: meets threshold)
    #     slot_delta=2 → actual=7   (PASS) +1 if subiter boundary crossed
    # Therefore, the only failure-producing slot configuration in the
    # current formula is slot_delta == 0 (or negative). Tests below
    # discriminate via the `actual` field's exact value at the boundary,
    # via cross-graph routing through diagnose_missing_edge, and via
    # subiter-aware variations that change `actual` without changing pass.
    # -------------------------------------------------------------------------

    def test_mfma_acc_chain_diagnose_missing_edge_with_nonzero_gap(self):
        """Cross-graph route: REF and SUBJ both contain three MFMAs and an
        ALU. In REF, the resolver pairs P (MFMA, slot=2 sub=0, writes v0..v3)
        with C (MFMA, slot=2 sub=1, reads v0..v1) — slot_delta == 0 → the
        TimingTooCloseFailure that compare_graphs synthesizes via
        diagnose_missing_edge MFMA branch carries `expected=3, actual=0`.

        Discriminates from `test_mfma_acc_chain_zero_gap_via_diagnose_missing_edge`
        on TWO axes the prior test did NOT exercise:
          (a) REF and SUBJ have STRUCTURALLY different graphs — REF lacks
              the shadowing ALU (proves the cross-graph diff isolates the
              MFMA→MFMA edge as missing in subj rather than absent in both).
          (b) Asserts the failure's discriminating fields
              (expected_quad_cycles, actual_quad_cycles, producer.category,
              consumer.category) — proving the gap arithmetic populates the
              failure record correctly, not just that dispatch fires.

        A non-zero `actual` would discriminate further but is currently
        unreachable: the formula yields actual ≥ 3 for every slot_delta > 0,
        so no slot configuration produces a failure with actual ∈ {1, 2}.
        Tracked in `rocm-libraries-w7f` (timeline-aware migration)."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            # REF: P writes v0..v3, C immediately reads v0..v1. NO shadowing
            # ALU between them — P->C survives as the missing-from-subj edge.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        subj_alu = VXorB32(dst=vgpr(0, 2), src0=vgpr(40, 1), src1=vgpr(41, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        failures = compare_graphs(g_ref, g_subj, raise_on_unexplained=False)
        mfma_timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
            and getattr(f.consumer, "category", None) == "MFMA"
        ]
        assert mfma_timing, (
            f"Expected diagnose_missing_edge MFMA branch to emit "
            f"TimingTooCloseFailure on the MFMA→MFMA edge missing in subj; "
            f"got: {[type(f).__name__ for f in failures]}"
        )
        f = mfma_timing[0]
        # Discriminating fields — the gap arithmetic must populate these:
        assert f.expected_quad_cycles == 3, (
            f"expected_quad_cycles should be QUAD_CYCLES_STANDARD_MFMA_FINISH "
            f"(3), got {f.expected_quad_cycles}."
        )
        assert f.actual_quad_cycles == 0, (
            f"actual_quad_cycles should be 0 for slot_delta == 0, got "
            f"{f.actual_quad_cycles}."
        )
        assert f.producer.category == "MFMA"
        assert f.consumer.category == "MFMA"

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
                make_lr(8, 4, 64, slot=0, category="LRA0"),
                make_swait(slot=1, dscnt=0),
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
        failures_cross = compare_graphs(g_ref, g_subj,
                                        raise_on_unexplained=False)
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

        # Direct check on _quad_cycle_gap_ok: at the boundary, ok=True with
        # actual == expected == 3. A future strict-inequality regression
        # would flip ok to False here.
        # Find the MFMA→MFMA edge for the direct invariant assertion.
        acc_edge = None
        for e in g_subj.edges:
            if (getattr(e.producer, "category", None) == "MFMA"
                    and getattr(e.consumer, "category", None) == "MFMA"):
                acc_edge = e
                break
        assert acc_edge is not None, "expected MFMA→MFMA acc-chain edge"
        ok, exp, act = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0)
        assert ok, (
            f"_quad_cycle_gap_ok must accept the boundary case "
            f"(actual={act}, expected={exp}); ok was False."
        )
        assert exp == 3 and act == 3, (
            f"At slot_delta==1, expected==actual==3; got "
            f"expected={exp}, actual={act}."
        )

    def test_mfma_to_mfma_cross_subiter_routing(self):
        """Cross-subiter MFMA→MFMA edge: producer at vmfma=0 (subiter 0),
        consumer at vmfma=2 (subiter 1) with num_mfma_per_subiter=2. The
        subiter-aware correction in `_quad_cycle_gap_ok` (post-cmw) adds
        `subiter_delta` quad-cycles to `actual`, so the SAME (slot_delta=2)
        configuration produces a DIFFERENT `actual` depending on whether
        the two MFMAs lie in the same subiter or in different subiters.

        This actively trips the cross-subiter routing path (cmw correction)
        rather than the OrderInverted same-subiter gate from sub-task 11
        (commit f594ff4b09): that gate applies only to ALU producers, not
        to MFMA, so MFMA cross-subiter routing here is governed exclusively
        by `_quad_cycle_gap_ok`'s subiter-delta term.

        Discrimination point: actual_same_subiter (nmps=4, subiter_delta=0)
        != actual_cross_subiter (nmps=2, subiter_delta=1) for the same
        slot_delta=2. A regression that dropped the subiter_delta term
        (the cmw fix) would collapse both to the same value."""
        def _build_cap():
            return make_capture(BODY_LABEL_ML, [
                make_lr(8, 4, 64, slot=0, category="LRA0"),
                make_swait(slot=1, dscnt=0),
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

        ok_s, exp_s, act_s = _quad_cycle_gap_ok(
            e_same.producer, e_same.consumer, 4)
        ok_c, exp_c, act_c = _quad_cycle_gap_ok(
            e_cross.producer, e_cross.consumer, 2)

        # Both pass (actual ≥ expected) — slot_delta=2 produces actual ≥ 7.
        assert ok_s and ok_c, (
            f"Both should pass: same-subiter actual={act_s}/exp={exp_s}, "
            f"cross-subiter actual={act_c}/exp={exp_c}."
        )
        # Discriminator: routing differs — cross-subiter `actual` is HIGHER
        # than same-subiter `actual` by the +subiter_delta correction.
        assert act_c > act_s, (
            f"Cross-subiter routing must add the subiter-delta correction: "
            f"same-subiter actual={act_s}, cross-subiter actual={act_c}. "
            f"A regression that dropped the +subiter_delta term would "
            f"collapse both to the same value."
        )
        # And the difference equals the subiter_delta (1 boundary crossed).
        assert act_c - act_s == 1, (
            f"Cross-subiter correction should be exactly subiter_delta==1; "
            f"got delta={act_c - act_s}."
        )

    def test_mfma_producer_multi_consumer_varied_gaps(self):
        """A single MFMA producer feeds three consumer MFMAs at slot deltas
        0, 1, 2 (i.e. gaps actual=0, 3, 7). The dispatch must short-circuit
        per-edge — only the gap=0 consumer triggers TimingTooCloseFailure;
        the gap=3 and gap=7 consumers pass.

        This catches a regression that early-exits after the first failure
        (e.g. bails out of `validate_edge_wait_coverage`'s edge loop on the
        first TimingTooClose), or one that flags ALL edges from a producer
        once any one of them fails.

        Layout: P at slot=2 writes v0..v3. Three consumers pick up disjoint
        slices of P's accumulator (v0..v1, v2..v3, v0..v1) at slots 2/3/4
        respectively. Each consumer is a distinct MFMA (different c_dst)
        so identities are unique."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            # Producer at slot=2 writes v0..v3 (4-vgpr accumulator).
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            # C1 at slot=2 sub=1 reads v0..v1 — gap=0 → MUST FAIL.
            make_mfma(c_dst_start=40, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
            # C2 at slot=3 reads v2..v3 — gap=3 → MUST PASS (boundary).
            make_mfma(c_dst_start=44, a_src_start=2, b_src_start=32,
                      slot=3, sequence=0, a_src_count=2),
            # C3 at slot=4 reads v0..v1 — gap=7 → MUST PASS.
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
        # Exactly ONE MFMA→MFMA timing failure: the slot_delta==0 consumer.
        actuals = [getattr(f, "actual_quad_cycles", None) for f in mfma_timing]
        assert len(mfma_timing) == 1, (
            f"Expected exactly 1 MFMA→MFMA TimingTooCloseFailure (the "
            f"slot_delta==0 consumer); got {len(mfma_timing)} with "
            f"actuals={actuals}"
        )
        f = mfma_timing[0]
        assert f.actual_quad_cycles == 0, (
            f"The single failure must be the gap=0 consumer; got "
            f"actual={f.actual_quad_cycles}."
        )
        assert f.expected_quad_cycles == 3
        # Sanity: at least 3 MFMA→MFMA edges exist in the graph (one per
        # consumer). Confirms the dispatch saw all three edges, not just
        # the first.
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
        """Mutation smell-test: monkeypatch `_classify_edge_coverage` to a
        no-op (returns []) and confirm at least one of the new
        TimingTooClose tests no longer detects the failure. This proves the
        new tests ACTIVELY exercise the dispatch logic — if they pass with
        the dispatch disabled, they're pinning behavior unrelated to the
        MFMA gap check.

        Targets the within-graph route (`validate_edge_wait_coverage` →
        `_classify_edge_coverage`). Cross-graph route (`compare_graphs` →
        `diagnose_missing_edge`) has its own dispatch and is not affected
        by this patch — that's the desired narrow scope of the mutation:
        confirm `_classify_edge_coverage` is the source of the within-graph
        TimingTooClose verdict.
        """
        from unittest.mock import patch

        # Re-build the same fixture as test_mfma_producer_multi_consumer_varied_gaps:
        # gap=0 consumer at slot=2 sub=1 must produce a TimingTooCloseFailure
        # under the real dispatch.
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, sequence=0, a_src_count=2),
            make_mfma(c_dst_start=40, a_src_start=0, b_src_start=32,
                      slot=2, sequence=1, a_src_count=2),
        ])
        g = build_dataflow_graph(_wrap(cap))

        # Sanity: under the real dispatch, the failure fires.
        real_failures = validate_edge_wait_coverage(g)
        real_timing = [
            f for f in real_failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
        ]
        assert real_timing, (
            "Pre-mutation sanity: the real _classify_edge_coverage must "
            "emit a TimingTooCloseFailure on the zero-gap fixture. If this "
            "fails, the test's premise is wrong, not the dispatch."
        )

        # Mutation: patch _classify_edge_coverage to a no-op. The
        # within-graph dispatch should now produce zero failures.
        with patch(
            "Tensile.Components.ScheduleCapture._classify_edge_coverage",
            return_value=[],
        ):
            mutated_failures = validate_edge_wait_coverage(g)
            mutated_timing = [
                f for f in mutated_failures
                if isinstance(f, TimingTooCloseFailure)
                and getattr(f.producer, "category", None) == "MFMA"
            ]
            assert not mutated_timing, (
                f"Mutation smell-test failed: after patching "
                f"_classify_edge_coverage to a no-op, the dispatch still "
                f"emitted {len(mutated_timing)} MFMA TimingTooCloseFailure "
                f"failures. This means at least one of the new tests is "
                f"NOT exercising _classify_edge_coverage — the dispatch "
                f"path is elsewhere. Investigate the source of the "
                f"residual failure."
            )


# =============================================================================
# P0 — GRInc SRD WAW invisible to subsequent GR (BufferLoad)
# =============================================================================
# `incrementSrd` (KernelWriterAssembly.py:8910) writes Srd{tc}+0/+1 with
# SAddU32/SAddCU32. The next BufferLoad in this iteration reads Srd{tc} via
# constructor param 2 (_inst_buffer_srd at ScheduleCapture.py:1267). If the
# SAddU32 is reordered after the BufferLoad, the load reads stale SRD and
# fetches wrong memory.


class TestGRIncSRDChain:
    def test_grinc_srd_waw_before_buffer_load(self):
        # Reference: SAddU32(Srd+0) then BufferLoad(reads Srd+0..3).
        srd_add = SAddU32(dst=sgpr(20, 1), src0=sgpr(20, 1), src1=sgpr(100, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(srd_add, category="GRIncA", mfma_index=0, sequence=0),
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
                    slot=0, category="GRA", sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject: BufferLoad before SAddU32. Stale SRD.
        srd_add2 = SAddU32(dst=sgpr(20, 1), src0=sgpr(20, 1), src1=sgpr(100, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
                    slot=0, category="GRA", sequence=0),
            _tag(srd_add2, category="GRIncA", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
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
        the pack before the LR consumes uninitialized data."""
        from rocisa.instruction import VCvtPkF32toBF16
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1), src0=vgpr(8, 1), src1=vgpr(9, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            _tag(cvt, category="PackA0", mfma_index=0, sequence=0),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 40, 32, slot=2, a_src_count=2),
        ])
        cvt2 = VCvtPkF32toBF16(dst=vgpr(40, 1), src0=vgpr(8, 1), src1=vgpr(9, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(cvt2, category="PackA0", mfma_index=0, sequence=0),
            make_lr(8, 4, 64, slot=0, category="LRA0", sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 40, 32, slot=2, a_src_count=2),
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
# Bead wx9.8: `_VSwapRule` (ScheduleCapture.py, before `_GenericALURule`)
# publishes both operands as reads AND writes. With the symmetric model the
# shared register carries WAR/WAW/RAW edges in BOTH orderings, and swapping
# the pair flips which instruction is producer vs consumer on each edge —
# so the edge KEY (producer.identity, consumer.identity, register, kind)
# differs between REF and SUBJ and the one-directional `compare_graphs`
# (`missing = ref - subj`) detects the reorder. xfail marker dropped.


class TestVSwapPair:
    def test_vswap_pair_reorder_invisible(self):
        sw1 = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
        sw2 = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            _tag(sw1, category="PackA0", mfma_index=0, sequence=0),
            _tag(sw2, category="PackA0", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        sw1b = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
        sw2b = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            _tag(sw2b, category="PackA0", mfma_index=0, sequence=0),
            _tag(sw1b, category="PackA0", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "Two VSwapB32 instructions sharing v1 reordered — final v1 "
            "value differs but the graph cannot tell."
        )


# =============================================================================
# P1 — VAddCO / VAddCCO VCC chain
# =============================================================================
# globalReadIncrement non-buffer path (KWA:9120-9146) emits
# VAddCOU32 (writes vgpr + VCC carry-out) followed by VAddCCOU32 (reads
# VCC carry-in to do 64-bit add). VCC is implicit — not in dst/src
# attributes — so it can't be tracked through register overlap. But the
# vgpr dst-of-#1 and dst1-of-#1 are also written; if a later instruction
# reads them and then we reorder, that should also surface.


class TestVCCCarryChain:
    def test_vcc_carry_chain_reorder(self):
        """Lower-half add (VAddCOU32) writes v100 and the carry-out VCC;
        upper-half add (VAddCCOU32) writes v101 and reads the carry-in VCC.

        The two instructions touch DISJOINT vgprs (lo: v100, v50; hi: v101,
        v51), so without VCC tracking there is NO edge in the reference
        graph between them and the reorder is invisible. With the wx9.9
        `_VCCRule` publishing VCC as a synthetic resource, the reference
        order forms a VCC RAW edge (lo → hi) that the subject's swapped
        order destroys, surfacing as an `OrderInvertedFailure`.
        """
        from rocisa.container import VCC
        lo = VAddCOU32(dst=vgpr(100, 1), dst1=VCC(),
                       src0=vgpr(100, 1), src1=vgpr(50, 1))
        hi = VAddCCOU32(dst=vgpr(101, 1), dst1=VCC(),
                        src0=vgpr(101, 1), src1=vgpr(51, 1), src2=VCC())
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(lo, category="GRIncA", mfma_index=0, sequence=0),
            _tag(hi, category="GRIncA", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 100, 32, slot=2, a_src_count=2),
        ])
        lo2 = VAddCOU32(dst=vgpr(100, 1), dst1=VCC(),
                        src0=vgpr(100, 1), src1=vgpr(50, 1))
        hi2 = VAddCCOU32(dst=vgpr(101, 1), dst1=VCC(),
                         src0=vgpr(101, 1), src1=vgpr(51, 1), src2=VCC())
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(hi2, category="GRIncA", mfma_index=0, sequence=0),
            _tag(lo2, category="GRIncA", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 100, 32, slot=2, a_src_count=2),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "VAddCO/VAddCCO 64-bit add pair reordered — VCC chain broken; "
            "graph cannot detect this."
        )

    def test_vcc_rule_invariants(self):
        """Lock down the wx9.9 contract: `_is_vcc` recognizes VCC sentinels,
        `_vcc_resource()` is a singleton with regType="vcc"/regNum=2, and
        `_VCCRule` publishes VCC as the carry-out write of VAddCOU32 and
        the carry-in read of VAddCCOU32.
        """
        from rocisa.container import VCC
        from Tensile.Components.ScheduleCapture import (
            _is_vcc, _is_register, _vcc_resource, _VCCRule,
        )

        v = VCC()
        # Inspection invariants — VCC is opaque to the generic resolver,
        # so we identify it by class name.
        assert _is_vcc(v) is True
        assert _is_register(v) is False  # _is_register stays strict; that's the gate
        # Regular registers must not be misclassified as VCC.
        assert _is_vcc(vgpr(100, 1)) is False

        # Singleton invariant — every VCC resource is the same object so
        # producers and consumers hash to the same byte keys.
        r1 = _vcc_resource()
        r2 = _vcc_resource()
        assert r1 is r2
        assert r1.regType == "vcc"
        assert r1.regIdx == 0
        assert r1.regNum == 2

        # _VCCRule produces correct (reads, writes) for VAddCOU32 (carry-out).
        rule = _VCCRule()
        lo = VAddCOU32(dst=vgpr(100, 1), dst1=VCC(),
                       src0=vgpr(100, 1), src1=vgpr(50, 1))
        assert rule.applies(lo)
        reads, writes = rule.extract(lo)
        # Writes: vgpr dst at slot 0, VCC dst1 at slot 1.
        assert len(writes) == 2
        assert writes[1] is _vcc_resource()
        # Reads: vgpr src0, vgpr src1 — no VCC.
        assert all(not _is_vcc(r) for r in reads)
        assert len(reads) == 2

        # _VCCRule produces correct (reads, writes) for VAddCCOU32 (carry chain).
        hi = VAddCCOU32(dst=vgpr(101, 1), dst1=VCC(),
                        src0=vgpr(101, 1), src1=vgpr(51, 1), src2=VCC())
        assert rule.applies(hi)
        reads, writes = rule.extract(hi)
        # Writes: vgpr dst at slot 0, VCC dst1 at slot 1.
        assert len(writes) == 2
        assert writes[1] is _vcc_resource()
        # Reads: src0, src1, and VCC src2 (carry-in) — exactly 3.
        assert len(reads) == 3
        assert reads[2] is _vcc_resource()


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
        wrong LDS half."""
        # Reference: LRS swap, then DSLoad reading the just-swapped vgpr.
        lrs = VXorB32(dst=vgpr(40, 1), src0=vgpr(60, 1), src1=vgpr(40, 1))
        ld_ref = DSLoadB128(dst=vgpr(8, 4), src=vgpr(40, 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(lrs, category="LRSA0", mfma_index=0, sequence=0),
            _tag(ld_ref, category="LRA0", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject: same instructions, swap reordered after the load.
        lrs2 = VXorB32(dst=vgpr(40, 1), src0=vgpr(60, 1), src1=vgpr(40, 1))
        ld_subj = DSLoadB128(dst=vgpr(8, 4), src=vgpr(40, 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(ld_subj, category="LRA0", mfma_index=0, sequence=0),
            _tag(lrs2, category="LRSA0", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
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
        consuming DSStoreB128 — store writes to pre-swap LDS half."""
        # Reference: LWS swap, then DSStore using the just-swapped vgpr.
        lws = VXorB32(dst=vgpr(50, 1), src0=vgpr(70, 1), src1=vgpr(50, 1))
        st_ref = DSStoreB128(dstAddr=vgpr(50, 1), src=vgpr(8, 4))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(lws, category="LWSA", mfma_index=0, sequence=0),
            _tag(st_ref, category="LWA", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 100, 32, slot=2, a_src_count=4),
        ])
        # Subject: store before swap.
        lws2 = VXorB32(dst=vgpr(50, 1), src0=vgpr(70, 1), src1=vgpr(50, 1))
        st_subj = DSStoreB128(dstAddr=vgpr(50, 1), src=vgpr(8, 4))
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(st_subj, category="LWA", mfma_index=0, sequence=0),
            _tag(lws2, category="LWSA", mfma_index=0, sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 100, 32, slot=2, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj), (
            "LWS VXorB32 writing LocalWriteAddrA reordered after the "
            "DSStoreB128 reading it — should have produced an "
            "OrderInvertedFailure on the LDS-address vgpr RAW edge."
        )

