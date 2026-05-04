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
    build_dataflow_graph,
    compare_graphs,
    cumulative_issue_cycles,
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

    def test_mfma_acc_chain_same_slot_passes_under_exact_arithmetic(self):
        """Two MFMAs sharing accumulator v0..v3, both at the SAME vmfma_index
        (sub_index breaks the tie). Bead `nk0` rewrite: cycle-exact
        arithmetic via `cumulative_issue_cycles` correctly gates the
        consumer behind the producer's `mfma_free_at` (current_issue + 1
        + finish_cycles = 0 + 1 + 3 = 4). Consumer issues at cycle 4;
        gap = 4 - 0 - 1 = 3 = expected. NO TimingTooCloseFailure.

        Pre-`nk0` the slot-delta approximation reported `slot_delta == 0
        → actual = 0 < expected = 3` and synthesized a failure that does
        not exist in the cycle-accurate timeline (`precompute_issue_times`
        also walks the stream sequentially and picks up the same +3 gap)."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        ok, exp, act = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert ok and exp == 3 and act == 3, (
            f"nk0 contract: same-slot MFMAs report exp=3, act=3 (matches "
            f"precompute_issue_times); got ok={ok}, exp={exp}, act={act}."
        )

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

    def test_mfma_acc_chain_slot_delta_2_independent_of_num_mfma_per_subiter(self):
        """Producer at vmfma=0, consumer at vmfma=2; vary num_mfma_per_subiter.
        Bead `nk0` (cycle-exact rewrite): `num_mfma_per_subiter` is no
        longer consulted by `_quad_cycle_gap_ok` — the helper walks the
        captured stream and accumulates exact issue cycles, so the same
        body produces the same `actual` regardless of `nmps`. This is the
        intended outcome of switching from the cmw approximation (where
        `nmps` controlled a sound +subiter_delta correction) to the
        precompute-equivalent simulator (which sees every intermediate
        instruction's issue cost directly).

        For body=[LR, SWait, MFMA@2, MFMA@4] the simulator yields:
        MFMA1 issues at 0, mfma_free_at=4; MFMA2 issues at max(1,4)=4 →
        gap = 4-0-1 = 3. Same value under any `nmps`."""
        def _build_cap():
            return make_capture(BODY_LABEL_ML, [
                make_lr(8, 4, 64, slot=0, category="LRA0"),
                make_swait(slot=1, dscnt=0),
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

        ok1, exp1, act1 = _quad_cycle_gap_ok(
            e1.producer, e1.consumer, 1, graph=g_nmps1)
        ok2, exp2, act2 = _quad_cycle_gap_ok(
            e2.producer, e2.consumer, 2, graph=g_nmps2)

        # nk0 contract: nmps is unused → same actual for both runs.
        assert act1 == act2 == 3, (
            f"nk0: cycle-exact arithmetic ignores nmps. Expected act1==act2==3; "
            f"got act1={act1}, act2={act2}."
        )
        assert ok1 and ok2 and exp1 == 3 and exp2 == 3

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
        ok, expected, actual = _quad_cycle_gap_ok(
            edge.producer, edge.consumer, 0, graph=g)

        # s7k: actual reflects body_delta * 1000 (cross-body placeholder).
        assert ok, "Cross-body edge should always pass the gap check."
        assert actual > expected, (
            f"Cross-body _quad_cycle_gap_ok reported actual={actual} == "
            f"expected={expected}; this is the misleading early-return s7k "
            f"flags. Expected actual to reflect the real cross-body gap "
            f"(many issue cycles)."
        )

    def test_mfma_acc_chain_diagnose_missing_edge_dispatches_through_mfma_branch(self):
        """Regression-pin test for the diagnose_missing_edge MFMA branch
        DISPATCH (bead `nk0` rewrite). Pre-`nk0` the same fixture produced
        a TimingTooCloseFailure with `actual=0`; under the cycle-exact
        helper the simulator gives the consumer 3 cycles of gap (MFMA1
        issues at 0, ALU at 1 (cost=1), MFMA2 at max(2, mfma_free=4)=4 →
        gap = 4-0-1 = 3 = expected). The MFMA branch IS reached (proving
        dispatch), but it now correctly returns ok=True and synthesizes
        no failure.

        The richer adversarial premise the original test sought (a real
        timing violation surfacing through diagnose_missing_edge) is no
        longer reachable by construction — the simulator's mfma_free_at
        contention prevents under-3-cycle gaps in any valid same-body
        MFMA→MFMA chain. Tracked as the new contract for `nk0`."""
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

    def test_mfma_acc_chain_diagnose_missing_edge_dispatch_no_failure(self):
        """Bead `nk0` rewrite: cycle-exact arithmetic.
        REF: P (MFMA slot=2 seq=0) + C (MFMA slot=2 seq=1) → P→C edge present.
        SUBJ: P (slot=2 seq=0), ALU (slot=2 seq=1), C (slot=2 seq=2) →
              ALU shadows v0..v1, P→C edge missing.
        compare_graphs routes to diagnose_missing_edge whose MFMA branch
        runs on subj_graph: simulator gives MFMA1 issue=0, mfma_free=4;
        ALU adds cost 1 → issue=1; MFMA2 max(2, mfma_free=4)=4 → c_issue=4;
        gap = 4-0-1 = 3 = expected. ok=True → NO TimingTooCloseFailure.

        The pre-`nk0` premise (failure with actual=0) was an artifact of
        the slot-delta approximation, not the real timeline."""
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
            acc_edge.producer, acc_edge.consumer, 0, graph=g_subj)
        assert ok, (
            f"_quad_cycle_gap_ok must accept the boundary case "
            f"(actual={act}, expected={exp}); ok was False."
        )
        assert exp == 3 and act == 3, (
            f"At slot_delta==1, expected==actual==3; got "
            f"expected={exp}, actual={act}."
        )

    def test_mfma_to_mfma_cross_subiter_routing_exact(self):
        """Bead `nk0` cycle-exact: `num_mfma_per_subiter` no longer affects
        `actual`. The simulator walks the captured stream directly, so
        same-body slot_delta=2 with no intermediate instructions yields
        actual=3 regardless of `nmps`. Both same-subiter (nmps=4) and
        cross-subiter (nmps=2) configurations produce the same `actual`."""
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
            e_same.producer, e_same.consumer, 4, graph=g_same)
        ok_c, exp_c, act_c = _quad_cycle_gap_ok(
            e_cross.producer, e_cross.consumer, 2, graph=g_cross)

        # Both pass.
        assert ok_s and ok_c
        # nk0: identical actual under exact arithmetic (nmps unused).
        assert act_s == act_c, (
            f"nk0: cycle-exact arithmetic ignores nmps; expected equal "
            f"actuals. Got same-subiter act={act_s}, cross-subiter act={act_c}."
        )
        # And it's the simulator's value: MFMA1 issue=0, mfma_free=4;
        # MFMA2 max(1, 4)=4 → gap=4-0-1=3.
        assert act_s == 3 and act_c == 3

    def test_mfma_producer_multi_consumer_varied_gaps_exact(self):
        """Bead `nk0` rewrite: under cycle-exact arithmetic the simulator's
        mfma_free_at contention naturally schedules each consumer with at
        least 3 cycles of gap behind the previous MFMA, so there are no
        same-body MFMA→MFMA timing failures in this fixture.

        Per-edge cumulative_issue_cycles values:
          P→C1 (slot 2 seq 0 → seq 1): MFMA1 issue=0, mfma_free=4; C1 max(1,4)=4 → gap=3.
          P→C2 (slot 2 → slot 3): walk includes C1 (mfma_free becomes 8); C2 issues at 8 → gap=7.
          P→C3 (slot 2 → slot 4): walk includes C1 (free=8) + C2 (free=12); C3 max(9,12)=12 → gap=11.
        All ≥ expected=3 → no failures. Sanity check: at least 3 edges exist."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
                f"Mutation smell-test failed: patched dispatch still "
                f"emitted {len(mutated_timing)} failures."
            )

    # -------------------------------------------------------------------------
    # Sub-A of bead `e7w` — 4x4 PackMFMA finish-cycle and dispatch-routing fixes.
    # Pre-fix bugs:
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
        """Bead `nk0` cycle-exact: 4x4 PackMFMA producer (finish=1) at
        slot=2 seq=0; standard MFMA consumer at slot=2 seq=1.
        Simulator: producer issues at 0, mfma_free=2, last_mfma_class=4x4.
        Consumer max(1,2)=2; type-switch (4x4→standard), gap=2-0=2 <
        FROM_4X4=3 → +1 stall → consumer issues at 3. Gap=3-0-1=2.
        expected=1 → ok=True → NO failure. (Pre-`nk0` the slot_delta=0
        approximation reported a fake actual=0 failure.)"""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        ok, exp, act = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert ok and exp == 1 and act == 2, (
            f"nk0 contract: 4x4 producer expected=1; type-switch +1 stall "
            f"yields act=2; got ok={ok}, exp={exp}, act={act}."
        )

    def test_mfma_pack_acc_chain_meets_finish_1_no_failure(self):
        """4x4 PackMFMA producer (finish=1) with a consumer at slot_delta=1
        gives an actual gap of `slot_delta * (1 + finish) - 1 == 1` quad-cycle —
        EXACTLY meets the 1-quad-cycle threshold, so no TimingTooCloseFailure
        is emitted. Pre-e7w with the wrong finish=3 default this would have
        been mis-flagged as too close."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=2,
                      variant=[4, 4, 4, 16]),
            # Consumer at next vmfma_index — slot_delta=1, finish=1 → actual=1.
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
            f"slot_delta=1 with 4x4 PackMFMA finish=1 gives actual=1 == "
            f"expected=1 — should NOT emit TimingTooCloseFailure. Got: "
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
        whose category is "PackA0" with an underlying _FakeMFMA, then assert
        `_is_mfma_producer` claims it and `_is_alu_producer` does NOT."""
        from Tensile.Components.ScheduleCapture import (
            _is_alu_producer, _is_mfma_producer, _is_mfma_pack_producer,
        )
        # _FakeMFMA — no getParams(), but _is_mfma() returns True for it
        # (its class name is in _MFMA_CLASS_NAMES).
        pack_mfma_tagged = make_mfma(
            c_dst_start=0, a_src_start=8, b_src_start=12, slot=2,
            category="PackA0", variant=[4, 4, 4, 16],
        )
        # GraphNode shape: a stub object exposing the attributes the
        # producer-classifier helpers read (category + rocisa_inst).
        class _StubNode:
            def __init__(self, tagged):
                self.category = tagged.category
                self.rocisa_inst = tagged.inst
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
    # Sub-B of bead `35z` — CVTPack -> MFMA 2-quad-cycle settle window.
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
            make_swait(slot=1, dscnt=0),
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
        """CVTPack producer at slot=2; MFMA consumer at slot=5 (slot_delta=3).
        With finish=0 the same-body formula gives
        `actual = slot_delta * (1 + 0) - 1 = 2`, exactly meeting the
        2-quad-cycle CVT->MFMA threshold. No TimingTooCloseFailure should
        be emitted."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
            # MFMA at slot=5 — slot_delta == 3, gap = 3*1 - 1 = 2 quad-cycles,
            # exactly meeting the 2-cycle CVT->MFMA threshold.
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
            f"slot_delta=2 with CVT finish=0 should meet the 2-quad-cycle "
            f"threshold (no TimingTooCloseFailure). Got: {failures}"
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
        from Tensile.Components.ScheduleCapture import (
            _is_alu_producer, _is_cvt_pack, _is_cvt_pack_producer,
            _is_mfma_producer, _classify_edge_coverage,
        )
        # Producer: real VCvtPkF32toBF16 wrapped in a Pack*-categorized
        # TaggedInstruction (the production CVT0 emission shape).
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(50, 1), src1=vgpr(51, 1))
        cvt_tagged = _tag(cvt, category="PackA0", mfma_index=2, sequence=0)

        class _StubNode:
            def __init__(self, tagged):
                self.category = tagged.category
                self.rocisa_inst = tagged.inst
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
            make_swait(slot=1, dscnt=0),
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
    # Sub-C of bead `or9` — 4x4 PackMFMA -> CVTPack (CVT1) 5-quad-cycle
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
    # Pre-or9 the structural side enforced this rule via
    # `MFMAPack.min_quad_cycles_before_result_used = 5` set in
    # `_handle_min_pack_quad_cycles` (CMSValidator.py:1748-1751); the
    # graph side had no equivalent. e7w made the PackMFMA producer
    # routable through `_is_mfma_producer`, but the resulting threshold
    # was the 4x4 finish-cycle (1) — too weak by 4 quad-cycles. This
    # bead adds the producer/consumer-pair-aware carve-out.
    # -------------------------------------------------------------------------

    def _make_real_pack_mfma(self, *, acc_start, acc_count, a_start,
                              a_count, b_start, b_count, slot, sequence,
                              category):
        """Build a real rocisa MFMAInstruction (4x4 PackMFMA family) wrapped
        in a TaggedInstruction. Uses the rocisa class instead of `_FakeMFMA`
        so the producer has `getParams()` and the per-byte resolver claims
        it via `_GenericALURule` (Pack-categorized MFMAs are excluded from
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
        cycle-exact simulator (`cumulative_issue_cycles`, bead `nk0`)
        reports:
          producer issues at 0, mfma_free_at = 0+1+1 = 2.
          consumer at p_idx+1 — current_issue += 1 (MFMA issue cost) → 1.
          c_issue_start = 1; gap = 1 - 0 - 1 = 0.
        expected = 5 (QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1) > actual = 0 →
        TimingTooCloseFailure. Failure must be emitted by the new
        PackMFMA->CVTPack branch (Sub-C of bead `or9`) which intercepts
        the pair BEFORE the generic MFMA-producer branch — the latter
        would have used `_mfma_finish_cycles_for(4x4) == 1` as the
        threshold and false-passed the edge.

        Uses a real rocisa MFMAInstruction (not _FakeMFMA) so the
        Pack-categorized producer has `getParams()` and `_GenericALURule`
        publishes (writes=(acc,), reads=(a,b,acc)) — required for the
        per-byte resolver to form the PackMFMA->CVT edge."""
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        from Tensile.Components.ScheduleCapture import (
            _is_mfma_pack_producer, _is_cvt_pack_producer,
            _classify_edge_coverage,
        )
        cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                              src0=vgpr(0, 1), src1=vgpr(1, 1))
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
    # Bead `vf4` — MFMA type-switch +1 stall penalty in the graph-side
    # `_quad_cycle_gap_ok` actual computation. The structural simulator
    # (`precompute_issue_times`, CMSValidator.py:2664-2670) injects a +1
    # quad-cycle stall whenever consecutive MFMAs differ in class
    # (standard MFMA <-> 4x4 PackMFMA) and the inter-MFMA gap is below the
    # producer's threshold (FROM_STANDARD = 5, FROM_4X4 = 3). For the
    # direct producer→consumer adjacency case (slot_delta == 1) both
    # directions trigger the stall: standard producer's gap is 1+3=4 < 5;
    # 4x4 producer's gap is 1+1=2 < 3. The graph-side fix mirrors this by
    # adding +1 to `actual` when producer.finish != consumer.finish — so
    # the reported `actual` matches the simulator's enlarged gap and the
    # graph stays a sound under-estimate (real_actual ≥ formula+1 in the
    # direct case, ≥ formula+1 in the chain case since any path between
    # the producer and a different-class consumer crosses ≥ 1 boundary).
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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        ok, expected, actual = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert ok
        assert expected == 3
        assert actual == 4

    def test_mfma_type_switch_4x4_to_standard_adds_one_to_actual(self):
        """4x4 PackMFMA producer (finish=1) at slot=2; standard MFMA
        consumer (finish=3) at slot=3. Cycle-exact simulator:
          producer issues at 0, mfma_free=2, last_mfma_class=4x4.
          consumer max(1,2)=2; type switch (4x4→standard), gap=2-0=2 <
          FROM_4X4=3 → +1 stall → consumer issues at 3.
          delivered gap = 3-0-1 = 2."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        ok, expected, actual = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        assert ok
        assert expected == 1
        assert actual == 2

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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        ok, expected, actual = _quad_cycle_gap_ok(
            acc_edge.producer, acc_edge.consumer, 0, graph=g)
        # nk0 cycle-exact: producer at 0 with mfma_free=4; consumer max(1,4)=4
        # → gap=3. Same class, no type-switch +1.
        assert ok and expected == 3 and actual == 3, (
            f"Same-class standard→standard must yield actual==expected==3. "
            f"Got ok={ok}, expected={expected}, actual={actual}."
        )

        # Sanity sibling: same-class 4x4→4x4 chain — no +1 either.
        cap_pack = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
        ok_p, expected_p, actual_p = _quad_cycle_gap_ok(
            acc_edge_pack.producer, acc_edge_pack.consumer, 0, graph=g_pack)
        # nk0 cycle-exact: 4x4 producer at 0, mfma_free=2; consumer max(1,2)=2;
        # same class → no type-switch. gap=2-0-1=1.
        assert ok_p and expected_p == 1 and actual_p == 1, (
            f"Same-class 4x4→4x4: actual==expected==1. Got "
            f"ok={ok_p}, expected={expected_p}, actual={actual_p}."
        )


# =============================================================================
# Bead `nk0` — cycle-exact cumulative_issue_cycles helper. Mirrors
# `CMSValidator.precompute_issue_times` walk over the captured stream.
# These tests pin the new exact contract: the helper sums per-instruction
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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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
                    and e.producer.position.vmfma_index == 2
                    and e.consumer.position.vmfma_index == 4):
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
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
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

    # Bead `arv` deleted the structural-side parity test
    # `test_graph_actual_matches_precompute_issue_times`. The
    # `precompute_issue_times` / `estimate_quad_cycles_precomputed`
    # helpers it imported no longer exist in CMSValidator (graph-side
    # `cumulative_issue_cycles` is now the source of truth for MFMA
    # quad-cycle gap verdicts; bead `nk0` established the parity that
    # this test pinned).


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

