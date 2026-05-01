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
    OrderInvertedFailure,
)

from dataflow_fixtures import (
    make_lr, make_gr, make_mfma, make_swait, make_capture,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap(ml_capture):
    """Wrap a single main-loop capture into a FourPartCapture, fillering the
    other 3 bodies with a no-op MFMA so build_dataflow_graph's
    non-empty-body precondition is satisfied."""
    def _filler(label):
        return make_capture(label, [make_mfma(
            c_dst_start=200, a_src_start=204, b_src_start=208, slot=0,
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
        slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=mfma_index, sequence=sequence),
    )


_GAP_XFAIL_REASON = (
    "Documented gap in ScheduleCapture._writes / _reads (lines 1607-1629). "
    "Only LR/GR/LW/MFMA are tracked; this class returns [] from both. To "
    "remove the xfail: extend _writes/_reads to handle the class, ensure "
    "_reg_overlaps treats the affected register type symmetrically, and "
    "verify the new edges don't break existing tests."
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

    @pytest.mark.xfail(reason=_GAP_XFAIL_REASON, strict=True)
    def test_dtl_m0_update_before_buffer_load(self):
        # Reference: m0 set, then BufferLoad consumes m0.
        m0_set = SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA", 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(m0_set, category="GRA", mfma_index=0, sequence=0),
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
                    slot=0, category="GRA", sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject: same instructions, m0-update issued AFTER the BufferLoad.
        # The load sees stale m0.
        m0_set2 = SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA", 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
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

    @pytest.mark.xfail(reason=_GAP_XFAIL_REASON, strict=True)
    def test_dtl_m0_add_update_before_buffer_load(self):
        """Same as above but the m0 update is the SAddU32 form
        (DTL + IncLdsBufSwitch / DTL + ExpandPointerSwap path)."""
        m0_add = SAddU32(dst=mgpr(0),
                         src0=sgpr("LocalWriteAddrA", 1),
                         src1=sgpr("LDSBufferWriteInc", 1))
        ref_cap = make_capture(BODY_LABEL_ML, [
            _tag(m0_add, category="GRA", mfma_index=0, sequence=0),
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
                    slot=0, category="GRA", sequence=1),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        m0_add2 = SAddU32(dst=mgpr(0),
                          src0=sgpr("LocalWriteAddrA", 1),
                          src1=sgpr("LDSBufferWriteInc", 1))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_gr(8, 4, srd_sgpr_start=20, immediate_offset=0,
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
# P0 — MFMA accumulator RAW chain
# =============================================================================
# `_reads(MFMA)` correctly enumerates a, b, AND c_dst (read-modify-write at
# ScheduleCapture.py:1623-1628). But `_writes(MFMA)` returns []. So a
# subsequent MFMA reading the same accumulator vgpr forms no edge to its
# producing MFMA. Reordering two MFMAs that share an accumulator silently
# corrupts the accumulator value.


class TestMFMASelfRAW:
    @pytest.mark.xfail(reason=_GAP_XFAIL_REASON, strict=True)
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


# =============================================================================
# P0 — GRInc SRD WAW invisible to subsequent GR (BufferLoad)
# =============================================================================
# `incrementSrd` (KernelWriterAssembly.py:8910) writes Srd{tc}+0/+1 with
# SAddU32/SAddCU32. The next BufferLoad in this iteration reads Srd{tc} via
# constructor param 2 (_inst_buffer_srd at ScheduleCapture.py:1267). If the
# SAddU32 is reordered after the BufferLoad, the load reads stale SRD and
# fetches wrong memory.


class TestGRIncSRDChain:
    @pytest.mark.xfail(reason=_GAP_XFAIL_REASON, strict=True)
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
    @pytest.mark.xfail(reason=_GAP_XFAIL_REASON, strict=True)
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
# Reversing them changes the final state. Today VSwap returns [] from both
# _writes and _reads.


class TestVSwapPair:
    @pytest.mark.xfail(reason=_GAP_XFAIL_REASON, strict=True)
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
    @pytest.mark.xfail(reason=_GAP_XFAIL_REASON, strict=True)
    def test_vcc_carry_chain_reorder(self):
        """Lower-half add writes v100; upper-half add (VAddCCOU32) reads
        the same vgpr position and the implicit VCC. Reorder breaks the
        64-bit add even ignoring VCC, because the vgpr WAW is wrong."""
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
