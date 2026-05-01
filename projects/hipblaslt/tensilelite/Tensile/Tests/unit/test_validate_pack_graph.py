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
"""Graph-native ports of test_ValidatePack.py — bead ola.4 (partial).

Sub-task ola.4 of the CMS validation migration epic (`br show
rocm-libraries-ola`): replace the structural rule `add_pack_constraints`
in CMSValidator.py with graph-side equivalents.

Scope of this file (initial migration):
  * LR -> Pack RAW dependency tests (TestLRAfterPack / _LR1 / _LR3): the
    Pack reads a vgpr the LR writes; reordering Pack before LR is a
    cross-graph order inversion. Graph machinery: `_GenericALURule` claims
    Pack VCvt* writes/reads, `compare_graphs` emits OrderInvertedFailure,
    `validate_edge_wait_coverage` emits MissingWait / WaitInsufficient on
    the LR -> Pack vlcnt drain.
  * Pack -> MFMA RAW (TestPackAfterMFMA / _LR1 / _LR3): the MFMA reads a
    vgpr the Pack writes; reordering Pack to issue at-or-after the MFMA is
    detected by `compare_graphs` Phase-1 order check. ALU producers have
    no SWait coverage requirement so no wait-coverage failure follows.

Out of scope (deferred to sibling beads — see `br show rocm-libraries-ola.4`
discussion and the parent epic's "Sequencing" section):

  * MiddlePack pair-interleaving (`WrongInterleavingFailure`): the f32
    pair-consumer ordering invariant in `_hook_up_packs_f32` has no
    graph-side classifier today. Recommendation: add a sibling sub-bead
    `ola.4.MiddlePack` for the new graph-side rule.
  * CVT-pair quad-cycle timing (`TimingTooCloseFailure` for ALU-producer
    Pack -> MFMA gaps): graph's `_quad_cycle_gap_ok` only dispatches on
    MFMA producers. Pack-as-producer quad-cycle gaps are owned by the
    in-flight `35z` quad-cycle work (see CMSValidator.estimate_quad_cycles
    + Pack.min_quad_cycles_before_result_used).
  * MFMA reorder + multi-tile + swap-pack tests: depend on Pack -> Pack
    pair ordering and CVT-pair timing, both blocked above.

Production-side deletions deferred until all 16 test classes have
graph-native equivalents (see deletion checklist in the bead's
"Deletion checklist" section). The legacy `test_ValidatePack.py` is
KEPT for now to preserve coverage during the incremental migration.
"""

from rocisa.container import vgpr
from rocisa.instruction import VCvtPkF32toBF16, VPermB32

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    OrderInvertedFailure,
    MissingWaitFailure,
)

from dataflow_fixtures import (
    make_capture, make_lr, make_mfma, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# =============================================================================
# Helpers
# =============================================================================
# Pack instructions in real CMS captures span VPermB32 (BF16), VCvtPk*
# (TF32), VLShift* (TF32 emulation), VPack* (variants). For the LR->Pack
# RAW edge tests we only need ONE Pack-shaped instruction whose reads
# overlap the LR's writes. `_GenericALURule` claims all of them via the
# dst_then_srcs convention, so VCvtPkF32toBF16 is a good representative.


def _pack_vcvt(out_vgpr: int, lr_vgpr: int, *, slot: int, sequence: int = 0,
               category: str = "PackA0") -> TaggedInstruction:
    """A Pack VCvtPkF32toBF16 reading two LR-output vgprs.

    `lr_vgpr` is the base of the LR's vgpr range; the VCvt reads
    [lr_vgpr] and [lr_vgpr+1] (two source operands). Tests should pick
    `lr_vgpr` to overlap a `make_lr(dst_vgpr_start=lr_vgpr, dst_vgpr_count=2)`
    so the dataflow graph forms an LR -> Pack RAW edge.
    """
    inst = VCvtPkF32toBF16(
        dst=vgpr(out_vgpr, 1),
        src0=vgpr(lr_vgpr, 1),
        src1=vgpr(lr_vgpr + 1, 1),
    )
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def _pack_vperm(out_vgpr: int, lr_vgpr: int, *, slot: int, sequence: int = 0,
                category: str = "PackA0") -> TaggedInstruction:
    """A Pack VPermB32 (BF16-style) reading two LR-output vgprs.

    BF16 packs map to v_perm_b32 in real schedules. Same shape as the
    VCvt helper but using a different rocisa class so the edge formation
    is exercised across both Pack flavors. selector is sgpr — but we
    can pass src1 as another vgpr; rocisa accepts it for class shape.
    """
    inst = VPermB32(
        dst=vgpr(out_vgpr, 1),
        src0=vgpr(lr_vgpr, 1),
        src1=vgpr(lr_vgpr + 1, 1),
        src2=vgpr(lr_vgpr + 2, 1),
    )
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


# =============================================================================
# LR -> Pack RAW edge tests (port of TestLRAfterPack family)
# =============================================================================
# Original tests assert ConstraintViolationFailure(producer=LR, consumer=Pack)
# when the Pack issues before its LR is guaranteed complete. Graph-native
# equivalent: `compare_graphs` emits OrderInvertedFailure when subj's Pack
# is positioned BEFORE subj's LR (variant A); `validate_edge_wait_coverage`
# emits MissingWaitFailure(counter_kind="vlcnt") when the LR's drain SWait
# sits AFTER the Pack consumer (variant B).
#
# Note: the legacy tests use slot indices 0..7 and a single-LR-per-slot
# convention. The graph-native fixtures use explicit register identities,
# so each test wires only ONE LR + ONE Pack pair — sufficient to fire the
# edge classifier under test. Multi-LR / multi-Pack tests are folded into
# the graph-side per-edge model implicitly.


class TestLRAfterPackGraph(GraphNativeValidationTest):
    """Graph-native port of test_ValidatePack.py::TestLRAfterPack.

    Models the LR0 -> PackA0 RAW dependency. The legacy class has three
    tests: variant A (LR positionally after Pack), variant B (LR before
    Pack but SWait after Pack), and a baseline-passing case.
    """

    def test_LR0_A_LR_after_Pack(self):
        """Subj reorders PackA0 to issue BEFORE LRA0. compare_graphs's
        Phase-1 order check fires because the LR -> Pack RAW edge has
        producer.position > consumer.position in subj.
        """
        # Reference: LR @ slot 0, Pack @ slot 3, MFMA @ slot 4 — canonical.
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        # Subject: Pack @ slot 0 (BEFORE the LR @ slot 1) — order inversion.
        subj_cap = make_capture(BODY_LABEL_ML, [
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=0, category="PackA0"),
            make_lr(8, 2, lds_offset=64, slot=1, category="LRA0", sequence=1),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.compare(self.wrap_single_body(ref_cap),
                                self.wrap_single_body(subj_cap))
        f = self.assert_failures_contain(
            failures, cls=OrderInvertedFailure,
        )
        # Pin identity: producer is LR (the v8 writer), consumer is Pack.
        assert f.producer.category == "LRA0"
        assert f.consumer.category == "PackA0"

    def test_LR0_B_waitcnt_after_Pack(self):
        """Subj has LR @ slot 0, Pack @ slot 1, SWait(dscnt=0) @ slot 2.
        The LR -> Pack vlcnt drain is missing in the window between LR
        (slot 0) and Pack (slot 1). validate_edge_wait_coverage emits
        MissingWaitFailure for the LR -> Pack edge.

        Note: DSLoad uses dscnt (dscnt drain). Subj's SWait IS at slot 2
        with dscnt=0, but it sits AFTER the Pack consumer. The window
        check looks for dscnt SWaits between producer and consumer.
        """
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=1, category="PackA0"),
            # SWait sits AFTER the Pack — doesn't cover the LR -> Pack edge.
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(subj_cap)))
        # MissingWait on dscnt for the LR -> Pack edge.
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure,
            counter_kind="dscnt",
        )

    def test_LR0_baseline_passing(self):
        """Canonical placement: LR @ 0, SWait(dscnt=0) @ 2, Pack @ 3, MFMA @ 6.
        validate_edge_wait_coverage finds the dscnt drain in the window
        between LR and Pack and emits no failure. MFMA is slotted with a
        2-cycle settle window after the CVTPack producer (35z's
        QUAD_CYCLES_CVT_BEFORE_MFMA = 2 requirement).
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=6, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)


class TestLRAfterPack_BF16Graph(GraphNativeValidationTest):
    """Graph-native port covering the BF16/VPermB32 Pack flavor.

    Original BF16 Pack tests (TestValidatePackBF16) used VPermB32 packs
    instead of VCvt*. Same RAW dependency shape; ensures the Pack edge
    formation works for both Pack rocisa flavors.
    """

    def test_BF16_pack_after_LR_baseline_passing(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vperm(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)

    def test_BF16_pack_before_LR_swait_too_late(self):
        """SWait sits after the Pack — Pack reads stale data."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, lds_offset=64, slot=0, category="LRA0"),
            _pack_vperm(out_vgpr=40, lr_vgpr=8, slot=1, category="PackA0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(subj_cap)))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# =============================================================================
# Pack -> MFMA RAW edge tests (port of TestPackAfterMFMA family)
# =============================================================================
# Original tests assert ConstraintViolationFailure(producer=Pack,
# consumer=MFMA) when the Pack issues at-or-after the MFMA that consumes
# its result. Graph-native equivalent: `compare_graphs` Phase-1 order
# check fires when subj's Pack.position > MFMA.position. ALU producers
# have no wait-coverage requirement, so no wait-coverage failure follows.


class TestPackAfterMFMAGraph(GraphNativeValidationTest):
    """Graph-native port of test_ValidatePack.py::TestPackAfterMFMA."""

    def test_LR0_Pack_after_MFMA(self):
        """Subj reorders PackA0 to slot 5 — AFTER the MFMA at slot 4 that
        reads its output vgpr. Phase-1 order check fires.
        """
        # Reference: Pack @ 3, MFMA @ 4 — canonical order.
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        # Subject: Pack @ 5, AFTER the MFMA @ 4.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=5, category="PackA0"),
        ])
        failures = self.compare(self.wrap_single_body(ref_cap),
                                self.wrap_single_body(subj_cap))
        f = self.assert_failures_contain(failures, cls=OrderInvertedFailure)
        # The reordered edge: Pack -> MFMA RAW on v40.
        assert f.producer.category == "PackA0"
        assert f.consumer.category == "MFMA"

    def test_LR0_baseline_passing(self):
        """Canonical Pack-before-MFMA: no failure."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.compare(self.wrap_single_body(ref_cap),
                                self.wrap_single_body(cap))
        self.assert_no_failures(failures)


class TestLRAfterPack_LR1Graph(GraphNativeValidationTest):
    """Graph-native port of TestLRAfterPack_LR1.

    LR1 -> PackA1 RAW edge in the same subiter. Real schedules use
    UsePLRPack=True to place PackA1 inside the loop; graph-native
    fixtures don't care about kernel flags, only the explicit register
    chain.

    Note: original LR1 test asserts the LR1 producer / Pack1 consumer
    pairing fires "issued too early" via the legacy ConstraintViolationFailure.
    The graph-native equivalent uses Pack VCvt with subiter index 1 in
    its category (PackA1) and LRA1 (subiter 1), so the within-subiter
    order check applies.
    """

    def test_LR1_A_LR_after_Pack(self):
        """Subj reorders PackA1 to issue BEFORE LRA1 (same subiter 1)."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, lds_offset=128, slot=0, category="LRA1"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=60, lr_vgpr=50, slot=3, category="PackA1"),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            _pack_vcvt(out_vgpr=60, lr_vgpr=50, slot=0, category="PackA1"),
            make_lr(50, 2, lds_offset=128, slot=1, category="LRA1", sequence=1),
            make_swait(slot=2, dscnt=0),
        ])
        failures = self.compare(self.wrap_single_body(ref_cap),
                                self.wrap_single_body(subj_cap))
        f = self.assert_failures_contain(failures, cls=OrderInvertedFailure)
        assert f.producer.category == "LRA1"
        assert f.consumer.category == "PackA1"

    def test_LR1_B_waitcnt_after_Pack(self):
        """LR1 at slot 0, Pack1 at slot 1, SWait sits AFTER Pack — coverage gap."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 2, lds_offset=128, slot=0, category="LRA1"),
            _pack_vcvt(out_vgpr=60, lr_vgpr=50, slot=1, category="PackA1"),
            make_swait(slot=2, dscnt=0),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(subj_cap)))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# Note on TestPackAfterMFMA_LR1 / _LR3 (legacy test_ValidatePack.py): those
# classes are redundant in the graph-native model. The legacy tests
# couldn't fire Pack1/Pack3 -> next-iter-MFMA in a single-iter timeline,
# so they fell back to exercising PackA0 -> MFMA — which is already
# covered by TestPackAfterMFMAGraph::test_LR0_Pack_after_MFMA above.
# In the graph-native model, kernel config (UsePLRPack, ForceUnrollSubIter,
# UseDirect32XEmulation) does not change graph edge formation, so a
# parallel LR1/LR3 graph variant would assert the same thing as LR0.
