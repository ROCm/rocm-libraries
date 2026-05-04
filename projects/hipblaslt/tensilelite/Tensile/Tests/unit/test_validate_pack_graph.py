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
    Pack -> MFMA gaps): now graph-side via `_cvt_to_mfma_gap_ok` (bead
    `35z`) and `_mfma_pack_to_cvt_gap_ok` (bead `or9`). Bead `8nz` deleted
    the structural-side mirror (`estimate_quad_cycles`,
    `Pack.min_quad_cycles_before_result_used`); coverage lives in
    `test_dataflow_graph_register_gaps.py`.
  * MFMA reorder + multi-tile + swap-pack tests: depend on Pack -> Pack
    pair ordering and CVT-pair timing, both blocked above.

Production-side deletions deferred until all 16 test classes have
graph-native equivalents (see deletion checklist in the bead's
"Deletion checklist" section). The legacy `test_ValidatePack.py` is
KEPT for now to preserve coverage during the incremental migration.
"""

from rocisa.container import vgpr
from rocisa.instruction import (
    PVCvtBF16toFP32,
    VCvtPkF32toBF16,
    VPermB32,
    VSubF32,
)

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    OrderInvertedFailure,
    MissingWaitFailure,
    WrongInterleavingFailure,
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


# =============================================================================
# MiddlePack pair-interleaving tests (bead `dpi`)
# =============================================================================
# Graph-native ports of the legacy
# `TestValidatePackTF32CrossPackInterleaving::test_failing_interleaved` and
# `TestValidatePackTF32MultipleGroups::test_failing_two_groups_fully_interleaved`
# in test_ValidatePack.py — both depend on the structural rule
# `_hook_up_packs_f32` + `MiddlePack.validate` in CMSValidator.py, which
# pairs middle-16 packs by adjacency in their category-local list (e.g.
# pair (0,1), (2,3), ... within PackA0) and asserts that the next
# MiddlePack in the GLOBAL stream after a pair-leader is its
# pair_consumer (the partner sharing its temp VGPR). The graph-side
# classifier `validate_middle_pack_pair_interleaving` (ScheduleCapture.py)
# walks the unified node stream once and emits WrongInterleavingFailure
# with the same shape the structural rule produces.
#
# The legacy tests stay on (parallel coverage) until ola.4 phase-2 deletes
# the structural rule; this file covers the graph-side classifier.


def _pack_middle_pvcvt(out_vgpr: int, src_vgpr: int, *, slot: int,
                       sequence: int = 0,
                       category: str = "PackA0") -> TaggedInstruction:
    """A MiddlePack-shaped instruction (PVCvtBF16toFP32 / `p_v_cvt_f32_bf16`).

    PVCvtBF16toFP32 is one of the four rocisa classes the production
    PACK_TYPE_MAP binds to the `MiddlePack` validator dataclass
    (CMSValidator.py:627). It carries the `dpi` classifier's pair-leader
    semantics if its category-local list places it at an even index. Here
    we use it as the leader of each pair; pair the `_pack_middle_vsubf32`
    helper below with it as the consumer for parity with the production
    `(v_cvt_f32_bf16, v_sub_f32)` pair shape — but the graph-side classifier
    only cares that BOTH halves are in `_MIDDLE_PACK_CLASS_NAMES`, not that
    they're a specific class pair.
    """
    inst = PVCvtBF16toFP32(dst=vgpr(out_vgpr, 1), src=vgpr(src_vgpr, 1))
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def _pack_middle_vsub(out_vgpr: int, src0_vgpr: int, src1_vgpr: int, *,
                     slot: int, sequence: int = 0,
                     category: str = "PackA0") -> TaggedInstruction:
    """A MiddlePack-shaped instruction (VSubF32 / `v_sub_f32`).

    The other half of a real production middle-16 pair (the
    error-term-subtraction step). Used here as the pair-CONSUMER while the
    PVCvtBF16toFP32 above acts as the LEADER. Either ordering would also
    fire the `dpi` classifier; the convention mirrors production stream
    order.
    """
    inst = VSubF32(dst=vgpr(out_vgpr, 1),
                   src0=vgpr(src0_vgpr, 1),
                   src1=vgpr(src1_vgpr, 1))
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


class TestMiddlePackPairInterleavingGraph(GraphNativeValidationTest):
    """Bead `dpi`: graph-native MiddlePack pair-interleaving classifier.

    Mirrors the legacy `MiddlePack.validate` semantics on the dataflow
    graph side: for each MiddlePack pair-leader (even-indexed within its
    category's middle-16 sub-stream), the next MiddlePack in the GLOBAL
    stream order must be the leader's pair-consumer (the next index in the
    same category). Otherwise emit `WrongInterleavingFailure(pack,
    expected_next, actual_next)`.

    Negative tests use unique category labels (PackA0 / PackB0) so the
    classifier's per-category pairing matches the production rule's
    per-group pairing in `_hook_up_middle_16_pairs`.
    """

    def test_middlepack_pair_contiguous_passing(self):
        """One PackA0 middle-16 pair (leader at slot 4, consumer at slot 5),
        feeding an MFMA at slot 8. No intervening MiddlePack. The classifier
        emits no failure.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            # Pair leader: PVCvtBF16toFP32 reading the LR's v8.
            _pack_middle_pvcvt(out_vgpr=40, src_vgpr=8, slot=4,
                               category="PackA0"),
            # Pair consumer: VSubF32 immediately after — back-to-back.
            _pack_middle_vsub(out_vgpr=41, src0_vgpr=40, src1_vgpr=9,
                              slot=5, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=8, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        # No WrongInterleavingFailure — pair is contiguous.
        for f in failures:
            assert not isinstance(f, WrongInterleavingFailure), (
                f"unexpected WrongInterleavingFailure: {f.format()}"
            )

    def test_middlepack_pair_interleaved_with_mfma_emits_wrong_interleaving(self):
        """PackA0 pair interleaved with a PackB0 MiddlePack: leader at slot 4,
        an OTHER middle-16 (PackB0) at slot 5, consumer at slot 6. The
        classifier fires WrongInterleavingFailure with pack=PackA0[leader],
        expected_next=PackA0[consumer], actual_next=PackB0[interloper].

        Note: the legacy test name says "interleaved with MFMA" but the
        actual structural-rule trigger is "another MiddlePack between the
        pair", because the pair-VGPR-share invariant is about pack-pack
        ordering, not pack-MFMA ordering. An intervening pure-MFMA does
        NOT violate the invariant (MFMAs do not contend for the temp VGPR).
        Mirrors `test_failing_interleaved` (PackB0[1] -> PackA0[2,2] ->
        PackB0[3]) — the issue is PackA0's middle-16 sitting between
        PackB0's pair leader and consumer.
        """
        # PackA0 pair: leader @ slot 4, would-be consumer @ slot 6.
        # PackB0 pair: leader @ slot 5 (between them), consumer @ slot 7.
        # Stream order of MiddlePacks: PackA0[L]@4, PackB0[L]@5, PackA0[C]@6, PackB0[C]@7.
        # PackA0[L]'s next MiddlePack in stream is PackB0[L], NOT PackA0[C] -> FAIL.
        # PackB0[L]'s next MiddlePack in stream is PackA0[C], NOT PackB0[C] -> FAIL.
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, lds_offset=64, slot=0, category="LRA0"),
            make_lr(20, 4, lds_offset=128, slot=1, category="LRB0"),
            make_swait(slot=2, dscnt=0),
            _pack_middle_pvcvt(out_vgpr=40, src_vgpr=8, slot=4,
                               category="PackA0"),  # PackA0 leader
            _pack_middle_pvcvt(out_vgpr=50, src_vgpr=20, slot=5,
                               category="PackB0"),  # interloper (PackB0 leader)
            _pack_middle_vsub(out_vgpr=41, src0_vgpr=40, src1_vgpr=9,
                              slot=6, category="PackA0"),  # PackA0 consumer
            _pack_middle_vsub(out_vgpr=51, src0_vgpr=50, src1_vgpr=21,
                              slot=7, category="PackB0"),  # PackB0 consumer
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        wrong_interleavings = [f for f in failures
                               if isinstance(f, WrongInterleavingFailure)]
        # Two violations: PackA0[L] and PackB0[L] both have a stranger as next.
        assert len(wrong_interleavings) == 2, (
            f"expected 2 WrongInterleavingFailures, got {len(wrong_interleavings)}: "
            + ", ".join(f.format() for f in wrong_interleavings)
        )
        # Verify the PackA0 leader's failure: leader@4, expected consumer@6,
        # actual next is PackB0 leader @ 5.
        f_a = next(f for f in wrong_interleavings if f.pack.category == "PackA0")
        assert f_a.pack.position.vmfma_index == 4
        assert f_a.expected_next.category == "PackA0"
        assert f_a.expected_next.position.vmfma_index == 6
        assert f_a.actual_next.category == "PackB0"
        assert f_a.actual_next.position.vmfma_index == 5

    def test_middlepack_two_groups_fully_interleaved(self):
        """Two pairs interleaved across categories — mirror of
        `test_failing_two_groups_fully_interleaved` in the legacy file.

        Stream: A0_L, B0_L, A0_C, B0_C. Both leaders see the wrong next.
        Two `WrongInterleavingFailure` (one per violating pair) expected.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=2, dscnt=0),
            _pack_middle_pvcvt(out_vgpr=40, src_vgpr=8, slot=4,
                               category="PackA0"),
            _pack_middle_pvcvt(out_vgpr=50, src_vgpr=20, slot=5,
                               category="PackB0"),
            _pack_middle_vsub(out_vgpr=41, src0_vgpr=40, src1_vgpr=9,
                              slot=6, category="PackA0"),
            _pack_middle_vsub(out_vgpr=51, src0_vgpr=50, src1_vgpr=21,
                              slot=7, category="PackB0"),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        wrong_interleavings = [f for f in failures
                               if isinstance(f, WrongInterleavingFailure)]
        assert len(wrong_interleavings) == 2, (
            f"expected 2 WrongInterleavingFailures, got {len(wrong_interleavings)}"
        )
        # Both leaders should have been flagged.
        flagged_categories = sorted(f.pack.category for f in wrong_interleavings)
        assert flagged_categories == ["PackA0", "PackB0"]

    def test_middlepack_non_pair_writes_no_false_positive(self):
        """A single MiddlePack in PackA0 (no sibling — odd-length category list)
        followed by an unrelated PackB0 MiddlePack. With only ONE PackA0 the
        category-local list has length 1 so no pair forms; the classifier
        must not fire. Same for PackB0.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=2, dscnt=0),
            # Single PackA0 MiddlePack (no pair).
            _pack_middle_pvcvt(out_vgpr=40, src_vgpr=8, slot=4,
                               category="PackA0"),
            # Single PackB0 MiddlePack (no pair).
            _pack_middle_pvcvt(out_vgpr=50, src_vgpr=20, slot=5,
                               category="PackB0"),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        for f in failures:
            assert not isinstance(f, WrongInterleavingFailure), (
                f"unexpected WrongInterleavingFailure on unpaired MiddlePacks: "
                f"{f.format()}"
            )
