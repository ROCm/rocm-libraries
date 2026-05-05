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
"""Graph-native ports of test_ValidatePack.py — bead ola.4 + bead wao.

Sub-task ola.4 of the CMS validation migration epic (`br show
rocm-libraries-ola`): replace the structural rule `add_pack_constraints`
in CMSValidator.py with graph-side equivalents. Bead wao closes out the
final 5 legacy classes whose tests had no obvious graph-native twin in
the deletion commit (`afb69530da`).

Scope of this file:
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
  * MiddlePack pair-interleaving (`WrongInterleavingFailure`): the f32
    pair-consumer ordering invariant in `_hook_up_packs_f32` is mirrored
    graph-side by `validate_middle_pack_pair_interleaving` (bead `dpi`).
  * Bead `wao` additions: BF16 LR1 -> Pack1 (v_perm flavor), Pack/MFMA
    same-slot boundary, and the VSwap-Pack class — VSwap reading LR-output
    and Pack-before-Swap reorder. See `TestSwapPackGraph` below.

# =============================================================================
# Migration table — test_ValidatePack.py (deleted afb69530da)
# =============================================================================
# Legend: PORTED (covered by a new graph-native test in this file or
#         test_dataflow_graph_register_gaps.py),
#         REDUNDANT (already covered by an existing graph-native test —
#         entry names that test),
#         OBSOLETE (the underlying scenario has no meaning in the
#         graph-native model: the structural validator ran kernel-flag
#         arithmetic the graph doesn't replicate).
#
# Bead wao: TestValidatePackBF16PLRPack (5 tests)
#   test_passing_plr_pack
#       -> TestLRAfterPack_BF16Graph::test_BF16_pack_after_LR_baseline_passing
#                                                        [REDUNDANT]
#   test_passing_plr_pack_different_number_LRs
#       -> n/a   [OBSOLETE: LR-count variation has no graph meaning;
#                 graph tracks register identities, not counts]
#   test_fail_too_early_plr_pack
#       -> TestLRAfterPack_BF16_LR1Graph::test_BF16_LR1_pack_before_LR_swait_too_late
#                                                        [PORTED in wao]
#   test_fail_too_early_more_lrs_plr_pack
#       -> n/a   [OBSOLETE: LR-count variation, same reason as above]
#   test_fail_too_early_less_lrs_plr_pack
#       -> n/a   [OBSOLETE: LR-count variation, same reason as above]
#
# Bead wao: TestValidatePackTF32MFMAReorder (2 tests)
#   test_passing
#       -> TestLRAfterPackGraph::test_LR0_baseline_passing
#                                                        [REDUNDANT]
#   test_failing_too_early
#       -> TestLRAfterPackGraph::test_LR0_A_LR_after_Pack
#                                                        [REDUNDANT:
#         the mfmaReorder kernel parameter only feeds the structural
#         validator's MFMA-position math; LR -> Pack RAW edge formation
#         is independent of MFMA reorder.]
#
# Bead wao: TestValidatePackTF32MFMA4x4x4MultipleTiles (4 tests)
#   test_passing_multiple_tiles_different_timings
#       -> TestLRAfterPackGraph::test_LR0_baseline_passing
#                                                        [REDUNDANT]
#   test_passing_tile1_packs_after_tile0_deadline
#       -> n/a   [OBSOLETE: tile-offset arithmetic is structural-validator
#                 specific; graph tracks per-register dependencies that
#                 inherently honor tile boundaries.]
#   test_failing_tile1_packs_actually_too_late
#       -> TestPackAfterMFMASameSlotGraph::test_pack_same_slot_as_mfma_orderinverted
#                                                        [PORTED in wao:
#         the boundary case Pack@N and MFMA@N at SAME mfma_index, Pack
#         appearing AFTER MFMA in stream order. Existing graph test only
#         covers the strict Pack@N+1 / MFMA@N case.]
#   test_broken_pack_schedule_with_mfma_reorder
#       -> TestPackAfterMFMAGraph::test_LR0_Pack_after_MFMA
#                                                        [REDUNDANT:
#         the underlying issue (Pack issued after consumer MFMA) is
#         already covered. The legacy test asserts on isValid()'s exact
#         error message string, which is structural-validator-specific.]
#
# Bead wao: TestValidatePackTF32MFMA4x4x4SwapPacks (6 tests)
#   test_passing_vw_combinations
#       -> n/a   [OBSOLETE: VectorWidth kernel parameter does not affect
#                 graph-native edge formation; per-register identity is
#                 sufficient.]
#   test_passing_multiple_groups_with_swaps
#       -> n/a   [OBSOLETE: same reason — multi-group/VW configuration
#                 has no graph-native counterpart.]
#   test_fail_swap_before_lr_done
#       -> TestSwapPackGraph::test_swap_lr_raw_swait_after_swap
#                                                        [PORTED in wao]
#   test_fail_regular_pack_before_swap_done
#       -> TestSwapPackGraph::test_pack_before_swap_orderinverted
#                                                        [PORTED in wao]
#   test_swap_depends_on_specific_lrs_vw4
#       -> n/a   [OBSOLETE: relies on production-side per-LR-index swap
#                 mapping (`_logical_reg_to_lr_index`); graph-native
#                 model uses real register identities, so per-LR partial
#                 SWaitCnt coverage is exercised by the existing dscnt
#                 wait-coverage tests in test_dataflow_graph_register_gaps
#                 .py without needing VW=4-specific construction.]
#   test_cvt0_depends_on_specific_swaps_vw4
#       -> n/a   [OBSOLETE: same — per-group swap-mapping logic is
#                 structural-validator-specific.]
#
# TestValidatePackTF32MFMA4x4x4::test_passing_snop
#       -> test_dataflow_graph_register_gaps.py::TestCumulativeIssueCycles::
#          test_snop_wait_state_dominates_4x4_mfma_finish (and siblings)
#                                                        [REDUNDANT:
#         restored coverage in the validator-branch yh0 commit
#         (8c945ed8b5) — SNop wait_state propagation is graph-native.]
#
# Production-side `add_pack_constraints` deletion is in scope for a later
# bead (see `br show rocm-libraries-ola.4` "Deletion checklist").
"""

from rocisa.container import vgpr
from rocisa.instruction import (
    PVCvtBF16toFP32,
    VCvtPkF32toBF16,
    VPermB32,
    VSubF32,
    VSwapB32,
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
# Original tests asserted on a now-deleted single-schedule constraint
# Failure (Pack rule retired in `ola.4` phase 2; the dataclass was
# deleted in `pcz`). Graph-native equivalent: `compare_graphs` emits
# OrderInvertedFailure when subj's Pack is positioned BEFORE subj's LR
# (variant A); `validate_edge_wait_coverage` emits
# MissingWaitFailure(counter_kind="vlcnt") when the LR's drain SWait
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

        Bead `2bu.4` switched `_cvt_to_mfma_gap_ok` to the cycle-exact
        `cumulative_issue_cycles` walk, which only counts instructions
        actually present in the captured stream (not slot-index gaps).
        Two intervening LRs (cost 1 each) populate the captured stream
        between Pack and MFMA so the cycle-exact walk computes
        `gap = CVT(1) + LR(1) + LR(1) - 1 = 2`, exactly meeting the
        threshold. Pre-`2bu.4` the slot-delta arithmetic gave gap=2 from
        the slot-index gap alone (slot_delta=3 → 2); the cycle-exact
        walk requires real intervening instructions.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_lr(60, 1, lds_offset=80, slot=4, category="LRA1"),
            make_lr(61, 1, lds_offset=84, slot=5, category="LRA1"),
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
# Original tests asserted on a now-deleted single-schedule constraint
# Failure (deleted in `pcz`). Graph-native equivalent: `compare_graphs`
# Phase-1 order check fires when subj's Pack.position > MFMA.position
# (OrderInvertedFailure). ALU producers have no wait-coverage requirement,
# so no wait-coverage failure follows.


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

    Note: original LR1 test asserted the LR1 producer / Pack1 consumer
    pairing fires "issued too early" via a now-deleted single-schedule
    constraint Failure (deleted in `pcz`). The graph-native equivalent
    uses Pack VCvt with subiter index 1 in its category (PackA1) and
    LRA1 (subiter 1), so the within-subiter order check applies.
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
                f"unexpected WrongInterleavingFailure: {repr(f)}"
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
            + ", ".join(repr(f) for f in wrong_interleavings)
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
                f"{repr(f)}"
            )


# =============================================================================
# Bead `wao` ports — see migration table in module docstring.
# =============================================================================
# Each test below has a docstring naming the legacy test it ports.
# Tests are alive (no xfail/skip) and use real `TaggedInstruction`s wrapping
# rocisa instruction classes through the existing `dataflow_fixtures` and
# `_pack_*` helpers above.


class TestLRAfterPack_BF16_LR1Graph(GraphNativeValidationTest):
    """Bead `wao`: BF16 LR1 -> Pack1 (v_perm flavor) coverage.

    Existing graph-native coverage uses VCvt-style packs for the LR1 case
    (`TestLRAfterPack_LR1Graph`) and VPerm-style packs only for LR0
    (`TestLRAfterPack_BF16Graph`). The legacy `TestValidatePackBF16PLRPack`
    runs UsePLRPack=True so PackA1/PackB1 are inside the loop and pinned
    on the LR1 -> Pack1 edge. This class covers the cross-product
    explicitly so a future regression in the v_perm-flavored Pack edge
    formation for subiter-1 categories cannot escape the suite.
    """

    def test_BF16_LR1_pack_before_LR_swait_too_late(self):
        """Ports test_ValidatePack.py::TestValidatePackBF16PLRPack::
        test_fail_too_early_plr_pack.

        Legacy test pinned PackA1[5] issued before SWaitCnt for LRA1[4]
        (SWaitCnt at idx=6) using the BF16 v_perm Pack flavor. Graph-native
        equivalent: an LRA1 -> PackA1 vlcnt edge whose drain SWait sits
        AFTER the Pack consumer fires `MissingWaitFailure`.
        """
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 4, lds_offset=128, slot=0, category="LRA1"),
            _pack_vperm(out_vgpr=60, lr_vgpr=50, slot=1, category="PackA1"),
            # SWait sits AFTER Pack — doesn't cover the LRA1 -> PackA1 edge.
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=60, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(subj_cap)))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )

    def test_BF16_LR1_pack_after_LR_baseline_passing(self):
        """Positive baseline for the LRA1 -> PackA1 v_perm edge: SWait sits
        BETWEEN the LR and the Pack so the dscnt drain covers the window.
        Mirror of `test_passing_plr_pack` for the LR1 v_perm slice
        (the LR0/v_perm side is already covered by
        `TestLRAfterPack_BF16Graph::test_BF16_pack_after_LR_baseline_passing`).
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(50, 4, lds_offset=128, slot=0, category="LRA1"),
            make_swait(slot=2, dscnt=0),
            _pack_vperm(out_vgpr=60, lr_vgpr=50, slot=3, category="PackA1"),
            make_mfma(c_dst_start=0, a_src_start=60, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)


class TestPackAfterMFMASameSlotGraph(GraphNativeValidationTest):
    """Bead `wao`: Pack/MFMA same-slot boundary coverage.

    Existing graph-native test (`TestPackAfterMFMAGraph::test_LR0_Pack_after_MFMA`)
    verifies the strictly-after case (Pack at slot 5, MFMA at slot 4).
    The boundary case where Pack and MFMA share an mfma_index but Pack
    appears AFTER MFMA in stream order (sequence 1 vs sequence 0) was
    untested. The legacy
    `TestValidatePackTF32MFMA4x4x4MultipleTiles::test_failing_tile1_packs_actually_too_late`
    pinned exactly this — `producer_idx=9, consumer_idx=9`. Locking it
    in here guards the sub_index/sequence-based tie-break in
    `compare_graphs` Phase-1 against future regressions.
    """

    def test_pack_same_slot_as_mfma_orderinverted(self):
        """Ports test_ValidatePack.py::TestValidatePackTF32MFMA4x4x4MultipleTiles::
        test_failing_tile1_packs_actually_too_late.

        Subj places PackA0 at the SAME mfma_index as the consuming MFMA
        but with sequence=1 (after the MFMA in stream order). The graph
        ref places Pack one slot earlier. compare_graphs Phase-1 fires
        `OrderInvertedFailure(producer=PackA0, consumer=MFMA)` because
        the producer's effective position now exceeds the consumer's.
        """
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=3, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        # Subject: Pack at slot 4 sequence=1 — SAME mfma_index as MFMA but
        # appearing AFTER it in stream order.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=4, a_src_count=1),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=4, sequence=1,
                       category="PackA0"),
        ])
        failures = self.compare(self.wrap_single_body(ref_cap),
                                self.wrap_single_body(subj_cap))
        f = self.assert_failures_contain(failures, cls=OrderInvertedFailure)
        # Pin identity: the Pack -> MFMA RAW edge is the one that flipped.
        assert f.producer.category == "PackA0"
        assert f.consumer.category == "MFMA"


# =============================================================================
# VSwapB32 helper for SwapPack tests
# =============================================================================
# Real schedules introduce VSwapB32 instructions at the head of a Pack
# group when LocalReadVectorWidth > 1 and VectorWidth > 1: the wider LR
# loads place A and B halves in interleaved registers, and the swaps
# transpose them before the regular VCvt packs run. The swaps belong to
# the same PackA0 / PackB0 stream as the VCvt packs, so they share the
# Pack category in the dataflow graph.
#
# `_VSwapRule` (ScheduleCapture.py:1970) publishes BOTH operands as
# reads AND writes (symmetric R+W) so reorderings of swap pairs produce
# distinguishable edge keys in `compare_graphs`. For the tests below we
# only need ONE swap reading an LR-output vgpr (LR -> Swap RAW edge) and
# one swap whose write is consumed by a later VCvt Pack (Swap -> Pack
# RAW edge).


def _swap_pack(dst_vgpr: int, src_vgpr: int, *, slot: int,
               sequence: int = 0,
               category: str = "PackA0") -> TaggedInstruction:
    """A VSwapB32 wrapped as a Pack-category TaggedInstruction.

    `v_swap_b32 dst, src` swaps the two registers — both are read AND
    both are written. `_VSwapRule` claims it ahead of the asymmetric
    `_GenericALURule` so the dataflow graph carries the correct
    bidirectional edge set.

    `dst_vgpr` should be the register the swap WRITES that a later Pack
    or MFMA consumer needs to depend on; `src_vgpr` is the partner the
    swap also touches. Tests can wire `dst_vgpr` to overlap an LR's
    output vgpr to form an LR -> Swap RAW edge.
    """
    inst = VSwapB32(dst=vgpr(dst_vgpr, 1), src=vgpr(src_vgpr, 1))
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


class TestSwapPackGraph(GraphNativeValidationTest):
    """Bead `wao`: VSwap-Pack coverage.

    Two ports from `TestValidatePackTF32MFMA4x4x4SwapPacks`:

      * Swap reads LR-output vgpr; SWaitCnt covering the LR drain sits
        AFTER the swap (legacy `test_fail_swap_before_lr_done`).
      * Regular VCvt Pack reads a register that an earlier swap wrote;
        the Pack is reordered to issue BEFORE the swap (legacy
        `test_fail_regular_pack_before_swap_done`).

    Other tests in the legacy class (`test_passing_vw_combinations`,
    `test_passing_multiple_groups_with_swaps`, `test_swap_depends_on_*_vw4`,
    `test_cvt0_depends_on_*_vw4`) are dispositioned OBSOLETE in the
    migration table above — they exercise structural-validator-side
    VW/group/per-LR-index arithmetic that the graph-native model
    short-circuits via real register identity tracking.
    """

    def test_swap_lr_raw_swait_after_swap(self):
        """Ports test_ValidatePack.py::TestValidatePackTF32MFMA4x4x4SwapPacks::
        test_fail_swap_before_lr_done.

        Legacy test pinned `producer=LRA0@0, consumer=PackA0@0` — the
        SwapPack at idx 0 (categorized as PackA0) reads an LRA0-output
        vgpr but the SWaitCnt covering LRA0's dscnt drain sits after the
        swap. Graph-native equivalent: VSwapB32's symmetric R+W publishes
        a read of the LR-output vgpr, the LR -> Swap edge has no covering
        SWait in its window, `validate_edge_wait_coverage` emits
        `MissingWaitFailure(counter_kind="dscnt")`.
        """
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            # Swap touches v8 — reads LR's output before the SWait drain.
            _swap_pack(dst_vgpr=8, src_vgpr=16, slot=1, category="PackA0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=4, a_src_count=2),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(subj_cap)))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )

    def test_swap_lr_baseline_passing(self):
        """Positive baseline: SWait sits BETWEEN the LR and the swap so the
        dscnt drain covers the LR -> Swap edge. Mirror of
        `test_passing_vw_combinations` shape, but identity-driven and
        kernel-flag-free.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            _swap_pack(dst_vgpr=8, src_vgpr=16, slot=2, category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=4, a_src_count=2),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)

    def test_pack_before_swap_orderinverted(self):
        """Ports test_ValidatePack.py::TestValidatePackTF32MFMA4x4x4SwapPacks::
        test_fail_regular_pack_before_swap_done.

        Legacy test pinned `producer=PackA0(swap)@2, consumer=PackA0(cvt)@1`
        — the regular CVT Pack at idx 1 reads vgpr v8 but the SwapPack at
        idx 2 (intended to write v8 first) is scheduled later, so the CVT
        Pack reads stale data. Graph-native equivalent: in the reference
        graph the Swap -> CVT-Pack RAW edge has Swap before Pack; in the
        subject graph the Pack appears before the Swap, so the Phase-1
        order check fires `OrderInvertedFailure(producer=PackA0(swap),
        consumer=PackA0(pack))`.
        """
        # Reference: Swap @ slot 2 writes v8, then VCvt Pack @ slot 4 reads v8.
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            _swap_pack(dst_vgpr=8, src_vgpr=16, slot=2, category="PackA0"),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=4, sequence=1,
                       category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=5, a_src_count=1),
        ])
        # Subject: VCvt Pack @ slot 1 reads v8 BEFORE the Swap @ slot 3.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 2, lds_offset=64, slot=0, category="LRA0"),
            _pack_vcvt(out_vgpr=40, lr_vgpr=8, slot=1, category="PackA0"),
            make_swait(slot=2, dscnt=0),
            _swap_pack(dst_vgpr=8, src_vgpr=16, slot=3, sequence=1,
                       category="PackA0"),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=5, a_src_count=1),
        ])
        failures = self.compare(self.wrap_single_body(ref_cap),
                                self.wrap_single_body(subj_cap))
        # The reorder destroys the Swap -> CVT-Pack edge — Phase-1 sees
        # Pack(consumer) before Swap(producer) in subj.
        f = self.assert_failures_contain(failures, cls=OrderInvertedFailure)
        # Both the producer and consumer are PackA0-category nodes; the
        # producer is the swap (writes v8) and the consumer is the VCvt
        # (reads v8) — but the legacy test pinned the categories alone,
        # not the rocisa class, so we mirror that.
        assert f.producer.category == "PackA0"
        assert f.consumer.category == "PackA0"
