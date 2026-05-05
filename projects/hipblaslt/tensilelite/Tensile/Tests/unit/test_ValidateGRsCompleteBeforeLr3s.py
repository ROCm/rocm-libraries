################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
"""Graph-native port of test_ValidateGRsCompleteBeforeLr3s.

The legacy file exercised the LR3 fallback in the structural rule
``set_gr_needed_by_from_lrs``: when LRA1/LRB1 are absent
(ForceUnrollSubIter mode), the rule's needed_by target switches from
LR1 to LR3. The structural rule has been removed.

Graph-side, the rule classification is unified: ``_collect_barrier_edges``
treats every category in ``{LRA0, LRA1, LRA3, LRB0, LRB1, LRB3}`` as a
valid LR consumer for the GR -> SWait(vlcnt) -> SBarrier -> LR pattern
(ScheduleCapture.py:2843). So the LR3 fallback is just "the LR consumer's
category happens to be LRA3/LRB3 instead of LRA1/LRB1" — same edge_kind,
same wait/barrier coverage rules. This file ports the LR1 negative tests
to assert the same Failures with LR3 categories, demonstrating the
unification.
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    MissingBarrierFailure,
    MissingWaitFailure,
    OrderInvertedFailure,
)

from dataflow_fixtures import (
    make_capture, make_gr, make_lr, make_sbarrier, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# Same register layout as the LR1 file. LR3 categories are the only
# difference (LRA3/LRB3 in place of LRA1/LRB1).
_GR_VGPR_BASE = 40
_GR_VGPR_COUNT = 4
_GR_SRD = 12
_GR_LDS_OFFSET = 64

_LR_VGPR_BASE = 8
_LR_VGPR_COUNT = 4
_LR_LDS_OFFSET = 64


def _gr(slot: int, category: str = "GRA", *, sequence: int = 0,
        vgpr_base: int = _GR_VGPR_BASE):
    return make_gr(
        vgpr_base, _GR_VGPR_COUNT, srd_sgpr_start=_GR_SRD,
        immediate_offset=_GR_LDS_OFFSET, slot=slot,
        category=category, sequence=sequence,
    )


def _lr3(slot: int, category: str = "LRA3", *, sequence: int = 0,
         vgpr_base: int = _LR_VGPR_BASE):
    """Single LR3 consumer at the given slot. LR3 is the
    ForceUnrollSubIter-mode fallback target for the GR -> LR pattern."""
    return make_lr(
        vgpr_base, _LR_VGPR_COUNT, lds_offset=_LR_LDS_OFFSET, slot=slot,
        category=category, sequence=sequence,
    )


# =============================================================================
# Positive tests — pattern correct, edge forms with LR3 consumer
# =============================================================================


class TestGRBeforeLR3_Positive(GraphNativeValidationTest):
    """Schedules with LRA3 / LRB3 consumers (ForceUnrollSubIter shape).
    Pattern collector emits gr_to_lr_lds_reuse edge for any LR* consumer."""

    def test_LR3_simple_case_success(self):
        """Canonical placement: GR @ 0, SWait+SBarrier @ 2, LR3 @ 7."""
        cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRB", vgpr_base=44),
            make_swait(slot=2, vlcnt=0),
            make_sbarrier(slot=2, sequence=1),
            _lr3(slot=7, category="LRA3"),
            _lr3(slot=7, category="LRB3", vgpr_base=12, sequence=1),
        ])
        subj = self.build_graph(self.wrap_single_body(cap))
        self.assert_edge_exists(subj, edge_kind="gr_to_lr_lds_reuse",
                                producer_category="GRA",
                                consumer_category="LRA3")
        failures = self.validate_waits(subj)
        self.assert_no_failures(failures)


# =============================================================================
# Negative tests — mirror the LR1 file with LRA3/LRB3 consumers
# =============================================================================


class TestGRBeforeLR3_Negatives(GraphNativeValidationTest):
    """Each method mirrors a TestValidateGRsCompleteBeforeLr1s negative
    test with LR3 consumers, demonstrating the unified
    gr_to_lr_lds_reuse coverage."""

    def _ref(self):
        """Reference: full GR -> SWait(vlcnt=0) -> SBarrier -> LR3 chain."""
        return make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=2, vlcnt=0),
            make_sbarrier(slot=2, sequence=1),
            _lr3(slot=7, category="LRA3"),
        ])

    def test_LR3_grs_swait_on_wrong_counter(self):
        """SWait fires with dscnt=0 instead of vlcnt -> WaitOnWrongCounter."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=2, dscnt=0),    # wrong counter
            make_sbarrier(slot=2, sequence=1),
            _lr3(slot=7, category="LRA3"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure,
            counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA3"
        assert len(f.nearby_other_counter_waits) >= 1

    def test_LR3_grs_no_swait_at_all(self):
        """No SWait of any kind in the window -> MissingWait(vlcnt)."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_sbarrier(slot=2),
            _lr3(slot=7, category="LRA3"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA3"

    def test_LR3_no_sbarrier(self):
        """SBarrier omitted -> MissingBarrier(role='needed_by')."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=2, vlcnt=0),
            _lr3(slot=7, category="LRA3"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingBarrierFailure, role="needed_by",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA3"

    def test_LR3_swait_after_sbarrier(self):
        """SBarrier sits BEFORE the SWait -> required ordering inverted ->
        MissingBarrier classification (wait drains, no barrier follows)."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_sbarrier(slot=1),                  # barrier BEFORE wait
            make_swait(slot=2, vlcnt=0),
            _lr3(slot=7, category="LRA3"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        self.assert_failures_contain(
            failures, cls=MissingBarrierFailure, role="needed_by",
        )

    def test_LR3_guaranteed_after_first_lr3(self):
        """4 GRs in flight; SWait+SBarrier at slot 4 land AFTER the FIRST
        LRA3 consumer at slot 3 — wait outside [GR, LRA3) window for that
        consumer, so the gr_to_lr_lds_reuse edge is missing in subj.
        diagnose_missing_edge classifies as MissingWait(vlcnt)."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRA", sequence=1),
            _gr(slot=0, category="GRA", sequence=2),
            _gr(slot=0, category="GRA", sequence=3),
            make_swait(slot=2, vlcnt=4),
            make_sbarrier(slot=2, sequence=1),
            _lr3(slot=7, category="LRA3"),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRA", sequence=1),
            _gr(slot=0, category="GRA", sequence=2),
            _gr(slot=0, category="GRA", sequence=3),
            _lr3(slot=3, category="LRA3"),
            make_swait(slot=4, vlcnt=4),    # AFTER LRA3 consumer
            make_sbarrier(slot=4, sequence=1),
        ])
        failures = self.compare(
            self.wrap_single_body(ref_cap),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA3"

    def test_LR3_swap_global_read_order_failure(self):
        """SwapGlobalReadOrder=True LR3 variant. GRB at slot=3 sits AFTER
        LRA3 at slot=2 in subj -> OrderInverted on the cross-graph
        GRB -> LRA3 edge."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRB", vgpr_base=44),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=1, sequence=1),
            _lr3(slot=2, category="LRB3", vgpr_base=12),
            _lr3(slot=5, category="LRA3"),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=1, sequence=1),
            _lr3(slot=2, category="LRA3"),
            _gr(slot=3, category="GRB", vgpr_base=44),
            make_swait(slot=4, vlcnt=0),
            make_sbarrier(slot=4, sequence=1),
            _lr3(slot=5, category="LRB3", vgpr_base=12),
        ])
        failures = self.compare(
            self.wrap_single_body(ref_cap),
            self.wrap_single_body(subj_cap),
            raise_on_unexplained=False,
        )
        f = self.assert_failures_contain(failures, cls=OrderInvertedFailure)
        assert f.producer.category in {"GRA", "GRB"}
        assert f.consumer.category in {"LRA3", "LRB3"}
