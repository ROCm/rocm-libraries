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
"""Self-test for ``graph_native_validation_base.GraphNativeValidationTest``.

Exercises every helper against synthetic captures so the migration agents
(ola.1..ola.4) inherit a base class proven to work end-to-end before any
real migration consumes it.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    MissingBarrierFailure,
    MissingWaitFailure,
    OrderInvertedFailure,
    SCCConflictFailure,
    SchedulePosition,
    TimingTooCloseFailure,
    WaitInsufficientFailure,
    WrongInterleavingFailure,
)

from dataflow_fixtures import (
    make_capture, make_gr, make_lr, make_mfma, make_sbarrier, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# =============================================================================
# Capture wrapping & graph construction
# =============================================================================


class TestWrapAndBuild(GraphNativeValidationTest):
    def test_wrap_single_body_default_fillers(self):
        """Wrapping a single ML body produces a four-body FourPartCapture
        whose ``main_loop[0]`` is the input and whose three other bodies are
        single-MFMA fillers in non-overlapping high-vgpr ranges."""
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        cap = self.wrap_single_body(ml_cap)
        assert cap.main_loop == {0: ml_cap}
        assert cap.num_codepaths == 1
        assert cap.source == "cms"
        # Filler bodies exist and are non-empty.
        assert 0 in cap.main_loop_prev
        assert 0 in cap.n_gl
        assert 0 in cap.n_ll
        assert len(cap.main_loop_prev[0].instructions) >= 1
        assert len(cap.n_gl[0].instructions) >= 1
        assert len(cap.n_ll[0].instructions) >= 1

    def test_build_graph_returns_dataflow_graph(self):
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        graph = self.build_graph(self.wrap_single_body(ml_cap))
        # DataflowGraph carries .nodes and .edges
        assert hasattr(graph, "nodes")
        assert hasattr(graph, "edges")
        assert len(graph.edges) >= 1

    def test_compare_returns_empty_for_identical_graphs(self):
        """compare() on two identical captures returns an empty failure list."""
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        ref_cap = self.wrap_single_body(ml_cap)
        # Build a second identical ML capture to avoid object-identity reuse.
        ml_cap2 = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj_cap = self.wrap_single_body(ml_cap2)
        failures = self.compare(ref_cap, subj_cap)
        self.assert_no_failures(failures)


# =============================================================================
# Failure-list assertions
# =============================================================================


class TestAssertFailure(GraphNativeValidationTest):
    def _missing_swait_failures(self):
        """Build a capture whose LR -> MFMA edge has no covering SWait, so
        validate_waits returns a MissingWaitFailure."""
        subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ]))
        return self.validate_waits(self.build_graph(subj_cap))

    def test_assert_failure_matches_class_only(self):
        """assert_failure(failures, cls=Cls) returns the matching failure."""
        failures = self._missing_swait_failures()
        # Only one MissingWaitFailure should be present (one missing edge).
        f = self.assert_failure(failures, cls=MissingWaitFailure)
        assert isinstance(f, MissingWaitFailure)

    def test_assert_failure_matches_class_and_field(self):
        """assert_failure with field constraints returns only the matching one."""
        failures = self._missing_swait_failures()
        f = self.assert_failure(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )
        assert f.counter_kind == "dscnt"

    def test_assert_failure_class_alias(self):
        """``cls_`` is accepted as an alias for ``cls``."""
        failures = self._missing_swait_failures()
        f = self.assert_failure(failures, cls_=MissingWaitFailure)
        assert isinstance(f, MissingWaitFailure)

    def test_assert_failure_no_match_raises(self):
        failures = self._missing_swait_failures()
        with pytest.raises(AssertionError, match="expected exactly 1 SCCConflictFailure"):
            self.assert_failure(failures, cls=SCCConflictFailure)

    def test_assert_failure_field_mismatch_raises(self):
        failures = self._missing_swait_failures()
        with pytest.raises(AssertionError, match="expected exactly 1 MissingWaitFailure"):
            self.assert_failure(
                failures, cls=MissingWaitFailure, counter_kind="vlcnt",
            )

    def test_assert_failure_missing_cls_raises(self):
        with pytest.raises(TypeError, match="missing required keyword 'cls'"):
            self.assert_failure([])

    def test_assert_failure_unknown_field_raises(self):
        """Unknown attributes count as a non-match (filter, not error)."""
        failures = self._missing_swait_failures()
        with pytest.raises(AssertionError, match="found 0"):
            self.assert_failure(
                failures, cls=MissingWaitFailure, nonexistent_attr="x",
            )

    def test_assert_failures_contain_finds_match(self):
        failures = self._missing_swait_failures()
        f = self.assert_failures_contain(failures, cls=MissingWaitFailure)
        assert isinstance(f, MissingWaitFailure)

    def test_assert_failures_contain_no_match_raises(self):
        failures = self._missing_swait_failures()
        with pytest.raises(AssertionError, match="no SCCConflictFailure"):
            self.assert_failures_contain(failures, cls=SCCConflictFailure)

    def test_assert_no_failures_passes_on_empty(self):
        self.assert_no_failures([])

    def test_assert_no_failures_raises_on_nonempty(self):
        failures = self._missing_swait_failures()
        with pytest.raises(AssertionError, match="expected empty failure list"):
            self.assert_no_failures(failures)


# =============================================================================
# Edge-set assertions
# =============================================================================


class TestEdgeAssertions(GraphNativeValidationTest):
    def _build_lr_mfma_graph(self):
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ]))
        return self.build_graph(cap)

    def test_assert_edge_exists_match_kind_only(self):
        graph = self._build_lr_mfma_graph()
        edge = self.assert_edge_exists(graph, edge_kind="raw_intrawave")
        assert edge.edge_kind == "raw_intrawave"

    def test_assert_edge_exists_match_categories(self):
        graph = self._build_lr_mfma_graph()
        edge = self.assert_edge_exists(
            graph,
            edge_kind="raw_intrawave",
            producer_category="LRA0",
            consumer_category="MFMA",
        )
        assert edge.producer.category == "LRA0"
        assert edge.consumer.category == "MFMA"

    def test_assert_edge_exists_no_match_raises(self):
        graph = self._build_lr_mfma_graph()
        with pytest.raises(AssertionError, match="no edge matching"):
            self.assert_edge_exists(graph, edge_kind="lr_to_gr_lds_reuse")

    def test_assert_edge_absent_passes_when_no_match(self):
        graph = self._build_lr_mfma_graph()
        # No GR in this capture -> no LR-to-GR LDS-reuse edge.
        self.assert_edge_absent(graph, edge_kind="lr_to_gr_lds_reuse")

    def test_assert_edge_absent_raises_when_match_exists(self):
        graph = self._build_lr_mfma_graph()
        with pytest.raises(AssertionError, match="found unexpected edge"):
            self.assert_edge_absent(graph, edge_kind="raw_intrawave")


# =============================================================================
# Wait-coverage validator integration
# =============================================================================


class TestValidateWaits(GraphNativeValidationTest):
    def test_clean_schedule_no_failures(self):
        """A valid LR -> SWait(dscnt=0) -> MFMA produces no failures."""
        graph = self.build_graph(self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])))
        self.assert_no_failures(self.validate_waits(graph))

    def test_wait_insufficient(self):
        """5 LRs in flight, SWait(dscnt=2) only drains 3 -> insufficient
        for the youngest (consumed by MFMA)."""
        graph = self.build_graph(self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
            make_lr(16, 4, 96, slot=2, category="LRA0"),
            make_lr(20, 4, 112, slot=3, category="LRA0"),
            make_lr(24, 4, 128, slot=4, category="LRA0"),
            make_swait(slot=5, dscnt=2),
            make_mfma(0, 24, 32, slot=6, a_src_count=4),
        ])))
        failures = self.validate_waits(graph)
        self.assert_failures_contain(failures, cls=WaitInsufficientFailure)


# =============================================================================
# Cross-graph comparison via compare()
# =============================================================================


class TestCompareGraphs(GraphNativeValidationTest):
    def test_missing_barrier_diagnosed(self):
        """Reference has LR0 -> SWait -> SBarrier -> GR; subject deletes
        the barrier -> compare() returns MissingBarrierFailure."""
        ref_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_sbarrier(slot=2),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ]))
        subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            # SBarrier deleted
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ]))
        failures = self.compare(ref_cap, subj_cap)
        f = self.assert_failures_contain(failures, cls=MissingBarrierFailure)
        assert f.role == "must_start_after"


# =============================================================================
# Lifted typed-Failure helpers — exercise each against a synthetic Failure
# =============================================================================
# These test the helpers, not the Failure-emitting rules. We construct
# Failure instances directly with stand-in node objects so the helpers'
# field assertions are exercised end-to-end.


class _FakeNode:
    """Minimal stand-in for GraphNode / ValidatorInstruction with .name and
    .issued_at.vmfma_index — the only attributes the lifted helpers read."""
    def __init__(self, name, vmfma_index, *, use_position=False):
        self.name = name
        if use_position:
            # GraphNode-style: only .position
            self.position = SchedulePosition(loop_index=1, vmfma_index=vmfma_index, sub_index=0)
        else:
            # ValidatorInstruction-style: .issued_at
            self.issued_at = SchedulePosition(loop_index=1, vmfma_index=vmfma_index, sub_index=0)


class TestLiftedHelpers(GraphNativeValidationTest):
    def test_assert_order_inverted_with_issued_at(self):
        """ValidatorInstruction-style nodes (.issued_at)."""
        f = OrderInvertedFailure(
            producer=_FakeNode("LRA0", -1),
            consumer=_FakeNode("PackA0", 2),
            default_producer_position=SchedulePosition(1, -1, 0),
            default_consumer_position=SchedulePosition(1, 2, 0),
        )
        self.assert_order_inverted(
            f, producer_name="LRA0", producer_idx=-1,
            consumer_name="PackA0", consumer_idx=2,
        )

    def test_assert_order_inverted_with_position_attribute(self):
        """GraphNode-style nodes (.position only) — used by graph-native
        Failures emitted by validate_edge_wait_coverage."""
        f = OrderInvertedFailure(
            producer=_FakeNode("GRA", 0, use_position=True),
            consumer=_FakeNode("LRA1", 6, use_position=True),
            default_producer_position=SchedulePosition(1, 0, 0),
            default_consumer_position=SchedulePosition(1, 6, 0),
        )
        self.assert_order_inverted(
            f, producer_name="GRA", producer_idx=0,
            consumer_name="LRA1", consumer_idx=6,
        )

    def test_assert_order_inverted_name_mismatch_raises(self):
        f = OrderInvertedFailure(
            producer=_FakeNode("LRA0", 0),
            consumer=_FakeNode("MFMA", 4),
            default_producer_position=SchedulePosition(1, 0, 0),
            default_consumer_position=SchedulePosition(1, 4, 0),
        )
        with pytest.raises(AssertionError, match="producer.name"):
            self.assert_order_inverted(
                f, producer_name="WRONG", producer_idx=0,
                consumer_name="MFMA", consumer_idx=4,
            )

    def test_assert_timing_too_close(self):
        f = TimingTooCloseFailure(
            producer=_FakeNode("PackA0", 4),
            consumer=_FakeNode("MFMA", 4),
            expected_quad_cycles=2,
            actual_quad_cycles=1,
        )
        self.assert_timing_too_close(
            f, producer_name="PackA0", producer_idx=4,
            consumer_name="MFMA", consumer_idx=4,
            expected_quad_cycles=2, actual_quad_cycles=1,
        )

    def test_assert_wrong_interleaving(self):
        f = WrongInterleavingFailure(
            pack=_FakeNode("PackA0", 0),
            expected_next=_FakeNode("PackB0", 1),
            actual_next=_FakeNode("PackA1", 2),
        )
        self.assert_wrong_interleaving(
            f, pack_name="PackA0", pack_idx=0,
            expected_next_name="PackB0", expected_next_idx=1,
            actual_next_name="PackA1", actual_next_idx=2,
        )
