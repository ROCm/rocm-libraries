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
"""Graph-native CMS validation test base.

Replacement for ``cms_validation_base.CMSValidationTestBase``. Where the
legacy base built a Timeline + ran structural-rule passes + asserted on
``timeline._last_failure``, this base operates entirely on the dataflow
graph: tests build a ``FourPartCapture`` with the existing
``dataflow_fixtures`` helpers, run ``build_dataflow_graph`` +
``compare_graphs`` / ``validate_edge_wait_coverage``, and assert on the
returned ``Failure`` list.

Migration path for ``ola.1``..``ola.4``:

  class TestGRFinishBeforeLR(GraphNativeValidationTest):
      def test_negative_missing_swait(self):
          subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
              make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                      slot=0, category="GRA"),
              # SWait deleted -> coverage failure on gr_to_lr_lds_reuse edge
              make_sbarrier(slot=2),
              make_lr(40, 4, 64, slot=3, category="LRA1"),
          ]))
          subj = self.build_graph(subj_cap)
          failures = validate_edge_wait_coverage(subj)
          self.assert_failure(failures, cls=MissingWaitFailure,
                              counter_kind="vlcnt")

The legacy ``assert_order_inverted`` / ``assert_timing_too_close`` /
``assert_scc_conflict`` / ``assert_out_of_order_sequence`` /
``assert_wrong_interleaving`` helpers are lifted onto this base unchanged
because they assert on typed Failure fields and are independent of how
the failure was produced.
"""

from typing import Any, Optional

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    DataflowGraph,
    Failure,
    FourPartCapture,
    LoopBodyCapture,
    build_dataflow_graph,
    compare_graphs,
    validate_edge_wait_coverage,
)

from dataflow_fixtures import make_capture, make_mfma


# Sentinel used by _matches_fields to distinguish "attribute missing" from
# "attribute is None / 0 / equal to the expected value". Module-level so
# all instances share identity.
_MISSING = object()


# Filler-MFMA register ranges for unused bodies. Picked high enough that
# they don't accidentally form edges against producers/consumers in the
# body under test (which conventionally lives in the v0..v100 range).
# Each body gets a distinct triple so cross-body acc-chain edges from
# MFMARule don't collapse into one identity.
_FILLER_RANGES = {
    BODY_LABEL_ML_PREV: (200, 204, 208),
    BODY_LABEL_NGL:     (220, 224, 228),
    BODY_LABEL_NLL:     (240, 244, 248),
}


def _filler_body(label: str) -> LoopBodyCapture:
    c, a, b = _FILLER_RANGES[label]
    return make_capture(label, [make_mfma(
        c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
    )])


class GraphNativeValidationTest:
    """Base class for graph-native CMS validation tests.

    Provides:

    * ``wrap_single_body(ml_capture, ...)`` — convert a single-body
      ``LoopBodyCapture`` into the four-body ``FourPartCapture`` required
      by ``build_dataflow_graph``. Filler bodies use high-vgpr ranges so
      they don't pollute the test's edge set.
    * ``build_graph(capture)`` — thin wrapper around ``build_dataflow_graph``.
    * ``compare(ref_capture, subj_capture)`` — two-graph comparison
      returning the ``Failure`` list.
    * ``validate_waits(graph)`` — single-graph wait-coverage check.
    * ``assert_failure(failures, *, cls, **fields)`` — assert exactly one
      failure of the given class matches the field constraints, return it.
    * ``assert_failures_contain(failures, *, cls, **fields)`` — at least
      one matching failure exists; return it.
    * ``assert_no_failures(failures)`` — positive-test assertion.
    * ``assert_edge_exists(graph, *, edge_kind, producer_category,
      consumer_category)`` — find at least one edge matching the
      (kind, producer-category, consumer-category) triple.

    The legacy ``assert_order_inverted`` / ``assert_timing_too_close`` /
    ``assert_scc_conflict`` / ``assert_out_of_order_sequence`` /
    ``assert_wrong_interleaving`` helpers are lifted unchanged from
    ``cms_validation_base.CMSValidationTestBase`` because they bind to
    typed Failure fields, not to how the Failure was produced.
    """

    # =========================================================================
    # Capture wrapping & graph construction
    # =========================================================================

    def wrap_single_body(
        self,
        ml_capture: LoopBodyCapture,
        *,
        ml_prev: Optional[LoopBodyCapture] = None,
        ngl: Optional[LoopBodyCapture] = None,
        nll: Optional[LoopBodyCapture] = None,
        num_mfma: int = 1,
        num_codepaths: int = 1,
        source: str = "cms",
        num_mfma_per_subiter: int = 0,
    ) -> FourPartCapture:
        """Wrap an ML body capture as a single-codepath ``FourPartCapture``.

        Bodies left as ``None`` are filled with a degenerate single-MFMA
        capture using high-vgpr ranges so the filler doesn't form edges
        against the body under test. The test must keep its own producers
        and consumers below vgpr index 200 to avoid collisions.
        """
        return FourPartCapture(
            main_loop={0: ml_capture},
            main_loop_prev={
                0: ml_prev if ml_prev is not None else _filler_body(BODY_LABEL_ML_PREV)
            },
            n_gl={0: ngl if ngl is not None else _filler_body(BODY_LABEL_NGL)},
            n_ll={0: nll if nll is not None else _filler_body(BODY_LABEL_NLL)},
            num_mfma=num_mfma,
            num_codepaths=num_codepaths,
            source=source,
            num_mfma_per_subiter=num_mfma_per_subiter,
        )

    def build_graph(self, capture: FourPartCapture) -> DataflowGraph:
        """Thin wrapper around ``build_dataflow_graph``.

        Trivial today; named explicitly so tests don't have to import
        ``build_dataflow_graph`` from ``ScheduleCapture`` and so future
        instrumentation (timing, edge-count assertions) can be hooked
        here without touching every test.
        """
        return build_dataflow_graph(capture)

    def compare(
        self,
        ref_capture: FourPartCapture,
        subj_capture: FourPartCapture,
        *,
        raise_on_unexplained: bool = True,
    ) -> list:
        """Build both graphs, run ``compare_graphs``, return failures."""
        ref = self.build_graph(ref_capture)
        subj = self.build_graph(subj_capture)
        return compare_graphs(ref, subj, raise_on_unexplained=raise_on_unexplained)

    def validate_waits(
        self,
        graph: DataflowGraph,
        *,
        raise_on_unexplained: bool = False,
    ) -> list:
        """Thin wrapper around ``validate_edge_wait_coverage``.

        Defaults ``raise_on_unexplained=False`` to match production: the
        intent of negative tests is to inspect the failure list, not to
        crash on an unclassified edge.
        """
        return validate_edge_wait_coverage(
            graph, raise_on_unexplained=raise_on_unexplained,
        )

    # =========================================================================
    # Failure-list assertions
    # =========================================================================

    @staticmethod
    def _matches_fields(failure: Failure, fields: dict) -> bool:
        """Return True if every (attr, expected) pair matches on ``failure``.

        Missing attributes are treated as a non-match (NOT an error) so
        callers can pass field constraints that filter on a subclass-specific
        attribute without having to know which Failure subclass each entry
        in the list is.
        """
        for attr_name, expected_value in fields.items():
            actual_value = getattr(failure, attr_name, _MISSING)
            if actual_value is _MISSING:
                return False
            if actual_value != expected_value:
                return False
        return True

    @staticmethod
    def assert_failure(failures: list, **kwargs: Any) -> Failure:
        """Assert exactly one failure of ``cls`` (and matching fields) is present.

        Usage::

            f = self.assert_failure(failures, cls=MissingWaitFailure,
                                    counter_kind="dscnt")

        Returns the matched failure so callers can do further field
        introspection — e.g. chain into ``self.assert_order_inverted(f, ...)``.

        ``cls`` is consumed from ``kwargs`` (not a named parameter) so the
        familiar ``cls=`` keyword from the legacy ``CMSValidationTestBase``
        plumbing works without colliding with the classmethod-implicit
        ``cls``. ``cls_`` is also accepted as an alias.
        """
        cls_ = kwargs.pop("cls", None)
        if cls_ is None:
            cls_ = kwargs.pop("cls_", None)
        if cls_ is None:
            raise TypeError(
                "assert_failure(...): missing required keyword 'cls' (or 'cls_')"
            )
        candidates = [
            f for f in failures
            if isinstance(f, cls_)
            and GraphNativeValidationTest._matches_fields(f, kwargs)
        ]
        if len(candidates) != 1:
            class_counts: dict = {}
            for f in failures:
                name = type(f).__name__
                class_counts[name] = class_counts.get(name, 0) + 1
            raise AssertionError(
                f"assert_failure: expected exactly 1 {cls_.__name__} matching "
                f"fields {kwargs!r}, found {len(candidates)}. "
                f"All failures: {class_counts!r}"
            )
        return candidates[0]

    @staticmethod
    def assert_failures_contain(failures: list, **kwargs: Any) -> Failure:
        """Assert at least one failure of ``cls`` (and matching fields) is present.

        Returns the FIRST matching failure for further field introspection.
        Use this when the rule under test is allowed to emit more than one
        failure (e.g. one per producer/consumer in a many-to-one edge).
        """
        cls_ = kwargs.pop("cls", None)
        if cls_ is None:
            cls_ = kwargs.pop("cls_", None)
        if cls_ is None:
            raise TypeError(
                "assert_failures_contain(...): missing required keyword "
                "'cls' (or 'cls_')"
            )
        for f in failures:
            if isinstance(f, cls_) and GraphNativeValidationTest._matches_fields(f, kwargs):
                return f
        class_counts: dict = {}
        for f in failures:
            name = type(f).__name__
            class_counts[name] = class_counts.get(name, 0) + 1
        raise AssertionError(
            f"assert_failures_contain: no {cls_.__name__} matching fields "
            f"{kwargs!r} found. All failures: {class_counts!r}"
        )

    @staticmethod
    def assert_no_failures(failures: list) -> None:
        """Assert the failure list is empty (positive-test path)."""
        if failures:
            descriptions = [f"{type(f).__name__}: {f.format()}" for f in failures]
            raise AssertionError(
                f"assert_no_failures: expected empty failure list, got "
                f"{len(failures)} failures:\n  " + "\n  ".join(descriptions)
            )

    # =========================================================================
    # Edge-set assertions
    # =========================================================================

    @staticmethod
    def assert_edge_exists(
        graph: DataflowGraph,
        *,
        edge_kind: Optional[str] = None,
        producer_category: Optional[str] = None,
        consumer_category: Optional[str] = None,
    ):
        """Assert at least one edge matches the given filters; return the first.

        All filters are optional; an unset filter matches anything. Use this
        to pin "the graph DOES contain the GR -> LR LDS-reuse edge we
        expected" in a positive test, distinct from "the graph PASSES
        wait-coverage" (which is what ``assert_no_failures`` covers).
        """
        for e in graph.edges:
            if edge_kind is not None and e.edge_kind != edge_kind:
                continue
            if producer_category is not None and e.producer.category != producer_category:
                continue
            if consumer_category is not None and e.consumer.category != consumer_category:
                continue
            return e
        edge_summary = [
            (e.producer.category, e.consumer.category, e.edge_kind)
            for e in graph.edges
        ]
        raise AssertionError(
            f"assert_edge_exists: no edge matching "
            f"edge_kind={edge_kind!r}, producer_category={producer_category!r}, "
            f"consumer_category={consumer_category!r}. "
            f"Edges in graph: {edge_summary!r}"
        )

    @staticmethod
    def assert_edge_absent(
        graph: DataflowGraph,
        *,
        edge_kind: Optional[str] = None,
        producer_category: Optional[str] = None,
        consumer_category: Optional[str] = None,
    ) -> None:
        """Inverse of ``assert_edge_exists``: assert no matching edge exists."""
        for e in graph.edges:
            if edge_kind is not None and e.edge_kind != edge_kind:
                continue
            if producer_category is not None and e.producer.category != producer_category:
                continue
            if consumer_category is not None and e.consumer.category != consumer_category:
                continue
            raise AssertionError(
                f"assert_edge_absent: found unexpected edge "
                f"({e.producer.category} -> {e.consumer.category}, "
                f"kind={e.edge_kind!r})"
            )

    # =========================================================================
    # Typed-Failure field assertions (lifted from CMSValidationTestBase
    # unchanged — they bind to typed Failure fields, independent of how
    # the failure was produced).
    # =========================================================================

    @staticmethod
    def assert_order_inverted(failure, *, producer_name, producer_idx,
                              consumer_name, consumer_idx):
        """Assert OrderInvertedFailure / ConstraintViolationFailure carries
        the expected producer/consumer identity AND vmfma_index positions.

        Pinning positions matters because a name-only check passes if any
        of N similarly-named instructions violate ordering — but we want
        to verify the *specific* schedule slot the test set up is the one
        being flagged.

        Works for any Failure subclass with ``.producer`` and ``.consumer``
        nodes that expose ``.name`` and ``.issued_at.vmfma_index`` /
        ``.position.vmfma_index``.
        """
        assert failure.producer.name == producer_name, (
            f"producer.name: expected {producer_name!r}, got {failure.producer.name!r}"
        )
        prod_pos = (
            getattr(failure.producer, "issued_at", None)
            or failure.producer.position
        )
        assert prod_pos.vmfma_index == producer_idx, (
            f"producer position.vmfma_index: expected {producer_idx}, "
            f"got {prod_pos.vmfma_index}"
        )
        assert failure.consumer.name == consumer_name, (
            f"consumer.name: expected {consumer_name!r}, got {failure.consumer.name!r}"
        )
        cons_pos = (
            getattr(failure.consumer, "issued_at", None)
            or failure.consumer.position
        )
        assert cons_pos.vmfma_index == consumer_idx, (
            f"consumer position.vmfma_index: expected {consumer_idx}, "
            f"got {cons_pos.vmfma_index}"
        )

    @staticmethod
    def assert_timing_too_close(failure, *, producer_name, producer_idx,
                                consumer_name, consumer_idx,
                                expected_quad_cycles, actual_quad_cycles):
        """Assert TimingTooCloseFailure carries the expected producer/consumer
        positions AND quad-cycle counts."""
        assert failure.producer.name == producer_name
        prod_pos = (
            getattr(failure.producer, "issued_at", None)
            or failure.producer.position
        )
        assert prod_pos.vmfma_index == producer_idx
        assert failure.consumer.name == consumer_name
        cons_pos = (
            getattr(failure.consumer, "issued_at", None)
            or failure.consumer.position
        )
        assert cons_pos.vmfma_index == consumer_idx
        assert failure.expected_quad_cycles == expected_quad_cycles, (
            f"expected_quad_cycles: expected {expected_quad_cycles}, "
            f"got {failure.expected_quad_cycles}"
        )
        assert failure.actual_quad_cycles == actual_quad_cycles, (
            f"actual_quad_cycles: expected {actual_quad_cycles}, "
            f"got {failure.actual_quad_cycles}"
        )

    @staticmethod
    def assert_scc_conflict(failure, *, conflicting_name, grinc_name,
                            conflicting_index, interval_start, interval_end):
        """Assert SCCConflictFailure carries the expected conflicting/GRInc
        identities AND the SCC-protected interval bounds."""
        assert failure.conflicting_name == conflicting_name, (
            f"conflicting_name: expected {conflicting_name!r}, got {failure.conflicting_name!r}"
        )
        assert failure.grinc_name == grinc_name, (
            f"grinc_name: expected {grinc_name!r}, got {failure.grinc_name!r}"
        )
        assert failure.conflicting_index == conflicting_index, (
            f"conflicting_index: expected {conflicting_index}, got {failure.conflicting_index}"
        )
        assert failure.interval_start == interval_start, (
            f"interval_start: expected {interval_start}, got {failure.interval_start}"
        )
        assert failure.interval_end == interval_end, (
            f"interval_end: expected {interval_end}, got {failure.interval_end}"
        )

    @staticmethod
    def assert_out_of_order_sequence(failure, *, schedule_key, sequence,
                                     bad_value, bad_index, prev_value):
        """Assert OutOfOrderSequenceFailure (kind='sequence') carries the
        expected schedule-key, full sequence, and bad-position triple."""
        assert failure.kind == "sequence", (
            f"kind: expected 'sequence', got {failure.kind!r}"
        )
        assert failure.schedule_key == schedule_key, (
            f"schedule_key: expected {schedule_key!r}, got {failure.schedule_key!r}"
        )
        assert failure.sequence == sequence, (
            f"sequence: expected {sequence!r}, got {failure.sequence!r}"
        )
        assert failure.bad_value == bad_value, (
            f"bad_value: expected {bad_value}, got {failure.bad_value}"
        )
        assert failure.bad_index == bad_index, (
            f"bad_index: expected {bad_index}, got {failure.bad_index}"
        )
        assert failure.prev_value == prev_value, (
            f"prev_value: expected {prev_value}, got {failure.prev_value}"
        )

    @staticmethod
    def assert_wrong_interleaving(failure, *, pack_name, pack_idx,
                                  expected_next_name, expected_next_idx,
                                  actual_next_name, actual_next_idx):
        """Assert WrongInterleavingFailure carries the expected pack +
        expected/actual successor identities and positions."""
        assert failure.pack.name == pack_name
        pack_pos = (
            getattr(failure.pack, "issued_at", None)
            or failure.pack.position
        )
        assert pack_pos.vmfma_index == pack_idx
        assert failure.expected_next.name == expected_next_name
        exp_pos = (
            getattr(failure.expected_next, "issued_at", None)
            or failure.expected_next.position
        )
        assert exp_pos.vmfma_index == expected_next_idx
        assert failure.actual_next.name == actual_next_name
        act_pos = (
            getattr(failure.actual_next, "issued_at", None)
            or failure.actual_next.position
        )
        assert act_pos.vmfma_index == actual_next_idx
