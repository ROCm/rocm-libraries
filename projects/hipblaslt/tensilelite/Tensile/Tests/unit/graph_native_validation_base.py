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
``assert_scc_conflict`` / ``assert_wrong_interleaving`` helpers are lifted onto this base unchanged
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
    FourPartCapture,
    LoopBodyCapture,
    build_dataflow_graph,
    compare_graphs,
    validate_edge_wait_coverage,
)
from Tensile.Components.CMSValidator import (
    Failure,
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
        """Assert the failure list is empty (positive-test path).

        Renders each unexpected Failure via repr() rather than .format() —
        the helper has no capture in scope (test infrastructure layer above
        the per-body context), and Failure.format requires a real capture
        for the per-category-stream [N] index. repr() shows the dataclass
        fields, which is sufficient diagnostic output for "this list was
        supposed to be empty".
        """
        if failures:
            descriptions = [f"{type(f).__name__}: {f!r}" for f in failures]
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
    def _label_category(label) -> str:
        """Extract bare category from a FailureNodeLabel.primary string.

        Primary is either 'Category' (plain MFMA) or 'Category[N]'. Strip
        the [N] suffix so the helper can compare against the bare-category
        names callers pass via `producer_name=` / `consumer_name=`.
        """
        primary = label.primary
        bracket = primary.find("[")
        return primary[:bracket] if bracket != -1 else primary

    @staticmethod
    def _label_idx(label) -> int:
        """Extract vmfma_index from a FailureNodeLabel.position string.

        Position is '@ idx=N' under the CMS-side provider; parse the
        integer suffix so position checks survive the string-only label API.
        """
        pos = label.position
        eq = pos.rfind("=")
        return int(pos[eq + 1:])

    def assert_order_inverted(self, failure, *, producer_name, producer_idx,
                              consumer_name, consumer_idx):
        """Assert OrderInvertedFailure carries the expected producer/consumer
        identity AND vmfma_index positions.

        Pinning positions matters because a name-only check passes if any
        of N similarly-named instructions violate ordering — but we want
        to verify the *specific* schedule slot the test set up is the one
        being flagged.

        Works for any Failure subclass with ``.producer`` and ``.consumer``
        nodes that expose ``.name`` and ``.issued_at.vmfma_index`` /
        ``.position.vmfma_index``.
        """
        prod_cat = self._label_category(failure.producer)
        prod_idx = self._label_idx(failure.producer)
        cons_cat = self._label_category(failure.consumer)
        cons_idx = self._label_idx(failure.consumer)
        assert prod_cat == producer_name, (
            f"producer category: expected {producer_name!r}, got {prod_cat!r}"
        )
        assert prod_idx == producer_idx, (
            f"producer vmfma_index: expected {producer_idx}, got {prod_idx}"
        )
        assert cons_cat == consumer_name, (
            f"consumer category: expected {consumer_name!r}, got {cons_cat!r}"
        )
        assert cons_idx == consumer_idx, (
            f"consumer vmfma_index: expected {consumer_idx}, got {cons_idx}"
        )

    def assert_timing_too_close(self, failure, *, producer_name, producer_idx,
                                consumer_name, consumer_idx,
                                expected_quad_cycles, actual_quad_cycles):
        """Assert TimingTooCloseFailure carries the expected producer/consumer
        positions AND quad-cycle counts."""
        assert self._label_category(failure.producer) == producer_name
        assert self._label_idx(failure.producer) == producer_idx
        assert self._label_category(failure.consumer) == consumer_name
        assert self._label_idx(failure.consumer) == consumer_idx
        assert failure.expected_quad_cycles == expected_quad_cycles, (
            f"expected_quad_cycles: expected {expected_quad_cycles}, "
            f"got {failure.expected_quad_cycles}"
        )
        assert failure.actual_quad_cycles == actual_quad_cycles, (
            f"actual_quad_cycles: expected {actual_quad_cycles}, "
            f"got {failure.actual_quad_cycles}"
        )

    def assert_wrong_interleaving(self, failure, *, pack_name, pack_idx,
                                  expected_next_name, expected_next_idx,
                                  actual_next_name, actual_next_idx):
        """Assert OverriddenInputFailure carries the expected pack +
        expected/actual successor identities and positions.

        Argument names retain the legacy pack/expected_next/actual_next
        vocabulary (caller convenience). Internally these map to the
        unified shape: pack -> producer, expected_next -> consumer,
        actual_next -> intervening_writer.
        """
        assert self._label_category(failure.producer) == pack_name
        assert self._label_idx(failure.producer) == pack_idx
        assert self._label_category(failure.consumer) == expected_next_name
        assert self._label_idx(failure.consumer) == expected_next_idx
        assert self._label_category(failure.intervening_writer) == actual_next_name
        assert self._label_idx(failure.intervening_writer) == actual_next_idx
