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
"""diagnose_extra_edge and the symmetric (subj − ref) direction of compare_graphs.

Tests for rocm-libraries-67us (C3e): EdgeRoutedDifferentlyFailure classifier
and the extra_keys processing block inside compare_graphs.

Test coverage:
  1. diagnose_extra_edge Phase 0: CaptureConsistencyError when ref has no
     writer for one of subj's byte_keys.
  2. diagnose_extra_edge Phase 1: EdgeRoutedDifferentlyFailure when ref's
     closest-prior writer has a different identity than subj's producer.
  3. diagnose_extra_edge Phase 1: UnexplainedExtraEdgeError when ref's
     closest-prior writer matches subj's producer identity exactly (the edge
     should have canceled in set-diff — validator-bug path).
  4. compare_graphs symmetric direction: extra_keys processing emits
     EdgeRoutedDifferentlyFailure for an asymmetric subj-extra edge pair.
  5. compare_graphs with subj == ref returns [] (regression guard for the
     symmetric direction being added — extra_keys must be empty when graphs
     are identical).
  6. EdgeRoutedDifferentlyFailure.format() output structure (string content
     assertions on key phrases).
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    FourPartCapture,
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    CaptureConsistencyError,
    UnexplainedExtraEdgeError,
)
from Tensile.Components.CMSValidator import (
    EdgeRoutedDifferentlyFailure,
    build_dataflow_graph,
    compare_graphs,
    diagnose_extra_edge,
    _DEFAULT_CDNA4_ARCH_PROFILE,
)

from dataflow_fixtures import (
    make_lr, make_gr, make_mfma, make_swait, make_sbarrier, make_capture,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap(ml_capture, *, ml_prev=None, ngl=None, nll=None):
    """Wrap an ML LoopBodyCapture into a FourPartCapture."""
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
        main_loop_prev={0: ml_prev if ml_prev is not None else _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: ngl if ngl is not None else _filler(BODY_LABEL_NGL)},
        n_ll={0: nll if nll is not None else _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


# =============================================================================
# Fixtures
# =============================================================================


def _build_graph_lr_swait_mfma(lr_dst_start=8, lr_dst_count=4, lr_lds_offset=64,
                                *, lr_slot=0, mfma_a_start=8, mfma_a_count=4):
    """Standard LR -> SWait(dscnt=0) -> MFMA graph.

    Produces one raw_intrawave edge from the LR to the MFMA. Both graphs
    share the same instruction identities — the 'standard ref' shape used
    throughout these tests.
    """
    cap = make_capture(BODY_LABEL_ML, [
        make_lr(lr_dst_start, lr_dst_count, lr_lds_offset,
                slot=lr_slot, category="LRA0"),
        make_swait(slot=lr_slot + 1, dscnt=0),
        make_mfma(c_dst_start=0, a_src_start=mfma_a_start,
                  b_src_start=32, slot=lr_slot + 2,
                  a_src_count=mfma_a_count),
    ])
    return build_dataflow_graph(_wrap(cap))


# =============================================================================
# Test 1 — diagnose_extra_edge Phase 0: CaptureConsistencyError
# =============================================================================


class TestDiagnoseExtraEdgePhase0CaptureConsistencyError:
    """Phase 0 gate: if ref has no writer for any of subj's byte_keys, raise
    CaptureConsistencyError. Tested by removing all byte_key_writers from the
    ref_graph object and calling diagnose_extra_edge directly."""

    def test_raises_capture_consistency_error_when_ref_has_no_writers(self):
        # Build two standard identical graphs; we'll borrow one edge from
        # subj_graph and patch the ref_graph's byte_key_writers to empty.
        g_subj = _build_graph_lr_swait_mfma()
        g_ref = _build_graph_lr_swait_mfma()

        # Pick the first edge in the subject graph (LR -> MFMA raw_intrawave).
        # It has non-empty producer_write_byte_key.
        subj_edge = next(
            e for e in g_subj.edges
            if e.edge_kind == "raw_intrawave" and e.producer_write_byte_key
        )

        # Patch ref graph to have no byte_key_writers for any of subj's byte_keys.
        empty_ref_graph = g_ref
        original_writers = dict(empty_ref_graph.byte_key_writers)
        for bk in subj_edge.producer_write_byte_key:
            empty_ref_graph.byte_key_writers.pop(bk, None)

        with pytest.raises(CaptureConsistencyError, match="same-instruction-set contract"):
            diagnose_extra_edge(subj_edge, g_subj, empty_ref_graph)

        # Restore (avoid test contamination of shared object).
        empty_ref_graph.byte_key_writers.update(original_writers)


# =============================================================================
# Test 2 — diagnose_extra_edge Phase 1: EdgeRoutedDifferentlyFailure
# =============================================================================


class TestDiagnoseExtraEdgePhase1RoutedDifferently:
    """Phase 1: ref has writers for all of subj's byte_keys, but the closest-prior
    writer in ref has a different identity than subj's producer. Emits
    EdgeRoutedDifferentlyFailure.

    Build strategy: construct a subj graph where LR-A writes bytes that are also
    written by LR-B in ref (different LDS offsets → different identity). The
    subj schedule pairs LR-A with the MFMA consumer; the ref schedule pairs LR-B
    with the same consumer. The byte_keys overlap (same vgpr range), so the
    edge_key for subj's LR-A->MFMA edge is NOT canceled by ref's LR-B->MFMA edge
    (they have different source_module_id / emission_ordinal prefix → distinct
    8-field keys).

    We test diagnose_extra_edge directly to avoid the compare_graphs data-flow
    category-count gate (which requires both graphs to have equal per-category
    node counts — but this is also tested in Test 4 via compare_graphs).
    """

    def test_edge_routed_differently_failure_emitted(self):
        # Subject: LR at vgpr 8 (LDS offset 64) → SWait → MFMA reading vgpr 8.
        # Reference: LR at vgpr 8 (LDS offset 128) → SWait → MFMA reading vgpr 8.
        # Same vgpr destination => same byte_keys; different LDS source offsets
        # => different instructions => different identities.
        g_subj = _build_graph_lr_swait_mfma(
            lr_dst_start=8, lr_dst_count=4, lr_lds_offset=64)
        g_ref = _build_graph_lr_swait_mfma(
            lr_dst_start=8, lr_dst_count=4, lr_lds_offset=128)

        # Pick the subj's LR->MFMA raw_intrawave edge.
        subj_edge = next(
            e for e in g_subj.edges
            if e.edge_kind == "raw_intrawave" and e.producer_write_byte_key
        )

        result = diagnose_extra_edge(subj_edge, g_subj, g_ref)

        assert len(result) == 1, f"Expected 1 failure, got {result!r}"
        assert isinstance(result[0], EdgeRoutedDifferentlyFailure), (
            f"Expected EdgeRoutedDifferentlyFailure, got {type(result[0])}"
        )

    def test_failure_fields_populated(self):
        """subj_producer and subj_consumer are non-None; ref_producer is non-None
        (ref had a closest-prior writer); byte_keys is non-empty."""
        g_subj = _build_graph_lr_swait_mfma(
            lr_dst_start=8, lr_dst_count=4, lr_lds_offset=64)
        g_ref = _build_graph_lr_swait_mfma(
            lr_dst_start=8, lr_dst_count=4, lr_lds_offset=128)
        subj_edge = next(
            e for e in g_subj.edges
            if e.edge_kind == "raw_intrawave" and e.producer_write_byte_key
        )

        failure = diagnose_extra_edge(subj_edge, g_subj, g_ref)[0]
        assert failure.subj_producer is not None
        assert failure.subj_consumer is not None
        assert failure.ref_producer is not None
        assert failure.byte_keys, "byte_keys should be non-empty"


# =============================================================================
# Test 3 — diagnose_extra_edge Phase 1: UnexplainedExtraEdgeError
# =============================================================================


class TestDiagnoseExtraEdgeUnexplainedExtraEdgeError:
    """Phase 1 identity-match path: ref's closest-prior writer has the SAME
    identity as subj's producer for every byte_key. The edge should have
    canceled in set-diff — this is a validator bug. Raise UnexplainedExtraEdgeError.

    Construction: use identical captures for ref and subj. Both have the same LR
    producing the same byte_keys. The LR->MFMA edge appears in BOTH graphs, so
    set-diff would cancel it. We call diagnose_extra_edge DIRECTLY (bypassing the
    set-diff) to confirm the validator-bug path is exercised correctly.
    """

    def test_raises_unexplained_extra_edge_error(self):
        g_subj = _build_graph_lr_swait_mfma()
        g_ref = _build_graph_lr_swait_mfma()

        # Pick the subj LR->MFMA edge (same instruction in both graphs).
        subj_edge = next(
            e for e in g_subj.edges
            if e.edge_kind == "raw_intrawave" and e.producer_write_byte_key
        )
        # ref has the SAME LR producer (identical identity) for these byte_keys.
        # Calling diagnose_extra_edge directly bypasses set-diff cancellation.
        with pytest.raises(UnexplainedExtraEdgeError):
            diagnose_extra_edge(subj_edge, g_subj, g_ref)


# =============================================================================
# Test 4 — compare_graphs symmetric direction: extra_keys processing
# =============================================================================


class TestCompareGraphsSymmetricDirection:
    """compare_graphs processes extra_keys (subj − ref) via diagnose_extra_edge.

    Structural note on why pure build_dataflow_graph graphs cannot easily exercise
    extra_keys through compare_graphs:

    For extra_keys (subj − ref) to be non-empty, two conditions must hold:
      (a) subj has an edge with a key that ref does not have; and
      (b) the per-category node count gate at compare_graphs entry must pass.

    The 8-field edge_key includes (source_module_id, emission_ordinal) from the
    producer. If ref and subj are built from captures with the same per-category
    node counts, both LR producers have the same source_module_id (None by default)
    and the same emission_ordinal (0 for the first LR of a given render) — so their
    edge_keys are identical and extra_keys is always empty.

    To get a non-empty extra_keys with equal counts, producers would need different
    source_module_ids. But if source_module_ids differ, the ref's producer identity
    is absent from subj, which causes diagnose_missing_edge's Phase 0 identity
    lookup to raise CaptureConsistencyError before the extra_keys loop is reached.

    Therefore: the compare_graphs wiring test uses a mock to inject a non-empty
    extra_keys scenario directly, verifying that compare_graphs calls
    diagnose_extra_edge and includes its failures in the returned list.
    The diagnose_extra_edge function itself is tested end-to-end in Tests 1–3.
    Test 5 below tests the symmetric zero-delta case through real graphs.
    """

    def test_extra_keys_wiring_via_mock(self):
        """compare_graphs calls diagnose_extra_edge for each key in extra_keys and
        extends the failures list with its return values.

        Uses unittest.mock to inject a synthetic extra_keys scenario: the
        subject graph reports one extra edge key (not in ref). The
        diagnose_extra_edge function is patched to return a sentinel failure.
        The test verifies that compare_graphs calls diagnose_extra_edge and
        includes the sentinel failure in its output.

        Why mock: for extra_keys to be non-empty without triggering the
        category-count gate, producers in ref and subj must differ in
        source_module_id. But different source_module_ids cause diagnose_missing_edge
        to raise CaptureConsistencyError for the missing_keys direction before
        extra_keys is processed. Injecting a synthetic extra_keys via mock is the
        only reliable way to test the wiring without side-effects from the
        missing_keys path.
        """
        from unittest.mock import patch, MagicMock
        import Tensile.Components.CMSValidator as cmsv

        g_ref = _build_graph_lr_swait_mfma()
        g_subj = _build_graph_lr_swait_mfma()  # identical — extra_keys would be {}

        # The sentinel failure that our patched diagnose_extra_edge returns.
        sentinel_failure = EdgeRoutedDifferentlyFailure(
            byte_keys=(("v", 8),),
            byte_key_routing={},
        )

        fake_extra_key = ("sentinel_mod", 99, (("v", 8),), (("v", 8),),
                          "raw_intrawave", (0,), 0, 0)

        # Build a mock subj edge that maps to fake_extra_key in subj_edges_by_key.
        mock_subj_edge = MagicMock()
        mock_subj_edge.producer.identity = ("render", "sentinel_mod", 99)
        mock_subj_edge.producer_write_byte_key = (("v", 8),)
        mock_subj_edge.consumer_read_byte_key = (("v", 8),)
        mock_subj_edge.edge_kind = "raw_intrawave"
        mock_subj_edge.intra_operand_byte_offset = (0,)
        mock_subj_edge.src_operand_slot = 0
        mock_subj_edge.sink_operand_slot = 0

        # Augmented subj edge_keys: original keys + the fake_extra_key.
        original_keys = g_subj.edge_keys()
        subj_keys_with_extra = original_keys | {fake_extra_key}

        with patch.object(g_subj, 'edge_keys', return_value=subj_keys_with_extra), \
             patch.object(g_subj, 'edges', [mock_subj_edge] + list(g_subj.edges)), \
             patch.object(cmsv, 'diagnose_extra_edge',
                          return_value=[sentinel_failure]) as mock_diag:
            # missing_keys = ref_keys - subj_keys: ref_keys == original_keys,
            # subj_keys == original_keys | {fake_extra_key}.
            # missing_keys = {} (nothing in ref that's not in subj).
            # extra_keys = subj_keys - ref_keys = {fake_extra_key}.
            failures = compare_graphs(g_ref, g_subj)

        mock_diag.assert_called_once()
        assert sentinel_failure in failures, (
            f"Expected sentinel_failure in compare_graphs output; got {failures!r}"
        )

    def test_symmetric_identical_graphs_no_extra_keys_failures(self):
        """compare_graphs(g, g) must return [] — extra_keys is empty when both
        graphs are identical. Regression guard: the symmetric direction must not
        introduce spurious failures on the identity case.
        """
        g = _build_graph_lr_swait_mfma()
        assert compare_graphs(g, g) == []


# =============================================================================
# Test 5 — EdgeRoutedDifferentlyFailure.format() output
# =============================================================================


class TestEdgeRoutedDifferentlyFailureFormat:
    """format() output must contain the key phrases that diagnose the failure.

    Assertions are on string content (substrings), not exact wording. Key
    required phrases:
      - subj_producer.primary (consumer reads FROM this)
      - ref_producer.primary (reference routed through this)
      - 'byte_keys'
      - 'DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3' (citation from plan §3)
    """

    def _make_label(self, primary: str, position: str = "@ idx=0"):
        from Tensile.Components.CMSValidator import FailureNodeLabel
        return FailureNodeLabel(primary=primary, position=position,
                                body_label=BODY_LABEL_ML)

    def test_format_contains_required_phrases(self):
        failure = EdgeRoutedDifferentlyFailure(
            subj_producer=self._make_label("LRA0[0]"),
            subj_consumer=self._make_label("MFMA[0]"),
            ref_producer=self._make_label("LRA0[1]"),
            byte_keys=(("v", 8), ("v", 9), ("v", 10), ("v", 11)),
            byte_key_routing={
                ("v", 8):  (("shadow_render", "mod_ref", 1), ("subj_render", "mod_subj", 0)),
            },
        )
        rendered = failure.format()
        assert "LRA0[0]" in rendered, f"subj_producer.primary not in output: {rendered!r}"
        assert "LRA0[1]" in rendered, f"ref_producer.primary not in output: {rendered!r}"
        assert "byte_keys" in rendered, f"'byte_keys' not in output: {rendered!r}"
        assert "DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3" in rendered, (
            f"doc citation not in output: {rendered!r}"
        )

    def test_format_with_none_ref_producer(self):
        """When ref_producer is None (ref had no prior writer), the output
        mentions 'no prior writer' at this consumer position."""
        failure = EdgeRoutedDifferentlyFailure(
            subj_producer=self._make_label("LRA0[0]"),
            subj_consumer=self._make_label("MFMA[0]"),
            ref_producer=None,
            byte_keys=(("v", 8),),
            byte_key_routing={},
        )
        rendered = failure.format()
        assert "no prior writer" in rendered, (
            f"Expected 'no prior writer' phrase when ref_producer is None: {rendered!r}"
        )
