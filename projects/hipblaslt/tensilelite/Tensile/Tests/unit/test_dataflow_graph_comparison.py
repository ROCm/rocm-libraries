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
"""compare_graphs and diagnose_missing_edge.

Each missing edge in the CMS graph (relative to the default reference)
runs through the diagnostic classifier, which returns a list of typed
Failures. Tests assert on Failure type and field, NOT on string content.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    FourPartCapture,
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    build_dataflow_graph,
    compare_graphs,
    diagnose_missing_edge,
    OrderInvertedFailure,
    MissingWaitFailure,
    WaitOnWrongCounterFailure,
    WaitInsufficientFailure,
    MissingBarrierFailure,
    CaptureConsistencyError,
    UnexplainedMissingEdgeError,
)

from dataflow_fixtures import (
    make_lr, make_gr, make_mfma, make_swait, make_sbarrier, make_capture,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap(ml_capture, *, ml_prev=None, ngl=None, nll=None):
    def _filler(label):
        return make_capture(label, [make_mfma(
            c_dst_start=200, a_src_start=204, b_src_start=208, slot=0,
        )])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: ml_prev if ml_prev is not None else _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: ngl if ngl is not None else _filler(BODY_LABEL_NGL)},
        n_ll={0: nll if nll is not None else _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
    )


# =============================================================================
# Positive — clean comparison emits no Failures
# =============================================================================


class TestCleanComparison:
    def test_identical_graphs_no_failures(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(cap))
        # compare_graphs to itself -> []
        assert compare_graphs(g_ref, g_ref) == []

    def test_same_dataflow_different_positions_no_failures(self):
        """Two captures with same instructions but different SWait/SNop
        positions; both graphs form the same edge set."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject: same producer/consumer/wait counter, different stream slots.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=5, category="LRA0"),
            make_swait(slot=6, dscnt=0),
            make_mfma(0, 8, 32, slot=7, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        assert compare_graphs(g_ref, g_subj) == []


# =============================================================================
# Negative — per-Failure-class diagnosis
# =============================================================================


class TestPerFailureDiagnosis:
    def _build_ref(self):
        """Standard reference: LR -> SWait(dscnt=0) -> MFMA forming one
        raw_intrawave edge."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        return build_dataflow_graph(_wrap(cap))

    def test_missing_wait_diagnosis(self):
        """Subject capture deletes the SWait — no waits at all in the window."""
        ref = self._build_ref()
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(ref, subj)
        assert any(isinstance(f, MissingWaitFailure) for f in failures)
        mw = next(f for f in failures if isinstance(f, MissingWaitFailure))
        assert mw.counter_kind == "dscnt"

    def test_wait_on_wrong_counter_diagnosis(self):
        """Subject replaces SWait(dscnt=0) with SWait(vlcnt=0)."""
        ref = self._build_ref()
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, vlcnt=0),    # wrong counter
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(ref, subj)
        assert any(isinstance(f, WaitOnWrongCounterFailure) for f in failures)
        wf = next(f for f in failures if isinstance(f, WaitOnWrongCounterFailure))
        assert wf.expected_counter == "dscnt"
        assert len(wf.wrong_counter_waits) >= 1

    def test_wait_insufficient_diagnosis(self):
        """Subject replaces SWait(dscnt=0) with SWait(dscnt=2) while 5 LRs
        are in flight."""
        # Reference: 5 LRs + SWait(dscnt=0) + MFMA on the 5th LR's regs.
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),     # LR_a
            make_lr(12, 4, 80, slot=1, category="LRA0"),    # LR_b
            make_lr(16, 4, 96, slot=2, category="LRA0"),    # LR_c
            make_lr(20, 4, 112, slot=3, category="LRA0"),   # LR_d
            make_lr(24, 4, 128, slot=4, category="LRA0"),   # LR_e
            make_swait(slot=5, dscnt=0),
            make_mfma(0, 24, 32, slot=6, a_src_count=4),    # reads LR_e's regs
        ])
        ref = build_dataflow_graph(_wrap(ref_cap))
        # Subject: same 5 LRs, but SWait(dscnt=2) leaves the youngest 2 pending.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
            make_lr(16, 4, 96, slot=2, category="LRA0"),
            make_lr(20, 4, 112, slot=3, category="LRA0"),
            make_lr(24, 4, 128, slot=4, category="LRA0"),
            make_swait(slot=5, dscnt=2),                    # only drains the oldest 3
            make_mfma(0, 24, 32, slot=6, a_src_count=4),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(ref, subj)
        assert any(isinstance(f, WaitInsufficientFailure) for f in failures)

    def test_missing_barrier_must_start_after_diagnosis(self):
        """Subject deletes the SBarrier between LR0's SWait and GR."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_sbarrier(slot=2),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ])
        ref = build_dataflow_graph(_wrap(ref_cap))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            # SBarrier deleted
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(ref, subj)
        mb = [f for f in failures if isinstance(f, MissingBarrierFailure)]
        assert len(mb) >= 1
        assert mb[0].role == "must_start_after"

    def test_missing_barrier_needed_by_diagnosis(self):
        """Subject deletes the SBarrier between GR's SWait and LR1."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=2),
            make_lr(8, 4, 64, slot=3, category="LRA1"),
        ])
        ref = build_dataflow_graph(_wrap(ref_cap))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            # SBarrier deleted
            make_lr(8, 4, 64, slot=3, category="LRA1"),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(ref, subj)
        mb = [f for f in failures if isinstance(f, MissingBarrierFailure)]
        assert len(mb) >= 1
        assert mb[0].role == "needed_by"

    def test_missing_wait_suppresses_missing_barrier(self):
        """Subject deletes BOTH the SWait and the SBarrier in an LR0 -> GR
        sequence -> only MissingWaitFailure (no MissingBarrierFailure)."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_sbarrier(slot=2),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ])
        ref = build_dataflow_graph(_wrap(ref_cap))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            # Both SWait and SBarrier deleted.
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(ref, subj)
        # Has at least one MissingWaitFailure
        assert any(isinstance(f, MissingWaitFailure) for f in failures)
        # No MissingBarrierFailure on the same edge
        assert not any(isinstance(f, MissingBarrierFailure) for f in failures)


# =============================================================================
# Failure type assertions
# =============================================================================


class TestFailureTypeContract:
    def test_failure_isinstance_dispatch(self):
        """Returned Failures route through isinstance — no .kind discriminator."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        ref = build_dataflow_graph(_wrap(ref_cap))
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(ref, subj)
        for f in failures:
            # Must be a typed Failure subclass — duck typing on .kind is wrong.
            from Tensile.Components.ScheduleCapture import Failure
            assert isinstance(f, Failure)


# =============================================================================
# Identity-coverage assertion at compare_graphs entry
# =============================================================================


class TestIdentityCoverage:
    def test_compare_graphs_identity_mismatch_raises(self):
        """Two graphs whose node identity sets differ -> raises."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject has an EXTRA LR — an identity not in the reference.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),  # extra
            make_swait(slot=2, dscnt=0),
            make_mfma(0, 8, 32, slot=3, a_src_count=4),
        ])
        ref = build_dataflow_graph(_wrap(ref_cap))
        subj = build_dataflow_graph(_wrap(subj_cap))
        with pytest.raises(CaptureConsistencyError):
            compare_graphs(ref, subj)


# =============================================================================
# Render-string identity robustness
# =============================================================================
# Identity is built from canonical render-strings, NOT from structured
# register-field tuples. This makes the comparison robust to:
# - register naming variations (symbolic vs numeric vs mixed) — the
#   render-string IS the canonical form
# - cosmetic differences (comment text, whitespace) — stripped/normalized
#   in _canonical_render


class TestRenderStringIdentity:
    def test_identity_is_assembly_text(self):
        """The identity tuple's third element should be the canonical
        rendered assembly. This is what makes identity robust to
        register-naming variations."""
        from Tensile.Components.ScheduleCapture import _identity_for
        lr = make_lr(8, 4, 64, slot=0, category="LRA0").inst
        ident = _identity_for(lr, BODY_LABEL_ML)
        assert ident[0] == "LR"             # class tag
        assert isinstance(ident[1], int)    # loop_index
        assert isinstance(ident[2], str)    # render-string
        # The render contains a vgpr reference and the LDS offset.
        # Synthetic fixtures construct RegisterContainer with a RegName,
        # so the render uses the symbolic form (v8 is the name we chose).
        assert "vgpr" in ident[2] or "v[" in ident[2]
        assert "lds[64]" in ident[2]

    def test_comment_differences_dont_change_identity(self):
        """Two instructions with identical operations but different
        comments (e.g. 'init' vs 'reload') should have the same identity.
        Confirms that _canonical_render strips comments."""
        from Tensile.Components.ScheduleCapture import _canonical_render

        # Use real rocisa instances to test the comment-stripping path.
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr
        a = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0), comment="init")
        b = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0), comment="reload")
        # render-strings might or might not include comment in str();
        # _canonical_render strips it either way.
        assert _canonical_render(a) == _canonical_render(b)

    def test_whitespace_normalization(self):
        """Multiple spaces / tabs / newlines collapse to single spaces."""
        from Tensile.Components.ScheduleCapture import _canonical_render

        class _Spaced:
            def __str__(self):
                return "v_mfma  v[0:3], \tv[8:9], \n v[32:33]"

        result = _canonical_render(_Spaced())
        # No double spaces, no tabs, no newlines
        assert "  " not in result
        assert "\t" not in result
        assert "\n" not in result
        assert result == "v_mfma v[0:3], v[8:9], v[32:33]"

    def test_mixed_symbolic_numeric_registers_in_same_inst(self):
        """An instruction with MIXED symbolic and numeric registers (the
        F32X TF32 emulation pattern: symbolic ValuA_T + numeric scratch +
        symbolic ValuA_X) renders consistently. If both captures emit the
        same inst with the same register identifiers, identities match."""
        from Tensile.Components.ScheduleCapture import _identity_for, _canonical_render
        from rocisa.instruction import MFMAInstruction
        from rocisa.container import vgpr
        from rocisa.enum import InstType

        # Construct two identical mixed-register MFMAs (same args).
        def _build():
            return MFMAInstruction(
                instType=InstType.INST_BF16, accType=InstType.INST_F32,
                variant=[4, 4, 4, 16], mfma1k=False,
                acc=vgpr("ValuA_T0_I0", 4),       # symbolic
                a=vgpr(74, 2),                    # numeric scratch
                b=vgpr("ValuA_X0_I0", 2),         # symbolic
            )

        a = _build()
        b = _build()
        # Even though the MFMA mixes symbolic and numeric register kinds,
        # the rendered string is deterministic; both instances render
        # identically; identities match.
        assert _canonical_render(a) == _canonical_render(b)
        assert _identity_for(a, BODY_LABEL_ML) == _identity_for(b, BODY_LABEL_ML)
        # Sanity check: render contains all three reg kinds.
        rendered = _canonical_render(a)
        assert "vgprValuA_T0_I0" in rendered    # symbolic acc
        assert "v[74:75]" in rendered           # numeric a
        assert "vgprValuA_X0_I0" in rendered    # symbolic b

    def test_symbolic_and_numeric_for_same_logical_reg_unchanged(self):
        """DOCUMENTED LIMITATION: if two captures construct the SAME
        logical register with DIFFERENT identifiers (one symbolic, one
        numeric for the same physical reg), the render-strings differ
        and identities differ. This case doesn't arise in practice
        because both captures consume the same kernel writer state, but
        it's worth pinning down expected behavior."""
        from Tensile.Components.ScheduleCapture import _canonical_render
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr

        sym = DSLoadB128(dst=vgpr("ValuA_X0_I0", 4), src=vgpr(0))
        num = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0))
        # Different render-strings -> different identities. This is by
        # design: a name-resolution table would be required to canonicalize
        # them as equivalent, and that's a known follow-up if a real
        # use case emerges.
        assert _canonical_render(sym) != _canonical_render(num)


# =============================================================================
# Phase-0 missing-node defense in diagnose_missing_edge
# =============================================================================


class TestDiagnoseMissingEdgeDefenses:
    def test_diagnose_missing_edge_with_missing_node_raises(self):
        """Bypass the entry assertion (test-only) and call diagnose with a
        node missing from the subject graph -> raises CaptureConsistencyError
        (NOT assert; survives python -O)."""
        from Tensile.Components.ScheduleCapture import (
            DataflowGraph, DataflowEdge, GraphNode, GraphPosition,
        )
        # A reference edge that references identities the subject doesn't have.
        ref_producer = GraphNode(
            identity=("LR", 1, ("vgpr", 8, 4), 64),
            position=GraphPosition(1, 0, 0),
            category="LRA0", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="LRA0[0]",
        )
        ref_consumer = GraphNode(
            identity=("MFMA", 1, ("vgpr", 0, 4), ("vgpr", 8, 2), ("vgpr", 32, 2)),
            position=GraphPosition(1, 2, 0),
            category="MFMA", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="MFMA",
        )
        ref_edge = DataflowEdge(
            producer=ref_producer, consumer=ref_consumer,
            register=None, edge_kind="raw_intrawave",
        )
        # Subject graph has no nodes — both lookups will fail.
        subj_graph = DataflowGraph(nodes={}, edges=[], captures={})
        with pytest.raises(CaptureConsistencyError):
            diagnose_missing_edge(ref_edge, subj_graph)
