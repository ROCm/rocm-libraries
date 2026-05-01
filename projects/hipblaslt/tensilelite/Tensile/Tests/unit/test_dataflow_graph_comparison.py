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
    validate_edge_wait_coverage,
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

    def test_redundant_swaits_and_barriers_in_subject_no_failures(self):
        """CMS may emit arbitrary numbers of redundant SWaits / SBarriers
        and still be correct. The cross-graph identity set must not include
        them, so that subject can have MORE waits/barriers than reference
        without producing 'in subject not reference' diffs that would
        previously have surfaced as identity-coverage warnings.

        Per-edge correctness is unaffected: the reference's LR->MFMA edge
        still finds a covering wait in the subject.
        """
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject: identical dataflow + several redundant waits & barriers.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),         # primary drain
            make_swait(slot=2, dscnt=0),         # redundant
            make_sbarrier(slot=3),               # redundant safety barrier
            make_swait(slot=4, vlcnt=0),         # different counter, no-op here
            make_mfma(0, 8, 32, slot=5, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        # Identity sets should differ ONLY on producer/consumer nodes,
        # which here are identical -> no failures, no identity mismatch.
        assert compare_graphs(g_ref, g_subj) == []
        # Defensive: the SWait/SBarrier nodes must not have leaked into
        # nodes_by_identity. Subject and reference must have the SAME
        # identity-set sizes despite subject having 3 extra sync ops.
        assert set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())

    def test_lcc_excluded_from_identity_set(self):
        """Loop-counter code (category 'LCC') is emitted by CMS inside the
        macro body but by the default-side scheduler OUTSIDE _loopBody (in
        closeLoop). The two schedulers capture different scopes for LCC.
        The identity-set comparison must exclude cls=LCC so the asymmetry
        doesn't surface as a CMS-only diff.

        Use a stand-in instruction class — cls=LCC comes from the category
        mapping (LCC -> 'LCC'), so any inst type works.
        """
        from dataclasses import dataclass

        @dataclass
        class _LccInst:
            def __str__(self):
                return "s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1"

        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, SlotKey, SLOT_KIND_MFMA,
        )
        ti_lcc = TaggedInstruction(
            inst=_LccInst(),
            category="LCC",
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=0, sequence=0),
        )
        # Reference has the LCC; subject doesn't (mirrors CMS vs default).
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            ti_lcc,
            make_mfma(0, 8, 32, slot=3, a_src_count=4),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=3, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        # Identity sets must match (LCC excluded from both).
        assert set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())
        # No LCC in either identity set.
        assert not any(ident[0] == "LCC" for ident in g_ref.nodes.keys())
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
        """Schedule has the LR and the consuming MFMA but no SWait between
        them. With register-resolution edges, the LR->MFMA edge forms in
        BOTH ref and subject (compare_graphs sees no diff). The validator
        validate_edge_wait_coverage flags the subject's missing wait.
        """
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = validate_edge_wait_coverage(subj)
        assert any(isinstance(f, MissingWaitFailure) for f in failures)
        mw = next(f for f in failures if isinstance(f, MissingWaitFailure))
        assert mw.counter_kind == "dscnt"

    def test_wait_on_wrong_counter_diagnosis(self):
        """Schedule has SWait(vlcnt=0) where SWait(dscnt=0) is needed for
        an LR -> MFMA edge. validate_edge_wait_coverage emits
        WaitOnWrongCounterFailure."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, vlcnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = validate_edge_wait_coverage(subj)
        assert any(isinstance(f, WaitOnWrongCounterFailure) for f in failures)
        wf = next(f for f in failures if isinstance(f, WaitOnWrongCounterFailure))
        assert wf.expected_counter == "dscnt"
        assert len(wf.wrong_counter_waits) >= 1

    def test_wait_insufficient_diagnosis(self):
        """Schedule has 5 LRs in flight + SWait(dscnt=2) which only drains
        the oldest 3. The MFMA reads the youngest LR's register (LR_e at
        regIdx=24) — the LR_e -> MFMA edge forms by register resolution,
        and validate_edge_wait_coverage emits WaitInsufficientFailure
        because LR_e's queue position (4) exceeds the SWait's cap (2)."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),     # LR_a
            make_lr(12, 4, 80, slot=1, category="LRA0"),    # LR_b
            make_lr(16, 4, 96, slot=2, category="LRA0"),    # LR_c
            make_lr(20, 4, 112, slot=3, category="LRA0"),   # LR_d
            make_lr(24, 4, 128, slot=4, category="LRA0"),   # LR_e
            make_swait(slot=5, dscnt=2),                    # drains oldest 3
            make_mfma(0, 24, 32, slot=6, a_src_count=4),    # reads LR_e
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = validate_edge_wait_coverage(subj)
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
        # Synthetic fixtures construct RegisterContainer with regType="v"
        # and a symbolic RegName (e.g. "v8"), so the render is something
        # like `v[vgprv8:vgprv8+3]`.
        assert "v[" in ident[2]
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

    def test_category_overrides_isinstance_class_tag(self):
        """A pack-categorized MFMAInstruction (TF32 bf16-emulation pattern)
        must not get cls_tag='MFMA' in the identity tuple — that confuses
        compare_graphs into reporting it as a missing main-loop MFMA when
        the two captures see different counts of pack-MFMAs.

        With category provided, _identity_for must use the category as the
        discriminator: PackA{u}/PackB{u} -> 'PACK', LRA{u} -> 'LR', etc.
        """
        from Tensile.Components.ScheduleCapture import (
            _identity_for, _class_tag_from_category,
        )
        from rocisa.instruction import MFMAInstruction
        from rocisa.container import vgpr
        from rocisa.enum import InstType

        pack_mfma = MFMAInstruction(
            instType=InstType.INST_BF16, accType=InstType.INST_F32,
            variant=[4, 4, 4, 16], mfma1k=False,
            acc=vgpr("ValuA_T0_I0", 4),
            a=vgpr(74, 2),
            b=vgpr("ValuA_X0_I0", 2),
        )
        # Without category -> isinstance fallback says 'MFMA'.
        assert _identity_for(pack_mfma, BODY_LABEL_ML)[0] == "MFMA"
        # With pack category -> 'PACK'.
        assert _identity_for(pack_mfma, BODY_LABEL_ML, category="PackA0")[0] == "PACK"
        assert _identity_for(pack_mfma, BODY_LABEL_ML, category="PackB3")[0] == "PACK"
        # Sanity-check the underlying mapping for the other categories.
        assert _class_tag_from_category("LRA0", pack_mfma) == "LR"
        assert _class_tag_from_category("LRB3", pack_mfma) == "LR"
        assert _class_tag_from_category("LWA",  pack_mfma) == "LW"
        assert _class_tag_from_category("GRA",  pack_mfma) == "GR"
        assert _class_tag_from_category("GRIncA", pack_mfma) == "GRINC"
        assert _class_tag_from_category("LRSA", pack_mfma) == "LRS"
        assert _class_tag_from_category("LWSA", pack_mfma) == "LWS"
        assert _class_tag_from_category("LCC",  pack_mfma) == "LCC"
        # category="SYNC" lumps SWaitCnt and SBarrier together (capture-side
        # bucket); fall back to isinstance to disambiguate.
        assert _class_tag_from_category("SYNC", pack_mfma) == "MFMA"  # isinstance fallback
        from rocisa.instruction import SWaitCnt, SBarrier
        sw = SWaitCnt(comment="t")
        sb = SBarrier()
        assert _class_tag_from_category("SYNC", sw) == "SWAIT"
        assert _class_tag_from_category("SYNC", sb) == "SBARRIER"
        assert _class_tag_from_category("BARRIER", pack_mfma) == "SBARRIER"
        # Unrecognized category falls back to isinstance.
        assert _class_tag_from_category("UNKNOWN", pack_mfma) == "MFMA"
        assert _class_tag_from_category(None, pack_mfma) == "MFMA"


# =============================================================================
# Phase-0 missing-node defense in diagnose_missing_edge
# =============================================================================


class TestDiagnoseMissingEdgeDefenses:
    def test_diagnose_missing_edge_with_missing_node_raises(self):
        """Bypass the entry assertion (test-only) and call diagnose with a
        node missing from the subject graph -> raises CaptureConsistencyError
        (NOT assert; survives python -O)."""
        from Tensile.Components.ScheduleCapture import (
            DataflowGraph, DataflowEdge, GraphNode, SchedulePosition,
        )
        # A reference edge that references identities the subject doesn't have.
        # Use the canonical short-form regType "v" (matches real rocisa).
        ref_producer = GraphNode(
            identity=("LR", 1, ("v", 8, 4), 64),
            position=SchedulePosition(1, 0, 0),
            category="LRA0", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="LRA0[0]",
        )
        ref_consumer = GraphNode(
            identity=("MFMA", 1, ("v", 0, 4), ("v", 8, 2), ("v", 32, 2)),
            position=SchedulePosition(1, 2, 0),
            category="MFMA", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="MFMA",
        )
        ref_edge = DataflowEdge(
            producer=ref_producer, consumer=ref_consumer,
            resource=None, edge_kind="raw_intrawave",
        )
        # Subject graph has no nodes — both lookups will fail.
        subj_graph = DataflowGraph(nodes={}, edges=[], captures={})
        with pytest.raises(CaptureConsistencyError):
            diagnose_missing_edge(ref_edge, subj_graph)


# =============================================================================
# Coverage gap — reversed GRIncA scalar chain is invisible to compare_graphs
# =============================================================================
#
# `verify_ascending_order` (CMSValidator.py:3345) catches CMS schedules that
# emit category instructions out of order — e.g. GRIncA at vmfma_indices
# [3,2,1,0] instead of [0,1,2,3]. The scheduler walks the optSchedule list
# left-to-right with no sort (CustomSchedule.py:400-423), so reversal silently
# emits the chain in reverse order.
#
# A reversed GRIncA chain has real dataflow violations:
#   #2 SCSelectB32 writes incLower; #4 SAddU32 reads incLower (RAW)
#   #6 SSubU32 writes ShadowLimit+0; #9 SCSelectB32 reads ShadowLimit+0 (RAW)
# Reversal puts the consumers before the producers — wrong code.
#
# Today the dataflow graph cannot see this. `_writes()` and `_reads()` in
# ScheduleCapture.py only handle LR/GR/LW/MFMA — scalar ALU instructions
# return [] from both, so they form no edges. compare_graphs therefore
# treats reversed and normal GRIncA as identical graphs.
#
# This test asserts the *desired* behavior (compare_graphs detects the
# reversal) so the failure documents the gap. If/when `_writes` and `_reads`
# are extended to recognize scalar ALU registers, this test starts passing
# and `verify_ascending_order` becomes a redundant defense.


class TestGRIncReorderDetection:
    """Document that compare_graphs is blind to reversed GRIncA chains.

    Builds a reference capture with a normal-order GRInc-like scalar chain
    and a subject capture with the same instructions reversed. Under correct
    behavior, compare_graphs would surface either an OrderInvertedFailure on
    the intra-chain RAW edges or a missing-edge diagnostic. Today it returns
    [] because scalar ALU writes/reads aren't tracked.
    """

    def _make_grinc_chain(self, *, base_slot: int, reversed_order: bool):
        """Build a 6-instruction GRIncA-like chain with intra-chain RAW deps.

        Real GRIncA (with Use64bShadowLimit=1) emits 9 SCC-coupled
        instructions. We model the simplest 4-step subset that has clear
        register-level RAW edges:

          #1 SCSelectB32 dst=s100 (incLower)            — producer of incLower
          #2 SCSelectB32 dst=s101 (incUpper)            — producer of incUpper
          #3 SAddU32     dst=s10, src1=s100             — RAW from #1
          #4 SAddCU32    dst=s11, src1=s101             — RAW from #2
          #5 SSubU32     dst=s20, src1=s100             — RAW from #1
          #6 SSubBU32    dst=s21, src1=s101             — RAW from #2

        category="GRIncA" tags every instruction; their stream positions
        (SchedulePosition.sub_index) distinguish them within the slot.
        """
        from rocisa.instruction import (
            SCSelectB32, SAddU32, SAddCU32, SSubU32, SSubBU32,
        )
        from rocisa.container import sgpr
        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, SlotKey, SLOT_KIND_MFMA,
        )

        # Producers of incLower / incUpper.
        i1 = SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1), src1=sgpr(51, 1))
        i2 = SCSelectB32(dst=sgpr(101, 1), src0=sgpr(52, 1), src1=sgpr(53, 1))
        # Carry chain: SAddU32 reads incLower; SAddCU32 reads incUpper.
        i3 = SAddU32(dst=sgpr(10, 1), src0=sgpr(10, 1), src1=sgpr(100, 1))
        i4 = SAddCU32(dst=sgpr(11, 1), src0=sgpr(11, 1), src1=sgpr(101, 1))
        # Borrow chain: SSubU32 reads incLower; SSubBU32 reads incUpper.
        i5 = SSubU32(dst=sgpr(20, 1), src0=sgpr(20, 1), src1=sgpr(100, 1))
        i6 = SSubBU32(dst=sgpr(21, 1), src0=sgpr(21, 1), src1=sgpr(101, 1))

        chain = [i1, i2, i3, i4, i5, i6]
        if reversed_order:
            chain = list(reversed(chain))

        tagged = []
        for seq, inst in enumerate(chain):
            tagged.append(TaggedInstruction(
                inst=inst,
                category="GRIncA",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=base_slot, sequence=seq),
            ))
        return tagged

    def test_reversed_grinc_chain_should_be_detected(self):
        # Reference: an LR -> SWait -> MFMA baseline plus a normal-order
        # GRInc-like chain in the same body.
        ref_chain = self._make_grinc_chain(base_slot=0, reversed_order=False)
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            *ref_chain,
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])

        # Subject: same instructions, GRInc chain reversed in stream order.
        # The 4 RAW edges within the chain (incLower -> add/sub, incUpper ->
        # addc/subb) now have producer issued AFTER consumer — should
        # produce OrderInvertedFailure or missing-edge diagnostic.
        subj_chain = self._make_grinc_chain(base_slot=0, reversed_order=True)
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            *subj_chain,
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])

        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(g_ref, g_subj)

        # The DESIRED behavior: graph detects intra-chain order inversion.
        assert failures, (
            "compare_graphs should have flagged the reversed GRIncA chain "
            "(producer SCSelectB32 issued after consumer SAddU32) but "
            "returned no failures — the dataflow graph is blind to scalar "
            "ALU register dependencies."
        )
        assert any(isinstance(f, OrderInvertedFailure) for f in failures), (
            "Expected at least one OrderInvertedFailure for the reversed "
            f"GRIncA chain; got {[type(f).__name__ for f in failures]}."
        )

    def test_ascending_order_rule_does_catch_reversed_grinc(self):
        """Companion: prove that the structural rule `verify_ascending_order`
        DOES catch the same kind of error at the optSchedule level. This is
        why the rule remains necessary even after the dataflow-graph
        comparison is in place."""
        from Tensile.Components.CMSValidator import verify_ascending_order
        from Tensile.Components.ScheduleCapture import OutOfOrderSequenceFailure

        class _FakeSchedInfo:
            optSchedule = {"GRIncA": [[5, 4, 3, 2, 1, 0]]}

        ok, msg = verify_ascending_order(_FakeSchedInfo(), context=None, code_path=0)
        assert not ok, "verify_ascending_order should have rejected reversed GRIncA"
        # The rule constructs an OutOfOrderSequenceFailure internally; here
        # we just confirm the message identifies the offending key.
        assert "GRIncA" in msg
        assert "0" in msg  # the bad value


class TestVgprChainReorderDetection:
    """Coverage proof for non-GRIncA categories — bead wx9.10 task #4.

    Mirrors `TestGRIncReorderDetection` but exercises a vgpr ALU chain
    with a different category tag (`LRA0`, modeling a Local Read Address
    advancement sequence). The point is to demonstrate that the
    scalar-ALU coverage in `_GenericALURule` is operand-shape-driven and
    not GRIncA-specific: any reversed RAW chain across vgpr or sgpr
    registers should be caught by `compare_graphs` once the producer is
    issued after the consumer.
    """

    def _make_vgpr_chain(self, *, base_slot: int, reversed_order: bool):
        """3-instruction vgpr RAW chain.

          #1 VAddU32   dst=v100, src0=v50, src1=v51       — produces v100
          #2 VAddU32   dst=v101, src0=v100, src1=v52      — RAW from #1
          #3 VAddU32   dst=v102, src0=v101, src1=v53      — RAW from #2
        """
        from rocisa.instruction import VAddU32
        from rocisa.container import vgpr
        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, SlotKey, SLOT_KIND_MFMA,
        )

        i1 = VAddU32(dst=vgpr(100), src0=vgpr(50), src1=vgpr(51))
        i2 = VAddU32(dst=vgpr(101), src0=vgpr(100), src1=vgpr(52))
        i3 = VAddU32(dst=vgpr(102), src0=vgpr(101), src1=vgpr(53))

        chain = [i1, i2, i3]
        if reversed_order:
            chain = list(reversed(chain))

        tagged = []
        for seq, inst in enumerate(chain):
            tagged.append(TaggedInstruction(
                inst=inst,
                category="LRA0",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=base_slot, sequence=seq),
            ))
        return tagged

    def test_reversed_vgpr_chain_should_be_detected(self):
        """A reversed vgpr RAW chain (different category, different op
        family from GRIncA) must also surface as `OrderInvertedFailure`.
        Proves the wx9.10 coverage isn't GRIncA-specific.
        """
        ref_chain = self._make_vgpr_chain(base_slot=0, reversed_order=False)
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            *ref_chain,
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj_chain = self._make_vgpr_chain(base_slot=0, reversed_order=True)
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            *subj_chain,
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])

        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(g_ref, g_subj)

        assert failures, (
            "compare_graphs should have flagged the reversed vgpr chain "
            "(producer VAddU32 issued after consumer VAddU32) but "
            "returned no failures — coverage of vgpr ALU dependencies "
            "regressed."
        )
        assert any(isinstance(f, OrderInvertedFailure) for f in failures), (
            "Expected at least one OrderInvertedFailure for the reversed "
            f"vgpr chain; got {[type(f).__name__ for f in failures]}."
        )
