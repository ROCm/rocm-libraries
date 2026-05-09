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
    CaptureConsistencyError,
    UnexplainedMissingEdgeError,
)
from Tensile.Components.CMSValidator import (
    OrderInvertedFailure,
    MissingWaitFailure,
    WaitInsufficientFailure,
    MissingBarrierFailure,
    build_dataflow_graph,
    compare_graphs,
    diagnose_missing_edge,
    validate_edge_wait_coverage,
    _DEFAULT_CDNA4_ARCH_PROFILE,
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
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
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

    def test_lcc_included_in_identity_set(self):
        """Loop-counter code (category 'LCC' — `SSubU32` + `SCmpEQI32`,
        per `Components/LCC_AUDIT.md`) is a full graph participant. Its
        per-instruction issue cycles contribute to
        `cumulative_issue_cycles` walks, which drives cross-body cycle
        counting. Identical captures with an LCC node both yield the
        same identity set including the LCC entry, and `compare_graphs`
        returns empty (no spurious LCC-only diff).

        Note: in real captures the default-side scheduler emits LCC
        outside `_loopBody` while CMS bakes it into the macro, so the
        two captures CAN differ in LCC presence. That asymmetry surfaces
        through `compare_graphs`; this test only pins the symmetric case
        to confirm LCC nodes participate in identity comparison.
        """
        from dataclasses import dataclass

        @dataclass
        class _LccInst:
            def __str__(self):
                return "s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1"

        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, WrappedInstruction, SlotKey, SLOT_KIND_MFMA,
        )

        def _build_ti():
            return TaggedInstruction(
                wrapped=WrappedInstruction(_LccInst()),
                category="LCC",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=0, sequence=0),
            )

        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            _build_ti(),
            make_mfma(0, 8, 32, slot=3, a_src_count=4),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            _build_ti(),
            make_mfma(0, 8, 32, slot=3, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        # Identity sets match — both include the LCC node.
        assert set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())
        assert any(ident[0] == "LCC" for ident in g_ref.nodes.keys()), (
            f"LCC nodes should participate in the identity set as of 2bu.2; "
            f"got cls tags {sorted({i[0] for i in g_ref.nodes.keys()})}"
        )
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
        MissingWaitFailure with nearby_other_counter_waits surfaced (the
        wrong-counter SWait at slot=1). The former
        WaitOnWrongCounterFailure was collapsed into MissingWaitFailure
        — see bead `hof`."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, vlcnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        subj = build_dataflow_graph(_wrap(subj_cap))
        failures = validate_edge_wait_coverage(subj)
        miss = [f for f in failures if isinstance(f, MissingWaitFailure)]
        assert miss, f"Expected MissingWaitFailure, got: {[type(f).__name__ for f in failures]}"
        wf = miss[0]
        assert wf.counter_kind == "dscnt"
        assert len(wf.nearby_wait_indices) >= 1

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
            from Tensile.Components.CMSValidator import Failure
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
        from Tensile.Components.CMSValidator import _identity_for
        lr = make_lr(8, 4, 64, slot=0, category="LRA0").wrapped.rocisa_inst
        ident = _identity_for(lr, BODY_LABEL_ML)
        assert ident[0] == "LR"             # class tag
        assert isinstance(ident[1], int)    # loop_index
        assert isinstance(ident[2], str)    # render-string
        # The render contains a vgpr reference and the LDS offset.
        # `make_lr` builds a real `rocisa.DSLoadB128`, which renders as
        # `ds_read_b128 v[8:11], v0 offset:64`.
        assert "v[" in ident[2]
        assert "offset:64" in ident[2]

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
        from Tensile.Components.CMSValidator import _identity_for
        from Tensile.Components.ScheduleCapture import _canonical_render
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
        """RESOLUTION CONTRACT (rocm-libraries-bb34): when two captures
        construct the SAME logical register with DIFFERENT identifiers
        (one symbolic, one numeric for the same physical reg), the
        in-stream `name_to_idx` lookup populated by
        `collect_regset_stream` resolves them to the SAME numeric
        byte-key in `_byte_keys_for_resource`. The render-strings still
        differ (this test pins the canonical-render contract for the
        identity-tuple path, which is unchanged); the byte-key
        equivalence is asserted separately to lock in the resolution
        behavior that wic4 / prp2 / oram depend on.
        """
        from Tensile.Components.ScheduleCapture import (
            _canonical_render, _byte_keys_for_resource,
        )
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr

        sym = DSLoadB128(dst=vgpr("ValuA_X0_I0", 4), src=vgpr(0))
        num = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0))
        # Render-strings still differ (the identity tuple is render-based;
        # operand-level resolution lives at edge-formation time, not at
        # canonical-render time).
        assert _canonical_render(sym) != _canonical_render(num)
        # When a RegSet binding for ValuA_X0_I0 -> 8 is in scope, the
        # symbolic and numeric refs to the same physical reg produce the
        # SAME numeric byte-keys — the load-bearing equivalence for
        # cross-form latest-writer dedup in build_dataflow_graph Phase 2.
        name_to_idx = {"ValuA_X0_I0": 8}
        sym_keys = _byte_keys_for_resource(sym.dst, name_to_idx=name_to_idx)
        num_keys = _byte_keys_for_resource(num.dst, name_to_idx=name_to_idx)
        assert sym_keys == num_keys
        assert sym_keys == (("v", 8), ("v", 9), ("v", 10), ("v", 11))

    def test_category_overrides_isinstance_class_tag(self):
        """A pack-categorized MFMAInstruction (TF32 bf16-emulation pattern)
        must not get cls_tag='MFMA' in the identity tuple — that confuses
        compare_graphs into reporting it as a missing main-loop MFMA when
        the two captures see different counts of pack-MFMAs.

        With category provided, _identity_for must use the category as the
        discriminator: PackA{u}/PackB{u} -> 'PACK', LRA{u} -> 'LR', etc.
        """
        from Tensile.Components.CMSValidator import (
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
# In-stream RegSet resolution (rocm-libraries-bb34, Option (e))
# =============================================================================
# These tests pin the resolution branch in `_byte_keys_for_resource`. The
# branch makes symbolic operand references resolve to the same numeric
# byte-key as the corresponding numeric reference, when a RegSet binding
# for the symbolic name is in scope. Without resolution (no name_to_idx),
# the legacy symbolic-keying path is preserved — that's what makes the
# branch additive rather than a behavior change for symbolic-only refs.


class TestRegSetResolution:
    def test_symbolic_resolves_via_regset_to_numeric_byte_key(self):
        """A symbolic operand whose bare name is in `name_to_idx` produces
        the SAME byte-key tuple as a direct numeric reference to the
        resolved index. This is the core resolution contract."""
        from Tensile.Components.ScheduleCapture import _byte_keys_for_resource
        from rocisa.container import vgpr

        sym = vgpr("ValuA_T0_I0", 4)
        num = vgpr(76, 4)
        name_to_idx = {"ValuA_T0_I0": 76}
        sym_keys = _byte_keys_for_resource(sym, name_to_idx=name_to_idx)
        num_keys = _byte_keys_for_resource(num, name_to_idx=name_to_idx)
        assert sym_keys == num_keys
        assert sym_keys == (("v", 76), ("v", 77), ("v", 78), ("v", 79))

    def test_symbolic_not_in_table_falls_through_unchanged(self):
        """A symbolic operand whose name is NOT in the table preserves
        the legacy symbolic-keying behavior — three-tuple (rt, name,
        offset+i). This is the load-bearing safety property: the
        resolution branch is additive only."""
        from Tensile.Components.ScheduleCapture import _byte_keys_for_resource
        from rocisa.container import vgpr

        sym = vgpr("LocalReadAddrA", 1)
        # Empty / None / missing-name table all preserve symbolic keys.
        for table in (None, {}, {"SomeOtherName": 42}):
            keys = _byte_keys_for_resource(sym, name_to_idx=table)
            assert keys == (("v", "LocalReadAddrA", 0),)

    def test_two_operands_same_phys_reg_collapse_after_resolution(self):
        """Two operands referencing the SAME physical register under
        different forms (one symbolic, one numeric) produce identical
        byte-keys after resolution — this is the latest-writer dedup
        invariant that makes intra-graph 7a not surface as false-positive
        edges."""
        from Tensile.Components.ScheduleCapture import _byte_keys_for_resource
        from rocisa.container import vgpr

        # Producer writes vgpr(76, 4); consumer reads vgprValuA_T0_I0(4).
        write_resource = vgpr(76, 4)
        read_resource = vgpr("ValuA_T0_I0", 4)
        name_to_idx = {"ValuA_T0_I0": 76}

        write_keys = set(_byte_keys_for_resource(write_resource, name_to_idx=name_to_idx))
        read_keys = set(_byte_keys_for_resource(read_resource, name_to_idx=name_to_idx))
        # Full overlap — every byte the consumer reads, the producer
        # wrote. Without resolution, the read keys would be
        # ("v", "ValuA_T0_I0", 0..3) and miss the latest-writer entries
        # at ("v", 76..79), yielding a phantom missing edge.
        assert write_keys == read_keys
        assert read_keys == {("v", 76), ("v", 77), ("v", 78), ("v", 79)}

    def test_symbolic_with_offset_resolves_with_offset_added(self):
        """A symbolic operand with a getTotalOffsets() offset resolves
        as `name_to_idx[bare] + offset + i` per byte. Mirrors the
        numeric-form `regIdx + i` exactly.
        """
        from Tensile.Components.ScheduleCapture import _byte_keys_for_resource
        from rocisa.container import vgpr

        sym = vgpr("Base", 2)
        # Inject a non-zero offset onto the symbolic vgpr's regName —
        # the resolution branch reads .name and .getTotalOffsets() off
        # the regName object, and addOffset(n) makes
        # getTotalOffsets() return n.
        sym.regName.addOffset(3)
        name_to_idx = {"Base": 100}
        keys = _byte_keys_for_resource(sym, name_to_idx=name_to_idx)
        # Expected: 100 + 3 + 0, 100 + 3 + 1
        assert keys == (("v", 103), ("v", 104))

    def test_collect_regset_stream_value_and_ref_chain(self):
        """`collect_regset_stream` resolves both value-form
        (`name -> int`) and ref-form (`name -> ref + offset`) RegSet
        bindings, mirroring rocisa's `RegSet::setIdx` semantics. The
        ref-form chain is replayed in passes so a binding can depend on
        an anchor declared earlier OR later in emission order — three
        passes are sufficient for production kernels (verified
        empirically against TF32/BF16/FP16/FP8 captures)."""
        from Tensile.Components.ScheduleCapture import collect_regset_stream
        from rocisa.code import Module, RegSet

        # Stand-in writer with a `module` that holds RegSet directives
        # in the same shape as KernelWriterAssembly emits them.
        class _FakeStates:
            class _MXSA:
                startVgprValu = None
            mxsa = _MXSA()

        class _FakeWriter:
            states = _FakeStates()
            sgprs = {"KernArgAddress": 0, "WorkGroup0": 2}

            def __init__(self):
                self.module = Module("kernel-body")
                # Value-form: vgprValuA_T0_I0 -> idx 76.
                self.module.add(RegSet("v", "vgprValuA_T0_I0", 76, 0))
                # Ref-form: vgprValuB_X0_I0_BASE -> vgprValuA_T0_I0 + 16.
                self.module.add(RegSet("v", "vgprValuB_X0_I0_BASE",
                                       "vgprValuA_T0_I0", 16))
                # Two-level ref-form: vgprValuB_X0_I0 ->
                # vgprValuB_X0_I0_BASE + 0.
                self.module.add(RegSet("v", "vgprValuB_X0_I0",
                                       "vgprValuB_X0_I0_BASE", 0))

        writer = _FakeWriter()
        n2i = collect_regset_stream(writer)
        # Vgpr resolutions.
        assert n2i["ValuA_T0_I0"] == 76
        assert n2i["ValuB_X0_I0_BASE"] == 92
        assert n2i["ValuB_X0_I0"] == 92
        # Sgpr pool surfaces under bare names too.
        assert n2i["KernArgAddress"] == 0
        assert n2i["WorkGroup0"] == 2


# =============================================================================
# Phase-0 missing-node defense in diagnose_missing_edge
# =============================================================================


class TestDiagnoseMissingEdgeDefenses:
    def test_diagnose_missing_edge_with_missing_node_raises(self):
        """Bypass the entry assertion (test-only) and call diagnose with a
        node missing from the subject graph -> raises CaptureConsistencyError
        (NOT assert; survives python -O)."""
        from Tensile.Components.ScheduleCapture import SchedulePosition
        from Tensile.Components.CMSValidator import (
            DataflowEdge, DataflowGraph, GraphNode,
        )
        # A reference edge that references identities the subject doesn't have.
        # Use the canonical short-form regType "v" (matches real rocisa).
        ref_producer = GraphNode(
            identity=("LR", 1, ("v", 8, 4), 64),
            position=SchedulePosition(1, 0),
            category="LRA0", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="LRA0[0]",
        )
        ref_consumer = GraphNode(
            identity=("MFMA", 1, ("v", 0, 4), ("v", 8, 2), ("v", 32, 2)),
            position=SchedulePosition(1, 2),
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
# Reversed GRIncA scalar chain — detected via dataflow graph (post wx9.10)
# =============================================================================
#
# Historical context: a structural rule `verify_ascending_order` previously
# caught CMS schedules that emitted category instructions out of order — e.g.
# GRIncA at vmfma_indices [3,2,1,0] instead of [0,1,2,3]. The scheduler walks
# the optSchedule list left-to-right with no sort (CustomSchedule.py:400-423),
# so reversal silently emits the chain in reverse order.
#
# A reversed GRIncA chain has real dataflow violations:
#   #2 SCSelectB32 writes incLower; #4 SAddU32 reads incLower (RAW)
#   #6 SSubU32 writes ShadowLimit+0; #9 SCSelectB32 reads ShadowLimit+0 (RAW)
# Reversal puts the consumers before the producers — wrong code.
#
# `_writes()` / `_reads()` model scalar ALU registers, so these reversals
# form RAW edges that compare_graphs flips into typed
# OrderInvertedFailures. The (now-removed) `verify_ascending_order`
# structural rule used to cover this; the test below is the replacement.


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
            TaggedInstruction, WrappedInstruction, SlotKey, SLOT_KIND_MFMA,
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
                wrapped=WrappedInstruction(inst),
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

class TestVgprChainReorderDetection:
    """Coverage proof for non-GRIncA categories.

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
            TaggedInstruction, WrappedInstruction, SlotKey, SLOT_KIND_MFMA,
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
                wrapped=WrappedInstruction(inst),
                category="LRA0",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=base_slot, sequence=seq),
            ))
        return tagged

    def test_reversed_vgpr_chain_should_be_detected(self):
        """A reversed vgpr RAW chain (different category, different op
        family from GRIncA) must also surface as `OrderInvertedFailure`.
        Proves the scalar-ALU coverage isn't GRIncA-specific.
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


# =============================================================================
# Allocation invariance + legitimate-reorder + OrderInverted (rocm-libraries-wx9.3)
# =============================================================================
# The Phase 2 contract (memo §6.1, clarified 2026-05-08):
#
#   edge_key = (src_role, src_position, sink_role, sink_position,
#               edge_kind, intra_operand_byte_offset)
#
# `intra_operand_byte_offset` is the offset WITHIN the connected operand
# (0..N-1 for an N-byte read), NOT an absolute physical-register byte-key.
# This is what makes the edge identity allocation-invariant: two graphs
# with the same topology but different physical register allocations
# produce identical edge keys.


class TestEdgeIdentityAllocationInvariance:
    """Edge identities are allocation-invariant.

    Two graphs with the same topology but different absolute register
    allocations (e.g. LR's destination is `vgpr(8, 4)` in one and
    `vgpr(12, 4)` in the other, with the consuming MFMA's `a` input
    pointed at the same vgpr accordingly) must produce the SAME edge
    identity tuples.
    """

    def test_lr_to_mfma_identities_equal_under_allocation_change(self):
        # Allocation A: LR writes vgpr(8,4); MFMA reads vgpr(8,4).
        cap_a = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Allocation B: LR writes vgpr(12,4); MFMA reads vgpr(12,4).
        cap_b = make_capture(BODY_LABEL_ML, [
            make_lr(12, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 12, 32, slot=2, a_src_count=4),
        ])
        g_a = build_dataflow_graph(_wrap(cap_a))
        g_b = build_dataflow_graph(_wrap(cap_b))

        # Edge identities must match exactly. The byte_offset component must
        # encode intra-operand position (0..3 for the 4-byte LR->MFMA dataflow)
        # — NOT absolute physical-register byte-keys (('v', 8) vs ('v', 12)).
        assert g_a.edge_keys() == g_b.edge_keys(), (
            "Edge identities must be allocation-invariant: same topology, "
            "different register allocations -> same edge keys.\n"
            f"A: {sorted(g_a.edge_keys())}\nB: {sorted(g_b.edge_keys())}"
        )
        # Sanity: the keys are not empty (we actually built edges).
        assert len(g_a.edge_keys()) > 0


class TestLegitimateCmsReorderNoFailure:
    """A legitimate CMS reorder (producer-before-consumer relative order
    preserved, but at different stream positions) produces no failures.

    The classifier-pipeline branch in `diagnose_missing_edge` returns []
    when both endpoints exist in subj with the same relative order as in
    ref, even when the edge identity (which now includes positions) doesn't
    match between graphs.
    """

    def test_reorder_preserving_producer_before_consumer_no_failures(self):
        # Reference: LR at slot=0, SWait at slot=1, MFMA at slot=2.
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 8, 32, slot=2, a_src_count=4),
        ])
        # Subject: same instructions, MFMA pushed out to slot=5; LR stays
        # at slot=0; SWait moves to slot=4. Producer still before consumer
        # in subj (slot 0 < slot 5). The CMS-reordered edge has different
        # positions on subj side — but order is preserved.
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=4, dscnt=0),
            make_mfma(0, 8, 32, slot=5, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        # Legitimate CMS reorder -> no failures.
        failures = compare_graphs(g_ref, g_subj)
        assert failures == [], (
            "Legitimate CMS reorder (producer-before-consumer preserved) "
            f"must yield no failures; got: {[type(f).__name__ for f in failures]}"
        )


class TestGenuineOrderInversionStillDetected:
    """A genuine producer/consumer inversion (consumer-before-producer in
    subj, producer-before-consumer in ref) must still surface
    `OrderInvertedFailure`.
    """

    def test_inverted_lr_mfma_pair_emits_order_inverted(self):
        # Reference: LR writes v100 then MFMA reads it.
        # Use a vgpr ALU chain so reordering doesn't violate sync semantics
        # (LR/MFMA reorder would also fire MissingWait). Use a within-body
        # pure-ALU producer/consumer pair which Phase 1 OrderInverted catches.
        from rocisa.instruction import VAddU32
        from rocisa.container import vgpr
        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, WrappedInstruction, SlotKey, SLOT_KIND_MFMA,
        )

        def _vadd_pair(producer_seq, consumer_seq):
            # producer: writes v100. consumer: reads v100 -> writes v101.
            prod = VAddU32(dst=vgpr(100), src0=vgpr(50), src1=vgpr(51))
            cons = VAddU32(dst=vgpr(101), src0=vgpr(100), src1=vgpr(52))
            tagged_p = TaggedInstruction(
                wrapped=WrappedInstruction(prod), category="LRA0",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=0, sequence=producer_seq),
            )
            tagged_c = TaggedInstruction(
                wrapped=WrappedInstruction(cons), category="LRA0",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=0, sequence=consumer_seq),
            )
            return tagged_p, tagged_c

        # Reference: producer at sequence=0, consumer at sequence=1.
        ref_p, ref_c = _vadd_pair(0, 1)
        ref_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            ref_p, ref_c,
            make_swait(slot=2, dscnt=0),
            make_mfma(0, 8, 32, slot=3, a_src_count=4),
        ])
        # Subject: SAME nodes (so identities match), but stream positions
        # inverted: producer at sequence=1, consumer at sequence=0. This is
        # a real reorder of a real dependency.
        subj_p, subj_c = _vadd_pair(1, 0)
        subj_cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            subj_c, subj_p,
            make_swait(slot=2, dscnt=0),
            make_mfma(0, 8, 32, slot=3, a_src_count=4),
        ])
        g_ref = build_dataflow_graph(_wrap(ref_cap))
        g_subj = build_dataflow_graph(_wrap(subj_cap))
        failures = compare_graphs(g_ref, g_subj)
        assert any(isinstance(f, OrderInvertedFailure) for f in failures), (
            "Genuine producer/consumer inversion must surface "
            f"OrderInvertedFailure; got: {[type(f).__name__ for f in failures]}"
        )
