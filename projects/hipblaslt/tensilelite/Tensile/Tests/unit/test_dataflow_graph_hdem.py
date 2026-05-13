################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""rocm-libraries-hdem (Approach A + Approach E) regression tests.

Pins the body-blindness contract introduced by `rocm-libraries-hdem`:

* Approach A drops `loop_index` from `TaggedInstruction.identity_for`'s
  tuple, so identity is now `(canonical_render, emission_ordinal)` —
  body-blind by construction.
* Approach E rewrites `DataflowGraph.edge_keys` to use
  `(producer.identity, consumer.identity, edge_kind,
  intra_operand_byte_offset, src_operand_slot, sink_operand_slot)` —
  body-blind via identity-collapse on both endpoints.

Together the two changes close the cross-body pipelining false-positive
that `compare_graphs` fired pre-hdem on the motivating UsePLRPack case
(Pack code in PRO body under CMS vs ML-1 body under default; identical
dataflow, different bodies).

See `Tensile/Components/ORAM1_PRINCIPLED_APPROACH_INVESTIGATION.md`
(§4 / §6 / §7) and `Tensile/Components/HDEM_IMPLEMENTATION.md` for
the full design and verification context.

Test classes (per bead spec test #1-#3):

  TestCrossBodyPipeliningMatches
    Synthetic cross-body Pack pipelining: Pack producer in PRO on the
    reference side, same Pack producer in ML on the subject side,
    feeding the same MFMA. Pre-hdem these had different identities AND
    different edge-keys. Post-hdem they collapse and `compare_graphs`
    returns []. (Pre-fix regression — would fail under the pre-hdem
    edge_keys + identity_for shapes.)

  TestCrossIterationRotationStillDistinct
    Same body, two iterations of register rotation: iter N writes vgpr+0,
    iter N+1 writes vgpr+16. Different canonical_render -> different
    identities -> different edge-keys. Both reference and subject have
    both iterations, so `compare_graphs` returns []. Pin that A+E does
    not break the rotation case (and that Approach D is not required).

  TestCrossBodyExtraWriteSurfaces
    A genuinely-extra Pack producer in one capture but not the other,
    constructed so its identity does NOT collapse with anything in the
    subject graph. Asserts the divergence surfaces as a non-empty
    Failure list — the residual-detection path the bead spec calls
    out (test #3).
"""

import pytest

from Tensile.Components.CMSValidator import (
    GraphNode,
    build_dataflow_graph,
    compare_graphs,
    _DEFAULT_CDNA4_ARCH_PROFILE,
)
from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    BODY_LABEL_PROLOGUE,
    FourPartCapture,
)
from dataflow_fixtures import (
    make_capture,
    make_lr,
    make_mfma,
    make_swait,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap_with_pro(prologue_cap, ml_cap):
    """Build a FourPartCapture with a non-empty prologue + ML body and
    filler MFMAs in ML-1 / NGL / NLL so build_dataflow_graph accepts
    every required body.
    """
    return FourPartCapture(
        prologue=prologue_cap,
        main_loop_prev={0: make_capture(BODY_LABEL_ML_PREV, [
            make_mfma(0, 200, 232, slot=0, a_src_count=1)])},
        main_loop={0: ml_cap},
        n_gl={0: make_capture(BODY_LABEL_NGL, [
            make_mfma(0, 208, 240, slot=0, a_src_count=1)])},
        n_ll={0: make_capture(BODY_LABEL_NLL, [
            make_mfma(0, 216, 248, slot=0, a_src_count=1)])},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _wrap_no_pro(ml_cap):
    """FourPartCapture with no prologue, same filler bodies."""
    return FourPartCapture(
        prologue=None,
        main_loop_prev={0: make_capture(BODY_LABEL_ML_PREV, [
            make_mfma(0, 200, 232, slot=0, a_src_count=1)])},
        main_loop={0: ml_cap},
        n_gl={0: make_capture(BODY_LABEL_NGL, [
            make_mfma(0, 208, 240, slot=0, a_src_count=1)])},
        n_ll={0: make_capture(BODY_LABEL_NLL, [
            make_mfma(0, 216, 248, slot=0, a_src_count=1)])},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


# =============================================================================
# Test #1 (per bead spec) — cross-body pipelining matches under A+E
# =============================================================================


class TestCrossBodyPipeliningMatches:
    """Pre-fix regression — would fail under the pre-hdem edge_keys.

    Reference: LR -> SWait -> Pack(PRO body) -> MFMA(ML body).
    Subject:   LR -> SWait -> Pack(ML  body) -> MFMA(ML body).

    Same canonical_render for the Pack on both sides, same per-(body,
    render) emission_ordinal=0 in each body, same physical bytes
    flow. Under pre-hdem the identities differed by `loop_index`
    (BODY_LABEL_PROLOGUE => loop_index=-1; BODY_LABEL_ML => 1) and the
    edge-keys differed by `producer.position` — `compare_graphs` fired
    a false-positive. Under hdem A+E both layers are body-blind and
    the comparison returns [].
    """

    def test_cross_body_pack_pipelining_matches(self):
        # REF: Pack producer in PRO body.
        pack_inst = make_lr(40, 1, 64, slot=0, category="PackA0")
        # ML body has the consumer MFMA reading v40.
        ref_pro = make_capture(BODY_LABEL_PROLOGUE, [
            pack_inst,
        ])
        ref_ml = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(0, 40, 60, slot=1, a_src_count=1),
        ])
        ref_cap = _wrap_with_pro(ref_pro, ref_ml)

        # SUBJ: same Pack producer, but in ML body instead of PRO.
        subj_pack_inst = make_lr(40, 1, 64, slot=0, category="PackA0")
        subj_ml = make_capture(BODY_LABEL_ML, [
            subj_pack_inst,
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 40, 60, slot=2, a_src_count=1),
        ])
        subj_cap = _wrap_no_pro(subj_ml)

        g_ref = build_dataflow_graph(ref_cap)
        g_subj = build_dataflow_graph(subj_cap)
        failures = compare_graphs(g_ref, g_subj)
        assert failures == [], (
            "rocm-libraries-hdem: cross-body Pack pipelining "
            "(reference Pack in PRO body, subject Pack in ML body, "
            "same canonical_render, same per-body ordinal, same "
            "consuming MFMA) must compare-equal under Approach A "
            "(identity body-blind) + Approach E (edge_keys "
            "body-blind via identity collapse). "
            f"Got {len(failures)} failures: "
            f"{[type(f).__name__ for f in failures[:5]]}"
        )

    def test_cross_body_pack_pipelining_node_collapse(self):
        """Pin the identity-collapse half of A+E: a Pack-producer
        identity in PRO body and the same-render-and-ordinal Pack
        producer in ML body must yield the SAME identity tuple.
        """
        pack_inst_pro = make_lr(40, 1, 64, slot=0, category="PackA0")
        pack_inst_ml = make_lr(40, 1, 64, slot=0, category="PackA0")
        ident_pro = pack_inst_pro.identity_for(BODY_LABEL_PROLOGUE)
        ident_ml = pack_inst_ml.identity_for(BODY_LABEL_ML)
        assert ident_pro == ident_ml, (
            "Approach A: identity must be body-blind. "
            f"PRO ident={ident_pro!r}, ML ident={ident_ml!r}"
        )
        # And the tuple shape is the new 2-tuple.
        assert len(ident_pro) == 2


# =============================================================================
# Test #2 (per bead spec) — cross-iteration register rotation still distinct
# =============================================================================


class TestCrossIterationRotationStillDistinct:
    """Pin the rotation case: same body, two iterations writing
    different physical bytes (vgpr+0 vs vgpr+16). The canonical_render
    differs (different operand text), so the two emissions get
    distinct identities and distinct edge-keys. Both REF and SUBJ
    have both iterations, so `compare_graphs` returns [].

    This pins ORAM1 §3.5 / §6 S2: A+E does not break the rotation
    case, and Approach D (canonical_render rotation normalization) is
    not required to handle it.
    """

    def test_two_iterations_register_rotation_match(self):
        # REF and SUBJ are identical: two iterations writing different
        # vgpr offsets, both consumed by an MFMA reading the same
        # offset.
        def _build():
            return make_capture(BODY_LABEL_ML, [
                # iter N: pack writes v40
                make_lr(40, 1, 64, slot=0, category="PackA0"),
                make_swait(slot=1, dscnt=0),
                make_mfma(0, 40, 60, slot=2, a_src_count=1),
                # iter N+1: pack writes v56 (rotation cycle slot 1)
                make_lr(56, 1, 64, slot=3, category="PackA0", sequence=1),
                make_swait(slot=4, dscnt=0),
                make_mfma(1, 56, 76, slot=5, a_src_count=1),
            ])
        g_ref = build_dataflow_graph(_wrap_no_pro(_build()))
        g_subj = build_dataflow_graph(_wrap_no_pro(_build()))
        failures = compare_graphs(g_ref, g_subj)
        assert failures == [], (
            "Cross-iteration register rotation: two same-body iterations "
            "writing different physical bytes must match across "
            "identical captures (different canonical_renders -> distinct "
            "identities -> distinct edge keys; both sides have both "
            "edges -> set match). "
            f"Got: {[type(f).__name__ for f in failures[:5]]}"
        )

    def test_two_iterations_distinct_identities(self):
        """Pin the per-iteration identity-distinctness: the rotated
        emissions get distinct identities even within the same body
        because their canonical_renders differ.
        """
        a = make_lr(40, 1, 64, slot=0, category="PackA0")
        b = make_lr(56, 1, 64, slot=3, category="PackA0", sequence=1)
        ident_a = a.identity_for(BODY_LABEL_ML)
        ident_b = b.identity_for(BODY_LABEL_ML)
        assert ident_a != ident_b, (
            "Rotated emissions writing different physical bytes must "
            "have distinct identities (different canonical_renders)."
        )


# =============================================================================
# Test #3 (per bead spec) — cross-body extra-write surfaces
# =============================================================================


class TestCrossBodyExtraWriteSurfaces:
    """Cross-body extra-write divergence pin.

    Constructs a scenario where the reference graph contains a Pack
    producer that has NO body-collapsed counterpart in the subject
    graph (different canonical_render, different ordinal, different
    consumer). The divergence MUST surface as a non-empty Failure
    list — this is the residual-detection path the bead spec calls
    out (and the `_data_flow_ids` gate may catch it directly when the
    extra producer's identity is not present in the subject identity
    set).
    """

    def test_extra_pack_producer_with_no_subject_counterpart_surfaces(self):
        # REF: Pack writes v100, MFMA reads v100. The Pack's
        # canonical_render contains "v100" so the identity is unique
        # to this byte.
        ref_ml = make_capture(BODY_LABEL_ML, [
            make_lr(100, 1, 64, slot=0, category="PackA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 100, 132, slot=2, a_src_count=1),
        ])
        # SUBJ: NO Pack writing v100; the MFMA reading v100 is also
        # absent (so we don't trigger a missing-consumer raise).
        # Subject simply has a different MFMA reading different vgprs.
        subj_ml = make_capture(BODY_LABEL_ML, [
            make_lr(40, 1, 64, slot=0, category="PackA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(0, 40, 60, slot=2, a_src_count=1),
        ])
        ref_cap = _wrap_no_pro(ref_ml)
        subj_cap = _wrap_no_pro(subj_ml)

        g_ref = build_dataflow_graph(ref_cap)
        g_subj = build_dataflow_graph(subj_cap)

        # Compare; expect either CaptureConsistencyError (if the
        # identity-set gate catches the extra producer) or a non-empty
        # failure list (if the gate misses but the edge layer
        # surfaces it).
        from Tensile.Components.ScheduleCapture import (
            CaptureConsistencyError,
        )
        try:
            failures = compare_graphs(g_ref, g_subj)
        except CaptureConsistencyError:
            # Gate caught the divergence — pinned outcome.
            return
        assert failures, (
            "Cross-body extra-write divergence: the reference contains "
            "a Pack writing v100 + MFMA reading v100 with no body-"
            "collapsed counterpart in subject; the divergence must "
            "surface either as CaptureConsistencyError at the "
            "identity-set gate or as a non-empty Failure list at the "
            "edge layer (per ORAM1 §6.1 mitigation argument). Got "
            f"empty failures."
        )

    def test_extra_pack_in_pro_with_distinct_consumer_surfaces(self):
        """Variant: extra Pack producer in PRO body with a unique
        consumer that does not exist in subject. Even when the Pack's
        identity might collapse with an ML-body Pack of the same
        render, the consuming MFMA's identity in REF (with its
        specific operand register set) is absent in SUBJ — the
        identity-set gate fires.
        """
        # REF: PRO has Pack writing v150; ML has MFMA reading v150.
        ref_pro = make_capture(BODY_LABEL_PROLOGUE, [
            make_lr(150, 1, 64, slot=0, category="PackA0"),
        ])
        ref_ml = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(0, 150, 170, slot=1, a_src_count=1),
        ])
        # SUBJ: no PRO; ML has MFMA reading completely different
        # registers.
        subj_ml = make_capture(BODY_LABEL_ML, [
            make_mfma(0, 100, 132, slot=0, a_src_count=1),
        ])
        ref_cap = _wrap_with_pro(ref_pro, ref_ml)
        subj_cap = _wrap_no_pro(subj_ml)

        g_ref = build_dataflow_graph(ref_cap)
        g_subj = build_dataflow_graph(subj_cap)

        from Tensile.Components.ScheduleCapture import (
            CaptureConsistencyError,
        )
        try:
            failures = compare_graphs(g_ref, g_subj)
        except CaptureConsistencyError:
            return
        assert failures, (
            "Cross-body extra-write with distinct consumer must surface "
            "as a CaptureConsistencyError raise OR a non-empty Failure "
            f"list. Got empty failures."
        )
