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
"""Reproducer for the p39d GR-in-ML OrderInverted residual class.

Bead: ``rocm-libraries-p39d`` — "GR OrderInverted residual class generalizes
beyond nyb5's 3-failure pin."

Structural template: ``test_cross_subiter_pack_artifact.py`` (synthetic-minimal
pattern, real rocisa instructions, hand-trimmed scale-down) +
``test_cross_subiter_alu_carveout_real_kernel.py`` (real-kernel anchor).
The combined structure here: Phase 1 anchors the failure in the real kernel
build; Phase 2 isolates the same failure shape in a GR-only trimmed capture
to prove the residual is intrinsic to GR ordering, not noise from surrounding
LR/LW/MFMA instructions.

Mechanism (confirmed by 3ija triage):
    The canonical _128x128x32_TF32 TN kernel uses DirectToLds (DTL) mode for
    its A and B global-read streams.  Each DTL "group" in the ML body consists
    of:

      1. An m0-setter — ``s_mov_b32 m0, <addr>`` or ``s_add_u32 m0, m0, N``
         (category GRA or GRB, scalar ALU).  This sets the LDS write address
         for the next DTL load.

      2. A DTL buffer_load — ``buffer_load_dwordx4 ... lds`` (category GRA or
         GRB, rocisa class BufferLoadB128 with is_dtl=True).  This reads m0
         implicitly as the LDS destination address.

    The dataflow graph captures the m0 dependency: m0-setter → DTL-load.

    The non-CMS reference build (Approach A) and the CMS build emit these
    GRA/GRB groups in DIFFERENT ORDERS within the ML body because the CMS
    per-tile schedule sets ``UsePLRPack=True`` on the kernel dict BEFORE SIA3
    runs — changing the global-read scheduling order on the CMS side relative
    to the non-CMS reference (which uses an unmutated kernel dict per
    ``2LZD_INVESTIGATION.md §6.2 Q2``).

    The result: 2 edge keys in the ref graph are missing from the CMS graph,
    where the CMS schedule emitted certain DTL-load / m0-setter pairs in the
    reversed order relative to the non-CMS reference.  These 2 missing edges
    surface as ``OrderInvertedFailure`` on GRB → GRB pairs in the ML body
    (per 3ija §2 table: _128x128x32_TF32 TN has 2× GRA→GRA + 1× GRB→GRA
    ordered failures — but the exact count may vary with the subgraph that
    reaches the classifier after the identity-set gate).

Expected state (nyb5 baseline, ``users/alvasile/validator_long_term_plans``
branch tip):

* Phase 1 (real-kernel anchor): ``compare_graphs`` raises
  ``CaptureConsistencyError`` before reaching OrderInverted classification,
  because the non-CMS reference build's LR/MFMA identity set differs from the
  CMS build's identity set (28 identities per side, split ``{LR:14, MFMA:14}``).
  The GRA/GRB ordering residuals are obscured by this earlier gate.  The test
  pins the CaptureConsistencyError shape and asserts the GRA/GRB ordering issue
  is NOT the source of the identity mismatch (i.e. GRA/GRB are absent from the
  per-side identity-diff summary).

* Phase 2 (trimmed minimal): a GR-only capture pair with 2 instructions — one
  m0-setter (``s_mov_b32 m0, s10``, category GRA) and one DTL buffer_load
  (reads m0, category GRB) — ordered differently in ref vs. subj.  The
  identity sets are identical (same GRA and GRB nodes on both sides), so
  ``compare_graphs`` proceeds past the identity-set gate and surfaces exactly
  1 ``OrderInvertedFailure`` on the GRA → GRB dependency whose order was
  inverted.  This proves the ordering issue is intrinsic: given only the GR
  stream, the validator WOULD fire even without the LR/MFMA noise.

When p39d is fixed (any of options 1/2/3 from the bead — extend
``_NO_DATAFLOW_IDENTITY_CATEGORIES``, add per-edge tolerance, or make GR
identity body-blind), Phase 2 MUST be updated: the assertion on
``len(failures) == 1`` flips to ``failures == []``.  The Phase 1 pinned
``CaptureConsistencyError`` shape may or may not change depending on which
fix is chosen.
"""

import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Shared kernel config — canonical TF32 4x4 TN from nyb5
# ---------------------------------------------------------------------------

_CANONICAL_KERNEL_CONFIG = {
    'ProblemType': {
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    'MatrixInstruction': [16, 16, 32, 1, 1, 4, 4, 2, 2],
    'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 1,
    'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
    'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
    'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
    'SourceSwap': 1, 'StreamK': 0,
    # rocm-libraries-2bww: required for the ('TN', False, 1) branch.
    'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
}


# ===========================================================================
# Phase 1 — Real-kernel anchor
# ===========================================================================

@pytest.fixture(scope="module")
def p39d_real_kernel_captures(isa_infrastructure):
    """Build the canonical TF32 4x4 TN kernel via the Approach A two-build
    path and return ``(ref_cap, cms_cap)`` where:

      * ``ref_cap`` = ``build_non_cms_reference`` (Build #2, non-CMS schedule,
        unmutated kernel dict per Q2 framing).
      * ``cms_cap`` = ``_last_cms_capture`` (Build #1, CMS schedule).

    Module-scoped: the two builds take ~5-10s; they are shared across both
    Phase 1 tests.

    The nyb5-baseline behavior is that ``compare_graphs(ref_graph, subj_graph)``
    raises ``CaptureConsistencyError`` before reaching the OrderInverted check,
    because the non-CMS and CMS builds produce different LR/MFMA identities (28
    per side, split ``{LR:14, MFMA:14}``).  This is a known Q2-expected
    divergence pinned by
    ``test_non_cms_reference_compare_graphs_surfaces_only_known_residuals``.
    """
    _isa, isaInfoMap, asm = isa_infrastructure
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from Tensile.Components.CustomSchedule.approach_a import build_non_cms_reference

    config = dict(_CANONICAL_KERNEL_CONFIG)

    # Build #1: CMS path — extract the CMS-side FourPartCapture.
    cms_solution = _make_solution(config, asm, isaInfoMap)
    cms_writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        cms_writer._getKernelSource(cms_solution)
    except Exception:
        # Pre-existing shadow-vs-CMS divergence may raise. The CMS-side
        # FourPartCapture is populated before the assert fires.
        pass
    cms_cap = cms_writer._last_cms_capture
    assert cms_cap is not None, (
        "CMS build did not populate _last_cms_capture; the kernelBody "
        "post-loop assembly stage did not run."
    )

    # Build #2: non-CMS reference via Approach A's helper.
    # rocm-libraries-l1l6: build_non_cms_reference takes the Solution
    # directly and reads the pre-CMS-derivation snapshot via
    # cms_solution.pre_cms_state().
    ref_cap = build_non_cms_reference(cms_solution, asm, isaInfoMap)

    return ref_cap, cms_cap


def test_real_kernel_anchor_gr_ordering_residual_shape(
    p39d_real_kernel_captures,
):
    """Phase 1: Real-kernel anchor for the p39d GR OrderInverted residual.

    The canonical TF32 4x4 TN kernel with Build #1 (CMS) vs. Build #2
    (non-CMS reference via Approach A).

    rocm-libraries-oplb (post-count-based-gate): the entry-level identity-set
    gate in ``compare_graphs`` was replaced with a per-(rocisa-derived
    InstructionCategory) count gate. Under the count gate, both pipelines
    emit the same number of LR/MFMA/GR nodes for this kernel — the gate
    PASSES (no CaptureConsistencyError at entry). The downstream comparison
    proceeds to ``edge_keys()`` and then ``diagnose_missing_edge`` for the
    missing edges.

    The edge layer still embeds ``(producer.identity, consumer.identity,
    ...)`` in its edge-key tuples, so the T/X register-naming divergence
    (UsePLRPack double-register-file convention) still propagates here:
    ref-edge endpoint identities do not resolve in the subject graph, and
    Phase 0 of ``diagnose_missing_edge`` raises ``CaptureConsistencyError``
    with the "identity-coverage check at compare_graphs entry was bypassed"
    message — that is the newly-surfaced shape pinned below.

    The p39d GR OrderInverted residual is now structurally closer to the
    surface, but is still gated by the edge-layer T/X identity divergence.
    Resolving p39d in the real kernel requires the follow-up edge-layer
    fix (byte-key matching per the Approach-E reference in CMSValidator.py)
    so the LR/MFMA edges classify cleanly and only the genuine GR ordering
    residual reaches OrderInverted classification.

    NOTE: This assertion was tightened post-oplb's count-based gate.
    """
    from Tensile.Components.CMSValidator import build_dataflow_graph, compare_graphs
    from Tensile.Components.ScheduleCapture import CaptureConsistencyError

    ref_cap, cms_cap = p39d_real_kernel_captures
    ref_graph = build_dataflow_graph(ref_cap)
    subj_graph = build_dataflow_graph(cms_cap)

    with pytest.raises(CaptureConsistencyError) as excinfo:
        compare_graphs(ref_graph, subj_graph)

    msg = str(excinfo.value)

    # Pin: the edge-layer Phase-0 bypass message — the count-based gate
    # passes and the edge layer hits a missing endpoint identity.
    assert "identity-coverage check at compare_graphs entry was bypassed" in msg, (
        f"Expected the diagnose_missing_edge Phase-0 'bypass' message "
        f"(edge-layer T/X register-naming divergence after the new "
        f"count-based gate passes); full message:\n{msg}"
    )
    assert "p_id=" in msg and "c_id=" in msg, (
        f"Expected both p_id= and c_id= in the bypass message; "
        f"full message:\n{msg}"
    )

    # Pin: the new count-based gate did NOT fire — counts match across
    # sides for this kernel (which is the whole point of the oplb gate
    # change).
    assert "data-flow per-category node counts differ" not in msg, (
        f"The new count-based gate fired — that would mean the two pipelines "
        f"actually emitted different numbers of LR/MFMA/GR nodes (a real "
        f"pipeline-integrity bug). Investigate before re-pinning.\n{msg}"
    )


# ===========================================================================
# Phase 2 — Trimmed minimal reproducer
# ===========================================================================
#
# The Phase 1 anchor confirms that the real kernel's GR ordering issue is
# hidden behind the LR/MFMA identity-set gate.  Phase 2 isolates the ordering
# behavior in a GR-only capture pair to prove it is INTRINSIC to the GR
# ordering itself.
#
# Mechanism (confirmed by 3ija triage + direct graph inspection):
#   The p39d GRA/GRB ordering residuals arise from the DTL GR stream's
#   m0-dependency chain:
#
#     s_mov_b32 m0, s[addr]      — m0-setter (category GRA, writes m0)
#     buffer_load_dwordx4 ... lds — DTL load  (category GRA, reads m0)
#     s_add_u32 m0, m0, 4224     — m0-increment (category GRA, reads+writes m0)
#     buffer_load_dwordx4 ... lds — DTL load   (category GRA, reads m0)
#
#   Each m0-setter/increment → DTL-load pair forms a dataflow edge via m0.
#   When CMS reorders these pairs (e.g. emits DTL[B+2] before DTL[B+1]),
#   the m0-increment that feeds DTL[B+2] now appears AFTER DTL[B+2] in the
#   CMS stream — the edge key (m0-increment, DTL[B+2]) is missing from the
#   CMS graph, surfacing as OrderInvertedFailure.
#
# Instruction subset for Phase 2 (minimal: 1 m0-setter + 1 DTL-load):
#   GRA_setter : s_mov_b32 m0, s10                     (category "GRA")
#   GRB_loader : buffer_load_dwordx4 v254, s[4:7] lds  (category "GRB")
#
#   Both instructions interact via the m0 register:
#     - GRA_setter writes m0 (the LDS write address for DTL)
#     - GRB_loader reads m0 implicitly (its LDS destination address)
#
#   Default (ref) order: GRA_setter, GRB_loader → m0 is set before it is read.
#   The per-byte resolver records GRA_setter as the latest m0 writer, so the
#   GRB_loader's m0 read creates the edge GRA_setter → GRB_loader.
#
#   CMS (subj) order: GRB_loader, GRA_setter → the DTL load executes BEFORE
#   the m0-setter.  No prior m0 writer exists when GRB_loader runs, so NO
#   m0 edge forms between them.  The ref's GRA → GRB edge-key is absent.
#
#   diagnose_missing_edge then checks:
#     default_p_before_c = True  (GRA at pos=0 < GRB at pos=1 in ref)
#     subj_p_before_c   = False  (GRA at pos=1 > GRB at pos=0 in subj)
#   → OrderInvertedFailure(producer=GRA, consumer=GRB).
#
#   The cross-subiter ALU carve-out does NOT fire because both instructions
#   occupy the same mfma_index slot (0), so their subiter values are equal:
#   p_node.subiter(nmps) == c_node.subiter(nmps) → carve-out condition False.
#
#   Why different categories (GRA vs GRB) for setter and loader?
#   Using distinct categories gives distinct node identities (the canonical
#   render strings differ anyway: s_mov_b32 vs buffer_load_dwordx4), but also
#   makes the failure label unambiguous: producer.category=GRA, consumer.category=GRB.
#   The 3ija triage reports mixed GRA→GRA and GRB→GRA failures in the real
#   kernel; this fixture reproduces the cross-category variant (GRA→GRB) which
#   is also present in the real kernel's ML body.

# m0 is the shared resource connecting the setter (GRA) to the DTL load (GRB).
# Use mgpr(0) — the numeric form that matches m0_resource()'s byte key ('m', 0).
# (mgpr('m0', 1) uses a symbolic name that produces key ('m', 'm0', 0) —
#  incompatible with m0_resource()'s ('m', 0) key; confirmed by byte-key audit.)
_M0_REGISTER_IDX = 0   # mgpr(0) = the m0 hardware register

# GRA setter: writes m0 from sgpr s10 (an arbitrary source sgpr).
_GRA_SETTER_SRC_SGPR = 10

# GRB loader: DTL BufferLoadB128, reads m0 implicitly, uses a distinct SRD
# (s[4:7]) so its render-string differs from any GRA instructions.
_GRB_LOADER_SRD_SGPR = 4   # s[4:7]
_GRB_LOADER_VADDR_VGPR = 254  # placeholder vaddr, not tracked in dataflow


def _make_gra_m0_setter(slot):
    """Build a TaggedInstruction wrapping ``s_mov_b32 m0, s10`` (category GRA).

    This is the minimal m0-setter pattern extracted from the real kernel's GRA
    group.  The rocisa SMovB32 writes ``mgpr(0)`` (m0 hardware register, byte
    key ``('m', 0)``), which the downstream DTL buffer_load reads implicitly
    via ``m0_resource()`` (same byte key).  The GRA category is recognized by
    ``_is_recognized_capture_category`` (prefix "GR"), so ``_make_node`` does
    not raise ``CaptureUnknownInstructionError``.
    """
    from Tensile.Components.ScheduleCapture import (
        SLOT_KIND_MFMA, SlotKey, TaggedInstruction, WrappedInstruction,
    )
    from rocisa.container import mgpr, sgpr
    from rocisa.instruction import SMovB32
    m0 = mgpr(_M0_REGISTER_IDX)
    inst = SMovB32(dst=m0, src=sgpr(_GRA_SETTER_SRC_SGPR, 1))
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category="GRA",
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=0),
    )


def _make_grb_dtl_loader(slot):
    """Build a TaggedInstruction wrapping a DTL buffer_load_dwordx4 (category GRB).

    DTL mode (``dst=None``, ``mubuf->lds=True``) causes the
    ``_BufferLoadRule`` to publish an implicit m0 read via ``m0_resource()``
    (byte key ``('m', 0)``) alongside the vaddr and saddr reads.  The GRB
    category is recognized by ``_is_recognized_capture_category`` (prefix "GR").
    """
    from Tensile.Components.ScheduleCapture import (
        SLOT_KIND_MFMA, SlotKey, TaggedInstruction, WrappedInstruction,
    )
    from rocisa.container import MUBUFModifiers, vgpr, sgpr
    from rocisa.instruction import BufferLoadB128
    mubuf = MUBUFModifiers(offen=True, offset12=0, lds=True)
    inst = BufferLoadB128(
        dst=None,
        vaddr=vgpr(_GRB_LOADER_VADDR_VGPR, 1),
        saddr=sgpr(_GRB_LOADER_SRD_SGPR, 4),
        soffset=0,
        mubuf=mubuf,
    )
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category="GRB",
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=0),
    )


def _build_ref_capture():
    """Default (non-CMS) order: GRA setter at slot 0, GRB DTL-loader at slot 1.

    Per-byte resolver walk:
      slot 0 — GRA setter writes m0 (byte key ('m', 0)).
               latest_writer[('m', 0)] = GRA_setter_node
      slot 1 — GRB loader reads m0 implicitly.
               _resolve_producers finds GRA_setter_node as writer of ('m', 0).
               Edge GRA_setter → GRB_loader emitted with edge_kind=raw_intrawave.
    """
    from Tensile.Components.ScheduleCapture import BODY_LABEL_ML
    from dataflow_fixtures import make_capture
    return make_capture(BODY_LABEL_ML, [
        _make_gra_m0_setter(slot=0),
        _make_grb_dtl_loader(slot=1),
    ])


def _build_subj_capture():
    """CMS (reordered) order: GRB DTL-loader at slot 0, GRA setter at slot 1.

    Per-byte resolver walk:
      slot 0 — GRB loader reads m0 implicitly.
               No prior m0 writer → no m0 edge formed.
      slot 1 — GRA setter writes m0.
               latest_writer[('m', 0)] = GRA_setter_node (no reader follows).

    Result: NO edge between GRA_setter and GRB_loader in this graph.  The ref's
    GRA → GRB edge-key (GRA_setter.identity, GRB_loader.identity, 'raw_intrawave',
    ...) is absent from subj.  diagnose_missing_edge fires: default_p_before_c=True
    (GRA at slot 0 < GRB at slot 1 in ref), subj_p_before_c=False (GRA at slot 1
    > GRB at slot 0 in subj) → OrderInvertedFailure.
    """
    from Tensile.Components.ScheduleCapture import BODY_LABEL_ML
    from dataflow_fixtures import make_capture
    return make_capture(BODY_LABEL_ML, [
        _make_grb_dtl_loader(slot=0),
        _make_gra_m0_setter(slot=1),
    ])


def _wrap_in_four_part(ml_capture):
    """Wrap a single ML body in a FourPartCapture with minimal filler bodies.

    Filler bodies use vgpr ranges well above any GR/m0 resources used in the
    ML body (high-numbered vgpr indices), avoiding any aliasing.
    """
    from Tensile.Components.CMSValidator import _DEFAULT_CDNA4_ARCH_PROFILE
    from Tensile.Components.ScheduleCapture import (
        BODY_LABEL_ML_PREV, BODY_LABEL_NGL, BODY_LABEL_NLL, FourPartCapture,
    )
    from dataflow_fixtures import make_capture, make_mfma

    def _filler(label, c, a, b):
        return make_capture(label, [
            make_mfma(c_dst_start=c, a_src_start=a, b_src_start=b, slot=0),
        ])

    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV, 100, 104, 108)},
        n_gl={0: _filler(BODY_LABEL_NGL, 120, 124, 128)},
        n_ll={0: _filler(BODY_LABEL_NLL, 140, 144, 148)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


class TestP39dGrOrderInvertedMinimal:
    """Phase 2: trimmed GR-only minimal reproducer for the p39d residual.

    Two GR instructions connected by the m0 register (DTL LDS address):
      - GRA setter: s_mov_b32 m0, s10 (writes m0, category GRA)
      - GRB loader: buffer_load_dwordx4 ... lds (reads m0, category GRB)

    In the ref capture: GRA setter then GRB loader — m0 is set before it is
    read, creating a GRA → GRB dataflow edge.

    In the subj capture: GRB loader then GRA setter — the m0 read precedes
    the write, so no m0 edge exists between them.  The ref's GRA → GRB
    edge-key is absent from subj → OrderInvertedFailure.

    The identity sets are identical (GRA and GRB present in both captures →
    no CaptureConsistencyError), so compare_graphs reaches the ordering check.
    The cross-subiter ALU carve-out does NOT fire because both instructions
    occupy the same mfma_index slot (subiter equal).

    NOTE: When p39d is fixed (bead options 1/2/3), the assertion
    ``len(failures) == 1`` must flip to ``failures == []`` — the fix will
    suppress this GR ordering inversion as an expected scheduler difference
    per Q2.
    """

    def test_ref_graph_has_gra_to_grb_m0_edge(self):
        """The default-order capture (GRA setter then GRB loader) produces a
        dataflow edge GRA → GRB through the m0 register.

        Per-byte resolver:
          1. GRA setter writes m0 → latest_writer[('m', 0)] = GRA_setter_node
          2. GRB loader reads m0 (implicit DTL slot) → produces GRA → GRB edge
        """
        from Tensile.Components.CMSValidator import build_dataflow_graph

        ref_cap = _build_ref_capture()
        g_ref = build_dataflow_graph(_wrap_in_four_part(ref_cap))

        gr_edges = [
            e for e in g_ref.edges
            if getattr(e.producer, "category", "") in {"GRA", "GRB"}
            and getattr(e.consumer, "category", "") in {"GRA", "GRB"}
        ]
        assert len(gr_edges) == 1, (
            f"Expected exactly 1 GR→GR edge in the default capture "
            f"(GRA setter → GRB loader via m0); "
            f"got {len(gr_edges)}: "
            f"{[(e.producer.category, e.consumer.category) for e in gr_edges]}"
        )
        assert gr_edges[0].producer.category == "GRA", (
            f"GRA setter must be the producer (earlier m0 writer); "
            f"got {gr_edges[0].producer.category}."
        )
        assert gr_edges[0].consumer.category == "GRB", (
            f"GRB loader must be the consumer (m0 reader); "
            f"got {gr_edges[0].consumer.category}."
        )
        # Confirm the edge resource is m0 (regType='m', regIdx=0).
        res = gr_edges[0].resource
        assert getattr(res, "regType", None) == "m", (
            f"Edge resource must be the m0 register (regType='m'); "
            f"got regType={getattr(res, 'regType', None)}."
        )

    def test_subj_graph_has_no_gra_to_grb_m0_edge(self):
        """The CMS-order capture (GRB loader then GRA setter) produces NO
        edge from GRA to GRB through m0.

        When the DTL load (GRB) executes BEFORE the m0-setter (GRA), no prior
        m0 writer exists in the resolver — the implicit m0 read produces no
        edge.  The GRA_setter → GRB_loader edge-key from the ref is absent.
        """
        from Tensile.Components.CMSValidator import build_dataflow_graph

        subj_cap = _build_subj_capture()
        g_subj = build_dataflow_graph(_wrap_in_four_part(subj_cap))

        gr_edges = [
            e for e in g_subj.edges
            if getattr(e.producer, "category", "") in {"GRA", "GRB"}
            and getattr(e.consumer, "category", "") in {"GRA", "GRB"}
        ]
        assert len(gr_edges) == 0, (
            f"Expected NO GR→GR edge in the CMS capture (DTL load precedes "
            f"m0-setter, so no m0 dependency edge forms); "
            f"got {len(gr_edges)}: "
            f"{[(e.producer.category, e.consumer.category) for e in gr_edges]}"
        )

    def test_compare_graphs_surfaces_orderinverted_on_gr_pair(self):
        """End-to-end trimmed reproducer: compare_graphs on the GR-only
        capture pair surfaces exactly 1 OrderInvertedFailure with
        producer.category=GRA and consumer.category=GRB, both in the ML body.

        The identity sets are identical (GRA and GRB present in both captures
        → no CaptureConsistencyError), so compare_graphs reaches Phase 1 of
        diagnose_missing_edge for the ref's GRA → GRB edge.

        Phase 1 sees:
          default_p_before_c = True  (GRA at slot 0 < GRB at slot 1 in ref)
          subj_p_before_c   = False  (GRA at slot 1 > GRB at slot 0 in subj)
          _is_alu_producer(GRA) = True (SMovB32 is scalar ALU)
          subiter(GRA) == subiter(GRB) (same mfma_index → no carve-out)
        → OrderInvertedFailure on (GRA, GRB).

        This confirms the residual is INTRINSIC to GR ordering: given only
        the GR stream (no LR/LW/MFMA noise), the validator fires on the
        m0-dependency ordering inversion.

        NOTE: When p39d is fixed (bead options 1/2/3), this assertion must flip
        to ``assert failures == []`` — the fix will suppress this GR ordering
        inversion as an expected scheduler difference per Q2.
        """
        from Tensile.Components.CMSValidator import (
            OrderInvertedFailure, build_dataflow_graph, compare_graphs,
        )

        ref_graph = build_dataflow_graph(_wrap_in_four_part(_build_ref_capture()))
        subj_graph = build_dataflow_graph(_wrap_in_four_part(_build_subj_capture()))

        # The identity-set gate must pass: same GRA and GRB identities + MFMA
        # fillers on each side.  If compare_graphs raises CaptureConsistencyError
        # here, the fixture wiring is wrong (filler bodies may produce unexpected
        # identity differences).
        failures = compare_graphs(ref_graph, subj_graph)

        assert len(failures) == 1, (
            f"Expected exactly 1 OrderInvertedFailure on the GRA→GRB ordering "
            f"inversion (m0-setter before DTL-loader in ref, reversed in subj); "
            f"got {len(failures)} failures: "
            f"{[(type(f).__name__, getattr(f, 'producer', None), getattr(f, 'consumer', None)) for f in failures]}"
        )
        failure = failures[0]
        assert isinstance(failure, OrderInvertedFailure), (
            f"Expected OrderInvertedFailure; got {type(failure).__name__}."
        )
        assert getattr(failure.producer, "category", "") == "GRA", (
            f"Expected producer.category='GRA' (the m0-setter, first in the "
            f"default order); got {getattr(failure.producer, 'category', None)}."
        )
        assert getattr(failure.consumer, "category", "") == "GRB", (
            f"Expected consumer.category='GRB' (the DTL loader, m0 reader); "
            f"got {getattr(failure.consumer, 'category', None)}."
        )
        assert getattr(failure.producer, "body_label", None) == "ML", (
            f"Expected producer in ML body; got "
            f"{getattr(failure.producer, 'body_label', None)}."
        )
        assert getattr(failure.consumer, "body_label", None) == "ML", (
            f"Expected consumer in ML body; got "
            f"{getattr(failure.consumer, 'body_label', None)}."
        )
