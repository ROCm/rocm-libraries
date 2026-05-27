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
# running this test under pytest also produces visual dataflow graphs as PNG at
# `oplb_artifacts/figures/`.
"""Reproducer for the rocm-libraries-oplb CaptureConsistencyError on
LR/MFMA identity divergence — specifically the T0_I0/X0_I0 register-naming
divergence.

Bead: ``rocm-libraries-oplb`` — CaptureConsistencyError on LR/MFMA identities
masks downstream residuals in real-vs-real ``compare_graphs``.

Structural template: ``test_p39d_gr_orderinverted_minimal.py`` (Phase 1
real-kernel anchor + Phase 2 trimmed minimal pattern) combined with
``test_cross_subiter_pack_artifact.py`` (assertion-pinning style).

Mechanism (confirmed by oplb investigation artifacts in
``oplb_artifacts/README.md``):

    The _192x256x32_TF32 TN kernel uses ``UsePLRPack=True``.  In the CMS
    variant, the ``customMainLoopSchedule`` assigns local-read destinations
    to a **double register file**: alternating between ``vgprValuA_T0_I0+N``
    (the "T0" buffer) and ``vgprValuA_X0_I0+N`` (the "X0" buffer), which
    occupy different physical vgpr ranges.  The non-CMS build assigns
    destinations from a single ``vgprValuA_T0_I0+N`` register array with
    linearly increasing offsets.

    The dataflow graph identity is ``(canonical_render, emission_ordinal)``
    where ``canonical_render`` is the rendered assembly string.  Because
    ``ds_read_b128 v[28:31], ...`` and ``ds_read_b128 v[4:7], ...`` render
    differently, even when reading the same LDS offset, they produce
    **distinct identity tuples**.  ``compare_graphs`` therefore sees a
    non-empty identity-set symmetric difference and raises
    ``CaptureConsistencyError`` before reaching any ordering / wait
    analysis.

    Result (nyb5 baseline, ``users/alvasile/validator_long_term_plans``
    branch tip):
      * 52 identity-set divergences per side: 26 LR + 26 MFMA.
      * LR divergences: CMS X0_I0 reads map to high vgpr indices absent
        from the non-CMS T0_I0-only graph, and vice versa.
      * MFMA divergences: downstream MFMAs read the PLR-renamed vgprs,
        so their identity tuples differ symmetrically.

Fix direction (from the investigation): symbolic-register-name
canonicalization in ``identity_for`` or ``WrappedInstruction.canonical_str``
— normalize ``_T0_I0`` and ``_X0_I0`` suffixes to the same logical slot
before constructing the identity tuple (option D from the bead).  When
fixed, all three Phase asserts flip to "no failures" / "graphs equivalent."

Phase 1 — real-kernel anchor
    Builds the _192x256x32_TF32 TN kernel via ``_make_solution`` +
    ``build_non_cms_reference``.  Asserts ``CaptureConsistencyError`` with
    the documented 52-per-side identity-set divergence.  Pins that the
    diverging categories are LR and MFMA (not GRA/GRB), confirming the
    bug is register-naming, not GR ordering.

Phase 2 — trimmed minimal reproducer
    Two hand-built ``LoopBodyCapture`` objects using real rocisa
    ``DSLoadB128`` instructions and one ``VCvtPkF32toBF16`` Pack:

    * CMS-side (subj): 4 LR reads with ALTERNATING T0 (low vgpr indices)
      and X0 (high vgpr indices) destinations — the UsePLRPack double-
      register-file pattern.  One Pack consuming a T0-index destination.

    * non-CMS-side (ref): 4 LR reads with LINEAR T0-only destinations.
      One Pack consuming the linear T0 destination.

    ``compare_graphs`` raises ``CaptureConsistencyError`` on the LR
    identity divergence.

Phase 3 — PNG dataflow-graph figures
    As a side effect of the test run, generates visual PNG graphs for
    both captures into ``oplb_artifacts/figures/``.

When oplb is fixed (register-name canonicalization in ``identity_for``
or ``WrappedInstruction.canonical_str`` — see bead option D), all three
Phase tests must be updated: Phase 1 and Phase 2 ``CaptureConsistencyError``
assertions flip to "no failures," and the Phase 2 LR divergence count
asserts flip to 0.
"""

import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Shared kernel config — _192x256x32_TF32 TN
# Mirrors BenchmarkProblemSizeGroup 1 of
# Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml:38-76
# ---------------------------------------------------------------------------

_192X256X32_TF32_TN_CONFIG = {
    'ProblemType': {
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    'MatrixInstruction': [16, 16, 32, 1, 1, 6, 8, 2, 2],
    'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 1,
    'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
    'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
    'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
    'SourceSwap': 1, 'StreamK': 0,
    'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
}


# ===========================================================================
# Phase 1 — Real-kernel anchor
# ===========================================================================

@pytest.fixture(scope="module")
def oplb_real_kernel_captures(isa_infrastructure):
    """Build the _192x256x32_TF32 TN kernel via the Approach A two-build path
    and return ``(ref_cap, cms_cap)`` where:

      * ``ref_cap`` = ``build_non_cms_reference`` (Build #2, non-CMS schedule,
        unmutated kernel dict per Q2 framing).
      * ``cms_cap`` = ``_last_cms_capture`` (Build #1, CMS schedule).

    Module-scoped: the two builds take ~5-10 seconds; they are shared across
    both Phase 1 tests.

    The nyb5-baseline behavior is that ``compare_graphs(ref_graph, subj_graph)``
    raises ``CaptureConsistencyError`` because the two builds produce different
    LR/MFMA identity sets: 52 per side, split ``{LR: 26, MFMA: 26}``.  This is
    the register-naming divergence pinned by this test class.
    """
    _isa, isaInfoMap, asm = isa_infrastructure
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from Tensile.Components.CustomSchedule.approach_a import build_non_cms_reference

    config = dict(_192X256X32_TF32_TN_CONFIG)

    # Build #1: CMS path — extract the CMS-side FourPartCapture.
    cms_solution = _make_solution(config, asm, isaInfoMap)
    cms_writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        cms_writer._getKernelSource(cms_solution)
    except Exception:
        # Pre-existing LR/MFMA identity-naming divergence may raise here
        # (that is exactly the bug this test pins). The CMS-side
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


def test_real_kernel_anchor_lr_naming_divergence_shape(
    oplb_real_kernel_captures,
):
    """Phase 1: Real-kernel anchor for the oplb LR/MFMA register-naming divergence.

    The _192x256x32_TF32 TN kernel with Build #1 (CMS, UsePLRPack=True) vs.
    Build #2 (non-CMS reference via Approach A) surfaces a
    ``CaptureConsistencyError`` at ``compare_graphs`` entry because the two
    builds' LR/MFMA identity sets differ.

    Documented divergence (from oplb investigation artifacts):
      * 52 identities per side: 26 LR + 26 MFMA.
      * CMS side: alternating vgprValuA_T0_I0+N and vgprValuA_X0_I0+N
        destinations (the UsePLRPack double-register-file naming).
      * non-CMS side: only vgprValuA_T0_I0+N with linearly increasing indices.

    This test additionally asserts that the identity-set mismatch is ENTIRELY
    in the LR and MFMA categories — GRA and GRB do NOT appear in the per-side
    identity diff.  This confirms the bug is register-naming (T0 vs X0), not
    GR ordering (which is the separate rocm-libraries-p39d issue).

    NOTE: When oplb is fixed (register-name canonicalization in identity_for
    or WrappedInstruction.canonical_str — bead option D), this test must flip
    to assert no CaptureConsistencyError is raised.
    """
    from Tensile.Components.CMSValidator import build_dataflow_graph, compare_graphs
    from Tensile.Components.ScheduleCapture import CaptureConsistencyError

    ref_cap, cms_cap = oplb_real_kernel_captures
    ref_graph = build_dataflow_graph(ref_cap)
    subj_graph = build_dataflow_graph(cms_cap)

    with pytest.raises(CaptureConsistencyError) as excinfo:
        compare_graphs(ref_graph, subj_graph)

    msg = str(excinfo.value)

    # Pin: exactly 52 identities per side (LR+MFMA divergence from the
    # UsePLRPack T0_I0/X0_I0 double-register-file naming).
    assert "in reference but not subject: 52 identities" in msg, (
        f"Expected 52 ref-but-not-subj identities (LR+MFMA T0/X0 divergence); "
        f"full message:\n{msg}"
    )
    assert "in subject but not reference: 52 identities" in msg, (
        f"Expected 52 subj-but-not-ref identities (LR+MFMA T0/X0 divergence); "
        f"full message:\n{msg}"
    )

    # Pin: the categories in the diff are LR and MFMA — the register-naming
    # divergence from UsePLRPack.
    assert "'LR'" in msg, (
        f"Expected 'LR' category in the identity-diff summary (LR reads with "
        f"different vgpr destinations); full message:\n{msg}"
    )
    assert "'MFMA'" in msg, (
        f"Expected 'MFMA' category in the identity-diff summary (MFMAs reading "
        f"differently-named vgprs); full message:\n{msg}"
    )

    # Negative check: GRA and GRB must NOT appear in the identity diff.
    # The register-naming bug is in LR/MFMA — GR instructions use the same
    # physical addressing on both sides (the T0/X0 split is LR-buffer-side
    # only, not in the GR-to-LDS path).
    for forbidden_cat in ("'GRA'", "'GRB'", "'GR'"):
        assert forbidden_cat not in msg, (
            f"Category {forbidden_cat} appeared in the identity-diff summary, "
            f"which would mean GR instructions are part of the LR/MFMA naming "
            f"divergence — unexpected for oplb. "
            f"Full message:\n{msg}"
        )


def test_real_kernel_anchor_first3_renders_show_lr_naming_difference(
    oplb_real_kernel_captures,
):
    """Phase 1 (supplementary): The 'first 3' identity renders in the error
    message expose the ds_read_b128 vgpr-range difference between CMS (X0_I0 /
    high-index vgprs) and non-CMS (T0_I0-only / low-index linear vgprs).

    The ``compare_graphs`` error message includes
    ``first 3: sorted(only_ref)[:3]`` tuples — these are rendered assembly
    strings.  On the non-CMS side the first 3 identities are from the linear
    T0_I0 range (low vgpr indices like v[0:3], v[4:7]).  On the CMS side the
    first 3 identities include the X0_I0 vgpr range (high indices, distinct
    from any non-CMS render string).

    Because both sides read ds_read_b128 instructions and these dominate the
    identity diff, all rendered strings in the 'first 3' must start with
    'ds_read_b128' — confirming the divergence is in LR instructions.

    NOTE: When oplb is fixed, this test must flip to verify that no
    CaptureConsistencyError is raised (and so 'first 3' strings are absent).
    """
    from Tensile.Components.CMSValidator import build_dataflow_graph, compare_graphs
    from Tensile.Components.ScheduleCapture import CaptureConsistencyError

    ref_cap, cms_cap = oplb_real_kernel_captures
    ref_graph = build_dataflow_graph(ref_cap)
    subj_graph = build_dataflow_graph(cms_cap)

    with pytest.raises(CaptureConsistencyError) as excinfo:
        compare_graphs(ref_graph, subj_graph)

    msg = str(excinfo.value)

    # The first-3 renders in the error message for the ref side (non-CMS,
    # linear T0 only) and the subj side (CMS, alternating T0/X0) are
    # sorted identity tuples. Both sides show ds_read_b128 in the first 3
    # (LR instructions dominate the diff because they appear first in sorted
    # order vs MFMA).
    #
    # We cannot assert specific vgpr numbers because physical register
    # assignment varies with the kernel config. We CAN assert:
    #   (a) 'ds_read_b128' appears in the message (LR instructions diverge)
    #   (b) Both sides list identities — 'first 3:' appears twice
    assert "ds_read_b128" in msg, (
        f"Expected 'ds_read_b128' in the error message first-3 sample renders "
        f"(LR instructions are the primary diverging category).\n{msg}"
    )
    assert msg.count("first 3:") == 2, (
        f"Expected exactly 2 'first 3:' sections (one per side); "
        f"got {msg.count('first 3:')}.\n{msg}"
    )


# ===========================================================================
# Phase 2 — Trimmed minimal reproducer
# ===========================================================================
#
# The Phase 1 anchor confirms that the real kernel's LR/MFMA identity sets
# diverge due to the UsePLRPack T0_I0/X0_I0 double-register-file naming.
# Phase 2 isolates this in a hand-crafted capture pair using real rocisa
# instructions, proving the divergence is intrinsic to the naming difference
# and reproducible without a full kernel build.
#
# Instruction plan (from oplb_artifacts/oplb_lr_cms.txt and oplb_lr_noncms.txt):
#
#   CMS-side (UsePLRPack=True) — alternating T0 / X0 destinations:
#     LR0: ds_read_b128 v[T0+12:T0+15], v255 offset:9504    (T0 buffer, rIdx=0)
#     LR1: ds_read_b128 v[X0+28:X0+31], v255 offset:9568    (X0 buffer, rIdx=1)
#     LR2: ds_read_b128 v[T0+16:T0+19], v255 offset:16896   (T0 buffer, rIdx=0)
#     LR3: ds_read_b128 v[X0+36:X0+39], v255 offset:16960   (X0 buffer, rIdx=1)
#     Pack: VCvtPkF32toBF16 dst=v50, src0=v12, src1=v16     (consumes T0 reads)
#
#   non-CMS-side (single register file) — linear T0 only:
#     LR0: ds_read_b128 v[T0+0:T0+3],   v255 offset:9504    (same offsets)
#     LR1: ds_read_b128 v[T0+4:T0+7],   v255 offset:9568
#     LR2: ds_read_b128 v[T0+8:T0+11],  v255 offset:16896
#     LR3: ds_read_b128 v[T0+12:T0+15], v255 offset:16960
#     Pack: VCvtPkF32toBF16 dst=v50, src0=v0, src1=v8       (consumes linear T0)
#
#   Physical vgpr choices:
#     T0 base = 0  (simulates vgprValuA_T0_I0 = 0)
#     X0 base = 20 (simulates vgprValuA_X0_I0 = 20; non-overlapping with T0 range)
#
# Identity divergence: LR0..LR3 on both sides have different dst vgprs →
# 4 LR identities in ref-but-not-subj, 4 in subj-but-not-ref.
# Pack identity is the same on both sides (same dst vgpr 50, same LDS offsets
# are not part of Pack identity — Pack reads from vgpr, not LDS).
# Note: Pack dst/src form its identity via canonical render. If Pack srcs
# differ across sides, Pack also shows in the diff — we accept this as part
# of the trimmed reproducer.

# Physical vgpr base addresses for the minimal capture.
# T0 base: simulates vgprValuA_T0_I0 = 0 (first physical register).
_T0_BASE = 0
# X0 base: simulates vgprValuA_X0_I0 = 20 (second PLR pack buffer,
# non-overlapping with T0 range). In the real kernel X0 is at a higher index.
_X0_BASE = 20

# LDS offsets from oplb_lr_cms.txt (A-side reads, LDS0 buffer):
_OFFSETS = [9504, 9568, 16896, 16960]

# Pack scratch vgpr — same on both sides so the Pack identity doesn't
# add to the divergence count (making the LR-only divergence the subject).
_PACK_DST_VGPR = 50

# LR address placeholder vgpr — same on both sides (not tracked by dataflow).
_LR_ADDR_VGPR = 255


def _build_cms_capture():
    """CMS-side (UsePLRPack=True): alternating T0/X0 LR destinations.

    The first of each pair (rIdx=0) writes to T0 base + offset N.
    The second (rIdx=1) writes to X0 base + offset M (a different physical
    register range), producing an identity distinct from the non-CMS T0-only
    reads.

    Vgpr ranges chosen to match the real kernel pattern from
    oplb_artifacts/oplb_lr_cms.txt:
      T0_I0+12 = v[12:15], X0_I0+28 = v[28:31]  → offset 9504, 9568
      T0_I0+16 = v[16:19], X0_I0+36 = v[36:39]  → offset 16896, 16960

    (X0 base here is 20, so X0+8 = v[28:31], X0+16 = v[36:39].)
    """
    from rocisa.instruction import DSLoadB128, VCvtPkF32toBF16
    from rocisa.container import DSModifiers, vgpr
    from Tensile.Components.ScheduleCapture import (
        BODY_LABEL_ML, SLOT_KIND_MFMA, SlotKey, TaggedInstruction,
        WrappedInstruction, _populate_wrapper, assign_emission_ordinals,
        LoopBodyCapture,
    )
    # T0 destinations: 12, 16 (at T0_BASE=0, indices +12, +16)
    # X0 destinations: 28, 36 (at X0_BASE=20, indices +8, +16; physical = 28, 36)
    lr_params = [
        (_T0_BASE + 12, _OFFSETS[0], 'LRA0', 0),   # T0_I0+12, rIdx=0
        (_X0_BASE + 8,  _OFFSETS[1], 'LRA0', 1),   # X0_I0+28 (=20+8), rIdx=1
        (_T0_BASE + 16, _OFFSETS[2], 'LRA0', 2),   # T0_I0+16, rIdx=0
        (_X0_BASE + 16, _OFFSETS[3], 'LRA0', 3),   # X0_I0+36 (=20+16), rIdx=1
    ]
    insts = []
    for dst_start, lds_offset, cat, slot_idx in lr_params:
        raw = DSLoadB128(
            dst=vgpr(dst_start, 4),
            src=vgpr(_LR_ADDR_VGPR, 1),
            ds=DSModifiers(offset=lds_offset),
        )
        wi = WrappedInstruction(raw)
        ti = TaggedInstruction(
            wrapped=wi, category=cat,
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=slot_idx, sequence=0),
        )
        _populate_wrapper(wi, category=cat)
        insts.append(ti)

    # Pack: consumes a T0-indexed LR output (v[_T0_BASE+12]).
    # Using VCvtPkF32toBF16 as in the real TF32 emulation kernel.
    pack_raw = VCvtPkF32toBF16(
        dst=vgpr(_PACK_DST_VGPR, 1),
        src0=vgpr(_T0_BASE + 12, 1),
        src1=vgpr(_T0_BASE + 16, 1),
    )
    pack_wi = WrappedInstruction(pack_raw)
    pack_ti = TaggedInstruction(
        wrapped=pack_wi, category='PackA0',
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=4, sequence=0),
    )
    _populate_wrapper(pack_wi, category='PackA0')
    insts.append(pack_ti)

    assign_emission_ordinals(insts)
    return LoopBodyCapture(instructions=insts)


def _build_noncms_capture():
    """non-CMS-side (single register file): linear T0-only LR destinations.

    Same LDS offsets as the CMS capture, but all four reads go to the T0
    range with linearly increasing indices (simulating vgprValuA_T0_I0+0,
    +4, +8, +12 — no X0 buffer).

    Identity comparison: these 4 LR identities are DISTINCT from the 4 LR
    identities in the CMS capture (different physical vgpr dst ranges), so
    ``compare_graphs`` fires ``CaptureConsistencyError``.
    """
    from rocisa.instruction import DSLoadB128, VCvtPkF32toBF16
    from rocisa.container import DSModifiers, vgpr
    from Tensile.Components.ScheduleCapture import (
        BODY_LABEL_ML, SLOT_KIND_MFMA, SlotKey, TaggedInstruction,
        WrappedInstruction, _populate_wrapper, assign_emission_ordinals,
        LoopBodyCapture,
    )
    # T0-only destinations: 0, 4, 8, 12 (linear vgprValuA_T0_I0+N)
    lr_params = [
        (_T0_BASE + 0,  _OFFSETS[0], 'LRA0', 0),
        (_T0_BASE + 4,  _OFFSETS[1], 'LRA0', 1),
        (_T0_BASE + 8,  _OFFSETS[2], 'LRA0', 2),
        (_T0_BASE + 12, _OFFSETS[3], 'LRA0', 3),
    ]
    insts = []
    for dst_start, lds_offset, cat, slot_idx in lr_params:
        raw = DSLoadB128(
            dst=vgpr(dst_start, 4),
            src=vgpr(_LR_ADDR_VGPR, 1),
            ds=DSModifiers(offset=lds_offset),
        )
        wi = WrappedInstruction(raw)
        ti = TaggedInstruction(
            wrapped=wi, category=cat,
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=slot_idx, sequence=0),
        )
        _populate_wrapper(wi, category=cat)
        insts.append(ti)

    # Pack: consumes a T0-indexed LR output (v[_T0_BASE+0]).
    pack_raw = VCvtPkF32toBF16(
        dst=vgpr(_PACK_DST_VGPR, 1),
        src0=vgpr(_T0_BASE + 0, 1),
        src1=vgpr(_T0_BASE + 8, 1),
    )
    pack_wi = WrappedInstruction(pack_raw)
    pack_ti = TaggedInstruction(
        wrapped=pack_wi, category='PackA0',
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=4, sequence=0),
    )
    _populate_wrapper(pack_wi, category='PackA0')
    insts.append(pack_ti)

    assign_emission_ordinals(insts)
    return LoopBodyCapture(instructions=insts)


def _wrap_in_four_part(ml_capture):
    """Wrap a single ML body in a FourPartCapture with minimal filler bodies.

    Filler bodies use vgpr ranges well above any LR/Pack resources used in
    the ML body, avoiding aliasing.
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
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV, 200, 204, 208)},
        n_gl={0: _filler(BODY_LABEL_NGL, 220, 224, 228)},
        n_ll={0: _filler(BODY_LABEL_NLL, 240, 244, 248)},
        num_mfma=1, num_codepaths=1, source='cms',
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _generate_png_figures(noncms_capture, cms_capture):
    """Generate PNG dataflow-graph figures for both captures.

    Writes into ``oplb_artifacts/figures/`` (relative to the tensilelite
    working directory, which is the test runner's cwd when run with
    ``PYTHONPATH=$PWD python -m pytest ...`` from the tensilelite dir).

    This function is called BEFORE ``compare_graphs`` so the figures are
    generated regardless of whether the comparison raises.

    Returns ``(noncms_png_path, cms_png_path)`` — the paths written.
    """
    from Tensile.Components.CMSValidator import build_dataflow_graph
    from Tensile.Components.CMSValidatorVisualization import visualize_dataflow_graph

    ref_graph = build_dataflow_graph(noncms_capture)
    subj_graph = build_dataflow_graph(cms_capture)

    artifacts_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "oplb_artifacts", "figures",
    )
    artifacts_dir = os.path.normpath(artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)

    noncms_path = os.path.join(artifacts_dir, "oplb_minimal_noncms.png")
    cms_path = os.path.join(artifacts_dir, "oplb_minimal_cms.png")

    visualize_dataflow_graph(
        ref_graph, noncms_path,
        title="oplb minimal: non-CMS (T0_I0 linear, single register file)",
    )
    visualize_dataflow_graph(
        subj_graph, cms_path,
        title="oplb minimal: CMS (alternating T0_I0/X0_I0, UsePLRPack)",
    )
    return noncms_path, cms_path


class TestOplbRegisterNamingMinimal:
    """Phase 2: trimmed LR-naming minimal reproducer for the oplb residual.

    Two LR instruction sets with the same LDS offsets but different vgpr
    destinations:
      - CMS-side: alternating T0 (low indices) and X0 (high indices) — the
        UsePLRPack double-register-file pattern
      - non-CMS-side: linear T0-only destinations

    ``compare_graphs`` raises ``CaptureConsistencyError`` because the LR
    identity tuples (which include the canonical assembly render including
    the vgpr range) differ between the two captures.

    NOTE: When oplb is fixed (bead option D — symbolic register
    canonicalization), the assertion ``len(lr_only_in_ref) == 4`` must flip
    to ``0`` — the fix will normalize T0_I0/X0_I0 into the same identity.
    """

    def test_cms_capture_has_alternating_t0_x0_identities(self):
        """The CMS capture (alternating T0/X0) produces LR identity tuples
        with vgpr ranges from BOTH the T0 base and X0 base.

        T0 range starts at _T0_BASE (0), X0 range starts at _X0_BASE (20).
        The four LR identities include two T0-range reads (v[12:15], v[16:19])
        and two X0-range reads (v[28:31], v[36:39]).
        """
        from Tensile.Components.CMSValidator import (
            build_dataflow_graph, _DATA_FLOW_CATEGORIES, _category,
        )
        from Tensile.Components.ScheduleCapture import BODY_LABEL_ML

        cms_cap = _build_cms_capture()
        graph = build_dataflow_graph(_wrap_in_four_part(cms_cap))

        lr_identities = {
            n.identity for n in graph.nodes.values()
            if _category(n.rocisa_inst) is not None
            and _category(n.rocisa_inst).value == "LR"
        }
        # Expect 4 distinct LR identities (one per ds_read_b128 instruction)
        assert len(lr_identities) == 4, (
            f"Expected 4 LR identities in the CMS capture "
            f"(4 ds_read_b128 instructions); got {len(lr_identities)}."
        )
        # At least one identity should have a high-index vgpr range (X0 base)
        # and at least one should have a low-index vgpr range (T0 base).
        # The rendered form is 'ds_read_b128 v[N:N+3], ...' — check that the
        # vgpr start indices span both ranges.
        renders = [ident[0] for ident in lr_identities]
        t0_range_present = any(
            f"v[{_T0_BASE + n}" in r
            for r in renders
            for n in [12, 16]
        )
        x0_range_present = any(
            f"v[{_X0_BASE + n}" in r
            for r in renders
            for n in [8, 16]
        )
        assert t0_range_present, (
            f"Expected T0-base vgpr indices in CMS LR identities; "
            f"renders={renders}"
        )
        assert x0_range_present, (
            f"Expected X0-base vgpr indices in CMS LR identities; "
            f"renders={renders}"
        )

    def test_noncms_capture_has_linear_t0_only_identities(self):
        """The non-CMS capture (T0-only) produces LR identity tuples with
        vgpr ranges only in the T0 base range (low indices).

        No X0-base reads appear — all four identities use v[N:N+3] with
        N in {0, 4, 8, 12} (T0_BASE + offset).
        """
        from Tensile.Components.CMSValidator import (
            build_dataflow_graph, _category,
        )

        noncms_cap = _build_noncms_capture()
        graph = build_dataflow_graph(_wrap_in_four_part(noncms_cap))

        lr_identities = {
            n.identity for n in graph.nodes.values()
            if _category(n.rocisa_inst) is not None
            and _category(n.rocisa_inst).value == "LR"
        }
        assert len(lr_identities) == 4, (
            f"Expected 4 LR identities in the non-CMS capture; "
            f"got {len(lr_identities)}."
        )
        renders = [ident[0] for ident in lr_identities]
        x0_range_absent = not any(
            f"v[{_X0_BASE + n}" in r
            for r in renders
            for n in range(0, 20, 4)
        )
        assert x0_range_absent, (
            f"Expected only T0-base vgpr indices in non-CMS LR identities "
            f"(no X0-range reads); renders={renders}"
        )

    def test_compare_graphs_raises_capture_consistency_error_on_lr_divergence(self):
        """Phase 2 end-to-end: compare_graphs on the minimal T0/X0 capture pair
        raises CaptureConsistencyError with exactly 4 LR identities diverging
        per side (the 4 ds_read_b128 instructions with different vgpr dsts).

        Also generates PNG figures for both captures as a side effect (into
        oplb_artifacts/figures/). Figures are generated BEFORE compare_graphs
        runs so they exist even when the test passes (and the comparison raises).

        NOTE: When oplb is fixed, the assertion must flip to verify that
        compare_graphs does NOT raise — the 4 LR divergences will be collapsed
        to 0 by register-name canonicalization.
        """
        from Tensile.Components.CMSValidator import (
            build_dataflow_graph, compare_graphs,
        )
        from Tensile.Components.ScheduleCapture import CaptureConsistencyError

        noncms_cap = _build_noncms_capture()
        cms_cap = _build_cms_capture()

        # Phase 3: generate PNG figures before the comparison (so the
        # figures exist regardless of test outcome).
        try:
            _generate_png_figures(
                _wrap_in_four_part(noncms_cap),
                _wrap_in_four_part(cms_cap),
            )
        except Exception as fig_exc:
            # Figure generation is best-effort; a missing matplotlib/networkx
            # in the test environment should not block the core assertion.
            import warnings
            warnings.warn(
                f"PNG figure generation failed (non-fatal): {fig_exc}",
                stacklevel=2,
            )

        ref_graph = build_dataflow_graph(_wrap_in_four_part(noncms_cap))
        subj_graph = build_dataflow_graph(_wrap_in_four_part(cms_cap))

        with pytest.raises(CaptureConsistencyError) as excinfo:
            compare_graphs(ref_graph, subj_graph)

        msg = str(excinfo.value)

        # Pin: exactly 4 LR identities diverge on each side (one per
        # ds_read_b128 instruction with a different vgpr destination).
        assert "in reference but not subject: 4 identities" in msg, (
            f"Expected 4 ref-but-not-subj LR identities (the non-CMS T0-only "
            f"reads, absent from the CMS X0-interleaved capture);\n{msg}"
        )
        assert "in subject but not reference: 4 identities" in msg, (
            f"Expected 4 subj-but-not-ref LR identities (the CMS X0-range "
            f"reads, absent from the non-CMS T0-only capture);\n{msg}"
        )
        # Both sides report category LR
        assert "'LR'" in msg, (
            f"Expected 'LR' category in the identity diff (ds_read_b128 "
            f"instructions are classified as LR);\n{msg}"
        )
        # Confirm the divergence count is purely from LR (not MFMA), since
        # the trimmed capture has no MFMAs in the ML body.
        assert "'MFMA'" not in msg, (
            f"Expected no 'MFMA' category in the trimmed capture's identity diff "
            f"(no ML-body MFMAs in the minimal fixture);\n{msg}"
        )

    def test_png_figures_are_written(self):
        """Phase 3: PNG figure files exist and are non-empty after generation.

        Generates the figures directly (independent of the compare_graphs
        test) and asserts that both output files are real PNGs (file size > 0).
        """
        noncms_cap = _build_noncms_capture()
        cms_cap = _build_cms_capture()

        noncms_path, cms_path = _generate_png_figures(
            _wrap_in_four_part(noncms_cap),
            _wrap_in_four_part(cms_cap),
        )

        assert os.path.exists(noncms_path), (
            f"non-CMS PNG figure was not written to {noncms_path}"
        )
        assert os.path.exists(cms_path), (
            f"CMS PNG figure was not written to {cms_path}"
        )
        noncms_size = os.path.getsize(noncms_path)
        cms_size = os.path.getsize(cms_path)
        assert noncms_size > 0, (
            f"non-CMS PNG figure at {noncms_path} is empty (0 bytes)"
        )
        assert cms_size > 0, (
            f"CMS PNG figure at {cms_path} is empty (0 bytes)"
        )
