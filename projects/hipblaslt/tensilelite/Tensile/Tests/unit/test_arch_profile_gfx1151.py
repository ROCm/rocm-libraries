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
"""Smoke tests for the gfx1151 (RDNA 3.5) ArchProfile path.

Confirms:
  1. The CDNA 4 default profile remains bit-identical when no profile is
     attached (regression guard for the ArchProfile refactor).
  2. The gfx1151 profile resolver returns the placeholder profile for the
     (11, 5, 1) ISA tuple.
  3. A synthetic gfx1151-shaped capture (WMMA + LR + LW) builds via
     `build_dataflow_graph` without raising
     `CaptureUnknownInstructionError`.
  4. The four pair-specific quad-cycle helpers consult the resolved
     profile (different expected values for CDNA 4 vs gfx1151).

The full live-build audit (assembly emit + captured CMS schedules) is
out of sandbox reach; that work is filed as sub-bead `l6q.1.live`.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    ArchProfile,
    BODY_LABEL_ML,
    DataflowGraph,
    FourPartCapture,
    _DEFAULT_CDNA4_ARCH_PROFILE,
    _GFX1151_ARCH_PROFILE,
    _cvt_to_mfma_gap_ok,
    _mfma_pack_to_cvt_gap_ok,
    _min_issue_quad_cycles_for,
    _mfma_finish_cycles_for,
    _resolve_arch_profile,
    _resolve_arch_profile_for_isa,
    build_dataflow_graph,
)

from Tensile.Tests.unit.dataflow_fixtures import (
    make_capture,
    make_lr,
    make_lw,
    make_mfma,
)


def test_default_profile_is_cdna4():
    """The unattached path resolves to the CDNA 4 default profile.

    Guards the bit-identical contract for every existing test that does
    not attach an ArchProfile to its FourPartCapture / DataflowGraph.
    """
    profile = _resolve_arch_profile(None)
    assert profile is _DEFAULT_CDNA4_ARCH_PROFILE
    assert profile.name == "CDNA4"
    assert profile.isa == (9, 5, 0)
    # Historical literals (CDNA 4 ISA section 7.6).
    assert profile.standard_mfma_finish_cycles == 3
    assert profile.mfma_4x4_finish_cycles == 1
    assert profile.cvt_before_mfma_quad_cycles == 2
    assert profile.mfma_4x4_before_cvt_quad_cycles == 5
    assert profile.mfma_type_switch_threshold_from_standard == 5
    assert profile.mfma_type_switch_threshold_from_4x4 == 3
    assert profile.default_issue_quad_cycles == 1


def test_gfx1151_profile_resolves_by_isa():
    """`(11, 5, 1)` resolves to the gfx1151 profile."""
    profile = _resolve_arch_profile_for_isa((11, 5, 1))
    assert profile is _GFX1151_ARCH_PROFILE
    assert profile.name == "GFX1151"
    assert profile.isa == (11, 5, 1)


def test_unknown_isa_falls_back_to_cdna4():
    """Unknown ISA tuples must NOT raise — fall back to CDNA 4 (the
    historical hardcoded path) so uncharacterized arches keep working.
    """
    profile = _resolve_arch_profile_for_isa((42, 0, 0))
    assert profile is _DEFAULT_CDNA4_ARCH_PROFILE


def test_none_isa_falls_back_to_cdna4():
    """`kernel["ISA"]` may be missing on test fixtures — must default."""
    profile = _resolve_arch_profile_for_isa(None)
    assert profile is _DEFAULT_CDNA4_ARCH_PROFILE


def test_gfx1151_profile_values_match_rdna35_isa():
    """Sub-beads l6q.1.t1, l6q.1.t2, l6q.1.t4 (alias beads xlh, j3n, b0t)
    replaced the conservative placeholders with values sourced from the
    RDNA 3.5 ISA reference. This test pins the new contract:

      - WMMA finish window: 1 (RDNA 3.5 §7.9.1, page 77 — "1 V_NOP or
        unrelated VALU instruction in between two WMMA instructions").
      - 4x4 family fields: 0 (no 4x4 WMMA family on RDNA 3.5; Table 33
        page 75 lists only 16x16x16 opcodes — fields are unreachable).
      - Wave32 base issue cost: 1 (RDNA 3.5 §2.1, page 9).

    Two fields remain conservative placeholders pending future audit:
      - `cvt_before_mfma_quad_cycles` (TF32 emul off on gfx1151 today).
      - `mfma_type_switch_threshold_from_standard` (ISA inconclusive;
        §7.9.1 documents the case as "Hardware may stall" without a
        quantified cycle window).
    """
    gfx = _GFX1151_ARCH_PROFILE
    # Documented values from the RDNA 3.5 ISA.
    assert gfx.standard_mfma_finish_cycles == 1
    assert gfx.mfma_4x4_finish_cycles == 0
    assert gfx.mfma_4x4_before_cvt_quad_cycles == 0
    assert gfx.mfma_type_switch_threshold_from_4x4 == 0
    assert gfx.default_issue_quad_cycles == 1
    # Conservative placeholders that remain (ISA inconclusive / branch
    # unreachable on gfx1151 today).
    assert gfx.cvt_before_mfma_quad_cycles >= _DEFAULT_CDNA4_ARCH_PROFILE.cvt_before_mfma_quad_cycles
    assert (gfx.mfma_type_switch_threshold_from_standard
            >= _DEFAULT_CDNA4_ARCH_PROFILE.mfma_type_switch_threshold_from_standard)


def test_gfx1151_capture_builds_without_unknown_instruction_error():
    """Smoke test: a synthetic gfx1151-shaped LR/LW/WMMA body builds via
    `build_dataflow_graph` without raising
    `CaptureUnknownInstructionError`.

    WMMA on gfx1151 reuses the `MFMAInstruction` rocisa class (the
    rendered mnemonic differs — `v_wmma_*` vs `v_mfma_*` — but the
    class identity is the same). The validator's `_is_mfma`
    discriminator therefore already claims WMMA without any code
    change; this test guards the "no missing classification" property.
    """
    # Construct a minimal gfx1151-style main loop:
    #   LW v[8:11] -> LDS@0
    #   LR v[12:13] <- LDS@0
    #   WMMA v[16:19] += v[12:13] x v[14:15]   (16x16x16 shape)
    body_instructions = [
        make_lw(src_vgpr_start=8, src_vgpr_count=4, lds_offset=0, slot=0),
        make_lr(dst_vgpr_start=12, dst_vgpr_count=2, lds_offset=0, slot=1),
        make_mfma(c_dst_start=16, a_src_start=12, b_src_start=14, slot=2,
                  variant=[16, 16, 16, 1]),
    ]
    body = make_capture(BODY_LABEL_ML, body_instructions)

    capture = FourPartCapture(
        main_loop={0: body},
        main_loop_prev={},
        n_gl={},
        n_ll={},
        num_mfma=1,
        num_codepaths=1,
        source="default-sia3",
        num_mfma_per_subiter=1,
        arch_profile=_GFX1151_ARCH_PROFILE,
    )

    # Must not raise CaptureUnknownInstructionError.
    graph = build_dataflow_graph(capture)
    assert graph is not None
    assert graph.arch_profile is _GFX1151_ARCH_PROFILE

    # Sanity: we got at least the WMMA node in the graph.
    mfma_nodes = [n for n in graph.nodes.values() if n.category == "MFMA"]
    assert len(mfma_nodes) == 1, (
        f"Expected exactly one MFMA-categorized node (the WMMA), got "
        f"{len(mfma_nodes)}."
    )


def test_finish_cycles_uses_arch_profile():
    """`_mfma_finish_cycles_for` returns per-arch values when a profile
    is passed.
    """
    # Standard MFMA shape -> profile.standard_mfma_finish_cycles.
    inst_std = type("FakeMFMA", (), {"variant": [16, 16, 16, 1]})()
    cdna_finish = _mfma_finish_cycles_for(inst_std, _DEFAULT_CDNA4_ARCH_PROFILE)
    gfx_finish = _mfma_finish_cycles_for(inst_std, _GFX1151_ARCH_PROFILE)
    assert cdna_finish == _DEFAULT_CDNA4_ARCH_PROFILE.standard_mfma_finish_cycles
    assert gfx_finish == _GFX1151_ARCH_PROFILE.standard_mfma_finish_cycles
    # The two MUST differ given the placeholder profile is conservatively
    # larger; this guards against future "oops both arches collapsed to 1"
    # regressions.
    assert cdna_finish != gfx_finish

    # 4x4 PackMFMA shape (per-arch override applies here too).
    inst_4x4 = type("FakeMFMA4x4", (), {"variant": [4, 4, 4, 16]})()
    assert _mfma_finish_cycles_for(inst_4x4, _DEFAULT_CDNA4_ARCH_PROFILE) == 1
    assert (_mfma_finish_cycles_for(inst_4x4, _GFX1151_ARCH_PROFILE)
            == _GFX1151_ARCH_PROFILE.mfma_4x4_finish_cycles)


def test_min_issue_cycles_uses_arch_profile_default():
    """`_min_issue_quad_cycles_for` honors `profile.default_issue_quad_cycles`
    for non-SNop instructions.
    """
    fake_alu = type("FakeALU", (), {})()
    # Both profiles set default_issue_quad_cycles == 1 today; assert that
    # the helper returns the profile's value rather than a hardcoded 1.
    assert (_min_issue_quad_cycles_for(fake_alu, _DEFAULT_CDNA4_ARCH_PROFILE)
            == _DEFAULT_CDNA4_ARCH_PROFILE.default_issue_quad_cycles)
    assert (_min_issue_quad_cycles_for(fake_alu, _GFX1151_ARCH_PROFILE)
            == _GFX1151_ARCH_PROFILE.default_issue_quad_cycles)


def test_pair_helpers_consult_graph_arch_profile():
    """`_cvt_to_mfma_gap_ok` and `_mfma_pack_to_cvt_gap_ok` read their
    expected-cycle threshold from the graph's `arch_profile`.

    Verified by attaching the gfx1151 profile to a synthetic graph and
    confirming the expected value differs from CDNA 4.
    """
    # Build a degenerate graph whose only purpose is to carry the
    # arch_profile field; the helpers fall through their `subj_graph
    # is None` branch by checking the graph for arch first.
    cdna_graph = DataflowGraph(
        nodes={}, edges=[], captures={},
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )
    gfx_graph = DataflowGraph(
        nodes={}, edges=[], captures={},
        arch_profile=_GFX1151_ARCH_PROFILE,
    )
    # Producer / consumer placeholders — their identity doesn't matter
    # for the expected-value branch since the helpers compute
    # `actual = cumulative_issue_cycles(...)` which returns 0 on an
    # empty graph (consumer-not-found defensive return).
    fake_node = type("FakeNode", (), {
        "rocisa_inst": None,
        "position": None,
        "category": "MFMA",
        "tagged_inst": None,
        "body_label": BODY_LABEL_ML,
    })()
    # `_cvt_to_mfma_gap_ok` returns (ok, expected, actual). expected =
    # profile.cvt_before_mfma_quad_cycles.
    _, expected_cdna, _ = _cvt_to_mfma_gap_ok(fake_node, fake_node, cdna_graph)
    _, expected_gfx, _ = _cvt_to_mfma_gap_ok(fake_node, fake_node, gfx_graph)
    assert expected_cdna == _DEFAULT_CDNA4_ARCH_PROFILE.cvt_before_mfma_quad_cycles
    assert expected_gfx == _GFX1151_ARCH_PROFILE.cvt_before_mfma_quad_cycles
    assert expected_cdna != expected_gfx, (
        "CDNA 4 and gfx1151 must report different expected cycles for the "
        "CVT->MFMA gap; the placeholder profile picks a conservative bump."
    )

    # Same shape for `_mfma_pack_to_cvt_gap_ok`.
    _, expected_cdna2, _ = _mfma_pack_to_cvt_gap_ok(fake_node, fake_node, cdna_graph)
    _, expected_gfx2, _ = _mfma_pack_to_cvt_gap_ok(fake_node, fake_node, gfx_graph)
    assert (expected_cdna2
            == _DEFAULT_CDNA4_ARCH_PROFILE.mfma_4x4_before_cvt_quad_cycles)
    assert (expected_gfx2
            == _GFX1151_ARCH_PROFILE.mfma_4x4_before_cvt_quad_cycles)
    assert expected_cdna2 != expected_gfx2


def test_arch_profile_is_frozen():
    """`ArchProfile` must be immutable so the singleton defaults can be
    safely shared across the suite without any caller mutating them.
    """
    with pytest.raises(Exception):
        # frozen dataclass raises FrozenInstanceError; we only assert
        # that mutation is rejected, not the exact exception type.
        _DEFAULT_CDNA4_ARCH_PROFILE.standard_mfma_finish_cycles = 999
