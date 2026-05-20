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
"""Tail-loop emit content assertions for the subtile BF16 any-K tail.

Drives `KernelWriter._emitTailLoopScaffoldSubtile` at ASEM ∈ {32, 8,
2, 1} via `setdefault_tail_scaffold_kernel_keys(..., asem=...)` and
pins the K%32 / K%8 / K%2 / odd-K emit shape.
"""
import re

import pytest

from Tensile.Tests.unit._subtile_tailloop_fixtures import (
    build_minimal_subtile_kwa,
    setdefault_tail_scaffold_kernel_keys,
    wrap_with_skiptoend,
)
from Tensile.Tests.unit.test_subtile_tailloop_emit import (
    _create_kernel,
    _extract_tail_section,
)


# ── Driver ──────────────────────────────────────────────────────────────────

def _emit_anyk_tail_asm(*, asem: int, pgr: int = 0, MT0: int = 128, MT1: int = 128) -> str:
    """Emit a BF16 subtile tail scaffold at the requested ASEM."""
    kernel = _create_kernel(MT0=MT0, MT1=MT1, fp4=False,
                            depthU=64, no_tail_loop=False)
    setdefault_tail_scaffold_kernel_keys(kernel, pgr, asem=asem)

    kwa = build_minimal_subtile_kwa(kernel)
    tPA = {"is_sparse": False, "tpsMetadata": None}
    tPB = {"is_sparse": False, "tpsMetadata": None}
    module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
    return wrap_with_skiptoend(module)


# ── Tests: K%8 (ASEM=8) ──────────────────────────────────────────────────────

class TestAnyKEmit_K8:
    """K%8 reuses the K32 emit path unchanged: the per-lane mask
    granularity is already `numMIInUnroll=8` for bf16.
    """

    def test_k8_emit_matches_k32_shape(self):
        asm_k32 = _emit_anyk_tail_asm(asem=32, pgr=0)
        asm_k8 = _emit_anyk_tail_asm(asem=8, pgr=0)
        tail_k32 = _extract_tail_section(asm_k32)
        tail_k8 = _extract_tail_section(asm_k8)
        assert tail_k32, "K32 baseline emitted no tail block"
        assert tail_k8, "K8 emit produced no tail block"

        cmp_k32 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k32)
        cmp_k8 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k8)
        cnd_k32 = re.findall(r"v_cndmask_b32.*[vV]alu", tail_k32)
        cnd_k8 = re.findall(r"v_cndmask_b32.*[vV]alu", tail_k8)

        assert len(cmp_k8) == len(cmp_k32), (
            f"K%8 (ASEM=8) lane-mask cmp count {len(cmp_k8)} must "
            f"equal K%32 (ASEM=32) baseline {len(cmp_k32)}. K%8 is "
            f"expected to reuse the K32 emit path unchanged."
        )
        assert len(cnd_k8) == len(cnd_k32), (
            f"K%8 (ASEM=8) lane-mask cndmask count {len(cnd_k8)} must "
            f"equal K%32 (ASEM=32) baseline {len(cnd_k32)}."
        )


# ── Tests: K%2 (ASEM=2) ──────────────────────────────────────────────────────

class TestAnyKEmit_K2:
    """K%2 emits the per-VGPR refinement (`_emitTailByteShiftMaskSubtile`
    Step 1) on top of the coarse per-lane cndmask. Negative pins
    verify the helper does NOT emit a `v_lshl(rev)_b64` byte-shift
    pattern (the name predates that approach being abandoned).
    """

    def test_k2_emits_per_vgpr_refinement(self):
        asm_k32 = _emit_anyk_tail_asm(asem=32, pgr=0)
        asm_k2 = _emit_anyk_tail_asm(asem=2, pgr=0)
        tail_k2 = _extract_tail_section(asm_k2)
        tail_k32 = _extract_tail_section(asm_k32)
        assert tail_k2, "K%2 emit produced no tail block"
        assert tail_k32, "K%32 baseline emit produced no tail block"

        # Step-1 refinement adds per-VGPR cmps beyond the per-mmak
        # baseline cmp. Strictly greater than the K32 cmp count is
        # the structural fingerprint of the per-VGPR loop firing.
        cmp_k2 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k2)
        cmp_k32 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k32)
        assert len(cmp_k2) > len(cmp_k32), (
            f"K%2 emit must add per-VGPR cmps beyond the K%32 baseline; "
            f"got cmp_k2={len(cmp_k2)} vs cmp_k32={len(cmp_k32)}."
        )

        # Per-VGPR cndmask zero against valuA / valuB.
        cnd_match = re.search(
            r"v_cndmask_b32[^\n]*(?:valuA|valuB|Valu[AB])", tail_k2
        ) or re.search(r"v_cndmask_b32[^\n]*per-VGPR byte refine", tail_k2)
        assert cnd_match, (
            "K%2 emit missing per-VGPR `v_cndmask_b32` against "
            "ValuA/ValuB. Refinement step 1 must zero each boundary "
            "VGPR. Tail excerpt:\n" + tail_k2[:1500]
        )

        # ir-th VGPR's K_pos offset must appear in the v_add chain.
        # For vgprPerInUnroll=4 and elementsPerVgpr=2 the offsets are
        # {0, 2, 4, 6} per mmak slice. At least one of {2, 4, 6}
        # must show up (0 also shows up as the original kPosCur
        # for the coarse mask).
        kpos_offsets = re.findall(
            r"v_add_u32\s+v\d+,\s+(\d+),\s+v\d+\s*//\s*(?:byteRefine|kPosCur|byteHi)",
            tail_k2,
        )
        observed_offsets = {int(x) for x in kpos_offsets}
        # ir=1, 2, 3 -> offsets 2, 4, 6 (or larger when mmak>0).
        # Only require ir=1 (offset 2) so the test stays robust to
        # other mmak slice offsets.
        assert 2 in observed_offsets, (
            f"K%2 per-VGPR refinement missing ir=1 K_pos offset (=2) "
            f"in kPosBase add. Observed offsets: {sorted(observed_offsets)}. "
            f"Tail excerpt:\n{tail_k2[:1500]}"
        )

        # Negative: byte-shift refinement must NOT be emitted.
        assert re.search(r"v_lsh(?:l|lrev)_b64", tail_k2) is None, (
            "K%2 emit must NOT contain v_lshl(rev)_b64."
        )
        assert re.search(r"s_and_b32[^\n]*\b7\b", tail_k2) is None, (
            "K%2 emit must NOT compute `K_remain & 7`."
        )


# ── Tests: K%1 (odd K, ASEM=1) ───────────────────────────────────────────────

class TestAnyKEmit_K1:
    """ASEM=1 (odd K). `buffer_load_*_d16 ... lds` is not legal on
    gfx950, so the scaffold relies on OOB-clipped wide DTL + the
    runtime-gated hi16 clear in `_emitTailByteShiftMaskSubtile` Step 2
    (`s_and_b32 LoopCounterL, 1` -> branch -> `v_and_b32 0xFFFF` +
    `v_cndmask_b32`).
    """

    def test_k1_no_narrow_load_and_has_hi16_clear(self):
        asm = _emit_anyk_tail_asm(asem=1, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail, "ASEM=1 emit produced no tail block"

        # Narrow d16 load must not be emitted (illegal on gfx950).
        assert re.search(
            r"buffer_load_d16_b16|buffer_load_short_d16",
            asm,
        ) is None, (
            "ASEM=1 emit must NOT contain the narrow DTL load: "
            "`buffer_load_*_d16 ... lds` is not legal on gfx950."
        )

        skip_label = re.search(
            r"label_SubtileTailByteHi16Skip", tail
        )
        assert skip_label is not None, (
            "ASEM=1 emit missing odd-K hi16 skip label "
            "`SubtileTailByteHi16Skip`."
        )

        # Runtime K-odd gate (s_and_b32 ..., 1).
        and_one = re.search(
            r"s_and_b32[^\n]*\b1\b[^\n]*LoopCounterL",
            tail,
        ) or re.search(
            r"s_and_b32[^\n]*LoopCounterL[^\n]*\b1\b",
            tail,
        )
        assert and_one is not None, (
            "ASEM=1 emit missing `s_and_b32 ..., LoopCounterL, 1` "
            "(K_remain & 1 gate). Tail excerpt:\n" + tail[:2000]
        )

        and_ffff = re.search(r"v_and_b32[^\n]*0xffff", tail, re.IGNORECASE)
        assert and_ffff is not None, (
            "ASEM=1 emit missing `v_and_b32 ..., 0xFFFF, ...` for "
            "hi16 clear of the odd-K boundary VGPR."
        )

        cnd_hi = re.search(
            r"v_cndmask_b32[^\n]*hi16",
            tail,
        )
        assert cnd_hi is not None, (
            "ASEM=1 emit missing `v_cndmask_b32` for odd-K hi16 "
            "clear. Tail excerpt:\n" + tail[:2000]
        )


# ── Regression net: K%32 emit must stay identical ────────────────────────────

class TestAnyKEmit_K32Unchanged:
    """K%32 (ASEM=32) regression net: structural fingerprint (cmp /
    cndmask presence + absence of any-K helper opcodes) so the K32
    emit cannot silently drift when the any-K helpers change.
    """

    K32_FINGERPRINT_HAS_LANE_CMP = True
    K32_FINGERPRINT_HAS_CNDMASK_VALUA = True
    K32_FINGERPRINT_HAS_CNDMASK_VALUB = True
    K32_FINGERPRINT_NO_LSHL_B64 = True
    K32_FINGERPRINT_NO_D16_B16 = True
    K32_FINGERPRINT_NO_AND_7 = True

    def test_k32_unchanged_with_new_helpers(self):
        asm = _emit_anyk_tail_asm(asem=32, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail, "K%32 emit produced no tail block"

        cmp_match = re.search(r"v_cmp_ge_i32.*LoopCounterL", tail)
        assert (cmp_match is not None) == self.K32_FINGERPRINT_HAS_LANE_CMP, (
            "K%32 emit lane cmp shape changed"
        )
        cnd_a = re.search(r"v_cndmask_b32.*[vV]aluA", tail)
        assert (cnd_a is not None) == self.K32_FINGERPRINT_HAS_CNDMASK_VALUA, (
            "K%32 emit cndmask valuA shape changed"
        )
        cnd_b = re.search(r"v_cndmask_b32.*[vV]aluB", tail)
        assert (cnd_b is not None) == self.K32_FINGERPRINT_HAS_CNDMASK_VALUB, (
            "K%32 emit cndmask valuB shape changed"
        )

        no_lshl = re.search(r"v_lsh(?:l|lrev)_b64", tail) is None
        assert no_lshl == self.K32_FINGERPRINT_NO_LSHL_B64, (
            "K%32 emit must NOT contain v_lshl(rev)_b64"
        )
        no_d16 = re.search(
            r"buffer_load_d16_b16|buffer_load_short_d16|BufferLoadD16B16",
            tail,
        ) is None
        assert no_d16 == self.K32_FINGERPRINT_NO_D16_B16, (
            "K%32 emit must NOT contain buffer_load_d16_b16"
        )
        no_and_7 = re.search(
            r"s_and_b32[^\n]*(?:\b7\b|\b8\s*-\s*1\b)", tail
        ) is None
        assert no_and_7 == self.K32_FINGERPRINT_NO_AND_7, (
            "K%32 emit must NOT compute K_remain mod 8 via "
            "`s_and_b32 ..., 7`"
        )
