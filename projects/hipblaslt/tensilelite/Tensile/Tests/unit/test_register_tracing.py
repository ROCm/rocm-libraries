################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
"""Tests for register-based dependency tracing in derive_pack_must_start_after.

ola.4 phase-2 deletion: every helper this file tests
(`derive_pack_must_start_after`, `set_pack_needed_by_from_mfma_operands`,
`set_lr_needed_by_from_mfma_operands`, `_hook_up_packs_f32_mfma`,
`_build_reg_to_lr_map`, `_get_lrs_for_pack`, `add_pack_constraints`) was
deleted as part of removing the structural Pack rule. The invariants
those helpers encoded — Pack -> Pack and LR -> Pack RAW/WAR ordering,
Pack -> MFMA needed_by chains — are now enforced graph-side by
`_GenericALURule` + `validate_edge_wait_coverage` + `compare_graphs`,
covered by `test_validate_pack_graph.py` and
`test_dataflow_graph_register_gaps.py`.

This module is `pytest.skip`-ed at collection time because its imports
no longer resolve. The file is preserved so the migration provenance
(test names + per-class invariant docstrings) stays browsable for
post-mortem review; it should be deleted by a follow-up cleanup bead.
"""

import pytest

pytest.skip(
    "test_register_tracing.py: helpers it tests were deleted by ola.4 "
    "phase-2; graph-side coverage in test_validate_pack_graph.py and "
    "test_dataflow_graph_register_gaps.py supersedes this module.",
    allow_module_level=True,
)

from rocisa.instruction import (
    VPermB32, VSwapB32, DSLoadB128, MFMAInstruction, VCvtPkF32toBF16,
    SWaitCnt, SBarrier,
)
from rocisa.container import vgpr, sgpr
from rocisa.enum import InstType

from Tensile.Components.CMSValidator import (
    Pack, CVTPack, MiddlePack, MFMAPack, SwapPack,
    LocalRead, MFMA,
    resolve_pack_type, PACK_TYPE_MAP,
    _compute_swap_register_pairs,
    VGPRS_PER_CONVERSION_GROUP,
    get_dst_range, get_src_ranges, get_reg_range,
    SchedulePosition,
    create_unified_timeline,
    validate_timeline, ValidationContext, isValid,
)
from Tensile.Components.CustomSchedule import ScheduleInfo, hasCustomSchedule
from cms_test_utils import (
    make_mock_mfma_code, make_mock_id_map,
    _make_mock_lr, _make_mock_packs_bf16, _make_mock_packs_tf32_4x4,
    _make_mock_swap_packs,
    generate_real_idmap,
    LRA_BASE, LRB_BASE, PACK_A_DST_BASE, PACK_B_DST_BASE,
    CVT0_BASE, TF32_INTERMEDIATE_BASE, MFMA_PACK_A_BASE,
)


# --- Helper to create ValidatorInstruction objects with mock rocisa_inst ---

_sub_counter = 0

def _make_pos(vmfma: int, loop: int = 1) -> SchedulePosition:
    """Create a SchedulePosition for testing."""
    global _sub_counter
    _sub_counter += 1
    return SchedulePosition(loop_index=loop, vmfma_index=vmfma, sub_index=_sub_counter)


def _make_lr(name: str, vmfma: int, issue_index: int, rocisa_inst) -> LocalRead:
    """Create a LocalRead with the given position and rocisa instruction."""
    lr = LocalRead(name=name, issued_at=_make_pos(vmfma), issue_index=issue_index)
    lr.rocisa_inst = rocisa_inst
    return lr


def _make_pack(name: str, vmfma: int, issue_index: int, rocisa_inst,
               cls=Pack, group_index: int = 0) -> Pack:
    """Create a Pack (or subclass) with the given position and rocisa instruction."""
    p = cls(name=name, issued_at=_make_pos(vmfma), issue_index=issue_index,
            group_index=group_index)
    p.rocisa_inst = rocisa_inst
    return p


# =============================================================================
# Test: BF16 register deps match positional
# =============================================================================

class TestBF16RegisterDeps:
    """Verify derive_pack_must_start_after matches _hook_up_packs_bf16 for BF16."""

    def test_single_pack_depends_on_two_lrs(self):
        """BF16 Pack[0] reads from LR[0] and LR[1] — register trace should find both,
        reduced to the latest."""
        mock_lrs = _make_mock_lr(4, base_reg=LRA_BASE)
        mock_packs = _make_mock_packs_bf16(1, pack_dst_base=PACK_A_DST_BASE,
                                            lr_base=LRA_BASE, num_lrs=4)

        # LR[0] at vmfma 0, LR[1] at vmfma 2 (later = higher constraint)
        lrs = [
            _make_lr("LRA0", 0, 0, mock_lrs[0]),
            _make_lr("LRA0", 2, 1, mock_lrs[1]),
            _make_lr("LRA0", 4, 2, mock_lrs[2]),
            _make_lr("LRA0", 6, 3, mock_lrs[3]),
        ]
        packs = [
            _make_pack("PackA0", 3, 0, mock_packs[0]),
        ]

        result = derive_pack_must_start_after(packs, lrs)
        assert 0 in result
        assert len(result[0]) == 1
        # Pack[0] element_idx=0, reads from LR[0] (v[1000]) and LR[1] (v[1004])
        # LR[1] is at vmfma 2 (later), so max dep should be LR[1]
        assert result[0][0] is lrs[1], \
            f"Expected dep on LR[1] (vmfma=2), got dep on {result[0][0].name}@{result[0][0].issued_at.vmfma_index}"

    def test_multiple_packs_map_to_correct_lr_pairs(self):
        """Each BF16 pack maps to a different LR pair via element_idx."""
        mock_lrs = _make_mock_lr(8, base_reg=LRA_BASE)
        mock_packs = _make_mock_packs_bf16(4, pack_dst_base=PACK_A_DST_BASE,
                                            lr_base=LRA_BASE, num_lrs=8)

        lrs = [_make_lr("LRA0", i, i, mock_lrs[i]) for i in range(8)]
        packs = [_make_pack("PackA0", 10 + i, i, mock_packs[i]) for i in range(4)]

        result = derive_pack_must_start_after(packs, lrs)

        # Pack[0]: element_idx=0, LR[0]+LR[1] → max = LR[1] (vmfma=1)
        assert result[0][0] is lrs[1]
        # Pack[1]: element_idx=1, LR[2]+LR[3] → max = LR[3] (vmfma=3)
        assert result[1][0] is lrs[3]
        # Pack[2]: element_idx=2, LR[4]+LR[5] → max = LR[5] (vmfma=5)
        assert result[2][0] is lrs[5]
        # Pack[3]: element_idx=3, LR[6]+LR[7] → max = LR[7] (vmfma=7)
        assert result[3][0] is lrs[7]

    def test_wrapping_element_idx(self):
        """Packs beyond num_element_pairs wrap around to earlier LR pairs."""
        mock_lrs = _make_mock_lr(4, base_reg=LRA_BASE)
        mock_packs = _make_mock_packs_bf16(4, pack_dst_base=PACK_A_DST_BASE,
                                            lr_base=LRA_BASE, num_lrs=4)

        # 4 LRs → 2 element pairs. Packs 0,1 map to pairs 0,1. Packs 2,3 wrap to 0,1.
        lrs = [_make_lr("LRA0", i * 2, i, mock_lrs[i]) for i in range(4)]
        packs = [_make_pack("PackA0", 10 + i, i, mock_packs[i]) for i in range(4)]

        result = derive_pack_must_start_after(packs, lrs)

        # Pack[0] and Pack[2] map to same LR pair (element_idx 0)
        assert result[0][0].issued_at.vmfma_index == result[2][0].issued_at.vmfma_index
        # Pack[1] and Pack[3] map to same LR pair (element_idx 1)
        assert result[1][0].issued_at.vmfma_index == result[3][0].issued_at.vmfma_index


# =============================================================================
# Test: TF32 4x4 register deps
# =============================================================================

class TestTF32_4x4RegisterDeps:
    """Verify derive_pack_must_start_after for TF32 4x4 MFMA pack groups."""

    def _make_group(self, lr_base=LRA_BASE, pack_dst_base=PACK_A_DST_BASE):
        """Create one group of 10 TF32 4x4 packs with matching LRs."""
        mock_lrs_raw = _make_mock_lr(2, base_reg=lr_base)
        mock_packs_raw = _make_mock_packs_tf32_4x4(10, pack_dst_base=pack_dst_base,
                                                     lr_base=lr_base)
        # 2 LRs providing 8 VGPRs for the group
        lrs = [_make_lr("LRA0", i, i, mock_lrs_raw[i]) for i in range(2)]

        # 10 packs: CVT0[0..3], MFMAPack[4..5], CVT1[6..9]
        packs = []
        for i in range(10):
            cls, _ = resolve_pack_type(mock_packs_raw[i])
            packs.append(_make_pack("PackA0", 10 + i, i, mock_packs_raw[i],
                                    cls=cls, group_index=0))
        return lrs, packs

    def test_cvt0_depends_on_lr(self):
        """CVT0 packs read from LR destination registers."""
        lrs, packs = self._make_group()
        result = derive_pack_must_start_after(packs, lrs)

        # CVT0[0..3] should each depend on an LR
        for i in range(4):
            assert len(result[i]) == 1, f"CVT0[{i}] should have exactly 1 dep"
            dep = result[i][0]
            assert isinstance(dep, LocalRead), f"CVT0[{i}] should depend on an LR, got {type(dep).__name__}"

    def test_mfmapack_depends_on_cvt0(self):
        """MFMAPack reads from CVT0 output registers (.b operand)."""
        lrs, packs = self._make_group()
        result = derive_pack_must_start_after(packs, lrs)

        # MFMAPack[0] (idx 4): reads CVT0[0..1] output via .b
        assert len(result[4]) == 1
        dep4 = result[4][0]
        assert dep4.issue_index in (0, 1), f"MFMAPack[0] should dep on CVT0[0] or CVT0[1], got idx={dep4.issue_index}"

        # MFMAPack[1] (idx 5): reads CVT0[2..3] output via .b
        assert len(result[5]) == 1
        dep5 = result[5][0]
        assert dep5.issue_index in (2, 3), f"MFMAPack[1] should dep on CVT0[2] or CVT0[3], got idx={dep5.issue_index}"

    def test_cvt1_war_chain(self):
        """CVT1 packs have WAR deps from the reverse-write pattern.

        CVT1[0] (idx 6): RAW from MFMAPack[1] (idx 5)
        CVT1[1] (idx 7): RAW from MFMAPack[1] + WAR from CVT1[0] → max = CVT1[0]
        CVT1[2] (idx 8): RAW from MFMAPack[0] + WAR from CVT1[1] → max = CVT1[1]
        CVT1[3] (idx 9): RAW from MFMAPack[0] + WAR from CVT1[1] → max = CVT1[1]
        """
        lrs, packs = self._make_group()
        result = derive_pack_must_start_after(packs, lrs)

        # CVT1[0] (idx 6): depends on MFMAPack[1] (idx 5)
        assert result[6][0].issue_index == 5, \
            f"CVT1[0] should dep on MFMAPack[1] (idx 5), got idx={result[6][0].issue_index}"

        # CVT1[1] (idx 7): WAR on CVT1[0] (idx 6) — CVT1[1] writes a reg that CVT1[0] read
        assert result[7][0].issue_index == 6, \
            f"CVT1[1] should dep on CVT1[0] (idx 6) via WAR, got idx={result[7][0].issue_index}"

        # CVT1[2] (idx 8): WAR on CVT1[1] (idx 7)
        assert result[8][0].issue_index == 7, \
            f"CVT1[2] should dep on CVT1[1] (idx 7) via WAR, got idx={result[8][0].issue_index}"

        # CVT1[3] (idx 9): WAR on CVT1[1] (idx 7), NOT CVT1[2] (idx 8)
        assert result[9][0].issue_index == 7, \
            f"CVT1[3] should dep on CVT1[1] (idx 7) via WAR, got idx={result[9][0].issue_index}"

    def test_no_war_for_middlepack(self):
        """MiddlePack should NOT get WAR deps — pair ordering is handled separately."""
        from rocisa.instruction import VSubF32
        tmp_reg = TF32_INTERMEDIATE_BASE

        # Create a minimal scenario: two MiddlePacks sharing a tmp register
        mock_lr = _make_mock_lr(1, base_reg=LRA_BASE)
        lr = _make_lr("LRA0", 0, 0, mock_lr[0])

        # MiddlePack[0] writes to tmp, reads from CVT0 output
        mp0_inst = VSubF32(dst=vgpr(tmp_reg, 1), src0=vgpr(CVT0_BASE, 1),
                           src1=vgpr(LRA_BASE, 1))
        mp0 = _make_pack("PackA0", 5, 0, mp0_inst, cls=MiddlePack)

        # MiddlePack[1] reads from tmp, writes to middle output
        mp1_inst = VSubF32(dst=vgpr(TF32_INTERMEDIATE_BASE + 100, 1),
                           src0=vgpr(LRA_BASE + 1, 1), src1=vgpr(tmp_reg, 1))
        mp1 = _make_pack("PackA0", 6, 1, mp1_inst, cls=MiddlePack)

        # MiddlePack[2] writes to SAME tmp (WAR with mp1 reading tmp)
        mp2_inst = VSubF32(dst=vgpr(tmp_reg, 1), src0=vgpr(CVT0_BASE, 1),
                           src1=vgpr(LRA_BASE + 2, 1))
        mp2 = _make_pack("PackA0", 7, 2, mp2_inst, cls=MiddlePack)

        result = derive_pack_must_start_after([mp0, mp1, mp2], [lr])

        # mp2 writes to tmp which mp1 read — but WAR is skipped for MiddlePack
        # So mp2 should only have RAW dep (on whoever wrote its src registers)
        if result[2]:
            dep = result[2][0]
            # mp2 should NOT depend on mp1 via WAR
            assert dep is not mp1, \
                "MiddlePack should not get WAR deps — pair ordering handled by pair_consumer"


# =============================================================================
# Test: Direct register chain verification
# =============================================================================

class TestRegisterChainIntegrity:
    """Verify that mock instruction register ranges form proper chains."""

    def test_bf16_lr_dst_overlaps_pack_src(self):
        """BF16 Pack src registers must fall within LR dst ranges."""
        mock_lrs = _make_mock_lr(4, base_reg=LRA_BASE)
        mock_packs = _make_mock_packs_bf16(2, pack_dst_base=PACK_A_DST_BASE,
                                            lr_base=LRA_BASE, num_lrs=4)

        lr0_dst = get_dst_range(mock_lrs[0])  # ('v', 1000, 1004)
        lr1_dst = get_dst_range(mock_lrs[1])  # ('v', 1004, 1008)
        pack0_src = get_src_ranges(mock_packs[0])  # should include regs in lr0 and lr1

        # Pack[0] element_idx=0, reads from LR[0] and LR[1]
        vgpr_srcs = [(b, s, e) for b, s, e in pack0_src if b == 'v']
        assert len(vgpr_srcs) == 2, f"Pack[0] should have 2 VGPR sources, got {len(vgpr_srcs)}"

        # One src should be in LR[0]'s range, the other in LR[1]'s range
        src_in_lr0 = any(lr0_dst[1] <= s < lr0_dst[2] for _, s, _ in vgpr_srcs)
        src_in_lr1 = any(lr1_dst[1] <= s < lr1_dst[2] for _, s, _ in vgpr_srcs)
        assert src_in_lr0, f"Pack[0] should have a src in LR[0] range {lr0_dst}"
        assert src_in_lr1, f"Pack[0] should have a src in LR[1] range {lr1_dst}"

    def test_tf32_4x4_cvt0_reads_lr_space(self):
        """TF32 4x4 CVT0 packs read from LR destination registers."""
        mock_lrs = _make_mock_lr(2, base_reg=LRA_BASE)
        mock_packs = _make_mock_packs_tf32_4x4(10, pack_dst_base=PACK_A_DST_BASE,
                                                 lr_base=LRA_BASE)

        lr0_dst = get_dst_range(mock_lrs[0])  # ('v', 1000, 1004)
        cvt0_src = get_src_ranges(mock_packs[0])
        vgpr_srcs = [(b, s, e) for b, s, e in cvt0_src if b == 'v']

        # CVT0[0] reads 2 consecutive VGPRs from LR[0]'s range
        for _, s, _ in vgpr_srcs:
            assert lr0_dst[1] <= s < lr0_dst[2], \
                f"CVT0[0] src v[{s}] should be within LR[0] range {lr0_dst}"

    def test_tf32_4x4_cvt0_writes_to_cvt0_space(self):
        """TF32 4x4 CVT0 destination is in the CVT0 intermediate space."""
        mock_packs = _make_mock_packs_tf32_4x4(10, pack_dst_base=PACK_A_DST_BASE,
                                                 lr_base=LRA_BASE)
        for i in range(4):
            dst = get_dst_range(mock_packs[i])
            assert dst[0] == 'v' and CVT0_BASE <= dst[1] < CVT0_BASE + 100, \
                f"CVT0[{i}] dst {dst} should be in CVT0 space v[{CVT0_BASE}+]"

    def test_tf32_4x4_mfmapack_reads_cvt0_output(self):
        """MFMAPack .b reads from CVT0 output space."""
        mock_packs = _make_mock_packs_tf32_4x4(10, pack_dst_base=PACK_A_DST_BASE,
                                                 lr_base=LRA_BASE)
        # MFMAPack[0] is at index 4
        mfma_pack = mock_packs[4]
        assert hasattr(mfma_pack, 'b'), "MFMAPack should have .b attribute"
        b_range = get_reg_range(mfma_pack.b)
        assert b_range[0] == 'v' and CVT0_BASE <= b_range[1] < CVT0_BASE + 100, \
            f"MFMAPack[0] .b {b_range} should read from CVT0 space v[{CVT0_BASE}+]"

    def test_tf32_4x4_cvt1_reverse_write_pattern(self):
        """CVT1 packs write in reverse order within pack output space."""
        mock_packs = _make_mock_packs_tf32_4x4(10, pack_dst_base=PACK_A_DST_BASE,
                                                 lr_base=LRA_BASE)
        cvt1_dsts = []
        for i in range(6, 10):
            dst = get_dst_range(mock_packs[i])
            cvt1_dsts.append(dst[1])

        # Should be in descending order: 5003, 5002, 5001, 5000
        assert cvt1_dsts == sorted(cvt1_dsts, reverse=True), \
            f"CVT1 dsts should be in reverse order, got {cvt1_dsts}"

    def test_tf32_4x4_cvt1_war_register_overlap(self):
        """CVT1[1] writes a register that CVT1[0] read — verifying WAR hazard exists."""
        mock_packs = _make_mock_packs_tf32_4x4(10, pack_dst_base=PACK_A_DST_BASE,
                                                 lr_base=LRA_BASE)
        # CVT1[0] (idx 6) reads some registers
        cvt1_0_srcs = get_src_ranges(mock_packs[6])
        cvt1_0_src_indices = set()
        for b, s, e in cvt1_0_srcs:
            if b == 'v':
                for r in range(s, e):
                    cvt1_0_src_indices.add(r)

        # CVT1[1] (idx 7) dst register
        cvt1_1_dst = get_dst_range(mock_packs[7])

        # CVT1[1]'s dst should overlap with one of CVT1[0]'s src registers
        assert cvt1_1_dst[1] in cvt1_0_src_indices, \
            f"CVT1[1] dst v[{cvt1_1_dst[1]}] should be in CVT1[0]'s src set {cvt1_0_src_indices} for WAR"

    def test_mfma_a_reads_pack_output_space(self):
        """MFMA .a should reference the PackA output space."""
        mfmas = make_mock_mfma_code(4)
        a_range = get_reg_range(mfmas[0].a)
        assert a_range[0] == 'v' and PACK_A_DST_BASE <= a_range[1] < PACK_A_DST_BASE + 100, \
            f"MFMA[0] .a {a_range} should read from PackA output space v[{PACK_A_DST_BASE}+]"

    def test_mfma_b_reads_pack_output_space(self):
        """MFMA .b should reference the PackB output space."""
        mfmas = make_mock_mfma_code(4)
        b_range = get_reg_range(mfmas[0].b)
        assert b_range[0] == 'v' and PACK_B_DST_BASE <= b_range[1] < PACK_B_DST_BASE + 100, \
            f"MFMA[0] .b {b_range} should read from PackB output space v[{PACK_B_DST_BASE}+]"

    def test_mfmapack_dst_overlaps_cvt1_src(self):
        """get_dst_range(MFMAPack) must return the .acc range, and CVT1 must read it.

        Load-bearing precondition for set_pack_needed_by_from_mfma_operands:
        MFMAPack→CVT1 chain depends on MFMAPack's dst (the accumulator)
        being readable as a register range that overlaps a CVT1 src range.
        """
        mock_packs = _make_mock_packs_tf32_4x4(10, pack_dst_base=PACK_A_DST_BASE,
                                                 lr_base=LRA_BASE)
        # MFMAPack[0] (idx 4): .acc writes to LR T-space; CVT1[2] (idx 8) reads it.
        mp0_dst = get_dst_range(mock_packs[4])
        assert mp0_dst is not None, "MFMAPack[0] dst must be non-None"
        cvt1_2_srcs = [s for b, s, e in get_src_ranges(mock_packs[8]) if b == mp0_dst[0]
                       for s in range(s, e)]
        mp0_offsets = set(range(mp0_dst[1], mp0_dst[2]))
        assert any(s in mp0_offsets for s in cvt1_2_srcs), \
            f"CVT1[2] srcs must overlap MFMAPack[0] dst {mp0_dst}, got srcs at {cvt1_2_srcs}"

        # MFMAPack[1] (idx 5): .acc writes to pack output space; CVT1[0] (idx 6) reads it.
        mp1_dst = get_dst_range(mock_packs[5])
        assert mp1_dst is not None, "MFMAPack[1] dst must be non-None"
        cvt1_0_srcs = [s for b, s, e in get_src_ranges(mock_packs[6]) if b == mp1_dst[0]
                       for s in range(s, e)]
        mp1_offsets = set(range(mp1_dst[1], mp1_dst[2]))
        assert any(s in mp1_offsets for s in cvt1_0_srcs), \
            f"CVT1[0] srcs must overlap MFMAPack[1] dst {mp1_dst}, got srcs at {cvt1_0_srcs}"


# =============================================================================
# Test: TF32 4x4 with VW>1 swap packs — positional vs register-based
# =============================================================================

class TestTF32_4x4_SwapPackDeps:
    """Verify derive_pack_must_start_after matches _hook_up_packs_f32_mfma
    for TF32 4x4 with VW>1 swap packs.

    With VW=2, there are 4 swap packs before 10 regular packs per group.
    Each swap transposes a register pair in the LR space. CVT0 packs then
    depend on swaps (for transposed regs) or LRs (for non-transposed regs).

    The test creates instructions using the real interleaving pattern from
    _compute_swap_register_pairs, then verifies both code paths agree.
    """

    def _build_vw2_instructions(self, lr_base=LRA_BASE, pack_dst_base=PACK_A_DST_BASE):
        """Build a complete set of VW=2 swap + 10 regular packs with matching LRs.

        Uses the real interleaving pattern from _compute_swap_register_pairs.
        Returns (lrs, all_packs) where all_packs includes swap packs + regular packs.
        """
        vw = 2
        n_lrs = 8  # DS_READ_CONV_TABLE for VW=2 has 8 entries
        total_regs = VGPRS_PER_CONVERSION_GROUP * vw  # 16
        n_swaps = 4 * (vw - 1)  # 4 swaps for VW=2

        # Get the real swap register pairs
        swap_pairs = _compute_swap_register_pairs(vw, total_regs)
        assert len(swap_pairs) == n_swaps

        # Create LR mock instructions: 8 LRs, each loading 4 VGPRs
        mock_lrs_raw = _make_mock_lr(n_lrs, base_reg=lr_base)
        lrs = [_make_lr("LRA0", i, i, mock_lrs_raw[i]) for i in range(n_lrs)]

        # Create swap mock instructions using real register pairs
        swap_raw = _make_mock_swap_packs(n_swaps, lr_base=lr_base, vw=vw)
        swap_packs = [_make_pack("PackA0", 2, i, swap_raw[i], cls=SwapPack)
                      for i in range(n_swaps)]

        # Create regular pack mock instructions (10 per group)
        regular_raw = _make_mock_packs_tf32_4x4(10, pack_dst_base=pack_dst_base,
                                                  lr_base=lr_base)
        regular_packs = []
        for i in range(10):
            cls, _ = resolve_pack_type(regular_raw[i])
            regular_packs.append(_make_pack("PackA0", 3 + i, n_swaps + i,
                                            regular_raw[i], cls=cls, group_index=0))

        all_packs = swap_packs + regular_packs
        return lrs, all_packs, vw

    def test_swap_packs_read_from_lr_space(self):
        """Swap pack src and dst registers should be in the LR register space."""
        vw = 2
        swap_raw = _make_mock_swap_packs(4, lr_base=LRA_BASE, vw=vw)
        for i, sp in enumerate(swap_raw):
            dst = get_dst_range(sp)
            src = get_src_ranges(sp)
            assert dst is not None, f"SwapPack[{i}] should have a dst"
            assert dst[0] == 'v', f"SwapPack[{i}] dst should be a VGPR"
            assert LRA_BASE <= dst[1] < LRA_BASE + 100, \
                f"SwapPack[{i}] dst v[{dst[1]}] should be in LR space v[{LRA_BASE}+]"
            vgpr_srcs = [r for r in src if r[0] == 'v']
            assert len(vgpr_srcs) >= 1, f"SwapPack[{i}] should have at least 1 VGPR src"
            for _, s, _ in vgpr_srcs:
                assert LRA_BASE <= s < LRA_BASE + 100, \
                    f"SwapPack[{i}] src v[{s}] should be in LR space v[{LRA_BASE}+]"

    def test_swap_deps_on_lrs_via_register_tracing(self):
        """Each swap pack should depend on LRs via register tracing."""
        lrs, all_packs, vw = self._build_vw2_instructions()
        result = derive_pack_must_start_after(all_packs, lrs)

        # First 4 packs are swaps — each should depend on an LR
        for i in range(4):
            assert i in result, f"SwapPack[{i}] should have a result entry"
            assert len(result[i]) == 1, f"SwapPack[{i}] should have exactly 1 dep"
            dep = result[i][0]
            assert isinstance(dep, LocalRead), \
                f"SwapPack[{i}] should depend on an LR, got {type(dep).__name__}"

    def test_cvt0_deps_on_swap_or_lr_via_register_tracing(self):
        """CVT0 packs after swaps should depend on SwapPack (for swapped regs) or LR."""
        lrs, all_packs, vw = self._build_vw2_instructions()
        result = derive_pack_must_start_after(all_packs, lrs)

        # CVT0 packs are at indices 4..7 (after 4 swap packs)
        for i in range(4, 8):
            assert i in result, f"CVT0[{i-4}] should have a result entry"
            assert len(result[i]) == 1, f"CVT0[{i-4}] should have exactly 1 dep"
            dep = result[i][0]
            # Should depend on either a SwapPack or an LR
            assert isinstance(dep, (SwapPack, LocalRead)), \
                f"CVT0[{i-4}] should depend on SwapPack or LR, got {type(dep).__name__}"

    def test_positional_and_register_agree_for_swap_packs(self):
        """The positional code and register-based code should produce the same
        effective max dependency for every pack in a VW=2 TF32 4x4 group.

        This is the key equivalence test. It calls _hook_up_packs_f32_mfma
        (positional) and derive_pack_must_start_after (register-based) on the
        same pack set and asserts the max dep (by done_idx) matches for each pack.
        """
        lrs, all_packs, vw = self._build_vw2_instructions()

        # Run positional code
        _hook_up_packs_f32_mfma(all_packs, lrs, vw)

        # Run register-based code
        reg_deps = derive_pack_must_start_after(all_packs, lrs)

        # Compare for each pack
        for pack in all_packs:
            if pack.issue_index not in reg_deps:
                continue
            reg_list = reg_deps[pack.issue_index]
            if not reg_list:
                continue
            pos_latest = max(pack.must_start_after, key=lambda d: d.done_idx()) \
                if pack.must_start_after else None
            reg_latest = reg_list[0]
            if pos_latest is None:
                continue
            assert pos_latest.done_idx() == reg_latest.done_idx(), \
                f"Pack[{pack.issue_index}] ({type(pack).__name__}): " \
                f"positional dep={pos_latest.name}@{pos_latest.done_idx()}, " \
                f"register dep={reg_latest.name}@{reg_latest.done_idx()}"


# =============================================================================
# Test: Real idMap from kernel writer — validates full pipeline
# =============================================================================

class TestRealIdMapValidation:
    """Verify that using real kernel-writer-produced idMaps eliminates all
    dual-path mismatches between positional and register-based must_start_after.

    These tests use the isa_infrastructure fixture (session-scoped, ~3.8s init)
    and generate_real_idmap (~0.14s per kernel) to produce real rocisa instruction
    objects with correct register assignments.
    """

    def test_tf32_4x4_tn_real_idmap_validates(self, isa_infrastructure):
        """TF32 4x4 TN schedule validates with real idMap (no mismatch warnings)."""
        isa, isaInfoMap, asm = isa_infrastructure
        config = {
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
        }
        id_map, mfma_code, solution = generate_real_idmap(config, asm, isaInfoMap)

        # Get the ScheduleInfo for this kernel
        has_schedule, schedule_info = hasCustomSchedule(solution)
        assert has_schedule, "Kernel should have a CMS schedule"

        # Validate with real idMap
        ctx = ValidationContext(kernel=solution, id_map=id_map, mfma_code=mfma_code)
        valid, message = isValid(schedule_info, ctx)
        assert valid, f"Schedule should pass validation: {message}"

    def test_tf32_4x4_tn_real_idmap_has_packs(self, isa_infrastructure):
        """Real idMap should contain pack instructions with named registers."""
        isa, isaInfoMap, asm = isa_infrastructure
        config = {
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
        }
        id_map, mfma_code, solution = generate_real_idmap(config, asm, isaInfoMap)

        assert 'PackA0' in id_map, "idMap should contain PackA0"
        packs = id_map['PackA0']
        assert len(packs) > 0, "PackA0 should have instructions"

        # Verify instructions have named registers (not numeric mocks)
        dst = get_dst_range(packs[0])
        assert dst is not None, "Pack should have a dst range"
        assert dst[0] != 'v' or dst[1] > 100, \
            f"Real pack should use named registers, got numeric {dst}"


# =============================================================================
# Test: set_pack_needed_by_from_mfma_operands — chain identity & filtering
# =============================================================================

def _make_mfma(name: str, vmfma: int, rocisa_inst, loop: int = 1) -> MFMA:
    m = MFMA(name=name, issued_at=_make_pos(vmfma, loop=loop))
    m.rocisa_inst = rocisa_inst
    return m


class TestSetPackNeededByFromMFMAOperands:
    """Direct tests for set_pack_needed_by_from_mfma_operands.

    Verifies the register-traced needed_by reproduces the chain semantics
    of _set_pack_needed_by: CVT0 → MFMAPack → CVT1 → real_MFMA, with the
    earliest-by-issued_at consumer winning.
    """

    def _make_tf32_4x4_group(self, vmfma_offsets, lr_base=LRA_BASE,
                             pack_dst_base=PACK_A_DST_BASE, loop: int = 1):
        """Build one 10-pack TF32 4x4 group with controllable per-pack vmfma.

        vmfma_offsets is a list of 10 vmfma indices for CVT0[0..3] +
        MFMAPack[0,1] + CVT1[0..3].
        """
        assert len(vmfma_offsets) == 10
        mock = _make_mock_packs_tf32_4x4(10, pack_dst_base=pack_dst_base, lr_base=lr_base)
        packs = []
        for i in range(10):
            cls, _ = resolve_pack_type(mock[i])
            p = _make_pack("PackA0", vmfma_offsets[i], i, mock[i], cls=cls, group_index=0)
            # Override loop on the SchedulePosition (helper defaults to loop=1)
            p.issued_at = SchedulePosition(loop_index=loop,
                                           vmfma_index=p.issued_at.vmfma_index,
                                           sub_index=p.issued_at.sub_index)
            packs.append(p)
        return packs

    def test_bf16_pack_needed_by_is_consuming_real_mfma(self):
        """BF16 PackA0 dst is read by a real MFMA — needed_by should point to it."""
        mock_lrs = _make_mock_lr(4, base_reg=LRA_BASE)
        mock_packs = _make_mock_packs_bf16(2, pack_dst_base=PACK_A_DST_BASE,
                                            lr_base=LRA_BASE, num_lrs=4)
        packs = [_make_pack("PackA0", 5 + i, i, mock_packs[i]) for i in range(2)]
        # Build a real MFMA that reads PackA0 dst (mfma.a is in PACK_A_DST_BASE space)
        mfmas = make_mock_mfma_code(2)
        real_mfmas = [_make_mfma("MFMA", 10 + i, mfmas[i]) for i in range(2)]

        result = set_pack_needed_by_from_mfma_operands(packs, real_mfmas)

        assert "PackA0" in result
        # Both packs should find a real MFMA as their consumer
        for i in range(2):
            assert i in result["PackA0"], f"PackA0[{i}] should have a needed_by"
            consumer = result["PackA0"][i]
            assert isinstance(consumer, MFMA), f"PackA0[{i}].needed_by must be a real MFMA"
            assert consumer in real_mfmas, "Consumer must be one of the real MFMAs we created"

    def test_tf32_4x4_intra_chain_identity(self):
        """For a TF32 4x4 group, verify each pack's needed_by is the SPECIFIC
        intra-chain consumer that _set_pack_needed_by would pick:
          cvt0[0,1].needed_by = mfma_packs[0]
          cvt0[2,3].needed_by = mfma_packs[1]
          mfma_packs[0].needed_by = cvt1[2]
          mfma_packs[1].needed_by = cvt1[0]
          cvt1[i].needed_by = real_MFMA (set far enough in the future)
        """
        # Vmfma layout that respects the natural ordering and gives
        # CVT0 < MFMAPack < CVT1 < external_MFMA
        packs = self._make_tf32_4x4_group(
            [10, 10, 11, 11, 15, 15, 20, 20, 21, 21]
        )
        # Real MFMA that reads the pack output (PackA dst space) — fires after CVT1
        mfmas = make_mock_mfma_code(1)
        real_mfma = _make_mfma("MFMA", 30, mfmas[0])

        result = set_pack_needed_by_from_mfma_operands(packs, [real_mfma])
        per_name = result["PackA0"]

        # CVT0 chain identity: cvt0[0,1] feed mfma_packs[0]; cvt0[2,3] feed mfma_packs[1]
        assert per_name[0] is packs[4], f"CVT0[0].needed_by must be mfma_packs[0], got {per_name[0]}"
        assert per_name[1] is packs[4], f"CVT0[1].needed_by must be mfma_packs[0], got {per_name[1]}"
        assert per_name[2] is packs[5], f"CVT0[2].needed_by must be mfma_packs[1], got {per_name[2]}"
        assert per_name[3] is packs[5], f"CVT0[3].needed_by must be mfma_packs[1], got {per_name[3]}"

        # MFMAPack chain identity: mfma_packs[0] feeds cvt1[2] (idx 8); mfma_packs[1] feeds cvt1[0] (idx 6)
        assert per_name[4] is packs[8], f"mfma_packs[0].needed_by must be cvt1[2] (idx 8), got {per_name[4]}"
        assert per_name[5] is packs[6], f"mfma_packs[1].needed_by must be cvt1[0] (idx 6), got {per_name[5]}"

        # CVT1 packs feed the external real MFMA
        for i in range(6, 10):
            assert per_name[i] is real_mfma, f"cvt1[{i-6}].needed_by must be the external MFMA, got {per_name[i]}"

    def test_earliest_of_multiple_candidates_wins(self):
        """When multiple instructions read the same dst register at different
        vmfma slots (and all fire after the pack), the earliest-by-issued_at
        candidate wins. This is the rule that reproduces _set_pack_needed_by's
        external-vs-intra-chain override at lines 1668-1670."""
        packs = self._make_tf32_4x4_group(
            [10, 10, 11, 11, 15, 15, 20, 20, 21, 21]
        )
        # Two real MFMAs, both reading PackA output (covering CVT1.dst).
        # MFMA at vmfma=22 fires before MFMA at vmfma=30. CVT1 packs (vmfma
        # 20-21) must pick the earlier MFMA (vmfma=22), not the later (30).
        mfmas = make_mock_mfma_code(1)
        early = _make_mfma("MFMA", 22, mfmas[0])
        late = _make_mfma("MFMA", 30, mfmas[0])

        result = set_pack_needed_by_from_mfma_operands(packs, [early, late])
        per_name = result["PackA0"]

        for i in range(6, 10):
            consumer = per_name[i]
            assert consumer is early, \
                f"cvt1[{i-6}] should pick the earliest MFMA (vmfma=22), got vmfma={consumer.issued_at.vmfma_index}"

    def test_register_reuse_across_iterations_filtered_out(self):
        """A pack must NOT pick a same-loop reader issued BEFORE itself.

        Construct a TF32 4x4 group where CVT0 reads from a register that an
        earlier-iteration CVT1 writes. In the linear instruction list of one
        loop body, that earlier-iteration's CVT1 is the SAME object as the
        current iteration's CVT1 (single occurrence per loop body). It must
        not be picked as the consumer of CVT0 from a later schedule slot.
        """
        # Order: CVT0 packs at vmfma=10-11, CVT1 packs at vmfma=5-6
        # (CVT1 is issued BEFORE CVT0 — a contrived but illustrative case).
        # MFMAPack at vmfma=15.
        packs = self._make_tf32_4x4_group(
            [10, 10, 11, 11, 15, 15, 5, 5, 6, 6]
        )
        mfmas = make_mock_mfma_code(1)
        real_mfma = _make_mfma("MFMA", 30, mfmas[0])

        result = set_pack_needed_by_from_mfma_operands(packs, [real_mfma])
        per_name = result["PackA0"]

        # CVT1[0..3] (vmfma 5-6) come BEFORE CVT0 (vmfma 10-11). For CVT1's
        # dst lookup, the only candidates issued strictly after vmfma 5/6 are
        # the external MFMA (vmfma 30) and MFMAPacks (vmfma 15). MFMAPacks
        # don't read pack output space. So CVT1's needed_by must be the
        # external MFMA, not a sibling CVT1.
        for i in range(6, 10):
            consumer = per_name[i]
            assert consumer is real_mfma, \
                f"cvt1[{i-6}].needed_by must be external MFMA (vmfma=30), " \
                f"got {consumer.name}@vmfma={consumer.issued_at.vmfma_index}"

    def test_loop_scoping_no_cross_loop_consumer(self):
        """Inputs must be loop-scoped; the function trusts the caller. Verify
        that when only same-loop instructions are passed in, all assigned
        needed_by share the pack's loop_index (the function's correctness
        precondition holds when caller scopes the inputs)."""
        loop_idx = 2
        packs = self._make_tf32_4x4_group(
            [10, 10, 11, 11, 15, 15, 20, 20, 21, 21],
            loop=loop_idx,
        )
        mfmas = make_mock_mfma_code(1)
        real_mfma = _make_mfma("MFMA", 30, mfmas[0], loop=loop_idx)

        result = set_pack_needed_by_from_mfma_operands(packs, [real_mfma])

        for issue_index, consumer in result["PackA0"].items():
            assert consumer.issued_at.loop_index == loop_idx, \
                f"pack[{issue_index}].needed_by has loop {consumer.issued_at.loop_index}, " \
                f"expected {loop_idx}"

    def test_swap_packs_excluded(self):
        """SwapPacks must not appear in the result and must not be assigned as consumers."""
        packs = self._make_tf32_4x4_group(
            [10, 10, 11, 11, 15, 15, 20, 20, 21, 21]
        )
        # Add a SwapPack that reads from CVT0 output space (artificially)
        # so we can verify it's filtered as a candidate even if its register
        # access pattern would otherwise match.
        from rocisa.instruction import VSwapB32
        from rocisa.container import vgpr
        swap_inst = VSwapB32(dst=vgpr(CVT0_BASE + 100, 1), src=vgpr(CVT0_BASE, 1))
        swap = _make_pack("PackA0", 12, 100, swap_inst, cls=SwapPack)
        packs.append(swap)

        mfmas = make_mock_mfma_code(1)
        real_mfma = _make_mfma("MFMA", 30, mfmas[0])

        result = set_pack_needed_by_from_mfma_operands(packs, [real_mfma])

        # SwapPack must not be in the result
        assert 100 not in result.get("PackA0", {}), "SwapPack must not get a needed_by entry"
        # CVT0[0]'s needed_by must still be the MFMAPack, not the SwapPack
        # (SwapPack reads CVT0_BASE which is CVT0[0]'s dst — would falsely match
        # if SwapPack weren't filtered as a candidate).
        assert result["PackA0"][0] is packs[4], \
            f"CVT0[0].needed_by must be mfma_packs[0], not the SwapPack. Got {result['PackA0'][0]}"


# =============================================================================
# Test: set_lr_needed_by_from_mfma_operands — chain follow + filtering
# =============================================================================


class TestSetLrNeededByFromMFMAOperands:
    """Direct tests for set_lr_needed_by_from_mfma_operands.

    Verifies the LR's chain-follow terminates at a real MFMA via the
    inline pack-chain computation (no dependence on pack.needed_by).
    Mirrors TestSetPackNeededByFromMFMAOperands six-test pattern.
    """

    def _make_tf32_4x4_group(self, vmfma_offsets, lr_base=LRA_BASE,
                             pack_dst_base=PACK_A_DST_BASE, loop: int = 1):
        """Build one 10-pack TF32 4x4 group with controllable per-pack vmfma."""
        assert len(vmfma_offsets) == 10
        mock = _make_mock_packs_tf32_4x4(10, pack_dst_base=pack_dst_base, lr_base=lr_base)
        packs = []
        for i in range(10):
            cls, _ = resolve_pack_type(mock[i])
            p = _make_pack("PackA0", vmfma_offsets[i], i, mock[i], cls=cls, group_index=0)
            p.issued_at = SchedulePosition(loop_index=loop,
                                           vmfma_index=p.issued_at.vmfma_index,
                                           sub_index=p.issued_at.sub_index)
            packs.append(p)
        return packs

    @staticmethod
    def _set_lr_loop(lr: LocalRead, loop: int) -> None:
        """Override an LR's loop_index (helper defaults to loop=1)."""
        lr.issued_at = SchedulePosition(loop_index=loop,
                                        vmfma_index=lr.issued_at.vmfma_index,
                                        sub_index=lr.issued_at.sub_index)

    def test_bf16_lr_needed_by_is_real_mfma(self):
        """Single-hop LR→Pack→MFMA: chain follow finds the Pack, then
        Pack.needed_by (via inline pack-chain) is the real MFMA."""
        mock_lrs = _make_mock_lr(4, base_reg=LRA_BASE)
        mock_packs = _make_mock_packs_bf16(4, pack_dst_base=PACK_A_DST_BASE,
                                            lr_base=LRA_BASE, num_lrs=4)
        # LRs at vmfma 0,1,2,3 (before packs at 5..)
        lrs = [_make_lr("LRA0", i, i, mock_lrs[i]) for i in range(4)]
        # Packs at vmfma 5..8 (after LRs, before MFMAs)
        packs = [_make_pack("PackA0", 5 + i, i, mock_packs[i]) for i in range(4)]
        # Real MFMA at vmfma=10 reads PackA0 dst (PACK_A_DST_BASE + 0..4).
        mfmas = make_mock_mfma_code(1)
        real_mfma = _make_mfma("MFMA", 10, mfmas[0])

        result = set_lr_needed_by_from_mfma_operands(lrs, packs, [real_mfma])

        assert "LRA0" in result, f"Expected LRA0 in result, got keys {list(result.keys())}"
        for i in range(4):
            assert i in result["LRA0"], f"LRA0[{i}] missing from result"
            consumer = result["LRA0"][i]
            assert consumer is real_mfma, \
                f"LRA0[{i}].needed_by must be the real MFMA, got {consumer}"

    def test_tf32_4x4_lr_chain_terminates_at_real_mfma(self):
        """Four-hop LR→CVT0→MFMAPack→CVT1→real_MFMA: chain follow walks
        through every Pack subclass (CVTPack, MFMAPack, CVTPack) and
        terminates at the real MFMA, NOT at any intermediate pack."""
        # LRs at vmfma 0..1 (before any pack). LR[0] writes regs
        # LRA_BASE+0..4 (consumed by CVT0[0,1]); LR[1] writes
        # LRA_BASE+4..8 (consumed by CVT0[2,3]).
        mock_lrs = _make_mock_lr(2, base_reg=LRA_BASE)
        lrs = [_make_lr("LRA0", i, i, mock_lrs[i]) for i in range(2)]

        packs = self._make_tf32_4x4_group(
            [10, 10, 11, 11, 15, 15, 20, 20, 21, 21]
        )
        mfmas = make_mock_mfma_code(1)
        real_mfma = _make_mfma("MFMA", 30, mfmas[0])

        result = set_lr_needed_by_from_mfma_operands(lrs, packs, [real_mfma])

        assert "LRA0" in result
        for i in range(2):
            assert i in result["LRA0"], f"LRA0[{i}] missing"
            consumer = result["LRA0"][i]
            assert consumer is real_mfma, \
                f"LRA0[{i}].needed_by must be the real MFMA at vmfma=30, " \
                f"got {consumer.name}@vmfma={consumer.issued_at.vmfma_index}. " \
                f"Chain should be LR→CVT0→MFMAPack→CVT1→real_MFMA."

    def test_earliest_lr_consumer_wins(self):
        """When LR.dst is read by multiple candidates (all issued after LR),
        the chain follows the earliest-by-issued_at immediate consumer."""
        # LR[0] at vmfma=0; two MFMAs read it directly at vmfma=5 and vmfma=10.
        # The earliest (vmfma=5) wins.
        mock_lrs = _make_mock_lr(1, base_reg=PACK_A_DST_BASE)  # LR writes pack-dst space
        # Reuse the LR's "dst" as a register MFMAs would read.
        lrs = [_make_lr("LRA0", 0, 0, mock_lrs[0])]
        mfmas = make_mock_mfma_code(1)
        early = _make_mfma("MFMA", 5, mfmas[0])
        late = _make_mfma("MFMA", 10, mfmas[0])

        result = set_lr_needed_by_from_mfma_operands(lrs, [], [early, late])

        consumer = result["LRA0"][0]
        assert consumer is early, \
            f"LRA0[0] should pick earliest MFMA (vmfma=5), got vmfma={consumer.issued_at.vmfma_index}"

    def test_lr_register_reuse_filtered_by_strict_lt(self):
        """An LR cannot pick a candidate issued at/before itself.

        Place the LR AFTER all packs — it represents the next iteration's
        LR overwriting registers in the same loop body. The packs (issued
        before the LR) read prior-iteration data, not this LR's data.
        Result: the LR has no valid consumer in this scope.
        """
        # Packs at vmfma 5..8; LR at vmfma 20 (after all packs).
        mock_packs = _make_mock_packs_bf16(4, pack_dst_base=PACK_A_DST_BASE,
                                            lr_base=LRA_BASE, num_lrs=4)
        packs = [_make_pack("PackA0", 5 + i, i, mock_packs[i]) for i in range(4)]
        mock_lrs = _make_mock_lr(1, base_reg=LRA_BASE)
        lrs = [_make_lr("LRA0", 20, 0, mock_lrs[0])]
        mfmas = make_mock_mfma_code(1)
        # MFMA at vmfma=25 (after the LR, but only reads pack output, not LR.dst).
        real_mfma = _make_mfma("MFMA", 25, mfmas[0])

        result = set_lr_needed_by_from_mfma_operands(lrs, packs, [real_mfma])

        # LR at vmfma=20 has packs at vmfma 5..8 reading its dst (LRA_BASE+0..4),
        # but they're issued BEFORE the LR — strict < filter rejects them.
        # The MFMA at vmfma=25 doesn't read LR.dst directly, so no consumer.
        assert result.get("LRA0", {}).get(0) is None, \
            "LR at vmfma=20 must have no consumer; packs at vmfma 5..8 are " \
            "issued before it and should be filtered by strict <."

    def test_lr_chain_dies_when_swap_pack_only_consumer(self):
        """When LR.dst is consumed only by a SwapPack, the chain dies and
        the LR is omitted (caller-side fallback semantics)."""
        from rocisa.instruction import VSwapB32
        from rocisa.container import vgpr

        # Construct one LR; its only consumer is a SwapPack.
        mock_lrs = _make_mock_lr(1, base_reg=LRA_BASE)
        lrs = [_make_lr("LRA0", 0, 0, mock_lrs[0])]

        # SwapPack reads LRA_BASE..+1, writes elsewhere. SwapPacks are
        # filtered from the candidates set, so LR's chain finds no
        # immediate consumer and is omitted.
        swap_inst = VSwapB32(dst=vgpr(SWAP_BASE := 8000, 1), src=vgpr(LRA_BASE, 1))
        swap = _make_pack("PackA0", 5, 100, swap_inst, cls=SwapPack)

        result = set_lr_needed_by_from_mfma_operands(lrs, [swap], [])

        assert result.get("LRA0", {}).get(0) is None, \
            "LR consumed only by a SwapPack must be omitted from result"

    def test_lr_loop_scoping(self):
        """When caller passes loop-scoped inputs, all assigned consumers
        share the LR's loop_index."""
        loop_idx = 2

        # LR in loop 2.
        mock_lrs = _make_mock_lr(2, base_reg=LRA_BASE)
        lrs = [_make_lr("LRA0", i, i, mock_lrs[i]) for i in range(2)]
        for lr in lrs:
            self._set_lr_loop(lr, loop_idx)

        packs = self._make_tf32_4x4_group(
            [10, 10, 11, 11, 15, 15, 20, 20, 21, 21],
            loop=loop_idx,
        )
        mfmas = make_mock_mfma_code(1)
        real_mfma = _make_mfma("MFMA", 30, mfmas[0], loop=loop_idx)

        result = set_lr_needed_by_from_mfma_operands(lrs, packs, [real_mfma])

        for issue_index, consumer in result.get("LRA0", {}).items():
            assert consumer.issued_at.loop_index == loop_idx, \
                f"LRA0[{issue_index}].needed_by has loop {consumer.issued_at.loop_index}, " \
                f"expected {loop_idx}"


# =============================================================================
# Parity test removed by bead ola.3 phase-2.
# =============================================================================
# The deleted ``TestSetLrNeededByParity`` compared
# ``set_lr_needed_by_from_mfma_operands`` (register-based) against
# ``set_lr_needed_by_for_VMFMA`` (positional baseline). The positional
# baseline was the implementation of the deleted
# ``add_local_read_constraints`` rule; with the rule and helper gone,
# there is no longer a baseline to compare against.
#
# The register-based ``set_lr_needed_by_from_mfma_operands`` continues
# to ship — it is consumed by ``hook_up_packs`` (still alive under
# ola.4 phase-1). Direct coverage of that helper lives in
# ``TestSetLrNeededByFromMFMAOperands`` above.
