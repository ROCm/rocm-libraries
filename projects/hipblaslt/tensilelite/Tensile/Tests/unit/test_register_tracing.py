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

These tests verify that the register-based dependency derivation produces
the same must_start_after constraints as the positional code in
_hook_up_packs_bf16, _hook_up_packs_f32, and _hook_up_packs_f32_mfma.
"""

import pytest
from rocisa.instruction import (
    VPermB32, VSwapB32, DSLoadB128, MFMAInstruction, VCvtPkF32toBF16,
    SWaitCnt, SBarrier,
)
from rocisa.container import vgpr, sgpr
from rocisa.enum import InstType

from Tensile.Components.CMSValidator import (
    Pack, CVTPack, MiddlePack, MFMAPack, SwapPack,
    LocalRead, MFMA,
    derive_pack_must_start_after,
    resolve_pack_type, PACK_TYPE_MAP,
    _compute_swap_register_pairs, _build_reg_to_lr_map,
    _hook_up_packs_f32_mfma,
    VGPRS_PER_CONVERSION_GROUP,
    get_dst_range, get_src_ranges, get_reg_range,
    SchedulePosition,
    create_unified_timeline,
    add_pack_constraints, add_local_read_constraints,
    validate_timeline, ValidationContext, isValid,
    _get_lrs_for_pack,
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
