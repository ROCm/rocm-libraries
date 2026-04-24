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
################################################################################
"""Mock builders for CMS Validator unit tests.

These create minimal rocisa instruction objects that satisfy the validator's
type inspection and register extraction without requiring a full kernel writer.

Register spaces are widely separated to avoid collisions:
    LRA:              v[1000..]   (LR[i].dst = v[1000 + i*4 : +4])
    LRB:              v[2000..]
    CVT0 intermediate: v[3000..]  (CVT0 dst -> MFMAPack .b / Middle-16 src)
    TF32 intermediate: v[4000..]  (MFMAPack .acc / Middle-16 tmp+dst -> CVT1 src)
    PackA output:      v[5000..]  (Pack dst -> MFMA .a)
    PackB output:      v[6000..]  (Pack dst -> MFMA .b)
    MFMA acc:          v[7000..]
    Swap packs:        v[8000..]
    MFMAPack .a:       v[9000..]  (identity matrix operand -- outside all spaces)
"""

from rocisa.instruction import (
    MFMAInstruction, VPermB32, DSLoadB128, SWaitCnt, SBarrier, SNop,
    SMovB32, SAddU32, BufferLoadB128, SCmpEQU32, SCSelectB32,
    SAddCU32, SSubU32,
)
from rocisa.container import vgpr, sgpr
from rocisa.enum import InstType

# --- Register space constants ---
LRA_BASE = 1000
LRB_BASE = 2000
CVT0_BASE = 3000
TF32_INTERMEDIATE_BASE = 4000
PACK_A_DST_BASE = 5000
PACK_B_DST_BASE = 6000
MFMA_ACC_BASE = 7000
SWAP_BASE = 8000
MFMA_PACK_A_BASE = 9000


def make_mock_mfma_code(num_mfma) -> list:
    """Build a list of mock MFMAInstruction objects.

    Each MFMA reads from PackA output space (.a) and PackB output space (.b),
    matching the register chains set up by _make_mock_packs_*.
    """
    if not num_mfma:
        return []
    mfmas = []
    for i in range(int(num_mfma)):
        acc_start = MFMA_ACC_BASE + i * 4
        a_start = PACK_A_DST_BASE + i * 4
        b_start = PACK_B_DST_BASE + i * 4
        mfmas.append(MFMAInstruction(
            instType=InstType.INST_BF16, accType=InstType.INST_F32,
            variant=[16, 16, 32, 1], mfma1k=False,
            acc=vgpr(acc_start, 4), a=vgpr(a_start, 4),
            b=vgpr(b_start, 4), acc2=vgpr(acc_start, 4),
        ))
    return mfmas


def _make_mock_lr(count: int, base_reg: int = LRA_BASE) -> list:
    """Create mock DSLoadB128 instructions for local reads.

    LR[i] writes 4 VGPRs: v[base_reg + i*4 : base_reg + i*4 + 4].
    """
    return [DSLoadB128(dst=vgpr(base_reg + i * 4, 4), src=vgpr(0, 1))
            for i in range(count)]


def _make_mock_packs_bf16(count: int, pack_dst_base: int = PACK_A_DST_BASE,
                          lr_base: int = LRA_BASE, num_lrs: int = 8) -> list:
    """Create mock VPermB32 instructions for BF16 packs.

    Each pack reads from TWO different LRs (D0 and D1 dimensions), matching
    the positional code's element_idx mapping:
      element_idx = i % num_element_pairs
      LR indices: element_idx*2 (D0) and element_idx*2+1 (D1)

    Pack[i]: src0 = one VGPR from LR[element_idx*2 + 1]
             src1 = one VGPR from LR[element_idx*2]
             dst  = v[pack_dst_base + i]
    """
    num_element_pairs = max(num_lrs // 2, 1)
    items = []
    for i in range(count):
        element_idx = i % num_element_pairs
        # src0 from LR[element_idx*2 + 1], src1 from LR[element_idx*2]
        # Each LR occupies 4 VGPRs at lr_base + lr_idx*4
        lr_idx_0 = element_idx * 2      # D0 LR
        lr_idx_1 = element_idx * 2 + 1  # D1 LR
        items.append(VPermB32(
            dst=vgpr(pack_dst_base + i, 1),
            src0=vgpr(lr_base + lr_idx_1 * 4, 1),  # one reg from D1 LR
            src1=vgpr(lr_base + lr_idx_0 * 4, 1),  # one reg from D0 LR
            src2=sgpr(0, 1),
        ))
    return items


def _make_mock_gr(count: int, dtl: bool = True) -> list:
    """Create mock GR instruction sequences.

    For DTL=1: interleaved [SMovB32, BufferLoadB128] pairs (count must be even).
    For DTL=0: all BufferLoadB128 (any count).
    """
    items = []
    if dtl:
        # DTL=1: interleaved SMovB32/SAddU32 + BufferLoadB128 pairs
        for i in range(count):
            if i % 2 == 0:
                if i == 0:
                    items.append(SMovB32(dst=sgpr(0, 1), src=sgpr(1, 1)))
                else:
                    items.append(SAddU32(dst=sgpr(0, 1), src0=sgpr(0, 1), src1=4096))
            else:
                items.append(BufferLoadB128(dst=None, vaddr=vgpr(i, 1),
                                           saddr=sgpr(0, 4), soffset=0))
    else:
        # Non-DTL: all are buffer loads
        for i in range(count):
            items.append(BufferLoadB128(dst=None, vaddr=vgpr(i, 1),
                                       saddr=sgpr(0, 4), soffset=0))
    return items


def _make_mock_grinc(count: int) -> list:
    """Create mock scalar instructions for GRInc sequences."""
    items = []
    for i in range(count):
        items.append(SCmpEQU32(src0=sgpr(0, 1), src1=sgpr(1, 1)))
    return items


def _make_mock_swap_packs(count: int, lr_base: int = LRA_BASE, vw: int = 1) -> list:
    """Create mock VSwapB32 instructions for swap packs.

    When vw > 1, uses the real interleaving pattern from
    _compute_swap_register_pairs to determine which register pairs get
    swapped. Each swap reads and writes two VGPRs within the LR space,
    offset by lr_base.

    When vw <= 1 (no swaps needed), returns empty list.
    """
    from rocisa.instruction import VSwapB32
    if vw <= 1:
        return []
    from Tensile.Components.CMSValidator import _compute_swap_register_pairs, VGPRS_PER_CONVERSION_GROUP
    total_regs = VGPRS_PER_CONVERSION_GROUP * vw
    swap_pairs = _compute_swap_register_pairs(vw, total_regs)
    items = []
    for i in range(min(count, len(swap_pairs))):
        reg_src, reg_dst = swap_pairs[i]
        items.append(VSwapB32(dst=vgpr(lr_base + reg_dst, 1),
                              src=vgpr(lr_base + reg_src, 1)))
    # If count > len(swap_pairs), pad with identity swaps (shouldn't happen in practice)
    while len(items) < count:
        items.append(VSwapB32(dst=vgpr(lr_base, 1), src=vgpr(lr_base + 1, 1)))
    return items


def _make_mock_swap(count: int) -> list:
    """Create mock SMovB32 for LR/LW swap operations."""
    from rocisa.instruction import VXorB32
    return [VXorB32(dst=vgpr(0, 1), src0=vgpr(0, 1), src1=vgpr(1, 1))
            for _ in range(count)]


def _make_mock_lw_swap(count: int) -> list:
    """Create mock SXorB32 for LW swap operations."""
    from rocisa.instruction import SXorB32
    return [SXorB32(dst=sgpr(0, 1), src0=sgpr(0, 1), src1=sgpr(1, 1))
            for _ in range(count)]


def _make_mock_lcc() -> list:
    """Create mock loop counter code (SSubU32 + SCmpEQI32)."""
    from rocisa.instruction import SCmpEQI32
    return [
        SSubU32(dst=sgpr(0, 1), src0=sgpr(0, 1), src1=1),
        SCmpEQI32(src0=sgpr(0, 1), src1=2),
    ]


def _make_mock_packs_tf32_4x4(count: int, pack_dst_base: int = PACK_A_DST_BASE,
                               lr_base: int = LRA_BASE) -> list:
    """Create mock instructions for TF32 4x4 MFMA pack groups.

    Group structure per 10 instructions:
      CVT0[0..3]:     v_cvt_pk_bf16_f32, reads from LR space, writes to CVT0 intermediate
      MFMAPack[0]:    v_mfma_f32_4x4x4, reads CVT0 output (.b), writes back to LR space (.acc)
      MFMAPack[1]:    v_mfma_f32_4x4x4, reads CVT0 output (.b), writes to pack output space (.acc)
      CVT1[0..3]:     v_cvt_pk_bf16_f32, reads from MFMA acc outputs, writes to pack output (reverse order)

    The CVT1 reverse-write pattern creates WAR hazards that derive_pack_must_start_after
    captures: CVT1[j] overwrites a register that CVT1[j-1] previously read.
    """
    from rocisa.instruction import VCvtPkF32toBF16
    items = []
    group_size = 10
    for i in range(count):
        group = i // group_size
        idx_in_group = i % group_size
        if idx_in_group < 4:
            # CVT0[j]: reads from LR space, writes to CVT0 intermediate
            j = idx_in_group
            items.append(VCvtPkF32toBF16(
                dst=vgpr(CVT0_BASE + group * 4 + j, 1),
                src0=vgpr(lr_base + group * 8 + j * 2, 1),
                src1=vgpr(lr_base + group * 8 + j * 2 + 1, 1),
            ))
        elif idx_in_group == 4:
            # MFMAPack[0]: .b reads CVT0[0..1], .acc writes back to LR T-space
            items.append(MFMAInstruction(
                instType=InstType.INST_BF16, accType=InstType.INST_F32,
                variant=[4, 4, 4, 1], mfma1k=False,
                acc=vgpr(lr_base + group * 8, 4),
                a=vgpr(MFMA_PACK_A_BASE, 2),
                b=vgpr(CVT0_BASE + group * 4, 2),
                acc2=vgpr(lr_base + group * 8, 4),
            ))
        elif idx_in_group == 5:
            # MFMAPack[1]: .b reads CVT0[2..3], .acc writes to pack output space
            items.append(MFMAInstruction(
                instType=InstType.INST_BF16, accType=InstType.INST_F32,
                variant=[4, 4, 4, 1], mfma1k=False,
                acc=vgpr(pack_dst_base + group * 4, 4),
                a=vgpr(MFMA_PACK_A_BASE, 2),
                b=vgpr(CVT0_BASE + group * 4 + 2, 2),
                acc2=vgpr(pack_dst_base + group * 4, 4),
            ))
        else:
            # CVT1[j]: reverse order writes into pack output space
            # CVT1[0] (idx 6): reads from MFMAPack[1] acc (pack output), writes highest dst
            # CVT1[1] (idx 7): reads from MFMAPack[1] acc, writes next dst (WAR on CVT1[0])
            # CVT1[2] (idx 8): reads from MFMAPack[0] acc (LR space), writes next dst (WAR on CVT1[1])
            # CVT1[3] (idx 9): reads from MFMAPack[0] acc (LR space), writes lowest dst (WAR on CVT1[1])
            j = idx_in_group - 6  # 0, 1, 2, 3
            dst_offset = 3 - j  # reverse: 3, 2, 1, 0
            if j < 2:
                # CVT1[0,1]: source from MFMAPack[1] acc (pack output space)
                src0_reg = pack_dst_base + group * 4 + (1 - j) * 2
                src1_reg = pack_dst_base + group * 4 + (1 - j) * 2 + 1
            else:
                # CVT1[2,3]: source from MFMAPack[0] acc (LR T-space)
                src0_reg = lr_base + group * 8 + (3 - j) * 2
                src1_reg = lr_base + group * 8 + (3 - j) * 2 + 1
            items.append(VCvtPkF32toBF16(
                dst=vgpr(pack_dst_base + group * 4 + dst_offset, 1),
                src0=vgpr(src0_reg, 1),
                src1=vgpr(src1_reg, 1),
            ))
    return items


def _make_mock_packs_tf32(count: int, pack_dst_base: int = PACK_A_DST_BASE,
                          lr_base: int = LRA_BASE) -> list:
    """Create mock instructions for TF32 (non-4x4) pack groups.

    Group structure per 24 instructions:
      CVT0[0..3]:      v_cvt_pk_bf16_f32, reads from LR space, writes to CVT0 intermediate
      Middle-16[0..15]: v_sub_f32 pairs sharing a tmp register per group
      CVT1[0..3]:      v_cvt_pk_bf16_f32, reads from middle output, writes to pack output (reverse)

    Middle-16 pairs share a "tmp" register (TF32_INTERMEDIATE_BASE + group):
      Even k: writes to tmp (v_cvt_f32_bf16 equivalent)
      Odd k:  reads from tmp (v_sub_f32 equivalent), writes to middle output space
    This creates within-pair RAW deps matching positional {5:[4], 7:[6], ...}.

    CVT1 uses reverse-write pattern (same as TF32 4x4) for WAR hazard chain.
    """
    from rocisa.instruction import VCvtPkF32toBF16, VSubF32 as _VSubF32
    items = []
    group_size = 24
    for i in range(count):
        group = i // group_size
        idx_in_group = i % group_size
        if idx_in_group < 4:
            # CVT0[j]: reads from LR space, writes to CVT0 intermediate
            j = idx_in_group
            items.append(VCvtPkF32toBF16(
                dst=vgpr(CVT0_BASE + group * 4 + j, 1),
                src0=vgpr(lr_base + group * 8 + j * 2, 1),
                src1=vgpr(lr_base + group * 8 + j * 2 + 1, 1),
            ))
        elif idx_in_group < 20:
            # Middle-16[k]: pairs sharing tmp register.
            # Real code: v_cvt_f32_bf16 (even) writes to tmp from CVT0 output,
            #            v_sub_f32 (odd) reads tmp and the T-reg, writes to output.
            # Both read from CVT0 output space, NOT directly from LR space.
            k = idx_in_group - 4  # 0..15
            tmp_reg = TF32_INTERMEDIATE_BASE + group
            mid_out_base = TF32_INTERMEDIATE_BASE + 100 + group * 16
            cvt0_idx = k // 4  # which CVT0 this pair depends on
            if k % 2 == 0:
                # Even (v_cvt_f32_bf16): reads from CVT0 output, writes to tmp
                items.append(_VSubF32(
                    dst=vgpr(tmp_reg, 1),
                    src0=vgpr(CVT0_BASE + group * 4 + cvt0_idx, 1),
                    src1=vgpr(CVT0_BASE + group * 4 + cvt0_idx, 1),
                ))
            else:
                # Odd (v_sub_f32): reads from CVT0 output and tmp, writes to middle output
                items.append(_VSubF32(
                    dst=vgpr(mid_out_base + k, 1),
                    src0=vgpr(CVT0_BASE + group * 4 + cvt0_idx, 1),
                    src1=vgpr(tmp_reg, 1),
                ))
        else:
            # CVT1[j]: reverse order writes into pack output space.
            # In real code, v4..v7 (src, dst=False) and v4d..v7d (dst, dst=True) are
            # the SAME X registers. CVT1 reads and writes within the pack output block,
            # creating the WAR hazard chain (each CVT1 overwrites a register the
            # previous CVT1 read).
            #   CVT1[0]: reads pack_dst+2, pack_dst+3, writes pack_dst+3
            #   CVT1[1]: reads pack_dst+0, pack_dst+1, writes pack_dst+2 (WAR on CVT1[0])
            #   CVT1[2]: reads from middle/LR space, writes pack_dst+1 (WAR on CVT1[1])
            #   CVT1[3]: reads from middle/LR space, writes pack_dst+0 (WAR on CVT1[1])
            j = idx_in_group - 20  # 0, 1, 2, 3
            dst_offset = 3 - j  # reverse: 3, 2, 1, 0
            pack_base = pack_dst_base + group * 4
            if j < 2:
                # CVT1[0,1]: source from pack output space (same as dst space)
                src0_reg = pack_base + (1 - j) * 2
                src1_reg = pack_base + (1 - j) * 2 + 1
            else:
                # CVT1[2,3]: source from middle-16 output space
                mid_out_base = TF32_INTERMEDIATE_BASE + 100 + group * 16
                src0_reg = mid_out_base + (3 - j) * 4 + 3
                src1_reg = mid_out_base + (3 - j) * 4 + 1
            items.append(VCvtPkF32toBF16(
                dst=vgpr(pack_base + dst_offset, 1),
                src0=vgpr(src0_reg, 1),
                src1=vgpr(src1_reg, 1),
            ))
    return items


def make_mock_id_map(schedule_info, kernel=None) -> dict:
    """Build a minimal mock id_map from a ScheduleInfo.

    For each key in optSchedule, creates a list of mock rocisa instruction
    objects of the correct length with the correct types for the validator's
    type resolution to work.

    If kernel is provided, uses it to determine TF32 emulation mode for
    generating the correct pack instruction types.
    """
    id_map = {}
    opt = schedule_info.optSchedule

    # Determine pack mode from kernel if available
    is_tf32 = kernel.get("UseF32XEmulation", False) if kernel else False
    is_4x4_tf32 = kernel.get("UseMFMAF32XEmulation", False) if kernel else False

    for key in opt:
        # Get the number of instructions from the first code path
        val = opt[key]
        if not val:
            count = 0
        elif isinstance(val[0], list):
            count = len(val[0])
        else:
            count = len(val)  # Single code path — val is the list directly

        if key == "SYNC":
            id_map[key] = list(schedule_info.syncCode) if schedule_info.syncCode else []
        elif key == "SNOP":
            id_map[key] = list(schedule_info.snopCode) if schedule_info.snopCode else []
        elif key.startswith("LRA") and not key.startswith("LRS"):
            id_map[key] = _make_mock_lr(count, base_reg=LRA_BASE)
        elif key.startswith("LRB") and not key.startswith("LRS"):
            id_map[key] = _make_mock_lr(count, base_reg=LRB_BASE)
        elif key == "GRA":
            id_map[key] = _make_mock_gr(count, dtl=kernel.get("DirectToLds", True) if kernel else True)
        elif key == "GRB":
            id_map[key] = _make_mock_gr(count, dtl=kernel.get("DirectToLds", True) if kernel else True)
        elif key == "GRIncA":
            id_map[key] = _make_mock_grinc(count)
        elif key == "GRIncB":
            id_map[key] = _make_mock_grinc(count)
        elif key.startswith("Pack"):
            is_a = key.startswith("PackA")
            lr_base = LRA_BASE if is_a else LRB_BASE
            pack_dst_base = PACK_A_DST_BASE if is_a else PACK_B_DST_BASE
            if is_4x4_tf32 or is_tf32:
                # Compute number of leading SwapPacks based on VW and TLUA/TLUB
                n_swaps = 0
                if kernel:
                    tlu_key = "TLUA" if is_a else "TLUB"
                    needs_transpose = kernel.get("ProblemType", {}).get(tlu_key, False)
                    if needs_transpose:
                        vw = kernel.get("VectorWidthA" if is_a else "VectorWidthB", 1)
                        if vw > 1:
                            n_swaps = 4 * (vw - 1)
                vw_for_swap = kernel.get("VectorWidthA" if is_a else "VectorWidthB", 1) if kernel else 1
                swap_items = _make_mock_swap_packs(n_swaps, lr_base=lr_base, vw=vw_for_swap)
                remaining = count - n_swaps
                if is_4x4_tf32:
                    pack_items = _make_mock_packs_tf32_4x4(remaining,
                                                           pack_dst_base=pack_dst_base,
                                                           lr_base=lr_base)
                else:
                    pack_items = _make_mock_packs_tf32(remaining,
                                                       pack_dst_base=pack_dst_base,
                                                       lr_base=lr_base)
                id_map[key] = swap_items + pack_items
            else:
                # Find corresponding LR count for element_idx mapping
                lr_name = key.replace("Pack", "LR")
                lr_val = opt.get(lr_name, [])
                if lr_val and isinstance(lr_val[0], list):
                    num_lrs = len(lr_val[0])
                elif lr_val:
                    num_lrs = len(lr_val)
                else:
                    num_lrs = 8  # default
                id_map[key] = _make_mock_packs_bf16(count,
                                                     pack_dst_base=pack_dst_base,
                                                     lr_base=lr_base,
                                                     num_lrs=num_lrs)
        elif key == "LRSA" or key == "LRSB":
            id_map[key] = _make_mock_swap(count)
        elif key == "LWSA" or key == "LWSB":
            id_map[key] = _make_mock_lw_swap(count)
        elif key == "LWA" or key == "LWB":
            id_map[key] = []  # Empty for DTL=1
        elif key == "LCC":
            # Create enough LCC entries — repeat the pattern if count > 2
            base = _make_mock_lcc()
            id_map[key] = (base * ((count // len(base)) + 1))[:count]
        else:
            # Unknown key — create empty list
            id_map[key] = [None] * count

    return id_map
