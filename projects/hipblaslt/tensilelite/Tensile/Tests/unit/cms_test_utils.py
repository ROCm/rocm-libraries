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
"""

from rocisa.instruction import (
    MFMAInstruction, VPermB32, DSLoadB128, SWaitCnt, SBarrier, SNop,
    SMovB32, SAddU32, BufferLoadB128, SCmpEQU32, SCSelectB32,
    SAddCU32, SSubU32,
)
from rocisa.container import vgpr, sgpr
from rocisa.enum import InstType


def make_mock_mfma_code(num_mfma) -> list:
    """Build a list of mock MFMAInstruction objects.

    Each MFMA gets unique accumulator and source register ranges so that
    register-based dependency tracing can distinguish them.
    """
    if not num_mfma:
        return []
    mfmas = []
    for i in range(int(num_mfma)):
        acc_start = i * 4
        a_start = 1000 + i * 4
        b_start = 2000 + i * 4
        mfmas.append(MFMAInstruction(
            instType=InstType.INST_BF16, accType=InstType.INST_F32,
            variant=[16, 16, 32, 1], mfma1k=False,
            acc=vgpr(acc_start, 4), a=vgpr(a_start, 4),
            b=vgpr(b_start, 4), acc2=vgpr(acc_start, 4),
        ))
    return mfmas


def _make_mock_lr(count: int, base_reg: int = 100) -> list:
    """Create mock DSLoadB128 instructions for local reads."""
    return [DSLoadB128(dst=vgpr(base_reg + i * 4, 4), src=vgpr(0, 1))
            for i in range(count)]


def _make_mock_packs_bf16(count: int, base_reg: int = 500) -> list:
    """Create mock VPermB32 instructions for BF16 packs."""
    return [VPermB32(dst=vgpr(base_reg + i, 1),
                     src0=vgpr(base_reg + 100 + i, 1),
                     src1=vgpr(base_reg + 200 + i, 1),
                     src2=sgpr(0, 1))
            for i in range(count)]


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


def _make_mock_swap_packs(count: int) -> list:
    """Create mock VSwapB32 instructions for swap packs at the start of TF32 pack sequences."""
    from rocisa.instruction import VSwapB32
    return [VSwapB32(dst=vgpr(i, 1), src=vgpr(i + 1, 1)) for i in range(count)]


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


def _make_mock_packs_tf32_4x4(count: int, base_reg: int = 500) -> list:
    """Create mock instructions for TF32 4x4 MFMA pack groups.

    Group structure: [CVT×4, MFMA×2, CVT×4] = 10 per group.
    Uses VCvtPkF32toBF16 for CVT and MFMAInstruction for MFMA packs.
    """
    from rocisa.instruction import VCvtPkF32toBF16
    items = []
    group_size = 10
    for i in range(count):
        idx_in_group = i % group_size
        if 4 <= idx_in_group < 6:
            # MFMA pack
            items.append(MFMAInstruction(
                instType=InstType.INST_BF16, accType=InstType.INST_F32,
                variant=[4, 4, 4, 1], mfma1k=False,
                acc=vgpr(base_reg + i * 4, 4), a=vgpr(base_reg + 1000, 2),
                b=vgpr(base_reg + i * 2, 2), acc2=vgpr(base_reg + i * 4, 4),
            ))
        else:
            # CVT pack
            items.append(VCvtPkF32toBF16(
                dst=vgpr(base_reg + i, 1),
                src0=vgpr(base_reg + 200 + i, 1),
                src1=vgpr(base_reg + 300 + i, 1),
            ))
    return items


def _make_mock_packs_tf32(count: int, base_reg: int = 500) -> list:
    """Create mock instructions for TF32 (non-4x4) pack groups.

    Group structure: [CVT×4, Middle×16, CVT×4] = 24 per group.
    Uses VCvtPkF32toBF16 for CVT and VSubF32 for middle-16.
    """
    from rocisa.instruction import VCvtPkF32toBF16, VSubF32 as _VSubF32
    items = []
    group_size = 24
    for i in range(count):
        idx_in_group = i % group_size
        if 4 <= idx_in_group < 20:
            # Middle-16 pack
            items.append(_VSubF32(
                dst=vgpr(base_reg + i, 1),
                src0=vgpr(base_reg + 200 + i, 1),
                src1=vgpr(base_reg + 300 + i, 1),
            ))
        else:
            # CVT pack
            items.append(VCvtPkF32toBF16(
                dst=vgpr(base_reg + i, 1),
                src0=vgpr(base_reg + 200 + i, 1),
                src1=vgpr(base_reg + 300 + i, 1),
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
            id_map[key] = _make_mock_lr(count, base_reg=100)
        elif key.startswith("LRB") and not key.startswith("LRS"):
            id_map[key] = _make_mock_lr(count, base_reg=200)
        elif key == "GRA":
            id_map[key] = _make_mock_gr(count, dtl=kernel.get("DirectToLds", True) if kernel else True)
        elif key == "GRB":
            id_map[key] = _make_mock_gr(count, dtl=kernel.get("DirectToLds", True) if kernel else True)
        elif key == "GRIncA":
            id_map[key] = _make_mock_grinc(count)
        elif key == "GRIncB":
            id_map[key] = _make_mock_grinc(count)
        elif key.startswith("Pack"):
            if is_4x4_tf32 or is_tf32:
                # Compute number of leading SwapPacks based on VW and TLUA/TLUB
                n_swaps = 0
                if kernel:
                    is_a = key.startswith("PackA")
                    tlu_key = "TLUA" if is_a else "TLUB"
                    needs_transpose = kernel.get("ProblemType", {}).get(tlu_key, False)
                    if needs_transpose:
                        vw = kernel.get("VectorWidthA" if is_a else "VectorWidthB", 1)
                        if vw > 1:
                            n_swaps = 4 * (vw - 1)
                swap_items = _make_mock_swap_packs(n_swaps)
                remaining = count - n_swaps
                if is_4x4_tf32:
                    pack_items = _make_mock_packs_tf32_4x4(remaining)
                else:
                    pack_items = _make_mock_packs_tf32(remaining)
                id_map[key] = swap_items + pack_items
            else:
                id_map[key] = _make_mock_packs_bf16(count)
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
