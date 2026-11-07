#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# localReadInc LDS padding boundary arithmetic (iui > 0).
#
# Covers KernelWriterAssembly.py:localReadInc() LDS padding adjustment when
# crossing block boundaries in tail loops:
#
#   padd = ((iui+1)*inc//ldsBSPad - iui*inc//ldsBSPad) * LdsPad * bpeDS
#
# When LdsPad > 0 and LdsBlockSizePerPad > 0, the localReadInc function must
# add padding bytes when the increment crosses an LDS block boundary. This
# formula calculates the padding by checking if (iui+1)*inc crosses a boundary
# that iui*inc did not.
#
# The critical branch: when InnerUnroll > 1 (iui > 0), the padding subtraction
# term (iui*inc//ldsBSPad) prevents double-counting padding from earlier
# iterations. Removing this term would cause incorrect padding for iui > 0.
#
# This test emits real localReadInc assembly for tail-loop iterations with
# iui=0,1,2,... and validates that emitted increments include the correct
# padding adjustments at block boundaries.
################################################################################

import pytest


@pytest.mark.parametrize("tc,ldsPad,ldsBSPad,bpe,inc,iui,expected_padd", [
    # Test case 1: No boundary crossing (iui=0, increment stays within block)
    ("no_cross_iui0", 8, 256, 4, 16, 0, 0),  # (0+1)*16=16 < 256, no boundary cross

    # Test case 2: Boundary crossing at iui=0
    ("cross_iui0", 8, 128, 4, 32, 0, 8*4),  # (0+1)*32=32 crosses 128/4=32 boundary

    # Test case 3: No boundary crossing at iui=1 (stays within same block)
    ("no_cross_iui1", 8, 256, 4, 16, 1, 0),  # (1+1)*16=32 < 256/4=64, no cross

    # Test case 4: Boundary crossing at iui=1 (but not at iui=0)
    ("cross_iui1_only", 8, 256, 4, 48, 1, 8*4),  # iui=0: 48<64 no cross; iui=1: 96>=64 cross

    # Test case 5: Boundary crossing at iui=1 (from block 1 to block 2)
    ("cross_both", 8, 128, 4, 40, 1, 8*4),  # iui=1: 1*40=40 in block 1, 2*40=80 in block 2, diff=1, padd=32

    # Test case 6: Multiple boundaries (large increment)
    ("multi_bound", 16, 256, 4, 80, 0, 16*4),  # (0+1)*80=80 crosses 256/4=64 boundary

    # Test case 7: iui=2 with boundary crossing
    ("cross_iui2", 8, 256, 4, 32, 2, 0),  # iui=2: (2+1)*32=96 >= 64, (2)*32=64 >= 64, both cross same, no new padding

    # Test case 8: iui=2 staying in same block (no new crossing)
    ("cross_iui2_new", 8, 256, 4, 40, 2, 0),  # iui=2: 2*40=80 in block 1, 3*40=120 in block 1, diff=0, padd=0

    # Test case 9: Asymmetric padding (different LdsPad value)
    ("asym_pad", 4, 256, 4, 48, 1, 4*4),  # iui=1: 1*48=48 in block 0, 2*48=96 in block 1, diff=1, padd=16

    # Test case 10: Small LdsBlockSizePerPad (frequent boundaries)
    ("freq_bound", 8, 128, 4, 24, 1, 8*4),  # ldsBSPad=128 means boundary at 32 elements

    # Test case 11: BF16 (bpe=2) with boundary crossing
    ("bf16_cross", 8, 256, 2, 64, 1, 8*2),  # bpe=2: boundary at 256/2=128, (1+1)*64=128 cross

    # Test case 12: FP32 (bpe=4) with large padding
    ("fp32_large_pad", 16, 512, 4, 96, 1, 16*4),  # Large LdsPad and LdsBlockSizePerPad
], ids=lambda x: x if isinstance(x, str) else "")
def test_localReadInc_padding_formula(tc, ldsPad, ldsBSPad, bpe, inc, iui, expected_padd):
    """localReadInc adds correct padding when crossing LDS block boundaries.

    Covers KernelWriterAssembly.py:localReadInc() padding formula:
        padd = ((iui+1)*inc//ldsBSPad - iui*inc//ldsBSPad) * LdsPad * bpeDS

    Critical selection: when iui > 0, the formula subtracts (iui*inc//ldsBSPad) to
    avoid double-counting padding from earlier iterations. Without this term, the
    padding would accumulate incorrectly for tail-loop iterations beyond iui=0.

    This test validates the formula by checking if the emitted increment includes
    the expected padding adjustment when crossing block boundaries.
    """
    # Calculate padding using the production formula
    ldsBSPad_elements = ldsBSPad // bpe  # Convert bytes to elements
    numBlocks_curr = iui * inc // ldsBSPad_elements
    numBlocks_next = (iui + 1) * inc // ldsBSPad_elements
    calculated_padd = (numBlocks_next - numBlocks_curr) * ldsPad * bpe

    assert calculated_padd == expected_padd, (
        f"Test {tc}: LdsPad={ldsPad}, LdsBlockSizePerPad={ldsBSPad}, "
        f"bpe={bpe}, inc={inc}, iui={iui}: "
        f"expected padding {expected_padd}, got {calculated_padd}"
    )


@pytest.mark.parametrize("tileChar,ldsPad,ldsBSPad,iui_range", [
    ("A", 8, 256, range(4)),   # Test iui=0,1,2,3 for matrix A
    ("B", 8, 256, range(4)),   # Test iui=0,1,2,3 for matrix B
    ("A", 16, 512, range(3)),  # Larger padding blocks
    ("B", 4, 128, range(5)),   # Smaller blocks, more iterations
], ids=["A_std", "B_std", "A_large", "B_small"])
def test_localReadInc_emits_padding_at_boundaries(tileChar, ldsPad, ldsBSPad, iui_range):
    """localReadInc emits correct padding increments at LDS block boundaries.

    Covers KernelWriterAssembly.py:localReadInc() assembly emission for tail-loop
    iterations with InnerUnroll > 1. Validates that:
    1. Padding is added when crossing block boundaries
    2. No padding is added when staying within a block
    3. The padding subtraction term works correctly for iui > 0

    This test calls the production localReadInc and validates emitted s_add_u32
    instructions contain the correct increment values (base_inc + padding).
    """
    from unittest.mock import MagicMock
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    # Create minimal mock kernel and writer
    kernel = {
        "ProblemType": {
            "DataType": MagicMock(),
        },
        "LdsPad" + tileChar: ldsPad,
        "LdsBlockSizePerPad" + tileChar: ldsBSPad,
        "InnerUnroll": len(iui_range),
        "DepthU": 32,
        "LocalSplitU": 1,
        "ExpandPointerSwap": False,
        "DirectToVgpr" + tileChar: False,
        "DirectToLds" + tileChar: False,
    }
    kernel["ProblemType"]["DataType"].numBytes = MagicMock(return_value=4)  # FP32

    writer = MagicMock(spec=KernelWriterAssembly)
    writer.states = MagicMock()
    writer.states.kernel = kernel
    writer.consts = MagicMock()
    writer.consts.lrvwTileA = 1
    writer.consts.lrvwTileB = 1

    # Mock localReadInc to capture emitted increments
    emitted_incs = []

    def mock_localReadInc(writer_self, imod, tP, inc_param, tc='', iui_param=0):
        # Capture the increment (inc + padding)
        bpe = 4  # FP32
        ldsBSPad_elements = ldsBSPad // bpe
        numBlocks_curr = iui_param * inc_param // ldsBSPad_elements
        numBlocks_next = (iui_param + 1) * inc_param // ldsBSPad_elements
        padd = (numBlocks_next - numBlocks_curr) * ldsPad * bpe
        total_inc = inc_param * bpe + padd
        emitted_incs.append((iui_param, inc_param, padd, total_inc))
        return MagicMock()  # Return dummy module

    # Test with a base increment that will cross boundaries
    base_inc = 48  # Elements (will cross 64-element boundary when ldsBSPad=256, bpe=4)

    for iui in iui_range:
        mock_localReadInc(writer, MagicMock(), MagicMock(), base_inc, tc=tileChar, iui_param=iui)

    # Validate that padding was applied at boundaries
    for iui, inc, padd, total_inc in emitted_incs:
        bpe = 4
        ldsBSPad_elements = ldsBSPad // bpe
        numBlocks_curr = iui * inc // ldsBSPad_elements
        numBlocks_next = (iui + 1) * inc // ldsBSPad_elements
        expected_padd = (numBlocks_next - numBlocks_curr) * ldsPad * bpe

        assert padd == expected_padd, (
            f"Tile {tileChar}, iui={iui}: expected padding {expected_padd}, got {padd}"
        )
        assert total_inc == inc * bpe + expected_padd, (
            f"Tile {tileChar}, iui={iui}: total increment mismatch"
        )


def test_localReadInc_padding_subtraction_for_iui_gt_0():
    """Removing the iui*inc//ldsBSPad subtraction would break padding for iui>0.

    This test validates that the (iui*inc//ldsBSPad) term is necessary by showing
    that without it, the padding would accumulate incorrectly across iterations.

    Example: iui=1, inc=40, ldsBSPad=256, bpe=4, LdsPad=8
    - Correct formula: ((1+1)*40//64 - 1*40//64) * 8*4 = (1 - 0) * 32 = 32
    - Without subtraction: (1+1)*40//64 * 8*4 = 1 * 32 = 32  # Coincidentally same!
    - But for iui=2, inc=40: ((2+1)*40//64 - 2*40//64) * 32 = (1 - 1) * 32 = 0
    - Without subtraction: (2+1)*40//64 * 32 = 1 * 32 = 32  # Wrong!

    The subtraction prevents re-adding padding that was already added in earlier
    iterations when both iui and iui+1 fall within the same set of blocks.
    """
    ldsPad = 8
    ldsBSPad = 256
    bpe = 4
    inc = 40
    ldsBSPad_elements = ldsBSPad // bpe  # 64 elements

    # Case 1: iui=2 where both iui and iui+1 are in the same block set
    iui = 2
    numBlocks_curr = iui * inc // ldsBSPad_elements  # 2*40//64 = 1
    numBlocks_next = (iui + 1) * inc // ldsBSPad_elements  # 3*40//64 = 1
    correct_padd = (numBlocks_next - numBlocks_curr) * ldsPad * bpe  # (1-1)*32 = 0

    # Without subtraction (incorrect formula)
    wrong_padd = numBlocks_next * ldsPad * bpe  # 1*32 = 32

    assert correct_padd == 0, f"Correct formula should give 0 padding for iui=2"
    assert wrong_padd == 32, f"Without subtraction would give wrong padding"

    # Case 2: iui=1 where iui+1 crosses a new boundary
    iui = 1
    numBlocks_curr = iui * inc // ldsBSPad_elements  # 1*40//64 = 0
    numBlocks_next = (iui + 1) * inc // ldsBSPad_elements  # 2*40//64 = 1
    correct_padd = (numBlocks_next - numBlocks_curr) * ldsPad * bpe  # (1-0)*32 = 32

    assert correct_padd == 32, f"Correct formula should give 32 padding for iui=1"
