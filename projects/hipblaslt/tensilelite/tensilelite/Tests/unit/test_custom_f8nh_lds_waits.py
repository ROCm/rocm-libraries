# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check LDS-read dependencies within the encoded gfx942 F8NH compute blocks."""

import re

import pytest

from tensilelite.resources import custom_kernel_text

pytestmark = pytest.mark.unit


def _lds_hazards(source):
    # These sources store one instruction per .long directive. Decode the
    # gfx942 DS, SOPP and MFMA fields needed to check their local dependencies.
    instructions = [
        [int(word, 16) for word in re.findall(r"0x[0-9A-Fa-f]+", line)]
        for line in source.splitlines()
        if line.startswith(".long ")
    ]
    pending = []
    hazards = []
    reads = mfmas = 0
    for position, words in enumerate(instructions):
        word = words[0]
        opcode = word >> 16
        if opcode in (0xBF81, 0xBF82, 0xBF84, 0xBF85, 0xBF86, 0xBF87, 0xBF88, 0xBF89):
            # The check is confined to straight-line blocks; it makes no claim
            # about dependencies crossing a branch.
            pending.clear()
        elif opcode == 0xBF8C:
            keep = (word >> 8) & 0xF
            pending = pending[-keep:] if keep else []
        elif word >> 26 == 0x36:
            assert opcode in (0xD8EC, 0xD9FE, 0xD81A, 0xD9BE), hex(word)
            if opcode in (0xD8EC, 0xD9FE):
                first = words[1] >> 24
                width = 2 if opcode == 0xD8EC else 4
                pending.append(set(range(first, first + width)))
                reads += 1
            else:
                pending.append(set())
        elif word >> 26 == 0x30:
            # These compute blocks do not interleave scalar loads and LDS reads.
            assert not any(pending), "Scalar load encountered within an LDS-read interval"
            pending.clear()
        elif word & 0xFFFFFF00 == 0xD3CD8000:
            # v_mfma_f32_16x16x16_f16 reads two consecutive VGPRs per input.
            first = (words[1] & 0x1FF) - 256
            second = ((words[1] >> 9) & 0x1FF) - 256
            assert min(first, second) >= 0
            used = {first, first + 1, second, second + 1}
            outstanding = set().union(*pending)
            if used & outstanding:
                hazards.append((position, sorted(used & outstanding)))
            mfmas += 1
    assert reads >= 20 and mfmas >= 50, "The expected compute blocks were not checked"
    return hazards


@pytest.mark.parametrize("variant", [0, 1])
def test_f8nh_mfma_operands_have_completed_lds_reads(variant):
    kernel = f"Custom_Cijk_Ailk_Bljk_F8NH_HHS_BH_Bias_AS_SAB_SAV_shortname{variant}_gfx942"
    assert not _lds_hazards(custom_kernel_text(kernel))
