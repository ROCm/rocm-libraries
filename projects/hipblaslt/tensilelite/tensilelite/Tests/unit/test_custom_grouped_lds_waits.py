# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check LDS-read dependencies within the encoded gfx942 grouped compute blocks."""

import re

import pytest

from tensilelite.resources import custom_kernel_text

pytestmark = pytest.mark.unit


def _lds_hazards(source):
    # These sources store one instruction per .long directive. Decode the
    # gfx942 DS, SOPP and MFMA fields needed to check their local dependencies.
    pending = []
    hazards = []
    reads = mfmas = 0
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line.startswith(".long "):
            continue
        words = [int(word, 16) for word in re.findall(r"0x[0-9A-Fa-f]+", line.split("//")[0])]
        word = words[0]
        opcode = word >> 16
        if opcode in (0xBF81, 0xBF82, 0xBF84, 0xBF85, 0xBF86, 0xBF87, 0xBF88, 0xBF89):
            # The check is confined to straight-line blocks; it makes no claim
            # about dependencies crossing a branch.
            pending.clear()
        elif opcode == 0xBF8C:
            keep = (word >> 8) & 0xF
            if keep < 15:
                pending = pending[-keep:] if keep else []
        elif word >> 26 == 0x36:
            assert opcode in (0xD8EC, 0xD9FE, 0xD81A), hex(word)
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
        elif word & 0xFFFF7F00 == 0xD3CD0000:
            # v_mfma_f32_16x16x16_f16 reads two consecutive VGPRs per input.
            first = (words[1] & 0x1FF) - 256
            second = ((words[1] >> 9) & 0x1FF) - 256
            assert min(first, second) >= 0
            used = {first, first + 1, second, second + 1}
            outstanding = set().union(*pending)
            if used & outstanding:
                hazards.append((line_number, sorted(used & outstanding)))
            mfmas += 1
    return hazards, reads, mfmas


@pytest.mark.parametrize("lgkmcnt", [0, 1, 2, 9, 15])
def test_lds_checker_waits_for_read_before_younger_write(lgkmcnt):
    # With a read followed by a write, a limit of one must complete the read.
    # Two leaves the MFMA operands pending; fifteen does not wait at all.
    source = f"""
.long 0xD8EC0460, 0x34000069 // ds_read_b64 v[52:53], v105 offset:1120
.long 0xD81A0000, 0x00005466 // ds_write_b32 v102, v84
.long {0xBF8CC07F | (lgkmcnt << 8):#010x} // s_waitcnt lgkmcnt({lgkmcnt})
.long 0xD3CD8010, 0x04421134 // v_mfma_f32_16x16x16_f16 a[16:19], v[52:53], v[8:9], a[16:19]
"""
    expected = [] if lgkmcnt <= 1 else [(5, [52, 53])]
    assert _lds_hazards(source) == (expected, 1, 1)


@pytest.mark.parametrize("dtype", ["HHS", "HSS"])
@pytest.mark.parametrize("variant", [0, 1, 2, 3])
def test_grouped_mfma_operands_have_completed_lds_reads(dtype, variant):
    kernel = f"Custom_Cijk_Ailk_Bljk_{dtype}_BH_Bias_GG_AS_SAV_UserArgs_shortname{variant}_gfx942"
    hazards, reads, mfmas = _lds_hazards(custom_kernel_text(kernel))
    assert reads >= 20 and mfmas >= 50, "The expected compute blocks were not checked"
    assert not hazards, f"{kernel}: (source line, pending VGPRs): {hazards}"
