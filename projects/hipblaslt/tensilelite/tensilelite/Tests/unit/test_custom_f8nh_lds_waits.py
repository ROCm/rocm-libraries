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
            # Fifteen disables the LDS wait; it does not complete older operations.
            if keep < 15:
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
                hazards.append((line_number, sorted(used & outstanding)))
            mfmas += 1
    return hazards, reads, mfmas


@pytest.mark.parametrize("lgkmcnt", [0, 1, 2, 15])
@pytest.mark.parametrize("younger_write", [False, True])
def test_lds_checker_waits_for_both_reads(lgkmcnt, younger_write):
    # One pending operation is safe only when a write follows both operand reads.
    source = """.long 0xD9FE0000, 0x18000000 // ds_read_b128 v[24:27], v0
.long 0xD9FE0010, 0x1C000000 // ds_read_b128 v[28:31], v0 offset:16
"""
    if younger_write:
        source += ".long 0xD81A0000, 0x00002000 // ds_write_b32 v0, v32\n"
    source += f".long {0xBF8CC07F | (lgkmcnt << 8):#010x}\n"
    source += ".long 0xD3CD8000, 0x0402111C // MFMA reads v[28:29] and v[8:9]\n"
    expected = [] if lgkmcnt <= int(younger_write) else [(len(source.splitlines()), [28, 29])]
    assert _lds_hazards(source) == (expected, 2, 1)


def test_lds_checker_disabled_wait_keeps_all_pending_reads():
    source = ".long 0xD8EC0000, 0x22000000 // ds_read_b64 v[34:35], v0\n"
    source += ".long 0xD81A0000, 0x00002000 // ds_write_b32 v0, v32\n" * 15
    source += ".long 0xBF8C0F78 // s_waitcnt vmcnt(8), with no LDS wait\n"
    source += ".long 0xD3CD8000, 0x04021122 // MFMA reads v[34:35] and v[8:9]\n"
    assert _lds_hazards(source) == ([(18, [34, 35])], 1, 1)


@pytest.mark.parametrize("variant", [0, 1])
def test_f8nh_mfma_operands_have_completed_lds_reads(variant):
    kernel = f"Custom_Cijk_Ailk_Bljk_F8NH_HHS_BH_Bias_AS_SAB_SAV_shortname{variant}_gfx942"
    hazards, reads, mfmas = _lds_hazards(custom_kernel_text(kernel))
    assert reads >= 20 and mfmas >= 50, "The expected compute blocks were not checked"
    assert not hazards, f"{kernel}: (source line, pending VGPRs): {hazards}"
