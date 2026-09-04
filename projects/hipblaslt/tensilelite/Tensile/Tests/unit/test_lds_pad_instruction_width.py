# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The unroll-major LDS pad must be measured in local-read INSTRUCTION widths.

An LDS op is serviced 32 DWORDs per cycle, so a local read moving W bytes per
lane covers 32*4/W lanes per pass, and lane n of a pass sits at byte n*S. Those
lanes land on distinct banks iff

    S = W * odd

i.e. the row stride is an ODD multiple of the per-lane instruction width. A pad
of an EVEN number of instruction widths cannot change that parity, so it cannot
fix a conflict -- which is what `optPad = LocalReadVectorWidth` produces once a
row takes more than one instruction. On shipped configurations that is gfx10/
gfx11 WMMA at 16-bit, where MIInputPerThread is overridden to MatrixInstK.

The clamp is expressed as a cap at MAX_LOCAL_READ_BYTES rather than by
predicting which instruction the writer picks: below the cap the row IS one
instruction, so optPad stays at lrvw either way, and at or above it the writer
issues b128s. See test_cap_equals_per_instruction_width_on_every_shipped_type.
"""

import pytest

from Tensile.SolutionStructs.LdsPadding import MAX_LOCAL_READ_BYTES

pytestmark = pytest.mark.unit


def _opt_pad(lrvw, numBytes):
    """The clamp as Solution.calcLdsPad applies it."""
    return min(lrvw, int(MAX_LOCAL_READ_BYTES // numBytes))


# Every element size reachable from DataType/MacDataType across all shipped
# logic. 0.75 (fp6/bf6) is deliberately absent -- no shipped logic selects a
# 6-bit MacDataType, and it is the only width where the cap and a
# per-instruction width would disagree (see the test below).
SHIPPED_ELEMENT_BYTES = (0.5, 1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# The cap is the per-instruction width, for everything that ships
# ---------------------------------------------------------------------------

def _block_width_registers(widthRegisters):
    """Registers moved by one local read, per memoryInstructions["LocalRead"].

    Widest-first first fit, the non-combining rule of
    KernelWriterAssembly.findMemoryInstructionForWidthStride(). blockWidth 6
    (b192) is included even though rocisa expands it to ds_load_b128 +
    ds_load_b64 -- the point here is to reproduce the selector, not the ISA.
    """
    for blockWidth in (6, 4, 2, 2, 1, 1):
        if widthRegisters >= blockWidth and widthRegisters % blockWidth == 0:
            return blockWidth
    return 1


@pytest.mark.parametrize("numBytes", SHIPPED_ELEMENT_BYTES)
@pytest.mark.parametrize("lrvw", [1, 2, 4, 8, 16, 32])
def test_cap_equals_per_instruction_width_on_every_shipped_type(lrvw, numBytes):
    """Capping at 16 B and asking the instruction table agree everywhere that
    ships, so the simpler form loses nothing. They part only at 0.75 B/element,
    which no shipped MacDataType uses."""
    perInstruction = _block_width_registers(int(lrvw * numBytes // 4)) * 4
    assert _opt_pad(lrvw, numBytes) == min(lrvw, int(perInstruction // numBytes))


# ---------------------------------------------------------------------------
# The parity law
# ---------------------------------------------------------------------------

def _lanes_hit_distinct_banks(strideBytes, blockBytes, padBytes, instrBytes):
    banks, bankBytes = 32, 4
    lanesPerPass = banks * bankBytes // instrBytes
    seen = {}
    for lane in range(lanesPerPass):
        addr = lane * strideBytes
        if blockBytes:
            addr += (addr // blockBytes) * padBytes
        for reg in range(instrBytes // bankBytes):
            dword = addr // bankBytes + reg
            seen.setdefault(dword % banks, set()).add(dword)
    return max(len(v) for v in seen.values()) == 1


@pytest.mark.parametrize("multiple, clean", [(1, True), (2, False), (3, True), (4, False)])
def test_only_odd_instruction_multiples_are_conflict_free(multiple, clean):
    stride, instr = 128, MAX_LOCAL_READ_BYTES   # gfx1151 repro: DepthU 32 * bf16 * VW 2
    assert _lanes_hit_distinct_banks(stride, stride, multiple * instr, instr) is clean


def test_legacy_pad_is_an_even_multiple_and_therefore_inert():
    """Pin the defect: lrvw=16 at bf16 is 32B = 2 instruction widths."""
    lrvw, bpe, stride = 16, 2, 32 * 2 * 2
    assert (lrvw * bpe) // MAX_LOCAL_READ_BYTES == 2   # legacy pad = 2 widths, even
    assert not _lanes_hit_distinct_banks(stride, 128, lrvw * bpe, MAX_LOCAL_READ_BYTES)
    assert _opt_pad(lrvw, bpe) == 8                    # clamped to one width
    assert _lanes_hit_distinct_banks(stride, 128, _opt_pad(lrvw, bpe) * bpe,
                                     MAX_LOCAL_READ_BYTES)


# ---------------------------------------------------------------------------
# The clamp is a no-op wherever a row is a single instruction (all of MFMA/CDNA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lrvw, bpe",
    [(1, 4), (2, 4), (4, 4),        # readRegs 1, 2, 4
     (2, 2), (4, 2), (8, 2),
     (4, 1), (8, 1), (16, 1),
     (1, 8), (2, 8), (1, 16)],      # double / complex
)
def test_clamp_is_a_noop_for_single_instruction_rows(lrvw, bpe):
    """readRegs <= 4 is everything MFMA/CDNA can reach -- the b192 reject in
    Solution.py caps non-WMMA there -- so CDNA layouts must not move."""
    assert int(lrvw * bpe // 4) <= 4
    assert _opt_pad(lrvw, bpe) == lrvw


@pytest.mark.parametrize("lrvw, bpe, expected", [(16, 2, 8), (32, 1, 16)])
def test_clamp_halves_the_wide_wmma_rows(lrvw, bpe, expected):
    """The readRegs == 8 rows. gfx11 16-bit WMMA reaches this via the
    MIInputPerThread = MatrixInstK override; the 8-bit row is exercised for the
    rule's own sake, not because a shipped arch produces it."""
    assert int(lrvw * bpe // 4) == 8
    assert _opt_pad(lrvw, bpe) == expected


def test_cap_is_b128():
    """If a genuinely wider local read appears -- b256 already exists on the
    LocalWrite side -- this constant and the pad rule must be revisited
    together. b192 does not count: rocisa expands it to b128 + b64."""
    assert MAX_LOCAL_READ_BYTES == 16
