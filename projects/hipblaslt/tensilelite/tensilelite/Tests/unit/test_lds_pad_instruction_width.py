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

The clamp is expressed as a cap at Solution.MAX_NUM_DS_LOAD_BYTES rather than
by predicting which instruction the writer picks: below the cap the row IS one
instruction, so optPad stays at lrvw either way, and at or above it the writer
issues b128s. See test_cap_equals_per_instruction_width_on_every_shipped_type.
The cap is in LDS elements (DataType under ConvertAfterDS, else MacDataType).
"""

import copy

import pytest

from tensilelite.SolutionStructs.Solution import Solution

MAX_LOCAL_READ_BYTES = Solution.MAX_NUM_DS_LOAD_BYTES

pytestmark = pytest.mark.unit


_opt_pad = Solution.localReadInstructionPad


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


def test_six_bit_elements_are_not_clamped():
    """16 B is 21.33 6-bit elements; clamping to 21 would give a 15.75 B pad."""
    assert _opt_pad(32, 0.75) == 32


def test_cap_is_b128():
    """If a genuinely wider local read appears -- b256 already exists on the
    LocalWrite side -- this constant and the pad rule must be revisited
    together. b192 does not count: rocisa expands it to b128 + b64."""
    assert MAX_LOCAL_READ_BYTES == 16


# ---------------------------------------------------------------------------
# End to end through Solution: the pad calcLdsPad hands the kernel writer
# ---------------------------------------------------------------------------

@pytest.fixture
def _restore_globals():
    from tensilelite.Common.GlobalParameters import defaultSolution, globalParameters
    from tensilelite.Common.ValidParameters import validParameters
    from test_halfplr_streamk_rejects import _PRISTINE_DEFAULT_SOLUTION

    saved = [(d, copy.deepcopy(dict(d)))
             for d in (globalParameters, validParameters, defaultSolution)]
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    yield
    for d, value in saved:
        d.clear()
        d.update(value)


_BF16_TN = {
    "OperationType": "GEMM", "DataType": "B", "DestDataType": "B", "ComputeDataType": "s",
    "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
    "UseBeta": True, "Batched": True,
}


def _derive(arch, assembler, lrvw, problemType=_BF16_TN, depthU=32,
            mi=(16, 16, 16, 1, 1, 1, 2, 2, 2), wave=32, **extra):
    """TN, MI16x16x16, LdsPad auto; defaults are the gfx1151 repro at bf16
    (MT32x64, wave32)."""
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.GlobalParameters import assignGlobalParameters
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import \
        matrixInstructionToMIParameters
    from toolchain_test_helpers import shared_isa_info_map_or_skip

    iim = shared_isa_info_map_or_skip(arch)
    assignGlobalParameters({}, iim)
    isa = gfxToIsa(arch)
    problemType = dict(problemType)
    mi = list(mi)
    params = {
        "ProblemType": problemType, "ISA": isa, "MatrixInstruction": mi, "WavefrontSize": wave,
        "WorkGroup": [wave, 4, 1], "DepthU": depthU, "LocalReadVectorWidth": lrvw,
        "LdsPadA": -1, "LdsPadB": -1, "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
    }
    params.update(extra)
    params.update(matrixInstructionToMIParameters(mi, isa, params["WavefrontSize"], problemType,
                                                  params["WorkGroup"], iim))
    return Solution(params, False, True, False, assembler, iim)


def _conflict_free(sol, tc, ldsBpe):
    """The parity model on the layout the solution actually uses. Adjacent
    lanes are VectorWidth rows apart (LraTileAssignment: bnOffset * vw)."""
    stride = sol["DepthU"] * ldsBpe * sol[f"VectorWidth{tc}"]
    return _lanes_hit_distinct_banks(stride, sol[f"LdsBlockSizePerPad{tc}"],
                                     int(sol[f"LdsPad{tc}"] * ldsBpe), MAX_LOCAL_READ_BYTES)


@pytest.mark.parametrize("depthU", [32, 64])
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
def test_gfx11_wmma_auto_pad_is_one_instruction_width(_restore_globals, assembler, arch, depthU):
    """32 B/lane rows (two b128 reads). The unclamped pad was 16 elements =
    32 B = two instruction widths, which the parity model reports as
    conflicting on this layout."""
    sol = _derive(arch, assembler, lrvw=16, depthU=depthU)
    assert sol["Valid"]
    assert sol["LocalReadVectorWidthA"] == 16
    for tc in "AB":
        assert sol[f"UnrollMajorLDS{tc}"]
        assert sol[f"LdsPad{tc}"] * 2 == MAX_LOCAL_READ_BYTES
        assert _conflict_free(sol, tc, 2)


def test_single_instruction_row_keeps_its_pad(_restore_globals, assembler):
    """A row that is one b128 read (lrvw 8 at bf16, readRegs 4) is not
    clamped, so the MI16x16 doubling after the clamp still gives 2 * lrvw."""
    sol = _derive("gfx1201", assembler, lrvw=8)
    assert sol["Valid"]
    assert sol["LocalReadVectorWidthA"] == 8
    assert sol["LdsPadA"] == sol["LdsPadB"] == 16


@pytest.mark.parametrize("arch, dataType, lrvw, depthU, grvw, pad", [
    # fp8 held in LDS as fp8: 16 B rows, develop's 16 B pad is conflict-free.
    ("gfx1201", "F8", 16, 64, 8, 16),
    # fp32 held in LDS: the MI16x16 doubling decides in MacDataType units.
    ("gfx1201", "S", 8, 32, None, 16),
    ("gfx1151", "S", 16, 64, None, 16),
])
def test_convert_after_ds_keeps_its_pad(_restore_globals, assembler, arch, dataType, lrvw,
                                        depthU, grvw, pad):
    """When LDS holds DataType instead of MacDataType the clamp is skipped,
    so these pads equal develop's. Clamping in MacDataType bytes gave 8 for
    the fp8 row (half a b128); clamping in LDS elements moved the fp32 rows
    to a pad the doubling still leaves at an even number of reads."""
    problemType = dict(_BF16_TN, DataType="H", DestDataType="H",
                       DataTypeA=dataType, DataTypeB=dataType)
    extra = {"GlobalReadVectorWidthA": grvw, "GlobalReadVectorWidthB": grvw} if grvw else {}
    sol = _derive(arch, assembler, lrvw=lrvw, problemType=problemType, depthU=depthU,
                  ConvertAfterDS=True, **extra)
    assert sol["Valid"]
    assert sol["ConvertAfterDS"]
    assert sol["LocalReadVectorWidthA"] == lrvw
    assert sol["LdsPadA"] == sol["LdsPadB"] == pad


@pytest.mark.parametrize("arch, dataType, bpe, lrvw, pad", [
    ("gfx1201", "B", 2, 16, 8),     # explicit wide LRVW on gfx12
    ("gfx1100", "I8", 1, 32, 16),   # 8-bit, 32 B rows
])
def test_explicit_wide_wmma_rows_are_clamped(_restore_globals, assembler, arch, dataType,
                                             bpe, lrvw, pad):
    """Any WMMA row wider than one b128 gets a one-instruction pad."""
    isInt = dataType == "I8"
    problemType = dict(_BF16_TN, DataType=dataType, DestDataType="I" if isInt else dataType,
                       ComputeDataType="i" if isInt else "s")
    sol = _derive(arch, assembler, lrvw=lrvw, problemType=problemType, depthU=64)
    assert sol["Valid"]
    assert sol["LocalReadVectorWidthA"] == lrvw
    assert sol["LdsPadA"] == sol["LdsPadB"] == pad
    assert pad * bpe == MAX_LOCAL_READ_BYTES


@pytest.mark.parametrize("arch, mi, lrvw, pad", [
    ("gfx942", (16, 16, 16, 1, 1, 1, 2, 2, 2), 8, 16),
    ("gfx950", (32, 32, 16, 1, 1, 1, 2, 2, 2), 8, 8),
])
def test_mfma_pad_is_unchanged(_restore_globals, assembler, arch, mi, lrvw, pad):
    """fp32 inputs held in LDS and converted to f16 after the read: in LDS
    elements a row is two b128, but the MFMA rule is sized in MacDataType
    units and must not move (pads as on develop). Clamping in LDS elements
    gave 8 on gfx942 and 4 on gfx950."""
    problemType = dict(_BF16_TN, DataType="H", DestDataType="H",
                       DataTypeA="S", DataTypeB="S")
    sol = _derive(arch, assembler, lrvw=lrvw, problemType=problemType,
                  depthU=64 if mi[0] == 32 else 32, mi=mi, wave=64, ConvertAfterDS=True)
    assert sol["Valid"]
    assert sol["LdsPadA"] == sol["LdsPadB"] == pad
