#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the w4a16 in-kernel group dequantization path.

Two layers:

* Validator tests -- pure Python, no toolchain. They pin the shapes
  ``UseScaleAB="Block"`` is allowed to run on.
* Codegen tests -- emit a real gfx1151 kernel and inspect the assembly. These
  need an amdclang++ that targets gfx1151 and are skipped otherwise.

The codegen assertions are deliberately about *structure* (one scale load per A
load, one dequantize block per LDS write, no VGPR aliasing) rather than exact
instruction text, so ordinary scheduling changes do not churn them.
"""

import os
import re
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if TENSILE_ROOT not in sys.path:
    sys.path.insert(0, TENSILE_ROOT)

pytestmark = pytest.mark.unit

ARCH = "gfx1151"


# ---------------------------------------------------------------------------
# Solution construction
# ---------------------------------------------------------------------------
def _w4a16_params(iim, isa, *, blockSize=32, depthU=128, glvwA=8, zeroPoint=False, **overrides):
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    problemType = {
        "OperationType": "GEMM",
        "DataType": "B",
        "DataTypeA": "I4",
        "MacDataTypeA": "B",
        "DestDataType": "B",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
        "UseScaleAB": "Block",
        "ScaleBlockSizeA": blockSize,
        "ScaleZeroPointA": zeroPoint,
    }
    problemType.update(overrides.pop("problemType", {}))

    params = {
        "ProblemType": problemType,
        "ISA": isa,
        "MatrixInstruction": [16, 16, 16, 1, 1, 2, 2, 2, 2],
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": depthU,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        # Explicit 0: MT64x64 x DepthU=128 is double-buffered bf16, which is
        # exactly the gfx1151 64 KiB LDS budget -- any auto pad overflows it.
        "LdsPadA": 0,
        "LdsPadB": 0,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": glvwA,
        "GlobalReadVectorWidthB": -1,
        # gfx11 WMMA has MIInputPerThread == MatrixInstK == 16, so the auto
        # width (8) is narrower than one MI input and gets rejected. 16 is what
        # the shipped gfx1151 logic uses.
        "LocalReadVectorWidth": 16,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "StreamK": 0,
        "PrefetchAcrossPersistent": 0,
        "PrefetchGL2": 0,
        "UseSubtileImpl": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "ClusterLocalRead": 0,
        "ConvertAfterDS": False,
        "AssertSummationElementMultiple": depthU,
    }
    params.update(overrides)
    params.update(
        matrixInstructionToMIParameters(
            params["MatrixInstruction"], isa, params["WavefrontSize"],
            params["ProblemType"], params["WorkGroup"], iim,
        )
    )
    return params


@pytest.fixture(scope="module")
def toolchain():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    try:
        cxx = validateToolchain("amdclang++")
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"no amdclang++: {exc}")
    isa = gfxToIsa(ARCH)
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip(f"amdclang++ here does not support {ARCH}")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    return isa, iim, assembler, cxx


def _solution(toolchain, **kw):
    from Tensile.SolutionStructs.Solution import Solution

    isa, iim, assembler, _cxx = toolchain
    return Solution(_w4a16_params(iim, isa, **kw), False, True, False, assembler, iim)


def _emit(toolchain, **kw):
    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    isa, iim, assembler, cxx = toolchain
    sol = _solution(toolchain, **kw)
    assert sol.get("Valid") is True, "w4a16 solution should derive cleanly"

    kernels = generateKernelObjectsFromSolutions([sol])
    kernel = kernels[0]
    ri = rocisa.rocIsa.getInstance()
    # The fixture already resolved the compiler through TensileLite's validator,
    # which searches the ROCm install; shutil.which alone would need it on PATH.
    ri.init(tuple(kernel["ISA"]), cxx)
    ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])

    kwa = KernelWriterAssembly(assembler, DebugConfig())
    kernel.duplicate = False
    kernel["BaseName"] = getKernelFileBase(False, kernel)
    res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
    assert res.err == 0, f"kernel emit failed err={res.err}"
    src = res.src
    if isinstance(src, (bytes, bytearray)):
        src = src.decode(errors="replace")
    return kernel, src


# ---------------------------------------------------------------------------
# Validator (no toolchain needed beyond Solution construction)
# ---------------------------------------------------------------------------
def test_block_dequant_solution_is_valid(toolchain):
    # The last two have DepthU < G, where one group spans 2 and 4 K iterations.
    for blockSize, depthU in ((32, 64), (32, 128), (128, 128), (128, 64), (128, 32)):
        sol = _solution(toolchain, blockSize=blockSize, depthU=depthU)
        assert sol.get("Valid") is True, f"G={blockSize} DepthU={depthU} should be valid"


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"blockSize": 96}, "ScaleBlockSizeA"),          # unsupported group size
        # Neither divides the other, so one iteration would straddle two groups.
        ({"blockSize": 128, "depthU": 96}, "divide one another"),
        ({"glvwA": 16}, "one dword per A load"),         # >1 dword per load
        ({"problemType": {"TransposeA": False}}, "TransposeA"),
        ({"ConvertAfterDS": True}, "ConvertAfterDS"),
        ({"GlobalSplitU": 2}, "GlobalSplitU"),
        ({"StaggerU": 32}, "StaggerU"),
        # SIA=1 and PGR>=2 are each fine alone; only together do they
        # desynchronise the scale from the A data it scales.
        ({"PrefetchGlobalRead": 2, "ScheduleIterAlg": 1}, "ScheduleIterAlg"),
    ],
)
def test_block_dequant_rejects_unsupported_shapes(toolchain, kw, reason):
    sol = _solution(toolchain, **kw)
    assert sol.get("Valid") is not True, f"expected a reject mentioning {reason!r}"


@pytest.mark.parametrize("kw", [{"PrefetchGlobalRead": 2}, {"ScheduleIterAlg": 1}])
def test_prefetch_and_schedule_are_each_fine_alone(toolchain, kw):
    """The reject above is on the *combination*, so pin that neither setting is
    rejected by itself -- otherwise the gate would silently cost every solution
    a scheduling knob it is entitled to."""
    assert _solution(toolchain, **kw).get("Valid") is True


def test_iters_per_group_is_the_counter_period():
    """The period blockScaleAIncrement counts to. An off-by-one here scales A by
    a neighbouring K group, which is wrong by a factor rather than a little."""
    from Tensile.SolutionStructs.Problem import blockDequantItersPerGroupA

    pt = {"UseScaleAB": "Block", "ScaleBlockSizeA": 128}
    assert blockDequantItersPerGroupA(pt, 128) == 1
    assert blockDequantItersPerGroupA(pt, 64) == 2
    assert blockDequantItersPerGroupA(pt, 32) == 4
    # DepthU > G still advances every iteration; the period never goes below 1.
    assert blockDequantItersPerGroupA(pt, 256) == 1
    # Block dequantize off entirely: no scale tensor, so no counter.
    assert blockDequantItersPerGroupA({"UseScaleAB": "", "ScaleBlockSizeA": 0}, 64) == 1


def test_scale_block_size_without_block_mode_is_rejected(toolchain):
    sol = _solution(toolchain, problemType={"UseScaleAB": ""})
    assert sol.get("Valid") is not True


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------
def test_emits_one_scale_load_per_a_load(toolchain):
    kernel, src = _emit(toolchain)
    numLoads = kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"]

    aLoads = re.findall(r"buffer_load_b32 v\[vgprG2LA\+\d+\].*sgprSrdA", src)
    scaleLoads = re.findall(
        r"buffer_load_d16_b16 v\[vgprG2LScaleA\+(\d+)\].*sgprSrdScaleA", src)
    # Both the prefetch and the main loop issue a full set.
    assert len(aLoads) > 0
    assert len(scaleLoads) > 0
    assert len(scaleLoads) == len(aLoads), "one scale load per A load"
    # Every load index in a set is distinct and covers 0..numLoads-1.
    assert set(int(i) for i in scaleLoads) == set(range(numLoads))


def test_emits_dequantize_per_local_write(toolchain):
    kernel, src = _emit(toolchain)
    numLoads = kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"]

    # One saved source dword, 8 sign-extracts and 4 packs per A local write.
    saves = re.findall(r"w4a16: save packed int4 dword", src)
    bfes = re.findall(r"v_bfe_i32 .*w4a16: sign-extend int4", src)
    # gfx950/gfx12.5 pack with v_cvt_pk_bf16_f32; everywhere else the pack is
    # open-coded and ends in v_pack_b32_f16. Match on the comment tag so this
    # counts packs on either path.
    packs = re.findall(r"(?:v_cvt_pk_bf16_f32|v_pack_b32_f16) .*w4a16: pack 2 bf16", src)
    assert saves, "no dequantize block emitted"
    assert len(bfes) == 8 * len(saves)
    assert len(packs) == 4 * len(saves)
    assert len(saves) % numLoads == 0

    # The scale must be converted per load, not hoisted to a single value.
    cvts = re.findall(r"v_lshlrev_b32 \S+, 16, v\[vgprG2LScaleA\+(\d+)\]", src)
    assert set(int(i) for i in cvts) == set(range(numLoads))


def test_dequantize_reads_source_before_overwriting_it(toolchain):
    """The bf16 output overlaps the packed int4 source inside each G2L block, so
    the source dword must be copied to a temp first."""
    _, src = _emit(toolchain)
    for m in re.finditer(
        r"v_mov_b32 (v\d+), v\[vgprG2LA\+(\d+)\+0\]\s+// w4a16: save packed int4", src
    ):
        tmp = m.group(1)
        # The first bfe after the save must read the temp, not G2LA.
        rest = src[m.end():m.end() + 400]
        firstBfe = re.search(r"v_bfe_i32 \S+, (\S+),", rest)
        assert firstBfe, "no bfe after the save"
        assert firstBfe.group(1) == tmp, (
            f"dequantize reads {firstBfe.group(1)} instead of the saved copy {tmp}")


def test_scale_srd_increment_matches_group_size(toolchain):
    """DepthU >= G: a literal advance of DepthU/G scale elements every iteration."""
    for blockSize, depthU in ((32, 128), (128, 128), (32, 64)):
        _, src = _emit(toolchain, blockSize=blockSize, depthU=depthU)
        want = (depthU // blockSize) * 2  # bf16 scale elements -> bytes
        incs = re.findall(
            r"s_add_u32 s\[sgprSrdScaleA\+0\], s\[sgprSrdScaleA\+0\], (\S+)\s+// scaleA SRD",
            src)
        assert incs, f"no scale SRD increment for G={blockSize} DepthU={depthU}"
        for got in incs:
            assert int(got, 0) == want, (
                f"G={blockSize} DepthU={depthU}: increment {got}, want {want}")
        # The limit must move with the base or edge tiles would read past the end.
        assert re.search(
            r"s_sub_u32 s\[sgprSrdScaleA\+2\], s\[sgprSrdScaleA\+2\], (\S+)\s+// scaleA limit",
            src)


@pytest.mark.parametrize("depthU,itersPerGroup", [(64, 2), (32, 4)])
def test_scale_srd_advances_every_nth_iteration(toolchain, depthU, itersPerGroup):
    """DepthU < G: the group outlives the iteration, so the pointer must step
    once every itersPerGroup iterations instead of every one.

    Checked on the emitted instructions rather than on results, because a
    pointer that advanced every iteration would still produce plausible numbers
    -- just scaled by the wrong group -- on any input whose scales do not vary
    much along K.
    """
    _, src = _emit(toolchain, blockSize=128, depthU=depthU)

    # The counter wraps with an AND, whose mask fixes the period.
    mask = re.findall(
        r"s_and_b32 (\S+), s\[sgprScaleAKCnt\], (\S+)\s+// scaleA: SCC", src)
    assert mask, f"no scale counter mask emitted for DepthU={depthU}"
    for _dst, got in mask:
        assert int(got, 0) == itersPerGroup - 1, (
            f"DepthU={depthU}: mask {got}, want {itersPerGroup - 1}")

    # Counter starts at zero, so the increment closing iteration 0 does nothing.
    assert re.search(r"s_mov_b32 s\[sgprScaleAKCnt\], 0\b", src), "counter not zeroed"

    # One step is one group: 2 bytes of bf16 scale, whatever DepthU is.
    sel = re.findall(r"s_cselect_b32 (\S+), 0, (\S+)\s+// scaleA: advance", src)
    assert sel, "no conditional scale advance"
    for _dst, got in sel:
        assert int(got, 0) == 2, f"DepthU={depthU}: step {got}, want 2 bytes"

    # The advance must be register-sourced now; a literal would mean the
    # every-iteration path leaked through.
    for got in re.findall(
            r"s_add_u32 s\[sgprSrdScaleA\+0\], s\[sgprSrdScaleA\+0\], (\S+)\s+// scaleA SRD",
            src):
        assert got.startswith("s"), f"DepthU={depthU}: unconditional advance by {got}"


FP16_UNSIGNED = {
    "DataType": "H", "MacDataTypeA": "H", "DestDataType": "H",
    "Int4EncodingA": "UnsignedBias8",
}


def test_fp16_unsigned_dequantizes_in_packed_fp16(toolchain):
    """An fp16 MAC type plus unsigned int4 nibbles takes the packed lowering:
    half the VALU of the f32 one, with permutations to restore sequential order."""
    _, whole = _emit(toolchain, blockSize=128, depthU=64, zeroPoint=True,
                     problemType=dict(FP16_UNSIGNED))
    # Scope to the dequantize: the epilogue converts its f32 accumulators to
    # fp16 too, and that is not what this is about.
    src = whole[whole.index("w4a16: save packed int4 dword 0"):whole.index("ds_store")]

    # One magic per nibble position, each with the exponent that makes its
    # nibble's lowest bit worth 1.0: 1024.0 at mantissa bits 0-3, 64.0 at 4-7.
    # They sit in SGPRs because the fused mask+OR has spent its one literal.
    for magic in ("0x64006400", "0x54005400"):
        assert re.search(r"s_mov_b32 s\[sgprScaleAPkMagic\+\d\], " + magic, whole), \
            "fp16 magic constant %s missing" % magic
    for mask in ("0xf000f", "0xf000f0"):
        assert re.search(
            r"v_and_or_b32 \S+, \S+, " + mask + r", s\[sgprScaleAPkMagic\+\d\]", src), \
            "no fused mask+OR for nibble mask %s" % mask
    assert re.search(r"v_pk_add_f16 ", src), "no packed subtract"
    assert re.search(r"v_pk_mul_f16 ", src), "no packed scale"

    # The f32 lowering's fingerprints must all be gone from the dequantize.
    assert "0x43004300" not in src, "still using the bf16 magic constant"
    assert not re.search(r"v_or_b32 ", src), "mask and OR not fused"
    assert not re.search(r"v_cvt_f16_f32 ", src), "still converting f32 -> fp16"

    # A fused v_pk_fma_f16 would need -(1024+z)*s pre-rounded to fp16, an error
    # the same size as the answer. Pin that it is not used here.
    assert not re.search(r"v_pk_fma_f16 ", src), "fused FMA reintroduces the cancellation"


def test_fp16_unsigned_lifts_nibbles_in_place(toolchain):
    """Two of the four nibble pairs are lifted where they already lie, by
    choosing the magic to match the nibble's position rather than shifting it
    to bit 0. The other two need mantissa bits 8-11, which are the exponent, so
    one shift per dword remains -- but only one, not three."""
    _, whole = _emit(toolchain, blockSize=128, depthU=64, zeroPoint=True,
                     problemType=dict(FP16_UNSIGNED))
    body = whole[whole.index("w4a16: save packed int4 dword 0"):whole.index("ds_store")]
    shifts = re.findall(r"v_lshrrev_b32 ", body)
    assert len(shifts) == 1, (
        "expected a single shift per dword, found %d" % len(shifts))


def test_fp16_unsigned_packed_is_half_the_valu(toolchain):
    """The whole point of the packed lowering. Counted rather than asserted in
    prose so a later change that quietly reinstates the f32 path is caught."""
    _, pk = _emit(toolchain, blockSize=128, depthU=64, zeroPoint=True,
                  problemType=dict(FP16_UNSIGNED))
    bf = dict(FP16_UNSIGNED)
    bf.update({"DataType": "B", "MacDataTypeA": "B", "DestDataType": "B"})
    _, f32 = _emit(toolchain, blockSize=128, depthU=64, zeroPoint=True,
                   problemType=bf)

    def valuBetween(src, first, last):
        body = src[src.index(first):src.index(last)]
        return len([l for l in body.splitlines() if l.startswith("v_")])

    pkN = valuBetween(pk, "w4a16: save packed int4 dword 0", "ds_store")
    f32N = valuBetween(f32, "w4a16: save packed int4 dword 0", "ds_store")
    assert pkN * 2 <= f32N, (
        "packed lowering is %d VALU against the f32 path's %d; expected at most half"
        % (pkN, f32N))


def test_fp16_scale_is_masked_before_duplication(toolchain):
    """The scale arrives via buffer_load_d16_b16, which leaves the upper half of
    the register at whatever it held. Duplicating without masking first puts
    that garbage in the low lane and the kernel returns NaN."""
    _, src = _emit(toolchain, blockSize=128, depthU=64, zeroPoint=True,
                   problemType=dict(FP16_UNSIGNED))
    dup = re.search(r"v_and_b32 (\S+), 0xffff, \S+.*\n.*v_lshl_or_b32 \1, \1, 16, \1", src)
    assert dup, "scale duplicated without masking the stale d16 high half"


def test_scale_offset_uses_group_shift(toolchain):
    for blockSize in (32, 128):
        _, src = _emit(toolchain, blockSize=blockSize, depthU=128)
        shift = blockSize.bit_length() - 1
        assert re.search(
            rf"v_lshrrev_b32 \S+, {shift}, \S+\s+// scaleA: kGroup = k/{blockSize}", src)


# ---------------------------------------------------------------------------
# Host / client plumbing
# ---------------------------------------------------------------------------
def test_problem_type_carries_block_scale_to_predicates(toolchain):
    """The library-logic ProblemType must round-trip the group size, and emit it
    as a predicate so only matching kernels are selected. The scale type is not
    among them: it follows B, which already has a predicate of its own."""
    from Tensile.Contractions import ProblemType as CProblemType

    sol = _solution(toolchain)
    assert sol.get("Valid") is True
    pt = CProblemType.FromOriginalState(sol["ProblemType"].state)

    assert pt.useScaleAB == "Block"
    assert pt.scaleBlockSizeA == 32
    assert not hasattr(pt, "scaleTypeA")
    # A is int4 in memory but bf16 at the MAC.
    assert pt.aType.toName() == "Int4"
    assert pt.computeInputTypeA.toName() == "BFloat16"

    states = [p.state() for p in pt.predicates(includeBatch=True, includeType=True)]
    byType = {s["type"]: s["value"] for s in states if isinstance(s, dict) and "type" in s}
    assert byType.get("UseScaleAB") == "Block"
    assert byType.get("ScaleBlockSizeA") == 32
    # No DataTypeScaleA predicate: TypesEqual already pins B, and the scale
    # follows it.
    assert "DataTypeScaleA" not in byType
    assert "TypesEqual" in byType


def test_block_scale_predicates_absent_when_mode_off(toolchain):
    """A plain bf16 solution must not carry the w4a16 predicates, or it would
    stop matching ordinary problems."""
    from Tensile.Contractions import ProblemType as CProblemType

    sol = _solution(
        toolchain,
        problemType={
            "UseScaleAB": "",
            "ScaleBlockSizeA": 0,
            "DataTypeA": "B",
            "MacDataTypeA": "B",
        },
    )
    assert sol.get("Valid") is True
    pt = CProblemType.FromOriginalState(sol["ProblemType"].state)
    states = [p.state() for p in pt.predicates(includeBatch=True, includeType=True)]
    types = {s["type"] for s in states if isinstance(s, dict) and "type" in s}
    assert "ScaleBlockSizeA" not in types
    assert "DataTypeScaleA" not in types


def test_client_config_carries_block_scale(toolchain, tmp_path):
    """ClientWriter must pass the group size and scale type to the benchmark
    client, otherwise its CPU reference cannot apply the scale."""
    from Tensile.ClientWriter import writeClientConfigIni  # noqa: F401  (import check)
    import Tensile.ClientWriter as CW
    import inspect

    src = inspect.getsource(CW)
    assert "param('scale-a-block', problemType.scaleBlockSizeA)" in src
    assert "param('scale-a-type', problemType.bType.toName())" in src


def test_no_vgpr_aliasing_between_scale_and_other_state(toolchain):
    """GlobalReadOffsetScaleA / G2LScaleA are carved out of the static VGPR
    layout; make sure they do not overlap A/B state or each other."""
    kernel, src = _emit(toolchain)
    numLoads = kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"]

    sets = dict(re.findall(r"^\.set (vgpr\w+), (\d+)$", src, re.M))
    for name in ("vgprGlobalReadOffsetScaleA", "vgprG2LScaleA"):
        assert name in sets, f"{name} not declared"

    ranges = {
        "GlobalReadOffsetA": (int(sets["vgprGlobalReadOffsetA"]), numLoads),
        "GlobalReadOffsetB": (int(sets["vgprGlobalReadOffsetB"]),
                              kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"]),
        "GlobalReadOffsetScaleA": (int(sets["vgprGlobalReadOffsetScaleA"]), numLoads),
        "G2LScaleA": (int(sets["vgprG2LScaleA"]), numLoads),
    }
    occupied = {}
    for name, (start, count) in ranges.items():
        for v in range(start, start + count):
            assert v not in occupied, f"vgpr {v} shared by {name} and {occupied[v]}"
            occupied[v] = name


def test_no_vgpr_aliasing_for_zero_point_state(toolchain):
    """The zero-point offset/data registers are extra static allocations; make
    sure they do not overlap the scale registers or A/B state."""
    kernel, src = _emit(toolchain, zeroPoint=True)
    numLoads = kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"]

    sets = dict(re.findall(r"^\.set (vgpr\w+), (\d+)$", src, re.M))
    for name in ("vgprGlobalReadOffsetScaleZeroA", "vgprG2LScaleZeroA"):
        assert name in sets, f"{name} not declared"

    ranges = {
        "GlobalReadOffsetA": (int(sets["vgprGlobalReadOffsetA"]), numLoads),
        "GlobalReadOffsetScaleA": (int(sets["vgprGlobalReadOffsetScaleA"]), numLoads),
        "G2LScaleA": (int(sets["vgprG2LScaleA"]), numLoads),
        "G2LScaleZeroA": (int(sets["vgprG2LScaleZeroA"]), numLoads),
        "GlobalReadOffsetScaleZeroA": (int(sets["vgprGlobalReadOffsetScaleZeroA"]), numLoads),
        "GlobalReadByteOffsetScaleZeroA": (int(sets["vgprGlobalReadByteOffsetScaleZeroA"]), numLoads),
    }
    occupied = {}
    for name, (start, count) in ranges.items():
        for v in range(start, start + count):
            assert v not in occupied, f"vgpr {v} shared by {name} and {occupied[v]}"
            occupied[v] = name


# ---------------------------------------------------------------------------
# Asymmetric: per-group zero-points
# ---------------------------------------------------------------------------
def test_zero_point_solution_is_valid(toolchain):
    # The last two have DepthU < G, exercising the zero-point pointer's
    # every-Nth-iteration walk as well as the scale pointer's.
    for blockSize, depthU in ((32, 64), (32, 128), (128, 128), (128, 64), (128, 32)):
        sol = _solution(toolchain, blockSize=blockSize, depthU=depthU, zeroPoint=True)
        assert sol.get("Valid") is True, f"G={blockSize} DepthU={depthU} should be valid"


@pytest.mark.parametrize(
    "kw",
    [
        # One group per iteration: the M-major layout could not express this
        # (half-a-byte advance, per-row nibble flip), the K-group-major one can.
        {"blockSize": 128, "depthU": 128},
        {"blockSize": 32, "depthU": 32},
    ],
)
def test_zero_point_accepts_odd_groups_per_iteration(toolchain, kw):
    sol = _solution(toolchain, zeroPoint=True, **kw)
    assert sol.get("Valid") is True, (
        "odd DepthU/ScaleBlockSizeA must be legal under the K-group-major "
        "zero-point layout (got a reject for %r)" % kw)


# ---------------------------------------------------------------------------
# Weight encodings (Int4EncodingA)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("encoding", ["Signed", "UnsignedBias8"])
@pytest.mark.parametrize("zeroPoint", [False, True])
def test_int4_encoding_solutions_are_valid(toolchain, encoding, zeroPoint):
    sol = _solution(toolchain, zeroPoint=zeroPoint,
                    problemType={"Int4EncodingA": encoding})
    assert sol.get("Valid") is True, f"{encoding} zp={zeroPoint} should be valid"


def test_int4_encoding_rejects_unknown_name(toolchain):
    sol = _solution(toolchain, problemType={"Int4EncodingA": "Nonsense"})
    assert sol.get("Valid") is not True


def test_signed_encoding_sign_extends_and_needs_no_bias(toolchain):
    """The default encoding reads two's-complement nibbles and, with no
    zero-point, scales with a plain multiply."""
    _, src = _emit(toolchain)
    assert re.search(r"v_bfe_i32 .*w4a16: sign-extend int4", src)
    assert not re.search(r"v_bfe_u32 .*w4a16: zero-extend int4", src)
    assert re.search(r"v_mul_f32 .*w4a16: dequantize", src)


def test_unsigned_bias8_zero_extends_and_folds_the_bias(toolchain):
    """UnsignedBias8 reads raw nibbles and turns the symmetric multiply into an
    FMA against a precomputed -8*s."""
    _, src = _emit(toolchain, problemType={"Int4EncodingA": "UnsignedBias8"})
    assert re.search(r"v_bfe_u32 .*w4a16: zero-extend int4", src)
    assert not re.search(r"v_bfe_i32 .*w4a16: sign-extend int4", src)
    # 8.0f, the implicit zero-point, folded into the per-load bias.
    assert re.search(r"v_mul_f32 \S+, 0x41000000", src), "expected an 8.0f * s bias"
    assert re.search(r"v_fma_f32 .*w4a16: q\*s - z\*s", src)


# ---------------------------------------------------------------------------
# fp16 MAC type
# ---------------------------------------------------------------------------
_FP16 = {"DataType": "H", "MacDataTypeA": "H", "DestDataType": "H", "DataTypeScaleA": "H"}


@pytest.mark.parametrize("encoding", ["Signed", "UnsignedBias8"])
def test_fp16_mac_type_is_valid(toolchain, encoding):
    pt = dict(_FP16)
    pt["Int4EncodingA"] = encoding
    sol = _solution(toolchain, problemType=pt)
    assert sol.get("Valid") is True, f"fp16 + {encoding} should be valid"


def test_fp16_uses_the_native_convert_not_the_open_coded_rounding(toolchain):
    """bf16 has no f32->bf16 convert on gfx11 so the pack is open-coded; fp16
    does, so the same dequantize should collapse to cvt/cvt/pack."""
    _, src = _emit(toolchain, problemType=dict(_FP16))
    assert re.search(r"v_cvt_f16_f32 .*w4a16: f32 -> fp16", src)
    assert re.search(r"w4a16: pack 2 fp16", src)
    assert re.search(r"v_cvt_f32_f16 .*scaleA: fp16 -> f32", src)
    # None of the bf16 rounding emulation should survive.
    assert not re.search(r"w4a16: lsb of the bf16 mantissa", src)
    assert not re.search(r"w4a16: check Nan", src)


def test_fp16_dequantize_is_cheaper_than_bf16(toolchain):
    """The whole reason fp16 is worth having on gfx11: a real convert replaces
    the 5-instruction-per-value bf16 rounding emulation."""
    tagged = re.compile(r"^\s*v_\w+\b.*//\s*(w4a16|scaleZeroA|scaleA):")

    def count(pt):
        _, src = _emit(toolchain, problemType=pt)
        return sum(1 for line in src.splitlines() if tagged.match(line))

    fp16, bf16 = count(dict(_FP16)), count({})
    # Measured 1.63-1.85x on gfx1151 depending on encoding / zero-point; assert
    # a floor well below that so ordinary scheduling churn does not trip it.
    assert fp16 * 3 < bf16 * 2, f"fp16 should be >1.5x cheaper than bf16 ({fp16} vs {bf16})"


def test_scale_type_follows_b(toolchain):
    """The scale has no parameter of its own: it is read with B's type.

    It is widened by the MAC type's own conversion, so a bf16 scale under an fp16
    MAC type would be decoded with the wrong exponent bias. Deriving it removes
    the chance of that disagreement rather than validating against it.

    (A MacDataTypeA that disagrees with DataType is caught harder still, by the
    typed-GEMM table in _checkIfSupportedGEMMType, so it never reaches here.)"""
    _, bf16 = _emit(toolchain)
    _, fp16 = _emit(toolchain, problemType=dict(_FP16))
    assert re.search(r"scaleA: bf16 -> f32", bf16)
    assert not re.search(r"scaleA: fp16 -> f32", bf16)
    assert re.search(r"scaleA: fp16 -> f32", fp16)
    assert not re.search(r"scaleA: bf16 -> f32", fp16)


def test_mismatched_mac_type_and_b_is_rejected(toolchain):
    pt = {"MacDataTypeA": "H"}  # DataType stays bf16
    with pytest.raises(Exception, match="not supported yet"):
        _solution(toolchain, problemType=pt)




def test_zero_point_nibble_comes_from_the_row_not_the_scale_offset(toolchain):
    """Packing the zero-points along M is what removes the DepthU/G parity rule.

    Its observable signature is that the nibble select reads bit 0 of the
    zero-point offset register (a pure function of the row) rather than the
    scale offset, so it is loop-invariant.
    """
    _, src = _emit(toolchain, blockSize=128, depthU=128, zeroPoint=True)
    assert re.search(r"v_and_b32 \S+, 1, v\[vgprGlobalReadOffsetScaleZeroA\+\d+\]", src), \
        "nibble must be derived from the zero-point offset register"
    assert not re.search(r"v_and_b32 \S+, 2, v\[vgprGlobalReadOffsetScaleA\+\d+\]", src), \
        "nibble must no longer be derived from the scale offset"
    # Packing along M makes one K-group exactly one byte, so a single group per
    # iteration advances the zero-point SRD by a literal 1 byte.
    assert re.search(r"s_add_u32 s\[sgprSrdScaleZeroA\+0\], s\[sgprSrdScaleZeroA\+0\], 0x1\b",
                     src), "one group per iteration should advance the SRD by 1 byte"


def test_zero_point_without_block_mode_is_rejected(toolchain):
    sol = _solution(toolchain, zeroPoint=True,
                    problemType={"UseScaleAB": "", "ScaleBlockSizeA": 0})
    assert sol.get("Valid") is not True


def test_zero_point_carried_to_predicates(toolchain):
    from Tensile.Contractions import ProblemType as CProblemType

    sol = _solution(toolchain, zeroPoint=True)
    pt = CProblemType.FromOriginalState(sol["ProblemType"].state)
    assert pt.scaleZeroPointA is True
    states = [p.state() for p in pt.predicates(includeBatch=True, includeType=True)]
    byType = {s["type"]: s["value"] for s in states if isinstance(s, dict) and "type" in s}
    assert byType.get("ScaleZeroPointA") is True

    # ...and the symmetric solution must advertise the opposite, or an
    # asymmetric problem could select a symmetric kernel.
    symPt = CProblemType.FromOriginalState(_solution(toolchain)["ProblemType"].state)
    symStates = [p.state() for p in symPt.predicates(includeBatch=True, includeType=True)]
    symByType = {s["type"]: s["value"] for s in symStates
                 if isinstance(s, dict) and "type" in s}
    assert symByType.get("ScaleZeroPointA") is False


def test_zero_point_kernel_name_differs(toolchain):
    """The two variants must not collide in the kernel cache."""
    sym = _solution(toolchain)["ProblemType"].__str__()
    asym = _solution(toolchain, zeroPoint=True)["ProblemType"].__str__()
    assert sym != asym
    assert "ZP" in asym and "ZP" not in sym


def test_emits_one_zero_point_load_per_a_load(toolchain):
    kernel, src = _emit(toolchain, zeroPoint=True)
    numLoads = kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"]
    loads = re.findall(r"buffer_load_[a-z0-9_]*\s+v\[?vgprG2LScaleZeroA", src)
    assert len(loads) >= numLoads, (
        f"expected >= {numLoads} zero-point loads, found {len(loads)}")


def test_zero_point_uses_fma_not_mul(toolchain):
    """Folding -z*s into the FMA is what keeps the per-element cost identical to
    the symmetric path; a regression to mul+sub would double the ALU work."""
    _, sym = _emit(toolchain)
    _, asym = _emit(toolchain, zeroPoint=True)

    assert "w4a16: q*s - z*s" in asym
    assert asym.count("v_fma_f32") > sym.count("v_fma_f32")
    # Same number of per-element dequantize ops in both.
    assert asym.count("w4a16: sign-extend int4") == sym.count("w4a16: sign-extend int4")


def test_zero_point_srd_advances_with_the_k_loop(toolchain):
    """The bug this pins: without a SrdScaleZeroA increment, K > DepthU reads
    the first iteration's zero-points for every iteration."""
    _, src = _emit(toolchain, zeroPoint=True)
    assert "scaleZeroA SRD += inc(lower)" in src
    assert "scaleZeroA limit -= inc" in src


def test_no_zero_point_state_when_symmetric(toolchain):
    """A symmetric kernel must not allocate or reference any zero-point state."""
    _, src = _emit(toolchain)
    for sym in ("G2LScaleZeroA", "SrdScaleZeroA", "AddressScaleZeroA"):
        assert sym not in src, f"{sym} leaked into a symmetric kernel"


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_zero_point_byte_offsets_are_hoisted(toolchain, block_size):
    kernel, src = _emit(toolchain, blockSize=block_size, depthU=64, zeroPoint=True,
                        problemType=dict(FP16_UNSIGNED))
    prologue, loop = src.split("label_LoopBeginL:", 1)
    loop = loop.split("label_LoopEndL:", 1)[0]
    count = kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"]
    for index in range(count):
        offset = f"v[vgprGlobalReadByteOffsetScaleZeroA+{index}]"
        assert f"v_mov_b32 {offset}," in prologue
        assert f"v_mov_b32 {offset}," not in loop
        assert f"buffer_load_d16_u8 v[vgprG2LScaleZeroA+{index}], {offset}," in loop
    assert "drop the nibble bit" not in src


@pytest.mark.parametrize("zero_point", [False, True])
def test_unsigned_fp16_restores_consecutive_pairs(toolchain, zero_point):
    _, src = _emit(toolchain, blockSize=32, depthU=64, zeroPoint=zero_point,
                   problemType=dict(FP16_UNSIGNED))
    body = src[src.index("w4a16: save packed int4 dword 0"):src.index("ds_store")]
    assert body.count("v_perm_b32 ") == 4
    assert "0x5040100" in src and "0x7060302" in src
    # Each four-instruction stage completes before the next begins.
    assert body.rindex("v_and_or_b32") < body.index("v_pk_add_f16")
    assert body.rindex("v_pk_add_f16") < body.index("v_pk_mul_f16")
    assert body.rindex("v_pk_mul_f16") < body.index("v_perm_b32")
