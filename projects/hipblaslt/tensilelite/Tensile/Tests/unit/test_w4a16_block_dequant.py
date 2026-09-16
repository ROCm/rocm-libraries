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
        "DataTypeScaleA": "B",
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
    for blockSize, depthU in ((32, 64), (32, 128), (128, 128)):
        sol = _solution(toolchain, blockSize=blockSize, depthU=depthU)
        assert sol.get("Valid") is True, f"G={blockSize} DepthU={depthU} should be valid"


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"blockSize": 96}, "ScaleBlockSizeA"),          # unsupported group size
        ({"blockSize": 128, "depthU": 64}, "DepthU"),    # DepthU % G != 0
        ({"glvwA": 16}, "one dword per A load"),         # >1 dword per load
        ({"problemType": {"TransposeA": False}}, "TransposeA"),
        ({"ConvertAfterDS": True}, "ConvertAfterDS"),
        ({"GlobalSplitU": 2}, "GlobalSplitU"),
        ({"StaggerU": 32}, "StaggerU"),
    ],
)
def test_block_dequant_rejects_unsupported_shapes(toolchain, kw, reason):
    sol = _solution(toolchain, **kw)
    assert sol.get("Valid") is not True, f"expected a reject mentioning {reason!r}"


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
    for blockSize, depthU in ((32, 64), (32, 128), (128, 128)):
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
@pytest.mark.parametrize("encoding", ["Signed", "UnsignedBias8", "UnsignedBias8ExLlama"])
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


@pytest.mark.parametrize("encoding", ["Signed", "UnsignedBias8", "UnsignedBias8ExLlama"])
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


def test_exllama_uses_the_magic_pair_extract(toolchain):
    """The ExLlama shuffle puts elements 2k and 2k+1 in the two halves of the
    dword, so one mask+or lifts both as bf16 128+q and widening to f32 is a
    shift -- no v_cvt and no per-element bit-field extract."""
    _, src = _emit(toolchain, problemType={"Int4EncodingA": "UnsignedBias8ExLlama"})
    assert re.search(r"v_and_b32 \S+, 0xf000f, ", src), "expected the 0x000F000F pair mask"
    assert re.search(r"v_or_b32 \S+, 0x43004300, ", src), "expected the bf16 128+q magic"
    assert not re.search(r"v_cvt_f32_i32 .*w4a16: int4 -> f32", src), \
        "the magic path should not need an int->float convert"
    assert not re.search(r"w4a16: (sign|zero)-extend int4", src), \
        "the magic path should not need per-element bfe"
    # 136 = the magic 128 plus the implicit zero-point 8.
    assert re.search(r"v_mul_f32 \S+, 0x43080000", src), "expected a 136.0f * s bias"


def test_exllama_does_not_use_dot2_bf16(toolchain):
    """v_dot2_bf16_bf16 would collapse the dequantize to two instructions per
    element, but on gfx1151 it truncates toward zero and its accumulate is
    ~2^-24 asymmetric. Because the magic bias is a deliberate cancellation
    ((128+q)*s - 136*s), that lands a full ulp low on every positive weight --
    a signed bias that accumulates over K. Pin the f32 lowering so the
    optimisation is not silently reintroduced."""
    _, src = _emit(toolchain, problemType={"Int4EncodingA": "UnsignedBias8ExLlama"})
    assert "v_dot2_bf16_bf16" not in src
    assert re.search(r"v_fma_f32 .*w4a16:", src), "the f32 fma lowering should be in use"


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
