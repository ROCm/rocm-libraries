# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DirectToVgprB on a TN bf16 WMMA v1 kernel with 32-byte global loads (AIESW-40348).

Four coupled changes are pinned here.

1. The ``GlobalRead`` MemoryInstruction table stopped at ``b192`` (24 bytes), so
   a 32-byte vector had no entry. That table -- not ``chooseGlobalRead``, which
   happily emulates 32 and 64 bytes -- is what sizes the tiled global read:
   ``nrcvpi`` comes from ``globalReadInstruction.totalWidth``, and both the G2L
   destination stride and ``numVgprG2L`` follow from it. Without a matching
   entry the emitter split the vector into two instructions that each still
   loaded the full 32 bytes, so it issued twice the loads at half the register
   stride and wrote past the end of the G2L buffer (over ``vgprSerial``, which
   faulted the GPU). A ``b256`` entry makes it one instruction again.

2. ``Solution.py``'s ``GlobalReadVectorWidth * numBytes`` bound tracks the
   widest entry in that table, so it moves 24 -> 32 with it.

3. ``numVgprG2L`` for DirectToVgpr was sized from the *unique* tile volume. A
   WMMA v1 operand is replicated across the half-waves, so an LDS-staged kernel
   gets the replication out of its ds_read but DirectToVgpr has to load it
   twice; ``Solution.dtvOperandDuplication`` accounts for that.

4. The WMMA v1 tail-loop K-masking in ``KernelWriterAssembly.mfmaIter`` built
   its operand register names as literal ``"Valu<tc>_X<m>_I<iui>+..."`` strings
   instead of going through ``generateSrcStrForMFMA`` like the WMMA
   instructions themselves. With ``DirectToVgpr<tc>`` the operand lives in the
   ``G2L<tc>`` buffer and no ``Valu<tc>`` symbol is emitted at all, so those
   references were to undefined symbols and the kernel failed to assemble
   ("expected absolute expression"). Only WMMA v1 took that branch -- MFMA and
   WMMA v2/v3 already used the helper -- which is why this surfaced on gfx11.

The codegen tests below emit a real kernel and check both that every ``vgpr*``
symbol it references is defined by a ``.set`` (what the assembler enforces) and
that no G2L reference runs past the buffer that was allocated for it (what the
page fault came from). Numerical correctness is covered on hardware by
``Tests/common/gemm/gfx11/dtvb_grvw32_bf16_tn_gfx11.yaml``.
"""

import contextlib
import copy
import io
import re
import shutil

import pytest

pytestmark = pytest.mark.unit


GRVW_REJECT_B = "GRVWB * DataTypeB.numBytes() > 32"

# gfx1151 (RDNA 3.5) is the WMMA v1 target this was found on; gfx1100 takes the
# same code path.
_GFX = "gfx1151"

# ``.set vgprFoo, ...`` definitions and ``vgprFoo`` references. Comments are
# stripped before scanning: they contain prose like "// vgprs" that is not a
# symbol reference.
_SET_RE = re.compile(r"^\.set\s+(vgpr\w+)", re.M)
_REF_RE = re.compile(r"\bvgpr[A-Za-z]\w*")


# ---------------------------------------------------------------------------
# Toolchain fixtures (CPU only: emit is exercised, assembling is not).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def isa_info_map():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa(_GFX)
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip(f"amdclang++ in this environment does not support {_GFX}")
    if not iim[isa].asmCaps.get("HasWMMA_V1"):
        pytest.skip(f"{_GFX} caps report no HasWMMA_V1")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _globals(isa_info_map):
    """Assign the process-global parameters for this arch, restore afterwards.

    ``assignGlobalParameters`` / ``Solution.__init__`` mutate ``globalParameters``,
    ``validParameters`` and ``defaultSolution`` in place; unrestored that leaks
    into unrelated tests.
    """
    from Tensile.Common.GlobalParameters import (
        globalParameters,
        assignGlobalParameters,
        defaultSolution,
    )
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    assignGlobalParameters({}, isa_info_map)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


# ---------------------------------------------------------------------------
# Base solution: TN bf16 WMMA 16x16x16, DepthU 32, DirectToVgprB, GRVWB=16
# (32-byte global loads). Each test flips exactly one knob.
# ---------------------------------------------------------------------------
def _make_params(isa_info_map, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa(_GFX)
    # [M, N, K, B, ?, MIWaveTile0, MIWaveTile1, WaveGroup0, WaveGroup1]
    mi = [16, 16, 16, 1, 1, 2, 2, 1, 2]
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,   # TN: TLUA == TLUB == False
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
    }
    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [32, 2, 1],
        "WavefrontSize": 32,
        "DepthU": 32,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,      # DirectToVgpr + TLU=False needs PLR >= 1
        "ScheduleIterAlg": 3,        # DirectToVgpr needs SIA >= 3
        "StaggerU": 8,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "InnerUnroll": 1,
        "TransposeLDS": 1,           # DirectToVgprB needs UnrollMajorLDSB != TLUB
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": 2,
        "VectorWidthB": 2,
        "StoreVectorWidth": 2,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 16,  # * 2 bytes == 32-byte global load
        "LocalReadVectorWidth": 16,
        "SourceSwap": True,
        "MIArchVgpr": True,
        "ExpandPointerSwap": True,
        "DirectToVgprA": False,
        "DirectToVgprB": True,
        "WorkGroupMapping": 1,
        "ClusterLocalRead": 0,
    }
    params.update(overrides)
    params.update(
        matrixInstructionToMIParameters(
            mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], isa_info_map
        )
    )
    return params


def _derive(isa_info_map, assembler, **overrides):
    """Construct a Solution with reject printing on; return (solution, stdout)."""
    from Tensile.SolutionStructs.Solution import Solution

    params = _make_params(isa_info_map, **overrides)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sol = Solution(params, False, True, False, assembler, isa_info_map)
    return sol, buf.getvalue()


def _emit(isa_info_map, assembler, **overrides):
    """Emit assembly text for the solution described by ``overrides``."""
    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    sol, _ = _derive(isa_info_map, assembler, **overrides)
    assert sol["Valid"], "solution under test was rejected before it could be emitted"

    kernel = generateKernelObjectsFromSolutions([sol])[0]
    ri = rocisa.rocIsa.getInstance()
    ri.init(tuple(kernel["ISA"]), shutil.which("amdclang++"))
    ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
    kernel.duplicate = False
    kernel["BaseName"] = getKernelFileBase(False, kernel)

    res = processKernelSource(
        KernelWriterAssembly(assembler, DebugConfig()),
        ri.getData(),
        ri.getOutputOptions(),
        False,
        kernel,
    )
    assert res.err == 0
    src = res.src
    if isinstance(src, (bytes, bytearray)):
        src = src.decode(errors="replace")
    return src


def _emit_with_writer(isa_info_map, assembler, **overrides):
    """Like ``_emit`` but also hands back the writer, for its register counts."""
    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    sol, _ = _derive(isa_info_map, assembler, **overrides)
    assert sol["Valid"], "solution under test was rejected before it could be emitted"

    kernel = generateKernelObjectsFromSolutions([sol])[0]
    ri = rocisa.rocIsa.getInstance()
    ri.init(tuple(kernel["ISA"]), shutil.which("amdclang++"))
    ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
    kernel.duplicate = False
    kernel["BaseName"] = getKernelFileBase(False, kernel)

    writer = KernelWriterAssembly(assembler, DebugConfig())
    res = processKernelSource(
        writer, ri.getData(), ri.getOutputOptions(), False, kernel
    )
    assert res.err == 0
    src = res.src
    if isinstance(src, (bytes, bytearray)):
        src = src.decode(errors="replace")
    return src, writer


def _undefined_vgpr_symbols(src):
    """vgpr symbols referenced by the code but never defined by a ``.set``."""
    code = "\n".join(line.split("//", 1)[0] for line in src.splitlines())
    return sorted(set(_REF_RE.findall(code)) - set(_SET_RE.findall(src)))


# ``v[vgprG2LB+8+0+2]`` / ``v[vgprG2LB+8:vgprG2LB+8+3]``: capture every offset
# chain hung off the G2LB base symbol so the highest register it reaches can be
# compared against the buffer that was actually allocated.
_G2L_REF_RE = re.compile(r"\bvgprG2L([AB])((?:\+\d+)*)")


def _max_g2l_offset(src, tc):
    code = "\n".join(line.split("//", 1)[0] for line in src.splitlines())
    offsets = [
        sum(int(n) for n in re.findall(r"\d+", m.group(2)))
        for m in _G2L_REF_RE.finditer(code)
        if m.group(1) == tc
    ]
    return max(offsets) if offsets else -1


# ---------------------------------------------------------------------------
# 1. The GRVW rejection bound.
# ---------------------------------------------------------------------------
def test_grvw_32_bytes_accepted_with_dtvb(isa_info_map, assembler, _globals):
    sol, out = _derive(isa_info_map, assembler)
    assert sol["Valid"], out
    assert GRVW_REJECT_B not in out
    assert sol["DirectToVgprB"]
    assert sol["GlobalReadVectorWidthB"] == 16


def test_grvw_32_bytes_accepted_without_dtvb(isa_info_map, assembler, _globals):
    """The 32-byte load is legal on its own, independent of DirectToVgpr."""
    sol, out = _derive(isa_info_map, assembler, DirectToVgprB=False)
    assert sol["Valid"], out
    assert GRVW_REJECT_B not in out


def test_grvw_64_bytes_still_rejected(isa_info_map, assembler, _globals):
    """32 bytes is the bound, not "no bound": 64 bytes must still be rejected."""
    sol, out = _derive(
        isa_info_map, assembler, DirectToVgprB=False, GlobalReadVectorWidthB=32
    )
    assert not sol["Valid"]
    assert GRVW_REJECT_B in out


# ---------------------------------------------------------------------------
# 2. Tail-loop operand naming.
# ---------------------------------------------------------------------------
def test_dtvb_kernel_references_only_defined_vgpr_symbols(
    isa_info_map, assembler, _globals
):
    """The regression itself: 144 references to an undefined ``vgprValuB_X*``."""
    src = _emit(isa_info_map, assembler)
    assert _undefined_vgpr_symbols(src) == []
    # With DirectToVgprB the B operand lives in G2LB and ValuB is never
    # allocated, so nothing may name it -- including the tail loop.
    assert "vgprValuB" not in src
    assert "vgprG2LB" in src


def test_non_dtvb_kernel_still_masks_valub_in_tail_loop(
    isa_info_map, assembler, _globals
):
    """Control: without DirectToVgprB the tail loop keeps masking ValuB.

    Pins that routing the operand name through ``generateSrcStrForMFMA`` did
    not silently drop the masking for the ordinary LDS-staged case.
    """
    src = _emit(isa_info_map, assembler, DirectToVgprB=False)
    assert _undefined_vgpr_symbols(src) == []
    assert "vgprValuB" in src
    # v_cndmask against the shifted copy is the tail-loop K-edge zero fill.
    assert re.search(r"v_cndmask_b32 v\[vgprValuB_X\d+_I\d+", src)


# ---------------------------------------------------------------------------
# 3. The 32-byte global read is one instruction, and the G2L buffer holds it.
# ---------------------------------------------------------------------------
def test_32_byte_global_read_is_a_single_vector(isa_info_map, assembler, _globals):
    """A 32-byte vector must map to one table entry, not be split in two.

    Splitting it is what desynchronised bpl (still the whole vector) from the
    destination stride (only ``totalWidth``).
    """
    _, writer = _emit_with_writer(isa_info_map, assembler)
    tPB = writer.tPB
    assert tPB["bpeGR"] * tPB["glvw"] == 32
    assert tPB["globalReadInstruction"].totalWidth == 8
    assert tPB["nrcv"] // tPB["nrcvpi"] == 1


@pytest.mark.parametrize("dtvb", [True, False])
def test_g2l_buffer_covers_every_reference(isa_info_map, assembler, _globals, dtvb):
    """No G2LB reference may run past the buffer allocated for it.

    Before the fix the DirectToVgpr kernel addressed ``G2LB+32..47`` out of a
    32-register allocation, i.e. over ``vgprSerial``, which is what faulted the
    GPU. Checking it here does not need hardware.
    """
    src, writer = _emit_with_writer(isa_info_map, assembler, DirectToVgprB=dtvb)
    allocated = writer.states.b.numVgprG2LAllocated
    assert allocated > 0
    assert _max_g2l_offset(src, "B") < allocated


def test_dtv_operand_duplication_factor(isa_info_map, _globals):
    """WMMA v1 wave32 with a 16-wide instruction replicates the operand twice."""
    from Tensile.SolutionStructs.Solution import Solution

    state = _make_params(isa_info_map)
    assert Solution.dtvOperandDuplication(state, "A", isa_info_map) == 2
    assert Solution.dtvOperandDuplication(state, "B", isa_info_map) == 2

    noMi = dict(state)
    noMi["EnableMatrixInstruction"] = False
    assert Solution.dtvOperandDuplication(noMi, "B", isa_info_map) == 1
