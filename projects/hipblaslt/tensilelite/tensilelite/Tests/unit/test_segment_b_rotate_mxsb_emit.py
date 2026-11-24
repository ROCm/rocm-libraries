# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Codegen regression for LDSSegmentBRotate on a full-wave MXSB tile: MXF4 32x16x128
# with SourceSwap=True (effective MatrixInstN == wave size), where the MXSB scales have
# no half-wave select to flip. The kernel keeps two MXSB read addresses
# (LocalReadAddrMXSB+0/+1) that address the two vIdx halves; per SIMD pair they are
# swapped through MXSBRotateLRDelta so the scales follow the rotated B columns.
# CPU-only gfx1250 codegen (stinkytofu); no GPU is touched.
################################################################################

import copy
import re
import shutil

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution, globalParameters
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from tensilelite.Common.GlobalParameters import assignGlobalParameters

    saved = (copy.deepcopy(dict(globalParameters)), copy.deepcopy(dict(validParameters)),
             copy.deepcopy(dict(defaultSolution)))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    for target, state in zip((globalParameters, validParameters, defaultSolution), saved):
        target.clear()
        target.update(state)


def _params(iim, vwb, rotate):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = [32, 16, 128, 1, 1, 8, 4, 2, 2]  # MT256x256, M/N swapped by SourceSwap
    problem_type = {
        "OperationType": "GEMM", "DataType": "F4", "DestDataType": "s", "ComputeDataType": "s",
        "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
        "UseBeta": True, "Batched": True, "MXBlockA": 32, "MXBlockB": 32,
        "DataTypeMXSA": "E8", "DataTypeMXSB": "E8",
    }
    params = {
        "ProblemType": problem_type, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 32, "DepthU": 256,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
        "TransposeLDS": -1, "LdsPadA": -1, "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1, "1LDSBuffer": 0,
        "VectorWidthA": 8, "VectorWidthB": vwb, "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1, "LocalReadVectorWidth": -1,
        "SourceSwap": True, "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "TDMInst": 3, "TDMIterateMode": -1,
        "HalfPLR": 1, "TDMSplit": False, "LDSTrInst": False, "LDSSegmentInterleave": 1,
        "LDSSegmentBRotate": rotate, "StreamK": 0, "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": False, "StoreRemapVectorWidth": 0, "DirectToVgprA": False,
        "DirectToVgprB": False, "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 1,
        "UseSgprForGRO": 0, "ForceDisableShadowInit": True,
    }
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], iim))
    return params


def _emit_asm(iim, assembler, vwb, rotate):
    import rocisa
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.SolutionStructs.Naming import getKernelFileBase
    from tensilelite.create_library.run import (generateKernelObjectsFromSolutions,
                                                  processKernelSource)
    from rocisa_test_state import preserve_rocisa_kernel_state

    sol = Solution(_params(iim, vwb, rotate), False, True, False, assembler, iim)
    assert sol.get("Valid"), "solution did not validate"
    pieces = []
    with preserve_rocisa_kernel_state():
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        for kernel in generateKernelObjectsFromSolutions([sol]):
            ri = rocisa.rocIsa.getInstance()
            ri.init(tuple(kernel["ISA"]),
                    shutil.which("amdclang++") or "/usr/bin/amdclang++")
            ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
            kernel.duplicate = False
            kernel["BaseName"] = getKernelFileBase(False, kernel)
            res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
            assert res.err == 0, "kernel codegen failed (err=%s)" % res.err
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            pieces.append(src or "")
    return "\n".join(pieces)


@pytest.fixture(scope="module")
def asm_cache(_gp_gfx1250, gfx1250_iim, assembler):
    cache = {}

    def get(vwb, rotate):
        if (vwb, rotate) not in cache:
            cache[(vwb, rotate)] = _emit_asm(gfx1250_iim, assembler, vwb, rotate)
        return cache[(vwb, rotate)]

    return get


def _mxsb_loads(asm):
    """Address register of every MXSB ds_load, in program order."""
    return re.findall(r"ds_load_b\d+ v\[vgprValuMXSB\S*?\], v\[vgprLocalReadAddrMXSB\+(\d)", asm)


@pytest.mark.parametrize("vwb, nV", [(2, 2), (1, 4)])
def test_split_mxsb_rotate_uses_two_read_addresses(asm_cache, vwb, nV):
    asm = asm_cache(vwb, 1)
    assert re.search(r"^\.set sgprMXSBRotateLRDelta,", asm, re.M)
    # Pair p reads the vIdx halves in order (p, 1-p): delta = H - p*2H.
    assert re.search(r"s_mul_i32 s\[sgprMXSBRotateLRDelta\], \S+, \d+ +// MXSB-rotate: pair \* 2H", asm)
    assert re.search(r"s_sub_u32 s\[sgprMXSBRotateLRDelta\], \d+, s\[sgprMXSBRotateLRDelta\]", asm)
    loads = _mxsb_loads(asm)
    # The first half of the vIdx range reads +0, the second half reads +1.
    assert set(loads) == {"0", "1"}
    assert loads.count("0") >= 1 and loads.count("1") >= 1


@pytest.mark.parametrize("vwb", [2, 1])
def test_split_mxsb_fan_out_rebuilds_second_address(asm_cache, vwb):
    asm = asm_cache(vwb, 1)
    fan_outs = re.findall(
        r"v_add_nc_u32 v\[vgprLocalReadAddrMXSB\+1[^\]]*\], s\[sgprMXSBRotateLRDelta\], "
        r"v\[vgprLocalReadAddrMXSB\+0[^\]]*\]", asm)
    # Init plus one rebuild after every LDS buffer swap in the loops.
    assert len(fan_outs) >= 3
    # The tail increment moves both addresses by the same step, so the delta stays valid.
    incs = re.findall(r"v_add_co_u32 v\[vgprLocalReadAddrMXSB\+(\d)[^\]]*\], vcc_lo, \S+, "
                      r"v\[vgprLocalReadAddrMXSB\+\1[^\]]*\] +// lrMXSB \+=", asm)
    assert sorted(incs) == ["0", "1"]


def test_no_rotate_keeps_single_mxsb_address(asm_cache):
    asm = asm_cache(2, 0)
    assert "MXSBRotateLRDelta" not in asm
    assert "LocalReadAddrMXSB+1" not in asm
