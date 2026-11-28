################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
# ISA-level regression for the 16-bit-output MXFP8 subtile store layout.
#
# Every subtile D store sits behind an out-of-bounds skip branch.  If such a
# branch can be taken while a C load issued for beta is still outstanding, a
# persistent kernel can carry that load into the next tile, where it lands in
# registers the next tile is already using.  MXFP8 kernels therefore take the
# MXFP4 grouped layout: OOB guards and D stores follow all of the beta math.
#
# The scan below tracks every C load until an s_waitcnt retires it and fails on
# any skip branch reached with a C load outstanding.
#
# It needs the gfx950 assembly toolchain (amdclang++); it skips otherwise.
################################################################################

import re
import shutil

import pytest

_VMEM = re.compile(r"^(buffer|global|flat|scratch)_(load|store|atomic)")
_SKIP = re.compile(r"^s_cbranch_\w+ label_subtile_(all_stores_end|skip_store)")

# (MatrixInstruction, DepthU, PrefetchGlobalRead, PrefetchLocalRead, extra parameters)
_CONFIGS = {
    "du128-pgr2": ([16, 16, 128, 1, 1, 8, 4, 2, 4], 128, 2, 0, {"StreamK": 0}),
    "du128-pgr2-streamk": ([16, 16, 128, 1, 1, 8, 8, 2, 2], 128, 2, 1,
                           {"TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid"}),
    "du256-pgr1": ([16, 16, 128, 1, 1, 8, 8, 2, 2], 256, 1, 1, {"StreamK": 0}),
}


def _emit(mi, depthU, pgr, plr, extra):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults
    from tensilelite.Common.GlobalParameters import assignGlobalParameters
    from tensilelite.SolutionStructs.Solution import Solution
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    cxx = validateToolchain("amdclang++")
    if not cxx or not shutil.which(cxx):
        pytest.skip("gfx950 assembly toolchain (amdclang++) unavailable")

    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    asm = makeAssemblyToolchain(cxx, bundler, "default").assembler
    assignGlobalParameters({}, iim)

    pt = {"OperationType": "GEMM", "DataType": "F8", "DestDataType": "b",
          "ComputeDataType": "s", "HighPrecisionAccumulate": True,
          "TransposeA": True, "TransposeB": False, "UseBeta": True,
          "Batched": True, "StridedBatched": True, "MXBlockA": 32, "MXBlockB": 32}
    params = {"ProblemType": pt, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 64, "DepthU": depthU,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr,
        "ScheduleIterAlg": 3, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
        "LdsPadA": -1, "LdsPadB": -1, "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0, "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1, "LocalReadVectorWidth": -1,
        "SourceSwap": False, "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "DirectToLds": 1, "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": True, "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False, "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 4}
    params.update(extra)
    params.update(matrixInstructionToMIParameters(
        mi, isa, 64, pt, params["WorkGroup"], iim))
    sol = Solution(params, False, True, False, asm, iim)
    assert sol.get("Valid"), "MXFP8 subtile solution did not validate"

    from tensilelite.create_library.run import (
        generateKernelObjectsFromSolutions, processKernelSource)
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.Common.Types import DebugConfig
    import rocisa

    kwa = KernelWriterAssembly(asm, DebugConfig())
    for kernel in generateKernelObjectsFromSolutions([sol]):
        ri = rocisa.rocIsa.getInstance()
        ri.init(tuple(kernel["ISA"]), shutil.which(cxx))
        ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
        kernel["BaseName"] = "mxfp8_grouped_store_probe"
        try:
            kernel.duplicate = False
        except Exception:
            pass
        res = processKernelSource(
            kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        assert res.err == 0, "kernel codegen failed (err=%s)" % res.err
        src = res.src
        if isinstance(src, (bytes, bytearray)):
            src = src.decode(errors="replace")
        return src
    pytest.skip("no kernel object generated")


def _vmcnt(code):
    if not code.startswith("s_waitcnt"):
        return None
    m = re.search(r"vmcnt\((\d+)\)", code)
    if m:
        return int(m[1])
    return 0 if "cnt(" not in code else None


def _scan(src):
    """Return (C loads, skip branches, skip branches reached with a C load outstanding)."""
    pending, loads, skips, sites = [], 0, 0, []
    for n, line in enumerate(src.splitlines(), 1):
        code = line.split("//")[0].strip()
        if not code:
            continue
        count = _vmcnt(code)
        if count is not None:
            pending = [age for age in pending if age < count]
        elif _VMEM.match(code):
            pending = [age + 1 for age in pending]
            if "load C" in line:
                pending.append(0)
                loads += 1
        elif _SKIP.match(code):
            skips += 1
            if pending:
                sites.append((n, code))
        elif code.startswith(("s_setpc", "s_endpgm")):
            pending = []
    return loads, skips, sites


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_no_store_skip_branch_with_c_loads_in_flight(name):
    src = _emit(*_CONFIGS[name])
    loads, skips, sites = _scan(src)
    assert loads > 0 and skips > 0, \
        "kernel has no beta C loads or no store-skip branches; the scan checks nothing"
    assert not sites, \
        "%d store-skip branch(es) taken with C loads in flight, first at asm line %d: %s" % (
            len(sites), sites[0][0], sites[0][1])
