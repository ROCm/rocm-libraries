# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared builders for the gfx1250 swizzled-TDM (SwizzleTensorA/B + TDM) unit tests.

``swizzled_params`` builds a TN gfx1250 solution parameter dict around the
configuration the swizzle suites use; ``derive`` runs the real Solution
derivation; ``codegen`` runs CPU-only kernel codegen (no assembly) and returns
the kernel source together with the kernel writer, whose states describe the
emitted register layout.
"""

import shutil

import pytest

# Base fork parameters shared by the swizzle-A / swizzle-B suites.
BASE_FORK = {
    "UseSgprForGRO": 0, "ForceDisableShadowInit": True, "DepthU": 256, "TDMInst": 3,
    "HalfPLR": 0, "TDMIterateMode": 0, "PrefetchGL2": 0, "TransposeLDS": -1,
    "LDSTrInst": False, "StaggerU": 0, "ScheduleIterAlg": 4, "PrefetchLocalRead": 1,
    "PrefetchGlobalRead": 2, "StoreRemapVectorWidth": 0, "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer", "GlobalReadVectorWidthA": -1,
    "GlobalReadVectorWidthB": -1, "VectorWidthA": 1, "VectorWidthB": 1,
    "LocalReadVectorWidth": -1, "1LDSBuffer": 0, "SourceSwap": True, "ClusterDim": [1, 1],
    "WorkGroupMapping": 1, "StoreVectorWidth": -1, "WavefrontSize": 32,
    "ExpandPointerSwap": False, "CompactLoopStore": False, "PreloadKernArgs": False,
    "KernelLanguage": "Assembly", "InnerUnroll": 1, "StreamK": 0,
    "DirectToVgprA": False, "DirectToVgprB": False,
}

# Per data type: MatrixInstruction K, default DepthU and ProblemType overrides.
DTYPES = {
    "f8":   dict(miK=128, du=256, pt={"DataType": "F8", "DestDataType": "B"}),
    "mxf8": dict(miK=128, du=256, pt={"DataType": "F8", "DestDataType": "B", "MXBlockA": 32, "MXBlockB": 32}),
    "f4":   dict(miK=128, du=256, pt={"DataType": "F4", "DestDataType": "B"}),
    "mxf4": dict(miK=128, du=256, pt={"DataType": "F4", "DestDataType": "B", "MXBlockA": 32, "MXBlockB": 32}),
    "bf16": dict(miK=32,  du=64,  pt={"DataType": "B", "DestDataType": "B"}),
    "f16":  dict(miK=32,  du=64,  pt={"DataType": "H", "DestDataType": "H"}),
    "i8":   dict(miK=64,  du=128, pt={"DataType": "I8", "DestDataType": "I", "ComputeDataType": "I"}),
    "f32":  dict(miK=4,   du=32,  pt={"DataType": "S", "DestDataType": "S"}),
}


def swizzled_params(iim, side="B", dtype="f8", wt=(2, 2), wg=(2, 2), pt=None, workgroup=(16, 16, 1), miMN=(16, 16), **fork):
    """Solution parameters for a TN swizzle-``side`` ("A", "B" or "AB") gfx1250 kernel."""
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

    d = DTYPES[dtype]
    isa = gfxToIsa("gfx1250")
    problem_type = {
        "OperationType": "GEMM", "ComputeDataType": "s", "HighPrecisionAccumulate": True,
        "TransposeA": True, "TransposeB": False, "UseBeta": True, "Batched": True,
        "SupportUserArgs": True, "UseBias": 0,
        "SwizzleTensorA": side in ("A", "AB"), "SwizzleTensorB": side in ("B", "AB"),
    }
    problem_type.update(d["pt"])
    problem_type.update(pt or {})
    params = dict(BASE_FORK)
    params["DepthU"] = d["du"]
    # A swizzled operand reads unpadded LDS (the solution adds its own VW>1 pad); the other uses auto pad.
    for tc in "AB":
        swz = problem_type["SwizzleTensor" + tc]
        params["LdsPad" + tc] = 0 if swz else -1
        params["LdsBlockSizePerPad" + tc] = 0 if swz else -1
    params.update(fork)
    mi = [miMN[0], miMN[1], d["miK"], 1, 1, wt[0], wt[1], wg[0], wg[1]]
    params.update({"ProblemType": problem_type, "ISA": isa, "MatrixInstruction": mi,
                   "WorkGroup": list(workgroup)})
    params.update(matrixInstructionToMIParameters(mi, isa, params["WavefrontSize"], problem_type,
                                                  params["WorkGroup"], iim))
    return params


def derive(params, assembler, iim):
    """Run the real Solution derivation (prints its reject reasons)."""
    from tensilelite.SolutionStructs.Solution import Solution
    return Solution(params, False, True, False, assembler, iim)


def codegen(sol, assembler):
    """CPU-only codegen of ``sol``: returns (kernel source text, kernel writer)."""
    import rocisa
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.Toolchain.Validators import validateToolchain
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions, processKernelSource

    cxx = validateToolchain("amdclang++")
    if not cxx or not shutil.which(cxx):
        pytest.skip("gfx1250 assembly toolchain (amdclang++) unavailable")
    kernels = generateKernelObjectsFromSolutions([sol])
    assert kernels, "no kernel object generated"
    kernel = kernels[0]
    kwa = KernelWriterAssembly(assembler, DebugConfig())
    ri = rocisa.rocIsa.getInstance()
    ri.init(tuple(kernel["ISA"]), shutil.which(cxx))
    ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
    kernel["BaseName"] = "swizzled_tdm_probe"
    try:
        kernel.duplicate = False
    except Exception:
        pass
    res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
    src = res.src.decode(errors="replace") if isinstance(res.src, (bytes, bytearray)) else res.src
    return res.err, src, kwa
