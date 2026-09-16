#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit a single gfx1151 kernel's assembly, for eyeballing the w4a16 dequant path.

Development helper, not a test. Builds one Solution from a small parameter dict
and runs the same codegen path TensileCreateLibrary uses, writing the .s to
stdout or a file.

    python3 scripts/gen_w4a16_kernel.py --mode w4a16 -o /tmp/w4a16.s
    python3 scripts/gen_w4a16_kernel.py --mode bf16          # baseline
    python3 scripts/gen_w4a16_kernel.py --mode f8f16         # existing 2x narrow->wide cvt path
"""

import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))


def buildParams(mode, arch, depthU, blockSize, glvwA, zeroPoint=False):
    problemType = {
        "OperationType": "GEMM",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,   # TN: A is [M][K], K contiguous
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
    }
    if mode == "w4a16":
        problemType["DataTypeA"] = "I4"
        problemType["MacDataTypeA"] = "B"
        problemType["UseScaleAB"] = "Block"
        problemType["ScaleBlockSizeA"] = blockSize
        problemType["ScaleZeroPointA"] = zeroPoint
    elif mode == "f8f16":
        # The existing narrow->wide convert-before-LDS path (2x expansion); the
        # closest thing in-tree to what w4a16 needs, so useful as a reference.
        problemType["DataType"] = "H"
        problemType["DestDataType"] = "H"
        problemType["DataTypeA"] = "F8"
        problemType["MacDataTypeA"] = "H"
        problemType["MacDataTypeB"] = "H"
    elif mode != "bf16":
        raise SystemExit("unknown mode %s" % mode)

    params = {
        "ProblemType": problemType,
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
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": glvwA,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
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
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="w4a16", choices=("w4a16", "bf16", "f8f16"))
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--depthu", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    # -1 = let Solution derive it. int4 wants an explicit 16 (= 8 bytes/thread).
    ap.add_argument("--glvw-a", type=int, default=None)
    ap.add_argument("--zero-point", action="store_true", help="asymmetric: per-group int4 zero-points")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    import rocisa
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler

    isa = gfxToIsa(args.arch)
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        raise SystemExit("amdclang++ here does not support %s" % args.arch)

    assignGlobalParameters({"PrintSolutionRejectionReason": True}, iim)
    globalParameters["PrintSolutionRejectionReason"] = True

    glvwA = args.glvw_a
    if glvwA is None:
        glvwA = 8 if args.mode == "w4a16" else -1
    params = buildParams(args.mode, args.arch, args.depthu, args.block_size, glvwA, args.zero_point)
    params["ISA"] = isa
    params.update(
        matrixInstructionToMIParameters(
            params["MatrixInstruction"], isa, params["WavefrontSize"],
            params["ProblemType"], params["WorkGroup"], iim,
        )
    )

    sol = Solution(params, False, True, False, assembler, iim)
    if not sol.get("Valid"):
        raise SystemExit("solution rejected (reason printed above)")

    kernels = generateKernelObjectsFromSolutions([sol])
    if not kernels:
        raise SystemExit("no kernels generated")
    kernel = kernels[0]

    asmpath = shutil.which("amdclang++") or cxx
    ri = rocisa.rocIsa.getInstance()
    ri.init(tuple(kernel["ISA"]), asmpath)
    ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])

    kwa = KernelWriterAssembly(assembler, DebugConfig())
    kernel.duplicate = False
    kernel["BaseName"] = getKernelFileBase(False, kernel)
    res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
    src = res.src
    if isinstance(src, (bytes, bytearray)):
        src = src.decode(errors="replace")
    if res.err:
        sys.stderr.write("codegen returned err=%s\n" % res.err)

    sys.stderr.write("kernel: %s\n" % kernel["BaseName"])
    if args.output:
        with open(args.output, "w") as f:
            f.write(src)
        sys.stderr.write("wrote %s (%d bytes)\n" % (args.output, len(src)))
    else:
        sys.stdout.write(src)
    return 1 if res.err else 0


if __name__ == "__main__":
    sys.exit(main())
