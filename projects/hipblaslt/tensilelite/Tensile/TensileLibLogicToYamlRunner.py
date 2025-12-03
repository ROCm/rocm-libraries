################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import os
import subprocess
import yaml
import shutil
import argparse
from io import StringIO
from pathlib import Path
from Tensile import TensileCreateLibrary, TensileLibLogicToYaml, Tensile
import sys


def runHipblaslt(gemmLog, hipblasltBench, deviceId, yamlData):
    with open(f"{gemmLog}", "w") as f:
        subprocess.run(
            [f"{hipblasltBench}", "--device", f"{deviceId}", "--yaml", "-"],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            input=yamlData.encode("utf-8"),
        )


def extractSolution(gemmLog):
    blocks = None

    try:
        blocks = open(f"{gemmLog}", "r").read().split("[0]:")[1:]
    except FileNotFoundError:
        print(f"{gemmLog} does not exist in path!")
        raise
    if not blocks:
        raise ValueError("The benchmark output file may be corrupted.")

    size = blocks[0].split("\n")[1].split(",")[4:8]
    size = [int(size[i]) for i in [0, 1, 3, 2]]

    solutionIndex = (
        blocks[0].split("\n")[2].strip("--Solution index").strip(":")[1:].strip()
    )
    solutionName = (
        blocks[0].split("\n")[3].strip("--Solution name").strip("Cijk")[1:].strip()
    )

    if not solutionName or not solutionIndex:
        raise ValueError("The benchmark output file may be corrupted.")

    return solutionIndex, solutionName


def fixExactSolution(configYaml, yamlData):
    data = yaml.safe_load(StringIO(yamlData))

    newPattern = (
        rf"\[{data[0]['M']}, {data[0]['N']}, {data[0]['batch_count']}, {data[0]['K']}\]"
    )

    try:
        if Path(f"{configYaml}").is_file():
            subprocess.run(
                [
                    "sed",
                    "-i",
                    rf"s/^\(.*Exact:\).*/\1 {newPattern}/",
                    f"{configYaml}",
                ],
                stderr=subprocess.STDOUT,
                check=True,
            )
    except FileNotFoundError:
        print(f"{configYaml} does not exist in path!")
        raise


def generateConfigLib(configYaml, matchTable, solutionIndex):
    # find matching library from solution index in gemm.log
    table = open(matchTable, "r").read().split(f"{solutionIndex}:")[1:2]
    line = table[0].strip(":").split("\n")[0:2]
    libName = line[0].strip("[")[2:-1]
    internalSolutionIndex = line[1].strip("]").strip()

    # generate config yaml from library
    sys.argv = [
        "",
        "--input",
        f"{libName}",
        "--indices",
        f"{internalSolutionIndex}",
        "--output",
        f"{configYaml}",
    ]

    TensileLibLogicToYaml.main()

    return internalSolutionIndex


def buildTensile(tensilelitePath, clientPath, buildDir):
    if not os.path.isfile(clientPath):
        print("Building tensilelite client...")
        subprocess.call(
            ["invoke", "build-client", "--build-dir", buildDir], cwd=tensilelitePath
        )


def generateLiblogic(clientPath, workDir, configYaml):
    shutil.rmtree(workDir, ignore_errors=True)

    sys.argv = [
        "",
        "--prebuilt-client",
        f"{clientPath}",
        f"{configYaml}",
        f"{workDir}",
    ]

    Tensile.main()


def createLibrary(tensilePath, liblogicPath, arch):
    shutil.rmtree(tensilePath, ignore_errors=True)

    sys.argv = [
        "",
        "--code-object-version",
        "4",
        "--library-format",
        "msgpack",
        "--architecture",
        f"{arch}",
        f"{liblogicPath}",
        f"{tensilePath}",
        "HIP",
    ]
    TensileCreateLibrary.run()


def main(hipblasltPath, deviceId, workspace, arch, yamlData):
    hipblasltBench = os.path.join(
        hipblasltPath, "build/release/clients/hipblaslt-bench"
    )
    matchTable = os.path.join(
        hipblasltPath, "build/release/device-library/MatchTable.yaml"
    )
    tensilelitePath = os.path.join(hipblasltPath, "tensilelite")

    # run test gemm
    gemmLog = os.path.join(workspace, "gemm.log")
    runHipblaslt(gemmLog, hipblasltBench, deviceId, yamlData)

    # extract solutin index and name
    solutionIndex, solutionName = extractSolution(gemmLog)

    # find matching library from solution index in gemm.log
    # generate config yaml from library
    configYaml = os.path.join(workspace, "config.yaml")
    internalSolutionIndex = generateConfigLib(configYaml, matchTable, solutionIndex)
    # update output the output filename with the index
    configYaml = os.path.join(workspace, f"config_{internalSolutionIndex}.yaml")

    # if library is origami or we have multiple gemms for one index, we need to replace exact solution
    fixExactSolution(configYaml, yamlData)

    # if tensile-client does not exist, build it
    buildDir = os.path.join(workspace, "build_tmp")
    clientPath = os.path.join(buildDir, "tensilelite/client/tensilelite-client")
    buildTensile(tensilelitePath, clientPath, buildDir)

    # call Tensile to generate liblogic
    workDir = os.path.join(workspace, "WDirDevice")
    generateLiblogic(clientPath, workDir, configYaml)

    # create a library from the new library logic
    newLiblogicPath = os.path.join(workDir, "3_LibraryLogic")
    newTensilePath = os.path.join(workspace, "tensile")
    createLibrary(newTensilePath, newLiblogicPath, arch)

    # run hblt again
    os.environ["HIPBLASLT_TENSILE_LIBPATH"] = f"{newTensilePath}/library"
    gemmOutLog = os.path.join(workspace, "gemm_out.log")
    runHipblaslt(gemmOutLog, hipblasltBench, deviceId, yamlData)

    # extract solution name
    _, solutionNameOut = extractSolution(gemmOutLog)

    # match solution names
    solutionName = solutionName.split("UserArgs_")[1].strip()
    solutionNameOut = solutionNameOut.split("UserArgs_")[1].strip()

    if solutionName != solutionNameOut:
        ValueError("The generated and existing libraries do not match!")

    return 1


yamlData = "- {function: matmul, M: 768, N: 3072, K: 2048, lda: 2048, ldb: 2048, ldc: 768, ldd: 768, stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, alpha: 1.000000, beta: 0.000000, transA: T, transB: N, batch_count: 1, scaleA: 0, scaleB: 0, scaleC: 0, scaleD: 0, swizzleA: false, swizzleB: false, scaleAlpha_vector: false, gradient: false, use_e: false, bias_vector: false, bias_source: d, a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, bias_type: f32_r, compute_type: c_f32_r, activation_type: none, flush: false, any_stride: true, rotating: 512, cold_iters: 0, iters: 1, print_kernel_info: true}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Test utility for testing TensileLibLogicToYaml script."
    )
    parser.add_argument("--hipblaslt", help="Path to hipBLASLt", type=str)
    parser.add_argument(
        "--workspace",
        help="Path to the working space, all files will be saved here. Default is current dir.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--device", help="Device to run the initial benchmarks on.", type=int, default=0
    )
    parser.add_argument(
        "--arch",
        help="Architecture. Support only gfx950 currently. Default is gfx950",
        type=str,
        default="gfx950",
    )

    args = parser.parse_args()

    hipblasltPath = args.hipblaslt
    arch = args.arch
    deviceId = args.device
    workspace = os.path.abspath(args.workspace)

    os.makedirs(workspace, exist_ok=True)

    main(hipblasltPath, deviceId, workspace, arch, yamlData)
