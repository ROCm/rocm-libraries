################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# Generates four rocblas-bench input files from the library logic files: beta 0/1 + rotating buffer 0/1. The cold and hot iteration counts are based on the performance in the library logic.

# Usage:
# $ python3 rocblas-benchInputCreator.py [-v] [-i <init>] <lib logic dir> <output dir>

# creates the benchmark yamls and verification files with default iterations and initialization:
# $ python3 rocblas-benchInputCreator.py -v ../libLogics ./

# creates the benchmark yamls for 3s of benchamrking with default initialization:
# $ python3 rocblas-benchInputCreator.py -v -d 3.0 ../libLogics ./

# creates the benchmark yamls and verification files with hpl initialization:
# $ python3 rocblas-benchInputCreator.py -v -i hpl ../libLogics ./

# creates the benchmark yamls using the default initialization (trig or int)
# $ python3 rocblas-benchInputCreator.py ../libLogics ./

import argparse
import os
import yaml
import math

from yaml import SafeDumper as yamlDumper
from yaml import SafeLoader as yamlLoader

typeIndexToName = {0: "f32_r", 1: "f64_r", 2: "f32_c", 3: "f64_c", 4: "f16_r", 5: "i8_r", 6: "i32_r", 7: "bf16_r", 8: "i8_r", 10: "f8_r", 11: "bf8_r", 12: "f8b8", 13: "b8f8"}

def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"libLogic" : "Input library logic file",
         "outDir"   : "Output directory for rocBLAS-bench yaml files",
         "verify"   : "Also output verify version of yaml files",
         "initial"  : "Matrix initialization: hpl, trig, int. The default is trig for non Int8 datatype, and int for Int8.",
         "duration" : "total benchmark duration in seconds. Default is 0 (10/2 iterations)"
    }

    argParser.add_argument("libLogic", metavar="logic-file", type=str, help=h["libLogic"])
    argParser.add_argument("outDir", metavar="output-dir", type=str, help=h["outDir"])
    argParser.add_argument("--verify", "-v", action="store_true", help=h["verify"])
    argParser.add_argument("--initialization", "-i", action="store", type=str, default = 'trig',  help=h["initial"])
    argParser.add_argument("--duration", "-d", action="store", type=float, default = 0.0,  help=h["duration"])

    return argParser.parse_args()

def getProblemType(problem):
    # transA/B, a/b/c/d/compute_type
    problemDict = {}

    if problem["ComplexConjugateA"]:
        problemDict["transA"] = "C"
    elif problem["TransposeA"]:
        problemDict["transA"] = "T"
    else:
        problemDict["transA"] = "N"

    if problem["ComplexConjugateB"]:
        problemDict["transB"] = "C"
    elif problem["TransposeB"]:
        problemDict["transB"] = "T"
    else:
        problemDict["transB"] = "N"

    problemDict["a_type"] = typeIndexToName[problem["DataType"]]
    problemDict["b_type"] = typeIndexToName[problem["DataType"]]
    problemDict["c_type"] = typeIndexToName[problem["DestDataType"]]
    problemDict["d_type"] = typeIndexToName[problem["DestDataType"]]

    f8gemm = True if (problem["DataType"]>=10) else False # is it f8

    if "ComputeDataType" in problem:
        compType = typeIndexToName[problem["ComputeDataType"]]
        if f8gemm: # f8 gemm
          if (typeIndexToName[problem["DataType"]] =="f8b8" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for F8B8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["composite_compute_type"] = "f8_bf8_f32"
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for B8F8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["composite_compute_type"] = "bf8_f8_f32"
          elif (typeIndexToName[problem["DataType"]] =="f8_r" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for F8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["composite_compute_type"] = "f8_f8_f32"
          elif (typeIndexToName[problem["DataType"]] =="bf8_r" and typeIndexToName[problem["DestDataType"]]=="f16_r"): # for B8HS
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["composite_compute_type"] = "bf8_bf8_f32"
          elif (typeIndexToName[problem["DataType"]] =="f8b8" and typeIndexToName[problem["DestDataType"]]=="f32_r"): # for B8SS
            problemDict["a_type"] = "f8_r"
            problemDict["b_type"] = "bf8_r"
            problemDict["composite_compute_type"] = "f32"
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="f32_r"): # for B8SS
            problemDict["a_type"] = "bf8_r"
            problemDict["b_type"] = "f8_r"
            problemDict["composite_compute_type"] = "f32"
          elif (typeIndexToName[problem["DataType"]] =="b8f8" and typeIndexToName[problem["DestDataType"]]=="bf8_r"): # for B8F8B8S
            problemDict["a_type"] = "bf8_r"
            problemDict["b_type"] = "f8_r"
            problemDict["composite_compute_type"] = "f32"
          elif (typeIndexToName[problem["DataType"]] =="f8b8" and typeIndexToName[problem["DestDataType"]]=="bf8_r"): # for F8B8B8S
            problemDict["a_type"] = "f8_r"
            problemDict["b_type"] = "bf8_r"
            problemDict["composite_compute_type"] = "f32"
          else:
            problemDict["composite_compute_type"] = "f32"
        else:
          problemDict["compute_type"] = compType
    else:
        if problemDict["a_type"] == "f16_r" and problem["HighPrecisionAccumulate"]:
            problemDict["compute_type"] = "f32_r"
        elif problem["DataType"] == 5:
            problemDict["compute_type"] = "i32_r"
        else:
            problemDict["compute_type"] = problemDict["a_type"]

    if "F32XdlMathOp" in problem and problem["F32XdlMathOp"]==9: # XF32
        problemDict["math_mode"] = 1

    return problemDict

def getSizeParams(size, transA, transB):
    # M/N/K, batch_count, lda/b/c/d
    sizeDict = {}
    sizeDict["M"] = size[0]
    sizeDict["N"] = size[1]
    sizeDict["K"] = size[3]

    if size[2] != 1:
        sizeDict["batch_count"] = size[2]
        # rocBLAS-bench will handle setting default strides

    if len(size)==8: # ld defined in the library logic
        sizeDict["ldc"] = size[4]
        sizeDict["ldd"] = size[5]
        sizeDict["lda"] = size[6]
        sizeDict["ldb"] = size[7]
    else: # no ld defined in the library logic
        sizeDict["ldc"] = size[0]
        sizeDict["ldd"] = size[0]
        if not transA and not transB: # NN
            sizeDict["lda"] = size[0]
            sizeDict["ldb"] = size[3]
        elif transA and not transB:   # TN
            sizeDict["lda"] = size[3]
            sizeDict["ldb"] = size[3]
        elif not transA and transB:   # NT
            sizeDict["lda"] = size[0]
            sizeDict["ldb"] = size[1]
        else:                         # TT
            sizeDict["lda"] = size[3]
            sizeDict["ldb"] = size[1]

    return sizeDict


def dumpYaml(outDir, outputfile,postfix, content):
    name = outputfile+postfix
    benchPath = os.path.join(outDir, name)
    with open(benchPath, "w") as f:
        yaml.dump(content, f, yamlDumper, default_flow_style=None, sort_keys=False, width=5000)
        f.write(f"# End of {name} \n")

def createYaml(args, outputfile, problem, sizeMappings, verify):
    bench = []
    benchStrided = []
    benchGeneralBatched = []

    bench_rotating = []
    benchStrided_rotating = []
    benchGeneralBatched_rotating = []

    bench_beta0 = []
    benchStrided_beta0 = []
    benchGeneralBatched_beta0 = []

    bench_beta0_rotating = []
    benchStrided_beta0_rotating = []
    benchGeneralBatched_beta0_rotating = []

    bench_verify = []
    benchStrided_verify = []
    benchGeneralBatched_verify = []

    # get GEMM function and matrix orientation - Fixed for each library
    problemParams = getProblemType(problem)
    transA = problem["TransposeA"]
    transB = problem["TransposeB"]

    # check if this is f8/b8:
    f8gemm = True if (problem["DataType"]>=10) else False

    if verify:
        otherParams_verify = {"alpha": 1, "beta": 1, "iters": 1, "cold_iters": 0, "norm_check": 1}

    #initialization
    if (args.initialization=='hpl' and problemParams["a_type"]!="i8_r"):
        init = {"initialization": "hpl"}
    elif (args.initialization=='trig' and problemParams["a_type"]!="i8_r"):
        init = {"initialization": "trig_float"}
    elif args.initialization== 'int':
        init = {"initialization": "rand_int"}
    else:
      print(f"Initialization {args.initialization} is not allowed for int8 datatype. Initialization changed to rand_int.")
      init = {"initialization": "rand_int"}

    # check if the library is General Batched based on the library name
    generalBatched = True if "_GB.yaml" in os.path.split(args.libLogic)[-1] else False

    # create rocBLAS-bench call for each size in logic file
    for (size, perf) in sizeMappings: # size[0] = M, size[1] = N, size[2] = batch_count, size[3] = K, size[4] = ldc, size[5] = ldd, size[6] = lda, size[7] = ldb

        params = {}

        if (not generalBatched and size[2] == 1 and not f8gemm):  # non-f8, non-batched gemm (serves both HPA and non-HPA)
            params["rocblas_function"] = "rocblas_gemm_ex"
        elif (not generalBatched and size[2] != 1 and not f8gemm): # non-f8, strided_batched gemm (serves both HPA and non-HPA)
            params["rocblas_function"] = "rocblas_gemm_strided_batched_ex"
        elif not generalBatched: # f8
            params["rocblas_function"] = "rocblas_gemm_ex3"
        elif (generalBatched and not f8gemm):  # non-f8, general batched gemm (serves both HPA and non-HPA) currently there is no f8 general batched
            params["rocblas_function"] = "rocblas_gemm_batched_ex"
        else:
            raise RuntimeError(" F8 GEMM is not supporting General Batched.")

        sizeParams = getSizeParams(size, transA, transB)

        if  args.duration>0.0:
            latency = 2*sizeParams['M']*sizeParams['N']*sizeParams['K']/perf[1]/1000 # us
            latency *= sizeParams["batch_count"] if "batch_count" in sizeParams else 1
            cold_iters = math.ceil( args.duration* 1e6 / latency)
            iters = cold_iters
            coe = 1.15
        else:
            cold_iters = 2
            iters = 10
            coe = 1

        otherParams = {"alpha": 1, "beta": 1, "iters": iters, "cold_iters": cold_iters}
        otherParams_rotating = {**otherParams, "flush_memory_size": 536870812}

        otherParams_beta0 = {"alpha": 1, "beta": 0, "iters": math.ceil( coe * iters), "cold_iters": math.ceil( coe * cold_iters)}
        otherParams_beta0_rotating = {**otherParams_beta0, "flush_memory_size": 536870812}

        params.update(problemParams)
        params.update(sizeParams)
        params.update(init)

        if (size[2] == 1 and not generalBatched):
            bench.append({**params, **otherParams})
            bench_rotating.append({**params, **otherParams_rotating})
            bench_beta0.append({**params, **otherParams_beta0})
            bench_beta0_rotating.append({**params, **otherParams_beta0_rotating})
            if verify:
                bench_verify.append({**params, **otherParams_verify})

        elif (generalBatched):
            benchGeneralBatched.append({**params, **otherParams})
            benchGeneralBatched_rotating.append({**params, **otherParams_rotating})
            benchGeneralBatched_beta0.append({**params, **otherParams_beta0})
            benchGeneralBatched_beta0_rotating.append({**params, **otherParams_beta0_rotating})
            if verify:
                benchGeneralBatched_verify.append({**params, **otherParams_verify})
        else:
            benchStrided.append({**params, **otherParams})
            benchStrided_rotating.append({**params, **otherParams_rotating})
            benchStrided_beta0.append({**params, **otherParams_beta0})
            benchStrided_beta0_rotating.append({**params, **otherParams_beta0_rotating})
            if verify:
                benchStrided_verify.append({**params, **otherParams_verify})

    # write output
    if len(bench) > 0:
        dumpYaml(args.outDir, outputfile,"_bench.yaml", bench)
        dumpYaml(args.outDir, outputfile,"_bench_beta0.yaml", bench_beta0)
        dumpYaml(args.outDir, outputfile, "_bench_rotating.yaml", bench_rotating)
        dumpYaml(args.outDir, outputfile, "_bench_beta0_rotating.yaml", bench_beta0_rotating)
        if verify:
            dumpYaml(args.outDir, outputfile, "_verify.yaml", bench_verify)

    if len(benchStrided) > 0:
        dumpYaml(args.outDir, outputfile, "_bench-strided.yaml", benchStrided)
        dumpYaml(args.outDir, outputfile, "_bench-strided_beta0.yaml", benchStrided_beta0)
        dumpYaml(args.outDir, outputfile, "_bench-strided_rotating.yaml", benchStrided_rotating)
        dumpYaml(args.outDir, outputfile, "_bench-strided_beta0_rotating.yaml", benchStrided_beta0_rotating)
        if verify:
            dumpYaml(args.outDir, outputfile, "_verify-strided.yaml", benchStrided_verify)

    if len(benchGeneralBatched) > 0:
        dumpYaml(args.outDir, outputfile, "_bench-general-batched.yaml", benchGeneralBatched)
        dumpYaml(args.outDir, outputfile, "_bench-general-batched_beta0.yaml", benchGeneralBatched_beta0)
        dumpYaml(args.outDir, outputfile, "_bench-general-batched_rotating.yaml", benchGeneralBatched_rotating)
        dumpYaml(args.outDir, outputfile, "_bench-general-batched_beta0_rotating.yaml", benchGeneralBatched_beta0_rotating)
        if verify:
            dumpYaml(args.outDir, outputfile, "_verify-general-batched.yaml", benchGeneralBatched_verify)

def main():
    args = parseArgs()

    if not (args.initialization in ['hpl', 'trig', 'int']):
        raise RuntimeError(f"Initialization {args.initialization} is not allowed. Choose from hpl, trig, or int.")

    for libname in os.listdir(args.libLogic):
        output = os.path.splitext(libname)[0]
        print(f" working on {output}")
        yamlName = os.path.join(args.libLogic,libname)
        with open(yamlName) as f:
            logicData = yaml.load(f, yamlLoader)

        try:
            os.makedirs(args.outDir)
        except OSError:
            pass

        problem = logicData[4]
        sizeMappings = logicData[7]

        createYaml(args, output, problem, sizeMappings, args.verify)

if __name__ == "__main__":
    main()
