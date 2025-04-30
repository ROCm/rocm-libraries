################################################################################
#
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

# This script reads the output of rocblas-bench with ROCBLAS_LAYER=2 (default) or ROCBLAS_LAYER=4, and summarizes the sizes based on GEMM types.
# It covers I8/HHS/HSS/BBS/BSS/SGEMM/DGEMM/CGEMM/ZGEMM.

# Usage
# python3 .\rocblas-parser.py --input .\parser\test2.log --output .\parser\summary.log -b .\parser\bench.yaml

import os
import argparse
import yaml

from yaml import SafeDumper as yamlDumper


def parseBenchCofnig():
    argParser = argparse.ArgumentParser()

    h = {"inputfile" : "input file",
         "outputfile" : "output file containing the summary of all GEMM sizes ()",
         "blas" : "ROCBLAS_LAYER, default:2, other choice is 4",
         "verify"   : "Also output verify version of rocblas-bench yaml files",
         "bench" : "create rocblas-bench output file: bench.log",
         "initial"  : "Matrix initialization: hpl, trig, int. The default is trig for non Int8 datatype, and int for Int8."         
    }

    argParser.add_argument("--input",            action="store", metavar="rocblas_log-file", type=str, help=h["inputfile"])
    argParser.add_argument("--output",           action="store", metavar="output-dir",       type=str, help=h["outputfile"])
    argParser.add_argument("--bench", "-b",      action="store", metavar="rocblas-bench",    type=str, default='', help=h["bench"])
    argParser.add_argument("--blas_layer", "-l", action="store", metavar="ROCBLAS_LAYER",    type=int, default = 2,   help=h["blas"])
    argParser.add_argument("--verify", "-v",     action="store_true", help=h["verify"])
    argParser.add_argument("--initialization", "-i", action="store", type=str, default = 'trig',  help=h["initial"])

    return argParser.parse_args()

def gemmfinder(config):
    output ="unrecognized"

    if   (config.a_type == "bf16_r" and config.b_type == "bf16_r"): 
        if (config.c_type == "bf16_r"  and config.d_type == "bf16_r" and config.compute_type == "f32_r"): 
            output = "BBS"
        elif (config.c_type == "bf32_r"  and config.d_type == "bf32_r" and config.compute_type == "f32_r"): 
            output = "BSS"
    elif (config.a_type == "f16_r" and config.b_type == "f16_r"): 
        if (config.c_type == "f16_r"  and config.d_type == "f16_r" and config.compute_type == "f32_r"): 
            output = "HHS"
        elif (config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r"): 
            output = "HSS"
    elif (config.a_type == "i8_r" and config.b_type == "i8_r" and config.c_type == "i_r32"  and config.d_type == "i_r32" and config.compute_type == "i_r32" ):     
            output = "I8II"
    elif (((config.f == "gemm_ex" or config.f == "gemm_strided_batched_ex") or (config.rocblas_function == "rocblas_gemm_ex" or config.rocblas_function == "rocblas_gemm_strided_batched_ex"))  and config.a_type == "f32_r" and config.b_type == "f32_r"  and config.c_type == "f32_r"  and config.d_type == "f32_r" and config.compute_type == "f32_r") or ((config.f == "gemm" or config.f == "gemm_strided_batched") and config.r == "f32_r"): 
        output = "SGEMM"
    elif ((config.f == "gemm_ex" or config.f == "gemm_strided_batched_ex") and config.a_type == "f64_r" and config.b_type == "f64_r"  and config.c_type == "f64_r"  and config.d_type == "f64_r" and config.compute_type == "f64_r") or ((config.f == "gemm" or config.f == "gemm_strided_batched") and config.r == "f64_r"): 
        output = "DGEMM"
    elif (((config.f == "gemm_ex" or config.f == "gemm_strided_batched_ex") or (config.rocblas_function == "rocblas_gemm_ex" or config.rocblas_function == "rocblas_gemm_strided_batched_ex"))  and config.a_type == "f32_c" and config.b_type == "f32_c"  and config.c_type == "f32_c"  and config.d_type == "f32_c" and config.compute_type == "f32_c") or ((config.f == "gemm" or config.f == "gemm_strided_batched") and config.r == "f32_c"): 
        output = "CGEMM"
    elif (((config.f == "gemm_ex" or config.f == "gemm_strided_batched_ex") or (config.rocblas_function == "rocblas_gemm_ex" or config.rocblas_function == "rocblas_gemm_strided_batched_ex"))  and config.a_type == "f64_c" and config.b_type == "f64_c"  and config.c_type == "f64_c"  and config.d_type == "f64_c" and config.compute_type == "f64_c") or ((config.f == "gemm" or config.f == "gemm_strided_batched") and config.r == "f64_c"): 
        output = "ZGEMM"

    if   ((config.transposeA == "N" and config.transposeB == "N") or (config.transA == "N" and config.transB == "N")):
        output += "_NN"
    elif ((config.transposeA == "N" and config.transposeB == "T") or (config.transA == "N" and config.transB == "T")):
        output += "_NT"    
    elif ((config.transposeA == "T" and config.transposeB == "N") or (config.transA == "T" and config.transB == "N")):
        output += "_TN"    
    elif ((config.transposeA == "T" and config.transposeB == "T") or (config.transposeA == "T" and config.transposeB == "T")):
        output += "_TT"    

    if (config.f == "gemm_strided_batched_ex" or config.f == "gemm_strided_batched" or config.f == "rocblas_gemm_strided_batched_ex" or config.f == "rocblas_gemm_strided_batched"): 
        output +="_SB"

    return output

def readfile_rocblas_layer(filename, blas_layer):

    argParser = argparse.ArgumentParser()

    argParser.add_argument("--rocblas_function", action="store", type=str, default = '')
    argParser.add_argument("-f", action="store", type=str, default = '')
    argParser.add_argument("-r", action="store", type=str, default = '')
    
    argParser.add_argument("--transA", action="store", type=str, default = '')
    argParser.add_argument("--transB", action="store", type=str, default = '')
    argParser.add_argument("--transposeA", action="store", type=str, default = '')
    argParser.add_argument("--transposeB", action="store", type=str, default = '')
    if blas_layer==4:
        argParser.add_argument("--M", action="store", type=int, default = '')
        argParser.add_argument("--N", action="store", type=int, default = '')
        argParser.add_argument("--K", action="store", type=int, default = '')
    else:
        argParser.add_argument("-m", action="store", type=int, default = '')
        argParser.add_argument("-n", action="store", type=int, default = '')
        argParser.add_argument("-k", action="store", type=int, default = '')

    argParser.add_argument("--lda", action="store", type=int, default = '')
    argParser.add_argument("--ldb", action="store", type=int, default = '')
    argParser.add_argument("--ldc", action="store", type=int, default = '')
    
    argParser.add_argument("--a_type", action="store", type=str, default = '')
    argParser.add_argument("--b_type", action="store", type=str, default = '')
    argParser.add_argument("--c_type", action="store", type=str, default = '')
    argParser.add_argument("--d_type", action="store", type=str, default = '')
    
    argParser.add_argument("--batch_count",  action="store", type=int, default = 0)
    argParser.add_argument("--compute_type", action="store", type=str, default = '')
    
    argParser.add_argument("--alpha", type=float)
    argParser.add_argument("--beta", action="store", type=float, default = 1.0)

    argParser.add_argument("--call_count",  action="store", type=int, default = 0)

    matrices = {}

    config = open(filename, 'r')
    Lines = config.readlines()

    count = 0 # total size in the file
    duplicate_count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        if blas_layer==4:
            line = line.replace("- {","")
            line = line.replace("}","")
            line = line.replace(":","")
            line = line.replace(", "," --")
            line = line.replace("'","")
            line = line.replace('"',"")
            line = line.replace("rocblas_function","--rocblas_function")
        
        config, unknown = argParser.parse_known_args(line.strip().split())
        
        gemmtype = gemmfinder(config)

        if not (gemmtype in matrices):
            matrices[gemmtype] = []

        if blas_layer==2:
            if ("_SB" in gemmtype): # for strided
                newSize = (config.m, config.n, config.batch_count, config.k, config.lda, config.ldb, config.ldc, config.alpha, config.beta)
                if not (newSize in matrices[gemmtype]): 
                    matrices[gemmtype].append(newSize) ## batch
                else:
                    duplicate_count += 1
            else: # for non-strided sizes
                newSize = (config.m, config.n, config.k, config.lda, config.ldb, config.ldc, config.alpha, config.beta)
                if not (newSize in matrices[gemmtype]):
                    matrices[gemmtype].append(newSize) 
                else:
                    duplicate_count += 1
        elif blas_layer==4:
            if ("_SB" in gemmtype): # for strided
                newSize = (config.M, config.N, config.batch_count, config.K, config.lda, config.ldb, config.ldc, config.alpha, config.beta, config.call_count)
                if not (newSize in matrices[gemmtype]): 
                    matrices[gemmtype].append(newSize) ## batch
                else:
                    duplicate_count += 1
            else: # for non-strided sizes
                newSize = (config.M, config.N, config.K, config.lda, config.ldb, config.ldc, config.alpha, config.beta, config.call_count)
                if not (newSize in matrices[gemmtype]):
                    matrices[gemmtype].append(newSize) 
                else:
                    duplicate_count += 1

    #sorting
    matrices = dict(sorted(matrices.items()))
    for gemmtype in matrices:
      if ("_SB" in gemmtype): # for strided
        matrices[gemmtype].sort(key=sortfuc_strided)
      else:
        matrices[gemmtype].sort(key=sortfuc)

    return [matrices,count,duplicate_count]

def sortfuc(size):
    return (size[0],size[1],size[2])

def sortfuc_strided(size):
    return (size[0],size[1],size[3], size[2])


def createOutput(filename, matrices,count,duplicate_count, blas_layer):
  count_unique = 0
  with open(filename,'w') as f:
    f.write(f"\n -- total bench in the file: {count} - duplicates: {duplicate_count}")
    
    for gemmtype in matrices:
      strided = "SB" in gemmtype

      f.write(f"\n\n--- {gemmtype} (with duplicates): {len(matrices[gemmtype])} \n")
      previous_size = (0,0,0,0,0,0,0) if strided else (0,0,0,0,0,0)
      for size in matrices[gemmtype]:
        duplicate = "duplicate"
        if size[:-2] != previous_size:
            duplicate = ""
            count_unique+=1

        if (strided): # for strided
            gemmconfig = f" - [ {size[0]}, {size[1]}, {size[2]}, {size[3]} ] # lda: {size[4]} ldb: {size[5]} ldc: {size[6]} alpha: {size[7]} beta: {size[8]}"
            if (blas_layer==4):
                gemmconfig +=f" call_count: {size[9]}  {duplicate}\n"
            else:
                gemmconfig +=f"  {duplicate}\n"
        else:
            gemmconfig = f" - [ {size[0]}, {size[1]}, 1, {size[2]} ] # lda: {size[3]} ldb: {size[4]} ldc: {size[5]} alpha: {size[6]} beta: {size[7]}"
            if (blas_layer==4):
                gemmconfig +=f" call_count: {size[8]}  {duplicate}\n"
            else:
                gemmconfig +=f"  {duplicate}\n"

        f.write(gemmconfig)
        previous_size = size[:-2]

    f.write(f"\n -- total unique is: {count_unique}")

def create_rocBLAS_bench(benchFile, matrices,verify,initialization):

    bench = []

    for gemmtype in matrices:
        print(gemmtype)
        # get GEMM function type: a/b/c/d/compute_type
        problemDict = {}
        problemDict["rocblas_function"] = "rocblas_gemm_strided_batched_ex" if "SB" in gemmtype else  "rocblas_gemm_ex"

        if ("HHS" in gemmtype):
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["c_type"] = "f16_r"
            problemDict["d_type"] = "f16_r"
            problemDict["compute_type"] = "f32_r"
    
        elif ("HSS" in gemmtype):
            problemDict["a_type"] = "f16_r"
            problemDict["b_type"] = "f16_r"
            problemDict["c_type"] = "f32_r"
            problemDict["d_type"] = "f32_r"
            problemDict["compute_type"] = "f32_r"

        elif ("BBS" in gemmtype):
            problemDict["a_type"] = "bf16_r"
            problemDict["b_type"] = "bf16_r"
            problemDict["c_type"] = "bf16_r"
            problemDict["d_type"] = "bf16_r"
            problemDict["compute_type"] = "f32_r"
    
        elif ("BSS" in gemmtype):
            problemDict["a_type"] = "bf16_r"
            problemDict["b_type"] = "bf16_r"
            problemDict["c_type"] = "f32_r"
            problemDict["d_type"] = "f32_r"
            problemDict["compute_type"] = "f32_r"

        elif ("SGEMM" in gemmtype):
            problemDict["a_type"] = "f32_r"
            problemDict["b_type"] = "f32_r"
            problemDict["c_type"] = "f32_r"
            problemDict["d_type"] = "f32_r"
            problemDict["compute_type"] = "f32_r"

        elif ("DGEMM" in gemmtype):
            problemDict["a_type"] = "f64_r"
            problemDict["b_type"] = "f64_r"
            problemDict["c_type"] = "f64_r"
            problemDict["d_type"] = "f64_r"
            problemDict["compute_type"] = "f64_r"

        # get GEMM function and matrix orientation: transA/B, a/b/c/d/compute_type
        if ("NN" in gemmtype):
            problemDict["transA"] = "N"
            problemDict["transB"] = "N"
        elif ("NT" in gemmtype):
            problemDict["transA"] = "N"
            problemDict["transB"] = "T"
        elif ("TN" in gemmtype):
            problemDict["transA"] = "T"
            problemDict["transB"] = "N"
        elif ("TT" in gemmtype):
            problemDict["transA"] = "T"
            problemDict["transB"] = "T"

        if "SB" in gemmtype:
            otherParams = {"iters": 5000, "cold_iters": 10000}
        else:
            otherParams = {"iters": 10000, "cold_iters": 20000}

        #initialization
        if (initialization=='hpl'):
            init = {"initialization": "hpl"}
        elif (initialization=='trig'):
            init = {"initialization": "trig_float"}
        elif initialization== 'int':
            init = {"initialization": "rand_int"}

        # # check if the library is General Batched based on the library name
        # generalBatched = True if "_GB.yaml" in os.path.split(args.libLogic)[-1] else False

        # create rocBLAS-bench call for each size
        for size in matrices[gemmtype]:
 
            sizeDict = {}
            # M/N/K, batch_count, lda/b/c/d
            if ("SB" in gemmtype): # for strided
                sizeDict["M"] = size[0]
                sizeDict["N"] = size[1]
                sizeDict["batch_count"] = size[2]
                sizeDict["K"] = size[3]
                sizeDict["lda"] = size[4]
                sizeDict["ldb"] = size[5]
                sizeDict["ldc"] = size[6]
                sizeDict["ldd"] = size[6]
                sizeDict["alpha"] = size[7]
                sizeDict["beta"] = size[8]
            else: 
                sizeDict["M"] = size[0]
                sizeDict["N"] = size[1]
                # sizeDict["batch_count"] = size[2]
                sizeDict["K"] = size[2]
                sizeDict["lda"] = size[3]
                sizeDict["ldb"] = size[4]
                sizeDict["ldc"] = size[5]
                sizeDict["ldd"] = size[5]
                sizeDict["alpha"] = size[6]
                sizeDict["beta"] = size[7]
            #else: general batched

            params = {}
            params.update(problemDict)
            params.update(sizeDict)
            params.update(init)
            params.update(otherParams)

            bench.append(params)


        # write output
        with open(benchFile, "w") as f:
            if len(bench) > 0:
                yaml.dump(bench, f, yamlDumper, default_flow_style=None, sort_keys=False, width=5000)

def main():
    args = parseBenchCofnig()    

    # check if input exists
    if not os.path.isfile(args.input): 
       raise FileNotFoundError("{0} input file does not exist!".format(args.input))

    matrices,count,duplicate_count = readfile_rocblas_layer(args.input, args.blas_layer)

    createOutput(args.output, matrices,count,duplicate_count, args.blas_layer)
    
    # create rocblas-bench input file
    if args.bench != '': 
        verify = True if args.verify else False
        create_rocBLAS_bench(args.bench, matrices, verify, args.initialization)

    print("Done!")

if __name__ == "__main__":
    main()
