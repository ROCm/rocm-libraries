################################################################################
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import argparse
from collections import defaultdict
import glob
import itertools
import multiprocessing as mp
import os
import re
import subprocess
try:
    import yaml
except ImportError:
    assert 0 and \
        "You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions."

from dataclasses import dataclass, field
try:
    from tqdm import tqdm
except ImportError:
    assert 0 and \
            "You must install tqdm."
from typing import List
try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader
    assert 0 and "CSafeLoader not installed. Fallback to SafeLoader."

#####################################################
# Parameters
#####################################################
globalParameters = {}
globalParameters["BuildDir"] = ""
globalParameters["WorkingDir"] = {}
globalParameters["WorkingDir"]["Bench"] = "0_Bench"
globalParameters["WorkingDir"]["LogicYaml"] = "1_LogicYaml"

defaultBenchOptions = {"ProblemType": {
    "TransposeA": 0,
    "TransposeB": 0,
    "ComputeInputDataType": "s",
    "ComputeDataType": "s",
    "DataTypeC": "s",
    "DataTypeD": "s",
    "UseBias": False
}, "TestConfig": {
    "ColdIter": 20,
    "Iter": 100,
    "AlgoMethod": "all",
    "RequestedSolutions": 2, # Only works in AlgoMethod heuristic
    "SolutionIndex": None, # Only works in AlgoMethod index
    "ApiMethod": "cpp",
    "RotatingBuffer": 512,
    "Device": 0,
}, "TuningParameters": {
    "SplitK": [0]
}, "ProblemSizes": []}
defaultCreateLogicOptions = {}  # Currently unused

#####################################################
# Functions
#####################################################
#########
# Need to find a way to call these functions from Tensile package.
def ensurePath(path):
  try:
    os.makedirs(path)
  except FileExistsError:
    pass
  except OSError:
    str1 = "Failed to create directory \"%s\" " % (path)
    assert 0 and str1
  return path

def readYaml(filename):
    try:
        with open(filename, "r") as f:
            data = yaml.load(f, yamlLoader)
            return data
    except:
        str1 = "Failed to read yaml file %s"%filename
        assert 0 and str1

def writeYAML(filename, data, **kwargs):
    """Writes data to file in YAML format."""
    # set default kwags for yaml dump
    if "explicit_start" not in kwargs:
        kwargs["explicit_start"] = True
    if "explicit_end" not in kwargs:
        kwargs["explicit_end"] = True
    if "default_flow_style" not in kwargs:
        kwargs["default_flow_style"] = None

    with open(filename, "w") as f:
        yaml.dump(data, f, **kwargs)
#########

def dataType2Bench(dataType):
    if dataType == "H":
        return "f16_r"
    elif dataType == "B":
        return "bf16_r"
    elif dataType == "S":
        return "f32_r"
    elif dataType == "FP8N":
        return "f8_fnuz_r"
    elif dataType == "FP8":
        return "f8_r"
    else:
        assert 0

def computeType2Bench(dataTypeA, dataTypeB, computeInputDataType, computeDataType):
    needCvt = False
    if (dataTypeA != computeInputDataType) or (dataTypeB != computeInputDataType):
        needCvt = True
    if computeDataType == "S":
        if needCvt:
            if computeInputDataType == "B":
                return "f32_bf16_r"
            else:
                return "f32_f16_r"
        else:
            return "f32_r"
    else:
        assert 0

def findExact(config):
    print("Running benchmarks")
    # Write default values
    for keyOuter in defaultBenchOptions:
        if keyOuter not in config:
            config[keyOuter] = defaultBenchOptions[keyOuter]
        else:
            for key in defaultBenchOptions[keyOuter]:
                if config[keyOuter] == None:
                    config[keyOuter] = {}
                if key not in config[keyOuter]:
                    config[keyOuter][key] = defaultBenchOptions[keyOuter][key]
    # Sort and remove duplicated
    config["ProblemSizes"].sort()
    config["ProblemSizes"] = [i for i, _ in itertools.groupby(config["ProblemSizes"])]
    # Fix format
    if "DataTypeA" not in config["ProblemType"]:
        config["ProblemType"]["DataTypeA"] = config["ProblemType"]["ComputeInputDataType"]
    if "DataTypeB" not in config["ProblemType"]:
        config["ProblemType"]["DataTypeB"] = config["ProblemType"]["ComputeInputDataType"]
    assert config["ProblemType"]["DataTypeC"] == config["ProblemType"]["DataTypeD"]
    config["ProblemType"]["TransposeA"] = "T" if config["ProblemType"]["TransposeA"] else "N"
    config["ProblemType"]["TransposeB"] = "T" if config["ProblemType"]["TransposeB"] else "N"
    config["ProblemType"]["DataTypeA"] = config["ProblemType"]["DataTypeA"].upper()
    config["ProblemType"]["DataTypeB"] = config["ProblemType"]["DataTypeB"].upper()
    config["ProblemType"]["DataTypeC"] = config["ProblemType"]["DataTypeC"].upper()
    config["ProblemType"]["DataTypeD"] = config["ProblemType"]["DataTypeD"].upper()
    config["ProblemType"]["ComputeInputDataType"] = config["ProblemType"]["ComputeInputDataType"].upper()
    config["ProblemType"]["ComputeDataType"] = config["ProblemType"]["ComputeDataType"].upper()
    config["TestConfig"]["AlgoMethod"] = config["TestConfig"]["AlgoMethod"].lower()
    config["TestConfig"]["ApiMethod"] = config["TestConfig"]["ApiMethod"].lower()
    if config["TestConfig"]["AlgoMethod"] == "heuristic":
        if config["TestConfig"]["RequestedSolutions"] < 2:
            assert 0 and "RequestedSolutions must > 2 if Algomethod == heuristic"
    elif config["TestConfig"]["AlgoMethod"] == "index":
        if config["TestConfig"]["SolutionIndex"] == None:
            assert 0 and "SolutionIndex cannot be None if Algomethod == index"
    else:
        config["TestConfig"]["SolutionIndex"] = 0

    # Convert to TL format. ex. F8H_HHS
    aType = dataType2Bench(config["ProblemType"]["DataTypeA"])
    bType = dataType2Bench(config["ProblemType"]["DataTypeB"])
    cType = dataType2Bench(config["ProblemType"]["DataTypeC"])
    dType = dataType2Bench(config["ProblemType"]["DataTypeD"])
    computeType = computeType2Bench(config["ProblemType"]["DataTypeA"],
                                    config["ProblemType"]["DataTypeB"],
                                    config["ProblemType"]["ComputeInputDataType"],
                                    config["ProblemType"]["ComputeDataType"])
    if config["ProblemType"]["DataTypeA"] != config["ProblemType"]["DataTypeB"] or \
        config["ProblemType"]["DataTypeA"] != config["ProblemType"]["ComputeInputDataType"]:
        gemm_type = "%s%s_%s%s%s"%(config["ProblemType"]["DataTypeA"],
                                   config["ProblemType"]["DataTypeB"],
                                   config["ProblemType"]["ComputeInputDataType"],
                                   config["ProblemType"]["DataTypeD"],
                                   config["ProblemType"]["ComputeDataType"])
    else:
        gemm_type = "%s%s%s"%(config["ProblemType"]["ComputeInputDataType"],
                              config["ProblemType"]["DataTypeD"],
                              config["ProblemType"]["ComputeDataType"])
    if config["ProblemType"]["UseBias"]:
        gemm_type += "_Bias"

    execBenchPath = globalParameters["BuildDir"] + "/clients/staging/hipblaslt-bench"

    for size in config["ProblemSizes"]:
        filename = "result_%s%s_%s_%dx%dx%dx%d.txt"%(config["ProblemType"]["TransposeA"],
                                                     config["ProblemType"]["TransposeB"],
                                                     gemm_type,
                                                     size[0],
                                                     size[1],
                                                     size[2],
                                                     size[3])
        print("--Running size: %s"%(filename))
        command = [execBenchPath,
                "--print_kernel_info",
                "--device", str(config["TestConfig"]["Device"]),
                "--transA", config["ProblemType"]["TransposeA"],
                "--transB", config["ProblemType"]["TransposeB"],
                "--a_type", aType,
                "--b_type", bType,
                "--c_type", cType,
                "--d_type", dType,
                "--compute_type", computeType,
                "--algo_method", config["TestConfig"]["AlgoMethod"],
                "--api_method", config["TestConfig"]["ApiMethod"],
                "--requested_solution", str(config["TestConfig"]["RequestedSolutions"]),
                "--solution_index", str(config["TestConfig"]["SolutionIndex"]),
                "-j", str(config["TestConfig"]["ColdIter"]), "-i", str(config["TestConfig"]["Iter"]),
                "-m", str(size[0]), "-n", str(size[1]), "-k", str(size[3]), "--batch_count", str(size[2])]

        if config["ProblemType"]["UseBias"]:
            command.append("--bias_vector")

        if config["TestConfig"]["RotatingBuffer"] > 0:
            command.append("--rotating")
            command.append(str(config["TestConfig"]["RotatingBuffer"]))

        if "SplitK" in config["TuningParameters"]:
            for splitk in config["TuningParameters"]["SplitK"]:
                if splitk < 0 or splitk > 255:
                    assert 0 and "splitk's range is 0~255"
                command.append("--splitk")
                command.append(str(splitk))

        filePath = os.path.abspath(globalParameters["WorkingDir"]["Bench"] + "/" + filename)
        with open(filePath, "w") as f:
            subprocess.run(command, stdout=f)

@dataclass
class yamlListInfo:
    problemSizes: List[int] = field(init=False)
    localSolutionIndex: int = -1
    tflops: float = 0.0
    splitK: int   = 0

    def __post_init__(self):
        self.problemSizes = []

def fetchDataFromLogic(yamlFilePath, infoList):
    data = readYaml(yamlFilePath)
    # Skip exact yaml files
    libraryType = None
    if len(data) > 11 and data[11]:
        libraryType = data[11]
    else:
        str1 = "Library logic file {} is missing required field matching property." \
                .format(yamlFilePath)
        assert 0 and str1

    str1 = ""
    # index 5: solution
    # index 7: exactLogic
    # index 8: rangeLogic
    solutionList = data[5]
    newSolutionList = []
    local2NewLocalTable = {}
    for info in infoList:
        solution = solutionList[info.localSolutionIndex]
        if ((solution["GlobalSplitU"] == info.splitK) or (info.splitK == 0)) and (libraryType == "Equality"):
            str1 += "Equality solution found for %s, skipping.\n"%solution["SolutionNameMin"]
            continue
        gsu = solution["GlobalSplitU"] if info.splitK == 0 else info.splitK
        key = (info.localSolutionIndex, gsu)
        if key not in local2NewLocalTable:
            newSolutionList.append(solutionList[info.localSolutionIndex])
            local2NewLocalTable[key] = len(newSolutionList) - 1
            assert newSolutionList[-1]["SolutionIndex"] == info.localSolutionIndex
            newSolutionList[-1]["SolutionIndex"] = local2NewLocalTable[key]
            if newSolutionList[-1]["GlobalSplitU"] != gsu:
                oldStr = "_GSU" + str(newSolutionList[-1]["GlobalSplitU"])
                newStr = "_GSU" + str(gsu)
                if oldStr in newSolutionList[-1]["SolutionNameMin"]:
                    newSolutionList[-1]["SolutionNameMin"] = newSolutionList[-1]["SolutionNameMin"].replace(oldStr, newStr)
                newSolutionList[-1]["GlobalSplitU"] = gsu
    data[5] = newSolutionList
    exactLogicList = []
    for info in infoList:
        solution = solutionList[info.localSolutionIndex]
        gsu = solution["GlobalSplitU"] if info.splitK == 0 else info.splitK
        key = (info.localSolutionIndex, gsu)
        if key in local2NewLocalTable:
            exactLogicList.append([info.problemSizes, [local2NewLocalTable[key], info.tflops]])
    data[7] = exactLogicList
    data[8] = None
    data[11] = "Equality"

    yamlFileName = os.path.abspath(globalParameters["WorkingDir"]["LogicYaml"] + "/" + os.path.basename(yamlFilePath))
    writeYAML(yamlFileName, data, explicit_start=False, explicit_end=False)
    return str1

def CreateExact(config):
    print("Creating exact logic")
    tableFile = globalParameters["BuildDir"] + "/library/MatchTable.yaml"
    print("--Reading matching table: %s"%tableFile)
    tableData = readYaml(tableFile)
    print("--Reading bench files")
    benchList = glob.glob(globalParameters["WorkingDir"]["Bench"] + "/result_*_*_*x*x*x*.txt")
    yamlList = defaultdict(list)
    for benchFile in benchList:
        print(" --Found file %s"%benchFile)
        solutionIndex = -1
        perfData = []
        with open(benchFile, "r") as f:
            fList = f.readlines()[::-1]
            for num, line in enumerate(fList):
                if "Winner:" in line:
                    perfData = fList[num-3:num]
                    break

        if len(perfData) != 3:
            str1 = "Winner/ solution index not found in file %s"%(benchFile)
            assert 0 and str1

        # Get perf results
        headers = perfData[2].split(',')
        values  = perfData[1].split(',')
        solutionIndex = int(re.search(r'\d+', perfData[0]).group())
        # Get values
        m   = int(values[headers.index('m')])
        n   = int(values[headers.index('n')])
        b   = int(values[headers.index('batch_count')])
        k   = int(values[headers.index('k')])
        lda = int(values[headers.index('lda')])
        ldb = int(values[headers.index('ldb')])
        ldc = int(values[headers.index('ldc')])
        ldd = int(values[headers.index('ldd')])
        tflops = float( values[headers.index('hipblaslt-Gflops')] )
        splitK = int(values[headers.index('splitK')]) if 'splitK' in headers else 0

        data = tableData[solutionIndex]
        yamlFilePath           = data[0]
        yamlLocalSolutionIndex = data[1]
        yli = yamlListInfo()
        yli.problemSizes = [m, n, b, k, lda, ldb, ldc, ldd]
        yli.localSolutionIndex = yamlLocalSolutionIndex
        yli.tflops = tflops
        yli.splitK = splitK
        yamlList[yamlFilePath].append(yli)

    pool = mp.Pool(8)
    jobs = []
    for yamlFilePath, infoList in yamlList.items():
        job = pool.apply_async(fetchDataFromLogic, (yamlFilePath, infoList, ))
        jobs.append(job)

    logInfo = []
    for job in tqdm(jobs, "Writing logic yaml files"):
        str1 = job.get()
        if str1:
            logInfo.append(str1)
    for infoStr in logInfo:
        print("[Info] " + infoStr)

    pool.close()
    pool.join()

# script run from commandline
if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("config_file", type=os.path.realpath, nargs="+",
            help="Benchmark config.yaml file")
    argParser.add_argument("build_path", type=os.path.realpath, \
            help="Path to hipblaslt build_path (build/release)")
    argParser.add_argument("output_path", type=os.path.realpath, \
            help="Path to conduct benchmark and write output files")
    args = argParser.parse_args()

    # Update global parameters
    globalParameters["BuildDir"] = args.build_path
    globalParameters["WorkingDir"]["RootDir"] = ensurePath(args.output_path)
    globalParameters["WorkingDir"]["Bench"] = ensurePath(os.path.abspath(globalParameters["WorkingDir"]["RootDir"] + "/" + globalParameters["WorkingDir"]["Bench"]))
    globalParameters["WorkingDir"]["LogicYaml"] = ensurePath(os.path.abspath(globalParameters["WorkingDir"]["RootDir"] + "/" + globalParameters["WorkingDir"]["LogicYaml"]))

    configPaths = args.config_file
    config = readYaml(configPaths[0])

    if "Bench" in config:
        findExact(config["Bench"])
    if "CreateLogic" in config:
        CreateExact(config["CreateLogic"])
