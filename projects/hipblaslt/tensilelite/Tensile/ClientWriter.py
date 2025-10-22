################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import inspect
import os
import subprocess
import shlex
import shutil

from pathlib import Path
from enum import Enum
from glob import glob

from Tensile.SolutionStructs.Problem import ProblemType, ProblemSizesMock, ProblemSizesMockDummy
from Tensile.SolutionStructs import ActivationArgs, BiasTypeArgs, FactorDimArgs
from Tensile.Toolchain.Component import Assembler

import rocisa

from . import ROOT_PATH
from . import LibraryIO
from Tensile.Common import ensurePath, print1, printExit, printWarning, ClientExecutionLock,\
                           LIBRARY_LOGIC_DIR, LIBRARY_CLIENT_DIR
from Tensile.Common.Architectures import isaToGfx
from Tensile.Common.GlobalParameters import globalParameters
from .TensileCreateLibrary import copyStaticFiles
from .Contractions import FreeIndex, BatchIndex
from .Contractions import ProblemType as ContractionsProblemType

class DataInitName(Enum):
  Zero = 0
  One = 1
  Two = 2
  Random = 3
  NaN = 4
  Inf = 5
  BadInput = 6
  BadOutput = 7
  SerialIdx = 8
  SerialDim0 = 9
  SerialDim1 = 10
  Identity = 11
  TrigSin = 12
  TrigCos = 13
  TrigAbsSin = 14
  TrigAbsCos = 15
  RandomNarrow = 16
  NegOne = 17
  Max = 18
  DenormMin = 19
  DenormMax = 20
  RandomNegPosLimited = 21
  TrigIndSin = 23
  TrigIndCos = 24
  TrigIndAbsSin = 25
  TrigIndAbsCos = 26

class ClientLogLevel(Enum):
  Error = 0
  Terse = 1
  Verbose = 2
  Debug = 3


################################################################################
# Main
################################################################################
def main(config, assembler: Assembler, cCompiler: str, isaInfoMap, outputPath: Path, deviceId: int, gfxName: str):

  libraryLogicPath = ensurePath(outputPath / LIBRARY_LOGIC_DIR)
  clientLibraryPath = ensurePath(outputPath / LIBRARY_CLIENT_DIR)
  sourcePath = ensurePath(clientLibraryPath / "source")
  copyStaticFiles(sourcePath)

  ##############################################################################
  # Read Logic Files
  ##############################################################################
  logicFiles = [os.path.join(libraryLogicPath, f) for f \
      in os.listdir(libraryLogicPath) \
      if (os.path.isfile(os.path.join(libraryLogicPath, f)) \
      and os.path.splitext(f)[1]==".yaml")]
  print1("LogicFiles: %s" % logicFiles)
  functions = []
  functionNames = []

  # Get rocIsa path, remove this when subprocess is removed
  module_path = os.path.dirname(inspect.getfile(rocisa))
  env = os.environ.copy()
  if 'PYTHONPATH' in env:
    if not module_path in env['PYTHONPATH']:
        env["PYTHONPATH"] = module_path + ":" + env["PYTHONPATH"]
  else:
    env["PYTHONPATH"] = module_path

  createLibraryScript = getBuildClientLibraryScript(clientLibraryPath, libraryLogicPath, str(assembler.path), isaToGfx(list(isaInfoMap.keys())[0]))
  subprocess.run(shlex.split(createLibraryScript), env=env, cwd=clientLibraryPath)
  coList = glob(os.path.join(clientLibraryPath, "library/*.co"))
  yamlList = glob(os.path.join(clientLibraryPath, "library/*.yaml"))

  clientParametersPaths = []
  splitGSU = False
  printSolutionRejectionReason = True
  printIndexAssignmentInfo = False
  for logicFileName in logicFiles:
    (scheduleName, _, problemType, _, exactLogic, newLibrary) \
        = LibraryIO.parseLibraryLogicFile(logicFileName,
                                          assembler,
                                          splitGSU,
                                          printSolutionRejectionReason,
                                          printIndexAssignmentInfo,
                                          isaInfoMap,
                                          globalParameters["LazyLibraryLoading"])
    functions.append((scheduleName, problemType))
    functionNames.append("tensile_%s" % (problemType))
    problemSizes = ProblemSizesMock(exactLogic) if exactLogic else ProblemSizesMockDummy()
    if len(problemType["BiasDataTypeList"]) > 0:
      biasTypeArgs = BiasTypeArgs(problemType, [problemType["BiasDataTypeList"][0]])
    else:
      biasTypeArgs = ""

    activationEnums = [[{'Enum': 'relu'}]]
    factorDimEnums = [0]
    # Reading the activation args from the LibraryClient section in the config YAML.
    # Example: enable relu and gelu activation and using none to run without activation
    #    LibraryClient:
    #      - ActivationArgs:
    #        - [Enum: none]
    #        - [Enum: gelu]
    #        - [Enum: relu]
    icacheFlushArgs = [False,]
    if len(config) > 0:
      for lc in config[0:]:
        if "ActivationArgs" in lc:
          activationEnums = lc["ActivationArgs"]
          break
        if "FactorDimArgs" in lc:
          factorDimEnums = lc["FactorDimArgs"]
        if "ICacheFlush" in lc:
          icacheFlushArgs = lc["ICacheFlush"]
    isForAll = True if problemType["ActivationType"] in ['all', 'hipblaslt_all'] else False
    activationArgs = ActivationArgs(problemType, activationEnums) if isForAll else ""
    factorDimArgs = FactorDimArgs(problemType, factorDimEnums)

    clientParametersPaths.append(writeClientConfig(
                                  forBenchmark=False,
                                  solutions=None,
                                  problemSizes=problemSizes,
                                  biasTypeArgs=biasTypeArgs,
                                  factorDimArgs=factorDimArgs,
                                  activationArgs=activationArgs,
                                  icacheFlushArgs=icacheFlushArgs,
                                  stepName=str(ProblemType(problemType, False)),
                                  stepBaseDir=str(clientLibraryPath),
                                  newLibrary=newLibrary,
                                  configBase="ClientParameters_%s"%str(ProblemType(problemType, False)),
                                  codeObjectFiles=coList,
                                  deviceId=deviceId,
                                  gfxName=gfxName,
                                  tileAwareSelection=False,
                                  libraryFile=yamlList[0]))

  forBenchmark = False
  problemSizes = None

  ##############################################################################
  # Run Build Script
  ##############################################################################
  # if redo=true, clobber the build directory

  if globalParameters["ForceRedoLibraryClient"]:
    shutil.rmtree(os.path.join(clientLibraryPath, "build"), \
        ignore_errors=True)

  forBenchmark = False
  enableTileSelection = False
  returncode = runClient(libraryLogicPath, forBenchmark, enableTileSelection, str(assembler.path), cCompiler, clientLibraryPath, clientParametersPaths)

  return returncode

################################################################################
# Write Run Script
################################################################################
def runNewClient(scriptPath, clientParametersPath, cxxCompiler: str, cCompiler: str, clientBuildDir=None):

  clientExe = getClientExecutablePath()
  iniFile = "--config-file={}".format(clientParametersPath)
  args = [clientExe, iniFile]

  try:
    subprocess.run(args, check=True)
  except (subprocess.CalledProcessError, OSError) as e:
    printWarning("ClientWriter Benchmark Process exited with error: {}".format(e))


def runClient(libraryLogicPath, forBenchmark, enableTileSelection, cxxCompiler: str, cCompiler: str, outputPath, configPaths=None):

  buildPath = ensurePath(outputPath / "build")

  runScriptName = writeRunScript(buildPath, forBenchmark, enableTileSelection, cxxCompiler, cCompiler, buildPath, configPaths)
  with ClientExecutionLock(globalParameters["ClientExecutionLockPath"]):
    process = subprocess.Popen(runScriptName, cwd=buildPath)
    process.communicate()

  if process.returncode:
    printWarning("ClientWriter Benchmark Process exited with code %u" % process.returncode)

  return process.returncode

def getBuildClientLibraryScript(buildPath, libraryLogicPath, cxxCompiler, targetGfx):
  import io
  runScriptFile = io.StringIO()

  callCreateLibraryCmd = ROOT_PATH + "/bin/TensileCreateLibrary"

  if not globalParameters["LazyLibraryLoading"]:
    callCreateLibraryCmd += " --no-lazy-library-loading"

  if globalParameters.get("AsmDebug", False):
    callCreateLibraryCmd += " --asm-debug"

  if globalParameters["KeepBuildTmp"]:
    callCreateLibraryCmd += " --keep-build-tmp"

  if globalParameters["DisableAsmComments"]:
    callCreateLibraryCmd += " --disable-asm-comments"

  callCreateLibraryCmd += " --architecture=" + targetGfx
  callCreateLibraryCmd += " --code-object-version=" + globalParameters["CodeObjectVersion"]
  callCreateLibraryCmd += " --cxx-compiler=" + cxxCompiler
  callCreateLibraryCmd += " --library-format=" + globalParameters["LibraryFormat"]

  callCreateLibraryCmd += " %s" % libraryLogicPath
  callCreateLibraryCmd += " %s" % buildPath #" ../source"
  callCreateLibraryCmd += " %s\n" % globalParameters["RuntimeLanguage"]

  runScriptFile.write(callCreateLibraryCmd)

  return runScriptFile.getvalue()


def writeRunScript(path, forBenchmark, enableTileSelection, cxxCompiler: str, cCompiler: str, buildDir, configPaths=None):
  if configPaths is None:
    configPaths = []

    configPaths.append(os.path.join(buildDir, "../source/ClientParameters.ini"))
    if enableTileSelection is True:
      configPaths.append(os.path.join(buildDir, "../source/ClientParameters_Granularity.ini"))

  # create run.bat or run.sh which builds and runs
  runScriptName = os.path.join(buildDir, \
    "run.%s" % ("bat" if os.name == "nt" else "sh") )
  runScriptFile = open(runScriptName, "w")
  if os.name != "nt":
    runScriptFile.write("#!/bin/bash\n\n")

  runScriptFile.write("set -ex\n")


  if forBenchmark:
    if os.name == "nt":
      runScriptFile.write(os.path.join(globalParameters["CMakeBuildType"], \
          "client.exe") )
    else:
      if globalParameters["PinClocks"] and globalParameters["ROCmSMIPath"]:
        runScriptFile.write("%s -d 0 --setfan 255 --setsclk 7\n" % globalParameters["ROCmSMIPath"])
        runScriptFile.write("sleep 1\n")
        runScriptFile.write("%s -d 0 -a\n" % globalParameters["ROCmSMIPath"])

      runScriptFile.write("set +e\n")


    if globalParameters["DataInitTypeA"] == -1 :
        globalParameters["DataInitTypeA"] = globalParameters["DataInitTypeAB"]
    if globalParameters["DataInitTypeB"] == -1 :
        globalParameters["DataInitTypeB"] = globalParameters["DataInitTypeAB"]

    runScriptFile.write("ERR1=0\n")

    clientExe = getClientExecutablePath()
    for configFile in configPaths:
      runScriptFile.write("{} --config-file {}\n".format(clientExe, configFile))
    runScriptFile.write("ERR2=$?\n\n")

    runScriptFile.write("""
ERR=0
if [[ $ERR1 -ne 0 ]]
then
    echo one
    ERR=$ERR1
fi
if [[ $ERR2 -ne 0 ]]
then
    echo two
    ERR=$ERR2
fi
""")

    if os.name != "nt":
      if globalParameters["PinClocks"] and globalParameters["ROCmSMIPath"]:
        runScriptFile.write("%s -d 0 --resetclocks\n" % globalParameters["ROCmSMIPath"])
        runScriptFile.write("%s -d 0 --setfan 50\n" % globalParameters["ROCmSMIPath"])
  else:
    for configFile in configPaths:
      runScriptFile.write("{} --config-file {} --best-solution 1\n".format(getClientExecutablePath(), configFile))

  if os.name != "nt":
    runScriptFile.write("exit $ERR\n")
  runScriptFile.close()
  if os.name != "nt":
    os.chmod(runScriptName, 0o777)
  return runScriptName


def toCppBool(yamlBool):
  return "true" if yamlBool else "false"

def getMaxSolutionSizes(solutions, solutionSummationSizes):

  maxK = max(solutionSummationSizes)
  maxMT0 = 0
  maxMT1 = 0
  for solution in solutions:

    wg = solution["WorkGroup"]
    tt = solution["ThreadTile"]
    mt0 = wg[0] * tt[0]
    mt1 = wg[1] * tt[1]

    if (mt0 > maxMT0):
      maxMT0 = mt0

    if (mt1 > maxMT1):
      maxMT1 = mt1

  return [maxMT0, maxMT1, maxK]

def checkConstStride(constStrideMap, keyIdx):
  finalVal = None
  for (mapIdx, val) in constStrideMap:
    if keyIdx == mapIdx:
      finalVal = val
  #print ("idx=", keyIdx, "=", finalVal)
  return finalVal


def problemSizeParams(problemType, problem, factorDim):

    numIndices = len(problemType.indices)
    rv = []

    if problem.stridesA:
        astrides = list(problem.stridesA)
    else:
        astrides = [-1] * problemType.aDims
    for sc in problemType.setConstStrideA:
        index = problemType.indices[sc[0]]
        if type(index) == FreeIndex:
            assert(index.isA)
            astrides[index.i] = sc[1]
        else:
            astrides[index.a] = sc[1]

    if problem.stridesB:
      bstrides = list(problem.stridesB)
    else:
      bstrides = [-1] * problemType.bDims
    for sc in problemType.setConstStrideB:
        index = problemType.indices[sc[0]]
        if type(index) == FreeIndex:
            assert(not index.isA)
            bstrides[index.i] = sc[1]
        else:
            bstrides[index.b] = sc[1]

    if problem.stridesC:
      cstrides = list(problem.stridesC)
    else:
      cstrides = [-1] * problemType.cDims

    if problem.stridesD:
      dstrides = list(problem.stridesD)
    else:
      dstrides = [-1] * problemType.dDims

    if len(problem.sizes) == numIndices:
        None
    elif len(problem.sizes) == numIndices + 4:
        # FIXME-problem, this is Exact format with strides tacked onto sizes as 4 extra pams
        # should just set problem.stride* appropriately when reading the Yaml and not deal with extra fields here
        if astrides[1] == -1:
          astrides[1] = problem.sizes[numIndices+2]
        elif astrides[1] != problem.sizes[numIndices+2]:
          raise RuntimeError("problem-specified lda(%u) conflicts with setConstStrideA(%u)" % \
              (astrides[1], problem.sizes[numIndices+2]))

        if bstrides[1] == -1:
          bstrides[1] = problem.sizes[numIndices+3]
        elif bstrides[1] != problem.sizes[numIndices+3]:
          raise RuntimeError("problem-specified ldb(%u) conflicts with setConstStrideB(%u)" % \
              (bstrides[1], problem.sizes[numIndices+3]))

        if cstrides[1] == -1:
          cstrides[1] = problem.sizes[numIndices+1]

        if dstrides[1] == -1:
          dstrides[1] = problem.sizes[numIndices+0]

    else:
        raise RuntimeError(
            "Invalid number of problem type indices: {0} - Indices: {1}, problemSize: {2}".format(len(problem.sizes), numIndices,
            ', '.join(map(str, problem.sizes))))

    problemSizeArg = ('problem-size', ','.join(map(str, problem.sizes[:numIndices])))
    rv.insert(0, problemSizeArg)

    rv.append(('a-strides', ",".join(map(str, astrides))))
    rv.append(('b-strides', ",".join(map(str, bstrides))))
    if cstrides:
      rv.append(('c-strides', ",".join(map(str, cstrides))))
    if dstrides:
      rv.append(('d-strides', ",".join(map(str, dstrides))))
      if problemType.useE:
          rv.append(('e-strides', ",".join(map(str, dstrides))))
    if problemType.useBias:
      length = problem.sizes[0]
      err_str = "M"
      if problemType.sparse:
        if len(factorDim) > 1:
          length = max(problem.sizes[0], problem.sizes[1])
          err_str = "max(M,N)"
        elif 1 in factorDim:
          length = problem.sizes[1]
          err_str = "N"
      biasstrides = [1, length, 0]
      for sc in problemType.setConstStrideBias:
        index = problemType.indices[sc[0]]
        if type(index) == BatchIndex:
            biasstrides[2] = sc[1]
      if biasstrides[2] == -1:
        biasstrides[2] = length
      elif biasstrides[2] != 0 and biasstrides[2] < length:
        raise RuntimeError("problem-specified bias stride(%u) must >= %s (%u)" % \
              (biasstrides[2], err_str, length))
      rv.append(('bias-strides', ",".join(map(str, biasstrides))))

    return rv

def dataInitParams(problemType):
    initA = globalParameters['DataInitTypeA']
    initB = globalParameters['DataInitTypeB']
    initC = globalParameters['DataInitTypeC']
    initD = globalParameters['DataInitTypeD']
    initE = globalParameters['DataInitTypeE']
    initAlpha = globalParameters['DataInitTypeAlpha']
    initBeta  = globalParameters['DataInitTypeBeta']
    initBias  = globalParameters['DataInitTypeBias']
    initScaleA  = globalParameters['DataInitTypeScaleA']
    initScaleB  = globalParameters['DataInitTypeScaleB']
    initScaleC  = globalParameters['DataInitTypeScaleC']
    initScaleD  = globalParameters['DataInitTypeScaleD']
    initScaleAlphaVec  = globalParameters['DataInitTypeScaleAlphaVec']

    if not problemType.useBeta:
        initBeta = 0

    if initA == -1: initA = globalParameters['DataInitTypeAB']
    if initB == -1: initB = globalParameters['DataInitTypeAB']

    return [('init-a',             DataInitName(initA).name),
            ('init-b',             DataInitName(initB).name),
            ('init-c',             DataInitName(initC).name),
            ('init-d',             DataInitName(initD).name),
            ('init-e',             DataInitName(initE).name),
            ('init-alpha',         DataInitName(initAlpha).name),
            ('init-beta',          DataInitName(initBeta).name),
            ('init-bias',          DataInitName(initBias).name),
            ('init-scaleA',        DataInitName(initScaleA).name),
            ('init-scaleB',        DataInitName(initScaleB).name),
            ('init-scaleC',        DataInitName(initScaleC).name),
            ('init-scaleD',        DataInitName(initScaleD).name),
            ('init-scaleAlphaVec', DataInitName(initScaleAlphaVec).name)]

def boundsCheckName(mode):
    if mode == 0: return 'Disable'
    if mode == 1: return 'NaN'
    if mode == 2: return 'GuardPageFront'
    if mode == 3: return 'GuardPageBack'
    if mode == 4: return 'GuardPageAll'

def pruneModeName(mode):
    if mode == 0: return 'PruneRandom'
    if mode == 1: return 'PruneXX00'
    if mode == 2: return 'PruneX0X0'
    if mode == 3: return 'Prune0XX0'
    if mode == 4: return 'PruneX00X'
    if mode == 5: return 'Prune0X0X'
    if mode == 6: return 'Prune00XX'

def writeClientConfigIni(forBenchmark, problemSizes, biasTypeArgs, factorDimArgs, activationArgs, icacheFlushArgs, problemType, sourceDir, codeObjectFiles, resultsFileName, parametersFilePath, deviceId: int, gfxName: str, libraryFile=None, probSolMap={}):

    assert os.path.exists(sourceDir), f"sourceDir={sourceDir} does not exist"

    with open(parametersFilePath, "w") as f:
        def param(key, value):
            f.write("{}={}\n".format(key, value))

        if libraryFile is None:
          libraryFilename = "TensileLibrary.yaml" if globalParameters["LibraryFormat"] == "yaml" else "TensileLibrary.dat"
          libraryFile = os.path.join(sourceDir, "library", libraryFilename)
        param("library-file", libraryFile)

        for coFile in codeObjectFiles:
            if 'gfx' not in coFile or gfxName in coFile:
                param("code-object", os.path.join(sourceDir,coFile))

        param('results-file', resultsFileName)
        param('performance-metric', globalParameters["PerformanceMetric"])
        param('problem-identifier', problemType.operationIdentifier)
        param('compute-input-type', problemType.computeInputType.toName())
        param('a-type',     problemType.aType.toName())
        param('b-type',     problemType.bType.toName())
        param('c-type',     problemType.cType.toName())
        param('d-type',     problemType.dType.toName())
        if problemType.useE:
            param('e-type',     problemType.eType.toName())
        if problemType.outputAmaxD:
            param('amaxD-type',     problemType.amaxDType.toName())
        param('alpha-type', problemType.alphaType.toName())
        param('beta-type',  problemType.betaType.toName())
        param('f32-xdl-math-op', problemType.f32XdlMathOp.toName())
        param('activation-compute-type', problemType.activationComputeDataType.toName())
        param('use-gradient', problemType.useGradient)
        param('use-bias',   problemType.useBias)
        param('bias-source',   problemType.biasSrcWhiteList[0])
        param('use-e', problemType.useE)
        param('output-amaxD', problemType.outputAmaxD)
        param('use-scaleAB',   problemType.useScaleAB)
        param('use-scaleCD',   problemType.useScaleCD)
        param('use-scaleAlphaVec',   problemType.useScaleAlphaVec)
        param('swizzle-tensor-a', problemType.swizzleTensorA)
        param('swizzle-tensor-b', problemType.swizzleTensorB)
        if biasTypeArgs:
          for btype in biasTypeArgs.biasTypes:
            param('bias-type-args',  btype.toName())
        if factorDimArgs:
          for fdim in factorDimArgs.factorDims:
            param('factor-dim-args', fdim)


        if icacheFlushArgs:
          for opt in icacheFlushArgs:
            param('icache-flush-args', opt)

        param('sparse',   problemType.sparse)
        param('high-precision-accumulate', problemType.highPrecisionAccumulate)
        param('strided-batched', problemType.stridedBatched)
        param('grouped-gemm', problemType.groupedGemm)

        probIdx = 0
        for problem in problemSizes.problems:
            for key,value in problemSizeParams(problemType, problem, factorDimArgs.factorDims):
                param(key,value)
            if probIdx in probSolMap:
                solutionIdx = probSolMap[probIdx]
                param('prob-sol-map', str(f'{probIdx},{solutionIdx}'),)
            probIdx += 1

        if activationArgs:
          for setting in activationArgs.settingList:
            param('activation-enum-args', setting.activationEnum.toEnum())
        param('activation-type', problemType.activationType.toEnum())
        param('activation-no-guard', problemType.activationNoGuard)
        if globalParameters["DataInitValueActivationArgs"]:
          param('activation-additional-args', ','.join(map(str, globalParameters["DataInitValueActivationArgs"])))

        param("device-idx",               deviceId)

        param("init-seed",                globalParameters["DataInitSeed"])

        for key,value in dataInitParams(problemType):
            param(key, value)

        param("c-equal-d",                globalParameters["CEqualD"])

        if globalParameters["PrintTensorA"]:
          param("print-tensor-a",         1)
        if globalParameters["PrintTensorB"]:
          param("print-tensor-b",         1)
        if globalParameters["PrintTensorC"]:
          param("print-tensor-c",         1)
        if globalParameters["PrintTensorD"]:
          param("print-tensor-d",         1)
        if globalParameters["PrintTensorRef"]:
          param("print-tensor-ref",       1)
        if globalParameters["PrintTensorBias"]:
          param("print-tensor-bias",      1)
        if globalParameters["PrintTensorScaleAlphaVec"]:
          param("print-tensor-scale-alpha-vec",      1)
        if globalParameters["PrintTensorAmaxD"]:
          param("print-tensor-amaxd",      1)
        if globalParameters["DumpTensors"]:
          param("dump-tensors",           1)
        if globalParameters["ExitOnFails"] > 1:
          param("exit-on-error", 1)

        param('prune-mode',               pruneModeName(int(globalParameters["PruneSparseMode"])))
        param("bounds-check",             boundsCheckName(int(globalParameters["BoundsCheck"])))
        param("print-valids",             globalParameters["ValidationPrintValids"])
        param("print-max",                globalParameters["ValidationMaxToPrint"])
        param("num-benchmarks",           globalParameters["NumBenchmarks"])

        numElementsToValidate = globalParameters["NumElementsToValidate"]
        if not forBenchmark:
         if globalParameters["NumElementsToValidateWinner"] == -1 or numElementsToValidate == -1:
           numElementsToValidate = -1
         else:
           numElementsToValidate = max(globalParameters["NumElementsToValidateWinner"], globalParameters["NumElementsToValidate"])
        param("num-elements-to-validate", numElementsToValidate)
        param("num-enqueues-per-sync",    globalParameters["EnqueuesPerSync"])
        param("max-enqueues-per-sync",    globalParameters["MaxEnqueuesPerSync"])
        param("num-syncs-per-benchmark",  globalParameters["SyncsPerBenchmark"])
        param("skip-slow-solution-ratio", globalParameters["SkipSlowSolutionRatio"])
        param("use-gpu-timer",            globalParameters["KernelTime"])
        param("hardware-monitor",         globalParameters["HardwareMonitor"])
        param("num-warmups",              globalParameters["NumWarmups"])
        param("min-flops-per-sync",       globalParameters["MinFlopsPerSync"])
        param("sleep-percent",            globalParameters["SleepPercent"])
        param("perf-l2-read-hits",        globalParameters["PerfModelL2ReadHits"])
        param("perf-l2-write-hits",       globalParameters["PerfModelL2WriteHits"])
        param("perf-l2-read-bw-mul",      globalParameters["PerfModelL2ReadBwMul"])
        param("perf-read-efficiency",     globalParameters["PerfModelReadEfficiency"])
        param("csv-export-extra-cols",    globalParameters["CSVExportWinner"])
        param("csv-merge-same-problems",  globalParameters["CSVMergeSameProblemID"])
        param("log-level",                ClientLogLevel(globalParameters["ClientLogLevel"]).name)
        param("max-workspace-size",       globalParameters["MaxWorkspaceSize"])
        param("PrintWinnersOnly",         globalParameters["PrintWinnersOnly"])
        param("granularity-threshold",    globalParameters["GranularityThreshold"])
        param("pristine-on-gpu",          globalParameters["PristineOnGPU"])

        param("library-update-file",      globalParameters["LibraryUpdateFile"])
        param("library-update-comment",   globalParameters["LibraryUpdateComment"])

        param("use-user-args",            globalParameters["UseUserArgs"])
        param("rotating-buffer-size",     globalParameters["RotatingBufferSize"])
        param("rotating-buffer-mode",     globalParameters["RotatingMode"])


def writeClientConfig(
      forBenchmark,
      solutions,
      problemSizes,
      biasTypeArgs,
      factorDimArgs,
      activationArgs,
      icacheFlushArgs,
      stepName,
      stepBaseDir,
      newLibrary,
      codeObjectFiles,
      tileAwareSelection,
      deviceId: int,
      gfxName: str,
      configBase = "ClientParameters",
      libraryFile = None,
      probSolMap = {}
    ):

    sourceDir = os.path.join(stepBaseDir, "source")

    if tileAwareSelection:
      filename = os.path.join(sourceDir, "%s_Granularity.ini"%configBase)
    else:
      filename = os.path.join(sourceDir, "%s.ini"%configBase)

    if len(newLibrary.solutions)==0:
      raise RuntimeError ("No valid solutions found")

    resultsFileName = None
    if tileAwareSelection:
      resultsFileName = os.path.join(stepBaseDir, "../Data", stepName+"_Granularity.csv")
    else:
      resultsFileName = os.path.join(stepBaseDir, "../Data", stepName+".csv")

    newSolution = next(iter(newLibrary.solutions.values()))
    writeClientConfigIni(forBenchmark, problemSizes, biasTypeArgs, factorDimArgs, activationArgs, icacheFlushArgs, newSolution.problemType, sourceDir, codeObjectFiles, resultsFileName, filename, deviceId, gfxName, libraryFile, probSolMap)

    return filename

def CreateBenchmarkClientParametersForSizes(libraryRootPath, problemSizes, dataFilePath, configFile, deviceId, gfxName, problemTypeDict=None):

    libraryPath = os.path.join(libraryRootPath, "library")
    libraryFiles = [os.path.join(libraryPath, f) for f in os.listdir(libraryPath)]
    codeObjectFiles = [f for f in libraryFiles if f.endswith("co")]

    if problemTypeDict:
      problemType = ContractionsProblemType.FromOriginalState(problemTypeDict)
    else:
      # if the we can library contains meta data then we can get the problem type this data
      metaDataFilePath = os.path.join(libraryPath, "metadata.yaml")
      if not os.path.exists(metaDataFilePath):
        printExit ("meta data file %s does not exist" % metaDataFilePath)
      metaData = LibraryIO.read(metaDataFilePath)
      problemTypeDict = metaData["ProblemType"]
      problemType = ContractionsProblemType.FromOriginalState(problemTypeDict)

    writeClientConfigIni(True, problemSizes, "", "", "", "", problemType, libraryRootPath, codeObjectFiles, dataFilePath, configFile, deviceId, gfxName)

def getClientExecutablePath():
  clientExe = globalParameters.get("PrebuiltClient")

  if not os.path.isfile(clientExe):
    raise FileNotFoundError(
        f"Tensile client executable not found at '{clientExe}'.\n"
        "Please ensure the client is built or provide a valid path using the --prebuilt-client flag.\n"
        "To build, run: `invoke build-client` (you may need to `pip3 install invoke` first).\n"
        "For custom cmake build instructions, please refer to the README in next-cmake."
    )
  return clientExe
