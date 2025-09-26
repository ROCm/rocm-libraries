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

from . import __version__
from . import Parallel
from .Utilities.ConditionalImports import print, TENSILE_TERM_COLORS
from .Utilities.Toolchain import getVersion
from collections import OrderedDict

from copy import deepcopy
from .AsmCaps import getCapabilitiesCache
from typing import Any, NamedTuple, Optional, Tuple, Dict

import math
import os.path
import subprocess
import sys
import time
import warnings
import re

startTime = time.time()

ParallelMap = Parallel.ParallelMap

IsaVersion = Tuple[int, int, int]

class SemanticVersion(NamedTuple):
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

class DeveloperWarning(Warning):
    """Custom warning for Tensile developers.

    This warning can be safely ignored when running any Tensile applications as a user.
    """


# print level
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints

################################################################################
# Global Parameters
################################################################################
globalParameters = OrderedDict()
workingDirectoryStack = []

########################################
# common
########################################
globalParameters["MinimumRequiredVersion"] = "0.0.0" # which version of tensile is required to handle all the features required by this configuration file
globalParameters["PerformanceMetric"] = "DeviceEfficiency" # performance metric for benchmarking; one of {DeviceEfficiency, CUEfficiency}
globalParameters["PrintLevel"] = 1                # how much info to print in generator. 0=none, 1=standard, 2=add code commands, 3=verbose
globalParameters["ClientLogLevel"] = 3            # the log level of client. 0=Error, 1=Terse, 2=Verbose, 3=Debug (Aligned with ResultReporter.hpp)
# benchmarking
globalParameters["KernelTime"] = False            # T=use device timers, F=use host timers
globalParameters["PreciseKernelTime"] = True      # T=On hip, use the timestamps for kernel start and stop rather than separate events.  Can provide more accurate kernel timing.  For GlobalSplitU kernels, recommend disabling this to provide consistent
# timing between GSU / non-GSU kernels
globalParameters["CodeFromFiles"] = True          # if False byte arrays will be generated during Benchmarking phase as before
globalParameters["SortProblems"] = False          # sort problems by size; else use order in YAML file
globalParameters["PinClocks"] = False             # T=pin gpu clocks and fan, F=don't
globalParameters["HardwareMonitor"] = True        # False: disable benchmarking client monitoring clocks using rocm-smi.
globalParameters["NumBenchmarks"] = 1             # how many benchmark data points to collect per problem/solution
globalParameters["NumWarmups"] = 0                # how many warmup runs to perform before benchmark
globalParameters["SyncsPerBenchmark"] = 1         # how iterations of the stream synchronization for-loop to do per benchmark data point
globalParameters["EnqueuesPerSync"] = 1           # how many solution enqueues to perform per synchronization
globalParameters["SleepPercent"] = 300            # how long to sleep after every data point: 25 means 25% of solution time. Sleeping lets gpu cool down more.
globalParameters["FlushCount"] = 1                # Number of copies of arrays to allocate for cache flushing in timing code.
                                                  # Functions are called iters times in a timing loop.
                                                  # If the problem memory footprint is small enough, then arrays will be cached.
                                                  # flush_count can be used to prevent caching.
                                                  # For example, for sgemm with transA=transB=N:
                                                  # problem_memory_footprint = (m*k + k*n + m*n) * sizeof(float).
                                                  # To flush arrays before reusing set:
                                                  # flush_count >= 1 + cache_size / problem_memory_footprint
                                                  # Note that in the calculation of flush_count any padding from leading
                                                  # dimensions are not loaded to cache and not included in the problem_memory_footprint.
                                                  # If you specify flush_count you cannot also specify flush_memory_size)
globalParameters["FlushMemorySize"] = 0           # Bytes of memory that will be occupied by arrays. Used only in timing code for cache flushing. Set to greater than
                                                  # cache size, so that arrays are flushed from cache before they are reused. When the size of arrays (the problem_memory_footprint)
                                                  # is smaller than flush_memory_size, then flush_count copies of arrays are allocated where:
                                                  # flush_count = flush_memory_size / problem_memory_footprint.
                                                  # For sgemm with transA=transB=N
                                                  # problem_memory_footprint = (m*k + k*n + m*n) * sizeof(float). Note that any padding from leading
                                                  # dimensions are not loaded to cache and not included in the problem_memory_footprint.
                                                  # If you specify flush_memory_size you cannot also specify flush_count)
                                                  # Also note that Tensile allocates enough memory once at setup to accommodate
                                                  # the largest problem. Similarly, the largest problem will be used to calculate flush_count.
                                                  # Configs with largely contrasting sizes may not guarantee cache eviction for the smaller problems

# validation
globalParameters["NumElementsToValidate"] = 128   # number of elements to validate, 128 will be evenly spaced out (with prime number stride) across C tensor
globalParameters["BoundsCheck"] = 0   # Bounds check
#1: Perform bounds check to find out of bounds reads/writes.  NumElementsToValidate must be -1.
#2: Perform bounds check by front side guard page
#3: Perform bounds check by back side guard page
#4: Perform bounds check by both back and front side guard page

globalParameters["ValidationMaxToPrint"] = 4      # maximum number of mismatches to print
globalParameters["ValidationPrintValids"] = False # print matches too
# steps
globalParameters["ForceRedoBenchmarkProblems"] = True # if False and benchmarking already complete, then benchmarking will be skipped when tensile is re-run
globalParameters["ForceRedoLibraryLogic"] = True      # if False and library logic already analyzed, then library logic will be skipped when tensile is re-run
globalParameters["ForceRedoLibraryClient"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run

# Compare CPU reference convolution model vs golden tensor contraction model
# Useful to test if conversion from tensor contraction is working as expected
# In this mode, the filter,stride,dilation are specified in the problem type.
# If the problem type uses constant Filter,Stride,Dilation,Pad* (ie these are not 'N'), then the
# specified constant MUST match the dimension in the problem or the tensile runtime will assert.
# The batch size, spatial dims, Cin, and Cout are always read from the problem description.
globalParameters["ConvolutionVsContraction"] = False

globalParameters["ShowProgressBar"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["SolutionSelectionAlg"] = 1          # algorithm to determine which solutions to keep. 0=removeLeastImportantSolutions, 1=keepWinnerSolutions (faster)
globalParameters["ExpandRanges"] = True          # expand ranges into exact configs before writing logic file.  False ignores ranges.
globalParameters["GenerateSourcesAndExit"] = False # Exit after kernel source generation.
globalParameters["WavefrontWidth"] = 64     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["ExitOnFails"] = 1     # 1: Exit after benchmark run if failures detected.  2: Exit during benchmark run.
globalParameters["CpuThreads"] = -1  # How many CPU threads to use for kernel generation. N=min(nproc,N). Setting CpuThreads < 1 (ie: 0 or -1) will use max threads (nproc)

# even if error occurs in kernel generation (ie due to resource overflow),
# generate the kernel source anyway.  Tensile will also attempt to run
# the kernel.  Useful to examine and debug overflow errors.
globalParameters["ForceGenerateKernel"] = 0

########################################
# optimization knob controls
########################################

globalParameters["UnrollLoopEfficiencyEnable"] = False   # if True split(S) MAC&LDS in each unroll iteration into n smaller groups..

########################################
# less common
########################################
globalParameters["CMakeBuildType"] = "Release"            # whether benchmark clients and library client should be release or debug
globalParameters["PrintSolutionRejectionReason"] = False  # when a solution is marked as invalid, print why
globalParameters["LibraryFormat"] = "msgpack"             # set library backend (either yaml or msgpack)
globalParameters["EmbedLibrary"] = None                   # whether library should be embedded or not

# True/False: CSV will/won't export WinnerGFlops, WinnerTimeUS, WinnerIdx, WinnerName.
# TODO - if no side-effect, we can set default to True. This can make analyzing "LibraryLogic" (AddFromCSV) faster
globalParameters["CSVExportWinner"] = False

# (When NumBenchmarks > 1). True: CSV will merge the rows of same Problem-ID. False: Each problem will write out "NumBenchmarks" rows
#   In old client - No effect, since in old client, CSV file only exports the last benchmark, somehow is not correct because the previous benchmarks are discarded
#   In new client - csv file exports "NumBenchmarks" rows for every problem. This also make the later analyzing slower
#                   Set this to "True" can merge the rows for same problem, hence can reduce the csv file size and speed up the later analyzing
# TODO - if side-effect, we can set default to True. This can make "getResults()" / "AddFromCSV()" faster
globalParameters["CSVMergeSameProblemID"] = False

# how to initialize tensor data
# serial-in-u will use a sequence that increments in the K dimension
# This is a predictable patterns that can be checked as the kernel runs to detect
# when the wrong data is being used.
# trig_float initializes with the sin function to have non-zero values in the mantissa
# and exponent. It cannot be used for int8 or int32. Need to use tensileAlmostEqual
# not tensileEqual for checking the result.
# See ClientWriter.py, the DataInitName(Enum) for a list of initialization patterns
#       - Problem-Independent: 0=0, 1=1, 2=2, 3=rand, 4=Nan, 5=Infinity, 6=BadInput(Nan), 7=BadOutput(Inf), 16=RandomNarrow
#       - Problem-dependent: 8=SerialID, 9=SerialDim0, 10=SerialDim1, 11=Identity, 12~15= Cos/Sin, Abs or Not
#       For A, B, C, D: All the InitMode (0~16) can be used
#       For Alpha/Beta: Only problem-independent init (0~7, 16) can be used,
#                       problem-dependent init (8~15) would cause a exception (Invalid InitMode) in New Client
globalParameters["DataInitTypeAB"] = 3
globalParameters["DataInitTypeA"] = -1
globalParameters["DataInitTypeB"] = -1
globalParameters["DataInitTypeC"]  = 3
globalParameters["DataInitTypeD"]  = 0
globalParameters["DataInitTypeAlpha"] = 2
globalParameters["DataInitTypeBeta"] = 2
globalParameters["CEqualD"] = False               # Set to true if testing for the case where the pointer to C is the same as D.
globalParameters["BufferOffsetA"] = 0             # data offset of buffer A
globalParameters["BufferOffsetB"] = 0             # data offset of buffer B
globalParameters["BufferOffsetC"] = 0             # data offset of buffer C
globalParameters["BufferOffsetD"] = 0             # data offset of buffer D

# build parameters
globalParameters["CMakeCXXFlags"] = ""            # pass flags to cmake
globalParameters["CMakeCFlags"] = ""              # pass flags to cmake
globalParameters["DebugKernel"] = False           # assembly only, kernel gets buffer for debug "printing"; kernel writes data to memory, gets copied to host and printed
globalParameters["LibraryPrintDebug"] = False     # solutions will print enqueue info when enqueueing a kernel

# debug for assembly
globalParameters["EnableAsserts"] = False         # Enable assembly debug assert
globalParameters["EnableDebugA"] = False          # Enable / Disable CheckValue1A
globalParameters["EnableDebugB"] = False          # Enable / Disable CheckValue1B
globalParameters["EnableDebugC"] = False          # Enable / Disable CheckValueC
globalParameters["ExpectedValueC"] = 16.0         # Expected C Value when CheckValueC, debug for Alpha*A*B
globalParameters["ForceCExpectedValue"] = False   # Force C to "DebugExpectedValueC", debug for global write
globalParameters["DebugSkipAtomic"] = False       # Reject kernels that contain atomics to only run non-atomic kernels
globalParameters["DebugSkipNonAtomic"] = False    # Reject kernels that do no contain atomics to only run atomic kernels

# Tensor printing controls:
globalParameters["PrintConvolutionUsage"] = 0      # Print Convolution usage info. 1=tensor fields,2=boilerplate info,4=print tensor mappings for specified ConvProblems
globalParameters["PrintTensorA"] = 0          # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0          # Print TensorB after initialization
globalParameters["PrintTensorC"] = 0          # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintTensorD"] = 0          # Print TensorD.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintTensorRef"] = 0          # Print reference tensor.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintIndexAssignments"] = 0      # Print the tensor index assignment info
globalParameters["PrintWinnersOnly"] = False      # Only print the solutions which become the fastest
globalParameters["DumpTensors"] = False        # If True, dump tensors to binary files instead of printing them.

# If PrintMax* is greater than the dimension, the middle elements will be replaced with "..."


# device selection
globalParameters["Platform"] = 0                  # select opencl platform
globalParameters["Device"] = 0                    # select hip device or opencl device within platform

# shouldn't need to change
globalParameters["DeviceLDS"] = 65536             # LDS bytes per CU, for computing occupancy
globalParameters["MaxLDS"] = 65536                # max LDS a kernel should attempt to use
globalParameters["MaxDepthU"] = 1024              # max DepthU value to allow
globalParameters["ShortNames"] = False            # on windows kernel names can get too long; =True will convert solution/kernel names to serial ids
globalParameters["MergeFiles"] = True             # F=store every solution and kernel in separate file; T=store all solutions in single file
globalParameters["NumMergedFiles"] = 1            # The number of files that kernels should be split between when merging

globalParameters["MaxFileName"] = 64              # If a file name would be longer than this, shorten it with a hash.
globalParameters["SupportedISA"] = [(8,0,3),
                                    (9,0,0), (9,0,6), (9,0,8), (9,0,10),
                                    (9,4,2), (9,5,0),
                                    (10,1,0), (10,1,1), (10,1,2), (10,3,0), (10,3,1), (10,3,2),
                                    (11,0,0), (11,0,1), (11,0,2), (11,0,3),
                                    (11,5,0), (11,5,1),
                                    (12,0,0), (12,0,1)] # assembly kernels writer supports these architectures

globalParameters["KeepBuildTmp"] = True                           # Do not remove build artifacts during the build process or build_tmp after build completes
globalParameters["GenerateManifestAndExit"] = False               # Output manifest file with list of expected library objects and exit
globalParameters["VerifyManifest"] = False                        # Verify manifest file against generated library files and exit.
globalParameters["BenchmarkProblemsPath"] = "1_BenchmarkProblems" # subdirectory for benchmarking phases
globalParameters["BenchmarkDataPath"] = "2_BenchmarkData"         # subdirectory for storing final benchmarking data
globalParameters["LibraryLogicPath"] = "3_LibraryLogic"           # subdirectory for library logic produced by analysis
globalParameters["LibraryClientPath"] = "4_LibraryClient"         # subdirectory for building example library client
globalParameters["ClientExecutionLockPath"] = None                # Path for a file lock to ensure only one client is executed at once.  filelock module is required if this is enabled.
globalParameters["LibraryUpdateFile"] = ""                        # File name for writing indices and speeds suitable for updating an existing library logic file
globalParameters["LibraryUpdateComment"] = False                  # Include solution name as a comment in the library update file
globalParameters["DictLibraryLogic"] = False

# internal, i.e., gets set during startup
globalParameters["CurrentISA"] = (0,0,0)
globalParameters["ROCmAgentEnumeratorPath"] = None      # /opt/rocm/bin/rocm_agent_enumerator
globalParameters["ROCmSMIPath"] = None                  # /opt/rocm/bin/rocm-smi
globalParameters["AssemblerPath"] = None                # /opt/rocm/llvm/bin/clang++
globalParameters["WorkingPath"] = os.getcwd()           # path where tensile called from
globalParameters["IndexChars"] =  "IJKLMNOPQRSTUVWXYZ"  # which characters to use for C[ij]=Sum[k] A[ik]*B[jk]
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))            # path to Tensile/Tensile.py
globalParameters["SourcePath"] = os.path.join(globalParameters["ScriptPath"], "Source") # path to Tensile/Source/
globalParameters["HipClangVersion"] = "0.0.0"

globalParameters["RuntimeLanguage"] = "HIP"

globalParameters["CodeObjectVersion"] = "default"
globalParameters["CxxCompiler"] = "amdclang++" if os.name != "nt" else "clang++"
globalParameters["CCompiler"] = "amdclang" if os.name != "nt" else "clang"
globalParameters["Architecture"] = "all"

# might be deprecated
globalParameters["ClientArgs"] = ""

# perf model
globalParameters["PerfModelL2ReadHits"] = 0.0
globalParameters["PerfModelL2WriteHits"] = 0.15
globalParameters["PerfModelL2ReadBwMul"] = 2
globalParameters["PerfModelReadEfficiency"] = 0.85

# limitation for training
globalParameters["MaxWorkspaceSize"] = 32 * 1024 * 1024 # max workspace for training (32M)

# control if a solution is run for a given problem
globalParameters["GranularityThreshold"] = 0.0

# directory where custom kernels are located
globalParameters["CustomKernelDirectory"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CustomKernels")

globalParameters["PristineOnGPU"] = True # use Pristine memory on Tensile training verification or not

globalParameters["SeparateArchitectures"] = False # write Tensile library metadata to separate files for each architecture

globalParameters["LazyLibraryLoading"] = False # Load library and code object files when needed instead of at startup

globalParameters["IgnoreAsmCapCache"] = False # Ignore checking for discrepancies between derived and cached asm caps

globalParameters["ExperimentalLogicDir"] = "/experimental/"

# Save a copy - since pytest doesn't re-run this initialization code and YAML files can override global settings - odd things can happen
defaultGlobalParameters = deepcopy(globalParameters)

# Translate GPU targets to filter filenames in Tensile_LOGIC directory
architectureMap = {
  'all':'_','gfx000':'none', 'gfx803':'r9nano', 'gfx900':'vega10', 'gfx900:xnack-':'vega10',
  'gfx906':'vega20', 'gfx906:xnack+':'vega20', 'gfx906:xnack-':'vega20',
  'gfx908':'arcturus','gfx908:xnack+':'arcturus', 'gfx908:xnack-':'arcturus',
  'gfx90a':'aldebaran', 'gfx90a:xnack+':'aldebaran', 'gfx90a:xnack-':'aldebaran',
  'gfx942':'aquavanjaram942', 'gfx942:xnack+':'aquavanjaram942', 'gfx942:xnack-':'aquavanjaram942',
  'gfx950':'gfx950', 'gfx950:xnack+':'gfx950', 'gfx950:xnack-':'gfx950',
  'gfx1010':'navi10', 'gfx1011':'navi12', 'gfx1012':'navi14',
  'gfx1030':'navi21', 'gfx1031':'navi22', 'gfx1032':'navi23', 'gfx1034':'navi24', 'gfx1035':'rembrandt',
  'gfx1100':'navi31', 'gfx1101':'navi32', 'gfx1102':'navi33', 'gfx1103':'gfx1103',
  'gfx1150':'strixpoint', 'gfx1151':'strixhalo',
  'gfx1200':'gfx1200',
  'gfx1201':'gfx1201'
}

def getArchitectureName(gfxName: str) -> Optional[str]:
  """Maps the provided Gfx architecture to its common name using the **architectureMap**.

  Args:
      gfxName: Gfx architecture name to map.

  Returns:
      Common name associated with the provided **gfxName** if it matches or is a substring
      of a key in **architectureMap**, otherwise None.
  """
  if gfxName in architectureMap:
    return architectureMap[gfxName]
  else:
    for archKey in architectureMap:
      if gfxName in archKey:
        return architectureMap[archKey]
    return None


################################################################################
# Enumerate Valid Solution Parameters
################################################################################
validWorkGroups = []
for numThreads in range(32, 1025, 32):
  for nsg in [ 1, 2, 4, 8, 16, 32, 64, 96, 128, 256 ]:
    for sg0 in range(1, numThreads//nsg+1):
      sg1 = numThreads//nsg//sg0
      if sg0*sg1*nsg == numThreads:
          workGroup = [sg0, sg1, nsg]
          validWorkGroups.append(workGroup)

validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + list(range(20, 256, 4))
validThreadTiles = []
for i in validThreadTileSides:
  for j in validThreadTileSides:
    validThreadTiles.append([i, j])

validActivationFormats = ('NCHW', 'NHWC', 'CNHW', 'NCDHW', 'NDHWC', 'CNDHW')
validWeightFormats = ('KCYX', "KYXC", "CKYX", "CYXK",  'KCZYX', 'CKZYX', 'CZYXK')
validMacroTileSides = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 6, 12, 24, 48, 96, 192, 384, 768 ]
validMacroTiles = []
validISA = [(0,0,0)]
validISA.extend(globalParameters["SupportedISA"])
depthUs = list(range(-16, 0))
depthUs.extend(list(range(2,globalParameters["MaxDepthU"]+1,1)))
for i in validMacroTileSides:
  for j in validMacroTileSides:
    validMacroTiles.append([i, j])

validMFMA = {}
validMFMA["H"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [16,16,32,1], [4,4,4,16]]
validMFMA["S"] = [[32,32,1,2], [32,32,2,1], [16,16,1,4], [16,16,4,1], [4,4,1,16]]
validMFMA["B"] = [[32,32,2,2], [32,32,4,1], [16,16,2,4], [16,16,8,1], [4,4,2,16]]
validMFMA["D"] = [[16,16,4,1], [4,4,4,4]]
validMFMA["B1k"] = validMFMA["H"]
validMFMA["C"] = validMFMA["S"]
validMFMA["Z"] = validMFMA["D"]
validMFMA["X"] = [[32,32,4,1], [16,16,8,1]]
validMFMA["F8"] = [[32,32,16,1], [16,16,32,1]]
validMFMA["B8"] = validMFMA["F8"]
validMFMA["F8B8"] = validMFMA["F8"]
validMFMA["B8F8"] = validMFMA["F8"]
validMFMA["I8_908"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["I8_940"] = [[32,32,4,2], [32,32,16,1], [16,16,4,4], [16,16,32,1], [4,4,4,16]]
validMFMA["I8"] = validMFMA["H"] + validMFMA["F8"]
validWMMA = [[16,16,16,1], ]
validTT = 64
validMFMA["_format9"] = []

for MFMA in [validMFMA["H"], validMFMA["S"], validMFMA["B"], validMFMA["D"], validMFMA["X"], validMFMA["F8"], validWMMA]:
  for MI in MFMA:
    for bm in range(int(math.log(MI[3],2))+1):
      for tt0 in range(1,validTT+1):
        for tt1 in range(1,validTT+1):
          for wave_m in range (3):
            for wave_n in range(3):
              validMFMA["_format9"].append([MI[0],MI[1],MI[2],MI[3],2**bm,tt0,tt1,2**wave_m, 2**wave_n])

validMatrixInstructions = [[], [-1]] + validMFMA["H"] + validMFMA["S"] + validMFMA["B"] + validMFMA["D"] + validMFMA["X"] + validMFMA["F8"]
validMatrixInstructions = validMatrixInstructions + validMFMA["_format9"]

# The supported typed GEMM, each entry is (Ti, To, Tc).
# DataType (Ti)        = The data-type of the input matrices: A/B
# DestDataType (To)    = The data-type of the output matrices: C/D
# ComputeDataType (Tc) = The data-type of computation: alpha/beta:
# CInternal: basically should == ComputeDataType

# Align the supported GEMM type with rocBLAS: [A/B/ C/D/ alpha/beta]
#   (rocblas/library/include/internal/rocblas_functions.h)
# GEMM (HPA=F, the data type of input, output, and computation are all the same.)
#   - HGEMM: [H/H/ H/H/ H/H]
#   - SGEMM: [S/S/ S/S/ S/S]
#   - DGEMM: [D/D/ D/D/ D/D]
#   - CGEMM: [C/C/ C/C/ C/C]
#   - ZGEMM: [Z/Z/ Z/Z/ Z/Z]
# GEMM_Ex: (HPA=T, Computation is in a higher precision data-type)
#   - GEMM_EX (HHS): [H/H/ H/H/ S/S]
#   - GEMM_EX (HSS): [H/H/ S/S/ S/S]
#   - GEMM_EX (BBS): [B/B/ B/B/ S/S]
#   - GEMM_EX (BSS): [B/B/ S/S/ S/S]
#   - GEMM_EX (I8II): [I8/I8/ I/I/ I/I]
#   - GEMM_EX (4xi8II): [4xi8/4xi8/ I/I/ I/I], tensile packs 4 i8 to 4xi8 with some restrictions
# This is used in SolutionStruct.py::checkIfSupportedGEMMType()
validGEMMTypes = [ ('D','D','D'), ('S','S','S'), ('Z','Z','Z'), ('C','C','C'), \
                   ('H','H','H'), ('H','H','S'), ('H','S','S'), \
                   ('B','B','S'), ('B','S','S'), \
                   ('I8','I','I'), ('4xi8','I','I'), \
                   ('F8','S','S'), ('B8','S','S'), \
                   ('F8B8','S','S'), ('B8F8', 'S', 'S'), \
                   ('F8','F8','S'), ('B8','B8','S'), \
                   ('F8B8','B8','S'), ('B8F8', 'B8', 'S'), \
                   ('F8','H','S'), ('B8','H','S'), \
                   ('F8B8','H','S'), ('B8F8','H','S') ]

# All HPA types are listed here (HPA=T). The name of the library logic files for these types is:
# *_TiToTc_BH*.yaml where Ti, Tc, and To are the data types of A/B, C/D, and computation, respectively.
# The name of the library logic files for non-HPA (HPA=F) types is: *_TiB*.yaml.
HPATypes = [ ('H','S','S'), ('H','H','S'), ('B','B','S'), ('B','S','S'), ('I8','I','I'), ('4xi8','I','I'), \
             ('F8','S','S'), ('B8','S','S'), ('F8B8','S','S'), ('B8F8', 'S', 'S'), \
             ('F8B8','B8','S'), ('B8F8', 'B8', 'S'), \
             ('F8','H','S'), ('B8','H','S'), ('F8B8','H','S'), ('B8F8','H','S'), \
             ('F8','F8', 'S'), ('B8', 'B8', 'S') ]

validParameters = {
    "LoopDoWhile":                [ False, True ], # Source. True=DoWhile, False=For loop
    "LoopTail":                   [ False, True ], # tail loop handles non multiples of unrolled summation loop

    # threads load elements from global into registers, then write from registers to LDS
    # these options affect those read/write patterns
    # coalesce-group=True  means adjacent threads will     read adjacent addresses; if the data needs to be transposed then adjacent threads will NOT write adjacent elements to LDS.
    # coalesce-group=False means adjacent threads will NOT read adjacent addresses; if the data needs to be transposed then adjacent threads will     write adjacent elements to LDS.
    # this parameter really only matters for transposing
    # =False means the L1 cache will do the transposing work and it is quite fast; then data is written coalesced (no bank conflicts) to LDS.
    # =True means the transpose will happen while writing to LDS, this usually has bank conflicts, but it appears the throughput is still fast enough to not slow the VALUs down.
    # it appears that the L1 cache can still achieve quite a bit of performance for GRCG=False, but overall it's usually faster to read coalesced
    "GlobalReadCoalesceGroupA":   [ False, True ],
    "GlobalReadCoalesceGroupB":   [ False, True ],

    # for transposes, this option governs how short-vectors should be read from global and written to lds
    # it is impossible to transpose data while operating on short-vectors for GlobalRead,LocalWrite and LocalRead; an odd number of those must be transposing and operating on vector components.
    # since data will be read from lds many more times than it will be written, data must always end up in lds such that short-vectors can be read from lds
    # =True means read short-vector from global and write its components to lds
    # =False means read vector components from global so that a full short-vector can be written to lds
    # both options were supported until a refactoring of the short-vector code (necessary to enable assembly) broke it. Since =True always seems to be faster, no time has been spend on fixing =False
    #  it may still work in source, but just not in assembly. The problem is the order in which elements are stored into vgprs, is different than the order in which they are written to lds. In source each
    #  loaded element gets a variable name which in independent of the order that they are written in the source code, but in assembly the values are just assigned vgprs in order and that order needs to be shuffles.
    "GlobalReadCoalesceVectorA":  [        True ], # FIXME =False worked before the vector refactor; fixing requires re-ordering load/store indices; but they aren't the faster option so not worth time right now
    "GlobalReadCoalesceVectorB":  [        True ],

    # original global read to lds is interlace, [w0,w1,w2,w3,w0,w1,w2,w3,w0,w1,w2,w3,w0,w1,w2,w3]
    # when WaveSeparateGlobalRead is enabled, LDS is divided to number of waves part.
    # each wave load a block memory to lds,     [w0,w0,w0,w0,w1,w1,w1,w1,w2,w2,w2,w2,w3,w3,w3,w3]
    # -1 is selected by logic, 0 disable, 1 enable.
    "WaveSeparateGlobalReadA":    [ 0, 1 ],
    "WaveSeparateGlobalReadB":    [ 0, 1 ],

    # directToLds (asyncDMA) feature do not work very efficiently for lower precisions fp16/bf16/i8. directToLds feature does not
    # support destination offset in LDS , no padding support to avoid LDS bank conflicts during data movement LDS->VGPR
    # This feature enumerates elements in summation Index dimension into different thread lanes during global fetch while
    # keeping memory fetch efficiency same as non directToLds and avoids bank conflicts when data moved from LDS->VGPR.
    # fragmenting threads required to fetch #elements in coalescing dimension. rather than using contiguous depthu/GLVW into
    # fragments occupying upper and lower 32 threads or fragment into 4x16 threads.
    # for higher precision like f64, The feature should also help to avoid LDS bank conflicts for Transpose data layout case
    # use primarily for direct to LDS feature with transpose data layout
    # for example A matrices = Transpose layout
    # ThreadSeparateGlobalReadA = 0   DepthU=64 GLVW=8 T0,T1,T2,T3,T4,T5,T6,T7  fetching  64 elements
    # ThreadSeparateGlobalReadA = 1   DepthU=64 GLVW=8 T0,T1,T2,T3, T32,T33,T34,T35
    # ThreadSeparateGlobalReadA = 2   DepthU=64 GLVW=8 T0,T1,T16,T17,T32,T33,T48,T49  fetching  64 elements
    # use =2 for 16x16x4xfp16 instructions
    # should work with WaveSeparateGlobalRead
    # Feature should help depthU*bpe requiring more than 4 threads.
    # SplitGlobalRead is integrated into ThreadSeparateGlobalRead

    "ThreadSeparateGlobalReadA":    [ 0, 1, 2, 4 ],
    "ThreadSeparateGlobalReadB":    [ 0, 1, 2, 4 ],

    # PrefetchGlobalRead = 1:
    # Requires 2X LDS space, and VGPRs for buffering data on way into LDS
    #   prefetch / double-buffer reads from global memory -> vgprs -> lds.
    #
    # PrefetchGlobalRead = 2:
    # Do another prefetch while writing data from vgpr to lds.
    #   prefetch / double-buffer reads from global memory -> vgprs --> lds.
    #                                                              |-> prefetch reads
    "PrefetchGlobalRead":         [ 0, 1, 2 ],

    # number of iteration prefetch local reads from lds to VGPRs buffer = PLR % LoopIter
    # number of VGPRs buffer = min(PLR+1,LoopIters)
    # LoopIters = DepthU / LocalSplitU
    # (LoopIters /= MatrixInstruction_K)
    # ex. MT64x128x16_MI32x32x4x2_PLR1, we'll have 4 LoopIters, prefetch read 1 iteration, with 2 VGPRs buffer (2=min(1+1,4))
    #     before loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[0], iter3:plr[0] MAC_r[1]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[0], iter3:       MAC_r[1]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR3, we'll have 4 LoopIters, prefetch read 3 iteration, with 4 VGPRs buffer (4=min(3+1,4))
    #     before loop:       plr[0] plr[1] plr[2]
    #           loop: iter0:plr[3] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[2], iter3:plr[2] MAC_r[3]
    #   no load loop: iter0:plr[3] MAC_r[0], iter1:       MAC_r[1], iter2:       MAC_r[2], iter3:       MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR5, we'll have 4 LoopIters, prefetch read 5%4=1 iteration, with 4 VGPRs buffer (4=min(5+1,4))
    #     before loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:plr[0] MAC_r[3]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:       MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR5_LRVW8, we'll have 4 LoopIters, prefetch read 5%4=1 iteration, with 4 VGPRs buffer (4=min(5+1,4)) , each read read 2 iterations
    #     before loop:       plr[0:1]
    #           loop: iter0:plr[2:3] MAC_r[0], iter1: MAC_r[1], iter2: MAC_r[2], iter3:plr[0:1] MAC_r[3]
    #   no load loop: iter0:plr[2:3] MAC_r[0], iter1: MAC_r[1], iter2: MAC_r[2], iter3:         MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR7, we'll have 4 LoopIters, prefetch read 7%4=3 iteration, with 4 VGPRs buffer (=min(7+1,4)) --> Exactly the same as PLR3
    #     before loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:plr[0] MAC_r[3]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:       MAC_r[3]
    "PrefetchLocalRead":          list(range(128+1)),

    # We use double LDS buffer when PrefetchGlobalRead.
    # While it reads data from LDS[0]/[1], it prefetch global data and writes to LDS[1]/[0]
    # If we can make sure all data are read from LDS to register before writing data to LDS, we can use 1 LDS buffer to save LDS memory.
    # this can help to generate Kernel that LDS usage originally exceed MaxLDS if using double LDS buffer,
    # or help to increase Occupancy.
    #     1 means: Force to use 1 LDS Buffer even with PrefetchGlobalRead
    #    -1 means: generator will use 1 LDS buffer only when LDS exceed MaxLDS
    # Use case:
    #    SIA2: 1LDSBuffer is set to 1 natively
    #    SIA3: 1LDSBuffer works only when PGR=True
    # TODO: optimize scheduling to support more cases.
    "1LDSBuffer": [-1 ,0, 1],

    # Split the unroll summation into multiple sections and combine the sections
    # GSU applies only to the unroll summation dimension
    "GlobalSplitU":               list(range(1, 4096+1)),

    # Chooses how to do GlobalSplitU:
    # - SingleBuffer: uses atomic operation to accumulate on one buffer
    # - MultipleBuffer: each GSU group writes to its own buffer and the postGSU accumulates the buffer
    # if GlobalSplitU=1, this parameter will be ignored (and will be set to SingleBuffer if it is
    # MultipleBuffer for consistency in lib logics).
    # GSU/GSUAlo can be used with all gemm types, except for I8II.
    # When GSU>1, we need extra kernels (other than the main assembly kernel) to do the computations. The language of these
    # kernels are HIP source and will be dropped in 1_BenchmarkProblems/Cijk_*/*_Final/source/Kernels.?pp:
    #   - pre-kernel: for Global Accumulation.
    #   - postGSU: for accumulating AxB and alpha*AxB+beta*C from the buffer.
    #
    # Table below shows the required kernels and workspace for each gemm function.
    # ------------------------------------------------------------------------------------------
    # | gemm function          | non-HPA gemm                  | HPA gemm (excluding int8)     |
    # | GSU Algorithm          | SingleBuffer | MultipleBuffer | SingleBuffer | MultipleBuffer |
    # | pre-kernel             | yes          | no             | yes          | no             |
    # | main kernel            | yes          | yes            | yes          | yes            |
    # | postGSU                | no           | yes            | yes          | yes            |
    # | _GlobalAccumulation    | None         | MultipleBuffer | SingleBuffer | MultipleBuffer |
    # | _WorkspaceSizePerElemC | 0            | Tc*GSU         | Tc           | Tc*GSU         |
    # | WorkspaceSize          | 0            | Tc*GSU*m*n     | Tc*m*n       | Tc*GSU*m*n     |
    # ------------------------------------------------------------------------------------------
    #
    # Note that the workspace in MultipleBuffer algo is used for accumulating the C matrix. The workspace for
    # HPA-SingleBuffer is used to convert the output from ComputeDataType to DestDataType.
    "GlobalSplitUAlgorithm":      ["SingleBuffer", "MultipleBuffer"],

    # When splitting up the summation between workgroups, there are two options for organizing which workgroup will do what
    # If we begin with N workgroups and set GSU=4, there will now be 4N workgroups
    # GSUWGMRR=False means workgroup 0,1,2,3 will all work on the same tile; =True means workgroup 0, N-1, 2N-1, 3N-1 will all work on the same tile
    "GlobalSplitUWorkGroupMappingRoundRobin":     [ False, True ],
    # GSUSARR=False means the 4 workgroups do whole chunks of the summation: k=0 -> K/4-1, k=K/4 -> 2K/4-1, k=2K/4 -> 3K/4-1, k=3K/4 -> 4K/4-1
    # GSUSARR=True means the 4 workgroups round robin split up the chunks of the summation: k=0 -> DU-1, 4DU -> 5DU-1, ...; k=1DU -> 2DU-1, 5DU -> 6DU-1...; ...
    "GlobalSplitUSummationAssignmentRoundRobin":  [ False, True ],

    # Enable atomic_add instruction for GlobalSplitU with SingleBuffer
    # So far, f32 only.
    # NOTE: This is not recommended
    "GlobalSplitUAtomicAdd":      [ False, True ],

    # in opencl for some compilers, performance improved by putting a memfence after each sub-iteration; it prevented the loads of one sub-iteration from being moved
    # into a prior iteration, which would help latency but it consumed more vgprs which was a net loss
    "UnrollMemFence":             [ False, True ],

    # not used yet; will refer to combining multiple reads into single instruction
    # such as ds_read_b32 -> ds_read2_b32
    # the pro is that it cuts in half the number of instructions
    # the con is that bits per offset is half, so arithmetic might be required to increment and reset offset vgprs
    "GlobalRead2A":               [ False, True ],
    "GlobalRead2B":               [ False, True ],
    "LocalWrite2A":               [ False, True ],
    "LocalWrite2B":               [ False, True ],
    "LocalRead2A":                [ False, True ],
    "LocalRead2B":                [ False, True ],

    # don't create a whole copy of the Unroll loop with loads removed - instead
    # use buffer limits to suppress global loads and ignore unnecessary ds_reads
    "SuppressNoLoadLoop":         [False, True],

    # For PrefetchGlobalRead=1, create a second copy of the unroll loop with
    # the LDS pointer swaps expanded into inline constants for LDS read and write instructions
    # This eliminates 4 vector XOR instructions used for pointer swap
    "ExpandPointerSwap":          [False, True],

    # Schedule global reads and global read increments into LocalRead iterations
    # Can reduce pressure on local read instruction dispatch queue
    # 0=perform global reads at start of instruction loop
    # 1=schedule into the local read instruction iterations
    "ScheduleGlobalRead":         [0, 1],

    # Schedule local writes into LocalRead iterations.
    # Can reduce pressure on local read instruction dispatch queue
    "ScheduleLocalWrite":         [0, 1],

    # Scheduling algorithm to use for each iteration:
    # 0 = minimal/no scheduling.  Global Read and increments, followed by local reads,
    # followed by local writes, followed by MACs
    "ScheduleIterAlg":            [0, 1, 2, 3],

    # Optimizing Local Write Vmcnt in PreLoop when PGR is on, especially for PAP
    # 0: no optimization, force wait vmcnt 0
    # 1: do optimization, in PAP, this can avoid ds_write waiting for previous global store
    # Can always be True, set to False for debugging or comparison
    "OptPreLoopVmcnt":            [False, True],

    # For MatrixInstruction and SIA3, number of GlobalReadInstruction between mfma
    # the purpose of this parameter is to control density of global read instruction scheduling
    # Scheduling global read back to back can have better memory efficiency
    # However, when full of vmem FIFO, it will block other instruction to be issued
    # Range from 0.01 to 32
    #         0.1 means 1 GR per 10 mfma
    #           5 means 5 GR per 1 mfma
    "GlobalReadPerMfma":       [ i/100 for i in range(1,3200)],
    #
    # For MatrixInstruction and SIA3, number of LocalWriteInstruction between mfma
    # the purpose of this parameter is to control density of local write instruction scheduling
    # In PGR1, we want to schedule local write more denser, so we can have more
    #          latency to hide global read
    # In PGR2, since LW is followed by GR, every LW has same whole loop latency
    #          to hide global read. We want to schedule LW less denser, can
    #          avoid full of vmem FIFO.
    # Range from 0.01 to 32
    #         0.1 means 1 LW per 10 mfma
    #           5 means 5 LW per 1 mfma
    # -1 will derived an optimized value internally
    # -2 will derived an optimized value and override LWPM silently (debug only, not recommended)
    "LocalWritePerMfma":       [ i/100 for i in range(1,3200)] + [ -1 ],

    # LDD Support
    # Allow LDD and StrideD to != LDC and StrideC for LDD <= LDC and LDD == M
    # TODO: remove. legacy logic yaml in rocblas contains true and false for this parameter
    # remove this parameter will cause two kernels have same.
    # so we can't remove it until we clean logic yaml in rocblas
    "LdcEqualsLdd":               [ False, True ],

    # Interleave alpha scale calculation with beta loads and address calcs - rather
    # than as a separate block of instructions
    "InterleaveAlpha":             [0, 1],

    # Create a copy of NoLoadLoop which interleaves the stores with the final mac
    # calculation and may perform other optimizations
    # 0 = no interleave
    # 1 = interleave one stores after required macs have completed execution
    # 2 = interleave two stores after required macs have completed execution
    "OptNoLoadLoop":               [0, 1, 2],

    # Prefetch across persistent kernel iterations - the no-load-loop computes the
    # tile assignment and next global read offset and launches the buffer loads for
    # the next tile in the sequence.
    "PrefetchAcrossPersistent":    [0, 1],

    # Changes the behavior of prefetch across persistent.
    # Mode 0 is default, works for all sizes
    # Mode 1 disables static tile setup for prefetch and merges prefetch with ord. noLoadLoop,
    "PrefetchAcrossPersistentMode": [0, 1],

    "BufferLoad":                 [ False, True ],
    "BufferStore":                [ False, True ],

    # Attempt to load directly from global memory into Vgpr.
    # Assembly only
    "DirectToVgprA":              [ False, True ],
    "DirectToVgprB":              [ False, True ],

    # Attempt to load directly from global memory into LDS.
    # Assembly only
    # Requires BufferLoad, assembler support for lds modifier on buffer
    # loads (checked automatically), GlobalVectorWidth=1 (this is hw
    # requirement) and A/B must not require any transpose.
    # DirectToLds reduces load latency and eliminates the
    # G2L registers used to stage data.  Also replaces the
    # local write offset with an SGPR.
    # For an 8x8 TT with PrefetchGlobalRead=1 this can save 33 VGPRs.
    #    - Requirements for DirectToLds=1:
    #      GlobalLoadVectorWidth * bpe should be 4
    #      TransposeLDS = 1 for TLU=0 case
    # old DirectToLds parameter is replaced with DirectToLdsA, B
    #"DirectToLds":                [ False, True ],
    "DirectToLdsA":                [ False, True ],
    "DirectToLdsB":                [ False, True ],

    # Load options:
    # (GRO = Global Read Offset)
    # BufferLoad=0:
    #  = Use global_load instructions with 64 bit GRO for each load
    #    + supports sizes up to 2^64
    #    - uses many VGPR for addressing
    #    - uses execmask+compares for edge detection
    # BufferLoad=1:
    #  = Use buffer load instructions with 32-bit offset
    #    + Less VGPRS (32b offset vs 64-bit) needed for addressing
    #    + Uses hardware buffer limit for edge detection
    #    - Limited range - the bot-right corner of macro-tile (plus padding=GRVW
    #        for shift-pointer, if ShiftPtr is required) must be within 2^32.
    #      ShiftPtrPad = MayShift ? GRVW*BPE : 0
    #      For TLU=1: Unroll*StrideA1 + ShiftPtrPad <= 2^32
    #      For TLU=0: MT*StrideA1 + ShiftPtrPad <= 2^32
    #      These conditions should be checked using Assert - TODO
    #  = UseSgprForGRO=1:
    #    + Attempt to use SGPR for Global Read Offsets.
    #    + Use one VGPR base GRO + many SGPR GRO rather than many VGPR GRO.
    #    + Each SGPR stores an offset from base GlobalReadOffset+0.
    #    - Requirements for UseSgprForGRO=1:
    #      - BufferLoad=1
    #      - Use appropriate Assert*ElementMultiple or GRVW=1 to eliminate need for ShiftPtr
    #        (UseSgprForGRO does not support ShiftPtr since ShiftPtr needs to potentially shift GRO)
    #  = KernelWriterAssembly also supports 64-bit 2D buffer size (see use64bPbcLimit)
    #    - Requires 4 instructions to move scalar limit and a couple SGPR
    #    - Enabled by default.  If the overhead matters we can add asserts/YAML parm to specialize
    #  = UseInstOffsetForGRO=1:
    #    + Attempt to use Instruction offset for Global Read Offsets.
    #    + This feature avoid updating m0 for subsequent GRO(s) for directToLds feature
    #    - Requirements for UseInstOffsetForGRO=1:
    #      - BufferLoad=1
    #      - DirectToLds=1

    #  converting m0 update from LocalWriteAddrSGpr using  is usually win
    # -1 attempt to use a heuristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseInstOffsetForGRO":              [ -1, 0, 1],


    # Converting VGPR GRO into SGPR GRO is usually a win
    # However, the mode may exhaust all available SGPR, in particular for large unroll
    # -1 attempt to use a heuristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseSgprForGRO":              [ -1, 0, 1],

    # Some work-items in the group may not participate in the final buffer load.
    # Allows more flexibility in choosing DepthU.
    # 1= allocate extra addressing vgpr for edge cases
    # 2= use temp vgpr inside unroll loop, may save 1 VPR if both A and B have a fractional edge but costs v_alu
    "FractionalLoad":             [ 0, 1, 2] ,

    # Use a 64-bit shadow limit register to allow buffers larger than 2^32 bytes
    "Use64bShadowLimit":   [ True, False],

    # Attempt to vectorize atomics
    # 1,2 : Number of elements to vectorize
    # -1 : Maximum supported value
    # This defines width of atomic_cmpswap (bpe(external) * VAW * 32bit = width of atomic_cmpswap (b32 or b64))
    # AtomicAdd case, only 1 supported
    "VectorAtomicWidth":          [ -1, 1, 2] ,

    # Assertion properties
    # These provide information or assertions that the problem size meets certain requirements
    # for sizes or alignments.  The kernel generator can use this information to produce
    # a kernel which uses those assertions to produce a faster kernel.
    #
    # If modifying or adding Assertions also change ProblemProperties class in TensileTypes.h

    # Kernel generator will assume that the summation size is some multiple of the element size
    # and uses this to optimize the kernel.
    # This can result in more efficient kernels, but requires runtime checking to ensure the specified
    # summation value meets the requirements.
    # (Recommended AF1EM value is 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertSummationElementMultiple>1:
    #  - If >=2 for half:
    #     - Tail loop loads can be vectorized 2X to use dword
    #     - Enables asm kernels on V20
    #     - Can use DirectToLds for both unroll and tail loops
    #  - Tail loop can be unrolled up to InnerUnroll amount if AssertSummationElementMultiple%InnerUnroll==0
    #  - GlobalSplitU>1 case:
    #   - Optimizations enabled by AssertSummationElementMultiple>1 will be adjusted as follows.
    #     ASEM%GSU == 0 and ASEM//GSU will be used for optimizations instead of ASEM
    #     For example, if ASEM is 8 and GSU is 2, K is multiple of 8 but K is divided by GSU.
    #     In that case, we can still guarantee K/GSU is multiple of 4 (= ASEM/GSU) and
    #     we can use ASEM//GSU=4 for optimizations
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertSummationElementMultiple": [1,2,4,8,16,32,64,128,256,512,1024],

    # Kernel generator will assume that the FreeIndex[0] size is some multiple of the element size
    # and uses this to optimize the kernel.
    # FreeIndex[0] is usually letter "I"
    # (Recommended AF0EM value for the best performance is 16 for I8, 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertFree0ElementMultiple>1:
    # Load optimizations:
    #  - For TLU=1 matrix, if AF1WM>=GLVW then can enable UseSgprForGRO
    #      - Reduces registers used for address calculations
    #      - Enables FractionalLoad for more flexibility in address calcs
    #      - Removes address shift/unshift code
    #    - UseSgprForGRO will only be enabled if all matrices meet assertion requirements.
    #
    # Store Optimizations:
    #  - Can vectorize stores in edge tiles.  Vector width can be up to AF0EM.
    #   (since C matrix is always coalesced in Free0 index direction and this assertion guarantees the index element multiple)
    #
    # TailLoop Optimizations:
    #  - enable wider global load with AF0EM > 1 for A + TLU, AF1EM > 1 for B + TLU
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree0ElementMultiple" : [1,2,4,8,16],

    # Kernel generator will assume that the FreeIndex[1] size is some multiple of the element size
    # and uses this to optimize the kernel.
    # FreeIndex[1] is usually letter "J"
    # (Recommended AF1EM value for the best performance is 16 for I8, 8 for half, 4 for single, 2 for double)

    # Optimizations enabled by AssertFree1ElementMultiple>1:
    #  - See above AssertFree0ElementMultiple "Load optimizations"

    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree1ElementMultiple" : [1,2,4,8,16],

    # Some kernels only work for certain sizes, see ProblemProperties in TensileTypes for exact defs
    "AssertMinApproxSize" : [0,1,2,3],


    # Assertions/Predicates that require stride to be specified value.
    # Dictionary of pairs of {position:constValue}
    # Unlike SetConstStride*, these use a position in the IndexAssignments* field:
    #   EX: "{2:0}"  means IndexAssignmentsB[2] must be 0 to run the solution.
    # Use this syntax to specify multiple Fork values in a YAML config file.

    #- AssertStrideAEqual:
    #  - {5: 2, 6: 2} # these are two AssertStrideAEqual predicates for the same solution.
    #  - {5: 2}       # this is a second solution generated with a single predicate.

    # Like other assertions, these are used when kernel is generated and checked before running kernel.
    "AssertStrideAEqual":  -1,

    "AssertStrideBEqual":  -1,

    "AssertStrideCEqual":  -1,
    "AssertStrideDEqual":  -1,

    # Assertions that require stride to be specified value.
    # Dictionary of pairs of {index, constValue}.
    # Index is a member of the global index assignments.
    "AssertSizeEqual":       -1,
    "AssertSizeGreaterThan": -1,
    "AssertSizeLessThan":    -1,
    "AssertSizeMultiple":    -1,

    # Assertions that require arithmetic intensity to be specified value.
    # Arithmetic intensity measures the ratio of computation to memory bandwidth required for a problem.
    # These predicates can be used to adjust solution selection compute-bound or memory-bound problems.
    "AssertAIGreaterThanEqual": -1,
    "AssertAILessThanEqual":    -1,

    #Assert values for alpha and beta
    "AssertBetaValue":       [False, 1, -1],
    "AssertAlphaValue":      [False, 1, -1],

    #Assert C==D
    "AssertCEqualsD": [False, True],

    # Generate code inside kernel to check Assertions on Tensor dimensions
    "CheckTensorDimAsserts":               [False, True],

    # Generate code inside kernel to check several dimension overflow cases, in particular around use of 32-bit calcs
    # 0 = no check, 1=checks for cases that should be avoided through assertions and kernel selection,
    # 2=checks for cases that should never happen
    "CheckDimOverflow":               [0,1,2],

    # Stagger the start summation position of the tiles.
    # Elements from the summation dimension are loaded at offsets rather than all starting at 0.
    # StaggerU is the max 'clicks' of StaggerUStride bytes where each wg starts ; see StaggerUMapping
    # for how the specific stagger for a given wg is determined.
    #
    # The tile assignment C are same as with StaggerOffset=0 ; the difference is the
    # order that the summation elements are added.
    # GRO will wrap back to the row start when the edge is reached.
    #
    # This can be effective for TLU=0 style matrices where the K dimension is a large power-of-2.
    # In this case the start of each row of the tile is separated by an exact power-of-2
    # which causes poor dram, cache, and tlb behavior.  V20 has 16 channels each 256 bytes wide.

    # StaggerU adjusts the start position in the summation (aka 'U') dimension
    # to avoid these conflicts.  Both A and B matrix start at the adjusted position.
    # If >0 specifies the offset in multiples of the macro-tile "unroll" dim
    #  - Higher values will spread traffic to more channels but provide less L2 re-use.
    #  - StaggerU and WorkGroupMapping interact and should be tuned together -
    #    The WGM controls how tiles are assigned in C matrix, while StaggerU controls where those
    #    tiles start reading their summation dim params.
    #  - StaggerU requires BufferLoad==1 and is silently ignored if BufferLoad==0
    "StaggerU":              [0,2,4,8,16,32,64],

    # Stride in bytes for each staggerU 'click'.
    # 256 is recommended since this is the width of memory channel (on gfx803,gfx900,gf906) - so
    # each click will start in a new memory channel and spread traffic among the 16 available channels.
    # For example StaggerUStride=256 and StaggerU=8 will use 8 unique starting points
    # in summation dimension, each offset by 256-bytes - provided the tensor dims are large
    # enough to support this.
    # StaggerUStride will be internally increased so it is an integer multiple of DepthU*BpeAB.
    # (the implementation requires this - the unroll iteration accesses data in steps of
    # DepthU*BPE
    # SUS=0 is only valid if SU=0
    "StaggerUStride":        [0,16,32,64,128,256,512,1024,2048],

    # How the tile assignment (wg0, wg1, wg2) controls the initial StaggerU offset:
    # 0: Use wg0
    # 1: Use wg1
    # 2: Use wg2
    # 3: Use wgSerial, wgSerial = wg0 + wg1 * nwg0 + wg2 * (nwg0 * nwg1)
    # 4: Debug mode, offset each tile max allowed StaggerU.  This just moves hot-spot
    #    to a different bank since all workgroups still start at same point.
    "StaggerUMapping":       [0,1,2,3,4],


    # 0=don't use magic div (source only)
    # 1=magic div alg #1.  Slightly faster but limited range (if magic number is 2^32)
    # 2=magic div alg#2.  Slightly slower but handles all unsigned ints up to 2^32
    "MagicDivAlg":       [0,1,2],

    # For Block Mapping type:
    # 0   : Use hardware-assigned wg number with no remapping.
    # N   : WG block width.  "Wrap" to a new wg1 "row" assignment after N WGs assigned in that row.
    # < 0 : Swaps the position of wg0 and wg1.  Does not change NumWorkGroups* or ProblemNumWorkGroups*.
    #       Can be effective in M>N case.
    #       -1 is same as 1
    # Tensor C always mapped with first free coord as fastest moving
    # (Elements in this dimension are sequential in memory.
    #
    # For 2D non-batched Matrix this means index order is I, then J
    # For 2D batched Matrix this means index order is I, then J, then K.
    #
    # Then for 2D case:
    #   - If drawn in row-major format, I is the width and J is the height.
    #   - WGM determines dimensions of the box used to assign tiles from C
    #   - WGM is the height of the box (in the J dimension)
    #   - Given WGM, the box width (in I dim) is determined by number of CUs
    #   - The box always moves across matrixC in the fastest-moving "I" dim, then
    #     wraps to next J.  TODO - might be useful to change this?
    #
    # Examples for 2D matrix:
    # WGM=8:  on CU64 machine this is a square box
    # WGM=1:  Short/Fat - this will cover maximum width in I dimension of C.  This matches hardware assigned mapping.
    # WGM=64: Tall/Skinny - this will cover maximum width in J dimension of C.
    #
    # Formula for wgSerial:
    # wgSerial = wg0 + (wg1 % WorkGroupMapping) * nwg0
    "WorkGroupMapping":           list(range(-1024,1024+1)),  # change a workgroup's id so that the all the workgroups on the gpu at a time are hitting L2 cache the best
    "WorkGroupMappingType":       ["B", "Z"],           # Blocking, Z-order (not any faster than blocking, especially for the arithmetic it requires)
    "MaxOccupancy":               list(range(1, 40+1)),       # wg / CU; if cache thrashing is hurting performance, this allocates extra lds to artificially limit occupancy
    "WorkGroup":                  validWorkGroups,      # ( wg0 x wg1 x LocalSplitU ) dimensions of the workgroup which will operate on a tile and share lds

    #ThreadTile: ( tt0 x tt1 ) dimensions of the C tile that each thread works on,
    # TT=4 and VW=4 means a thread will work on a tight 4x4 tile of C, where VW=1 means the tile will work on 16 spread out values
    # Generally, the VW determines the consecutive a WI will work on, then it will skip ahead SG0*VW elements to get to the next row of VGPR inputs
    "ThreadTile":                 validThreadTiles,
    "MacroTile":                  validMacroTiles,      # MT0 = wg0*tt0, MT1 = wg1*tt1

    # Which instruction to use for MAC: MAD or FMA
    "MACInstruction":             ["MAD", "FMA"],
    "WavefrontSize":              [32, 64],

    # Which type of memory modifiers to use, GLC/SLC or SC0/SC1
    "MemoryModifierFormat":         ["GLC", "SC0"],

    # MatrixInstruction: (M x N x K x B)
    # XDLOPS tile definition, only valid for gfx908, gfx90a
    # MxNxKxB specifies matrix instruction variants
    #  MxNxB determines the shape of the C tile each instruction worked on
    #      K determines the unroll depth
    # If empty, do not use these instructions
    #
    # Alternative format: (M x N x K x B x MIBlockM x WaveTileM x WaveTileN x WaveM x WaveN)
    # (Note: MxN means M-by-N in the following comments)
    # MIBlockM determines how many blocks along M dimension for multi-block MI variants. Concrete examples:
    #  - MI 16x16x1x4 (4-block variant) with MIBlockM=4 -> (16x16)*(4x1)=64x16 tile per instruction executed
    #  - MI 32x32x1x2 (2-block variant) with MIBlockM=1 -> (32x32)*(1x2)=32x64 tile per instruction executed
    # WaveTileM/N are dimensions of the C tile each wave works on, and is close to the concept of ThreadTile in classic VALU kernels
    #  - WT 4x1 -> each wave executes 4x1 matrix instructions on the C tile of total area (4*MITileM)x(1*MITileN)
    # WaveM/N are dimensions of waves spawned for one workgroup where each wave consists of 64 threads
    #  - Wave2x2 -> a total of 4 waves in one workgroup of shape 2x2
    # Putting it all together:
    #  - [32, 32, 1, 2,  1,  4, 1,  2, 2]
    #     ^^^^^^^^^^^^   ^   ^^^^   ^^^^
    #      MatrixInst  BlkM   WT    Wave
    #  - means (32x64) per MI * (4x1) per wave * (2x2) per workgroup = (32*4*2)x(64*1*2) = 256x128 macro tile
    # Tensile will ignore the parameters ThreadTile and WorkGroup when the alternative format is used
    "MatrixInstruction":          validMatrixInstructions,

    # StoreRemap: Optimize MatrixInstruction store patterns to enhance performance.
    #             MI output data between each threads are along N dims.
    #             But global memory is along M dim continuous.
    #             That mean global write between each threads are not continuous.
    #             Therefore, store performance for MI instruction is poor.
    # How StoreRemap works in final store stage:
    #             1. Put all thread output data into LDS.
    #             2. All thread read data from LDS along M dims.
    #                (match global Memory continuous direction)
    #             3. All thread write out data into global memory.
    # 0:   Disable StoreRemap (default)
    # 1~8: Enable StoreRemap and set the global write vector width
    # Suggest optimum value: fp32 = [2,4], fp16 or bf16 = [4,8] (dwordx2 and dowrdx4)
    # -1:  Use dwordx2 if support SRVW, or set SRVW to 0
    "StoreRemapVectorWidth":      [-1,0,1,2,4,8,16],

    # SourceSwap: Optimizes MatrixInstruction store pattern by swapping mfma input order.
    "SourceSwap":                 [False, True],

    # AtomicAddC: If CEqualsD and Beta=1, use atomic add instead of load/store.
    "AtomicAddC":                 [False, True],

    # Following parameters are designed for store scheduling.
    # (store stands for load from C (with beta) and store to C/D)
    #
    # we want to hide store behind unroll loop
    #   1. if we can launch 2 WorkGroups per CU (occupancy >= 2, large M/N)
    #   2. if there are remaining global memory bandwidth in unroll loop (compute bound kernel)
    #
    # we can hide store behind the other WG's loop by lowering priority of store
    #   priority of loop is the same as priority of store
    #     WG0: ￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣\__
    #         |<-- loop --->|<-- store -->|end
    #
    #     WG1: ___________________________/￣￣￣￣￣￣￣￣￣￣￣￣\__
    #         |<--------- loop ------------------->|<-- store -->|end
    #
    #   priority of loop is higher than priority of store
    #     WG0: ￣￣￣￣￣￣￣\____________________
    #         |<-- loop --->|<------ store ----->|end
    #
    #     WG1: _____________/￣￣￣￣￣\__________________
    #         |<------- loop -------->|<----- store ---->|end
    "StorePriorityOpt":           [False, True],
    #
    # If we issue store in short period of time, kernel will become from compute bound to memory bound
    # 0 means issue instructions as many as possible if VGPR available
    "NumElementsPerBatchStore":   list(range(0, 256)),
    #
    # add sync after per batch store in order to store contiguous elements
    # add sleep after per batch store in order to distribute store over whole loops
    # NOTE: this parameter is highly depends on size_k
    # 0 means no sync and sleep
    "StoreSyncOpt":               list(range(0, 256)),
    #
    # There are index or address calculation between global instructions.
    # issue global instruction b2b has better performance
    "GroupLoadStore":             [False, True],
    #
    # Do storeC (output of GEMM) in unroll Loop; When PK enabled, storeC Code section can be
    # moved into unroll Loop code section for tiles[0..N-2], storeC scheduled in PK[1..N-1]
    # Enable this feature when PK is enabled
    # Enable this feature when you have 2 or More Tiles/CU
    # disable StoreSyncOpt, StorePriorityOpt,GroupLoadStore feature when this feature is enabled
    # enable PersistentKernel , PrefetchAcrossPersistent
    "StoreCInUnroll":             [False, True],
    #
    # StoreCInUnrollInterval is to specify the MFMA interval between 2 StoreC/AtomicAdd.
    # (This is effective only for StoreVectorWidth=1)
    # Actual MCMA interval is StoreCInUnrollInterval * (1/ LocalWritePerMfma).
    # For example, if StoreCInUnrollInterval=3, LocalWritePerMfma=0.5, StoreC/AtomicAddC inserted
    # at every 6 MFMAs (interval = 6)
    "StoreCInUnrollInterval":     list(range(1, 16)),
    #
    # StoreCInUnrollExact is to optimize specific K size by removing arbitrary K support code
    # 128x128 tile case, only K=512 is covered by StoreCInUnroll
    "StoreCInUnrollExact":        [False, True],
    #
    # StoreCInUnrollPostLoop is to add extra post loop to execute remaining LoadC/StoreC for K < supported minimumK for StoreCInUnroll
    "StoreCInUnrollPostLoop":     [False, True],

    # In order to remove the copying from Acc vgpr to Arch vgpr, only use Arch vgprs for v_mfma_xxx.
    # Only support for kernel whose totalVgpr counts less than 256 and gcn that has control bit ACC_CD.
    "MIArchVgpr":               [False, True],

    # Disable overlapping AB-tile vgpr and read/write addr vgprs with C-tile vgprs
    # Valid only for MatrixInstruction enabled kernels, which by default overlaps
    # C-tile w/ AB-tile until it's due for v_accvgpr_read before the write-back. Illustrated below:
    # |<----------------------- valuC ----------------------->|
    # |<--- valuA/B --->|<-- R/W pointers -->|xxx|<- Spares ->|
    #                                          ^        ^
    #         (Reserved by persistent kernels) ^        ^
    #                       (Utilized by register pool) ^
    "DisableVgprOverlapping":     [False, True],

    # If positive, each switch includes switches <= the specified switch.
    # For example 3 will enable NoPostLoop+NoGlobalRead+NoLocalWrite
    # If negative, setting is precise and will disable only the specified code piece.
    # intended use is to evaluate which sections of the kernel are taking most of the execution time
    # 0=Baseline
    # 1= +NoPostLoop
    # 2= +NoGlobalRead
    # 3= +NoLocalWrite
    # 4= +NoLocalRead
    # 5= +NoWait +NoSync
    # 6= +NoMAC
    # 7= +NoPreLoop+ NoGlobalReadInc
    # 9= NullKernel
    # 10= +invalid LocalReadA (use invalid vgpr offset(LdsOOB)). Negative only.
    # 11= +invalid LocalReadB (use invalid vgpr offset(LdsOOB)). Negative only.
    # 12= +invalid LocalReadA+B (use invalid vgpr offset(LdsOOB)). Negative only.
    # 13= +invalid LocalWriteA (use invalid vgpr offset(LdsOOB)). Negative only.
    # 14= +invalid LocalWriteB (use invalid vgpr offset(LdsOOB)). Negative only.
    # 15= +invalid LocalWriteA+B (use invalid vgpr offset(LdsOOB)). Negative only.
    # 16= +invalid GlobalReadA (use srdA[2]=0, BufferLoad only). Negative only.
    # 17= +invalid GlobalReadB (use srdB[2]=0, BufferLoad only). Negative only.
    # 18= +invalid GlobalReadA+B (use srdB[2]=0, BufferLoad only). Negative only.
    # For example set DisableKernelPieces: [0,1,2,3,4,5,6,7,9]
    #   this will create a set of kernels with progressively more pieces of the kernel disabled
    "DisableKernelPieces":        list(range(-18,10)),         # disable pieces of the kernel, for performance isolation

    # assume atomics always work correctly.
    "DisableAtomicFail": [False, True],

    # alternate implementation for fp16 HPA MFMA
    "Fp16AltImpl": [False, True],
    # fp16 alternate implementation round mode: false for truncate, true for round near zero
    "Fp16AltImplRound": [False, True],

    # StreamK (SK) kernels divide work evenly among CUs by splitting along MT and K dimensions.
    # Total work units are calculated as (#MTs x #LoopIters) and divided among workgroups.
    # In most cases each workgroup will calculate a partial tile that are accumulated in a fixup step in the same kernel
    # 0 : Standard data-parallel kernel
    # 1 : Basic StreamK
    # 2 : Two-Tile StreamK (each WG completes an even number of sk iterations, followed by an even number of dp tiles)
    # 3 : Two-Tile StreamK with DP before SK tiles
    # StreamK kernels can adjust the number of CUs being used.
    # Using fewer sometimes increases overall throughput by allowing other kernels to run in parallel.
    # StreamK grid is controlled by setting these enviornment variables:
    # TENSILE_STREAMK_FIXED_GRID lets you override the default grid size with a specific number
    #   0 = override disabled (default)
    # TENSILE_STREAMK_FULL_TILES sets the number of full tiles to be included in stream-k work
    #   -1 = use prediction model for best performance (default)
    #   0 = only remainder tiles run in stream-k
    #   1+ = remainder + 1 (or more) full grids of tiles run in stream-k
    # TENSILE_STREAMK_DYNAMIC_GRID enables dynamic grid mode, which automatically limits the number of CUs used:
    #   0 = Off, use all CUs (default)
    #   1 = Only reduce CUs for small problems to number of output tiles when num_tiles < CU count.
    #   2 = Also reduce CUs used for large sizes to improve data-parallel portion and reduce power.
    #   3 = Analytically predict the best grid-size by weighing the cost of the fix-up step and the cost of processing MACs.
    # TENSILE_STREAMK_MAX_CUS allows the user to manually set maximum number of CUs used, which could free up some CUs for
    #   other operations to run in parallel with gemm.
    # TENSILE_STREAMK_GRID_MULTIPLIER lets you set how many workgroups are created per CU being used.
    #   1 = 1 WG per CU (default), for example. 2 will launch WGs = 2 x CU count.
    # The priority of these environment variables is defined as follows:
    # TENSILE_STREAMK_FIXED_GRID > TENSILE_STREAMK_DYNAMIC_GRID > TENSILE_STREAMK_MAX_CUS > TENSILE_STREAMK_GRID_MULTIPLIER
    "StreamK": [0, 1, 2, 3],

    # Determines if StreamK kernel uses atomics
    # 0: uses workspace to store partial tiles, accumulate in deterministic fix-up step
    # 1: uses atomics to accumulate partial tiles
    "StreamKAtomic": [0, 1],

    # Enables XCC-based remapping of workgroups, set the value to the number of XCCs
    # for the device/configuration being used
    # 0: uses default workgroup assignment
    # 2+: remaps workgroups to be contiguous within an XCC for a given number of XCCs
    "StreamKXCCMapping": [0] + list(range(2, 9)),

    # Debug settings for stream-k kernels to disable parts of the kernel
    #   Bit 0: Don't generate fixup code
    #   Bit 1: Don't generate write to partials code
    # Both parts can be disabled together
    #   0 = Debug mode off, generate full kernel
    #   1 = No fixup
    #   2 = No partials
    #   3 = Nofixup and no partials
    "DebugStreamK": [0, 1, 2, 3],

    # 0  : standard launch
    # N>0 : launch persistent kernel with N workgroups per compute unit
    #       - Recommended min is enough WG to use all resources on the CU
    #       - Higher values result in shorter-running WG which are less 'persistent'
    #         this increases the switch time between work-groups but results in
    #         more opportunities to schedule other WG or recover if a wg runs long
    #         or all compute units were not available before the launch.
    #       - Host code will not launch more groups than tiles in the C space
    # -1 : Automatically choose a "heuristic" value that can possibly get a better gain: (TilesPerWorkgroup = 1~2)
    #      Not based on any theory, but on some experiment observation, can be used to reduce the kernels
    #      Recommend [-1,0,1] for basic tuning
    # Assertions/Requirements: NumWorkGroups0 * NumWorkGroups1 < 2^32
    "PersistentKernel":           range(-1,512+1) ,       # Use persistent kernel.

    # True:  Batch dimension (WG.z) is also considered in persistent kernel
    # False: Not considered
    #        for problems with large batch-size, PKAB = True could help
    #        for problems with only one batch, PKAB = True/False should make no difference
    "PersistentKernelAlongBatch": [False,True],

    # Allow macro-tile to span batch dimensions and thus a single workgroup can work across batch dimensions.
    # This can improve utilization, in particular if macro-tile is larger than the lower dimensions.
    # The byte address of the last element in the packed array must fit in 2^32.
    # 0x0 = each workgroup works on a single batch dim.
    # 0x1 = pack Batch dimensions into wg0/A - works if all batch strides for B==0.
    #       Also must set AssertFree0ElementMultiple to >= GlobalLoadVectorWidthA
    # 0x2 = pack Batch dimensions into wg1/B - works if all batch strides for A==0
    #       Also must set AssertFree1ElementMultiple to >= GlobalLoadVectorWidthB
    # 0x3 = pack batch dims into both A and B. Could support any stride for A and B. (Not supported yet)
    "PackBatchDims":             [0,1,2],

    # Pack free dimensions
    # If True, allow macro-tile to span free dimensions.  Single workgroup can work across multiple free dimensions.
    # If False, macro-tile is always Free0*Free1.  Additional free dimensions are not supported.
    "PackFreeDims":              [False, True],

    # Pack summation dims
    # If 0, a for loops are generated for each summation dimension.
    # If 1, summation dims are packed into a single loop and extracted as needed using mod/shift.  The innermost summation
    #  dimension must be an integer multiple of the unroll loop - in other words the load tile is contiguous in memory.
    #  In this mode, tensile can still prefetch data across the load tile dimension.
    # If 2, summations dims are packed into a single loop as above.  In addition, the load tile does not need to be
    #  contiguous in memory and can span summation dimensions. (not supported yet)
    "PackSummationDims":         [0,1],

    # debug mode, uses the PackSummationDims method to increment the unroll loop counter
    "UnrollIncIsDepthU":         [0,1],

    # Granularity allowed when packing tensor dims.
    # Lower values are finer granularity which requires more dimension division operations on store path
    # but supports more flexible tensor dimes.
    # Higher values are coarser values - less dimension division operations but tensor dims must meet
    # more stringent element multiple requirements
    # 0x1 : Any dimension supported, compute dims after each element (not supported yet)
    # 0x2 : VectorWidth must not span tensor dim
    "PackGranularity": [2],

    # Controls desired width (#elements) for loads from global memory -> LDS.
    # and eliminates the pointer unshift logic
    # Setting different GlobalLoadVectorWidth for A,B is now supported
    # (GlobalReadVectorWidth is still valid to set the same value to both A and B)
    # -1 : Set GlobalLoadVectorWidthA/B = VectorWidth
    # NOTE: for input bpe=32, max GRVW is 4  (to fit dwordX4) (FP32), min GRVW is 1 (dword)
    #                 bpe=16, max GRVW is 8  (to fit dwordX4) (FP16), min GRVW is 2 (dword)
    #                 bpe=8,  max GRVW is 16 (to fit dwordX4) (INT8), min GRVW is 4 (dword)
    # NOTE: GlobalLoadVectorWidthA/B can be auto-adjusted in SolutionStruct.py
    "GlobalLoadVectorWidthA":     [ -1, 1, 2, 3, 4, 6, 8, 16 ],
    "GlobalLoadVectorWidthB":     [ -1, 1, 2, 3, 4, 6, 8, 16 ],

    # legacy setting for global load width
    #   -1 : use GlobalLoadVectorWidthA, GlobalLoadVectorWidthB
    #  > 0 : GlobalLoadVectorWidthA=GlobalLoadVectorWidthB=GlobalReadVectorWidth
    "GlobalReadVectorWidth":      [ -1, 1, 2, 3, 4, 6, 8, 16 ],

    # Controls desired width (#elements) for loads from LDS -> VGPR.
    # -1 : Set LocalReadVectorWidth =  MIInputPerThread if MatrixInstruction else VectorWidth
    #  1 cannot be used for half type.
    # used in combination with TransposeLDS=True
    # in TransposeLDS=1 case, use wider load to fetch elements in summation dimension from LDS
    # helps optimizing instruction scheduling between MFMA and nonMFMA instructions
    # NOTE: for input bpe=32, max LRVW is 4  (to fit ds_read_b128) (FP32)
    #                 bpe=16, max LRVW is 8  (to fit ds_read_b128) (FP16)
    #                 bpe=8,  max LRVW is 16 (to fit ds_read_b128) (INT8)

    "LocalReadVectorWidth":      [ -1, 1, 2, 4, 8, 16 ],

    # threads should read/write/operate on this many contiguous elements from the C matrix.
    # If VW=4 then thread0 will process 4 consecutive C elements, then thread1 next 4, etc.
    # If the ThreadTile is > VectorWidth then thread0 will next operate on the 4 elements in C at (4*NumThreads)
    # Typically the load vector width and store vector width are directly related to the VW.
    # The global load width is closely related to the width of local stores so
    # GlobalLoadVectorWidthA/B also controls local write width.
    # Local read width also matches since VectorWidth consecutive elements must be read
    # Typically matching 16 bytes is good choice since the stores will be optimally coalesced with 16 bytes/WI.
    # -1 means use the largest vector width up to 128 bits.
    # Using a VW too large which results in >16bytes/thread isn't supported
    "VectorWidth":                [ -1, 1, 2, 3, 4, 6, 8, 16 ],
    # VectorWidth for B (MatrixInstruction only)
    "VectorWidthB":               [ -1, 1, 2, 4, 8, 16 ],

    # If 0, store 1 element per instruction.
    # If 1, store vector-width elements per instruction.
    # if -1, store vector-wide elements per instruction unless PBD would not generate a valid kernel
    "VectorStore":                    [-1, 0, 1],

    # Controls desired width (#elements) for stores from reg to global memory.
    # When MatrixInstruction == None, derived parameter gwvw takes precedence.
    # -1 : Set StoreVectorWidth = VectorWidth
    "StoreVectorWidth":           [ -1, 1, 2, 3, 4, 6, 8, 16 ],

    # place upper and lower limits on the skinny-ness of macro tiles; shape=1 means square tile, like 64x64. shape=4 means 4x64 or 64x4 or 128x8...
    # these will just mark some kernels as invalid so that fewer kernels will be checked
    "MacroTileShapeMin":          list(range(1, 512+1)),
    "MacroTileShapeMax":          list(range(1, 512+1)),

    # when loading all the data from global into lds requires multiple load instructions, these parameters govern which
    # loads will pull which rectangle of data from global into lds
    # NLC=1 means one load along the coalesced dimension, which results in the most coalescing possible
    # NLC=-1 looks for the largest number of reads along the coalesced dimension which results in the least amount of coalescing;
    # however in this case the stride between one load and another is a static value, therefore buffer loads only need one set of registers
    # whereas the =1 case has a stride which is a multiple of a kernel argument and therefore needs one address per load in the perpendicular dimension
    "NumLoadsCoalescedA":         list(range(-1, 64+1)),
    "NumLoadsCoalescedB":         list(range(-1, 64+1)),

    # DepthU, LocalSplitU (which is the 3rd number in WorkGroup), and LoopUnroll are closely related
    # LoopUnroll=4 means there are 4 sub-iterations within the loop, 4 actual iterations written in the code.
    # LocalSplit=2 means the workgroup is split up into 2 subgroups, and each subgroup is doing different parts of the summation.
    # subgroup0 does k=0-3, 8-11... and subgroup1 does k=4-7, 12-15...
    # So, each iteration through the summation loop, which has 4 actual sub-iterations, does 8 summation iterations, because each subgroup did 4;
    # and when data is read from global memory the threads read 8 elements along the summation dimension.
    # DepthU = LoopUnroll * LocalSplitU = 4*2 in this case
    # it made more sense for the user to directly control LocalSplitU and DepthU, then derive afterwards LoopUnroll=DepthU/LocalSplitU
    # -1 : Only allow GLVW=1
    # -2 : Only allow max(GLVWA,GLVWB) < VW ?
    # -3 : Only allow min(GLVWA,GLVWB) < VW ?
    "DepthU":                     depthUs,

    # DepthULdsDivisor (Split LDS) determines how we pipeline the data from global memory to LDS
    # Instead of moving all in-flight data from the register buffer (G2L) to the LDS at once, we divide the G2L buffer into N portions and
    # write each portion of the G2L to LDS, read from LDS and do the actual matrix multiply-accumulate, before moving on to the portion and so on.
    # This helps cut down LDS usage by the value of the divisor. Helps increase CU occupancy or DepthU if kernel was previously LDS limited.
    #
    # The premise is to be able to fetch 256B (equivalent to 128 half's or 64 single's) in TN layout problems to maximize L2 utilization. This
    # was previously a problem for TN since it implies DepthU is large, and that leads to oversubscription of LDS.
    #
    # Preconditions:
    # ScheduleIterAlg=3, TransposeLDS=1, PGR=0/1 excluding 2, DirectToLds=0 (DirectToLds=0 because part of the data loaded *need* to reside in registers),
    # nRegs per load >= DepthULdsDivisor (since we artificially require at least 1 register per LDS write)
    #
    # Example: DepthULdsDivisor=2
    # v0, v1, v2, v3 | v0, v1, v2, v3 | ... ----> unroll dim
    # -----Thd 0----- -----Thd 1-----   ...
    # 1st subloop writes v0,v1 to LDS
    # 2nd subloop writes v2,v3 to LDS
    "DepthULdsDivisor":           [1, 2, 4],

    # integer amount of padding to put into LDS, in 2016 this didn't seem to help performance, profilers were showing that channel conflicts weren't really hurting
    # performance so this has been deprecated and probably doesn't work
    # -1 means use same padding as the VectorWidth if TLU=0 else 0.  (Padding only helps when transpose is required)
    # With MatrixInstruction: -1 means max(GRVW,MIInput) if TLU=0
    # SourceKernel case, convert -1 to 0. Please manually set LdsPad for SourceKernel
    # SourceKernel requires LdsPadA==LdsPadB
    "LdsPadA":                     list(range(-1, 128)),
    "LdsPadB":                     list(range(-1, 128)),

    # Padding boundary for LDS. defines block-size for pad insertion. for every 'LdsBlockSizePerPad' bytes, LDS padding (pad value from LdsPad parameter)
    # is added (readOffset aware of the pad and adjusts offset value based on this parameter value).
    # Only support LdsBlockSizePerPad >= unrollDepth * BPE
    # 0 means disable LdsBlockSizePerPad,
    # -1 means round up to nearest power of 2 begin with 128
    # SourceKernel case, convert -1 to 0. Please manually set LdsBlockSizePerPad for SourceKernel
    "LdsBlockSizePerPadA":          [-1, 0, 64, 128, 256, 512, 1024, 2048, 4096],
    "LdsBlockSizePerPadB":          [-1, 0, 64, 128, 256, 512, 1024, 2048, 4096],

    # Transpose LDS format. Local store in Coalesced dimension , same as optimized global fetch dimension . applicable only in TLU=0 case for miSIMD(s)
    # TODO: No code for -1 ?
    "TransposeLDS":                [-1, 1, 0],

    # UnrollMajorLDSA, UnrollMajorLDSB is to use Transpose LDS format for either TLU = 0 or 1
    # If this is true, this overwrites UnrollMajorLDSA, B set by TranposeLDS
    # Using UnrollMajorLDSA or B for TLU=1 can be beneficial for smaller data types (need to combine with LdsPad and/or LdsBlockSizePerPad)
    "UnrollMajorLDSA":             [False, True],
    "UnrollMajorLDSB":             [False, True],

    # Add extra miLatencyLeft to improve local read scheduling
    # Adding more room for scheduling local read instructions
    # Increasing this might result in overflowedResources=5 error.
    # No need to increase miLatencyLeft in that case.
    "ExtraMiLatencyLeft":         list(range(0,9,2)),

    # Add extra latency to calculate number of MFMA to insert between local read and wait
    # Negative value means reduce interval between local read and wait (for DirectToVgpr only)
    "ExtraLatencyForLR":          list(range(0,17,2)) + list(range(-80,0,10)),

    # Allocate dedicated vgpr for local read with packing
    #   False: use tmp vgpr. Less vgpr usage, but not best for local read scheduling
    #   True: use dedicated vgpr for local read with packing. Best for local read scheduling, but need more vgpr
    # This is effective only when we need packing (UnrollMajorLDSA (or B) is False and bpe is less than 4 (HasEccHalf case).
    # Apply this to HasEccHalf case only.
    # Not effective for PrefetchLocalRead <= 1
    "VgprForLocalReadPacking":     [False, True],

    # ClusterLocalRead enables wider local read and packing with v_perm_b32 for 8bit or 16bit data
    # Works with VgprForLocalReadPacking=True
    "ClusterLocalRead":            [False, True],

    # tinkered with adding extra syncs or waits in the assembly kernels to see if it would improve the sequencing between workgroups, "fully synchronous scheduling" is WAY more promising; this can be deprecated
    "PerformanceSyncLocation":    list(range(-1, 16*16+1)),
    "PerformanceWaitLocation":    list(range(-1, 16*16+1)),
    "PerformanceWaitCount":       list(range(-1, 16)),

    # add gls or slc after global memory read/writes to change caching, not caching the writes is promising and improved performance a tiny bit
    # 1: glc, 2: slc, 3: glc+slc
    # 0: wave (none), 1: group (sc0), 2: device (sc1), 3: system (sc0+sc1) , 4-7: add "nt"
    "NonTemporalD":               list(range(0,8)),
    "NonTemporalC":               list(range(0,8)),
    "NonTemporalA":               list(range(0,8)),
    "NonTemporalB":               list(range(0,8)),

    # force sc0/sc1 bits on all stores, "Auto" for auto select by arch
    "ForceStoreSC1":              ["Auto", False, True],

    # guard against out of bounds reads
    # None: don't guard
    # Branch: use if statements (source only, and doesn't support VW)
    # ShiftPtr: shift read pointers to be in bounds, then unshift registers (source & assembly),
    # ShiftPtr does not support very small problem dims < global load vector width since the shift
    # would move outside the array bounds.
    # If GLVW==1 or Assert*ElementMultiple for the coalesced dim is > GRVW, then shifting is not
    # necessary and the shift/unshift code will not be generated
    "EdgeType":                   [ "Branch", "ShiftPtr", "None" ], # None=don't guard against ou

    # Group together unroll iterations inside the unroll loop.
    # For example, InnerUnroll=2 will fetch LDS for two unroll iterations
    "InnerUnroll":                [1,2,4,8,16,32,64],

    # Arrange elements in LDS so N elements consecutive in U-dim are adjacent in LDS
    # 1 is default and results in no interleaving.
    # Implementation only supports LocalDotLayout that is a power-of-two
    "LocalDotLayout":             [1,2,4,8],

    # Aggressive performance mode
    # Some of these may cause instability, particularly s_setprio
    # 0=none, 1=add setprio, 2=add setprio and modify LDS to allow only 2 waves/simd
    "AggressivePerfMode":       [0,1,2],

    # Use the feature whereby 56 bytes of kernel arguments can be preloaded in SGPRs
    # before the kernel begins executing.  This is currently only supported in cases
    # where that is enough to initiate the first load of the A and B tensors before
    # the remaining arguments have been loaded.
    # -1: Use if it's supported (by the GPU and in combination with other features)
    #  0: Don't use
    #  1: Use, reject solution if not supported.
    "PreloadKernelArguments": [-1, 0, 1],

    # If PreloadKernelArguments is 1, specifies whether we delay initiating the load
    # of the remaining arguments until after the load of A and B has been initiated.
    "DelayRemainingArguments": [False, True],

    # Kernels should be written in assembly or source
    # if assembly, ISA will determine architecture
    # if source, Runtime will determine language
    # later on, we'll relax this to inner kernel languages and outer kernel languages, such as inline asm embedded in ocl or in llvm
    "KernelLanguage":             [ "Assembly", "Source" ],
    "ISA":                        validISA,       # arch for assembly kernels

    # Replaces assembly kernels if they are found in the directory Tensile/Tensile/ReplacementKernels
    "ReplacementKernel":          [False, True],

    # Name of the custom kernel located in globalParameters["CustomKernelDirectory"].
    # a custom kernel is a user written assembly kernel with its associated configuration parameters included in a custom.config section
    # inside the yaml block between the --- and ... markers.  These parameters are only used for information purposes, not kernel generation.
    # Ex:
    # custom.config:
    #   ProblemType:
    #     OperationType: GEMM
    #     etc...
    #   ThreadTile: [8, 8]
    #   etc...
    #
    # Custom kernels can be included in a BenchmarkProblemSizeGroup by having their name (without file extension) listed under the "CustomKernels"
    # category alongside InitialSolutionParameters, BenchmarkCommonParameters, etc...
    "CustomKernelName":            -1,

    # Will allow a kernel to be accepted even when checks determine it's not viable.
    # Intended for use with custom kernels which have confirmed to be correct
    "NoReject":                    [False, True],

    "MinVgprNumber":                list(range(0,256)),

    "MaxVgprNumber":                list(range(0,257)),
    # min K size to use GlobalSplitU algorithm
    "MinKForGSU":                   [16,32,64,128,256]
    }

def getNameAbbreviation(name):
  specialValues = {
    'MACInstruction': '' # Conflicts with MatrixInstruction, but _MAD and _FMA should be enough differentiation for the kernel name.
  }
  if name in specialValues: return specialValues[name]
  return ''.join(c for c in name if not c.islower())

parameterNameAbbreviations = {name: getNameAbbreviation(name) for name in list(validParameters.keys()) + ["Kernel"]}

# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"LoopDoWhile":               [ False ] },
    {"LoopTail":                  [ True ] },
    {"EdgeType":                  [ "Branch" ] },
    {"InnerUnroll":               [ 1 ] },
    {"LocalDotLayout":            [ 1 ] },
    {"AggressivePerfMode":        [ 1 ] },
    {"PreloadKernelArguments":    [ 0 ] },
    {"KernelLanguage":            [ "Source" ] },
    {"LdsPadA":                   [ -1 ] },
    {"LdsPadB":                   [ -1 ] },
    {"LdsBlockSizePerPadA":       [ -1 ] },
    {"LdsBlockSizePerPadB":       [ -1 ] },
    {"TransposeLDS":              [ 0 ] },
    {"UnrollMajorLDSA":           [ False ] },
    {"UnrollMajorLDSB":           [ False ] },
    {"ExtraMiLatencyLeft":        [ 0 ] },
    {"ExtraLatencyForLR":         [ 0 ] },
    {"VgprForLocalReadPacking":   [ False ] },
    {"ClusterLocalRead":          [ False ] },
    {"MaxOccupancy":              [ 40 ] },
    {"VectorWidth":               [ -1 ] },
    {"VectorWidthB":              [ -1 ] },
    {"VectorStore":               [ -1 ] },
    {"StoreVectorWidth":          [ -1 ] },
    {"GlobalLoadVectorWidthA":    [ -1 ] },
    {"GlobalLoadVectorWidthB":    [ -1 ] },
    {"GlobalReadVectorWidth":     [ -1 ] },
    {"LocalReadVectorWidth":      [ -1 ] },
    {"GlobalReadCoalesceVectorA": [ True ] },
    {"GlobalReadCoalesceVectorB": [ True ] },
    {"WaveSeparateGlobalReadA":   [ 0 ] },
    {"WaveSeparateGlobalReadB":   [ 0 ] },
    {"GlobalReadCoalesceGroupA":  [ True ] },
    {"GlobalReadCoalesceGroupB":  [ True ] },
    {"PrefetchGlobalRead":        [ 1 ] },
    {"PrefetchLocalRead":         [ 1 ] },
    {"UnrollMemFence":            [ False ] },
    {"GlobalRead2A":              [ True ] },
    {"GlobalRead2B":              [ True ] },
    {"LocalWrite2A":              [ True ] },
    {"LocalWrite2B":              [ True ] },
    {"LocalRead2A":               [ True ] },
    {"LocalRead2B":               [ True ] },
    {"SuppressNoLoadLoop":        [ False ]},
    {"ExpandPointerSwap":         [ True ]},

    {"ScheduleGlobalRead":        [ 1 ] },
    {"ScheduleLocalWrite":        [ 1 ] },
    {"ScheduleIterAlg":           [ 1 ] },
    {"OptPreLoopVmcnt":           [ True ] },

    {"LdcEqualsLdd":              [ False ] },

    {"GlobalReadPerMfma":         [ 1 ] },
    {"LocalWritePerMfma":         [ -1 ] },

    {"InterleaveAlpha":           [ 0 ] },
    {"OptNoLoadLoop":             [ 1 ] },
    {"PrefetchAcrossPersistent":  [ 0 ] },
    {"PrefetchAcrossPersistentMode": [ 0 ] },

    {"BufferLoad":                [ True ] },
    {"BufferStore":               [ True ] },
    {"DirectToVgprA":             [ False ] },
    {"DirectToVgprB":             [ False ] },
    {"DirectToLdsA":              [ False ] },
    {"DirectToLdsB":              [ False ] },
    {"UseSgprForGRO":             [ -1 ] },
    {"UseInstOffsetForGRO":       [ 0 ] },
    {"AssertSummationElementMultiple": [ 1 ] },
    {"AssertFree0ElementMultiple": [ 1 ] },
    {"AssertFree1ElementMultiple": [ 1 ] },
    {"AssertMinApproxSize":        [ -1 ] },
    {"AssertStrideAEqual":        [ {} ] },
    {"AssertStrideBEqual":        [ {} ] },
    {"AssertStrideCEqual":        [ {} ] },
    {"AssertStrideDEqual":        [ {} ] },
    {"AssertSizeEqual":           [ {} ] },
    {"AssertSizeGreaterThan":     [ {} ] },
    {"AssertSizeMultiple":        [ {} ] },
    {"AssertSizeLessThan":        [ {} ] },
    {"AssertAIGreaterThanEqual":  [ -1 ] },
    {"AssertAILessThanEqual":     [ -1 ] },
    {"AssertAlphaValue":          [ False ]},
    {"AssertBetaValue":           [ False ]},
    {"AssertCEqualsD":            [ False ]},
    {"CheckTensorDimAsserts"      : [ False ] },
    {"CheckDimOverflow"           : [ 0 ] },

    {"StaggerU":                  [ 32 ] },   # recommend [0,32]
    {"StaggerUStride":            [ 256 ] },  # recommend 256 for V10,V20
    {"StaggerUMapping":           [ 0 ] },    # recommend [0,1]
    {"MagicDivAlg":               [ 2 ] },
    {"GlobalSplitU":              [ 1 ] },
    {"GlobalSplitUAlgorithm":     [ "SingleBuffer" ] },
    {"GlobalSplitUSummationAssignmentRoundRobin": [ True ] },
    {"GlobalSplitUWorkGroupMappingRoundRobin":    [ False ] },
    {"GlobalSplitUAtomicAdd":     [ False ] },
    {"MacroTileShapeMin":         [ 1 ] },
    {"MacroTileShapeMax":         [ 64 ] },
    {"StreamK":                   [ 0 ] },
    {"StreamKAtomic":             [ 0 ] },
    {"StreamKXCCMapping":         [ 0 ] },
    {"DebugStreamK":              [ 0 ] },
    {"PersistentKernel":          [ 0 ] },
    {"PersistentKernelAlongBatch":[ False ] },    # May be default True is better ?
    {"PackBatchDims":             [ 0 ] },
    {"PackFreeDims":              [ 1 ] },
    {"PackSummationDims":         [ 0 ] },
    {"UnrollIncIsDepthU":         [ 0 ] },
    {"PackGranularity":           [ 2 ] },
    {"FractionalLoad":            [ 0 ] },
    {"Use64bShadowLimit":         [ 1 ] },
    {"VectorAtomicWidth":         [ -1 ] },
    {"NumLoadsCoalescedA":        [ 1 ] },
    {"NumLoadsCoalescedB":        [ 1 ] },
    {"WorkGroup":                 [ [16,16,1]] },
    {"WorkGroupMappingType":      [ "B" ] },
    {"WorkGroupMapping":          [ 8 ] },
    {"ThreadTile":                [ [4,4] ] },
    {"MACInstruction":            [ "FMA" ]}, # Default to FMA, matches MAC performance and integrates additional flags
    {"WavefrontSize":             [ 64 ]},
    {"MemoryModifierFormat":      [ "" ] },
    {"MatrixInstruction":         [ [] ] },
    {"DisableVgprOverlapping":    [ False ] },
    {"1LDSBuffer":                [ 0 ] },
    {"DisableAtomicFail":         [ 0 ] },
    {"DisableKernelPieces":       [ 0 ] },
    {"DepthU":                    [ -1 ] },
    {"DepthULdsDivisor":          [ 1 ] },
    {"PerformanceSyncLocation":   [ -1 ] },
    {"PerformanceWaitLocation":   [ -1 ] },
    {"PerformanceWaitCount":      [ -1 ] },
    {"NonTemporalD":              [ 0 ] },
    {"NonTemporalC":              [ 0 ] },
    {"NonTemporalA":              [ 0 ] },
    {"NonTemporalB":              [ 0 ] },
    {"ForceStoreSC1":             [ "Auto" ] },
    {"ReplacementKernel":         [ False ] },
    {"CustomKernelName":          [ "" ] },
    {"NoReject":                  [ False ]},
    {"MinVgprNumber":             [0]},
    {"MaxVgprNumber":             [256]},
    {"StoreRemapVectorWidth":     [ 0 ] },
    {"SourceSwap":                [ False ] },
    {"AtomicAddC":                [ False ] },
    {"StorePriorityOpt":          [ False ] },
    {"NumElementsPerBatchStore":  [ 0 ] },
    {"StoreSyncOpt":              [ 0 ] },
    {"GroupLoadStore":            [ False ] },
    {"MIArchVgpr":                [ False ] },
    {"StoreCInUnroll":            [ False ] },
    {"StoreCInUnrollInterval":    [ 1 ] },
    {"StoreCInUnrollExact":       [ False ] },
    {"StoreCInUnrollPostLoop":    [ False ] },
    {"Fp16AltImpl":               [ False ] },
    {"Fp16AltImplRound":          [ False ] },
    {"ThreadSeparateGlobalReadA": [ 0 ] },
    {"ThreadSeparateGlobalReadB": [ 0 ] },
    {"MinKForGSU":                [256]},
    ]

# dictionary of defaults comprised of default option for each parameter
defaultSolution = {}
for paramDict in defaultBenchmarkCommonParameters:
  for key, value in paramDict.items():
    defaultSolution[key] = value[0]
# other non-benchmark options for solutions

# valid fields in ConvolutionConfig and explanations:
validConvolutionConfig= [
    # For OperationType == Convolution*
    # Examples: NCHW, NHWC, NCDHW, more
    # *HW* and *YX*   create solution with 2 spatial dimensions.
    # *DHW* and *ZYX* create solution with 3 spatial dimensions.
    "TensorAFormat",           # see validTensorAFormats
    "TensorBFormat",           # see validTensorBFormats
    "TensorDFormat",           # see validTensorDFormats

    # Each of the params below specifies dimensions separated by 'x".
    # -  The notation follows 'convolution' convention so fastest-moving dimensions are last,
    #    and should mirror the order of the spatial dimension in the activation format.
    #    For example, in NCHW format Filter=3x1 is 3 in the H dimension and 1 in the W dimension.
    # -  2 or 3 dimensions are supported 'Filter:3x1' or 'Filter:3x3x1'.
    # - Use an integer to create a kernel with a compile-time constant
    #   Use "N" to create flexible kernel the value provided at runtime via appropriate
    #   size and stride values.
    # - 0 specifies the default.  Defaults below shown for 2 spatial dimensions; a 3-dimensional
    #   default will be created if the formats request 3 spacial dimensions.
    "Filter",                   # examples: 1x1,3x3,1x7,7x1,NxN,Nx5,3x3x3.  Default=1x1/1x1x1.
    "Stride",                   # examples 1x1,2x2,1xN, 2x2x2.  Default=1x1/1x1x1.
    "Dilation",                 # examples 1x1,2x2,1xN, 2x2x2.  Default=1x1/1x1x1.

    # Pad at start of each filter dimension. Recommend 0x0 when possible or NxN otherwise.
    # (performance difference from compile-time padding is not significant)
    "PadStart",                 # examples:1x1, 2x3, 2x2x2, NxN.  Default=0x0/0x0x0.
    # Pad at end of each filter dimension
    "PadEnd",                   # examples:1x1, 2x3, 2x2x2, NxN.  Default=0x0/0x0x0.

    # For grouped convolutions:
    "GroupCount",

    # pack spatial dims (d,h,w) into single tensor dim when possible
    # This is preferred for cases where these dimensions are packed in memory
    # since it reduces addressing overhead and will produce a more efficient kernel
    # Default is 1, multiple dimensions will be created if needed for strides or other cases.
    "PackedSpatialDims",

    # pack filter dims (z,y,x) into single tensor dim when possible.
    # This is preferred for cases where these dimensions are packed in memory
    # since it reduces addressing overhead and will produce a more efficient kernel
    # Default is 1, multiple dimensions will be created if needed for dilations or other cases.
    "PackedFilterDims",

    # If 1:
    #  - Unroll index is the channel index
    #  - if PackSummationDims=0, this is likely highest perf since it provides a larger
    #    iteration count for the unroll loop.
    # If 0:
    #   - Unroll index is filter index (Forward,BackwardData) or spatial index (BackwardWeights)
    #   - provides better cache locality for most formats, but tighter looping.
    #   - Likely a good idea with PackSummationDims=1 since there is only one unroll loop.
    "UnrollOnChannel",

    # Input spatial dimensions (D,H,W)
    # Optional parameter for debug and testing.  This does not impact kernel generation.
    # If set,then each problem dimension size/stride will be checked to ensure they are
    # correctly specified. (TBD)
    # Also used by test benches to compute consistent strides and sizes for auto-generated
    # problem sizes and strides.
    'Spatial',              # examples 56x56, 7x7.

    ]

################################################################################
# Default Problem Type
################################################################################
defaultProblemType = {
    # =GEMM uses TransposeA,B parameters and makes the problem type more readable for users
    # =TensorContraction  requires specifying
    "OperationType":            "GEMM",           # GEMM, TensorContraction, ConvolutionForward, ConvolutionBackwardData, ConvolutionBackwardWeights

    "ConvolutionConfig":        [],               # See validConvolutionConfig

    "DataType":                 0,                # data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DestDataType":             0,                # destination data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "ComputeDataType":          0,                # compute data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType

    "UseBeta":                  True,             # =True use beta parameter (asm will check for B=0 and optimize the write for that), =False don't use beta parameter
    "HighPrecisionAccumulate":  False,            # f32 += f16*f16
    "SilentHighPrecisionAccumulate": False,       # Keep kernel names the same for HPA mode.  Useful for testing.
    "F32XdlMathOp":             0,                # reducing intermediate precision from f32 to a specific type, such as "x", as listed in SolutionStructs.py::DataType.
                                                  # in:f32, intermediate:xf32, out:f32. f32 = xf32(f32) * xf32(f32)

    "ComplexConjugateA":        False,            # complex data should be conjugated for "C" transpose case
    "ComplexConjugateB":        False,

    # for OperationType == GEMM
    "TransposeA":               False,            # =True means transA="T" or "C", =False means transA = "N"
    "TransposeB":               True,
    "Batched":                  False,            # add batching dimension
    "StridedBatched":           True,             # use to select general batch or strided batch

    # for OperationType == TensorContraction
    # - Indices < NumIndicesC are Free or Batch indices and appear in C and D
    # - Indices which appear in both A and B, and are < NumIndicesC are batch.  A and B must have same number of batch indices.
    # - Indices which appear in both A and B, and are >= NumIndicesC are summation. A and B must have same number of summation indices.
    # - Indices which appear in A or B (but not both), are Free.  A and B may have different numbers of free indices.
    # - Summation loops are nested from smallest index number to largest, with the largest summation index as the 'unroll' loop.
    # - Memory order of C and D matrices is always 0..NumIndicesC-1, with 0 as the fastest-moving.
    #   - By choosing index assignments the output can be 'transposed'.  For example if IA=[1,2] IB=[0,2] then 0 is the coalesced dim for C/D.
    #   - Likewise batch index may be assigned between two free indices to control the output order, ie to write in CNHW format.
    #   - For example : IA=[0,1,3] IB=[2,1,3].  0,2 are free indices;  1 is batch.
    "IndexAssignmentsA":        [0, 2],
    "IndexAssignmentsB":        [1, 2],
    "NumIndicesC":              2,

    # use initial strides for AB.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - GlobalReadWidth must be 1 (since elements are not guaranteed to be adjacent in memory)
    "UseInitialStridesAB":      False,

    # use initial strides for CD.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - Additional multiply on the store address path
    #   -VectorStore must be 0.  If VectorStore is -1, it will be silently set to 0 internally.
    "UseInitialStridesCD":      False,

    "AllowNoFreeDims":          False,  # allow A or B to specify no free dims
                                        # (if false, A and B must have at least one free dim)
                                        # (if true, A and B must have at least one free or batch dim)

    # SetConstStride* sets the specified stride in the problem.
    # These no longer generate predicates - see AssertStrideEqualA/B below
    # List of pairs of [index, constValue].
    # Index is a member of the global index assignments (not an offset into IndexAssignmentsA/B)
    # EX: SetConstStrideA: [ [3, 1], [2, 4] ] sets
    #     strideA for index3 to constant '1' and stride for index2 to constant '4'.
    "SetConstStrideA":          [],
    "SetConstStrideB":          [],

    # ZeroPad:
    # Zero-pad will add leading and trailing "pad" elements to the specified 'anchor'
    # dimension when accessed by specified summation dimension.
    #
    # Format is list of tuples of [freeDim, sumDim, padStart, padEnd].
    #  - freeDim is the anchor where the zero-pad starts.
    #  - sumDim is the summation dim to which the padding checking is added.
    #  - padStart is the number of elements to pad before the Start element
    #  - padEnd is the number of elements to pad before the last element.

    # - Terms:
    #   - Start is the first summation element
    #   - FreeSize is the size of the specified free dimension (freeDim)
    #   - SumSize is the size of the specified summation dimension (sumDim)
    # - Pad Ranges:
    #   - Ranges show below are inclusive on the start element and exclusive on the last element.
    #     For example, [0,3) is 0,1,2.
    #    - Elements in the region [Start-padStart, Start) are in the leading pad region and will return 0.
    #    - Elements in the memory region [Start + freeSize + sumSize - padEnd,  Start + freeSize + sumSize)
    #     are in the trailing pad region and will return 0.
    #    - Code actually checks for elementMem < padStart or elementMem>=elementEdge
    #      - elementMem is the memory offset of the element from the tensor base
    #      - elementEdge is FreeSize*FreeStride + (SumSize-1)*SumStride - padEnd
    #        - FreeStride is typically spatial*convolutionStride
    #        - SumStride is typically spatial*dilation
    #        - PadStart and PadStop should be scaled by spatial on the host before calling the kernel.
    #          (spatial is not available inside the kernel)
    #      - The GPU implementations shift the load tile by -padStart, then return 0s for any address <=0.
    #        The elementEdge is also shifted by -padStart.  This allows the global read offset to be used for the
    #        edge comparison.  Edge comparisons are performed with vector instructions so each work-item computes
    #        a different in/out value.
    #      - Multiple summations OR together their edge checks, so any OOB edge returns 0 for the load.
    #
    # - Strides:
    #   - padStart and padStop are passed as kernel arguments. These are scaled by the spatial dim on the host;
    #   - No memory access is performed for elements in the Pad regions.
    #   - The Pad regions are handled by manipulating the tensor addressing and are not visible in actual memory.
    #     For example, a tensor with 2 rows, 16 elements/row, padStart=padEnd=2 occupies 32 elements in memory (not 40)
    #   - Typical use case is to set summationStride < freeSize, with padStart+padEnd+1 == summationStride.
    # - Caveats:
    #  - ZeroPad requires that the ElementEdge <= 2^32:
    #    This is SizeFree+SizeSum + Pad_Leading + PadTrailingPad + padding=GRVW for shift-pointer) bytes < 2^32
    #    Likely this is less than the standard buffer load limits (bottom-right corner of macro-tile)

    #  EX: ZeroPadA: [ [0,1,  2,3]] # TensorA free index 0 with sum index 1 has leading pad=2 and trailing pad=3
    # Note nesting of brackets ; the parm can contain multiple padding tuples.
    #  EX: ZeroPadA: [ [0,1, -1,-1]]# Pads are dynamic and passed as part of the problem.

    "ZeroPadA":                 [], # [ [0,1, 2,3]]
    "ZeroPadB":                 [], # Not fully supported/tested yet

    # Summation dimension indices
    "MirrorDimsA":              [],
    "MirrorDimsB":              [],

    # for LD description
    "NumIndicesLD":             4,
    "IndexAssignmentsLD":       [3, 4, 5, 6],      # order is LDD, LDC, LDA, LDB

    # Tile aware solution selection
    "TileAwareSelection":       False,

    # FP16 Alternate Implementation
    "Fp16AltImpl":              False,
    "Fp16AltImplRound":         False,

    # Use unpack version of up-conversion instruction for f8/b8.
    "Fp8NoPackUpConversion" :   False,

    # S/W clipping of f32 to f8/b8 down conversion. When it is set, the kernel clips any value which is greater
    # than max_f8_value (e.g., 240.0 for f8) to max_f8_value in down conversion. NaN and +/-INF are propagated.
    # By default, it is set for f8 kernels.
    "Fp32toFp8SWClip" :         True,

    # only in-device SR for now
    "StochasticRounding" :      False,  # By default, IEEE RNE rounding

    # Rounding mode for f32 to f8 down conversion
    # TODO in Future:
    # There are two different rounding modes for f32 to f8 down conversion: [0]: IEEE RNE mode and [1/2]: stochastic mode.
    # For stochastic mode, there are two implementations to use random numbers in H/W instruction:
    #   In-device [1]: we need to pass the seed of random number and kernel will generate the pseudo-random numbers
    #   RND-table [2]: we need to pass a table of random numbers to the kernel, NOT implemented yet
    #"StochasticRounding" :     0  # [0,1,2]   0=NA, 1=in-device, 2=RND Table. By default, IEEE RNE rounding
    }

defaultProblemSizes = [{"Range": [ [2880], 0, 0 ]}]
defaultBenchmarkFinalProblemSizes = [{"Range": [
    [64, 64, 64, 512], 0, 0 ]}]
defaultBatchedProblemSizes = [{"Range": [ [2880], 0, [1], 0 ]}]
defaultBatchedBenchmarkFinalProblemSizes = [{"Range": [
    [64, 64, 64, 512], 0, [1], 0 ]}]


defaultSolutionSummationSizes = [32,64,96,128,256,512,1024,2048,4096,8192,16192]


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "ScheduleName":       "Tensile",
    "DeviceNames":  "fallback",
    "ArchitectureName": "gfx000",
    "SolutionImportanceMin":      0.01, # = 0.01=1% total time saved by keeping this solution
    }


################################################################################
# Searching Nested Lists / Dictionaries
# to see if keys exist and what their values are
################################################################################
# param name in structures?
def inListOfDictionaries(param, dictionaries):
  for dictionary in dictionaries:
    if param in dictionary:
      return True
  return False
def inListOfListOfDictionaries(param, dictionaries):
  for dictionaryList in dictionaries:
    if inListOfDictionaries(param, dictionaryList):
      return True
  return False
def inListOfLists(param, lists):
  for l in lists:
    if param in l:
      return True
  return False

# get param values from structures.
def hasParam( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      if hasParam(name, l):
        return True
    return False
  elif isinstance(structure, dict):
    return name in structure
  else:
    return name == structure
    #printExit("structure %s is not list or dict" % structure)

def getParamValues( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      param = getParamValues(name, l)
      if param != None:
        return param
    return None
  elif isinstance(structure, dict):
    if name in structure:
      return structure[name]
    else:
      return None
  else:
    printExit("structure %s is not list or dict" % structure)

################################################################################
# Print Debug
################################################################################
def tPrint(verbosity: int, arg) -> None:
    """Conditionally prints input to stdout.

    If the global print level is greater than or equal to the verbosity,
    the argument is printed to stdout.

    Args:
        verbosity: Level to use for printing arg.
        arg: Item to print to stdout.
    """
    if globalParameters["PrintLevel"] >= verbosity:
        print(arg)
        sys.stdout.flush()

def printWarning(message: str, category=UserWarning):
  warnings.warn(message, category)
  sys.stdout.flush()

def printExit(message):
  if TENSILE_TERM_COLORS:
        message = f"[bold red]{message}[/bold red]"
  print(message)
  sys.stdout.flush()
  sys.exit(-1)

################################################################################
# Locate Executables
# rocm-smi, hip-clang, rocm_agent_enumerator, clang-offload-bundler
################################################################################
def isExe( filePath ):
  return os.path.isfile(filePath) and os.access(filePath, os.X_OK)
def locateExe( defaultPath, exeName ): # /opt/rocm/bin, hip-clang
  # look in defaultPath first
  exePath = os.path.join(defaultPath, exeName)
  if isExe(exePath):
    return exePath
  # look in PATH second
  for path in os.environ["PATH"].split(os.pathsep):
    exePath = os.path.join(path, exeName)
    if isExe(exePath):
      return exePath
  return None

def GetAsmCaps(isaVersion: IsaVersion, hipVersion: SemanticVersion, cachedAsmCaps: Dict[IsaVersion, dict]) -> Dict[IsaVersion, dict]:
  """ Determine assembler capabilities by testing short instructions sequences """
  if globalParameters["AssemblerPath"] is not None:

    derivedAsmCaps = {}
    derivedAsmCaps["SupportedISA"]          = tryAssembler(isaVersion, "")
    derivedAsmCaps["HasExplicitCO"]         = tryAssembler(isaVersion, "v_add_co_u32 v0,vcc,v0,1")
    derivedAsmCaps["HasExplicitNC"]         = tryAssembler(isaVersion, "v_add_nc_u32 v0,v0,1")

    # Syntax of DirectToLds loads has changed: destination vgpr should be omitted
    # Old syntax should be removed in a future update as it is no longer supported
    derivedAsmCaps["HasDirectToLdsDest"]    = tryAssembler(isaVersion, "buffer_load_dword v40, v36, s[24:27], s28 offen offset:0 lds") \
                                           or tryAssembler(isaVersion, "buffer_load_b32 v40, v36, s[24:27], s28 offen offset:0 lds")
    derivedAsmCaps["HasDirectToLdsNoDest"]  = tryAssembler(isaVersion, "buffer_load_dword v36, s[24:27], s28 offen offset:0 lds") \
                                           or tryAssembler(isaVersion, "buffer_load_b32 v36, s[24:27], s28 offen offset:0 lds")

    derivedAsmCaps["HasAddLshl"]            = tryAssembler(isaVersion, "v_add_lshl_u32 v47, v36, v34, 0x2")
    derivedAsmCaps["HasLshlOr"]             = tryAssembler(isaVersion, "v_lshl_or_b32 v47, v36, 0x2, v34")
    derivedAsmCaps["HasSMulHi"]             = tryAssembler(isaVersion, "s_mul_hi_u32 s47, s36, s34")

    derivedAsmCaps["HasWMMA"]               = tryAssembler(isaVersion, "v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]")
    derivedAsmCaps["HasMFMA"]               = tryAssembler(isaVersion, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]") \
                                           or tryAssembler(isaVersion, "v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31]")
    derivedAsmCaps["HasMFMA_constSrc"]      = tryAssembler(isaVersion, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, 0") \
                                           or tryAssembler(isaVersion, "v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, 0")
    derivedAsmCaps["HasMFMA_vgpr"]          = tryAssembler(isaVersion, "v_mfma_f32_32x32x2bf16 v[0:31], v32, v33, v[0:31]") \
                                           or tryAssembler(isaVersion, "v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[0:31]")
    derivedAsmCaps["HasMFMA_f64"]           = tryAssembler(isaVersion, "v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]") \
                                           or tryAssembler(isaVersion, "v_mfma_f64_16x16x4_f64 v[0:7], v[32:33], v[36:37], v[0:7]")
    derivedAsmCaps["HasMFMA_bf16_original"] = tryAssembler(isaVersion, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]")
    derivedAsmCaps["HasMFMA_bf16_1k"]       = tryAssembler(isaVersion, "v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]")
    derivedAsmCaps["HasMFMA_xf32"]          = tryAssembler(isaVersion, "v_mfma_f32_32x32x4_xf32 a[0:15], v[32:33], v[36:37], a[0:15]")
    derivedAsmCaps["HasMFMA_f8"]            = tryAssembler(isaVersion, "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]")
    derivedAsmCaps["HasMFMA_b8"]            = tryAssembler(isaVersion, "v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]")
    derivedAsmCaps["HasMFMA_i8_908"]        = tryAssembler(isaVersion, "v_mfma_i32_32x32x8i8 a[0:15], v2, v3, a[0:15]")
    derivedAsmCaps["HasMFMA_i8_940"]        = tryAssembler(isaVersion, "v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15]")

    derivedAsmCaps["v_mac_f16"]             = tryAssembler(isaVersion, "v_mac_f16 v47, v36, v34")

    derivedAsmCaps["v_fma_f16"]             = tryAssembler(isaVersion, "v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]")
    derivedAsmCaps["v_fmac_f16"]            = tryAssembler(isaVersion, "v_fma_f16 v47, v36, v34")

    derivedAsmCaps["v_pk_fma_f16"]          = tryAssembler(isaVersion, "v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]")
    derivedAsmCaps["v_pk_fmac_f16"]         = tryAssembler(isaVersion, "v_pk_fma_f16 v47, v36, v34")

    derivedAsmCaps["v_mad_mix_f32"]         = tryAssembler(isaVersion, "v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]")
    derivedAsmCaps["v_fma_mix_f32"]         = tryAssembler(isaVersion, "v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]")

    derivedAsmCaps["v_dot2_f32_f16"]        = tryAssembler(isaVersion, "v_dot2_f32_f16 v20, v36, v34, v20")
    derivedAsmCaps["v_dot2c_f32_f16"]       = tryAssembler(isaVersion, "v_dot2c_f32_f16 v47, v36, v34") \
                                           or tryAssembler(isaVersion, "v_dot2acc_f32_f16 v47, v36, v34")

    derivedAsmCaps["v_dot4_i32_i8"]         = tryAssembler(isaVersion, "v_dot4_i32_i8 v47, v36, v34")
    derivedAsmCaps["v_dot4c_i32_i8"]        = tryAssembler(isaVersion, "v_dot4c_i32_i8 v47, v36, v34")
    derivedAsmCaps["VOP3v_dot4_i32_i8"]     = tryAssembler(isaVersion, "v_dot4_i32_i8 v47, v36, v34, v47")

    derivedAsmCaps["v_mac_f32"]             = tryAssembler(isaVersion, "v_mac_f32 v20, v21, v22")
    derivedAsmCaps["v_fma_f32"]             = tryAssembler(isaVersion, "v_fma_f32 v20, v21, v22, v23")
    derivedAsmCaps["v_fmac_f32"]            = tryAssembler(isaVersion, "v_fmac_f32 v20, v21, v22")

    derivedAsmCaps["v_fma_f64"]             = tryAssembler(isaVersion, "v_fma_f64 v[20:21], v[22:23], v[24:25], v[20:21]")

    derivedAsmCaps["v_mov_b64"]             = tryAssembler(isaVersion, "v_mov_b64 v[20:21], v[22:23]")

    derivedAsmCaps["HasAtomicAdd"]          = tryAssembler(isaVersion, "buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0")
    derivedAsmCaps["HasGLCModifier"]        = tryAssembler(isaVersion, "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc")
    derivedAsmCaps["HasNTModifier"]         = tryAssembler(isaVersion, "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, nt")

    if tryAssembler(isaVersion, "s_waitcnt vmcnt(63)"):
      derivedAsmCaps["MaxVmcnt"] = 63
    elif tryAssembler(isaVersion, "s_waitcnt vmcnt(15)"):
      derivedAsmCaps["MaxVmcnt"] = 15
    else:
      derivedAsmCaps["MaxVmcnt"] = 0

    # TODO- Need to query the max cap, just like vmcnt as well?
    derivedAsmCaps["MaxLgkmcnt"] = 15

    derivedAsmCaps["KernargPreloading"] = tryAssembler(isaVersion, """
      TestKernel:
      s_endpgm
      .amdhsa_kernel TestKernel
      .amdhsa_next_free_vgpr 8
      .amdhsa_next_free_sgpr 4
      .amdhsa_group_segment_fixed_size 0 // lds bytes
      .amdhsa_user_sgpr_kernarg_segment_ptr 1
      .amdhsa_user_sgpr_kernarg_preload_length 3
      .amdhsa_user_sgpr_kernarg_preload_offset 0
      .amdhsa_accum_offset 4
      .end_amdhsa_kernel
      """)

    derivedAsmCaps["SupportedSource"] = True

    ignoreCacheCheck = globalParameters["IgnoreAsmCapCache"]

    # disable cache checking for < rocm 5.3
    if len(hipVersion) >= 2:
      ignoreCacheCheck = ignoreCacheCheck or \
                         hipVersion.major < 5 or \
                         (hipVersion.major == 5 and hipVersion.minor <= 2)

    if not derivedAsmCaps["SupportedISA"] and cachedAsmCaps[isaVersion]["SupportedISA"]:
      printWarning("Architecture {} not supported by ROCm {}".format(isaVersion, globalParameters['HipClangVersion']), DeveloperWarning)
      ignoreCacheCheck = True

    # check if derived caps matches asm cap cache
    if not ignoreCacheCheck:
      exitFlag = False
      # rocm<=6.0, ignore KernargPreloading
      if hipVersion.major <= 5 or (hipVersion.major == 6 and hipVersion.minor == 0):
        derivedAsmCapsCopy = deepcopy(derivedAsmCaps)
        # copy KernargPreloading from CACHED_ASM_CAPS (to ignore this)
        derivedAsmCapsCopy["KernargPreloading"] = cachedAsmCaps[isaVersion]["KernargPreloading"]
        # compare with copied version (need to keep original value)
        if derivedAsmCapsCopy != cachedAsmCaps[isaVersion]:
          exitFlag = True
      # rocm>=6
      elif derivedAsmCaps != cachedAsmCaps[isaVersion]:
        exitFlag = True
      if exitFlag:
        printWarning("Cached asm caps differ from derived asm caps for {}".format(isaVersion))
    return derivedAsmCaps
  else:
    printWarning("Assembler not present, asm caps loaded from cache are unverified")
    return cachedAsmCaps[isaVersion]

def GetArchCaps(isaVersion):
  rv = {}
  rv["HasEccHalf"]         = (isaVersion==(9,0,6) or isaVersion==(9,0,8) or isaVersion==(9,0,10) or \
                              isaVersion==(9,4,2) or isaVersion==(9,5,0))
  rv["Waitcnt0Disabled"]   = (isaVersion==(9,0,8) or isaVersion==(9,0,10) or isaVersion==(9,4,2) or isaVersion==(9,5,0))
  rv["SeparateVscnt"]      = isaVersion[0] in (10, 11)
  rv["CMPXWritesSGPR"]     = isaVersion[0] not in (10, 11, 12)
  rv["HasWave32"]          = isaVersion[0] in (10, 11, 12)
  rv["HasAccCD"]           = (isaVersion==(9,0,10) or isaVersion==(9,4,2) or isaVersion==(9,5,0))
  rv["ArchAccUnifiedRegs"] = (isaVersion==(9,0,10) or isaVersion==(9,4,2) or isaVersion==(9,5,0))
  rv["VgprBank"]           = isaVersion[0] in (10, 11, 12)
  rv["InstRename"]         = isaVersion[0]>=11
  rv["CrosslaneWait"]      = (isaVersion==(9,4,2) or isaVersion==(9,5,0))
  rv["ForceStoreSC1"]      = False
  rv["HasDTLx4"]           = isaVersion==(9,5,0)

  return rv

def tryAssembler(isaVersion, asmString, debug=False, *options):
  """
  Try to assemble the asmString for the specified target processor
  Success is defined as assembler returning no error code or stderr/stdout
  """
  options = list(options)
  if globalParameters["PrintLevel"] >= 3:
    debug = True

  if isaVersion[0] >= 10:
    options += ['-mwavefrontsize64']

  assembler = globalParameters['AssemblerPath']
  if assembler is None:
    raise ValueError('No assembler available; set TENSILE_ROCM_ASSEMBLER_PATH to point to ROCm Clang.')
  args = [assembler, '-x', 'assembler',
          '-target', 'amdgcn-amdhsa',
          '-mcpu='+gfxName(isaVersion),
          *options,
          '-']

  result = subprocess.run(args, input=asmString.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  output = result.stdout.decode()

  if debug:
    print("isaVersion: ", isaVersion)
    print("asm_cmd:", ' '.join(args))
    print("asmString: ", asmString)
    print("output: ", output)
    print("return code: ", result.returncode)

  if output != "" or result.returncode != 0:
    return False
  return True

def gfxArch(name):
    import re
    match = re.search(r'gfx([0-9a-fA-F]{3,})', name)
    if not match: return None

    ipart = match.group(1)

    step = int(ipart[-1], 16)
    ipart = ipart[:-1]

    minor = int(ipart[-1])
    ipart = ipart[:-1]

    major = int(ipart)

    rv = (major, minor, step)

    return rv

def gfxName(arch):
    # convert last digit to hex because reasons
    name = str(arch[0]) + str(arch[1]) + ('%x' % arch[2])
    return 'gfx' + ''.join(map(str,name))


def detectIsaWindows(output):
    i = 0
    for line in output:
      if 'gcnArchName' in line:
        arch = gfxArch(line.split()[1].strip())
        if arch and arch in globalParameters["SupportedISA"]:
          tPrint(1, f"# Detected GPU {i}:    {gfxName(arch)}")
          globalParameters["CurrentISA"] = arch
          i += 1


def detectIsaLinux(output):
    for i, line in enumerate(output):
      arch = gfxArch(line.strip())
      if arch and arch in globalParameters["SupportedISA"]:
          tPrint(1, f"# Detected GPU {i}:    {gfxName(arch)}")
          globalParameters["CurrentISA"] = arch


def detectGlobalCurrentISA():
  """
  Returns returncode if detection failure
  """
  global globalParameters

  if not (globalParameters["CurrentISA"] == (0,0,0) and globalParameters["ROCmAgentEnumeratorPath"]):
    return -1

  enumerator = globalParameters["ROCmAgentEnumeratorPath"]
  process = subprocess.run([enumerator], stdout=subprocess.PIPE)
  output = process.stdout.decode().splitlines()

  if os.name == "nt":
    detectIsaWindows(output)
  else:
    detectIsaLinux(output)

  if process.returncode:
    printWarning(f"{globalParameters['ROCmAgentEnumeratorPath']} exited with code {process.returncode}")
  return process.returncode

def restoreDefaultGlobalParameters():
  """
  Restores `globalParameters` back to defaults.
  """
  global globalParameters
  global defaultGlobalParameters
  # Can't just assign globalParameters = deepcopy(defaultGlobalParameters) because that would
  # result in dangling references, specifically in Tensile.Tensile().
  globalParameters.clear()
  for key, value in deepcopy(defaultGlobalParameters).items():
    globalParameters[key] = value

def printTable(rows):
  rows = list([[str(cell) for cell in row] for row in rows])
  colWidths = list([max([len(cell) for cell in col]) for col in zip(*rows)])

  for row in rows:
    for (width, cell) in zip(colWidths, row):
      pad = ' ' * (width - len(cell))
      print(pad, cell, sep='', end=' ')
    print()

def printCapTable(parameters):
  import itertools
  archs = [(0,0,0)] + parameters["SupportedISA"]
  gfxNames = list(map(gfxName, archs))

  headerRow = ['cap'] + gfxNames

  def capRow(caps, cap):
    return [cap] + [('1' if cap in caps[arch] and caps[arch][cap] else '0') for arch in archs]

  allAsmCaps = set(itertools.chain(*[caps.keys() for arch, caps in parameters["AsmCaps"].items()]))
  allAsmCaps = sorted(allAsmCaps, key=lambda k: (k.split("_")[-1], k))
  asmCapRows = [capRow(parameters["AsmCaps"], cap) for cap in allAsmCaps]

  allArchCaps = set(itertools.chain(*[caps.keys() for arch, caps in parameters["ArchCaps"].items()]))
  allArchCaps = sorted(allArchCaps)
  archCapRows = [capRow(parameters["ArchCaps"], cap) for cap in allArchCaps]

  printTable([headerRow] + asmCapRows + archCapRows)


def splitArchs():
    # Helper for architecture
    def isSupported(arch):
        return (
            globalParameters["AsmCaps"][arch]["SupportedISA"]
            and globalParameters["AsmCaps"][arch]["SupportedSource"]
        )

    if ";" in globalParameters["Architecture"]:
        wantedArchs = globalParameters["Architecture"].split(";")
    else:
        wantedArchs = globalParameters["Architecture"].split("_")
    archs = []
    cmdlineArchs = []

    if "all" in wantedArchs:
        for arch in globalParameters["SupportedISA"]:
            if isSupported(arch):
                if arch == (9, 0, 6) or arch == (9, 0, 8) or arch == (9, 0, 10):
                    if arch == (9, 0, 10):
                        archs += [gfxName(arch) + "-xnack+"]
                        cmdlineArchs += [gfxName(arch) + ":xnack+"]
                    archs += [gfxName(arch) + "-xnack-"]
                    cmdlineArchs += [gfxName(arch) + ":xnack-"]
                else:
                    archs += [gfxName(arch)]
                    cmdlineArchs += [gfxName(arch)]
    else:
        for arch in wantedArchs:
            archs += [re.sub(":", "-", arch)]
            cmdlineArchs += [arch]
    return archs, cmdlineArchs


def populateCapabilities(
    globalParameters: Dict[str, Any], cachedAsmCaps: Dict[IsaVersion, dict], hipVer: SemanticVersion
):
    """Populates the assembler and archiecture capabilities based on the compiler and ISA.

    This function updates the **globalParameters** and **cachedAsmCaps** dictionaries with
    the assembler and architecture capabilities for each supported ISA version. It checks
    the compatibility of each ISA version with the current ROCm stack version and skips any
    unsupported ISA versions, issuing a warning for each skipped version.

    Args:
        globalParameters: A dictionary containing global parameters of the
            application, including the HipClang version and supported ISA versions.
        cachedAsmCaps: A dictionary to be populated with the assembler
            capabilities for each ISA version.
        hipVer: The hip compiler version.

    Note:
        This function modifies `globalParameters` and `cachedAsmCaps` in place.
    """
    supportedISA = globalParameters["SupportedISA"]
    to_remove = []

    emptyCache = not bool(globalParameters["AsmCaps"])

    for v in supportedISA + [(0, 0, 0)]:
        if v[0] == 12 and not (
            hipVer.major > 6 or (hipVer.major == 6 and hipVer.minor >= 3)
        ):
            printWarning(f"{gfxName(v)} isn't supported by ROCm {hipVer}, libraries will not be generated...")
            to_remove.append(v)
            continue

        if emptyCache or not globalParameters["CacheAsmCaps"]:
            globalParameters["AsmCaps"][v] = GetAsmCaps(v, hipVer, cachedAsmCaps)

        globalParameters["ArchCaps"][v] = GetArchCaps(v)

    # Efficiently remove unsupported ISA versions after iterating
    for v in to_remove:
        cachedAsmCaps.pop(v, None)  # Safely attempt to remove v from cachedAsmCaps
        supportedISA.remove(v)

################################################################################
################################################################################
def assignGlobalParameters( config, capabilitiesCache: Optional[dict] = None ):
  """
  Assign Global Parameters
  Each global parameter has a default parameter, and the user
  can override them, overriding happens here
  """

  global globalParameters

  # Minimum Required Version
  if "MinimumRequiredVersion" in config:
    if not versionIsCompatible(config["MinimumRequiredVersion"]):
      printExit("Config file requires version=%s is not compatible with current Tensile version=%s" \
          % (config["MinimumRequiredVersion"], __version__) )

  if "HipConfig" in config:
    output = getVersion(config["HipConfig"], regex=r'(.+)')
    globalParameters["HipClangVersion"] = output.strip()
    tPrint(1, f"# Found HIP version: {globalParameters['HipClangVersion']}")
  else:
    raise ValueError("HipConfig not specified in config, could not set HipClangVersion")

  # User-specified global parameters
  tPrint(3, "GlobalParameters:")
  for key in globalParameters:
    defaultValue = globalParameters[key]
    if key in config:
      configValue = config[key]
      if configValue == defaultValue:
        tPrint(3, " %24s: %8s (same)" % (key, configValue))
      else:
        tPrint(3, " %24s: %8s (overridden)" % (key, configValue))
    else:
      tPrint(3, " %24s: %8s (unspecified)" % (key, defaultValue))

  if "KeepBuildTmp" in config:
    globalParameters["KeepBuildTmp"] = config["KeepBuildTmp"]

  globalParameters["ROCmPath"] = "/opt/rocm"
  if "ROCM_PATH" in os.environ:
    globalParameters["ROCmPath"] = os.environ.get("ROCM_PATH")
  if "TENSILE_ROCM_PATH" in os.environ:
    globalParameters["ROCmPath"] = os.environ.get("TENSILE_ROCM_PATH")
  if os.name == "nt" and "HIP_DIR" in os.environ:
    globalParameters["ROCmPath"] = os.environ.get("HIP_PATH") # windows has no ROCM
  globalParameters["CmakeCxxCompiler"] = None
  if "CMAKE_CXX_COMPILER" in os.environ:
    globalParameters["CmakeCxxCompiler"] = os.environ.get("CMAKE_CXX_COMPILER")
  if "CMAKE_C_COMPILER" in os.environ:
    globalParameters["CmakeCCompiler"] = os.environ.get("CMAKE_C_COMPILER")

  globalParameters["ROCmBinPath"] = os.path.join(globalParameters["ROCmPath"], "bin")

  # ROCm Agent Enumerator Path
  if "ROCmAgentEnumeratorPath" in config:
    globalParameters["ROCmAgentEnumeratorPath"] = config["ROCmAgentEnumeratorPath"]
  else:
    raise ValueError("ROCmAgentEnumeratorPath not specified in config")
  if "CxxCompiler" in config:
    globalParameters["CxxCompiler"] = config["CxxCompiler"]
  else:
    raise ValueError("CxxCompiler not specified in config")
  if "CCompiler" in config:
    globalParameters["CCompiler"] = config["CCompiler"]
  else:
    raise ValueError("CCompiler not specified in config")
  if "Assembler" in config:
    globalParameters["AssemblerPath"] = config["Assembler"]
  else:
    raise ValueError("Assembler not specified in config")
  if "OffloadBundler" in config:
    globalParameters["ClangOffloadBundlerPath"] = config["OffloadBundler"]
  else:
    raise ValueError("OffloadBundler not specified in config")

  globalParameters["ROCmSMIPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm-smi")

  globalParameters["ExtractKernelPath"] = locateExe(os.path.join(globalParameters["ROCmPath"], "hip/bin"), "extractkernel")

  # read current gfx version
  returncode = detectGlobalCurrentISA()
  if globalParameters["CurrentISA"] == (0,0,0):
    printWarning("Did not detect SupportedISA: %s; cannot benchmark assembly kernels." % globalParameters["SupportedISA"])
  if returncode:
    if os.name == "nt":
      globalParameters["CurrentISA"] = (9,0,6)
      printWarning("Failed to detect ISA so forcing (gfx906) on windows")
  isasWithDisabledHWMonitor = ((9,4,2), (9,5,0), (11,0,0), (11,0,1), (11,0,2), (11,0,3), (11,5,0), (11,5,1), (12,0,0), (12,0,1))
  if globalParameters["CurrentISA"] in isasWithDisabledHWMonitor:
    isaString = ', '.join(map(gfxName, isasWithDisabledHWMonitor))
    printWarning(f"HardwareMonitor currently disabled for {isaString}")
    globalParameters["HardwareMonitor"] = False

  if "IgnoreAsmCapCache" in config:
    globalParameters["IgnoreAsmCapCache"] = config["IgnoreAsmCapCache"]

  globalParameters["CacheAsmCaps"] = True if capabilitiesCache is not None else False
  globalParameters["AsmCaps"] = capabilitiesCache if globalParameters["CacheAsmCaps"] else {}
  globalParameters["ArchCaps"] = {}

  hipVersion = SemanticVersion(
    *[int(c.split("-")[0]) for c in globalParameters["HipClangVersion"].split(".")[:3]]
  )
  cachedAsmCaps = getCapabilitiesCache(hipVersion)
  populateCapabilities(globalParameters, cachedAsmCaps, hipVersion)

  if globalParameters["PrintLevel"] >= 2:
    printCapTable(globalParameters)

  if globalParameters["AsmCaps"] != cachedAsmCaps and globalParameters["PrintLevel"] >= 1:
    import pprint
    printWarning("ASM Caps differ from cache. New caps:")
    print("####################")
    print("CACHED_ASM_CAPS = \\\n")
    pprint.pprint(globalParameters["AsmCaps"])
    print("####################")

  globalParameters["SupportedISA"] = list([i for i in globalParameters["SupportedISA"] if globalParameters["AsmCaps"][i]["SupportedISA"]])

  validParameters["ISA"] = [(0,0,0), *globalParameters["SupportedISA"]]

  if "MergeFiles" in config and "NumMergedFiles" in config:
    if not config["MergeFiles"] and config["NumMergedFiles"] > 1:
      config["NumMergedFiles"] = 1
      printWarning("--num-merged-files and --no-merge-files specified, ignoring --num-merged-files")

  rejectGlobalParameters = {
    "LogicPath",
    "OutputPath",
    "EmbedLibraryKey",
    "Version",
    "ClientConfig",
    "WriteMasterSolutionIndex",
    "HipConfig",
    "OffloadBundler",
    "Assembler",
  }
  for key in config:
    if key in rejectGlobalParameters:
      continue
    value = config[key]
    if key not in globalParameters:
      printWarning("Global parameter %s = %s unrecognized." % ( key, value ), DeveloperWarning)
    globalParameters[key] = value

def setupRestoreClocks():
  import atexit
  def restoreClocks():
    if globalParameters["PinClocks"]:
      rsmi = globalParameters["ROCmSMIPath"]
      subprocess.call([rsmi, "-d", "0", "--resetclocks"])
      subprocess.call([rsmi, "-d", "0", "--setfan", "50"])
  atexit.register(restoreClocks)
setupRestoreClocks()

################################################################################
# Assign Parameters
# populate dst with src[key] else give it the default/backup value
################################################################################
def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, \
    defaultDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = deepcopy(sourceDictionary[key])
  else:
    destinationDictionary[key] = deepcopy(defaultDictionary[key])

# populate dst with src[key] else abort since it's required
def assignParameterRequired(destinationDictionary, key, sourceDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = deepcopy(sourceDictionary[key])
  else:
    printExit("Parameter \"%s\" must be defined in dictionary %s" % (key, sourceDictionary) )


################################################################################
# Push / Pop Working Path
# store a WorkingPath where to write files (like benchmark files)
################################################################################
def pushWorkingPath( foldername ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  globalParameters["WorkingPath"] = \
      os.path.join(globalParameters["WorkingPath"], foldername )
  return ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  if len(workingDirectoryStack) == 0:
    globalParameters["WorkingPath"] = \
      os.path.split(globalParameters["WorkingPath"])[0]
  else:
    globalParameters["WorkingPath"] = workingDirectoryStack.pop()
def ensurePath(path):
  try:
    os.makedirs(path)
  except FileExistsError:
    pass
  except OSError:
    printExit("Failed to create directory \"%s\" " % (path) )
  return path
def setWorkingPath( fullPathName ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  workingDirectoryStack.append(globalParameters["WorkingPath"])
  globalParameters["WorkingPath"] = ensurePath(fullPathName)


def roundUp(f):
  return (int)(math.ceil(f))

################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def versionIsCompatible(queryVersionString):
  (qMajor, qMinor, qStep) = queryVersionString.split(".")
  (tMajor, tMinor, tStep) = __version__.split(".")

  # major version must match exactly
  if qMajor != tMajor:
    return False

  # minor.patch version must be >=
  if int(qMinor) > int(tMinor):
    return False
  if qMinor == tMinor:
    if int(qStep) > int(tStep):
      return False
  return True

def getCOVFromParam(versionString):
  if versionString == "default" or versionString == "V4":
    return 4
  elif versionString == "V5":
    return 5
  printExit("Unknown CodeObjectVersion %s" % (versionString))

def ClientExecutionLock():
  if not globalParameters["ClientExecutionLockPath"]:
    return open(os.devnull)

  import filelock
  return filelock.FileLock(globalParameters["ClientExecutionLockPath"])

# convert python list to C++ initializer style syntax
def listToInitializer(l):
  return "{" + ','.join(map(str, l)) + "}"


from copy import copy
class Backup:
  """RAII class to restore backed up fields from object"""
  fields = {}
  object = None
  def __init__(self, object, **fields):
    self.object = object
    for k, v in fields.items():
        self.fields[k] = copy(v)
  def __del__(self):
    for k, v in self.fields.items():
        setattr(self.object, k, v)

# Append copyrights to all files generated by tensile since they belong to Tensile intellectual property
CMakeHeader = """################################################################################
# Copyright (C) 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
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

###################################################
# This file was generated by Tensile:             #
# https://github.com/ROCmSoftwarePlatform/Tensile #
###################################################


"""

CHeader = """/*******************************************************************************
* Copyright (C) 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

/**************************************************
* This file was generated by Tensile:             *
* https://github.com/ROCmSoftwarePlatform/Tensile *
**************************************************/


"""

HR = "################################################################################"
