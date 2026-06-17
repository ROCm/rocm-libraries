################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import itertools
import os.path
import subprocess
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

from Tensile import __version__

from .Architectures import isaToGfx
from .Types import IsaVersion, IsaInfo
from .Utilities import locateExe, versionIsCompatible, print1, print2, printExit, printWarning, \
     getVerbosity
from .ValidParameters import validParameters

startTime = time.time()

globalParameters = OrderedDict()
globalParameters["MinimumRequiredVersion"] = (
    "0.0.0"  # which version of tensile is required to handle all the features required by this configuration file
)
globalParameters["PerformanceMetric"] = (
    "DeviceEfficiency"  # performance metric for benchmarking; one of {DeviceEfficiency, CUEfficiency}
)
globalParameters["ClientLogLevel"] = (
    3  # the log level of client. 0=Error, 1=Terse, 2=Verbose, 3=Debug (Aligned with ResultReporter.hpp)
)
# benchmarking
globalParameters["KernelTime"] = False  # T=use device timers, F=use host timers
globalParameters["PreciseKernelTime"] = (
    True  # T=On hip, use the timestamps for kernel start and stop rather than separate events.  Can provide more accurate kernel timing.  For GlobalSplitU kernels, recommend disabling this to provide consistent
)
# timing between GSU / non-GSU kernels
globalParameters["PinClocks"] = False  # T=pin gpu clocks and fan, F=don't
globalParameters["HardwareMonitor"] = (
    True  # False: disable benchmarking client monitoring clocks using rocm-smi.
)
globalParameters["MinFlopsPerSync"] = (
    1  # Minimum number of flops per sync to increase stability for small problems
)
globalParameters["NumBenchmarks"] = (
    1  # how many benchmark data points to collect per problem/solution
)
globalParameters["SyncsPerBenchmark"] = (
    1  # how iterations of the stream synchronization for-loop to do per benchmark data point
)
globalParameters["EnqueuesPerSync"] = 1  # how many solution enqueues to perform per synchronization
globalParameters["MaxEnqueuesPerSync"] = -1  # max solution enqueues to perform per synchronization
globalParameters["SleepPercent"] = (
    300  # how long to sleep after every data point: 25 means 25% of solution time. Sleeping lets gpu cool down more.
)
globalParameters["SkipSlowSolutionRatio"] = 0.0  # Skip slow solution during warm-up stage.
# The valid range of this ratio is (0.0 ~ 1.0), and 0.0 means no skipping.
# Skip condition:  warm-up time * ratio > current best sol's warm-up time
# Suggestion:
#     Small size :  0.5
#     Medium size: 0.75
#     Large size :  0.9

# validation
globalParameters["NumElementsToValidate"] = (
    128  # number of elements to validate, 128 will be evenly spaced out (with prime number stride) across C tensor
)
globalParameters["NumElementsToValidateWinner"] = (
    0  # number of elements to validate in LibraryClient stage, the exact number to be validated is max(NumElementsToValidate,NumElementsToValidateWinner)
)
globalParameters["BoundsCheck"] = 0  # Bounds check
# 1: Perform bounds check to find out of bounds reads/writes.  NumElementsToValidate must be -1.
# 2: Perform bounds check by front side guard page
# 3: Perform bounds check by back side guard page
# 4: Perform bounds check by both back and front side guard page

globalParameters["ValidationMaxToPrint"] = 4  # maximum number of mismatches to print
globalParameters["ValidationPrintValids"] = False  # print matches too
# steps
globalParameters["ForceRedoBenchmarkProblems"] = (
    True  # if False and benchmarking already complete, then benchmarking will be skipped when tensile is re-run
)
globalParameters["ForceRedoLibraryLogic"] = (
    True  # if False and library logic already analyzed, then library logic will be skipped when tensile is re-run
)
globalParameters["ForceRedoLibraryClient"] = (
    True  # if False and library client already built, then building library client will be skipped when tensile is re-run
)

globalParameters["ShowProgressBar"] = (
    True  # if False and library client already built, then building library client will be skipped when tensile is re-run
)
globalParameters["SolutionSelectionAlg"] = (
    1  # algorithm to determine which solutions to keep. 0=removeLeastImportantSolutions, 1=keepWinnerSolutions (faster)
)
globalParameters["GenerateSourcesAndExit"] = False  # Exit after kernel source generation.
globalParameters["ExitOnFails"] = (
    1  # 1: Exit after benchmark run if failures detected.  2: Exit during benchmark run.
)
globalParameters["CpuThreads"] = (
    -1
)  # How many CPU threads to use for kernel generation.  0=no threading, -1 == nproc, N=min(nproc,N).  TODO - 0 sometimes fails with a kernel name error?  0 does not check error codes correctly
# If True: after each kernel is assembled, verify that StinkyTofu's total instruction encoding
# size (sum of each instruction's encoded byte length from the Stinky pass pipeline) matches the
# compiled total instruction size (ELF ``.text`` size of the ``.o``, via llvm-readelf/readelf). In
# other words: computed encoding total == compiled total instruction size.
#
# Why: command-processor (CP) prefetch uses that computed total so the right amount of
# kernel code is prefetched. When you add instructions, literals, or passes, this catches mistakes
# where the pipeline’s instruction-size analysis no longer matches real assembly output.
#
# Scope: only kernels emitted through the StinkyTofu assembly path carry the embedded total in the
# generated ``.s`` and participate in this check; that same encoding total feeds
# ``.amdhsa_inst_pref_size`` in ``.amdhsa_kernel`` metadata for CP instruction prefetch.
# Currently this applies to gfx1250 only (StinkyTofu gfx1250 emitter and CP prefetch on that arch).
# Testing: ``tox`` turns on ``CheckASMCodeSize=True`` for the default Tensile pytest runs (see
# ``tox.ini``), so gfx1250 kernels built during those tests are verified automatically.
globalParameters["CheckASMCodeSize"] = False
globalParameters["NumWarmups"] = 0
globalParameters["TimingInstrumentation"] = False  # Enable detailed timing instrumentation output
globalParameters["ParallelGpuExecution"] = 1  # Number of GPUs for parallel client execution (0=auto-detect, 1=serial, N=use N GPUs)

globalParameters["PythonProfile"] = False  # Enable python profiling

globalParameters["ISA"] = []

########################################
# less common
########################################
globalParameters["CMakeBuildType"] = (
    "Release"  # whether benchmark clients and library client should be release or debug
)
globalParameters["LogicFormat"] = "yaml"  # set library backend (yaml, or json)
globalParameters["LibraryFormat"] = "yaml"  # set library backend (yaml, or msgpack)
globalParameters["MXScaleFormat"] = 0  # MX scale data format (0=none, 1=pre-swizzle for GPU kernel layout). Only the gfx950 subtile MX kernels need the pre-swizzle; gfx1250 reads canonical scales. The two gfx950 yamls that need it set MXScaleFormat: 1 explicitly.

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
#       - Problem-Independent: 0=0, 1=1, 2=2, 3=rand, 4=Nan, 5=Infinity, 6=BadInput(Nan), 7=BadOutput(Inf), 16=RandomNarrow,
#                              21=RandomNegPosLimited(-128~128 or -1~1), 23~26=Ind Cos/Sin Abs or Not
#       - Problem-dependent: 8=SerialID, 9=SerialDim0, 10=SerialDim1, 11=Identity, 12~15= Cos/Sin, Abs or Not
#       For A, B, C, D: All the InitMode (0~16) can be used
#       For Alpha/Beta: Only problem-independent init (0~7, 16, 23~26) can be used,
#                       problem-dependent init (8~15) would cause a exception (Invalid InitMode) in New Client
globalParameters["DataInitTypeAB"] = 3
globalParameters["DataInitTypeA"] = -1
globalParameters["DataInitTypeB"] = -1
globalParameters["DataInitTypeC"] = 3
globalParameters["DataInitTypeD"] = 0
globalParameters["DataInitTypeE"] = 0
globalParameters["DataInitTypeAlpha"] = 2
globalParameters["DataInitTypeBeta"] = 2
globalParameters["DataInitTypeBias"] = 3
globalParameters["DataInitTypeScaleA"] = 2
globalParameters["DataInitTypeScaleB"] = 2
globalParameters["DataInitTypeScaleC"] = 2
globalParameters["DataInitTypeScaleD"] = 2
globalParameters["DataInitTypeScaleAlphaVec"] = 3
globalParameters["DataInitTypeMXSA"] = 1
globalParameters["DataInitTypeMXSB"] = 1
globalParameters["DataInitValueActivationArgs"] = [2.0, 2.0]
globalParameters["CEqualD"] = (
    False  # Set to true if testing for the case where the pointer to C is the same as D.
)
# When this parameter is set to 0, the Tensile client will use srand(time(NULL)).
# If not 0 the Tensile client will use srand(seed).
globalParameters["DataInitSeed"] = 0
globalParameters["PruneSparseMode"] = (
    0  # Prune mode for Sparse Matrix: 0=random, 1=XX00, 2=X0X0, 3=0XX0, 4=X00X, 5=0X0X, 6=00XX
)

# build parameters
globalParameters["CMakeCXXFlags"] = ""  # pass flags to cmake
globalParameters["CMakeCFlags"] = ""  # pass flags to cmake
globalParameters["AsanBuild"] = False  # build with asan
#globalParameters["SaveTemps"] = False  # Generate intermediate results of hip kernels
globalParameters["KeepBuildTmp"] = False  # If true, do not remove artifacts in build_tmp

# debug for assembly
#globalParameters["SplitGSU"] = False  # Split GSU kernel into GSU1 and GSUM

# Tensor printing controls:
globalParameters["PrintTensorA"] = 0  # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0  # Print TensorB after initialization
globalParameters["PrintTensorC"] = (
    0  # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
)
globalParameters["PrintTensorD"] = (
    0  # Print TensorD.  0x1=after init; 0x2=after copy-back; 0x3=both
)
globalParameters["PrintTensorRef"] = (
    0  # Print reference tensor.  0x1=after init; 0x2=after copy-back; 0x3=both
)
globalParameters["PrintTensorBias"] = 0  # Print TensorBias after initialization
globalParameters["PrintTensorScaleAlphaVec"] = 0  # Print TensorScaleAlphaVec after initialization
globalParameters["PrintTensorAmaxD"] = 0  # Print AmaxD after validation
globalParameters["PrintWinnersOnly"] = False  # Only print the solutions which become the fastest
globalParameters["PrintCodeCommands"] = (
    False  # print the commands used to generate the code objects (asm,link,hip-clang, etc)
)
globalParameters["DumpTensors"] = (
    False  # If True, dump tensors to binary files instead of printing them.
)

# If PrintMax* is greater than the dimension, the middle elements will be replaced with "..."


# device selection
globalParameters["Platform"] = 0  # select opencl platform

# shouldn't need to change
globalParameters["ClientExecutionLockPath"] = (
    None  # Path for a file lock to ensure only one client is executed at once.  filelock module is required if this is enabled.
)
globalParameters["LibraryUpdateFile"] = (
    ""  # File name for writing indices and speeds suitable for updating an existing library logic file
)
globalParameters["LibraryUpdateComment"] = (
    False  # Include solution name as a comment in the library update file
)

# internal, i.e., gets set during startup
globalParameters["ROCmSMIPath"] = None  # /opt/rocm/bin/rocm-smi
globalParameters["HipClangVersion"] = "0.0.0"

# default runtime is selected based on operating system, user can override
if os.name == "nt":
    globalParameters["RuntimeLanguage"] = "HIP"
else:
    globalParameters["RuntimeLanguage"] = "HIP"

globalParameters["CodeObjectVersion"] = "4"

# perf model
globalParameters["PerfModelL2ReadHits"] = 0.0
globalParameters["PerfModelL2WriteHits"] = 0.15
globalParameters["PerfModelL2ReadBwMul"] = 2
globalParameters["PerfModelReadEfficiency"] = 0.85

# limitation for training
globalParameters["MaxWorkspaceSize"] = 128 * 1024 * 1024  # max workspace for training (128MB)
#globalParameters["MinKForGSU"] = 32  # min K size to use GlobalSplitU algorithm (only for HPA now)

# control if a solution is run for a given problem
globalParameters["GranularityThreshold"] = 0.0

# control if a solution is run for a given performance prediction
# if enabled, the solutions will be run in the order of the performance prediction, from fatest to slowest.
#   PredictionThreshold > 1 : Regular tuning, no sorting with performance prediction.
#   PredictionThreshold == 1: Regular tuning, but sorted with performance prediction.
#   PredictionThreshold < 1 : Sort and use the `PredictionThreshold * NumSolutions`-th performance prediction value as the threshold,
#                              and run the solutions with better prediction value than the threshold. Usually only run the
#                              `PredictionThreshold`-percent of solutions.
#   PredictionTHreshold == 0: Run the single solution with best prediction value.
globalParameters["PredictionThreshold"] = 2.0

globalParameters["PristineOnGPU"] = (
    True  # use Pristine memory on Tensile trainning verification or not
)

globalParameters["SeparateArchitectures"] = (
    False  # write Tensile library metadata to separate files for each architecture
)

globalParameters["LazyLibraryLoading"] = (
    False  # Load library and code object files when needed instead of at startup
)

globalParameters["EnableMarker"] = False  # Enable Tensile markers

globalParameters["UseUserArgs"] = False

globalParameters["RotatingBufferSize"] = 0  # Size in MB
globalParameters["RotatingMode"] = (
    0  # Default is 0, allocated in order A0B0C0D0..ANBNCNDN. 1 is in order A0 pad B0 pad .... AN pad BN pad.
)
# Mode 0 requires memcpy everytime when the problem changes to reset the data, but mode 1 doesn't.

# I-cache rotation (used by client --icache-rotate-copies / --icache-rotate-size).
# IcacheRotateCopies: number of EXTRA hipModule_t copies of each --code-object to load
#   for I-cache cold-miss tests. 0 = disabled (default). N (>0) = load N extras
#   (total = N+1 modules). -1 = auto: extras = max(rotating buffer num,
#   cache-overflow term). The cache-overflow term is
#   IcacheRotateSize * 2 * 1024 / min(kernel_start->label_GW_End) on Linux, and
#   IcacheRotateSize directly on non-Linux (no <elf.h> available).
globalParameters["IcacheRotateCopies"] = 0
# IcacheRotateSize: cache budget (in KB) used by the auto path's cache-overflow term.
#   Linux: effective bytes = IcacheRotateSize * 2 * 1024. Default 64 -> 128 KB,
#   loosely targeting ~2x a typical L1 I-cache.
#   Non-Linux: used directly as the raw extras count (no ELF parsing).
globalParameters["IcacheRotateSize"] = 64

globalParameters["BuildIdKind"] = "sha1"
globalParameters["AsmDebug"] = (
    False  # Set to True to keep debug information for compiled code objects
)

globalParameters["UseEffLike"] = True  # Set to False to use winnerGFlops as the performance metric

globalParameters["DisableAsmComments"] = False  # Set to True to disable assembly comments in generated assembly code

globalParameters["RocProfCounter"] = None # No rocprof counter

# StinkyTofu debug level (applies per-PM: outer PM + each ScopeAdaptor inner PM)
# 0: Silent (default)
# 1: Pass names + AnalysisManager cache activity to stdout
# 2: Initial IR + IR after each pass to per-PM files:
#    kernel-OuterPM-{before,after}_passes.txt     (outer PM)
#    <groupName>-{before,after}_passes.txt        (single-region adapter)
#    <group1>+<group2>-{before,after}_passes.txt  (multi-region adapter)
#    wholeKernel-{before,after}_passes.txt        (whole-kernel adapter)
globalParameters["StinkyTofuDebugLevel"] = 0

# StinkyTofu selective pass IR dump (applies per-PM, same file naming as DebugLevel 2)
# Comma-separated pass names to print IR before/after (case-sensitive)
# e.g. "CFG Builder" or "RedundantMovEliminationPass, StinkyDAGSchedulerPass"
# Unmatched pass names are silently ignored
globalParameters["StinkyTofuPrintBeforePass"] = ""
globalParameters["StinkyTofuPrintAfterPass"] = ""

# StinkyTofu internal pass debug logging & instruction-order snapshot (global — applies to all PMs)
# Comma-separated pass names (case-sensitive) to:
#   1. Enable PASS_DEBUG output for the listed passes
#   2. Record before/after instruction order JSON when StinkyTofuPassOrderSnapshotJson is set
# e.g. "StinkyDAGSchedulerPass"
# Unmatched pass names are silently ignored
globalParameters["StinkyTofuDebugPass"] = ""

# Before/after instruction-order JSON for tools/stinkytofu-analysis (empty = disabled).
# When set, PassManager records snapshots for passes listed in StinkyTofuDebugPass
# (defaults to StinkyDAGSchedulerPass only when StinkyTofuDebugPass is empty).
# Note: multiple kernels may overwrite the same file unless you use a unique path per build.
globalParameters["StinkyTofuPassOrderSnapshotJson"] = ""

# StinkyTofu optimization remarks (stderr).  Unlike PASS_DEBUG (for compiler
# developers), remarks are for kernel developers who want to understand generated
# code quality — e.g. how many regions a loop was split into, what caused the
# splits, and how many s_nop cycles were wasted.
globalParameters["StinkyTofuEnableRemarks"] = False

globalParameters["DisableSTWaitCnt"] = True

# Save a copy - since pytest doesn't re-run this initialization code and YAML files can override global settings - odd things can happen
# we should do this here...
defaultGlobalParameters = deepcopy(globalParameters)


################################################################################
# Tensile internal parameters
################################################################################
# These parameters are not adjustable by the config yamls. They change with the
# generator versions
internalParameters = {
    # Each universal kernel will generate one PostGSU(GlobalSplitUPGR) kernel
    "GlobalSplitUPGR": 16
}

# These parameters are used in ContractionSolutions for user arguments support.
defaultInternalSupportParams = {
    "KernArgsVersion": 2,
    # Information about user input internal kernel argument support
    # Change this to False if the CustomKernel does not support.
    "SupportUserGSU": True,
    # This is a little different from GSU because GSU is already a parameter,
    # but WGM is not.
    "SupportCustomWGM": True,
    "SupportCustomStaggerU": True,
    # Use GG as G's backend
    "UseUniversalArgs": True,
    "UseSFC": False,
}

# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"InnerUnroll": [1]},
    {"KernelLanguage": ["Assembly"]},
    {"LdsPadA": [-1]},
    {"LdsPadMXSA": [ -1 ] },
    {"LdsPadB": [-1]},
    {"LdsPadMXSB": [ -1 ] },
    {"LdsPadMetadata": [0]},
    {"LdsBlockSizePerPadA": [-1]},
    {"LdsBlockSizePerPadMXSA": [ -1 ] },
    {"LdsBlockSizePerPadB": [-1]},
    {"LdsBlockSizePerPadMXSB": [ -1 ] },
    {"LdsBlockSizePerPadMetadata": [0]},
    {"TransposeLDS": [-1]},
    {"TransposeLDSMetadata": [-1]},
    {"MaxOccupancy": [40]},
    {"MaxLDS": [-1]},
    {"VectorWidthA": [-1]},
    {"VectorWidthB": [-1]},
    {"VectorStore": [-1]},
    {"StoreVectorWidth": [-1]},
    {"GlobalReadVectorWidthA": [-1]},
    {"GlobalReadVectorWidthB": [-1]},
    {"LocalReadVectorWidth": [-1]},
    {"LocalReadVectorWidthA": [-1]},
    {"LocalReadVectorWidthB": [-1]},
    {"WaveSeparateGlobalReadA": [0]},
    {"WaveSeparateGlobalReadB": [0]},
    {"WaveSeparateGlobalReadMetadata": [0]},
    {"UnrollLoopSwapGlobalReadOrder": [0]},
    {"PrefetchGlobalRead": [1]},
    {"PrefetchLocalRead": [1]},
    {"PrefetchGL2": [0]},
    {"ClusterLocalRead": [1]},
    {"SuppressNoLoadLoop": [False]},
    {"ExpandPointerSwap": [True]},
    {"ScheduleGlobalRead": [1]},
    {"ScheduleLocalWrite": [1]},
    {"ScheduleIterAlg": [3]},
    {"GlobalReadPerMfma": [1]},
    {"LocalWritePerMfma": [-1]},
    {"InterleaveAlpha": [0]},
    {"OptNoLoadLoop": [1]},
    {"BufferLoad": [True]},
    {"BufferStore": [True]},
    {"DirectToVgprA": [False]},
    {"DirectToVgprB": [False]},
    {"DirectToVgprMXSA": [False]},
    {"DirectToVgprMXSB": [False]},
    {"DirectToVgprSparseMetadata": [False]},
    {"DirectToLds": [0]},
    {"DirectToLdsMetadata": [1]},
    {"UseSubtileImpl": [False]},
    {"UseSgprForGRO": [-1]},
    {"UseInstOffsetForGRO": [0]},
    {"AssertSummationElementMultiple": [1]},
    {"AssertFree0ElementMultiple": [1]},
    {"AssertFree1ElementMultiple": [1]},
    {"AssertAIGreaterThanEqual": [-1]},
    {"AssertAILessThanEqual": [-1]},
    {"StaggerU": [32]},  # recommend [0,32]
    {"StaggerUStride": [256]},  # recommend 256 for V10,V20
    {"StaggerUMapping": [0]},  # recommend [0,1]
    {"MagicDivAlg": [2]},
    {"GlobalSplitU": [1]},
    {"GlobalSplitUAlgorithm": ["MultipleBuffer"]},
    {"GlobalSplitUCoalesced": [False]},
    {"GlobalSplitUWorkGroupMappingRoundRobin": [False]},
    {"Use64bShadowLimit": [True]},
    {"Use64bShadowLimitMX": [False]}, # Disable Use64bShadowLimit for MXSA/B by default
    {"NumLoadsCoalescedA": [1]},
    {"NumLoadsCoalescedB": [1]},
    {"WorkGroup": [[16, 16, 1]]},
    {"WorkGroupMapping": [8]},
    {"WorkGroupMappingXCC": [1]},
    {"WorkGroupMappingXCCGroup": [-1]},
    {"ThreadTile": [[4, 4]]},
    {"WavefrontSize": [-1]},
    {"MatrixInstruction": [[]]},
    {"1LDSBuffer": [0]},
    {"DepthU": [-1]},
    {"NonTemporalE": [0]},
    {"NonTemporalD": [0]},
    {"NonTemporalC": [0]},
    {"NonTemporalA": [0]},
    {"NonTemporalMXSA": [ 0 ] },
    {"NonTemporalB": [0]},
    {"NonTemporalMXSB": [ 0 ] },
    {"NonTemporalWS": [0]},
    {"NonTemporalMetadata": [0]},
    {"NonTemporal": [-1]},
    {"TemporalHint": [-1]},
    {"TemporalHintE": [0]},
    {"TemporalHintD": [0]},
    {"TemporalHintC": [0]},
    {"TemporalHintA": [0]},
    {"TemporalHintMXSA": [0]},
    {"TemporalHintB": [0]},
    {"TemporalHintMXSB": [0]},
    {"TemporalHintWS": [0]},
    {"TemporalHintMetadata": [0]},
    {"NonVolatile": [-1]},
    {"NonVolatileE": [0]},
    {"NonVolatileD": [0]},
    {"NonVolatileC": [0]},
    {"NonVolatileA": [0]},
    {"NonVolatileMXSA": [0]},
    {"NonVolatileB": [0]},
    {"NonVolatileMXSB": [0]},
    {"NonVolatileWS": [0]},
    {"NonVolatileMetadata": [0]},
    {"PreloadKernArgs": [True]},
    {"CustomKernelName": [""]},
    {"NoReject": [False]},
    {"StoreRemapVectorWidth": [0]},
    {"SourceSwap": [False]},
    {"StorePriorityOpt": [False]},
    {"NumElementsPerBatchStore": [0]},
    {"StoreSyncOpt": [0]},
    {"GroupLoadStore": [False]},
    {"MIArchVgpr": [False]},
    {"StreamK": [0]},
    {"StreamKForceDPOnly": [0]},
    {"StreamKAtomic": [0]},
    {"StreamKXCCMapping": [0]},
    {"StreamKFixupTreeReduction": [0]},
    {"DebugStreamK": [0]},
    {"DebugPersistentKernelLoopForever": [False]},
    {"ActivationFused": [True]},
    {"ActivationFuncCall": [True]},
    {"ActivationAlt": [False]},
    {"WorkGroupReduction": [False]},
    {"ConvertAfterDS": [False]},
    {"ForceDisableShadowInit": [False]},
    {"InitCIterWmma": [0]},
    {"LDSTrInst": [False]},
    {"WaveSplitK": [ False ]},
    {"MbskPrefetchMethod": [-1]},
    {"PrefetchAcrossPersistent": [0]},
    {"UseCustomMainLoopSchedule": [-1]},
    {"SpaceFillingAlgo": [[]]},
    {"SFCWGM": [[[1,1],[1,1]]]},
    {"AdaptiveGemm": [0]},
    {"AdaptiveGemmGSUA": [0]},
    {"AdaptiveGemmNTAB": [0]},
    {"ExtraMiLatencyLeft": [-1]},
    {"ExtraLatencyForLR": [0]},
    {"TailloopInNll": [False]},
    {"SwapGlobalReadOrder": [0]},
    {"ScheduleGROverBarrier": [-1]},
    {"DtlPlusLdsBuf": [-1]},
    {"MinGRIncPerMfma": [-1]},
    {"UsePLRPack": [0]},
    {"TDMInst": [0]},
    {"TDMSplit": [False]},
    {"MXScaleFormat": ["Auto"]},
    {"MXLoadInst": ["Auto"]},
    # SwInstructionPrefetch — True: reserve one scratch SGPR so StinkyTofu can insert software
    # instruction prefetch when the ISA supports it (SwPrefetchInsertionPass).
    # Purpose: CP prefetch covers only a bounded window; very large kernels can see early kernel
    # code evicted from the I-cache before it runs. Software prefetch helps keep instruction fetch
    # ahead of execution. False: no SGPR reserved; Stinky prefetch pass disabled for that kernel.
    {"SwInstructionPrefetch": [True]},
    # ClusterDim — workgroup cluster dimensions [x, y] for clustered kernel launch.
    # [1, 1] disables clustering. Non-[1, 1] enables Multicast so workgroups within
    # a cluster can share data loaded via TDM-multicast, reducing redundant global reads.
    {"ClusterDim": [[1, 1]]},
    {"HalfPLR": [0]},
    {"TDMIterateMode": [0]}
]

# dictionary of defaults comprised of default option for each parameter
defaultSolution = {}
for paramDict in defaultBenchmarkCommonParameters:
    for key, value in paramDict.items():
        defaultSolution[key] = value[0]
# other non-benchmark options for solutions



defaultProblemSizes = [{"Range": [[2880], 0, 0]}]
defaultBenchmarkFinalProblemSizes = [{"Range": [[64, 64, 64, 512], 0, 0]}]
defaultBatchedProblemSizes = [{"Range": [[2880], 0, [1], 0]}]
defaultBatchedBenchmarkFinalProblemSizes = [{"Range": [[64, 64, 64, 512], 0, [1], 0]}]


defaultSolutionSummationSizes = [32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16192]


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "ScheduleName": "Tensile",
    "DeviceNames": "fallback",
    "ArchitectureName": "gfx000",
    "LibraryType": "GridBased",
    "SolutionImportanceMin": 0.01,  # = 0.01=1% total time saved by keeping this solution
}

# Per-key expected-type overrides for the LibraryLogic block. The
# defaults map gives a single example value per key; for keys that
# legitimately accept multiple Python types (e.g. DeviceNames can be a
# single str like the default "fallback" OR a list of strings naming
# multiple ASIC device IDs that share the same library logic), the
# override widens the accepted type set. Without this widening the
# strict gate in generateLogic() rejects the common list form.
libraryLogicTypeOverrides = {
    "DeviceNames": {str, list},
}


################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
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


# hopefully the isaInfoMap keys only contain isas we plan to build and not all
def printCapabilitiesTable(isaInfoMap: Dict[str, IsaInfo]):
    """
    Prints a capability table for the given parameters and ISA information map.

    Args:
        supportedIsas: The ISAs to show in the table.
        isaInfoMap: The ISA information map containing assembler and architecture capabilities.
    """

    def printTable(rows):
        rows = [[str(cell) for cell in row] for row in rows]
        colWidths = [max(len(cell) for cell in col) for col in zip(*rows)]

        for row in rows:
            print(" ".join(cell.ljust(width) for cell, width in zip(row, colWidths)))

    def capRow(isaInfoMap, cap, capType):
        return [cap] + [
            "1" if cap in getattr(info, capType) and getattr(info, capType)[cap] else "-"
            for info in isaInfoMap.values()
        ]

    gfxs = list(map(isaToGfx, isaInfoMap.keys()))
    headerRow = ["Capability"] + gfxs

    allAsmCaps = sorted(
        set(itertools.chain(*[info.asmCaps for info in isaInfoMap.values()])),
        key=lambda k: (k.split("_")[-1], k),
    )
    asmCapRows = [capRow(isaInfoMap, cap, "asmCaps") for cap in allAsmCaps]

    allArchCaps = sorted(set(itertools.chain(*[info.archCaps for info in isaInfoMap.values()])))
    archCapRows = [capRow(isaInfoMap, cap, "archCaps") for cap in allArchCaps]

    printTable([headerRow] + asmCapRows + archCapRows)


# Override table for globalParameters keys whose default value is None
# (and therefore have no usable type to derive from type(default)). Each
# entry is the set of permitted types for the user-supplied value.
# Skipping None-defaulted keys wholesale was a coverage gap that allowed
# e.g. RocProfCounter: 42 to pass silently.
globalParameterTypeOverrides = {
    "ClientExecutionLockPath": {type(None), str},   # path or unset
    "ROCmSMIPath":             {type(None), str},   # path, populated at startup
    "CmakeCxxCompiler":        {type(None), str},   # path, populated at startup
    "RocProfCounter":          {type(None), str},   # counter spec or None
}


def _assertOverrideTableCovers(defaults_dict, override_dict):
    """Coverage guard: every None-defaulted key has an override entry.

    Asserts the invariant that any None-defaulted key in globalParameters
    carries a corresponding type-override annotation. Exercised in CI by
    test_assignGlobalParameters_types.py /
    test_registry_disjoint_property.py rather than at module-import time,
    so the coverage gap is caught by a failing test (with the same precise
    message) instead of coupling the invariant to every import of this
    module.
    """
    missing = [k for k, v in defaults_dict.items()
               if v is None and k not in override_dict]
    if missing:
        raise RuntimeError(
            "globalParameterTypeOverrides is missing entries for the "
            f"following None-defaulted globalParameters: {missing!r}. "
            "Add type annotations for each to "
            "Tensile/Common/GlobalParameters.py."
        )


def _assertGlobalParametersAreValid(config, ignoreKeys):
    """Raise ConfigTypeError for the first unknown or mistyped key in *config*.

    No side effects: no subprocess, no ``locateExe``, no ISA mutation,
    no writes into the module-level ``globalParameters`` registry.

    Shared by production (``assignGlobalParameters``) and the corpus test
    so the two never diverge on what counts as a clean GlobalParameters block.

    Args:
        config: the raw ``GlobalParameters:`` dict from the YAML.
        ignoreKeys: keys with a sanctioned opt-out from the strict gate.
    """
    from .TypeValidationErrors import ConfigTypeError, formatMismatch

    # MinimumRequiredVersion is validated separately at the top of
    # assignGlobalParameters; skip it in the type loop here.
    typeCheckSkip = {"MinimumRequiredVersion"}

    for key in config:
        if key in ignoreKeys:
            continue
        value = config[key]
        if key not in globalParameters:
            raise ConfigTypeError(
                f"Unknown global parameter '{key}' = {value!r}. "
                f"Add it to globalParameters in GlobalParameters.py if it is real, "
                f"or remove it from the config."
            )

        if key in typeCheckSkip:
            continue

        if key in globalParameterTypeOverrides:
            expectedTypes = globalParameterTypeOverrides[key]
        else:
            default = globalParameters[key]
            if default is None:
                continue
            expectedTypes = {type(default)}

        if type(value) not in expectedTypes:
            raise ConfigTypeError(formatMismatch("", f"GlobalParameters.{key}", value, expectedTypes))


# Keys that may appear in a GlobalParameters: block but are not (or no longer)
# entries in the globalParameters registry. _assertGlobalParametersAreValid
# skips these silently. Shared with the corpus test so the two never diverge.
_GLOBAL_PARAMETER_IGNORE_KEYS = [
    # --- CLI / TensileCreateLibrary-level config consumed outside the
    #     globalParameters registry (each has its own argparse dest or
    #     direct arguments[...] reader) ---
    "Architecture",       # build-arch list, read directly in Tensile.py
    "PrintLevel",         # verbosity, read by setVerbosity in TensileCreateLibrary/Run.py
    "Device",             # device id, read from config in Tensile.py
    "UseCompression",     # code-object compression toggle, set in ParseArguments / read in Run.py
    "CxxCompiler",        # --cxx-compiler arg, resolved in Toolchain layer, not a registry value
    "CCompiler",          # --c-compiler arg, resolved in Toolchain layer, not a registry value
    "OffloadBundler",     # --offload-bundler arg, resolved in Toolchain layer
    "Assembler",          # --assembler arg, resolved in Toolchain layer
    "LogicPath",          # library-logic dir, read by TensileCreateLibrary/Run.py
    "LogicFilter",        # logic-file glob, read by TensileCreateLibrary/Run.py
    "OutputPath",         # positional output dir arg in Tensile.py / RetuneLibrary
    "Experimental",       # --experimental logic-dir toggle in ParseArguments
    "GenSolTable",        # --gen-sol-table toggle in ParseArguments
    # Keys with a sanctioned opt-out from the strict gate. Three categories:
    #   - Dead (no consumer anywhere; safe to silently drop):
    "AMDGPUArchPath",          # removed dc2c963c Mar2025; arch detection moved to Toolchain/
    "DataInitTypeeScaleE",     # never registered/consumed (double-e typo; scale-E init never implemented)
    "DeviceLDS",               # removed 7770c97e May2025; superseded by archCaps["DeviceLDS"]
    "MaxFileName",             # removed d170037b Feb2025; superseded by MAX_FILENAME_LENGTH constant
    "MergeFiles",              # removed 2d2e1496 Jan2025; code always merges now
    "MinKForGSU",              # removed dc2c963c Mar2025; superseded by MIN_K_FOR_GSU constant in Contractions.py
    "NewClient",               # removed dc2c963c Mar2025; old "must be 2" guard is meaningless now
    "ROCmAgentEnumeratorPath", # reverted 4a5aa3cb Mar2026; tool selection now via Toolchain/Validators.py
    "UseGPUTimer",             # never registered; always a duplicate of KernelTime (the real key)
    #   - Live but read via DebugConfig (makeDebugConfig in
    #     Tensile/Common/Types.py) directly from the raw config dict
    #     after assignGlobalParameters, bypassing the globalParameters
    #     registry:
    "ForceGenerateKernel",        # DebugConfig.forceGenerateKernel, read by makeDebugConfig
    "PrintIndexAssignmentInfo",   # DebugConfig.printIndexAssignmentInfo, read by makeDebugConfig
    "PrintSolutionRejectionReason", # DebugConfig.printSolutionRejectionReason, read by makeDebugConfig
    #   - Dead predecessor of PrintIndexAssignmentInfo (renamed
    #     dc2c963c Mar2025); kept so the one stale YAML
    #     (sgemm_xf32_asm.yaml) does not trip the gate. Follow-up:
    #     rename the YAML key to PrintIndexAssignmentInfo.
    "PrintIndexAssignments",
    #   - Misplaced solution parameter: registered in
    #     defaultBenchmarkCommonParameters (solution-level), not
    #     globalParameters; YAMLs that put it under GlobalParameters:
    #     have the value silently dropped. Follow-up: relocate to
    #     BenchmarkCommonParameters: / ForkParameters: in the YAMLs.
    "MaxLDS",
]


def assignGlobalParameters(config, isaInfoMap: Dict[IsaVersion, IsaInfo]):
    """
    Assign Global Parameters
    Each global parameter has a default parameter, and the user
    can override them, those overridings happen here
    """

    global globalParameters

    # Minimum Required Version
    if "MinimumRequiredVersion" in config:
        if not versionIsCompatible(config["MinimumRequiredVersion"]):
            printExit(
                "Config file requires version=%s is not compatible with current Tensile version=%s"
                % (config["MinimumRequiredVersion"], __version__)
            )

    # User-specified global parameters
    print2("GlobalParameters:")
    for key in globalParameters:
        defaultValue = globalParameters[key]
        if key in config:
            configValue = config[key]
            if configValue == defaultValue:
                print2(" %24s: %8s (same)" % (key, configValue))
            else:
                print2(" %24s: %8s (overriden)" % (key, configValue))
        else:
            print2(" %24s: %8s (unspecified)" % (key, defaultValue))

    globalParameters["ROCmPath"] = "/opt/rocm"
    if "ROCM_PATH" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("ROCM_PATH")
    if "TENSILE_ROCM_PATH" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("TENSILE_ROCM_PATH")
    if os.name == "nt" and "HIP_DIR" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("HIP_DIR")  # windows has no ROCM
    globalParameters["CmakeCxxCompiler"] = None
    if "CMAKE_CXX_COMPILER" in os.environ:
        globalParameters["CmakeCxxCompiler"] = os.environ.get("CMAKE_CXX_COMPILER")
    if "CMAKE_C_COMPILER" in os.environ:
        globalParameters["CmakeCCompiler"] = os.environ.get("CMAKE_C_COMPILER")

    globalParameters["ROCmBinPath"] = os.path.join(globalParameters["ROCmPath"], "bin")
    try:
        globalParameters["ROCmSMIPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm-smi")
    except OSError:
        if os.name == "nt":
            # rocm-smi is not presently supported on Windows so do not require it.
            pass
        else:
            raise

    if "AsanBuild" in config:
        globalParameters["AsanBuild"] = config["AsanBuild"]

    if "KeepBuildTmp" in config:
        globalParameters["KeepBuildTmp"] = config["KeepBuildTmp"]

    if "CodeObjectVersion" in config:
        globalParameters["CodeObjectVersion"] = config["CodeObjectVersion"]

    if getVerbosity() >= 1:
        printCapabilitiesTable(isaInfoMap)

    isaList = list(isaInfoMap.keys())
    validParameters["ISA"] = [IsaVersion(0, 0, 0), *isaList]

    # For ubuntu platforms, call dpkg to grep the version of hip-clang.  This check is platform specific, and in the future
    # additional support for yum, dnf zypper may need to be added.  On these other platforms, the default version of
    # '0.0.0' will persist

    # Due to platform.linux_distribution() being deprecated, just try to run dpkg regardless.
    # The alternative would be to install the `distro` package.
    # See https://docs.python.org/3.7/library/platform.html#platform.linux_distribution

    # The following try except block computes the hipcc version
    # TODO: hipcc is deprecated, this block should be removed.
    try:
        compiler = "hipcc"
        output = subprocess.run(
            [compiler, "--version"], check=True,
            stdout=subprocess.PIPE,
            # Avoids some warning spam on Windows.
            stderr=subprocess.DEVNULL,
        ).stdout.decode()

        for line in output.split("\n"):
            if "HIP version" in line:
                globalParameters["HipClangVersion"] = line.split()[2]
                print1("# Found hipcc version " + globalParameters["HipClangVersion"])

    except (subprocess.CalledProcessError, OSError) as e:
        printWarning("Error: {} running {} {} ".format("hipcc", "--version", e))

    ignoreKeys = _GLOBAL_PARAMETER_IGNORE_KEYS

    from .TypeValidationErrors import _STRICT_GATE_ENABLED

    if not _STRICT_GATE_ENABLED:
        for key in config:
            if key in ignoreKeys:
                continue
            value = config[key]
            if key not in globalParameters:
                printWarning("Global parameter %s = %s unrecognised." % (key, value))
            globalParameters[key] = value
        return

    _assertGlobalParametersAreValid(config, ignoreKeys)
    for key in config:
        if key in ignoreKeys:
            continue
        globalParameters[key] = config[key]


def setupRestoreClocks():
    import atexit

    def restoreClocks():
        # Clocks will only be pinned if rocm-smi is available, therefore
        # we only need to restore if found.
        if globalParameters["PinClocks"]:
            rsmi = globalParameters["ROCmSMIPath"]
            if rsmi is not None:
                subprocess.call([rsmi, "-d", "0", "--resetclocks"])
                subprocess.call([rsmi, "-d", "0", "--setfan", "50"])

    atexit.register(restoreClocks)

setupRestoreClocks()
