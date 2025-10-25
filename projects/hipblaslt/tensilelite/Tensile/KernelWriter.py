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

from rocisa import rocIsa, countInstruction, countGlobalRead, \
            countLocalRead, countLocalWrite, countDSStoreB256, getMFMAs
from rocisa.code import Module, TextBlock, StructuredModule, KernelBody
from rocisa.container import RegisterContainer, replaceHolder, HWRegContainer
from rocisa.label import LabelManager
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.instruction import BufferLoadB128, BufferLoadB32, BufferLoadB64, \
  BufferLoadD16B16, BufferLoadD16U8, DSLoad2B32, DSLoad2B64, DSLoadB128, \
  DSLoadB32, DSLoadB64, DSLoadB64TrB16, DSLoadInstruction, DSLoadU16, \
  DSLoadU8, DSStore2B32, DSStore2B64, DSStoreB128, DSStoreB16, DSStoreB256, \
  DSStoreB32, DSStoreB64, DSStoreB8, DSStoreInstruction, FlatLoadB128, FlatLoadB32, \
  FlatLoadB64, FlatStoreB128, FlatStoreB32, FlatStoreB64, Instruction, \
  MFMAInstruction, SBarrier, SBranch, SCBranchSCC0, SCBranchSCC1, SCBranchVCCNZ, SCmpLeU32, \
  SMFMAInstruction, SNop, SSetPrior, SSetRegIMM32B32, SSubU32, SWaitCnt, SWaitAlu, \
  SLongBranchPositive, VFmaMixF32, VMadMixF32, VMovB32
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType

from .KernelWriterModules import *
from .Component import Component, LraTileProperties
from .Components.Signature import UserArgumentsInfo
from .Components.CustomSchedule import customMainLoopSchedule
from .SolutionStructs import Solution, isPackedIndex
from .AsmMemoryInstruction import MemoryInstruction
from .Activation import ActivationModule
from .Common import printWarning, roundUp, print2, DebugConfig, DataDirection, \
  INDEX_CHARS, IsaVersion
from Tensile.SolutionStructs.Naming import getKernelNameMin
from Tensile.Toolchain.Component import Assembler

import math
import abc
import sys
import collections
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional,Tuple, Type
from math import ceil

# Make const values immutable
@dataclass(frozen=True)
class ConstValues():
  initLdsValue:int  = 0xFFFFFFFF  # Value to use for LDS Init, if enabled
  initSgprValue:int = 0x0  # Value to use for Sgpr Init, if enabled
  initVgprValue:int = 0xFFFFFFFF  # Value to use for Vgpr Init, if enabled

  maxOccupancy: int = 10

  ldsOOB: int       = 0xF00000

@dataclass
class MatrixInfo:
  numVgprValu: int               = -1
  numVgprValuPack: int           = -1
  startVgprValu: int             = -1
  startVgprValuPack: int         = -1
  startVgprValuPackTemp: int     = -1

  numSgprStrides: int            = -1

@dataclass
class ABMatrixInfo(MatrixInfo):
  numVgprValuPerBlock: int       = -1
  numVgprG2L: int                = -1
  numVgprG2LAllocated: int       = -1
  numVgprG2LTailLoopAllocated: int= -1
  startVgprG2L: Optional[int]    = None
  numVgprLocalReadAddr:int       = -1
  startVgprLocalReadAddr: int    = -1
  numVgprLocalWriteAddr: int     = -1
  startVgprLocalWriteAddr: int   = -1
  numVgprLocalWriteAddrTailLoop: int= -1
  numVgprGlobalReadOffsets: int  = -1
  startVgprGlobalReadOffset: int = -1
  numVgprLocalReadSwapAddr: int  = -1
  startVgprLocalReadSwapAddr: int= -1
  numVgprLocalWriteSwapAddr: int  = -1
  startVgprLocalWriteSwapAddr: int= -1
  numSgprGlobalReadIncs: int     = -1

# States
@dataclass
class StateValues:
  version: Tuple[int, int, int]
  kernel: dict
  kernelName: str
  language: str  = "ASM"
  asmCaps: dict  = field(init=False)
  archCaps: dict = field(init=False)
  regCaps: dict  = field(init=False)
  laneSGPRCount: int = field(init=False)

  # These values may differ between platforms, so put them here.
  # registers per global address
  rpga = 2 # 64-bit
  # registers per local address
  rpla = 1 # 32-bit
  # registers per global 32-bit offset (some intructions only support 32-bit offset)
  rpgo = 1 # 32-bit
  # registers per element
  bpr: int = 4 # all registers are 32bit
  # default setup
  # AB=DataType / Cexternal=DestDataType / Cinternal=Accumulation (MAC or MFMA)
  bpeAB: int = field(init=False)
  bpeE: int = field(init=False)
  # Cexternal = the "current" kernel output type,
  # - default: the "current" kernel is a non-GSU-kernel,
  #     Cexternal (= DestDataType) and is the final gemm result
  #
  # - For GSU: the "current" kernel is a GSU-kernel,
  #     this kernel returns a temp buffer with same type as Cinternal.
  #     Later, another kernel will accumulate this buffer
  #     and convert the final result to Cexternal (= DestDataType) as the gemm result
  bpeCexternalGSU1: int = field(init=False)
  bpeCexternal: int     = field(init=False)
  # already covers: dgemm, cgemm, zgemm, sgemm
  #               : hgemm  + !HPA ([H/H/H] compute = internal = f16)
  #               : hgemm  +  HPA ([H/H/S] or [H/S/S] compute = internal = f32)
  #               : bfgemm +  HPA ([B/B/S] or [H/S/S] compute = internal = f32)
  #               : int8x4-gemm   (internal = i32)
  bpeCinternal: int = field(init=False)

  # KernelWriter
  invalidLSUCode: bool                   = False
  inTailLoop: bool                       = False
  overflowedResources: int               = 0
  ## Schedule
  scheduleGlobalRead: int                = 0
  scheduleLocalWrite: int                = 0
  scheduleIterAlg: int                   = 0
  ## ShadowInit
  doShadowInit: int                      = 0
  ## Loop
  actualSummationLoops: int              = 0
  otherSummationLoops: int               = 0
  otherSummations: int                   = 0

  indexChars: List[int]                  = field(init=False)
  unrollIdx: int                         = -1
  unrollChar: str                        = ""
  tileChar0: str                         = ""
  tileChar1: str                         = ""

  numItersPLR: int                       = 0
  numVgprBuffer: int                     = 0
  numVgprBufferPackA: int                = 0
  numVgprBufferPackB: int                = 0
  numVgprBufferPackMetadata: int         = 0
  lrvwTileA: int                         = 0
  lrvwTileB: int                         = 0
  lrvwTileMetadata: int                         = 0 # For Sparse Metadat
  lrvwUnrollA: int                       = 0
  lrvwUnrollB: int                       = 0
  lrvwUnrollMetadata: int                       = 0 # For Sparse Metadat

  numMfmaPerIter: int                    = 0
  SubTileIdx: int                       = 0
  numReadsIterCoalescedA: int            = 0
  numReadsIterCoalescedB: int            = 0
  numReadsIterCoalescedMetadata: int     = 0
  numIterPerCoalescedReadA: int          = 0
  numIterPerCoalescedReadB: int          = 0
  numIterPerCoalescedReadMetadata: int   = 0
  numReadsPerUnrollA: int                = 0
  numReadsPerUnrollB: int                = 0
  numReadsPerUnrollMetadata: int         = 0

  # KernelWriterAssembly
  mixinst: Optional[Type[Instruction]]   = None
  globalReadIncsUseVgpr: bool            = False
  groOffsetInMacroTile: int              = 0
  use64bShadowLimit: bool                = True
  preventVgprOverflowDuringNewTile: int  = -1
  interleaveStoreVmcnt: bool             = False
  srdShiftLeft:dict                      = field(init=False)
  checkGRO: bool                         = False
  combineLocalAddresses: bool            = False # Debug
  unifiedVgprRegs: bool                  = False
  useAtomicAdd: bool                     = False
  serializedStore: bool                  = False

  a: ABMatrixInfo                        = field(default_factory=ABMatrixInfo)
  b: ABMatrixInfo                        = field(default_factory=ABMatrixInfo)
  c: MatrixInfo                          = field(default_factory=MatrixInfo)
  d: MatrixInfo                          = field(default_factory=MatrixInfo)
  e: MatrixInfo                          = field(default_factory=MatrixInfo)
  bias: MatrixInfo                       = field(default_factory=MatrixInfo)
  m: ABMatrixInfo                        = field(default_factory=ABMatrixInfo)       # For Sparse Metadata
  totalAgprs: int                        = 0
  maxLimitAgprs: int                     = 0
  totalMixedAgprs: int                   = 0
  totalVgprs: int                        = 0
  totalSgprs: int                        = 0
  lastValuAB: int                        = 0
  lastVgprForReads: int                  = 0
  startVgpr: int                         = 0
  startVgprAddressDbg: int               = -1
  startVgprAlphaTmp: int                 = -1
  startVgprSerial: int                   = -1
  startVgprCvt: int                      = -1

  numSgprSizesSum: int                   = 0
  numSgprSizesFree: int                  = 0
  numActivationTypeArgSize: int          = 0
  numActivationArgSize: int              = 0
  numactivationArgTotalSize: int         = 0
  numSgprAddressScaleA: int              = 0
  numSgprAddressScaleB: int              = 0
  numSgprAddressScaleC: int              = 0
  numSgprAddressScaleD: int              = 0
  numSgprAddressDbg: int                 = 0

  firstInitSgpr: int                     = -1
  nonPostLoopSgpr: List[str]             = field(init=False)
  userArgsInfo: UserArgumentsInfo        = field(default_factory=UserArgumentsInfo)
  numSgprToLoad: int                     = 0 # For kernel args
  preloadGuard: List[int]                = field(init=False)  # For preload kernel args guard
  numSgprPreload: int                    = 0 # For kernel args
  numSgprAlpha: int                      = 0 # For user arguments
  numSgprBeta: int                       = 0 # For user arguments
  numStoreSgprNames: List[str]           = field(init=False) # For post-loop kernel args
  numStoreSgprNameSizes: List[int]       = field(init=False) # For post-loop kernel args
  numStoreSgprToLoad: int                = 0 # For post-loop kernel args
  numStoreSgprNames2: List[str]          = field(init=False) # For post-loop kernel args
  numStoreSgprNameSizes2: List[int]      = field(init=False) # For post-loop kernel args
  numStoreSgprToLoad2: int               = 0 # For post-loop kernel args
  numStoreSgprInst: int                  = 0 # For pose-loop kernel args
  numStoreSgprInstExt: int               = 0 # For pose-loop kernel args
  numSgprAddressBias: int                = 0
  numSgprAddressGSUSync: int             = 0
  numSgprStreamK: int                    = 0
  BiasType: int                          = 0
  BiasStride: int                        = 0
  FactorDim: int                         = 0
  freeSgprVarPool: set                   = field(init=False)

  numReadsPerIterA: int                  = 0
  numReadsPerIterB: int                  = 0
  numReadsPerIterMetadata: int           = 0
  localReadDoCntA: int                   = 0
  localReadDoCntB: int                   = 0
  localReadDoCntMetadata: int            = 0
  savedLocalReadDoCntA: int              = 0
  savedLocalReadDoCntB: int              = 0
  savedLocalReadDoCntMetadata: int       = 0

  dtvKIntervalA: int                     = 1
  dtvKIntervalB: int                     = 1
  ## MFMA
  miLatency: int                         = 0
  miLatencyLeft: int                     = 0
  miDependency: int                      = 0
  numMfmaForLR: int                      = 1
  grEndMfmaIndex: int                    = -1
  sync1LdsMfmaIndex: int                 = -1
  lwStartMfmaIndex: int                  = -1
  lwEndMfmaIndex: int                    = -1
  numMfmaForNextLoopLR: int              = -1
  syncPlrMfmaIndex: int                  = -1
  numGlobalReadInsPerMfma: int           = 0
  numLocalWriteModPerMfma: int           = 0
  HHH_WMMA: bool                         = False
  tmpvgpr: List[int]                     = field(init=False) # vgpr storage for localread
  numPackCvt: int                        = 0

  lraTileProperties: Dict[int, LraTileProperties] = field(init=False)

  WGMTransformLevels: int                = -1

  # Epilogue states
  preloadScaleA = False
  preloadScaleB = False
  useBias       = DataDirection.NONE
  needBiasType  = False

  def __post_init__(self):
    """ How many SGPRs does it take to have one bit per lane? """
    self.laneSGPRCount = 2
    if "WavefrontSize" in self.kernel and self.kernel["WavefrontSize"] == 32:
      self.laneSGPRCount = 1

    self.indexChars   = []  # Workaround
    self.srdShiftLeft = {}  # Workaround

    self.perIterLocalWriteCanSkip = []

    self.lraTileProperties = {}  # Workaround

    self.numStoreSgprNames = []
    self.numStoreSgprNameSizes = []

    self.nonPostLoopSgpr = []

    self.preloadGuard = []
    self.tmpvgpr = {}
    self.freeSgprVarPool = set()

@dataclass
class StateVgprs:
  coord0: int = -1
  coord1: int = -1

  # StoreRemapVectorWidth
  storeRemapLW: int           = -1
  storeRemapLR: int           = -1
  storeRemapCoord0: int       = -1
  storeRemapCoord1: int       = -1
  storeRemapOffsetCoord1: int = -1

  # BufferStore
  cinRowPtr: int  = -1
  coutRowPtrBias: int = -1
  coutRowPtrE: int = -1
  coutRowPtrD: int = -1

  # FlatStore
  addrE: int    = -1
  addrD: int    = -1
  addrC: int    = -1
  addrBias: int = -1

  globalReadRegisters: Dict[str, int] = field(init=False)

  def __post_init__(self):
    self.globalReadRegisters = {}
    self.globalReadRegisters['A'] = []
    self.globalReadRegisters['B'] = []

@dataclass
class CodeModules:
  accVgprRead: Optional[Module]               = None
  accVgprWrite: Optional[Module]              = None
  mulAlphaMultipleBuffer: Optional[Module]    = None
  mulAlphaOther: Optional[Module]             = None
  localWriteA: Optional[Module]               = None
  localWriteB: Optional[Module]               = None
  dtlsM0UpdateA: Optional[Module]             = None
  dtlsM0UpdateB: Optional[Module]             = None
  globalReadA: Optional[Module]               = None
  globalReadB: Optional[Module]               = None
  globalReadIncrements: Optional[Module]      = None
  ## MFMA
  unrollLoopHeader: Optional[Module]                                  = None
  perIterGlobalRead: Optional[List[Module]]                           = None
  perIterLocalWrite: Optional[List[Tuple[List[int], Module]]]         = None
  perIterLocalWriteCodeNGLL: Optional[List[Tuple[List[int], Module]]] = None

@dataclass
class ExternClasses:
  activation: ActivationModule = ActivationModule()
  biasSumUnroll: Optional[Component.SumUnroll] = None


################################################################################
# Kernel Writer
################################################################################
class KernelWriter(metaclass=abc.ABCMeta):
  #__metaclass__=abc.ABCMeta

  ##############################################################################
  # Init
  ##############################################################################
  def __init__(
      self,
      assembler: Assembler,
      debugConfig: DebugConfig,
    ):
    self.assembler = assembler
    self.debugConfig = debugConfig

    self.do = {}
    self.do["PreLoop"]     = True
    self.do["GlobalReadA"] = True
    self.do["GlobalReadB"] = True
    self.do["GlobalInc"]   = True
    self.do["LocalWriteA"]  = True
    self.do["LocalWriteB"]  = True
    self.do["LocalWriteMetadata"]  = True
    self.do["LocalWriteCVT"]  = True
    self.do["LocalReadA"]  = True
    self.do["LocalReadB"]  = True
    self.do["LocalReadMetadata"]  = True
    self.do["Wait"]        = True
    self.do["Sync"]        = True
    self.do["MAC"]         = True
    self.do["PostLoop"]    = True
    self.do["ApplyAlpha"]  = True
    self.do["GlobalWrite"] = True
    self.do["EdgeWrite"]   = True
    self.do["KeepDirectToLdsAlloc"] = False  # If true, keep regs used for LDS alloc even if not used
    self.do["OptimizeNumItersPLR0"] = True
    self.do["AutoSplitDsWrite"] = True

    self.do["executeToInitEnd"] = 0
    self.do["executeToPrefetchEnd"] = 0
    self.do["executeToLoopEnd"] = 0

    # Various debug flags and modes
    self.db = {}
    self.db["EnableAsserts"]       = self.debugConfig.enableAsserts  # Enable assertion codegen. Requires 2 SGPR.
    self.db["DebugKernelMaxItems"] = 16  # Capture first N(=16) print values, ignore subsequent.  If -1, debug writing is faster but writing more than 16 values is undefined.

    # Chicken bit to add conservative synchronization at strategic points:
    # 0x01 = waitcnt + barrier after vector load
    # 0x02 = waitcnt at self._wait() for globalRead
    # 0x04 = waitcnt at self._wait() for localWrite
    # 0x08 = waitcnt at self._wait() for localRead
    # 0x10 = waitcnt after summation iteration, this can catch lingering ds or vm activity from summation loop
    # 0x20 = waitcnt before each write batch
    # 0x40 = waitcnt after each write batch
    self.db["ConservativeWaitCnt"] = 0x00

    self.db["InitLds"]     = False  # Initialize LDS at start of kernel

    # InitSgpr and InitVgpr can initialize at various points:
    #  0x1: Init at kernel start
    #  0x2: Init at end of summation loop (after tail too) - this is just before store loop
    self.db["InitSgpr"]   = 0x0  # init SGPRs

    self.db["InitVgpr"]   = 0x0  # init VGPRs

    # Debug and Check flags:
    # Check A and B values loaded from memory to ensure they are 1
    # Requires DataInitTypeAB=1.
    # Only works if the problem uses full tiles (no edges)
    # Mismatches will assert (generate GPUVM fault)
    self.db["CheckValue1A"] = self.debugConfig.enableDebugA
    self.db["CheckValue1B"] = self.debugConfig.enableDebugB
    self.db["CheckValue1Metadata"] = False
    # Check value in C matrix.
    # Caveats:
    #  - Only works for single, or Half/BF with HPA.
    #  - Checks after alpha calc for each element.  Later elements (in the TT) will not yet have applied their alpha.
    #  - Only works if matrix is integral multiple of macro-tile (no edges) - check is dumb so doesn't know
    #    which work-items are outside the valid edge.
    #  - Does not work in OptNoLoadLoop
    self.db["CheckValueC"]  = self.debugConfig.enableDebugC
    # value expected if CheckValueC is set. Use '.' for FP.
    # For example could be 16.0 if U=8 and alpha=2
    self.db["ValueCExpectedValue"] = self.debugConfig.expectedValueC

    # Force an expected value for all C outputs.
    # May be useful for checking store path
    # See same caveats as CheckValueC
    self.db["ForceExpectedValue"]  = self.debugConfig.forceCExpectedValue

    # Force VSerial value into the output, this will
    # not match reference but can be useful to see which work-items are
    # storing which values
    # See same caveats as CheckValueC
    self.db["ForceVSerial"] = False

    # can't do both of these since they both override output
    assert (not (self.db["ForceExpectedValue"] and self.db["ForceVSerial"]))

    self.db["ForceInputValueA"] = False
    self.db["ForceInputValueB"] = False
    self.db["ForceInputValueMetadata"] = False
    self.db["ForceValueA"] = 1.0
    self.db["ForceValueB"] = 1.0
    self.db["ForceValueMetadata"] = 1.0

    self.db["CheckStoreC"] = -1 # -1 disables, reload and verify output data.  Specify expected constant value.
    #self.db["CheckStoreC"] = 1024.0 # possible value

    self.db["ForceEdgeStores"] = 0 # 1=force use of edge store path for all tiles,  2=add assert in non-edge stores
    self.db["AssertNoEdge"] = 0 # Add assert in edge store code so crashes if executed

    # print vgpr register pool checkins and checkouts
    self.db["PrintRP"] = False
    self.db["AssertOnSgprOverflow"] = False
    self.db["PrintStoreRegisterDb"] = False

    self.labels = LabelManager()

    # KernelWriter values
    self.consts = ConstValues()
    self.states = StateValues((0,0,0), {}, "")
    self.vgprs  = StateVgprs()

    self.exclasses = ExternClasses()

  ##############################################################################
  # makeSchedule:  Schedule work into interations.

  # Tensile uses a two-level scheduler.  This the first-level, which
  # schedules global reads, global incs, and local writes into iteration.
  # Then makeSubIterSchedule schedules the instructions within the iteration.
  #
  # Inputs:
  #   localWriteEndIter: loop iteration where last writes should be inserted
  #      If scheduleLocalWrite=0, all writes will be be placed in this iteration.
  #      If scheduleLocalWrite=1, the scheduler will work backwards from this
  #      iteration.
  #
  # Outputs:
  #   self.codes.unrollLoopHeader:
  #      - Code module that should be added into the unroll loop header
  #        In unscheduled code this contains global loads and global address increment
  #   self.codes.perIterGlobalRead[], self.codes.perIterLocalWrite[]
  #      - List indexed by unroll iteration.
  #        Each entry in the list is a code module that should be added into that iteration.
  #        May be None, indicating no extra code for that iteration
  #   self.states.grEndMfmaIndex
  #   self.states.lwStartMfmaIndex
  #   self.states.lwEndMfmaIndex
  #   self.states.syncPlrMfmaIndex
  #   self.states.numMfmaForNextLoopLR
  # This routine is responsible for setting the schedule including determining
  # that all necessary dependency are met.  The driver code in kernelBody
  # blindly follows the plan set in unrollLoopHeaderCode and perIterCode
  ##############################################################################
  def makeSchedule(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, skipGlobalReadInc=False, firstIter=False, lastLoop=False, lastLc=False, isNGLL=False):

    self.codes.unrollLoopHeader = Module()
    # schedule of work for each local_read iteration:
    self.codes.perIterGlobalRead = [ Module() for i in range (kernel["LoopIters"]) ]
    self.codes.perIterLocalWrite = [ [[], Module()] for i in range (kernel["LoopIters"]) ]
    if lastLc:
      self.codes.perIterLocalWriteCodeNGLL = [ [[], Module()] for i in range (kernel["LoopIters"]) ]
    self.states.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    assert([item.name for item in self.codes.globalReadIncrements.items()] == ['globalReadIncrementA', 'globalReadIncrementB'])

    globalReadIncACode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementB")

    if skipGlobalReadInc:
      globalReadIncACode  = Module()
      globalReadIncBCode  = Module()

    siaComponent = Component.SIA.find(self)
    siaComponent.schedIntoIteration(self, kernel, tensorParametersA, tensorParametersB, \
      localWriteEndIter, firstIter, lastLoop, lastLc, globalReadIncACode, \
      globalReadIncBCode, isNGLL)

  ##############################################################################
  # packItemsConditional: pack src items into dst items until numPack or searchString is found
  # returns number of items packed
  ##############################################################################
  def _packItemsConditional(numPack, srcPackItems, dstPackItems, searchStrings):
    numPacked = 0
    if numPack == 0:
      numPack = 999
    for n in range(numPack):
      if srcPackItems:
        numPacked += 1
        item = srcPackItems.pop(0)
        dstPackItems.append(item)
        itemStr = str(item)
        for string in searchStrings:
          if string in itemStr:
            return numPacked
    return numPacked

  ##############################################################################
  # Schedule work into the each unroll loop iteration
  # localReadCode is the local reads for this loop iteration
  #  (returned by localReadDo). The instructions in localReadCode
  #  will retain their relative order, but may be interleaved
  #  with instructions from otherCode.

  # globalReadCode is the 'other' buffer loads and addr increments
  # localWriteCode is the 'other' local writes
  #  to schedule in with the ds reads.  The instructions
  #  will retain their relative order, but may be interleaved
  #  with instructions from localReadCode.

  # pointerCode contains local pointer changes (if needed)
  # waitCode contains s_waitcnt before macs.
  #   - Cannot be "" or None
  #   - may be empty Module if not waiting is desired (perhaps for debug)
  #   - may be multiple instructions (ConservativeWaitCnt)
  #   - typically is a single SWaitCnt.  This routine will
  #     modify the dscnt to account for any scheduling decisions.
  #     If this is not desired, add the waitCnt to pointerCode and
  #     set waitCode to an empty module
  # macIterCode contains the mac iters.  May be a macro call.
  #
  # returns: a Module with the combined, optimally scheduled
  #  localReadCode + otherCode
  ##############################################################################
  def _makeSubIterSchedule(self, kernel, tPA, tPB, localReadCode, iteration, pointerLWCode, pointerLRCode, waitCode, macIterCode, \
      waitLWCode = Module(), syncCode = Module(), packCode = Module(), prevIterCode = Module(), NLLlast = False):

    iterCode = Module()
    globalReadCode       = deepcopy(self.codes.perIterGlobalRead[iteration])
    localWriteCodeCounts = self.codes.perIterLocalWrite[iteration][0]
    localWriteCode       = self.codes.perIterLocalWrite[iteration][1]
    isBarrier            = kernel["LoopIters"] - self.states.numItersPLR
    hasLocalRead = countLocalRead(localReadCode)
    # Default schedule is other, local reads, then local writes:
    if self.states.scheduleIterAlg==0:
      # simple schedule, just add the modules in-order
      iterCode.add(globalReadCode)
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      iterCode.add(localReadCode)
      iterCode.add(localWriteCode)
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(waitCode)
      iterCode.add(packCode)
      iterCode.add(macIterCode)
    elif self.states.scheduleIterAlg == 1:
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      #import pdb
      #pdb.set_trace()
      # simple algorithm - do half the reads first:
      readsToSchedule = countLocalRead(localReadCode) / 2
      #localReadCode.prettyPrint()
      readItems = localReadCode.flatitems()
      while readItems:
        item = readItems.pop(0)
        #print "readsToSchedule=", readsToSchedule, "item=", item
        iterCode.add(item)
        readsThisItem = countLocalRead(item)
        if readsThisItem:
          assert readsThisItem==1, "Scheduler assumes 1 read per item"
          readsToSchedule = readsToSchedule - 1
          if readsToSchedule == 0:
            break

      iterCode.add(globalReadCode)

      # add rest of the reads here
      for item in readItems:
        iterCode.add(item)

      #move down write to be the last
      iterCode.add(localWriteCode)
      # tack on the pointer and mac code:
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(waitCode)
      iterCode.add(packCode)
      iterCode.add(macIterCode)
    elif self.states.scheduleIterAlg == 2:
    # SIA2 use only 1 iteration and separate compute and fetch by raising compute priority
    # 2 workgroup interleave, while WG0/WG1 doing compute, WG1/WG0 doing fetch
    # EPS need to be 1, or valu instruction will break interleave
      iterCode.add(globalReadCode)
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      iterCode.add(localReadCode)
      iterCode.add(waitCode)

      # interleave pack code
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      if self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]:
        instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      else:
        instPerRegPack = 1 if (kernel["ProblemType"]["DataType"].numRegisters() == 0.25) else 0
      instPerPack    = int(kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.states.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.states.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.states.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.states.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      macIterItems = macIterCode.flatitems()
      # pop the first code which is s_nop 1 for packing
      item = macIterItems.pop(0) if isinstance(macIterItems[0], SNop) else None

      numMfmaPerIter = self.states.numMfmaPerIter
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      for i in range(numMfmaPerIter):
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTileA"] == 0 else 0
          # blockWidth < 1, means 0.5 or 0.25 (BF,H,Int8)
          if self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]:
            packAIdx = packAIdx if tPA["bpe"] < 4 and not kernel["UnrollMajorLDSA"] else 0
            packBIdx = packBIdx if tPB["bpe"] < 4 and not kernel["UnrollMajorLDSB"] else 0
          else:
            packAIdx = packAIdx if tPA["localReadInstruction"].blockWidth == 0.25 else 0
            packBIdx = packAIdx if tPB["localReadInstruction"].blockWidth == 0.25 else 0

          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma, "2" means A & B
          if packItems:
            for j in range(instPerPack):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.add(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.add(SNop(waitState=0, comment="VALU packing writes to be consumed by matrix instruction"))
              curPackIdx += 1
        if i == 0:
          if not packItems:
            tmpVgpr = self.vgprPool.checkOut(1)
            iterCode.add(VMovB32(dst=vgpr(tmpVgpr), src="0x0", comment="valu operation to have different priority"))
            self.vgprPool.checkIn(tmpVgpr)
          iterCode.add(SSetPrior(prior=3, comment="Raise priority while processing macs"))
        item = macIterItems.pop(0)
        iterCode.add(item)
      while macIterItems:
        iterCode.add(macIterItems.pop(0))

      iterCode.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))
      if kernel["1LDSBuffer"]:
        barrier = Module()
        barrier.addComment0("1 LDS buffer: read-sync-write")
        barrier.add(SWaitCnt(dscnt=0, comment=""))
        barrier.add(SBarrier())
        iterCode.add(barrier)
      iterCode.add(localWriteCode)
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(SSetPrior(prior=2, comment="Raise priority while processing macs"))
    elif self.states.scheduleIterAlg == 3:
      iterCode.addComment0(" grEndMfmaIndex:%u, lwStartMfmaIndex:%u, lwEndMfmaIndex:%u "\
                          %(self.states.grEndMfmaIndex, self.states.lwStartMfmaIndex, self.states.lwEndMfmaIndex))
      iterCode.addComment0(" numMfmaForLR:%u, syncPlrMfmaIndex:%u "\
                           %(self.states.numMfmaForNextLoopLR, self.states.syncPlrMfmaIndex))
      #####
      # Prepare and Assign parameter
      ####
      if iteration == 0:
        self.localReadsVacancy = []
        self.localReadsWait = [ [] for j in range(kernel["LoopIters"])]
      self.localReadsWait[iteration] = waitCode
      numMfmaPerIter = self.states.numMfmaPerIter
      isBarrier = kernel["LoopIters"] - self.states.numItersPLR
      writeItems = list(localWriteCode.items())
      macIterItems = macIterCode.flatitems()
      skipLocalWriteWaitcnt = 0
      localReadsWaitcnt = 0
      localReadsIssuedInThisIter = 0
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0
      packMIdx = 0

      schedulePackConsiderMetadata = kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]
      numPackedA = 0
      numPackedB = 0
      numPackedM = 0
      #####
      # Prepare localReadCode
      ####
      localReadCodeAB = Module()
      for iui in range(kernel["InnerUnroll"]):
        localReadCodeA = localReadCode.findNamedItem("LocalReadDoA_I%s"%(iui))
        localReadCodeB = localReadCode.findNamedItem("LocalReadDoB_I%s"%(iui))
        localReadCodeM = localReadCode.findNamedItem("LocalReadDoMetadata_I%s"%(iui))
        # In case localReadDo not generate localReadCode Module
        # and findNamedItem will return None type
        # TODO: findNamedItem return Module() if not found
        if not localReadCodeA:
          localReadCodeA = Module()
        if not localReadCodeB:
          localReadCodeB = Module()
        if not localReadCodeM:
          localReadCodeM = Module()
        if localReadCodeA.items():
          localReadCodeAB.add(localReadCodeA.popFirstItem())
        if localReadCodeM.items():
          localReadCodeAB.add(localReadCodeM.popFirstItem())
        if localReadCodeB.items():
          localReadCodeAB.add(localReadCodeB.popFirstItem())
        if localReadCodeA.itemsSize():
          localReadCodeAB.addItems(localReadCodeA.popFirstNItems(localReadCodeA.itemsSize()))
        if localReadCodeM.itemsSize():
          localReadCodeAB.addItems(localReadCodeM.popFirstNItems(localReadCodeM.itemsSize()))
        if localReadCodeB.itemsSize():
          localReadCodeAB.addItems(localReadCodeB.popFirstNItems(localReadCodeB.itemsSize()))
      localReadItems = localReadCodeAB.flatitems()
      localReadItemsThisLoop = localReadItems if iteration < isBarrier else []
      localReadItemsNextLoop = localReadItems if iteration >= isBarrier else []

      #####
      # Prepare pack Code                for B:
      # since the mfma reuse B first =>    for A: mfma[A][B]
      # we need 1 vector A and 1 vector B for first mfma
      # then we prepare remaining A, then remaining B
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      ####
      if self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]:
        instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      else:
        instPerRegPack = 1 if (kernel["ProblemType"]["DataType"].numRegisters() == 0.25) else 0
      instPerPackA    = 0 if kernel["UnrollMajorLDSA"] else int(kernel["MIInputPerThreadA"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      instPerPackB    = 0 if kernel["UnrollMajorLDSB"] else int(kernel["MIInputPerThreadB"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      if kernel["ConvertAfterDS"]:
        if kernel["ProblemType"]["DataTypeA"].isAnyFloat8():
          if kernel["UnrollMajorLDSA"]:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackA = 2 * self.states.numReadsIterCoalescedA if(iteration % self.states.numReadsIterCoalescedA == 0) else 0
            else:
              instPerPackA = 6 * self.states.numReadsIterCoalescedA if(iteration % self.states.numReadsIterCoalescedA == 0) else 0
          elif self.states.lrvwTileA == 1:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackA = 4
            else:
              instPerPackA = 8
          elif self.states.lrvwTileA == 2:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackA = 8
            else:
              instPerPackA = 16
          elif self.states.lrvwTileA == 4:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackA = 20
            else:
              instPerPackA = 36
          elif self.states.lrvwTileA == 8:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackA = 44
            else:
              instPerPackA = 76
        if kernel["ProblemType"]["DataTypeB"].isAnyFloat8():
          if kernel["UnrollMajorLDSB"]:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackB = 2 * self.states.numReadsIterCoalescedB if(iteration % self.states.numReadsIterCoalescedB == 0) else 0
            else:
              instPerPackB = 6 * self.states.numReadsIterCoalescedB if(iteration % self.states.numReadsIterCoalescedB == 0) else 0
          elif self.states.lrvwTileB == 1:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackB = 4
            else:
              instPerPackB = 8
          elif self.states.lrvwTileB == 2:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackB = 8
            else:
              instPerPackB = 16
          elif self.states.lrvwTileB == 4:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackB = 20
            else:
              instPerPackB = 36
          elif self.states.lrvwTileB == 8:
            if self.states.asmCaps["Hascvtf16_fp8"]:
              instPerPackB = 44
            else:
              instPerPackB = 76

      instPerPackM = 0
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"] and not kernel["UnrollMajorLDSMetadata"]:
        instPerPackM = 1
        if self.states.lrvwTileMetadata > 1:
          if kernel["MIInputPerThreadMetadata"] == 1:
            instPerPackM = 1.5
          elif kernel["MIInputPerThreadMetadata"] == 4:
            instPerPackM = 3
        elif kernel["MIInputPerThreadMetadata"] == 4:
          instPerPackM = 3
      packItems = []
      packItemsA = []
      packItemsB = []
      packItemsM = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB,self.states.numReadsIterCoalescedMetadata)) ]
        packINtemsA = packINtems
        packINtemsB = packINtems
        packINtemsM = packINtems
        if schedulePackConsiderMetadata:
          packINtemsA = [ [] for j in range(max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB,self.states.numReadsIterCoalescedMetadata)) ]
          packINtemsB = [ [] for j in range(max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB,self.states.numReadsIterCoalescedMetadata)) ]
          packINtemsM = [ [] for j in range(max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB,self.states.numReadsIterCoalescedMetadata)) ]

        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        packM = packCode.findNamedItem("packMetadata_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Module()
        packBItems = packB.flatitems()
        if not packM:
          packM = Module()
        packMItems = packM.flatitems()

        if packAItems:
          if kernel["ConvertAfterDS"] and kernel["ProblemType"]["DataTypeA"].isAnyFloat8():
            for n in range(instPerPackA):
              packINtemsA[0].append(packAItems.pop(0))
          else:
            for j in range(self.states.numReadsIterCoalescedA):
              for n in range(instPerPackA):
                packINtemsA[j].append(packAItems.pop(0))

        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          for j in range(self.states.numReadsIterCoalescedMetadata):
            for n in range(ceil(instPerPackM)):
              if packMItems:
                packINtemsM[j].append(packMItems.pop(0))
              else:
                break

        if packBItems:
          if kernel["ConvertAfterDS"] and kernel["ProblemType"]["DataTypeB"].isAnyFloat8():
            for n in range(instPerPackB):
              packINtemsB[0].append(packBItems.pop(0))
          else:
            for j in range(self.states.numReadsIterCoalescedB):
              for n in range(instPerPackB):
                packINtemsB[j].append(packBItems.pop(0))

        scheduleTF32Emu = kernel["UseF32XEmulation"]
        if scheduleTF32Emu:
          if kernel["UseDirect32XEmulation"]:
            # none needed for direct
            instPerPackA = 0
            instPerPackB = 0
          else:
            instPerPackA = self.states.numPackCvt
            instPerPackB = self.states.numPackCvt
          while packAItems or packBItems:
            KernelWriter._packItemsConditional(instPerPackA, packAItems, packItems, ["__TF32_1", "__TF32_2"])
            KernelWriter._packItemsConditional(instPerPackB, packBItems, packItems, ["__TF32_1", "__TF32_2"])
        else:
          while packAItems:
            if kernel["ConvertAfterDS"] and kernel["ProblemType"]["DataTypeA"].isAnyFloat8():
              for n in range(instPerPackA):
                if packAItems:
                  packINtemsA[0].append(packAItems.pop(0))
                else:
                  break
            else:
              for j in range(self.states.numReadsIterCoalescedA):
                for n in range(instPerPackA):
                  if packAItems:
                    packINtemsA[j].append(packAItems.pop(0))
                  else:
                    break
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            while packMItems:
              for j in range(self.states.numReadsIterCoalescedMetadata):
                for n in range(ceil(instPerPackM)):
                  if packMItems:
                    packINtemsM[j].append(packMItems.pop(0))
                  else:
                    break
          while packBItems:
            if kernel["ConvertAfterDS"] and kernel["ProblemType"]["DataTypeB"].isAnyFloat8():
              for n in range(instPerPackB):
                if packBItems:
                  packINtemsB[0].append(packBItems.pop(0))
                else:
                  break
            else:
              for j in range(self.states.numReadsIterCoalescedB):
                for n in range(instPerPackB):
                  if packBItems:
                    packINtemsB[j].append(packBItems.pop(0))
                  else:
                    break

        for j in range(max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB,self.states.numReadsIterCoalescedMetadata)):
          if schedulePackConsiderMetadata:
            packItemsA += packINtemsA.pop(0)
            packItemsB += packINtemsB.pop(0)
            packItemsM += packINtemsM.pop(0)
          elif not scheduleTF32Emu:
            packItems += packINtems.pop(0)

        if schedulePackConsiderMetadata:
          packItems = packItemsA + packItemsB + packItemsM
      # remove s_nop for packing
      # we will add s_nop if needed
      if macIterItems:
        if isinstance(macIterItems[0], SNop):
          macIterItems.pop(0)

      ####
      # scheduled local read to previous iterations
      ####
      if kernel["ClusterLocalRead"]:
        for vacancy in self.localReadsVacancy:
          # {"items","latencyLeft","atIter","atMfmaIndex","noReadsAtThisIter"}
          for localRead in list(localReadItemsThisLoop):
            if vacancy["latencyLeft"] >= localRead.issueLatency() * 2:
              vacancy["latencyLeft"] -= localRead.issueLatency() * 2
              vacancy["items"].add(localRead)
              localReadItemsThisLoop.remove(localRead)
              if vacancy["atMfmaIndex"] > self.states.sync1LdsMfmaIndex and kernel["1LDSBuffer"]:
                self.states.overflowedResources = 5
              # update waitCnt
              if self.states.numItersPLR:
                for readsIter in range(vacancy["atIter"], iteration + self.states.numItersPLR):
                  if (vacancy["atMfmaIndex"] % numMfmaPerIter == 0 or readsIter != vacancy["atIter"]) and \
                      (vacancy["noReadsAtThisIter"] or readsIter <= vacancy["atIter"] + self.states.numItersPLR):
                    if isinstance(self.localReadsWait[readsIter], SWaitCnt):
                      self.localReadsWait[readsIter].dscnt += 1
            else:
              # make sure the localread sequence remain the same
              vacancy["latencyLeft"] = 0
      numReadsInst = len(localReadItemsThisLoop) if iteration < isBarrier else len(localReadItemsNextLoop)

      # Add space to avoid LR FIFO stall
      # lrStallLatencyBuffer:
      # 40 quad-cycle - 4 x miLatency for b128
      # 20 quad-cycle - 4 x miLatency for b64 (equal to one miLatency)
      # 10 quad-cycle - 4 x miLatency for b32 (no stall)
      # so no stall happen for b64/b32/b16
      if iteration == 0:
        self.localReadThisLoopFIFO = []
        self.localReadNextLoopFIFO = []
      def checkLocalReadFIFOFull(currentMFMA, fifo, lrItems, numLR, numLREven):
        numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1] * kernel["LocalSplitU"]
        if numLREven >= 1.0:
          numToBeSched = max(ceil(numLREven), numLR)
          for n in range(numToBeSched):
            if len(lrItems) <= n:
              break
            item = lrItems[n]
            if not isinstance(item, DSLoadB128):
              continue
            if len(fifo) < (16 / numWaves):
              fifo.append(currentMFMA)
            else:
              fifo.pop(0)
              fifo.append(currentMFMA)
          return numToBeSched
        numToBeIssued = 0
        for n in range(numLR):
          if len(lrItems) <= n:
            break
          item = lrItems[n]
          if not isinstance(item, DSLoadB128):
            numToBeIssued += 1
            continue
          # The FIFO length is 16 so that each wave has 16/numWaves buffer.
          lrStallLatencyBuffer = 40 - ((16 / numWaves) * self.states.miLatency)
          if len(fifo) < (16 / numWaves):
            fifo.append(currentMFMA)
          else:
            oldMFMA = fifo[0]
            if (currentMFMA - oldMFMA) * self.states.miLatency >= lrStallLatencyBuffer:
              fifo.pop(0)
              fifo.append(currentMFMA)
            else:
              break
          numToBeIssued += 1
        return numToBeIssued

      oneBufferScheduling = kernel["1LDSBuffer"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]

      insertedPackA = 0
      insertedPackB = 0
      insertedPackM = 0

      def hasDependency(lr: DSLoadInstruction, inst: Instruction) -> bool:
        lrDataReg = lr.dst

        if isinstance(inst, MFMAInstruction):
          srcRegs = [inst.a, inst.b,]
        elif isinstance(inst, SMFMAInstruction):
          srcRegs = [inst.a, inst.b, inst.metadata]
        else:
          srcRegs = inst.srcs

        return any((lrDataReg & r) for r in srcRegs if isinstance(r, RegisterContainer))

      def hasAnyDependency(lr: DSLoadInstruction, insts: List[Instruction]):
        return any(hasDependency(lr, inst) for inst in insts)

      def calculateRangeAndUpdateCounter(itemCounter, writeCounters, length):
        newItemCounter = itemCounter + length
        numLoops = 0
        for count in writeCounters:
          if count > newItemCounter:
            break
          numLoops += 1
        return numLoops, newItemCounter

      itemCounter = 0
      for i in range(numMfmaPerIter):
        mfmaIndex = iteration * numMfmaPerIter + i
        insertInst = countInstruction(iterCode)
        iterCode.addComment0(" mfmaIndex:%u " %(mfmaIndex))

        ####
        # scheduled local read
        ####
        numReadsInst = len(localReadItemsThisLoop)
        readLeft = numReadsInst
        latencyLeft = self.states.miLatencyLeft
        if iteration < isBarrier:
          # with PrefetchLocalRead, localreads can interleave with mfma
          if self.states.numItersPLR:
            # take ds_write into account to schedule ds_read, assume A and B localwrite have same width (TLDS=1)
            if (mfmaIndex >= self.states.lwStartMfmaIndex) and not countGlobalRead(globalReadCode):
              writeItemLength = (localWriteCodeCounts[-1] - itemCounter) if localWriteCodeCounts else 0
              writeItemLength = min(writeItemLength, self.states.numLocalWriteModPerMfma)
              numLoops, _ = calculateRangeAndUpdateCounter(itemCounter, localWriteCodeCounts, writeItemLength)
              for j in range(min(len(writeItems), numLoops)):
                if countLocalWrite(writeItems[j]):
                  latencyLeft -= (tPA["localWriteInstruction"].issueLatency*2)
            readLeftLROPT = 0
            for j in range(len(localReadItemsThisLoop)):
              latencyLeft -= localReadItemsThisLoop[j].issueLatency()*2
              readLeftLROPT += 1 if latencyLeft >= 0 else 0
            # at least 1 instruction
            readLeftLROPT = max(readLeftLROPT,1)
            # evenly schedule localread with each mfma
            readLeftLREven = numReadsInst / (numMfmaPerIter - i)
            # we want no localreads at first mfma
            if (iteration == 0) and numMfmaPerIter != 1:
              if i == 0:
                readLeftLREven = 0
                readLeftLROPT = 0
              # rest mfma help to schedule those localReads
              else:
                readLeftLREven = numReadsInst / (numMfmaPerIter - i)
            # if there are too many localreads, change strategy to even.
            readLeft = checkLocalReadFIFOFull(mfmaIndex, self.localReadThisLoopFIFO, localReadItemsThisLoop, readLeftLROPT, readLeftLREven)
          elif kernel["EnableMatrixInstruction"] and self.do["OptimizeNumItersPLR0"]:
            # if numItersPLR == 0, try to schedule local reads with instruction level prefetch.
            mfmas = getMFMAs(macIterCode)
            if i + 1 != numMfmaPerIter:
              numLocalReadShouldSchedule = 0
              # prefetch load for next wave tile along M since we re-use B first.
              tileM: int = kernel["MIWaveTileA"]
              instsToCheck = mfmas[i:min(i+tileM+1, numMfmaPerIter)] + packItems
              localReadItemsThisLoop = sorted(localReadItemsThisLoop, key=lambda o: hasAnyDependency(o, instsToCheck), reverse=True)

              for lr in localReadItemsThisLoop:
                if hasAnyDependency(lr, instsToCheck):
                  numLocalReadShouldSchedule += 1
                else:
                  break
              readLeft = numLocalReadShouldSchedule
              if len(localReadItemsThisLoop) and readLeft == 0:
                numRemainMfmas = numMfmaPerIter - i - 1
                avgNumMfmasPerLr = numRemainMfmas // len(localReadItemsThisLoop)

                if avgNumMfmasPerLr > 0 and i % avgNumMfmasPerLr == 0:
                  readLeft = 1
              latencyLeft -= readLeft * tPA["localWriteInstruction"].issueLatency * 2
            else:
              readLeft = len(localReadItemsThisLoop)
              latencyLeft -= sum(j.issueLatency()*2 for j in localReadItemsThisLoop)
          else:
            latencyLeft -= sum(j.issueLatency()*2 for j in localReadItemsThisLoop)

        # force to schedule all remaining localreads before start to schedule localwrite.
        if mfmaIndex == self.states.sync1LdsMfmaIndex and oneBufferScheduling:
          iterCode.addComment0("schedule remaining localreads for one buffer scheduling")
          while (localReadItemsThisLoop):
            item = localReadItemsThisLoop.pop(0)
            iterCode.add(item)
            localReadsIssuedInThisIter += 1
            if (i == 0):
              localReadsWaitcnt += 1
          item = Module()
          iterCode.add(item)
          self.localReadsVacancy.append({ "items": item, \
                                          "latencyLeft": sys.maxsize, \
                                          "atIter": iteration, \
                                          "atMfmaIndex": mfmaIndex, \
                                          "noReadsAtThisIter": numReadsInst == 0, \
                                        })
        # if start to schedule localwrite, but still have localreads not scheduled yet,
        # reject to use 1LDSB, since it will write and read same lds buffer at same time.
        if mfmaIndex > self.states.sync1LdsMfmaIndex and localReadItemsThisLoop and oneBufferScheduling:
          #TODO: can we remove this restriction?
          self.states.overflowedResources = 5
        for j in range(readLeft):
          if localReadItemsThisLoop:
            item = localReadItemsThisLoop.pop(0)
            iterCode.add(item)
            localReadsIssuedInThisIter += 1
            if (i == 0):
              localReadsWaitcnt += 1
        if not localReadItemsThisLoop and latencyLeft > 0 and iteration < isBarrier and \
            not(mfmaIndex > self.states.sync1LdsMfmaIndex and kernel["1LDSBuffer"]):
          item = Module()
          item.addComment0("localReadsVacancy: latencyLeft %d"%(latencyLeft))
          iterCode.add(item)
          self.localReadsVacancy.append({ "items": item, \
                                          "latencyLeft": latencyLeft, \
                                          "atIter": iteration, \
                                          "atMfmaIndex": mfmaIndex, \
                                          "noReadsAtThisIter": numReadsInst == 0, \
                                        })

        ####
        # scheduled global read
        ####
        loadModules = globalReadCode.popFirstNItems(min(globalReadCode.itemsSize(), self.states.numGlobalReadInsPerMfma))
        iterCode.addItems(loadModules)
        # schedule remaining globalReadInst
        if mfmaIndex == self.states.grEndMfmaIndex:
          while globalReadCode.itemsSize() and \
              (countGlobalRead(globalReadCode) or kernel["PrefetchGlobalRead"] == 2):
            loadModule = globalReadCode.popFirstItem()
            iterCode.add(loadModule)
        # schedule remaining globalReadIncInst
        if (i == numMfmaPerIter - 1 and globalReadCode.itemsSize()) or \
           (i == 0 and globalReadCode.itemsSize() and (iteration == 1 and kernel["ForceUnrollSubIter"])):
          loadModules = globalReadCode.popFirstNItems(globalReadCode.itemsSize())
          iterCode.addItems(loadModules)

        ####
        # scheduled local write
        ####
        if kernel["1LDSBuffer"] and mfmaIndex == self.states.sync1LdsMfmaIndex:
          barrier = Module()
          barrier.addComment0("1 LDS buffer: read-sync-write")
          barrier.add(SWaitCnt(dscnt=0, comment=""))
          barrier.add(SBarrier())
          iterCode.add(barrier)

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            lwStartOffset = 0
            if (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
              lwStartOffset = 2
            #  if (mfmaIndex == self.states.lwStartMfmaIndex or mfmaIndex == self.states.syncPlrMfmaIndex+2):
            if (mfmaIndex == self.states.lwStartMfmaIndex + lwStartOffset or mfmaIndex == self.states.syncPlrMfmaIndex+1) :
              flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == self.states.grEndMfmaIndex or mfmaIndex == self.states.syncPlrMfmaIndex+1) :
            withGL = (not NLLlast)
            withDTLload = (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and withGL
            startIndex = 0 if withDTLload else 1
            if (mfmaIndex == startIndex or withGL and mfmaIndex == self.states.syncPlrMfmaIndex+1):
              flagInsert = True
          if flagInsert:
            iterCode.add(SSetPrior(prior=3, comment="store optimization"))
        if (mfmaIndex >= self.states.lwStartMfmaIndex):
          numLoops, itemCounter = calculateRangeAndUpdateCounter(itemCounter, localWriteCodeCounts, self.states.numLocalWriteModPerMfma)
          for j in range(min(len(writeItems), numLoops)):
            # in case there are localWrite and globalread in same iteration
            # we need to make sure globalRead before localWrite
            if writeItems and not countGlobalRead(globalReadCode):
              localWriteCodeCounts.pop(0)
              writeItem = writeItems.pop(0)
              iterCode.add(writeItem)
              # if there is localWrite at first mfma, need to skip it in waitcnt.
              if i == 0:
                skipLocalWriteWaitcnt += countLocalWrite(writeItem) + countDSStoreB256(writeItem)
              if not localReadItemsThisLoop:
                self.states.perIterLocalWriteCanSkip[iteration] += countLocalWrite(writeItem) + countDSStoreB256(writeItem)
          if kernel["ForceUnrollSubIter"] and (writeItems and i == (numMfmaPerIter - 1)):
            # if ForceUnrollSubIter, we need to schedule all localWrite in last mfma
            while writeItems:
              writeItem = writeItems.pop(0)
              iterCode.add(writeItem)
        if mfmaIndex == self.states.lwEndMfmaIndex:
          while writeItems:
            localWriteCodeCounts.pop(0)
            writeItem = writeItems.pop(0)
            # generate all remaining pre code before the first Store C
            iterCode.add(writeItem)
            if i == 0:
              skipLocalWriteWaitcnt += countLocalWrite(writeItem) + countDSStoreB256(writeItem)
            if not localReadItemsThisLoop:
              self.states.perIterLocalWriteCanSkip[iteration] += countLocalWrite(writeItem) + countDSStoreB256(writeItem)

        ####
        # scheduled pointer
        ####
        if mfmaIndex == self.states.lwEndMfmaIndex:
          iterCode.add(pointerLWCode)
        if i == numMfmaPerIter - 1:
          iterCode.add(pointerLRCode)

        ####
        # scheduled sync
        ####
        if mfmaIndex == self.states.syncPlrMfmaIndex and self.states.numItersPLR:
          iterCode.add(waitLWCode)
          iterCode.add(syncCode)

        ####
        # scheduled local read for next loop
        # localReads for next loop should after barrier
        ####
        latencyLeft = self.states.miLatencyLeft
        numReadsInst = len(localReadItemsNextLoop)
        if self.states.numItersPLR and iteration >= isBarrier:
          readLeftLROPT = 0
          for j in range(len(localReadItemsNextLoop)):
            latencyLeft -= localReadItemsNextLoop[j].issueLatency()*2
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # at least 1 instruction
          readLeftLROPT = max(readLeftLROPT,1)
          readLeftLREven = numReadsInst / (numMfmaPerIter - i)
          # we want no localreads at barrier mfma
          if (iteration == isBarrier) and numMfmaPerIter != 1:
            numMfmaForLR = self.states.numMfmaForNextLoopLR
            if i < numMfmaPerIter - numMfmaForLR:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = numReadsInst / (numMfmaPerIter - i)
          # if there are too many localreads, change strategy to even.
          readLeft = checkLocalReadFIFOFull(mfmaIndex, self.localReadNextLoopFIFO, localReadItemsNextLoop, readLeftLROPT, readLeftLREven)
        for j in range(readLeft):
          if localReadItemsNextLoop:
            item = localReadItemsNextLoop.pop(0)
            iterCode.add(item)
            if (i == 0):
              localReadsWaitcnt += 1

        ####
        # scheduled wait localReads
        ####
        if self.states.numItersPLR == 0 and kernel["EnableMatrixInstruction"] and self.do["OptimizeNumItersPLR0"]:
          dscnt = -1
          mfmas = getMFMAs(macIterCode)
          ## To support do["MAC"] is False
          mfma = [mfmas[i],] if len(mfmas) > 0 else []
          instsToCheck = mfma + packItems
          numDsInsts = 0
          lastDsCnt = -1
          for ds in filter(lambda j: isinstance(j, (DSLoadInstruction, DSStoreInstruction, SWaitCnt)), reversed(prevIterCode.flatitems() + iterCode.flatitems())):
            if isinstance(ds, DSLoadInstruction) and hasAnyDependency(ds, instsToCheck):
              break

            if isinstance(ds, DSLoadInstruction) and not hasAnyDependency(ds, instsToCheck) or isinstance(ds, DSStoreInstruction):
              numDsInsts += 1
            elif isinstance(ds, SWaitCnt):
              if ds.dscnt >= 0 and lastDsCnt == -1:
                lastDsCnt = ds.dscnt + numDsInsts

          if lastDsCnt != numDsInsts:
            if lastDsCnt == -1 or (lastDsCnt >= 0 and lastDsCnt > numDsInsts):
              waitDsRead = SWaitCnt(dscnt=numDsInsts, comment="Wait for dependent lr")
              iterCode.add(waitDsRead)
        else:
          if kernel["UnrollMajorLDSB"] and not (kernel["ProblemType"]["DataTypeB"].isAnyFloat8() and kernel["ConvertAfterDS"]):
            if iteration == 0 and i == kernel["MIWaveTileA"]:
              # add 1 more waitcnt before using ds read data
              waitCode2 = deepcopy(waitCode)
              waitCode2.dscnt = localReadsIssuedInThisIter
              iterCode.add(waitCode2)
          if i == 0:
            iterCode.add(waitCode)

        ####
        # scheduled pack
        ####
        _instPerPackA = 0
        _instPerPackB = 0
        _instPerPackM = 0
        instPackLast = []
        if packItems:

          # check the remain latency before mfma
          currentInsertInst = countInstruction(iterCode) - insertInst
          latencyLeft = self.states.miLatencyLeft - currentInsertInst
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          _instPerPackA = instPerPackA if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packAIdx += _instPerPackA
          _instPerPackB = instPerPackB if i % kernel["MIWaveTileA"] == 0 else 0
          packBIdx += _instPerPackB
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            if kernel["ProblemType"]["Sparse"] == 2:
              _instPerPackM = instPerPackM if i % kernel["MIWaveTileA"] == 0 else 0
            else:
              _instPerPackM = instPerPackM if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0

            packMIdx += _instPerPackM
          # blockWidth < 1, means 0.5 or 0.25 (BF,H,Int8)
          if self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]:
            packAIdx = packAIdx if tPA["bpe"] < 4 and (not kernel["UnrollMajorLDSA"] or kernel["ConvertAfterDS"]) else 0
            packBIdx = packBIdx if tPB["bpe"] < 4 and (not kernel["UnrollMajorLDSB"] or kernel["ConvertAfterDS"]) else 0
          else:
            packAIdx = packAIdx if tPA["localReadInstruction"].blockWidth == 0.25 else 0
            packBIdx = packBIdx if tPB["localReadInstruction"].blockWidth == 0.25 else 0
          numPack = (packAIdx + packBIdx)
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            packMIdx = packMIdx if not kernel["UnrollMajorLDSMetadata"] else 0
            numPack += packMIdx
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u, packMIdx:%u" %(packAIdx,packBIdx,packMIdx))
          else:
            iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))

          if not schedulePackConsiderMetadata:
              if kernel["UseF32XEmulation"]:
                tmp = []
                curPackIdx += KernelWriter._packItemsConditional(instPerPackA, packItems, tmp, ["__TF32_1_A", "__TF32_2_A"])
                for n in tmp:
                  iterCode.add(n)
              else:
                # we put 2 pack in each mfma
                for j in range(instPerPackA):
                  if packItems:
                    iterCode.add(packItems.pop(0))
                    curPackIdx += 1
              if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
                for j in range(ceil(instPerPackM)):
                  if packItems:
                    iterCode.add(packItems.pop(0))
                    curPackIdx += 1
              if kernel["UseF32XEmulation"]:
                tmp = []
                curPackIdx += KernelWriter._packItemsConditional(instPerPackB, packItems, tmp, ["__TF32_1_B", "__TF32_2_B"])
                for n in tmp:
                  iterCode.add(n)
              else:
                for j in range(instPerPackB):
                  if packItems:
                    iterCode.add(packItems.pop(0))
                    curPackIdx += 1
              # since packed register need to wait 2 quad cycle to finish packing
              # we insert pack instruction if we can, or s_nop
              while curPackIdx < numPack+2:
                if packItems:
                  iterCode.add(packItems.pop(0))
                  curPackIdx += 1
                else:
                  iterCode.add(SNop(waitState=1, comment="VALU packing writes to be consumed by matrix instruction"))
                  curPackIdx += 1
                  break
              if kernel["UseF32XEmulation"]:
                # HACK add dummy waits btween swap and mfmas. TODO: improve pack scheduling to avoid this
                numDummy = 0 if kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16 else 1
                for numd in range(numDummy):
                  iterCode.add(SNop(waitState=0, comment="VALU packing writes to be consumed by matrix instruction"))
          else:

            desiredPack = instPerPackA + instPerPackB + ceil(instPerPackM)
            # Step 1
            # put the required pack into mfma iter
            for j in range(_instPerPackA):
              if packItemsA:
                # Skip if the required pack has already been placed in the previous mfma iter.
                if numPackedA >= packAIdx:
                  break
                iterCode.add(packItemsA.pop(0))
                curPackIdx += 1
                numPackedA += 1
                latencyLeft -= 1
                insertedPackA += 1
                desiredPack -= 1
                if len(instPackLast) == 2:
                  instPackLast.pop(0)
                instPackLast.append("A")


            # check if the inserted pack instructions are fulfilled instPerPack.
            # If not, insert the next pack instructions until satisfied.
            # The unsatisfied is usually caused by Step 3 of the previous round.
            instPackLeft = (insertedPackA % instPerPackA) if instPerPackA > 0 else 0
            instPackLeft = (instPerPackA - instPackLeft) if instPackLeft > 0 else 0
            if instPackLeft > 0:
              insertedPackA = 0
              for j in range(instPackLeft):
                if packItemsA:
                  iterCode.add(packItemsA.pop(0))
                  curPackIdx += 1
                  numPackedA += 1
                  latencyLeft -= 1
                  desiredPack -= 1
                  if len(instPackLast) == 2:
                    instPackLast.pop(0)
                  instPackLast.append("A")

            if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
              # put the required pack into mfma iter
              for j in range(ceil(_instPerPackM)):
                if packItemsM:
                  # Skip if the required pack has already been placed in the previous mfma iter.
                  if numPackedM >= packMIdx:
                    break
                  iterCode.add(packItemsM.pop(0))
                  curPackIdx += 1
                  numPackedM += 1
                  latencyLeft -= 1
                  insertedPackM += 1
                  desiredPack -= 1
                  if len(instPackLast) == 2:
                    instPackLast.pop(0)
                  instPackLast.append("M")

              # check if the inserted pack instructions are fulfilled instPerPack.
              # If not, insert the next pack instructions until satisfied.
              # The unsatisfied is usually caused by Step 3 of the previous round.
              instPackLeft = (insertedPackM % ceil(instPerPackM)) if instPerPackM > 0 else 0
              instPackLeft = (ceil(instPerPackM) - instPackLeft) if instPackLeft > 0 else 0
              if instPackLeft > 0:
                insertedPackM = 0
                for j in range(instPackLeft):
                  if packItemsM:
                    iterCode.add(packItemsM.pop(0))
                    curPackIdx += 1
                    numPackedM += 1
                    latencyLeft -= 1
                    desiredPack -= 1
                    if len(instPackLast) == 2:
                      instPackLast.pop(0)
                    instPackLast.append("M")

            # put the required pack into mfma iter
            for j in range(_instPerPackB):
              if packItemsB:
                # Skip if the required pack has already been placed in the previous mfma iter.
                if numPackedB >= packBIdx:
                  break
                iterCode.add(packItemsB.pop(0))
                curPackIdx += 1
                numPackedB += 1
                latencyLeft -= 1
                insertedPackB += 1
                desiredPack -= 1
                if len(instPackLast) == 2:
                  instPackLast.pop(0)
                instPackLast.append("B")

            # check if the inserted pack instructions are fulfilled instPerPack.
            # If not, insert the next pack instructions until satisfied.
            # The unsatisfied is usually caused by Step 3 of the previous round.
            instPackLeft = (insertedPackB % instPerPackB) if instPerPackB > 0 else 0
            instPackLeft = (instPerPackB - instPackLeft) if instPackLeft > 0 else 0
            if instPackLeft > 0:
              insertedPackB = 0
              for j in range(instPackLeft):
                if packItemsB:
                  iterCode.add(packItemsB.pop(0))
                  curPackIdx += 1
                  numPackedB += 1
                  latencyLeft -= 1
                  desiredPack -= 1
                  if len(instPackLast) == 2:
                    instPackLast.pop(0)
                  instPackLast.append("B")

            # Step 2
            # put the desired pack into mfma iter
            if latencyLeft > 0:
              remainDesiredPack = desiredPack
              for j in range(remainDesiredPack):
                if packItemsA:
                  iterCode.add(packItemsA.pop(0))
                  curPackIdx += 1
                  numPackedA += 1
                  latencyLeft -= 1
                  desiredPack -= 1
                  if len(instPackLast) == 2:
                    instPackLast.pop(0)
                  instPackLast.append("A")

            if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"] and latencyLeft > 0:
              remainDesiredPack = desiredPack
              for j in range(remainDesiredPack):
                if packItemsM:
                  iterCode.add(packItemsM.pop(0))
                  curPackIdx += 1
                  numPackedM += 1
                  latencyLeft -= 1
                  desiredPack -= 1
                  if len(instPackLast) == 2:
                    instPackLast.pop(0)
                  instPackLast.append("M")

            if latencyLeft > 0:
              remainDesiredPack = desiredPack
              for j in range(remainDesiredPack):
                if packItemsB:
                  iterCode.add(packItemsB.pop(0))
                  curPackIdx += 1
                  numPackedB += 1
                  latencyLeft -= 1
                  desiredPack -= 1
                  if len(instPackLast) == 2:
                    instPackLast.pop(0)
                  instPackLast.append("B")

            # Step 3
            # since packed register need to wait 2 quad cycle to finish packing
            # we insert pack instruction if we can, or s_nop
            remainLatency = 0
            iterCode.addComment0("pack scheduling: curPackIdx:%u, numPack:%u, instPackLast:%s" %(curPackIdx,numPack,instPackLast))
            if curPackIdx < numPack + 2:
              if len(instPackLast):
                remainLatency = 2
              else:
                remainLatency = 0
                remainPacked = len(packItemsA) + len(packItemsB) + len(packItemsM)
                desiredPack = max(0, desiredPack)
                if remainPacked > 0:
                  remainLatency = min(remainPacked, desiredPack)
            elif curPackIdx >= numPack:
              # when the number of inserted packs is >= the number of desired packs
              # check the last 2 inserted packs to see if we need to add extra instructions after the last inersted pack.
              remainLatency = len(instPackLast)
              while len(instPackLast):
                instLast = instPackLast.pop()
                if instLast == "A":
                  if numPackedA <= packAIdx:
                    break
                elif instLast == "M":
                  if numPackedM <= packMIdx:
                    break
                elif instLast == "B":
                  if numPackedB <= packBIdx:
                    break
                remainLatency -= 1
            instPackLast.clear()

            # if there is only on pack remain in the pool, inserts it as well.
            # this may help to reduce the extra s_nop.
            rA = len(packItemsA)
            rB = len(packItemsB)
            rM = len(packItemsM)
            if (rA + rB + rM - remainLatency) == 1:
                remainLatency += 1

            while remainLatency > 0:
              if packItemsA:
                iterCode.add(packItemsA.pop(0))
                curPackIdx += 1
                numPackedA += 1
                remainLatency -= 1
                insertedPackA += 1
              elif packItemsM:
                iterCode.add(packItemsM.pop(0))
                curPackIdx += 1
                numPackedM += 1
                remainLatency -= 1
                insertedPackM += 1
              elif packItemsB:
                iterCode.add(packItemsB.pop(0))
                curPackIdx += 1
                numPackedB += 1
                remainLatency -= 1
                insertedPackB += 1
              else:
                latency = remainLatency - 1
                iterCode.add(SNop(waitState=latency, comment="VALU packing writes to be consumed by matrix instruction"))
                remainLatency -= (latency+1)

        if not schedulePackConsiderMetadata:
          if i == numMfmaPerIter - 1:
            while packItems:
              iterCode.add(packItems.pop(0))
        else:
          if i == numMfmaPerIter - 1:
            while packItemsA:
              iterCode.add(packItemsA.pop(0))
            while packItemsM:
              iterCode.add(packItemsM.pop(0))
            while packItemsB:
              iterCode.add(packItemsB.pop(0))

        ####
        # scheduled mfma dependency
        ####
        if mfmaIndex > 0:
          currentInsertInst = countInstruction(iterCode) - insertInst
          if currentInsertInst < self.states.miDependency:
            iterCode.add(SNop(waitState=(self.states.miDependency-currentInsertInst-1), comment="Dependency"))

        ####
        # scheduled mfma
        ####
        iterCode.add(macIterItems.pop(0) if macIterItems else Module())

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            #  if (mfmaIndex == self.states.syncPlrMfmaIndex or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)):
            if (mfmaIndex == self.states.syncPlrMfmaIndex - 1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
                flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == mfmaIndex == self.states.syncPlrMfmaIndex - 1 or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
            insertPos1 = self.states.grEndMfmaIndex
            if not kernel["NoLdsWriteCode"]:
              insertPos1 = self.states.lwStartMfmaIndex - 1
            withGL = (not NLLlast)
            if withGL and (mfmaIndex == insertPos1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) or \
               (not withGL) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter // 2 - 1):
              flagInsert = True
          if flagInsert:
            iterCode.add(SSetPrior(prior=0, comment="store optimization"))
      while macIterItems:
        iterCode.add(macIterItems.pop(0))
    else:
      assert 0, "Unsupported scheduleIterAlg=%u"%self.states.scheduleIterAlg

    if isinstance(waitCode, SWaitCnt):

      # Set the waitCount, based on the new iter schedule
      dscnt = waitCode.dscnt
      localReads = 0
      localWrites = 0
      if kernel["EnableMatrixInstruction"]:
        # dataAtIter      : the data we wait is read at which iteration
        # numReadsIter    : in this loop, number of iteration we have read (data used in current loop)
        dataAtIterA = iteration//self.states.numIterPerCoalescedReadA - self.states.numItersPLR
        dataAtIterB = iteration//self.states.numIterPerCoalescedReadB - self.states.numItersPLR
        numReadsIterA = min(iteration+1, kernel["LoopIters"]//self.states.numIterPerCoalescedReadA - self.states.numItersPLR)
        numReadsIterB = min(iteration+1, kernel["LoopIters"]//self.states.numIterPerCoalescedReadB - self.states.numItersPLR)
        skipReadsIterA = numReadsIterA - dataAtIterA - 1 if not dataAtIterA < max(dataAtIterA,dataAtIterB) else 0
        skipReadsIterB = numReadsIterB - dataAtIterB - 1 if not dataAtIterB < max(dataAtIterA,dataAtIterB) else 0
        # numPrefetchIter : in this loop, number of prefetch iteration we have read (data used in next loop)
        # currently we have localReadA and localReadB if iteration >= isBarrier
        # some case will not have localReads if PGR=0 or NoLoadLoop
        # known bug: wider localread + numItersPLR>1 may have chance to fail.
        numPrefetchIter = (iteration//(kernel["LoopIters"]-self.states.numItersPLR))*((iteration+1)-(kernel["LoopIters"]-self.states.numItersPLR)) if kernel["PrefetchGlobalRead"] else 0
        numPrefetchIter = 0 if iteration >= isBarrier and not hasLocalRead else numPrefetchIter
        skipReadsIterA += numPrefetchIter
        skipReadsIterB += numPrefetchIter
        # here the reads are prefetches so can skip them in the waitcnt
        # how many localreads can skip is based on how many iterations we prefetch.
        localReadsA = 0 if kernel["DirectToVgprA"] else self.states.numReadsPerIterA * skipReadsIterA
        localReadsB = 0 if kernel["DirectToVgprB"] else self.states.numReadsPerIterB * skipReadsIterB
        localReads += localReadsA + localReadsB
        # some of localReads is interleaved after waitcnt in SIA3
        if kernel["ScheduleIterAlg"] == 3 and self.states.numItersPLR and\
          (iteration < numReadsIterA or iteration < numReadsIterB or numPrefetchIter) and\
          not kernel["ForceUnrollSubIter"]:
          if ((iteration < numReadsIterA and not dataAtIterA < max(dataAtIterA,dataAtIterB)) or numPrefetchIter) and (not kernel["DirectToVgprA"]):
            localReads -= self.states.numReadsPerIterA
          if ((iteration < numReadsIterB and not dataAtIterB < max(dataAtIterA,dataAtIterB)) or numPrefetchIter) and (not kernel["DirectToVgprB"]):
            localReads -= self.states.numReadsPerIterB
          localReads += localReadsWaitcnt
          if iteration == 0 and kernel["UnrollMajorLDSB"] and not (kernel["ProblemType"]["DataTypeB"].isAnyFloat8() and kernel["ConvertAfterDS"]):
            # We issued LR with A[0]->B[0]->A[1:]->B[1:] order.
            # We need to calculate how many B[N:] needed by 1st mfma.
            # If not UnrollMajorLDSB, we have packB which will be issued ahead.
            # If ConvertAfterDS and DataTypeB=FP8, we have cvtB which will be issued ahead.
            localReadsNotWaited = self.states.numReadsPerIterB//kernel["InnerUnroll"] - self.states.numReadsPerUnrollB
            if localReadsNotWaited > 0:
              localReads += localReadsNotWaited
        elif kernel["ForceUnrollSubIter"]:
          localReads = 0

        dscnt += localReads
        iterCode.addComment0("numPrefetchIter=%u" % numPrefetchIter)
        iterCode.addComment0("dataAtIterA=%u numReadsIterA=%u skipReadsIterA=%u readsPerIterA=%u" % (dataAtIterA, numReadsIterA, skipReadsIterA, self.states.numReadsPerIterA))
        iterCode.addComment0("dataAtIterB=%u numReadsIterB=%u skipReadsIterB=%u readsPerIterB=%u" % (dataAtIterB, numReadsIterB, skipReadsIterB, self.states.numReadsPerIterB))
        if kernel["ScheduleIterAlg"] == 0 or kernel["ScheduleIterAlg"] == 1:
          for i in range (max(dataAtIterA,dataAtIterB),iteration+1):
            localWrites += countLocalWrite(self.codes.perIterLocalWrite[i][1])
            localWrites += countDSStoreB256(self.codes.perIterLocalWrite[i][1])
        # ScheduleIterAlg=2, localwrite is after waitCnt, no need to count it's current iteration.
        if kernel["ScheduleIterAlg"] == 3:
          for i in range (max(dataAtIterA,dataAtIterB)+1,iteration):
            localWrites += countLocalWrite(self.codes.perIterLocalWrite[i][1])
            localWrites += countDSStoreB256(self.codes.perIterLocalWrite[i][1])
          if kernel["ScheduleLocalWrite"] > 0:
            # current iteration localWrite count
            localWrites += skipLocalWriteWaitcnt
            # dataAtIter iteration localWrite count
            if self.states.numItersPLR:
              skipPreIterLW = self.states.perIterLocalWriteCanSkip[max(dataAtIterA,dataAtIterB)]
              localWrites += skipPreIterLW
        dscnt += localWrites
      else:
        for item in list(iterCode.items()):
          localReads  = countLocalRead(item)
          localWrites = countLocalWrite(item) + countDSStoreB256(item)
          if self.states.numItersPLR:
            # SQ: If PrefetchLocalRead = 1 and DepthU == LocalSplitU, then there is no double
            #  buffering and we must wait for all localReads but not localWrites.
            #  In that case, LoopIters == 1:
            if kernel["LoopIters"] > 1:
              # here the reads are prefetches so can skip them in the waitcnt
              dscnt += localReads
            # and the writes are targetting another section of LDS and are
            # synchronized through a different waitcnt than this one
            # (which is always just before the macs)
            dscnt += localWrites
          else:
            # we need to wait for all preceding reads before the macs
            # so only opportunity for optimization is if the writes are at the end
            if localReads:
              dscnt = 0 # reset to wait for all reads
            else:
              dscnt = localWrites  # this only survives if writes are at the end

      waitCode.comment += " old=%u, new=%u newLW=%u newLR=%u" % (waitCode.dscnt, dscnt, localWrites, localReads)
      if iteration == 0:
        waitCode.comment += " for iteration == 0"
      waitCode.dscnt = dscnt

    return iterCode

  ##############################################################################
  # returns list of modules or text
  ##############################################################################
  def setupNewTile(self, kernel, tensorParametersA, tensorParametersB, isOptNLL=False, forceNoTileCode=False, forceNoGRCode=False):
    module = Module("setupNewTile")

    ####################################
    # Global Read Addresses
    ####################################
    module.addComment2("Begin setupNewTile")

    # work-group assignments
    module.addComment1("global read addresses: work-group")
    if not forceNoTileCode:
      module.add(self.graWorkGroup(kernel, tensorParametersA, tensorParametersB))


    self.dontAppendCode = forceNoTileCode

    tPM = tensorParametersA["tpsMetadata"]
    tPMRef = tensorParametersA

    if kernel["ProblemType"]["Sparse"] == 2:
      tPM = tensorParametersB["tpsMetadata"]
      tPMRef = tensorParametersB

    if kernel["StreamK"] != 0:
      module.add(self.localReadAddresses(kernel, tensorParametersA, tensorParametersB, tPM))
      module.add(self.localWriteAddresses(kernel, tensorParametersA, tensorParametersB, tPM))

    module.add(self.removeGRSrdVariableSgprsFromPool(kernel))

    # tile assignments
    module.addComment1("global read addresses: tile offset assignment a")
    module.add(self.graTileAssignment(kernel, tensorParametersA))
    if kernel["ProblemType"]["Sparse"]:
      module.addComment1("global read addresses: tile offset assignment metadata")
      if kernel["DirectToVgprSparseMetadata"]:
        # calculate tile assignment and store into each vgprGlobalReadOffsetMetadata
        module.add(self.graMetadataTileAssignment(kernel, tPMRef))
      else:
        module.add(self.graTileAssignment(kernel, tPM))
    module.addComment1("global read addresses: tile offset assignment b")
    module.add(self.graTileAssignment(kernel, tensorParametersB))

    # unroll assignments
    module.addComment1("global read addresses: unroll assignment a")
    module.add(self.graUnrollAssignment(kernel, tensorParametersA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.addComment1("global read addresses: unroll assignment metadata")
      module.add(self.graUnrollAssignment(kernel, tPM))
    module.addComment1("global read addresses: unroll assignment b")
    module.add(self.graUnrollAssignment(kernel, tensorParametersB))

    # other free indices
    if kernel["ProblemType"]["NumIndicesC"] > 2:
      module.addComment1("global read addresses: other free assignments")
      module.add(self.graOtherFreeAssignments())

    # other summation indices
    if self.states.otherSummations:
      module.addComment1("global read addresses: other summation assignments")
      module.add(self.graOtherSummationAssignments(kernel))

    # tile offsets
    module.addComment1("global read addresses: tile offsets a")
    module.add(self.graTileOffsets(kernel, tensorParametersA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.addComment1("global read addresses: tile offsets metadata")
      # Using A or B's margin to instead Metadata's margin
      module.add(self.graTileOffsets(kernel, tPM, tPMRef["glvw"] if tPMRef["rtv"] else 1))
    module.addComment1("global read addresses: tile offsets b")
    module.add(self.graTileOffsets(kernel, tensorParametersB))

    # unroll offsets
    module.addComment1("global read addresses: unroll offsets a")
    module.add(self.graUnrollOffsets(kernel, tensorParametersA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.addComment1("global read addresses: unroll offsets metadata")
      module.add(self.graUnrollOffsets(kernel, tPM))
    module.addComment1("global read addresses: unroll offsets b")
    module.add(self.graUnrollOffsets(kernel, tensorParametersB))

    # tile edges
    if kernel["EdgeType"] == "ShiftPtr":
      if self.states.useBias == DataDirection.WRITE and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
        # Not supported
        assert not forceNoTileCode
      # Shift here has two purposes:
      #  1. Ensure the loads are in-bounds to prevent fault.
      #     BufferLoad uses the buffer limit hardware and does not require bounds checking for this case
      #  2. Shift-left a wide vector load to ensure it is completely in-bounds.
      #     If this occurs we need to 'unshift' the C values (see shiftVectorComponents)
      #     BufferLoad does support this shifting, but if GuaranteeNoPartial=1 then
      #     it can be guaranteed that no shifting is required.
      if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"]) and not forceNoTileCode:
        module.addComment1("global read addresses: shift a")
        module.add(self.graShift(kernel, tensorParametersA))
        if tensorParametersA["is_sparse"] and kernel["DirectToVgprSparseMetadata"]:
          module.addComment1("global read addresses: shift metadata")
          module.add(self.graMetadataShift(kernel, tensorParametersA))

      if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialMetadata"]) and not forceNoTileCode \
        and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.addComment1("global read addresses: shift metadata")
        # Using A's margin to instead Metadata's margin
        module.add(self.graShift(kernel, tPM, tPMRef["glvw"] if tPMRef["rtv"] else 1))

      if not (kernel["BufferLoad"] and  kernel["GuaranteeNoPartialB"]) and not forceNoTileCode:
        module.addComment1("global read addresses: shift b")
        module.add(self.graShift(kernel, tensorParametersB))
        if tensorParametersB["is_sparse"] and kernel["DirectToVgprSparseMetadata"]:
          module.addComment1("global read addresses: shift metadata")
          module.add(self.graMetadataShift(kernel, tensorParametersB))

    # addresses
    if not forceNoTileCode:
      module.addComment1("global read addresses: addresses a")
      module.add(self.graAddresses(kernel, tensorParametersA))
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.addComment1("global read addresses: addresses metadata")
        module.add(self.graAddresses(kernel, tPM))
      module.addComment1("global read addresses: addresses b")
      module.add(self.graAddresses(kernel, tensorParametersB))

    # workgoup SGPRs no longer needed

    module.add(self.removeGROffsetsVariableSgprsFromPool(kernel))

    # final offsets
    module.addComment1("global read addresses: final offsets a")
    module.add(self.graFinalOffsets(kernel, tensorParametersA))
    if kernel["ProblemType"]["Sparse"]:
      module.addComment1("global read addresses: final offsets metadata")
      if kernel["DirectToVgprSparseMetadata"]:
        module.add(self.graMetadataFinalOffsets(kernel, tPMRef))
      else:
        module.add(self.graFinalOffsets(kernel, tPM))
    module.addComment1("global read addresses: final offsets b")
    module.add(self.graFinalOffsets(kernel, tensorParametersB))
    self.dontAppendCode = False
    self.dontAppendCode = self.dontAppendCode or forceNoTileCode

    # Add increment code
    gsuComponent = Component.GSU.find(self)
    module.add(gsuComponent.setupNewTile(self, kernel, tensorParametersA, tensorParametersB, tPM))

    self.dontAppendCode = self.dontAppendCode or forceNoTileCode

    ###########################################################################
    # summations loops: open
    ###########################################################################

    # declare loop num iter
    if not forceNoTileCode:
      module.addComment0("declare loop num iterations")

    # perform initC in the shadow of the prefetch
    # Prefetch occurs at start of unroll loop
    # If we have multiple summation indices (otherSummationLoops>0),
    # we can't init in shadow of this prefetch
    # since that would initC inside the other summation loops

    if self.states.doShadowInit != 2:
      module.add(self.initC(kernel))
      if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
        module.add(self.initSumUnroll(kernel))

    # open non-unrolled summation loops
    if not forceNoTileCode:
      for i in range(kernel["ProblemType"]["NumIndicesSummation"]-1):
        module.addComment1("summation loop %u"%i)
        module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, i))
        if self.states.actualSummationLoops>1:
          module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, i))
      module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx))

    if not forceNoTileCode:
      module.add(self.declareStaggerParms(kernel))
      module.add(self.calculateStagger(kernel, tensorParametersA))
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.add(self.calculateStagger(kernel,tPM))
      module.add(self.calculateStagger(kernel, tensorParametersB))

    # LRO and LWA as assigned
    # init lds read pointers before each unrolled loop
    module.addComment0("local read addresses: init pointers a")
    module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.addComment0("local read addresses: init pointers metadata")
      module.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))
    module.addComment0("local read addresses: init pointers b")
    module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))

    if self.do["executeToInitEnd"]:
      module.add(self.functionEnd(kernel, addLabel=False))

    ####################################
    # prefetch: unrolled loop prefix
    ####################################
    if kernel["PrefetchGlobalRead"]:
      # if DirectToVgpr is enabled and swapGlobalRead is true, swap the order of global read (B->A)
      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      if self.isSwapGlobalReadOrderForDtvOrDtl(kernel, prefetch1=True):
        tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
      pfi = 1
      module.addComment1("prefetch: global -> local")
      module.add(self.openSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=isOptNLL))
      moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters1st)
      module.add(replaceHolder(moduleTmp, 0))
      module.add(self.globalReadDo(kernel, 0, tensorParameters1st))
      moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters2nd, True)
      module.add(replaceHolder(moduleTmp, 0))
      module.add(self.globalReadDo(kernel, 0, tensorParameters2nd))
      tPA = tensorParametersA
      tPB = tensorParametersB
      if kernel["PrefetchGlobalRead"] == 2:
        # PGR2 + DTV case, skip GR inc
        if kernel["DirectToVgprA"]:
          tPA = None
        if kernel["DirectToVgprB"]:
          tPB = None
      module.add(self.globalReadIncrementAB(kernel, tPA, tPB, self.states.unrollIdx, pfi))

    module.addComment2("End setupNewTile")

    return module

  ##############################################################################
  # No Load Loop Body
  ##############################################################################
  def noLoadLoopBody( self, kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast, NLLindex=0, NLLnum=1):
    if kernel["UseCustomMainLoopSchedule"]:
      module = Module()
      if isNGLL:
        module.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=-1, comment="Wait for all PGR to complete"))
        module.add(SBarrier(comment=""))
        module.addComment0("Code-path 0, useGR=0, usePLR=1, useGRInc=1, useLoop = 0")
        module.add(TextBlock("MAINLOOP 0 0 1 1 0\n"))
      else:
        module.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=-1, comment="Wait for all PGR to complete"))
        module.add(SBarrier(comment=""))
        module.addComment0("Code-path 0, useGR=0, usePLR=0, useGRInc=0, useLoop = 0")
        module.add(TextBlock("MAINLOOP 0 0 0 0 0\n"))
      return module
    module = Module("noLoadLoopBody")
    expand = kernel["ExpandPointerSwap"]
    lastuIdx = False
    pflr     = self.states.numItersPLR
    localWriteEndIter = kernel["LoopIters"] - self.states.numItersPLR - 1
    dsWriteBA = False
    if isNGLL and kernel["UnrollLoopSwapGlobalReadOrder"] == 1 and NLLindex == 0 and NLLnum == 2:
      dsWriteBA = True

    # vregSetIdx for DTV GlobalRead
    # isSecondBuf case, use second set
    vregSetIdxGR = 0
    vregSetIdxMFMA = 0
    isDTVAB = kernel["DirectToVgprA"] or kernel["DirectToVgprB"]
    if isDTVAB:
      if isNGLL:
        if NLLindex == 0 and NLLnum == 2:
          vregSetIdxGR = 1
      else:
        if NLLindex == 1 and NLLnum == 2:
          vregSetIdxGR = 1
      # for MFMA, use opposite side
      vregSetIdxMFMA = 1 - vregSetIdxGR
    tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]

    for uIdx in range(0, kernel["LoopIters"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      isLastLoop = not isNGLL
      if u == 0:
        if not isLastLoop:
          # For UnrollLoopSwapGlobalReadOrder == 1 or DirectToVgprA/B, we will have 2 NGLLs,
          # One with local write A then B and another with B then A.
          if dsWriteBA == True:
            # In the current scheduling, we always schedule lwa first then lwb second.
            # Put B in lwa code can easily change the order.
            self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersB, swapAB=1)
            self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersA, swapAB=1)
          else:
            self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersA)  # local write in loopcnt N targets data for loopcnt N+1
            self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersB)
          # unrolled loop: global read A, B
          # DirectToVgprA + PGR2: we need DTVA GR in NGLL
          if kernel["DirectToVgprA"]:
            self.codes.globalReadA = self.globalReadDo(kernel, 1, tensorParametersA, unrollLoopIdx=0, g2lBufIdx=vregSetIdxGR)
          else:
            self.codes.globalReadA = StructuredModule() # empty
          # DirectToVgprB + PGR2: we need DTVB GR in NGLL
          if kernel["DirectToVgprB"]:
            self.codes.globalReadB = self.globalReadDo(kernel, 1, tensorParametersB, unrollLoopIdx=0, g2lBufIdx=vregSetIdxGR)
          else:
            self.codes.globalReadB = StructuredModule() # empty

        else:
          self.codes.localWriteA = Module()
          self.codes.localWriteB = Module()

        if not isNGLL or kernel["ExpandPointerSwap"] or kernel["UnrollLoopSwapGlobalReadOrder"] or isDTVAB:
          # PAP would have GlobalRead and GlobalInc, but no localWrite
          # Get the perIterGlobalReadCode code for PAP (if PAP=On), else would be empty
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, skipGlobalReadInc=False, lastLoop=NLLlast, isNGLL=isNGLL)
          module.add(self.codes.unrollLoopHeader)

      if kernel["ForceUnrollSubIter"]:
        self.states.lwStartMfmaIndex = (kernel["MIWaveTile"][1] + kernel["MIWaveTile"][0])//2 -1

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = 1 if kernel["ForceUnrollSubIter"] else (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
        isSwapAndResetLwoIter = (u == self.states.lwEndMfmaIndex//(self.states.numMfmaPerIter))

      extraComment = ""
      if isLastLoop:
        extraComment += " (last unrolled loop)"
      else:
        if isResetLroIter:
            extraComment += " (reset local read pointers iteration) "
        if isSwapAndResetLwoIter:
            extraComment += " (swap and reset local write pointers iteration) "
        if isSwapLroIter:
            extraComment += " (swap local read pointers iteration) "
      if kernel["ForceUnrollSubIter"]:
        module.addComment1("subiter %u"%(u))
      else:
        module.addComment1("iter %u%s"%(u,extraComment))
      plrIdx = (u+pflr) % self.states.numVgprBuffer
      plrIdxDTV = (u+pflr) % kernel["LoopIters"]
      localReads = Module()

      pointerLWCode = Module()
      pointerLRCode = Module()
      waitCode = Module()  # may be overwritten (not added to) below
      macIterCode = Module()
      waitLWCode = Module()
      syncCode = Module()

      vregSetIdxLR = 0
      #needNextBufLR = True
      if isDTVAB:
        # vregSetIdx for DTV + packing
        vregSetIdxLR = vregSetIdxMFMA
        if kernel["LoopIters"] > 1 and u+pflr >= kernel["LoopIters"] and not (isNGLL and NLLindex == 1):
          # use next vregSet for the last loopIter
          # exception for isNGLL and NLLindex == 1 case
          # in this case, we need to flip twice (for NGLL to NLL, even to odd)
          # then, we do not need to flip vregSetIdxLR
          vregSetIdxLR = 1 - vregSetIdxLR

      hasLiveLdsData = kernel["PrefetchGlobalRead"]
      hasLiveLdsData = hasLiveLdsData and not isLastLoop
      # for DirectToVgpr + pack
      # DTV pack case, need to call localReadDo to generate local read code (pack) for next NLL
      #  OptNLL: NLLindex= 0, 1
      #  OrdNLL: NLLindex= 0
      needExtraDTVLocalReadDo = (NLLlast and (NLLindex == 0 or isOptNLL) and u > localWriteEndIter)
      needNextBufLR = not needExtraDTVLocalReadDo
      hasLiveLdsData = hasLiveLdsData or needExtraDTVLocalReadDo
      # reads for current loop are done in previous iteration because of wider local read
      doReadA = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadA - self.states.numItersPLR)
      doReadB = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadB - self.states.numItersPLR)
      doReadM = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMetadata - self.states.numItersPLR)
      if kernel["ForceUnrollSubIter"]:
        doReadA = 1 if u == 0 else 0
        doReadB = 1 if u == 0 else 0
        doReadM = 1 if u == 0 else 0
      # reads for next loop
      doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
      doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
      doReadM = doReadM or (hasLiveLdsData and u > localWriteEndIter)
      doReadM = doReadM and (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"])
      for iui in range(0,kernel["InnerUnroll"]):
        doReadA = doReadA and iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]
        doReadM = doReadM and iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]
        if (doReadA or doReadB or doReadM) and kernel["ForceUnrollSubIter"]:
          pack[1] = Module()
        if doReadA:
          localReads.addComment1("local read a")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadA
          if self.states.packDTVA or self.states.convDTVA:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadA + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeA, packCodeA = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedA, 0, tensorParametersA)
          if needNextBufLR:
            localReads.add(localReadCodeA)
          pack[plrIdx*self.states.numIterPerCoalescedReadA].add(packCodeA)
          if kernel["ForceUnrollSubIter"]:
            pack[1].add(packCodeA)
        if doReadM:
          localReads.addComment1("local read metadata")
          localReadCodeM, packCodeM = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
          if needNextBufLR:
            localReads.add(localReadCodeM)
          pack[plrIdx*self.states.numIterPerCoalescedReadMetadata].add(packCodeM)
          if kernel["ForceUnrollSubIter"]:
            pack[1].add(packCodeM)
        if doReadB:
          localReads.addComment1("local read b")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadB
          if self.states.packDTVB or self.states.convDTVB:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadB + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeB, packCodeB = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedB, 0, tensorParametersB)
          if kernel["ForceUnrollSubIter"]:
            pack[1].add(packCodeB)
          if needNextBufLR:
            localReads.add(localReadCodeB)
          pack[plrIdx*self.states.numIterPerCoalescedReadB].add(packCodeB)
        if (not isResetLroIter or iui != kernel["InnerUnroll"]-1):
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadM:
            localReads.addComment1("local read increment metadata")
            localReads.add(self.localReadInc(kernel, iui, tPM))
          if doReadB:
            localReads.addComment1("local read increment b")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersB))

      if not isLastLoop:
        if kernel["PrefetchGlobalRead"]:
          # put barrier at localWriteEndIter+1
          if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
            if not kernel["NoLdsWriteCode"]:
              waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
            if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
              # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
              waitLWCode.add(self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, isNLL=(not isNGLL), beforeBarrier=True))
            elif kernel["PrefetchGlobalRead"]==2 and (kernel["DirectToLdsA"] and kernel["DirectToLdsB"]):
              waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for global reads with lds"))
            syncCode.add(self._syncThreads(kernel))

          if isSwapAndResetLwoIter: # ResetLroIter
            # local write for next iter, used to have local writes here
            pointerLWCode.addComment1("local write swap offsets a")
            pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
            pointerLWCode.addComment1("local write swap offsets b")
            pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

          if isSwapLroIter: # ResetLroIter
            # Swap, reset, or increment the LRO:
            # force internalPointerSwap = False in NGLL case
            internalPointerSwap = expand and not isNGLL
            if not kernel["ForceUnrollSubIter"] or (doReadA and (u<localWriteEndIter)):
              pointerLRCode.addComment1("local read swap offsets a")
              pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA))
            if not kernel["ForceUnrollSubIter"] or (doReadB and (u<localWriteEndIter)):
              pointerLRCode.addComment1("local read swap offsets b")
              pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersB))
            if (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]) and\
               (not kernel["ForceUnrollSubIter"] or (doReadM and (u<localWriteEndIter))):
              pointerLRCode.addComment1("local read swap offsets metadata")
              pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tPM))

        if isResetLroIter: # ResetLroIter
          pointerLRCode.addComment1("local read init pointers a")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
          pointerLRCode.addComment1("local read init pointers b")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            pointerLRCode.addComment1("local read init pointers metadata")
            pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))

      # we initiate dscnt to 0, then assigning it correct value in makeSubIterSchedule()
      waitCode = self._wait(kernel, tensorParametersA, tensorParametersB, \
          -1, 0, 0, \
          "wait for prior local read local write")
      # DirectToVgpr case, wait for global read as well as local read/write
      if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
        # not generate wait here
        #  1) local write code in previous u (u-1) has local write (it comes with waitcnt vmcnt)
        countLW = 0
        if (u > 0):
          countLW += countLocalWrite(self.codes.perIterLocalWrite[u-1][1])
        if countLW == 0:
          module.add(self.getWaitcntCodeForDirectToVgpr(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, u, isNLL=(not isNGLL), NLLlast=NLLlast))

      luIdx = u % self.states.numVgprBuffer # local to use for MACs
      if kernel["ForceUnrollSubIter"]:
        luIdx = u # Pack[0] is for A0,B0 Pack[1] is for A1,B1. Pack[2] and Pack[3] are empty
      if kernel["EnableMatrixInstruction"]:
        macIterCode.add(self.mfmaIter(kernel, tensorParametersA, tensorParametersB, u, kernel["InnerUnroll"], vregSetIdxMFMA, unrollIdx = u))
      else:
        macIterCode.add(self.macIter(kernel, tensorParametersA, tensorParametersB, luIdx, kernel["InnerUnroll"], True))
      if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
        tP = tensorParametersA if kernel["ProblemType"]["BiasSrc"] == "A" else tensorParametersB
        macIterCode.add(self.exclasses.biasSumUnroll.loopSum(self, kernel, tP, u, kernel["InnerUnroll"]))

      subIterCode = self._makeSubIterSchedule(kernel, tensorParametersA, tensorParametersB, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx], module, NLLlast)
      module.add(subIterCode)
      self.states.SubTileIdx = (self.states.SubTileIdx + 1) % kernel["numSubTiles"]
      pack[luIdx] = Module()
    return module

  ##############################################################################
  # noLoadLoop
  # Create the no load loop (NLL)
  #
  # isOptNLL : the NLL is to be optimized for the alpha=1 and non-edge case
  ##############################################################################
  def noLoadLoop( self, kernel, tensorParametersA, tensorParametersB, isOptNLL, isNGLL, pack, NLLindex=0, NLLnum=1):
    module = Module("noLoadLoop")
    LoopNameComment = "NoGlobalLoadLoop" if isNGLL else "NoLoadLoop"
    isOptNLLComment = "Opt" if isOptNLL else "Ord"
    startComment = "%s. %s - Begin " % (isOptNLLComment, LoopNameComment)
    if NLLnum > 1:
      startComment = startComment + "%u/%u"%(NLLindex+1, NLLnum)
    module.addComment2(startComment)
    NLLfirst = True
    NLLlast = True
    PGR2 = kernel["PrefetchGlobalRead"] == 2
    if PGR2:
      # PGR=2 case NoLoadLoop(NLL) is generated twice
      # we need to distinguish them to generate proper code at each NLL
      if isNGLL:
        NLLlast = False
      else:
        # PGR=2 and not isNGLL means second NoLoadLoop for PGR2.
        # Need to avoid generating duplicated code which is already generated in NGLL(first NoLoadLoop for PGR=2)
        NLLfirst = False
    if isNGLL:
      self.codes.perIterLocalWrite = self.codes.perIterLocalWriteCodeNGLL
      self.states.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
      if kernel["ExpandPointerSwap"] == 1:
        self.codes.globalReadA = StructuredModule() # empty
        self.codes.globalReadB = StructuredModule() # empty
    #else:
    if not isNGLL:
      self.codes.dtlsM0UpdateA = StructuredModule()
      self.codes.globalReadA = StructuredModule() # empty
      self.codes.dtlsM0UpdateB = StructuredModule()
      self.codes.globalReadB = StructuredModule() # empty
      # PGR2 + DTV case, we still need global read inc code in NLL
      tPAlocal = None
      tPBlocal = None
      if PGR2 and kernel["DirectToVgprA"]:
        tPAlocal = tensorParametersA # generate inc A code
      if PGR2 and kernel["DirectToVgprB"]:
        tPBlocal = tensorParametersB # generate inc B code
      self.codes.globalReadIncrements = self.globalReadIncrementAB(kernel, tPAlocal, tPBlocal, self.states.unrollIdx, 0)
      self.codes.localWriteA = Module()
      self.codes.localWriteB = Module()

    openSum = self.openSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL, isNGLL=isNGLL, NLLindex=NLLindex, NLLnum=NLLnum)
    module.add(openSum)

    if not self.states.numItersPLR and not kernel["UseCustomMainLoopSchedule"]:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        vlcntVal = 1 if kernel["PrefetchGlobalRead"] == 2 and isNGLL else 0
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, vlcntVal, -1, -1, "10wait for global read"))
      if not kernel["NoLdsWriteCode"]:
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "4wait for local write"))
      module.add(self._syncThreads(kernel))

    # generate no Load Loop Body code
    module.add(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, pack, isOptNLL, isNGLL, NLLfirst, NLLlast, NLLindex=NLLindex, NLLnum=NLLnum))

    if self.do["executeToLoopEnd"] and isOptNLL:
      module.add(self.functionEnd(kernel, addLabel=False))

    # Close code is necessary for both first and last (NGLL case(=NLLfirst) needs label)
    module.add(self.closeSumAtLeastUnroll(kernel, tensorParametersA, tensorParametersB, prefetch=False, isOptNLL=isOptNLL, isNGLL=isNGLL, isNotLast=(NLLindex<(NLLnum-1))))

    if self.states.FactorDim == 3:
      self.updateBranchPlaceHolder(module, ["skipOptNLL_placeholder", "skipOptNLL_scc1_placeholder"] , ["OptNLL_End", "OptNLL_End"], ["SCBranchSCC0", "SCBranchSCC1"])
    return module

  ##############################################################################
  # Loop Body
  # When UnrollLoopSwapGlobalReadOrder is enabled,
  # dsWriteBA is to do ds_write B first.
  # grBA is to do buffer_load B first.
  ##############################################################################
  def _loopBody( self, kernel, tensorParametersA, tensorParametersB, pack, lc, loopCopies, finalLoop, firstIter=False, dsWriteBA=False, grBA=False, isDTVGRSecondBuf=False, skipClose=False ):
    module = Module("loopBody")
    expand = kernel["ExpandPointerSwap"]

    # not generate openLoop for firstIter
    if not firstIter:
      module.addComment2("Unrolled Loop %u/%u - Begin" % (lc+1, loopCopies))
    if kernel["PrefetchGlobalRead"] and not self.states.numItersPLR and not kernel["ScheduleIterAlg"] == 2 and not kernel["UseCustomMainLoopSchedule"]:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        vlcntVal = 1 if kernel["PrefetchGlobalRead"] == 2 else 0
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, vlcntVal, -1, -1, "11wait for global read"))
      if not kernel["NoLdsWriteCode"]:
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "1wait for local write"))
      module.add(self._syncThreads(kernel, "4sync for global read"))

    module.addComment1("Begin Each Unroll: Check VGPR.checkin for INT8 LW")

    # swap the order of global read (B->A)
    # - swapAB (grBA=True)
    # - isSwapGlobalReadOrderForDtvOrDtl is true
    tensorParameters1st = tensorParametersA
    tensorParameters2nd = tensorParametersB
    tc1 = 'A'
    tc2 = 'B'
    if grBA==True or self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
      tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
      tc1, tc2 = tc2, tc1

    # unrolled loop: global read A, B
    # M0 update for directToLds, skip if using custom main loop schedule
    self.codes.dtlsM0UpdateA = self.directToLdsM0Update(kernel, 1, tensorParameters1st, kernel["UseCustomMainLoopSchedule"])
    self.codes.dtlsM0UpdateB = self.directToLdsM0Update(kernel, 1, tensorParameters2nd, True)

    g2lBufIdx1st = 0
    if grBA==True or (kernel["DirectToVgpr%s"%tc1] and isDTVGRSecondBuf):
      # use second buffer
      g2lBufIdx1st = 1
    self.codes.globalReadA = self.globalReadDo(kernel, 1, tensorParameters1st, unrollLoopIdx=lc, g2lBufIdx=g2lBufIdx1st)
    g2lBufIdx2nd = 0
    if grBA==True or (kernel["DirectToVgpr%s"%tc2] and isDTVGRSecondBuf):
      # use second buffer
      g2lBufIdx2nd = 1
    self.codes.globalReadB = self.globalReadDo(kernel, 1, tensorParameters2nd, unrollLoopIdx=lc, g2lBufIdx=g2lBufIdx2nd)

    # unrolled loop: increment global read addresses
    self.codes.globalReadIncrements = self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, 0)
    globalReadIncACode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementB")

    if not kernel["NoLdsWriteCode"]:
      self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersA)
      self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersB)
    else:
      self.codes.localWriteA = Module()
      self.codes.localWriteB = Module()

    # localWriteEndIter is used to determine which iteration to put sync
    # if PGR=0, GR,LW,sync,LR will put at front of loop.
    localWriteEndIter = kernel["LoopIters"] - self.states.numItersPLR - 1

    # Schedule the global read, global read inc, and writes:
    unrollLoopHeaderCodeScheduled = False
    if not kernel["PrefetchGlobalRead"]:
      unrollLoopHeaderCodeScheduled = True
      self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter=firstIter)
      module.add(self.codes.unrollLoopHeader)

    # if not prefetch global, localWrite before mac's
    if not kernel["PrefetchGlobalRead"]:
      # unrolled loop: local write A, B
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "5wait for global read"))
      module.add(self._syncThreads(kernel, "PGR=0, prior iter done reading lds"))
      if not kernel["NoLdsWriteCode"]:
        module.addComment1("local write a")
        tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA)
        module.add(tempLWCodeModA)
        module.addComment1("local write b")
        tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB)
        module.add(tempLWCodeModB)
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
      module.add(self._syncThreads(kernel))
      # debug Local state
      """
      module.add("    /* print Local state */" + self.endLine)
      module.add("    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine)
      module.add("      printf(\\\"localMemory[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" )
          % self.endLine
      module.add("    }" + self.endLine)
      """

    tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]

    # unrolled loop: prefetch local
    if self.states.numItersPLR and not kernel["PrefetchGlobalRead"]:
      for plrIdx in range(0, self.states.numItersPLR):
        pack[plrIdx] = Module()
        for iui in range(0,kernel["InnerUnroll"]):
          if iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]:
            module.addComment1("prefetch local a")
            localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadA, iui*self.states.numReadsIterCoalescedA, 0, tensorParametersA)
            module.add(localReadCodeA)
            pack[plrIdx].add(packCodeA)
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            if iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]: # no local read code if DirectToVgpr is enabled
              module.addComment1("prefetch local metadata")
              localReadCodeM, packCodeM = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
              module.add(localReadCodeM)
              pack[plrIdx].add(packCodeM)
          if iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]:
            module.addComment1("prefetch local b")
            localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadB, iui*self.states.numReadsIterCoalescedB, 0, tensorParametersB)
            module.add(localReadCodeB)
            pack[plrIdx].add(packCodeB)

          if iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]:
            module.addComment0("local read increment a")
            module.add(self.localReadInc(kernel, iui, tensorParametersA))
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            numReadsIterCoalesced = self.states.numReadsIterCoalescedB if kernel["ProblemType"]["Sparse"] == 2 else self.states.numReadsIterCoalescedA
            if iui*numReadsIterCoalesced < kernel["InnerUnroll"]:
              module.addComment0("local read increment metadata")
              module.add(self.localReadInc(kernel, iui, tPM))
          if iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]:
            module.addComment0("local read increment b")
            module.add(self.localReadInc(kernel, iui, tensorParametersB))

    pflr     = self.states.numItersPLR  # how many pf already done above

    # Store instruction streams across all iterations
    MfmaCodeAllIters = Module()
    LRSwapAAllIters = Module()
    LRSwapBAllIters = Module()
    LWSwapAAllIters = Module()
    LWSwapBAllIters = Module()
    LRCodeAAllIters = []
    LRCodeBAllIters = []
    PackCodeAAllIters = []
    PackCodeBAllIters = []

    ############################################################################
    # unrolled loop: mac iterations
    ############################################################################

    # double/quadruple the number of compute loop for each DepthU's worth of data read
    for uIdx in range(0, kernel["LoopIters"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      if kernel["UseCustomMainLoopSchedule"]:
        LRCodeAAllIters.append(Module())
        LRCodeBAllIters.append(Module())
        PackCodeAAllIters.append(Module())
        PackCodeBAllIters.append(Module())
      if u==0: # if at start of subloop...
        # ...update local write code
        if not kernel["NoLdsWriteCode"]:
          if dsWriteBA:
            self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersB, swapAB=1)
            self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersA, swapAB=1)
          else:
            self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersA)
            self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersB)
        else:
          self.codes.localWriteA = Module()
          self.codes.localWriteB = Module()

        if not unrollLoopHeaderCodeScheduled:
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter=firstIter, lastLoop=False, lastLc=(lc==loopCopies-1))
          module.add(self.codes.unrollLoopHeader)

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      if kernel["ForceUnrollSubIter"]:
        self.states.lwStartMfmaIndex = (kernel["MIWaveTile"][1] + kernel["MIWaveTile"][0])//2 - 1

      isResetLroIter = 1 if kernel["ForceUnrollSubIter"] else (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
        isSwapAndResetLwoIter = (u == self.states.lwEndMfmaIndex//(self.states.numMfmaPerIter))
      extraComment = ""
      if isResetLroIter:
        extraComment += " (reset local read pointers iteration) "
      if isSwapAndResetLwoIter:
        extraComment += " (swap and reset local write pointers iteration) "
      if isSwapLroIter:
        extraComment += " (swap local read pointers iteration) "

      if kernel["ForceUnrollSubIter"]:
        module.addComment1("subiter %u"%(u))
      else:
        module.addComment1("iter %u%s"%(u,extraComment))

      plrIdx = (u+pflr) % self.states.numVgprBuffer
      plrIdxDTV = (u+pflr) % kernel["LoopIters"]

      # vregSetIdx for DTV
      # use oppsite side of GR buffer
      vregSetIdxMFMA = 1 if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and (not isDTVGRSecondBuf) else 0
      vregSetIdxLR = vregSetIdxMFMA
      if kernel["LoopIters"] > 1 and u+pflr >= kernel["LoopIters"]:
        # use next vregSet for the last loopIter (exception: LoopIters==1)
        # LoopIters==1 case, local read is for the current iteration and not for the next iteration
        vregSetIdxLR = (vregSetIdxLR + 1) % loopCopies
        # final loop case, use vregSetIdx for noLoadLoop (NGLL(PGR2) or NLL(PGR1))
        if finalLoop:
          # finalLoop case, this is for NoLoadLoop (NGLL(PGR2) or NLL(PGR1))
          # PGR2 case, next is NGLL. Use first set
          # PGR1 case, next is NLL. Use second set
          vregSetIdxLR = 0 if kernel["PrefetchGlobalRead"] == 2 else 1

      localReads = Module()
      localReadsA = Module()
      localReadsB = Module()
      localReadsM = Module()

      pointerLWCode = Module()
      pointerLRCode = Module()
      waitCode = Module()  # may be overwritten (not added to) below
      macIterCode = Module()
      waitLWCode = Module()
      syncCode = Module()

      hasLiveLdsData = kernel["PrefetchGlobalRead"]
      # reads for current loop are done in previous iteration because of wider local read
      doReadA = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadA - self.states.numItersPLR)
      doReadB = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadB - self.states.numItersPLR)
      doReadM = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMetadata - self.states.numItersPLR)

      if kernel["ForceUnrollSubIter"]:
        doReadA = 1 if u == 0 else 0
        doReadB = 1 if u == 0 else 0
        doReadM = 1 if u == 0 else 0

      # reads for next loop
      doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
      doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
      doReadM = doReadM or (hasLiveLdsData and u > localWriteEndIter)
      doReadM = doReadM and (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"])
      for iui in range(0,kernel["InnerUnroll"]):
        doReadA = doReadA and iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]
        doReadM = doReadM and iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]
        if (doReadA or doReadB or doReadM) and kernel["ForceUnrollSubIter"]:
          pack[1] = Module()
        if doReadA:
          localReads.addComment1("local read a")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadA
          if self.states.packDTVA or self.states.convDTVA:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadA + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeA, packCodeA = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedA, 0, tensorParametersA)
          localReads.add(localReadCodeA)
          localReadsA.add(localReadCodeA)
          pack[plrIdx*self.states.numIterPerCoalescedReadA].add(packCodeA)
          if kernel["UseCustomMainLoopSchedule"]:
            LRCodeAAllIters[uIdx].add(localReadCodeA)
            PackCodeAAllIters[uIdx].add(packCodeA)
          if kernel["ForceUnrollSubIter"]:
            pack[1].add(packCodeA)
        if doReadM:
          localReads.addComment1("local read metadata")
          localReadCodeM, packCodeM = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
          localReads.add(localReadCodeM)
          localReadsM.add(localReadCodeM)
          pack[plrIdx*self.states.numIterPerCoalescedReadMetadata].add(packCodeM)
          if kernel["ForceUnrollSubIter"]:
            pack[1].add(packCodeM)
        if doReadB:
          localReads.addComment1("local read b")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadB
          if self.states.packDTVB or self.states.convDTVB:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadB + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeB, packCodeB = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedB, 0, tensorParametersB)
          localReads.add(localReadCodeB)
          localReadsB.add(localReadCodeB)

          pack[plrIdx*self.states.numIterPerCoalescedReadB].add(packCodeB)
          if kernel["UseCustomMainLoopSchedule"]:
            LRCodeBAllIters[uIdx].add(localReadCodeB)
            PackCodeBAllIters[uIdx].add(packCodeB)
          if kernel["ForceUnrollSubIter"]:
            pack[1].add(packCodeB)

        # Don't increment the LRO if we are going to reset them below:
        if not isResetLroIter or iui != kernel["InnerUnroll"]-1:
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadM:
            localReads.addComment1("local read increment metadata")
            localReads.add(self.localReadInc(kernel, iui, tPM))
          if doReadB:
            localReads.addComment1("local read increment b")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersB))

      if kernel["PrefetchGlobalRead"]:
        # wait code for DirectToVgpr
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          module.add(self.getWaitcntCodeForDirectToVgpr(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, u))
        # put barrier at localWriteEndIter+1
        if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
          if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
            vlcntVal = 1 if kernel["PrefetchGlobalRead"] == 2 else 0
            waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, vlcntVal, -1, -1, \
                                      "wait for previous set of global reads"))
          # (no local write code. Global read wait for DirectToLds is already done)
          if not kernel["NoLdsWriteCode"]:
            waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
          skipForceWaitcnt0 = False
          if kernel["DirectToVgprA"] or kernel["DirectToVgprB"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
            # DTVA/B or DTLA/B case, skip generating force waitcnt0
            skipForceWaitcnt0 = True
          syncCode.add(self._syncThreads(kernel, skipForceWaitcnt0=skipForceWaitcnt0))

        if isSwapAndResetLwoIter: # ResetLroIter
          if kernel["ExpertSchedulingMode"] > 0:
            pointerLWCode.add(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))

          # local write for next iter, used to have local writes here
          pointerLWCode.addComment1("local write swap offsets a")
          pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
          pointerLWCode.addComment1("local write swap offsets b")
          pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
          if kernel["UseCustomMainLoopSchedule"]:
            LWSwapAAllIters.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
            LWSwapBAllIters.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

        if isSwapLroIter: # ResetLroIter
          if kernel["ExpertSchedulingMode"] > 0:
            pointerLRCode.add(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))
          # Swap, reset, or increment the LRO:
          if not kernel["ForceUnrollSubIter"] or (doReadA and (u<localWriteEndIter)):
            pointerLRCode.addComment1("local read swap offsets a")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
            if kernel["UseCustomMainLoopSchedule"]:
              LRSwapAAllIters.add(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"] and\
             (not kernel["ForceUnrollSubIter"] or (doReadM and (u<localWriteEndIter))):
            pointerLRCode.addComment1("local read swap offsets metadata")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tPM))
          if not kernel["ForceUnrollSubIter"] or (doReadB and (u<localWriteEndIter)):
            pointerLRCode.addComment1("local read swap offsets b")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tensorParametersB))
            if kernel["UseCustomMainLoopSchedule"]:
              LRSwapBAllIters.add(self.localReadSwapOffsets(kernel, expand, tensorParametersB))


      if isResetLroIter: # ResetLroIter
        pointerLRCode.addComment1("local read init pointers a")
        pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          pointerLRCode.addComment1("local read init pointers metadata")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))
        pointerLRCode.addComment1("local read init pointers b")
        pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))

      waitCode = self._wait(kernel, tensorParametersA, tensorParametersB, \
          -1, 0, 0, \
          "wait for prior local read local write")

      luIdx = u % self.states.numVgprBuffer # local to use for MACs
      if kernel["EnableMatrixInstruction"]:
        mfmaIter = self.mfmaIter(kernel, tensorParametersA, tensorParametersB, u, kernel["InnerUnroll"], vregSetIdxMFMA, unrollLoopIdx=lc, unrollIdx = u)
        if kernel["UseCustomMainLoopSchedule"]:
          MfmaCodeAllIters.add(mfmaIter)
        else:
          macIterCode.add(mfmaIter)
      else:
        macIterCode.add(self.macIter(kernel, tensorParametersA, tensorParametersB, luIdx, kernel["InnerUnroll"], True))
      if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
        tP = tensorParametersA if kernel["ProblemType"]["BiasSrc"] == "A" else tensorParametersB
        macIterCode.add(self.exclasses.biasSumUnroll.loopSum(self, kernel, tP, u, kernel["InnerUnroll"]))

      ###### unroll loop efficiency implementation######################################
      # unroll loop efficiency implementation
      ## split A&B fetch&MAC code into multiple groups
      ## splitting strategy   based on TT size
      ## 6x4 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of A{1-2)B(0) , A(3),B(1) (not implemented yet)
      ## 4x6 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of B{1-2)A(0) , B(3),A(1)
      ## 4x4 -> split into group of 8 and 8  MAC(s)
      ## 6x6 -> split into group of 12 MAC(s)
      ## 8x4/4x8 -> split into group of 16 and 16  MAC(s)
      ## 8x8 -> split into group of 16 MAC(s)
      ## supports only PLR=0
      ###############################################################################

      # Is this test necessary because of the global variable this if was previously always true
      # after removing the global variable it is always false...
      # if self.states.numItersPLR:
      if not kernel["UseCustomMainLoopSchedule"]:
        if kernel["ForceUnrollSubIter"]:
          luIdx = u
        subIterCode = self._makeSubIterSchedule(kernel, tensorParametersA, tensorParametersB, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx], module)
        module.add(subIterCode) # add scheduled "other", local reads, local writes

      self.states.SubTileIdx = (self.states.SubTileIdx + 1) % kernel["numSubTiles"]
      pack[luIdx] = Module()


    if kernel["UseCustomMainLoopSchedule"]:
      optSchedule, numCodePath = customMainLoopSchedule(self, kernel, tensorParametersA, tensorParametersB, globalReadIncACode, globalReadIncBCode, \
                                                   LRCodeAAllIters, PackCodeAAllIters, LRCodeBAllIters, PackCodeBAllIters, \
                                                   LRSwapAAllIters, LRSwapBAllIters, self.codes.globalReadA, self.codes.globalReadB, \
                                                   LWSwapAAllIters, LWSwapBAllIters, MfmaCodeAllIters, \
                                                   self.closeLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, False))
      module.add(optSchedule)
      module.add(self.simdSpecDispatch(kernel, numCodePath))

    # close unrolled loop
    endStr = ""
    if loopCopies == 2:
      finalStr = " (final)" if lc == 1 else ""
      endStr = " %u/%u%s"%(lc+1, loopCopies, finalStr)
    module.addComment2("Unrolled Loop - End%s"%(endStr))

    oddLabel = (lc == 0 and loopCopies == 2)
    if not skipClose and not kernel["UseCustomMainLoopSchedule"]:
      module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, finalLoop, oddLabel=oddLabel))
    return module

  ##############################################################################
  # Kernel Body
  ##############################################################################
  def kernelBody( self, kernel, tensorParametersA, tensorParametersB ):
    expand = kernel["ExpandPointerSwap"]
    self.dontAppendCode = False

    tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]

    ####################################
    # Begin String
    moduleKernelBody = KernelBody("kernelBody")

    ####################################
    # Function Signature
    ####################################
    fs = self.functionSignature()
    moduleKernelBody.addSignature(fs)

    module = Module("body")
    module.add(Label("ASM_Start", "Main body of the asm kernel"))
    module.add(self.defineAndResources(kernel, tensorParametersA, tensorParametersB, tPM))

    # Initialize stream-k loop
    skComponent = Component.StreamK.find(self)
    module.add(skComponent.preLoop(self, kernel))


    # Open persistent loop
    loopComponent = Component.PersistentLoop.find(self)

    module.add(loopComponent.openPersistentLoop(self, kernel))
    module.add(self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isOptNLL=False))

    if self.do["executeToPrefetchEnd"]:
      module.add(self.functionEnd(kernel, addLabel=False))
    if kernel["ForceUnrollSubIter"]:
      pack = [ Module() for i in range (4) ] # 4 buffers for ForceUnrollSubIter
    else:
      pack = [ Module() for i in range (self.states.numVgprBuffer) ]
    self.preLoopLocalWriteCode = None

    if kernel["PrefetchGlobalRead"]:
      if self.states.doShadowInit:
        module.add(self.openShadowInit())
        # SrdD/SrdC are used starting now, remove from sgpr pool
        self.removeSgprVarFromPool("SrdD")
        self.removeSgprVarFromPool("SrdC")
        module.add(self.globalWriteWorkGroupInit(kernel))
        if self.states.doShadowInit == 2:
          module.add(self.initC(kernel)) # initC while waiting for global reads
          if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
            module.add(self.initSumUnroll(kernel))
        module.add(self.closeShadowInit(kernel))

      # Wait for PGR code in setupNewTile
      module.add(self.getWaitcntCodeForPGR(kernel, tensorParametersA, tensorParametersB, "wait for global read"))
      # These cases loop back and run the prefetch loop again
      # we need an extra barrier to ensure that the ds_reads (either for SR or MFMA) from previous iteration
      # have finished before we generate the prefetch for the next summation index.
      if kernel["StreamK"] > 0 or self.states.actualSummationLoops>1:
        module.add(SBarrier(comment="For stream-k / persistent loop"))

      # local write
      self.preLoopLocalWriteCode = self.preLoopLocalWriteDo(kernel, tensorParametersA, tensorParametersB)
      module.add(self.preLoopLocalWriteCode)
      # swap local ptrs
      module.addComment1("local write swap a")
      module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      module.addComment1("local write swap b")
      module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

      if kernel["PrefetchGlobalRead"] == 2:
        module.add(self.openPrefetchGlobalRead2(kernel))
        # For UnrollLoopSwapGlobalReadOrder, we also need to swap ds write A/B order.
        # In scheduling, we always schedule lwa first then lwb second,
        # Putting lwb in lwa's code object can easily change the order.
        # swap the order of global read (B->A)
        # - swapAB (grBA=True)
        # - isSwapGlobalReadOrderForDtvOrDtl is true
        tensorParameters1st = tensorParametersA
        tensorParameters2nd = tensorParametersB
        tc1 = 'A'
        tc2 = 'B'
        if kernel["UnrollLoopSwapGlobalReadOrder"] == 1 or self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
          tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
          tc1, tc2 = tc2, tc1
        # skip second PGR if DTV is true
        skip1st = kernel["DirectToVgpr%s"%tc1]
        skip2nd = kernel["DirectToVgpr%s"%tc2]
        if not skip1st:
          g2lBufIdx1st = 0
          if kernel["UnrollLoopSwapGlobalReadOrder"] == 1 or kernel["DirectToVgpr%s"%tc1]:
            # use second buffer
            g2lBufIdx1st = 1
          module.add(self.directToLdsM0Update(kernel, 1, tensorParameters1st))
          module.add(self.globalReadDo(kernel, 0, tensorParameters1st, g2lBufIdx=g2lBufIdx1st))
        if not skip2nd:
          g2lBufIdx2nd = 0
          if kernel["UnrollLoopSwapGlobalReadOrder"] == 1 or kernel["DirectToVgpr%s"%tc2]:
            # use second buffer
            g2lBufIdx2nd = 1
          module.add(self.directToLdsM0Update(kernel, 1, tensorParameters2nd))
          module.add(self.globalReadDo(kernel, 0, tensorParameters2nd, g2lBufIdx=g2lBufIdx2nd))

        # swap local ptrs again if DirectToLds is enabled
        if kernel["DirectToLdsA"]:
          module.addComment1("local write swap a")
          module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
        if kernel["DirectToLdsB"]:
          module.addComment1("local write swap b")
          module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

        module.add(self.closePrefetchGlobalRead2())

      self.states.subTileIdx = 0

      if kernel["ForceUnrollSubIter"]:
        kernel["LoopIters"] = kernel["numSubTiles"] * kernel["numSubTiles"]
        self.states.numMfmaPerIter = self.states.numMfmaPerIter//kernel["LoopIters"]
        self.states.numItersPLR = 1

      # prefetch-local
      if self.states.numItersPLR:
        # not generate wait for local write if LDS write code is not generated
        if not kernel["NoLdsWriteCode"]:
          module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "0prefetch wait for local write"))
        module.add(self._syncThreads(kernel))

        # in some cases need an extra copy of the LDS read with appropriate double buffer offsets
        for plrIdx in range(0, self.states.numItersPLR):
          pack[plrIdx] = Module()
          for espi in range(0, 1):
            for iui in range(0,kernel["InnerUnroll"]):
              if iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                module.addComment1("local read prefetch a")
                localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadA, iui*self.states.numReadsIterCoalescedA, espi, tensorParametersA)
                module.add(localReadCodeA)
                pack[plrIdx].add(packCodeA)
              if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
                if iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]:
                  module.addComment1("local read prefetch metadata")
                  localReadCodeM, packCodeM = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, espi, tPM)
                  module.add(localReadCodeM)
                  pack[plrIdx].add(packCodeM)
              if iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                module.addComment1("local read prefetch b")
                localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadB, iui*self.states.numReadsIterCoalescedB, espi, tensorParametersB)
                module.add(localReadCodeB)
                pack[plrIdx].add(packCodeB)
              if not kernel["ForceUnrollSubIter"] and (iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]):
                module.addComment1("local read inc a")
                module.add(self.localReadInc(kernel, iui, tensorParametersA))
              if not kernel["ForceUnrollSubIter"] and (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]):
                if iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]: # no local read code if DirectToVgpr is enabled
                  module.addComment1("local read inc metadata")
                  module.add(self.localReadInc(kernel, iui, tPM))
              if not kernel["ForceUnrollSubIter"] and (iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]):
                module.addComment1("local read inc b")
                module.add(self.localReadInc(kernel, iui, tensorParametersB))
              if kernel["UsePLRPack"] and self.states.numItersPLR:
                module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA and LRB to complete"))
                module.add(pack[plrIdx])
        self.states.SubTileIdx = (self.states.SubTileIdx + 1) % kernel["numSubTiles"]
      elif self.states.numItersPLR == 0 and kernel["UseCustomMainLoopSchedule"]:
        # For numItersPLR=0 and CMS we prefetch only half the local reads
        localReadCodeA = Module()
        localReadCodeB = Module()
        packCodeA = Module()
        packCodeB = Module()
        for espi in range(0, 1):
          localReadCodeA_, packCodeA_ = self.localReadDo(kernel, 0, 0, espi, tensorParametersA)
          localReadCodeB_, packCodeB_ = self.localReadDo(kernel, 0, 0, espi, tensorParametersB)
          localReadCodeA.add(localReadCodeA_)
          localReadCodeB.add(localReadCodeB_)
          packCodeA.add(packCodeA_)
          packCodeB.add(packCodeB_)
        localReadCodeA = localReadCodeA.flatitems()
        localReadCodeB = localReadCodeB.flatitems()
        packCodeA = packCodeA.flatitems()
        packCodeB = packCodeB.flatitems()

        module.add(self._syncThreads(kernel, "Wait for PGR1 of all waves to complete"))
        for lri in range(len(localReadCodeA) // 2):
          module.add(localReadCodeA[lri])
        for lri in range(len(localReadCodeB) // 2):
          module.add(localReadCodeB[lri])
        module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA and LRB to complete"))
        if kernel["UsePLRPack"]:
          for pi in range(len(packCodeA) // 2):
            module.add(packCodeA[pi])
          for pi in range(len(packCodeB) // 2):
            module.add(packCodeB[pi])

      module.add(self.closeSumAtLeastUnroll(kernel, tensorParametersA, tensorParametersB, prefetch=True, isOptNLL=False, isNGLL=False))

    loopCopies = 2 if expand else 1
    isDTV = (kernel["DirectToVgprA"] or kernel["DirectToVgprB"])
    isULSGRO = kernel["UnrollLoopSwapGlobalReadOrder"] == 1
    needSecondLoop = (not expand) and (isULSGRO or isDTV) # need 2 MainLoop for 2 buffers (PGR1/2)
    needSecondNGLL = (not expand) and (isULSGRO or isDTV) and kernel["PrefetchGlobalRead"] == 2# need 2 NGLL for 2 buffers (PGR2)
    if loopCopies == 1 and needSecondLoop:
      # force to generate 2 loop bodies
      loopCopies = 2

    # open unrolled summation loop
    module.addComment2("Unrolled Loop(s) - Begin")
    module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, beginLabelOnly=False))

    if needSecondLoop and kernel["PrefetchGlobalRead"] == 2:
      # force to generate 2 loop bodies (PGR2 only)
      # TODO: unify 2 loop bodies code generation with else case

      # second LW buffer check for UnrollLoopSwapGlobalReadOrder
      dsWriteBA = True if isULSGRO else False
      # second GR buffer check for DTV
      isDTVGRSecondBuf = True if isDTV else False
      module.add(self._loopBody( kernel, tensorParametersA, tensorParametersB, pack, 0, loopCopies, False , dsWriteBA=dsWriteBA, isDTVGRSecondBuf=isDTVGRSecondBuf, skipClose=True))

      loopLabelToNoGRloopAfterABLoop = Label("NoGRloopAfterABLoop", "" )
      loopCounter = self.loopCounter(kernel, self.states.unrollIdx)
      module.add(SSubU32(dst=loopCounter, src0=loopCounter, \
                         src1=1, \
                         comment="dec counterL"))
      module.add(SCmpLeU32(src0=loopCounter, \
                           src1=hex(2), \
                          comment="counteL<=2"))
      module.add(SCBranchSCC1(labelName=loopLabelToNoGRloopAfterABLoop.getLabelName(), comment="exit LoopL" ))
      # grBA check for UnrollLoopSwapGlobalReadOrder
      grBA = True if isULSGRO else False
      module.add(self._loopBody( kernel, tensorParametersA, tensorParametersB, pack, 1, loopCopies, True , grBA=grBA))
    else:
      for lc in range(0, loopCopies):
        # second GR buffer check for DTV
        isDTVGRSecondBuf = True if isDTV and lc == 0 else False
        # loop body code generation
        finalLoop = lc == loopCopies - 1
        module.add(self._loopBody( kernel, tensorParametersA, tensorParametersB, pack, lc, loopCopies, finalLoop, isDTVGRSecondBuf=isDTVGRSecondBuf ))
    if kernel["ExpertSchedulingMode"] > 0:
      module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0,2]), src=0x0, comment="enable hardware dependency checking"))

    module.addComment1("Before NLL: Check VGPR.checkin for INT8 LW")

    # swap local write, read again before noLoadLoop if PrefetchGlobalRead and DirectToLds is enabled
    # In DirectToLds enabled case, local write address is necessary for prefetch global read (for m0).
    # However, even exit with DirectToLds will not pass with this code (limitation).
    if kernel["PrefetchGlobalRead"] and kernel["ExpandPointerSwap"]:
      # local write for next iter, used to have local writes here
      if(kernel["DirectToLdsA"]):
        module.addComment1("local write swap offsets a")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      if(kernel["DirectToLdsB"]):
        module.addComment1("local write swap offsets b")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))

    if kernel["PrefetchGlobalRead"] == 2:
      # NGLL code generation
      NGLLindex = 0
      NGLLnum = 2 if needSecondNGLL else 1
      if needSecondNGLL:
        # generate extra NGLL for second GR buffer
        module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=True, pack=pack, NLLindex=NGLLindex, NLLnum=NGLLnum))
        module.add(loopLabelToNoGRloopAfterABLoop)
        if kernel["ExpertSchedulingMode"] > 0:
          module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0,2]), src=0x0, comment="enable hardware dependency checking"))
        NGLLindex += 1
      module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=True, pack=pack, NLLindex=NGLLindex, NLLnum=NGLLnum))

    # This "NoLoad" loop is a copy of the unroll loop but with global loads + LDS writes removed
    # doShadowInit is required since this pushes up the store SRD initialization before the NLL
    # OptNLL only allowed for single summation index  - for multiple summation we (currently)
    # execute the NLL inside each unroll iteration not just once at the end.
    if kernel["PrefetchGlobalRead"]:
      if not kernel["SuppressNoLoadLoop"]:
        NeedNLLOddEven  = isDTV # need odd+even NLL for 2 buffers (PGR1/2)
        NLLnum = 2 if NeedNLLOddEven else 1
        gsuComponent = Component.GSU.find(self)
        module.add(gsuComponent.noLoadLoop(self, kernel, tensorParametersA, tensorParametersB, pack))
        for NLLindex in range(0, NLLnum):
          self.saveLocalPointers(kernel, tensorParametersA, tensorParametersB)
          # copy pack
          if NLLindex == NLLnum - 1 or (self.states.packDTVA or self.states.packDTVB or self.states.convDTVA or self.states.convDTVB):
            # last NLL or  pack DTV case, no deep copy for pack
            # pack code for local prefetch is generated in noLoadLoopBody and used for DTV even
            deepCopyPack = pack
          else:
            # deepCopy packCode for OptNLL noLoadLoop
            deepCopyPack = deepcopy(pack)
          module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=False, pack=deepCopyPack, NLLindex=NLLindex, NLLnum=NLLnum))
          self.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)

    if self.states.actualSummationLoops>1:
      module.addComment1("remove stagger offsets")
      module.add(self.removeStagger(kernel, tensorParametersA))
      module.add(self.removeStagger(kernel, tensorParametersB))

    self.vgprPool.add(self.states.a.startVgprValu , \
        self.states.lastValuAB - self.states.a.startVgprValu, "ValuAB")
    module.addComment1("Tail: add ValuA/B vgpr buffer [%u...%u) to pool" % \
        (self.states.a.startVgprValu, self.states.lastValuAB))
    self.vgprPool.add(self.states.lastValuAB , \
        self.states.lastVgprForReads - self.states.lastValuAB, "address vgpr")
    module.addComment1("Tail: add address/G2L vgpr [%u...%u) to pool" % \
        (self.states.lastValuAB, self.states.lastVgprForReads))

    self.removeSgprVarFromPool("SrdWS")

    if not kernel["NoTailLoop"]:
      ########################################
      # Tail Loop
      # which means tail loop not needed.
      ########################################
      self.states.inTailLoop = True
      module.addComment2("Tail Loop")

      # need to unroll tail loop for the following cases
      mEnd = 1
      if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"]:
        mEnd = kernel["LoopIters"]
      if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
        mEnd = kernel["DepthU"]//(kernel["MatrixInstK"]*kernel["LocalSplitU"])

      # Update local write pointers in case the upcoming global reads are writing directly to LDS:
      if kernel["PrefetchGlobalRead"]:
        module.addComment1("local write reset offsets a")
        module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersA))
        if kernel["ExpandPointerSwap"]:
          # reset local write offset in asm code as well
          module.add(self.localWriteResetOffsets(kernel, False, tensorParametersA))
        module.addComment1("local write reset offsets b")
        module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersB))
        if kernel["ExpandPointerSwap"]:
          # reset local write offset in asm code as well
          module.add(self.localWriteResetOffsets(kernel, False, tensorParametersB))

      # tail: global read
      # Check out VGPR for DTVA
      vDtvResources = self.tailLoopAllocDTVVgpr(kernel, tensorParametersA, tensorParametersB)
      for item in vDtvResources:
        if item[0] != -1:
          module.add(item[1])

      # Check out VGPR for G2l
      moduleMacroG2lVgpr, vgprG2L = self.tailLoopAllocG2LVgpr(kernel)
      module.add(moduleMacroG2lVgpr)

      # Check out VGPR for LW
      moduleMacroDTLLWVgpr, vgprLW = self.tailLoopAllocDTLLWVgpr(kernel)
      module.add(moduleMacroDTLLWVgpr)

      module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, -1))
      if self.states.actualSummationLoops==1:
        module.addComment1("remove stagger offsets for tail loop")
        module.add(self.removeStagger(kernel, tensorParametersA))
        module.add(self.removeStagger(kernel, tensorParametersB))

      # if swapGlobalRoad is true, swap the order of global read (B->A)
      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      tailLoopOpt1st = kernel["tailLoopOptA"]
      tailLoopOpt2nd = kernel["tailLoopOptB"]

      tc1 = 'A'
      tc2 = 'B'
      if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
        tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
        tailLoopOpt1st, tailLoopOpt2nd = tailLoopOpt2nd, tailLoopOpt1st
        tc1, tc2 = tc2, tc1

      # globalReadMode = 2 -> optimized by using more vgpr to reorder GR, waitcnt, v_or_b32 instructions.
      # globalReadMode = 3 -> optimized by using wider global read instructions.
      globalReadMode1st = 2 if (((tensorParameters1st["glvw"] * tensorParameters1st["bpeGR"]) < 4) or \
                               tailLoopOpt1st == False) else 3
      globalReadMode2nd = 2 if (((tensorParameters2nd["glvw"] * tensorParameters2nd["bpeGR"]) < 4) or \
                               tailLoopOpt2nd == False) else 3

      globalReadMode1st = 3 if tensorParameters1st["isSwizzled"] else globalReadMode1st
      globalReadMode2nd = 3 if tensorParameters2nd["isSwizzled"] else globalReadMode2nd

      # Use mode 3 for ss_bss type
      if tensorParameters1st["bpeGR"] == 4 and tensorParameters1st["bpeDS"] == 2:
        globalReadMode1st = 3
      if tensorParameters2nd["bpeGR"] == 4 and tensorParameters2nd["bpeDS"] == 2:
        globalReadMode2nd = 3

      if kernel["DirectToLdsA"] and kernel["NonDTLTailLoopA"]:
        if tc1 == 'A':
          globalReadMode1st = 2
        elif tc2 == 'A':
          globalReadMode2nd = 2

      if kernel["DirectToLdsB"] and kernel["NonDTLTailLoopB"]:
        if tc1 == 'B':
          globalReadMode1st = 2
        elif tc2 == 'B':
          globalReadMode2nd = 2

      module.addComment1("Update M0 for DTLDS")
      moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters1st)
      module.add(replaceHolder(moduleTmp, 0))
      module.addComment1("Tail global read %s"%tc1)
      if tailLoopOpt1st and (globalReadMode1st == 2):
        module.add(self.doTailLoopOpt(kernel, tensorParameters1st))
      else:
        module.add(self.globalReadDo(kernel, globalReadMode1st, tensorParameters1st))
      module.addComment1("Update M0 for DTLDS")
      moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters2nd, True)
      module.add(replaceHolder(moduleTmp, 0))
      module.addComment1("Tail global read %s"%tc2)
      if tailLoopOpt2nd and (globalReadMode2nd == 2):
        module.add(self.doTailLoopOpt(kernel, tensorParameters2nd))
      else:
        module.add(self.globalReadDo(kernel, globalReadMode2nd, tensorParameters2nd))

      doA = False
      doB = False
      if globalReadMode1st == 3:
        if tc1 == 'A':
          doA = True if (tensorParameters1st["bpeGR"] % 4 != 0) and (not kernel["ProblemType"]["TLU%s"%(tc1)]) else False
        else:
          doB = True if (tensorParameters1st["bpeGR"] % 4 != 0) and (not kernel["ProblemType"]["TLU%s"%(tc1)]) else False
      if globalReadMode2nd == 3:
        if tc2 == 'A':
          doA = True if (tensorParameters2nd["bpeGR"] % 4 != 0) and (not kernel["ProblemType"]["TLU%s"%(tc2)]) else False
        else:
          doB = True if (tensorParameters2nd["bpeGR"] % 4 != 0) and (not kernel["ProblemType"]["TLU%s"%(tc2)]) else False

      if doA or doB:
        if tc1 == 'A':
          module.add(self.tailLoopGlobalRead(kernel, tensorParameters1st, tensorParameters2nd, doA, doB))
        else:
          module.add(self.tailLoopGlobalRead(kernel, tensorParameters2nd, tensorParameters1st, doA, doB))
      module.add(self._wait(kernel, tensorParameters1st, tensorParameters2nd, 0, -1, -1, "2wait for global read"))
      module.add(self._syncThreads(kernel))

      # init local write offsets to nondtl loads in tail loop.
      module.add(self.lwaInitAddressesForDTLTailLoop(kernel, tensorParameters1st))
      module.add(self.lwaInitAddressesForDTLTailLoop(kernel, tensorParameters2nd))

      # the following read/write addresses could be modified in recalcLocal(Read|Write)Addresses due to policy change
      self.oriLraA = None # back up original local read address vgpr
      self.oriLraB = None
      self.oriLraM = None
      self.oriLwaA = None # back up original local write address vgpr
      self.oriLwaB = None
      self.oriLwaM = None
      if not kernel["NoLdsWriteCode"] or kernel["NonDTLTailLoopA"] or kernel["NonDTLTailLoopB"]:
        # tail: local write
        module.addComment1("local write a")
        tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA)
        module.add(tempLWCodeModA)
        module.addComment1("local write b")
        tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB)
        module.add(tempLWCodeModB)
      # change local read policy from wider local read to one unit of K at a time
      # DirectToVgpr case, use original wider local read instead of recalculating local read address
      if not (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
        module.addComment1("Recalc local read offsets")
        module.add(self.recalcLocalReadAddressesAB(kernel, tensorParametersA, tensorParametersB))
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "5wait for local write"))
      module.add(self._syncThreads(kernel))
      #module.add(self.dumpLds(kernel, 0, 8))

      # tail: free G2L Vgpr
      module.add(self.tailLoopFreeVgpr(vgprG2L, moduleMacroG2lVgpr))

      # tail: free Vgpr for local writes
      module.add(self.tailLoopFreeVgpr(vgprLW, moduleMacroDTLLWVgpr))

      # Check out VGPR for ALU
      valuResources = self.tailLoopAllocValuVgpr(kernel, tensorParametersA, tensorParametersB, tPM)
      for item in valuResources:
        if item[0] != -1:
          module.add(item[1])

      # tail: re-init local read addresses
      if kernel["PrefetchGlobalRead"]:
        module.addComment1("Tail: local read reset offsets a")
        module.add(self.localReadResetOffsets(kernel, tensorParametersA))
        module.addComment1("Tail: local read reset offsets b")
        module.add(self.localReadResetOffsets(kernel, tensorParametersB))
        module.addComment1("Tail: local read init pointers a")
        module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
        module.addComment1("Tail: local read init pointers b")
        module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          module.addComment1("local read reset offsets metadata")
          module.add(self.localReadResetOffsets(kernel, tPM))
          module.addComment1("local read init pointers metadata")
          module.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))
      # tail: macs
      module.addComment1("tail loop: macs")
      module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, -1, None))

      # Try to use InnerUnroll in the tail loop if allowed:
      KinInnerUnroll = kernel["InnerUnroll"]
      if kernel["EnableMatrixInstruction"]:
        KinInnerUnroll *= kernel["MatrixInstK"]

      tailLoopInnerUnroll = 1
      # dot2: currently force tailLoopInnerUnroll = 1
      if (not kernel["UseDotInstruction"]) and (kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0):
        tailLoopInnerUnroll = kernel["InnerUnroll"]

      shiftK = Module()
      for mValue in range(mEnd):
        if mEnd > 1:
          # print tail loop counter if mEnd>1 (means do tail loop unroll)
          module.addComment1("tail loop unroll iter %u"%(mValue))
        pack[0] = Module()
        for iui in range(0, tailLoopInnerUnroll):
          # local read buffer id. No prefetch in tail loop case.
          bufIdx = mValue % self.states.numVgprBuffer
          # DTV case, use different bufIdx for all loop iterations
          bufIdxDTV = mValue % kernel["LoopIters"]
          bufIdxA = (bufIdxDTV if kernel["DirectToVgprA"] else bufIdx) // self.states.numReadsIterCoalescedA
          bufIdxB = (bufIdxDTV if kernel["DirectToVgprB"] else bufIdx) // self.states.numReadsIterCoalescedB
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedA == 0:
            # Reading 16-bit data from LDS requires packing when ECC enabled
            module.addComment1("local read a")
            localReadCodeA, packCodeA = self.localReadDo(kernel, bufIdxA*self.states.numIterPerCoalescedReadA, iui*self.states.numIterPerCoalescedReadA, 0, tensorParametersA)
            module.add(localReadCodeA)
            if kernel["UseF32XEmulation"]:
              shiftK.add(packCodeA)
            else:
              pack[0].add(packCodeA)
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            if mValue*self.states.numIterPerCoalescedReadMetadata < mEnd:
              module.addComment1("local read metadata")
              localReadCodeM, packCodeM = self.localReadDo(kernel, bufIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
              module.add(localReadCodeM)
              pack[0].add(packCodeM)
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedB == 0:
            module.addComment1("local read b")
            localReadCodeB, packCodeB = self.localReadDo(kernel, bufIdxB*self.states.numIterPerCoalescedReadB, iui*self.states.numIterPerCoalescedReadB, 0, tensorParametersB)
            module.add(localReadCodeB)
            if kernel["UseF32XEmulation"]:
              shiftK.add(packCodeB)
            else:
              pack[0].add(packCodeB)
          # adjustment for DirectToLds case
          iuiParam = iui + tailLoopInnerUnroll * mValue//self.states.numReadsIterCoalescedA
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedA == 0:
            module.addComment1("local read inc a")
            module.add(self.localReadInc(kernel, iuiParam, tensorParametersA))
          iuiParam = iui + tailLoopInnerUnroll * mValue
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            module.addComment1("local read inc metadata")
            module.add(self.localReadInc(kernel, iuiParam, tPM))
          iuiParam = iui + tailLoopInnerUnroll * mValue//self.states.numReadsIterCoalescedB
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedB == 0:
            module.addComment1("local read inc b")
            module.add(self.localReadInc(kernel, iuiParam, tensorParametersB))
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "4wait for local read"))

        module.add(pack[0])
        pack[0] = Module()

        if kernel["EnableMatrixInstruction"]:
          # always use vregSetIdx=0 for DirectToVgpr + tail loop
          vregSetIdxMFMA = 0
          module.add(self.mfmaIter(kernel, tensorParametersA, tensorParametersB, mValue, tailLoopInnerUnroll, vregSetIdxMFMA, 0, tail = True, unrollIdx = mValue, postShiftK = shiftK))
        else: # mac instruction
          module.add(self.macIter(kernel, tensorParametersA, tensorParametersB, mValue, tailLoopInnerUnroll, True, True))
        if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
          tP = tensorParametersA if kernel["ProblemType"]["BiasSrc"] == "A" else tensorParametersB
          module.add(self.exclasses.biasSumUnroll.loopSum(self, kernel, tP, 0, tailLoopInnerUnroll))

        finalLoop = mValue == mEnd - 1
        module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, -1, finalLoop, skipCondJumpCounter=mValue))
      # always emit the skip-tail-loop label
      module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, -1, None, emitEndLabelOnly=True))

      # Check in VGPR for VALU
      for item in valuResources:
        if item[0] != -1:
          module.add(self.tailLoopFreeVgpr(item[0], item[1]))

      # Check in VGPR for DTV
      for item in vDtvResources:
        if item[0] != -1:
          module.add(self.tailLoopFreeVgpr(item[0], item[1]))

      # tail: close
      self.states.inTailLoop = False

      # FIXME: Add back.
      if mEnd == 1:
        #add misc vgpr to vgprPool
        self.vgprPool.add(self.states.startVgprMisc , \
          self.states.startVgpr - self.states.startVgprMisc, "misc vgpr") # Add as available
        module.addComment1("Tail: add MISC Vgpr [%u...%u) to pool" % \
                          (self.states.startVgprMisc, self.states.startVgpr))
        if self.states.a.startVgprLocalReadAddr > self.states.startVgpr:
          # Not in the MISC. Release here.
          self.vgprPool.add(self.states.a.startVgprLocalReadAddr , \
            self.states.a.numVgprLocalReadAddr, "LocalReadAddrA vgpr") # Add as available
          module.addComment1("Tail: add LocalReadAddrA Vgpr [%u...%u) to pool" % \
                            (self.states.a.startVgprLocalReadAddr, self.states.a.startVgprLocalReadAddr + self.states.a.numVgprLocalReadAddr))
        if self.states.b.startVgprLocalReadAddr > self.states.startVgpr:
          # Not in the MISC. Release here.
          self.vgprPool.add(self.states.b.startVgprLocalReadAddr , \
            self.states.b.numVgprLocalReadAddr, "LocalReadAddrB vgpr") # Add as available
          module.addComment1("Tail: add LocalReadAddrB Vgpr [%u...%u) to pool" % \
                            (self.states.b.startVgprLocalReadAddr, self.states.b.startVgprLocalReadAddr + self.states.b.numVgprLocalReadAddr))

    if self.do["executeToLoopEnd"]:
      module.add(self.functionEnd(kernel, addLabel=False))

    # extra summation loops: global increment and close
    for i in reversed(range(self.states.otherSummationLoops)):
      module.addComment1("global read inc AB")
      module.add(self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, i, 0))
      module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, i, True))

    module.add(self.endSummation(kernel, tensorParametersA, tensorParametersB))
    if not self.states.doShadowInit:
      module.add(self.globalWriteWorkGroupInit(kernel))

    ####################################
    # Shift Vector Components
    ####################################
    # TODO: support edge case for dot2
    if kernel["EdgeType"] == "ShiftPtr" and not kernel["UseDotInstruction"]:
      # GuaranteeNoPartial means each component in the vector loads is always valid.  In this case we
      # don't need the unshift code

      # shift vector components d0
      if not kernel["GuaranteeNoPartialA"] and tensorParametersA["rtv"]:
        module.addComment1("shift vector components d0")
        module.add(self.shiftVectorComponents(kernel, tensorParametersA))

      # shift vector components d1, for MFMA version, B never entered this
      if not kernel["GuaranteeNoPartialB"] and tensorParametersB["rtv"]:
        module.addComment1("shift vector components d1")
        module.add(self.shiftVectorComponents(kernel, tensorParametersB))

    # dot2: WaveSplitK reduction
    if kernel["NumWaveSplitK"] > 1:
      module.add(self.waveSplitKReduction(kernel))

    ####################################
    # LocalSplitU reduction
    ####################################
    #if kernel["NumThreads"]%kernel["MacroTile0"] == 0:
    if kernel["LocalSplitU"] > 1:
      module.addComment1("LocalSplitU: local write and read")
      lsuComponent = Component.LSU.find(self)
      module.add(lsuComponent.writeReadReduction(self, kernel))

      # LocalSplitU: global write indices
      module.addComment1("LocalSplitU: global write indices")
      module.add(lsuComponent.globalWriteIndices(self, kernel))

      # LocalSplitU: global write
      module.addComment1("LocalSplitU: global write")
      module.add(lsuComponent.globalWrite(self, kernel, tensorParametersA, tensorParametersB))

    else:
      ####################################
      # NOT LocalSplitU
      ####################################

      # global write indices
      module.addComment1("not-LocalSplitU: global write indices")
      module.add(self.notLocalSplitUGlobalWriteIndices(kernel))

      # global write
      module.addComment1("not-LocalSplitU: global write")
      module.add(self.notLocalSplitUGlobalWrite(kernel, tensorParametersA, tensorParametersB))

    module.add(self.functionEnd(kernel, addLabel=True))

    # Add a label at the end of the asm for indexing.
    module.add(Label("ASM_End", "The end of the kernel"))

    moduleKernelBody.addBody(module)
    self.checkResources(kernel, moduleKernelBody) # check resource available or not

    # Tensile instruction pass, temporarily disable due to build time.
    # Kernels with epilog especially with activation is too long (50000~ lines).
    # Need to refactor global write elements.
    ripo = rocIsaPassOption()
    ripo.removeDupFunc = bool(kernel["ActivationFuncCall"])
    ripo.numWaves = kernel["NumThreads"] // kernel["WavefrontSize"]
    if kernel["ProblemType"]["ActivationType"] == "all":
      ripo.removeDupAssign = False
    if self.states.archCaps["HasSchedMode"]:
      ripo.insertDelayAlu = True
    passResult = rocIsaPass(moduleKernelBody, ripo)
    kernel["MathClocksUnrolledLoop"] = passResult.cycles


    error = self.states.overflowedResources
    print2(f"  found error code {error} with overflowed resources set to {self.states.overflowedResources}")

    return (error, str(moduleKernelBody))

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def _initKernel(self, kernel, tensorParametersA, tensorParametersB):
    assert kernel["KernelLanguage"] == "Assembly"
    self.language   = "ASM"
    # ISA version, such as 803
    version = tuple(kernel["ISA"])
    ti = rocIsa.getInstance()
    ti.setKernel(version, kernel["WavefrontSize"])

    self.consts = ConstValues()
    self.states = StateValues(version=version, kernel=kernel, kernelName=getKernelNameMin(kernel, self.debugConfig.splitGSU))
    self.vgprs  = StateVgprs()
    self.sgprs  = collections.OrderedDict()
    self.codes  = CodeModules()
    self.labels = LabelManager()

    # external classes
    if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
      self.exclasses.biasSumUnroll = Component.SumUnroll.find(self)
      assert self.exclasses.biasSumUnroll


    if kernel["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
      self.exclasses.activation.setUseCache(True)
    self.exclasses.activation.setGuard(not kernel["ProblemType"]["ActivationNoGuard"])
    self.exclasses.activation.setAlt(kernel["ActivationAlt"])

    self.states.asmCaps  = ti.getAsmCaps()
    self.states.archCaps = ti.getArchCaps()
    self.states.regCaps  = ti.getRegCaps()

    self.asmAssert = Assert(self.states.laneSGPRCount, kernel["WavefrontSize"], self.db["EnableAsserts"])

    # Only assembly supports scheduling
    if kernel["KernelLanguage"] == "Assembly":
      self.states.scheduleGlobalRead = kernel["ScheduleGlobalRead"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"] # flat updates lgkmcnt counts = hard to schedule flat loads
      self.states.scheduleLocalWrite = kernel["ScheduleLocalWrite"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"]  # flat updates lgkmcnt counts = hard to schedule writes and loads?
      self.states.scheduleIterAlg = kernel["ScheduleIterAlg"]
    else:
      self.states.scheduleGlobalRead = 0
      self.states.scheduleLocalWrite = 0
      self.states.scheduleIterAlg = 0

    self.states.actualSummationLoops = kernel["ProblemType"]["NumIndicesSummation"]
    self.states.otherSummationLoops  = self.states.actualSummationLoops-1
    self.states.otherSummations      = kernel["ProblemType"]["NumIndicesSummation"]-1 # not loops but summations vars

    # doShadowInit performs initialization in the 'shadow' of the global mem prefetch
    if not kernel["ForceDisableShadowInit"]:
      if kernel["PrefetchGlobalRead"]:
        if self.states.actualSummationLoops == 1:
          self.states.doShadowInit = 2 # 2 is both store setup and initC
        else:
          # can't do shadow initC with multiple summation since this resets the ValuC counters
          # on each unroll iteration.
          self.states.doShadowInit = 1 # 1 is just store setup

    self.states.indexChars = []
    for i in range(0, len(INDEX_CHARS)):
      self.states.indexChars.append(INDEX_CHARS[i])
    self.states.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.states.indexChars[kernel["ProblemType"]["Index0"]]
    self.states.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.states.indexChars[kernel["ProblemType"]["Index1"]]
    self.states.unrollIdx = kernel["ProblemType"]["NumIndicesSummation"]-1
    self.states.unrollChar = \
        self.states.indexChars[kernel["ProblemType"]["IndicesSummation"][\
        self.states.unrollIdx]]
    self.states.tileChar0 = self.states.indexChars[kernel["ProblemType"]["Index0"]]
    self.states.tileChar1 = self.states.indexChars[kernel["ProblemType"]["Index1"]]

    """
    if kernel["ProblemType"]["Tensor0"]==0:
      kernel["ThreadTileA"] = kernel["ThreadTile0"]
      kernel["ThreadTileB"] = kernel["ThreadTile1"]
      kernel["SubGroupA"] = kernel["SubGroup0"]
      kernel["SubGroupB"] = kernel["SubGroup1"]
      kernel["MacroTileA"] = kernel["MacroTile0"]
      kernel["MacroTileB"] = kernel["MacroTile1"]
    else:
      kernel["ThreadTileB"] = kernel["ThreadTile0"]
      kernel["ThreadTileA"] = kernel["ThreadTile1"]
      kernel["SubGroupB"] = kernel["SubGroup0"]
      kernel["SubGroupA"] = kernel["SubGroup1"]
      kernel["MacroTileB"] = kernel["MacroTile0"]
      kernel["MacroTileA"] = kernel["MacroTile1"]
    """

    """
    # original parameters
    NumLoadsCoalesced -> NumLoadsPerpendicular
    # new intermediate parameters
    numReadsTile # nrt
    numReadsUnroll # nru
    numWritesCoal # nwc
    numWritesPerp # nwp
    numWritesCoalVecComp # nwvc
    numWritesPerpVecComp # nwvp
    """

    if kernel["EnableMatrixInstruction"] and kernel["LocalReadVectorWidth"] >= kernel["MIInputPerThread"]:
      WLR = max(kernel["LocalReadVectorWidth"]//kernel["MIInputPerThread"], 1)
      self.states.numItersPLR = kernel["PrefetchLocalRead"]%(kernel["LoopIters"]//WLR)
    else:
      self.states.numItersPLR = kernel["PrefetchLocalRead"]%(kernel["LoopIters"])

    if kernel["ClusterLocalRead"]:
      self.states.numVgprBuffer = kernel["LoopIters"]
    else:
      self.states.numVgprBuffer = kernel["PrefetchLocalRead"] + 1

    if kernel["ClusterLocalRead"]:
      self.states.numVgprBufferPackA = kernel["LoopIters"]
      self.states.numVgprBufferPackB = kernel["LoopIters"]
    else:
      self.states.numVgprBufferPackA = self.states.numItersPLR + 1
      self.states.numVgprBufferPackB = self.states.numItersPLR + 1

    # TODO load sub-vector
    vwa = kernel["GlobalReadVectorWidthA"]
    vwb = kernel["GlobalReadVectorWidthB"]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      vwm = kernel["GlobalReadVectorWidthMetadata"]

    # force lrvwTile = 1 for numBytes >= 4 + MIInputPerThread > 1
    # TODO: implement extra logic to swap vgprs after local read to suport lrvwTile > 1 for umBytes >= 4 + MIInputPerThread > 1
    forceLrvwTile1 = kernel["ProblemType"]["DataType"].numBytes() >= 4 and (kernel["EnableMatrixInstruction"] and kernel["MIInputPerThread"] > 1)
    if not kernel["UnrollMajorLDSA"] and not forceLrvwTile1:
      self.states.lrvwTileA = kernel["VectorWidthA"]
    else:
      self.states.lrvwTileA = 1

    if not kernel["UnrollMajorLDSB"] and not forceLrvwTile1:
      self.states.lrvwTileB = kernel["VectorWidthB"]
    else:
      self.states.lrvwTileB = 1

    # DirectToVgpr + pack (v_perm)
    self.states.packDTVA = kernel["DirectToVgprA"] and self.states.lrvwTileA > 1 and kernel["MIInputPerThread"] > 1
    self.states.packDTVB = kernel["DirectToVgprB"] and self.states.lrvwTileB > 1 and kernel["MIInputPerThread"] > 1

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      if kernel["ClusterLocalRead"]:
        self.states.numVgprBufferPackMetadata = kernel["LoopIters"]
      else:
        self.states.numVgprBufferPackMetadata = self.states.numItersPLR + 1
      if not kernel["UnrollMajorLDSMetadata"]:
        self.states.lrvwTileMetadata = kernel["VectorWidthMetadata"]
      else:
        self.states.lrvwTileMetadata = 1
      if self.states.lrvwTileMetadata > 1:
        self.states.numVgprBufferPackB = 1

    if self.states.lrvwTileA > 1 and (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16() or \
      kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()):
      self.states.numVgprBufferPackA = 1

    if self.states.lrvwTileB > 1 and (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16() or \
      kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()):
      self.states.numVgprBufferPackB = 1

    if kernel["UnrollMajorLDSA"]:
      divider = 2 if (kernel["ProblemType"]["Sparse"] == 1) else 1
      self.states.lrvwUnrollA = kernel["LocalReadVectorWidth"] // divider
    else:
      self.states.lrvwUnrollA = 1

    if kernel["UnrollMajorLDSB"]:
      divider = 2 if (kernel["ProblemType"]["Sparse"] == 2) else 1
      self.states.lrvwUnrollB = kernel["LocalReadVectorWidth"] // divider
    else:
      self.states.lrvwUnrollB = 1

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      if kernel["UnrollMajorLDSMetadata"]:
        self.states.lrvwUnrollMetadata = kernel["MIInputPerThreadMetadata"]
      else:
        self.states.lrvwUnrollMetadata = 1

    # Wider LocalRead
    if kernel["EnableMatrixInstruction"]:
      self.states.numReadsIterCoalescedA = ceil(self.states.lrvwUnrollA / kernel["MIInputPerThreadA"])
      self.states.numReadsIterCoalescedB = ceil(self.states.lrvwUnrollB / kernel["MIInputPerThreadB"])
    else:
      self.states.numReadsIterCoalescedA = self.states.lrvwUnrollA // kernel["NumDotElements"] if kernel["UseDotInstruction"] else 1
      self.states.numReadsIterCoalescedB = self.states.lrvwUnrollB // kernel["NumDotElements"] if kernel["UseDotInstruction"] else 1
    self.states.numIterPerCoalescedReadA = max(1,self.states.numReadsIterCoalescedA//kernel["InnerUnroll"])
    self.states.numIterPerCoalescedReadB = max(1,self.states.numReadsIterCoalescedB//kernel["InnerUnroll"])

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      if kernel["EnableMatrixInstruction"]:
        self.states.numReadsIterCoalescedMetadata = ceil(self.states.lrvwUnrollMetadata / kernel["MIInputPerThreadMetadata"])
      else:
        self.states.numReadsIterCoalescedMetadata  = 1
    else:
      self.states.numReadsIterCoalescedMetadata  = 1
    self.states.numIterPerCoalescedReadMetadata = max(1,self.states.numReadsIterCoalescedMetadata//kernel["InnerUnroll"])

    if kernel["ScheduleIterAlg"] == 3 or kernel["ScheduleIterAlg"] == 2:
      self.states.numMfmaPerIter = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1] * kernel["InnerUnroll"]
      if kernel["ProblemType"]["DataType"].isComplex(): self.states.numMfmaPerIter *= 4
      elif kernel["UseF32XEmulation"]: self.states.numMfmaPerIter *= 3

    # NamedTuple is immutable
    class intermediateTPValues(NamedTuple):
      numReadsTile: int = -1
      numReadsUnroll: int = -1
      readTileDimVector: bool = False
      writeTileDimComponents: bool = False
      numWritesCoalVecComp: int = -1
      numWritesPerpVecComp: int = -1
      # convert tile/unroll to para/perp
      numReadsCoalVecComp: int = -1
      numReadsPerpVecComp: int = -1

    def readWriteVectors(mat, vw, kernel):
      ########################################
      # read vectors or vector components
      ########################################
      # Two dim: tile and unroll
      if kernel["ProblemType"]["TLU%s"%mat]: # NT no transpose
        numReadsTile = kernel["NumLoadsCoalesced%s"%mat]
        numReadsUnroll = kernel["NumLoadsPerpendicular%s"%mat]
        readTileDimVector = True # Vector
      else: # TN yes transpose
        numReadsTile = kernel["NumLoadsPerpendicular%s"%mat]
        numReadsUnroll = kernel["NumLoadsCoalesced%s"%mat]
        readTileDimVector = False # Scalar

      numReadsCoalVecComp   = vw
      numReadsPerpVecComp   = 1

      ########################################
      # write vectors or vector components
      ########################################
      if kernel["ProblemType"]["TLU%s"%mat] != kernel["UnrollMajorLDS%s"%mat]: # NT no transpose
        writeTileDimComponents = False # Vector
        # writeCoal indicates writes should be done in the coal dim or else perp
        numWritesCoalVecComp = vw
        numWritesPerpVecComp = 1
      else: # TN yes transpose
        writeTileDimComponents = True
        numWritesCoalVecComp = 1
        numWritesPerpVecComp = vw

      return intermediateTPValues(numReadsTile, numReadsUnroll, readTileDimVector, \
        writeTileDimComponents, numWritesCoalVecComp, numWritesPerpVecComp, \
        numReadsCoalVecComp, numReadsPerpVecComp)

    itP = dict()
    itP["A"] = readWriteVectors("A", vwa, kernel)
    itP["B"] = readWriteVectors("B", vwb, kernel)

    self.getTensorParameters(tensorParametersA, kernel, itP, "A")
    self.getTensorParameters(tensorParametersB, kernel, itP, "B")

    tensorParametersA["PackedIndices"] = kernel["PackedC%uIndicesX"%tensorParametersA["tile01Idx"]]
    tensorParametersB["PackedIndices"] = kernel["PackedC%uIndicesX"%tensorParametersB["tile01Idx"]]

    tensorParametersM = None
    tensorParametersA["tpsMetadata"] = None
    tensorParametersB["tpsMetadata"] = None

    if kernel["ProblemType"]["Sparse"]:
      if not kernel["DirectToVgprSparseMetadata"]:
        itP["Metadata"] = readWriteVectors("Metadata", vwm, kernel)
        tensorParametersM = {}
        self.getTensorParameters(tensorParametersM, kernel, itP, "Metadata")
        tensorParametersM["localReadOffset"] = 0
        tensorParametersM["PackedIndices"] = kernel["PackedC%uIndicesX"%tensorParametersM["tile01Idx"]]
        if  kernel["ProblemType"]["Sparse"] == 2:
          tensorParametersB["tpsMetadata"] = tensorParametersM
        else:
          tensorParametersA["tpsMetadata"] = tensorParametersM

    # init these here in case some kernel pieces are disabled for performance exploration:
    tensorParametersA["localReadOffset"] = 0
    tensorParametersB["localReadOffset"] = 0

    # DirectToVgpr + input conversion
    self.states.convDTVA = kernel["DirectToVgprA"] and kernel["ConvertAfterDS"] and tensorParametersA["bpe"] > tensorParametersA["bpeGR"]
    self.states.convDTVB = kernel["DirectToVgprB"] and kernel["ConvertAfterDS"] and tensorParametersB["bpe"] > tensorParametersB["bpeGR"]

    #---
    # Internal optimization and debug controls.
    # These have a default which is almost always faster so don't make a full-blown YAML parm
    # But have a control here so we can disable for debugging and also easily tell
    # which parts of the code were changed to support the new mode.
    self.states.globalReadIncsUseVgpr = False if kernel["BufferLoad"] else True

    # If True, GRO are expressed as offsets from the beginning of the macro-tile, and the SRD
    # is set to the beginning of the macro-tile.
    # If False, GRO are expressed as offsets from the beginning of the lowest 2 dimensions
    # in the tensor.
    # True can allow Buffer-Based logic to have significantly higher range and handle larger tensors
    # groOffsetInMacroTile doesn't work with pointer-shift because it sets the SRD to point to the
    # start of the macro-tile - if we overhang by small number of elements (<GRVW) then can't shift
    # back to get all the data.
    # groOffsetInMacroTile doesn't work with packed dims since these need to set SRD to the tensor base
    # then extract the packed dimensions from the flattened index (including the workgroup) and scale by strides
    # - the index is per-work-item so can't put work-group into the SRD
    if len(kernel["PackedC0IndicesX"])==1 and len(kernel["PackedC1IndicesX"])==1 and kernel["BufferLoad"]:
      self.states.groOffsetInMacroTile = 1
    else:
      self.states.groOffsetInMacroTile = 0

    # use 64-bit buffer limit shadow register
    # but not implemented or tested
    self.states.use64bShadowLimit = kernel["Use64bShadowLimit"] and kernel["BufferLoad"]

    # Check if the address setup code for LWA and GRO causes register growth.
    # This is not an error condition but bears further investigation.
    # Realistically we just have the GlobalToLocal VGPRs, all else is growth.
    self.states.preventVgprOverflowDuringNewTile = 0 and not self.debugConfig.forceGenerateKernel

    # For Beta:
    # Rather than waiting for all loads to finish with s_waitcnt vmcnt(0), interleave
    # appropriate vmcnts into the stores so they issue as loads become available
    self.states.interleaveStoreVmcnt = (not kernel["GroupLoadStore"]) and kernel["BufferStore"]

    # if >0, shift the start of the SRD left by specified #elements (not bytes)
    # Gives pointer shift some room to move left, even into the previous macro-tile
    # This slightly reduces the range of the GRO since they have to include the offset
    # Pointer shift still cannot be used with very small matrices < GRVW
    self.states.srdShiftLeft["A"] = kernel["GlobalReadVectorWidthA"]
    self.states.srdShiftLeft["B"] = kernel["GlobalReadVectorWidthB"]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      self.states.srdShiftLeft["Metadata"] = kernel["GlobalReadVectorWidthMetadata"]

    # checkGRO requires useSgprForGRO=0 so that code allocates and uses
    # the VGPRs that are used for the GRO offset checking
    assert not (kernel["_UseSgprForGRO"] and self.states.checkGRO)

    self.states.doubleVgpr = False
    if self.states.archCaps["ArchAccUnifiedRegs"] or (kernel["WavefrontSize"] == 32):
      self.states.doubleVgpr = True

    if kernel["EnableMatrixInstruction"]:
      if (kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64') and (not self.states.asmCaps["HasMFMA_f64"]):
        raise RuntimeError("FP64 MatrixInstruction not supported for {0}".format(self.states.version))
      elif not ( self.states.asmCaps["HasMFMA"] or self.states.asmCaps["HasWMMA"]):
        raise RuntimeError("MatrixInstruction not supported for {0}".format(self.states.version))

      if kernel["MFMA_BF16_1K"] and not self.states.asmCaps["HasMFMA_bf16_1k"]:
        raise RuntimeError("BF16_1k MatrixInstruction not supported for {0}".format(self.states.version))

      if kernel["ProblemType"]["Sparse"] and not self.states.asmCaps["HasSMFMA"]:
        raise RuntimeError("Sparse MatrixInstruction not supported for {0}".format(self.states.version))

      if (kernel["EnableF32XdlMathOp"] and kernel["ProblemType"]["F32XdlMathOp"].isXFloat32() and (not self.states.asmCaps["HasMFMA_xf32"])):
        if not kernel["UseF32XEmulation"]:
          raise RuntimeError("XF32 MatrixInstruction not supported for {0}".format(self.states.version))

    if not self.states.asmCaps["HasDirectToLds"]:
      kernel["DirectToLdsA"] = False
      kernel["DirectToLdsB"] = False
      kernel["LocalWriteUseSgprA"] = False # Requires DirectToLdsA
      kernel["LocalWriteUseSgprB"] = False # Requires DirectToLdsB

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      kernel["LocalWriteUseSgprMetadata"] = False

    # The inst HasAtomicAdd is using is not compatible with int32.
    self.states.useAtomicAdd = (self.states.asmCaps["HasAtomicAdd"] and kernel["ProblemType"]["ComputeDataType"].isSingle()) and \
                        (kernel["_GlobalAccumulation"] == 'SingleBuffer')

    if self.states.asmCaps["v_fma_mix_f32"]:
      self.states.mixinst = VFmaMixF32
    elif self.states.asmCaps["v_mad_mix_f32"]:
      self.states.mixinst = VMadMixF32


    self.states.bpeAB = int(self.states.bpr * kernel["ProblemType"]["DataType"].numRegisters())
    self.states.bpeE  = int(self.states.bpr * kernel["ProblemType"]["DataTypeE"].numRegisters())
    self.states.bpeCinternal = int(self.states.bpr * kernel["ProblemType"]["ComputeDataType"].numRegisters())

    self.states.bpeCexternalGSU1 = int(self.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())
    self.states.bpeCexternal = self.states.bpeCexternalGSU1
    if kernel["_GlobalAccumulation"] and kernel["_GlobalAccumulation"] != 'PartialsBuffer':
      self.states.bpeCexternal = self.states.bpeCinternal


    # special case for wmma h and b
    if (kernel["EnableMatrixInstruction"]
            and self.states.asmCaps["HasWMMA_V1"]
            and (kernel["ProblemType"]["ComputeDataType"].numRegisters() == 0.5)):
        self.states.bpeCinternal = 4
        if kernel["_GlobalAccumulation"]:
            self.states.bpeCexternal = 2

    self.states.HHH_WMMA = kernel["EnableMatrixInstruction"] \
                                and self.states.asmCaps["HasWMMA"] \
                                and kernel["ProblemType"]["DestDataType"].isHalf() \
                                and (not kernel["ProblemType"]["HighPrecisionAccumulate"])

    # HPA not allowed in dgemm, cgemm, zgemm, sgemm
    if kernel["ProblemType"]["HighPrecisionAccumulate"] and \
       not (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16() or \
          kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isInt8() or \
          kernel["ProblemType"]["DataType"].is8bitFloat()):
        print("HighPrecisionAccumulate only valid when DataType is half, bf16, Int8x4, Int8, fp8, bf8. Forcing HPA to False")
        kernel["ProblemType"]["HighPrecisionAccumulate"] = False

    assert self.states.bpeAB == tensorParametersA["bpe"]
    assert self.states.bpeAB == tensorParametersB["bpe"]

    #######################################L
    # Available Memory Instructions
    ########################################

    # name, numAddresses, numOffsets, offsetMultiplier, blockWidth, formatting):
    ########################################
    # Local Read
    _ds_load_b128 = MemoryInstruction(DSLoadB128,  1, 1, 4, 4)
    _ds_load2_b64 = MemoryInstruction(DSLoad2B64,  1, 2, 2, 2)
    _ds_load_b64 = MemoryInstruction(DSLoadB64,    1, 1, 2, 2)
    _ds_load2_b32 = MemoryInstruction(DSLoad2B32,  1, 2, 1, 1)
    _ds_load_b32 = MemoryInstruction(DSLoadB32,    1, 1, 1, 1)
    _ds_load_u16 = MemoryInstruction(DSLoadU16,    1, 1, 1, 0.5)
    _ds_load_u8 = MemoryInstruction(DSLoadU8,      1, 1, 1, 0.25)
    _ds_load_b64_tr_b16 = MemoryInstruction(DSLoadB64TrB16,    1, 1, 2, 2)

    ########################################
    # Local Write
    _ds_store_b256 = MemoryInstruction(DSStoreB256,  1, 1, 8, 8)
    _ds_store_b128 = MemoryInstruction(DSStoreB128,  1, 1, 4, 4)
    _ds_store2_b64 = MemoryInstruction(DSStore2B64,  1, 2, 2, 2)
    _ds_store_b64 = MemoryInstruction(DSStoreB64,    1, 1, 2, 2)
    _ds_store2_b32 = MemoryInstruction(DSStore2B32,  1, 2, 1, 1)
    _ds_store_b32 = MemoryInstruction(DSStoreB32,    1, 1, 1, 1)
    _ds_store_b16 = MemoryInstruction(DSStoreB16,    1, 1, 1, 0.5)
    _ds_store_b8 = MemoryInstruction(DSStoreB8,      1, 1, 1, 0.25)
    ########################################
    # Global Read
    _flat_load_b128 = MemoryInstruction(FlatLoadB128, 1, 0, 0, 4)
    _flat_load_b64 = MemoryInstruction(FlatLoadB64,   1, 0, 0, 2)
    _flat_load_b32 = MemoryInstruction(FlatLoadB32,   1, 0, 0, 1)

    _buffer_load_b128 = MemoryInstruction(BufferLoadB128, 1, 0, 0, 4)
    _buffer_load_b64 = MemoryInstruction(BufferLoadB64, 1, 0, 0, 2)
    _buffer_load_b32 = MemoryInstruction(BufferLoadB32, 1, 0, 0, 1)
    # generate half directly w/o using the format string to handle hi/lo correctly
    _buffer_load_d16_b16 = MemoryInstruction(BufferLoadD16B16, 1, 0, 0, 0.5)
    # generate byte directly w/o using the format string to handle hi/lo correctly
    _buffer_load_d16_u8 = MemoryInstruction(BufferLoadD16U8, 1, 0, 0, 0.25)

    self.buff_load_inst_offset_max = 4096

    ########################################
    # Global Write
    _flat_store_b128 = MemoryInstruction(FlatStoreB128, 1, 0, 0, 4)
    _flat_store_b64  = MemoryInstruction(FlatStoreB64,  1, 0, 0, 2)
    _flat_store_b32  = MemoryInstruction(FlatStoreB32,  1, 0, 0, 1)

    ########################################
    # Available Memory Instructions per Architecture
    # gfx701 "Hawaii"
    # gfx801 "Carrizo"
    # gfx802 "Tonga"
    # gfx803 "Fiji"
    # gfx900
    ########################################
    if (kernel["BufferLoad"]):
      chosen_load_b128 = _buffer_load_b128
      chosen_load_b64  = _buffer_load_b64
      chosen_load_b32  = _buffer_load_b32
      chosen_load_b16  = _buffer_load_d16_b16
      chosen_load_b8   = _buffer_load_d16_u8
    else:
      chosen_load_b128 = _flat_load_b128
      chosen_load_b64  = _flat_load_b64
      chosen_load_b32  = _flat_load_b32
      chosen_load_b16  = _flat_load_b32 # not supported
      chosen_load_b8   = _flat_load_b32 # not supported

    chosen_store_b128 = _flat_store_b128
    chosen_store_b64  = _flat_store_b64
    chosen_store_b32  = _flat_store_b32

    self.memoryInstructions = {
          "GlobalRead" : [ chosen_load_b128, chosen_load_b64, chosen_load_b32,
                           chosen_load_b16, chosen_load_b8 ],
          "GlobalWrite": [ chosen_store_b128, chosen_store_b64, chosen_store_b32 ],
          "LocalRead"  : [ _ds_load_b128, _ds_load2_b64, _ds_load_b64,
                           _ds_load2_b32, _ds_load_b32, _ds_load_u16, _ds_load_u8,
                           _ds_load_b64_tr_b16],
          "LocalWrite" : [ _ds_store_b256, _ds_store_b128, _ds_store2_b64,
                           _ds_store_b64, _ds_store2_b32, _ds_store_b32,
                           _ds_store_b16, _ds_store_b8 ]
        }

    ####################################
    # choose memory instructions
    ####################################

    ########################################

    instructions = self.memoryInstructions
    self.initGlobalReadMemoryInstruction(instructions, tensorParametersA, self.states.bpr)
    self.initGlobalReadMemoryInstruction(instructions, tensorParametersB, self.states.bpr)
    self.initLocalWriteMemoryInstruction(instructions, kernel, tensorParametersA, self.states.bpr)
    self.initLocalWriteMemoryInstruction(instructions, kernel, tensorParametersB, self.states.bpr)
    self.initLocalReadMemoryInstruction(instructions, kernel, tensorParametersA, self.states.bpr)
    self.initLocalReadMemoryInstruction(instructions, kernel, tensorParametersB, self.states.bpr)

    if tensorParametersM is not None:
      self.initGlobalReadMemoryInstruction(instructions, tensorParametersM, self.states.bpr)
      self.initLocalWriteMemoryInstruction(instructions, kernel, tensorParametersM, self.states.bpr)
      self.initLocalReadMemoryInstruction(instructions, kernel, tensorParametersM, self.states.bpr)

    # global reads per instruction
    tensorParametersA["nrcvpi"] = int((tensorParametersA["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersA["bpeGR"])
    tensorParametersB["nrcvpi"] = int((tensorParametersB["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersB["bpeGR"])
    tensorParametersA["nwcvpi"] = int((tensorParametersA["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersA["bpeDS"])
    tensorParametersB["nwcvpi"] = int((tensorParametersB["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersB["bpeDS"])

    if tensorParametersM is not None:
      tensorParametersM["nrcvpi"] = int((tensorParametersM["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersM["bpeDS"])
      tensorParametersM["nwcvpi"] = int((tensorParametersM["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersM["bpeDS"])
    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
    if kernel["EnableMatrixInstruction"]:
      #jgolds bpeCinternal because we are allocating accumulation registers here
      self.states.c.numVgprValu = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.states.bpeCinternal)//self.states.bpr

      # pack or input conversion DTV case, need double buffer (LoopIters * 2)
      numVgprBufferA = self.states.numVgprBuffer if not (self.states.packDTVA or self.states.convDTVA) else kernel["LoopIters"] * 2
      numVgprBufferB = self.states.numVgprBuffer if not (self.states.packDTVB or self.states.convDTVB) else kernel["LoopIters"] * 2
      valuBlocks  = self.states.numVgprBuffer * kernel["InnerUnroll"] # for Sparse
      valuBlocksA = numVgprBufferA * kernel["InnerUnroll"]
      valuBlocksB = numVgprBufferB * kernel["InnerUnroll"]

      self.states.a.numVgprValuPerBlock = kernel["MIWaveTileA"] * kernel["MIInputPerThreadA"] * tensorParametersA["bpe"] // self.states.bpr
      self.states.b.numVgprValuPerBlock = kernel["MIWaveTileB"] * kernel["MIInputPerThreadB"] * tensorParametersB["bpe"] // self.states.bpr

      # change numVgprValuAPerBlock to 0 if DirectToVgpr is enabled (except for DTV + (pack or input conversion))
      if kernel["DirectToVgprA"] and not (self.states.packDTVA or self.states.convDTVA):
        self.states.a.numVgprValuPerBlock = 0
      if kernel["DirectToVgprB"] and not (self.states.packDTVB or self.states.convDTVB):
        self.states.b.numVgprValuPerBlock = 0

      self.states.a.numVgprValu = self.states.a.numVgprValuPerBlock * valuBlocksA
      if self.states.lrvwTileA > 1 and tensorParametersA["bpe"] < 4 and not (kernel["UsePLRPack"] and self.states.numItersPLR):
        self.states.a.numVgprValu = self.states.a.numVgprValuPerBlock * kernel["InnerUnroll"]

      self.states.b.numVgprValu = self.states.b.numVgprValuPerBlock * valuBlocksB
      if self.states.lrvwTileB > 1 and tensorParametersB["bpe"] < 4 and not (kernel["UsePLRPack"] and self.states.numItersPLR):
        self.states.b.numVgprValu = self.states.b.numVgprValuPerBlock * kernel["InnerUnroll"]

    else: # mac instruction
      valuBlocksA = (1 + kernel["PrefetchLocalRead"]) * kernel["InnerUnroll"]
      valuBlocksB = (1 + kernel["PrefetchLocalRead"]) * kernel["InnerUnroll"]

      if kernel["UseDotInstruction"]:
        # dot2: at least read NumDotElements elements
        self.states.a.numVgprValuPerBlock = kernel["ThreadTileA"] * tensorParametersA["bpe"] * kernel["NumDotElements"] // self.states.bpr
        self.states.b.numVgprValuPerBlock = kernel["ThreadTileB"] * tensorParametersB["bpe"] * kernel["NumDotElements"] // self.states.bpr
      else:
        self.states.a.numVgprValuPerBlock = kernel["ThreadTileA"] * tensorParametersA["bpe"] // self.states.bpr
        self.states.b.numVgprValuPerBlock = kernel["ThreadTileB"] * tensorParametersB["bpe"] // self.states.bpr

      self.states.c.numVgprValu = kernel["ThreadTile0"] * kernel["ThreadTile1"] * kernel["ProblemType"]["ComputeDataType"].numRegisters()
      self.states.a.numVgprValu = self.states.a.numVgprValuPerBlock * valuBlocksA
      self.states.b.numVgprValu = self.states.b.numVgprValuPerBlock * valuBlocksB

    if kernel["ProblemType"]["Sparse"]:
      if kernel["DirectToVgprSparseMetadata"]:
        miWaveTile = kernel["MIWaveTileB"] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTileA"]
        self.states.m.numVgprValuPerBlock = miWaveTile * kernel["LoopIters"] #every 8bit need 1 register
        valuBlocks = (kernel["PrefetchGlobalRead"] + 1)
        self.states.m.numVgprValu = self.states.m.numVgprValuPerBlock * valuBlocks
      else:
        if self.states.lrvwTileMetadata > 1:
          self.states.m.numVgprValuPerBlock = kernel["MIWaveTileMetadata"] * roundUp(kernel["MIInputPerThreadMetadata"] / self.states.bpr)
        else:
          self.states.m.numVgprValuPerBlock = kernel["MIWaveTileMetadata"] * kernel["MIInputPerThreadMetadata"]
        self.states.m.numVgprValu = self.states.m.numVgprValuPerBlock * valuBlocks
        if self.states.lrvwTileMetadata > 1 and tensorParametersM["bpe"] < 4:
          self.states.m.numVgprValu = self.states.m.numVgprValuPerBlock * kernel["InnerUnroll"]


    ####################################
    # num vgprs: global -> local elements
    self.states.a.numVgprG2L = 0
    numVgprG2LAllocatedLocal = 0


    bpeMax = tensorParametersA["bpeDS"] if kernel["ConvertAfterDS"] else max(tensorParametersA["bpeGR"], tensorParametersA["bpe"])
    statesANumVgprG2L = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] * \
        kernel["GlobalReadVectorWidthA"] * bpeMax) / (float)(self.states.bpr))
    tpA      = self.states.bpr if bpeMax * vwa < self.states.bpr else bpeMax * vwa
    tpALocal = self.states.bpr if tensorParametersA["bpe"] * vwa < self.states.bpr else tensorParametersA["bpe"] * vwa
    numVgprG2LAllocatedLocal = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] * \
        tpALocal) / (float)(self.states.bpr))
    if (self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]) and (bpeMax * vwa < self.states.bpr):
      # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
      # We should optimize the usage for better performance.
      statesANumVgprG2LAllocated = statesANumVgprG2L * (int)(self.states.bpr/(bpeMax * vwa))
    else:
      statesANumVgprG2LAllocated = statesANumVgprG2L
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      self.states.a.numVgprG2L = statesANumVgprG2L
      self.states.a.numVgprG2LAllocated = statesANumVgprG2LAllocated
      self.states.a.numVgprG2LTailloopAllocated = self.states.a.numVgprG2LAllocated
    else:
      self.states.a.numVgprG2L = 0
      self.states.a.numVgprG2LAllocated = 0
      self.states.a.numVgprG2LTailloopAllocated = statesANumVgprG2LAllocated
    # using _ds_store_b8: need one more vgpr space to do lshr
    if tensorParametersA["localWriteInstruction"].blockWidth == 0.25:
      self.states.a.numVgprG2L = self.states.a.numVgprG2L * 2
      self.states.a.numVgprG2LAllocated += numVgprG2LAllocatedLocal
      self.states.a.numVgprG2LTailloopAllocated += numVgprG2LAllocatedLocal
    # double numVgprG2L if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.states.a.numVgprG2L *= 2
      self.states.a.numVgprG2LAllocated *= 2
      bpeA = tensorParametersA["bpe"]
      bpeGRA = tensorParametersA["bpeGR"]
      if kernel["ConvertAfterDS"] and bpeA > bpeGRA:
        # DTV + covertAfterDS case, we need to allocate vgpr based on after conversion
        self.states.a.numVgprG2L *= (bpeA // bpeGRA)
        self.states.a.numVgprG2LAllocated *= (bpeA // bpeGRA)

    self.states.b.numVgprG2L = 0
    numVgprG2LAllocatedLocal = 0

    bpeMax = tensorParametersB["bpeDS"] if kernel["ConvertAfterDS"] else max(tensorParametersB["bpeGR"], tensorParametersB["bpe"])
    statesBNumVgprG2L = roundUp((kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"] * \
        kernel["GlobalReadVectorWidthB"] * bpeMax) / (float)(self.states.bpr))
    tpB      = self.states.bpr if bpeMax * vwb < self.states.bpr else bpeMax * vwb
    tpBLocal = self.states.bpr if tensorParametersB["bpe"] * vwb < self.states.bpr else tensorParametersB["bpe"] * vwb
    numVgprG2LAllocatedLocal = roundUp((kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"] * \
        tpBLocal) / (float)(self.states.bpr))
    if (self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]) and (bpeMax * vwb < self.states.bpr):
      # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
      # We should optimize the usage for better performance.
      statesBNumVgprG2LAllocated = statesBNumVgprG2L * (int)(self.states.bpr/(bpeMax * vwb))
    else:
      statesBNumVgprG2LAllocated = statesBNumVgprG2L
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      self.states.b.numVgprG2L = statesBNumVgprG2L
      self.states.b.numVgprG2LAllocated = statesBNumVgprG2LAllocated
      self.states.b.numVgprG2LTailloopAllocated = self.states.b.numVgprG2LAllocated
    else:
      self.states.b.numVgprG2L = 0
      self.states.b.numVgprG2LAllocated = 0
      self.states.b.numVgprG2LTailloopAllocated = statesBNumVgprG2LAllocated
    # using _ds_store_b8: need one more vgpr space to do lshr
    if tensorParametersB["localWriteInstruction"].blockWidth == 0.25:
      self.states.b.numVgprG2L = self.states.b.numVgprG2L * 2
      self.states.b.numVgprG2LAllocated += numVgprG2LAllocatedLocal
      self.states.b.numVgprG2LTailloopAllocated += numVgprG2LAllocatedLocal
    # double numVgprG2L if DirectToVgpr is enabled
    if kernel["DirectToVgprB"]:
      self.states.b.numVgprG2L *= 2
      self.states.b.numVgprG2LAllocated *= 2
      bpeB = tensorParametersB["bpe"]
      bpeGRB = tensorParametersB["bpeGR"]
      if kernel["ConvertAfterDS"] and bpeB > bpeGRB:
        # DTV + covertAfterDS case, we need to allocate vgpr based on after conversion
        self.states.b.numVgprG2L *= (bpeB // bpeGRB)
        self.states.b.numVgprG2LAllocated *= (bpeB // bpeGRB)

    self.states.m.numVgprG2L = 0
    if kernel["ProblemType"]["Sparse"]:
      if not kernel["DirectToVgprSparseMetadata"]:
        self.states.m.numVgprG2L = roundUp((kernel["NumLoadsCoalescedMetadata"] * kernel["NumLoadsPerpendicularMetadata"] * \
          kernel["GlobalReadVectorWidthMetadata"] * tensorParametersM["bpeDS"]) / (float)(self.states.bpr))
        if self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]:
          tpM = self.states.bpr if tensorParametersM["bpeDS"] * vwm < self.states.bpr else tensorParametersM["bpeDS"] * vwm
          self.states.m.numVgprG2LAllocated = roundUp((kernel["NumLoadsCoalescedMetadata"] * kernel["NumLoadsPerpendicularMetadata"] * \
            tpM) / (float)(self.states.bpr))
          self.states.m.numVgprG2L = self.states.m.numVgprG2LAllocated
        # using _ds_store_b8: need one more vgpr space to do lshr
        if tensorParametersM["localWriteInstruction"].blockWidth == 0.25:
          self.states.m.numVgprG2L = self.states.m.numVgprG2L * 2
          self.states.m.numVgprG2LAllocated = self.states.m.numVgprG2LAllocated * 2
      else:
        self.states.m.numVgprG2LAllocated = 0
    ####################################
    # num vgprs: local read addresses
    self.states.a.numVgprLocalReadAddr = 1 * self.states.rpla
    self.states.b.numVgprLocalReadAddr = 1 * self.states.rpla
    self.states.m.numVgprLocalReadAddr = 1 * self.states.rpla
    self.states.a.numVgprLocalWriteAddr = 0 if kernel["LocalWriteUseSgprA"] else 1 * self.states.rpla
    self.states.b.numVgprLocalWriteAddr = 0 if kernel["LocalWriteUseSgprB"] else 1 * self.states.rpla
    self.states.m.numVgprLocalWriteAddr = 0 if kernel["ProblemType"]["Sparse"] and kernel["LocalWriteUseSgprMetadata"] else 1 * self.states.rpla
    self.states.a.numVgprLocalReadSwapAddr = 0
    self.states.b.numVgprLocalReadSwapAddr = 0
    self.states.m.numVgprLocalReadSwapAddr = 0
    self.states.a.numVgprLocalWriteSwapAddr = 0
    self.states.b.numVgprLocalWriteSwapAddr = 0
    self.states.m.numVgprLocalWriteSwapAddr = 0
    self.states.a.numVgprLocalWriteAddrTailLoop = 0 if not (kernel["DirectToLdsA"] and kernel["NonDTLTailLoopA"]) else 1 * self.states.rpla
    self.states.b.numVgprLocalWriteAddrTailLoop = 0 if not (kernel["DirectToLdsB"] and kernel["NonDTLTailLoopB"]) else 1 * self.states.rpla

    numVgprMultiplierA = 1
    numVgprMultiplierB = 1
    numVgprMultiplierMetadata = 1
    maxLDSConstOffset = self.states.regCaps["maxLDSConstOffset"]

    if self.states.archCaps["DeviceLDS"] > maxLDSConstOffset:
      hasMultipleBuffer = kernel["ExpandPointerSwap"] and not kernel["1LDSBuffer"] and not kernel["StoreSwapAddr"]

      numVgprMultiplier = 1 if not hasMultipleBuffer else (kernel["LdsOffsetA_Blk"] // maxLDSConstOffset + 1)

      numVgprMultiplierA = max(numVgprMultiplier, kernel["LdsNumElementsAlignedA"] // maxLDSConstOffset + 1)
      numVgprMultiplierB = max(numVgprMultiplier, kernel["LdsNumElementsAlignedB"] // maxLDSConstOffset + 1)
      numVgprMultiplierMetadata = max(numVgprMultiplier, kernel["LdsNumElementsAlignedMetadata"] // maxLDSConstOffset + 1)


    self.states.a.numVgprLocalReadAddr *= numVgprMultiplierA
    self.states.a.numVgprLocalWriteAddr *= numVgprMultiplierA
    self.states.a.numVgprLocalWriteAddrTailLoop *= numVgprMultiplierA

    self.states.b.numVgprLocalReadAddr *= numVgprMultiplierB
    self.states.b.numVgprLocalWriteAddr *= numVgprMultiplierB
    self.states.b.numVgprLocalWriteAddrTailLoop *= numVgprMultiplierB

    self.states.m.numVgprLocalReadAddr *= numVgprMultiplierMetadata
    self.states.m.numVgprLocalWriteAddr *= numVgprMultiplierMetadata


    if not (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]):
      self.states.m.numVgprLocalReadAddr = 0
    # do not allocate local read address register if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.states.a.numVgprLocalReadAddr = 0
    if kernel["DirectToVgprB"]:
      self.states.b.numVgprLocalReadAddr = 0

    if not (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]):
      self.states.m.numVgprLocalWriteAddr = 0
    # do not allocate local write address register if DirectToVgpr is enabled
    if kernel["DirectToVgprA"] or kernel["DirectToLdsA"]:
      self.states.a.numVgprLocalWriteAddr = 0
    if kernel["DirectToVgprB"] or kernel["DirectToLdsB"]:
      self.states.b.numVgprLocalWriteAddr = 0

    if kernel["StoreSwapAddr"]:
      if self.states.a.numVgprLocalReadAddr > 0:
        self.states.a.numVgprLocalReadSwapAddr = 1
      if self.states.b.numVgprLocalReadAddr > 0:
        self.states.b.numVgprLocalReadSwapAddr = 1
      if self.states.m.numVgprLocalReadAddr > 0:
        self.states.m.numVgprLocalReadSwapAddr = 1
      if not kernel["LocalWriteUseSgprA"] and self.states.a.numVgprLocalWriteAddr > 0:
        self.states.a.numVgprLocalWriteSwapAddr = 1
      if not kernel["LocalWriteUseSgprB"] and self.states.b.numVgprLocalWriteAddr > 0:
        self.states.b.numVgprLocalWriteSwapAddr = 1
      if kernel["ProblemType"]["Sparse"] and not kernel["LocalWriteUseSgprMetadata"] and self.states.m.numVgprLocalWriteAddr > 0:
        self.states.m.numVgprLocalWriteSwapAddr = 1

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalReadVectorWidthA"]
    numGlobalReadInstructionsA = (numGlobalReadsA * tensorParametersA["bpeGR"])//\
        (tensorParametersA["globalReadInstruction"].blockWidth * 4)

    if kernel["BufferLoad"]:
      self.states.a.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsA * self.states.rpgo)
    else:
      numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.states.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalReadVectorWidthB"]
    numGlobalReadInstructionsB = (numGlobalReadsB * tensorParametersB["bpeGR"])// \
        (tensorParametersB["globalReadInstruction"].blockWidth * 4)
    if kernel["BufferLoad"]:
      self.states.b.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsB * self.states.rpgo)
    else:
      numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.states.rpga

    if self.states.globalReadIncsUseVgpr:
      numVgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.states.rpga
      numVgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.states.rpga
    else:
      numVgprGlobalReadIncsA = 0
      numVgprGlobalReadIncsB = 0

    if tensorParametersM is not None:
      numGlobalReadsMetadata = kernel["NumLoadsCoalescedMetadata"] \
          * kernel["NumLoadsPerpendicularMetadata"] * kernel["GlobalReadVectorWidthMetadata"]
      numGlobalReadInstructionsMetadata = (numGlobalReadsMetadata * tensorParametersM["bpe"])//\
          (tensorParametersM["globalReadInstruction"].blockWidth * 4)
      if kernel["BufferLoad"]:
        self.states.m.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsMetadata * self.states.rpgo)
      if self.states.globalReadIncsUseVgpr:
        numVgprGlobalReadIncsMetadata = kernel["ProblemType"]["NumIndicesSummation"] \
            * self.states.rpga
      else:
        numVgprGlobalReadIncsMetadata = 0

    numVgprAddressDbg = self.states.rpga if self.debugConfig.debugKernel else 0

    ####################################
    # num vgprs: c write address
    # 1 address where to write first value
    # 1 tmp address where to write current value

    ####################################
    # VGPR Assignment
    ####################################
    vgprIdx = 0
    self.states.totalAgprs      = 0
    self.states.totalMixedAgprs = 0
    self.states.maxLimitAgprs   = self.states.regCaps["PhysicalMaxVgpr"] - self.states.regCaps["MaxVgpr"]
    self.states.c.startVgprValu = vgprIdx; vgprIdx += self.states.c.numVgprValu

    if kernel["EnableMatrixInstruction"]:
      # MI kernels can overlap C-tile w/ AB-tile up until writeback. Illustrated below:
      # |<-------------- valuC -------------->|<-->|
      # |------------|-----------|------------|----|
      #   lastValuAB ^           ^            ^    ^  (ValuA, ValuB)
      #         lastVgprForReads ^            ^    ^  (localWriteAddr, globalReadOffser, G2L, localReadAddr)
      #                             lastValuC ^    ^  (ValuC)
      #                               vgprForStore ^  (other vgpr used in store section)
      self.states.serializedStore = True

      ########################################
      # AGPR Allocation
      ########################################
      if not kernel["MIArchVgpr"]:
        self.states.totalAgprs = self.states.c.numVgprValu
        if self.states.totalAgprs > self.states.maxLimitAgprs:
          self.states.totalMixedAgprs = self.states.totalAgprs - self.states.maxLimitAgprs
          self.states.totalAgprs      = self.states.maxLimitAgprs
        vgprIdx = self.states.totalMixedAgprs
        self.states.c.numVgprValu = self.states.totalMixedAgprs

    #----------------------------------
    # Move to the front and bypass to tail loop
    self.states.startVgprMisc = vgprIdx

    # BufferLoad:
    # Uses a resource descriptor (SRD) which is stored in 4 SGPRs and thus shared by all work-items.
    # Each work-item also uses  a unique 32-bit offset into vgprGlobalReadOffset.  These offsets are set when
    # the tile is initialized and stay constant through the execution of the kernel.
    # The base address in the SRD is updated when the algorithm moves to a new tile
    # BufferLoad disables the gptGlobalReadAddr used in flat addressing.
    if kernel["BufferLoad"]:
      self.startVgprGlobalReadOffsetA = vgprIdx
      vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.states.a.numVgprGlobalReadOffsets
      self.startVgprGlobalReadOffsetB = vgprIdx
      vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.states.b.numVgprGlobalReadOffsets
      if kernel["ProblemType"]["Sparse"]:
        self.startVgprGlobalReadOffsetMetadata = vgprIdx
        if kernel["DirectToVgprSparseMetadata"]:
          miWaveTile = kernel["MIWaveTileB"] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTileA"]
          vgprIdx += miWaveTile
        else:
          vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.states.m.numVgprGlobalReadOffsets
    else:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.startVgprGlobalReadAddressesA = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesA
      self.startVgprGlobalReadAddressesB = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesB

    if not kernel["LocalWriteUseSgprA"]:
      self.states.a.startVgprLocalWriteAddr = vgprIdx
      vgprIdx += self.states.a.numVgprLocalWriteAddr

    if not kernel["LocalWriteUseSgprB"]:
      self.states.b.startVgprLocalWriteAddr = vgprIdx
      vgprIdx += self.states.b.numVgprLocalWriteAddr

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      if self.states.combineLocalAddresses:
        self.states.m.startVgprLocalWriteAddr = self.states.m.startVgprLocalReadAddr
      else:
        self.states.m.startVgprLocalWriteAddr = vgprIdx
        vgprIdx += self.states.m.numVgprLocalWriteAddr

    self.startVgprGlobalReadIncsA = vgprIdx
    vgprIdx += numVgprGlobalReadIncsA
    self.startVgprGlobalReadIncsB = vgprIdx
    vgprIdx += numVgprGlobalReadIncsB
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      self.startVgprGlobalReadIncsMetadata = vgprIdx
      vgprIdx += numVgprGlobalReadIncsMetadata

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      self.states.m.startVgprLocalReadAddr = vgprIdx
      vgprIdx += self.states.m.numVgprLocalReadAddr
    self.states.a.startVgprLocalReadAddr = vgprIdx
    vgprIdx += self.states.a.numVgprLocalReadAddr
    self.states.b.startVgprLocalReadAddr = vgprIdx
    vgprIdx += self.states.b.numVgprLocalReadAddr

    # ----------------------------
    # TODO: alignment hack, figure out a better solution
    boolMoveLocalReadAddrA = False
    boolMoveLocalReadAddrB = False
    if (vgprIdx % 2) == 1:
      if (self.states.a.numVgprLocalReadAddr % 2) == 1:
        boolMoveLocalReadAddrA = True
        self.states.b.startVgprLocalReadAddr -= self.states.a.numVgprLocalReadAddr
        vgprIdx -= self.states.a.numVgprLocalReadAddr
      elif (self.states.b.numVgprLocalReadAddr % 2) == 1:
        boolMoveLocalReadAddrB = True
        vgprIdx -= self.states.b.numVgprLocalReadAddr

    # ----------------------------
    # TODO: alignment hack, figure out a better solution
    vgprIdx = ((vgprIdx+1)//2)*2
    # Avoid bank conflict between VgprA and VgprC
    if(self.states.archCaps["VgprBank"]):
      if (self.states.c.startVgprValu % 4) != (vgprIdx % 4):
        vgprIdx += 2
    # dot2: alignment hack for wider local read
    if kernel["UseDotInstruction"] and kernel["InnerUnroll"] > 1:
      vgprIdx = ((vgprIdx+3)//4)*4
    self.states.a.startVgprValu  = vgprIdx
    self.states.startVgpr        = vgprIdx
    vgprIdx += self.states.a.numVgprValu
    numVgprValuPackA = 0
    if tensorParametersA["bpe"] < 4 and not kernel["UnrollMajorLDSA"] and not kernel["enableLDSTrA"]:
      self.states.a.startVgprValuPack = vgprIdx
      if self.states.lrvwTileA > 1:
        numVgprValuPackA = ceil(kernel["VectorWidthA"] * tensorParametersA["bpe"] / self.states.bpr) * kernel["MIWaveTileA"] // kernel["VectorWidthA"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadA"]
        if self.states.packDTVA:
          # pack DTV case, double the number
          numVgprValuPackA *= 2
        elif (kernel["UsePLRPack"] and self.states.numItersPLR):
          numVgprValuPackA //= 2
      else:
        numVgprValuPackA = self.states.a.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackA * (int(4/tensorParametersA["bpeDS"]) - 1)
    vgprIdx += numVgprValuPackA
    self.states.a.startVgprG2L = None
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      # DirectToVgpr + pack or input conversion case, overlap G2L and ValuPack
      if self.states.packDTVA:
        self.states.a.startVgprG2L = self.states.a.startVgprValuPack
      elif self.states.convDTVA:
        self.states.a.startVgprG2L = self.states.a.startVgprValu
      # if PGR = True, PAP could be possibly enabled, we move G2LA later to prevent it from being reclaimed
      # otherwise, put G2L here since it can overlap valu
      if (not kernel["PrefetchGlobalRead"]): # g2l can overlap valu
        self.states.a.startVgprG2L = self.states.a.startVgprValu
        vgprIdx = self.states.a.startVgprValu  \
            + max(self.states.a.numVgprValu + numVgprValuPackA, self.states.a.numVgprG2LAllocated)

    # TODO: alignment hack, figure out a better solution
    if(self.states.archCaps["VgprBank"]):
      residual = (vgprIdx % 4)
      if (residual % 2) == 0:
        # if 2-aligned bank(bank0 and bank2), move to bank1 or bank3.
        vgprIdx += 1
    else:
      vgprIdx = ((vgprIdx+1)//2)*2
    self.states.b.startVgprValu  = vgprIdx
    vgprIdx += self.states.b.numVgprValu
    numVgprValuPackB = 0
    if tensorParametersB["bpe"] < 4 and not kernel["UnrollMajorLDSB"] and not kernel["enableLDSTrB"]:
      self.states.b.startVgprValuPack = vgprIdx
      if self.states.lrvwTileB > 1:
        numVgprValuPackB = ceil(kernel["VectorWidthB"] * tensorParametersB["bpe"] / self.states.bpr) * kernel["MIWaveTileB"] // kernel["VectorWidthB"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadB"]
        if self.states.packDTVB:
          # pack DTV case, double the number
          numVgprValuPackB *= 2
        elif (kernel["UsePLRPack"] and self.states.numItersPLR):
          numVgprValuPackB //= 2
      else:
        numVgprValuPackB = self.states.b.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackB * (int(4/tensorParametersB["bpeDS"]) - 1)
    vgprIdx += numVgprValuPackB
    self.states.b.startVgprG2L = None
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      # DirectToVgpr + pack  or input conversion case, overlap G2L and ValuPack
      if self.states.packDTVB:
        self.states.b.startVgprG2L = self.states.b.startVgprValuPack
      elif self.states.convDTVB:
        self.states.b.startVgprG2L = self.states.b.startVgprValu
      # if PGR = True, PAP could be possibly enabled, we move G2LB later to prevent it from being reclaimed
      # otherwise, put G2L here since it can overlap valu
      if (not kernel["PrefetchGlobalRead"]): # g2l can overlap valu
        self.states.b.startVgprG2L = self.states.b.startVgprValu
        vgprIdx = self.states.b.startVgprValu \
            + max(self.states.b.numVgprValu + numVgprValuPackB, self.states.b.numVgprG2LAllocated)

    if ((tensorParametersA["bpe"] < 4 and not kernel["UnrollMajorLDSA"]) or                                 \
        (tensorParametersB["bpe"] < 4 and not kernel["UnrollMajorLDSB"]) or                                 \
        (not kernel["UnrollMajorLDSMetadata"] and kernel["MIInputPerThreadMetadata"] == 4))                \
        and (kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()):
      self.states.a.startVgprValuPackTemp = vgprIdx
      self.states.b.startVgprValuPackTemp = vgprIdx
      vgprIdx += 1

    self.states.a.startVgprValuCvtTemp = -1
    self.states.b.startVgprValuCvtTemp = -1
    if kernel["ConvertAfterDS"]:
      if ((tensorParametersA["bpe"] > tensorParametersA["bpeDS"]) and kernel["ProblemType"]["DataTypeA"].is8bitFloat()):
        self.states.a.startVgprValuCvtTemp = vgprIdx
      if ((tensorParametersB["bpe"] > tensorParametersB["bpeDS"]) and kernel["ProblemType"]["DataTypeB"].is8bitFloat()):
        self.states.b.startVgprValuCvtTemp = vgprIdx
      if self.states.a.startVgprValuCvtTemp != -1 or self.states.b.startVgprValuCvtTemp != -1:
        vgprIdx += 2

    if kernel["ProblemType"]["Sparse"]:
      if kernel["DirectToVgprSparseMetadata"]:
        self.states.m.startVgprValu = vgprIdx
        vgprIdx += self.states.m.numVgprValu
      else:
        # TODO: alignment hack, figure out a better solution
        vgprIdx = ((vgprIdx+1)//2)*2
        if(self.states.archCaps["VgprBank"]):
          vgprIdx += 1
        self.states.m.startVgprValu  = vgprIdx
        vgprIdx += self.states.m.numVgprValu
        numVgprValuPackMetadata = 0
        if not kernel["UnrollMajorLDSMetadata"]:
          self.states.m.startVgprValuPack = vgprIdx
          if self.states.lrvwTileMetadata > 1:
            miWaveTile = kernel["MIWaveTileB"] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTileA"]
            numVgprValuPackMetadata = roundUp(kernel["VectorWidthMetadata"] * tensorParametersM["bpe"] / self.states.bpr) * miWaveTile // kernel["VectorWidthMetadata"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadMetadata"]
          else:
            numVgprValuPackMetadata = self.states.m.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackMetadata * (int(4/tensorParametersM["bpe"]) - 1)
        vgprIdx += numVgprValuPackMetadata
        self.states.m.startVgprG2L = None
        if not kernel["PrefetchGlobalRead"]: # g2l can overlap valu
          self.states.m.startVgprG2L = self.states.m.startVgprValu
          vgprIdx = self.states.m.startVgprValu  \
              + max(self.states.m.numVgprValu + numVgprValuPackMetadata, self.states.m.numVgprG2LAllocated)

    # Registers allocated above this point can be used as temps during setup
    # Registers above here are reserved in initC, near the end of the setup
    # code
    self.states.lastValuAB = vgprIdx

    #-----------
    self.states.firstVgprForReads = vgprIdx
    if self.states.a.startVgprG2L is None and self.states.a.numVgprG2LAllocated > 0:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.states.a.startVgprG2L = vgprIdx
      if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
        vgprIdx += self.states.a.numVgprG2LAllocated*2
      else:
        vgprIdx += self.states.a.numVgprG2LAllocated

    if self.states.b.startVgprG2L is None and self.states.b.numVgprG2LAllocated > 0:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.states.b.startVgprG2L = vgprIdx
      if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
        vgprIdx += self.states.b.numVgprG2LAllocated*2
      else:
        vgprIdx += self.states.b.numVgprG2LAllocated

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      if self.states.m.startVgprG2L is None:
        # TODO: alignment hack, figure out a better solution
        vgprIdx = ((vgprIdx+1)//2)*2
        self.states.m.startVgprG2L = vgprIdx; vgprIdx += self.states.m.numVgprG2LAllocated

    # GlobalRead, LocalWrite, LocalRead, G2L can be reclaimed, extend the "lastVgprForReads" value
    self.states.lastVgprForReads = vgprIdx

    if boolMoveLocalReadAddrA:
      self.states.a.startVgprLocalReadAddr = vgprIdx
      vgprIdx += self.states.a.numVgprLocalReadAddr
    elif boolMoveLocalReadAddrB:
      self.states.b.startVgprLocalReadAddr = vgprIdx
      vgprIdx += self.states.b.numVgprLocalReadAddr

    #-----------
    if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"]:
      if kernel["ProblemType"]["BiasSrc"] == "A":
        self.states.bias.numVgprValu = kernel["MIWaveTile"][0]
      elif kernel["ProblemType"]["BiasSrc"] == "B":
        self.states.bias.numVgprValu = kernel["MIWaveTile"][1]
      else:
        self.states.bias.numVgprValu = 0
      self.states.bias.numVgprValu *= max(kernel["ProblemType"]["ComputeDataType"].numRegisters(), 1)
    else:
      self.states.bias.numVgprValu = 0
    self.states.bias.startVgprValu = vgprIdx
    vgprIdx += self.states.bias.numVgprValu

    #-----------
    if kernel["ProblemType"]["OutputAmaxD"]:
      self.startVgprAmaxOut = vgprIdx
      self.startVgprAmaxOutB = vgprIdx + 1
      vgprIdx += 2

    self.states.startVgprAddressDbg = vgprIdx
    vgprIdx += numVgprAddressDbg

    # for zgemm + (SCIU or MIAV) case, allocate 4 vgpr for alpha calculation (cannot use tmp vgpr in unroll loop or write batch)
    if kernel["ProblemType"]["DataType"].isComplex() \
        and (kernel["StoreCInUnroll"] or kernel["MIArchVgpr"]) \
        and (kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
      # need proper alignment
      vgprIdx = ((vgprIdx+2 - 1)//2)*2
      self.states.startVgprAlphaTmp = vgprIdx
      vgprIdx += 4

    # for swapping vgpr offsets of different lds buffers
    if self.states.a.numVgprLocalReadSwapAddr > 0:
      self.states.a.startVgprLocalReadSwapAddr = vgprIdx
      vgprIdx += 1
    if self.states.m.numVgprLocalReadSwapAddr > 0:
      self.states.m.startVgprLocalReadSwapAddr = vgprIdx
      vgprIdx += 1
    if self.states.b.numVgprLocalReadSwapAddr > 0:
      self.states.b.startVgprLocalReadSwapAddr = vgprIdx
      vgprIdx += 1
    if self.states.a.numVgprLocalWriteSwapAddr > 0:
      self.states.a.startVgprLocalWriteSwapAddr = vgprIdx
      vgprIdx += 1
    if self.states.m.numVgprLocalWriteSwapAddr > 0:
      self.states.m.startVgprLocalWriteSwapAddr = vgprIdx
      vgprIdx += 1
    if self.states.b.numVgprLocalWriteSwapAddr > 0:
      self.states.b.startVgprLocalWriteSwapAddr = vgprIdx
      vgprIdx += 1

    # TODO: Serial is always the first/last register in the pool so the store
    # code doesn't have to deal with fragmentation
    self.states.startVgprSerial = vgprIdx
    vgprIdx += 1 # for vgpr serial id

    if kernel["UseDirect32XEmulation"]:
      numVgprsEmu = (self.states.a.numVgprValu + self.states.b.numVgprValu) // 2
      if (vgprIdx >= (self.states.regCaps["MaxVgpr"] - numVgprsEmu)):
        kernel["UseDirect32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = True
      else:
        #align 64 bit
        vgprIdx = ((vgprIdx+1)//2)*2
        self.states.startVgprCvt = vgprIdx
        vgprIdx += numVgprsEmu # for vgpr 32XEmulation

    self.states.totalVgprs = max(vgprIdx, self.states.c.numVgprValu)
    if self.states.totalVgprs < 0 or self.states.totalVgprs > self.states.regCaps["MaxVgpr"]:
      raise RuntimeError("Generating asm kernel error: total vgpr: %u not in [0, %u].\n" % (self.states.totalVgprs, self.states.regCaps["MaxVgpr"]))

    agprLimit = self.states.regCaps["PhysicalMaxVgpr"] - self.states.regCaps["MaxVgpr"]
    if self.states.totalAgprs > agprLimit:
      raise RuntimeError("Generating asm kernel error: total agpr: %u not in [0, %u].\n" % (self.states.totalAgprs, agprLimit) )

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    self.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=True, printRP=False)
    numSgprAddressD = self.states.rpga # til end
    numSgprAddressC = self.states.rpga # til end
    numSgprAddressA = self.states.rpga # til read offsets
    numSgprAddressB = self.states.rpga # til read offsets
    numSgprAddressWS = self.states.rpga
    numSgprAddressFlags = self.states.rpga

    numSgprAddressMetadata = self.states.rpga if kernel["ProblemType"]["Sparse"] else 0

    # would not less than 1 reg,
    # since even if ComputeType = H, we still pass the arg as a 32-bit (concate two 16-bit)
    numSgprAlpha = max(1,int(self.states.bpeCinternal/4))
    numSgprBeta  = max(1,int(self.states.bpeCinternal/4)) if kernel["ProblemType"]["UseBeta"] else 0
    self.states.e.numSgprStrides = kernel["ProblemType"]["NumIndicesC"]
    self.states.d.numSgprStrides = kernel["ProblemType"]["NumIndicesC"]
    self.states.c.numSgprStrides = kernel["ProblemType"]["NumIndicesC"]
    self.states.a.numSgprStrides = len(kernel["ProblemType"]["IndexAssignmentsA"])
    self.states.b.numSgprStrides = len(kernel["ProblemType"]["IndexAssignmentsB"])
    if not kernel["ProblemType"]["UseInitialStridesCD"]:
      self.states.e.numSgprStrides -= 1
      self.states.d.numSgprStrides -= 1
      self.states.c.numSgprStrides -= 1
    if not kernel["ProblemType"]["UseInitialStridesAB"]:
      self.states.a.numSgprStrides -= 1
      self.states.b.numSgprStrides -= 1
    if kernel["ProblemType"]["Sparse"]:
      self.states.m.numSgprStrides = len(kernel["ProblemType"]["IndexAssignmentsMetadata"])
      if not kernel["ProblemType"]["UseInitialStridesAB"]:
        self.states.m.numSgprStrides -= 1
    else:
      self.states.m.numSgprStrides = 0
    self.states.numSgprSizesSum = kernel["ProblemType"]["NumIndicesSummation"]
    self.states.numSgprSizesFree = kernel["ProblemType"]["NumIndicesC"]
    self.states.numActivationTypeArgSize = 0 # Will change to 1 if activationType == All
    self.states.numActivationArgSize = max(1, int(kernel["ProblemType"]["DestDataType"].numRegisters()))
    self.states.numactivationArgTotalSize = self.states.numActivationArgSize * kernel["ProblemType"]["ActivationType"].getAdditionalArgNum()
    self.states.numSgprAddressDbg = self.states.rpga if self.debugConfig.debugKernel else 0

    ####################################
    # num sgprs: global read increments
    if self.states.globalReadIncsUseVgpr:
      self.states.a.numSgprGlobalReadIncs = 0
      self.states.b.numSgprGlobalReadIncs = 0
      self.states.m.numSgprGlobalReadIncs = 0
    else:
      self.states.a.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo
      self.states.b.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo
      self.states.m.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo

    ########################################
    # SGPR Assignment according to AMDGPU-ABI
    ########################################
    self.defineSgpr("KernArgAddress", self.states.rpga)
    assert(self.sgprs["KernArgAddress"] ==  0) # kernarg is passed to kernel as SGPR0

    self.defineSgpr("WorkGroup0", 1)
    self.defineSgpr("WorkGroup1", 1)

    wg=2

    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        self.defineSgpr("WorkGroup%u"%wg, 1)
        wg+=1

    # SGPR above are user SGPR which are set by GPU hardware when the kernel is launched
    self.states.firstInitSgpr = self.sgprPool.size()

    if kernel["ProblemType"]["SupportUserArgs"]:
      self.defineSgpr("ArgType", 1)  # 0: normal, 1: hbm, 2: user args

    # To avoid corrupting tmp sgprs that may be used around the assert,
    # reserve some sgprs to save/restore the execmask
    if self.db["EnableAsserts"]:
      self.defineSgpr("SaveExecMask", 2, 2)

    if kernel["GlobalSplitU"] != 0:
      self.defineSgpr("GSUSumIdx", 2, 2)
      self.defineSgpr("GSULog2BpeC", 1)
      self.defineSgpr("GSULog2BpeD", 1)
    self.defineSgpr("StaggerU", 1)
    self.defineSgpr("WGM", 1)

    if kernel["LocalSplitU"] > 1:
      self.defineSgpr("LSUTailLoopOffset", 1)

    # for packed batches without stride restrictions need to do something different here
    assert sorted(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]) == \
           sorted(set(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]))
    for idxChar in kernel["PackedC0IdxChars"][:-1]:
      if kernel["MagicDivAlg"]==2:
        self.defineSgpr("MagicAbitSize%s"%idxChar, 1)
    for idxChar in kernel["PackedC1IdxChars"][:-1]:
      if kernel["MagicDivAlg"]==2:
        self.defineSgpr("MagicAbitSize%s"%idxChar, 1)

    # product of all packed dims in the 0 or 1 dimensions:
    if len(kernel["PackedC0IndicesX"]) > 1:
      self.defineSgpr("PackedSize0", 1)
    if len(kernel["PackedC1IndicesX"]) > 1:
      self.defineSgpr("PackedSize1", 1)

    # contractions with multiple summations will use multiple LoopCounters, if PSD=0
    for i in range(kernel["ProblemType"]["NumIndicesSummation"]):
      self.defineSgpr(self.loopCounterName(kernel,i), 1)

    self.defineSgpr("OrigLoopCounter", 1)

    if self.debugConfig.debugKernel:
      self.defineSgpr("AddressDbg", self.states.numSgprAddressDbg)
      self.defineSgpr("DebugKernelItems", 1)

    # the sgprs overlap with wg ids
    if self.states.doShadowInit and kernel["BufferStore"]:
      self.defineSgpr("SrdD", 4, 4)
      self.defineSgpr("SrdC", 4, 4)

    self.defineSgpr("NumWorkGroups0", 1)
    self.defineSgpr("NumWorkGroups1", 1)

    if kernel["BufferStore"] and (kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel'):
      self.defineSgpr("WSDstart", 2, 2)

    # Calculate numSgpr preload
    self.states.preloadGuard = []
    self.states.numSgprPreload = 0
    if kernel["PreloadKernArgs"]:
      # Max num spgrs can be setup by CP is only 16 for now
      # kernel argument buffer address needs 2 sgprs
      # Workgroup ID x, y, z need 3 sgprs
      numWorkgroupIDSgpr = kernel["ProblemType"]["NumIndicesC"]
      self.states.numSgprPreload = 16 - self.states.rpga - kernel["ProblemType"]["NumIndicesC"]

      # Safe guard for preload arguments
      while(1):
        tmpSgpr = self.sgprPool.checkOut(1, preventOverflow=False)
        if tmpSgpr >= 16:
          self.sgprPool.checkIn(tmpSgpr)
          break
        self.states.preloadGuard.append(tmpSgpr)

    ###################################
    # Get kernel argument start here
    ###################################
    # get aligned Sgpr index for wider s_load
    self.defineSgpr("SizesFree", self.states.numSgprSizesFree,4)
    # fill empty Sgpr slot caused by Sgpr alignment,
    # because we need following defineSgpr use continuous sgpr
    SgprSlot = []
    currentSize = self.sgprPool.size()
    while (1):
      tempSgpr = self.sgprPool.checkOut(1,"fill empty slot temporarily",preventOverflow=False)
      if tempSgpr >= currentSize:
        self.sgprPool.checkIn(tempSgpr)
        break
      SgprSlot.append(tempSgpr)
    self.defineSgpr("SizesSum", self.states.numSgprSizesSum)
    self.defineSgpr("AddressD", numSgprAddressD)
    self.defineSgpr("AddressC", numSgprAddressC)
    self.defineSgpr("AddressA", numSgprAddressA)
    self.defineSgpr("AddressB", numSgprAddressB)
    if kernel["ProblemType"]["Sparse"]:
      self.defineSgpr("AddressMetadata", numSgprAddressMetadata)
    if kernel["StreamK"] > 0 and kernel["StreamKAtomic"] == 0:
      self.defineSgpr("AddressWS", numSgprAddressWS)
      self.defineSgpr("AddressFlags", numSgprAddressFlags)
      self.states.numSgprStreamK += numSgprAddressWS + numSgprAddressFlags

    #asm input interface depen
    self.defineSgpr("StridesD", self.states.d.numSgprStrides)
    self.defineSgpr("StridesC", self.states.c.numSgprStrides)
    self.defineSgpr("StridesA", self.states.a.numSgprStrides)
    self.defineSgpr("StridesB", self.states.b.numSgprStrides)
    if kernel["ProblemType"]["Sparse"]:
      self.defineSgpr("StridesMetadata", self.states.m.numSgprStrides)

    # for packed batches without stride restrictions need to do something different here
    assert sorted(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]) == \
           sorted(set(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]))
    for idxChar in kernel["PackedC0IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1)
    for idxChar in kernel["PackedC1IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1)

    GSUAMBSK = 0
    if (kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel'):
      GSUAMBSK = 1

    self.defineSgpr("Alpha", numSgprAlpha, numSgprAlpha)
    self.states.numSgprAlpha = numSgprAlpha
    if kernel["ProblemType"]["UseBeta"]:
      self.defineSgpr("Beta", numSgprBeta, numSgprBeta)
      self.states.numSgprBeta = numSgprBeta

    if kernel["StreamK"]:
      # StreamK args
      self.defineSgpr("ItersPerTile", 1)
      self.defineSgpr("MagicNumberItersPerTile", 1)
      self.defineSgpr("MagicShiftItersPerTile", 1)
      self.defineSgpr("TotalIters", 1)
      self.defineSgpr("SKItersPerWG", 1)
      self.states.numSgprStreamK += 5
      if kernel["StreamK"] >= 2: # Two-tile SK
        self.defineSgpr("skGrid", 1)
        self.defineSgpr("skTiles", 1)
        self.states.numSgprStreamK += 2

    if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("LocalWriteAddrA", 1)
    if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("LocalWriteAddrB", 1)

    # Allocate registers to swap between lds buffers
    if kernel["StoreSwapAddr"]:
      if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("SwapA", 1)
      if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("SwapB", 1)
      if kernel["ProblemType"]["Sparse"] and kernel["LocalWriteUseSgprMetadata"]:
        self.defineSgpr("SwapMetadata", 1)


    if GSUAMBSK:
      self.defineSgpr("AddressTD", numSgprAddressD, align=2)
      self.states.numSgprAddressGSUSync += numSgprAddressD
      self.defineSgpr("Synchronizer", 2, align=2)
      self.states.numSgprAddressGSUSync += 2
      self.defineSgpr("GSUSync", 1)
      self.states.numSgprAddressGSUSync += 1

    if kernel["GlobalSplitU"] != 0:
      self.defineSgpr("GSU", 1)  # Can't move to the front because of the preload arguments

    if kernel["StreamK"]:
      # StreamK vars
      self.defineSgpr("StreamKIdx", 1)
      self.defineSgpr("StreamKIter", 1)
      self.defineSgpr("StreamKIterEnd", 1)
      self.defineSgpr("StreamKLocalStart", 1)
      self.defineSgpr("StreamKLocalEnd", 1)
      if len(kernel["SpaceFillingAlgo"]):
        self.defineSgpr("StreamKTileID", 1)
      if kernel["StreamKAtomic"] == 0:
        self.defineSgpr("SrdWS", 4, 4)

    # These SGPRs aren't used right away, add them to spr pool temporarily
    if self.states.doShadowInit and kernel["BufferStore"]:
      self.addSgprVarToPool("SrdC")
    if kernel["StreamK"] and kernel["StreamKAtomic"] == 0:
      self.addSgprVarToPool("SrdWS")

    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    for key, _ in self.sgprs.items():
      self.states.nonPostLoopSgpr.append(key)
    # Manually remove some additional unused sgpr
    for i in range(kernel["ProblemType"]["NumIndicesSummation"]):
      self.states.nonPostLoopSgpr.remove(self.loopCounterName(kernel,i))
    self.states.nonPostLoopSgpr.remove("OrigLoopCounter")

    if not kernel["StreamK"]:
      # Persistent loop requires arguments to remain for next tile
      self.states.nonPostLoopSgpr.remove("WGM")
      self.states.nonPostLoopSgpr.remove("AddressA")
      self.states.nonPostLoopSgpr.remove("AddressB")
      self.states.nonPostLoopSgpr.remove("StridesA")
      self.states.nonPostLoopSgpr.remove("StridesB")

    self.states.preloadScaleA = False
    self.states.preloadScaleB = False
    if kernel["ProblemType"]["UseScaleAB"] == "Scalar":
      if kernel["ProblemType"]["DataTypeA"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters():
        self.states.preloadScaleA = True
      if kernel["ProblemType"]["DataTypeB"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters():
        self.states.preloadScaleB = True

      for preloadScale, name in zip([self.states.preloadScaleA, self.states.preloadScaleB], ['A','B']):
        if preloadScale:
          self.defineSgpr("AddressScale%s"%name, 2, 2)
          self.defineSgpr("Scale%s"%name, numSgprAlpha, numSgprAlpha if numSgprAlpha > 1 else 2)


    self.states.numSgprToLoad = self.states.numSgprSizesFree + self.states.numSgprSizesSum + \
      numSgprAddressD + numSgprAddressC + numSgprAddressA + numSgprAddressB + numSgprAlpha + numSgprAddressMetadata + \
      (numSgprBeta if kernel["ProblemType"]["UseBeta"] else 0) + \
      self.states.d.numSgprStrides + self.states.c.numSgprStrides + self.states.a.numSgprStrides + self.states.b.numSgprStrides + self.states.m.numSgprStrides + \
      self.states.numSgprStreamK + \
      len(kernel["PackedC0IdxChars"][:-1])*2 + len(kernel["PackedC1IdxChars"][:-1])*2
    # Get kernel argument end here
    ###################################

    # put unused Sgpr back to SgprPool
    while SgprSlot:
      tempSgpr = SgprSlot.pop(0)
      self.sgprPool.checkIn(tempSgpr)

    if self.sgprPool.size() > self.states.regCaps["MaxSgpr"]:
      print ("warning: Number of first half of defined SGPRS (%d) overflowed max SGPRS (%d)." \
               % (self.sgprPool.size(), self.states.regCaps["MaxSgpr"]))

    ########################################
    # Register Pools
    ########################################
    #print "TotalVgprs", self.states.totalVgprs
    self.vgprPool = RegisterPool(self.states.totalVgprs, RegisterType.Vgpr, defaultPreventOverflow=False,
                                 printRP=self.db["PrintRP"])
    self.savedVgprPool = None
    self.savedSgprPool = None

    ## accumulator Buffer for storeCinUnroll feature
    self.agprPool = RegisterPool(self.states.totalAgprs, RegisterType.Accvgpr, defaultPreventOverflow=False, printRP=False)

    ########################################
    # reads Per Iteration
    ########################################
    if kernel["EnableMatrixInstruction"]:
      if kernel["UnrollMajorLDSA"]:
        self.states.numReadsPerUnrollA = ceil(tensorParametersA["bpe"] * kernel["MIInputPerThreadA"] / int(tensorParametersA["localReadInstruction"].blockWidth * 4))
      else:
        self.states.numReadsPerUnrollA = kernel["MIInputPerThreadA"]
      numA = kernel["InnerUnroll"]*(kernel["MIWaveTile"][0] * self.states.numReadsPerUnrollA) // tensorParametersA["localReadInstruction"].numOffsets
      if self.states.lrvwTileA > 1:
        numA = numA // kernel["VectorWidthA"]

      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        if kernel["UnrollMajorLDSMetadata"]:
          self.states.numReadsPerUnrollMetadata = ceil(tensorParametersM["bpe"] * kernel["MIInputPerThreadMetadata"] / int(tensorParametersM["localReadInstruction"].blockWidth * 4))
        else:
          self.states.numReadsPerUnrollMetadata = kernel["MIInputPerThreadMetadata"]
        tileM = kernel["MIWaveTile"][1] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTile"][0]
        numM = kernel["InnerUnroll"]*(tileM * self.states.numReadsPerUnrollMetadata) // tensorParametersM["localReadInstruction"].numOffsets

      if kernel["UnrollMajorLDSB"]:
        self.states.numReadsPerUnrollB = ceil(tensorParametersB["bpe"] * kernel["MIInputPerThreadB"] / int(tensorParametersB["localReadInstruction"].blockWidth * 4))
      else:
        self.states.numReadsPerUnrollB = kernel["MIInputPerThreadB"]
      numB = kernel["InnerUnroll"]*(kernel["MIWaveTile"][1] * self.states.numReadsPerUnrollB) // tensorParametersB["localReadInstruction"].numOffsets
      if self.states.lrvwTileB > 1:
        numB = numB // kernel["VectorWidthB"]

      # wider localread has 2 mode
      # 1. using larger IU to coalesced localread, only half of local reads in 1 iteration
      # 2. using larger PLR to read more iterations, same number local reads in 1 iteration
      if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedA:
        numA //= self.states.numReadsIterCoalescedA
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedMetadata:
          numM //= self.states.numReadsIterCoalescedMetadata
      if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedB:
        numB //= self.states.numReadsIterCoalescedB

    else: # mac instruction
      if kernel["UseDotInstruction"]:
        # dot2: InnerUnroll are used for wider local read
        numA = kernel["ThreadTile0"] // tensorParametersA["localReadInstruction"].numOffsets
        numB = kernel["ThreadTile1"] // tensorParametersB["localReadInstruction"].numOffsets
      else:
        numA = kernel["InnerUnroll"]*(kernel["ThreadTile0"] // kernel["VectorWidthA"]) // tensorParametersA["localReadInstruction"].numOffsets
        numB = kernel["InnerUnroll"]*(kernel["ThreadTile1"] // kernel["VectorWidthB"]) // tensorParametersB["localReadInstruction"].numOffsets

    if not kernel["DirectToVgprA"]:
      self.states.numReadsPerIterA = numA
    if not kernel["DirectToVgprB"]:
      self.states.numReadsPerIterB = numB
    self.states.localReadDoCntA   = 0
    self.states.localReadDoCntB   = 0
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      self.states.numReadsPerIterMetadata = numM
      self.states.localReadDoCntMetadata  = 0

    if kernel["EnableMatrixInstruction"]:
      from rocisa.instruction import getMFMAIssueLatency, getSMFMAIssueLatency
      self.states.miLatency, miIssueLatency = getSMFMAIssueLatency(kernel["ProblemType"]["DataType"].toEnum(), kernel["MatrixInstM"], kernel["MatrixInstB"]) if kernel["ProblemType"]["Sparse"] else \
                                              getMFMAIssueLatency(kernel["ProblemType"]["DataType"].toEnum(), kernel["MatrixInstM"], kernel["MatrixInstB"])

      # give 1 quad-cycle buffer to prevend bubble from sync
      miLatencyBuffer = 1
      self.states.miLatencyLeft = max(self.states.miLatency - miLatencyBuffer - miIssueLatency,0)

      # Special dependency cases
      if kernel["ProblemType"]["ComputeDataType"].isDouble():
        if kernel["MatrixInstruction"] == [4, 4, 4, 4]:
          if kernel['ISA'] == IsaVersion(9,0,10):
            self.states.miDependency = 4


    # shift vectors determined later

    canCheckValueC = (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and \
                      kernel["ProblemType"]["HighPrecisionAccumulate"]
    canCheckValueC = canCheckValueC or kernel["ProblemType"]["DataType"].isSingle()
    canCheckValueC = canCheckValueC or (kernel["ProblemType"]["DataType"].isInt8() and kernel["ProblemType"]["HighPrecisionAccumulate"])
    assert not self.db["CheckValueC"] or canCheckValueC

    # Epilogue related
    self.states.useBias = DataDirection.NONE
    self.states.needBiasType = False
    if kernel["ProblemType"]["UseBias"]:
      if kernel["ProblemType"]["Gradient"]:
        if kernel["ProblemType"]["BiasSrc"] == "D":
          self.states.useBias = DataDirection.WRITE
        elif kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B":
          self.states.useBias = DataDirection.WRITE
      else:
        self.states.useBias = DataDirection.READ
      # Need bias type if the kernel supports multiple bias type.
    if self.states.useBias == DataDirection.READ or (self.states.useBias == DataDirection.WRITE and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B")):
      self.states.needBiasType = True
    else:
      self.states.needBiasType = False

    #########################################################
    # Below calculates the number of sgprs needed in epilogue
    #########################################################
    self.states.numStoreSgprNames = []
    self.states.numStoreSgprNameSizes = []
    storeSgprLoad = 0
    enableFactorDim = False;
    if kernel["ProblemType"]["UseScaleAB"]:
      self.states.numSgprAddressScaleA = self.states.rpga if (not self.states.preloadScaleA) else 0
      self.states.numSgprAddressScaleB = self.states.rpga if (not self.states.preloadScaleB) else 0
      storeSgprLoad += self.states.numSgprAddressScaleA + self.states.numSgprAddressScaleB
      if self.states.numSgprAddressScaleA:
        self.states.numStoreSgprNames.append("AddressScaleA")
        self.states.numStoreSgprNameSizes.append(self.states.numSgprAddressScaleA)
      if self.states.numSgprAddressScaleB:
        self.states.numStoreSgprNames.append("AddressScaleB")
        self.states.numStoreSgprNameSizes.append(self.states.numSgprAddressScaleB)
    if kernel["ProblemType"]["UseScaleCD"]:
      self.states.numSgprAddressScaleC = self.states.rpga
      self.states.numSgprAddressScaleD = self.states.rpga
      storeSgprLoad += self.states.numSgprAddressScaleC + self.states.numSgprAddressScaleD
      if self.states.numSgprAddressScaleC:
        self.states.numStoreSgprNames.append("AddressScaleC")
        self.states.numStoreSgprNameSizes.append(self.states.numSgprAddressScaleC)
      if self.states.numSgprAddressScaleD:
        self.states.numStoreSgprNames.append("AddressScaleD")
        self.states.numStoreSgprNameSizes.append(self.states.numSgprAddressScaleD)
    if kernel["ProblemType"]["UseScaleAlphaVec"]:
        storeSgprLoad += self.states.rpga
        self.states.numStoreSgprNames.append("AddressScaleAlphaVec")
        self.states.numStoreSgprNameSizes.append(self.states.rpga)
        self.states.FactorDim = max(self.states.FactorDim, kernel["ProblemType"]["UseScaleAlphaVec"])
        if self.states.FactorDim == 3:
          enableFactorDim = True
    if self.states.useBias != DataDirection.NONE:
      # Does not support atomic yet
      self.states.BiasType   = 0
      self.states.BiasStride = 0
      self.states.numSgprAddressBias = self.states.rpga # 64-bit
      self.states.numStoreSgprNames.append("AddressBias")
      self.states.numStoreSgprNameSizes.append(self.states.numSgprAddressBias)
      if self.states.needBiasType:
        self.states.BiasType   = 1
        self.states.BiasStride = 1
        self.states.numStoreSgprNames.append("BiasType")
        self.states.numStoreSgprNameSizes.append(self.states.BiasType)
        self.states.numStoreSgprNames.append("BiasStride")
        self.states.numStoreSgprNameSizes.append(self.states.BiasStride)
        self.states.FactorDim = max(self.states.FactorDim, kernel["ProblemType"]["UseBias"])
        if self.states.FactorDim == 3:
            enableFactorDim = True
      storeSgprLoad += self.states.numSgprAddressBias + self.states.BiasType + self.states.BiasStride

    if enableFactorDim:
      self.states.numStoreSgprNames.append("FactorDim")
      self.states.numStoreSgprNameSizes.append(1)
      storeSgprLoad += 1

    if kernel["ProblemType"]["UseE"]:
      storeSgprLoad += self.states.rpga + self.states.e.numSgprStrides
      self.states.numStoreSgprNames.append("AddressE")
      self.states.numStoreSgprNameSizes.append(self.states.rpga)
      self.states.numStoreSgprNames.append("StridesE")
      self.states.numStoreSgprNameSizes.append(self.states.e.numSgprStrides)
    runActivation = True if ((kernel["ProblemType"]["ActivationType"] != 'none') \
        and kernel["ActivationFused"]) else False
    if runActivation:
      for name in kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        self.states.numStoreSgprNames.append(name)
        self.states.numStoreSgprNameSizes.append(self.states.numActivationArgSize)
      if kernel["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
        self.states.numActivationTypeArgSize = 1
        self.states.numStoreSgprNames.append("ActivationType")
        self.states.numStoreSgprNameSizes.append(1)
      storeSgprLoad += self.states.numActivationTypeArgSize + self.states.numactivationArgTotalSize
    self.states.numStoreSgprToLoad = storeSgprLoad


    if self.db["InitLds"] : print ("\n***WARNING: InitLds enabled, may impact performance\n")
    if self.db["InitSgpr"] : print ("\n***WARNING: InitSgpr enabled, may impact performance\n")
    if self.db["InitVgpr"] : print ("\n***WARNING: InitVgpr enabled, may impact performance\n")
    if self.db["ConservativeWaitCnt"] : print ("\n***WARNING: ConservativeWaitCnt enabled, may impact performance\n")
    if self.do["KeepDirectToLdsAlloc"] : print ("\n***WARNING: KeepDirectToLdsAlloc enabled, may impact performance\n")
    if self.db["CheckValue1A"] : print ("\n***WARNING: CheckValue1A enabled, may impact performance\n")
    if self.db["CheckValue1B"] : print ("\n***WARNING: CheckValue1B enabled, may impact performance\n")
    if self.db["CheckValueC"] : print ("\n***WARNING: CheckValueC enabled, may impact performance\n")
    if self.db["ForceExpectedValue"] : print ("\n***WARNING: ForceExpectedValue enabled, may impact functionality\n")
    if self.db["ForceVSerial"] : print ("\n***WARNING: ForceVSerial enabled, will impact functionality\n")
    if self.db["ForceInputValueA"] : print ("\n***WARNING: ForceInputValueA enabled, may impact functionality\n")
    if self.db["ForceInputValueB"] : print ("\n***WARNING: ForceInputValueB enabled, may impact functionality\n")
    if self.db["CheckStoreC"] >=0  : print ("\n***WARNING: CheckStoreC enabled, may impact performance\n")
    if self.db["ForceEdgeStores"] : print ("\n***WARNING: ForceEdgeStores enabled, may impact performance\n")
    if self.db["AssertNoEdge"] : print ("\n***WARNING: AssertNoEdge enabled, may impact functionality and performance\n")
    if self.db["PrintRP"] : print ("\n***WARNING: PrintRP enabled, may generate verbose output\n")

  ##############################################################################
  # Function Signature
  ##############################################################################
  @abc.abstractmethod
  def functionSignature(self):
    return ""

  ##############################################################################
  # Local Read Addresses
  ##############################################################################
  @abc.abstractmethod
  def localReadAddresses(self, kernel, tPA, tPB, tPM):
    return ""

  ##############################################################################
  # Local Write Addresses
  ##############################################################################
  @abc.abstractmethod
  def localWriteAddresses(self, kernel, tPA, tPB, tPM):
    return ""

  ##############################################################################
  # Allocate Resources
  ##############################################################################
  @abc.abstractmethod
  def defineAndResources(self, kernel, tPA, tPB, tPM):
    return ""

  ##############################################################################
  # Allocate GR Address Resources
  ##############################################################################
  @abc.abstractmethod
  def removeGRSrdVariableSgprsFromPool(self, kernel):
    return ""

  ##############################################################################
  # Allocate GR Address Resources
  ##############################################################################
  @abc.abstractmethod
  def removeGROffsetsVariableSgprsFromPool(self, kernel):
    return ""

  ##############################################################################
  # Check Resources
  ##############################################################################
  @abc.abstractmethod
  def checkResources(self, kernel, mkb) -> None:
    pass

  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  @abc.abstractmethod
  def graWorkGroup(self, kernel):
    return ""

  ##############################################################################
  # Get Params For Tensor A/B
  ##############################################################################
  def getTensorParameters(self, tP, kernel, itP, cM):
    tP["mirror"] = bool(kernel["ProblemType"]["MirrorDims%s" % (cM)])

    if cM == "A" or (kernel["ProblemType"]["Sparse"] == 1 and cM == "Metadata"): # A
      tP["tensorIdx"] = 0                                   # tensor index A=0, B=1
      tP["tileChar"] = self.states.tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) \
        else self.states.tileChar1                       # tile char I0 or J1
    elif cM == "B" or (kernel["ProblemType"]["Sparse"] ==2 and cM == "Metadata"): # B
      tP["tensorIdx"] = 1
      tP["tileChar"] = self.states.tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) \
        else self.states.tileChar1

    tP["isA"] = (cM == "A")                                      # is this tensor A
    tP["isB"] = (cM == "B")                                      # is this tensor B
    tP["isM"] = (cM == "Metadata")                               # is this tensor Metadata

    bpe = int(kernel["ProblemType"]["DataType"].numBytes()) if not tP["isM"] else 1
    bpetc = int(kernel["ProblemType"]["DataType%s"%cM].numBytes()) if not tP["isM"] else 1
    bpeA = int(kernel["ProblemType"]["DataTypeA"].numBytes()) if not tP["isM"] else 1

    tP["bpe"] = bpe
    tP["bpeA"]  = (bpeA if kernel["ConvertAfterDS"] else bpe)
    tP["bpeGR"] = bpetc
    tP["bpeDS"] = (bpetc if kernel["ConvertAfterDS"] else bpe)
    tP["tensorChar"] = cM                                        # tensor character A/B
    tP["tileIdx"] = kernel["ProblemType"]["Index01%s"%cM]        # is the tile dimension of A the 0th or 1th index, i.e. Aki, tileIdx=0
    tP["tile01Idx"] = 1 if tP["tileIdx"] else 0
    tP["lsc"] = "LSC%s"%cM                                       # load size coalesced A/B, number of elements that get loaded along coalesced dimension with each load
    tP["lsp"] = "LSP%s"%cM                                       # load size perpendicular A/B, number of elements that get loaded along non-coalesced dimension with each load
    tP["lvc"] = "LVC%s"%cM                                       # "load size" in terms of number of short-vectors and not elements
    tP["lvp"] = "LVP%s"%cM                                       # "load size" in terms of number of short-vectors and not elements
    tP["rtv"] = itP[cM].readTileDimVector                        # bool in the tile dimension, reads will read vectors
    tP["wg"] = "WorkGroup%u" % (tP["tile01Idx"])                 # these are storing the actual strong to lookup the number from kernel dictionary
    tP["sg"] = "SubGroup%u" % (tP["tile01Idx"])
    tP["tt"] = "ThreadTile%u" % (tP["tile01Idx"])
    tP["mt"] = "MacroTile%u" % (tP["tile01Idx"])
    tP["tlu"] = kernel["ProblemType"]["TLU%s"%cM]                # thread stride is less than unroll stride, i.e., not transposing matrix
    tP["ia"] = kernel["ProblemType"]["IndexAssignments%s"%cM]    # array of index assignments
    tP["nrt"] = itP[cM].numReadsTile                             # number of reads along tile dimension
    tP["nru"] = itP[cM].numReadsUnroll                           # number of reads along unroll dimension
    tP["nrc"] = kernel["NumLoadsCoalesced%s"%cM]                 # number of reads along coalesced dimension
    tP["nrcv"] = itP[cM].numReadsCoalVecComp                     # number of vector components along coalesced dimension
    tP["nrp"] = kernel["NumLoadsPerpendicular%s"%cM]             # number of reads along perpendicular dimension
    tP["nrpv"] = itP[cM].numReadsPerpVecComp                     # number of vector components along perpendicular dimension
    tP["nwcv"] = itP[cM].numWritesCoalVecComp                    # number of vector component writes along coalesced dimension
    tP["nwpv"] = itP[cM].numWritesPerpVecComp                    # number of vector component writes along perpendicular dimension
    tP["glvw"] = kernel["GlobalReadVectorWidth%s"%cM]
    tP["wtc"] = itP[cM].writeTileDimComponents                   # write vector components along tile dimension
    tP["idx"] = kernel["ProblemType"]["Index%d"%tP["tensorIdx"]] # index 0 is tile dimension belonging to A. Note 'idx' may not be in tP['ia'].
    tP["NonTemporal"] = kernel["NonTemporal%s"%cM]               # non-temporal read type
    tP["shiftGR"] = 0 if (tP["bpeGR"] >= tP["bpeDS"]) else int(tP["glvw"] // 2 * (tP["bpeDS"] / self.states.bpr))  # Shift global read register for cvt spaces
    tP["bpeRatio"] = tP["bpeDS"] // tP["bpeGR"] if tP["bpeGR"] < tP["bpeDS"] else 1                                # g2lIdx multiplier

    tP["is_sparse"] = (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"])
    # KernelWriterAssembly
    tP["localReadSwapByteOffset"]  = 0
    tP["localWriteSwapByteOffset"] = 0
    tP["gpr"] = {}
    tP["metadataWriteSwapByteOffset"] = 0
    tP["isSwizzled"] = (kernel["ProblemType"]["SwizzleTensorB"] and tP["isB"]) or (kernel["ProblemType"]["SwizzleTensorA"] and tP["isA"])

    if tP["isSwizzled"]:
      # 16 means bytes of buffer_load_dwordx4
      tP["swizzlePackK"] = 16 // kernel["MIInputPerThread%s"%cM] // kernel["ProblemType"]["DataType%s"%cM].numBytes()
      tP["swizzleK"] = kernel["MatrixInstK"] * tP["swizzlePackK"]

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherFreeAssignments(self):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherSummationAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B
  ##############################################################################
  @abc.abstractmethod
  def graShift(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def graAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments A/B
  # This function declares the increments
  ##############################################################################
  @abc.abstractmethod
  def graIncrements(self, kernel, loopIdx, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaFirstOffset(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignment(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lraFinalOffset(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  @abc.abstractmethod
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lraDeclareAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Recalculate local read addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalReadAddressesAB(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Recalculate local write addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalWriteAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Define stagger parms that will be used in calculateStagger
  ##############################################################################
  @abc.abstractmethod
  def declareStaggerParms(self, kernel):
    return ""


  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  @abc.abstractmethod
  def calculateStagger(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Remove stagger offset (before tail loop)
  ##############################################################################
  @abc.abstractmethod
  def removeStagger(self, kernel):
    return ""

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  @abc.abstractmethod
  def calculateLoopNumIter(self, kernel, tPA, tPB, loopIdx):
    return ""


  ##############################################################################
  # openShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def openShadowInit(self):
    return ""

  ##############################################################################
  # closeShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def closeShadowInit(self, kernel):
    return ""

  ##############################################################################
  # Initialize C
  ##############################################################################
  @abc.abstractmethod
  def initC(self, kernel):
    return ""

  ##############################################################################
  # Initialize Summ unroll
  ##############################################################################
  @abc.abstractmethod
  def initSumUnroll(self, kernel):
    return ""

  ##############################################################################
  # Open Loop
  # loopIdx<0 : tail loop
  ##############################################################################
  @abc.abstractmethod
  def openLoop(self, kernel, tPA, tPB, loopIdx, noLabelGen, beginLabelOnly):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, tPA, tPB, loopIdx, \
                finalLoop, emitEndLabelOnly, oddLabel=False):
    return ""

  ##############################################################################
  # End Summation
  ##############################################################################
  @abc.abstractmethod
  def endSummation(self, kernel, tPA, tPB, noSkipLoad = True, label = None, isOptNLL = False):
    return ""

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  @abc.abstractmethod
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isNGLL=False, NLLindex=0, NLLnum=1):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel, tPA, tPB, prefetch, isOptNLL, isNGLL):
    return ""

  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrementAB(self, kernel, tPA, tPB, loopIdx, prefetchIndex):
    return ""

  ##############################################################################
  # Global Read: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def globalReadDo(self, kernel, mode, tP):
    return ""

  ##############################################################################
  # directToLds m0 update: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def directToLdsM0Update(self, kernel, mode, tP, skipWait):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteDo(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Read: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadResetOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInitPointers(self, kernel, tPA, tP):
    return ""

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInc(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadDo(self, kernel, bufferIdx, innerUnrollIndex, epsi, tP):
    return ""

  ##############################################################################
  # Shift Vector Components d0/1
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents(self, kernel, tP):
    return ""

  ##############################################################################
  # globalWriteWorkGroupInit:
  # Perform work-group granularity init
  ##############################################################################
  @abc.abstractmethod
  def globalWriteWorkGroupInit(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWrite(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWrite(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # PrefetchGlobalRead2
  ##############################################################################
  @abc.abstractmethod
  def openPrefetchGlobalRead2(self, kernel):
    return ""

  @abc.abstractmethod
  def closePrefetchGlobalRead2(self):
    return ""

  ##############################################################################
  # Function End
  ##############################################################################
  @abc.abstractmethod
  def functionEnd(self, kernel, addLabel=True):
    return ""

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def getWaitcntCodeForDirectToVgpr(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, u, isNLL=False, beforeBarrier=False, NLLlast=False, oddLast=False):
    return ""

  ##############################################################################
  # waitcnt code for PrefetchGlobalRead
  ##############################################################################
  @abc.abstractmethod
  def getWaitcntCodeForPGR(self, kernel, tensorParametersA, tensorParametersB, comment):
    return ""

  ##############################################################################
  # isSwapGlobalReadOrderForDtvOrDtl
  ##############################################################################
  @abc.abstractmethod
  def isSwapGlobalReadOrderForDtvOrDtl(self, kernel, prefetch1=False):
    return ""

  ##############################################################################
  # WaveSplitK Reduction
  ##############################################################################
  @abc.abstractmethod
  def waveSplitKReduction(self, kernel):
    return ""

  ##############################################################################
  # SIMD Specialized Dispatch
  ##############################################################################
  @abc.abstractmethod
  def simdSpecDispatch(self, kernel, numCodePath):
    return ""

  ##############################################################################
  # longBranchScc0 - 32 bit offset
  ##############################################################################
  @abc.abstractmethod
  def longBranchScc0(self, label: Label, posNeg: int=0, comment=""):
    return ""

  ##############################################################################
  # longBranchScc1 - 32 bit offset
  ##############################################################################
  @abc.abstractmethod
  def longBranchScc1(self, label: Label, posNeg: int=0, comment=""):
    return ""

  ##############################################################################
  # longBranchVccnz - 32 bit offset
  ##############################################################################
  @abc.abstractmethod
  def longBranchVccnz(self, label: Label, posNeg: int=0, comment=""):
    return ""

  ##############################################################################
  # WaitCnt
  ##############################################################################
  def _wait(self, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, skipLocalRead, comment, skipGlobalReadInst=-1):
    if not self.do["Wait"]: return Module("noWait")
    return wait(self.states, kernel, tPA, tPB, skipGlobalRead, \
      skipLocalWrite, skipLocalRead, self.db["ConservativeWaitCnt"], comment, skipGlobalReadInst)

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def _syncThreads(self, kernel, comment="", skipForceWaitcnt0=False):
    if self.do["Sync"]:
      return syncThreads(kernel, self.states.archCaps, self.states.asmCaps, comment, skipForceWaitcnt0=skipForceWaitcnt0)
    return Module("SyncThreads (Empty)")

  ##############################################################################
  #
  #   Internal helper functions for entry functions
  #
  ##############################################################################


  def _getKernelSource(self, kernel: Solution):
    """
    Returns the source of the kernel, either C++ or assembly.
    """

    fileString = ""
    tensorParametersA = {}
    tensorParametersB = {}
    self._initKernel(kernel, tensorParametersA, tensorParametersB)
    self.stringIdx = 0
    (error, kb) = self.kernelBody(kernel, tensorParametersA, tensorParametersB)
    fileString += str(kb)

    if error != 0:
      if self.debugConfig.forceGenerateKernel:
        printWarning("Generating kernel source resulted in error {}, but ForceGenerateKernel=1 so saving source".format(error))
      else:
        raise RuntimeError("Generating kernel source resulted in error {}".format(error))
    return fileString


  @abc.abstractmethod
  def getSourceFileString(self, kernel) -> Tuple[int, str]:
    """
    Returns a string suitable for placing in Kernels.cpp.  This means the actual kernel source in the case
    of a source kernel, or an assembled code object byte array definition in the case of an assembly kernel.

    In the case of an assembly kernel, this function has the side effect of creating the following files:
     * An assembly source file
     * An object file
     * A code object file
     * A Python script which can create byte array variable definitions.
    """

  def getHeaderFileString(self, kernel):
    kernelName = getKernelNameMin(kernel, self.debugConfig.splitGSU)
    fileString = "" # CHeader
    fileString += "extern const unsigned char %s_coba[]; // code object byte array\n" % kernelName

    return fileString


  def setRocIsa(self, data, outOptions):
    ti = rocIsa.getInstance()
    ti.setData(data)
    ti.setOutputOptions(outOptions)

  def updateBranchPlaceHolder(self, module, placeholders, targets, operations):
    phs = [ ph for ph in placeholders ]
    found_phs = {}

    def findInstByName(module, names, count):
      for inst in module.items():
        if isinstance(inst, Module):
          if inst.name in phs:
            found_phs[count] = inst
            count += 1
          else:
            count = findInstByName(inst, names, count)
        elif (not isinstance(inst, TextBlock)):
          count += 1
      return count

    currentInstLength = findInstByName(module, phs, 0)
    if len(found_phs) == 0:
      return

    counts = reversed(sorted(found_phs.keys())) if sys.version_info < (3,8) else reversed(found_phs.keys())
    for count in counts:
      _placeholder = found_phs[count]
      ph_idx = placeholders.index(_placeholder.name)

      _target = Label(targets[ph_idx], "")
      _operation = operations[ph_idx]

      _placeholder.name = "Branch_%s"%_target.getLabelName()
      if _operation == "SCBranchSCC0":
        if currentInstLength - count + 1 >= 16384:
          with self.allocTmpSgpr(3) as tmpSgprInfo:
              _placeholder.add(self.longBranchScc0(_target, 1, tmpSgprInfo))
        else:
          _placeholder.add(SCBranchSCC0(labelName=_target.getLabelName()))
      elif _operation == "SCBranchSCC1":
        if currentInstLength - count + 1 >= 16384:
          with self.allocTmpSgpr(3) as tmpSgprInfo:
              _placeholder.add(self.longBranchScc1(_target, 1, tmpSgprInfo))
        else:
          _placeholder.add(SCBranchSCC1(labelName=_target.getLabelName()))
      elif _operation == "SCBranchVCCNZ":
        if currentInstLength - count + 1 >= 16384:
          with self.allocTmpSgpr(3) as tmpSgprInfo:
              _placeholder.add(self.longBranchVccnz(_target, 1, tmpSgprInfo))
        else:
          _placeholder.add(SCBranchVCCNZ(labelName=_target.getLabelName()))
      elif _operation == "SBranch":
        if currentInstLength - count + 1 >= 16384:
          with self.allocTmpSgpr(3) as tmpSgprInfo:
            _placeholder.add(SLongBranchPositive(_target, tmpSgprInfo))
        else:
          _placeholder.add(SBranch(labelName=_target.getLabelName()))
      currentInstLength += countInstruction(_placeholder)

  @property
  def isa(self):
    return self.states.version

##############################################################################
# Assert
##############################################################################

def bomb(scratchVgpr, cookie=None):
    """
    Cause a GPUVM fault.
    Instruction after the bomb will write the cookie to SGPR0, so you can see the cookie in the
    backtrace. Useful for locating which spot in code generated the bomb
    vgprAddr controls which vgpr to overwrite with the null pointer address
    """

    module = Module("bomb")
    vgprAddr = scratchVgpr

    if cookie != None:
        if cookie < 0:
            module.add(Label("bomb_neg%u" % abs(cookie), ""))
        else:
            module.add(Label("bomb_%u" % abs(cookie), ""))
    module.add(VMovB32(dst=vgpr(vgprAddr+0), src=0))
    module.add(VMovB32(dst=vgpr(vgprAddr+1), src=0))
    module.add(FlatLoadB32(dst=vgpr(vgprAddr), vaddr=vgpr(vgprAddr,2), comment="bomb - force fault" ))

    # This move does not execute but appears in the instruction stream immediately following
    # the faulting load:
    if cookie != None:
        module.add(SMovB32(dst=sgpr(0), src=cookie, comment="bomb cookie=%d(0x%x)"%(cookie,cookie&0xffffffff)))

    return module

class Assert():
    def __init__(self, laneSGPRCount, wavefrontSize, enableAsserts):
        self.printedAssertCnt = 0
        self.laneSGPRCount = laneSGPRCount
        self.wavefrontSize = wavefrontSize
        self.enableAsserts = enableAsserts

    ##############################################################################
    # assertCommon : Common routine for all assert functions.
    # On entry, we have already set the exec-mask so any enabled lanes should bomb
    ##############################################################################
    def assertCommon(self, vtmp, cookie=-1):
        module = Module("assertCommon")
        if self.enableAsserts:
            self.printedAssertCnt += 1
            # Default cookie for asserts is negative of printed #asserts
            # Can be used to roughly identify which assert in the code is firing
            module.add(bomb(vtmp, cookie if cookie != -1 else -self.printedAssertCnt))
        return module

    ##############################################################################
    # assertCmpCommon : Common routine for all assert comparison functions
    ##############################################################################
    def assertCmpCommon(self, inst, val0, val1, vtmp, cookie=-1):
        assert issubclass(inst, VCmpXInstruction)
        module = Module("assertCmpCommon")
        if self.enableAsserts:
            SOrSaveExecBX = SOrSaveExecB64 if self.wavefrontSize == 64 else SOrSaveExecB32
            module.add(SOrSaveExecBX(dst=sgpr("SaveExecMask",self.laneSGPRCount), src=0, \
                comment="assert: saved execmask"))
            module.add(inst(dst=VCC(), src0=val0, src1=val1, comment="v_cmp")) # type: ignore
            module.add(self.assertCommon(vtmp, cookie))
            module.add(SOrSaveExecBX(dst=VCC(), src=sgpr("SaveExecMask",self.laneSGPRCount), \
                comment="assert: restore execmask"))
        return module

    ##############################################################################
    # Handle different conditions for the asserts:
    # These support uin32 compare, float could be added later
    # Asserts currently modify vcc
    ##############################################################################
    def eq(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXNeU32, val0, val1, vtmp, cookie)

    def eq_u16(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXNeU16, val0, val1, vtmp, cookie)

    def ne(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXEqU32, val0, val1, vtmp, cookie)

    def lt_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXGeU32, val0, val1, vtmp, cookie)

    def gt_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXLeU32, val0, val1, vtmp, cookie)

    def le_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXGtU32, val0, val1, vtmp, cookie)

    def ge_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXLtU32, val0, val1, vtmp, cookie)

    def ge_i32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXLtI32, val0, val1, vtmp, cookie)

    # can left shift w/o losing non-zero bits:
    def no_shift_of(self, val0, shift, stmp, vtmp, cookie=-1):
        module = Module("Assert no shift of")
        # TODO - use BFE here:
        module.add(SMovB32(dst=stmp, src=hex((shift-1) << (32-log2(shift))), comment="assert_no_shift_of - compute mask"))
        module.add(SAndB32(dst=stmp, src0=stmp, src1=val0, comment="assert_no_shift_of"))
        module.add(self.eq(stmp, 0, vtmp, cookie))
        return module

    # asserts if val0 is not an integer multiple of multiple2
    # multiple2 must be a constant and power of 2
    # for example assert_multiple(A, 8) will assert if A is not multiple of 8
    def multiple_b32(self, sval, multiple2, vtmp, cookie=-1):
        module = Module("Assert multiple b32")
        if self.enableAsserts:

            stmp = sgpr("SaveExecMask") # repurpose to get a tmp sgpr
            SAndBX = SAndB64 if self.wavefrontSize else SAndB32
            module.add(SAndBX(dst=stmp, src0=sval, src1=multiple2-1, comment="mask" ))
            module.add(SCmpEQU32(src0=stmp, src1=0, comment="if maskedBits==0 then SCC=1 == no fault" ))
            SMovBX = SMovB64 if self.wavefrontSize else SMovB32
            module.add(SMovBX(dst=sgpr("SaveExecMask",self.laneSGPRCount), src=-1))
            SCMovBX= SCMovB64 if self.wavefrontSize else SCMovB32
            module.add(SCMovBX(dst=sgpr("SaveExecMask", self.laneSGPRCount),  src=0, comment="Clear exec mask"))

            SAndSaveExecBX = SAndSaveExecB64 if self.wavefrontSize else SAndSaveExecB32
            module.add(SAndSaveExecBX(dst=sgpr("SaveExecMask",self.laneSGPRCount), src=sgpr("SaveExecMask",self.laneSGPRCount), \
                comment="assert: saved execmask"))

            module.add(self.assertCommon(vtmp, cookie))

            SOrSaveExecBX = SOrSaveExecB64 if self.wavefrontSize else SOrSaveExecB32
            module.add(SOrSaveExecBX(dst=VCC(), src=sgpr("SaveExecMask",self.laneSGPRCount), \
                comment="assert: restore execmask"))

        return module

    # assert v0 + expectedScalarDiff == v1
    # Verify that each element in v1 is scalar offset from v0
    def assert_vector_diff(self, v0, v1, expectedScalarDiff, cmpvtmp, vtmp, cookie=-1):
        module = Module("assert_vector_diff")
        module.add(VAddCOU32(dst=vgpr(cmpvtmp), \
                       dst1=VCC(), \
                       src0=expectedScalarDiff, \
                       src1=v0, \
                       comment="assert_vector_diff add expectedScalarDiff"))
        module.add(self.eq(vgpr(cmpvtmp), v1, vtmp, cookie))
        return module
