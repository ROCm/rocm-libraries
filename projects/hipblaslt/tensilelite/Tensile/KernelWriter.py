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

from rocisa import rocIsa, countInstruction, countGlobalRead, \
            countLocalRead, countLocalWrite, countWeightedLocalRead, countWeightedLocalWrite, countMFMA, getMFMAs
from rocisa.code import Module, TextBlock, StructuredModule, KernelBody, RegSet
from rocisa.container import RegisterContainer, replaceHolder, HWRegContainer, VCC, MemTokenData, sgpr, vgpr
from rocisa.label import LabelManager
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.instruction import BufferLoadB128, BufferLoadB192, BufferLoadB32, BufferLoadB64, BufferLoadB96, \
  BufferLoadD16B16, BufferLoadD16U8, DSLoad2B32, DSLoad2B64, DSLoadB128, \
  DSLoadB32, DSLoadB64, DSLoadB192, DSStoreB192, DSLoadB64TrB16, DSLoadB128TrB16, \
  DSLoadB64TrB8, DSLoadB64TrB4, DSLoadB96TrB6, DSLoadInstruction, DSLoadU16, \
  DSLoadU8, DSStore2B32, DSStore2B64, DSStoreB128, DSStoreB16, DSStoreB96, DSStoreB256, \
  DSStoreB32, DSStoreB64, DSStoreB8, DSStoreInstruction, FlatLoadB128, FlatLoadB192, FlatLoadB32, \
  FlatLoadB64, FlatStoreB128, FlatStoreB32, FlatStoreB64, Instruction, MacroInstruction, \
  MFMAInstruction, MXMFMAInstruction, SAndB32, SBarrier, SBranch, SCBranchSCC0, SCBranchSCC1, SCBranchVCCNZ, SCmpEQU32, SCmpEQU64, SCmpGeU32, SCmpLeU32, \
  SCSelectB32, SLShiftLeftB32, SLShiftRightB32, SMFMAInstruction, SMovB32, SMovB64, SNop, SEndpgm, SOrB32, SSetPrior, SSetRegIMM32B32, SSubU32, SWaitCnt, SWaitAlu, \
  SLongBranchPositive, VFmaMixF32, VMadMixF32, VMovB32, VAndB32, VCmpEQU32, VCndMaskB32, VMovB64, VReadfirstlaneB32, VNop, TensorLoadToLds, SCMovB32, SCMovB64
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType, DataTypeEnum

from .KernelWriterModules import *

from .Component import Component, LraTileProperties
from .Components.Signature import UserArgumentsInfo
from .Components.CustomSchedule import customMainLoopSchedule
from .Components.Subtile.Kernel import *
from .SolutionStructs import Solution, isPackedIndex
from .SolutionStructs.Utilities import getMiInputType
from .AsmMemoryInstruction import MemoryInstruction
from .Activation import ActivationModule
from .Common import printWarning, roundUp, print2, DebugConfig, DataDirection, \
  INDEX_CHARS, IsaVersion, log2
from .Common.GlobalParameters import globalParameters
from Tensile.SolutionStructs.Naming import getKernelNameMin
from Tensile.Toolchain.Component import Assembler

import rocisa
import math
import abc
import sys
import os
import time
import collections
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional,Tuple, Type
from math import ceil, prod
import itertools

# TODO: DEBUG ONLY, remove later
from pprint import pprint

# Make const values immutable
@dataclass(frozen=True)
class ConstValues():
  initLdsValue:int  = 0xFFFFFFFF  # Value to use for LDS Init, if enabled
  initSgprValue:int = 0x0  # Value to use for Sgpr Init, if enabled
  initVgprValue:int = 0xFFFFFFFF  # Value to use for Vgpr Init, if enabled

  ldsOOB: int       = 0xF00000

@dataclass
class MatrixInfo:
  numVgprValu: int               = -1
  numVgprValuPack: int           = -1
  startVgprValu: int             = -1
  startVgprValuPack: int         = -1
  startVgprValuPackTemp: int     = -1

  numSgprStrides: int            = -1
  tileInfo: object = field(init=False)  # TileInfo for all tile types



@dataclass
class ABMatrixInfo(MatrixInfo):
  numVgprValuPerBlock: int       = -1
  numVgprG2L: int                = -1
  numVgprG2LAllocated: int       = -1
  numVgprG2LTailLoopAllocated: int= -1
  startVgprG2L: Optional[int]    = None
  numVgprLocalReadAddr:int       = -1
  startVgprLocalReadAddr: int    = -1
  startVgprLocalReadAddrOrig: int= -1
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
  useConstSgprGlobalReadIncs: bool = False
  numPackCvt: int                = 0
  useTransposeCodeThis: bool     = False
  useTransposeCodeNext: bool     = False
  TF32EmuInterleaveTreg: bool    = False
  useDirect32XEmulationThis: bool = False
  useDirect32XEmulationNext: bool = False
  numVgprEmu: int                = -1
  startVgprCvt: int              = -1
  tmpVgprCvtSub: int             = -1  # per-tensor temp VGPR for XF32 cvt+sub residual

  gNLCPermBlock: int             = -1
  gNLCPerpStride: int            = -1
  gRDtlSwizzlePerpBlockSize: int    = -1
  gRDtlSwizzleParaBlockSize: int    = -1
  startVgprGL2PrefetchAddr: int = -1

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
  # A(B)=MacDataTypeA(B) / Cexternal=DestDataType / Cinternal=Accumulation (MAC or MFMA)
  bpeA: float = field(init=False)
  bpeB: float = field(init=False)
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
  numVgprBufferPackMXSA: int             = 0
  numVgprBufferPackMXSB: int             = 0
  numVgprBufferPackMetadata: int         = 0
  numPackBuffer: int                     = 0
  lrvwTileA: int                         = 0
  lrvwTileB: int                         = 0
  lrvwTileMXSA: int                      = 0
  lrvwTileMXSB: int                      = 0
  lrvwTileMetadata: int                  = 0 # For Sparse Metadat
  lrvwUnrollA: int                       = 0
  lrvwUnrollB: int                       = 0
  lrvwUnrollMXSA: int                    = 0
  lrvwUnrollMXSB: int                    = 0
  lrvwUnrollMetadata: int                = 0 # For Sparse Metadat

  numMfmaPerIter: int                    = 0
  SubTileIdx: int                       = 0
  subtileLdsSwizzle: bool                = False
  numReadsIterCoalescedA: int            = 0
  numReadsIterCoalescedB: int            = 0
  numReadsIterCoalescedMXSA: int         = 0
  numReadsIterCoalescedMXSB: int         = 0
  numReadsIterCoalescedMetadata: int     = 0
  numIterPerCoalescedReadA: int          = 0
  numIterPerCoalescedReadB: int          = 0
  numIterPerCoalescedReadMXSA: int       = 0
  numIterPerCoalescedReadMXSB: int       = 0
  numIterPerCoalescedReadMetadata: int   = 0
  numReadsPerUnrollA: int                = 0
  numReadsPerUnrollB: int                = 0
  numReadsPerUnrollMXSA: int             = 0
  numReadsPerUnrollMXSB: int             = 0
  numReadsPerUnrollMetadata: int         = 0

  # KernelWriterAssembly
  mixinst: Optional[Type[Instruction]]   = None
  globalReadIncsUseVgpr: bool            = False
  groOffsetInMacroTile: int              = 0
  use64bShadowLimit: bool                = True
  use64bShadowLimitMX: bool              = False
  preventVgprOverflowDuringNewTile: int  = -1
  interleaveStoreVmcnt: bool             = False
  srdShiftLeft:dict                      = field(init=False)
  checkGRO: bool                         = False
  combineLocalAddresses: bool            = False # Debug
  unifiedVgprRegs: bool                  = False
  useAtomicAdd: bool                     = False
  serializedStore: bool                  = False
  storeAlign8: bool                      = False
  subtileTotalMOffsetSgpr: Optional[int] = None

  a: ABMatrixInfo                        = field(default_factory=ABMatrixInfo)
  b: ABMatrixInfo                        = field(default_factory=ABMatrixInfo)
  mxsa: ABMatrixInfo                     = field(default_factory=ABMatrixInfo)
  mxsb: ABMatrixInfo                     = field(default_factory=ABMatrixInfo)
  c: MatrixInfo                          = field(default_factory=MatrixInfo)
  d: MatrixInfo                          = field(default_factory=MatrixInfo)
  e: MatrixInfo                          = field(default_factory=MatrixInfo)
  bias: MatrixInfo                       = field(default_factory=MatrixInfo)
  m: ABMatrixInfo                        = field(default_factory=ABMatrixInfo)       # For Sparse Metadata
  startMXDummyValuVgpr: int              = 0
  totalAgprs: int                        = 0
  maxLimitAgprs: int                     = 0
  totalMixedAgprs: int                   = 0
  totalVgprs: int                        = 0
  totalSgprs: int                        = 0
  lastValuAB: int                        = 0
  lastValuMXSAB: int                     = 0
  lastVgprForReads: int                  = 0
  startVgpr: int                         = 0
  startVgprAddressDbg: int               = -1
  startVgprAlphaTmp: int                 = -1
  startVgprSerial: int                   = -1
  startVgprSKConsts: int                 = -1
  numVgprSKConsts: int                   = 0
  startVgprIdentityMatrix: int           = -1

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
  localReadDoCntMXSA: int                = 0
  localReadDoCntB: int                   = 0
  localReadDoCntMXSB: int                = 0
  localReadDoCntMetadata: int            = 0
  savedLocalReadDoCntA: int              = 0
  savedLocalReadDoCntB: int              = 0
  savedLocalReadDoCntMXSA: int           = 0
  savedLocalReadDoCntMXSB: int           = 0
  savedLocalReadDoCntMetadata: int       = 0

  ldsStartOffsetA: int                   = -1
  ldsStartOffsetB: int                   = -1
  ldsStartOffsetMXSA: int                = -1
  ldsStartOffsetMXSB: int                = -1
  ldsTotalSize: int                      = 0

  dtvKIntervalA: int                     = 1
  dtvKIntervalB: int                     = 1

  halfPLRGroups: List[int]               = field(init=False)
  ## MFMA
  miLatency: int                         = 0
  miLatencyLeft: int                     = 0
  miDependency: int                      = 0
  miVALUInstrDataHazard: int             = 0 # 2nd VALU instruction is not any XDL WMMA/SWMMAC instruction
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
  tmpvgpr: List[int]                     = field(init=False) # vgpr dict for localread
  packDTVA: bool                         = False
  packDTVB: bool                         = False

  lraTileProperties: Dict[int, LraTileProperties] = field(init=False)

  WGMTransformLevels: int                = -1
  tailloopInNll: bool                    = False
  tailloopInNllmaxUnit: int              = 0
  staggerUCode: bool                     = 0
  scheduleGROverBarrier: bool            = False
  numLDSBlk: int                         = 0
  IncLdsBufSwitch: bool                  = False
  oneBufferScheduling: bool              = False
  doPackPreSchedulingThisLoop: bool      = False
  doPackPreSchedulingNextLoop: bool      = False
  doFullPackCodePrefetch: bool           = False
  lockLdsReadTokenSwap: bool             = False
  useCommonSgprSwap: bool                = False

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
  localWriteMXSA: Optional[Module]            = None
  localWriteMXSB: Optional[Module]            = None
  dtlsM0UpdateA: Optional[Module]             = None
  dtlsM0UpdateB: Optional[Module]             = None
  dtlsM0UpdateMXSA: Optional[Module]          = None
  dtlsM0UpdateMXSB: Optional[Module]          = None
  globalReadA: Optional[Module]               = None
  globalReadB: Optional[Module]               = None
  globalReadMXSA: Optional[Module]            = None
  globalReadMXSB: Optional[Module]            = None
  globalReadMetadata: Optional[Module]         = None
  globalReadIncrements: Optional[Module]      = None
  gl2Prefetch: Optional[Module]               = None
  gl2PrefetchIncrement: Optional[Module]      = None
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
    self.do["GlobalReadMXSA"] = True
    self.do["GlobalReadMXSB"] = True
    self.do["GlobalReadMetadata"] = True
    self.do["GlobalInc"]   = True
    self.do["LocalWriteA"]  = True
    self.do["LocalWriteB"]  = True
    self.do["LocalWriteMXSA"]  = True
    self.do["LocalWriteMXSB"]  = True
    self.do["LocalWriteMetadata"]  = True
    self.do["LocalWriteCVT"]  = True
    self.do["LocalReadA"]  = True
    self.do["LocalReadB"]  = True
    self.do["LocalReadMXSA"] = True
    self.do["LocalReadMXSB"] = True
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
    self.do["EmulatedECCBufferLoad"] = False

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
    self.db["CheckValue1MXSA"] = False
    self.db["CheckValue1MXSB"] = False
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
    self.db["ForceInputValueMXSA"] = False
    self.db["ForceInputValueMXSB"] = False
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
    self.states.scheduledGRInstCounts = 0

    globalReadIncACode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementB")

    if skipGlobalReadInc:
      globalReadIncACode  = Module()
      globalReadIncBCode  = Module()

    siaComponent = Component.SIA.find(self)
    if siaComponent:
      siaComponent.schedIntoIteration(self, kernel, tensorParametersA, tensorParametersB, \
        localWriteEndIter, firstIter, lastLoop, lastLc, globalReadIncACode, \
        globalReadIncBCode, isNGLL)

  ##############################################################################
  # packItemsConditional: pack src items into dst items until numPack or searchString is found
  # returns number of items packed
  ##############################################################################
  def _packItemsConditional(self, numPack, srcPackItems, dstPackItems, searchStrings):
    numPacked = 0
    final = False
    finalStr = "pack final end"
    if numPack == 0:
      numPack = 999
    for n in range(numPack):
      if srcPackItems:
        item = srcPackItems.pop(0)
        dstPackItems.append(item)
        numPacked += 1
        itemStr = str(item.comment)
        for string in searchStrings:
          if string in itemStr:
            final = finalStr in itemStr
            # check if the next item is nop or not
            # if nop, pop
            if srcPackItems:
              item = srcPackItems.pop(0)
              if isinstance(item, SNop):
                # add nop. Do not count nop
                dstPackItems.append(item)
              else:
                # push back
                srcPackItems.insert(0, item)
            return numPacked, final
    return numPacked, final

  ##############################################################################
  # Interleave A, B pack code based on scheduling order
  # A0,B0,A1,A2,,,B1,B2,.
  ##############################################################################
  def isMFMAIn(self,items):
    for inst in items:
      if isinstance(inst, MFMAInstruction):
        return True
    return False
  def countAfter2ndMFMA(self,items):
    mfmaCount = 0
    mfmaBlockCount = 0
    after2ndMfma = 0
    after2ndMfmaCountStart = False
    for inst in items:
      if after2ndMfmaCountStart:
        after2ndMfma += 1
      if isinstance(inst, MFMAInstruction):
        mfmaCount += 1
      if mfmaBlockCount == 0 and mfmaCount == 2:
        # ater reaching the first 2 mfma, start counting after2ndMfma
        after2ndMfmaCountStart = True
    return after2ndMfma
  def insertNopForInterleavePackAB(self,dstPackItems, currPackItems, numA, numB, final, prefetch, mfma):
    waitState = -1
    if prefetch and mfma and not final:
      # prefetch + mfma included case, insert nop
      if (numA > 0 and numB == 0) or (numA == 0 and numB > 0):
        waitState = 4
      else:
        # both numA and numB case
        # nop insertion is determined by number of instruction after 2nd mfma
        # case 1: full interleave case
        #         gather both mfma A and B at once
        #         in this case, the 2nd mfma is referred first.
        #         we need at least 5 instrcutions to avoid inserting nop.
        #         insert nop if number of ist after 2nd mfma is 4 or less
        # case 2: interleave per block
        #         still need to insert nop 0 for mfma and not final
        # parse current pack items
        mfmaCount = 0
        mfmaBlockCount = 0
        after2ndMfma = 0
        after2ndMfmaCountStart = False
        for item in currPackItems:
          if after2ndMfmaCountStart:
            after2ndMfma += 1
          if mfmaBlockCount == 0 and mfmaCount == 2:
            # ater reaching the first 2 mfma, start counting after2ndMfma
            after2ndMfmaCountStart = True
          if isinstance(item, MFMAInstruction):
            mfmaCount += 1
          else:
            if mfmaCount > 0:
              mfmaBlockCount += 1
            mfmaCount = 0
        if mfmaCount > 0:
          mfmaBlockCount += 1
        if mfmaBlockCount == 1 and after2ndMfma <= 4:
          # need at least 5 instruction in mfmaBlockCount==1 case (full interleave case)
          waitState = 4 - after2ndMfma
        elif mfmaBlockCount >= 2 and mfmaCount == 2:
          # interleave case, we still need nop 0
          waitState = 0

    if waitState >= 0:
      dstPackItems.append(SNop(waitState=waitState, comment="nop for x32f emulation"))

  def _interleavePackAB(self, kernel, packAItems, packBItems, dstPackItems, prefetch=False, searchStrings=["__TF32_1", "__TF32_2"]):
    carryOverPackItems = []
    while packAItems or packBItems:
      tmpPackItems = []
      numA, finalA = self._packItemsConditional(0, packAItems, tmpPackItems, searchStrings)
      numB, finalB = self._packItemsConditional(0, packBItems, tmpPackItems, searchStrings)
      final = finalA or finalB
      mfma = self.isMFMAIn(tmpPackItems)
      if numA > 0 and numB > 0:
        # both A and B exist, add to dst
        dstPackItems += tmpPackItems
        self.insertNopForInterleavePackAB(dstPackItems, tmpPackItems, numA, numB, final, prefetch, mfma)
      elif (numA > 0 and numB == 0) or (numA == 0 and numB > 0):
        doInterleave = True
        # no Treg (means use tmp reg) case, we use same tmp reg and cannot interleave same side
        if numA > 0 and not (self.states.a.useDirect32XEmulationThis and self.states.a.useDirect32XEmulationNext):
          doInterleave = False
        if numB > 0 and not (self.states.b.useDirect32XEmulationThis and self.states.b.useDirect32XEmulationNext):
          doInterleave = False

        # only one side
        if (not final):
          # if we already have 2 items in carry over, we interleave the current item with the previous iteration
          if len(carryOverPackItems) == 2:
            # previous iter 1st half (not final)
            dstPackItems += carryOverPackItems.pop(0)
            # current iter 1st half
            dstPackItems += tmpPackItems
            # previous iter 2nd half (not final)
            dstPackItems += carryOverPackItems.pop(0)
            if prefetch and mfma:
              # add s_nop 1
              dstPackItems.append(SNop(waitState=1, comment="nop for x32f emulation"))
          else:
            # carry over current item to the next iteration
            carryOverPackItems.append(tmpPackItems)
        else:
          # final case
          # carry over should start from non final
          if len(carryOverPackItems) == 0:
            dstPackItems += tmpPackItems
          else:
            # carry over current item to the next iteration
            carryOverPackItems.append(tmpPackItems)
        if not doInterleave:
          # interleave disabled case, force to add to dst
          while len(carryOverPackItems) :
            item = carryOverPackItems.pop(0)
            dstPackItems += item
            if self.isMFMAIn(item):
              numAfter = self.countAfter2ndMFMA(item)
              # add s_nop for MFMA for the remaining items
              if numAfter <= 4:
                dstPackItems.append(SNop(waitState=4-numAfter, comment="nop for x32f emulation"))

    # add remaining carry over items
    while len(carryOverPackItems) > 0:
      item = carryOverPackItems.pop(0)
      # add back to dst
      dstPackItems += item
      mfma = self.isMFMAIn(item)
      # nop insert for MFMA4x4 (for prefetch/tail loop)
      if mfma:
        numAfter = self.countAfter2ndMFMA(item)
        # add s_nop for MFMA for the remaining items
        if numAfter <= 4:
          dstPackItems.append(SNop(waitState=4-numAfter, comment="nop for x32f emulation"))

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
      waitLWCode = Module(), syncCode = Module(), packCode = Module(), packPreCode = Module(), prevIterCode = Module(), localReadCodeSecondHalf = Module(), NLLlast = False, \
                   tailloopInNll = False, isNLLorNGLL=False):

    iterCode = Module()
    globalReadCode       = deepcopy(self.codes.perIterGlobalRead[iteration])
    localWriteCodeCounts = self.codes.perIterLocalWrite[iteration][0]
    localWriteCode       = self.codes.perIterLocalWrite[iteration][1]
    isBarrier            = kernel["LoopIters"] - self.states.numItersPLR
    if self.states.doFullPackCodePrefetch and kernel["ForceUnrollSubIter"]:
      # hack for doFullPackCodePrefetch and SubIter
      # move isBarrier 1 iter ahead (calculate isBarrier based on syncPlrMfmaIndex)
      isBarrier = self.states.syncPlrMfmaIndex // self.states.numMfmaPerIter
    hasLocalRead = countLocalRead(localReadCode)
    scheduleIterAlg = self.states.scheduleIterAlg
    if (NLLlast and tailloopInNll):
      # use scheduleIterAlg = 0 for NLLlast and tailloopInNll case
      scheduleIterAlg = 0
    # Default schedule is other, local reads, then local writes:
    if scheduleIterAlg == 0:
      # simple schedule, just add the modules in-order
      if kernel["HalfPLR"]:
        assert len(packCode.flatitems()) == 0, "Pack code should be empty for half PLR case"
        if kernel["PrefetchGlobalRead"] < 2:
          iterCode.add(globalReadCode)
        iterCode.add(waitLWCode)
        iterCode.add(syncCode)
        iterCode.add(localReadCode)
        iterCode.add(localWriteCode)
        iterCode.add(waitCode)
        macCnt = countMFMA(macIterCode)
        assert macCnt % 2 == 0, "HalfPLR does not support odd number of matrix instructions"
        macToSchedule = macCnt // 2
        macItems = macIterCode.flatitems()
        while macItems:
          item = macItems.pop(0)
          iterCode.add(item)
          macsThisItem = countMFMA(item)
          if macsThisItem:
            assert macsThisItem==1, "Scheduler assumes 1 mac per item"
            macToSchedule -= 1
            if macToSchedule == 0:
              break
        iterCode.add(localReadCodeSecondHalf.popFirstItem())
        # when HalfPLR == 3, we need to add mac before the second half of LR of B (the inner tile index for mac)
        if kernel["HalfPLR"] == 3:
          iterCode.addItems(macItems)
          iterCode.add(localReadCodeSecondHalf)
          macItems = []
        iterCode.add(pointerLWCode)
        iterCode.add(pointerLRCode)
        if kernel["PrefetchGlobalRead"] >= 2: 
          iterCode.add(globalReadCode)
        # add rest of the mac here
        iterCode.addItems(macItems)
      else:
        if(kernel["PrefetchGlobalRead"] >= 2):
          iterCode.add(waitLWCode)
          iterCode.add(syncCode)
          iterCode.add(localReadCode)
          iterCode.add(localWriteCode)
          iterCode.add(pointerLWCode)
          iterCode.add(pointerLRCode)
          # PLR=0 + NoLdsWriteCode: drain dscnt before async LDS write to avoid
          # self-wave WAR (ds_load vs tensor_load_to_lds/DTL on the same buffer).
          if not self.states.numItersPLR and kernel["NoLdsWriteCode"]:
            iterCode.add(waitCode)
            iterCode.add(globalReadCode)
          else:
            iterCode.add(globalReadCode)
            iterCode.add(waitCode)
          # iterCode.add(packPreCode)
          iterCode.add(packCode)
          iterCode.add(macIterCode)
        else:
          iterCode.add(globalReadCode)
          iterCode.add(waitLWCode)
          iterCode.add(syncCode)
          iterCode.add(localReadCode)
          iterCode.add(localWriteCode)
          iterCode.add(pointerLWCode)
          iterCode.add(pointerLRCode)
          iterCode.add(waitCode)
          iterCode.add(packPreCode)
          iterCode.add(packCode)
          iterCode.add(macIterCode)
    elif scheduleIterAlg == 1:
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      #import pdb
      #pdb.set_trace()
      # simple algorithm - do half the reads first:
      # TODO: remove this half logic after stinkytofu works.
      readsToSchedule = countLocalRead(localReadCode) / 2
      #localReadCode.prettyPrint()
      readItems = localReadCode.flatitems()
      while readItems:
        item = readItems.pop(0)
        #print "readsToSchedule=", readsToSchedule, "item=", item
        item.name += " iter%s"%(iteration)  # tag for group
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

      if kernel["1LDSBuffer"]:
        if localWriteCode.itemsSize() > 0:
          barrier = Module()
          barrier.addComment0("1 LDS buffer: read-sync-write")
          barrier.add(SWaitCnt(dscnt=0, comment=""))
          _barrier = SBarrier()
          barrier.add(_barrier)
          iterCode.add(barrier)

      #move down write to be the last
      iterCode.add(localWriteCode)
      # tack on the pointer and mac code:
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(waitCode)
      iterCode.add(packPreCode)
      iterCode.add(packCode)
      iterCode.add(macIterCode)
    elif scheduleIterAlg == 2:
    # SIA2 use only 1 iteration and separate compute and fetch by raising compute priority
    # 2 workgroup interleave, while WG0/WG1 doing compute, WG1/WG0 doing fetch
    # EPS need to be 1, or valu instruction will break interleave
      iterCode.add(globalReadCode)
      iterCode.add(waitLWCode)
      iterCode.add(syncCode)
      iterCode.add(localReadCode)
      iterCode.add(waitCode)
      packCode.add(packPreCode)

      # interleave pack code
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      if self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]:
        instPerRegPackA = 1 / kernel["ProblemType"]["MacDataTypeA"].numRegisters() - 1
        instPerRegPackB = 1 / kernel["ProblemType"]["MacDataTypeB"].numRegisters() - 1
      else:
        instPerRegPackA = 1 if (kernel["ProblemType"]["MacDataTypeA"].numRegisters() == 0.25) else 0
        instPerRegPackB = 1 if (kernel["ProblemType"]["MacDataTypeB"].numRegisters() == 0.25) else 0
      instPerPackA    = 0 if kernel["UnrollMajorLDSA"] else int(kernel["MIInputPerThreadA"] * kernel["ProblemType"]["MacDataTypeA"].numRegisters() * instPerRegPackA)
      instPerPackB    = 0 if kernel["UnrollMajorLDSB"] else int(kernel["MIInputPerThreadB"] * kernel["ProblemType"]["MacDataTypeB"].numRegisters() * instPerRegPackB)
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
            for n in range(instPerPackA):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.states.numReadsIterCoalescedB):
            for n in range(instPerPackB):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.states.numReadsIterCoalescedA):
            for n in range(instPerPackA):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.states.numReadsIterCoalescedB):
            for n in range(instPerPackB):
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
          packAIdx += instPerPackA if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPackB if i % kernel["MIWaveTileA"] == 0 else 0
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
            for j in range(instPerPackA):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPackB):
              iterCode.add(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPackA):
                iterCode.add(packItems.pop(0))
                curPackIdx += 1
              for j in range(instPerPackB):
                iterCode.add(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.add(SNop(waitState=0, comment="VALU packing writes to be consumed by matrix instruction"))
              curPackIdx += 1
        if i == 0:
          if not packItems:
            tmpVgpr = self.vgprPool.checkOut(1, tag="_makeSubIterSchedule_tmpVgpr")
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
        _barrier = SBarrier()
        barrier.add(_barrier)
        iterCode.add(barrier)
      iterCode.add(localWriteCode)
      iterCode.add(pointerLWCode)
      iterCode.add(pointerLRCode)
      iterCode.add(SSetPrior(prior=2, comment="Raise priority while processing macs"))
    elif scheduleIterAlg == 3:
      oneBufferScheduling = self.states.oneBufferScheduling
      iterCode.addComment0(" grEndMfmaIndex:%u, lwStartMfmaIndex:%u, lwEndMfmaIndex:%u "\
                          %(self.states.grEndMfmaIndex, self.states.lwStartMfmaIndex, self.states.lwEndMfmaIndex))
      commentsync1Lds = ", sync1LdsMfmaIndex:%u"%self.states.sync1LdsMfmaIndex if oneBufferScheduling else ""
      iterCode.addComment0(" numMfmaForLR:%u, syncPlrMfmaIndex:%u %s"\
                           %(self.states.numMfmaForNextLoopLR, self.states.syncPlrMfmaIndex, commentsync1Lds))
      #####
      # Prepare and Assign parameter
      ####
      if iteration == 0:
        self.localReadsVacancy = []
        self.localReadsWait = [ [] for j in range(kernel["LoopIters"])]
      self.localReadsWait[iteration] = waitCode
      numMfmaPerIter = self.states.numMfmaPerIter
      writeItems = list(localWriteCode.items())
      macIterItems = macIterCode.flatitems()
      skipLocalWriteWaitcnt = 0
      localReadsWaitcnt = 0
      localReadsIssuedInThisIter = 0
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0
      packMXSAIdx = 0
      packMXSBIdx = 0
      packMIdx = 0

      schedulePackConsiderMetadata = kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]
      numPackedA = 0
      numPackedB = 0
      numPackedM = 0
      pointerLWCodeInserted = False
      PointerLRCodeItems = pointerLRCode.flatitems()
      lenPointerLRCode = 0
      # count number of PointerLRCode (excluding comment)
      for item in PointerLRCodeItems:
        if isinstance(item, Instruction):
          lenPointerLRCode += 1
      #####
      # Prepare localReadCode
      ####
      localReadCodeAB = Module()
      for iui in range(kernel["InnerUnroll"]):
        localReadCodeA = localReadCode.findNamedItem("LocalReadDoA_I%s"%(iui))
        localReadCodeB = localReadCode.findNamedItem("LocalReadDoB_I%s"%(iui))
        localReadCodeMXSA = localReadCode.findNamedItem("LocalReadDoMXSA_I%s"%(iui))
        localReadCodeMXSB = localReadCode.findNamedItem("LocalReadDoMXSB_I%s"%(iui))
        localReadCodeM = localReadCode.findNamedItem("LocalReadDoMetadata_I%s"%(iui))
        # In case localReadDo not generate localReadCode Module
        # and findNamedItem will return None type
        # TODO: findNamedItem return Module() if not found
        if not localReadCodeA:
          localReadCodeA = Module()
        if not localReadCodeB:
          localReadCodeB = Module()
        if not localReadCodeMXSA:
          localReadCodeMXSA = Module()
        if not localReadCodeMXSB:
          localReadCodeMXSB = Module()
        if not localReadCodeM:
          localReadCodeM = Module()

        if localReadCodeA.items():
          localReadCodeAB.add(localReadCodeA.popFirstItem())
        if localReadCodeMXSA.items():
          localReadCodeAB.add(localReadCodeMXSA.popFirstItem())
        if localReadCodeMXSB.items():
          localReadCodeAB.add(localReadCodeMXSB.popFirstItem())
        if localReadCodeM.items():
          localReadCodeAB.add(localReadCodeM.popFirstItem())
        if localReadCodeB.items():
          localReadCodeAB.add(localReadCodeB.popFirstItem())

        if localReadCodeA.itemsSize():
          localReadCodeAB.addItems(localReadCodeA.popFirstNItems(localReadCodeA.itemsSize()))
        if localReadCodeMXSA.itemsSize():
          localReadCodeAB.addItems(localReadCodeMXSA.popFirstNItems(localReadCodeMXSA.itemsSize()))
        if localReadCodeMXSB.itemsSize():
          localReadCodeAB.addItems(localReadCodeMXSB.popFirstNItems(localReadCodeMXSB.itemsSize()))
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
        instPerRegPackA = 1 / kernel["ProblemType"]["MacDataTypeA"].numRegisters() - 1
        instPerRegPackB = 1 / kernel["ProblemType"]["MacDataTypeB"].numRegisters() - 1
        instPerRegPackMX = 4 - 1
      else:
        instPerRegPackA = 1 if (kernel["ProblemType"]["MacDataTypeA"].numRegisters() == 0.25) else 0
        instPerRegPackB = 1 if (kernel["ProblemType"]["MacDataTypeB"].numRegisters() == 0.25) else 0
        instPerRegPackMX = 1
      instPerPackMXSA = 0
      instPerPackMXSB = 0
      if kernel["ProblemType"]["MXBlockA"] and (not kernel["UnrollMajorLDSMXSA"]):
        instPerPackMXSA = int(kernel["MIInputPerThreadMXSA"] * kernel["ProblemType"]["DataTypeMXSA"].numRegisters() * instPerRegPackMX)
      if kernel["ProblemType"]["MXBlockB"] and (not kernel["UnrollMajorLDSMXSB"]):
        instPerPackMXSB = int(kernel["MIInputPerThreadMXSB"] * kernel["ProblemType"]["DataTypeMXSB"].numRegisters() * instPerRegPackMX)

      instPerPackA    = 0 if kernel["UnrollMajorLDSA"] else int(kernel["MIInputPerThreadA"] * kernel["ProblemType"]["MacDataTypeA"].numRegisters() * instPerRegPackA)
      instPerPackB    = 0 if kernel["UnrollMajorLDSB"] else int(kernel["MIInputPerThreadB"] * kernel["ProblemType"]["MacDataTypeB"].numRegisters() * instPerRegPackB)
      if kernel["ConvertAfterDS"]:
        if kernel["ProblemType"]["DataTypeA"].isAnyFloat8():
          if kernel["UnrollMajorLDSA"]:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackA = 2 * self.states.numReadsIterCoalescedA if(iteration % self.states.numReadsIterCoalescedA == 0) else 0
            else:
              instPerPackA = 6 * self.states.numReadsIterCoalescedA if(iteration % self.states.numReadsIterCoalescedA == 0) else 0
          elif self.states.lrvwTileA == 1:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackA = 4
            else:
              instPerPackA = 8
          elif self.states.lrvwTileA == 2:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackA = 8
            else:
              instPerPackA = 16
          elif self.states.lrvwTileA == 4:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackA = 20
            else:
              instPerPackA = 36
          elif self.states.lrvwTileA == 8:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackA = 44
            else:
              instPerPackA = 76
        if kernel["ProblemType"]["DataTypeB"].isAnyFloat8():
          if kernel["UnrollMajorLDSB"]:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackB = 2 * self.states.numReadsIterCoalescedB if(iteration % self.states.numReadsIterCoalescedB == 0) else 0
            else:
              instPerPackB = 6 * self.states.numReadsIterCoalescedB if(iteration % self.states.numReadsIterCoalescedB == 0) else 0
          elif self.states.lrvwTileB == 1:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackB = 4
            else:
              instPerPackB = 8
          elif self.states.lrvwTileB == 2:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackB = 8
            else:
              instPerPackB = 16
          elif self.states.lrvwTileB == 4:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
              instPerPackB = 20
            else:
              instPerPackB = 36
          elif self.states.lrvwTileB == 8:
            if self.states.asmCaps["Hascvtf16_fp8_sf32"]:
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
          elif kernel["MIInputPerThreadMetadata"] == 8:
            instPerPackM = 6
        elif kernel["MIInputPerThreadMetadata"] == 4:
          instPerPackM = 3
        elif kernel["MIInputPerThreadMetadata"] == 8:
          instPerPackM = 6
      packItems = []
      packItemsA = []
      packItemsB = []
      packItemsMXSA = []
      packItemsMXSB = []
      packItemsM = []
      packPreItems = []
      packPreItemsA = []
      packPreItemsB = []
      scheduleTF32Emu = kernel["UseF32XEmulation"]
      maxReadIterCoal = max(self.states.numReadsIterCoalescedA, self.states.numReadsIterCoalescedB, self.states.numReadsIterCoalescedMetadata, \
                            self.states.numReadsIterCoalescedMXSA, self.states.numReadsIterCoalescedMXSB)
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(maxReadIterCoal) ]
        packINtemsA = packINtems
        packINtemsB = packINtems
        packINtemsMXSA = packINtems
        packINtemsMXSB = packINtems
        packINtemsM = packINtems
        if schedulePackConsiderMetadata:
          packINtemsA    = [ [] for j in range(maxReadIterCoal) ]
          packINtemsB    = [ [] for j in range(maxReadIterCoal) ]
          packINtemsMXSA = [ [] for j in range(maxReadIterCoal) ]
          packINtemsMXSB = [ [] for j in range(maxReadIterCoal) ]
          packINtemsM    = [ [] for j in range(maxReadIterCoal) ]

        # Add pack pre and pack code (put pack pre first)
        # They can be in either packCode or packPreCode)
        packA = packCode.findNamedItem("packA_I%s Pre"%(iui))
        packB = packCode.findNamedItem("packB_I%s Pre"%(iui))
        #TODO: do we nned packPreMXSA?
        packMXSA = packCode.findNamedItem("packMXSA_I%s"%(iui))
        packMXSB = packCode.findNamedItem("packMXSB_I%s"%(iui))
        packA2 = packCode.findNamedItem("packA_I%s"%(iui))
        packB2 = packCode.findNamedItem("packB_I%s"%(iui))
        packM = packCode.findNamedItem("packMetadata_I%s"%(iui))
        packPreA = Module()
        packPreB = Module()
        if packPreCode != None:
          packPreA = packPreCode.findNamedItem("packA_I%s Pre"%(iui))
          packPreB = packPreCode.findNamedItem("packB_I%s Pre"%(iui))
          packPreA2 = packPreCode.findNamedItem("packA_I%s"%(iui))
          packPreB2 = packPreCode.findNamedItem("packB_I%s"%(iui))
          if not packPreA:
            packPreA = Module()
          if packPreA2:
            packPreA.add(packPreA2)
          if not packPreB:
            packPreB = Module()
          if packPreB2:
            packPreB.add(packPreB2)
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Module()
        if packA2:
          packA.add(packA2)
        packAItems = packA.flatitems()
        if not packB:
          packB = Module()
        if packB2:
          packB.add(packB2)
        packBItems = packB.flatitems()
        if not packMXSA:
          packMXSA = Module()
        packMXSAItems = packMXSA.flatitems()
        if not packMXSB:
          packMXSB = Module()
        packMXSBItems = packMXSB.flatitems()
        if not packM:
          packM = Module()
        packMItems = packM.flatitems()
        if not packPreA:
          packPreA = Module()
          packPreB = Module()
        packPreAItems = packPreA.flatitems()
        if not packPreB:
          packPreB = Module()
        packPreBItems = packPreB.flatitems()

        if packAItems:
          if kernel["ConvertAfterDS"] and kernel["ProblemType"]["DataTypeA"].isAnyFloat8():
            for n in range(instPerPackA):
              packINtemsA[0].append(packAItems.pop(0))
          else:
            for j in range(self.states.numReadsIterCoalescedA):
              for n in range(instPerPackA):
                packINtemsA[j].append(packAItems.pop(0))

        if packMXSAItems:
          for j in range(self.states.numReadsIterCoalescedMXSA):
            for n in range(instPerPackMXSA):
              packINtemsMXSA[j].append(packMXSAItems.pop(0))

        if packMXSBItems:
          for j in range(self.states.numReadsIterCoalescedMXSB):
            for n in range(instPerPackMXSB):
              packINtemsMXSB[j].append(packMXSBItems.pop(0))

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

        if scheduleTF32Emu:
          instPerPackA = len(packAItems)
          instPerPackB = len(packBItems)
          firstDone = False
          # Gather A, B Pre conversion code based on scheduling order
          self._interleavePackAB(kernel, packPreAItems, packPreBItems, packPreItems, prefetch=True, searchStrings=["__TF32_1", "__TF32_2"])
          # Gather A, B conversion code based on scheduling order
          self._interleavePackAB(kernel, packAItems, packBItems, packItems, prefetch=True, searchStrings=["__TF32_1", "__TF32_2"])

        else:
          while packAItems or packMXSAItems:
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
            if packMXSAItems:
              for j in range(self.states.numReadsIterCoalescedMXSA):
                for n in range(instPerPackMXSA):
                  if packMXSAItems:
                    packINtemsMXSA[j].append(packMXSAItems.pop(0))
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
          while packMXSBItems or packBItems:
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
            if packMXSBItems:
              for j in range(self.states.numReadsIterCoalescedMXSB):
                for n in range(instPerPackMXSB):
                  if packMXSBItems:
                    packINtemsMXSB[j].append(packMXSBItems.pop(0))
                  else:
                    break

        for j in range(maxReadIterCoal):
          if schedulePackConsiderMetadata:
            packItemsA    += packINtemsA.pop(0)
            packItemsB    += packINtemsB.pop(0)
            packItemsMXSA += packINtemsMXSA.pop(0)
            packItemsMXSB += packINtemsMXSB.pop(0)
            packItemsM    += packINtemsM.pop(0)
          elif not scheduleTF32Emu:
            packItems += packINtems.pop(0)

        if schedulePackConsiderMetadata:
          packItems = packItemsA + packItemsB + packItemsMXSA + packItemsMXSB + packItemsM

      packPreItemsThisLoop = packPreItems if iteration < isBarrier else []
      # no need to generate pack Pre code for next loop in NLL last case (no next loop)
      packPreItemsNextLoop = packPreItems if iteration >= isBarrier and (not NLLlast) else []
      packPreItems = packPreItemsThisLoop + packPreItemsNextLoop
      #####
      # Prepare pack Pre Code
      # So far, this is for TF32 Emulation wider local read tranpose code (v_swap_b32) only
      numPackPre = len(packPreItems)
      doPackPreSchedule = numPackPre > 0
      addWaitForPack = True
      if doPackPreSchedule:
        startPrePackIndex = min(self.states.numMfmaForNextLoopLR + 1, numMfmaPerIter - 1)
        if self.states.doFullPackCodePrefetch and kernel["ForceUnrollSubIter"] and (iteration >= isBarrier or iteration == 1):
          # full pack code + subIter prefetch case
          # ItemsThisLoop case: start after local read
          # ItemsNextLoop case: local read is moved 1 iter ahead. Start pack scheduling from i = 0
          startPrePackIndex = 0
          if iteration == 1:
            # wait at start is not needed for B
            addWaitForPack = False
        numMfmaForPrePack = numMfmaPerIter - startPrePackIndex
        numPackPreInstPerMfma = numPackPre // numMfmaForPrePack
        numPackPreInstMod = numPackPre % numMfmaForPrePack
        mfmaAdjust = 0

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
              if vacancy["atMfmaIndex"] > self.states.sync1LdsMfmaIndex and oneBufferScheduling:
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

      insertedPackA = 0
      insertedPackB = 0
      insertedPackM = 0

      def hasDependency(lr: DSLoadInstruction, inst: Instruction) -> bool:
        lrDataReg = lr.dst

        if isinstance(inst, MFMAInstruction):
          srcRegs = [inst.a, inst.b,]
        elif isinstance(inst, SMFMAInstruction):
          srcRegs = [inst.a, inst.b, inst.metadata]
        elif isinstance(inst, MXMFMAInstruction):
          srcRegs = [inst.a, inst.b, inst.mxsa, inst.mxsb,]
        else:
          if hasattr(inst, 'srcs'):
            srcRegs = inst.srcs
          else:
            srcRegs = []

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
      finalCountA = 0
      finalCountB = 0
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
          elif kernel["EnableMatrixInstruction"] and self.do["OptimizeNumItersPLR0"] and not scheduleTF32Emu:
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
            localReadsIssuedInThisIter += countWeightedLocalRead(item)
            if (i == 0):
              localReadsWaitcnt += countWeightedLocalRead(item)
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
          # If DTLA and DTLB we can ignore this, since the global reads will be scheduled following
          # the local reads, and there is an lgkmcnt(0) to wait for the remaining local reads
          if not ((iteration == isBarrier - 1) and (kernel["DirectToLdsA"] and kernel["DirectToLdsB"])):
            self.states.overflowedResources = 5
        for j in range(readLeft):
          if localReadItemsThisLoop:
            item = localReadItemsThisLoop.pop(0)
            iterCode.add(item)
            localReadsIssuedInThisIter += countWeightedLocalRead(item)
            if (i == 0):
              localReadsWaitcnt += countWeightedLocalRead(item)
        if not localReadItemsThisLoop and latencyLeft > 0 and iteration < isBarrier and \
            not(mfmaIndex > self.states.sync1LdsMfmaIndex and oneBufferScheduling):
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
              (countGlobalRead(globalReadCode) or kernel["PrefetchGlobalRead"] >= 2):
            loadModule = globalReadCode.popFirstItem()
            iterCode.add(loadModule)
            self.states.scheduledGRInstCounts += countGlobalRead(loadModule)
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
          _barrier = SBarrier()
          barrier.add(_barrier)
          iterCode.add(barrier)

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] >= 2:
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
              self.states.scheduledGRInstCounts += countGlobalRead(writeItem) # PGR2 case GR is in localWriteCode
              # if there is localWrite at first mfma, need to skip it in waitcnt.
              if i == 0:
                skipLocalWriteWaitcnt += countWeightedLocalWrite(writeItem)
              if not localReadItemsThisLoop:
                self.states.perIterLocalWriteCanSkip[iteration] += countWeightedLocalWrite(writeItem)
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
              skipLocalWriteWaitcnt += countWeightedLocalWrite(writeItem)
            if not localReadItemsThisLoop:
              self.states.perIterLocalWriteCanSkip[iteration] += countWeightedLocalWrite(writeItem)

        ####
        # scheduled pointer
        ####
        # pointerLWCode with sgpr can break GlobalReadInc because s_xor_b32 changes SCC
        # generate pointerLWCode after GlobalReadInc to avoid incorrect GlobalReadInc
        LocalWriteUseSgpr = kernel["LocalWriteUseSgprA"] or kernel["LocalWriteUseSgprB"]
        if ((not LocalWriteUseSgpr) and mfmaIndex == self.states.lwEndMfmaIndex) or \
           (((mfmaIndex >= self.states.lwEndMfmaIndex and globalReadCode.itemsSize() == 0) or i == numMfmaPerIter - 1) and \
            (not pointerLWCodeInserted)):
          iterCode.add(pointerLWCode)
          pointerLWCodeInserted = True
        # schedule PointerLRCode into multiple MFMA (only v operation case (means excludes IncLdsBufSwitch))
        if lenPointerLRCode > 2 and not self.states.IncLdsBufSwitch:
          if len(localReadItemsThisLoop) == 0:
            if i >= numMfmaPerIter - lenPointerLRCode:
              while PointerLRCodeItems:
                item = PointerLRCodeItems.pop(0)
                iterCode.add(item)
                if isinstance(item, Instruction):
                  break
            if i == numMfmaPerIter - 1:
              while PointerLRCodeItems:
                iterCode.add(PointerLRCodeItems.pop(0))
        elif i == numMfmaPerIter - 1:
          iterCode.add(pointerLRCode)

        ####
        # scheduled sync
        ####
        if mfmaIndex == self.states.syncPlrMfmaIndex and self.states.numItersPLR:
          # scheduleGROverBarrier case, add GR wait at barrier sync with the number of GR already scheduled
          if self.states.scheduleGROverBarrier and (not isNLLorNGLL):
            numWait = kernel["PrefetchGlobalRead"] - 2
            iterCode.add(self._wait(kernel, tPA, tPB, numWait, -1, -1, "wait for previous set of global reads", \
                                    skipGlobalReadInst=self.states.scheduledGRInstCounts))
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
            # doPackPreSchedulingNextLoop case(except for doPackPreSchedulingNextLoop), schedule local read for next loop from i = 1
            startLR = numMfmaPerIter - numMfmaForLR
            if self.states.doPackPreSchedulingNextLoop:
              startLR = min(numMfmaPerIter -1 , (self.states.syncPlrMfmaIndex % numMfmaPerIter) + 1)
            if i < startLR:
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
              localReadsWaitcnt += countWeightedLocalRead(item)

        ####
        # scheduled wait localReads
        ####
        subIterMxNeedsDependentLrWait = (
          kernel["ForceUnrollSubIter"] and self.states.version == (12,5,0)
          and (kernel["ProblemType"]["MXBlockA"] or kernel["ProblemType"]["MXBlockB"])
        )
        if ((self.states.numItersPLR == 0 and self.do["OptimizeNumItersPLR0"]) or subIterMxNeedsDependentLrWait) \
            and kernel["EnableMatrixInstruction"] and not scheduleTF32Emu:
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
          if (kernel["UnrollMajorLDSB"] and not (kernel["ProblemType"]["DataTypeB"].isAnyFloat8() and kernel["ConvertAfterDS"])) and not kernel["UseF32XEmulation"]:
            if iteration == 0 and i == (kernel["MIWaveTileA"] // kernel["numSubTiles"]):
              # add 1 more waitcnt before using ds read data
              waitCode2 = deepcopy(waitCode)
              waitCode2.dscnt = localReadsIssuedInThisIter
              iterCode.add(waitCode2)
          if i == 0:
            # skip generating waitcnt at i==0 in doPackPreSchedule case
            if self.states.doPackPreSchedulingThisLoop:
              # doPackPreSchedulingThisLoop case, local read wait is already done for pre Pack.
              pass
            elif self.states.doPackPreSchedulingNextLoop and mfmaIndex == 0:
              # doPackPreSchedulingNextLoop case, we can skip only mfmaIndex == 0 (for prefetch only)
              pass
            else:
              iterCode.add(waitCode)


        ####
        # scheduled pack pre for this loop / next loop
        ####
        if doPackPreSchedule and packPreItems and i >= startPrePackIndex:
          if i == startPrePackIndex and addWaitForPack:
            iterCode.add(SWaitCnt(dscnt=0, comment="wait for current LocalRead packing"))
          count = 0
          adjust = 0
          mfmaCount = 0
          # add mfmaAdjust and reset
          adjust += mfmaAdjust
          mfmaAdjust = 0
          if numPackPreInstMod > 0:
            adjust += 1
            numPackPreInstMod -= 1
          numInst = numPackPreInstPerMfma + adjust
          while packPreItems and count < numInst:
            inst = packPreItems.pop(0)
            if isinstance(inst, MFMAInstruction):
              mfmaCount += 1
            else:
              mfmaCount = 0
            if count == numInst - 1:
              # if no mfma and next is mfma
              #  case 1: numInst > 1, push back
              #  case 2: numInst == 1, add all continuous mfma + 1 inst
              if mfmaCount == 0 and packPreItems:
                instNext = packPreItems.pop(0)
                if isinstance(instNext, MFMAInstruction):
                  if numInst > 1:
                    # push back next
                    packPreItems.insert(0, instNext)
                    # push back current inst
                    packPreItems.insert(0, inst)
                    # go out of while loop
                    break
                  else:
                    # mfma and numInst == 1 case, add all mfma
                    iterCode.add(inst)
                    inst = instNext
                    mfmaCount += 1
                else:
                  # next is not mfma
                  # push back next
                  packPreItems.insert(0, instNext)
              # last inst to schedule
              # we would not like to put mfma at the end
              while mfmaCount > 0:
                iterCode.add(inst)
                inst = packPreItems.pop(0)
                if isinstance(inst, MFMAInstruction):
                  mfmaCount += 1
                else:
                  # pop 1 more
                  iterCode.add(inst)
                  inst = packPreItems.pop(0)
                  mfmaCount = 0
            count += 1
            iterCode.add(inst)

        ####
        # scheduled pack
        ####
        _instPerPackA = 0
        _instPerPackMXSA = 0
        _instPerPackMXSB = 0
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
          _instPerPackMXSA = instPerPackMXSA if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packMXSAIdx += _instPerPackMXSA
          _instPerPackMXSB = instPerPackMXSB if i % kernel["MIWaveTileA"] == 0 else 0
          packMXSBIdx += _instPerPackMXSB
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
            packMXSAIdx = packMXSAIdx if ("MX" in tPA) and (not kernel["UnrollMajorLDSMXSA"]) else 0
            packMXSBIdx = packMXSBIdx if ("MX" in tPB) and (not kernel["UnrollMajorLDSMXSB"]) else 0
          else:
            packAIdx = packAIdx if tPA["localReadInstruction"].blockWidth == 0.25 else 0
            packBIdx = packBIdx if tPB["localReadInstruction"].blockWidth == 0.25 else 0
            packMXSAIdx = 0
            packMXSBIdx = 0
          numPack = (packAIdx + packBIdx + packMXSAIdx + packMXSBIdx)
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            packMIdx = packMIdx if not kernel["UnrollMajorLDSMetadata"] else 0
            numPack += packMIdx
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u, packMIdx:%u" %(packAIdx,packBIdx,packMIdx))
          else:
            iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))

          if not schedulePackConsiderMetadata:
              insertABcount = 0
              isLastNop = False
              isWmmaXf32Direct = kernel["UseF32XEmulation"] and kernel.get("UseDirect32XEmulation", False) and not kernel["UseMFMAF32XEmulation"]
              if isWmmaXf32Direct:
                # XF32 direct emulation with multigroup WMMA V3: place ALL pack
                # items (including the rearrange swaps) before the first WMMA to
                # ensure all bf16_hi and bf16_lo values are fully packed.
                while packItems:
                  iterCode.add(packItems.pop(0))
                  curPackIdx += 1
              elif kernel["UseF32XEmulation"]:
                if instPerPackA > 0 or instPerPackB == 0 and packItems:
                  tmp = []
                  num, _ = self._packItemsConditional(instPerPackA, packItems, tmp, ["__TF32_1_A", "__TF32_2_A"])
                  if num > 0:
                    insertABcount += 1
                    instPerPackA -= num
                  if not firstDone and instPerPackB == 0:
                    num, _ = self._packItemsConditional(instPerPackA, packItems, tmp, ["__TF32_1_A", "__TF32_2_A"])
                    if num > 0:
                      insertABcount += 1
                      instPerPackA -= num
                  firstDone = True
                  for n in tmp:
                    isLastNop = isinstance(n, SNop)
                    iterCode.add(n)
              else:
                # we put 2 pack in each mfma
                for j in range(instPerPackA):
                  if packItems:
                    iterCode.add(packItems.pop(0))
                    curPackIdx += 1
              if not isWmmaXf32Direct:
                for j in range(instPerPackMXSA):
                  if packItems:
                    iterCode.add(packItems.pop(0))
                    curPackIdx += 1
                for j in range(instPerPackMXSB):
                  if packItems:
                    iterCode.add(packItems.pop(0))
                    curPackIdx += 1
                if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
                  for j in range(ceil(instPerPackM)):
                    if packItems:
                      iterCode.add(packItems.pop(0))
                      curPackIdx += 1
                if kernel["UseF32XEmulation"]:
                  if instPerPackB > 0 or instPerPackA == 0 and packItems:
                    tmp = []
                    num, _ = self._packItemsConditional(instPerPackB, packItems, tmp, ["__TF32_1_B", "__TF32_2_B"])
                    if num > 0:
                      insertABcount += 1
                      instPerPackB -= num
                    if not firstDone and instPerPackA == 0:
                      num, _ = self._packItemsConditional(instPerPackB, packItems, tmp, ["__TF32_1_B", "__TF32_2_B"])
                      if num > 0:
                        insertABcount += 1
                        instPerPackB -= num
                    firstDone = True
                    for n in tmp:
                      isLastNop = isinstance(n, SNop)
                      iterCode.add(n)
                else:
                  for j in range(instPerPackB):
                    if packItems:
                      iterCode.add(packItems.pop(0))
                      curPackIdx += 1
              # since packed register need to wait 2 quad cycle to finish packing
              # we insert pack instruction if we can, or s_nop
              while curPackIdx < numPack+2 and not kernel["UseF32XEmulation"]:
                if packItems:
                  iterCode.add(packItems.pop(0))
                  curPackIdx += 1
                else:
                  iterCode.add(SNop(waitState=1, comment="VALU packing writes to be consumed by matrix instruction"))
                  curPackIdx += 1
                  break
              if kernel["UseF32XEmulation"] and not isLastNop:
                # HACK add dummy waits btween swap and mfmas. TODO: improve pack scheduling to avoid this
                numDummy = 0 if kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16 else 1
                # nop insertion for MFMA is done in _interleavePackAB
                #if kernel["UseMFMAF32XEmulation"] and not final:
                #  # only one side remaining case, add dummy by 2 sicne we do not interleave PackA and PackB
                #  if insertABcount == 1:
                #    numDummy += 2
                #for numd in range(numDummy):
                iterCode.add(SNop(waitState=numDummy - 1, comment="VALU packing writes to be consumed by matrix instruction (HACK)"))
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
            iterCode.addComment0("pack scheduling: curPackIdx:%u, numPack:%u, instPackLast:%s" % (curPackIdx, numPack, instPackLast))
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
              remainLatency = 2 if len(instPackLast) > 0 else 0
              # Per-type pop counters: track how many packs of each type have been
              # popped (= how many extras of that type have already been accounted for).
              popCountA = popCountB = popCountM = 0
              while len(instPackLast):
                instLast = instPackLast.pop()
                # extraPackX <= popCountX means we've run out of the extras of type X
                # and the current pack produces a register for the coming SMFMAC instruction.
                # Hence, we need to insert an extra pack or s_nop between
                # the current pack instruction and the coming SMFMAC instruction.
                if instLast == "A":
                  extraPackA = numPackedA - packAIdx
                  if extraPackA <= popCountA:
                    break
                  popCountA += 1
                elif instLast == "M":
                  extraPackM = numPackedM - packMIdx
                  if extraPackM <= popCountM:
                    break
                  popCountM += 1
                elif instLast == "B":
                  extraPackB = numPackedB - packBIdx
                  if extraPackB <= popCountB:
                    break
                  popCountB += 1
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
          if kernel["PrefetchGlobalRead"] >= 2:
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
      assert 0, "Unsupported scheduleIterAlg=%u"%scheduleIterAlg

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
        dataAtIterMXSA = iteration//self.states.numIterPerCoalescedReadMXSA - self.states.numItersPLR if kernel["ProblemType"]["MXBlockA"] else (-self.states.numItersPLR)
        dataAtIterMXSB = iteration//self.states.numIterPerCoalescedReadMXSB - self.states.numItersPLR if kernel["ProblemType"]["MXBlockB"] else (-self.states.numItersPLR)
        hasMetadata = kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]
        dataAtIterMetadata = iteration//self.states.numIterPerCoalescedReadMetadata - self.states.numItersPLR if hasMetadata else (-self.states.numItersPLR)
        maxDataAtIter = max(dataAtIterA, dataAtIterB, dataAtIterMXSA, dataAtIterMXSB, dataAtIterMetadata)

        numReadsIterA = min(iteration+1, kernel["LoopIters"]//self.states.numIterPerCoalescedReadA - self.states.numItersPLR)
        numReadsIterB = min(iteration+1, kernel["LoopIters"]//self.states.numIterPerCoalescedReadB - self.states.numItersPLR)
        numReadsIterMXSA = min(iteration+1, kernel["LoopIters"]//self.states.numIterPerCoalescedReadMXSA - self.states.numItersPLR) if kernel["ProblemType"]["MXBlockA"] else 0
        numReadsIterMXSB = min(iteration+1, kernel["LoopIters"]//self.states.numIterPerCoalescedReadMXSB - self.states.numItersPLR) if kernel["ProblemType"]["MXBlockB"] else 0
        numReadsIterMetadata = min(iteration+1, kernel["LoopIters"]//self.states.numIterPerCoalescedReadMetadata - self.states.numItersPLR) if hasMetadata else 0
        maxNumberReadIter = max(numReadsIterA, numReadsIterMXSA, numReadsIterB, numReadsIterMXSB, numReadsIterMetadata)

        skipReadsIterA = numReadsIterA - dataAtIterA - 1 if not dataAtIterA < maxDataAtIter else 0
        skipReadsIterMXSA = numReadsIterMXSA - dataAtIterMXSA - 1 if not dataAtIterMXSA < maxDataAtIter else 0
        skipReadsIterB = numReadsIterB - dataAtIterB - 1 if not dataAtIterB < maxDataAtIter else 0
        skipReadsIterMXSB = numReadsIterMXSB - dataAtIterMXSB - 1 if not dataAtIterMXSB < maxDataAtIter else 0
        skipReadsIterMetadata = numReadsIterMetadata - dataAtIterMetadata - 1 if (hasMetadata and not dataAtIterMetadata < maxDataAtIter) else 0

        # numPrefetchIter : in this loop, number of prefetch iteration we have read (data used in next loop)
        # currently we have localReadA and localReadB if iteration >= isBarrier
        # some case will not have localReads if PGR=0 or NoLoadLoop
        # known bug: wider localread + numItersPLR>1 may have chance to fail.
        numPrefetchIter = (iteration//(kernel["LoopIters"]-self.states.numItersPLR))*((iteration+1)-(kernel["LoopIters"]-self.states.numItersPLR)) if kernel["PrefetchGlobalRead"] else 0
        numPrefetchIter = 0 if iteration >= isBarrier and not hasLocalRead else numPrefetchIter
        skipReadsIterA += numPrefetchIter
        skipReadsIterB += numPrefetchIter
        skipReadsIterMXSA += numPrefetchIter
        skipReadsIterMXSB += numPrefetchIter
        if hasMetadata:
          skipReadsIterMetadata += numPrefetchIter

        # here the reads are prefetches so can skip them in the waitcnt
        # how many localreads can skip is based on how many iterations we prefetch.
        readFactorA = 2 if (tPA["localReadInstruction"].blockWidth == 6) else 1
        readFactorB = 2 if (tPB["localReadInstruction"].blockWidth == 6) else 1
        localReadsA = 0 if kernel["DirectToVgprA"] else self.states.numReadsPerIterA * skipReadsIterA * readFactorA
        localReadsB = 0 if kernel["DirectToVgprB"] else self.states.numReadsPerIterB * skipReadsIterB * readFactorB
        localReadsMXSA = 0 if ((not kernel["ProblemType"]["MXBlockA"]) or kernel["DirectToVgprMXSA"]) else self.states.numReadsPerIterMXSA * skipReadsIterMXSA
        localReadsMXSB = 0 if ((not kernel["ProblemType"]["MXBlockB"]) or kernel["DirectToVgprMXSB"]) else self.states.numReadsPerIterMXSB * skipReadsIterMXSB
        localReadsMetadata = self.states.numReadsPerIterMetadata * skipReadsIterMetadata if hasMetadata else 0
        # HalfPLR: only half of ds_read is issued before wmma
        localReadsA = localReadsA // 2 if kernel["HalfPLRA"] else localReadsA
        localReadsB = localReadsB // 2 if kernel["HalfPLRB"] else localReadsB
        localReads += (localReadsA + localReadsB + localReadsMXSA + localReadsMXSB + localReadsMetadata)
        # some of localReads is interleaved after waitcnt in SIA3
        if scheduleIterAlg == 3 and self.states.numItersPLR and (iteration < maxNumberReadIter or numPrefetchIter) and\
          not kernel["UseF32XEmulation"]:
          if ((iteration < numReadsIterA and not dataAtIterA < maxDataAtIter) or numPrefetchIter) and (not kernel["DirectToVgprA"]):
            localReads -= self.states.numReadsPerIterA * readFactorA
          if kernel["ProblemType"]["MXBlockA"]:
            if ((iteration < numReadsIterMXSA and not dataAtIterMXSA < maxDataAtIter) or numPrefetchIter) and (not kernel["DirectToVgprA"]):
              localReads -= self.states.numReadsPerIterMXSA
          if ((iteration < numReadsIterB and not dataAtIterB < maxDataAtIter) or numPrefetchIter) and (not kernel["DirectToVgprB"]):
            localReads -= self.states.numReadsPerIterB * readFactorB
          if kernel["ProblemType"]["MXBlockB"]:
            if ((iteration < numReadsIterMXSB and not dataAtIterMXSB < maxDataAtIter) or numPrefetchIter) and (not kernel["DirectToVgprB"]):
              localReads -= self.states.numReadsPerIterMXSB
          if hasMetadata:
            if ((iteration < numReadsIterMetadata and not dataAtIterMetadata < maxDataAtIter) or numPrefetchIter):
              localReads -= self.states.numReadsPerIterMetadata

          localReads += localReadsWaitcnt
          if iteration == 0 and kernel["UnrollMajorLDSB"] and not (kernel["ProblemType"]["DataTypeB"].isAnyFloat8() and kernel["ConvertAfterDS"]):
            # We issued LR with A[0]->B[0]->A[1:]->B[1:] order.
            # We need to calculate how many B[N:] needed by 1st mfma.
            # If not UnrollMajorLDSB, we have packB which will be issued ahead.
            # If ConvertAfterDS and DataTypeB=FP8, we have cvtB which will be issued ahead.
            localReadsNotWaited = (self.states.numReadsPerIterB//kernel["InnerUnroll"] - self.states.numReadsPerUnrollB) * readFactorB
            if localReadsNotWaited > 0:
              localReads += localReadsNotWaited
        elif kernel["UseF32XEmulation"]:
          localReads = 0

        dscnt += localReads
        iterCode.addComment0("numPrefetchIter=%u" % numPrefetchIter)
        iterCode.addComment0("dataAtIterA=%u numReadsIterA=%u skipReadsIterA=%u readsPerIterA=%u" % (dataAtIterA, numReadsIterA, skipReadsIterA, self.states.numReadsPerIterA))
        iterCode.addComment0("dataAtIterB=%u numReadsIterB=%u skipReadsIterB=%u readsPerIterB=%u" % (dataAtIterB, numReadsIterB, skipReadsIterB, self.states.numReadsPerIterB))
        if kernel["ProblemType"]["MXBlockA"]:
          iterCode.addComment0("dataAtIterMXSA=%u numReadsIterMXSA=%u skipReadsIterMXSA=%u readsPerIterMXSA=%u" % (dataAtIterMXSA, numReadsIterMXSA, skipReadsIterMXSA, self.states.numReadsPerIterMXSA))
        if kernel["ProblemType"]["MXBlockB"]:
          iterCode.addComment0("dataAtIterMXSB=%u numReadsIterMXSB=%u skipReadsIterMXSB=%u readsPerIterMXSB=%u" % (dataAtIterMXSB, numReadsIterMXSB, skipReadsIterMXSB, self.states.numReadsPerIterMXSB))
        if hasMetadata:
          iterCode.addComment0("dataAtIterMetadata=%u numReadsIterMetadata=%u skipReadsIterMetadata=%u readsPerIterMetadata=%u" % (dataAtIterMetadata, numReadsIterMetadata, skipReadsIterMetadata, self.states.numReadsPerIterMetadata))
        if scheduleIterAlg == 0 or scheduleIterAlg == 1:
          for i in range (max(dataAtIterA,dataAtIterB),iteration+1):
            localWrites += countWeightedLocalWrite(self.codes.perIterLocalWrite[i][1])
        # ScheduleIterAlg=2, localwrite is after waitCnt, no need to count it's current iteration.
        if scheduleIterAlg == 3:
          for i in range (max(dataAtIterA,dataAtIterB)+1,iteration):
            localWrites += countWeightedLocalWrite(self.codes.perIterLocalWrite[i][1])
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
          localReads  = countWeightedLocalRead(item)
          localWrites = countWeightedLocalWrite(item)
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

    # TDM + PGR>=2 + NoLdsWriteCode + SIA in {2,3} case: the SIA2/SIA3 schedulers
    # route PGR>=2 global reads through itemsGRToSchedLater and pair them with
    # local writes inside schedLocalWrite. TDM has no local writes
    # (NoLdsWriteCode=True) so schedLocalWrite never iterates and the GR items
    # are silently dropped. Placing them earlier in iterCode (e.g. via
    # perIterGlobalRead[swapIter]) would also land them BEFORE pointerLWCode (the
    # LDS swap) and BEFORE the barrier in syncCode, causing the new
    # tensor_load_to_lds to write to the bank still being consumed by ds_loads.
    # Append the deferred TDM tensor_load_to_lds at the very end of the swap
    # iter's iterCode so the load uses the post-swap descriptor and targets the
    # bank whose data was just fully consumed; the next outer cycle's syncPlr
    # wait/barrier waits on it before its ds_loads.
    # TODO: This is intentionally narrow rather than a canonical scheduler
    # phase. A cleaner long-term model would add an explicit post-swap global
    # read lane shared by SIA0/SIA2/SIA3, but doing that first would touch the
    # broader GR/LW scheduling contract. Keep this local until the post-swap GR
    # phase can be introduced and validated across DTL/DTV/sparse scheduling.
    if scheduleIterAlg in (2, 3) \
        and kernel["enableTDMA"] and kernel["enableTDMB"] \
        and kernel["NoLdsWriteCode"] and kernel["PrefetchGlobalRead"] >= 2:
      if scheduleIterAlg == 3:
        swapIter = self.states.lwEndMfmaIndex // self.states.numMfmaPerIter
      else:
        swapIter = kernel["LoopIters"] - self.states.numItersPLR - 1
      if iteration == swapIter:
        tdmInj = Module("tdmInjectGRForPGR2")
        tdmInj.addComment1("tdm: inject GR for PGR>=2 (scheduler drops these when no LWs to pair with)")
        for item in self.codes.globalReadA.middle.items():
          tdmInj.add(deepcopy(item))
        for item in self.codes.globalReadMXSA.middle.items():
          tdmInj.add(deepcopy(item))
        for item in self.codes.globalReadMXSB.middle.items():
          tdmInj.add(deepcopy(item))
        for item in self.codes.globalReadB.middle.items():
          tdmInj.add(deepcopy(item))
        if tdmInj.itemsSize() > 1:
          iterCode.add(tdmInj)

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

    # Subtile computes wave offsets from vgpr("Serial") inline, so WaveIdx
    # is only needed during kernel-arg init.  Free it before graWorkGroup
    # to reduce SGPR pressure for StreamK temp allocations.
    if kernel.get("UseSubtileImpl") and (kernel["enableTDMA"] or kernel["enableTDMB"]) \
        and "WaveIdx" in self.sgprs and not kernel.get("ClusterBarrier"):
      module.add(self.undefineSgpr("WaveIdx"))

    # work-group assignments
    module.addComment1("global read addresses: work-group") # is this comment needed?
    if not forceNoTileCode:
      module.add(self.graWorkGroup(kernel, tensorParametersA, tensorParametersB))


    self.dontAppendCode = forceNoTileCode

    tPM = tensorParametersA["tpsMetadata"]
    tPMRef = tensorParametersA

    if kernel["ProblemType"]["Sparse"] == 2:
      tPM = tensorParametersB["tpsMetadata"]
      tPMRef = tensorParametersB

    if kernel["StreamK"] != 0:
      if not kernel["UseSubtileImpl"]:
        module.add(self.localReadAddresses(kernel, tensorParametersA, tensorParametersB, tPM))
        module.add(self.localWriteAddresses(kernel, tensorParametersA, tensorParametersB, tPM))

    if kernel["PrefetchGL2"]:
      module.add(self.gl2PrefetchCalcAddr(kernel, tensorParametersA, tensorParametersB))

    tdmA: bool = kernel["enableTDMA"]
    tdmB: bool = kernel["enableTDMB"]
    tdmMetadata: bool = kernel["enableTDMMetadata"]
    tdmInited: bool = False
    tdmMetadataInited: bool = False

    # TODO: This can probably be moved later, after setupnewtile
    # Always release GR-related SGPRs from the pool. regardless of whether
    # it is a TDM or non-TDM kernel.
    module.add(self.removeGRSrdVariableSgprsFromPool(kernel))

    # tile assignments
    if not kernel["UseSubtileImpl"]:
      # Wave-separated TDM increment: subtile uses per-wave descriptors and
      # does not need parity-based increment selection.
      if tdmA and tdmB and kernel["NumWaves"] > 1 and not kernel["UseSubtileImpl"]:
        module.add(self.initTDMDescriptorWaveSeparated(kernel, tensorParametersA, tensorParametersB))
        if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
          module.add(self.initTDMDescriptorWaveSeparated(kernel, tensorParametersA["MX"], tensorParametersB["MX"]))
        module.add(self.tdmGlobalOffsetWaveSeparated(kernel, tensorParametersA, tensorParametersB))
        if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
          module.add(self.tdmGlobalOffsetWaveSeparated(kernel, tensorParametersA["MX"], tensorParametersB["MX"]))
        tdmInited = True

      # Tile offset assignment A(MXSA)
      #TODO: TDM handles MXSA and MXSB
      if tdmA:
        if not tdmInited:
          module.add(self.tdmGlobalOffset(kernel, tensorParametersA))
          if kernel["UseSubtileImpl"]:
            module.add(initTDMDescriptorSubtile(self, kernel, tensorParametersA))
          else:
            module.add(self.initTDMDescriptor(kernel, tensorParametersA))
      else:
        module.addComment1("global read addresses: tile offset assignment a")
        module.add(self.graTileAssignment(kernel, tensorParametersA))
      if kernel["ProblemType"]["MXBlockA"]:
        if not tdmA:
          module.addComment1("global read addresses: tile offset assignment mxsa")
          module.add(self.graTileAssignment(kernel, tensorParametersA["MX"]))
      # Tile offset assignment Metadata
      if kernel["ProblemType"]["Sparse"]:
        if tdmMetadata:
          if not tdmMetadataInited:
            # TODO: TDM global offset for metadata
            module.add(self.tdmGlobalOffset(kernel, tPM))
            module.add(self.initTDMDescriptor(kernel, tPM))
            tdmMetadataInited = True
        else:
          module.addComment1("global read addresses: tile offset assignment metadata")
          if kernel["DirectToVgprSparseMetadata"]:
            # calculate tile assignment and store into each vgprGlobalReadOffsetMetadata
            module.add(self.graMetadataTileAssignment(kernel, tPMRef))
          else:
            module.add(self.graTileAssignment(kernel, tPM))
      # Tile offset assignment B(MXSB)
      if kernel["ProblemType"]["MXBlockB"]:
        if not tdmB:
          module.addComment1("global read addresses: tile offset assignment mxsb")
          module.add(self.graTileAssignment(kernel, tensorParametersB["MX"]))
      if tdmB:
        if not tdmInited:
          module.add(self.tdmGlobalOffset(kernel, tensorParametersB))
          if kernel["UseSubtileImpl"]:
            module.add(initTDMDescriptorSubtile(self, kernel, tensorParametersB))
          else:
            module.add(self.initTDMDescriptor(kernel, tensorParametersB))
      else:
        module.addComment1("global read addresses: tile offset assignment b")
        module.add(self.graTileAssignment(kernel, tensorParametersB))

      # Unroll assignment A(MXSA)
      if not tdmA:
        module.addComment1("global read addresses: unroll assignment a")
        module.add(self.graUnrollAssignment(kernel, tensorParametersA))
      if kernel["ProblemType"]["MXBlockA"]:
        if not tdmA:
          module.addComment1("global read addresses: unroll assignment mxsa")
          module.add(self.graUnrollAssignment(kernel, tensorParametersA["MX"]))
      # Unroll assignment Metadata
      if not tdmMetadata and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.addComment1("global read addresses: unroll assignment metadata")
        module.add(self.graUnrollAssignment(kernel, tPM))
      # Unroll assignment B(MXSB)
      if kernel["ProblemType"]["MXBlockB"]:
        if not tdmB:
          module.addComment1("global read addresses: unroll assignment mxsb")
          module.add(self.graUnrollAssignment(kernel, tensorParametersB["MX"]))
      if not tdmB:
        module.addComment1("global read addresses: unroll assignment b")
        module.add(self.graUnrollAssignment(kernel, tensorParametersB))

      # other free indices
      if not (tdmA or tdmB):
        if kernel["ProblemType"]["NumIndicesC"] > 2:
          module.addComment1("global read addresses: other free assignments")
          module.add(self.graOtherFreeAssignments())

        # other summation indices
        if self.states.otherSummations:
          module.addComment1("global read addresses: other summation assignments")
          module.add(self.graOtherSummationAssignments(kernel))

      # Tile offsets A(MXSA)
      if not tdmA:
        module.addComment1("global read addresses: tile offsets a")
        module.add(self.graTileOffsets(kernel, tensorParametersA))
      if kernel["ProblemType"]["MXBlockA"]:
        module.addComment1("global read addresses: tile offsets mxsa")
        if not tdmA:
          module.add(self.graTileOffsets(kernel, tensorParametersA["MX"]))
      # Tile offsets Metadata
      if not tdmMetadata and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.addComment1("global read addresses: tile offsets metadata")
        # Using A or B's margin to instead Metadata's margin
        module.add(self.graTileOffsets(kernel, tPM, tPMRef["glvw"] if tPMRef["rtv"] else 1))
      # Tile offsets B(MXSB)
      if kernel["ProblemType"]["MXBlockB"]:
        module.addComment1("global read addresses: tile offsets mxsb")
        if not tdmB:
          module.add(self.graTileOffsets(kernel, tensorParametersB["MX"]))
      if not tdmB:
        module.addComment1("global read addresses: tile offsets b")
        module.add(self.graTileOffsets(kernel, tensorParametersB))

      # Unroll offsets A(MXSA)
      if not tdmA:
        module.addComment1("global read addresses: unroll offsets a")
        module.add(self.graUnrollOffsets(kernel, tensorParametersA))
      if kernel["ProblemType"]["MXBlockA"]:
        module.addComment1("global read addresses: unroll offsets mxsa")
        if not tdmA:
          module.add(self.graUnrollOffsets(kernel, tensorParametersA["MX"]))
      # Unroll offsets Metadata
      if not tdmMetadata and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.addComment1("global read addresses: unroll offsets metadata")
        module.add(self.graUnrollOffsets(kernel, tPM))
      # Unroll offsets B(MXSB)
      if kernel["ProblemType"]["MXBlockB"]:
        module.addComment1("global read addresses: unroll offsets mxsb")
        if not tdmB:
          module.add(self.graUnrollOffsets(kernel, tensorParametersB["MX"]))

      if not tdmB:
        module.addComment1("global read addresses: unroll offsets b")
        module.add(self.graUnrollOffsets(kernel, tensorParametersB))

      # Free sgpr that will not be used
      if kernel["Multicast"] and kernel["TDMInst"] != 0:
        tdmA: bool = kernel["enableTDMA"]
        tdmB: bool = kernel["enableTDMB"]
        tdmM: bool = kernel["enableTDMMetadata"]
        if tdmA and tdmB and kernel["NumWaves"] > 1:
          module.add(self.undefineSgpr("MulticastMask"))
        else:
          module.add(self.undefineSgpr("MulticastMaskA"))
          module.add(self.undefineSgpr("MulticastMaskB"))
        if tdmM:
          module.add(self.undefineSgpr("MulticastMaskMetadata"))

      # tile edges
      if kernel["EdgeType"] == "ShiftPtr" and not tdmA and not tdmB:
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
        if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"]) and not forceNoTileCode and not kernel["UseGeneralizedNLCOneA"] \
          and not tensorParametersA["isSwizzled"]:
          module.addComment1("global read addresses: shift a")
          module.add(self.graShift(kernel, tensorParametersA))
          if tensorParametersA["is_sparse"] and kernel["DirectToVgprSparseMetadata"]:
            module.addComment1("global read addresses: shift metadata")
            module.add(self.graMetadataShift(kernel, tensorParametersA))
          if kernel["ProblemType"]["MXBlockA"]:
            module.addComment1("global read addresses: shift mxsa")
            module.add(self.graShiftMX(kernel, tensorParametersA["MX"], tensorParametersA))

        if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialMetadata"]) and not forceNoTileCode \
          and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          module.addComment1("global read addresses: shift metadata")
          # Using A's margin to instead Metadata's margin
          module.add(self.graShift(kernel, tPM, tPMRef["glvw"] if tPMRef["rtv"] else 1))

        if not (kernel["BufferLoad"] and  kernel["GuaranteeNoPartialB"]) and not forceNoTileCode and not kernel["UseGeneralizedNLCOneB"] \
          and not tensorParametersB["isSwizzled"]:
          module.addComment1("global read addresses: shift b")
          module.add(self.graShift(kernel, tensorParametersB))
          if tensorParametersB["is_sparse"] and kernel["DirectToVgprSparseMetadata"]:
            module.addComment1("global read addresses: shift metadata")
            module.add(self.graMetadataShift(kernel, tensorParametersB))
          if kernel["ProblemType"]["MXBlockB"]:
            module.addComment1("global read addresses: shift mxsb")
            module.add(self.graShiftMX(kernel, tensorParametersB["MX"], tensorParametersB))

      # addresses
      def releaseTensorTmpGprs(tP):
        self.vgprPool.checkIn(tP["gpr"]["lwoT"])
        tP["gpr"]["lwoT"] = None
        self.vgprPool.checkIn(tP["gpr"]["uReg2"])
        tP["gpr"]["uReg2"] = None

        self.vgprPool.checkIn(tP["gpr"]["uReg"])
        tP["gpr"]["uReg"] = None
        if "subIterReg" in tP["gpr"]:
          if tP["gpr"]["subIterReg"] is not None:
            self.vgprPool.checkIn(tP["gpr"]["subIterReg"])
          tP["gpr"]["subIterReg"] = None

      # addresses
      if not forceNoTileCode:
        # Addresses A(MXSA)
        if not tdmA:
          module.addComment1("global read addresses: addresses a")
          module.add(self.graAddresses(kernel, tensorParametersA))
        if not tdmA and kernel["ProblemType"]["MXBlockA"]:
          module.addComment1("global read addresses: addresses mxsa")
          module.add(self.graAddresses(kernel, tensorParametersA["MX"]))
        # Addresses Metadata
        if not tdmMetadata and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          module.addComment1("global read addresses: addresses metadata")
          module.add(self.graAddresses(kernel, tPM))
        # Addresses B(MXSB)
        if not tdmB and kernel["ProblemType"]["MXBlockB"]:
          module.addComment1("global read addresses: addresses mxsb")
          module.add(self.graAddresses(kernel, tensorParametersB["MX"]))
        if not tdmB:
          module.addComment1("global read addresses: addresses b")
          module.add(self.graAddresses(kernel, tensorParametersB))

      # workgroup SGPRs no longer needed
      if not tdmA or kernel["NumWaves"] < 2:
        module.add(self.removeGROffsetsVariableSgprsFromPool(kernel))

      # Final offsets A(MXSA)
      if not tdmA:
        module.addComment1("global read addresses: final offsets a")
        module.add(self.graFinalOffsets(kernel, tensorParametersA))
        # releaseTensorTmpGprs(tensorParametersA)
      if not tdmA and kernel["ProblemType"]["MXBlockA"]:
        module.addComment1("global read addresses: final offsets mxsa")
        module.add(self.graFinalOffsets(kernel, tensorParametersA["MX"]))
      if not tdmMetadata and kernel["ProblemType"]["Sparse"]:
        module.addComment1("global read addresses: final offsets metadata")
        if kernel["DirectToVgprSparseMetadata"]:
          module.add(self.graMetadataFinalOffsets(kernel, tPMRef))
        else:
          module.add(self.graFinalOffsets(kernel, tPM))
      # Final offsets B(MXSB)
      if not tdmB and kernel["ProblemType"]["MXBlockB"]:
        module.addComment1("global read addresses: final offsets mxsb")
        module.add(self.graFinalOffsets(kernel, tensorParametersB["MX"]))
      if not tdmB:
        module.addComment1("global read addresses: final offsets b")
        module.add(self.graFinalOffsets(kernel, tensorParametersB))
        # releaseTensorTmpGprs(tensorParametersB)

      self.dontAppendCode = False
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode

      # Add increment code
      gsuComponent = Component.GSU.find(self)
      module.add(gsuComponent.setupNewTile(self, kernel, tensorParametersA, tensorParametersB, tPM))

      #TODO: TDM wave separated
      if tdmA and tdmB and kernel["NumWaves"] > 1:
        module.add(self.tdmSetupIncrementWaveSeparated(kernel, tensorParametersA, tensorParametersB))

        if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
          module.add(self.tdmSetupIncrementWaveSeparated(kernel, tensorParametersA["MX"], tensorParametersB["MX"]))

        if kernel["StreamK"] > 0:
          module.add(self.tdmApplyStreamKOffsetWaveSeparated(kernel, tensorParametersA, tensorParametersB))
          if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
            module.add(self.tdmApplyStreamKOffsetWaveSeparated(kernel, tensorParametersA["MX"], tensorParametersB["MX"]))

        if not self.states.staggerUCode:
          module.add(self.releaseGlobalReadIncsSgprsAfterTdmWaveSep(kernel))

      # WaveIdx already freed for subtile (before graWorkGroup above)
      if (kernel["enableTDMA"] or kernel["enableTDMB"]) and not kernel["ClusterBarrier"] \
          and not kernel.get("UseSubtileImpl"):
        module.add(self.undefineSgpr("WaveIdx"))

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

      if not forceNoTileCode and self.states.staggerUCode:
        module.add(self.declareStaggerParms(kernel))
        # Calculate stagger A(MXSA)
        module.add(self.calculateStagger(kernel, tensorParametersA))
        if kernel["ProblemType"]["MXBlockA"]:
          module.add(self.calculateStagger(kernel, tensorParametersA["MX"]))
        if kernel["ProblemType"]["MXBlockB"]:
          module.add(self.calculateStagger(kernel, tensorParametersB["MX"]))
        # Calculate stagger Metadata
        if not tdmMetadata and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          module.add(self.calculateStagger(kernel,tPM))
        # Calculate stagger B(MXSB)
        module.add(self.calculateStagger(kernel, tensorParametersB))
      # LRO and LWA as assigned
      # init lds read pointers before each unrolled loop
      module.addComment0("local read addresses: init pointers a")
      module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
      if kernel["ProblemType"]["MXBlockA"]:
        module.addComment0("local read addresses: init pointers mxsa")
        module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA["MX"]))
      if kernel["ProblemType"]["MXBlockB"]:
        module.addComment0("local read addresses: init pointers mxsb")
        module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB["MX"]))
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.addComment0("local read addresses: init pointers metadata")
        module.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))
      module.addComment0("local read addresses: init pointers b")
      module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))
      if self.states.IncLdsBufSwitch:
        # IncLdsBufSwitch case, need to initialize local write inc register
        module.addComment0("local write addresses: reset inc")
        module.add(self.localWriteResetOffsets(kernel,  False, tensorParametersA))

      if ((kernel["DirectToLdsA"] or kernel["DirectToLdsB"])
          and not (tdmA and tdmB)
          and self.isPrefetchAcrossPersistentEnabled(kernel)):
        module.add(self.papDtlRestoreLdsBank(kernel, tensorParametersA, tensorParametersB))

      if (tdmA and tdmB and prod(kernel["MIWaveGroup"]) > 1
          and self.isPrefetchAcrossPersistentEnabled(kernel)):
        module.add(self.papTdmRestoreLdsBank(kernel, tensorParametersA, tensorParametersB))

      if self.do["executeToInitEnd"]:
        module.add(self.functionEnd(kernel, addLabel=False))

      ####################################
      # prefetch: unrolled loop prefix
      ####################################
      if kernel["PrefetchGlobalRead"]:
        # if DirectToVgpr is enabled and swapGlobalRead is true, swap the order of global read (B->A)
        tensorParameters1st = tensorParametersA
        tensorParameters2nd = tensorParametersB
        tdm1st, tdm2nd = kernel["enableTDMA"], kernel["enableTDMB"]
        tdmMetadata = kernel["enableTDMMetadata"]
        if self.isSwapGlobalReadOrderForDtvOrDtl(kernel, prefetch1=True):
          tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
          tdm1st, tdm2nd = tdm2nd, tdm1st
        pfi = 1 if kernel["PrefetchGlobalRead"] < 3 else kernel["PrefetchGlobalRead"] - 1
        module.addComment1("prefetch: global -> local")
        module.add(self.openSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=isOptNLL))
        usePrimedSkip = self.isPrefetchAcrossPersistentEnabled(kernel)
        lbl_prefetchPrimedMerge = Label(self.labels.getNameInc("SK_PrefetchPrimedMerge"), "")
        if usePrimedSkip:
          module.add(SCmpEQU32(src0=sgpr("SkPrefetchPrimed"), src1=0, comment="tail prefetch already issued first PGR group?"))
          module.add(SCBranchSCC0(labelName=lbl_prefetchPrimedMerge.getLabelName(), comment="skip first PGR group if primed"))
        moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters1st)
        module.add(replaceHolder(moduleTmp, 0))

        module.add(self.globalReadDo(kernel, 0, tensorParameters1st, tPM=tPM))
        # PAP+MX keeps MX G2L in the durable read range, so scale loads are part
        # of the same first-PGR handoff as main A/B. When SkPrefetchPrimed is
        # already set, the branch above skips this whole first-PGR group.
        if "MX" in tensorParameters1st:
          moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters1st["MX"], skipWait=True)
          module.add(replaceHolder(moduleTmp, 0))
          module.add(self.globalReadDo(kernel, 0, tensorParameters1st["MX"]))
        if "MX" in tensorParameters2nd:
          moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters2nd["MX"], skipWait=True)
          module.add(replaceHolder(moduleTmp, 0))
          module.add(self.globalReadDo(kernel, 0, tensorParameters2nd["MX"]))
        skip2ndWaitForDtl = kernel["DirectToLds%s"%tensorParameters1st["tensorChar"]]
        moduleTmp = self.directToLdsM0Update(kernel, 0, tensorParameters2nd, skip2ndWaitForDtl)
        module.add(replaceHolder(moduleTmp, 0))
        module.add(self.globalReadDo(kernel, 0, tensorParameters2nd, tPM=tPM))
        if tdmMetadata and kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]
          moduleTmp = self.directToLdsM0Update(kernel, 0, tPM)
          module.add(replaceHolder(moduleTmp, 0))
          module.add(self.globalReadDo(kernel, 0, tPM))
        if usePrimedSkip:
          module.add(lbl_prefetchPrimedMerge)
          module.add(SMovB32(dst=sgpr("SkPrefetchPrimed"), src=0, comment="clear after first PGR group merge"))
        tPA = tensorParametersA
        tPB = tensorParametersB
        if kernel["PrefetchGlobalRead"] == 2:
          # PGR2 + DTV case, skip GR inc
          if kernel["DirectToVgprA"]:
            tPA = None
          if kernel["DirectToVgprB"]:
            tPB = None
        module.add(self.globalReadIncrementAB(kernel, tPA, tPB, self.states.unrollIdx, pfi))
        # swap Tensor memToken
        self.states.ldsTensorTokenIdx = \
            self.states.memTokenLdsBuffer1 if self.states.ldsTensorTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

    module.addComment2("End setupNewTile")

    return module

  ##############################################################################
  # Setup next-tile PAP loads only.
  ##############################################################################
  def setupPrefetchAcrossPersistentLoads(self, kernel, tensorParametersA, tensorParametersB, isOptNLL=True):
    module = Module("setupPrefetchAcrossPersistentLoads")
    module.addComment2("Begin setupPrefetchAcrossPersistentLoads")

    if not kernel["PrefetchGlobalRead"]:
      module.addComment2("End setupPrefetchAcrossPersistentLoads")
      return module

    def usesCanonicalSrd(tP):
      tc = tP["tensorChar"]
      if tc in ["A", "MXSA"]:
        return not kernel["enableTDMA"]
      if tc in ["B", "MXSB"]:
        return not kernel["enableTDMB"]
      return True

    def usesShadowLimit(tP):
      tc = tP["tensorChar"]
      if tc in ["MXSA", "MXSB"]:
        return self.states.use64bShadowLimitMX
      return self.states.use64bShadowLimit

    # PAP borrows the current descriptor registers long enough to compute and
    # issue next-tile first-PGR loads. Restore them before current-tail code
    # resumes; only the issued loads and SkPrefetchPrimed are durable.
    def checkpointDescriptorState(tP, tmpSgpr):
      tc = tP["tensorChar"]
      module.addComment0("PAP: snapshot current %s descriptor" % tc)
      module.add(SMovB64(dst=sgpr(tmpSgpr + 0, 2), src=sgpr("Srd%s+0" % tc, 2), comment="checkpoint Srd%s[0:1]" % tc))
      if usesShadowLimit(tP):
        module.add(SMovB64(dst=sgpr(tmpSgpr + 2, 2), src=sgpr("ShadowLimit%s+0" % tc, 2), comment="checkpoint ShadowLimit%s" % tc))
      else:
        module.add(SMovB64(dst=sgpr(tmpSgpr + 2, 2), src=sgpr("Srd%s+2" % tc, 2), comment="checkpoint Srd%s[2:3]" % tc))

    def restoreShadowLimitSrd(tP, tmpSgpr):
      tc = tP["tensorChar"]
      module.add(SCmpEQU32(src0=sgpr("ShadowLimit%s+1" % tc), src1=0, comment="are we within 2^32?"))
      module.add(SCSelectB32(dst=sgpr("Srd%s+2" % tc), src0=sgpr("ShadowLimit%s+0" % tc), src1="BufferLimit", comment="Move shadow to real if we are within 2^32"))
      if self.states.version[:2] == (12, 5):
        # gfx125x encodes low record-count bits in Srd+1, so repack after
        # restoring Srd+2 from ShadowLimit.
        module.addComment0("Shift num records for gfx125x")
        module.add(SAndB32(sgpr(tmpSgpr), sgpr("Srd%s+2" % tc), 0x7F))
        module.add(SLShiftLeftB32(sgpr(tmpSgpr), 25, sgpr(tmpSgpr)))
        module.add(SAndB32(sgpr("Srd%s+1" % tc), sgpr("Srd%s+1" % tc), 0x1FFFFFF))
        module.add(SOrB32(sgpr("Srd%s+1" % tc), sgpr("Srd%s+1" % tc), sgpr(tmpSgpr)))
        module.add(SLShiftRightB32(sgpr("Srd%s+2" % tc), 7, sgpr("Srd%s+2" % tc)))

    def restoreDescriptorState(tP, tmpSgpr):
      tc = tP["tensorChar"]
      module.addComment0("PAP: restore current %s descriptor" % tc)
      module.add(SMovB64(dst=sgpr("Srd%s+0" % tc, 2), src=sgpr(tmpSgpr + 0, 2), comment="restore Srd%s[0:1]" % tc))
      if usesShadowLimit(tP):
        module.add(SMovB64(dst=sgpr("ShadowLimit%s+0" % tc, 2), src=sgpr(tmpSgpr + 2, 2), comment="restore ShadowLimit%s" % tc))
        restoreShadowLimitSrd(tP, tmpSgpr)
      else:
        module.add(SMovB64(dst=sgpr("Srd%s+2" % tc, 2), src=sgpr(tmpSgpr + 2, 2), comment="restore Srd%s[2:3]" % tc))

    def borrowedStaggerState():
      # calculateStagger mutates this state for the next tile. The current NLL
      # and tail path can still observe it after PAP, so treat it as borrowed.
      if not self.states.staggerUCode:
        return []

      state = [("StaggerUIter", 1), ("WrapUA", 2), ("WrapUB", 2)]
      isDTVAorB = kernel["DirectToVgprA"] != kernel["DirectToVgprB"]
      if kernel["PrefetchGlobalRead"] >= 2 and isDTVAorB:
        state.append(("StaggerUIterDTV", 1))
      if kernel["ProblemType"]["MXBlockA"]:
        state.append(("WrapUMXSA", 2))
      if kernel["ProblemType"]["MXBlockB"]:
        state.append(("WrapUMXSB", 2))
      if kernel["ProblemType"]["Sparse"]:
        state.append(("WrapUMetadata", 2))
      return state

    def copyBorrowedStaggerState(dstBase, srcBase, state, comment):
      offset = 0
      for name, size in state:
        for i in range(size):
          regName = name if i == 0 else "%s+%u" % (name, i)
          module.add(SMovB32(dst=sgpr(dstBase + offset + i) if dstBase is not None else sgpr(regName),
                             src=sgpr(srcBase + offset + i) if srcBase is not None else sgpr(regName),
                             comment="%s %s" % (comment, regName)))
        offset += size

    def checkpointBorrowedStaggerState(tmpSgpr, state):
      copyBorrowedStaggerState(tmpSgpr, None, state, "checkpoint")

    def restoreBorrowedStaggerState(tmpSgpr, state):
      copyBorrowedStaggerState(None, tmpSgpr, state, "restore")

    def emitTensorLoadUsesStagger(tP):
      return self.states.staggerUCode and kernel["BufferLoad"] and usesCanonicalSrd(tP)

    def tensorGlobalReadOffsetInfo(tP):
      tc = tP["tensorChar"]
      if tc == "A":
        return self.startVgprGlobalReadOffsetA, self.states.a.numVgprGlobalReadOffsets
      if tc == "B":
        return self.startVgprGlobalReadOffsetB, self.states.b.numVgprGlobalReadOffsets
      if tc == "MXSA":
        return self.startVgprGlobalReadOffsetMXSA, self.states.mxsa.numVgprGlobalReadOffsets
      if tc == "MXSB":
        return self.startVgprGlobalReadOffsetMXSB, self.states.mxsb.numVgprGlobalReadOffsets
      return None, 0

    def refreshesShiftPtrGlobalReadOffsets(tP):
      if not (kernel["BufferLoad"] and usesCanonicalSrd(tP)):
        return False
      if kernel.get("EdgeType") != "ShiftPtr" or kernel["_UseSgprForGRO"]:
        return False
      if kernel["enableTDMA"] or kernel["enableTDMB"]:
        return False
      _, count = tensorGlobalReadOffsetInfo(tP)
      return count > 0

    def copyGlobalReadOffsets(dstBase, srcBase, tP, comment):
      tc = tP["tensorChar"]
      _, count = tensorGlobalReadOffsetInfo(tP)
      for i in range(count):
        module.add(VMovB32(dst=vgpr(dstBase + i) if dstBase is not None else vgpr("GlobalReadOffset%s+%u" % (tc, i)),
                           src=vgpr(srcBase + i) if srcBase is not None else vgpr("GlobalReadOffset%s+%u" % (tc, i)),
                           comment="%s vgprGlobalReadOffset%s+%u" % (comment, tc, i)))

    def checkpointGlobalReadOffsets(tP, tag):
      _, count = tensorGlobalReadOffsetInfo(tP)
      tmpVgpr = self.vgprPool.checkOutAligned(count, 1, tag)
      copyGlobalReadOffsets(tmpVgpr, None, tP, "checkpoint current")
      return tmpVgpr

    def restoreGlobalReadOffsets(tP, tmpVgpr):
      copyGlobalReadOffsets(None, tmpVgpr, tP, "restore current")
      self.vgprPool.checkIn(tmpVgpr)

    def refreshShiftPtrGlobalReadOffsets(tP):
      tc = tP["tensorChar"]
      module.addComment1("PAP: refresh next-tile ShiftPtr global-read offsets %s" % tc)
      module.add(self.lwaTileAssignment(kernel, tP))
      module.add(self.graTileAssignment(kernel, tP))
      module.add(self.graUnrollAssignment(kernel, tP))
      module.add(self.graTileOffsets(kernel, tP))
      module.add(self.graUnrollOffsets(kernel, tP))
      if tc in ["A", "B"]:
        if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartial%s" % tc]) \
          and not kernel["UseGeneralizedNLCOne%s" % tc] and not tP["isSwizzled"]:
          module.add(self.graShift(kernel, tP))
      elif tc in ["MXSA", "MXSB"]:
        parentTc = "A" if tc == "MXSA" else "B"
        if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartial%s" % parentTc]) \
          and not kernel["UseGeneralizedNLCOne%s" % parentTc] and not tP["isSwizzled"]:
          parentTP = tensorParametersA if tc == "MXSA" else tensorParametersB
          module.add(self.graShiftMX(kernel, tP, parentTP))
      module.add(self.graAddresses(kernel, tP))
      module.add(self.graFinalOffsets(kernel, tP))

    def emitFirstPgrLoad(tP, skipWait):
      if emitTensorLoadUsesStagger(tP):
        module.add(self.calculateStagger(kernel, tP))
      moduleTmp = self.directToLdsM0Update(kernel, 0, tP, skipWait)
      module.add(replaceHolder(moduleTmp, 0))
      module.add(self.globalReadDo(kernel, 0, tP))

    def emitTensorLoad(tP, skipWait=False):
      tc = tP["tensorChar"]
      module.addComment1("PAP: next-tile addresses and first-PGR load %s" % tc)
      if kernel["BufferLoad"] and usesCanonicalSrd(tP):
        with self.allocTmpSgpr(4, 2, "PAP%sSrdSnapshot" % tc) as tmpSgprInfo:
          checkpointDescriptorState(tP, tmpSgprInfo.idx)
          if refreshesShiftPtrGlobalReadOffsets(tP):
            tmpVgpr = checkpointGlobalReadOffsets(tP, "PAP%sGROSnapshot" % tc)
            refreshShiftPtrGlobalReadOffsets(tP)
            emitFirstPgrLoad(tP, skipWait)
            restoreGlobalReadOffsets(tP, tmpVgpr)
          else:
            module.add(self.graAddresses(kernel, tP))
            emitFirstPgrLoad(tP, skipWait)
          restoreDescriptorState(tP, tmpSgprInfo.idx)
      else:
        emitFirstPgrLoad(tP, skipWait)

    # StreamK PAP contract:
    #   Durable state: issued first-PGR loads for the next persistent iteration,
    #   SkPrefetchPrimed, and the saved PAP TDM/DTL LDS-bank bits encoded there.
    #   Borrowed state: current tile identity, descriptors/shadow limits,
    #   StaggerU registers, and LDS bank/descriptor state needed when current
    #   NLL or tail code resumes. Each borrowed group must be restored before
    #   leaving this handoff or by the caller immediately after it.
    tensorParameters1st = tensorParametersA
    tensorParameters2nd = tensorParametersB
    if self.isSwapGlobalReadOrderForDtvOrDtl(kernel, prefetch1=True):
      tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st

    prefetchTensorParameters = [tensorParameters1st]
    if "MX" in tensorParameters1st:
      prefetchTensorParameters.append(tensorParameters1st["MX"])
    if "MX" in tensorParameters2nd:
      prefetchTensorParameters.append(tensorParameters2nd["MX"])
    prefetchTensorParameters.append(tensorParameters2nd)
    papPrefetchUsesStagger = any(emitTensorLoadUsesStagger(tP) for tP in prefetchTensorParameters)

    def emitPrefetchGroup():
      module.addComment1("PAP: prefetch first global -> local group")
      module.add(self.openSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=isOptNLL))
      if papPrefetchUsesStagger:
        module.add(self.declareStaggerParms(kernel))
      emitTensorLoad(tensorParameters1st)
      if "MX" in tensorParameters1st:
        emitTensorLoad(tensorParameters1st["MX"], skipWait=True)
      if "MX" in tensorParameters2nd:
        emitTensorLoad(tensorParameters2nd["MX"], skipWait=True)
      skip2ndWaitForDtl = kernel["DirectToLds%s" % tensorParameters1st["tensorChar"]]
      emitTensorLoad(tensorParameters2nd, skipWait=skip2ndWaitForDtl)

    staggerState = borrowedStaggerState() if papPrefetchUsesStagger else []
    if staggerState:
      numStaggerSnapshotSgprs = sum(size for _, size in staggerState)
      with self.allocTmpSgpr(numStaggerSnapshotSgprs, 2, "PAPStaggerSnapshot") as tmpSgprInfo:
        checkpointBorrowedStaggerState(tmpSgprInfo.idx, staggerState)
        emitPrefetchGroup()
        restoreBorrowedStaggerState(tmpSgprInfo.idx, staggerState)
    else:
      emitPrefetchGroup()
    module.add(SMovB32(dst=sgpr("SkPrefetchPrimed"), src=1, comment="first PGR group for next persistent iter prefetched"))
    if ((kernel["DirectToLdsA"] or kernel["DirectToLdsB"])
        and not (kernel["enableTDMA"] and kernel["enableTDMB"])):
      module.add(self.papDtlSaveLdsBank(kernel, tensorParametersA, tensorParametersB))
    self.states.ldsTensorTokenIdx = \
        self.states.memTokenLdsBuffer1 if self.states.ldsTensorTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

    module.addComment2("End setupPrefetchAcrossPersistentLoads")
    return module

  ##############################################################################
  # Setup next-tile PAP loads for Subtile scheduler kernels.
  ##############################################################################
  def papCheckpointCurrentTileIdentityVgprs(self, kernel, prevTile):
    module = Module("papCheckpointCurrentTileIdentityVgprs")
    for name in self.papTileIdentityNames(kernel):
      module.add(VMovB32(dst=vgpr(prevTile[name]), src=sgpr(name),
                         comment="checkpoint %s for Subtile PAP restore" % name))
    return module

  def papRestoreCurrentTileIdentityVgprs(self, kernel, prevTile):
    module = Module("papRestoreCurrentTileIdentityVgprs")
    for name in self.papTileIdentityNames(kernel):
      module.add(VReadfirstlaneB32(dst=sgpr(name), src=vgpr(prevTile[name]),
                                   comment="restore current %s after Subtile PAP" % name))
    return module

  def prefetchAcrossPersistentSubtile(self, kernel, tensorParametersA, tensorParametersB, preloopGrModule=None, skipBarrier=False):
    module = Module("prefetchAcrossPersistentSubtile")
    if not (kernel.get("UseSubtileImpl") and self.isPrefetchAcrossPersistentEnabled(kernel)):
      return module
    if tensorParametersA is None or tensorParametersB is None:
      return module

    skipLabel = Label(self.labels.getNameInc("SK_SkipNllSubtilePAP"), "")
    module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Parallel reduction: skip Subtile PAP"))
    module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(), comment=""))
    module.add(SCmpGeU32(src0=sgpr("StreamKIter"), src1=sgpr("StreamKIterEnd"), comment="No next persistent iteration"))
    module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(), comment=""))
    if not skipBarrier:
      module.add(SBarrier(comment="Subtile PAP: sync before next-tile prefetch"))

    skComponent = Component.StreamK.find(self)
    tileIdentityNames = self.papTileIdentityNames(kernel)
    prevTileBase = self.vgprPool.checkOutAligned(len(tileIdentityNames), 1, "Subtile PAP tile identity")
    prevTile = {name: prevTileBase + i for i, name in enumerate(tileIdentityNames)}
    module.add(self.papCheckpointCurrentTileIdentityVgprs(kernel, prevTile))
    module.add(skComponent.prefetchAcrossPersistentSetupNextTile(self, kernel, tensorParametersA, tensorParametersB, skipLroReset=True))
    module.add(self.setupPrefetchAcrossPersistentSubtileLoads(kernel, tensorParametersA, tensorParametersB, preloopGrModule))
    module.add(self.papRestoreCurrentTileIdentityVgprs(kernel, prevTile))
    self.vgprPool.checkIn(prevTileBase)

    module.add(skipLabel)
    return module

  def setupPrefetchAcrossPersistentSubtileLoads(self, kernel, tensorParametersA, tensorParametersB, preloopGrModule=None):
    module = Module("setupPrefetchAcrossPersistentSubtileLoads")
    module.addComment2("Begin setupPrefetchAcrossPersistentSubtileLoads")

    def emitTensorLoad(tP):
      tc = tP["tensorChar"]
      module.addComment1("Subtile PAP: first-PGR load %s" % tc)
      if tc in ["A", "B"]:
        module.add(globalReadDoSubtile(tc, self, kernel))
      else:
        module.add(globalReadDoScaleSubtile(tc, self, kernel))

    module.add(globalReadDTLInitCommonSgpr(self, kernel))
    if kernel["ProblemType"].get("MXBlockA", 0) or kernel["ProblemType"].get("MXBlockB", 0):
      module.add(globalReadScaleSwizzledDTLInitCommonSgpr(self, kernel))

    module.add(self.graAddresses(kernel, tensorParametersA))
    if kernel["ProblemType"].get("MXBlockA", 0) and "MX" in tensorParametersA:
      module.add(self.graAddresses(kernel, tensorParametersA["MX"]))
    module.add(self.graAddresses(kernel, tensorParametersB))
    if kernel["ProblemType"].get("MXBlockB", 0) and "MX" in tensorParametersB:
      module.add(self.graAddresses(kernel, tensorParametersB["MX"]))

    if preloopGrModule is not None:
      module.add(preloopGrModule)
    else:
      emitTensorLoad(tensorParametersA)
      emitTensorLoad(tensorParametersB)
      if kernel["ProblemType"].get("MXBlockA", 0) and "MX" in tensorParametersA:
        emitTensorLoad(tensorParametersA["MX"])
      if kernel["ProblemType"].get("MXBlockB", 0) and "MX" in tensorParametersB:
        emitTensorLoad(tensorParametersB["MX"])
    module.add(SMovB32(dst=sgpr("SkPrefetchPrimed"), src=1, comment="Subtile PAP: first PRELOOP GR group prefetched"))

    module.addComment2("End setupPrefetchAcrossPersistentSubtileLoads")
    return module
  ##############################################################################
  # get conditions to skip local read write wait
  ##############################################################################
  def getConditionToSkipLocalReadWriteWait( self, kernel, u ):
    oneBufferScheduling = self.states.oneBufferScheduling
    condSkip = False
    # skip wait for SIA=3 and oneLDSBuffer scheduling and PLR and PGR==2 and u > localWriteStartIter
    # in this case, all local read is executed before 1LDSBuffer sync and no need to wait for local read
    if (kernel["_ScheduleIterAlg"] == 3 and oneBufferScheduling and kernel["PrefetchLocalRead"] and kernel["PrefetchGlobalRead"] >= 2):
      localWriteStartIter = self.states.lwStartMfmaIndex//self.states.numMfmaPerIter
      if u > localWriteStartIter:
        condSkip = True
    return not condSkip

  ##############################################################################
  # No Load Loop Body
  ##############################################################################
  def noLoadLoopBody( self, kernel, tensorParametersA, tensorParametersB, pack, packPre, isOptNLL, isNGLL, NLLfirst, NLLlast, NLLindex=0, NLLnum=1, \
                      useTailloopInNll=False, remainPgr=0):
    UnrollLoopSwapGlobalReadOrder = kernel["UnrollLoopSwapGlobalReadOrder"]
    if kernel["DirectToLdsA"] and kernel["DirectToLdsB"] and kernel["PrefetchGlobalRead"] >= 2:
      # DTLA+DTLB code case, NGLL code is exactly same and no need to consider UnrollLoopSwapGlobalReadOrder
      UnrollLoopSwapGlobalReadOrder = 0
    if kernel["UseCustomMainLoopSchedule"]:
      strNtab = "" if kernel["AdaptiveGemmNTAB"] == 0 else "_NTA0_NTB0"
      module = Module()
      if isNGLL:
        module.addComment0("Code-path 0, useGR=0, usePLR=1, useGRInc=1, useLoop = 0")
        module.add(MacroInstruction(name="MAINLOOP%s"%strNtab, args=[0,0,1,1,0]))
      else:
        module.addComment0("Code-path 0, useGR=0, usePLR=0, useGRInc=0, useLoop = 0")
        module.add(MacroInstruction(name="MAINLOOP%s"%strNtab, args=[0,0,0,0,0]))
      return module
    module = Module("noLoadLoopBody")
    expand = kernel["ExpandPointerSwap"]
    lastuIdx = False
    pflr     = self.states.numItersPLR
    localWriteEndIter = kernel["LoopIters"] - self.states.numItersPLR - 1
    dsWriteBA = False
    if isNGLL and UnrollLoopSwapGlobalReadOrder == 1 and NLLindex == 0 and NLLnum == 2:
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

    # initialize SubTileIdx
    self.states.SubTileIdx = 1 if self.states.numItersPLR and kernel["numSubTiles"] else 0

    if isNGLL:
      self.codes.perIterGlobalRead = [ Module() for i in range (kernel["LoopIters"]) ]

    for uIdx in range(0, kernel["LoopIters"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      isLastLoop = not isNGLL
      if u == 0:
        if not isLastLoop:
          if not kernel["NoLdsWriteCode"]:
            # For UnrollLoopSwapGlobalReadOrder == 1 or DirectToVgprA/B, we will have 2 NGLLs,
            # One with local write A then B and another with B then A.
            if dsWriteBA == True:
              # In the current scheduling, we always schedule lwa first then lwb second.
              # Put B in lwa code can easily change the order.
              self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersB, swapAB=1)
              if "MX" in tensorParametersB:
                self.codes.localWriteMXSA = self.localWriteDo(kernel, tensorParametersB["MX"], swapAB=1)
              else:
                self.codes.localWriteMXSA = Module()
              if "MX" in tensorParametersA:
                self.codes.localWriteMXSB = self.localWriteDo(kernel, tensorParametersA["MX"], swapAB=1)
              else:
                self.codes.localWriteMXSB = Module()
              self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersA, swapAB=1)
            else:
              self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersA)  # local write in loopcnt N targets data for loopcnt N+1
              if "MX" in tensorParametersA:
                self.codes.localWriteMXSA = self.localWriteDo(kernel, tensorParametersA["MX"])  # local write in loopcnt N targets data for loopcnt N+1
              else:
                self.codes.localWriteMXSA = Module()
              if "MX" in tensorParametersB:
                self.codes.localWriteMXSB = self.localWriteDo(kernel, tensorParametersB["MX"])
              else:
                self.codes.localWriteMXSB = Module()
              self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersB)
          else:
            self.codes.localWriteA = Module()
            self.codes.localWriteMXSA = Module()
            self.codes.localWriteMXSB = Module()
            self.codes.localWriteB = Module()
          # unrolled loop: global read A, B
          # DirectToVgprA + PGR2: we need DTVA GR in NGLL
          if kernel["DirectToVgprA"]:
            self.codes.globalReadA = self.globalReadDo(kernel, 1, tensorParametersA, unrollLoopIdx=0, g2lBufIdx=vregSetIdxGR)
          else:
            self.codes.globalReadA = StructuredModule() # empty
          if ("MX" in tensorParametersA) and kernel["DirectToVgprMXSA"]:
            self.codes.globalReadMXSA = self.globalReadDo(kernel, 1, tensorParametersA["MX"], unrollLoopIdx=0, g2lBufIdx=vregSetIdxGR)
          else:
            self.codes.globalReadMXSA = StructuredModule() # empty
          # DirectToVgprB + PGR2: we need DTVB GR in NGLL
          if ("MX" in tensorParametersB) and kernel["DirectToVgprMXSB"]:
            self.codes.globalReadMXSB = self.globalReadDo(kernel, 1, tensorParametersB["MX"], unrollLoopIdx=0, g2lBufIdx=vregSetIdxGR)
          else:
            self.codes.globalReadMXSB = StructuredModule() # empty
          # DirectToVgprB + PGR2: we need DTVB GR in NGLL
          if kernel["DirectToVgprB"]:
            self.codes.globalReadB = self.globalReadDo(kernel, 1, tensorParametersB, unrollLoopIdx=0, g2lBufIdx=vregSetIdxGR)
          else:
            self.codes.globalReadB = StructuredModule() # empty
          if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"]:
            pass # non-TDM path does not have globalReadMetadata
          elif kernel["enableTDMMetadata"]:
            self.codes.globalReadMetadata = StructuredModule() # empty

        else:
          self.codes.localWriteA = Module()
          self.codes.localWriteB = Module()
          self.codes.localWriteMXSA = Module()
          self.codes.localWriteMXSB = Module()
          self.codes.globalReadMetadata = StructuredModule() # empty

        callMakeSchedule = not isNGLL or kernel["ExpandPointerSwap"] or UnrollLoopSwapGlobalReadOrder or isDTVAB or \
          (kernel["PrefetchGlobalRead"] >= 3 and isNGLL) or \
          (self.states.doPackPreSchedulingNextLoop and isNGLL)
        if callMakeSchedule:
          # PAP would have GlobalRead and GlobalInc, but no localWrite
          # Get the perIterGlobalReadCode code for PAP (if PAP=On), else would be empty
          skipGlobalReadInc = False
          if kernel["PrefetchGlobalRead"] >= 3 and isNGLL:
            # PGR>=3 and NGLL case, we need GR Inc only for the first NGLL and skip them afterwards
            skipGlobalReadInc = remainPgr < kernel["PrefetchGlobalRead"] - 1
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, skipGlobalReadInc=skipGlobalReadInc, lastLoop=NLLlast, isNGLL=isNGLL)
          module.add(self.codes.unrollLoopHeader)

        if isNGLL and not callMakeSchedule:
          globalReadIncACode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementA")
          globalReadIncBCode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementB")
          self.codes.perIterGlobalRead[u].addComment1("Global Read IncA")
          self.codes.perIterGlobalRead[u].add(globalReadIncACode)
          self.codes.perIterGlobalRead[u].addComment1("Global Read IncB")
          self.codes.perIterGlobalRead[u].add(globalReadIncBCode)

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = 1 if kernel["ForceUnrollSubIter"] else (u == localWriteEndIter)
      isSwapAndResetLwoIter = (u == localWriteEndIter)
      isSwapLroIter = isResetLroIter
      if kernel["_ScheduleIterAlg"] == 3:
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
      packPreIdx = (u+pflr) % self.states.numPackBuffer # pack store and pack pre read
      packIdx = u % self.states.numPackBuffer # pack read
      # hack: full pack prefetch case, move local read for next loop 1 iter ahead (no change for CMS)
      uNext = u
      if kernel["ForceUnrollSubIter"] and self.states.doFullPackCodePrefetch and u >= self.states.numVgprBuffer and not kernel["UseCustomMainLoopSchedule"]:
        if u == kernel["LoopIters"] - 2:
          uNext = kernel["LoopIters"] - 1
          self.states.SubTileIdx = 0 # hack: force to idx0 for next loop
        elif u == kernel["LoopIters"] - 1:
          uNext = kernel["LoopIters"] - 2
      packStoreIdx = (uNext+pflr) % self.states.numPackBuffer

      localReads = Module("local read")

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
      doReadA    = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadA - self.states.numItersPLR)
      doReadMXSA = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMXSA - self.states.numItersPLR)
      doReadMXSB = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMXSB - self.states.numItersPLR)
      doReadB    = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadB - self.states.numItersPLR)
      doReadM    = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMetadata - self.states.numItersPLR)
      if kernel["ForceUnrollSubIter"]:
        doReadA = 1 if u == 0 else 0
        doReadMXSA = 1 if u == 0 else 0
        doReadMXSB = 1 if u == 0 else 0
        doReadB = 1 if u == 0 else 0
        doReadM = 1 if u == 0 else 0
      # reads for next loop
      doNext = uNext > localWriteEndIter
      doReadA = doReadA or (hasLiveLdsData and doNext)
      doReadMXSA = doReadMXSA or (hasLiveLdsData and doNext)
      doReadMXSA = doReadMXSA and kernel["ProblemType"]["MXBlockA"]
      doReadMXSB = doReadMXSB or (hasLiveLdsData and doNext)
      doReadMXSB = doReadMXSB and kernel["ProblemType"]["MXBlockB"]
      doReadB = doReadB or (hasLiveLdsData and doNext)
      doReadM = doReadM or (hasLiveLdsData and doNext)
      doReadM = doReadM and (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"])
      if ((hasLiveLdsData and doNext) or (self.states.numItersPLR == 0 and uIdx == 0)) and not self.states.lockLdsReadTokenSwap:
        # swap LR buffer token only when the LR buffer actually changes
        self.states.ldsReadTokenIdx = self.states.memTokenLdsBuffer1 if self.states.ldsReadTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0
      if isOptNLL and not self.states.lockLdsReadTokenSwap:
        # After entering OptNLL body, keep the read token fixed.
        self.states.lockLdsReadTokenSwap = True
      for iui in range(0,kernel["InnerUnroll"]):
        # use full prefetch only for next loop
        usePLRPackA = self.states.doFullPackCodePrefetch and (doNext or u == 0)
        usePLRPackB = self.states.doFullPackCodePrefetch and (doNext or kernel["ForceUnrollSubIter"])
        usePLRPackBNext = usePLRPackB and (kernel["ForceUnrollSubIter"] and u == 0)
        doReadA = doReadA and iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadMXSA = doReadMXSA and iui*self.states.numReadsIterCoalescedMXSA < kernel["InnerUnroll"]
        doReadMXSB = doReadMXSB and iui*self.states.numReadsIterCoalescedMXSB < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]
        doReadM = doReadM and iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]
        if doReadA:
          localReads.addComment1("local read a")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadA
          if self.states.packDTVA or self.states.convDTVA:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadA + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeA, packCodeA, packPreA = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedA, 0, tensorParametersA)
          if needNextBufLR:
            localReads.add(localReadCodeA)
          # packPre code
          if doNext and self.states.doPackPreSchedulingNextLoop or self.states.doPackPreSchedulingThisLoop:
            # do pack pre scheduling for this loop. Put packPreCode to packPre
            packPre[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packPreA)
          else:
            # otherwise, put pack pre to pack
            pack[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packPreA)
          # pack code
          if usePLRPackA:
            packPre[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packCodeA)
          else:
            pack[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packCodeA)
        if doReadMXSA:
          localReads.addComment1("local read mxsa")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadMXSA
          if self.states.packDTVA or self.states.convDTVA:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadMXSA + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeMXSA, packCodeMXSA, packPreMXSA = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedMXSA, 0, tensorParametersA["MX"])
          if needNextBufLR:
            localReads.add(localReadCodeMXSA)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSA].add(packPreMXSA)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSA].add(packCodeMXSA)
        if doReadM:
          localReads.addComment1("local read metadata")
          localReadCodeM, packCodeM, packPreM = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
          if needNextBufLR:
            localReads.add(localReadCodeM)
          pack[plrIdx*self.states.numIterPerCoalescedReadMetadata].add(packCodeM)
          if kernel["ForceUnrollSubIter"]:
            pack[1].add(packCodeM)
        if doReadMXSB:
          localReads.addComment1("local read mxsb")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadMXSB
          if self.states.packDTVB or self.states.convDTVB:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadMXSB + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeMXSB, packCodeMXSB, packPreMXSB = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedMXSB, 0, tensorParametersB["MX"])
          if needNextBufLR:
            localReads.add(localReadCodeMXSB)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSB].add(packPreMXSB)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSB].add(packCodeMXSB)
        if doReadB:
          localReads.addComment1("local read b")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadB
          if self.states.packDTVB or self.states.convDTVB:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadB + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeB, packCodeB, packPreB = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedB, 0, tensorParametersB)
          if needNextBufLR:
            localReads.add(localReadCodeB)
          # packPre code
          if doNext and self.states.doPackPreSchedulingNextLoop or self.states.doPackPreSchedulingThisLoop:
            # do pack pre scheduling for this loop. Put packPreCode to packPre
            packPre[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packPreB)
          else:
            # otherwise, put pack pre to pack
            pack[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packPreB)
          # pack code
          if usePLRPackB:
            if usePLRPackBNext:
              packPre[(packStoreIdx+1)*self.states.numIterPerCoalescedReadB].add(packCodeB)
            else:
              packPre[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packCodeB)
          else:
            pack[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packCodeB)
        if (not isResetLroIter or iui != kernel["InnerUnroll"]-1):
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadMXSA:
            localReads.addComment1("local read increment mxsa")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA["MX"]))
          if doReadM:
            localReads.addComment1("local read increment metadata")
            localReads.add(self.localReadInc(kernel, iui, tPM))
          if doReadMXSB:
            localReads.addComment1("local read increment mxsb")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersB["MX"]))
          if doReadB:
            localReads.addComment1("local read increment b")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersB))

      if not isLastLoop:
        if kernel["PrefetchGlobalRead"]:
          # put barrier at localWriteEndIter+1
          if (u == localWriteEndIter+1 and not self.states.doFullPackCodePrefetch) or \
             (self.states.doFullPackCodePrefetch and (u == self.states.syncPlrMfmaIndex // self.states.numMfmaPerIter)) or \
             (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["_ScheduleIterAlg"] == 2) or \
             (u == (localWriteEndIter+1)%kernel["LoopIters"] and not self.states.numItersPLR \
              and kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["_ScheduleIterAlg"] == 0):
            # PLR=0 + TDM + SIA=0 wrap-around handling, mirroring the main-loop
            # emit (see equivalent block ~line 4085). With numItersPLR==0 the
            # canonical isBarrier (= localWriteEndIter+1) equals LoopIters and
            # the `u == isBarrier` clause above never fires inside the loop
            # body; emit at the wrapped position (u == 0) instead. SIA gating
            # is narrow on purpose: SIA=2 is handled by the prior clause;
            # SIA=3 takes a dedicated codegen path; SIA=1 is not exercised
            # with TDM in any current test.
            if not kernel["NoLdsWriteCode"]:
              waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
            elif kernel["enableTDMA"] and kernel["enableTDMB"]:
              waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for TDM tensor loads"))
            if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
              # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
              waitLWCode.add(self.getWaitcntCodeForDirectToVgpr(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, u, \
                             isNLL=(not isNGLL), beforeBarrier=True))
            elif kernel["PrefetchGlobalRead"]>=2 and (kernel["DirectToLdsA"] and kernel["DirectToLdsB"]):
              waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, remainPgr-1, -1, -1, "wait for global reads with lds"))
            syncCode.add(self._syncThreads(kernel, "noLoadLoop sync"))

          if isSwapAndResetLwoIter:
            # local write for next iter, used to have local writes here
            # Swap offsets A(MXSA)
            if kernel["enableTDMA"]:
              pointerLWCode.addComment1("tdm swap offsets a")
              pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tensorParametersA))
            elif not kernel["NoLdsWriteCode"]:
              pointerLWCode.addComment1("local write swap offsets a")
              pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))

            if "MX" in tensorParametersA:
              pointerLWCode.addComment1("local write swap offsets mxsa")
              if kernel["enableTDMA"]:
                pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tensorParametersA["MX"]))
              elif not kernel["NoLdsWriteCode"]:
                pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA["MX"]))
            # Swap offsets B(MXSB)
            if "MX" in tensorParametersB:
              pointerLWCode.addComment1("local write swap offsets mxsb")
              #TODO: TDM refactor
              if kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["NumWaves"] == 1:
                pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tensorParametersB["MX"]))
              elif not kernel["enableTDMB"] and not kernel["NoLdsWriteCode"]:
                pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB["MX"]))
            if kernel["enableTDMB"]:
              #TODO: TDM refactor
              if kernel["NumWaves"] == 1:
                pointerLWCode.addComment1("tdm swap offsets b")
                pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tensorParametersB))
            elif not kernel["NoLdsWriteCode"]:
              pointerLWCode.addComment1("local write swap offsets b")
              pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
            # Swap offsets Metadata
            if kernel["enableTDMMetadata"] and not kernel["DirectToVgprSparseMetadata"]:
                pointerLWCode.addComment1("tdm swap offsets metadata")
                pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tPM))
            # Swap local write memory token
            self.states.ldsWriteTokenIdx = \
              self.states.memTokenLdsBuffer1 if self.states.ldsWriteTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

          if isSwapLroIter: # ResetLroIter
            # Swap, reset, or increment the LRO:
            # force internalPointerSwap = False in NGLL case
            internalPointerSwap = expand and not isNGLL
            if not kernel["ForceUnrollSubIter"] or (doReadA and (u<localWriteEndIter)):
              pointerLRCode.addComment1("local read swap offsets a")
              pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA))
            if kernel["ProblemType"]["MXBlockA"] and ((not kernel["ForceUnrollSubIter"]) or (doReadMXSA and (u<localWriteEndIter))):
              pointerLRCode.addComment1("local read swap offsets mxsa")
              pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA["MX"]))
            if kernel["ProblemType"]["MXBlockB"] and ((not kernel["ForceUnrollSubIter"]) or (doReadMXSB and (u<localWriteEndIter))):
              pointerLRCode.addComment1("local read swap offsets mxsb")
              pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersB["MX"]))
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
          if kernel["ProblemType"]["MXBlockA"]:
            pointerLRCode.addComment1("local read init pointers mxsa")
            pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA["MX"]))
          if kernel["ProblemType"]["MXBlockB"]:
            pointerLRCode.addComment1("local read init pointers mxsb")
            pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB["MX"]))
          pointerLRCode.addComment1("local read init pointers b")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            pointerLRCode.addComment1("local read init pointers metadata")
            pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))

      elif kernel["enableTDMA"] and kernel["enableTDMB"]:
        if isSwapLroIter: # ResetLroIter
          # Swap, reset, or increment the LRO:
          # force internalPointerSwap = False in NGLL case
          internalPointerSwap = False
          pointerLRCode.addComment1("local read swap offsets a")
          pointerLRCode.addComment1("isSwapLroIter = %d"%(isSwapLroIter))
          pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA))
          if kernel["ProblemType"]["MXBlockA"]:
            pointerLRCode.addComment1("local read swap offsets mxsa")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA["MX"]))
          if kernel["ProblemType"]["MXBlockB"]:
            pointerLRCode.addComment1("local read swap offsets mxsb")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersB["MX"]))
          pointerLRCode.addComment1("local read swap offsets b")
          pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersB))
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            pointerLRCode.addComment1("local read swap offsets metadata")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, internalPointerSwap, tPM))

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
      if kernel["EnableMatrixInstruction"]:
        postShiftK = Module()
        if kernel["UseF32XEmulation"] and useTailloopInNll:
          # useTailloopInNll case, we need to generate TF32 pack code after ShiftK
          packItems = []
          for iui in range(kernel["InnerUnroll"]):
            # schedule both packPre and pack
            packAPre = pack[packIdx].findNamedItem("packA_I%s Pre"%(iui))
            packBPre = pack[packIdx].findNamedItem("packB_I%s Pre"%(iui))
            packA = pack[packIdx].findNamedItem("packA_I%s"%(iui))
            packB = pack[packIdx].findNamedItem("packB_I%s"%(iui))
            if packAPre == None:
              packAPre = Module()
            if packBPre == None:
              packBPre = Module()
            packAItems = packA.flatitems()
            packBItems = packB.flatitems()
            # Gather A, B conversion code based on scheduling order
            self._interleavePackAB(kernel, packAItems, packBItems, packItems, prefetch=True, searchStrings=["__TF32_1", "__TF32_2"])
          # put packPre back to pack[packIdx]
          pack[packIdx] = Module()
          pack[packIdx].add(packAPre)
          pack[packIdx].add(packBPre)
          for item in packItems:
            postShiftK.add(item)
        macIterCode.add(self.mfmaIter(kernel, tensorParametersA, tensorParametersB, u, kernel["InnerUnroll"], vregSetIdxMFMA, unrollIdx = u, tail = useTailloopInNll, postShiftK = postShiftK))
      else:
        macIterCode.add(self.macIter(kernel, tensorParametersA, tensorParametersB, luIdx, kernel["InnerUnroll"], True))
      if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
        tP = tensorParametersA if kernel["ProblemType"]["BiasSrc"] == "A" else tensorParametersB
        macIterCode.add(self.exclasses.biasSumUnroll.loopSum(self, kernel, tP, u, kernel["InnerUnroll"]))
      subIterCode = self._makeSubIterSchedule(kernel, tensorParametersA, tensorParametersB, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[packIdx], packPre[packPreIdx], \
                      module, NLLlast=NLLlast, tailloopInNll=useTailloopInNll, isNLLorNGLL=True)
      module.add(subIterCode)
      self.states.SubTileIdx = (self.states.SubTileIdx + 1) % kernel["numSubTiles"]
      pack[packIdx] = Module()
      packPre[packPreIdx] = Module()

      # tail loop in NoLoadLoop case, generate close loop code for TailLoop here (except for last loop iteration)
      if useTailloopInNll:
        finalLoop = (u == kernel["LoopIters"] - 1)
        NLLindexLast = (NLLindex == NLLnum - 1)
        module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, -2, finalLoop, skipCondJumpCounter=u, NLLindexLast=NLLindexLast))

    return module

  ##############################################################################
  # noLoadLoop
  # Create the no load loop (NLL)
  #
  # isOptNLL : the NLL is to be optimized for the alpha=1 and non-edge case
  ##############################################################################
  def noLoadLoop( self, kernel, tensorParametersA, tensorParametersB, isOptNLL, isNGLL, pack, packPre, NLLindex=0, NLLnum=1, useTailloopInNll=False, remainPgr=0):
    module = Module("noLoadLoop")
    LoopNameComment = "NoGlobalLoadLoop" if isNGLL else "NoLoadLoop"
    if useTailloopInNll:
      LoopNameComment = "TailLoop in " + LoopNameComment
    if remainPgr >= 1:
      LoopNameComment += "_%d"%remainPgr
    isOptNLLComment = "Opt" if isOptNLL else "Ord"
    startComment = "%s. %s - Begin " % (isOptNLLComment, LoopNameComment)
    if NLLnum > 1:
      startComment = startComment + "%u/%u"%(NLLindex+1, NLLnum)
    module.addComment2(startComment)
    NLLfirst = True
    NLLlast = True
    PGR = kernel["PrefetchGlobalRead"]
    if PGR >= 3 and isNGLL:
      # PGR>=3 case, add a label for early exit destination
      LabelNGLL = Label("NoGlobalLoadLoop_%d"%remainPgr, "")
      module.add(LabelNGLL)
      # PGR>=3 and NGLL case, we need GR inc code only at the first NGLL (remainPgr = PGR - 2)
      # Later, we need to make it empty to avoid generating unnecessary inc code
      if remainPgr == PGR - 2:
        self.codes.globalReadIncrements = self.globalReadIncrementAB(kernel, None, None, self.states.unrollIdx, 0)

    if PGR >= 2:
      # PGR>=2 case NoLoadLoop(NLL) is generated twice
      # we need to distinguish them to generate proper code at each NLL
      if isNGLL:
        NLLlast = False
      else:
        # PGR=2 and not isNGLL means second NoLoadLoop for PGR2.
        # Need to avoid generating duplicated code which is already generated in NGLL(first NoLoadLoop for PGR=2)
        NLLfirst = False
    if isNGLL:
      self.codes.perIterLocalWrite = deepcopy(self.codes.perIterLocalWriteCodeNGLL)
      # perIterLocalWriteCodeNGLL may be generated before ldsWriteTokenIdx flips to
      # the current LDS buffer, so refresh each local-write instruction token here.
      localWriteMemToken = MemTokenData([self.states.ldsWriteTokenIdx])
      for _, localWriteCode in self.codes.perIterLocalWrite:
        for item in localWriteCode.flatitems():
          if isinstance(item, DSStoreInstruction):
            item.setMemToken(localWriteMemToken)
            # Keep debug/comment text aligned with the actual updated memory token.
            if isinstance(item.comment, str):
              syncComment = "sync LDS%u"%(self.states.ldsWriteTokenIdx)
              item.comment = item.comment.replace("sync LDS0", syncComment).replace("sync LDS1", syncComment)
      self.states.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
      if kernel["ExpandPointerSwap"] == 1 or self.states.scheduleIterAlg==0:
        self.codes.globalReadA = StructuredModule() # empty
        self.codes.globalReadMXSA = StructuredModule() # empty
        self.codes.globalReadMXSB = StructuredModule() # empty
        self.codes.globalReadB = StructuredModule() # empty
    #else:
    if not isNGLL:
      self.codes.dtlsM0UpdateA = StructuredModule()
      self.codes.globalReadA = StructuredModule() # empty
      self.codes.dtlsM0UpdateMXSA = StructuredModule()
      self.codes.globalReadMXSA = StructuredModule() # empty
      self.codes.dtlsM0UpdateMXSB = StructuredModule()
      self.codes.globalReadMXSB = StructuredModule() # empty
      self.codes.dtlsM0UpdateB = StructuredModule()
      self.codes.globalReadB = StructuredModule() # empty
      # PGR2 + DTV case, we still need global read inc code in NLL
      tPAlocal = None
      tPBlocal = None
      if PGR == 2 and kernel["DirectToVgprA"]:
        tPAlocal = tensorParametersA # generate inc A code
      if PGR == 2 and kernel["DirectToVgprB"]:
        tPBlocal = tensorParametersB # generate inc B code
      if PGR <= 2:
        self.codes.globalReadIncrements = self.globalReadIncrementAB(kernel, tPAlocal, tPBlocal, self.states.unrollIdx, 0)
      self.codes.localWriteA = Module()
      self.codes.localWriteB = Module()

    # re-calculate loop counter for tailloopInNll
    if useTailloopInNll:
      module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, -1, tailloopInNll=True, NLLindex=NLLindex))

    openSum = self.openSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL, isNGLL=isNGLL, NLLindex=NLLindex, NLLnum=NLLnum, \
                                        tailloopInNll=useTailloopInNll)
    module.add(openSum)

    papPriorSync = False
    if not self.states.numItersPLR and not kernel["UseCustomMainLoopSchedule"]:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        vlcntVal = remainPgr if kernel["PrefetchGlobalRead"] >= 2 and isNGLL else 0
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, vlcntVal, -1, -1, "10wait for global read"))
      if not kernel["NoLdsWriteCode"]:
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "4wait for local write"))
      module.add(self._syncThreads(kernel, "wait for local write done, sync"))
      papPriorSync = True
    elif kernel["enableTDMA"] and kernel["enableTDMB"]:
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for tensor load to finish"))
      module.add(self._syncThreads(kernel, "wait for tensor load done, sync"))
      papPriorSync = True

    isPapNll = (not isOptNLL and not isNGLL and NLLindex == NLLnum - 1
                and self.isPrefetchAcrossPersistentEnabled(kernel))

    if isPapNll:
      module.add(self.prefetchAcrossPersistent(kernel, tensorParametersA, tensorParametersB,
                                               skipBarrier=papPriorSync))

    # generate no Load Loop Body code
    module.add(self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, pack, packPre, isOptNLL, isNGLL, NLLfirst, NLLlast, NLLindex=NLLindex, \
                                   NLLnum=NLLnum, useTailloopInNll=useTailloopInNll, remainPgr=remainPgr))

    if self.do["executeToLoopEnd"] and isOptNLL:
      module.add(self.functionEnd(kernel, addLabel=False))

    # Close code is necessary for both first and last (NGLL case(=NLLfirst) needs label)
    module.add(self.closeSumAtLeastUnroll(kernel, tensorParametersA, tensorParametersB, prefetch=False, isOptNLL=isOptNLL, isNGLL=isNGLL, \
                                          isNotLast=(NLLindex<(NLLnum-1)), tailloopInNll=useTailloopInNll, remainPgr=remainPgr))

    if self.states.FactorDim == 3:
      self.updateBranchPlaceHolder(module, ["skipOptNLL_placeholder", "skipOptNLL_scc1_placeholder"] , ["OptNLL_End", "OptNLL_End"], ["SCBranchSCC0", "SCBranchSCC1"])
    return module

  ##############################################################################
  # Loop Body
  # When UnrollLoopSwapGlobalReadOrder is enabled,
  # dsWriteBA is to do ds_write B first.
  # grBA is to do buffer_load B first.
  ##############################################################################
  def _loopBody( self, kernel, tensorParametersA, tensorParametersB, pack, packPre, lc, loopCopies, finalLoop, firstIter=False, dsWriteBA=False, grBA=False, isDTVGRSecondBuf=False, skipClose=False, initCIter=False, nta=0, ntb=0 ):
    module = Module("loopBody")
    expand = kernel["ExpandPointerSwap"]
    # initialize SubTileIdx
    self.states.SubTileIdx = 1 if self.states.numItersPLR and kernel["numSubTiles"] else 0

    label_unrolledLoop_lc1 = Label("unrolledLoop_lc1", "", alignment=16)

    # For loopCopies>=2 (EPS uses 2; HalfPLR uses 3), place the lc=1 SBranch target
    #   EPS    : initC[lc=0] -> lc=1 -> lc=0 -> lc=1 -> ...
    #   HalfPLR: initC[lc=0] -> lc=1 -> lc=2 -> lc=0 -> lc=1 -> lc=2 -> ...
    if loopCopies >= 2 and lc == 1 and not initCIter:
      module.add(label_unrolledLoop_lc1)

    # not generate openLoop for firstIter
    if not firstIter and not initCIter:
      module.addComment2("Unrolled Loop %u/%u - Begin" % (lc+1, loopCopies))
    if kernel["PrefetchGlobalRead"] and not self.states.numItersPLR and not kernel["_ScheduleIterAlg"] == 2 and not kernel["UseCustomMainLoopSchedule"]:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        vlcntVal = kernel["PrefetchGlobalRead"] - 1 if kernel["PrefetchGlobalRead"] >= 2 else 0
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, vlcntVal, -1, -1, "11wait for global read"))
      if not kernel["NoLdsWriteCode"]:
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "1wait for local write"))
      module.add(self._syncThreads(kernel, "4sync for global read, PGR->LW needs sync"))
    elif kernel["PrefetchGlobalRead"] and kernel["enableTDMA"] and kernel["enableTDMB"]:
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for tensor load to finish"))
      module.add(self._syncThreads(kernel, "wait for tensor load to finish, PGR->LW needs sync"))

    module.addComment1("Begin Each Unroll: Check VGPR.checkin for INT8 LW")

    # swap the order of global read (B->A)
    # - swapAB (grBA=True)
    # - isSwapGlobalReadOrderForDtvOrDtl is true
    tensorParameters1st = tensorParametersA
    tensorParameters2nd = tensorParametersB
    tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]
    tc1 = 'A'
    tc2 = 'B'
    if grBA==True or self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
      tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
      tc1, tc2 = tc2, tc1

    # unrolled loop: global read A, B
    # M0 update for directToLds, skip if using custom main loop schedule
    self.codes.dtlsM0UpdateA = self.directToLdsM0Update(kernel, 1, tensorParameters1st, kernel["UseCustomMainLoopSchedule"])
    if "MX" in tensorParameters1st:
      self.codes.dtlsM0UpdateMXSA = self.directToLdsM0Update(kernel, 1, tensorParameters1st["MX"], skipWait=True)
    else:
      self.codes.dtlsM0UpdateMXSA = StructuredModule()
    if "MX" in tensorParameters2nd:
      self.codes.dtlsM0UpdateMXSB = self.directToLdsM0Update(kernel, 1, tensorParameters2nd["MX"], skipWait=True)
    else:
      self.codes.dtlsM0UpdateMXSB = StructuredModule()
    # skip wait for DTL if global load 1st is DTL
    skip2ndWaitForDtl = kernel["DirectToLds%s"%tc1] or kernel["UseCustomMainLoopSchedule"]
    self.codes.dtlsM0UpdateB = self.directToLdsM0Update(kernel, 1, tensorParameters2nd, skip2ndWaitForDtl)
    tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]

    if kernel["PrefetchGlobalRead"] > 1:
      # swap Tensor memToken before doing global read
      self.states.ldsTensorTokenIdx = \
          self.states.memTokenLdsBuffer1 if self.states.ldsTensorTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

    g2lBufIdx1st = 0
    if grBA==True or (kernel["DirectToVgpr%s"%tc1] and isDTVGRSecondBuf):
      # use second buffer
      g2lBufIdx1st = 1
    self.codes.globalReadA = self.globalReadDo(kernel, 1, tensorParameters1st, unrollLoopIdx=lc, g2lBufIdx=g2lBufIdx1st, tPM=tPM)
    if "MX" in tensorParameters1st:
      g2lBufIdx1st = 0
      if grBA==True or (kernel["DirectToVgprMXS%s"%tc1] and isDTVGRSecondBuf):
        # use second buffer
        g2lBufIdx1st = 1
      self.codes.globalReadMXSA = self.globalReadDo(kernel, 1, tensorParameters1st["MX"], unrollLoopIdx=lc, g2lBufIdx=g2lBufIdx1st)
    else:
      self.codes.globalReadMXSA = StructuredModule()

    # unrolled loop: global read Metadata
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"] and kernel["enableTDMMetadata"]:
      self.codes.globalReadMetadata = self.globalReadDo(kernel, 1, tPM, unrollLoopIdx=lc, g2lBufIdx=0)
    else:
      self.codes.globalReadMetadata = StructuredModule() # empty

    if "MX" in tensorParameters2nd:
      g2lBufIdx2nd = 0
      if grBA==True or (kernel["DirectToVgprMXS%s"%tc2] and isDTVGRSecondBuf):
        # use second buffer
        g2lBufIdx2nd = 1
      self.codes.globalReadMXSB = self.globalReadDo(kernel, 1, tensorParameters2nd["MX"], unrollLoopIdx=lc, g2lBufIdx=g2lBufIdx2nd)
    else:
      self.codes.globalReadMXSB = StructuredModule()
    g2lBufIdx2nd = 0
    if grBA==True or (kernel["DirectToVgpr%s"%tc2] and isDTVGRSecondBuf):
      # use second buffer
      g2lBufIdx2nd = 1
    self.codes.globalReadB = self.globalReadDo(kernel, 1, tensorParameters2nd, unrollLoopIdx=lc, g2lBufIdx=g2lBufIdx2nd, tPM=tPM)

    if kernel["PrefetchGlobalRead"] <= 1:
      # swap Tensor memToken after doing global read
      self.states.ldsTensorTokenIdx = \
          self.states.memTokenLdsBuffer1 if self.states.ldsTensorTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

    # unrolled loop: increment global read addresses
    self.codes.globalReadIncrements = self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, 0)
    globalReadIncACode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.codes.globalReadIncrements.findNamedItem("globalReadIncrementB")

    self.codes.gl2PrefetchIncrement = Module()
    self.codes.gl2Prefetch = Module()
    if kernel["PrefetchGL2"]:
      loopCounter = self.loopCounter(kernel, self.states.unrollIdx)
      self.codes.gl2PrefetchIncrement = Module()
      self.codes.gl2PrefetchIncrement.add(SCmpLeU32(loopCounter, kernel["PrefetchGlobalRead"] + kernel["PrefetchGL2"], \
        comment="counterL<=PGR+GL2"))
      self.codes.gl2PrefetchIncrement.add(SCMovB32(sgpr("GL2PrefetchIncA"), 0))
      self.codes.gl2PrefetchIncrement.add(SCMovB32(sgpr("GL2PrefetchIncB"), 0))
      if kernel["ProblemType"]["MXBlockA"]:
        self.codes.gl2PrefetchIncrement.add(SCMovB32(sgpr("GL2PrefetchIncMXSA"), 0))
      if kernel["ProblemType"]["MXBlockB"]:
        self.codes.gl2PrefetchIncrement.add(SCMovB32(sgpr("GL2PrefetchIncMXSB"), 0))
      self.codes.gl2PrefetchIncrement.add(self.gl2PrefetchIncrementAddr(kernel, tensorParametersA, tensorParametersB))
      self.codes.gl2Prefetch = Module()
      self.codes.gl2Prefetch.add(self.gl2PrefetchIssueLoad(kernel, tensorParametersA, tensorParametersB))
      

    if not kernel["NoLdsWriteCode"]:
      self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersA)
      if "MX" in tensorParametersA:
        self.codes.localWriteMXSA = self.localWriteDo(kernel, tensorParametersA["MX"])
      else:
        self.codes.localWriteMXSA = Module()
      if "MX" in tensorParametersB:
        self.codes.localWriteMXSB = self.localWriteDo(kernel, tensorParametersB["MX"])
      else:
        self.codes.localWriteMXSB = Module()
      self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersB)
    else:
      self.codes.localWriteA = Module()
      self.codes.localWriteMXSA = Module()
      self.codes.localWriteMXSB = Module()
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
      module.add(self._syncThreads(kernel, "PGR=0, prior iter done reading lds, LW wait LR, sync"))
      if not kernel["NoLdsWriteCode"]:
        module.addComment1("local write a")
        tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA)
        module.add(tempLWCodeModA)
        if "MX" in tensorParametersA:
          module.addComment1("local write mxsa")
          tempLWCodeModMXSA = self.localWriteDo(kernel, tensorParametersA["MX"])
          module.add(tempLWCodeModMXSA)
        if "MX" in tensorParametersB:
          module.addComment1("local write mxsb")
          tempLWCodeModMXSB = self.localWriteDo(kernel, tensorParametersB["MX"])
          module.add(tempLWCodeModMXSB)
        module.addComment1("local write b")
        tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB)
        module.add(tempLWCodeModB)
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
        module.add(self._syncThreads(kernel, "After LW code, sync"))
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
        packPre[plrIdx] = Module()
        for iui in range(0,kernel["InnerUnroll"]):
          if iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]:
            module.addComment1("prefetch local a")
            localReadCodeA, packCodeA, packPreA = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadA, iui*self.states.numReadsIterCoalescedA, 0, tensorParametersA)
            module.add(localReadCodeA)
            pack[plrIdx].add(packPreA) # no packPre scheduling for PGR0
            pack[plrIdx].add(packCodeA)
          if kernel["ProblemType"]["MXBlockA"]:
            if iui*self.states.numReadsIterCoalescedMXSA < kernel["InnerUnroll"]:
              module.addComment1("prefetch local mxsa")
              localReadCodeMXSA, packCodeMXSA, packPreMXSA = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMXSA, iui*self.states.numReadsIterCoalescedMXSA, 0, tensorParametersA["MX"])
              module.add(localReadCodeMXSA)
              pack[plrIdx].add(packPreMXSA)
              pack[plrIdx].add(packCodeMXSA)
          if kernel["ProblemType"]["MXBlockB"]:
            if iui*self.states.numReadsIterCoalescedMXSB < kernel["InnerUnroll"]:
              module.addComment1("prefetch local mxsb")
              localReadCodeMXSB, packCodeMXSB, packPreMXSB = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMXSB, iui*self.states.numReadsIterCoalescedMXSB, 0, tensorParametersB["MX"])
              module.add(localReadCodeMXSB)
              pack[plrIdx].add(packPreMXSB)
              pack[plrIdx].add(packCodeMXSB)
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            if iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]: # no local read code if DirectToVgpr is enabled
              module.addComment1("prefetch local metadata")
              localReadCodeM, packCodeM, packPreM = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
              module.add(localReadCodeM)
              pack[plrIdx].add(packPreM) # no packPre scheduling for PGR0
              pack[plrIdx].add(packCodeM)
          if iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]:
            module.addComment1("prefetch local b")
            localReadCodeB, packCodeB, packPreB = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadB, iui*self.states.numReadsIterCoalescedB, 0, tensorParametersB)
            module.add(localReadCodeB)
            pack[plrIdx].add(packPreB) # no packPre scheduling for PGR0
            pack[plrIdx].add(packCodeB)

          if iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]:
            module.addComment0("local read increment a")
            module.add(self.localReadInc(kernel, iui, tensorParametersA))
          if kernel["ProblemType"]["MXBlockA"]:
            if iui*self.states.numReadsIterCoalescedMXSA < kernel["InnerUnroll"]:
              module.addComment0("local read increment mxsa")
              module.add(self.localReadInc(kernel, iui, tensorParametersA["MX"]))
          if kernel["ProblemType"]["MXBlockB"]:
            if iui*self.states.numReadsIterCoalescedMXSB < kernel["InnerUnroll"]:
              module.addComment0("local read increment mxsb")
              module.add(self.localReadInc(kernel, iui, tensorParametersB["MX"]))
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            numReadsIterCoalesced = self.states.numReadsIterCoalescedB if kernel["ProblemType"]["Sparse"] == 2 else self.states.numReadsIterCoalescedA
            if iui*numReadsIterCoalesced < kernel["InnerUnroll"]:
              module.addComment0("local read increment metadata")
              module.add(self.localReadInc(kernel, iui, tPM))
          if iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]:
            module.addComment0("local read increment b")
            module.add(self.localReadInc(kernel, iui, tensorParametersB))

    pflr = self.states.numItersPLR  # how many pf already done above

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

      if initCIter and uIdx==0:
        module.addComment2("Init C with wmma(mfma) - Begin")

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
            if "MX" in tensorParametersB:
              self.codes.localWriteMXSA = self.localWriteDo(kernel, tensorParametersB["MX"], swapAB=1)
            else:
              self.codes.localWriteMXSA = Module()
            if "MX" in tensorParametersA:
              self.codes.localWriteMXSB = self.localWriteDo(kernel, tensorParametersA["MX"], swapAB=1)
            else:
              self.codes.localWriteMXSB = Module()
            self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersA, swapAB=1)
          else:
            self.codes.localWriteA = self.localWriteDo(kernel, tensorParametersA)
            if "MX" in tensorParametersA:
              self.codes.localWriteMXSA = self.localWriteDo(kernel, tensorParametersA["MX"])
            else:
              self.codes.localWriteMXSA = Module()
            if "MX" in tensorParametersB:
              self.codes.localWriteMXSB = self.localWriteDo(kernel, tensorParametersB["MX"])
            else:
              self.codes.localWriteMXSB = Module()
            self.codes.localWriteB = self.localWriteDo(kernel, tensorParametersB)
        else:
          self.codes.localWriteA = Module()
          self.codes.localWriteMXSA = Module()
          self.codes.localWriteMXSB = Module()
          self.codes.localWriteB = Module()

        if not unrollLoopHeaderCodeScheduled:
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter=firstIter, lastLoop=False, lastLc=(lc==loopCopies-1))
          module.add(self.codes.unrollLoopHeader)

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = 1 if kernel["ForceUnrollSubIter"] else (u == localWriteEndIter)
      isSwapAndResetLwoIter = (u == localWriteEndIter)
      isSwapLroIter = isResetLroIter
      if kernel["_ScheduleIterAlg"] == 3:
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
      packPreIdx = (u+pflr) % self.states.numPackBuffer
      packIdx = u % self.states.numPackBuffer
      # hack: full pack prefetch case, move local read for next loop 1 iter ahead (no change for CMS)
      uNext = u
      if kernel["ForceUnrollSubIter"] and self.states.doFullPackCodePrefetch and u >= self.states.numVgprBuffer and not kernel["UseCustomMainLoopSchedule"]:
        if u == kernel["LoopIters"] - 2:
          uNext = kernel["LoopIters"] - 1
          self.states.SubTileIdx = 0 # hack: force to idx0 for next loop
        elif u == kernel["LoopIters"] - 1:
          uNext = kernel["LoopIters"] - 2
      packStoreIdx = (uNext+pflr) % self.states.numPackBuffer

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
          vregSetIdxLR = 0 if kernel["PrefetchGlobalRead"] >= 2 and kernel["PrefetchGlobalRead"] % 2 == 0 else 1

      localReads = Module()
      localReadsA = Module()
      localReadsMXSA = Module()
      localReadsMXSB = Module()
      localReadsB = Module()
      localReadsM = Module()
      localReadsSecondHalf = Module()

      pointerLWCode = Module()
      pointerLRCode = Module()
      waitCode = Module()  # may be overwritten (not added to) below
      macIterCode = Module()
      waitLWCode = Module()
      syncCode = Module()

      hasLiveLdsData = kernel["PrefetchGlobalRead"]
      # reads for current loop are done in previous iteration because of wider local read
      doReadA    = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadA - self.states.numItersPLR)
      doReadMXSA = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMXSA - self.states.numItersPLR)
      doReadMXSB = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMXSB - self.states.numItersPLR)
      doReadB    = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadB - self.states.numItersPLR)
      doReadM    = (u < kernel["LoopIters"]/self.states.numIterPerCoalescedReadMetadata - self.states.numItersPLR)

      if kernel["ForceUnrollSubIter"]:
        doReadA = 1 if u == 0 else 0
        doReadMXSA = 1 if u == 0 else 0
        doReadMXSB = 1 if u == 0 else 0
        doReadB = 1 if u == 0 else 0
        doReadM = 1 if u == 0 else 0

      # reads for next loop
      doNext = uNext > localWriteEndIter
      doReadA = doReadA or (hasLiveLdsData and doNext)
      doReadMXSA = doReadMXSA or (hasLiveLdsData and doNext)
      doReadMXSA = doReadMXSA and kernel["ProblemType"]["MXBlockA"]
      doReadMXSB = doReadMXSB or (hasLiveLdsData and doNext)
      doReadMXSB = doReadMXSB and kernel["ProblemType"]["MXBlockB"]
      doReadB = doReadB or (hasLiveLdsData and doNext)
      doReadM = doReadM or (hasLiveLdsData and doNext)
      doReadM = doReadM and (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"])
      # Prefetch reads for next loop target LDS1; current iteration reads target LDS0
      if hasLiveLdsData and doNext and not self.states.lockLdsReadTokenSwap:
        # swap LR buffer
        self.states.ldsReadTokenIdx = self.states.memTokenLdsBuffer1 if self.states.ldsReadTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0
      if kernel["HalfPLR"]:
        self.states.halfPLRGroups = self.getHalfPLRGroups(kernel, lc, (u+pflr))
      for iui in range(0,kernel["InnerUnroll"]):
        # use full prefetch only for next loop
        usePLRPackA = self.states.doFullPackCodePrefetch and (doNext or u == 0)
        usePLRPackB = self.states.doFullPackCodePrefetch and (doNext or kernel["ForceUnrollSubIter"])
        usePLRPackBNext = usePLRPackB and (kernel["ForceUnrollSubIter"] and u == 0)
        doReadA = doReadA and iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]
        doReadMXSA = doReadMXSA and iui*self.states.numReadsIterCoalescedMXSA < kernel["InnerUnroll"]
        doReadMXSB = doReadMXSB and iui*self.states.numReadsIterCoalescedMXSB < kernel["InnerUnroll"]
        doReadB = doReadB and iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]
        doReadM = doReadM and iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]
        if doReadA:
          localReads.addComment1("local read a")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadA
          if self.states.packDTVA or self.states.convDTVA:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadA + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeA, packCodeA, packPreA = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedA, 0, tensorParametersA)
          if kernel["HalfPLRA"]:
            lrCnt = countLocalRead(localReadCodeA)
            assert lrCnt % 2 == 0, "HalfPLR does not support odd number of local reads"
            halfCnt = lrCnt // 2
            readItems = localReadCodeA.flatitems()
            while readItems:
              item = readItems.pop(0)
              localReads.add(item)
              halfCnt -= countLocalRead(item)
              if halfCnt <= 0:
                assert halfCnt == 0, "two half not balanced"
                break
            tmpModule = Module()
            tmpModule.addItems(readItems)
            localReadsSecondHalf.add(tmpModule)
          else:
            localReads.add(localReadCodeA)
          localReadsA.add(localReadCodeA)
          # packPre code
          if doNext and self.states.doPackPreSchedulingNextLoop or self.states.doPackPreSchedulingThisLoop:
            # do pack pre scheduling for this loop. Put packPreCode to packPre
            packPre[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packPreA)
          else:
            # otherwise, put pack pre to pack
            pack[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packPreA)
          # pack code
          if usePLRPackA:
            # put pack code to packPre
            packPre[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packCodeA)
          else:
            pack[packStoreIdx*self.states.numIterPerCoalescedReadA].add(packCodeA)
          if kernel["UseCustomMainLoopSchedule"]:
            LRCodeAAllIters[uIdx].add(localReadCodeA)
            PackCodeAAllIters[uIdx].add(packPreA)
            PackCodeAAllIters[uIdx].add(packCodeA)
        if doReadMXSA:
          localReads.addComment1("local read maxa")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadMXSA
          if self.states.packDTVA or self.states.convDTVA:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadMXSA + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeMXSA, packCodeMXSA, packPreMXSA = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedMXSA, 0, tensorParametersA["MX"])
          localReads.add(localReadCodeMXSA)
          localReadsMXSA.add(localReadCodeMXSA)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSA].add(packPreMXSA)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSA].add(packCodeMXSA)
          if kernel["UseCustomMainLoopSchedule"]:
            LRCodeAAllIters[uIdx].add(localReadCodeMXSA)
            PackCodeAAllIters[uIdx].add(packPreMXSA)
            PackCodeAAllIters[uIdx].add(packCodeMXSA)
        if doReadMXSB:
          localReads.addComment1("local read mxsb")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadMXSB
          if self.states.packDTVB or self.states.convDTVB:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadMXSB + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeMXSB, packCodeMXSB, packPreMXSB = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedMXSB, 0, tensorParametersB["MX"])
          localReads.add(localReadCodeMXSB)
          localReadsMXSB.add(localReadCodeMXSB)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSB].add(packPreMXSB)
          pack[packStoreIdx*self.states.numIterPerCoalescedReadMXSB].add(packCodeMXSB)
          if kernel["UseCustomMainLoopSchedule"]:
            LRCodeAAllIters[uIdx].add(localReadCodeMXSB)
            PackCodeAAllIters[uIdx].add(packPreMXSB)
            PackCodeAAllIters[uIdx].add(packCodeMXSB)
        if doReadM:
          localReads.addComment1("local read metadata")
          plrIdxM = plrIdx*self.states.numIterPerCoalescedReadMetadata
          localReadCodeM, packCodeM, packPreM = self.localReadDo(kernel, plrIdxM, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
          localReads.add(localReadCodeM)
          localReadsM.add(localReadCodeM)
          pack[plrIdxM].add(packPreM)
          pack[plrIdxM].add(packCodeM)
        if doReadB:
          localReads.addComment1("local read b")
          bufferIdx = plrIdx*self.states.numIterPerCoalescedReadB
          if self.states.packDTVB or self.states.convDTVB:
            # DTV + pack or input conversion case, offset bufferIdx for local read packing instructions
            bufferIdx = plrIdxDTV*self.states.numIterPerCoalescedReadB + vregSetIdxLR * kernel["LoopIters"]
          localReadCodeB, packCodeB, packPreB = self.localReadDo(kernel, bufferIdx, iui*self.states.numReadsIterCoalescedB, 0, tensorParametersB)
          if kernel["HalfPLRB"]:
            lrCnt = countLocalRead(localReadCodeB)
            assert lrCnt % 2 == 0, "HalfPLR does not support odd number of local reads"
            halfCnt = lrCnt // 2
            readItems = localReadCodeB.flatitems()
            while readItems:
              item = readItems.pop(0)
              localReads.add(item)
              halfCnt -= countLocalRead(item)
              if halfCnt <= 0:
                assert halfCnt == 0, "two half not balanced"
                break
            tmpModule = Module()
            tmpModule.addItems(readItems)
            localReadsSecondHalf.add(tmpModule)
          else:
            localReads.add(localReadCodeB)
          localReadsB.add(localReadCodeB)
          # packPre code
          if doNext and self.states.doPackPreSchedulingNextLoop or self.states.doPackPreSchedulingThisLoop:
            # do pack pre scheduling for this loop. Put packPreCode to packPre
            packPre[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packPreB)
          else:
            # otherwise, put pack pre to pack
            pack[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packPreB)
          # pack code
          if usePLRPackB:
            # put pack code to packPre
            if usePLRPackBNext:
              # usePLRPackBNext case, put to the next pre pack
              packPre[(packStoreIdx+1)*self.states.numIterPerCoalescedReadB].add(packCodeB)
            else:
              packPre[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packCodeB)
          else:
            pack[packStoreIdx*self.states.numIterPerCoalescedReadB].add(packCodeB)
          if kernel["UseCustomMainLoopSchedule"]:
            LRCodeBAllIters[uIdx].add(localReadCodeB)
            PackCodeBAllIters[uIdx].add(packPreB)
            PackCodeBAllIters[uIdx].add(packCodeB)

        # Don't increment the LRO if we are going to reset them below:
        if not isResetLroIter or iui != kernel["InnerUnroll"]-1:
          if doReadA:
            localReads.addComment1("local read increment a")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA))
          if doReadMXSA:
            localReads.addComment1("local read increment mxsa")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersA["MX"]))
          if doReadMXSB:
            localReads.addComment1("local read increment mxsb")
            localReads.add(self.localReadInc(kernel, iui, tensorParametersB["MX"]))
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
        isBarrier = localWriteEndIter+1 if not self.states.doFullPackCodePrefetch else (u == self.states.syncPlrMfmaIndex // self.states.numMfmaPerIter)
        if (u == isBarrier) or \
           (self.states.doFullPackCodePrefetch and (u == self.states.syncPlrMfmaIndex // self.states.numMfmaPerIter)) or \
           (u == isBarrier%kernel["LoopIters"] and kernel["_ScheduleIterAlg"] == 2) or \
           (u == isBarrier%kernel["LoopIters"] and not self.states.numItersPLR \
            and kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["_ScheduleIterAlg"] == 0):
          # The 4th clause handles PLR=0 + TDM + SIA=0: numItersPLR==0 makes
          # isBarrier (= localWriteEndIter+1) equal to LoopIters, so the
          # canonical "u == isBarrier" emit above never fires inside the loop
          # body. The TDM tensor_load_to_lds issued at end of iter 0 would be
          # unsynced against the next trip's local reads from the swapped LDS
          # buffer. Emit the wait_tensorcnt+barrier at the wrapped position so
          # the previous trip's TDM has landed before this trip reads from it.
          # SIA gating is narrow on purpose: SIA=2 is handled above; SIA=3
          # uses a dedicated codegen path; SIA=1 isn't exercised with TDM.
          if (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and not self.states.scheduleGROverBarrier:
            vlcntVal = kernel["PrefetchGlobalRead"] - 1 if kernel["PrefetchGlobalRead"] >= 2 else 0
            waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, vlcntVal, -1, -1, \
                                      "wait for previous set of global reads"))
          elif kernel["enableTDMA"] and kernel["enableTDMB"]:
            # TDM case: tensor_load_to_lds instructions (issued in prior iter) write to LDS via the
            # tensor counter. A s_wait_tensorcnt 0 is required before the barrier to guarantee all
            # TDM stores to LDS have landed before other waves read from that LDS buffer.
            waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, \
                                      "wait for TDM global reads"))
          # (no local write code. Global read wait for DirectToLds is already done)
          if not kernel["NoLdsWriteCode"]:
            waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
          elif kernel["enableTDMA"] and kernel["enableTDMB"]:
            waitLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for TDM tensor loads"))
          skipForceWaitcnt0 = False
          if kernel["DirectToVgprA"] or kernel["DirectToVgprB"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"] or \
             (kernel["enableTDMA"] and kernel["enableTDMB"]):
            # DTVA/B, DTLA/B, or TDM case: global read wait is handled above, skip force waitcnt0
            skipForceWaitcnt0 = True
          syncCode.add(self._syncThreads(kernel, "PGR, and wait until LW done to sync", skipForceWaitcnt0=skipForceWaitcnt0))
          if kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["PrefetchGlobalRead"] == 1:
            syncCode.add(self._syncThreads(kernel, "PGR1 and TDM, another wait to sync", skipForceWaitcnt0=skipForceWaitcnt0))

        if isSwapAndResetLwoIter: # ResetLroIter
          if kernel["ExpertSchedulingMode"] > 0:
            pointerLWCode.add(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))
          if kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["_ScheduleIterAlg"] == 0 and kernel["PrefetchGlobalRead"] == 2:
            if not self.states.numItersPLR:
              pointerLWCode.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, \
                "wait for local read before cross-wave TDM swap sync"))
            pointerLWCode.add(self._syncThreads(
              kernel,
              "Waiting current LR finish for next GR(TDM), sync"))
          # local write for next iter, used to have local writes here
          # Swap offsets A(MXSA)
          if kernel["enableTDMA"]:
            pointerLWCode.addComment1("tdm swap offsets a")
            pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tensorParametersA))
          else:
            pointerLWCode.addComment1("local write swap offsets a")
            pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
          if kernel["ProblemType"]["MXBlockA"]:
            pointerLWCode.addComment1("local write swap offsets mxsa")
            if kernel["enableTDMA"]:
              pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tensorParametersA["MX"]))
            else:
              pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA["MX"]))
          # Swap offsets B(MXSB)
          if kernel["ProblemType"]["MXBlockB"]:
            if kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["NumWaves"] == 1:
              pointerLWCode.addComment1("local write swap offsets mxsb")
            elif not kernel["enableTDMB"]:
              pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB["MX"]))
          if kernel["enableTDMB"]:
            #TODO: TDM refactor
            if kernel["NumWaves"] == 1:
              pointerLWCode.addComment1("tdm swap offsets b")
              pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tensorParametersB))
          else:
            pointerLWCode.addComment1("local write swap offsets b")
            pointerLWCode.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
            if kernel["UseCustomMainLoopSchedule"]:
              LWSwapAAllIters.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
              LWSwapBAllIters.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
          # Swap offsets Metadata
          if kernel["enableTDMMetadata"]:
              pointerLWCode.addComment1("tdm swap offsets metadata")
              pointerLWCode.add(self.tdmSwapLdsOffset(kernel, tPM))
          # Swap local write memory token
          self.states.ldsWriteTokenIdx = \
            self.states.memTokenLdsBuffer1 if self.states.ldsWriteTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

        if isSwapLroIter: # ResetLroIter
          if kernel["ExpertSchedulingMode"] > 0:
            pointerLRCode.add(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))
          # Swap, reset, or increment the LRO:
          if not kernel["ForceUnrollSubIter"] or (doReadA and (u<localWriteEndIter)):
            pointerLRCode.addComment1("local read swap offsets a")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
            if kernel["UseCustomMainLoopSchedule"]:
              LRSwapAAllIters.add(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
          if kernel["ProblemType"]["MXBlockA"] and (not kernel["ForceUnrollSubIter"] or (doReadMXSA and (u<localWriteEndIter))):
            pointerLRCode.addComment1("local read swap offsets mxsa")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tensorParametersA["MX"]))
          if kernel["ProblemType"]["MXBlockB"] and (not kernel["ForceUnrollSubIter"] or (doReadMXSB and (u<localWriteEndIter))):
            pointerLRCode.addComment1("local read swap offsets mxsb")
            pointerLRCode.add(self.localReadSwapOffsets(kernel, expand, tensorParametersB["MX"]))
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
        if kernel["ProblemType"]["MXBlockA"]:
          pointerLRCode.addComment1("local read init pointers mxsa")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA["MX"]))
        if kernel["ProblemType"]["MXBlockB"]:
          pointerLRCode.addComment1("local read init pointers mxsb")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB["MX"]))
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          pointerLRCode.addComment1("local read init pointers metadata")
          pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))
        pointerLRCode.addComment1("local read init pointers b")
        pointerLRCode.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))
      if self.getConditionToSkipLocalReadWriteWait(kernel, u):
        waitCode = self._wait(kernel, tensorParametersA, tensorParametersB, \
            -1, 0, 0, \
            "wait for prior local read local write")

      luIdx = u % self.states.numVgprBuffer # local to use for MACs
      if kernel["EnableMatrixInstruction"]:
        mfmaIter = self.mfmaIter(kernel, tensorParametersA, tensorParametersB, u, kernel["InnerUnroll"], vregSetIdxMFMA, unrollLoopIdx=lc, unrollIdx = u, initCIterWmma=(initCIter and uIdx == 0))
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
        subIterCode = self._makeSubIterSchedule(kernel, tensorParametersA, tensorParametersB, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[packIdx], packPre[packPreIdx], module, localReadsSecondHalf)
        module.add(subIterCode) # add scheduled "other", local reads, local writes

      self.states.SubTileIdx = (self.states.SubTileIdx + 1) % kernel["numSubTiles"]
      # reset pack/packPre code
      pack[packIdx] = Module()
      packPre[packPreIdx] = Module()


    if kernel["UseCustomMainLoopSchedule"]:
      optSchedule, numCodePath = customMainLoopSchedule(self, kernel, tensorParametersA, tensorParametersB, globalReadIncACode, globalReadIncBCode, \
                                                   LRCodeAAllIters, PackCodeAAllIters, LRCodeBAllIters, PackCodeBAllIters, \
                                                   LRSwapAAllIters, LRSwapBAllIters, self.codes.globalReadA, self.codes.globalReadB, \
                                                   LWSwapAAllIters, LWSwapBAllIters, MfmaCodeAllIters, \
                                                   self.closeLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, False, nta=nta, ntb=ntb), nta=nta, ntb=ntb)
      module.add(optSchedule)
      module.add(self.simdSpecDispatch(kernel, numCodePath, nta=nta, ntb=ntb))

    # close unrolled loop
    endStr = ""
    if loopCopies == 2:
      finalStr = " (final)" if lc == 1 else ""
      endStr = " %u/%u%s"%(lc+1, loopCopies, finalStr)
    if not initCIter:
      module.addComment2("Unrolled Loop - End%s"%(endStr))

    if initCIter:
      module.addComment2("Init C with wmma(mfma) - End")
    oddLabel = (lc == 0 and loopCopies == 2)
    if not skipClose and not kernel["UseCustomMainLoopSchedule"]:
        if not initCIter:
          module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, finalLoop, oddLabel=oddLabel, nta=nta, ntb=ntb))
        elif loopCopies >= 2:
          # initC(lc=0) + multi-copy (EPS loopCopies=2 / HalfPLR loopCopies=3)
          # initC(lc=0) -> (lc=1) -> (lc=0 in EPS / lc=2 in HalfPLR) -> ...
          module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, finalLoop=False, oddLabel=(loopCopies == 2), nta=nta, ntb=ntb))
          module.add(SBranch(label_unrolledLoop_lc1.getLabelName()))
        else:
          # initC (no EPS, no HalfPLR): closeLoop(finalLoop=False) does dec + exit-check
          module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, finalLoop=False, oddLabel=False, nta=nta, ntb=ntb))
    return module

  ##############################################################################
  # Creates a negative identity matrix for 4x4x4_16b MFMA
  # Each 4x4 block is set to this:
  # -1  0  0
  #  0 -1  0
  #  0  0 -1
  ##############################################################################
  def createNegIdentityMatrix(self, kernel):
    module = Module("NegIdentityMatrix")
    tmp = self.vgprPool.checkOut(1, tag="createNegIdentityMatrix_tmpVgpr")
    lane4 = self.vgprPool.checkOut(1, tag="createNegIdentityMatrix_lane4")
    mfmaHigh = vgpr(self.states.startVgprIdentityMatrix)
    mfmaLow = vgpr(self.states.startVgprIdentityMatrix+1)

    module.addComment0("Create a negative identity matrix used by TF32 MFMA emulation.")
    module.add(VAndB32(dst=vgpr(lane4),src0=3, src1=vgpr("Serial"), comment="lane % 4"))
    module.add(VMovB64(dst=vgpr(self.states.startVgprIdentityMatrix,2),src=0))
    module.add(VMovB32(dst=vgpr(tmp), src="0xbf80", comment=""))

    module.add(VCmpEQU32(dst=VCC(), src0=0, src1=vgpr(lane4), comment="Lane %4 == 0 ?"))
    module.add(SNop(1, comment=""))
    module.add(VCndMaskB32(dst=mfmaHigh, src0=mfmaHigh, src1=vgpr(tmp), src2= VCC(), comment=""))

    module.add(VCmpEQU32(dst=VCC(), src0=2, src1=vgpr(lane4), comment="Lane %4 == 2 ?"))
    module.add(SNop(1, comment=""))
    module.add(VCndMaskB32(dst=mfmaLow, src0=mfmaLow, src1=vgpr(tmp), src2= VCC(), comment=""))

    module.add(VMovB32(dst=vgpr(tmp), src="0xbf800000", comment=""))

    module.add(VCmpEQU32(dst=VCC(), src0=1, src1=vgpr(lane4), comment="Lane %4 == 1 ?"))
    module.add(SNop(1, comment=""))
    module.add(VCndMaskB32(dst=mfmaHigh, src0=mfmaHigh, src1=vgpr(tmp), src2= VCC(), comment=""))

    module.add(VCmpEQU32(dst=VCC(), src0=3, src1=vgpr(lane4), comment="Lane %4 == 3 ?"))
    module.add(SNop(1, comment=""))
    module.add(VCndMaskB32(dst=mfmaLow, src0=mfmaLow, src1=vgpr(tmp), src2= VCC(), comment=""))

    self.vgprPool.checkIn(tmp)
    self.vgprPool.checkIn(lane4)
    return module

  ##############################################################################
  # GSU bit-mask helper (introduced for AdaptiveGemmNTAB)
  ##############################################################################
  def gsuMaskHex(self, kernel):
    # GSU bit width depends on InternalArgsSupport.version:
    #   v <  3: GSU is bits 0..13 (mask 0x3FFF)
    #   v >= 3: GSU narrowed to bits 0..11 (mask 0x0FFF) so bits 12/13
    #           can carry NTA / NTB for AdaptiveGemmNTAB.
    isp = kernel.get("InternalSupportParams", {})
    version = isp.get("KernArgsVersion", 0) if isp else 0
    return hex(0x0FFF) if version >= 3 else hex(0x3FFF)

  ##############################################################################
  # Kernel Body - Subtiled version
  ##############################################################################
  def kernelBodySubtile(self, kernel, tensorParametersA, tensorParametersB):
    # Store tensor params so emitter can access them for TDM descriptor reprogramming
    self.tPA = tensorParametersA
    self.tPB = tensorParametersB
    # First VGPR of the D-tile accumulator allocation.  On gfx1250 (MIArchVgpr),
    # WMMA writes directly to regular VGPRs, so the post-loop store path aliases
    # ValuC to these VGPRs instead of allocating a separate range.
    self._subtileDtileBaseVgpr = None
    #expand = kernel["ExpandPointerSwap"]
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
    if kernel.get("PrefetchAcrossPersistent"):
      module.add(SMovB32(dst=sgpr("SkPrefetchPrimed"), src=0, comment="PrefetchAcrossPersistent: not primed at kernel entry"))

    # Should check for is swizzled instead of usesubtileimpl
    # TODO: Move this calculation to host-side?
    if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockA"] and kernel["UseSubtileImpl"]:
      module.addComment("Scale StridesMXSA by 32")
      module.add(SLShiftLeftB32(sgpr("StridesMXSA"), 5, sgpr("StridesMXSA")))
      module.add(SLShiftLeftB32(sgpr("StridesMXSB"), 5, sgpr("StridesMXSB")))

    # Open persistent loop
    loopComponent = Component.PersistentLoop.find(self)

    module.add(loopComponent.openPersistentLoop(self, kernel))

    atileInfo = self.states.a.tileInfo
    btileInfo = self.states.b.tileInfo
    # TODO: Need corresponding ctileInfo for GSU/StreamK
    dtileInfo = self.states.d.tileInfo
    mxsatileInfo = self.states.mxsa.tileInfo if kernel["ProblemType"].get("MXBlockA", 0) else None
    mxsbtileInfo = self.states.mxsb.tileInfo if kernel["ProblemType"].get("MXBlockB", 0) else None

    module.addComment0("Number of subtiles for A: %u"%(len(atileInfo.localSubtiles)))
    module.addComment0("Number of subtiles for B: %u"%(len(btileInfo.localSubtiles)))

    module.add(self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isOptNLL=False))

    # Subtile defers TDM descriptor SGPR allocation until after setupNewTile
    # (which includes StreamK preLoop + graWorkGroup) so freed SK-constant
    # slots can serve as temps during those phases.
    if kernel.get("UseSubtileImpl"):
      module.addModuleAsFlatItems(self.defineTdmSgprs(kernel))
    module.add(self.removeGROffsetsVariableSgprsFromPool(kernel))
    #self.removeSgprVarFromPool("SrdD")
    #self.removeSgprVarFromPool("SrdC")

    ##
    # TODOBS: need to add init c code, and also init sum unroll code.
    #

    if not (kernel["enableTDMA"] and kernel["enableTDMB"]):
      module.add(globalReadDTLInitCommonSgpr(self, kernel))

    if mxsatileInfo != None and mxsbtileInfo != None:
      if not (kernel["enableTDMA"] and kernel["enableTDMB"]):
        module.add(globalReadScaleSwizzledDTLInitCommonSgpr(self, kernel))

    # TODOBS: globalWriteWorkGroupInit can be emitted here or later on, check..
    if self.states.doShadowInit:
      #module.add(self.openShadowInit())
      # SrdD/SrdC are used starting now, remove from sgpr pool
      self.removeSgprVarFromPool("SrdD")
      self.removeSgprVarFromPool("SrdC")
      self.removeSgprVarFromPool("SrdWS")
      module.add(self.globalWriteWorkGroupInit(kernel))
      #if self.states.doShadowInit == 2:
      #  module.add(self.initC(kernel)) # initC while waiting for global reads
      #  if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
      #    module.add(self.initSumUnroll(kernel))
      #module.add(self.closeShadowInit(kernel))


    hasTDM = kernel["enableTDMA"] and kernel["enableTDMB"]
    module.addComment1("global read addresses: addresses a")
    if not hasTDM:
      module.add(self.graAddresses(kernel, tensorParametersA))
      if kernel["ProblemType"]["MXBlockA"]:
        module.addComment1("global read addresses: addresses mxsa")
        module.add(self.graAddresses(kernel, tensorParametersA["MX"]))
      module.addComment1("global read addresses: addresses b")
      module.add(self.graAddresses(kernel, tensorParametersB))
      if kernel["ProblemType"]["MXBlockB"]:
        module.addComment1("global read addresses: addresses mxsb")
        module.add(self.graAddresses(kernel, tensorParametersB["MX"]))



    # List of tiles that need to be read form
    readtileInfoList = [atileInfo, btileInfo, mxsatileInfo, mxsbtileInfo]

    # Printout tile info
    for tileInfo in [atileInfo, btileInfo, mxsatileInfo, mxsbtileInfo, dtileInfo]:
      if tileInfo != None:
        module.addComment0(str(tileInfo))

    # Allocate registers for GR/LR
    for tileInfo in [atileInfo, btileInfo, mxsatileInfo, mxsbtileInfo]:
      if tileInfo is None:
        continue
      tileInfo.allocOffsetRegisters(self, kernel)
      module.addComment("Allocating v%s for %s GR"%(str(tileInfo.sharedVgprGROffset), tileInfo.tc))
      module.addComment("Allocating v%s for %s LR"%(str(tileInfo.sharedVgprLROffset), tileInfo.tc))
      module.addComment("Allocating v%s for %s LR Swap"%(str(tileInfo.sharedVgprLROffsetSwap), tileInfo.tc))

    if hasTDM:
      module.add(tdmGlobalOffsetSubtile(self, kernel, tensorParametersA))
      module.add(initTDMDescriptorSubtile(self, kernel, tensorParametersA))
      module.add(tdmGlobalOffsetSubtile(self, kernel, tensorParametersB))
      module.add(initTDMDescriptorSubtile(self, kernel, tensorParametersB))
    if not hasTDM:
      module.add(graTileAssignment(self, kernel))
    module.add(lraTileAssignment(self, kernel))
    module.add(localReadDTLInitCommonSwapVgpr(self, kernel))

    if not hasTDM:
      module.add(graTileAssignmentScaleSwizzled(self, kernel))
    module.add(lraTileAssignmentScaleSwizzled(self, kernel))

    module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx))

    # Apply StreamK K-offset to TDM global addresses.
    # calculateLoopNumIter sets StreamKLocalStart; offset Address{A,B} so
    # TDM reads from the correct K-slice when StreamK splits K across WGs.
    if hasTDM and kernel["StreamK"]:
      module.add(tdmApplyStreamKOffsetSubtile(self, kernel, tensorParametersA))
      module.add(tdmApplyStreamKOffsetSubtile(self, kernel, tensorParametersB))

    dtileInfo.allocVgprTileRegisters_legacy(self, kernel)

    if dtileInfo.vgprTiles:
      self._subtileDtileBaseVgpr = dtileInfo.vgprTiles[0].regList.indices[0]

    module.add(initVgprTilesToZero(self, kernel, dtileInfo))

    for vtiles in dtileInfo.vgprTiles:
      regStr = "Vgpr" if vtiles.regList.pool == self.vgprPool else "Agpr"
      module.addComment("%ss used for %s mma tile %u: %s"%(regStr, dtileInfo.tc, dtileInfo.vgprTiles.index(vtiles), str(vtiles)))

    if self.do["executeToPrefetchEnd"]:
      module.add(self.functionEnd(kernel, addLabel=False))

    module.add(mainLoop(self, kernel))

    # Deallocate offset registers
    for tileInfo in [atileInfo, btileInfo, mxsatileInfo, mxsbtileInfo]:
      if tileInfo is None:
        continue
      tileInfo.deallocOffsetRegisters(self, kernel)

    # For subtile kernels, free SGPRs that were only needed during the main loop
    self.states.storeAlign8 = False
    if kernel["UseSubtileImpl"]:
      # Remove SrdWS from the free pool before defineSgprIdx calls below,
      # otherwise checkOutAligned can grab registers overlapping SrdWS.
      if not self.states.doShadowInit and kernel["StreamK"] and kernel["StreamKAtomic"] == 0:
        self.removeSgprVarFromPool("SrdWS")

      result = self.undefineSubtileMainLoopSgprs(kernel)
      if result is not None:
        module.add(result)
      # Immediately allocate permanent SGPRs for subtile M/N guards.
      # Must be done here (before endSummation) so they get indices from
      # the freshly-freed swap/LocalWriteBaseAddr range.
      self.states.subtileM32ValidBlocksSgpr = self.defineSgprIdx("SubtileMGuard", 1)
      self.states.subtileN16ValidBlocksSgpr = self.defineSgprIdx("SubtileNGuard", 1)
      module.add(RegSet("s", "sgprSubtileMGuard", self.states.subtileM32ValidBlocksSgpr))
      module.add(RegSet("s", "sgprSubtileNGuard", self.states.subtileN16ValidBlocksSgpr))
      self.states.nonPostLoopSgpr.append("SubtileMGuard")
      self.states.nonPostLoopSgpr.append("SubtileNGuard")
      self.states.storeAlign8 = True

    # Start of post-loop code
    if 1:
      module.addComment0(" =============================================================== ")
      module.addComment0(" =================== Start of post-loop code =================== ")
      module.addComment0(" =============================================================== ")

      # ValuC aliases D-tile VGPRs on MIArchVgpr (see _subtileDtileBaseVgpr)
      if kernel["UseSubtileImpl"] and kernel["MIArchVgpr"] and self._subtileDtileBaseVgpr is not None:
        self.states.c.startVgprValu = self._subtileDtileBaseVgpr
      else:
        self.states.c.startVgprValu = self.vgprPool.checkOutAligned(1, 4, tag="postLoop_startVgprValu")

      module.addComment0("ValuC range: [%u-%u), %s"%(self.states.c.startVgprValu, self.states.c.startVgprValu+self.states.c.numVgprValu, \
                             "serializedStore enabled" if self.states.serializedStore else ""))
      module.add(RegSet("v", "vgprValuC", self.states.c.startVgprValu))
      self.states.serializedStore = True


      # SrdWS must be removed from the free pool before endSummation so that
      # defineSgpr() calls within endSummation don't see it as Available while
      # defineMultiSgprIndex (checkOutMulti) may have grabbed those registers.
      if not self.states.doShadowInit:
        self.removeSgprVarFromPool("SrdWS")
      module.add(self.endSummation(kernel, tensorParametersA, tensorParametersB))
      if not self.states.doShadowInit:
        self.removeSgprVarFromPool("SrdD")
        self.removeSgprVarFromPool("SrdC")
        module.add(self.globalWriteWorkGroupInit(kernel))

      ####################################
      # NOT LocalSplitU
      ####################################



      # global write indices
      module.addComment1("not-LocalSplitU: global write indices")
      module.add(self.notLocalSplitUGlobalWriteIndices(kernel))

      # global write
      #module.addComment1("not-LocalSplitU: global write")
      storeModule, deferredGSU0 = self.notLocalSplitUGlobalWrite(kernel, tensorParametersA, tensorParametersB)
      module.add(storeModule)

      if not (kernel["UseSubtileImpl"] and kernel["MIArchVgpr"] and self._subtileDtileBaseVgpr is not None):
        self.vgprPool.checkIn(self.states.c.startVgprValu)

    # Deallocate registers used for C/D tiles after store code instructions are emitted
    dtileInfo.deallocVgprTileRegisters_legacy(self, kernel)

    hasDeferredGSU0 = deferredGSU0 and len(deferredGSU0.items()) > 0
    hasDeferredFixup = hasattr(self.states, 'deferredFixupModule') and self.states.deferredFixupModule is not None
    hasDeferredEdge = hasattr(self.states, 'deferredEdgeModules') and self.states.deferredEdgeModules
    hasDeferredPartials = hasattr(self.states, 'deferredPartialsModule') and self.states.deferredPartialsModule is not None
    hasDeferredActivation = hasattr(self.states, 'deferredActivationModules') and self.states.deferredActivationModules is not None
    hasAnyDeferred = hasDeferredGSU0 or hasDeferredFixup or hasDeferredEdge or hasDeferredPartials

    if hasAnyDeferred:
      loopComponent = Component.PersistentLoop.find(self)
      module.add(loopComponent.closePersistentLoop(self, kernel))
      # After persistent loop exits, skip over all deferred blocks
      kernelEndLabel = Label("KernelEnd", "")
      with self.allocTmpSgpr(3, tag="kernelBodySubtile_tmpSgprInfo") as tmpSgprInfo:
        module.add(SLongBranchPositive(kernelEndLabel, tmpSgprInfo, comment="persistent loop done, skip deferred blocks"))
      module.addComment0("#" * 60)
      module.addComment0("#" * 60)
      module.addComment0("##")
      module.addComment0("##  DEFERRED BLOCKS START")
      module.addComment0("##  The following code blocks have been moved here from their")
      module.addComment0("##  original inline positions to keep the optimized NonEdge")
      module.addComment0("##  beta=0 store path close to the main loop.")
      module.addComment0("##  Each block is reached via unconditional branch from its")
      module.addComment0("##  original label stub and returns via branch-back.")
      module.addComment0("##")
      module.addComment0("#" * 60)
      module.addComment0("#" * 60)
      if hasDeferredFixup:
        module.appendModule(self.states.deferredFixupModule)
        self.states.deferredFixupModule = None
      if hasDeferredEdge:
        for edgeMod in self.states.deferredEdgeModules:
          module.appendModule(edgeMod)
        self.states.deferredEdgeModules = []
      if hasDeferredPartials:
        module.appendModule(self.states.deferredPartialsModule)
        self.states.deferredPartialsModule = None
      if hasDeferredGSU0:
        module.appendModule(deferredGSU0)
      if hasDeferredActivation:
        module.add(SBranch(labelName=kernelEndLabel.getLabelName(), comment="skip over activation functions"))
        module.appendModule(self.states.deferredActivationModules)
        self.states.deferredActivationModules = None
      module.add(kernelEndLabel)
      if kernel["ProblemType"]["OutputAmaxD"]:
        module.add(self.insertAmaxD(kernel))
      module.add(SEndpgm(comment="Kernel End"))
    else:
      # If activation was deferred but no other deferred blocks exist, emit it before functionEnd
      if hasDeferredActivation:
        module.add(SBranch(labelName="KernelEnd", comment="skip over activation functions"))
        module.appendModule(self.states.deferredActivationModules)
        self.states.deferredActivationModules = None
      module.add(self.functionEnd(kernel, addLabel=True))

    # Add a label at the end of the asm for indexing.
    module.add(Label("ASM_End", "The end of the kernel"))

    moduleKernelBody.addBody(module)
    self.checkResources(kernel, moduleKernelBody) # check resource available or not


    # TODO: Check what does this do and enable this if needed
    # Tensile instruction pass, temporarily disable due to build time.
    # Kernels with epilog especially with activation is too long (50000~ lines).
    # Need to refactor global write elements.
    #ripo = rocIsaPassOption()
    #ripo.removeDupFunc = bool(kernel["ActivationFuncCall"])
    #ripo.numWaves = kernel["NumThreads"] // kernel["WavefrontSize"]
    #if kernel["ProblemType"]["ActivationType"] == "all":
    #  ripo.removeDupAssign = False
    #if self.states.archCaps["HasSchedMode"]:
    #  ripo.insertDelayAlu = True
    #passResult = rocIsaPass(moduleKernelBody, ripo)
    #kernel["MathClocksUnrolledLoop"] = passResult.cycles


    error = self.states.overflowedResources
    print2(f"  found error code {error} with overflowed resources set to {self.states.overflowedResources}")

    return (error, str(moduleKernelBody))


  ##############################################################################
  # StreamK Constants In VGPRs
  ##############################################################################
  def isStreamKConstantsToVgprEnabled(self, kernel):
    # SK4 uses a different runtime-argument set and still references those
    # constants directly in the dynamic queue path.
    return kernel["ISA"] == IsaVersion(12,5,0) and kernel["StreamK"] != 4

  def acquireStreamKConstSgpr(self, kernel, name):
    if self.isStreamKConstantsToVgprEnabled(kernel):
      idx = self.sgprPool.checkOut(1, name, preventOverflow=False)
      if idx + 1 > self.states.regCaps["MaxSgpr"]:
        self.states.overflowedResources = 2
      return idx
    return name

  def releaseStreamKConstSgpr(self, nameOrIdx):
    if isinstance(nameOrIdx, int):
      self.sgprPool.checkIn(nameOrIdx)

  ##############################################################################
  # Move StreamK Constants to VGPRs
  ##############################################################################
  def moveStreamKConstantsToVgpr(self, kernel):
    """Move StreamK constant SGPRs (kernel args) to VGPRs to reduce SGPR pressure.

    Uses statically allocated VGPRs (startVgprSKConsts) that don't overlap with
    MXS/ValuAB/ValuC regions. At usage sites, v_readfirstlane_b32 brings values
    back to temp SGPRs as needed.
    """
    module = Module("Move StreamK constants to VGPRs")
    self.states.skConstVgprs = {}

    consts = ["ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile", "SKItersPerWG"]
    if kernel["StreamK"] >= 2:
      consts += ["skGrid", "skTiles"]

    baseVgpr = self.states.startVgprSKConsts
    for i, name in enumerate(consts):
      v = baseVgpr + i
      self.states.skConstVgprs[name] = v
      module.add(VMovB32(dst=vgpr(v), src=sgpr(name), comment="Save %s to VGPR v%u" % (name, v)))

    # Fully free the SGPR slots so defineVariableSgprs can reuse them.
    # undefineSgpr checks them back into sgprPool (Available) AND emits
    # .set UNDEF so the assembler catches any stale references.
    # addSgprVarToPool would only put them in freeSgprVarPool which
    # defineSgpr intentionally blocks from reuse (see defineSgpr lines 514-518).
    for name in consts:
      module.add(self.undefineSgpr(name))

    # StreamKIdx is a var (not kernel arg) — value set later in preLoop
    v = baseVgpr + len(consts)
    self.states.skConstVgprs["StreamKIdx"] = v

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
    module.add(self.disableWmmaArbStall())

    # gfx1250 moves SK constants to VGPRs inside defineAndResources so the
    # freed SGPR slots can be reused before defineVariableSgprs runs.

    # Initialize stream-k loop
    skComponent = Component.StreamK.find(self)
    module.add(skComponent.preLoop(self, kernel))
    if kernel.get("PrefetchAcrossPersistent"):
      module.add(SMovB32(dst=sgpr("SkPrefetchPrimed"), src=0, comment="PrefetchAcrossPersistent: not primed at kernel entry"))

    # MFMA F32XEmulation negative identity matrix
    if kernel["UseMFMAF32XEmulation"]:
      module.add(self.createNegIdentityMatrix(kernel))

    # Open persistent loop
    loopComponent = Component.PersistentLoop.find(self)

    module.add(loopComponent.openPersistentLoop(self, kernel))
    module.add(self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isOptNLL=False))

    if self.do["executeToPrefetchEnd"]:
      module.add(self.functionEnd(kernel, addLabel=False))
    self.states.numPackBuffer = 4 if kernel["ForceUnrollSubIter"] else self.states.numVgprBuffer # 2 buffers for ForceUnrollSubIter
    pack = [ Module() for i in range (self.states.numPackBuffer ) ]
    packPre = [ Module() for i in range (self.states.numPackBuffer ) ]
    self.preLoopLocalWriteCode = None

    # InitCIterWmma is resolved to 0/1 in SolutionStructs/Solution.py (-1 auto path).
    initCIterWmma = bool(kernel["InitCIterWmma"])

    if kernel["PrefetchGlobalRead"]:
      if self.states.doShadowInit:
        module.add(self.openShadowInit())
        # SrdD/SrdC are used starting now, remove from sgpr pool
        self.removeSgprVarFromPool("SrdD")
        self.removeSgprVarFromPool("SrdC")
        module.add(self.globalWriteWorkGroupInit(kernel))
        if self.states.doShadowInit == 2:
          module.add(self.initC(kernel, initCIterWmma)) # initC while waiting for global reads
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
      if not kernel["NoLdsWriteCode"]:
        preLoopLocalWriteCode = self.preLoopLocalWriteDo(kernel, tensorParametersA, tensorParametersB)
        module.add(preLoopLocalWriteCode)
      # elif kernel["enableTDMA"] and kernel["enableTDMB"]:
      #   preLoopLocalWriteCode = self.preLoopLocalWriteDoMX(kernel, tensorParametersA, tensorParametersB)
      #   module.add(preLoopLocalWriteCode)

      #TODO: TDM
      # Swap local ptrs A(MXSA)
      if kernel["enableTDMA"]:
        module.addComment1("TDM swap lds a")
        module.add(self.tdmSwapLdsOffset(kernel, tensorParametersA))
      else:
        module.addComment1("local write swap a")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      if "MX" in tensorParametersA:
        module.addComment1("local write swap mxsa")
        if kernel["enableTDMA"]:
          module.add(self.tdmSwapLdsOffset(kernel, tensorParametersA["MX"]))
        else:
          module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA["MX"]))
      # Swap local ptrs B(MXSB)
      if "MX" in tensorParametersB:
        module.addComment1("local write swap mxsb")
        if kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["NumWaves"] == 1:
          module.add(self.tdmSwapLdsOffset(kernel, tensorParametersB["MX"]))
        elif not kernel["enableTDMB"]:
          module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB["MX"]))

      if kernel["enableTDMB"]:
        #TODO: TDM refactor
        if kernel["NumWaves"] == 1:
          module.addComment1("TDM swap lds b")
          module.add(self.tdmSwapLdsOffset(kernel, tensorParametersB))
      else:
        module.addComment1("local write swap b")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
      # Swap Metadata
      if kernel["enableTDMMetadata"]:
          module.addComment1("TDM swap lds metadata")
          module.add(self.tdmSwapLdsOffset(kernel, tPM))
      # swap local write memory token
      self.states.ldsWriteTokenIdx = \
          self.states.memTokenLdsBuffer1 if self.states.ldsWriteTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

      # prefetch global read for PGR>=2
      if kernel["PrefetchGlobalRead"] >= 2:
        for idxPgr in range(1, kernel["PrefetchGlobalRead"]):
          module.add(self.openPrefetchGlobalRead2orMore(kernel, idxPgr))
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
          skipMXS1st = kernel["DirectToVgprMXS%s"%tc1] if ("MX" in tensorParameters1st) else True
          skipMXS2nd = kernel["DirectToVgprMXS%s"%tc2] if ("MX" in tensorParameters2nd) else True
          skip2nd = kernel["DirectToVgpr%s"%tc2]

          # skip wait for DTL if global load 1st is DTL
          skip1stWaitForDtl = idxPgr > 1 or kernel["NoLdsWriteCode"]
          skip2ndWaitForDtl = kernel["DirectToLds%s"%tc1] or idxPgr > 1
          if not skip1st:
            g2lBufIdx1st = 0
            if kernel["UnrollLoopSwapGlobalReadOrder"] == 1 or kernel["DirectToVgpr%s"%tc1]:
              # use second buffer
              g2lBufIdx1st = 1
            module.add(self.directToLdsM0Update(kernel, 1, tensorParameters1st, skip1stWaitForDtl))
            module.add(self.globalReadDo(kernel, 0, tensorParameters1st, g2lBufIdx=g2lBufIdx1st, tPM=tPM))
          if not skipMXS1st:
            g2lBufIdx1st = 0
            if kernel["UnrollLoopSwapGlobalReadOrder"] == 1 or kernel["DirectToVgprMXS%s"%tc1]:
              # use second buffer
              g2lBufIdx1st = 1
            module.add(self.directToLdsM0Update(kernel, 1, tensorParameters1st["MX"], skipWait=True))
            module.add(self.globalReadDo(kernel, 0, tensorParameters1st["MX"], g2lBufIdx=g2lBufIdx1st))
          if not skipMXS2nd:
            g2lBufIdx2nd = 0
            if kernel["UnrollLoopSwapGlobalReadOrder"] == 1 or kernel["DirectToVgprMXS%s"%tc2]:
              # use second buffer
              g2lBufIdx2nd = 1
            module.add(self.directToLdsM0Update(kernel, 1, tensorParameters2nd["MX"], skipWait=True))
            module.add(self.globalReadDo(kernel, 0, tensorParameters2nd["MX"], g2lBufIdx=g2lBufIdx2nd))
          if not skip2nd:
            g2lBufIdx2nd = 0
            if kernel["UnrollLoopSwapGlobalReadOrder"] == 1 or kernel["DirectToVgpr%s"%tc2]:
              # use second buffer
              g2lBufIdx2nd = 1
            module.add(self.directToLdsM0Update(kernel, 1, tensorParameters2nd, skip2ndWaitForDtl))
            module.add(self.globalReadDo(kernel, 0, tensorParameters2nd, g2lBufIdx=g2lBufIdx2nd, tPM=tPM))
          # Sparse: prefetch metadata global read for PGR>=2 (mirrors setupNewTile PGR=1 prefetch).
          if kernel["enableTDMMetadata"] and not kernel["DirectToVgprSparseMetadata"]:
            module.add(self.directToLdsM0Update(kernel, 1, tPM))
            module.add(self.globalReadDo(kernel, 0, tPM, g2lBufIdx=0))
          if idxPgr < kernel["PrefetchGlobalRead"] - 1:
            # generate GR inc code except for the last prefetch
            prefetchIdx = kernel["PrefetchGlobalRead"] - 1 - idxPgr
            module.add(self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, prefetchIdx))
            # swap Tensor memToken
            self.states.ldsTensorTokenIdx = \
                self.states.memTokenLdsBuffer1 if self.states.ldsTensorTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

          # swap local ptrs again if DirectToLds is enabled
          skipMetaSwap = kernel["ProblemType"]["Sparse"] and not kernel["DirectToLdsMetadata"]
          if kernel["DirectToLdsA"]:
            module.addComment1("local write swap a")
            module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA, prefetch=True, skipMetaSwap=skipMetaSwap))
          if ("MX" in tensorParametersA) and kernel["DirectToLdsMXSA"]:
            module.addComment1("local write swap mxsa")
            module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA["MX"]))
          if ("MX" in tensorParametersB) and kernel["DirectToLdsMXSB"]:
            module.addComment1("local write swap mxsb")
            module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB["MX"]))
          if kernel["DirectToLdsB"]:
            module.addComment1("local write swap b")
            module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB, prefetch=True, skipMetaSwap=skipMetaSwap))
          # swap ldsDirectToLDSTokenIdx
          self.states.ldsDirectToLDSTokenIdx = \
            self.states.memTokenLdsBuffer1 if self.states.ldsDirectToLDSTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

        # generate exit code
        for idxPgr in range(0, kernel["PrefetchGlobalRead"] + 1):
          module.add(self.closePrefetchGlobalRead2orMore(kernel, tensorParametersA, tensorParametersB, idxPgr))

      self.states.subTileIdx = 0

      # Init for 3LDSBlk. LRAddr related initialization
      if self.states.IncLdsBufSwitch:
        # init Sreg only. Call A only because Sreg is common for A and B
        module.add(self.lraAddressesInitFor3LDSBlk(kernel, tensorParametersA, True, False))

      # sync threads before prefetch-local
      if self.states.numItersPLR:
        # not generate wait for local write if LDS write code is not generated
        if not kernel["NoLdsWriteCode"]:
          module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "0prefetch wait for local write"))
        module.add(self._syncThreads(kernel, "LW to PLR, sync"))

        usePLRPack = self.states.doFullPackCodePrefetch or (kernel["UseCustomMainLoopSchedule"] and kernel["UsePLRPack"])

      # prefetch-local
      if self.states.numItersPLR:
        # in some cases need an extra copy of the LDS read with appropriate double buffer offsets
        for plrIdx in range(0, self.states.numItersPLR):
          packPre[plrIdx] = Module()
          pack[plrIdx] = Module()
          # we need to create separate module for prefetch pack
          # pack[plrIdx] will be updated in loopBody and the new code added loopBody will be also added prefetch part
          # if we insert pack[plrIdx] here.
          packPrePrefetchA = Module("Pack pre A code for prefetch")
          packPrePrefetchB = Module("Pack pre B code for prefetch")
          if kernel["HalfPLR"]:
            self.states.halfPLRGroups = self.getHalfPLRGroups(kernel, 0, 0)
          for espi in range(0, 1):
            for iui in range(0,kernel["InnerUnroll"]):
              if iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                module.addComment1("local read prefetch a")
                localReadCodeA, packCodeA, packPreA = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadA, iui*self.states.numReadsIterCoalescedA, espi, tensorParametersA)
                module.add(localReadCodeA)
                if self.states.doPackPreSchedulingNextLoop or usePLRPack:
                  packPrePrefetchA.add(packPreA)
                else:
                  pack[plrIdx].add(packPreA)
                if usePLRPack:
                  # usePLRPack case, generate all pack code at prefetch phase
                  # CMS only needs 1st PLRPack code insertion in preLoop, but localReadDo returns only 1st PLRPack code.
                  # No need to check iui==0
                  packPrePrefetchA.add(packCodeA)
                else:
                  pack[plrIdx].add(packCodeA)
              if kernel["ProblemType"]["MXBlockA"]:
                if iui*self.states.numReadsIterCoalescedMXSA < kernel["InnerUnroll"]:
                  module.addComment1("local read prefetch mxsa")
                  localReadCodeMXSA, packCodeMXSA, packPreMXSA = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMXSA, iui*self.states.numReadsIterCoalescedMXSA, espi, tensorParametersA["MX"])
                  module.add(localReadCodeMXSA)
                  pack[plrIdx].add(packPreMXSA)
                  pack[plrIdx].add(packCodeMXSA)
              if kernel["ProblemType"]["MXBlockB"]:
                if iui*self.states.numReadsIterCoalescedMXSB < kernel["InnerUnroll"]:
                  module.addComment1("local read prefetch mxsb")
                  localReadCodeMXSB, packCodeMXSB, packPreMXSB = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMXSB, iui*self.states.numReadsIterCoalescedMXSB, espi, tensorParametersB["MX"])
                  module.add(localReadCodeMXSB)
                  pack[plrIdx].add(packPreMXSB)
                  pack[plrIdx].add(packCodeMXSB)
              if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
                if iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]:
                  module.addComment1("local read prefetch metadata")
                  localReadCodeM, packCodeM, packPreM = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, espi, tPM)
                  module.add(localReadCodeM)
                  # no packPre scheduling support for M
                  pack[plrIdx].add(packPreM)
                  pack[plrIdx].add(packCodeM)
              if iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                module.addComment1("local read prefetch b")
                localReadCodeB, packCodeB, packPreB = self.localReadDo(kernel, plrIdx*self.states.numIterPerCoalescedReadB, iui*self.states.numReadsIterCoalescedB, espi, tensorParametersB)
                module.add(localReadCodeB)
                if self.states.doPackPreSchedulingNextLoop or usePLRPack:
                  packPrePrefetchB.add(packPreB)
                else:
                  pack[plrIdx].add(packPreB)
                if usePLRPack:
                  # usePLRPack case, generate all pack code at prefetch phase
                  # CMS only needs 1st PLRPack code insertion in preLoop, but localReadDo returns only 1st PLRPack code.
                  # No need to check iui==0
                  packPrePrefetchB.add(packCodeB)
                else:
                  pack[plrIdx].add(packCodeB)
              if not kernel["ForceUnrollSubIter"] and (iui*self.states.numReadsIterCoalescedA < kernel["InnerUnroll"]):
                module.addComment1("local read inc a")
                module.add(self.localReadInc(kernel, iui, tensorParametersA))
              if kernel["ProblemType"]["MXBlockA"] and (not kernel["ForceUnrollSubIter"] and (iui*self.states.numReadsIterCoalescedMXSA < kernel["InnerUnroll"])):
                if iui*self.states.numReadsIterCoalescedMXSA < kernel["InnerUnroll"]:
                  module.addComment1("local read inc mxsa")
                  module.add(self.localReadInc(kernel, iui, tensorParametersA["MX"]))
              if kernel["ProblemType"]["MXBlockB"] and (not kernel["ForceUnrollSubIter"] and (iui*self.states.numReadsIterCoalescedMXSB < kernel["InnerUnroll"])):
                if iui*self.states.numReadsIterCoalescedMXSB < kernel["InnerUnroll"]:
                  module.addComment1("local read inc mxsb")
                  module.add(self.localReadInc(kernel, iui, tensorParametersB["MX"]))
              if not kernel["ForceUnrollSubIter"] and (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]):
                if iui*self.states.numReadsIterCoalescedMetadata < kernel["InnerUnroll"]: # no local read code if DirectToVgpr is enabled
                  module.addComment1("local read inc metadata")
                  module.add(self.localReadInc(kernel, iui, tPM))
              if not kernel["ForceUnrollSubIter"] and (iui*self.states.numReadsIterCoalescedB < kernel["InnerUnroll"]):
                module.addComment1("local read inc b")
                module.add(self.localReadInc(kernel, iui, tensorParametersB))
          # Gather A, B conversion code based on scheduling order
          packPrePrefetchItems = []
          self._interleavePackAB(kernel, packPrePrefetchA.flatitems(), packPrePrefetchB.flatitems(), packPrePrefetchItems, prefetch=True)
          if len(packPrePrefetchItems) > 0:
            module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA and LRB to complete (for pre Pack code)"))

            module.addItems(packPrePrefetchItems)
          else:
            # no packPre code. Disable packPre scheduling
            self.states.doPackPreSchedulingThisLoop = False
            self.states.doPackPreSchedulingNextLoop = False

        self.states.SubTileIdx = (self.states.SubTileIdx + 1) % kernel["numSubTiles"]
      elif self.states.numItersPLR == 0 and kernel["UseCustomMainLoopSchedule"]:
        # For numItersPLR=0 and CMS we prefetch only half the local reads
        localReadCodeA = Module()
        localReadCodeB = Module()
        packCodeA = Module()
        packCodeB = Module()
        for espi in range(0, 1):
          localReadCodeA_, packCodeA_, packPreCodeA_ = self.localReadDo(kernel, 0, 0, espi, tensorParametersA)
          localReadCodeB_, packCodeB_, packPreCodeB_ = self.localReadDo(kernel, 0, 0, espi, tensorParametersB)
          localReadCodeA.add(localReadCodeA_)
          localReadCodeB.add(localReadCodeB_)
          packCodeA.add(packPreCodeA_)
          packCodeA.add(packCodeA_)
          packCodeB.add(packPreCodeB_)
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

    if kernel["PrefetchGL2"]:
      skipGL2Label = Label("Skip_GL2", "")
      loopCounter = self.loopCounter(kernel, self.states.unrollIdx)
      module.add(SCmpLeU32(src0=loopCounter, src1=hex(kernel["PrefetchGlobalRead"]), \
        comment="counterL<=PGR"))
      module.add(SCBranchSCC1(labelName=skipGL2Label.getLabelName(), comment=""))
      module.add(self.gl2PrefetchIssueLoad(kernel, tensorParametersA, tensorParametersB))
      if kernel["PrefetchGL2"] == 2:
        module.add(SCmpLeU32(src0=loopCounter, src1=hex(kernel["PrefetchGlobalRead"]+1), comment="counterL<=PGR+1"))
        module.add(SCBranchSCC1(labelName=skipGL2Label.getLabelName(), comment=""))
        module.add(self.gl2PrefetchIncrementAddr(kernel, tensorParametersA, tensorParametersB))
        module.add(self.gl2PrefetchIssueLoad(kernel, tensorParametersA, tensorParametersB))
      module.add(skipGL2Label)

    loopCopies = 3 if kernel["HalfPLR"] else 2 if expand else 1
    isDTV = (kernel["DirectToVgprA"] or kernel["DirectToVgprB"])
    isULSGRO = kernel["UnrollLoopSwapGlobalReadOrder"] == 1
    # DTLA+DTLB+PGR2 case, NGLL code is same and no need to genenerate even/odd NGLL
    isULSGROForNGLL = isULSGRO and (not (kernel["DirectToLdsA"] and kernel["DirectToLdsB"] and kernel["PrefetchGlobalRead"]>=2))
    needSecondLoop = (not expand) and (isULSGRO or isDTV) # need 2 MainLoop for 2 buffers (PGR1/2)
    needSecondNGLL = (not expand) and (isULSGROForNGLL or isDTV) and kernel["PrefetchGlobalRead"] >= 2# need 2 NGLL for 2 buffers (PGR>=2)
    if loopCopies == 1 and needSecondLoop:
      # force to generate 2 loop bodies
      loopCopies = 2

    loopLabelToNoGRloopAfterABLoop = Label("NoGRloopAfterABLoop", "" )

    def _kernelBody(pack, packPre, nta, ntb):
      # open unrolled summation loop
      module.addComment2("Unrolled Loop(s) - Begin")
      if kernel["enableTDMA"] and kernel["enableTDMB"] and not kernel["PrefetchGlobalRead"]:
        module.add(SBarrier(comment="TDM PGR=0: prime barrier before loop"))
      module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx, beginLabelOnly=False, beforeInitCIter=initCIterWmma, nta=nta, ntb=ntb))

      #init C with wmma(mfma) instead of v_mov, then jump to unrolled loop iter1
      if initCIterWmma:
        temp_states = deepcopy(self.states)
        temp_kernel = deepcopy(kernel)
        temp_A = deepcopy(tensorParametersA)
        temp_B = deepcopy(tensorParametersB)
        temp_pack = deepcopy(pack)
        temp_packPre = deepcopy(packPre)
        module.add(self._loopBody( temp_kernel, temp_A, temp_B, temp_pack, temp_packPre, 0, loopCopies, 0==(loopCopies-1), \
                                   isDTVGRSecondBuf=isDTV, initCIter=initCIterWmma, nta=nta, ntb=ntb))
        module.add(self.openLoop( temp_kernel, temp_A, temp_B, self.states.unrollIdx, beginLabelOnly=True, nta=nta, ntb=ntb))
        self.states = temp_states

      loop = Module("loopBody")
      if needSecondLoop and kernel["PrefetchGlobalRead"] >= 2:
        # force to generate 2 loop bodies (PGR2 only)
        # TODO: unify 2 loop bodies code generation with else case

        # second LW buffer check for UnrollLoopSwapGlobalReadOrder
        dsWriteBA = True if isULSGRO else False
        # second GR buffer check for DTV
        isDTVGRSecondBuf = True if isDTV else False
        loop.add(self._loopBody( kernel, tensorParametersA, tensorParametersB, pack, packPre, 0, loopCopies, False , dsWriteBA=dsWriteBA, isDTVGRSecondBuf=isDTVGRSecondBuf, skipClose=True, nta=nta, ntb=ntb))

        # loop counter decrement
        loopCounter = self.loopCounter(kernel, self.states.unrollIdx)
        loop.add(SSubU32(dst=loopCounter, src0=loopCounter, \
                           src1=1, \
                           comment="dec counterL"))
        endCounter = kernel["PrefetchGlobalRead"]
        loop.add(SCmpLeU32(src0=loopCounter, \
                             src1=hex(endCounter), \
                            comment="counteL<=%d"%endCounter))
        loop.add(SCBranchSCC1(labelName=loopLabelToNoGRloopAfterABLoop.getLabelName(), comment="exit LoopL" ))
        # grBA check for UnrollLoopSwapGlobalReadOrder
        grBA = True if isULSGRO else False
        loop.add(self._loopBody( kernel, tensorParametersA, tensorParametersB, pack, packPre, 1, loopCopies, True , grBA=grBA, nta=nta, ntb=ntb))
      else:
        for lc in range(0, loopCopies):
          # second GR buffer check for DTV
          isDTVGRSecondBuf = True if isDTV and lc == 0 else False
          # loop body code generation
          finalLoop = lc == loopCopies - 1
          loop.add(self._loopBody( kernel, tensorParametersA, tensorParametersB, pack, packPre, lc, loopCopies, finalLoop, isDTVGRSecondBuf=isDTVGRSecondBuf, nta=nta, ntb=ntb ))
          if self.states.numItersPLR == 0 and not finalLoop:
            # swap LDS read buffer
            self.states.ldsReadTokenIdx = self.states.memTokenLdsBuffer1 if self.states.ldsReadTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

      module.add(loop)

    if kernel["AdaptiveGemmNTAB"] == 0:
      _kernelBody(pack, packPre, 0, 0)
    else:
      # AdaptiveGemmNTAB: dispatch is decided host-side and forwarded via
      # internalArg0 bits 12 (NTA) and 13 (NTB). Host clears those bits when
      # the chosen NT value would be 0, and only sets them after compressing
      # GSU into bits 0..11. Requires InternalArgsSupport.version >= 3, which
      # codegen guarantees by bumping KernArgsVersion when AdaptiveGemmNTAB!=0.
      originalNta = tensorParametersA["NonTemporal"]
      originalNtb = tensorParametersB["NonTemporal"]

      ntCombos = [[0, 0], [0, 4], [4, 0]]
      ntLabels = [Label("LoopBody_NTA{}_NTB{}".format(nta, ntb), "") for nta, ntb in ntCombos]
      ntLabelDone = Label("LoopBody_NTA_NTB_Done", "")

      # Bit layout in sgpr("GSU") (== internalArg0 low 16b):
      #   bit 12 -> NTA (1 means use NTA=4)
      #   bit 13 -> NTB (1 means use NTB=4)
      kNtaMask = 0x1000
      kNtbMask = 0x2000

      module.addComment1("AdaptiveGemmNTAB: 3-way bit-based dispatch from internalArg0")

      with self.allocTmpSgpr(1) as tmpNtBitsInfo:
        tmpNtBits = tmpNtBitsInfo.idx
        # Extract both NT bits then dispatch on the 3 combos.
        # ntCombos order: [0,0]=idx0, [0,4]=idx1, [4,0]=idx2
        # Use long branches because each kernelBody is large (>128KB possible).
        module.add(SAndB32(dst=sgpr(tmpNtBits), src0=sgpr("GSU"),
                           src1=hex(kNtaMask | kNtbMask), comment="extract NTA|NTB bits"))
        module.add(SCmpEQU32(src0=sgpr(tmpNtBits), src1=hex(kNtbMask),
                             comment="NTB=4 only ?"))
        module.add(self.longBranchScc1(ntLabels[1], 1, comment="-> NTA0_NTB4"))
        module.add(SCmpEQU32(src0=sgpr(tmpNtBits), src1=hex(kNtaMask),
                             comment="NTA=4 only ?"))
        module.add(self.longBranchScc1(ntLabels[2], 1, comment="-> NTA4_NTB0"))
        # Fall through to ntLabels[0] (NTA0_NTB0)

      # _kernelBody() mutates self.states (ldsWriteTokenIdx,
      # setMemTokenInsts, localReadDoCnt*, ...), the pack/packPre dicts
      # and the tensorParameters dicts as a side-effect of codegen.
      # Because only one body executes at runtime but all are emitted
      # back-to-back, we must checkpoint this state before the first
      # body and restore it before every subsequent body, otherwise the
      # later bodies are generated from a corrupted starting state
      # (e.g. wrong LDS swap buffer / local-read offsets) and produce
      # garbage at runtime.
      stateSnap   = deepcopy(self.states.__dict__)
      packSnap    = deepcopy(pack)
      packPreSnap = deepcopy(packPre)
      tPASnap     = deepcopy(tensorParametersA)
      tPBSnap     = deepcopy(tensorParametersB)

      def _restoreNtabState():
        # pack / packPre are lists of Module objects; tensorParameters* are dicts.
        # Rebuild contents in-place so references held by callers remain valid.
        # Also preserve aliases that existed before the NTAB snapshot. Codegen
        # often assumes kernel is self.states.kernel and mutates that shared
        # metadata in-place. If another restored metadata object must remain
        # aliased with an outer active object, explicitly rebind it here as well.
        self.states.__dict__.clear()
        self.states.__dict__.update(deepcopy(stateSnap))
        self.states.kernel = kernel
        pack[:]    = deepcopy(packSnap)
        packPre[:] = deepcopy(packPreSnap)
        tensorParametersA.clear(); tensorParametersA.update(deepcopy(tPASnap))
        tensorParametersB.clear(); tensorParametersB.update(deepcopy(tPBSnap))

      for idx, (nta, ntb) in enumerate(ntCombos):
        if idx > 0:
          _restoreNtabState()
        module.add(ntLabels[idx])
        tensorParametersA["NonTemporal"] = nta
        tensorParametersB["NonTemporal"] = ntb
        _kernelBody(pack, packPre, nta, ntb)
        # All paths but the last one need to skip the rest. Use long
        # branches everywhere because each kernelBody can easily exceed
        # the +/-128KB reach of a short s_branch.
        if idx < len(ntCombos) - 1:
          with self.allocTmpSgpr(3) as tmpSgprInfo:
            module.add(SLongBranchPositive(ntLabelDone, tmpSgprInfo,
                                           comment="long jump to NTAB done"))

      module.add(ntLabelDone)

      tensorParametersA["NonTemporal"] = originalNta
      tensorParametersB["NonTemporal"] = originalNtb

    if kernel["ExpertSchedulingMode"] > 0:
      module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0,2]), src=0x0, comment="enable hardware dependency checking"))
    if kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["_ScheduleIterAlg"] == 0 and kernel["PrefetchGlobalRead"] == 2:
      module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0,2]), src=0x0, comment="disable expert scheduling mode = 0"))
    module.addComment1("Before NLL: Check VGPR.checkin for INT8 LW")

    # swap local write, read again before noLoadLoop if PrefetchGlobalRead and DirectToLds is enabled
    # In DirectToLds enabled case, local write address is necessary for prefetch global read (for m0).
    # However, even exit with DirectToLds will not pass with this code (limitation).
    if kernel["PrefetchGlobalRead"] and kernel["ExpandPointerSwap"]:
      # local write for next iter, used to have local writes here
      if(kernel["DirectToLdsA"]):
        module.addComment1("local write swap offsets a")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      if ("MX" in tensorParametersA) and (kernel["DirectToLdsMXSA"]):
        module.addComment1("local write swap offsets mxsa")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersA["MX"]))
      if ("MX" in tensorParametersB) and (kernel["DirectToLdsMXSB"]):
        module.addComment1("local write swap offsets mxsb")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB["MX"]))
      if(kernel["DirectToLdsB"]):
        module.addComment1("local write swap offsets b")
        module.add(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
      # swap ldsDirectToLDSTokenIdx
      self.states.ldsDirectToLDSTokenIdx = \
          self.states.memTokenLdsBuffer1 if self.states.ldsDirectToLDSTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0

    for remainPgr in range(kernel["PrefetchGlobalRead"]-1, 0, -1) if not kernel["SuppressNoLoadLoop"] else []:
      # NGLL code generation for PGR>=2
      NGLLindex = 0
      NGLLnum = 2 if needSecondNGLL else 1
      if needSecondNGLL:
        # generate extra NGLL for second GR buffer
        module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=True, pack=pack, packPre=packPre, \
                                   NLLindex=NGLLindex, NLLnum=NGLLnum, remainPgr=remainPgr))
        module.add(loopLabelToNoGRloopAfterABLoop)
        if kernel["ExpertSchedulingMode"] > 0:
          module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0,2]), src=0x0, comment="enable hardware dependency checking"))
        NGLLindex += 1
      elif isULSGRO and not isULSGROForNGLL and remainPgr == kernel["PrefetchGlobalRead"]-1:
        # generate loopLabelToNoGRloopAfterABLoop at first iteration of remainPgr
        # This is to jump to the same NGLL from even and odd main loop
        module.add(loopLabelToNoGRloopAfterABLoop)
      module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=True, pack=pack, packPre=packPre, \
                                 NLLindex=NGLLindex, NLLnum=NGLLnum, remainPgr=remainPgr))

    # This "NoLoad" loop is a copy of the unroll loop but with global loads + LDS writes removed
    # doShadowInit is required since this pushes up the store SRD initialization before the NLL
    # OptNLL only allowed for single summation index  - for multiple summation we (currently)
    # execute the NLL inside each unroll iteration not just once at the end.
    if kernel["PrefetchGlobalRead"]:
      if not kernel["SuppressNoLoadLoop"]:
        if self.states.tailloopInNll:
          # deepCopy packCode for tailloopInNll
          backupPack = deepcopy(pack)
          backupPackPre = deepcopy(packPre)
        NeedNLLOddEven  = isDTV # need odd+even NLL for 2 buffers (PGR1/2)
        NLLnum = 2 if NeedNLLOddEven else 1
        gsuComponent = Component.GSU.find(self)
        module.add(gsuComponent.noLoadLoop(self, kernel, tensorParametersA, tensorParametersB, pack, packPre))
        for NLLindex in range(0, NLLnum):
          self.saveLocalPointers(kernel, tensorParametersA, tensorParametersB)
          # copy pack
          if NLLindex == NLLnum - 1 or (self.states.packDTVA or self.states.packDTVB or self.states.convDTVA or self.states.convDTVB):
            # last NLL or  pack DTV case, no deep copy for pack
            # pack code for local prefetch is generated in noLoadLoopBody and used for DTV even
            deepCopyPack = pack
            deepCopyPackPre = packPre
          else:
            # deepCopy packCode for OptNLL noLoadLoop
            deepCopyPack = deepcopy(pack)
            deepCopyPackPre = deepcopy(packPre)
          module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=False, pack=deepCopyPack, packPre=deepCopyPackPre, \
                                     NLLindex=NLLindex, NLLnum=NLLnum, useTailloopInNll=self.states.tailloopInNll))
          self.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)

        if self.states.tailloopInNll:
          # tailloopInNll case, generate another set of NoLoadLoop for tailloopInNll not applicable case
          # restore backup for pack code
          pack = backupPack
          packPre = backupPackPre
          for NLLindex in range(0, NLLnum):
            self.saveLocalPointers(kernel, tensorParametersA, tensorParametersB)
            # copy pack
            if NLLindex == NLLnum - 1 or (self.states.packDTVA or self.states.packDTVB or self.states.convDTVA or self.states.convDTVB):
              # last NLL or  pack DTV case, no deep copy for pack
              # pack code for local prefetch is generated in noLoadLoopBody and used for DTV even
              deepCopyPack = pack
              deepCopyPackPre = packPre
            else:
              # deepCopy packCode for OptNLL noLoadLoop
              deepCopyPack = deepcopy(pack)
              deepCopyPackPre = deepcopy(packPre)
            module.add(self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=False, pack=deepCopyPack, packPre=deepCopyPackPre, NLLindex=NLLindex, NLLnum=NLLnum))
            self.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)

    self.postMainLoopBarrierCheckAndReset(kernel, module)

    if self.states.actualSummationLoops>1 and self.states.staggerUCode:
      module.addComment1("remove stagger offsets")
      module.add(self.removeStaggerAB(kernel, tensorParametersA, tensorParametersB))

    module.add(VNop(self.states.miVALUInstrDataHazard, "Add v_nop before releasing ValuA/B"))
    self.vgprPool.add(self.states.a.startVgprValu , \
        self.states.lastValuAB - self.states.a.startVgprValu, "ValuAB")
    module.addComment1("Tail: add ValuA/B vgpr buffer [%u...%u) to pool" % \
        (self.states.a.startVgprValu, self.states.lastValuAB))
    if not self.isPrefetchAcrossPersistentEnabled(kernel):
      self.vgprPool.add(self.states.lastValuAB , \
          self.states.lastVgprForReads - self.states.lastValuAB, "address vgpr")
      module.addComment1("Tail: add address/G2L vgpr [%u...%u) to pool" % \
          (self.states.lastValuAB, self.states.lastVgprForReads))
    else:
      module.addComment1("PAP: G2L vgpr [%u...%u) held for prefetched data" % \
          (self.states.lastValuAB, self.states.lastVgprForReads))

    self.removeSgprVarFromPool("SrdWS")

    if not kernel["NoTailLoop"]:
      ########################################
      # Tail Loop
      # which means tail loop needed.
      ########################################
      self.states.inTailLoop = True
      module.addComment2("Tail Loop")

      # F32XEmu case, need to disable index transpose and generate transpose code for wider local read
      # Disable doFullPackCodePrefetchas.
      # Not TF32EmuInterleaveTreg case, disable useDirect32XEmulationNext
      if kernel["UseF32XEmulation"]:
        self.states.doFullPackCodePrefetch = False
        if not self.states.a.TF32EmuInterleaveTreg:
          self.states.a.useDirect32XEmulationThis = False
          self.states.a.useDirect32XEmulationNext = False
        if self.states.lrvwTileA > 1:
          self.states.a.useTransposeCodeThis = True
          self.states.a.useTransposeCodeNext = True
        if not self.states.b.TF32EmuInterleaveTreg:
          self.states.b.useDirect32XEmulationThis = False
          self.states.b.useDirect32XEmulationNext = False
        if self.states.lrvwTileB > 1:
          self.states.b.useTransposeCodeThis = True
          self.states.b.useTransposeCodeNext = True

      # need to unroll tail loop for the following cases
      mEnd = 1
      if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"]:
        mEnd = kernel["LoopIters"]
      if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
        mEnd = kernel["DepthU"]//(kernel["MatrixInstK"]*kernel["LocalSplitU"])

      # Update local write pointers in case the upcoming global reads are writing directly to LDS:
      if kernel["PrefetchGlobalRead"]:
        #TODO: TDM
        module.addComment1("local write reset offsets a")
        if not kernel["enableTDMA"]:
          module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersA))
          if kernel["ExpandPointerSwap"]:
            # reset local write offset in asm code as well
            module.add(self.localWriteResetOffsets(kernel, False, tensorParametersA))

        module.addComment1("local write reset offsets b")
        if not kernel["enableTDMB"]:
          module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersB))
          if kernel["ExpandPointerSwap"]:
            # reset local write offset in asm code as well
            module.add(self.localWriteResetOffsets(kernel, False, tensorParametersB))

        if kernel["ProblemType"]["MXBlockA"]:
          if not kernel["enableTDMA"]:
            module.addComment1("local write reset offsets mxsa")
            module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersA["MX"]))
            if kernel["ExpandPointerSwap"]:
              # reset local write offset in asm code as well
              module.add(self.localWriteResetOffsets(kernel, False, tensorParametersA["MX"]))

        if kernel["ProblemType"]["MXBlockB"]:
          if not kernel["enableTDMA"]:
            module.addComment1("local write reset offsets mxsb")
            module.add(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersB["MX"]))
            if kernel["ExpandPointerSwap"]:
              # reset local write offset in asm code as well
              module.add(self.localWriteResetOffsets(kernel, False, tensorParametersB["MX"]))

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

      is_wmma_v3 = self.states.asmCaps.get("HasWMMA_V3", False)
      if not is_wmma_v3:
        module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, -1))
      if self.states.actualSummationLoops==1 and self.states.staggerUCode:
        module.addComment1("remove stagger offsets for tail loop")
        if is_wmma_v3:
          skipRemoveStaggerLabel = Label("SkipRemoveStagger", "")
          module.add(SCmpEQU32(src0=sgpr("OrigLoopCounter"), src1=0, comment="skip if main loop was not executed"))
          module.add(SCBranchSCC1(labelName=skipRemoveStaggerLabel.getLabelName(), comment="skip removeStagger"))
        module.add(self.removeStaggerAB(kernel, tensorParametersA, tensorParametersB))
        if is_wmma_v3:
          module.add(skipRemoveStaggerLabel)
      if is_wmma_v3:
        module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, -1))

      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      tailLoopOpt1st = kernel["tailLoopOptA"] and self.do["GlobalReadA"]
      tailLoopOpt2nd = kernel["tailLoopOptB"] and self.do["GlobalReadB"]

      tc1 = 'A'
      tc2 = 'B'

      # if swapGlobalRoad is true, swap the order of global read (B->A)
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

      if (kernel["enableTDMA"] or kernel["enableTDMB"]) and not kernel["1LDSBuffer"]:
        module.add(self._syncThreads(kernel, "Barrier before tail TDM loads (WAR hazard with NLL LDS reads)"))

      if kernel["enableTDMA"] and kernel["enableTDMB"]:
        if kernel["NumWaves"] > 1:
          if self.isPrefetchAcrossPersistentEnabled(kernel):
            module.add(self.papResetTDMDescriptorForTailWaveSeparated(kernel, tensorParametersA, tensorParametersB))
          else:
            module.add(self.resetTDMDescriptorForTailWaveSeparated(kernel, tensorParametersA, tensorParametersB))
          if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
            if self.isPrefetchAcrossPersistentEnabled(kernel):
              module.add(self.papResetTDMDescriptorForTailWaveSeparated(kernel, tensorParametersA["MX"], \
                                                                        tensorParametersB["MX"]))
            else:
              module.add(self.resetTDMDescriptorForTailWaveSeparated(kernel, tensorParametersA["MX"], \
                                                                     tensorParametersB["MX"]))
        else:
          module.add(self.resetTDMDescriptorForTail(kernel, tensorParametersA))
          module.add(self.resetTDMDescriptorForTail(kernel, tensorParametersB))
          if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
            module.add(self.resetTDMDescriptorForTail(kernel, tensorParametersA["MX"]))
            module.add(self.resetTDMDescriptorForTail(kernel, tensorParametersB["MX"]))

      # reset memToken for tail loop
      self.states.ldsWriteTokenIdx = self.states.memTokenLdsBuffer0
      self.states.ldsReadTokenIdx = self.states.memTokenLdsBuffer0
      module.addComment1("Update M0 for DTLDS")
      moduleTmp = self.directToLdsM0Update(kernel, 2, tensorParameters1st)
      module.add(replaceHolder(moduleTmp, 0))
      module.addComment1("Tail global read %s"%tc1)
      if tailLoopOpt1st and (globalReadMode1st == 2):
        module.add(self.doTailLoopOpt(kernel, tensorParameters1st))
        module.addComment1("Update M0 for DTLDS")
        moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters2nd, True)
        module.add(replaceHolder(moduleTmp, 0))
        module.addComment1("Tail global read %s"%tc2)
        if tailLoopOpt2nd and (globalReadMode2nd == 2):
          module.add(self.doTailLoopOpt(kernel, tensorParameters2nd))
        else:
          module.add(self.globalReadDo(kernel, globalReadMode2nd, tensorParameters2nd))
      else:
        # skip wait for DTL if global load 1st is DTL
        skip2ndWaitForDtl = kernel["DirectToLds%s"%tc1]
        module.add(self.globalReadDo(kernel, globalReadMode1st, tensorParameters1st))

        if "MX" in tensorParameters1st:
          module.addComment1("Update M0 for DTLDS")
          moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters1st["MX"], skipWait=skip2ndWaitForDtl)
          module.add(replaceHolder(moduleTmp, 0))
          module.addComment1("Tail global read MXS%s"%tc1)
          if tailLoopOpt1st and (globalReadMode1st == 2):
            module.add(self.doTailLoopOpt(kernel, tensorParameters1st["MX"]))
          else:
            module.add(self.globalReadDo(kernel, globalReadMode1st, tensorParameters1st["MX"]))

        if "MX" in tensorParameters2nd:
          module.addComment1("Update M0 for DTLDS")
          moduleTmp = self.directToLdsM0Update(kernel, 1, tensorParameters2nd["MX"], skipWait=skip2ndWaitForDtl)
          module.add(replaceHolder(moduleTmp, 0))
          module.addComment1("Tail global read MXS%s"%tc2)
          if tailLoopOpt2nd and (globalReadMode2nd == 2):
            module.add(self.doTailLoopOpt(kernel, tensorParameters2nd["MX"]))
          else:
            module.add(self.globalReadDo(kernel, globalReadMode2nd, tensorParameters2nd["MX"]))

        module.addComment1("Update M0 for DTLDS")
        moduleTmp = self.directToLdsM0Update(kernel, 2, tensorParameters2nd, skip2ndWaitForDtl)
        module.add(replaceHolder(moduleTmp, 0))
        module.addComment1("Tail global read %s"%tc2)
        if tailLoopOpt2nd and (globalReadMode2nd == 2):
          module.add(self.doTailLoopOpt(kernel, tensorParameters2nd))
        else:
          module.add(self.globalReadDo(kernel, globalReadMode2nd, tensorParameters2nd))

      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"] and kernel["enableTDMMetadata"]:
        module.addComment1("Update M0 for DTLDS")
        moduleTmp = self.directToLdsM0Update(kernel, 1, tPM)
        module.add(replaceHolder(moduleTmp, 0))
        module.addComment1("Tail global read metadata")
        if tailLoopOpt2nd and (globalReadMode2nd == 2):
          module.add(self.doTailLoopOpt(kernel, tPM))
        else:
          module.add(self.globalReadDo(kernel, globalReadMode2nd, tPM))

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
      if kernel["ProblemType"]["MXBlockA"]:
        doA = False
      if kernel["ProblemType"]["MXBlockB"]:
        doB = False

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
      if kernel["ProblemType"]["MXBlock%s"%tc1]:
        module.add(self.lwaInitAddressesForDTLTailLoop(kernel, tensorParameters1st["MX"]))
      if kernel["ProblemType"]["MXBlock%s"%tc2]:
        module.add(self.lwaInitAddressesForDTLTailLoop(kernel, tensorParameters2nd["MX"]))
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.add(self.lwaInitAddressesForDTLTailLoop(kernel, tPM))

      # the following read/write addresses could be modified in recalcLocal(Read|Write)Addresses due to policy change
      self.oriLraA = None # back up original local read address vgpr
      self.oriLraB = None
      self.oriLraM = None
      self.oriLwaA = None # back up original local write address vgpr
      self.oriLwaB = None
      self.oriLwaM = None
      if not kernel["NoLdsWriteCode"] or kernel["NonDTLTailLoopA"] or kernel["NonDTLTailLoopB"] or kernel["NonDTLTailLoopMetadata"]:
        # TODO: Check correctness for TDM
          # Tail: local write A(MXSA)
          if kernel["ProblemType"]["MacDataTypeA"].is6bitFloat() or kernel["ProblemType"]["MacDataTypeB"].is6bitFloat():
            module.add(self.shiftVgpr6bitFloat(tensorParametersA, tensorParametersB))
          module.addComment1("local write a")
          if not kernel["enableTDMA"]:
            tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA)
            module.add(tempLWCodeModA)
          if "MX" in tensorParametersA:
            module.addComment1("local write mxsa")
            if not kernel["enableTDMA"]:
              tempLWCodeModMXSA = self.localWriteDo(kernel, tensorParametersA["MX"])
              module.add(tempLWCodeModMXSA)
          # Tail: local write B(MXSB)
          if "MX" in tensorParametersB:
            module.addComment1("local write mxsb")
            if not kernel["enableTDMB"]:
              tempLWCodeModMXSB = self.localWriteDo(kernel, tensorParametersB["MX"])
              module.add(tempLWCodeModMXSB)
          module.addComment1("local write b")
          if not kernel["enableTDMB"]:
            tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB)
            module.add(tempLWCodeModB)
      # change local read policy from wider local read to one unit of K at a time
      # DirectToVgpr case, use original wider local read instead of recalculating local read address
      if not (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
        module.addComment1("Recalc local read offsets")
        module.add(self.recalcLocalReadAddressesAB(kernel, tensorParametersA, tensorParametersB))
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "5wait for local write"))
      module.add(self._syncThreads(kernel, "Tail loop LW->LR, sync LDS0", memoryToken=[self.states.memTokenLdsBuffer0]))
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
        # StreamK tail workgroups may start from a partial tile slice, so SIA0
        # also needs the LRO reset before tail MACs.
        if kernel["_ScheduleIterAlg"] != 0 or kernel["StreamK"]:
          module.addComment1("Tail: local read reset offsets a")
          module.add(self.localReadResetOffsets(kernel, tensorParametersA))
          if kernel["ProblemType"]["MXBlockA"]:
            module.addComment1("Tail: local read reset offsets mxsa")
            module.add(self.localReadResetOffsets(kernel, tensorParametersA["MX"]))
          if kernel["ProblemType"]["MXBlockB"]:
            module.addComment1("Tail: local read reset offsets mxsb")
            module.add(self.localReadResetOffsets(kernel, tensorParametersB["MX"]))
          module.addComment1("Tail: local read reset offsets b")
          module.add(self.localReadResetOffsets(kernel, tensorParametersB))

        module.addComment1("Tail: local read init pointers a")
        module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA))
        if kernel["ProblemType"]["MXBlockA"]:
          module.addComment1("Tail: local read init pointers mxsa")
          module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersA["MX"]))
        if kernel["ProblemType"]["MXBlockB"]:
          module.addComment1("Tail: local read init pointers mxsb")
          module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB["MX"]))
        module.addComment1("Tail: local read init pointers b")
        module.add(self.localReadInitPointers(kernel, tensorParametersA, tensorParametersB))

        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          module.addComment1("local read reset offsets metadata")
          module.add(self.localReadResetOffsets(kernel, tPM))
          module.addComment1("local read init pointers metadata")
          module.add(self.localReadInitPointers(kernel, tensorParametersA, tPM))

        if (kernel["enableTDMA"] and kernel["enableTDMB"]
            and self.isPrefetchAcrossPersistentEnabled(kernel)):
          module.add(self.papTdmShiftTailLdsBank(kernel, tensorParametersA, tensorParametersB))

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

      self.states.SubTileIdx = 0 # reset SubTileIdx before TailLoop local read
      for mValue in range(mEnd):
        shiftK = Module()
        if mEnd > 1:
          # print tail loop counter if mEnd>1 (means do tail loop unroll)
          module.addComment1("tail loop unroll iter %u"%(mValue))
        pack[0] = Module()
        for iui in range(0, tailLoopInnerUnroll):
          packCodeA = Module()
          packCodeB = Module()
          packPreA = Module()
          packPreB = Module()
          # local read buffer id. No prefetch in tail loop case.
          bufIdx = mValue % self.states.numVgprBuffer
          # DTV case, use different bufIdx for all loop iterations
          bufIdxDTV = mValue % kernel["LoopIters"]
          bufIdxA = (bufIdxDTV if kernel["DirectToVgprA"] else bufIdx) // self.states.numReadsIterCoalescedA
          if kernel["ProblemType"]["MXBlockA"]:
            bufIdxMXSA = (bufIdxDTV if kernel["DirectToVgprMXSA"] else bufIdx) // self.states.numReadsIterCoalescedMXSA
          if kernel["ProblemType"]["MXBlockB"]:
            bufIdxMXSB = (bufIdxDTV if kernel["DirectToVgprMXSB"] else bufIdx) // self.states.numReadsIterCoalescedMXSB
          bufIdxB = (bufIdxDTV if kernel["DirectToVgprB"] else bufIdx) // self.states.numReadsIterCoalescedB
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedA == 0:
            # Reading 16-bit data from LDS requires packing when ECC enabled
            module.addComment1("local read a")
            localReadCodeA, packCodeA, packPreA = self.localReadDo(kernel, bufIdxA*self.states.numIterPerCoalescedReadA, iui*self.states.numIterPerCoalescedReadA, 0, tensorParametersA)
            module.add(localReadCodeA)
            if not kernel["UseF32XEmulation"]:
              pack[0].add(packCodeA)
          if kernel["ProblemType"]["MXBlockA"]:
            if mValue < mEnd and mValue % self.states.numReadsIterCoalescedMXSA == 0:
              # Reading 16-bit data from LDS requires packing when ECC enabled
              module.addComment1("local read maxa")
              localReadCodeMXSA, packCodeMXSA, packPreMXSA = self.localReadDo(kernel, bufIdxMXSA*self.states.numIterPerCoalescedReadMXSA, iui*self.states.numIterPerCoalescedReadMXSA, 0, tensorParametersA["MX"])
              module.add(localReadCodeMXSA)
              pack[0].add(packPreMXSA)
              pack[0].add(packCodeMXSA)
          if kernel["ProblemType"]["MXBlockB"]:
            if mValue < mEnd and mValue % self.states.numReadsIterCoalescedMXSB == 0:
              # Reading 16-bit data from LDS requires packing when ECC enabled
              module.addComment1("local read maxb")
              localReadCodeMXSB, packCodeMXSB, packPreMXSB = self.localReadDo(kernel, bufIdxMXSB*self.states.numIterPerCoalescedReadMXSB, iui*self.states.numIterPerCoalescedReadMXSB, 0, tensorParametersB["MX"])
              module.add(localReadCodeMXSB)
              pack[0].add(packPreMXSB)
              pack[0].add(packCodeMXSB)
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            if mValue*self.states.numIterPerCoalescedReadMetadata < mEnd:
              module.addComment1("local read metadata")
              localReadCodeM, packCodeM, packPreM = self.localReadDo(kernel, bufIdx*self.states.numIterPerCoalescedReadMetadata, iui*self.states.numReadsIterCoalescedMetadata, 0, tPM)
              module.add(localReadCodeM)
              pack[0].add(packPreM)
              pack[0].add(packCodeM)
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedB == 0:
            module.addComment1("local read b")
            localReadCodeB, packCodeB, packPreB = self.localReadDo(kernel, bufIdxB*self.states.numIterPerCoalescedReadB, iui*self.states.numIterPerCoalescedReadB, 0, tensorParametersB)
            module.add(localReadCodeB)
            if not kernel["UseF32XEmulation"]:
              pack[0].add(packCodeB)
          if kernel["UseF32XEmulation"]:
            # Gather A, B conversion code based on scheduling order
            if len(packPreA.flatitems()) or len(packPreB.flatitems()):
              # pack Pre
              packPreABItems = []
              self._interleavePackAB(kernel, packPreA.flatitems(), packPreB.flatitems(), packPreABItems, prefetch=True)
              pack[0].addItems(packPreABItems)
            # pack
            packABItems = []
            # use interleave only for mfma (cvt_sub/dot2 causes tmp vgpr overlapping)
            if kernel["UseMFMAF32XEmulation"]:
              self._interleavePackAB(kernel, packCodeA.flatitems(), packCodeB.flatitems(), packABItems, prefetch=True)
            else:
              # cvt_sub/dot2 case, add A and B separately (call _interleavePackAB twice for A and B separately)
              emptyModule = Module()
              self._interleavePackAB(kernel, packCodeA.flatitems(), emptyModule.flatitems(), packABItems, prefetch=True)
              self._interleavePackAB(kernel, emptyModule.flatitems(), packCodeB.flatitems(), packABItems, prefetch=True)
            shiftK.addItems(packABItems)

          # adjustment for DirectToLds case
          iuiParam = iui + tailLoopInnerUnroll * mValue//self.states.numReadsIterCoalescedA
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedA == 0:
            module.addComment1("local read inc a")
            module.add(self.localReadInc(kernel, iuiParam, tensorParametersA))
          if kernel["ProblemType"]["MXBlockA"]:
            iuiParam = iui + tailLoopInnerUnroll * mValue//self.states.numReadsIterCoalescedMXSA
            if mValue < mEnd and mValue % self.states.numReadsIterCoalescedMXSA == 0:
              module.addComment1("local read inc mxsa")
              module.add(self.localReadInc(kernel, iuiParam, tensorParametersA["MX"]))
          if kernel["ProblemType"]["MXBlockB"]:
            iuiParam = iui + tailLoopInnerUnroll * mValue//self.states.numReadsIterCoalescedMXSB
            if mValue < mEnd and mValue % self.states.numReadsIterCoalescedMXSB == 0:
              module.addComment1("local read inc mxsb")
              module.add(self.localReadInc(kernel, iuiParam, tensorParametersB["MX"]))
          iuiParam = iui + tailLoopInnerUnroll * mValue
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            module.addComment1("local read inc metadata")
            module.add(self.localReadInc(kernel, iuiParam, tPM))
          iuiParam = iui + tailLoopInnerUnroll * mValue//self.states.numReadsIterCoalescedB
          if mValue < mEnd and mValue % self.states.numReadsIterCoalescedB == 0:
            module.addComment1("local read inc b")
            module.add(self.localReadInc(kernel, iuiParam, tensorParametersB))
        module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "4wait for local read"))
        self.states.SubTileIdx = (self.states.SubTileIdx + 1) % kernel["numSubTiles"]

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
    elif self.states.tailloopInNll:
      # always emit the skip-tail-loop label for NoTailLoop + tailloopInNll
      module.add(self.closeLoop(kernel, tensorParametersA, tensorParametersB, -1, None, emitEndLabelOnly=True))

    if self.states.tailloopInNll:
      # generate wait code for early exit
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, " tailloopInNll: wait for global read"))
      module.add(self._wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, " tailloopInNll: wait for local read"))
      module.add(self._syncThreads(kernel, "tailloopInNll: wait until LR done, sync LDS0", memoryToken=[self.states.memTokenLdsBuffer0]))

    if self.states.lastValuMXSAB:
      self.vgprPool.add(0 , self.states.lastValuMXSAB, "ValuMXSAB")
      module.addComment1("Tail: add ValuA/B vgpr buffer [%u...%u) to pool" % \
          (0, self.states.lastValuMXSAB))

    if self.do["executeToLoopEnd"]:
      module.add(self.functionEnd(kernel, addLabel=False))

    # extra summation loops: global increment and close
    for i in reversed(range(self.states.otherSummationLoops)):
      module.addComment1("global read inc AB")
      module.add(self.globalReadIncrementAB(kernel, tensorParametersA, tensorParametersB, i, 0))
      # swap Tensor memToken
      self.states.ldsTensorTokenIdx = \
          self.states.memTokenLdsBuffer1 if self.states.ldsTensorTokenIdx == self.states.memTokenLdsBuffer0 else self.states.memTokenLdsBuffer0
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

      # shift vector components d1
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
      storeModule, _ = self.notLocalSplitUGlobalWrite(kernel, tensorParametersA, tensorParametersB)
      module.add(storeModule)

    # Emit any deferred activation modules (set during globalWriteElements)
    if hasattr(self.states, 'deferredActivationModules') and self.states.deferredActivationModules is not None:
      module.add(SBranch(labelName="KernelEnd", comment="skip over activation functions"))
      module.appendModule(self.states.deferredActivationModules)
      self.states.deferredActivationModules = None

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

    # Post-rocIsaPass: rescan actual register usage and update kernel descriptor
    # + CUOccupancy for ArchAccUnifiedRegs ISAs where removeDuplicateAssignment
    # may have reduced the instruction-level VGPR count below the pool estimate.
    self.updateOccupancyFromScan(kernel, moduleKernelBody)

    # Initialize stModule as None (will be set for supported architectures)
    stModule = None

    # Run StinkyTofu conversion for supported architectures
    t0_start = time.perf_counter()
    stinky_opt_level = kernel.get("_StinkyTofuOptLevel")
    if stinky_opt_level is not None and rocisa.isSupportedByStinkyTofu(self.states.version):
      print2(f"StinkyTofu: Converting kernel to stinkytofu IR for gfx{self.states.version[0]}{self.states.version[1]}{self.states.version[2]}...")

      moduleKernelBody.body.setParent()

      # Set StinkyTofu module options
      stinky_module_options = {"OptLevel": stinky_opt_level,
                               "EnableRemarks": bool(globalParameters.get("StinkyTofuEnableRemarks") or False),
                               "DebugLevel": int(globalParameters.get("StinkyTofuDebugLevel") or 0),
                               "PrintBeforePass": str(globalParameters.get("StinkyTofuPrintBeforePass") or ""),
                               "PrintAfterPass": str(globalParameters.get("StinkyTofuPrintAfterPass") or ""),
                               "DebugPass": str(globalParameters.get("StinkyTofuDebugPass") or ""),
                               "PassOrderSnapshotJson": str(globalParameters.get("StinkyTofuPassOrderSnapshotJson") or ""),
                               "EnableWaitCntInsertion": True if stinky_opt_level != 0 else not globalParameters.get("DisableSTWaitCnt", True),
                               # True: expert scheduling mode2; False: mode 0. Independent of ScheduleIterAlg/OptLevel.
                               "EnableESM2": kernel["EnableStinkyTofuESM2"],
                               "TileA0": kernel["ThreadTile0"],
                               "TileB0": kernel["ThreadTile1"],
                               "TileM0": kernel["MacroTile0"],
                               "wavefrontSize": kernel["WavefrontSize"],
                               "SubGroup0": kernel["SubGroup0"],
                               "SubGroup1": kernel["SubGroup1"],
                               "WaveGroup0": kernel["MIWaveGroup"][0],
                               "WaveGroup1": kernel["MIWaveGroup"][1],
                               "VectorWidthA": kernel["VectorWidthA"],
                               "VectorWidthB": kernel["VectorWidthB"],
                               "GlobalReadVectorWidthA": kernel["GlobalReadVectorWidthA"],
                               "GlobalReadVectorWidthB": kernel["GlobalReadVectorWidthB"],
                               "DirectToLdsA": bool(kernel["DirectToLdsA"]),
                               "DirectToLdsB": bool(kernel["DirectToLdsB"]),
                               "UseSgprForGRO": kernel["_UseSgprForGRO"],
                               # -1 disables SwInstructionPrefetch in Gfx1250Backend; else scratch pool index
                               "SwPrefetchScratchSgpr": int(self.sgprs.get("SwPrefetchScratch", -1)),
                               # Cluster-barrier handshake insertion in Gfx1250Backend
                               # (kernel-scope at every OptLevel when set).
                               "ClusterBarrier": bool(kernel.get("ClusterBarrier", False)),
                               # PrefetchGlobalRead (PGR) passed to InsertClusterBarrierPass.
                               # Gates Rule 3 (`LCL <= PGR` skip) and Rule 4 (`LCL == PGR+1`
                               # skip in fresh-gate mode; inherits upstream `LCL == PGR` cmp
                               # in inherited-SCC mode). Rule 1 uses `LCL == 0`, not PGR.
                               # Defaults to 1 when unset.
                               "PrefetchGlobalRead": int(kernel.get("PrefetchGlobalRead", 1)),
                               # PrefetchLocalRead (PLR) passed to InsertClusterBarrierPass.
                               # When PLR == 0, enables Rule 3 anchor mode (b): if no
                               # `s_barrier_wait -1` precedes `label_openLoopL:`, synthesize
                               # a workgroup sync inside the LCL-gated signal block.
                               # Defaults to 1 (fallback off) when unset.
                               "PrefetchLocalRead": int(kernel.get("PrefetchLocalRead", 1)),
                              }

      print2(f"StinkyTofu module options: {stinky_module_options}")
      # Convert rocisa module to stinkytofu with signature
      # Returns a KernelBody wrapper that includes signature and instruction module
      # - runOptimizationPipeline() optimizes the instruction body
      # - emitAssembly() outputs complete kernel: signature + optimized instructions
      t1a_start = time.perf_counter()
      stModule = rocisa.toStinkyTofuModule(moduleKernelBody.body, self.states.version, "kernel_name",
                                           signature=fs,
                                           options=stinky_module_options)
      t1a_end = time.perf_counter()
      print2(f"StinkyTofu (1a) toStinkyTofuModule: {t1a_end - t1a_start:.4f}s")

      # Run pipeline — builder handles O0 internally (skips optimization,
      # still runs required passes like InsertVgprMsb)
      t1b_start = time.perf_counter()
      stModule.runOptimizationPipeline()
      t1b_end = time.perf_counter()
      print2(f"StinkyTofu (1b) pipeline: {t1b_end - t1b_start:.4f}s")

    error = self.states.overflowedResources
    print2(f"  found error code {error} with overflowed resources set to {self.states.overflowedResources}")

    # Check if StinkyTofu assembly output should be used
    if stModule is not None:
      t2_start = time.perf_counter()
      st_asm = stModule.emitAssembly()
      t2_end = time.perf_counter()
      print2(f"StinkyTofu (2) emitAssembly: {t2_end - t2_start:.4f}s")

      if os.environ.get("ENABLE_DEBUG_STINKYTOFU_ASM") is not None:
        t3_start = time.perf_counter()
        import hashlib, fcntl
        os.makedirs("cmpasm/orig", exist_ok=True)
        os.makedirs("cmpasm/st", exist_ok=True)

        kernel_name = self.states.kernelName
        name_hash = hashlib.sha256(kernel_name.encode()).hexdigest()[:8]
        base_name = f"{name_hash}"

        orig_path = f"cmpasm/orig/{base_name}.s"
        st_path = f"cmpasm/st/{base_name}.s"
        suffix = 0
        while os.path.exists(orig_path) or os.path.exists(st_path):
          orig_path = f"cmpasm/orig/{base_name}_{suffix}.s"
          st_path = f"cmpasm/st/{base_name}_{suffix}.s"
          suffix += 1

        t_str_start = time.perf_counter()
        orig_asm_str = str(moduleKernelBody)
        t_str_end = time.perf_counter()
        print2(f"Rocisa::str(moduleKernelBody): {t_str_end - t_str_start:.4f}s")

        with open(orig_path, "w") as f:
          f.write(orig_asm_str)
        with open(st_path, "w") as f:
          f.write(st_asm)

        manifest = f"cmpasm/manifest.txt"
        with open(manifest, "a") as f:
          fcntl.flock(f, fcntl.LOCK_EX)
          f.write(f"{os.path.basename(orig_path)} -> {kernel_name}\n")
          fcntl.flock(f, fcntl.LOCK_UN)

        t3_end = time.perf_counter()
        print2(f"StinkyTofu (3) debug file write: {t3_end - t3_start:.4f}s")

      t0_end = time.perf_counter()
      print2(f"StinkyTofu (0) total: {t0_end - t0_start:.4f}s")

      print2(f"Using StinkyTofu Assembly")
      return (error, st_asm)
    else:
      print2(f"Using Original Rocisa Assembly")
      return (error, str(moduleKernelBody))

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def _initKernel(self, kernel, tensorParametersA, tensorParametersB):
    assert kernel["KernelLanguage"] == "Assembly"
    self.language   = "ASM"
    # ISA version, such as 803
    version = tuple(kernel["ISA"])
    isgfx950 = kernel["ISA"][:2] == (9, 5)
    ti = rocIsa.getInstance()
    ti.setKernel(version, kernel["WavefrontSize"])

    self.consts = ConstValues()
    self.states = StateValues(version=version, kernel=kernel, kernelName=getKernelNameMin(kernel, self.debugConfig.splitGSU))
    # LDS swizzling for subtile bank-conflict avoidance (gfx950 only;
    # gfx1250 TDM uses a different LDS layout without swizzling).
    self.states.subtileLdsSwizzle = kernel.get("UseSubtileImpl", False) and isgfx950
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

    def selectGRSubtileShape(tc):
      """Select GR subtile shape based on wave cooperation level.

      The GR tile shape represents the effective coverage per GR load round.
      When multiple waves cooperate on a single GR load (waves_coop >= 4),
      the GR tile expands from (1,2) to (2,2) MMA tiles.

      For A: waves_coop = numWaves / MIWaveGroup[0]
      For B: waves_coop = numWaves / MIWaveGroup[1]

      Wave config → GR shape:
        1x4 WG: A=(2,2), B=(1,2)   — A has 4 cooperating waves
        4x1 WG: A=(1,2), B=(2,2)   — B has 4 cooperating waves
        2x2 WG: A=(1,2), B=(1,2)   — 2 cooperating waves each
        1x1 WG: A=(1,2), B=(1,2)   — 1 wave each

      LR subtile shape is always (1,2) regardless of wave config.
      """
      mi = kernel["MIWaveGroup"]
      numWaves = mi[0] * mi[1]
      wg_idx = 0 if tc == 'A' else 1
      wg_m = mi[wg_idx]
      waves_coop = numWaves // wg_m
      return (2, 2) if waves_coop >= 4 else (1, 2)

    def initSubTileInfo(tc):
      tileMap = {
        'A' : self.states.a,
        'B' : self.states.b,
        'D' : self.states.d,
        'MXSA' : self.states.mxsa,
        'MXSB' : self.states.mxsb,
      }
      matrixInfo = tileMap[tc]
      if tc == 'D':
        geometry = selectDGeometry(kernel)
        matrixInfo.tileInfo = TileInfo(geometry, tc, self, kernel)
      elif tc in ('A', 'B'):
        grShape = selectGRSubtileShape(tc)
        matrixInfo.grSubtileShape = grShape
        geometry = selectABGeometry(kernel, tc)
        matrixInfo.tileInfo = TileInfo(geometry, tc, self, kernel)
      elif tc in ('MXSA', 'MXSB'):
        geometry = selectMXScaleGeometry(kernel, tc)
        matrixInfo.tileInfo = TileInfo(geometry, tc, self, kernel)


    if kernel["UseSubtileImpl"]:
      initSubTileInfo('A')
      initSubTileInfo('B')
      initSubTileInfo('D')

      if kernel["ProblemType"].get("MXBlockA", 0) > 0:
        initSubTileInfo('MXSA')
      if kernel["ProblemType"].get("MXBlockB", 0) > 0:
        initSubTileInfo('MXSB')

      self.ldsStartOffsetA = 0
      aTileInfo = self.states.a.tileInfo
      bTileInfo = self.states.b.tileInfo
      numASubtiles = int(aTileInfo.globalSubtileGrid[0] * aTileInfo.globalSubtileGrid[1])
      numBSubtiles = int(bTileInfo.globalSubtileGrid[0] * bTileInfo.globalSubtileGrid[1])
      readSize = 2*aTileInfo.subtileSize
      # Align A and B sizes to readSize for DTL 2xsubtile reads
      sizeA = int(((numASubtiles * aTileInfo.subtileSize + readSize-1) // readSize) * readSize)
      sizeB = int(((numBSubtiles * bTileInfo.subtileSize + readSize-1) // readSize) * readSize)
      self.ldsStartOffsetB = sizeA
      sizeMXSA = 0
      sizeMXSB = 0
      if kernel["ProblemType"].get("MXBlockA", 0) > 0 and kernel["ProblemType"].get("MXBlockB", 0) > 0:
        mxsaTileInfo = self.states.mxsa.tileInfo
        mxsbTileInfo = self.states.mxsb.tileInfo

        # For Swizzled scale we use extra LDS space for now to allow wider DTL loads
        numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
        sizeMXSA = mxsaTileInfo.loadWidthGR * kernel["WavefrontSize"] * numWaves
        sizeMXSB = mxsbTileInfo.loadWidthGR * kernel["WavefrontSize"] * numWaves
        self.ldsStartOffsetMXSA = sizeA + sizeB
        self.ldsStartOffsetMXSB = sizeA + sizeB + sizeMXSA

      self.ldsTotalSize = sizeA + sizeB + sizeMXSA + sizeMXSB

      kernel["LdsNumBytes"] = max(1, int(self.ldsTotalSize * kernel["NumLdsBlk"]))
      if kernel["LdsNumBytes"] > self.states.archCaps["DeviceLDS"]:
        self.states.overflowedResources = 8


    #print(self.states.a.tileInfo.getLocalSubtileId(1,0))

    #exit(1)

    self.states.tailloopInNll = kernel["TailloopInNll"]
    hasMx = kernel["ProblemType"]["MXBlockA"] or kernel["ProblemType"]["MXBlockB"]
    usesTDM = kernel["enableTDMA"] or kernel["enableTDMB"]
    usesDTL = kernel["DirectToLdsA"] or kernel["DirectToLdsB"]
    # MX PAP on the canonical buffer-load path cannot afford the borrowed
    # stagger-state checkpoint. Current MX PAP configs use StaggerU=0 here.
    disableStaggerForMxPap = kernel["PrefetchAcrossPersistent"] and hasMx and not (usesTDM or usesDTL)
    # remove staggerU code for the following cases
    # - tailloopInNll (cannot support staggerU)
    # - StreamK + MX (not enough sgpr. gfx950 only for now)
    # - MX PAP without TDM/DTL (not enough sgpr)
    self.states.staggerUCode = True
    if self.states.tailloopInNll or \
       (kernel["StreamK"] and \
        hasMx and isgfx950) or \
       disableStaggerForMxPap or \
       kernel["UseSubtileImpl"]:
      self.states.staggerUCode = False
    self.states.tailloopInNllmaxUnit = 1
    if self.states.tailloopInNll:
      tluA = kernel["ProblemType"]["TLUA"]
      tluB = kernel["ProblemType"]["TLUB"]
      bpeGRA = kernel["ProblemType"]["DataTypeA"].numBytes()
      bpeGRB = kernel["ProblemType"]["DataTypeB"].numBytes()
      grvwa = kernel["GlobalReadVectorWidthA"]
      grvwb = kernel["GlobalReadVectorWidthB"]
      asem = kernel["AssertSummationElementMultiple"]
      unitA = 1
      unitB = 1
      if ((not tluA) and (bpeGRA * asem < 4) and grvwa > 1):
        unitA = int(4 / (bpeGRA * asem))
      if ((not tluB) and (bpeGRB * asem < 4) and grvwb > 1):
        unitB = int(4 / (bpeGRB * asem))
      self.states.tailloopInNllmaxUnit = max(unitA, unitB)

    # Only assembly supports scheduling
    if kernel["KernelLanguage"] == "Assembly":
      self.states.scheduleGlobalRead = kernel["ScheduleGlobalRead"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"] # flat updates lgkmcnt counts = hard to schedule flat loads
      self.states.scheduleLocalWrite = kernel["ScheduleLocalWrite"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"]  # flat updates lgkmcnt counts = hard to schedule writes and loads?
      self.states.scheduleIterAlg = kernel["_ScheduleIterAlg"]
    else:
      self.states.scheduleGlobalRead = 0
      self.states.scheduleLocalWrite = 0
      self.states.scheduleIterAlg = 0

    self.states.actualSummationLoops = kernel["ProblemType"]["NumIndicesSummation"]
    self.states.otherSummationLoops  = self.states.actualSummationLoops-1
    self.states.otherSummations      = kernel["ProblemType"]["NumIndicesSummation"]-1 # not loops but summations vars

    # enable scheduling GR (in LWcode for PGR2) over barrier sync
    self.states.scheduleGROverBarrier = kernel["ScheduleGROverBarrier"]

    # doShadowInit performs initialization in the 'shadow' of the global mem prefetch
    # TODO re-enable..
    if not kernel["ForceDisableShadowInit"] and not kernel["UseSubtileImpl"]:
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

    if kernel["EnableMatrixInstruction"] and kernel["LocalReadVectorWidthA"] >= kernel["MIInputPerThread"]:
      WLR = int(max(kernel["LocalReadVectorWidthA"]//kernel["MIInputPerThread"], 1))
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
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.numVgprBufferPackMXSA = kernel["LoopIters"]
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.numVgprBufferPackMXSB = kernel["LoopIters"]
    else:
      self.states.numVgprBufferPackA = self.states.numItersPLR + 1
      self.states.numVgprBufferPackB = self.states.numItersPLR + 1
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.numVgprBufferPackMXSA = self.states.numItersPLR + 1
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.numVgprBufferPackMXSB = self.states.numItersPLR + 1

    if kernel["enableLDSTrA"]:
      self.states.numVgprBufferPackA = 0

    if kernel["enableLDSTrB"]:
      self.states.numVgprBufferPackB = 0

    # TODO load sub-vector
    vwa = kernel["GlobalReadVectorWidthA"]
    vwb = kernel["GlobalReadVectorWidthB"]
    if kernel["ProblemType"]["MXBlockA"]:
      vwmxsa = kernel["GlobalReadVectorWidthMXSA"]
    if kernel["ProblemType"]["MXBlockB"]:
      vwmxsb = kernel["GlobalReadVectorWidthMXSB"]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      vwm = kernel["GlobalReadVectorWidthMetadata"]

    # force lrvwTile = 1 for numBytes >= 4 + MIInputPerThread > 1
    # Exempt from forcing lrvwTile=1:
    #   - MFMA-based XF32 emulation (gfx950): pack code already handles lrvwTile > 1
    #   - CMS kernels: schedules are designed with lrvwTile > 1 and manage pack-code placement explicitly
    # Non-MFMA XF32 (e.g. gfx1250 WMMA) without CMS must use lrvwTile=1 because
    # the default scheduler's local reads don't match the XF32 pack code's expectations.
    # TODO: implement extra logic to swap vgprs after local read to suport lrvwTile > 1 for numBytes >= 4 + MIInputPerThread > 1
    isCMS = kernel["UseCustomMainLoopSchedule"]
    isMfmaXf32 = kernel["UseMFMAF32XEmulation"]
    forceLrvwTile1A = kernel["ProblemType"]["MacDataTypeA"].numBytes() >= 4 and \
      (kernel["EnableMatrixInstruction"] and kernel["MIInputPerThread"] > 1) and \
      not (kernel["UseF32XEmulation"] and (isMfmaXf32 or isCMS))
    if not kernel["UnrollMajorLDSA"] and not forceLrvwTile1A:
      self.states.lrvwTileA = kernel["VectorWidthA"]
    else:
      self.states.lrvwTileA = 1
    forceLrvwTile1B = kernel["ProblemType"]["MacDataTypeB"].numBytes() >= 4 and \
      (kernel["EnableMatrixInstruction"] and kernel["MIInputPerThreadB"] > 1) and \
      not (kernel["UseF32XEmulation"] and (isMfmaXf32 or isCMS))
    if not kernel["UnrollMajorLDSB"] and not forceLrvwTile1B:
      self.states.lrvwTileB = kernel["VectorWidthB"]
    else:
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.lrvwTileMXSB = 1
      self.states.lrvwTileB = 1

    if kernel["ProblemType"]["MXBlockA"]:
      if ((self.states.asmCaps["HasWMMA_V3"] and kernel["MXScaleFormat"] == "InMemorySwizzle")
          or (not kernel["UnrollMajorLDSMXSA"] and not forceLrvwTile1A)):
        self.states.lrvwTileMXSA = kernel["VectorWidthA"]
      else:
        self.states.lrvwTileMXSA = 1

    if kernel["ProblemType"]["MXBlockB"]:
      if ((self.states.asmCaps["HasWMMA_V3"] and kernel["MXScaleFormat"] == "InMemorySwizzle")
          or (not kernel["UnrollMajorLDSMXSB"] and not forceLrvwTile1B)):
        self.states.lrvwTileMXSB = kernel["VectorWidthB"]
      else:
        self.states.lrvwTileMXSB = 1

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

    if self.states.lrvwTileA > 1 and (kernel["ProblemType"]["MacDataTypeA"].isHalf() or kernel["ProblemType"]["MacDataTypeA"].isBFloat16() or \
      kernel["ProblemType"]["MacDataTypeA"].isInt8() or kernel["ProblemType"]["MacDataTypeA"].is8bitFloat()):
      self.states.numVgprBufferPackA = 1
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.numVgprBufferPackMXSA = 1

    if self.states.lrvwTileB > 1 and (kernel["ProblemType"]["MacDataTypeB"].isHalf() or kernel["ProblemType"]["MacDataTypeB"].isBFloat16() or \
      kernel["ProblemType"]["MacDataTypeB"].isInt8() or kernel["ProblemType"]["MacDataTypeB"].is8bitFloat()):
      self.states.numVgprBufferPackB = 1
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.numVgprBufferPackMXSB = 1

    if kernel["UnrollMajorLDSA"]:
      self.states.lrvwUnrollA = kernel["LocalReadVectorWidthA"]
    else:
      self.states.lrvwUnrollA = 1
    if kernel["ProblemType"]["MXBlockA"]:
      self.states.lrvwUnrollMXSA = 1

    if kernel["UnrollMajorLDSB"]:
      self.states.lrvwUnrollB = kernel["LocalReadVectorWidthB"]
    else:
      self.states.lrvwUnrollB = 1
    if kernel["ProblemType"]["MXBlockB"]:
      self.states.lrvwUnrollMXSB = 1

    if kernel["ProblemType"]["MXBlockA"]:
      if kernel["UnrollMajorLDSMXSA"]:
        self.states.lrvwUnrollMXSA = kernel["LocalReadVectorWidthMXS"]
      else:
        self.states.lrvwUnrollMXSA = 1

    if kernel["ProblemType"]["MXBlockB"]:
      if kernel["UnrollMajorLDSMXSB"]:
        self.states.lrvwUnrollMXSB = kernel["LocalReadVectorWidthMXS"]
      else:
        self.states.lrvwUnrollMXSB = 1

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
    self.states.numReadsIterCoalescedMXSA  = 1
    self.states.numIterPerCoalescedReadMXSA = max(1, self.states.numReadsIterCoalescedMXSA // kernel["InnerUnroll"])
    self.states.numReadsIterCoalescedMXSB  = 1
    self.states.numIterPerCoalescedReadMXSB = max(1, self.states.numReadsIterCoalescedMXSB // kernel["InnerUnroll"])

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      if kernel["EnableMatrixInstruction"]:
        self.states.numReadsIterCoalescedMetadata = ceil(self.states.lrvwUnrollMetadata / kernel["MIInputPerThreadMetadata"])
      else:
        self.states.numReadsIterCoalescedMetadata = 1
    else:
      self.states.numReadsIterCoalescedMetadata  = 1
    self.states.numIterPerCoalescedReadMetadata = max(1,self.states.numReadsIterCoalescedMetadata // kernel["InnerUnroll"])

    if kernel["_ScheduleIterAlg"] == 3 or kernel["_ScheduleIterAlg"] == 2:
      self.states.numMfmaPerIter = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1] * kernel["InnerUnroll"]
      if kernel["ProblemType"]["DataType"].isComplex(): self.states.numMfmaPerIter *= 4
      elif kernel["UseF32XEmulation"]: self.states.numMfmaPerIter *= 3

    if kernel["ForceUnrollSubIter"]:
      kernel["LoopIters"] = kernel["numSubTiles"] * kernel["numSubTiles"]
      self.states.numMfmaPerIter = self.states.numMfmaPerIter//kernel["LoopIters"]
      self.states.numItersPLR = 1

    # set number of LDS Block
    self.states.numLDSBlk = kernel["NumLdsBlk"]
    # use inc to switch Lds Buffers (instead of using xor)
    self.states.IncLdsBufSwitch = kernel["NumLdsBlk"] >= 3
    # oneBufferScheduling
    self.states.oneBufferScheduling = (kernel["1LDSBuffer"]) or \
                                      ((kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and \
                                       self.states.numLDSBlk == kernel["PrefetchGlobalRead"])
    # common sgprSwap
    # enable it for the following case.
    #  DTLA+B + numLDSBlk==2 + (not EPS) + (not sparse) + (MX or (StoreSwapAddr and (not CMS))
    self.states.useCommonSgprSwap = False
    if kernel["DirectToLds"] == 1 and self.states.numLDSBlk == 2 and \
       (not kernel["ExpandPointerSwap"]) and (not kernel["ProblemType"]["Sparse"]) and \
       ((kernel["ProblemType"]["MXBlockA"] or kernel["ProblemType"]["MXBlockB"]) or \
         kernel["StoreSwapAddr"] and not kernel["UseCustomMainLoopSchedule"]):
      self.states.useCommonSgprSwap = True
    # set memory token by LDS buffer setting
    self.states.memTokenLdsBufferMeta = 4
    if kernel["1LDSBuffer"]:
      self.states.memTokenLdsBuffer0 = 0
      self.states.memTokenLdsBuffer1 = 0
    else:
      self.states.memTokenLdsBuffer0 = 0
      self.states.memTokenLdsBuffer1 = 1
    self.states.ldsReadTokenIdx = self.states.memTokenLdsBuffer0
    self.states.ldsTensorTokenIdx = self.states.memTokenLdsBuffer0
    self.states.ldsDirectToLDSTokenIdx = self.states.memTokenLdsBuffer0
    self.states.ldsWriteTokenIdx = self.states.memTokenLdsBuffer0
    self.states.lockLdsReadTokenSwap = False

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

    tensorParametersMXSA = None
    if kernel["ProblemType"]["MXBlockA"]:
      itP["MXSA"] = readWriteVectors("MXSA", vwmxsa, kernel)
      tensorParametersMXSA = {}
      self.getTensorParameters(tensorParametersMXSA, kernel, itP, "MXSA")
      tensorParametersMXSA["PackedIndices"] = kernel["PackedC%uIndicesX"%tensorParametersMXSA["tile01Idx"]]
      tensorParametersA["MX"] = tensorParametersMXSA

    tensorParametersMXSB = None
    if kernel["ProblemType"]["MXBlockB"]:
      itP["MXSB"] = readWriteVectors("MXSB", vwmxsb, kernel)
      tensorParametersMXSB = {}
      self.getTensorParameters(tensorParametersMXSB, kernel, itP, "MXSB")
      tensorParametersMXSB["PackedIndices"] = kernel["PackedC%uIndicesX"%tensorParametersMXSB["tile01Idx"]]
      tensorParametersB["MX"] = tensorParametersMXSB

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
    # Keep gfx950's dedicated MX shadow-limit switch, but match legacy gfx1250
    # behavior where MX tensors share the same shadow-limit mode as A/B.
    if isgfx950:
      self.states.use64bShadowLimitMX = kernel["Use64bShadowLimitMX"] and kernel["BufferLoad"]
    else:
      self.states.use64bShadowLimitMX = self.states.use64bShadowLimit

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
    self.states.srdShiftLeft["A"] = kernel["GlobalReadVectorWidthA"] if not kernel["UseSubtileImpl"] else 0
    self.states.srdShiftLeft["B"] = kernel["GlobalReadVectorWidthB"] if not kernel["UseSubtileImpl"] else 0
    if kernel["ProblemType"]["MXBlockA"]:
      # use MXS version for gfx950 only
      self.states.srdShiftLeft["MXSA"] = kernel["GlobalReadVectorWidthMXSA"] if isgfx950 and not kernel["UseSubtileImpl"] else kernel["GlobalReadVectorWidthA"]
    if kernel["ProblemType"]["MXBlockB"]:
      # use MXS version for gfx950 only
      self.states.srdShiftLeft["MXSB"] = kernel["GlobalReadVectorWidthMXSB"] if isgfx950 and not kernel["UseSubtileImpl"] else kernel["GlobalReadVectorWidthB"]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      self.states.srdShiftLeft["Metadata"] = kernel["GlobalReadVectorWidthMetadata"]

    # checkGRO requires useSgprForGRO=0 so that code allocates and uses
    # the VGPRs that are used for the GRO offset checking
    assert not (kernel["_UseSgprForGRO"] and self.states.checkGRO)

    self.states.doubleVgpr = False
    if self.states.archCaps["ArchAccUnifiedRegs"] or (kernel["WavefrontSize"] == 32):
      self.states.doubleVgpr = True

    if kernel["EnableMatrixInstruction"]:
      if (kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64') and not (self.states.asmCaps["HasMFMA_f64"] or self.states.asmCaps["HasWMMA_V3_f64"]):
        raise RuntimeError("FP64 MatrixInstruction not supported for {0}".format(self.states.version))
      elif not ( self.states.asmCaps["HasMFMA"] or self.states.asmCaps["HasWMMA"]):
        raise RuntimeError("MatrixInstruction not supported for {0}".format(self.states.version))

      if kernel["MFMA_BF16_1K"] and not self.states.asmCaps["HasMFMA_bf16_1k"]:
        raise RuntimeError("BF16_1k MatrixInstruction not supported for {0}".format(self.states.version))

      # Modification: Add "HasSWMMAC" to asmCaps for wave-32. Separatlly check for SMFMA and SWMMAC.
      if kernel["ProblemType"]["Sparse"]:
        if kernel["WavefrontSize"] == 32 and not self.states.asmCaps["HasSWMMAC"]:
          raise RuntimeError("Sparse MatrixInstruction SWMMAC not supported for {0}".format(self.states.version))
        if kernel["WavefrontSize"] == 64 and not self.states.asmCaps["HasSMFMA"]:
          raise RuntimeError("Sparse MatrixInstruction SMFMA not supported for {0}".format(self.states.version))

      if (kernel["EnableF32XdlMathOp"] and kernel["ProblemType"]["F32XdlMathOp"].isXFloat32() and (not self.states.asmCaps["HasMFMA_xf32"])):
        if not kernel["UseF32XEmulation"]:
          raise RuntimeError("XF32 MatrixInstruction not supported for {0}".format(self.states.version))

    if not self.states.asmCaps["HasDirectToLds"]:
      kernel["DirectToLdsA"] = False
      kernel["DirectToLdsB"] = False
      kernel["LocalWriteUseSgprA"] = False # Requires DirectToLdsA
      kernel["LocalWriteUseSgprB"] = False # Requires DirectToLdsB

    if kernel["ProblemType"]["Sparse"] and not (kernel["DirectToVgprSparseMetadata"] or kernel["DirectToLdsMetadata"]):
      kernel["LocalWriteUseSgprMetadata"] = False

    # The inst HasAtomicAdd is using is not compatible with int32.
    self.states.useAtomicAdd = (self.states.asmCaps["HasAtomicAdd"] and kernel["ProblemType"]["ComputeDataType"].isSingle()) and \
                        (kernel["_GlobalAccumulation"] == 'SingleBuffer')

    if self.states.asmCaps["v_fma_mix_f32"]:
      self.states.mixinst = VFmaMixF32
    elif self.states.asmCaps["v_mad_mix_f32"]:
      self.states.mixinst = VMadMixF32


    self.states.bpeA = self.states.bpr * kernel["ProblemType"]["MacDataTypeA"].numRegisters()
    self.states.bpeB = self.states.bpr * kernel["ProblemType"]["MacDataTypeB"].numRegisters()
    self.states.bpeE = int(self.states.bpr * kernel["ProblemType"]["DataTypeE"].numRegisters())
    self.states.bpeCinternal = int(self.states.bpr * kernel["ProblemType"]["ComputeDataType"].numRegisters())

    self.states.bpeCexternalGSU1 = int(self.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())
    self.states.bpeCexternal = self.states.bpeCexternalGSU1
    if kernel["GlobalSplitU"] > 0 and kernel["_GlobalAccumulation"] and kernel["_GlobalAccumulation"] != 'PartialsBuffer':
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
          kernel["ProblemType"]["DataType"].is8bitFloat() or kernel["ProblemType"]["DataType"].is6bitFloat() or \
          kernel["ProblemType"]["DataType"].isFloat4()):
        print("HighPrecisionAccumulate only valid when DataType is half, bf16, Int8x4, Int8, fp8, bf8, fp6, bf6, fp4. Forcing HPA to False")
        kernel["ProblemType"]["HighPrecisionAccumulate"] = False

    assert self.states.bpeA == tensorParametersA["bpe"]
    assert self.states.bpeB == tensorParametersB["bpe"]

    #######################################L
    # Available Memory Instructions
    ########################################

    # name, numAddresses, numOffsets, offsetMultiplier, blockWidth, formatting):
    ########################################
    # Local Read
    _ds_load_b192 = MemoryInstruction(DSLoadB192,  1, 1, 6, 6)
    _ds_load_b128 = MemoryInstruction(DSLoadB128,  1, 1, 4, 4)
    _ds_load2_b64 = MemoryInstruction(DSLoad2B64,  1, 2, 2, 2)
    _ds_load_b64 = MemoryInstruction(DSLoadB64,    1, 1, 2, 2)
    _ds_load2_b32 = MemoryInstruction(DSLoad2B32,  1, 2, 1, 1)
    _ds_load_b32 = MemoryInstruction(DSLoadB32,    1, 1, 1, 1)
    _ds_load_u16 = MemoryInstruction(DSLoadU16,    1, 1, 1, 0.5)
    _ds_load_u8 = MemoryInstruction(DSLoadU8,      1, 1, 1, 0.25)
    _ds_load_b64_tr_b16 = MemoryInstruction(DSLoadB64TrB16,    1, 1, 2, 2, bpe=2)
    _ds_load_b128_tr_b16 = MemoryInstruction(DSLoadB128TrB16,    1, 1, 4, 4, bpe=2)
    _ds_load_b64_tr_b8 = MemoryInstruction(DSLoadB64TrB8,    1, 1, 2, 2, bpe=1)
    _ds_load_b64_tr_b4 = MemoryInstruction(DSLoadB64TrB4,    1, 1, 2, 2, bpe=0.5)
    _ds_load_b96_tr_b6 = MemoryInstruction(DSLoadB96TrB6,    1, 1, 3, 3, bpe=0.75)

    ########################################
    # Local Write
    _ds_store_b256 = MemoryInstruction(DSStoreB256,  1, 1, 8, 8)
    _ds_store_b192 = MemoryInstruction(DSStoreB192,  1, 1, 6, 6)
    _ds_store_b128 = MemoryInstruction(DSStoreB128,  1, 1, 4, 4)
    _ds_store_b96 = MemoryInstruction(DSStoreB96,  1, 1, 3, 3)
    _ds_store2_b64 = MemoryInstruction(DSStore2B64,  1, 2, 2, 2)
    _ds_store_b64 = MemoryInstruction(DSStoreB64,    1, 1, 2, 2)
    _ds_store2_b32 = MemoryInstruction(DSStore2B32,  1, 2, 1, 1)
    _ds_store_b32 = MemoryInstruction(DSStoreB32,    1, 1, 1, 1)
    _ds_store_b16 = MemoryInstruction(DSStoreB16,    1, 1, 1, 0.5)
    _ds_store_b8 = MemoryInstruction(DSStoreB8,      1, 1, 1, 0.25)
    ########################################
    # Global Read
    _flat_load_b192 = MemoryInstruction(FlatLoadB192, 1, 0, 0, 6)
    _flat_load_b128 = MemoryInstruction(FlatLoadB128, 1, 0, 0, 4)
    _flat_load_b64 = MemoryInstruction(FlatLoadB64,   1, 0, 0, 2)
    _flat_load_b32 = MemoryInstruction(FlatLoadB32,   1, 0, 0, 1)

    _buffer_load_b192 = MemoryInstruction(BufferLoadB192, 1, 0, 0, 6)
    _buffer_load_b128 = MemoryInstruction(BufferLoadB128, 1, 0, 0, 4)
    _buffer_load_b96 = MemoryInstruction(BufferLoadB96, 1, 0, 0, 3)
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
      chosen_load_b192 = _buffer_load_b192
      chosen_load_b128 = _buffer_load_b128
      chosen_load_b96 = _buffer_load_b96
      chosen_load_b64  = _buffer_load_b64
      chosen_load_b32  = _buffer_load_b32
      chosen_load_b16  = _buffer_load_d16_b16
      chosen_load_b8   = _buffer_load_d16_u8
    else:
      #TODO: add flatl_load_b96
      chosen_load_b192 = _flat_load_b192
      chosen_load_b128 = _flat_load_b128
      chosen_load_b64  = _flat_load_b64
      chosen_load_b32  = _flat_load_b32
      chosen_load_b16  = _flat_load_b32 # not supported
      chosen_load_b8   = _flat_load_b32 # not supported

    chosen_store_b128 = _flat_store_b128
    chosen_store_b64  = _flat_store_b64
    chosen_store_b32  = _flat_store_b32

    self.memoryInstructions = {
          "GlobalRead" : [ chosen_load_b192, chosen_load_b128, chosen_load_b96, chosen_load_b64,
                           chosen_load_b32, chosen_load_b16, chosen_load_b8 ],
          "GlobalWrite": [ chosen_store_b128, chosen_store_b64, chosen_store_b32 ],
          "LocalRead"  : [ _ds_load_b192, _ds_load_b128, _ds_load2_b64, _ds_load_b64,
                           _ds_load2_b32, _ds_load_b32, _ds_load_u16, _ds_load_u8],
          "TrLocalRead": [_ds_load_b64_tr_b16, _ds_load_b128_tr_b16, _ds_load_b64_tr_b8,
                          _ds_load_b64_tr_b4, _ds_load_b96_tr_b6],
          "LocalWrite" : [ _ds_store_b256, _ds_store_b192, _ds_store_b128, _ds_store_b96,
                           _ds_store2_b64, _ds_store_b64, _ds_store2_b32,
                           _ds_store_b32, _ds_store_b16, _ds_store_b8 ]
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

    if tensorParametersMXSA is not None:
      self.initGlobalReadMemoryInstruction(instructions, tensorParametersMXSA, self.states.bpr)
      self.initLocalWriteMemoryInstruction(instructions, kernel, tensorParametersMXSA, self.states.bpr)
      self.initLocalReadMemoryInstruction(instructions, kernel, tensorParametersMXSA, self.states.bpr)

    if tensorParametersMXSB is not None:
      self.initGlobalReadMemoryInstruction(instructions, tensorParametersMXSB, self.states.bpr)
      self.initLocalWriteMemoryInstruction(instructions, kernel, tensorParametersMXSB, self.states.bpr)
      self.initLocalReadMemoryInstruction(instructions, kernel, tensorParametersMXSB, self.states.bpr)

    if tensorParametersM is not None:
      self.initGlobalReadMemoryInstruction(instructions, tensorParametersM, self.states.bpr)
      self.initLocalWriteMemoryInstruction(instructions, kernel, tensorParametersM, self.states.bpr)
      self.initLocalReadMemoryInstruction(instructions, kernel, tensorParametersM, self.states.bpr)

    # global reads per instruction
    tensorParametersA["nrcvpi"] = int((tensorParametersA["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersA["bpeGR"])
    tensorParametersB["nrcvpi"] = int((tensorParametersB["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersB["bpeGR"])
    tensorParametersA["nwcvpi"] = int((tensorParametersA["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersA["bpeDS"])
    tensorParametersB["nwcvpi"] = int((tensorParametersB["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersB["bpeDS"])

    if tensorParametersMXSA is not None:
      tensorParametersMXSA["nrcvpi"] = int((tensorParametersMXSA["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersMXSA["bpeGR"])
      tensorParametersMXSA["nwcvpi"] = int((tensorParametersMXSA["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersMXSA["bpeDS"])

    if tensorParametersMXSB is not None:
      tensorParametersMXSB["nrcvpi"] = int((tensorParametersMXSB["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersMXSB["bpeGR"])
      tensorParametersMXSB["nwcvpi"] = int((tensorParametersMXSB["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersMXSB["bpeDS"])

    if tensorParametersM is not None:
      tensorParametersM["nrcvpi"] = int((tensorParametersM["globalReadInstruction"].totalWidth*self.states.bpr)/tensorParametersM["bpeDS"])
      tensorParametersM["nwcvpi"] = int((tensorParametersM["localWriteInstruction"].totalWidth*self.states.bpr)/tensorParametersM["bpeDS"])

    ####################################
    # VGPR Allocation
    ####################################

    def vgprAllocationImplClassic():
      ####################################
      # num vgprs: valu
      if kernel["EnableMatrixInstruction"]:
        #jgolds bpeCinternal because we are allocating accumulation registers here
        self.states.c.numVgprValu = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.states.bpeCinternal)//self.states.bpr

        # pack or input conversion DTV case, need double buffer (LoopIters * 2)
        numVgprBufferA = self.states.numVgprBuffer if not (self.states.packDTVA or self.states.convDTVA) else kernel["LoopIters"] * 2
        numVgprBufferB = self.states.numVgprBuffer if not (self.states.packDTVB or self.states.convDTVB) else kernel["LoopIters"] * 2
        valuBlocks  = self.states.numVgprBuffer * kernel["InnerUnroll"] # for Sparse
        valuBlocksA = (1.5 if kernel["HalfPLRA"] else numVgprBufferA) * kernel["InnerUnroll"]
        valuBlocksB = (1.5 if kernel["HalfPLRB"] else numVgprBufferB) * kernel["InnerUnroll"]
        if kernel["ProblemType"]["MXBlockA"]:
          valuBlocksMXSA = numVgprBufferA * kernel["InnerUnroll"]
        if kernel["ProblemType"]["MXBlockB"]:
          valuBlocksMXSB = numVgprBufferB * kernel["InnerUnroll"]

        self.states.a.numVgprValuPerBlock = int(kernel["MIWaveTileA"] * kernel["MIInputPerThreadA"] * tensorParametersA["bpe"] // self.states.bpr)
        self.states.b.numVgprValuPerBlock = int(kernel["MIWaveTileB"] * kernel["MIInputPerThreadB"] * tensorParametersB["bpe"] // self.states.bpr)

        #TODO: remove this if upcoming compiler changes applied
        if kernel["ProblemType"]["MacDataTypeA"].numBytes() == 0.75 or kernel["ProblemType"]["MacDataTypeB"].numBytes() == 0.75:
          if kernel["enableLDSTrA"]:
            numVgprPerSubIter = int(kernel["MIInputPerThreadA"] * tensorParametersA["bpe"] // self.states.bpr)
            numVgprPerLocalRead, vgprAlignment = 3, 4
            numLoads = numVgprPerSubIter // numVgprPerLocalRead
            self.states.a.numVgprValuPerBlock = kernel["MIWaveTileA"] * numLoads * vgprAlignment

          if kernel["enableLDSTrB"]:
            numVgprPerSubIter = int(kernel["MIInputPerThreadB"] * tensorParametersB["bpe"] // self.states.bpr)
            numVgprPerLocalRead, vgprAlignment = 3, 4
            numLoads = numVgprPerSubIter // numVgprPerLocalRead
            self.states.b.numVgprValuPerBlock = kernel["MIWaveTileB"] * numLoads * vgprAlignment

        # change numVgprValuAPerBlock to 0 if DirectToVgpr is enabled (except for DTV + (pack or input conversion))
        if kernel["DirectToVgprA"] and not (self.states.packDTVA or self.states.convDTVA):
          self.states.a.numVgprValuPerBlock = 0
        if kernel["DirectToVgprB"] and not (self.states.packDTVB or self.states.convDTVB):
          self.states.b.numVgprValuPerBlock = 0

        self.states.a.numVgprValu = int(self.states.a.numVgprValuPerBlock * valuBlocksA)
        if self.states.lrvwTileA > 1 and tensorParametersA["bpe"] < 4 and not (kernel["UsePLRPack"] and self.states.numItersPLR):
          self.states.a.numVgprValu = self.states.a.numVgprValuPerBlock * kernel["InnerUnroll"]

        self.states.b.numVgprValu = int(self.states.b.numVgprValuPerBlock * valuBlocksB)
        if self.states.lrvwTileB > 1 and tensorParametersB["bpe"] < 4 and not (kernel["UsePLRPack"] and self.states.numItersPLR):
          self.states.b.numVgprValu = self.states.b.numVgprValuPerBlock * kernel["InnerUnroll"]

        if kernel["ProblemType"]["MXBlockA"]:
          self.states.mxsa.numVgprValuPerBlock = kernel["MIWaveTileMXSA"] * kernel["MIInputPerThreadMXSA"] // self.states.bpr
          # workaround for gfx950
          # need to allocate same amount of MIWaveTile
          if isgfx950:
            self.states.mxsa.numVgprValuPerBlock = kernel["MIWaveTileMXSA"]
          if kernel["DirectToVgprMXSA"] and not (self.states.packDTVA or self.states.convDTVA):
            self.states.mxsa.numVgprValuPerBlock = 0
          # MX scale registers are consumed by local-read and wmma paths; avoid
          # emitting unresolved vgprValuMXSA symbols when integer division rounds to 0.
          elif self.states.mxsa.numVgprValuPerBlock == 0:
            self.states.mxsa.numVgprValuPerBlock = kernel["MIWaveTileMXSA"]
          self.states.mxsa.numVgprValu = self.states.mxsa.numVgprValuPerBlock * valuBlocksMXSA
          if not self.states.asmCaps["HasWMMA_V3"] and self.states.lrvwTileMXSA > 1:
            self.states.mxsa.numVgprValu = self.states.mxsa.numVgprValuPerBlock * kernel["InnerUnroll"]

        if kernel["ProblemType"]["MXBlockB"]:
          self.states.mxsb.numVgprValuPerBlock = kernel["MIWaveTileMXSB"] * kernel["MIInputPerThreadMXSB"] // self.states.bpr
          # workaround for gfx950
          # need to allocate same amount of MIWaveTile
          if isgfx950:
            self.states.mxsb.numVgprValuPerBlock = kernel["MIWaveTileMXSB"]
          if kernel["DirectToVgprMXSB"] and not (self.states.packDTVB or self.states.convDTVB):
            self.states.mxsb.numVgprValuPerBlock = 0
          # MX scale registers are consumed by local-read and wmma paths; avoid
          # emitting unresolved vgprValuMXSB symbols when integer division rounds to 0.
          elif self.states.mxsb.numVgprValuPerBlock == 0:
            self.states.mxsb.numVgprValuPerBlock = kernel["MIWaveTileMXSB"]
          self.states.mxsb.numVgprValu = self.states.mxsb.numVgprValuPerBlock * valuBlocksMXSB
          if not self.states.asmCaps["HasWMMA_V3"] and self.states.lrvwTileMXSB > 1:
            self.states.mxsb.numVgprValu = self.states.mxsb.numVgprValuPerBlock * kernel["InnerUnroll"]

      else: # mac instruction
        valuBlocksA = (1 + kernel["PrefetchLocalRead"]) * kernel["InnerUnroll"]
        valuBlocksB = (1 + kernel["PrefetchLocalRead"]) * kernel["InnerUnroll"]

        if kernel["UseDotInstruction"]:
          # dot2: at least read NumDotElements elements
          self.states.a.numVgprValuPerBlock = int(kernel["ThreadTileA"] * tensorParametersA["bpe"] * kernel["NumDotElements"] // self.states.bpr)
          self.states.b.numVgprValuPerBlock = int(kernel["ThreadTileB"] * tensorParametersB["bpe"] * kernel["NumDotElements"] // self.states.bpr)
        else:
          self.states.a.numVgprValuPerBlock = int(kernel["ThreadTileA"] * tensorParametersA["bpe"] // self.states.bpr)
          self.states.b.numVgprValuPerBlock = int(kernel["ThreadTileB"] * tensorParametersB["bpe"] // self.states.bpr)

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
          self.states.m.numVgprValuPerBlock = kernel["MIWaveTileMetadata"] * roundUp(kernel["MIInputPerThreadMetadata"] / self.states.bpr)
          if kernel["enableLDSTrMetadata"]:
            multiplyBy = 1 if kernel["MIInputPerThreadMetadata"] // self.states.bpr == 2 else 2
            self.states.m.numVgprValuPerBlock *= multiplyBy
          self.states.m.numVgprValu = self.states.m.numVgprValuPerBlock * valuBlocks
          if self.states.lrvwTileMetadata > 1 and tensorParametersM["bpe"] < 4:
            self.states.m.numVgprValu = self.states.m.numVgprValuPerBlock * kernel["InnerUnroll"]


      ####################################
      # num vgprs: global -> local elements A
      self.states.a.numVgprG2L = 0
      numVgprG2LAllocatedLocal = 0

      bpeMax = tensorParametersA["bpeDS"] if kernel["ConvertAfterDS"] else max(tensorParametersA["bpeGR"], tensorParametersA["bpe"])
      statesANumVgprG2L = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] * \
          kernel["GlobalReadVectorWidthA"] * bpeMax) / (float)(self.states.bpr))
      tpA      = self.states.bpr if bpeMax * vwa < self.states.bpr else bpeMax * vwa
      tpALocal = self.states.bpr if tensorParametersA["bpe"] * vwa < self.states.bpr else tensorParametersA["bpe"] * vwa
      numVgprG2LAllocatedLocal = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] * \
          tpALocal) / (float)(self.states.bpr))
      if (self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]):
        if bpeMax * vwa < self.states.bpr:
          # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
          # We should optimize the usage for better performance.
          statesANumVgprG2LAllocated = statesANumVgprG2L * (int)(self.states.bpr/(bpeMax * vwa))
        else:
          # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
          # We should optimize the usage for better performance.
          statesANumVgprG2LAllocated = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] * \
            tpA) / (float)(self.states.bpr))
          #TODO: remove this if upcoming compiler changes getting merged
          if tensorParametersA["globalReadInstruction"].blockWidth == 3:
            statesANumVgprG2LAllocated = roundUp(statesANumVgprG2LAllocated * 4 / 3)
      else:
        statesANumVgprG2LAllocated = statesANumVgprG2L

      if (not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]) and not kernel["enableTDMA"]:
        self.states.a.numVgprG2L = statesANumVgprG2L
        self.states.a.numVgprG2LAllocated = statesANumVgprG2LAllocated
        self.states.a.numVgprG2LTailloopAllocated = statesANumVgprG2LAllocated if tensorParametersA["globalReadInstruction"].blockWidth != 6 else roundUp(statesANumVgprG2LAllocated * 4 / 3)
      else:
        self.states.a.numVgprG2L = 0
        self.states.a.numVgprG2LAllocated = 0
        self.states.a.numVgprG2LTailloopAllocated = statesANumVgprG2LAllocated if tensorParametersA["globalReadInstruction"].blockWidth != 6 else roundUp(statesANumVgprG2LAllocated * 4 / 3)
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
          self.states.a.numVgprG2L *= int(bpeA // bpeGRA)
          self.states.a.numVgprG2LAllocated *= int(bpeA // bpeGRA)

      # num vgprs: global -> local elements : MXSA
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numVgprG2L = 0
        numVgprG2LMXSAllocatedLocal = 0

        statesMXSANumVgprG2L = roundUp((kernel["NumLoadsCoalescedMXSA"] * kernel["NumLoadsPerpendicularMXSA"] * \
            kernel["GlobalReadVectorWidthMXSA"]) / (float)(self.states.bpr))
        tpMXSALocal = self.states.bpr if vwmxsa < self.states.bpr else vwmxsa
        numVgprG2LMXSAllocatedLocal = roundUp((kernel["NumLoadsCoalescedMXSA"] * kernel["NumLoadsPerpendicularMXSA"] * \
            tpMXSALocal) / (float)(self.states.bpr))

        if (self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]) and (vwmxsa < self.states.bpr):
          # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
          # We should optimize the usage for better performance.
          statesMXSANumVgprG2LAllocated = statesMXSANumVgprG2L * (int)(self.states.bpr/vwmxsa)
        else:
          statesMXSANumVgprG2LAllocated = statesMXSANumVgprG2L

        if (not kernel["DirectToLdsMXSA"] or self.do["KeepDirectToLdsAlloc"]) and not kernel["enableTDMA"]:
          self.states.mxsa.numVgprG2L = statesMXSANumVgprG2L
          self.states.mxsa.numVgprG2LAllocated = statesMXSANumVgprG2LAllocated
          self.states.mxsa.numVgprG2LTailloopAllocated = self.states.mxsa.numVgprG2LAllocated
        else:
          self.states.mxsa.numVgprG2L = 0
          self.states.mxsa.numVgprG2LAllocated = 0
          self.states.mxsa.numVgprG2LTailloopAllocated = statesMXSANumVgprG2LAllocated
        # using _ds_store_b8: need one more vgpr space to do lshr
        if tensorParametersMXSA["localWriteInstruction"].blockWidth == 0.25:
          self.states.mxsa.numVgprG2L = self.states.mxsa.numVgprG2L * 2
          self.states.mxsa.numVgprG2LAllocated += numVgprG2LMXSAllocatedLocal
          self.states.mxsa.numVgprG2LTailloopAllocated += numVgprG2LMXSAllocatedLocal
        # double numVgprG2L if DirectToVgpr is enabled
        if kernel["DirectToVgprMXSA"]:
          self.states.mxsa.numVgprG2L *= 2
          self.states.mxsa.numVgprG2LAllocated *= 2

      # num vgprs: global -> local elements : B
      self.states.b.numVgprG2L = 0
      numVgprG2LAllocatedLocal = 0

      bpeMax = tensorParametersB["bpeDS"] if kernel["ConvertAfterDS"] else max(tensorParametersB["bpeGR"], tensorParametersB["bpe"])
      statesBNumVgprG2L = roundUp((kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"] * \
          kernel["GlobalReadVectorWidthB"] * bpeMax) / (float)(self.states.bpr))
      tpB      = self.states.bpr if bpeMax * vwb < self.states.bpr else bpeMax * vwb
      tpBLocal = self.states.bpr if tensorParametersB["bpe"] * vwb < self.states.bpr else tensorParametersB["bpe"] * vwb
      numVgprG2LAllocatedLocal = roundUp((kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"] * \
          tpBLocal) / (float)(self.states.bpr))

      if (self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]):
        if bpeMax * vwb < self.states.bpr:
          # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
          # We should optimize the usage for better performance.
          statesBNumVgprG2LAllocated = statesBNumVgprG2L * (int)(self.states.bpr/(bpeMax * vwb))
        else:
          # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
          # We should optimize the usage for better performance.
          statesBNumVgprG2LAllocated = roundUp((kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"] * \
            tpB) / (float)(self.states.bpr))
          if tensorParametersB["globalReadInstruction"].blockWidth == 3:
            statesBNumVgprG2LAllocated = roundUp(statesBNumVgprG2LAllocated * 4 / 3)
      else:
        statesBNumVgprG2LAllocated = statesBNumVgprG2L
      if (not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]) and not kernel["enableTDMB"]:
        self.states.b.numVgprG2L = statesBNumVgprG2L
        self.states.b.numVgprG2LAllocated = statesBNumVgprG2LAllocated
        self.states.b.numVgprG2LTailloopAllocated = statesBNumVgprG2LAllocated if tensorParametersB["globalReadInstruction"].blockWidth != 6 else roundUp(statesBNumVgprG2LAllocated * 4 / 3)
      else:
        self.states.b.numVgprG2L = 0
        self.states.b.numVgprG2LAllocated = 0
        self.states.b.numVgprG2LTailloopAllocated = statesBNumVgprG2LAllocated if tensorParametersB["globalReadInstruction"].blockWidth != 6 else roundUp(statesBNumVgprG2LAllocated * 4 / 3)
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
          self.states.b.numVgprG2L *= int(bpeB // bpeGRB)
          self.states.b.numVgprG2LAllocated *= int(bpeB // bpeGRB)

      # num vgprs: global -> local elements : MXSB
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numVgprG2L = 0
        numVgprG2LMXSBllocatedLocal = 0

        statesMXSBNumVgprG2L = roundUp((kernel["NumLoadsCoalescedMXSB"] * kernel["NumLoadsPerpendicularMXSB"] * \
            kernel["GlobalReadVectorWidthMXSB"]) / (float)(self.states.bpr))
        tpMXSBLocal = self.states.bpr if vwmxsb < self.states.bpr else vwmxsb
        numVgprG2LMXSBllocatedLocal = roundUp((kernel["NumLoadsCoalescedMXSB"] * kernel["NumLoadsPerpendicularMXSB"] * \
            tpMXSBLocal) / (float)(self.states.bpr))

        if (self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]) and (vwmxsb < self.states.bpr):
          # This check is to reserve porential usage of VGPRs for gfx12 8-bit code gen
          # We should optimize the usage for better performance.
          statesMXSBNumVgprG2LAllocated = statesMXSBNumVgprG2L * (int)(self.states.bpr/vwmxsb)
        else:
          statesMXSBNumVgprG2LAllocated = statesMXSBNumVgprG2L

        if (not kernel["DirectToLdsMXSB"] or self.do["KeepDirectToLdsAlloc"]) and not kernel["enableTDMB"]:
          self.states.mxsb.numVgprG2L = statesMXSBNumVgprG2L
          self.states.mxsb.numVgprG2LAllocated = statesMXSBNumVgprG2LAllocated
          self.states.mxsb.numVgprG2LTailloopAllocated = self.states.mxsb.numVgprG2LAllocated
        else:
          self.states.mxsb.numVgprG2L = 0
          self.states.mxsb.numVgprG2LAllocated = 0
          self.states.mxsb.numVgprG2LTailloopAllocated = statesMXSBNumVgprG2LAllocated
        # using _ds_store_b8: need one more vgpr space to do lshr
        if tensorParametersMXSB["localWriteInstruction"].blockWidth == 0.25:
          self.states.mxsb.numVgprG2L = self.states.mxsb.numVgprG2L * 2
          self.states.mxsb.numVgprG2LAllocated += numVgprG2LMXSBllocatedLocal
          self.states.mxsb.numVgprG2LTailloopAllocated += numVgprG2LMXSBllocatedLocal
        # double numVgprG2L if DirectToVgpr is enabled
        if kernel["DirectToVgprMXSB"]:
          self.states.mxsb.numVgprG2L *= 2
          self.states.mxsb.numVgprG2LAllocated *= 2

      # num vgprs: global -> local elements : Metadata
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
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numVgprLocalReadAddr = 1 * self.states.rpla
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numVgprLocalReadAddr = 1 * self.states.rpla

      self.states.a.numVgprLocalWriteAddr = 0 if kernel["LocalWriteUseSgprA"] or kernel["enableTDMA"] else 1 * self.states.rpla
      self.states.b.numVgprLocalWriteAddr = 0 if kernel["LocalWriteUseSgprB"] or kernel["enableTDMB"] else 1 * self.states.rpla
      self.states.m.numVgprLocalWriteAddr = 0 if kernel["ProblemType"]["Sparse"] and kernel["LocalWriteUseSgprMetadata"] else 1 * self.states.rpla
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numVgprLocalWriteAddr = 0 if kernel["LocalWriteUseSgprMXSA"] or kernel["enableTDMA"] else 1 * self.states.rpla
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numVgprLocalWriteAddr = 0 if kernel["LocalWriteUseSgprMXSB"] or kernel["enableTDMB"] else 1 * self.states.rpla

      self.states.a.numVgprLocalReadSwapAddr = 0
      self.states.b.numVgprLocalReadSwapAddr = 0
      self.states.m.numVgprLocalReadSwapAddr = 0
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numVgprLocalReadSwapAddr = 0
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numVgprLocalReadSwapAddr = 0

      self.states.a.numVgprLocalWriteSwapAddr = 0
      self.states.b.numVgprLocalWriteSwapAddr = 0
      self.states.m.numVgprLocalWriteSwapAddr = 0
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numVgprLocalWriteSwapAddr = 0
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numVgprLocalWriteSwapAddr = 0

      self.states.a.numVgprLocalWriteAddrTailLoop = 0 if not (kernel["DirectToLdsA"] and kernel["NonDTLTailLoopA"]) else 1 * self.states.rpla
      self.states.b.numVgprLocalWriteAddrTailLoop = 0 if not (kernel["DirectToLdsB"] and kernel["NonDTLTailLoopB"]) else 1 * self.states.rpla
      self.states.m.numVgprLocalWriteAddrTailLoop = 0 if not kernel["ProblemType"]["Sparse"] or not (kernel["DirectToLdsMetadata"] and kernel["NonDTLTailLoopMetadata"]) else 1 * self.states.rpla
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numVgprLocalWriteAddrTailLoop = 0 if not (kernel["DirectToLdsMXSA"] and kernel["NonDTLTailLoopMXSA"]) else 1 * self.states.rpla
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numVgprLocalWriteAddrTailLoop = 0 if not (kernel["DirectToLdsMXSB"] and kernel["NonDTLTailLoopMXSB"]) else 1 * self.states.rpla

      numVgprMultiplierA = 1
      numVgprMultiplierB = 1
      numVgprMultiplierMXSA = 1
      numVgprMultiplierMXSB = 1
      numVgprMultiplierMetadata = 1
      if kernel["ProblemType"]["MXBlockA"]:
        numVgprMultiplierMXSA = 1
      if kernel["ProblemType"]["MXBlockB"]:
        numVgprMultiplierMXSB = 1

      maxLDSConstOffset = self.states.regCaps["maxLDSConstOffset"]
      if self.states.archCaps["DeviceLDS"] > maxLDSConstOffset:
        hasMultipleBuffer = kernel["ExpandPointerSwap"] and not kernel["1LDSBuffer"] and not kernel["StoreSwapAddr"]
        maxOffsetA = kernel["LdsNumElementsAlignedA"]
        maxOffsetB = kernel["LdsNumElementsAlignedB"]
        maxOffsetMXSA = kernel["LdsNumElementsAlignedMXSA"]
        maxOffsetMXSB = kernel["LdsNumElementsAlignedMXSB"]
        maxOffsetMetadata = kernel["LdsNumElementsAlignedMetadata"]
        maxOffsetMetadata = kernel["LdsNumElementsAlignedMetadata"]
        if hasMultipleBuffer:
          maxOffsetA += kernel["LdsOffsetA_Blk"]
          maxOffsetB += kernel["LdsOffsetA_Blk"]
          maxOffsetMXSA += kernel["LdsOffsetA_Blk"]
          maxOffsetMXSB += kernel["LdsOffsetA_Blk"]
          maxOffsetMetadata += kernel["LdsOffsetA_Blk"]

        numVgprMultiplierA = maxOffsetA // maxLDSConstOffset + 1
        numVgprMultiplierB = maxOffsetB // maxLDSConstOffset + 1
        numVgprMultiplierMXSA = maxOffsetMXSA // maxLDSConstOffset + 1
        numVgprMultiplierMXSB = maxOffsetMXSB // maxLDSConstOffset + 1
        numVgprMultiplierMetadata = maxOffsetMetadata // maxLDSConstOffset + 1

      self.states.a.numVgprLocalReadAddr *= numVgprMultiplierA
      self.states.a.numVgprLocalWriteAddr *= numVgprMultiplierA
      self.states.a.numVgprLocalWriteAddrTailLoop *= numVgprMultiplierA

      self.states.b.numVgprLocalReadAddr *= numVgprMultiplierB
      self.states.b.numVgprLocalWriteAddr *= numVgprMultiplierB
      self.states.b.numVgprLocalWriteAddrTailLoop *= numVgprMultiplierB

      self.states.m.numVgprLocalReadAddr *= numVgprMultiplierMetadata
      self.states.m.numVgprLocalWriteAddr *= numVgprMultiplierMetadata
      self.states.m.numVgprLocalWriteAddrTailLoop *= numVgprMultiplierMetadata
      if not (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]):
        self.states.m.numVgprLocalReadAddr = 0

      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numVgprLocalReadAddr *= numVgprMultiplierMXSA
        self.states.mxsa.numVgprLocalWriteAddr *= numVgprMultiplierMXSA
        self.states.mxsa.numVgprLocalWriteAddrTailLoop *= numVgprMultiplierMXSA

      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numVgprLocalReadAddr *= numVgprMultiplierMXSB
        self.states.mxsb.numVgprLocalWriteAddr *= numVgprMultiplierMXSB
        self.states.mxsb.numVgprLocalWriteAddrTailLoop *= numVgprMultiplierMXSB

      # do not allocate local read address register if DirectToVgpr is enabled
      if kernel["DirectToVgprA"]:
        self.states.a.numVgprLocalReadAddr = 0
      if kernel["ProblemType"]["MXBlockA"] and kernel["DirectToVgprMXSA"]:
        self.states.mxsa.numVgprLocalReadAddr = 0
        self.states.mxsa.numVgprLocalWriteAddr = 0
      if kernel["DirectToVgprB"]:
        self.states.b.numVgprLocalReadAddr = 0
      if kernel["ProblemType"]["MXBlockB"] and kernel["DirectToVgprMXSB"]:
        self.states.mxsb.numVgprLocalReadAddr = 0
        self.states.mxsb.numVgprLocalWriteAddr = 0
      if not (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]):
        self.states.m.numVgprLocalWriteAddr = 0

      if kernel["ProblemType"]["MXBlockA"] and kernel["DirectToVgprMXSA"]:
        self.states.mxsa.numVgprLocalReadAddr = 0
      if kernel["ProblemType"]["MXBlockB"] and kernel["DirectToVgprMXSB"]:
        self.states.mxsb.numVgprLocalReadAddr = 0


      # do not allocate local write address register if DirectToVgpr is enabled
      if kernel["DirectToVgprA"] or kernel["DirectToLdsA"]:
        self.states.a.numVgprLocalWriteAddr = 0
      if kernel["DirectToVgprB"] or kernel["DirectToLdsB"]:
        self.states.b.numVgprLocalWriteAddr = 0

      if kernel["ProblemType"]["MXBlockA"] and (kernel["DirectToVgprMXSA"] or kernel["DirectToLdsMXSA"]):
        self.states.mxsa.numVgprLocalWriteAddr = 0
      if kernel["ProblemType"]["MXBlockB"] and (kernel["DirectToVgprMXSB"] or kernel["DirectToLdsMXSB"]):
        self.states.mxsb.numVgprLocalWriteAddr = 0

      if kernel["StoreSwapAddr"]:
        if self.states.a.numVgprLocalReadAddr > 0:
          self.states.a.numVgprLocalReadSwapAddr = 1
        if self.states.b.numVgprLocalReadAddr > 0:
          self.states.b.numVgprLocalReadSwapAddr = 1
        if self.states.m.numVgprLocalReadAddr > 0:
          self.states.m.numVgprLocalReadSwapAddr = 1
        if kernel["ProblemType"]["MXBlockA"] and (self.states.mxsa.numVgprLocalReadAddr > 0):
          self.states.mxsa.numVgprLocalReadSwapAddr = 1
        if kernel["ProblemType"]["MXBlockB"] and (self.states.mxsb.numVgprLocalReadAddr > 0):
          self.states.mxsb.numVgprLocalReadSwapAddr = 1

        if not kernel["LocalWriteUseSgprA"] and (self.states.a.numVgprLocalWriteAddr > 0):
          self.states.a.numVgprLocalWriteSwapAddr = 1
        if not kernel["LocalWriteUseSgprB"] and (self.states.b.numVgprLocalWriteAddr > 0):
          self.states.b.numVgprLocalWriteSwapAddr = 1
        if kernel["ProblemType"]["Sparse"] and (not kernel["LocalWriteUseSgprMetadata"]) and (self.states.m.numVgprLocalWriteAddr > 0):
          self.states.m.numVgprLocalWriteSwapAddr = 1
        if kernel["ProblemType"]["MXBlockA"] and (not kernel["LocalWriteUseSgprMXSA"]) and (self.states.mxsa.numVgprLocalWriteAddr > 0):
          self.states.mxsa.numVgprLocalWriteSwapAddr = 1
        if kernel["ProblemType"]["MXBlockB"] and (not kernel["LocalWriteUseSgprMXSB"]) and (self.states.mxsb.numVgprLocalWriteAddr > 0):
          self.states.mxsb.numVgprLocalWriteSwapAddr = 1

      ####################################
      # num vgprs: global read addresses A
      numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
          * kernel["NumLoadsPerpendicularA"] * kernel["GlobalReadVectorWidthA"]
      numGlobalReadInstructionsA = int(numGlobalReadsA * tensorParametersA["bpeGR"])//\
          (tensorParametersA["globalReadInstruction"].blockWidth * 4)

      if kernel["enableTDMA"]:
        self.states.a.numVgprGlobalReadOffsets = 0
      elif kernel["BufferLoad"]:
        self.states.a.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsA * self.states.rpgo)
      else:
        numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.states.rpga

      if self.states.globalReadIncsUseVgpr:
        numVgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
            * self.states.rpga
      else:
        numVgprGlobalReadIncsA = 0

      # num vgprs: global read addresses MXSA
      if kernel["ProblemType"]["MXBlockA"]:
        numGlobalReadsMXSA = kernel["NumLoadsCoalescedMXSA"] \
            * kernel["NumLoadsPerpendicularMXSA"] * kernel["GlobalReadVectorWidthMXSA"]
        numGlobalReadInstructionsMXSA = int(numGlobalReadsMXSA / \
            (tensorParametersMXSA["globalReadInstruction"].blockWidth * 4))

        if kernel["enableTDMA"]:
          self.states.mxsa.numVgprGlobalReadOffsets = 0
        elif kernel["BufferLoad"]:
          self.states.mxsa.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsMXSA * self.states.rpgo)
        else:
          numVgprGlobalReadAddressesMXSA = numGlobalReadInstructionsMXSA * self.states.rpga

        if self.states.globalReadIncsUseVgpr:
          numVgprGlobalReadIncsMXSA = kernel["ProblemType"]["NumIndicesSummation"] \
              * self.states.rpga
        else:
          numVgprGlobalReadIncsMXSA = 0

      # num vgprs: global read addresses B
      numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
          * kernel["NumLoadsPerpendicularB"] * kernel["GlobalReadVectorWidthB"]
      numGlobalReadInstructionsB = int(numGlobalReadsB * tensorParametersB["bpeGR"])// \
          (tensorParametersB["globalReadInstruction"].blockWidth * 4)

      if kernel["enableTDMB"]:
        self.states.b.numVgprGlobalReadOffsets = 0
      elif kernel["BufferLoad"]:
        self.states.b.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsB * self.states.rpgo)
      else:
        numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.states.rpga

      if self.states.globalReadIncsUseVgpr:
        numVgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
            * self.states.rpga
      else:
        numVgprGlobalReadIncsB = 0

      if kernel["ProblemType"]["MXBlockB"]:
        numGlobalReadsMXSB = kernel["NumLoadsCoalescedMXSB"] \
            * kernel["NumLoadsPerpendicularMXSB"] * kernel["GlobalReadVectorWidthMXSB"]
        numGlobalReadInstructionsMXSB = int(numGlobalReadsMXSB / \
            (tensorParametersMXSB["globalReadInstruction"].blockWidth * 4))

        if kernel["enableTDMB"]:
          self.states.mxsb.numVgprGlobalReadOffsets = 0
        elif kernel["BufferLoad"]:
          self.states.mxsb.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsMXSB * self.states.rpgo)
        else:
          numVgprGlobalReadAddressesMXSB = numGlobalReadInstructionsMXSB * self.states.rpga

        if self.states.globalReadIncsUseVgpr:
          numVgprGlobalReadIncsMXSB = kernel["ProblemType"]["NumIndicesSummation"] \
              * self.states.rpga
        else:
          numVgprGlobalReadIncsMXSB = 0

      # num vgprs: global read addresses M
      if tensorParametersM is not None:
        numGlobalReadsMetadata = kernel["NumLoadsCoalescedMetadata"] \
            * kernel["NumLoadsPerpendicularMetadata"] * kernel["GlobalReadVectorWidthMetadata"]
        numGlobalReadInstructionsMetadata = int(numGlobalReadsMetadata * tensorParametersM["bpe"])//\
            (tensorParametersM["globalReadInstruction"].blockWidth * 4)
        if kernel["BufferLoad"]:
          self.states.m.numVgprGlobalReadOffsets = roundUp(numGlobalReadInstructionsMetadata * self.states.rpgo)
        if self.states.globalReadIncsUseVgpr:
          numVgprGlobalReadIncsMetadata = kernel["ProblemType"]["NumIndicesSummation"] \
              * self.states.rpga
        else:
          numVgprGlobalReadIncsMetadata = 0


      def GNLCOInit(tc):
        abmatrixinfo = self.states.a
        if tc == 'A':
          abmatrixinfo = self.states.a
        elif tc == 'B':
          abmatrixinfo = self.states.b
        elif tc == 'MXSA':
          abmatrixinfo = self.states.mxsa
        elif tc == 'MXSB':
          abmatrixinfo = self.states.mxsb
        elif tc == 'Metadata':
          abmatrixinfo = self.states.m

        if (tc in ("A", "B")) and kernel["DirectToLds%s"%tc] and kernel["UseGeneralizedNLCOne%s"%tc]:
          isMixedPrec = (kernel["ProblemType"]["DataTypeA"].numBytes() != kernel["ProblemType"]["DataTypeB"].numBytes())
          lrvw = kernel["LocalReadVectorWidth"]
          grvw = kernel["GlobalReadVectorWidth%c"%tc]
          bpe = kernel["ProblemType"]["DataType%s"%tc].numBytes()
          LdsStride = kernel["VectorWidth%s"%tc] * bpe * kernel["DepthU"]
          MinLdsBlockSizePerPad = (kernel[f"GlobalReadVectorWidth%s"%tc] * bpe) * kernel["WavefrontSize"]
          isM0PadEnough = LdsStride >= MinLdsBlockSizePerPad

          # Currently only supported for 16b, DTL, TLU=0 and grvw == lrvw
          if kernel["ProblemType"]["DataType"].numBytes() == 2 and not isMixedPrec \
             and kernel["ProblemType"]["TLU%s"%tc] == 0 and lrvw == grvw and \
             not isM0PadEnough:
            abmatrixinfo.gRDtlSwizzlePerpBlockSize = kernel["VectorWidth%s"%tc]
            abmatrixinfo.gRDtlSwizzleParaBlockSize = kernel["MatrixInstK"] // (kernel["LocalReadVectorWidth"])
          else:
            abmatrixinfo.gRDtlSwizzlePerpBlockSize = 0
            abmatrixinfo.gRDtlSwizzleParaBlockSize = 0

          ntpl = kernel["NumTotalPackedLoads%s"%tc]
          if kernel["ProblemType"]["TLU%s"%tc] == 1 and not kernel["enableLDSTr%s"%tc]:
            usePerpPerm = False
          elif kernel["ProblemType"]["TLU%s"%tc] == 1 and kernel["enableLDSTr%s"%tc]:
            usePerpPerm = (ntpl & (ntpl-1)) == 0
          else:
            # Currently only VW=1,2 is supported due to how the local read offset
            # is currently computed. Supporting VW=1,2 only required small modifications
            # to the offset calc.
            # TODO: Add support for VW=4,8, this will require more changes in LR offset
            # calculations
            usePerpPerm = False if kernel["VectorWidth%s"%tc] > 2 or kernel["ProblemType"]["DataType"].numBytes() == 2 else True

          permBlock = kernel["MatrixInstK"] if kernel["ProblemType"]["TLU%s"%tc] == 1 \
            else kernel["VectorWidth%s"%tc] * kernel["MatrixInstM"]
          abmatrixinfo.gNLCPermBlock  = permBlock
          abmatrixinfo.gNLCPerpStride = min([8, 2**int(math.log(ntpl, 2)), permBlock]) if usePerpPerm else 1
        else:
          abmatrixinfo.gNLCPerpStride = 1
          abmatrixinfo.gNLCPermBlock = 1
          abmatrixinfo.gRDtlSwizzlePerpBlockSize = 0
          abmatrixinfo.gRDtlSwizzleParaBlockSize = 0

      GNLCOInit('A')
      GNLCOInit('B')
      GNLCOInit('MXSA')
      GNLCOInit('MXSB')
      GNLCOInit('Metadata')

      numVgprAddressDbg = self.states.rpga if self.debugConfig.debugKernel else 0

      ####################################
      # num vgprs: c write address
      # 1 address where to write first value
      # 1 tmp address where to write current value

      ####################################
      # VGPR Assignment
      ####################################
      vgprIdx = 0
      papEnabled = self.isPrefetchAcrossPersistentEnabled(kernel)

      if bool(kernel["ProblemType"]["MXBlockA"]) ^ bool(kernel["ProblemType"]["MXBlockB"]):
        self.states.startMXDummyValuVgpr = vgprIdx
        vgprIdx += 2

      # TODO: alignment hack, figure out a better solution
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.startVgprValu = vgprIdx
        vgprIdx += self.states.mxsa.numVgprValu
        numVgprValuPackMXSA = 0
        if not kernel["UnrollMajorLDSMXSA"]:
          self.states.mxsa.startVgprValuPack = vgprIdx
          if self.states.lrvwTileMXSA > 1:
            numVgprValuPackMXSA = ceil(kernel["VectorWidthMXSA"] / self.states.bpr) * kernel["MIWaveTileMXSA"] // kernel["VectorWidthMXSA"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadMXSA"]
            if self.states.packDTVA:
              # pack DTV case, double the number
              numVgprValuPackMXSA *= 2
          else:
            numVgprValuPackMXSA = self.states.mxsa.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackMXSA * int(4 - 1)
        vgprIdx += numVgprValuPackMXSA
        self.states.mxsa.startVgprG2L = None
        if not kernel["DirectToLdsMXSA"] or self.do["KeepDirectToLdsAlloc"]:
          # DirectToVgpr + pack or input conversion case, overlap G2L and ValuPack
          if self.states.packDTVA and not papEnabled:
            self.states.mxsa.startVgprG2L = self.states.mxsa.startVgprValuPack
          elif self.states.convDTVA and not papEnabled:
            self.states.mxsa.startVgprG2L = self.states.mxsa.startVgprValu
          # if PGR = True, PAP could be possibly enabled, we move G2LA later to prevent it from being reclaimed
          # otherwise, put G2L here since it can overlap valu
          if (not kernel["PrefetchGlobalRead"]) and not papEnabled: # g2l can overlap valu
            self.states.mxsa.startVgprG2L = self.states.mxsa.startVgprValu
            vgprIdx = self.states.mxsa.startVgprValu  \
                + max(self.states.mxsa.numVgprValu + numVgprValuPackMXSA, self.states.mxsa.numVgprG2LAllocated)

      if kernel["ProblemType"]["MXBlockB"]:
        # TODO: alignment hack, figure out a better solution
        if(self.states.archCaps["VgprBank"]):
          residual = (vgprIdx % 4)
          if (residual % 2) == 0:
            # if 2-aligned bank(bank0 and bank2), move to bank1 or bank3.
            vgprIdx += 1
          if kernel["ISA"][:2] == (12, 5):
            vgprIdx = ((vgprIdx+1)//2)*2
        else:
          vgprIdx = ((vgprIdx+1)//2)*2

        self.states.mxsb.startVgprValu = vgprIdx
        vgprIdx += self.states.mxsb.numVgprValu
        numVgprValuPackMXSB = 0
        if not kernel["UnrollMajorLDSMXSB"]:
          self.states.mxsb.startVgprValuPack = vgprIdx
          if self.states.lrvwTileMXSB > 1:
            numVgprValuPackMXSB = ceil(kernel["VectorWidthMXSB"] / self.states.bpr) * kernel["MIWaveTileMXSB"] // kernel["VectorWidthMXSB"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadMXSB"]
            if self.states.packDTVB:
              # pack DTV case, double the number
              numVgprValuPackMXSB *= 2
          else:
            numVgprValuPackMXSB = self.states.mxsb.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackMXSB * int(4 - 1)
        vgprIdx += numVgprValuPackMXSB
        self.states.mxsb.startVgprG2L = None
        if not kernel["DirectToLdsMXSB"] or self.do["KeepDirectToLdsAlloc"]:
          # DirectToVgpr + pack or input conversion case, overlap G2L and ValuPack
          if self.states.packDTVB and not papEnabled:
            self.states.mxsb.startVgprG2L = self.states.mxsb.startVgprValuPack
          elif self.states.convDTVB and not papEnabled:
            self.states.mxsb.startVgprG2L = self.states.mxsb.startVgprValu
          # if PGR = True, PAP could be possibly enabled, we move G2LA later to prevent it from being reclaimed
          # otherwise, put G2L here since it can overlap valu
          if (not kernel["PrefetchGlobalRead"]) and not papEnabled: # g2l can overlap valu
            self.states.mxsb.startVgprG2L = self.states.mxsb.startVgprValu
            vgprIdx = self.states.mxsb.startVgprValu  \
                + max(self.states.mxsb.numVgprValu + numVgprValuPackMXSB, self.states.mxsb.numVgprG2LAllocated)

      if kernel["ProblemType"]["MXBlockA"] and not papEnabled:
        if self.states.mxsa.startVgprG2L is None and self.states.mxsa.numVgprG2LAllocated > 0:
          # TODO: alignment hack, figure out a better solution
          vgprIdx = ((vgprIdx+1)//2)*2
          self.states.mxsa.startVgprG2L = vgprIdx;
          if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
            vgprIdx += self.states.mxsa.numVgprG2LAllocated * 2
          else:
            vgprIdx += self.states.mxsa.numVgprG2LAllocated

      if kernel["ProblemType"]["MXBlockB"] and not papEnabled:
        if self.states.mxsb.startVgprG2L is None and self.states.mxsb.numVgprG2LAllocated > 0:
          # TODO: alignment hack, figure out a better solution
          vgprIdx = ((vgprIdx+1)//2)*2
          self.states.mxsb.startVgprG2L = vgprIdx;
          if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
            vgprIdx += self.states.mxsb.numVgprG2LAllocated * 2
          else:
            vgprIdx += self.states.mxsb.numVgprG2LAllocated

      vgprIdx = (vgprIdx+1)//2*2
      self.states.lastValuMXSAB = vgprIdx

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
          vgprIdx = self.states.c.startVgprValu + self.states.totalMixedAgprs
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
        if kernel["ProblemType"]["MXBlockA"]:
          self.startVgprGlobalReadOffsetMXSA = vgprIdx
          vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.states.mxsa.numVgprGlobalReadOffsets
        if kernel["ProblemType"]["MXBlockB"]:
          self.startVgprGlobalReadOffsetMXSB = vgprIdx
          vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.states.mxsb.numVgprGlobalReadOffsets
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
        if kernel["ProblemType"]["MXBlockA"]:
          self.startVgprGlobalReadAddressesMXSA = vgprIdx
          vgprIdx += numVgprGlobalReadAddressesMXSA
        if kernel["ProblemType"]["MXBlockB"]:
          self.startVgprGlobalReadAddressesMXSB = vgprIdx
          vgprIdx += numVgprGlobalReadAddressesMXSB

      if not kernel["LocalWriteUseSgprA"]:
        self.states.a.startVgprLocalWriteAddr = vgprIdx
        vgprIdx += self.states.a.numVgprLocalWriteAddr

      if not kernel["LocalWriteUseSgprB"]:
        self.states.b.startVgprLocalWriteAddr = vgprIdx
        vgprIdx += self.states.b.numVgprLocalWriteAddr

      # num vgprs: GL2 prefetch addresses
      if kernel["PrefetchGL2"]:
        self.gl2PrefetchInit(kernel, tensorParametersA, tensorParametersB)

      if kernel["ProblemType"]["MXBlockA"]:
        if not kernel["LocalWriteUseSgprMXSA"]:
          self.states.mxsa.startVgprLocalWriteAddr = vgprIdx
          vgprIdx += self.states.mxsa.numVgprLocalWriteAddr

      if kernel["ProblemType"]["MXBlockB"]:
        if not kernel["LocalWriteUseSgprMXSB"]:
          self.states.mxsb.startVgprLocalWriteAddr = vgprIdx
          vgprIdx += self.states.mxsb.numVgprLocalWriteAddr

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
      if kernel["ProblemType"]["MXBlockA"]:
        self.startVgprGlobalReadIncsMXSA = vgprIdx
        vgprIdx += numVgprGlobalReadIncsMXSA
      if kernel["ProblemType"]["MXBlockB"]:
        self.startVgprGlobalReadIncsMXSB = vgprIdx
        vgprIdx += numVgprGlobalReadIncsMXSB
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        self.startVgprGlobalReadIncsMetadata = vgprIdx
        vgprIdx += numVgprGlobalReadIncsMetadata

      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        self.states.m.startVgprLocalReadAddr = vgprIdx
        vgprIdx += self.states.m.numVgprLocalReadAddr
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.startVgprLocalReadAddr = vgprIdx
        vgprIdx += self.states.mxsa.numVgprLocalReadAddr
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.startVgprLocalReadAddr = vgprIdx
        vgprIdx += self.states.mxsb.numVgprLocalReadAddr
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

      if self.states.IncLdsBufSwitch:
        # Need backup for the first LocalReadAddr only (others will be calculated from the first one)
        self.states.a.startVgprLocalReadAddrOrig = vgprIdx
        vgprIdx += 1 if self.states.a.numVgprLocalReadAddr > 0 else 0
        if kernel["ProblemType"]["MXBlockA"]:
          self.states.mxsa.startVgprLocalReadAddrOrig = vgprIdx
          vgprIdx += 1 if self.states.mxsa.numVgprLocalReadAddr > 0 else 0
        self.states.b.startVgprLocalReadAddrOrig = vgprIdx
        vgprIdx += 1 if self.states.b.numVgprLocalReadAddr > 0 else 0
        if kernel["ProblemType"]["MXBlockB"]:
          self.states.mxsb.startVgprLocalReadAddrOrig = vgprIdx
          vgprIdx += 1 if self.states.mxsb.numVgprLocalReadAddr > 0 else 0

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

      self.states.startVgpr = vgprIdx

      self.states.a.startVgprValu = vgprIdx
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
      if (not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]) and not kernel["enableTDMA"]:
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
        if kernel["ISA"][:2] == (12, 5):
          vgprIdx = ((vgprIdx+1)//2)*2
      else:
        vgprIdx = ((vgprIdx+1)//2)*2

      self.states.b.startVgprValu = vgprIdx
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
      if (not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]) and not kernel["enableTDMB"]:
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
          (kernel["ProblemType"]["Sparse"] and not kernel["UnrollMajorLDSMetadata"] and (kernel["MIInputPerThreadMetadata"] == 4))) \
          and (kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()) or \
          (self.states.asmCaps["HasSWMMAC_gfx1250"] and kernel["ProblemType"]["Sparse"] and not kernel["UnrollMajorLDSMetadata"]):
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
          # gfx1250
          if self.states.m.numVgprValu >= 2:
            vgprIdx = ((vgprIdx+1)//2)*2
          self.states.m.startVgprValu = vgprIdx
          vgprIdx += self.states.m.numVgprValu
          numVgprValuPackMetadata = 0
          if not kernel["UnrollMajorLDSMetadata"] and not kernel["enableLDSTrMetadata"]:
            self.states.m.startVgprValuPack = vgprIdx
            if self.states.lrvwTileMetadata > 1:
              numVgprValuPackMetadata = roundUp(kernel["VectorWidthMetadata"] * tensorParametersM["bpe"] / self.states.bpr) * kernel["MIWaveTileMetadata"] // kernel["VectorWidthMetadata"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadMetadata"]
            else:
              numVgprValuPackMetadata = (kernel["MIInputPerThreadMetadata"]-1) * kernel["MIWaveTileMetadata"] * kernel["InnerUnroll"] * self.states.numVgprBufferPackMetadata
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

      if papEnabled and kernel["ProblemType"]["MXBlockA"]:
        if self.states.mxsa.startVgprG2L is None and self.states.mxsa.numVgprG2LAllocated > 0:
          # Keep MX scale PAP data in the durable read range instead of the
          # low MXS/VALU area that post-loop code may reuse.
          vgprIdx = ((vgprIdx+1)//2)*2
          self.states.mxsa.startVgprG2L = vgprIdx
          if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
            vgprIdx += self.states.mxsa.numVgprG2LAllocated*2
          else:
            vgprIdx += self.states.mxsa.numVgprG2LAllocated

      if papEnabled and kernel["ProblemType"]["MXBlockB"]:
        if self.states.mxsb.startVgprG2L is None and self.states.mxsb.numVgprG2LAllocated > 0:
          # Keep MX scale PAP data in the durable read range instead of the
          # low MXS/VALU area that post-loop code may reuse.
          vgprIdx = ((vgprIdx+1)//2)*2
          self.states.mxsb.startVgprG2L = vgprIdx
          if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
            vgprIdx += self.states.mxsb.numVgprG2LAllocated*2
          else:
            vgprIdx += self.states.mxsb.numVgprG2LAllocated

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

      # for cgemm or zgemm + MIAV case, allocate 2 or 4 vgpr for alpha calculation (cannot use tmp vgpr in write batch)
      if kernel["ProblemType"]["DataType"].isComplex() \
        and kernel["MIArchVgpr"]:

        # need proper alignment
        vgprIdx = ((vgprIdx+2 - 1)//2)*2
        self.states.startVgprAlphaTmp = vgprIdx
        vgprIdx += kernel["ProblemType"]["DataType"].numRegisters()

      # for swapping vgpr offsets of different lds buffers
      if self.states.a.numVgprLocalReadSwapAddr > 0:
        self.states.a.startVgprLocalReadSwapAddr = vgprIdx
        vgprIdx += 1
      if self.states.mxsa.numVgprLocalReadSwapAddr > 0:
        self.states.mxsa.startVgprLocalReadSwapAddr = vgprIdx
        vgprIdx += 1
      if self.states.m.numVgprLocalReadSwapAddr > 0:
        self.states.m.startVgprLocalReadSwapAddr = vgprIdx
        vgprIdx += 1
      if self.states.b.numVgprLocalReadSwapAddr > 0:
        self.states.b.startVgprLocalReadSwapAddr = vgprIdx
        vgprIdx += 1
      if self.states.mxsb.numVgprLocalReadSwapAddr > 0:
        self.states.mxsb.startVgprLocalReadSwapAddr = vgprIdx
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
      if kernel["ProblemType"]["MXBlockA"]:
        if self.states.mxsa.numVgprLocalWriteSwapAddr > 0:
          self.states.mxsa.startVgprLocalWriteSwapAddr = vgprIdx
          vgprIdx += 1
      if kernel["ProblemType"]["MXBlockB"]:
        if self.states.mxsb.numVgprLocalWriteSwapAddr > 0:
          self.states.mxsb.startVgprLocalWriteSwapAddr = vgprIdx
          vgprIdx += 1

      # X32F Emulation initializations
      # meaning of variables
      # useDirect32XEmulation (separate values for A and B):
      #   True: allocate extra buffer (either full (tranpose only) or interleave) to eliminate extra v_mov
      #   False: use temp Treg only for conversion (need some v_mov)
      # TF32EmuUseTransposeCode (separate values for A and B. For wider local read(lrvwTile>1) only):
      #   True: Generate extra transpose code (with v_swap)
      #   False: Use index tranpose and no tranpose code
      #          This is for cvt + sub only (means not dot2, not mfma)
      # TF32EmuInterleaveTreg:
      #   True: Allocate T reg with interleaving X regs for dest of local read
      #            T0-3
      #            X4-7
      #            T4-7
      #            X8-11
      #            ....
      #         This works with useDirect32XEmulation=Trie
      #         Wider local read case, we need TransposeCode=True
      #   False: Does not use interleave layout
      #         ider local read + index transpose case, this needs to be False
      def initTF32Emu():
        # for UseF32XEmulation only
        if not kernel["UseF32XEmulation"]:
          return 0, 0
        self.states.a.useDirect32XEmulationThis = self.states.a.useDirect32XEmulationNext = kernel["UseDirect32XEmulation"]
        self.states.b.useDirect32XEmulationThis = self.states.b.useDirect32XEmulationNext = kernel["UseDirect32XEmulation"]
        self.states.mxsa.useDirect32XEmulationThis = False
        self.states.mxsb.useDirect32XEmulationThis = False
        self.states.a.TF32EmuUseTransposeCode = False
        self.states.b.TF32EmuUseTransposeCode = False
        self.states.mxsa.TF32EmuUseTransposeCode = False
        self.states.mxsb.TF32EmuUseTransposeCode = False
        self.states.a.TF32EmuInterleaveTreg = False
        self.states.b.TF32EmuInterleaveTreg = False
        self.states.mxsa.TF32EmuInterleaveTreg = False
        self.states.mxsb.TF32EmuInterleaveTreg = False
        # do prefetch and scheduling for full pack code
        # this sceduling opt is for non CMS. No need to enable it for CMS
        self.states.doFullPackCodePrefetch = kernel["UsePLRPack"] and not kernel["UseCustomMainLoopSchedule"]
        # prefetch pack/prePack scheduling for non CMS only
        # We do not enable any ppack scheduling optimizations for PLR=0
        if (not kernel["UseCustomMainLoopSchedule"]) and self.states.numItersPLR:
          # enabhe prepack scheduling for this loop only for DTLA + B
          if kernel["DirectToLds"] == 1:
            # do packPre scheduling for This loop only not CLR or SubIter
            self.states.doPackPreSchedulingThisLoop = (not kernel["ClusterLocalRead"]) or kernel["ForceUnrollSubIter"]
          self.states.doPackPreSchedulingNextLoop = True
        if self.states.tailloopInNll:
          # disable all TF32 scheduling if tailloopInNll is enabled
          self.states.doFullPackCodePrefetch = False
          self.states.doPackPreSchedulingThisLoop = False
          self.states.doPackPreSchedulingNextLoop = False
        numVgprsEmuA = initTF32EmuAB(self.states.a, self.states.lrvwTileA)
        numVgprsEmuB = initTF32EmuAB(self.states.b, self.states.lrvwTileB)
        return numVgprsEmuA, numVgprsEmuB
      def initTF32EmuAB(sAorB: ABMatrixInfo, lrvwTile):
        # for UseF32XEmulation only
        if not kernel["UseF32XEmulation"]:
          return 0
        # number of Vreg for interleaveTreg. Half of ValuA or B. Need same block number as Valu
        numVForInterleave = sAorB.numVgprValu // 2
        numVForIndexTranspose = sAorB.numVgprValuPerBlock
        if kernel["ForceUnrollSubIter"]:
          # SubIter case, we devide local read into half at each prefetch
          numVForIndexTranspose //= 2
        # full prefetch pack case, we need to allocate full ValuA/B buffers
        if self.states.doFullPackCodePrefetch:
          if kernel["UseDirect32XEmulationInterleaveTreg"]:
            # use conventional Treg allocatin (interleaved Treg and Xreg)
            numV = numVForInterleave
            # enable TF32EmuInterleaveTreg
            sAorB.TF32EmuInterleaveTreg = True
          else:
            # allocate single full buffer as dest of local read
            numV = numVForIndexTranspose
          sAorB.useDirect32XEmulationThis = True
          sAorB.useDirect32XEmulationNext = True
          if kernel["UseMFMAF32XEmulation"]:
            # use transpose code for MFMA
            sAorB.useTransposeCodeThis = True
            sAorB.useTransposeCodeNext = True
          return numV
        # reg layout setting
        # At init stage, seting is same for this and next
        if sAorB.useDirect32XEmulationThis:
          # enable TF32EmuInterleaveTreg
          sAorB.TF32EmuInterleaveTreg = True
          numV = numVForInterleave
          if lrvwTile > 1:
            # useDirect32XEmulation case
            # Use wider local read + transpose code
            sAorB.useTransposeCodeThis = True
            sAorB.useTransposeCodeNext = True
        else:
           # no useDirect32XEmulation case, use temp reg version
           numV = adjustNumVForTF32Emu(sAorB, lrvwTile)
        return numV

      def adjustNumVForTF32Emu(sAorB: ABMatrixInfo, lrvwTile):
        # for UseF32XEmulation only
        if not kernel["UseF32XEmulation"]:
          return 0
        # no T reg for both This and Next Loop
        if lrvwTile > 1:
          # use tranpose code for wider local read
          sAorB.useTransposeCodeThis = True
          sAorB.useTransposeCodeNext = True
        numV = 0
        # disable TF32EmuInterleaveTreg
        sAorB.TF32EmuInterleaveTreg = False
        sAorB.useDirect32XEmulationThis = False
        sAorB.useDirect32XEmulationNext = False
        return numV

      def checkVregOverflowTF32Emu(vgprIdx, numV):
        # for UseF32XEmulation only
        if not kernel["UseF32XEmulation"]:
          return False
        # Do not allow adjustment for CMS or doFullPackCodePrefetch
        if kernel["UseCustomMainLoopSchedule"] or self.states.doFullPackCodePrefetch:
          return False
        # We need to consider 2 more vreg (Serial tmp)
        # Looks like we need more tmp vreg at tailloop
        # So far, max 32 tmp vregs might be used.
        # set 2 + 32 as buffer (tentative)
        # MFMA case, need 2 more
        bufferVregNum = 2 + 32
        if kernel["UseMFMAF32XEmulation"]:
          bufferVregNum += 2
        return vgprIdx + bufferVregNum + numV > self.states.regCaps["MaxVgpr"]

      # initial TF32Emu setting
      numVgprsEmuA, numVgprsEmuB = initTF32Emu()
      # numVreg adjustment
      # step 1 Adjustment for lrvwTileA/B==1
      #   start from B
      needAdjustment = checkVregOverflowTF32Emu(vgprIdx, numVgprsEmuA + numVgprsEmuB)
      if needAdjustment and self.states.lrvwTileB == 1:
        numVgprsEmuB = adjustNumVForTF32Emu(self.states.b, self.states.lrvwTileB)
      #   then, check A
      needAdjustment = checkVregOverflowTF32Emu(vgprIdx, numVgprsEmuA + numVgprsEmuB)
      if needAdjustment and self.states.lrvwTileA == 1 and not self.states.doFullPackCodePrefetch:
        numVgprsEmuA = adjustNumVForTF32Emu(self.states.a, self.states.lrvwTileA)
      # step 2 Adjustment for lrvwTileA/B>1
      #   start from B
      needAdjustment = checkVregOverflowTF32Emu(vgprIdx, numVgprsEmuA + numVgprsEmuB)
      if needAdjustment and self.states.lrvwTileB > 1 and not self.states.doFullPackCodePrefetch:
        numVgprsEmuB = adjustNumVForTF32Emu(self.states.b, self.states.lrvwTileB)
      #   then, checkA
      needAdjustment = checkVregOverflowTF32Emu(vgprIdx, numVgprsEmuA + numVgprsEmuB)
      if needAdjustment and self.states.lrvwTileA > 1 and not self.states.doFullPackCodePrefetch:
        numVgprsEmuA = adjustNumVForTF32Emu(self.states.a, self.states.lrvwTileA)
      # final adjustment
      # disable UseMFMAF32XEmulation and save 2 vregs
      if kernel["UseMFMAF32XEmulation"]:
        if checkVregOverflowTF32Emu(vgprIdx, numVgprsEmuA + numVgprsEmuB):
          # disable UseMFMAF32XEmulation and use Dot2 instead
          kernel["UseMFMAF32XEmulation"] = False
          kernel["UseDot2F32XEmulation"] = True

      # vreg allocation for UseMFMAF32XEmulation
      if kernel["UseMFMAF32XEmulation"]:
        vgprIdx = ((vgprIdx+1)//2)*2 #align 64 bit
        self.states.startVgprIdentityMatrix = vgprIdx
        vgprIdx+=2
      numVgprsEmu = numVgprsEmuA + numVgprsEmuB
      self.states.a.numVgprEmu = numVgprsEmuA
      self.states.b.numVgprEmu = numVgprsEmuB
      if numVgprsEmu > 0:
        #align 64 bit
        vgprIdx = ((vgprIdx+1)//2)*2
        self.states.a.startVgprCvt = vgprIdx
        vgprIdx += numVgprsEmuA # for vgpr 32XEmulation A
        self.states.b.startVgprCvt = vgprIdx
        vgprIdx += numVgprsEmuB # for vgpr 32XEmulation B

      if kernel["StreamK"] and self.isStreamKConstantsToVgprEnabled(kernel):
        numSKConsts = 5  # ItersPerTile, MagicNumberItersPerTile, MagicShiftItersPerTile, SKItersPerWG, StreamKIdx
        if kernel["StreamK"] >= 2:
          numSKConsts += 2  # skGrid, skTiles
        self.states.startVgprSKConsts = vgprIdx
        self.states.numVgprSKConsts = numSKConsts
        vgprIdx += numSKConsts

      # Per-tensor temp VGPRs for XF32 cvt+sub residual computation.
      # A and B each get a dedicated temp so that _interleavePackAB can
      # safely interleave their residual sequences without clobbering.
      if kernel["UseF32XEmulation"] and not kernel["UseMFMAF32XEmulation"] and not kernel["UseDot2F32XEmulation"]:
        self.states.a.tmpVgprCvtSub = vgprIdx
        vgprIdx += 1
        self.states.b.tmpVgprCvtSub = vgprIdx
        vgprIdx += 1
  
      if kernel["PrefetchGL2"]:
        vgprIdx = int((vgprIdx + 1) / 2) * 2
        self.states.a.startVgprGL2PrefetchAddr = vgprIdx
        vgprIdx += tensorParametersA["gl2nl"] * self.states.rpga
        self.states.b.startVgprGL2PrefetchAddr = vgprIdx
        vgprIdx += tensorParametersB["gl2nl"] * self.states.rpga
        if kernel["ProblemType"]["MXBlockA"]:
          self.states.mxsa.startVgprGL2PrefetchAddr = vgprIdx
          vgprIdx += tensorParametersA["MX"]["gl2nl"] * self.states.rpga
        if kernel["ProblemType"]["MXBlockB"]:
          self.states.mxsb.startVgprGL2PrefetchAddr = vgprIdx
          vgprIdx += tensorParametersB["MX"]["gl2nl"] * self.states.rpga      

      # TODO: Serial is always the first/last register in the pool so the store
      # code doesn't have to deal with fragmentation
      self.states.startVgprSerial = vgprIdx
      vgprIdx += 1 # for vgpr serial id

      self.states.totalVgprs = max(vgprIdx, self.states.c.numVgprValu)
      if self.states.totalVgprs < 0 or self.states.totalVgprs > self.states.regCaps["MaxVgpr"]:
        raise RuntimeError("Generating asm kernel error: total vgpr: %u not in [0, %u].\n" % (self.states.totalVgprs, self.states.regCaps["MaxVgpr"]))

      agprLimit = self.states.regCaps["PhysicalMaxVgpr"] - self.states.regCaps["MaxVgpr"]
      if self.states.totalAgprs > agprLimit:
        raise RuntimeError("Generating asm kernel error: total agpr: %u not in [0, %u].\n" % (self.states.totalAgprs, agprLimit) )
  
      # VGPR alloc marker


    def vgprAllocationImplSubtile():
      self.states.maxLimitAgprs   = self.states.regCaps["PhysicalMaxVgpr"] - self.states.regCaps["MaxVgpr"]
      if kernel["EnableMatrixInstruction"]:
        #jgolds bpeCinternal because we are allocating accumulation registers here
        self.states.c.numVgprValu = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.states.bpeCinternal)//self.states.bpr

      vgprIdx = 0
      self.states.startVgprSerial = vgprIdx;
      vgprIdx += 1
      #self.states.c.startVgprValu = vgprIdx;

      #vgprIdx += self.states.c.numVgprValu
      if kernel["StreamK"] and self.isStreamKConstantsToVgprEnabled(kernel):
        numSKConsts = 5
        if kernel["StreamK"] >= 2:
          numSKConsts += 2
        self.states.startVgprSKConsts = vgprIdx
        self.states.numVgprSKConsts = numSKConsts
        vgprIdx += numSKConsts

      self.states.totalVgprs = vgprIdx

      return



    # Dispatch to different VGPR allocation logic for subtile-based impl
    if kernel["UseSubtileImpl"]:
      vgprAllocationImplSubtile()
    else:
      vgprAllocationImplClassic()

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
    if kernel["ProblemType"]["MXBlockA"]:
      numSgprAddressMXSA = self.states.rpga
    if kernel["ProblemType"]["MXBlockB"]:
      numSgprAddressMXSB = self.states.rpga
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
    if kernel["ProblemType"]["MXBlockA"]:
      self.states.mxsa.numSgprStrides = len(kernel["ProblemType"]["IndexAssignmentsMXSA"])
    if kernel["ProblemType"]["MXBlockB"]:
      self.states.mxsb.numSgprStrides = len(kernel["ProblemType"]["IndexAssignmentsMXSB"])
    if not kernel["ProblemType"]["UseInitialStridesCD"]:
      self.states.e.numSgprStrides -= 1
      self.states.d.numSgprStrides -= 1
      self.states.c.numSgprStrides -= 1
    if not kernel["ProblemType"]["UseInitialStridesAB"]:
      self.states.a.numSgprStrides -= 1
      self.states.b.numSgprStrides -= 1
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numSgprStrides -= 1
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numSgprStrides -= 1
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
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numSgprGlobalReadIncs = 0
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numSgprGlobalReadIncs = 0
      self.states.m.numSgprGlobalReadIncs = 0
    else:
      self.states.a.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo
      self.states.b.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.mxsa.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.mxsb.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo
      self.states.m.numSgprGlobalReadIncs = kernel["ProblemType"]["NumIndicesSummation"] * self.states.rpgo
      # check for constSgprGlobalReadInc
      # use const version of GlobalReadInc for the following case
      # - StreamK (not GSU)
      # - no staggerUCode
      # - TLUA false for A, TLUB false for B
      # - numSgprGlobalReadIncs is 1
      if kernel["StreamK"] and (not self.states.staggerUCode):
        if kernel["ProblemType"]["TLUA"] == False:
          if self.states.a.numSgprGlobalReadIncs == 1:
            # use const GR Inc
            self.states.a.useConstSgprGlobalReadIncs = True
            # do not allocate GRInc sgpr
            self.states.a.numSgprGlobalReadIncs = 0
          if kernel["ProblemType"]["MXBlockA"]:
            if self.states.mxsa.numSgprGlobalReadIncs == 1:
              # use const GR Inc
              self.states.mxsa.useConstSgprGlobalReadIncs = True
              # do not allocate GRInc sgpr
              self.states.mxsa.numSgprGlobalReadIncs = 0
        if kernel["ProblemType"]["TLUB"] == False:
          if self.states.b.numSgprGlobalReadIncs == 1:
            # use const GR Inc
            self.states.b.useConstSgprGlobalReadIncs = True
            # do not allocate GRInc sgpr
            self.states.b.numSgprGlobalReadIncs = 0
          if kernel["ProblemType"]["MXBlockB"]:
            if self.states.mxsb.numSgprGlobalReadIncs == 1:
              # use const GR Inc
              self.states.mxsb.useConstSgprGlobalReadIncs = True
              # do not allocate GRInc sgpr
              self.states.mxsb.numSgprGlobalReadIncs = 0

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

    if kernel["enableTDMA"] or kernel["enableTDMB"]:
      self.defineSgpr("WaveIdx", 1)

    if kernel["Multicast"]:
      tdmA: bool = kernel["enableTDMA"]
      tdmB: bool = kernel["enableTDMB"]
      tdmM: bool = kernel["enableTDMMetadata"]
      if tdmA and tdmB and kernel["NumWaves"] > 1:
        self.defineSgpr("MulticastMask", 1)
      else:
        self.defineSgpr("MulticastMaskA", 1)
        self.defineSgpr("MulticastMaskB", 1)
      if tdmM:
        self.defineSgpr("MulticastMaskMetadata", 1)

    # SGPR above are user SGPR which are set by GPU hardware when the kernel is launched
    self.states.firstInitSgpr = self.sgprPool.size()

    # Commenting the conditional assignment of SGPR for ArgType
    # since the General Batched GEMM will also use this value like
    # Grouped GEMM but has to piggy back on Strided Batched GEMM logic.
    #if kernel["ProblemType"]["SupportUserArgs"]:
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

    # Whole-kernel scratch for StinkyTofu SwPrefetchInsertionPass (user SGPR; pool order);
    if rocisa.isSupportedByStinkyTofu(self.states.version) and bool(kernel.get("SwInstructionPrefetch", True)):
      self.defineSgpr("SwPrefetchScratch", 1)

    if self.debugConfig.debugKernel:
      self.defineSgpr("AddressDbg", self.states.numSgprAddressDbg)
      self.defineSgpr("DebugKernelItems", 1)

    # the sgprs overlap with wg ids
    # TODO: For subtileimpl, consider shadowInit param as well
    if self.states.doShadowInit and kernel["BufferStore"]:
      self.defineSgpr("SrdD", 4, 4)
      self.defineSgpr("SrdC", 4, 4)

    self.defineSgpr("NumWorkGroups0", 1)
    self.defineSgpr("NumWorkGroups1", 1)

    # Calculate numSgpr preload
    self.states.preloadGuard = []
    self.states.numSgprPreload = 0
    if kernel["PreloadKernArgs"]:
      # kernel argument buffer address needs 2 sgprs
      # Workgroup ID x, y, z need 3 sgprs
      self.states.numSgprPreload = self.states.archCaps["MaxSgprPreload"] - self.states.rpga - kernel["ProblemType"]["NumIndicesC"]

      # Safe guard for preload arguments
      while(1):
        tmpSgpr = self.sgprPool.checkOut(1, tag="preloadGuard_tmpSgpr", preventOverflow=False)
        if tmpSgpr >= self.states.archCaps["MaxSgprPreload"]:
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
    if kernel["ProblemType"]["MXBlockA"]:
      self.defineSgpr("AddressMXSA", numSgprAddressMXSA)
    self.defineSgpr("AddressB", numSgprAddressB)
    if kernel["ProblemType"]["MXBlockB"]:
      self.defineSgpr("AddressMXSB", numSgprAddressMXSB)
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
    if kernel["ProblemType"]["MXBlockA"]:
      self.defineSgpr("StridesMXSA", self.states.mxsa.numSgprStrides)
    self.defineSgpr("StridesB", self.states.b.numSgprStrides)
    if kernel["ProblemType"]["MXBlockB"]:
      self.defineSgpr("StridesMXSB", self.states.mxsb.numSgprStrides)
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

    self.defineSgpr("Alpha", numSgprAlpha, numSgprAlpha)
    self.states.numSgprAlpha = numSgprAlpha
    if kernel["ProblemType"]["UseBeta"]:
      self.defineSgpr("Beta", numSgprBeta, numSgprBeta)
      self.states.numSgprBeta = numSgprBeta

    if kernel["StreamK"] == 4:
      self.defineSgpr("ItersPerTile", 1)
      self.defineSgpr("TotalItems", 1)
      self.defineSgpr("skTiles", 1)
      self.defineSgpr("SKSplit", 1)
      self.defineSgpr("SKItersPerWI", 1)
      self.defineSgpr("skGrid", 1)
      self.states.numSgprStreamK += 6
    elif kernel["StreamK"]:
      # StreamK args
      self.defineSgpr("ItersPerTile", 1)
      self.defineSgpr("MagicNumberItersPerTile", 1)
      self.defineSgpr("MagicShiftItersPerTile", 1)
      self.defineSgpr("SKItersPerWG", 1)
      self.states.numSgprStreamK += 4
      if kernel["StreamK"] >= 2: # Two-tile SK
        self.defineSgpr("skGrid", 1)
        self.defineSgpr("skTiles", 1)
        self.states.numSgprStreamK += 2

    if not kernel["UseSubtileImpl"]:
      if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("LocalWriteAddrA", 1)
      if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("LocalWriteAddrB", 1)
      if kernel["ProblemType"]["Sparse"] and kernel["LocalWriteUseSgprMetadata"]:
        self.defineSgpr("LocalWriteAddrMetadata", 1)
      if kernel["ProblemType"]["MXBlockA"] and kernel["LocalWriteUseSgprMXSA"]:
          self.defineSgpr("LocalWriteAddrMXSA", 1)
      if kernel["ProblemType"]["MXBlockB"] and kernel["LocalWriteUseSgprMXSB"]:
          self.defineSgpr("LocalWriteAddrMXSB", 1)

    # Allocate registers to swap between lds buffers
    if self.states.useCommonSgprSwap and not kernel["UseSubtileImpl"]:
      self.defineSgpr("SwapCommon", 1)
    elif not kernel["UseSubtileImpl"] and (kernel["StoreSwapAddr"]):
      if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("SwapA", 1)
      if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("SwapB", 1)
      if kernel["ProblemType"]["MXBlockA"] and kernel["LocalWriteUseSgprMXSA"]:
          self.defineSgpr("SwapMXSA", 1)
      if kernel["ProblemType"]["MXBlockB"] and kernel["LocalWriteUseSgprMXSB"]:
          self.defineSgpr("SwapMXSB", 1)
      if kernel["ProblemType"]["Sparse"] and kernel["LocalWriteUseSgprMetadata"]:
        self.defineSgpr("SwapMetadata", 1)

    if ((kernel["GlobalSplitU"] == -1 or kernel["GlobalSplitU"] > 0) and (kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel" or kernel["AdaptiveGemmGSUA"] == 1)):
      self.defineSgpr("AddressTD", numSgprAddressD, align=2)
      self.states.numSgprAddressGSUSync += numSgprAddressD
      self.defineSgpr("Synchronizer", 2, align=2)
      self.states.numSgprAddressGSUSync += 2
      self.defineSgpr("GSUSync", 1)
      self.states.numSgprAddressGSUSync += 1

    if kernel["GlobalSplitU"] != 0 or kernel["AdaptiveGemmNTAB"] != 0:
      self.defineSgpr("GSU", 1)  # Can't move to the front because of the preload arguments

    # Collect SGPRs to allocate via the deferred interleaved loop.
    # Using a list allows allocation order to be controlled, minimising
    # alignment holes (e.g. keeping the pool on a 4-aligned boundary
    # immediately before any 4-aligned SrdWS allocation).
    requiredUnalignedSgprVar = []
    requiredAligned4SgprVar = []

    if kernel["StreamK"] == 4:
      requiredUnalignedSgprVar += [
        "StreamKIdx",
        "StreamKTileIdx",
        "StreamKPartialIdx",
        "StreamKLocalStart",
        "StreamKLocalEnd",
      ]
      if kernel["StreamKAtomic"] == 0:
        requiredAligned4SgprVar.append("SrdWS")
    elif kernel["StreamK"]:
      if not self.isStreamKConstantsToVgprEnabled(kernel):
        requiredUnalignedSgprVar.append("StreamKIdx")
      requiredUnalignedSgprVar += [
        "StreamKIter",
        "StreamKIterEnd",
        "StreamKLocalStart",
        "StreamKLocalEnd",
      ]
      if len(kernel["SpaceFillingAlgo"]):
        requiredUnalignedSgprVar.append("StreamKTileID")
      if kernel.get("PrefetchAcrossPersistent"):
        requiredUnalignedSgprVar.append("SkPrefetchPrimed")
        self.states.numSgprStreamK += 1
      if kernel["StreamKAtomic"] == 0:
        requiredAligned4SgprVar.append("SrdWS")

    if kernel["UseSubtileImpl"]:
      # DTL addressing SGPRs not needed when TDM handles global-to-LDS
      if not (kernel["enableTDMA"] and kernel["enableTDMB"]):
        requiredUnalignedSgprVar.append("LocalWriteBaseAddrA")
        requiredUnalignedSgprVar.append("LocalWriteBaseAddrB")
        if kernel["ProblemType"]["MXBlockA"]:
          requiredUnalignedSgprVar.append("LocalWriteBaseAddrMXSA")
        if kernel["ProblemType"]["MXBlockB"]:
          requiredUnalignedSgprVar.append("LocalWriteBaseAddrMXSB")
        requiredUnalignedSgprVar.append("SwapA")
        requiredUnalignedSgprVar.append("SwapB")
        if kernel["ProblemType"]["MXBlockA"]:
          requiredUnalignedSgprVar.append("SwapMXSA")
        if kernel["ProblemType"]["MXBlockB"]:
          requiredUnalignedSgprVar.append("SwapMXSB")
      if kernel["ProblemType"]["Sparse"] and kernel["LocalWriteUseSgprMetadata"]:
        requiredUnalignedSgprVar.append("SwapMetadata")

    # Actual allocation: prioritise 4-aligned SGPRs whenever the pool is
    # already on a 4-aligned boundary, otherwise consume unaligned ones.
    while len(requiredUnalignedSgprVar) or len(requiredAligned4SgprVar):
      if self.sgprPool.size() % 4 == 0 and len(requiredAligned4SgprVar):
        var = requiredAligned4SgprVar.pop()
        self.defineSgpr(var, 4, 4)
      elif len(requiredUnalignedSgprVar):
        var = requiredUnalignedSgprVar.pop()
        self.defineSgpr(var, 1)

    # These SGPRs aren't used right away, add them to sgpr pool temporarily
    if self.states.doShadowInit and kernel["BufferStore"]:
      self.addSgprVarToPool("SrdC")
    if kernel["StreamK"] and kernel["StreamKAtomic"] == 0:
      self.addSgprVarToPool("SrdWS")

    # gfx1250 frees the SK constant SGPRs later in moveStreamKConstantsToVgpr
    # after their values have been copied to VGPRs. Freeing them here would let
    # temp allocs clobber kernel arguments before they are copied.
    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    for key, _ in self.sgprs.items():
      self.states.nonPostLoopSgpr.append(key)
    # Manually remove some additional unused sgpr.
    # With PrefetchAcrossPersistent, loop counters must survive the
    # post-loop store phase so setupNewTile can use them in the
    # prefetch tail.
    keepLoopCounters = kernel["StreamK"] and kernel.get("PrefetchAcrossPersistent")
    if not keepLoopCounters:
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
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.nonPostLoopSgpr.remove("AddressMXSA")
        self.states.nonPostLoopSgpr.remove("StridesMXSA")
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.nonPostLoopSgpr.remove("AddressMXSB")
        self.states.nonPostLoopSgpr.remove("StridesMXSB")

    self.states.preloadScaleA = False
    self.states.preloadScaleB = False
    if kernel["ProblemType"]["UseScaleAB"] == "Scalar":
      if kernel["ProblemType"]["DataTypeA"].numRegisters() > kernel["ProblemType"]["MacDataTypeA"].numRegisters():
        self.states.preloadScaleA = True
      if kernel["ProblemType"]["DataTypeB"].numRegisters() > kernel["ProblemType"]["MacDataTypeB"].numRegisters():
        self.states.preloadScaleB = True

      for preloadScale, name in zip([self.states.preloadScaleA, self.states.preloadScaleB], ['A','B']):
        if preloadScale:
          self.defineSgpr("AddressScale%s"%name, 2, 2)
          self.defineSgpr("Scale%s"%name, numSgprAlpha, numSgprAlpha if numSgprAlpha > 1 else 2)


    self.states.numSgprToLoad = self.states.numSgprSizesFree + self.states.numSgprSizesSum + \
      numSgprAddressD + numSgprAddressC + numSgprAddressA + numSgprAddressB + numSgprAlpha + numSgprAddressMetadata + \
      (numSgprAddressMXSA if kernel["ProblemType"]["MXBlockA"] else 0) + \
      (numSgprAddressMXSB if kernel["ProblemType"]["MXBlockB"] else 0) + \
      (numSgprBeta if kernel["ProblemType"]["UseBeta"] else 0) + \
      self.states.d.numSgprStrides + self.states.c.numSgprStrides + self.states.a.numSgprStrides + self.states.b.numSgprStrides + self.states.m.numSgprStrides + \
      (self.states.mxsa.numSgprStrides if kernel["ProblemType"]["MXBlockA"] else 0) + \
      (self.states.mxsb.numSgprStrides if kernel["ProblemType"]["MXBlockB"] else 0) + \
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
      factorSubIterA = kernel["numSubTiles"]
      if kernel["UnrollMajorLDSA"] or kernel["enableLDSTrA"]:
        self.states.numReadsPerUnrollA = ceil(tensorParametersA["bpe"] * kernel["MIInputPerThreadA"] / int(tensorParametersA["localReadInstruction"].blockWidth * 4))
      else:
        self.states.numReadsPerUnrollA = kernel["MIInputPerThreadA"]
        if kernel["ForceUnrollSubIter"] and self.states.numReadsPerUnrollA > 1:
          self.states.numReadsPerUnrollA //= kernel["numSubTiles"]
          factorSubIterA = 1
      numA = kernel["InnerUnroll"]*(kernel["MIWaveTile"][0] * self.states.numReadsPerUnrollA) // tensorParametersA["localReadInstruction"].numOffsets
      if self.states.lrvwTileA > 1:
        numA = numA // kernel["VectorWidthA"]
      if kernel["ForceUnrollSubIter"]:
        numA = numA // factorSubIterA
      if kernel["ProblemType"]["MXBlockA"]:
        if kernel["UnrollMajorLDSMXSA"]:
          self.states.numReadsPerUnrollMXSA = ceil(kernel["MIInputPerThreadMXSA"] / int(tensorParametersA["MX"]["localReadInstruction"].blockWidth * 4))
        else:
          self.states.numReadsPerUnrollMXSA = kernel["MIInputPerThreadMXSA"]
        numMXSA = kernel["InnerUnroll"] * kernel["MIWaveTile"][0] // tensorParametersMXSA["localReadInstruction"].numOffsets
        if self.states.lrvwTileMXSA > 1:
          if self.states.asmCaps["HasWMMA_V3"]:
            mxUnit = kernel["MatrixInstK"] // kernel["ProblemType"]["MXBlockA"]
            numMXSA = numMXSA // int(tensorParametersA["MX"]["localReadInstruction"].blockWidth * 4 // mxUnit)
          else:
            numMXSA = numMXSA // kernel["VectorWidthA"]

      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        if kernel["UnrollMajorLDSMetadata"] or kernel["enableLDSTrMetadata"]:
          self.states.numReadsPerUnrollMetadata = ceil(tensorParametersM["bpe"] * kernel["MIInputPerThreadMetadata"] / int(tensorParametersM["localReadInstruction"].blockWidth * 4))
        else:
          self.states.numReadsPerUnrollMetadata = kernel["MIInputPerThreadMetadata"]
        tileM = kernel["MIWaveTile"][1] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTile"][0]
        numM = kernel["InnerUnroll"]*(tileM * self.states.numReadsPerUnrollMetadata) // tensorParametersM["localReadInstruction"].numOffsets
      if kernel["ForceUnrollSubIter"]:
        self.states.numReadsPerUnrollMetadata //= kernel["numSubTiles"]

      factorSubIterB = kernel["numSubTiles"]

      if kernel["UnrollMajorLDSB"] or kernel["enableLDSTrB"]:
        self.states.numReadsPerUnrollB = ceil(tensorParametersB["bpe"] * kernel["MIInputPerThreadB"] / int(tensorParametersB["localReadInstruction"].blockWidth * 4))
      else:
        self.states.numReadsPerUnrollB = kernel["MIInputPerThreadB"]
        if kernel["ForceUnrollSubIter"] and self.states.numReadsPerUnrollB > 1:
          self.states.numReadsPerUnrollB //= kernel["numSubTiles"]
          factorSubIterB = 1
      numB = kernel["InnerUnroll"]*(kernel["MIWaveTile"][1] * self.states.numReadsPerUnrollB) // tensorParametersB["localReadInstruction"].numOffsets
      if self.states.lrvwTileB > 1:
        numB = numB // kernel["VectorWidthB"]
      if kernel["ForceUnrollSubIter"]:
        numB = numB // factorSubIterB
      if kernel["ProblemType"]["MXBlockB"]:
        if kernel["UnrollMajorLDSMXSB"]:
          self.states.numReadsPerUnrollMXSB = ceil(kernel["MIInputPerThreadMXSB"] / int(tensorParametersB["MX"]["localReadInstruction"].blockWidth * 4))
        else:
          self.states.numReadsPerUnrollMXSB = kernel["MIInputPerThreadMXSB"]

        numMXSB = kernel["InnerUnroll"] * kernel["MIWaveTile"][1] // tensorParametersMXSB["localReadInstruction"].numOffsets
        if self.states.lrvwTileMXSB > 1:
          if self.states.asmCaps["HasWMMA_V3"]:
            mxUnit = kernel["MatrixInstK"] // kernel["ProblemType"]["MXBlockB"]
            numMXSB = numMXSB // int(tensorParametersB["MX"]["localReadInstruction"].blockWidth * 4 // mxUnit)
          else:
            numMXSB = numMXSB // kernel["VectorWidthB"]

      # wider localread has 2 mode
      # 1. using larger IU to coalesced localread, only half of local reads in 1 iteration
      # 2. using larger PLR to read more iterations, same number local reads in 1 iteration
      if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedA:
        numA //= self.states.numReadsIterCoalescedA
      if kernel["ProblemType"]["MXBlockA"]:
        if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedMXSA:
          numMXSA //= self.states.numReadsIterCoalescedMXSA
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedMetadata:
          numM //= self.states.numReadsIterCoalescedMetadata
      if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedB:
        numB //= self.states.numReadsIterCoalescedB
      if kernel["ProblemType"]["MXBlockB"]:
        if kernel["InnerUnroll"] >= self.states.numReadsIterCoalescedMXSB:
          numMXSB //= self.states.numReadsIterCoalescedMXSB

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
      if kernel["ProblemType"]["MXBlockA"]:
        self.states.numReadsPerIterMXSA = numMXSA
    if not kernel["DirectToVgprB"]:
      self.states.numReadsPerIterB = numB
      if kernel["ProblemType"]["MXBlockB"]:
        self.states.numReadsPerIterMXSB = numMXSB

    self.states.localReadDoCntA = 0
    if kernel["ProblemType"]["MXBlockA"]:
      self.states.localReadDoCntMXSA = 0
    self.states.localReadDoCntB = 0
    if kernel["ProblemType"]["MXBlockB"]:
      self.states.localReadDoCntMXSB = 0
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      self.states.numReadsPerIterMetadata = numM
      self.states.localReadDoCntMetadata  = 0

    if kernel["EnableMatrixInstruction"]:
      from rocisa.instruction import getMFMAIssueLatency, getSMFMAIssueLatency
      miInputType = getMiInputType(kernel)

      if kernel["ProblemType"]["Sparse"]:
        self.states.miLatency, miIssueLatency = getSMFMAIssueLatency(
            miInputType.toEnum(), kernel["MatrixInstM"], kernel["MatrixInstB"])
      else:
        self.states.miLatency, miIssueLatency = getMFMAIssueLatency(
            miInputType.toEnum(), kernel["MatrixInstM"], kernel["MatrixInstB"])
      # TODO: Avoid the logic which does not make sense.
      # For gfx950, we can't issue any VALU or DS instruction in next 4 cycles.
      # Changed the value based on this and also to mitigate some instruction scheduling issues.
      # Invalidate this adjustment if ExtraMiLatencyLeft is >= 0
      if not kernel["ProblemType"]["Sparse"] and kernel['ISA'] == IsaVersion(9,5,0) and miInputType.numBytes() == 2 and kernel["ExtraMiLatencyLeft"]== -1:
        self.states.miLatency = kernel["MatrixInstM"] // 2
        miIssueLatency = 2

      # give 1 quad-cycle buffer to prevend bubble from sync
      miLatencyBuffer = 1
      self.states.miLatencyLeft = max(self.states.miLatency - miLatencyBuffer - miIssueLatency,0)
      # add extra miLatencyLeft from parameter
      if kernel["ExtraMiLatencyLeft"] > 0:
        self.states.miLatencyLeft += kernel["ExtraMiLatencyLeft"]

      # Special dependency cases
      if kernel["ProblemType"]["ComputeDataType"].isDouble():
        if kernel["MatrixInstruction"] == [4, 4, 4, 4]:
          if kernel['ISA'] == IsaVersion(9,0,10):
            self.states.miDependency = 4

      # Next VALU instruction with source same as the previous XDL S/WMMA instruction's Matrix D (RAW)
      # Next VALU instruction with same vdst as the previous XDL S/WMMA instruction's Matrix D (WAW)
      # Next VALU instruction with same vdst as the previous XDL S/WMMA instruction's Matrix A/B/Index (WAR)
      self.states.miVALUInstrDataHazard = 0
      if self.states.version == (12,5,0):
        if kernel["ProblemType"]["Sparse"]:
          if (kernel["ProblemType"]["DataType"].isHalf() or \
            kernel["ProblemType"]["DataType"].isBFloat16() or \
            kernel["ProblemType"]["DataType"].is8bitFloat()):
            self.states.miVALUInstrDataHazard = 2
          elif kernel["ProblemType"]["DataType"].isInt8():
            self.states.miVALUInstrDataHazard = 4
        else:
          if (kernel["ProblemType"]["DataType"].isHalf() or \
            kernel["ProblemType"]["DataType"].isBFloat16() or \
            (kernel["ProblemType"]["DataType"].is8bitFloat() and kernel["MatrixInstK"] <= 64) or \
            kernel["ProblemType"]["DataType"].is6bitFloat() or \
            kernel["ProblemType"]["DataType"].isFloat4()):
            self.states.miVALUInstrDataHazard = 4
          elif (kernel["ProblemType"]["DataType"].isInt8() or \
            (kernel["ProblemType"]["DataType"].is8bitFloat() and kernel["MatrixInstK"] > 64)):
            self.states.miVALUInstrDataHazard = 8

      # num (GRInc) instruction per mfma
      minInst = kernel["MinGRIncPerMfma"]
      self.states.numInstPerMfma = max(roundUp(self.states.miLatencyLeft/2), minInst)

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

  def updateOccupancyFromScan(self, kernel, mkb) -> None:
    """Override in KernelWriterAssembly to rescan actual register usage after
    rocIsaPass optimizations and correct kernel["CUOccupancy"] + the kernel
    descriptor's next_free_vgpr for ArchAccUnifiedRegs ISAs."""
    pass

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

  def tpBpe(self, kernel, key, cM):
    if cM in ("Metadata", "MXSA", "MXSB"):
      return 1
    else:
      return kernel["ProblemType"][key].numBytes()

  ##############################################################################
  # Get Input Tensor Bpe
  ##############################################################################
  def tpBpe(self, kernel, key, cM):
    if cM in ("Metadata", "MXSA", "MXSB"):
      return 1
    else:
      return kernel["ProblemType"][key].numBytes()

  ##############################################################################
  # Get Params For Tensor A/B
  ##############################################################################
  def getTensorParameters(self, tP, kernel, itP, cM):
    tP["mirror"] = bool(kernel["ProblemType"]["MirrorDims%s" % (cM)])

    if ("A" in cM) or (kernel["ProblemType"]["Sparse"] == 1 and cM == "Metadata"): # A
      tP["tensorIdx"] = 0                                   # tensor index A=0, B=1
      tP["tileChar"] = self.states.tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) \
        else self.states.tileChar1                       # tile char I0 or J1
    elif ("B" in cM) or (kernel["ProblemType"]["Sparse"] ==2 and cM == "Metadata"): # B
      tP["tensorIdx"] = 1
      tP["tileChar"] = self.states.tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) \
        else self.states.tileChar1

    tP["isA"] = (cM == "A")                                      # is this tensor A
    tP["isB"] = (cM == "B")                                      # is this tensor B
    tP["isMXSA"] = (cM == "MXSA")                                # is this tensor MXSA
    tP["isMXSB"] = (cM == "MXSB")                                # is this tensor MXSB
    tP["isM"] = (cM == "Metadata")                               # is this tensor Metadata

    bpe = self.tpBpe(kernel, f"MacDataType{cM}", cM)
    bpetc = self.tpBpe(kernel, f"DataType{cM}", cM)
    bpeA = self.tpBpe(kernel, "DataTypeA", cM)

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
    tP["ntpl"] = kernel["NumTotalPackedLoads%s"%cM]
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
      tP["swizzlePackK"] = 16 // kernel["MIInputPerThread%s"%cM] // int(kernel["ProblemType"]["DataType%s"%cM].numBytes())
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
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lraAddressesInitFor3LDSBlk(self, kernel, tP, initSreg, initVreg):
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
  def removeStaggerAB(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  @abc.abstractmethod
  def calculateLoopNumIter(self, kernel, tPA, tPB, loopIdx, tailloopInNll=False, NLLindex=0):
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
  def openLoop(self, kernel, tPA, tPB, loopIdx, noLabelGen, beginLabelOnly, nta=0, ntb=0):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, tPA, tPB, loopIdx, \
                finalLoop, emitEndLabelOnly=False, oddLabel=False, \
                skipCondJumpCounter=-1, NLLlast=False, \
                nta=0, ntb=0):
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
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isNGLL=False, NLLindex=0, NLLnum=1, NLLindexLast=False):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel, tPA, tPB, prefetch, isOptNLL, isNGLL, isNotLast=False, remainPgr=0):
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
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP, prefetch=False, skipMetaSwap=False):
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
  def openPrefetchGlobalRead2orMore(self, kernel, idxPgr):
    return ""

  @abc.abstractmethod
  def closePrefetchGlobalRead2orMore(self, kernel, tensorParametersA, tensorParametersB, idxPgr):
    return ""

  ##############################################################################
  # PAP helpers
  ##############################################################################
  def isPrefetchAcrossPersistentEnabled(self, kernel):
    """Return True when PAP is enabled for this kernel."""
    return (kernel["StreamK"] == 3
            and kernel.get("PrefetchAcrossPersistent", 0)
            and not kernel.get("SuppressNoLoadLoop", False)
            and kernel["PrefetchGlobalRead"] >= 1
            and not kernel.get("UseCustomMainLoopSchedule", 0))

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
  def simdSpecDispatch(self, kernel, numCodePath, nta=0, ntb=0):
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
    if kernel["enableTDMA"] and kernel["enableTDMB"] and skipGlobalRead > -1:
      return tdmWait(self.states, kernel, tPA, tPB, skipGlobalRead, comment)
    return wait(self.states, kernel, tPA, tPB, skipGlobalRead, \
      skipLocalWrite, skipLocalRead, self.db["ConservativeWaitCnt"], comment, skipGlobalReadInst)

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def _syncThreads(self, kernel, comment="", skipForceWaitcnt0=False, memoryToken=None):
    if self.do["Sync"]:
      return syncThreads(kernel, self.states.archCaps, self.states.asmCaps, comment, skipForceWaitcnt0=skipForceWaitcnt0, memoryToken=memoryToken)
    return Module("SyncThreads (Empty)")

  ##############################################################################
  # PostMainLoopBarrierCheckAndReset
  # Rebuild SBarrier placement by memory-token read/write phase transitions.
  ##############################################################################
  def postMainLoopBarrierCheckAndReset(self, kernel, rootModule):
    """
    Rebuild SBarrier placement from dynamic memory-token usage:
      - Token count is discovered from instruction token API at runtime
      - tensor_load / ds_write => write access
      - ds_read => read access
      - Writing -> Read transition: insert barrier, state=Reading
      - Reading -> Write transition: insert barrier, state=Writing
    """
    numWaves = kernel["NumThreads"] // kernel["WavefrontSize"]
    stOptLevel = kernel.get("_StinkyTofuOptLevel", 0)
    scheduleIterAlg = kernel.get("_ScheduleIterAlg", self.states.scheduleIterAlg)
    if stOptLevel != 3 or scheduleIterAlg != 0:
      print2(f"[postMainLoopBarrierCheckAndReset] skip: _StinkyTofuOptLevel={stOptLevel}, _ScheduleIterAlg={scheduleIterAlg} (expect 3 and 0)")
      return
    if not kernel["enableTDMA"] or not kernel["enableTDMB"]:
      print2(f"[postMainLoopBarrierCheckAndReset] skip: enableTDMA={kernel['enableTDMA']}, enableTDMB={kernel['enableTDMB']} (both must be True)")
      return
    if numWaves == 1:
      print2(f"[postMainLoopBarrierCheckAndReset] skip: numWaves={numWaves} (must be > 1)")
      return

    removedCount = 0
    insertedCount = 0

    # Pass-1: remove existing barriers first.
    modulesToScan = [rootModule]
    while modulesToScan:
      currentModule = modulesToScan.pop()
      keptItems = []
      for item in currentModule.items():
        if isinstance(item, Module):
          modulesToScan.append(item)
          keptItems.append(item)
        elif isinstance(item, SBarrier):
          removedCount += 1
          continue
        else:
          keptItems.append(item)
      currentModule.setItems(keptItems)

    # Pass-2: insert barriers by token state transitions.
    tokenState = {}
    branchTokenStateSnapshot = {}

    def _isOptNllEndLabelName(labelName):
      return isinstance(labelName, str) and "OptNLL_End" in labelName

    def _classifyTokenAccess(inst: Instruction):
      # tensor_load / ds_write => write, ds_read => read
      if isinstance(inst, DSStoreInstruction):
        return "write"
      if isinstance(inst, TensorLoadToLds):
        return "write"
      if isinstance(inst, DSLoadInstruction):
        return "read"
      return None

    def _getTokenList(inst: Instruction):
      # This pass only relies on the existing mem-token API.
      if hasattr(inst, "getMemToken"):
        memTokenObj = inst.getMemToken()
        if memTokenObj is not None and hasattr(memTokenObj, "tokens"):
          return list(memTokenObj.tokens)
      return []

    def _accessPhase(access):
      return "writing" if access == "write" else "reading"

    def _conflicts(access, state):
      return (access == "read" and state == "writing") or \
             (access == "write" and state == "reading")

    def _isUnrollLoopBeginLabel(labelName):
      # The repeating unroll loop is delimited by a "LoopBegin<char>" label and
      # a backward branch that targets it. Tail-loop begin labels ("TailLoopBegin")
      # are excluded - they are handled by their own pass and are not the main
      # unroll loop we model the back-edge for.
      return isinstance(labelName, str) and "LoopBegin" in labelName and "TailLoopBegin" not in labelName

    def _detectLoopHeadInfo():
      # Detect the real loop span(s) from the back-edge, not from module names.
      # A module named "loopBody" also contains the odd/even-iter exit code and
      # the loop-end label that execute AFTER the back-branch, so its last token
      # access is not the loop tail. Instead, flatten leaves in program order,
      # find each backward branch to a "LoopBegin" label, and treat
      # [begin .. back-branch] as the loop body.
      #
      # Returns beginLabelName -> {token: [firstAccess, tailState]} where:
      #   firstAccess: access ("read"/"write") of the token's FIRST occurrence in
      #                the body (what the back-edge feeds into).
      #   tailState:   phase ("reading"/"writing") of the token's LAST occurrence
      #                in the body (the phase the back-edge carries out).
      flatLeaves = []
      def _flattenLeaves(mod: Module):
        for item in mod.items():
          if isinstance(item, Module):
            _flattenLeaves(item)
          else:
            flatLeaves.append(item)
      _flattenLeaves(rootModule)

      labelDefIndex = {}
      for idx, leaf in enumerate(flatLeaves):
        if hasattr(leaf, "getLabelName") and not isinstance(leaf, Instruction):
          name = leaf.getLabelName()
          labelDefIndex.setdefault(name, idx)

      # beginLabelName -> (beginIdx, backBranchIdx) using the widest back-edge span.
      loopSpanByLabel = {}
      for idx, leaf in enumerate(flatLeaves):
        target = getattr(leaf, "labelName", None)
        if target is None or not _isUnrollLoopBeginLabel(target):
          continue
        beginIdx = labelDefIndex.get(target, None)
        if beginIdx is None or beginIdx >= idx:
          continue  # forward branch, not a back-edge
        prev = loopSpanByLabel.get(target)
        if prev is None or idx > prev[1]:
          loopSpanByLabel[target] = (beginIdx, idx)

      headInfo = {}
      for beginName, (beginIdx, branchIdx) in loopSpanByLabel.items():
        info = {}
        for k in range(beginIdx, branchIdx + 1):
          leaf = flatLeaves[k]
          if not isinstance(leaf, Instruction):
            continue
          access = _classifyTokenAccess(leaf)
          tokens = _getTokenList(leaf)
          if access is None or not tokens:
            continue
          phase = _accessPhase(access)
          for token in tokens:
            if token not in info:
              info[token] = [access, phase]  # [firstAccess, tailState]
            else:
              info[token][1] = phase          # update tail to the last access
        headInfo[beginName] = info
      return headInfo

    # Back-edge modeling is only needed for PrefetchGlobalRead < 2. With
    # PrefetchGlobalRead >= 2 the pipelined prologue pre-stages the next
    # iteration's LDS data, so the steady-state phase is already established and
    # a plain single linear pass is correct - leaving loopHeadInfo empty makes
    # _rewriteModuleInOrder degrade to exactly that linear pass.
    loopEntryOverride = {}
    loopPendingTokens = set()
    loopHeadInfo = _detectLoopHeadInfo() if kernel["PrefetchGlobalRead"] < 2 else {}

    def _rewriteModuleInOrder(mod: Module):
      nonlocal insertedCount
      rewrittenItems = []
      for item in mod.items():
        if hasattr(item, "getLabelName") and not isinstance(item, Instruction):
          labelName = item.getLabelName()
          if _isOptNllEndLabelName(labelName) and labelName in branchTokenStateSnapshot:
            # recover token state from the snapshot
            tokenState.clear()
            tokenState.update(deepcopy(branchTokenStateSnapshot[labelName]))
          if labelName in loopHeadInfo:
            # Entering the unroll loop. The loop-head barrier is driven purely by
            # the back-edge (loop-tail) state, so a steady-state iteration only
            # gets a barrier when the carried phase truly conflicts with the
            # first access. The first iteration's pre-loop conflict, if any and
            # not already covered by a back-edge barrier, is satisfied ONCE by a
            # barrier hoisted into the prologue (emitted right before the loop
            # label) instead of one that re-fires every iteration.
            prologueBarrierTokens = []
            for token, (firstAccess, tailState) in loopHeadInfo[labelName].items():
              preState = tokenState.get(token, "standby")
              loopEntryOverride[token] = tailState
              loopPendingTokens.add(token)
              if _conflicts(firstAccess, preState) and not _conflicts(firstAccess, tailState):
                prologueBarrierTokens.append(token)
            if prologueBarrierTokens:
              uniqueTokens = sorted(set(prologueBarrierTokens))
              syncComments = ", ".join([f"sync LDS{token}" for token in uniqueTokens])
              barrier = SBarrier(comment=f"auto token transition barrier (loop prologue), {syncComments}")
              barrier.setMemToken(MemTokenData(uniqueTokens))
              rewrittenItems.append(barrier)
              insertedCount += 1
          rewrittenItems.append(item)
          continue

        if isinstance(item, Module):
          _rewriteModuleInOrder(item)
          rewrittenItems.append(item)
          continue
        if not isinstance(item, Instruction):
          rewrittenItems.append(item)
          continue

        branchLabelName = getattr(item, "labelName", None)
        if _isOptNllEndLabelName(branchLabelName) and branchLabelName not in branchTokenStateSnapshot:
          # Save token state at the first branch to OptNLL_End.
          branchTokenStateSnapshot[branchLabelName] = deepcopy(tokenState)
        if branchLabelName in loopHeadInfo:
          # Reached the loop back-branch: drop any stale loop-entry overrides.
          loopEntryOverride.clear()
          loopPendingTokens.clear()

        access = _classifyTokenAccess(item)
        tokens = _getTokenList(item)
        if access is None and tokens:
          print2(f"[postMainLoopBarrierCheckAndReset] WARNING: instruction {type(item).__name__} has tokens {tokens} but no classified access — barrier may be missing")
        if access is None or not tokens:
          rewrittenItems.append(item)
          continue

        barrierTokens = []
        for token in tokens:
          if token in loopPendingTokens:
            # First access of this token inside the loop body: evaluate it
            # against the back-edge (loop-tail) state.
            state = loopEntryOverride.get(token, tokenState.get(token, "standby"))
            loopPendingTokens.discard(token)
          else:
            state = tokenState.get(token, "standby")
          if _conflicts(access, state):
            barrierTokens.append(token)

        if barrierTokens:
          uniqueTokens = sorted(set(barrierTokens))
          syncComments = ", ".join([f"sync LDS{token}" for token in uniqueTokens])
          barrier = SBarrier(comment=f"auto token transition barrier, {syncComments}")
          barrier.setMemToken(MemTokenData(uniqueTokens))
          rewrittenItems.append(barrier)
          insertedCount += 1

        nextState = _accessPhase(access)
        for token in tokens:
          tokenState[token] = nextState

        rewrittenItems.append(item)

      mod.setItems(rewrittenItems)

    _rewriteModuleInOrder(rootModule)
    print2(f"[postMainLoopBarrierCheckAndReset] removed {removedCount} barriers, inserted {insertedCount} barriers")
    return

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
    if not kernel["UseSubtileImpl"]:
      (error, kb) = self.kernelBody(kernel, tensorParametersA, tensorParametersB)
    else:
      (error, kb) = self.kernelBodySubtile(kernel, tensorParametersA, tensorParametersB)

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
        if currentInstLength - count + 1 >= self.states.asmCaps["ShortBranchMaxLength"]:
          with self.allocTmpSgpr(3, tag="updateBranchPlaceHolder_tmpSgprInfo") as tmpSgprInfo:
              _placeholder.add(self.longBranchScc0(_target, 1, tmpSgprInfo))
        else:
          _placeholder.add(SCBranchSCC0(labelName=_target.getLabelName()))
      elif _operation == "SCBranchSCC1":
        if currentInstLength - count + 1 >= self.states.asmCaps["ShortBranchMaxLength"]:
          with self.allocTmpSgpr(3, tag="updateBranchPlaceHolder_tmpSgprInfo2") as tmpSgprInfo:
              _placeholder.add(self.longBranchScc1(_target, 1, tmpSgprInfo))
        else:
          _placeholder.add(SCBranchSCC1(labelName=_target.getLabelName()))
      elif _operation == "SCBranchVCCNZ":
        if currentInstLength - count + 1 >= self.states.asmCaps["ShortBranchMaxLength"]:
          with self.allocTmpSgpr(3, tag="updateBranchPlaceHolder_tmpSgprInfo3") as tmpSgprInfo:
              _placeholder.add(self.longBranchVccnz(_target, 1, tmpSgprInfo))
        else:
          _placeholder.add(SCBranchVCCNZ(labelName=_target.getLabelName()))
      elif _operation == "SBranch":
        if currentInstLength - count + 1 >= self.states.asmCaps["ShortBranchMaxLength"]:
          with self.allocTmpSgpr(3, tag="updateBranchPlaceHolder_tmpSgprInfo4") as tmpSgprInfo:
            _placeholder.add(SLongBranchPositive(_target, tmpSgprInfo))
        else:
          _placeholder.add(SBranch(labelName=_target.getLabelName()))
      currentInstLength += countInstruction(_placeholder)

  @property
  def isa(self):
    return self.states.version

  def initTDMDescriptor(self, kernel, tP) -> Module:
    assert False, "Should be overrided"

  def initTDMDescriptorWaveSeparated(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def tdmGlobalOffset(self, kernel, tP) -> Module:
    assert False, "Should be overrided"

  def tdmGlobalOffsetWaveSeparated(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def tdmApplyStreamKOffsetWaveSeparated(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def papResetTDMDescriptorForTailWaveSeparated(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def papTdmUpdateDescriptor(self, kernel, tPA, tPB, preservePapBank=True) -> Module:
    assert False, "Should be overrided"

  def papTdmSaveLdsBank(self, kernel) -> Module:
    assert False, "Should be overrided"

  def papTdmRestoreLdsBank(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def papDtlSaveLdsBank(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def papDtlRestoreLdsBank(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def papTdmShiftTailLdsBank(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def tdmIncrementAB(self, kernel, tP, loopIdx=None, prefetchIndex=0) -> Module:
    assert False, "Should be overrided"

  def tdmIncrementABWaveSperated(self, kernel, tPA, tPB, loopIdx=None, prefetchIndex=0) -> Module:
    assert False, "Should be overrided"

  def tdmSetupIncrementWaveSeparated(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"
  
  @abc.abstractmethod
  def gl2PrefetchInit(self, kernel, tPA, tPB):
    return ""
  
  @abc.abstractmethod
  def gl2PrefetchCalcAddr(self, kernel, tPA, tPB) -> Module:
    return ""
  
  @abc.abstractmethod
  def gl2PrefetchIssueLoad(self, kernel, tPA, tPB) -> Module:
    return ""
  
  @abc.abstractmethod
  def gl2PrefetchIncrementAddr(self, kernel, tPA, tPB) -> Module:
    return ""
  

  def resetTDMDescriptorForTail(self, kernel, tP) -> Module:
    assert False, "Should be overrided"

  def resetTDMDescriptorForTailWaveSeparated(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

  def graIncrementMask(self, kernel, tPA, tPB) -> Module:
    assert False, "Should be overrided"

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
