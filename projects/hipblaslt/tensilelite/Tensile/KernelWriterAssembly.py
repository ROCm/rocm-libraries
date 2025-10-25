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


from rocisa import countInstruction, countGlobalRead, countSMemLoad, findInstCount
from rocisa.asmpass import getActFuncModuleName, getActFuncBranchModuleName
from rocisa.code import KernelBody, Label, Macro, Module, RegSet, SrdUpperValue, \
                        StructuredModule, TextBlock, ValueEndif, ValueIf, ValueSet, SignatureBase
from rocisa.container import DSModifiers, SDWAModifiers, VOP3PModifiers, \
                      MUBUFModifiers, SMEMModifiers, EXEC, VCC, RegisterContainer, \
                      DPPModifiers, vgpr, sgpr, accvgpr, mgpr, ContinuousRegister, \
                      HWRegContainer, GLOBALModifiers
from rocisa.instruction import SGetPositivePCOffset, SLongBranchPositive, SCLongBranchScc0, SCLongBranchScc1, SCLongBranchVccnz, \
                        SMulInt64to32, VCvtBF16toFP32
from rocisa.functions import vectorStaticDivide, vectorStaticRemainder, vectorUInt32CeilDivideAndRemainder, \
                        vectorStaticDivideAndRemainder, scalarStaticDivideAndRemainder, scalarStaticCeilDivide, \
                        scalarStaticRemainder, scalarUInt32DivideAndRemainder, sMagicDiv, vectorStaticMultiply, \
                        vectorStaticMultiplyAdd, scalarStaticMultiply64, BranchIfZero, BranchIfNotZero, DSInit, \
                        ArgumentLoader
from rocisa.enum import InstType, SelectBit
from rocisa.macro import MacroVMagicDiv, PseudoRandomGenerator
from . import CUSTOM_KERNEL_PATH
from rocisa.instruction import BranchInstruction, BufferLoadB128, BufferLoadB32, \
  BufferLoadB64, BufferLoadD16B16, BufferLoadD16HIB16, BufferLoadD16HIU8, \
  BufferLoadD16U8, BufferStoreB128, BufferStoreB16, BufferStoreB32, BufferStoreB64, \
  BufferStoreB8, BufferStoreD16HIB16, CommonInstruction, DSBPermuteB32, DSLoadB128, \
  DSLoadB16, DSLoadB32, DSLoadB64, DSLoadU16, DSStoreB128, DSStoreB16, DSStoreB32, \
  DSStoreB64, DSStoreB8, DSStoreInstruction, FlatLoadB128, FlatLoadB32, FlatLoadB64, \
  FlatLoadD16B16, FlatLoadD16HIB16, FlatStoreB128, FlatStoreB32, FlatStoreB64, \
  FlatStoreD16B16, FlatStoreD16HIB16, MFMAInstruction, MUBUFReadInstruction, \
  MacroInstruction, SAShiftRightI32, SAbsI32, SAddCU32, SAddI32, SAddU32, SAndB32, \
  SAndB64, SAndN2B32, SAtomicDec, SBarrier, SBfmB32, SBitcmp1B32, SBranch, SCBranchSCC0, \
  SCBranchSCC1, SCBranchVCCNZ, SCBranchVCCZ, SCMovB32, SCSelectB32, SCmpEQI32, \
  SCmpEQU32, SCmpEQU64, SCmpGeI32, SCmpGeU32, SCmpGtI32, SCmpGtU32, SCmpKEQU32, \
  SCmpKGeU32, SCmpKGtU32, SCmpKLGU32, SCmpLeI32, SCmpLeU32, SCmpLgU32, SCmpLtU32, SCmpLtI32, \
  SEndpgm, SFf1B32, SGetRegB32, SFlbitI32B32, SLShiftLeft2AddU32, SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, \
  SLShiftRightB64, SLoadB32, SLoadB64, SMFMAInstruction, SMemLoadInstruction, SMaxU32, SMinI32, \
  SMinU32, SMovB32, SMovB64, SMulHIU32, SMulI32, SNop, SOrB32, SOrSaveExecB32, \
  SOrSaveExecB64, SSExtI16toI32, SSetPCB64, SSetRegIMM32B32, SSetPrior, SSubBU32, SSubI32, SSubU32, \
  SWaitCnt, SWaitAlu, SXorB32, VAShiftRightI32, VAccvgprReadB32, VAccvgprWrite, VAccvgprWriteB32, \
  VAdd3U32, VAddCCOU32, VAddCOU32, VAddF32, VAddF64, VAddLShiftLeftU32, VAddU32, VAndB32, \
  VBfeU32, VCmpEQI32, VCmpEQU32, VCmpGEI32, VCmpGEU32, VCmpGtU32, VCmpLeI32, VCmpLtI32, \
  VCmpLtU32, VCmpUF32, VCmpXGeU32, VCmpXLtU32, VCmpXLtU64, VCndMaskB32, VCvtF16toF32, \
  VCvtF32toF16, VCvtFP8toF32, VCvtInstruction, VCvtPkF32toBF16, VCvtPkF32toBF8, \
  VCvtPkF32toFP8, VCvtPkFP8toF32, VCvtSRF32toBF8, VCvtSRF32toFP8, VCvtScaleFP8toF16, \
  VCvtScalePkF16toBF8, VCvtScalePkF16toFP8, VCvtScalePkFP8toF16, VLShiftLeftB32, \
  VLShiftLeftB64, VLShiftRightB32, VMadU32U24, VMaxF32, VMinI32, VMovB32, VMovB64, VMulF32, \
  VMulHIU32, VMulLOU32, VMulPKF32S, VMulU32U24, VNotB32, VOrB32, VPackF16toB32, \
  VPrngB32, VReadfirstlaneB32, VSubF32, VSubI32, VSubU32, VXorB32, GlobalLoadTR8B64, GlobalLoadTR16B128

from .Component import Component
from .KernelWriterModules import *
from .SolutionStructs import isPackedIndex
from .AsmStoreState import StoreState, VectorDataTypes
from .Activation import ActivationType
from .CustomKernels import isCustomKernelConfig
from .Common import roundUp, log2, ceilDivide
from Tensile.Common import print2, printExit, printWarning, INDEX_CHARS, DebugConfig, DataDirection
from Tensile.Common.DataType import DataType
from Tensile.Common.RegisterPool import RegisterPool, allocTmpGpr, allocTmpGprList
from .Components.WorkGroupMappingAlgos import DefaultWGM, wgmXCC, SpaceFillingCurveWalk

from Tensile.KernelWriter import KernelWriter
from Tensile.SolutionStructs.Naming import getKernelFileBase
from Tensile.Toolchain.Component import Assembler

from math import ceil, log
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, NamedTuple, Optional, Tuple, Union

import os
@dataclass
class TailOptParams:
  idx:                 int         = 0
  behavior:            str         = ""
  jumpLabel:           Label       = None
  kLabelsList:         List[Label] = None
  tmpVgpr:             int         = None
  kSgpr:               int         = None
  vgprSlot:            List[int]   = None
  loadNum:             int         = 0
  periodParam:         List[int]   = None
  preDirectToLdsLoads: int         = 0
  firstLoop:           int         = 0
  finalLoop:           int         = 0

@dataclass
class GprInfo:
  addrVgpr:            List[str]   = field(default_factory=list)
  offset:              List[str]   = field(default_factory=list)

@dataclass
class GlobalReadGprRecord:
  a: GprInfo                       = field(default_factory=GprInfo)
  b: GprInfo                       = field(default_factory=GprInfo)

################################################################################
# Assembly Kernel
################################################################################
class KernelWriterAssembly(KernelWriter):

  ##############################################################################
  # Init
  ##############################################################################
  def __init__(
      self,
      assembler: Assembler,
      debugConfig: DebugConfig,
    ):
    super(KernelWriterAssembly, self).__init__(assembler, debugConfig)
    self.globalread_gpr_record = GlobalReadGprRecord()

  def _getCustomKernelSource(self, kernel, CustomKernelDirectory):
    kernelName = getKernelFileBase(self.debugConfig.splitGSU, kernel)
    with open(os.path.join(CustomKernelDirectory, (kernelName + ".s"))) as f:
      rocmVersion = self.assembler.rocm_version
      if not (rocmVersion.major >= 6 and rocmVersion.patch >= 32650):
        code = []
        for line in f.readlines():
          if "amdhsa_user_sgpr_kernarg_preload" not in line:
            code.append(line)
        code = "".join(code)
      else:
        code = f.read()

    self.tPA = {}
    self.tPB = {}
    self.states.kernel = kernel
    self.states.language = "ASM"
    self.states.version = kernel["ISA"]

    return code


  def getSourceFileString(self, kernel) -> Tuple[int, str]:
    assert kernel["KernelLanguage"] == "Assembly"
    # Skip if .o files will have already been built for this file
    if kernel.duplicate:
      self.language = "ASM"
      return (0, "") # should this be an non zero number

    try:
      code = self._getCustomKernelSource(kernel, CUSTOM_KERNEL_PATH) if isCustomKernelConfig(kernel) else self._getKernelSource(kernel)
      errcode = 0
    except RuntimeError as e:
      printWarning(f"Failed to generate assembly source code for {kernel}: {e}")
      code = ""
      errcode = -2
    return (errcode, code)

  def getSgprOccupancy(self, sgprs):
    return self.states.regCaps["PhysicalMaxSgpr"]//sgprs

  def getVgprOccupancy(self, numThreads, vgprs, doubleVgpr=False):
    multiplier = int(ceil(max(numThreads, 256) / 256.0)) # example: wg=512 multiplier=2, 1024=4
    maxOccupancy = self.consts.maxOccupancy//multiplier

    vgprAllocateAligned = 4    if not doubleVgpr else 8
    totalVgprs = self.states.regCaps["MaxVgpr"] if not doubleVgpr else self.states.regCaps["MaxVgpr"]*2
    vgprsAligned = int(ceil(vgprs/vgprAllocateAligned))*vgprAllocateAligned
    vgprsAligned *= multiplier

    if   vgprsAligned > totalVgprs:  return 0
    elif vgprsAligned < 1:           return maxOccupancy
    occupancy = min(totalVgprs//vgprsAligned, maxOccupancy)

    #print("vgprs = ", vgprs, "vgprsAligned = ", vgprsAligned, "unifiedVgprRegs = " ,unifiedVgprRegs, "Occupancy = ", occupancy)

    return occupancy

  ########################################
  def getOccupancy(self, numThreads, vgprs, sgprs, ldsSize, accvgprs=0, doubleVgpr=False):

    deviceLdsSize = self.states.archCaps["DeviceLDS"]
    ldsLimitedOccupancy = self.getLdsLimitedOccupancy(deviceLdsSize, ldsSize)

    if not doubleVgpr:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, vgprs,          doubleVgpr)
      accvgprLimitedOccupancy = self.getVgprOccupancy(numThreads, accvgprs,       doubleVgpr)
    else:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, ceil(vgprs/8)*8+accvgprs, doubleVgpr)
      accvgprLimitedOccupancy = vgprLimitedOccupancy
    sgprLimitedOccupancy = self.getSgprOccupancy(sgprs)

    return min(ldsLimitedOccupancy, vgprLimitedOccupancy, accvgprLimitedOccupancy, sgprLimitedOccupancy)

  # TODO: also consider sgpr
  def getMaxRegsForOccupancy(self, numThreads, vgprs, sgprs, ldsSize, accvgprs=0, doubleVgpr=False):
    lastVgprs = vgprs
    considerAccVgprs = 0       if not doubleVgpr else accvgprs
    totalVgprs = self.states.regCaps["MaxVgpr"] if not doubleVgpr else self.states.regCaps["MaxVgpr"]*2

    initOccupancy = self.getOccupancy(numThreads, vgprs, sgprs, ldsSize, accvgprs, doubleVgpr)
    if initOccupancy == 0: return lastVgprs, 1

    def getVgpr(vgpr, doubleVgpr):
      return vgpr if not doubleVgpr else ceil(vgpr/8)*8
    while (getVgpr(vgprs, doubleVgpr) + considerAccVgprs) < totalVgprs and vgprs < self.states.regCaps["MaxVgpr"]:
      vgprs += 1
      if self.getVgprOccupancy(numThreads, getVgpr(vgprs, doubleVgpr) + considerAccVgprs, doubleVgpr) >= initOccupancy:
        lastVgprs = vgprs
        next
      else:
        break

    return lastVgprs, initOccupancy

  @staticmethod
  def getLdsLimitedOccupancy(deviceLdsSize, ldsSize):
    # As ldsSize gets large, rounding might push us slightly higher than deviceLdsSize.
    # Clamp at deviceLdsSize
    ldsSize = min(ldsSize + 255, deviceLdsSize) & 0xffffff00 # 256-byte granularity

    ldsLimitedOccupancy = deviceLdsSize//ldsSize
    return ldsLimitedOccupancy

  @staticmethod
  def getLdsSize(kernel):
    ldsSize = kernel["LdsNumBytes"]
    return ldsSize

  ########################################
  def sizeRef(self, idx):
    """
    Return sgpr() or const with the specified size
    See above definitions for how these are mapped to Free or Sum sizes
    based on the problem definition.
    """
    idxChar= INDEX_CHARS[idx]
    return sgpr("Size%s"%idxChar)

  def loopChar(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return INDEX_CHARS[loopDim]

  def loopSizeRef(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return self.sizeRef(loopDim)

  def loopCounterName(self, kernel, loopIdx):
    return "LoopCounter%s"%(self.loopChar(kernel, loopIdx))

  def loopCounter(self, kernel, loopIdx):
    """
    Return loopCounter for loopIdx wrapped in "SGPR" syntax
    loop idx is 0...unrollIdx
    """
    return sgpr(self.loopCounterName(kernel,loopIdx))

  def checkLastIter(self, kernel, comment="at last iteration?"):
    """ Return last iteration of unroll loop. """
    return SCmpEQU32(src0=self.loopCounter(kernel, self.states.unrollIdx), src1=0, comment=comment)

  def isConstUnitStride(self, stride):
      if isinstance(stride, RegisterContainer):
        return False
      return stride.startswith("const")

  ########################################
  def strideRef(self, tc, dim):
    """
    Return sgpr with specified stride or define starting with const if constant.
    dim is index 0...max indices and is in global index space.
    """
    problemType = self.states.kernel["ProblemType"]
    if tc in ['A','B', "Metadata"]:
      if not problemType["UseInitialStridesAB"] and \
          dim == problemType["IndexAssignments%s"%tc][0]:
        return ("constStride%s%s"%(tc,self.states.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.states.indexChars[dim]))
    elif tc in ['E','D','C','TD']:
      if not problemType["UseInitialStridesCD"] and dim == 0:
        return ("constStride%s%s"%(tc,self.states.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.states.indexChars[dim]))
    else:
      raise ValueError("unexpected tensorChar='%s' in stride function"%tc)

  ##############################################################################
  # Find Memory Instruction For Width and Stride
  ##############################################################################
  def findMemoryInstructionForWidthStride(self, width, strides, combine, \
      instructions):
    for i in range(0, len(instructions)):
      instruction = instructions[i]
      numAddresses = instruction.numAddresses
      numOffsets = instruction.numOffsets
      offsetMultiplier = instruction.offsetMultiplier
      blockWidth = instruction.blockWidth
      valid = True
      if width < blockWidth:
        valid = False
      if combine: # try to combine ops
        if numOffsets > 0: # if inst combines using offsets
          for stride in strides:
            if stride % offsetMultiplier != 0:
              valid = False
      else: # don't try to combine ops
        if numOffsets > 1 or numAddresses > 1:
          valid = False
      if valid:
        return i
      else:
        continue

    printWarning("Could not find valid memory instruction for width=%f" % width)
    return len(instructions)

  ##############################################################################
  # Select Memory Instruction
  # when selecting instruction, need to support stride in both dims
  ##############################################################################
  def selectMemoryInstruction(self,
      operation, # ReadGlobal, WriteLocal, ReadLocal
      width, # num registers 1 chunk
      write2, # Para, Perp, None
      para2, # NumLoadsPara >= 2
      perp2, # NumLoadsPerp >= 2
      strides ):

    #instructions = self.memoryArchitecture[operation]
    instructions = self.memoryInstructions[operation]
    # try to combine
    if (write2 == "Coalesced" and para2) \
        or (write2 == "Perpendicular" and perp2):
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, True, instructions)
    # don't or can't combine
    else:
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, False, instructions)

    if instructionIdx < len(instructions): # found
      return instructionIdx
    else:
      raise RuntimeError("Could not find valid memory instruction for operation=%s, width=%f, kernel=%s" %(operation, width, self.states.kernelName))

  def initGlobalReadMemoryInstruction(self, instructions, tP, bpr):
    # globalRead instruction; no flat_load2_*
    globalReadWidth = float(tP["nrcv"]*tP["bpeGR"])/bpr
    globalRead2Coalesced = tP["nrc"] > 1
    globalRead2Perpendicular = tP["nrp"] > 1
    globalReadInstructionIdx = self.selectMemoryInstruction("GlobalRead", globalReadWidth, \
                                False, \
                                globalRead2Coalesced, globalRead2Perpendicular, [] )

    tP["globalReadInstruction"] = instructions["GlobalRead"][globalReadInstructionIdx]

  def initLocalWriteMemoryInstruction(self, instructions, kernel, tP, bpr):
    ########################################
    # localWrite instruction
    # for local, tile->para, unroll->perp
    # wtc = writeTileDimComponents
    localWriteWidth = tP["nwcv"]*tP["bpeDS"]//bpr
    if localWriteWidth < 1:
      localWriteWidth = (1.0*tP["nwcv"]*tP["bpeDS"])/bpr
    localWrite2Coalesced = tP["nrc"]>1 or tP["wtc"]
    localWrite2Perpendicular = tP["nrp"]>1
    # localWrite stride tile
    if tP["tlu"]:
      if tP["wtc"]:
        localWriteStrideTile = 1
      else:
        localWriteStrideTile = kernel[tP["lsc"]]
    else:
      localWriteStrideTile = kernel[tP["lsp"]]
    localWriteStrideTile = localWriteStrideTile*tP["bpeDS"]//bpr
    # localWrite stride unroll
    if tP["tlu"]:
      localWriteStrideUnroll = kernel[tP["lsc"]]*kernel[tP["mt"]]
    else:
      if tP["wtc"]:
        localWriteStrideUnroll = 1*kernel[tP["mt"]]
      else:
        localWriteStrideUnroll = kernel[tP["lsc"]]*kernel[tP["mt"]]
    localWriteStrideUnroll = \
        (localWriteStrideUnroll*tP["bpeDS"])//bpr
    localWriteInstructionIdx = self.selectMemoryInstruction("LocalWrite", localWriteWidth, \
                                False, \
                                localWrite2Coalesced, localWrite2Perpendicular,
                                [localWriteStrideTile, localWriteStrideUnroll] )

    tP["localWrite2Coalesced"]     = localWrite2Coalesced
    tP["localWrite2Perpendicular"] = localWrite2Perpendicular
    tP["localWriteStrideTile"]     = localWriteStrideTile
    tP["localWriteStrideUnroll"]   = localWriteStrideUnroll
    tP["localWriteInstruction"]    = instructions["LocalWrite"][localWriteInstructionIdx]

  def initLocalReadMemoryInstruction(self, instructions, kernel, tP, bpr):

    tChar = "A" if tP["isA"] else "B" if tP["isB"] else "Metadata"
    if kernel["UnrollMajorLDS%s"%tChar]:
      if tChar == "A":
        localReadWidth = (self.states.lrvwUnrollA * tP["bpeDS"]) / bpr
      if tChar == "B":
        localReadWidth = (self.states.lrvwUnrollB * tP["bpeDS"]) / bpr
      if tChar == "Metadata":
        localReadWidth = (self.states.lrvwUnrollMetadata * tP["bpeDS"]) / bpr
    else:
      if tChar == "A":
        localReadWidth = (self.states.lrvwTileA * tP["bpeDS"]) / bpr
      if tChar == "B":
        localReadWidth = (self.states.lrvwTileB * tP["bpeDS"]) / bpr
      if tChar == "Metadata":
        localReadWidth = (self.states.lrvwTileMetadata * tP["bpeDS"]) / bpr

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    localReadStrideCoalesced = kernel[tP["tt"]] * tP["bpeDS"] // bpr
    localRead2Coalesced = False
    tP["enableLDSTr"] = False
    if tChar != "Metadata":
        tP["enableLDSTr"] = kernel["enableLDSTr%s"%tChar]
    length = len(instructions["LocalRead"])

    if tP["enableLDSTr"]:
        localReadInstructionIdx = length - 1
    else:
        localReadInstructionIdx = self.selectMemoryInstruction("LocalRead", localReadWidth, \
                                   False, \
                                   localRead2Coalesced, localRead2Perpendicular,
                                   [localReadStrideCoalesced] )
    tP["localRead2Coalesced"]      = localRead2Coalesced
    tP["localRead2Perpendicular"]  = localRead2Perpendicular
    tP["localReadStrideCoalesced"] = localReadStrideCoalesced
    tP["localReadInstruction"]     = instructions["LocalRead"][localReadInstructionIdx]

  def allocTmpSgpr(self, num: int, alignment=None, tag=None):
    def overflowListener(e):
      self.states.overflowedResources = 2
      if self.db["AssertOnSgprOverflow"]:
        raise e

    tmpSgpr = allocTmpGpr(self.sgprPool, num, self.states.regCaps["MaxSgpr"], alignment, tag, overflowListener)
    return tmpSgpr

  def allocTmpSgprList(self, nums: List[int], alignments: Optional[List[int]]=None, tag=None):
    def overflowListener(e):
      self.states.overflowedResources = 2
      if self.db["AssertOnSgprOverflow"]:
        raise e

    tmpSgpr = allocTmpGprList(self.sgprPool, nums, self.states.regCaps["MaxSgpr"], alignments, tag, overflowListener)
    return tmpSgpr

  def defineMultiVgprIndex(self, names: List[str], numVgprs: List[int], align=1):
    assert(len(names) == len(numVgprs))
    vgprIdxVec = self.vgprPool.checkOutMulti(numVgprs, align, tags=names)
    return vgprIdxVec

  def defineSgprIdx(self, name, numSgprs, align=1):
    if numSgprs == 0: return

    sgprIdx = self.sgprPool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=False)
    #self.sgprIdx = roundUpToNearestMultiple(self.sgprIdx,align)
    #print (name, "->", self.sgprIdx, "+", numSgprs)
    self.sgprs[name] = sgprIdx

    return sgprIdx

  def defineSgpr(self, name, numSgprs, align=1):
    # check if previous define sgprs are being used..
    for s in self.states.freeSgprVarPool:
      self.setSgprToInUseState(s)
    ret = RegSet("s", "sgpr"+name, self.defineSgprIdx(name, numSgprs, align))
    for s in self.states.freeSgprVarPool:
      self.setSgprToFreeState(s)
    return ret

  def defineMultiSgprIndex(self, names: List[str], numSgprs: List[int], align=1):
    assert(len(names) == len(numSgprs))

    sgprIdxVec = self.sgprPool.checkOutMulti(numSgprs, align, tags=names)
    #self.sgprIdx = roundUpToNearestMultiple(self.sgprIdx,align)
    #print (name, "->", self.sgprIdx, "+", numSgprs)
    for idx, name in enumerate(names):
      self.sgprs[name] = sgprIdxVec[idx]

    return sgprIdxVec

  def setSgprToInUseState(self, name):
    self.sgprPool.removeFromCheckOut(self.sgprs[name])
    return RegSet("s", "sgpr"+name, self.sgprs[name])

  def undefineSgpr(self, name):
    self.sgprPool.checkIn(self.sgprs[name])
    # undefine a sgpr string twice will cause compiler error.
    # User must not add the UNDEF code module except it is the last one.
    return ValueSet(name="sgpr"+name, value="UNDEF", format = -1)

  def setSgprToFreeState(self, name):
    self.sgprPool.addFromCheckOut(self.sgprs[name])
    # undefine a sgpr string twice will cause compiler error.
    # Must call setSgprToInUseState again before calling setSgprToFreeState.
    return ValueSet(name="sgpr"+name, value="UNDEF", format = -1)

  def addSgprVarToPool(self, name):
    if name not in self.sgprs.keys() or (name in self.states.freeSgprVarPool):
      return
    self.states.freeSgprVarPool.add(name)
    ret = self.setSgprToFreeState(name)
    return ret

  def removeSgprVarFromPool(self, name):
    if name not in self.sgprs.keys() or (name not in self.states.freeSgprVarPool):
      return
    self.states.freeSgprVarPool.remove(name)
    ret = self.setSgprToInUseState(name)
    return ret

  def removeGRSrdVariableSgprsFromPool(self, kernel):
    module = Module("RemoveGRSrdSgprsFromPool")

    if kernel["BufferLoad"]:
       # resource descriptor (SRD) A and B, must be aligned on 4-SGPR boundary
      self.removeSgprVarFromPool("SrdA")
      self.removeSgprVarFromPool("SrdB")

    if self.states.use64bShadowLimit:
      self.removeSgprVarFromPool("ShadowLimitA")
      self.removeSgprVarFromPool("ShadowLimitB")

    self.removeSgprVarFromPool("WrapUA")
    self.removeSgprVarFromPool("WrapUB")

    return module

  def removeGROffsetsVariableSgprsFromPool(self, kernel):
    module = Module("RemoveGROffsetSgprsFromPool")

    self.removeSgprVarFromPool("GlobalReadIncsA")
    self.removeSgprVarFromPool("GlobalReadIncsB")

    return module

  def defineVariableSgprs(self, kernel):
    module = Module("DefineVariableSgpr")
    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals

    if kernel["BufferLoad"]:
       # resource descriptor (SRD) A and B, must be aligned on 4-SGPR boundary
      module.add(self.defineSgpr("SrdA", 4, 4))
      module.add(self.defineSgpr("SrdB", 4, 4))
      self.addSgprVarToPool("SrdA")
      self.addSgprVarToPool("SrdB")
      if kernel["ProblemType"]["Sparse"]:
        module.add(self.defineSgpr("SrdMetadata", 4, 4))

    if self.states.use64bShadowLimit:
      module.add(self.defineSgpr("ShadowLimitA", 2, 2))
      module.add(self.defineSgpr("ShadowLimitB", 2, 2))
      self.addSgprVarToPool("ShadowLimitA")
      self.addSgprVarToPool("ShadowLimitB")
      if kernel["ProblemType"]["Sparse"]:
        module.add(self.defineSgpr("ShadowLimitMetadata", 2, 2))

    module.add(self.defineSgpr("StaggerUIter", 1))  # stagger loop iterations, used for various iter counts in the code
    isDTVAorB = (kernel["DirectToVgprA"] != kernel["DirectToVgprB"]) #  only one of them is enabled
    if kernel["PrefetchGlobalRead"] == 2 and isDTVAorB:
      # PGR2 + DTVA or B (only 1 side), need separate StaggerUIter for DTV load
      module.add(self.defineSgpr("StaggerUIterDTV", 1))  # stagger loop iterations, used for various iter counts in the code
    module.add(self.defineSgpr("WrapUA", 2))  # Bytes to add to SrdA to reset address from N-1 iter to AddressA
    module.add(self.defineSgpr("WrapUB", 2))  # Bytes to add to SrdB to reset address from N-1 iter to AddressB
    if kernel["ProblemType"]["Sparse"]:
      module.add(self.defineSgpr("WrapUMetadata", 2))  # Bytes to add to SrdMetadata to reset address from N-1 iter to AddressMetadata
    self.addSgprVarToPool("WrapUA")
    self.addSgprVarToPool("WrapUB")


    module.add(self.defineSgpr("GlobalReadIncsA", self.states.a.numSgprGlobalReadIncs))
    module.add(self.defineSgpr("GlobalReadIncsB", self.states.b.numSgprGlobalReadIncs))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.add(self.defineSgpr("GlobalReadIncsMetadata", self.states.m.numSgprGlobalReadIncs))
    self.addSgprVarToPool("GlobalReadIncsA")
    self.addSgprVarToPool("GlobalReadIncsB")


    needPackK16  = False
    needPackK8Lw = False
    needPackK8Hi = False

    if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
      if self.states.lrvwTileA > 1 or self.states.lrvwTileB > 1:
        needPackK16 = True
      if self.states.lrvwTileMetadata > 1:
        needPackK8Lw = True
      if self.states.lrvwTileMetadata > 2:
        needPackK8Hi = True
    elif kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat():
      if self.states.lrvwTileA > 1 or self.states.lrvwTileB > 1 or self.states.lrvwTileMetadata > 1:
        needPackK8Lw = True
      if self.states.lrvwTileA > 2 or self.states.lrvwTileB > 2 or self.states.lrvwTileMetadata > 2:
        needPackK8Hi = True

    # need extra sgpr for packing metadata
    tPackM = "M" if needPackK16 and needPackK8Lw else ""

    if needPackK16:
        module.add(self.defineSgpr("PackKForV0", 1))
        module.add(self.defineSgpr("PackKForV1", 1))
        if kernel["StreamK"] != 0:
          self.states.nonPostLoopSgpr.append("PackKForV0")
          self.states.nonPostLoopSgpr.append("PackKForV1")
    if needPackK8Lw:
      module.add(self.defineSgpr("PackKFor%sV0"%tPackM, 1))
      module.add(self.defineSgpr("PackKFor%sV1"%tPackM, 1))
      if kernel["StreamK"] != 0 and not needPackK16 :
        self.states.nonPostLoopSgpr.append("PackKForV0")
        self.states.nonPostLoopSgpr.append("PackKForV1")
    if needPackK8Hi:
      module.add(self.defineSgpr("PackKFor%sV2"%tPackM, 1))
      module.add(self.defineSgpr("PackKFor%sV3"%tPackM, 1))
      if kernel["StreamK"] != 0 and not needPackK16:
        self.states.nonPostLoopSgpr.append("PackKForV2")
        self.states.nonPostLoopSgpr.append("PackKForV3")

    if kernel["ProblemType"]["StochasticRounding"]:
      module.add(self.defineSgpr("RNDSeed", 1))

    if kernel["_UseSgprForGRO"]:
      needFirstSgprOffset = kernel["DirectToLdsA"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.states.a.numVgprGlobalReadOffsets if needFirstSgprOffset else (self.states.a.numVgprGlobalReadOffsets-1)
      if numberOfSgpr > 0:
        module.add(self.defineSgpr("ScalarGlobalReadOffsetA", numberOfSgpr))

      needFirstSgprOffset = kernel["DirectToLdsB"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.states.b.numVgprGlobalReadOffsets if needFirstSgprOffset else (self.states.b.numVgprGlobalReadOffsets-1)
      if numberOfSgpr > 0:
        module.add(self.defineSgpr("ScalarGlobalReadOffsetB", numberOfSgpr))

      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        needFirstSgprOffset = kernel["DirectToLdsMetadata"] and kernel["UseInstOffsetForGRO"]
        numberOfSgpr = self.states.m.numVgprGlobalReadOffsets if needFirstSgprOffset else (self.states.m.numVgprGlobalReadOffsets-1)
        if numberOfSgpr > 0:
          module.add(self.defineSgpr("ScalarGlobalReadOffsetMetadata", numberOfSgpr))

    # debug flag to allocate dummy / unused sgpr
    # useful when comparing code that adds new kernel arguments to see what
    # was actually changed
    numDummySgpr= 0
    for i in range(numDummySgpr):
      module.add(self.defineSgpr("DummySgpr%d"%i, 1))

    if self.sgprPool.size() > self.states.regCaps["MaxSgpr"]:
      print ("warning: Number of defined SGPRS (%d) overflowed max SGPRS (%d)." \
               % (self.sgprPool.size(), self.states.regCaps["MaxSgpr"]))

    # End of define sgprs
    #------------------------

    #########################################################
    # Below calculates the number of sgprs needed not in epilogue
    #########################################################
    self.states.numStoreSgprNames2 = []
    self.states.numStoreSgprNameSizes2 = []
    storeSgprLoad2 = 0

    if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      storeSgprLoad2 += self.states.rpga
      self.states.numStoreSgprNames2.append("AddressTD")
      self.states.numStoreSgprNameSizes2.append(self.states.rpga)

    if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      storeSgprLoad2 += self.states.rpga
      self.states.numStoreSgprNames2.append("Synchronizer")
      self.states.numStoreSgprNameSizes2.append(self.states.rpga)

    self.states.numStoreSgprToLoad2 = storeSgprLoad2

    return module

  ##############################################################################
  def functionSignature(self) -> SignatureBase:
    """
    Function Signature
    called after rest of code
    """
    signatureClass = Component.Signature.find(self)
    signature = signatureClass(self)
    return signature

  def macroAndSet(self, kernel, tPA, tPB) -> Module:
    module = Module("MacroNSet")
    module.add(MacroVMagicDiv(kernel["MagicDivAlg"]))

    tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]
    ########################################
    # VGPR Macros
    ########################################
    module.addComment2("VGPR Assignments")
    module.addComment0("ValuC range: [%u-%u), %s"%(self.states.c.startVgprValu, self.states.c.startVgprValu+self.states.c.numVgprValu, \
                           "serializedStore enabled" if self.states.serializedStore else ""))
    module.add(RegSet("v", "vgprValuC", self.states.c.startVgprValu))

    module.addComment0("ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx")
    # PLR index: from X0 to X<LoopIters-1> (at most) -> VGPRs will be duplicated LoopIters times (at most)
    # eg, if LoopIters = 4, there would be at most 4*VGPRs
    PLR = self.states.numVgprBuffer
    numBi = PLR
    ri = 0
    moduleVgprMacro = Module("VALU/G2L Vgpr Macro")
    moduleVgprMacroValuA = Module("VALUA Vgpr Macro")
    moduleVgprMacroValuB = Module("VALUB Vgpr Macro")
    moduleVgprMacroValuAPack = Module("VALUA Pack Vgpr Macro")
    moduleVgprMacroValuBPack = Module("VALUB Pack Vgpr Macro")
    moduleVgprMacroValuM = Module("VALUMetadata Vgpr Macro")
    moduleVgprMacroValuMPack = Module("VALUMetadata Pack Vgpr Macro")
    moduleVgprMacroG2LA = Module("G2LA Vgpr Macro")
    moduleVgprMacroG2LB = Module("G2LB Vgpr Macro")
    module.add(RegSet("v", "vgprBase", self.states.startVgpr))
    if self.states.a.numVgprValu > 0: # Do not generate vgprValuA if numVgprValuA is 0
      numBiFactor = numBi
      if kernel["DirectToVgprA"] and (self.states.packDTVA or self.states.convDTVA):
        # DirectToVgpr case, we need LoopIters * 2 buffers
        numBiFactor = kernel["LoopIters"] * 2
      if self.states.lrvwTileA > 1:
        moduleVgprMacro.add(RegSet("v", "vgprValuA_X0_I0_BASE", "vgprBase", self.states.a.startVgprValu - self.states.startVgpr))
        for bi in range(0,numBiFactor): # buffer indices
          for iui in range(0, kernel["InnerUnroll"]):
            moduleVgprMacroValuA.add(RegSet("v", "vgprValuA_X%u_I%u"%(bi,iui), "vgprValuA_X0_I0_BASE", ri))
            ri += self.states.a.numVgprValuPerBlock
          if tPA["bpe"] < 4 and not kernel["UnrollMajorLDSA"] and not (kernel["UsePLRPack"] and self.states.numItersPLR):
            ri = 0
        ri = 0
        if tPA["bpe"] < 4 and not kernel["UnrollMajorLDSA"]:
          moduleVgprMacro.add(RegSet("v", "vgprValuA_X0_I0_D0_PACK", "vgprBase", self.states.a.startVgprValuPack - self.states.startVgpr))
          for bi in range(0,numBiFactor): # buffer indices
            for iui in range(0, kernel["InnerUnroll"]):
              for data in range(0,kernel["MIInputPerThreadA"]):
                moduleVgprMacroValuAPack.add(RegSet("v", "vgprValuA_X%u_I%u_D%u"%(bi,iui,data),"vgprValuA_X0_I0_D0_PACK", ri))
                ri += ceil(kernel["VectorWidthA"] * tPA["bpe"] / self.states.bpr) * kernel["MIWaveTileA"] // kernel["VectorWidthA"]
            if (kernel["UsePLRPack"] and self.states.numItersPLR):
              ri = 0
      else:
        moduleVgprMacro.add(RegSet("v", "vgprValuA_X0_I0_BASE", "vgprBase", self.states.a.startVgprValu - self.states.startVgpr))
        for bi in range(0,numBiFactor): # buffer indices
          for iui in range(0, kernel["InnerUnroll"]):
            moduleVgprMacroValuA.add(RegSet("v", "vgprValuA_X%u_I%u"%(bi,iui), "vgprValuA_X0_I0_BASE", ri))
            if kernel["UseDirect32XEmulation"]:
              moduleVgprMacroValuA.add(RegSet("v", "vgprValuA_T%u_I%u"%(bi,iui), self.states.startVgprCvt, ri // 2))
            ri += self.states.a.numVgprValuPerBlock
        ri = 0
        if tPA["bpe"] < 4 and not kernel["UnrollMajorLDSA"]:
          moduleVgprMacro.add(RegSet("v", "vgprValuA_X0_I0_D0_PACK", "vgprBase", self.states.a.startVgprValuPack - self.states.startVgpr))
          for data in range(1,int(self.states.bpr/tPA["bpeDS"])):
            for bi in range(0,numBiFactor): # buffer indices
              if bi % self.states.numVgprBufferPackA == 0:
                ri = (data-1) * kernel["InnerUnroll"] * self.states.numVgprBufferPackA * self.states.a.numVgprValuPerBlock
              for iui in range(0, kernel["InnerUnroll"]):
                moduleVgprMacroValuAPack.add(RegSet("v", "vgprValuA_X%u_I%u_D%u"%(bi,iui,data),"vgprValuA_X0_I0_D0_PACK", ri))
                ri += self.states.a.numVgprValuPerBlock

    ri = 0
    if self.states.b.numVgprValu > 0: # Do not generate vgprValuB if numVgprValuB is 0
      numBiFactor = numBi
      if kernel["DirectToVgprB"] and (self.states.packDTVB or self.states.convDTVB):
        # DirectToVgpr case, we need LoopIters * 2 buffers
        numBiFactor = kernel["LoopIters"] * 2
      if self.states.lrvwTileB > 1:
        moduleVgprMacro.add(RegSet("v", "vgprValuB_X0_I0_BASE", "vgprBase", self.states.b.startVgprValu - self.states.startVgpr))
        for bi in range(0,numBiFactor): # buffer indices
          for iui in range(0, kernel["InnerUnroll"]):
            moduleVgprMacroValuB.add(RegSet("v", "vgprValuB_X%u_I%u"%(bi,iui), "vgprValuB_X0_I0_BASE", ri))
            ri += self.states.b.numVgprValuPerBlock
          if (tPB["bpe"] < 4 and not kernel["UnrollMajorLDSB"]) and not (kernel["UsePLRPack"] and self.states.numItersPLR):
            ri = 0
        ri = 0
        if tPB["bpe"] < 4 and not kernel["UnrollMajorLDSB"]:
          moduleVgprMacro.add(RegSet("v", "vgprValuB_X0_I0_D0_PACK", "vgprBase", self.states.b.startVgprValuPack - self.states.startVgpr))
          for bi in range(0,numBiFactor): # buffer indices
            for iui in range(0, kernel["InnerUnroll"]):
              for data in range(0,kernel["MIInputPerThreadB"]):
                moduleVgprMacroValuBPack.add(RegSet("v", "vgprValuB_X%u_I%u_D%u"%(bi,iui,data), "vgprValuB_X0_I0_D0_PACK", ri))
                ri += ceil(kernel["VectorWidthB"] * tPB["bpe"] / self.states.bpr) * kernel["MIWaveTileB"] // kernel["VectorWidthB"]
            if (kernel["UsePLRPack"] and self.states.numItersPLR):
              ri = 0
      else:
        moduleVgprMacro.add(RegSet("v", "vgprValuB_X0_I0_BASE", "vgprBase", self.states.b.startVgprValu - self.states.startVgpr))
        for bi in range(0,numBiFactor): # buffer indices
          for iui in range(0, kernel["InnerUnroll"]):
            moduleVgprMacroValuB.add(RegSet("v", "vgprValuB_X%u_I%u"%(bi,iui), "vgprValuB_X0_I0_BASE", ri))
            if kernel["UseDirect32XEmulation"]:
              moduleVgprMacroValuB.add(RegSet("v", "vgprValuB_T%u_I%u"%(bi,iui), self.states.startVgprCvt + (self.states.b.startVgprValu - self.states.startVgpr) // 2, ri // 2))
            ri += self.states.b.numVgprValuPerBlock
        ri = 0
        if tPB["bpe"] < 4 and not kernel["UnrollMajorLDSB"]:
          moduleVgprMacro.add(RegSet("v", "vgprValuB_X0_I0_D0_PACK", "vgprBase", self.states.b.startVgprValuPack - self.states.startVgpr))
          for data in range(1,int(self.states.bpr/tPB["bpeDS"])):
            for bi in range(0,numBiFactor): # buffer indices
              if bi % self.states.numVgprBufferPackB == 0:
                ri = (data-1) * kernel["InnerUnroll"] * self.states.numVgprBufferPackB * self.states.b.numVgprValuPerBlock
              for iui in range(0, kernel["InnerUnroll"]):
                moduleVgprMacroValuBPack.add(RegSet("v", "vgprValuB_X%u_I%u_D%u"%(bi,iui,data), "vgprValuB_X0_I0_D0_PACK", ri))
                ri += self.states.b.numVgprValuPerBlock

    if kernel["ConvertAfterDS"]:
      cvtTemp = max(self.states.a.startVgprValuCvtTemp, self.states.b.startVgprValuCvtTemp)
      if (cvtTemp != -1):
         moduleVgprMacro.add(RegSet("v", "vgprCvtTemp", "vgprBase", cvtTemp - self.states.startVgpr))

    if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
      moduleVgprMacro.add(RegSet("v", "vgprValuSum", "vgprBase", self.states.bias.startVgprValu - self.states.startVgpr))

    if kernel["ProblemType"]["Sparse"]:
      if kernel["DirectToVgprSparseMetadata"]:
        moduleVgprMacro.add(RegSet("v", "vgprValuMetadata_X0_I0_BASE", "vgprBase", self.states.m.startVgprValu - self.states.startVgpr))
        moduleVgprMacroValuM.add(RegSet("v", "vgprValuMetadata", "vgprValuMetadata_X0_I0_BASE", 0))
      else:
        ri = 0
        if self.states.m.numVgprValu > 0: # Do not generate vgprValu if numVgprValu is 0
          if self.states.lrvwTileMetadata > 1:
            moduleVgprMacro.add(RegSet("v", "vgprValuMetadata_X0_I0_BASE", "vgprBase", self.states.m.startVgprValu - self.states.startVgpr))
            for bi in range(0,PLR): # buffer indices
              for iui in range(0, kernel["InnerUnroll"]):
                moduleVgprMacroValuM.add(RegSet("v", "vgprValuMetadata_X%u_I%u"%(bi,iui), "vgprValuMetadata_X0_I0_BASE", ri))
                ri += self.states.m.numVgprValuPerBlock
              if not kernel["UnrollMajorLDSMetadata"]:
                ri = 0
            ri = 0
            if not kernel["UnrollMajorLDSMetadata"]:
              miWaveTile = kernel["MIWaveTileB"] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTileA"]
              moduleVgprMacro.add(RegSet("v", "vgprValuMetadata_X0_I0_D_PACK", "vgprBase", self.states.m.startVgprValuPack - self.states.startVgpr))
              for data in range(0,kernel["MIInputPerThreadMetadata"]):
                for bi in range(0,PLR): # buffer indices
                  for iui in range(0, kernel["InnerUnroll"]):
                    moduleVgprMacroValuMPack.add(RegSet("v", "vgprValuMetadata_X%u_I%u_D%u"%(bi,iui,data), "vgprValuMetadata_X0_I0_D_PACK", ri))
                    ri += ceil(kernel["VectorWidthMetadata"] * tPM["bpe"] / self.states.bpr) * miWaveTile // kernel["VectorWidthMetadata"]
          else:
            moduleVgprMacro.add(RegSet("v", "vgprValuMetadata_X0_I0_BASE", "vgprBase", self.states.m.startVgprValu - self.states.startVgpr))
            for bi in range(0,PLR): # buffer indices
              for iui in range(0, kernel["InnerUnroll"]):
                moduleVgprMacroValuM.add(RegSet("v", "vgprValuMetadata_X%u_I%u"%(bi,iui), "vgprValuMetadata_X0_I0_BASE", ri))
                ri += self.states.m.numVgprValuPerBlock

    if not kernel["LocalWriteUseSgprA"] and self.states.a.numVgprLocalWriteAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteAddrA", \
          self.states.a.startVgprLocalWriteAddr))
      if self.states.a.numVgprLocalWriteAddr > 1:
        module.add(RegSet("v", "vgprLocalWriteAddrOverhangA", \
            self.states.a.startVgprLocalWriteAddr+1))
    if not kernel["LocalWriteUseSgprB"] and self.states.b.numVgprLocalWriteAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteAddrB", \
          self.states.b.startVgprLocalWriteAddr))
      if self.states.b.numVgprLocalWriteAddr > 1:
        module.add(RegSet("v", "vgprLocalWriteAddrOverhangB", \
            self.states.b.startVgprLocalWriteAddr+1))
    if self.states.m.numVgprLocalWriteAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteAddrMetadata", \
          self.states.m.startVgprLocalWriteAddr))
      if self.states.m.numVgprLocalWriteAddr > 1:
        module.add(RegSet("v", "vgprLocalWriteAddrOverhangMetadata", \
            self.states.m.startVgprLocalWriteAddr+1))
    if kernel["BufferLoad"]:
      module.add(RegSet("v", "vgprGlobalReadOffsetA", \
          self.startVgprGlobalReadOffsetA))
      module.add(RegSet("v", "vgprGlobalReadOffsetB", \
          self.startVgprGlobalReadOffsetB))
      if kernel["ProblemType"]["Sparse"]:
        module.add(RegSet("v", "vgprGlobalReadOffsetMetadata", \
            self.startVgprGlobalReadOffsetMetadata))
    else:
      module.add(RegSet("v", "vgprGlobalReadAddrA", \
          self.startVgprGlobalReadAddressesA))
      module.add(RegSet("v", "vgprGlobalReadAddrB", \
          self.startVgprGlobalReadAddressesB))

    if self.states.a.startVgprG2L is not None:
      moduleVgprMacro.add(RegSet("v", "vgprG2LA_BASE", "vgprBase", self.states.a.startVgprG2L - self.states.startVgpr))
    if self.states.b.startVgprG2L is not None:
      moduleVgprMacro.add(RegSet("v", "vgprG2LB_BASE", "vgprBase", self.states.b.startVgprG2L - self.states.startVgpr))
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      moduleVgprMacroG2LA.add(RegSet("v", "vgprG2LA", "vgprG2LA_BASE", 0))
      if kernel["DirectToVgprA"]:
        # additional definition G2LA2 for swapping register sets
        moduleVgprMacroG2LA.add(RegSet("v", "vgprG2LA2", "vgprG2LA_BASE", self.states.a.numVgprG2LAllocated//2))
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      moduleVgprMacroG2LB.add(RegSet("v", "vgprG2LB", "vgprG2LB_BASE", 0))
      if kernel["DirectToVgprB"]:
        # additional definition G2LB2 for swapping register sets
        moduleVgprMacroG2LB.add(RegSet("v", "vgprG2LB2", "vgprG2LB_BASE", self.states.b.numVgprG2LAllocated//2))
    if kernel["UnrollLoopSwapGlobalReadOrder"] and not kernel["DirectToLdsA"] and not kernel["DirectToLdsB"]:
      if kernel["ULSGRODoubleG2L"] == 0:
        moduleVgprMacroG2LA.add(RegSet("v", "vgprG2LB2", "vgprG2LA_BASE", 0))
        moduleVgprMacroG2LA.add(RegSet("v", "vgprG2LA2", "vgprG2LA_BASE", self.states.b.numVgprG2LAllocated))
      else:
        moduleVgprMacroG2LA.add(RegSet("v", "vgprG2LA2", "vgprG2LA_BASE", self.states.a.numVgprG2LAllocated))
        moduleVgprMacroG2LB.add(RegSet("v", "vgprG2LB2", "vgprG2LB_BASE", self.states.b.numVgprG2LAllocated))

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      moduleVgprMacro.add(RegSet("v", "vgprG2LMetadata", "vgprBase", self.states.m.startVgprG2L - self.states.startVgpr))

    if ((tPA["bpe"] < 4 and not kernel["UnrollMajorLDSA"]) or                                              \
        (tPB["bpe"] < 4 and not kernel["UnrollMajorLDSB"]) or                                              \
        (not kernel["UnrollMajorLDSMetadata"] and kernel["MIInputPerThreadMetadata"] == 4))                \
        and (kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()):
      moduleVgprMacro.add(RegSet("v", "vgprPackTemp", "vgprBase", self.states.a.startVgprValuPackTemp - self.states.startVgpr))

    if self.states.globalReadIncsUseVgpr:
      module.add(RegSet("v", "vgprGlobalReadIncsA", \
          self.startVgprGlobalReadIncsA))
      module.add(RegSet("v", "vgprGlobalReadIncsB", \
          self.startVgprGlobalReadIncsB))
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        module.add(RegSet("v", "vgprGlobalReadIncsMetadata", \
            self.startVgprGlobalReadIncsMetadata))
    if self.states.a.numVgprLocalReadAddr > 0:
      module.add(RegSet("v", "vgprLocalReadAddrA", \
          self.states.a.startVgprLocalReadAddr))
    if self.states.b.numVgprLocalReadAddr > 0:
      module.add(RegSet("v", "vgprLocalReadAddrB", \
          self.states.b.startVgprLocalReadAddr))
    if self.states.m.numVgprLocalReadAddr > 0:
      module.add(RegSet("v", "vgprLocalReadAddrMetadata", \
          self.states.m.startVgprLocalReadAddr))
    if self.states.a.numVgprLocalReadSwapAddr > 0:
      module.add(RegSet("v", "vgprLocalReadSwapAddrA", \
          self.states.a.startVgprLocalReadSwapAddr))
    if self.states.b.numVgprLocalReadSwapAddr > 0:
      module.add(RegSet("v", "vgprLocalReadSwapAddrB", \
          self.states.b.startVgprLocalReadSwapAddr))
    if self.states.m.numVgprLocalReadSwapAddr > 0:
      module.add(RegSet("v", "vgprLocalReadSwapAddrMetadata", \
          self.states.m.startVgprLocalReadSwapAddr))
    if self.states.a.numVgprLocalWriteSwapAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteSwapAddrA", \
          self.states.a.startVgprLocalWriteSwapAddr))
    if self.states.b.numVgprLocalWriteSwapAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteSwapAddrB", \
          self.states.b.startVgprLocalWriteSwapAddr))
    if self.states.m.numVgprLocalWriteSwapAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteSwapAddrMetadata", \
          self.states.m.startVgprLocalWriteSwapAddr))

    if kernel["ProblemType"]["OutputAmaxD"]:
      module.add(RegSet("v", "vgprAmaxOut", self.startVgprAmaxOut))
      module.add(RegSet("v", "vgprAmaxOutB", self.startVgprAmaxOutB))

    if kernel["ProblemType"]["DataType"].isDoubleComplex() and kernel["MIArchVgpr"]:
      module.add(RegSet("v", "vgprAlphaTmp", \
          self.states.startVgprAlphaTmp))

    module.add(RegSet("v", "vgprSerial", self.states.startVgprSerial))

    if self.debugConfig.debugKernel:
      module.add(RegSet("v", "vgprAddressDbg", \
          self.states.startVgprAddressDbg))
    #module.addComment0("Occu: %u waves/simd" % self.numWavesPerSimd )
    # module.addComment0("Num VGPR=%u"%self.vgprPool.size())
    # module.addComment0("Num AccVGPR=%u"%self.agprPool.size())
    self.moduleVgprMacro = moduleVgprMacro
    self.moduleVgprMacroValuA = moduleVgprMacroValuA
    self.moduleVgprMacroValuB = moduleVgprMacroValuB
    self.moduleVgprMacroValuAPack = moduleVgprMacroValuAPack
    self.moduleVgprMacroValuBPack = moduleVgprMacroValuBPack
    self.moduleVgprMacroValuM = moduleVgprMacroValuM
    self.moduleVgprMacroValuMPack = moduleVgprMacroValuMPack
    self.moduleVgprMacroValuA = moduleVgprMacroValuA
    self.moduleVgprMacroG2LA = moduleVgprMacroG2LA
    self.moduleVgprMacroG2LB = moduleVgprMacroG2LB
    module.addComment2("VGPR Macro Assignments")
    module.add(self.moduleVgprMacro)
    module.add(self.moduleVgprMacroValuA)
    module.add(self.moduleVgprMacroValuB)
    module.add(self.moduleVgprMacroValuAPack)
    module.add(self.moduleVgprMacroValuBPack)
    module.add(self.moduleVgprMacroValuM)
    module.add(self.moduleVgprMacroValuMPack)
    module.add(self.moduleVgprMacroG2LA)
    module.add(self.moduleVgprMacroG2LB)

    ########################################
    # SGPR Macros
    ########################################
    module.addComment2("SGPR Assignments")

    # Emit declarations for all sgprs allocated with defineSgpr
    # in the order they were declared
    for skey in self.sgprs:
      module.add(RegSet("s", "sgpr"+skey, self.sgprs[skey]))
    # module.addComment0("max SGPR=%u"%self.sgprPool.size())

    if kernel["StreamK"] == 2 or kernel["StreamK"] == 3:
      module.addSpaceLine()
      module.addComment0("StreamK Parallel Reduction Assignments")
      module.add(RegSet("s", "sgprSkSplit", "sgprskTiles", 0))
      module.add(RegSet("s", "sgprSkPartialIdx", "sgprBeta", 0))

    module.addSpaceLine()
    module.addComment0("Size Assignments")
    problemType = kernel["ProblemType"]
    for idx in range(max(problemType["IndexAssignmentsA"] + problemType["IndexAssignmentsB"])+1):
      idxChar= INDEX_CHARS[idx]
      if idx in problemType["IndicesFree"] or idx in problemType["IndicesBatch"]:
        idxType="Free"
      elif idx in problemType["IndicesSummation"]:
        idxType="Sum"
        idx = idx - problemType["NumIndicesC"]
      else:
        raise ValueError("unexpected index type in size assignments")

      module.add(RegSet("s", "sgprSize%s"%(idxChar), \
                  "sgprSizes%s"%idxType, idx))

    module.addSpaceLine()
    module.addComment0("Stride Assignments")
    for tc in ('D','C'):
      for idx in range(0, problemType["NumIndicesC"]):
        i = idx
        idxChar= self.states.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesCD"]:
          module.add(ValueSet("constStride%s%s"%(tc,idxChar), 1))
        else:
          if not kernel["ProblemType"]["UseInitialStridesCD"]:
            i = i-1
          module.add(RegSet("s", "sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s"%tc, i))

    tcList = ["A", "B"]
    if kernel["ProblemType"]["Sparse"]:
      tcList.append("Metadata")
    for tc in tcList:
      for i, idx in enumerate(problemType["IndexAssignments%s"%tc]):
        idxChar= self.states.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesAB"]:
          module.add(ValueSet("constStride%s%s"%(tc,idxChar), 1))
        else:
          if not kernel["ProblemType"]["UseInitialStridesAB"]:
            i = i-1
          module.add(RegSet("s", "sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s"%tc, i))

    module.addSpaceLine()
    module.add(ValueSet("MT0", kernel["MacroTile0"]))
    module.add(ValueSet("MT1", kernel["MacroTile1"]))
    module.add(ValueSet("DepthU", kernel["DepthU"]))
    module.add(ValueSet("BpeA", tPA["bpe"]))
    module.add(ValueSet("BpeALog2", log2(tPA["bpe"])))
    module.add(ValueSet("BpeB", tPB["bpe"]))
    module.add(ValueSet("BpeBLog2", log2(tPB["bpe"])))
    module.add(ValueSet("BpeAGR", tPA["bpeGR"]))
    module.add(ValueSet("BpeAGRLog2", log2(tPA["bpeGR"])))
    module.add(ValueSet("BpeBGR", tPB["bpeGR"]))
    module.add(ValueSet("BpeBGRLog2", log2(tPB["bpeGR"])))
    # TODO- get real value from MI-InstM/N
    if kernel["ProblemType"]["SwizzleTensorA"]:
      module.add(ValueSet("MI_M", 16))
    if kernel["ProblemType"]["SwizzleTensorB"]:
      module.add(ValueSet("MI_N", 16))
    #
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.add(ValueSet("BpeMetadata", tPM["bpe"]))
      module.add(ValueSet("BpeMetadataLog2", log2(tPM["bpe"])))
    module.addComment0("Number of elements to shift-left SRD")
    module.add(ValueSet("SrdShiftLeftA", self.states.srdShiftLeft['A']))
    module.add(ValueSet("SrdShiftLeftB", self.states.srdShiftLeft['B']))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.add(ValueSet("SrdShiftLeftMetadata", self.states.srdShiftLeft["Metadata"]))
    if kernel["BufferLoad"] or kernel["BufferStore"]:
      module.addComment0("2GB limit - set offsets to -1 to exceed this and clamp")
      module.add(ValueSet("BufferLimit", 0xffffffff, format=1))
      #TODO-64 : This is max 32-bit negative value, the tail loop
      # does incrementally step through the GRO and increment GRO
      # which are initialized with this value
      module.add(ValueSet("BufferOOB", 0x80000000, format=1))

      srdUpperValue = SrdUpperValue(self.states.version)
      module.addComment2("Bits 127:96 of SRD.\n" + srdUpperValue.desc())
      module.add(ValueSet("Srd127_96", srdUpperValue.getValue(), format=1))

    if ((kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel')):
      module.addComment0("MT offset for 64b address (=MT0*MT1*bpeC)")
      reductionOffset = int(kernel["MacroTile0"]*kernel["MacroTile1"]*self.states.bpeCinternal)
      reductionOffsetLow32 = reductionOffset & 0xFFFFFFFF
      reductionOffsetHigh32 = (reductionOffset >> 32) & 0xFFFFFFFF
      module.add(ValueSet("MTOffset", reductionOffsetLow32, format=1))
      module.add(ValueSet("MTOffsetH32", reductionOffsetHigh32, format=1))

    ########################################
    # Global Offsets
    ########################################
    # justOffset32 means we should only write the 32-bit offset
    # This is used in Buffer addressing modes.
    # Flat addressing modes expect the GLOBAL_OFFSET to initialize a full 64-bit address

    GOList =  [ \
        ("C", list(range(0, kernel["ProblemType"]["NumIndicesC"])), kernel["BufferStore"], None, False), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"], kernel["BufferLoad"], tPA, False), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"], kernel["BufferLoad"], tPB, False) ]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      GOList.append(("Metadata", kernel["ProblemType"]["IndexAssignmentsMetadata"], kernel["BufferLoad"], tPM, False))
    if kernel["ProblemType"]["SwizzleTensorA"]:
      GOList.append(("A", kernel["ProblemType"]["IndexAssignmentsA"], kernel["BufferLoad"], tPA, True))
    if kernel["ProblemType"]["SwizzleTensorB"]:
      GOList.append(("B", kernel["ProblemType"]["IndexAssignmentsB"], kernel["BufferLoad"], tPB, True))

    for (tc, indices, justOffset32, tP, isSwizzled) in GOList:

      # BufferStore does not use this macro so don't generate it:
      if tc == "C" and kernel["BufferStore"]:
        continue

      # function name and comment
      suffix_tc = tc + "_SWIZZLED" if isSwizzled else tc
      module.addComment1("Global Offset %s"%suffix_tc)
      numDim = len(indices)
      idxChars = []
      for i in indices:
        idxChars.append(self.states.indexChars[i])

      # macro declaration
      calcDims = [] # dimensions which are participating in the address calc (ignores other summation)
      mirrorSumDims = []
      macroArgs = []
      for i in range(0, numDim):
        if tc == 'C':
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesCD"]
          idxChar = self.states.indexChars[i]
        else:
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesAB"]
          idxChar = self.states.indexChars[tP['ia'][i]]

        # tile index or unroll vgpr or summation
        # other summation (other than unroll) are included in the GLOBAL_OFFSET macro but not used in address calc
        if     tc in ('A','C') and indices[i] == kernel["ProblemType"]["Index0"] \
            or tc in ('B','C', "Metadata") and indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          macroArgs.append("vgprOffset%s:req" % idxChars[i])
          calcDims.append(i)
        elif indices[i] in kernel["ProblemType"]["IndicesSummation"]:
          # other summation index (not unroll)
          if tc in ('A', 'B', "Metadata") and indices[i] in kernel["ProblemType"]["MirrorDims%s" % tc]:
            mirrorSumDims.append(i)
          continue
        else:
          # other batch or free index
          if isPackedIndex(kernel, indices[i]):
            calcDims.append(i)
            macroArgs.append("vgprOffset%s:req" % idxChars[i])
          elif not justOffset32: # buffer/justOffset32 scalars are included in SRD not the offset, so skip here
            calcDims.append(i)
            macroArgs.append("sgprOffset%s:req" % idxChars[i])

      macro = Macro("GLOBAL_OFFSET_%s" % suffix_tc, ["vgprAddr:req", *macroArgs, "vgprTmp:req"])

      # Each index may be skipped, scaled by stride, or unscaled
      # If destLo is unset, no accumulation is necessary.

      # if the first index (i==0) is unscaled (UseInitialStrides),
      # it can be combined at the next update or moved at end
      # (if there is no next update)

      pendingOffset = None # offset pending for accumulation
      offsetIsVgpr = False # True if the source is VGPR ; False if SGPR
      destLo = None

      # true for first addr calc. In this case, we can directly write addr
      # rather than accumulating through a tmp
      writeDirectToAddr = 1

      # mirror other summation indices
      for i in mirrorSumDims:
        if writeDirectToAddr:
          dest = vgpr("Addr+0", isMacro=True)
          needAdd = 0 # don't need add since writing address directly.
          writeDirectToAddr = 0
        else:
          dest = vgpr("Tmp+0", isMacro=True)
          needAdd = 1
        macro.add(VSubU32(dst=dest, \
                src0=sgpr("Size%s"%INDEX_CHARS[indices[i]]), \
                src1=1, \
                comment="mirror %s%s 1"%(tc, INDEX_CHARS[indices[i]])))
        macro.add(VMulLOU32(dst=dest, \
                src0=dest, \
                src1=self.strideRef(tc, indices[i]), \
                comment="mirror %s%s 2"%(tc, INDEX_CHARS[indices[i]])))

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = vgpr("Addr+0", isMacro=True)
          destHi = vgpr("Addr+1", isMacro=True)

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          macro.add(VAddCOU32(dst=destLo, \
            dst1=VCC(), \
            src0=srcLo, \
            src1="v[\\vgprTmp+0]", \
            comment="accumulate %s lower"%idxChar))

      for i in calcDims:
        # should have eliminated these above
        idx = indices[i]
        isMirrorIdx = tc in ('A', 'B', "Metadata") and idx in kernel["ProblemType"]["MirrorDims%s" % tc]
        assert not (idx in kernel["ProblemType"]["IndicesSummation"] and idx != kernel["ProblemType"]["IndexUnroll"])

        if indices[i] == kernel["ProblemType"]["Index0"] \
            or indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          offsetIsVgpr = True
        # other c index sgpr (free or batch)
        elif indices[i] < kernel["ProblemType"]["NumIndicesC"]:
          if isPackedIndex(kernel, indices[i]):
            offsetIsVgpr = True
          else:
            offsetIsVgpr = False
        else:
          assert(0) # no other type allowed

        # offset is VGPR or SGPR string to use for the offset
        if offsetIsVgpr:
          offset = "v[\\vgprOffset%s]" % idxChars[i]
        else:
          offset = "s[\\sgprOffset%s]" % idxChars[i]

        # macro.addComment0("dim%s pendingOffset=%s offset=%s offsetIsVgpr=%s" \
        #    % (self.states.indexChars[indices[i]], pendingOffset, offset, offsetIsVgpr))

        needAdd = 0
        # should be indices[i]??
        if i==0 and not useInitialStrides:
          # slide into next address calc - can do addr = pendingOffset + nextAddrCalc
          pendingOffset = offset
          writeDirectToAddr = 0
        else:
          # tile index or unroll vgpr
          if offsetIsVgpr:
            if writeDirectToAddr:
              destLo = vgpr("Addr+0", isMacro=True)
              destHi = vgpr("Addr+1", isMacro=True)
              needAdd = 0 # don't need add since writing address directly.
              writeDirectToAddr = 0
            else:
              destLo = vgpr("Tmp+0", isMacro=True)
              destHi = vgpr("Tmp+1", isMacro=True)
              needAdd = 1
            if isMirrorIdx:
              macro.add(VSubI32(
                dst=vgpr("Tmp+0", isMacro=True),
                src0=sgpr("Size%s"%INDEX_CHARS[idx]), \
                src1=offset, \
                comment="mirror %s%s 1"%(tc, INDEX_CHARS[indices[i]])))
              macro.add(VSubI32(\
                dst=vgpr("Tmp+0", isMacro=True),
                src0="v[\\vgprTmp+0]", \
                src1=1, \
                comment="mirror %s%s 2"%(tc, INDEX_CHARS[indices[i]])))
              offset = "v[\\vgprTmp+0]"

            # offset * stride
            if isSwizzled:
              # Swizzle directly uses the offset base that has already been multiplied by the stride.
              macro.add(VMovB32(dst=destLo, src=offset))
            else:
              macro.add(VMulLOU32(dst=destLo,
                src0=self.strideRef(tc, indices[i]), \
                src1=offset, \
                comment="mul d%u lower"%i))
            if not justOffset32:
              macro.add(VMulHIU32(dst=destHi,
                  src0=self.strideRef(tc, indices[i]), \
                  src1=offset, \
                  comment="mul d%u upper"%i))
          else: # offset is SGPR:
            assert not isMirrorIdx
            if not justOffset32:
              # buffer mode (aka justOffset32) does scalars into SRD not offset
              macro.add(VMovB32(dst=vgpr("Tmp+2", isMacro=True), src="s[\\sgprOffset%s]"%idxChars[i], \
                  comment="sgprOffset -> vgprTmp+2"))
              # offset * stride
              macro.add(VMulLOU32(dst=vgpr("Tmp+0", isMacro=True), \
                  src0=self.strideRef(tc, indices[i]), src1="v[\\vgprTmp+2]",  \
                  comment="other stride mul d%u lower"%i))
              macro.add(VMulHIU32(dst=vgpr("Tmp+1", isMacro=True), \
                  src0=self.strideRef(tc, indices[i]), src1="v[\\vgprTmp+2]",  \
                  comment="mul d%u upper"%i))
              needAdd = 1

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = vgpr("Addr+0", isMacro=True)
          destHi = vgpr("Addr+1", isMacro=True)
          # addr += offset * stride (lo) : accumulate just-computed address term into addr

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          macro.add(VAddCOU32(dst=destLo, dst1=VCC(), \
            src0=srcLo, src1="v[\\vgprTmp+0]", \
            comment="accumulate %s lower"%idxChar))

          # addr += offset * stride (hi)
          if not justOffset32:
            macro.add(VAddCCOU32(dst=vgpr("Addr+1", isMacro=True), dst1=VCC(), \
                src0="v[\\vgprTmp+1]", src1=srcHi, src2=VCC(), \
                comment="accumulate %s upper"%idxChar))
          pendingOffset = None

      # pendingOffset but never got a chance to apply it,
      # need to just add an explicit move or add:
      # this can happen for small-order tensors
      if pendingOffset != None:
        destLo = vgpr("Addr+0", isMacro=True)
        if writeDirectToAddr:
          macro.add(VMovB32(dst=destLo, src=offset, comment="setup d0 lower"))
          if not justOffset32:
            macro.add(VMovB32(dst=vgpr("Addr+1", isMacro=True), src=0, comment="d0 upper"))
        else:
          macro.add(VAddCOU32(dst=destLo, dst1=VCC(), \
            src0=destLo, src1=pendingOffset, \
            comment="accumulate final pendingOffset"))


      if tP != None and kernel["BufferLoad"] and self.states.srdShiftLeft[tc]:
        macro.add(VAddU32(dst=vgpr("Addr+0", isMacro=True), \
            src0=hex(self.states.srdShiftLeft[tc]), \
            src1="v[\\vgprAddr+0]", \
            comment="add prepad for pointer shift"))

      bpeGR = tP["bpeGR"] if not tP["isM"] else tP["bpe"]
      # addr *= bytes/element
      if justOffset32:
        macro.add(vectorStaticMultiply(vgpr("Addr+0", isMacro=True), vgpr("Addr+0", isMacro=True), bpeGR, None, "offset *= bytes/element"))
      else:
        macro.add(VLShiftLeftB64(dst=vgpr("Addr+0", 2, isMacro=True), \
            shiftHex=hex(log2(bpeGR)), \
            src="v[\\vgprAddr+0:\\vgprAddr+1]", \
            comment="offset *= bytes/element"))
      module.add(macro)

    if kernel["ProblemType"]["StochasticRounding"] and not self.states.asmCaps["v_prng_b32"] :
      module.add(PseudoRandomGenerator())

    if not kernel["EnableMatrixInstruction"]:
      # Macro MAC
      PLR = kernel["PrefetchLocalRead"] \
        if kernel["PrefetchLocalRead"] < kernel["LoopIters"] \
        else kernel["LoopIters"] - 1
      for m in range(0, 1+PLR):
        macro = Macro("MAC_%ux%u_X%u" % (kernel["ThreadTile0"], kernel["ThreadTile1"], m), [])
        component = Component.MAC.find(self)
        if not component:
          printExit("Assembly doesn't support datatype %s" % kernel["ProblemType"]["DataType"])
        innerModule = component(self, tPA, tPB, m, kernel["InnerUnroll"])
        for item in innerModule.items():
            macro.add(item)
        module.add(macro)

      # dot2: oneIUI for tail loop
      if kernel["UseDotInstruction"]:
        macro = Macro("MAC_%ux%u_X%u_OneIUI" % (kernel["ThreadTile0"],kernel["ThreadTile1"], 0), [""])
        component = Component.MAC.find(self)
        if not component:
          printExit("Assembly doesn't support datatype %s" % kernel["ProblemType"]["DataType"])
        innerModule = component(self, tPA, tPB, 0, 1)
        for item in innerModule.items():
            macro.add(item)
        module.add(macro)

    module.setNoOpt(True)
    return module

  def checkResources(self, kernel, mkb: KernelBody):
    # register allocation
    totalVgprs = self.vgprPool.size()
    totalAgprs = self.agprPool.size()
    totalSgprs = self.sgprPool.size()

    mkb.setGprs(totalVgprs=totalVgprs, totalAgprs=totalAgprs, totalSgprs=totalSgprs)

    if self.vgprPool.size() > self.states.regCaps["MaxVgpr"]:
      self.states.overflowedResources = 1
    elif self.sgprPool.size() > self.states.regCaps["MaxSgpr"]:
      self.states.overflowedResources = 2

    # TODO: Add target occupancy or kept the occupancy settings from globalWriteBatch
    kernel["CUOccupancy"] = self.getOccupancy(kernel["NumThreads"], self.vgprPool.size(), self.sgprPool.size(), \
      self.getLdsSize(kernel), self.agprPool.size(), self.states.doubleVgpr)
    if kernel["ScheduleIterAlg"] == 2 and kernel["CUOccupancy"] < 2:
      self.states.overflowedResources = 6

    vgprPerThreadPerOccupancy = self.states.regCaps["PhysicalMaxVgprCU"] // kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy // mkb.getNextFreeVgpr()
    if numWorkGroupsPerCU < 1:
      self.states.overflowedResources = 4

    if self.states.invalidLSUCode:
      self.states.overflowedResources = 7

    self.vgprPool.checkFinalState()

    if self.states.overflowedResources:
      mkb.body.add(ValueEndif("overflowed resources"))
      if self.states.overflowedResources == 1:
        msg = "too many vgprs"
      elif self.states.overflowedResources == 2:
        msg = "too many sgprs"
      elif self.states.overflowedResources == 3:
        msg = "half store requires at least two elements per batch"
      elif self.states.overflowedResources == 4:
        msg = "Occupancy limit"
      elif self.states.overflowedResources == 5:
        if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
          msg = "cannot schedule local read with DirectToLds"
        else:
          msg = "reading and writing LDS at same time require 2 LDS buffer"
      elif self.states.overflowedResources == 6:
        msg = "SIA2 better with occupancy 2"
      elif self.states.overflowedResources == 7:
        msg = "invalid LSU code due to assertion fail"
      else:
        msg = "unknown"

      if self.debugConfig.printSolutionRejectionReason:
        printWarning("%s overflowed resources.  errorCode=%d, msg=\"%s\", vgprs=%u, sgprs=%u" \
          % (self.states.kernelName, self.states.overflowedResources, msg, \
          self.vgprPool.size(), self.sgprPool.size()))
      mkb.body.add(SEndpgm(comment="overflowed resources"), 0)
      mkb.body.add(ValueIf(value=0), 1)

  ##############################################################################
  # code phrase for load batched address from array of buffer pointer
  ##############################################################################
  def loadBatchedAddress(self, kernel, Batch, tmpSgprResource: ContinuousRegister):
    tmpSgpr = tmpSgprResource.idx
    laneSC = tmpSgprResource.size
    module = Module("loadBatchedAddress %s" % Batch)
    module.addSpaceLine()

    # handle Batch C/D
    if not kernel["_GlobalAccumulation"]:
      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(Batch), src1=0x8, comment="offset of global buffer address"))
          module.add(SLoadB64(dst=sgpr("AddressD", 2), base=sgpr("AddressD",2), soffset=sgpr(tmpSgpr), comment="load global buffer D address"))

      endCheckLabel = Label(self.labels.getName(f"label_skip_c_buffer_deref_{Batch}"), "")
      module.add(BranchIfZero("Beta", kernel["ProblemType"]["ComputeDataType"].toEnum(), tmpSgpr, laneSC, endCheckLabel, \
                     kernel['WavefrontSize']))

      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(Batch), src1=0x8, comment="offset of global buffer address"))
          module.add(SLoadB64(dst=sgpr("AddressC", 2), base=sgpr("AddressC",2), soffset=sgpr(tmpSgpr), comment="load global buffer C address"))

      module.add(endCheckLabel)

    #handle Batch A/B
    endCheckLabel = Label(self.labels.getName(f"label_skip_ab_buffer_deref_{Batch}"), "")
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(1), comment="check summation size"))
    for i in range(0, self.states.numSgprSizesSum):
      module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr("SizesSum+%u"%(i)), src1=sgpr(tmpSgpr), comment="check summation size"))
    module.add(SCmpEQU32(src0=sgpr(tmpSgpr), src1=0, comment="skip buffer deref is size of summation is 0"))
    module.add(SCBranchSCC1(labelName=endCheckLabel.getLabelName(), comment="skip buffer deref is size of summation is 0"))
    module.add(BranchIfZero("Alpha", kernel["ProblemType"]["ComputeDataType"].toEnum(), tmpSgpr, laneSC, endCheckLabel, \
                                 kernel['WavefrontSize']))

    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(Batch), src1=0x8, comment="offset of global buffer address"))
    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        module.add(SLoadB64(dst=sgpr("AddressA", 2), base=sgpr("AddressA",2), soffset=sgpr(tmpSgpr), comment="load global buffer A address"))
        module.add(SLoadB64(dst=sgpr("AddressB", 2), base=sgpr("AddressB",2), soffset=sgpr(tmpSgpr), comment="load global buffer B address"))

    module.add(endCheckLabel)

    return module

  ##############################################################################
  def getKernelArgLoadModule(self, kernel, sgprStartIdx, numsOfLoad, preloadNum):
    kernelArgs = Module("load arguments")
    kernelArgs.addComment1("Load Kernel Args")
    if self.debugConfig.debugKernel:
      kernelArgs.add(self.argLoader.loadKernArg("AddressDbg", "KernArgAddress", dword=2))
    self.argLoader.resetOffset()
    kernelArgs.addModuleAsFlatItems(self.argLoader.loadAllKernArg(sgprStartIdx, "KernArgAddress", numsOfLoad, preloadNum))
    if kernel["ProblemType"]["UseScaleAB"] == "Scalar":
      sgprOffset = self.argLoader.getOffset()
      for preloadScale, name in zip([self.states.preloadScaleA, self.states.preloadScaleB], ['A','B']):
        if preloadScale:
          kernelArgs.add(self.argLoader.loadKernArg("AddressScale%s"%name, "KernArgAddress", sgprOffset=hex(sgprOffset), dword=2))
        sgprOffset += (self.states.rpga * self.states.bpr)
    return kernelArgs

  def localReadAddresses(self, kernel, tPA, tPB, tPM):
    module = Module("Local Read Addresses")

    ####################################
    # Local Read Addresses
    ####################################
    module.addComment2("Local Read Addresses")

    # tile assignments
    module.addComment1("local read addresses: tile assignments a/b")
    module.add(self.lraTileAssignment(kernel, tPA, tPB))

    # final offsets
    module.addComment1("local read addresses: final offsets a")
    module.add(self.lraFinalOffset(kernel, tPA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.addComment1("local read addresses: final offsets metadata")
      module.add(self.lraFinalOffset(kernel, tPM))
    module.addComment1("local read addresses: final offsets b")
    module.add(self.lraFinalOffset(kernel, tPB))

    # declare addresses
    module.addComment1("local read addresses: declare addresses a")
    module.add(self.lraDeclareAddresses(kernel, tPA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.addComment1("local read addresses: declare addresses metadata")
      module.add(self.lraDeclareAddresses(kernel, tPM))
    module.addComment1("local read addresses: declare addresses b")
    module.add(self.lraDeclareAddresses(kernel, tPB))

    if self.states.a.numVgprLocalReadAddr > 0:
      module.add(self.lraSwapAddressesForDTLPad(kernel, tPA))
    if self.states.b.numVgprLocalReadAddr > 0:
      module.add(self.lraSwapAddressesForDTLPad(kernel, tPB))
    if self.states.m.numVgprLocalReadAddr > 0:
      module.add(self.lraSwapAddressesForDTLPad(kernel, tPM))

    return module

  def localWriteAddresses(self, kernel, tPA, tPB, tPM):
    module = Module("Local Write Addresses")

    ####################################
    # Local Write Addresses
    ####################################
    module.addComment2("Local Write Addresses")

    # tile assignments
    module.add(self.lwaTileAssignment(kernel, tPA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.add(self.lwaTileAssignment(kernel, tPM))
    module.add(self.lwaTileAssignment(kernel, tPB))

    # unroll assignments
    module.add(self.lwaUnrollAssignment(kernel, tPA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.add(self.lwaUnrollAssignment(kernel, tPM))
    module.add(self.lwaUnrollAssignment(kernel, tPB))

    # first offsets
    module.addComment1("local write addresses: first offset a")
    module.add(self.lwaFirstOffset(kernel, tPA))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      module.addComment1("local write addresses: first offset metadata")
      module.add(self.lwaFirstOffset(kernel, tPM))
    module.addComment1("local write addresses: first offset b")
    module.add(self.lwaFirstOffset(kernel, tPB))

    return module

  def remapWgSerial(self, kernel, earlyStop=True):
    module = Module("RemapWgSerial")
    ########
    # remap wg serial to wg0,wg1,wg2
    ########
    # FIXME: Here does not support UseBatch: False
    if "WorkGroup2" in self.sgprs:
      with self.allocTmpSgpr(2) as tmpSgpr:
        module.addComment1("remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0)")
        module.addComment0("wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1))")
        tmpVgpr     = self.vgprPool.checkOut(2)
        tmpVgprRes  = ContinuousRegister(tmpVgpr, 2)
        module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1")))
        if kernel["GlobalSplitU"] != 0:
          module.add(SAndB32(dst=sgpr(tmpSgpr.idx+1), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
          module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr(tmpSgpr.idx), src1=sgpr(tmpSgpr.idx+1)))
        module.add(scalarUInt32DivideAndRemainder(qReg=tmpSgpr.idx, dReg="WorkGroup0", divReg=tmpSgpr.idx, rReg=tmpSgpr.idx+1,\
                                        tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))
        module.add(SMovB32(dst=sgpr("WorkGroup2"), src=sgpr(tmpSgpr.idx)))

        # Convert 1D to 2D
        sgprWGM = "WGM"
        module.addComment0("idxWG01 = idxWG012 - wg2 * numWG0 * numWG1")
        module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("NumWorkGroups1"), src1=sgpr("NumWorkGroups0")))
        module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr(tmpSgpr.idx), src1=sgpr("WorkGroup2")))
        if kernel["GlobalSplitU"] != 0:
          module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr(tmpSgpr.idx), src1=sgpr(tmpSgpr.idx+1)))
        module.add(SSubU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr.idx)))
        module.addComment0("wg1 = idxWG01 * smallMagicNumber(1/numWG0)")
        module.add(scalarUInt32DivideAndRemainder(qReg=tmpSgpr.idx, dReg="WorkGroup0", divReg="NumWorkGroups0", rReg=tmpSgpr.idx+1,\
                                        tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))
        self.vgprPool.checkIn(tmpVgpr)
        module.add(SMovB32(dst=sgpr("WorkGroup1"), src=sgpr(tmpSgpr.idx)))
        module.addComment0("wg0 = idxWG01 - wg1 * numWG0")
        module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0")))
        module.add(SSubU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr.idx)))

      # early stop if wgIdx exceed wg needed
      if earlyStop:
        module.addComment1("Early stop if wg exceed")
        module.add(SCmpGeU32(src0=sgpr("WorkGroup2"), src1=sgpr("SizesFree+2")))
        label_EarlyStop = Label(self.labels.getNameInc("EarlyStop_if_wg_exceed"), "")
        label_nonEarlyStop = Label(self.labels.getNameInc("NoEarlyStop_wgExceed"), "")
        module.add(SCBranchSCC0(labelName=label_nonEarlyStop.getLabelName()))
        module.add(label_EarlyStop)
        module.add(SEndpgm())
        module.add(label_nonEarlyStop)
    return module

  def defineAndResources(self, kernel, tPA, tPB, tPM):
    module = Module("allocateResources")
    module.add(self.macroAndSet(kernel, tPA, tPB))

    module.addComment2("Allocate Resources")
    moduleArgs = Module("load arguments")
    moduleRegInit = Module("Init regs")

    tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]

    sgprNumsOfGemm = None

    if self.do["PreLoop"]:
      ### temp sgpr for groupedgemm ###
      # can be start from sgpr_preload_end
      sgprNumsOfGemm = self.sgprPool.checkOut(1, preventOverflow=False)

      self.kernArgOffset = 0
      self.argLoader = ArgumentLoader()
      self.externalArgLoader = ArgumentLoader()
      ########################################
      # Common parameters
      sgprArgType = self.sgprPool.checkOut(1, preventOverflow=False)
      commonArgs = Module("load arguments")
      commonArgs.addComment1("Load num of Gemms")
      commonArgs.add(self.argLoader.loadKernArg(sgprNumsOfGemm, "KernArgAddress", 0, dword=1))

      sgprPackedArgs = self.sgprPool.checkOut(1, preventOverflow=False)
      # Load combined internal arguments
      commonArgs.addComment1("Load packed kernel args (StaggerU/GSU)")
      commonArgs.add(self.argLoader.loadKernArg(sgprPackedArgs, "KernArgAddress", 4, dword=1))
      commonArgs.addComment1("Load WGM data")
      commonArgs.add(self.argLoader.loadKernArg("WGM", "KernArgAddress", 8, dword=1))
      tmpSgprNumWorkGroups = self.sgprPool.checkOut(1, preventOverflow=False)
      commonArgs.addComment1("Load num of WGs")
      commonArgs.add(self.argLoader.loadKernArg(tmpSgprNumWorkGroups, "KernArgAddress", 12, dword=1))
      ########################################
      # kernel args parameters
      load = self.states.numSgprToLoad
      sgprStart = self.sgprs["SizesFree"]

      ########################################
      # load ws/ user args
      hbmArgs = Module("load HBM arguments")
      hbmArgs.addComment1("Load address of kernel arguments")
      hbmArgs.add(self.argLoader.loadKernArg("KernArgAddress", "KernArgAddress", self.states.userArgsInfo.commonArgsSize, dword=2))

      moduleArgs.addModuleAsFlatItems(deepcopy(commonArgs))
      moduleArgs.add(SWaitCnt(kmcnt=0, comment="load args"))
      moduleArgs.add(SLShiftRightB32(dst=sgpr(sgprArgType), shiftHex=hex(30), src=sgpr(sgprNumsOfGemm), comment="Get arg type"))
      moduleArgs.add(SAndB32(dst=sgpr(sgprNumsOfGemm), src0=hex(0x3FFFFFFF), src1=sgpr(sgprNumsOfGemm), comment="Get nums of gemm"))

      if ((kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel')):
        extReadEpilogueLabeltmp    = Label(label=self.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
        moduleArgs.addComment0("Check if custom structure pointer is null")
        if kernel["ProblemType"]["SupportUserArgs"]:
          moduleArgs.add(SCmpEQU32(src0=sgpr(sgprArgType), src1=2, comment="ArgType == 2 ?"))
          moduleArgs.add(SCBranchSCC0(labelName=extReadEpilogueLabeltmp.getLabelName()))
        moduleArgs.addComment1("Grouped Gemm: Load address of external kernel arguments")
        moduleArgs.add(self.argLoader.loadKernArg("AddressTD", "KernArgAddress", hex(self.states.userArgsInfo.commonArgsSize+16), dword=2))
        moduleArgs.add(self.argLoader.loadKernArg("Synchronizer", "KernArgAddress", hex(self.states.userArgsInfo.commonArgsSize+8), dword=2))
        moduleArgs.add(extReadEpilogueLabeltmp)

      moduleArgs.add(SCmpEQU32(src0=sgpr(sgprArgType), src1=(0), comment="Is kernel args"))
      labelHBM = Label("HBMArgs", comment="")
      labelLoadEnd = Label("LoadArgsEnd", comment="")
      moduleArgs.add(SCBranchSCC0(labelName=labelHBM.getLabelName()))
      moduleArgs.add(SAddU32(dst=sgpr("KernArgAddress"), src0=sgpr("KernArgAddress"), src1=hex(self.states.userArgsInfo.commonArgsSize), comment="Shift common args"))
      moduleArgs.add(SAddCU32(dst=sgpr("KernArgAddress+1"), src0=sgpr("KernArgAddress+1"), src1=0))
      moduleArgs.addModuleAsFlatItems(self.getKernelArgLoadModule(kernel, sgprStart, load, 0))
      if self.states.numSgprPreload > 0:
        moduleArgs.add(SWaitCnt(kmcnt=0, comment="preload"))
      moduleArgs.add(SBranch(labelName=labelLoadEnd.getLabelName()))
      moduleArgs.add(labelHBM)
      moduleArgs.addModuleAsFlatItems(deepcopy(hbmArgs))
      moduleArgs.add(SWaitCnt(kmcnt=0, comment="wait for args to load"))
      moduleArgs.add(labelLoadEnd)

      if self.states.numSgprPreload > 0:
        common_kern_entry  = Label(label="common_kernel_entry", comment="for both preload/non-preload common code")


        #For groupgemm, the preload happened prior to this stage
        moduleArgs.add(SBranch(common_kern_entry.getLabelName())) # jump to common path
        total_inst_dwords = 0
        for inst in moduleArgs.items():
          if isinstance(inst, (BranchInstruction, SWaitCnt, CommonInstruction)):
            total_inst_dwords = total_inst_dwords + 1
          elif isinstance(inst, (SMemLoadInstruction)):
            total_inst_dwords = total_inst_dwords + 2
        assert total_inst_dwords <= 64
        moduleArgs.addComment1("pad %u snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue" % (64 - total_inst_dwords))
        for i in range(64 - total_inst_dwords):
          moduleArgs.add(SNop(waitState=0, comment=""))
        moduleArgs.add(Label("Preload_Offset_Start", ""))
        # Common args preload
        preloadSgprStartIdx = self.states.rpga
        moduleArgs.add(SAndB32(dst=sgpr(sgprNumsOfGemm), src0=hex(0x3FFFFFFF), src1=sgpr(preloadSgprStartIdx), comment="Get nums of gemm"))
        moduleArgs.add(SLShiftRightB32(dst=sgpr(sgprArgType), shiftHex=hex(30), src=sgpr(preloadSgprStartIdx), comment="Get arg type"))

        if ((kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel')):
          extReadEpilogueLabeltmp    = Label(label=self.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
          moduleArgs.addComment0("Check if custom structure pointer is null")
          if kernel["ProblemType"]["SupportUserArgs"]:
            moduleArgs.add(SCmpEQU32(src0=sgpr(sgprArgType), src1=2, comment="ArgType == 2 ?"))
            moduleArgs.add(SCBranchSCC0(labelName=extReadEpilogueLabeltmp.getLabelName()))
          moduleArgs.add(SMovB32(dst=sgpr("Synchronizer+1"), src=sgpr(preloadSgprStartIdx+7), comment="Load Synchronizer data"))
          moduleArgs.add(SMovB32(dst=sgpr("Synchronizer"), src=sgpr(preloadSgprStartIdx+6), comment="Load Synchronizer data"))
          moduleArgs.add(SMovB32(dst=sgpr("AddressTD+1"), src=sgpr(preloadSgprStartIdx+9), comment="Load AddressTD data"))
          moduleArgs.add(SMovB32(dst=sgpr("AddressTD"), src=sgpr(preloadSgprStartIdx+8), comment="Load AddressTD data"))
          moduleArgs.add(extReadEpilogueLabeltmp)

        moduleArgs.add(SMovB32(dst=sgpr(sgprPackedArgs), src=sgpr(preloadSgprStartIdx+1), comment="Preload internal args"))
        moduleArgs.add(SCmpEQU32(src0=sgpr(sgprArgType), src1=(0), comment="Is kernel args"))
        preloadLabelHBM = Label("Preload_HBMArgs", comment="")
        perloadLabelLoadEnd = Label("Preload_LoadArgsEnd", comment="")
        moduleArgs.add(SCBranchSCC0(labelName=preloadLabelHBM.getLabelName()))
        moduleArgs.add(SAddU32(dst=sgpr("KernArgAddress"), src0=sgpr("KernArgAddress"), src1=hex(self.states.userArgsInfo.commonArgsSize), comment="Shift common args"))
        moduleArgs.add(SAddCU32(dst=sgpr("KernArgAddress+1"), src0=sgpr("KernArgAddress+1"), src1=0))
        self.argLoader.resetOffset()
        moduleArgs.addModuleAsFlatItems(self.getKernelArgLoadModule(kernel, sgprStart, load, self.states.numSgprPreload - self.states.userArgsInfo.commonArgsNum))
        i = self.states.userArgsInfo.commonArgsNum
        while i < self.states.numSgprPreload:
          dstIndex = sgprStart+i-self.states.userArgsInfo.commonArgsNum
          srcIndex = preloadSgprStartIdx+i
          if (i + 1 < self.states.numSgprPreload) and (dstIndex % 2 == 0) and (srcIndex % 2 == 0):
            moduleArgs.add(SMovB64(dst=sgpr(sgprStart+i-self.states.userArgsInfo.commonArgsNum, 2), src=sgpr(preloadSgprStartIdx+i, 2), comment="move preload data to correct sgpr"))
            i += 2
          else:
            moduleArgs.add(SMovB32(dst=sgpr(sgprStart+i-self.states.userArgsInfo.commonArgsNum), src=sgpr(preloadSgprStartIdx+i), comment="move preload data to correct sgpr"))
            i += 1
        moduleArgs.add(SBranch(labelName=perloadLabelLoadEnd.getLabelName()))
        moduleArgs.add(preloadLabelHBM)
        moduleArgs.add(SMovB64(dst=sgpr("KernArgAddress", 2), src=sgpr(preloadSgprStartIdx+4, 2), comment="Load address of kernel arguments"))
        moduleArgs.add(perloadLabelLoadEnd)
        moduleArgs.add(SMovB32(dst=sgpr("WGM"), src=sgpr(preloadSgprStartIdx+2), comment="Preload internal args2"))
        moduleArgs.add(SMovB32(dst=sgpr(tmpSgprNumWorkGroups), src=sgpr(preloadSgprStartIdx+3), comment="Load num of WGs"))
        # add common kern entry label
        moduleRegInit.add(common_kern_entry)
        for i in range(kernel["ProblemType"]["NumIndicesC"]):
          moduleRegInit.add(SMovB32(dst=sgpr("WorkGroup0+%u"%i), src=sgpr(preloadSgprStartIdx+self.states.numSgprPreload+i), \
                      comment="restore workgroup id"))

      moduleRegInit.add(SAndB32(dst=sgpr("StaggerU"), src0=sgpr(sgprPackedArgs), src1=hex(0xFFFF0000), comment="Restore StaggerU related vars"))
      moduleRegInit.add(SLShiftRightB32(dst=sgpr("StaggerU"), shiftHex=hex(16), src=sgpr("StaggerU")))
      if kernel["GlobalSplitU"] != 0:
        moduleRegInit.add(SAndB32(dst=sgpr("GSU"), src0=sgpr(sgprPackedArgs), src1=hex(0xFFFF), comment="Restore GSUConfig and GSU"))

      if kernel["ProblemType"]["SupportUserArgs"]:
        moduleRegInit.add(SMovB32(dst=sgpr("ArgType"),src=sgpr(sgprArgType)))

    self.sgprPool.checkIn(sgprPackedArgs)

    if kernel["StorePriorityOpt"]:
      moduleRegInit.add(SSetPrior(prior=3, comment="optimization store"))

    if self.do["PreLoop"]:
      if self.db["InitSgpr"] & 0x1:
        moduleRegInit.addComment1("Init SGPRs")
        for i in range(self.states.firstInitSgpr, self.sgprPool.size()):
          moduleRegInit.add(SMovB32(dst=sgpr(i), src=hex(self.consts.initSgprValue), comment="InitSgpr&0x1"))
        moduleRegInit.addSpaceLine()

      if self.db["InitVgpr"] & 0x1:
        moduleRegInit.addComment1("Init VGPRs")
        for i in range(1, self.states.totalVgprs):
          moduleRegInit.add(VMovB32(dst=vgpr(i), src=hex(self.consts.initVgprValue), comment="InitVgpr&0x1"))
        moduleRegInit.addSpaceLine()

      # set m0
      moduleRegInit.add(SMovB32(dst=mgpr(0), src=hex(kernel["LdsNumBytes"]),
          comment="LDS clamp at %u bytes"%(kernel["LdsNumBytes"])))

      # set Serial id vgpr
      moduleRegInit.add(VMovB32(dst=vgpr("Serial"), src=vgpr(0), comment="thread serial id"))

      if self.states.kernel["WavefrontSize"] == 32:
        moduleRegInit.add(SMovB32(dst=VCC(setHi=True), src=0, comment="Ensure hi bits are zero"))

      # SrdD can be used as temp sgprs for a bit
      if self.states.doShadowInit and kernel["BufferStore"]:
        self.addSgprVarToPool("SrdD")

      waitForScaleAB = False
      moduleScaleAB = Module("Load ScaleAB")
      if kernel["ProblemType"]["UseScaleAB"] == "Scalar":
        for preloadScale, name in zip([self.states.preloadScaleA, self.states.preloadScaleB], ['A','B']):
          if preloadScale:
            waitForScaleAB = True
            moduleScaleAB.add(SMovB32(dst=sgpr("Scale%s"%name), src=1.0 , comment="init as 1" ))
            label  = Label(self.labels.getNameInc("Scale%sValid"%name), "")
            moduleScaleAB.add(BranchIfZero("AddressScale%s"%name, DataType('int64').toEnum(), -1, int(kernel["WavefrontSize"]/32), label, kernel["WavefrontSize"]))
            # load scale data
            moduleScaleAB.add(SLoadB32(dst=sgpr("Scale%s"%name), base=sgpr("AddressScale%s"%name,2), soffset=0, comment="load scale%s"%name))
            moduleScaleAB.add(label)

      moduleWg = Module("Calculate Workgroup")

      # C regs are not used during initialization so mark them as available -
      # we will claim then just before the start of the unroll loop:
      self.vgprPool.add(self.states.a.startVgprValu , \
          self.states.lastValuAB - self.states.a.startVgprValu , "ValuAB") # Add as available
      moduleWg.addComment0("init: add vgpr [%u...%u) to pool" % \
                          (self.states.a.startVgprValu, self.states.lastValuAB+self.states.a.startVgprValu))

      self.vgprPool.add(self.states.c.startVgprValu, \
        self.states.c.numVgprValu, "ValuC-Block") # Add as available
      moduleWg.addComment0("init: add vgpr [%u...%u) to pool" % \
                          (self.states.c.startVgprValu, self.states.c.startVgprValu+self.states.c.numVgprValu))

      numAccvgprs = self.states.totalAgprs
      self.agprPool.add(0, numAccvgprs, "ValuC-Block")
      moduleWg.addComment0("init: add agpr [%u...%u) to pool" % \
                          (0, numAccvgprs))

      if kernel["StreamK"] == 0:
        moduleWg.add(self.localReadAddresses(kernel, tPA, tPB, tPM))
        moduleWg.add(self.localWriteAddresses(kernel, tPA, tPB, tPM))

      def waitForArgsToLoad():
        if kernel["ProblemType"]["SupportUserArgs"]:
          moduleWg.add(SWaitCnt(kmcnt=0, comment="wait for %u/%u bytes of kern args" % \
                        (self.argLoader.getOffset() - (self.states.numSgprPreload*4), self.externalArgLoader.getOffset())))
        else:
          moduleWg.add(SWaitCnt(kmcnt=0, comment="wait for %u bytes of kern args" % \
                              (self.argLoader.getOffset() - (self.states.numSgprPreload*4))))
        moduleWg.addModuleAsFlatItems(moduleScaleAB)

      def calculateWG():
        #### calculate numWorkGroup ####
        qReg = self.vgprPool.checkOut(4)
        dReg = qReg + 1
        divReg = qReg + 2
        rReg = qReg + 3
        moduleWg.add(VMovB32(dst=vgpr(divReg), src="MT0", comment="set MT0 into sgpr"))
        moduleWg.add(VMovB32(dst=vgpr(dReg), src=sgpr("SizesFree+0"), comment="set Free0 size"))
        moduleWg.add(vectorUInt32CeilDivideAndRemainder(qReg=qReg, dReg=dReg, divReg=divReg, rReg=rReg, doRemainder=False))
        moduleWg.add(VMovB32(dst=vgpr(divReg), src="MT1", comment="set MT1 into sgpr"))
        moduleWg.add(VMovB32(dst=vgpr(dReg), src=sgpr("SizesFree+1"), comment="set Free1 size"))
        moduleWg.add(VReadfirstlaneB32(dst=sgpr("NumWorkGroups0"), src=vgpr(qReg), comment="set back to numWorkGroup0"))
        moduleWg.add(vectorUInt32CeilDivideAndRemainder(qReg=qReg, dReg=dReg, divReg=divReg, rReg=rReg, doRemainder=False))
        if self.states.archCaps["TransOpWait"]:
          moduleWg.add(SNop(waitState=0, comment="1 wait states"))
        moduleWg.add(VReadfirstlaneB32(dst=sgpr("NumWorkGroups1"), src=vgpr(qReg), comment="set back to numWorkGroup1"))
        self.vgprPool.checkIn(qReg)

      if self.states.numSgprPreload > 0:
        calculateWG()
        waitForArgsToLoad()
      else:
        waitForArgsToLoad()
        calculateWG()

      if not kernel["ProblemType"]["StridedBatched"]:
        with self.allocTmpSgpr(self.states.laneSGPRCount) as tmpSgpr:
          moduleWg.add(self.loadBatchedAddress(kernel, "WorkGroup2", tmpSgpr))
        moduleWg.add(SWaitCnt(kmcnt=0, comment="wait global buffer address ready"))
      elif waitForScaleAB:
        moduleWg.add(SWaitCnt(kmcnt=0, comment="wait for scaleA/B to load"))


      labelMultiGemm = Label(label="MultiGemm", comment="")
      labelMultiGemmEnd = Label(label="MultiGemmEnd", comment="")
      module.add(moduleArgs)
      module.add(moduleRegInit)

      # Reorder WGIDs
      module.add(wgmXCC(self, kernel, tmpSgprNumWorkGroups))

      self.sgprPool.checkIn(tmpSgprNumWorkGroups)
      tmpSgprNumWorkGroups = None
      module.add(SCmpEQU32(src0=sgpr(sgprArgType), src1=0))
      self.sgprPool.checkIn(sgprArgType)
      sgprArgType = None # Cannot be used after this point
      module.add(SCBranchSCC0(labelName=labelMultiGemm.getLabelName()))
      module.add(deepcopy(moduleWg))
      if kernel["StreamK"] == 0:
        module.add(self.remapWgSerial(kernel, earlyStop=False))

      module.add(SBranch(labelName=labelMultiGemmEnd.getLabelName()))
      module.add(labelMultiGemm)

      # Return the sgprs cause after this point preloads ends
      for i in self.states.preloadGuard:
        self.sgprPool.checkIn(i)
      self.states.preloadGuard = []

      numStoreSgprToLoad = self.states.numStoreSgprToLoad
      if kernel["ProblemType"]["UseScaleAB"] == "Scalar":
        if self.states.preloadScaleA:
          numStoreSgprToLoad += 2
        if self.states.preloadScaleB:
          numStoreSgprToLoad += 2
      ###### GroupedGemm  ############
      ######
      # linear search
      ######
      with self.allocTmpSgpr(8, 2) as tmpSgpr:
        tmpSgprM = self.sgprs["SizesFree"]
        tmpSgprN = tmpSgprM+1
        tmpSgprB = tmpSgprN+1
        tmpSgprArgAddress0 = tmpSgpr.idx
        tmpSgpr0 = tmpSgpr.idx + 2
        tmpSgprNumWG0 = tmpSgpr.idx + 4
        tmpSgprNumWG1 = tmpSgpr.idx + 5
        tmpSgprAddrM = tmpSgpr.idx + 6
        tmpSgprAccumTiles = tmpSgpr.idx + 7
        tmpSgprLoopCounter = self.sgprs["NumWorkGroups0"]
        tmpSgprArgOffsett = self.sgprs["NumWorkGroups1"]

        # offset KernArgAddress to address of M
        extValidLabel    = Label(label="IsExternalValid", comment="")
        extValidLabelEnd = Label(label="IsExternalValidEnd", comment="")
        if kernel["ProblemType"]["SupportUserArgs"]:
          module.addComment1("Check if custom structure pointer is null")
          module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
          module.add(SCBranchSCC1(labelName=extValidLabel.getLabelName(), comment="branch if ArgType == 2"))
          if ((kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel')):
            module.add(SMovB32(dst=sgpr(tmpSgprArgOffsett), src=(self.argLoader.getOffset() + (numStoreSgprToLoad * 4) + (self.states.numSgprAddressGSUSync)*4)))
          else:
            module.add(SMovB32(dst=sgpr(tmpSgprArgOffsett), src=(self.argLoader.getOffset() + (numStoreSgprToLoad * 4))))
          module.add(SMulI32(dst=sgpr(tmpSgprAddrM), src0=sgpr(sgprNumsOfGemm), src1=4)) # offset wgTable
          module.add(SMovB64(dst=sgpr(tmpSgprArgAddress0,2), src=sgpr("KernArgAddress",2)))
          module.add(SBranch(extValidLabelEnd.getLabelName()))
          module.add(extValidLabel)
          module.add(SMovB32(dst=sgpr(tmpSgprArgOffsett), src=self.states.userArgsInfo.totalSize))
          module.add(SMovB32(dst=sgpr(tmpSgprAddrM), src=0))
          module.add(SMovB64(dst=sgpr(tmpSgprArgAddress0,2), src=sgpr("KernArgAddress",2)))
          module.add(extValidLabelEnd)
        else:
          module.add(SMovB32(dst=sgpr(tmpSgprArgOffsett), src=(self.argLoader.getOffset() + (numStoreSgprToLoad * 4))))
          module.add(SMulI32(dst=sgpr(tmpSgprAddrM), src0=sgpr(sgprNumsOfGemm), src1=4)) # offset wgTable
          module.add(SMovB64(dst=sgpr(tmpSgprArgAddress0,2), src=sgpr("KernArgAddress",2)))

        # prefetch 1 arg load
        module.addComment1("Grouped Gemm:: prefetch 1 arg load")
        module.add(SMovB32(dst=sgpr(tmpSgprLoopCounter), src=1))
        module.add(SMovB32(dst=sgpr(tmpSgprAccumTiles), src=0))
        module.add(self.argLoader.loadKernArg(tmpSgprM, tmpSgprArgAddress0, sgpr(tmpSgprAddrM), dword=4))
        #module.add(SCmpKEQU32(src=sgpr(sgprNumsOfGemm), simm16=1, comment="if gemm_count is 1?"))
        module.add(self.getSCMPKInstruction("EQU32", sgprNumsOfGemm, 1, comment="if gemm_count is 1?"))
        label_noLoadLoop = Label("wgTable_noLoadLoop", "")
        module.add(SCBranchSCC1(labelName=label_noLoadLoop.getLabelName()))

        # Start to search
        module.addComment1("Grouped Gemm:: accumulate numTiles for each gemm")
        module.addComment0("Grouped Gemm:: loop start")
        label_Loop_gemm_count = Label("Loop_GemmCount", "")
        module.add(label_Loop_gemm_count)
        module.add(SWaitCnt(kmcnt=0))
        # calculate numTiles
        regStateRes = ContinuousRegister(idx=tmpSgpr0, size=2)
        module.add(scalarStaticCeilDivide(qReg=sgpr(tmpSgprNumWG0), dReg=sgpr(tmpSgprM), divisor=kernel["MacroTile0"], tmpSgprRes=regStateRes))
        module.add(scalarStaticCeilDivide(qReg=sgpr(tmpSgprNumWG1), dReg=sgpr(tmpSgprN), divisor=kernel["MacroTile1"], tmpSgprRes=regStateRes))
        # accumulate tiles of each gemm
        module.add(SMulI32(dst=sgpr(tmpSgprNumWG0), src0=sgpr(tmpSgprNumWG0), src1=sgpr(tmpSgprNumWG1)))
        module.add(SMulI32(dst=sgpr(tmpSgprNumWG0), src0=sgpr(tmpSgprNumWG0), src1=sgpr(tmpSgprB)))
        if kernel["GlobalSplitU"] != 0:
          module.add(SAndB32(dst=sgpr(tmpSgprNumWG1), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
          module.add(SMulI32(dst=sgpr(tmpSgprNumWG0), src0=sgpr(tmpSgprNumWG0), src1=sgpr(tmpSgprNumWG1)))
        module.add(SAddU32(dst=sgpr(tmpSgprAccumTiles), src0=sgpr(tmpSgprAccumTiles), src1=sgpr(tmpSgprNumWG0)))
        # check wgIndex >= AccumTiles?
        module.add(SCmpLtU32(src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgprAccumTiles)))
        label_FOUND = Label("FOUND", "")
        module.add(SCBranchSCC1(labelName=label_FOUND.getLabelName()))

        gsuComponent = Component.GSU.find(self)
        module.add(gsuComponent.defineAndResources(self, kernel, tmpSgpr0, tmpSgprM, tmpSgprN, tmpSgprNumWG0, tmpSgprAccumTiles))

        module.add(SAddU32(dst=sgpr(tmpSgprAddrM), src0=sgpr(tmpSgprAddrM), src1=sgpr(tmpSgprArgOffsett)))
        module.add(self.argLoader.loadKernArg(tmpSgprM, tmpSgprArgAddress0, sgpr(tmpSgprAddrM), dword=4))
        module.add(SAddU32(dst=sgpr(tmpSgprLoopCounter), src0=sgpr(tmpSgprLoopCounter), src1=1))
        # loop gemm count
        module.add(SCmpLtU32(src0=sgpr(tmpSgprLoopCounter), src1=sgpr(sgprNumsOfGemm)))
        module.add(SCBranchSCC1(labelName=label_Loop_gemm_count.getLabelName()))

        # noLoadLoop
        module.addComment1("Grouped Gemm:: noLoadLoop")
        module.add(label_noLoadLoop)
        module.add(SWaitCnt(kmcnt=0))
        # calculate numTiles
        regStateRes = ContinuousRegister(idx=tmpSgpr0, size=2)
        module.add(scalarStaticCeilDivide(qReg=sgpr(tmpSgprNumWG0), dReg=sgpr(tmpSgprM), divisor=kernel["MacroTile0"], tmpSgprRes=regStateRes))
        module.add(scalarStaticCeilDivide(qReg=sgpr(tmpSgprNumWG1), dReg=sgpr(tmpSgprN), divisor=kernel["MacroTile1"], tmpSgprRes=regStateRes))
        # accumulate tiles of each gemm
        tmpSgprGSU = tmpSgpr.idx
        module.add(SMulI32(dst=sgpr(tmpSgprNumWG0), src0=sgpr(tmpSgprNumWG0), src1=sgpr(tmpSgprNumWG1)))
        module.add(SMulI32(dst=sgpr(tmpSgprNumWG0), src0=sgpr(tmpSgprNumWG0), src1=sgpr(tmpSgprB)))
        if kernel["GlobalSplitU"] != 0:
          module.add(SAndB32(dst=sgpr(tmpSgprGSU), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
          module.add(SMulI32(dst=sgpr(tmpSgprNumWG0), src0=sgpr(tmpSgprNumWG0), src1=sgpr(tmpSgprGSU)))
        module.add(SAddU32(dst=sgpr(tmpSgprAccumTiles), src0=sgpr(tmpSgprAccumTiles), src1=sgpr(tmpSgprNumWG0)))

        # gemmIndex found
        tmpSgprWgLeft = tmpSgpr.idx
        tmpSgprGemmIdxLeft = tmpSgprWgLeft + 1
        module.addComment1("Grouped Gemm:: gemmIndex found")
        module.add(label_FOUND)
        module.add(SSubU32(dst=sgpr(tmpSgprGemmIdxLeft), src0=sgpr(tmpSgprLoopCounter), src1=1))
        module.add(SSubU32(dst=sgpr(tmpSgprWgLeft), src0=sgpr(tmpSgprAccumTiles), src1=sgpr(tmpSgprNumWG0)))

        ########
        # load arguments of gemm
        ########
        module.add(SSubU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgprWgLeft)))
        extLabel    = Label(label="LoadExternalStruct", comment="")
        extLabelEnd = Label(label="LoadExternalStructEnd", comment="")
        if kernel["ProblemType"]["SupportUserArgs"]:
          module.addComment0("Check if custom structure pointer is null")
          module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
          module.add(SCBranchSCC1(labelName=extLabel.getLabelName(), comment="branch if ArgType == 2"))
        module.addComment1("Grouped Gemm: offset argument address to gemm")
        module.addComment0("Grouped Gemm: offset address from wg_table_start to args_start")
        module.add(SLShiftLeft2AddU32(dst=sgpr("KernArgAddress"), src0=sgpr(sgprNumsOfGemm), src1=sgpr("KernArgAddress")))
        module.add(SAddCU32(dst=sgpr("KernArgAddress+1"), src0=sgpr("KernArgAddress+1"), src1=0))
        module.addComment0("Grouped Gemm: offset address from args_start to gemm_start")
        if ((kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel')):
          module.add(SMulI32(dst=sgpr(tmpSgprGemmIdxLeft), src0=sgpr(tmpSgprGemmIdxLeft),\
                            src1=(self.argLoader.getOffset() + (numStoreSgprToLoad * 4) + (self.states.numSgprAddressGSUSync)*4)))
        else:
          module.add(SMulI32(dst=sgpr(tmpSgprGemmIdxLeft), src0=sgpr(tmpSgprGemmIdxLeft),\
                            src1=(self.argLoader.getOffset() + (numStoreSgprToLoad * 4))))
        module.add(SAddU32(dst=sgpr("KernArgAddress"), src0=sgpr("KernArgAddress"), src1=sgpr(tmpSgprGemmIdxLeft)))
        module.add(SAddCU32(dst=sgpr("KernArgAddress+1"), src0=sgpr("KernArgAddress+1"), src1=0))
        module.add(self.getKernelArgLoadModule(kernel, sgprStart, load, 4))
        if kernel["ProblemType"]["SupportUserArgs"]:
          module.add(SBranch(extLabelEnd.getLabelName()))
          module.add(extLabel)
          module.addComment0("Grouped Gemm: offset address from args_start to gemm_start")
          # Currently a magic number cause the structure is fixed, should the structure gen by python so we can know the size?
          module.add(SMulI32(dst=sgpr(tmpSgprGemmIdxLeft), src0=sgpr(tmpSgprGemmIdxLeft),src1=self.states.userArgsInfo.totalSize))
          module.add(SAddU32(dst=sgpr("KernArgAddress"), src0=sgpr("KernArgAddress"), src1=sgpr(tmpSgprGemmIdxLeft)))
          module.add(SAddCU32(dst=sgpr("KernArgAddress+1"), src0=sgpr("KernArgAddress+1"), src1=0))
          moduleExternalArgs = Module("Load external Arguments")
        # Here alpha and beta in user args are fixed sizes, so we need to exclude beta and read it with a different offset
          load = load - self.states.numSgprBeta
          moduleExternalArgs.addModuleAsFlatItems(self.externalArgLoader.loadAllKernArg(sgprStart, "KernArgAddress", load, 4))
          offset = self.externalArgLoader.getOffset() + self.states.bpr * (self.states.userArgsInfo.alphaMaxRegisterSize - self.states.numSgprAlpha)
          self.externalArgLoader.setOffset(offset)
          moduleExternalArgs.addComment("Read Beta")
          moduleExternalArgs.addModuleAsFlatItems(self.externalArgLoader.loadAllKernArg(self.sgprs["Beta"], "KernArgAddress", self.states.numSgprBeta))
          offset = self.externalArgLoader.getOffset() + self.states.bpr * (self.states.userArgsInfo.betaMaxRegisterSize - self.states.numSgprBeta)
          if kernel["ProblemType"]["UseScaleAB"] == "Scalar":
            sgprOffset = self.externalArgLoader.getOffset()
            for preloadScale, name in zip([self.states.preloadScaleA, self.states.preloadScaleB], ['A','B']):
              if preloadScale:
                moduleExternalArgs.add(self.externalArgLoader.loadKernArg("AddressScale%s"%name, "KernArgAddress", sgprOffset=hex(sgprOffset), dword=2))
              sgprOffset += self.states.userArgsInfo.scaleASize
          self.externalArgLoader.setOffset(offset)
          module.add(moduleExternalArgs)
          module.add(extLabelEnd)

      # Update label
      labels = []
      for inst in moduleWg.items():
        if isinstance(inst, Label):
          self.labels.getNameInc(inst.label)
          labels.append([inst.label, self.labels.getNameInc(inst.label)])
          inst.label = labels[-1][1]
      for inst in moduleWg.items():
        if isinstance(inst, BranchInstruction):
          removeIdx = -1
          for idx, label in enumerate(labels):
            if inst.labelName == Label(label=label[0],comment="").getLabelName():
              inst.labelName = Label(label=label[1],comment="").getLabelName()
              removeIdx = idx
          if removeIdx != -1:
            del labels[removeIdx]
      module.add(moduleWg)

      earlyReturnModule = Module("Early stop if N(SizeFreeJ) == 0")
      earlyReturnModule.addComment1("Early stop if N(SizeFreeJ) == 0")
      earlyReturnModule.add(SCmpEQU32(sgpr("SizeJ"), 0))
      earlyReturnLabel = Label("EarlyStop_if_N_is_0", "")
      noEarlyReturnLabel = Label("NoEarlyStop_N0", "")
      earlyReturnModule.add(SCBranchSCC0(noEarlyReturnLabel.getLabelName()))
      earlyReturnModule.add(earlyReturnLabel)
      earlyReturnModule.add(SEndpgm())
      earlyReturnModule.add(noEarlyReturnLabel)
      module.add(earlyReturnModule)
      if kernel["StreamK"] == 0:
        module.add(self.remapWgSerial(kernel))
      module.addSpaceLine()
      module.add(labelMultiGemmEnd)

    # CheckIn temp sgprs
    if sgprNumsOfGemm:
      self.sgprPool.checkIn(sgprNumsOfGemm)
      sgprNumsOfGemm = None

    # define the rest of sgprs
    module.addModuleAsFlatItems(self.defineVariableSgprs(kernel))

    needPackK16  = False
    needPackK8Lw = False
    needPackK8Hi = False

    if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
      if self.states.lrvwTileA > 1 or self.states.lrvwTileB > 1:
        needPackK16 = True
      if self.states.lrvwTileMetadata > 1:
        needPackK8Lw = True
      if self.states.lrvwTileMetadata > 2:
        needPackK8Hi = True
    elif kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat():
      if self.states.lrvwTileA > 1 or self.states.lrvwTileB > 1 or self.states.lrvwTileMetadata > 1:
        needPackK8Lw = True
      if self.states.lrvwTileA > 2 or self.states.lrvwTileB > 2 or self.states.lrvwTileMetadata > 2:
        needPackK8Hi = True

    # need extra sgpr for packing metadata
    tPackM = "M" if needPackK16 and needPackK8Lw else ""
    if needPackK16:
        module.add(SMovB32(dst=sgpr("PackKForV0"), src="0x05040100", comment=""))
        module.add(SMovB32(dst=sgpr("PackKForV1"), src="0x07060302", comment=""))
    if needPackK8Lw:
        module.add(SMovB32(dst=sgpr("PackKFor%sV0"%tPackM), src="0x0c0c0400", comment=""))
        module.add(SMovB32(dst=sgpr("PackKFor%sV1"%tPackM), src="0x0c0c0501", comment=""))
    if needPackK8Hi:
        module.add(SMovB32(dst=sgpr("PackKFor%sV2"%tPackM), src="0x0c0c0602", comment=""))
        module.add(SMovB32(dst=sgpr("PackKFor%sV3"%tPackM), src="0x0c0c0703", comment=""))

    # self.states.groOffsetInMacroTile == 1 case, subtract pre-pad here
    if self.states.groOffsetInMacroTile:
      prePad = self.states.srdShiftLeft["A"] * tPA["bpeGR"] # leave room in case we have to pointer shift
      module.add(SSubU32(dst=sgpr("AddressA+0"), src0=sgpr("AddressA+0"), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
      module.add(SSubBU32(dst=sgpr("AddressA+1"), src0=sgpr("AddressA+1"), src1=0, comment="pre-pad to make room for possible pointer shift"))
      prePad = self.states.srdShiftLeft["B"] * tPB["bpeGR"] # leave room in case we have to pointer shift
      module.add(SSubU32(dst=sgpr("AddressB+0"), src0=sgpr("AddressB+0"), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
      module.add(SSubBU32(dst=sgpr("AddressB+1"), src0=sgpr("AddressB+1"), src1=0, comment="pre-pad to make room for possible pointer shift"))
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        prePad = self.states.srdShiftLeft["Metadata"] * tPM["bpe"] # leave room in case we have to pointer shift
        module.add(SSubU32(dst=sgpr("AddressMetadata+0"), src0=sgpr("AddressMetadata+0"), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
        module.add(SSubBU32(dst=sgpr("AddressMetadata+1"), src0=sgpr("AddressMetadata+1"), src1=0, comment="pre-pad to make room for possible pointer shift"))

    # Check alpha == 0, is done before kernel body
    # so if alpha/beta=Half, they haven't been converted to f32
    # This means we can use ComputeDataType as AlphaType (even <h,h,h,h,"h,h"> +"HPA")
    if self.do["ApplyAlpha"]:
      if self.states.useBias == DataDirection.WRITE and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
        # Temporarily turn off "Set summation dim=0 if Alpha == 0" cause reduction.
        pass
      else:
        module.addComment1("Short circuit condition if Alpha == 0, then sumDims=0")
        endCheckLabel = Label("AlphaNonZero", "")
        module.add(BranchIfNotZero("Alpha", kernel["ProblemType"]["ComputeDataType"].toEnum(), endCheckLabel))

        # Conditional set summation dimensions to 0 on SCC==1
        for i in range(0, self.states.numSgprSizesSum):
          module.add(SMovB32(dst=sgpr("SizesSum+%u"%(i)), src=0, comment="Set summation dim=0 if Alpha == 0"))
        # Short circuit for stream-k is handled in the stream-k component by skipping partial tiles and setting loop counter to 0

        # Jump here if alpha is non-zero
        module.add(endCheckLabel)

    if kernel["MagicDivAlg"]==2:
      for idxChar in sorted(set(kernel["PackedC0IdxChars"][:-1] + kernel["PackedC1IdxChars"][:-1])):
          module.add(SLShiftRightB32(dst=sgpr("MagicAbitSize%s"%idxChar), src=sgpr("MagicShiftSize%s"%idxChar), shiftHex=31, comment="extract abit"))
          module.add(SAndB32(dst=sgpr("MagicShiftSize%s"%idxChar), src0=sgpr("MagicShiftSize%s"%idxChar), src1=hex(0x7fffffff), comment="remove abit"))

    ########################################
    # Debug Buffer
    if self.debugConfig.debugKernel:
      module.addComment1("Debug Buffer")

      # nwg0 FIXME use NumWorkGroups0
      nwg0 = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOutAligned(2, 2)
      tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
      module.addComment("nwg0 = (size%s + MT%s - 1) / MT%s;" \
          % (self.states.tileChar0, self.states.tileChar0, self.states.tileChar0))
      module.add(VMovB32(dst=vgpr(tmpVgpr), src=hex(kernel["MacroTile0"]-1), comment="MT0-1"))
      module.add(VAddCOU32(dst=vgpr(nwg0), dst1=VCC(), src0=sgpr("SizesFree+0"), \
          src1=vgpr(tmpVgpr), comment="%s = size0+MT0-1"%vgpr(nwg0)))
      module.add(vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgprRes))
      self.vgprPool.checkIn(tmpVgpr)
      self.nipt = 16 # num integers per thread
      v = self.vgprPool.checkOut(3)
      module.add(VMovB32(dst=vgpr(v), src=sgpr("WorkGroup0"), comment="%s=wg0"%vgpr(v) ))
      module.add(VMovB32(dst=vgpr(v+1), src=sgpr("WorkGroup1"), comment="%s=wg1"%vgpr(v+1) ))
      module.add(VMulLOU32(dst=vgpr(v+1), src0=vgpr(v+1), src1=vgpr(nwg0), \
          comment="%s=wg1*nwg0"%vgpr(v+1) ))
      module.add(VAddCOU32(dst=vgpr(v), dst1=VCC(), src0=vgpr(v), src1=vgpr(v+1), \
          comment="%s=wg1*nwg0+wg0"%vgpr(v) ))
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(vectorStaticMultiply(vgpr(v), vgpr(v), kernel["NumThreads"], tmpSgprInfo))
      module.add(VAddCOU32(dst=vgpr(v), dst1=VCC(), src0=vgpr(v), src1=vgpr("Serial"), \
          comment="%s=tid+NT*(wg1*nwg0+wg0)=serial"%vgpr(v) ))
      module.add(VMulLOU32(dst=vgpr(v), src0=hex(self.nipt*4), src1=vgpr(v), \
          comment="%s=serial*nipt*4"%vgpr(v) ))
      module.add(VMovB32(dst=vgpr(v+1), src=0))
      module.add(VAddCOU32(dst=vgpr("AddressDbg"), dst1=VCC(), src0=sgpr("AddressDbg"), src1=vgpr(v), \
          comment="%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") ))
      module.add(VMovB32(dst=vgpr(v+2), src=sgpr("AddressDbg+1"), comment="%s=AddressD1"%vgpr(v+2) ))
      module.add(VAddCCOU32(dst=vgpr("AddressDbg+1"), dst1=VCC(), src0=vgpr(v+2), \
          src1=vgpr(v+1), src2=VCC(), comment="%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") ))
      module.add(SMovB32(dst=sgpr("DebugKernelItems"), src=0))
      self.vgprPool.checkIn(v)
      self.vgprPool.checkIn(nwg0)

    if self.db["InitLds"]:
      tmp = ContinuousRegister(idx=self.vgprPool.checkOut(2), size=2)
      module.add(DSInit(tmp, kernel["NumThreads"], kernel["LdsNumBytes"], self.consts.initLdsValue))
      self.vgprPool.checkIn(tmp.idx)

    return module

  def extractPackedCoord1ToRowStart(self, kernel, packedC1, packedCoordVgpr, storeChar):
    if kernel["ProblemType"]["UseE"] and ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0):
      printExit("extractPackedCoord1ToRowStart doe not support with output E.")
    # calculate packed rowStart vgpr
    # vgprTmp assignments:
    #   - tmp+0 is the incoming packed coordinate 1, used on replay too
    #   - tmp+1 is DIV output
    #   - tmp+2 is scratch
    #   - tmp+3 holds thread rowStart free1 offset
    module = Module("extractPackedCoord1ToRowStart")
    tmpV0 = self.vgprPool.checkOut(4)
    tmpV1 = tmpV0 + 1
    tmpV2 = tmpV0 + 2
    tmpV3 = tmpV0 + 3

    module.add(VMovB32(dst=vgpr(tmpV0), src=vgpr(packedCoordVgpr),  comment="copy coord1 then unpack"))
    for i,idx in enumerate(packedC1[:-1]):
      idxChar= INDEX_CHARS[idx]
      module.addComment0("extract %s"%self.sizeRef(idx))
      module.add(MacroInstruction(name="V_MAGIC_DIV", \
                args=[tmpV1, vgpr(tmpV0), sgpr("MagicNumberSize%s"%idxChar), \
                sgpr("MagicShiftSize%s"%idxChar), (sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0")]))
      module.add(VMulLOU32(dst=vgpr(tmpV2), src0=vgpr(tmpV1), src1=self.sizeRef(idx), comment="remainder part 1"))
      module.add(VSubU32(dst=vgpr(tmpV2), src0=vgpr(tmpV0), src1=vgpr(tmpV2), comment="remainder part 2"))
      if i==0:
        module.add(VMulLOU32(dst=vgpr(tmpV3), src0=vgpr(tmpV2), \
                  src1=self.strideRef(storeChar, idx), comment="addrCalc <- scaled extracted dim"))
      else:
        module.add(VMulLOU32(dst=vgpr(tmpV2), src0=vgpr(tmpV2), \
                  src1=self.strideRef(storeChar, idx), comment="scale extracted dim"))
        module.add(VAddU32(dst=vgpr(tmpV3), src0=vgpr(tmpV3), \
                  src1=vgpr(tmpV2), comment="addrCalc += scaled extracted dim "))

      if i < len(packedC1)-2:
        module.add(VMovB32(dst=vgpr(tmpV0), src=vgpr(tmpV1), \
                  comment="Copy remaining bits for next divide"))

    module.addComment0("extract final %s"%self.sizeRef(packedC1[-1]))
    module.add(VMulLOU32(dst=vgpr(tmpV2), src0=vgpr(tmpV1), \
              src1=self.strideRef(storeChar, packedC1[-1]), comment="scale final extracted dim"))
    module.add(VAddU32(dst=vgpr(self.vgprs.coutRowPtrD), src0=vgpr(tmpV3), \
              src1=vgpr(tmpV2), comment="rowStart += scaled extracted dim "))

    self.vgprPool.checkIn(tmpV0)
    return module


  ##############################################################################
  # Global Read Addresses: WorkGroup
  ##############################################################################
  def graWorkGroup(self, kernel, tPA, tPB):
    module = Module("graWorkGroup")
    module.addComment0("graWorkGroup mapping")

    skComponent = Component.StreamK.find(self)
    module.add(skComponent.graWorkGroup(self, kernel, tPA, tPB))

    gsuComponent = Component.GSU.find(self)
    module.add(gsuComponent.graWorkGroup(self, kernel))

    # Start of WGM algos
    sgprWGM = "WGM"
    if len(kernel["SpaceFillingAlgo"]):
      module.addComment0("Start of Generic WGM algo")
      self.states.WGMTransformLevels = len(kernel["SpaceFillingAlgo"])
      module.add(SpaceFillingCurveWalk(self, kernel, sgprWGM))
    else:
      module.add(DefaultWGM(self, kernel, sgprWGM))

    return module




  def graMetadataTileAssignment(self, kernel, tP):
    module = Module("graMetadataTileAssignment")
    module.addComment0("calculate metadata gra tile assignment")

    if kernel["DirectToVgprSparseMetadata"]:

      # alloc vgpr
      wReg    = self.vgprPool.checkOut(1,"wReg") # quotient
      tReg    = self.vgprPool.checkOut(1,"tReg") # remainder
      tmpVgpr = self.vgprPool.checkOutAligned(2,2,"tmpVgpr")
      dummy   = self.vgprPool.checkOut(1,"dummy")

      tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

      # get constant parameter

      waveWidth        = kernel["WavefrontSize"]

      # parameter for get each type index
      num1DBlocks      = kernel["MatrixInstBM"]
      num1DWaves       = kernel["MIWaveGroup"][0]
      dividedForWaveId = waveWidth

      # strider for each type of index
      offsetWave       = kernel["MatrixInstM"] * num1DBlocks

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        # tile offset
        module.add(vectorStaticRemainder(dummy, tReg, "Serial", waveWidth, tmpVgprRes, tmpSgprInfo, \
            comment="0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
        module.add(vectorStaticRemainder(dummy, tReg, tReg, kernel["MatrixInstM"], tmpVgprRes, tmpSgprInfo, \
            comment="1. tile offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstM"]))

        # wave offset
        if num1DWaves > 1:
          module.add(vectorStaticDivide(wReg, "Serial", dividedForWaveId, tmpVgprRes, \
              "2. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
          module.add(vectorStaticRemainder(dummy, wReg, wReg, num1DWaves, tmpVgprRes, tmpSgprInfo, \
              "2. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
          module.add(vectorStaticMultiplyAdd(vgpr("GlobalReadOffsetMetadata"), vgpr(wReg), offsetWave, vgpr(tReg), tmpSgprInfo, \
                                       "2. wave offset in M dimen: wOffset = wtid0 * W0 offset(%u); 2. tile coord = tileOffset + wOffset" % offsetWave))
        else:
          module.add(VMovB32(vgpr("GlobalReadOffsetMetadata"), vgpr(tReg), \
              comment="2. tile coord = tileOffset"))

        for graIdx in range (1, kernel["MIWaveTile"][0]):
          MIWaveGroup0ShapeSize = kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0]
          vgprGro = "GlobalReadOffsetMetadata+%u"%(graIdx)
          module.add(VMovB32(vgpr(vgprGro), hex(MIWaveGroup0ShapeSize*graIdx), comment="7. coord offset of WaveTile %u"%graIdx))
          module.add(VAddU32(vgpr(vgprGro), vgpr(vgprGro), vgpr("GlobalReadOffsetMetadata"), \
              "3. final coord0: tile coord + waveTile coord"))

      # release register
      self.vgprPool.checkIn(tReg)
      self.vgprPool.checkIn(wReg)
      self.vgprPool.checkIn(tmpVgpr)
      self.vgprPool.checkIn(dummy)

    return module

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  # global read addresses: tile offset assignment (message from .s)
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    module = Module("graTileAssignment")
    tc = tP["tensorChar"]
    tReg =  tP["gpr"]["lwoT"]

    module.addComment0("graTileAssignment%s = %s" % (tc, vgpr(tReg)))

    if self.states.groOffsetInMacroTile:
      tReg2 = tReg
      # treg2 and treg same register and value - we store the 'static'
      # part of the address calculation in the SRD to maximize the
      # range of the 32-bit GRO
    else:
      tReg2 = self.vgprPool.checkOut(1, 'treg2', self.states.preventVgprOverflowDuringNewTile)

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      if not self.states.groOffsetInMacroTile:
        tmpVgpr = self.vgprPool.checkOut(1, 'graTA vgpr', self.states.preventVgprOverflowDuringNewTile)
        # Buffer Load will set the SRD to start of the MacroTile
        # So don't add the static wg-related component here - save for later.
        module.add(vectorStaticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]], tmpSgprInfo))  # workgroup
        module.add(VAddCOU32(dst=vgpr(tReg2), dst1=VCC(), src0=vgpr(tmpVgpr), \
            src1=vgpr(tReg), comment="gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
            % (tc, tOpStr, divisorName, tc, tc) ))
        self.vgprPool.checkIn(tmpVgpr)

    tP["gpr"]["tReg"] = tReg2

    return Module("graTileAssignment (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Unroll Assignment
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    module = Module("graUnrollAssignment")
    # note groOffsetInMacroTile rolls these into SRD so don't change here:
    if not self.states.groOffsetInMacroTile:
      gsuOffset = self.vgprPool.checkOut(1, "gsuOffset", self.states.preventVgprOverflowDuringNewTile)
      module.add(VMovB32(dst=vgpr(gsuOffset), src=sgpr("GSUSumIdx"), comment="=gsuSumIdx"))

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        # graUnrollAssignment += gsuSumIdx*DepthU
        module.add(vectorStaticMultiply(vgpr(gsuOffset), vgpr(gsuOffset), kernel["DepthU"], tmpSgprInfo))

      module.add(VAddCOU32(dst=vgpr(tP["gpr"]["uReg"]), dst1=VCC(), \
          src0=vgpr(gsuOffset), src1=vgpr(tP["gpr"]["uReg"]), \
          comment="graUnrollAssignment += gsuOffset"))
      self.vgprPool.checkIn(gsuOffset)
    else:
      module.addComment0(str(vgpr(tP["gpr"]["uReg"])))

    return Module("graUnrollAssignment (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self):
    module = Module("graOtherFreeAssignments")
    module.addComment0("s[sgprWorkGroup2]")
    return module

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    module = Module("graOtherSummationAssignments")
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      module.add(ValueSet(name="globalReadOffsetA%s" % self.states.indexChars[index], value=0))
      module.add(ValueSet(name="globalReadOffsetB%s" % self.states.indexChars[index], value=0))
    return module

  def alignTo(self, dstBase: Union[int, str], srcBase: Union[int, str], alignment: int) -> Module:
    assert (alignment & (alignment - 1)) == 0 and f"Alignment must be power of two"
    module = Module()
    module.addComment(f"Align to {alignment}")
    module.add(SAddU32(sgpr(dstBase), sgpr(srcBase), alignment-1))
    module.add(SLShiftRightB32(dst=sgpr(dstBase), src=sgpr(dstBase), shiftHex=log2(alignment)))
    module.add(SLShiftLeftB32(dst=sgpr(dstBase), src=sgpr(dstBase), shiftHex=log2(alignment)))
    return module

  def setTo(self, dstBase: Union[int, str], srcBase: Union[int, str]) -> Module:
    module = Module()
    module.add(SMovB32(dst=sgpr(dstBase), src=sgpr(srcBase), comment="Set to %s"%(srcBase)))
    return module

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP, margin=-1):
    module = Module("graTileOffsets")
    tc = tP["tensorChar"]
    tP["vgprPackedOffsets"] = None
    tP["vgprTileOffsetsCheckOut"] = False
    tP["numVgprTileOffsets"] = 0
    if kernel["_UseSgprForGRO"]:
      # Let the vgprTileOffsets checkin handle tReg later since these are same vgpr
      tP["vgprTileOffsets"] = tP["gpr"]["tReg"]
    else:
      numTileOffsets = tP["nrt"]
      tP["vgprTileOffsets"] = self.vgprPool.checkOut(numTileOffsets, "vgprTileOffsets", self.states.preventVgprOverflowDuringNewTile)
      tP["vgprTileOffsetsCheckOut"] = True
      v = tP["vgprTileOffsets"]
      numExtraPackedOffsetsPerTile = len(tP["PackedIndices"])-1
      if numExtraPackedOffsetsPerTile:
        tP["vgprPackedOffsets"] = self.vgprPool.checkOut(numExtraPackedOffsetsPerTile * numTileOffsets, "vgprPackedOffsets", self.states.preventVgprOverflowDuringNewTile)
      strideIdx = tP["lsc"] if tP["tlu"] else tP["lsp"]
      stride = kernel[strideIdx]
      # adjustment for DirectToVgpr + tlu=False + VW > 1 case
      strideInterleave = False
      if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%c"%tc] and (not tP["tlu"]) and kernel["VectorWidth%s"%tc] > 1:
        strideInterleave = True
        stride = stride * kernel["VectorWidth%s"%tc] - (kernel["VectorWidth%s"%tc] - 1)
        strideMask = (kernel["VectorWidth%s"%tc] - 1)

      if tP["isM"] and not margin == -1:
        # margin is the number of how many continuous element need to be read
        # shift metadata global read offset to align A's global read offset.
        module.add(VLShiftRightB32(dst=vgpr(v), shiftHex=hex(log2(margin)), src=vgpr(tP["gpr"]["tReg"]), comment="gro%s%s_%u /= %d"%(tP["tensorChar"], tP["tileChar"], 0, margin)))
        module.add(VLShiftLeftB32(dst=vgpr(v), shiftHex=hex(log2(margin)), src=vgpr(v), comment="gro%s%s_%u *= %d"%(tP["tensorChar"], tP["tileChar"], 0, margin)))
      else:
        if not tP["isSwizzled"]:
          module.add(VMovB32(dst=vgpr(v), src=vgpr(tP["gpr"]["tReg"]), comment="gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], 0) ))
        else:
          lsu = kernel["LocalSplitU"] # localSplitU
          if tP["isA"]:
            WvG_MorN = kernel["MIWaveGroup"][0]
            swzMorN = kernel["MatrixInstM"]
          elif tP["isB"]:
            WvG_MorN = kernel["MIWaveGroup"][1]
            swzMorN = kernel["MatrixInstN"]
          swzStride = tP["swizzleK"]
          swzBlockSize = swzMorN * swzStride
          vw = kernel[f"VectorWidth{tc}"]
          kPack = tP["swizzlePackK"]
          laneSize = int(kernel["MatrixInstK"] / 4) * kPack  # the size of one swizzle's lane

          with self.allocTmpSgpr(2) as tmpSgprInfo:
            swzBlkVWSizeSgpr = tmpSgprInfo.idx
            swzStridePerWave = swzBlkVWSizeSgpr + 1

            module.addComment0("SWZ-%s: calculation of gro%s%s_%u"%(tc, tP["tensorChar"], tP["tileChar"], 0))
            module.add(VMovB32(dst=vgpr(v), src=vgpr(tP["gpr"]["tReg"])))
            swzBlkVWSizeVgpr = self.vgprPool.checkOut(1)
            module.add(SMovB32(dst=sgpr(swzBlkVWSizeSgpr), src=hex(swzBlockSize * vw), \
              comment="swizzled block * VW = (MI_MorN(%u) * MI_K(%u) * kPack) * VW(%u)" %(swzMorN, int(swzStride / kPack), vw)))
            module.add(VMovB32(dst=vgpr(swzBlkVWSizeVgpr), src=sgpr(swzBlkVWSizeSgpr)))
            module.add(VMulU32U24(dst=vgpr(v), src0=vgpr(v), src1=vgpr(swzBlkVWSizeVgpr)))

            # LSU part
            if lsu > 1:
              tmpVgprRes = None
              wave_id    = self.vgprPool.checkOut(1) # quotient
              # constant
              du         = kernel["DepthU"]
              lsuStride  = du // lsu
              numWaves   = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
              # codes
              module.add(vectorStaticDivide(wave_id, "Serial", kernel["WavefrontSize"] * numWaves, tmpVgprRes, comment="LSU offset: Get LSU wave_id"))
              module.add(VLShiftLeftB32(dst=vgpr(wave_id), shiftHex=hex(log2(lsuStride*swzMorN)), src=vgpr(wave_id), comment="LSU offset: LSU_wave_id*MI_MorN(%u)*lsuStrideK(%u)"%(swzMorN, lsuStride)))
              module.add(VAddU32(dst=vgpr(v), src0=vgpr(wave_id), src1=vgpr(v), comment="tileOffset += LSU offset"))
              self.vgprPool.checkIn(wave_id)

            # don't emit the code if we don't need to do more than one nrt
            if tP["nrt"] > 1:
              module.addComment0("swzStridePerWave = (number of swizzle block in K) * WaveGroup_MorN")
              module.addComment(f"Align to {swzStride}")
              module.add(SAddU32(sgpr(swzStridePerWave), sgpr("SizesSum"), swzStride-1)),
              module.add(SLShiftRightB32(dst=sgpr(swzStridePerWave), src=sgpr(swzStridePerWave), shiftHex=hex(log2(swzStride)),
                                        comment="numKr = DimK / swizzleK")),
              module.add(SMulI32(dst=sgpr(swzStridePerWave), src0=hex(WvG_MorN), src1=sgpr(swzStridePerWave),
                                comment="numKr *= MI_WaveGroup, wave-M (SWZ-A) or wave-N (SWZ-B)"))

              module.addComment0("swzBlkVWOffset = swzBlkWvGSize - laneSize * (VW - 1)")
              module.add(SMulI32(dst=sgpr(swzBlkVWSizeSgpr), src0=sgpr(swzStridePerWave), src1=sgpr(swzBlkVWSizeSgpr), \
                comment="swzBlkWvGSize = numKr * (swzBlockSize * VW)"))
              module.add(SSubU32(dst=sgpr(swzBlkVWSizeSgpr), src0=sgpr(swzBlkVWSizeSgpr), src1=(laneSize * (vw - 1)), \
                comment="swzBlkVWOffset = swzBlkWvGSize - laneSize * (VW - 1)"))
              module.add(VMovB32(dst=vgpr(swzBlkVWSizeVgpr), src=sgpr(swzBlkVWSizeSgpr)) )

      for l in range(1, tP["nrt"]):
        strideValue = stride
        if strideInterleave and (l & strideMask) != 0:
          strideValue = 1
        if not tP["isSwizzled"]:
          module.add(VAddCOU32(dst=vgpr(v+l), dst1=VCC(), src0=strideValue, \
            src1=vgpr(v+l-1), comment="gro%s%s_%u += %s"%(tP["tensorChar"], tP["tileChar"], l, strideIdx) ))
        # swizzle
        else:
          # VW > 1
          if (strideInterleave and (l & strideMask) != 0):
            module.add(VAddCOU32(dst=vgpr(v+l), dst1=VCC(), src0=laneSize, \
              src1=vgpr(v+l-1), comment="SWZ-%s: gro%s%s_%u"%(tc, tP["tensorChar"], tP["tileChar"], l) ))
          # VW == 1
          else:
            module.add(VAddCOU32(dst=vgpr(v+l), dst1=VCC(), src0=vgpr(swzBlkVWSizeVgpr), \
              src1=vgpr(v+l-1), comment="SWZ-%s: gro%s%s_%u"%(tc, tP["tensorChar"], tP["tileChar"], l) ))
      if tP["isSwizzled"]:
        self.vgprPool.checkIn(swzBlkVWSizeVgpr)

      # TODO- check for swizzle
      if numExtraPackedOffsetsPerTile:
        tmpV = self.vgprPool.checkOutAligned(2,2,"packTmp", self.states.preventVgprOverflowDuringNewTile)

        for l in range(0, tP["nrt"]):
          lastGroVgpr = vgpr(v+l)
          lastGroIdx = tP["PackedIndices"][0]
          module.addSpaceLine()
          for p in range(0, numExtraPackedOffsetsPerTile):
            groIdx  = tP["PackedIndices"][p+1]
            groChar = INDEX_CHARS[tP["PackedIndices"][p+1]]
            groVgpr = vgpr(tP["vgprPackedOffsets"] + l*numExtraPackedOffsetsPerTile + p)
            pChar = INDEX_CHARS[tP["PackedIndices"][p]]
            module.add(MacroInstruction(name="V_MAGIC_DIV", \
                args=[tmpV, lastGroVgpr, sgpr("MagicNumberSize%s"%pChar), \
                sgpr("MagicShiftSize%s"%pChar), (sgpr("MagicAbitSize%s"%pChar) if kernel["MagicDivAlg"]==2 else "0")] ))
            module.add(VMovB32(dst=groVgpr, src=vgpr(tmpV), comment="extract gro%s%s_%u (%s)"%(tc,groChar,l,groVgpr)))
            module.add(VMulLOU32(dst=vgpr(tmpV), src0=groVgpr, src1=sgpr("SizesFree+%u"%lastGroIdx), comment="remainder part 1"))
            module.add(VSubU32(dst=lastGroVgpr, src0=lastGroVgpr, src1=vgpr(tmpV), \
                comment="remove extracted bits from gro%s%s_%u (%s)"%(tc, INDEX_CHARS[lastGroIdx], l, lastGroVgpr)))
            lastGroVgpr = groVgpr
            lastGroIdx = groIdx
        self.vgprPool.checkIn(tmpV)

      # groOffsetInMacroTile uses same register for both of these, don't free it here:
      if tP["gpr"]["lwoT"] != tP["gpr"]["tReg"] :
        self.vgprPool.checkIn(tP["gpr"]["tReg"])
        tP["gpr"]["tReg"] = None
    return Module("graTileOffsets (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    module = Module("graUnrollOffsets")
    tc = tP["tensorChar"]
    if kernel["_UseSgprForGRO"]:
      tP["gpr"]["unrollOffsets"] = tP["gpr"]["uReg"]
    # swizzle
    elif tP["isSwizzled"]:
      numUnrollOffsets = tP["nru"]
      tP["gpr"]["unrollOffsets"] = self.vgprPool.checkOut(numUnrollOffsets, "unrollOffsets", self.states.preventVgprOverflowDuringNewTile)
      v = tP["gpr"]["unrollOffsets"]

      # Please note that the following description uses swizzle A as an example.
      # The notation of A and M should be replaced with B and N when referring to swizzle B.
      #
      # swizzle offset formula = F(row, col) + swzBlkOffsetM(tid)    (unit: element)
      #   row, col: local index in a swizzled block
      #     row = (tid / swzM) * laneSize
      #     col = (tid % (swzM / VWA)) * VWA
      #       laneSize: number of elements loaded by a single lane.
      #   F(row, col): swizzled offset when vector width is 1.
      #     F(row, col) = row % laneSize + col * laneSize + (row / laneSize) * swzM * laneSize
      #   swzBlkOffsetM(tid) = [tid / (swzM / VWA)] * swzSizeK
      #     swzSizeK = numSwzBlkK * numElmInSwzBlk
      #       numSwzBlkK: number of swizzled block in K dimension.
      #       numElmInSwzBlk: number of element in a swizzled block.

      if tP["isA"]:
        swzMorN = kernel["MatrixInstM"]
      elif tP["isB"]:
        swzMorN = kernel["MatrixInstN"]
      swzStride = tP["swizzleK"]
      vw = kernel[f"VectorWidth{tc}"]
      kPack = tP["swizzlePackK"]
      laneSize = int(kernel["MatrixInstK"] / 4) * kPack  # the size of one swizzle's lane
      numElmInSwzBlk = swzMorN * swzStride

      # Calculate local index in a swizzled block
      row = self.vgprPool.checkOut(1)
      col = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOut(1)
      module.addComment0("SWZ-%s: r = swzRow = (tid / swzMorN(%u)) * laneSize(%u)"%(tc, swzMorN, laneSize))
      module.add(VAndB32(dst=vgpr(v), src0=vgpr("Serial"), src1=(kernel["WavefrontSize"]-1), comment="tid"))
      module.add(VLShiftRightB32(dst=vgpr(row), shiftHex=hex(int(log(swzMorN, 2))), src=vgpr(v)))
      module.add(VLShiftLeftB32(dst=vgpr(row), shiftHex=hex(int(log(laneSize, 2))), src=vgpr(row)))
      module.addComment0("SWZ-%s: c = swzCol = [tid mod (swzMorN(%u) / VW(%u))] * VW(%u)"%(tc, swzMorN, vw, vw))
      module.add(VAndB32(dst=vgpr(col), src0=vgpr(v), src1=int(swzMorN/vw)-1))
      if vw > 1:
        module.add(VLShiftLeftB32(dst=vgpr(col), shiftHex=hex(int(log(vw, 2))), src=vgpr(col)))

      module.addComment0("SWZ-%s: F = r mod laneSize + c * laneSize + (r/laneSize) * swzMorN * laneSize"%tc)
      module.add(VLShiftRightB32(dst=vgpr(tmpVgpr), shiftHex=hex(int(log(laneSize, 2))), src=vgpr(row), comment="tmp = r / laneSize"))
      module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(int(log(swzMorN * laneSize, 2))), src=vgpr(tmpVgpr), comment="tmp *= swzMorN * laneSize"))
      module.add(VMadU32U24(dst=vgpr(tmpVgpr), src0=vgpr(col), src1=laneSize, src2=vgpr(tmpVgpr), comment="tmp += c * laneSize"))
      module.add(VAndB32(dst=vgpr(row), src0=vgpr(row), src1=laneSize-1, comment="tmp2 = row mod laneSize"))
      if vw == 1:
        module.add(VAddU32(dst=vgpr(v), src0=vgpr(tmpVgpr), src1=vgpr(row), comment="offset = tmp + tmp2"))

      else:
        fVgpr = self.vgprPool.checkOut(1)
        module.add(VAddU32(dst=vgpr(fVgpr), src0=vgpr(tmpVgpr), src1=vgpr(row), comment="F = tmp + tmp2"))

        module.addComment0("SWZ-%s: swzSizeK = numSwzBlkK * numElmInSwzBlk"%tc)
        swzSizeK = self.vgprPool.checkOut(1)
        tmpSgpr = self.sgprPool.checkOut(1)
        module.add(SAddU32(sgpr(tmpSgpr), sgpr("SizesSum"), swzStride - 1, comment="Align to %u"%swzStride))
        module.add(SLShiftRightB32(dst=sgpr(tmpSgpr), src=sgpr(tmpSgpr), shiftHex=hex(log2(swzStride)), comment="numKr = DimK / %u"%swzStride))
        module.add(SMulI32(dst=sgpr(tmpSgpr, 1), src0=sgpr(tmpSgpr, 1), src1=hex(numElmInSwzBlk)))
        module.add(VMovB32(dst=vgpr(swzSizeK), src=sgpr(tmpSgpr, 1)))
        self.sgprPool.checkIn(tmpSgpr)

        module.addComment0("SWZ-%s: swizzle offset = F + [(tid / (swzMorN(%u) / VW(%u))) mod VW] * swzSizeK"%(tc, swzMorN, vw))
        module.add(VLShiftRightB32(dst=vgpr(tmpVgpr), shiftHex=hex(int(log((swzMorN/vw), 2))), src=vgpr(v), comment="tmp = tid / (swzMorN / VW)"))
        module.add(VAndB32(dst=vgpr(tmpVgpr), src0=vgpr(tmpVgpr), src1=vw-1, comment="tmp = tmp mod VW"))
        module.add(VMadU32U24(dst=vgpr(v), src0=vgpr(tmpVgpr), src1=vgpr(swzSizeK), src2=vgpr(fVgpr), comment="offset = tmp * swzSizeK + F"))
        self.vgprPool.checkIn(fVgpr)
        self.vgprPool.checkIn(swzSizeK)

      self.vgprPool.checkIn(row)
      self.vgprPool.checkIn(col)
      self.vgprPool.checkIn(tmpVgpr)

      for l in range(1, tP["nru"]):
        module.add(VAddCOU32(dst=vgpr(v+l), dst1=VCC(), src0=hex(numElmInSwzBlk), \
            src1=vgpr(v+l-1), comment="gro%s%s_%u"%(tP["tensorChar"], self.states.unrollChar, l)))

    else:
      numUnrollOffsets = tP["nru"]
      tP["gpr"]["unrollOffsets"] = self.vgprPool.checkOut(numUnrollOffsets, "unrollOffsets", self.states.preventVgprOverflowDuringNewTile)
      v = tP["gpr"]["unrollOffsets"]
      strideIdx = (tP["lsp"] if tP["tlu"] else tP["lsc"])
      stride = kernel[strideIdx]
      if (tc == "A" or tc == "B") and kernel["DirectToVgpr%s"%tc] and kernel["LocalSplitU"] > 1:
        # DTV + LSU case, we need to divide stride by LSU
        stride = stride // kernel["LocalSplitU"]
      prevStride = 0
      totalStride = 0
      dtvKInterval = self.states.dtvKIntervalA if tP["isA"] else self.states.dtvKIntervalB
      module.add(VMovB32(dst=vgpr(v), src=vgpr(tP["gpr"]["uReg"]), comment="gro%s%s_%u"%(tP["tensorChar"], self.states.unrollChar, 0)))
      for l in range(1, tP["nru"]):
        totalStride += stride
        if dtvKInterval > 1:
          # DirectToVgpr + k interval > 1 case, stride * dtvKInterval is added every dtvKInterval.
          # Add mod in mod != 0 case
          totalStride = stride * (l - (l % dtvKInterval)) + (l % dtvKInterval)
        currStride = totalStride - prevStride
        prevStride = totalStride
        module.add(VAddCOU32(dst=vgpr(v+l), dst1=VCC(), src0=currStride, \
            src1=vgpr(v+l-1), comment="gro%s%s_%u + %s"%(tP["tensorChar"], self.states.unrollChar, l, strideIdx)))
      #self.vgprPool.checkIn(tP["gpr"]["uReg"])
    return Module("graUnrollOffsets (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Shift A/B
  # See if the load (including vw) will extend past the 'free' dim of the
  # tensor.  If so clip to the last legal value which is inside the array
  ##############################################################################
  def graMetadataShift(self, kernel, tP):
    module = Module("graMetadataShift")

    if kernel["DirectToVgprSparseMetadata"]:
      margin = tP["glvw"] if tP["rtv"] else 1

      # Subtract the static component from SizesFree:
      with self.allocTmpSgpr(2+self.states.laneSGPRCount) as tmpSgprInfo:
        edgeSgpr = tmpSgprInfo.idx
        shiftSgpr    = edgeSgpr+1
        laneMaskSgpr = edgeSgpr+2
        module.add(SMulI32(dst=sgpr(edgeSgpr), src0=sgpr(tP["wg"]), src1=kernel[tP["mt"]], comment="WorkGroup[01] * MT"))
        module.add(SSubI32(dst=sgpr(edgeSgpr), src0=self.sizeRef(tP["idx"]), src1=sgpr(edgeSgpr), comment="edge = Size%s - WG*MT"%(tP["tileChar"])))
        module.add(SAndB32(dst=sgpr(shiftSgpr), src0=sgpr(edgeSgpr), src1=hex(margin-1), comment="edge size & (glvw-1)"))
        module.add(SSubU32(dst=sgpr(shiftSgpr), src0=hex(margin), src1=sgpr(shiftSgpr), comment="shift size = glvw - (edge size & (glvw-1))"))
        module.add(SAndN2B32(dst=sgpr(edgeSgpr), src0=sgpr(edgeSgpr), src1=hex(margin-1), comment="edgeCoord = edge & !(glvw-1)"))

        # apply shiftPointer into vgpr offset
        shiftedCoord = self.vgprPool.checkOut(1, "shiftedCoord", self.states.preventVgprOverflowDuringNewTile)
        for graIdx in range (0, kernel["MIWaveTile"][0]):
          vgprGro = "GlobalReadOffsetMetadata+%u"%(graIdx)
          # check if in shift area
          module.add(VCmpLeI32(dst=sgpr(laneMaskSgpr,self.states.laneSGPRCount), src0=sgpr(edgeSgpr), src1=vgpr(vgprGro), comment="edgeCoord <= coord"))
          # calculate shifted coord
          module.add(VSubI32(dst=vgpr(shiftedCoord), src0=vgpr(vgprGro), src1=sgpr(shiftSgpr), comment="shiftedCoord = coord - shift size"))
          # apply shift if condition is true
          module.add(VCndMaskB32(dst=vgpr(vgprGro), src0=vgpr(vgprGro), src1=vgpr(shiftedCoord), src2=sgpr(laneMaskSgpr,self.states.laneSGPRCount),
                      comment="coord =  (cond) ? shifted coord : ori coord"))
        self.vgprPool.checkIn(shiftedCoord)

    return module

  ##############################################################################
  def graShift(self, kernel, tP, margin=-1):
    # graShift requires a vgpr for each address component (so each component
    # can be examined and shifted if necessary) - therefore does not work
    # with UseSgprForGRO.
    assert(not kernel["_UseSgprForGRO"]), "%s"%self.states.kernelName

    module = Module("graShift")
    #tc = tP["tensorChar"]
    # edge value
    marginO = margin
    # for the edge case, using A's margin to instead Metadata's margin,
    # otherwise, loaded data of A and Metadata will not match
    if margin == -1:
      margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1, "edge", self.states.preventVgprOverflowDuringNewTile)

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      if self.states.groOffsetInMacroTile:
        # Subtract the static component from SizesFree:
        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tP["wg"]), src1=kernel[tP["mt"]], comment="WorkGroup[01] * MT"))
        module.add(SSubU32(dst=sgpr(tmpSgpr), src0=self.sizeRef(tP["idx"]), src1=sgpr(tmpSgpr), \
                  comment="edge = Size%s - WG*MT"%(tP["tileChar"])))
        # use math here to use unsigned (to increase range)
        #  - add srdShiftLeft to tmpSgpr - ensure it is always positive
        #  - below add srdShiftLeft to a tmp copy of the offset used for the compare
        # edge = (Size - WG*MT) - margin = the last valid load position that won't cause OOB
        # offset = the current load position for this thread
        # so if offset is larger than edge, we go back to the edge position
        module.add(SSubU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=margin, comment="edge -= margin(%u)"%(margin)))
        module.add(VMovB32(dst=vgpr(edge), src=sgpr(tmpSgpr), comment="edge vgpr = Size%s- WG*MT - margin(%u)"%(tP["tileChar"], margin) ))
        #shiftedEdge = self.vgprPool.checkOut(1, "shiftedEdge", self.states.preventVgprOverflowDuringNewTile)
        #module.add(VAddCOU32(dst=vgpr(shiftedEdge), dst1=VCC(), src0=vgpr(edge), src1=self.states.srdShiftLeft[tc],
        #             comment="shiftedEdge = edge + srdShiftLeft({})".format(self.states.srdShiftLeft[tc])))
      else:
        module.add(SSubU32(dst=sgpr(tmpSgpr), src0=self.sizeRef(tP["idx"]), src1=margin, \
            comment="edge = Size%s-%u"%(tP["tileChar"], margin) ))
        module.add(VMovB32(dst=vgpr(edge), src=sgpr(tmpSgpr), \
            comment="edge vgpr = Size%s-%u"%(tP["tileChar"], margin) ))

    # shift offsets
    vSrc = tP["vgprTileOffsets"]
    vDst = tP["vgprTileOffsets"]
    with self.allocTmpSgpr(self.states.laneSGPRCount) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      for l in range(0, tP["nrt"]):
        # compare
        cmpCommentText = "offset < edge"
        if self.states.groOffsetInMacroTile:
          module.add(VMinI32(dst=vgpr(vDst+l), src0=vgpr(edge), src1=vgpr(vSrc+l),
                      comment="offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge)))
        else:
          module.add(VCmpLtU32(dst=sgpr(tmpSgpr,self.states.laneSGPRCount), src0=vgpr(vSrc+l), src1=vgpr(edge),
                      comment="shiftedOffset < shiftedEdge"))
          # shift
          module.add(VCndMaskB32(dst=vgpr(vDst+l), src0=vgpr(edge), src1=vgpr(vSrc+l), src2=sgpr(tmpSgpr,self.states.laneSGPRCount),
                      comment="offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge)))
    # For metadata and using A's margin, shift extra tail offset
    if tP["isM"] and not marginO == -1:
      module.add(VAndB32(dst=vgpr(edge), src0=(margin-1), src1=vgpr(tP["gpr"]["tReg"]), comment="shifTailOffstet = tailOffset %% %d"%(margin)))
      for l in range(0, tP["nrt"]):
        module.add(VAddU32(dst=vgpr(vDst+l), src0=vgpr(edge), src1=vgpr(vSrc+l),
                        comment="offset += shifTailOffstet"))
    self.vgprPool.checkIn(edge)
    return module

  ##############################################################################
  # Global Read Addresses: Final Offsets metadata
  ##############################################################################
  def graMetadataFinalOffsets(self, kernel, tP):
    module = Module("graMetadataFinalOffsets")
    module.addComment1("calculate metadata gra final offset")


    # alloc vgpr
    kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
    tmpVgpr = self.vgprPool.checkOutAligned(2,2,"tmpVgpr")
    dummy   = self.vgprPool.checkOut(1,"dummy")

    tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

    # get constant parameter
    tc               = tP["tensorChar"]
    waveWidth        = kernel["WavefrontSize"]
    inputPerThread   = kernel["MIInputPerThread"]

    # parameter for get each type index
    dividendForKId   = kernel["MatrixInstM"]
    unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]

    # strider for each type of index
    strideTile       = self.sizeRef(unrollSummation[-1])
    strideK          = inputPerThread

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      # unroll offset
      module.add(vectorStaticRemainder(dummy, kReg, "Serial", waveWidth, tmpVgprRes, tmpSgprInfo, \
          "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
      module.add(vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgprRes, \
          "1. unroll offset: kIdx = wtid / (MIN(%u) )" % (kernel["MatrixInstN"])))
      module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), strideK, tmpSgprInfo, \
          "1. unroll offset: grKOffset = kIdx * mStride(%u)" % strideK))

    # Calculate final element offset
    for graIdx in range (0, kernel["MIWaveTile"][0]):
      vgprGro = "GlobalReadOffsetMetadata+%u"%(graIdx)
      module.add(VMulLOU32(vgpr(vgprGro), vgpr(vgprGro), strideTile, \
          "2. tile offset: tile coord * sizeL"))
      module.add(VAddU32(vgpr(vgprGro), vgpr(vgprGro), vgpr(kReg), \
          "2. final global read offset: fgrOffset = tile Offset + unroll Offset"))
      # elements to bytes
      module.add(vectorStaticDivide(vgprGro, vgprGro, 8, tmpVgprRes, \
        "  3. bytes offset : bnIdx = global read elememt offset / 8"))

    module.add(Label("graFinalMeta", ""))

    # release register
    self.vgprPool.checkIn(kReg)
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(dummy)

    return module

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    module = Module("graFinalOffsets")
    tc = tP["tensorChar"]
    tmp = self.vgprPool.checkOut(3, "tmp", self.states.preventVgprOverflowDuringNewTile)
    graIdx = 0
    swapPerpPara = (((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]) and (not tP["tlu"]) and tP["nrp"] > 1)

    # both UseSgprForGRO and DTVA/B are enabled
    if ((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]) and kernel["_UseSgprForGRO"]:
      if tP["tlu"]:
        sizeOfaPart = kernel["LocalReadVectorWidth"]
        numOfVW = tP["nrp"]//sizeOfaPart
        for sPerp in range(0, tP["nrpv"]*numOfVW):
          for perp in range(0, sizeOfaPart):
            for para in range(0, tP["nrc"]):
              for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
                # single loop
                singleModule, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
                module.add(singleModule)
      else:
        numOfVW = kernel["MIWaveTile%c"%tc]//kernel["VectorWidth%s"%tc] if kernel["VectorWidth%c"%tc]>1 else 1
        sizeOfaPart = tP["nrp"]//numOfVW
        # swap para and perp
        for para in range(0, tP["nrc"]):
          for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
            for sPerp in range(0, tP["nrpv"]*numOfVW):
              for perp in range(0, sizeOfaPart):
                # single loop
                singleModule, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
                module.add(singleModule)
    # DTVA/B always go this way, including swizzled
    elif (not swapPerpPara):
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
              # single loop
              singleModule, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              module.add(singleModule)
    else:
      # swap para and perp
      for para in range(0, tP["nrc"]):
        for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
          for perp in range(0, tP["nrp"]):
            for sPerp in range(0, tP["nrpv"]):
              # single loop
              singleModule, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              module.add(singleModule)

    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    tP["gpr"]["lwoT"] = None
    # if kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1:
    self.vgprPool.checkIn(tP["gpr"]["uReg2"])
    tP["gpr"]["uReg2"] = None

    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    tP["gpr"]["uReg"] = None
    if "subIterReg" in tP["gpr"]:
      if tP["gpr"]["subIterReg"] is not None:
        self.vgprPool.checkIn(tP["gpr"]["subIterReg"])
      tP["gpr"]["subIterReg"] = None

    if tP["vgprTileOffsetsCheckOut"]:
      self.vgprPool.checkIn(tP["vgprTileOffsets"])
      tP["vgprTileOffsets"] = None
      tP["vgprTileOffsetsCheckOut"] = False
      # _UseSgprForGRO uses same vgpr for ureg and tP["gpr"]["unrollOffsets"] so
      # let checkin(ureg) do the checkin
      # vgprTileOffsets is renamed version of treg/lwo so checkin here

    if not kernel["_UseSgprForGRO"]:
      self.vgprPool.checkIn(tP["gpr"]["unrollOffsets"])
      tP["gpr"]["unrollOffsets"] = None

    if tP["vgprPackedOffsets"] != None:
      self.vgprPool.checkIn(tP["vgprPackedOffsets"])
      tP["vgprPackedOffsets"] = None

    self.vgprPool.checkIn(tmp)

    return Module("Global Read Addresses: Final Offsets A/B (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B (single loop)
  ##############################################################################
  def graFinalOffsetsSingleLoop(self, kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara):
    module = Module("graFinalOffsetsSingleLoop")
    problemType = kernel["ProblemType"]
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    # single loop start

    # vgpr assignments
    if tP["tlu"]:
      vgprTile   = tP["vgprTileOffsets"]   + para*tVW + sPara*tVS
      vgprUnroll = tP["gpr"]["unrollOffsets"] + perp*uVW + sPerp*uVS
    else:
      vgprTile   = tP["vgprTileOffsets"]   + perp*tVW + sPara*tVS
      vgprUnroll = tP["gpr"]["unrollOffsets"] + para*uVW + sPerp*uVS

    if graIdx==0 or not kernel["_UseSgprForGRO"]:
      # emit global offset macro
      # TODO -refactor this and macro def to pass all indices, use the ones we need
      if kernel["BufferLoad"]:
        if tP["isSwizzled"]:
          bfName = "GLOBAL_OFFSET_%s_SWIZZLED" % tP["tensorChar"]
        else:
          bfName = "GLOBAL_OFFSET_%s" % tP["tensorChar"]
        bfArgs = ["vgprGlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)]
      else:
        bfName = "GLOBAL_OFFSET_%s" % tP["tensorChar"]
        bfArgs = ["vgprGlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx)]
      packedIter = 0 #iterator through ia
      iaToGpr = [None] * problemType["TotalIndices"]
      for i in tP["ia"]:
        if i < problemType["NumIndicesC"]:
          if i == tP["tileIdx"]:
            iaToGpr[i] = vgprTile
            bfArgs.append( "%2u" % iaToGpr[i] )
          else:
            if isPackedIndex(kernel,i):
              iaToGpr[i] = tP["vgprPackedOffsets"] + \
                            (vgprTile-tP["vgprTileOffsets"])*(len(tP["PackedIndices"])-1) + \
                            packedIter
              bfArgs.append( "%2u" % (iaToGpr[i]) )
              packedIter += 1
            else:
              # just a group index
              if not kernel["BufferLoad"]:  # buffer load adds these to SRD not the GLOBAL_OFFSET here
                bfArgs.append( "sgprWorkGroup%u"%i )
        else: # summation index
          if i == problemType["IndexUnroll"]:
            iaToGpr[i] = vgprUnroll
            bfArgs.append( "%2u" % iaToGpr[i] )
          # other summation indices are ignored

      bfArgs.append( "%u" % tmp )
      bfComment = "gRO%s_%u_%u_%u_%u" % (tP["tensorChar"], para, sPara, perp, sPerp)
      module.add(MacroInstruction(name=bfName, args=bfArgs, comment=bfComment))

      with self.allocTmpSgpr(2) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx

        # modify start
        if (not kernel["_UseSgprForGRO"]) and kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
          # add room for instruction offset
          groVgpr = "GlobalReadOffset%s+%u" % (tP["tensorChar"], graIdx)
          module.add(SMovB32(dst=sgpr(tmpSgpr), src=self.buff_load_inst_offset_max))
          module.add(VAddU32(dst=vgpr(groVgpr), src0=vgpr(groVgpr), src1=sgpr(tmpSgpr), comment="shift for UseInstOffsetForGRO"))

          ldsInc = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalReadVectorWidth%c"%tc] * tP["bpeGR"]
          if kernel["LdsBlockSizePerPad%s"%tc] != 0:
            ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeGR"]
          else:
            padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
            ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpeGR"]

          # buffer_load only support 12 bit instruction offset
          # we have to increase m0 if offset is larger thant 12 bits
          # so only keep 12 bit offset and subtract it on global address
          # global address will add back by buffer_load instruction offset
          ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
          if (ldsInc != 0):
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=ldsInc))
            module.add(VSubU32(dst=vgpr(groVgpr), src0=vgpr(groVgpr), src1=sgpr(tmpSgpr), comment="sub offset for buffer_load instoffset"))

    def computeScalarGroImpl(scalarGro):
      # this needs unroll stride in some cases and free stride in others
      # if we have multiple free strides - what is expected behavior?
      # could just extract the first free dimension from A?
      stride1 = "Stride%s%s"%(tc,self.states.indexChars[tP["idx"]])
      if tP["tlu"]:
        tileStride   = kernel[tP["lsc"]] * (para*tVW + sPara*tVS)
        unrollStride = 0

        if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]:
          unrollStride = perp*uVW + (kernel[tP["lsp"]] * sPerp * kernel["LocalReadVectorWidth"])
        else:
          unrollStride = kernel[tP["lsp"]] * (perp*uVW + sPerp*uVS)

        unrollSummation = [ i for i in tP["ia"] if i in problemType["IndicesSummation"] ]
        strideU = "Stride%s%s"%(tc,self.states.indexChars[unrollSummation[-1]])
        module.add(SMulI32(dst=sgpr(scalarGro), src0=sgpr(strideU), src1=unrollStride, \
                    comment="compute offset diff (scaled unrollDim)"))
        if tileStride:
          module.add(SAddU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=tileStride, \
                    comment="compute offset diff (tileDim)"))
      else:
        tileStride = 0
        if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc] and kernel["VectorWidth%s"%tc] > 1:
          tileStride = perp*tVW + (kernel[tP["lsp"]] * sPerp * kernel["VectorWidth%s"%tc])
        else:
          tileStride = kernel[tP["lsp"]] * (perp*tVW + sPara*tVS)

        unrollStride = kernel[tP["lsc"]] * (para*uVW + sPerp*uVS)
        strideF = "Stride%s%s"%(tc,self.states.indexChars[tP['tileIdx']])
        module.add(SMulI32(dst=sgpr(scalarGro), src0=sgpr(strideF), src1=tileStride, \
                    comment="compute offset diff (scaled tileDim)"))
        if unrollStride:
          module.add(SAddU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=unrollStride, \
                    comment="compute offset diff (unrollDim)"))

      # Using offsets so GRO holds a byte offset not an element offset
      # So scale here before comparison:
      if log2(tP["bpeGR"]) > 0:
        module.add(SLShiftLeftB32(
            dst=sgpr(scalarGro), \
            src=sgpr(scalarGro), \
            shiftHex=hex(log2(tP["bpeGR"])), \
            comment="scalar offset *= bytes/element"))
      else:
        module.addCommentAlign("scalar offset *= bytes/element (multiplier is 1, do nothing)")

      if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
        # add room for instruction offset
        module.add(SAddU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=self.buff_load_inst_offset_max, comment="shift for UseInstOffsetForGRO"))

        ldsInc = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalReadVectorWidth%c"%tc] * tP["bpeGR"]
        if kernel["LdsBlockSizePerPad%s"%tc] != 0:
          ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeGR"]
        else:
          padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
          ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpeGR"]

        # buffer_load only support 12 bit instruction offset
        # we have to increase m0 if offset is larger thant 12 bits
        # so only keep 12 bit offset and subtract it on global address
        # global address will add back by buffer_load instruction offset
        ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
        if (ldsInc != 0):
          module.add(SSubU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=ldsInc, comment="sub offset for buffer_load instoffset"))

      if self.states.checkGRO:
        # Debug mode to verify that the computed offsets are offset by the expected scalar
        print(tc, "tileStride=", tileStride, "unrollStride=", unrollStride, \
              "stride=%s"%(stride1))

        module.add(self.getVectorDiffAssert(vgpr("GlobalReadOffset%s+%u"%(tc,0)), \
                                            vgpr("GlobalReadOffset%s+%u"%(tc,graIdx)), \
                                            sgpr(scalarGro)))

    needFirstSgprOffset = kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]

    if (kernel["_UseSgprForGRO"] or self.states.checkGRO) and (needFirstSgprOffset or graIdx > 0):
      # compute offsets for scalar global read offsets:
      if kernel["_UseSgprForGRO"]:
        tmpIdx = graIdx if needFirstSgprOffset else graIdx-1
        scalarGro = "ScalarGlobalReadOffset%s+%u"%(tc, tmpIdx)
        computeScalarGroImpl(scalarGro)
      else:
        # TODO: need for investagation for replacing by allocTmpSgpr
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          scalarGro = tmpSgprInfo.idx
          computeScalarGroImpl(scalarGro)

    graIdx += self.states.rpgo if kernel["BufferLoad"] else self.states.rpga

    return module, graIdx

  def computeMetaDataSrd(self, kernel, tP, tc, indices):
    module = Module("computeMetaDataSrd")

    wroteTileStart = True
    with self.allocTmpSgpr(2 + 2 + 2) as tmpSgprInfo:
      stmp = tmpSgprInfo.idx
      gsuoffset = stmp
      tensorSize = stmp
      tileStart = stmp+2
      blockOffset = stmp+4
      actualBatchSize = stmp+5 #for broadcast
      actualBatchIndex = stmp+5 #for broadcast

      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(tP["wg"]), kernel[tP["mt"]], comment="WorkGroup[01] * MT"))

      unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]
      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart), sgpr(tileStart+1), sgpr(tileStart+0), self.sizeRef(unrollSummation[-1]), \
                                comment="scaled tile-offset by Summation size"))

      depthU = kernel["DepthU"]
      gsucLabel    = Label(label=self.labels.getNameInc("GSUC_M"), comment="")
      gsucLabelEnd = Label(label=self.labels.getNameInc("GSUC_M_End"), comment="")
      module.add(SAndB32(dst=sgpr(tmpSgprInfo.idx), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))
      gsuOffsetStr = "gsuOffset = DepthU*GSUSumIdx"
      module.add(SCBranchSCC1(labelName=gsucLabel.getLabelName(), comment="branch if GSUC == 1"))
      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), depthU, sgpr("GSUSumIdx"), comment=gsuOffsetStr))
      module.add(SBranch(gsucLabelEnd.getLabelName()))
      module.add(gsucLabel)
      gsuOffsetStr = "gsuOffset = DepthU*accumulatedNumOfLoopCounterL"
      loopCounterName = self.loopCounterName(kernel, self.states.unrollIdx)
      module.add(SLShiftRightB32(dst=sgpr(loopCounterName), src=sgpr("SizesSum"), shiftHex=log2(depthU), \
                                  comment="s[%s] = s[sgprSizesSum] / %s"%(loopCounterName, depthU)))
      module.add(self.calculateLoopNumIterOffsetGsu(kernel, loopCounterName, tmpSgprInfo))
      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(stmp+0), depthU, comment=gsuOffsetStr))
      module.add(gsucLabelEnd)
      module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(stmp+0), comment="accum GsuOffet term to tilestart"))
      module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(stmp+1), comment="accum GsuOffet term to tilestart"))

      sizeIndex = [ dim for dim in tP["ia"] ]
      assert(len(sizeIndex) >= 2)
      module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(blockOffset), sgpr(blockOffset+1), self.sizeRef(sizeIndex[0]), self.sizeRef(sizeIndex[1]), \
                                comment="calculate metadata tensor size"))
      for dim in sizeIndex[2:]:
        module.add(SCmpEQU32(src0=sgpr("Stride%s%s"%(tc,self.states.indexChars[tP['ia'][dim]])), src1=0, comment="broadcast %s?"%tc))
        module.add(SCSelectB32(dst=sgpr(actualBatchSize), src0=hex(1) , src1=self.sizeRef(sizeIndex[dim]), comment="set batchSize as 1 for boardcast %s"%tc))
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tensorSize), sgpr(tensorSize+1), sgpr(blockOffset), sgpr(actualBatchSize), \
                                comment="calculate metadata tensor size"))
        module.add(SCSelectB32(dst=sgpr(actualBatchIndex), src0=0 , src1=sgpr("WorkGroup2"), comment="set batchIndex as 0 for boardcast %s"%tc))

      if self.states.use64bShadowLimit:
        limitTmp0 = "ShadowLimitMetadata"
        limitTmp1 = "ShadowLimitMetadata+1"
      else:
        limitTmp0 = stmp+0
        limitTmp1 = stmp+1

      module.add(SSubU32(sgpr(limitTmp0), sgpr(tensorSize), sgpr(tileStart+0), "sub tileStart"))
      module.add(SSubBU32(sgpr(limitTmp1), sgpr(tensorSize+1), sgpr(tileStart+1), "sub tileStart"))

      if self.states.use64bShadowLimit:
        module.add(SLShiftRightB64(sgpr(limitTmp0,2), hex(log2(8)), sgpr(limitTmp0,2), "Set limit to use bytes"))
        module.add(SCmpEQU32(sgpr(limitTmp1), 0, "are we within 2^32?"))
        module.add(SCSelectB32(sgpr("SrdMetadata+2"), sgpr(limitTmp0), "BufferLimit", "Move shadow to real if we are within 2^32"))
      else:
        module.add(SLShiftRightB32(sgpr("SrdMetadata+2"), hex(log2(8)), sgpr(limitTmp0), "Set limit to use bytes"))

      numDim = len(indices)
      wg=2
      for i in range(0, numDim):
        idx = indices[i]
        if not ( idx == kernel["ProblemType"]["Index0"] \
            or idx == kernel["ProblemType"]["Index1"] \
            or idx in kernel["ProblemType"]["IndicesSummation"] \
            or isPackedIndex(kernel, idx)):
          assert(wg==2)
          if not wroteTileStart:
            module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(blockOffset), sgpr("WorkGroup2"), comment="block offset*WG"))
            wroteTileStart = True
          else:
            module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(blockOffset), sgpr(actualBatchIndex), comment="block offset*WG"))
            module.add(SAddU32(sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum wg term to tilestart"))
            module.add(SAddCU32(sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum wg term to tilestart"))

      if wroteTileStart:
        module.add(SLShiftRightB64(sgpr(tileStart,2), hex(log2(8)), sgpr(tileStart,2), "Set limit to use bytes"))
        module.add(SAddU32(sgpr("SrdMetadata+0"), sgpr("AddressMetadata+0"), sgpr(tileStart+0), "SRD base = Address+ tileStart0"))
        module.add(SAddCU32(sgpr("SrdMetadata+1"), sgpr("AddressMetadata+1"), sgpr(tileStart+1), "SRD base = Address+ tileStart1"))
      else:
        module.add(SMovB64(sgpr("SrdMetadata+0", 2), sgpr("AddressMetadata+0", 2), "init SRD base address" ))

      module.add(SMovB32(sgpr("SrdMetadata+3"), "Srd127_96", "Set bits 127_96 in SRD"))
    return module

  ##############################################################################
  # Add the constant offsets to the specified srd.
  # Srd is set to point to the base of the tile. All offsets except lowest-order
  # 2d dims are computed into the SRD.
  # GRO are offset from the tile SRD and the first GRO will be 0
  # Only called for BufferLoad=1 (or eventually BufferStore=1)
  ##############################################################################
  def computeLoadSrd(self, kernel, tP, tc, indices, bpe):
    module = Module("computeLoadSrd")
    with self.allocTmpSgpr(2 + 2 + (0 if self.states.use64bShadowLimit else 2)) as tmpSgprInfo:
      stmp = tmpSgprInfo.idx
      tileStart = stmp+2
      if self.states.use64bShadowLimit:
        tensor2dSize0 = "ShadowLimit%s+0"%tc
        tensor2dSize1 = "ShadowLimit%s+1"%tc
      else:
        tensor2dSize0 = stmp+4
        tensor2dSize1 = stmp+5
      wroteTileStart = False
      #---
      # Compute tileStart #elements from the 2D array start
      # Add tile (and unroll if GSU) component into SRD - SRD will point to beginning of the macro-tile:
      if self.states.groOffsetInMacroTile:
        # packed modes can't use this mode, and code here assumes 1 index.
        assert(len(kernel["PackedC0IndicesX"])==1)
        assert(len(kernel["PackedC1IndicesX"])==1)

        wroteTileStart = True
        #tP['ia'][1]

        # This is guaranteed to fit in 32-bit since the WG*MT is a number of elements in some unsigned direction:
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(tP["wg"]), kernel[tP["mt"]], comment="WorkGroup[01] * MT"))
        strideF = self.strideRef(tc, tP['tileIdx'])
        if not self.isConstUnitStride(strideF):
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart), sgpr(tileStart+1), sgpr(tileStart+0), \
                    strideF, comment="tlu=0, scaled tile-offset by stride"))

        skComponent = Component.StreamK.find(self)
        module.add(skComponent.computeLoadSrd(self, kernel, tc, stmp))

        gsuComponent = Component.GSU.find(self)
        module.add(gsuComponent.computeLoadSrd(self, kernel, tP, stmp, tileStart))

      # Output : tileStart[0:1] have offset in elements from the 2D start of the tile.
      # if groOffsetInMacroTile=1, 2DStart + tileStart gives the the start of the macro-tile;
      # This is used to compute the limit.
      # Later we modify tileStart to include batch and higher-order dims and add this to SRD.

      #---
      # Compute BUFFER Limit:
      prePad = self.states.srdShiftLeft[tc] * tP["bpeGR"] # leave room in case we have to pointer shift

      if not wroteTileStart:
        module.add(SMovB64(dst=sgpr(tileStart, 2), src=0, comment="set default tileStart"))

      #Calculate tensor 2d size
      if self.states.use64bShadowLimit or ((not self.states.use64bShadowLimit) and tensor2dSize0 % 2 == 0):
        module.add(SMovB64(dst=sgpr(tensor2dSize0, 2), src=0x1, comment="Init tensor size"))
      else:
        module.add(SMovB32(dst=sgpr(tensor2dSize0), src=0x1, comment="Init tensor size"))
        module.add(SMovB32(dst=sgpr(tensor2dSize1), src=0x0, comment="init tensor size"))


      numDim = len(indices)
      for i in range(0, numDim):
        idx = indices[i]
        if idx == kernel["ProblemType"]["Index0"] \
            or idx == kernel["ProblemType"]["Index1"] \
            or idx in kernel["ProblemType"]["IndicesSummation"] \
            or isPackedIndex(kernel, idx):
          stride = self.strideRef(tc,idx)
          size =   self.sizeRef(idx)
          # The sizeL of a structure sparsity 2:4 matrix is half of the dense matrix.
          if (idx in kernel["ProblemType"]["IndicesSummation"]) and     \
             ((tP["isA"] and kernel["ProblemType"]["Sparse"] == 1) or   \
             (tP["isB"] and kernel["ProblemType"]["Sparse"] == 2)) :
            module.add(SLShiftRightB32(dst=sgpr(stmp), src=size, shiftHex=0x1, comment="(size/2)"))
            module.add(SSubU32(dst=sgpr(stmp), src0=sgpr(stmp), src1=0x1, comment="(size/2-1)"))
          else:
            if tP["isSwizzled"]:
              if idx in kernel["ProblemType"]["IndicesSummation"]:
                module.addModuleAsFlatItems(self.alignTo(stmp, "SizeL", tP["swizzleK"]))
                module.add(SSubU32(dst=sgpr(stmp), src0=sgpr(stmp), src1=1, comment="SWZ-%s align: (sizeL-1)"%tc))
              elif tP["isA"] and idx == kernel["ProblemType"]["Index0"]:
                module.addModuleAsFlatItems(self.alignTo(stmp, "SizeI", 16))
                module.add(SSubU32(dst=sgpr(stmp), src0=sgpr(stmp), src1=1, comment="SWZ-%s align: (sizeM-1)"%tc))
              elif tP["isB"] and idx == kernel["ProblemType"]["Index1"]:
                module.addModuleAsFlatItems(self.alignTo(stmp, "SizeJ", 16))
                module.add(SSubU32(dst=sgpr(stmp), src0=sgpr(stmp), src1=1, comment="SWZ-%s align: (sizeN-1)"%tc))
              else:
                module.add(SSubU32(dst=sgpr(stmp), src0=size, src1=0x1, comment="(size-1)"))
            else:
              module.add(SSubU32(dst=sgpr(stmp), src0=size, src1=0x1, comment="(size-1)"))
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), stride, \
                      sgpr(stmp), comment="stride x (size-1)"))
          module.add(SAddU32(dst=sgpr(tensor2dSize0), src0=sgpr(tensor2dSize0), src1=sgpr(stmp+0), comment="sum tensor size"))
          module.add(SAddCU32(dst=sgpr(tensor2dSize1), src0=sgpr(tensor2dSize1), src1=sgpr(stmp+1), comment="sum tensor size"))

      if self.states.use64bShadowLimit:
        limitTmp0 = "ShadowLimit%s+0"%tc
        limitTmp1 = "ShadowLimit%s+1"%tc
      else:
        limitTmp0 = stmp+0
        limitTmp1 = stmp+1

      module.add(SSubU32(dst=sgpr(limitTmp0), src0=sgpr(tensor2dSize0), src1=sgpr(tileStart+0), comment="sub tileStart"))
      module.add(SSubBU32(dst=sgpr(limitTmp1), src0=sgpr(tensor2dSize1), src1=sgpr(tileStart+1), comment="sub tileStart"))

      if self.states.use64bShadowLimit:
        # Set initial buffer limit
        # if the limit is >64bit, incrementSrd decrements the shadow as the SRD increments,
        # and when we get within 32-bit we start to step down the SRD
        # if the limit is <32bits, set it accurately here:
        # Note lshl_b64 the higher-numbered SGPR has the upper 32-bits
        if log2(tP["bpeGR"]) > 0:
          module.add(SLShiftLeftB64(dst=sgpr("ShadowLimit%s"%tc,2),  src=sgpr("ShadowLimit%s"%tc,2), \
              shiftHex=hex(log2(tP["bpeGR"])), comment="Set limit to use bytes"))
        else:
          module.addCommentAlign("Set limit to use bytes (byte is 1, do nothing)")
        if prePad:
          module.add(SAddU32(dst=sgpr("ShadowLimit%s+0"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1=prePad, comment="extend limit for pre-pad"))
          module.add(SAddCU32(dst=sgpr("ShadowLimit%s+1"%tc), src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="extend limit for pre-pad"))

        if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
          module.add(SAddU32(dst=sgpr("ShadowLimit%s+0"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1=self.buff_load_inst_offset_max, comment="extend limit for directToLDS instruction offset"))
          module.add(SAddCU32(dst=sgpr("ShadowLimit%s+1"%tc), src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="extend limit for directToLDS instruction offset"))

        module.add(SCmpEQU32(src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="are we within 2^32?"))
        module.add(SCSelectB32(dst=sgpr("Srd%s+2"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1="BufferLimit", comment="Move shadow to real if we are within 2^32"))
      else:
        # put limit directly into SRD:
        if log2(tP["bpeGR"]) > 0:
          module.add(SLShiftLeftB32(dst=sgpr("Srd%s+2"%tc), src=sgpr(stmp+0), shiftHex=hex(log2(tP["bpeGR"])), comment="Set limit to use bytes"))
        else:
          module.addCommentAlign("Set limit to use bytes (byte is 1, do nothing)")
        module.add(SAddU32(dst=sgpr("Srd%s+2"%tc), src0=sgpr("Srd%s+2"%tc), src1=prePad, comment="extend limit for pre-pad"))

      # Apply any high-order address components to the tileStart and eventually the SRD - batch idx for batched gemm
      if kernel["ProblemType"]["StridedBatched"]:
        wg=2 # TODO - refactor since only WG2 is supported and this is always batch
        for i in range(1, numDim):
          idx = indices[i]
          if idx == kernel["ProblemType"]["Index0"] \
              or idx == kernel["ProblemType"]["Index1"] \
              or idx in kernel["ProblemType"]["IndicesSummation"] \
              or isPackedIndex(kernel, idx):
                continue # these will be captured in GRO not the SRD (or other summations are always 0)
          else:
            assert(wg==2) # can only have one wg2 with a batch. Other dimensions should be packed into wg0/wg1
            stride = "Stride%s%s"%(tc,self.states.indexChars[tP['ia'][i]])
            if not wroteTileStart:
              module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(stride), sgpr("WorkGroup2"), comment="Stride*WG"))
              wroteTileStart = True
            else:
              module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(stride), sgpr("WorkGroup2"), comment="Stride*WG"))
              module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(stmp+0), comment="accum wg term to tilestart"))
              module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(stmp+1), comment="accum wg term to tilestart"))
            wg+=1

    # Add the tile start to the SRD
    if wroteTileStart:
      module.add(scalarStaticMultiply64(sgpr(tileStart,2), sgpr(tileStart,2), tP["bpeGR"], None, "tileStart *= BPE"))
      module.add(SAddU32(dst=sgpr("Srd%s+0"%tc), src0=sgpr("Address%s+0"%tc), src1=sgpr(tileStart+0), comment="SRD base = Address+ tileStart0"))
      module.add(SAddCU32(dst=sgpr("Srd%s+1"%tc), src0=sgpr("Address%s+1"%tc), src1=sgpr(tileStart+1), comment="SRD base = Address+ tileStart1"))
    else:
      module.add(SMovB64(dst=sgpr("Srd%s"%tc, 2), src=sgpr("Address%s"%tc, 2), comment="init SRD base address"))

    # self.states.groOffsetInMacroTile == 1 case,  pre-pad is already subtracted from AddressA/B
    if prePad and self.states.groOffsetInMacroTile == 0:
      module.add(SSubU32(dst=sgpr("Srd%s+0"%tc), src0=sgpr("Srd%s+0"%tc), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
      module.add(SSubBU32(dst=sgpr("Srd%s+1"%tc), src0=sgpr("Srd%s+1"%tc), src1=0, comment="pre-pad to make room for possible pointer shift"))

    if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
      module.add(SSubU32(dst=sgpr("Srd%s+0"%tc), src0=sgpr("Srd%s+0"%tc), src1=self.buff_load_inst_offset_max, comment="make room for directToLDS instruction offset"))
      module.add(SSubBU32(dst=sgpr("Srd%s+1"%tc), src0=sgpr("Srd%s+1"%tc), src1=0, comment="make room for directToLDS instruction offset"))

    module.add(SMovB32(dst=sgpr("Srd%s+3"%tc), src="Srd127_96", comment="Set bits 127_96 in SRD"))

    #if tP["isB"]:
    #  module.add(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup1"), 0xA))

    return module

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  def graAddresses(self, kernel, tP):
    module = Module("graAddresses")
    tc = tP["tensorChar"]
    graIdx = 0

    if tP["isSwizzled"]:
      # "StrideA0I" or "StrideB1J"
      strideName = "Stride%s%s"%(tc,self.states.indexChars[tP["idx"]])
      module.addModuleAsFlatItems(self.setTo(strideName, "SizeL"))
      module.addModuleAsFlatItems(self.alignTo(strideName, strideName, tP["swizzleK"]))

    if kernel["BufferLoad"]:
      # maxAddrSgpr = size[n] * stride[n-1]
      module.addComment0("max read offset = size[n] * stride[n-1]")
      module.add(self.computeLoadSrd(kernel, tP, tc, kernel["ProblemType"]["IndexAssignments%s"%tc], tP["bpeGR"]))

      if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"]:
        if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]):
          module.add(self.computeMetaDataSrd(kernel, tP, tc, kernel["ProblemType"]["IndexAssignments%s"%tc]))

    else:
      tmp = self.vgprPool.checkOut(2, "tmp", self.states.preventVgprOverflowDuringNewTile)

      skComponent = Component.StreamK.find(self)
      module.add(skComponent.graAddresses(self, kernel, tc, tmp))

      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):

              comment = "gRA%s_%u_%u_%u_%u = addr%s+grO%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp, \
                  tP["tensorChar"], tP["tensorChar"], \
                  para, sPara, perp, sPerp )
              module.add(VAddCOU32(
                  dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  dst1=VCC(), \
                  src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  src1=vgpr(tmp+0), \
                  comment=comment+" (lower)"))
              module.add(VAddCCOU32(
                  dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  dst1=VCC(), \
                  src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  src1=vgpr(tmp+1), \
                  src2=VCC(), \
                  comment=comment+" (upper)"))
              graIdx += self.states.rpga
      self.vgprPool.checkIn(tmp)

    return module

  ##############################################################################
  # Global Read Addresses: Increments
  # Define graIncrements, called once for each summation
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    module = Module("graIncrements")
    tc = tP["tensorChar"]

    dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
    loopChar = self.states.indexChars[dimIdx]

    stride = self.strideRef(tc, dimIdx)
    isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

    #print (tc, ": loopIdx=", loopIdx, "dimIdx=", dimIdx, "strideIdx=", strideIdx)

    assert(self.states.unrollIdx == kernel["ProblemType"]["NumIndicesSummation"]-1)
    if loopIdx==self.states.unrollIdx:
      gsuComponent = Component.GSU.find(self)
      module.add(gsuComponent.graIncrements(self, kernel, loopIdx, tP))
    else:
      # other summation
      if self.states.globalReadIncsUseVgpr:
        printExit("NumIndicesSummation=%u not yet supported in assembly unless globalReadIncsUseVgpr==0" \
            % kernel["ProblemType"]["NumIndicesSummation"] )
      else:
        graInc = "GlobalReadIncs%s+%u"%(tc, loopIdx)
        # subtract increments done by the inner iterations
        # may be negative:
        loopIdxPrev = loopIdx + 1
        dimIdxPrev    = kernel["ProblemType"]["IndicesSummation"][loopIdxPrev] # dimension index
        loopCharPrev  = self.states.indexChars[dimIdxPrev]
        stridePrev = self.strideRef(tc, dimIdxPrev)
        isMirrorIdxPrev = dimIdxPrev in kernel["ProblemType"]["MirrorDims%s"%tc]

        module.addComment1("compute globalReadInc for higher-level loop")

        with self.allocTmpSgpr(3) as tmpSgprInfo:
          tmpSgpr = tmpSgprInfo.idx
          # Summations always appear in both A and B, can compute number of iterations just once:
          if loopIdxPrev==self.states.unrollIdx:
            loopCounterName= self.loopCounterName(kernel, self.states.unrollIdx)
            if tP["isA"]:
              quotient = loopCounterName
              dividend = "SizesSum+%u"%self.states.unrollIdx
              divisor = kernel["DepthU"]
              if kernel["NoTailLoop"] and kernel["AssertSummationElementMultiple"] % kernel["DepthU"] != 0:
                # round up SizesSum/DepthU for noTailLoop case
                module.add(SAddI32(dst=sgpr(quotient), src0=(divisor - 1), src1=sgpr(dividend), \
                    comment="round up SizeSum / DepthU" ))
                module.add(scalarStaticDivideAndRemainder(quotient, None, quotient, \
                            divisor, tmpSgprInfo, 0))
              else:
                module.add(scalarStaticDivideAndRemainder(quotient, None, dividend, \
                            divisor, tmpSgprInfo, 0))

              gsuComponent = Component.GSU.find(self)
              if kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1:
                module.add(gsuComponent.calculateLoopNumIterGsu(self, kernel, loopCounterName, tmpSgprInfo))
              module.add(gsuComponent.graIncrementsRestore(self, kernel, loopCounterName))

            module.add(SMulI32(dst=sgpr(graInc), src0=stridePrev, src1=sgpr(loopCounterName), \
                  comment="tmp <- stride%s%s * myWgUnrollIters" %(tc, loopCharPrev)))
          else:
            module.add(SMulI32(dst=sgpr(graInc), src0=stridePrev, src1=self.sizeRef(dimIdxPrev), \
                  comment="tmp <- stride%s%s * size%s%s" %(tc, loopCharPrev, tc, loopCharPrev)))

        # subtract amount that previous inner loop will have already incremented:
        # graInc is used as temp for the prev loop calc
        if isMirrorIdx and isMirrorIdxPrev:
          module.add(SSubI32(dst=sgpr(graInc), \
              src0=sgpr(graInc), \
              src1=stride, \
              comment="incr%s%s = <prev-incs> - stride%s%s"%(tc, loopChar, tc, loopChar) ))
        elif isMirrorIdx:
          module.add(SAddI32(dst=sgpr(graInc), \
              src0=stride, \
              src1=sgpr(graInc), \
              comment="incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) ))
          module.add(SSubI32(dst=sgpr(graInc), \
              src0=0, \
              src1=sgpr(graInc), \
              comment="incr%s%s = - (stride%s%s + <prev-incs>)"%(tc, loopChar, tc, loopChar) ))
        elif isMirrorIdxPrev:
          module.add(SAddI32(dst=sgpr(graInc), \
              src0=stride, \
              src1=sgpr(graInc), \
              comment="incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) ))
        else:
          module.add(SSubI32(dst=sgpr(graInc), \
              src0=stride, \
              src1=sgpr(graInc), \
              comment="incr%s%s = stride%s%s - <prev-incs>"%(tc, loopChar, tc, loopChar) ))

        module.add(SLShiftLeftB32(
            dst=sgpr(graInc), \
            src=sgpr(graInc), \
            shiftHex="BpeGR%sLog2"%tc,
            comment="<- scale by bpeDS"))

    return Module("graIncrements (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    module = Module("lwaTileAssignment")
    tc = tP["tensorChar"]

    divisorName = tP["lvc"]
    divisor = kernel[divisorName]

    # DTV case, use tlu path
    isDTVAB = (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]
    isTr = (tc == "A" or tc == "B") and kernel["enableGLTr%s"%tc]
    isSwizzledOrTr = tP["isSwizzled"] or isTr
    swizzledOrTrName = ""
    if tP["isSwizzled"]:
      swizzledOrTrName = "SwizzleTensor%s" % tc
    elif isTr:
      swizzledOrTrName = "GLTr%s"%tc

    # swizzled goes to else
    if (tP["tlu"] or isDTVAB) and (not isSwizzledOrTr):
      rReg = self.vgprPool.checkOut(1, "lwaTA rReg0", self.states.preventVgprOverflowDuringNewTile) # tile = serial%divisor
      qReg = self.vgprPool.checkOut(1, "lwaTA qReg0", self.states.preventVgprOverflowDuringNewTile) # unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprPool.checkOut(1, 'lwaTA qReg1', self.states.preventVgprOverflowDuringNewTile) # tile = serial/divisor
      rReg = self.vgprPool.checkOut(1, 'lwaTA rReg1', self.states.preventVgprOverflowDuringNewTile) # unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"

    module.addComment0("%s = %u" % (divisorName, kernel[divisorName]))
    module.addComment0("%s = %s-unroll = serial%s%s" \
        % (vgpr(uReg), tc, uOpStr, divisorName) )

    tmpVgpr = self.vgprPool.checkOutAligned(2, 2, 'lwaTA vgpr', self.states.preventVgprOverflowDuringNewTile)
    tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

    dividendReg = "Serial" # local serial

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      dividendReg = self.vgprPool.checkOut(1, "idInWave", self.states.preventVgprOverflowDuringNewTile)
      dummy       = self.vgprPool.checkOut(1, "dummy", self.states.preventVgprOverflowDuringNewTile)
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(vectorStaticRemainder(dummy, dividendReg, "Serial", kernel["WavefrontSize"], tmpVgprRes, tmpSgprInfo))

    # store DirectToVgpr K interval for later use
    dtvKInterval = 1

    # swizzle
    if isSwizzledOrTr:
      module.addComment0("TileAssignment for DirectToVgpr%s and %s" % (tc, swizzledOrTrName))
      module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, kernel["WavefrontSize"], tmpVgprRes))

      # Calc numKr
      if tP["isSwizzled"]:
        tmp = self.sgprPool.checkOut(1)
        numKr = sgpr(tmp)
        swzStride = tP["swizzleK"]
        module.addComment(f"Align to {swzStride}")
        module.add(SAddU32(numKr, sgpr("SizesSum"), swzStride-1))
        module.add(SLShiftRightB32(dst=numKr, shiftHex=hex(log2(swzStride)), src=numKr,  comment="%s: numKr = DimK / %s"%(swizzledOrTrName, swzStride)))
      elif isTr:
        numKr = kernel["MatrixInstK"] // tP["glvw"]

        tmp = self.vgprPool.checkOut(2)
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          module.add(vectorStaticDivide(tmp, rReg, tP["bpeGR"]*8, tmpVgprRes, comment="%s: %s = %s / %s"%(swizzledOrTrName, vgpr(tmp), vgpr(rReg), tP["bpeGR"]*8)))
          module.add(vectorStaticMultiply(vgpr(tmp), vgpr(tmp), tP["bpeGR"]*4, tmpSgprInfo, comment="%s: %s = %s * %s"%(swizzledOrTrName, vgpr(tmp), vgpr(tmp), tP["bpeGR"]*4)))
        module.add(vectorStaticRemainder(tmp+1, rReg, rReg, tP["bpeGR"]*4, tmpVgprRes, tmpSgprInfo, comment="%s: %s = %s %% %s"%(swizzledOrTrName, vgpr(tmp), vgpr(rReg), tP["bpeGR"]*4)))
        module.add(VAddU32(dst=vgpr(rReg), src0=vgpr(tmp), src1=vgpr(rReg), comment="%s: %s = %s + %s"%(swizzledOrTrName, vgpr(rReg), vgpr(rReg), vgpr(tmp))))

      WvG_M = kernel["MIWaveGroup"][0]
      if tP["isA"]:
        module.add(VAndB32(dst=vgpr(qReg), src0=hex(WvG_M-1), src1=vgpr(qReg), comment="%s: wave_id (along_M) %%= MIWG[0]"%(swizzledOrTrName)))
        module.add(VMulU32U24(dst=vgpr(qReg), src0=numKr, src1=vgpr(qReg), comment="%s: wave_id (along_M) *= numKr"%(swizzledOrTrName)))
      elif tP["isB"]:
        # NB:
        #   Calc of w_id is: /= MIWG[0], not %= MIWG[1]
        module.add(VLShiftRightB32(dst=vgpr(qReg), shiftHex=log2(WvG_M), src=vgpr(qReg), comment="%s: wave_id (along_N) /= MIWG[0]"%(swizzledOrTrName)))
        if kernel["LocalSplitU"] > 1:
          # LSU for swizzleB: make sure wave_id is in the range of MIWG[1], so need extra %=MIWG[1]
          WvG_N = kernel["MIWaveGroup"][1]
          module.add(VAndB32(dst=vgpr(qReg), src0=hex(WvG_N-1), src1=vgpr(qReg), comment="%s: LSU Case: wave_id (along_N) %%= MIWG[1]"%(swizzledOrTrName)))
        module.add(VMulU32U24(dst=vgpr(qReg), src0=numKr, src1=vgpr(qReg), comment="%s: wave_id (along_N) *= numKr"%(swizzledOrTrName)))

      if tP["isSwizzled"]:
        self.sgprPool.checkIn(tmp)
      elif isTr:
        module.add(VBfeU32(dst=vgpr(tmp), src0=vgpr(dividendReg), src1=tP["bpeGR"]+1, src2=1, comment="%s: offset for the right half of the tile"%(swizzledOrTrName)))
        module.add(VAddU32(dst=vgpr(qReg), src0=vgpr(tmp), src1=vgpr(qReg), comment="%s: wave_id += offset for the right half of the tile"%(swizzledOrTrName)))
        self.vgprPool.checkIn(tmp)
    elif isDTVAB:
      # offset calculation for DirectToVgpr
      # call function from LraTileAssignmentMFMA for DirectToVgpr
      module.addComment0("TileAssignment for DirectToVgpr%s" % tc)
      component = Component.LraTileAssignment.find(self)
      module.add(component.LraTileAssignmentCode(self, kernel, tP, tReg, uReg, tmpVgprRes, dividendReg=dividendReg, isDTVAB=True))

      # The other side of lrvw
      if tP["isA"]:
        # the other is B
        tluOther = kernel["ProblemType"]["TLUB"]
        if tluOther:
          lrvwOther = self.states.lrvwTileB
        else:
          lrvwOther = self.states.lrvwUnrollB
      else:
        # the other is A
        tluOther = kernel["ProblemType"]["TLUA"]
        if tluOther:
          lrvwOther = self.states.lrvwTileA
        else:
          lrvwOther = self.states.lrvwUnrollA
      if lrvwOther >= 2 and (not tluOther) and tP["tlu"]:
        # DirectToVgpr + LocalReadVectorWidth>=2 case, multiply qReg by lrvwOther
        dtvKInterval = lrvwOther
      if  tluOther and tP["tlu"]:
        # DirectToVgpr + both TLU case, multiply qReg by kernel["MIInputPerThread"]
        dtvKInterval = kernel["MIInputPerThread"]
      module.add(vectorStaticMultiply(vgpr(qReg), vgpr(qReg), dtvKInterval, None))

      # DTV+localSplitU case. Calculate LSU offset here
      if kernel["LocalSplitU"] > 1:
        # allocate resources
        wave_id    = self.vgprPool.checkOut(1) # quotient
        # constant
        lsu         = kernel["LocalSplitU"]
        du          = kernel["DepthU"]
        lsuStride   = du // lsu
        numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
        # generate instruction
        module.add(vectorStaticDivide(wave_id, "Serial", kernel["WavefrontSize"] * numWaves, tmpVgprRes, comment="LSU offset: Get LSU wave_id"))
        module.add(VMovB32(dst=vgpr(tmpVgpr), src=hex(lsuStride)))
        module.add(VMulLOU32(dst=vgpr(wave_id), src0=vgpr(tmpVgpr), src1=vgpr(wave_id), \
          comment="LSU offset: lsuoffset = wave_id*lsuStride(%u)" % (lsuStride)))
        module.add(VAddU32(dst=vgpr(qReg), src0=vgpr(wave_id), src1=vgpr(qReg), \
          comment="LSU Offset: offset += lsuoffset" ))
        self.vgprPool.checkIn(wave_id)

    else:
      module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))

    if kernel["WaveSeparateGlobalRead%s"%tc] == 1:
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr), src=vgpr("Serial"), comment="WaveIdxWavefrontWidth"))
        module.add(SLShiftRightB32(dst=sgpr(tmpSgpr), src=sgpr(tmpSgpr), shiftHex=hex(log2(kernel["WavefrontSize"])), comment="WaveId"))
        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=(kernel[tP["lsp"]] * tP["nrp"]), \
            comment="Each wave loads continuous lsp(%u)*nrp(%u) columns" % (kernel[tP["lsp"]], tP["nrp"])))
        module.add(VAddU32(dst=vgpr(qReg), src0=sgpr(tmpSgpr), src1=vgpr(qReg), \
            comment="Add back to column index"))
      self.vgprPool.checkIn(dividendReg)
      self.vgprPool.checkIn(dummy)
    elif kernel["WaveSeparateGlobalRead%s"%tc] == 2:
      module.add(VLShiftRightB32(vgpr(dividendReg), hex(log2(kernel["WavefrontSize"])), vgpr("Serial"), "WaveID"))
      module.add(VMovB32(vgpr(dummy), kernel["NumLoadsPerpendicular%s"%tc]*kernel["NumThreads"]//kernel["WavefrontSize"], comment="Global Read Wave: add back to cloumn index"))
      module.add(VMulLOU32(vgpr(qReg), vgpr(dummy), vgpr(qReg), "Global Read Wave: add back to cloumn index"))
      module.add(VAddU32(vgpr(qReg), vgpr(dividendReg), vgpr(qReg), "Global Read Wave: add back to cloumn index"))
      self.vgprPool.checkIn(dividendReg)
      self.vgprPool.checkIn(dummy)

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      if tP["glvw"] > 1:
        if tP["tlu"]:
          module.addComment0("tile *= glvw")
          module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), tP["glvw"], tmpSgprInfo))
        else:
          module.addComment0("unroll *= glvw")
          module.add(vectorStaticMultiply(vgpr(uReg), vgpr(uReg), tP["glvw"], tmpSgprInfo))


    uReg2 = self.vgprPool.checkOut(1, "uReg2", self.states.preventVgprOverflowDuringNewTile)
    module.add(VMovB32(dst=vgpr(uReg2), src=vgpr(uReg), comment="copy for GlobalSplitU"))
    tP["gpr"]["uReg2"] = uReg2
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["uReg"] = uReg
    self.vgprPool.checkIn(tmpVgpr)

    # store DirectToVgpr K interval for later use
    if tP["isA"]:
      self.states.dtvKIntervalA = dtvKInterval
    elif tP["isB"]:
      self.states.dtvKIntervalB = dtvKInterval

    return module

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    module = Module("lwaUnrollAssignment")
    uReg = tP["gpr"]["uReg2"]# if kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1 else "uReg"]
    module.addComment0("lwaUnrollAssignment%s = %s" % (tP["tensorChar"], vgpr(uReg)))
    return module

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP):
    module = Module("lwaFirstOffset")
    tc = tP["tensorChar"]
    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.states.unrollChar, tP["tileChar"])
    uReg = tP["gpr"]["uReg2"] # if kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1 else "uReg"]
    if kernel["LocalWriteUseSgpr%s"%tc]:
      destVgpr = self.vgprPool.checkOut(1, "destVgpr", self.states.preventVgprOverflowDuringNewTile)
    else:
      destVgpr = "LocalWriteAddr%s"%tc

    if kernel["UnrollMajorLDS%s" % tc]:
      lds_stride = kernel["_DepthU%s"%tc] + LdsPad
      module.add(VMulU32U24(dst=vgpr(destVgpr), src0=hex(lds_stride), src1=vgpr(tP["gpr"]["lwoT"]), \
          comment="lw%s%s**(DepthU_Compute + PAD)"%(tc, self.states.unrollChar)))
      if log2(tP["bpeDS"]) > 0:
        module.add(VAddLShiftLeftU32(dst=vgpr(destVgpr), src0=vgpr(uReg), src1=vgpr(destVgpr), shiftHex=hex(log2(tP["bpeDS"])), \
            comment="lwFO%s = (lw%s%s + lw%s%s*(DepthU+PAD))*bpeDS" % (tc, tc, tc, tc, self.states.unrollChar) ))
      else:
        module.add(VAddU32(dst=vgpr(destVgpr), src0=vgpr(uReg), src1=vgpr(destVgpr), \
            comment="lwFO%s = (lw%s%s + lw%s%s*(DepthU+PAD))*bpeDS(1)" % (tc, tc, tc, tc, self.states.unrollChar) ))
    else:
      lds_stride = kernel["MacroTile%s"%tc] + LdsPad
      module.add(VMulU32U24(dst=vgpr(destVgpr), src0=hex(lds_stride), src1=vgpr(uReg), \
          comment="lw%s%s**(MT%s + PAD)"%(tc, self.states.unrollChar, tc)))
      if log2(tP["bpeDS"]) > 0:
        module.add(VAddLShiftLeftU32(dst=vgpr(destVgpr), src0=vgpr(tP["gpr"]["lwoT"]), src1=vgpr(destVgpr), shiftHex=hex(log2(tP["bpeDS"])), \
            comment="lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpeDS" % (tc, tc, tc, tc, self.states.unrollChar, tP["tileChar"]) ))
      else:
        module.add(VAddU32(dst=vgpr(destVgpr), src0=vgpr(tP["gpr"]["lwoT"]), src1=vgpr(destVgpr), \
            comment="lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpeDS(1)" % (tc, tc, tc, tc, self.states.unrollChar, tP["tileChar"]) ))

    # LdsBlockSizePerPad: add padding
    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      tmpVgpr = self.vgprPool.checkOutAligned(2, 2)
      tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
      module.add(vectorStaticDivide(tmpVgpr, destVgpr, kernel["LdsBlockSizePerPad%s"%tc], tmpVgprRes, \
        "padding %u per block %u" % (kernel["LdsPad%s"%tc] * tP["bpeDS"], kernel["LdsBlockSizePerPad%s"%tc])))
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(vectorStaticMultiplyAdd(vgpr(destVgpr), vgpr(tmpVgpr), kernel["LdsPad%s"%tc] * tP["bpeDS"], vgpr(destVgpr), tmpSgprInfo, \
          "padding %u per block %u" % (kernel["LdsPad%s"%tc] * tP["bpeDS"], kernel["LdsBlockSizePerPad%s"%tc])))
      self.vgprPool.checkIn(tmpVgpr)

    if tP["isB"]:
      if kernel["LdsOffsetB"] != 0:
        module.add(VAddCOU32(
            dst=vgpr(destVgpr), \
            dst1=VCC(), \
            src0=hex(kernel["LdsOffsetB"]), \
            src1=vgpr(destVgpr), \
            comment="lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u" % (tP["tileChar"], \
            self.states.unrollChar, tP["tileChar"], kernel["LdsOffsetB"]) ))
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"] and tP["isM"]:
      if kernel["LdsOffsetMetadata"] != 0: # LdsOffsetMetadata can be 0 if DirectToVgprSparseMetadata is enabled
        module.add(VAddCOU32(
            dst=vgpr(destVgpr), \
            dst1=VCC(), \
            src0=hex(kernel["LdsOffsetMetadata"]), \
            src1=vgpr(destVgpr), \
            comment="lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_METADATA=%u" % (tP["tileChar"], \
            self.states.unrollChar, tP["tileChar"], kernel["LdsOffsetMetadata"])))

    numLwa = 0
    if tP["isA"]:
      numLwa = self.states.a.numVgprLocalWriteAddr
    elif tP["isB"]:
      numLwa = self.states.b.numVgprLocalWriteAddr
    elif tP["isM"]:
      numLwa = self.states.m.numVgprLocalWriteAddr

    if numLwa > 1:
      finalVgpr64K = vgpr("LocalWriteAddr%s+1"%tc)
      module.add(VAddU32(dst=finalVgpr64K, src0=0x10000, src1= vgpr(destVgpr), \
          comment="Final Offset Plus 64K" ))
    if numLwa > 2:
      finalVgpr128K = vgpr("LocalWriteAddr%s+2"%tc)
      module.add(VAddU32(dst=finalVgpr128K, src0= 0x20000, src1= vgpr(destVgpr), \
          comment="Final Offset Plus 128K" ))

    #LSC_ * LSP_
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    validWIPerLoad     = kernel[tP["lsc"]] * kernel[tP["lsp"]]//tP["glvw"]
    validBytesPerLoad  = kernel[tP["lsc"]] * kernel[tP["lsp"]] * numBytesPerElement
    maxBytesPerLoad    = kernel["NumThreads"] * tP["glvw"] * numBytesPerElement

    if kernel["WaveSeparateGlobalRead%s"%tc] == 1:
      validBytesPerLoad *= (kernel["NumThreads"] // self.states.kernel["WavefrontSize"])
    elif kernel["WaveSeparateGlobalRead%s"%tc] == 2:
      if kernel["ProblemType"]["TLU%s"%tc]:
        validBytesPerLoad *= (kernel["DepthU"] // kernel["NumLoadsPerpendicular%s"%tc] // (kernel["NumThreads"] // kernel["WavefrontSize"]))
      else:
        validBytesPerLoad *= (kernel["MacroTile%s"%tc] // kernel["NumLoadsPerpendicular%s"%tc] // (kernel["NumThreads"] // kernel["WavefrontSize"]))

    isDTVAB = ((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc])
    assert (validBytesPerLoad <= maxBytesPerLoad) or isDTVAB
    assert (kernel[tP["lsc"]] * kernel[tP["lsp"]] % tP["glvw"] == 0) or isDTVAB

    if validBytesPerLoad != maxBytesPerLoad:
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.add(SMovB32(dst=sgpr(tmpSgpr), src=validWIPerLoad, \
            comment="lsc*lsp=%u*%u"%(kernel[tP["lsc"]],kernel[tP["lsp"]] )))
        module.add(VCmpLtU32(
            dst=VCC(), \
            src0=vgpr("Serial"), \
            src1=sgpr(tmpSgpr), \
            comment="fractional: ensure tid < global read tile elements"))
        tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.states.preventVgprOverflowDuringNewTile)
        module.add(VMovB32(dst=vgpr(tmpVgpr), src=hex(self.consts.ldsOOB)))
        module.add(VCndMaskB32(
                    dst=vgpr(destVgpr), \
                    src0=vgpr(tmpVgpr), \
                    src1=vgpr(destVgpr), \
                    comment="Mask load so out-of-gr-tile bounds returns 0"))
        self.vgprPool.checkIn(tmpVgpr)

    if kernel["LocalWriteUseSgpr%s"%tc]:
      # TODO: Can refactor code above to Compute this directly:
      if self.states.archCaps["CrosslaneWait"]:
        module.add(SNop(waitState=0, comment="1 wait states required before reading vgpr by lane"))
      module.add(VReadfirstlaneB32(
          dst=sgpr("LocalWriteAddr%s"%tc), \
          src=vgpr(destVgpr), \
          comment="Copy lds write address VGPR to SGPR"))
      self.vgprPool.checkIn(destVgpr)

    if kernel["StoreSwapAddr"]:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        # needed for the VReadfirstlaneB32 in the prior code block
        if self.states.archCaps["CrosslaneWait"]:
          module.add(SNop(waitState=0, comment="1 wait states"))
        module.add(SAddU32(dst=sgpr("Swap%s"%tc), src0=sgpr("LocalWriteAddr%s"%tc), src1=kernel["LdsOffsetA_Blk"], comment="Calculate starting lds addr of second buffer"))
        module.add(SXorB32(dst=sgpr("Swap%s"%tc), src0=sgpr("Swap%s"%tc), src1=sgpr("LocalWriteAddr%s"%tc), comment="xor both lds buffer offsets to enable swapping"))
      else:
        module.add(VAddU32(dst=vgpr("LocalWriteSwapAddr%s"%tc), src0=kernel["LdsOffsetA_Blk"], src1=vgpr("LocalWriteAddr%s"%tc), \
                           comment="starting lds addr of second buffer" ))
        module.add(VXorB32(dst=vgpr("LocalWriteSwapAddr%s"%tc), \
                          src0=vgpr("LocalWriteSwapAddr%s"%tc), \
                          src1=vgpr("LocalWriteAddr%s"%tc), \
                          comment="xor both lds offsets to enable swapping"))

    return Module("lwaUnrollAssignment (Empty)") if self.dontAppendCode or isDTVAB else module

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  def lraTileAssignment(self, kernel, tPA, tPB):
    module = Module("lraTileAssignment")

    component = Component.LraTileAssignment.find(self)

    tP0 = tPA if tPB["tile01Idx"] else tPB
    tP1 = tPB if tPB["tile01Idx"] else tPA

    if component:
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP0["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        module.add(component(self, kernel, tP0))
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP1["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        module.add(component(self, kernel, tP1))
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]
        module.add(component(self, kernel, tPM))

    return module

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    module = Module("lraFinalOffset")

    tc = tP["tensorChar"]
    # do not generate local read code if DirectToVgpr is enabled
    if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]:
      return Module("lraFinalOffset (Empty)")

    if kernel["EnableMatrixInstruction"]:
      # allocate resources
      wave_id    = self.vgprPool.checkOut(1) # quotient
      rReg       = self.vgprPool.checkOut(1) # remainder, unused here
      tmpVgpr    = self.vgprPool.checkOutAligned(2, 2,"tmpVgpr")
      tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

      # constant
      tc          = tP["tensorChar"]
      tile01      = tP["tile01Idx"]
      LdsPad      = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
      mtAddPad    = kernel["MacroTile%u" % tile01] + LdsPad
      umlds       = kernel["UnrollMajorLDS%s" % tc]
      lsu         = kernel["LocalSplitU"]
      du          = kernel["DepthU"]
      lsuStride   = du // lsu
      numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]

      # generate instruction
      module.add(vectorStaticDivide(wave_id, "Serial", kernel["WavefrontSize"], tmpVgprRes))
      module.add(vectorStaticDivide(wave_id, wave_id, numWaves, tmpVgprRes, comment="LSU offset: Get LSU wave_id"))
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        if umlds == False:
          module.add(SMovB32(dst=sgpr(tmpSgpr), src=mtAddPad*lsuStride, \
            comment="LSU offset: stride = lsuStride(%u)*(MT%u(%u) + PAD%u(%u))" % (lsuStride,tile01, kernel["MacroTile%u" % tile01], tile01, LdsPad)))
        else:
          module.add(SMovB32(dst=sgpr(tmpSgpr), src=lsuStride, \
            comment="LSU offset: stride = lsuStride(%u) when umlds==True" % (lsuStride)))
        module.add(VMulLOU32(dst=vgpr(wave_id), src0=sgpr(tmpSgpr), src1=vgpr(wave_id), \
          comment="LSU offset: lsuoffset = wave_id*lsuStride*(MT%u+PAD)"%tile01))

      # final offset
      finalVgpr = vgpr("LocalReadAddr%s"%tc)
      if log2(tP["bpeDS"]) > 0:
        module.add(VAddLShiftLeftU32(dst=finalVgpr, src0=vgpr(wave_id), src1=vgpr(tP["gpr"]["lro"]), shiftHex=hex(log2(tP["bpeDS"])), \
          comment="Final Offset: offset = (lro%s+lsuoffset)*bpeDS" % tile01 ))
      else:
        module.add(VAddU32(dst=finalVgpr, src0=vgpr(wave_id), src1=vgpr(tP["gpr"]["lro"]), \
          comment="Final Offset: offset = (lro%s+lsuoffset)*bpeDS(1)" % tile01 ))

      # LdsBlockSizePerPad: add padding
      if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] !=0:
        module.add(vectorStaticDivide(rReg, "LocalReadAddr%s"%tc, kernel["LdsBlockSizePerPad%s"%tc], tmpVgprRes, \
          "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc] * tP["bpeDS"], kernel["LdsBlockSizePerPad%s"%tc])))
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          module.add(vectorStaticMultiplyAdd(vgpr("LocalReadAddr%s"%tc), vgpr(rReg), kernel["LdsPad%s"%tc] * tP["bpeDS"], vgpr("LocalReadAddr%s"%tc), tmpSgprInfo, \
                                       "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc] * tP["bpeDS"], kernel["LdsBlockSizePerPad%s"%tc])))

      # release resources
      self.vgprPool.checkIn(tmpVgpr)
      self.vgprPool.checkIn(wave_id)
      self.vgprPool.checkIn(rReg)
      self.vgprPool.checkIn(tP["gpr"]["lro"])

    else:
      # dot2: support WaveSplitk
      # constant
      tile01      = tP["tile01Idx"]
      LdsPad      = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
      divisor     = kernel["SubGroup0"] * kernel["SubGroup1"]
      mtAddPad    = kernel["MacroTile%u" % tile01] + LdsPad
      lrvw        = self.states.lrvwUnrollA if tc == 'A' else self.states.lrvwUnrollB

      # final offset
      finalVgpr = vgpr("LocalReadAddr%s"%tc)

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        if kernel["UseDotInstruction"]:
          kidx = self.vgprPool.checkOut(1) # remainder
          module.add(VAndB32(dst=vgpr(kidx), src0=(kernel["NumWaveSplitK"]-1), src1=vgpr("Serial"), comment="kidx = Serial % NumWaveSplitK"))
          module.add(vectorStaticMultiply(vgpr(kidx), vgpr(kidx), lrvw, tmpSgprInfo, \
            "*= lrvw"))
          # Final offset
          module.add(VAddLShiftLeftU32(dst=finalVgpr, shiftHex=hex(log2(tP["bpe"])), src0=vgpr(kidx), src1=vgpr(tP["gpr"]["lro"]), \
            comment="Final Offset: add padding %u per block %u" % (kernel["LdsPad%s"%tc] * tP["bpeDS"], kernel["LdsBlockSizePerPad%s"%tc])))
          self.vgprPool.checkIn(kidx)
        else:
          sgid = self.vgprPool.checkOut(1) # quotient
          vtmp = self.vgprPool.checkOut(1) # tmp
          vCont = ContinuousRegister(vtmp, 1)
          module.add(vectorStaticDivide(sgid, "Serial", divisor, vCont, \
            "LSU offset: sgid = Serial / subGroup(%u)" % divisor))
          module.add(vectorStaticMultiply(vgpr(sgid), vgpr(sgid), mtAddPad, tmpSgprInfo, \
            "LSU offset: lsuoffset = sgid*(MT%u+PAD)"%tile01))
          # module.add(SMovB32(dst=sgpr(tmpSgpr), src=mtAddPad*lsuStride, \
          #   comment="LSU offset: stride = lsuStride(%u)*(MT%u(%u) + PAD%u(%u))" % (lsuStride,tile01, kernel["MacroTile%u" % tile01], tile01, LdsPad)))
          module.add(vectorStaticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), kernel["VectorWidthB"], tmpSgprInfo, \
            "Final Offset: lr%sOffset * VW" % tc))
          module.add(VAddLShiftLeftU32(dst=finalVgpr, shiftHex=hex(log2(tP["bpe"])), src0=vgpr(sgid), src1=vgpr(tP["gpr"]["lro"]), \
            comment="Final Offset: add padding %u per block %u" % (kernel["LdsPad%s"%tc] * tP["bpeDS"], kernel["LdsBlockSizePerPad%s"%tc])))
          self.vgprPool.checkIn(sgid)
          self.vgprPool.checkIn(vtmp)

      # release resources
      self.vgprPool.checkIn(tP["gpr"]["lro"])

      # LdsBlockSizePerPad: add padding
      if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] !=0:
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          tmpSgpr = tmpSgprInfo.idx
          rReg    = self.vgprPool.checkOut(1) # remainder, unused here
          module.add(vectorStaticDivide(rReg, "LocalReadAddr%s"%tc, kernel["LdsBlockSizePerPad%s"%tc], tmpSgpr, \
            "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
          module.add(vectorStaticMultiplyAdd(vgpr("LocalReadAddr%s"%tc), vgpr(rReg), kernel["LdsPad%s"%tc] * tP["bpe"], vgpr("LocalReadAddr%s"%tc), tmpSgprInfo, \
            "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc] * tP["bpeDS"], kernel["LdsBlockSizePerPad%s"%tc])))
          self.vgprPool.checkIn(rReg)

    return module

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    module = Module("lraOffsetConversionForDTLandNLC")
    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
    divisorName = tP["lvc"]
    divisor = kernel[divisorName]
    width = kernel["WavefrontSize"] if tP["tlu"] else kernel["DepthU"]
    if divisor < width:
      # DirectToLds + above conditions, rotate offset_val bits to adjust LDS offset
      lowerScale = tP["nrc"]
      upperScale = (kernel["WavefrontSize"] // divisor)
      # bit rotation necessary only when nrc > 1
      if lowerScale > 1:
        tile01 = tP["tile01Idx"]
        rightShift = int(log2(lowerScale)) # assuming power of 2
        leftShift = int(log2(upperScale)) # assuming power of 2
        line = kernel["MacroTile%u" % tile01] if tP["tlu"] else kernel["DepthU"]
        ldsLineSize = line * tP["bpe"] // lowerScale
        maskBitsLow = (lowerScale - 1) * ldsLineSize
        maskBitsHigh = (upperScale - 1) * lowerScale * ldsLineSize
        maskBitsAll = (maskBitsLow | maskBitsHigh)

        # offset_val conversion
        low = offset_val & maskBitsLow
        high = offset_val & maskBitsHigh
        low <<= leftShift
        high >>= rightShift
        val = low | high
        offset_val = (offset_val & (~maskBitsAll)) | val

        # generate asm code
        if generateAsm:
          with self.allocTmpSgpr(1) as tmpSgprInfo:
            tmpSgpr2 = tmpSgprInfo.idx
            module.add(VAndB32(dst=vgpr(tmp1), src0=hex(maskBitsLow), src1=finalVgpr, \
              comment="Offset rotation for DirectToLds + %s > 1"%tP["lsc"]))
            module.add(VAndB32(dst=vgpr(tmp2), src0=hex(maskBitsHigh), src1=finalVgpr))
            module.add(VLShiftLeftB32(dst=vgpr(tmp1), shiftHex=hex(leftShift), src=vgpr(tmp1)))
            module.add(VLShiftRightB32(dst=vgpr(tmp2), shiftHex=hex(rightShift), src=vgpr(tmp2)))
            module.add(VOrB32(dst=vgpr(tmp1), src0=vgpr(tmp1), src1=vgpr(tmp2)))
            module.add(SMovB32(dst=sgpr(tmpSgpr2), src=hex(maskBitsAll)))
            module.add(VNotB32(dst=vgpr(tmp2), src=sgpr(tmpSgpr2)))
            module.add(VAndB32(dst=finalVgpr, src0=vgpr(tmp2), src1=finalVgpr))
            module.add(VOrB32(dst=finalVgpr, src0=vgpr(tmp1), src1=finalVgpr))

    return module, offset_val

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    module = Module("lraDeclareAddresses")
    if tP["isA"]:
      module.addComment0("N/A")

    else:
      # no need to generate add code if LdsOffset is 0 or DirectToVgprB
      if kernel["LdsOffset%s"%tP["tensorChar"]] == 0 or tP["isB"] and kernel["DirectToVgprB"]:
        module = Module("lraDeclareAddresses (Empty)")
      else:
        module.add(VAddCOU32(
            dst=vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
            dst1=VCC(), \
            src0=hex(kernel["LdsOffset%s"%tP["tensorChar"]]), \
            src1=vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
            comment=" += LdsOffset%s (lower)"%tP["tensorChar"]))
    numLra = 0
    if tP["isA"]:
      numLra = self.states.a.numVgprLocalReadAddr
    elif tP["isB"]:
      numLra = self.states.b.numVgprLocalReadAddr
    elif tP["isM"]:
      numLra = self.states.m.numVgprLocalReadAddr

    if numLra > 1:
      finalVgpr64K = vgpr("LocalReadAddr%s+1"%tP["tensorChar"])
      module.add(VAddU32(dst=finalVgpr64K, src0=0x10000, src1= vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          comment="Final vgprLocalReadAddr%s+1 Offset Plus 64K"%tP["tensorChar"] ))
    if numLra > 2:
      finalVgpr128K = vgpr("LocalReadAddr%s+2"%tP["tensorChar"])
      module.add(VAddU32(dst=finalVgpr128K, src0=0x20000, src1= vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          comment="Final vgprLocalReadAddr%s+2 Offset Plus 128K"%tP["tensorChar"] ))
    return module

  def lraSwapAddressesForDTLPad(self, kernel, tP):
    module = Module("lraSwapAddressesForDTLPad")

    tc = tP["tensorChar"]
    if kernel["StoreSwapAddr"]:
      module.add(VAddU32(dst=vgpr("LocalReadSwapAddr%s"%tc), src0=kernel["LdsOffsetA_Blk"], src1=vgpr("LocalReadAddr%s"%tc), \
                         comment="Calculate starting lds addr of second buffer" ))
      module.add(VXorB32(dst=vgpr("LocalReadSwapAddr%s"%tc), \
                         src0=vgpr("LocalReadSwapAddr%s"%tc), \
                         src1=vgpr("LocalReadAddr%s"%tc), \
                         comment="xor both lds buffer offsets to enable swapping"))
    return module


  def lwaInitAddressesForDTLTailLoop(self, kernel, tP):
    module = Module("lwaInitAddressesForDTLTailLoop")

    tc = tP["tensorChar"]
    waveSize = kernel["WavefrontSize"]
    if kernel["DirectToLds%c"%tc] and kernel["NonDTLTailLoop%s"%tc]:
      module.addComment0("Set local write offsets for %c to be same as DTL %uB load"%(tc, kernel["GlobalReadVectorWidth%c"%tc] * tP["bpe"]))
      module.add(VAndB32(dst=vgpr("LocalWriteAddr%c"%tc), src0=(waveSize - 1), src1=vgpr("Serial"), comment="Serial % wavesize"))
      module.add(VLShiftLeftB32(dst=vgpr("LocalWriteAddr%c"%tc), shiftHex=hex(log2(kernel["GlobalReadVectorWidth%c"%tc] * tP["bpe"])), src=vgpr("LocalWriteAddr%c"%tc), comment=""))
      module.add(VAddU32(dst=vgpr("LocalWriteAddr%s"%tc), src0=sgpr("LocalWriteAddr%c"%tc), src1=vgpr("LocalWriteAddr%s"%tc), \
                         comment="" ))
      numLw = 0
      if tP["isA"]:
        numLw = self.states.a.numVgprLocalWriteAddrTailLoop
      elif tP["isB"]:
        numLw = self.states.b.numVgprLocalWriteAddrTailLoop
      for i in range(1,numLw):
        module.add(VAddU32(dst=vgpr("LocalWriteAddr%s + %s"%(tc, i)), src0=(i * 0x10000), \
                           src1=vgpr("LocalWriteAddr%s"%tc), comment="" ))
    return module

  ##############################################################################
  # openShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def openShadowInit(self):
    module = Module("openShadowInit")
    module.add(Label("ShadowInitStart", ""))
    return module

  ##############################################################################
  # closeShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def closeShadowInit(self, kernel):
    module = Module("closeShadowInit")
    assert(self.states.doShadowInit and kernel["PrefetchGlobalRead"])

    module.add(self.checkLastIter(kernel))
    if kernel["SuppressNoLoadLoop"]:
      loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx]]
      lastIterEnd = Label("LoopEnd%s"%loopChar, "")
    else:
      lastIterEnd = Label("PrefetchGlobalLastIterEnd", "")

    # This branch could potentially be very far e.g. > SIMM16
    module.addComment1("after InitC, skip to end of prefetch last iter if numIter==0")
    # use positive offset only long jump
    with self.allocTmpSgpr(3) as tmpSgprInfo:
      module.add(self.longBranchScc1(lastIterEnd, posNeg=1, tmpSgprInfo=tmpSgprInfo))

    return module

  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    module = Module("initC")
    self.vgprPool.remove(self.states.c.startVgprValu, self.states.c.numVgprValu, "ValuC")
    module.addComment1("initC: remove ValuC vgpr buffer [%u...%u) from pool"%(self.states.c.startVgprValu, self.states.c.startVgprValu+self.states.c.numVgprValu))
    numAccvgprs = self.states.totalAgprs
    self.agprPool.remove(0, numAccvgprs, "ValuC")
    module.addComment1("initC: remove acc vgpr buffer [%u...%u) from pool"%(0, numAccvgprs))
    self.vgprPool.remove(self.states.a.startVgprValu , self.states.lastValuAB - self.states.a.startVgprValu , "ValuAB")
    module.addComment1("initC: remove ValuA/B vgpr buffer [%u...%u) from pool"%(self.states.a.startVgprValu , self.states.lastValuAB))
    numCVgpr = self.states.c.numVgprValu + numAccvgprs

    if kernel["MfmaInitCVgprs"] == True and self.states.asmCaps["HasMFMA_f8f6f4"] and numCVgpr >= 32:
      tmpVgpr = self.vgprPool.checkOutAligned(2,2,"tmp vgpr for lds init C registers")
      module.add(VMovB64(dst=vgpr(tmpVgpr,2), src=0, comment="A/B=0"))

      chunkSize = 16
      for i in range(0, min(16, numCVgpr)):
        copyInst = VMovB32 if i >= numAccvgprs else VAccvgprWrite
        regStr = vgpr("ValuC+%u"%(i-numAccvgprs)) if i >= numAccvgprs else accvgpr(i)
        module.add(copyInst(dst=regStr, src=0, comment="initC"))

      for i in range(min(16, numCVgpr), numCVgpr, 16):
        if i + 16 <= numCVgpr:
          module.add(MFMAInstruction(instType=InstType.INST_I8, accType=InstType.INST_I32, variant=[32,32,16,1], mfma1k=False, \
                                     acc=accvgpr(i,16), a=vgpr(tmpVgpr,2), b=vgpr(tmpVgpr,2), acc2=accvgpr(0,16), \
                                     comment="initC: [%u, %u]"%(i, i+15)))
        else:
          for j in range(i,min(i+16, numCVgpr)):
            copyInst = VMovB32 if j >= numAccvgprs else VAccvgprWrite
            regStr = vgpr("ValuC+%u"%(j-numAccvgprs)) if j >= numAccvgprs else accvgpr(j)
            module.add(copyInst(dst=regStr, src=0, comment="initC"))

      self.vgprPool.checkIn(tmpVgpr)
    else:
      if kernel["LdsInitCVgprs"]:
        tmpAddr = self.vgprPool.checkOut(1,"tmp vgpr for lds init C registers")
        module.add(VMovB32(dst=vgpr(tmpAddr), src=self.consts.ldsOOB, comment="set out-of-bound addr"))

      for i in range(0, numCVgpr):
        copyInst = VMovB32 if i >= numAccvgprs else VAccvgprWrite
        regStr = vgpr("ValuC+%u"%(i-numAccvgprs)) if i >= numAccvgprs else accvgpr(i)
        if not kernel["LdsInitCVgprs"]:
          module.add(copyInst(dst=regStr, src=0, comment="initC"))
        else:
          module.add(DSLoadB32(dst=regStr, src=vgpr(tmpAddr), ds=DSModifiers(offset=0), comment="initC"))

      if kernel["LdsInitCVgprs"]:
        self.vgprPool.checkIn(tmpAddr)

    return module

  def initSumUnroll(self, kernel):
    return self.exclasses.biasSumUnroll.initSumUnroll(self, kernel)

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  # Output: Sets sgpr(StaggerRowMask)
  ##############################################################################
  def declareStaggerParms(self, kernel):
    module = Module("declareStaggerParms")
    #Calculate StaggerUIter
    with self.allocTmpSgpr(4) as tmpSgprInfo:
      beginStaggerUIterLabel = Label("beginStaggerUIter",comment="")
      endStaggerUIterLabel = Label("endStaggerUIter", comment="")
      tmpSgpr = tmpSgprInfo.idx
      currentStaggerU = tmpSgpr
      shiftedStaggerU = tmpSgpr + 1
      staggerUMask = tmpSgpr + 1
      staggerUStrideShift = tmpSgpr + 2
      staggerUMapping = tmpSgpr + 3
      module.add(SAndB32(dst=sgpr(staggerUStrideShift), src0=sgpr("StaggerU"), src1=hex(0x1F00)))
      module.add(SLShiftRightB32(dst=sgpr(staggerUStrideShift), shiftHex=hex(8), src=sgpr(staggerUStrideShift)))
      module.add(SAndB32(dst=sgpr(staggerUMapping), src0=sgpr("StaggerU"), src1=hex(0xE000)))
      module.add(SAndB32(dst=sgpr("StaggerU"), src0=sgpr("StaggerU"), src1=hex(0xFF)))
      module.add(SMovB32(dst=sgpr(currentStaggerU), src=sgpr("StaggerU"), comment="init staggerU"))
      module.add(beginStaggerUIterLabel)
      module.add(SLShiftLeftB32(dst=sgpr(shiftedStaggerU), src=sgpr(currentStaggerU), \
              shiftHex=sgpr(staggerUStrideShift), comment="shift by StaggerUStride"))
      module.add(SCmpGeU32(src0=sgpr("OrigLoopCounter"), src1=sgpr(shiftedStaggerU), \
          comment="loopCount >= current shift Count" ))
      module.add(SCBranchSCC1(labelName=endStaggerUIterLabel.getLabelName(), comment="jump to end"))
      module.add(SLShiftRightB32(dst=sgpr(currentStaggerU), src=sgpr(currentStaggerU), \
              shiftHex=1, comment="step down to smaller stagger"))
      module.add(SBranch(labelName=beginStaggerUIterLabel.getLabelName(), comment="jump to begin"))
      module.add(endStaggerUIterLabel)
      module.add(SSubU32(dst=sgpr(staggerUMask), src0=sgpr(currentStaggerU), src1=1, comment="staggerU mask"))
      module.add(SCmpGeU32(src0=sgpr(currentStaggerU), src1=1, \
          comment="if current staggerU >= 1" ))
      module.add(SCSelectB32(dst=sgpr("StaggerUIter"), src0=sgpr(staggerUMask), src1=0, comment="set Mask"))

      staggerInput = tmpSgpr
      staggerLabel = Label("staggerInputEnd", comment="")
      for i in range(0, 5):
        label = Label("StaggerUMapping_%d"%(i + 1), comment="")
        module.add(SCmpEQU32(src0=sgpr(staggerUMapping), src1=hex(i << 13)))
        if i != 4:
          module.add(SCBranchSCC1(labelName=label.getLabelName()))
        else:
          module.add(SCBranchSCC1(labelName=staggerLabel.getLabelName()))
        if i == 0:
          module.add(SMovB32(dst=sgpr(staggerInput), src=sgpr("WorkGroup0")))
        elif i == 1:
          module.add(SMovB32(dst=sgpr(staggerInput), src=sgpr("WorkGroup1")))
        elif i == 2 and len(kernel["ProblemType"]["IndicesBatch"]) > 2:
          module.add(SMovB32(dst=sgpr(staggerInput), src=sgpr("WorkGroup2")))
        elif i == 3:
          wgSerial = staggerInput
          tmp = tmpSgpr+1
          if len(kernel["ProblemType"]["IndicesBatch"]) > 2:
            module.add(SMulI32(dst=sgpr(wgSerial), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), \
              comment="wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0"))
            module.add(SMulI32(dst=sgpr(wgSerial), src0=sgpr(wgSerial), src1=sgpr("WorkGroup2")))
          module.add(SMulI32(dst=sgpr(tmp), src0=sgpr("NumWorkGroups0"), src1=sgpr("WorkGroup1")))
          module.add(SAddU32(dst=sgpr(wgSerial), src0=sgpr(wgSerial), src1=sgpr(tmp)))
          module.add(SAddU32(dst=sgpr(wgSerial), src0=sgpr(wgSerial), src1=sgpr("WorkGroup0")))
        else:
          module.add(SMovB32(dst=sgpr(staggerInput), src=hex(-1)))
        module.add(SBranch(staggerLabel.getLabelName()))
        if i != 4:
          module.add(label)
        else:
          module.add(staggerLabel)

      module.add(SAndB32(dst=sgpr("StaggerUIter"), src0=sgpr("StaggerUIter"), \
                src1=sgpr(staggerInput), \
                comment="Compute actual stagger start for this tile"))
      module.add(SLShiftLeftB32(dst=sgpr("StaggerUIter"), src=sgpr("StaggerUIter"), \
                shiftHex=sgpr(staggerUStrideShift), comment="shift by StaggerUStride"))

      skComponent = Component.StreamK.find(self)
      module.add(skComponent.declareStaggerParms(self, kernel))

    return module

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  def calculateStagger(self, kernel, tP):
    imod = Module("calculateStagger")
    tc = tP["tensorChar"]

    assert (kernel["BufferLoad"])

    with self.allocTmpSgpr(3) as tmpSgprInfo:
      staggerTmp    = tmpSgprInfo.idx
      incSparseSgpr = tmpSgprInfo.idx + 2

      #---
      imod.addComment1("SRDs += (StaggerUIter) * GlobalReadIncs%s+%u"% (tc, self.states.unrollIdx))

      # Calculate the stagger byte offset
      imod.addModuleAsFlatItems(self.s_mul_i64_i32(
                sgpr(staggerTmp), sgpr(staggerTmp+1), \
                sgpr("StaggerUIter"), sgpr("GlobalReadIncs%s+%u"%(tc, self.states.unrollIdx)), \
                " stagger byte offset"))

      # Amount of bytes to add to get back to start.
      # on the llop iteration which matches StaggerUIter, this offset added instead of GlobalReadInc
      imod.addModuleAsFlatItems(self.s_mul_i64_i32(sgpr("WrapU%s+0"%tc), sgpr("WrapU%s+1"%tc), \
                self.loopCounter(kernel, self.states.unrollIdx), sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                "Number of bytes accessed by the unroll loop"))

      imod.add(SSubU32(dst=sgpr("WrapU%s+0"%tc),  \
                src0=sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                src1=sgpr("WrapU%s+0"%tc), \
                comment="remove one iteration"))
      imod.add(SSubBU32(dst=sgpr("WrapU%s+1"%tc), \
                src0=0, \
                src1=sgpr("WrapU%s+1"%tc), \
                comment="remove one iteration"))

      imod.add(self.incrementSrd(tP, sgpr(staggerTmp), sgpr(staggerTmp+1)))

      if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"] and \
         ((kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"])):
        imod.addComment1("SRDs += (StaggerUIter) * GlobalReadIncsMetadata")

        tc = "Metadata"
        if kernel["DirectToVgprSparseMetadata"]:
          incSparse = incSparseSgpr
          imod.add(self.calculateIncrementMetadata(kernel, incSparse))
        else:
          incSparse = "GlobalReadIncsMetadata+%u"%(self.states.unrollIdx)
        imod.addModuleAsFlatItems(self.s_mul_i64_i32( \
                        sgpr(staggerTmp), sgpr(staggerTmp+1), \
                        sgpr("StaggerUIter"), sgpr(incSparse), " stagger byte offset of metadata"))
        # Amount of bytes to add to get back to start.
        # on the llop iteration which matches StaggerUIter, this offset added instead of GlobalReadInc
        imod.addModuleAsFlatItems(self.s_mul_i64_i32( \
                  sgpr("WrapU%s+0"%tc), sgpr("WrapU%s+1"%tc), \
                  self.loopCounter(kernel, self.states.unrollIdx), sgpr(incSparse), \
                  "Number of bytes accessed by the unroll loop"))

        imod.add(SSubU32(sgpr("WrapU%s+0"%tc), sgpr(incSparse), sgpr("WrapU%s+0"%tc), " remove one iteration"))
        imod.add(SSubBU32(sgpr("WrapU%s+1"%tc), 0, sgpr("WrapU%s+1"%tc), " remove one iteration"))

        if kernel["DirectToVgprSparseMetadata"]:
          imod.add(self.incrementMetadataSrd(sgpr(staggerTmp), sgpr(staggerTmp+1)))
        else:
          imod.add(self.incrementSrd(tP["tpsMetadata"], sgpr(staggerTmp), sgpr(staggerTmp+1)))

    if tP["isB"]:
      isDTVAorB = (kernel["DirectToVgprA"] != kernel["DirectToVgprB"]) #  only one of them is enabled
      if kernel["PrefetchGlobalRead"] == 2 and isDTVAorB:
        # PGR2 + DTVA or B (only 1 side), need separate StaggerUIter for DTV load
        imod.add(SAddU32(dst=sgpr("StaggerUIterDTV"), src0=sgpr("StaggerUIter"), \
                src1=(1), \
                comment="Subtract (PGR-1); StaggerUIter now contains target iteration to wrap"))
      # Convert passed in S' to S for easy loop comparison.  S=S-(PGR-1)'
      imod.add(SAddU32(dst=sgpr("StaggerUIter"), src0=sgpr("StaggerUIter"), \
              src1=(2 if kernel["PrefetchGlobalRead"] else 1), \
              comment="Subtract (PGR-1); StaggerUIter now contains target iteration to wrap"))
    return imod

  ##############################################################################
  # Remove stagger offset (before tail loop)
  # |          |           |   |
  # |-- S'*I --|
  # |---------- W' --------|-I-|
  #           ^ current SRD pos
  # ^unrollLoopStart           ^tailLoopStart   (in summation0 dimension)

  #
  # S = sgprStaggerUIter = S+(PGR+1)'
  # W = sgprWrapU
  # PGR = kernel["PrefetchGlobalRead"]
  #
  # S' = StaggUIter that is passed into the kernel = -PGR+1+S
  # S'*I is also the global read offset (from unrollLoopStart) at unroll loop exit ?
  # I = GlobalReadIncs
  # W' = W

  # Need to move it to tailLoopStart

  # To compute position where tail loop should start:
  #  = W' - S'*I + I
  #  = W - (S+PGR+1)*I) + I
  #  = W - (S+PGR+1)*I + I
  #  = W - (S+2+PGR)*I
  ##############################################################################
  def removeStagger(self, kernel, tP):
    imod = Module("removeStagger")
    tc = tP["tensorChar"]
    with self.allocTmpSgpr(3) as tmpSgprInfo:
      tmp = tmpSgprInfo.idx
      tmpIncSparse = tmpSgprInfo.idx + 2
      # might be able to refactor this to eliminate signed math
      imod.add(SSubI32(dst=sgpr(tmp), src0=3 if kernel["PrefetchGlobalRead"] else 2, \
              src1=sgpr("StaggerUIter")))
      imod.addModuleAsFlatItems(self.s_mul_i64_i32_u32(sgpr(tmp), sgpr(tmp+1), \
                  sgpr(tmp), sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                  "start offset S in bytes"))
      imod.add(SSubU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr("WrapU%s"%tc), comment="S - WrapU"))
      imod.add(SSubBU32(dst=sgpr(tmp+1), src0=sgpr(tmp+1), src1=sgpr("WrapU%s+1"%(tc)), comment="S - WrapU"))

      imod.add(self.incrementSrd(tP, sgpr(tmp), sgpr(tmp+1)))

      if kernel["ProblemType"]["Sparse"] and \
         ((kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"])):
        tc = "Metadata"
        if kernel["DirectToVgprSparseMetadata"]:
          incSparse = tmpIncSparse
          imod.add(self.calculateIncrementMetadata(kernel, incSparse))
        else:
          incSparse = "GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)

        # might be able to refactor this to eliminate signed math
        imod.add(SSubI32(dst=sgpr(tmp), src0=3 if kernel["PrefetchGlobalRead"] else 2, \
                src1=sgpr("StaggerUIter")))
        imod.addModuleAsFlatItems(self.s_mul_i64_i32_u32(sgpr(tmp), sgpr(tmp+1), \
                    sgpr(tmp), sgpr(incSparse), \
                     "start offset S in bytes"))
        imod.add(SSubU32(sgpr(tmp), sgpr(tmp), sgpr("WrapU%s"%tc), "S - WrapU"))
        imod.add(SSubBU32(sgpr(tmp+1), sgpr(tmp+1), sgpr("WrapU%s+1"%(tc)), "S - WrapU"))

        if kernel["DirectToVgprSparseMetadata"]:
          imod.add(self.incrementMetadataSrd(sgpr(tmp), sgpr(tmp+1)))
        else:
          imod.add(self.incrementSrd(tP["tpsMetadata"], sgpr(tmp), sgpr(tmp+1)))

    return imod

  ##############################################################################
  # Set parameters UNDEF in the macro module
  ##############################################################################
  def undefineMacroModule(self, macroModule):
    imod = Module()
    for inst in macroModule.items():
      if isinstance(inst, (RegSet, ValueSet)):
        imod.add(ValueSet(name=inst.name, value="UNDEF", format = -1))
      elif isinstance(inst, Module):
        imod.add(self.undefineMacroModule(inst))
    return imod

  ##############################################################################
  # Check in VGPRs which is allocated for tail loop
  ##############################################################################
  def tailLoopFreeVgpr(self, vgprBase, moduleMacro):
    imod = Module("tailLoopFreeVgpr")
    if vgprBase != -1:
      imod.add(self.undefineMacroModule(moduleMacro))
      self.vgprPool.checkIn(vgprBase)
    return imod

  ##############################################################################
  # Check out VGPRs which is used for tail loop
  # The functions will retuen either
  # a single pair of (imod, vgprBase)
  # or
  # a array of pairs of (imod, vgprBase)'s
  # User must call tailLoopFreeVgpr(vgprBase, imod) to release the resources.
  ##############################################################################
  def tailLoopAllocValuVgpr(self, kernel, tensorParametersA, tensorParametersB, tensorParametersM):
    imodA            = Module("tailLoopAllocValuAVgpr")
    vgprBaseA        = -1
    numValuA         = 0
    numVgprValuPackA = 0
    if self.states.a.numVgprValu > 0 and not kernel["DirectToVgprA"]:
      numValuA = self.states.a.numVgprValu
      if tensorParametersA["bpe"] < 4 and not kernel["UnrollMajorLDSA"] and not kernel["enableLDSTrA"]:
        if self.states.lrvwTileA > 1:
          numVgprValuPackA = ceil(kernel["VectorWidthA"] * tensorParametersA["bpe"] / self.states.bpr) * kernel["MIWaveTileA"] // kernel["VectorWidthA"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadA"]
          if self.states.packDTVA:
            # pack DTV case, double the number
            numVgprValuPackA *= 2
          elif (kernel["UsePLRPack"] and self.states.numItersPLR):
            numVgprValuPackA //= 2
        else:
          numVgprValuPackA = self.states.a.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackA * (int(4/tensorParametersA["bpeDS"]) - 1)
      vgprBaseA = self.vgprPool.checkOutAligned(numValuA + numVgprValuPackA, 2)
      imodA.add(RegSet("v", "vgprValuA_X0_I0_BASE", vgprBaseA))
      imodA.add(self.moduleVgprMacroValuA)
      if numVgprValuPackA > 0:
        imodA.add(RegSet("v", "vgprValuA_X0_I0_D0_PACK", vgprBaseA + numValuA))
        imodA.add(self.moduleVgprMacroValuAPack)

    imodB            = Module("tailLoopAllocValuBVgpr")
    vgprBaseB        = -1
    numValuB         = 0
    numVgprValuPackB = 0
    if self.states.b.numVgprValu > 0 and not kernel["DirectToVgprB"]:
      numValuB = self.states.b.numVgprValu
      if tensorParametersB["bpe"] < 4 and not kernel["UnrollMajorLDSB"] and not kernel["enableLDSTrB"]:
        if self.states.lrvwTileB > 1:
          numVgprValuPackB = ceil(kernel["VectorWidthB"] * tensorParametersB["bpe"] / self.states.bpr) * kernel["MIWaveTileB"] // kernel["VectorWidthB"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadB"]
          if self.states.packDTVB:
            # pack DTV case, double the number
            numVgprValuPackB *= 2
          elif (kernel["UsePLRPack"] and self.states.numItersPLR):
            numVgprValuPackB //= 2
        else:
          numVgprValuPackB = self.states.b.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackB * (int(4/tensorParametersB["bpeDS"]) - 1)

      vgprBaseB = self.vgprPool.checkOutAligned(numValuB + numVgprValuPackB, 2)
      imodB.add(RegSet("v", "vgprValuB_X0_I0_BASE", vgprBaseB))
      imodB.add(self.moduleVgprMacroValuB)
      if numVgprValuPackB > 0:
        imodB.add(RegSet("v", "vgprValuB_X0_I0_D0_PACK", vgprBaseB + numValuB))
        imodB.add(self.moduleVgprMacroValuBPack)

    imodM                   = Module("tailLoopAllocValuMetadataVgpr")
    vgprBaseM               = -1
    numValuM                = 0
    numVgprValuPackMetadata = 0
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      numValuM = self.states.m.numVgprValu
      if not kernel["UnrollMajorLDSMetadata"]:
        if self.states.lrvwTileMetadata > 1:
          miWaveTile = kernel["MIWaveTileB"] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTileA"]
          numVgprValuPackMetadata = roundUp(kernel["VectorWidthMetadata"] * tensorParametersM["bpe"] / self.states.bpr) * miWaveTile // kernel["VectorWidthMetadata"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadMetadata"]
        else:
          numVgprValuPackMetadata = self.states.m.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackMetadata * (int(4/tensorParametersM["bpe"]) - 1)
      # No PGR metadata for tail loop.
      vgprBaseM = self.vgprPool.checkOutAligned(max(numValuM + numVgprValuPackMetadata, self.states.m.numVgprG2LAllocated), 2)
      imodM.add(RegSet("v", "vgprValuMetadata_X0_I0_BASE", vgprBaseM))
      imodM.add(self.moduleVgprMacroValuM)
      if numVgprValuPackMetadata > 0:
        imodM.add(RegSet("v", "vgprValuMetadata_X0_I0_D_PACK", vgprBaseM + numValuM))
        imodM.add(self.moduleVgprMacroValuMPack)

    imodMisc        = Module("tailLoopAllocValuMiscVgpr")
    vgprBaseMisc    = -1
    numVgprPackTemp = 0
    if ((tensorParametersA["bpe"] < 4 and not kernel["UnrollMajorLDSA"])                                   \
        or (tensorParametersB["bpe"] < 4 and not kernel["UnrollMajorLDSB"])                                \
        or (not kernel["UnrollMajorLDSMetadata"] and kernel["MIInputPerThreadMetadata"] == 4))             \
        and (kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()):
      numVgprPackTemp = 1
    numVgprCvtTemp = 0
    if self.states.a.startVgprValuCvtTemp != -1 or self.states.b.startVgprValuCvtTemp != -1:
      numVgprCvtTemp = 2
    if numVgprPackTemp + numVgprCvtTemp > 0:
      vgprBaseMisc = self.vgprPool.checkOut(numVgprPackTemp + numVgprCvtTemp)
      if numVgprPackTemp > 0:
        imodMisc.add(RegSet("v", "vgprPackTemp", vgprBaseMisc))
      if numVgprCvtTemp > 0:
        imodMisc.add(RegSet("v", "vgprCvtTemp", vgprBaseMisc + numVgprPackTemp))

    return ([vgprBaseA, imodA],[vgprBaseB, imodB],[vgprBaseM, imodM],[vgprBaseMisc, imodMisc])

  def tailLoopAllocG2LVgpr(self, kernel):
    imod     = Module("tailLoopAllocG2LVgpr")
    vgprBase = -1
    numG2LA  = 0
    numG2LB  = 0
    numG2LMetadata  = 0
    if not kernel["DirectToVgprA"]:
      if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
        numG2LA = self.states.a.numVgprG2LTailloopAllocated*2
      else:
        numG2LA = self.states.a.numVgprG2LTailloopAllocated
    if not kernel["DirectToVgprB"]:
      if ("ULSGRODoubleG2L" in kernel) and kernel["ULSGRODoubleG2L"] == 1:
        numG2LB = self.states.b.numVgprG2LTailloopAllocated*2
      else:
        numG2LB = self.states.b.numVgprG2LTailloopAllocated
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      numG2LMetadata = self.states.m.numVgprG2LAllocated

    if numG2LB > 0:
      # TODO: B alignment hack
      numG2LA = ((numG2LA+1)//2)*2
    if numG2LMetadata > 0:
      # TODO: M alignment hack
      numTemp1 = numG2LA + numG2LB
      numTemp2 = ((numTemp1+1)//2)*2
      numG2LB += (numTemp2 - numTemp1)
    if numG2LA + numG2LB + numG2LMetadata > 0:
      vgprBase = self.vgprPool.checkOutAligned(numG2LA + numG2LB + numG2LMetadata, 2)
      imod.addComment0("Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (%d,%d,%d)"%(numG2LA,numG2LB,numG2LMetadata))
    if numG2LA > 0:
      imod.add(RegSet("v", "vgprG2LA_BASE", vgprBase))
      if kernel["DirectToLdsA"] and kernel["NonDTLTailLoopA"]:
        imod.add(RegSet("v", "vgprG2LA", "vgprG2LA_BASE", 0))
      else:
        imod.add(self.moduleVgprMacroG2LA)
    if numG2LB > 0:
      imod.add(RegSet("v", "vgprG2LB_BASE", vgprBase + numG2LA))
      if kernel["DirectToLdsB"] and kernel["NonDTLTailLoopB"]:
        imod.add(RegSet("v", "vgprG2LB", "vgprG2LB_BASE", 0))
      else:
        imod.add(self.moduleVgprMacroG2LB)
    if numG2LMetadata > 0:
      imod.add(RegSet("v", "vgprG2LMetadata", vgprBase + numG2LA + numG2LB))

    return imod, vgprBase

  def tailLoopAllocDTLLWVgpr(self, kernel):
    imod     = Module("tailLoopAllocDTLLWVgpr")
    vgprBase = -1
    numLWA   = self.states.a.numVgprLocalWriteAddrTailLoop
    numLWB   = self.states.b.numVgprLocalWriteAddrTailLoop

    if numLWA + numLWB > 0:
      vgprBase = self.vgprPool.checkOutAligned(numLWA + numLWB, 2)
      imod.addComment0("Check out VGPR (numLWA,numLWB) = (%d,%d)"%(numLWA,numLWB))
    if numLWA > 0 and (kernel["DirectToLdsA"] and kernel["NonDTLTailLoopA"]):
      imod.add(RegSet("v", "vgprLocalWriteAddrA", vgprBase))
    if numLWB > 0 and (kernel["DirectToLdsB"] and kernel["NonDTLTailLoopB"]):
      imod.add(RegSet("v", "vgprLocalWriteAddrB", vgprBase + numLWA))

    return imod, vgprBase


  def tailLoopAllocDTVVgpr(self, kernel, tensorParametersA, tensorParametersB):
    imodA     = Module("tailLoopAllocDTVVgprA")
    vgprBaseA = -1
    if kernel["DirectToVgprA"]:
      if self.states.packDTVA or self.states.convDTVA:
        numValuA = self.states.a.numVgprValu
        numVgprValuPackA = 0
        if tensorParametersA["bpe"] < 4 and not kernel["UnrollMajorLDSA"]:
          if self.states.lrvwTileA > 1:
            numVgprValuPackA = ceil(kernel["VectorWidthA"] * tensorParametersA["bpe"] / self.states.bpr) * kernel["MIWaveTileA"] // kernel["VectorWidthA"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadA"]
            if self.states.packDTVA:
              # pack DTV case, double the number
              numVgprValuPackA *= 2
          else:
            numVgprValuPackA = self.states.a.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackA * (int(4/tensorParametersA["bpeDS"]) - 1)

        vgprBaseA = self.vgprPool.checkOutAligned(numValuA + numVgprValuPackA, 2)
        imodA.add(RegSet("v", "vgprValuA_X0_I0_BASE", vgprBaseA))
        if numVgprValuPackA > 0:
          imodA.add(RegSet("v", "vgprValuA_X0_I0_D0_PACK", vgprBaseA + numValuA))
        if self.states.packDTVA:
          #DTVA is a.startVgprValuPack
          imodA.add(RegSet("v", "vgprG2LA_BASE", vgprBaseA + numValuA))
        elif self.states.convDTVA:
          #DTVA is a.startVgprValu
          imodA.add(RegSet("v", "vgprG2LA_BASE", vgprBaseA))
        imodA.add(self.moduleVgprMacroG2LA)
        imodA.add(self.moduleVgprMacroValuA)
        imodA.add(self.moduleVgprMacroValuAPack)
      else:
        #DTVA is G2LA itself
        if kernel["ULSGRODoubleG2L"] == 1:
          numG2LA = self.states.a.numVgprG2LAllocated*2
        else:
          numG2LA = self.states.a.numVgprG2LAllocated
        vgprBaseA = self.vgprPool.checkOutAligned(numG2LA, 2)
        imodA.add(RegSet("v", "vgprG2LA_BASE", vgprBaseA))
        imodA.add(self.moduleVgprMacroG2LA)

    imodB     = Module("tailLoopAllocDTVVgprA")
    vgprBaseB = -1
    if kernel["DirectToVgprB"]:
      if self.states.packDTVB or self.states.convDTVB:
        numValuB = self.states.b.numVgprValu
        numVgprValuPackB = 0
        if tensorParametersB["bpe"] < 4 and not kernel["UnrollMajorLDSB"]:
          if self.states.lrvwTileB > 1:
            numVgprValuPackB = ceil(kernel["VectorWidthB"] * tensorParametersB["bpe"] / self.states.bpr) * kernel["MIWaveTileB"] // kernel["VectorWidthB"] * kernel["InnerUnroll"] * self.states.numVgprBuffer * kernel["MIInputPerThreadB"]
            if self.states.packDTVB:
              # pack DTV case, double the number
              numVgprValuPackB *= 2
          else:
            numVgprValuPackB = self.states.b.numVgprValuPerBlock * kernel["InnerUnroll"] * self.states.numVgprBufferPackB * (int(4/tensorParametersB["bpeDS"]) - 1)

        vgprBaseB = self.vgprPool.checkOutAligned(numValuB + numVgprValuPackB, 2)
        imodB.add(RegSet("v", "vgprValuB_X0_I0_BASE", vgprBaseB))
        if numVgprValuPackB > 0:
          imodB.add(RegSet("v", "vgprValuB_X0_I0_D0_PACK", vgprBaseB + numValuB))
        if self.states.packDTVB:
          #DTVB is b.startVgprValuPack
          imodB.add(RegSet("v", "vgprG2LB_BASE", vgprBaseB + numValuB))
        elif self.states.convDTVB:
          #DTVB is b.startVgprValu
          imodB.add(RegSet("v", "vgprG2LB_BASE", vgprBaseB))
        imodB.add(self.moduleVgprMacroG2LB)
        imodB.add(self.moduleVgprMacroValuB)
        imodB.add(self.moduleVgprMacroValuBPack)
      else:
        #DTVB is G2LB itself
        if kernel["ULSGRODoubleG2L"] == 1:
          numG2LB = self.states.b.numVgprG2LAllocated*2
        else:
          numG2LB = self.states.b.numVgprG2LAllocated
        vgprBaseB = self.vgprPool.checkOutAligned(numG2LB, 2)
        imodB.add(RegSet("v", "vgprG2LB_BASE", vgprBaseB))
        imodB.add(self.moduleVgprMacroG2LB)

    imodM     = Module("tailLoopAllocDTVVgprA")
    vgprBaseM = -1
    if kernel["ProblemType"]["Sparse"]:
      if kernel["DirectToVgprSparseMetadata"]:
        numValuM = self.states.m.numVgprValu
        vgprBaseM = self.vgprPool.checkOut(numValuM)
        imodM.add(RegSet("v", "vgprValuMetadata_X0_I0_BASE", vgprBaseM))
        imodM.add(self.moduleVgprMacroValuM)

    return ([vgprBaseA, imodA],[vgprBaseB, imodB],[vgprBaseM, imodM])

  ##############################################################################
  # Using wider load instructions to improve the GR efficiency in tail loop.
  # If loading size is smaller than a dword(32bit), it will return 0 instead.
  # Need to use buffer_load_d16 to load the data which is out of boundary.
  ##############################################################################
  def tailLoopGlobalRead(self, kernel, tPA, tPB, doA, doB):
    imod = Module("tailLoopGlobalRead")

    glvwWorkaround = 8 * kernel["ProblemType"]["DataType"].numRegisters()
    dataTypeA = kernel["ProblemType"]["DataType"] if tPA["glvw"] < glvwWorkaround else \
                kernel["ProblemType"]["DataTypeA"]
    dataTypeB = kernel["ProblemType"]["DataType"] if tPB["glvw"] < glvwWorkaround else \
                kernel["ProblemType"]["DataTypeB"]
    numElementsPerLoadA = -1
    numElementsPerLoadB = -1
    if dataTypeA.isHalf() or dataTypeA.isBFloat16():
      if tPA["glvw"] > 1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
        numElementsPerLoadA = 2
    if dataTypeB.isHalf() or dataTypeB.isBFloat16():
      if tPB["glvw"] > 1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
        numElementsPerLoadB = 2
    if numElementsPerLoadA == 2:
      doA = False
    if numElementsPerLoadB == 2:
      doB = False

    tagList = ["AddressA", "AddressB", "WrapUA", "WrapUB", "StaggerU", "WGM", \
               "StaggerUIter", "GlobalReadIncsA", "GlobalReadIncsB", \
               "sgprStridesA", "sgprStridesB", "sgprShadowLimitA", "sgprShadowLimitB"]
    lastRegTag = None
    spool = self.sgprPool.getPool()
    imod.addComment1("release sgprs that will not be used")
    for i in range(0, self.sgprPool.size()):
      regTag = spool[i].tag
      if regTag != lastRegTag:
        lastRegTag = regTag
        if (lastRegTag not in self.states.nonPostLoopSgpr) and \
           (spool[i].status == RegisterPool.Status.InUse) and \
           (lastRegTag in tagList):
          imod.add(self.undefineSgpr(regTag))

    loadALabel  = Label(label="LoadA", comment="")
    loadBLabel  = Label(label="LoadB", comment="")
    mergeALabel = Label(label="MergeA", comment="")
    mergeBLabel = Label(label="MergeB", comment="")
    tailGlobalLoadEndLabel   = Label(label="TailGlobalLoadEnd", comment="")
    checkOtherLoadALabel = Label(label="CheckOtherLoadA", comment="")
    checkAddrALabel = Label(label="CheckAddrA", comment="")
    checkOtherLoadBLabel = Label(label="CheckOtherLoadB", comment="")
    checkAddrBLabel = Label(label="CheckAddrB", comment="")
    lspA = kernel[tPA["lsp"]]
    lscA = kernel[tPA["lsc"]]
    lspB = kernel[tPB["lsp"]]
    lscB = kernel[tPB["lsc"]]
    nlcA = kernel["NumLoadsCoalescedA"]
    nlcB = kernel["NumLoadsCoalescedB"]
    nlpA = kernel["NumLoadsPerpendicularA"]
    nlpB = kernel["NumLoadsPerpendicularB"]

    numElementsPer4BytesA = int(4 / tPA["bpeGR"])
    numElementsPer4BytesB = int(4 / tPB["bpeGR"])
    maxNumOOBElementsA = numElementsPer4BytesA - 1
    maxNumOOBElementsB = numElementsPer4BytesB - 1

    numDwordA = (tPA["glvw"] * tPA["bpeGR"]) >> 2
    numDwordB = (tPB["glvw"] * tPB["bpeGR"]) >> 2
    numDwordA = 1 if numDwordA == 0 else numDwordA
    numDwordB = 1 if numDwordB == 0 else numDwordB
    numTmpVgpr = maxNumOOBElementsA * numDwordA + maxNumOOBElementsB * numDwordB # 2 fo 16b

    numSingleSgpr = 10
    numPairSgpr = 2
    singSgprList = []  # store single sgpr
    pairSgprList = []  # store aligned2 sgpr

    for num in range(numSingleSgpr):
      tmpSgpr = self.sgprPool.checkOut(1, preventOverflow=False)
      singSgprList.append(tmpSgpr)
    for num in range(numPairSgpr):
      tmpSgpr = self.sgprPool.checkOutAligned(2, 2, preventOverflow=False)
      pairSgprList.append(tmpSgpr)

    # If load bytes of each thread is a multiple of 4-byte, data can be fully loaded by using
    # buffer_load_dword, if not, load single element by buffer_load_short_d16 additionally. It's
    # affected by WaveSeparateGlobalRead, MT, DU.
    # This function finds the last element locates in which global read tile (load range of each
    # global read instruction.
    # sReloadFlag: 0 is no need to reload.
    # sLoadTileIdx: The last element locates in which load tile.
    # sLoadNum: How many elements need to be reloaded.
    # sValidBytes: How many valid bytes can be loaded in tail loop.
    def generateFindTheLastElementLocation(tc):
      do = doA if  tc == "A" else doB
      strSize = "SizeI" if tc == "A" else "SizeJ"
      strMacroTile = "MacroTile0" if tc == "A" else "MacroTile1"
      strWG = "WorkGroup0" if tc == "A" else "WorkGroup1"
      tP = tPA if tc == "A" else tPB
      nlp = nlpA if tc == "A" else nlpB
      nlc = nlcA if tc == "A" else nlcB
      lsp = lspA if tc == "A" else lspB
      lsc = lscA if tc == "A" else lscB
      strWSGR = "WaveSeparateGlobalReadA" if tc == "A" else "WaveSeparateGlobalReadB"
      maxNumOOBElements = maxNumOOBElementsA if tc == "A" else maxNumOOBElementsB

      sReloadFlag = singSgprList[2] if tc == "A" else singSgprList[3]
      sTmp0 = singSgprList[0] if tc == "A" else singSgprList[1]
      sTmp1 = singSgprList[4] if tc == "A" else singSgprList[5]
      sTmp2 = singSgprList[6] if tc == "A" else singSgprList[7]
      sTmp3 = singSgprList[8] if tc == "A" else singSgprList[9]
      sx2Tmp0 = pairSgprList[0]
      sx2Tmp1 =  pairSgprList[1]
      sLoadTileIdx = sTmp0
      sLoadNum = sTmp1
      sValidBytes = sTmp2

      loopIdx = self.states.unrollIdx
      if do:
        imod.addComment("Calculate %s %% %s"%(strSize, strMacroTile))
        imod.add(SMulI32(dst=sgpr(sTmp0), src0=sgpr(strWG), src1=kernel[strMacroTile], \
                         comment="Calculate the remaining dimension along I/J direction."))
        imod.add(SSubU32(dst=sgpr(sTmp0), src0=sgpr(strSize), src1=sgpr(sTmp0), \
                         comment="Calculate the remaining dimension along I/J direction."))
        imod.add(SMulI32(dst=sgpr(sTmp0), src0=sgpr(sTmp0), src1=tP["bpeGR"], \
                         comment="In bytes"))
        imod.add(SAndB32(dst=sgpr(sTmp1), src0=sgpr("SizeL"), src1=(kernel["DepthU"] - 1), \
                         comment="Calculate the remaining dimension along L direction."))
        imod.add(SLShiftRightB32(dst=sgpr(sx2Tmp1), shiftHex=hex(log2(lsc)), \
                                 src=sgpr(sTmp1), comment="Divided by lsc(%s)"%(lsc)))
        imod.add(self.s_mul_u64_u32(sgpr(sValidBytes), sgpr(sReloadFlag), sgpr(sTmp0), sgpr(sTmp1), \
                               comment="Calculate total number of valid elements."))
        imod.add(SCmpGtU32(src0=sgpr(sReloadFlag), src1=0))
        imod.add(SCMovB32(dst=sgpr(sValidBytes), src=hex(0xFFFFFFFF), \
                          comment="If valid elements > max(U32), set the value to max"))

        if (kernel[strWSGR] == 0):
          tmpSgpr = sTmp0
        else:
          tmpSgpr = sTmp3
        imod.add(SSubU32(dst=sgpr(sTmp1), src0=sgpr(strSize), src1=1, \
                         comment="sLoadTileIdx starts from 0"))
        imod.addComment("Calculate %s - 1 %% %s"%(strSize, strMacroTile))
        imod.add(scalarStaticDivideAndRemainder(tmpSgpr, tmpSgpr, sTmp1, \
                                                kernel[strMacroTile], \
                                                ContinuousRegister(sx2Tmp0, 2), 1))
        if (kernel[strWSGR] > 0):
          imod.addComment("Calculate %s %% (nlp * lsp)"%(tmpSgpr))
          imod.add(scalarStaticDivideAndRemainder(sTmp0, sTmp0, tmpSgpr, \
                                                  (nlp * lsp), \
                                                  ContinuousRegister(sx2Tmp0, 2), 1))
        imod.add(SLShiftRightB32(dst=sgpr(sTmp0), shiftHex=hex(log2(lsp)), \
                                 src=sgpr(sTmp0), comment="Divide lsp to get the load tile index"))
        imod.add(SMulI32(dst=sgpr(sTmp0), src0=sgpr(sTmp0), src1=nlc, comment="Multiply nlc"))
        imod.add(SAddI32(dst=sgpr(sLoadTileIdx), src0=sgpr(sTmp0), src1=sgpr(sx2Tmp1), \
                         comment=""))
        imod.add(scalarStaticDivideAndRemainder(sx2Tmp0, sTmp1, "SizesSum+%u"%loopIdx, \
                                                kernel["DepthU"], ContinuousRegister(sx2Tmp0, 2), \
                                                2))
        imod.add(SAndB32(dst=sgpr(sLoadNum), src0=sgpr(sTmp1), src1=(tP["glvw"] - 1), \
                         comment="sLoadNum = (SizesSum+%u"%(loopIdx)+" mod DU) & glvw"))
        imod.add(SAndB32(dst=sgpr(sReloadFlag), src0=sgpr(sLoadNum), src1=hex(maxNumOOBElements), \
                         comment=""))
    # The end of generateFindTheLastElementLocation

    # Generate code for jumping branch to correct load tile (sLoadTileIdx) and load correct number
    # of elements by checking sLoadNum.
    # Extract the global load instraction and merge instruction of each load tile and load number
    # from globalReadGuardK.
    # What "LOAD" do is buffer_load_short_d16 v0, vAddr, sgprSrd, soffset.
    # What "MERGE" do is s_waitcnt vmcnt(0)
    #                    v_or_b32 vData, vData, v0
    # Flow: GRA -> GRB -> MergeA -> MergeB.
    def generateLoadAndMergeSingleElement():
      def generateJumpBranch(idx):
        def generateDetailsForBehavior(tP, tmpVgpr, behavior, jumpLabel, tileSgpr, kSgpr):
            self.param = TailOptParams()
            tc = tP["tensorChar"]
            bpe = tP["bpeGR"]
            numThreadsPerp = kernel[tP["lsp"]]
            numThreadsCoal = int(kernel["NumThreads"] / numThreadsPerp)
            nlp = kernel["NumLoadsPerpendicular%c"%(tc)]
            nlc = kernel["NumLoadsCoalesced%c"%(tc)]
            numTiles = nlp * nlc
            numElementsPer4Bytes = int(4 / bpe)
            def generateFunction(idx, bevavior, jumpLabel, tileSgpr, kSgpr):
              tileLabelsList = []
              kLabelsList = []
              if (idx >= 0):
                labelStr = bevavior + "_"+tc+str(idx)
                labelTmp = Label(label=labelStr, comment="")
                tileLabelsList.append(labelTmp)
                if (idx != 0):
                  imod.add(SCmpEQU32(src0=sgpr(tileSgpr), src1=idx, comment=""))
                  imod.add(SCBranchSCC1(labelName=labelTmp.getLabelName(), comment=""))

                generateFunction(idx-1, bevavior, jumpLabel, tileSgpr, kSgpr)

                imod.add(labelTmp)
                for i in range(tP["glvw"], 0, -1):
                  if i % numElementsPer4Bytes != 0:
                    labelStrK = labelStr+"_K"+str(i)
                    labelTmp = Label(label=labelStrK, comment="")
                    kLabelsList.append(labelTmp)
                if idx == (numTiles - 1):
                  finalLoop = 1
                else:
                  finalLoop = 0

                self.param.idx = idx
                self.param.jumpLabel = jumpLabel
                self.param.tmpVgpr = tmpVgpr
                self.param.kLabelsList = kLabelsList
                self.param.behavior = behavior
                self.param.kSgpr = kSgpr
                self.param.finalLoop = finalLoop
                imod.add(self.globalReadGuardK(kernel, tP, 1, self.param))
            generateFunction(numTiles - 1, behavior, jumpLabel, tileSgpr, kSgpr)
        # The End of generateDetailsForBehavior

        tc = ["A", "B"]
        do = [doA, doB]
        tP = [tPA, tPB]
        loadLabel = [loadALabel, loadBLabel]
        sReloadFlag = [sReloadFlagA, sReloadFlagB]
        sLoadTileIdx = [sLoadTileIdxA, sLoadTileIdxB]
        sLoadNum = [sLoadNumA, sLoadNumB]
        tmpVgprStartIdx = [0, (maxNumOOBElementsA * numDwordA)]
        mergeLabel = [mergeALabel, mergeBLabel]
        numElementsPerLoad = [numElementsPerLoadA, numElementsPerLoadB]
        if (idx >= 0 and idx < len(tc)):
          if tc[idx] == "A":
            jumpLabel = loadBLabel if doA and doB else mergeALabel
          else:
            jumpLabel = mergeALabel if doA and doB else mergeBLabel
          if do[idx]:
            imod.addComment1("load single element for %s"%(tc[idx]))
            imod.add(loadLabel[idx])
            imod.add(SCmpEQU32(src0=sgpr(sReloadFlag[idx]), src1=0, \
                               comment="Valid loading size per thread is multiples of 4 bytes"))
            imod.add(SCBranchSCC1(labelName=jumpLabel.getLabelName(),
                                  comment="Skip loading "+tc[idx]))
            generateDetailsForBehavior(tP[idx], tmpVgpr + tmpVgprStartIdx[idx], "LOAD", jumpLabel,
                                       sLoadTileIdx[idx], sLoadNum[idx])

          generateJumpBranch(idx+1)

          idx = idx ^ 1
          if do[idx]:
            if tc[idx] == "A":
              jumpLabel = mergeBLabel if doA and doB else checkOtherLoadALabel
            else:
              jumpLabel = checkOtherLoadALabel if doA and doB else checkOtherLoadBLabel
            imod.addComment1("merge single element for %s"%(tc[idx]))
            imod.add(mergeLabel[idx])
            imod.add(SCmpEQU32(src0=sgpr(sReloadFlag[idx]), src1=0, \
                               comment="Valid loading size per thread is multiples of 4 bytes"))
            imod.add(SCBranchSCC1(labelName=jumpLabel.getLabelName(),
                                  comment="Skip mergeing "+tc[idx]))
            generateDetailsForBehavior(tP[idx], tmpVgpr + tmpVgprStartIdx[idx], "MERGE", \
                                       jumpLabel, sLoadTileIdx[idx], sLoadNum[idx])
      generateJumpBranch(0)
      # The end of generateJumpBranch
    # The end of generateFindTheLastElementLocation

    # Generate tht jump branch for checking other load tiles.
    def jumpLabel(tP, tmpSgpr, jumpLabel):
      tc = tP["tensorChar"]
      nlp = kernel["NumLoadsPerpendicular%c"%(tc)]
      nlc = kernel["NumLoadsCoalesced%c"%(tc)]
      numTiles = nlp * nlc
      def func(idx, tmpSgpr):
        tileLabelsList = []
        if (idx >= 0):
          labelStr = tc+str(idx)
          labelTmp = Label(label=labelStr, comment="")
          tileLabelsList.append(labelTmp)
          if (idx != 0):
            imod.add(SCmpEQU32(src0=sgpr(tmpSgpr), src1=idx, comment=""))
            imod.add(SCBranchSCC1(labelName=labelTmp.getLabelName(), comment=""))
          func(idx-1, tmpSgpr)
          imod.add(labelTmp)
          if tc == "A":
            addrbase = self.globalread_gpr_record.a.addrVgpr[idx]
            offset = self.globalread_gpr_record.a.offset[idx]
          else:
            addrbase = self.globalread_gpr_record.b.addrVgpr[idx]
            offset = self.globalread_gpr_record.b.offset[idx]
          if kernel["_UseSgprForGRO"]:
            imod.add(VAddU32(dst=vgpr(tmpVgpr), src0=vgpr(addrbase), src1=offset))
          else:
            imod.add(VMovB32(dst=vgpr(tmpVgpr), src=vgpr(addrbase)))
          if idx != (numTiles - 1):
            imod.add(SBranch(labelName=jumpLabel.getLabelName(), comment=""))
      func(numTiles - 1, tmpSgpr)
    # The end of jumpLabel.

    # Even we have reloaded the load tile which the last element locates in, there might be some
    # other tiles access the last element./
    # So we check the upper load next step, and do the check recursively untill no threads in the
    # load tile accesses the last element.
    def generateReLoadLoop(tc):
      do = doA if tc == "A" else doB
      nlp = nlpA if tc == "A" else nlpB
      nlc = nlcA if tc == "A" else nlcB
      checkOtherLoadLabel = checkOtherLoadALabel if tc == "A" else checkOtherLoadBLabel
      sLoadTileIdx = sLoadTileIdxA if tc == "A" else sLoadTileIdxB
      tP = tPA if tc == "A" else tPB
      checkAddrLabel = checkAddrALabel if tc == "A" else checkAddrBLabel
      sValidBytes = sValidBytesA if tc == "A" else sValidBytesB
      loadLabel = loadALabel if tc == "A" else loadBLabel
      imod.add(checkOtherLoadLabel)

      # At the beginning of the loop, jump to the next step if:
      #   no need to reload (sReloadFlagA/B == 0) or
      #   sLoadCnt == total number of global load subtiles.
      # Increase sLoadCnt nlc in every loop.
      # For generating for loop B:
      #   LoopA/B share the same sgpr to record the load cnt. Reset it at the beginning of loop B.
      #   Force sReloadFlagB to be 0 to skip reload B when reload A. Reset it at the beginning of
      #   loop B.
      if do:
        if tc == "A":
          imod.add(SCmpEQU32(src0=sgpr(sReloadFlagA), src1=0,
                             comment="Noneed to load single element fo A?"))
          if doB:
            imod.add(SCBranchSCC1(labelName=checkOtherLoadBLabel.getLabelName(), comment=""))
          else:
            imod.add(SCBranchSCC1(labelName=tailGlobalLoadEndLabel.getLabelName(), comment=""))
          imod.add(SAddU32(dst=sgpr(sLoadCnt), src0=sgpr(sLoadCnt), src1=nlc))
          imod.add(SCmpEQU32(src0=sgpr(sLoadCnt), src1=(nlp * nlc),
                             comment="Have reloaded all subtiles?"))
          if doB:
            imod.add(SCMovB32(dst=sgpr(sLoadCnt), src=0, comment="Reset loop count"))
            imod.add(SCBranchSCC1(labelName=checkOtherLoadBLabel.getLabelName(), comment=""))
          else:
            imod.add(SCBranchSCC1(labelName=tailGlobalLoadEndLabel.getLabelName(), comment=""))
        else:
          if doA:
            imod.add(SMovB32(dst=sgpr(sReloadFlagA), src=0, comment="Force to skip reload A"))
            imod.add(SCmpEQU32(src0=sgpr(sLoadCnt), src1=0, comment="Loop start?"))
            imod.add(SCMovB32(dst=sgpr(sReloadFlagB), src=sgpr(sBackupSkipLoadB),
                              comment="Restore sReloadFlagB for B"))
          imod.add(SCmpEQU32(src0=sgpr(sReloadFlagB), src1=0,
                             comment="Noneed to load single element for B?"))
          imod.add(SCBranchSCC1(labelName=tailGlobalLoadEndLabel.getLabelName(), comment=""))
          imod.add(SAddU32(dst=sgpr(sLoadCnt), src0=sgpr(sLoadCnt), src1=nlc))
          imod.add(SCmpEQU32(src0=sgpr(sLoadCnt), src1=(nlp * nlc),
                             comment="Have reloaded all subtiles?"))
          imod.add(SCBranchSCC1(labelName=tailGlobalLoadEndLabel.getLabelName(), comment=""))

        imod.add(SSubI32(dst=sgpr(sLoadTileIdx), src0=sgpr(sLoadTileIdx), src1=nlc, \
                         comment="Check the upper subtile"))
        imod.add(SCmpLtI32(src0=sgpr(sLoadTileIdx), src1=0, comment=""))
        imod.add(SCSelectB32(dst=sgpr(sCmpLoadStartAddrStatusx2), src0=(nlp * nlc), src1=0, \
                             comment="Back to the last subtile"))
        imod.add(SAddI32(dst=sgpr(sLoadTileIdx), src0= sgpr(sLoadTileIdx),
                         src1=sgpr(sCmpLoadStartAddrStatusx2), \
                         comment="If currently reload the first subtile, \
                                  check the last subtile next."))
        jumpLabel(tP, sLoadTileIdx, checkAddrLabel)
        imod.add(checkAddrLabel)
        imod.add(VSubU32(dst=vgpr(tmpVgpr), src0=vgpr(tmpVgpr),
                         src1=self.states.srdShiftLeft[tc] * tP["bpeGR"], comment="sub prepad"))
        loadRangePerThread = tP["glvw"] * tP["bpeGR"] - 1
        imod.add(VAddU32(dst=vgpr(tmpVgpr+1), src0=vgpr(tmpVgpr), src1=loadRangePerThread, \
                         comment="Calculate load range per thread"))
        imod.add(VCmpLtI32(dst=sgpr(sCmpLoadStartAddrStatusx2, 2), src0=vgpr(tmpVgpr), \
                           src1=sgpr(sValidBytes), \
                           comment="If loading start address < total valid bytes?"))
        imod.add(VCmpGEI32(dst=sgpr(sCmpLoadEndAddrStatusx2, 2), src0=vgpr(tmpVgpr+1), \
                           src1=sgpr(sValidBytes), \
                           comment="If loading end address >= total valid bytes?"))
        imod.add(SAndB32(dst=sgpr(sCmpLoadStartAddrStatusx2), \
                         src0=sgpr(sCmpLoadStartAddrStatusx2), \
                         src1=sgpr(sCmpLoadEndAddrStatusx2), \
                         comment="Find threads which access the last element"))
        imod.add(SAndB32(dst=sgpr(sCmpLoadStartAddrStatusx2+1), \
                         src0=sgpr(sCmpLoadStartAddrStatusx2+1), \
                         src1=sgpr(sCmpLoadEndAddrStatusx2+1), \
                         comment="Find thread that access the last element"))
        imod.add(SAddU32(dst=sgpr(sCmpLoadStartAddrStatusx2), \
                         src0=sgpr(sCmpLoadStartAddrStatusx2), \
                         src1=sgpr(sCmpLoadStartAddrStatusx2+1), \
                         comment="Find thread that access the last element"))
        imod.add(SCmpLgU32(src0=sgpr(sCmpLoadStartAddrStatusx2), src1=0, \
                           comment="Have threads access the last element?"))
        if tc == "A":
          if doB:
            imod.add(SCMovB32(dst=sgpr(sReloadFlagB), src=0, comment="Skip reload B temporarily"))
            imod.add(SCSelectB32(dst=sgpr(sLoadCnt), src0=sgpr(sLoadCnt), src1=0, \
                                 comment="Reset loop count if needed"))
        imod.add(SCBranchSCC1(labelName=loadLabel.getLabelName(), comment="Reload "+tc))
    # The End of generateReLoadLoop

    # These sgprs must be the same as used in generateFindTheLastElementLocation
    sLoadTileIdxA = singSgprList[0]
    sLoadTileIdxB = singSgprList[1]
    sReloadFlagA = singSgprList[2]
    sReloadFlagB = singSgprList[3]
    sLoadNumA = singSgprList[4]
    sLoadNumB = singSgprList[5]
    sValidBytesA = singSgprList[6]
    sValidBytesB = singSgprList[7]
    sLoadCnt = singSgprList[8]
    sBackupSkipLoadB = singSgprList[9]
    sCmpLoadStartAddrStatusx2 = pairSgprList[0]
    sCmpLoadEndAddrStatusx2 = pairSgprList[1]

    imod.addComment1("find the last element location for a")
    generateFindTheLastElementLocation("A")
    imod.addComment1("find the last element location for b")
    generateFindTheLastElementLocation("B")

    if doA or doB:
      tmpVgpr = self.vgprPool.checkOut(numTmpVgpr)
      imod.add(SMovB32(dst=sgpr(sLoadCnt), src=0, comment="Set loop count = 0"))
      if doA and doB:
        imod.add(SMovB32(dst=sgpr(sBackupSkipLoadB), src=sgpr(sReloadFlagB), \
                 comment="Backup and will be restored in label_CheckB_OOB"))

    generateLoadAndMergeSingleElement()

    imod.addComment1("reload loop for a: check if there's other load range need to be reloaded")
    generateReLoadLoop("A")
    imod.addComment1("reload loop for b: check if there's other load range need to be reloaded")
    generateReLoadLoop("B")

    imod.addComment1("global read for tail done")
    imod.add(tailGlobalLoadEndLabel)

    if (doA and kernel["DirectToLds%s"%tPA["tensorChar"]]) or \
       (doB and kernel["DirectToLds%s"%tPB["tensorChar"]]):
      imod.add(SMovB32(dst=mgpr(0), src=hex(kernel["LdsNumBytes"]), \
            comment="Restore LDS clamp at %u bytes HERE"%(kernel["LdsNumBytes"])))

    if doA or doB:
      self.vgprPool.checkIn(tmpVgpr)
    for s in singSgprList:
      self.sgprPool.checkIn(s)
    for s in pairSgprList:
      self.sgprPool.checkIn(s)

    return imod

  ##############################################################################
  # Emit code to compute loop iterations for GSU.
  # See same function in KernelWriterSource.py for background explanation
  # This function is used to compute number of loop iters and also
  # for computing the global read increment for GSU case.
  # For multiple summation, the number of loop iterations needs to be reset
  # for each iteration so replicate the code in addr inc and at open of unroll loop

  # tmpSgpr is allocation of at least 3 tmpSgpr

  # Output: SGPR(destName) contains the number of unroll iterations for
  # this workgroup.
  ##############################################################################
  def calculateLoopNumIterGsu(self, kernel, destName, tmpSgprRes: ContinuousRegister):
    module = Module("calculateLoopNumIterGsu")

    loopCounter = sgpr(destName)
    quotient = destName
    remainder = "GSUSumIdx+1" # numIterPerWgRemainder
    dividend = destName

    tmpVgpr = self.vgprPool.checkOut(2,"tmp")
    tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
    module.add(SAndB32(dst=sgpr(remainder), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
    module.add(scalarUInt32DivideAndRemainder(quotient, dividend, remainder, remainder, tmpVgprRes, wavewidth=kernel["WavefrontSize"]))
    self.vgprPool.checkIn(tmpVgpr)

    # if gsuSumIdx < numIterPerWgRemainder
    module.add(SAddU32(dst=sgpr(tmpSgprRes.idx), src0=1, src1=loopCounter, comment="tmp<-numIterMyWg+1"))
    module.add(SCmpLtU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), comment="gsuSumIdx < numIterPerWgRemainder"))
    module.add(SCMovB32(dst=loopCounter, src=sgpr(tmpSgprRes.idx), comment="numIterMyWg++ if needed"))

    return module

  def calculateLoopNumIterOffsetGsu(self, kernel, destName, tmpSgprRes: ContinuousRegister):
    module = Module("calculateLoopNumIterOffsetGsu")

    loopCounter = sgpr(destName)
    quotient = destName
    remainder = "GSUSumIdx+1" # numIterPerWgRemainder
    dividend = destName

    tmpVgpr = self.vgprPool.checkOut(2,"tmp")
    tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
    module.add(SAndB32(dst=sgpr(remainder), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
    module.add(scalarUInt32DivideAndRemainder(quotient, dividend, remainder, remainder, tmpVgprRes, wavewidth=kernel["WavefrontSize"]))
    self.vgprPool.checkIn(tmpVgpr)

    # calculate offset number of loop iterations for each wg
    stmp = tmpSgprRes.idx
    module.add(SMulI32(dst=sgpr(stmp+1), src0=loopCounter, src1=sgpr("GSUSumIdx"), comment="quotient*GSUSumIdx"))
    module.add(SAddU32(dst=sgpr(stmp+0), src0=1, src1=loopCounter, comment="quotient+1"))
    module.add(SAddU32(dst=sgpr(stmp+1), src0=sgpr(stmp+1), src1=sgpr("GSUSumIdx+1"), comment="quotient*GSUSumIdx+remainder"))
    module.add(SMulI32(dst=sgpr(stmp+0), src0=sgpr(stmp+0), src1=sgpr("GSUSumIdx"), comment="(quotient+1)*GSUSumIdx"))
    # if gsuSumIdx < numIterPerWgRemainder
    module.add(SCmpLtU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), comment="gsuSumIdx < numIterPerWgRemainder"))
    module.add(SCSelectB32(dst=sgpr(stmp+0), src0=sgpr(stmp+0), src1=sgpr(stmp+1), comment="(quotient+1)*GSUSumIdx if needed"))

    return module

  ##############################################################################
  # Calculate Loop Num Iter
  # loopIdx is the index of the loop (used for contractions with multiple summations)
  # 0 is outermost; self.states.unrollIdx is the unroll index.
  # -1 is tail loop (used only for the unroll loop)
  ##############################################################################
  def calculateLoopNumIter(self, kernel, tPA, tPB, loopIdx):
    module = Module("calculateLoopNumIter")

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.states.unrollIdx
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    loopChar = self.states.indexChars[loopDim]

    ########################################
    # Tail Loop
    if tailLoop:
      with self.allocTmpSgpr(4) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        loopCounterName = self.loopCounterName(kernel, loopIdx)
        module.addSpaceLine()
        if kernel["SuppressNoLoadLoop"]:
          # If the tail loop is suppressed, then final iterations will have moved the Srd base forward
          # (and also moved back the srd shadow limit) and slammed Limit to 0, so need to 'undo'
          # those increments - see setTailSrd
          assert(kernel["PrefetchGlobalRead"] == 1) #if >1 would need a multiply here
          module.add(SCmpEQU32(src0=sgpr("OrigLoopCounter"), src1=0, comment="completely skipped unroll loop?"))
          module.add(SCSelectB32(dst=sgpr(tmpSgpr+0), src0=0, src1=sgpr("GlobalReadIncsA"), comment="force to 0?"))
          module.add(SCSelectB32(dst=sgpr(tmpSgpr+1), src0=0, src1=sgpr("GlobalReadIncsB"), comment="force to 0?"))
          module.add(self.setTailSrd(tPA, sgpr(tmpSgpr+0)))
          module.addSpaceLine()
          module.add(self.setTailSrd(tPB, sgpr(tmpSgpr+1)))
          module.addSpaceLine()
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            module.add(SCSelectB32(dst=sgpr(tmpSgpr+0), src0=0, src1=sgpr("GlobalReadIncsMetadata"), comment="force to 0?"))
            tP = tPB if kernel["ProblemType"]["Sparse"] == 2 else tPA
            module.add(self.setTailSrd(tP, sgpr(tmpSgpr+0)))
            module.addSpaceLine()
        # LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
        module.addComment("numIter%s = LOCAL_SPLITU * min(size%s %% LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)" \
            % (self.states.unrollChar, self.states.unrollChar))

        # size % DepthU
        module.add(scalarStaticDivideAndRemainder(tmpSgpr, loopCounterName, \
          "SizesSum+%u"%loopIdx, kernel["DepthU"], ContinuousRegister(tmpSgpr+2, 2), 2))
        loopCounter = sgpr(loopCounterName)

        if kernel["LocalSplitU"] > 1:
          # we cannot set loopCounter zero and skip tail loop because we need all waves to do global read.
          # in order to check the k index, we have to keep the offset to check boundary.
          #  | lsu0 | lsu1 | lsu2 | lsu3|
          #  |----d----|
          #  |  o0  |o1|    keep this offset of each lsu.
          # For example, 'o1' is the offset of lsu1. '-o0' is the offset of lsu0. This offset can be negative.

          dividend               = tmpSgpr+2
          tmpVgpr    = self.vgprPool.checkOutAligned(2, 2,"tmpVgpr")
          tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
          wave_id    = self.vgprPool.checkOut(1)
          numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]

          module.add(SMovB32(dst=sgpr(dividend), src=hex(kernel["DepthU"]//kernel["LocalSplitU"]), comment="DEPTHU / LOCAL_SPLITU" ))
          module.add(vectorStaticDivide(wave_id, "Serial", kernel["WavefrontSize"], tmpVgprRes))
          module.add(vectorStaticDivide(wave_id, wave_id, numWaves, tmpVgprRes, comment="LSU offset: Get LSU wave_id"))
          #module.add(VAddU32(vgpr(wave_id), vgpr(wave_id), 1, "add 1"))
          module.add(VMulLOU32(vgpr(wave_id), sgpr(dividend), vgpr(wave_id), comment="multiply by (DEPTHU / LOCAL_SPLITU)"))
          module.add(SNop(waitState=0, comment="Wait to read lane"))
          module.add(VReadfirstlaneB32(dst=sgpr("LSUTailLoopOffset"), src=vgpr(wave_id), comment="Update Alpha"))
          module.add(SSubI32(
              dst=sgpr("LSUTailLoopOffset"), \
              src0=loopCounter, \
              src1=sgpr("LSUTailLoopOffset"), \
              comment="lsu offset" ))
          module.add(SMinU32(dst=loopCounter, src0=sgpr(dividend), src1=loopCounter, comment="" ))
          self.vgprPool.checkIn(wave_id)
          self.vgprPool.checkIn(tmpVgpr)

      skComponent = Component.StreamK.find(self)
      module.add(skComponent.tailLoopNumIter(self, kernel, loopCounter))

      gsuComponent = Component.GSU.find(self)
      module.add(gsuComponent.tailLoopNumIter(self, kernel, loopCounter))

      # if tail numIter == 0 skip altogether
      skipTailLoopLabel = Label.getFormatting("SkipTailLoop%s"%(loopChar) )
      module.add(SCmpEQU32(src0=loopCounter, src1=0, comment="numIter%s == 0"%loopChar ))
      module.add(SMovB32(dst=sgpr("OrigLoopCounter"), src=0, comment="repurpose to count each localRead increment"))
      module.add(SCBranchSCC1(labelName=skipTailLoopLabel, \
                comment="skip to end of tail loop b/c numIter==0"))

    ########################################
    # Unrolled Loop
    elif loopIdx == self.states.unrollIdx:
      loopCounterName = self.loopCounterName(kernel, loopIdx)
      loopCounter = sgpr(loopCounterName)
      if not self.do["PreLoop"]: module.add(ValueEndif())

      with self.allocTmpSgpr(3) as tmpSgprInfo:
        skComponent = Component.StreamK.find(self)
        module.add(skComponent.calculateLoopNumIter(self, kernel, loopCounterName, loopIdx, tmpSgprInfo))

        gsuComponent = Component.GSU.find(self)
        module.add(gsuComponent.calculateLoopNumIter(self, kernel, loopCounterName, tmpSgprInfo))

        module.add(SMovB32(dst=sgpr("OrigLoopCounter"), \
                  src=loopCounter, \
                  comment="copy loop counter"))
    else:
      # other summation, not unroll loop
      #printExit("no assembly support for 2+ dimensional summation")
      module.addComment1("%sother summation, numIter%s = size%s" \
          % (self.indent, loopChar, loopChar))
      loopCounter = self.loopCounter(kernel, loopIdx)
      module.add(SMovB32(dst=loopCounter, \
                src=sgpr("SizesSum+%u"%loopIdx), \
                comment="init loop counter"))

    return module

  ##############################################################################
  # Calculate Metadata offset
  ##############################################################################
  def calculateIncrementMetadata(self, kernel, sgprOut):
    module = Module("calculateIncrementMetadata")
    gsuComponent = Component.GSU.find(self)
    module.add(gsuComponent.calculateIncrementMetadata(self, kernel, sgprOut))
    return module

  ##############################################################################
  # Open Loop
  ##############################################################################
  def openLoop(self, kernel, tPA, tPB, loopIdx, noLabelGen=False, beginLabelOnly=False):
    module = Module("openLoop")
    # TODO - rewrite this function to simplify control-flow between tail-loop / unroll loop
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.states.unrollIdx
      self.states.inTailLoop = True
    loopChar = self.states.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if not tailLoop and not noLabelGen:
      module.add(Label("openLoop%s"%loopChar, ""))
    loopLabelBegin = Label("%sLoopBegin%s"%("Tail" if tailLoop else "", loopChar), "", alignment=16 )
    loopLabelEnd = Label("%sLoopEnd%s"%("Tail" if tailLoop else "", loopChar), "" )

    if beginLabelOnly:
      # generate only beginLabel, then, return
      module.add(loopLabelBegin)
      return module

    # is numIter at least 1? otherwise skip to end
    # PGL needs a skip-check here if not bufferload
    # If kernel["SuppressNoLoadLoop"] we don't have a special loop for the 'last iter'
    loopCounter = self.loopCounter(kernel, loopIdx)
    if tailLoop:
      endCounter = 0
    elif kernel["PrefetchGlobalRead"] == 1:
      if kernel["SuppressNoLoadLoop"]:
        endCounter =  0
      else:
        endCounter = 1
    elif kernel["PrefetchGlobalRead"] == 2:
      if kernel["SuppressNoLoadLoop"]:
        endCounter =  1
      else:
        endCounter = 2
    else:
      endCounter =  0

    if tailLoop:
      # begin loop
      if not noLabelGen:
        module.add(loopLabelBegin)

    else: # not tailloop:

      if loopIdx == self.states.unrollIdx:
        # 1 loop check is necessary only when AssertSummationElementMultiple % (DepthU * 2) != 0
        if kernel["PrefetchGlobalRead"] == 2 and kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) != 0:
          module.add(SCmpEQU32(
              src0=loopCounter, \
              src1=hex(endCounter-1), \
              comment="LoopCounter%s < EndCounter"%(loopChar) ))
          toPGR1 = Label.getFormatting(self.labels.getName("toPGR1"))
          module.add(SCBranchSCC1(labelName=toPGR1, comment="PGR=2 but only 1 loop, toPGR1"))

        module.add(SCmpLeU32(
            src0=loopCounter, \
            src1=hex(endCounter), \
            comment="LoopCounter%s < EndCounter"%(loopChar) ))
        jumpLabel = loopLabelEnd
        if kernel["PrefetchGlobalRead"]==2 and (not kernel["SuppressNoLoadLoop"]) and kernel["ExpandPointerSwap"]:
          # PGR=2 and EPS and no SuppressNoLoadLoop case, need to jump to EvenExit
          jumpLabel = Label("LoopEnd%s_evenexit"%(loopChar), "" )
        module.add(SCBranchSCC1(labelName=jumpLabel.getLabelName(), \
                  comment="do not enter Loop%s"%loopChar ))

      if kernel["ExpertSchedulingMode"] > 0:
        expertSchedulingMode = int(kernel["ExpertSchedulingMode"])
        module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0,2]), src=expertSchedulingMode, comment="disable conservative hardware dependency checking to allow scheduling by software"))

      if not noLabelGen:
        module.add(loopLabelBegin)

      if loopIdx != self.states.unrollIdx:
        # reset LRO since these may have changed due to odd-iter exit ?
        if kernel["PrefetchGlobalRead"]:
          module.addComment0("openLoop - reset LRO for possible odd-iter exit")
          module.add(self.localReadResetOffsets(kernel, tPA))
          module.add(self.localReadResetOffsets(kernel, tPB))
          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]
            module.add(self.localReadResetOffsets(kernel, tPM))

    return module

  ##############################################################################
  # Close Loop
  # finalLoop : final unroll loop
  ##############################################################################
  def closeLoop(self, kernel, tPA, tPB, loopIdx, finalLoop, emitEndLabelOnly=False, oddLabel=False, skipCondJumpCounter=-1):
    module = Module("closeLoop")
    if emitEndLabelOnly:
      loopIdx = self.states.unrollIdx
      loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      module.add(Label("SkipTailLoop%s"%(loopChar), ""))
      return module

    tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]

    finalJump = SCBranchSCC0
    nonFinalJumpNeeded = True

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.states.unrollIdx
      loopChar = self.states.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = Label("TailLoopBegin%s"%(loopChar), "" )
      loopLabelEnd = Label("TailLoopEnd%s"%(loopChar), "" )
      loopLabelEndOddExit = Label("TailLoopEnd%s_oddexit"%(loopChar), "unroll loop odditer exit" )
      loopCounter = self.loopCounter(kernel, loopIdx)
      numReadsIterCoalescedA = self.states.numReadsIterCoalescedA
      numReadsIterCoalescedB = self.states.numReadsIterCoalescedB
      numReadsIterCoalesced = max(numReadsIterCoalescedA, numReadsIterCoalescedB)

      unrollInc      = 1
      KinInnerUnroll = kernel["InnerUnroll"]
      # dot2: force InnerUnroll = 1 in tail loop
      if kernel["UseDotInstruction"]:
        unrollInc *= kernel["NumDotElements"] * kernel["NumWaveSplitK"]
      else:
        if kernel["EnableMatrixInstruction"]:
          unrollInc      *= kernel["MatrixInstK"] * numReadsIterCoalesced
          KinInnerUnroll *= kernel["MatrixInstK"]
        if kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0:
          unrollInc *= kernel["InnerUnroll"]

      skipCondJump = False
      if skipCondJumpCounter >= 0 and (skipCondJumpCounter%numReadsIterCoalesced < numReadsIterCoalesced - 1):
        # skip conditional jump when numReadsIterCoalesced > 1 and skipCondJumpCounter is not the last in numReadsIterCoalesced
        # to support numReadsIterCoalesced > 1, MatrixInstK * numReadsIterCoalesced needs to be executed
        # e.g.) MatrixInstK=4, numReadsIterCoalesced=2
        #    skipCondJumpCounter==0 case: execute K=0,2,4,6  (here, no exit(means skip cond jump) to execute odd K(1,3,5,7))
        #    skipCondJumpCounter==1 case: execute K=1,3,5,7  (here, all K=0-7 are done. check condition and jump if tail loop is done)
        # skipCondJump=True is not to exit after skipCondJumpCounter==0.
        skipCondJump = True
      nonFinalJumpNeeded = not skipCondJump
      if not skipCondJump:
        module.addComment1("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

        module.add(SSubI32(
            dst=loopCounter, \
            src0=loopCounter, \
            src1=hex(unrollInc), \
            comment="dec counter%s (tailLoop)"%(loopChar) ))

        # Track # LDS reads?
        module.add(SAddU32(
          dst=sgpr("OrigLoopCounter"), \
          src0=sgpr("OrigLoopCounter"), \
          src1=hex(unrollInc),
          comment="inc counter%s"%(loopChar) ))

        endCounter = 0
        if kernel["LocalSplitU"] > 1:
          module.add(SSubI32(
              dst=sgpr("LSUTailLoopOffset"), \
              src0=sgpr("LSUTailLoopOffset"), \
              src1=hex(unrollInc), \
              comment="LSU offset dec counter%s (tailLoop)"%(loopChar) ))
          tmp = self.sgprPool.checkOut(1)
          module.add(SMinI32(dst=sgpr(tmp), src0=loopCounter, src1=sgpr("LSUTailLoopOffset"), comment="check lsu offset too"))
          module.add(SCmpLeI32(
              src0=sgpr(tmp), \
              src1=hex(endCounter), \
              comment="counter%s<=%d"%(loopChar,endCounter) ))
          self.sgprPool.checkIn(tmp)
        else:
          module.add(SCmpLeI32(
              src0=loopCounter, \
              src1=hex(endCounter), \
              comment="counter%s<=%d"%(loopChar,endCounter) ))
    else: # not tailloop
      loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = Label("LoopBegin%s"%(loopChar), "" )
      loopLabelEnd = Label("LoopEnd%s"%(loopChar), "" )
      loopLabelEndOddExit = Label("LoopEnd%s_oddexit"%(loopChar), "unroll loop odditer exit" )
      loopLabelEndEvenExit = Label("LoopEnd%s_evenexit"%(loopChar), "unroll loop eveniter exit" )
      loopCounter = self.loopCounter(kernel, loopIdx)
      module.addComment1("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

      # If PrefetchGlobalRead=1 the loads in the loop prefetch next macro-tile
      # For the final trip through the unroll loop we need to ensure those loads stay in bounds.

      # One technique is to create a copy of the unroll loop with all loads removed.
      # However buffer load doesn't need this loop copy since we OOB loads can be suppressed by buffer limit hardware
      # So can do one more iteration (endCounter==0) in the main unroll loop, and adjust the pointer
      # increments appropriately.
      # Also sum idx other than unroll always compare against 0 (there is no PGR to account for)
      if kernel["PrefetchGlobalRead"] == 1 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.states.unrollIdx:
        endCounter = 1
      elif kernel["PrefetchGlobalRead"] == 2 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.states.unrollIdx:
        endCounter = 2
      else:
        endCounter = 0

      if kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) == 0 and endCounter > 0:
        # if AssertSummationElementMultiple is multiple of DepthU*2, loop exit is necessary only once in 2 Loop iterations
        #  In endCounter % 2 == 1 case, exit at lc % 2 == 0 (= oddLabel). It means no exit if not oddLabel
        #  In endCounter % 2 == 0 case, exit at lc % 2 == 1 (= not oddLabel). It means no exit if oddLabel
        # No exit case, no code is necessary except for final Loop

        # decrement by 2 if PGR=2 and StaggerU is 0, else 1
        if kernel["PrefetchGlobalRead"]==2:
          with self.allocTmpSgpr(2) as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            module.add(SCmpEQU32(src0=sgpr("StaggerU"), src1=0))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=hex(2), src1=hex(1)))
            decCode = SSubU32(dst=loopCounter, src0=loopCounter, \
                src1=sgpr(tmpSgpr), \
                comment="dec counter%s"%(loopChar) )
        else:
          decCode = SSubU32(dst=loopCounter, src0=loopCounter, \
              src1=1, \
              comment="dec counter%s"%(loopChar) )
        condCode = SCmpEQI32(src0=loopCounter, \
            src1=hex(endCounter), \
            comment="counter%s==%d"%(loopChar,endCounter) )

        noExit = False

        if endCounter%2 != 0:
          if not oddLabel:
            noExit = True
        else:
          if oddLabel:
            noExit = True

        if noExit:
          # No exit. No dec code if decValue is 2
          if decValue == 2:
            decCode = ""
          condCode = ""
          nonFinalJumpNeeded = False
          if finalLoop:
            # No exit and finalLoop case, use s_branch (no condition)
            finalJump = SBranch

        if decCode: module.add(decCode)
        if condCode: module.add(condCode)
      else:
        module.add(SSubU32(
            dst=loopCounter, src0=loopCounter, \
            src1=1, \
            comment="dec counter%s"%(loopChar) ))

        module.add(SCmpEQI32(
            src0=loopCounter, \
            src1=hex(endCounter), \
            comment="counter%s==%d"%(loopChar,endCounter) ))

    jumpLabel = loopLabelEnd
    if not tailLoop and not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
      # in this case, odd or/and even code is generated and use odd/even exit to avoid skipping odd/even code
      # (end label is generated after odd/even code)
      jumpLabel = loopLabelEndOddExit if oddLabel else loopLabelEndEvenExit
    if not finalLoop:
      if nonFinalJumpNeeded:
        # just an exit check, else fall through to the next loop copy
        module.add(SCBranchSCC1(labelName=jumpLabel.getLabelName(), comment="exit Loop%s"%loopChar ))
    else: #finalLoop:
      module.add(finalJump(labelName=loopLabelBegin.getLabelName(), comment="restart Loop%s"%(loopChar)))

      if not tailLoop and loopIdx == self.states.unrollIdx:
        oddIterPreCode = Module()
        oddIterCode = Module()
        evenIterPreCode = Module()
        evenIterCode = Module()
        if not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
          oddIterPreCode.add(loopLabelEndOddExit)
          # In this case we kept the 'no-load' loop which has LDS offsets assuming first bank of LDS
          # if we exit the main loop at an odd iter - need to swap LDS read pointers
          # so the ds_reads read from the 'high' buffer of LDS
          oddIterPreCode.addComment1("Select high bank of LDS")
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprA"] and not kernel["StoreSwapAddr"]:
            if kernel["ExpertSchedulingMode"] > 0:
              oddIterCode.add(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))
            oddIterCode.add(self.localReadSwapOffsets(kernel, False, tPA))
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprB"] and not kernel["StoreSwapAddr"]:
            if kernel["ExpertSchedulingMode"] > 0:
              oddIterCode.add(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))
            oddIterCode.add(self.localReadSwapOffsets(kernel, False, tPB))

          if kernel["ProblemType"]["Sparse"]:
            if kernel["DirectToVgprSparseMetadata"]:
              oddIterCode.add(SWaitCnt(vlcnt=0, comment="wait for global read before moving metadata to target vgpr"))
              for i in range(0, self.states.m.numVgprValuPerBlock):
                oddIterCode.add(VMovB32(vgpr("ValuMetadata+%u"%i), vgpr("ValuMetadata+%u+%u"%(self.states.m.numVgprValuPerBlock, i)), \
                                        comment="copy ValuMetadata blk1 to blk0"))
            else:
              oddIterCode.add(self.localReadSwapOffsets(kernel, False, tPM))

          evenIterPreCode.add(loopLabelEndEvenExit)
          # generate even code here (so far, for PrefetchGlobalRead=2 only)
          if kernel["PrefetchGlobalRead"]==2:
            if not kernel["DirectToVgprA"] or not kernel["DirectToVgprB"]:
              if kernel["ExpertSchedulingMode"] > 0:
                evenIterCode.add(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))

            # Generate local write address code only for PrefetchGlobalRead==2
            if not kernel["DirectToLdsA"]:
              evenIterCode.add(self.localWriteSwapOffsets(kernel, False, tPA))
            if not kernel["DirectToLdsB"]:
              evenIterCode.add(self.localWriteSwapOffsets(kernel, False, tPB))
            evenIterCode.add(self.localWriteSwapOffsets(kernel, True, tPA))
            evenIterCode.add(self.localWriteSwapOffsets(kernel, True, tPB))

        # generate even, odd exit code
        # not oddLabel case, order is even -> odd
        firstPreCode = evenIterPreCode
        firstCode = evenIterCode
        secondPreCode = oddIterPreCode
        secondCode = oddIterCode
        if oddLabel:
          # oddLabel case, swap the order (odd -> even)
          firstPreCode, secondPreCode = secondPreCode, firstPreCode
          firstCode, secondCode = secondCode, firstCode

        module.add(firstPreCode)
        module.add(firstCode)

        # if secondCode exist, add jump to skip secondCode
        if secondCode.count():
          module.add(SBranch(labelName=loopLabelEnd.getLabelName(), \
                    comment="exit unroll loop%s (and skip second exit code)"%(loopChar)))
        module.add(secondPreCode)
        module.add(secondCode)

      module.add(loopLabelEnd)

      if tailLoop:
        if len(kernel["ProblemType"]["IndicesSummation"]) > 1 or kernel["StreamK"]:
          # recover the 'damage' done to LRO:

          # if LRA is backed-up before (wlr case), we simply restore the addr (sub inc*loop doesn't work)
          tPList = []
          if self.oriLraA != None:
            module.add(VMovB32(dst=vgpr("LocalReadAddrA"), src=vgpr(self.oriLraA), comment="restore LRA"))
            self.vgprPool.checkIn(self.oriLraA)
            self.oriLraA = None
          else:
            tPList.append(tPA)
          if self.oriLraB != None:
            module.add(VMovB32(dst=vgpr("LocalReadAddrB"), src=vgpr(self.oriLraB), comment="restore LRA"))
            self.vgprPool.checkIn(self.oriLraB)
            self.oriLraB = None
          else:
            tPList.append(tPB)

          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            if self.oriLraM != None:
              module.add(VMovB32(dst=vgpr("LocalReadAddrMetadata"), src=vgpr(self.oriLraM), comment="restore LRA"))
              self.vgprPool.checkIn(self.oriLraM)
              self.oriLraM= None
            else:
              tPList.append(tPM)

          for tP in tPList:
            tc     = tP["tensorChar"]
            LdsPad = kernel["LdsPad%s" % tc] * tP["bpeDS"] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
            inc    = kernel["LocalSplitU"] * (kernel["MacroTile%s"%tc] + LdsPad) * tP["bpeDS"]

            # aligned with localReadInc
            if kernel["EnableMatrixInstruction"]:
              if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
                inc = kernel["LocalSplitU"] * tP["bpeDS"]
              # No need to *= K, because LoopCounter is increased by K each time
              # inc *= kernel["MatrixInstK"]

            if not (tP["isA"] and kernel["DirectToVgprA"] or tP["isB"] and kernel["DirectToVgprB"]): # no local read code if DirectToVgpr is enabled
              with self.allocTmpSgpr(1) as tmpSgprInfo:
                stmp = tmpSgprInfo.idx
                module.add(SMovB32(dst=sgpr(stmp), src=inc, comment="tailloop lds offset"))
                module.add(SMulI32(dst=sgpr(stmp), src0=sgpr("OrigLoopCounter"), src1=sgpr(stmp), comment="scale by mul"))
                module.add(VSubU32(dst=vgpr("LocalReadAddr%s"%tc), src0=vgpr("LocalReadAddr%s"%tc), src1=sgpr(stmp), comment="remove lro damage"))
          # if LWA is backed-up before, we simply restore the addr
          if self.oriLwaA != None:
            module.add(VMovB32(dst=vgpr("LocalWriteAddrA"), src=vgpr(self.oriLwaA), comment="restore LWA"))
            module.add(VMovB32(dst=vgpr("LocalWriteAddrB"), src=vgpr(self.oriLwaB), comment="restore LWA"))
            self.vgprPool.checkIn(self.oriLwaA)
            self.vgprPool.checkIn(self.oriLwaB)
            self.oriLwaA = None
            self.oriLwaB = None

            if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
              module.add(VMovB32(dst=vgpr("LocalWriteAddrMetadata"), src=vgpr(self.oriLwaM), comment="restore LWA"))
              self.vgprPool.checkIn(self.oriLwaM)
              self.oriLwaM = None
    # restore all threads
    if tailLoop and kernel["LocalSplitU"] > 1:
      sgprCnt = self.states.laneSGPRCount
      waveSize = kernel["WavefrontSize"]
      module.addComment1("restore full exec mask")

      with self.allocTmpSgpr(sgprCnt) as tmpSgprInfo:
        fullExec = tmpSgprInfo.idx
        activeMask    = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
        SMovBX        = SMovB32 if (waveSize == 32) else SMovB64
        SOrSaveExecBX = SOrSaveExecB32 if (waveSize == 32) else SOrSaveExecB64
        module.add(SMovBX(dst=sgpr(fullExec,sgprCnt), src=activeMask, comment="restore all threads active"))
        module.add(SOrSaveExecBX (dst=sgpr(fullExec,sgprCnt), src=sgpr(fullExec,sgprCnt), comment="full mask -> exec" ))
    return module

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self, kernel, tPA, tPB, noSkipLoad = True, label = None, isOptNLL = False):
    module = Module("endSummation")

    module.add(Label((self.labels.getUniqueNamePrefix("Summation_End") if label is None else label), ""))

    if kernel["StorePriorityOpt"]:
      module.add(SSetPrior(prior=0, comment="optimization store"))

    if isOptNLL:
      vbegin = self.states.startVgprMisc
      vsize = self.states.lastVgprForReads - vbegin
      if self.states.a.startVgprLocalReadAddr > self.states.lastVgprForReads:
        vsize = self.states.a.startVgprLocalReadAddr + self.states.a.numVgprLocalReadAddr - vbegin
      if self.states.b.startVgprLocalReadAddr > self.states.lastVgprForReads:
        vsize = self.states.b.startVgprLocalReadAddr + self.states.b.numVgprLocalReadAddr - vbegin
      self.vgprPool.add(vbegin, vsize, "endSummation")
      module.addComment0("endSummation: add vgpr [%u...%u) to pool" % \
                        (vbegin, vbegin+vsize))

    # Write bias A, B data to LDS
    if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):

      tP = tPA if kernel["ProblemType"]["BiasSrc"] == "A" else tPB
      module.add(self.exclasses.biasSumUnroll.storeSumLDS(self, kernel, tP))

      vbegin = self.states.bias.startVgprValu
      vsize = self.states.bias.numVgprValu
      self.vgprPool.add(vbegin, vsize, "free vgpr of biasSumUnroll")
      module.addComment0("endSummation: add vgpr [%u...%u) to pool" % \
                        (vbegin, vbegin+vsize))

    lastRegTag=None

    spool = self.sgprPool.getPool()
    for i in range(0, self.sgprPool.size()):
      regTag = spool[i].tag
      if regTag != lastRegTag:
        lastRegTag = regTag
        if (lastRegTag not in self.states.nonPostLoopSgpr) and (spool[i].status == RegisterPool.Status.InUse):
          if label == "Summation_End_OptNLL":
            self.undefineSgpr(regTag)
          else:
            module.add(self.undefineSgpr(regTag))

    if self.db["InitVgpr"] & 0x2:
      module.add(self.vgprPool.initTmps(self.consts.initVgprValue,start=0, stop=100))
    if 0: # FIXME: Can remove?
      for i in range(0,16+1):
         module.add(VMovB32(dst=vgpr(21), src=vgpr(21), comment="hack tmp in pool"))

    # this doesn't seem to do anything
    if self.db["InitSgpr"] & 0x2:
      module.add(self.sgprPool.initTmps(self.consts.initSgprValue))

    if self.db["ConservativeWaitCnt"] & 0x10:
      module.add(SBarrier(comment="debug"))
      module.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=0, comment=""))

    if kernel["SuppressNoLoadLoop"]:
      module.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=0, comment="wait for all summation activity"))

    # Load bias data from LDS and write to global

    ########################################
    # Load kernel args needed by global write batch
    module.addComment0("load store sgprs")
    # Define sgprs for kernel args
    runActivation = True if ((kernel["ProblemType"]["ActivationType"] != 'none') \
        and kernel["ActivationFused"]) else False

    def fixPreloadOffset(offset, sgpxIdxVec, numStoreSgprToLoad):
      item = None
      startVgprName = sgpxIdxVec[0]
      if kernel["ProblemType"]["UseScaleAB"] == "Scalar":
        if (kernel["ProblemType"]["DataTypeA"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters()) and (kernel["ProblemType"]["DataTypeB"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters()):
          self.argLoader.setOffset(offset + ((self.states.rpga * self.states.bpr) * 2))
        elif kernel["ProblemType"]["DataTypeA"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters():
          assert sgpxIdxVec[0] == self.sgprs["AddressScaleB"]
          self.argLoader.setOffset(offset + (self.states.rpga * self.states.bpr))
        elif kernel["ProblemType"]["DataTypeB"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters():
          assert sgpxIdxVec[0] == self.sgprs["AddressScaleA"]
          item = self.argLoader.loadKernArg(self.sgprs["AddressScaleA"], "KernArgAddress", dword=2)
          if len(sgpxIdxVec) > 1:
            startVgprName = sgpxIdxVec[1]
          numStoreSgprToLoad -= self.states.rpga
          self.argLoader.setOffset(offset + ((self.states.rpga * self.states.bpr) * 2))
      return (item, startVgprName, numStoreSgprToLoad)

    if self.states.numStoreSgprToLoad:
      sgpxIdxVec = self.defineMultiSgprIndex(self.states.numStoreSgprNames, self.states.numStoreSgprNameSizes, align=4)
      for name in self.states.numStoreSgprNames:
          module.add(RegSet("s", "sgpr"+name, self.sgprs[name]))
      if noSkipLoad and kernel["GlobalSplitU"] != 0:
        gsuLabel = Label(label=self.labels.getNameInc("GSU"), comment="")
        with self.allocTmpSgpr(1) as tmpSgprGSU:
          module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
          module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
        if (kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel'):
          module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="branch if GSU != 1"))
      if kernel["ProblemType"]["SupportUserArgs"]:
        extReadEpilogueLabel    = Label(label=self.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
        extReadEpilogueLabelEnd = Label(label=self.labels.getNameInc("LoadExternalEpilogueStructEnd"), comment="")
        module.addComment0("Check if custom structure pointer is null")
        module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
        module.add(SCBranchSCC1(labelName=extReadEpilogueLabel.getLabelName(), comment="branch if ArgType == 2"))
        argOffset = self.argLoader.getOffset() # Backup offset
        numStoreSgprToLoad = self.states.numStoreSgprToLoad
        (item, startVgprName, numStoreSgprToLoad) = fixPreloadOffset(argOffset, sgpxIdxVec, numStoreSgprToLoad)
        if item:
          module.add(item)
        loadModule = module.addModuleAsFlatItems(self.argLoader.loadAllKernArg(startVgprName, "KernArgAddress", numStoreSgprToLoad))
        self.states.numStoreSgprInst = countSMemLoad(loadModule)
        self.argLoader.setOffset(argOffset) # Restore offset
        module.add(SBranch(extReadEpilogueLabelEnd.getLabelName()))
        module.add(extReadEpilogueLabel)
        extArgOffset = self.externalArgLoader.getOffset()
        backupExtArgOffset = extArgOffset
        loadList = [[-1, 0, extArgOffset]]
        extArgOffset += self.states.userArgsInfo.scaleASize
        if (kernel["ProblemType"]["UseScaleAB"] == "Scalar" and (not self.states.preloadScaleA)) or kernel["ProblemType"]["UseScaleAB"] == "Vector":
          if loadList[-1][0] == -1:
            loadList[-1][0] = self.sgprs["AddressScaleA"]
          loadList[-1][1] += self.states.userArgsInfo.scaleASize
        else:
          loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        extArgOffset += self.states.userArgsInfo.scaleBSize
        if (kernel["ProblemType"]["UseScaleAB"] == "Scalar" and (not self.states.preloadScaleB)) or kernel["ProblemType"]["UseScaleAB"] == "Vector":
          if loadList[-1][0] == -1:
            loadList[-1][0] = self.sgprs["AddressScaleB"]
          loadList[-1][1] += self.states.userArgsInfo.scaleBSize
        else:
          loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        extArgOffset += self.states.userArgsInfo.scaleCSize + self.states.userArgsInfo.scaleDSize
        if kernel["ProblemType"]["UseScaleCD"]:
          if loadList[-1][0] == -1:
            loadList[-1][0] = self.sgprs["AddressScaleC"]
          loadList[-1][1] += self.states.userArgsInfo.scaleCSize + self.states.userArgsInfo.scaleDSize
        else:
          loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        extArgOffset += self.states.userArgsInfo.scaleAlphaVecSize
        if kernel["ProblemType"]["UseScaleAlphaVec"]:
          if loadList[-1][0] == -1:
            loadList[-1][0] = self.sgprs["AddressScaleAlphaVec"]
          loadList[-1][1] += self.states.userArgsInfo.scaleAlphaVecSize
        else:
          loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        extArgOffset += self.states.userArgsInfo.biasSize
        if self.states.numSgprAddressBias:
          biasLoadSize = (self.states.numSgprAddressBias + self.states.BiasType + self.states.BiasStride) * 4
          if loadList[-1][0] == -1:
            loadList[-1][0] = self.sgprs["AddressBias"]
          loadList[-1][1] += biasLoadSize
          if biasLoadSize < self.states.userArgsInfo.biasSize:
            loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        else:
            loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.

        extArgOffset += self.states.userArgsInfo.factorDimSize
        if self.states.FactorDim == 3:
          if loadList[-1][0] == -1:
            loadList[-1][0] = self.sgprs["FactorDim"]
          loadList[-1][1] += self.states.userArgsInfo.factorDimSize
        else:
          loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.

        extArgOffset += self.states.userArgsInfo.eSize
        if kernel["ProblemType"]["UseE"]:
          if loadList[-1][0] == -1:
            loadList[-1][0] = self.sgprs["AddressE"]
          loadList[-1][1] += self.states.userArgsInfo.eSize
        else:
          loadList.append([-1, 0, extArgOffset])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        extArgOffset += self.states.userArgsInfo.activationSize
        if runActivation:
          needActTypeArg = 1 if self.states.numActivationTypeArgSize else 0
          actNames = kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList()
          actLoadSize = (len(actNames) * self.states.numActivationArgSize + needActTypeArg) * 4
          if (actLoadSize == self.states.userArgsInfo.activationSize) or len(actNames) > 0:
            if loadList[-1][0] == -1:
              loadList[-1][0] = self.sgprs[actNames[0]]
            loadList[-1][1] += actLoadSize
          else:
            loadList.append(["ActivationType", actLoadSize])  # Need to start a new loadAllKernArg cause no AdditionalArgStringList is needed
            loadList.append([-1, 0, extArgOffset - (needActTypeArg * 4)])  # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        else:
          loadList.append([-1, 0, extArgOffset])   # Need to start a new loadAllKernArg cause the argument is not consecutively anymore.
        # Start reading arguments
        loadModuleExt = Module("Count Inst")
        for loadInfo in loadList:
          if loadInfo[0] == -1:
            continue
          dwordLen = loadInfo[1] // 4
          self.externalArgLoader.setOffset(loadInfo[2])
          loadModuleExt.addModuleAsFlatItems(module.addModuleAsFlatItems(self.externalArgLoader.loadAllKernArg(loadInfo[0], "KernArgAddress", dwordLen)))
        self.states.numStoreSgprInstExt = countSMemLoad(loadModuleExt)
        self.externalArgLoader.setOffset(backupExtArgOffset)
        module.add(extReadEpilogueLabelEnd)
      else:
        argOffset = self.argLoader.getOffset() # Backup offset
        numStoreSgprToLoad = self.states.numStoreSgprToLoad
        (item, startVgprName, numStoreSgprToLoad) = fixPreloadOffset(argOffset, sgpxIdxVec, numStoreSgprToLoad)
        if item:
          module.add(item)
        loadModule = module.addModuleAsFlatItems(self.argLoader.loadAllKernArg(startVgprName, "KernArgAddress", numStoreSgprToLoad))
        self.states.numStoreSgprInst = countSMemLoad(loadModule)
        self.argLoader.setOffset(argOffset) # Restore offset
      if noSkipLoad and kernel["GlobalSplitU"] != 0:
        module.add(gsuLabel)

    ########################################
    # Load kernel args needed by global write batch

    # Define sgprs for kernel args
    if self.states.numStoreSgprToLoad2:
      module.addComment0("load store sgprs2")
      for name in self.states.numStoreSgprNames2:
          module.add(RegSet("s", "sgpr"+name, self.sgprs[name]))

      argOffset = self.argLoader.getOffset() # Backup offset
      if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and (kernel["ProblemType"]["DataTypeA"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters() or kernel["ProblemType"]["DataTypeB"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters()):
        self.argLoader.setOffset(argOffset + (self.states.numStoreSgprToLoad)*4 + (self.states.rpga * self.states.bpr)) # Restore offset
      else:
        self.argLoader.setOffset(argOffset + (self.states.numStoreSgprToLoad)*4) # Restore offset
      numStoreSgprToLoad = self.states.numStoreSgprToLoad2

      if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and (kernel["ProblemType"]["DataTypeA"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters() or kernel["ProblemType"]["DataTypeB"].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters()):
        argOffsettmp = (argOffset + (self.states.numStoreSgprToLoad)*4 + (self.states.rpga * self.states.bpr)) # Restore offset
      else:
        argOffsettmp = (argOffset + (self.states.numStoreSgprToLoad)*4) # Restore offset

      extReadEpilogueLabeltmp    = Label(label=self.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
      module.addComment0("Check if custom structure pointer is null")
      if kernel["ProblemType"]["SupportUserArgs"]:
        module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
        module.add(SCBranchSCC1(labelName=extReadEpilogueLabeltmp.getLabelName()))

      module.add(self.argLoader.loadKernArg("AddressTD", "KernArgAddress", sgprOffset=hex(argOffsettmp), dword=2))
      module.add(self.argLoader.loadKernArg("Synchronizer", "KernArgAddress", sgprOffset=hex(argOffsettmp + (2)*4), dword=2))

      module.add(extReadEpilogueLabeltmp)

      self.argLoader.setOffset(argOffset) # Restore offset

    # define the rest sgprs
    if (not self.states.doShadowInit) and kernel["BufferStore"]:
      self.defineSgpr("SrdD", 4, 4)
      self.defineSgpr("SrdC", 4, 4)
      module.add(RegSet("s", "sgprSrdC", self.sgprs["SrdC"]))
      module.add(RegSet("s", "sgprSrdD", self.sgprs["SrdD"]))
    if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
      self.defineSgpr("SrdScaleA", 4, 4)
      self.defineSgpr("SrdScaleB", 4, 4)
      module.add(RegSet("s", "sgprSrdScaleA", self.sgprs["SrdScaleA"]))
      module.add(RegSet("s", "sgprSrdScaleB", self.sgprs["SrdScaleB"]))
    if kernel["ProblemType"]["UseScaleAlphaVec"]:
      self.defineSgpr("SrdScaleAlphaVec", 4, 4)
      module.add(RegSet("s", "sgprSrdScaleAlphaVec", self.sgprs["SrdScaleAlphaVec"]))
    if self.states.useBias != DataDirection.NONE:
      self.defineSgpr("SrdBias", 4, 4)
      module.add(RegSet("s", "sgprSrdBias", self.sgprs["SrdBias"]))

    if (kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel'):
      self.defineSgpr("SrdTD", 4, 4)
      module.add(RegSet("s", "sgprSrdTD", self.sgprs["SrdTD"]))
      self.defineSgpr("GSUSync", 1)
      module.add(RegSet("s", "sgprGSUSync", self.sgprs["GSUSync"]))

    if kernel["ProblemType"]["UseE"]:
      self.defineSgpr("SrdE", 4, 4)
      module.add(RegSet("s", "sgprSrdE", self.sgprs["SrdE"]))
      for idx in range(0, kernel["ProblemType"]["NumIndicesC"]):
        i = idx
        idxChar= self.states.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesCD"]:
          module.add(ValueSet("constStrideE%s"%idxChar, 1))
        else:
          if not kernel["ProblemType"]["UseInitialStridesCD"]:
            i = i-1
          module.add(RegSet("s", "sgprStrideE%s"%idxChar, \
                    "sgprStridesE", i))
    if kernel["ProblemType"]["UseScaleCD"]:
      assert kernel["ProblemType"]["ComputeDataType"].isSingle()
      self.defineSgpr("ScaleD", 2, 2)
      module.add(RegSet("s", "sgprScaleD", self.sgprs["ScaleD"]))

    if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      self.defineSgpr("SrdSync", 4, 4)
      module.add(RegSet("s", "sgprSrdSync", self.sgprs["SrdSync"]))

    # Load kernel args end
    ########################################

    # copy accumulated C from agpr to vgpr
    if kernel["EnableMatrixInstruction"]:
      #TODO avoid s_nop if its possible
      #instCycles = kernel["MatrixInstM"] // 2 # 32x32 is 64 cycles, 16x16 is 32 cycles, 4x4 is 8 cycles
      #module.add(SNop(waitState=instCycles))
      module.addComment1("Mapping of Acc register -> C Vgpr register")
      self.codes.accVgprRead = mapAcctoArchRegs(kernel, self.states.maxLimitAgprs, write=False)
      if kernel["StreamK"] > 0 and kernel["StreamKAtomic"] == 0:
        self.codes.accVgprWrite = mapAcctoArchRegs(kernel, self.states.maxLimitAgprs, write=True)
      if kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
        self.codes.accVgprWrite = mapAcctoArchRegs(kernel, self.states.maxLimitAgprs, write=True)
      if kernel["MIArchVgpr"]:
        module.addComment1("Multiply MI out register with Alpha -> C Vgpr register")
        self.codes.mulAlphaMultipleBuffer = moveMIoutToArch(kernel, self.states.startVgprAlphaTmp)
        self.codes.mulAlphaOther = mulMIoutAlphaToArch(kernel, self.states.startVgprAlphaTmp)

    return module

  def mfmaIter_waitCount(self, kernel):
    if self.states.version in [(9,4,2), (9,5,0)]:
      dataType = kernel["ProblemType"]["DataType"]
      miM = kernel["MatrixInstM"]
      miN = kernel["MatrixInstN"]
      if dataType.isSingle() or dataType.isHalf() or dataType.isBFloat16():
          if miM == 4 and miN == 4:
              return 2
    return 0

  ##############################################################################
  # src A,B str for MFMA
  ##############################################################################
  def generateSrcStrForMFMA(self, kernel, tP, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, idxAB, bk=None):
    tc = tP["tensorChar"]

    statesAorB = self.states.a if tP["isA"] else self.states.b
    numVgprValuPerBlock = kernel["MIWaveTile%c"%tc] * kernel["MIInputPerThread%c"%tc] * tP["bpe"] // self.states.bpr
    numIterPerCoalescedRead = self.states.numIterPerCoalescedReadA if tP["isA"] else self.states.numIterPerCoalescedReadB
    numReadsIterCoalesced   = self.states.numReadsIterCoalescedA   if tP["isA"] else self.states.numReadsIterCoalescedB

    # calculate vgprBufferA_new ( or B) and offset for DirectToVgpr. Use u instead of m (number of local prefetch buffer does not matter)
    m_or_u = u if kernel["DirectToVgpr%c"%tc] else m
    vgprBuffer_new = (m_or_u//numIterPerCoalescedRead)*numIterPerCoalescedRead
    vgprBuffer_new_offset = m_or_u%numIterPerCoalescedRead*innerUnroll*vgprPerInput
    # DirectToVgpr + pack special case
    # offset vgprBuffer_new
    packDTV = self.states.packDTVA if tP["isA"] else self.states.packDTVB
    convDTV = self.states.convDTVA if tP["isA"] else self.states.convDTVB
    if packDTV or convDTV:
      # DTV + pack case, offset bufferIdx for local read packing instructions
      numBi = kernel["LoopIters"]
      vgprBuffer_new += vregSetIdx * numBi

    iui_new = (iui//numReadsIterCoalesced)*numReadsIterCoalesced
    iui_new_offset = iui%numReadsIterCoalesced*vgprPerInput
    ab_new = idxAB*vgprPerInput*numReadsIterCoalesced
    abStr = "Valu%c_X%u_I%u+%u+%u+%u" % (tc, vgprBuffer_new, iui_new, ab_new, vgprBuffer_new_offset, iui_new_offset)
    if kernel["UseDirect32XEmulation"] and bk != None and (int(bk) % 8) < 4:
      abStr = "Valu%c_T%u_I%u+%u+%u+%u" % (tc, vgprBuffer_new, iui_new, ab_new // 2, vgprBuffer_new_offset, iui_new_offset)
    if kernel["DirectToVgpr%c"%tc] and not (packDTV or convDTV):
      # overwrite aStr/bStr for DirectToVgpr (except for pack DTV case)
      numVgprPerBlock = statesAorB.numVgprG2LAllocated
      numVgprPerBlock //= 2 # DTV case, buffer is doubled. //2 to calculate single size
      ab_new += vregSetIdx * numVgprPerBlock + ( vgprBuffer_new * innerUnroll) * numVgprValuPerBlock
      abStr  = "G2L%c+%u+%u" % (tc, ab_new, vgprBuffer_new_offset)

    if bk != None:
      abStr += "+%u"%(bk)
    return abStr

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, tPA, tPB, bufferIdx, iuiCount, useMacro, isTail=False):
    imod = Module("macIter_X%u_I%u"%(bufferIdx, iuiCount))

    # if kernel["ProblemType"]["DataType"].isHalf():
    #   # imod.addText(".align32 8, 0xbf800001", "align v_pk_fma")   # Align v_pk_fma instructions used in MAC_ blocks
    #   imod.addText(".align32 8, 0xbf800001\n")   # Align v_pk_fma instructions used in MAC_ blocks

    # dot2: add shiftK module to prevent read out of K bound
    shiftK           = Module("shiftK")
    inputType        = kernel["ProblemType"]["DataType"]
    numRegistersIn   = inputType.numRegisters()
    loopCounterName  = self.loopCounterName(kernel, self.states.unrollIdx)
    numInput         = kernel["NumDotElements"] if kernel["UseDotInstruction"] else 1
    kPerIter         = numInput * kernel["NumWaveSplitK"]
    vgprPerInput     = 1
    shiftPerElement  = int(numRegistersIn * 32)
    s_nop            = 0
    # alloc vgpr
    kReg    = None
    abReg   = None
    dummy   = -1

    if isTail and (kernel["AssertSummationElementMultiple"] % kPerIter != 0):
      kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
      loopCntSgpr = loopCounterName

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        shiftK.add(vectorStaticRemainder(dummy, kReg, "Serial", kernel["NumWaveSplitK"], None, tmpSgprInfo))

      numTmpSgpr = 4 if (vgprPerInput > 2) else 3

      with self.allocTmpSgpr(numTmpSgpr, alignment=1) as tmpSgprInfo:
        if tmpSgprInfo.idx % 2 == 0:
          tmpSgprX2 = tmpSgprInfo.idx
          tmpSgprX1 = tmpSgprInfo.idx+2
        else:
          tmpSgprX2 = tmpSgprInfo.idx+1
          tmpSgprX1 = tmpSgprInfo.idx

        # replace 0 for differnet thread
        shiftK.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), numInput, tmpSgprInfo))
        shiftK.add(VCmpGEI32(dst=sgpr(tmpSgprX2, self.states.laneSGPRCount), src0=vgpr(kReg), src1=sgpr(loopCntSgpr), comment="check K index >= Size L"))
        for a in range(0, kernel["ThreadTileA"]):
          for iui in range(0, iuiCount):
            aStr = vgpr("ValuA_X%u_I%u+%u" % (bufferIdx, iui, a*vgprPerInput), 1)
            shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=0, src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment="set 0 if K_idx >= sizeL"))

        for b in range(0, kernel["ThreadTileB"]):
          for iui in range(0, iuiCount):
            bStr = vgpr("ValuB_X%u_I%u+%u" % (bufferIdx, iui, b*vgprPerInput), 1)
            shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=0, src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment="set 0 if K_idx >= sizeL"))

        # replace 0 for some thread
        if kernel["AssertSummationElementMultiple"] % numInput != 0:
          abReg   = self.vgprPool.checkOutAligned(vgprPerInput, vgprPerInput, "abReg")
          shiftK.add(VSubU32(dst=vgpr(kReg), src0=sgpr(loopCntSgpr), src1=vgpr(kReg), comment="get distance between size and k index"))
          shiftK.add(VCmpLtI32(dst=sgpr(tmpSgprX2, self.states.laneSGPRCount), src0=vgpr(kReg), src1=numInput, comment="set partial 0 if distance less than numInput"))
          shiftK.add(SAndB32(dst=sgpr(tmpSgprX1), src0=sgpr(loopCntSgpr), src1=numInput-1, comment="get inputs for edge thread"))
          shiftK.add(SSubU32(dst=sgpr(tmpSgprX1), src0=numInput, src1=sgpr(tmpSgprX1), comment="use shift to fill 0 for outside element"))
          shiftK.add(SLShiftLeftB32(dst=sgpr(tmpSgprX1), shiftHex=log2(shiftPerElement), src=sgpr(tmpSgprX1), comment="use shift to fill 0 for outside element"))

          # currently vgprPerInput must be 1
          VShiftLeft = VLShiftLeftB32

          for a in range(0, kernel["ThreadTileA"]):
            for iui in range(0, iuiCount):
              aStr = vgpr("ValuA_X%u_I%u+%u" % (bufferIdx, iui, a*vgprPerInput), vgprPerInput)
              shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=aStr, comment=""))
              for bk in range(0, vgprPerInput):
                aStr  = vgpr("ValuA_X%u_I%u+%u+%u" % (bufferIdx, iui, a*vgprPerInput, bk), 1)
                shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment=""))

          for b in range(0, kernel["ThreadTileB"]):
            for iui in range(0, iuiCount):
              bStr = vgpr("ValuB_X%u_I%u+%u" % (bufferIdx, iui, b*vgprPerInput), vgprPerInput)
              shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=bStr, comment=""))
              for bk in range(0, vgprPerInput):
                bStr  = vgpr("ValuB_X%u_I%u+%u+%u" % (bufferIdx, iui, b*vgprPerInput, bk), 1)
                shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment=""))

      if kernel["LocalSplitU"] > 1:
        self.sgprPool.checkIn(loopCntSgpr)

      s_nop = 2

    if s_nop != 0:
      imod.add(SNop(waitState=(s_nop - 1), comment=""))

    if kernel["InnerUnroll"] > 1 and iuiCount==1:
      # This it tail-loop case where we just want one IUI,
      instr = "MAC_%ux%u_X%u_OneIUI" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx)
    else:
      if not useMacro:
        printExit("MAC doesn't support useMacro=False")
      instr = "MAC_%ux%u_X%u" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx)

    # release register
    if kReg is not None: self.vgprPool.checkIn(kReg)
    if abReg is not None: self.vgprPool.checkIn(abReg)

    if self.do["MAC"]:
      imod.add(shiftK)
      imod.add(MacroInstruction(name=instr, args=[]))
      imod.addSpaceLine()
    return imod

  ##############################################################################
  # ACC Vgpr R/W Function
  ##############################################################################
  def accVgprReadWriteFunction(self, kernel, idx, read=True):
    if not kernel["MIArchVgpr"]:
      if idx >= self.states.maxLimitAgprs:
        return VMovB32
      else:
        return VAccvgprReadB32 if read else VAccvgprWriteB32
    else:
      return VMovB32
  def accVgprReadWriteIndex(self, kernel, idx, sz=1):
    if not kernel["MIArchVgpr"]:
      if idx >= self.states.maxLimitAgprs:
        return vgpr(idx - self.states.maxLimitAgprs, sz)
      else:
        return accvgpr(idx, sz)
    else:
      return vgpr(idx, sz)

  ##############################################################################
  # MFMA Iteration
  ##############################################################################
  def mfmaIter(self, kernel, tPA, tPB, u, innerUnroll, vregSetIdx, unrollLoopIdx = 0, unrollIdx = 0, tail = False, firstIter = False, postShiftK = Module()):
    imod = Module("mi")
    shiftK = Module("shiftK")
    m = (u) % (self.states.numVgprBuffer) # local to use for MACs

    def dataTypeToMfmaInstTypePair(dataType: DataType, sourceSwap: bool) -> Tuple[InstType, InstType]:
      miInTypeStr  = dataType.toNameAbbrev()
      miInInstType = dataTypeNameAbbrevToInstType(miInTypeStr, sourceSwap) # v_mfma_[...xK]<InType>
      miOutInstType = dataTypeNameAbbrevToInstType(dataType.MIOutputTypeNameAbbrev()) # v_mfma_<OutType>..
      return miInInstType, miOutInstType

    def dataTypeNameAbbrevToInstType(abbrev: str, sourceSwap: bool = False) -> InstType:
      if abbrev == 'f64':
          return InstType.INST_F64
      elif abbrev == 'f32':
          return InstType.INST_F32
      elif abbrev == 'f16':
          return InstType.INST_F16
      elif abbrev == 'i32':
          return InstType.INST_I32
      elif abbrev == 'i8':
          return InstType.INST_I8
      elif abbrev == 'bf16':
          return InstType.INST_BF16
      elif abbrev == 'xf32':
          return InstType.INST_XF32
      elif abbrev == 'fp8_fp8':
          return InstType.INST_F8
      elif abbrev == 'bf8_bf8':
          return InstType.INST_BF8
      elif (abbrev == 'fp8_bf8' and sourceSwap == False) or \
          (abbrev == 'bf8_fp8' and sourceSwap == True):
          return InstType.INST_F8_BF8
      elif (abbrev == 'bf8_fp8' and sourceSwap == False) or \
          (abbrev == 'fp8_bf8' and sourceSwap == True):
          return InstType.INST_BF8_F8
      else:
          assert("Unsupported data type.")
      return InstType.INST_NOTYPE

    miInputType      = kernel["ProblemType"]["F32XdlMathOp"] if kernel["EnableF32XdlMathOp"] else kernel["ProblemType"]["DataType"]
    # calculate constant
    is_mfma          = self.states.asmCaps["HasMFMA"]
    is_wmma_v1          = self.states.asmCaps["HasWMMA_V1"]
    is_wmma_v2          = self.states.asmCaps["HasWMMA_V2"]
    numRegistersIn   = miInputType.numRegisters()
    numRegistersOut  = kernel["MIRegPerOut"]
    loopCounterName  = self.loopCounterName(kernel, self.states.unrollIdx)
    accs_per_wave    = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] \
                       // self.states.kernel["WavefrontSize"] * numRegistersOut
    dividerFortidInK = kernel["MatrixInstN"] * kernel["MatrixInstB"]
    numMIInputA      = kernel["MIInputPerThreadA"]
    numMIInputB      = kernel["MIInputPerThreadB"]
    numMIInput       = max(numMIInputA,numMIInputB)
    miInInstType, miOutInstType = dataTypeToMfmaInstTypePair(miInputType, kernel["SourceSwap"])
    neg_flag         = True if ((not is_mfma) and (miInInstType == InstType.INST_I8)) else False
    miInInstType     = InstType.INST_U8 if ((not is_mfma) and miInInstType == InstType.INST_I8) else miInInstType
    miOutInstType    = miOutInstType if is_mfma else dataTypeNameAbbrevToInstType(kernel["ProblemType"]["ComputeDataType"].toNameAbbrev())
    numReadsIterCoalescedA = self.states.numReadsIterCoalescedA
    numReadsIterCoalescedB = self.states.numReadsIterCoalescedB
    numReadsIterCoalesced = max(numReadsIterCoalescedA, numReadsIterCoalescedB)

    vgprPerInputA    = int(numMIInputA * numRegistersIn)
    vgprPerInputB    = int(numMIInputB * numRegistersIn)
    vgprPerInput     = max(vgprPerInputA,vgprPerInputB)
    shiftPerElement  = int(numRegistersIn * 32)
    s_nop            = 0
    accumRegType     = "acc" if not kernel["MIArchVgpr"] else "v"
    mfma_1k          = True if kernel["MFMA_BF16_1K"] else False
    accStoreCIdx     = 0
    # alloc vgpr
    kReg_first = None
    kReg    = None
    abReg   = None
    tmpVgpr = None
    dummy   = -1

    if (numRegistersIn < 1) and ((kernel["UnrollMajorLDSA"] == False) or (kernel["UnrollMajorLDSB"] == False)):
      s_nop = 2

    if kernel["ConvertAfterDS"] and (numRegistersIn < 1) and ((tPA["bpe"] > tPA["bpeDS"]) or (tPB["bpe"] > tPB["bpeDS"])):
      s_nop = 2

    # here we remap index to where it read for wider local read
    # ex. if we read 2 iteration at a time,
    #   original   : _ds_load_b64  valuA_X0_I0
    #   read 2 iter: _ds_load_b128 valuA_X0_I0 (we read valuA_X0_I0 and valuA_X1_I0)
    # instead of using valuA_X1_I0, we use valuA_X0_I0+2 as mfma input

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      vgprPerInputM    = 1
      vgprBufferM_new = (m//self.states.numIterPerCoalescedReadMetadata)*self.states.numIterPerCoalescedReadMetadata
      vgprBufferM_new_offset = m%self.states.numIterPerCoalescedReadMetadata*kernel["InnerUnroll"]*vgprPerInputM

    # handle multiple K element in MFMA instruction
    # MIK=1 case, we still need this code for Coalesced case
    if tail and (kernel["MatrixInstK"] > 1 or numReadsIterCoalescedA > 1 or numReadsIterCoalescedB > 1):
      if not is_wmma_v1: #mfma or wmma_v2
        # ToDo: Avoid using extra kReg_first for wmma_v2 case
        kReg_first  = self.vgprPool.checkOut(1,"kReg_first") # the first vgpr of remainder
        kReg        = self.vgprPool.checkOut(1,"kReg") # remainder
        if kernel["LocalSplitU"] > 1:
          loopCntSgpr = self.sgprPool.checkOut(1)
        else:
          loopCntSgpr = loopCounterName

        with self.allocTmpSgpr(1) as tmpSgprInfo:
          shiftK.add(vectorStaticRemainder(dummy, kReg_first, "Serial", self.states.kernel["WavefrontSize"], tmpVgpr, tmpSgprInfo))
          shiftK.add(vectorStaticDivide(kReg_first, kReg_first, dividerFortidInK, tmpVgpr))
        numTmpSgpr = 4 if (vgprPerInput > 2 and is_wmma_v2) else 3 # ToDo: check if this is correct for sparse mfma

        with self.allocTmpSgpr(numTmpSgpr, alignment=1) as tmpSgprInfo:
          if tmpSgprInfo.idx % 2 == 0:
            tmpSgprX2 = tmpSgprInfo.idx
            tmpSgprX1 = tmpSgprInfo.idx+2
            if numTmpSgpr == 4:
               tmpSgprX3 = tmpSgprInfo.idx+3
          else:
            tmpSgprX2 = tmpSgprInfo.idx+1
            tmpSgprX1 = tmpSgprInfo.idx
            if numTmpSgpr == 4:
               tmpSgprX3 = tmpSgprInfo.idx+3

          def findSparseOffset(isA:bool):
            blocksPerTGroupSMFMA = 1
            elementsPerBlockSMFMA = 1
            blockOffsetSMFMA =1
            if kernel["ProblemType"]["Sparse"] != 0:
                if kernel["MIInputPerThread"] * kernel["ProblemType"]["DataTypeB"].numBytes() > 16: # double K
                    isSparseTrack = (kernel["ProblemType"]["Sparse"] == 1 and isA) or (kernel["ProblemType"]["Sparse"] == 2 and not isA)
                    # gfx950 sparse track only has one block for each thread group.
                    # TODO adjust this value for other arch.
                    blocksPerTGroupSMFMA = 1 if isSparseTrack else 2
                    if blocksPerTGroupSMFMA > 1:
                        threadGroups = kernel["MatrixInstK"] // kernel["MIInputPerThread"]
                        elementsPerBlockSMFMA = kernel["MIInputPerThread"] // blocksPerTGroupSMFMA
                        blockStride = elementsPerBlockSMFMA * threadGroups
                        blockOffsetSMFMA = blockStride - elementsPerBlockSMFMA
            return blocksPerTGroupSMFMA, elementsPerBlockSMFMA, blockOffsetSMFMA

          blocksPerTGroupSMFMAA, elementsPerBlockSMFMAA, blockOffsetSMFMAA = findSparseOffset(True)
          blocksPerTGroupSMFMAB, elementsPerBlockSMFMAB, blockOffsetSMFMAB = findSparseOffset(False)
          # replace 0 for differnet thread
          if kernel["ProblemType"]["Sparse"] == 1 and numMIInput//8 >= 1:
            vgprPerSet0Group = 1
          elif vgprPerInputA <= 2:
            shiftK.add(vectorStaticMultiply(vgpr(kReg_first), vgpr(kReg_first), numMIInput * numReadsIterCoalesced, tmpSgprInfo))
            kStepForCoalesced = (u%numReadsIterCoalesced) * numMIInput
            if kStepForCoalesced > 0:
              shiftK.add(VAddU32(vgpr(kReg_first), hex(kStepForCoalesced), vgpr(kReg_first), "k += (u%%numReadsIterCoalesced) * numMIInput"))
            if kernel["LocalSplitU"] > 1:
              shiftK.add(SMinI32(dst=sgpr(loopCntSgpr), src0=sgpr(loopCounterName), src1=sgpr("LSUTailLoopOffset"), comment="check lsu bound"))
            shiftK.add(VCmpGEI32(dst=sgpr(tmpSgprX2, self.states.laneSGPRCount), src0=vgpr(kReg_first), src1=sgpr(loopCntSgpr), comment="check K index >= Size L")) #ToDo: check value of laneSGPRCount
            vgprPerSet0Group = vgprPerInputA
          elif is_wmma_v2 and vgprPerInputA > 2:
            vgprPerSet0Group = 4
          else:
            vgprPerSet0Group = 2
          numSet0GroupA = vgprPerInputA//vgprPerSet0Group
          for group in range(0, numSet0GroupA):
            if numSet0GroupA > 1 or (is_wmma_v2 and vgprPerInputA > 2):
              if group == 0:
                if kernel["ProblemType"]["Sparse"]:
                    multiplyBy = numMIInput//blocksPerTGroupSMFMAA
                else:
                    multiplyBy = numMIInput//2 if vgprPerInputA == 8 else numMIInput
                shiftK.add(vectorStaticMultiply(vgpr(kReg_first), vgpr(kReg_first), multiplyBy, tmpSgprInfo))
                shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg_first), 0, ""))
              elif blocksPerTGroupSMFMAA == 2 and (group * vgprPerSet0Group) == (elementsPerBlockSMFMAA * numRegistersIn):
                kIncA = blockOffsetSMFMAA + (numMIInput//numSet0GroupA) * max(group - 1, 0)
                shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg), kIncA, "add part of K"))
              else:
                kIncA = numMIInput//numSet0GroupA
                if self.states.asmCaps["HasMFMA_f8f6f4"]:
                  if group == 2 and vgprPerInputA == 8: #special layout for F8
                    kIncA = 56 if kernel["MatrixInstK"] == 128 else 24
                    if kernel["UseF32XEmulation"]:
                      kIncA = 14 if kernel["MatrixInstK"] == 32 else 6
                shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg), kIncA, "add part of K"))
              if kernel["LocalSplitU"] > 1:
                shiftK.add(SMinI32(dst=sgpr(loopCntSgpr), src0=sgpr(loopCounterName), src1=sgpr("LSUTailLoopOffset"), comment="check lsu bound"))
              shiftK.add(VCmpGEI32(dst=sgpr(tmpSgprX2, self.states.laneSGPRCount), src0=vgpr(kReg), src1=sgpr(loopCntSgpr), comment="check K index >= Size L"))

            if not tPA["isSwizzled"]:
              for bk in range(0, vgprPerSet0Group):
                for a in range(0, kernel["MIWaveTileA"]):
                  for iui in range(0, innerUnroll):
                    aStr = vgpr(self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInputA, m, u, iui, a, bk=bk + group * vgprPerSet0Group), 1)
                    shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=0, src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment="set 0 if K_idx >= sizeL"))

          if kernel["ProblemType"]["Sparse"] == 2 and numMIInput//8 >= 1:
            shiftK.add(vectorStaticRemainder(dummy, kReg_first, "Serial", kernel["WavefrontSize"], tmpVgpr, tmpSgprInfo))
            shiftK.add(vectorStaticDivide(kReg_first, kReg_first, dividerFortidInK, tmpVgpr))
            vgprPerSet0Group = 1
          elif vgprPerInputB <= 2:
            vgprPerSet0Group = vgprPerInputB
          elif is_wmma_v2 and vgprPerInputB > 2:
            shiftK.add(vectorStaticRemainder(dummy, kReg_first, "Serial", kernel["WavefrontSize"], tmpVgpr, tmpSgprInfo))
            shiftK.add(vectorStaticDivide(kReg_first, kReg_first, dividerFortidInK, tmpVgpr))
            vgprPerSet0Group = 4
          else:
            shiftK.add(vectorStaticRemainder(dummy, kReg_first, "Serial", kernel["WavefrontSize"], tmpVgpr, tmpSgprInfo))
            shiftK.add(vectorStaticDivide(kReg_first, kReg_first, dividerFortidInK, tmpVgpr))
            vgprPerSet0Group = 2

          numSet0GroupB = vgprPerInputB//vgprPerSet0Group
          for group in range(0, numSet0GroupB):
            if numSet0GroupB > 1 or (is_wmma_v2 and vgprPerInputB > 2):
              if group == 0:
                if kernel["ProblemType"]["Sparse"]:
                    multiplyBy = numMIInput//blocksPerTGroupSMFMAB
                else:
                    multiplyBy = numMIInput//2 if vgprPerInputB == 8 else numMIInput
                shiftK.add(vectorStaticMultiply(vgpr(kReg_first), vgpr(kReg_first), multiplyBy, tmpSgprInfo))
                shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg_first), 0, ""))
              elif blocksPerTGroupSMFMAB == 2 and (group * vgprPerSet0Group) == (elementsPerBlockSMFMAB * numRegistersIn):
                kIncB = blockOffsetSMFMAB + (numMIInput//numSet0GroupB) * max(group - 1, 0)
                shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg), kIncB, "add part of K"))
              else:
                kIncB = numMIInput//numSet0GroupB
                if group == 2 and vgprPerInputB == 8:
                   kIncB = 56 if kernel["MatrixInstK"] == 128 else 24
                   if kernel["UseF32XEmulation"]:
                     kIncB = 14 if kernel["MatrixInstK"] == 32 else 6
                shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg), kIncB, "add part of K"))
              # replace 0 for differnet thread
              if kernel["LocalSplitU"] > 1:
                shiftK.add(SMinI32(dst=sgpr(loopCntSgpr), src0=sgpr(loopCounterName), src1=sgpr("LSUTailLoopOffset"), comment="check lsu bound"))
              shiftK.add(VCmpGEI32(dst=sgpr(tmpSgprX2, self.states.laneSGPRCount), src0=vgpr(kReg), src1=sgpr(loopCntSgpr), comment="check K index >= Size L"))
            for bk in range(0, vgprPerSet0Group):
              for b in range(0, kernel["MIWaveTileB"]):
                for iui in range(0, innerUnroll):
                  bStr = vgpr(self.generateSrcStrForMFMA(kernel, tPB, innerUnroll, vregSetIdx, vgprPerInputB, m, u, iui, b, bk=bk + group*vgprPerSet0Group), 1)
                  shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=0, src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment="set 0 if K_idx >= sizeL"))

          # replace 0 for same thread
          if numMIInput > 1 and kernel["AssertSummationElementMultiple"] < 8:
            alignment = vgprPerInput if is_wmma_v2 else (2 if vgprPerInput > 1 else 1)
            abReg   = self.vgprPool.checkOutAligned(vgprPerInput, alignment, "abReg")
            if (vgprPerInput < 4 and is_mfma) or is_wmma_v2:
              shiftK.add(VSubU32(dst=vgpr(kReg), src0=sgpr(loopCntSgpr), src1=vgpr(kReg_first), comment="get distance between size and k index"))
              shiftK.add(VCmpLtI32(dst=sgpr(tmpSgprX2, self.states.laneSGPRCount), src0=vgpr(kReg), src1=numMIInput, comment="set partial 0 if distance less than input per thread"))
            TailLoop_SkipZeroOutMask = Label((self.labels.getUniqueNamePrefix("TailLoop_SkipZeroOutMask")), comment="")
            shiftK.add(SAndB32(dst=sgpr(tmpSgprX1), src0=sgpr("SizeL"), src1=8-1, comment="if summation is multiple of 8, skip masking"))
            shiftK.add(SCmpEQU32(src0=sgpr(tmpSgprX1), src1=0))
            shiftK.add(SCBranchSCC1(labelName=TailLoop_SkipZeroOutMask.getLabelName(), comment="skip mask"))
            shiftK.add(SAndB32(dst=sgpr(tmpSgprX1), src0=sgpr(loopCntSgpr), src1=numMIInput-1, comment="get inputs for edge thread"))
            shiftK.add(SSubU32(dst=sgpr(tmpSgprX1), src0=numMIInput, src1=sgpr(tmpSgprX1), comment="use shift to fill 0 for outside element"))
            shiftK.add(SLShiftLeftB32(dst=sgpr(tmpSgprX1), shiftHex=log2(shiftPerElement), src=sgpr(tmpSgprX1), comment="use shift to fill 0 for outside element"))

            if vgprPerInput == 1:
              VShiftLeft = VLShiftLeftB32
            elif vgprPerInput >= 2:   # for 2, 4, 8
              VShiftLeft = VLShiftLeftB64
            if vgprPerInput == 4 and is_wmma_v2:
              tmpVgpr2   = self.vgprPool.checkOutAligned(2, 2, "tmpVgpr2")

            for a in range(0, kernel["MIWaveTileA"]):
              for iui in range(0, innerUnroll):
                aStr_base = self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a)
                aStr = vgpr(aStr_base, min(2, vgprPerInput))
                if is_wmma_v2:
                  if vgprPerInput == 4:
                    a_64_shift = Label(label=self.labels.getNameInc("a_64_Shift"), comment="")
                    a_32_shift = Label(label=self.labels.getNameInc("a_32_Shift"), comment="")
                    a_common = Label(label=self.labels.getNameInc("a_shift_end"), comment="")
                    aStr1 = vgpr(aStr_base + "+1", min(2, vgprPerInput))
                    aStr2 = vgpr(aStr_base + "+2", min(2, vgprPerInput))

                    shiftK.add(SMovB32(dst=sgpr(tmpSgprX3), src=sgpr(tmpSgprX1), comment="sgpr used for minic shift 128 bit"))
                    shiftK.add(SCmpGeI32(src0=sgpr(tmpSgprX3), src1=64, comment="check offset > 63"))
                    shiftK.add(SCBranchSCC1(labelName=a_64_shift.getLabelName(), comment="jump when positive"))
                    shiftK.add(SCmpGeI32(src0=sgpr(tmpSgprX3), src1=32, comment="check offset > 32"))
                    shiftK.add(SCBranchSCC1(labelName=a_32_shift.getLabelName(), comment="jump when positive"))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=aStr, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg), src=vgpr(tmpVgpr2),comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+1), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=aStr1, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+2), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=aStr2, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+3), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(SBranch(a_common.getLabelName()))
                    shiftK.add(a_32_shift)
                    shiftK.add(SSubU32(dst=sgpr(tmpSgprX3), src0=sgpr(tmpSgprX3), src1=32, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg), src=0, comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=aStr, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+1), src=vgpr(tmpVgpr2),comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+2), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=aStr1, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+3), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(SAddU32(dst=sgpr(tmpSgprX3), src0=sgpr(tmpSgprX3), src1=32, comment=""))
                    shiftK.add(SBranch(a_common.getLabelName()))
                    shiftK.add(a_64_shift)
                    shiftK.add(SSubU32(dst=sgpr(tmpSgprX3), src0=sgpr(tmpSgprX3), src1=64, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg), src=0, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+1), src=0, comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(abReg+2, 2), shiftHex=sgpr(tmpSgprX3), src=aStr, comment=""))
                    shiftK.add(SAddU32(dst=sgpr(tmpSgprX3), src0=sgpr(tmpSgprX3), src1=64, comment=""))
                    shiftK.add(a_common)
                  else:
                    shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=aStr, comment=""))
                  for bk in range(0, vgprPerInput):
                    aStr = vgpr(self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a, bk=bk), 1)
                    shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment=""))
                else: # mfma
                  if kernel["ProblemType"]["Sparse"]:
                    if vgprPerInput == 2:
                      aStr = vgpr(self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a), 2)
                      shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=aStr, comment=""))
                  elif vgprPerInput <= 2:
                    shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=aStr, comment=""))
                  if vgprPerInput >= 4 and vgprPerInput % 2 == 0:
                    for ivgpr in range(0, vgprPerInput, 2):
                      aStr = vgpr(self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a, bk=ivgpr), 2)
                      shiftK.add(VShiftLeft(dst=vgpr(abReg+ivgpr, 2), shiftHex=sgpr(tmpSgprX1), src=aStr, comment=""))
                  for bk in range(0, vgprPerInput):
                    aStr = vgpr(self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a, bk=bk), 1)
                    if vgprPerInput >= 4:
                      if bk == 0:
                        shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg_first), numMIInput//numSet0GroupA, "add part of K"))
                      elif bk % 2 == 0:
                        kIncA = numMIInput//numSet0GroupA
                        if bk == 4 and self.states.asmCaps["HasMFMA_f8f6f4"]:
                          kIncA = 56 if kernel["MatrixInstK"] == 128 else 24
                          if kernel["UseF32XEmulation"]:
                            kIncA = 14 if kernel["MatrixInstK"] == 32 else 6
                        shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg), kIncA, "add part of K"))
                      # replace 0 for differnet thread
                      if kernel["LocalSplitU"] > 1:
                        shiftK.add(SMinI32(dst=sgpr(loopCntSgpr), src0=sgpr(loopCounterName), src1=sgpr("LSUTailLoopOffset"), comment="check lsu bound"))
                      shiftK.add(VCmpGEI32(dst=sgpr(tmpSgprX2,2), src0=vgpr(kReg), src1=sgpr(loopCntSgpr), comment="check K index >= Size L"))
                    shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgprX2,2), comment=""))

            for b in range(0, kernel["MIWaveTileB"]):
              for iui in range(0, innerUnroll):
                bStr_base = self.generateSrcStrForMFMA(kernel, tPB, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, b)
                bStr = vgpr(bStr_base, min(2, vgprPerInput))
                if is_wmma_v2:
                  if vgprPerInput == 4:
                    b_64_shift = Label(label=self.labels.getNameInc("b_64_Shift"), comment="")
                    b_32_shift = Label(label=self.labels.getNameInc("b_32_Shift"), comment="")
                    b_common = Label(label=self.labels.getNameInc("b_shift_end"), comment="")
                    bStr1 = vgpr(bStr_base + "+1", min(2, vgprPerInput))
                    bStr2 = vgpr(bStr_base + "+2", min(2, vgprPerInput))
                    shiftK.add(SMovB32(dst=sgpr(tmpSgprX3), src=sgpr(tmpSgprX1), comment="sgpr used for minic shift 128 bit"))
                    shiftK.add(SCmpGeI32(src0=sgpr(tmpSgprX3), src1=64, comment="check offset >63"))
                    shiftK.add(SCBranchSCC1(labelName=b_64_shift.getLabelName(), comment="jump when positive"))
                    shiftK.add(SCmpGeI32(src0=sgpr(tmpSgprX3), src1=32, comment="check offset >32"))
                    shiftK.add(SCBranchSCC1(labelName=b_32_shift.getLabelName(), comment="jump when positive"))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=bStr, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg), src=vgpr(tmpVgpr2),comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+1), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=bStr1, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+2), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=bStr2, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+3), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(SBranch(b_common.getLabelName()))
                    shiftK.add(b_32_shift)
                    shiftK.add(SSubU32(dst=sgpr(tmpSgprX3), src0=sgpr(tmpSgprX3), src1=32, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg), src=0, comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=bStr, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+1), src=vgpr(tmpVgpr2),comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+2), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(tmpVgpr2, 2), shiftHex=sgpr(tmpSgprX3), src=bStr1, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+3), src=vgpr(tmpVgpr2+1),comment=""))
                    shiftK.add(SBranch(b_common.getLabelName()))
                    shiftK.add(b_64_shift)
                    shiftK.add(SSubU32(dst=sgpr(tmpSgprX3), src0=sgpr(tmpSgprX3), src1=64, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg), src=0, comment=""))
                    shiftK.add(VMovB32(dst=vgpr(abReg+1), src=0, comment=""))
                    shiftK.add(VShiftLeft(dst=vgpr(abReg+2,2), shiftHex=sgpr(tmpSgprX3), src=bStr, comment=""))
                    shiftK.add(b_common)
                  else:
                    shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=bStr, comment=""))
                  for bk in range(0, vgprPerInput):
                    bStr = vgpr(self.generateSrcStrForMFMA(kernel, tPB, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, b, bk=bk), 1)
                    shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgprX2, self.states.laneSGPRCount), comment=""))
                else: # mfma
                  if kernel["ProblemType"]["Sparse"]:
                    if vgprPerInput == 2:
                      aStr = vgpr(self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a), 2)
                      shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=aStr, comment=""))
                  elif vgprPerInput <= 2:
                    shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgprX1), src=bStr, comment=""))
                  if vgprPerInput >= 4 and vgprPerInput % 2 == 0:
                    for ivgpr in range(0, vgprPerInput, 2):
                      bStr = vgpr(self.generateSrcStrForMFMA(kernel, tPB, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, b, bk=ivgpr), 2)
                      shiftK.add(VShiftLeft(dst=vgpr(abReg+ivgpr, 2), shiftHex=sgpr(tmpSgprX1), src=bStr, comment=""))
                  for bk in range(0, vgprPerInput):
                    bStr = vgpr(self.generateSrcStrForMFMA(kernel, tPB, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, b, bk=bk), 1)
                    if vgprPerInput >= 4:
                      if bk == 0:
                        shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg_first), numMIInput//numSet0GroupB, "add part of K"))
                      elif bk % 2 == 0:
                        kIncB = numMIInput//numSet0GroupB
                        if bk == 4: # when vgprPerInput == 8
                          kIncB = 56 if kernel["MatrixInstK"] == 128 else 24
                          if kernel["UseF32XEmulation"]:
                            kIncB = 14 if kernel["MatrixInstK"] == 32 else 6
                        shiftK.add(VAddU32(vgpr(kReg), vgpr(kReg), kIncB, "add part of K"))
                      # replace 0 for differnet thread
                      if kernel["LocalSplitU"] > 1:
                        shiftK.add(SMinI32(dst=sgpr(loopCntSgpr), src0=sgpr(loopCounterName), src1=sgpr("LSUTailLoopOffset"), comment="check lsu bound"))
                      shiftK.add(VCmpGEI32(dst=sgpr(tmpSgprX2,2), src0=vgpr(kReg), src1=sgpr(loopCntSgpr), comment="check K index >= Size L"))
                    shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgprX2,2), comment=""))
            shiftK.add(TailLoop_SkipZeroOutMask)
            if vgprPerInput == 4 and is_wmma_v2:
              if tmpVgpr2 is not None: self.vgprPool.checkIn(tmpVgpr2)
        if kernel["LocalSplitU"] > 1:
          self.sgprPool.checkIn(loopCntSgpr)
      else: #wmma
        iui = 0
        abReg      = self.vgprPool.checkOutAligned(2, 2, "abReg")
        with self.allocTmpSgpr(3) as tmpSgprInfo:
          sgprShift = tmpSgprInfo.idx
          sgpr64bIdx = sgprShift + 1
          sgprMask   = sgprShift + 2

          shiftK.add(SSubI32(dst=sgpr(sgprShift), src0=sgpr(loopCounterName), src1=1, comment="calculate 64bit groups index"))
          shiftK.add(SLShiftRightB32(dst=sgpr(sgpr64bIdx), src=sgpr(sgprShift), shiftHex=log2(64 // (numRegistersIn * 32)), comment="calculate 64bit groups index"))
          shiftK.add(SAndB32(dst=sgpr(sgprShift), src0=sgpr(sgprShift), src1=int((64 // (numRegistersIn * 32))-1), comment="calculate shift value"))
          shiftK.add(SSubI32(dst=sgpr(sgprShift), src0=int((64 // (numRegistersIn * 32))-1), src1=sgpr(sgprShift), comment="calculate shift value"))
          shiftK.add(SLShiftLeftB32(dst=sgpr(sgprShift), shiftHex=log2(numRegistersIn * 32), src=sgpr(sgprShift),  comment="calculate shift value"))

          for it in range(int((kernel["MatrixInstK"] * numRegistersIn) // 2)): # handle 64 bit per iteration
            shiftK.add(VCmpEQI32(dst=sgpr(sgprMask), src0=sgpr(sgpr64bIdx), src1=it, comment='handle this 64bit group: part 1'))
            for a in range(0, kernel["MIWaveTileA"]):
              aStr = vgpr("ValuA_X%u_I%u+%u+%u" % (m, iui, a*vgprPerInput, it*2), 2)
              shiftK.add(VLShiftLeftB64(dst=vgpr(abReg,2), shiftHex=sgpr(sgprShift), src=aStr, comment=f"shfit for ValuA[{it*2}:{it*2+1}]"))
              for bk in range(2):
                aStr = vgpr("ValuA_X%u_I%u+%u+%u+%u" % (m, iui, a*vgprPerInput, it*2, bk))
                shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=vgpr(abReg+bk), src2=sgpr(sgprMask), comment="shift if in this 64b group"))
            for b in range(0, kernel["MIWaveTileB"]):
              bStr = vgpr("ValuB_X%u_I%u+%u+%u" % (m, iui, b*vgprPerInput, it*2), 2)
              shiftK.add(VLShiftLeftB64(dst=vgpr(abReg,2), shiftHex=sgpr(sgprShift), src=bStr, comment=f"shfit for ValuB[{it*2}:{it*2+1}]"))
              for bk in range(2):
                bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u" % (m, iui, b*vgprPerInput, it*2, bk))
                shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=vgpr(abReg+bk), src2=sgpr(sgprMask), comment="shift if in this 64b group"))
            if it > 0:
              shiftK.add(VCmpLtI32(dst=sgpr(sgprMask), src0=sgpr(sgpr64bIdx), src1=it, comment='handle this 64bit group: part 2'))
              for a in range(0, kernel["MIWaveTileA"]):
                for bk in range(2):
                  aStr = vgpr("ValuA_X%u_I%u+%u+%u+%u" % (m, iui, a*vgprPerInput, it*2, bk))
                  shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=0, src2=sgpr(sgprMask), comment="shift if in this 64b group"))
              for b in range(0, kernel["MIWaveTileB"]):
                for bk in range(2):
                  bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u" % (m, iui, b*vgprPerInput, it*2, bk))
                  shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=0, src2=sgpr(sgprMask), comment="shift if in this 64b group"))

      s_nop = 2

    if s_nop != 0:
      imod.add(SNop(waitState=(s_nop - 1), comment=""))

    prevAccIdx = -1
    for iui in range(0, innerUnroll):
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        iuiM_new = (iui//self.states.numReadsIterCoalescedMetadata)*self.states.numReadsIterCoalescedMetadata
        iuiM_new_offset = iui%self.states.numReadsIterCoalescedMetadata*vgprPerInputM

      zgemmVaddSrcCheck = [[], [], []] # to avoid generating redundant v_add
      outer = 1
      loopSwap = False
      # complex case, swap inner loop and outer loop so that idxA comes outer
      # this is to re-use same tmp vgpr to nagate ai or ar
      if kernel["ProblemType"]["DataType"].isComplex() and tPB["tile01Idx"]:
        outer = 0
        loopSwap = True
      inner = 1 - outer # inner is the opposite of outer

      idxOuter_start = 0
      idxInner_start = 0
      idxOuter_stop = kernel["MIWaveTile"][outer]
      idxInner_stop = kernel["MIWaveTile"][inner]
      numSubTiles = kernel["numSubTiles"]
      if numSubTiles > 1 and not self.states.inTailLoop:
        # iter  (idxOuter_start, idxOuter_stop)    (idxInner_start, idxInner_stop)        MFMA
        #  0          (0,4)                             (0,4)                          MFMA(A0,B0) 
        #  1          (0,4)                             (4,8)                          MFMA(A1,B0) 
        #  2          (4,8)                             (0,4)                          MFMA(A0,B1)
        #  3          (4,8)                             (4,8)                          MFMA(A1,B1)
        outerBy2=(kernel["MIWaveTile"][outer]//numSubTiles)
        innerBy2=(kernel["MIWaveTile"][inner]//numSubTiles)
        outerMod2=(kernel["MIWaveTile"][outer]%numSubTiles)
        innerMod2=(kernel["MIWaveTile"][inner]%numSubTiles)
        idxHalfO = u//numSubTiles
        idxHalfI = u % numSubTiles
        idxOuter_start = (outerBy2 + outerMod2)*idxHalfO 
        idxInner_start = (innerBy2 + innerMod2)*idxHalfI 
        idxOuter_stop = kernel["MIWaveTile"][outer] - (1-idxHalfO)* outerBy2
        idxInner_stop = kernel["MIWaveTile"][inner] - (1-idxHalfI)* innerBy2

      for idxOuter in range(idxOuter_start, idxOuter_stop):
        for idxInner in range(idxInner_start, idxInner_stop):
          idx0 = idxInner
          idx1 = idxOuter
          if loopSwap:
            idx0, idx1 = idx1, idx0
          accIdx   = idx1 * kernel["MIWaveTile"][0] + idx0
          accStart = accIdx * accs_per_wave
          accEnd   = accStart + accs_per_wave - 1

          idxA     = idx0 if tPB["tile01Idx"] else idx1
          idxB     = idx1 if tPB["tile01Idx"] else idx0
          aStr_base = self.generateSrcStrForMFMA(kernel, tPA, innerUnroll, vregSetIdx, vgprPerInputA, m, u, iui, idxA)
          bStr_base = self.generateSrcStrForMFMA(kernel, tPB, innerUnroll, vregSetIdx, vgprPerInputB, m, u, iui, idxB)
          aStr     = vgpr(aStr_base, vgprPerInputA)
          bStr     = vgpr(bStr_base, vgprPerInputB)
          Str0     = aStr if tPB["tile01Idx"] else bStr
          Str1     = bStr if tPB["tile01Idx"] else aStr

          if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            idxM     = idxB if kernel["ProblemType"]["Sparse"] == 2 else idxA
            m_new    = idxM*self.states.numReadsIterCoalescedMetadata
            mStr     = "ValuMetadata_X%u_I%u+%u+%u+%u" % (vgprBufferM_new, iuiM_new, m_new, vgprBufferM_new_offset, iuiM_new_offset)
            mStr     = vgpr(mStr, vgprPerInputM)

          if kernel["ProblemType"]["DataType"].isComplex():
            # override because complex mul is emulated by 4 mfma insts
            # TODO: adopt component system
            miInInstType = miOutInstType #"f32" for SingleComplex, "f64" for DoubleComplex
            ccA = kernel["ProblemType"]["ComplexConjugateA"]
            ccB = kernel["ProblemType"]["ComplexConjugateB"]
            ccVgprs = [None]*3 # three terms that can be negated: [real1, imag0, imag1]
            ccInsts = [None]*3
            accImOffset = accVgprImagNumOffset(kernel)
            accStartSrcImg = accStartSrc+accImOffset
            accEndSrcImg = accStartSrcImg + accs_per_wave - 1

            # vgpr A,B setting. In complex case, numRegistersIn does not match. Use numRegistersOut instead
            ar_base = aStr_base
            ai_base = ar_base + "+%u"%numRegistersOut
            ar = vgpr(ar_base, numRegistersOut)
            ai = vgpr(ai_base, numRegistersOut)
            br_base = bStr_base
            bi_base = br_base + "+%u"%numRegistersOut
            br = vgpr(br_base, numRegistersOut)
            bi = vgpr(bi_base, numRegistersOut)
            minus_ar = ar.getMinus()
            minus_ai = ai.getMinus()
            if miOutInstType == InstType.INST_F32:
              VAddX = VAddF32
            elif miOutInstType == InstType.INST_F64:
              VAddX = VAddF64
            else:
              printExit("Unsupported v_add type %s"%miOutInstType)
            offsetVgpr = [0,0,0]
            forceGenerate = ccA and ccB # so far, v_add is always necessary for ccA and ccB case
            if ccA == ccB:
              arrayIndex = 0
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate r1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = VAddX(dst=vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), src0=minus_ai, src1=0, comment="Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
            if ccA:
              arrayIndex = 1
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i0")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = VAddX(dst=vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), src0=minus_ai, src1=0, comment="Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
            if ccB:
              arrayIndex = 2
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ar not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = VAddX(dst=vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), src0=minus_ar, src1=0, comment="Ar=-Ar")
                zgemmVaddSrcCheck[arrayIndex].append(ar)
            (src0, src1) = (br, ar) if kernel["SourceSwap"] else (ar, br)
            for inst in ccInsts:
              if inst is not None:
                imod.add(inst)
            variant = [kernel["MatrixInstM"], kernel["MatrixInstN"], kernel["MatrixInstK"], kernel["MatrixInstB"]]
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), a=src0, b=src1, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), \
                     comment="Cr += Ar*Br"))
            (src0, src1) = (bi, (vgpr(ccVgprs[0] + offsetVgpr[0], numRegistersOut) if ccVgprs[0] else ai)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[0] + offsetVgpr[0], numRegistersOut) if ccVgprs[0] else ai), bi)
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), a=src0, b=src1, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), \
                     comment="Cr += %sAi*Bi"%("-" if ccVgprs[0] else "")))
            (src0, src1) = (br, (vgpr(ccVgprs[1] + offsetVgpr[1], numRegistersOut) if ccVgprs[1] else ai)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[1] + offsetVgpr[1], numRegistersOut) if ccVgprs[1] else ai), br)
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=self.accVgprReadWriteIndex(kernel, (accStart+accImOffset), (accEnd-accStart+1)), a=src0, b=src1, acc2=self.accVgprReadWriteIndex(kernel, accStartSrcImg, (accEndSrcImg-accStartSrcImg+1)), \
                     comment="Ci += %sAi*Br"%("-" if ccVgprs[1] else "")))
            (src0, src1) = (bi, (vgpr(ccVgprs[2] + offsetVgpr[2], numRegistersOut) if ccVgprs[2] else ar)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[2] + offsetVgpr[2], numRegistersOut) if ccVgprs[2] else ar), bi)
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=self.accVgprReadWriteIndex(kernel, (accStart+accImOffset+accStoreCIdx), (accEnd-accStart+1)), a=src0, b=src1, acc2=self.accVgprReadWriteIndex(kernel, accStartSrcImg, (accEndSrcImg-accStartSrcImg+1)), \
                     comment="Ci += %sAr*Bi"%("-" if ccVgprs[2] else "")))
            for v in ccVgprs:
              if v is not None: self.vgprPool.checkIn(v)
          else:

            if kernel["SourceSwap"]:
              src0 = Str1
              src1 = Str0
            else:
              src0 = Str0
              src1 = Str1

            variant = [kernel["MatrixInstM"], kernel["MatrixInstN"], kernel["MatrixInstK"], kernel["MatrixInstB"]]

            waits = self.mfmaIter_waitCount(kernel)
            if waits > 0 and prevAccIdx == accIdx:
              imod.add(SNop(waits - 1, "Wait for C"))
            if(kernel["ProblemType"]["Sparse"]):
              if kernel["DirectToVgprSparseMetadata"]:
                miWaveTile = kernel["MIWaveTileB"] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MIWaveTileA"]
                idx = idx1 if kernel["ProblemType"]["Sparse"] == 2 else idx0
                accInStart = miWaveTile * kernel["LoopIters"] * unrollLoopIdx + idx * kernel["LoopIters"] + unrollIdx
                imod.add(SMFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                        acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                        a=src0, b=src1, metadata=vgpr("ValuMetadata+%u"%(accInStart)), \
                                        comment="left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
              else:
                imod.add(SMFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                           acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                           a=src0, b=src1, metadata=mStr, \
                           comment="left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
            else:
              if kernel["UseF32XEmulation"]:
                abOffsetStr = "+2" if kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16 else "+4"
                if kernel["SourceSwap"]:
                  src1_0     = vgpr(aStr_base[:-4], vgprPerInputA / 2)
                  src1_1     = vgpr(aStr_base[:-4] + abOffsetStr, vgprPerInputA / 2)
                  src0_0     = vgpr(bStr_base[:-4], vgprPerInputB / 2)
                  src0_1     = vgpr(bStr_base[:-4] + abOffsetStr, vgprPerInputB / 2)
                else:
                  src1_0     = vgpr(bStr_base[:-4], vgprPerInputB / 2)
                  src1_1     = vgpr(bStr_base[:-4] + abOffsetStr, vgprPerInputB / 2)
                  src0_0     = vgpr(aStr_base[:-4], vgprPerInputA / 2)
                  src0_1     = vgpr(aStr_base[:-4] + abOffsetStr, vgprPerInputA / 2)

                imod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                       acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                       a=src0_0, b=src1_0, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), neg=neg_flag,\
                                       comment="src0_h*src1_h, left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
                if kernel["SourceSwap"]:
                  imod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                        acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                        a=src0_0, b=src1_1, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), neg=neg_flag,\
                                        comment="src0_h*src1_l, left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
                  imod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                        acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                        a=src0_1, b=src1_0, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), neg=neg_flag,\
                                        comment="src0_l*src1_h, left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
                else:
                  imod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                        acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                        a=src0_1, b=src1_0, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), neg=neg_flag,\
                                        comment="src0_l*src1_h, left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
                  imod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                        acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                        a=src0_0, b=src1_1, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), neg=neg_flag,\
                                        comment="src0_h*src1_l, left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
              else:
                imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                       acc=self.accVgprReadWriteIndex(kernel, (accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                       a=src0, b=src1, acc2=self.accVgprReadWriteIndex(kernel, accStart, (accEnd-accStart+1)), neg=neg_flag,\
                                       comment="left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))
            prevAccIdx = accIdx

      if kernel["ExpertSchedulingMode"] > 0:
        imod.add(SWaitAlu(va_vdst=0, comment="wait for the current iter's writes to complete"))

    # release register
    if kReg_first is not None: self.vgprPool.checkIn(kReg_first)
    if kReg is not None: self.vgprPool.checkIn(kReg)
    if abReg is not None: self.vgprPool.checkIn(abReg)
    if tmpVgpr is not None: self.vgprPool.checkIn(tmpVgpr)

    mfmaMod = Module("mfmaCode")
    if self.do["MAC"]:
      shiftK.add(postShiftK)
      mfmaMod.add(shiftK)
      mfmaMod.add(imod)

    return mfmaMod

  ##############################################################################
  # At Least 1 Unroll
  # prefetch means this is in the prefetch code, either before unroll loop
  # or in the PAP code.
  # isOptNLL : this is for the store-interleaved NLL optimization
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isNGLL=False, NLLindex=0, NLLnum=1):
    isLongBranch = False
    if kernel["EnableMatrixInstruction"] and kernel["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
      acclen = getAccToArchLen(kernel)
      # Just a rough calculation
      if acclen > 100 or (kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel'):
        isLongBranch = True

    module = Module("openSumAtLeastUnroll")
    if prefetch:
      if not isOptNLL:
        module.add(self.checkLastIter(kernel))
        if kernel["StorePriorityOpt"]:
          module.add(SSetPrior(prior=0, comment="optimization store"))
        if self.states.doShadowInit:
          shadowName = Label.getFormatting("ShadowInitStart")
          module.add(SCBranchSCC1(labelName=shadowName, \
              comment="skip to ShadowInitStart iter b/c numIter==0"))
        else:
          # This branch could potentially be very far e.g. > SIMM16
          module.addComment1("after InitC, skip to end of prefetch last iter if numIter==0")
          # use positive offset only long jump
          if kernel["SuppressNoLoadLoop"]:
            loopChar = self.states.indexChars[ \
                kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx]]
            lastIterEnd = Label("LoopEnd%s"%loopChar, "")
            module.add(SCBranchSCC1(labelName=lastIterEnd, \
                       comment="skip to unrollLoop end loop%s iter b/c numIter==0" % loopChar))
          else:
            lastIterEnd = Label("PrefetchGlobalLastIterEnd", "")
            # use positive offset only long jump
            with self.allocTmpSgpr(3) as tmpSgprInfo:
              module.add(self.longBranchScc1(lastIterEnd, posNeg=1, tmpSgprInfo=tmpSgprInfo))

    else:
      if isOptNLL and NLLindex==0:
        skipOptNLL = Label("OptNLL_End", "")
        with self.allocTmpSgpr(4) as tmpSgprInfo:
          tmpSgpr = tmpSgprInfo.idx
          placeHolder="skipOptNLL_placeholder" if self.states.FactorDim == 3 else None
          module.add(self.checkIsBetaZero(kernel, tmpSgprInfo, skipOptNLL, isLongBranch=isLongBranch, placeHolder=placeHolder, posNeg=1))

          # check alpha
          if self.do["ApplyAlpha"]:
            # (The new hgemm (h,h,h,h,s,s) is included in ComputeType=Single)
            if kernel["ProblemType"]["ComputeDataType"].isHalf():
              if kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["StreamK"]:
                module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=1.0, comment="Alpha == 1.0 ?"))
              else:
                # for (h,h,h,h,h,h) no HPA,
                module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x3c003c00), comment="Packed alpha==1.0"))
                module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=sgpr(tmpSgpr), comment="alpha == 1.0?"))

            # Shouldn't go here. Currently, DataType=B->ComputeDataType=S
            # (bf-gemm is included in ComputeType=Single)
            elif kernel["ProblemType"]["ComputeDataType"].isBFloat16():
              module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x3f803f80), comment="Packed alpha==1.0"))
              module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=sgpr(tmpSgpr), comment="alpha == 1.0?"))

            elif kernel["ProblemType"]["ComputeDataType"].isInt32():
              module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=1, comment="Alpha == 1.0 ?"))

            # This covers sgemm, bfgemm + HPA (b,b,b,b,s,s), and also hgemm (h,h,h,h,s,s)
            elif kernel["ProblemType"]["ComputeDataType"].isSingle():
              module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=1.0, comment="Alpha == 1.0 ?"))

            elif kernel["ProblemType"]["ComputeDataType"].isDouble():
              module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(0x00000000), comment="Low part of double 1.0"))
              module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0x3ff00000), comment="High part of double 1.0"))
              module.add(SCmpEQU64(src0=sgpr("Alpha",2), src1=sgpr(tmpSgpr,2), comment="Alpha == 1.0 ?"))

            elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
              module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(1.0), comment="Real part of 1.0"))
              module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0.0), comment="Imaginary part of 1.0"))
              module.add(SCmpEQU64(src0=sgpr("Alpha",2), src1=sgpr(tmpSgpr,2), comment="Alpha == 1.0 ?"))

            elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
              module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(0x00000000), comment="lsb of real part of 1.0"))
              module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0x3ff00000), comment="msb of real part of 1.0"))
              module.add(SCmpEQU64(src0=sgpr("Alpha",2), src1=sgpr(tmpSgpr,2), comment="Alpha.real == 1.0 ?"))
              if placeHolder == None:
                module.add(SCBranchSCC0(labelName=skipOptNLL.getLabelName(), comment="branch if alpha.real != 1"))
              else:
                skipOptNLLModule = Module("skipOptNLL_placeholder")
                skipOptNLLModule.addComment1("branch if alpha.real != 1")
                module.add(skipOptNLLModule)
              module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(0x00000000), comment="lsb of imag part of 0.0"))
              module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0x00000000), comment="msb of imag part of 0.0"))
              module.add(SCmpEQU64(src0=sgpr("Alpha+2",2), src1=sgpr(tmpSgpr,2), comment="Alpha.imag == 0.0 ?"))

            if placeHolder == None:
              if isLongBranch:
                module.add(self.longBranchScc0(skipOptNLL, posNeg=1, tmpSgprInfo=tmpSgprInfo, comment="branch if alpha != 1"))
              else:
                module.add(SCBranchSCC0(labelName=skipOptNLL.getLabelName(), comment="branch if alpha != 1"))
            else:
              skipOptNLLModule = Module("skipOptNLL_placeholder")
              skipOptNLLModule.addComment1("branch if alpha != 1")
              module.add(skipOptNLLModule)
            module.addSpaceLine()

          placeHolder = "skipOptNLL_scc1_placeholder" if self.states.FactorDim == 3 else None
          module.add(self.checkIsEdge(kernel, tmpSgprInfo, skipOptNLL, skipOptNLL, isLongBranch=isLongBranch, placeHolder=placeHolder))
          module.addSpaceLine()

          # Check tail loop required:
          # Skip tail loop check if noTailLoop is true
          if not kernel["NoTailLoop"]:
            loopChar = self.states.indexChars[ \
                kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx]]
            module.add(scalarStaticDivideAndRemainder(tmpSgpr, tmpSgpr+1, "SizesSum+%u"%self.states.unrollIdx, \
                      kernel["DepthU"], ContinuousRegister(tmpSgpr+2, 2), 2))
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr+1), src1=0, comment="numIter%s == 0"%loopChar ))
            if placeHolder == None:
              if isLongBranch:
                module.add(self.longBranchScc0(skipOptNLL, posNeg=1, tmpSgprInfo=tmpSgprInfo, comment="skip if tail loop required"))
              else:
                module.add(SCBranchSCC0(labelName=skipOptNLL.getLabelName(), comment="skip if tail loop required"))
            else:
              skipOptNLLModule = Module("skipOptNLL_placeholder")
              skipOptNLLModule.addComment1("skip if tail loop required")
              module.add(skipOptNLLModule)

        # save the vgprPool for generating the normal path.
        # so we can avoid explicit vgpr check-in/out
        self.savedVgprPool = deepcopy(self.vgprPool)
        self.savedSgprPool = deepcopy(self.sgprPool)

        # comment out the following codes that attempt to reduce vgpr consumption
        # however, the kernel vgpr count is governed by peak vgpr consumption so saving
        # a few here shouldn't affect kernel's overall vgpr consumption.
        # the following code is for reference and will be removed in the future
        """
        added = [] # track registers added to pool
        if kernel["PrefetchGlobalRead"]:
          if not kernel["DirectToLdsA"]:
            added.append(self.vgprPool.addRange(self.states.a.startVgprG2L, \
                self.states.a.startVgprG2L+self.states.a.numVgprG2L-1, "startOptNLL"))
            added.append(self.vgprPool.addRange(self.states.a.startVgprLocalWriteAddr, \
                         self.states.a.startVgprLocalWriteAddr, "startOptNLL"))
          if not kernel["DirectToLdsB"]:
            added.append(self.vgprPool.addRange(self.states.b.startVgprG2L, \
                self.states.b.startVgprG2L+self.states.b.numVgprG2L-1, "startOptNLL"))
            added.append(self.vgprPool.addRange(self.states.b.startVgprLocalWriteAddr, \
                         self.states.b.startVgprLocalWriteAddr, "startOptNLL"))

        if kernel["BufferLoad"]:
          added.append(self.vgprPool.addRange(self.startVgprGlobalReadOffsetA, \
              self.startVgprGlobalReadOffsetB, "startOptNLL"))
        else:
          added.append(self.vgprPool.addRange(self.startVgprGlobalReadAddressesA, \
              self.startVgprGlobalReadAddressesB, "startOptNLL"))
        module.addComment1("reclaim VGPRS: " + ", ".join(added))
        """

      if (not isNGLL) and NLLnum == 2:
        OptOrOrd = "Opt" if isOptNLL else "Ord"
        loopLabel2ndNLL = Label("%sNLL_second"%(OptOrOrd), "second %s NoLoadLoop entry"%OptOrOrd )
        # NLL + double buffer (NLLnum==2) case (PGR1/2), we need to generate 2 NLL (first and second buffer)
        if NLLindex == 0:
          # first NLL, jump to second code if OrigLoopCounter is odd
          module.add(SBitcmp1B32(src0=sgpr("OrigLoopCounter"), src1=0, comment="test if OrigLoopCounter is Odd ?"))
          module.add(SCBranchSCC1(labelName=loopLabel2ndNLL.getLabelName(), comment="jump to second NoLoadLoop" ))
        else: # NLLindex==1
          module.add(loopLabel2ndNLL)

    return module

  ##############################################################################
  def closeSumAtLeastUnroll(self, kernel, tPA, tPB, prefetch, isOptNLL, isNGLL, isNotLast=False):
    module = Module("closeSumAtLeastUnroll")
    if not prefetch:
      if isNGLL:
        if kernel["ProblemType"]["Sparse"] and kernel["PrefetchGlobalRead"] == 2 and kernel["DirectToVgprSparseMetadata"]:
          for i in range(0, self.states.m.numVgprValuPerBlock):
            module.add(VMovB32(vgpr("ValuMetadata+%u"%i), vgpr("ValuMetadata+%u+%u"%(self.states.m.numVgprValuPerBlock, i)), \
                                    comment="copy ValuMetadata blk1 to blk0"))
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          # PGR2 + DTVA/B case, we need to decrement loop counter after NGLL
          # This needs to be before toPGR1 label to avoid decrementing loop counter in loopCounter==1 case
          loopIdx = self.states.unrollIdx
          loopChar = self.states.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
          loopCounter = self.loopCounter(kernel, loopIdx)
          module.add(SSubU32(dst=loopCounter, src0=loopCounter, src1=1, comment="dec counter%s"%(loopChar)))
        toPGR1 = Label(self.labels.getName("toPGR1"), "")
        if isNotLast:
          module.add(SBranch(labelName=toPGR1.getLabelName(), comment="Branch to toPGR1"))
        else:
          module.add(toPGR1)
      else:
        suffix = "OptNLL" if isOptNLL else "OrdNLL"
        toPGR1end = Label(self.labels.getName("toPGR1end_%s"%suffix), "")
        if isNotLast:
          module.add(SBranch(labelName=toPGR1end.getLabelName(), comment="Branch to toPGR1end"))
        else:
          module.add(toPGR1end)
          if isOptNLL:
              endSumLabel = "Summation_End_OptNLL"

              module.addComment0("Stores for OptNLL")
              module.add(self.endSummation(kernel, tPA, tPB, False, endSumLabel, isOptNLL))

              # dot2: WaveSplitK reduction for opt NLL
              if kernel["NumWaveSplitK"] > 1:
                module.add(self.waveSplitKReduction(kernel))

              # perhaps could work with LSU>1 by adding other indices here, but not tested
              assert (kernel["LocalSplitU"] == 1)
              module.add(self.notLocalSplitUGlobalWriteIndices(kernel))

              # add stores for opt NLL
              (fullVw, elements, fullVw_1, elements_1) = self.notLocalFullTileElements(kernel, False)
              alpha = False
              beta = False
              module.add(self.globalWriteElements(kernel, tPA, tPB, [fullVw], [fullVw_1], [elements], [elements_1], True, applyAlpha=alpha, betas=[beta], edges=[False]))

              self.cleanupGlobalWrite(kernel)
              module.addSpaceLine()
              module.add(self.functionEnd(kernel, addLabel=False))
              module.add(Label("OptNLL_End", ""))

          else:
            module.add(Label("PrefetchGlobalLastIterEnd", ""))

    # swap back vgpr pool if any
    def swapBackGprPool(gprPool, savedGprPool, isNotLast):
      if savedGprPool != None and (not isNotLast):
        # in case pool size in current path is larger than pool size in main path
        # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
        newSize = gprPool.size()
        savedGprPool.appendPool(newSize)
        return savedGprPool, None # restore vgprPool before alternate path
      return gprPool, savedGprPool

    self.vgprPool, self.savedVgprPool = swapBackGprPool(self.vgprPool, self.savedVgprPool, isNotLast)
    self.sgprPool, self.savedSgprPool = swapBackGprPool(self.sgprPool, self.savedSgprPool, isNotLast)
    return module

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def incrementSrd(self, tP, incLower, incUpper):
    imod = Module("incrementSrd")
    tc = tP["tensorChar"]

    imod.add(SAddU32(dst=sgpr("Srd%s+0"%(tc)), \
                     src0=sgpr("Srd%s+0"%(tc)), \
                     src1=incLower, \
                     comment="gra SRD += inc(lower)" ))
    imod.add(SAddCU32(dst=sgpr("Srd%s+1"%(tc)), \
                      src0=sgpr("Srd%s+1"%(tc)), \
                      src1=incUpper, \
                      comment="gra SRD += inc(upper)" ))

    # also have to move the boundary since we change the base
    # so less buffers to the edge:
    if self.states.use64bShadowLimit:
      imod.add(SSubU32(dst=sgpr("ShadowLimit%s+0"%tc), \
                       src0=sgpr("ShadowLimit%s+0"%tc), \
                       src1=incLower, \
                       comment="limit -= inc)"))
      imod.add(SSubBU32(dst=sgpr("ShadowLimit%s+1"%tc), \
                        src0=sgpr("ShadowLimit%s+1"%tc), \
                        src1=incUpper, \
                        comment="limit -= inc)" ))
      imod.add(SCmpEQU32(src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="are we within 2^32?"))
      if 1: # self.states.staggerU:
        # staggerU case, need to restore BufferLimit when ShadowLimit goes to negative value
        imod.add(SCSelectB32(dst=sgpr("Srd%s+2"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1="BufferLimit", comment="Move shadow to real if we are within 2^32"))
      else:
        imod.add(SCMovB32(dst=sgpr("Srd%s+2"%tc), src=sgpr("ShadowLimit%s+0"%tc), comment="Move shadow to real if we are within 2^32"))
    else:
      imod.add(SSubU32(dst=sgpr("Srd%s+2"%(tc)), \
                       src0=sgpr("Srd%s+2"%(tc)), \
                       src1=incLower, \
                       comment="limit -= inc)" ))
    return imod

  def incrementMetadataSrd(self, incSparseLower, incSparseUpper):
    imod = Module("incrementMetadataSrd")

    imod.add(SAddU32(sgpr("SrdMetadata+0"), \
                     sgpr("SrdMetadata+0"), \
                     incSparseLower, \
                     "gra SRD += incSparse(lower)"))
    imod.add(SAddCU32(sgpr("SrdMetadata+1"), \
                      sgpr("SrdMetadata+1"), \
                      incSparseUpper, \
                      "gra SRD += incSparse(uppper)" ))

    # also have to move the boundary since we change the base
    # so less buffers to the edge:
    if self.states.use64bShadowLimit:
      imod.add(SSubU32(sgpr("ShadowLimitMetadata+0"), \
                       sgpr("ShadowLimitMetadata+0"), \
                       incSparseLower, \
                       "limit -= incSparse(lower)"))
      imod.add(SSubBU32(sgpr("ShadowLimitMetadata+1"), \
                        sgpr("ShadowLimitMetadata+1"), \
                        incSparseUpper, \
                       "limit -= incSparse(uppper)" ))
      imod.add(SCmpEQU32(sgpr("ShadowLimitMetadata+1"), 0, "are we within 2^32?"))
      if 1: # self.states.staggerU:
        # staggerU case, need to restore BufferLimit when ShadowLimit goes to negative value
        imod.add(SCSelectB32(dst=sgpr("SrdMetadata+2"), src0=sgpr("ShadowLimitMetadata+0"), src1="BufferLimit", comment="Move shadow to real if we are within 2^32"))
      else:
        imod.add(SCMovB32(sgpr("SrdMetadata+2"), sgpr("ShadowLimitMetadata+0"), "Move shadow to real if we are within 2^32"))
    else:
      imod.addInst(SSubU32(sgpr("SrdMetadata+2"), \
                           sgpr("SrdMetadata+2"), \
                           incSparseLower, \
                           "limit -= inc)" ))
    return imod

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def setTailSrd(self, tP, incLower):
    # In SuppressNoLoadLoop, the final loop iteration moves the SRD base forward
    # and the ShadowLimit backwards by one extra 'click' of GlobalReadIncs[AB].
    # Note the ShadowLimit may become negative - for example edge tiles where the
    # increment is > tile width.
    # The SuppressNoLoadLoop mode also forces the SRD limit to 0 on the final iteration.
    # The code here undoes the final click step by moving the base backwards and the
    # limit forwards (reading from the ShadowLimit).
    # It only works if use64bShadowLimit is enabled (since this enables use of the ShadowLimit)

    tc = tP["tensorChar"]
    module = Module("setTailSrd")
    incUpper = 0

    module.add(SSubU32(dst=sgpr("Srd%s+0"%(tc)), \
          src0=sgpr("Srd%s+0"%(tc)), src1=incLower, \
          comment="gra SRD -= inc(lower)" ))
    module.add(SSubBU32(dst=sgpr("Srd%s+1"%(tc)), \
          src0=sgpr("Srd%s+1"%(tc)), src1=incUpper, \
          comment="gra SRD -= inc(upper)" ))

    # using Shadow limit here which only works with 64-bit PBC:
    assert(self.states.use64bShadowLimit)

    module.add(SAddU32(dst=sgpr("ShadowLimit%s+0"%tc), \
          src0=sgpr("ShadowLimit%s+0"%tc), src1=incLower, \
          comment="limit -= inc)"))
    module.add(SAddCU32(dst=sgpr("ShadowLimit%s+1"%tc), \
          src0=sgpr("ShadowLimit%s+1"%tc), src1=incUpper, \
          comment="limit -= inc)" ))
    module.add(SCmpEQU32(src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="are we within 2^32?"))
    module.add(SCMovB32(dst=sgpr("Srd%s+2"%tc), src=sgpr("ShadowLimit%s+0"%tc), comment="Move shadow to real if we are within 2^32"))

    return module

  ##############################################################################
  # Global Read: Increment A/B
  # loopIdx is summation idx:
  #   self.states.unrollIdx, or an idx from 0..NumIndicesSummation
  # prefetchIndex is >0 (1...PrefetchGlobalRead) if this increment follows a
  #   global prefetch or 0 otherwise
  ##############################################################################
  def globalReadIncrement(self, kernel, imod, loopIdx, tP, prefetchIndex):
    if not self.do["GlobalInc"]: return ""
    tc = tP["tensorChar"]
    loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]

    imod.addComment1("global read inc %s loop%s"%(tc,loopChar))

    if kernel["BufferLoad"]:
      # TODO - does this handle N-dim tensors correctly?
      #if tP["isB"]:
      #  module.add(SMovB32(dst=sgpr("OffsetB"), src=sgpr("SrdB+0"), comment="hack to save"))
      if loopIdx == self.states.unrollIdx: # and self.states.staggerU
        # add a wrap increment, if needed:
        with self.allocTmpSgpr(4) as tmpSgprInfo:
          incLower = tmpSgprInfo.idx
          incUpper = incLower + 1
          tmpS =    incLower + 2
          tmpIncSparse = incLower + 3
          suStr = "StaggerUIter"
          tcOther = "B" if tP["isA"] else "A"
          if kernel["PrefetchGlobalRead"] == 2 and (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc] and (not kernel["DirectToVgpr%s"%tcOther]):
            suStr += "DTV"
          if prefetchIndex:
            imod.add(SAddU32(dst=sgpr(tmpS), src0=self.loopCounter(kernel, self.states.unrollIdx), src1=prefetchIndex, comment="remove pf(%u)"%prefetchIndex))
            imod.add(SCmpEQU32(src0=sgpr(suStr), src1=sgpr(tmpS), comment="Is this wrapIter? (pf)"))
          else:
            imod.add(SCmpEQU32(src0=self.loopCounter(kernel, self.states.unrollIdx), \
                      src1=sgpr(suStr), comment="Is this the wrapIter?"))
          imod.add(SCSelectB32(dst=sgpr(incLower), src0=sgpr("WrapU%s+0"%tc), src1=sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                      comment="incLower <- ?"))
          imod.add(SCSelectB32(dst=sgpr(incUpper), src0=sgpr("WrapU%s+1"%tc), src1=0,
                      comment="incUpper <- ?"))
          imod.add(self.incrementSrd(tP, sgpr(incLower), sgpr(incUpper)))

          if kernel["ProblemType"]["Sparse"]:
            if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) :
              tc = "Metadata"
              if kernel["DirectToVgprSparseMetadata"]:
                incSparse = tmpIncSparse
                imod.add(self.calculateIncrementMetadata(kernel, incSparse))
              else:
                incSparse = "GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)
              if prefetchIndex:
                imod.add(SCmpEQU32(src0=sgpr("StaggerUIter"), src1=sgpr(tmpS), comment="Is this wrapIter? (pf)"))
              else:
                imod.add(SCmpEQU32(src0=self.loopCounter(kernel, self.states.unrollIdx), \
                        src1=sgpr("StaggerUIter"), comment="Is this the wrapIter?"))
              imod.add(SCSelectB32(dst=sgpr(incLower), src0=sgpr("WrapU%s+0"%tc), src1=sgpr(incSparse), \
                          comment="incLower <- ?"))
              imod.add(SCSelectB32(dst=sgpr(incUpper), src0=sgpr("WrapU%s+1"%tc), src1=0,
                          comment="incUpper <- ?"))
              if kernel["DirectToVgprSparseMetadata"]:
                imod.add(self.incrementMetadataSrd(sgpr(incLower), sgpr(incUpper)))
              else:
                imod.add(self.incrementSrd(tP["tpsMetadata"], sgpr(incLower), sgpr(incUpper)))

      else:
        if loopIdx != self.states.unrollIdx or (tc in ('A', 'B') and kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"%tc]):
          with self.allocTmpSgpr(1) as tmpSgprInfo:
            incUpper = tmpSgprInfo.idx
            # GRO may be negative for other summation if stride-other < stride-unroll or if mirror dim.
            imod.add(SAShiftRightI32(dst=sgpr(incUpper), shiftHex=31, src=sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), comment="sign-extend"))
            imod.add(self.incrementSrd(tP, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), sgpr(incUpper)))
        else:
          incUpper = 0 # GRO is positive for loop unroll
          imod.add(self.incrementSrd(tP, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), hex(incUpper)))

        if kernel["ProblemType"]["Sparse"]:
          if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) :
            tc = "Metadata"
            imod.addComment1("global read inc metadata loop%s"%(loopChar))
            if kernel["DirectToVgprSparseMetadata"]:
              with self.allocTmpSgpr(1) as tmpSgprInfo:
                incSparse = tmpSgprInfo.idx
                imod.add(self.calculateIncrementMetadata(kernel, incSparse))
                imod.add(self.incrementMetadataSrd(sgpr(incSparse), 0))
            else:
              incSparse = "GlobalReadIncs%s+%u"%(tc,loopIdx)
              if loopIdx != self.states.unrollIdx or (kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"%tc]):
                with self.allocTmpSgpr(1) as tmpSgprInfo:
                  incUpper = tmpSgprInfo.idx
                  # GRO may be negative for other summation if stride-other < stride-unroll or if mirror dim.
                  imod.add(SAShiftRightI32(dst=sgpr(incUpper), shiftHex=31, src=sgpr(incSparse), comment="sign-extend"))
                  imod.add(self.incrementMetadataSrd(sgpr(incSparse), sgpr(incUpper)))
              else:
                incUpper = 0 # GRO is positive for loop unroll
                imod.add(self.incrementSrd(tP["tpsMetadata"], sgpr(incSparse), hex(incUpper)))

    else:
      graIdx = 0
      for _ in range(0, tP["nrp"]):
        for _ in range(0, tP["nrpv"]):
          for _ in range(0, tP["nrc"]):
            for _ in range(0, tP["nrcv"]//tP["nrcvpi"]):
              if self.states.globalReadIncsUseVgpr:
                imod.add(VAddCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    src1=vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], 2*loopIdx)), \
                    comment="gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar)))
                imod.add(VAddCCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    src1=vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], 2*loopIdx)), \
                    src2=VCC(), \
                    comment="gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar)))
              else:
                imod.add(VAddCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    src1=sgpr("GlobalReadIncs%s+%u"%(tP["tensorChar"], loopIdx)), \
                    comment="gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar)))
                imod.add(VAddCCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    src1=0,
                    src2=VCC(), \
                    comment="gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar)))
              graIdx += self.states.rpga

  def globalReadIncrementAB(self, kernel, tPA, tPB, loopIdx, prefetchIndex):
    imod = Module("globalReadIncrementAB")

    incCodeA = imod.add(Module("globalReadIncrementA"))
    if tPA != None:
      self.globalReadIncrement(kernel, incCodeA, loopIdx, tPA, prefetchIndex)
    incCodeB = imod.add(Module("globalReadIncrementB"))
    if tPB != None:
      self.globalReadIncrement(kernel, incCodeB, loopIdx, tPB, prefetchIndex)
    return imod

  ##############################################################################
  # Global Read:
  # globalReadTrueGuardK is called for loads in the tail loop
  # Must ensure each load is in bounds - either using buffer bounds
  # or exec-mask checks.
  ##############################################################################
  def globalReadGuardK(self, kernel, tP, doTailOpt = 0, optParams = None):
    module = Module("globalReadGuardK")
    tc = tP["tensorChar"]
    problemType = self.states.kernel["ProblemType"]
    numElementsPer4Bytes = int(4 / tP["bpeGR"])
    isTr = (tc == "A" or tc == "B") and kernel["enableGLTr%s"%tc]

    ########################################
    # Calculate Max Addr
    ########################################

    if isTr:
      # DirectToVgpr case, we need to calculate max address
      module.addComment1("Max read address offset for GLTr%s"%tc)

      maxGroVgpr = self.vgprPool.checkOut(1)

      tmpVgpr = self.vgprPool.checkOutAligned(2, 2, 'tmpVgpr')
      tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

      tmp = self.vgprPool.checkOut(1)
      tmp2 = self.vgprPool.checkOut(1)

      WvG_M = kernel["MIWaveGroup"][0]
      numKr = kernel["MatrixInstK"] // tP["glvw"]

      module.addComment0("calc last tile offset")
      module.add(vectorStaticDivide(maxGroVgpr, "Serial", kernel["WavefrontSize"], tmpVgprRes))
      if tP["isA"]:
        module.add(VAndB32(dst=vgpr(maxGroVgpr), src0=hex(WvG_M-1), src1=vgpr(maxGroVgpr), comment="GLTr%s: wave_id (along_M) mod MIWG[0]"%tc))
        module.add(VMulU32U24(dst=vgpr(maxGroVgpr), src0=numKr, src1=vgpr(maxGroVgpr), comment="GLTr%s: wave_id (along_M) *= numKr"%tc))
      elif tP["isB"]:
        # NB:
        #   Calc of w_id is: /= MIWG[0], not %= MIWG[1]
        module.add(VLShiftRightB32(dst=vgpr(maxGroVgpr), shiftHex=log2(WvG_M), src=vgpr(maxGroVgpr), comment="GLTr%s: wave_id (along_N) /= MIWG[0]"%tc))
        module.add(VMulU32U24(dst=vgpr(maxGroVgpr), src0=numKr, src1=vgpr(maxGroVgpr), comment="GLTr%s: wave_id (along_N) *= numKr"%tc))

      module.add(VBfeU32(dst=vgpr(tmp2), src0=vgpr("Serial"), src1=tP["bpeGR"]+1, src2=1, comment="GLTr%s: offset for the right half of the tile"%(tc)))
      module.add(VAddU32(dst=vgpr(maxGroVgpr), src0=vgpr(tmp2), src1=vgpr(maxGroVgpr), comment="GLTr%s: wave_id += offset for the right half of the tile"%(tc)))

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        if tP["glvw"] > 1:
          if tP["tlu"]:
            module.add(vectorStaticMultiply(vgpr(maxGroVgpr), vgpr(maxGroVgpr), tP["glvw"], tmpSgprInfo, comment="GLTr%s: tile * glvw(%u)"%(tc, tP["glvw"])))
          else:
            module.add(vectorStaticMultiply(vgpr(maxGroVgpr), vgpr(maxGroVgpr), tP["glvw"], tmpSgprInfo, comment="GLTr%s: unroll * glvw(%u)"%(tc, tP["glvw"])))

      strideIdx = tP["lsc"] if tP["tlu"] else tP["lsp"]
      stride = kernel[strideIdx]
      module.add(VAddU32(dst=vgpr(maxGroVgpr), src0=vgpr(maxGroVgpr), src1=stride*(tP["nrt"]-1)))

      module.addComment0("calc last unroll offset")
      module.add(VMovB32(dst=vgpr(tmp), src=sgpr("SizesSum+%u"%self.states.unrollIdx)))
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(vectorStaticRemainder(tmp, tmp2, tmp, kernel["DepthU"], tmpVgprRes, tmpSgprInfo))

      module.addComment0("final offset")
      module.add(VSubU32(dst=vgpr(tmp2), src0=vgpr(tmp2), src1=1, comment="GLTr%s: unroll idx - 1"%(tc)))
      bfArgs = (maxGroVgpr, maxGroVgpr, tmp2, tmp)
      module.add(MacroInstruction(name="GLOBAL_OFFSET_%s"%tc, args=bfArgs))

      module.addSpaceLine()

      self.vgprPool.checkIn(tmp)
      self.vgprPool.checkIn(tmp2)
      self.vgprPool.checkIn(tmpVgpr)
    elif not kernel["BufferLoad"]:
      with self.allocTmpSgpr(2) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        maxAddrSgpr = tmpSgpr
        module.addComment0("flat addressing - max read address = size[n] * stride[n-1]")
        dim = len(tP["ia"])-1 # dim
        sizeIdx = tP["ia"][dim]
        sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
        if sizeIdxIsSum:
          sizeIdx -= kernel["ProblemType"]["NumIndicesC"]
        # TODO-multiply by largest stride
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(maxAddrSgpr+0), sgpr(maxAddrSgpr+1),  \
                    sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
                    sgpr("Stride%s%s"%(tc, self.states.indexChars[tP['ia'][-1]])), \
                    comment="64b tensor%s size in elements"%tc))
        if log2(tP["bpeGR"]) > 0:
          module.add(SLShiftLeftB64(dst=sgpr(maxAddrSgpr,2), src=sgpr(maxAddrSgpr,2), \
            shiftHex=hex(log2(tP["bpeGR"])), comment="<- tensor%s size in bytes"%tc))
        else:
          module.addCommentAlign("<- tensor%s size in bytes (byte is 1, do nothing)")
        module.add(SAddU32(
            dst=sgpr(maxAddrSgpr+0), \
            src0=sgpr(self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"]), \
            src1=sgpr(maxAddrSgpr+0), \
            comment="prepend address lower"))
        module.add(SAddCU32(
            dst=sgpr(maxAddrSgpr+1), \
            src0=sgpr((self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"])+1), \
            src1=sgpr(maxAddrSgpr+1), \
            comment="prepend address upper"))
        # sgpr->vgpr
        maxAddrVgpr = self.vgprPool.checkOutAligned(2, 2, "maxAddrVgpr")
        module.add(VMovB32(dst=vgpr(maxAddrVgpr+0), src=sgpr(maxAddrSgpr+0), comment="sgpr->vgpr"))
        module.add(VMovB32(dst=vgpr(maxAddrVgpr+1), src=sgpr(maxAddrSgpr+1), comment="sgpr->vgpr"))

        # full exec mask
        fullExec = tmpSgpr
        sgprCnt = self.states.laneSGPRCount
        waveSize = kernel["WavefrontSize"]
        activeMask = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
        SMovBX     = SMovB32 if (waveSize == 32) else SMovB64
        module.add(SMovBX(dst=sgpr(fullExec,sgprCnt), src=activeMask, comment="to restore all threads active"))
        bpeVgpr = self.vgprPool.checkOut(1, "bpeVgpr")
        module.add(VMovB32(dst=vgpr(bpeVgpr), src=hex(tP["bpeGR"]), comment="bpeGR"))

        # can remove this?
        zeroVgpr = self.vgprPool.checkOut(1,"zeroVgpr")
        module.add(VMovB32(dst=vgpr(zeroVgpr), src=0, comment="zero"))

    def globalReadGuardKBody(tP, optParams = None):
      if optParams != None:
        jumpLabel = optParams.jumpLabel
        idx = optParams.idx
        tmpVgpr = optParams.tmpVgpr
        kLabelsList = optParams.kLabelsList
        behavior = optParams.behavior
        kSgpr = optParams.kSgpr
        vgprSlot = optParams.vgprSlot
        periodParam = optParams.periodParam
        loadNum = optParams.loadNum
        preDirectToLdsLoads = optParams.preDirectToLdsLoads
        finalLoop = optParams.finalLoop
      else:
        jumpLabel = None
        idx = 0
        tmpVgpr = None
        kLabelsList = []
        behavior = ""
        kSgpr = None
        vgprSlot = []
        periodParam = []
        loadNum = 0
        preDirectToLdsLoads = 0
        finalLoop = 0

      tc = tP["tensorChar"]
      self.vgprs.globalReadRegisters[tc] = []

      tcDataType = "" if tc == "Metadata" else tc
      graIdx = 0
      g2lIdx = 0
      loadWidth = tP["globalReadInstruction"].totalWidth
      # FIXME: Don't know why for grvw == 1, need further investigate
      glvwWorkaround = 8 * kernel["ProblemType"]["DataType"].numRegisters()
      dataType = kernel["ProblemType"]["DataType"] if tP["glvw"] < glvwWorkaround else kernel["ProblemType"]["DataType%s"%tcDataType]

      isGlc = bool(tP["NonTemporal"] & 0x1)
      isSlc = bool(tP["NonTemporal"] & 0x2)
      isNT  = bool(tP["NonTemporal"] & 0x4)
      isLds = True if (kernel["DirectToLds%s"%tc] and not kernel["NonDTLTailLoop%s"%tc]) else False
      isTr = (tc == "A" or tc == "B") and kernel["enableGLTr%s"%tc]
      is16b = dataType.isHalf() or dataType.isBFloat16()

      directToLdsLoads = 0
      if doTailOpt == 2 and behavior == "LOAD":
          directToLdsLoads += preDirectToLdsLoads
      prevLdsOffset    = 0

      instOffset = 0
      loopCnt = -1
      vgprIdx = 0
      loadCnt = 0
      perpStart = periodParam[0] if doTailOpt == 2 else 0
      perpEnd = periodParam[1] if doTailOpt == 2 else tP["nrp"]
      sPerpStart = periodParam[2] if doTailOpt == 2 else 0
      sPerpEnd = periodParam[3] if doTailOpt == 2 else tP["nrpv"]
      paraStart = periodParam[4] if doTailOpt == 2 else 0
      paraEnd = periodParam[5] if doTailOpt == 2 else tP["nrc"]
      sParaStart = periodParam[6] if doTailOpt == 2 else 0
      sParaEnd = periodParam[7] if doTailOpt == 2 else (tP["nrcv"]//tP["nrcvpi"])
      rStart = periodParam[8] if doTailOpt == 2 else 0
      rEnd = periodParam[9] if doTailOpt == 2 else 0

      requiredSgprs = min(self.sgprPool.size(), self.states.regCaps["MaxSgpr"])
      maxVgprs, occupancy = self.getMaxRegsForOccupancy(kernel["NumThreads"], self.vgprPool.size(), requiredSgprs, \
                                                      self.getLdsSize(kernel), self.agprPool.size(), self.states.doubleVgpr)

      tmpVgprPool = []
      packInst = Module()
      maxNumTempToUse = max(maxVgprs - self.vgprPool.size(), 1)

      for perp in range(perpStart, perpEnd):
        for sPerp in range(sPerpStart, sPerpEnd):
          for para in range(paraStart, paraEnd):
            for sPara in range(sParaStart, sParaEnd):
              i = sPara + (tP["nrcv"] // tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
              loopCnt = sPara + para * sParaEnd + sPerp * paraEnd * sParaEnd  + perp * sPerpEnd * paraEnd * sParaEnd

              graIdx = i * self.states.rpgo if kernel["BufferLoad"] else i * self.states.rpga
              g2lIdx = i * loadWidth * tP["bpeRatio"]
              if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc] and kernel["ConvertAfterDS"] and not isTr:
                # DTV + ConvertAfterDS case, if bpe > bpeGR, we need to shift g2lIdx for conversion
                if tP["bpe"] > tP["bpeGR"]:
                  g2lIdx *= tP["bpe"] // tP["bpeGR"]

              destVgprHi = None
              tmpVgprIdx = tmpVgpr
              dataIsByte = False
              packInt8Code = None
              eccOffset = 0

              instOffsetInc = 0 # increment value for instOffset. Need to apply after r loop

              r = 0
              numLoadVectorComp = int(loadWidth*self.states.bpr//tP["bpeGR"])

              if (kernel["ProblemType"]["DataType%s"%tcDataType].isDouble() or kernel["ProblemType"]["DataType%s"%tcDataType].isSingle())\
                 and kernel["BufferLoad"]:
                # adjustment for {d,s}gemm + BufferLoad
                # use same buffer_load instruction for tail loop as out of tail loop
                # this is mandatory for DirectToLds case. Also, it improves tail loop performance.
                numLoadVectorComp = numLoadVectorComp // kernel["GlobalReadVectorWidth%c"%tc]

              if isLds or isTr:
                numLoadVectorComp = 1

              int8TempVgpr = numLoadVectorComp - 1

              # for each component in vector
              while r < numLoadVectorComp:
                if doTailOpt == 1  and i == idx and ((r+1) % numElementsPer4Bytes != 0):
                  if kLabelsList != None:
                    module.add(kLabelsList.pop())
                    module.add(SCmpGeU32(src0=sgpr(kSgpr), src1=(r + 1), comment=""))
                    module.add(SCBranchSCC0(labelName=jumpLabel.getLabelName(), comment=""))

                numElementsPerLoad = 1

                if kernel["ConvertAfterDS"]:
                    dataType = kernel["ProblemType"]["DataType%s"%tcDataType]
                if dataType.isInt8() or dataType.is8bitFloat() or tP["isM"]:
                  # TODO-Int8, Check this:
                  # if tP["glvw"]>1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
                  # # Pack two FP16 values into a single load dword x2
                  #   numElementsPerLoad = 2
                  # elif self.states.archCaps["HasEccHalf"]:
                  #   destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi')
                  if isTr:
                    numElementsPerLoad = kernel["GlobalReadVectorWidth%c"%tc]
                  if (not tP["isM"]) and isLds:
                    if not kernel["NonDTLTailLoop%c"%tc]:
                      numElementsPerLoad = kernel["GlobalReadVectorWidth%c"%tc]
                    dataIsByte = False
                  else:
                    dataIsByte = True
                  # Check out 3 regs once , for component 1,2,3 (r = 1,2,3)
                  if doTailOpt == 1:
                    if r == 1:
                      packInt8Code = Module()
                    destVgprHi = tmpVgprIdx
                  else:
                    if doTailOpt == 2:
                      if r == 1:
                        packInt8Code = Module()
                      if r != 0:
                        destVgprHi = vgprSlot[vgprIdx]
                        if r >= rStart and r < rEnd and (r % 4 != 0):
                          vgprIdx = vgprIdx + 1
                    elif r == 1:
                      packInt8Code = Module()
                      destVgprHi = self.vgprPool.checkOut( int8TempVgpr , 'destVgprHi')
                  regIdx = r // 4
                  if (tP["localWriteInstruction"].blockWidth <= 0.5) and (r%2 == 0) and not tP["isM"]:
                      numVgprG2L = self.states.a.numVgprG2L if tc == 'A' else self.states.b.numVgprG2L
                      eccBpe = tP["bpeDS"] if kernel["ConvertAfterDS"] else max(tP["bpeGR"], tP["bpe"])
                      eccOffset = _getEccOffset(tP["globalReadInstruction"].totalWidth, bpr=self.states.bpr, bpe=eccBpe, \
                        glvw=tP["glvw"], idx=loopCnt, numVgprG2L=numVgprG2L)
                elif dataType.isHalf() or dataType.isBFloat16():
                  if isTr:
                    numElementsPerLoad = kernel["GlobalReadVectorWidth%c"%tc]
                  elif tP["glvw"]>1 and kernel["AssertSummationElementMultiple"] % 2 == 0 and not isLds:
                  # Pack two FP16 values into a single load dword x2
                    numElementsPerLoad = 2
                  elif isLds and not kernel["NonDTLTailLoop%c"%tc]:
                    numElementsPerLoad = kernel["GlobalReadVectorWidth%c"%tc]
                  elif self.states.archCaps["HasEccHalf"]:
                    # In some cards, loading half types into register will zero out
                    # the other half. Therefore we need to load into a separate register
                    # then pack 2 registers into one
                    if doTailOpt == 1:
                      destVgprHi = tmpVgprIdx
                    else:
                      if (tP["localWriteInstruction"].blockWidth == 0.5) and (r%2 == 0):
                        numVgprG2L = self.states.a.numVgprG2L if tc == 'A' else self.states.b.numVgprG2L
                        eccBpe = tP["bpeDS"] if kernel["ConvertAfterDS"] else max(tP["bpeGR"], tP["bpe"])
                        eccOffset = _getEccOffset(tP["globalReadInstruction"].totalWidth, bpr=self.states.bpr, bpe=eccBpe, \
                        glvw=tP["glvw"], idx=loopCnt, numVgprG2L=numVgprG2L)
                      else:
                        if doTailOpt == 2:
                          destVgprHi = vgprSlot[vgprIdx]
                          if r >= rStart and r < rEnd and (r % 2 != 0):
                            vgprIdx = vgprIdx + 1
                        else: #doTailOpt == 0 case
                          if r % 2 == 1:
                            tmpVgprPool.append(self.vgprPool.checkOut(1, 'destVgprHi'))
                            destVgprHi = tmpVgprPool[-1]
                  regIdx = r // 2
                elif dataType.isInt8x4() or dataType.isSingle():
                  # Only supported for buffer loads since it has OOB checks
                  if kernel["BufferLoad"]:
                    numElementsPerLoad = kernel["GlobalReadVectorWidth%c"%tc]
                  regIdx = r
                elif dataType.isDouble():
                  # Only supported for buffer loads since it has OOB checks
                  if kernel["BufferLoad"]:
                    numElementsPerLoad = kernel["GlobalReadVectorWidth%c"%tc] # adjust numElementsPerLoad for DGEMM
                  regIdx = r*2
                elif dataType.isSingleComplex():
                  regIdx = r*2
                elif dataType.isDoubleComplex() :
                  regIdx = r*4
                else:
                  printWarning("DataType unsupported")
                if doTailOpt == 0:
                  module.addComment0("g2l=%u, load component %u"%(g2lIdx, r))

                offset = 0
                hi8 = 0
                hi16 = 0

                if kernel["BufferLoad"] or isTr:
                  # Use buffer limit to stay in-bounds - the limit was set to edge when SRD initialized
                  # and each increment of SRD base in the unroll loop does a corresponding decrement
                  # of the srd limit - so base+limit stays constant and also points at maximum
                  # element that should be accessed.
                  if kernel["_UseSgprForGRO"]:
                    offsetVgpr = "GlobalReadOffset%s+0"%(tc)
                  else:
                    offsetVgpr = "GlobalReadOffset%s+%u"%(tc, graIdx)

                  # Vgpr for GRO
                  if not kernel["_UseSgprForGRO"]:
                    soffset = "0"
                  # instruction offset with Sgpr for GRO
                  elif isLds and kernel["UseInstOffsetForGRO"]:
                    soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx))
                  # Sgpr for GRO
                  else:
                    soffset = "0" if graIdx == 0 else sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

                  unrollMirrorWithSoffset = kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx] in problemType["MirrorDims%s"%tc] and soffset != "0"
                  # ScalarGlobalReadOffset should be negative value with unroll mirroring.
                  # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
                  if unrollMirrorWithSoffset:
                    codeMod = Module("mirrorIdx%u"%loopCnt)
                    codeMod.add(VSubU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset, comment="mirror unroll: GRO=GRO-SGRO, soffset=0"))
                    module.add(codeMod)
                    soffset_prev = soffset
                    soffset = "0"

                  if isLds:
                    # need to increment ldsInc only once per each loopCnt
                    # this is pre count up, so increment it at r == 0
                    if r == 0:
                      ldsInc = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalReadVectorWidth%c"%tc] * tP["bpeGR"]
                    else:
                      ldsInc = 0
                    if kernel["LdsBlockSizePerPad%s"%tc] != 0:
                      ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeGR"]
                    else:
                      padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
                      ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpeGR"]
                    if kernel["UseInstOffsetForGRO"]:
                      # buffer_load only support 12 bit instruction offset
                      # we have to increase m0 if offset is larger thant 12 bits
                      if instOffset >= self.buff_load_inst_offset_max:
                        inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                        module.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=inc, comment="Move LDS write address to next base" ))
                        instOffset -= inc
                    elif directToLdsLoads != 0 and ldsInc > 0:
                        if tP["nrc"] > 1:
                          # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                          divisorName = tP["lvc"]
                          divisor = kernel[divisorName]
                          # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                          wSize = max(self.states.kernel["WavefrontSize"], divisor)
                          lscaOffset = para * wSize * tP["bpeGR"] * tP["glvw"]
                          ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                          ldsInc = ldsOffset - prevLdsOffset
                          prevLdsOffset = ldsOffset

                        if (doTailOpt == 0) or (doTailOpt == 2 and behavior == "LOAD") or\
                           (doTailOpt == 1 and i == idx and ((r+1) % numElementsPer4Bytes != 0) and r != 0):
                          module.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=ldsInc, comment="Move LDS write address to next line" ))
                    destVgpr=0
                    self.vgprs.globalReadRegisters[tc].append(0)
                  else:
                    g2lIdxM = i * max(loadWidth * tP["bpeRatio"], 1)
                    destVgpr="G2L%s+%u+%u"%(tc, g2lIdx + tP["shiftGR"] if not tP["isM"] else g2lIdxM, regIdx+eccOffset)
                    self.vgprs.globalReadRegisters[tc].append( (g2lIdx + tP["shiftGR"] if not tP["isM"] else g2lIdxM) + regIdx+eccOffset)

                  offset = r * tP["bpeGR"] + instOffset
                  comment = "load one buffer value"

                  if (dataType.isHalf() or dataType.isBFloat16()) and not tP["isM"]:
                    if numElementsPerLoad==2:
                      # Pack two FP16 values into a single load dword x2
                      r += 1 # skip next element since we loaded 2X here
                      comment = "load packed 2X half buffer value"
                    elif not isLds:
                      hi16=loopCnt%2 if tP["glvw"]==1 else r%2
                      comment="load one buffer value"

                  if ((dataType.isInt8() or dataType.is8bitFloat()) \
                               and not tP["isM"]) or (tP["isM"] and destVgprHi != None):
                    # TODO-Int8, Check this:
                    #if numElementsPerLoad==2:
                      # Pack two FP16 values into a single load dword x2
                    #  r += 1 # skip next element since we loaded 2X here
                    #  comment = "load packed 2X half buffer value"
                    if not isLds:
                      hi8  = (loopCnt%4) %2 if tP["glvw"]==1 else (r%4) %2
                      hi16 = False if tP["glvw"]==1 else (r%4)//2
                      comment="load one buffer value"

                  useBuffer = not isTr
                  bpl = numElementsPerLoad*(tP["bpeGR"] if not tP["isM"] else tP["bpe"]) # bytesPerLoad

                  # if hi8=1 or hi16=1 (component 1,2,3 for int8) or (component 1 for half), use the temp destVgprHi
                  # but only when hi16=1 we use the _d16_hi version instruction, see the below visualized int8 comment
                  if doTailOpt == 1:
                    loadVgpr = destVgprHi
                  else:
                    loadVgpr = destVgprHi if ((hi16 or hi8) and destVgprHi != None) else destVgpr
                  self.vgprs.globalReadRegisters[tc][-1] = destVgprHi if ((hi16 or hi8) and destVgprHi != None) else self.vgprs.globalReadRegisters[tc][-1]
                  if (kernel["ProblemType"]["DataType%s"%tcDataType].isInt8() or kernel["ProblemType"]["DataType%s"%tcDataType].is8bitFloat() or tP["isM"]) \
                    and (not self.states.archCaps["HasEccHalf"]) and useBuffer:
                    module.add(VMovB32(dst=vgpr(loadVgpr), src=0, comment="set to zero to avoid unexpected value"))

                  if isTr:
                    module.add(VMinI32(dst=vgpr(offsetVgpr), src0=vgpr(maxGroVgpr), src1=vgpr(offsetVgpr), comment="truncated load: clamp GRO to legal range"))

                  if doTailOpt == 1:
                    if behavior == "LOAD" and i == idx:
                      if (numElementsPerLoad == 2 and r % numElementsPer4Bytes != 0) or \
                         (numElementsPerLoad != 2 and ((r + 1) % numElementsPer4Bytes != 0)):
                        module.addComment0("g2l=%u, load component %u"%(g2lIdx, r))
                        module.add(self.chooseGlobalRead(useBuffer, \
                                  bpl, destVgpr=loadVgpr, \
                                  addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 2 if isTr else 4), \
                                  soffset=soffset, offset=offset, \
                                  glc=isGlc, slc=isSlc, nt=isNT, lds=isLds, \
                                  tr=isTr, hi16=hi16, \
                                  comment=comment))
                        tmpVgprIdx += 1

                        if (numElementsPerLoad == 2 and r == (numLoadVectorComp - 1)) or \
                           (numElementsPerLoad != 2 and (r + 1) == (numLoadVectorComp - 1)):
                          module.add(SBranch(labelName=jumpLabel.getLabelName(), comment=""))
                  else:
                    if (doTailOpt == 0) or (doTailOpt == 2 and behavior == "LOAD" and r >= rStart and r < rEnd):

                      module.add(self.chooseGlobalRead(useBuffer, \
                                bpl, destVgpr=loadVgpr, \
                                addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 2 if isTr else 4), \
                                soffset=soffset, offset=offset, \
                                glc=isGlc, slc=isSlc, nt=isNT, lds=isLds, \
                                tr=isTr, hi16=hi16, \
                                comment=comment))
                      loadCnt = loadCnt + 1
                  if unrollMirrorWithSoffset:
                    codeMod = Module("mirrorIdx%u"%loopCnt)
                    codeMod.add(VAddU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset_prev, comment="mirror unroll: restore GRO=GRO+SGRO"))
                    module.add(codeMod)

                  if isLds and kernel["UseInstOffsetForGRO"]:
                    instOffsetInc += ldsInc

                else: # Not buffer load, ie 'flat' load
                  # mask if current address if in bounds
                  module.add(VCmpXLtU64(dst=VCC(), \
                      src0=vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), \
                      src1=vgpr(maxAddrVgpr,2), \
                      comment="addr < maxAddr"))
                  hi16=(kernel["ProblemType"]["DataType%s"%tcDataType].isHalf() or kernel["ProblemType"]["DataType%s"%tcDataType].isBFloat16()) and r%2==1
                  destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx)
                  # load one element from address
                  module.add(self.chooseGlobalRead(False, \
                            tP["bpeGR"], destVgpr=destVgprHi if (hi16 and destVgprHi != None) else destVgpr, \
                            addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                            soffset=0, offset=0, \
                            glc=isGlc, slc=isSlc, nt=isNT, lds=isLds, \
                            hi16=hi16, \
                            comment="load one flat value"))

                  # restore full exec mask
                  SOrSaveExecBX = SOrSaveExecB32 if self.states.kernel["WavefrontSize"] == 32 else SOrSaveExecB64
                  module.add(SOrSaveExecBX(dst=VCC(), src=sgpr(fullExec,self.states.laneSGPRCount), comment="all threads active"))

                  # increment address by 1 element (BPE)
                  module.add(VAddCOU32(
                      dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                      dst1=VCC(), \
                      src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                      src1=vgpr(bpeVgpr), comment="gra += 1 (lower)"))
                  module.add(VAddCCOU32(
                      dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                      dst1=VCC(), \
                      src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                      src1=vgpr(zeroVgpr), \
                      src2=VCC(), comment="gra += 1 (upper)"))

                # int8 byte:
                # |--------|--------|--------|---V0---|, r = 0, hi8=0, hi16=0, load d16
                # |--------|--------|--------|---V1---|, r = 1, hi8=1, hi16=0, load d16
                # |--------|---V2---|--------|--------|, r = 2, hi8=0, hi16=1, load d16_hi
                # |--------|---V3---|--------|--------|, r = 3, hi8=1, hi16=1, load d16_hi
                # V1, V3 -> shift left 8 bits, or 4 regs (pack)
                # DestV0|=(V1 << 8), DestV0|= V2, DestV0|=(V3 << 8)
                # Int8 (byte)
                if doTailOpt == 2 and r >= rStart and r < rEnd:
                  loadNum = loadNum - 1
                if dataIsByte and (destVgprHi != None):
                  if doTailOpt == 1:
                    if behavior == "MERGE" and i == idx and ((r + 1) % numElementsPer4Bytes != 0):
                      # hi8  -> r = 1,3
                      # hi16 -> r = 2,3
                      module.add(SWaitCnt(vlcnt=0, comment=""))
                      if hi8:
                        module.add(VLShiftLeftB32(dst=vgpr(destVgprHi), shiftHex=hex(0x8), src=vgpr(destVgprHi), comment="shift left to higher 8 bits"))
                      module.add(VOrB32(dst=vgpr(destVgpr), src0=vgpr(destVgpr), src1=vgpr(destVgprHi), comment="pack a sub 8-bit with dest"))
                      tmpVgprIdx += 1
                      if (r + 1) == (numLoadVectorComp - 1):
                        module.add(SBranch(labelName=jumpLabel.getLabelName(), comment=""))
                  else:
                    if (doTailOpt == 0) or (doTailOpt == 2 and behavior == "MERGE" and r >= rStart and r < rEnd):
                      # hi8  -> r = 1,3
                      # hi16 -> r = 2,3
                      if hi8 or hi16:
                        # r = 1,2,3, vlcnt needed for one packing
                        if doTailOpt == 0:
                          packInt8Code.add(SWaitCnt(vlcnt=(int8TempVgpr-r), comment=""))
                        else:
                          packInt8Code.add(SWaitCnt(vlcnt=(loadNum), comment=""))
                      if hi8:
                        # r = 1,3,   shift needed
                        packInt8Code.add(VLShiftLeftB32(dst=vgpr(destVgprHi), shiftHex=hex(0x8), src=vgpr(destVgprHi), comment="shift left to higher 8 bits"))
                      if hi8 or hi16:
                        # r = 1,2,3, packing
                        packInt8Code.add(VOrB32(dst=vgpr(destVgpr), src0=vgpr(destVgpr), src1=vgpr(destVgprHi), comment="pack a sub 8-bit with dest"))
                      destVgprHi += 1

                # Half
                elif destVgprHi != None:
                  if doTailOpt == 1:
                    if behavior == "MERGE" and i == idx:
                      if (numElementsPerLoad == 2 and r % numElementsPer4Bytes != 0) or \
                         (numElementsPerLoad != 2 and ((r + 1) % numElementsPer4Bytes != 0)):
                        module.add(SWaitCnt(vlcnt=0, comment=""))
                        # 8b checks are needed for mixed 16b/8b precisions
                        if kernel["ProblemType"]["DataType%s"%tcDataType].is8bitFloat():
                          module.add(VLShiftRightB32(dst=vgpr(destVgprHi), shiftHex=hex(8), src=vgpr(destVgprHi), comment="shift right to lower 8 bits"))
                        module.add(VOrB32(dst=vgpr(destVgpr), src0=vgpr(destVgpr), src1=vgpr(destVgprHi), comment="HasEccHalf: pack"))
                        if kernel["ProblemType"]["DataType%s"%tcDataType].is8bitFloat() and (g2lIdx % 2 == 1):
                          module.add(VLShiftLeftB32(dst=vgpr(destVgpr), shiftHex=hex(16), src=vgpr(destVgpr), comment="shift left to higher 16 bits"))
                        tmpVgprIdx += 1
                        if (numElementsPerLoad == 2 and r == (numLoadVectorComp - 1)) or\
                           (numElementsPerLoad != 2 and (r + 1) == (numLoadVectorComp - 1)):
                          module.add(SBranch(labelName=jumpLabel.getLabelName(), comment=""))
                  else:
                    if r % 2 == 1:
                      # 8b checks are needed for mixed 16b/8b precisions
                      if kernel["ProblemType"]["DataType%s"%tcDataType].is8bitFloat():
                        packInst.add(VLShiftRightB32(dst=vgpr(destVgprHi), shiftHex=hex(8), src=vgpr(destVgprHi), comment="shift right to lower 8 bits"))
                      packInst.add(VOrB32(dst=vgpr(destVgpr), src0=vgpr(destVgpr), src1=vgpr(destVgprHi), comment="HasEccHalf: pack"))
                      if kernel["ProblemType"]["DataType%s"%tcDataType].is8bitFloat() and (g2lIdx % 2 == 1):
                        packInst.add(VLShiftLeftB32(dst=vgpr(destVgpr), shiftHex=hex(16), src=vgpr(destVgpr), comment="shift left to higher 16 bits"))
                      vlcntVal = 0 if doTailOpt == 0 else loadNum
                      if (doTailOpt == 0 and len(tmpVgprPool) == maxNumTempToUse) or (doTailOpt == 2 and behavior == "MERGE" and r >= rStart and r < rEnd):
                        module.add(SWaitCnt(vlcnt=(vlcntVal), comment=""))
                        module.add(packInst)
                        packInst = Module()
                else:
                  if doTailOpt == 1 and i == idx and behavior == "MERGE" and\
                     ((numElementsPerLoad == 2 and r == (numLoadVectorComp - 1)) or\
                     (numElementsPerLoad != 2 and (r + 1) == (numLoadVectorComp - 1))):
                    module.add(SBranch(labelName=jumpLabel.getLabelName(), comment=""))
                # For half (bf16). Note: for int8, we will checkin after loading all components
                if (destVgprHi != None) and (not dataIsByte):
                  if len(tmpVgprPool) == maxNumTempToUse and doTailOpt == 0:
                    for v in tmpVgprPool:
                      self.vgprPool.checkIn(v)
                    tmpVgprPool = []
                  destVgprHi = None

                r += 1 # next component (for half, byte)

              # end R loop

              instOffset += instOffsetInc # add increment value for instOffset. Need to apply after r loop
              # increment once per r loop (at the end)
              directToLdsLoads+=1

              # for int8:
              # we do the 3 packs, and checking the 3 extra vgprs after loading all components
              if dataIsByte and int8TempVgpr and not isLds:
                assert packInt8Code != None and destVgprHi != None
                module.add(packInt8Code)
                if doTailOpt == 0:
                  self.vgprPool.checkIn(destVgprHi - int8TempVgpr)
                destVgprHi = None

      # Handles doTailOpt==0 packing after all previous buffer loads have completed
      if is16b and doTailOpt == 0 and tmpVgprPool:
        module.add(SWaitCnt(vlcnt=0, comment="Wait for previous GR to finish"))
        module.add(packInst)
        for v in tmpVgprPool:
          self.vgprPool.checkIn(v)
        tmpVgprPool = []

      if kernel["ProblemType"]["Sparse"]:
        if kernel["DirectToVgprSparseMetadata"]:
          miWaveTile = kernel["MIWaveTileB"] if (tP["is_sparse"] and tP["isB"]) else kernel["MIWaveTileA"] if (tP["is_sparse"] and tP["isA"]) else 0
          for wtIdx in range(0, miWaveTile):
            offsetVgpr= "GlobalReadOffsetMetadata+%u"%wtIdx
            for unrollIdx in range(0, kernel["LoopIters"]):
              bpl = kernel["MIInputPerThread"]//8 # bytes per load: 1 byte for fp16,bf16, 2 bytes for int8
              constOffset = unrollIdx * kernel["MatrixInstK"] // 8
              for byteIdx in range(0, bpl):
                constOffset += byteIdx
                if byteIdx == 0:
                  # For PGR=2 read metadata into the 3rd blk of vpgrValuMetadata
                  if kernel["PrefetchGlobalRead"] == 2:
                    offsetBlk = self.states.m.numVgprValuPerBlock * 2
                  else:
                    offsetBlk = 0
                  destVgprLow="ValuMetadata+%u+%u"%(offsetBlk, (wtIdx*kernel["LoopIters"]+unrollIdx))
                  destVgpr=destVgprLow
                else:
                  destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi')
                  destVgpr=destVgprHi
                module.add(self.chooseGlobalRead(kernel["BufferLoad"], \
                          1, \
                          destVgpr=destVgpr, \
                          addr0=vgpr(offsetVgpr), addr1=sgpr("SrdMetadata",4), \
                          soffset=0, offset=constOffset, \
                          glc=isGlc, slc=isSlc, nt=isNT, lds=isLds, \
                          hi16=0, \
                          comment="G -> Reg ValuMetadata"))
              if bpl == 2: #pack 2bytes
                module.add(SWaitCnt(vlcnt=0))
                module.add(VLShiftLeftB32(dst=vgpr(destVgprHi), shiftHex="0x8", src=vgpr(destVgprHi), comment="shift left to higher 8 bits"))
                module.add(VOrB32(dst=vgpr(destVgprLow), src0=vgpr(destVgprLow), src1=vgpr(destVgprHi), comment="pack 2 bytes"))
                self.vgprPool.checkIn(destVgprHi)
                destVgprHi = None

      return loadCnt, self.vgprs.globalReadRegisters[tc], directToLdsLoads

    loadCnt, vgprList, directToLdsLoads = globalReadGuardKBody(tP, optParams)

    if doTailOpt == 0 or \
       (doTailOpt == 2 and optParams.finalLoop == 1 and optParams.behavior == "MERGE"):
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        if tP["is_sparse"]:
            globalReadGuardKBody(tP["tpsMetadata"])

      if self.db["ConservativeWaitCnt"] & 0x1:
          module.add(SBarrier(comment="debug"))
          module.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=0, comment=""))
          module.add(SBarrier(comment="debug"))

      # TODO - can remove one of these m0 restores if A and B both TLU
      if kernel["DirectToLds%s"%tP["tensorChar"]]:
        module.add(SMovB32(dst=mgpr(0), src=hex(kernel["LdsNumBytes"]), \
            comment="Restore LDS clamp at %u bytes HERE"%(kernel["LdsNumBytes"])))

      if isTr:
        self.vgprPool.checkIn(maxGroVgpr)
      elif not kernel["BufferLoad"]:
        self.vgprPool.checkIn(maxAddrVgpr)
        self.vgprPool.checkIn(bpeVgpr)
        self.vgprPool.checkIn(zeroVgpr)

    if doTailOpt == 2:
      return module, loadCnt, vgprList, directToLdsLoads
    else:
      return module

  ##############################################################################
  # Redorder the wait instructions to reduce overall waiting time in tail loop.
  # It needs more vgprs to store data which is needed to be merged into a dword.
  ##############################################################################
  def doTailLoopOpt(self, kernel, tP):
    module = Module("doTailLoop")
    tc = tP["tensorChar"]
    orinrp = tP["nrp"]
    orinrpv = tP["nrpv"]
    orinrc = tP["nrc"]
    orinrcv_div_nrcvpi = tP["nrcv"] // tP["nrcvpi"]
    loadWidth = tP["globalReadInstruction"].totalWidth
    oriNumLoadVectorComp = (int(loadWidth*self.states.bpr//tP["bpeGR"]))
    numElementsPerLoad = 1
    glvwWorkaround = 8 * kernel["ProblemType"]["DataType"].numRegisters()
    dataType = kernel["ProblemType"]["DataType"] if tP["glvw"] < glvwWorkaround \
               else kernel["ProblemType"]["DataType%s"%tc]

    if dataType.isHalf() or dataType.isBFloat16():
      if tP["glvw"] > 1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
        # Pack two FP16 values into a single load dword x2
        numElementsPerLoad = 2
        oriNumLoadVectorComp = oriNumLoadVectorComp // 2
    totalVgprNum = (tP["nrp"] * tP["nrpv"] * tP["nrc"] * (tP["nrcv"] // tP["nrcvpi"])) * (oriNumLoadVectorComp)
    if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] \
       and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
      totalVgprNum += 1

    numLoadVectorComp = oriNumLoadVectorComp
    nrp = orinrp
    nrpv = orinrpv
    nrc = orinrc
    nrcv_div_nrcvpi = orinrcv_div_nrcvpi
    currentSize = self.vgprPool.size()
    VgprSlot = []
    VgprSlotBk = []

    maxVgpr = 50
    while (len(VgprSlot) < totalVgprNum and len(VgprSlot) <= maxVgpr):
      tempVgpr = self.vgprPool.checkOut(1,"")
      if tempVgpr >= currentSize:
        self.vgprPool.checkIn(tempVgpr)
        break
      if kernel["ProblemType"]["Gradient"] and kernel["ProblemType"]["UseBias"] \
         and (kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B"):
        if tempVgpr != self.states.bias.startVgprValu:
          VgprSlot.append(tempVgpr)
      else:
        VgprSlot.append(tempVgpr)
      VgprSlotBk.append(tempVgpr)

    loopNum = 1
    finalVgprNum = totalVgprNum
    while (finalVgprNum > len(VgprSlot) and totalVgprNum > 0):
      if nrp > 1:
        nrp -= 1
      elif nrpv > 1:
        nrpv -=  1
      elif nrc > 1:
        nrc -= 1
      elif nrcv_div_nrcvpi > 1:
        nrcv_div_nrcvpi -= 1
      else:
        numLoadVectorComp = numLoadVectorComp - 1
      finalVgprNum = (nrp * nrpv * nrc * nrcv_div_nrcvpi) * (numLoadVectorComp)

    nrpLoopNum = ceil(tP["nrp"] / nrp)
    nrpvLoopNum = ceil(tP["nrpv"] / nrpv)
    nrcLoopNum = ceil(tP["nrc"] / nrc)
    nrcv_div_nrcvpiLoopNum = ceil(tP["nrcv"] // tP["nrcvpi"] / nrcv_div_nrcvpi)
    loadVectorCompLoopNum = int(ceil(oriNumLoadVectorComp / numLoadVectorComp))
    totalLoopNum = nrpLoopNum * nrpvLoopNum * nrcLoopNum * nrcv_div_nrcvpiLoopNum * loadVectorCompLoopNum

    i = 0
    globalReadRegisters = []
    directToLdsLoads = 0
    finalLoop = 0

    self.optParamsLoad = TailOptParams()
    self.optParamsMerge = TailOptParams()
    for nrp_idx in range(nrpLoopNum):
      for nrpv_idx in range(nrpvLoopNum):
        for nrc_idx in range(nrcLoopNum):
          for nrcv_div_nrcvpi_idx in range(nrcv_div_nrcvpiLoopNum):
            for loadVectorComp_idx in range(loadVectorCompLoopNum):
              firstLoop = 0
              if numElementsPerLoad == 2:
                periodParam = [nrp_idx * nrp, min(orinrp, (nrp_idx + 1) * nrp), \
                               nrpv_idx * nrpv, min(orinrpv, (nrpv_idx + 1) * nrpv), \
                               nrc_idx * nrc, min(orinrc, (nrc_idx + 1) * nrc), \
                               nrcv_div_nrcvpi_idx * nrcv_div_nrcvpi, \
                               min(orinrcv_div_nrcvpi, (nrcv_div_nrcvpi_idx + 1) * nrcv_div_nrcvpi), \
                               loadVectorComp_idx * numLoadVectorComp * 2, \
                               min(oriNumLoadVectorComp * 2, (loadVectorComp_idx + 1) * numLoadVectorComp * 2)]
              else:
                periodParam = [nrp_idx * nrp, min(orinrp, (nrp_idx + 1) * nrp), \
                               nrpv_idx * nrpv, min(orinrpv, (nrpv_idx + 1) * nrpv), \
                               nrc_idx * nrc, min(orinrc, (nrc_idx + 1) * nrc), \
                               nrcv_div_nrcvpi_idx * nrcv_div_nrcvpi, \
                               min(orinrcv_div_nrcvpi, (nrcv_div_nrcvpi_idx + 1) * nrcv_div_nrcvpi), \
                               loadVectorComp_idx * numLoadVectorComp, \
                               min(oriNumLoadVectorComp, (loadVectorComp_idx + 1) * numLoadVectorComp)]
              if i == 0:
                firstLoop = 1

              self.optParamsLoad.behavior = "LOAD"
              self.optParamsLoad.vgprSlot = VgprSlot
              self.optParamsLoad.periodParam = periodParam
              self.optParamsLoad.directToLdsLoads = directToLdsLoads
              self.optParamsLoad.firstLoop = firstLoop
              self.optParamsLoad.finalLoop = finalLoop
              imod, loadCnt, vgprList, directToLdsLoads = self.globalReadDo(kernel, 2, tP, -1, 0, 2, self.optParamsLoad)
              module.add(imod)

              globalReadRegisters = globalReadRegisters + vgprList

              if i == totalLoopNum - 1:
                finalLoop = 1

              self.optParamsMerge.behavior = "MERGE"
              self.optParamsMerge.vgprSlot = VgprSlot
              self.optParamsMerge.periodParam = periodParam
              self.optParamsMerge.loadNum = loadCnt
              self.optParamsMerge.directToLdsLoads = directToLdsLoads
              self.optParamsMerge.firstLoop = firstLoop
              self.optParamsMerge.finalLoop = finalLoop
              imod, loadCnt, vgprList, directToLdsLoads = self.globalReadDo(kernel, 2, tP, -1, 0, 2, self.optParamsMerge)
              module.add(imod)

              i += 1

    self.vgprs.globalReadRegisters[tc] = globalReadRegisters
    while VgprSlotBk:
      tempVgpr = VgprSlotBk.pop(0)
      self.vgprPool.checkIn(tempVgpr)

    return module

  ##############################################################################
  # DirectToLds M0 update: Do It A/B
  ##############################################################################
  def directToLdsM0Update(self, kernel, mode, tP, skipWait = False):
    tc = tP["tensorChar"]
    imod = Module("directToLdsM0Update%s_%u"%(tc,mode))
    DtldsModule = imod.add(Module("dtls_offset%s"%tP["tensorChar"]))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod
    if kernel["DirectToLds%s"%tc]:
      # DirectToLds only enabled for TLU=1 cases, where the registers are directly copied into LDS
      # for cases both A&B are DTLS, updating m0 for each GlobalRead requires instruction schedule
      # along with global reads
      assert (kernel["LocalWriteUseSgpr%s"%tc])
      if kernel["ExpandPointerSwap"]:
        DtldsModule.add(SAddU32(dst=mgpr(0), src0=sgpr("LocalWriteAddr%s"%tc), \
                      src1=tP["localWriteSwapByteOffset"], comment="m0 <- LDS write address"))
      else:
        DtldsModule.add(SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddr%s"%tc), comment="m0 <- LDS write address"))

      # PrefetchGlobalRead=2 case, generate local read wait for DirectToLds
      if kernel["PrefetchGlobalRead"]==2 and not skipWait:
        # do not generate local read wait for PGR=2
        DtldsModule.addComment0("before DirectToLds load, ensure prior ds_reads have finished")
        DtldsModule.add(SWaitCnt(dscnt=0, comment=""))
        if not kernel["NoLdsWriteCode"]:
          waitStr = 0
          DtldsModule.add(SWaitCnt(vlcnt=waitStr, comment=""))
        DtldsModule.add(SBarrier())

    return imod

  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, mode, tP, unrollLoopIdx=-1, g2lBufIdx=0, \
                   doTailOpt = 0, optParams = None):
    tc = tP["tensorChar"]
    problemType = self.states.kernel["ProblemType"]
    imod = StructuredModule("globalReadDo%s_%u"%(tc,mode))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]:
      return imod

    # sizeK % LOCAL_DEPTHU
    guardK = (mode==2)

    loopIdx = self.states.unrollIdx # TODO - does this handle multiple summation indices?
    if kernel["SuppressNoLoadLoop"]:
      if mode==1 and tP["isA"]:
        imod.header.add(SCmpEQI32(
              src0=self.loopCounter(kernel, loopIdx), \
              src1=1, \
              comment="%s"%"is this the last iteration"))
        imod.header.add(SCMovB32(
              dst=sgpr("SrdA+2"), src=0,
              comment="Set limit to 0 for last iteration"))
        imod.header.add(SCMovB32(
              dst=sgpr("SrdB+2"), src=0,
              comment="Set limit to 0 for last iteration"))

    # set the first tc for below wait code for DirectToLds
    tc1st = 'A'
    # if DirectToVgpr is enabled and swapGlobalRead is true, change the first to B
    if self.isSwapGlobalReadOrderForDtvOrDtl(kernel, prefetch1=(mode==0)):
      tc1st = 'B'
    if tc == tc1st and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) \
                   and not kernel["PrefetchGlobalRead"]==2:
      if doTailOpt == 0 or \
        (doTailOpt == 2 and optParams.behavior == "LOAD" and optParams.firstLoop == 1):
        # generate local read wait for DirectToLds except for
        # PrefetchGlobalRead=2 (for PGR=2, generate wait after m0 value setting)
        imod.header.addComment0("before DirectToLds load, ensure prior ds_reads have finished")
        if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and not guardK and mode != 3:
          # no need to generate sync here if DirectToVgpr is enabled and not tail loop
          imod.header.add(SWaitCnt(dscnt=0, comment="wait for LDS read/write"))
        else:
          imod.header.add(self._syncThreads(kernel))

    if guardK:
      if doTailOpt == 0:
        imod.middle.add(self.globalReadGuardK(kernel, tP))
        return imod
      else:
        module, loadCnt, vgprList, directToLdsLoads = \
          self.globalReadGuardK(kernel, tP, doTailOpt, optParams)

        imod.middle.add(module)
        return imod, loadCnt, vgprList, directToLdsLoads

    # else not-guardK below:
    record = [False, False]  # record global read gpr info for A/B
    record[0] = True if tc == "A" else False
    record[1] = True if tc == "B" else False
    def globalReadBody(tP):
      tc = tP["tensorChar"]
      self.vgprs.globalReadRegisters[tc] = []
      graIdx = 0
      g2lIdx = 0
      loadWidth = tP["globalReadInstruction"].totalWidth # load width in elements?
      bpe = tP["bpeGR"] if not tP["isM"] else tP["bpe"]
      bpl = bpe * tP["glvw"]  # bytes per load
      isGlc = bool(tP["NonTemporal"] & 0x1)
      isSlc = bool(tP["NonTemporal"] & 0x2)
      isNT  = bool(tP["NonTemporal"] & 0x4)
      isLds = True if kernel["DirectToLds%s"%tc] else False
      isTr = (tc == "A" or tc == "B") and kernel["enableGLTr%s"%tc]

      directToLdsLoads = 0
      instOffset       = 0
      prevLdsOffset    = 0

      if g2lBufIdx >= 1:
        # G2L vgpr base string. DirectToVgpr or swapAB case. Need to toggle destination vreg set
        destVgprPrefix = "G2L%s%u"%(tc, g2lBufIdx + 1)
      else:
        destVgprPrefix = "G2L%s"%(tc)

      loopCnt = -1
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
              if tc != "Metadata" and kernel["DirectToVgpr%s"%tc] and kernel["reorderGRInstForDTV%s"%tc]:
                i = perp + tP["nrp"] * (sPerp + tP["nrpv"] * (para + sPara * tP["nrc"]))
              else:
                i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
              loopCnt += 1
              graIdx = i * self.states.rpgo if kernel["BufferLoad"] else i * self.states.rpga
              g2lIdx = i * loadWidth * tP["bpeRatio"]
              if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc] and kernel["ConvertAfterDS"]:
                # DTV + ConvertAfterDS case, if bpe > bpeGR, we need to shift g2lIdx for conversion
                if tP["bpe"] > tP["bpeGR"]:
                  g2lIdx *= tP["bpe"] // tP["bpeGR"]
              # Each load may contains a small bundle of instructions, package them together in loadModule:
              loadModule = Module("load%u"%loopCnt)
              imod.middle.add(loadModule)

              if (self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]) and not tP["isM"]:
                numVgprG2L = self.states.a.numVgprG2L if tc == 'A' else self.states.b.numVgprG2L if tc =='B' else self.states.m.numVgprG2L
                eccBpe = tP["bpeDS"] if kernel["ConvertAfterDS"] else max(tP["bpeGR"], tP["bpe"])
                eccOffset = _getEccOffset(loadWidth, bpr=self.states.bpr, bpe=eccBpe, \
                  glvw=tP["glvw"], idx=i, numVgprG2L=numVgprG2L)
              else:
                eccOffset = 0

              if kernel["BufferLoad"]:
                if kernel["_UseSgprForGRO"]:
                  offsetVgpr= "GlobalReadOffset%s+0"%(tc)
                else:
                  offsetVgpr= "GlobalReadOffset%s+%u"%(tc, graIdx)

                # vgpr for GRO
                if not kernel["_UseSgprForGRO"]:
                  soffset = "0"
                # instruction offset with Sgpr for GRO
                elif kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx))
                # Sgpr for GRO
                else:
                  soffset = "0" if graIdx == 0 else sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))
                unrollMirrorWithSoffset = kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx] in problemType["MirrorDims%s"%tc] and soffset != "0"
                # ScalarGlobalReadOffset should be negative value with unroll mirroring.
                # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
                if unrollMirrorWithSoffset:
                  codeMod = Module("mirrorIdx%u"%loopCnt)
                  codeMod.add(VSubU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset, comment="mirror unroll: GRO=GRO-SGRO, soffset=0"))
                  loadModule.add(codeMod)
                  soffset_prev = soffset
                  soffset = "0"

                if kernel["DirectToLds%s"%tc]:
                  # use bpe with GlobalReadVectorWidth
                  ldsInc = (self.states.kernel["WavefrontSize"] * kernel["GlobalReadVectorWidth%c"%tc] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"] * kernel["GlobalReadVectorWidth%c"%tc]) * tP["bpeGR"]
                  if kernel["LdsBlockSizePerPad%s"%tc] != 0:
                    ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeGR"]
                  else:
                    padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
                    ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpeGR"]

                  if kernel["UseInstOffsetForGRO"]:
                    # buffer_load only support 12 bit instruction offset
                    # we have to increase m0 if offset is larger thant 12 bits
                    if instOffset >= self.buff_load_inst_offset_max:
                      inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                      loadModule.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=inc, comment="Move LDS write address to next base" ))
                      instOffset -= inc
                  elif directToLdsLoads != 0:
                    # m0 offset conversion (only for UseInstOffsetForGRO == 0)
                    # in tP["glvw"] == 1 and tP["nrc"] > 1 case, only m0 offset conversion is necessary. row and column index conversion is not necessary.
                    if tP["nrc"] > 1:
                      # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                      divisorName = tP["lvc"]
                      divisor = kernel[divisorName]
                      # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                      wSize = max(self.states.kernel["WavefrontSize"], divisor)
                      lscaOffset = para * wSize * tP["bpeGR"] * tP["glvw"]
                      ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                      ldsInc = ldsOffset - prevLdsOffset
                      prevLdsOffset = ldsOffset
                    loadModule.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=ldsInc, comment="Move LDS write address to next line" ))
                  directToLdsLoads+=1
                  destVgpr=0
                  self.vgprs.globalReadRegisters[tc].append(0)
                else:
                  g2lIdxM = i * max(loadWidth * tP["bpeRatio"], 1)
                  destVgpr = destVgprPrefix + "+%u"%((g2lIdx+eccOffset+tP["shiftGR"]) if not tP["isM"] else g2lIdxM)
                  self.vgprs.globalReadRegisters[tc].append(g2lIdx+eccOffset+tP["shiftGR"] if not tP["isM"] else g2lIdxM)
                  if tP["isM"]:
                    assert(graIdx <= self.states.m.numVgprG2LAllocated)

                # TODO: is it possible to load only hi16 when no in tail? (need to check INT8 too)
                datatype = kernel["ProblemType"]["DataType%s"%tc] if kernel["ConvertAfterDS"] else kernel["ProblemType"]["DataType"]
                isHigh16Bits = (datatype.isHalf() or datatype.isBFloat16()) and loopCnt%2==1 if not tP["isM"] else False
                if tc == "A" and record[0] == True:
                  self.globalread_gpr_record.a.addrVgpr.append(offsetVgpr)
                  self.globalread_gpr_record.a.offset.append(soffset)
                elif tc == "B" and record[1] == True:
                  self.globalread_gpr_record.b.addrVgpr.append(offsetVgpr)
                  self.globalread_gpr_record.b.offset.append(soffset)

                useBuffer = not isTr
                loadModule.add( self.chooseGlobalRead(useBuffer, \
                          bpl, destVgpr=destVgpr, \
                          addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 2 if isTr else 4), \
                          soffset=soffset, offset=instOffset, \
                          glc=isGlc, slc=isSlc, nt=isNT, lds=isLds, \
                          tr=isTr, hi16=isHigh16Bits , \
                          comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp)))

                if unrollMirrorWithSoffset:
                  codeMod = Module("mirrorIdx%u"%loopCnt)
                  codeMod.add(VAddU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset_prev, comment="mirror unroll: restore GRO=GRO+SGRO"))
                  loadModule.add(codeMod)

                if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                    instOffset += ldsInc

                #print "IM=", type(imod.instList[-1]), imod.instList[-1],
              else: # not buffer load
                destVgpr = destVgprPrefix + "+%u"%(g2lIdx + tP["shiftGR"])
                loadModule.add( self.chooseGlobalRead(False, \
                          bpl, \
                          destVgpr=destVgpr, \
                          addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                          soffset=0, offset=0, \
                          glc=isGlc, slc=isSlc, nt=isNT, lds=isLds, \
                          hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                          comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp )))

      if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"]:
        if tP["is_sparse"]:
          if kernel["PrefetchGlobalRead"] == 1 and unrollLoopIdx % 2 == 0:
            offsetBlk = self.states.m.numVgprValuPerBlock
          elif kernel["PrefetchGlobalRead"] == 2:
            offsetBlk = self.states.m.numVgprValuPerBlock * 2
          else:
            offsetBlk = 0
          miWaveTile = kernel["MIWaveTileA"] if tP["isA"] else kernel["MIWaveTileB"]
          for wtIdx in range(0, miWaveTile):
            offsetVgpr= "GlobalReadOffsetMetadata+%u"%wtIdx
            for unrollIdx in range(0, kernel["LoopIters"]):
              bpl = kernel["MIInputPerThread"]//8 # bytes per load: 1 byte for fp16,bf16, 2 bytes for int8
              constOffset = unrollIdx * kernel["MatrixInstK"] // 8
              codeMod = Module("load metadata%u"%loopCnt)
              imod.middle.add(codeMod)
              codeMod.add( self.chooseGlobalRead(kernel["BufferLoad"], \
                        bpl, \
                        destVgpr="ValuMetadata+%u+%u"%(offsetBlk, (wtIdx*kernel["LoopIters"]+unrollIdx)), \
                        addr0=vgpr(offsetVgpr), addr1=sgpr("SrdMetadata",4), \
                        soffset=0, offset=constOffset, \
                        glc=isGlc, slc=isSlc, nt=isNT, lds=isLds, \
                        hi16=0, \
                        comment="G -> Reg ValuMetadata"))
    if tc == "A" and record[0] == True:
      self.globalread_gpr_record.a.addrVgpr = []
      self.globalread_gpr_record.a.offset = []
    elif tc == "B" and record[1] == True:
      self.globalread_gpr_record.b.addrVgpr = []
      self.globalread_gpr_record.b.offset = []

    globalReadBody(tP)

    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"] and tP["is_sparse"]:
        globalReadBody(tP["tpsMetadata"])

    if self.db["ConservativeWaitCnt"] & 0x1:
        imod.footer.add(SBarrier(comment="debug"))
        imod.footer.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=0, comment="conservative wait"))
        imod.footer.add(SBarrier(comment="debug"))
        #module.add(self.getCmpAssert(self.asmAssert.lt, vgpr("Serial"), 64)) # examine second wavefront

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]] and not (mode == 1 and kernel["PrefetchGlobalRead"]==2):
      dst = mgpr(0)
      src = hex(kernel["LdsNumBytes"])
      comment = "Restore LDS clamp at %u bytes"%(kernel["LdsNumBytes"])
      # PGR=2 case, footer is located before global read. To avoid setting clamp before global read, store lds clamp code in middle
      if kernel["PrefetchGlobalRead"] == 2:
        imod.middle.add(SMovB32(dst=dst, src=src, comment=comment))
      else:
        imod.footer.add(SMovB32(dst=dst, src=src, comment=comment))

    return imod



  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    tc = tP["tensorChar"]
    if not self.do["LocalWrite%s"%tc]: return Module("localWriteSwapOffsets (No local write%s)"%tc)
    needSwap = False if kernel["1LDSBuffer"] else True
    doMetadataCheck = kernel["ProblemType"]["Sparse"] and \
                      ((kernel["ProblemType"]["Sparse"] ==2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]))
    needMetaSwap = needSwap and doMetadataCheck
    # swap not needed if DirectToVgpr is enabled (do not use DTVA/B for metaData. Change needSwap after setting needMetaSwap)
    if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]:
      needSwap = False
    if doMetadataCheck:
      if kernel["DirectToVgprSparseMetadata"]:
        needMetaSwap = (kernel["PrefetchGlobalRead"] == 2 and kernel["ExpandPointerSwap"])

    if not (needSwap or needMetaSwap): return Module("localWriteSwapOffsets (Empty)")
    module = Module("localWriteSwapOffsets")

    def localWriteSwapXOR(tc, src0Val, numLwa):
      if kernel["LocalWriteUseSgpr%s"%tc]:
        module.add(SXorB32(
          dst=sgpr("LocalWriteAddr%s"%tc), \
          src0=src0Val, \
          src1=sgpr("LocalWriteAddr%s"%tc), \
          comment="swap Red Blk SGPR"))
      else:
        module.add(VXorB32(
            dst=vgpr("LocalWriteAddr%s"%tc), \
            src0=src0Val, \
            src1=vgpr("LocalWriteAddr%s"%tc), \
            comment="swap Red Blk"))
        for i in range(1,numLwa):
          module.add(VAddU32(
            dst=vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
            src0=(i * self.states.regCaps["maxLDSConstOffset"]), \
            src1=vgpr("LocalWriteAddr%s"%tc), \
            comment="Final Offset Plus %uK"%((i * self.states.regCaps["maxLDSConstOffset"]) / 1024)))

    if needSwap:
      #fixme-iui  need to use wrapping increment for double or triple buffering:

      if internalPointerSwap and not kernel["StoreSwapAddr"]:
        tP["localWriteSwapByteOffset"] = 0 if tP["localWriteSwapByteOffset"] else kernel["LdsOffsetA_Blk"]
        module.addComment1("(EPS=1) local write swap internal offset -> %u" % tP["localWriteSwapByteOffset"])
      else:
        src0Val = None
        if kernel["StoreSwapAddr"]:
          if kernel["LocalWriteUseSgpr%s"%tc]:
            src0Val = sgpr("Swap%s"%tc)
          else:
            src0Val = vgpr("LocalWriteSwapAddr%s"%tc)
        else:
          # Using inlined constants
          src0Val = hex(kernel["LdsOffsetA_Blk"])
        numLwa = self.states.a.numVgprLocalWriteAddr if tP["isA"] else self.states.b.numVgprLocalWriteAddr
        localWriteSwapXOR(tc, src0Val, numLwa)

    # This used to control where to store the metadata
    if needMetaSwap:
      if kernel["DirectToVgprSparseMetadata"]:
        tP["metadataWriteSwapByteOffset"] = 0 if tP["metadataWriteSwapByteOffset"] else self.states.m.numVgprValuPerBlock
        module.addComment1("metadata write swap offset -> %u" % tP["metadataWriteSwapByteOffset"])
      else:
        tc = "Metadata"
        tPM = tP["tpsMetadata"]
        if internalPointerSwap and not kernel["StoreSwapAddr"]:
          tPM["localWriteSwapByteOffset"] = 0 if tPM["localWriteSwapByteOffset"] else kernel["LdsOffsetA_Blk"]
          module.addComment1("(EPS=1) local write swap internal offset -> %u" % tPM["localWriteSwapByteOffset"])
        else:
          if kernel["StoreSwapAddr"]:
            if kernel["LocalWriteUseSgpr%s"%tc]:
              src0Val = sgpr("Swap%s"%tc)
            else:
              src0Val = vgpr("LocalWriteSwapAddr%s"%tc)
          else:
            # Using inlined constants
            src0Val = hex(kernel["LdsOffsetA_Blk"])
          numLwa = self.states.m.numVgprLocalWriteAddr
          localWriteSwapXOR(tc, src0Val, numLwa)
    return module

  ##############################################################################
  # Local Write: Reset Offsets A/B
  # used for global-read + tail-loop to reset to writing in red
  ##############################################################################
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    tc = tP["tensorChar"]
    if not self.do["LocalWrite%s"%tc]: return Module("localWriteResetOffsets (no local write%s)"%tc)
    needReset = not (kernel["1LDSBuffer"])
    doMetadataCheck = kernel["ProblemType"]["Sparse"] and \
                      ((kernel["ProblemType"]["Sparse"] ==2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]))
    needMetaReset = needReset and doMetadataCheck
    # reset not needed if DirectToVgpr is enabled (do not use DTVA/B for metaData. Change needReset after setting needMetaReset)
    if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]:
      needReset = False
    if doMetadataCheck:
      if kernel["DirectToVgprSparseMetadata"]:
        needMetaReset = (kernel["PrefetchGlobalRead"] == 2 and kernel["ExpandPointerSwap"])
    if not (needReset or needMetaReset): # no local write code if DirectToVgpr is enabled
      return Module("localWriteResetOffsets (Empty)")
    module = Module("localWriteResetOffsets")
    if needReset:
      resetMask = hex(kernel["LdsOffsetA_Blk"]-1 | self.consts.ldsOOB)
      if internalPointerSwap or kernel["StoreSwapAddr"]:
        if internalPointerSwap:
          tP["localWriteSwapByteOffset"] = 0
        else:
          if kernel["LocalWriteUseSgpr%s"%tc]:
            tmpsgpr = self.sgprPool.checkOut(1)
            module.add(SXorB32(
              dst=sgpr(tmpsgpr), \
              src0=sgpr("Swap%s"%tc), \
              src1=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
              comment="Get other lds buffer offset value"))
            module.add(SMinU32(dst=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
                               src0=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
                               src1=sgpr(tmpsgpr), comment="Set LWA to first buffer offset" ))
            self.sgprPool.checkIn(tmpsgpr)
          else:
            tmpvgpr = self.vgprPool.checkOut(1)
            module.add(VXorB32(
              dst=vgpr(tmpvgpr), \
              src0=vgpr("LocalWriteSwapAddr%s"%tc), \
              src1=vgpr("LocalWriteAddr%s"%tc), \
              comment="Get other lds buffer offset"))
            module.add(VMinI32(dst=vgpr("LocalWriteAddr%s"%tc), \
                               src0=vgpr("LocalWriteAddr%s"%tc), \
                               src1=vgpr(tmpvgpr),
                               comment="Set LWA to first buffer offset"))
            self.vgprPool.checkIn(tmpvgpr)
      else:
        if kernel["LocalWriteUseSgpr%s"%tc]:
          module.add(SAndB32(
              dst=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
              src0=resetMask, \
              src1=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
              comment="reset to Red"))
        else:
          module.add(VAndB32(
            dst=vgpr("LocalWriteAddr%s"%tc), \
            src0=resetMask, \
            src1=vgpr("LocalWriteAddr%s"%tc), \
            comment="reset to Red"))
    if needMetaReset:
      if kernel["DirectToVgprSparseMetadata"]:
        tP["metadataWriteSwapByteOffset"] = 0
        module.addComment1("reset metadata write offset to %u" % tP["metadataWriteSwapByteOffset"])
      else:
        tPM = tP["tpsMetadata"]
        resetMask = hex(kernel["LdsOffsetA_Blk"]-1 | self.consts.ldsOOB)
        if internalPointerSwap:
          tPM["localWriteSwapByteOffset"] = 0
        elif kernel["StoreSwapAddr"]:
          if kernel["LocalWriteUseSgpr%s"%tPM["tensorChar"]]:
            tmpsgpr = self.sgprPool.checkOut(1)
            module.add(SXorB32(
              dst=sgpr(tmpsgpr), \
              src0=sgpr("Swap%s"%tPM["tensorChar"]), \
              src1=sgpr("LocalWriteAddr%s"%tPM["tensorChar"]), \
              comment="Get other lds buffer offset value"))
            module.add(SMinU32(dst=sgpr("LocalWriteAddr%s"%tPM["tensorChar"]), \
                               src0=sgpr("LocalWriteAddr%s"%tPM["tensorChar"]), \
                               src1=sgpr(tmpsgpr), comment="Set LWA to first buffer offset" ))
            self.sgprPool.checkIn(tmpsgpr)
          else:
            tmpvgpr = self.vgprPool.checkOut(1)
            module.add(VXorB32(
              dst=vgpr(tmpvgpr), \
              src0=vgpr("LocalWriteSwapAddr%s"%tPM["tensorChar"]), \
              src1=vgpr("LocalWriteAddr%s"%tPM["tensorChar"]), \
              comment="Get other lds buffer offset"))
            module.add(VMinI32(dst=vgpr("LocalWriteAddr%s"%tPM["tensorChar"]), \
                               src0=vgpr("LocalWriteAddr%s"%tPM["tensorChar"]), \
                               src1=vgpr(tmpvgpr),
                               comment="Set LWA to first buffer offset"))
            self.vgprPool.checkIn(tmpvgpr)
        else:
          module.add(VAndB32(
            dst=vgpr("LocalWriteAddr%s+%u"%(tPM["tensorChar"], 0)), \
            src0=resetMask, \
            src1=vgpr("LocalWriteAddr%s+%u"%(tPM["tensorChar"], 0)), \
            comment="reset to Red"))

    numLwa = 0
    if tP["isA"]:
      numLwa = self.states.a.numVgprLocalWriteAddr
    elif tP["isB"]:
      numLwa = self.states.b.numVgprLocalWriteAddr
    elif tP["isM"]:
      numLwa = self.states.m.numVgprLocalWriteAddr

    for i in range(1, numLwa):
      module.add(VAddU32(
        dst=vgpr("LocalWriteAddr%s+%u"%(tc, i)), \
        src0=(i * self.states.regCaps["maxLDSConstOffset"]), \
        src1=vgpr("LocalWriteAddr%s"%tc), \
        comment="Final Offset Plus %uK"%((i * self.states.regCaps["maxLDSConstOffset"]) / 1024)))
    return module

  ##############################################################################
  # Calculate offset to use for LDS write
  # Intro:
  #   Each WI has a 2D tile index (coal, perp).
  #     - Code above computes global mem address by scaling one dim by the
  #       lda and adding the other.
  #     - Here we compute a linear LDS offset by scaling one dim by the MT
  #       dim and adding the other.
  #   Result is we map a tile from global memory into LDS.  Consecutive LDS
  #   locations contain elements from different summation 'rows' - therefore
  #   loading a row of LDS will feed computations for different C tile indices.
  # Notes:
  #   Total load insts is nrc * nrp which load the macro-tile.
  #   Par and coalesced are ~synonyms referring to same dimension
  #   Either nrpv or nrvc must be 1 - can't have vectors in both dimensions.
  #     Thus either sPerp or sPara is 0.
  # Inputs:
  #   perp : index of the load in perp dimension (0...nrp)
  #   par  : index of the load in the para dim (0...nrc)
  #   sPerp : component index of the perp vector (0...nrpv)
  #   sPara : component index of the par vector (0...nrcv)
  # Outputs:
  #   offsetBytes : Offset in bytes for the _ds_store instruction
  #   i : i-th instruction
  #   comment : Comment with the text version of the formula
  #############################################################################
  def calculateLdsWriteOffset(self, perp, para, sPerp, sPara, kernel, tP):
    tc = tP["tensorChar"]
    mask = 0
    #print "tc ", tc, " perp ", perp, " para ", para, " sPerp ", sPerp, " sPara ", sPara
    lscaOffset = para * kernel[tP["lsc"]]
    perp_masked = perp
    perp_rem = 0
    lspaOffset = perp_masked * kernel[tP["lsp"]]
    rem = 0

    # Add component offset to interleave from different regs
    # and compute mysterious "i"
    assert(sPerp==0 or sPara==0)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset += sPerp & mask
      lscaOffset += sPara
      rem = (sPerp & ~mask)
      i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp_masked))
      #print "nrcv ", tP["nrcv"], " nrcvpi ", tP["nrcvpi"], " nrc ", tP["nrc"], " nrpv ", tP["nrpv"]
    else:
      lscaOffset += sPara
      lspaOffset += sPerp
      rem = 0
      i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))

    #if not tP["tlu"]:
    #  tmp = sPara
    #  sPara = sPerp
    #  sPerp = tmp
    # print("0lspaOffset", lspaOffset)
    # print("0lscaOffset", lscaOffset)

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    lds_stride = (kernel["_DepthU%s"%tc] + LdsPad) if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] \
            else (kernel[tP["mt"]] + LdsPad)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset *= lds_stride
      lspaOffset += rem + perp_rem
    else:
      lscaOffset *= lds_stride
      lscaOffset += rem

    # print("1lspaOffset", lspaOffset)
    # print("1lscaOffset", lscaOffset)
    #if tP["tlu"]:
    #  lspaOffset *= tP["glvw"]
    #  lscaOffset *= tP["glvw"]

    # print("2lspaOffset", lspaOffset)
    # print("2lscaOffset", lscaOffset)
    offsetElements = (lspaOffset + lscaOffset)
    # print("offsetElements", offsetElements)
    offsetBytes   = offsetElements*tP["bpeDS"]

    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      offsetBytes   = offsetBytes + (offsetBytes // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"]

    offsetBytes += tP["localWriteSwapByteOffset"]

    #print("offsetBytes", offsetBytes)
    #print "offset", offset

    comment = "lwo%s_%u_%u_%u_%u = (%s%d*%s)" \
        % (tP["tensorChar"], \
        para, sPara, perp, sPerp, \
        (("%u + "%sPara) if tP["wtc"] else ""), \
        para, tP["lsc"] )
    if not tP["tlu"]:
      comment += "*(MT%s+PAD)" % (tP["tileChar"])
    comment += " + (%d*%s)" % (perp, tP["lsp"])
    if tP["tlu"]:
      comment += "(*MT%s+PAD)" % (tP["tileChar"])
    comment += " = %u" % (offsetBytes)

    return (offsetBytes, i, comment)

  def recalcLocalWriteAddresses(self, kernel, tP):

    tc = tP["tensorChar"]

    module = Module("recalcLocalWriteAddresses")
    module.addComment1("recalculate LocalWriteAddr{}".format(tc))

    lwvw = getattr(self, "localWriteWidth{}".format(tc))
    newInstIdx = self.selectMemoryInstruction("LocalWrite", lwvw, \
        False, \
        tP["localWrite2Coalesced"], tP["localWrite2Perpendicular"],
        [tP["localWriteStrideTile"], tP["localWriteStrideUnroll"]] )
    tP["localWriteInstruction"] = self.memoryInstructions["LocalWrite"][newInstIdx]

    loopComponent = Component.PersistentLoop.find(self)
    module.add(loopComponent.recalcLocalWriteAddresses(self, kernel, tc))

    # local write tile assignments
    module.add(self.lwaTileAssignment(kernel, tP))
    # local write unroll assignments
    module.add(self.lwaUnrollAssignment(kernel, tP))
    # local write local write first offsets
    module.add(self.lwaFirstOffset(kernel, tP))

    # global read tile assignment
    module.add(self.graTileAssignment(kernel, tP))
    # global read tile offsets
    module.add(self.graTileOffsets(kernel, tP))
    # global read unroll offsets
    module.add(self.graUnrollOffsets(kernel, tP))
    # still needed for vgpr resource management
    # intentionally not emitting code
    self.graFinalOffsets(kernel, tP)

    if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"]:
      if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]):
        graMetadataFinalOffsets(kernel, tP)

    return module

  def recalcLocalReadAddressesAB(self, kernel, tPA, tPB):
    imod = Module()

    if self.states.inTailLoop:
      # it do 1 iteration each loop in tail loop, and is no use to wider local read next iteration.
      # In 1 block MI, it remap localReadAddr in order to let each thread wider local read continuous k
      # this decrease performance since it require more loop to handle continuous k in each thread.
      # reCalculating localread address because we disable wider local read in tail loop
      if not kernel["UseDotInstruction"] and ((self.states.numReadsIterCoalescedA > 1 or self.states.numReadsIterCoalescedB > 1)):
        loopComponent = Component.PersistentLoop.find(self)
        imod.add(loopComponent.recalcLocalReadAddressesAB(self, kernel))

        self.states.numReadsIterCoalescedA = 1
        self.states.numReadsIterCoalescedB = 1
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          self.states.numReadsIterCoalescedMetadata = 1
        tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]
        imod.add(self.lraTileAssignment(kernel, tPA, tPB))
        imod.add(self.lraFinalOffset(kernel, tPA))
        imod.add(self.lraDeclareAddresses(kernel, tPA))
        imod.add(self.lraFinalOffset(kernel, tPB))
        imod.add(self.lraDeclareAddresses(kernel, tPB))
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          imod.add(self.lraFinalOffset(kernel, tPM))
          imod.add(self.lraDeclareAddresses(kernel, tPM))
        localRead2Perpendicular = False
        instructions = self.memoryInstructions

        if kernel["UnrollMajorLDSA"]:
          localReadWidth = (kernel["MIInputPerThreadA"] * tPA["bpeDS"]) // self.states.bpr
          localReadInstructionIdxA = \
            self.selectMemoryInstruction("LocalRead", localReadWidth, \
            False, \
            tPA["localRead2Coalesced"], localRead2Perpendicular,
            [tPB["localReadStrideCoalesced"]] )
          tPA["localReadInstruction"] = instructions["LocalRead"][localReadInstructionIdxA]


        if kernel["UnrollMajorLDSB"]:
          localReadWidth = (kernel["MIInputPerThreadB"] * tPB["bpeDS"]) // self.states.bpr
          localReadInstructionIdxB = \
            self.selectMemoryInstruction("LocalRead", localReadWidth, \
            False, \
            tPB["localRead2Coalesced"], localRead2Perpendicular,
            [tPB["localReadStrideCoalesced"]] )
          tPB["localReadInstruction"] = instructions["LocalRead"][localReadInstructionIdxB]

        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          localReadWidth = tPM["bpeDS"] / self.states.bpr
          if kernel["UnrollMajorLDSMetadata"]:
            localReadWidth = (kernel["MIInputPerThreadMetadata"] * tPM["bpeDS"]) // self.states.bpr

          localReadInstructionIdxM = \
            self.selectMemoryInstruction("LocalRead", localReadWidth, \
            False, \
            tPM["localRead2Coalesced"], localRead2Perpendicular,
            [ tPM["localReadStrideCoalesced"]] )
          tPM["localReadInstruction"] = instructions["LocalRead"][ \
            localReadInstructionIdxM]
      # dot2: remove wider local read in tail loop, need to recalculate lra offset
      elif kernel["UseDotInstruction"] and (kernel["InnerUnroll"] > 1):
        self.states.numReadsIterCoalescedA = 1
        self.states.numReadsIterCoalescedB = 1
        self.states.lrvwUnrollA //= kernel["InnerUnroll"]
        self.states.lrvwUnrollB //= kernel["InnerUnroll"]
        localRead2Perpendicular = False
        instructions = self.memoryInstructions
        if kernel["UnrollMajorLDSA"]:
          localReadWidth = (kernel["NumDotElements"] * tPA["bpeDS"]) // self.states.bpr
          localReadInstructionIdxA = \
            self.selectMemoryInstruction("LocalRead", localReadWidth, \
            False, \
            tPA["localRead2Coalesced"], localRead2Perpendicular,
            [tPB["localReadStrideCoalesced"]] )
          tPA["localReadInstruction"] = instructions["LocalRead"][localReadInstructionIdxA]
        if kernel["UnrollMajorLDSB"]:
          localReadWidth = (kernel["NumDotElements"] * tPB["bpeDS"]) // self.states.bpr
          localReadInstructionIdxB = \
            self.selectMemoryInstruction("LocalRead", localReadWidth, \
            False, \
            tPB["localRead2Coalesced"], localRead2Perpendicular,
            [tPB["localReadStrideCoalesced"]] )
          tPB["localReadInstruction"] = instructions["LocalRead"][localReadInstructionIdxB]
        imod.add(self.lraTileAssignment(kernel, tPA, tPB))
        imod.add(self.lraFinalOffset(kernel, tPA))
        imod.add(self.lraDeclareAddresses(kernel, tPA))
        imod.add(self.lraFinalOffset(kernel, tPB))
        imod.add(self.lraDeclareAddresses(kernel, tPB))

    return imod

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    imod = Module()

    LWDoMod = imod.add(Module())
    LWDoA = self.localWriteDo(kernel, tPA) if self.do["LocalWrite%s"%tPA["tensorChar"]] else Module()
    LWDoB = self.localWriteDo(kernel, tPB) if self.do["LocalWrite%s"%tPB["tensorChar"]] else Module()
    LWDoMod.addComment1("local write a")
    LWDoMod.add(LWDoA)
    LWDoMod.addComment1("local write b")
    LWDoMod.add(LWDoB)
    return imod

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  def localWriteDo(self, kernel, tP, swapAB=0):

    tc = tP["tensorChar"]
    imod = Module()
    isBpeInputLarger = True if tP["bpeGR"] > tP["bpe"] else False

    def localWriteBody(tP):
      tc = tP["tensorChar"]

      instruction = tP["localWriteInstruction"]
      numBlocks = instruction.numBlocks
      numOffsets = instruction.numOffsets
      blockWidth = instruction.blockWidth
      #offsetMultiplier = instruction.offsetMultiplier
      g2lIdx = 0

      tmpLocalWriteAddr = -1

      # using _ds_store_b8: need one more vgpr space to do lshr
      tmpVgprOffset = ((self.states.a.numVgprG2L if (tP['tensorChar'] == 'A') else self.states.m.numVgprG2L if tP["isM"] else self.states.b.numVgprG2L) / 2) if (blockWidth == 0.25) else 0

      # if transposing, positions of sPerp and sPara are transposed
      instructionCnt = 0
      Hcvt2BMap = {}
      g2lIdxDict = {}
      regTmpVgprBlock = None

      if swapAB == 1:
        destVgprPrefix = "G2L%s2"%(tc)
      else:
        destVgprPrefix = "G2L%s"%(tc)
      for perp in range(0, tP["nrp"]):
        localWriteCode = imod.add(Module("LocalWrite%u perp=%d"%(instructionCnt,perp)))
        lwa = "LocalWriteAddr%s"%tc  # default

        for para in range(0, tP["nrc"]):
          if para>=1:
            localWriteCode = imod.add(Module("LocalWrite%u perp=%d para=%d"%(instructionCnt,perp,para)))

          for s in range(0, max(tP["nwcv"],tP["nwpv"])//tP["nwcvpi"]):
            localWriteCVTCode = Module()
            sPerp = 0
            sPara = 0
            needToSplitMetadata = False
            metadataScalar = 1.0
            if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
              if tP["wtc"]:
                sPerp = s
            else:
              if tP["wtc"]:
                sPara = s
                if tP["isM"]:
                  needToSplitMetadata = True
                  metadataScalar = max(1, self.states.bpr // ( tP["glvw"] * tP["bpeGR"]))
            #print("perp:{}/{} para:{}/{} sPerp:{} sPara:{}".format(perp,tP["nrp"],para,tP["nrc"],sPerp,sPara))
            (offset, i, comment) = self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP)

            # Need refactor, the pattern < glvw in fp8 is not the same as the original.
            # Thus the offset calculation here does not match global read.
            if tP["bpeGR"] == 4 and tP["bpeDS"] == 2:
              # float to fp16/bf16, keep idx in the lower idx.
              g2lIdx = int((i * blockWidth) * metadataScalar) * 2
            elif tP["glvw"] <= 2:
              # In Metadata use metadataScalar to group # of i
              g2lIdx = (i * blockWidth) * metadataScalar
              if isBpeInputLarger:
                g2lIdx *= (tP["bpeGR"]// tP["bpeDS"])
              g2lIdx = int(g2lIdx)
            else:
              g2lIdx = int((i * blockWidth) * metadataScalar)
              if isBpeInputLarger:
                g2lIdx *= (tP["bpeGR"]// tP["bpeDS"])

            graIdx = i * self.states.rpgo if kernel["BufferLoad"] else i * self.states.rpga

            if tP["isM"]:
              if not needToSplitMetadata:
                g2lIdx = graIdx * ceil(blockWidth)

            # If g2lIdx is already in the dict and blockWidth < 1, the data may
            # be packed into one register.
            instHi = 0
            if g2lIdx in g2lIdxDict:
              g2lIdxDict[g2lIdx] += 1
            else:
              g2lIdxDict[g2lIdx] = 0
            instHi = g2lIdxDict[g2lIdx]

            if self.states.archCaps["HasEccHalf"] or not self.states.asmCaps["HasWMMA_V1"]:
              numVgprG2L = self.states.a.numVgprG2L if tc == 'A' else self.states.b.numVgprG2L if tc == 'B' else self.states.m.numVgprG2L
              eccinstHi = instHi
              # FIXME: Workaround, unique pattern in 8bit + glvw == 2...
              if tP["bpeDS"] == tP["bpeGR"] and (tP["globalReadInstruction"].totalWidth) == 0.5 and (blockWidth == 0.25) and not tP["isM"]:
                eccinstHi = i // 2
              eccBpe = tP["bpeDS"] if kernel["ConvertAfterDS"] else max(tP["bpeGR"], tP["bpe"])
              eccOffset = _getEccOffset(tP["globalReadInstruction"].totalWidth, bpr=self.states.bpr, bpe=eccBpe, \
                glvw=tP["glvw"], idx=eccinstHi, numVgprG2L=numVgprG2L)
            else:
              eccOffset = 0

            # TODO- INT8: check uDu
            if (blockWidth == 0.25) and ((s % 4) == 0) and (not tP["isM"] or needToSplitMetadata):
                src = destVgprPrefix + "+%u" % (g2lIdx + eccOffset)
                dst = destVgprPrefix + "+%u+%u" % (tmpVgprOffset, g2lIdx)
                if tP["bpeDS"] != tP["bpeGR"]:
                  if kernel["ProblemType"]["DataType%s"%tc].isHalf():
                    if tP["glvw"] > 1:
                      dst = destVgprPrefix + "+%u+%u" % (tmpVgprOffset, g2lIdx // 2)
                      localWriteCVTCode.add(VPackF16toB32(dst=vgpr(dst), src0=vgpr(src), src1=vgpr(destVgprPrefix + "+%u" % (g2lIdx+1)), \
                                        vop3=VOP3PModifiers(op_sel=[1,1,0]), comment="Pack with neighbor"))
                      localWriteCVTCode.add(VPackF16toB32(dst=vgpr(src), src0=vgpr(src), src1=vgpr(destVgprPrefix + "+%u" % (g2lIdx+1)), \
                                        vop3=VOP3PModifiers(op_sel=[0,0,0]), comment="Pack with neighbor"))
                  else:
                    printExit("Unsupported combination DataType%s (%s) -> DataType (%s)"%(tc, kernel["ProblemType"]["DataType%s"%tc].toChar(), kernel["ProblemType"]["DataType"].toChar()))
                elif tP["glvw"] > 1:
                  localWriteCVTCode.add(VMovB32(dst=vgpr(dst), src=vgpr(src), comment="another VGPR storing lshr 8-bit value"))
                  localWriteCVTCode.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(8), src=vgpr(dst), comment="G2L Vpgr >> 8"))
                  if kernel["ExpertSchedulingMode"] > 0:
                    localWriteCVTCode.add(SWaitAlu(va_vdst=0, comment="wait for writes to complete"))

            #comment = "Reg -> L %u_%u_%u_%u"%(para, sPara, perp, sPerp)
            isHigh16Bits = False
            isCvtHighBits = False
            datatype = kernel["ProblemType"]["DataType%s"%tc] if kernel["ConvertAfterDS"] else kernel["ProblemType"]["DataType"]
            if (datatype.isHalf() or datatype.isBFloat16()) and not tP["isM"]:
              if s%2==1:
                isHigh16Bits = True
              if (blockWidth == 0.5) and (instHi % 2 == 1):
                isHigh16Bits = True
              if kernel["ProblemType"]["DataType%s"%tc].isAnyFloat8():
                if g2lIdx%2 == 1:
                  isCvtHighBits = True
            #       |  hi16  |  hi16  |        |        |
            #       |  hi8   |        |   hi8  |        |
            #############################################
            # VGPR: |---w4---|---w3---|---w2---|---w1---| -> b8_d16: get w1 / _b8_d16_hi: get w3
            # LSHR: |--------|---w4---|--------|---w2---| -> b8_d16: get w2 / _b8_d16_hi: get w4
            elif datatype.isInt8() or datatype.is8bitFloat() or tP["isM"]:
              isHigh16Bits = (s % 4) > 1 # 2,3
              # TODO
              # if tP["glvw"]==1 and instructionCnt%2==1:
              #   isHigh16Bits = True

            paramList = []
            numsOfRegister = []
            globalBlockWidth = tP["globalReadInstruction"].totalWidth
#            print("tc = ", tc, ", numBlocks = ", numBlocks)
#            print("regs: ", self.vgprs.globalReadRegisters[tc])
#            print("destVgprPrefix = ", destVgprPrefix, ", blockWidth = ", blockWidth)
            for _ in range(0, numBlocks):
              # FIXME: In the future all registers should pass from global read instead of recalculate them
              if globalBlockWidth == 1 and blockWidth == 0.5 and tP["glvw"] == 1:
                if isHigh16Bits:
                  paramList.append(vgpr(destVgprPrefix + "+%u"%(g2lIdx + 1)))
                else:
                  paramList.append(vgpr(destVgprPrefix + "+%u"%(g2lIdx)))
                numsOfRegister.append(blockWidth)
              elif globalBlockWidth == blockWidth and tP["glvw"] == 1:
#                print("destVgprPrefix = ", destVgprPrefix)
#                print("tc = ", tc, ", i", i)
#                print(self.vgprs.globalReadRegisters[tc][i])
#                print("blockWidth = ", blockWidth)
#                print(vgpr(destVgprPrefix + "+%u"%(self.vgprs.globalReadRegisters[tc][i]), blockWidth))
#                print("i = ", i)
#                print("_ = ", _)
                paramList.append(vgpr(destVgprPrefix + "+%u"%(self.vgprs.globalReadRegisters[tc][i]), blockWidth))
#                print("DONE")
              elif blockWidth == 1:
                paramList.append(vgpr(destVgprPrefix + "+%u"%(g2lIdx)))
                numsOfRegister.append(1)
              elif blockWidth == 0.25 and ((s % 2) == 1): # Int8, s = 1 or 3 (high8Bits)
                if tP["bpeDS"] != tP["bpeGR"] and tmpVgprOffset != 0:
                  paramList.append(vgpr(destVgprPrefix + "+%u+%u"%(tmpVgprOffset, g2lIdx // 2)))
                else:
                  paramList.append(vgpr(destVgprPrefix + "+%u+%u"%(tmpVgprOffset, g2lIdx)))
                numsOfRegister.append(1)
              else:
                paramList.append(vgpr(destVgprPrefix + "+%u"%(g2lIdx + eccOffset), blockWidth))
                numsOfRegister.append(blockWidth)
              if self.db["ForceInputValue%s"%tc]:
                localWriteCVTCode.add(VMovB32(dst=vgpr(destVgprPrefix + "+%u"%(g2lIdx)), src=self.db["ForceValue%s"], comment="ForceInputValue"))
              if (kernel["ProblemType"]["DataType"].isBFloat16() and kernel["ProblemType"]["DataType%s"%tc].isHalf()) and not tP["isM"]:
                numIters = 1 if blockWidth <= 1 else blockWidth
                vgprTmp = self.vgprPool.checkOut(2)
                for iter in range(0, numIters):
                  f16Tobf16Idx = g2lIdx + iter
                  if f16Tobf16Idx in Hcvt2BMap:
                    Hcvt2BMap[f16Tobf16Idx] += 2
                  else:
                    Hcvt2BMap[f16Tobf16Idx] = 0
                  f16Tobf16Idx += Hcvt2BMap[f16Tobf16Idx]
                  sdwa = SDWAModifiers(src0_sel=SelectBit.WORD_1)
                  localWriteCVTCode.add(VCvtF16toF32(dst=vgpr(vgprTmp), src=vgpr(destVgprPrefix + "+%u"%(f16Tobf16Idx))))
                  localWriteCVTCode.add(VCvtF16toF32(dst=vgpr(vgprTmp+1), src=vgpr(destVgprPrefix + "+%u"%(f16Tobf16Idx)),sdwa=sdwa))
                  localWriteCVTCode.add(VPackF16toB32(dst=vgpr(destVgprPrefix + "+%u"%(f16Tobf16Idx)), src0=vgpr(vgprTmp), src1=vgpr(vgprTmp+1),
                                    vop3=VOP3PModifiers(op_sel=[1,1,0])))
                self.vgprPool.checkIn(vgprTmp)

            for oIdx in range(0, numOffsets):
              paramList.append(offset)

            #print "offset", offset

            # Need cvt
            if tP["bpeDS"] != tP["bpeGR"]:
              assert numBlocks == 1
              if (kernel["ProblemType"]["DataType%s"%tc].isSingle() and kernel["ProblemType"]["DataType"].isHalf()):
                newBlockWidth = (tP["bpeGR"] / tP["bpe"]) * blockWidth
                if newBlockWidth == 1:
                  dst_sel = SelectBit.WORD_1 if isHigh16Bits else SelectBit.WORD_0
                  new_src = deepcopy(paramList[0])
                  if isHigh16Bits:
                    new_src.regName.addOffset(1)
                  localWriteCVTCode.add(VCvtF32toF16(dst=paramList[0], src=new_src, sdwa=SDWAModifiers(dst_sel=dst_sel), comment="convert C to fp16"))
                else:
                  for vi in range(0, int(newBlockWidth)):
                    dst_sel = SelectBit.WORD_1 if vi%2==1 else SelectBit.WORD_0
                    localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi)), sdwa=SDWAModifiers(dst_sel=dst_sel), comment="convert C to fp16"))
              elif (kernel["ProblemType"]["DataType%s"%tc].isHalf() and kernel["ProblemType"]["DataType"].is8bitFloat()):
                #HH_F8/B8/F8B8/B8F8_
                toF8 = False
                if (kernel["ProblemType"]["DataType"].isAnyFloat8() or
                  (tc == "A" and kernel["ProblemType"]["DataType"].isAnyFloat8BFloat8()) or
                  (tc == "B" and kernel["ProblemType"]["DataType"].isAnyBFloat8Float8())):
                  toF8 = True

                newBlockWidth = (tP["bpeGR"] / tP["bpe"]) * blockWidth
                if newBlockWidth == 0.5:
                  # We cannot use optimized f16->fp8 instruction here as we have only one half element to be converted
                  # and we only have a pack instruction which can convert 2 half elements to 2 f8 element
                  if kernel["ProblemType"]["StochasticRounding"]:
                    if self.states.asmCaps["v_prng_b32"]:
                      vgprTmp = self.vgprPool.checkOutAligned(2, 2)
                    else:
                      vgprTmp = self.vgprPool.checkOutAligned(4, 2)
                  else:
                    vgprTmp = self.vgprPool.checkOutAligned(1, 2)
                  src_sel = SelectBit.WORD_1 if isHigh16Bits else SelectBit.WORD_0
                  sel = 1 if isHigh16Bits else 0
                  localWriteCVTCode.add(VCvtF16toF32(dst=vgpr(vgprTmp), src=paramList[0], sdwa=SDWAModifiers(src0_sel=src_sel), comment="convert to F32"))

                  # ScaleA/B
                  if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and kernel["ProblemType"]["DataType%s"%tc].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters():
                    localWriteCVTCode.add(VMulF32(dst=vgpr(vgprTmp), src0=vgpr(vgprTmp), src1=sgpr("Scale%s"%tc), comment="Input *= scale %s"%tc))

                  if kernel["ProblemType"]["StochasticRounding"]:
                    vRand = vgprTmp+1 #seed
                    if self.states.asmCaps["v_prng_b32"]:
                      localWriteCVTCode.add(VPrngB32(dst=vgpr(vRand),src=vgpr(vgprTmp),comment="Psudo Random Number Generator"))
                    else:
                      vTemp0 = vgprTmp+2
                      vTemp1 = vgprTmp+3
                      localWriteCVTCode.add(MacroInstruction(name="PRND_GENERATOR", args=[vRand, vgprTmp, vTemp0, vTemp1]))

                    if (toF8):
                      localWriteCVTCode.add(VCvtSRF32toFP8(dst=paramList[0], src0=vgpr(vgprTmp), src1=vgpr(vRand), vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="Convert to FP8"))
                    else:
                      localWriteCVTCode.add(VCvtSRF32toBF8(dst=paramList[0], src0=vgpr(vgprTmp), src1=vgpr(vRand), vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="Convert to BF8"))
                  else:
                    if (toF8):
                      localWriteCVTCode.add(VCvtPkF32toFP8(dst=paramList[0], src0=vgpr(vgprTmp), src1=vgpr(vgprTmp), vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="Convert to FP8"))
                    else:
                      localWriteCVTCode.add(VCvtPkF32toBF8(dst=paramList[0], src0=vgpr(vgprTmp), src1=vgpr(vgprTmp), vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="Convert to BF8"))
                  self.vgprPool.checkIn(vgprTmp)
                else:
                  if kernel["ProblemType"]["StochasticRounding"]:
                    if self.states.asmCaps["v_prng_b32"] :
                      vgprTmp = self.vgprPool.checkOutAligned(3, 2)
                    else:
                      vgprTmp = self.vgprPool.checkOutAligned(5, 2)
                  else:
                    vgprTmp = self.vgprPool.checkOutAligned(2, 2)
                  vgprTmp2 = vgprTmp + 1
                  for vi in range(0, int(newBlockWidth)):
                    sel = 1 if vi %2 == 1 else 0
                    if self.states.asmCaps["Hascvtfp8_f16"] and not kernel["ProblemType"]["StochasticRounding"] and not kernel["ProblemType"]["UseScaleAB"] == "Scalar":
                      if (toF8):
                        localWriteCVTCode.add(VCvtScalePkF16toFP8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi)), scale=0x3f800000,\
                                                              vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="convert F16 to F8"))
                      else:
                        localWriteCVTCode.add(VCvtScalePkF16toBF8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi)), scale=0x3f800000,\
                                                              vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="convert F16 to BF8"))
                    else:
                      localWriteCVTCode.add(VCvtF16toF32(dst=vgpr(vgprTmp), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi)), sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                      localWriteCVTCode.add(VCvtF16toF32(dst=vgpr(vgprTmp2), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi)), sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_1), comment="convert to F32"))

                      if kernel["ProblemType"]["StochasticRounding"]:
                        # ScaleA/B, sgpr upper is dummy.
                        if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and kernel["ProblemType"]["DataType%s"%tc].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters():
                          localWriteCVTCode.add(VMulPKF32S(dst=vgpr(vgprTmp, 2), src0=vgpr(vgprTmp, 2), src1=sgpr("Scale%s"%tc, 2), vop3=VOP3PModifiers(op_sel_hi=[1,0,1]), comment="Input *= scale %s"%tc))
                        vRand = vgprTmp+2
                        if self.states.asmCaps["v_prng_b32"]:
                          localWriteCVTCode.add(VPrngB32(dst=vgpr(vRand),src=vgpr(vgprTmp),comment="Psudo Random Number Generator"))
                        else:
                          vTemp0 = vgprTmp+3
                          vTemp1 = vgprTmp+4
                          localWriteCVTCode.add(MacroInstruction(name="PRND_GENERATOR", args=[vRand, vgprTmp, vTemp0, vTemp1]))
                        if (toF8):
                          localWriteCVTCode.add(VCvtSRF32toFP8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src0=vgpr(vgprTmp), src1=vgpr(vRand), vop3=VOP3PModifiers(op_sel=[0,0,0,sel]), comment="Convert to FP8"))
                          if self.states.asmCaps["v_prng_b32"]:
                            localWriteCVTCode.add(VPrngB32(dst=vgpr(vRand),src=vgpr(vgprTmp2),comment="Psudo Random Number Generator"))
                          else:
                            localWriteCVTCode.add(MacroInstruction(name="PRND_GENERATOR", args=[vRand, vgprTmp2, vTemp0, vTemp1]))
                          localWriteCVTCode.add(VCvtSRF32toFP8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src0=vgpr(vgprTmp2), src1=vgpr(vRand), vop3=VOP3PModifiers(op_sel=[0,0,1,sel]), comment="Convert to FP8"))
                        else:
                          localWriteCVTCode.add(VCvtSRF32toBF8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src0=vgpr(vgprTmp), src1=vgpr(vRand), vop3=VOP3PModifiers(op_sel=[0,0,0,sel]), comment="Convert to BF8"))
                          if self.states.asmCaps["v_prng_b32"]:
                            localWriteCVTCode.add(VPrngB32(dst=vgpr(vRand),src=vgpr(vgprTmp2),comment="Psudo Random Number Generator"))
                          else:
                            localWriteCVTCode.add(MacroInstruction(name="PRND_GENERATOR", args=[vRand, vgprTmp2, vTemp0, vTemp1]))
                          localWriteCVTCode.add(VCvtSRF32toBF8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src0=vgpr(vgprTmp2), src1=vgpr(vRand), vop3=VOP3PModifiers(op_sel=[0,0,1,sel]), comment="Convert to BF8"))
                      else:
                        # ScaleA/B, sgpr upper is dummy.
                        if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and kernel["ProblemType"]["DataType%s"%tc].numRegisters() > kernel["ProblemType"]["DataType"].numRegisters():
                          localWriteCVTCode.add(VMulPKF32S(dst=vgpr(vgprTmp, 2), src0=vgpr(vgprTmp, 2), src1=sgpr("Scale%s"%tc, 2), vop3=VOP3PModifiers(op_sel_hi=[1,0,1]), comment="Input *= scale %s"%tc))

                        if (toF8):
                          localWriteCVTCode.add(VCvtPkF32toFP8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src0=vgpr(vgprTmp), src1=vgpr(vgprTmp2), vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="Convert to FP8"))
                        else:
                          localWriteCVTCode.add(VCvtPkF32toBF8(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src0=vgpr(vgprTmp), src1=vgpr(vgprTmp2), vop3=VOP3PModifiers(op_sel=[0,0,sel]), comment="Convert to BF8"))
                  self.vgprPool.checkIn(vgprTmp)

              elif (kernel["ProblemType"]["DataType%s"%tc].isAnyFloat8() and kernel["ProblemType"]["DataType"].isHalf()):
                newBlockWidth = tP["globalReadInstruction"].blockWidth
                if newBlockWidth == 0.25:
                  new_src = deepcopy(paramList[0])
                  if tP["glvw"] == 1:
                    vgprTmp = self.vgprPool.checkOut(1)
                    new_src.regName.addOffset(tP["shiftGR"])
                    src_sel = SelectBit.BYTE_2 if isHigh16Bits else SelectBit.BYTE_0
                    dst_sel = SelectBit.WORD_1 if isHigh16Bits else SelectBit.WORD_0
                    if self.states.asmCaps["Hascvtf16_fp8"]:
                      sel = [0,1,1,0] if isHigh16Bits else [0,0,0,0]
                      localWriteCVTCode.add(VCvtScaleFP8toF16(dst=paramList[0], src=new_src, scale=0x3f800000, vop3=VOP3PModifiers(op_sel=sel), comment="A convert fp8 to f16"))
                    else:
                      localWriteCVTCode.add(VCvtFP8toF32(dst=vgpr(vgprTmp), src=new_src , sdwa=SDWAModifiers(src0_sel=src_sel), comment="convert C to fp32"))
                      localWriteCVTCode.add(VCvtF32toF16(dst=paramList[0], src=vgpr(vgprTmp), sdwa=SDWAModifiers(dst_sel=dst_sel), comment="convert C to fp16"))
                    self.vgprPool.checkIn(vgprTmp)
                  else:
                    if isCvtHighBits and isHigh16Bits:
                      src_sel = SelectBit.BYTE_3
                    elif isHigh16Bits and (not isCvtHighBits):
                      new_src.regName.addOffset(tP["shiftGR"])
                      src_sel = SelectBit.BYTE_1
                    elif (not isHigh16Bits) and isCvtHighBits:
                      src_sel = SelectBit.BYTE_2
                    else:
                      new_src.regName.addOffset(tP["shiftGR"])
                      src_sel = SelectBit.BYTE_0
                    dst_sel = SelectBit.WORD_1 if isHigh16Bits else SelectBit.WORD_0
                    if self.states.asmCaps["Hascvtf16_fp8"]:
                        if isHigh16Bits:
                          if isCvtHighBits:
                            sel = [1,1,1,0]
                          else:
                            sel = [0,1,1,0]
                        else:
                          if isCvtHighBits:
                            sel = [0,1,0,0]
                          else:
                            sel = [0,0,0,0]

                    if new_src == paramList[0]:
                      if self.states.asmCaps["Hascvtf16_fp8"]:
                        localWriteCVTCode.add(VCvtScaleFP8toF16(dst=paramList[0], src=new_src , scale=0x3f800000, vop3=VOP3PModifiers(op_sel=sel), comment="B convert fp8 to f16"))
                      else:
                        if src_sel == SelectBit.BYTE_0 or src_sel == SelectBit.BYTE_2:
                          if regTmpVgprBlock == None:
                            regTmpVgprBlock = self.vgprPool.checkOutAligned(2, 2)
                          src_sel2 = SelectBit.WORD_0 if src_sel == SelectBit.BYTE_0 else SelectBit.WORD_1
                          localWriteCVTCode.add(VCvtPkFP8toF32(dst=vgpr(regTmpVgprBlock, 2), src=new_src , sdwa=SDWAModifiers(src0_sel=src_sel2), comment="convert C to fp32"))
                          localWriteCVTCode.add(VCvtF32toF16(dst=paramList[0], src=vgpr(regTmpVgprBlock), sdwa=SDWAModifiers(dst_sel=dst_sel), comment="convert C to fp16"))
                        else:
                          localWriteCVTCode.add(VCvtF32toF16(dst=paramList[0], src=vgpr(regTmpVgprBlock + 1), sdwa=SDWAModifiers(dst_sel=dst_sel), comment="convert C to fp16"))
                    else:
                      vgprTmp = self.vgprPool.checkOut(1)
                      if self.states.asmCaps["Hascvtf16_fp8"]:
                        localWriteCVTCode.add(VCvtScaleFP8toF16(dst=paramList[0], src=new_src, scale=0x3f800000, vop3=VOP3PModifiers(op_sel=sel), comment="C convert fp8 to f16"))
                      else:
                        localWriteCVTCode.add(VCvtFP8toF32(dst=vgpr(vgprTmp), src=new_src , sdwa=SDWAModifiers(src0_sel=src_sel), comment="convert C to fp32"))
                        localWriteCVTCode.add(VCvtF32toF16(dst=paramList[0], src=vgpr(vgprTmp), sdwa=SDWAModifiers(dst_sel=dst_sel), comment="convert C to fp16"))
                      self.vgprPool.checkIn(vgprTmp)
                elif newBlockWidth == 0.5:
                  vgprTmp = self.vgprPool.checkOutAligned(2, 2)
                  src_sel = SelectBit.WORD_1 if isCvtHighBits else SelectBit.WORD_0
                  modNum = max(1, int(newBlockWidth / blockWidth))
                  if self.states.asmCaps["Hascvtf16_fp8"]:
                    sel  = [1,0,0,0] if isCvtHighBits else [0,0,0,0]
                    localWriteCVTCode.add(VCvtScalePkFP8toF16(dst=vgpr(destVgprPrefix + "+%u"%(g2lIdx)), src=vgpr(destVgprPrefix + "+%u"%(g2lIdx)), scale=0x3f800000,\
                                                          vop3=VOP3PModifiers(op_sel=sel), comment="D convert fp8 to f16"))
                  else:
                    if (not isHigh16Bits) and (g2lIdx % modNum == 0):
                      localWriteCVTCode.add(VCvtPkFP8toF32(dst=vgpr(vgprTmp, 2), src=vgpr(destVgprPrefix + "+%u"%(g2lIdx)), sdwa=SDWAModifiers(src0_sel=src_sel), comment="convert to F32"))
                      localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u"%(g2lIdx)), src=vgpr(vgprTmp), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                      if (newBlockWidth <= blockWidth):
                        localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u"%(g2lIdx)), src=vgpr(vgprTmp+1), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                    elif (newBlockWidth > blockWidth):
                      localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u"%(g2lIdx)), src=vgpr(vgprTmp+1), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                  self.vgprPool.checkIn(vgprTmp)
                else:
                  modNum = max(1, int(newBlockWidth / blockWidth))
                  vgprTmp = self.vgprPool.checkOutAligned(2, 2)
                  if (not isHigh16Bits) and (newBlockWidth <= blockWidth):
                    for vi in range(0, int(newBlockWidth)):
                      if self.states.asmCaps["Hascvtf16_fp8"]:
                        localWriteCVTCode.add(VCvtScaleFP8toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2)),     src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx+tP["shiftGR"], vi)), scale=0x3f800000,\
                                                            vop3=VOP3PModifiers(op_sel=[0,0,0,0]), comment="E convert fp8 to f16"))
                        localWriteCVTCode.add(VCvtScaleFP8toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2)),     src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx+tP["shiftGR"], vi)), scale=0x3f800000,\
                                                            vop3=VOP3PModifiers(op_sel=[1,0,1,0]), comment="E convert fp8 to f16"))
                        localWriteCVTCode.add(VCvtScaleFP8toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2 + 1)), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx+tP["shiftGR"], vi)), scale=0x3f800000,\
                                                            vop3=VOP3PModifiers(op_sel=[0,1,0,0]), comment="E convert fp8 to f16"))
                        localWriteCVTCode.add(VCvtScaleFP8toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2 + 1)), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx+tP["shiftGR"], vi)), scale=0x3f800000,\
                                                            vop3=VOP3PModifiers(op_sel=[1,1,1,0]), comment="E convert fp8 to f16"))
                      else:
                        localWriteCVTCode.add(VCvtPkFP8toF32(dst=vgpr(vgprTmp, 2), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx+tP["shiftGR"], vi)), sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                        localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2)), src=vgpr(vgprTmp), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                        localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2)), src=vgpr(vgprTmp+1), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                        localWriteCVTCode.add(VCvtPkFP8toF32(dst=vgpr(vgprTmp, 2), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx+tP["shiftGR"], vi)), sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_1), comment="convert to F32"))
                        localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2 + 1)), src=vgpr(vgprTmp), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                        localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi * 2 + 1)), src=vgpr(vgprTmp+1), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                  else:
                    if (not isHigh16Bits):
                      idxMod = g2lIdx % modNum
                      g2lIdxTmp = g2lIdx - idxMod
                      vi = idxMod // 2
                      selectBit = SelectBit.WORD_0 if idxMod % 2 == 0 else SelectBit.WORD_1
                      interOffset = 0 if idxMod % 2 == 0 else 1
                      if self.states.asmCaps["Hascvtf16_fp8"]:
                        sel = 1 if idxMod % 2 == 0 else 1
                        localWriteCVTCode.add(VCvtScalePkFP8toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdxTmp, vi * 2 + interOffset)),\
                                                              src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdxTmp+tP["shiftGR"], vi)), scale=0x3f800000,\
                                                              vop3=VOP3PModifiers(op_sel=[0,0,0,sel]), comment="F convert fp8 to f16"))
                      else:
                        localWriteCVTCode.add(VCvtPkFP8toF32(dst=vgpr(vgprTmp, 2), src=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdxTmp+tP["shiftGR"], vi)), sdwa=SDWAModifiers(src0_sel=selectBit), comment="convert to F32"))
                        localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdxTmp, vi * 2 + interOffset)), src=vgpr(vgprTmp), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                        if blockWidth != 0.5:
                          localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdxTmp, vi * 2 + interOffset)), src=vgpr(vgprTmp+1), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                    elif blockWidth == 0.5:
                      localWriteCVTCode.add(VCvtF32toF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdxTmp, vi * 2 + interOffset)), src=vgpr(vgprTmp+1), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                  self.vgprPool.checkIn(vgprTmp)
              elif (kernel["ProblemType"]["DataType%s"%tc].isSingle() and kernel["ProblemType"]["DataType"].isBFloat16()):
                newBlockWidth = (tP["bpeGR"] / tP["bpe"]) * blockWidth
                if tP["glvw"] == 1:
                  # use the same VGPR
                  localWriteCVTCode.add(VCvtPkF32toBF16(dst=paramList[0], src0=paramList[0], src1=paramList[0], comment="convert one fp32 to bf16 to the same VGPR"))
                elif newBlockWidth == 1:
                  new_src = deepcopy(paramList[0])
                  if isHigh16Bits:
                    new_src.regName.addOffset(1)
                  localWriteCVTCode.add(VCvtPkF32toBF16(dst=paramList[0], src0=new_src, src1=new_src, comment="convert one fp32 to bf16 idx=%d"%g2lIdx))
                else:
                  for vi in range(0, int(newBlockWidth)):
                    if vi%2 == 0:
                      localWriteCVTCode.add(VCvtPkF32toBF16(dst=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi//2)), src0=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi)), src1=vgpr(destVgprPrefix + "+%u+%u"%(g2lIdx, vi+1)), comment="convert/pack fp32 to bf16"))
              else:
                printExit("Unsupported combination DataType%s (%s) -> DataType (%s)"%(tc, kernel["ProblemType"]["DataType%s"%tc].toChar(), kernel["ProblemType"]["DataType"].toChar()))

            LocalWriteX = tP["localWriteInstruction"].getInst(isHigh16Bits)
            if numBlocks == 1:
              if (paramList[1] >= 0x20000):
                  olwa = "LocalWriteAddr%s+2"%tc  # default
                  dstAddr=vgpr(olwa)
                  paramList[1] = paramList[1] - 131072
              elif (paramList[1] >= 0x10000):
                  olwa = "LocalWriteAddr%s+1"%tc  # default
                  dstAddr=vgpr(olwa)
                  paramList[1] = paramList[1] - 65536
              else:
                dstAddr=vgpr(lwa)
              ds        = DSModifiers(na=1, offset=paramList[1])
              writeInst = LocalWriteX(dstAddr=dstAddr, src=paramList[0], ds=ds, comment=comment)
            else:
              ds        = DSModifiers(na=2, offset0=paramList[2], offset1=paramList[3])
              writeInst = LocalWriteX(dstAddr=vgpr(lwa), src0=paramList[0], src1=paramList[1], ds=ds, comment=comment)
            if self.do["LocalWriteCVT"]:
              localWriteCode.add(localWriteCVTCode)
            if self.do["LocalWrite%s"%tc]:
              localWriteCode.add(writeInst)
              instructionCnt += 1 if blockWidth < 8 else 2

      if regTmpVgprBlock != None:
        self.vgprPool.checkIn(regTmpVgprBlock)

      if tmpLocalWriteAddr != -1:
        self.vgprPool.checkIn(tmpLocalWriteAddr)

      #if vgprFp32NanInfFlag != None:
        #self.vgprPool.checkIn(vgprFp32NanInfFlag)

      if kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"]:
          miWaveTile = kernel["MIWaveTileB"] if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) else kernel["MIWaveTileA"] if (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) else 0
          if kernel["PrefetchGlobalRead"] == 2 and miWaveTile > 0:
            #vpgr to store metadata
            offsetBlk = tP["metadataWriteSwapByteOffset"]
            instructionCnt = -1
            for wtIdx in range(0, miWaveTile):
              for unrollIdx in range(0, kernel["LoopIters"]):
                instructionCnt +=1
                localWriteCode = imod.add(Module("MetadataWrite%u "%(instructionCnt)))
                localWriteCode.add(VMovB32( \
                  vgpr("ValuMetadata+%u+%u"%(offsetBlk, (wtIdx * kernel["LoopIters"]+unrollIdx))), \
                  vgpr("ValuMetadata+%u+%u"%(self.states.m.numVgprValuPerBlock * 2, (wtIdx*kernel["LoopIters"]+unrollIdx))), \
                  comment="copy ValuMetadata from blk2"))

      if 0 and tP["isB"]: # post-lds-write
        localWriteCode.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=0, comment=""))
        localWriteCode.add(SBarrier(comment="dump LDS"))
        localWriteCode.add(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup0"),1))

    # Enable local write if not DTL or using nonDTL loads in tail loop
    if (not kernel["DirectToLds%s"%tc]) or \
       (((tP["isA"] and kernel["NonDTLTailLoopA"]) or \
        (tP["isB"] and kernel["NonDTLTailLoopB"])) and self.states.inTailLoop):
      # Skip local write if DTVA or DTVB
      if not ((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]):
        localWriteBody(tP)
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        if tP["is_sparse"]:
          localWriteBody(tP["tpsMetadata"])

    return imod

  ##############################################################################
  # Local Read: Swap Offsets A/B
  # internalPointerSwap: swap internally tracked offsets - rather than
  #    emit specific instructions to do the pointer swap
  ##############################################################################
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]):
      return Module("localReadSwapOffsets (no local read)")
    if kernel["1LDSBuffer"] or ((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]): # no local read code if DirectToVgpr is enabled
      return Module("localReadSwapOffsets (Empty)")
    module = Module("localReadSwapOffsets")

    if internalPointerSwap or kernel["StoreSwapAddr"]:
      if not kernel["StoreSwapAddr"]:
        tP["localReadSwapByteOffset"] = 0 if tP["localReadSwapByteOffset"] else kernel["LdsOffsetA_Blk"]
        module.addComment1("local read swap internal offset -> %u" % tP["localReadSwapByteOffset"])
      else:
        module.add(VXorB32(
          dst=vgpr("LocalReadAddr%s"%tc), \
          src0=vgpr("LocalReadSwapAddr%s"%tc), \
          src1=vgpr("LocalReadAddr%s"%tc), \
          comment="swap Red Blk"))
    else:
      module.add(VXorB32(
        dst=vgpr("LocalReadAddr%s"%tc), \
        src0=hex(kernel["LdsOffsetA_Blk"]), \
        src1=vgpr("LocalReadAddr%s"%tc), \
        comment="swap Red Blk"))

    numLra = 0
    if tP["isA"]:
      numLra = self.states.a.numVgprLocalReadAddr
    elif tP["isB"]:
      numLra = self.states.b.numVgprLocalReadAddr
    elif tP["isM"]:
      numLra = self.states.m.numVgprLocalReadAddr

    for i in range(1,numLra):
      module.add(VAddU32(
        dst=vgpr("LocalReadAddr%s+%u"%(tc,i)), \
        src0=(i * self.states.regCaps["maxLDSConstOffset"]), \
        src1=vgpr("LocalReadAddr%s"%tc), \
        comment="Final Offset Plus %uK"%((i * self.states.regCaps["maxLDSConstOffset"]) / 1024)))

    return module

  ##############################################################################
  # Local Read: Reset Offsets A/B
  # x % n == n & (n-1) for n power of 2
  # tP[localReadOffset] maintains running count of offsets
  # This is called from the tail loop to reset read offsets?
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return Module("localReadResetOffsets (no local read)")
    if kernel["1LDSBuffer"] or ((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]): # no local read code if DirectToVgpr is enabled
      return Module("localReadResetOffsets (Empty)")
    module = Module("localReadResetOffsets")
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadSwapByteOffset"] = 0
      module.addComment1("localReadResetOffsets")
      tP["localReadOffset"] = 0
      module.addComment0("handled internally")

    if kernel["StoreSwapAddr"]:
      # Reset offset, by picking smaller of the two
      tmpvgpr = self.vgprPool.checkOut(1) # contains other offsets
      module.add(VXorB32(
        dst=vgpr(tmpvgpr), \
        src0=vgpr("LocalReadSwapAddr%s"%tc), \
        src1=vgpr("LocalReadAddr%s"%tc), \
        comment="Get other lds buffer offset value"))
      module.add(VMinI32(dst=vgpr("LocalReadAddr%s"%tc), src0=vgpr("LocalReadAddr%s"%tc), src1=vgpr(tmpvgpr),
                      comment="Set LRA to first buffer offset"))
      self.vgprPool.checkIn(tmpvgpr)
    else:

      module.add(VAndB32(
        dst=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], 0)), \
        src0=hex(kernel["LdsOffsetA_Blk"]-1), \
        src1=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], 0)), \
        comment="reset Red,Blk -> Red"))

    numLra = 0
    if tP["isA"]:
      numLra = self.states.a.numVgprLocalReadAddr
    elif tP["isB"]:
      numLra = self.states.b.numVgprLocalReadAddr
    elif tP["isM"]:
      numLra = self.states.m.numVgprLocalReadAddr

    for i in range(1,numLra):
      module.add(VAddU32(
        dst=vgpr("LocalReadAddr%s+%u"%(tc,i)), \
        src0=(i * self.states.regCaps["maxLDSConstOffset"]), \
        src1=vgpr("LocalReadAddr%s"%tc), \
        comment="Final Offset Plus %uK"%((i * self.states.regCaps["maxLDSConstOffset"]) / 1024)))

    return module

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tPA, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]) or ((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]): # no local read code if DirectToVgpr is enabled
      return Module("localReadInitPointers (Empty)")
    module = Module("localReadInitPointers")
    if tPA["localReadInstruction"].numOffsets == 1:
      module.addComment1("localReadInitPointers")
      tP["localReadOffset"] = 0
    else:
      numLra = 0
      if tP["isA"]:
        numLra = self.states.a.numVgprLocalReadAddr
      elif tP["isB"]:
        numLra = self.states.b.numVgprLocalReadAddr
      elif tP["isM"]:
        numLra = self.states.m.numVgprLocalReadAddr
      module.add(VAndB32(
        dst=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], 0)), \
        src0=hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]-1), \
        src1=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], 0)), \
        comment="init Red,Blk -> Red"))
      for i in range(1, numLra):
        module.add(VAddU32(
          dst=vgpr("LocalReadAddr%s+%u"%(tc,i)), \
          src0=(i * self.states.regCaps["maxLDSConstOffset"]), \
          src1=vgpr("LocalReadAddr%s"%tc), \
          comment="Final Offset Plus %uK"%((i * self.states.regCaps["maxLDSConstOffset"]) / 1024)))
    return module

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    tc = tP["tensorChar"]
    if (not self.do["LocalRead%s" % tc]) or ((tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]): # no local read code if DirectToVgpr is enabled
      return Module("localReadInc (Empty)")

    module = Module("localReadInc")

    offsetInc = 0
    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0

    if self.states.inTailLoop:
      if kernel["UseDotInstruction"]:
        # dot2
        inc = self.states.lrvwUnrollA * kernel["NumWaveSplitK"] * tP["bpeDS"]
        comment = "(LocalReadVectorWidth*NumWaveSplitK*bpeDS)"
      else:
        inc = (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad) * tP["bpeDS"]
        comment = " ((MT+PAD)*bpeDS)"
      if kernel["EnableMatrixInstruction"]:
        matrixInstK = kernel["MatrixInstK"]
        if kernel["UnrollMajorLDS%s" % tc]:
          inc = tP["bpeDS"] * max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB)
          comment = " (bpeDS)"
        inc *= matrixInstK
        if kernel["ProblemType"]["Sparse"]:
          if (kernel["ProblemType"]["Sparse"] == 2 and tc == "B") or (kernel["ProblemType"]["Sparse"] == 1 and tc == "A"):
            inc //= 2
          elif tc == "Metadata":
            inc //= 8

      if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
        inc = inc + (inc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"]

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(inc), comment="inc"))
        numLra = 0
        if tP["isA"]:
          numLra = self.states.a.numVgprLocalReadAddr
        elif tP["isB"]:
          numLra = self.states.b.numVgprLocalReadAddr
        elif tP["isM"]:
          numLra = self.states.m.numVgprLocalReadAddr
        for i in range(numLra):
          module.add(VAddCOU32(
              dst=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], i)), \
              dst1=VCC(), \
              src0=sgpr(tmpSgpr), \
              src1=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], i)), \
              comment="lr%s += %u%s"%(tP["tensorChar"], inc, comment) ))
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        if kernel["EnableMatrixInstruction"]:
          if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            offsetInc = kernel["MatrixInstK"] * max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB)
            if kernel["ProblemType"]["Sparse"]:
              if (kernel["ProblemType"]["Sparse"] == 2 and tc == "B") or (kernel["ProblemType"]["Sparse"] == 1 and tc == "A"):
                offsetInc //= 2
              elif tc == "Metadata":
                offsetInc //= 8
          else:
            if tc == "A":
              sparseA = kernel["ProblemType"]["Sparse"] == 1
              lrvw = kernel["LocalReadVectorWidth"] // (2 if sparseA else 1)
              wlr = lrvw//kernel["MIInputPerThreadA"]
              wlr = 1 if wlr == 0 else wlr
              if kernel["ProblemType"]["Sparse"] and lrvw < kernel["MIInputPerThreadA"]:
                if not sparseA:
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"])
                else:
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*lrvw//kernel["MIInputPerThreadA"])
              elif (self.states.localReadDoCntA)%(wlr):
                offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * kernel["MIInputPerThreadA"]
              else:
                isTF32EmuGfx950 = kernel["UseF32XEmulation"] and  (kernel["MatrixInstK"] >= 32 or (kernel["MatrixInstM"] == 32 and kernel["MatrixInstK"] == 16))
                if kernel["UnrollMajorLDSA"] == False and kernel["MatrixInstK"] >= 64 or isTF32EmuGfx950: # Special handling for new f8 mfmas and tf32
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"])
                else:
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*lrvw//kernel["MIInputPerThreadA"]-kernel["MIInputPerThreadA"]*(lrvw//kernel["MIInputPerThreadA"]-1))
                if sparseA:
                  offsetInc //= 2
            elif tc == "Metadata":
              lrvw = kernel["LocalReadVectorWidth"] // 8
              if lrvw < kernel["MIInputPerThreadMetadata"]:
                offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*lrvw//kernel["MIInputPerThreadMetadata"]) // 4
              elif (self.states.localReadDoCntMetadata)%(lrvw//kernel["MIInputPerThreadMetadata"]):
                offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * kernel["MIInputPerThreadMetadata"]
              else:
                offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*lrvw//kernel["MIInputPerThreadMetadata"]-kernel["MIInputPerThreadMetadata"]*(lrvw//kernel["MIInputPerThreadMetadata"]-1))
                offsetInc //= 8
            else:
              sparseB = kernel["ProblemType"]["Sparse"] == 2
              lrvw = kernel["LocalReadVectorWidth"] // (2 if sparseB else 1)
              wlr = lrvw//kernel["MIInputPerThreadB"]
              wlr = 1 if wlr == 0 else wlr
              #if (self.states.localReadDoCntB)%(lrvw//kernel["MIInputPerThreadB"]):
              if kernel["ProblemType"]["Sparse"] and lrvw < kernel["MIInputPerThreadB"]:
                if not sparseB:
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"])
                else:
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*lrvw//kernel["MIInputPerThreadB"])
              elif (self.states.localReadDoCntB)%(wlr):
                offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * kernel["MIInputPerThreadB"]

              else:
                isTF32EmuGfx950 = kernel["UseF32XEmulation"] and  (kernel["MatrixInstK"] >= 32 or (kernel["MatrixInstM"] == 32 and kernel["MatrixInstK"] == 16))
                if kernel["UnrollMajorLDSB"] == False and kernel["MatrixInstK"] >= 64 or isTF32EmuGfx950: # Special handling for new f8 mfmas and tf32
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"])
                else:
                  offsetInc = (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*lrvw//kernel["MIInputPerThreadB"]-kernel["MIInputPerThreadB"]*(lrvw//kernel["MIInputPerThreadB"]-1))

                if sparseB:
                  offsetInc //= 2
        else:
          # dot2
          offsetInc = self.states.lrvwUnrollA * kernel["NumWaveSplitK"] if kernel["UseDotInstruction"] else (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad)
        tP["localReadOffset"] += offsetInc
        module.addComment0("N/A, lro->%d" % tP["localReadOffset"])
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          module.addComment0("self.localReadDoCntA %d self.localReadDoCntB %d self.localReadDoCntMetadata %d" % (self.states.localReadDoCntA,self.states.localReadDoCntB, self.states.localReadDoCntMetadata))
        else:
          module.addComment0("self.localReadDoCntA %d self.localReadDoCntB %d" % (self.states.localReadDoCntA,self.states.localReadDoCntB))
      else:
        inc = (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad)
        numLra = 0
        if tP["isA"]:
          numLra = self.states.a.numVgprLocalReadAddr
        elif tP["isB"]:
          numLra = self.states.b.numVgprLocalReadAddr
        elif tP["isM"]:
          numLra = self.states.m.numVgprLocalReadAddr
        for i in range(numLra):
          module.add(VAddCOU32(
              dst=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], i)), \
              dst1=VCC(), \
              src0=hex(inc), \
              src1=vgpr("LocalReadAddr%s+%u"%(tP["tensorChar"], i)), \
              comment="lr%s += %u ((MT+Pad)*bpe)"%(tP["tensorChar"], inc) ))

    return module

  ##############################################################################
  # Local Read: Do It A/B
  # iui = Inner Unroll Idx
  # uIdx - Unroll Idx
  # epsi = expand pointer swap index. Only used for PAP
  ##############################################################################
  def localReadDo(self, kernel, bufferIdx, iui, epsi, tP):

    if not self.do["LocalRead%s" % tP["tensorChar"]]:
      imod = Module("LocalReadDo%s_I%s" % (tP["tensorChar"], iui))
      pack = Module("pack%s_I%s" % (tP["tensorChar"], iui))
      return imod, pack

    component = Component.LocalRead.find(self)
    if component:
      return component(self, kernel, bufferIdx, iui, epsi, tP)

  ##############################################################################
  # Save the local read pointers, for example when creating a duplicated
  # optimized path (like optNLL)
  ##############################################################################
  def saveLocalPointers(self, kernel, tPA, tPB):
    tPA["savedLocalReadOffset"] = tPA["localReadOffset"]
    tPB["savedLocalReadOffset"] = tPB["localReadOffset"]
    tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      tPM["savedLocalReadOffset"] = tPM["localReadOffset"]
    self.states.savedLocalReadDoCntA = self.states.localReadDoCntA
    self.states.savedLocalReadDoCntB = self.states.localReadDoCntB
    self.states.savedLocalReadDoCntMetadata = self.states.localReadDoCntMetadata
    if kernel["ExpandPointerSwap"]:
      tPA["savedLocalWriteSwapByteOffset"] = tPA["localWriteSwapByteOffset"]
      tPB["savedLocalWriteSwapByteOffset"] = tPB["localWriteSwapByteOffset"]
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        tPM["savedLocalWriteSwapByteOffset"] = tPM["localWriteSwapByteOffset"]
  ##############################################################################
  # Restore the saved local read pointers
  # Must be paired with an earlier call to savePointers
  ##############################################################################
  def restoreLocalPointers(self, kernel, tPA, tPB):
    tPA["localReadOffset"] = tPA["savedLocalReadOffset"]
    tPB["localReadOffset"] = tPB["savedLocalReadOffset"]
    tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
      tPM["localReadOffset"] = tPM["savedLocalReadOffset"]
    self.states.localReadDoCntA = self.states.savedLocalReadDoCntA
    self.states.localReadDoCntB = self.states.savedLocalReadDoCntB
    self.states.localReadDoCntMetadata = self.states.savedLocalReadDoCntMetadata
    if kernel["ExpandPointerSwap"]:
      tPA["localWriteSwapByteOffset"] = tPA["savedLocalWriteSwapByteOffset"]
      tPB["localWriteSwapByteOffset"] = tPB["savedLocalWriteSwapByteOffset"]
      if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        tPM["localWriteSwapByteOffset"] = tPM["savedLocalWriteSwapByteOffset"]
  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    component = Component.ShiftVectorComponents.find(self)
    if component:
      return component(self, kernel, tP)

  ##############################################################################
  # computeStoreSrd
  # Add tile assignment fields to store srd
  # This is based on WG not the WI/TT assignment
  ##############################################################################
  def computeStoreSrdStart(self, kernel, srdTcList: list, sgprBpeList = [], useSize: list = [], noMultipleBuffer = False):
    module = Module("computeStoreSrdStart")

    if useSize:
      assert len(srdTcList) == len(useSize)
    else:
      useSize = [False for _ in srdTcList]

    with self.allocTmpSgpr(3) as tmpSgprInfo:
      tmpS0 = tmpSgprInfo.idx
      tmpS1 = tmpS0+1
      wgMT1 = tmpS0+2

      # Compute and save wg1*MT1 - the element offset that is top of the macro-tile in output space
      assert kernel["BufferStore"]
      module.addSpaceLine()
      module.add(SMulI32(
          dst=sgpr(wgMT1), \
          src0="MT1", \
          src1=sgpr("WorkGroup1"), \
          comment="<- wg1*MT1"))

      # Overall strategy is to set the SRD to the top-left of the macro-tile.
      # TT offsets are from this base (and include the column)

      # In non-packed mode:
      # higher-order tensor dims are static since this kernel operates within
      # the 2D Tensor formed by Index0 and Indexa.
      # Index0 and Index1 vary for each work-item (aka 'dynamic') so roll these into the VGPR

      # In packed mode:
      # Higher-order dimensions may be packed into coord0 / coord1 - see rowstart calculation below

      # Walk through addressing components (each tensor index) in C
      # For static dims add to SrdC / SrdD to compute a new base.
      # For dynamic (based on TT assignment) - save in coutRowPtrD in computeStoreVgprs,
      # which saves the TT assignment for each WI scaled by StrideC0
      # TODO - future opportunities for store vgpr and other optimization
      #  - coutRowPtrD and tid1 are strongly related - can we merge or remove one of these?
      # Packed follows same philosophy but may have more vector components
      indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
      numDim = len(indices)
      addrSrcSgpr = "Address" # use "Address" only for the first iteration
      for i in range(1, numDim):
        if i == kernel["ProblemType"]["Index0"]:
          # Used if the output is transposed?
          addToSrd = False
        elif i == kernel["ProblemType"]["Index1"] and len(kernel["PackedC1IndicesX"]) == 1:
          coord = sgpr(wgMT1)
          addToSrd = True
        elif i != kernel["ProblemType"]["Index0"] and i != kernel["ProblemType"]["Index1"] and not isPackedIndex(kernel, i):
          # group index, this is higher-order Tensor dimension, just add to SRD base:
          isStridedBuffer = kernel["ProblemType"]["StridedBatched"] or kernel["_GlobalAccumulation"]
          coord = sgpr("WorkGroup2") if isStridedBuffer else None
          addToSrd = True if isStridedBuffer else False
        else:
          # could be packed higher-order index, just ignore
          coord = None
          addToSrd = False

        if not sgprBpeList:
          sgprBpeList = [""] * len(srdTcList)
        assert len(srdTcList) == len(sgprBpeList)
        if addToSrd:
          for mat, sgprBpe, us in zip(srdTcList, sgprBpeList, useSize):
            bpe = self.states.bpeCinternal if mat =="Bias" else (self.states.bpeE if mat == "E" else self.states.bpeCexternal)
            bpe = int(self.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters()) if kernel["_GlobalAccumulation"]  == 'MultipleBuffer'  and mat =="C" else bpe
            bpe = sgpr(sgprBpe) if sgprBpe else log2(bpe)  # sgprBpe cannot be 0
            # These are constant across all workitems, just add to the SRD:
            if us:
              if i == 0:
                # Skip cause stride = 1 if use size instead
                continue
              if i > 1:
                strideC0 = "Size%s"%(INDEX_CHARS[0])
                strideC1 = "Size%s"%(INDEX_CHARS[1])
                module.add(SMulI32(dst=sgpr(tmpS0), src0=sgpr(strideC0), src1=sgpr(strideC1)))
                for x in range(2, i - 1):
                  strideC = "Size%s"%(INDEX_CHARS[x])
                  module.add(SMulI32(dst=sgpr(tmpS0), src0=sgpr(tmpS0), src1=sgpr(strideC)))
                module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(tmpS0), comment="Scale%s %s by Stride"%(mat, coord)))
              else:
                strideC = "Size%s"%(INDEX_CHARS[i-1])
                module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(strideC), comment="Scale%s %s by Stride"%(mat, coord)))
            else:
              strideC = "Stride%s%s"%(mat, self.states.indexChars[i])
              module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(strideC), comment="Scale%s %s by Stride"%(mat, coord)))
            module.add(SLShiftLeftB64(dst=sgpr(tmpS0,2), src=sgpr(tmpS0,2), shiftHex=bpe, comment="scale by bpe"))
            module.add(SAddU32(dst=sgpr("Srd%s+0"%mat), src0=sgpr("%s%s+0"%(addrSrcSgpr, mat)), src1=sgpr(tmpS0), comment="add lo to SRD"))
            module.add(SAddCU32(dst=sgpr("Srd%s+1"%mat), src0=sgpr("%s%s+1"%(addrSrcSgpr, mat)), src1=sgpr(tmpS1), comment="add hi to SRD"))

          module.addSpaceLine()

          addrSrcSgpr = "Srd" # update src Sgpr for the second or later iterations

    if noMultipleBuffer:
      return module

    gsuComponent = Component.GSU.find(self)
    module.add(gsuComponent.computeStoreSrdStart(self, kernel))

    skComponent = Component.StreamK.find(self)
    module.add(skComponent.computeStoreSrdStart(self, kernel))

    for cdir in (0,1):
      indices = kernel["PackedC%uIndicesX"%cdir]
      packedSizes = "PackedSize%u"%cdir
      if len(indices) > 1:
        for i,idx in enumerate(indices[1:]):
          if i==0:
            module.add(SMulI32(dst=sgpr(packedSizes), src0=self.sizeRef(indices[0]), \
                      src1=self.sizeRef(idx), comment="first packed size"))
          else:
            module.add(SMulI32(dst=sgpr(packedSizes), src0=sgpr(packedSizes), \
                      src1=self.sizeRef(idx), comment="first packed size"))

    return module

  ##############################################################################
  # computeStoreVgprs
  # Compute workitem/TT offsets in VGPRS
  # and coord0/coord1
  ##############################################################################
  def computeStoreVgprs(self, kernel, divisor=None, tid0Scale=None, tid1Scale=None):
    module = Module("computeStoreVgprs")
    module.addComment0("computeStoreVgprs")
    component = Component.ComputeStoreVgprs.find(self)
    if component:
      if kernel["EnableMatrixInstruction"]:
        module.add(component(self, kernel))
      else:
        module.add(component(self, kernel, divisor, tid0Scale, tid1Scale))
    return module

  ##############################################################################
  # globalWriteWorkGroupInit:
  ##############################################################################

  def SrdTDInit(self, kernel):
    module = Module("SrdTDInit")
    tmpspgr0 = self.sgprPool.checkOut(1)
    tmpspgr = self.sgprPool.checkOutAligned(2, 4, preventOverflow=False)

    module.addComment0("calculate SrdTD address")

    module.add(SMovB32(dst=sgpr("SrdTD+2"), src="BufferOOB"))
    module.add(SMovB32(dst=sgpr("SrdTD+3"), src="Srd127_96", comment="Set bits 127_96 in post-loop SRD"))

    module.add(SMulI32(dst=sgpr(tmpspgr0), src0="MT1", src1=sgpr("WorkGroup1"), comment=""))
    module.add(SMulHIU32(dst=sgpr(tmpspgr+1), src0=sgpr(tmpspgr0), src1=sgpr("StrideD1J"), comment=""))
    module.add(SMulI32(dst=sgpr(tmpspgr+0), src0=sgpr(tmpspgr0), src1=sgpr("StrideD1J"), comment=""))

    bpe = int(self.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters()) # self.states.bpeCinternal
    module.add(SLShiftLeftB64(dst=sgpr(tmpspgr,2), src=sgpr(tmpspgr,2), shiftHex=log2(bpe), comment="scale by bpe"))

    module.add(SAddU32(dst=sgpr("SrdTD+0"), src0=sgpr("AddressTD+0"), src1=sgpr(tmpspgr+0), comment="add lo to SRTD" ))
    module.add(SAddCU32(dst=sgpr("SrdTD+1"), src0=sgpr("AddressTD+1"), src1=sgpr(tmpspgr+1), comment="add hi to SRTD" ))

    module.add(SMulHIU32(dst=sgpr(tmpspgr+1), src0=sgpr("WorkGroup2"), src1=sgpr("StrideDK"), comment=""))
    module.add(SMulI32(dst=sgpr(tmpspgr+0), src0=sgpr("WorkGroup2"), src1=sgpr("StrideDK"), comment=""))

    module.add(SLShiftLeftB64(dst=sgpr(tmpspgr,2), src=sgpr(tmpspgr,2), shiftHex=log2(bpe), comment="scale by bpe"))

    module.add(SAddU32(dst=sgpr("SrdTD+0"), src0=sgpr("SrdTD+0"), src1=sgpr(tmpspgr+0), comment="add lo to SRTD" ))
    module.add(SAddCU32(dst=sgpr("SrdTD+1"), src0=sgpr("SrdTD+1"), src1=sgpr(tmpspgr+1), comment="add hi to SRTD" ))

    self.sgprPool.checkIn(tmpspgr0)
    self.sgprPool.checkIn(tmpspgr)

    return module

  def globalWriteWorkGroupInit(self, kernel):
    module = Module("globalWriteWorkGroupInit")
    if kernel["BufferStore"]:
      module.add(self.allocPostLoopSrd("D"))
      module.add(self.allocPostLoopSrd("C"))
      sgprBpeList = ["GSULog2BpeC", "GSULog2BpeD"] if kernel["GlobalSplitU"] != 0 else []

      # Set BPE based on reduction algorithm
      if kernel["StreamK"] == 3:
        sgprLog2BpeC = self.sgprPool.checkOut(1, preventOverflow=False)
        sgprLog2BpeD = self.sgprPool.checkOut(1, preventOverflow=False)

        module.add(SMovB32(dst=sgpr(sgprLog2BpeC), src=log2(self.states.bpeCexternalGSU1)))
        module.add(SMovB32(dst=sgpr(sgprLog2BpeD), src=log2(self.states.bpeCexternalGSU1)))

        bpeDoneLabel = Label(label=self.labels.getNameInc("BPEDone"), comment="")

        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=bpeDoneLabel.getLabelName(), comment="If synchronizer, use regular output BPE"))
        module.add(SCmpEQU32(src0=sgpr("skTiles"), src1=1, comment="split == 1 ?"))
        module.add(SCBranchSCC1(labelName=bpeDoneLabel.getLabelName(), comment="If split == 1, use reguler output BPE"))

        # BPE for parallel reduction
        module.add(SMovB32(dst=sgpr(sgprLog2BpeC), src=log2(int(self.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters()))))
        module.add(SMovB32(dst=sgpr(sgprLog2BpeD), src=log2(self.states.bpeCinternal)))

        module.add(bpeDoneLabel)
        sgprBpeList = [sgprLog2BpeC, sgprLog2BpeD]

      module.add(self.computeStoreSrdStart(kernel, ["C", "D"], sgprBpeList=sgprBpeList))
      if kernel["GlobalSplitU"] != 0:
        module.add(self.undefineSgpr("GSULog2BpeC"))
      if kernel["StreamK"] == 0:
        module.add(self.undefineSgpr("AddressC"))

      if kernel["StreamK"] == 3:
        self.sgprPool.checkIn(sgprLog2BpeD)
        self.sgprPool.checkIn(sgprLog2BpeC)
    return module

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    module = Module("localSplitUGlobalWriteIndices")

    # Add LSU Offset back
    packedC1 = kernel["PackedC1IndicesX"]
    strideC1 = "StrideC%s" % (self.states.indexChars[packedC1[0]])
    strideD1 = "StrideD%s" % (self.states.indexChars[packedC1[0]])
    wave_id = self.vgprPool.checkOut(1, "tmpWaveID")
    tmpVgpr1 = self.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
    tmpVgpr1Res = ContinuousRegister(tmpVgpr1, 2)
    module.add(vectorStaticDivide(wave_id, "Serial", kernel["WavefrontSize"], tmpVgpr1Res))
    numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
    module.add(vectorStaticDivide(wave_id, wave_id, numWaves, tmpVgpr1Res))

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      if self.LSUValidOffset0 > 0:
        module.add(SMovB32(dst=sgpr(tmpSgpr), \
            src=hex(self.LSUValidOffset0), comment="a valid offset"))
        module.add(VMulLOU32(dst=vgpr(tmpVgpr1), src0=vgpr(wave_id), src1=sgpr(tmpSgpr), comment="wave LSU offset"))
        module.add(VAddU32(dst=vgpr(self.vgprs.coord0), src0=vgpr(tmpVgpr1), src1=vgpr(self.vgprs.coord0), comment="coord0 += LSU offset0"))
      else:
        module.addComment0("valid offset coord0 is zero.")

      if self.LSUValidOffset1 > 0:
        module.add(SMovB32(dst=sgpr(tmpSgpr), \
            src=hex(self.LSUValidOffset1), comment="a valid offset"))
        module.add(VMulLOU32(dst=vgpr(tmpVgpr1), src0=vgpr(wave_id), src1=sgpr(tmpSgpr), comment="wave LSU offset"))
        module.add(VAddU32(dst=vgpr(self.vgprs.coord1), src0=vgpr(tmpVgpr1), src1=vgpr(self.vgprs.coord1), comment="coord1 += LSU offset1"))
        module.add(VAddU32(dst=vgpr(self.vgprs.coord1InMT), src0=vgpr(tmpVgpr1), src1=vgpr(self.vgprs.coord1InMT), comment="coord1InMT += LSU offset1"))

        # this code is from CouputeStoreVgprs. coord 1 : offset part
        packedC1 = kernel["PackedC1IndicesX"]
        strideC1 = "StrideC%s" % (self.states.indexChars[packedC1[0]])
        strideD1 = "StrideD%s" % (self.states.indexChars[packedC1[0]])
        module.add(VMulLOU32(dst=vgpr(self.vgprs.cinRowPtr), src0=vgpr(self.vgprs.coord1InMT), src1=sgpr(strideC1), comment=" offset 1"))
        module.add(VMulLOU32(dst=vgpr(self.vgprs.coutRowPtrD), src0=vgpr(self.vgprs.coord1InMT), src1=sgpr(strideD1), comment=" offset 1"))
        if kernel["ProblemType"]["UseE"] and ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0):
            module.add(VMovB32(dst=vgpr(self.vgprs.coutRowPtrE), src=vgpr(self.vgprs.coord1InMT), comment=" save offset 1 for E"))
        if self.vgprs.coutRowPtrBias != -1:
            index = packedC1[0] - 1
            strideW1 = "Size%s" % "I" if index == 0 else ("J" if index == 1 else (self.states.indexChars[index]))
            module.add(VMulLOU32(dst=vgpr(self.vgprs.coutRowPtrBias), src0=vgpr(self.vgprs.coord1InMT), src1=sgpr(strideW1), comment=" offset 1"))
      else:
        module.addComment0("valid offset coord1 is zero.")

    self.vgprPool.checkIn(tmpVgpr1)
    self.vgprPool.checkIn(wave_id)
    self.vgprPool.checkIn(self.lsuCoordOffset)
    self.vgprPool.checkIn(self.vgprs.coord0InMT)
    self.vgprPool.checkIn(self.vgprs.coord1InMT)

    if kernel["BufferStore"]:
      #print "----AddressC-LocalSplitU"
      #print self.vgprPool.state()
      self.vgprs.addrE    = -1
      self.vgprs.addrD    = -1
      self.vgprs.addrC    = -1
      self.vgprs.addrBias = -1
      self.vgprs.addrScaleAVec = -1
      self.vgprs.addrScaleBVec = -1
      self.vgprs.addrScaleAlphaVec = -1
    else:
      self.vgprs.addrD = self.vgprPool.checkOut(2)
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+0), \
          src=sgpr("AddressD+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+1), \
          src=sgpr("AddressD+1"), \
          comment="sgpr -> vgpr"))
      self.vgprs.addrC = self.vgprPool.checkOut(2)
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+0), \
          src=sgpr("AddressC+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+1), \
          src=sgpr("AddressC+1"), \
          comment="sgpr -> vgpr"))

      if kernel["GlobalSplitU"] != 0:
        gsuLabel = Label(label=self.labels.getNameInc("GSU"), comment="")
        with self.allocTmpSgpr(1) as tmpSgprGSU:
          module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
          module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
        module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="branch if GSU != 1"))
      if kernel["ProblemType"]["UseE"]:
        self.vgprs.addrE = self.vgprPool.checkOut(2, 'addrE')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrE+0), \
            src=sgpr("AddressE+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrE+1), \
            src=sgpr("AddressE+1"), \
            comment="sgpr -> vgpr"))
      if self.states.useBias == DataDirection.READ:
        self.vgprs.addrBias = self.vgprPool.checkOut(2, 'addrBias')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+0), \
            src=sgpr("AddressBias+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+1), \
            src=sgpr("AddressBias+1"), \
            comment="sgpr -> vgpr"))
      if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
        self.vgprs.addrScaleAVec = self.vgprPool.checkOut(2, 'addrScaleAVec')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAVec+0), \
            src=sgpr("AddressScaleA+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAVec+1), \
            src=sgpr("AddressScaleA+1"), \
            comment="sgpr -> vgpr"))
        self.vgprs.addrScaleBVec = self.vgprPool.checkOut(2, 'addrScaleVVec')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleBVec+0), \
            src=sgpr("AddressScaleB+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleBVec+1), \
            src=sgpr("AddressScaleB+1"), \
            comment="sgpr -> vgpr"))
      if kernel["ProblemType"]["UseScaleAlphaVec"]:
        self.vgprs.addrScaleAlphaVec = self.vgprPool.checkOut(2, 'addrScaleAlphaVec')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAlphaVec+0), \
            src=sgpr("AddressScaleAlphaVec+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAlphaVec+1), \
            src=sgpr("AddressScaleAlphaVec+1"), \
            comment="sgpr -> vgpr"))
      if kernel["GlobalSplitU"] != 0:
        module.add(gsuLabel)

    return module

  ##############################################################################
  def allocPostLoopSrd(self, ch: str):
    module = Module("allocPostLoopSrd")
    # Buffer-load uses one base read pointer stored in the SRD - set it here:
    module.add(SMovB64(dst=sgpr("Srd%s+0"%ch, 2), src=sgpr("Address%s+0"%ch, 2), comment="init SRD base address" ))
    module.add(SMovB32(dst=sgpr("Srd%s+2"%ch), src="BufferOOB"))
    module.add(SMovB32(dst=sgpr("Srd%s+3"%ch), src="Srd127_96", comment="Set bits 127_96 in post-loop SRD"))
    module.addSpaceLine()
    return module

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    module = Module("notLocalSplitUGlobalWriteIndices")

    if kernel["EnableMatrixInstruction"]:
      module.add(self.computeStoreVgprs(kernel))
    else:
      module.add(self.computeStoreVgprs(kernel,
                                        divisor = kernel["SubGroup0"],
                                        tid0Scale = kernel["VectorWidthA"],
                                        tid1Scale = kernel["VectorWidthB"]))

    if kernel["BufferStore"]:
      #print "----AddressC-nonLSU-----"
      #print self.vgprPool.state()
      self.vgprs.addrE    = -1
      self.vgprs.addrD    = -1
      self.vgprs.addrC    = -1
      self.vgprs.addrBias = -1
      self.vgprs.addrScaleAVec = -1
      self.vgprs.addrScaleBVec = -1
      self.vgprs.addrScaleAlphaVec = -1
    else:
      self.vgprs.addrD = self.vgprPool.checkOut(2, 'addrD')
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+0), \
          src=sgpr("AddressD+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+1), \
          src=sgpr("AddressD+1"), \
          comment="sgpr -> vgpr"))
      self.vgprs.addrC = self.vgprPool.checkOut(2, 'addrC')
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+0), \
          src=sgpr("AddressC+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+1), \
          src=sgpr("AddressC+1"), \
          comment="sgpr -> vgpr"))

      if kernel["GlobalSplitU"] != 0:
        gsuLabel = Label(label=self.labels.getNameInc("GSU"), comment="")
        with self.allocTmpSgpr(1) as tmpSgprGSU:
          module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
          module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
        module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="branch if GSU != 1"))
      if kernel["ProblemType"]["UseE"]:
        self.vgprs.addrE = self.vgprPool.checkOut(2, 'addrE')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrE+0), \
            src=sgpr("AddressE+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrE+1), \
            src=sgpr("AddressE+1"), \
            comment="sgpr -> vgpr"))
      if self.states.useBias == DataDirection.READ:
        self.vgprs.addrBias = self.vgprPool.checkOut(2, 'addrBias')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+0), \
            src=sgpr("AddressBias+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+1), \
            src=sgpr("AddressBias+1"), \
            comment="sgpr -> vgpr"))
      if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
        self.vgprs.addrScaleAVec = self.vgprPool.checkOut(2, 'addrScaleAVec')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAVec+0), \
            src=sgpr("AddressScaleA+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAVec+1), \
            src=sgpr("AddressScaleA+1"), \
            comment="sgpr -> vgpr"))
        self.vgprs.addrScaleBVec = self.vgprPool.checkOut(2, 'addrScaleBVec')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleBVec+0), \
            src=sgpr("AddressScaleB+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleBVec+1), \
            src=sgpr("AddressScaleB+1"), \
            comment="sgpr -> vgpr"))
      if kernel["ProblemType"]["UseScaleAlphaVec"]:
        self.vgprs.addrScaleAlphaVec = self.vgprPool.checkOut(2, 'addrScaleAlphaVec')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAlphaVec+0), \
            src=sgpr("AddressScaleAlphaVec+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleAlphaVec+1), \
            src=sgpr("AddressScaleAlphaVec+1"), \
            comment="sgpr -> vgpr"))
      if kernel["GlobalSplitU"] != 0:
        module.add(gsuLabel)
    return module

  ##############################################################################
  # Release any resources used by the global write
  def cleanupGlobalWrite(self, kernel):
    self.vgprPool.checkIn(self.vgprs.coord0)
    self.vgprPool.checkIn(self.vgprs.coord1)

    if kernel["StoreRemapVectorWidth"]:
      self.vgprPool.checkIn(self.vgprs.storeRemapLW)
      self.vgprPool.checkIn(self.vgprs.storeRemapLR)
      self.vgprPool.checkIn(self.vgprs.storeRemapCoord0)
      self.vgprPool.checkIn(self.vgprs.storeRemapCoord1)
      self.vgprPool.checkIn(self.vgprs.storeRemapOffsetCoord1)
      for v in self.vgprs.storeRemapAS:
        self.vgprPool.checkIn(v)
    if kernel["BufferStore"]:
      self.vgprPool.checkIn(self.vgprs.cinRowPtr)
      self.vgprPool.checkIn(self.vgprs.coutRowPtrD)
      if self.vgprs.coutRowPtrE != -1:
        self.vgprPool.checkIn(self.vgprs.coutRowPtrE)
        self.vgprs.coutRowPtrE = -1
      if self.vgprs.coutRowPtrBias != -1:
        self.vgprPool.checkIn(self.vgprs.coutRowPtrBias)
        self.vgprs.coutRowPtrBias = -1
    if not kernel["BufferStore"]:
      self.vgprPool.checkIn(self.vgprs.addrD)
      self.vgprPool.checkIn(self.vgprs.addrC)
      if self.vgprs.addrE != -1:
        self.vgprPool.checkIn(self.vgprs.addrE)
        self.vgprs.addrE = -1
      if self.states.useBias == DataDirection.READ:
        self.vgprPool.checkIn(self.vgprs.addrBias)
        self.vgprs.addrBias = -1
      if self.vgprs.addrScaleAVec != -1:
        self.vgprPool.checkIn(self.vgprs.addrScaleAVec)
        self.vgprs.addrScaleAVec = -1
      if self.vgprs.addrScaleBVec != -1:
        self.vgprPool.checkIn(self.vgprs.addrScaleBVec)
        self.vgprs.addrScaleBVec = -1
      if self.vgprs.addrScaleAlphaVec != -1:
        self.vgprPool.checkIn(self.vgprs.addrScaleAlphaVec)
        self.vgprs.addrScaleAlphaVec = -1

  ##############################################################################
  # Return max global write vector width, in elements
  def maxGwvw(self, kernel):
    atomic = (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer' and kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel')

    if kernel["BufferStore"]:
      if atomic:
        return self.getVectorAtomicWidth(kernel)
      else:
        return 1000  # no limit
    else:
      if atomic:
        return 1  # flat vector atomic is not tested
      else:
        return 1000  # no limit

  def getVectorAtomicWidth(self, kernel):
    if kernel["ProblemType"]["DataType"].isHalf() and (not kernel["_GlobalAccumulation"]):
      return 2
    return 1

  ##############################################################################
  # Partition thread-tile into writeElements for store code
  # This function creates the writeElement mapping for full tiles
  # (ie non-edge cases)
  ##############################################################################
  def notLocalFullTileElements(self, kernel, edge):
    component = Component.NotLocalFullTileElements.find(self)
    if component:
      return component(self, kernel, edge)

  ##############################################################################
  # Store Remap: Local Write
  ##############################################################################
  def storeRemapAddLocalWrite(self, kernel, ss, addrCalc, srcVgpr):
    """
    Add localWrite for the element with addrCalc and srcVgpr.
    """

    module = Module("storeRemapAddLocalWrite srcVgpr %s"%str(srcVgpr))

    bps = self.states.bpeCexternal * ss.cfg.gwvw
    rpv = self.states.bpeCexternal * ss.cfg.gwvw / self.states.bpr

    addr0  = vgpr(self.vgprs.storeRemapLW)
    offset = addrCalc.coordOffset0 * self.states.bpeCexternal
    ds     = DSModifiers(offset=offset)

    if bps==1:
      module.add(DSStoreB8(dstAddr=addr0, src=vgpr(srcVgpr, rpv*4), \
                ds=ds, comment="storeRemap lw"))
    elif bps==2:
      module.add(DSStoreB16(dstAddr=addr0, src=vgpr(srcVgpr, rpv*2), \
                ds=ds, comment="storeRemap lw"))
    elif bps==4:
      module.add(DSStoreB32(dstAddr=addr0, src=vgpr(srcVgpr, rpv), \
                ds=ds, comment="storeRemap lw"))
    elif bps==8:
      module.add(DSStoreB64(dstAddr=addr0, src=vgpr(srcVgpr, rpv), \
                ds=ds, comment="storeRemap lw"))
    elif bps==16:
      module.add(DSStoreB128(dstAddr=addr0, src=vgpr(srcVgpr, rpv), \
                ds=ds, comment="storeRemap lw"))
    else:
      assert 0, "StoreRemap: bad bps!"

    return module

  ##############################################################################
  # Store Remap: Local Read and Global Write
  ##############################################################################
  def storeRemapAddStore(self, kernel, tmpVgpr, tmpS01, edge, StoreRemapLastBatch):
    module = Module("storeRemapAddStore")

    module.add(SWaitCnt(dscnt=0, comment="wait for LDS write"))

    numStoreInst = 0

    #Data exchange between different waves
    #Make sure LDS writes are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      # FIXME: Indent?
      module.add(SBarrier(comment="wait all lds write finished"))
    module.addSpaceLine()

    gwvw = kernel["StoreRemapVectorWidth"]
    nElements = kernel["MacroTile0"]*kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]//self.states.kernel["WavefrontSize"]

    bpe = self.states.bpeCexternal
    bps = bpe * gwvw
    rpe = self.states.bpeCexternal / self.states.bpr
    rpv = rpe * gwvw

    storeRegs = self.vgprs.storeRemapAS
    src = vgpr(self.vgprs.storeRemapLR)
    for rIdx, i in enumerate(range(0, nElements, gwvw)):
      offset = self.storeRemapLrOffset * bpe * (i//gwvw)
      ds  = DSModifiers(offset=offset)
      dst = vgpr(storeRegs[rIdx], rpv)
      if bps==4:
        module.add(DSLoadB32(dst=dst, src=src, ds=ds, comment="storeRemap lr"))
      elif bps==8:
        module.add(DSLoadB64(dst=dst, src=src, ds=ds, comment="storeRemap lr"))
      elif bps==16:
        module.add(DSLoadB128(dst=dst, src=src, ds=ds, comment="storeRemap lr"))
      else:
        assert 0, "StoreRemap: bad bps!"

    module.addSpaceLine()

    # Global Write
    addr1 = sgpr("SrdD", 4)
    packedD1 = kernel["PackedC1IndicesX"]
    strideD1 = "StrideD%s" % (self.states.indexChars[packedD1[0]])

    vTmp = self.vgprPool.checkOut(1, "SR Store temp addr0")
    addr0 = vgpr(vTmp)

    isGlc = bool(kernel["NonTemporalD"] & 0x1)
    isSlc = bool(kernel["NonTemporalD"] & 0x2)
    isNT  = bool(kernel["NonTemporalD"] & 0x4)
    if kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
      isGlc = True
      isSlc = True

    if not edge:
      for rIdx, i in enumerate(range(0, nElements, gwvw)):
        if i == 0:
          module.add(VMovB32(dst=addr0, src=vgpr(self.vgprs.storeRemapOffsetCoord1), comment="coord1"))
        else:
          currentStep = i//gwvw
          module.add(VAddU32(dst=addr0, src0=vgpr(self.vgprs.storeRemapOffsetCoord1), src1=(self.storeRemapNCPL * currentStep), comment="coord1 += nColPerLoad"))

        module.add(VMulLOU32(dst=addr0, src0=addr0, src1=sgpr(strideD1), comment="coord1 offset =  coord1 * StrideD"))
        module.add(VAddLShiftLeftU32(dst=addr0, src0=addr0, src1=vgpr(self.vgprs.storeRemapCoord0), shiftHex=hex(log2(bpe)), comment="global write D address"))

        dscnt = min((nElements-i)//gwvw - 1, 15)
        module.add(SWaitCnt(dscnt=dscnt, comment="wait for LDS read"))

        numStoreInst += 1

        module.add(self.chooseGlobalWrite(True, bps, storeRegs[rIdx], rpv, addr0, addr1, 0, glc=isGlc, slc=isSlc, nt=isNT, comment="store D StoreRemapVectorWidth"))

    else:
      tmpS23 = tmpS01+self.states.laneSGPRCount
      coord0 = tmpVgpr
      coord1 = coord0+1
      srvw = kernel["StoreRemapVectorWidth"]
      edgeVw = min(kernel["AssertFree0ElementMultiple"],kernel["StoreRemapVectorWidth"])
      bps = self.states.bpeCexternal * edgeVw
      rpv = self.states.bpeCexternal / self.states.bpr * edgeVw
      for rIdx, i in enumerate(range(0, nElements, srvw)):
        for vi in range (0, srvw, edgeVw):

          if vi == 0:
            dscnt = min((nElements-i)//srvw - 1, 15)
            module.add(SWaitCnt(dscnt=dscnt, comment="wait for LDS read"))

          sizeBoundary = [0,0]
          sizeBoundary[0] = \
              sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index0"])
          sizeBoundary[1] = \
              sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index1"])

          currentStep = i//srvw

          # calculate global coordination
          module.add(VAddU32(dst=vgpr(coord1), src0=vgpr(self.vgprs.storeRemapCoord1), src1=(self.storeRemapNCPL * currentStep), comment="coord1 += nColPerLoad"))
          module.add(VAddU32(dst=vgpr(coord0), src0=vgpr(self.vgprs.storeRemapCoord0), src1=vi, comment="coord0 += element index of load vector"))
          module.add(VAddU32(dst=addr0, src0=vgpr(self.vgprs.storeRemapOffsetCoord1), src1=(self.storeRemapNCPL * currentStep), \
                      comment="offset coord1 += nColPerLoad"))

          module.add(VCmpLtU32(dst=sgpr(tmpS01,self.states.laneSGPRCount), src0=vgpr(coord0), src1=sizeBoundary[0], comment="coord0 < size0" ))
          module.add(VCmpLtU32(dst=sgpr(tmpS23,self.states.laneSGPRCount), src0=vgpr(coord1), src1=sizeBoundary[1], comment="coord1 < size1" ))
          SAndBX = SAndB32 if self.states.kernel["WavefrontSize"] == 32 else SAndB64
          module.add(SAndBX(
                      dst=sgpr(tmpS23,self.states.laneSGPRCount),
                      src0=sgpr(tmpS01,self.states.laneSGPRCount),
                      src1=sgpr(tmpS23,self.states.laneSGPRCount), comment="in0 && in1" ))

          module.add(VMulLOU32(dst=addr0, src0=addr0, src1=sgpr(strideD1), comment="coord1 element offset =  coord1 * StrideD"))
          module.add(VAddLShiftLeftU32(dst=addr0, src0=addr0, src1=vgpr(coord0), shiftHex=hex(log2(bpe)), comment="scale to BPE"))
          module.add(VCndMaskB32(dst=addr0, src0=-1, src1=addr0, src2=sgpr(tmpS23,self.states.laneSGPRCount), comment="clip if OOB. offset" ))

          sumIdx = storeRegs[rIdx] + int(vi*rpe)
          numStoreInst += 1
          if bps == 2:
            module.add(self.chooseGlobalWrite(True, bpe, sumIdx, rpe, addr0, addr1, 0, glc=isGlc, slc=isSlc, nt=isNT, hi16=vi%2, comment="store D StoreRemapVectorWidth"))
          else:
            module.add(self.chooseGlobalWrite(True, bps, sumIdx, rpv, addr0, addr1, 0, glc=isGlc, slc=isSlc, nt=isNT, comment="store D StoreRemapVectorWidth"))

          if bps == 1:
            module.add(VAShiftRightI32(dst=vgpr("ValuC+%u"%sumIdx), shiftHex=8, src=vgpr("ValuC+%u"%sumIdx), comment=" shift 1 byte" ))

    module.addSpaceLine()
    self.vgprPool.checkIn(vTmp)

    #Data exchange between different waves
    #Make sure LDS reads are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      module.add(SBarrier(comment="wait all lds read finished"))

    return module, numStoreInst

  ##############################################################################
  # Store remap compute vgprs:
  ##############################################################################
  def storeRemapComputeStoreVgprs(self, kernel):
    module = Module("storeRemapComputeStoreVgprs")
    module.addComment0("Store Remap Local Write address")

    with self.allocTmpSgpr(2) as tmpSgprInfo:
      tmpS0 = tmpSgprInfo.idx
      wgMT1 = tmpS0 + 1

      wg0="WorkGroup0"
      wg1="WorkGroup1"

      tid0 = self.vgprPool.checkOut(1, "SR coord0")
      tid1 = self.vgprPool.checkOut(1, "SR coord1")
      coord1Offset = self.vgprPool.checkOut(1, "SR coord1 offset")
      storeRemapLW = self.vgprPool.checkOut(1, "SR local write")
      storeRemapLR = self.vgprPool.checkOut(1, "SR local read")

      tmpV0 = self.vgprPool.checkOut(5, "tmpV0")
      waveCoord0 = tmpV1 = tmpV0+1
      ldsStride = tmpV0+2
      coord0 = tmpV0+3
      waveCoord1 = tmpV0+4

      gwvw = kernel["StoreRemapVectorWidth"]
      ldsPad = max(kernel["StoreRemapVectorWidth"],kernel["MIOutputVectorWidth"])

      #calculate local write Address: v[vgprLocalWriteAddrC]
      module.add(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.states.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], \
        ContinuousRegister(tmpV0, 2)))

      module.add(VMulLOU32(dst=vgpr(waveCoord1),
                src0=hex(kernel["MatrixInstN"]), src1=vgpr(tid1), comment="coord1 offset of LDS for each Wave"))
      module.add(VAndB32(dst=vgpr(tid1),
                src0=hex(kernel["MatrixInstN"]-1), src1=vgpr("Serial"), comment="coord1 offset of LDS for each thread"))
      module.add(VAddU32(dst=vgpr(tid1), src0=vgpr(waveCoord1), src1=vgpr(tid1), comment="coord1 offset in MacroTile"))
      module.add(VMovB32(dst=vgpr(ldsStride), src=hex(kernel["MacroTile0"]+ldsPad), \
                comment="lds stride = MT0 + PAD"))
      module.add(VMulLOU32(dst=vgpr(tmpV0), src0=vgpr(tid1), src1=vgpr(ldsStride), \
                comment="lds coord1 offset = Col-id* lds stride"))

      module.add(vectorStaticDivideAndRemainder(waveCoord0, tid0, tid0, self.states.kernel["WavefrontSize"], ContinuousRegister(tmpV0, 2)))
      module.add(VLShiftRightB32(dst=vgpr(coord0),
                shiftHex=hex(log2(kernel["MatrixInstN"])), src=vgpr(tid0), \
                comment="tid / matrixInstN"))

      if kernel["MIOutputVectorWidth"] > 1:
        module.add(VLShiftLeftB32(dst=vgpr(coord0), shiftHex=hex(log2(kernel["MIOutputVectorWidth"])), src=vgpr(coord0), \
                comment="lds coord0 offset *= 4 (each thread hold 4 element)"))

      module.add(VMadU32U24(dst=vgpr(coord0), src0=(kernel["MatrixInstM"]*kernel["MatrixInstBM"]), src1=vgpr(waveCoord0), src2=vgpr(coord0), \
                comment="coord0 += waveCoord0 * wave M shape(blockM*MiM)"))

      module.add(VAddLShiftLeftU32(
        dst=vgpr(storeRemapLW), \
        src0=vgpr(tmpV0), \
        src1=vgpr(coord0), \
        shiftHex=sgpr("GSULog2BpeD"), \
        comment="local write C address"))

      module.addSpaceLine()
      # calculate local read address : v[vgprLocalReadAddrC]

      module.addComment0("Store Remap Local Read address")

      module.add(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.states.kernel["WavefrontSize"], \
        ContinuousRegister(tmpV0, 2)))
      module.add(VMulLOU32(dst=vgpr(waveCoord1),
                src0=hex(kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]), src1=vgpr(tid1), comment="coord1 offset of LDS for each Wave"))

      nThreadPerCol = kernel["MacroTile0"] // gwvw
      nColPerLoad = self.states.kernel["WavefrontSize"] // nThreadPerCol
      self.storeRemapLrOffset = (kernel["MacroTile0"]+ldsPad) * nColPerLoad
      self.storeRemapNCPL = nColPerLoad

      module.add(VLShiftRightB32(dst=vgpr(tmpV1),\
                shiftHex=hex(log2(nThreadPerCol)), src=vgpr(tid0), \
                comment="tid / nThreadPerCol"))
      module.add(VAddU32(dst=vgpr(coord1Offset), src0=vgpr(waveCoord1), src1=vgpr(tmpV1), comment="coord1 offset in MacroTile"))
      module.add(VMulLOU32(dst=vgpr(tmpV0), src0=vgpr(coord1Offset), src1=vgpr(ldsStride), \
                comment="lds coord1 offset = Col-id* lds stride"))

      module.add(VAndB32(dst=vgpr(coord0), src0=hex(nThreadPerCol-1), src1=vgpr(tid0),
                comment="coord0 offset of LDS for each thread"))
      module.add(VLShiftLeftB32(dst=vgpr(coord0), shiftHex=hex(log2(gwvw)), src=vgpr(coord0), \
                comment="lds coord0 offset *= gwvw (each thread hold gwvw element)"))

      module.add(VAddLShiftLeftU32(
                dst=vgpr(storeRemapLR), \
                src0=vgpr(tmpV0), \
                src1=vgpr(coord0), \
                shiftHex=sgpr("GSULog2BpeD"), \
                comment="local read C address"))
      module.addSpaceLine()

      # calculate global write coord0 and coord1
      module.addComment0("Store Remap global write coord0 and coord1")
      module.add(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.states.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], \
        ContinuousRegister(tmpV0, 2)))

      ColsPerBlockShape = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

      module.add(VMulLOU32(dst=vgpr(waveCoord1), src0=hex(ColsPerBlockShape), src1=vgpr(tid1),
                comment="coord1 offset of global memory for each Wave"))

      module.add(vectorStaticDivideAndRemainder(tid1, tid0, tid0, self.states.kernel["WavefrontSize"], \
        ContinuousRegister(tmpV0, 2)))
      module.add(VMadU32U24(dst=vgpr(waveCoord1), src0=(kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]), src1=vgpr(tid1), src2=vgpr(waveCoord1), \
                comment="waveCoord1 += waveCoord0 * MiN / WaveGroupM"))

      module.add(VLShiftRightB32(dst=vgpr(tmpV1), shiftHex=hex(log2(nThreadPerCol)), src=vgpr(tid0), \
                comment="tid / nThreadPerCol"))

      module.add(VAddU32(dst=vgpr(coord1Offset), src0=vgpr(waveCoord1), src1=vgpr(tmpV1), comment="coord1 offset in MacroTile"))

      module.add(SMulI32(dst=sgpr(tmpS0), src0=hex(kernel["MacroTile0"]), src1=sgpr(wg0), comment="%s = wg0*MT0"%sgpr(tmpS0)))

      module.add(VAddCOU32(dst=vgpr(tid0), dst1=VCC(), src0=sgpr(tmpS0), src1=vgpr(coord0), comment="coord0 = coord0 + wg0 * MT0"))

      module.add(SMulI32(dst=sgpr(wgMT1), src0="MT1", src1=sgpr(wg1), comment="<- wg1*MT1"))
      module.add(VAddCOU32(dst=vgpr(tid1), dst1=VCC(), src0=sgpr(wgMT1), src1=vgpr(coord1Offset), comment="coord1 = tid1*VW + wg1*MT1"))

      module.addSpaceLine()

      module.add(self._syncThreads(kernel, "StoreRemap Start"))

      self.vgprs.storeRemapLW = storeRemapLW  #local write
      self.vgprs.storeRemapLR = storeRemapLR  #local read
      self.vgprs.storeRemapCoord0 = tid0      #global coord0
      self.vgprs.storeRemapCoord1 = tid1      #global coord1
      self.vgprs.storeRemapOffsetCoord1 = coord1Offset #offset coord1

      self.vgprPool.checkIn(tmpV0)

      nElements = kernel["MacroTile0"]*kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]//self.states.kernel["WavefrontSize"]
      rpe = self.states.bpeCexternal / self.states.bpr
      rpv = rpe * gwvw
      self.vgprs.storeRemapAS = []
      for i in range(0, nElements, gwvw):
        self.vgprs.storeRemapAS.append(self.vgprPool.checkOutAligned(int(rpv), int(rpv), "store element d"))
    return module

  ##############################################################################
  # Not LocalSplitU: Global Write
  # Determine write batching pattern
  # element() specifies TT 'coordinate' to write
  # vectorWidths specifies width of vector to store
  # TODO - why does this use VectorWidth to control store width?  Could be GlobalWriteVectorWidth?
  #
  # Function creates one mapping for full tiles and one for edge tiles,
  # then calls globalWriteElements to generate the code for the new tiles.
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel, tPA, tPB):
    if not self.do["PostLoop"]: return ""
    elements   = [[] for y in range(2)] # 2D array for Full, Edge
    elements_1 = [[] for y in range(2)] # 2D array for Full, Edge

    (fullVw, elements[False], fullVw_1, elements_1[False]) = self.notLocalFullTileElements(kernel, False)
    (edgeVw, elements[True], edgeVw_1, elements_1[True] )  = self.notLocalFullTileElements(kernel, True)

    # print("len(elements[False])= ", len(elements[False]))
    # print("len(elements[True])= ", len(elements[True]))
    vectorWidths   = [fullVw, edgeVw]
    vectorWidths_1 = [fullVw, edgeVw_1]

    noGSUBranch = (kernel["GlobalSplitU"] == 0 and kernel["StreamK"] != 3)
    module = Module("notLocalSplitUGlobalWrite")
    module.add(self.globalWriteElements(kernel, tPA, tPB, vectorWidths, vectorWidths_1, elements, elements_1, noGSUBranch=noGSUBranch))

    self.cleanupGlobalWrite(kernel)

    return module

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel, tPA, tPB):
    if not self.do["PostLoop"]: return ""

    elements_0 = [[] for y in range(2)]
    elements_1 = [[] for y in range(2)]
    elements_f0  = [[] for y in range(2)]
    elements_f1  = [[] for y in range(2)]
    (fullVw, elements_0[False], fullVw_1, elements_1[False]) = self.notLocalFullTileElements(kernel, False)
    (edgeVw, elements_0[True], edgeVw_1, elements_1[True] )  = self.notLocalFullTileElements(kernel, True)
    edgeScaled_0 = len(elements_0[True]) // len(elements_1[False])
    edgeScaled_1 = len(elements_1[True]) // len(elements_1[False])
    noEgScaled_0 = len(elements_0[False]) // len(elements_1[False])

    for i in range(0, len(elements_1[False])):
      element = elements_1[False][i]
      if element in self.LSUelementsPerLSUWave:
        elements_f1[False].append(element)
        for j in range(0, edgeScaled_0):
          # in general, edge will affect vc0 dimension.
          element = elements_0[True][i*edgeScaled_0+j]
          elements_f0[True].append(element)
        for j in range(0, edgeScaled_1):
          # in general, edge will affect vc0 dimension.
          element = elements_1[True][i*edgeScaled_1+j]
          elements_f1[True].append(element)
        for j in range(0, noEgScaled_0):
          # in general, edge will affect vc0 dimension.
          element = elements_0[False][i*noEgScaled_0+j]
          elements_f0[False].append(element)

    vectorWidths   = [fullVw, edgeVw]
    vectorWidths_1 = [fullVw_1, edgeVw_1]

    noGSUBranch = (kernel["GlobalSplitU"] == 0 and kernel["StreamK"] != 3)
    module = Module("localSplitUGlobalWrite")
    module.add(self.globalWriteElements(kernel, tPA, tPB, vectorWidths, vectorWidths_1, elements_f0, elements_f1, noGSUBranch=noGSUBranch))
    self.cleanupGlobalWrite(kernel)
    self.vgprPool.checkIn(self.accVgprLdsReduction)
    return module

  ##############################################################################
  # checkIsBetaZero
  # tmpSgpr is one temp sgpr
  # betaLabel is label to branch to if beta != 0
  ##############################################################################
  def checkIsBetaZero(self, kernel, tmpSgprInfo, betaLabel, isLongBranch=False, placeHolder=None, posNeg: int=0):
    module = Module("checkIsBetaZero label %s"%betaLabel)
    assert(isinstance(betaLabel, Label))
    betaLabelName = betaLabel.getLabelName()
    if kernel["ProblemType"]["UseBeta"]:
      if self.states.bpeCinternal <= self.states.bpr: # 1 register to check for Beta==0
        #module.add(SCmpKEQU32(src=sgpr("Beta"), simm16=0, comment="Beta == 0"))
        module.add(self.getSCMPKInstruction("EQU32", "Beta", 0, comment="Beta == 0"))
      else: # multiple registers to check for Beta==0
        module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=sgpr("Beta+0"), comment="tmp = Beta[0]"))
        for i in range(1, self.states.bpeCinternal//self.states.bpr):
          module.add(SOrB32(dst=sgpr(tmpSgprInfo.idx), src0=sgpr("Beta+%u"%i), src1=sgpr(tmpSgprInfo.idx), comment="tmp |= Beta[%u] " % i))
        #module.add(SCmpKEQU32(src=sgpr(tmpSgprInfo.idx), simm16=0, comment="Beta == 0"))
        module.add(self.getSCMPKInstruction("EQU32", tmpSgprInfo.idx, 0, comment="Beta == 0"))
      if placeHolder == None:
        if isLongBranch:
          module.add(self.longBranchScc0(betaLabel, posNeg, tmpSgprInfo))
        else:
          module.add(SCBranchSCC0(labelName=betaLabelName, comment="Branch if Beta is not zero"))
      else:
        placeHolderModule = Module(placeHolder)
        module.add(placeHolderModule)
    module.addSpaceLine()
    return module

  ##############################################################################
  # checkIsEdge
  # tmpSgpr must have at least 4 free SGPR
  # isEdgeTarget is the branch target if edges are required
  ##############################################################################
  def checkIsEdge(self, kernel, tmpSgprInfo, isEdgeTargetMT0, isEdgeTargetMT1, isLongBranch=False, placeHolder=None):
    assert(isinstance(isEdgeTargetMT0, Label) and isinstance(isEdgeTargetMT1, Label))
    isEdgeTargetMT0Label = isEdgeTargetMT0.getLabelName()
    isEdgeTargetMT1Label = isEdgeTargetMT1.getLabelName()
    module = Module("checkIsEdge")
    tmpS0  = tmpSgprInfo.idx
    tmpS1  = tmpS0 + 1
    tmpS23 = tmpS1 + 1

    wg0="WorkGroup0"
    wg1="WorkGroup1"

    # check edge0 ###
    # s23 = rMT0 = Size0 % MT0
    #--
    sizeBoundary = [0,0]
    sizeBoundary[0] = \
        sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index0"])
    sizeBoundary[1] = \
        sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index1"])

    module.add(scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[0], kernel["MacroTile0"], \
      ContinuousRegister(tmpS23, 2), 2))
    # s23 = nwg0-1
    module.add(SAddU32(dst=sgpr(tmpS1), src0=hex(-1), src1=sgpr("NumWorkGroups0")))
    module.add(SCmpGeU32(src0=sgpr(wg0), src1=sgpr(tmpS1), comment="wg0 >= nwg0-1 ?"))
    module.add(SCSelectB32(dst=sgpr(tmpS0), src0=sgpr(tmpS0), src1=0, comment="set rMT0"))
    # s01 now = myMT0 = wg0 < nwg0-1 ? MT0 : rMT0

    # if rMT0 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      #module.add(SCmpKGtU32(src=sgpr(tmpS0), simm16=0, comment="rMT0 > 0"))
      module.add(self.getSCMPKInstruction("GTU32", tmpS0, 0, comment="rMT0 > 0"))
      if self.db["ForceEdgeStores"]:
        module.add(SCmpEQU32(src0=sgpr(tmpS0), src1=sgpr(tmpS0), comment="ForceEdgeStores!"))
      if placeHolder == None:
        if isLongBranch:
          module.add(self.longBranchScc1(isEdgeTargetMT0, posNeg=1, tmpSgprInfo=tmpSgprInfo, comment="jump if edges required"))
        else:
          module.add(SCBranchSCC1(labelName=isEdgeTargetMT0Label, comment="jump if edges required"))
      else:
        placeHolderModule = Module(placeHolder)
        placeHolderModule.addComment1("jump if edges required")
        module.add(placeHolderModule)

    # check edge1 ###
    # TODO-packed - this only needs to change to handle packing into C1 index
    # change would be similar to above - multiply by product of packed sizes in C1
    # --

    # s23 = rMT1 = Size1 % MT1
    module.add(scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[1], kernel["MacroTile1"], \
      ContinuousRegister(tmpS23, 2), 2))
    # s01 now = myMT1 = wg1 < nwg1-1 ? MT1 : rMT1

    # s23 = nwg1-1
    module.add(SAddU32(dst=sgpr(tmpS1), src0=hex(-1), src1=sgpr("NumWorkGroups1")))
    module.add(SCmpGeU32(src0=sgpr(wg1), src1=sgpr(tmpS1), comment="wg1 >= nwg1-1"))
    module.add(SCSelectB32(dst=sgpr(tmpS0), src0=sgpr(tmpS0), src1=0, comment="set rMT1"))

    # if rMT1 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      #module.add(SCmpKGtU32(src=sgpr(tmpS0), simm16=0, comment="rMT1 > 0"))
      module.add(self.getSCMPKInstruction("GTU32", tmpS0, 0, comment="rMT1 > 0"))
      if placeHolder == None:
        if isLongBranch:
          module.add(self.longBranchScc1(isEdgeTargetMT1, posNeg=1, tmpSgprInfo=tmpSgprInfo, comment="jump if edges required"))
        else:
          module.add(SCBranchSCC1(labelName=isEdgeTargetMT1Label, comment="jump if edges required"))
      else:
        placeHolderModule = Module(placeHolder)
        placeHolderModule.addComment1("jump if edges required")
        module.add(placeHolderModule)
    return module

  ##############################################################################
  # checkIsFactorDimZero
  # tmpSgpr is one temp sgpr
  # factorDimLabel is label to branch to if factorDim != 0
  ##############################################################################
  def checkIsFactorDimZero(self, kernel, tmpSgprInfo, factorDimLabel, isLongBranch=False, posNeg: int=0):
    module = Module("checkIsFactorDimZero label %s"%factorDimLabel)
    assert(isinstance(factorDimLabel, Label))
    factorDimLabelName = factorDimLabel.getLabelName()
    if kernel["ProblemType"]["UseBias"] or kernel["ProblemType"]["UseScaleAlphaVec"]:
      if self.states.bpeCinternal <= self.states.bpr: # 1 register to check for Beta==0
        module.add(self.getSCMPKInstruction("EQU32", "FactorDim", 0, comment="FactorDim == 0"))
      else: # multiple registers to check for Beta==0
        module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=sgpr("FactorDim+0"), comment="tmp = FactorDim[0]"))
        for i in range(1, self.states.bpeCinternal//self.states.bpr):
          module.add(SOrB32(dst=sgpr(tmpSgprInfo.idx), src0=sgpr("FactorDim+%u"%i), src1=sgpr(tmpSgprInfo.idx), comment="tmp |= FactorDim[%u] " % i))
        module.add(self.getSCMPKInstruction("EQU32", tmpSgprInfo.idx, 0, comment="FactorDim == 0"))
      if isLongBranch:
        module.add(self.longBranchScc0(factorDimLabel, posNeg, tmpSgprInfo))
      else:
        module.add(SCBranchSCC0(labelName=factorDimLabelName, comment="Branch if FactorDim is not zero"))
    module.addSpaceLine()
    return module

  ##############################################################################
  # Global Write Elements
  ##############################################################################
  class BF16CVTVgprStruct(NamedTuple): # class for bf16 vgprs
    vgprBf16Temp: int = -1
    vgprBf16Mask: int = -1
    vgprFp32Nan: int = -1
    vgprBf16Inc: int = -1

  class FP8CVTVgprStruct(NamedTuple):
    vgprFp8NanInf: int = -1
    vgprFp8Temp: int   = -1
    vgprFp8Min: int    = -1
    vgprFp8Max: int    = -1

  class BF8CVTVgprStruct(NamedTuple):
    vgprBF8NanInf: int = -1
    vgprBF8Temp: int   = -1
    vgprBF8Min: int    = -1
    vgprBF8Max: int    = -1

  class I8CVTVgprStruct(NamedTuple):
    vgprI8Temp0: int   = -1
    vgprI8Temp1: int   = -1
    vgprI8Mask0: int   = -1
    vgprI8Mask1: int   = -1

  class ActivationSetPCStruct(NamedTuple):
    sgprOffsetActivation: int = -1
    sgprOffsetBack: int = -1
    vgprActCopy: int = -1

  def globalWriteElements(self, kernel, tPA, tPB, vectorWidths_2, vectorWidths_1, elements_2, elements_1,
                          noGSUBranch=False,
                          applyAlpha=True, # defaults to generating *=alpha codes
                          betas=None, # if left unspecified, then let global parameter decide
                          edges=None):
    if not self.do["PostLoop"]: return Module("GlobalWriteElements (Empty)")
    module = Module("GlobalWriteElements")

    module.addComment2("Global Write Elements")
    if kernel["ProblemType"]["OutputAmaxD"]:
        module.add(VMovB32(dst=vgpr("AmaxOut"), src="0"))
    if self.states.numStoreSgprToLoad or self.states.numStoreSgprToLoad2: # Wait for kernel args
      module.add(SWaitCnt(kmcnt=0, comment="wait for %u bytes of kern args."%((self.states.numStoreSgprToLoad+self.states.numStoreSgprToLoad2) * 4)))

    skBackup           = kernel["StreamK"]
    gsuBackup          = kernel["GlobalSplitU"]
    gsuAccumBackup     = kernel["_GlobalAccumulation"]
    bpeCexternalBackup = self.states.bpeCexternal
    afcBackup          = kernel["ActivationFuncCall"]
    useBiasBackup      = self.states.useBias
    betasBackup    = betas
    edgesBackup    = edges
    gsuLimit = 1 if noGSUBranch or self.debugConfig.splitGSU else 2
    if gsuLimit > 1:
      gsuLabel = Label(label=self.labels.getNameInc("GSU"), comment="")
      if kernel["StreamK"]:
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="Branch to stream-k store code"))
        module.add(SCmpEQU32(src0=sgpr("skTiles"), src1=1, comment="split == 1 ?"))
        # TODO May need long branch??
        module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if split == 1"))
      else:
        skipSrdTDLabel = Label(label=self.labels.getNameInc("Skip_SrdTD"), comment="")
        with self.allocTmpSgpr(1) as tmpSgprGSU:
          module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
          module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
        if (kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel'):
          module.add(self.longBranchScc1(label=gsuLabel, posNeg=1, comment="long branch if GSU == 1"))
          # module.add(SCBranchSCC1(labelName=skipSrdTDLabel.getLabelName(), comment="branch if GSU == 1"))
          module.add(self.SrdTDInit(kernel))
          module.add(skipSrdTDLabel)
        else:
          module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))

    gsuLimitRange = range(0, gsuLimit) # generate GSU1 and GSUM label
    # TODO: generate only GSUM label for MBSK
    # if (kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel'):
    #   gsuLimitRange = [0]

    for gsuLimitIdx in gsuLimitRange:
      if gsuLimit > 1:
        betas = betasBackup
        edges = edgesBackup
        if gsuLimitIdx == 0:
          self.states.bpeCexternal = self.states.bpeCinternal
          if (kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel'):
            self.states.useBias = self.states.useBias if self.states.useBias == DataDirection.WRITE else DataDirection.NONE
          if self.states.useBias == DataDirection.WRITE and kernel["ProblemType"]["BiasSrc"] == "D":
            self.states.useBias = DataDirection.NONE
          if (kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel'):
            kernel["LdsOffsetBias"] = kernel["LdsOffsetBiasGSU"]
            kernel["ActivationFuncCall"] = False
          kernel["GlobalSplitU"] = 2
          kernel["_GlobalAccumulation"] = kernel["_GlobalAccumulation"]
          if kernel["StreamK"]:
            kernel["StreamK"] = 0
            kernel["_GlobalAccumulation"] = "MultipleBuffer"
          vectorWidths = vectorWidths_2
          elements     = elements_2
        else:
          module.add(gsuLabel)
          self.states.bpeCexternal = self.states.bpeCexternalGSU1
          self.states.useBias = useBiasBackup
          kernel["LdsOffsetBias"] = kernel["LdsOffsetBiasNonGSU"]
          kernel["ActivationFuncCall"] = afcBackup
          kernel["GlobalSplitU"] = 1
          kernel["_GlobalAccumulation"] = None
          kernel["StreamK"] = skBackup
          if kernel["StreamK"]:
            kernel["GlobalSplitU"] = gsuBackup
            kernel["_GlobalAccumulation"] = gsuAccumBackup
          vectorWidths = vectorWidths_1
          elements     = elements_1
      else:
        if (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1) and (kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel'):
          self.states.useBias = self.states.useBias if self.states.useBias == DataDirection.WRITE else DataDirection.NONE
          kernel["LdsOffsetBias"] = kernel["LdsOffsetBiasGSU"]
          kernel["ActivationFuncCall"] = False
          vectorWidths = vectorWidths_2
          elements     = elements_2
        else:
          kernel["LdsOffsetBias"] = kernel["LdsOffsetBiasNonGSU"]
          vectorWidths = vectorWidths_1
          elements     = elements_1
      '''
      Post process for loop
      '''
      ssslist = []
      useSize = []

      if gsuLimit > 1 and gsuLimitIdx > 0:
        if kernel["ProblemType"]["UseScaleAB"]:
          if not self.states.preloadScaleA:
            module.add(self.setSgprToInUseState("AddressScaleA"))
            if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
              module.add(self.setSgprToInUseState("SrdScaleA"))
          if not self.states.preloadScaleB:
            module.add(self.setSgprToInUseState("AddressScaleB"))
            if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
              module.add(self.setSgprToInUseState("SrdScaleB"))
        if kernel["ProblemType"]["UseScaleAlphaVec"]:
          module.add(self.setSgprToInUseState("AddressScaleAlphaVec"))
          module.add(self.setSgprToInUseState("SrdScaleAlphaVec"))

      isSingleKernel = ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel') or kernel["StreamK"] > 0
      # Issue read scale A/B value for later use
      if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and \
        isSingleKernel and \
        ((kernel["ProblemType"]["DataTypeA"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters()) or \
        (kernel["ProblemType"]["DataTypeB"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters())):
        assert(kernel["ProblemType"]["ComputeDataType"].isSingle())
        sgprScaleA = self.sgprPool.checkOut(1, preventOverflow=False)
        sgprScaleB = self.sgprPool.checkOut(1, preventOverflow=False)
        for i,name in enumerate(['A','B']):
          if kernel["ProblemType"]["DataType%s"%name].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters():
            sgprScale = sgprScaleA if name == 'A' else sgprScaleB
            module.add(SMovB32(dst=sgpr(sgprScale), src=1.0 , comment="init as 1" ))
            label  = Label(self.labels.getNameInc("Scale%sValid"%name), "")
            module.add(BranchIfZero("AddressScale%s"%name, DataType('int64').toEnum(), -1, int(kernel["WavefrontSize"]/32), label, kernel["WavefrontSize"]))
            # load scale data
            module.add(SLoadB32(dst=sgpr(sgprScale), base=sgpr("AddressScale%s"%name,2), soffset=0, comment="load scale%s"%name))
            module.add(label)

      # Issue read scale C/D value for later use
      if kernel["ProblemType"]["UseScaleCD"] and ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0):
        module.add(SMovB32(dst=sgpr("ScaleD"), src=1.0 , comment="init as 1" ))
        module.add(SMovB32(dst=sgpr("ScaleD+1"), src=1.0 , comment="init as 1" ))
        label  = Label(self.labels.getNameInc("ScaleDValid"), "")
        module.add(BranchIfZero("AddressScaleD", DataType('int64').toEnum(), -1, int(kernel["WavefrontSize"]/32), label, kernel["WavefrontSize"]))
        # load scale data
        module.add(SLoadB32(dst=sgpr("ScaleD"), base=sgpr("AddressScaleD",2), soffset=0, comment="load scaleD"))
        module.add(label)
        sgprScaleC = self.sgprPool.checkOut(1, preventOverflow=False)
        module.add(SMovB32(dst=sgpr(sgprScaleC), src=1.0 , comment="init as 1" ))
        label  = Label(self.labels.getNameInc("ScaleCValid"), "")
        module.add(BranchIfZero("AddressScaleC", DataType('int64').toEnum(), -1, int(kernel["WavefrontSize"]/32), label, kernel["WavefrontSize"]))
        # load scale data
        module.add(SLoadB32(dst=sgpr(sgprScaleC), base=sgpr("AddressScaleC",2), soffset=0, comment="load scaleC"))
        module.add(label)

      factorDims = [0]
      if self.states.FactorDim == 3:
        factorDims.append(1)
      elif self.states.FactorDim == 2:
        factorDims = [1]

      vectorDataTypes = VectorDataTypes()
      if (kernel["ProblemType"]["UseScaleAlphaVec"]) and isSingleKernel:
        labelStr = self.labels.getNameInc("ScaleAlphaVec")
        if self.states.FactorDim == 3:
          with self.allocTmpSgpr(1,1) as tmpSgprRes:
            tmpSgpr = tmpSgprRes.idx
            module.add(SCmpKEQU32(src=sgpr("FactorDim"), simm16=0, comment="FactorDim == 0"))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr("SizeI"), src1=sgpr("SizeJ")))
            module.add(allocPostLoopSrdSuppress("ScaleAlphaVec", labelStr, sgprLength=sgpr(tmpSgpr)))
        elif self.states.FactorDim == 2:
          module.add(allocPostLoopSrdSuppress("ScaleAlphaVec", labelStr, sgprLength=sgpr("SizeJ")))
        else:
          module.add(allocPostLoopSrdSuppress("ScaleAlphaVec", labelStr, sgprLength=sgpr("SizeI")))
        module.add(SMulI32(dst=sgpr("SrdScaleAlphaVec+2"), src0=hex(self.states.bpeCinternal), src1=sgpr("SrdScaleAlphaVec+2"), comment="ScaleAlphaVec scaled by BPE"))# scaled by BPE
        for d in range(len(factorDims)):
          vectorDataTypes.scaleAlpha(d).dataType = kernel["ProblemType"]["ComputeDataType"]

      # Add ScaleABVec support here
      # Issue read scale A/B vector value for later use
      if ((kernel["ProblemType"]["UseScaleAB"] == "Vector")) and isSingleKernel:
        labelStrA = self.labels.getNameInc("ScaleAVec")
        labelStrB = self.labels.getNameInc("ScaleBVec")
        module.add(allocPostLoopSrdSuppress("ScaleA", labelStrA, sgprLength=sgpr("SizeI")))
        module.add(allocPostLoopSrdSuppress("ScaleB", labelStrB, sgprLength=sgpr("SizeJ")))
        module.add(SMulI32(dst=sgpr("SrdScaleA+2"), src0=hex(self.states.bpeCinternal), src1=sgpr("SrdScaleA+2"), comment="ScaleAVec scaled by BPE"))# scaled by BPE
        module.add(SMulI32(dst=sgpr("SrdScaleB+2"), src0=hex(self.states.bpeCinternal), src1=sgpr("SrdScaleB+2"), comment="ScaleBVec scaled by BPE"))# scaled by BPE
        vectorDataTypes.scaleA.dataType = kernel["ProblemType"]["ComputeDataType"]
        vectorDataTypes.scaleB.dataType = kernel["ProblemType"]["ComputeDataType"]

      factorDim0Label = Label(self.labels.getNameInc("Load_FactorDim_0"), "")
      factorDim1Label = Label(self.labels.getNameInc("Load_FactorDim_1"), "")

      # Add bias lds
      isLdsLoaded = False
      if self.states.useBias == DataDirection.READ and isSingleKernel:
        # Init bias Srd
        labelStr = self.labels.getNameInc("Bias")
        with self.allocTmpSgpr(1,1) as tmpSgprRes:
          set_bs_label = Label(self.labels.getNameInc("Dont_Set_BiasStride"), "")
          tmpSgpr = tmpSgprRes.idx
          module.add(SAddU32(dst=sgpr(tmpSgpr), src0=sgpr("WorkGroup2"), src1=hex(1)))
          module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr("BiasStride"), src1=sgpr(tmpSgpr), comment="stride * (wg+1)"))
          module.add(SCmpEQU32(sgpr(tmpSgpr), 0, comment="bias stride = 0?"))
          if self.states.FactorDim == 3:
            module.add(SCBranchSCC0(set_bs_label.getLabelName()))
            module.add(self.getSCMPKInstruction("EQU32", "FactorDim", 0, comment="FactorDim == 0"))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr("SizeI"), src1=sgpr("SizeJ")))
            module.add(set_bs_label)
          elif self.states.FactorDim == 2:
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr("SizeJ"), src1=sgpr(tmpSgpr)))
          else:
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr("SizeI"), src1=sgpr(tmpSgpr)))
          # module.add(SMovB32(dst=sgpr(tmpSgpr), src=128, comment="set bias stride"))
          module.add(allocPostLoopSrdSuppress("Bias", labelStr, sgprLength=sgpr(tmpSgpr)))

        loadBiasEndLabel = Label(self.labels.getNameInc("Load_Bias_End"), "")
        if self.states.FactorDim == 3:
          module.add(factorDim0Label)
          module.add(self.getSCMPKInstruction("LGU32", "FactorDim", 0, comment="FactorDim != 0"))
          module.add(SCBranchSCC1(factorDim1Label.getLabelName(), "Branch if true"))

        for d in range(len(factorDims)):
          # Calculate max vgpr for bias read
          vectorDataTypes.bias(d).dataType = kernel["ProblemType"]["BiasDataTypeList"][0]
          totalTmpVgpr = self.getNumOfTempVgprs(vectorDataTypes, kernel, 1, factorDims[d])
          tmpVgpr      = self.vgprPool.checkOutAligned(totalTmpVgpr, 2, "store tmps")
          tmpVgprRes   = ContinuousRegister(idx=tmpVgpr, size=4)

          if d == 1:
            module.add(factorDim1Label)
          multiBiasTypeLabel = []
          for i in kernel["ProblemType"]["BiasDataTypeList"]:
            name = self.labels.getNameInc("Load_Bias%s_%u"%(i.toNameAbbrev(), factorDims[d]))
            multiBiasTypeLabel.append(Label(name, ""))
          multiBiasTypeLabel.append(loadBiasEndLabel)
          offsetVgpr  = self.vgprPool.checkOut(1)
          if len(kernel["ProblemType"]["BiasDataTypeList"]) == 1:
            vectorDataTypes.bias(d).dataType = kernel["ProblemType"]["BiasDataTypeList"][0]
            module.add(self.readVectorToLDS(vectorDataTypes, kernel, 1, offsetVgpr, tmpVgprRes, factorDims[d]))
            if len(factorDims) == 2:
              if d == 0:
                module.add(SBranch(labelName=loadBiasEndLabel.getLabelName(), comment="Branch to load bias end"))
              else:
                module.add(loadBiasEndLabel)
          else:
            for i, label in enumerate(multiBiasTypeLabel[1:]):
              typeValue = kernel["ProblemType"]["BiasDataTypeList"][i].value
              module.add(multiBiasTypeLabel[i])
              #module.add(SCmpKLGU32(sgpr("BiasType"), typeValue, "BiasType != %u"%typeValue))
              module.add(self.getSCMPKInstruction("LGU32", "BiasType", typeValue, comment="BiasType != %u"%typeValue))
              module.add(SCBranchSCC1(label.getLabelName(), "Branch if true"))
              vectorDataTypes.bias(d).dataType = kernel["ProblemType"]["BiasDataTypeList"][i]
              module.add(self.readVectorToLDS(vectorDataTypes, kernel, 1, offsetVgpr, tmpVgprRes, factorDims[d]))
              module.add(SBranch(labelName=loadBiasEndLabel.getLabelName(), comment="Branch to load bias end"))
            if d == len(factorDims) -1:
              module.add(loadBiasEndLabel)
          isLdsLoaded = True
          self.vgprPool.checkIn(offsetVgpr)
          self.vgprPool.checkIn(tmpVgpr)
      elif self.states.useBias == DataDirection.WRITE:
        labelStr = self.labels.getNameInc("Bias")
        if kernel["ProblemType"]["BiasSrc"] == "A" or kernel["ProblemType"]["BiasSrc"] == "B":
          # Calculate max vgpr for bias write A, B
          tP          = tPA if kernel["ProblemType"]["BiasSrc"] == "A" else tPB
          tile01      = tP["tile01Idx"]
          maxKId      = self.states.lraTileProperties[tile01].maxKId
          biasMaxVgpr = kernel["VectorWidthA"] * kernel["ProblemType"]["ComputeDataType"].numRegisters() * maxKId
          maxAlign    = max(1, (kernel["VectorWidthA"] + 1) // 2 * 2)
          tmpVgpr     = self.vgprPool.checkOutAligned(biasMaxVgpr, maxAlign, "store tmps")
          tmpVgprRes  = ContinuousRegister(idx=tmpVgpr, size=biasMaxVgpr)

          # Skip bias store
          skipGlobalStoreLabel = Label(self.labels.getNameInc("SkipBiasStore"), comment="")
          wgIdx = 1 if tile01 == 0 else 0
          #module.add(SCmpKLGU32(sgpr("WorkGroup%d"%wgIdx), 0, "WorkGroup%d != 0"%wgIdx))
          module.add(self.getSCMPKInstruction("LGU32", "WorkGroup%d"%wgIdx, 0, comment="WorkGroup%d != 0"%wgIdx))
          module.add(SCBranchSCC1(skipGlobalStoreLabel.getLabelName(), "Branch if true"))
          if (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1) and kernel["GlobalSplitUAlgorithm"] == "MultipleBuffer":
            sourceAddress = "D"
          else:
            sourceAddress = "Bias"
          numRecordsStr = "SizeI" if kernel["ProblemType"]["BiasSrc"] == "A" else "SizeJ"
          # Init bias Srd
          module.add(allocPostLoopSrdSuppressRaw("Bias", sourceAddress, labelStr, sgprLength=sgpr(numRecordsStr)))
          if sourceAddress == "D":
            module.add(self.undefineSgpr("AddressD"))
          multiBiasTypeLabel = []
          for i in kernel["ProblemType"]["BiasDataTypeList"]:
            name = self.labels.getNameInc("Write_Bias%s"%i.toNameAbbrev())
            multiBiasTypeLabel.append(Label(name, ""))
          writeBiasEndLabel = Label(self.labels.getNameInc("Write_Bias_End"), "")
          multiBiasTypeLabel.append(writeBiasEndLabel)
          # Get gwvw
          '''
          gwvw is set to max(mt // kernel["NumThreads"], kernel["VectorWidthA"]) instead of kernel["VectorWidthA"] is that we don't want batch exists.
          If VW is set to 1, MT=512, and flat work group = 256. We will have to set gwvw to 2 to store all the bias data.
          '''
          tile01 = tP["tile01Idx"]
          mt     = kernel["MacroTile%u" % tile01]
          gwvw   = min(max(mt // kernel["NumThreads"], kernel["VectorWidthA"]), tP["glvw"])
          offsetVgpr  = self.vgprPool.checkOut(gwvw)
          with self.allocTmpSgpr(5, 2) as tmpSgprRes:
            if kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1:
              module.add(self.writeBiasToGlobal(kernel["ProblemType"]["ComputeDataType"], kernel, tP, gwvw, offsetVgpr, tmpSgprRes, tmpVgprRes))
            elif len(kernel["ProblemType"]["BiasDataTypeList"]) == 1:
              module.add(self.writeBiasToGlobal(kernel["ProblemType"]["BiasDataTypeList"][0], kernel, tP, gwvw, offsetVgpr, tmpSgprRes, tmpVgprRes))
            else:
              for i, label in enumerate(multiBiasTypeLabel[1:]):
                typeValue = kernel["ProblemType"]["BiasDataTypeList"][i].value
                module.add(multiBiasTypeLabel[i])
                #module.add(SCmpKLGU32(sgpr("BiasType"), typeValue, "BiasType != %u"%typeValue))
                module.add(self.getSCMPKInstruction("LGU32", "BiasType", typeValue, comment="BiasType != %u"%typeValue))
                module.add(SCBranchSCC1(label.getLabelName(), "Branch if true"))
                module.add(self.writeBiasToGlobal(kernel["ProblemType"]["BiasDataTypeList"][i], kernel, tP, gwvw, offsetVgpr, tmpSgprRes, tmpVgprRes))
                module.add(SBranch(labelName=writeBiasEndLabel.getLabelName(), comment="Branch to write bias end"))
              module.add(writeBiasEndLabel)
          self.vgprPool.checkIn(offsetVgpr)
          self.vgprPool.checkIn(tmpVgpr)
          module.add(skipGlobalStoreLabel)
        else:
          # Init bias Srd
          module.add(allocPostLoopSrdSuppress("Bias", labelStr, hex(0x80000000)))
          ssslist.append("Bias")
          useSize.append(True)

      if vectorDataTypes.isValid() and (not isLdsLoaded):
        if self.states.FactorDim == 3:
          module.add(factorDim0Label)
          module.add(self.getSCMPKInstruction("LGU32", "FactorDim", 0, comment="FactorDim != 0"))
          module.add(SCBranchSCC1(factorDim1Label.getLabelName(), "Branch if true"))
        labelDimEnd = Label(self.labels.getNameInc("MultiDimEnd"), "")
        for d in range(len(factorDims)):
          totalTmpVgpr = self.getNumOfTempVgprs(vectorDataTypes, kernel, 1, factorDims[d])
          tmpVgpr      = self.vgprPool.checkOutAligned(totalTmpVgpr, 2, "store tmps")
          tmpVgprRes   = ContinuousRegister(idx=tmpVgpr, size=4)
          offsetVgpr  = self.vgprPool.checkOut(1)
          if d == 1:
            module.add(factorDim1Label)
          module.add(self.readVectorToLDS(vectorDataTypes, kernel, 1, offsetVgpr, tmpVgprRes, factorDims[d]))
          if self.states.FactorDim == 3 and d == 0:
            module.add(SBranch(labelName=labelDimEnd.getLabelName(), comment="Branch to load end"))
          self.vgprPool.checkIn(offsetVgpr)
          self.vgprPool.checkIn(tmpVgpr)
        if len(factorDims) > 1:
          module.add(labelDimEnd)

      # Undefine LDS load related sgprs
      if gsuLimit > 1 and gsuLimitIdx == 0:
        if kernel["ProblemType"]["UseScaleAB"]:
          if not self.states.preloadScaleA:
            module.add(self.setSgprToFreeState("AddressScaleA"))
            if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
              module.add(self.setSgprToFreeState("SrdScaleA"))
          if not self.states.preloadScaleB:
            module.add(self.setSgprToFreeState("AddressScaleB"))
            if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
              module.add(self.setSgprToFreeState("SrdScaleB"))
        if kernel["ProblemType"]["UseScaleAlphaVec"]:
          module.add(self.setSgprToFreeState("AddressScaleAlphaVec"))
          module.add(self.setSgprToFreeState("SrdScaleAlphaVec"))
      else:
        if kernel["ProblemType"]["UseScaleAB"]:
          if not self.states.preloadScaleA:
            module.add(self.undefineSgpr("AddressScaleA"))
            if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
              module.add(self.undefineSgpr("SrdScaleA"))
          if not self.states.preloadScaleB:
            module.add(self.undefineSgpr("AddressScaleB"))
            if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
              module.add(self.undefineSgpr("SrdScaleB"))
        if kernel["ProblemType"]["UseScaleAlphaVec"]:
          module.add(self.undefineSgpr("AddressScaleAlphaVec"))
          module.add(self.undefineSgpr("SrdScaleAlphaVec"))

      if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and (((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0) or kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel') and \
        ((kernel["ProblemType"]["DataTypeA"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters()) or \
        (kernel["ProblemType"]["DataTypeB"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters())):
        assert(kernel["ProblemType"]["ComputeDataType"].isSingle())
        newAlphaVgpr = self.vgprPool.checkOut(1)
        module.add(VMovB32(dst=vgpr(newAlphaVgpr), src=sgpr("Alpha")))
        module.add(SWaitCnt(kmcnt=0, comment="wait for scaleAB load"))
        if kernel["ProblemType"]["DataTypeA"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters():
          module.add(VMulF32(dst=vgpr(newAlphaVgpr), src0=vgpr(newAlphaVgpr), src1=sgpr(sgprScaleA)))
        if kernel["ProblemType"]["DataTypeB"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters():
          module.add(VMulF32(dst=vgpr(newAlphaVgpr), src0=vgpr(newAlphaVgpr), src1=sgpr(sgprScaleB)))
        module.add(SNop(waitState=0, comment="1 wait states"))
        if kernel["StreamK"] > 0:
          oldAlpha = self.sgprPool.checkOut(1)
          module.add(SMovB32(dst=sgpr(oldAlpha), src=sgpr("Alpha"), comment="Save alpha value"))
        module.add(VReadfirstlaneB32(dst=sgpr("Alpha"), src=vgpr(newAlphaVgpr), comment="Update Alpha"))
        self.vgprPool.checkIn(newAlphaVgpr)
        self.sgprPool.checkIn(sgprScaleA)
        self.sgprPool.checkIn(sgprScaleB)

      # Update beta
      if kernel["ProblemType"]["UseScaleCD"] and ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0):
        assert(kernel["ProblemType"]["ComputeDataType"].isSingle())
        newBetaVgpr = self.vgprPool.checkOut(1)
        module.add(VMovB32(dst=vgpr(newBetaVgpr), src=sgpr("Beta")))
        if (not ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel')) or kernel["StreamK"] > 0:
          module.add(SWaitCnt(kmcnt=0, comment="wait for scaleC load"))
        module.add(VMulF32(dst=vgpr(newBetaVgpr), src0=vgpr(newBetaVgpr), src1=sgpr(sgprScaleC)))
        module.add(SNop(waitState=0, comment="1 wait states"))
        module.add(VReadfirstlaneB32(dst=sgpr("Beta"), src=vgpr(newBetaVgpr), comment="Update Beta"))
        self.vgprPool.checkIn(newBetaVgpr)
        self.sgprPool.checkIn(sgprScaleC)
        # Copy scaleD for PK calculations
        module.add(SMovB32(dst=sgpr("ScaleD+1"), src=sgpr("ScaleD")))

      if kernel["ProblemType"]["UseE"] and ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0):
        # Update E offset1
        strideE1 = "StrideE%s" % (self.states.indexChars[kernel["PackedC1IndicesX"][0]])
        module.add(VMulLOU32(dst=vgpr(self.vgprs.coutRowPtrE), src0=vgpr(self.vgprs.coutRowPtrE), src1=sgpr(strideE1), comment=" offset 1"))
        labelEStr = self.labels.getNameInc("E")
        module.add(allocPostLoopSrdSuppress("E", labelEStr, hex(0x80000000)))
        ssslist.append("E")
        useSize.append(False)

      if ssslist:
        module.add(self.computeStoreSrdStart(kernel, ssslist, useSize=useSize, noMultipleBuffer=True))

      '''
      Post process for loop end
      '''

      atomic = (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer' and kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel')
      activation = self.exclasses.activation

      # write possibilities and labels
      # if beta/edge combo not specified fall back to global param definition
      if betas is None:
        hasBeta = kernel["ProblemType"]["UseBeta"] and \
          (kernel["_GlobalAccumulation"] != 'MultipleBuffer' or \
          kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel' or \
          kernel["GlobalSplitU"] == 0)
        betas = [False, True] if hasBeta else [False]
      if edges is None:
        edges = [False, True] if self.do["EdgeWrite"] else [False]
      if factorDims is None:
        factorDims = [0, 1] if self.states.FactorDim == 3 else [1] if self.states.FactorDim == 2 else [0]
      writeLabels = {}
      splitMN = False
      if True in edges:
        if vectorWidths[0] != vectorWidths[1]:
            splitMN = True
      for beta in betas:
        writeLabels[beta] = {}
        for edge in edges:
          writeLabels[beta]["EdgeCheck0"] = Label(self.labels.getNameInc("GW_B%u_E%u_EdgeCheck0" % ( 1 if beta else 0, 1 if edge else 0) ), "")
          writeLabels[beta]["EdgeCheck1"] = Label(self.labels.getNameInc("GW_B%u_E%u_EdgeCheck1" % ( 1 if beta else 0, 1 if edge else 0) ), "")
          writeLabels[beta][edge] = {}
          if len(factorDims) == 1:
            if edge:
              if splitMN:
                writeLabels[beta][edge][factorDims[0]] = []
                writeLabels[beta][edge][factorDims[0]].append(Label(self.labels.getNameInc("GW_B%u_E%u_M" % ( 1 if beta else 0, 1 if edge else 0) ), ""))
                writeLabels[beta][edge][factorDims[0]].append(Label(self.labels.getNameInc("GW_B%u_E%u_N" % ( 1 if beta else 0, 1 if edge else 0) ), ""))
              else:
                writeLabels[beta][edge][factorDims[0]] = [Label(self.labels.getNameInc("GW_B%u_E%u" % ( 1 if beta else 0, 1 if edge else 0) ), "")]
            else:
              writeLabels[beta][edge][factorDims[0]] = [Label(self.labels.getNameInc("GW_B%u_E%u" % ( 1 if beta else 0, 1 if edge else 0) ), "")]
          else:
            for factorDim in factorDims:
              if edge:
                if splitMN:
                  writeLabels[beta][edge][factorDim] = []
                  writeLabels[beta][edge][factorDim].append(Label(self.labels.getNameInc("GW_B%u_E%u_FD%u_M" % ( 1 if beta else 0, 1 if edge else 0, factorDim) ), ""))
                  writeLabels[beta][edge][factorDim].append(Label(self.labels.getNameInc("GW_B%u_E%u_FD%u_N" % ( 1 if beta else 0, 1 if edge else 0, factorDim) ), ""))
                else:
                  writeLabels[beta][edge][factorDim]= [Label(self.labels.getNameInc("GW_B%u_E%u_FD%u" % ( 1 if beta else 0, 1 if edge else 0, factorDim) ), "")]
              else:
                writeLabels[beta][edge][factorDim] = [Label(self.labels.getNameInc("GW_B%u_E%u_FD%u" % ( 1 if beta else 0, 1 if edge else 0, factorDim) ), "")]
      endLabel = Label(self.labels.getNameInc("GW_End"), "")

      # Layout
      """
      if B1 goto label_B1
      if E1 goto label_B0_E1_FD0
      if BD1 goto label_B0_E0_FD1
      label_B0_E0_FD0:
      writes
      goto label_End
      label_B0_E0_FD1:
      writes
      goto label_End
      label_B0_E1_FD0:
      if BD1 goto label_B0_E1_FD1
      writes
      goto label_End
      label_B0_E1_FD1:
      writes
      goto label_End
      label_B1:
      if E1 goto label_B1_E1_FD0
      if BD1 goto label_B1_E0_FD1
      label_B1_E0_FD0:
      writes
      goto label_End
      label_B1_E0_FD1:
      writes
      goto label_End
      label_B1_E1_FD0:
      if BD1 goto label_B1_E1_FD1
      writes
      goto label_End
      label_B1_E1_FD1:
      writes
      goto label_End
      label_End
      """

      ########################################
      # Vgprs
      maxAlign = 2
      if kernel["BufferStore"]:
        numTmpVgpr = 2
        if len(kernel["PackedC0IndicesX"]) > 1:
          numTmpVgpr += 1
      else:
        numTmpVgpr = 2 + 3 # GLOBAL_OFFSET_C needs 3, plus 2 tmps?
      # BufferOOB
      if (kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel") or kernel["BufferStore"]:
        numTmpVgpr += 1
      else:
        numTmpVgpr = None
      # dot2: for WaveSplitK reduction
      if kernel["NumWaveSplitK"] > 1:
        numTmpVgpr += 1
      # Get max vgpr and sgpr for activation
      actPCGwvwVgpr = 0
      actPCMaxTempSgpr = 0
      actTempSgpr = 0
      actExportType = ActivationType.Export.GRADONLY if kernel["ProblemType"]["Gradient"] else ActivationType.Export.NORMAL
      if kernel["ActivationFuncCall"] or \
        ((((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0) and kernel["ActivationFused"]) and \
        (kernel["ProblemType"]["ActivationType"] != 'none')):
        maxVw = max(vectorWidths)
        # Here is where activation creates cache if cache is enabled
        usage = activation.getAllGprUsage(kernel["ProblemType"]["ActivationComputeDataType"], kernel["ProblemType"]["ActivationType"], exportType=actExportType)
        actPCMaxTempVgpr = 0
        for _, gprs in usage.items():
          actPCMaxTempVgpr = max(actPCMaxTempVgpr, gprs["vgpr"])
          actPCMaxTempSgpr = max(actPCMaxTempSgpr, gprs["sgpr"])
        actPCGwvwVgpr = int(ceil(maxVw * kernel["ProblemType"]["ActivationComputeDataType"].numRegisters()))
        numTmpVgpr = max(numTmpVgpr, actPCMaxTempVgpr + actPCGwvwVgpr)
      if kernel["ProblemType"]["UseE"] and (not kernel["ProblemType"]["Gradient"]):
        maxVw = max(vectorWidths)
        gwvwVgpr = int(ceil(maxVw * kernel["ProblemType"]["ActivationComputeDataType"].numRegisters()))
        if kernel["ActivationFuncCall"]:
          gwvwVgpr += actPCMaxTempVgpr + actPCGwvwVgpr
        numTmpVgpr = max(numTmpVgpr, gwvwVgpr)
      tmpVgpr = self.vgprPool.checkOutAligned(numTmpVgpr, maxAlign, "store tmps")
      tmpVgpr = ContinuousRegister(idx=tmpVgpr, size=numTmpVgpr)

      cvtVgprStruct  = None
      cvtVgpr        = None
      if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        cvtVgpr = self.vgprPool.checkOut(4)
        cvtVgprStruct = self.BF16CVTVgprStruct(vgprBf16Temp=cvtVgpr, vgprBf16Mask=(cvtVgpr+1), \
                                               vgprFp32Nan=(cvtVgpr+2), vgprBf16Inc=(cvtVgpr+3))
      elif kernel["ProblemType"]["DestDataType"].isAnyFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        cvtVgpr = self.vgprPool.checkOut(4)
        cvtVgprStruct = self.FP8CVTVgprStruct(vgprFp8Temp=cvtVgpr, vgprFp8NanInf=(cvtVgpr+1), \
                                              vgprFp8Min=(cvtVgpr+2), vgprFp8Max=(cvtVgpr+3))
      elif kernel["ProblemType"]["DestDataType"].isAnyBFloat8():
        cvtVgpr = self.vgprPool.checkOut(4)
        cvtVgprStruct = self.BF8CVTVgprStruct(vgprBF8Temp=cvtVgpr, vgprBF8NanInf=(cvtVgpr+1), \
                                              vgprBF8Min=(cvtVgpr+2), vgprBF8Max=(cvtVgpr+3))
      elif kernel["ProblemType"]["DestDataType"].isInt8():
        cvtVgpr = self.vgprPool.checkOut(4)
        cvtVgprStruct = self.I8CVTVgprStruct(vgprI8Temp0=cvtVgpr, vgprI8Temp1=(cvtVgpr+1), \
                                             vgprI8Mask0=(cvtVgpr+2), vgprI8Mask1=(cvtVgpr+3))

      activationSetPCStruct = None
      activationLabelList = None
      activationEnumStrList = None
      toActModuleList = None
      isInsertActFunctionCallAddrCalc = True
      if kernel["ActivationFuncCall"]:
        sgprOffsetActivation = self.sgprPool.checkOutAligned(2, 2, preventOverflow=False)
        sgprOffsetBack = self.sgprPool.checkOutAligned(2, 2, preventOverflow=False)
        activationSetPCStruct = self.ActivationSetPCStruct(sgprOffsetActivation=sgprOffsetActivation, \
          sgprOffsetBack=sgprOffsetBack, vgprActCopy=tmpVgpr.idx)
        activationCDataType = kernel["ProblemType"]["ActivationComputeDataType"]
        activationLabelList = {}
        toActModuleList = {}
        supportedBy = ActivationType.SupportedBy.ALL if kernel["ProblemType"]["ActivationType"] == 'all' else ActivationType.SupportedBy.HIPBLASLT
        activationEnumStrList = ActivationType.getEnumStrList(activationCDataType, supportedBy, exportType=actExportType)
        for gwvw in vectorWidths:
          if gwvw in activationLabelList:
            continue
          activationLabelList[gwvw] = []
          toActModuleList[gwvw] = []
          for enumStr in activationEnumStrList:
            name = self.labels.getNameInc("Activation_%s_VW%u"% (enumStr.capitalize(), gwvw))
            activationLabelList[gwvw].append(Label(name, ""))
            toActModuleList[gwvw].append(Label("To_%s"% (name), ""))
        # Add branch here if all elements are identical
        if vectorWidths.count(vectorWidths[0]) == len(vectorWidths):
          isInsertActFunctionCallAddrCalc = False
          module.add(self.insertActFunctionCallAddrCalc(activationSetPCStruct.sgprOffsetActivation, \
            vectorWidths[0], toActModuleList, activationEnumStrList, activationLabelList))

      ########################################
      # Sgprs

      # allocate tmps for the store header (before the batch implementations)
      # branch B1 or B0
      # betaLabel = Label("GW_Beta", "") if (gsuLimit > 1) and (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1) else Label(self.labels.getNameInc("GW_Beta"), "")
      betaLabel = Label(self.labels.getNameInc("GW_Beta"), "")
      skPartialsLabel = Label(label=self.labels.getNameInc("SK_Partials"), comment="")
      skComponent = Component.StreamK.find(self)
      module.add(skComponent.storeBranches(self, kernel, skPartialsLabel, vectorWidths_1, elements_1, tmpVgpr.idx, cvtVgprStruct))

      if (kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel'):
        reductionStartLabel = Label(self.labels.getNameInc("Reduction_Start"), comment="Reduction start")
        reductionEndLabel = Label(self.labels.getNameInc("Reduction_End"), comment="Reduction end")
        # TODO: enabled when only GSUM label for MBSKGSU1
        # with self.allocTmpSgpr(1) as tmpSgprGSU:
        #   module.addComment1("cases with GSU>1 need reduction")
        #   module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
        #   module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
        # module.add(SCBranchSCC0(labelName=reductionStartLabel.getLabelName(), comment="branch if GSU != 1"))
        # module.add(SMovB64(dst=sgpr("SrdTD+0", 2), src=sgpr("SrdD+0", 2), comment="SrdTD = SrdD for GSU == 1"))
        # module.add(SMovB64(dst=sgpr("SrdTD+2", 2), src=sgpr("SrdD+2", 2), comment="SrdTD = SrdD for GSU == 1"))
        # module.add(SBranch(labelName=reductionEndLabel.getLabelName(), comment="branch if GSU == 1"))

      gsuComponent = Component.GSU.find(self)
      if kernel["GlobalSplitU"] > 1 and kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
        self.defineSgpr("GSUStartWGIdx", 1)
        module.add(RegSet("s", "sgprGSUStartWGIdx", self.sgprs["GSUStartWGIdx"]))
        module.addSpaceLine()
        module.add(reductionStartLabel)
        module.add(gsuComponent.reductionBranches(self, kernel, tPB, vectorWidths_1, elements_1, tmpVgpr, cvtVgprStruct, \
          vectorDataTypes, factorDims, reductionEndLabel, endLabel))
        module.add(reductionEndLabel)
        module.add(self.undefineSgpr("GSUStartWGIdx"))
        module.addSpaceLine()


      betaModules = Module("Betas")
      currentInstLength = 0
      for idx0 in reversed(range(len(betas))):
        beta = betas[idx0]
        if beta and kernel["_GlobalAccumulation"] == "SingleBuffer" and (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1):
          continue
        betaModule = Module("Beta_%u"%idx0)
        # start B1
        if beta:
          betaModule.add(betaLabel)
        mod_pos = len(betaModule.items())
        # by now we either jumped to E1 or stayed at E0
        for idx1 in reversed(range(len(edges))):
          edge = edges[idx1]
          loopMN = 2 if (edge and splitMN) else 1
          for idxMN in range(loopMN):
            edgeStr = ""
            if loopMN == 2:
              edgeStr = "_M" if idxMN == 0 else "_N"
            edgeModule = Module("edge_%u%s"%(idx1, edgeStr))

            vectorWidthsNew = vectorWidths
            elementsNew     = elements
            if edge and idxMN == 1:
              vectorWidthsNew = [vectorWidths[0], vectorWidths[0]]
              elementsNew     = [elements[0], elements[0]]

            edge_mode_pos = 0
            for idx2 in range(len(factorDims)):
              edge_mode_pos, currentInstLength, activationTypeStr = \
                  self.globalWriteElementBatch(kernel, tPA, tPB, activation,
                                              applyAlpha, beta, edge, atomic,
                                              vectorWidthsNew, elementsNew, activationLabelList,
                                              tmpVgpr, cvtVgprStruct, activationSetPCStruct, activationEnumStrList,
                                              actPCMaxTempSgpr, isInsertActFunctionCallAddrCalc, toActModuleList,
                                              edgeModule, writeLabels, endLabel,
                                              edge_mode_pos, currentInstLength,
                                              idx0, idx1, idx2, idxMN, vectorDataTypes, factorDims)
            if len(factorDims) == 2:
              isLongBranch = True if currentInstLength >= 16384 else False
              with self.allocTmpSgpr(3) as tmpSgprInfo:
                checkIsFactorDimZero = edgeModule.add(self.checkIsFactorDimZero(kernel, tmpSgprInfo, \
                  writeLabels[beta][edge][factorDims[1]][idxMN], isLongBranch=isLongBranch), pos=edge_mode_pos)
                currentInstLength += countInstruction(checkIsFactorDimZero)

            betaModule.add(edgeModule, pos=mod_pos)

        ########################################
        # branch if Edge0 or Edge1
        if False in edges and True in edges:
          isLongBranch = True if currentInstLength >= 16384 else False
          with self.allocTmpSgpr(4) as tmpSgprInfo:
            labelMT1 = writeLabels[beta][True][factorDims[0]][0] if len(writeLabels[beta][True][factorDims[0]]) == 1 else writeLabels[beta][True][factorDims[0]][1]
            checkIsEdge = betaModule.add(self.checkIsEdge(kernel, tmpSgprInfo, \
              writeLabels[beta][True][factorDims[0]][0], labelMT1, isLongBranch=isLongBranch), pos=mod_pos)
            currentInstLength += countInstruction(checkIsEdge)
        betaModules.add(betaModule, pos=0)

      # Check if branch exceeds
      if False in betas and True in betas:
        isBetaLongBranch = False
        count = 0
        count, found = findInstCount(betaModules, betaLabel, count)
        if found:
          if count >= 16384:
            isBetaLongBranch = True
          with self.allocTmpSgpr(3 if isBetaLongBranch else 1) as tmpSgprInfo:
            module.add(self.checkIsBetaZero(kernel, tmpSgprInfo, betaLabel, isBetaLongBranch, posNeg=1))
      module.appendModule(betaModules)

      if activationLabelList:
        assert activationEnumStrList and activationSetPCStruct
        for key, activationLabelModules in activationLabelList.items():
          gwvw = key
          actModules = Module(getActFuncModuleName(gwvw, \
            activationSetPCStruct.vgprActCopy, tmpVgpr.idx, actTempSgpr))
          for index, activationLabelModule in enumerate(activationLabelModules):
            actModule = Module(activationLabelModule.getLabelName())
            actModule.add(activationLabelModule)
            activationTypeStr = activationEnumStrList[index]
            vgprIdx = activationSetPCStruct.vgprActCopy
            if self.insertActivationAfterPacked(kernel, activationTypeStr):
              actModule.appendModule(self.getActivationDestDataType(kernel, activation, \
                activationTypeStr, gwvw, vgprIdx, vgprIdx, (tmpVgpr.idx + actPCGwvwVgpr), \
                actTempSgpr))
            else:
              actModule.appendModule(self.getActivationActivationComputeType(kernel, activation, \
                activationTypeStr, gwvw, vgprIdx, vgprIdx, (tmpVgpr.idx + actPCGwvwVgpr), \
                actTempSgpr))
            actModule.add(SSetPCB64(src=sgpr(activationSetPCStruct.sgprOffsetBack,2)))
            actModules.add(actModule)
          module.add(actModules)
        self.sgprPool.checkIn(activationSetPCStruct.sgprOffsetActivation)
        self.sgprPool.checkIn(activationSetPCStruct.sgprOffsetBack)

      module.add(skComponent.writePartials(self, kernel, skPartialsLabel, vectorWidths_1, elements_1, tmpVgpr.idx, cvtVgprStruct, endLabel))

      # End label
      module.add(endLabel)

      if kernel["ProblemType"]["UseScaleAB"] == "Scalar" and kernel["StreamK"] > 0 and \
        ((kernel["ProblemType"]["DataTypeA"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters()) or \
        (kernel["ProblemType"]["DataTypeB"].numRegisters() <= kernel["ProblemType"]["DataType"].numRegisters())):
        assert(kernel["ProblemType"]["ComputeDataType"].isSingle())
        module.add(SMovB32(dst=sgpr("Alpha"), src=sgpr(oldAlpha), comment="Restore alpha value"))
        self.sgprPool.checkIn(oldAlpha)

      if self.states.FactorDim == 3:
        self.updateBranchPlaceHolder(module, ["end_placeholder"], [endLabel.label], ["SBranch"])
      self.vgprPool.checkIn(tmpVgpr.idx)
      if cvtVgpr is not None:
        self.vgprPool.checkIn(cvtVgpr)
      if gsuLimit > 1 and gsuLimitIdx == 0:
        with self.allocTmpSgpr(3) as tmpSgprInfo:
          module.add(SLongBranchPositive(Label("KernelEnd", ""), tmpSgprInfo))

    kernel["GlobalSplitU"] = gsuBackup
    kernel["_GlobalAccumulation"] = gsuAccumBackup
    self.states.bpeCexternal = bpeCexternalBackup
    return module

  def getMBSKGSUTotal(self, kernel):
    if kernel["MbskPrefetchMethod"]:
      return kernel["NumMbskPrefetchElements"] + 1
    GSUtotal = 16
    if (kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1]) * (kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]) > 8:
      GSUtotal = int(GSUtotal/int((kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1]) * (kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1])/8))
    GSUtotal = max(2,GSUtotal)
    return GSUtotal

  def setOccupancy(self, kernel):
    # Use VGPR up to next occupancy threshold:
    # Account for additional temp sgprs that will be required for code gen, up to physical limits. +5 approximates upper end of required temp space for GSUSynccodegenOpt
    requiredSgprs = min(self.sgprPool.size() + 5, self.states.regCaps["MaxSgpr"])
    maxVgprs, occupancy = self.getMaxRegsForOccupancy(kernel["NumThreads"], self.vgprPool.size(), requiredSgprs, \
                                                      self.getLdsSize(kernel), self.agprPool.size(), self.states.doubleVgpr)
    # Set occupancy limit for register pools
    # TODO: Support gfx12
    if kernel["ISA"][0] != 12:
      self.vgprPool.setOccupancyLimit(self.states.regCaps["MaxVgpr"], self.states.regCaps["PhysicalMaxVgpr"] // occupancy)
      self.sgprPool.setOccupancyLimit(self.states.regCaps["MaxSgpr"], self.states.regCaps["PhysicalMaxSgpr"] // occupancy)
    return maxVgprs, occupancy

  def refineOccupancy(self, kernel, atomic, elements, actPCMaxTempSgpr, \
                      edgeI, gwvw, maxVgprs, ss):
    # Get estimated numVgprAvailable
    # print("Max vgprs =", maxVgprs, self.vgprPool.size(), self.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align))
    numVgprAvailable = self.vgprPool.availableBlockMaxVgpr(maxVgprs, ss.numVgprsPerElement, ss.align)

    # Grow the register pool if needed - we need enough regs for at least one element
    # Unfortunate since this means the write logic is setting the VGPR requirement
    # for the entire kernel but at least we have a functional kernel.
    # Before growing the pool, see if we can shrink the write vector width instead?
    # TODO : the vgprSerial is needed for-ever and if we grow here will split the
    # range of the tmps.  Maybe want to move vgprSerial to first vgpr?

    # TODO: Minimum elems for StoreRemap
    # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
    minElements = 2 if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) else 1
    if numVgprAvailable < minElements*ss.numVgprsPerElement:
      self.vgprPool.resetOccupancyLimit()
      print2("info: growing pool += %d * %d for GlobalWrite\n" \
          % (minElements,ss.numVgprsPerElement))
      self.vgprPool.growPool(0, minElements, ss.numVgprsPerElement, \
        "grow-pool for GlobalWrite")
      maxVgprsN, occupancy = self.setOccupancy(kernel)
      if maxVgprs != maxVgprsN:
        #print("refineOccupancy maxVgprs, new", maxVgprsN, "old", maxVgprs)
        return self.refineOccupancy(kernel, atomic, elements, actPCMaxTempSgpr, edgeI, gwvw, maxVgprsN, ss)
      numVgprAvailable = self.vgprPool.available()

    # print("NumVgprAvailable", numVgprAvailable)
    if ss.numVgprsPerElement:
      numElementsPerBatch = numVgprAvailable // ss.numVgprsPerElement
    else:
      numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

    assert(self.states.c.numVgprValu % gwvw == 0) # sanity check

    numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

    # print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
    #     "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
    if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
      numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

    # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
    if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
      # only do an even number of halves - since these share hi/lo pieces of some registers?
      if numElementsPerBatch > 1:
        numElementsPerBatch = int(numElementsPerBatch/2)*2
      # dot2: no this constraint
      elif not kernel["EnableMatrixInstruction"] and not kernel["UseDotInstruction"]:
        # The globalWriteBatch routine below can't handle odd elements per batch
        # and 0 elements per batch is illegal.
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        # It might be possible to fix globalWriteBatch to handle this case but these
        # are likely to be low-performing so likely not worth optimizing.
        print2("WARNING: half requires at least two elements per batch")
        self.states.overflowedResources = 3

    assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%self.states.kernelName

    #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
    if atomic and (ss.optSingleColVgpr or ss.optSharedColVgpr):
      # hack to avoid re-using address vgpr across rows
      # atomics need to perform several memory operations
      # if the batch spans multiple rows, need multiple address vgpr
      # which is not currently supported in the two opt*ColVgpr modes
      firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
      numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

    # check best numElementsPerBatch to handle a column block
    # elements of column block must be multiple size of numElementsPerBatch
    nBatchesPerRow = 0
    if kernel["StoreRemapVectorWidth"]:
      firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0] # format for element = (tt1, tt0, vc1, vc0)
      # find the largest factor and smaller than numElementPerBatch
      nBatchesPerRow = 1
      for d in range(1, len(firstRow)+1):
        largestFactor = len(firstRow)//d
        if len(firstRow)%d == 0 and largestFactor <= numElementsPerBatch:
          numElementsPerBatch = largestFactor
          nBatchesPerRow = d
          break

    # if no atomics and no edge, then write whole vectors
    #if not atomic and not edge:
    #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
    #  #print "  NumVectorsPerBatch", numVectorsPerBatch
    #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
    numBatches = max(1, ceilDivide(len(elements[edgeI]),numElementsPerBatch))

    # Grow pool if needed
    # Get true numVgprAvailable
    numVgprAvailable = self.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)
    totalNeededVgpr = ss.numVgprsPerElement * numElementsPerBatch
    # print("Available vgprs =", numVgprAvailable, "Needed vgprs =", totalNeededVgpr, "pool size =", self.vgprPool.size())
    if numVgprAvailable < totalNeededVgpr:
      self.vgprPool.resetOccupancyLimit()
      print2("info: growing pool += %d * %d for GlobalWrite\n" \
          % (numBatches,ss.numVgprsPerElement))
      availableBlock = min(0, self.vgprPool.available() - numVgprAvailable)
      self.vgprPool.growPool(0, totalNeededVgpr + availableBlock, 1, "grow-pool for GlobalWrite")
      maxVgprsN, occupancy = self.setOccupancy(kernel)
      if maxVgprs != maxVgprsN:
        #print("refineOccupancy maxVgprs, new", maxVgprsN, "old", maxVgprs)
        return self.refineOccupancy(kernel, atomic, elements, actPCMaxTempSgpr, edgeI, gwvw, maxVgprsN, ss)

    # # Get true numVgprAvailable
    # numVgprAvailable = self.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)
    # print("Available vgprs =", numVgprAvailable, "pool size =", self.vgprPool.size())

    numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

    if actPCMaxTempSgpr:
      numSgprs = max(actPCMaxTempSgpr, numSgprs)

    self.sgprPool.resetOccupancyLimit()
    self.sgprPool.checkIn(self.sgprPool.checkOutAligned(numSgprs, 2, preventOverflow=False))
    maxVgprsN, occupancy = self.setOccupancy(kernel)
    if maxVgprs != maxVgprsN:
      #print("refineOccupancy maxVgprs, new", maxVgprsN, "old", maxVgprs)
      return self.refineOccupancy(kernel, atomic, elements, actPCMaxTempSgpr, edgeI, gwvw, maxVgprsN, ss)
    return numElementsPerBatch, nBatchesPerRow, numBatches, numSgprs


  ##############################################################################
  # globalWriteElementBatch :
  ##############################################################################
  def globalWriteElementBatch(self, kernel, tPA, tPB, activation, \
                              applyAlpha, beta, edge, atomic, \
                              vectorWidths, elements, activationLabelList, \
                              tmpVgpr, cvtVgprStruct, activationSetPCStruct, activationEnumStrList, \
                              actPCMaxTempSgpr, isInsertActFunctionCallAddrCalc, toActModuleList, \
                              edgeModule, writeLabels, endLabel, \
                              edge_mode_pos, currentInstLength, \
                              idx0, idx1, idx2, idxMN, vectorDataTypes, factorDims):
    factorDim = factorDims[idx2]
    edgeModule.add(writeLabels[beta][edge][factorDim][idxMN])
    if idx2 == 0:
      edge_mode_pos = len(edgeModule.items())

    # for storeRemap edge case, non-beta still can enable vector stores
    if kernel["StoreRemapVectorWidth"] and not beta:
      edgeI = False
    else:
      edgeI = edge
    #edgeI = True  # set to True to disable vector stores
    gwvw = vectorWidths[edgeI]

    #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

    ########################################
    # Calculate Vgprs for Write Batching
    ########################################
    self.vgprPool.resetOccupancyLimit()
    self.sgprPool.resetOccupancyLimit()

    # Temporarily grow pool for sgpr
    sgprList = []
    if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      sgprList.append(self.sgprPool.checkOut(1, preventOverflow=False))
      sgprList.append(self.sgprPool.checkOut(1, preventOverflow=False))
      sgprList.append(self.sgprPool.checkOut(1, preventOverflow=False))
      sgprList.append(self.sgprPool.checkOutAligned(2, 2, preventOverflow=False))
      sgprList.append(self.sgprPool.checkOutAligned(2, 2, preventOverflow=False))
      sgprList.append(self.sgprPool.checkOutAligned(4, 4, preventOverflow=False))
      for s in sgprList:
        self.sgprPool.checkIn(s)
    if actPCMaxTempSgpr > 0:
      self.sgprPool.checkIn(self.sgprPool.checkOutAligned(actPCMaxTempSgpr, 2 if actPCMaxTempSgpr > 1 else 1, preventOverflow=False))

    tmpVgprDynamic = None
    tmpVgprDynamicSize  = 0
    tmpVgprDynamicAlign = 0
    if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      GSUTotal = self.getMBSKGSUTotal(kernel)
      vgprMbsk = (GSUTotal-1) * gwvw * max(1, kernel["ProblemType"]["DestDataType"].numRegisters())
      tmpVgprDynamicSize  = vgprMbsk
      tmpVgprDynamicAlign = 4
    if tmpVgprDynamicSize > 0:
      tmpVgprDynamic = ContinuousRegister(idx=self.vgprPool.checkOutAligned(tmpVgprDynamicSize, tmpVgprDynamicAlign), size=tmpVgprDynamicSize)

    maxVgprs, occupancy = self.setOccupancy(kernel)

    ss = StoreState(self, kernel, gwvw, edge, beta, atomic, elements[edgeI], vectorDataTypes, dim=factorDim)

    actPCMaxTempSgpr_ = None
    if activationLabelList and isInsertActFunctionCallAddrCalc:
      assert activationSetPCStruct, activationEnumStrList and activationLabelList and toActModuleList
      actPCMaxTempSgpr_ = actPCMaxTempSgpr

    numElementsPerBatch, nBatchesPerRow, numBatches, numSgprs = self.refineOccupancy(kernel, atomic, elements, actPCMaxTempSgpr_, edgeI, gwvw, maxVgprs, ss)

    # set atomicW after we potentially resize GWVW
    atomicW = min(gwvw, self.getVectorAtomicWidth(kernel))

    if activationLabelList and isInsertActFunctionCallAddrCalc:
      edgeModule.add(self.insertActFunctionCallAddrCalc(activationSetPCStruct.sgprOffsetActivation, \
        gwvw, toActModuleList, activationEnumStrList, activationLabelList, \
        idx0, idx1))

    if self.db["PrintStoreRegisterDb"]:
      print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", ss.numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
      print ("numSgprs=", numSgprs, "sgprPool.size()=", self.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
            "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
      print(self.sgprPool.state())
    edgeModule.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
        (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
    #edgeModule.addComment("storeStats, %d, %d, %d"% (edgeI, numSgprs, numElementsPerBatch))
    # so if we don't have *GPR resources to handle a larger batch then need
    # to mark overflowedResources rather than generate a kernel that won't work.
    # Activation
    actLoopEndLabel, actLoopLabelModules, actLoopEnumStrList = self.initActivationLoop(kernel, beta, edge)
    actLoopModuleList = []
    actLoopModuleCodeLength = []
    with self.allocTmpSgpr(numSgprs, 2) as tmpSgprRes:
      for index, activationLabelModule in enumerate(actLoopLabelModules):
        actLoopModule = Module("Activation Loop %s"%index)
        activationTypeStr = actLoopEnumStrList[index]
        if activationLabelModule:
          actLoopModule.add(activationLabelModule)

        tmpSgpr = tmpSgprRes.idx
        actTempSgpr = tmpSgpr # Get sgpr start address, should always be the same
        elementSgprs = tmpSgpr + ss.cfg.numTempSgprPerBatch
        codeAccVgprRead = deepcopy(self.codes.accVgprRead) if self.states.serializedStore else None
        mulAlpha = self.codes.mulAlphaMultipleBuffer if (kernel["_GlobalAccumulation"] == 'MultipleBuffer' or kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel') else self.codes.mulAlphaOther
        codeMulAlpha = deepcopy(mulAlpha) if self.states.serializedStore else None

        self.alphaBeforeLoadC = False
        if kernel["MIArchVgpr"] and applyAlpha and not kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
          codeAccVgprRead = None

          #Only apply when 2 wave optimization features are enabled
          if (kernel["StorePriorityOpt"] or kernel["StoreSyncOpt"]) and beta:
            self.alphaBeforeLoadC = True
          #When LSU>1, don't use the VGPRs from the endSum.
          if (kernel["LocalSplitU"] > 1):
            codeMulAlpha = None
        else:
          codeMulAlpha = None

        biasLocalBarrierInit = False
        # If LSU, the VGPRs are from LSU reduction.
        # We need a variable to read from correct VGPR index when numBatches > 1.
        ss.lsuStartVgprOffset = 0
        for batchIdx in range(0, numBatches):
          elementStartIdx = batchIdx * numElementsPerBatch
          elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
          elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
          #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,ss.numVgprsPerElement ))
          # elementVgprs can be large and should be perfectly tuned to the number of available
          # VGPRS.  We do not want to accidentally overflow and grow the pool here:

          if kernel["StoreRemapVectorWidth"]:
            #Indication if this batch is last batch for this column block shape
            self.StoreRemapLastBatch = 1 if (batchIdx+1) % nBatchesPerRow == 0 else 0

          actLoopModule.add(self.globalWriteBatch(kernel, tPA, tPB, activation, ss, batchIdx, \
              applyAlpha, beta, edge, atomic, gwvw, atomicW, \
              elementsThisBatch, self.vgprs.addrE, self.vgprs.addrD, self.vgprs.addrC, self.vgprs.addrBias, \
              self.vgprs.addrScaleAVec, self.vgprs.addrScaleBVec, self.vgprs.addrScaleAlphaVec, \
              biasLocalBarrierInit, tmpVgpr, tmpVgprDynamic, cvtVgprStruct, activationSetPCStruct, \
              activationTypeStr, elementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, factorDim))
          biasLocalBarrierInit = True

        ss.resetState()
        actLoopModuleList.append(actLoopModule)
        actLoopModuleCodeLength.append(countInstruction(actLoopModule))

    #################
    # Free after final vgpr vcalculation
    if tmpVgprDynamic:
      self.vgprPool.checkIn(tmpVgprDynamic.idx)

    if len(actLoopLabelModules) > 1:
      actInstCounter = 0
      # Add activation branch
      for index, actLoopLabelModule in enumerate(actLoopLabelModules):
        enumIndex = ActivationType.getEnumIndex(actLoopEnumStrList[index])
        #edgeModule.add(SCmpKEQU32(sgpr("ActivationType"), enumIndex, "activationType == %u"%enumIndex))
        edgeModule.add(self.getSCMPKInstruction("EQU32", "ActivationType", enumIndex, comment="activationType == %u"%enumIndex))
        if actInstCounter >= 16384:
          edgeModule.add(self.longBranchScc1(actLoopLabelModule, posNeg=1, comment="Branch if true"))
        else:
          edgeModule.add(SCBranchSCC1(actLoopLabelModule.getLabelName(), "Branch if true"))
        actInstCounter += actLoopModuleCodeLength[index]
      # Add jump to activation end
      for index, _ in enumerate(actLoopLabelModules):
        actLoopModule = actLoopModuleList[index]
        if (index < (len(actLoopLabelModules) - 1)):
          if actInstCounter >= 16384:
            with self.allocTmpSgpr(3) as tmpSgprInfo:
              actLoopModule.add(SLongBranchPositive(actLoopEndLabel, tmpSgprInfo))
          else:
            actLoopModule.add(SBranch(labelName=actLoopEndLabel.getLabelName()))
        actInstCounter -= actLoopModuleCodeLength[index]

    # Append to edgeModule
    for actLoopModule in actLoopModuleList:
      edgeModule.appendModule(actLoopModule)
    # Add actLoopEndLabel if needed
    if len(actLoopLabelModules) > 1:
      edgeModule.add(actLoopEndLabel)

    if len(factorDims) == 1:
      if currentInstLength >= 16384:
        with self.allocTmpSgpr(3) as tmpSgprInfo:
          edgeModule.add(SLongBranchPositive(endLabel, tmpSgprInfo, comment="jump to end"))
      else:
        edgeModule.add(SBranch(labelName=endLabel.getLabelName(), comment="jump to end"))
    else:
      end_placeholder = Module("end_placeholder")
      edgeModule.add(end_placeholder)
    currentInstLength += countInstruction(edgeModule)
    del ss

    return edge_mode_pos, currentInstLength, activationTypeStr

  ##############################################################################
  # chooseGlobalRead :
  # create the load instruction for requested vector width and other parms
  # return an Inst class
  #
  # bpl = bytes per load op
  # tr = transpose read (for matrix instruction)
  ##############################################################################
  def chooseGlobalRead(self, useBuffer, bpl, destVgpr, \
                       addr0, addr1, soffset, offset, \
                       glc=False, slc=False, nt=False, lds=False, \
                       tr=False, hi16=0, comment="load C"):
    # rpv = regs per vector
    rpv = bpl/4.0

    if useBuffer:
      rv = Module("Global Read")
      mubuf = MUBUFModifiers(offen=True, offset12=offset, glc=glc, slc=slc, nt=nt, lds=lds)

      # Nested buffer load implementation function for easy branching for soffset
      def bufferLoadImpl(soffset):
        nonlocal rv
        factor = max(1, 4//bpl)
        dst = None if lds else vgpr(destVgpr, rpv*factor)
        if bpl==1 and hi16:
          rv.add(BufferLoadD16HIU8(dst=dst, vaddr=addr0, saddr=addr1, \
                                  soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==1 and not hi16:
          rv.add(BufferLoadD16U8(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==2 and hi16:
          rv.add(BufferLoadD16HIB16(dst=dst, vaddr=addr0, saddr=addr1, \
                                    soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==2 and not hi16:
          rv.add(BufferLoadD16B16(dst=dst, vaddr=addr0, saddr=addr1, \
                                  soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==4:
          rv.add(BufferLoadB32(dst=dst, vaddr=addr0, saddr=addr1, \
                              soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==8:
          rv.add(BufferLoadB64(dst=dst, vaddr=addr0, saddr=addr1, \
                              soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==16:
          rv.add(BufferLoadB128(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==32:
          # split into two dwordx4 loads. Second load offset is +0.5 bpl
          rv = Module("emulated _buffer_load_b256")
          dst = None if lds else vgpr(destVgpr, rpv//2)
          rv.add(BufferLoadB128(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))
          mubuf2 = MUBUFModifiers(offen=True, offset12=int(offset + bpl/2), glc=glc, slc=slc, nt=nt, lds=lds)
          if isinstance(destVgpr, str):
            dst2 = destVgpr + "+" + str(int(rpv//2))
          elif isinstance(destVgpr, int):
            dst2 = int(destVgpr + int(rpv//2))
          dst = None if lds else vgpr(dst2, rpv//2)
          rv.add(BufferLoadB128(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf2, comment=comment))
          return rv
        elif bpl==64 and not lds:
          rv = Module("emulated _buffer_load_b512")
          # +0.25
          dst = vgpr(destVgpr, rpv//4)
          rv.add(BufferLoadB128(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))

          mubuf2 = MUBUFModifiers(offen=True, offset12=int(offset + bpl/4), glc=glc, slc=slc, nt=nt, lds=lds)
          dst2 = destVgpr + "+" + str(int(rpv//4)) if isinstance(destVgpr, str) else int(destVgpr + int(rpv//4))

          dst = vgpr(dst2, rpv//4)
          rv.add(BufferLoadB128(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf2, comment=comment))
          #+0.5
          mubuf3 = MUBUFModifiers(offen=True, offset12=int(offset + bpl/2), glc=glc, slc=slc, nt=nt, lds=lds)
          dst3 = destVgpr + "+" + str(int(rpv//2)) if isinstance(destVgpr, str) else int(destVgpr + int(rpv//2))

          dst = vgpr(dst3, rpv//4)
          rv.add(BufferLoadB128(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf3, comment=comment))
          #+0.75
          mubuf4 = MUBUFModifiers(offen=True, offset12= int(offset + 3*bpl/4), glc=glc, slc=slc, nt=nt, lds=lds)
          dst4 = destVgpr + "+" + str(3*int(rpv//4)) if isinstance(destVgpr, str) else int(destVgpr + 3*int(rpv//4))

          dst = vgpr(dst4, rpv//4)
          rv.add(BufferLoadB128(dst=dst, vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf4, comment=comment))
        else:
          assert 0, "%s\nchooseGlobalRead: bad bpl %u"%(self.states.kernelName,bpl)

      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      if offset >= 4096:
        if soffset in (0, "0"):
          mubuf = MUBUFModifiers(offen=True, offset12=0, glc=glc, slc=slc, nt=nt, lds=lds)
          with self.allocTmpSgpr(1) as tmpSgprInfo:
            soffset = sgpr(tmpSgprInfo.idx)
            rv.add(SMovB32(dst=soffset, src=offset, comment="large offset"))
            bufferLoadImpl(soffset)
        else:
          assert 0, "offset too large and soffset set"
      else:
        bufferLoadImpl(soffset)

      return rv
    elif tr:
      # global_load offset field is 24-bit.
      if offset >= 1<<24:
        raise RuntimeError("chooseGlobalRead: offset too large for global_load %u"%(offset))

      modifier = GLOBALModifiers(offset=offset)

      if bpl==8:
        return GlobalLoadTR8B64(dst=vgpr(destVgpr, rpv), vaddr=addr0, saddr=addr1, modifier=modifier, comment=comment)
      elif bpl==16:
        return GlobalLoadTR16B128(dst=vgpr(destVgpr, rpv), vaddr=addr0, saddr=addr1, modifier=modifier, comment=comment)
    else:
      flat = FLATModifiers(glc=glc, slc=slc, lds=lds)
      if bpl==2 and hi16:
        return FlatLoadD16HIB16(dst=vgpr(destVgpr, rpv*2), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==2 and not hi16:
        return FlatLoadD16B16(dst=vgpr(destVgpr, rpv*2), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==4:
        return FlatLoadB32(dst=vgpr(destVgpr, rpv), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==8:
        return FlatLoadB64(dst=vgpr(destVgpr, rpv), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==16:
        return FlatLoadB128(dst=vgpr(destVgpr, rpv), vaddr=addr0, flat=flat, comment=comment)
      else:
        assert 0, "chooseGlobalRead: bad bpl"

  ##############################################################################
  def chooseGlobalWrite(self, useBuffer, bps, srcVgpr, rpv, \
                        addr0, addr1, offset, soffset=0, \
                        glc=False, slc=False, nt=False, hi16=0, comment="store"):
    """
    create the store instruction for requested vector width and other parms
    rpv = regs per vector
    """

    module = Module("chooseGlobalWrite %s -> %s (%s)"%(srcVgpr, addr0, addr1))

    def bufferStoreImpl(tmpSgpr, mubuf):
      if bps==1 and hi16:
        module.add(BufferStoreD16HIB16(src=vgpr(srcVgpr, rpv*4), vaddr=addr0, \
                                       saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))
      elif bps==1 and not hi16:
        module.add(BufferStoreB8(src=vgpr(srcVgpr, rpv*4), vaddr=addr0, \
                                 saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))
      elif bps==2 and hi16:
        module.add(BufferStoreD16HIB16(src=vgpr(srcVgpr, rpv*2), vaddr=addr0, \
                                       saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))
      elif bps==2 and not hi16:
        module.add(BufferStoreB16(src=vgpr(srcVgpr, rpv*2), vaddr=addr0, \
                                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))
      elif bps==4:
        module.add(BufferStoreB32(src=vgpr(srcVgpr, rpv), vaddr=addr0, \
                                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))
      elif bps==8:
        module.add(BufferStoreB64(src=vgpr(srcVgpr, rpv), vaddr=addr0, \
                                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))
      elif bps==16:
        module.add(BufferStoreB128(src=vgpr(srcVgpr, rpv), vaddr=addr0, \
                                   saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))

      elif bps >= 32 and bps % 32 == 0:
        # split into several dwordx4 loads. Offset the next by +0.5 bps
        rounds = bps // 16
        shiftByte = bps // rounds
        shiftRpv = rpv // rounds
        module.add(BufferStoreB128(src=vgpr(srcVgpr, shiftRpv), vaddr=addr0, \
                                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment=comment))
        for i in range(1, rounds):
          offset2 = offset+shiftByte*i
          if offset2 >= 4096:
            module.add(SMovB32(dst=tmpSgpr, src=offset2, comment="large offset"))
            offset2 = 0
          mubuf2 = MUBUFModifiers(offen=True, offset12=offset2, glc=glc, slc=slc, nt=nt, isStore=True)
          vgprOff = int(srcVgpr + shiftRpv * i) if isinstance(srcVgpr, int) else f"{srcVgpr}+{int(shiftRpv * i)}"
          module.add(BufferStoreB128(src=vgpr(vgprOff, shiftRpv), vaddr=addr0, \
                saddr=addr1, soffset=tmpSgpr, mubuf=mubuf2, comment=comment))
      else:
        assert 0, "bad bps"

    if useBuffer:
      mubuf = MUBUFModifiers(offen=True, offset12=offset, glc=glc, slc=slc, nt=nt, isStore=True)
      if soffset != 0:
        assert offset < 4096, "sgpr offset provided with large const offset"
      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      maxShift = max(bps - 16, 0) #if bps = 32 or bps = 64
      if (offset + maxShift) >= 4096:
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          tmpSgpr = sgpr(tmpSgprInfo.idx)
          if offset >= 4096:
            module.add(SMovB32(dst=tmpSgpr, src=offset, comment="large offset"))
            mubuf = MUBUFModifiers(offen=True, offset12=0, glc=glc, slc=slc, nt=nt, isStore=True)
          bufferStoreImpl(tmpSgpr, mubuf)
      else:
        bufferStoreImpl(soffset, mubuf)

    else:
      flat = FLATModifiers(glc=glc, slc=slc, isStore=True)
      if bps==2 and hi16:
        module.add(FlatStoreD16HIB16(vaddr=addr0, src=vgpr(srcVgpr*2), flat=flat, comment=comment))
      elif bps==2 and not hi16:
        module.add(FlatStoreD16B16(vaddr=addr0, src=vgpr(srcVgpr, rpv*2), flat=flat, comment=comment))
      elif bps==4:
        module.add(FlatStoreB32(vaddr=addr0, src=vgpr(srcVgpr, rpv), flat=flat, comment=comment))
      elif bps==8:
        module.add(FlatStoreB64(vaddr=addr0, src=vgpr(srcVgpr, rpv), flat=flat, comment=comment))
      elif bps==16:
        module.add(FlatStoreB128(vaddr=addr0, src=vgpr(srcVgpr, rpv), flat=flat, comment=comment))
      else:
         assert 0, "bad bps"

    return module

  def addVecGlobalLoad(self, dataType, kernel, vecVgpr, addr0, addr1, offset, gwvw, comment=""):
    """
    Add vec for the element with addrCalc, elementIdx, and vecVgpr.
    vecVgpr is one or more vgpr :temp vGPR ( = gwvw * numbytes // 4 + 1 if cvt is needed)
    """
    # Add vec here
    module = Module(comment)
    bps = dataType.numBytes() * gwvw

    useBuffer = kernel["BufferLoad"]
    if dataType.isHalf() or dataType.isBFloat16():
      module.add(self.chooseGlobalRead(useBuffer, bps, vecVgpr, \
                        addr0, addr1, soffset=0, offset=offset, hi16=0, comment=comment))
    elif dataType.isInt32() or dataType.isSingle():
      module.add(self.chooseGlobalRead(useBuffer, bps, vecVgpr, \
                        addr0, addr1, soffset=0, offset=offset, comment=comment))
    elif dataType.isDouble() or dataType.isSingleComplex() :
      module.add(self.chooseGlobalRead(useBuffer, bps, vecVgpr, \
                        addr0, addr1, soffset=0, offset=offset, comment=comment))
    else:
      printExit("Unsupported %s type %s."%(comment, str(dataType)))
    return module

  ##############################################################################
  def addScaleVecLoad(self, kernel, ss, name: str, srdName: str, addrScaleVecVgpr, scaleVecVgpr, gwvw, scaleVecOffset, addVecPostfix = True):
    """
    Add scaleAlphaVec for the element with addrCalc, elementIdx, and scaleVecVgpr.
    scaleVecVgpr is one or more vgpr :temp vGPR ( = gwvw * numbytes // 4 + 1 if cvt is needed)
    """
    module = Module("addScale%sVec"%srdName)
    if kernel["ProblemType"]["UseScale%s"%name] and ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel' or kernel["StreamK"] > 0):
      bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw
      if kernel["BufferLoad"]:
        addr0 = vgpr(addrScaleVecVgpr)
        addr1 = sgpr("SrdScale%sVec"%srdName, 4) if addVecPostfix else sgpr("SrdScale%s"%srdName, 4)
      else:
        addr0 = vgpr(addrScaleVecVgpr,2)
        addr1 = ""

      useBuffer = kernel["BufferLoad"]

      if kernel["ProblemType"]["ComputeDataType"].isHalf() or kernel["ProblemType"]["ComputeDataType"].isBFloat16():
        module.add(self.chooseGlobalRead(useBuffer, bps, scaleVecVgpr, \
                          addr0, addr1, soffset=0, offset=scaleVecOffset, hi16=0, comment="load scale%sVecH"%srdName))
      elif kernel["ProblemType"]["ComputeDataType"].isInt32() or kernel["ProblemType"]["ComputeDataType"].isSingle():
        module.add(self.chooseGlobalRead(useBuffer, bps, scaleVecVgpr, \
                          addr0, addr1, soffset=0, offset=scaleVecOffset, comment="load scale%sVecI"%srdName))
      elif kernel["ProblemType"]["ComputeDataType"].isDouble() or kernel["ProblemType"]["ComputeDataType"].isSingleComplex() :
        module.add(self.chooseGlobalRead(useBuffer, bps, scaleVecVgpr, \
                          addr0, addr1, soffset=0, offset=scaleVecOffset, comment="load scale%sVec"%srdName))
      else:
        printExit("Unsupported scale%sVec type %s."%(srdName, str(kernel["ProblemType"]["ComputeDataType"])))

    return module

  def addLdsLoad(self, dataType, dstVgpr, srcAddrVgpr, dsOffset, gwvw, comment=""):
      module = Module(comment)
      dst = vgpr(dstVgpr)
      src = vgpr(srcAddrVgpr)
      ds = DSModifiers(offset=dsOffset)
      bpl = dataType.numBytes() * gwvw
      if bpl==2:
        module.add(DSLoadU16(dst=dst, src=src, ds=ds, comment=comment))
      elif bpl==4:
        module.add(DSLoadB32(dst=dst, src=src, ds=ds, comment=comment))
      elif bpl==8:
        module.add(DSLoadB64(dst=vgpr(dstVgpr, 2), src=src, ds=ds, comment=comment))
      elif bpl==16:
        module.add(DSLoadB128(dst=vgpr(dstVgpr, 4), src=src, ds=ds, comment=comment))
      elif bpl==32:
        module.add(DSLoadB128(dst=vgpr(dstVgpr, 4), src=src, ds=ds, comment=comment))
        ds = DSModifiers(offset=int(dsOffset+bpl/2))
        module.add(DSLoadB128(dst=vgpr(dstVgpr+4, 4), src=src, ds=ds, comment=comment))
      elif bpl==64:
        module.add(DSLoadB128(dst=vgpr(dstVgpr, 4), src=src, ds=ds, comment=comment))
        ds = DSModifiers(offset=int(dsOffset+bpl/4))
        module.add(DSLoadB128(dst=vgpr(dstVgpr+4, 4), src=src, ds=ds, comment=comment))
        ds = DSModifiers(offset=int(dsOffset+2*bpl/4))
        module.add(DSLoadB128(dst=vgpr(dstVgpr+8, 4), src=src, ds=ds, comment=comment))
        ds = DSModifiers(offset=int(dsOffset+3*bpl/4))
        module.add(DSLoadB128(dst=vgpr(dstVgpr+12, 4), src=src, ds=ds, comment=comment))
      else:
        assert 0, "bad bpl"
      return module

  def addBiasLoad(self, dataType, kernel, gwvw, addrCalc, biasVgpr, factorDim, isLocal=False):
    if isLocal and (self.states.useBias == DataDirection.READ):
      return self.addLdsLoad(dataType, biasVgpr, addrCalc.addrBiasVgpr, addrCalc.biasOffset[factorDim], gwvw, comment="Load Bias")

    if self.states.useBias == DataDirection.READ:
      if kernel["BufferLoad"]:
        addr0 = vgpr(addrCalc.addrBiasVgpr)
        addr1 = sgpr("SrdBias", 4)
      else:
        addr0 = vgpr(addrCalc.addrBiasVgpr,2)
        addr1 = ""
    else:
      return Module("Empty load")
    return self.addVecGlobalLoad(dataType, kernel, biasVgpr, addr0, addr1, addrCalc.biasOffset[factorDim], gwvw, comment="Load Bias")

  ##############################################################################
  def addStore(self, kernel, ss, tc: str, addrCalc, sumIdx, tmpS01, edge, wsOffset=0, comment="addStore"):
    """
    Add stores for the element with addrCalc and sumIdx.
    tmpS01 is a single :temp sGPR
    """
    module = Module("addStore sumIdx %s"%(str(sumIdx)))
    if self.do["GlobalWrite"]:
      # perform vector stores here, so no VI indexing.
      # if GWVW > Vw, might need to support loops to
      # implement wider stores
      isGlc = False
      isSlc = False
      isNT = False

      if tc == 'D':
        isGlc = bool(kernel["NonTemporalD"] & 0x1)
        isSlc = bool(kernel["NonTemporalD"] & 0x2)
        isNT  = bool(kernel["NonTemporalD"] & 0x4)
        bps = self.states.bpeCexternal * ss.cfg.gwvw
        rpv = self.states.bpeCexternal * ss.cfg.gwvw / self.states.bpr

        if kernel["BufferStore"]:
          addr0 = vgpr(addrCalc.addrDVgpr)
          addr1 = sgpr("SrdD", 4)
        else:
          addr0 = vgpr(addrCalc.addrDVgpr,2)
          addr1 = ""
        if ss.optSrdIncForRow and addrCalc.rowInc:
          module.add(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS01))

        dataType     = kernel["ProblemType"]["DestDataType"]
        globalOffset = addrCalc.globalOffset
      elif tc == 'TD':
        isGlc = True
        isSlc = True

        bps = kernel["ProblemType"]["DestDataType"].numBytes() * ss.cfg.gwvw
        rpv = kernel["ProblemType"]["DestDataType"].numBytes() * ss.cfg.gwvw / self.states.bpr
        if kernel["BufferStore"]:
          addr0 = vgpr(addrCalc.addrGSUSyncVgprs)
          addr1 = sgpr("SrdTD", 4)
        else:
          addr0 = vgpr(addrCalc.addrGSUSyncVgprs,2)
          addr1 = ""
        if ss.optSrdIncForRow and addrCalc.rowInc:
          module.add(addrCalc.incrementToNextRow(kernel, "TD", ss, tmpS01))
        dataType     = kernel["ProblemType"]["DestDataType"]
        globalOffset = addrCalc.globalOffset
        globalOffset = int((globalOffset/self.states.bpeCexternal) * self.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())
      elif tc == 'WS':
        isGlc = True
        isSlc = True
        isNT  = bool(kernel["NonTemporalD"] & 0x4)

        bps = self.states.bpeCinternal * ss.cfg.gwvw
        rpv = self.states.bpeCinternal * ss.cfg.gwvw / self.states.bpr
        if kernel["BufferStore"]:
          addr0 = vgpr(addrCalc.addrDVgpr)
          addr1 = sgpr("SrdWS", 4)
        else:
          addr0 = vgpr(addrCalc.addrDVgpr,2)
          addr1 = ""
        dataType     = kernel["ProblemType"]["ComputeDataType"]
        globalOffset = 0
      elif tc == 'Bias':
        bps = self.states.bpeCinternal * ss.cfg.gwvw
        rpv = self.states.bpeCinternal * ss.cfg.gwvw / self.states.bpr

        if kernel["BufferStore"]:
          addr0 = vgpr(addrCalc.addrBiasVgpr)
          addr1 = sgpr("Srd%s"%tc, 4)
        else:
          addr0 = vgpr(addrCalc.addrBiasVgpr,2)
          addr1 = ""
        if ss.optSrdIncForRow and addrCalc.rowInc:
          module.add(addrCalc.incrementToNextRow(kernel, tc, ss, tmpS01, bpeType=self.states.bpeCinternal))
        dataType     = kernel["ProblemType"]["ComputeDataType"]
        globalOffset = addrCalc.globalOffsetInternal
      elif tc == 'E':
        bps = self.states.bpeE * ss.cfg.gwvw
        rpv = self.states.bpeE * ss.cfg.gwvw / self.states.bpr

        if kernel["BufferStore"]:
          addr0 = vgpr(addrCalc.addrEVgpr)
          addr1 = sgpr("Srd%s"%tc, 4)
        else:
          addr0 = vgpr(addrCalc.addrEVgpr,2)
          addr1 = ""
        if ss.optSrdIncForRow and addrCalc.rowInc:
          module.add(addrCalc.incrementToNextRow(kernel, tc, ss, tmpS01, bpeType=self.states.bpeE))
        dataType     = kernel["ProblemType"]["DataTypeE"]
        globalOffset = addrCalc.globalOffsetE
      else:
        printExit("Unsupported store tc %s"%tc)

      useBuffer = kernel["BufferStore"]
      if dataType.isHalf() or dataType.isBFloat16():
        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (H,H,H,H,H,H), internal H
          if self.states.asmCaps["HasWMMA_V1"] and kernel["EnableMatrixInstruction"]:
            module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                addr0, addr1, globalOffset, soffset=wsOffset, \
                glc=isGlc, slc=isSlc, nt=isNT, hi16=0, comment=comment))
          else:
            module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx//2, rpv, \
                addr0, addr1, globalOffset, soffset=wsOffset, \
                glc=isGlc, slc=isSlc, nt=isNT, hi16=sumIdx%2, comment=comment))
        else:
          # (B,B,B,B,S,S), internal S
          # (H,H,H,H,H,H), internal S
          # (H,H,H,H,S,S), internal S
          module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
              addr0, addr1, globalOffset, soffset=wsOffset, \
              glc=isGlc, slc=isSlc, nt=isNT, hi16=0, comment=comment))
      elif dataType.isInt32() or dataType.isSingle():
        module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
            addr0, addr1, globalOffset, soffset=wsOffset, \
            glc=isGlc, slc=isSlc, nt=isNT, comment=comment))
      elif dataType.isDouble() or dataType.isSingleComplex():
        module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx*2, rpv, \
            addr0, addr1, globalOffset, soffset=wsOffset, \
            glc=isGlc, slc=isSlc, nt=isNT, comment=comment))
      elif dataType.isDoubleComplex():
        rps = dataType.numRegisters()
        module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx*rps, rpv, \
            addr0, addr1, globalOffset, soffset=wsOffset, \
            glc=isGlc, slc=isSlc, nt=isNT, comment=comment))
      elif dataType.isInt8() or dataType.isAnyFloat8() or dataType.isAnyBFloat8() or dataType.isAnyFloat8BFloat8() or dataType.isAnyBFloat8Float8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
              addr0, addr1, globalOffset, soffset=wsOffset, \
              glc=isGlc, slc=isSlc, nt=isNT, comment=comment))
    return module

  ##############################################################################
  # Global Read Input
  ##############################################################################
  def readInput(self, kernel, ss, tc: str, dataType, addrCalc, vc0, data, gwvw, addr, tmpS01):
    module = Module("read%sInput"%tc)
    bps = dataType.numBytes() * gwvw
    useBuffer = kernel["BufferStore"]

    if kernel["BufferStore"]:
      addr0 = vgpr(addr)
      addr1 = sgpr("Srd%s"%tc, 4)
    else:
      addr0 = vgpr(addr,2)
      addr1 = ""

    isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
    isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
    isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

    soffset = 0
    if tc == 'E':
      globalOffset = addrCalc.globalOffsetE
      bpeType = self.states.bpeE
    elif tc == 'WS':
      soffset = tmpS01
      globalOffset = 0
      bpeType = self.states.bpeCinternal
    else:
      if dataType == kernel["ProblemType"]["ComputeDataType"]:
        globalOffset = addrCalc.globalOffsetInternal
        bpeType = self.states.bpeCinternal
      else:
        globalOffset = addrCalc.globalOffset
        bpeType = self.states.bpeCexternal
        if tc == 'C':
          if kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
            globalOffset = int((globalOffset/self.states.bpeCexternal) * self.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())

    isWorkspace = tc == 'WS'
    if ss.optSrdIncForRow and addrCalc.rowInc and not isWorkspace:
      module.add(addrCalc.incrementToNextRow(kernel, tc, ss, tmpS01, bpeType=bpeType))

    if dataType.isHalf():
      hi16 = 0 if self.states.HHH_WMMA else (vc0 % 2)
      module.add(self.chooseGlobalRead(useBuffer, bps, data, \
          addr0, addr1, soffset=soffset, offset=globalOffset, \
          glc=isGlc, slc=isSlc, nt=isNT, lds=False, hi16=hi16, \
          comment="load %s"%tc))
    elif dataType.isInt8() or dataType.is8bitFloat():
     module.add(self.chooseGlobalRead(useBuffer, bps, data, \
          addr0, addr1, soffset=soffset, offset=globalOffset, \
          glc=isGlc, slc=isSlc, nt=isNT, lds=False, \
          #hi16=vc0 % 4,
          comment="load %s"%tc))
    elif dataType.isBFloat16() or \
         dataType.isInt32() or \
         dataType.isSingle() or \
         dataType.isDouble() or \
         dataType.isSingleComplex() or \
         dataType.isDoubleComplex():
      module.add(self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=soffset, offset=globalOffset, \
                glc=isGlc, slc=isSlc, nt=isNT, lds=False, \
                comment="load %s"%tc))

    return module

  ##############################################################################
  # Global Write Batch
  ##############################################################################
  def globalWriteBatch(self, kernel, tPA, tPB, activation, ss: StoreState, batchIdx, \
      applyAlpha, beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrE, addrD, addrC, addrBias, \
      addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, biasLocalBarrierInit: bool, \
      tmpVgpr, tmpVgprDynamic, cvtVgprStruct, activationSetPCStruct, activationTypeStr, \
      batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, factorDim) -> Module:
      packdata = Component.PackData.find(self)
      gwriter  = Component.GlobalWriteComponents.find(self)
      return gwriter(kernel, tPA, tPB, activation, ss, \
        batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
        batchElements, addrE, addrD, addrC, addrBias, \
        addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, biasLocalBarrierInit, \
        tmpVgpr, tmpVgprDynamic, cvtVgprStruct, activationSetPCStruct, activationTypeStr, \
        batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, packdata, self, factorDim, \
        self.assembler.version)

  ##############################################################################
  def openPrefetchGlobalRead2(self, kernel):
    imod = Module()
    loopCounter = self.loopCounter(kernel, self.states.unrollIdx)
    imod.add(SCmpEQU32(src0=loopCounter, src1=hex(1), comment="PGR=2 but only 1 loop"))
    skipPGR2 = Label(self.labels.getName("skipPGR2"), "")
    imod.add(SCBranchSCC1(labelName=skipPGR2.getLabelName(), comment="PGR=2 but only 1 loop"))
    return imod

  def closePrefetchGlobalRead2(self):
    imod = Module()
    skipPGR2 = Label(self.labels.getName("skipPGR2"), "")
    imod.add(skipPGR2)
    return imod

  ########################################
  # Read vector to LDS
  ########################################
  def calculateVectorGlobalOffset(self, kernel, offsetVgpr, tmpSgpr, dim):
    module = Module("")
    def calculateVectorGlobalOffsetCommon(s):
      module.add(SMulI32(dst=sgpr(s), src0=kernel["MacroTile%d"%dim], src1=sgpr("WorkGroup%d"%dim), comment="wgp%d * MT%d"%(dim, dim)))
      module.add(VAddU32(dst=vgpr(offsetVgpr), src0=sgpr(s), src1=vgpr("Serial"), comment=f"coord {dim} = wgp{dim} * MT{dim} + thread offset"))
    if tmpSgpr:
      calculateVectorGlobalOffsetCommon(tmpSgpr)
    else:
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        calculateVectorGlobalOffsetCommon(tmpSgpr)
    return module

  def calculateVectorGlobalStride(self, offsetInVgpr, offsetOutVgpr, tmpSgpr, dim, strideName:str):
    module = Module("")
    def calculateVectorGlobalStrideCommon(s):
      module.add(SMulI32(dst=sgpr(s), src0=sgpr(strideName), src1=sgpr("WorkGroup2"), comment="Stride * WG"))
      module.add(VAddU32(dst=vgpr(offsetOutVgpr), src0=sgpr(s), src1=vgpr(offsetInVgpr), comment=f"coord {dim} = wgp{dim} * MT{dim} + thread offset + Stride * WG"))
    if tmpSgpr:
      calculateVectorGlobalStrideCommon(tmpSgpr)
    else:
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        calculateVectorGlobalStrideCommon(tmpSgpr)
    return module

  def getGlobalShiftOffset(self, kernel, dataType, gwvw):
    numVgprs  = int(ceil(dataType.numRegisters() * gwvw))
    reg = dataType.numRegisters() if dataType.numRegisters() >= kernel["ProblemType"]["ComputeDataType"].numRegisters() \
      else kernel["ProblemType"]["ComputeDataType"].numRegisters()
    return gwvw * reg - numVgprs

  def getTurn(self, kernel, gwvw, dim):
    divisor = kernel["SubGroup0"] * kernel["SubGroup1"]
    turn    = ceil(kernel["MacroTile%d"%dim] / (divisor * gwvw))
    return turn, divisor

  def addVectorGlobalLoad(self, kernel, srdName: str, offsetVgpr, shiftOffset, dataType, bpe, gwvw, tmpVgpr1Res: ContinuousRegister, dstOffset, dim):
    module        = Module("")
    tmpVgpr1      = tmpVgpr1Res.idx + dstOffset
    turn, divisor = self.getTurn(kernel, gwvw, dim)
    addr0         = vgpr(offsetVgpr)
    addr1         = sgpr("Srd%s"%srdName, 4)
    offset        = (divisor * gwvw) * bpe

    for i in range(turn):
      if i != 0:
        module.add(VAddU32(dst=vgpr(offsetVgpr), src0=offset, src1=vgpr(offsetVgpr), comment="add subgroup offset"))
      module.add(self.addVecGlobalLoad(dataType, kernel, tmpVgpr1 + shiftOffset, addr0, addr1, 0, gwvw, comment="Load %s"%srdName))
      # TODO: Will this work if gwvw > 1?
      tmpVgpr1 += 1
    return module

  def addVectorLocalStore(self, kernel, addressStr: str, offsetVgpr, shiftOffset, dataType, gwvw, tmpVgpr1Res: ContinuousRegister, srcOffset, subGroupOffset, dim, setToOne=False, comment=""):
    module        = Module("")
    tmpVgpr1      = tmpVgpr1Res.idx + srcOffset
    turn, divisor = self.getTurn(kernel, gwvw, dim)
    offset        = (divisor * gwvw) * self.states.bpeCinternal

    if setToOne:
      module.add(VCmpGtU32(dst=sgpr("Address%s"%addressStr, self.states.laneSGPRCount), src0=sgpr("Srd%s+2"%addressStr), src1=0, comment=" == 0 ?"))
      # Set maskConst to 1.0 or 1
      if kernel["ProblemType"]["ComputeDataType"].isSingle():
        maskConst = 1.0
      elif kernel["ProblemType"]["ComputeDataType"].isInt32():
        maskConst = 1

    turnOffset = 0
    for i in range(turn):
      if i != 0:
        turnOffset += offset
      bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw
      ds  = DSModifiers(offset=(subGroupOffset[0] + turnOffset))
      dst = vgpr(offsetVgpr)
      for vi in range(gwvw):
        # Does not support hi/lo yet
        shiftOffset2 = shiftOffset + int(vi * dataType.numRegisters())
        if kernel["ProblemType"]["ComputeDataType"].isSingle():
          if dataType.isHalf():
            module.add(VCvtF16toF32(dst=vgpr(tmpVgpr1 + vi + i * gwvw), src=vgpr(tmpVgpr1 + shiftOffset2+ i * gwvw), comment="convert to FP32"))
          elif dataType.isBFloat16():
            module.add(VCvtBF16toFP32(dst=vgpr(tmpVgpr1 + vi + i * gwvw), src=vgpr(tmpVgpr1 + shiftOffset2+ i * gwvw), vgprMask=None, vi=0))
          elif dataType == kernel["ProblemType"]["ComputeDataType"]:
            pass # Same, no need to convert
          else:
            printExit("[Compute fp32] Unrecognized data type %s."%str(dataType))
        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          if dataType == kernel["ProblemType"]["ComputeDataType"]:
            pass # Same, no need to convert
          else:
            printExit("[Compute int32] Unrecognized data type %s."%str(dataType))
        else:
          printExit("Does not support ComputeDataType == %s"%str(kernel["ProblemType"]["ComputeDataType"]))
        if setToOne:
          module.add(VCndMaskB32(
            dst=vgpr(tmpVgpr1 + i * gwvw), \
            src1=vgpr(tmpVgpr1 + i * gwvw), \
            src0=maskConst, \
            src2=sgpr("Address%s"%addressStr, self.states.laneSGPRCount), \
            comment="1. mul 1 if 0"))
      if bps==2:
        module.add(DSStoreB16(dstAddr=dst, src=vgpr(tmpVgpr1 + i * gwvw), ds=ds, comment=comment))
      elif bps==4:
        module.add(DSStoreB32(dstAddr=dst, src=vgpr(tmpVgpr1 + i * gwvw), ds=ds, comment=comment))
      elif bps==8:
        module.add(DSStoreB64(dstAddr=dst, src=vgpr(tmpVgpr1 + i * gwvw * 2, 2), ds=ds, comment=comment))
      else:
        assert 0
    return module

  def getNumOfTempVgprs(self, vectorDataTypes: VectorDataTypes, kernel, gwvw, dim):
    biasDataType   = vectorDataTypes.bias(dim).dataType
    scaleADataType = vectorDataTypes.scaleA.dataType
    scaleBDataType = vectorDataTypes.scaleB.dataType
    scaleAlphaDataType = vectorDataTypes.scaleAlpha(dim).dataType

    # Calculate nums of vgpr for store data
    totalReg  = 0
    regPerVec = gwvw * kernel["ProblemType"]["ComputeDataType"].numRegisters()
    if biasDataType:
      vectorDataTypes.bias(dim).dstVgpr = totalReg
      vectorDataTypes.bias(dim).turn = self.getTurn(kernel, gwvw, dim)[0]
      totalReg = totalReg + (self.getTurn(kernel, gwvw, dim)[0] * regPerVec)
    if scaleAlphaDataType:
      vectorDataTypes.scaleAlpha(dim).dstVgpr = totalReg
      vectorDataTypes.scaleAlpha(dim).turn = self.getTurn(kernel, gwvw, dim)[0]
      totalReg = totalReg + (self.getTurn(kernel, gwvw, dim)[0] * regPerVec)
    if scaleADataType:
      vectorDataTypes.scaleA.dstVgpr = totalReg
      vectorDataTypes.scaleA.turn = self.getTurn(kernel, gwvw, 0)[0]
      totalReg = totalReg + (self.getTurn(kernel, gwvw, 0)[0] * regPerVec)
    if scaleBDataType:
      vectorDataTypes.scaleB.dstVgpr = totalReg
      vectorDataTypes.scaleB.turn = self.getTurn(kernel, gwvw, 1)[0]
      totalReg = totalReg + (self.getTurn(kernel, gwvw, 1)[0] * regPerVec)

    # Check how many additional sgpr is needed for global read
    tmpVgprNum = 0
    offsetVgprStart = totalReg
    # Only vector without stride input can add to dimKey
    dimKey = {}
    if biasDataType:
      vectorDataTypes.bias(dim).offsetVgpr = offsetVgprStart
      tmpVgprNum = tmpVgprNum + 1
    if scaleAlphaDataType:
      if (scaleAlphaDataType, 1) in dimKey:
        vectorDataTypes.scaleAlpha(dim).offsetVgpr = dimKey[(scaleAlphaDataType, dim)]
      else:
        vectorDataTypes.scaleAlpha(dim).offsetVgpr = offsetVgprStart + tmpVgprNum
        tmpVgprNum = tmpVgprNum + 1
    if scaleADataType:
      if (scaleADataType, 0) in dimKey:
        vectorDataTypes.scaleA.offsetVgpr = dimKey[(scaleADataType, dim)]
      else:
        vectorDataTypes.scaleA.offsetVgpr = offsetVgprStart + tmpVgprNum
        tmpVgprNum = tmpVgprNum + 1
    if scaleBDataType:
      if (scaleBDataType, 1) in dimKey:
        vectorDataTypes.scaleB.offsetVgpr = dimKey[(scaleBDataType, dim)]
      else:
        vectorDataTypes.scaleB.offsetVgpr = offsetVgprStart + tmpVgprNum
        tmpVgprNum = tmpVgprNum + 1
    return totalReg + tmpVgprNum

  def readVectorToLDS(self, vectorDataTypes: VectorDataTypes, kernel, gwvw, offsetVgpr, tmpVgpr1Res: ContinuousRegister, dim, tmpSgpr = None):
    assert gwvw == 1
    # Params
    biasDataType         = vectorDataTypes.bias(dim).dataType
    scaleADataType       = vectorDataTypes.scaleA.dataType
    scaleBDataType       = vectorDataTypes.scaleB.dataType
    scaleAlphaDataType   = vectorDataTypes.scaleAlpha(dim).dataType
    biasBpe              = int(self.states.bpr * biasDataType.numRegisters()) if biasDataType else 0
    scaleABpe            = int(self.states.bpr * scaleADataType.numRegisters()) if scaleADataType else 0
    scaleBBpe            = int(self.states.bpr * scaleBDataType.numRegisters()) if scaleBDataType else 0
    scaleAlphaBpe        = int(self.states.bpr * scaleAlphaDataType.numRegisters()) if scaleAlphaDataType else 0
    biasDstVgpr          = vectorDataTypes.bias(dim).dstVgpr
    scaleADstVgpr        = vectorDataTypes.scaleA.dstVgpr
    scaleBDstVgpr        = vectorDataTypes.scaleB.dstVgpr
    scaleAlphaDstVgpr    = vectorDataTypes.scaleAlpha(dim).dstVgpr
    biasOffsetVgpr       = vectorDataTypes.bias(dim).offsetVgpr + tmpVgpr1Res.idx
    scaleAOffsetVgpr     = vectorDataTypes.scaleA.offsetVgpr + tmpVgpr1Res.idx
    scaleBOffsetVgpr     = vectorDataTypes.scaleB.offsetVgpr + tmpVgpr1Res.idx
    scaleAlphaOffsetVgpr = vectorDataTypes.scaleAlpha(dim).offsetVgpr + tmpVgpr1Res.idx

    module = Module("ReadVecToLds")
    module.addComment2("Read vector to LDS")
    # Calculate global offset- macro tile X part
    ## Common codes
    module.addModuleAsFlatItems(self.calculateVectorGlobalOffset(kernel, offsetVgpr, tmpSgpr, dim))
    ## Scale for each component
    offsetIsInit = {}
    if biasDataType:
      # Recalculate bias length
      module.add(SMulI32(dst=sgpr("SrdBias+2"), src0=hex(biasBpe), src1=sgpr("SrdBias+2"), comment="scaled by BPE"))
      if biasOffsetVgpr not in offsetIsInit:
        offsetIsInit[biasOffsetVgpr] = 1
        module.addModuleAsFlatItems(self.calculateVectorGlobalStride(offsetVgpr, biasOffsetVgpr, tmpSgpr, dim, "BiasStride"))
        module.add(VLShiftLeftB32(dst=vgpr(biasOffsetVgpr), \
                                  shiftHex=hex(log2(biasBpe)), \
                                  src=vgpr(biasOffsetVgpr), \
                                  comment="Global bias address scaled by BPE"))
    offsetSequences = []
    if dim == 0:
      offsetSequences.append([scaleAlphaDataType, scaleAlphaOffsetVgpr, scaleAlphaBpe, "scaleAlpha", dim])
      offsetSequences.append([scaleADataType, scaleAOffsetVgpr, scaleABpe, "scaleA", 0])
      offsetSequences.append([scaleBDataType, scaleBOffsetVgpr, scaleBBpe, "scaleB", 1])
    else:
      offsetSequences.append([scaleAlphaDataType, scaleAlphaOffsetVgpr, scaleAlphaBpe, "scaleAlpha", dim])
      offsetSequences.append([scaleBDataType, scaleBOffsetVgpr, scaleBBpe, "scaleB", 1])
      offsetSequences.append([scaleADataType, scaleAOffsetVgpr, scaleABpe, "scaleA", 0])
    for index, offsetSequence in enumerate(offsetSequences):
      if offsetSequence[4] != dim:
        dimAnother = 1 if dim == 0 else 0
        module.addModuleAsFlatItems(self.calculateVectorGlobalOffset(kernel, offsetVgpr, tmpSgpr, dimAnother))
      if offsetSequence[0] and (offsetSequence[1] not in offsetIsInit):
        offsetIsInit[offsetSequence[1]] = 1
        module.add(VLShiftLeftB32(dst=vgpr(offsetSequence[1]), \
                                  shiftHex=hex(log2(offsetSequence[2])), \
                                  src=vgpr(offsetVgpr), \
                                  comment="Global %s address scaled by BPE"%offsetSequence[3]))

    # global load
    globalLoadsModule = Module("Global Loads")
    if biasDataType:
      biasShiftOffset = self.getGlobalShiftOffset(kernel, biasDataType, gwvw)
      globalLoadsModule.addModuleAsFlatItems(self.addVectorGlobalLoad(kernel, "Bias", biasOffsetVgpr, biasShiftOffset, biasDataType, biasBpe, gwvw, tmpVgpr1Res, biasDstVgpr, dim))
    if scaleAlphaDataType:
      scaleAlphaShiftOffset = self.getGlobalShiftOffset(kernel, scaleAlphaDataType, gwvw)
      globalLoadsModule.addModuleAsFlatItems(self.addVectorGlobalLoad(kernel, "ScaleAlphaVec", scaleAlphaOffsetVgpr, scaleAlphaShiftOffset, scaleAlphaDataType, scaleAlphaBpe, gwvw, tmpVgpr1Res, scaleAlphaDstVgpr, dim))
    if scaleADataType:
      scaleAShiftOffset = self.getGlobalShiftOffset(kernel, scaleADataType, gwvw)
      globalLoadsModule.addModuleAsFlatItems(self.addVectorGlobalLoad(kernel, "ScaleA", scaleAOffsetVgpr, scaleAShiftOffset, scaleADataType, scaleABpe, gwvw, tmpVgpr1Res, scaleADstVgpr, 0))
    if scaleBDataType:
      scaleBShiftOffset = self.getGlobalShiftOffset(kernel, scaleBDataType, gwvw)
      globalLoadsModule.addModuleAsFlatItems(self.addVectorGlobalLoad(kernel, "ScaleB", scaleBOffsetVgpr, scaleBShiftOffset, scaleBDataType, scaleBBpe, gwvw, tmpVgpr1Res, scaleBDstVgpr, 1))
    # Count global loads
    vlcnt = 0
    for item in globalLoadsModule.items():
      if isinstance(item, MUBUFReadInstruction):
        vlcnt = vlcnt + 1
    module.add(globalLoadsModule)
    assert vlcnt > 0

    # Local write
    # In local write, all vector shares the same offsetVgpr since the internal data types are all the same.
    module.add(VLShiftLeftB32(dst=vgpr(offsetVgpr), \
                              shiftHex=hex(log2(self.states.bpeCinternal)), \
                              src=vgpr("Serial"), \
                              comment="Local address scaled by BPE"))
    if kernel["LdsOffsetBias"] != 0:
      module.add(VAddU32(dst=vgpr(offsetVgpr), \
                         src0=(kernel["LdsOffsetBias"]), \
                         src1=vgpr(offsetVgpr), \
                         comment="add lds offset"))

    # Get all local stores
    storeModules = Module("Store")
    subGroupOffset = [0]
    if biasDataType:
      vectorDataTypes.bias(dim).ldsOffset = subGroupOffset[0]
      storeModules.add(self.addVectorLocalStore(kernel, "Bias", offsetVgpr, biasShiftOffset, biasDataType, gwvw, tmpVgpr1Res, biasDstVgpr, subGroupOffset, dim, comment="store bias"))
      subGroupOffset[0] += kernel["NumThreads"] * kernel["ProblemType"]["ComputeDataType"].numBytes() * vectorDataTypes.bias(dim).turn
    if scaleAlphaDataType:
      vectorDataTypes.scaleAlpha(dim).ldsOffset = subGroupOffset[0]
      storeModules.add(self.addVectorLocalStore(kernel, "ScaleAlphaVec", offsetVgpr, scaleAlphaShiftOffset, scaleAlphaDataType, gwvw, tmpVgpr1Res, scaleAlphaDstVgpr, subGroupOffset, dim, setToOne=True, comment="store scaleAlpha"))
      subGroupOffset[0] += kernel["NumThreads"] * kernel["ProblemType"]["ComputeDataType"].numBytes() * vectorDataTypes.scaleAlpha(dim).turn
    if scaleADataType:
      vectorDataTypes.scaleA.ldsOffset = subGroupOffset[0]
      storeModules.add(self.addVectorLocalStore(kernel, "ScaleA", offsetVgpr, scaleAShiftOffset, scaleADataType, gwvw, tmpVgpr1Res, scaleADstVgpr, subGroupOffset, 0, setToOne=True, comment="store scaleA"))
      subGroupOffset[0] += kernel["NumThreads"] * kernel["ProblemType"]["ComputeDataType"].numBytes() * vectorDataTypes.scaleA.turn
    if scaleBDataType:
      vectorDataTypes.scaleB.ldsOffset = subGroupOffset[0]
      storeModules.add(self.addVectorLocalStore(kernel, "ScaleB", offsetVgpr, scaleBShiftOffset, scaleBDataType, gwvw, tmpVgpr1Res, scaleBDstVgpr, subGroupOffset, 1, setToOne=True, comment="store scaleB"))
      subGroupOffset[0] += kernel["NumThreads"] * kernel["ProblemType"]["ComputeDataType"].numBytes() * vectorDataTypes.scaleB.turn
    # We move s_barrier before local load. Add barrier here to avoid race condition if lds offset starts from 0
    if kernel["LdsOffsetBias"] == 0:
      module.add(SBarrier(comment="wait for all global loads."))

    # rearrange them and add waitcnt
    for storeModule in storeModules.items():
      isAdded = False
      if isinstance(storeModule, Module):
        for item in storeModule.items():
          if (not isAdded) and isinstance(item, (VCvtInstruction, DSStoreInstruction, VCndMaskB32, VLShiftLeftB32, VAndB32)):
            vlcnt = vlcnt - 1
            module.add(SWaitCnt(vlcnt=(vlcnt), comment="wait for global load"))
            module.add(item)
            isAdded = True
          else:
            module.add(item)
          # restore after ds_store
          if isinstance(item, DSStoreInstruction):
            isAdded = False
      else:
        module.add(storeModule)

    return module

  '''
  Read reduction results from LDS
  In edge case the gwvw will be set to 1.
  '''
  def writeBiasToBlobalLdsRead(self, kernel, offsetVgpr, gwvw, maxKId, outVgpr):
    module = Module("WriteBiasToGlobalLdsRead")
    module.add(VLShiftLeftB32(dst=vgpr(offsetVgpr), \
                              shiftHex=hex(log2(self.states.bpeCinternal)), \
                              src=vgpr(offsetVgpr), \
                              comment="Local bias address scaled by BPE"))

    if kernel["LdsOffsetBias"] != 0:
      module.add(VAddU32(dst=vgpr(offsetVgpr), \
                         src0=(kernel["LdsOffsetBias"]), \
                         src1=vgpr(offsetVgpr), \
                         comment="add bias lds offset"))

    srcAddr = vgpr(offsetVgpr)
    outVgprN = outVgpr
    if maxKId == 1:
      gwvwK = 1
      bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw
      ds  = DSModifiers(offset=0)
      if bps==2:
        module.add(DSLoadB16(dst=vgpr(outVgprN), src=srcAddr, ds=ds, comment="load bias"))
      elif bps==4:
        module.add(DSLoadB32(dst=vgpr(outVgprN), src=srcAddr, ds=ds, comment="load bias"))
      elif bps==8:
        gwvwK = 2
        module.add(DSLoadB64(dst=vgpr(outVgprN, 2), src=srcAddr, ds=ds, comment="load bias"))
      else:
        assert 0
      outVgprN += gwvwK
    else:
      dsOffset = 0
      for _ in range(0, gwvw):
        gwvwK = 1
        idx = 0
        while idx < maxKId:
          gwvwK = 2 if (idx + 1 < maxKId * gwvw) else 1
          bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvwK
          ds  = DSModifiers(offset=dsOffset)
          if bps==2:
            module.add(DSLoadB16(dst=vgpr(outVgprN), src=srcAddr, ds=ds, comment="load bias"))
          elif bps==4:
            module.add(DSLoadB32(dst=vgpr(outVgprN), src=srcAddr, ds=ds, comment="load bias"))
          elif bps==8:
            module.add(DSLoadB64(dst=vgpr(outVgprN, 2), src=srcAddr, ds=ds, comment="load bias"))
          else:
            assert 0
          outVgprN += gwvwK
          idx += gwvwK
          dsOffset += bps
    return module, outVgprN

  '''
  This is a tiny version of globalWriteElements for Bias reduction.
  Every MT saves MTx1 bias data to global.

  When GSU = 1, the address is set to the bias pointer.
  When GSU > 1 in multiple buffer mode, the address is set to the work space pointer.

  Wider DS load is always enabled.
  Wider global store only enables when freeElementMultiple % gwvw == 0 since each thread only stores 1, 2 elements.
  '''
  def writeBiasToGlobal(self, biasDataType, kernel, tP, gwvw, offsetVgpr, tmpSgprRes, tmpVgpr1Res: ContinuousRegister):
    tile01 = tP["tile01Idx"]
    mt     = kernel["MacroTile%u" % tile01]
    maxKId = self.states.lraTileProperties[tile01].maxKId
    assert tmpSgprRes.size >= 1
    assert tmpVgpr1Res.size >= kernel["VectorWidthA"] * kernel["ProblemType"]["ComputeDataType"].numRegisters() * maxKId


    assert gwvw % 2 == 0 or gwvw == 1

    # Params
    biasBpe = int(self.states.bpr * biasDataType.numRegisters())
    module = Module("WriteBiasToGlobal")
    module.addComment2("Write Bias to Global")
    module.add(SBarrier(comment="wait for bias lds store."))

    # Recalculate bias length
    gsuComponent = Component.GSU.find(self)
    module.add(gsuComponent.writeBiasToGlobal(self, kernel, biasDataType, tP, tmpSgprRes, biasBpe))

    # Num records
    module.add(SMulI32(dst=sgpr("SrdBias+2"), src0=hex(biasBpe), src1=sgpr("SrdBias+2"), comment="scaled by BPE"))

    # Local read
    # remaining size % VW
    module.add(vectorStaticMultiply(vgpr(offsetVgpr), vgpr("Serial"), maxKId, tmpSgprRes, \
            "offset = serial * maxKId"))
    module.add(vectorStaticMultiply(vgpr(offsetVgpr), vgpr(offsetVgpr), gwvw, tmpSgprRes, \
            "apply VectorWidth: offset = bnOffset * vw(%u)" % gwvw))

    enableEdge = False
    serialOffsetVgpr  = tmpVgpr1Res.idx
    serialOffsetVgpr2 = tmpVgpr1Res.idx + 1
    # ShiftPtr
    if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"]) and (kernel["ProblemType"]["BiasSrc"] == "A"):
      enableEdge = True
    if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialB"]) and (kernel["ProblemType"]["BiasSrc"] == "B"):
      enableEdge = True
    if kernel["EdgeType"] == "ShiftPtr" and enableEdge:
      jumpLabel    = Label(self.labels.getNameInc("ShiftPtrSkip"), comment="Skip shift ptr")
      jumpLabelEnd = Label(self.labels.getNameInc("ShiftPtrEnd"), comment="Skip shift ptr end")
      assert tmpSgprRes.size >= 5 # For ShiftPtr
      assert tP["glvw"] % gwvw == 0 # Make sure the magic trick works (serial = serial * gwvw)
      tmpSgpr = tmpSgprRes.idx
      tmpSgprShift = ContinuousRegister(idx=tmpSgprRes.idx+2, size=3)
      margin = tP["glvw"] if tP["rtv"] else 1
      module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tP["wg"]), src1=kernel[tP["mt"]], comment="WorkGroup[01] * MT"))
      module.add(SSubU32(dst=sgpr(tmpSgpr), src0=self.sizeRef(tP["idx"]), src1=sgpr(tmpSgpr), \
                comment="edge = Size%s - WG*MT"%(tP["tileChar"])))
      module.add(SCmpGeU32(src0=sgpr(tmpSgpr), src1=kernel[tP["mt"]], comment="skip shift ptr if edge >= MT"))
      module.add(SCBranchSCC1(labelName=jumpLabel.getLabelName(), comment="" ))
      module.add(scalarStaticRemainder(tmpSgpr+1, tmpSgpr+1, tmpSgpr, tP["glvw"], tmpSgprShift, comment="remainder = edge %% glvw(%d)"%tP["glvw"]))
      module.add(SSubU32(dst=sgpr(tmpSgpr+1), src0=hex(tP["glvw"]), src1=sgpr(tmpSgpr+1), comment="shift = glvw(%d) - remainder"%tP["glvw"]))
      #module.add(SCmpKEQU32(src=sgpr(tmpSgpr+1), simm16=hex(tP["glvw"]), comment="if shift == glvw(%d)?"%tP["glvw"]))
      module.add(self.getSCMPKInstruction("EQU32", tmpSgpr+1, tP["glvw"], comment="if shift == glvw(%d)?"%tP["glvw"]))
      module.add(SCMovB32(dst=sgpr(tmpSgpr+1), src=0, comment="shift = glvw(%d) ? 0 : NOP"%(tP["glvw"])))
      module.add(SSubU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=margin, comment="edge -= margin(%u)"%(margin)))
      module.add(SCMovB32(dst=sgpr(tmpSgpr), src=0, comment="edge = edge < 0 ? 0 : edge")) # Saturation
      if gwvw != 1:
        '''
        Edge case, the gwvw is set to 1 instead, do boundary check on every element.

        if gwvw > 1:
            for g in rangw(gwvw):
              if(out_of_bound(g))
                offset_at_g += shit_offset
              ds_read vpgr_at_g, offset_at_g
        else:
          if(out_of_bound(g))
            offset_at_g += shit_offset
          ds_read vpgr_at_g, offset_at_g
        '''
        module.add(vectorStaticMultiply(vgpr(serialOffsetVgpr), vgpr("Serial"), gwvw, tmpSgprShift, comment="serial = serial * gwvw"))
        for g in range(gwvw):
          if g != 0:
            module.add(VAddU32(dst=vgpr(offsetVgpr+g), src0=vgpr(offsetVgpr+(g-1)), src1=maxKId, comment="new offset = offset + maxKId"))
        module.add(VMulLOU32(dst=vgpr(serialOffsetVgpr2), src0=hex(maxKId), src1=sgpr(tmpSgpr+1), comment="ds_offset = K * offset"))
        for g in range(gwvw):
          module.add(VCmpXGeU32(dst=EXEC(), src0=vgpr(serialOffsetVgpr), src1=sgpr(tmpSgpr), comment="needs shift if serial > edge"))
          module.add(VAddU32(dst=vgpr(offsetVgpr+g), src0=vgpr(offsetVgpr+g), src1=vgpr(serialOffsetVgpr2), comment="real offset = offset + ds_offset"))
          if g < gwvw - 1:
              module.add(VAddU32(dst=vgpr(serialOffsetVgpr), src0=1, src1=vgpr(serialOffsetVgpr), comment="inc += 1"))
      else:
        module.add(VCmpXGeU32(dst=EXEC(), src0=vgpr("Serial"), src1=sgpr(tmpSgpr), comment="needs shift if serial > edge"))
        module.add(VMulLOU32(dst=vgpr(serialOffsetVgpr), src0=hex(maxKId), src1=sgpr(tmpSgpr+1), comment="ds_offset = K * offset"))
        module.add(VAddU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=vgpr(serialOffsetVgpr), comment="real offset = offset + ds_offset"))
      module.add(SMovB64(dst=EXEC(), src=-1, comment="reset mask"))
      outVgprNext = tmpVgpr1Res.idx
      # Shiftptr
      for g in range(gwvw):
        wb2blr, outVgprNext = self.writeBiasToBlobalLdsRead(kernel, offsetVgpr + g, 1, maxKId, outVgprNext)
        module.add(wb2blr)
      module.add(SBranch(labelName=jumpLabelEnd.getLabelName(), comment=""))
      module.add(jumpLabel)
      # Shiftptr case, but no need to shift
      wb2blrGwvw, _ = self.writeBiasToBlobalLdsRead(kernel, offsetVgpr, gwvw, maxKId, tmpVgpr1Res.idx)
      module.add(wb2blrGwvw)
      module.add(jumpLabelEnd)
    else:
      # Non-shiftptr case
      wb2blrGwvw, _ = self.writeBiasToBlobalLdsRead(kernel, offsetVgpr, gwvw, maxKId, tmpVgpr1Res.idx)
      module.add(wb2blrGwvw)


    module.add(SWaitCnt(dscnt=0, comment="wait for bias lds load"))
    # Sum K (if needed)
    '''
    The vgprs are rearranged in this step.
    For example, gwvw = 2, k = 2, we have [v6, v7] [v8, v9]
    v6 = v6 + v7
    v7 = v8 + v9
    '''
    tmpVgpr1 = tmpVgpr1Res.idx
    tmpVgprN = tmpVgpr1 + 1
    if maxKId != 1:
      for gidx in range(0, gwvw):
        tmpVgprAccum = tmpVgpr1 + gidx
        if gidx != 0:
          module.add(VMovB32(dst=vgpr(tmpVgprAccum), src=vgpr(tmpVgprN), comment="Copy address"))
          tmpVgprN += 1
        for idx in range(1, maxKId):
          if kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(VAddF32(dst=vgpr(tmpVgprAccum), src0=vgpr(tmpVgprN), src1=vgpr(tmpVgprAccum), comment="Sum K"))
          else:
            assert 0
          tmpVgprN += 1
    # Convert
    freeElementMultiple = kernel["AssertFree%dElementMultiple"%tile01]
    enablePack = True if (freeElementMultiple % gwvw == 0) else False
    tmpVgprN = tmpVgpr1
    if biasDataType != kernel["ProblemType"]["ComputeDataType"]:
      bf16CVTVgprStruct = None
      bf16CVTVgpr       = None
      if biasDataType.isBFloat16():
        bf16CVTVgpr = self.vgprPool.checkOut(4)
        bf16CVTVgprStruct = self.BF16CVTVgprStruct(vgprBf16Temp=bf16CVTVgpr, vgprBf16Mask=(bf16CVTVgpr+1), \
                                           vgprFp32Nan=(bf16CVTVgpr+2), vgprBf16Inc=(bf16CVTVgpr+3))
        module.add(VMovB32(vgpr(bf16CVTVgprStruct.vgprBf16Mask), "0xffff0000", comment="mask for pack two bfloat16 element to 32bit" ))
        module.add(VMovB32(vgpr(bf16CVTVgprStruct.vgprFp32Nan), "0x7fff0000", comment="fp32 Nan" ))
        module.add(VMovB32(vgpr(bf16CVTVgprStruct.vgprBf16Inc), "0x7fff", comment="rounding bias for bfloat16" ))
      for vi in range(gwvw):
        # Does not support hi/lo yet
        if kernel["ProblemType"]["ComputeDataType"].isSingle():
          if biasDataType.isHalf():
            module.add(VCvtF32toF16(dst=vgpr(tmpVgprN), src=vgpr(tmpVgprN), comment="convert to FP16"))
            if vi % 2 == 1 and enablePack:
              module.add(VPackF16toB32(dst=vgpr(tmpVgprN - 1), src0=vgpr(tmpVgprN - 1), src1=vgpr(tmpVgprN), \
                         comment="Pack with neighbor"))
          elif biasDataType.isBFloat16():
            module.add(VCmpUF32(dst=sgpr(tmpSgprRes.idx,2), src0=vgpr(tmpVgprN), src1=vgpr(tmpVgprN), comment="check Nan"))
            module.add(VBfeU32(dst=vgpr(bf16CVTVgprStruct.vgprBf16Temp), src0=vgpr(tmpVgprN), src1=16, src2=1, \
                         comment="Non-Nan case: store lsb of bf16" ))
            module.add(VAdd3U32(dst=vgpr(bf16CVTVgprStruct.vgprBf16Temp), src0=vgpr(tmpVgprN), src1=vgpr(bf16CVTVgprStruct.vgprBf16Temp), \
                         src2=vgpr(bf16CVTVgprStruct.vgprBf16Inc), comment="Non-Nan case: add lsb and the increment for rounding" ))
            module.add(VCndMaskB32(dst=vgpr(tmpVgprN), src0=vgpr(bf16CVTVgprStruct.vgprBf16Temp), \
                         src1=vgpr(bf16CVTVgprStruct.vgprFp32Nan), src2=sgpr(tmpSgprRes.idx,2)))
            module.add(VLShiftRightB32(dst=vgpr(tmpVgprN), shiftHex=16, src=vgpr(tmpVgprN), comment="convert to bf16"))
            if vi % 2 == 1 and enablePack:
              module.add(VPackF16toB32(dst=vgpr(tmpVgprN - 1), src0=vgpr(tmpVgprN - 1), src1=vgpr(tmpVgprN), \
                         comment="Pack with neighbor"))
          elif biasDataType == kernel["ProblemType"]["ComputeDataType"]:
            pass # Same, no need to convert
          else:
            printExit("Unrecognized bias type %s."%str(biasDataType))
          tmpVgprN += 1
        else:
          printExit("Does not support ComputeDataType != float")
      if bf16CVTVgpr != None:
        self.vgprPool.checkIn(bf16CVTVgpr)
    # Global write
    # Calculate global offset- macro tile 0 part
    tmpSgpr = tmpSgprRes.idx
    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=mt, src1=sgpr("WorkGroup%u" % tile01), comment="wgp * MT"))
    module.add(vectorStaticMultiply(vgpr(offsetVgpr), vgpr("Serial"), gwvw, tmpSgprRes, \
            "apply VectorWidth: offset = serial * vw(%u)" % gwvw))
    module.add(VAddU32(dst=vgpr(offsetVgpr), src0=sgpr(tmpSgpr), src1=vgpr(offsetVgpr), comment="coord = wgp * MT + thread offset"))
    module.add(VLShiftLeftB32(dst=vgpr(offsetVgpr), \
                              shiftHex=hex(log2(biasBpe)), \
                              src=vgpr(offsetVgpr), \
                              comment="Global bias address scaled by BPE"))
    with self.allocTmpSgpr(1, 1) as tmpSgprRes:
      module.add(SMovB32(dst=sgpr(tmpSgprRes.idx), src=hex(mt//gwvw), comment="%d=%d//%d"%(mt//gwvw, mt, gwvw)))
      module.add(VCmpXLtU32(dst=EXEC(), src0=vgpr("Serial"), src1=sgpr(tmpSgprRes.idx), comment="if serial < MacroTile%d/gwvw"%tile01))
    addr0 = vgpr(offsetVgpr)
    addr1 = sgpr("SrdBias", 4)
    dataType = biasDataType
    useBuffer = kernel["BufferLoad"]
    tmpVgprN = tmpVgpr1
    if enablePack: # no partial
      bps = biasDataType.numBytes() * gwvw
      rpe = biasDataType.numBytes() / self.states.bpr
      rpv = rpe * gwvw
      if dataType.isHalf() or dataType.isBFloat16():
        module.add(self.chooseGlobalWrite(useBuffer, bps, tmpVgprN, rpv, \
                          addr0, addr1, offset=0, hi16=0, comment="global store bias"))
      elif dataType.isInt32() or dataType.isSingle():
        module.add(self.chooseGlobalWrite(useBuffer, bps, tmpVgprN, rpv, \
                          addr0, addr1, offset=0, comment="global store bias"))
      elif dataType.isDouble() or dataType.isSingleComplex() :
        module.add(self.chooseGlobalWrite(useBuffer, bps, tmpVgprN, rpv, \
                          addr0, addr1, offset=0, comment="global store bias"))
    else: # edge
      tmpVgprNStep = max(1, biasDataType.numRegisters())
      globalOffset = 0
      for gidx in range(0, gwvw):
        bps = biasDataType.numBytes()
        rpe = biasDataType.numBytes() / self.states.bpr
        rpv = rpe
        if dataType.isHalf() or dataType.isBFloat16():
          module.add(self.chooseGlobalWrite(useBuffer, bps, tmpVgprN, rpv, \
                            addr0, addr1, offset=globalOffset, hi16=0, comment="global store bias"))
        elif dataType.isInt32() or dataType.isSingle():
          module.add(self.chooseGlobalWrite(useBuffer, bps, tmpVgprN, rpv, \
                            addr0, addr1, offset=globalOffset, comment="global store bias"))
        elif dataType.isDouble() or dataType.isSingleComplex() :
          module.add(self.chooseGlobalWrite(useBuffer, bps, tmpVgprN, rpv, \
                            addr0, addr1, offset=globalOffset, comment="global store bias"))
        tmpVgprN += tmpVgprNStep
        globalOffset += biasBpe
    module.add(SMovB64(dst=EXEC(), src=-1, comment="Reset exec mask"))
    return module

  ########################################
  # Amax related
  ########################################
  def amax_define_load_res(self) -> Module:
    module = Module("AmaxD Set and Load")
    module.addComment0("AmaxD Set and Load")

    self.amaxVgprIdxVec = self.defineMultiVgprIndex(self.amaxVgprNames, self.amaxVgprSizes, align=1)
    for i in range(0, len(self.amaxVgprNames)):
      name = self.amaxVgprNames[i]
      idx = self.amaxVgprIdxVec[i]
      module.add(RegSet("v", "vgpr"+name, idx))

    module.addSpaceLine()
    module.add(self.defineSgpr("Src", 4, 4))
    module.add(self.defineSgpr("Dst", 4, 4))
    module.add(self.defineSgpr("Offset", 1))
    module.add(self.defineSgpr("Tmp", 6, 2))
    module.add(self.defineSgpr("NumGroup", 1))
    module.add(self.defineSgpr("WGIdx", 1))
    module.addSpaceLine()

    # defineMulti (ensure they are checkout together) for SGPRs that are used to load args
    self.amaxSgprIdxVec = self.defineMultiSgprIndex(self.amaxSgprArgNames, self.amaxSgprArgSizes, align=4)
    for name in self.amaxSgprArgNames:
      module.add(RegSet("s", "sgpr"+name, self.sgprs[name]))
    module.addSpaceLine()

    # TODO- why we don't directly update the offset in the last argLoader ?
    argOffset = self.argLoader.getOffset()
    argOffset += (self.states.numStoreSgprToLoad + self.states.numStoreSgprToLoad2) * 4
    module.add(self.argLoader.loadKernArg("AddrAmaxOut", "KernArgAddress", sgprOffset=hex(argOffset), dword=4))
    argOffset += 16 # advance dwordx4
    module.add(self.argLoader.loadKernArg("AddressSy", "KernArgAddress", sgprOffset=hex(argOffset), dword=2))
    module.add(SMulI32(sgpr("NumGroup"), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), "get total num_wgs"))
    module.add(SMulI32(sgpr("WGIdx"), sgpr("WorkGroup1"), sgpr("NumWorkGroups0"), "wgId = wg1 * numWG0"))
    module.add(SAddI32(sgpr("WGIdx"), sgpr("WGIdx"), sgpr("WorkGroup0"), "wgId += wg0"))
    module.addSpaceLine()
    module.add(SWaitCnt(kmcnt=0))
    module.addSpaceLine()

    return module

  def amax_intra_wave_reduction(self, kernel, postfix) -> Module:
    wave_size = kernel["WavefrontSize"]
    label = Label(f"permute_{postfix}", f"permute_{postfix}")

    mod = Module("intra_wave_reduction")
    mod.addComment0("intra_wave_reduction")

    mod.add(SMovB32(sgpr("Tmp"), 1))
    mod.add(label)
    mod.addSpaceLine()
    mod.add(VAddU32(vgpr("Tmp"), sgpr("Tmp"), vgpr("Serial")))
    mod.add(VAndB32(vgpr("Tmp"), wave_size-1, vgpr("Tmp")))
    mod.add(VLShiftLeftB32(vgpr("Tmp"), 0x2, vgpr("Tmp")))
    mod.addSpaceLine()
    mod.add(DSBPermuteB32(vgpr("AmaxOutB"), vgpr("Tmp"), vgpr("AmaxOut")))
    mod.add(SWaitCnt(dscnt=0))
    mod.addSpaceLine()
    # TODO- F16
    mod.add(VMaxF32(vgpr("AmaxOut"), vgpr("AmaxOut"), vgpr("AmaxOutB")))
    mod.add(SLShiftLeftB32(sgpr("Tmp"), 1, sgpr("Tmp")))
    mod.add(SCmpLtU32(sgpr("Tmp"), wave_size))
    mod.add(SCBranchSCC1(label.getLabelName()))
    mod.addSpaceLine()
    return mod

  def amax_inter_wave_reduction(self, kernel) -> Module:
    wave_size = kernel["WavefrontSize"]
    numWorkItems = kernel["NumThreads"]
    amaxOutType = kernel["ProblemType"]["DataTypeAmaxD"]
    amax_lds_start = kernel["LdsBytesNoAmax"]

    label_wave_inter = Label("wave_inter", 'wave_inter')
    label_wave_upper = Label("wave_upper", 'wave_upper')
    label_wave_lower = Label("wave_lower", 'wave_lower')
    label_wave_empty = Label("wave_empty", 'wave_empty')
    label_wave_end   = Label("wave_end",   'wave_end')

    mod = Module("inter_wave_reduction")
    mod.addComment0("inter_wave_reduction")

    mod.add(VLShiftRightB32(vgpr("Widx"), int(log2(wave_size)), vgpr("Serial")))
    mod.add(SMovB32(sgpr("Offset"), numWorkItems // wave_size))
    mod.add(label_wave_inter)
    mod.add(SLShiftRightB32(sgpr("Offset"), 1, sgpr("Offset")))
    mod.add(SCmpEQU32(sgpr("Offset"), 0))
    mod.add(SCBranchSCC1(label_wave_end.getLabelName()))
    mod.add(SLShiftLeftB32(sgpr("Tmp"), 1, sgpr("Offset")))
    mod.add(VCmpLtU32(sgpr("Tmp+2",2), vgpr("Widx"), sgpr("Tmp")))
    mod.add(VCmpGEU32(sgpr("Tmp+4",2), vgpr("Widx"), sgpr("Offset")))
    mod.add(SAndB64(VCC(), sgpr("Tmp+2",2), sgpr("Tmp+4",2)))
    mod.add(SCBranchVCCNZ(label_wave_upper.getLabelName()))
    mod.add(VCmpLtU32(VCC(), vgpr("Widx"), sgpr("Offset")))
    mod.add(SCBranchVCCNZ(label_wave_lower.getLabelName()))
    mod.add(SBranch(label_wave_empty.getLabelName()))

    mod.add(label_wave_upper)
    mod.add(VSubU32(vgpr("Tmp"), vgpr("Widx"), sgpr("Offset")))
    mod.add(VLShiftLeftB32(vgpr("Tmp"), int(log2(amaxOutType.numBytes())), vgpr("Tmp")))

    # TODO- select inst
    ds = DSModifiers(offset=amax_lds_start)
    mod.add(DSStoreB32(vgpr("Tmp"), vgpr("AmaxOut"), ds))
    mod.add(SWaitCnt(dscnt=0))
    mod.add(SBarrier())
    mod.add(SBranch(label_wave_inter.getLabelName()))
    mod.add(label_wave_lower)
    mod.add(SBarrier())
    mod.add(VLShiftLeftB32(vgpr("Tmp"), int(log2(amaxOutType.numBytes())), vgpr("Widx")))

    # TODO- select inst
    mod.add(DSLoadB32(vgpr("AmaxOutB"), vgpr("Tmp"), ds))
    mod.add(SWaitCnt(dscnt=0))
    # TODO- F16
    mod.add(VMaxF32(vgpr("AmaxOut"), vgpr("AmaxOut"), vgpr("AmaxOutB")))
    mod.add(SBranch(label_wave_inter.getLabelName()))
    mod.add(label_wave_empty)
    mod.add(SBarrier())
    mod.add(SBranch(label_wave_inter.getLabelName()))
    mod.add(label_wave_end)
    mod.addSpaceLine()
    return mod

  def amax_broadcast(self, kernel) -> Module:
    amax_lds_start = kernel["LdsBytesNoAmax"]

    label_lower = Label("broadcast_lower", f'broadcast_lower')
    label_end = Label("broadcast_end", f'broadcast_end')

    mod = Module("broadcast")
    mod.addComment0("broadcast")
    mod.add(VCmpEQU32(VCC(), vgpr("Widx"), 0))
    mod.add(SCBranchVCCZ(label_lower.getLabelName()))

    # TODO- select inst
    ds = DSModifiers(offset=amax_lds_start)
    mod.add(DSStoreB32(vgpr("Widx"), vgpr("AmaxOut"), ds))
    mod.add(SWaitCnt(dscnt=0))
    mod.add(SBarrier())
    mod.add(SBranch(label_end.getLabelName()))
    mod.add(label_lower)
    mod.add(SBarrier())
    mod.add(VMovB32(vgpr("Tmp"), 0))

    # TODO- select inst
    mod.add(DSLoadB32(vgpr("AmaxOut"), vgpr("Tmp"), ds))
    mod.add(SWaitCnt(dscnt=0))
    mod.add(label_end)
    mod.addSpaceLine()
    mod.addSpaceLine()
    return mod

  def amax_output_result(self, kernel) -> Module:
    wave_size = kernel["WavefrontSize"]
    amaxInType = kernel["ProblemType"]["ComputeDataType"]
    amaxOutType = kernel["ProblemType"]["DataTypeAmaxD"]

    mod = Module("output_result")
    mod.addComment0("output_result")

    label_end = Label("end", 'end')
    label_final_loop = Label("final_loop", 'final_loop')
    label_final_output = Label("final_output", 'final_output')
    mod.addSpaceLine()

    mod.add(VReadfirstlaneB32(sgpr("Tmp"), vgpr("Serial")))
    mod.add(SCmpEQU32(sgpr("Tmp"), 0))
    mod.add(SCBranchSCC0(label_end.getLabelName()))
    mod.addSpaceLine()

    # if self.arch.find("gfx94") != -1:
    mod.addSpaceLine()
    mod.add(SCmpEQU32(sgpr("NumGroup"), 1))
    mod.add(SCBranchSCC1(label_final_output.getLabelName()))

    mod.add(SLShiftLeftB32(sgpr("Tmp"), int(log2(amaxInType.numBytes())), sgpr("NumGroup")))
    mod.add(SMovB32(sgpr("Dst+0"), sgpr("AddressWk+0")))
    mod.add(SMovB32(sgpr("Dst+1"), sgpr("AddressWk+1")))
    mod.add(SMovB32(sgpr("Dst+2"), sgpr("Tmp")))
    mod.add(SMovB32(sgpr("Dst+3"), "Srd127_96"))

    mod.add(SLShiftLeftB32(sgpr("Offset"), int(log2(amaxInType.numBytes())), sgpr("WGIdx")))
    mod.add(VMovB32(vgpr("Offset"), 0))

    # TODO- select inst
    mod.add(BufferStoreB32(vgpr("AmaxOut"), vgpr("Offset"), sgpr("Dst",4), sgpr("Offset"), MUBUFModifiers(offen=True, glc=True, slc=True)))
    mod.add(SWaitCnt(vlcnt=0))
    mod.addSpaceLine()

    mod.add(SSubI32(sgpr("Tmp"), sgpr("NumGroup"), 1))
    mod.add(SAtomicDec(sgpr("Tmp"), sgpr("AddressSy",2), SMEMModifiers(glc=True)))
    mod.add(SWaitCnt(vlcnt=0, kmcnt=0))
    mod.add(SCmpEQU32(sgpr("Tmp"), 1))
    mod.add(SCBranchSCC0(label_end.getLabelName()))
    mod.addSpaceLine()

    mod.add(SLShiftLeftB32(sgpr("Tmp"), int(log2(amaxInType.numBytes())), sgpr("NumGroup")))
    mod.add(SMovB32(sgpr("Src+0"), sgpr("AddressWk+0")))
    mod.add(SMovB32(sgpr("Src+1"), sgpr("AddressWk+1")))
    mod.add(SMovB32(sgpr("Src+2"), sgpr("Tmp")))
    mod.add(SMovB32(sgpr("Src+3"), "Srd127_96"))
    mod.addSpaceLine()

    mod.add(VLShiftLeftB32(vgpr("Offset"), int(log2(amaxOutType.numBytes())), vgpr("Serial")))
    mod.addSpaceLine()

    mod.add(VMovB32(vgpr("AmaxOut"), "0"))
    mod.addSpaceLine()
    mod.add(label_final_loop)

    # TODO- select inst
    mod.add(BufferLoadB32(vgpr(f"Value"), vgpr("Offset"), sgpr("Src",4), 0, MUBUFModifiers(offen=True, glc=True, slc=True)))
    mod.add(SWaitCnt(vlcnt=0))
    mod.addSpaceLine()

    # TODO- F16?
    mod.add(VMaxF32(vgpr("AmaxOut"), vgpr("AmaxOut"), vgpr("Value", isAbs=True)))
    mod.addSpaceLine()

    mod.add(SMovB32(sgpr("Tmp"), wave_size * amaxInType.numBytes()))
    mod.add(VAddU32(vgpr("Offset"), vgpr("Offset"), sgpr("Tmp")))
    mod.addSpaceLine()

    mod.add(SSubI32(sgpr("NumGroup"), sgpr("NumGroup"), wave_size))
    mod.add(SCmpGtI32(sgpr("NumGroup"), 0))
    mod.add(SCBranchSCC1(label_final_loop.getLabelName()))
    mod.addSpaceLine()

    mod.add(self.amax_intra_wave_reduction(kernel, "final"))
    mod.addSpaceLine()
    mod.add(label_final_output)

    mod.add(SMovB32(sgpr("Dst+0"), sgpr("AddrAmaxOut+0")))
    mod.add(SMovB32(sgpr("Dst+1"), sgpr("AddrAmaxOut+1")))
    mod.add(SMovB32(sgpr("Dst+2"), amaxOutType.numBytes()))
    mod.add(SMovB32(sgpr("Dst+3"), "Srd127_96"))
    mod.addSpaceLine()

    mod.add(VMovB32(vgpr("Offset"), 0))

    # TODO- select inst
    mod.add(BufferStoreB32(vgpr("AmaxOut"), vgpr("Offset"), sgpr("Dst",4), 0, MUBUFModifiers(offen=True)))
    mod.addSpaceLine()
    mod.add(label_end)
    mod.addSpaceLine()

    return mod

  def insertAmaxD(self, kernel):
    module = Module("AmaxD Output")
    module.addComment0("AmaxD Output")

    self.amaxVgprNames = ["Widx", "Offset", "Tmp", "Value"]
    self.amaxVgprSizes = [1, 1, 1, 1]
    self.amaxSgprArgNames = ["AddrAmaxOut", "AddressWk", "AddressSy"]
    self.amaxSgprArgSizes = [2, 2, 2]

    module.addSpaceLine()
    module.add(SBarrier())
    module.add(self.amax_define_load_res())
    module.add(self.amax_intra_wave_reduction(kernel, "middle"))
    module.add(self.amax_inter_wave_reduction(kernel))
    module.add(self.amax_broadcast(kernel))
    module.add(self.amax_output_result(kernel))

    for i in self.amaxVgprIdxVec:
        self.vgprPool.checkIn(i)
    for i in self.amaxSgprIdxVec:
        self.sgprPool.checkIn(i)

    return module

  ########################################
  # Activation related
  ########################################
  def initActivationLoop(self, kernel, beta, edge):
    # Create a suffix and check if the string exists
    activationLabelSuffix = self.labels.getNameInc( \
      "%s%s"%("_Beta" if beta else "", "_Edge" if edge else ""))
    activationCDataType = kernel["ProblemType"]["ActivationComputeDataType"]
    activationType = kernel["ProblemType"]["ActivationType"]
    activationEndLabel = Label("Activation_End%s"%activationLabelSuffix, "")
    activationLabelModules = []
    activationEnumStrList = []
    if kernel["ActivationFuncCall"]:
      activationLabelModules.append("")
      activationEnumStrList.append("none")
    elif (((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel' or kernel["StreamK"] > 0) and kernel["ActivationFused"]) and \
      (activationType != 'none'):
      if activationType in ['all', 'hipblaslt_all']:
        exportType = ActivationType.Export.GRADONLY if kernel["ProblemType"]["Gradient"] else ActivationType.Export.NORMAL
        supportedBy = ActivationType.SupportedBy.ALL if activationType == 'all' else ActivationType.SupportedBy.HIPBLASLT
        activationEnumStrList = ActivationType.getEnumStrList(activationCDataType, supportedBy, exportType=exportType)
        for _, enumStr in enumerate(activationEnumStrList):
          activationLabelModule = Label("Activation_%s%s"% (enumStr.capitalize(), activationLabelSuffix), "")
          activationLabelModules.append(activationLabelModule)
      else:
        activationEnumStrList.append(str(kernel["ProblemType"]["ActivationType"]).lower())
    else:
      activationLabelModules.append("")
      activationEnumStrList.append("none")
    return activationEndLabel, activationLabelModules, activationEnumStrList

  def insertActFunctionCallAddrCalc(self, sgprOffset, gwvw, \
    toActModuleList, activationEnumStrList, activationLabelList, \
    betaIdx = -1, edgeIdx = -1):
    activationLabelModules = activationLabelList[gwvw]
    module = Module(getActFuncBranchModuleName())
    setAddrEndLabel = Label(self.labels.getNameInc("ActivationSetPCAddrEnd"), "")
    toActModules = deepcopy(toActModuleList[gwvw])
    for index, toActModule in enumerate(toActModules):
      if betaIdx >= 0 and edgeIdx >= 0:
        toActModule.label = self.labels.getNameInc(toActModule.label + "_beta_%u_edge_%u"%(betaIdx, edgeIdx))
      if index != 0:
        enumIndex = ActivationType.getEnumIndex(activationEnumStrList[index])
        #module.add(SCmpKEQU32(sgpr("ActivationType"), enumIndex, "activationType == %u"%enumIndex))
        module.add(self.getSCMPKInstruction("EQU32", "ActivationType", enumIndex, comment="activationType == %u"%enumIndex))
        module.add(SCBranchSCC1(toActModule.getLabelName(), "Branch if true"))
    for index, activationLabelModule in enumerate(activationLabelModules):
      toActModule = toActModules[index]
      module.add(toActModule)
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.addModuleAsFlatItems(SGetPositivePCOffset(sgprOffset, activationLabelModule, tmpSgprInfo))
      module.add(SBranch(setAddrEndLabel.getLabelName()))
    module.add(setAddrEndLabel)
    return module

  def insertActivationAfterPacked(self, kernel, activationTypeStr):
    result = False
    if kernel["ProblemType"]["UseScaleCD"] and ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["StreamK"] > 0):
      return result
    elif ((kernel["ProblemType"]["ActivationType"] != 'none') and \
      ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel' or kernel["StreamK"] > 0) and kernel["ActivationFused"]):
      if kernel["ActivationFuncCall"]:
        return (kernel["ProblemType"]["ActivationComputeDataType"] == kernel["ProblemType"]["DestDataType"])
      elif kernel["ProblemType"]["DestDataType"].isBFloat16() and (activationTypeStr == 'abs'):
        result = True
      elif kernel["ProblemType"]["DestDataType"].isHalf() and \
        ((activationTypeStr == 'abs') or (activationTypeStr == 'relu')):
        result = True
      elif kernel["ProblemType"]["ActivationComputeDataType"] == kernel["ProblemType"]["DestDataType"]:
        result = True
    return result

  def getActivationDestDataType(self, kernel, activation, activationTypeStr: str, gwvw, \
  elementSumIdxIn, elementSumIdxOut, tmpVgpr, tmpSgpr):
    module = Module("ActivationAfterPack")
    for vi in range(0, gwvw):
      sumIdxVIn  = elementSumIdxIn + vi
      sumIdxVOut = elementSumIdxOut + vi
      if kernel["ProblemType"]["DestDataType"].isHalf() or \
          kernel["ProblemType"]["DestDataType"].isBFloat16():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # Generate single f16 code if edge is detected.
          if ((vi + 1) == gwvw) and ((gwvw % 2) == 1):
            activation.setUsePK(False)
          # Original packed route
          elif vi%2 == 1:
            assert (gwvw % 2 == 0)
          else:
            continue
          vgprIn  = elementSumIdxIn + vi//2
          vgprOut = elementSumIdxOut + vi//2

        else:
          if (sumIdxVIn % 2 != 0):
            continue
          vgprIn  = sumIdxVIn // 2
          vgprOut = sumIdxVOut // 2
      elif kernel["ProblemType"]["DestDataType"].isSingle():
        vgprIn  = sumIdxVIn
        vgprOut = sumIdxVOut
      elif kernel["ProblemType"]["DestDataType"].isDouble():
        vgprIn  = sumIdxVIn * 2
        vgprOut = sumIdxVOut * 2
      elif kernel["ProblemType"]["DestDataType"].isInt32():
        vgprIn  = sumIdxVIn
        vgprOut = sumIdxVOut
      else:
        raise RuntimeError("Unsupported data type %s for activation vgpr index."%str(self.states.kernel["ProblemType"]["DestDataType"]))
      # Here we still use DestDataType cause the data is ready to be written to global
      actModule = activation.getModule(self.states.kernel["ProblemType"]["DestDataType"], activationTypeStr, vgprIn, vgprOut)
      module.add(activation.assignGpr(actModule, tmpVgpr, tmpSgpr))
      activation.setUsePK(True)
    return module

  def getActivationActivationComputeType(self, kernel, activation, activationTypeStr: str, gwvw, \
    elementSumIdxIn, elementSumIdxOut, tmpVgpr, tmpSgpr, satInt8=False, enableValuCPrefix=False):
    module = Module("ActivationBeforePack")
    if satInt8:
      activation.setSaturationForInt8(True)
    if enableValuCPrefix:
      # The register is from ValuC allocation instead of allocated by itself
      activation.setVgprPrefixFormat("ValuC+%u")
    for vi in range(0, gwvw):
      vgprIn  = elementSumIdxIn + vi
      vgprOut = elementSumIdxOut + vi
      actModule = activation.getModule(kernel["ProblemType"]["ActivationComputeDataType"], activationTypeStr, vgprIn, vgprOut)
      module.add(activation.assignGpr(actModule, tmpVgpr, tmpSgpr))
    activation.setSaturationForInt8(False)
    activation.setVgprPrefixFormat("")
    return module

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel, addLabel=True):
    imod = Module()
    loopComponent = Component.PersistentLoop.find(self)
    imod.add(loopComponent.closePersistentLoop(self, kernel))
    if addLabel:
      imod.add(Label("KernelEnd", ""))

      # TODO- refine this part, put outside of this function
      if kernel["ProblemType"]["OutputAmaxD"]:
        imod.add(self.insertAmaxD(kernel))

    imod.add(SEndpgm(comment="Kernel End"))
    return imod

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  def getWaitcntCodeForDirectToVgpr(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, u, isNLL=False, beforeBarrier=False, NLLlast=False, oddLast=False):
    module = Module("DTV wait")
    # generate wait for DTV
    # TODO: add precise wait count logic for DTVA + DTVB
    if kernel["DirectToVgprA"] and kernel["DirectToVgprB"]:
      needToWait = 0
      waitComment = "global read wait for DirectToVgpr"
      module.add(self.getWaitcntCodeForDTVSub(kernel, tensorParametersA, tensorParametersB, needToWait, waitComment))
    elif kernel["DirectToVgprA"] ^ kernel["DirectToVgprB"]:
      numGlobalReadA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"]
      numGlobalReadB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"]
      numReadsIterCoalesced = self.states.numReadsIterCoalescedA if kernel["DirectToVgprA"] else self.states.numReadsIterCoalescedB
      numReadsIterCoalesced *= kernel["NumLoadsCoalescedA"] if kernel["reorderGRInstForDTVA"] and \
                                                               kernel["NumLoadsCoalescedA"] % 2 == 0 else 1
      numReadsIterCoalesced *= kernel["NumLoadsCoalescedB"] if kernel["reorderGRInstForDTVB"] and \
                                                               kernel["NumLoadsCoalescedB"] % 2 == 0 else 1

      numGlobalReadNonDTV = 0
      if not kernel["DirectToVgprA"]:
        numGlobalReadNonDTV += numGlobalReadA
      if not kernel["DirectToVgprB"]:
        numGlobalReadNonDTV += numGlobalReadB
      numGlobalReadDTV = numGlobalReadA + numGlobalReadB - numGlobalReadNonDTV
      waitComment = "global read wait for DirectToVgpr"
      # delay DirectToVgpr global read (from previous iteration) which is not referred yet (do not delay in beforeBarrier case)
      numRegsIn1set = (numGlobalReadDTV * numReadsIterCoalesced) // kernel["LoopIters"]
      numSet = (u + numReadsIterCoalesced) // numReadsIterCoalesced
      numSetMod = (u + numReadsIterCoalesced) % numReadsIterCoalesced
      if (not beforeBarrier) and numSetMod > 0:
        # if mod > 0, wait is already done by mod == 0 case and no need to wait for same set of global read
        return module
      needToWait = numGlobalReadDTV - numSet * numRegsIn1set
      if isNLL:
        # NLL case, no (non DTV) global load A, B in no load loop. Reset numGlobalReadNonDTV
        numGlobalReadNonDTV = 0
      if kernel["PrefetchGlobalRead"] == 2:
        # PGR=2 case, add numGlobalReadNonDTV for second set of prefetch
        needToWait += numGlobalReadNonDTV
      if u > 0:
        # count number of global read for i < u
        count = 0
        for i in range(u):
          globalReadStr = ' '.join([str(x) for x in self.codes.perIterGlobalRead[i].flatitems()])
          count += countGlobalRead(self.codes.perIterGlobalRead[i])
          # PGR=2 case, global read is in LocalWriteCode
          count += countGlobalRead(self.codes.perIterLocalWrite[i][1])
        needToWait += count
        if u == localWriteEndIter + 1 and beforeBarrier:
          # beforeBarrier case, reduce the amount of non-Vgpr global read
          needToWait -= numGlobalReadNonDTV
      # adjustment for oddLast
      # oddLast case or ScheduleIterAlg < 3 case, ignore all of above and set 0
      if oddLast or kernel["ScheduleIterAlg"] < 3:
        needToWait = 0

      # generate waitcnt code
      module.add(self.getWaitcntCodeForDTVSub(kernel, tensorParametersA, tensorParametersB, needToWait, waitComment))
    return module

  ##############################################################################
  # waitcnt code for PrefetchGlobalRead
  ##############################################################################
  def getWaitcntCodeForPGR(self, kernel, tensorParametersA, tensorParametersB, comment):
    module = Module("PGR wait")
    # generate wait
    if (kernel["DirectToVgprA"] and kernel["DirectToVgprB"]):
      # DTVA and DTVB case, wait code for prefetch is unnecessary
      return module
    count = 0
    if kernel["DirectToVgprA"]:
      # increase vlcnt for DTVA (no need to wait for DTVA global load)
      count += kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"]
    if kernel["DirectToVgprB"]:
      # increase vlcnt for DTVB (no need to wait for DTVB global load)
      count += kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"]

    module.add(self.getWaitcntCodeForDTVSub(kernel, tensorParametersA, tensorParametersB, count, comment))
    return module

  ##############################################################################
  # waitcnt code for DirectToVgpr Sub function
  ##############################################################################
  def getWaitcntCodeForDTVSub(self, kernel, tensorParametersA, tensorParametersB, count, waitComment):
    return self._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, waitComment, skipGlobalReadInst = count)

  ##############################################################################
  # isSwapGlobalReadOrderForDtvOrDtl
  ##############################################################################
  def isSwapGlobalReadOrderForDtvOrDtl(self, kernel, prefetch1=False):
    # check the conditions to swap global read order (from A, B to B, A)
    if kernel["DirectToVgprA"] and kernel["DirectToVgprB"]:
      # DTVA+DTVB case, no swap
      return False
    elif kernel["DirectToVgprA"]:
      # DTVA. Swap with the following conditions
      # - PGR=2 and prefetch1
      # - PGR=1
      if  kernel["PrefetchGlobalRead"] == 2 and prefetch1:
        return True
      elif kernel["PrefetchGlobalRead"] == 1:
        return True
    elif kernel["DirectToVgprB"]:
      # DTVB. Swap with the following conditions
      # - PGR=2 and not prefetch1
      if kernel["PrefetchGlobalRead"] == 2 and (not prefetch1):
        return True
    elif (not kernel["DirectToLdsA"]) and kernel["DirectToLdsB"]:
      # (no DTLA) + DTLB (need to put DTLB first)
      return True
    elif kernel["SwapGlobalReadOrder"] == True:
      return True
    return False

  ##############################################################################
  # WaveSplitK Reduction
  ##############################################################################
  def waveSplitKReduction(self, kernel):
    module = Module("WaveSplitK Reduction")
    module.addComment2("WaveSplitK Reduction")
    # the last instruction may use ValuC VGPR (ex. no tailloop case), need 2 snop
    module.add(SNop(waitState=0, comment="1 wait states"))
    module.add(SNop(waitState=0, comment="1 wait states"))
    accK = 1
    while accK < kernel["NumWaveSplitK"]:
      # if only one VGPR, need to insert one SNOP in between
      if (kernel["ThreadTile1"] == 1 and kernel["ThreadTile0"] == 1) and accK != 1:
        module.add(SNop(waitState=0, comment="1 wait states"))
      for block1 in range(0, kernel["ThreadTile1"]):
        for block0 in range(0, kernel["ThreadTile0"]):
          vgprIdx = block0 + block1 * kernel["ThreadTile0"]
          if accK <= 8:
            module.add(VAddF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr("ValuC+%d"%vgprIdx), src1=vgpr("ValuC+%d"%vgprIdx), dpp=DPPModifiers(bound_ctrl=0, row_shr=accK), comment="Reduce %s elements"%(accK*2) ))
          else:
            module.add(VAddF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr("ValuC+%d"%vgprIdx), src1=vgpr("ValuC+%d"%vgprIdx), dpp=DPPModifiers(bound_ctrl=0, row_bcast=accK-1), comment="Reduce %s elements"%(accK*2) ))
      accK *= 2

    module.addSpaceLine()
    return module

  def simdSpecDispatch(self, kernel, numCodePath):
    module = Module()

    loopLabelBegin = []
    loopLabelSkipBegin = []

    loopChar = self.states.indexChars[kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx]]

    # Wait for all LR in pre-loop to complete before we start main loop
    module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LR in pre-loop to complete"))

    if numCodePath == 1:
      module.add(TextBlock("MAINLOOP 0\n"))
      module.add(SCBranchSCC0(labelName="label_LoopBegin%s"%(loopChar), comment="" ))
      module.add(Label("LoopEnd%s"%(loopChar), "" ))
      return module


    module.addComment0("SIMD specialized dispatch")

    for l in range(numCodePath):
      loopLabelBegin.append(Label("LoopBegin%s_%u"%(loopChar, l), "", alignment=16))
      loopLabelSkipBegin.append(Label("LoopSkipBegin%s_%u"%(loopChar, l), "" ))
    loopLabelEnd = Label("LoopEnd%s"%(loopChar), "" )

    tmpSgpr = self.sgprPool.checkOut(1)
    numbits = 1 if numCodePath == 2 else 2
    module.add(SGetRegB32(dst=sgpr(tmpSgpr), src="hwreg(HW_REG_HW_ID, 4, %u)"%numbits))

    tmpSgpr1 = self.sgprPool.checkOutAligned(3, 2)
    sgprPC = ContinuousRegister(tmpSgpr1, 3)
    for l in range(numCodePath):
      module.addComment0("Branch to code-path %u logic"%l)
      module.add(SCmpEQU32(src0=sgpr(tmpSgpr), src1=hex(l), comment="" ))
      #module.add(SCBranchSCC1(labelName=loopLabelBegin[l].getLabelName(), comment="" ))
      module.add(SCLongBranchScc1(loopLabelBegin[l], sgprPC, \
                                  loopLabelSkipBegin[l].getLabelName(), \
                                  loopLabelBegin[l].getLabelName()))
    self.sgprPool.checkIn(tmpSgpr1)
    self.sgprPool.checkIn(tmpSgpr)

    for l in range(numCodePath):
      module.addComment0("SIMD %u code-path"%l)
      module.add(loopLabelBegin[l])
      module.add(TextBlock("MAINLOOP %u\n"%l))
      module.add(SCBranchSCC0(labelName=loopLabelBegin[l].getLabelName(), comment="" ))
      tmpSgpr1 = self.sgprPool.checkOutAligned(2, 2)
      sgprPC = ContinuousRegister(tmpSgpr1, 3)
      #module.add(SBranch(labelName=loopLabelEnd.getLabelName(), comment="" ))
      module.add(SLongBranchPositive(loopLabelEnd, sgprPC))
      self.sgprPool.checkIn(tmpSgpr1)

    # close unrolled loop
    endStr = ""
    module.addComment2("Unrolled Loop - End%s"%(endStr))
    module.add(loopLabelEnd)

    return module

  ##############################################################################
  # Wrappers
  ##############################################################################

  ##############################################################################
  # longBranchScc0 - 32 bit offset
  # Conditional branch to label when SCC == 0
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc0(self, label: Label, posNeg: int=0, tmpSgprInfo: Optional[ContinuousRegister]=None, comment=""):
    if tmpSgprInfo:
      return SCLongBranchScc0(label, tmpSgprInfo, \
        self.labels.getUniqueNamePrefix("NoBranch"), \
        self.labels.getUniqueNamePrefix("Positive"),
        posNeg, comment)
    else:
      with self.allocTmpSgpr(3) as tmpSgprInfo:
        return SCLongBranchScc0(label, tmpSgprInfo, \
          self.labels.getUniqueNamePrefix("NoBranch"), \
          self.labels.getUniqueNamePrefix("Positive"),
          posNeg, comment)

  ##############################################################################
  # longBranchScc1 - 32 bit offset
  # Conditional branch to label when SCC == 1
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc1(self, label: Label, posNeg: int=0, tmpSgprInfo: Optional[ContinuousRegister]=None, comment=""):
    if tmpSgprInfo:
      return SCLongBranchScc1(label, tmpSgprInfo, \
        self.labels.getUniqueNamePrefix("NoBranch"), \
        self.labels.getUniqueNamePrefix("Positive"),
        posNeg, comment)
    else:
      with self.allocTmpSgpr(3) as tmpSgprInfo:
        return SCLongBranchScc1(label, tmpSgprInfo, \
          self.labels.getUniqueNamePrefix("NoBranch"), \
          self.labels.getUniqueNamePrefix("Positive"),
          posNeg, comment)

##############################################################################
  # longBranchVccnz - 32 bit offset
  # Conditional branch to label when VCC != 0
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchVccnz(self, label: Label, posNeg: int=0, tmpSgprInfo: Optional[ContinuousRegister]=None, comment=""):
    if tmpSgprInfo:
      return SCLongBranchVccnz(label, tmpSgprInfo, \
        self.labels.getUniqueNamePrefix("NoBranch"), \
        self.labels.getUniqueNamePrefix("Positive"),
        posNeg, comment)
    else:
      with self.allocTmpSgpr(3) as tmpSgprInfo:
        return SCLongBranchVccnz(label, tmpSgprInfo, \
          self.labels.getUniqueNamePrefix("NoBranch"), \
          self.labels.getUniqueNamePrefix("Positive"),
          posNeg, comment)

  def sMagicDivWrapper(self, dest, dividend, magicNumber, magicShift):
    tmpVgpr = self.vgprPool.checkOut(2)
    tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
    module = sMagicDiv(dest, self.states.asmCaps["HasSMulHi"], dividend, magicNumber, magicShift, tmpVgprRes)
    self.vgprPool.checkIn(tmpVgpr)
    return module

  def s_mul_u64_u32 (self, dst0, dst1,  src0, src1, tmpVgpr=None, comment=""):
    vtmp0 = self.vgprPool.checkOut(2) if tmpVgpr == None else tmpVgpr
    tmpVgprRes = ContinuousRegister(vtmp0, 2)
    module = SMulInt64to32(dst0, dst1, src0, src1, tmpVgprRes, bool(self.states.asmCaps["HasSMulHi"]), False, comment)
    if tmpVgpr == None:
      self.vgprPool.checkIn(vtmp0)
    return module

  def s_mul_i64_i32 (self, dst0, dst1,  src0, src1, comment):
    vtmp0 = self.vgprPool.checkOut(2)
    tmpVgprRes = ContinuousRegister(vtmp0, 2)
    module = SMulInt64to32(dst0, dst1, src0, src1, tmpVgprRes, bool(self.states.asmCaps["HasSMulHi"]), True, comment)
    self.vgprPool.checkIn(vtmp0)
    return module

  def s_mul_i64_i32_u32 (self, dst0, dst1,  src0, src1, comment):
    module = Module("S_MUL_I64_I32_U32")
    vtmp0 = self.vgprPool.checkOut(2)
    negativeLabel = Label((self.labels.getUniqueNamePrefix("Negative")), comment="")
    multiplydoneLabel = Label((self.labels.getUniqueNamePrefix("MultiplyDone")), comment="")
    module.add(SCmpGeI32(src0, 0))
    module.add(SCBranchSCC0(labelName=negativeLabel.getLabelName(), comment=""))
    tmpVgprRes = ContinuousRegister(vtmp0, 2)
    module.add(SMulInt64to32(dst0, dst1, src0, src1, tmpVgprRes, bool(self.states.asmCaps["HasSMulHi"]), False, comment))
    module.add(SBranch(labelName=multiplydoneLabel.getLabelName(), comment=""))
    module.add(negativeLabel)
    module.add(SAbsI32(src0, src0, comment=""))
    module.add(SMulInt64to32(dst0, dst1, src0, src1, tmpVgprRes, bool(self.states.asmCaps["HasSMulHi"]), False, comment))
    module.add(SXorB32(dst0, dst0, hex(0xFFFFFFFF), comment=""))
    module.add(SXorB32(dst1, dst1, hex(0xFFFFFFFF), comment=""))
    module.add(SAddU32(dst0, dst0, hex(0x1), comment=""))
    module.add(SAddCU32(dst1, dst1, 0, comment=""))
    module.add(multiplydoneLabel)
    self.vgprPool.checkIn(vtmp0)
    return module

  def getCmpAssert(self, function, val0, val1, cookie=-1):
    scratchVgpr = self.vgprPool.checkOut(2)
    function(val0, val1, scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)

  def getMultipleB32Assert(self, sval, multiple2, cookie=-1):
    scratchVgpr = self.vgprPool.checkOut(2)
    self.asmAssert.multiple_b32(sval, multiple2, scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)

  def getVectorDiffAssert(self, v0, v1, expectedScalarDiff, cookie=-1):
    cmpvtmp = self.vgprPool.checkOut(1)
    vtmp = self.vgprPool.checkOut(2)
    self.asmAssert.assert_vector_diff(v0, v1, expectedScalarDiff, cmpvtmp, vtmp, cookie)
    self.vgprPool.checkIn(vtmp)
    self.vgprPool.checkIn(cmpvtmp)

  #to-do, tmp solution, need to move to instruction.py
  def getSCMPKInstruction(self, instOP, s0, simm16, comment="") -> Module:
    imodscmpk = Module()
    Inst0 = SCmpKEQU32
    Inst1 = SCmpEQU32
    if instOP == "EQU32":
      Inst0 = SCmpKEQU32
      Inst1 = SCmpEQU32
    elif instOP == "LGU32":
      Inst0 = SCmpKLGU32
      Inst1 = SCmpLgU32
    elif instOP == "GEU32":
      Inst0 = SCmpKGeU32
      Inst1 = SCmpGeU32
    elif instOP == "GTU32":
      Inst0 = SCmpKGtU32
      Inst1 = SCmpGtU32
    else:
      assert 0,"getSCMPKInstruction failed"
    if self.states.asmCaps["HasSCMPK"]:
      imodscmpk.add(Inst0(sgpr(s0), simm16, comment=comment))
    else:
      tmpScmp = self.sgprPool.checkOut(1, preventOverflow=False)
      imodscmpk.add(SMovB32(sgpr(tmpScmp), simm16))
      imodscmpk.add(Inst1(src0=sgpr(s0), src1=sgpr(tmpScmp), comment=comment))
      self.sgprPool.checkIn(tmpScmp)
    return imodscmpk

def _getEccOffset(totalWidth, bpr, bpe, glvw, idx, numVgprG2L):
  if totalWidth < 1: # Need extra offset if global read < 1
    modVal = int(bpr / (bpe * glvw))
    left = idx % modVal
    return numVgprG2L * left
  else:
    return 0
