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

from . import Code
from .Common import gfxName, globalParameters, getCOVFromParam, tPrint, printExit, printWarning, roundUp
from .Component import Component
from .KernelWriter import KernelWriter
from .SolutionStructs import isPackedIndex
from .Utils import ceil_divide
from .AsmMemoryInstruction import MemoryInstruction, getGlcBitName, getSlcBitName
from .AsmRegisterPool import RegisterPool
from .AsmUtils import inst, vgpr, sgpr, accvgpr, log2, vectorStaticDivideAndRemainder, vectorStaticDivide, vectorStaticRemainder, scalarStaticDivideAndRemainder, vectorStaticMultiply, staticMultiply, scalarStaticMultiply, instCommentOnly

from math import ceil, trunc, modf, log
from copy import deepcopy
import collections
import os
import shlex
from enum import Enum

class ZeroPadReg:
  class State(Enum):
    Allocated=0
    MacroDef=1
    CalculatedAddr=2
  def __init__(self, zp, regName, vgprIdx, perp, sPerp, para, sPara):
    self.zp = zp
    self.state = ZeroPadReg.State.Allocated
    self.regName = regName
    self.vgprIdx= vgprIdx
    self.perp = perp
    self.sPerp = sPerp
    self.para = para
    self.sPara = sPara

  def isMatch(self, perp, sPerp, para, sPara):
    return self.perp==perp and self.sPerp==sPerp and self.para==para and self.sPara==sPara

class PreLoopVmcntCase(Enum):
  Undefined = 0
  Basic_Load = 1
  OptNLL_Store = 2
  OrdNLL_E1_Store = 3
  OrdNLL_B1_Store = 4

################################################################################
# Assembly Kernel
################################################################################
class KernelWriterAssembly(KernelWriter):

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming, removeTemporaries=True ):
    super(KernelWriterAssembly, self).__init__( \
        kernelMinNaming, kernelSerialNaming, removeTemporaries)
    self.do = {}
    self.do["PreLoop"]     = True
    self.do["GlobalReadA"] = True
    self.do["GlobalReadB"] = True
    self.do["GlobalInc"]   = True
    self.do["LocalWrite"]  = True
    self.do["LocalReadA"]  = True
    self.do["LocalReadB"]  = True
    self.do["Wait"]        = True
    self.do["Sync"]        = True
    self.do["MAC"]         = True
    self.do["PostLoop"]    = True
    self.do["ApplyAlpha"]  = True
    self.do["GlobalWrite"] = True
    self.do["EdgeWrite"]   = True
    self.do["KeepDirectToLdsAlloc"] = False  # If true, keep regs used for LDS alloc even if not used

    # Remove me if 906 can work with beta in SGPR
    # Also can push alpha/beta recalc back to host for HPA mode
    self.betaInSgpr = True

    # Various debug flags and modes
    self.db = {}
    self.db["EnableAsserts"]       = globalParameters["EnableAsserts"]  # Enable assertion codegen. Requires 2 SGPR.
    self.db["DebugKernelMaxItems"] = 16  # Capture first N(=16) print values, ignore subsequent.  If -1, debug writing is faster but writing more than 16 values is undefined.

    # Chicken bit to add conservative synchronization at strategic points:
    # 0x01 = waitcnt + barrier after vector load
    # 0x02 = waitcnt at self.wait() for globalRead
    # 0x04 = waitcnt at self.wait() for localWrite
    # 0x08 = waitcnt at self.wait() for localRead
    # 0x10 = waitcnt after summation iteration, this can catch lingering ds or vm activity from summation loop
    # 0x20 = waitcnt before each write batch
    # 0x40 = waitcnt after each write batch
    self.db["ConservativeWaitCnt"] = 0x00

    self.db["InitLds"]     = False  # Initialize LDS at start of kernel
    self.printedAssertCnt  = 0
    self.initLdsValue     = 0xFFFFFFFF  # Value to use for LDS Init, if enabled

    # InitSgpr and InitVgpr can initialize at various points:
    #  0x1: Init at kernel start
    #  0x2: Init at end of summation loop (after tail too) - this is just before store loop
    self.db["InitSgpr"]   = 0x0  # init SGPRs
    self.initSgprValue    = 0x0  # Value to use for Sgpr Init, if enabled

    self.db["InitVgpr"]   = 0x0  # init VGPRs
    self.initVgprValue    = 0xFFFFFFFF  # Value to use for Vgpr Init, if enabled

    # Debug and Check flags:
    # Check A and B values loaded from memory to ensure they are 1
    # Requires DataInitTypeAB=1.
    # Only works if the problem uses full tiles (no edges)
    # Mismatches will assert (generate GPUVM fault)
    self.db["CheckValue1A"] = globalParameters["EnableDebugA"]
    self.db["CheckValue1B"] = globalParameters["EnableDebugB"]

    # Check value in C matrix.
    # Caveats:
    #  - Only works for single, or Half/BF with HPA.
    #  - Checks after alpha calc for each element.  Later elements (in the TT) will not yet have applied their alpha.
    #  - Only works if matrix is integral multiple of macro-tile (no edges) - check is dumb so doesn't know
    #    which work-items are outside the valid edge.
    #  - Does not work in OptNoLoadLoop
    self.db["CheckValueC"]  = globalParameters["EnableDebugC"]
    # value expected if CheckValueC is set. Use '.' for FP.
    # For example could be 16.0 if U=8 and alpha=2
    self.db["ValueCExpectedValue"] = globalParameters["ExpectedValueC"]

    # Force an expected value for all C outputs.
    # May be useful for checking store path
    # See same caveats as CheckValueC
    self.db["ForceExpectedValue"]  = globalParameters["ForceCExpectedValue"]

    # Force VSerial value into the output, this will
    # not match reference but can be useful to see which work-items are
    # storing which values
    # See same caveats as CheckValueC
    self.db["ForceVSerial"] = False

    # can't do both of these since they both override output
    assert (not (self.db["ForceExpectedValue"] and self.db["ForceVSerial"]))

    self.db["ForceInputValueA"] = False
    self.db["ForceInputValueB"] = False
    self.db["ForceValueA"] = 1.0
    self.db["ForceValueB"] = 1.0

    self.db["CheckStoreC"] = -1 # -1 disables, reload and verify output data.  Specify expected constant value.
    #self.db["CheckStoreC"] = 1024.0 # possible value

    self.db["ForceEdgeStores"] = 0 # 1=force use of edge store path for all tiles,  2=add assert in non-edge stores
    self.db["AssertNoEdge"] = 0 # Add assert in edge store code so crashes if executed

    # print vgpr register pool checkins and checkouts
    self.db["PrintRP"] = 0
    self.db["AssertOnSgprOverflow"] = False
    self.db["PrintStoreRegisterDb"] = False

    # Number of times localReadDo(localWriteDo) has been called by the code-generator.
    # Used to control debug enablement.
    # Note this increments as the assembly code is generated not as it executes
    # so it can be used to determine which iteration of the unroll is being generated
    self.localReadDoCnt   = 0
    self.localWriteDoCnt  = 0

    self.maxVgprs = 256
    # max allowed is 112 out of 112 , 6 is used by hardware 4 SGPRs are wasted
    self.maxSgprs = 102
    self.maxAgprs = 256
    self.maxOccupancy = 10

    self.endLine = "\n"
    self.syncStr = "s_barrier"
    self.commentPrefix = "/*"
    self.commentSuffix = "*/"
    self.commentHR = "*"*40
    self.indent = ""
    self.labels = {}
    self.localReadOffsetA = 0
    self.localReadOffsetB = 0
    self.inTailLoop = False

  @property
  def vcc(self) -> str:
    if self.kernel["WavefrontSize"] == 64:
      return "vcc"
    else:
      return "vcc_lo"

  @property
  def exec(self) -> str:
    if self.kernel["WavefrontSize"] == 64:
      return "exec"
    else:
      return "exec_lo"

  @property
  def laneSGPRCount(self) -> int:
    """ How many SGPRs does it take to have one bit per lane? """
    if self.kernel["WavefrontSize"] == 64:
      return 2
    else:
      return 1

  @property
  def rpga(self) -> int:
    """ (2) registers per global address (64-bit) """
    return 2

  @property
  def rpla(self):
    """ (1) registers per local address (32-bit) """
    return 1

  @property
  def rpgo(self):
    """ (1) registers per global 32-bit offset (some intructions only support 32-bit offset) """
    return 1

  def getCompileArgs(self, sourceFileName, objectFileName, *moreArgs, isa=None, wavefrontSize=None):
    if isa is None:
      isa = self.version
    if wavefrontSize is None:
      wavefrontSize = self.kernel["WavefrontSize"]

    launcher = shlex.split(os.environ.get('Tensile_ASM_COMPILER_LAUNCHER', ''))

    assembler = globalParameters['AssemblerPath']
    if assembler is None:
      raise ValueError('No assembler available; set TENSILE_ROCM_ASSEMBLER_PATH to point to ROCm Clang.')
    rv = launcher + [assembler,
          '-x', 'assembler',
          '-target', 'amdgcn-amd-amdhsa']

    cov = getCOVFromParam(globalParameters["CodeObjectVersion"])
    rv += ['-mcode-object-version=%s' % (cov)]

    rv += ['-mcpu=' + gfxName(isa)]

    if wavefrontSize == 64:
      rv += ['-mwavefrontsize64']
    else:
      rv += ['-mno-wavefrontsize64']

    rv += moreArgs

    rv += ['-c', '-o', objectFileName, sourceFileName]

    return rv

  def getLinkCodeObjectArgs(self, objectFileNames, coFileName, *moreArgs):
    assembler = globalParameters['AssemblerPath']
    if assembler is None:
      raise ValueError('No assembler available; set TENSILE_ROCM_ASSEMBLER_PATH to point to ROCm Clang.')
    rv = [assembler,
          '-target', 'amdgcn-amd-amdhsa']
    # Add build-id for builds with rocm 5.3+
    compilerVer = globalParameters['HipClangVersion'].split(".")[:2]
    compilerVer = [int(c) for c in compilerVer]
    if len(compilerVer) >= 2 and (compilerVer[0] > 5 or (compilerVer[0] == 5 and compilerVer[1] > 2)):
      rv += ["-Xlinker", "--build-id"]
    rv += moreArgs
    rv += ['-o', coFileName] + objectFileNames
    return rv

  def getVgprOccupancy(self, numThreads, vgprs, doubleVgpr=False):
    multiplier = int(ceil(max(numThreads, 256) / 256.0)) # example: wg=512 multiplier=2, 1024=4
    maxOccupancy = self.maxOccupancy//multiplier

    vgprAllocateAligned = 4    if not doubleVgpr else 8
    totalVgprs = self.maxVgprs if not doubleVgpr else self.maxVgprs*2
    vgprsAligned = int(ceil(vgprs/vgprAllocateAligned))*vgprAllocateAligned
    vgprsAligned *= multiplier

    if   vgprsAligned > totalVgprs:  return 0
    elif vgprsAligned < 1:           return maxOccupancy
    occupancy = min(totalVgprs//vgprsAligned, maxOccupancy)

    #print("vgprs = ", vgprs, "vgprsAligned = ", vgprsAligned, "doubleVgpr = " ,doubleVgpr, "Occupancy = ", occupancy)

    return occupancy

  ########################################
  def getOccupancy(self, numThreads, vgprs, ldsSize, accvgprs=0, doubleVgpr=False):

    ldsLimitedOccupancy = self.getLdsLimitedOccupancy(ldsSize)

    if not doubleVgpr:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, vgprs,          doubleVgpr)
      accvgprLimitedOccupancy = self.getVgprOccupancy(numThreads, accvgprs,       doubleVgpr)
    else:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, vgprs+accvgprs, doubleVgpr)
      accvgprLimitedOccupancy = vgprLimitedOccupancy

    return min(ldsLimitedOccupancy, vgprLimitedOccupancy, accvgprLimitedOccupancy)

  # TODO: also consider sgpr
  def getMaxRegsForOccupancy(self, numThreads, vgprs, ldsSize, accvgprs=0, doubleVgpr=False):
    lastVgprs = vgprs
    considerAccVgprs = 0       if not doubleVgpr else accvgprs
    totalVgprs = self.maxVgprs if not doubleVgpr else self.maxVgprs*2

    initOccupancy = self.getOccupancy(numThreads, vgprs, ldsSize, accvgprs, doubleVgpr)
    if initOccupancy == 0: return lastVgprs

    while (vgprs + considerAccVgprs) < totalVgprs and vgprs < self.maxVgprs:
      vgprs += 1
      if self.getVgprOccupancy(numThreads, vgprs + considerAccVgprs, doubleVgpr) >= initOccupancy:
        lastVgprs = vgprs
        next
      else:
        break

    return lastVgprs

  @staticmethod
  def getLdsLimitedOccupancy(ldsSize):
    maxLds = 65536
    # As ldsSize gets large, rounding might push us slightly higher than maxLds.
    # Clamp at maxLds
    ldsSize = min(ldsSize + 255, maxLds) & 0x1ff00 # 256-byte granularity

    ldsLimitedOccupancy = maxLds//ldsSize
    return ldsLimitedOccupancy

  @staticmethod
  def getLdsSize(kernel):
    ldsSize = kernel["LdsNumElements"] * kernel["ProblemType"]["DataType"].numBytes()
    return ldsSize

  ########################################
  def sizeRef(self, idx):
    """
    Return sgpr() or const with the specified size
    See above definitions for how these are mapped to Free or Sum sizes
    based on the problem definition.
    """
    idxChar= globalParameters["IndexChars"][idx]
    return sgpr("Size%s"%idxChar)

  def loopChar(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return globalParameters["IndexChars"][loopDim]

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
    if self.unrollIncIsDepthU:
      return inst("s_cmp_gt_u32", "DepthU", \
          sgpr("UnrollLoopLastIter"), comment)
    else:
      return inst("s_cmp_eq_u32", self.loopCounter(kernel, self.unrollIdx), \
          0, comment)

  def isConstUnitStride(self, stride):
      return stride.startswith("const")

  ########################################
  def strideRef(self, tc, dim):
    """
    Return sgpr with specified stride or define starting with const if constant.
    dim is index 0...max indices and is in global index space.
    """
    problemType = self.kernel["ProblemType"]
    if tc == 'WS': # WS used by Stream-K kernel
      tc = 'D'
    if tc in ['A','B']:
      if not problemType["UseInitialStridesAB"] and \
          dim == problemType["IndexAssignments%s"%tc][0]:
        return ("constStride%s%s"%(tc,self.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.indexChars[dim]))
    elif tc in ['D','C']:
      if not problemType["UseInitialStridesCD"] and dim == 0:
        return ("constStride%s%s"%(tc,self.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.indexChars[dim]))
    else:
      raise ValueError("unexpected tensorChar='%s' in stride function"%tc)

  ########################################
  # Get Label
  # return label number - create new if it doesn't already exist
  ########################################
  def getLabelNum(self, name):
    if name not in self.labels:
      self.labels[name] = len(self.labels)
    return self.labels[name]

  ########################################
  # return label name including a unique number
  # create new if it doesn't already exist
  ########################################
  def getNamedLabel(self, name):
    if name not in self.labels:
      self.labels[name] = "%s_%u" % (name, len(self.labels))
    return self.labels[name]

  ########################################
  # return label name that is always unique
  # useful when trying to re-use subroutines that create labels
  ########################################
  def getNamedLabelUnique(self, name):
    key = name + "_" + str(len(self.labels))
    self.labels[key] = key
    return key

  ########################################
  # return string that defines a unique named name_number
  ########################################
  def getNamedLabelDef(self, name, labelComment=""):
    t = "%s: // %s\n" % (self.getNamedLabel(name), labelComment)
    return t

  ########################################
  # return string that defines a unique numeric label
  # labelComment is a comment string if this is a label definition
  ##############################################################################
  def getLabelDef(self,name,labelComment=""):
    t = "label_%04u: // %s %s\n" % (self.getLabelNum(name), name, labelComment)
    return t

  ##############################################################################
  # define a label and return undecorated label_%4u - suitable for using as jump target
  ##############################################################################
  def getLabelTarget(self,name,labelDef=None):
    t = "label_%04u" % (self.getLabelNum(name))
    return t

  ##############################################################################
  def getUniqLabel(self):
    name = "uniq_label_" + str(len(self.labels))
    return self.getLabelNum(name)

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
      raise RuntimeError("Could not find valid memory instruction for operation=%s, width=%f, kernel=%s" %(operation, width, self.kernelName))

  class TmpSgpr:
    """ A temporary register which is automatically returned to sgpr pool when class is destroyed. """
    def __init__(self, regPool, num, align, tag=None):
      self.regPool = regPool
      self.regIdx = regPool.checkOutAligned(num, align, tag=tag, preventOverflow=False)

    def idx(self):
      return self.regIdx

    def __int__(self):
      return self.idx()

    def __del__(self):
      self.regPool.checkIn(self.regIdx)

  def getTmpSgpr(self, num, align=None, tag=None):
    if align==None:
      align = 1 if num==1 else 2
    if tag==None:
      tag = "getTmpSgpr(%d)"%num

    t = self.TmpSgpr(self.sgprPool, num, align, tag)
    if t.idx()+num > self.maxSgprs:
      self.overflowedResources = 2
      if self.db["AssertOnSgprOverflow"]:
        assert(t.idx()+num <= self.maxSgprs)
    return t

  def definePreloadedSgprPool(self, numSgprs):
    self.sgprPool.add(self.sgprPool.size(), numSgprs, newStatus = RegisterPool.Status.AvailableForPreload)

  def defineSgpr(self, name, numSgprs, align=1, kernarg=None, preload=False):
    if numSgprs == 0: return
    if kernarg is None:
      kernarg = preload

    if self.kernel["PreloadKernelArguments"] != 1:
      preload = False

    sgprIdx = self.sgprPool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=False, kernarg=kernarg, preload=preload)

    if preload:
      self.preloadedKernargs[0] = min(sgprIdx, self.preloadedKernargs[0])
      self.preloadedKernargs[1] = max(sgprIdx+numSgprs, self.preloadedKernargs[1])
    elif kernarg:
      self.loadedKernargs[0] = min(sgprIdx, self.loadedKernargs[0])
      self.loadedKernargs[1] = max(sgprIdx+numSgprs, self.loadedKernargs[1])

    #self.sgprIdx = roundUpToNearestMultiple(self.sgprIdx,align)
    #print (name, "->", self.sgprIdx, "+", numSgprs)
    self.sgprs[name] = sgprIdx

    return sgprIdx

  def undefineSgpr(self, name):
    self.sgprPool.checkIn(self.sgprs[name])
    # later references will result in compile-time error (with odd 'error: expected relocatable expression')
    # and 'Kernel ... not found in any loaded module'
    # TODO: temporarily disable undef as it seems to have issues
    # return ".set sgpr%s, UNDEF\n" % name
    return ".set %s, UNDEF\n" % name

  def defineVariableSgprs(self, kernel):
    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    # self.lastPostLoopSgpr = self.sgprPool.size()

    if self.unrollIncIsDepthU:
      # product of all summation dimensions, this also will be divided if GSU is enabled
      self.defineSgpr("UnrollLoopLastIter", 1)

    if kernel["StreamK"]:
      # StreamK vars
      self.defineSgpr("StreamKIdx", 1)
      self.defineSgpr("StreamKIter", 1)
      self.defineSgpr("StreamKIterEnd", 1)
      self.defineSgpr("StreamKLocalStart", 1)
      self.defineSgpr("StreamKLocalEnd", 1)
      if kernel["StreamKAtomic"] == 0:
        self.defineSgpr("SrdWS", 4, 4)

    if kernel["PackSummationDims"] and kernel["GlobalSplitU"]>1:
      self.defineSgpr("GsuNumIter%s"%self.loopChar(kernel,self.unrollIdx), 1)

    for tc in ('A', 'B'):
      for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
        (freeDim, sumDim, padStart, padEnd) = zp
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        # These will eventually be read as kernel args:
        self.defineSgpr("ElementEdge%s%s"%(tc, sumDimChar),1)
        if kernel["PackSummationDims"]:
          self.defineSgpr("Iter%s"%(sumDimChar),1)

    if not self.isInitCodeOptLW:
      # init code optimization: define PerpOverhangVccA, B just after kernel args (need this before undef OFFSET)
      if kernel["FractionalLoad"] == 2:
        if kernel["fractionalPerpOverhangA"]:
          self.defineSgpr("PerpOverhangVccA", 2, 2)
        if kernel["fractionalPerpOverhangB"]:
          self.defineSgpr("PerpOverhangVccB", 2, 2)
    if self.use64bShadowLimit:
      # If need more SGPR could overlap this with the Tensor2dSize regs
      self.defineSgpr("ShadowLimitA", 2, 2)
      self.defineSgpr("ShadowLimitB", 2, 2)

    if kernel["PackSummationDims"]:
      for tc in ('A','B'):
        self.defineSgpr("InitialSrd%sBase"%tc, 2)
        self.defineSgpr("InitialSrd%sLimit"%tc, 2 if self.use64bShadowLimit else 1)

    if self.staggerU:
      self.defineSgpr("StaggerUIter", 1)  # stagger loop iterations, used for various iter counts in the code
      self.defineSgpr("WrapUA", 2)  # Bytes to add to SrdA to reset address from N-1 iter to AddressA
      self.defineSgpr("WrapUB", 2)  # Bytes to add to SrdB to reset address from N-1 iter to AddressB

    if kernel["PersistentKernel"]:
      self.defineSgpr("SerialWorkGroupIter", 1) # Track sequential persistent wg
      # self.defineSgpr("PersistentLoopIter", 1) # Back-up: The count of current persistent loop, not needed now
      if kernel["PersistentKernelAlongBatch"]:
        self.defineSgpr("WGKSerial", 1)  # for persistent kernel along batch, wgK of PK-remapping
        self.defineSgpr("WGIJSerial", 1)  # for persistent kernel along batch, wgIJ of PK-remapping
    if self.prefetchAcrossPersistent0:
      self.defineSgpr("PrevWorkGroup0", 1) # WorkGroup0 from prev iteration, use for stores
      self.defineSgpr("PrevWorkGroup1", 1) # WorkGroup0 from prev iteration, use for stores
      # self.defineSgpr("PrevWorkGroup2", 1) # WorkGroup0 from prev iteration, use for stores

    if self.canOptimizePreLoopLWVmcnt:
      self.defineSgpr("PreLoopLWVmcntCase", 1) # Indicating which case for optimizing PreLoop Vmcnt (based on the Store Inst)

    self.defineSgpr("GlobalReadIncsA", self.numSgprGlobalReadIncsA)
    self.defineSgpr("GlobalReadIncsB", self.numSgprGlobalReadIncsB)

    # packTmp vgpr for wider local read + v_perm
    lrvwTileMax = max(self.lrvwTileA, self.lrvwTileB)
    if lrvwTileMax > 1:
      numElemPerReg = self.bpr // kernel["ProblemType"]["DataType"].numBytes()
      if numElemPerReg > 1:
        for b in range(min(numElemPerReg, lrvwTileMax)):
          self.defineSgpr("PackKForV%u"%b, 1)

    if not self.isInitCodeOptLW:
      # init code optimization: define LocalWriteUseSgprA, B just after kernel args (need this before undef OFFSET)
      if kernel["LocalWriteUseSgprA"]:
          self.defineSgpr("LocalWriteAddrA", 1)
      if kernel["LocalWriteUseSgprB"]:
          self.defineSgpr("LocalWriteAddrB", 1)

    if kernel["_UseSgprForGRO"]:
      needFirstSgprOffset = kernel["DirectToLdsA"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.numGlobalReadOffsetsA if needFirstSgprOffset else (self.numGlobalReadOffsetsA-1)
      self.defineSgpr("ScalarGlobalReadOffsetA", numberOfSgpr)

      needFirstSgprOffset = kernel["DirectToLdsB"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.numGlobalReadOffsetsB if needFirstSgprOffset else (self.numGlobalReadOffsetsB-1)
      self.defineSgpr("ScalarGlobalReadOffsetB", numberOfSgpr)

    # Register to store rounding term if fp16 alt rnz
    if kernel["ProblemType"]["Fp16AltImpl"] and kernel["ProblemType"]["Fp16AltImplRound"]:
      self.defineSgpr("Fp16AltOffset", 1)
      self.defineSgpr("Fp16AltNanCheck", 2, 2)

    # dedicated sgpr(S) for storeC VGPR indexing
    # sgpr semaphore for message synchronization between different part of code section
    if kernel["StoreCInUnroll"]:
      needAddrC = not kernel["AssertCEqualsD"] and kernel["ProblemType"]["UseBeta"]
      self.defineSgpr("StoreCIndex0",1)
      self.defineSgpr("StoreCOffsetAddr",1)
      self.defineSgpr("StoreCEnableCountInit",1)
      self.defineSgpr("StoreCEnableCount",1)
      self.defineSgpr("StoreCAvail",2,2)
      if needAddrC:
        self.defineSgpr("SrdCBackup",2,2)
      self.defineSgpr("SrdDBackup",2,2)
      if needAddrC:
        self.defineSgpr("CAddrInc",1)
      self.defineSgpr("DAddrInc",1)
      # initialization of StoreCInUnroll C/D addr inc
      self.initializeStoreCInUnrollAddrIncValues(kernel)
      if self.StoreCInUnrollAddrIncHoffset > 0 or self.StoreCInUnrollNumInterleaveV > 1:
        if needAddrC:
          self.defineSgpr("CAddrIncV1",1)
        self.defineSgpr("DAddrIncV1",1)
        if self.StoreCInUnrollAddrIncV2Iterations > 0:
          if needAddrC:
            self.defineSgpr("CAddrIncV2",1)
          self.defineSgpr("DAddrIncV2",1)
        if self.StoreCInUnrollAddrIncV3Iterations > 0:
          if needAddrC:
            self.defineSgpr("CAddrIncV3",1)
          self.defineSgpr("DAddrIncV3",1)

    # debug flag to allocate dummy / unused sgpr
    # useful when comparing code that adds new kernel arguments to see what
    # was actually changed
    numDummySgpr= 0
    for i in range(numDummySgpr):
      self.defineSgpr("DummySgpr%d"%i, 1)

    if self.sgprPool.size() >= self.maxSgprs:
      print ("warning: Number of defined SGPRS (%d) overflowed max SGPRS (%d)." \
               % (self.sgprPool.size(), self.maxSgprs))

    # TODO-persistent - likely recompute some of the registers above.
    if kernel["PersistentKernel"] or kernel["StreamK"]:
      self.lastPostLoopSgpr = self.sgprPool.size()

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterAssembly, self).initKernel(kernel, tPA, tPB)
    problemType = kernel["ProblemType"]

    dkp = kernel["DisableKernelPieces"]
    self.do["NullKernel"]  = dkp >= 9 or dkp == -9

    self.kernel = kernel

    # init these here in case some kernel pieces are disabled for performance exploration:
    tPA["localReadOffset"] = 0
    tPB["localReadOffset"] = 0

    self.sgprs=collections.OrderedDict()
    self.preloadedSgprs = collections.OrderedDict()
    self.loadedKernargs = [1000000, -1]
    self.preloadedKernargs = [1000000, -1]

    self.LdsOOB = 0xF00000

    #---
    # Internal optimization and debug controls.
    # These have a default which is almost always faster so don't make a full-blown YAML parm
    # But have a control here so we can disable for debugging and also easily tell
    # which parts of the code were changed to support the new mode.
    self.globalReadIncsUseVgpr = False if kernel["BufferLoad"] else True

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
    # ZeroPad requires groOffsetInMacroTile since it needs the gro offsets in each dimension to include
    # the tile components, since those same vars are used to compute the ZP offsets used for edge comparisons.
    if problemType["ZeroPadA"] == [] and problemType["ZeroPadB"] == [] and \
       len(kernel["PackedC0IndicesX"])==1 and len(kernel["PackedC1IndicesX"])==1 and kernel["BufferLoad"]:
      self.groOffsetInMacroTile = 1
    else:
      self.groOffsetInMacroTile = 0

    self.use64bPackSumOffset = 0  # use 2 SGPR for extracting packed summation dims.  Not supported, but this marks eventual required changes

    # use 64-bit buffer limit shadow register
    # PackSummationDims does not support shadow limit - the address calc code would need to restore the shadow limit, which is possible
    # but not implemented or tested
    self.use64bShadowLimit = kernel["Use64bShadowLimit"] and kernel["BufferLoad"] and not kernel["PackSummationDims"]

    # Check if the address setup code for LWA and GRO causes register growth.
    # This is not an error condition but bears further investigation.
    # In particular if PrefetchAcrossPersistent=1 then the NewTile setup code
    # will be run before the no-load-loop iteration where registers are still
    # tight.  Realistically we just have the GlobalToLocal VGPRs, all else is
    # growth.
    self.preventVgprOverflowDuringNewTile = 0 and not globalParameters["ForceGenerateKernel"]

    # For Beta:
    # Rather than waiting for all loads to finish with s_waitcnt vmcnt(0), interleave
    # appropriate vmcnts into the stores so they issue as loads become available
    self.interleaveStoreVmcnt = (not kernel["GroupLoadStore"]) and kernel["BufferStore"]

    # if >0, shift the start of the SRD left by specified #elements (not bytes)
    # Gives pointer shift some room to move left, even into the previous macro-tile
    # This slightly reduces the range of the GRO since they have to include the offset
    # Pointer shift still cannot be used with very small matrices < GRVW
    self.srdShiftLeft = {}
    # GRVW<1 case, bpe calculation is included in offset. Need bpe here
    bpeOffsetA = 1 if kernel["GlobalLoadVectorWidthA"] >= 1 else self.tPA["bpe"]
    bpeOffsetB = 1 if kernel["GlobalLoadVectorWidthB"] >= 1 else self.tPB["bpe"]
    self.srdShiftLeft["A"] = int(kernel["GlobalLoadVectorWidthA"] * bpeOffsetA)
    self.srdShiftLeft["B"] = int(kernel["GlobalLoadVectorWidthB"] * bpeOffsetB)

    self.checkGRO = False
    # checkGRO requires useSgprForGRO=0 so that code allocates and uses
    # the VGPRs that are used for the GRO offset checking
    assert not (kernel["_UseSgprForGRO"] and self.checkGRO)

    # init code optimization setting
    # generate local read/write address code before wait for kernel arg load
    # enable only for Asm
    self.isInitCodeOptLR = True
    self.isInitCodeOptLW = self.isInitCodeOptLR
    # disable init code opt for the following conditions
    # - enabledSplitLDS and UnrollMajorLDSA or B: uReg will be overwritten (cannot move local write code before global read offset code)
    # - PersistentKernel: need local write and tile code inside PK loop (cannot move to before wait for kernel arg load)
    # - not BufferLoad: requires WorkGroup0,1 (means need to calculate WGM)
    #-  groOffsetInMacroTile=0: requires WorkGroup0,1 (means need to calculate WGM)
    if ((kernel.enabledSplitLDS and (kernel["UnrollMajorLDSA"] or kernel["UnrollMajorLDSB"])) or \
        kernel["PersistentKernel"] or kernel["StreamK"] or \
        (not kernel["BufferLoad"]) or \
        self.groOffsetInMacroTile == 0):
      # disable init opt for local write
      self.isInitCodeOptLW = False
    # disable init code opt for local read if kernel["PreloadKernelArguments"] == 1
    if kernel["PreloadKernelArguments"] == 1:
      self.isInitCodeOptLR = False

    # Debug mode to explore combining VGPRs.
    # Saves VGPRs but doesn't generate correct answer
    self.combineLocalAddresses = 0

    # ISA version, such as 803
    self.version = globalParameters["CurrentISA"]
    if "ISA" in kernel:
      self.version = tuple(kernel["ISA"])
    if not globalParameters["AsmCaps"][self.version]["SupportedISA"]:
      defaultIsa = (9,0,0)
      print("warning: ISA:", self.version, " is not supported; overriding with ", defaultIsa)
      self.version = defaultIsa

    self.doubleVgpr = False
    if globalParameters["ArchCaps"][self.version]["ArchAccUnifiedRegs"] or (kernel["WavefrontSize"] == 32):
      self.doubleVgpr = True

    if kernel["EnableMatrixInstruction"]:
      if (kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64') and (not self.asmCaps["HasMFMA_f64"]):
        raise RuntimeError("FP64 MatrixInstruction not supported for {0}".format(self.version))
      elif not (self.asmCaps["HasMFMA"] or self.asmCaps["HasWMMA"]):
        raise RuntimeError("MatrixInstruction not supported for {0}".format(self.version))

      if kernel["MFMA_BF16_1K"] and not self.asmCaps["HasMFMA_bf16_1k"]:
        raise RuntimeError("BF16_1k MatrixInstruction not supported for {0}".format(self.version))

      if kernel["ProblemType"]["Fp16AltImpl"] and not self.asmCaps["HasMFMA_bf16_1k"]:
        raise RuntimeError("Fp16AltImpl not supported for {0}".format(self.version))

      if (kernel["EnableF32XdlMathOp"] and kernel["ProblemType"]["F32XdlMathOp"].isXFloat32() and (not self.asmCaps["HasMFMA_xf32"])):
        raise RuntimeError("XF32 MatrixInstruction not supported for {0}".format(self.version))

    self.AsmBugs = {}
    self.AsmBugs["ExplicitCO"] = globalParameters["AsmCaps"][self.version]["HasExplicitCO"]
    self.AsmBugs["ExplicitNC"] = globalParameters["AsmCaps"][self.version]["HasExplicitNC"]

    hasDtl = globalParameters["AsmCaps"][self.version]["HasDirectToLdsDest"] or globalParameters["AsmCaps"][self.version]["HasDirectToLdsNoDest"]
    if not hasDtl:
      if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
        printExit("DirectToLds requested, but not available on this architecture ( {} )".format(self.version))
      kernel["DirectToLdsA"] = False
      kernel["DirectToLdsB"] = False
      kernel["LocalWriteUseSgprA"] = False # Requires DirectToLdsA
      kernel["LocalWriteUseSgprB"] = False # Requires DirectToLdsB

    # set up useAtomicAdd
    # TODO Stream-K In future change, either gneeralize GSUAA option, or add toggle to control SK behaviour
    self.useAtomicAdd = kernel["GlobalSplitUAtomicAdd"]

    # OptPreLoopVmcnt for PAP:
    # the vmcnt for _ds_store in pre-loop can be optimized to skip the store of prev PKLoop
    #
    # a dictionary storing the vmcnt numbers for each case:
    # case 1: first PK-Loop (no previous store), cnt = #-basic-globalload
    # case 2: after Opt.NLL (no Beta), cnt = #-prev-store (no beta,no edge) +  #-basic-globalload
    # case 3: after Ord.NLL (with Edge but No Beta), cnt = #-prev-store (edge store) +  #-basic-globalload
    # case 4: after Ord.NLL (with Beta), cnt = no needed for vmcnt
    self.preLoopVmcntDict = { \
      PreLoopVmcntCase.Basic_Load:0, \
      PreLoopVmcntCase.OptNLL_Store:0, \
      PreLoopVmcntCase.OrdNLL_E1_Store:0 }
      # Case4: No need to count store vmcnt for next PreLoop since OrdNLL_B1_Store already has vmcnts waiting for loading beta
      # PreLoopVmcntCase.OrdNLL_B1_Store:0 }

    # a dictionary storing the keywords to be replaced for each case:
    # case 1: replace the vmcnt("Basic_Load") with vmcnt(N)
    # case 2: replace the vmcnt("OptNLL_Store" + "Basic_Load") with vmcnt(M1+N)
    # case 3: replace the vmcnt("OrdNLL_E1_Store" + "Basic_Load") with vmcnt(M2+N)
    # case 4: s_waitcnt vmcnt will be removed, no need to replace
    self.preLoopCaseToReplaceKWList = { \
      PreLoopVmcntCase.Basic_Load     :[PreLoopVmcntCase.Basic_Load], \
      PreLoopVmcntCase.OptNLL_Store   :[PreLoopVmcntCase.Basic_Load, PreLoopVmcntCase.OptNLL_Store], \
      PreLoopVmcntCase.OrdNLL_E1_Store:[PreLoopVmcntCase.Basic_Load, PreLoopVmcntCase.OrdNLL_E1_Store] }
      # PreLoopVmcntCase.OrdNLL_B1_Store:[PreLoopVmcntCase.Basic_Load, PreLoopVmcntCase.OrdNLL_B1_Store] }

    self.useManualVmcnt = False
    self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined


    #######################################L
    # Available Memory Instructions
    ########################################

    # name, numAddresses, numOffsets, offsetMultiplier, blockWidth, formatting):
    ########################################
    # Local Read
    _ds_load_b128 = MemoryInstruction("_ds_load_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_load2_b64 = MemoryInstruction("_ds_load2_b64",  1, 2, 2, 2, \
        "%s, %s offset0:%s, offset1:%s", kernel["MemoryModifierFormat"] )
    _ds_load_b64 = MemoryInstruction("_ds_load_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_load2_b32 = MemoryInstruction("_ds_load2_b32",  1, 2, 1, 1, \
        "%s, %s offset0:%s offset1:%s", kernel["MemoryModifierFormat"] )
    _ds_load_b32 = MemoryInstruction("_ds_load_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_load_u16 = MemoryInstruction("_ds_load_u16",    1, 1, 1, 0.5, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_load_u8 = MemoryInstruction("_ds_load_u8",      1, 1, 1, 0.25, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    ########################################
    # Local Write
    _ds_store_b128 = MemoryInstruction("_ds_store_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_store2_b64 = MemoryInstruction("_ds_store2_b64",  1, 2, 2, 2, \
        "%s, %s, %s offset0:%s, offset1:%s", kernel["MemoryModifierFormat"] )
    _ds_store_b64 = MemoryInstruction("_ds_store_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_store2_b32 = MemoryInstruction("_ds_store2_b32",  1, 2, 1, 1, \
        "%s, %s, %s offset0:%s offset1:%s", kernel["MemoryModifierFormat"] )
    _ds_store_b32 = MemoryInstruction("_ds_store_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_store_b16 = MemoryInstruction("_ds_store_b16",    1, 1, 1, 0.5, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    _ds_store_b8 = MemoryInstruction("_ds_store_b8",      1, 1, 1, 0.25, \
        "%s, %s offset:%s", kernel["MemoryModifierFormat"] )
    ########################################
    # Global Read
    _global_load_b128 = MemoryInstruction("_global_load_b128", 1, 0, 0, 4, \
        "UNUSED %s, %s, %s", kernel["MemoryModifierFormat"] )
    _global_load_b64 = MemoryInstruction("_global_load_b64",   1, 0, 0, 2, \
        "UNUSED %s, %s, %s", kernel["MemoryModifierFormat"] )
    _global_load_b32 = MemoryInstruction("_global_load_b32",   1, 0, 0, 1, \
        "UNUSED %s, %s, %s", kernel["MemoryModifierFormat"] )

    _buffer_load_b128 = MemoryInstruction("_buffer_load_b128", 1, 0, 0, 4, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s", kernel["MemoryModifierFormat"] )
    _buffer_load_b64 = MemoryInstruction("_buffer_load_b64", 1, 0, 0, 2, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s", kernel["MemoryModifierFormat"] )
    _buffer_load_b32 = MemoryInstruction("_buffer_load_b32", 1, 0, 0, 1, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s", kernel["MemoryModifierFormat"] )
    # generate half directly w/o using the format string to handle hi/lo correctly
    _buffer_load_d16_b16 = MemoryInstruction("_buffer_load_d16_b16", 1, 0, 0, 0.5, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s", kernel["MemoryModifierFormat"] )
    # generate byte directly w/o using the format string to handle hi/lo correctly
    _buffer_load_d16_u8 = MemoryInstruction("_buffer_load_d16_u8", 1, 0, 0, 0.25, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s", kernel["MemoryModifierFormat"] )

    self.buff_load_inst_offset_max = 4096

    ########################################
    # Global Write
    _global_store_b128 = MemoryInstruction("_global_store_b128", 1, 0, 0, 4, \
        "%s, %s, %s", kernel["MemoryModifierFormat"], kernel["ForceStoreSC1"] )
    _global_store_b64  = MemoryInstruction("_global_store_b64",  1, 0, 0, 2, \
        "%s, %s, %s", kernel["MemoryModifierFormat"], kernel["ForceStoreSC1"] )
    _global_store_b32  = MemoryInstruction("_global_store_b32",  1, 0, 0, 1, \
        "%s, %s, %s", kernel["MemoryModifierFormat"], kernel["ForceStoreSC1"] )

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
      chosen_load_b128 = _global_load_b128
      chosen_load_b64  = _global_load_b64
      chosen_load_b32  = _global_load_b32
      chosen_load_b16  = _global_load_b32 # not supported
      chosen_load_b8   = _global_load_b32 # not supported

    chosen_store_b128 = _global_store_b128
    chosen_store_b64  = _global_store_b64
    chosen_store_b32  = _global_store_b32


    self.memoryInstructions = {
          "GlobalRead" : [ chosen_load_b128, chosen_load_b64, chosen_load_b32,
                           chosen_load_b16, chosen_load_b8 ],
          "GlobalWrite": [ chosen_store_b128, chosen_store_b64, chosen_store_b32 ],
          "LocalRead"  : [ _ds_load_b128, _ds_load2_b64, _ds_load_b64,
                           _ds_load2_b32, _ds_load_b32, _ds_load_u16, _ds_load_u8 ],
          "LocalWrite" : [ _ds_store_b128, _ds_store2_b64, _ds_store_b64, _ds_store2_b32,
                           _ds_store_b32, _ds_store_b16, _ds_store_b8 ]
        }

    if self.asmCaps["v_fma_mix_f32"]:
      self.mixinst = "v_fma_mix_f32"
    elif self.asmCaps["v_mad_mix_f32"]:
      self.mixinst = "v_mad_mix_f32"
    else:
      self.mixinst = "NOT_SUPPORTED"

    self.overflowedResources = 0 # if true, comment out whole kernel

    self.kernelName = self.getKernelName(kernel)
    self.inTailLoop = False
    self.serializedStore = False
    self.codeAccVgprRead = None
    self.codeMulAlpha = None

    ##initializing code modules for StoreCInUnroll
    self.BetaOpTemplate = None
    self.AlphaOpTemplate = None
    self.LoadCTemplate = None
    self.StoreCTemplate = None
    self.accVgprTemplate = None

    if kernel["StoreCInUnroll"]:
      self.BetaOpTemplate = Code.OpTemplate("BetaOpCode")
      self.AlphaOpTemplate = Code.OpTemplate("AlphaOpCode")
      self.LoadCTemplate = Code.MemOpTemplate("LoadCCode")
      self.StoreCTemplate = Code.MemOpTemplate("StoreCCode")
      self.accVgprTemplate = Code.OpTemplate("accVgprCode")

    # condition(s) to allocate tile offset and unroll offset registers for PK kernel
    self.useGlobalReadTileVgpr = False
    if kernel["StoreCInUnroll"]:
      # limit to StoreCInUnroll case only
      self.useGlobalReadTileVgpr = True

    # registers per element
    self.bpr = 4 # all registers are 32bit

    # default setup
    # AB=DataType / Cexternal=DestDataType / Cinternal=Accumulation (MAC or MFMA)
    self.bpeAB = int(self.bpr * kernel["ProblemType"]["DataType"].numRegisters())

    # Cexternal = the "current" kernel output type,
    # - default: the "current" kernel is a non-GSU-kernel,
    #     Cexternal (= DestDataType) and is the final gemm result
    #
    # - For GSU: the "current" kernel is a GSU-kernel,
    #     this kernel returns a temp buffer with same type as Cinternal.
    #     Later, another kernel will accumulate this buffer
    #     and convert the final result to Cexternal (= DestDataType) as the gemm result
    self.bpeCexternal = int(self.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())

    # already covers: dgemm, cgemm, zgemm, sgemm
    #               : hgemm  + !HPA ([H/H/H] compute = internal = f16)
    #               : hgemm  +  HPA ([H/H/S] or [H/S/S] compute = internal = f32)
    #               : bfgemm +  HPA ([B/B/S] or [H/S/S] compute = internal = f32)
    #               : int8x4-gemm   (internal = i32)
    self.bpeCinternal = int(self.bpr * kernel["ProblemType"]["ComputeDataType"].numRegisters())

    if kernel["_GlobalAccumulation"] and kernel["_GlobalAccumulation"] != 'PartialsBuffer':
          self.bpeCexternal = self.bpeCinternal

    # special case for wmma h and b
    if (kernel["EnableMatrixInstruction"]
            and globalParameters["AsmCaps"][self.version]["HasWMMA"]
            and (kernel["ProblemType"]["ComputeDataType"].numRegisters() == 0.5)):
        self.bpeCinternal = 4
        if kernel["_GlobalAccumulation"]: # TODO SK and kernel["_GlobalAccumulation"] != 'PartialsBuffer':
            self.bpeCexternal = 2

    #jgolds Need to check device for support
    # HPA not allowed in dgemm, cgemm, zgemm, sgemm
    if kernel["ProblemType"]["HighPrecisionAccumulate"] and \
       not (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16() or \
          kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isInt8() or \
          kernel["ProblemType"]["DataType"].is8bitFloat()):
        print("HighPrecisionAccumulate only valid when DataType is half, bf16, Int8x4, Int8, fp8, bf8. Forcing HPA to False")
        kernel["ProblemType"]["HighPrecisionAccumulate"] = False

    assert self.bpeAB == tPA["bpe"]
    assert self.bpeAB == tPB["bpe"]

    ####################################
    # choose memory instructions
    ####################################

    ########################################
    # globalReadA instruction; no flat_load2_*
    self.globalReadWidthA = float(tPA["nrcv"]*tPA["bpe"])/self.bpr
    self.globalRead2CoalescedA = kernel["NumLoadsCoalescedA"]>1 \
        or self.readCoalescedComponentsA
    self.globalRead2PerpendicularA = kernel["NumLoadsPerpendicularA"] > 1 \
        or self.readPerpendicularComponentsA
    self.globalReadInstructionIdxA = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthA, \
        kernel["GlobalRead2A"], \
        self.globalRead2CoalescedA, self.globalRead2PerpendicularA, [] )
    ########################################
    # globalReadB instruction; no flat_load2_
    self.globalReadWidthB = float(tPB["nrcv"]*tPB["bpe"])/self.bpr
    self.globalRead2CoalescedB = kernel["NumLoadsCoalescedB"]>1 \
        or self.readCoalescedComponentsB
    self.globalRead2PerpendicularB = kernel["NumLoadsPerpendicularB"] > 1 \
        or self.readPerpendicularComponentsB
    self.globalReadInstructionIdxB = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthB, \
        kernel["GlobalRead2B"], \
        self.globalRead2CoalescedB, self.globalRead2PerpendicularB, [] )

    ########################################
    # localWriteA instruction
    # for local, tile->para, unroll->perp
    #self.localWriteWidthA = 1 if (self.writeTileDimComponentsA \
    #    or self.writeUnrollDimComponentsA) else kernel["VectorWidth"]
    self.localWriteWidthA = int(tPA["nwcv"]*tPA["bpe"])//self.bpr
    if self.localWriteWidthA < 1:
      self.localWriteWidthA = (1.0*tPA["nwcv"]*tPA["bpe"])/self.bpr
    self.localWrite2CoalescedA = tPA["nrc"]>1 \
        or self.writeTileDimComponentsA
    self.localWrite2PerpendicularA = tPA["nrp"]>1 \
        or self.writeUnrollDimComponentsA
    # localWriteA stride tile
    if kernel["ProblemType"]["TLUA"]:
      if self.writeTileDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSCA"]
        self.localWriteJoinTileA = "Coalesced"
    else:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSPA"]
        self.localWriteJoinTileA = "Perpendicular"
    self.localWriteStrideTileA = self.localWriteStrideTileA*tPA["bpe"]//self.bpr
    # localWriteA stride unroll
    if kernel["ProblemType"]["TLUA"]:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Perpendicular"
    else:
      if self.writeTileDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Coalesced"
    self.localWriteStrideUnrollA = \
        (self.localWriteStrideUnrollA*tPA["bpe"])//self.bpr
    self.localWriteInstructionIdxA = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthA, \
        kernel["LocalWrite2A"], \
        self.localWrite2CoalescedA, self.localWrite2PerpendicularA,
        [self.localWriteStrideTileA, self.localWriteStrideUnrollA] )

    ########################################
    # localWriteB instruction
    # for local, tile->para, unroll->perp
    #self.localWriteWidthB = 1 if (self.writeTileDimComponentsB \
    #    or self.writeUnrollDimComponentsB) else kernel["VectorWidth"]
    self.localWriteWidthB = int(tPB["nwcv"]*tPB["bpe"])//self.bpr
    if self.localWriteWidthB < 1:
      self.localWriteWidthB = (1.0*tPB["nwcv"]*tPB["bpe"])/self.bpr
    self.localWrite2CoalescedB = tPB["nrc"]>1 \
        or self.writeTileDimComponentsB
    self.localWrite2PerpendicularB = tPB["nrp"]>1 \
        or self.writeUnrollDimComponentsB
    # localWriteB stride tile
    if kernel["ProblemType"]["TLUB"]:
      if self.writeTileDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSCB"]
        self.localWriteJoinTileB = "Coalesced"
    else:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSPB"]
        self.localWriteJoinTileB = "Perpendicular"
    self.localWriteStrideTileB = (self.localWriteStrideTileB*tPB["bpe"])//self.bpr
    # localWriteB stride unroll
    if kernel["ProblemType"]["TLUB"]:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Perpendicular"
    else:
      if self.writeTileDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Coalesced"
    self.localWriteStrideUnrollB = \
        (self.localWriteStrideUnrollB*tPB["bpe"])//self.bpr
    self.localWriteInstructionIdxB = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthB, \
        kernel["LocalWrite2B"], \
        self.localWrite2CoalescedB, self.localWrite2PerpendicularB,
        [self.localWriteStrideTileB, self.localWriteStrideUnrollB] )

    ########################################
    # localRead A
    localReadWidth = (kernel["VectorWidthA"] * tPA["bpe"]) // self.bpr
    if kernel["EnableMatrixInstruction"]:
      localReadWidth = self.lrvwTileA * tPA["bpe"] / self.bpr
    if kernel["UnrollMajorLDSA"] or kernel["DirectToVgprA"]:
      # DTVA case, no local read is actually generated, but this still affects (dummy) local read scheduling for DTVA
      localReadWidth = (self.lrvwA * tPA["bpe"]) // self.bpr

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedA = \
        kernel["ThreadTile0"] * tPA["bpe"]//self.bpr
    self.localRead2CoalescedA = kernel["ThreadTile0"]//kernel["VectorWidthA"] > 1
    self.localReadInstructionIdxA = \
        self.selectMemoryInstruction("LocalRead", localReadWidth, \
        kernel["LocalRead2A"], \
        self.localRead2CoalescedA, localRead2Perpendicular,
        [self.localReadStrideCoalescedA] )
    tPA["localReadSwapByteOffset"] = 0
    tPB["localReadSwapByteOffset"] = 0
    tPA["localWriteSwapByteOffset"] = 0
    tPB["localWriteSwapByteOffset"] = 0

    ########################################
    # localRead B
    localReadWidth = (kernel["VectorWidthB"] * tPB["bpe"]) // self.bpr
    if kernel["EnableMatrixInstruction"]:
      localReadWidth = self.lrvwTileB * tPB["bpe"] / self.bpr
    if kernel["UnrollMajorLDSB"] or kernel["DirectToVgprB"] or (kernel["DirectToVgprA"] and self.useWiderLocalReadB):
      # DTVB case, no local read is actually generated, but this still affects (dummy) local read scheduling for DTVB
      localReadWidth = (self.lrvwB * tPB["bpe"]) // self.bpr

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedB = \
    kernel["ThreadTile1"] * tPB["bpe"]//self.bpr
    self.localRead2CoalescedB = kernel["ThreadTile1"]//kernel["VectorWidthB"] > 1
    self.localReadInstructionIdxB = \
        self.selectMemoryInstruction("LocalRead", localReadWidth, \
        kernel["LocalRead2B"], \
        self.localRead2CoalescedB, localRead2Perpendicular,
        [self.localReadStrideCoalescedB] )

    instructions = self.memoryInstructions
    self.globalReadInstructionA = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxA]
    self.globalReadInstructionB = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxB]
    self.localWriteInstructionA = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxA]
    self.localWriteInstructionB = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxB]
    self.localReadInstructionA = instructions["LocalRead"][ \
        self.localReadInstructionIdxA]
    self.localReadInstructionB = instructions["LocalRead"][ \
        self.localReadInstructionIdxB]
    # global reads per instruction
    tPA["nrcvpi"] = ((self.globalReadInstructionA.totalWidth*self.bpr)/tPA["bpe"])
    if tPA["nrcvpi"] >= 1:
      # convert to int if >= 1
      tPA["nrcvpi"] = int(tPA["nrcvpi"])
    tPB["nrcvpi"] = ((self.globalReadInstructionB.totalWidth*self.bpr)/tPB["bpe"])
    if tPB["nrcvpi"] >= 1:
      # convert to int if >= 1
      tPB["nrcvpi"] = int(tPB["nrcvpi"])
    tPA["nwcvpi"] = int((self.localWriteInstructionA.totalWidth*self.bpr)/tPA["bpe"])
    tPB["nwcvpi"] = int((self.localWriteInstructionB.totalWidth*self.bpr)/tPB["bpe"])
    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
    #jgolds bpeCinternal because we are allocating accumulation registers here
    self.numVgprValuC = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.bpeCinternal)//self.bpr

    PLR = kernel["PrefetchLocalRead"] if kernel["PrefetchLocalRead"] < kernel["LoopIters"] else kernel["LoopIters"] - 1

    PLRplus1A = (1+PLR) if not self.packDTVA else kernel["LoopIters"]
    PLRplus1B = (1+PLR) if not self.packDTVB else kernel["LoopIters"]
    valuBlocksA = PLRplus1A * kernel["InnerUnroll"]
    valuBlocksB = PLRplus1B * kernel["InnerUnroll"]
    if kernel["EnableMatrixInstruction"]:
      self.numVgprValuAPerBlock = kernel["MIWaveTileA"] * kernel["MIInputPerThread"] * tPA["bpe"] // self.bpr
      self.numVgprValuBPerBlock = kernel["MIWaveTileB"] * kernel["MIInputPerThread"] * tPB["bpe"] // self.bpr
    else:
      self.numVgprValuAPerBlock = kernel["ThreadTileA"] * tPA["bpe"] // self.bpr
      self.numVgprValuBPerBlock = kernel["ThreadTileB"] * tPB["bpe"] // self.bpr
      if kernel["ProblemType"]["DataType"].isBFloat16():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          self.numVgprValuAPerBlock = kernel["ThreadTileA"]
          self.numVgprValuBPerBlock = kernel["ThreadTileB"]
      elif kernel["ProblemType"]["DataType"].isInt8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          if kernel["LocalDotLayout"] == 1:
            self.numVgprValuAPerBlock = kernel["ThreadTileA"]
            self.numVgprValuBPerBlock = kernel["ThreadTileB"]
      elif kernel["ProblemType"]["DataType"].is8bitFloat(): # F8 is always HPA
        if kernel["LocalDotLayout"] == 1:
          self.numVgprValuAPerBlock = kernel["ThreadTileA"]
          self.numVgprValuBPerBlock = kernel["ThreadTileB"]

    # change numVgprValuAPerBlock to 0 for A if DirectToVgpr is enabled
    if kernel["DirectToVgprA"] and not self.packDTVA:
      self.numVgprValuAPerBlock = 0
    self.numVgprValuA = self.numVgprValuAPerBlock * valuBlocksA
    # change numVgprValuBPerBlock to 0 for B if DirectToVgpr is enabled
    if kernel["DirectToVgprB"] and not self.packDTVB:
      self.numVgprValuBPerBlock = 0
    self.numVgprValuB = self.numVgprValuBPerBlock * valuBlocksB

    if self.archCaps["HasEccHalf"]:
        self.needPackA = tPA["bpe"] < self.bpr and (not kernel["UnrollMajorLDSA"]) and kernel["EnableMatrixInstruction"]
        self.needPackB = tPB["bpe"] < self.bpr and (not kernel["UnrollMajorLDSB"]) and kernel["EnableMatrixInstruction"]
    else:
        self.needPackA = self.localReadInstructionA.blockWidth == 0.25 and kernel["EnableMatrixInstruction"]
        self.needPackB = self.localReadInstructionB.blockWidth == 0.25 and kernel["EnableMatrixInstruction"]

    ####################################
    # num vgprs: valuPack
    self.numVgprValuPackA = 0
    self.numElemPerBprA = (4 / tPA["bpe"])
    needVgprForPack = self.needPackA and kernel["VgprForLocalReadPacking"]
    if needVgprForPack:
      if self.lrvwTileA > 1:
        self.numVgprValuA //= PLRplus1A
        numLoadPerReg = max(1, int(self.numElemPerBprA)//self.lrvwTileA)
        valuBlocksPack = PLRplus1A * numLoadPerReg
      else:
         valuBlocksPack = (int(self.numElemPerBprA) - 1)
      if self.packDTVA:
        # pack DTV case, double the number of blocks
        valuBlocksPack *= 2
      self.numVgprValuPackA = self.numVgprValuA * valuBlocksPack

    self.numVgprValuPackB =0
    self.numElemPerBprB = (4 / tPB["bpe"])
    needVgprForPack = self.needPackB and kernel["VgprForLocalReadPacking"]
    if needVgprForPack:
      if self.lrvwTileB > 1:
        self.numVgprValuB //= PLRplus1B
        numLoadPerReg = max(1, int(self.numElemPerBprB)//self.lrvwTileB)
        valuBlocksPack = PLRplus1B * numLoadPerReg
      else:
        valuBlocksPack = (int(self.numElemPerBprB) - 1)
      if self.packDTVB:
        # pack DTV case, double the number of blocks
        valuBlocksPack *= 2
      self.numVgprValuPackB = self.numVgprValuB * valuBlocksPack

    ####################################
    # num vgprs: global -> local elements
    self.numVgprG2LA = 0
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      self.numVgprG2LA = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] * \
        kernel["GlobalLoadVectorWidthA"] * tPA["bpe"]) / (float)(self.bpr))
    # using _ds_store_b8: need one more vgpr space to do lshr
    if self.localWriteInstructionA.blockWidth == 0.25:
      self.numVgprG2LA = self.numVgprG2LA * 2
    # double numVgprG2LA if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.numVgprG2LA = self.numVgprG2LA * 2

    self.numVgprG2LB = 0
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      self.numVgprG2LB = roundUp((kernel["NumLoadsCoalescedB"] * kernel["NumLoadsPerpendicularB"] * \
        kernel["GlobalLoadVectorWidthB"] * tPB["bpe"]) / (float)(self.bpr))
    # using _ds_store_b8: need one more vgpr space to do lshr
    if self.localWriteInstructionB.blockWidth == 0.25:
      self.numVgprG2LB = self.numVgprG2LB * 2
    # double numVgprG2LB if DirectToVgpr is enabled
    if kernel["DirectToVgprB"]:
      self.numVgprG2LB = self.numVgprG2LB * 2

    ####################################
    # num vgprs: local read addresses
    self.numVgprLocalReadAddressesA = 1 * self.rpla
    self.numVgprLocalReadAddressesB = 1 * self.rpla
    # do not allocate local read address register if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.numVgprLocalReadAddressesA = 0
    if kernel["DirectToVgprB"]:
      self.numVgprLocalReadAddressesB = 0

    ####################################
    # num vgprs: local write addresses
    #numLocalWritesA = kernel["NumLoadsCoalescedA"] \
    #    * nlp * self.numWriteVectorComponentsA
    #numLocalWriteInstructionsA = numLocalWritesA \
    #    / self.localWriteInstructionA[self.instructionIdxNumOffsets]
    self.numVgprLocalWriteAddressesA = 0 if kernel["LocalWriteUseSgprA"] else 1 * self.rpla
    # TODO - if we only have one local write - can just map the overhang register to the LWO
    if kernel["FractionalLoad"]==1 and kernel["fractionalPerpOverhangA"]:
      self.numVgprLocalWriteAddressesA += 1*self.rpla

    #numLocalWritesB = kernel["NumLoadsCoalescedB"] \
    #    * nlp * self.numWriteVectorComponentsB
    #numLocalWriteInstructionsB = numLocalWritesB \
    #    / self.localWriteInstructionB[self.instructionIdxNumOffsets]
    self.numVgprLocalWriteAddressesB = 0 if kernel["LocalWriteUseSgprB"] else 1 * self.rpla
    if kernel["FractionalLoad"]==1 and kernel["fractionalPerpOverhangB"]:
      self.numVgprLocalWriteAddressesB += 1*self.rpla

    # do not allocate local write address register if DirectToVgpr is enabled
    if kernel["DirectToVgprA"]:
      self.numVgprLocalWriteAddressesA = 0
    if kernel["DirectToVgprB"]:
      self.numVgprLocalWriteAddressesB = 0

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] \
        * self.numReadVectorComponentsA
    numGlobalReadInstructionsA = (numGlobalReadsA * tPA["bpe"])//\
        (self.globalReadInstructionA.blockWidth * 4)

    if kernel["BufferLoad"]:
      self.numGlobalReadOffsetsA = roundUp(numGlobalReadInstructionsA * self.rpgo)
    else:
      numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = (numGlobalReadsB * tPB["bpe"])// \
        (self.globalReadInstructionB.blockWidth * 4)
    if kernel["BufferLoad"]:
      self.numGlobalReadOffsetsB = roundUp(numGlobalReadInstructionsB * self.rpgo)
    else:
      numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.rpga
    if self.globalReadIncsUseVgpr:
      numVgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
      numVgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
    else:
      numVgprGlobalReadIncsA = 0
      numVgprGlobalReadIncsB = 0

    numVgprAddressDbg = self.rpga if globalParameters["DebugKernel"] else 0

    if kernel["ProblemType"]["Fp16AltImpl"]:
      self.numG2LpipeRegisters = 2
      if kernel["ProblemType"]["Fp16AltImplRound"]:
        self.numG2LpipeRegisters += 1

    ####################################
    # num vgprs: c write address
    # 1 address where to write first value
    # 1 tmp address where to write current value

    ####################################
    # VGPR Assignment
    ####################################
    vgprIdx = 0
    self.totalAgprs = 0
    self.startVgprValuC = vgprIdx; vgprIdx += self.numVgprValuC

    if kernel["EnableMatrixInstruction"]:
      # MI kernels can overlap C-tile w/ AB-tile up until writeback. Illustrated below:
      # |<-------------- valuC -------------->|
      # |------------|-----------|xx|---------|
      #   lastValuAB ^           ^  ^         ^
      #         lastVgprForReads ^  ^         ^
      #              startVgprReuse ^         ^
      #                             lastValuC ^
      # TODO a bit tricky. Better to manage all GPRs solely through RegisterPool
      # enable serializedStore only for EnableMatrixInstruction enabled case
      if kernel["EnableMatrixInstructionStore"]:
        self.serializedStore = True

      ########################################
      # AGPR Allocation
      ########################################
      if not kernel["MIArchVgpr"] and not kernel["StoreCInUnroll"]:
        self.totalAgprs = self.numVgprValuC
        # Reset vgprIdx. In case of LSU, need to keep C reg for reduction
        if kernel["LocalSplitU"]==1:
          vgprIdx = 0
          self.numVgprValuC = 0

      self.startaccValuC0 = None
      self.startaccValuC1 = None
      if kernel["StoreCInUnroll"]:
        self.startaccValuC0 = 0
        self.startaccValuC1 = self.numVgprValuC
        self.totalAgprs = 2*self.numVgprValuC  ## TODO fix for MT>128
        vgprIdx = 0
        self.numVgprValuC = 0

    # TODO: alignment hack, figure out a better solution
    vgprIdx = ((vgprIdx+1)//2)*2
    # Avoid bank conflict between VgprA and VgprC
    if self.archCaps["VgprBank"]:
      vgprIdx += 2
    self.startVgprValuA = vgprIdx; vgprIdx += self.numVgprValuA
    self.startVgprValuPackA = vgprIdx
    vgprIdx += self.numVgprValuPackA
    self.startVgprG2LA = None
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      if self.packDTVA:
        # DirectToVgpr + packing
        # overlap G2LA and ValuPackA
        self.startVgprG2LA = self.startVgprValuPackA
      # if PGR = True, PAP could be possibly enabled, we move G2LA later to prevent it from being reclaimed
      # otherwise, put G2L here since it can overlap valu
      elif not kernel["PrefetchGlobalRead"] and not kernel.enabledSplitLDS: # g2l can overlap valu
        self.startVgprG2LA = self.startVgprValuA
        vgprIdx = self.startVgprValuA \
            + max(self.numVgprValuA + self.numVgprValuPackA, self.numVgprG2LA)

    # TODO: alignment hack, figure out a better solution
    vgprIdx = ((vgprIdx+1)//2)*2
    if self.archCaps["VgprBank"]:
        vgprIdx += 1
    self.startVgprValuB = vgprIdx; vgprIdx += self.numVgprValuB
    self.startVgprValuPackB = vgprIdx
    vgprIdx += self.numVgprValuPackB
    self.startVgprG2LB = None
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      if self.packDTVB:
        # DirectToVgpr + packing
        # overlap G2LB and ValuPackB
        self.startVgprG2LB = self.startVgprValuPackB
      # if PGR = True, PAP could be possibly enabled, we move G2LB later to prevent it from being reclaimed
      # otherwise, put G2L here since it can overlap valu
      elif not kernel["PrefetchGlobalRead"] and not kernel.enabledSplitLDS: # g2l can overlap valu
        self.startVgprG2LB = self.startVgprValuB
        vgprIdx = self.startVgprValuB \
            + max(self.numVgprValuB + self.numVgprValuPackB, self.numVgprG2LB)

    # Registers allocated above this point can be used as temps during setup
    # Registers above here are reserved in initC, near the end of the setup
    # code
    self.lastValuAB = vgprIdx
    #----------------------------------
    # Point at last VGPR that can be reclaimed for use in the summation loop
    # If more VGPRs are added here be aware of the register reclaim code in
    # endSummation - registers that should be preserved after lastVgprForReads
    #
    # For PAP: decide the reclaim case
    # if we're not doing PAP, then the GlobalRead, LocalWrite, LocalRead, VgprG2L can be reclaimed
    # (and we'll extend the "lastVgprForReads" value later)
    # otherwise if we have PAP, they can't be reclaimed so we simply use the current vgprIdx
    self.lastVgprForReads = vgprIdx
    #----------------------------------

    if not kernel["LocalWriteUseSgprA"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesA = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesA = vgprIdx
        vgprIdx += self.numVgprLocalWriteAddressesA

    if not kernel["LocalWriteUseSgprB"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesB = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesB = vgprIdx
        vgprIdx += self.numVgprLocalWriteAddressesB

    # BufferLoad:
    # Uses a resource descriptor (SRD) which is stored in 4 SGPRs and thus shared by all work-items.
    # Each work-item also uses  a unique 32-bit offset into vgprGlobalReadOffset.  These offsets are set when
    # the tile is initialized and stay constant through the execution of the kernel.
    # The base address in the SRD is updated when the algorithm moves to a new tile
    # BufferLoad disables the gptGlobalReadAddr used in global addressing.
    if kernel["BufferLoad"]:
      self.startVgprGlobalReadOffsetA = vgprIdx
      vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.numGlobalReadOffsetsA
      self.startVgprGlobalReadOffsetB = vgprIdx
      vgprIdx += 1 if kernel["_UseSgprForGRO"] else self.numGlobalReadOffsetsB
      # allocate tile offset and unroll offset registers for PK kernel
      if self.useGlobalReadTileVgpr:
        self.startVgprGlobalReadTileOffsetA = vgprIdx
        vgprIdx += tPA["nrt"]
        self.startVgprGlobalReadUnrollOffsetA = vgprIdx
        vgprIdx += self.numGlobalReadOffsetsA
        self.startVgprGlobalReadTileOffsetB = vgprIdx
        vgprIdx += tPB["nrt"]
        self.startVgprGlobalReadUnrollOffsetB = vgprIdx
        vgprIdx += self.numGlobalReadOffsetsB

    else:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.startVgprGlobalReadAddressesA = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesA
      self.startVgprGlobalReadAddressesB = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesB

    self.zeroPadRegs={}
    self.zeroPadRegs['A'] = collections.OrderedDict()
    self.zeroPadRegs['B'] = collections.OrderedDict()
    for (tc,tP) in (('A',self.tPA),('B',self.tPB)):
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
              for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
                (freeDim, sumDim) = zp[:2]
                freeDimChar = globalParameters["IndexChars"][freeDim]
                sumDimChar  = globalParameters["IndexChars"][sumDim]
                zpName = "GlobalReadOffset%s_ZP%s%s_%d_%d_%d_%d" % \
                          (tc, freeDimChar, sumDimChar, para, sPara, perp, sPerp)

                assert (zpName not in self.zeroPadRegs[tc])
                self.zeroPadRegs[tc][zpName] = ZeroPadReg(zp, zpName, vgprIdx, \
                                                        perp, sPerp, para, sPara)
                vgprIdx += 1

    self.startVgprGlobalReadIncsA = vgprIdx
    vgprIdx += numVgprGlobalReadIncsA
    self.startVgprGlobalReadIncsB = vgprIdx
    vgprIdx += numVgprGlobalReadIncsB
    #-----------

    if self.startVgprG2LA is None:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.startVgprG2LA = vgprIdx; vgprIdx += self.numVgprG2LA

    if self.startVgprG2LB is None:
      # TODO: alignment hack, figure out a better solution
      vgprIdx = ((vgprIdx+1)//2)*2
      self.startVgprG2LB = vgprIdx; vgprIdx += self.numVgprG2LB

    # Check if PAP or not,
    # if not PAP GlobalRead, LocalWrite, LocalRead, G2L can be reclaimed, extend the "lastVgprForReads" value
    if not self.prefetchAcrossPersistent:
      self.lastVgprForReads = vgprIdx
    #-----------

    self.startVgprLocalReadAddressesA = vgprIdx
    vgprIdx += self.numVgprLocalReadAddressesA
    if self.combineLocalAddresses:
      self.startVgprLocalReadAddressesB = self.startVgprLocalReadAddressesA
    else:
      self.startVgprLocalReadAddressesB = vgprIdx
      vgprIdx += self.numVgprLocalReadAddressesB

    if kernel["ProblemType"]["Fp16AltImpl"]:
      self.G2Lpipe0 = vgprIdx
      self.G2Lpipe1 = self.G2Lpipe0 + 1
      vgprIdx += 2
      if kernel["ProblemType"]["Fp16AltImplRound"]:
        self.Fp16AltTmp = self.G2Lpipe1 + 1
        vgprIdx += 1

    # packTmp vgpr for wider local read + v_perm + 8bit
    lrvwTileMax = max(self.lrvwTileA, self.lrvwTileB)
    if lrvwTileMax > 1 and (kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()):
      self.vgprPackTemp = vgprIdx
      vgprIdx += 1

    self.startVgprAddressDbg = vgprIdx
    vgprIdx += numVgprAddressDbg

    # allocate VGPRS for loadC and storeC (dedicated for now)
    self.startVgprG2LC = None
    self.startVgprL2GC = None
    self.GlobalReadOffsetC = None
    self.GlobalWriteOffsetD = None
    if kernel["StoreCInUnroll"]:
       # TODO: alignment hack, figure out a better solution
      Calign = 2
      if kernel["StoreVectorWidth"] == 2:
        Calign = 4
      # need proper alignment for G2LC
      vgprIdx = ((vgprIdx+Calign - 1)//Calign)*Calign
      self.startVgprG2LC = vgprIdx
      vgprIdx += (kernel["VectorWidthA"] * tPA["bpe"]) // self.bpr
      self.startVgprL2GC = vgprIdx
      vgprIdx += ((kernel["VectorWidthA"] * tPA["bpe"]) // self.bpr)
      if not kernel["AtomicAddC"] and kernel["ProblemType"]["UseBeta"]:
        self.GlobalReadOffsetC = vgprIdx
        vgprIdx +=1
        if not kernel["StoreCInUnrollExact"]:
          self.GlobalReadOffsetCBackup = vgprIdx
          vgprIdx +=1
      self.GlobalWriteOffsetD = vgprIdx
      vgprIdx +=1
      if not kernel["StoreCInUnrollExact"]:
        self.GlobalWriteOffsetDBackup = vgprIdx
        vgprIdx +=1
      self.GlobalBufferOOB = vgprIdx
      vgprIdx +=1
    # for cgemm/zgemm + SCIU case, allocate 4 vgpr for alpha calculation (cannot use tmp vgpr in unroll loop or write batch)
    if kernel["ProblemType"]["DataType"].isComplex() \
        and kernel["StoreCInUnroll"] \
        and (kernel["GlobalSplitU"] == 1 or kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
      # need proper alignment
      vgprIdx = ((vgprIdx+2 - 1)//2)*2
      self.startVgprAlphaTmp = vgprIdx
      vgprIdx += kernel["ProblemType"]["DataType"].numRegisters();

    self.startVgprSerial = vgprIdx
    vgprIdx += 1 # for vgpr serial id

    # tmp vgprs
    #minVgprTmp = 1
    #if kernel["LoopTail"]:
    #  minVgprTmp += 4
    #if globalParameters["DebugKernel"]:
    #  minVgprTmp += 2
    #vgprIdx += minVgprTmp
    #tPrint(3, "%3u vgprs <- %s" % (vgprIdx, self.kernelName) )
    self.startVgprReuse = vgprIdx # for register reuse;

    self.totalVgprs = max(vgprIdx, self.numVgprValuC)
    if self.totalVgprs < kernel["MinVgprNumber"] or self.totalVgprs > kernel["MaxVgprNumber"]:
      raise RuntimeError("Generating asm kernel error: total vgpr: %u not in [%u, %u]. kernel=%s\n" % (self.totalVgprs, kernel["MinVgprNumber"], kernel["MaxVgprNumber"], self.kernelName))

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    self.sgprPool = RegisterPool(0, 's', defaultPreventOverflow=True, printRP=0)
    numSgprAddressD = self.rpga # til end
    numSgprAddressC = self.rpga # til end
    numSgprAddressA = self.rpga # til read offsets
    numSgprAddressB = self.rpga # til read offsets
    numSgprAddressWS = self.rpga
    numSgprAddressFlags = self.rpga
    # would not less than 1 reg,
    # since even if ComputeType = H, we still pass the arg as a 32-bit (concate two 16-bit)
    numSgprAlpha = max(1,int(self.bpeCinternal/4))
    numSgprBeta  = max(1,int(self.bpeCinternal/4)) if kernel["ProblemType"]["UseBeta"] else 0
    self.numSgprStridesD = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesC = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    self.numSgprStridesB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    if not kernel["ProblemType"]["UseInitialStridesCD"]:
      self.numSgprStridesD -= 1
      self.numSgprStridesC -= 1
    if not kernel["ProblemType"]["UseInitialStridesAB"]:
      self.numSgprStridesA -= 1
      self.numSgprStridesB -= 1
    self.numSgprSizesSum = kernel["ProblemType"]["NumIndicesSummation"]
    self.numSgprSizesFree = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprOffsetD = 0
    self.numSgprOffsetC = 0
    self.numSgprOffsetA = 0
    self.numSgprOffsetB = 0
    self.numSgprAddressDbg = self.rpga if globalParameters["DebugKernel"] else 0

    ####################################
    # num sgprs: global read increments
    if self.globalReadIncsUseVgpr:
      self.numSgprGlobalReadIncsA = 0
      self.numSgprGlobalReadIncsB = 0
    else:
      self.numSgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] * self.rpgo
      self.numSgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] * self.rpgo

    ########################################
    # SGPR Assignment according to AMDGPU-ABI
    ########################################
    self.defineSgpr("KernArgAddress", self.rpga)
    assert(self.sgprs["KernArgAddress"] ==  0) # kernarg is passed to kernel as SGPR0
    if self.kernel["PreloadKernelArguments"] == 1:
      self.definePreloadedSgprPool(14)

    #if kernel["WorkGroupMapping"]>=0 :
    #  self.defineSgpr("WorkGroup0", 1)
    #  self.defineSgpr("WorkGroup1", 1)
    #else:
    #  self.defineSgpr("WorkGroup1", 1)
    #  self.defineSgpr("WorkGroup0", 1)
    self.defineSgpr("WorkGroup0", 1)
    self.defineSgpr("WorkGroup1", 1)

    wg=2

    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        self.defineSgpr("WorkGroup%u"%wg, 1)
        wg+=1

    # SGPR above are user SGPR which are set by GPU hardware when the kernel is launched
    self.firstInitSgpr = self.sgprPool.size()

    # To avoid corrupting tmp sgprs that may be used around the assert,
    # reserve some sgprs to save/restore the execmask
    if self.db["EnableAsserts"]:
      self.defineSgpr("SaveExecMask", 2, 2)

    self.defineSgpr("GSUSumIdx", 2 if kernel["GlobalSplitU"] > 1 else 0)

    self.sumMagicParms = []
    if kernel["PackSummationDims"]:
      self.magicSumChars = [globalParameters["IndexChars"][c] for c in kernel["ProblemType"]["IndicesSummation"][1:]]

      self.sumMagicParms=["%s"%idxChar for idxChar in self.magicSumChars]
      if kernel["PackSummationDims"] and kernel["GlobalSplitU"] > 1 and self.sumMagicParms:
          self.sumMagicParms.append("%s_GsuRemainder"%self.unrollChar)

      for magicName in self.sumMagicParms:
        if kernel["MagicDivAlg"]==2:
          self.defineSgpr("MagicAbitSize%s"%magicName, 1)

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

    if kernel["PackSummationDims"]:
      self.defineSgpr(self.loopCounterName(kernel,self.unrollIdx), 1)
    else:
      # contractions with multiple summations will use multiple LoopCounters, if PSD=0
      for i in range(kernel["ProblemType"]["NumIndicesSummation"]):
        self.defineSgpr(self.loopCounterName(kernel,i), 1)

    self.defineSgpr("OrigLoopCounter", 1)

    if self.prefetchAcrossPersistent0 and (not self.noTailLoop):
      #if kernel["ExpandPointerSwap"]:
        # For ExpandPointerSwap + PAP, track which expanded loop iter to start on
        # global prefetches bounce between two LDS buffers, and the bounce state
        # must be maintained across PK boundaries.
        # If the no-load-loop is present it counts as one iteration and
        # So if K is even multiple of unroll then we exit at odd iteration
        # and each PK loop will start on the second expanded pointer swap
        # TODO- We use a temp Sgpr to track this?
        #self.defineSgpr("BreakAtEvenIter", 1)  # exit loop at LoopCopy2 (finish all EPS loops)
      self.defineSgpr("TailLoopCounter", 1)
    if globalParameters["DebugKernel"]:
      self.defineSgpr("AddressDbg", self.numSgprAddressDbg)
      self.defineSgpr("DebugKernelItems", 1)

    if kernel["BufferLoad"]:
       # resource descriptor (SRD) A and B, must be aligned on 4-SGPR boundary
      self.defineSgpr("SrdA", 4, 4)
      self.defineSgpr("SrdB", 4, 4)
    if kernel["BufferStore"]:
      self.defineSgpr("SrdD", 4, 4)
      self.defineSgpr("SrdC", 4, 4)

    ###################################
    # Get kernel argument start here
    self.defineSgpr("Tensor2dSizeA", 2, (2 if kernel["PreloadKernelArguments"] == 1 else 4), preload=True)
    self.defineSgpr("Tensor2dSizeB", 2, 2, preload=True)
    self.argAddressOffset = 6 * 4 # 8 bytes C, A, B

    self.defineSgpr("AddressD", numSgprAddressD, numSgprAddressD, kernarg=True)
    self.defineSgpr("AddressC", numSgprAddressC, numSgprAddressC, kernarg=True)
    self.defineSgpr("AddressA", numSgprAddressA, numSgprAddressA, preload=True)
    self.defineSgpr("AddressB", numSgprAddressB, numSgprAddressB, preload=True)
    if kernel["StreamK"] > 0 and kernel["StreamKAtomic"] == 0:
      self.defineSgpr("AddressWS", numSgprAddressWS, kernarg=True)
      self.defineSgpr("AddressFlags", numSgprAddressFlags, kernarg=True)

    if not kernel["ProblemType"]["StridedBatched"]:
      self.numSgprOffsetD = 2
      self.numSgprOffsetC = 2
      self.numSgprOffsetA = 2
      self.numSgprOffsetB = 2
      self.defineSgpr("OffsetD", self.numSgprOffsetD, kernarg=True)
      self.defineSgpr("OffsetC", self.numSgprOffsetC, kernarg=True)
      self.defineSgpr("OffsetA", self.numSgprOffsetA, kernarg=True)
      self.defineSgpr("OffsetB", self.numSgprOffsetB, kernarg=True)

    self.defineSgpr("Alpha", numSgprAlpha, numSgprAlpha, preload=True)
    if kernel["ProblemType"]["UseBeta"]:
      self.defineSgpr("Beta", numSgprBeta, numSgprBeta, kernarg=True)
    self.defineSgpr("StridesD", self.numSgprStridesD, kernarg=True)
    self.defineSgpr("StridesC", self.numSgprStridesC, kernarg=True)
    self.defineSgpr("StridesA", self.numSgprStridesA, preload=True)
    self.defineSgpr("StridesB", self.numSgprStridesB, preload=True)
    self.defineSgpr("SizesFree", self.numSgprSizesFree, kernarg=True)
    self.defineSgpr("SizesSum", self.numSgprSizesSum, preload=True)

    self.sumMagicParms = []
    if kernel["PackSummationDims"]:
      self.magicSumChars = [globalParameters["IndexChars"][c] for c in kernel["ProblemType"]["IndicesSummation"][1:]]
      self.sumMagicParms=["%s"%idxChar for idxChar in self.magicSumChars]
      if kernel["PackSummationDims"] and kernel["GlobalSplitU"] > 1 and self.sumMagicParms:
          self.sumMagicParms.append("%s_GsuRemainder"%self.unrollChar)
      for magicName in self.sumMagicParms:
        self.defineSgpr("MagicNumberSize%s"%magicName, 1, kernarg=True)
        self.defineSgpr("MagicShiftSize%s"%magicName, 1, kernarg=True)
    # for packed batches without stride restrictions need to do something different here
    assert sorted(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]) == \
           sorted(set(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]))
    for idxChar in kernel["PackedC0IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1, kernarg=True)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1, kernarg=True)
    for idxChar in kernel["PackedC1IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1, kernarg=True)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1, kernarg=True)
    for idx in kernel["ProblemType"]["IndicesSummation"]:
      for tc in ('A','B'):
        for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
          (freeDim, sumDim, padStart, padEnd) = zp
          if sumDim == idx:
            freeDimChar = globalParameters["IndexChars"][freeDim]
            sumDimChar  = globalParameters["IndexChars"][sumDim]
            # These will eventually be read as kernel args:
            self.defineSgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar),1, kernarg=True)
            self.defineSgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar),1, kernarg=True)
    if self.staggerU:
      self.defineSgpr("OrigStaggerUIter", 1, kernarg=True)  # Original stagger register.  Only needed for Persistent
    self.defineSgpr("NumWorkGroups0", 1, kernarg=True)
    self.defineSgpr("NumWorkGroups1", 1, kernarg=True)

    pkArgumentToLoad = 0
    if kernel["PersistentKernel"]:
      self.defineSgpr("MagicNumberProblemNumGroupTiles0", 1, kernarg=True) # Magic number to use for division
      self.defineSgpr("MagicShiftProblemNumGroupTiles0", 1, kernarg=True) # Magic shift/abit to use for division alg 2
      self.defineSgpr("GridNumWorkGroups0", 1, kernarg=True) # Magic number to use for division, persistent kernel - flattened wg0 (=all WGs)
      pkArgumentToLoad += 3
      if kernel["PersistentKernelAlongBatch"]:
        self.defineSgpr("NumWorkGroups2", 1, kernarg=True)  # for persistent kernel along batch
        self.defineSgpr("MagicNumProblemNumGroupTiles0By1", 1, kernarg=True)  # for PKAB, use for Magic Div Alg 2 by (nwg0*nwg1)
        self.defineSgpr("MagicShiftProblemNumGroupTiles0By1", 1, kernarg=True)  # for PKAB, use for Magic Div Alg 2 by (nwg0*nwg1)
        pkArgumentToLoad += 3
    skArgumentToLoad = 0
    if kernel["StreamK"]:
      # StreamK args
      self.defineSgpr("MagicNumberProblemNumGroupTiles0", 1, kernarg=True) # Magic number to use for division
      self.defineSgpr("MagicShiftProblemNumGroupTiles0", 1, kernarg=True) # Magic shift/abit to use for division alg 2
      self.defineSgpr("ItersPerTile", 1, kernarg=True)
      self.defineSgpr("MagicNumberItersPerTile", 1, kernarg=True)
      self.defineSgpr("MagicShiftItersPerTile", 1, kernarg=True)
      self.defineSgpr("MagicNumProblemNumGroupTiles0By1", 1, kernarg=True)  # for PKAB, use for Magic Div Alg 2 by (nwg0*nwg1)
      self.defineSgpr("MagicShiftProblemNumGroupTiles0By1", 1, kernarg=True)  # for PKAB, use for Magic Div Alg 2 by (nwg0*nwg1)
      self.defineSgpr("TotalIters", 1, kernarg=True)
      self.defineSgpr("SKItersPerWG", 1, kernarg=True)
      skArgumentToLoad += 9
      if kernel["StreamK"] >= 2: # Two-tile SK
        self.defineSgpr("skGrid", 1, kernarg=True)
        self.defineSgpr("skTiles", 1, kernarg=True)
        self.defineSgpr("skExtraIters", 1, kernarg=True)
        # self.defineSgpr("dpTilesPerWG", 1, kernarg=True)
        skArgumentToLoad += 3

    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    self.lastPostLoopSgpr = self.sgprPool.size()
    self.numSgprWGM = 0
    if abs(kernel["WorkGroupMapping"]) > 1:
      self.numSgprWGM = 3
      self.defineSgpr("NumFullBlocks", 1, kernarg=True) # Magic number to use for div by (NumWorkGroups1 % WGM)
      self.defineSgpr("WgmRemainder1", 1, kernarg=True) # Magic number to use for div by (NumWorkGroups1 % WGM)
      self.defineSgpr("MagicNumberWgmRemainder1", 1, kernarg=True) # Magic number to use for div by (NumWorkGroups1 % WGM)

    # SR only for F8 type
    if kernel["ProblemType"]["DataType"].is8bitFloat():
      if kernel["ProblemType"]["StochasticRounding"]: # in-device, only RNDSeed
        self.defineSgpr("RNDSeed", 1, kernarg=True)  # seed for random number generation

    #----

    if self.isInitCodeOptLW:
      # init code optimization: define PerpOverhangVccA, B just after kernel args (need this before undef OFFSET)
      if kernel["FractionalLoad"] == 2:
        if kernel["fractionalPerpOverhangA"]:
          self.defineSgpr("PerpOverhangVccA", 2, 2)
        if kernel["fractionalPerpOverhangB"]:
          self.defineSgpr("PerpOverhangVccB", 2, 2)
      # init code optimization: define LocalWriteUseSgprA, B just after kernel args (need this before undef OFFSET)
      if kernel["LocalWriteUseSgprA"]:
          self.defineSgpr("LocalWriteAddrA", 1)
      if kernel["LocalWriteUseSgprB"]:
          self.defineSgpr("LocalWriteAddrB", 1)

    self.numSgprToLoad = self.sgprPool.numKernargSGPRs
    if not kernel["ProblemType"]["StridedBatched"]:
      offsetD = self.sgprs["OffsetD"]
      self.argOffsetOffset = (offsetD - self.sgprPool.kernargStart) * 4
      assert self.sgprs["OffsetC"] == offsetD + 2
      assert self.sgprs["OffsetA"] == offsetD + 4
      assert self.sgprs["OffsetB"] == offsetD + 6

    # Get kernel argument end here
    ###################################

    ########################################
    # Register Pools
    ########################################
    #print "TotalVgprs", self.totalVgprs
    self.vgprPool = RegisterPool(self.totalVgprs, 'v', defaultPreventOverflow=False,
                                 printRP=self.db["PrintRP"])
    #print self.vgprPool.state()
    self.savedVgprPool = None
    self.savedSgprPool = None

    # C regs are not used during initialization so mark them as available -
    # we will claim then just before the start of the unroll loop:
    self.vgprPool.add(self.startVgprValuA, \
        self.lastValuAB - self.startVgprValuA, "ValuAB") # Add as available

    self.vgprPool.add(self.startVgprValuC, \
      self.numVgprValuC, "ValuC-Block") # Add as available
    #print self.vgprPool.state()
    ## accumulator Buffer for storeCinUnroll feature
    self.agprPool = RegisterPool(self.totalAgprs, 'a', defaultPreventOverflow=False, printRP=0)
    # C regs are not used during initialization so mark them as available -
    # we will claim then just before the start of the unroll loop:
    numAccvgprs = self.totalAgprs
    if kernel["StoreCInUnroll"]:
      numAccvgprs = self.totalAgprs - self.startaccValuC1
    self.agprPool.add(0, numAccvgprs, "ValuC-Block")

    # place any of these gpr inst values into tPA, tPB for later reference
    tPA["globalReadInstruction"] = self.globalReadInstructionA
    tPA["localWriteInstruction"] = self.localWriteInstructionA
    tPA["localReadInstruction"] = self.localReadInstructionA
    tPA["gpr"] = {}

    tPB["globalReadInstruction"] = self.globalReadInstructionB
    tPB["localWriteInstruction"] = self.localWriteInstructionB
    tPB["localReadInstruction"] = self.localReadInstructionB
    tPB["gpr"] = {}

    ########################################
    # reads Per Iteration
    ########################################
    if kernel["EnableMatrixInstruction"]:
      # setting numReadPerVector to 0 for DirectToVgpr makes performance a little worse.
      # so, keep this part unchanged.
      #self.numReadPerVectorA = 0 if kernel["DirectToVgprA"] else tPA["bpe"] * self.lrvwA // int(tPA["localReadInstruction"].blockWidth * 4)
      #self.numReadPerVectorB = 0 if kernel["DirectToVgprB"] else tPB["bpe"] * self.lrvwB // int(tPB["localReadInstruction"].blockWidth * 4)
      self.numReadPerVectorA = tPA["bpe"] * self.lrvwA // int(tPA["localReadInstruction"].blockWidth * 4)
      self.numReadPerVectorB = tPB["bpe"] * self.lrvwB // int(tPB["localReadInstruction"].blockWidth * 4)
      numA = kernel["InnerUnroll"]*(kernel["MIWaveTile"][0] * self.numReadPerVectorA) // tPA["localReadInstruction"].numOffsets
      numB = kernel["InnerUnroll"]*(kernel["MIWaveTile"][1] * self.numReadPerVectorB) // tPB["localReadInstruction"].numOffsets
      # wider localread has 2 mode
      # 1. using larger IU to coalesced localread, only half of local reads in 1 iteration
      # 2. using larger PLR to read more iterations, same number local reads in 1 iteration
      if kernel["InnerUnroll"] >= self.numReadsIterCoalescedA:
        numA //= self.numReadsIterCoalescedA
        if tPA["tlu"] and kernel["DirectToVgprA"]:
          # DTVA case, no local read is actually generated, but this still affects (dummy) local read scheduling for DTVA
          numA //= self.lrvwA
      if kernel["InnerUnroll"] >= self.numReadsIterCoalescedB:
        numB //= self.numReadsIterCoalescedB
      if self.useWiderLocalReadB:
        # use wider local read for UnrollMajorLDSB=False
        # numB needs to be divided by lrvwB
        numB //= self.lrvwB
    else:
      numB = kernel["InnerUnroll"]*(kernel["ThreadTile1"] // kernel["VectorWidth"]) // tPB["localReadInstruction"].numOffsets
      numA = kernel["InnerUnroll"]*(kernel["ThreadTile0"] // kernel["VectorWidth"]) // tPA["localReadInstruction"].numOffsets
    # Keep numA/B unchanged for DirectToVgprA/B
    # Setting numA/B=0 affects instruction scheduling and can cause performance drop
    #if kernel["DirectToVgprA"]:
    #  numA = 0
    #if kernel["DirectToVgprB"]:
    #  numB = 0
    self.numReadsPerIterA = numA
    self.numReadsPerIterB = numB
    self.localReadDoCntA   = 0
    self.localReadDoCntB   = 0

    if kernel["EnableMatrixInstruction"]:
      self.miLatency = kernel["MatrixInstM"] // 2
      miIssueLatency = 2
      if (self.version == (9,4,2)) and kernel["MatrixInstB"] == 1 and \
         (kernel["EnableF32XdlMathOp"] or \
          kernel["ProblemType"]["DataType"].is8bitFloat() or \
          kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16() or \
          kernel["ProblemType"]["DataType"].isInt8()):
        self.miLatency //= 2
        miIssueLatency //= 2
      # give 1 quad-cycle buffer to prevend bubble from sync
      miLatencyBuffer = 1
      self.miLatencyLeft = max(self.miLatency - miLatencyBuffer - miIssueLatency, 1) # minimum 1 to make scheduling work
      # add extra miLatencyLeft from parameter
      self.miLatencyLeft += kernel["ExtraMiLatencyLeft"]

    # pre-determine labels in order
    unrollChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
    self.labels = {}
    #self.getLabelNum("PrefetchGlobalBegin")
    self.getNamedLabel("PrefetchGlobalEnd")
    self.getNamedLabel("LoopBegin%s"%(unrollChar))
    self.getNamedLabel("LoopEnd%s"%(unrollChar))
    self.getNamedLabel("LoopEnd%s_oddexit"%(unrollChar))
    self.getNamedLabel("LoopEnd%s_evenexit"%(unrollChar))
    self.getNamedLabel("PrefetchGlobalLastIterEnd")
    self.getNamedLabel("TailLoopBegin%s"%(unrollChar))
    self.getNamedLabel("TailLoopEnd%s"%(unrollChar))
    self.getNamedLabel("SkipTailLoop%s"%(unrollChar))
    self.getNamedLabel("KernelEnd%s"%(unrollChar))
    # shift vectors determined later

    canCheckValueC = (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and \
                      kernel["ProblemType"]["HighPrecisionAccumulate"]
    canCheckValueC = canCheckValueC or kernel["ProblemType"]["DataType"].isSingle()
    canCheckValueC = canCheckValueC or (kernel["ProblemType"]["DataType"].isInt8() and kernel["ProblemType"]["HighPrecisionAccumulate"])
    canCheckValueC = canCheckValueC or kernel["ProblemType"]["DataType"].is8bitFloat() # F8 is always HPA
    assert not self.db["CheckValueC"] or canCheckValueC

    if self.db["InitLds"] : print ("\n***WARNING: InitLds enabled, may impact performance\n")
    if self.db["InitSgpr"] : print ("\n***WARNING: InitSgpr enabled, may impact performance\n")
    if self.db["InitVgpr"] : print ("\n***WARNING: InitVgpr enabled, may impact performance\n")
    if self.db["ConservativeWaitCnt"] : print ("\n***WARNING: ConservativeWaitCnt enabled, may impact performance\n")
    if self.do["KeepDirectToLdsAlloc"] : print ("\n***WARNING: KeepDirectToLdsAlloc enabled, may impact performance\n")
    if not kernel["LoopTail"] : print ("\n***WARNING: LoopTail disabled, kernel may not function correctly for all inputs\n")
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
    if kernel["CheckTensorDimAsserts"] : print ("\n***WARNING: CheckTensorDimAsserts enabled, may impact performance\n")
    if kernel["CheckDimOverflow"] : print ("\n***WARNING: CheckDimOverflow enabled, may impact performance\n")

  ##############################################################################
  # format macro
  def macroRegister(self, name, value, comment=""):
    if comment != "":
      assert '\n' not in comment, comment
      comment = " // " + comment
    return f".set {name}, {value}{comment}{self.endLine}"


  ##############################################################################
  # Function Prefix
  ##############################################################################
  def functionPrefix(self, kernel):
    return ""


  def defineMACs(self, kernel, m, innerUnroll):
    component = Component.MAC.find(self)
    if component:
      return component(self, m, innerUnroll)
    printExit("Assembly doesn't support %s" % kernel["ProblemType"]["DataType"])


  def defineMACMacro(self, kernel, innerUnroll, useMacro):
    """
    Defines a macro that performs one set of multiply-accumulate operations.
    """

    kStr = ""
    # Create a macro version that processes just one U iter
    # (used in tail loop in some cases)
    oneIUI = kernel["InnerUnroll"] > 1 and innerUnroll==1

    ########################################
    # MACs
    kStr += self.comment3("%dx%d thread-tile" \
        % (kernel["ThreadTile0"], kernel["ThreadTile1"]) )
    PLR = kernel["PrefetchLocalRead"] if kernel["PrefetchLocalRead"] < kernel["LoopIters"] else kernel["LoopIters"] - 1
    for m in range(0, 1+PLR):
      # Create a special macro that does one K iter if needed:
      ext = "_OneIUI" if oneIUI else ""
      if useMacro:
        kStr += ".macro MAC_%ux%u_X%u%s" \
            % (kernel["ThreadTile0"], kernel["ThreadTile1"], m, ext)
      kStr += self.endLine
      kStr += self.defineMACs(kernel, m, innerUnroll)
      if useMacro:
        kStr += ".endm%s" % self.endLine

    return kStr



  def defineVALUMacros(self):
    """
      Defines cross-architecture compatibility macros.
    """
    kStr = ""

    kStr += ".macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_add_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    # add w/o carry-out.  On older arch, vcc is still written
    kStr += self.endLine
    kStr += ".macro _v_add_u32 dst:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitNC"]:
        kStr += r"   v_add_nc_u32 \dst, \src0 \src1 \dpp" + self.endLine
    elif self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_add_u32 \dst, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_add_u32 \dst, vcc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    # add w/o carry-out.  On older arch, vcc is still written
    kStr += self.endLine
    kStr += ".macro _v_add_i32 dst:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitNC"]:
        kStr += r"   v_add_nc_i32 \dst, \src0 \src1 \dpp" + self.endLine
    elif self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_add_i32 \dst, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_add_i32 \dst, vcc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += self.endLine
    kStr += ".macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitNC"]:
        kStr += r"   v_add_co_ci_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    elif self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_addc_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += self.endLine
    kStr += ".macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_sub_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += self.endLine
    # sub w/o carry-out.  On older arch, vcc is still written.
    kStr += ".macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitNC"]:
        kStr += r"   v_sub_nc_u32 \dst, \src0, \src1 \dpp" + self.endLine
    elif self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_sub_u32 \dst, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_sub_u32 \dst, vcc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += self.endLine
    # sub w/o carry-out.  On older arch, vcc is still written.
    kStr += ".macro _v_sub_i32 dst:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitNC"]:
        kStr += r"   v_sub_nc_i32 \dst, \src0, \src1 \dpp" + self.endLine
    elif self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_sub_i32 \dst, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_sub_i32 \dst, vcc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    # Use combined add+shift, where available:
    kStr += self.endLine
    kStr += ".macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req" + self.endLine
    if globalParameters["AsmCaps"][self.version]["HasAddLshl"]:
      kStr += r"    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt" + self.endLine
    else:
      if self.AsmBugs["ExplicitCO"]:
        kStr += r"    v_add_co_u32 \dst, vcc, \src0, \src1" + self.endLine
      else:
        kStr += r"    v_add_u32 \dst, vcc, \src0, \src1" + self.endLine
      kStr += r"    v_lshlrev_b32 \dst, \shiftCnt, \dst" + self.endLine
    kStr += ".endm" + self.endLine

    # Use combined shift+add, where available:
    kStr += self.endLine
    kStr += ".macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req" + self.endLine
    if globalParameters["AsmCaps"][self.version]["HasAddLshl"]:
      kStr += r"    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt" + self.endLine
    else:
      kStr += r"    v_lshlrev_b32 \dst, \shiftCnt, \dst" + self.endLine
      if self.AsmBugs["ExplicitCO"]:
        kStr += r"    v_add_co_u32 \dst, vcc, \src0, \src1" + self.endLine
      else:
        kStr += r"    v_add_u32 \dst, vcc, \src0, \src1" + self.endLine
    kStr += ".endm" + self.endLine

    # Use combined shift+or, where available:
    kStr += self.endLine
    kStr += ".macro _v_lshl_or_b32 dst:req, src0:req, shiftCnt:req, src1:req" + self.endLine
    if globalParameters["AsmCaps"][self.version]["HasLshlOr"]:
      kStr += r"    v_lshl_or_b32 \dst, \src0, \shiftCnt, \src1" + self.endLine
    else:
      kStr += r"    v_lshlrev_b32 \dst, \shiftCnt, \src0" + self.endLine
      kStr += r"    v_or_b32 \dst, \dst, \src1" + self.endLine
    kStr += ".endm" + self.endLine

    # v_dot2acc & v_dot4_acc
    inst = 'v_dot2acc_f32_f16' if self.archCaps["InstRename"] else 'v_dot2c_f32_f16'
    kStr += self.endLine
    kStr += ".macro _v_dot2acc_f32_f16 dst, src0, src1"  + self.endLine
    kStr += f'{inst} \\dst, \\src0, \\src1' + self.endLine
    kStr += ".endm" + self.endLine

    return kStr


  def defineCMPXMacros(self):
    """
    Navi's cmpx instruction writes only to EXEC, not to SGPRs or to VCC.
    For now, replicate old behaviour with two instructions.
    """
    def macro(op, dtype):
      dict = {'op': op, 'dtype': dtype}
      mStr = ".macro _v_cmpx_{op}_{dtype} dst, src0, src1=".format(**dict) + self.endLine
      if self.archCaps["CMPXWritesSGPR"]:
        mStr += r"   v_cmpx_{op}_{dtype} \dst, \src0, \src1 ".format(**dict) + self.endLine
      else:
        mStr += r"   v_cmp_{op}_{dtype} \dst, \src0, \src1".format(**dict) + self.endLine
        if self.kernel["WavefrontSize"] == 64:
          mStr += r"   s_mov_b64 exec \dst" + self.endLine
        else:
          mStr += r"   s_mov_b32 exec_lo \dst" + self.endLine
      mStr += ".endm" + self.endLine
      return mStr

    ops = ['lt', 'eq', 'le', 'gt', 'ne', 'lg', 'ge', 'o', 'u']
    dtypes = list([sg + ln for sg in ['i','u'] for ln in ['16', '32', '64']])

    return self.endLine + self.endLine.join([macro(op, dtype) for op in ops for dtype in dtypes])


  def defineMACInstructionMacros(self):
    kStr = ""

    kStr += ".macro _v_mac_f32 c:req, a:req, b:req" + self.endLine
    if self.kernel["MACInstruction"] == "FMA":
      if self.asmCaps["v_fmac_f32"]:
        kStr += r"    v_fmac_f32 \c, \a, \b" + self.endLine
      elif self.asmCaps["v_fma_f32"]:
        kStr += r"    v_fma_f32 \c, \a, \b, \c" + self.endLine
      else:
        raise RuntimeError("FMA instruction specified but not supported on {}".format(self.kernel["ISA"]))
    elif self.asmCaps["v_mac_f32"]:
      kStr += r"    v_mac_f32 \c, \a, \b" + self.endLine
    else:
      raise RuntimeError("MAC instruction specified but not supported on {}".format(self.kernel["ISA"]))
    kStr += ".endmacro" + self.endLine

    return kStr


  def generalMacro(self, prefix, origin, replace, *args):
    kStr = ''
    kStr += f'.macro _{prefix}{origin}'
    for arg in args:
      kStr += f' {arg}'
    kStr += self.endLine

    kStr += f'    {prefix}{replace}'
    for arg in args:
      kStr += f' \\{arg}'
    kStr += self.endLine

    kStr += '.endm' + self.endLine
    return kStr


  def defineSLoadMacros(self):

    macro_list = {'b32' :'dword',
                  'b64' :'dwordx2',
                  'b128':'dwordx4',
                  'b256':'dwordx8',
                  'b512':'dwordx16'}
    kStr = self.comment('scale global load macros')
    for key in macro_list:
        origin = key
        replace = key if self.archCaps["InstRename"] else macro_list[key]
        kStr += self.generalMacro("s_load_", origin, replace, 'dst', 'base', 'offset') + self.endLine
    return kStr


  def defineDSMacros(self):

    kStr = self.comment('ds operation macros')

    width = ('u8', 'u8_d16_hi', 'u16', 'u16_d16_hi', 'b32', 'b64', 'b128')
    for w in width:
      origin = f'load_{w}'
      replace = f'load_{w}' if self.archCaps["InstRename"] else f'read_{w}'
      kStr += self.generalMacro('ds_', origin, replace, 'dst', 'src', 'offset') + self.endLine

    width = ('b8', 'b8_d16_hi', 'b16', 'b16_d16_hi', 'b32', 'b64', 'b128')
    for w in width:
      origin = f'store_{w}'
      replace = f'store_{w}' if self.archCaps["InstRename"] else f'write_{w}'
      kStr += self.generalMacro('ds_', origin, replace, 'dst', 'src', 'offset') + self.endLine

    width = ('b32', 'b64')
    op = {'load2' : 'read2',
          'store2': 'write2'}
    for key in op:
      for w in width:
        origin = f'{key}_{w}'
        replace = f'{key}_{w}' if self.archCaps["InstRename"] else f'{op[key]}_{w}'
        kStr += self.generalMacro('ds_', origin, replace, 'dst', 'src', 'offset1', 'offset2') + self.endLine

    return kStr


  def defineBufferMemoryMacros(self):
    kStr = self.comment('buffer memory operation macros')

    type_list = {
      'b32'       : 'dword',
      'b64'       : 'dwordx2',
      'b96'       : 'dwordx3',
      'b128'      : 'dwordx4',
      'd16_b16'   : 'short_d16',
      'd16_hi_b16': 'short_d16_hi',
      'd16_u8'    : 'ubyte_d16',
      'd16_hi_u8' : 'ubyte_d16_hi',
      'u16'       : 'ushort'
    }
    for t in type_list:
      origin  = f'{t}'
      replace = f'{t}' if self.archCaps["InstRename"] else f'{type_list[t]}'
      kStr += self.generalMacro('buffer_load_', origin, replace, 'dst', 'voffset', 'base', 'soffset', 'offen', 'ioffset', 'md0', 'md1', 'md2') + self.endLine

    # Extra macro for DirectToLds loads with no destination register
    type_list = {
      'b32'       : 'dword',
      'b64'       : 'dwordx2',
      'b128'      : 'dwordx4',
      'u16'       : 'ushort'
    }
    for t in type_list:
      origin  = f'{t}'
      replace = f'{type_list[t]}' if (self.version[0] < 11) else f'{t}'
      kStr += self.generalMacro('buffer_load_', origin + '_dtl', replace, 'voffset', 'base', 'soffset', 'offen', 'ioffset', 'md0', 'md1', 'md2') + self.endLine

    type_list = {
      'b32'       : 'dword',
      'b64'       : 'dwordx2',
      'b96'       : 'dwordx3',
      'b128'      : 'dwordx4',
      'b16'       : 'short',
      'd16_hi_b16': 'short_d16_hi',
      'b8'        : 'byte',
      'd16_hi_b8' : 'byte_d16_hi',
    }
    for t in type_list:
      origin  = f'{t}'
      replace = f'{t}' if self.archCaps["InstRename"] else f'{type_list[t]}'
      kStr += self.generalMacro('buffer_store_', origin, replace, 'src', 'voffset', 'base', 'soffset', 'offen', 'ioffset', 'md0', 'md1', 'md2') + self.endLine

    type_list = {'_b32': '',
                 '_b64': '_x2'}
    for t in type_list:
        origin  = f'{t}'
        replace = f'{t}' if self.archCaps["InstRename"] else f'{type_list[t]}'
        kStr += self.generalMacro('buffer_atomic_cmpswap', origin, replace, 'dst', 'voffset', 'base', 'soffset', 'offen', 'ioffset', 'md0', 'md1', 'md2') + self.endLine

    return kStr


  def defineGlobalMemoryMacros(self):
    kStr = self.comment('buffer memory operation macros')

    type_list = {
      'b32'       : 'dword',
      'b64'       : 'dwordx2',
      'b96'       : 'dwordx3',
      'b128'      : 'dwordx4',
      'd16_b16'   : 'short_d16',
      'd16_hi_b16': 'short_d16_hi',
      'd16_u8'    : 'ubyte_d16',
      'd16_hi_u8' : 'ubyte_d16_hi',
      'u16'       : 'ushort'
    }
    for t in type_list:
      origin  = f'{t}'
      replace = f'{t}' if self.archCaps["InstRename"] else f'{type_list[t]}'
      # use global_load/store instead of flat instructions
      kStr += self.generalMacro('global_load_', origin, replace, 'dst', 'base', 'src', 'ioffset', 'md0', 'md1', 'md2') + self.endLine

    type_list = {
      'b32'       : 'dword',
      'b64'       : 'dwordx2',
      'b96'       : 'dwordx3',
      'b128'      : 'dwordx4',
      'd16_b16'   : 'short',        # global_store name different from global_load for short on gfx9/10
      'd16_hi_b16': 'short_d16_hi',
      'd16_u8'    : 'ubyte_d16',
      'd16_hi_u8' : 'ubyte_d16_hi',
      'u16'       : 'ushort'
    }
    for t in type_list:
      origin  = f'{t}'
      replace = f'{t}' if self.archCaps["InstRename"] else f'{type_list[t]}'
      # use global_load/store instead of flat instructions
      kStr += self.generalMacro('global_store_', origin, replace, 'base', 'src', 'src2', 'md0', 'md1', 'md2') + self.endLine

    type_list = {'_b32': '',
                 '_b64': '_x2'}
    for t in type_list:
        origin  = f'{t}'
        replace = f'{t}' if self.archCaps["InstRename"] else f'{type_list[t]}'
        kStr += self.generalMacro('global_atomic_cmpswap', origin, replace, 'tmp', 'base', 'data', 'src', 'ioffset', 'md') + self.endLine

    return kStr


  def defineFeatureMacros(self):
    """
      Defines cross-architecture compatibility macros.
    """
    kStr = self.comment3("Asm syntax workarounds")
    kStr += self.defineVALUMacros()
    kStr += self.defineCMPXMacros()
    kStr += self.defineMACInstructionMacros()
    kStr += self.defineSLoadMacros()
    kStr += self.defineDSMacros()
    kStr += self.defineBufferMemoryMacros()
    kStr += self.defineGlobalMemoryMacros()

    return kStr


  ##############################################################################
  def functionSignature(self, kernel ):
    """
    Function Signature
    called after rest of code
    """
    kStr = ""

    signature = Component.Signature.find(self)
    kStr += signature(self)

    kStr += self.defineFeatureMacros()

    # Performs a division using 'magic number' computed on host
    # Argument requirements:
    #   - dstIdx must be two consecutive registers ; on exit the lower one will contain the quotient.  The upper is used as a temp.
    #   - First parm is passed as an integer vgpr index ; remaining are vgpr or sgpr symbolic names
    #   - dstIdx+1 cannot be same as dividend.  dividend+0 can be same as dividend and this may be useful for chaining divides.
    kStr += self.comment3("Magic div and mod functions")
    if kernel["MagicDivAlg"]==1: # TODO: remove me
        kStr += ".macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req" + self.endLine
        kStr += r"    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber" + self.endLine
        kStr += r"    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicNumber" + self.endLine
        kStr += r"    v_lshrrev_b64 v[\dstIdx:\dstIdx+1], \magicShift, v[\dstIdx:\dstIdx+1]" + self.endLine
        kStr += ".endm" + self.endLine
    elif kernel["MagicDivAlg"]==2:
        kStr += ".macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req" + self.endLine
        kStr += r"    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber" + self.endLine
        kStr += r"    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA" + self.endLine
        kStr += r"    _v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]" + self.endLine
        kStr += r"    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]" + self.endLine
        kStr += ".endm" + self.endLine

    ########################################
    # VGPR Macros
    ########################################
    kStr += self.comment3("VGPR Assignments")
    kStr += self.comment1("ValuC range: [%u-%u), %s"%(self.startVgprValuC, self.startVgprValuC+self.numVgprValuC, \
                           "serializedStore enabled" if self.serializedStore else ""))
    kStr += self.macroRegister("vgprValuC", self.startVgprValuC)

    kStr += self.comment1("ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx")
    # PLR index: from X0 to X<LoopIters-1> (at most) -> VGPRs will be duplicated LoopIters times (at most)
    # eg, if LoopIters = 4, there would be at most 4*VGPRs
    # PLR = kernel["PrefetchLocalRead"] if kernel["PrefetchLocalRead"] < kernel["LoopIters"] else kernel["LoopIters"] - 1
    PLR = min(kernel["PrefetchLocalRead"], kernel["LoopIters"]-1)
    numBi = PLR+1
    ri = 0
    if self.numVgprValuA > 0: # Do not generate vgprValuA if numVgprValuA is 0
      numBiFactor = numBi
      if kernel["DirectToVgprA"] and self.lrvwTileA > 1:
        # DirectToVgpr case, we need LoopIters * 2 buffers
        numBiFactor = kernel["LoopIters"] * 2
      for bi in range(0,numBiFactor): # buffer indices
        for iui in range(0, kernel["InnerUnroll"]):
          kStr += self.macroRegister("vgprValuA_X%u_I%u"%(bi,iui), self.startVgprValuA+ri)
          ri += self.numVgprValuAPerBlock
        if self.lrvwTileA > 1:
          ri = 0
      if self.numVgprValuPackA > 0:
        ri = 0
        if self.lrvwTileA > 1:
          for bi in range(0,numBiFactor): # buffer indices
            for iui in range(0, kernel["InnerUnroll"]):
              for data in range(0,kernel["MIInputPerThread"]):
                kStr += self.macroRegister("vgprValuA_X%u_I%u_D%u"%(bi,iui,data), self.startVgprValuPackA+ri)
                ri += ceil(self.lrvwTileA * self.tPA["bpe"] / self.bpr) * kernel["MIWaveTileA"] // self.lrvwTileA
        else:
          for data in range(1,int(self.numElemPerBprA)):
            for bi in range(0,numBi): # buffer indices
              for iui in range(0, kernel["InnerUnroll"]):
                kStr += self.macroRegister("vgprValuA_X%u_I%u_D%u"%(bi,iui,data), self.startVgprValuPackA+ri)
                ri += self.numVgprValuAPerBlock
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
        kStr += self.macroRegister("vgprG2LA", self.startVgprG2LA)
        if kernel["DirectToVgprA"]:
          # additional definition G2LA0, G2LA1 for swapping register sets
          kStr += self.macroRegister("vgprG2LA0", self.startVgprG2LA)
          kStr += self.macroRegister("vgprG2LA1", self.startVgprG2LA + self.numVgprG2LA//2)

    ri = 0
    if self.numVgprValuB > 0: # Do not generate vgprValuB if numVgprValuB is 0
      numBiFactor = numBi
      if kernel["DirectToVgprB"] and self.lrvwTileB > 1:
        # DirectToVgpr case, we need LoopIters * 2 buffers
        numBiFactor = kernel["LoopIters"] * 2
      for bi in range(0,numBiFactor): # buffer indices
        for iui in range(0, kernel["InnerUnroll"]):
          kStr += self.macroRegister("vgprValuB_X%u_I%u"%(bi,iui), self.startVgprValuB+ri)
          ri += self.numVgprValuBPerBlock
        if self.lrvwTileB > 1:
          ri = 0
      if self.numVgprValuPackB > 0:
        ri = 0
        if self.lrvwTileB > 1:
          for bi in range(0,numBiFactor): # buffer indices
            for iui in range(0, kernel["InnerUnroll"]):
              for data in range(0,kernel["MIInputPerThread"]):
                kStr += self.macroRegister("vgprValuB_X%u_I%u_D%u"%(bi,iui,data), self.startVgprValuPackB+ri)
                ri += ceil(self.lrvwTileB * self.tPB["bpe"] / self.bpr) * kernel["MIWaveTileB"] // self.lrvwTileB
        else:
          for data in range(1,int(self.numElemPerBprB)):
            for bi in range(0,numBiFactor): # buffer indices
              for iui in range(0, kernel["InnerUnroll"]):
                kStr += self.macroRegister("vgprValuB_X%u_I%u_D%u"%(bi,iui,data), self.startVgprValuPackB+ri)
                ri += self.numVgprValuBPerBlock
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
        kStr += self.macroRegister("vgprG2LB", self.startVgprG2LB)
        if kernel["DirectToVgprB"]:
          # additional definition G2LB0, G2LB1 for swapping register sets
          kStr += self.macroRegister("vgprG2LB0", self.startVgprG2LB)
          kStr += self.macroRegister("vgprG2LB1", self.startVgprG2LB + self.numVgprG2LB//2)
    if not kernel["LocalWriteUseSgprA"] and self.numVgprLocalWriteAddressesA > 0:
      kStr += self.macroRegister("vgprLocalWriteAddrA", \
          self.startVgprLocalWriteAddressesA)
      if self.numVgprLocalWriteAddressesA > 1:
        kStr += self.macroRegister("vgprLocalWriteAddrOverhangA", \
            self.startVgprLocalWriteAddressesA+1)
    if not kernel["LocalWriteUseSgprB"] and self.numVgprLocalWriteAddressesB > 0:
      kStr += self.macroRegister("vgprLocalWriteAddrB", \
          self.startVgprLocalWriteAddressesB)
      if self.numVgprLocalWriteAddressesB > 1:
        kStr += self.macroRegister("vgprLocalWriteAddrOverhangB", \
            self.startVgprLocalWriteAddressesB+1)
    if kernel["BufferLoad"]:
      kStr += self.macroRegister("vgprGlobalReadOffsetA", \
          self.startVgprGlobalReadOffsetA)
      kStr += self.macroRegister("vgprGlobalReadOffsetB", \
          self.startVgprGlobalReadOffsetB)
      if self.useGlobalReadTileVgpr:
        kStr += self.macroRegister("vgprGlobalReadTileOffsetA", \
            self.startVgprGlobalReadTileOffsetA)
        kStr += self.macroRegister("vgprGlobalReadUnrollOffsetA", \
            self.startVgprGlobalReadUnrollOffsetA)
        kStr += self.macroRegister("vgprGlobalReadTileOffsetB", \
            self.startVgprGlobalReadTileOffsetB)
        kStr += self.macroRegister("vgprGlobalReadUnrollOffsetB", \
            self.startVgprGlobalReadUnrollOffsetB)
    else:
      kStr += self.macroRegister("vgprGlobalReadAddrA", \
          self.startVgprGlobalReadAddressesA)
      kStr += self.macroRegister("vgprGlobalReadAddrB", \
          self.startVgprGlobalReadAddressesB)

    for tc in ('A','B'):
      for zpr in self.zeroPadRegs[tc].values():
        kStr += self.macroRegister("vgpr" + zpr.regName, zpr.vgprIdx)
        self.zpr = ZeroPadReg.State.MacroDef
    if self.globalReadIncsUseVgpr:
      kStr += self.macroRegister("vgprGlobalReadIncsA", \
          self.startVgprGlobalReadIncsA)
      kStr += self.macroRegister("vgprGlobalReadIncsB", \
          self.startVgprGlobalReadIncsB)
    if self.numVgprLocalReadAddressesA > 0:
      kStr += self.macroRegister("vgprLocalReadAddrA", \
          self.startVgprLocalReadAddressesA)
    if self.numVgprLocalReadAddressesB > 0:
      kStr += self.macroRegister("vgprLocalReadAddrB", \
          self.startVgprLocalReadAddressesB)

    if kernel["StoreCInUnroll"]:
      kStr += self.macroRegister("vgprG2LC", \
          self.startVgprG2LC)
      kStr += self.macroRegister("vgprL2GC", \
          self.startVgprL2GC)
      if not kernel["AtomicAddC"] and kernel["ProblemType"]["UseBeta"]:
        kStr += self.macroRegister("vgprGlobalReadOffsetC", \
            self.GlobalReadOffsetC)
        if not kernel["StoreCInUnrollExact"]:
          kStr += self.macroRegister("vgprGlobalReadOffsetCBackup", \
              self.GlobalReadOffsetCBackup)
      kStr += self.macroRegister("vgprGlobalWriteOffsetD", \
          self.GlobalWriteOffsetD)
      if not kernel["StoreCInUnrollExact"]:
        kStr += self.macroRegister("vgprGlobalWriteOffsetDBackup", \
            self.GlobalWriteOffsetDBackup)
      kStr += self.macroRegister("vgprGlobalBufferOOB", \
          self.GlobalBufferOOB)
    if kernel["ProblemType"]["DataType"].isComplex() and kernel["StoreCInUnroll"] \
       and (kernel["GlobalSplitU"] == 1 or kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
      kStr += self.macroRegister("vgprAlphaTmp", \
          self.startVgprAlphaTmp)

    if kernel["ProblemType"]["Fp16AltImpl"]:
      kStr += self.macroRegister("vgprG2Lpipe0", self.G2Lpipe0)
      kStr += self.macroRegister("vgprG2Lpipe1", self.G2Lpipe1)
      if kernel["ProblemType"]["Fp16AltImplRound"]:
        kStr += self.macroRegister("vgprFp16AltTmp", self.Fp16AltTmp)

    lrvwTileMax = max(self.lrvwTileA, self.lrvwTileB)
    if lrvwTileMax > 1 and (kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()):
      kStr += self.macroRegister("vgprPackTemp", self.vgprPackTemp)

    # Serial is always the last register in the pool so the store
    # code doesn't have to deal with fragmentation
    self.vgprstartSerial = self.vgprPool.size()-1
    kStr += self.macroRegister("vgprSerial", self.startVgprSerial)

    if kernel["StoreCInUnroll"]:
      kStr += self.macroRegister("accgprStoreCBuf0", self.startaccValuC0)
      kStr += self.macroRegister("accgprStoreCBuf1", self.startaccValuC1)

    if globalParameters["DebugKernel"]:
      kStr += self.macroRegister("vgprAddressDbg", \
          self.startVgprAddressDbg)
    #kStr += self.comment1("Occu: %u waves/simd" % self.numWavesPerSimd )
    kStr += self.comment1("Num VGPR=%u"%self.vgprPool.size())
    kStr += self.comment1("Num AccVGPR=%u"%self.agprPool.size())

    ########################################
    # SGPR Macros
    ########################################
    kStr += self.comment3("SGPR Assignments")

    # Emit declarations for all sgprs allocated with defineSgpr
    # in the order they were declared
    for skey in self.sgprs:
      regNum = self.sgprs[skey]
      count = self.sgprPool.checkOutSizeCache[skey] if skey in self.sgprPool.checkOutSizeCache else "unk"
      kStr += self.macroRegister("sgpr"+skey, regNum, f"({count})")
    kStr += self.comment1("max SGPR=%u"%self.sgprPool.size())

    kStr += "\n"
    kStr += self.comment1("Size Assignments")
    problemType = kernel["ProblemType"]
    for idx in range(max(problemType["IndexAssignmentsA"] + problemType["IndexAssignmentsB"])+1):
      idxChar= globalParameters["IndexChars"][idx]
      if idx in problemType["IndicesFree"] or idx in problemType["IndicesBatch"]:
        idxType="Free"
      elif idx in problemType["IndicesSummation"]:
        idxType="Sum"
        idx = idx - problemType["NumIndicesC"]
      else:
        raise ValueError("unexpected index type in size assignments")

      kStr += self.macroRegister("sgprSize%s"%(idxChar), \
                  "sgprSizes%s+%u"%(idxType, idx))

    kStr += "\n"
    kStr += self.comment1("Stride Assignments")
    for tc in ('D','C'):
      for idx in range(0, problemType["NumIndicesC"]):
        i = idx
        idxChar= self.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesCD"]:
          kStr += self.macroRegister("constStride%s%s"%(tc,idxChar), 1)
        else:
          if not kernel["ProblemType"]["UseInitialStridesCD"]:
            i = i-1
          kStr += self.macroRegister("sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s+%u"%(tc, i))

    for tc in ('A','B'):
      for i, idx in enumerate(problemType["IndexAssignments%s"%tc]):
        idxChar= self.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesAB"]:
          kStr += self.macroRegister("constStride%s%s"%(tc,idxChar), 1)
        else:
          if not kernel["ProblemType"]["UseInitialStridesAB"]:
            i = i-1
          kStr += self.macroRegister("sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s+%u"%(tc, i))

    kStr += "\n"
    kStr += self.macroRegister("MT0", kernel["MacroTile0"])
    kStr += self.macroRegister("MT1", kernel["MacroTile1"])
    kStr += self.macroRegister("DepthU", kernel["DepthU"])
    kStr += self.macroRegister("GSU", kernel["GlobalSplitU"])
    kStr += self.macroRegister("BpeA", self.tPA["bpe"])
    kStr += self.macroRegister("BpeALog2", log2(self.tPA["bpe"]))
    kStr += self.macroRegister("BpeB", self.tPB["bpe"])
    kStr += self.macroRegister("BpeBLog2", log2(self.tPB["bpe"]))
    kStr += self.comment1("Number of elements to shift-left SRD")
    kStr += self.macroRegister("SrdShiftLeftA", self.srdShiftLeft['A'])
    kStr += self.macroRegister("SrdShiftLeftB", self.srdShiftLeft['B'])

    if kernel["BufferLoad"] or kernel["BufferStore"]:
      kStr += self.comment1("2GB limit - set offsets to -1 to exceed this and clamp")
      # for A
      limitValue = 0 if self.enable["InvalidGlobalReadA"] else 0xffffffff
      kStr += self.macroRegister("BufferLimitA", hex(limitValue))
      # for B
      limitValue = 0 if self.enable["InvalidGlobalReadB"] else 0xffffffff
      kStr += self.macroRegister("BufferLimitB", hex(limitValue))
      #TODO-64 : This is max 32-bit negative value, the tail loop
      # does incrementally step through the GRO and increment GRO
      # which are initialized with this value
      #strOOB = "0xfffff000" if self.noTailLoop else "0x80000000" # use 32bit max (minus offset 12bit) in noTailLoop case
      strOOB = "0xfffff000" # use 32bit max (minus offset 12bit)
      kStr += self.macroRegister("BufferOOB", strOOB)

      srdUpperValue = Code.SrdUpperValue(self.version)
      kStr += self.comment3("Bits 127:96 of SRD.\n" + srdUpperValue.desc())
      kStr += self.macroRegister("Srd127_96", str(srdUpperValue))

    ########################################
    # Global Offsets
    ########################################
    # justOffset32 means we should only write the 32-bit offset
    # This is used in Buffer addressing modes.
    # Global addressing modes expect the GLOBAL_OFFSET to initialize a full 64-bit address
    for (tc, indices, justOffset32, tP) in [ \
        ("D", list(range(0, kernel["ProblemType"]["NumIndicesC"])), kernel["BufferStore"], None), \
        ("C", list(range(0, kernel["ProblemType"]["NumIndicesC"])), kernel["BufferStore"], None), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"], kernel["BufferLoad"], self.tPA), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"], kernel["BufferLoad"], self.tPB) ]:

      # BufferStore does not use this macro so don't generate it:
      if kernel["BufferStore"] and (tc == "C" or tc == "D"):
        continue

      kStr += self.comment("Global Offset %s"%tc)
      numDim = len(indices)
      idxChars = []
      for i in indices:
        idxChars.append(self.indexChars[i])

      packBatchDims = tP["PackBatchDims"] if tP != None else 0x3

      # macro declaration
      kStr += ".macro GLOBAL_OFFSET_%s vgprAddr:req"%tc
      calcDims = [] # dimensions which are participating in the address calc (ignores other summation)
      mirrorSumDims = []
      for i in range(0, numDim):
        if tc == 'C' or tc == 'D':
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesCD"]
          idxChar = self.indexChars[i]
        else:
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesAB"]
          idxChar = self.indexChars[tP['ia'][i]]

        # tile index or unroll vgpr or summation
        # other summation (other than unroll) are included in the GLOBAL_OFFSET macro but not used in address calc
        if     tc in ('A','C','D') and indices[i] == kernel["ProblemType"]["Index0"] \
            or tc in ('B','C','D') and indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          kStr += " vgprOffset%s:req" % idxChars[i]
          calcDims.append(i)
        elif indices[i] in kernel["ProblemType"]["IndicesSummation"]:
          # other summation index (not unroll)
          if tc in ('A', 'B') and indices[i] in kernel["ProblemType"]["MirrorDims%s" % tc]:
            mirrorSumDims.append(i)
          continue
        else:
          # other batch or free index
          isStridedBuffer = kernel["ProblemType"]["StridedBatched"] or (tc in ('C','D') and kernel["GlobalSplitU"] > 1 and kernel["_GlobalAccumulation"])
          if isPackedIndex(kernel, indices[i], packBatchDims):
            calcDims.append(i)
            kStr += " vgprOffset%s:req" % idxChars[i]
          elif not justOffset32 and isStridedBuffer:
            # buffer/justOffset32 scalars are included in SRD not the offset, so skip here
            # also skip case for general batch, unique pointers are loaded by workgroup
            calcDims.append(i)
            kStr += " sgprOffset%s:req" % idxChars[i]
      kStr += " vgprTmp:req" + self.endLine

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
          dest = "v[\\vgprAddr+0]"
          needAdd = 0 # don't need add since writing address directly.
          writeDirectToAddr = 0
        else:
          dest = "v[\\vgprTmp+0]"
          needAdd = 1
        kStr += inst("_v_sub_u32", \
                dest,
                sgpr("Size%s"%globalParameters["IndexChars"][indices[i]]), \
                "1", \
                "mirror %s%s 1"%(tc, globalParameters["IndexChars"][indices[i]]))
        kStr += inst("v_mul_lo_u32", \
                dest,
                dest, \
                self.strideRef(tc, indices[i]), \
                "mirror %s%s 2"%(tc, globalParameters["IndexChars"][indices[i]]))

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = "v[\\vgprAddr+0]"
          destHi = "v[\\vgprAddr+1]"

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          kStr += inst("_v_add_co_u32", \
            destLo, \
            self.vcc, \
            srcLo, \
            "v[\\vgprTmp+0]", \
            "accumulate %s lower"%idxChar)

      for i in calcDims:
        # should have eliminated these above
        idx = indices[i]
        isMirrorIdx = tc in ('A', 'B') and idx in kernel["ProblemType"]["MirrorDims%s" % tc]
        assert not (idx in kernel["ProblemType"]["IndicesSummation"] and idx != kernel["ProblemType"]["IndexUnroll"])

        if indices[i] == kernel["ProblemType"]["Index0"] \
            or indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          offsetIsVgpr = True
        # other c index sgpr (free or batch)
        elif indices[i] < kernel["ProblemType"]["NumIndicesC"]:
          if isPackedIndex(kernel, indices[i], packBatchDims):
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

        #kStr += self.comment1("dim%s pendingOffset=%s offset=%s offsetIsVgpr=%s" \
        #    % (self.indexChars[indices[i]], pendingOffset, offset, offsetIsVgpr))

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
              destLo = "v[\\vgprAddr+0]"
              destHi = "v[\\vgprAddr+1]"
              needAdd = 0 # don't need add since writing address directly.
              writeDirectToAddr = 0
            else:
              destLo = "v[\\vgprTmp+0]"
              destHi = "v[\\vgprTmp+1]"
              needAdd = 1
            if isMirrorIdx:
              kStr += inst("_v_sub_i32", \
                "v[\\vgprTmp+0]",
                sgpr("Size%s"%globalParameters["IndexChars"][idx]), \
                offset, \
                "mirror %s%s 1"%(tc, globalParameters["IndexChars"][indices[i]]))
              kStr += inst("_v_sub_i32", \
                "v[\\vgprTmp+0]",
                "v[\\vgprTmp+0]", \
                "1", \
                "mirror %s%s 2"%(tc, globalParameters["IndexChars"][indices[i]]))
              offset = "v[\\vgprTmp+0]"

            # offset * stride
            kStr += inst("v_mul_lo_u32", \
                destLo,
                self.strideRef(tc, indices[i]), \
                offset, \
                "mul d%u lower"%i)
            if not justOffset32:
              kStr += inst("v_mul_hi_u32", \
                  destHi,
                  self.strideRef(tc, indices[i]), \
                  offset, \
                  "mul d%u upper"%i)
          else: # offset is SGPR:
            assert not isMirrorIdx
            if not justOffset32:
              # buffer mode (aka justOffset32) does scalars into SRD not offset
              kStr += inst("v_mov_b32", \
                  "v[\\vgprTmp+2]", \
                  "s[\\sgprOffset%s]"%idxChars[i], \
                  "sgprOffset -> vgprTmp+2")
              # offset * stride
              kStr += inst("v_mul_lo_u32", \
                  "v[\\vgprTmp+0]", \
                  self.strideRef(tc, indices[i]), \
                  "v[\\vgprTmp+2]",  \
                  "other stride mul d%u lower"%i)
              kStr += inst("v_mul_hi_u32", \
                  "v[\\vgprTmp+1]", \
                  self.strideRef(tc, indices[i]), \
                  "v[\\vgprTmp+2]",  \
                  "mul d%u upper"%i)
              needAdd = 1

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = "v[\\vgprAddr+0]"
          destHi = "v[\\vgprAddr+1]"
          # addr += offset * stride (lo) : accumulate just-computed address term into addr

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          kStr += inst("_v_add_co_u32", \
            destLo, \
            self.vcc, \
            srcLo, \
            "v[\\vgprTmp+0]", \
            "accumulate %s lower"%idxChar)

          # addr += offset * stride (hi)
          if not justOffset32:
            kStr += inst("_v_addc_co_u32", \
                "v[\\vgprAddr+1]", \
                self.vcc, \
                "v[\\vgprTmp+1]",  \
                srcHi, \
                self.vcc, \
                "accumulate %s upper"%idxChar)
          pendingOffset = None

      # pendingOffset but never got a chance to apply it,
      # need to just add an explicit move or add:
      # this can happen for small-order tensors
      if pendingOffset != None:
        destLo = "v[\\vgprAddr+0]"
        if writeDirectToAddr:
          kStr += inst("v_mov_b32", destLo, offset, "setup d0 lower")
          if not justOffset32:
            kStr += inst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper")
        else:
          kStr += inst("_v_add_co_u32", \
            destLo, \
            self.vcc, \
            destLo, \
            pendingOffset, \
            "accumulate final pendingOffset")


      if tP != None and kernel["BufferLoad"] and self.srdShiftLeft[tc]:
        kStr += inst("_v_add_u32", \
            "v[\\vgprAddr+0]", \
            hex(self.srdShiftLeft[tc]), \
            "v[\\vgprAddr+0]", \
            "add prepad for pointer shift")

      # GRVW<1 case, bpe calculation needs to be done before glvw calculation. Hence, skip bpe calculation here
      useEarlyBpeCalc = (tc == "A" or tc == "B") and kernel["GlobalLoadVectorWidth%c"%tc] < 1
      if not useEarlyBpeCalc:
        # addr *= bytes/element
        bpe = self.bpeCexternal if (tc == "C" or tc == "D") else self.bpeAB
        if justOffset32:
          kStr += staticMultiply("v[\\vgprAddr+0]", "v[\\vgprAddr+0]", bpe, None, "offset *= bytes/element")
        else:
          kStr += inst("v_lshlrev_b64", \
              "v[\\vgprAddr+0:\\vgprAddr+1]", \
              hex(log2(bpe)), \
              "v[\\vgprAddr+0:\\vgprAddr+1]", \
              "offset *= bytes/element")
      #kStr += "s_endpgm\n"
      kStr += ".endm%s" % self.endLine

    ########################################
    # Dynamic Scalar Divide
    kStr += self.comment3("Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor;")
    kStr += ".macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp%s" % self.endLine
    sTmpStr = "s[\\sTmp]" if (self.kernel["WavefrontSize"] == 32) else "s[\\sTmp:\\sTmp+1]"
    kStr += inst("v_cvt_f32_u32", "v[\\vQuotient]",  "v[\\vDivisor]",  "" )
    kStr += inst("v_rcp_f32",     "v[\\vQuotient]",  "v[\\vQuotient]", "" )
    kStr += inst("v_mul_f32",     "v[\\vQuotient]",  "0x4f800000",     "v[\\vQuotient]", "" )
    kStr += inst("v_cvt_u32_f32", "v[\\vQuotient]",  "v[\\vQuotient]", "" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vDivisor]", "v[\\vQuotient]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vTmp0]",      "v[\\vDivisor]", "v[\\vQuotient]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vTmp1]",      self.vcc, hex(0),    "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ne_i32",  sTmpStr, hex(0),        "v[\\vTmp0]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vRemainder]", "v[\\vTmp1]",     "v[\\vRemainder]", sTmpStr, "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vRemainder]", "v[\\vRemainder]", "v[\\vQuotient]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vTmp0]",      self.vcc,            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("_v_add_co_u32",     "v[\\vQuotient]",  self.vcc,            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vTmp0]", sTmpStr, "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vDividend]", "" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vTmp0]",      self.vcc,            "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ge_u32",  sTmpStr, "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("_v_add_co_u32",     "v[\\vRemainder]", self.vcc,            hex(1), "v[\\vQuotient]", "" )
    kStr += inst("_v_add_co_u32",     "v[\\vTmp1]",      self.vcc, -1,        "v[\\vQuotient]", "" )
    kStr += inst("v_cmp_le_u32",  self.vcc,             "v[\\vDivisor]", "v[\\vTmp0]", "" )
    kStr += inst("s_and_b{}".format(self.kernel["WavefrontSize"]),     self.vcc,             sTmpStr,         self.vcc,     "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vRemainder]", self.vcc, "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vTmp1]",     "v[\\vQuotient]", sTmpStr, "" )
    kStr += inst("v_cmp_ne_i32",  self.vcc, hex(0),     "v[\\vDivisor]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  -1, "v[\\vQuotient]", self.vcc, "final result" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vRemainder]", self.vcc,            "v[\\vDividend]", "v[\\vRemainder]", "final result" )
    kStr += ".endm%s" % self.endLine

    ########################################
    # IN DEVICE RANDOM NUMBER GENERATOR
    if kernel["ProblemType"]["StochasticRounding"]:  # ask in-device RND generator
      prndgen = Component.PseudoRandomGenerator.find(self)
      kStr += prndgen(self)

    if not kernel["EnableMatrixInstruction"]:
      kStr += self.defineMACMacro(kernel, kernel["InnerUnroll"], True)
      if kernel["InnerUnroll"] > 1:
        kStr += self.defineMACMacro(kernel, 1, True) # define OneIter case

    if self.overflowedResources:
      if self.overflowedResources == 1:
        msg = "too many vgprs"
      elif self.overflowedResources == 2:
        msg = "too many sgprs"
      elif self.overflowedResources == 3:
        msg = "half store requires at least two elements per batch"
      elif self.overflowedResources == 4:
        msg = "Occupancy limit"
      elif self.overflowedResources == 5:
        if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
          msg = "cannot schedule local read with DirectToLds"
        else:
          msg = "reading and writing LDS at same time require 2 LDS buffer"
      elif self.overflowedResources == 6:
        msg = "SIA2 better with occupancy 2"
      elif self.overflowedResources == 7:
        msg = "too many accvgprs"
      else:
        msg = "unknown"

      if globalParameters["PrintSolutionRejectionReason"]:
        printWarning(
            f"{self.kernelName} overflowed resources. errorCode={self.overflowedResources}" \
            f", msg='{msg}', vgprs={self.vgprPool.size()}, sgprs={self.sgprPool.size()}"
            + f", accvgprs={self.agprPool.size()}" if kernel["EnableMatrixInstruction"] else ""
        )
      kStr += "s_endpgm // overflowed resources\n"
      kStr += ".if 0\n"

    return kStr

  ##############################################################################
  # Function Beginning
  ##############################################################################
  def functionSignaturePrefix(self, kernel): return ""
  def functionSignatureSuffix(self, kernel): return ""
  def functionBegin(self, kernel): return ""

  ##############################################################################
  # getKernArg
  # Write an argument to specified SGPR and move the kernArgOffset
  # if writeSgpr==0, just move the kernArgOffset - this is used to skip
  # unused parms
  ##############################################################################
  def getKernArg(self, parmName, writeSgpr=1):
    kStr = ""
    size = 1*4
    if writeSgpr:
      kStr += inst("_s_load_b32", sgpr(parmName), \
          sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
    self.kernArgOffset += size
    return kStr

  ##############################################################################
  # code phrase for load batched address from array of buffer pointer
  ##############################################################################
  def loadBatchedAddress(self, kernel, Batch, tmpSgpr):
    laneSC = self.laneSGPRCount
    kStr = self.endLine

    # handle Batch C/D
    if not kernel["_GlobalAccumulation"]:
      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(Batch), 0x8, "offset of global buffer address")
          kStr += inst("_s_load_b64", sgpr("AddressD", 2), sgpr("AddressD",2), sgpr(tmpSgpr), "load global buffer D address")

      endCheckLabel = self.getNamedLabel(f"label_skip_c_buffer_deref_{Batch}")
      if kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
        kStr += inst("v_cmp_eq_f64", sgpr(tmpSgpr, laneSC), sgpr("Beta", 2), 0.0, "Beta.real == 0.0 ?")
        kStr += inst("v_cmp_eq_f64", self.vcc, sgpr("Beta+2", 2), 0.0, "Beta.imag == 0.0 ?")
        kStr += inst(f"s_and_b{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), self.vcc, sgpr(tmpSgpr, laneSC), "Beta == 0 ?")
        kStr += inst(f"s_cmp_eq_u{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), hex(0), "branch if beta == 0")
        kStr += inst("s_cbranch_scc0 %s" % (endCheckLabel), "branch if beta == 0")
      elif kernel["ProblemType"]["ComputeDataType"].isDouble():
        kStr += inst("v_cmp_eq_f64", self.vcc, sgpr("Beta", 2), 0.0, "Beta == 0.0 ?")
        kStr += inst("s_cbranch_vccnz %s" % (endCheckLabel), "branch if Beta == 0")
      elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
        kStr += inst("v_cmp_eq_f32", sgpr(tmpSgpr, laneSC), sgpr("Beta"), 0.0, "Beta.real == 0.0f ?")
        kStr += inst("v_cmp_eq_f32", self.vcc, sgpr("Beta+1"), 0.0, "Beta.imag == 0.0f ?")
        kStr += inst(f"s_and_b{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), self.vcc, sgpr(tmpSgpr, laneSC), "Beta == 0 ?")
        kStr += inst(f"s_cmp_eq_u{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), hex(0), "branch if beta == 0")
        kStr += inst("s_cbranch_scc0 %s" % (endCheckLabel), "branch if beta == 0")
      elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
           kernel["ProblemType"]["ComputeDataType"].isHalf() or \
           kernel["ProblemType"]["ComputeDataType"].isBFloat16():
        kStr += inst("v_cmp_eq_f32", self.vcc, sgpr("Beta"), 0.0, "Beta == 0.0f ?")
        kStr += inst("s_cbranch_vccnz %s" % (endCheckLabel), "branch if beta == 0")
      else: # int32
        kStr += inst("s_cmp_eq_u32", sgpr("Beta"), 0, "Beta == 0 ?")
        kStr += inst("s_cbranch_scc1 %s" % (endCheckLabel), "branch if beta == 0")

      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(Batch), 0x8, "offset of global buffer address")
          kStr += inst("_s_load_b64", sgpr("AddressC", 2), sgpr("AddressC",2), sgpr(tmpSgpr), "load global buffer C address")

      kStr += self.getNamedLabelDef(f"label_skip_c_buffer_deref_{Batch}")

    #handle Batch A/B
    endCheckLabel = self.getNamedLabel(f"label_skip_ab_buffer_deref_{Batch}")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(1), "check summation size")
    for i in range(0, self.numSgprSizesSum):
      kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr("SizesSum+%u"%(i)), sgpr(tmpSgpr), "check summation size")
    kStr += inst("s_cmp_eq_u32", sgpr(tmpSgpr), hex(0), "skip buffer deref is size of summation is 0")
    kStr += inst("s_cbranch_scc1", endCheckLabel, "skip buffer deref is size of summation is 0")

    if kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
      kStr += inst("v_cmp_eq_f64", sgpr(tmpSgpr, laneSC), sgpr("Alpha", 2), 0.0, "Alpha.real == 0.0 ?")
      kStr += inst("v_cmp_eq_f64", self.vcc, sgpr("Alpha+2", 2), 0.0, "Alpha.imag == 0.0 ?")
      kStr += inst(f"s_and_b{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), self.vcc, sgpr(tmpSgpr, laneSC), "Alpha == 0 ?")
      kStr += inst(f"s_cmp_eq_u{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), hex(0), "branch if alpha == 0")
      kStr += inst("s_cbranch_scc0 %s" % (endCheckLabel), "branch if alpha == 0")
    elif kernel["ProblemType"]["ComputeDataType"].isDouble():
      kStr += inst("v_cmp_eq_f64", self.vcc, sgpr("Alpha", 2), 0.0, "Alpha == 0.0 ?")
      kStr += inst("s_cbranch_vccnz %s" % (endCheckLabel), "branch if Alpha == 0")
    elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
      kStr += inst("v_cmp_eq_f32", sgpr(tmpSgpr, laneSC), sgpr("Alpha"), 0.0, "Alpha.real == 0.0f ?")
      kStr += inst("v_cmp_eq_f32", self.vcc, sgpr("Alpha+1"), 0.0, "Alpha.imag == 0.0f ?")
      kStr += inst(f"s_and_b{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), self.vcc, sgpr(tmpSgpr, laneSC), "Alpha == 0 ?")
      kStr += inst(f"s_cmp_eq_u{kernel['WavefrontSize']}", sgpr(tmpSgpr, laneSC), hex(0), "branch if alpha == 0")
      kStr += inst("s_cbranch_scc0 %s" % (endCheckLabel), "branch if alpha == 0")
    elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
         kernel["ProblemType"]["ComputeDataType"].isHalf() or \
         kernel["ProblemType"]["ComputeDataType"].isBFloat16():
      kStr += inst("v_cmp_eq_f32", self.vcc, sgpr("Alpha"), 0.0, "Alpha == 0.0f ?")
      kStr += inst("s_cbranch_vccnz %s" % (endCheckLabel), "branch if alpha == 0")
    else: # int32
      kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), 0, "Alpha == 0 ?")
      kStr += inst("s_cbranch_scc1 %s" % (endCheckLabel), "branch if alpha == 0")

    kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(Batch), 0x8, "offset of global buffer address")
    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        kStr += inst("_s_load_b64", sgpr("AddressA", 2), sgpr("AddressA",2), sgpr(tmpSgpr), "load global buffer A address")
        kStr += inst("_s_load_b64", sgpr("AddressB", 2), sgpr("AddressB",2), sgpr(tmpSgpr), "load global buffer B address")

    kStr += self.getNamedLabelDef(f"label_skip_ab_buffer_deref_{Batch}")

    return kStr

    ########################################
    # load kernel args
  def loadKernelArguments(self):
    kStr = ""

    kStr += self.comment("Load Kernel Args")
    self.kernArgOffset = 0
    if self.kernel["PreloadKernelArguments"] == 1:
      self.kernArgOffset = self.sgprPool.numPreloadSGPRs * 4

    if globalParameters["DebugKernel"]:
      kStr += self.getKernArg("AddressDbg")
      kStr += self.getKernArg("AddressDbg+1")

    load = self.numSgprToLoad
    sgprStart = self.sgprPool.kernargStart
    while load > 0:
      if load >= 16:
        load -= 16
        kStr += inst("_s_load_b512", sgpr(sgprStart,16), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
        sgprStart += 16
        self.kernArgOffset += 16 * 4
        continue
      if load >= 8:
        load -= 8
        kStr += inst("_s_load_b256", sgpr(sgprStart,8), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
        sgprStart += 8
        self.kernArgOffset += 8 * 4
        continue
      if load >= 4:
        load -= 4
        kStr += inst("_s_load_b128", sgpr(sgprStart,4), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
        sgprStart += 4
        self.kernArgOffset += 4 * 4
        continue
      if load >= 2:
        load -= 2
        kStr += inst("_s_load_b64", sgpr(sgprStart,2), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
        sgprStart += 2
        self.kernArgOffset += 2 * 4
        continue
      if load >= 1:
        load -= 1
        kStr += inst("_s_load_b32", sgpr(sgprStart), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
        sgprStart += 1
        self.kernArgOffset += 1 * 4
        continue
    # currently align sgpr to kernel argument memory, and use s_load_bxxx to load argument as large as possible in one instruction
    # however, in order to match sgpr to kernel argument memory, some unnecessarily sgpr will also be defined, and caused wasting of sgpr.
    # TODO: more efficient way is to organize both sgpr and kernel argument memory in API

    # KernArgAddress needed for general batch after loading arguments
    if self.kernel["ProblemType"]["StridedBatched"] or not self.kernel["ProblemType"]["Batched"]:
      self.undefineSgpr("KernArgAddress")

    return kStr

  ##############################################################################
  def allocateResources(self, kernel, lraCode=None):
    kStr = ""

    if kernel["PreloadKernelArguments"] == 1:
      kStr += self.comment("256 bytes of s_nop")
      for i in range(64):
        kStr += inst("s_nop 0", "preload")

      kStr += self.comment("256 bytes of s_nop")
      kStr += "\n"
    kStr += f"{self.kernelName}_preloaded: // Kernel start when preloading\n"


    if kernel["StorePriorityOpt"]:
      kStr += inst("s_setprio 3", "optimization store")

    if self.do["NullKernel"]:
      kStr += inst("s_endpgm", "Skip the whole kernel")

    if self.do["PreLoop"]:
      if self.db["InitSgpr"] & 0x1:
        kStr += self.comment("Init SGPRs")
        for i in range(self.firstInitSgpr, self.sgprPool.size()):
          kStr += inst("s_mov_b32", sgpr(i), hex(self.initSgprValue), "InitSgpr&0x1")
        kStr += "\n"

      if self.db["InitVgpr"] & 0x1:
        kStr += self.comment("Init VGPRs")
        for i in range(1, self.totalVgprs):
          kStr += inst("v_mov_b32", vgpr(i), hex(self.initVgprValue), "InitVgpr&0x1")
        kStr += "\n"

      if self.kernel["WavefrontSize"] == 32:
        kStr += inst("s_mov_b32", "vcc_hi", "0", "Ensure hi bits are zero")

      if not self.kernel["DelayRemainingArguments"]:
        kStr += self.loadKernelArguments()

      # set m0
      kStr += inst("s_mov_b32", "m0", hex(kernel["LdsNumElements"] \
          * self.bpeAB), "LDS clamp at %u bytes" \
          %(kernel["LdsNumElements"] * self.bpeAB) )

      # set Serial id vgpr
      kStr += inst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")

      if kernel.enabledSetPrioSplitLDS:
        kStr += inst("s_setprio", "1", "prioritize init code so as to issue load sooner")

      # init code optimization
      # add local read/write code (before wait for kernel arg load)
      if lraCode != None :
        kStr += lraCode

      if kernel["PreloadKernelArguments"] != 1:
        kernArgBytes = self.sgprPool.numKernargSGPRs * 4
        kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for %u bytes of kern args" % kernArgBytes )

      if not kernel["ProblemType"]["StridedBatched"]:
        tmpSgpr = self.getTmpSgpr(self.laneSGPRCount).idx()
        kStr += self.loadBatchedAddress(kernel, "WorkGroup2", tmpSgpr)
        kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait global buffer address ready")
    else:
      kStr += ".if 0\n"

    # sgpr allocation optimization
    # (after batched address loaded) store addresses in SrdA,B,C,D instead of AddressA,B,C,D
    # then, release Address A,B,C,D after initial calculation is done
    # not applicable cases
    #  - PersistentKernel
    #  - CheckDimOverflow >= 2
    #  - convolution (not GEMM) (does not work)
    #  - not BufferLoad (for A,B)
    #  - not BufferStore (for C,D)
    #  - not PrefetchGlobalRead (for C,D)
    self.releaseSgprAdressAB = False
    self.sgprAddressStrAB = "Address"
    self.releaseSgprAdressCD = False
    self.releaseSgprAdressCDlater = False
    self.sgprAddressStrCD = "Address"
    if (not kernel["PersistentKernel"] and not kernel["StreamK"]) and kernel["CheckDimOverflow"]<2 and kernel["ProblemType"]["OperationType"] == 'GEMM':
      # A,B check
      if kernel["BufferLoad"]:
        self.releaseSgprAdressAB = True
        self.sgprAddressStrAB = "Srd"
      # C,D check
      if kernel["BufferStore"] and kernel["PrefetchGlobalRead"]:
        if kernel["PreloadKernelArguments"] != 1:
          self.releaseSgprAdressCD = True
          self.sgprAddressStrCD = "Srd"
        else:
          self.releaseSgprAdressCDlater = True

    # add offset to buffer
    if not kernel["ProblemType"]["StridedBatched"] and (not kernel["_GlobalAccumulation"] or kernel["_GlobalAccumulation"] == 'PartialsBuffer'):
      kStr += inst("s_lshl_b64", sgpr("OffsetD", 2), sgpr("OffsetD", 2), hex(log2(self.bpeCexternal)), "elements offset to bytes offset")
      kStr += inst("s_add_u32",  sgpr("%sD+0"%self.sgprAddressStrCD), sgpr("AddressD+0"), sgpr("OffsetD"), "add offset to buffer address")
      kStr += inst("s_addc_u32", sgpr("%sD+1"%self.sgprAddressStrCD), sgpr("AddressD+1"), sgpr("OffsetD+1"), "add offset to buffer address")

      kStr += inst("s_lshl_b64", sgpr("OffsetC", 2), sgpr("OffsetC", 2), hex(log2(self.bpeCexternal)), "elements offset to bytes offset")
      kStr += inst("s_add_u32",  sgpr("%sC+0"%self.sgprAddressStrCD), sgpr("AddressC+0"), sgpr("OffsetC"), "add offset to buffer address")
      kStr += inst("s_addc_u32", sgpr("%sC+1"%self.sgprAddressStrCD), sgpr("AddressC+1"), sgpr("OffsetC+1"), "add offset to buffer address")
    elif self.releaseSgprAdressCD:
      # copy AddressC,D to srdC,D to undefine AddressC,D
      kStr += inst("s_mov_b64", sgpr("%sC+0"%self.sgprAddressStrCD, 2), sgpr("AddressC+0", 2), "copy addressC")
      kStr += inst("s_mov_b64", sgpr("%sD+0"%self.sgprAddressStrCD, 2), sgpr("AddressD+0", 2), "copy addressD")

    dstAddressA0Str = sgpr("%sA+0"%self.sgprAddressStrAB)
    dstAddressA1Str = sgpr("%sA+1"%self.sgprAddressStrAB)
    dstAddressB0Str = sgpr("%sB+0"%self.sgprAddressStrAB)
    dstAddressB1Str = sgpr("%sB+1"%self.sgprAddressStrAB)
    srcPadA0Str     = dstAddressA0Str
    srcPadA1Str     = dstAddressA1Str
    srcPadB0Str     = dstAddressB0Str
    srcPadB1Str     = dstAddressB1Str
    if not kernel["ProblemType"]["StridedBatched"]:
      kStr += inst("s_lshl_b64", sgpr("OffsetA", 2), sgpr("OffsetA", 2), hex(log2(self.bpeAB)), "elements offset to bytes offset")
      kStr += inst("s_add_u32",  dstAddressA0Str, sgpr("AddressA+0"), sgpr("OffsetA"), "add offset to buffer address")
      kStr += inst("s_addc_u32", dstAddressA1Str, sgpr("AddressA+1"), sgpr("OffsetA+1"), "add offset to buffer address")

      kStr += inst("s_lshl_b64", sgpr("OffsetB", 2), sgpr("OffsetB", 2), hex(log2(self.bpeAB)), "elements offset to bytes offset")
      kStr += inst("s_add_u32",  dstAddressB0Str, sgpr("AddressB+0"), sgpr("OffsetB"), "add offset to buffer address")
      kStr += inst("s_addc_u32", dstAddressB1Str, sgpr("AddressB+1"), sgpr("OffsetB+1"), "add offset to buffer address")
    elif self.releaseSgprAdressAB:
      if self.groOffsetInMacroTile:
        # use AddressA/B registers as src of pre-pad add (no need to copy)
        srcPadA0Str = sgpr("AddressA+0")
        srcPadA1Str = sgpr("AddressA+1")
        srcPadB0Str = sgpr("AddressB+0")
        srcPadB1Str = sgpr("AddressB+1")
      else:
        # copy AddressA,B to srdA,B to undefine AddressA,B
        kStr += inst("s_mov_b64", sgpr("%sA+0"%self.sgprAddressStrAB, 2), sgpr("AddressA+0", 2), "copy addressA")
        kStr += inst("s_mov_b64", sgpr("%sB+0"%self.sgprAddressStrAB, 2), sgpr("AddressB+0", 2), "copy addressB")

    # self.groOffsetInMacroTile == 1 case, subtract pre-pad here
    if self.groOffsetInMacroTile:
      bpeOffsetA = 1 if self.tPA["glvw"] < 1 else self.tPA["bpe"] # glvw<1 case, no need to multiply bpe here
      prePad = self.srdShiftLeft["A"] * bpeOffsetA# leave room in case we have to pointer shift
      kStr += inst("s_sub_u32",  dstAddressA0Str, srcPadA0Str, prePad, "pre-pad to make room for possible pointer shift")
      kStr += inst("s_subb_u32",  dstAddressA1Str, srcPadA1Str, 0, "pre-pad to make room for possible pointer shift")
      bpeOffsetB = 1 if self.tPB["glvw"] < 1 else self.tPB["bpe"] # glvw<1 case, no need to multiply bpe here
      prePad = self.srdShiftLeft["B"] * bpeOffsetB # leave room in case we have to pointer shift
      kStr += inst("s_sub_u32",  dstAddressB0Str, srcPadB0Str, prePad, "pre-pad to make room for possible pointer shift")
      kStr += inst("s_subb_u32",  dstAddressB1Str, srcPadB1Str, 0, "pre-pad to make room for possible pointer shift")

    # undefine Offset sgpr (only for general batch)
    kStr += self.endLine
    if not kernel["ProblemType"]["StridedBatched"]:
      kStr += self.undefineSgpr("OffsetD")
      kStr += self.undefineSgpr("OffsetC")
      kStr += self.undefineSgpr("OffsetA")
      kStr += self.undefineSgpr("OffsetB")
    # undefine Address sgpr
    if self.releaseSgprAdressCD:
      kStr += self.undefineSgpr("AddressD")
      kStr += self.undefineSgpr("AddressC")
    if self.releaseSgprAdressAB:
      kStr += self.undefineSgpr("AddressA")
      kStr += self.undefineSgpr("AddressB")

    self.defineVariableSgprs(kernel)

    # Check alpha == 0, is done before kernel body
    # so if alpha/beta=Half, they haven't been converted to f32
    # This means we can use ComputeDataType as AlphaType (even <h,h,h,h,"h,h"> +"HPA")
    if self.do["ApplyAlpha"]:

      kStr += self.comment("Short circuit condition if Alpha == 0, then sumDims=0")
      endCheckLabel = "label_AlphaNonZero"
      branchNeeded = True
      if kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
        kStr += inst("v_cmp_eq_f64", self.vcc, sgpr("Alpha", 2), 0.0, "Alpha.real == 0.0 ?")
        kStr += inst("s_cbranch_vccz %s" % (endCheckLabel), "branch if Alpha.real != 0")
        kStr += inst("v_cmp_eq_f64", self.vcc, sgpr("Alpha+2", 2), 0.0, "Alpha.imag == 0.0 ?")
        kStr += inst("s_cbranch_vccz %s" % (endCheckLabel), "branch if Alpha.imag != 0")

      elif kernel["ProblemType"]["ComputeDataType"].isDouble():
        kStr += inst("v_cmp_eq_f64", self.vcc, sgpr("Alpha", 2), 0.0, "Alpha == 0.0 ?")
        kStr += inst("s_cbranch_vccz %s" % (endCheckLabel), "branch if Alpha != 0")

      elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
        kStr += inst("v_cmp_eq_f32", self.vcc, sgpr("Alpha"), 0.0, "Alpha.real == 0.0f ?")
        kStr += inst("s_cbranch_vccz %s" % (endCheckLabel), "branch if Alpha.real != 0")
        kStr += inst("v_cmp_eq_f32", self.vcc, sgpr("Alpha+1"), 0.0, "Alpha.imag == 0.0f ?")
        kStr += inst("s_cbranch_vccz %s" % (endCheckLabel), "branch if Alpha.imag != 0")

      # AlphaType is f32 or two-concated-f16, or two-concated-bf16(not support)
      elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
           kernel["ProblemType"]["ComputeDataType"].isHalf() or \
           kernel["ProblemType"]["ComputeDataType"].isBFloat16() or \
           kernel["ProblemType"]["ComputeDataType"].is8bitFloat():
        kStr += inst("v_cmp_eq_f32", self.vcc, sgpr("Alpha"), 0.0, "Alpha == 0.0f ?")
        kStr += inst("s_cbranch_vccz %s" % (endCheckLabel), "branch if alpha != 0")

      # AlphaType is int32
      else:
        kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), 0, "Alpha == 0 ?")
        #kStr += inst("s_cbranch_scc0 %s" % (endCheckLabel), "branch if alpha != 0")
        branchNeeded = False

      # Conditional set summation dimensions to 0 on SCC==1
      instStr = "s_mov_b32" if branchNeeded else "s_cmov_b32"
      for i in range(0, self.numSgprSizesSum):
        kStr += inst(instStr, sgpr("SizesSum+%u"%(i)), hex(0), "Set summation dim=0 if Alpha == 0")

      # Jump here if alpha is non-zero
      if branchNeeded:
        kStr += "%s:%s" % (endCheckLabel, self.endLine)

    for tc in ('A', 'B'):
      for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
        (freeDim, sumDim) = zp[:2]
        freeDimChar = globalParameters["IndexChars"][freeDim]
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        kStr += inst("s_lshl_b32", \
                     sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     "Bpe%sLog2"%tc, "")
        kStr += inst("s_lshl_b32", \
                     sgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     sgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     "Bpe%sLog2"%tc, "")

    if kernel["PersistentKernel"]:
      kStr += inst("s_mov_b32", sgpr("SerialWorkGroupIter"), sgpr("WorkGroup0"), "init SerialWorkGroupIter")
      # kStr += inst("s_mov_b32", sgpr("PersistentLoopIter"), 0, "init PersistentKernelLoop Iter")  # Back-up: not needed now

    if self.canOptimizePreLoopLWVmcnt:
      kStr += inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(1), "init PreLoopLWVmcntCase to 1")

    if kernel["MagicDivAlg"]==2:
      for magicName in self.sumMagicParms:
          kStr += inst("s_lshr_b32", sgpr("MagicAbitSize%s"%magicName), sgpr("MagicShiftSize%s"%magicName), 31,"extract abit")
          kStr += inst("s_and_b32",  sgpr("MagicShiftSize%s"%magicName), sgpr("MagicShiftSize%s"%magicName), hex(0x7fffffff), "remove abit")

      for idxChar in sorted(set(kernel["PackedC0IdxChars"][:-1] + kernel["PackedC1IdxChars"][:-1])):
          kStr += inst("s_lshr_b32", sgpr("MagicAbitSize%s"%idxChar), sgpr("MagicShiftSize%s"%idxChar), 31,"extract abit")
          kStr += inst("s_and_b32",  sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicShiftSize%s"%idxChar), hex(0x7fffffff), "remove abit")

    # setup rounding term if Fp16AltImpl is enabled, reusing sgpr to hold rounding term
    if kernel["ProblemType"]["Fp16AltImpl"] and kernel["ProblemType"]["Fp16AltImplRound"]:
      kStr += inst("s_mov_b32", sgpr("Fp16AltOffset"), "0x7FFF", "Set rounding coeff if Fp16AltImplNtz enabled")

    # setup src values for v_perm_b32
    lrvwTileMax = max(self.lrvwTileA, self.lrvwTileB)
    if lrvwTileMax > 1:
      if kernel["ProblemType"]["DataType"].numBytes() == 2:
        kStr += inst("s_mov_b32", sgpr("PackKForV0"), "0x05040100", "half pack value 0 for v_perm_b32")
        kStr += inst("s_mov_b32", sgpr("PackKForV1"), "0x07060302", "half pack value 1 for v_perm_b32")
      elif kernel["ProblemType"]["DataType"].numBytes() == 1:
        kStr += inst("s_mov_b32", sgpr("PackKForV0"), "0x0c0c0400", "byte pack value 0 for v_perm_b32")
        kStr += inst("s_mov_b32", sgpr("PackKForV1"), "0x0c0c0501", "byte pack value 1 for v_perm_b32")
        if lrvwTileMax > 2:
          kStr += inst("s_mov_b32", sgpr("PackKForV2"), "0x0c0c0602", "byte pack value 2 for v_perm_b32")
          kStr += inst("s_mov_b32", sgpr("PackKForV3"), "0x0c0c0703", "byte pack value 3 for v_perm_b32")

    ########################################
    # Debug Buffer
    if globalParameters["DebugKernel"]:
      kStr += self.comment("Debug Buffer")

      # nwg0 FIXME use NumWorkGroups0
      #kStr += self.assert_eq(vgpr(nwg0), sgpr("NumWorkGroups0")) # "bozo, remove me")
      nwg0 = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOut(1)
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += "// nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]-1), "MT0-1")
      kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr(tmpSgpr), "MT0-1")
      kStr += inst("_v_add_co_u32", vgpr(nwg0), self.vcc, sgpr("SizesFree+0"), \
          vgpr(tmpVgpr), "%s = size0+MT0-1"%vgpr(nwg0))
      kStr += vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpSgpr)
      self.vgprPool.checkIn(tmpVgpr)
      self.nipt = 16 # num integers per thread
      v = self.vgprPool.checkOut(3)
      kStr += inst("v_mov_b32", vgpr(v), sgpr("WorkGroup0"), "%s=wg0"%vgpr(v) )
      kStr += inst("v_mov_b32", vgpr(v+1), sgpr("WorkGroup1"), "%s=wg1"%vgpr(v+1) )
      kStr += inst("v_mul_lo_u32", vgpr(v+1), vgpr(v+1), vgpr(nwg0), \
          "%s=wg1*nwg0"%vgpr(v+1) )
      kStr += inst("_v_add_co_u32", vgpr(v), self.vcc, vgpr(v), vgpr(v+1), \
          "%s=wg1*nwg0+wg0"%vgpr(v) )
      kStr += staticMultiply(vgpr(v), vgpr(v), kernel["NumThreads"], sgpr(tmpSgpr))
      kStr += inst("_v_add_co_u32", vgpr(v), self.vcc, vgpr(v), vgpr("Serial"), \
          "%s=tid+NT*(wg1*nwg0+wg0)=serial"%vgpr(v) )
      kStr += inst("v_mul_lo_u32", vgpr(v), hex(self.nipt*4), vgpr(v), \
          "%s=serial*nipt*4"%vgpr(v) )
      kStr += inst("v_mov_b32", vgpr(v+1), 0, "")
      kStr += inst("_v_add_co_u32", vgpr("AddressDbg"), self.vcc, sgpr("AddressDbg"), \
          vgpr(v), "%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") )
      kStr += inst("v_mov_b32", vgpr(v+2), sgpr("AddressDbg+1"), "%s=AddressD1"%vgpr(v+2) )
      kStr += inst("_v_addc_co_u32", vgpr("AddressDbg+1"), self.vcc, vgpr(v+2), \
          vgpr(v+1), self.vcc, "%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") )
      kStr += inst("s_mov_b32", sgpr("DebugKernelItems"), 0, "")
      self.vgprPool.checkIn(v)
      self.vgprPool.checkIn(nwg0)

    if self.db["InitLds"]:
      kStr += self.initLds(kernel, self.initLdsValue)

    if kernel["CheckTensorDimAsserts"]:
      kStr += self.assert_multiple_b32(sgpr("SizesSum+%u"%(self.numSgprSizesSum-1)),
                kernel["AssertSummationElementMultiple"], 0x1001)
      kStr += self.assert_multiple_b32(sgpr("SizesFree+0"),
                kernel["AssertFree0ElementMultiple"], 0x1002)
      kStr += self.assert_multiple_b32(sgpr("SizesFree+1"),
                kernel["AssertFree1ElementMultiple"], 0x1003)

    return kStr

  ##############################################################################
  # Perform a magic division (mul by magic number and shift)
  # dest is two consec SGPR, used for intermediate temp as well as final result
  # result quotient returned in sgpr(dest,1)
  ##############################################################################
  def sMagicDiv(self, kernel, dest, dividend, magicNumber, magicShift):
    kStr = ""
    kStr += self.s_mul_u64_u32(sgpr(dest), sgpr(dest+1), dividend, magicNumber, "s_magic mul")
    kStr += inst("s_lshr_b64", sgpr(dest,2), sgpr(dest,2), magicShift, "sMagicDiv")
    return kStr

  ##############################################################################
  # Perform a sgpr version of magic division algo 2 (mul by magic number, Abit and shift)
  # dest is three consec SGPR, used for intermediate temp as well as final result
  # result quotient returned in sgpr(dest,1)
  ##############################################################################
  def sMagicDivAlg2(self, kernel, dest, dividend, magicNumber, magicShiftAbit):
    # dest+0: q,
    # dest+1: intermediate for magic div
    # dest+2: A tmpS to store the 'Abit' and the final Shift (use tmpS to save sgpr)
    tmpS = dest+2

    kStr = ""
    kStr += inst("s_mul_hi_u32", sgpr(dest+1), dividend, magicNumber, " s_magic mul, div alg 2")
    kStr += inst("s_lshr_b32", sgpr(tmpS), magicShiftAbit, 31, " tmpS = extract abit")                              # tmpS = MagicAbit
    kStr += inst("s_mul_i32", sgpr(dest), dividend, sgpr(tmpS), " s_magic mul, div alg 2")
    kStr += inst("s_add_u32", sgpr(dest), sgpr(dest), sgpr(dest+1), "")

    kStr += inst("s_and_b32",  sgpr(tmpS), magicShiftAbit, hex(0x7fffffff), " tmpS = remove abit to final shift")   # tmpS = MagicShift
    kStr += inst("s_lshr_b32", sgpr(dest), sgpr(dest), sgpr(tmpS), " sMagicDiv Alg 2")
    return kStr

  def extractPackedCoord1ToRowStart(self, kernel, packedC1, packedCoordVgpr, storeChar):
    # calculate packed rowStart vgpr
    # vgprTmp assignments:
    #   - tmp+0 is the incoming packed coordinate 1, used on replay too
    #   - tmp+1 is DIV output
    #   - tmp+2 is scratch
    #   - tmp+3 holds thread rowStart free1 offset
    kStr = ""
    tmpV0 = self.vgprPool.checkOut(4)
    tmpV1 = tmpV0 + 1
    tmpV2 = tmpV0 + 2
    tmpV3 = tmpV0 + 3

    #assert(kernel["LdcEqualsLdd"])
    kStr += inst("v_mov_b32", vgpr(tmpV0), vgpr(packedCoordVgpr),  "copy coord1 then unpack")
    for i,idx in enumerate(packedC1[:-1]):
      idxChar= globalParameters["IndexChars"][idx]
      kStr += self.comment1("extract %s"%self.sizeRef(idx))
      kStr += "V_MAGIC_DIV %s, %s, %s, %s, %s\n" % \
               (tmpV1, vgpr(tmpV0), sgpr("MagicNumberSize%s"%idxChar), \
                sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0")
      kStr += inst("v_mul_lo_u32", vgpr(tmpV2), vgpr(tmpV1), self.sizeRef(idx), "remainder part 1")
      kStr += inst("_v_sub_u32", vgpr(tmpV2), vgpr(tmpV0), vgpr(tmpV2), "remainder part 2")
      if i==0:
        kStr += inst("v_mul_lo_u32", vgpr(tmpV3), vgpr(tmpV2), \
                  self.strideRef(storeChar, idx), "addrCalc <- scaled extracted dim")
      else:
        kStr += inst("v_mul_lo_u32", vgpr(tmpV2), vgpr(tmpV2), \
                  self.strideRef(storeChar, idx), "scale extracted dim")
        kStr += inst("_v_add_u32", vgpr(tmpV3), vgpr(tmpV3), \
                  vgpr(tmpV2), "addrCalc += scaled extracted dim ")

      if i < len(packedC1)-2:
        kStr += inst("v_mov_b32", vgpr(tmpV0), vgpr(tmpV1), \
                  "Copy remaining bits for next divide")

    kStr += self.comment1("extract final %s"%self.sizeRef(packedC1[-1]))
    kStr += inst("v_mul_lo_u32", vgpr(tmpV2), vgpr(tmpV1), \
              self.strideRef(storeChar, packedC1[-1]), "scale final extracted dim")
    kStr += inst("_v_add_u32", vgpr(self.coutRowPtr), vgpr(tmpV3), \
              vgpr(tmpV2), "rowStart += scaled extracted dim ")

    self.vgprPool.checkIn(tmpV0)
    return kStr

  ##############################################################################
  # Open Persistent Loop
  # init iteration counter, define loop target
  ##############################################################################
  def openPersistentLoop(self, kernel):
    kStr = ""
    if kernel["PersistentKernel"] or kernel["StreamK"]:
      if kernel["StreamK"]:
        # Workload calculations
        if kernel["StreamKXCCMapping"]:
          sXCC = self.sgprPool.checkOut(1, "XCC", preventOverflow=0)
          sGridC = self.sgprPool.checkOut(1, "sGridC", preventOverflow=0)
          sGridF = self.sgprPool.checkOut(1, "sGridF", preventOverflow=0)
          sGridM = self.sgprPool.checkOut(1, "sGridM", preventOverflow=0)
          stmp = None
          sqtmp = None
          divisor = kernel["StreamKXCCMapping"]
          if ((divisor & (divisor - 1)) != 0): # Need temp registers if not power of 2
            stmp = self.sgprPool.checkOut(3, "stmp", preventOverflow=0)
            sqtmp = stmp + 2
          # sGridC = ceil(grid / xccm)
          kStr += inst("s_add_u32", sgpr(sXCC), sgpr("skGrid"), kernel["StreamKXCCMapping"] - 1, "ceil(grid/xccm)")
          kStr += scalarStaticDivideAndRemainder(sGridC, None, sGridC, kernel["StreamKXCCMapping"], stmp, doRemainder=0)
          # sGridF = floor(grid / xccm)
          # sGridM = grid % xccm
          kStr += scalarStaticDivideAndRemainder(sGridF, sGridM, "skGrid", kernel["StreamKXCCMapping"], stmp)
          # sXCC = wg0 % xccm
          # sqtmp is temp register for quotient for non-power-of-2 case
          # sqtmp overlaps temp registers, works in this case and output is discarded
          kStr += scalarStaticDivideAndRemainder(sqtmp, sXCC, "WorkGroup0", kernel["StreamKXCCMapping"], stmp, doRemainder=2)
          # Check if current XCC requires a remainder WG or not
          kStr += inst("s_cmp_lt_u32", sgpr(sXCC), sgpr(sGridM), "XCCM < Remainder")
          kStr += inst("s_cselect_b32", sgpr(sGridC), sgpr(sGridC), sgpr(sGridF), "Select multiplier")
          kStr += inst("s_cselect_b32", sgpr(sGridM), 0, sgpr(sGridM), "Select remainder")
          # WG = floor(wg0 / xccm) * xccm + XCCoffset + optional remainder
          kStr += scalarStaticDivideAndRemainder("WorkGroup0", None, "WorkGroup0", kernel["StreamKXCCMapping"], stmp, doRemainder=0)
          kStr += inst("s_mul_i32", sgpr(sXCC), sgpr(sXCC), sgpr(sGridC), "XCC group id")
          kStr += inst("s_add_u32", sgpr("WorkGroup0"), sgpr("WorkGroup0"), sgpr(sXCC), "Add XCC group offset")
          kStr += inst("s_add_u32", sgpr("WorkGroup0"), sgpr("WorkGroup0"), sgpr(sGridM), "Add remainder offset")
          self.sgprPool.checkIn(sXCC)
          self.sgprPool.checkIn(sGridC)
          self.sgprPool.checkIn(sGridF)
          self.sgprPool.checkIn(sGridM)
          if stmp is not None:
            self.sgprPool.checkIn(stmp)

        kStr += inst("s_mov_b32", sgpr("StreamKIdx"), sgpr("WorkGroup0"), "Save original StreamK index")
        if kernel["StreamK"] == 1: # Basic SK
          kStr += inst("s_mul_i32", sgpr("StreamKIter"), sgpr("StreamKIdx"), sgpr("SKItersPerWG"), "StreamK starting iteration")
          kStr += inst("s_add_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIter"), sgpr("SKItersPerWG"), "StreamK ending iteration")
          kStr += inst("s_min_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIterEnd"), sgpr("TotalIters"), "Cap ending iter at total iters")
          kStr += inst("s_cmp_lt_u32", sgpr("StreamKIter"), sgpr("StreamKIterEnd"), "Make sure there's work to do")
          kStr += self.longBranchScc0("label_%04u" % (self.getLabelNum("KernelEnd")), positiveOnly=True)
          # kStr += inst("s_cbranch_scc0", "label_%04u" % (self.getLabelNum("KernelEnd")), "edge case that work doesn't divide well")
          kStr += self.undefineSgpr("TotalIters")
        elif kernel["StreamK"] == 2: # Two-tile SK
          # iter count after all extra iters have been distributed
          kStr += inst("s_mul_i32", sgpr("StreamKIter"), sgpr("StreamKIdx"), sgpr("SKItersPerWG"), "StreamK starting iteration (case: after extra iters)")
          kStr += inst("s_add_u32", sgpr("StreamKIter"), sgpr("StreamKIter"), sgpr("skExtraIters"), "Add extra iters")
          kStr += inst("s_add_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIter"), sgpr("SKItersPerWG"), "StreamK ending iteration (case: after extra iters)")
          # iter count before all extra iters have been distributed
          # stmp+1 = SKItersPerWG + 1 extra iteration
          sIter = self.sgprPool.checkOut(2, "SKIter", preventOverflow=0)
          kStr += inst("s_add_u32", sgpr(sIter+1), sgpr("SKItersPerWG"), 1, "Spread out extra iterations")
          kStr += inst("s_mul_i32", sgpr(sIter), sgpr("StreamKIdx"), sgpr(sIter+1), "StreamK starting iteration (case: before extra iters)")
          kStr += inst("s_add_u32", sgpr(sIter+1), sgpr(sIter), sgpr(sIter+1), "StreamK ending iteration (case: before extra iters)")
          # select correct start/end iteration index
          kStr += inst("s_cmp_lt_u32", sgpr("StreamKIdx"), sgpr("skExtraIters"), "Check if lane gets an extra iteration")
          kStr += inst("s_cselect_b32", sgpr("StreamKIter"), sgpr(sIter), sgpr("StreamKIter"), "Set start iter")
          kStr += inst("s_cselect_b32", sgpr("StreamKIterEnd"), sgpr(sIter+1), sgpr("StreamKIterEnd"), "Set end iter")
          self.sgprPool.checkIn(sIter)
          # clamp to end of sk iterations
          # TODO maybe remove clamp, since extra iters code should guarantee total iterations match
          stmp = self.sgprPool.checkOut(1, "TotalSKIters", preventOverflow=0)
          kStr += inst("s_mul_i32", sgpr(stmp), sgpr("skTiles"), sgpr("ItersPerTile"), "Total SK iters")
          kStr += inst("s_min_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIterEnd"), sgpr(stmp), "Cap ending iter at total SK iters")
          self.sgprPool.checkIn(stmp)
          # check if this WG has no work to do
          kStr += inst("s_cmp_lt_u32", sgpr("StreamKIter"), sgpr("TotalIters"), "Make sure there's work to do")
          kStr += self.longBranchScc0("label_%04u" % (self.getLabelNum("KernelEnd")), positiveOnly=True)
        elif kernel["StreamK"] == 3: # Two-tile SK after DP
          # Do DP tiles before SK
          skInitDone = self.getNamedLabelUnique("SK_InitDone")
          kStr += inst("s_mul_i32", sgpr("StreamKIter"), sgpr("StreamKIdx"), sgpr("ItersPerTile"), "DP starting iteration (case: DP work to do)")
          kStr += inst("s_mov_b32", sgpr("StreamKIterEnd"), sgpr("TotalIters"), "DP ending iteration (case: only DP work to do)")
          stmp = self.sgprPool.checkOut(1, "TotalSKIters", preventOverflow=0)
          kStr += inst("s_mul_i32", sgpr(stmp), sgpr("skTiles"), sgpr("ItersPerTile"), "Total SK iters")
          kStr += inst("s_cmp_lt_u32", sgpr(stmp), sgpr("TotalIters"), "Check if there are DP tiles to do")
          kStr += inst("s_cbranch_scc1 %s" % skInitDone, "Done init")
          self.sgprPool.checkIn(stmp)

          # If there are no DP tiles to do, regular SK init
          # iter count after all extra iters have been distributed
          kStr += inst("s_mul_i32", sgpr("StreamKIter"), sgpr("StreamKIdx"), sgpr("SKItersPerWG"), "StreamK starting iteration (case: after extra iters)")
          kStr += inst("s_add_u32", sgpr("StreamKIter"), sgpr("StreamKIter"), sgpr("skExtraIters"), "Add extra iters")
          kStr += inst("s_add_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIter"), sgpr("SKItersPerWG"), "StreamK ending iteration (case: after extra iters)")
          # iter count before all extra iters have been distributed
          # stmp+1 = SKItersPerWG + 1 extra iteration
          sIter = self.sgprPool.checkOut(2, "SKIter", preventOverflow=0)
          kStr += inst("s_add_u32", sgpr(sIter+1), sgpr("SKItersPerWG"), 1, "Spread out extra iterations")
          kStr += inst("s_mul_i32", sgpr(sIter), sgpr("StreamKIdx"), sgpr(sIter+1), "StreamK starting iteration (case: before extra iters)")
          kStr += inst("s_add_u32", sgpr(sIter+1), sgpr(sIter), sgpr(sIter+1), "StreamK ending iteration (case: before extra iters)")
          # select correct start/end iteration index
          kStr += inst("s_cmp_lt_u32", sgpr("StreamKIdx"), sgpr("skExtraIters"), "Check if lane gets an extra iteration")
          kStr += inst("s_cselect_b32", sgpr("StreamKIter"), sgpr(sIter), sgpr("StreamKIter"), "Set start iter")
          kStr += inst("s_cselect_b32", sgpr("StreamKIterEnd"), sgpr(sIter+1), sgpr("StreamKIterEnd"), "Set end iter")
          self.sgprPool.checkIn(sIter)
          # clamp to end of sk iterations
          # TODO maybe remove clamp, since extra iters code should guarantee total iterations match
          stmp = self.sgprPool.checkOut(1, "TotalSKIters", preventOverflow=0)
          kStr += inst("s_mul_i32", sgpr(stmp), sgpr("skTiles"), sgpr("ItersPerTile"), "Total SK iters")
          kStr += inst("s_min_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIterEnd"), sgpr(stmp), "Cap ending iter at total SK iters")
          self.sgprPool.checkIn(stmp)

          kStr += "%s:\n" % (skInitDone)
          # check if this WG has no work to do
          kStr += inst("s_cmp_lt_u32", sgpr("StreamKIter"), sgpr("TotalIters"), "Make sure there's work to do")
          kStr += self.longBranchScc0("label_%04u" % (self.getLabelNum("KernelEnd")), positiveOnly=True)

      kStr += self.comment3("Persistent Loop Start")
      kStr += self.getLabelDef("PersistentLoopStart")
      # kStr += inst("s_add_u32", sgpr("PersistentLoopIter"), sgpr("PersistentLoopIter"), hex(1), "Inc PersistentLoop Iter")   # Back-up: not needed now
      #kStr += str(Code.WaitCnt(self.version, 0,0,"wait for outstanding stores"))

    return kStr

  ##############################################################################
  # Global Read Addresses: WorkGroup
  ##############################################################################
  def graWorkGroup(self, kernel, isPap):
    kStr = ""

    if kernel["StreamK"]:
      # StreamK workgroup mapping
      stmp = self.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=0)
      # Always reset pointers to handle odd-exit case which moves LRO to the upper bank
      if not self.prefetchAcrossPersistent and kernel["PrefetchGlobalRead"]:
        kStr += self.localReadResetOffsets(kernel, self.tPA)
        kStr += self.localReadResetOffsets(kernel, self.tPB)

      kStr += self.comment1("StreamK calculate tile idx and map to WG")

      # stmp = tile index
      kStr += self.sMagicDivAlg2(kernel, stmp, sgpr("StreamKIter"), sgpr("MagicNumberItersPerTile"), sgpr("MagicShiftItersPerTile"))
      # stmp+1 = tile start
      kStr += inst("s_mul_i32", sgpr(stmp+1), sgpr(stmp), sgpr("ItersPerTile"), "Tile start iteration")
      # stmp+2 = tile end
      kStr += inst("s_add_u32", sgpr(stmp+2), sgpr(stmp+1), sgpr("ItersPerTile"), "Tile end iteration")
      # local start
      kStr += inst("s_sub_u32", sgpr("StreamKLocalStart"), sgpr("StreamKIter"), sgpr(stmp+1), "Local iteration start")
      # local end (SK tile)
      kStr += inst("s_min_u32", sgpr("StreamKLocalEnd"), sgpr("StreamKIterEnd"), sgpr(stmp+2), "1. (Local) iteration end (SK tile)")
      kStr += inst("s_sub_u32", sgpr("StreamKLocalEnd"), sgpr("StreamKLocalEnd"), sgpr(stmp+1), "2. Local iteration end (SK tile)")

      if kernel["StreamK"] == 2: # Two-tile algorithm
        # local end (DP tile)
        # TODO This line isnt needed?
        kStr += inst("s_sub_u32", sgpr(stmp+3), sgpr(stmp+2), sgpr(stmp+1), "Local iteration end (DP tile)")
        # select correct local end
        kStr += inst("s_cmp_lt_u32", sgpr("StreamKIter"), sgpr("StreamKIterEnd"), "Check if in SK or DP section")
        kStr += inst("s_cselect_b32", sgpr("StreamKLocalEnd"), sgpr("StreamKLocalEnd"), sgpr(stmp+3), "Apply SK or DP end iteration")

        # Increment StreamK iteration
        # If moving from SK to DP, next iteration is first DP
        # stmp = offset to first DP tile
        kStr += inst("s_mul_i32", sgpr(stmp+3), sgpr("skTiles"), sgpr("ItersPerTile"), "Offset to first DP tile")
        kStr += inst("s_mul_i32", sgpr(stmp+1), sgpr("StreamKIdx"), sgpr("ItersPerTile"), "WG tile offset")
        kStr += inst("s_add_u32", sgpr(stmp+3), sgpr(stmp+3), sgpr(stmp+1), "DP start offset + WG offset")
        # If already in DP, add dpShift
        kStr += inst("s_mul_i32", sgpr(stmp+1), sgpr("skGrid"), sgpr("ItersPerTile"), "DP iterations shift")
        kStr += inst("s_add_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("StreamKIter"), "Add DP shift")
        # Save DP iter in stmp
        kStr += inst("s_cmp_lt_u32", sgpr("StreamKIter"), sgpr("StreamKIterEnd"), "Check if in SK or DP section")
        kStr += inst("s_cselect_b32", sgpr(stmp+3), sgpr(stmp+3), sgpr(stmp+1), "Select first DP tile, or add DP shift")
        # If staying in SK portion, next iteration is stmp+2
        kStr += inst("s_cmp_lt_u32", sgpr(stmp+2), sgpr("StreamKIterEnd"), "Check if there are more SK tiles")
        kStr += inst("s_cselect_b32", sgpr("StreamKIter"), sgpr(stmp+2), sgpr(stmp+3), "Select next SK or DP tile")

      elif kernel["StreamK"] == 3: # Two-tile SK after DP
        skUpdateDone = self.getNamedLabelUnique("SK_UpdateDone")
        # stmp+3 = Offset to first SK tile
        kStr += inst("s_mul_i32", sgpr(stmp+3), sgpr("skTiles"), sgpr("ItersPerTile"), "Total SK iters")
        kStr += inst("s_sub_u32", sgpr(stmp+3), sgpr("TotalIters"), sgpr(stmp+3), "Offset to first SK tile")
        # If in DP, add dpShift
        kStr += inst("s_mul_i32", sgpr(stmp+1), sgpr("skGrid"), sgpr("ItersPerTile"), "DP iterations shift")
        kStr += inst("s_add_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("StreamKIter"), "Add DP shift")
        # if stmp+1 < stmp+3, continue DP (add dpShift)
        kStr += inst("s_cmp_lt_u32", sgpr(stmp+1), sgpr(stmp+3), "Check if still in DP section")
        kStr += inst("s_cbranch_scc1 %s" % skUpdateDone, "Done update")
        # if StreamKIter >= stmp+3, continue SK (add skShift?)
        kStr += inst("s_mov_b32", sgpr(stmp+1), sgpr(stmp+2), "SK iterations shift")
        kStr += inst("s_cmp_le_u32", sgpr(stmp+3), sgpr("StreamKIter"), "Check if continuing in SK section")
        kStr += inst("s_cbranch_scc1 %s" % skUpdateDone, "Done update")
        # if stmp+1 > stmp+3 and StreamKIter < stmp+3, switch from DP to SK (add dpShift)
        # iter count after all extra iters have been distributed
        kStr += inst("s_mul_i32", sgpr("StreamKIter"), sgpr("StreamKIdx"), sgpr("SKItersPerWG"), "StreamK starting iteration (case: after extra iters)")
        kStr += inst("s_add_u32", sgpr("StreamKIter"), sgpr("StreamKIter"), sgpr("skExtraIters"), "Add extra iters")
        kStr += inst("s_add_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIter"), sgpr("SKItersPerWG"), "StreamK ending iteration (case: after extra iters)")
        # iter count before all extra iters have been distributed
        # stmp+1 = SKItersPerWG + 1 extra iteration
        sIter = self.sgprPool.checkOut(2, "SKIter", preventOverflow=0)
        kStr += inst("s_add_u32", sgpr(sIter+1), sgpr("SKItersPerWG"), 1, "Spread out extra iterations")
        kStr += inst("s_mul_i32", sgpr(sIter), sgpr("StreamKIdx"), sgpr(sIter+1), "StreamK starting iteration (case: before extra iters)")
        kStr += inst("s_add_u32", sgpr(sIter+1), sgpr(sIter), sgpr(sIter+1), "StreamK ending iteration (case: before extra iters)")
        # select correct start/end iteration index
        kStr += inst("s_cmp_lt_u32", sgpr("StreamKIdx"), sgpr("skExtraIters"), "Check if lane gets an extra iteration")
        kStr += inst("s_cselect_b32", sgpr("StreamKIter"), sgpr(sIter), sgpr("StreamKIter"), "Set start iter")
        kStr += inst("s_cselect_b32", sgpr("StreamKIterEnd"), sgpr(sIter+1), sgpr("StreamKIterEnd"), "Set end iter")
        self.sgprPool.checkIn(sIter)
        kStr += inst("s_add_u32", sgpr(stmp+1), sgpr("StreamKIter"), sgpr(stmp+3), "Offset to start of SK section")
        kStr += inst("s_add_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIterEnd"), sgpr(stmp+3), "Offset to start of SK section")
        # clamp to end of iterations
        # TODO maybe remove clamp, since extra iters code should guarantee total iterations match
        kStr += inst("s_min_u32", sgpr("StreamKIterEnd"), sgpr("StreamKIterEnd"), sgpr("TotalIters"), "Cap ending iter at total SK iters")
        # check if this WG has no work to do
        # TODO Shouldn't need this check!
        kStr += inst("s_cmp_lt_u32", sgpr("StreamKIter"), sgpr("TotalIters"), "Make sure there's work to do")
        kStr += self.longBranchScc0("label_%04u" % (self.getLabelNum("KernelEnd")), positiveOnly=True, tmpSgpr=stmp) # reuse tmp

        # If in SK, next iteration is stmp+2
        # Increment StreamK iteration
        kStr += "%s:\n" % (skUpdateDone)
        kStr += inst("s_mov_b32", sgpr("StreamKIter"), sgpr(stmp+1), "Store current iteration")

      else: # Basic Stream-K
        # Increment StreamK iteration
        kStr += inst("s_mov_b32", sgpr("StreamKIter"), sgpr(stmp+2), "Increment StreamK Iteration")

      # Map StreamK tile index to wg0/1
      kStr += self.comment1("Map StreamK tile index to wg0/1/2")
      kStr += self.sMagicDivAlg2(kernel, stmp+1, sgpr(stmp), sgpr("MagicNumProblemNumGroupTiles0By1"), sgpr("MagicShiftProblemNumGroupTiles0By1"))
      kStr += inst("s_mov_b32", sgpr("WorkGroup2"), sgpr(stmp+1), "wg2 = Tile Idx / problemNumGroupTiles0By1")
      kStr += inst("s_mul_i32", sgpr(stmp+1), sgpr(stmp+1), sgpr("NumWorkGroups0"), "remainder part 1 : quotient * divisor")
      kStr += inst("s_mul_i32", sgpr(stmp+1), sgpr(stmp+1), sgpr("NumWorkGroups1"), "remainder part 1 : quotient * divisor")
      kStr += inst("s_sub_u32", sgpr(stmp), sgpr(stmp), sgpr(stmp+1), "remainder")
      kStr += self.sMagicDivAlg2(kernel, stmp+1, sgpr(stmp), sgpr("MagicNumberProblemNumGroupTiles0"), sgpr("MagicShiftProblemNumGroupTiles0"))
      kStr += inst("s_mov_b32", sgpr("WorkGroup1"), sgpr(stmp+1), "wg1 = Tile Idx / problemNumGroupTiles0")
      kStr += inst("s_mul_i32", sgpr("WorkGroup0"), sgpr(stmp+1), sgpr("NumWorkGroups0"), "remainder part 1 : quotient * divisor")
      kStr += inst("s_sub_u32", sgpr("WorkGroup0"), sgpr(stmp), sgpr("WorkGroup0"), "wg0 = Tile Idx % problemNumGroupTiles0")

      kStr += "\n"

      self.sgprPool.checkIn(stmp)
      # del stmpRef

    if kernel["PersistentKernel"]:
      stmp = self.sgprPool.checkOutAligned(8, 4, "PKMappingTemp", preventOverflow=0)
      # Always reset pointers to handle odd-exit case which moves LRO to the upper bank
      if not self.prefetchAcrossPersistent and kernel["PrefetchGlobalRead"]:
        kStr += self.localReadResetOffsets(kernel, self.tPA)
        kStr += self.localReadResetOffsets(kernel, self.tPB)

      if kernel["PersistentKernelAlongBatch"]:
        # re-mapping WG2 to WGKSerial -> wg2
        # re-mapping SerialWorkGroupIter to WGIJSerial -> wg0/1
        kStr += self.comment1("compute SerialWorkGroupIter / problemNumGroupTiles0x1 (aka nWG0*nWG1)")
        kStr += self.sMagicDivAlg2(kernel, stmp, sgpr("SerialWorkGroupIter"), sgpr("MagicNumProblemNumGroupTiles0By1"), sgpr("MagicShiftProblemNumGroupTiles0By1"))
        kStr += inst("s_mov_b32", sgpr("WGKSerial"), sgpr(stmp), "wgKSerial = SerialWorkGroupIter / problemNumGroupTiles0x1")
        kStr += inst("s_mul_i32", sgpr("WGIJSerial"), sgpr(stmp)        , sgpr("NumWorkGroups0"), "for remainder: get quotient * NumWorkGroups0")
        kStr += inst("s_mul_i32", sgpr("WGIJSerial"), sgpr("WGIJSerial"), sgpr("NumWorkGroups1"), "for remainder: get quotient * NumWorkGroups0 * NumWorkGroups1")
        kStr += inst("s_sub_u32", sgpr("WGIJSerial"), sgpr("SerialWorkGroupIter"), sgpr("WGIJSerial"), "wgIJSerial = SerialWorkGroupIter % problemNumGroupTiles0x1")
        # WGIJSerial -> wg0/1
        kStr += self.comment1("compute WGIJSerial / problemNumGroupTiles0 (aka numWorkGroups0)")
        kStr += self.sMagicDivAlg2(kernel, stmp, sgpr("WGIJSerial"), sgpr("MagicNumberProblemNumGroupTiles0"), sgpr("MagicShiftProblemNumGroupTiles0"))
        kStr += inst("s_mov_b32", sgpr("WorkGroup1"), sgpr(stmp), "wg1 = WGIJSerial / problemNumGroupTiles0")
        kStr += inst("s_mul_i32", sgpr("WorkGroup0"), sgpr(stmp), sgpr("NumWorkGroups0"), "remainder part 1 : quotient * divisor")
        kStr += inst("s_sub_u32", sgpr("WorkGroup0"), sgpr("WGIJSerial"), sgpr("WorkGroup0"), "wg0 = WGIJSerial % problemNumGroupTiles0")

        # general batch
        if not kernel["ProblemType"]["StridedBatched"]:
          if len(kernel["ProblemType"]["IndicesBatch"]) > 0:
            kStr += self.endLine
            kStr += inst("_s_load_b256", sgpr("AddressD", 8), sgpr("KernArgAddress",2), hex(self.argAddressOffset), "reload DCAB address")
            kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for reload DCAB address")
            kStr += self.loadBatchedAddress(kernel, "WGKSerial", stmp)
            kStr += inst("_s_load_b256", sgpr(stmp, 8), sgpr("KernArgAddress",2), hex(self.argOffsetOffset),  "reload DCAB Offset")
            kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait global buffer adress ready")

            if not kernel["_GlobalAccumulation"]:
              kStr += inst("s_lshl_b64", sgpr(stmp+0, 2), sgpr(stmp+0, 2), hex(log2(self.bpeCexternal)), "elements offset to bytes offset")
              kStr += inst("s_add_u32",  sgpr("AddressD+0"), sgpr("AddressD+0"), sgpr(stmp + 0), "add offset to buffer address")
              kStr += inst("s_addc_u32", sgpr("AddressD+1"), sgpr("AddressD+1"), sgpr(stmp + 1), "add offset to buffer address")

              kStr += inst("s_lshl_b64", sgpr(stmp+2, 2), sgpr(stmp+2, 2), hex(log2(self.bpeCexternal)), "elements offset to bytes offset")
              kStr += inst("s_add_u32",  sgpr("AddressC+0"), sgpr("AddressC+0"), sgpr(stmp + 2), "add offset to buffer address")
              kStr += inst("s_addc_u32", sgpr("AddressC+1"), sgpr("AddressC+1"), sgpr(stmp + 3), "add offset to buffer address")

            kStr += inst("s_lshl_b64", sgpr(stmp+4, 2), sgpr(stmp+4, 2), hex(log2(self.bpeAB)), "elements offset to bytes offset")
            kStr += inst("s_add_u32",  sgpr("AddressA+0"), sgpr("AddressA+0"), sgpr(stmp + 4), "add offset to buffer address")
            kStr += inst("s_addc_u32", sgpr("AddressA+1"), sgpr("AddressA+1"), sgpr(stmp + 5), "add offset to buffer address")

            kStr += inst("s_lshl_b64", sgpr(stmp+6, 2), sgpr(stmp+6, 2), hex(log2(self.bpeAB)), "elements offset to bytes offset")
            kStr += inst("s_add_u32",  sgpr("AddressB+0"), sgpr("AddressB+0"), sgpr(stmp + 6), "add offset to buffer address")
            kStr += inst("s_addc_u32", sgpr("AddressB+1"), sgpr("AddressB+1"), sgpr(stmp + 7), "add offset to buffer address")

            if self.groOffsetInMacroTile:
              bpeOffsetA = 1 if self.tPA["glvw"] < 1 else self.tPA["bpe"] # glvw<1 case, no need to multiply bpe here
              prePad = self.srdShiftLeft["A"] * bpeOffsetA # leave room in case we have to pointer shift
              kStr += inst("s_sub_u32",  sgpr("AddressA+0"), sgpr("AddressA+0"), prePad, "pre-pad to make room for possible pointer shift")
              kStr += inst("s_subb_u32",  sgpr("AddressA+1"), sgpr("AddressA+1"), 0, "pre-pad to make room for possible pointer shift")
              bpeOffsetB = 1 if self.tPB["glvw"] < 1 else self.tPB["bpe"] # glvw<1 case, no need to multiply bpe here
              prePad = self.srdShiftLeft["B"] * bpeOffsetB # leave room in case we have to pointer shift
              kStr += inst("s_sub_u32",  sgpr("AddressB+0"), sgpr("AddressB+0"), prePad, "pre-pad to make room for possible pointer shift")
              kStr += inst("s_subb_u32",  sgpr("AddressB+1"), sgpr("AddressB+1"), 0, "pre-pad to make room for possible pointer shift")

      else:
        # SerialWorkGroupIter wg0/1
        kStr += self.comment1("compute SerialWorkGroupIter / problemNumGroupTiles0 (aka numWorkGroups0)")
        kStr += self.sMagicDivAlg2(kernel, stmp, sgpr("SerialWorkGroupIter"), sgpr("MagicNumberProblemNumGroupTiles0"), sgpr("MagicShiftProblemNumGroupTiles0"))
        kStr += inst("s_mov_b32", sgpr("WorkGroup1"), sgpr(stmp), "wg1 = SerialWorkGroupIter / problemNumGroupTiles0")
        kStr += inst("s_mul_i32", sgpr("WorkGroup0"), sgpr(stmp), sgpr("NumWorkGroups0"), "remainder part 1 : quotient * divisor")
        kStr += inst("s_sub_u32", sgpr("WorkGroup0"), sgpr("SerialWorkGroupIter"), sgpr("WorkGroup0"), "wg0 = SerialWorkGroupIter % problemNumGroupTiles0")

      #kStr += self.assert_ne(sgpr("SerialWorkGroupIter"), 2)
      kStr += "\n"
      self.sgprPool.checkIn(stmp)

    kStr += self.comment1("graWorkGroup mapping")
    if kernel["GlobalSplitU"] > 1:
      if kernel["GlobalSplitUWorkGroupMappingRoundRobin"]:
        # gsuSumIdx = wg1 / nwg1
        # wg1       = wg1 % nwg1

        # nwg1
        nwg1 = self.vgprPool.checkOut(1, "nwg1", self.preventVgprOverflowDuringNewTile)
        quotient = self.vgprPool.checkOut(1, "quotient", self.preventVgprOverflowDuringNewTile)
        tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
        tmpSgpr = self.sgprPool.checkOut(1, "GSUMappingTemp", preventOverflow=0)
        kStr += "// GSU-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
            % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
        kStr += inst("v_mov_b32", vgpr(nwg1), sgpr("SizesFree+1"), "")
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile1"]-1), "")
        kStr += inst("_v_add_co_u32", vgpr(nwg1), self.vcc, sgpr(tmpSgpr), vgpr(nwg1), \
            "%s = size1+MT1-1"%vgpr(nwg1))
        kStr += vectorStaticDivide(quotient, nwg1, kernel["MacroTile1"], tmpSgpr)
        self.vgprPool.checkIn(nwg1)
        nwg1 = quotient

        # wg1
        wg1 = self.vgprPool.checkOut(1, "wg1", self.preventVgprOverflowDuringNewTile)
        kStr += inst("v_mov_b32", vgpr(wg1), sgpr("WorkGroup1"), "wg1")

        # gsuSumIdx = wg1 / nwg1
        # wg1       = wg1 % nwg1
        quotient = self.vgprPool.checkOut(1, "quotient", self.preventVgprOverflowDuringNewTile)
        remainder = self.vgprPool.checkOut(1, "remainer", self.preventVgprOverflowDuringNewTile)
        tmpVgpr1 = self.vgprPool.checkOut(1, "tmpVgpr1", self.preventVgprOverflowDuringNewTile)
        dividend = wg1
        divisor = nwg1
        kStr += "DYNAMIC_VECTOR_DIVIDE %s %s %s %s %s %s %s%s" \
            % ( quotient, remainder, dividend, divisor, \
            tmpVgpr, tmpVgpr1, tmpSgpr, self.endLine )

        # move vgprs into sgprs
        kStr += inst("v_readfirstlane_b32", sgpr("GSUSumIdx"), \
            vgpr(quotient), "")
        kStr += inst("v_readfirstlane_b32", sgpr("WorkGroup1"), \
            vgpr(remainder), "")
        self.vgprPool.checkIn(tmpVgpr)
        self.vgprPool.checkIn(tmpVgpr1)
        self.vgprPool.checkIn(nwg1)
        self.vgprPool.checkIn(wg1)
        self.vgprPool.checkIn(quotient)
        self.vgprPool.checkIn(remainder)
        self.sgprPool.checkIn(tmpSgpr)
      else:
        kStr += "// GSU-not-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
            % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)

        # gsuSumIdx = wg1 % GSU
        # wg1       = wg1 / GSU
        tmpSgpr = self.sgprPool.checkOutAligned(3, 2, "GSUMappingTemp", preventOverflow=0)
        divisor = tmpSgpr+2
        kStr += inst("s_mov_b32", sgpr(divisor), sgpr("WorkGroup1"), \
            "copying for divisor")

        #tmp = self.vgprPool.checkOut(1)

        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
        #kStr += dump(vgpr(tmp)) # numerator

        kStr += scalarStaticDivideAndRemainder("WorkGroup1", "GSUSumIdx", \
            divisor, kernel["GlobalSplitU"], tmpSgpr, 1)

        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
        #kStr += dump(vgpr(tmp)) # quotient
        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("GSUSumIdx"), "gsusumidx")
        #kStr += dump(vgpr(tmp)) # remainder
        #self.vgprPool.checkIn(tmp)
        #kStr += "s_endpgm\n"
        self.sgprPool.checkIn(tmpSgpr)

    ########################################
    # Blocked rows or columns
    absWgm = abs(kernel["WorkGroupMapping"])
    if kernel["WorkGroupMappingType"] == "B" and kernel["WorkGroupMapping"] > 1:
      smallNumMagicShift = 31
      magicNumberWgm = ((1<<smallNumMagicShift) // absWgm + 1)

      # tmpSgpr = self.getTmpSgpr(4).idx()
      tmpSgpr = self.sgprPool.checkOutAligned(4, 2, "WGMappingTemp", preventOverflow=0)
      blockId2  = tmpSgpr+0
      wgSerial2 = tmpSgpr+1
      wgmDivisor = tmpSgpr+2
      wgmDivisorMagicNumber = tmpSgpr+3

      kStr += inst("s_mov_b32", sgpr(wgmDivisorMagicNumber), hex(magicNumberWgm)+'L', \
          "magic number for WGM==%u"%absWgm)
      # blockId and serial within block

      if kernel["WorkGroupMapping"]>=0 :
        firstNum = "0"
        secondNum = "1"
      else:
        firstNum = "1"
        secondNum = "0"
      firstWg = "WorkGroup%s"%firstNum
      secondWg = "WorkGroup%s"%secondNum

      # note this overwrites blockId2+1
      kStr += self.sMagicDiv(kernel, dest=blockId2, dividend=sgpr(secondWg), \
          magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift)
      kStr += inst("s_mul_i32", sgpr(wgSerial2), sgpr(blockId2), absWgm, "quotient * non-magic divisor")
      kStr += inst("s_sub_u32", sgpr(wgSerial2), sgpr(secondWg), sgpr(wgSerial2), "%s=remainder"%secondWg)
      kStr += inst("s_mul_i32", sgpr(wgSerial2), sgpr(wgSerial2), sgpr("NumWorkGroups%s"%firstNum), "(wg%s %% WGM)*nwg%s"%(secondNum, firstNum))
      kStr += inst("s_add_u32", sgpr(wgSerial2), sgpr(wgSerial2), sgpr(firstWg), "wgSerial = wg%s + (wg1 %% WGM)*nwg%s"%(firstNum, secondNum))

      kStr += inst("s_cmp_ge_u32", sgpr(blockId2), sgpr("NumFullBlocks"), "blockId >= numFullBlocks ?")
      # reuse wgmDivisorMagicNumber - may override with remainder here:
      kStr += inst("s_cmov_b32", sgpr(wgmDivisorMagicNumber), sgpr("MagicNumberWgmRemainder1"),  "")
      kStr += inst("s_cselect_b32", sgpr(wgmDivisor), sgpr("WgmRemainder1"), absWgm,  "")

      #assert(self.sgprs[firstWg] & 0x1 == 0) # must be even and ...
      #assert(self.sgprs[firstWg]+1 == self.sgprs[secondWg] ) # must be consecutive (for magic div below)
      kStr += self.sMagicDiv(kernel, dest=self.sgprs["WorkGroup0"], dividend=sgpr(wgSerial2), \
          magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift)
      if kernel["WorkGroupMapping"]<0 :
        kStr += inst("s_mov_b32", sgpr(firstWg), sgpr("WorkGroup0"), "mov for WGM<0")
      kStr += inst("s_mul_i32", sgpr(secondWg), sgpr(firstWg), sgpr(wgmDivisor), "quotient * non-magic divisor")
      kStr += inst("s_sub_u32", sgpr(secondWg), sgpr(wgSerial2), sgpr(secondWg), "%s=remainder"%secondWg)

      kStr += inst("s_mul_i32", sgpr(blockId2), sgpr(blockId2), absWgm, "blockId * WGM")

      kStr += inst("s_add_u32", sgpr(secondWg), sgpr(secondWg), \
          sgpr(blockId2), "wg%s += blockId * WGM"%secondNum)

      self.sgprPool.checkIn(tmpSgpr)

    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  # global read addresses: tile offset assignment (message from .s)
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]

    if tP["grcg"]:
      if tP["grcv"]:
        divisorName = tP["lvc"]
      else:
        # Fractional load use the more accurate lsc, multiply by VW later
        divisorName = tP["lsc"]
    else:
      if tP["grcv"]:
        divisorName = tP["lsp"]
      else:
        divisorName = tP["lvp"]
    divisor = kernel[divisorName]

    if tP["grcg"] == tP["tlu"]:
      rReg = self.vgprPool.checkOut(1, "graTA rReg0", self.preventVgprOverflowDuringNewTile) # gro-tile = serial%divisor
      qReg = self.vgprPool.checkOut(1, "graTA qReg0", self.preventVgprOverflowDuringNewTile) # gro-unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprPool.checkOut(1, 'graTA qReg1', self.preventVgprOverflowDuringNewTile) # gro-tile = serial/divisor
      rReg = self.vgprPool.checkOut(1, 'graTA rReg1', self.preventVgprOverflowDuringNewTile) # gro-unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"

    kStr += self.comment1("%s = %u" % (divisorName, kernel[divisorName]))
    if self.groOffsetInMacroTile:
      tReg2 = tReg
      # treg2 and treg same register and value - we store the 'static'
      # part of the address calculation in the SRD to maximize the
      # range of the 32-bit GRO
      kStr += self.comment1("%s = (local)gro%s-tile = serial%s%s (note (wg%s*MT%s) will be added to SRD)" \
          % (vgpr(tReg2), tc, tOpStr, divisorName, tc, tc) )
    else:
      tReg2 = self.vgprPool.checkOut(1, 'treg2', self.preventVgprOverflowDuringNewTile)
      kStr += self.comment1("%s = gro%s-tile = serial%s%s + (wg%s*MT%s)" \
          % (vgpr(tReg2), tc, tOpStr, divisorName, tc, tc) )

    kStr += self.comment1("%s = gro%s-unroll = serial%s%s" \
        % (vgpr(uReg), tc, uOpStr, divisorName) )

    tmpSgpr = self.getTmpSgpr(1).idx() if self.initOptLwaTmpSgpr == None else self.initOptLwaTmpSgpr

    dividendReg = "Serial" # local serial

    if kernel["WaveSeparateGlobalRead%s"%tc] or kernel["ThreadSeparateGlobalRead%s"%tc]:
      divisorVal = self.kernel["WavefrontSize"]
      if kernel["ThreadSeparateGlobalRead%s"%tc]:
        divisorVal //= kernel["ThreadSeparateGlobalRead%s"%tc] * 2
      dividendReg = self.vgprPool.checkOut(1, "idInWave", self.preventVgprOverflowDuringNewTile)
      kStr += vectorStaticRemainder(dividendReg, "Serial", divisorVal, tmpSgpr)
      # vgpr for WaveSeparateGlobalRead + add back to column index
      if tP["tlu"]:
        # tlu case, add wave offset to uReg
        wqreg = uReg
      else:
        # no tlu case, add wave offset to tReg
        wqreg = tReg

    # store DirectToVgpr K interval for later use
    dtvKInterval = 1

    if kernel["DirectToVgpr%s"%tc]:
      # offset calculation for DirectToVgpr
      # ported code from local read for DirectToVgpr
      # parameters
      tile01      = tP["tile01Idx"]
      waveWidth   = kernel["WavefrontSize"]
      dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
      num1DBlocks = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
      num1DWaves  = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
      if kernel["SourceSwap"]:
          dividedForBlkId  = kernel["MatrixInstM"] if (tile01 == 0) else (kernel["MatrixInstM"] * kernel["MatrixInstBM"])
      else:
          dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]
      # TLU case, glvw and vw are applied to the same direction. No need to apply both.
      # non TLU case, glvw and vw are applied to the different direction. We need to apply vw here.
      vectorWidth = 1 if tP["tlu"] else kernel["VectorWidth%s"%tc]
      strideTile  = 1 # tentative
      strideBlock = kernel["MatrixInstM"] * strideTile
      strideWave  = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth
      # tile offset
      kStr += vectorStaticRemainder(qReg, dividendReg, waveWidth, tmpSgpr, \
        "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth)
      kStr += vectorStaticRemainder(rReg, qReg, kernel["MatrixInstN"], tmpSgpr, \
        "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"])
      kStr += staticMultiply(vgpr(rReg), vgpr(rReg), strideTile, sgpr(tmpSgpr), \
        "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile)
      # block offset (no code. assuming num1DBlocks == 1)
      if num1DBlocks > 1:
          tmpVgpr = self.vgprPool.checkOut(1,"tmpVgpr")
          # generate the code only when num1DBlocks > 1.
          # if num1DBlocks is 1, % num1DBlocks is always 0 and no difference in rReg value
          kStr += vectorStaticDivide(tmpVgpr, qReg, dividedForBlkId, tmpSgpr, \
              "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId)
          kStr += vectorStaticRemainder(tmpVgpr, tmpVgpr, num1DBlocks, tmpSgpr, \
              "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks)  # assuming num1DBlocks is power of 2 to use same vreg for src and dst
          kStr += staticMultiply(vgpr(tmpVgpr), vgpr(tmpVgpr), strideBlock, sgpr(tmpSgpr), \
              "2. block offset: bnOffset = bnIdx * strideBlock(%u)" % strideBlock)
          kStr += inst("_v_add_u32", vgpr(rReg), vgpr(tmpVgpr), vgpr(rReg), \
              "3. add N and block offset: bnOffset = block and N offset")
          self.vgprPool.checkIn(tmpVgpr)
      else:
          # comment only because bnIdx = bnIdx % num1DBlocks(1) = 0
          kStr += instCommentOnly("2. block offset: bnIdx = bnIdx %% num1DBlocks(%u) is 0. do nothing" % num1DBlocks)
      kStr += staticMultiply(vgpr(rReg), vgpr(rReg), vectorWidth, sgpr(tmpSgpr), \
          "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth)
      # unroll offset
      # need division for qReg
      kStr += vectorStaticDivide(qReg, qReg, dividendForKId, tmpSgpr, \
          "5. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"]))

      # wave offset
      if num1DWaves > 1:
          # alloc vgpr
          wReg  = self.vgprPool.checkOut(1,"wReg") # quotient
          tmpVgpr = self.vgprPool.checkOut(1, 'graTA vgpr', self.preventVgprOverflowDuringNewTile)
          kStr += vectorStaticDivide(tmpVgpr, dividendReg, waveWidth, tmpSgpr)
          kStr += vectorStaticRemainder(wReg, tmpVgpr, num1DWaves, tmpSgpr)
          kStr += staticMultiply(vgpr(wReg), vgpr(wReg), strideWave, sgpr(tmpSgpr))
          kStr += inst("_v_add_u32", vgpr(rReg), vgpr(wReg), vgpr(rReg),"")
          # release register
          self.vgprPool.checkIn(wReg)
          self.vgprPool.checkIn(tmpVgpr)

      # localSplitU case. Calculate LSU offset here
      if kernel["LocalSplitU"] > 1:
          # alloc vgpr
          tmpVgpr = self.vgprPool.checkOut(1, 'graTA vgpr', self.preventVgprOverflowDuringNewTile)
          subGroup  = kernel["SubGroup0"] * kernel["SubGroup1"]
          dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
          kInterval = waveWidth // dividendForKId
          # generate instruction
          kStr += vectorStaticDivide(tmpVgpr, "Serial", subGroup, tmpSgpr, \
            "LSU offset: sgid = Serial / subGroup(%u)" % subGroup)
          kStr += staticMultiply(vgpr(tmpVgpr), vgpr(tmpVgpr), kInterval, sgpr(tmpSgpr), \
            "LSU offset: lsuOffset = sgid * kInterval(%u)" % (kInterval))
          kStr += inst("_v_add_u32", vgpr(qReg), vgpr(tmpVgpr), vgpr(qReg), \
            "LSU offset: lrOffset += lsuOffset")
          # release register
          self.vgprPool.checkIn(tmpVgpr)

      lrvwOther = self.lrvwB if tP["isA"] else self.lrvwA # The other side of lrvw
      tluOther = kernel["ProblemType"]["TLUB"] if tP["isA"] else kernel["ProblemType"]["TLUA"] # The other side of tlu
      if lrvwOther >= 2 and (not tluOther) and tP["tlu"]:
        # DirectToVgpr + LocalReadVectorWidth>=2 case, multiply qReg by lrvwOther
        dtvKInterval = lrvwOther
      if  tluOther and tP["tlu"]:
        # DirectToVgpr + both TLU case, multiply qReg by kernel["MIInputPerThread"]
        dtvKInterval = kernel["MIInputPerThread"]
      kStr += staticMultiply(vgpr(qReg), vgpr(qReg), dtvKInterval, sgpr(tmpSgpr))

    else:
      divisor2 = divisor
      if kernel["ThreadSeparateGlobalRead%s"%tc]:
        # fragment depthU among (kernel["ThreadSeparateGlobalRead%s"%tc] * 2) groups
        divisor2 //= (kernel["ThreadSeparateGlobalRead%s"%tc]*2)
      kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor2, tmpSgpr)

      #split Wavefront threads into groups (ThreadSeparateGlobalRead<<1) fetching depthU/ThreadSeparateGlobalRead<<1 elements per group
      if kernel["ThreadSeparateGlobalRead%s"%tc]:
        tmpVgpr = self.vgprPool.checkOut(1, 'graTA vgpr', self.preventVgprOverflowDuringNewTile)
        kStr += vectorStaticRemainder(dividendReg, "Serial",self.kernel["WavefrontSize"], tmpSgpr)
        kStr += inst("v_lshrrev_b32", vgpr(tmpVgpr), hex(log2(self.kernel["WavefrontSize"]//(kernel["ThreadSeparateGlobalRead%s"%tc]*2))), vgpr(dividendReg), "ThreadFragmentsize -- groups ( each group with #threads) required for depthU elements")
        kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), hex(log2(divisor2)),vgpr(tmpVgpr), "ElementIdx")
        kStr += inst("_v_add_u32", vgpr(rReg), vgpr(tmpVgpr), vgpr(rReg), "update gro-Unroll")
        self.vgprPool.checkIn(tmpVgpr)

    if not kernel["WaveSeparateGlobalRead%s"%tc] and kernel["ThreadSeparateGlobalRead%s"%tc]:
      kStr += inst("v_readfirstlane_b32", sgpr(tmpSgpr), vgpr("Serial"), "WaveIdxWavefrontWidth")
      kStr += inst("s_lshr_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), hex(log2(self.kernel["WavefrontSize"])), "WaveId")
      kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(tmpSgpr), (kernel["WavefrontSize"] // divisor), \
          "Global Read Wave: each wave loads continuous WavefrontSize(%u)/divisor(%u) columns" % (kernel["WavefrontSize"], divisor))
      kStr += inst("_v_add_u32", vgpr(wqreg), sgpr(tmpSgpr), vgpr(wqreg), \
          "Global Read Wave: add back to column index")
      self.vgprPool.checkIn(dividendReg)

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      kStr += inst("v_readfirstlane_b32", sgpr(tmpSgpr), vgpr("Serial"), "WaveIdxWavefrontWidth")
      kStr += inst("s_lshr_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), hex(log2(kernel["WavefrontSize"])), "WaveId")
      kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(tmpSgpr), kernel[tP["lsp"]] * tP["nrp"], \
          "Global Read Wave: each wave loads continuous lsp(%u)*nrp(%u) columns" % (kernel[tP["lsp"]], tP["nrp"]))
      kStr += inst("_v_add_u32", vgpr(wqreg), sgpr(tmpSgpr), vgpr(wqreg), \
          "Global Read Wave: add back to column index")
      self.vgprPool.checkIn(dividendReg)

    # GRVW<1 case, bpe calculation needs to be done before glvw calculation
    useEarlyBpeCalc = tP["glvw"] < 1
    bpeOffset = 1
    if useEarlyBpeCalc:
      # multiply bpe before multiplying glvw (to support glvw<1 case)
      bpeOffset = tP["bpe"]
      kStr += staticMultiply(vgpr(qReg), vgpr(qReg), bpeOffset, sgpr(tmpSgpr))
      kStr += staticMultiply(vgpr(rReg), vgpr(rReg), bpeOffset, sgpr(tmpSgpr))

    if tP["grcv"] == tP["tlu"]:
      kStr += self.comment1("gro-tile *= glvw")
      kStr += vectorStaticMultiply(tReg, tReg, tP["glvw"], tmpSgpr)
    else:
      kStr += self.comment1("gro-unroll *= glvw")
      kStr += vectorStaticMultiply(uReg, uReg, tP["glvw"], tmpSgpr)

    if not self.groOffsetInMacroTile:
      # Buffer Load will set the SRD to start of the MacroTile
      # So don't add the static wg-related component here - save for later.
      tmpVgpr = self.vgprPool.checkOut(1, 'graTA vgpr', self.preventVgprOverflowDuringNewTile)
      kStr += staticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]] * bpeOffset)  # workgroup
      kStr += inst("_v_add_co_u32", vgpr(tReg2), self.vcc, vgpr(tmpVgpr), \
          vgpr(tReg), "gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
          % (tc, tOpStr, divisorName, tc, tc) )
      self.vgprPool.checkIn(tmpVgpr)

    if kernel["GlobalSplitU"] > 1:
      uReg2 = self.vgprPool.checkOut(1, "uReg2", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(uReg2), vgpr(uReg), "copy for GlobalSplitU")
      tP["gpr"]["uReg2"] = uReg2
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["tReg"] = tReg2
    tP["gpr"]["uReg"] = uReg

    # store DirectToVgpr K interval for later use
    if tP["isA"]:
      self.dtvKIntervalA = dtvKInterval
    else:
      self.dtvKIntervalB = dtvKInterval

    return "" if self.dontAppendCode else kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    kStr = ""

    # GRVW<1 case, bpe calculation needs to be done before glvw calculation
    useEarlyBpeCalc = tP["glvw"] < 1
    bpeOffset = 1
    if useEarlyBpeCalc:
      # multiply bpe before multiplying glvw (to support glvw<1 case)
      bpeOffset = tP["bpe"]

    # note groOffsetInMacroTile rolls these into SRD so don't change here:
    if not self.groOffsetInMacroTile and kernel["GlobalSplitU"] > 1:
      gsuOffset = self.vgprPool.checkOut(1, "gsuOffset", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(gsuOffset), sgpr("GSUSumIdx"), "=gsuSumIdx")
      tmpSgpr = self.getTmpSgpr(1).idx()
      if kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        # graUnrollAssignment += gsuSumIdx*DepthU*bpeOffset
        kStr += staticMultiply(vgpr(gsuOffset), vgpr(gsuOffset), kernel["DepthU"] * bpeOffset, sgpr(tmpSgpr))
      else:
        # graUnrollAssignment += gsuSumIdx*(SizeU/GSU)
        sizeU = self.vgprPool.checkOut(1, "sizeU", self.preventVgprOverflowDuringNewTile)
        kStr += inst("v_mov_b32", vgpr(sizeU), sgpr("SizesSum+0"), \
            "=Size%s"%self.unrollChar)
        quotient = self.vgprPool.checkOut(1, "quotient", self.preventVgprOverflowDuringNewTile)
        kStr += vectorStaticDivide(quotient, sizeU, kernel["GlobalSplitU"], tmpSgpr)
        self.vgprPool.checkIn(sizeU)
        #kStr += " + (size%s/GLOBAL_SPLITU)*" % self.unrollChar
        kStr += inst("v_mul_lo_u32", vgpr(gsuOffset), vgpr(quotient), \
            vgpr(gsuOffset), "gsuOffset=gsuSumIdx*(SizeU/GSU)")
        # *= bpeOffset
        kStr += staticMultiply(vgpr(gsuOffset), vgpr(gsuOffset), bpeOffset, sgpr(tmpSgpr))
        self.vgprPool.checkIn(quotient)

      kStr += inst("_v_add_co_u32", vgpr(tP["gpr"]["uReg"]), self.vcc, \
          vgpr(gsuOffset), vgpr(tP["gpr"]["uReg"]), \
          "graUnrollAssignment += gsuOffset")
      self.vgprPool.checkIn(gsuOffset)
    else:
      kStr += self.comment1(vgpr(tP["gpr"]["uReg"]))

    return "" if self.dontAppendCode else kStr

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    kStr = ""
    if kernel["PersistentKernel"] and kernel["PersistentKernelAlongBatch"]:
      kStr += inst("s_mov_b32", sgpr("WorkGroup2"), sgpr("WGKSerial"), "init WG2 for this persistent loop")
    else:
      kStr += self.comment1(sgpr("WorkGroup2"))
    return kStr

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    kStr = ""
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      kStr += ".set globalReadOffsetA%s,  0%s" \
          % (self.indexChars[index], self.endLine)
      kStr += ".set globalReadOffsetB%s,  0%s" \
          % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    tP["vgprPackedOffsets"] = None
    tP["vgprTileOffsetsCheckOut"] = False
    tP["numVgprTileOffsets"] = 0
    if kernel["_UseSgprForGRO"]:
      # Let the vgprTileOffsets checkin handle tReg later since these are same vgpr
      tP["vgprTileOffsets"] = tP["gpr"]["tReg"]
    else:
      numTileOffsets = tP["nrt"]
      if tP["rtc"]:
        numTileOffsets *= tP["glvw"]
      if self.useGlobalReadTileVgpr:
        tP["vgprTileOffsets"] = self.startVgprGlobalReadTileOffsetA if tP["isA"] else self.startVgprGlobalReadTileOffsetB
        tP["numVgprTileOffsets"] = numTileOffsets # keep numTileOffsets for later use
      else:
        tP["vgprTileOffsets"] = self.vgprPool.checkOut(numTileOffsets, "vgprTileOffsets", self.preventVgprOverflowDuringNewTile)
        tP["vgprTileOffsetsCheckOut"] = True
      v = tP["vgprTileOffsets"]
      numExtraPackedOffsetsPerTile = len(tP["PackedIndices"])-1
      if numExtraPackedOffsetsPerTile:
        tP["vgprPackedOffsets"] = self.vgprPool.checkOut(numExtraPackedOffsetsPerTile * numTileOffsets, "vgprPackedOffsets", self.preventVgprOverflowDuringNewTile)
      strideIdx = tP["lsc"] if tP["tlu"] else tP["lsp"]
      stride = kernel[strideIdx]
      # adjustment for DirectToVgpr + tlu=False + VW > 1 case
      strideInterleave = False
      if kernel["DirectToVgpr%c"%tc] and (not tP["tlu"]) and kernel["VectorWidth%s"%tc] > 1:
        strideInterleave = True
        stride = stride * kernel["VectorWidth%s"%tc] - (kernel["VectorWidth%s"%tc] - 1)
        strideMask = (kernel["VectorWidth%s"%tc] - 1)

      bpeOffset = 1 if tP["glvw"] >= 1 else tP["bpe"] # glvw<1 case, need to multiply strideValue by bpe
      if tP["rtc"]:
        assert(numExtraPackedOffsetsPerTile == 0) # not supported here
        # l=0, s=0
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, 0) )
        # l=0, s>0
        for s in range(1, tP["glvw"]):
          kStr += inst("_v_add_co_u32", vgpr(v+s), self.vcc, 1, \
              vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, s) )
        for l in range(1, tP["nrt"]):
          # l>0, s=0
          strideValue = stride
          if strideInterleave and (l & strideMask) != 0:
            strideValue = 1
          kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]), self.vcc, strideValue * bpeOffset, \
              vgpr(v+(l-1)*tP["glvw"]), \
              "gro%s%s_%u_s%u + %s"%(tP["tensorChar"], tP["tileChar"], l, 0, strideIdx) )
          # l>0, s>0
          for s in range(1, tP["glvw"]):
            kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]+s), self.vcc, \
                1, vgpr(v+l*tP["glvw"]+(s-1)), \
                "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, s) )

      else:
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], 0) )
        for l in range(1, tP["nrt"]):
          strideValue = stride
          if strideInterleave and (l & strideMask) != 0:
            strideValue = 1
          kStr += inst("_v_add_co_u32", vgpr(v+l), self.vcc, strideValue * bpeOffset, \
              vgpr(v+l-1), "gro%s%s_%u += %s"%(tP["tensorChar"], tP["tileChar"], l, strideIdx) )
        if numExtraPackedOffsetsPerTile:
          tmpV = self.vgprPool.checkOutAligned(2,2,"packTmp", self.preventVgprOverflowDuringNewTile)

          for l in range(0, tP["nrt"]):
            lastGroVgpr = vgpr(v+l)
            lastGroIdx = tP["PackedIndices"][0]
            kStr += "\n"
            for p in range(0, numExtraPackedOffsetsPerTile):
              groIdx  = tP["PackedIndices"][p+1]
              groChar = globalParameters["IndexChars"][tP["PackedIndices"][p+1]]
              groVgpr = vgpr(tP["vgprPackedOffsets"] + l*numExtraPackedOffsetsPerTile + p)
              pChar = globalParameters["IndexChars"][tP["PackedIndices"][p]]
              kStr += "V_MAGIC_DIV %s, %s, %s, %s, %s\n" \
                  % (tmpV, lastGroVgpr, sgpr("MagicNumberSize%s"%pChar), \
                  sgpr("MagicShiftSize%s"%pChar), sgpr("MagicAbitSize%s"%pChar) if kernel["MagicDivAlg"]==2 else "0")
              kStr += inst("v_mov_b32", groVgpr, vgpr(tmpV), "extract gro%s%s_%u (%s)"%(tc,groChar,l,groVgpr))
              kStr += inst("v_mul_lo_u32", vgpr(tmpV), groVgpr, sgpr("SizesFree+%u"%lastGroIdx), "remainder part 1")
              kStr += inst("_v_sub_u32", lastGroVgpr, lastGroVgpr, vgpr(tmpV), \
                  "remove extracted bits from gro%s%s_%u (%s)"%(tc, globalParameters["IndexChars"][lastGroIdx], l, lastGroVgpr))
              lastGroVgpr = groVgpr
              lastGroIdx = groIdx
          self.vgprPool.checkIn(tmpV)

      # groOffsetInMacroTile uses same register for both of these, don't free it here:
      if tP["gpr"]["lwoT"] != tP["gpr"]["tReg"] :
        self.vgprPool.checkIn(tP["gpr"]["tReg"])
        tP["gpr"]["tReg"] = None
    return "" if self.dontAppendCode else kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    if kernel["_UseSgprForGRO"]:
      tP["gpr"]["unrollOffsets"] = tP["gpr"]["uReg"]
    else:
      numUnrollOffsets = tP["nru"]
      if tP["ruc"]:
        numUnrollOffsets *= tP["glvw"]
      if self.useGlobalReadTileVgpr:
        tP["gpr"]["unrollOffsets"] = self.startVgprGlobalReadUnrollOffsetA if tP["isA"] else self.startVgprGlobalReadUnrollOffsetB
      else:
        tP["gpr"]["unrollOffsets"] = self.vgprPool.checkOut(numUnrollOffsets, "unrollOffsets", self.preventVgprOverflowDuringNewTile)
      v = tP["gpr"]["unrollOffsets"]
      strideIdx = (tP["lsp"] if tP["tlu"] else tP["lsc"])
      stride = kernel[strideIdx]
      prevStride = 0
      totalStride = 0
      bpeOffset = 1 if tP["glvw"] >= 1 else tP["bpe"] # glvw<1 case, need to multiply strideValue by bpe
      dtvKInterval = self.dtvKIntervalA if tP["isA"] else self.dtvKIntervalB

      if tP["ruc"]:
        # l=0, s=0
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u_s%u"%(tc, self.unrollChar, 0, 0) )
        # l=0, s>0
        for s in range(1, tP["glvw"]):
          kStr += inst("_v_add_co_u32", vgpr(v+s), self.vcc, 1, \
              vgpr(v+s-1), "gro%s%s_%u_s%u"%(tc, self.unrollChar, 0, s) )
        for l in range(1, tP["nru"]):
          # l>0, s=0
          totalStride += stride
          if dtvKInterval > 1:
            # DirectToVgpr + k interval > 1 case, stride * dtvKInterval is added every dtvKInterval.
            # Add mod in mod != 0 case
            totalStride = stride * (l - (l % dtvKInterval)) + (l % dtvKInterval)
          currStride = totalStride - prevStride
          prevStride = totalStride
          kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]), self.vcc, currStride * bpeOffset, \
              vgpr(v+(l-1)*tP["glvw"]), \
              "gro%s%s_%u_s%u + %s"%(tc, self.unrollChar, l, 0, strideIdx) )
          # l>0, s>0
          for s in range(1, tP["glvw"]):
            kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]+s), self.vcc, \
                1, vgpr(v+l*tP["glvw"]+(s-1)), \
                "gro%s%s_%u_s%u"%(tc, self.unrollChar, 0, s) )
      else:
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u"%(tc, self.unrollChar, 0) )
        for l in range(1, tP["nru"]):
          totalStride += stride
          if dtvKInterval > 1:
            # DirectToVgpr + k interval > 1 case, stride * dtvKInterval is added every dtvKInterval.
            # Add mod in mod != 0 case
            totalStride = stride * (l - (l % dtvKInterval)) + (l % dtvKInterval)
          currStride = totalStride - prevStride
          prevStride = totalStride
          kStr += inst("_v_add_co_u32", vgpr(v+l), self.vcc, currStride * bpeOffset, \
              vgpr(v+l-1), "gro%s%s_%u + %s"%(tc, self.unrollChar, l, strideIdx) )
      #self.vgprPool.checkIn(tP["gpr"]["uReg"])
    return "" if self.dontAppendCode else kStr

  ##############################################################################
  # Global Read Addresses: Branch A/B
  ##############################################################################
  def graBranch(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B
  # See if the load (including vw) will extend past the 'free' dim of the
  # tensor.  If so clip to the last legal value which is inside the array
  ##############################################################################
  def graShift(self, kernel, tP):
    # FractionalLoad maps addresses in a different way?

    # graShift requires a vgpr for each address component (so each component
    # can be examined and shifted if necessary) - therefore does not work
    # with UseSgprForGRO.
    assert(not kernel["_UseSgprForGRO"])

    kStr = ""
    #tc = tP["tensorChar"]
    # edge value
    margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1, "edge", self.preventVgprOverflowDuringNewTile)

    # GRVW<1 case, bpe calculation is done before glvw calculation for offset calculation
    useEarlyBpeCalc = tP["glvw"] < 1

    if self.groOffsetInMacroTile:
      # Subtract the static component from SizesFree:
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(tP["wg"]), kernel[tP["mt"]], "WorkGroup[01] * MT")
      kStr += inst("s_sub_u32", sgpr(tmpSgpr), self.sizeRef(tP["idx"]), sgpr(tmpSgpr), \
                "edge = Size%s - WG*MT"%(tP["tileChar"]))
      # use math here to use unsigned (to increase range)
      #  - add srdShiftLeft to tmpSgpr - ensure it is always positive
      #  - below add srdShiftLeft to a tmp copy of the offset used for the compare
      # edge = (Size - WG*MT) - margin = the last valid load position that won't cause OOB
      # offset = the current load position for this thread
      # so if offset is larger than edge, we go back to the edge position
      kStr += inst("s_sub_u32", sgpr(tmpSgpr), sgpr(tmpSgpr), margin, "edge -= margin(%u)"%(margin))
      if useEarlyBpeCalc and tP["bpe"] > 1:
        kStr += inst("s_lshl_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), log2(tP["bpe"]), "*= BPE")
      kStr += inst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
          "edge vgpr = Size%s- WG*MT - margin(%u)"%(tP["tileChar"], margin) )
      #shiftedEdge = self.vgprPool.checkOut(1, "shiftedEdge", self.preventVgprOverflowDuringNewTile)
      #kStr += inst("_v_add_co_u32", vgpr(shiftedEdge), self.vcc, vgpr(edge), self.srdShiftLeft[tc],
      #             "shiftedEdge = edge + srdShiftLeft({})".format(self.srdShiftLeft[tc]))
    else:
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += inst("s_sub_u32", sgpr(tmpSgpr), self.sizeRef(tP["idx"]), margin, \
          "edge = Size%s-%u"%(tP["tileChar"], margin) )
      if useEarlyBpeCalc and tP["bpe"] > 1:
        kStr += inst("s_lshl_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), log2(tP["bpe"]), "*= BPE")
      kStr += inst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
          "edge vgpr = Size%s-%u"%(tP["tileChar"], margin) )

    if kernel["CheckDimOverflow"]:
      # if tensor is really skinny (SizesFree is less then glvw) then shifting fails-
      # can detect here if the computed edge after subtracting marging is <0
      kStr += self.assert_ge_i32(vgpr(edge), 0)
    #kStr += self.assert_ne(sgpr("WorkGroup0"),1)

    # shift offsets
    vSrc = tP["vgprTileOffsets"]
    if self.useGlobalReadTileVgpr:
      # self.useGlobalReadTileVgpr case, use new vgpr as dst to avoid overwritting GlobalReadTileVgpr with shifted value
      tP["vgprTileOffsets"] = self.vgprPool.checkOut(tP["numVgprTileOffsets"], "vgprTileOffsets", self.preventVgprOverflowDuringNewTile)
      tP["vgprTileOffsetsCheckOut"] = True
    vDst = tP["vgprTileOffsets"]
    tmpSgpr = self.getTmpSgpr(self.laneSGPRCount).idx()
    for l in range(0, tP["nrt"]):
      # compare
      cmpCommentText = "offset < edge"
      if self.groOffsetInMacroTile:
        #shiftedOffset = self.vgprPool.checkOut(1, "shiftedOffset", self.preventVgprOverflowDuringNewTile)
        #kStr += inst("_v_add_co_u32", vgpr(shiftedOffset), self.vcc, vgpr(vSrc+l), self.srdShiftLeft[tc], "shiftedOffset = offset + srdShiftLeft(%u)"%(self.srdShiftLeft[tc]))
        ## int cmp since if we are near the front of the tile this may go negative:
        #kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,self.laneSGPRCount), vgpr(shiftedOffset), vgpr(shiftedEdge),
        #             "shiftedOffset < shiftedEdge")
        #self.vgprPool.checkIn(shiftedOffset)
        kStr += inst("v_min_i32", vgpr(vDst+l), vgpr(edge), vgpr(vSrc+l),
                     "offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge))
      else:
        kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,self.laneSGPRCount), vgpr(vSrc+l), vgpr(edge),
                     "shiftedOffset < shiftedEdge")
        # shift
        kStr += inst("v_cndmask_b32", vgpr(vDst+l), vgpr(edge), vgpr(vSrc+l), sgpr(tmpSgpr,self.laneSGPRCount),
                     "offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge))
    self.vgprPool.checkIn(edge)
    #if self.groOffsetInMacroTile:
    #  self.vgprPool.checkIn(shiftedEdge)

    #if tP["isB"]:
    #  kStr += "s_endpgm\n"

    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    tmp = self.vgprPool.checkOut(3, "tmp", self.preventVgprOverflowDuringNewTile)
    graIdx = 0
    swapPerpPara = (((tc=="A" and kernel["DirectToVgprA"]) or (tc=="B" and kernel["DirectToVgprB"])) \
                    and (not tP["tlu"]) and tP["nrp"] > 1)

    if not swapPerpPara:
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
              # single loop
              singleStr, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              kStr += singleStr
    else:
      # swap para and perp
      for para in range(0, tP["nrc"]):
        for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
          for perp in range(0, tP["nrp"]):
            for sPerp in range(0, tP["nrpv"]):
              # single loop
              singleStr, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              kStr += singleStr

    if tP["vgprTileOffsetsCheckOut"]:
      self.vgprPool.checkIn(tP["vgprTileOffsets"])
      tP["vgprTileOffsets"] = None
      tP["vgprTileOffsetsCheckOut"] = False
      # _UseSgprForGRO uses same vgpr for ureg and tP["gpr"]["unrollOffsets"] so
      # let checkin(ureg) do the checkin
      # vgprTileOffsets is renamed version of treg/lwo so checkin here

    if not kernel["_UseSgprForGRO"] and not self.useGlobalReadTileVgpr:
      self.vgprPool.checkIn(tP["gpr"]["unrollOffsets"])
      tP["gpr"]["unrollOffsets"] = None

    if tP["vgprPackedOffsets"] != None:
      self.vgprPool.checkIn(tP["vgprPackedOffsets"])
      tP["vgprPackedOffsets"] = None

    self.vgprPool.checkIn(tmp)
    #if tP["isB"]:
    #  kStr += self.bomb(0x100)

    # ensure we computed all the required addresses above
    for zpr in self.zeroPadRegs[tc].values():
      assert(zpr.state == ZeroPadReg.State.CalculatedAddr)

    return "" if self.dontAppendCode else kStr

  ##############################################################################
  # LdsInc for DirectToLds
  ##############################################################################
  def getLdsIncForDTL(self, kernel, tP):
    tc = tP["tensorChar"]
    # DTL does not support loadBytePerElem > bpr
    loadBytePerElem = int(kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"])
    ldsInc = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * loadBytePerElem
    if kernel["LdsBlockSizePerPad%s"%tc] != 0:
      padInterval = kernel["LdsBlockSizePerPad%s"%tc]
    else:
      padInterval = (self.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.bpr
    ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]
    return ldsInc

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B (single loop)
  ##############################################################################
  def graFinalOffsetsSingleLoop(self, kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara):
    kStr = ""
    problemType = kernel["ProblemType"]
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    if tP["rtc"]:
      tVW = tP["glvw"]
      tVS = 1
    elif tP["ruc"]:
      uVW = tP["glvw"]
      uVS = 1

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
        kStr += "GLOBAL_OFFSET_%s vgprGlobalReadOffset%s+%u"%(tP["tensorChar"], tP["tensorChar"], graIdx)
      else:
        kStr += "GLOBAL_OFFSET_%s vgprGlobalReadAddr%s+%u"%(tP["tensorChar"], tP["tensorChar"], graIdx)
      packedIter = 0 #iterator through ia
      iaToGpr = [None] * problemType["TotalIndices"]
      for i in tP["ia"]:
        if i < problemType["NumIndicesC"]:
          if i == tP["tileIdx"]:
            iaToGpr[i] = vgprTile
            kStr += ", %2u" % iaToGpr[i]
          else:
            if isPackedIndex(kernel,i, tP["PackBatchDims"]):
              iaToGpr[i] = tP["vgprPackedOffsets"] + \
                            (vgprTile-tP["vgprTileOffsets"])*(len(tP["PackedIndices"])-1) + \
                            packedIter
              kStr += ", %2u" % (iaToGpr[i])
              packedIter += 1
            else:
              # just a group index
              if not kernel["BufferLoad"] and kernel["ProblemType"]["StridedBatched"]:
                # buffer load adds these to SRD not the GLOBAL_OFFSET here
                # general_batch case loads batch pointer, don't calculate based on workgroup/stride
                kStr += ", sgprWorkGroup%u"%i
        else: # summation index
          if i == problemType["IndexUnroll"]:
            iaToGpr[i] = vgprUnroll
            kStr += ", %2u" % iaToGpr[i]
          # other summation indices are ignored

      kStr += ", %u // gRO%s_%u_%u_%u_%u%s" % (tmp, tP["tensorChar"], \
          para, sPara, perp, sPerp, self.endLine)

      tmpSgpr = self.getTmpSgpr(2).idx()

      # modify start
      if (not kernel["_UseSgprForGRO"]) and kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
         # add room for instruction offset
        groVgpr = "GlobalReadOffset%s+%u" % (tP["tensorChar"], graIdx)
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), self.buff_load_inst_offset_max, "" )
        kStr += inst("_v_add_u32", vgpr(groVgpr), vgpr(groVgpr), sgpr(tmpSgpr), "shift for UseInstOffsetForGRO")

        # get ldsInc
        ldsInc = self.getLdsIncForDTL(kernel, tP)

        # buffer_load only support 12 bit instruction offset
        # we have to increase m0 if offset is larger thant 12 bits
        # so only keep 12 bit offset and subtract it on global address
        # global address will add back by buffer_load instruction offset
        ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
        if (ldsInc != 0):
          kStr += inst("s_mov_b32", sgpr(tmpSgpr), ldsInc, "" )
          kStr += inst("_v_sub_u32", vgpr(groVgpr), vgpr(groVgpr), sgpr(tmpSgpr), "sub offset for buffer_load instoffset")

      for zpr in [zpr for zpr in self.zeroPadRegs[tc].values() if zpr.isMatch(perp, sPerp, para, sPara)]:
        assert(zpr.state == ZeroPadReg.State.Allocated) # only calc address once
        zpr.state = ZeroPadReg.State.CalculatedAddr
        kStr += self.comment1(zpr.regName)
        (freeDim,sumDim) = zpr.zp[:2]
        freeDimChar = globalParameters["IndexChars"][freeDim]
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        assert(iaToGpr[freeDim] != None)
        kStr += inst("v_mul_lo_u32", \
                  vgpr(zpr.regName), \
                  vgpr(iaToGpr[freeDim]), \
                  self.strideRef(tc, freeDim), \
                  "zp.freeDim * strideFree")
        vgprOffset = vgpr(iaToGpr[sumDim]) if vgpr(iaToGpr[sumDim]) else 0
        if sumDim in kernel["ProblemType"]["MirrorDims%s"%tc]:
          kStr += inst("_v_sub_u32", \
                  vgpr(tmp), \
                  sgpr("Size%s"%sumDimChar), \
                  vgprOffset, \
                  "zp.sumDim mirror 1")
          kStr += inst("_v_sub_u32", \
                  vgpr(tmp), \
                  vgpr(tmp), \
                  "1", \
                  "zp.sumDim mirror 2")
          vgprOffset = vgpr(tmp)
        #iaToGpr[sumDim] will be 0 for other summation dims
        kStr += inst("v_mul_lo_u32", \
                  vgpr(tmp), \
                  vgprOffset, \
                  self.strideRef(tc, sumDim), \
                  "zp.sumDim * strideSum")
        kStr += inst("_v_add_u32", \
                  vgpr(zpr.regName), \
                  vgpr(zpr.regName), \
                  vgpr(tmp),
                  "zp.freeDim * strideFree + zp.sumDim * strideSum")
        kStr += inst("v_lshlrev_b32", \
                     vgpr(zpr.regName), \
                     "Bpe%sLog2"%tc, \
                     vgpr(zpr.regName), \
                     "scale to bpe")
        kStr += inst("_v_sub_u32",
                  vgpr(zpr.regName), \
                  vgpr(zpr.regName), \
                  sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                  "zp.freeDim * strideFree + zp.sumDim * strideSum PadStart")

      if kernel["BufferLoad"] and kernel["FractionalLoad"]:
        lastValidThread = kernel[tP["lsc"]]*kernel[tP["lsp"]]//tP["glvw"]
        if lastValidThread < kernel["NumThreads"]:
          kStr += "// Offset only valid for %u/%u threads inside the PerLoadTile\n" \
               % (lastValidThread, kernel["NumThreads"])
          kStr += inst("s_mov_b32", sgpr(tmpSgpr), lastValidThread, "" )
          kStr += inst("v_cmp_lt_u32", \
              self.vcc, \
              vgpr("Serial"), \
              sgpr(tmpSgpr), \
              "tid < valid-tid")
          boundsVgpr = self.vgprPool.checkOut(3)
          kStr += inst("s_mov_b32", sgpr(tmpSgpr), "BufferOOB", "" )
          kStr += inst("v_mov_b32", vgpr(boundsVgpr), sgpr(tmpSgpr), "" )
          kStr += inst("v_cndmask_b32", \
               vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)), \
               vgpr(boundsVgpr), \
               vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)), \
               self.vcc,
               "Mask load so OOB will return 0")
          self.vgprPool.checkIn(boundsVgpr)

    needFirstSgprOffset = kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]
    if (kernel["_UseSgprForGRO"] or self.checkGRO) and (needFirstSgprOffset or graIdx > 0):
      # compute offsets for scalar global read offsets:
      if kernel["_UseSgprForGRO"]:
        tmpIdx = graIdx if needFirstSgprOffset else graIdx-1
        scalarGro = "ScalarGlobalReadOffset%s+%u"%(tc, tmpIdx)
      else:
        scalarGro = self.getTmpSgpr(1).idx()

      # this needs unroll stride in some cases and free stride in others
      # if we have multiple free strides - what is expected behavior?
      # could just extract the first free dimension from A?
      stride1 = "Stride%s%s"%(tc,self.indexChars[tP["idx"]])
      if tP["tlu"]:
        tileStride   = kernel[tP["lsc"]] * (para*tVW + sPara*tVS)
        unrollStride = kernel[tP["lsp"]] * (perp*uVW + sPerp*uVS)
        unrollSummation = [ i for i in tP["ia"] if i in problemType["IndicesSummation"] ]
        strideU = "Stride%s%s"%(tc,self.indexChars[unrollSummation[-1]])
        kStr += inst("s_mul_i32", sgpr(scalarGro), sgpr(strideU), unrollStride, \
                     "compute offset diff (scaled unrollDim)")
        if tileStride:
          kStr += inst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), tileStride, \
                     "compute offset diff (tileDim)")
      else:
        tileStride   = kernel[tP["lsp"]] * (perp*tVW + sPara*tVS)
        unrollStride = kernel[tP["lsc"]] * (para*uVW + sPerp*uVS)
        strideF = "Stride%s%s"%(tc,self.indexChars[tP['tileIdx']])
        kStr += inst("s_mul_i32", sgpr(scalarGro), sgpr(strideF), tileStride, \
                     "compute offset diff (scaled tileDim)")
        if unrollStride:
          kStr += inst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), unrollStride, \
                     "compute offset diff (unrollDim)")

      # Using offsets so GRO holds a byte offset not an element offset
      # So scale here before comparison:
      kStr += inst("s_lshl_b32", \
          sgpr(scalarGro), \
          sgpr(scalarGro), \
          hex(log2(tP["bpe"])), \
          "scalar offset *= bytes/element")

      if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
        # add room for instruction offset
        kStr += inst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), self.buff_load_inst_offset_max, "shift for UseInstOffsetForGRO")

        # get ldsInc
        ldsInc = self.getLdsIncForDTL(kernel, tP)

        # buffer_load only support 12 bit instruction offset
        # we have to increase m0 if offset is larger thant 12 bits
        # so only keep 12 bit offset and subtract it on global address
        # global address will add back by buffer_load instruction offset
        ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
        if (ldsInc != 0):
          kStr += inst("s_sub_u32", sgpr(scalarGro), sgpr(scalarGro), ldsInc, "sub offset for buffer_load instoffset")

      if self.checkGRO:
        # Debug mode to verify that the computed offsets are offset by the expected scalar
        print(tc, "tileStride=", tileStride, "unrollStride=", unrollStride, \
              "stride=%s"%(stride1))

        kStr += self.assert_vector_diff(vgpr("GlobalReadOffset%s+%u"%(tc,0)), \
                                        vgpr("GlobalReadOffset%s+%u"%(tc,graIdx)), \
                                        sgpr(scalarGro))

    # dump final offsets
    # BufferLoad flavor:
    #if tP["isA"]:
    #  kStr += self.dump(vgpr("GlobalReadOffset%s+%u+0"%(tP["tensorChar"], graIdx)))
    # Global load flavor:
    #kStr += dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)))
    #kStr += dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)))
    graIdx += self.rpgo if kernel["BufferLoad"] else self.rpga

    return kStr, graIdx

  ##############################################################################
  # Add the constant offsets to the specified srd.
  # Srd is set to point to the base of the tile. All offsets except lowest-order
  # 2d dims are computed into the SRD.
  # GRO are offset from the tile SRD and the first GRO will be 0
  # Only called for BufferLoad=1 (or eventually BufferStore=1)
  ##############################################################################
  def computeLoadSrd(self, kernel, tP, tc, indices, bpe, isPap):
    kStr = ""

    stmpRef = self.getTmpSgpr(2)
    stmp = stmpRef.idx()
    tileStartRef = self.getTmpSgpr(2)
    tileStart = tileStartRef.idx()
    prePadSgprRef = self.getTmpSgpr(1)
    prePadSgpr = prePadSgprRef.idx()
    wroteTileStart = False
    #---
    # Compute tileStart #elements from the 2D array start
    # Add tile (and unroll if GSU) component into SRD - SRD will point to beginning of the macro-tile:
    if self.groOffsetInMacroTile:
      # packed modes can't use this mode, and code here assumes 1 index.
      assert(len(kernel["PackedC0IndicesX"])==1)
      assert(len(kernel["PackedC1IndicesX"])==1)

      wroteTileStart = True
      #tP['ia'][1]

      # This is guaranteed to fit in 32-bit since the WG*MT is a number of elements in some unsigned direction:
      kStr += self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(tP["wg"]), kernel[tP["mt"]], "WorkGroup[01] * MT")
      if kernel["CheckDimOverflow"] >=2:
        kStr += self.assert_eq(sgpr(tileStart+1),0)
      strideF = self.strideRef(tc, tP['tileIdx'])
      if not self.isConstUnitStride(strideF):
        kStr += self.s_mul_u64_u32(sgpr(tileStart), sgpr(tileStart+1), sgpr(tileStart+0), \
                   strideF, "tlu=0, scaled tile-offset by stride")

      if kernel["StreamK"]:
        # StreamK partial tile - offset to tile start index
        kStr += inst("s_mul_i32", sgpr(stmp), sgpr("StreamKLocalStart"), "DepthU", "StreamK tile start offset")
        strideL = self.strideRef(tc, kernel["ProblemType"]["IndicesSummation"][0])
        kStr += self.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), sgpr(stmp), strideL, "StreamK tile start offset")
        if kernel["CheckDimOverflow"] >=2:
          kStr += self.assert_eq(sgpr(stmp+1),0)
        kStr += inst("s_add_u32",  sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum GsuOffset term to tilestart")
        kStr += inst("s_addc_u32", sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum GsuOffset term to tilestart")

      if kernel["GlobalSplitU"] > 1:
        # Only GlobalSplitUSummationAssignmentRoundRobin supported for groOffsetInMacroTile - would need different math here for start:
        assert(kernel["GlobalSplitUSummationAssignmentRoundRobin"])

        kStr += self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), kernel["DepthU"], sgpr("GSUSumIdx"), "gsuOffset = DepthU*bpe*GSUSumIdx")
        if kernel["CheckDimOverflow"] >=2:
          kStr += self.assert_eq(sgpr(stmp+1),0)
        # TODO - PackSummationDims handling needs to handle multiple sum dims
        unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]
        stride = self.strideRef(tc,unrollSummation[-1])
        if tP["tlu"] and not self.isConstUnitStride(stride):
          # non-transpose case, unroll is in perp dim and should be scaled by unroll Stride
          kStr += self.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), sgpr(stmp+0), \
                    stride, "tlu=1, scaled unroll-offset by stride")

        kStr += inst("s_add_u32",  sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum GsuOffset term to tilestart")
        kStr += inst("s_addc_u32", sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum GsuOffset term to tilestart")

    # Output : tileStart[0:1] have offset in elements from the 2D start of the tile.
    # if groOffsetInMacroTile=1, 2DStart + tileStart gives the the start of the macro-tile;
    # This is used to compute the limit.
    # Later we modify tileStart to include batch and higher-order dims and add this to SRD.

    #---
    # Compute BUFFER Limit:
    bpeOffset = 1 if tP["glvw"] < 1 else tP["bpe"] # glvw>=1 case, need to multiply by bpe
    prePad = prePadConst = self.srdShiftLeft[tc] * bpeOffset # leave room in case we have to pointer shift
    # subtract the zeropad(s) from the SRD base
    # this causes small offsets (<pad) to result in large negative offsets and thus report as OOB
    for i,zp in enumerate(kernel["ProblemType"]["ZeroPad%s"%tc]):
      (freeDim,sumDim) = zp[:2]
      freeDimChar = globalParameters["IndexChars"][freeDim]
      sumDimChar  = globalParameters["IndexChars"][sumDim]
      # override the const pre-pad with an SGPR based on the leading/trailing items:
      prePad = sgpr(prePadSgpr)
      if i==0:
        kStr += inst("s_add_u32", prePad, \
                 sgpr("PadStart%s%s%s"%(tc, freeDimChar,sumDimChar)), \
                 prePadConst, "prePadSgpr = PadStart + ptr-shift-pad")
      else:
        kStr += inst("s_add_u32", prePad, \
                 prePad, sgpr("PadStart%s%s%s"%(tc,freeDimChar, sumDimChar)), \
                 "prepadSgpr += PadStart")

    if not wroteTileStart:
      kStr += inst("s_mov_b32", sgpr(tileStart+0), 0, "set default tileStart")
      kStr += inst("s_mov_b32", sgpr(tileStart+1), 0, "set default tileStart")

    if self.use64bShadowLimit:
      limitTmp0 = "ShadowLimit%s+0"%tc
      limitTmp1 = "ShadowLimit%s+1"%tc
    else:
      limitTmp0 = stmp+0
      limitTmp1 = stmp+1

    kStr += inst("s_sub_u32",  sgpr(limitTmp0), sgpr("Tensor2dSize%s"%tc), sgpr(tileStart+0), "sub tileStart")
    kStr += inst("s_subb_u32", sgpr(limitTmp1), sgpr("Tensor2dSize%s+1"%tc), sgpr(tileStart+1), "sub tileStart")

    if self.use64bShadowLimit:
      # Set initial buffer limit
      # if the limit is >64bit, incrementSrd decrements the shadow as the SRD increments,
      # and when we get within 32-bit we start to step down the SRD
      # if the limit is <32bits, set it accurately here:
      # Note lshl_b64 the higher-numbered SGPR has the upper 32-bits
      kStr += inst("s_lshl_b64", sgpr("ShadowLimit%s"%tc,2),  sgpr("ShadowLimit%s"%tc,2), \
          hex(log2(tP["bpe"])), "Set limit to use bytes")
      if prePad:
        kStr += inst("s_add_u32",  sgpr("ShadowLimit%s+0"%tc), sgpr("ShadowLimit%s+0"%tc), prePad, "extend limit for pre-pad")
        kStr += inst("s_addc_u32", sgpr("ShadowLimit%s+1"%tc), sgpr("ShadowLimit%s+1"%tc), 0, "extend limit for pre-pad")

      if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
        kStr += inst("s_add_u32",  sgpr("ShadowLimit%s+0"%tc), sgpr("ShadowLimit%s+0"%tc), self.buff_load_inst_offset_max, "extend limit for directToLDS instruction offset")
        kStr += inst("s_addc_u32", sgpr("ShadowLimit%s+1"%tc), sgpr("ShadowLimit%s+1"%tc), 0, "extend limit for directToLDS instruction offset")

      kStr += inst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
      kStr += inst("s_cselect_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "BufferLimit%s"%tc, "Move shadow to real if we are within 2^32")
    else:
      # put limit directly into SRD:
      kStr += inst("s_lshl_b32", sgpr("Srd%s+2"%tc), sgpr(stmp+0), hex(log2(tP["bpe"])), "Set limit to use bytes")
      kStr += inst("s_add_u32",  sgpr("Srd%s+2"%tc), sgpr("Srd%s+2"%tc), prePad, "extend limit for pre-pad")

    # Apply any high-order address components to the tileStart and eventually the SRD - batch idx for batched gemm
    if kernel["ProblemType"]["StridedBatched"]:
      numDim = len(indices)
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
          stride = "Stride%s%s"%(tc,self.indexChars[tP['ia'][i]])
          if not wroteTileStart:
            kStr += self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG")
            wroteTileStart = True
          else:
            kStr += self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG")
            kStr += inst("s_add_u32",  sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum wg term to tilestart")
            kStr += inst("s_addc_u32", sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum wg term to tilestart")
          wg+=1

    sgprAddress0 = sgpr("%s%s+0"%(self.sgprAddressStrAB,tc))
    sgprAddress1 = sgpr("%s%s+1"%(self.sgprAddressStrAB,tc))

    # Add the tile start to the SRD
    if wroteTileStart:
      kStr += scalarStaticMultiply(sgpr(tileStart,2), sgpr(tileStart,2), bpe, None, "tileStart *= BPE")
      kStr += inst("s_add_u32",  sgpr("Srd%s+0"%tc), sgprAddress0, sgpr(tileStart+0), "SRD base = Address+ tileStart0")
      kStr += inst("s_addc_u32", sgpr("Srd%s+1"%tc), sgprAddress1, sgpr(tileStart+1), "SRD base = Address+ tileStart1")
    else:
      kStr += inst("s_mov_b32", sgpr("Srd%s+0"%tc), sgprAddress0, "init SRD base address (lower )" )
      kStr += inst("s_mov_b32", sgpr("Srd%s+1"%tc), sgprAddress1, "init SRD base address (upper) + other fields" )

    # self.groOffsetInMacroTile == 1 case,  pre-pad is already subtracted from AddressA/B
    if prePad and self.groOffsetInMacroTile == 0:
      kStr += inst("s_sub_u32",  sgpr("Srd%s+0"%tc), sgpr("Srd%s+0"%tc), prePad, "pre-pad to make room for possible pointer shift")
      kStr += inst("s_subb_u32",  sgpr("Srd%s+1"%tc), sgpr("Srd%s+1"%tc), 0, "pre-pad to make room for possible pointer shift")

    if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
      kStr += inst("s_sub_u32",  sgpr("Srd%s+0"%tc), sgpr("Srd%s+0"%tc), self.buff_load_inst_offset_max, "make room for directToLDS instruction offset")
      kStr += inst("s_subb_u32",  sgpr("Srd%s+1"%tc), sgpr("Srd%s+1"%tc), 0, "make room for directToLDS instruction offset")

    # PAP case, setting value to Srd+3 is not necessary (already set before PK loop)
    if not isPap:
      kStr += inst("s_mov_b32", sgpr("Srd%s+3"%tc), "Srd127_96", "Set bits 127_96 in SRD")

    #if tP["isB"]:
    #  kStr += self.assert_ne(sgpr("WorkGroup1"), 0xA)

    if kernel["CheckDimOverflow"]>=2:
      # double-check to make sure the SRD limit is inside the allowed tensor:
      #   - compute size of tensor in elements (including all dimensions)
      #   - subtract the SRD base and SRD buffer limit
      #   - Make sure the 64bit result is >0
      kStr += inst("s_lshl_b64", sgpr(stmp,2), sgpr("Tensor2dSize%s"%tc,2), log2(bpe), "tensor size in bytes")
      kStr += inst("s_add_u32",  sgpr(stmp+0), sgpr(stmp+0), sgprAddress0, "add start ptr to compute tensor%s bot-right"%tc)
      kStr += inst("s_addc_u32", sgpr(stmp+1), sgpr(stmp+1), sgprAddress1, "add start ptr to compute tensor%s bot-right"%tc)
      kStr += inst("s_sub_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Srd%s+0"%tc), "sub SRD base")
      kStr += inst("s_subb_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("Srd%s+1"%tc), "sub SRD base")
      if self.use64bShadowLimit:
        kStr += inst("s_sub_u32", sgpr(stmp+0), sgpr(stmp+0), sgpr("ShadowLimit%s+0"%tc), "sub buffer size")
        kStr += inst("s_subb_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("ShadowLimit%s+1"%tc), "sub buffer size")
      else:
        kStr += inst("s_sub_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Srd%s+2"%tc), "sub buffer limit")

      kStr += self.assert_eq(sgpr(stmp+1), 0)  # must be 0 or we are way OOB
      kStr += self.assert_ge_u32(sgpr(stmp+0), 0) # diff greater than zero
      if 0 and tP["isB"]:
        t = self.vgprPool.checkOut(1, "t", self.preventVgprOverflowDuringNewTile)
        kStr += inst("s_add_u32", sgpr(stmp+0), sgpr("WorkGroup1"), sgpr("WorkGroup2"), "bozo, debug")
        kStr += inst("v_mov_b32", vgpr(t), 0x54, "")
        kStr += self.assert_ne(sgpr(stmp+0), vgpr(t) )
        self.vgprPool.checkIn(t)

    if kernel["PackSummationDims"]:
      kStr += self.comment("Save the initial SRD and limit for later address calculation")
      kStr += inst("s_mov_b32", sgpr("InitialSrd%sBase+0"%tc), sgpr("Srd%s+0"%tc), "save base")
      kStr += inst("s_mov_b32", sgpr("InitialSrd%sBase+1"%tc), sgpr("Srd%s+1"%tc), "save base")
      if self.use64bShadowLimit:
        kStr += inst("s_mov_b32", sgpr("InitialSrd%sLimit+0"%tc), sgpr("ShadowLimit%s+0"%tc), "save shadow limit")
        kStr += inst("s_mov_b32", sgpr("InitialSrd%sLimit+1"%tc), sgpr("ShadowLimit%s+1"%tc), "save shadow limit")
      else:
        kStr += inst("s_mov_b32", sgpr("InitialSrd%sLimit"%tc), sgpr("Srd%s+2"%tc), "save limit")

    # invalid global read for performance evaluation only
    if self.enable["InvalidGlobalRead%s"%tc]:
      kStr += inst("s_mov_b32", sgpr("Srd%s+2"%tc), hex(0), "set out-of-bound addr for performance evaluation only")
      if self.use64bShadowLimit:
        kStr += inst("s_mov_b32", sgpr("ShadowLimit%s+1"%tc), hex(0xffffffff), "set out-of-bound addr for performance evaluation only")

    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  def graAddresses(self, kernel, tP, isPap=False):
    kStr = ""
    tc = tP["tensorChar"]
    graIdx = 0

    if kernel["BufferLoad"]:
      # maxAddrSgpr = size[n] * stride[n-1]
      kStr += self.comment1("max read offset = size[n] * stride[n-1]")

      kStr += self.computeLoadSrd(kernel, tP, tc, kernel["ProblemType"]["IndexAssignments%s"%tc], tP["bpe"], isPap)

      #kStr += self.bomb(0x13) # after addresses and SRD set
    else:
      tmp = self.vgprPool.checkOut(2, "tmp", self.preventVgprOverflowDuringNewTile)

      if kernel["StreamK"]:
        # StreamK partial tile - offset to tile start index
        tmpOffset = self.sgprPool.checkOut(2, "skStartOffset", preventOverflow=0)
        kStr += inst("s_mul_i32", sgpr(tmpOffset), sgpr("StreamKLocalStart"), "DepthU*%d"%(tP["bpe"]), "StreamK tile start offset")
        strideL = self.strideRef(tc, kernel["ProblemType"]["IndicesSummation"][0])
        kStr += self.s_mul_u64_u32(sgpr(tmpOffset), sgpr(tmpOffset+1), sgpr(tmpOffset), strideL, "StreamK tile start offset")
        if kernel["CheckDimOverflow"] >=2:
          kStr += self.assert_eq(sgpr(tmpOffset+1),0)
        kStr += inst("s_add_u32",  sgpr(tmpOffset+0), sgpr(tmpOffset+0), sgpr("Address%s+0"%tP["tensorChar"]), "accum skOffset term to tilestart")
        kStr += inst("s_addc_u32", sgpr(tmpOffset+1), sgpr(tmpOffset+1), sgpr("Address%s+1"%tP["tensorChar"]), "accum skOffset term to tilestart")
        kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr(tmpOffset+0), "" )
        kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr(tmpOffset+1), "" )
        self.sgprPool.checkIn(tmpOffset)
      else:
        kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("Address%s+0"%tP["tensorChar"]), "" )
        kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("Address%s+1"%tP["tensorChar"]), "" )

      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):

              comment = "gRA%s_%u_%u_%u_%u = addr%s+grO%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp, \
                  tP["tensorChar"], tP["tensorChar"], \
                  para, sPara, perp, sPerp )
              kStr += inst("_v_add_co_u32", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  self.vcc, \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  vgpr(tmp+0), \
                  comment+" (lower)")
              kStr += inst("_v_addc_co_u32", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  self.vcc, \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  vgpr(tmp+1), \
                  self.vcc, \
                  comment+" (upper)")
              #kStr += dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)))
              #kStr += dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)))
              graIdx += self.rpga
      #kStr += "s_endpgm\n"
      self.vgprPool.checkIn(tmp)

    return kStr

  ##############################################################################
  # Global Read Addresses: Increments
  # Define graIncrements, called once for each summation
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    kStr = ""
    tc = tP["tensorChar"]

    dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
    loopChar = self.indexChars[dimIdx]

    stride = self.strideRef(tc, dimIdx)
    isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

    #print (tc, ": loopIdx=", loopIdx, "dimIdx=", dimIdx, "strideIdx=", strideIdx)

    gsu = 1
    if kernel["GlobalSplitU"] > 1 \
        and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
      gsu = kernel["GlobalSplitU"]

    assert(self.unrollIdx == kernel["ProblemType"]["NumIndicesSummation"]-1)
    if loopIdx==self.unrollIdx:
      if self.globalReadIncsUseVgpr:
        if kernel["PackSummationDims"]:
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), \
              stride, \
              "" )
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), \
              0,
              "" )
        else:
          tmpSgpr = self.getTmpSgpr(2).idx()
          kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), \
              "DepthU*%d"%(gsu*tP["bpe"]), stride, \
              "incr%s%s = %s*DepthU*bpe (unrollIdx)"%(tc, loopChar, stride) )
          # TODO - this should be mul-H??
          kStr += inst("s_mov_b32", \
              sgpr(tmpSgpr+1), \
              hex(0), \
              "(carry)")
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), \
              sgpr(tmpSgpr+0), \
              "" )
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), \
              sgpr(tmpSgpr+1), \
              "" )
      else: # not globalReadIncsUseVgpr, ie use SGPR

        if kernel["PackSummationDims"]:
          m = "Bpe%s"%(tc)
        else:
          m = "DepthU*Bpe%s"%(tc)
        if gsu>1:
          m += "*%d"%gsu

        if isMirrorIdx:
          m = "-%s"%(m)

        # multiply by stride, optimizing if unit stride
        if self.isConstUnitStride(stride):
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), m, \
              "incr%s (unrollIdx)"%(tc) )
        else:
          kStr += inst("s_mul_i32", sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), \
              m, stride, \
              "incr%s unrollIdx)"%(tc) )
    else:
      # other summation
      if self.globalReadIncsUseVgpr:
        printExit("NumIndicesSummation=%u not yet supported in assembly unless globalReadIncsUseVgpr==0" \
            % kernel["ProblemType"]["NumIndicesSummation"] )
      else:
        graInc = "GlobalReadIncs%s+%u"%(tc, loopIdx)
        if kernel["PackSummationDims"]:
          # simpler address calculation here - don't need to subtract prev iteration increments
          # since only one iteration
          if isMirrorIdx:
            kStr += inst("s_mul_i32", \
                sgpr(graInc), \
                stride, \
                "-Bpe%s"%tc,
                "<- scale by bpe")
          else:
            kStr += inst("s_lshl_b32", \
                sgpr(graInc), \
                stride, \
                "Bpe%sLog2"%tc,
                "<- scale by bpe")
        else:
          # subtract increments done by the inner iterations
          # may be negative:
          loopIdxPrev = loopIdx + 1
          dimIdxPrev    = kernel["ProblemType"]["IndicesSummation"][loopIdxPrev] # dimension index
          loopCharPrev  = self.indexChars[dimIdxPrev]
          stridePrev = self.strideRef(tc, dimIdxPrev)
          isMirrorIdxPrev = dimIdxPrev in kernel["ProblemType"]["MirrorDims%s"%tc]

          kStr += self.comment("compute globalReadInc for higher-level loop")

          tmpSgpr = self.getTmpSgpr(3).idx()
          # Summations always appear in both A and B, can compute number of iterations just once:
          if loopIdxPrev==self.unrollIdx:
            loopCounterName= self.loopCounterName(kernel, self.unrollIdx)
            if tP["isA"]:
              quotient = loopCounterName
              dividend = "SizesSum+%u"%self.unrollIdx
              divisor = kernel["DepthU"]
              asem = kernel["AssertSummationElementMultiple"]
              gsu = kernel["GlobalSplitU"]
              if self.noTailLoop and ((asem % gsu != 0) or ((asem//gsu) % kernel["DepthU"] != 0)):
                # round up SizesSum/DepthU for noTailLoop case
                kStr += inst("s_add_i32", sgpr(quotient), (divisor - 1), sgpr(dividend), \
                    "round up SizeSum / DepthU" )
                kStr += scalarStaticDivideAndRemainder(quotient, None, quotient, \
                            divisor, tmpSgpr+2, 0)
              else:
                kStr += scalarStaticDivideAndRemainder(quotient, None, dividend, \
                            divisor, tmpSgpr+2, 0)

              if kernel["GlobalSplitU"] > 1:
                kStr += self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgpr)

              kStr += inst("s_mul_i32", sgpr(loopCounterName), sgpr(loopCounterName), \
                        kernel["GlobalSplitU"]*kernel["DepthU"], \
                        "=loopCounterName*DepthU")
            kStr += inst("s_mul_i32", sgpr(graInc), stridePrev, sgpr(loopCounterName), \
                  "tmp <- stride%s%s * myWgUnrollIters" %(tc, loopCharPrev))
          else:
            kStr += inst("s_mul_i32", sgpr(graInc), stridePrev, self.sizeRef(dimIdxPrev), \
                  "tmp <- stride%s%s * size%s%s" %(tc, loopCharPrev, tc, loopCharPrev))

          # subtract amount that previous inner loop will have already incremented:
          # graInc is used as temp for the prev loop calc
          if isMirrorIdx and isMirrorIdxPrev:
            kStr += inst("s_sub_i32", sgpr(graInc), \
                sgpr(graInc), \
                stride, \
                "incr%s%s = <prev-incs> - stride%s%s"%(tc, loopChar, tc, loopChar) )
          elif isMirrorIdx:
            kStr += inst("s_add_i32", sgpr(graInc), \
                stride, \
                sgpr(graInc), \
                "incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) )
            kStr += inst("s_sub_i32", sgpr(graInc), \
                0, \
                sgpr(graInc), \
                "incr%s%s = - (stride%s%s + <prev-incs>)"%(tc, loopChar, tc, loopChar) )
          elif isMirrorIdxPrev:
            kStr += inst("s_add_i32", sgpr(graInc), \
                stride, \
                sgpr(graInc), \
                "incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) )
          else:
            kStr += inst("s_sub_i32", sgpr(graInc), \
                stride, \
                sgpr(graInc), \
                "incr%s%s = stride%s%s - <prev-incs>"%(tc, loopChar, tc, loopChar) )

          kStr += inst("s_lshl_b32", \
              sgpr(graInc), \
              sgpr(graInc), \
              "Bpe%sLog2"%tc,
              "<- scale by bpe")

        if 0 and tP["isB"] and loopIdx==0:
          kStr += self.bomb()
          #kStr += self.assert_ne(sgpr("WorkGroup1"),0)

    #kStr += dump(vgpr("GlobalReadIncs%s"%tP["tensorChar"]))
    #kStr += "s_endpgm\n"
    #if tP["isB"]:
    #  kStr += self.bomb(0x100)
    return "" if self.dontAppendCode else kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    return self.comment1("lwaTileAssignment%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["lwoT"])))

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    kStr = ""
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    kStr += self.comment1("lwaUnrollAssignment%s = %s" % (tP["tensorChar"], vgpr(uReg)))
    if kernel.enabledSplitLDS and kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      if self.inTailLoop:
        subIterReg = self.vgprPool.checkOut(1, "subIterReg")
        kStr += self.comment1("Each wg writes 1/%u of G2L data to LDS"%kernel["DepthULdsDivisor"])
        kStr += inst("v_lshrrev_b32", vgpr(subIterReg), log2(kernel["_DepthULds"]), vgpr(uReg), "sub_G2L_idx = uIdx / DepthU_Compute")
        kStr += inst("v_and_b32", vgpr(uReg), vgpr(uReg), kernel["_DepthULds"]-1, "unrollIdx = unrollIdx % DepthU_Compute")
        tP["gpr"]["subIterReg"] = subIterReg
      else:
        kStr += self.comment1("Each thd writes 1/%u of G2L data to LDS"%kernel["DepthULdsDivisor"])
        kStr += inst("v_lshrrev_b32", vgpr(uReg), log2(kernel["DepthULdsDivisor"]), vgpr(uReg), "sub_G2L_idx = uIdx / DepthULdsDivisor")
    return kStr

  ##############################################################################
  # Local Write Addresses: Release tile related vgpr
  ##############################################################################
  def lwaReleaseTileVgpr(self, kernel, tP):
    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    tP["gpr"]["lwoT"] = None
    if kernel["GlobalSplitU"] > 1:
      self.vgprPool.checkIn(tP["gpr"]["uReg2"])
      tP["gpr"]["uReg2"] = None
    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    tP["gpr"]["uReg"] = None
    if "subIterReg" in tP["gpr"]:
      if tP["gpr"]["subIterReg"] is not None:
        self.vgprPool.checkIn(tP["gpr"]["subIterReg"])
      tP["gpr"]["subIterReg"] = None
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  # uDu: which part of G2L buffer to write to LDS
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP, uDu=0):
    kStr = ""
    tc = tP["tensorChar"]
    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.unrollChar, tP["tileChar"])
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    if kernel["LocalWriteUseSgpr%s"%tc]:
      destVgpr = self.vgprPool.checkOut(1, "destVgpr", self.preventVgprOverflowDuringNewTile)
    else:
      destVgpr = "LocalWriteAddr%s"%tc

    dotInterleave = kernel["LocalDotLayout"]

    # GRVW<1 case, bpe calculation is done before glvw calculation for offset
    useEarlyBpeCalc = tP["glvw"] < 1

    if dotInterleave == 1:
      if kernel["UnrollMajorLDS%s" % tc]:
        lds_stride = kernel["_DepthULds"] + LdsPad
        kStr += inst("v_mul_u32_u24", vgpr(destVgpr), hex(lds_stride), vgpr(tP["gpr"]["lwoT"]), \
            "lw%s%s**(DepthU_Compute + PAD)"%(tP["tensorChar"], self.unrollChar))
        if useEarlyBpeCalc:
          kStr += inst("_v_add_u32", vgpr(destVgpr), vgpr(uReg), vgpr(destVgpr), \
              "lwFO%s = (lw%s%s + lw%s%s*(DepthU+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar) )
        else:
          kStr += inst("_v_add_lshl_u32", vgpr(destVgpr), vgpr(uReg), vgpr(destVgpr), hex(log2(tP["bpe"])), \
              "lwFO%s = (lw%s%s + lw%s%s*(DepthU+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar) )
      else:
        lds_stride = kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad
        kStr += inst("v_mul_u32_u24", vgpr(destVgpr), hex(lds_stride), vgpr(uReg), \
            "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
        if useEarlyBpeCalc:
          kStr += inst("_v_add_u32", vgpr(destVgpr), vgpr(tP["gpr"]["lwoT"]), vgpr(destVgpr), \
              "lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar, tP["tileChar"]) )
        else:
          kStr += inst("_v_add_lshl_u32", vgpr(destVgpr), vgpr(tP["gpr"]["lwoT"]), vgpr(destVgpr), hex(log2(tP["bpe"])), \
              "lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar, tP["tileChar"]) )

      # LdsBlockSizePerPad: add padding
      if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
        tmpSgpr = self.getTmpSgpr(1).idx() if self.initOptLwaTmpSgpr == None else self.initOptLwaTmpSgpr
        if self.isInitCodeOptLW:
          # init code opt case, cannot reuse ureg (it will be used later)
          tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
        else:
          # reuse uReg
          tmpVgpr = uReg
        kStr += vectorStaticDivide(tmpVgpr, destVgpr, kernel["LdsBlockSizePerPad%s"%tc], tmpSgpr, \
          "padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))
        kStr += staticMultiply(vgpr(tmpVgpr), vgpr(tmpVgpr), kernel["LdsPad%s"%tc] * tP["bpe"], sgpr(tmpSgpr), \
          "padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))
        kStr += inst("_v_add_u32", vgpr(destVgpr), vgpr(tmpVgpr), vgpr(destVgpr), \
          "add padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))
        if tmpVgpr != uReg:
          self.vgprPool.checkIn(tmpVgpr)
    else:
      ldlOffsetVgpr = self.vgprPool.checkOut(1, "ldlOffsetVgpr", self.preventVgprOverflowDuringNewTile)
      uRegScrap = self.vgprPool.checkOut(1, "uRegScrap", self.preventVgprOverflowDuringNewTile)
      if self.isInitCodeOptLW:
        # init code opt case, cannot reuse ureg (it will be used later)
        tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
      else:
        # reuse uReg
        tmpVgpr = uReg
      # likely broken for dot4, revisit
      # odd tiles will write to MT, even tiles to normal location
      bpeOffsetLDL = tP["bpe"] if useEarlyBpeCalc else 1 # glvw<1 case, need to multiply bpe here
      bpeComment = "*bpe" if useEarlyBpeCalc else ""
      LdlMinus1Bpe = (kernel["LocalDotLayout"]-1)*bpeOffsetLDL
      kStr += inst("v_and_b32", \
          vgpr(destVgpr), \
          ~(LdlMinus1Bpe), \
          vgpr(tP["gpr"]["lwoT"]), \
          "lwoT & ~((LDL-1)%s)"%bpeComment)
      # uReg bit 1 maps to LDS offset bit 1 (calculateLdsWriteOffset) or LocalWriteAddr (here)
      kStr += inst("v_and_b32", \
          vgpr(uRegScrap), \
          LdlMinus1Bpe, \
          vgpr(uReg), \
          "uReg & (LDL-1)%s)"%bpeComment)
      kStr += inst("v_and_b32", \
          vgpr(tmpVgpr), \
          ~(LdlMinus1Bpe), \
          vgpr(uReg), \
          "uReg & (LDL-1)%s)"%bpeComment)
      kStr += inst("v_and_b32", \
          vgpr(ldlOffsetVgpr), \
          LdlMinus1Bpe, \
          vgpr(tP["gpr"]["lwoT"]), \
          "lwoT & (LDL-1)%s)"%bpeComment)
      kStr += inst("_v_lshl_add_u32", \
          vgpr(tmpVgpr), \
          vgpr(ldlOffsetVgpr), \
          #log2(kernel["LocalDotLayout"]), \
          0, \
          vgpr(tmpVgpr), \
          "shift scrap by LDL")
      kStr += inst("v_mul_u32_u24", \
          vgpr(tmpVgpr), \
          hex(kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad), \
          vgpr(tmpVgpr), \
          "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
      kStr += inst("_v_add_co_u32", \
          vgpr(tmpVgpr), \
          self.vcc, \
          vgpr(uRegScrap), \
          vgpr(tmpVgpr), \
          "add scraps from LDL masking")
      if useEarlyBpeCalc:
        kStr += inst("_v_add_u32", \
            vgpr(destVgpr), \
            vgpr(tmpVgpr), \
            vgpr(destVgpr), \
            "")
      else:
        kStr += inst("_v_add_lshl_u32", \
            vgpr(destVgpr), \
            vgpr(tmpVgpr), \
            vgpr(destVgpr), \
            hex(log2(tP["bpe"])), \
            " *= bpe")
      self.vgprPool.checkIn(uRegScrap)
      if tmpVgpr != uReg:
        self.vgprPool.checkIn(tmpVgpr)
      self.vgprPool.checkIn(ldlOffsetVgpr)

    if tP["isB"]:
      if kernel["LdsOffsetB"] != 0: # LdsOffsetB can be 0 if DirectToVgprA is enabled
        kStr += inst("_v_add_co_u32", \
            vgpr(destVgpr), \
            self.vcc, \
            hex(kernel["LdsOffsetB"]*tP["bpe"]), \
            vgpr(destVgpr), \
            "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
            self.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.bpeAB) )

    # skip releasing tile related vreg here in init code opt case
    if not self.isInitCodeOptLW:
      self.vgprPool.checkIn(tP["gpr"]["lwoT"])
      tP["gpr"]["lwoT"] = None
      if kernel["GlobalSplitU"] > 1:
        self.vgprPool.checkIn(tP["gpr"]["uReg2"])
        tP["gpr"]["uReg2"] = None
    #LSC_ * LSP_
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    validWIPerLoad     = int(kernel[tP["lsc"]] * kernel[tP["lsp"]]/tP["glvw"])
    validBytesPerLoad  = kernel[tP["lsc"]] * kernel[tP["lsp"]] * numBytesPerElement
    maxBytesPerLoad    = int(kernel["NumThreads"] * tP["glvw"] * numBytesPerElement)

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      validBytesPerLoad *= (kernel["NumThreads"] // self.kernel["WavefrontSize"])

    assert (validBytesPerLoad <= maxBytesPerLoad)
    assert (tP["glvw"] < 1 or kernel[tP["lsc"]] * kernel[tP["lsp"]] % tP["glvw"] == 0)

    if validBytesPerLoad != maxBytesPerLoad:
      tmpSgpr = self.getTmpSgpr(1).idx() if self.initOptLwaTmpSgpr == None else self.initOptLwaTmpSgpr
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), validWIPerLoad, \
          "lsc*lsp=%u*%u"%(kernel[tP["lsc"]],kernel[tP["lsp"]] ))
      kStr += inst("v_cmp_lt_u32", \
          self.vcc, \
          vgpr("Serial"), \
          sgpr(tmpSgpr), \
          "fractional: ensure tid < global read tile elements")
      tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(tmpVgpr), hex(self.LdsOOB), "")
      kStr += inst("v_cndmask_b32", \
                  vgpr(destVgpr), \
                  vgpr(tmpVgpr), \
                  vgpr(destVgpr), \
                   self.vcc, \
                   "Mask load so out-of-gr-tile bounds returns 0")
      self.vgprPool.checkIn(tmpVgpr)

    elif self.inTailLoop and kernel.enabledSplitLDS: # where (DepthU for global read) != (DepthU for compute)
      tmpSgpr = self.getTmpSgpr(1).idx() if self.initOptLwaTmpSgpr == None else self.initOptLwaTmpSgpr

      # only for TN tensor + TN lds layout
      assert tP["tlu"] == 0
      kStr += inst("v_cmp_eq_u32",self.vcc, vgpr(tP["gpr"]["subIterReg"]), uDu, "if sub_g2l_idx == %u ?"%uDu)

      ldsOOB = self.vgprPool.checkOut(1, "lds OOB addr", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(ldsOOB), hex(self.LdsOOB), "lds OOB address")
      kStr += inst("v_cndmask_b32", \
                  vgpr(destVgpr), \
                  vgpr(ldsOOB), \
                  vgpr(destVgpr), \
                   self.vcc, \
                   "Mask threads not belonging to current sub_g2l_idx by assigning OOB")
      self.vgprPool.checkIn(ldsOOB)

    if kernel["LocalWriteUseSgpr%s"%tc]:
      # TODO: Can refactor code above to Compute this directly:
      if self.archCaps["CrosslaneWait"]:
        kStr += inst("s_nop 0", "1 wait states required before reading vgpr by lane")
      kStr += inst("v_readfirstlane_b32", \
          sgpr("LocalWriteAddr%s"%tc), \
          vgpr(destVgpr), \
          "Copy lds write address VGPR to SGPR")
      self.vgprPool.checkIn(destVgpr)

    if kernel["FractionalLoad"] and kernel["fractionalPerpOverhang%s"%tc]:
      overhang = kernel["fractionalPerpOverhang%s"%tc]
      validWI = int(overhang*kernel[tP["lsc"]]/tP["glvw"])
      if kernel["FractionalLoad"] == 2:
        mask = "PerpOverhangVcc%s"%tc
      else:
        mask = self.getTmpSgpr(2).idx() if self.initOptLwaTmpSgpr == None else self.initOptLwaTmpSgpr
      kStr += self.comment1("Compute fractional overhang")
      kStr += inst("s_mov_b32", sgpr(mask), validWI, \
          "overhang=%u, validWI=%u" % (overhang, validWI))
      kStr += inst("v_cmp_lt_u32", \
          sgpr(mask,2),
          vgpr("Serial"), \
          sgpr(mask,1),
          "fractional-overhang: some wi write to harmless LDS location")
      if kernel["FractionalLoad"] == 1:
        kStr += inst("v_cndmask_b32", \
                    vgpr("LocalWriteAddrOverhang%s"%tc), \
                    1.0, \
                    vgpr("LocalWriteAddr%s"%tc), \
                    sgpr(mask,2), \
                    "Mask load so out-of-gr-tile bounds returns 0. Note 1.0f=0x3f80000 which is large non-neg int")


    # skip releasing tile related vreg here in init code opt case
    if not self.isInitCodeOptLW:
      self.vgprPool.checkIn(tP["gpr"]["uReg"])
      tP["gpr"]["uReg"] = None
      if "subIterReg" in tP["gpr"]:
        if tP["gpr"]["subIterReg"] is not None:
          self.vgprPool.checkIn(tP["gpr"]["subIterReg"])
        tP["gpr"]["subIterReg"] = None
    # dump lds write offsets
    #if tP["isA"]:
      #kStr += self.dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
      #kStr += self.bomb(-40)

    # invalid local write for performance evaluation only
    if self.enable["InvalidLocalWrite%s"%tc]:
      kStr += inst("v_mov_b32", vgpr(destVgpr), self.LdsOOB, "set out-of-bound addr for performance evaluation only")

    # do not generate local write address code if DirectToVgpr is enabled
    return "" if self.dontAppendCode or kernel["DirectToVgpr%s"%tc] else kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  ##############################################################################
  def lwaFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B
  ##############################################################################
  def lwaDeclareAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Allocate tmpSgpr for initOpt
  ##############################################################################
  def lwaInitOptAllocate(self):
    maxTmpRegNum = 2
    self.initOptLwaTmpSgpr = None
    if self.isInitCodeOptLW:
      self.initOptLwaTmpSgpr = self.sgprPool.checkOutAligned(maxTmpRegNum, 2, preventOverflow=0)

  ##############################################################################
  # Local Write Addresses: Release tmpSgpr for initOpt
  ##############################################################################
  def lwaInitOptRelease(self):
    if self.initOptLwaTmpSgpr:
      self.sgprPool.checkIn(self.initOptLwaTmpSgpr)
      self.initOptLwaTmpSgpr = None

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  def lraTileAssignment(self, kernel, tPA, tPB):
    kStr = ""

    component = Component.LraTileAssignment.find(self)

    tP0 = tPA if tPB["tile01Idx"] else tPB
    tP1 = tPB if tPB["tile01Idx"] else tPA

    if component:
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP0["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        kStr += component(self, kernel, tP0)
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP1["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        kStr += component(self, kernel, tP1)

    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    kStr = ""

    # do not generate local read code if DirectToVgpr is enabled
    tc = tP["tensorChar"]
    if kernel["DirectToVgpr%s"%tc]:
      return kStr

    # allocate resources
    tmpSgpr = self.getTmpSgpr(1).idx()

    # constant
    tile01      = tP["tile01Idx"]
    LdsPad      = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
    divisor     = kernel["SubGroup0"] * kernel["SubGroup1"]
    mtAddPad    = kernel["MacroTile%u" % tile01] + LdsPad

    # final offset
    finalVgpr = vgpr("LocalReadAddr%s"%tc)

    # generate instruction
    if kernel["EnableMatrixInstruction"]:
      # MatrixInstruction case
      # no need to handle sgid here (already handled)
      kStr += staticMultiply(finalVgpr, vgpr(tP["gpr"]["lro"]), tP["bpe"], sgpr(tmpSgpr),  \
              "Final Offset: offset = (lro%s)*bpe" % tile01);
      if kernel["DirectToLds%s" % tc]:
        if kernel["NumLoadsCoalesced%s"%tc] > 1:
          tmp1 = self.vgprPool.checkOut(1,"tmp1")
          tmp2 = self.vgprPool.checkOut(1,"tmp2")

          # address conversion for DirectToLds + NumLoadsCoalesced > 1
          newStr, dummy = self.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val=0, generateAsm=True, \
                                                               finalVgpr=finalVgpr, tmp1=tmp1, tmp2=tmp2)
          kStr += newStr

          self.vgprPool.checkIn(tmp1)
          self.vgprPool.checkIn(tmp2)
    else:
      # non MatrixInstruction case
      # LSU offset
      sgid    = self.vgprPool.checkOut(1) # quotient
      kStr += vectorStaticDivide(sgid, "Serial", divisor, tmpSgpr, \
        "LSU offset: sgid = Serial / subGroup(%u)" % divisor)
      kStr += staticMultiply(vgpr(sgid), vgpr(sgid), mtAddPad, sgpr(tmpSgpr), \
        "LSU offset: lsuoffset = sgid*(MT%u+PAD)"%tile01)
      kStr += staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), kernel["VectorWidth"], sgpr(tmpSgpr), \
        "Final Offset: lr%sOffset * VW" % tc)
      # Final offset
      kStr += inst("_v_add_lshl_u32", finalVgpr, vgpr(sgid), vgpr(tP["gpr"]["lro"]), hex(log2(tP["bpe"])), \
        "Final Offset: offset = (lro%s*VW+lsuoffset)*bpe" % tile01 )
      self.vgprPool.checkIn(sgid)

    # release resources
    self.vgprPool.checkIn(tP["gpr"]["lro"])

    # LdsBlockSizePerPad: add padding
    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] !=0:
      rReg    = self.vgprPool.checkOut(1) # remainder, unused here
      kStr += vectorStaticDivide(rReg, "LocalReadAddr%s"%tc, kernel["LdsBlockSizePerPad%s"%tc], tmpSgpr, \
        "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))
      kStr += staticMultiply(vgpr(rReg), vgpr(rReg), kernel["LdsPad%s"%tc] * tP["bpe"], sgpr(tmpSgpr), \
        "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))
      kStr += inst("_v_add_u32", vgpr("LocalReadAddr%s"%tc), vgpr(rReg), vgpr("LocalReadAddr%s"%tc), \
        "Final Offset: add padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc]))
      self.vgprPool.checkIn(rReg)

    # invalid local read for performance evaluation only
    if self.enable["InvalidLocalRead%s"%tc]:
      kStr += inst("v_mov_b32", finalVgpr, self.LdsOOB, "set out-of-bound addr for performance evaluation only")

    return kStr

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    kStr = ""
    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
    if tP["grcg"]:
      if tP["grcv"]:
        divisorName = tP["lvc"]
      else:
        # Fractional load use the more accurate lsc, multiply by VW later
        divisorName = tP["lsc"]
    else:
      if tP["grcv"]:
        divisorName = tP["lsp"]
      else:
        divisorName = tP["lvp"]
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
          tmpSgpr2 = self.getTmpSgpr(1).idx()
          kStr += inst("v_and_b32", vgpr(tmp1), hex(maskBitsLow), finalVgpr, \
            "Offset rotation for DirectToLds + %s > 1"%tP["lsc"])
          kStr += inst("v_and_b32", vgpr(tmp2), hex(maskBitsHigh), finalVgpr, "")
          kStr += inst("v_lshlrev_b32", vgpr(tmp1), hex(leftShift), vgpr(tmp1), "")
          kStr += inst("v_lshrrev_b32", vgpr(tmp2), hex(rightShift), vgpr(tmp2), "")
          kStr += inst("v_or_b32", vgpr(tmp1), vgpr(tmp1), vgpr(tmp2), "")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr2), hex(maskBitsAll), "")
          kStr += inst("v_not_b32", vgpr(tmp2), sgpr(tmpSgpr2), "")
          kStr += inst("v_and_b32", finalVgpr, vgpr(tmp2), finalVgpr, "")
          kStr += inst("v_or_b32", finalVgpr, vgpr(tmp1), finalVgpr, "")

    return kStr, offset_val

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    if tP["isA"]:
      return self.comment1("N/A")
    else:
      # no local read code if DirectToVgpr is enabled
      # no need to generate add code if LdsOffset is 0
      if kernel["DirectToVgprB"] or kernel["LdsOffset%s"%tP["tensorChar"]] == 0:
        return ""
      return inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          self.vcc, \
          hex(kernel["LdsOffset%s"%tP["tensorChar"]]*tP["bpe"]), \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          " += LdsOffset%s (lower)"%tP["tensorChar"])

  ##############################################################################
  # openShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def openShadowInit(self, kernel):
    kStr = self.getNamedLabelDef("ShadowInitStart")
    if self.kernel["PreloadKernelArguments"] == 1:
      if self.kernel["DelayRemainingArguments"]:
        kStr += self.loadKernelArguments()
      kernArgBytes = self.sgprPool.numKernargSGPRs * 4
      kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for %u bytes of kern args" % kernArgBytes )
    return kStr

  ##############################################################################
  # closeShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def closeShadowInit(self, kernel):
    kStr = ""
    assert(self.doShadowInit and kernel["PrefetchGlobalRead"])

    kStr += self.checkLastIter(kernel)
    if kernel["SuppressNoLoadLoop"]:
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
      lastIterEnd = self.getNamedLabel("LoopEnd%s"%loopChar)
    else:
      lastIterEnd = self.getNamedLabel("PrefetchGlobalLastIterEnd")

    # This branch could potentially be very far e.g. > SIMM16
    kStr += self.comment("after InitC, skip to end of prefetch last iter if numIter==0")
    # use positive offset only long jump
    kStr += self.longBranchScc1(lastIterEnd, positiveOnly=True)

    return kStr

  ##############################################################################
  # longBranch - 32 bit offset
  # s_branch class instructions take a label operand which is truncated to 16 bit
  # If the target label address offset is greater than 16 bits, then
  # we must use a longer 32 bit version.
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranch(self, label, tmpSgpr=None):
    kStr = ""
    if tmpSgpr is None:
      tmpSgpr = self.getTmpSgpr(3).idx()
    positiveLabel = self.getNamedLabelUnique("Positive")
    kStr += inst("s_getpc_B64", sgpr(tmpSgpr,2), "addr of next instr")
    kStr += inst("s_add_i32",  sgpr(tmpSgpr+2), "%s"%label, hex(4), "target branch offset")
    kStr += inst("s_cmp_ge_i32", sgpr(tmpSgpr+2), hex(0), "check positive or negative")
    kStr += inst("s_cbranch_scc1 label_%s" % positiveLabel, "jump when positive")

    # negative offset
    kStr += inst("s_abs_i32",  sgpr(tmpSgpr+2), sgpr(tmpSgpr+2), "abs offset")
    kStr += inst("s_sub_u32",  sgpr(tmpSgpr),   sgpr(tmpSgpr),   sgpr(tmpSgpr+2), "sub target branch offset")
    kStr += inst("s_subb_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "sub high and carry")
    kStr += inst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%label)

    # positive offset
    kStr += "label_%s:%s"%(positiveLabel, self.endLine)
    kStr += inst("s_add_u32",  sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr(tmpSgpr+2), "add target branch offset")
    kStr += inst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "add high and carry")
    kStr += inst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%label)
    return kStr

  ##############################################################################
  # longBranchPositive - 32 bit offset (positive offset only)
  # s_branch class instructions take a label operand which is truncated to 16 bit
  # If the target label address offset is greater than 16 bits, then
  # we must use a longer 32 bit version.
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchPositive(self, label, tmpSgpr=None):
    kStr = ""
    if tmpSgpr is None:
      tmpSgpr = self.getTmpSgpr(3).idx()
    kStr += inst("s_getpc_B64", sgpr(tmpSgpr,2), "addr of next instr")
    kStr += inst("s_add_i32",  sgpr(tmpSgpr+2), "%s"%label, hex(4), "target branch offset")

    # positive offset
    kStr += inst("s_add_u32",  sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr(tmpSgpr+2), "add target branch offset")
    kStr += inst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "add high and carry")
    kStr += inst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%label)
    return kStr

  ##############################################################################
  # longBranchNegative - 32 bit offset (negative offset only)
  # s_branch class instructions take a label operand which is truncated to 16 bit
  # If the target label address offset is greater than 16 bits, then
  # we must use a longer 32 bit version.
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchNegative(self, label, tmpSgpr=None):
    kStr = ""
    if tmpSgpr is None:
      tmpSgpr = self.getTmpSgpr(3).idx()
    kStr += inst("s_getpc_B64", sgpr(tmpSgpr,2), "addr of next instr")
    kStr += inst("s_add_i32",  sgpr(tmpSgpr+2), "%s"%label, hex(4), "target branch offset")

    # negative offset
    kStr += inst("s_abs_i32",  sgpr(tmpSgpr+2), sgpr(tmpSgpr+2), "abs offset")
    kStr += inst("s_sub_u32",  sgpr(tmpSgpr),   sgpr(tmpSgpr),   sgpr(tmpSgpr+2), "sub target branch offset")
    kStr += inst("s_subb_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), 0, "sub high and carry")
    kStr += inst("s_setpc_b64", sgpr(tmpSgpr,2), "branch to %s"%label)
    return kStr

  ##############################################################################
  # longBranchScc0 - 32 bit offset
  # Conditional branch to label when SCC == 0
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc0(self, label, positiveOnly=False, negativeOnly=False, tmpSgpr=None):
    kStr = ""
    noBranchLabel = self.getNamedLabelUnique("NoBranch")
    kStr += inst("s_cbranch_scc1 label_%s" % noBranchLabel, "Only branch on scc0")
    if positiveOnly:
      kStr += self.longBranchPositive(label, tmpSgpr)
    elif negativeOnly:
      kStr += self.longBranchNegative(label, tmpSgpr)
    else:
      kStr += self.longBranch(label, tmpSgpr)
    kStr += "label_%s:%s"%(noBranchLabel, self.endLine)
    return kStr

  ##############################################################################
  # longBranchScc1 - 32 bit offset
  # Conditional branch to label when SCC == 1
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc1(self, label, positiveOnly=False, negativeOnly=False, tmpSgpr=None):
    kStr = ""
    noBranchLabel = self.getNamedLabelUnique("NoBranch")
    kStr += inst("s_cbranch_scc0 label_%s" % noBranchLabel, "Only branch on scc1")
    if positiveOnly:
      kStr += self.longBranchPositive(label, tmpSgpr)
    elif negativeOnly:
      kStr += self.longBranchNegative(label, tmpSgpr)
    else:
      kStr += self.longBranch(label, tmpSgpr)
    kStr += "label_%s:%s"%(noBranchLabel, self.endLine)
    return kStr

  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    kStr = ""
    kStr += self.comment("initC: remove C-tile %u-%u from pool"%(self.startVgprValuC, self.startVgprValuC+self.numVgprValuC))
    self.vgprPool.remove(self.startVgprValuC, self.numVgprValuC, "ValuC")
    numAccvgprs = self.totalAgprs
    if kernel["StoreCInUnroll"]:
      numAccvgprs -= self.startaccValuC1
    self.agprPool.remove(0, numAccvgprs, "ValuC")
    kStr += self.comment("initC: remove AB-tile %u-%u from pool"%(self.startVgprValuA, self.lastValuAB))
    self.vgprPool.remove(self.startVgprValuA, self.lastValuAB - self.startVgprValuA, "ValuAB")
    numCVgpr = max(self.numVgprValuC, numAccvgprs)

    startNumCVgpr = 0
    # init accvgpr opt. use src2=0. No need to initialize accVgpr
    if not self.useInitAccVgprOpt:
      if kernel["LdsInitCVgprs"]:
        tmpAddr = self.vgprPool.checkOut(1,"tmp vgpr for lds init C registers")
        kStr += inst("v_mov_b32", vgpr(tmpAddr), self.LdsOOB, "set out-of-bound addr")

      useVgpr = (not kernel["EnableMatrixInstruction"]) or kernel["MIArchVgpr"]
      if globalParameters["AsmCaps"][self.version]["v_mov_b64"] and (useVgpr or kernel["LdsInitCVgprs"]):
        # use 64 bit mov
        # (dst of v_mov_b64 is vgpr only (no acc support))
        numCVgprM2 = (numCVgpr//2)*2 # multiple of 2
        for i in range(startNumCVgpr, numCVgprM2, 2):
          regStr = vgpr("ValuC+%u"%i, 2) if useVgpr else accvgpr(i, 2)
          if not kernel["LdsInitCVgprs"]:
            kStr += inst("v_mov_b64", regStr, hex(0), "initC")
          else:
            kStr += inst("_ds_load_b64", regStr, vgpr(tmpAddr), "offset:0", "initC")
        # update start for remaining
        startNumCVgpr = numCVgprM2

      copyInsStr = "v_mov_b32" if useVgpr else "v_accvgpr_write"
      for i in range(startNumCVgpr, numCVgpr):
        regStr = vgpr("ValuC+%u"%i) if useVgpr else "acc%u"%i
        if not kernel["LdsInitCVgprs"]:
          kStr += inst(copyInsStr, regStr, hex(0), "initC")
        else:
          kStr += inst("_ds_load_b32", regStr, vgpr(tmpAddr), "offset:0", "initC")

      if kernel["LdsInitCVgprs"]:
        self.vgprPool.checkIn(tmpAddr)

    if kernel["PersistentKernel"]:
      # Move to next serial wg early since SerialWorkGroupIter is checked in several places below including tail loop which has multiple entry points
      # As a result be aware for much of the loop SerialWorkGroupIter points to the next tile not the current one
      kStr += self.comment1("move to next serial WG")
      kStr += inst("s_add_u32", sgpr("SerialWorkGroupIter"), \
          sgpr("SerialWorkGroupIter"), sgpr("GridNumWorkGroups0"), \
          "Move Serial forward by numworkgroups - will map to new wg0/wg1 later")
      if self.prefetchAcrossPersistent0:
        kStr += self.comment1("save PrevWorkGroup for stores here")
        kStr += inst("s_mov_b32", sgpr("PrevWorkGroup0"), sgpr("WorkGroup0"), "save for store code")
        kStr += inst("s_mov_b32", sgpr("PrevWorkGroup1"), sgpr("WorkGroup1"), "save for store code")

    return kStr

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    kStr =""
    if self.unrollIncIsDepthU:
      if kernel["GlobalSplitU"] > 1:
        tmpSgpr = self.getTmpSgpr(3).idx()
        quotient = "UnrollLoopLastIter"
        dividend = self.loopSizeRef(kernel, self.unrollIdx) # sumSize
        divisor = kernel["DepthU"]
        kStr += scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgpr, 0)
        kStr += self.calculateLoopNumIterGsu(kernel, "UnrollLoopLastIter", tmpSgpr)
        kStr += inst ("s_mul_i32", sgpr("UnrollLoopLastIter"), sgpr("UnrollLoopLastIter"), "DepthU", "scale")
      else:
        kStr += inst ("s_mov_b32", sgpr("UnrollLoopLastIter"), self.loopSizeRef(kernel, self.unrollIdx), "init")

      if kernel["PackSummationDims"]:
        if kernel["GlobalSplitU"]>1:
          kStr += inst ("s_mov_b32", sgpr("GsuNumIter%s"%self.loopChar(kernel,self.unrollIdx)), \
                        sgpr("UnrollLoopLastIter"), "save innermost iters for later unpacking")
        for idx in range(self.otherSummations):
          kStr += inst ("s_mul_i32", sgpr("UnrollLoopLastIter"), sgpr("UnrollLoopLastIter"), \
                            self.loopSizeRef(kernel, idx), "")

    return kStr

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  # Output: Sets sgpr(StaggerRowMask)
  ##############################################################################
  def declareStaggerParms(self, kernel):
    kStr=""
    tmpSgprRef = self.getTmpSgpr(2)
    tmpSgpr = tmpSgprRef.idx()
    if self.staggerU:
      # this could be dynamic?
      if kernel["StaggerUMapping"] == 0:
        staggerInput = sgpr("WorkGroup0")
      elif kernel["StaggerUMapping"] == 1:
        staggerInput = sgpr("WorkGroup1")
      elif kernel["StaggerUMapping"] == 2:
        staggerInput = sgpr("WorkGroup2")
      elif kernel["StaggerUMapping"] == 3:
        # wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0
        wgSerial = tmpSgpr
        tmp = tmpSgpr+1
        kStr += inst("s_mul_i32", sgpr(wgSerial), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), \
          "wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0")
        kStr += inst("s_mul_i32", sgpr(wgSerial), sgpr(wgSerial), sgpr("WorkGroup2"), "")
        kStr += inst("s_mul_i32", sgpr(tmp), sgpr("NumWorkGroups0"), sgpr("WorkGroup1"), "")
        kStr += inst("s_add_u32", sgpr(wgSerial), sgpr(wgSerial), sgpr(tmp), "")
        kStr += inst("s_add_u32", sgpr(wgSerial), sgpr(wgSerial), sgpr("WorkGroup0"), "")
        staggerInput = sgpr(wgSerial)
      elif kernel["StaggerUMapping"] == 4:
        staggerInput = -1

      kStr += inst("s_and_b32", sgpr("StaggerUIter"), sgpr("OrigStaggerUIter"), \
                    staggerInput, \
                    "Compute actual stagger start for this tile")
      if kernel["_staggerStrideShift"] > 0: # generate code only when it is necessary
        kStr += inst("s_lshl_b32", sgpr("StaggerUIter"), sgpr("StaggerUIter"), \
                  kernel["_staggerStrideShift"], "shift by StaggerUStride")

      if kernel["StreamK"]:
        # Set stagger=0 for partial tiles to avoid using stagger larger than workload
        kStr += inst("s_cmp_gt_u32", sgpr("StreamKLocalStart"), 0, "does wg start tile?")
        kStr += inst("s_cmov_b32", sgpr("StaggerUIter"), 0, "set stagger=0 for partial tiles")
        kStr += inst("s_cmp_lt_u32", sgpr("StreamKLocalEnd"), sgpr("ItersPerTile"), "does wg finish tile?")
        kStr += inst("s_cmov_b32", sgpr("StaggerUIter"), 0, "set stagger=0 for partial tiles")

    return kStr

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  def calculateStagger(self, kernel, tP):
    imod = Code.Module("calculateStagger")
    tc = tP["tensorChar"]

    if self.staggerU:
      assert (kernel["BufferLoad"])

      staggerTmpRef = self.getTmpSgpr(2)
      staggerTmp = staggerTmpRef.idx()

      #---
      imod.addComment1("SRDs += (StaggerUIter) * GlobalReadIncs%s+%u"% (tc, self.unrollIdx))

      # Calculate the stagger byte offset
      imod.addCode(self.s_mul_u64_u32(
                sgpr(staggerTmp), sgpr(staggerTmp+1), \
                sgpr("StaggerUIter"), sgpr("GlobalReadIncs%s+%u"%(tc, self.unrollIdx)), \
                " stagger byte offset"))

      # Amount of bytes to add to get back to start.
      # on the llop iteration which matches StaggerUIter, this offset added instead of GlobalReadInc
      imod.addCode(self.s_mul_u64_u32(sgpr("WrapU%s+0"%tc), sgpr("WrapU%s+1"%tc), \
                self.loopCounter(kernel, self.unrollIdx), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                "Number of bytes accessed by the unroll loop"))

      imod.addInst("s_sub_u32", sgpr("WrapU%s+0"%tc),  \
                sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                sgpr("WrapU%s+0"%tc), \
                "remove one iteration")
      imod.addInst("s_subb_u32", sgpr("WrapU%s+1"%tc), \
                0, \
                sgpr("WrapU%s+1"%tc), \
                "remove one iteration")

      imod.addCode(self.incrementSrd(kernel, tP, sgpr(staggerTmp), sgpr(staggerTmp+1)))

      if tP["isB"]:
        # Convert passed in S' to S for easy loop comparison.  S=S-(PGR-1)'
        imod.addInst("s_add_u32", sgpr("StaggerUIter"), sgpr("StaggerUIter"), \
                  (2 if kernel["PrefetchGlobalRead"] else 1), \
                  "Subtract (PGR-1); StaggerUIter now contains target iteration to wrap")
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
    imod = Code.Module("removeStagger")
    if self.staggerU:
      tc = tP["tensorChar"]
      tmpRef = self.getTmpSgpr(4)
      tmp = tmpRef.idx()
      tmpForInc = tmp
      tmpForExtra = tmp + 2
      # need to use extra 64bit mul to avoid negative value by subtraction
      #  ((3 or 2) - StaggerUIter) * GlobalReadIncs
      #   -> (3 or 2) * GlobalReadIncs - StaggerUIter * GlobalReadIncs
      extra = 3 if kernel["PrefetchGlobalRead"] else 2
      # tmpForInc = extra * GlobalReadIncs
      imod.addInst("s_mov_b32", sgpr(tmpForExtra), extra, "")
      imod.addCode(self.s_mul_u64_u32(sgpr(tmpForInc), sgpr(tmpForInc+1), \
                  sgpr(tmpForExtra), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                  "%u * GlobalReadIncs"%extra))
      # tmpForExtra = StaggerUIter * GlobalReadIncs
      imod.addCode(self.s_mul_u64_u32(sgpr(tmpForExtra), sgpr(tmpForExtra+1), \
                  sgpr("StaggerUIter"), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                  "StaggerUIter * GlobalReadIncs"))
      # tmpForInc = tmpForInc - tmpForExtra = (extra - StaggerUIter) * GlobalReadIncs
      imod.addInst("s_sub_u32", sgpr(tmpForInc), sgpr(tmpForInc), sgpr(tmpForExtra), "start offset S in bytes")
      imod.addInst("s_subb_u32", sgpr(tmpForInc+1), sgpr(tmpForInc+1), sgpr(tmpForExtra+1), "start offset S in bytes")
      # -= WrapU
      imod.addInst("s_sub_u32", sgpr(tmpForInc), sgpr(tmpForInc), sgpr("WrapU%s"%tc), "S - WrapU")
      imod.addInst("s_subb_u32", sgpr(tmpForInc+1), sgpr(tmpForInc+1), sgpr("WrapU%s+1"%(tc)), "S - WrapU")

      imod.addCode(self.incrementSrd(kernel, tP, sgpr(tmpForInc), sgpr(tmpForInc+1)))

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
  def calculateLoopNumIterGsu(self, kernel, destName, tmpSgpr):
    kStr = ""

    loopCounter = sgpr(destName)
    quotient = destName
    remainder = "GSUSumIdx+1" # numIterPerWgRemainder
    dividend = tmpSgpr+2 # numIterMyWg
    divisor = kernel["GlobalSplitU"]
    if log(divisor,2).is_integer():
      kStr += inst("s_mov_b32", sgpr(dividend), loopCounter, "copy for divide IterGsu" )
      kStr += scalarStaticDivideAndRemainder(quotient, remainder, dividend, divisor, tmpSgpr, 1)
    else:
      qReg = self.vgprPool.checkOut(1,"qReg")
      rReg = self.vgprPool.checkOut(1,"rReg")
      dReg = self.vgprPool.checkOut(1,"dReg")
      kStr += inst("v_mov_b32", vgpr(dReg), loopCounter, "copy for divide IterGsu")
      kStr += vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr)
      kStr += inst("v_readfirstlane_b32", sgpr(quotient), vgpr(qReg), "")
      kStr += inst("v_readfirstlane_b32", sgpr(remainder), vgpr(rReg), "")
      self.vgprPool.checkIn(dReg)
      self.vgprPool.checkIn(rReg)
      self.vgprPool.checkIn(qReg)

    # if gsuSumIdx < numIterPerWgRemainder
    kStr += inst("s_add_u32", sgpr(tmpSgpr), "1", \
                  loopCounter, "tmp<-numIterMyWg+" )
    kStr += inst("s_cmp_lt_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
        "gsuSumIdx < numIterPerWgRemainder" )
    kStr += inst("s_cmov_b32", loopCounter, sgpr(tmpSgpr), "numIterMyWg++ if needed" )

    return kStr

  ##############################################################################
  # Calculate Loop Num Iter
  # loopIdx is the index of the loop (used for contractions with multiple summations)
  # 0 is outermost; self.unrollIdx is the unroll index.
  # -1 is tail loop (used only for the unroll loop)
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx, isPap):
    kStr = ""

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    loopChar = self.indexChars[loopDim]

    ########################################
    # Tail Loop
    if tailLoop:
      tmpSgpr = self.getTmpSgpr(3).idx()
      if self.prefetchAcrossPersistent0:
        loopCounterName = "TailLoopCounter"
      else:
        loopCounterName = self.loopCounterName(kernel, loopIdx)
      kStr += "\n"
      if kernel["SuppressNoLoadLoop"]:
        # If the tail loop is suppressed, then final iterations will have moved the Srd base forward
        # (and also moved back the srd shadow limit) and slammed Limit to 0, so need to 'undo'
        # those increments - see setTailSrd
        assert(kernel["PrefetchGlobalRead"] == 1) #if >1 would need a multiply here
        kStr += inst("s_cmp_eq_u32", sgpr("OrigLoopCounter"), 0, "completely skipped unroll loop?")
        kStr += inst("s_cselect_b32", sgpr(tmpSgpr+0), 0, sgpr("GlobalReadIncsA"), "force to 0?")
        kStr += inst("s_cselect_b32", sgpr(tmpSgpr+1), 0, sgpr("GlobalReadIncsB"), "force to 0?")
        kStr += self.setTailSrd(kernel, self.tPA, sgpr(tmpSgpr+0))
        kStr += "\n"
        kStr += self.setTailSrd(kernel, self.tPB, sgpr(tmpSgpr+1))
        kStr += "\n"
        #kStr += self.bomb()

      if self.tailLoopInNLL:
        kStr += self.comment1("Tail Loop in NoLoadLoop")

      kStr += "%s//numIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
      # size % DepthU
      kStr += scalarStaticDivideAndRemainder(tmpSgpr+2, loopCounterName, "SizesSum+%u"%loopIdx, kernel["DepthU"], tmpSgpr, 2)
      loopCounter = sgpr(loopCounterName)

      if self.tailLoopInNLL:
        # tailLoop in NLL case, numIter will not be 0. Convert 0 to DepthU
        kStr += inst("s_cmp_eq_u32", loopCounter, \
            hex(0), "numIter%s == 0"%loopChar )
        kStr += inst("s_cmov_b32", loopCounter, hex(kernel["DepthU"]),  "Convert 0 to DepthU in tail loop NLL case")

      if kernel["LocalSplitU"] > 1 and (not (kernel["EnableMatrixInstruction"] and kernel["MatrixInstK"] > 1)):
        # (size % DepthU) + LSU - 1
        kStr += inst("s_add_u32", loopCounter, hex(kernel["LocalSplitU"]-1), loopCounter, "(size % DepthU) + LSU - 1" )
        # /= LSU
        dividend = loopCounter
        kStr += scalarStaticDivideAndRemainder( loopCounterName, None, dividend, kernel["LocalSplitU"], tmpSgpr, 0)

      # skip tail loop if StreamK WG not processing final iteration
      if kernel["StreamK"]:
        # Check if tile finished
        kStr += inst("s_cmp_lt_u32", sgpr("StreamKLocalEnd"), sgpr("ItersPerTile"), "Check if WG processes final iteration of tile")
        kStr += inst("s_cmov_b32", loopCounter, hex(0), "This WG not completing tile")

      # if GSU numIter=0 if gsuSumIdx != remainder
      if kernel["GlobalSplitU"] > 1:
        kStr += inst("s_cmp_lg_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
            "gsuSumIdx == numIterPerWgRemainder" )
        kStr += inst("s_cmov_b32", loopCounter, hex(0), "numIter=0 if gsuSumIdx!=remainder")

      # do not use early exit here in tailLoop in NLL case
      if not self.tailLoopInNLL:
        # if tail numIter == 0 skip altogether
        skipTailLoopLabel = self.getNamedLabel("SkipTailLoop%s"%(loopChar) )
        kStr += inst("s_cmp_eq_u32", loopCounter, \
            hex(0), "numIter%s == 0"%loopChar )
        kStr += inst("s_cbranch_scc1 %s"\
            % skipTailLoopLabel, \
            "skip to end of tail loop b/c numIter==0")
        kStr += inst("s_mov_b32", sgpr("OrigLoopCounter"), 0, \
            "repurpose to count each localRead increment")

    ########################################
    # Unrolled Loop
    elif loopIdx == self.unrollIdx:
      loopCounterName = self.loopCounterName(kernel, loopIdx)
      loopCounter = sgpr(loopCounterName)
      if not self.do["PreLoop"]: kStr += ".endif\n"

      sumSize = "SizesSum+%u"%loopIdx
      #sumSize = self.sumSize(kernel, loopIdx)
      if self.unrollIncIsDepthU:
        kStr += inst("s_mov_b32", loopCounter, 0,\
                  "init loop counter, unrollIncIsDepthU mode")

      else:
        # TODO - use named arguments
        tmpSgpr = self.getTmpSgpr(3).idx()
        quotient = loopCounterName
        dividend = sumSize
        divisor = kernel["DepthU"]
        asem = kernel["AssertSummationElementMultiple"]
        gsu = kernel["GlobalSplitU"]
        if kernel["StreamK"]:
          # Use StreamK params for loop count
          kStr += inst("s_sub_u32", sgpr(loopCounterName), sgpr("StreamKLocalEnd"), sgpr("StreamKLocalStart"), "StreamK loop counter = localEnd - localStart")
          # Adjust loop count for tail loop
          if not self.noTailLoop:
            loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
            kStr += scalarStaticDivideAndRemainder(tmpSgpr, tmpSgpr+1, "SizesSum+%u"%self.unrollIdx, kernel["DepthU"], tmpSgpr+2, 2)
            kStr += inst("s_cmp_eq_u32", sgpr(tmpSgpr+1), hex(0), "numIter%s == 0"%loopChar )
            kStr += inst("s_cselect_b32", sgpr(tmpSgpr), 0, 1, "check if size uses tail loop")
            kStr += inst("s_cmp_eq_u32", sgpr("StreamKLocalEnd"), sgpr("ItersPerTile"), "Check if WG processes final iteration of tile")
            kStr += inst("s_cselect_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), 0, "this WG runs tail loop")
            kStr += inst("s_sub_u32", sgpr(loopCounterName), sgpr(loopCounterName), sgpr(tmpSgpr), "Adjust loop counter for tail loop")
        elif self.noTailLoop and ((asem % gsu != 0) or ((asem//gsu) % kernel["DepthU"] != 0)):
          # round up SizesSum/DepthU for noTailLoop case
          kStr += inst("s_add_i32", sgpr(quotient), (divisor - 1), sgpr(dividend), \
              "round up SizeSum / DepthU" )
          kStr += scalarStaticDivideAndRemainder(quotient, None, quotient, divisor, tmpSgpr, 0)
        else:
          kStr += scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgpr, 0)
        # if GSU numIter++ if gsuSumIdx < remainder
        if kernel["GlobalSplitU"] > 1:
          kStr += self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgpr)

      kStr += inst("s_mov_b32", sgpr("OrigLoopCounter"), \
                loopCounter, \
                "copy loop counter")
      # We can use OrigLoopCounter & 1 instead of using another register for BreakAtEvenIter
      # calculate once and save: will this problem size exit at oddIter or evenIter?
      #if self.prefetchAcrossPersistent0 and kernel["ExpandPointerSwap"] and not isPap:
      #  kStr += inst("s_and_b32", sgpr("BreakAtEvenIter"), sgpr("OrigLoopCounter"), \
      #            0x1, "save unroll loop start position - copy1 or copy2")
    elif not kernel["PackSummationDims"]:
      # other summation, not unroll loop
      #printExit("no assembly support for 2+ dimensional summation")
      kStr += self.comment("%sother summation, numIter%s = size%s" \
          % (self.indent, loopChar, loopChar))
      loopCounter = self.loopCounter(kernel, loopIdx)
      kStr += inst("s_mov_b32", loopCounter, \
                sgpr("SizesSum+%u"%loopIdx), \
                "init loop counter")

    if not tailLoop:
      # compute element edge:
      problemType = kernel["ProblemType"]
      zpB = next((zpi for zpi in problemType["ZeroPadB"] if zpi[1] == loopDim), None)
      assert zpB==None # not supported

      zpA = next((zpi for zpi in problemType["ZeroPadA"] if zpi[1] == loopDim), None)
      if zpA:
        tc = 'A'
        (freeDim,sumDim) = zpA[:2]
        freeDimChar = globalParameters["IndexChars"][freeDim]
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        elementEdge = "ElementEdge%s%s" % (tc,sumDimChar)
        tmpSgpr = self.getTmpSgpr(1).idx()
        kStr += "\n"
        kStr += self.comment1("ElementEdge%s%s" % (tc, sumDimChar))
        kStr += inst("s_mul_i32", sgpr(elementEdge), \
                  self.strideRef('A', freeDim), \
                  self.sizeRef(freeDim), \
                  "strideFree*sizeFree")

        kStr += inst("s_sub_u32", sgpr(tmpSgpr), \
                   self.sizeRef(sumDim), 1, \
                   "strideSum*(sizeSum-1), step1")
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), \
                  self.strideRef('A', sumDim), \
                  sgpr(tmpSgpr),\
                   "strideSum*(sizeSum-1), step2")

        kStr += inst("s_add_u32", sgpr(elementEdge), \
                  sgpr(elementEdge), \
                  sgpr(tmpSgpr), \
                   "strideFree*sizeFree + strideSum*(sizeSum-1)")

        kStr += inst("s_lshl_b32", \
                  sgpr("ElementEdge%s%s"%(tc, sumDimChar)), \
                  sgpr("ElementEdge%s%s"%(tc, sumDimChar)), \
                  "Bpe%sLog2"%tc, "scale by bpe")

        kStr += inst("s_sub_u32", sgpr(elementEdge), \
                  sgpr(elementEdge), \
                  sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                  "sub PadStart*Bpe")
        kStr += inst("s_sub_u32", sgpr(elementEdge), \
                  sgpr(elementEdge), \
                  sgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                  "Final0: (strideFree*sizeFree - strideSum*(sizeSum-1))*BPE - padStart - padEnd")

        if kernel["PackSummationDims"]:
          kStr += inst("s_mov_b32", sgpr("Iter"+sumDimChar), 0, "init iterX")

        #assert(self.groOffsetInMacroTile==0)

    return kStr

  ##############################################################################
  # Open Loop
  # uDu: 'None' means not generating branching label which decides which part of G2L
  #      buffer to write to LDS
  ##############################################################################
  def openLoop(self, kernel, loopIdx, uDu=None, noLabelGen=False, beginLabelOnly=False):
    kStr = ""
    # TODO - rewrite this function to simplify control-flow between tail-loop / unroll loop
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
      self.inTailLoop = True
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if not tailLoop and not noLabelGen:
      kStr += "%s:\n" % self.getNamedLabel("openLoop%s"%loopChar)
    loopLabelBegin = self.getNamedLabel("%sLoopBegin%s%s"%("Tail" if tailLoop else "", loopChar, "_G2L%s"%uDu if uDu is not None else "" ) )
    loopLabelEnd = self.getNamedLabel("%sLoopEnd%s%s"%("Tail" if tailLoop else "", loopChar, "_G2L%s"%uDu if uDu is not None else "") )

    if beginLabelOnly:
      # generate only beginLabel, then, return
      kStr += "%s:%s" % (loopLabelBegin, self.endLine)
      return kStr

    # is numIter at least 1? otherwise skip to end
    # PGL needs a skip-check here if not bufferload
    # If kernel["SuppressNoLoadLoop"] we don't have a special loop for the 'last iter'
    loopCounter = self.loopCounter(kernel, loopIdx)
    if tailLoop:
      if self.prefetchAcrossPersistent0:
        loopCounter = sgpr("TailLoopCounter")
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
      # comment out since redundant
      """
      kStr += inst("s_cmp_le_u32", \
          loopCounter, \
          hex(endCounter), \
          "LoopCounter%s < EndCounter"%(loopChar) )
      kStr += inst("s_cbranch_scc1 %s"%loopLabelEnd, \
          "do not enter Loop%s"%loopChar )

      kStr += inst("s_mov_b32", sgpr("OrigLoopCounter"), 0, \
          "repurpose to count each localRead increment")
      """

      # LSU not all threads will do summation
      # Exec flag is not used for LSU + MFMA + MIK>1
      useExecFlag = kernel["LocalSplitU"] > 1 and (not (kernel["EnableMatrixInstruction"] and kernel["MatrixInstK"] > 1))
      if useExecFlag:
        tmpSgpr = self.getTmpSgpr(1).idx()
        kStr += self.comment("apply exec mask for LSU")
        tmpVgpr = self.vgprPool.checkOut(1,"tmpVgpr")
        sgId = self.vgprPool.checkOut(1,"sgId")
        numIter = self.vgprPool.checkOut(1,"numIter")
        divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
        kStr += vectorStaticDivide(sgId, "Serial", divisor, tmpSgpr)
        kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr("SizesSum+0"), "sizeU to vgpr")
        divisor = kernel["DepthU"]
        kStr += vectorStaticRemainder(numIter, tmpVgpr, divisor, tmpSgpr)
        #kStr += dump(vgpr(sgId))
        #kStr += dump(vgpr(numIter))
        self.vgprPool.checkIn(tmpVgpr)
        #self.tailNumIter = numIter
        #self.vgprPool.checkIn(numIter)
        # thread is active is sgId < numIter % LocalSplitU

      # begin loop
      if not noLabelGen:
        kStr += "%s:%s" % (loopLabelBegin, self.endLine)

      # LSU mask for this iteration
      if useExecFlag:
        kStr += inst("_v_cmpx_lt_u32", self.vcc, \
            vgpr(sgId), vgpr(numIter), "sgId < numIter")
        kStr += inst("_v_add_co_u32", vgpr(sgId), self.vcc, hex(kernel["LocalSplitU"]), \
            vgpr(sgId), "sgId+=LSU")
        self.vgprPool.checkIn(sgId)
        self.vgprPool.checkIn(numIter)
        #kStr += dump(vgpr(sgId))

    else: # not tailloop:

      if loopIdx == self.unrollIdx:
        # 1 loop check is necessary only when (AssertSummationElementMultiple/GlobalSplitU) % (DepthU * 2) != 0
        asem = kernel["AssertSummationElementMultiple"]
        gsu = kernel["GlobalSplitU"]
        if kernel["PrefetchGlobalRead"] == 2 and ((asem % gsu != 0) or ((asem//gsu) % (kernel["DepthU"] * 2) != 0)):
          if not self.unrollIncIsDepthU:
            kStr += inst("s_cmp_eq_u32", \
                loopCounter, \
                hex(endCounter-1), \
                "LoopCounter%s < EndCounter"%(loopChar) )
          else:
            kStr += inst("s_cmp_ge_u32", \
                loopCounter, \
                sgpr("UnrollLoopLastIter"), \
                "LoopCounter%s > EndCounter"%(loopChar) )
          toPGR1 = self.getLabelNum("toPGR1")
          kStr += inst("s_cbranch_scc1 label_%04u"%toPGR1, "PGR=2 but only 1 loop, toPGR1")

        if self.unrollIncIsDepthU:
          if kernel["PrefetchGlobalRead"] == 2:
            tmpSgpr = self.getTmpSgpr(1).idx()
            kStr += inst("s_add_u32", sgpr(tmpSgpr),\
                loopCounter, \
                 "DepthU", "")
            loopCounter = sgpr(tmpSgpr)
          kStr += inst("s_cmp_ge_u32", \
              loopCounter, \
              sgpr("UnrollLoopLastIter"), \
              "LoopCounter%s > EndCounter"%(loopChar) )
        else:
          kStr += inst("s_cmp_le_u32", \
              loopCounter, \
              hex(endCounter), \
              "LoopCounter%s < EndCounter"%(loopChar) )
        jumpLabel = loopLabelEnd
        if kernel["PrefetchGlobalRead"]==2 and (not kernel["SuppressNoLoadLoop"]) and kernel["ExpandPointerSwap"]:
          # PGR=2 and EPS and no SuppressNoLoadLoop case, need to jump to EvenExit
          jumpLabel = self.getNamedLabel("LoopEnd%s_evenexit"%(loopChar) )
        kStr += inst("s_cbranch_scc1 %s"%jumpLabel, \
            "do not enter Loop%s"%loopChar )

      # No need, we will always start from LoopCopy1
      # if self.prefetchAcrossPersistent and kernel["ExpandPointerSwap"]:
      #   kStr += inst("s_cmp_eq_u32", sgpr("BreakAtEvenIter"), 1, "test if BreakAtEvenIter == 1 ?")
      #   kStr += inst("s_cbranch_scc1", self.getLabelTarget("LoopCopy1"), "if == 1, then start from LoopCopy1")

      if not noLabelGen:
        kStr += "%s:%s" % (loopLabelBegin, self.endLine)

      if loopIdx != self.unrollIdx:
        # reset LRO since these may have changed due to odd-iter exit ?
        if kernel["PrefetchGlobalRead"]:
          kStr += self.comment1("openLoop - reset LRO for possible odd-iter exit")
          kStr += self.localReadResetOffsets(kernel, self.tPA)
          kStr += self.localReadResetOffsets(kernel, self.tPB)

    return kStr

  ##############################################################################
  # Close Loop
  # finalLoop : final unroll loop
  # uDu: 'None' means not generating branching label which decides which part of G2L
  #      buffer to write to LDS
  ##############################################################################
  def closeLoop(self, kernel, loopIdx, finalLoop, loopCopies, uDu=None, emitEndLabelOnly=False, oddLabel=False, \
                skipCondJumpCounter=-1, isOptNLL=False, skipJump=False):
    kStr = ""
    if emitEndLabelOnly:
      loopIdx = self.unrollIdx
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      kStr += "%s:%s"%(self.getNamedLabel("SkipTailLoop%s"%(loopChar)), self.endLine)
      return kStr

    finalJump = "s_cbranch_scc0"
    jumpNeeded = True

    #kStr += self.indent + self.syncStr + self.endLine
    #kStr += "s_endpgm\n"
    tailLoop = loopIdx < 0
    tailLabelStr = ""
    if tailLoop:
      loopIdx = self.unrollIdx
      tailLabelStr = "Tail"
    optNLLLabelStr = ""
    # tail loop case only, add OptNLL to the label
    if isOptNLL and tailLoop:
      optNLLLabelStr = "_OptNLL"
    extraStr = "_G2L%s"%uDu if uDu is not None else ""
    loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    loopLabelBegin = self.getNamedLabel("%sLoopBegin%s%s"%(tailLabelStr, loopChar, extraStr + optNLLLabelStr) )
    loopLabelEnd = self.getNamedLabel("%sLoopEnd%s%s"%(tailLabelStr, loopChar, extraStr + optNLLLabelStr) )
    loopLabelEndOddExit = self.getNamedLabel("%sLoopEnd%s%s_oddexit"%(tailLabelStr, loopChar, optNLLLabelStr) )
    loopLabelEndEvenExit = self.getNamedLabel("%sLoopEnd%s%s_evenexit"%(tailLabelStr, loopChar, optNLLLabelStr) )
    loopCounter = self.loopCounter(kernel, loopIdx)
    if tailLoop:
      if self.prefetchAcrossPersistent0:
        loopCounter = sgpr("TailLoopCounter")

      unrollInc      = 1
      numReadsIterCoalescedA = self.numReadsIterCoalescedA
      numReadsIterCoalescedB = self.numReadsIterCoalescedB
      numReadsIterCoalesced = max(numReadsIterCoalescedA, numReadsIterCoalescedB)
      KinInnerUnroll = kernel["InnerUnroll"]
      if kernel["EnableMatrixInstruction"]:
        unrollInc      *= kernel["MatrixInstK"] * numReadsIterCoalesced
        KinInnerUnroll *= kernel["MatrixInstK"]
        lsu            = kernel["LocalSplitU"]
        if lsu > 1 and kernel["MatrixInstK"] > 1:
          # exec flag is not used and loop counter is not divided by lsu
          # need to multiply inc values by lsu
          unrollInc      *= lsu
          KinInnerUnroll *= lsu

      asem = kernel["AssertSummationElementMultiple"]
      gsu = kernel["GlobalSplitU"]
      if ((asem%gsu == 0) and (asem//gsu) % KinInnerUnroll == 0):
        unrollInc *= kernel["InnerUnroll"]
      elif (kernel["LocalDotLayout"] > 1) and (kernel["InnerUnroll"] == kernel["LocalDotLayout"]):
        unrollInc *= kernel["InnerUnroll"]

      skipCondJump = skipJump
      if skipCondJumpCounter >= 0 and (skipCondJumpCounter%numReadsIterCoalesced < numReadsIterCoalesced - 1):
        # skip conditional jump when numReadsIterCoalesced > 1 and skipCondJumpCounter is not the last in numReadsIterCoalesced
        # to support numReadsIterCoalesced > 1, MatrixInstK * numReadsIterCoalesced needs to be executed
        # e.g.) MatrixInstK=4, numReadsIterCoalesced=2
        #    skipCondJumpCounter==0 case: execute K=0,2,4,6  (here, no exit(means skip cond jump) to execute odd K(1,3,5,7))
        #    skipCondJumpCounter==1 case: execute K=1,3,5,7  (here, all K=0-7 are done. check condition and jump if tail loop is done)
        # skipCondJump=True is not to exit after skipCondJumpCounter==0.
        skipCondJump = True
      jumpNeeded = not skipCondJump
      if not skipCondJump:
        kStr += self.comment("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

        kStr += inst("s_sub_i32", \
            loopCounter, \
            loopCounter, \
            hex(unrollInc), \
            "dec counter%s (tailLoop)"%(loopChar) )

        # Track # LDS reads?
        kStr += inst("s_add_u32", \
          sgpr("OrigLoopCounter"), \
          sgpr("OrigLoopCounter"), \
          hex(unrollInc),
          "inc counter%s"%(loopChar) )

        endCounter = 0
        kStr += inst("s_cmp_le_i32", \
            loopCounter, \
            hex(endCounter), \
          "counter%s<=%d"%(loopChar,endCounter) )
    else: # not tailloop
      kStr += self.comment("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

      if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
        assert (not kernel["SuppressNoLoadLoop"]) # not accounting for end-of-loop iteration change here in deprecated mode

        if kernel["PrefetchGlobalRead"] == 2:
          tmpSgpr = self.getTmpSgpr(1).idx()
          kStr += inst("s_add_u32", sgpr(tmpSgpr),\
              loopCounter, \
               "DepthU", "")
          kStr += inst("s_cmp_ge_u32", \
              sgpr(tmpSgpr), \
              sgpr("UnrollLoopLastIter"), \
              "LoopCounter%s + DU < EndCounter. Go to PGR1"%(loopChar) )
        else:
          kStr += inst("s_cmp_ge_u32", \
              loopCounter, \
              sgpr("UnrollLoopLastIter"), \
            "counter%s==0"%(loopChar) )
      else:
        # If PrefetchGlobalRead=1 the loads in the loop prefetch next macro-tile
        # For the final trip through the unroll loop we need to ensure those loads stay in bounds.

        # One technique is to create a copy of the unroll loop with all loads removed.
        # However buffer load doesn't need this loop copy since we OOB loads can be suppressed by buffer limit hardware
        # So can do one more iteration (endCounter==0) in the main unroll loop, and adjust the pointer
        # increments appropriately.
        # Also sum idx other than unroll always compare against 0 (there is no PGR to account for)
        if kernel["PrefetchGlobalRead"] == 1 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.unrollIdx:
          endCounter = 1
        elif kernel["PrefetchGlobalRead"] == 2 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.unrollIdx:
          endCounter = 2
        else:
          endCounter = 0

        asem = kernel["AssertSummationElementMultiple"]
        gsu = kernel["GlobalSplitU"]
        if (asem % gsu == 0) and (asem//gsu) % (kernel["DepthU"] * 2) == 0 and endCounter > 0:
          # if AssertSummationElementMultiple/GlobalSplitU is multiple of DepthU*2, loop exit is necessary only once in 2 Loop iterations
          #  In endCounter % 2 == 1 case, exit at lc % 2 == 0 (= oddLabel). It means no exit if not oddLabel
          #  In endCounter % 2 == 0 case, exit at lc % 2 == 1 (= not oddLabel). It means no exit if oddLabel
          # No exit case, no code is necessary except for final Loop

          # decrement by loopCopies if PGR=2 and StaggerU == 0, else 1
          oneLoop = loopIdx==0 and finalLoop and (not kernel["ExpandPointerSwap"])
          decValue = loopCopies if kernel["PrefetchGlobalRead"]==2 and kernel["StaggerU"] == 0 else 1
          decCode = inst("s_sub_u32", \
              loopCounter, loopCounter, \
              decValue, \
              "dec counter%s"%(loopChar) )
          condCode = inst("s_cmp_eq_i32", \
              loopCounter, \
              hex(endCounter), \
            "counter%s==%d"%(loopChar,endCounter) )

          noExit = False

          # noExit can be used only for multiple unroll loops case
          if not oneLoop:
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
            if finalLoop:
              # No exit and finalLoop case, use s_branch (no condition)
              finalJump = "s_branch"
            else:
              jumpNeeded = False

          kStr += decCode
          kStr += condCode
        else:
          kStr += inst("s_sub_u32", \
              loopCounter, loopCounter, \
              1, \
              "dec counter%s"%(loopChar) )

          kStr += inst("s_cmp_eq_i32", \
              loopCounter, \
              hex(endCounter), \
            "counter%s==%d"%(loopChar,endCounter) )

    if oddLabel and tailLoop:
      loopLabelEnd = loopLabelEndOddExit
    jumpLabel = loopLabelEnd
    if not tailLoop and not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
      # in this case, odd or/and even code is generated and use odd/even exit to avoid skipping odd/even code
      # (end label is generated after odd/even code)
      jumpLabel = loopLabelEndOddExit if oddLabel else loopLabelEndEvenExit

    # tail + SplitLds branch code
    kStrSLDS = ""
    if tailLoop and kernel.enabledSplitLDS:
      tailLoopLabelEnd = self.getNamedLabel(
        "TailLoopEnd%s%s"%(loopChar, "_G2L%s"%(kernel["DepthULdsDivisor"]-1) if kernel.enabledSplitLDS else "") )
      kStrSLDS += inst("s_cbranch_scc1", tailLoopLabelEnd, "break Loop%s"%loopChar)
      thresForNextSubLoop = (uDu+1)*(kernel["_DepthULds"])
      kStrSLDS += inst("s_cmp_ge_u32", sgpr("OrigLoopCounter"), thresForNextSubLoop,
        "OrigLoopCounter >= %u (G2L buffer %u/%u)"%(thresForNextSubLoop, uDu, kernel["DepthULdsDivisor"]) )

    if not finalLoop:
      if kStrSLDS != "":
        # tail + SplitLds branch case
        kStr += kStrSLDS
      elif jumpNeeded:
        # just an exit check, else fall through to the next loop copy
        kStr += inst("s_cbranch_scc1 %s"%(jumpLabel), "exit Loop%s"%loopChar )
    else: #finalLoop:

      # add tail + SplitLds branch case code (if exists)
      kStr += kStrSLDS

      if jumpNeeded:
        kStr += inst("%s %s"%(finalJump, loopLabelBegin), \
            "restart Loop%s"%(loopChar ))

      if not tailLoop and loopIdx == self.unrollIdx:
        oddIterPreCode = Code.Module()
        oddIterCode = Code.Module()
        evenIterPreCode = Code.Module()
        evenIterCode = Code.Module()
        if not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
          oddIterPreCode.addText("%s: // unroll loop odditer exit\n" % (loopLabelEndOddExit))
          # In this case we kept the 'no-load' loop which has LDS offsets assuming first bank of LDS
          # if we exit the main loop at an odd iter - need to swap LDS read pointers
          # so the ds_reads read from the 'high' buffer of LDS
          oddIterPreCode.addComment1("Select high bank of LDS")
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprA"]:
            oddIterCode.addText(self.localReadSwapOffsets(kernel, False, self.tPA))
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprB"]:
            oddIterCode.addText(self.localReadSwapOffsets(kernel, False, self.tPB))

          evenIterPreCode.addText("%s: // unroll loop eveniter exit\n" % (loopLabelEndEvenExit))
          # generate even code here (so far, for PrefetchGlobalRead=2 only)
          if kernel["PrefetchGlobalRead"]==2:
            # Generate local write address code only for PrefetchGlobalRead==2 (localWriteSwapOffsets does nothing if DirectToVgpr is enabled)
            # Code is unnecessary if DirectToLds is enabled, but internal SwapOffset is necessary if useInitAccVgprOpt is True
            if kernel["DirectToLdsA"]:
              if self.useInitAccVgprOpt:
                self.localWriteSwapOffsets(kernel, True, self.tPA)
            else:
              evenIterCode.addText(self.localWriteSwapOffsets(kernel, False, self.tPA))
            if kernel["DirectToLdsB"]:
              if self.useInitAccVgprOpt:
                self.localWriteSwapOffsets(kernel, True, self.tPB)
            else:
              evenIterCode.addText(self.localWriteSwapOffsets(kernel, False, self.tPB))
            # swap internal write pointer as well (except for useInitAccVgprOpt case)
            if not self.useInitAccVgprOpt:
              evenIterCode.addText(self.localWriteSwapOffsets(kernel, True, self.tPA))
              evenIterCode.addText(self.localWriteSwapOffsets(kernel, True, self.tPB))

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

        kStr += str(firstPreCode)
        kStr += str(firstCode)

        # if secondCode exist, add jump to skip secondCode
        if secondCode.count():
          kStr += inst("s_branch %s"%loopLabelEnd, \
              "exit unroll loop%s (and skip second exit code)"%(loopChar ))
        kStr += str(secondPreCode)
        kStr += str(secondCode)

      kStr += "%s:%s" % (loopLabelEnd, self.endLine)

      if tailLoop and not self.tailLoopInNLL:
        if kernel["PersistentKernel"] or kernel["StreamK"] or len(kernel["ProblemType"]["IndicesSummation"]) > 1:
          # recover the 'damage' done to LRO:
          stmp = self.getTmpSgpr(1).idx()

          # if LRA is backed-up before (wlr case), we simply restore the addr (sub inc*loop doesn't work)
          tPList = []
          if self.oriLraA != None:
            if not kernel["DirectToVgprA"]: # no local read code if DirectToVgpr is enabled
              kStr += inst("v_mov_b32", vgpr("LocalReadAddrA"), vgpr(self.oriLraA), "restore LRA")
            self.vgprPool.checkIn(self.oriLraA)
            self.oriLraA = None
          else:
            tPList.append(self.tPA)
          if self.oriLraB != None:
            if not kernel["DirectToVgprB"]: # no local read code if DirectToVgpr is enabled
              kStr += inst("v_mov_b32", vgpr("LocalReadAddrB"), vgpr(self.oriLraB), "restore LRA")
            self.vgprPool.checkIn(self.oriLraB)
            self.oriLraB = None
          else:
            tPList.append(self.tPB)
          for tP in tPList:
            tc     = tP["tensorChar"]
            LdsPad = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
            inc    = kernel["LocalSplitU"]*(kernel["MacroTile%s"%tc]+LdsPad)*tP["bpe"]

            # aligned with localReadInc
            if kernel["EnableMatrixInstruction"]:
              if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
                inc = kernel["LocalSplitU"] * tP["bpe"]
              # No need to *= K, because LoopCounter is increased by K each time
              # inc *= kernel["MatrixInstK"]

            if not kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
              kStr += inst("s_mov_b32", sgpr(stmp), inc, "tailloop lds offset")
              kStr += inst("s_mul_i32", sgpr(stmp), sgpr("OrigLoopCounter"), sgpr(stmp), "scale by mul")
              kStr += inst("_v_sub_u32", vgpr("LocalReadAddr%s"%tc), vgpr("LocalReadAddr%s"%tc), sgpr(stmp), "remove lro damage")
          # if LWA is backed-up before, we simply restore the addr
          if self.oriLwaA != None:
            if not kernel["DirectToVgprA"]: # no local write code if DirectToVgpr is enabled
              kStr += inst("v_mov_b32", vgpr("LocalWriteAddrA"), vgpr(self.oriLwaA), "restore LWA")
            if not kernel["DirectToVgprB"]: # no local write code if DirectToVgpr is enabled
              kStr += inst("v_mov_b32", vgpr("LocalWriteAddrB"), vgpr(self.oriLwaB), "restore LWA")
            self.vgprPool.checkIn(self.oriLwaA)
            self.vgprPool.checkIn(self.oriLwaB)
            self.oriLwaA = None
            self.oriLwaB = None

    # restore all threads
    if tailLoop and kernel["LocalSplitU"] > 1 and (not (kernel["EnableMatrixInstruction"] and kernel["MatrixInstK"] > 1)):
      sgprCnt = self.laneSGPRCount
      waveSize = kernel["WavefrontSize"]
      kStr += self.comment("restore full exec mask")
      fullExec = self.getTmpSgpr(sgprCnt).idx()
      activeMask = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
      kStr += inst("s_mov_b{}".format(waveSize), sgpr(fullExec,sgprCnt), activeMask, "restore all threads active")
      kStr += inst("s_or_saveexec_b{}".format(waveSize),  sgpr(fullExec,sgprCnt), sgpr(fullExec,sgprCnt), "full mask -> exec" )
    return kStr

  ##############################################################################
  def openLoopCopy(self, kernel, lc):
    return self.getLabelDef("LoopCopy%u"%(lc+1) )

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self, kernel, label = None, isOptNLL = False):
    kStr = ""

    kStr += "%s:\n" % (self.getNamedLabelUnique("Summation_End") if label is None else label)

    if kernel["StorePriorityOpt"]:
      kStr += inst("s_setprio 0", "optimization store")

    vbegin = self.startVgprValuA
    vsize = self.lastVgprForReads - self.startVgprValuA

    self.vgprPool.add(vbegin, vsize, "endSummation")
    kStr += self.comment1("endSummation: add vgpr [%u...%u) to pool" % \
            (vbegin, vbegin+vsize))

    # TODO this undef limits ability to define sgprs closer to the time it is used
    lastRegTag=None
    for i in range(self.lastPostLoopSgpr, self.sgprPool.size()):
      regTag = self.sgprPool.pool[i].tag
      if regTag != lastRegTag:
        lastRegTag = regTag
        if self.sgprPool.pool[i].status == RegisterPool.Status.InUse:
          kStr += self.undefineSgpr(regTag)

    if self.db["InitVgpr"] & 0x2:
      #kStr += self.vgprPool.initTmps(self.initVgprValue)
      kStr += self.vgprPool.initTmps(self.initVgprValue,start=0, stop=100)
    if 0:
      for i in range(0,16+1):
         #kStr += inst("v_mov_b32", vgpr(21), hex(self.initVgprValue), "hack tmp in pool")
         kStr += inst("v_mov_b32", vgpr(21), vgpr(21), "hack tmp in pool")

    # this doesn't seem to do anything - not being aggressive with lastPostLoopSgpr
    if self.db["InitSgpr"] & 0x2:
      kStr += self.sgprPool.initTmps(self.initSgprValue)

    if self.db["ConservativeWaitCnt"] & 0x10:
      kStr += "s_barrier // debug" + self.endLine
      kStr += "s_waitcnt lgkmcnt(0) & vmcnt(0)" + self.endLine
      if self.archCaps["SeparateVscnt"]:
        kStr += "s_waitcnt_vscnt null, 0" + self.endLine

    if kernel["SuppressNoLoadLoop"]:
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "wait for all summation activity")
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

    # copy accumulated C from agpr to vgpr
    if kernel["EnableMatrixInstruction"]:
      #TODO avoid s_nop if its possible
      #instCycles = kernel["MatrixInstM"] // 2 # 32x32 is 64 cycles, 16x16 is 32 cycles, 4x4 is 8 cycles
      #kStr += "s_nop %u\n" % instCycles
      kStr += self.MapAcctoArchRegs(kernel,option=0, isOptNLL=isOptNLL)
      if kernel["MIArchVgpr"] or kernel["StoreCInUnroll"]:
        if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
          kStr += self.MoveMIoutToArch(kernel)
        else:
          kStr += self.MulMIoutAlphaToArch(kernel)

    return kStr

  ##############################################################################
  # src A,B str for MFMA
  ##############################################################################
  def generateSrcStrForMFMA(self, kernel, tP, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, idxAB, bk=None):
    tc = tP["tensorChar"]

    numVgprValuPerBlock = kernel["MIWaveTile%c"%tc] * kernel["MIInputPerThread"] * tP["bpe"] // self.bpr
    numIterPerCoalescedRead = self.numIterPerCoalescedReadA if tP["isA"] else self.numIterPerCoalescedReadB
    numReadsIterCoalesced   = self.numReadsIterCoalescedA   if tP["isA"] else self.numReadsIterCoalescedB
    numVgprPerBlock         = self.numVgprG2LA              if tP["isA"] else self.numVgprG2LB
    numVgprPerBlock //= 2

    # calculate vgprBufferA_new ( or B) and offset for DirectToVgpr. Use u instead of m (number of local prefetch buffer does not matter)
    m_or_u = u if kernel["DirectToVgpr%c"%tc] else m
    vgprBuffer_new = (m_or_u//numIterPerCoalescedRead)*numIterPerCoalescedRead
    vgprBuffer_new_offset = m_or_u%numIterPerCoalescedRead*innerUnroll*vgprPerInput
    # DirectToVgpr + pack special case
    # offset vgprBuffer_new
    packDTV = self.packDTVA if tc == "A" else self.packDTVB
    if packDTV:
      # DTV + pack case, offset bufferIdx for local read packing instructions
      numBi = kernel["LoopIters"]
      vgprBuffer_new += vregSetIdx * numBi

    iui_new = (iui//numReadsIterCoalesced)*numReadsIterCoalesced
    iui_new_offset = iui%numReadsIterCoalesced*vgprPerInput
    ab_new = idxAB*vgprPerInput*numReadsIterCoalesced
    abStr = "Valu%c_X%u_I%u+%u+%u+%u" % (tc, vgprBuffer_new, iui_new, ab_new, vgprBuffer_new_offset, iui_new_offset)
    packDTV = self.packDTVA if tP["isA"] else self.packDTVB
    if kernel["DirectToVgpr%c"%tc] and not packDTV:
      # overwrite aStr/bStr for DirectToVgpr (except for pack DTV case)
      ab_new += vregSetIdx * numVgprPerBlock + ( vgprBuffer_new * innerUnroll) * numVgprValuPerBlock
      abStr  = "G2L%c+%u+%u" % (tc, ab_new, vgprBuffer_new_offset)

    if bk != None:
      abStr += "+%u"%(bk)
    return abStr

  ##############################################################################
  # MFMA Iteration
  ##############################################################################
  def mfmaIter(self, kernel, u, innerUnroll, vregSetIdx, lastKinloop=False, tail=False, firstIter=False):
    imod = Code.Module("mi")
    shiftK = Code.Module("shiftK")
    m = ((u) % (self.numVgprBuffer+1)) % kernel["LoopIters"] # local to use for MACs

    miInputType      =  kernel["ProblemType"]["F32XdlMathOp"] if kernel["EnableF32XdlMathOp"] else kernel["ProblemType"]["DataType"]

    # calculate constant
    is_mfma          = globalParameters["AsmCaps"][self.version]["HasMFMA"]

    numRegistersIn   = miInputType.numRegisters()
    numRegistersOut  = kernel["MIRegPerOut"]
    loopCounterName  = self.loopCounterName(kernel, self.unrollIdx)
    accs_per_wave    = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] \
                       / self.kernel["WavefrontSize"] * numRegistersOut
    dividerFortidInK = kernel["MatrixInstN"] * kernel["MatrixInstB"]
    numMIInput       = kernel["MIInputPerThread"]
    miInTypeName     = "bf16" if kernel["ProblemType"]["Fp16AltImpl"] else miInputType.toNameAbbrev() # v_mfma_[...xK]<InType>
    miInTypeName     = "iu8" if ((not is_mfma) and miInTypeName == "i8") else miInTypeName
    miOutTypeName    = miInputType.MIOutputTypeNameAbbrev() # v_mfma_<OutType>..
    miOutTypeName    = miOutTypeName if is_mfma else kernel["ProblemType"]["ComputeDataType"].toNameAbbrev()

    vgprPerInput     = int(numMIInput * numRegistersIn)
    shiftPerElement  = int(numRegistersIn * 32)
    s_nop            = 0
    accStoreCIdx     = self.startaccValuC1 if kernel["StoreCInUnroll"] and lastKinloop else 0


    if tail and self.prefetchAcrossPersistent0:
      loopCounterName = "TailLoopCounter"

    # alloc vgpr
    abReg   = None

    if (numRegistersIn < 1) and ((kernel["UnrollMajorLDSA"] == False) or (kernel["UnrollMajorLDSB"] == False)):
      s_nop = 2

    # here we remap index to where it read for wider local read
    # ex. if we read 2 iteration at a time,
    #   original   : _ds_load_b64  valuA_X0_I0
    #   read 2 iter: _ds_load_b128 valuA_X0_I0 (we read valuA_X0_I0 and valuA_X1_I0)
    # instead of using valuA_X1_I0, we use valuA_X0_I0+2 as mfma input

    numReadsIterCoalescedA = self.numReadsIterCoalescedA
    numReadsIterCoalescedB = self.numReadsIterCoalescedB

    # handle multiple K element in MFMA instruction
    if tail and kernel["MatrixInstK"] > 1:
      if is_mfma: # mfma
        # skip K mask code if BufferLoad + TLU
        # BufferLoad + TLU case, global read for out of range K is already 0 because out of range K is always out of array
        # (this is not true in TLU=False case or not BufferLoad case)
        # An exception case is DirectToLds. DTL load does not store 0 in LDS for out of range K.
        # We still need mask operation for DTL case
        needKMaskForA = not (kernel["BufferLoad"] and self.tPA["tlu"] and (not kernel["DirectToLdsA"]))
        needKMaskForB = not (kernel["BufferLoad"] and self.tPB["tlu"] and (not kernel["DirectToLdsB"]))
        if needKMaskForA or needKMaskForB:
          shiftK.addComment1("tail loop mfma iter %u: numReadsIterCoalescedA=%u, numReadsIterCoalescedB=%u"%(u, numReadsIterCoalescedA, numReadsIterCoalescedB))
          numReadsIterCoalesced = max(numReadsIterCoalescedA, numReadsIterCoalescedB)
          kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
          tmpSgpr = self.getTmpSgpr(3).idx()
          shiftK.addCode(vectorStaticRemainder(kReg, "Serial", kernel["WavefrontSize"], tmpSgpr))
          shiftK.addCode(vectorStaticDivide(kReg, kReg, dividerFortidInK, tmpSgpr))
          shiftK.addCode(staticMultiply(vgpr(kReg), vgpr(kReg), numMIInput * numReadsIterCoalesced, sgpr(tmpSgpr)))
          kStepForCoalesced = (u%numReadsIterCoalesced) * numMIInput
          if kStepForCoalesced > 0:
            shiftK.addCode(inst("_v_add_u32", vgpr(kReg), hex(kStepForCoalesced), vgpr(kReg), \
                                "k += (u%%numReadsIterCoalesced) * numMIInput"))
          lsu = kernel["LocalSplitU"]
          if lsu > 1:
            # LocalSplitU case, add sgid*kInterval*numReadsIterCoalesced to kReg
            kInterval = (kernel["WavefrontSize"] // dividerFortidInK) * numMIInput
            sgId = self.vgprPool.checkOut(1,"sgId")
            divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
            shiftK.addCode(vectorStaticDivide(sgId, "Serial", divisor, tmpSgpr))
            #shiftK.addCode(vectorStaticRemainder(sgId, sgId, lsu, tmpSgpr)) # lsu must be power of 2 to use same reg for src and dest
            shiftK.addCode(staticMultiply(vgpr(sgId), vgpr(sgId), kInterval*numReadsIterCoalesced, sgpr(tmpSgpr)))
            shiftK.addCode(inst("_v_add_u32", vgpr(kReg), vgpr(sgId), vgpr(kReg), "k += sgid * kInterval*numReadsIterCoalesced"))
            self.vgprPool.checkIn(sgId)

          # replace 0 for differnet thread
          shiftK.addCode(inst("v_cmp_ge_i32", sgpr(tmpSgpr, 2), vgpr(kReg), sgpr(loopCounterName), "check K index >= Size L"))
          for bk in range(0, vgprPerInput):
            if needKMaskForA:
              for a in range(0, kernel["MIWaveTileA"]):
                for iui in range(0, innerUnroll):
                  aStr_base = self.generateSrcStrForMFMA(kernel, self.tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a, bk)
                  aStr = vgpr(aStr_base, 1)
                  shiftK.addCode(inst("v_cndmask_b32", aStr, aStr, hex(0), sgpr(tmpSgpr, 2), "set 0 if K_idx >= sizeL"))
            if needKMaskForB:
              for b in range(0, kernel["MIWaveTileB"]):
                for iui in range(0, innerUnroll):
                  bStr_base = self.generateSrcStrForMFMA(kernel, self.tPB, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, b, bk)
                  bStr = vgpr(bStr_base, 1)
                  shiftK.addCode(inst("v_cndmask_b32", bStr, bStr, hex(0), sgpr(tmpSgpr, 2), "set 0 if K_idx >= sizeL"))

          # replace 0 for same thread
          # new logic to set partial 0
          # do not shift loaded value to skip setting 0 code for BufferLoad + TLU
          # if we shift loaded value, the location (K=0,1,...) does not match skipping shift for BufferLoad+TLU
          #  e.g.) set 0 to K=2 and 3
          #    (1) shift case: 16bit left shift (K1 K0 0 0)
          #    (2) skip shift: 0 bit shift (0 0 K1 K0)
          # instead, we use and operation to set 0 for out of range K
          #  e.g.) I8 case (numMIInput=4, vgprPerInput=1)
          #   if loopCounter < numMIInput, we need partial mask to set 0
          #   assuming loopCounter = 2, we need to set 0 for K=2 and 3
          #    Bit 0- 7: K=0 (keep)
          #    Bit 8-15: K=1 (keep)
          #    Bit16-23: K=2 (set 0)
          #    Bit24-31: K=3 (set 0)
          #   to set 0 for K=2,3, we need to generate mask=0x0000ffff.
          #    mask = 0xffffffff >> ((loopCounter%numMIInput) * shiftPerElement)
          #   newMask = select 0xffffffff if loopCounterName < kReg else mask (apply mask for loopCounterName>= kReg)
          #   newValue = current & newMask
          #   then, masked value is (0 0 K1 K0) and matches skip shift case
          # mask value is same for all A,B input. calculate mask only once at the beginning
          if numMIInput > 1:
            abReg   = self.vgprPool.checkOutAligned(vgprPerInput, 2 if vgprPerInput>1 else 1, "abReg")
            shiftK.addCode(inst("_v_sub_u32",    vgpr(kReg), sgpr(loopCounterName), vgpr(kReg), "get distance between size and k index"))
            shiftK.addCode(inst("v_cmp_lt_i32", sgpr(tmpSgpr,2), vgpr(kReg), numMIInput, "set partial 0 if distance less than input per thread"))
            shiftK.addCode(inst("s_and_b32",    sgpr(tmpSgpr+2), sgpr(loopCounterName), numMIInput-1, "get inputs for edge thread"))
            shiftK.addCode(inst("s_sub_u32",    sgpr(tmpSgpr+2), numMIInput, sgpr(tmpSgpr+2), "use shift to fill 0 for outside element"))
            shiftK.addCode(inst("s_lshl_b32",   sgpr(tmpSgpr+2), sgpr(tmpSgpr+2), log2(shiftPerElement), "use shift to fill 0 for outside element"))
            for bk in range(0, vgprPerInput):
              shiftK.addCode(inst("v_mov_b32", vgpr(abReg+bk), "-1", "set 0xffffffff"))
            shiftK.addCode(inst("v_lshrrev_b%u" % (vgprPerInput*32), vgpr(abReg, vgprPerInput), sgpr(tmpSgpr+2), vgpr(abReg, vgprPerInput), "rshift mask for partial k"))
            for bk in range(0, vgprPerInput):
              shiftK.addCode(inst("v_cndmask_b32", vgpr(abReg+bk), "-1", vgpr(abReg+bk), sgpr(tmpSgpr, 2), "select shifted mask for partial k"))
            if needKMaskForA:
              for a in range(0, kernel["MIWaveTileA"]):
                for iui in range(0, innerUnroll):
                  for bk in range(0, vgprPerInput):
                    aStr_base = self.generateSrcStrForMFMA(kernel, self.tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, a, bk)
                    aStr = vgpr(aStr_base, 1)
                    shiftK.addCode(inst("v_and_b32", aStr, aStr, vgpr(abReg+bk), ""))
            if needKMaskForB:
              for b in range(0, kernel["MIWaveTileB"]):
                for iui in range(0, innerUnroll):
                  for bk in range(0, vgprPerInput):
                    bStr_base = self.generateSrcStrForMFMA(kernel, self.tPB, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, b, bk)
                    bStr = vgpr(bStr_base, 1)
                    shiftK.addCode(inst("v_and_b32", bStr, bStr, vgpr(abReg+bk), ""))
          # release register
          self.vgprPool.checkIn(kReg)
          # use s_nop only when actual k mask code is generated here
          s_nop = 2

      else: # wmma
        iui = 0

        abReg      = self.vgprPool.checkOutAligned(2, 2, "abReg")
        sgprShift  = self.getTmpSgpr(3).idx()
        sgpr64bIdx = sgprShift + 1
        sgprMask   = sgprShift + 2

        shiftK.addCode(inst("s_sub_i32", sgpr(sgprShift), sgpr(loopCounterName), 1, "calculate 64bit groups index"))
        shiftK.addCode(inst("s_lshr_b32", sgpr(sgpr64bIdx), sgpr(sgprShift), log2(64 // (numRegistersIn * 32)), "calculate 64bit groups index"))
        shiftK.addCode(inst("s_and_b32", sgpr(sgprShift), sgpr(sgprShift), int((64 // (numRegistersIn * 32))-1), "calculate shift value"))
        shiftK.addCode(inst("s_sub_i32", sgpr(sgprShift), int((64 // (numRegistersIn * 32))-1), sgpr(sgprShift), "calculate shift value"))
        shiftK.addCode(inst("s_lshl_b32", sgpr(sgprShift), sgpr(sgprShift), log2(numRegistersIn * 32), "calculate shift value"))

        for it in range(int((kernel["MatrixInstK"] * numRegistersIn) // 2)): # handle 64 bit per iteration
          shiftK.addCode(inst("v_cmp_eq_i32", sgpr(sgprMask), sgpr(sgpr64bIdx), it, 'handle this 64bit group: part 1'))
          for a in range(0, kernel["MIWaveTileA"]):
            aStr = vgpr("ValuA_X%u_I%u+%u+%u" % (m, iui, a*vgprPerInput, it*2), 2)
            shiftK.addCode(inst("v_lshlrev_b64", vgpr(abReg,2), sgpr(sgprShift), aStr, f"shfit for ValuA[{it*2}:{it*2+1}]"))
            for bk in range(2):
              aStr = vgpr("ValuA_X%u_I%u+%u+%u+%u" % (m, iui, a*vgprPerInput, it*2, bk))
              shiftK.addCode(inst("v_cndmask_b32", aStr, aStr, vgpr(abReg+bk), sgpr(sgprMask), "shift if in this 64b group"))
          for b in range(0, kernel["MIWaveTileB"]):
            bStr = vgpr("ValuB_X%u_I%u+%u+%u" % (m, iui, b*vgprPerInput, it*2), 2)
            shiftK.addCode(inst("v_lshlrev_b64", vgpr(abReg,2), sgpr(sgprShift), bStr, f"shfit for ValuB[{it*2}:{it*2+1}]"))
            for bk in range(2):
              bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u" % (m, iui, b*vgprPerInput, it*2, bk))
              shiftK.addCode(inst("v_cndmask_b32", bStr, bStr, vgpr(abReg+bk), sgpr(sgprMask), "shift if in this 64b group"))
          if it > 0:
            shiftK.addCode(inst("v_cmp_lt_i32", sgpr(sgprMask), sgpr(sgpr64bIdx), it, 'handle this 64bit group: part 2'))
            for a in range(0, kernel["MIWaveTileA"]):
              for bk in range(2):
                aStr = vgpr("ValuA_X%u_I%u+%u+%u+%u" % (m, iui, a*vgprPerInput, it*2, bk))
                shiftK.addCode(inst("v_cndmask_b32", aStr, aStr, 0, sgpr(sgprMask), "shift if in this 64b group"))
            for b in range(0, kernel["MIWaveTileB"]):
              for bk in range(2):
                bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u" % (m, iui, b*vgprPerInput, it*2, bk))
                shiftK.addCode(inst("v_cndmask_b32", bStr, bStr, 0, sgpr(sgprMask), "shift if in this 64b group"))

        s_nop = 2

    # add shiftK code to the first mfma item
    shiftKStr = "".join([str(x) for x in shiftK.flatitems()])

    nopStr = ""
    if s_nop != 0:
      nopStr = "s_nop %u\n" % (s_nop - 1)
    # tail casa, always need to generate nop.
    # add nop to shiftKStr code to always generate nop code
    if tail:
      shiftKStr += nopStr
      nopStr = ""

    imod.addCode(nopStr)

    # release register
    if abReg is not None: self.vgprPool.checkIn(abReg)

    # conditions to reverse idxInner if idxOuter is odd (for better HW efficiency)
    # disable reverse idxInner if one of the following condition is true
    # - PGR=2 and u == LoopIters - 1 and DTV
    # - wmma
    # - complex type
    enableReverseInner = True
    if (kernel["PrefetchGlobalRead"] == 2 and u == kernel["LoopIters"] - 1 and \
        (kernel["DirectToVgprA"] or kernel["DirectToVgprB"])) or \
       (not is_mfma) or \
       (kernel["ProblemType"]["DataType"].isComplex()):
      enableReverseInner = False

    prevAccIdx = -1
    for iui in range(0, innerUnroll):
      firstIterIui0 = firstIter and iui == 0 # InnerUnroll case, use const 0 for src1 only if iui == 0
      zgemmVaddSrcCheck = [[], [], []] # to avoid generating redundant v_add
      outer = 1
      # swap inner loop and outer loop so that idxA comes outer
      if self.swapMfmaInnerLoop:
        outer = 0
      inner = 1 - outer # inner is the opposite of outer
      for idxOuter in range(0, kernel["MIWaveTile"][outer]):
        for idxInner in range(0, kernel["MIWaveTile"][inner]):
          # reverse idxInner if idxOuter is odd (for better HW efficiency)
          reverseInner = idxOuter & 1
          if enableReverseInner and reverseInner:
            idxInner = kernel["MIWaveTile"][inner] - 1 - idxInner
          idx0 = idxInner
          idx1 = idxOuter
          if self.swapMfmaInnerLoop:
            idx0, idx1 = idx1, idx0
          accIdx   = idx1 * kernel["MIWaveTile"][0] + idx0
          accStart = accIdx * accs_per_wave
          accEnd   = accStart + accs_per_wave - 1
          idxA     = idx0 if self.tPB["tile01Idx"] else idx1
          idxB     = idx1 if self.tPB["tile01Idx"] else idx0
          aStr_base = self.generateSrcStrForMFMA(kernel, self.tPA, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, idxA)
          bStr_base = self.generateSrcStrForMFMA(kernel, self.tPB, innerUnroll, vregSetIdx, vgprPerInput, m, u, iui, idxB)
          aStr     = vgpr(aStr_base, vgprPerInput)
          bStr     = vgpr(bStr_base, vgprPerInput)
          Str0     = aStr if self.tPB["tile01Idx"] else bStr
          Str1     = bStr if self.tPB["tile01Idx"] else aStr

          mfmaComponent = Component.MFMA.find(self)
          if kernel["ProblemType"]["DataType"].isComplex():
            # override because complex mul is emulated by 4 mfma insts
            # TODO: adopt component system
            miInTypeName = miOutTypeName #"f32" for SingleComplex, "f64" for DoubleComplex
            ccA = kernel["ProblemType"]["ComplexConjugateA"]
            ccB = kernel["ProblemType"]["ComplexConjugateB"]
            ccVgprs = [None]*3 # three terms that can be negated: [real1, imag0, imag1]
            ccInsts = [None]*3
            accImOffset = self.AccVgprImagNumOffset(kernel)
            accStartSrcImg = accStart+accImOffset
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
            v_add = "v_add_" + miOutTypeName
            # negate code for Ai
            # necessary when ccA == ccB or ccA
            # the location to add negation is different between ccA == ccB and ccA
            #   index = 0 for ccA == ccB
            #   index = 1 for ccA
            # if ccA == ccB and ccA, we need negation twice for index 0 and 1 (but we need to calculate -Ai only once)
            if ccA == ccB:
              arrayIndex = 0
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate r1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again)
              if (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = inst(v_add, vgpr(ccVgprs[arrayIndex], numRegistersOut), "-"+ai, "0", "Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
              # if ccA, copy ccVgprs[0] to ccVgpr[1] to avoid generating same negate value twice
              if ccA:
                ccVgprs[1] = ccVgprs[0]
            elif ccA:
              arrayIndex = 1
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i0")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again)
              if (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = inst(v_add, vgpr(ccVgprs[arrayIndex], numRegistersOut), "-"+ai, "0", "Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
            # negate code for Ar
            if ccB:
              arrayIndex = 2
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again)
              if (ar not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = inst(v_add, vgpr(ccVgprs[arrayIndex], numRegistersOut), "-"+ar, "0", "Ar=-Ar")
                zgemmVaddSrcCheck[arrayIndex].append(ar)
            imod.addInst(shiftKStr + \
                         "".join([inst for inst in ccInsts if inst is not None]) + \
                         mfmaComponent(self, accStart, accEnd, ar, br, accStart, accEnd, 0, firstIterIui0), "Cr += Ar*Br")
            src0 = vgpr(ccVgprs[1], numRegistersOut) if ccVgprs[1] else ai
            imod.addInst(mfmaComponent(self, accStart+accImOffset, accEnd+accImOffset, src0, br, accStartSrcImg, accEndSrcImg, 0, firstIterIui0), "Ci += %sAi*Br"%("-" if ccVgprs[1] else ""))
            src0 = vgpr(ccVgprs[0], numRegistersOut) if ccVgprs[0] else ai
            imod.addInst(mfmaComponent(self, accStart, accEnd, src0, bi, accStart, accEnd, accStoreCIdx, False), "Cr += %sAi*Bi"%("-" if ccVgprs[0] else ""))
            src0 = vgpr(ccVgprs[2], numRegistersOut) if ccVgprs[2] else ar
            imod.addInst(mfmaComponent(self, accStart+accImOffset, accEnd+accImOffset, src0, bi, accStartSrcImg, accEndSrcImg, accStoreCIdx, False), "Ci += %sAr*Bi"%("-" if ccVgprs[2] else ""))

            checked = []
            for v in ccVgprs:
              if (v is not None) and (v not in checked):
                self.vgprPool.checkIn(v)
                checked.append(v)
          else:
            waits = mfmaComponent.WaitCount(self)
            # if number of mfma in one unroll loop (without innerUnroll) is 1, accIdx is same as the previous iteration
            numMfma = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1]
            waitCode = ""
            if waits > 0 and (prevAccIdx == accIdx or numMfma == 1):
              waitCode += inst("s_nop %u"%(waits - 1), "Wait for C")
            imod.addCode(shiftKStr + waitCode + mfmaComponent(self, accStart, accEnd, Str0, Str1, accStart, accEnd, accStoreCIdx, firstIterIui0))
            prevAccIdx = accIdx

          # clear shiftKStr
          shiftKStr = ""

    mfmaMod = Code.Module("mfmaCode")
    mfmaMod.addCode(imod)

    return mfmaMod

  def removeExtraUnroll(self, kernel):
    kStr = ""

    tmpSgpr = self.getTmpSgpr(1).idx()
    loopCounterName = "TailLoopCounter" if self.prefetchAcrossPersistent0 else self.loopCounterName(kernel, self.unrollIdx)
    elementPerReg = self.bpr//self.bpeAB
    skipLabel = 'SkipCleanDirtyUnroll'

    kStr += inst("s_cmp_ge_u32", sgpr(loopCounterName), elementPerReg, "check any dirty unroll")
    kStr += inst("s_cbranch_scc1", skipLabel, "skip clean when no dirty unroll")
    kStr += self.endLine

    kStr += inst("s_and_b32", sgpr(tmpSgpr), sgpr(loopCounterName), elementPerReg-1, "how much element in vgpr")
    kStr += inst("s_sub_i32", sgpr(tmpSgpr), elementPerReg, sgpr(tmpSgpr), "how much dirty element need to be remove")
    kStr += inst("s_lshl_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), hex(log2(self.bpeAB*8)), "how much dirty bits need to be remove")
    kStr += self.endLine

    for blockA in range(0, kernel["ThreadTile0"]//elementPerReg):
      for iui in range(0, kernel["InnerUnroll"]):
        aStr = f'ValuA_X0_I{iui}+{blockA}'
        kStr += inst("v_lshlrev_b32", vgpr(aStr), sgpr(tmpSgpr), vgpr(aStr), "clean dirty unroll")
    kStr += self.endLine

    for blockB in range(0, kernel["ThreadTile1"]//elementPerReg):
      for iui in range(0, kernel["InnerUnroll"]):
        bStr = f'ValuB_X0_I{iui}+{blockB}'
        kStr += inst("v_lshlrev_b32", vgpr(bStr), sgpr(tmpSgpr), vgpr(bStr), "clean dirty unroll")
    kStr += self.endLine

    kStr += inst(skipLabel+':', "end of clean dirty unroll")
    kStr += self.endLine

    return kStr

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, bufferIdx, iuiCount, useMacro, isTail=False):
    imod = Code.Module("macIter_X%u_I%u"%(bufferIdx, iuiCount))

    if not self.do["MAC"]: return imod

    if isTail and (kernel["LocalDotLayout"] > 1) and (kernel["InnerUnroll"] == kernel["LocalDotLayout"]) \
        and ((kernel["AssertSummationElementMultiple"] % kernel["LocalDotLayout"]) != 0):
      imod.addText(self.removeExtraUnroll(kernel))

    if kernel["ProblemType"]["DataType"].isHalf():
      imod.addInst(".align32 8, 0xbf800001", "align v_pk_fma")   # Align v_pk_fma instructions used in MAC_ blocks

    if kernel["InnerUnroll"] > 1 and iuiCount==1:
      # This it tail-loop case where we just want one IUI,
      imod.addText("MAC_%ux%u_X%u_OneIUI" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx))
    else:
      if useMacro:
        imod.addText("MAC_%ux%u_X%u" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx))
      else:
        # Generate MAC calls inline
        imod.addText(self.defineMACs(kernel, bufferIdx, kernel["InnerUnroll"]))

    return imod

  ##############################################################################
  # MAC Iteration -alternate version
  ##############################################################################
  def macCode(self, kernel, bufferIdx, iuiCount):
    if not self.do["MAC"]: return ""
    imod = Code.Module("macIter_X%u_I%u"%(bufferIdx, iuiCount))

    if kernel["ProblemType"]["DataType"].isHalf():
      imod.addInst(".align32 8, 0xbf800001", "align v_pk_fma")   # Align v_pk_fma instructions used in MAC_ blocks

    doOnce = False
    beAggressive = kernel["AggressivePerfMode"]
    macIdx = 0

    # half precision
    if kernel["ProblemType"]["DataType"].isHalf():
      for blockB in range(0, kernel["ThreadTile1"]//2):
        for blockA in range(0, kernel["ThreadTile0"]//2):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # bf16 precision
    elif kernel["ProblemType"]["DataType"].isBFloat16():
      for blockB in range(0, kernel["ThreadTile1"]//2):
        for blockA in range(0, kernel["ThreadTile0"]//2):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # integer i8x4
    elif kernel["ProblemType"]["DataType"].isInt8x4():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # single precision
    elif kernel["ProblemType"]["DataType"].isSingle():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True
          if macIdx == kernel["PerformanceWaitLocation"]:
            imod.addCode(Code.WaitCnt(self.version, kernel["PerformanceWaitCount"],"extra wait for performance"))
          if macIdx == kernel["PerformanceSyncLocation"]:
            imod.addInst("s_barrier ","extra barrier for performance")
          macIdx += 1

    # double precision
    elif kernel["ProblemType"]["DataType"].isDouble():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # single precision complex
    elif kernel["ProblemType"]["DataType"].isSingleComplex():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # double precision complex
    elif kernel["ProblemType"]["DataType"].isDoubleComplex():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    else:
      printExit("Assembly doesn't support %s" % kernel["ProblemType"]["DataType"])

    if beAggressive and doOnce:
      imod.addInst("s_setprio ","0","Reset priority after macs")

    return imod

  ##############################################################################
  # At Least 1 Unroll
  # prefetch means this is in the prefetch code, either before unroll loop
  # or in the PAP code.
  # isPap means this is the PAP iteration, need to adjust the loop exit
  # isOptNLL : this is for the store-interleaved NLL optimization
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isPap):
    kStr = ""
    if prefetch:
      if not isOptNLL:
        kStr += self.checkLastIter(kernel)
        if not isPap:
          if kernel["StorePriorityOpt"]:
            kStr += inst("s_setprio 0", "optimization store")
          if self.doShadowInit:
            kStr += inst("s_cbranch_scc1 %s"\
                % self.getNamedLabel("ShadowInitStart"), \
                "skip to ShadowInitStart iter b/c numIter==0")
          else:
            loopChar = self.indexChars[ \
                kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
            labelName = self.getNamedLabel("LoopEnd%s"%loopChar)
            kStr += inst("s_cbranch_scc1 %s" % labelName,
                "skip to unrollLoop end loop%s iter b/c numIter==0" % loopChar)
        else:
          labelName =  "SkipPrefetchAcrossPersistent"
          kStr += inst("s_cbranch_scc1 %s"\
              % self.getNamedLabel(labelName), \
              "skip prefetch loads since numIter==0")
    elif isOptNLL:

      # When OptNLL + PAP enabled, but is the last tile so isPap=False (brief: T,T,F),
      # We don't need to append the code here (checking Alpha,Beta,Tail) since it is shared with (T,T,T)
      # Somehow we still need to do the register-pool backup...
      if self.prefetchAcrossPersistent and not isPap:
        self.savedVgprPool = deepcopy(self.vgprPool)
        self.savedSgprPool = deepcopy(self.sgprPool)
        return ""

      skipOptNLL = self.getNamedLabel("OptNLL_End")
      tmpSgpr = self.getTmpSgpr(4).idx()

      # skip beta check for StoreCInUnroll in OptNLL case
      if not kernel["StoreCInUnroll"]:
        kStr += self.checkIsBetaZero(kernel, skipOptNLL, tmpSgpr)

      # check alpha
      # skip alpha check for StoreCInUnroll in OptNLL case
      if self.do["ApplyAlpha"] and not kernel["StoreCInUnroll"]:
        # (The new hgemm (h,h,h,h,s,s) is included in ComputeType=Single)
        if kernel["ProblemType"]["ComputeDataType"].isHalf():

          if kernel["ProblemType"]["HighPrecisionAccumulate"] and \
             (kernel["PersistentKernel"] or kernel["StreamK"]):
            kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), "1.0", "Alpha == 1.0 ?")
          # Otherwise, Alpha is a packed F16 so far (if Non-PK, the cvt is done later in GW)
          else:
            # for (h,h,h,h,h,h) no HPA,
            kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0x3c003c00", "Packed alpha==1.0")
            kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), sgpr(tmpSgpr), "alpha == 1.0?")

        # Shouldn't go here. Currently, DataType=B->ComputeDataType=S
        # (bf-gemm is included in ComputeType=Single)
        elif kernel["ProblemType"]["ComputeDataType"].isBFloat16():
          kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0x3f803f80", "Packed alpha==1.0")
          kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), sgpr(tmpSgpr), "alpha == 1.0?")

        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), "1", "Alpha == 1.0 ?")

        # This covers sgemm, bfgemm + HPA (b,b,b,b,s,s), and also hgemm (h,h,h,h,s,s)
        elif kernel["ProblemType"]["ComputeDataType"].isSingle():
          #kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
          kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), "1.0", "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["ComputeDataType"].isDouble():
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), 0x00000000, "Low part of double 1.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0x3ff00000", "High part of double 1.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), "1.0", "Real part of 1.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0.0", "Imaginary part of 1.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), "0x00000000", "lsb of real part of 1.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0x3ff00000", "msb of real part of 1.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha.real == 1.0 ?")
          kStr += inst("s_cbranch_scc0 %s"%skipOptNLL, "branch if alpha.real != 1")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), "0x00000000", "lsb of imag part of 0.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0x00000000", "msb of imag part of 0.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha+2",2), sgpr(tmpSgpr,2), "Alpha.imag == 0.0 ?")

        kStr += inst("s_cbranch_scc0 %s"%skipOptNLL, "branch if alpha != 1")
        kStr += "\n"

      kStr += self.checkIsEdge(kernel, skipOptNLL, tmpSgpr)
      kStr += "\n"

      # Check tail loop required:
      # Skip tail loop check if noTailLoop is true
      if not self.noTailLoop:
        loopChar = self.indexChars[ \
            kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
        kStr += scalarStaticDivideAndRemainder(tmpSgpr, tmpSgpr+1, "SizesSum+%u"%self.unrollIdx, \
                  kernel["DepthU"], tmpSgpr+2, 2)
        kStr += inst("s_cmp_eq_u32", sgpr(tmpSgpr+1), \
            hex(0), "numIter%s == 0"%loopChar )
        kStr += inst("s_cbranch_scc0 %s"%skipOptNLL, \
            "skip if tail loop required")

      # save the vgprPool for generating the normal path.
      # dump the 'dirty' pool upon s_endpgm and swap back the 'clean' pool
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
          added.append(self.vgprPool.addRange(self.startVgprG2LA, \
              self.startVgprG2LA+self.numVgprG2LA-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.startVgprLocalWriteAddressesA, \
                       self.startVgprLocalWriteAddressesA, "startOptNLL"))
        if not kernel["DirectToLdsB"]:
          added.append(self.vgprPool.addRange(self.startVgprG2LB, \
              self.startVgprG2LB+self.numVgprG2LB-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.startVgprLocalWriteAddressesB, \
                       self.startVgprLocalWriteAddressesB, "startOptNLL"))

      if kernel["BufferLoad"]:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadOffsetA, \
            self.startVgprGlobalReadOffsetB, "startOptNLL"))
      else:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadAddressesA, \
            self.startVgprGlobalReadAddressesB, "startOptNLL"))
      kStr += self.comment("reclaim VGPRS: " + ", ".join(added))
      """

    return kStr

  ##############################################################################
  def getLocalReadSwapOffset(self, kernel, labelIndex):
    kStr = ""
    # if AssertSummationElementMultiple/GlobalSplitU is multiple of DepthU * 2, LoopCounter will not be Odd
    # then, we do not need odd check and only need code for even case
    asem = kernel["AssertSummationElementMultiple"]
    gsu = kernel["GlobalSplitU"]
    noOddFlag = (asem % gsu == 0) and ((asem//gsu) % (kernel["DepthU"] * 2) == 0)
    label = "SkipLroSwap%s"%str(labelIndex)
    if self.prefetchAcrossPersistent0 and kernel["ExpandPointerSwap"] and not noOddFlag:
      # We can use OrigLoopCounter & 1 instead of using BreakAtEvenIter==1
      #kStr += inst("s_cmp_eq_u32", sgpr("BreakAtEvenIter"), 1, "test if BreakAtEvenIter==1 ?")
      #kStr += inst("s_cbranch_scc1", self.getLabelTarget(label), "Skip LROSwap if BreakAtEvenIter==1")
      tmpSgpr = self.getTmpSgpr(1).idx()
      # forceBreakAtEvenCheck case (= PrefetchGlobalRead=2 and NGLL), swap is necessary if LoopCounter is even.
      # for other cases, swap is necessary if LoopCounter is odd
      scc0or1 = 1 #0 if forceBreakAtEvenCheck else 1
      oddOrEven = "Odd" #"Even" if forceBreakAtEvenCheck else "Odd"
      kStr += inst("s_and_b32",sgpr(tmpSgpr), sgpr("OrigLoopCounter"), 1, "test if LoopCounter is Odd ?")
      kStr += inst("s_cbranch_scc%u"%(scc0or1), self.getLabelTarget(label), "Skip LROSwap if LoopCounter is %s"%(oddOrEven))

    kStr += self.comment("(PAP) Select low bank of LDS, if high bank is selected before (loop odditer exit)" if kernel["ExpandPointerSwap"] \
      else "(PAP) local read swap offsets a, b")
    if not kernel["DirectToVgprA"]: # do not generate local read code if DirectToVgpr is enabled
      kStr += self.localReadSwapOffsets(kernel, False, self.tPA)
    if not kernel["DirectToVgprB"]: # do not generate local read code if DirectToVgpr is enabled
      kStr += self.localReadSwapOffsets(kernel, False, self.tPB)

    if kernel["ExpandPointerSwap"] and not noOddFlag:
      kStr += self.getLabelDef(label, "Skip LRO Swap\n")
    return kStr

  ##############################################################################
  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isPap, isNGLL):
    kStr = ""
    if not prefetch:
      if isNGLL:
        toPGR1 = self.getLabelNum("toPGR1")
        kStr += "label_%04u:%s" % (toPGR1, self.endLine)
      else:
        if isOptNLL:
            endSumLabel = self.getNamedLabel("Summation_End_OptNLL")

            kStr += self.comment1("Stores for OptNLL")
            kStr += self.endSummation(kernel, endSumLabel, isOptNLL)

            # perhaps could work with LSU>1 by adding other indices here, but not tested
            assert (kernel["LocalSplitU"] == 1)
            kStr += self.notLocalSplitUGlobalWriteIndices(kernel)

            # add stores for opt NLL
            (fullVw, elements) = self.notLocalFullTileElements(kernel, False)
            alpha = False
            beta = False
            if kernel["StoreCInUnroll"]:
              # in StoreCInUnroll case
              # enable alpha
              alpha = True
              # enable beta if necessary
              if not kernel["AtomicAddC"] and kernel["ProblemType"]["UseBeta"]:
                beta = True
            kStr += self.globalWriteElements(kernel, [fullVw], [elements], applyAlpha=alpha, betas=[beta], edges=[False], isOptNLL=True)

            self.cleanupGlobalWrite(kernel)
            kStr += "\n"
            kStr += str(self.functionEnd(kernel, False))
            #kStr += inst("s_branch %s"%summationEnd, "skip the OptNLL")

            label = self.getNamedLabel("OptNLL_End")
            kStr += "%s:%s" % (label, self.endLine)

        else:
          # local read swap offset code
          #kStr += self.getLocalReadSwapOffset(kernel,2)

          label = self.getNamedLabel("PrefetchGlobalLastIterEnd")
          kStr += "%s:%s" % (label, self.endLine)

    # swap back vgpr pool if any
    if self.savedVgprPool != None:
      # in case pool size in current path is larger than pool size in main path
      # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
      oldSize = self.savedVgprPool.size()
      newSize = self.vgprPool.size()
      if newSize > self.savedVgprPool.size():
        for i in range(oldSize,newSize):
          self.savedVgprPool.pool.append(self.savedVgprPool.Register(RegisterPool.Status.Available,"restore vgprPool"))
      self.vgprPool = self.savedVgprPool # restore vgprPool before alternate path
      self.savedVgprPool = None
    # swap back sgpr pool if any
    if self.savedSgprPool != None:
      # in case pool size in current path is larger than pool size in main path
      # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
      oldSize = self.savedSgprPool.size()
      newSize = self.sgprPool.size()
      if newSize > self.savedSgprPool.size():
        # fixed range to prevent overflowing resources in some cases
        for i in range(oldSize,newSize):
          self.savedSgprPool.pool.append(self.savedSgprPool.Register(RegisterPool.Status.Available,"restore sgprPool"))
      self.sgprPool = self.savedSgprPool # restore vgprPool before alternate path
      self.savedSgprPool = None
    return kStr

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def incrementSrd(self, kernel, tP, incLower, incUpper, checkShadowLimitCopy=True):
    imod = Code.Module("incrementSrd")
    tc = tP["tensorChar"]

    imod.addInst("s_add_u32", \
         sgpr("Srd%s+0"%(tc)), \
         sgpr("Srd%s+0"%(tc)), \
         incLower, \
        "gra SRD += inc(lower)" )
    imod.addInst("s_addc_u32 ", \
         sgpr("Srd%s+1"%(tc)), \
         sgpr("Srd%s+1"%(tc)), \
         incUpper, \
         "gra SRD += inc(upper)" )

    # also have to move the boundary since we change the base
    # so less buffers to the edge:
    if self.use64bShadowLimit:
      imod.addInst("s_sub_u32", \
          sgpr("ShadowLimit%s+0"%tc), \
          sgpr("ShadowLimit%s+0"%tc), \
          incLower, \
            "limit -= inc)")
      imod.addInst("s_subb_u32", \
          sgpr("ShadowLimit%s+1"%tc), \
          sgpr("ShadowLimit%s+1"%tc), \
          incUpper, \
            "limit -= inc)" )
      if checkShadowLimitCopy:
        imod.addInst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
        if self.staggerU:
          # staggerU case, need to restore BufferLimit when ShadowLimit goes to negative value
          imod.addInst("s_cselect_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "BufferLimit%s"%tc, "Move shadow to real if we are within 2^32")
        else:
          imod.addInst("s_cmov_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "Move shadow to real if we are within 2^32")
    else:
      imod.addInst("s_sub_u32", \
           sgpr("Srd%s+2"%(tc)), \
           sgpr("Srd%s+2"%(tc)), \
           incLower, \
            "limit -= inc)" )
    return imod

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def setTailSrd(self, kernel, tP, incLower):
    # In SuppressNoLoadLoop, the final loop iteration moves the SRD base forward
    # and the ShadowLimit backwards by one extra 'click' of GlobalReadIncs[AB].
    # Note the ShadowLimit may become negative - for example edge tiles where the
    # increment is > tile width.
    # The SuppressNoLoadLoop mode also forces the SRD limit to 0 on the final iteration.
    # The code here undoes the final click step by moving the base backwards and the
    # limit forwards (reading from the ShadowLimit).
    # It only works if use64bShadowLimit is enabled (since this enables use of the ShadowLimit)

    tc = tP["tensorChar"]
    kStr = ""
    incUpper = 0

    kStr += inst("s_sub_u32 ", \
         sgpr("Srd%s+0"%(tc)), \
         sgpr("Srd%s+0"%(tc)), \
         incLower, \
        "gra SRD -= inc(lower)" )
    kStr += inst("s_subb_u32 ", \
         sgpr("Srd%s+1"%(tc)), \
         sgpr("Srd%s+1"%(tc)), \
         incUpper, \
        "gra SRD -= inc(upper)" )

    # using Shadow limit here which only works with 64-bit PBC:
    assert(self.use64bShadowLimit)

    kStr += inst("s_add_u32", \
        sgpr("ShadowLimit%s+0"%tc), \
        sgpr("ShadowLimit%s+0"%tc), \
         incLower, \
          "limit -= inc)")
    kStr += inst("s_addc_u32", \
        sgpr("ShadowLimit%s+1"%tc), \
        sgpr("ShadowLimit%s+1"%tc), \
         incUpper, \
          "limit -= inc)" )
    kStr += inst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
    kStr += inst("s_cmov_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "Move shadow to real if we are within 2^32")

    return kStr

  ##############################################################################
  # Global Read: Increment A/B
  # loopIdx is summation idx:
  #   self.unrollIdx, or an idx from 0..NumIndicesSummation
  # prefetchIndex is >0 (1...PrefetchGlobalRead) if this increment follows a
  #   global prefetch or 0 otherwise
  # incs is number of increments to perform
  ##############################################################################
  def globalReadIncrement(self, kernel, imod, loopIdx, tP, prefetchIndex, incs=1):
    if not self.do["GlobalInc"]: return ""
    tc = tP["tensorChar"]
    loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]

    imod.addComment1("global read inc %s loop%s"%(tc,loopChar))

    if kernel["BufferLoad"]:
      # TODO - does this handle N-dim tensors correctly?
      #if tP["isB"]:
      #  kStr += inst("s_mov_b32", sgpr("OffsetB"), sgpr("SrdB+0"), "hack to save")
      if self.staggerU and loopIdx == self.unrollIdx:
        # add a wrap increment, if needed:
        incLower = self.getTmpSgpr(3).idx()
        incUpper = incLower + 1
        tmpS     = incLower + 2
        if prefetchIndex:
          imod.addInst("s_add_u32", sgpr(tmpS), self.loopCounter(kernel, self.unrollIdx), prefetchIndex, "remove pf(%u)"%prefetchIndex)
          imod.addInst("s_cmp_eq_u32",  sgpr("StaggerUIter"), sgpr(tmpS), "Is this wrapIter? (pf)")
        else:
          imod.addInst("s_cmp_eq_u32",  self.loopCounter(kernel, self.unrollIdx), \
                    sgpr("StaggerUIter"), "Is this the wrapIter?")
        #kStr += self.assert_scc_is_1() # break at the wrap iteration
        imod.addInst("s_cselect_b32", sgpr(incLower), sgpr("WrapU%s+0"%tc), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                    "incLower <- ?")
        imod.addInst("s_cselect_b32", sgpr(incUpper), sgpr("WrapU%s+1"%tc), 0,
                    "incUpper <- ?")
        imod.addCode(self.incrementSrd(kernel, tP, sgpr(incLower), sgpr(incUpper), checkShadowLimitCopy=True))
      else:
        if (tc in ('A', 'B') and kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"%tc]):
          if kernel["_UseSgprForGRO"]:
            offsetVgpr = "GlobalReadOffset%s+0"%(tc)
            imod.addInst("_v_add_i32", vgpr(offsetVgpr), vgpr(offsetVgpr), sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), "update offset for mirror dims")
          else:
            for perp in range(0, tP["nrp"]):
              for sPerp in range(0, tP["nrpv"]):
                for para in range(0, tP["nrc"]):
                  for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
                    i = sPara + int(tP["nrcv"] / tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
                    graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
                    offsetVgpr = "GlobalReadOffset%s+%u"%(tc, graIdx)
                    imod.addInst("_v_add_i32", vgpr(offsetVgpr), vgpr(offsetVgpr), sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), "update offset for mirror dims")
        else:
          if loopIdx != self.unrollIdx:
            incUpper = sgpr(self.getTmpSgpr(1).idx())
            # GRO may be negative for other summation if stride-other < stride-unroll or if mirror dim.
            imod.addInst("s_ashr_i32", incUpper, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), 31, "sign-extend")
          else:
            incUpper = 0 # GRO is positive for loop unroll
          imod.addCode( self.incrementSrd(kernel, tP, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), incUpper))
    else:
      graIdx = 0
      #for perp in range(0, tP["nrp"]):
      #  for para in range(0, tP["nrc"]):
      #    for s in range(0, tP["nrcv"]):
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
              if self.globalReadIncsUseVgpr:
                imod.addInst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], 2*loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                imod.addInst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], 2*loopIdx)), \
                    self.vcc, \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              else:
                imod.addInst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    sgpr("GlobalReadIncs%s+%u"%(tP["tensorChar"], loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                imod.addInst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    0,
                    self.vcc, \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              graIdx += self.rpga
      #kStr += dump(vgpr("GlobalReadAddrA+0"))
      #kStr += dump(vgpr("GlobalReadAddrA+1"))
      #kStr += "s_endpgm\n"

  def globalReadIncrementAB(self, kernel, loopIdx, prefetchIndex, incs=1):
    imod = Code.Module("globalReadIncrementAB%s")
    problemType = self.kernel["ProblemType"]
    unrollLoopCounter = self.loopCounter(kernel, self.unrollIdx)

    incCodeA = imod.addCode(Code.Module("globalReadIncrementA"))
    incCodeB = imod.addCode(Code.Module("globalReadIncrementB"))

    if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
      loopCounter = self.loopCounter(kernel, self.unrollIdx)
      incCodeA.addInst("s_add_u32",
                   loopCounter, loopCounter,
                   "DepthU",  "increment psdIter")

    if loopIdx==self.unrollIdx and kernel["PackSummationDims"] and self.actualSummationLoops==1:
      incSize = 2 if self.use64bPackSumOffset else 1
      tmpSgpr = self.getTmpSgpr(3 + 2*incSize + (3 if kernel["GlobalSplitU"]>1 else 0)).idx()
      inc ={}
      inc['A'] = tmpSgpr + 3
      inc['B'] = inc['A'] + incSize
      gsuMagic = inc['B'] + incSize

      psdPackedBits = "DepthU" if prefetchIndex>0 else unrollLoopCounter
      incCodeA.addComment1("extract indices here from %s"%psdPackedBits)
      for oSum in reversed(range(problemType["NumIndicesSummation"])):
        sumDim  = problemType["IndicesSummation"][oSum]
        sumChar = self.indexChars[sumDim]
        firstIter = (oSum==problemType["NumIndicesSummation"]-1)
        lastIter  = (oSum==0)

        incCodeA.addComment1("extract index %s"%sumChar)

        if not lastIter:
          if oSum==self.unrollIdx and kernel["GlobalSplitU"] > 1:
            # GSU divides the first loop counter size by some amount
            size = "GsuNumIter%s"%sumChar
          else:
            size = "Size%s"%sumChar

          if firstIter:
            psdPackedBits2 = psdPackedBits
          else:
            psdPackedBits2 = sgpr(tmpSgpr+2)
            incCodeA.addInst("s_mov_b32", psdPackedBits2, psdPackedBits, "copy psdPackedBits")

          if oSum==self.unrollIdx and kernel["GlobalSplitU"] > 1:
            # compare GSUA
            # cmov into temps for Size,Abit,Shift
            # divide and go.
            # need more temps for this, need divide routine to take 3 parms
            incCodeA.addInst("s_cmp_lt_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
                "gsuSumIdx < numIterPerWgRemainder" )
            incCodeA.addInst("s_cselect_b32", sgpr(gsuMagic+0), sgpr("MagicNumberSize%s_GsuRemainder"%sumChar),
                              sgpr("MagicNumberSize%s"%sumChar), "Use alternate divisor")
            incCodeA.addInst("s_cselect_b32", sgpr(gsuMagic+1), sgpr("MagicAbitSize%s_GsuRemainder"%sumChar),
                              sgpr("MagicAbitSize%s"%sumChar), "Use alternate divisor")
            incCodeA.addInst("s_cselect_b32", sgpr(gsuMagic+2), sgpr("MagicShiftSize%s_GsuRemainder"%sumChar),
                              sgpr("MagicShiftSize%s"%sumChar), "Use alternate divisor")
            incCodeA.addText(self.scalarMagicDivExplicit(tmpSgpr, psdPackedBits,
                              magicNumber=gsuMagic+0, magicAbit=gsuMagic+1, magicShift=gsuMagic+2))
          else:
            incCodeA.addText(self.scalarMagicDiv(tmpSgpr, psdPackedBits, sumChar))

          # TODO-64
          incCodeA.addInst("s_mul_i32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+0), sgpr(size), "remainder step 1")
          incCodeA.addInst("s_sub_u32", sgpr(tmpSgpr+1), psdPackedBits2, sgpr(tmpSgpr+1), "remainder step 2")
          iterX=sgpr(tmpSgpr+1)
        elif firstIter and lastIter:
          # just one iter, use loop counter directly not remainder
          iterX = psdPackedBits
        else:
          iterX=sgpr(tmpSgpr+0)


        for tc in ('A','B'):
          zp = next((zpi for zpi in problemType["ZeroPad"+tc] if zpi[1] == sumDim), None)
          if zp:
            incCodeA.addInst("s_mov_b32", sgpr("Iter"+sumChar), iterX, "save iterX")

        # update psdOffset. Inputs:
        #   - tmpSgpr+0== packedBits, and must be preserved for next iteration
        #   - iterX, number of iterations for this dim.  Used in A/B increment loop below
        for tc in ('A','B'):
          assert(not self.use64bPackSumOffset)
          if firstIter:
            #incCodeA.addText(self.s_mul_u64_u32(inc{'A'}+0, inc{'A'}+1, tmpSgpr+1, sgpr["GlobalReadIncs%s+%d"]))
            incCodeA.addInst("s_mul_i32", sgpr(inc[tc]), iterX, sgpr("GlobalReadIncs%s+%d"%(tc,oSum)),
                              "psdOffset%s += scale iter%s"%(tc,sumChar))
          else:
            incCodeA.addInst("s_mul_i32", sgpr(tmpSgpr+2), iterX, sgpr("GlobalReadIncs%s+%d"%(tc,oSum)), "Scale iter%s"%sumChar)
            incCodeA.addInst("s_add_u32", sgpr(inc[tc]+0), sgpr(inc[tc]+0), sgpr(tmpSgpr+2), "psdOffset%s += scale iter%s"%(tc,sumChar))
            #incCodeA.addText(self.s_mul_u64_u32(tmp+0, inc{'A'}+1, tmpSgpr+1, sgpr["GlobalReadIncsA"]))

          psdPackedBits = sgpr(tmpSgpr+0)

        if 0 and lastIter:
          incCodeA.addText(self.assert_ne(sgpr("LoopCounterM"), 8))

      assert(kernel["BufferLoad"])

      incCodeA.addText("\n")
      incCodeA.addComment1("Reset and increment SRDs")
      for tc in ('A','B'):
        incCodeA.addInst("s_mov_b32", sgpr("Srd%s+0"%tc), sgpr("InitialSrd%sBase+0"%tc), "restore base")
        incCodeA.addInst("s_mov_b32", sgpr("Srd%s+1"%tc), sgpr("InitialSrd%sBase+1"%tc), "restore base")
        if self.use64bShadowLimit:
          incCodeA.addInst("s_mov_b32", sgpr("ShadowLimit%s+0"%tc), sgpr("InitialSrd%sLimit+0"%tc), "restore shadow limit")
          incCodeA.addInst("s_mov_b32", sgpr("ShadowLimit%s+1"%tc), sgpr("InitialSrd%sLimit+1"%tc), "restore shadow limit")
          assert(0) # not tested, would maybe need to restore base too if limit 0
        else:
          incCodeA.addInst("s_mov_b32", sgpr("Srd%s+2"%tc), sgpr("InitialSrd%sLimit"%tc), "restore limit")


      # TODO - this skips over the stagger-u wrap codes
      def incrementSrdPsd(tc, tp):
        incCodeA.addText("\n")
        incUpperA = sgpr(inc[tc]+1) if self.use64bPackSumOffset else 0
        if bool(set(kernel["ProblemType"]["IndicesSummation"]).intersection(set(kernel["ProblemType"]["MirrorDims%s"%tc]))) and not self.use64bPackSumOffset:
          incUpperA = sgpr(self.getTmpSgpr(1).idx())
          incCodeA.addInst("s_ashr_i32", incUpperA, sgpr(inc[tc]), 31, "sign-extend")
        incCodeA.addCode(self.incrementSrd(kernel, tp, sgpr(inc[tc]), incUpperA))

      incrementSrdPsd('A', self.tPA)
      incrementSrdPsd('B', self.tPB)
    else:
      self.globalReadIncrement(kernel, incCodeA, loopIdx, self.tPA, prefetchIndex, incs)
      self.globalReadIncrement(kernel, incCodeB, loopIdx, self.tPB, prefetchIndex, incs)
    return imod

  ##############################################################################
  # Global Read:
  # globalReadGuardK is called for loads in the tail loop
  # Must ensure each load is in bounds - either using buffer bounds
  # or exec-mask checks.
  ##############################################################################
  def globalReadGuardK(self, kernel, tP, vregSetIdx):
    kStr = ""
    tc = tP["tensorChar"]
    problemType = self.kernel["ProblemType"]
    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth

    # need numElementsPerLoad to calculate address increment for BufferLoad=0
    # try to use the widest global load based on AssertFree0, 1 or AssertSummationElementMultiple
    afem = kernel["AssertFree0ElementMultiple"] if tP["isA"] else kernel["AssertFree1ElementMultiple"]
    asem = kernel["AssertSummationElementMultiple"]
    # need to adjust asem for GSU
    gsu = kernel["GlobalSplitU"]
    asem = 1 if asem%gsu !=0 else asem//gsu
    elemAlignAssert = afem if tP["tlu"] else asem
    elemAlignMin = min(tP["glvw"], elemAlignAssert)
    # if glvw > 1 and glvw is not power of 2, use 1
    if (tP["glvw"] > 1 and tP["glvw"] & (tP["glvw"] - 1)) != 0:
      elemAlignMin = 1
    MIInputPerThread = kernel["MIInputPerThread"] if kernel["EnableMatrixInstruction"] else 1
    if not tP["tlu"]:
      # TLU=False + TailLoop case, cannot make numElementsPerLoad larger than MIInputPerThread
      elemAlignMin = min(elemAlignMin, MIInputPerThread)
    numElementsPerLoad = elemAlignMin
    numBytesPerLoad = int(numElementsPerLoad * tP["bpe"])

    ########################################
    # Calculate Max Addr
    ########################################

    if not kernel["BufferLoad"]:
      tmpSgpr = self.getTmpSgpr(2).idx()
      maxAddrSgpr = tmpSgpr

      kStr += self.comment1("global addressing - max read address = Tensor2dSize%s"%tc)
      dim = len(tP["ia"])-1 # dim
      sizeIdx = tP["ia"][dim]
      sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
      if sizeIdxIsSum:
        sizeIdx -= kernel["ProblemType"]["NumIndicesC"]
      kStr += self.s_mul_u64_u32(sgpr(maxAddrSgpr+0), sgpr(maxAddrSgpr+1),  \
                  sgpr("WorkGroup%u"%(sizeIdx)),  \
                  sgpr("Stride%s%s"%(tc, self.indexChars[tP['ia'][-1]])), \
                  "64b tensor%s size in elements"%tc)
      kStr += inst("s_add_u32", \
          sgpr(maxAddrSgpr+0), \
          sgpr("Tensor2dSize%c"%tc), \
          sgpr(maxAddrSgpr+0), \
          "add Tensor2dSize%c"%tc)
      kStr += inst("s_addc_u32", \
          sgpr(maxAddrSgpr+1), \
          sgpr("Tensor2dSize%c+1"%tc), \
          sgpr(maxAddrSgpr+1), \
          "add Tensor2dSize%c"%tc)
      kStr += inst("s_lshl_b64", \
        sgpr(maxAddrSgpr,2), \
        sgpr(maxAddrSgpr,2), \
        hex(log2(tP["bpe"])), "<- tensor%s size in bytes"%tc)

      kStr += inst("s_add_u32", \
          sgpr(maxAddrSgpr+0), \
          sgpr(self.sgprs["Address%s"%tc]), \
          sgpr(maxAddrSgpr+0), \
          "prepend address lower")
      kStr += inst("s_addc_u32", \
          sgpr(maxAddrSgpr+1), \
          sgpr(self.sgprs["Address%s"%tc] +1), \
          sgpr(maxAddrSgpr+1), \
          "prepend address upper")
      # sgpr->vgpr
      maxAddrVgpr = self.vgprPool.checkOutAligned(2, 2, "maxAddrVgpr")
      kStr += inst("v_mov_b32", vgpr(maxAddrVgpr+0), sgpr(maxAddrSgpr+0), "sgpr->vgpr")
      kStr += inst("v_mov_b32", vgpr(maxAddrVgpr+1), sgpr(maxAddrSgpr+1), "sgpr->vgpr")

      # full exec mask
      fullExec = tmpSgpr
      sgprCnt = self.laneSGPRCount
      waveSize = kernel["WavefrontSize"]
      activeMask = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
      kStr += inst("s_mov_b{}".format(waveSize), sgpr(fullExec,sgprCnt), activeMask, "to restore all threads active")
      bpeVgpr = self.vgprPool.checkOut(1, "bpeVgpr")
      kStr += inst("v_mov_b32", vgpr(bpeVgpr), hex(numBytesPerLoad), "bpe*numElementsPerLoad")

      # can remove this?
      zeroVgpr = self.vgprPool.checkOut(1,"zeroVgpr")
      kStr += inst("v_mov_b32", vgpr(zeroVgpr), hex(0), "zero")

    extraFields = ""
    if (tP["NonTemporal"] & 1) != 0:
      extraFields += " " + getGlcBitName(kernel["MemoryModifierFormat"])
    if (tP["NonTemporal"] & 2) != 0:
      extraFields += " " + getSlcBitName(kernel["MemoryModifierFormat"])
    if (tP["NonTemporal"] & 4) != 0:
      extraFields += " nt"
    dtlNoDestVgpr = False
    if kernel["DirectToLds%s"%tc]:
      extraFields += " lds"
      dtlNoDestVgpr = globalParameters["AsmCaps"][self.version]["HasDirectToLdsNoDest"]


    directToLdsLoads = 0
    prevLdsOffset    = 0
    # print("tc={}, nrp={}, nrpv={}, nrc={}, nrcv/nrcvpi={}, zeroPad={}, sgprforGRO={}".format(tc, tP["nrp"], tP["nrpv"], tP["nrc"], int(tP["nrcv"]/tP["nrcvpi"], problemType["ZeroPad%s"%tc], kernel["UseSgprForGRO"]))
    if problemType["ZeroPad%s"%tc]:
      addrV = self.vgprPool.checkOut(1,"addrV")

    instOffset = 0
    loopCnt = -1

    mirrorDim = kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in problemType["MirrorDims%s"%tc]
    mirrorOffsetTemp = None
    if mirrorDim:
      mirrorOffsetTemp = self.vgprPool.checkOut(1, "mirrorOffset")

    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
            i = sPara + int(tP["nrcv"] / tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth

            destVgprHi = None
            dataIsByte = False
            packInt8Code = None

            instOffsetInc = 0 # increment value for instOffset. Need to apply after r loop

            r = 0
            numLoadVectorComp = (loadWidth*self.bpr/tP["bpe"])
            if numLoadVectorComp >= 1:
              # convert to int
              numLoadVectorComp = int(numLoadVectorComp)

            int8TempVgpr = numLoadVectorComp - 1
            # for each component in vector
            while r < numLoadVectorComp:
              loopCnt += 1
              if numBytesPerLoad == 1:
                # Check out 3 regs once , for component 1,2,3 (r = 1,2,3)
                if r == 1:
                  packInt8Code = Code.Module()
                  destVgprHi = self.vgprPool.checkOut( int8TempVgpr , 'destVgprHi')
                dataIsByte = True
              elif numBytesPerLoad == 2:
                if self.archCaps["HasEccHalf"]:
                  # In some cards, loading half types into register will zero out
                  # the other half. Therefore we need to load into a separate register
                  # then pack 2 registers into one
                  destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi')
              elif numBytesPerLoad > 16:
                printWarning("DataType unsupported")
              regIdx = r * tP["bpe"] // 4
              kStr += self.comment1("g2l=%u, load component %u"%(g2lIdx, r))

              offset = 0

              hi8 = 0
              hi16 = 0
              bpl = numElementsPerLoad*self.bpeAB # bytesPerLoad

              if numElementsPerLoad > 1:
                comment = "load packed %uX buffer value"%numElementsPerLoad
              else:
                comment = "load one buffer value"
              if numBytesPerLoad==2:
                hi16=loopCnt%2

              if numBytesPerLoad==1:
                hi8  = (loopCnt%4) %2
                hi16 = (loopCnt%4)//2

              # destVgpr setting (common for BL0 and BL1)
              if kernel["DirectToLds%s"%tc]:
                destVgpr=0
              elif kernel["DirectToVgpr%s"%tc]:
                numVgprG2L = self.numVgprG2LA if tP["isA"] else self.numVgprG2LB
                numVgprPerBlock = numVgprG2L // 2 # numVgprG2L is doubled for DirectToVgpr
                idx = g2lIdx + vregSetIdx * numVgprPerBlock
                destVgpr="G2L%s+%u+%u"%(tc, idx, regIdx)
              else:
                destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx)
              # if hi8=1 or hi16=1 (component 1,2,3 for int8) or (component 1 for half), use the temp destVgprHi
              # but only when hi16=1 we use the _d16_hi version instruction, see the below visualized int8 comment
              loadVgpr = destVgprHi if ((hi16 or hi8) and destVgprHi != None) else destVgpr

              # (Int8(bpl=1) + Not HasEccHalf) or (bpl < 4 + HasEccHalf + not bufferLoad case), need to initialize dest vgpr for global load
              # bpl < 4 case, we need to use multiple loads + or operation(s)
              # In that case, we need to ensure that register value of out of range is 0
              # (except for the case we use same destination for multiple loads (Not HasEccHalf)
              # Not buffer load case, we use exec flag and register value is unchanged if exec flag is 0.
              # If we do not initialize dest register here, some uninitialized value can be used for or operation
              if (bpl == 1 and (not self.archCaps["HasEccHalf"])) or \
                 (bpl < 4 and self.archCaps["HasEccHalf"] and not kernel["BufferLoad"]):
                kStr += inst("v_mov_b32", vgpr(loadVgpr), 0, "set to zero to avoid unexpected value for byte per load < 4")

              if kernel["BufferLoad"]:
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
                elif kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx))
                # Sgpr for GRO
                else:
                  soffset = "0" if graIdx == 0 else sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

                if problemType["ZeroPad%s"%tc] and not (kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]):
                  codeMod = Code.Module("guardZeroPad%u"%loopCnt)
                  offsetVgpr = self.guardZeroPad(kernel, tP, codeMod, offsetVgpr, soffset, addrV, perp, sPerp, para, sPara)
                  kStr += str(codeMod)

                if mirrorDim:
                  codeMod = Code.Module("mirrorIdx%u"%loopCnt)
                  codeMod.addInst("v_mov_b32", vgpr(mirrorOffsetTemp), vgpr(offsetVgpr), "copy offset to clamp mirror dim")
                  offsetVgpr = mirrorOffsetTemp

                  # ScalarGlobalReadOffset should be negative value with unroll mirroring.
                  # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
                  if soffset != "0":
                    codeMod.addInst("_v_sub_u32", vgpr(offsetVgpr), vgpr(offsetVgpr), soffset, "mirror unroll: GRO=GRO-SGRO, soffset=0")
                    # soffset_prev = soffset
                    soffset = "0"

                  # check for pre-padding
                  prePad = 0
                  if self.groOffsetInMacroTile:
                    prePad = self.srdShiftLeft[tc] * self.bpeAB

                  codeMod.addInst("v_max_i32", vgpr(offsetVgpr), prePad, vgpr(offsetVgpr), "clamp to beginning of buffer")
                  kStr += str(codeMod)

                if kernel["DirectToLds%s"%tc]:
                  # need to increment ldsInc only once per each loopCnt
                  # this is pre count up, so increment it at r == 0
                  if r == 0:
                    # get ldsInc
                    ldsInc = self.getLdsIncForDTL(kernel, tP)
                  else:
                    ldsInc = 0
                  #print("ldsInc", ldsInc)
                  #print("GlobalLoadVectorWidth", kernel["GlobalLoadVectorWidth%c"%tc])
                  #print("bpr", self.bpr)
                  if kernel["UseInstOffsetForGRO"]:
                    # buffer_load only support 12 bit instruction offset
                    # we have to increase m0 if offset is larger thant 12 bits
                    if instOffset >= self.buff_load_inst_offset_max:
                      inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                      kStr += inst("s_add_u32", "m0", "m0", inc, "Move LDS write address to next base" )
                      instOffset -= inc
                  elif directToLdsLoads != 0 and ldsInc > 0:
                      if tP["nrc"] > 1:
                        # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                        if tP["grcg"]:
                          if tP["grcv"]:
                            divisorName = tP["lvc"]
                          else:
                            # Fractional load use the more accurate lsc, multiply by VW later
                            divisorName = tP["lsc"]
                        else:
                          if tP["grcv"]:
                            divisorName = tP["lsp"]
                          else:
                            divisorName = tP["lvp"]
                        divisor = kernel[divisorName]
                        # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                        wSize = max(self.kernel["WavefrontSize"], divisor)
                        lscaOffset = para * wSize * int(tP["bpe"] * tP["glvw"])
                        ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                        ldsInc = ldsOffset - prevLdsOffset
                        prevLdsOffset = ldsOffset
                      kStr += inst("s_add_u32", "m0", "m0", ldsInc, "Move LDS write address to next line" )

                offset = r * tP["bpe"] + instOffset

                kStr += self.chooseGlobalRead(True, \
                          bpl, destVgpr=loadVgpr, \
                          addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                          soffset=soffset, offset=offset, \
                          extraFields=extraFields, \
                          dtlNoDestVgpr=dtlNoDestVgpr, \
                          hi16=hi16, \
                          comment=comment).toStr()

                if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  instOffsetInc += ldsInc
                # print("  bpl={}, destVgpr={}, soffset={}, offset={}, hi16={}".format(bpl, destVgpr, soffset, offset, hi16))

              else: # Not buffer load, ie 'global' load
                # mask if current address if in bounds

                kStr += inst("_v_cmpx_lt_u64", self.vcc, \
                    vgpr("GlobalReadAddr%s+%u"%(tc, graIdx),2), \
                    vgpr(maxAddrVgpr,2), \
                    "addr < maxAddr")

                # load one element from address
                kStr += self.chooseGlobalRead(False, \
                          bpl, destVgpr=loadVgpr, \
                          addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                          soffset=0, offset=0, \
                          extraFields=extraFields, \
                          dtlNoDestVgpr=dtlNoDestVgpr, \
                          hi16=hi16, \
                          comment="load one global value").toStr()

                # restore full exec mask
                kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr(fullExec,self.laneSGPRCount), \
                    "all threads active")

                # increment address by 1 element (BPE)
                kStr += inst("_v_add_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tc, graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+0"%(tc, graIdx)),  \
                    vgpr(bpeVgpr), "gra += 1 (lower)")
                kStr += inst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tc, graIdx)), \
                    self.vcc, \
                    vgpr("GlobalReadAddr%s+%u+1"%(tc, graIdx)), \
                    vgpr(zeroVgpr), \
                    self.vcc, \
                    "gra += 1 (upper)")

              # int8 byte:
              # |--------|--------|--------|---V0---|, r = 0, hi8=0, hi16=0, load d16
              # |--------|--------|--------|---V1---|, r = 1, hi8=1, hi16=0, load d16
              # |--------|---V2---|--------|--------|, r = 2, hi8=0, hi16=1, load d16_hi
              # |--------|---V3---|--------|--------|, r = 3, hi8=1, hi16=1, load d16_hi
              # V1, V3 -> shift left 8 bits, or 4 regs (pack)
              # DestV0|=(V1 << 8), DestV0|= V2, DestV0|=(V3 << 8)
              # Int8 (byte)
              if dataIsByte and (destVgprHi != None):
                # hi8  -> r = 1,3
                # hi16 -> r = 2,3
                if hi8 or hi16:
                  # r = 1,2,3, vmcnt needed for one packing
                  packInt8Code.addText("s_waitcnt vmcnt(%u)\n"%(int8TempVgpr-r) )
                if hi8:
                  # r = 1,3,   shift needed
                  packInt8Code.addInst("v_lshlrev_b32", vgpr(destVgprHi), "0x8", vgpr(destVgprHi), "shift left to higher 8 bits")
                if hi8 or hi16:
                  # r = 1,2,3, packing
                  packInt8Code.addInst("v_or_b32", vgpr(destVgpr), vgpr(destVgpr), vgpr(destVgprHi), "pack a sub 8-bit with dest")
                destVgprHi += 1

              # Half
              elif destVgprHi != None and (r//numElementsPerLoad) % 2 == 1:
                kStr += "s_waitcnt vmcnt(0)\n"
                kStr += "v_or_b32 " + vgpr(destVgpr) + ", " + vgpr(destVgpr) + ", " + vgpr(destVgprHi) + " // HasEccHalf: pack\n"

              # For half (bf16). Note: for int8, we will checkin after loading all components
              if (destVgprHi != None) and (not dataIsByte):
                self.vgprPool.checkIn(destVgprHi)
                destVgprHi = None

              r += numElementsPerLoad # next component (for half, byte)

            # end R loop

            instOffset += instOffsetInc # add increment value for instOffset. Need to apply after r loop
            # increment once per r loop (at the end)
            directToLdsLoads+=1

            # for int8:
            # we do the 3 packs, and checking the 3 extra vgprs after loading all components
            if numBytesPerLoad==1:
              assert packInt8Code != None and destVgprHi != None
              kStr += str(packInt8Code)
              self.vgprPool.checkIn(destVgprHi - int8TempVgpr)
              destVgprHi = None

    if mirrorDim:
      self.vgprPool.checkIn(mirrorOffsetTemp)

    if self.db["ConservativeWaitCnt"] & 0x1:
        kStr += "s_barrier // debug\n"
        kStr += "s_waitcnt lgkmcnt(0) & vmcnt(0)\n"
        if self.archCaps["SeparateVscnt"]:
          kStr += "s_waitcnt_vscnt null, 0\n"
        kStr += "s_barrier // debug\n"
        #kStr += self.assert_lt(vgpr("Serial"), 64) # examine second wavefront

    if problemType["ZeroPad%s"%tc]:
      self.vgprPool.checkIn(addrV)

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      kStr += inst("s_mov_b32", "m0", \
          hex(kernel["LdsNumElements"] * tP["bpe"]), \
          "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"]))

    if not kernel["BufferLoad"]:
      self.vgprPool.checkIn(maxAddrVgpr)
      self.vgprPool.checkIn(bpeVgpr)
      self.vgprPool.checkIn(zeroVgpr)

    return kStr

  ##############################################################################
  # guardZeroPad
  # add to code module the code to guard subsequent load
  # Inputs:
  #  - offsetVgpr contains GlobalReadOffset
  # Outputs:
  #  - addrV is temp vgpr, returns the guarded address (OOB lanes return -1)
  ##############################################################################
  def guardZeroPad(self, kernel, tP, codeMod, offsetVgpr, soffset, addrV, perp, sPerp, para, sPara):
    tmpSgpr = self.getTmpSgpr(2).idx()
    tc = tP["tensorChar"]
    zps = [zpr for zpr in self.zeroPadRegs[tc].values() if zpr.isMatch(perp, sPerp, para, sPara)]
    for i, zpr in enumerate(zps):
      #zpTmp = tmpSgpr + i + 1
      (freeDim,sumDim) = zpr.zp[:2]
      sumChar = self.indexChars[sumDim]

      codeMod.addComment1("guardZeroPad: "+zpr.regName)
      iterX = "Iter"+sumChar if kernel["PackSummationDims"] else tmpSgpr
      if not kernel["PackSummationDims"]:
        codeMod.addInst("s_sub_u32", sgpr(tmpSgpr), sgpr("Size%s"%sumChar) , sgpr("LoopCounter%s"%sumChar),
                          "loop = Size - remaining loop counter")
      codeMod.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr(iterX), \
                        self.strideRef(tc,sumDim), "LoopCounterZp*strideSum")
      codeMod.addInst("s_lshl_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), \
                        "Bpe%sLog2"%tc, "")
      if soffset != "0":
        assert (soffset == "0") # need to add to scalar above.  Can't happen with UseSgprForGRO=0
        codeMod.addInst("s_add_u32", sgpr(tmpSgpr), sgpr(tmpSgpr), soffset, "add soffset ")

      if sumDim in kernel["ProblemType"]["MirrorDims%s"%tc]:
        codeMod.addInst("_v_sub_u32", vgpr(addrV), vgpr(zpr.regName), sgpr(tmpSgpr), \
                        "<- GRO - scaled elementCounter")
      else:
        codeMod.addInst("_v_add_u32", vgpr(addrV), vgpr(zpr.regName), sgpr(tmpSgpr), \
                        "<- GRO + scaled elementCounter")

      cmpDest = self.vcc if i==0 else sgpr(tmpSgpr,self.laneSGPRCount) # first one writes vcc
      codeMod.addInst("v_cmp_ge_u32", cmpDest, vgpr(addrV), \
                        sgpr("ElementEdge%s%s"%(tc,sumChar)), \
                        "loopCounter*strideSum >= ElementEdge ?")

      if i>0:
        codeMod.addInst("s_or_b{}".format(self.kernel["WavefrontSize"]), self.vcc, self.vcc, sgpr(tmpSgpr,self.laneSGPRCount),"combine elementEdge masks")

      if i==len(zps)-1:
        codeMod.addInst("v_cndmask_b32", vgpr(addrV), vgpr(offsetVgpr), -1, self.vcc, \
                          "Set addresses in pad to large OOB value")

      #if soffset != "0":
      #  assert(sumChar == self.unrollChar) # don't think we need this for non-unroll dims
      #  #codeMod.addText(self.assert_ne(sgpr("WorkGroup0"),1))
      #codeMod.addText(self.bomb())

    return addrV

  ##############################################################################
  # DirectToLds M0 update: Do It A/B
  ##############################################################################
  def directToLdsM0Update(self, kernel, mode, tP, usePlaceHolder=False):
    tc = tP["tensorChar"]
    imod = Code.Module("directToLdsM0Update%s_%u"%(tc,mode))
    DtldsModule = imod.addCode(Code.Module("dtls_offset%s"%tP["tensorChar"]))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod
    if kernel["DirectToLds%s"%tc]:
      # DirectToLds only enabled for TLU=1 cases, where the registers are directly copied into LDS
      # for cases both A&B are DTLS, updating m0 for each GlobalRead requires instruction schedule
      # along with global reads
      assert (kernel["LocalWriteUseSgpr%s"%tc])
      if kernel["ExpandPointerSwap"]:
        DtldsModule.addInst("s_add_u32", "m0", sgpr("LocalWriteAddr%s"%tc), \
                      tP["localWriteSwapByteOffset"], "m0 <- LDS write address")
      else:
        DtldsModule.addInst("s_mov_b32", "m0", sgpr("LocalWriteAddr%s"%tc), "m0 <- LDS write address")

      # PrefetchGlobalRead=2 case, generate local read wait for DirectToLds
      # skip wait and barrier for B if both A and B use DirectToLds
      skipWait = kernel["DirectToLdsA"] and kernel["DirectToLdsB"] and tc == "B"
      if kernel["PrefetchGlobalRead"]==2 and not skipWait:
        # do not generate local read wait for PGR=2
        DtldsModule.addText(self.comment1("before DirectToLds load, ensure prior ds_reads have finished"))
        DtldsModule.addText("s_waitcnt lgkmcnt(0)" + self.endLine)
        # vmcnt is needed for DirectToLdsA + DirectToLdsB
        if (not kernel["NoLdsWriteCode"]) or (kernel["DirectToLdsA"] and kernel["DirectToLdsB"]):
          if usePlaceHolder:
            waitStr = "__placeholder__"
          else:
            waitStr = "0"
          DtldsModule.addText("s_waitcnt vmcnt(%s)"%waitStr + self.endLine)
        DtldsModule.addText("s_barrier" + self.endLine)

    return imod

  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, mode, tP, vregSetIdx=0):
    tc = tP["tensorChar"]
    problemType = self.kernel["ProblemType"]
    imod = Code.StructuredModule("globalReadDo%s_%u"%(tc,mode))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod

    # sizeK % LOCAL_DEPTHU
    guardK = (mode==2)

    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth # load width in elements?
    bpl = int(self.bpeAB * tP["glvw"]) # bytes per load
    instOffset = 0

    loopIdx = self.unrollIdx # TODO - does this handle multiple summation indices?
    if kernel["SuppressNoLoadLoop"]:
      if mode==1 and tP["isA"]:
        imod.header.addInst("s_cmp_eq_i32", \
              self.loopCounter(kernel, loopIdx), \
              "%u"% 1, \
              "%s"%"is this the last iteration")
        imod.header.addInst("s_cmov_b32", \
              sgpr("SrdA+2"), \
              0,
              "Set limit to 0 for last iteration")
        imod.header.addInst("s_cmov_b32", \
              sgpr("SrdB+2"), \
              0,
              "Set limit to 0 for last iteration")

    #tmpSgpr = self.getTmpSgpr(2).idx()
    # TODO - clean up here:
    # +0,+1 - general purpose tmp. i + 2 is the offset for zero-pad index X
    #tmpSgpr = self.getTmpSgpr(2+len(problemType["ZeroPad%s"%tc])).idx()
    #for i, zp in enumerate(problemType["ZeroPad%s"%tc]):
    #  zpTmp = tmpSgpr + i + 2
    #  imod.header.addComment1("Zeropad check:")
    #  (freeDim,sumDim)= zp[:2]
    #  sumChar = self.indexChars[sumDim]
    #  loopIdx = problemType["IndicesSummation"].index(sumDim)
    #  # TODO - fix for GSU, need LOCAL_DEPTHU*GSUp
    #  if guardK:
    #    imod.header.addInst("s_sub_u32", sgpr(zpTmp), self.sizeRef(freeDim), \
    #      self.loopCounter(kernel,loopIdx), "compute elementCounter%s, step2"%(sumChar))
    #  else:
    #    imod.header.addInst("s_mul_i32", sgpr(zpTmp), self.loopCounter(kernel,loopIdx), \
    #      "DepthU", "compute elementCounter%s, step1"%(sumChar))
    #    imod.header.addInst("s_sub_u32", sgpr(zpTmp), self.sizeRef(freeDim), \
    #      sgpr(zpTmp), "compute elementCounter%s, step2"%(sumChar))
    #  imod.header.addInst("s_mul_i32", sgpr(zpTmp), self.strideRef(tc,freeDim), sgpr(zpTmp), "scale by stride")
    #  imod.header.addInst("s_lshl_b32", sgpr(zpTmp), sgpr(zpTmp), log2(self.bpeAB), "scale by bpe")

    # set the first tc for below wait code for DirectToLds
    # if DirectToVgpr is enabled and swapGlobalRead is true, change the first to B
    tc1st = 'A'
    if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
      tc1st = 'B'

    if tc == tc1st and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and not kernel["PrefetchGlobalRead"]==2:
      # generate local read wait for DirectToLds except for PrefetchGlobalRead=2 (for PGR=2, generate wait after m0 value setting)
      imod.header.addText(self.comment1("before DirectToLds load, ensure prior ds_reads have finished"))
      if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and not guardK:
        # no need to generate sync here if DirectToVgpr is enabled and not tail loop
        imod.header.addText("s_waitcnt lgkmcnt(0)" + self.endLine)
      else:
        imod.header.addText(self.syncThreads(kernel))


    if guardK:
      imod.middle.addText(self.globalReadGuardK(kernel, tP, vregSetIdx))
      return imod

    # else not-guardK below:

    extraFields = ""
    if (tP["NonTemporal"] & 1) != 0:
      extraFields += " " + getGlcBitName(kernel["MemoryModifierFormat"])
    if (tP["NonTemporal"] & 2) != 0:
      extraFields += " " + getSlcBitName(kernel["MemoryModifierFormat"])
    if (tP["NonTemporal"] & 4) != 0:
      extraFields += " nt"
    dtlNoDestVgpr = False
    if kernel["DirectToLds%s"%tc]:
      extraFields += " lds"
      dtlNoDestVgpr = globalParameters["AsmCaps"][self.version]["HasDirectToLdsNoDest"]

    directToLdsLoads = 0
    instOffset       = 0
    prevLdsOffset    = 0

    loopCnt = -1
    if problemType["ZeroPad%s"%tc]:
      addrV = self.vgprPool.checkOut(1,"addrV")
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, int(tP["nrcv"]/tP["nrcvpi"])):
            i = sPara + (int(tP["nrcv"]/tP["nrcvpi"])) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            loopCnt += 1
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth
            # Each load may contains a small bundle of instructions, package them together in loadModule:
            loadModule = Code.Module("load%u"%loopCnt)
            imod.middle.addCode(loadModule)

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

              if problemType["ZeroPad%s"%tc] and not (kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]):
                codeMod = Code.Module("guardZeroPad%u"%loopCnt)
                offsetVgpr = self.guardZeroPad(kernel, tP, codeMod, offsetVgpr, soffset, addrV, perp, sPerp, para, sPara)
                loadModule.addCode(codeMod)

              unrollMirrorWithSoffset = kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in problemType["MirrorDims%s"%tc] and soffset != "0"
              # ScalarGlobalReadOffset should be negative value with unroll mirroring.
              # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
              if unrollMirrorWithSoffset:
                codeMod = Code.Module("mirrorIdx%u"%loopCnt)
                codeMod.addInst("_v_sub_u32", vgpr(offsetVgpr), vgpr(offsetVgpr), soffset, "mirror unroll: GRO=GRO-SGRO, soffset=0")
                loadModule.addCode(codeMod)
                soffset_prev = soffset
                soffset = "0"

              if kernel["DirectToLds%s"%tc]:
                # use bpe with GlobalLoadVectorWidth
                ldsInc = self.getLdsIncForDTL(kernel, tP)

                if kernel["UseInstOffsetForGRO"]:
                  # buffer_load only support 12 bit instruction offset
                  # we have to increase m0 if offset is larger thant 12 bits
                  if instOffset >= self.buff_load_inst_offset_max:
                    inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                    loadModule.addInst("s_add_u32", "m0", "m0", inc, "Move LDS write address to next base" )
                    instOffset -= inc
                elif directToLdsLoads != 0:
                  # m0 offset conversion (only for UseInstOffsetForGRO == 0)
                  # in tP["glvw"] == 1 and tP["nrc"] > 1 case, only m0 offset conversion is necessary. row and column index conversion is not necessary.
                  if tP["nrc"] > 1:
                    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                    if tP["grcg"]:
                      if tP["grcv"]:
                        divisorName = tP["lvc"]
                      else:
                        # Fractional load use the more accurate lsc, multiply by VW later
                        divisorName = tP["lsc"]
                    else:
                      if tP["grcv"]:
                        divisorName = tP["lsp"]
                      else:
                        divisorName = tP["lvp"]
                    divisor = kernel[divisorName]
                    # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                    wSize = max(self.kernel["WavefrontSize"], divisor)
                    lscaOffset = para * wSize * int(tP["bpe"] * tP["glvw"])
                    ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                    ldsInc = ldsOffset - prevLdsOffset
                    prevLdsOffset = ldsOffset
                  loadModule.addInst("s_add_u32", "m0", "m0", ldsInc, "Move LDS write address to next line" )
                directToLdsLoads+=1
                destVgpr=0
              elif kernel["DirectToVgpr%s"%tc]:
                # DirectToVgpr case. Need to toggle destination vreg set and adjust instOffset
                destVgpr="G2L%s%u+%u"%(tc, vregSetIdx, g2lIdx)
              else:
                destVgpr="G2L%s+%u"%(tc, g2lIdx)

              # TODO: is it possible to load only hi16 when no in tail? (need to check INT8 too)
              loadModule.addCode( self.chooseGlobalRead(kernel["BufferLoad"], \
                        bpl, destVgpr=destVgpr, \
                        addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                        soffset=soffset, offset=instOffset, \
                        extraFields=extraFields, \
                        dtlNoDestVgpr=dtlNoDestVgpr, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp)))

              if unrollMirrorWithSoffset:
                codeMod = Code.Module("mirrorIdx%u"%loopCnt)
                codeMod.addInst("_v_add_u32", vgpr(offsetVgpr), vgpr(offsetVgpr), soffset_prev, "mirror unroll: restore GRO=GRO+SGRO")
                loadModule.addCode(codeMod)

              if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  instOffset += ldsInc

              #print "IM=", type(imod.instList[-1]), imod.instList[-1],
            else: # not buffer load
              # load one element from address
              if kernel["DirectToVgpr%s"%tc]:
                # DirectToVgpr case. Need to toggle destination vreg set and adjust instOffset
                destVgpr="G2L%s%u+%u"%(tc, vregSetIdx, g2lIdx)
              else:
                destVgpr="G2L%s+%u"%(tc, g2lIdx)
              loadModule.addCode( self.chooseGlobalRead(False, \
                        bpl, \
                        destVgpr=destVgpr, \
                        addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                        soffset=0, offset=0, \
                        extraFields=extraFields, \
                        dtlNoDestVgpr=dtlNoDestVgpr, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp )))

    if self.db["ConservativeWaitCnt"] & 0x1:
        imod.footer.addInst( "s_barrier", "debug")
        imod.footer.addInst( "s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "conservative wait")
        if self.archCaps["SeparateVscnt"]:
          imod.footer.addInst( "s_waitcnt_vscnt", "null", "0", "stores")
        imod.footer.addInst( "s_barrier", "debug")
        #kStr += self.assert_lt(vgpr("Serial"), 64) # examine second wavefront

    # TODO - can remove one of these m0 restores if A and B both TLU
    # StoreCInUnroll + mode 1 (= unroll loop) + PGR=2 case, m0 clamp will be generated in StoreCInUnroll code. No need to generate here.
    if kernel["DirectToLds%s"%tP["tensorChar"]] and not (kernel["StoreCInUnroll"] and mode == 1 and kernel["PrefetchGlobalRead"]==2):
      inst = "s_mov_b32"
      dst = "m0"
      src = hex(kernel["LdsNumElements"] * tP["bpe"])
      comment = "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"])
      # PGR=2 case, footer is located before global read. To avoid setting clamp before global read, store lds clamp code in middle
      if kernel["PrefetchGlobalRead"] == 2:
        imod.middle.addInst(inst, dst, src, comment)
      else:
        imod.footer.addInst(inst, dst, src, comment)

    if problemType["ZeroPad%s"%tc]:
      self.vgprPool.checkIn(addrV)

    return imod


  ##############################################################################
  # Global Read A/B completed
  ##############################################################################
  def doneGlobalABReads(self, kernel):
    kStr = ""
    kStr += self.comment("Done global A/B reads")
    # TODO Many kernels can undefine this, but condition needs to be updated
    # Has a problem with tail loop in convolution kernels which have PK elements, but not marked PK
    # if kernel["BufferLoad"] and not kernel["PrefetchAcrossPersistent"]:
    #   kStr += self.undefineSgpr("SrdA")
    #   kStr += self.undefineSgpr("SrdB")
    # TODO Should be able to define WS here but it gets undef'd by endSummation
    # if kernel["StreamK"] >= 2:
    #   self.defineSgpr("SrdWS", 4, 4)
    return kStr


  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    if kernel["1LDSBuffer"]:
      return kStr
    tc = tP["tensorChar"]
    #fixme-iui  need to use wrapping increment for double or triple buffering:
    if internalPointerSwap:
      tP["localWriteSwapByteOffset"] = 0 if tP["localWriteSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
      kStr += self.comment("(EPS=1) local write swap internal offset -> %u" % tP["localWriteSwapByteOffset"])
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        kStr += inst("s_xor_b32", \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "swap Red Blk SGPR")
      elif not kernel["DirectToVgpr%s"%tc]: # no local write code if DirectToVgpr is enabled
        numLwa = self.numVgprLocalWriteAddressesA if tP["isA"] else self.numVgprLocalWriteAddressesB
        for i in range(0,numLwa):
          kStr += inst("v_xor_b32", \
              vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
              vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A/B
  # used for global-read + tail-loop to reset to writing in red
  ##############################################################################
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    if not self.do["LocalWrite"]: return ""
    tc = tP["tensorChar"]
    kStr = ""
    if kernel["1LDSBuffer"] or kernel["DirectToVgpr%s"%tc]: # no local write code if DirectToVgpr is enabled
      return kStr
    resetMask = hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1 | self.LdsOOB)
    if internalPointerSwap:
      tP["localWriteSwapByteOffset"] = 0
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        kStr += inst("s_and_b32", \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            resetMask, \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "reset to Red")
      else:
        kStr += inst("v_and_b32", \
            vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            resetMask, \
            vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "reset to Red")
    return kStr

  ##############################################################################
  # Local Write: Init Pointers A/B
  ##############################################################################
  def localWriteInitPointers(self, kernel, tP):
    return ""

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
  #   LocalDotLayout>1 will place N elements from same summation 'row' in
  #   adjacent dims, which is handy for feeding dot instructions.
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
  def calculateLdsWriteOffset(self, perp, para, sPerp, sPara, kernel, tP, localWriteCnt):
    tc = tP["tensorChar"]
    ldl = kernel["LocalDotLayout"]
    mask = ldl-1
    #print "tc ", tc, " perp ", perp, " para ", para, " sPerp ", sPerp, " sPara ", sPara
    lscaOffset = para * kernel[tP["lsc"]]
    perp_masked = perp
    perp_rem = 0
    if (ldl > 1):
      if (kernel[tP["mt"]] >= kernel["SubGroup0"] * kernel["SubGroup1"] * tP["glvw"]):
        # Since it will take multiple fetches to get a full MT, we map low bits of perp to small,
        # horizontal shift to fill in gaps we made by spacing out the data for LDL.
        # Other cases will be handled by low bits of uReg in lwaFirstOffset().
        perp_masked = perp & ~mask
        perp_rem = perp & mask
    lspaOffset = perp_masked * kernel[tP["lsp"]]
    rem = 0

    # Add component offset to interleave from different regs
    # and compute mysterious "i"
    assert(sPerp==0 or sPara==0)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset += sPerp & mask
      lscaOffset += sPara
      rem = (sPerp & ~mask) >> log2(ldl)
      if ldl > 1:
        #i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))
        i = localWriteCnt
      else:
        i = sPara + (int(tP["nrcv"]/tP["nrcvpi"])) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp_masked))
      #print "nrcv ", tP["nrcv"], " nrcvpi ", tP["nrcvpi"], " nrc ", tP["nrc"], " nrpv ", tP["nrpv"]
    else:
      lscaOffset += (sPara // ldl) * ldl
      lspaOffset += sPerp
      rem = sPara % ldl
      i = sPara + (int(tP["nrcv"]/tP["nrcvpi"])) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))

    #if not tP["tlu"]:
    #  tmp = sPara
    #  sPara = sPerp
    #  sPerp = tmp
    # print("0lspaOffset", lspaOffset)
    # print("0lscaOffset", lscaOffset)

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    lds_stride = (kernel["_DepthULds"] + LdsPad) if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] \
            else (kernel[tP["mt"]] + LdsPad)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset *= lds_stride
      lspaOffset += rem * ldl + perp_rem
    else:
      lscaOffset *= lds_stride
      lscaOffset += rem

    # print("1lspaOffset", lspaOffset)
    # print("1lscaOffset", lscaOffset)
    #if tP["tlu"] == tP["grcv"]:
    #  lspaOffset *= tP["glvw"]
    #  lscaOffset *= tP["glvw"]

    # print("2lspaOffset", lspaOffset)
    # print("2lscaOffset", lscaOffset)
    offsetElements = (lspaOffset + lscaOffset)
    # print("offsetElements", offsetElements)
    offsetBytes = offsetElements*tP["bpe"]

    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      offsetBytes = offsetBytes + (offsetBytes // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]

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
    comment += " + (%s%d*%s)" % (
        (("%u + "%sPerp) if tP["wuc"] else ""), perp, \
        tP["lsp"])
    if tP["tlu"]:
      comment += "(*MT%s+PAD)" % (tP["tileChar"])
    comment += " = %u" % (offsetBytes)

    return (offsetBytes, i, comment)

  def recalcLocalWriteAddresses(self, kernel, tP, uDu):

    tc = tP["tensorChar"]

    kStr = ""
    kStr += self.comment("recalculate LocalWriteAddr{}".format(tc))

    lwvw = getattr(self, "localWriteWidth{}".format(tc))
    newInstIdx = self.selectMemoryInstruction("LocalWrite", lwvw*kernel["DepthULdsDivisor"], \
        kernel["LocalWrite2A"], \
        self.localWrite2CoalescedA, self.localWrite2PerpendicularA,
        [self.localWriteStrideTileA, self.localWriteStrideUnrollA] )
    tP["localWriteInstruction"] = self.memoryInstructions["LocalWrite"][newInstIdx]

    if kernel["PersistentKernel"] or kernel["StreamK"]:
      if getattr(self, "oriLwa%s"%tc) is None:
        setattr(self, "oriLwa%s"%tc, self.vgprPool.checkOut(1, "OriLocalWriteddr%s"%tc) )
        kStr += inst("v_mov_b32", vgpr(getattr(self, "oriLwa%s"%tc)), vgpr("LocalWriteAddr%s"%tc), "back up LWA for persistent kernel + wider local read")

    # global read tile assignment
    kStr += self.graTileAssignment(kernel, tP)
    # global read tile offsets
    kStr += self.graTileOffsets(kernel, tP)
    # global read unroll offsets
    kStr += self.graUnrollOffsets(kernel, tP)
    # still needed for vgpr resource management
    # intentionally not emitting code
    self.graFinalOffsets(kernel, tP)

    # local write tile assignments
    kStr += self.lwaTileAssignment(kernel, tP)
    # local write unroll assignments
    kStr += self.lwaUnrollAssignment(kernel, tP)
    # local write local write first offsets
    kStr += self.lwaFirstOffset(kernel, tP, uDu)
    # local write final offsets
    kStr += self.lwaFinalOffsets(kernel, tP)
    # local write declare addresses
    kStr += self.lwaDeclareAddresses(kernel, tP)

    return kStr

  def recalcLocalReadAddressesAB(self, kernel):
    imod = Code.Module()

    if self.inTailLoop:
      # it do 1 iteration each loop in tail loop, and is no use to wider local read next iteration.
      # In 1 block MI, it remap localReadAddr in order to let each thread wider local read continuous k
      # this decrease performance since it require more loop to handle continuous k in each thread.
      # recalculate localReadAddr to cancel wider local read in tail loop
      # TODO: If DepthULdsDivisor>1, local read addr is incremented for each K the loop iterates, which
      # upon second sub-loop needs to be reset to its original value. Backing up local read address would
      # be nicer than recomputing them
      # MatrixInstB>1 case, recalc code is not necessary (regenerated code will be same as original local read offset code)
      # however, we need to update related variables below and regenerate local read instruction based on new numReadsIterCoalesced
      numReadsIterCoalescedA = self.numReadsIterCoalescedA
      numReadsIterCoalescedB = self.numReadsIterCoalescedB

      kStr = ""

      needRecalc = kernel.enabledSplitLDS or (numReadsIterCoalescedA > 1 or numReadsIterCoalescedB > 1) #and tP["isB"]:
      # backup LocalReadAddr
      # LdsPad + LBSPP case, need to backup LocalReadAddr even if recalc is not done
      needBackupLRAddr = needRecalc or (kernel["LdsPadA"] and kernel["LdsBlockSizePerPadA"] or kernel["LdsPadB"] and kernel["LdsBlockSizePerPadB"])

      if needBackupLRAddr:
        # need to back-up the LRA before reCalculation for wider local read (when no wlr, no need to do this)
        if kernel["PersistentKernel"] or kernel["StreamK"]:
          if self.oriLraA is None and not kernel["DirectToVgprA"]: # no local read code if DirectToVgpr is enabled
            self.oriLraA = self.vgprPool.checkOut(1, "OriLocalReadAddrA")
            kStr += inst("v_mov_b32", vgpr(self.oriLraA), vgpr("LocalReadAddrA"), "back up LRA for persistent kernel + wider local read")
          if self.oriLraB is None and not kernel["DirectToVgprB"]: # no local read code if DirectToVgpr is enabled
            self.oriLraB = self.vgprPool.checkOut(1, "OriLocalReadAddrB")
            kStr += inst("v_mov_b32", vgpr(self.oriLraB), vgpr("LocalReadAddrB"), "back up LRA for persistent kernel + wider local read")

      if needRecalc:
        self.numReadsIterCoalescedA = 1
        self.numReadsIterCoalescedB = 1
        self.numIterPerCoalescedReadA = max(1,self.numReadsIterCoalescedA//kernel["InnerUnroll"])
        self.numIterPerCoalescedReadB = max(1,self.numReadsIterCoalescedB//kernel["InnerUnroll"])
        self.lrvwA = kernel["MIInputPerThread"]
        self.lrvwB = kernel["MIInputPerThread"]

        kStrRecalc = ""
        kStrRecalc += (self.lraTileAssignment(kernel, self.tPA, self.tPB))
        kStrRecalc += (self.lraFinalOffset(kernel, self.tPA))
        kStrRecalc += (self.lraDeclareAddresses(kernel, self.tPA))
        kStrRecalc += (self.lraFinalOffset(kernel, self.tPB))
        kStrRecalc += (self.lraDeclareAddresses(kernel, self.tPB))
        if kernel["MatrixInstB"] == 1:
          # recalc code is necessary only for MatrixInstB=1
          kStr += kStrRecalc
        localRead2Perpendicular = False
        instructions = self.memoryInstructions

        if kernel["UnrollMajorLDSA"]:
          localReadWidth = (kernel["MIInputPerThread"] * self.tPA["bpe"]) // self.bpr
          self.localReadInstructionIdxA = \
            self.selectMemoryInstruction("LocalRead", localReadWidth, \
            kernel["LocalRead2A"], \
            self.localRead2CoalescedA, localRead2Perpendicular,
            [self.localReadStrideCoalescedA] )
          self.localReadInstructionA = instructions["LocalRead"][self.localReadInstructionIdxA]
          self.tPA["localReadInstruction"] = self.localReadInstructionA

        if kernel["UnrollMajorLDSB"]:
          localReadWidth = (kernel["MIInputPerThread"] * self.tPB["bpe"]) // self.bpr
          self.localReadInstructionIdxB = \
            self.selectMemoryInstruction("LocalRead", localReadWidth, \
            kernel["LocalRead2B"], \
            self.localRead2CoalescedB, localRead2Perpendicular,
            [self.localReadStrideCoalescedB] )
          self.localReadInstructionB = instructions["LocalRead"][ \
            self.localReadInstructionIdxB]
          self.tPB["localReadInstruction"] = self.localReadInstructionB

      if kStr != "":
        imod.addCode(kStr)

    return str(imod)

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    imod = Code.Module()

    # can't optimize, insert the general LWDo
    if not self.canOptimizePreLoopLWVmcnt:
      LWDoMod = imod.addCode(Code.Module())
      LWDoA = self.localWriteDo(kernel, tPA)
      LWDoB = self.localWriteDo(kernel, tPB)
      LWDoMod.addText(self.comment("local write a"))
      LWDoMod.addCode(LWDoA)
      LWDoMod.addText(self.comment("local write b"))
      LWDoMod.addCode(LWDoB)
      return imod

    # Opt for PAP waitcnt, 4 cases:
    # one for the first PK-loop, one for Opt-NLL, one for Edge, one for Beta
    basic_gl_Label = self.getNamedLabel("Basic_GL_Label")
    optNLL_lw_Label = self.getNamedLabel("OptNLL_LW_Label")
    ordNLL_E1_lw_Label = self.getNamedLabel("OrdNLL_E1_LW_Label")
    ordNLL_B1_lw_Label = self.getNamedLabel("OrdNLL_B1_LW_Label")
    optNLL_SCIUl_Label = self.getNamedLabel("OptNLL_SCIU_Label")
    lwEnd_Label = self.getNamedLabel("PreLoopLWEnd")

    self.useManualVmcnt = True
    self.vmcntDec = 0
    # Template LWDoCode, not added to imod. Using "__placeholder__" ( vmcnt("__placeholder__ + Basic_Load - Decrement") )
    LWDoCodeTemplate = Code.Module()
    if not kernel["NoLdsWriteCode"]:
      LWDoA = self.localWriteDo(kernel, tPA)
      LWDoB = self.localWriteDo(kernel, tPB)
      LWDoCodeTemplate.addText(self.comment("local write a"))
      LWDoCodeTemplate.addCode(LWDoA)
      LWDoCodeTemplate.addText(self.comment("local write b"))
      LWDoCodeTemplate.addCode(LWDoB)
    else:
        # no local write code case(DirectToVgpr + DirectToLds)
        # add only wait for global read here
        LWDoCodeTemplate.addText(self.comment("global read wait in no local write case"))
        LWDoCodeTemplate.addText("s_waitcnt vmcnt(__placeholder__+0+0)\n")

    codeTemplateStrList = LWDoCodeTemplate.flatitems()
    self.useManualVmcnt = False
    # "Basic_Load" should == the final number of vmcnt-decrement ( Since "Basic_Load - Decrement" would be 0 )
    self.preLoopVmcntDict[ PreLoopVmcntCase.Basic_Load ] = self.vmcntDec

    # Branch conditions
    BranchMod = imod.addCode(Code.Module("Branch Module"))

    # barrier, but can be skipped for the first PK Loop
    barrierComment = "for the second or later PKLoop, need to ensure the prev DS_READ for SR or MFMA are finished before LW\n"
    BranchMod.addInst("\ns_barrier",  "", barrierComment)

    if kernel["StoreCInUnroll"]:
      BranchMod.addInst("s_cmp_eq_u32", sgpr("PreLoopLWVmcntCase"), hex(5), "Case 5: PK Loop for StoreCInUnroll?")
      BranchMod.addInst("s_cbranch_scc1", optNLL_SCIUl_Label, "jump to Case 5")
    BranchMod.addInst("s_cmp_eq_u32", sgpr("PreLoopLWVmcntCase"), hex(1), "Case 1: First PK Loop?")
    BranchMod.addInst("s_cbranch_scc1", basic_gl_Label, "jump to Case 1, can skip the s_barrier")

    # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
    if not (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
      BranchMod.addInst("s_cmp_eq_u32", sgpr("PreLoopLWVmcntCase"), hex(2), "Case 2: Prev PK-Loop is Opt-NLL?")
      BranchMod.addInst("s_cbranch_scc1", optNLL_lw_Label, "jump to Case 2")
    if kernel["ProblemType"]["UseBeta"]:
      # UseBeta case, 4 is the last option
      # Use s_branch for 4 (not generate the condition to check if the value is 4
      BranchMod.addInst("s_cmp_eq_u32", sgpr("PreLoopLWVmcntCase"), hex(3), "Case 3: Prev PK-Loop is Ord-NLL with edge?")
      BranchMod.addInst("s_cbranch_scc1", ordNLL_E1_lw_Label, "jump to Case 3")
      # BranchMod.addInst("s_cmp_eq_u32", sgpr("PreLoopLWVmcntCase"), hex(4), "Case 4: Prev PK-Loop is Ord-NLL with beta?")
      # BranchMod.addInst("s_cbranch_scc1", ordNLL_B1_lw_Label, "jump to Case 4")
      BranchMod.addInst("s_branch", ordNLL_B1_lw_Label, "jump to Case 4")
    else:
      # no Beta case, 3 is the last option
      BranchMod.addInst("s_branch", ordNLL_E1_lw_Label, "jump to Case 3")

    # Fast duplicate LWDoCodeTemplate four times to different placeholder keywords for later replacement (after global write)
    # can avoid calling localWriteDo() for several times

    basicVmcntKW = PreLoopVmcntCase( PreLoopVmcntCase.Basic_Load ).name
    addWmcnt = "0"
    if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
      # DirectToVgpr case, only wait for the first iter for DirectToVgpr side global load
      numGlobalReadA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] * self.numReadVectorComponentsA
      numGlobalReadB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] * self.numReadVectorComponentsB
      numGlobalRead = numGlobalReadA if kernel["DirectToVgprA"] else numGlobalReadB
      # delay DirectToVgpr global read which is not referred yet
      addWmcnt = str(numGlobalRead - (0 + 1) * (numGlobalRead // kernel["LoopIters"]))

    # CASE 1:
    # replace vmcnt("__placeholder__ + Basic_Load - Decrement") to vmcnt("Basic_Load - Decrement")
    currCaseKW = basicVmcntKW
    LWDoCase1Mod = imod.addCode(Code.Module(currCaseKW))
    LWDoCase1Mod.addText("\n%s:" % basic_gl_Label)
    LWDoCase1Mod.addComment1("global-load-cnt = %s+%s"%(basicVmcntKW, addWmcnt))
    for item in codeTemplateStrList:
      LWDoCase1Mod.addText(str(item).replace("__placeholder__", addWmcnt))
    LWDoCase1Mod.addInst("s_branch", lwEnd_Label, "finish case, jump to end of LW")

    # CASE 2:
    # replace vmcnt("__placeholder__ + Basic_Load - Decrement") to vmcnt("OptNLL_Store + Basic_Load - Decrement")
    # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
    if not (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
      currCaseKW = PreLoopVmcntCase( PreLoopVmcntCase.OptNLL_Store ).name
      LWDoCase2Mod = imod.addCode(Code.Module(currCaseKW))
      LWDoCase2Mod.addText("\n%s:" % optNLL_lw_Label)
      LWDoCase2Mod.addComment1("prev-global-store-cnt = %s, global-load-cnt = %s+%s"%(currCaseKW, basicVmcntKW, addWmcnt))
      for item in codeTemplateStrList:
        LWDoCase2Mod.addText(str(item).replace("__placeholder__",str(currCaseKW) + "+" + addWmcnt))
      LWDoCase2Mod.addInst("s_branch", lwEnd_Label, "finish case, jump to end of LW")

    # CASE 3:
    # replace vmcnt("__placeholder__ + Basic_Load - Decrement") to vmcnt("OrdNLL_E1_Store + Basic_Load - Decrement")
    currCaseKW = PreLoopVmcntCase( PreLoopVmcntCase.OrdNLL_E1_Store ).name
    LWDoCase3Mod = imod.addCode(Code.Module(currCaseKW))
    LWDoCase3Mod.addText("\n%s:" % ordNLL_E1_lw_Label)
    LWDoCase3Mod.addComment1("prev-global-store-cnt = %s, global-load-cnt = %s+%s"%(currCaseKW, basicVmcntKW, addWmcnt))
    for item in codeTemplateStrList:
      LWDoCase3Mod.addText(str(item).replace("__placeholder__",str(currCaseKW) + "+" + addWmcnt))
    LWDoCase3Mod.addInst("s_branch", lwEnd_Label, "finish case, jump to end of LW")

    if kernel["ProblemType"]["UseBeta"]:
      # CASE 4:
      # replace vmcnt("__placeholder__ + Basic_Load - Decrement") to vmcnt("OrdNLL_B1_Store + Basic_Load - Decrement")
      currCaseKW = PreLoopVmcntCase( PreLoopVmcntCase.OrdNLL_B1_Store ).name
      LWDoCase4Mod = imod.addCode(Code.Module(currCaseKW))
      LWDoCase4Mod.addText("\n%s:" % ordNLL_B1_lw_Label)
      # special for case 4, prev store already did vmcnt(n) for loading beta, we don't need any vmcnt here
      # so only keep the lines without s_waitcnt vmcnt( __placeholder__ ), otherwise, discard them
      # LWDoCase4Mod.addComment1("prev-global-store-cnt = %s, global-load-cnt = %s"%(currCaseKW, basicVmcntKW))
      for item in codeTemplateStrList:
        if (str(item).find("__placeholder__") == -1):
          LWDoCase4Mod.addText(str(item))
      # if StoreCInUnroll is not enabled, this is the last case. No jump necessary.
      if kernel["StoreCInUnroll"]:
        LWDoCase4Mod.addInst("s_branch", lwEnd_Label, "finish case, jump to end of LW")

    if kernel["StoreCInUnroll"]:
      # CASE 5:
      # replace vmcnt("__placeholder__ + Basic_Load - Decrement") to vmcnt("Basic_Load - Decrement")
      # adjust addWmcnt for StoreC
      numStoreCInTemplate = self.getNumberOfStoreCInTemplate(kernel)
      addWmcntForStoreC = numStoreCInTemplate
      # PGR=2 and noLoadC case add numStoreCInTemplate once more for the second last iter
      needLoadC = (not kernel["AtomicAddC"]) and kernel["ProblemType"]["UseBeta"]
      if kernel["PrefetchGlobalRead"]==2 and not needLoadC:
        addWmcntForStoreC += numStoreCInTemplate
      addWmcnt = addWmcnt + "+" + str(addWmcntForStoreC)
      imod.addText("\n%s:" % optNLL_SCIUl_Label)
      imod.addComment1("global-load-cnt = %s+%s"%(basicVmcntKW, addWmcnt))
      for item in codeTemplateStrList:
        imod.addText(str(item).replace("__placeholder__", addWmcnt).replace("Basic_Load", str(self.vmcntDec)))

    # End
    imod.addText("\n%s:" % lwEnd_Label)

    return imod

  ##############################################################################
  # Replace the determined vmcnt in PreLoop LocalWrite
  ##############################################################################
  def replacePreLoopLWVmcnt(self, kernel):
    # This replaces the vmcnt keywords with the actual number
    # ("Basic_Load"/"OptNLL_Store"/"OrdNLL_E1_Store"/"OrdNLL_B1_Store")

    maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]

    # Iterate each PreLoopVmcnt case which needs to replace keyword to number
    for vmcntCase in self.preLoopCaseToReplaceKWList:
      toReplaceList = self.preLoopCaseToReplaceKWList[vmcntCase]
      # get the module corresponding to the case
      codeMod = self.preLoopLocalWriteCode.findNamedItem( PreLoopVmcntCase(vmcntCase).name )
      if codeMod:
        numItems = len(codeMod.itemList)
        # for each module, loop each item string, pop from head -> replace -> append to tail
        for idx in range(0,numItems):
          replacedCode = str(codeMod.itemList.pop(0))
          # Get the vmcnt keywords need to be replaced for this case
          # replace each keyword with actual number (calculated in global write)
          for toReplaceCase in toReplaceList:
            vmcntCaseKeyword = PreLoopVmcntCase(toReplaceCase).name
            replacedCode = replacedCode.replace(vmcntCaseKeyword, "%u"%(self.preLoopVmcntDict[toReplaceCase]))#
          #
          # Up to here, the replacedCode is "....vmcnt(A+B-C)", which is possible to exceed MaxVmcnt
          # So we need to do the final evaluation
          #
          valStartPos = replacedCode.find("vmcnt(")
          if valStartPos != -1:
            valEndPosEnd = replacedCode.find(")")
            valStartPos += 6
            # get the str of "A+B-C" to evaluate
            valueStr = replacedCode[valStartPos : valEndPosEnd]
            # replace "A+B-C" to final evaluated value, since we need to test min(value, maxVmcnt)
            # "..... vmcnt(" + final_value + ")", and add comment
            replacedCode = "%-50s // %s \n" %( \
              replacedCode[:valStartPos] + str( min(maxVmcnt, eval(valueStr)) ) + ")", \
              ("min(maxVmcnt, (%s))"%valueStr) \
              )

          codeMod.addText(replacedCode)

    return

  ##############################################################################
  # Local Write: Do It A/B
  # uDu: 'None' means to use fractional local write (where not all threads are active)
  #      when DepthULdsDivisor > 1
  ##############################################################################
  def localWriteDo(self, kernel, tP, uDu=0):
    if not self.do["LocalWrite"]: return "", -1

    tc = tP["tensorChar"]
    self.localWriteDoCnt += 1
    imod = Code.Module()

    if (not kernel["DirectToLds%s"%tc]) and (not kernel["DirectToVgpr%s"%tc]):
      instruction = tP["localWriteInstruction"]
      numBlocks = instruction.numBlocks
      numOffsets = instruction.numOffsets
      blockWidth = instruction.blockWidth
      #offsetMultiplier = instruction.offsetMultiplier
      g2lIdx = 0
      #kStr += dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
      if 0:
        print("\nLocalWrite", tP["tensorChar"])
        print("tlu", tP["tlu"])
        print("lsc", kernel[tP["lsc"]])
        print("lsp", kernel[tP["lsp"]])
        print("grcv", tP["grcv"])
        print("wtc", tP["wtc"])
        print("wuc", tP["wuc"])
        print("nrc", tP["nrc"])
        print("nrp", tP["nrp"])
        print("nwcv", tP["nwcv"])
        print("nwpv", tP["nwpv"])
        print("nrcvpi", tP["nrcvpi"])
        print("nwcvpi", tP["nwcvpi"])

      tmpLocalWriteAddr = -1

      # due to bug in scheduler, cannot use temp sgpr for now
      #tempFp16Sgpr = -1
      #if kernel["ProblemType"]["Fp16AltImpl"] and kernel["ProblemType"]["Fp16AltImplRound"]:
      #  tempFp16Sgpr = self.sgprPool.checkOutAligned(self.laneSGPRCount, 2, "FP16 Alt Round Temp SGPRs")

      # using _ds_store_b8: need one more vgpr space to do lshr
      tmpVgprOffset = ((self.numVgprG2LA if (tP['tensorChar'] == 'A') else self.numVgprG2LB) / 2) if (blockWidth == 0.25) else 0

      loopCnt = 0
      # if transposing, positions of sPerp and sPara are transposed
      instructionCnt = 0
      for perp in range(0, tP["nrp"]):
        localWriteCode = imod.addCode(Code.Module("LocalWrite%u perp=%d"%(instructionCnt,perp)))
        lwa = "LocalWriteAddr%s"%tc  # default
        if kernel["FractionalLoad"] and perp==tP["nrp"]-1:
          # add inline here:
          overhang = kernel["fractionalPerpOverhang%s"%tc]
          if overhang:
            if kernel["FractionalLoad"]==1:
              # Use already-computed vpr:
              lwa = "LocalWriteAddrOverhang%s"%tc
            elif kernel["FractionalLoad"]==2:
              if tmpLocalWriteAddr == -1:
                tmpLocalWriteAddr = self.vgprPool.checkOut(1,"tmpLocalWriteAddr")

              validWI = overhang*kernel[tP["lsc"]]/tP["glvw"]
              #print "%s: overhang=%u element validWI=%u" % (tc, overhang, validWI)
              localWriteCode.addText(self.comment1("LastPerp.  overhang=%u, mask WI>%u" % (overhang, validWI)))
              localWriteCode.addInst("v_cndmask_b32", \
                          vgpr(tmpLocalWriteAddr), \
                          1.0, \
                          vgpr("LocalWriteAddr%s"%tc), \
                          sgpr("PerpOverhangVcc%s"%tc,2), \
                          "Mask load so out-of-gr-tile bounds returns 0. Note 1.0f=0x3f80000 which is large non-neg int")
              lwa = tmpLocalWriteAddr

        for para in range(0, tP["nrc"]):
          if para>=1:
            localWriteCode = imod.addCode(Code.Module("LocalWrite%u perp=%d para=%d"%(instructionCnt,perp,para)))

          # insert the manual vmcnt for each nrc
          if self.useManualVmcnt == True:
            self.vmcntDec += 1
            localWriteCode.addText("s_waitcnt vmcnt(__placeholder__+%s-%u)\n" \
              %( PreLoopVmcntCase(PreLoopVmcntCase.Basic_Load).name, self.vmcntDec))

          for s in range(0, roundUp(max(tP["nwcv"],tP["nwpv"])/tP["nwcvpi"])):
            sPerp = 0
            sPara = 0
            if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
              if tP["wtc"] == tP["grcv"]:
                sPerp = s
              elif tP["wuc"] == tP["grcv"]:
                sPara = s
            else:
              if tP["wtc"] == tP["grcv"]:
                sPara = s
              elif tP["wuc"] == tP["grcv"]:
                sPerp = s

            #print("perp:{}/{} para:{}/{} sPerp:{} sPara:{} loopCnt:{}".format(perp,tP["nrp"],para,tP["nrc"],sPerp,sPara,loopCnt))
            (offset, i, comment) = self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP, loopCnt)

            if uDu is None:
              g2lIdx = int(i * blockWidth)
            else:
              # Example: DepthULdsDivisor=2
              # v0, v1, v2, v3 | v0, v1, v2, v3 | ... ----> unroll dim
              # -----Thd 0----- -----Thd 1-----   ...
              # 1st subloop writes v0,v1 to LDS
              # 2nd subloop writes v2,v3 to LDS
              g2lIdx = int((i * kernel["DepthULdsDivisor"] + uDu) * blockWidth)
              #print("uDu=%u, g2lIdx = %u, offset: %u"%(uDu, g2lIdx, offset))

            # TODO- INT8: check uDu
            if (blockWidth == 0.25) and ((s % 4) == 0):
                src = "G2L%s+%u" % (tc, g2lIdx)
                dst = "G2L%s+%u+%u" % (tc, tmpVgprOffset, g2lIdx)
                localWriteCode.addInst("v_mov_b32", vgpr(dst), vgpr(src), "another VGPR storing lshr 8-bit value")
                localWriteCode.addInst("v_lshrrev_b32", vgpr(dst), "0x8", vgpr(dst), "G2L Vpgr >> 8")

            paramList = []
            paramList.append(vgpr(lwa))
            for blockIdx in range(0, numBlocks):
              if blockWidth == 1:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
              elif blockWidth == 0.25 and ((s % 2) == 1): # Int8, s = 1 or 3 (high8Bits)
                paramList.append(vgpr("G2L%s+%u+%u"%(tc, tmpVgprOffset, g2lIdx)))
              else:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), blockWidth))
              if self.db["ForceInputValue%s"%tc]:
                localWriteCode.addInst("v_mov_b32", vgpr("G2L%s+%u"%(tc, g2lIdx)), self.db["ForceValue%s"%tc], "ForceInputValue")
              if kernel["ProblemType"]["Fp16AltImpl"]:
                numIters = 1 if blockWidth <= 1 else blockWidth
                for iter in range(0, numIters):
                  vgprsrc = vgpr("G2L%s+%u"%(tc, g2lIdx+iter))

                  localWriteCode.addInst("v_cvt_f32_f16", vgpr("G2Lpipe0"), vgprsrc,"")
                  if kernel["ProblemType"]["Fp16AltImplRound"]:
                    localWriteCode.addInst("v_cmp_u_f32", sgpr("Fp16AltNanCheck", self.laneSGPRCount), vgpr("G2Lpipe0"), vgpr("G2Lpipe0"), "check NaN" )
                    localWriteCode.addInst("v_add_u32", vgpr("Fp16AltTmp"), sgpr("Fp16AltOffset"), vgpr("G2Lpipe0"), "")
                    localWriteCode.addInst("v_cndmask_b32", vgpr("G2Lpipe0"), vgpr("Fp16AltTmp"), vgpr("G2Lpipe0"), sgpr("Fp16AltNanCheck",self.laneSGPRCount), "" )

                  localWriteCode.addInst("v_cvt_f32_f16", vgpr("G2Lpipe1"), vgprsrc, "src0_sel:WORD_1", "")
                  if kernel["ProblemType"]["Fp16AltImplRound"]:
                    localWriteCode.addInst("v_cmp_u_f32", sgpr("Fp16AltNanCheck",self.laneSGPRCount), vgpr("G2Lpipe1"), vgpr("G2Lpipe1"), "check NaN" )
                    localWriteCode.addInst("v_add_u32", vgpr("Fp16AltTmp"), sgpr("Fp16AltOffset"), vgpr("G2Lpipe1"), "")
                    localWriteCode.addInst("v_cndmask_b32", vgpr("G2Lpipe1"), vgpr("Fp16AltTmp"), vgpr("G2Lpipe1"), sgpr("Fp16AltNanCheck",self.laneSGPRCount), "" )

                  localWriteCode.addInst("v_pack_b32_f16", vgprsrc, vgpr("G2Lpipe0"),vgpr("G2Lpipe1"), "op_sel:[1,1,0]","")

            for oIdx in range(0, numOffsets):
              paramList.append(offset)

            #print "offset", offset

            paramTuple = tuple(paramList)
            #comment = "Reg -> L %u_%u_%u_%u"%(para, sPara, perp, sPerp)
            #comment += " #%u"%self.localWriteDoCnt
            nonTemporal = 0
            isHigh16Bits = False
            if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
              if s%2==1:
                isHigh16Bits = True
              if tP["glvw"]==1 and instructionCnt%2==1:
                isHigh16Bits = True

            #       |  hi16  |  hi16  |        |        |
            #       |  hi8   |        |   hi8  |        |
            #############################################
            # VGPR: |---w4---|---w3---|---w2---|---w1---| -> b8_d16: get w1 / _b8_d16_hi: get w3
            # LSHR: |--------|---w4---|--------|---w2---| -> b8_d16: get w2 / _b8_d16_hi: get w4
            elif kernel["ProblemType"]["DataType"].isInt8() \
                or kernel["ProblemType"]["DataType"].is8bitFloat():
              isHigh16Bits = (s % 4) > 1 # 2,3
              # TODO
              # if tP["glvw"]==1 and instructionCnt%2==1:
              #   isHigh16Bits = True
            localWriteCode.addCode(Code.LocalWriteInst( \
                instruction.IssueLatency, \
                tP["localWriteInstruction"].toCodeInst(paramTuple, \
                nonTemporal, isHigh16Bits),comment))
            instructionCnt += 1
            loopCnt+=1
      if tmpLocalWriteAddr != -1:
        self.vgprPool.checkIn(tmpLocalWriteAddr)

      # due to bug in scheduler, cannot use temp sgpr for now
      #if tempFp16Sgpr != -1:
      #  self.sgprPool.checkIn(tempFp16Sgpr)

    # localWriteDoCnt<=2 is prefetch if PrefetchGlobalRead:
    if 0 and tP["isB"]: # post-lds-write
    #if 0 and self.localWriteDoCnt >= 0:
      localWriteCode.addInst( "s_waitcnt lgkmcnt(0) & vmcnt(0)", "")
      if self.archCaps["SeparateVscnt"]:
        localWriteCode.addInst( "s_waitcnt_vscnt", "null", "0", "")
      localWriteCode.addInst("s_barrier", "dump LDS" )
      localWriteCode.addText(self.assert_ne(sgpr("WorkGroup0"),1))
      #localWriteCode.addText(self.bomb())

    return imod

  ##############################################################################
  # Local Read: Swap Offsets A/B
  # internalPointerSwap: swap internally tracked offsets - rather than
  #    emit specific instructions to do the pointer swap
  ##############################################################################
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]) or kernel["DirectToVgpr%s"%tc]: return "" # no local read code if DirectToVgpr is enabled
    kStr = ""
    if kernel["1LDSBuffer"]:
      return kStr
    if internalPointerSwap:
      tP["localReadSwapByteOffset"] = 0 if tP["localReadSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
      kStr += self.comment("local read swap internal offset -> %u" % tP["localReadSwapByteOffset"])
    else:
      kStr += inst("v_xor_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A/B
  # x % n == n & (n-1) for n power of 2
  # tP[localReadOffset] maintains running count of offsets
  # This is called from the tail loop to reset read offsets?
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""
    if kernel["1LDSBuffer"] or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return kStr
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadSwapByteOffset"] = 0
      kStr += self.comment("localReadResetOffsets")
      tP["localReadOffset"] = 0
      kStr += self.comment1("handled internally")
    kStr += inst("v_and_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]) or kernel["DirectToVgpr%s"%tc]: return "" # no local read code if DirectToVgpr is enabled
    kStr = ""
    if self.localReadInstructionA.numOffsets == 1:
      kStr += self.comment("localReadInitPointers")
      tP["localReadOffset"] = 0
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*tP["bpe"]-1), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "init Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read offset conversion for DirectToLds
  ##############################################################################
  def localReadOffsetConvForDTL(self, kernel, tP, offset_val):
    tc = tP["tensorChar"]

    # ThreadSeparateGlobalRead special case
    if kernel["ThreadSeparateGlobalRead%c"%tc]:
      # ThreadSeparateGlobalRead special case. Align with calculation in LocalRead
      lrdOffsetMod = (kernel["_DepthULds"]//(kernel["ThreadSeparateGlobalRead%c"%tc]*2))
      if (offset_val >= lrdOffsetMod):
        MblockSizePerLoad = int(kernel["WavefrontSize"] * kernel["GlobalLoadVectorWidth%c"%tc]) // kernel["_DepthULds"]
        offset_val_div = offset_val // lrdOffsetMod
        offset_val_mod = offset_val % lrdOffsetMod
        offset_val = offset_val_div * lrdOffsetMod * MblockSizePerLoad + offset_val_mod

    offset_val *= tP["bpe"]

    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
    dummy, offset_val = self.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val)

    return offset_val

  ##############################################################################
  # Local Read: Increment A/B sub function
  ##############################################################################
  def localReadIncSub(self, kernel, iui, tP, LdsPad):
    tc = tP["tensorChar"]

    # offset increment calculation for both tail loop and not tail loop cases
    inc_base = (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad)
    inc_base_lsu = kernel["LocalSplitU"] * inc_base
    numReadsIterCoalescedA = self.numReadsIterCoalescedA
    numReadsIterCoalescedB = self.numReadsIterCoalescedB
    if kernel["EnableMatrixInstruction"]:
      if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
        inc = kernel["LocalSplitU"] * kernel["MatrixInstK"] * max(numReadsIterCoalescedA, numReadsIterCoalescedB)
      else:
        if tc == "A":
          count = (iui+1) if self.inTailLoop else self.localReadDoCntA
          if kernel["MatrixInstB"] != 1 or self.lrvwA == self.lrvwB:
            inc = inc_base_lsu * kernel["MatrixInstK"] * numReadsIterCoalescedA
          else:
            if (count)%(kernel["LocalReadVectorWidth"]//self.lrvwA):
              inc = inc_base * self.lrvwA
            else:
              inc = inc_base * (kernel["LocalSplitU"] * kernel["MatrixInstK"]*kernel["LocalReadVectorWidth"]//self.lrvwA-self.lrvwA*(kernel["LocalReadVectorWidth"]//self.lrvwA-1))
        else:
          count = (iui+1) if self.inTailLoop else self.localReadDoCntB
          if kernel["MatrixInstB"] != 1 or self.lrvwA == self.lrvwB:
            inc = inc_base_lsu * kernel["MatrixInstK"] * numReadsIterCoalescedB
          else:
            if (count)%(kernel["LocalReadVectorWidth"]//self.lrvwB):
              inc = inc_base * self.lrvwB
            else:
              inc = inc_base * (kernel["LocalSplitU"] * kernel["MatrixInstK"]*kernel["LocalReadVectorWidth"]//self.lrvwB-self.lrvwB*(kernel["LocalReadVectorWidth"]//self.lrvwB-1))
    else:
      inc = inc_base_lsu

    return inc

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    tc = tP["tensorChar"]
    if not self.do["LocalRead%s" % tc] or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return ""

    kStr = ""

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0

    # offset increment calculation for both tail loop and not tail loop cases
    inc = self.localReadIncSub(kernel, iui, tP, LdsPad)

    if self.inTailLoop:
      comment = " (LSU*(MT+PAD)*bpe)"
      bpe = tP["bpe"]
      if kernel["EnableMatrixInstruction"]:
        if kernel["UnrollMajorLDS%s" % tc]:
          if kernel["DirectToLds%s" % tc]:
            # DirectToLds special case. Need special address conversion
            localReadOffset = inc
            prev_offset_val = localReadOffset * iui
            offset_val = localReadOffset * (iui + 1)
            # offset conversion or DirectToLds
            prev_offset_val= self.localReadOffsetConvForDTL(kernel, tP, prev_offset_val)
            offset_val= self.localReadOffsetConvForDTL(kernel, tP, offset_val)
            inc = offset_val - prev_offset_val
            comment = " (local read inc for DTL)"
            # DirectToLds case, no need to multiply by bpe. so set bpe=1 here
            bpe = 1
          else:
            comment = " (LSU*bpe)"
      inc *= bpe
      # adjust inc for LBSPP in TailLoop case
      if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
        incTotalPrev = 0
        # acculmulate total inc from 0 to iui-1
        for i in range(iui):
          incTotalPrev += self.localReadIncSub(kernel, i, tP, LdsPad) * bpe
        extraIncPrev = ((incTotalPrev) // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * bpe
        extraIncCurr = ((inc+incTotalPrev) // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * bpe
        inc += extraIncCurr - extraIncPrev
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(inc), "inc")
      kStr += inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          self.vcc, \
          sgpr(tmpSgpr), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "lr%s += %u%s"%(tP["tensorChar"], inc, comment) )
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        tP["localReadOffset"] += inc
        kStr += self.comment1("N/A, lro->%d" % tP["localReadOffset"])
        kStr += self.comment1("self.localReadDoCntA %d self.localReadDoCntB %d" % (self.localReadDoCntA,self.localReadDoCntB))
      else:
        inc = kernel["LocalSplitU"] * (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad)
        kStr += inst("_v_add_co_u32", \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            self.vcc, \
            hex(inc), \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            "lr%s += %u (LSU+(MT+Pad)*bpe"%(tP["tensorChar"], inc) )

    return kStr

  ##############################################################################
  # Local Read: Do It A/B
  # iui = Inner Unroll Idx
  # uIdx - Unroll Idx
  # epsi = expand pointer swap index. Only used for PAP
  ##############################################################################
  def localReadDo(self, kernel, bufferIdx, iui, epsi, tP):

    if not self.do["LocalRead%s" % tP["tensorChar"]]:
      imod = Code.Module("LocalReadDo%s_I%s" % (tP["tensorChar"], iui))
      pack = Code.Module("pack%s_I%s" % (tP["tensorChar"], iui))
      return imod, pack

    component = Component.LocalRead.find(self)
    if component:
      return component(self, bufferIdx, iui, epsi, tP)

  ##############################################################################
  # Save the local read pointers, for example when creating a duplicated
  # optimized path (like optNLL)
  ##############################################################################
  def saveLocalPointers(self, kernel):
    self.tPA["savedLocalReadOffset"] = self.tPA["localReadOffset"]
    self.tPB["savedLocalReadOffset"] = self.tPB["localReadOffset"]
    self.savedLocalReadDoCntA = self.localReadDoCntA
    self.savedLocalReadDoCntB = self.localReadDoCntB
    if kernel["ExpandPointerSwap"]:
      self.tPA["savedLocalWriteSwapByteOffset"] = self.tPA["localWriteSwapByteOffset"]
      self.tPB["savedLocalWriteSwapByteOffset"] = self.tPB["localWriteSwapByteOffset"]

  ##############################################################################
  # Restore the saved local read pointers
  # Must be paired with an earlier call to savePointers
  ##############################################################################
  def restoreLocalPointers(self, kernel):
    self.tPA["localReadOffset"] = self.tPA["savedLocalReadOffset"]
    self.tPB["localReadOffset"] = self.tPB["savedLocalReadOffset"]
    self.localReadDoCntA = self.savedLocalReadDoCntA
    self.localReadDoCntB = self.savedLocalReadDoCntB
    if kernel["ExpandPointerSwap"]:
      self.tPA["localWriteSwapByteOffset"] = self.tPA["savedLocalWriteSwapByteOffset"]
      self.tPB["localWriteSwapByteOffset"] = self.tPB["savedLocalWriteSwapByteOffset"]

  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    component = Component.ShiftVectorComponents.find(self)
    if component:
      return component(self, kernel, tP)

  ##############################################################################
  # Complex Declare Tmp Registers - SKIP
  ##############################################################################
  def complexDeclareTmpRegisters(self, kernel):
    kStr = ""
    return kStr

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    kStr = ""
    # wait for summation to be done with lds before writing reduction values
    # sync is already done before calling this function
    # kStr += self.syncThreads(kernel, "pre-lsu local write")

    if kernel["EnableMatrixInstruction"]:
      kStr += self.localSplitULocalWriteMFMA(kernel)
    else:
      kStr += self.localSplitULocalWriteNonMFMA(kernel)

    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all writes")
    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
    kStr += self.syncThreads(kernel, "post-lsu local write")
    #kStr += self.dumpLds(kernel, 0, 16)
    #kStr += self.bomb(5)
    return kStr

  ##############################################################################
  # LocalSplitU: Local Write for non-MFMA
  ##############################################################################
  def localSplitULocalWriteNonMFMA(self, kernel):
    kStr = ""

    lr0 = self.vgprPool.checkOut(1,"lr0")
    lr1 = self.vgprPool.checkOut(1,"lr1")
    sg = self.vgprPool.checkOut(1,"sg")
    copy = self.vgprPool.checkOut(1,"copy")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # lr0 = serial % SG0
    kStr += vectorStaticDivideAndRemainder(copy, lr0, "Serial", kernel["SubGroup0"], tmpSgpr)

    # lr1 = (serial / SG0) % SG1
    # sg  = (serial / SG0) / SG1
    kStr += vectorStaticDivideAndRemainder(sg, lr1, copy, kernel["SubGroup1"], tmpSgpr)

    # lr0 *= VW
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["VectorWidth"]*self.bpeCinternal), "VW")
    kStr += inst("v_mul_lo_u32", vgpr(lr0), sgpr(tmpSgpr), vgpr(lr0), \
        "lr0 *= VW")
    # lr1 *= VW*MT0
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["VectorWidth"]*kernel["MacroTile0"]*self.bpeCinternal), "VW*MT0")
    kStr += inst("v_mul_lo_u32", vgpr(lr1), sgpr(tmpSgpr), vgpr(lr1), \
        "lr1 *= VW*MT0")
    # sg  *= MT0*MT1
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["MacroTile0"]*kernel["MacroTile1"]*self.bpeCinternal), "MT0*MT1")
    kStr += inst("v_mul_lo_u32", vgpr(sg), sgpr(tmpSgpr), vgpr(sg), \
        "sg *= MT0*MT1")

    # thread offset
    addr = lr0
    kStr += inst("_v_add_co_u32", vgpr(addr), self.vcc, vgpr(lr1), vgpr(addr),  "")
    kStr += inst("_v_add_co_u32", vgpr(addr), self.vcc, vgpr(sg), vgpr(addr),  "threadOffset")
    self.vgprPool.checkIn(lr0)
    self.vgprPool.checkIn(lr1)
    self.vgprPool.checkIn(sg)
    self.vgprPool.checkIn(copy)

    # dump addr
    #kStr += dump(vgpr(addr))

    # do writes
    # LDS Layout example (for Sgemm, LSU=4, TT=8x8, WG=[8,4,4]), 128 WI/WG
    # VectorWidth = GlobalWriteVectorWidth = 4
    # SubGroup0 (WI:00-32)  : LDS 0x0000-
    # SubGroup1 (WI:33-64)  : LDS 0x2000-
    # SubGroup2 (WI:65-95)  : LDS 0x4000-
    # SubGroup3 (WI:96-127) : LDS 0x6000-

    # Interleave within a subgroup is interesting...
    #       Start LDS Addr
    # WI00 - 0x000
    # WI01 - 0x010
    # ...
    # WI07 - 0x070
    # WI08 - 0x400
    # WI09 - 0x410
    # ...
    # WI0F - 0x470
    # WI10 - 0x800
    # ...
    # ...
    # WI1f - 0xc70
    # WI20 - 0x1000  (start SubGroup1)

    # so a zoom-in on the pattern at beginning of LDS, for the case above:
    #   WI (hex) |x00-|x01-|...   |x07-|0x0-|0x1-|...|0x7-|0x0-| ... ... ||0x8-|
    # ValuC      |0123|0123|...   |0123|4567|4567|...|4567|89AB| ... ... ||0123
    #            |                     |                  |               |
    # LDS Addr  0x0                  0x80               0x100           0x400

    bytesPerElem = kernel["ProblemType"]["ComputeDataType"].numBytes()
    regsPerElem  = kernel["ProblemType"]["ComputeDataType"].numRegisters()
    bytesPerVector = kernel["VectorWidth"] * bytesPerElem
    bytesPerStep = min(bytesPerVector, 16) # max length of ds inst is 16 bytes(128bits)
    regsPerStep  = int((bytesPerStep+3)//4)
    elementStep = bytesPerStep // bytesPerElem

    for j in range(0, kernel["ThreadTile1"]//kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]//kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          for vc in range(0, kernel["VectorWidth"], elementStep):
            # for half, write 2 elements (4 bytes)
            # for single, write 1 element (4 bytes)
            # double doesn't work yet
            writeOffset = vc \
                + i*kernel["SubGroup0"]*kernel["VectorWidth"] \
                + s*kernel["MacroTile0"] \
                + j*kernel["MacroTile0"]*kernel["SubGroup1"]*kernel["VectorWidth"]
            regIdx = vc \
                + i*kernel["VectorWidth"] \
                + s*kernel["ThreadTile0"] \
                + j*kernel["ThreadTile0"]*kernel["VectorWidth"]
            regIdx = int(regIdx * regsPerElem)

            kStr += inst(f"_ds_store_b{bytesPerStep*8}", vgpr(addr), vgpr("ValuC+%u"%regIdx, regsPerStep), \
                "offset:%u"%(writeOffset*self.bpeCinternal), "j=%u i=%u s=%u vc=%u"%(j,i,s,vc))

    return kStr

  ##############################################################################
  # LocalSplitU: Local Write for MFMA
  ##############################################################################
  def localSplitULocalWriteMFMA(self, kernel):
    kStr = ""

    # MFMA + LSU case
    # Store intermediate sum to LDS with the same layout as non-MFMA case.
    # To do so, apply same calculation logic as global store for MFMA except for stride.
    # Stride for LSU will be MT0 * bpe

    # 1. move acc to vgpr
    kStr += self.MapAcctoArchRegs(kernel,option=0, isOptNLL=False)

    # 2. local write offset address
    # code from ComputeStoreVgprsMFMA(both Swap and non-Swap version)
    kStr += self.localSplitULocalWriteOffsetAddressMFMA(kernel)

    # 3. generate elements
    # code from notLocalSplitUGlobalWrite
    elements = [[] for y in range(1)] # 2D array for Full (Edge is unnecessary for LSU)
    (gwvw, elements) = self.notLocalFullTileElements(kernel, False)

    # 4. local write code
    # code from globalWriteElements and Batch
    kStr += self.lsuMFMAWriteElements(kernel, gwvw, elements)

    return kStr

  ##############################################################################
  # LocalSplitU: Local Write Offset Address for MFMA (Code from ComputeStoreVgprsMFMA(both Swap and non-Swap version))
  ##############################################################################
  def localSplitULocalWriteOffsetAddressMFMA(self, kernel):
    kStr = ""

    # alloc resources
    tid0 = self.vgprPool.checkOut(1)
    tid1 = self.vgprPool.checkOut(1)
    sg_id = self.vgprPool.checkOut(1)
    tmpSgpr  = self.getTmpSgpr(1).idx()

    # constant
    MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
    MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

    if kernel["SourceSwap"]:
      # SourceSwap case, use M
      MorN = "M"
    else:
      # non SourceSwap case, use N
      MorN = "N"
    matrixInstMorN = kernel["MatrixInst%s"%MorN] * kernel["MatrixInstB%s"%MorN] if (kernel["MatrixInst%s"%MorN] == 4) else kernel["MatrixInst%s"%MorN]

    # sg_id   = (serial//WaveFrontSize)//numWaveForSubGroup
    # wave_id = (serial//WaveFrontSize)%numWaveForSubGroup
    numWaveForSubGroup = (kernel["SubGroup0"]*kernel["SubGroup1"])//kernel["WavefrontSize"]
    if numWaveForSubGroup > 1:
      # numWaveForSubGroup==1 case, need both sg_id and wave_id
      wave_id = self.vgprPool.checkOut(1)
      tmpVgpr0 = self.vgprPool.checkOut(1,"tmpVgpr0")
      kStr += vectorStaticDivide(tmpVgpr0, "Serial", kernel["WavefrontSize"], tmpSgpr)
      kStr += vectorStaticDivideAndRemainder(sg_id, wave_id, tmpVgpr0, numWaveForSubGroup, tmpSgpr)
    else:
      # numWaveForSubGroup==1 case, wave_id is always 0. wave_id is nunecessary
      wave_id = None
      tmpVgpr0 = None
      kStr += vectorStaticDivide(sg_id, "Serial", kernel["WavefrontSize"] * numWaveForSubGroup, tmpSgpr)

    # coord 1 : thread part
    if kernel["SourceSwap"]:
      # SourceSwap case
      kStr += vectorStaticRemainder(tid1, "Serial", kernel["WavefrontSize"], tmpSgpr)
      kStr += vectorStaticDivide(tid1, tid1, matrixInstMorN, tmpSgpr)
      kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "thread0 * continuous_output")
      if kernel["MatrixInstBN"] > 1 and kernel["MatrixInstM"] == 4 and (kernel["MatrixInstN"] > kernel["MIOutputVectorWidth"]):
        # conversion for MI4x4 + MIBN>1
        # tid1 = (tid1/MIBN) + (tid1%MIBN)*(MIN//MIOVW)
        if tmpVgpr0==None:
          tmpVgpr0 = self.vgprPool.checkOut(1,"tmpVgpr0")
        kStr += vectorStaticDivideAndRemainder(tmpVgpr0, tid1, tid1, kernel["MatrixInstBN"], tmpSgpr) # using rReg=dReg (assuming MIBN is power of 2)
        kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["MatrixInstN"]//kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "(tid1%MIBN)*(MIN//MIOVW)")
        kStr += inst("_v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "tid1 = (tid1/MIBN) + (tid1%MIBN)*MIN")
    else:
      # non SourceSwap case
      kStr += vectorStaticRemainder(tid1, "Serial", matrixInstMorN, tmpSgpr)

    # coord 1 : wave part
    if kernel["MIWaveGroup"][1] > 1:
      kStr += vectorStaticDivide(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpSgpr)
      kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape1), vgpr(tmpVgpr0), "wave coordination offset 1")
      kStr += inst("_v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")

    # adjustment for B vector
    kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["VectorWidthB"], sgpr(tmpSgpr), "coordination 1 *= vwB")
    # tid1 *= MT0*bpe
    kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["MacroTile0"]*self.bpeCinternal, sgpr(tmpSgpr), "coordination 1 *= (MT0*bpe)")

    # coord 1 : LSU part
    sgOffset = kernel["MacroTile0"]*kernel["MacroTile1"]*self.bpeCinternal
    kStr += staticMultiply(vgpr(sg_id), vgpr(sg_id), sgOffset, sgpr(tmpSgpr), "sgOffset = sg_id * (MT0*MT1*bpe)")
    kStr += inst("_v_add_u32", vgpr(tid1), vgpr(sg_id), vgpr(tid1), "tid1 += sgOffset")

    # coord 0 : thread part
    if kernel["SourceSwap"]:
      # SourceSwap case
      kStr += vectorStaticRemainder(tid0, "Serial", matrixInstMorN, tmpSgpr)
    else:
      # non SourceSwap case
      kStr += vectorStaticRemainder(tid0, "Serial", kernel["WavefrontSize"], tmpSgpr)
      kStr += vectorStaticDivide(tid0, tid0, matrixInstMorN, tmpSgpr)
      kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "thread0 * continuous_output")
      if kernel["MatrixInstBM"] > 1 and kernel["MatrixInstN"] == 4 and (kernel["MatrixInstM"] > kernel["MIOutputVectorWidth"]):
        # conversion for MI4x4 + MIBM>1
        # tid0 = (tid0/MIBM) + (tid0%MIBM)*(MIM//MIOVW)
        if tmpVgpr0==None:
          tmpVgpr0 = self.vgprPool.checkOut(1,"tmpVgpr0")
        kStr += vectorStaticDivideAndRemainder(tmpVgpr0, tid0, tid0, kernel["MatrixInstBM"], tmpSgpr) # using rReg=dReg (assuming MIBM is power of 2)
        kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["MatrixInstM"]//kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "(tid1%MIBM)*(MIM//MIOVW)")
        kStr += inst("_v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "tid0 = (tid0/MIBM) + (tid0%MIBM)*MIM")

    # coord 0 : wave part
    if kernel["MIWaveGroup"][0] > 1:
      kStr += vectorStaticRemainder(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpSgpr)
      kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape0), vgpr(tmpVgpr0), "wave coordination offset 0")
      kStr += inst("_v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "tid0 += wave coordination offset 0")

    # tid0 *= VWA*bpe
    kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["VectorWidthA"]*self.bpeCinternal, sgpr(tmpSgpr), "coordination 0 *= (VWA*bpe)")

    # tid = tid0 + tid1
    kStr += inst("_v_add_u32", vgpr(tid0), vgpr(tid1), vgpr(tid0), "tid = tid0 + tid1")

    # release resource
    if tmpVgpr0 != None:
      self.vgprPool.checkIn(tmpVgpr0)
    if wave_id != None:
      self.vgprPool.checkIn(wave_id)
    self.vgprPool.checkIn(sg_id)
    self.vgprPool.checkIn(tid1)

    # Store tid to self.coord0 for future use. Release vreg for tid0 later
    self.coord0 = tid0

    return kStr

  ##############################################################################
  # LocalSplitU for MFMA Write Elements (code from globalWriteElements and Batch)
  ##############################################################################
  def lsuMFMAWriteElements(self, kernel, gwvw, elements):
    kStr = ""

    codeAccVgprRead = self.codeAccVgprRead
    bpr = self.bpr
    numVgprPerValuC = self.bpeCinternal//bpr

    matrixInstM  = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
    matrixInstN  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
    matrixInstBM = 1                                                if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
    matrixInstBN = 1                                                if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

    for elementIdx in range(len(elements)):
      # Step 1: calculate coordOffset1, coordOffset0, sumIdx (code from setupStoreElementsForBatch)
      element = elements[elementIdx]
      (d1,d0,vc1,vc0) = element

      coordOffset1 = 0

      MIOutputVectorWidth = kernel["MIOutputVectorWidth"]
      MFMAContinuousOutputs = MIOutputVectorWidth if kernel["SourceSwap"] else 1
      OutputsPerMIMN        = (matrixInstM * matrixInstN // kernel["WavefrontSize"]) if kernel["SourceSwap"] else 1
      WavefrontSize = kernel["WavefrontSize"]

      eIdx1        = d1 % (OutputsPerMIMN // MFMAContinuousOutputs)
      remain_d1    = d1 // (OutputsPerMIMN // MFMAContinuousOutputs)

      bIdx1     = remain_d1 % matrixInstBN
      remain_d1 = remain_d1 // matrixInstBN
      wtIdex    = remain_d1 % kernel["MIWaveTile"][1]

      coordOffset1  = eIdx1 * (WavefrontSize // matrixInstN) * MFMAContinuousOutputs
      coordOffset1 += bIdx1 * matrixInstN
      coordOffset1 += wtIdex * matrixInstN *  matrixInstBN * kernel["MIWaveGroup"][1]
      coordOffset1  = coordOffset1 * kernel["VectorWidthB"]  + vc1

      MFMAContinuousOutputs = 1 if kernel["SourceSwap"] else MIOutputVectorWidth
      OutputsPerMIMN        = 1 if kernel["SourceSwap"] else matrixInstM * matrixInstN // WavefrontSize

      eIdx0        = d0 % (OutputsPerMIMN // MFMAContinuousOutputs)
      remain_d0    = d0 // (OutputsPerMIMN // MFMAContinuousOutputs)

      bIdx0        = remain_d0 % matrixInstBM
      remain_d0    = remain_d0 // matrixInstBM
      wtIdex       = remain_d0 % kernel["MIWaveTile"][0]

      coordOffset0  = eIdx0  * (WavefrontSize // matrixInstM) * MFMAContinuousOutputs
      coordOffset0 += bIdx0  * matrixInstM
      coordOffset0 += wtIdex * matrixInstM * matrixInstBM * kernel["MIWaveGroup"][0]
      coordOffset0  = coordOffset0 * kernel["VectorWidthA"] + vc0

      # Step 2: generate move code from acc to vgpr (no move necessary for MIArchVgpr case (some exceptions))
      # loop over scalars within one store instruction
      sumIdx = None
      noMove = self.srcIdxList != None
      if not noMove:
        regsPerScalar = self.bpeCinternal//bpr # register per scalar
        alignment = numVgprPerValuC * gwvw
        sumIdx    = self.vgprPool.checkOutAligned(alignment, alignment, "vgprValuC")
        for vi in range(0, gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            tempStr = str(codeAccVgprRead.items().pop(0))
            kStr += tempStr.replace("__placeholder__", str(sumIdx + regsPerScalar*vi + rIdx))
        if not kernel["MIArchVgpr"]:
          kStr += inst("s_nop 1", "2 wait states required before reading vgpr")

      # Step 3: generate local store code
      bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw
      regsPerStep = bps // bpr
      writeOffset = coordOffset0 + coordOffset1 * kernel["MacroTile0"]
      if noMove:
        # no move case, directly use mfma dst vregs (stored in srcIdxList)
        regIdx = self.srcIdxList[elementIdx*regsPerStep]
      else:
        regIdx = sumIdx
      kStr += inst(f"_ds_store_b{bps*8}", vgpr(self.coord0), vgpr("ValuC+%u"%regIdx, regsPerStep), \
          "offset:%u"%(writeOffset*self.bpeCinternal), "")

      # release resource for sumIdx
      if sumIdx != None:
        self.vgprPool.checkIn(sumIdx)

    # release resource for tid
    self.vgprPool.checkIn(self.coord0)

    return kStr

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    # alloc resource
    tmpSgpr  = self.getTmpSgpr(1).idx()
    baseAddr = self.vgprPool.checkOut(1,"baseAddr")

    # calculate parameters
    bytesPerElem = kernel["ProblemType"]["ComputeDataType"].numBytes()
    regsPerElem  = kernel["ProblemType"]["ComputeDataType"].numRegisters()
    bytesPerVector = kernel["GlobalWriteVectorWidth"] * bytesPerElem
    bytesPerStep = 16
    while (bytesPerVector % bytesPerStep) != 0:
      bytesPerStep //= 2
    regsPerStep  = int((bytesPerStep+3)//4)
    elementStep = bytesPerStep // bytesPerElem

    # generate source
    kStr = ""
    kStr += staticMultiply(vgpr(baseAddr), vgpr("Serial"), bytesPerVector, sgpr(tmpSgpr))
    # Load values for each subgroup
    for r in range(0, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
          offset = s + i*kernel["NumThreads"]*kernel["GlobalWriteVectorWidth"] + r * kernel["MacroTile0"]*kernel["MacroTile1"]
          regIdx = int((s + i*kernel["GlobalWriteVectorWidth"] + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]) * regsPerElem)
          kStr += inst(f"_ds_load_b{bytesPerStep*8}", vgpr("ValuC+%u"%regIdx,regsPerStep), vgpr(baseAddr), \
              "offset:%u"%(offset*self.bpeCinternal), "r=%u i=%u s=%u"%(r,i,s))
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all reads")

    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

    # free resources
    self.vgprPool.checkIn(baseAddr)

    return kStr

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""

    is_fp16 = kernel["ProblemType"]["ComputeDataType"].isHalf()
    elementStep = 2 if is_fp16 else 1
    regsPerElem = kernel["ProblemType"]["ComputeDataType"].numRegisters()

    for r in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
          cIdx = int((s + i * kernel["GlobalWriteVectorWidth"]) * regsPerElem)
          regIdx = int((s + i * kernel["GlobalWriteVectorWidth"] + r * kernel["GlobalWriteVectorWidth"] * kernel["NumGlobalWriteVectorsPerThread"]) * regsPerElem)

          if is_fp16:
            kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%cIdx), vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), \
                         "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["ComputeDataType"].isInt32():
            kStr += inst("_v_add_i32", vgpr("ValuC+%u"%cIdx), vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), \
                         "c[%u] += c[%u]"%(cIdx, regIdx))
          elif kernel["ProblemType"]["ComputeDataType"].isSingle():
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%cIdx), vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), \
                         "c[%u] += c[%u]"%(cIdx, regIdx))
          elif kernel["ProblemType"]["ComputeDataType"].isDouble():
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%cIdx,2), vgpr("ValuC+%u" % regIdx,2), vgpr("ValuC+%u"%cIdx,2), \
                         "c[%u] += c[%u]"%(cIdx, regIdx))
          elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%(cIdx+0)), vgpr("ValuC+%u" % (regIdx+0)), vgpr("ValuC+%u"%(cIdx+0)), \
                         "c[%u] += c[%u], real part"%(cIdx, regIdx) )
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%(cIdx+1)), vgpr("ValuC+%u" % (regIdx+1)), vgpr("ValuC+%u"%(cIdx+1)), \
                         "c[%u] += c[%u], imaginary part"%(cIdx+1, regIdx+1) )
          elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%(cIdx+0),2), vgpr("ValuC+%u" % (regIdx+0),2), vgpr("ValuC+%u"%(cIdx+0),2), \
                         "c[%u] += c[%u], real part"%(cIdx, regIdx) )
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%(cIdx+2),2), vgpr("ValuC+%u" % (regIdx+2),2), vgpr("ValuC+%u"%(cIdx+2),2), \
                         "c[%u] += c[%u], imaginary part"%(cIdx+2, regIdx+2) )
          else:
            # TODO: int8
            assert(0) # unsupported data type, need to modify here and LSU write/read code
    return kStr

  ##############################################################################
  # computeStoreSrd
  # Add tile assignment fields to store srd
  # This is based on WG not the WI/TT assignment
  ##############################################################################
  def computeStoreSrdStart(self, kernel):
    kStr = ""

    tmpS0ref = self.getTmpSgpr(3)
    tmpS0 = tmpS0ref.idx()
    tmpS1 = tmpS0+1
    wgMT1 = tmpS0+2

    # Compute and save wg1*MT1 - the element offset that is top of the macro-tile in output space
    assert kernel["BufferStore"]
    kStr += "\n"
    kStr += inst("s_mul_i32", \
        sgpr(wgMT1), \
        "MT1", \
        sgpr("WorkGroup1"), \
        "<- wg1*MT1")

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
    # For dynamic (based on TT assignment) - save in coutRowPtr in computeStoreVgprs,
    # which saves the TT assignment for each WI scaled by StrideC0
    # TODO - future opportunities for store vgpr and other optimization
    #  - coutRowPtr and tid1 are strongly related - can we merge or remove one of these?
    # Packed follows same philosophy but may have more vector components
    indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
    numDim = len(indices)
    addrSrcSgpr = self.sgprAddressStrCD # use "Address" only for the first iteration
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

      if addToSrd:
        # These are constant across all workitems, just add to the SRD:
        strideC = "StrideC%s"%self.indexChars[i]
        kStr += self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(strideC), "CScale %s by Stride"%coord)
        kStr += inst("s_lshl_b64", sgpr(tmpS0,2), sgpr(tmpS0,2), log2(self.bpeCexternal), "scale by bpe")

        kStr += inst("s_add_u32",  sgpr("SrdC+0"), sgpr("%sC+0"%addrSrcSgpr), sgpr(tmpS0), "add lo to SRD")
        kStr += inst("s_addc_u32", sgpr("SrdC+1"), sgpr("%sC+1"%addrSrcSgpr), sgpr(tmpS1), "add hi to SRD")

        # These are constant across all workitems, just add to the SRD:
        stride = "StrideD%s" % (self.indexChars[i])
        kStr += self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(stride), "Scale %s by Stride"%coord)
        kStr += inst("s_lshl_b64", sgpr(tmpS0,2), sgpr(tmpS0,2), log2(self.bpeCexternal), "scale by bpe")

        kStr += inst("s_add_u32",  sgpr("SrdD+0"), sgpr("%sD+0"%addrSrcSgpr), sgpr(tmpS0), "add lo to SRD")
        kStr += inst("s_addc_u32", sgpr("SrdD+1"), sgpr("%sD+1"%addrSrcSgpr), sgpr(tmpS1), "add hi to SRD")

        kStr += "\n"

        addrSrcSgpr = "Srd" # update src Sgpr for the second or later iterations

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      # GSU algorithm 2: adjust output buffer address to per GSU buffer
      tmpFree0Ref = self.getTmpSgpr(2)
      tmpFree0 = tmpFree0Ref.idx()
      tmpFreeNRef = self.getTmpSgpr(2)
      tmpFreeN = tmpFreeNRef.idx()
      tmpSgprRef = self.getTmpSgpr(1)
      tmpSgpr = tmpSgprRef.idx()
      kStr += "// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s" % self.endLine
      kStr += self.s_mul_u64_u32(sgpr(tmpFree0+0), sgpr(tmpFree0+1), sgpr("SizesFree+0"), sgpr("GSUSumIdx"), "Free0")
      for i in range(1, numDim):
        kStr += inst("s_sub_u32",  sgpr(tmpSgpr), sgpr("SizesFree+%u"%i), 1, "Free%u" % i)
        kStr += inst("s_mul_i32",  sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr("GSUSumIdx"), "Free%u" % i)
        kStr += self.s_mul_u64_u32(sgpr(tmpFreeN+0), sgpr(tmpFreeN+1), sgpr(tmpSgpr), sgpr("StrideC%s"%self.indexChars[i]), "Free%u" % i)
        kStr += inst("s_add_u32",  sgpr(tmpFree0+0), sgpr(tmpFree0+0), sgpr(tmpFreeN+0), "Free%u" % i)
        kStr += inst("s_addc_u32", sgpr(tmpFree0+1), sgpr(tmpFree0+1), sgpr(tmpFreeN+1), "Free%u" % i)
      kStr += inst("s_lshl_b64", sgpr(tmpFree0+0,2), sgpr(tmpFree0+0,2), log2(self.bpeCexternal), "scale by bpe")
      kStr += inst("s_add_u32",  sgpr("SrdD+0"), sgpr("SrdD+0"), sgpr(tmpFree0+0), "add lo GSU offset to SRD")
      kStr += inst("s_addc_u32", sgpr("SrdD+1"), sgpr("SrdD+1"), sgpr(tmpFree0+1), "add hi GSU offset to SRD")

    for cdir in (0,1):
      indices = kernel["PackedC%uIndicesX"%cdir]
      packedSizes = "PackedSize%u"%cdir
      if len(indices) > 1:
        for i,idx in enumerate(indices[1:]):
          if i==0:
            kStr += inst("s_mul_i32", sgpr(packedSizes), self.sizeRef(indices[0]), \
                      self.sizeRef(idx), "first packed size")
          else:
            kStr += inst("s_mul_i32", sgpr(packedSizes), sgpr(packedSizes), \
                      self.sizeRef (idx), "first packed size")

    return kStr

  ##############################################################################
  # computeStoreVgprs
  # Compute workitem/TT offsets in VGPRS
  # and coord0/coord1
  # tid0Scale specifies the number of output elements in 0/coalesced dim
  # that should be written by each work-item in each batch element.
  ##############################################################################
  def computeStoreVgprs(self, kernel, divisor, tid0Scale, tid1Scale):
    kStr = ""
    kStr += self.comment1("computeStoreVgprs")
    component = Component.ComputeStoreVgprs.find(self)
    if component:
      kStr += component(self, kernel, divisor, tid0Scale, tid1Scale)
    return kStr

  ##############################################################################
  # globalWriteWorkGroupInitBeforePersistentLoop:
  ##############################################################################
  def globalWriteWorkGroupInitBeforePersistentLoop(self, kernel):
    kStr = ""
    if kernel["BufferStore"]:
      kStr += self.allocPostLoopSrd(kernel, "D")
      kStr += self.allocPostLoopSrd(kernel, "C")
    return kStr

  ##############################################################################
  # globalWriteWorkGroupInit:
  ##############################################################################
  def globalWriteWorkGroupInit(self, kernel):
    kStr = ""
    if kernel["BufferStore"]:
      # do allocPostLoopSrd separately (before persistent loop) in self.prefetchAcrossPersistent case
      if not self.prefetchAcrossPersistent:
        kStr += self.allocPostLoopSrd(kernel, "D")
        kStr += self.allocPostLoopSrd(kernel, "C")
      kStr += self.computeStoreSrdStart(kernel)
      # release CD for PKA case
      if self.releaseSgprAdressCDlater:
        kStr += self.undefineSgpr("AddressD")
        kStr += self.undefineSgpr("AddressC")
        self.releaseSgprAdressCDlater = False
    return kStr

  ##############################################################################
  # Global Write Indices common
  ##############################################################################
  def globalWriteIndicesCommon(self, kernel):
    kStr = ""

    if kernel["BufferStore"]:
      #print "----AddressC-LocalSplitU"
      #print self.vgprPool.state()
      self.addrD = -1
      self.addrC = -1
    else:
      self.addrD = self.vgprPool.checkOut(2)
      kStr += inst("v_mov_b32", \
          vgpr(self.addrD+0), \
          sgpr("AddressD+0"), \
          "sgpr -> vgpr")
      kStr += inst("v_mov_b32", \
          vgpr(self.addrD+1), \
          sgpr("AddressD+1"), \
          "sgpr -> vgpr")
      if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
        # (code from computeStoreSrdStart)
        # GSU algorithm 2: adjust output buffer address to per GSU buffer
        indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
        numDim = len(indices)
        tmpSgpr = self.getTmpSgpr(5).idx()
        kStr += "// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s" % self.endLine
        kStr += self.s_mul_u64_u32(sgpr(tmpSgpr+0), sgpr(tmpSgpr+1), sgpr("SizesFree+0"), sgpr("GSUSumIdx"), "Free0")
        for i in range(1, numDim):
          kStr += inst("s_sub_u32",  sgpr(tmpSgpr+4), sgpr("SizesFree+%u"%i), 1, "Free%u" % i)
          kStr += inst("s_mul_i32",  sgpr(tmpSgpr+4), sgpr(tmpSgpr+4), sgpr("GSUSumIdx"), "Free%u" % i)
          kStr += self.s_mul_u64_u32(sgpr(tmpSgpr+2), sgpr(tmpSgpr+3), sgpr(tmpSgpr+4), sgpr("StrideC%s"%self.indexChars[i]), "Free%u" % i)
          kStr += inst("s_add_u32",  sgpr(tmpSgpr+0), sgpr(tmpSgpr+0), sgpr(tmpSgpr+2), "Free%u" % i)
          kStr += inst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), sgpr(tmpSgpr+3), "Free%u" % i)
        kStr += inst("s_lshl_b64", sgpr(tmpSgpr+0,2), sgpr(tmpSgpr+0,2), log2(self.bpeCexternal), "scale by bpe")
        tmpVgpr = self.vgprPool.checkOut(2)
        kStr += inst("v_mov_b32", vgpr(tmpVgpr+0), sgpr(tmpSgpr+0), "sgpr -> vgpr")
        kStr += inst("v_mov_b32", vgpr(tmpVgpr+1), sgpr(tmpSgpr+1), "sgpr -> vgpr")
        kStr += inst("_v_add_co_u32",  vgpr(self.addrD+0), self.vcc, vgpr(tmpVgpr+0), vgpr(self.addrD+0), "add lo GSU offset to SRD")
        kStr += inst("_v_addc_co_u32", vgpr(self.addrD+1), self.vcc, vgpr(tmpVgpr+1), vgpr(self.addrD+1), self.vcc, "add hi GSU offset to SRD")
        self.vgprPool.checkIn(tmpVgpr)
        # addrC is not used for MultipleBuffer
        self.addrC = -1
      else:
        self.addrC = self.vgprPool.checkOut(2)
        kStr += inst("v_mov_b32", \
            vgpr(self.addrC+0), \
            sgpr("AddressC+0"), \
            "sgpr -> vgpr")
        kStr += inst("v_mov_b32", \
            vgpr(self.addrC+1), \
            sgpr("AddressC+1"), \
            "sgpr -> vgpr")

    return kStr

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""

    # lr0 = serial % SG0
    kStr += self.computeStoreVgprs(kernel, \
              divisor = kernel["MacroTile0"] // kernel["GlobalWriteVectorWidth"], \
              tid0Scale=kernel["GlobalWriteVectorWidth"], \
              tid1Scale=1)

    kStr += self.globalWriteIndicesCommon(kernel)

    return kStr

  ##############################################################################
  def allocPostLoopSrd(self, kernel, tc):
    kStr = ""
    sgprAddress0 = sgpr("Address%s+0"%tc)
    sgprAddress1 = sgpr("Address%s+1"%tc)
    # Buffer-load uses one base read pointer stored in the SRD - set it here:
    if not (self.releaseSgprAdressCD or self.releaseSgprAdressCDlater):
      kStr += inst("s_mov_b32", sgpr("Srd%s+0"%tc), sgprAddress0, "init SRD base address (lower)" )
      kStr += inst("s_mov_b32", sgpr("Srd%s+1"%tc), sgprAddress1, "init SRD base address (upper) + other fields" )
    kStr += inst("s_mov_b32", sgpr("Srd%s+2"%tc), "BufferOOB", "")
    kStr += inst("s_mov_b32", sgpr("Srd%s+3"%tc), "Srd127_96", "Set bits 127_96 in post-loop SRD")
    kStr += "\n"
    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not kernel["StoreCInUnroll"]:
      if not self.do["PostLoop"]: return ""
    kStr = ""

    kStr += self.computeStoreVgprs(kernel,
              divisor = kernel["SubGroup0"],\
              tid0Scale=kernel["VectorWidth"], \
              tid1Scale=kernel["VectorWidth"])

    kStr += self.globalWriteIndicesCommon(kernel)

    return kStr

  ##############################################################################
  # Release any resources used by the global write
  def cleanupGlobalWrite(self, kernel):
    self.vgprPool.checkIn(self.coord0)
    self.vgprPool.checkIn(self.coord1)

    if kernel["StoreRemapVectorWidth"]:
      self.vgprPool.checkIn(self.storeRemapLW)
      self.vgprPool.checkIn(self.storeRemapLR)
      self.vgprPool.checkIn(self.storeRemapCoord0)
      self.vgprPool.checkIn(self.storeRemapCoord1)
      self.vgprPool.checkIn(self.storeRemapOffsetCoord1)
    if kernel["BufferStore"]:
      self.vgprPool.checkIn(self.cinRowPtr)
      self.vgprPool.checkIn(self.coutRowPtr)
    if self.addrD >= 0:
      self.vgprPool.checkIn(self.addrD)
    if self.addrC >= 0:
      self.vgprPool.checkIn(self.addrC)

    if self.betaVgpr != None:
      self.vgprPool.checkIn(self.betaVgpr)

  ##############################################################################
  # Return max global write vector width, in elements
  def maxGwvw(self, kernel):
    #atomic = (kernel["GlobalSplitU"] > 1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')
    #
    #if kernel["BufferStore"]:
    #  if atomic:
    #    return kernel["VectorAtomicWidth"]
    #  else:
    #    return 1000  # no limit
    #else:
    #  if atomic:
    #    return 1  # flat vector atomic is not tested
    #  else:
    #    return 1000  # no limit
    return 1000  # no limit

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

    kStr = ""

    bps = self.bpeCexternal * ss.cfg.gwvw
    rpv = self.bpeCexternal * ss.cfg.gwvw / self.bpr

    addr0 = vgpr(self.storeRemapLW)
    offset =  addrCalc.coordOffset0 * self.bpeCexternal

    if bps==1:
      kStr += inst("ds_write_b8", addr0, vgpr(srcVgpr, rpv*4), \
                 "offset:%u"%offset, "storeRemap lw")
    elif bps==2:
      kStr += inst("_ds_store_b16", addr0, vgpr(srcVgpr, rpv*2), \
                 "offset:%u"%offset, "storeRemap lw")
    elif bps==4:
      kStr += inst("_ds_store_b32", addr0, vgpr(srcVgpr, rpv), \
                 "offset:%u"%offset, "storeRemap lw")
    elif bps==8:
      kStr += inst("_ds_store_b64", addr0, vgpr(srcVgpr, rpv), \
                 "offset:%u"%offset, "storeRemap lw")
    elif bps==16:
      kStr += inst("_ds_store_b128", addr0, vgpr(srcVgpr, rpv), \
                 "offset:%u"%offset, "storeRemap lw")
    else:
      assert 0, "StoreRemap: bad bps!"

    return kStr

  ##############################################################################
  # Store Remap: Local Read and Global Write
  ##############################################################################
  def storeRemapAddStore(self, kernel, ss, addrCalc, tmpVgpr, tmpS01, edge):
    kStr = ""

    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for LDS write" )

    numStoreInst = 0

    #Data exchange between different waves
    #Make sure LDS writes are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      kStr += self.indent + self.syncStr + " //wait all lds write finished" + self.endLine
    kStr += "\n"

    gwvw = kernel["StoreRemapVectorWidth"]
    nElements = kernel["MacroTile0"]*kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]//self.kernel["WavefrontSize"]

    bpe = self.bpeCexternal
    bps = bpe * gwvw
    rpe = self.bpeCexternal / self.bpr
    rpv = rpe * gwvw

    # num registers to check out
    storeRegs = []
    for i in range(0, nElements, gwvw):
      storeRegs.append(self.vgprPool.checkOutAligned(int(rpv), int(rpv), "store element d"))
    src = vgpr(self.storeRemapLR)
    for rIdx, i in enumerate(range(0, nElements, gwvw)):
      offset = self.storeRemapLrOffset * bpe * (i//gwvw)
      dst = vgpr(storeRegs[rIdx], rpv)
      if bps==4:
        kStr += inst("_ds_load_b32", dst, src, "offset:%u"%offset, "storeRemap lr")
      elif bps==8:
        kStr += inst("_ds_load_b64", dst, src, "offset:%u"%offset, "storeRemap lr")
      elif bps==16:
        kStr += inst("_ds_load_b128", dst, src, "offset:%u"%offset, "storeRemap lr")
      else:
        assert 0, "StoreRemap: bad bps!"

    kStr += "\n"

    # Global Write
    ntStr = ""
    if kernel.enabledSetPrioSplitLDS:
      kStr += inst("s_setprio", "1", "")
    if (kernel["NonTemporalD"] & 1) != 0 or kernel["ForceStoreSC1"]:
      ntStr += " " + getGlcBitName(kernel["MemoryModifierFormat"])
    if (kernel["NonTemporalD"] & 2) != 0 or kernel["ForceStoreSC1"]:
      ntStr += " " + getSlcBitName(kernel["MemoryModifierFormat"])
    if (kernel["NonTemporalD"] & 4) != 0:
      ntStr += " nt"

    addr1 = sgpr("SrdD", 4)
    packedD1 = kernel["PackedC1IndicesX"]
    strideD1 = "StrideD%s" % (self.indexChars[packedD1[0]])

    vTmp = self.vgprPool.checkOut(1, "SR Store temp addr0")
    addr0 = vgpr(vTmp)

    if not edge:
      for rIdx, i in enumerate(range(0, nElements, gwvw)):
        if i == 0:
          kStr += inst("v_mov_b32", addr0, vgpr(self.storeRemapOffsetCoord1), "coord1")
        else:
          currentStep = i//gwvw
          kStr += inst("_v_add_u32", addr0, vgpr(self.storeRemapOffsetCoord1), self.storeRemapNCPL * currentStep , "coord1 += nColPerLoad")

        kStr += inst("v_mul_lo_u32", addr0, addr0, sgpr(strideD1), "coord1 offset =  coord1 * StrideD")
        kStr += inst("_v_add_lshl_u32", addr0, addr0,  vgpr(self.storeRemapCoord0), hex(log2(bpe)), "global write D address")

        lgkmcnt = min((nElements-i)//gwvw - 1, 15)
        kStr += inst("s_waitcnt", "lgkmcnt(%u)"% lgkmcnt, "wait for LDS read" )

        numStoreInst += 1
        kStr += self.chooseGlobalWrite(True, bps, storeRegs[rIdx], rpv, addr0, addr1, 0, ntStr)
    else:
      tmpS23 = tmpS01+self.laneSGPRCount
      coord0 = tmpVgpr
      coord1 = coord0+1
      lrVw = kernel["StoreRemapVectorWidth"]
      edgeVw = min(kernel["AssertFree0ElementMultiple"],kernel["StoreRemapVectorWidth"])
      bps = self.bpeCexternal * edgeVw
      rpv = self.bpeCexternal / self.bpr * edgeVw

      ## extra vgpr for global write with StoreRemap when bps == 1
      vGWTmp = 0
      if bps == 1:
        vGWTmp = self.vgprPool.checkOut(1, "SR Store temp to shift byte")

      for rIdx, i in enumerate(range(0, nElements, lrVw)):
        for vi in range (0, lrVw, edgeVw):

          if vi == 0:
            lgkmcnt = min((nElements-i)//lrVw - 1, 15)
            kStr += inst("s_waitcnt", "lgkmcnt(%u)"% lgkmcnt, "wait for LDS read" )

          sizeBoundary = [0,0]
          sizeBoundary[0] = \
              sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index0"])
          sizeBoundary[1] = \
              sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index1"])

          currentStep = i//lrVw

          # calculate global coordination
          kStr += inst("_v_add_u32", vgpr(coord1), vgpr(self.storeRemapCoord1), self.storeRemapNCPL * currentStep , "coord1 += nColPerLoad")
          kStr += inst("_v_add_u32",vgpr(coord0), vgpr(self.storeRemapCoord0), vi , "coord0 += element index of load vector")
          kStr += inst("_v_add_u32", addr0, vgpr(self.storeRemapOffsetCoord1), self.storeRemapNCPL * currentStep , \
                        "offset coord1 += nColPerLoad")

          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS01,self.laneSGPRCount), vgpr(coord0), sizeBoundary[0], "coord0 < size0" )
          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS23,self.laneSGPRCount), vgpr(coord1), sizeBoundary[1], "coord1 < size1" )
          kStr += inst("s_and_b{}".format(self.kernel["WavefrontSize"]),
                       sgpr(tmpS23,self.laneSGPRCount),
                       sgpr(tmpS01,self.laneSGPRCount),
                       sgpr(tmpS23,self.laneSGPRCount), "in0 && in1" )

          kStr += inst("v_mul_lo_u32", addr0, addr0, sgpr(strideD1), "coord1 element offset =  coord1 * StrideD")
          kStr += inst("_v_add_lshl_u32", addr0, addr0,  vgpr(coord0), hex(log2(bpe)), "scale to BPE")
          kStr += inst("v_cndmask_b32", addr0, -1, addr0, sgpr(tmpS23,self.laneSGPRCount), "clip if OOB. offset" )

          sumIdx = storeRegs[rIdx] + int(vi*rpe)
          numStoreInst += 1
          if bps == 1:
            kStr += self.chooseGlobalWrite(True, bpe, sumIdx, rpe, addr0, addr1, 0, ntStr, hi16=vi%4, vb1Tmp=vGWTmp) ### need to check??
          elif bps == 2:
            kStr += self.chooseGlobalWrite(True, bpe, sumIdx, rpe, addr0, addr1, 0, ntStr, hi16=vi%2)
          else:
            kStr += self.chooseGlobalWrite(True, bps, sumIdx, rpv, addr0, addr1, 0, ntStr)

      if bps == 1:
        self.vgprPool.checkIn(vGWTmp)

    kStr += "\n"

    self.vgprPool.checkIn(vTmp)
    for v in storeRegs:
      self.vgprPool.checkIn(v)

    #Data exchange between different waves
    #Make sure LDS reads are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      kStr += self.indent + self.syncStr + " //wait all lds read finished" + self.endLine

    return kStr, numStoreInst

  ##############################################################################
  # Store remap compute vgprs:
  ##############################################################################
  def storeRemapComputeStoreVgprs(self, kernel):
    kStr = ""
    kStr += self.comment1("Store Remap Local Write address")

    tmpS0 = self.getTmpSgpr(2).idx()
    wgMT1 = tmpS0+1

    if self.prefetchAcrossPersistent:
      wg0="PrevWorkGroup0"
      wg1="PrevWorkGroup1"
    else:
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
    kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], \
      tmpS0)

    kStr += inst("v_mul_lo_u32", vgpr(waveCoord1),
                  hex(kernel["MatrixInstN"]), vgpr(tid1), "coord1 offset of LDS for each Wave")
    kStr += inst("v_and_b32", vgpr(tid1),
                  hex(kernel["MatrixInstN"]-1), vgpr("Serial"), "coord1 offset of LDS for each thread")
    kStr += inst("_v_add_u32", vgpr(tid1), vgpr(waveCoord1),vgpr(tid1),"coord1 offset in MacroTile")
    kStr += inst("v_mov_b32", vgpr(ldsStride), hex(kernel["MacroTile0"]+ldsPad), \
                    "lds stride = MT0 + PAD")
    kStr += inst("v_mul_lo_u32", vgpr(tmpV0), vgpr(tid1), vgpr(ldsStride), \
                  "lds coord1 offset = Col-id* lds stride")

    kStr += vectorStaticDivideAndRemainder(waveCoord0, tid0, tid0, self.kernel["WavefrontSize"], tmpS0)
    kStr += inst("v_lshrrev_b32", vgpr(coord0),
                hex(log2(kernel["MatrixInstN"])), vgpr(tid0), \
                "tid / matrixInstN")

    if kernel["MIOutputVectorWidth"] > 1:
      kStr += inst("v_lshlrev_b32", vgpr(coord0), hex(log2(kernel["MIOutputVectorWidth"])), vgpr(coord0), \
                    "lds coord0 offset *= 4 (each thread hold 4 element)")

    kStr += inst("v_mad_u32_u24", vgpr(coord0), kernel["MatrixInstM"]*kernel["MatrixInstBM"], vgpr(waveCoord0), vgpr(coord0), \
                  "coord0 += waveCoord0 * wave M shape(blockM*MiM)")

    kStr += inst("_v_add_lshl_u32", \
      vgpr(storeRemapLW), \
      vgpr(tmpV0), \
      vgpr(coord0), \
      hex(log2(self.bpeCexternal)), \
      "local write C address")

    kStr += "\n"
    # calculate local read address : v[vgprLocalReadAddrC]

    kStr += self.comment1("Store Remap Local Read address")

    kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.kernel["WavefrontSize"], tmpS0)
    kStr += inst("v_mul_lo_u32", vgpr(waveCoord1),
                  hex(kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]), vgpr(tid1), "coord1 offset of LDS for each Wave")

    nThreadPerCol = kernel["MacroTile0"] // gwvw
    nColPerLoad = self.kernel["WavefrontSize"] // nThreadPerCol
    self.storeRemapLrOffset = (kernel["MacroTile0"]+ldsPad) * nColPerLoad
    self.storeRemapNCPL = nColPerLoad

    kStr += inst("v_lshrrev_b32", vgpr(tmpV1),\
                hex(log2(nThreadPerCol)), vgpr(tid0), \
                "tid / nThreadPerCol")
    kStr += inst("_v_add_u32", vgpr(coord1Offset), vgpr(waveCoord1),vgpr(tmpV1),"coord1 offset in MacroTile")
    kStr += inst("v_mul_lo_u32", vgpr(tmpV0), vgpr(coord1Offset), vgpr(ldsStride), \
                  "lds coord1 offset = Col-id* lds stride")

    kStr += inst("v_and_b32", vgpr(coord0),
                  hex(nThreadPerCol-1), vgpr(tid0), "coord0 offset of LDS for each thread")
    kStr += inst("v_lshlrev_b32", vgpr(coord0), hex(log2(gwvw)), vgpr(coord0), \
                  "lds coord0 offset *= gwvw (each thread hold gwvw element)")

    kStr += inst("_v_add_lshl_u32", \
      vgpr(storeRemapLR), \
      vgpr(tmpV0), \
      vgpr(coord0), \
      hex(log2(self.bpeCexternal)), \
      "local read C address")
    kStr += "\n"

    # calculate global write coord0 and coord1
    kStr += self.comment1("Store Remap global write coord0 and coord1")
    kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], tmpS0)

    ColsPerBlockShape = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

    kStr += inst("v_mul_lo_u32", vgpr(waveCoord1),
                  hex(ColsPerBlockShape), vgpr(tid1), "coord1 offset of global memory for each Wave")

    kStr += vectorStaticDivideAndRemainder(tid1, tid0, tid0, self.kernel["WavefrontSize"], tmpS0)
    kStr += inst("v_mad_u32_u24", vgpr(waveCoord1), kernel["MatrixInstN"]//kernel["MIWaveGroup"][0], vgpr(tid1), vgpr(waveCoord1), \
                  "waveCoord1 += waveCoord0 * MiN / WaveGroupM")

    kStr += inst("v_lshrrev_b32", vgpr(tmpV1),\
                hex(log2(nThreadPerCol)), vgpr(tid0), \
                "tid / nThreadPerCol")

    kStr += inst("_v_add_u32", vgpr(coord1Offset), vgpr(waveCoord1),vgpr(tmpV1),"coord1 offset in MacroTile")

    kStr += inst("s_mul_i32", \
        sgpr(tmpS0), \
        hex(kernel["MacroTile0"]), \
        sgpr(wg0), \
        "%s = wg0*MT0"%sgpr(tmpS0))

    kStr += inst("_v_add_co_u32", vgpr(tid0), self.vcc, sgpr(tmpS0), vgpr(coord0), "coord0 = coord0 + wg0 * MT0")

    kStr += inst("s_mul_i32", \
        sgpr(wgMT1), \
        "MT1", \
        sgpr(wg1), \
        "<- wg1*MT1")
    kStr += inst("_v_add_co_u32", \
        vgpr(tid1), \
        self.vcc, \
        sgpr(wgMT1), \
        vgpr(coord1Offset), \
        "coord1 = tid1*VW + wg1*MT1")

    kStr += "\n"

    kStr += self.syncThreads(kernel, "StoreRemap Start")

    self.storeRemapLW = storeRemapLW  #local write
    self.storeRemapLR = storeRemapLR  #local read
    self.storeRemapCoord0 = tid0      #global coord0
    self.storeRemapCoord1 = tid1      #global coord1
    self.storeRemapOffsetCoord1 = coord1Offset #offset coord1

    self.vgprPool.checkIn(tmpV0)

    return kStr

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
  def notLocalSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    elements = [[] for y in range(2)] # 2D array for Full, Edge

    (fullVw, elements[False]) = self.notLocalFullTileElements(kernel, False)
    (edgeVw, elements[True])  = self.notLocalFullTileElements(kernel, True)

    # print("len(elements[False])= ", len(elements[False]))
    # print("len(elements[True])= ", len(elements[True]))
    vectorWidths = [fullVw, edgeVw]

    kStr = self.globalWriteElements(kernel, vectorWidths, elements)

    self.cleanupGlobalWrite(kernel)

    return kStr

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""

    fullVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    fullVw = min(fullVw, self.maxGwvw(kernel))
    elements = [[] for y in range(2)] # 2D array for Full, Edge
    # Full tile loop:
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], fullVw): # note step by fullVw
            element = (tt1, tt0, vc1, vc0)
            elements[False].append(element)

    # Edge tile loop - note if we know AF0EM we can can use a larger vector
    # and reduce the boundary checks accordingly.  But if no AF0EM guarantee
    # then use a conservative 1
    edgeVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    edgeVw = min(edgeVw, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
    assert(kernel["GlobalWriteVectorWidth"]%edgeVw == 0)
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], edgeVw):
            element = (tt1, tt0, vc1, vc0)
            elements[True].append(element)

    vectorWidths = [fullVw, edgeVw]
    kStr =  self.globalWriteElements(kernel, vectorWidths, elements)
    self.cleanupGlobalWrite(kernel)
    return kStr

  ##############################################################################
  # StoreState
  # tracks state that is preserved across globalWriteBatch calls:
  # init is called before globalWriteBatch
  # the KernelWriter object
  ##############################################################################
  class StoreState:

    ##############################################################################
    # Setup store config for number of sgpr and vgpr needed
    # These are set based on edge, atomic, etc - do not change during
    # the generation of the store code.
    ##############################################################################
    class StoreConstConfig:
      def __init__(self, kernelWriter, kernel, ss, gwvw, edge, beta, atomic, isWorkspace=False):
        self.gwvw = gwvw

        if ss.optSingleColVgpr:
          # use one vgpr (allocated in ss.sharedColDVgprs) for all addressing
          # - need 0 additional vgpr per element.
          self.numVgprsPerAddr = 0
        else:
          self.numVgprsPerAddr = kernelWriter.rpgo if kernel["BufferStore"] else kernelWriter.rpga

        if ss.optSGPRUsage == 'BufferLoad_Mask':
          self.numMaskSgprPerElement = 0
          self.numMaskSgprPerBatch   = 0
          self.numTempSgprPerBatch   = kernelWriter.laneSGPRCount
        elif ss.optSGPRUsage == 'BufferLoad_Edge_Mask':
          self.numMaskSgprPerElement = 0
          self.numMaskSgprPerBatch   = kernelWriter.laneSGPRCount
          self.numTempSgprPerBatch   = 2 * kernelWriter.laneSGPRCount
        else:
          # calculate number of instructions per gwvw
          # a separate mask is necessary for each single atomic instruction to avoid adding same value more than once
          numAtomic = 1
          if atomic:
            bytePerGwvw = kernelWriter.bpeCexternal * gwvw
            atomicW = min(gwvw, kernel["VectorAtomicWidth"])
            # bytePerAtomicW is either 4 (b32) or 8(b64). should not exceed 8.
            bytePerAtomicW = min(8, kernelWriter.bpeCexternal * atomicW)
            numAtomic = bytePerGwvw // bytePerAtomicW
          self.numMaskSgprPerElement = kernelWriter.laneSGPRCount * numAtomic # need 1 lane per atomic
          self.numMaskSgprPerBatch   = 0
          self.numTempSgprPerBatch   = 2 * kernelWriter.laneSGPRCount

        if self.numMaskSgprPerElement:
          numSgprAvailable = kernelWriter.maxSgprs - kernelWriter.sgprPool.size() + kernelWriter.sgprPool.availableBlockAtEnd()
          numSgprAvailable = numSgprAvailable & ~0x1 # make sure it's aligned
          #print("numSgprAvailable=", numSgprAvailable)
          self.numElementsPerBatchLimitedBySgprs = (numSgprAvailable - self.numTempSgprPerBatch - self.numMaskSgprPerBatch) // self.numMaskSgprPerElement
        else:
          self.numElementsPerBatchLimitedBySgprs = 9999 # no limit

        if self.numElementsPerBatchLimitedBySgprs<=0:
          kernelWriter.overflowedResources = 2
          self.numElementsPerBatchLimitedBySgprs = 1 # dummy value
            #assert self.numElementsPerBatchLimitedBySgprs > 0, "numElementsPerBatchLimitedBySgprs=0 for %s"%self.kernelName

        kernelWriter.HHH_WMMA = kernel["EnableMatrixInstructionStore"] \
                                and kernelWriter.asmCaps["HasWMMA"] \
                                and kernel["ProblemType"]["DestDataType"].isHalf() \
                                and (not kernel["ProblemType"]["HighPrecisionAccumulate"])

        bpeC = kernelWriter.bpeCexternal
        if isWorkspace:
          bpeC = kernelWriter.bpeCinternal

        if atomic:
          # flat atomics have another VGPR to allow different data for return#
          regsPerElement = 2
          # The atomic loop processes multiple elements in single instruction
          # so will use VGPR from consec elements? TODO
          self.numVgprsPerDataPerVI = (1.0 * regsPerElement * bpeC) / kernelWriter.bpr
        elif beta:
          self.numVgprsPerDataPerVI = (1.0 * bpeC) / kernelWriter.bpr
          if kernelWriter.HHH_WMMA:
              self.numVgprsPerDataPerVI = 1.0
        else:
          self.numVgprsPerDataPerVI = 0.0

        if kernelWriter.serializedStore:
          #self.numVgprPerValuC = kernel["MIRegPerOut"]
          # TODO check stream-k case
          self.numVgprPerValuC = kernelWriter.bpeCinternal//kernelWriter.bpr # vgpr needed from register pool
        else:
          self.numVgprPerValuC = 0 # null since they are already declared in macro part of assembly kernel

        # indicates each vector element is actually half -
        # changes vgpr allocation so two elements share a data vgpr
        # Really only used if gwvw=1 - edge cases
        #TODO: change name of halfdataRegPerVI to indicate both f8 & f16
        # exception: data vgpr cannot be shared if UseInitialStridesCD is enabled and card enable EccHalf,
        #            since each buffer_load_short would overwrite undefined 16bit as zero.
        self.halfDataRegPerVI = gwvw*self.numVgprsPerDataPerVI < 1.0 and not (kernel["ProblemType"]["UseInitialStridesCD"] and kernelWriter.archCaps["HasEccHalf"])

    # StoreState constructor:
    def __init__(self, kernelWriter, kernel, gwvw, edge, beta, atomic, elements, isWorkspace=False):
      self.kernelWriter = kernelWriter
      self.kernel = kernel

      #--
      # Optimizations for coord0/column address calculations:
      #
      # optSingleColVgpr:
      #  - works in cases where the data is written row by row to memory.
      # In this case we can use a single vgpr for addressing:
      #  - Use the load/store instruction offset (fixed at compile-time)
      #  - the horizontal addresses are fixed offsets from the base
      #  - as we move to a new row, increment the appropriate SRDs

      # optSharedColVgpr:
      #  - Each col gets it's own address, but elements in later rows with the same col will share VGPR.
      #  - allows cols to be non-adjacent
      #  - this is mutually exclusive with optSingleColVgpr - not as optimal but provides
      #    more flexibility.

      # optSrdIncForRow: optimize coord1/row address calculations:
      #  - Move the SRD between memory operations to get to new row
      #    atomic needs to reset the SRD to handle retry loop.  Then might work.

      self.optSingleColVgpr = 0
      self.optSharedColVgpr = 0
      self.optSrdIncForRow  = 0

      # opt*ColVgpr doesn't work for edge since each element requires own addr VGPR so
      #    we can perform bounds check and set to -1 for OOB accesses.
      # if optSingleColVgpr = optSharedColVgpr = 0, then each element gets
      #  1-2 VGPRs to track address.  Address calcs are performed independently
      #  for each element.

      # atomic contains multiple memory operations which need to preserve
      # the address for each load.  Memops in same row can use offsets
      # and share a base register but Memops in different rows need
      # different registers or need to inteligently reset the SRD.
      if kernel["BufferStore"] and not edge and not atomic:
        if len(kernel["PackedC0IndicesX"]) > 1:
          # packed mode needs a unique VGPR address calc for each column.
          self.optSharedColVgpr = 1
        elif len(kernel["PackedC1IndicesX"]) > 1:
          self.optSharedColVgpr = 0
          self.optSingleColVgpr = 0
        else:
          self.optSingleColVgpr = 1

        if not atomic and len(kernel["PackedC1IndicesX"]) == 1:
          self.optSrdIncForRow = 1

      if kernel["StoreRemapVectorWidth"] and not isWorkspace:
        self.optSrdIncForRow = 1

      if kernel["ProblemType"]["UseInitialStridesCD"]:
        self.optSingleColVgpr = 0 # BOZO, hack to disable this
        self.optSharedColVgpr = 0# BOZO, hack to disable this

      self.optSGPRUsage = None
      if kernel["BufferStore"] and (not atomic):
        self.optSGPRUsage = 'BufferLoad_Edge_Mask' if edge else 'BufferLoad_Mask'

      # can't have both of these enabled:
      assert (not (self.optSingleColVgpr and self.optSharedColVgpr))


      self.cfg = self.StoreConstConfig(kernelWriter, kernel, self, gwvw, edge, beta, atomic, isWorkspace)

      # Use to detect new rows:
      self.lastCoordOffset1 = 0

      # vgpr holding current coord, setup initial state
      self.coord1Vgpr = kernelWriter.coord1

      if self.optSharedColVgpr:
        numCols = len([e for e in elements if e[0] == 0 and e[2] == 0]) # count #elements with row d1=v1==0
        self.numAddrVgpr = numCols
        self.sharedColDVgprs = kernelWriter.vgprPool.checkOut(self.numAddrVgpr, "sharedColDVgprs for packed elements")
        if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
          self.sharedColCVgprs = kernelWriter.vgprPool.checkOut(self.numAddrVgpr, "sharedColCVgprs for packed elements")
        else:
          self.sharedColCVgprs = self.sharedColDVgprs
      elif self.optSingleColVgpr:
        self.numAddrVgpr = 1
        if kernel["StoreCInUnroll"]:
          self.sharedColDVgprs = "GlobalWriteOffsetD"
        else:
          self.sharedColDVgprs = kernelWriter.vgprPool.checkOut(1, "sharedColDVgprs")
        self.singleColDAddrUpdated = False
        self.singleColCAddrUpdated = False
        if kernel["ProblemType"]["UseBeta"] and not kernel["AtomicAddC"]:
          if kernel["StoreCInUnroll"]:
            self.sharedColCVgprs = "GlobalReadOffsetC"
          else:
            self.sharedColCVgprs = kernelWriter.vgprPool.checkOut(1, "sharedColCVgprs")
        else:
          self.sharedColCVgprs = self.sharedColDVgprs
      else:
        self.numAddrVgpr = 0
        self.sharedColDVgprs = None
        self.sharedColCVgprs = None

      # For detecting when we are running first batch
      self.firstBatch = True

    ##############################################################################
    # Setup data structures to feed store loops:
    #   self.elementAddr, self.elementData, self.elementMask, self.elementSumIdx
    # batchElements is a list of (d0,d1,v0,v1) for which stores to perform
    # batchElementSgprs is SGPRs to use for mask.  If None, elementMask is
    #  not initialized.
    #
    # Also create an AddrCalc for each memory operation.
    ##############################################################################
    def setupStoreElementsForBatch(self, kernel, gwvw, batchElements, batchElementSgprs, preventOverflow, isWorkspace=False):

      self.elementAddr = []
      self.elementData = []  # VGPR to use for element data, needed for atomic or beta
      self.elementMask = []  # SGPR to use for element mask
      self.elementSumIdx = []

      kw = self.kernelWriter
      dataType = kernel["ProblemType"]["DestDataType"]
      if isWorkspace:
        dataType = kernel["ProblemType"]["ComputeDataType"]

      if kernel["EnableMatrixInstructionStore"]:
        matrixInstM  = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
        matrixInstBM = 1                                                if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
        matrixInstBN = 1                                                if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

      lastData = 0
      for elementIdx in range(0, len(batchElements)):
        # Create the AddrCalc for each memory load/store
        # This is the control code that sets up the dest, source, offsets, etc and
        # identifies cases where the AddrCalc is a new row and therefore needs some
        # additional math.  Each AddrCalc contains isolated state sufficient to
        # perform any needed range checks and address calculations for the element.
        #
        # The AddrCalc creation code here maintains state across elements (including
        # across write batches) to remove replicated calculations.
        #
        # Later the AddrCalc::emitAddressSetupCode will emit the necessary code
        # Also allocate VGPR resources here, if needed.

        element = batchElements[elementIdx]
        (d1,d0,vc1,vc0) = element

        coordOffset1 = 0
        if kernel["EnableMatrixInstructionStore"]:
          MIOutputVectorWidth = kernel["MIOutputVectorWidth"]
          MFMAContinuousOutputs = MIOutputVectorWidth if kernel["SourceSwap"] else 1
          OutputsPerMIMN        = (matrixInstM * matrixInstN // self.kernel["WavefrontSize"]) if kernel["SourceSwap"] else 1

          eIdx1        = d1 % (OutputsPerMIMN // MFMAContinuousOutputs)
          remain_d1    = d1 // (OutputsPerMIMN // MFMAContinuousOutputs)

          bIdx1     = remain_d1 % matrixInstBN
          remain_d1 = remain_d1 // matrixInstBN
          wtIdex    = remain_d1 % kernel["MIWaveTile"][1]

          coordOffset1  = eIdx1 * (self.kernel["WavefrontSize"] // matrixInstN) * MFMAContinuousOutputs
          coordOffset1 += bIdx1 * matrixInstN
          coordOffset1 += wtIdex * matrixInstN *  matrixInstBN * kernel["MIWaveGroup"][1]
          coordOffset1 = coordOffset1 * kernel["VectorWidthB"] + vc1
        else:
          if kernel["LocalSplitU"] > 1:
            strideD1 = (kernel["NumThreads"]*kernel["VectorWidth"]//kernel["MacroTile0"])
          else:
            strideD1 = (kernel["SubGroup1"] * kernel["VectorWidth"])
          coordOffset1 = d1 * strideD1 + vc1

        newCoord1 = (self.firstBatch and elementIdx==0) or (coordOffset1 != self.lastCoordOffset1)

        # gpr and offset assignments for element
        coordOffset0 = 0
        if kernel["EnableMatrixInstructionStore"]:
          MFMAContinuousOutputs = 1 if kernel["SourceSwap"] else kernel["MIOutputVectorWidth"]
          OutputsPerMIMN        = 1 if kernel["SourceSwap"] else matrixInstM * matrixInstN // self.kernel["WavefrontSize"]

          eIdx0        = d0 % (OutputsPerMIMN // MFMAContinuousOutputs)
          remain_d0    = d0 // (OutputsPerMIMN // MFMAContinuousOutputs)

          bIdx0        = remain_d0 % matrixInstBM
          remain_d0    = remain_d0 // matrixInstBM
          wtIdex       = remain_d0 % kernel["MIWaveTile"][0]

          coordOffset0  = eIdx0  * (self.kernel["WavefrontSize"] // matrixInstM) * MFMAContinuousOutputs
          coordOffset0 += bIdx0  * matrixInstM
          coordOffset0 += wtIdex * matrixInstM * matrixInstBM * kernel["MIWaveGroup"][0]
          coordOffset0  = coordOffset0 * kernel["VectorWidth"] + vc0
        else:
          coordOffset0 = d0 * kernel["SubGroup0"]*kernel["VectorWidth"] + vc0

        if self.optSingleColVgpr:
          # use same address vgpr for all
          addrDVgpr = self.sharedColDVgprs
          addrCVgpr = self.sharedColCVgprs
        elif self.optSharedColVgpr:
          if kernel["EnableMatrixInstructionStore"]:
            elementCol = (d0 * kernel["MIOutputVectorWidth"] + vc0) / gwvw
          else:
            elementCol = (d0 * kernel["VectorWidth"] + vc0) / gwvw
          assert (modf(elementCol)[0] < 0.001)
          elementCol = trunc(elementCol)
          addrDVgpr = self.sharedColDVgprs+elementCol
          addrCVgpr = self.sharedColCVgprs+elementCol
          #print ("d0=", d0, "vc0=", vc0, "elementCol=", elementCol)
        else:
          # allocate new VGPR for each element:
          # need to allow expanding register pool (set preventOverflow=false)
          addrDVgpr = kw.vgprPool.checkOutAligned(self.cfg.numVgprsPerAddr, \
              int(ceil(self.cfg.numVgprsPerAddr)), "writeDBatch-addr for ei=%u"%(elementIdx), preventOverflow=False)
          if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
            addrCVgpr = kw.vgprPool.checkOutAligned(self.cfg.numVgprsPerAddr, \
                int(ceil(self.cfg.numVgprsPerAddr)), "loadCBatch-addr for ei=%u"%(elementIdx), preventOverflow=False)
          else:
            addrCVgpr = addrDVgpr

        self.elementAddr.append(kw.AddrCalc(kw, self, addrCVgpr, addrDVgpr, element, coordOffset0, \
          self.kernelWriter.coord1, coordOffset1, coordOffset1 - self.lastCoordOffset1, newCoord1))
        # if numVgprsPerDataPerVI == 0.5, then two consecutive elements
        # should have same data pointer, next should move.

        if self.cfg.numVgprsPerDataPerVI > 0:
          if self.cfg.halfDataRegPerVI:
            # TODO- check (H,H,H,H,S,S)
            # NOTE: Changed from DataType to DestDataType
            if kernel["ProblemType"]["HighPrecisionAccumulate"] and \
               (dataType.isBFloat16() or dataType.isHalf()):
              data = kw.vgprPool.checkOutAligned(int(2*self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw), \
                    int(ceil(int(2*self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw))), "writeBatch-data for ei=%u and ei=%u"%(elementIdx,elementIdx+1), preventOverflow=preventOverflow)
            elif dataType.is8bitFloat():
              numRegForData = int(ceil(self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw))
              data = kw.vgprPool.checkOutAligned(numRegForData, numRegForData, "writeBatch-data for ei=%u and ei=%u"%(elementIdx,elementIdx+1), preventOverflow=preventOverflow)
            else:
              if elementIdx%2 == 0:
                # allocate for two elements:
                data = kw.vgprPool.checkOutAligned(int(2*self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw), \
                       int(ceil(int(2*self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw))), "writeBatch-data for ei=%u and ei=%u"%(elementIdx,elementIdx+1), preventOverflow=preventOverflow)
                lastData = data
              else:
                data = lastData
                del lastData
          else:
            if self.cfg.numVgprsPerDataPerVI == 0.5:
              data = kw.vgprPool.checkOutAligned(int(ceil(self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw)), \
                    int(ceil(self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw)), "writeBatch-data for ei=%u"%elementIdx, preventOverflow=False)
            else:
              data = kw.vgprPool.checkOutAligned(int(self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw), \
                    int(ceil(self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw)), "writeBatch-data for ei=%u"%elementIdx, preventOverflow=False)
            #data = kw.vgprPool.checkOut(int(self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw), \
            #      "writeBatch-data for ei=%u"%elementIdx, preventOverflow=False)
        else:
          data = 0

        self.elementData.append(data)
        if batchElementSgprs != None:
          if self.optSGPRUsage:
            mask = batchElementSgprs
          else:
            mask = batchElementSgprs + self.cfg.numMaskSgprPerBatch + elementIdx * self.cfg.numMaskSgprPerElement
          self.elementMask.append(mask)

        #print "Edge=", edge, element
        sumIdx = 0
        if kernel["LocalSplitU"] > 1:
          sumIdx = kw.startVgprValuC + vc0 + d1*kernel["VectorWidthA"]
        else:
          bestVw                  = kernel["VectorWidthA"]
          elementsLoadedPerVw     = kernel["NumThreads"] * bestVw
          elementsLoadedPerbestVw = kernel["NumThreads"] * kernel["StoreVectorWidth"]

          if elementsLoadedPerVw < elementsLoadedPerbestVw:
            bestVw = kernel["StoreVectorWidth"]

          if kernel["EnableMatrixInstructionStore"]:
            alignment = self.cfg.numVgprPerValuC * self.cfg.gwvw
            sumIdx    = kw.vgprPool.checkOutAligned(self.cfg.numVgprPerValuC*self.cfg.gwvw, alignment, "vgprValuC") // self.cfg.numVgprPerValuC
          else:
            sumIdx = kw.startVgprValuC + vc0 + d0*kernel["VectorWidth"] + vc1*kernel["ThreadTile0"] + d1*kernel["VectorWidth"]*kernel["ThreadTile0"]
        self.elementSumIdx.append(sumIdx) # sumIdx is an element idx, need to div/2 for half
        self.lastCoordOffset1 = coordOffset1

    def checkInTempVgprC(self):
      if self.kernelWriter.serializedStore is False:
        return # early exit; currently only serializedStore==True checks out C-tile from register pool

      if len(self.elementSumIdx) > 0:
        for i in self.elementSumIdx:
          self.kernelWriter.vgprPool.checkIn(i * self.cfg.numVgprPerValuC)
          # print("checked in vgpr %u"%i)
        self.elementSumIdx = []

    def __del__(self):
      if (self.sharedColDVgprs != None) and not self.kernel["StoreCInUnroll"]:
        self.kernelWriter.vgprPool.checkIn(self.sharedColDVgprs)
        if (self.sharedColCVgprs != self.sharedColDVgprs):
          self.kernelWriter.vgprPool.checkIn(self.sharedColCVgprs)
      self.checkInTempVgprC()

  ##############################################################################
  # Fields associated with computing address
  ##############################################################################
  class AddrCalc:
    # rowInc is number of rows to add to the base address
    # coord0Vgpr : This is VGPR that holds coord0.  Coord0 is element-space
    #    packed index for the 0 coordinate of the C/D matrix.
    # coord1Vgpr : VGPR which tracks the last coord1 calculation.
    #          If this is new coord1, just overwrite it with latest calc.
    def __init__(self, kernelWriter, ss, addrCVgpr, addrDVgpr, element, \
        coordOffset0, coord1Vgpr, coordOffset1, rowInc, newCoord1):
      self.kernelWriter = kernelWriter

      # vgprs for address, could be more than one (for 64bit)
      self.addrDVgpr = addrDVgpr
      self.addrCVgpr = addrCVgpr
      self.coord1Vgpr = coord1Vgpr # vgpr that stores coord1Vgpr

      self.element = element
      self.coordOffset0 = coordOffset0
      self.coordOffset1 = coordOffset1
      self.rowInc = rowInc
      self.rowIncDirtyRowPtr = 0 # rowInc was used to modify rowPtr, need to recompute addr
      self.newCoord1 = newCoord1 # vgpr that stores newCoord1

      if ss.optSingleColVgpr:
        # optimized stores use the load offset for coordOffset0 calculations.
        self.globalOffset = coordOffset0 * kernelWriter.bpeCexternal
      else:
        # else non-opt stores include the coord0 offset into VGPR address calcs
        self.globalOffset = 0

    def addScaled(self, destV, src0, src1, scale1, tmpS01, comment=""):
      """
      Use minimally efficient instructions to add stride*scale
      """

      kStr = ""
      if scale1 == 1:
        kStr += inst("_v_add_u32", destV, src0, src1, comment)
      else:
        kStr += inst("s_mul_i32", sgpr(tmpS01), src1, scale1, "scale stride")
        kStr += inst("_v_add_u32", destV, src0,  sgpr(tmpS01), comment)
      return kStr


    def emitAddressCoordIncrement(self, kernel, ss, tmpVgpr, tmpS01, updateCoord1):
      """
      Emit code that computes the coord0 and coord1 for this element
      sets self.coord0Vgpr with the address that holds the coord0 value for this element.
      Input:
        - tmpVgpr is a 1 temporary VGPR used for coord0 calculation on edges
      """

      kStr = ""
      kw = self.kernelWriter
      (d1,d0,vc1,vc0) = self.element
      self.coord0Vgpr = None # will set below

      #kStr += self.kernelWriter.comment1("store addr=v%u coordOffset0=%u"% \
      #    (self.addr, self.coordOffset0))
      kStr += self.kernelWriter.comment1("(d1,vc1,d0,vc0)=(%u,%u,%u,%u)"\
          % (d1,vc1,d0,vc0))
      if ss.optSingleColVgpr:
        self.coord0Vgpr = kw.coord0
      elif not ss.optSharedColVgpr or (d1 == vc1 == 0):
        # not share mode or first row always does the address calc math:

        if self.coordOffset0 == 0:
          self.coord0Vgpr = kw.coord0
        elif self.coordOffset0 <= 64:
          self.coord0Vgpr = tmpVgpr
          kStr += inst("_v_add_co_u32", vgpr(self.coord0Vgpr), self.kernelWriter.vcc, vgpr(kw.coord0), self.coordOffset0, \
                    "coord0.1: coord0 += d0*sg0*VW + vc0")
        else:
          self.coord0Vgpr = tmpVgpr
          kStr += inst("s_mov_b32", sgpr(tmpS01), self.coordOffset0, "coordOffset0 d0=%u vc0=%u"%(d0, vc0))
          kStr += inst("_v_add_co_u32", vgpr(self.coord0Vgpr), self.kernelWriter.vcc, vgpr(kw.coord0), sgpr(tmpS01), \
                    "coord0.2: coord0 += d0*sg0*VW + vc0")

        if self.newCoord1:
          if not kernel["BufferStore"] or updateCoord1:
            if self.rowInc== 0:
              None
            elif self.rowInc <= 64:
              # rowInc fits in instruction:
              kStr += inst("_v_add_co_u32", vgpr(self.coord1Vgpr), self.kernelWriter.vcc, \
                        vgpr(self.kernelWriter.coord1), self.rowInc, \
                        "coord1.1: coord1Vgpr += d1*sg1*VW + vc1")
            else:
              kStr += inst("s_mov_b32", sgpr(tmpS01), self.rowInc, "rowInc d1=%u vc1=%u"%(d0, vc0))
              kStr += inst("_v_add_co_u32", vgpr(self.coord1Vgpr), self.kernelWriter.vcc, \
                        vgpr(self.kernelWriter.coord1), sgpr(tmpS01), \
                        "coord1.2: coord1 += d1*sg1*VW + vc1")
      return kStr

    # storeChar is 'C' or 'D'
    # elementVgpr is coord0Vgpr*strideCD0, or optimized to just coord0Vgpr if strideCD0 is unit const
    def emitExtractAndScalePackedDims(self, kernel, ss, tmpVgpr, storeChar):
      kStr = ""
      kw = self.kernelWriter
      packedIndices = kernel["PackedC0IndicesX"]
      packedBits = self.coord0Vgpr # start with coord0, will move to temp below
      rowPtr = kw.cinRowPtr if (storeChar == 'C') else kw.coutRowPtr
      addrVgpr = self.addrCVgpr if (storeChar == 'C') else self.addrDVgpr

      for i,idx in enumerate(packedIndices[:-1]):
        # vgprTmp assignments:
        #   - tmp+0 may be the incoming packed coordinate 0, used on replay too
        #   - tmp+1 is DIV output
        #   - tmp+2 is scratch
        idxChar= globalParameters["IndexChars"][idx]
        kStr += kw.comment1("extract %s"%kw.sizeRef(idx))
        assert(tmpVgpr+1 != packedBits) # bad since we still need packedBits below for remainder (can't overwrite here)
        kStr += "V_MAGIC_DIV %s, %s, %s, %s, %s\n" % \
                 (tmpVgpr+1, vgpr(packedBits), sgpr("MagicNumberSize%s"%idxChar), \
                  sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0")
        # tmpVgpr+1 returns the quotient, tmpVgpr+2 is overwritten

        # compute remainder, packedBits % sizeIdx - this is the 'extracted' index that must be scaled
        # remainder is mul and sub
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+1), kw.sizeRef(idx), \
                     "remainder part 1")
        kStr += inst("_v_sub_u32", vgpr(tmpVgpr+2), vgpr(packedBits), vgpr(tmpVgpr+2),
                      "remainder part 2")

        if i==0:
          kStr += inst("v_mul_lo_u32", vgpr(addrVgpr), vgpr(tmpVgpr+2), \
                    kw.strideRef(storeChar, idx), "addrCalc <- scaled extracted dim")
        else:
          kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+2), \
                    kw.strideRef(storeChar, idx), "scale extracted dim")
          kStr += inst("_v_add_u32", vgpr(addrVgpr), vgpr(addrVgpr), \
                    vgpr(tmpVgpr+2), "addrCalc += scaled extracted dim ")

        if i < len(packedIndices)-2:
          # TODO - might be able to eliminate this
          kStr += inst("v_mov_b32", vgpr(tmpVgpr+0), vgpr(tmpVgpr+1), \
                    "Copy remaining bits for next divide")
          packedBits = tmpVgpr+0

      if len(packedIndices)>1:
        # if we unpacked something, then scale it to BPE
        kStr += kw.comment1("extract final %s"%kw.sizeRef(packedIndices[-1]))
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+1), \
                  kw.strideRef(storeChar, packedIndices[-1]), "scale final extracted dim")
        kStr += inst("_v_add_u32", vgpr(addrVgpr), vgpr(addrVgpr), \
                  vgpr(tmpVgpr+2), "addrCalc += scaled extracted dim ")

        kStr += inst("_v_add_lshl_u32", vgpr(addrVgpr), \
                  vgpr(rowPtr), \
                  vgpr(addrVgpr), \
                  hex(log2(kw.bpeCexternal)), \
                  "packed: add rowPtr and scaleToBpe")

      return kStr

    def emitScaleToBpe(self, kernel, ss, tmpVgpr, singleUpdate, tc):
      """
      Needs 3 temporary VGPRs
      """

      kStr = ""
      if tc == 'WS': # StreamK workspace uses compute size
        tc = 'C'
      kw = self.kernelWriter
      (d1,d0,vc1,vc0) = self.element
      rowPtr = kw.cinRowPtr if (tc == 'C') else kw.coutRowPtr
      addrVgpr = self.addrCVgpr if (tc == 'C') else self.addrDVgpr
      # set when we generate code that updates the address
      # optSingleColVgpr and optSharedColVgpr attempt to minimize these updates
      updatedAddr = False

      # scale and set final address:
      stride0 = kw.strideRef(tc, 0)
      if kw.isConstUnitStride(stride0):
        elementVgpr = self.coord0Vgpr
      else:
        kStr += inst("v_mul_lo_u32", \
            vgpr(addrVgpr), \
            vgpr(self.coord0Vgpr), \
            stride0, \
            "scale element by non-unit stride")
        elementVgpr = addrVgpr

      if ss.optSingleColVgpr:
        # This is first element in the first batch, create a byte address that will
        # be re-used by subsequent elements:
        # if this element is firstInBatch - may need to set up a bpe-scaled row pointer for the batch:
        #  - need row-ptr start of each batch
        assert (kw.coord0 == self.coord0Vgpr) # elementAddr assignment above assumes these are the same
        if singleUpdate:
          updatedAddr = True
          singleColAddrUpdated = ss.singleColCAddrUpdated if (tc == 'C') else ss.singleColDAddrUpdated
          if not singleColAddrUpdated or not ss.optSrdIncForRow:
            if tc == 'C':
              ss.singleColCAddrUpdated = True
            else:
              ss.singleColDAddrUpdated = True
            kStr += inst("_v_add_lshl_u32", \
              vgpr(addrVgpr), \
              vgpr(rowPtr), \
              vgpr(elementVgpr), \
              hex(log2(kw.bpeCexternal)), \
              "optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=%d, coord0Vgpr=%d"%(kw.coord0, self.coord0Vgpr))
      elif ss.optSharedColVgpr:
        # Need an address calculation for the first address in each row:
        if d1==0 and vc1==0:
          packedIndices = kernel["PackedC0IndicesX"]
          if len(packedIndices) > 1:
            updatedAddr = True
            kStr += self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc)
          else:
            updatedAddr = True
            kStr += inst("_v_add_lshl_u32", \
              vgpr(addrVgpr), \
              vgpr(rowPtr), \
              vgpr(elementVgpr), \
              hex(log2(kw.bpeCexternal)), \
              "optSharedColVgpr scaleToBpe for first row: col addr <- cinRowPtr + coord0, scaled by BPE")
      else:
        # Generate final address calculation (to bytes) for each element
        # The unpacking takes 8-10 instructions so could be worth optimizing someday :
        # each col has same offset so could create a class to hold column-specific state including
        # the byte address offset for that col and the mask in/out.
        packedIndices = kernel["PackedC0IndicesX"]
        if len(packedIndices) > 1:
          updatedAddr = True
          kStr += self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc)
        else:
          updatedAddr = True
          kStr += inst("_v_add_lshl_u32", \
              vgpr(addrVgpr), \
              vgpr(rowPtr), \
              vgpr(elementVgpr), \
              hex(log2(kw.bpeCexternal)), \
              "scaleToBpe: accumulate d0 lower and *= bpe into Cin addr")

      # if not optSrdIncForRow then we may have moved the row pointer
      # and depending on paths above may not have refreshed addrVgpr already.
      # if so - do it here:
      if self.rowIncDirtyRowPtr and not updatedAddr:
        kStr += inst("_v_add_lshl_u32", \
          vgpr(addrVgpr), \
          vgpr(rowPtr), \
          vgpr(kw.coord0), \
          hex(log2(kw.bpeCexternal)), \
          "scaleToBpe: Update address with new rowPtr")

      return kStr

    def edgeProtectCode(self, kernel, edge, beta, atomic, mask, tmpSgpr):
      """
      Generate code to protect address offset in edge case
      """

      kStr = ""
      kw = self.kernelWriter
      tmpS01 = tmpSgpr
      tmpS23 = tmpSgpr+self.kernelWriter.laneSGPRCount

      laneSGPRCount = self.kernelWriter.laneSGPRCount
      wavefrontSize = kernel["WavefrontSize"]

      # Now do the edge check and compute the address in bytes:
      if kernel["BufferStore"]:
        if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and beta)):
          # Set address to -1 if OOB on either dimension
          # and only check the x/coord0 index here, save a couple inst
          sizeBoundary = [0,0]
          sizeBoundary[0] = \
              sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
              else kw.sizeRef(kernel["ProblemType"]["Index0"])
          sizeBoundary[1] = \
              sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
              else kw.sizeRef(kernel["ProblemType"]["Index1"])

          kStr += inst("v_cmp_lt_u32", sgpr(tmpS01,laneSGPRCount), vgpr(self.coord0Vgpr), sizeBoundary[0], "coord0 < size0" )
          kStr += inst("v_cmp_lt_u32", sgpr(mask,laneSGPRCount), vgpr(self.coord1Vgpr), sizeBoundary[1], "coord1 < size1" )
          kStr += inst("s_and_b{}".format(wavefrontSize), sgpr(mask,laneSGPRCount), sgpr(tmpS01,laneSGPRCount), sgpr(mask,laneSGPRCount), "in0 && in1" )
      else:
        kStr += inst("v_cmp_lt_u32", sgpr(tmpS01,laneSGPRCount), vgpr(self.coord0Vgpr), sgpr("SizesFree+0"), "coord0 < size0" )
        kStr += inst("v_cmp_lt_u32", sgpr(tmpS23,laneSGPRCount), vgpr(self.coord1Vgpr), sgpr("SizesFree+1"), "coord1 < size1" )
        kStr += inst("s_and_b{}".format(wavefrontSize),  sgpr(mask,laneSGPRCount), sgpr(tmpS01,laneSGPRCount), sgpr(tmpS23,laneSGPRCount), "in0 && in1" )

        if (beta or atomic):
          kStr += inst("s_mov_b{}".format(wavefrontSize), self.kernelWriter.exec, sgpr(mask,laneSGPRCount), "sgprs -> exec" )

      return kStr

    # TODO - mask should be part of AddrCalc state not passed as parm
    def emitAddressSetupCode(self, kernel, ss, tc, tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrVgpr):
      """
      Generate code to set up the address vgpr
      Input:
        tmpVgpr : two temp vgprs
      Output:
        Returns kStr with appropriate setup code
        Sets self.coord0Vgpr with vgpr that contains the coord0 for this element.  This enables
          optimization - if no setup code is required the coord0 can be the input.
      """

      kStr = ""
      kw = self.kernelWriter

      updateCoord1 = (edge or len(kernel["PackedC1IndicesX"]) > 1)
      if tc == 'C':
        kStr += self.emitAddressCoordIncrement(kernel, ss, tmpVgpr, tmpS01, updateCoord1)
        if not kernel["BufferStore"] and not beta: # Skip extra code for C if no beta
          return kStr

      # Move the row ptr VGPR
      # optSrdIncForRow moves the SRD so don't move here
      if not ss.optSrdIncForRow and kernel["BufferStore"]:
        if self.rowInc > 0:
          self.rowIncDirtyRowPtr = 1
          #assert (not kernel["ProblemType"]["UseInitialStridesCD"])
          kStr += kw.comment("Fix for UseInitialStridesCD, emitAddressSetupCode")

          if len(kernel["PackedC1IndicesX"]) == 1:
            strideChar = self.kernelWriter.indexChars[kernel["PackedC1IndicesX"][0]]
            kStr += self.addScaled(vgpr(kw.cinRowPtr),  vgpr(kw.cinRowPtr),  \
                      sgpr("StrideC%s"%strideChar), self.rowInc, tmpS01, "ROWINC- Move cinRowPtr to next row")
            kStr += self.addScaled(vgpr(kw.coutRowPtr), vgpr(kw.coutRowPtr), \
                      sgpr("StrideD%s"%strideChar), self.rowInc, tmpS01, "Move coutRowPtr to next row")
          elif len(kernel["PackedC1IndicesX"]) > 1:
            kStr += self.kernelWriter.extractPackedCoord1ToRowStart(kernel, kernel["PackedC1IndicesX"] , self.coord1Vgpr, 'D')

      # calculate global load offset
      # this has to be done after calculating new coord1
      if not kernel["BufferStore"] and (tc == 'D' or (tc == 'C' and beta)):
        # global: in-bounds exec mask
        # global offset macro (requires 3 tmpVgpr)
        # final address = C + index*bytes
        isStridedBuffer = kernel["ProblemType"]["StridedBatched"] or (kernel["GlobalSplitU"] > 1 and kernel["_GlobalAccumulation"])
        kStr += "GLOBAL_OFFSET_%s %u" % (tc, addrVgpr)
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          if i == kernel["ProblemType"]["Index0"]:
            kStr += ", %s" % (self.coord0Vgpr)
          elif i == kernel["ProblemType"]["Index1"]:
            kStr += ", %s" % (self.coord1Vgpr)
          elif isStridedBuffer: # just a group index
            kStr += ", sgprWorkGroup%u"%i
        kStr += ", %s%s" % ((tmpVgpr+2), kw.endLine)
        kStr += inst("v_mov_b32", vgpr(tmpVgpr+2), vgpr(addrVgpr+0), "temp store offset 0")
        kStr += inst("v_mov_b32", vgpr(tmpVgpr+3), vgpr(addrVgpr+1), "temp store offset 1")

      return kStr

    def emitLdChange(self, kernel, ss, tc, edge, beta, mask, singleUpdate, tmpVgpr, addrVgpr, BufAddr):
      """
      Generate code for final C read/D write address
      """

      laneSGPRCount = self.kernelWriter.laneSGPRCount

      kStr = ""
      if kernel["BufferStore"]:
        kStr += self.emitScaleToBpe(kernel, ss, tmpVgpr, singleUpdate, tc)
        if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and beta)):
          kStr += inst("v_cndmask_b32", vgpr(addrVgpr), -1, vgpr(addrVgpr), \
                       sgpr(mask,laneSGPRCount), "LD%s clip if OOB. offset" % tc )
      else:
        # store a copy of the offset in 2 of the tmpVgpr for D
        kStr += inst("_v_add_co_u32",  vgpr(addrVgpr+0), self.kernelWriter.vcc, vgpr(BufAddr+0), vgpr(tmpVgpr+2), \
                     "addrVgpr = %s + index*bytes (lo)" % tc)
        kStr += inst("_v_addc_co_u32", vgpr(addrVgpr+1), self.kernelWriter.vcc, vgpr(BufAddr+1), vgpr(tmpVgpr+3), \
                     self.kernelWriter.vcc, "addrVgpr = %s + index*bytes (hi)" % tc)
      return kStr

    def incrementToNextRow(self, kernel, tc, ss, stmp):
      """
      Generate code to move to the next row(s)
      If optSrdIncForRow, this will move the SRD forward
      If not, this could generate some other instructions
      """

      kStr = ""
      numRows = self.rowInc
      tmpBpe = self.kernelWriter.bpeCexternal
      if ss.optSrdIncForRow:
        if numRows:
          packedC1 = kernel["PackedC1IndicesX"]
          assert(len(packedC1) == 1)  # would need to extract each dim and scale
          strideCD1 = "Stride%s%s"%(tc,self.kernelWriter.indexChars[packedC1[0]])
          if tc == 'WS': # TODO StreamK revised large WS
            strideCD1 = "Stride%s%s"%('D',self.kernelWriter.indexChars[packedC1[0]])
          if numRows > 1:
            kStr += inst("s_mul_i32", sgpr(stmp), \
                         sgpr(strideCD1), \
                         numRows*tmpBpe, \
                         "scale Stride%s *= numRows(%u) * bpe"%(tc,numRows))
          else:
            kStr += inst("s_lshl_b32 ", \
                  sgpr(stmp), \
                  sgpr(strideCD1), \
                  log2(tmpBpe), \
                  "incToNextRow: Scale by BPE")

          kStr += inst("s_add_u32 ", \
               sgpr("Srd%s+0"%(tc)), \
               sgpr("Srd%s+0"%(tc)), \
               sgpr(stmp), \
               "incToNextRow: gra SRD += inc(lower)" )
          kStr += inst("s_addc_u32 ", \
               sgpr("Srd%s+1"%(tc)), \
               sgpr("Srd%s+1"%(tc)), \
               0, \
               "incToNextRow: gra SRD += inc(upper)" )

        None

      return kStr

  ##############################################################################
  # checkIsBetaZero
  # tmpSgpr is one temp sgpr
  # betaLabel is label to branch to if beta != 0
  ##############################################################################
  def checkIsBetaZero(self, kernel, betaLabel, tmpSgpr=None):
    kStr = ""
    tmpSgprRef = None
    if tmpSgpr == None:
      tmpSgprRef = self.getTmpSgpr(1)
      tmpSgpr = tmpSgprRef.idx()
    if kernel["ProblemType"]["UseBeta"]:
      if self.bpeCinternal <= self.bpr: # 1 register to check for Beta==0
        kStr += inst("s_cmpk_eq_u32", sgpr("Beta"), hex(0), "Beta == 0")
      else: # multiple registers to check for Beta==0
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), sgpr("Beta+0"), "tmp = Beta[0]")
        for i in range(1, self.bpeCinternal//self.bpr):
          kStr += inst("s_or_b32", sgpr(tmpSgpr), sgpr("Beta+%u"%i), sgpr(tmpSgpr), "tmp |= Beta[%u] " % i)
        kStr += inst("s_cmpk_eq_u32", sgpr(tmpSgpr), hex(0), "Beta == 0")
      kStr += inst("s_cbranch_scc0 %s" % betaLabel, \
          "Branch if Beta is not zero")
      kStr += "\n"
    return kStr

  ##############################################################################
  # checkIsEdge
  # tmpSgpr must have at least 6 free SGPR
  # isEdgeTarget is the branch target if edges are required
  ##############################################################################
  def checkIsEdge(self, kernel, isEdgeTarget, tmpSgpr=None):
    kStr = ""
    tmpSgprRef = None
    if tmpSgpr == None:
      tmpSgprRef = self.getTmpSgpr(3)
      tmpSgpr = tmpSgprRef.idx()
    tmpS0  = tmpSgpr
    tmpS1  = tmpS0 + 1
    tmpS23 = tmpS1 + 1

    if self.prefetchAcrossPersistent:
      wg0="PrevWorkGroup0"
      wg1="PrevWorkGroup1"
    else:
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

    kStr += scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[0], kernel["MacroTile0"], tmpS23, 2)
    # s23 = nwg0-1
    kStr += inst("s_add_u32", sgpr(tmpS1), hex(-1), sgpr("NumWorkGroups0"), "" )
    kStr += inst("s_cmp_ge_u32", sgpr(wg0), sgpr(tmpS1), "wg0 >= nwg0-1 ?")
    kStr += inst("s_cselect_b32", sgpr(tmpS0), sgpr(tmpS0), 0, "set rMT0")
    # s01 now = myMT0 = wg0 < nwg0-1 ? MT0 : rMT0

    # if rMT0 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      kStr += inst("s_cmpk_gt_u32", sgpr(tmpS0), hex(0), "rMT0 > 0")
      if self.db["ForceEdgeStores"]:
        kStr += inst("s_cmp_eq_u32", sgpr(tmpS0), sgpr(tmpS0), "ForceEdgeStores!")
      kStr += inst("s_cbranch_scc1 %s" % isEdgeTarget, "jump if edges required")

    # check edge1 ###
    # TODO-packed - this only needs to change to handle packing into C1 index
    # change would be similar to above - multiply by product of packed sizes in C1
    # --

    # s23 = rMT1 = Size1 % MT1
    kStr += scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[1], kernel["MacroTile1"], tmpS23, 2)
    # s01 now = myMT1 = wg1 < nwg1-1 ? MT1 : rMT1

    # s23 = nwg1-1
    kStr += inst("s_add_u32", sgpr(tmpS1), hex(-1), sgpr("NumWorkGroups1"), "" )
    kStr += inst("s_cmp_ge_u32", sgpr(wg1), sgpr(tmpS1), "wg1 >= nwg1-1")
    kStr += inst("s_cselect_b32", sgpr(tmpS0), sgpr(tmpS0), 0, "set rMT1")

    # if rMT1 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      kStr += inst("s_cmpk_gt_u32", sgpr(tmpS0), hex(0), "rMT1 > 0")
      kStr += inst("s_cbranch_scc1 %s" % isEdgeTarget, "jump if edges required")

    return kStr

  ##############################################################################
  # Global Write Batch
  ##############################################################################
  def fixupBatch(self, kernel, ss, batchIdx, edge, gwvw, \
      batchElements, coord0, coord1, addrD, addrC, \
      tmpVgpr, tmpCVTVgpr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeAccVgprWrite):
    kStr = ""

    kStr += self.comment1("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
              (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

    if kernel["StoreSyncOpt"]:
      kStr += "s_sleep %d // optimization: sync and wait\n" %(kernel["StoreSyncOpt"]-1)
      kStr += "s_barrier\n"

    # comment tt1, tt0, vc1, vc0
    # tt = thread tile, vc=vector component
    commentStr = "Fixup%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Edge" if edge else "", batchIdx)
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      commentStr += "(%u,%u,%u,%u:vw%u)" % \
        (element[0], element[1], element[2], element[3], gwvw)
      if elementIdx < len(batchElements)-1:
        commentStr += "; "
    kStr += self.comment3(commentStr)
    # print(self.kernelName)
    # print(commentStr)

    # allow expanding vgpr pool for OptNLL
    preventOverflow = True #(not isOptNLL)
    ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, preventOverflow=preventOverflow, isWorkspace=True)

    loadsIssued = 0
    storesIssued = 0
    tmpS01 = tmpSgpr # scratch sgprs

    wavelen = self.kernel["WavefrontSize"]
    laneSGPRC = self.laneSGPRCount
    # always use gwvw for buffer load C for atomic_cmpswap
    # bpm = self.bpeCexternal * atomicW
    # bpm = self.bpeCexternal * gwvw
    # vgprLoadDW = 1*(bpm//4)
    # atomic oparation width. 1 for b32, 2 for b64
    # atomicOpW = (atomicW * self.bpeCexternal) // 4
    # if atomicOpW > 2:
    #   # should not exceeding 2.
    #   atomicOpW = 2

    ########################################
    # calculate addr and masks
    kStr += self.comment("calc coords, apply mask, and issue loads (if necessary)")
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.db["ConservativeWaitCnt"] & 0x10:
      kStr += "s_barrier // debug\n"
      kStr += inst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      kStr += "s_barrier // debug\n"
    if not edge and self.db["ForceEdgeStores"]>=2:
      kStr += self.bomb() # should not get here
    if edge and self.db["AssertNoEdge"]:
      kStr += self.bomb() # should not get here

    atomicAddC = kernel["AtomicAddC"] and not edge

    ## create code Module to push mov vgpr,acc instructions
    if kernel["StoreCInUnroll"] and not edge:
      accVgprRead = Code.Module("movaccVgpr")
      self.StoreCUnrollLoadCWaitComment = "waitcnt for LoadC" # this will be used later to identify waitcnt for loadC

    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
      addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
      addrCalc = ss.elementAddr[elementIdx]
      data = ss.elementData[elementIdx]
      mask = ss.elementMask[elementIdx]
      sumIdx = ss.elementSumIdx[elementIdx]
      # d1 = element[0]
      # d0 = element[1]
      # vc1 = element[2]
      vc0 = element[3]

      storeWidth = kernel["StoreVectorWidth"]
      # storeWidth = 2
      if batchIdx == 0 and elementIdx == 0:
        kStr += staticMultiply(vgpr(addrCVgpr), vgpr("Serial"), storeWidth * self.bpeCinternal, sgpr(tmpS01))
        # kStr += inst("v_mul_lo_u32", , "Partials buffer address")
        kStr += inst("s_mov_b32", sgpr(tmpS01), 0, "Init sgpr offset")
      else:
        increment = (kernel["WavefrontSize"] * 4) * storeWidth * self.bpeCinternal
        kStr += inst("s_add_u32", sgpr(tmpS01), sgpr(tmpS01), increment, "Inc sgpr offset")

      kStr += self.readCInput(kernel, ss, addrCalc, vc0, data, gwvw, addrCVgpr, sgpr(tmpS01), 'WS')
      loadsIssued += 1

    ########################################
    # AccVgpr read
    if kernel.enabledSetPrioSplitLDS:
      kStr += inst("s_setprio", "0", "")
    if codeAccVgprRead is not None:
      regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(0, len(batchElements)):
        # loop over scalars within one store instruction
        for vi in range(0, gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            tempStr = str(codeAccVgprRead.items().pop(0))
            kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
            if kernel["StoreCInUnroll"] and not edge:
              tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
              accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

      if not kernel["MIArchVgpr"]:
        kStr += inst("s_nop 1", "2 wait states required before reading vgpr")

    ########################################
    # Not Atomic
    ########################################
    # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
    interleaveStoreVmcnt = self.interleaveStoreVmcnt and not edge
    for elementIdx in range(0, len(batchElements)):
      for vi in range(0, gwvw):
        sumIdxV = ss.elementSumIdx[elementIdx] + vi
        # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
        if kernel["ProblemType"]["ComputeDataType"].isInt32() or \
          kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
            if self.db["ForceExpectedValue"]:
              kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
            if self.db["ForceVSerial"]:
              kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
            if self.db["CheckValueC"]:
              kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
              kStr += self.assert_eq(vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01))

    ########################################
    # wait for batched load
    if not interleaveStoreVmcnt: # beta and
      kStr += inst("s_waitcnt", "vmcnt(0)", "wait C")
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

      # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
      # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
      # And this only occurs when beta=true, so case must not be 2 or 3
      assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, \
        "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

    kStr += self.comment("apply mask, calc new C and issue writes")
    #kStr += self.bomb() # can see store addresses just before the store inst

    if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
      vgprBf16Temp = tmpCVTVgpr
      vgprBf16Mask = vgprBf16Temp + 1
      vgprFp32Nan = vgprBf16Temp + 2
      vgprBf16Inc = vgprBf16Temp + 3
      kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
      kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
      kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )

    # DestDataType for 8bit Float can only be F8 or B8
    if kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8(): # F8 is always HPA
      # make vgprF8Temp0 always even to use pk instruction later
      if tmpCVTVgpr % 2 == 0:
        vgprF8Temp0 = tmpCVTVgpr
        vgprF8Temp1 = vgprF8Temp0 + 1
        vgprF8Max = vgprF8Temp0 + 2
        vgprF8Min = vgprF8Temp0 + 3
      else:
        vgprF8Max = tmpCVTVgpr
        vgprF8Temp0 = vgprF8Max + 1
        vgprF8Temp1 = vgprF8Max + 2
        vgprF8Min = vgprF8Max + 3

      if kernel["ProblemType"]["Fp32toFp8SWClip"]:
        # set flag of f32 NaN and +/- INF for v_cmp_class
        vgprFp32NanInfFlag = vgprF8Min + 1
        kStr += inst("v_mov_b32", vgpr(vgprFp32NanInfFlag), "0x207", "flag for Nan and +/- inf" )
        # set max/min values for clipping
        if kernel["ProblemType"]["DestDataType"].isFloat8():
          kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x43700000", "save 240.0f as max for clipping" )
          kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC3700000", "save -240.0f as min for clipping" )
        else: #BFloat8
          kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x47600000", "save 57344.0f as max for clipping" )
          kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC7600000", "save -57344`.0f as min for clipping" )

    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      addr = ss.elementAddr[elementIdx].addrDVgpr
      mask = ss.elementMask[elementIdx]
      addrCalc = ss.elementAddr[elementIdx]
      # d1 = element[0]
      # d0 = element[1]
      # vc1 = element[2]
      vc0 = element[3]
      sumIdx = ss.elementSumIdx[elementIdx]

      # apply in-bounds exec mask
      if edge and not kernel["BufferStore"]:
        kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec" )

      # if beta:
      # if GWVW=1 the half path still assumes we have
      # at least two stores so does some combining across VI -
      # for example assuming we can have two elements and can use pk_mul
      # here:
      if interleaveStoreVmcnt: # beta and
        if self.archCaps["SeparateVscnt"]:
          vmcnt = loadsIssued - elementIdx - 1
          vmComment = "{} = {} - {} - 1".format(vmcnt, loadsIssued, elementIdx)
        else:
          waitStoreCnt = storesIssued if not kernel["GroupLoadStore"] else 0
          vmcnt = loadsIssued - elementIdx + waitStoreCnt - 1
          vmComment = "{} = {} - {} + {} - 1".format(vmcnt, loadsIssued, elementIdx, waitStoreCnt)

        maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
        vmcnt = min(vmcnt, maxVmcnt)
        #print "wmvcnt=", vmcnt
        kStr += "\n"
        if not atomicAddC:
          kStr += inst("s_waitcnt", "vmcnt(%u)"%vmcnt, "wait C (interleaved) " + vmComment)

        # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
        # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
        # And this only occurs when beta=true, so case must not be 2 or 3
        assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, \
          "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

      for vi in range(0, gwvw):
        dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
        sumIdxV = ss.elementSumIdx[elementIdx] + vi
        if kernel["ProblemType"]["ComputeDataType"].isHalf():
          if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
            if self.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"]:
              dataV = ss.elementData[elementIdx] + int(vi / 2 * ss.cfg.numVgprsPerDataPerVI)
              # if (vi % 2) == 0:
              #     kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
              #         "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
              # else:
              if (vi % 2) != 0:
                  kStr += inst("v_lshrrev_b32", vgpr(dataV), 16, vgpr(dataV), \
                      "shift 16bit to get next half of packed ValueC")
              # dataV+0 = new c = old c*beta + rC
              kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%(sumIdxV)), vgpr(dataV), vgpr("ValuC+%u"%(sumIdxV)), \
                  "sum*alpha + C*beta")
            elif sumIdxV%2==0 or (not self.ss.cfg.halfDataRegPerVI and gwvw==1):
              # dataV+0 = new c = old c*beta
              # kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
              #     "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
              # dataV+0 = new c = old c*beta + rC
              kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%(sumIdxV//2)), vgpr(dataV), vgpr("ValuC+%u"%(sumIdxV//2)), \
                  "sum*alpha + C*beta")
            else:
              pass # add will have been done previously
          else: # HPA
            # dataV+0 = new c = old c*beta + rC
            # src0 = beta = f32 = opsel 00
            # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
            # src2 = sumIdxV = f32 = opsel 00
            dataCExternal = ss.elementData[elementIdx] + vi//2
            hi16 = (vi + gwvw*vc0) % 2
            # TODO try to replace with add? need opsel for f16 src
            # kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), sgpr("Beta"), \
            kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), 1, \
                vgpr(dataCExternal), vgpr("ValuC+%u"%sumIdxV), \
                "op_sel:[0,%u,0] op_sel_hi:[0,1,0]" % (hi16), \
                "//C*=beta")

        elif kernel["ProblemType"]["ComputeDataType"].isBFloat16():
          if kernel["ProblemType"]["HighPrecisionAccumulate"]:
            # dataV+0 = new c = old c*beta + rC
            # src0 = beta = f32 = opsel 00
            # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
            # src2 = sumIdxV = f32 = opsel 00
            dataCExternal = ss.elementData[elementIdx] + vi//2
            if (vi%2) == 1:
              kStr += inst("v_and_b32", vgpr(tmpVgpr), vgpr(dataCExternal), vgpr(vgprBf16Mask), "convert bf16 to fp32")
            else:
              kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), "16", vgpr(dataCExternal), "convert bf16 to fp32" )
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), vgpr(tmpVgpr), "accum partials")

        # 8bit-Float: dest can only be either F8 or B8
        elif kernel["ProblemType"]["ComputeDataType"].isFloat8() or kernel["ProblemType"]["ComputeDataType"].isBFloat8(): # F8 is always HPA
          # dataV+0 = new c = old c*beta + rC
          dataCExternal = ss.elementData[elementIdx] + vi//4

          #restructuring the code to handle edge case
          if kernel["ProblemType"]["Fp8NoPackUpConversion"] or gwvw == 1:
            byteSel = "src0_sel:BYTE_0" if vi%4==0 else "src0_sel:BYTE_1" if vi%4==1 else "src0_sel:BYTE_2" if vi%4==2 else "src0_sel:BYTE_3"
            if kernel["ProblemType"]["DestDataType"].isFloat8():
              kStr += "v_cvt_f32_fp8 v%u, %s %s   // convert fp8 in lo_byte[0] to f32%s"%(vgprF8Temp0, vgpr(dataCExternal), byteSel, self.endLine)
            else:
              kStr += "v_cvt_f32_bf8 v%u, %s %s   // convert bf8 in lo_byte[0] to f32%s"%(vgprF8Temp0, vgpr(dataCExternal), byteSel, self.endLine)
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprF8Temp0), "accum partials")
          else: ## use packing inst
            if kernel["ProblemType"]["DestDataType"].isFloat8():
              f32Tof8PkInst="V_cvt_pk_f32_fp8"
            else: #bf8
              f32Tof8PkInst="V_cvt_pk_f32_bf8"

            if (vi%2) == 1: ## vi%4 == 1 or vi%4 == 3
              wordSel = "" if vi%4 == 1 else "src0_sel:WORD_1"
              wordTxt = "lo_16" if vi%4==1 else "hi_16"
              kStr += f32Tof8PkInst + " v[%u:%u], %s  %s  // convert two f8 in %s to f32%s"%(vgprF8Temp0, vgprF8Temp1, vgpr(dataCExternal), wordSel, wordTxt, self.endLine)

              kStr += inst("v_add_f32", vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr(vgprF8Temp0), "accum partials")
              kStr += inst("v_add_f32", vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprF8Temp1), "accum partials")

        elif kernel["ProblemType"]["ComputeDataType"].isSingle():
          kStr += inst("v_add_f32", vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), "accum partials")

        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
          # v_mad_i32_i24
          kStr += inst("_v_add_u32", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), vgpr("ValuC+%u"%sumIdxV), \
              "accum partials")

        elif kernel["ProblemType"]["ComputeDataType"].isDouble():
          # dataV+0 = new c = old c*beta
          if not atomicAddC:
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%(sumIdxV*2),2), vgpr("ValuC+%u"%(sumIdxV*2),2), vgpr(dataV+0,2), "accum partials")

        # single precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
          kStr += inst("v_add_f32", vgpr("ValuC+%u"%(sumIdxV*2)), vgpr("ValuC+%u"%(sumIdxV*2)), vgpr(dataV+0), "accum partials real")
          kStr += inst("v_add_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr(dataV+1), "accum partials imag")

        # double precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
          kStr += inst("v_add_f64", vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr(dataV+0,2), "accum partials real")
          kStr += inst("v_add_f64", vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(dataV+2,2), "accum partials imag")

    ########################################
    # AccVgpr write
    if kernel.enabledSetPrioSplitLDS:
      kStr += inst("s_setprio", "0", "")
    if codeAccVgprWrite is not None:
      regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(0, len(batchElements)):
        # loop over scalars within one store instruction
        for vi in range(0, gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            tempStr = str(codeAccVgprWrite.items().pop(0))
            kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
            if kernel["StoreCInUnroll"] and not edge:
              tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
              accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

      if not kernel["MIArchVgpr"]:
        kStr += inst("s_nop 1", "2 wait states required before reading vgpr")

    #kStr += self.bomb(5)
    if self.db["CheckStoreC"]>=0:
      useBuffer = kernel["BufferStore"]
      # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
      # Note - TODO- CheckStoreC also won't work for StoreRemap
      kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      for elementIdx in range(0, len(batchElements)):
        addr = ss.elementAddr[elementIdx].addrDVgpr
        sumIdx = ss.elementSumIdx[elementIdx]

        bps = kernel["ProblemType"]["DestDataType"].numBytes() * gwvw
        if kernel["BufferStore"]:
          addr0 = vgpr(addr)
          addr1 = sgpr("SrdC", 4)
        else:
          addr0 = vgpr(addr,2)
          addr1 = ""

        if kernel["ProblemType"]["DestDataType"].isHalf() or kernel["ProblemType"]["DestDataType"].isBFloat16():
          if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
                      addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=sumIdx%2).toStr()
          else:
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                      addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=0).toStr()
        elif kernel["ProblemType"]["DestDataType"].isInt32() or kernel["ProblemType"]["DestDataType"].isSingle():
          kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                    addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        elif kernel["ProblemType"]["DestDataType"].isDouble() or kernel["ProblemType"]["DestDataType"].isSingleComplex() :
          kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
                    addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
          kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
                    addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
      kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

      # Add checks for expected values:
      kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["CheckStoreC"], "expected value")
      for elementIdx in range(0, len(batchElements)):
        sumIdx = ss.elementSumIdx[elementIdx]
        # Need to fix for other types:
        assert (kernel["ProblemType"]["DestDataType"].isSingle() or kernel["ProblemType"]["DestDataType"].isInt32())
        kStr += self.assert_eq(vgpr(sumIdx), sgpr(tmpS01))


    if edge and (not kernel["BufferStore"]): # atomic or
      # subsequent batch must start with full exec mask
      # BufferStore doesn't need exec since it used buffer range checking when
      # possible
      kStr += inst("s_mov_b{}".format(wavelen), self.exec, -1, "full mask -> exec" )

    if self.db["ConservativeWaitCnt"] & 0x40:
      kStr += "s_barrier // debug\n"
      kStr += inst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      kStr += "s_barrier // debug\n"
    ########################################
    # End Not Atomic
    ########################################

    # return registers to pool:
    lastData = -1
    for elementIdx in range(0, len(batchElements)):
      if not ss.sharedColDVgprs:
        addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
        addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
        self.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.vgprPool.checkIn(addrCVgpr)

      data = ss.elementData[elementIdx]
      if data != 0:
        if data != lastData:
          self.vgprPool.checkIn(data)
        lastData = data

    self.ss.firstBatch = False
    self.ss.checkInTempVgprC()

    if self.serializedStore:
      kStr += inst("s_nop 0", "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst")

    # Update the store cnt to preLoopVmcntDict for Case2/3
    # (No need to update for Case0:'Undefined' or Case4:'OrdNLL_B1_Store')
    if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
      if not self.archCaps["SeparateVscnt"]:
        self.preLoopVmcntDict[self.currPreLoopVmcntCase] += storesIssued

    return kStr

  ##############################################################################
  # Workspace SRD for FixupStep
  ##############################################################################
  def computeWorkspaceSrd(self, kernel, sCtaIdx, tmpSgpr = None):
    kStr = ""

    # Base Address
    kStr += inst("s_mov_b32", sgpr("SrdWS+0"), sgpr("AddressWS+0"), "init SRD base address (lower)" )
    kStr += inst("s_mov_b32", sgpr("SrdWS+1"), sgpr("AddressWS+1"), "init SRD base address (upper) + other fields" )
    kStr += inst("s_mov_b32", sgpr("SrdWS+2"), "BufferOOB", "")
    kStr += inst("s_mov_b32", sgpr("SrdWS+3"), "Srd127_96", "Set bits 127_96 in post-loop SRD")

    tmpSgprRef = None
    if tmpSgpr == None:
      tmpSgprRef = self.getTmpSgpr(1)
      tmpSgpr = tmpSgprRef.idx()

    assert kernel["BufferStore"]
    kStr += "\n"
    kStr += inst("s_mul_i32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]*kernel["MacroTile1"]*self.bpeCinternal), sCtaIdx, "Offset to correct partials tile")
    kStr += inst("s_add_u32",  sgpr("SrdWS+0"), sgpr("SrdWS+0"), sgpr(tmpSgpr), "add lo to SRD")
    kStr += inst("s_addc_u32", sgpr("SrdWS+1"), sgpr("SrdWS+1"), 0, "add hi to SRD")

    return kStr

  ##############################################################################
  # Fixup Step
  ##############################################################################
  def fixupStep(self, kernel, vectorWidths, elements, edges, tmpVgpr, tmpCVTVgpr, sCtaIdx, skStoreLabel):
    kStr = ""

    edgeLabels = {}
    for edge in edges:
      edgeLabels[edge] = self.getNamedLabelUnique("Fixup_E%u" % (1 if edge else 0))
    # branch if Edge0 or Edge1
    if False in edges and True in edges:
      kStr += self.checkIsEdge(kernel, "%s" % edgeLabels[True])

    # by now we either jumped to E1 or stayed at E0
    for edge in edges:
      # write label for batch case
      kStr += "%s:%s"%(edgeLabels[edge], self.endLine)

      PreLoopVmcntCaseStr = ""
      # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
      if self.canOptimizePreLoopLWVmcnt:
        if edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
          self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
        else:
          self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
        PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
          "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
        # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
        # OrdNLL_B1_Store is excluded
        if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
          self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

      edgeI = edge
      #edgeI = True  # set to True to disable vector stores
      gwvw = vectorWidths[edgeI]

      ########################################
      # Calculate Vgprs for Write Batching
      ########################################

      self.ss = self.StoreState(self, kernel, gwvw, edge, True, False, elements[edgeI], isWorkspace=True)

      # how many vgprs are needed for zero elements
      # 2 for addressC in vgpr for addition - already checked out
      # 2 for coord0,1 of thread - already checked out
      # 2 for tmp - already checked out

      # 5 = how many vgprs are needed per element (flat)
      #  - 2 for addr
      #  - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
      #  - if beta gwvw*rpe for new value
      #  - if atomic 2*rpe for old and cmp values

      # print("numVgprsPerAddr=%u, numVgprsPerDataPerVI=%u, numVgprPerValuC=%u"%(self.ss.cfg.numVgprsPerAddr, self.ss.cfg.numVgprsPerDataPerVI, self.ss.cfg.numVgprPerValuC))
      numVgprsPerElement = self.ss.cfg.numVgprPerValuC*gwvw + self.ss.cfg.numVgprsPerAddr + int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw))

      if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
        numVgprsPerElement += self.ss.cfg.numVgprsPerAddr

      #print self.vgprPool.state()
      # Use VGPR up to next occupancy threshold:
      maxVgprs = self.getMaxRegsForOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
                                              self.getLdsSize(kernel), self.agprPool.size(), self.doubleVgpr)
      if self.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
        len(elements[edgeI])
        tl = []
        for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
          tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
        for t in tl:
          self.vgprPool.checkIn(t)
      align = 1
      # align adjustment
      if self.ss.cfg.numVgprsPerAddr > 1:
        align = max(align, self.ss.cfg.numVgprsPerAddr)
      if self.ss.cfg.numVgprPerValuC*gwvw > 1:
        align = max(align, self.ss.cfg.numVgprPerValuC*gwvw)
      if int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
        align = max(align, int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)))
      numVgprAvailable = self.vgprPool.availableBlock(numVgprsPerElement, align)

      # Grow the register pool if needed - we need enough regs for at least one element
      # Unfortunate since this means the write logic is setting the VGPR requirement
      # for the entire kernel but at least we have a functional kernel.
      # Before growing the pool, see if we can shrink the write vector width instead?
      # TODO : the vgprSerial is needed for-ever and if we grow here will split the
      # range of the tmps.  Maybe want to move vgprSerial to first vgpr?

      # TODO: Minimum elems for StoreRemap
      # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
      minElements = 1
      if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
        minElements = 2
      elif kernel["ProblemType"]["DataType"].is8bitFloat():
        minElements = 4

      minNeeded = minElements * numVgprsPerElement
      shrinkDb = 0
      if shrinkDb:
        print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)
      if numVgprAvailable < minNeeded:
        gwvwOrig = gwvw
        currentOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
            self.vgprPool.size(), self.agprPool.size(), self.doubleVgpr)
        futureOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
            self.vgprPool.size() - numVgprAvailable + minNeeded, self.agprPool.size(), self.doubleVgpr)

        if shrinkDb:
          print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
              % (currentOccupancy, futureOccupancy, self.vgprPool.size(), \
                  numVgprAvailable, minElements*numVgprsPerElement))
        if futureOccupancy > currentOccupancy:
          if shrinkDb:
            print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                  (self.kernelName))
            print("   numVgprAvailable=", numVgprAvailable, \
                  "numVgprsPerElement=", numVgprsPerElement, \
                  "gwvw=", gwvw)
        elif gwvw != gwvwOrig:
          self.ss.gwvw = gwvw # make both representations consistent
          if shrinkDb:
            tPrint(3, "info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                % (self.kernelName, gwvwOrig, gwvw, currentOccupancy))

        if numVgprAvailable < minElements*numVgprsPerElement:
          tPrint(3, "info: growing pool += %d * %d for GlobalWrite\n" \
              % (minElements,numVgprsPerElement))
          tPrint(3, self.vgprPool.state())
          tl = []
          for i in range(0,minElements):
            tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
          for t in tl:
            self.vgprPool.checkIn(t)
          numVgprAvailable = self.vgprPool.available()
          tPrint(3, self.vgprPool.state())

      # print("NumVgprAvailable", numVgprAvailable)
      if numVgprsPerElement:
        numElementsPerBatch = numVgprAvailable // numVgprsPerElement
      else:
        numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

      assert(self.numVgprValuC % gwvw == 0) # sanity check

      numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

      if shrinkDb:
        print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", self.ss.cfg.numElementsPerBatchLimitedBySgprs, \
            "WARNING" if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
      if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
        numElementsPerBatch = self.ss.cfg.numElementsPerBatchLimitedBySgprs

      # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
      if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
        # only do an even number of halves - since these share hi/lo pieces of some registers?
        if numElementsPerBatch > 1:
          numElementsPerBatch = int(numElementsPerBatch/2)*2
        elif not kernel["EnableMatrixInstruction"]:
          # (excluding MFMA+LSU case. It can work without an issue)
          # The globalWriteBatch routine below can't handle odd elements per batch
          # and 0 elements per batch is illegal.
          # so if we don't have *GPR resources to handle a larger batch then need
          # to mark overflowedResources rather than generate a kernel that won't work.
          # It might be possible to fix globalWriteBatch to handle this case but these
          # are likely to be low-performing so likely not worth optimizing.
          if shrinkDb:
            print("WARNING: half requires at least two elements per batch")
          self.overflowedResources = 3
      #elif kernel["ProblemType"]["DataType"].is8bitFloat():
      #  if numElementsPerBatch > 1:
      #    numElementsPerBatch = int(numElementsPerBatch/4)*4

      assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%self.kernelName

      # if no atomics and no edge, then write whole vectors
      # ERROR commented out in globalWriteELements, causes numVectorsPerBatch to not be int
      # if not edge: # not atomic and
      #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
      #  #print "  NumVectorsPerBatch", numVectorsPerBatch
      #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
      numBatches = max(1, ceil_divide(len(elements[edgeI]),numElementsPerBatch))

      numSgprs = self.ss.cfg.numTempSgprPerBatch + self.ss.cfg.numMaskSgprPerBatch + self.ss.cfg.numMaskSgprPerElement * numElementsPerBatch

      if self.db["PrintStoreRegisterDb"]:
        print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
        print ("numSgprs=", numSgprs, "sgprPool.size()=", self.sgprPool.size(), "numTempSgprPerBatch=", self.ss.cfg.numTempSgprPerBatch,
                "numMaskSgprPerBatch=", self.ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", self.ss.cfg.numMaskSgprPerElement)
        print(self.sgprPool.state())
      kStr += self.comment("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
          (edgeI, numSgprs, self.ss.cfg.numTempSgprPerBatch, self.ss.cfg.numMaskSgprPerBatch, self.ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
      #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
      # so if we don't have *GPR resources to handle a larger batch then need
      # to mark overflowedResources rather than generate a kernel that won't work.

      tmpSgprRef = self.getTmpSgpr(numSgprs, 2)
      tmpSgpr = tmpSgprRef.idx()

      elementSgprs = tmpSgpr + self.ss.cfg.numTempSgprPerBatch

      codeAccVgprRead = deepcopy(self.codeAccVgprRead) if self.serializedStore else None
      codeAccVgprWrite = deepcopy(self.codeAccVgprWrite) if self.serializedStore else None

      kStr += self.computeWorkspaceSrd(kernel, sgpr(sCtaIdx), tmpSgpr)

      for batchIdx in range(0, numBatches):
        elementStartIdx = batchIdx * numElementsPerBatch
        elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
        elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
        #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
        # elementVgprs can be large and should be perfectly tuned to the number of available
        # VGPRS.  We do not want to accidentally overflow and grow the pool here:

        kStr += self.fixupBatch(kernel, self.ss, batchIdx, edge, gwvw, \
            elementsThisBatch, self.coord0, self.coord1, self.addrD, self.addrC, \
            tmpVgpr, tmpCVTVgpr, \
            elementSgprs, tmpSgpr, codeAccVgprRead, codeAccVgprWrite)
      # delay PreLoopVmcntCase code after globalWrite
      if self.canOptimizePreLoopLWVmcnt:
        kStr += PreLoopVmcntCaseStr

      del self.ss

      # Finish one write path, reset currPreLoopVmcntCase to Undefined
      self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined

      # kStr += inst("s_branch", skStoreLabel, "jump to store")

    return kStr

  def writePartials(self, kernel, vectorWidths, elements, edges, atomic, tmpVgpr, tmpCVTVgpr, isOptNLL, endLabel):
    kStr = ""

    partialsLabels = {}
    for edge in edges:
      partialsLabels[edge] = self.getNamedLabelUnique("GW_Partials_E%u" % ( 1 if edge else 0) )

    if False in edges and True in edges:
      kStr += self.checkIsEdge(kernel, "%s" % partialsLabels[True])

    for edge in edges:
      kStr += "%s:%s"%(partialsLabels[edge], self.endLine)
      kStr += self.computeWorkspaceSrd(kernel, sgpr("StreamKIdx"))
      kStr += self.partialsWriteProcedure(kernel, vectorWidths, elements, False, False, edge, atomic, tmpVgpr, tmpCVTVgpr, isOptNLL, endLabel)

    return kStr


  ##############################################################################
  # Partials Write Procedure
  ##############################################################################
  def partialsWriteProcedure(self, kernel, vectorWidths, elements, alpha, beta, edge, atomic, tmpVgpr, tmpCVTVgpr, isOptNLL, endLabel):
    kStr = ""

    PreLoopVmcntCaseStr = ""
    # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
    if self.canOptimizePreLoopLWVmcnt:
      if beta:
        self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_B1_Store
      elif edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
        self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
      else:
        self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
      PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
        "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
      # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
      # OrdNLL_B1_Store is excluded
      if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

    edgeI = edge
    #edgeI = True  # set to True to disable vector stores
    gwvw = vectorWidths[edgeI]
    #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

    ########################################
    # Calculate Vgprs for Write Batching
    ########################################

    self.ss = self.StoreState(self, kernel, gwvw, edge, beta, atomic, elements[edgeI], isWorkspace=True)

    # how many vgprs are needed for zero elements
    # 2 for addressC in vgpr for addition - already checked out
    # 2 for coord0,1 of thread - already checked out
    # 2 for tmp - already checked out

    # 5 = how many vgprs are needed per element (flat)
    #  - 2 for addr
    #  - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
    #  - if beta gwvw*rpe for new value
    #  - if atomic 2*rpe for old and cmp values

    # print("numVgprsPerAddr=%u, numVgprsPerDataPerVI=%u, numVgprPerValuC=%u"%(self.ss.cfg.numVgprsPerAddr, self.ss.cfg.numVgprsPerDataPerVI, self.ss.cfg.numVgprPerValuC))
    numVgprsPerElement = self.ss.cfg.numVgprPerValuC*gwvw + self.ss.cfg.numVgprsPerAddr + int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw))

    if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
      numVgprsPerElement += self.ss.cfg.numVgprsPerAddr

    #print self.vgprPool.state()
    # Use VGPR up to next occupancy threshold:
    maxVgprs = self.getMaxRegsForOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
                                            self.getLdsSize(kernel), self.agprPool.size(), self.doubleVgpr)
    if self.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
      len(elements[edgeI])
      tl = []
      for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
        tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
      for t in tl:
        self.vgprPool.checkIn(t)
    align = 1
    # align adjustment
    if self.ss.cfg.numVgprsPerAddr > 1:
      align = max(align, self.ss.cfg.numVgprsPerAddr)
    if self.ss.cfg.numVgprPerValuC*gwvw > 1:
      align = max(align, self.ss.cfg.numVgprPerValuC*gwvw)
    if int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
      align = max(align, int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)))
    numVgprAvailable = self.vgprPool.availableBlock(numVgprsPerElement, align)

    # Grow the register pool if needed - we need enough regs for at least one element
    # Unfortunate since this means the write logic is setting the VGPR requirement
    # for the entire kernel but at least we have a functional kernel.
    # Before growing the pool, see if we can shrink the write vector width instead?
    # TODO : the vgprSerial is needed for-ever and if we grow here will split the
    # range of the tmps.  Maybe want to move vgprSerial to first vgpr?

    # TODO: Minimum elems for StoreRemap
    # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
    minElements = 1
    if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
      minElements = 2
    elif kernel["ProblemType"]["DataType"].is8bitFloat():
      minElements = 4

    minNeeded = minElements * numVgprsPerElement
    shrinkDb = 0
    if shrinkDb:
      print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)
    if numVgprAvailable < minNeeded:
      gwvwOrig = gwvw
      currentOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
          self.vgprPool.size(), self.agprPool.size(), self.doubleVgpr)
      futureOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
          self.vgprPool.size() - numVgprAvailable + minNeeded, self.agprPool.size(), self.doubleVgpr)

      if shrinkDb:
        print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
            % (currentOccupancy, futureOccupancy, self.vgprPool.size(), \
                numVgprAvailable, minElements*numVgprsPerElement))
      if futureOccupancy > currentOccupancy:
        if shrinkDb:
          print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                (self.kernelName))
          print("   numVgprAvailable=", numVgprAvailable, \
                "numVgprsPerElement=", numVgprsPerElement, "atomic=", atomic, \
                "beta=", beta, "gwvw=", gwvw)
      elif gwvw != gwvwOrig:
        self.ss.gwvw = gwvw # make both representations consistent
        if shrinkDb:
          tPrint(3, "info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
              % (self.kernelName, gwvwOrig, gwvw, currentOccupancy))

      if numVgprAvailable < minElements*numVgprsPerElement:
        tPrint(3, "info: growing pool += %d * %d for GlobalWrite\n" \
            % (minElements,numVgprsPerElement))
        tPrint(3, self.vgprPool.state())
        tl = []
        for i in range(0,minElements):
          tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
        for t in tl:
          self.vgprPool.checkIn(t)
        numVgprAvailable = self.vgprPool.available()
        tPrint(3, self.vgprPool.state())

    # set atomicW after we potentially resize GWVW
    atomicW = min(gwvw, kernel["VectorAtomicWidth"])

    # print("NumVgprAvailable", numVgprAvailable)
    if numVgprsPerElement:
      numElementsPerBatch = numVgprAvailable // numVgprsPerElement
    else:
      numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

    assert(self.numVgprValuC % gwvw == 0) # sanity check

    numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

    if shrinkDb:
      print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", self.ss.cfg.numElementsPerBatchLimitedBySgprs, \
          "WARNING" if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
    if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
      numElementsPerBatch = self.ss.cfg.numElementsPerBatchLimitedBySgprs

    # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
    if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
      # only do an even number of halves - since these share hi/lo pieces of some registers?
      if numElementsPerBatch > 1:
        numElementsPerBatch = int(numElementsPerBatch/2)*2
      elif not kernel["EnableMatrixInstruction"]:
        # (excluding MFMA+LSU case. It can work without an issue)
        # The globalWriteBatch routine below can't handle odd elements per batch
        # and 0 elements per batch is illegal.
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        # It might be possible to fix globalWriteBatch to handle this case but these
        # are likely to be low-performing so likely not worth optimizing.
        if shrinkDb:
          print("WARNING: half requires at least two elements per batch")
        self.overflowedResources = 3
    #elif kernel["ProblemType"]["DataType"].is8bitFloat():
    #  if numElementsPerBatch > 1:
    #    numElementsPerBatch = int(numElementsPerBatch/4)*4

    assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%self.kernelName

    #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
    if atomic and (self.ss.optSingleColVgpr or self.ss.optSharedColVgpr):
      # hack to avoid re-using address vgpr across rows
      # atomics need to perform several memory operations
      # if the batch spans multiple rows, need multiple address vgpr
      # which is not currently supported in the two opt*ColVgpr modes
      firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
      numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

    numBatches = max(1, ceil_divide(len(elements[edgeI]),numElementsPerBatch))

    numSgprs = self.ss.cfg.numTempSgprPerBatch + self.ss.cfg.numMaskSgprPerBatch + self.ss.cfg.numMaskSgprPerElement * numElementsPerBatch

    if self.db["PrintStoreRegisterDb"]:
      print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
      print ("numSgprs=", numSgprs, "sgprPool.size()=", self.sgprPool.size(), "numTempSgprPerBatch=", self.ss.cfg.numTempSgprPerBatch,
              "numMaskSgprPerBatch=", self.ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", self.ss.cfg.numMaskSgprPerElement)
      print(self.sgprPool.state())
    kStr += self.comment("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
        (edgeI, numSgprs, self.ss.cfg.numTempSgprPerBatch, self.ss.cfg.numMaskSgprPerBatch, self.ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
    #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
    # so if we don't have *GPR resources to handle a larger batch then need
    # to mark overflowedResources rather than generate a kernel that won't work.
    tmpSgpr = self.getTmpSgpr(numSgprs, 2).idx()

    elementSgprs = tmpSgpr + self.ss.cfg.numTempSgprPerBatch

    codeAccVgprRead = deepcopy(self.codeAccVgprRead) if self.serializedStore else None
    codeMulAlpha    = deepcopy(self.codeMulAlpha) if self.serializedStore else None

    self.alphaBeforeLoadC = False
    useCodeMulAlpha =  kernel["MIArchVgpr"] and alpha and not (kernel["GlobalSplitU"] > 1)
    if useCodeMulAlpha: # do not set codeAccVgprRead=None if GSU>1
      codeAccVgprRead = None

      #Only apply when 2 wave optimization features are enabled
      if (kernel["StorePriorityOpt"] or kernel["StoreSyncOpt"]) and beta:
        self.alphaBeforeLoadC = True
    else:
      codeMulAlpha = None

    for batchIdx in range(0, numBatches):
      elementStartIdx = batchIdx * numElementsPerBatch
      elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
      elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
      #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
      # elementVgprs can be large and should be perfectly tuned to the number of available
      # VGPRS.  We do not want to accidentally overflow and grow the pool here:

      kStr += self.partialsWriteBatch(kernel, self.ss, batchIdx, alpha, beta, edge, atomic, gwvw, atomicW, \
          elementsThisBatch, self.coord0, self.coord1, self.addrD, self.addrC, \
          tmpVgpr, tmpCVTVgpr, \
          elementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, isOptNLL)
    # delay PreLoopVmcntCase code after globalWrite
    if self.canOptimizePreLoopLWVmcnt:
      kStr += PreLoopVmcntCaseStr

    # Set flag
    kStr += inst("s_waitcnt", "vmcnt(0)", "wait for data store")
    kStr += inst("s_barrier", "store all data before setting flag")
    kStr += inst("s_lshl_b32", sgpr(tmpSgpr), sgpr("StreamKIdx"), log2(4), "flag offset based on CTA index")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr+2), 1, "flag data")
    kStr += inst("s_store_dword", sgpr(tmpSgpr+2), sgpr("AddressFlags", 2), sgpr(tmpSgpr), "glc", "set flag")
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for flag") # TODO just for testing

    # TODO - if this is the last tile, don't need to jump to next instruction
    # NOTE: in SR kernel, we need long branch since PRNG explodes the line of codes
    if kernel["ProblemType"]["StochasticRounding"]:  # in-device RND
      endLabelName = "label_%s"%(endLabel)
      kStr += self.longBranchPositive(endLabelName)
    else:
      kStr += inst("s_branch", "label_%s"%endLabel, "jump to end")
    del self.ss

    # Finish one write path, reset currPreLoopVmcntCase to Undefined
    self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined

    return kStr


  ##############################################################################
  # Partials Write Batch
  ##############################################################################
  def partialsWriteBatch(self, kernel, ss, batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
      batchElements, coord0, coord1, addrD, addrC, \
      tmpVgpr, tmpCVTVgpr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, isOptNLL):
    kStr = ""

    kStr += self.comment1("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
              (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

    if kernel["StoreSyncOpt"]:
      kStr += "s_sleep %d // optimization: sync and wait\n" %(kernel["StoreSyncOpt"]-1)
      kStr += "s_barrier\n"

    # comment tt1, tt0, vc1, vc0
    # tt = thread tile, vc=vector component
    commentStr = "Partials Write%s%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Alpha" if applyAlpha else "", " Beta" if beta else "", " Edge" if edge else "", batchIdx)
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      commentStr += "(%u,%u,%u,%u:vw%u%s)" % \
        (element[0], element[1], element[2], element[3], gwvw,
         ":vaw:%u"%atomicW if atomic else "")
      if elementIdx < len(batchElements)-1:
        commentStr += "; "
    kStr += self.comment3(commentStr)

    # allow expanding vgpr pool for OptNLL
    preventOverflow = (not isOptNLL)
    ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, preventOverflow=preventOverflow, isWorkspace=True)

    storesIssued = 0
    tmpS01 = tmpSgpr # scratch sgprs

    ########################################
    # calculate addr and masks
    kStr += self.comment("calc coords, apply mask, and issue loads (if necessary)")
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.db["ConservativeWaitCnt"] & 0x10:
      kStr += "s_barrier // debug\n"
      kStr += inst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      kStr += "s_barrier // debug\n"
    if not edge and self.db["ForceEdgeStores"]>=2:
      kStr += self.bomb() # should not get here
    if edge and self.db["AssertNoEdge"]:
      kStr += self.bomb() # should not get here

    ## create code Module to push mov vgpr,acc instructions
    if kernel["StoreCInUnroll"] and not edge:
      accVgprRead = Code.Module("movaccVgpr")
      self.StoreCUnrollLoadCWaitComment = "waitcnt for LoadC" # this will be used later to identify waitcnt for loadC

    ########################################
    # AccVgpr read
    if kernel.enabledSetPrioSplitLDS:
      kStr += inst("s_setprio", "0", "")
    if codeAccVgprRead is not None:
      regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(0, len(batchElements)):
        # loop over scalars within one store instruction
        for vi in range(0, gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            tempStr = str(codeAccVgprRead.items().pop(0))
            kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
            if kernel["StoreCInUnroll"] and not edge:
              tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
              accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

      if not kernel["MIArchVgpr"]:
        kStr += inst("s_nop 1", "2 wait states required before reading vgpr")

    ########################################
    # Not Atomic
    ########################################
    # else:
    # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
    for elementIdx in range(0, len(batchElements)):
      for vi in range(0, gwvw):
        sumIdxV = ss.elementSumIdx[elementIdx] + vi
        # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
        if kernel["ProblemType"]["ComputeDataType"].isInt32() or \
            kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
            if self.db["ForceExpectedValue"]:
              kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
            if self.db["ForceVSerial"]:
              kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
            if self.db["CheckValueC"]:
              kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
              kStr += self.assert_eq(vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01))

    kStr += self.comment("apply mask, calc new C and issue writes")
    #kStr += self.bomb() # can see store addresses just before the store inst

    if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
      vgprBf16Temp = tmpCVTVgpr
      vgprBf16Mask = vgprBf16Temp + 1
      vgprFp32Nan = vgprBf16Temp + 2
      vgprBf16Inc = vgprBf16Temp + 3
      kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
      kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
      kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )

    # DestDataType for 8bit Float can only be F8 or B8
    if kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8(): # F8 is always HPA
      # make vgprF8Temp0 always even to use pk instruction later
      if tmpCVTVgpr % 2 == 0:
        vgprF8Temp0 = tmpCVTVgpr
        vgprF8Max = vgprF8Temp0 + 2
        vgprF8Min = vgprF8Temp0 + 3
      else:
        vgprF8Max = tmpCVTVgpr
        vgprF8Temp0 = vgprF8Max + 1
        vgprF8Min = vgprF8Max + 3

      if kernel["ProblemType"]["Fp32toFp8SWClip"]:
        # set flag of f32 NaN and +/- INF for v_cmp_class
        vgprFp32NanInfFlag = vgprF8Min + 1
        kStr += inst("v_mov_b32", vgpr(vgprFp32NanInfFlag), "0x207", "flag for Nan and +/- inf" )
        # set max/min values for clipping
        if kernel["ProblemType"]["DestDataType"].isFloat8():
          kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x43700000", "save 240.0f as max for clipping" )
          kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC3700000", "save -240.0f as min for clipping" )
        else: #BFloat8
          kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x47600000", "save 57344.0f as max for clipping" )
          kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC7600000", "save -57344`.0f as min for clipping" )

    storeCode = ""
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      addr = ss.elementAddr[elementIdx].addrDVgpr
      addrCalc = ss.elementAddr[elementIdx]
      sumIdx = ss.elementSumIdx[elementIdx]

      storeWidth = kernel["StoreVectorWidth"]
      # storeWidth = 2
      if batchIdx == 0 and elementIdx == 0:
        kStr += staticMultiply(vgpr(addr), vgpr("Serial"), storeWidth * self.bpeCinternal, sgpr(tmpS01))
        # kStr += inst("v_mul_lo_u32", , "Partials buffer address")
        kStr += inst("s_mov_b32", sgpr(tmpS01), 0, "Init sgpr offset")
      else:
        increment = (kernel["WavefrontSize"] * 4) * storeWidth * self.bpeCinternal
        kStr += inst("s_add_u32", sgpr(tmpS01), sgpr(tmpS01), increment, "Inc sgpr offset")

      # TODO StreamK need this packing code???
      # if self.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"] and kernel["ProblemType"]["DestDataType"].isHalf() and (not kernel["ProblemType"]["HighPrecisionAccumulate"]):
      #   for vi in range(0, gwvw):
      #     sumIdxV = ss.elementSumIdx[elementIdx] + vi
      #     if vi%2 == 1:
      #       d = ss.elementSumIdx[elementIdx] + vi//2
      #       kStr += inst("v_pack_b32_f16", vgpr(d), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%sumIdxV), "Pack with neighbor" )

      # if not kernel["StoreRemapVectorWidth"]:
      tmpStoreCode = self.addStore(kernel, ss, addrCalc, sumIdx, tmpS01, edge, 'WS', sgpr(tmpS01))
      if kernel["GroupLoadStore"]:
        storeCode += tmpStoreCode
      else:
        kStr += tmpStoreCode
      storesIssued += 1

    kStr += storeCode

    # return registers to pool:
    lastData = -1
    for elementIdx in range(0, len(batchElements)):
      if not ss.sharedColDVgprs:
        addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
        addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
        self.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.vgprPool.checkIn(addrCVgpr)

      data = ss.elementData[elementIdx]
      if data != 0:
        if data != lastData:
          self.vgprPool.checkIn(data)
        lastData = data

    self.ss.firstBatch = False
    self.ss.checkInTempVgprC()

    if self.serializedStore:
      kStr += inst("s_nop 0", "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst")

    # Update the store cnt to preLoopVmcntDict for Case2/3
    # (No need to update for Case0:'Undefined' or Case4:'OrdNLL_B1_Store')
    if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
      if not self.archCaps["SeparateVscnt"]:
        self.preLoopVmcntDict[self.currPreLoopVmcntCase] += storesIssued

    return kStr

  ##############################################################################
  # Global Write Procedure
  ##############################################################################
  def globalWriteProcedure(self, kernel, vectorWidths, elements, alpha, beta, edge, atomic, tmpVgpr, tmpCVTVgpr, isOptNLL, endLabel):
    kStr = ""

    PreLoopVmcntCaseStr = ""
    # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
    if self.canOptimizePreLoopLWVmcnt:
      if beta:
        self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_B1_Store
      elif edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
        self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
      else:
        self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
      PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
        "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
      # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
      # OrdNLL_B1_Store is excluded
      if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

    # for storeRemap edge case, non-beta still can enable vector stores
    if kernel["StoreRemapVectorWidth"] and not beta:
      edgeI = False
    else:
      edgeI = edge
    #edgeI = True  # set to True to disable vector stores
    # print(edgeI)
    # print(vectorWidths)
    gwvw = vectorWidths[edgeI]
    #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

    ########################################
    # Calculate Vgprs for Write Batching
    ########################################

    self.ss = self.StoreState(self, kernel, gwvw, edge, beta, atomic, elements[edgeI])

    # how many vgprs are needed for zero elements
    # 2 for addressC in vgpr for addition - already checked out
    # 2 for coord0,1 of thread - already checked out
    # 2 for tmp - already checked out

    # 5 = how many vgprs are needed per element (flat)
    #  - 2 for addr
    #  - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
    #  - if beta gwvw*rpe for new value
    #  - if atomic 2*rpe for old and cmp values

    # print("numVgprsPerAddr=%u, numVgprsPerDataPerVI=%u, numVgprPerValuC=%u"%(self.ss.cfg.numVgprsPerAddr, self.ss.cfg.numVgprsPerDataPerVI, self.ss.cfg.numVgprPerValuC))
    numVgprsPerElement = self.ss.cfg.numVgprPerValuC*gwvw + self.ss.cfg.numVgprsPerAddr + int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw))

    if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
      numVgprsPerElement += self.ss.cfg.numVgprsPerAddr

    #print self.vgprPool.state()
    # Use VGPR up to next occupancy threshold:
    maxVgprs = self.getMaxRegsForOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
                                            self.getLdsSize(kernel), self.agprPool.size(), self.doubleVgpr)
    if self.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
      len(elements[edgeI])
      tl = []
      for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
        tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
      for t in tl:
        self.vgprPool.checkIn(t)
    align = 1
    # align adjustment
    if self.ss.cfg.numVgprsPerAddr > 1:
      align = max(align, self.ss.cfg.numVgprsPerAddr)
    if self.ss.cfg.numVgprPerValuC*gwvw > 1:
      align = max(align, self.ss.cfg.numVgprPerValuC*gwvw)
    if int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
      align = max(align, int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)))
    numVgprAvailable = self.vgprPool.availableBlock(numVgprsPerElement, align)

    # Grow the register pool if needed - we need enough regs for at least one element
    # Unfortunate since this means the write logic is setting the VGPR requirement
    # for the entire kernel but at least we have a functional kernel.
    # Before growing the pool, see if we can shrink the write vector width instead?
    # TODO : the vgprSerial is needed for-ever and if we grow here will split the
    # range of the tmps.  Maybe want to move vgprSerial to first vgpr?

    # TODO: Minimum elems for StoreRemap
    # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
    minElements = 1
    if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
      minElements = 2
    elif kernel["ProblemType"]["DataType"].is8bitFloat():
      minElements = 4

    minNeeded = minElements * numVgprsPerElement
    shrinkDb = 0
    if shrinkDb:
      print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)
    if numVgprAvailable < minNeeded:
      gwvwOrig = gwvw
      currentOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
          self.vgprPool.size(), self.agprPool.size(), self.doubleVgpr)
      futureOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
          self.vgprPool.size() - numVgprAvailable + minNeeded, self.agprPool.size(), self.doubleVgpr)

      if shrinkDb:
        print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
            % (currentOccupancy, futureOccupancy, self.vgprPool.size(), \
                numVgprAvailable, minElements*numVgprsPerElement))
      if futureOccupancy > currentOccupancy:
        if shrinkDb:
          print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                (self.kernelName))
          print("   numVgprAvailable=", numVgprAvailable, \
                "numVgprsPerElement=", numVgprsPerElement, "atomic=", atomic, \
                "beta=", beta, "gwvw=", gwvw)
      elif gwvw != gwvwOrig:
        self.ss.gwvw = gwvw # make both representations consistent
        if shrinkDb:
          tPrint(3, "info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
              % (self.kernelName, gwvwOrig, gwvw, currentOccupancy))

      if numVgprAvailable < minElements*numVgprsPerElement:
        tPrint(3, "info: growing pool += %d * %d for GlobalWrite\n" \
            % (minElements,numVgprsPerElement))
        tPrint(3, self.vgprPool.state())
        tl = []
        for i in range(0,minElements):
          tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
        for t in tl:
          self.vgprPool.checkIn(t)
        numVgprAvailable = self.vgprPool.available()
        tPrint(3, self.vgprPool.state())

    # set atomicW after we potentially resize GWVW
    atomicW = min(gwvw, kernel["VectorAtomicWidth"])

    # print("NumVgprAvailable", numVgprAvailable)
    if numVgprsPerElement:
      numElementsPerBatch = numVgprAvailable // numVgprsPerElement
    else:
      numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

    assert(self.numVgprValuC % gwvw == 0) # sanity check

    numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

    if shrinkDb:
      print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", self.ss.cfg.numElementsPerBatchLimitedBySgprs, \
          "WARNING" if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
    if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
      numElementsPerBatch = self.ss.cfg.numElementsPerBatchLimitedBySgprs

    # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
    if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
      # only do an even number of halves - since these share hi/lo pieces of some registers?
      if numElementsPerBatch > 1:
        numElementsPerBatch = int(numElementsPerBatch/2)*2
      elif not kernel["EnableMatrixInstruction"]:
        # (excluding MFMA+LSU case. It can work without an issue)
        # The globalWriteBatch routine below can't handle odd elements per batch
        # and 0 elements per batch is illegal.
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        # It might be possible to fix globalWriteBatch to handle this case but these
        # are likely to be low-performing so likely not worth optimizing.
        if shrinkDb:
          print("WARNING: half requires at least two elements per batch")
        self.overflowedResources = 3
    #elif kernel["ProblemType"]["DataType"].is8bitFloat():
    #  if numElementsPerBatch > 1:
    #    numElementsPerBatch = int(numElementsPerBatch/4)*4

    assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%self.kernelName

    #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
    if atomic and (self.ss.optSingleColVgpr or self.ss.optSharedColVgpr):
      # hack to avoid re-using address vgpr across rows
      # atomics need to perform several memory operations
      # if the batch spans multiple rows, need multiple address vgpr
      # which is not currently supported in the two opt*ColVgpr modes
      firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
      numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

    # check best numElementsPerBatch to handle a column block
    # elements of column block must be multiple size of numElementsPerBatch
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
    numBatches = max(1, ceil_divide(len(elements[edgeI]),numElementsPerBatch))

    numSgprs = self.ss.cfg.numTempSgprPerBatch + self.ss.cfg.numMaskSgprPerBatch + self.ss.cfg.numMaskSgprPerElement * numElementsPerBatch

    if self.db["PrintStoreRegisterDb"]:
      print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
      print ("numSgprs=", numSgprs, "sgprPool.size()=", self.sgprPool.size(), "numTempSgprPerBatch=", self.ss.cfg.numTempSgprPerBatch,
              "numMaskSgprPerBatch=", self.ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", self.ss.cfg.numMaskSgprPerElement)
      print(self.sgprPool.state())
    kStr += self.comment("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
        (edgeI, numSgprs, self.ss.cfg.numTempSgprPerBatch, self.ss.cfg.numMaskSgprPerBatch, self.ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
    #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
    # so if we don't have *GPR resources to handle a larger batch then need
    # to mark overflowedResources rather than generate a kernel that won't work.
    tmpSgpr = self.getTmpSgpr(numSgprs, 2).idx()

    elementSgprs = tmpSgpr + self.ss.cfg.numTempSgprPerBatch

    codeAccVgprRead = deepcopy(self.codeAccVgprRead) if self.serializedStore else None
    codeMulAlpha    = deepcopy(self.codeMulAlpha) if self.serializedStore else None

    self.alphaBeforeLoadC = False
    useCodeMulAlpha =  kernel["MIArchVgpr"] and alpha and not (kernel["GlobalSplitU"] > 1)
    if useCodeMulAlpha: # do not set codeAccVgprRead=None if GSU>1
      codeAccVgprRead = None

      #Only apply when 2 wave optimization features are enabled
      if (kernel["StorePriorityOpt"] or kernel["StoreSyncOpt"]) and beta:
        self.alphaBeforeLoadC = True
    else:
      codeMulAlpha = None

    for batchIdx in range(0, numBatches):
      elementStartIdx = batchIdx * numElementsPerBatch
      elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
      elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
      #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
      # elementVgprs can be large and should be perfectly tuned to the number of available
      # VGPRS.  We do not want to accidentally overflow and grow the pool here:

      if kernel["StoreRemapVectorWidth"]:
        #Indication if this batch is last batch for this column block shape
        self.StoreRemapLastBatch = 1 if (batchIdx+1) % nBatchesPerRow == 0 else 0

      kStr += self.globalWriteBatch(kernel, self.ss, batchIdx, alpha, beta, edge, atomic, gwvw, atomicW, \
          elementsThisBatch, self.coord0, self.coord1, self.addrD, self.addrC, \
          tmpVgpr, tmpCVTVgpr, \
          elementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, isOptNLL)
    # delay PreLoopVmcntCase code after globalWrite
    if self.canOptimizePreLoopLWVmcnt:
      kStr += PreLoopVmcntCaseStr

    # TODO - if this is the last tile, don't need to jump to next instruction
    # NOTE: in SR kernel, we need long branch since PRNG explodes the line of codes
    if kernel["ProblemType"]["StochasticRounding"]:  # in-device RND
      endLabelName = "label_%s"%(endLabel)
      kStr += self.longBranchPositive(endLabelName)
    else:
      kStr += inst("s_branch", "label_%s"%endLabel, "jump to end")
    del self.ss

    # Finish one write path, reset currPreLoopVmcntCase to Undefined
    self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined

    return kStr


  ##############################################################################
  # Global Write Elements
  ##############################################################################
  def globalWriteElements(self, kernel, vectorWidths, elements,
                          applyAlpha=True, # defaults to generating *=alpha codes
                          betas=None, # if left unspecified, then let global parameter decide
                          edges=None,
                          isOptNLL=False): # if OptNLL or not (for StoreCInUnroll)
    if not kernel["StoreCInUnroll"]:
      if not self.do["PostLoop"]: return ""
    kStr = ""
    atomic = (kernel["GlobalSplitU"] > 1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')
    atomic = atomic or kernel["StreamKAtomic"] == 1
    useCodeMulAlpha =  kernel["MIArchVgpr"] and applyAlpha and not (kernel["GlobalSplitU"] > 1)

    # write possibilities and labels
    # if beta/edge combo not specified fall back to global param definition
    if betas is None:
      hasBeta = kernel["ProblemType"]["UseBeta"] and (kernel["GlobalSplitU"] == 1)
      betas = [False, True] if hasBeta else [False]
    if edges is None:
      edges = [False, True] if self.do["EdgeWrite"] else [False]
    writeLabels = {}
    for beta in betas:
      writeLabels[beta] = {}
      for edge in edges:
        writeLabels[beta]["EdgeCheck0"] = self.getNamedLabelUnique("GW_B%u_E%u_EdgeCheck0" % ( 1 if beta else 0, 1 if edge else 0) )
        writeLabels[beta]["EdgeCheck1"] = self.getNamedLabelUnique("GW_B%u_E%u_EdgeCheck1" % ( 1 if beta else 0, 1 if edge else 0) )
        writeLabels[beta][edge] = self.getNamedLabelUnique("GW_B%u_E%u" % ( 1 if beta else 0, 1 if edge else 0) )
      if not beta:
        betaLabel = self.getNamedLabelUnique("GW_Beta")
    endLabel = self.getNamedLabelUnique("GW_End")

    # Layout
    """
    if B1 goto label_B1
    if E1 goto label_B0_E1
    label_B0_E0:
    writes
    goto label_End
    label_B0_E1:
    writes
    goto label_End
    label_B1:
    if E1 goto label_B1_E1
    label_B1_E0:
    writes
    goto label_End
    label_B1_E1:
    writes
    goto label_End
    label_End
    """
    self.betaVgpr = None

    ########################################
    # Vgprs
    if kernel["BufferStore"]:
      numTmpVgpr = 2
      if len(kernel["PackedC0IndicesX"]) > 1:
        numTmpVgpr += 1
    else:
      numTmpVgpr = 2 + 3 # GLOBAL_OFFSET_C needs 3, plus 2 tmps?
    if useCodeMulAlpha and kernel["ProblemType"]["DataType"].isComplex():
      # codeMulAlpha and complex caes, use tmpVgpr for alpha calculation
      numTmpVgpr = max(numTmpVgpr, kernel["ProblemType"]["DataType"].numRegisters())
    tmpVgpr = self.vgprPool.checkOutAligned(numTmpVgpr, 2, "store tmps")

    isHpaBF16 = kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]
    isHpaF8 = kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8() # F8 is always HPA

    # need temp vgpr both for bf16 and f8
    if isHpaF8:
      rcnt = 4
      if kernel["ProblemType"]["Fp32toFp8SWClip"]:
        rcnt += 1
      if kernel["ProblemType"]["StochasticRounding"]:
        rcnt += 2
      tmpCVTVgpr = self.vgprPool.checkOut(rcnt)
    elif isHpaBF16:
      tmpCVTVgpr = self.vgprPool.checkOut(4)
    else:
      tmpCVTVgpr = None

    ########################################
    # Sgprs

    # allocate tmps for the store header (before the batch implementations)
    # tmpSgprRef = self.getTmpSgpr(4)
    # tmpSgpr = tmpSgprRef.idx()

    # branch B1 or B0
    betaLabel = self.getNamedLabelUnique("GW_Beta")
    skPartialsLabel = self.getNamedLabelUnique("SK_Partials")
    skFixupLabel = self.getNamedLabelUnique("SK_Fixup")
    skStoreLabel = self.getNamedLabelUnique("SK_Store")

    if kernel["StreamK"] > 0 and kernel["StreamKAtomic"] == 0:
      # StreamK store branches
      tmpSgpr = self.sgprPool.checkOut(4, "globalWriteElements", preventOverflow=0)
      # if we did not start the tile, store partials
      # branch to beta == 0 store path
      kStr += inst("s_cmp_eq_u32", sgpr("StreamKLocalStart"), 0, "does wg start tile?")
      kStr += inst("s_cbranch_scc0 %s" % skPartialsLabel, "Branch if not start tile, store partials")

      if kernel["DebugStreamK"] & 1 == 0:
        # if we started and finished the tile, regular store code
        # branch to regular store code, skip fixup step
        kStr += inst("s_cmp_eq_u32", sgpr("StreamKLocalEnd"), sgpr("ItersPerTile"), "does wg finish tile?")
        kStr += inst("s_cbranch_scc1 %s" % skStoreLabel, "Branch if started and finished tile, go to regular store code")

        # if we started the tile but did not finish it, fix up step
        # run fixup code before regular store code
        sCtaIdx = self.sgprPool.checkOut(1, "CtaIdx", preventOverflow=0) # self.defineSgpr("CtaIdx", 1)
        kStr += inst("s_add_u32", sgpr(sCtaIdx), sgpr("StreamKIdx"), 1, "input partial tile index")

        sFixupEnd = self.sgprPool.checkOut(1, "FixupEnd", preventOverflow=0) # self.defineSgpr("CtaEnd", 1)
        kStr += self.sMagicDivAlg2(kernel, tmpSgpr, sgpr("StreamKIterEnd"), sgpr("MagicNumberItersPerTile"), sgpr("MagicShiftItersPerTile"))
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(tmpSgpr), sgpr("ItersPerTile"), "start iteration of partial tile")
        kStr += inst("s_sub_u32", sgpr(sFixupEnd), sgpr("StreamKIterEnd"), sgpr(tmpSgpr), "calc iterations completed by this WG")

        kStr += "%s:\n" % (skFixupLabel)

        # Check flag
        kStr += inst("s_lshl_b32", sgpr(tmpSgpr), sgpr(sCtaIdx), log2(4), "flag offset based on CTA index")
        kStr += inst("s_load_dword", sgpr(tmpSgpr+2), sgpr("AddressFlags", 2), sgpr(tmpSgpr), "glc", "get flag")
        kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for flag load")
        if kernel["DebugStreamK"] & 2 == 0:
          kStr += inst("s_cmp_eq_u32", sgpr(tmpSgpr+2), 1, "check if ready")
          kStr += inst("s_cbranch_scc0 %s" % skFixupLabel, "if flag not set, wait and check again")

        self.sgprPool.checkIn(tmpSgpr)

        fixupEdge = [False] # Temporary hack to test no edge variant
        kStr += self.fixupStep(kernel, vectorWidths, elements, fixupEdge, tmpVgpr, tmpCVTVgpr, sCtaIdx, skStoreLabel)

        if kernel["StreamK"] >= 2:
          sIterCount = self.sgprPool.checkOut(1, "iterCount", preventOverflow=0)
          kStr += inst("s_add_u32", sgpr(sIterCount), sgpr("SKItersPerWG"), 1, "Add extra iter")
          kStr += inst("s_cmp_lt_u32", sgpr(sCtaIdx), sgpr("skExtraIters"), "Check if next WG had an extra iteration")
          kStr += inst("s_cselect_b32", sgpr(sIterCount), sgpr(sIterCount), sgpr("SKItersPerWG"), "Select correct number of iterations for next WG")
          kStr += inst("s_add_u32", sgpr(sFixupEnd), sgpr(sFixupEnd), sgpr(sIterCount), "next partial tile iteration")
          self.sgprPool.checkIn(sIterCount)
        kStr += inst("s_add_u32", sgpr(sCtaIdx), sgpr(sCtaIdx), 1, "next partial tile index")
        if kernel["StreamK"] == 1:
          kStr += inst("s_add_u32", sgpr(sFixupEnd), sgpr(sFixupEnd), sgpr("SKItersPerWG"), "next partial tile iteration")
        kStr += inst("s_cmp_lt_u32", sgpr(sFixupEnd), sgpr("ItersPerTile"), "done loading partial tiles?")
        kStr += inst("s_cbranch_scc1 %s" % skFixupLabel, "Branch to continue fixup loop")

        self.sgprPool.checkIn(sFixupEnd)
        self.sgprPool.checkIn(sCtaIdx)

      kStr += "%s:\n" % (skStoreLabel)

    if False in betas and True in betas:
      kStr += self.checkIsBetaZero(kernel, betaLabel)

    for beta in betas:
      # start B1
      if beta:
        kStr += "%s:\n"%(betaLabel)

      ########################################
      # branch if Edge0 or Edge1
      if False in edges and True in edges:
        kStr += self.checkIsEdge(kernel, "%s" % writeLabels[beta][True])

      # by now we either jumped to E1 or stayed at E0
      for edge in edges:
        # write label for batch case
        kStr += "%s:%s"%(writeLabels[beta][edge], self.endLine)
        kStr += self.globalWriteProcedure(kernel, vectorWidths, elements, applyAlpha, beta, edge, atomic, tmpVgpr, tmpCVTVgpr, isOptNLL, endLabel)

    if kernel["StreamK"] > 0 and kernel["StreamKAtomic"] == 0:
      kStr += "%s:\n" % (skPartialsLabel)
      if kernel["DebugStreamK"] & 2 == 0:
        fixupEdge = [False] # Temporary hack to test no edge variant
        kStr += self.writePartials(kernel, vectorWidths, elements, fixupEdge, atomic, tmpVgpr, tmpCVTVgpr, isOptNLL, endLabel)

    # End label
    kStr += "label_%s:%s"%(endLabel, self.endLine)
    self.vgprPool.checkIn(tmpVgpr)
    if tmpCVTVgpr is not None:
      self.vgprPool.checkIn(tmpCVTVgpr)
    return kStr

  ##############################################################################
  # chooseGlobalRead :
  # create the load instruction for requested vector width and other parms
  # return an Inst class
  #
  # bpl = bytes per load op
  ##############################################################################
  def chooseGlobalRead(self, useBuffer, bpl, destVgpr, \
                       addr0, addr1, soffset, offset, extraFields, dtlNoDestVgpr, hi16=0, ubyteLoad=0, comment="load C"):
  # rpv = regs per vector
    rpv = bpl/4.0

    if useBuffer:
      rv = Code.Module("Global Read")
      tailFields = "offen offset:%u"%offset
      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      if offset >= 4096:
        if soffset == 0 or soffset == "0":
          tailFields = "offen offset:0"
          soffset = sgpr(self.getTmpSgpr(1).idx())
          rv.addCode(inst("s_mov_b32", soffset, offset, "large offset"))
        else:
          assert 0, "offset too large and soffset set"
      if extraFields != "":
        tailFields += ", %s"% extraFields
      globalReadInst = None
      if bpl==1 and ubyteLoad:
        #TODO: need to use _buffer_load_u8 macro.. but conflict with buffer_load_d16_u8 in chosen_load_b8
        globalReadInst = "buffer_load_ubyte"
        rpv *= 4
      elif bpl==1 and hi16:
        globalReadInst = "_buffer_load_d16_hi_u8"
        rpv *= 4
      elif bpl==1 and not hi16:
        globalReadInst = "_buffer_load_d16_u8"
        rpv *= 4
      elif bpl==2 and hi16:
        globalReadInst = "_buffer_load_d16_hi_b16"
        rpv *= 2
      elif bpl==2 and not hi16:
        globalReadInst = "_buffer_load_d16_b16"
        rpv *= 2
      elif bpl==4:
        globalReadInst = "_buffer_load_b32"
      elif bpl==8:
        globalReadInst = "_buffer_load_b64"
      elif bpl==16:
        globalReadInst = "_buffer_load_b128"
      elif bpl==32:
        # split into two dwordx4 loads. Second load offset is +0.5 bpl
        tailFields1 = "offen offset:%u"%(offset + bpl/2)
        if extraFields != "":
          tailFields1 += ", %s"% extraFields

        rv = Code.Module("emulated _buffer_load_b256")
        rv.addCode(Code.GlobalReadInst("_buffer_load_b128", vgpr(destVgpr, rpv/2), addr0, \
                  addr1, soffset, tailFields, comment))
        rv.addCode(Code.GlobalReadInst("_buffer_load_b128", vgpr(int(destVgpr + rpv/2), rpv/2), addr0, \
                  addr1, soffset, tailFields1, comment))
        return rv
      else:
        assert 0, "chooseGlobalRead: bad bpl"

      if dtlNoDestVgpr:
        globalReadInst += "_dtl"
      args = [globalReadInst]
      if not dtlNoDestVgpr:
        args.append(vgpr(destVgpr, rpv))
      args.extend([addr0, addr1, soffset, tailFields, comment])
      rv.addCode(Code.GlobalReadInst(*args))
      return rv

    else:
      # use global_load instructions (instead of flat_load) for useBuffer=0
      # specify "off" to disable optional saddr
      # global_load uses offset13s (offset:{-4096..4095})
      # TODO: use saddr if offset is out of offset13s
      if offset > 4095 or offset < -4096:
          assert 0, "offset out of range for global_load"
      tailFields = "offset:%u"%offset
      if extraFields != "":
        tailFields += ", %s"% extraFields
      if bpl==1 and hi16:
        return Code.GlobalReadInst("_global_load_d16_hi_u8", vgpr(destVgpr, rpv*4), addr0, "off", tailFields, comment )
      elif bpl==1 and not hi16:
        return Code.GlobalReadInst("_global_load_d16_u8", vgpr(destVgpr, rpv*4), addr0, "off", tailFields, comment )
      elif bpl==2 and hi16:
        return Code.GlobalReadInst("_global_load_d16_hi_b16", vgpr(destVgpr, rpv*2), addr0, "off", tailFields, comment )
      elif bpl==2 and not hi16:
        return Code.GlobalReadInst("_global_load_d16_b16", vgpr(destVgpr, rpv*2), addr0, "off", tailFields, comment )
      elif bpl==4:
        return Code.GlobalReadInst("_global_load_b32", vgpr(destVgpr, rpv), addr0, "off", tailFields, comment )
      elif bpl==8:
        return Code.GlobalReadInst("_global_load_b64", vgpr(destVgpr, rpv), addr0, "off", tailFields, comment )
      elif bpl==16:
        return Code.GlobalReadInst("_global_load_b128", vgpr(destVgpr, rpv), addr0, "off", tailFields, comment )
      else:
        assert 0, "chooseGlobalRead: bad bpl"

  ##############################################################################
  def chooseGlobalWrite(self, useBuffer, bps, srcVgpr, rpv, \
                        addr0, addr1, offset, extraFields, hi16=0, vb1Tmp=0, soffset=None):
    """
    create the store instruction for requested vector width and other parms
    rpv = regs per vector
    """

    kStr = ""

    if useBuffer:
      tmpSgpr = 0
      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      if offset >= 4096:
        tmpSgpr = sgpr(self.getTmpSgpr(1).idx())
        kStr += inst("s_mov_b32", tmpSgpr, offset, "large offset")
        offset = 0

      if soffset != None:
        assert offset < 4096, "sgpr offset provided with large const offset"
        tmpSgpr = soffset

      if bps == 1 :
        #TODO: need to use _buffer_store_b8 macro.. but conflict with buffer_store_byte_d16 in chosen_store
        if hi16 == 0:
          kStr += inst("buffer_store_byte", vgpr(srcVgpr, rpv*4), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
        elif hi16 == 1:
          #NOTE: no buffer store to select 2nd byte [8:15], so shifting
          kStr += inst("v_lshrrev_b32", vgpr(vb1Tmp),  0x8, vgpr(srcVgpr, rpv*4), "vGWTmp = src >> 8")
          kStr += inst("buffer_store_byte", vgpr(vb1Tmp), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
        elif hi16 == 2:
          kStr += inst("buffer_store_byte_d16_hi", vgpr(srcVgpr, rpv*4), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
        else: #hi16 == 3
          #NOTE: no buffer store to select 4th byte [24:31], so shifting
          kStr += inst("v_lshrrev_b32", vgpr(vb1Tmp),  0x8, vgpr(srcVgpr, rpv*4), "vGWTmp = src >> 8")
          kStr += inst("buffer_store_byte_d16_hi", vgpr(vb1Tmp), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==2 and hi16:
        kStr += inst("_buffer_store_d16_hi_b16", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==2 and not hi16:
        kStr += inst("_buffer_store_b16", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==4:
        kStr += inst("_buffer_store_b32", vgpr(srcVgpr, rpv), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==8:
        kStr += inst("_buffer_store_b64", vgpr(srcVgpr, rpv), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==16:
        kStr += inst("_buffer_store_b128", vgpr(srcVgpr, rpv), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps == 32:
        # split into two dwordx4 loads. Offset the second by +0.5 bps
        kStr += inst("_buffer_store_b128", vgpr(srcVgpr, rpv/2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, extraFields, "store D")

        kStr += inst("_buffer_store_b128", vgpr(int(srcVgpr +rpv/2), rpv/2), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%(int(offset+bps/2)), extraFields, "store D")
      else:
        assert 0, "bad bps"
    else:
      if bps == 1 :
        if hi16 == 0:
          kStr += inst("global_store_byte", vgpr(srcVgpr, rpv*4), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, "off", extraFields, "store D")
        elif hi16 == 1:
          #NOTE: no flat store to select 2nd byte [8:15], so shifting
          kStr += inst("v_lshrrev_b32", vgpr(vb1Tmp),  0x8, vgpr(srcVgpr, rpv*4), "vGWTmp = src >> 8")
          kStr += inst("global_store_byte", vgpr(vb1Tmp), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, "off", extraFields, "store D")
        elif hi16 == 2:
          kStr += inst("global_store_byte_d16_hi", vgpr(srcVgpr, rpv*4), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, "off", extraFields, "store D")
        elif hi16 == 3:
          #NOTE: no flat store to select 4th byte [24:31], so shifting
          kStr += inst("v_lshrrev_b32", vgpr(vb1Tmp),  0x8, vgpr(srcVgpr, rpv*4), "vGWTmp = src >> 8")
          kStr += inst("global_store_byte_d16_hi", vgpr(vb1Tmp), addr0, \
                  addr1, tmpSgpr, "offen", "offset:%u"%offset, "off", extraFields, "store D")
        else:
          assert 0, "bad bps"
      elif bps==2 and hi16:
        kStr += inst("_global_store_d16_hi_b16", addr0, vgpr(srcVgpr*2), "off", extraFields, "store D" )
      elif bps==2 and not hi16:
        kStr += inst("_global_store_d16_b16", addr0, vgpr(srcVgpr, rpv*2), "off", extraFields, "store D" )
      elif bps==4:
        kStr += inst("_global_store_b32", addr0, vgpr(srcVgpr, rpv), "off", extraFields, "store D" )
      elif bps==8:
        kStr += inst("_global_store_b64", addr0, vgpr(srcVgpr, rpv), "off", extraFields, "store D" )
      elif bps==16:
        kStr += inst("_global_store_b128", addr0, vgpr(srcVgpr, rpv), "off", extraFields, "store D" )
      else:
         assert 0, "bad bps"

    return kStr

  ##############################################################################
  def addStore(self, kernel, ss, addrCalc, sumIdx, tmpS01, edge, tc='D', wsOffset=None):
    """
    Add stores for the element with addrCalc and sumIdx.
    tmpS01 is a single :temp sGPR
    """
    kStr = ""
    isWorkspace = tc == 'WS'
    dataType = kernel["ProblemType"]["DestDataType"]
    if isWorkspace:
      dataType = kernel["ProblemType"]["ComputeDataType"]
    if self.do["GlobalWrite"]:
      # perform vector stores here, so no VI indexing.
      # if GWVW > Vw, might need to support loops to
      # implement wider stores
      ntStr = ""
      if (kernel["NonTemporalD"] & 1) != 0 or kernel["ForceStoreSC1"] or isWorkspace:
        ntStr += " " + getGlcBitName(kernel["MemoryModifierFormat"])
      if (kernel["NonTemporalD"] & 2) != 0 or kernel["ForceStoreSC1"] or isWorkspace:
        ntStr += " " + getSlcBitName(kernel["MemoryModifierFormat"])
      if (kernel["NonTemporalD"] & 4) != 0:
        ntStr += " nt"

      bps = self.bpeCexternal * ss.cfg.gwvw
      rpv = self.bpeCexternal * ss.cfg.gwvw / self.bpr
      if isWorkspace:
        bps = self.bpeCinternal * ss.cfg.gwvw
        rpv = self.bpeCinternal * ss.cfg.gwvw / self.bpr

      if kernel["BufferStore"]:
        addr0 = vgpr(addrCalc.addrDVgpr)
        addr1 = sgpr("Srd%s" % (tc), 4)
      else:
        addr0 = vgpr(addrCalc.addrDVgpr,2)
        addr1 = ""

      useBuffer = kernel["BufferStore"]
      if ss.optSrdIncForRow and addrCalc.rowInc and not isWorkspace:
        kStr += addrCalc.incrementToNextRow(kernel, "D", ss, tmpS01)

      offset = addrCalc.globalOffset
      if isWorkspace:
        offset = 0

      if dataType.isHalf() or dataType.isBFloat16():

        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (H,H,H,H,H,H), internal H
          if globalParameters["AsmCaps"][self.version]["HasWMMA"] and kernel["EnableMatrixInstructionStore"]:
            kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, addr0, addr1, offset, ntStr, hi16=0, vb1Tmp=0, soffset=wsOffset)
          else:
            kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx//2, rpv, addr0, addr1, offset, ntStr, hi16=sumIdx%2, vb1Tmp=0, soffset=wsOffset)
        else:
          # (B,B,B,B,S,S), internal S
          # (H,H,H,H,H,H), internal S
          # (H,H,H,H,S,S), internal S
          kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                    addr0, addr1, offset, ntStr, hi16=0, vb1Tmp=0, soffset=wsOffset)
      #TODO: need to test on emulator, always HPA for f8, data is already packed in dest
      elif dataType.isFloat8() or dataType.isBFloat8():
          kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                    addr0, addr1, offset, ntStr, hi16=0, vb1Tmp=0, soffset=wsOffset)

      elif dataType.isInt32() or dataType.isSingle():
        kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                  addr0, addr1, offset, ntStr, hi16=0, vb1Tmp=0, soffset=wsOffset)
      elif dataType.isDouble() or dataType.isSingleComplex():
        if kernel["AtomicAddC"] and not edge:
          kStr += inst("buffer_atomic_add_f64", vgpr(sumIdx*2, 2), vgpr(addrCalc.addrDVgpr), sgpr("SrdD", 4), "0", "offen offset:{}".format(addrCalc.globalOffset), "AtomicAddC")
        else:
          kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx*2, rpv, \
                    addr0, addr1, offset, ntStr, hi16=0, vb1Tmp=0, soffset=wsOffset)
      elif dataType.isDoubleComplex():
        rps = dataType.numRegisters()
        kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx*rps, rpv, \
                  addr0, addr1, offset, ntStr, hi16=0, vb1Tmp=0, soffset=wsOffset)

    return kStr

  ##############################################################################
  # choose the ADD instruction for combining external C with internal C
  # used in atomic=1 case to compute expected external data
  ##############################################################################
  def chooseAddForAtomic(self, kernel, dst, src0, sumIdxV, comment, atomicOpW):
    kStr = ""
    regNum = 1
    if kernel["ProblemType"]["ComputeDataType"].isSingle() or kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
      addInst = "v_add_f32"
    elif kernel["ProblemType"]["ComputeDataType"].isHalf():
      addInst = "v_pk_add_f16"
    elif kernel["ProblemType"]["ComputeDataType"].isInt32():
      # assume v_add_i32 can be used in place of v_add_f32
      # need to add saturation directive to v_add_i32 instruction to clamp integer arithmetic
      addInst = "_v_add_i32"
    elif kernel["ProblemType"]["ComputeDataType"].isDouble() or kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
      addInst = "v_add_f64"
      regNum = 2
    else:
      assert(0) # unsupported data type,

    for idx in range(0,atomicOpW, regNum):
      kStr += inst(addInst, vgpr(dst+idx, regNum), vgpr(src0+idx, regNum), vgpr("ValuC+%u"%(sumIdxV+idx), regNum), comment)

    return kStr

  ##############################################################################
  # choose the atomic_cmpswap instruction for write attempt
  # used in atomic=1 case
  ##############################################################################
  def chooseAtomicCmpswap(self, kernel, addrCalc, addDst, offset, atomicOpW, memoryBit):
    kStr = ""
    # attempt write
    if self.do["GlobalWrite"]:
      # use cmpswap_b64 for DGEMM or cmpswap_b32 for DGEMM in CAS loop
      bits = 32 * atomicOpW
      if kernel["BufferStore"]:
        kStr += "_buffer_atomic_cmpswap_b%u %s, %s, %s %s %s   // %s%s" % \
            (bits, \
            vgpr(addDst,atomicOpW*2), \
            vgpr(addrCalc.addrDVgpr,1), \
            sgpr("SrdD", 4),  \
            "0 offen offset:%u" % (addrCalc.globalOffset + offset), \
            memoryBit,
            "attempt write", self.endLine )
      else:
        # not BufferStore case
        kStr += "_global_atomic_cmpswap_b%u %s, %s, %s, %s, %s, %s    // %s%s" % \
            (bits, vgpr(addDst,atomicOpW), vgpr(addrCalc.addrDVgpr,2), \
            vgpr(addDst,atomicOpW*2), "off", "offset:%u"%offset, memoryBit, "attempt write", self.endLine )
    else:
       # Fake successful CAS swap:
       for idx in range(atomicOpW):
         kStr += inst("v_mov_b32", vgpr(addDst+idx), vgpr(addDst+atomicOpW+idx), "Fake successful CAS" )

    return kStr

  ##############################################################################
  def applyAlpha(self, kernel, gwvw, elementSumIdx, elementIdx, tmpS01):
    kStr = ""

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      return kStr

    if self.do["ApplyAlpha"]:
      for vi in range(0, gwvw):
        sumIdxV = elementSumIdx[elementIdx] + vi

        if kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (h,h,h,h,h,h), internal alpha is f16 (2-16bits)
          if sumIdxV%2:
            kStr += inst("v_pk_mul_f16", vgpr("ValuC+%u"%(sumIdxV//2)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV//2)), "*= alpha sumIdx=%u vi=%u"%(elementSumIdx[elementIdx], vi))

        # Int8 (TODO- Int8x4 not checked, but should be OK)
        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          # below assume we use v_mul_lo_u32. Could also use v_mul_i32_i24.
          # kStr += inst("v_mul_i32_i24", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          kStr += inst("v_mul_lo_u32", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          if self.db["ForceExpectedValue"]:
            kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
          if self.db["CheckValueC"]:
            kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
            kStr += self.assert_eq(vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01))

        # sgemm, HPA-bfgemm(b,b,b,b,s,s), and HPA-hgemm(h,h,h,h,s,s)
        # (h,h,h,h,h,h) + HPA (will be converted to (h,h,h,h,s,s)), internal alpha is single
        elif kernel["ProblemType"]["ComputeDataType"].isSingle() or (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
          kStr += inst("v_mul_f32", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          if self.db["ForceExpectedValue"]:
            kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
          if self.db["ForceVSerial"]:
            kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
          if self.db["CheckValueC"]:
            kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
            kStr += self.assert_eq(vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01))

        # dgemm
        elif kernel["ProblemType"]["ComputeDataType"].isDouble():
          kStr += inst("v_mul_f64", vgpr("ValuC+%u"%(sumIdxV*2),2), sgpr("Alpha",2), vgpr("ValuC+%u"%(sumIdxV*2),2), "*= alpha")

        # single precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
          tmpVgpr = self.vgprPool.checkOut(1)
          kStr += inst("v_mov_b32", vgpr(tmpVgpr), vgpr("ValuC+%u"%(sumIdxV*2)), "store Cr")
          kStr += inst("v_mul_f32", vgpr("ValuC+%u"%(sumIdxV*2)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV*2)), "*= alpha ( Cr = Ar * Cr)")
          kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), "-" + sgpr("Alpha+1"), vgpr("ValuC+%u"%(sumIdxV*2+1)), "*= alpha ( Cr += -Ai * Ci )")
          kStr += inst("v_mul_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV*2+1)), "*= alpha ( Ci = Ar * Ci)")
          kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), sgpr("Alpha+1"), vgpr(tmpVgpr), "*= alpha ( Ci += Ai * Cr_backup )")
          self.vgprPool.checkIn(tmpVgpr)

        # double precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
          vtmp1 = self.vgprPool.checkOutAligned(2, 2)
          vtmp2 = self.vgprPool.checkOutAligned(2, 2)
          # tmp1 = a.real * b.real
          kStr += inst("v_mul_f64", vgpr(vtmp1,2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "")
          # tmp2 = a.imag * b.real
          kStr += inst("v_mul_f64", vgpr(vtmp2,2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "")
          # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
          kStr += "v_fma_f64 %s, %s, -%s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+0),2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(vtmp1,2), self.endLine)
          # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
          kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+2),2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(vtmp2,2), self.endLine)
          self.vgprPool.checkIn(vtmp1)
          self.vgprPool.checkIn(vtmp2)

    return kStr

  ##############################################################################
  # Global Read C Input
  ##############################################################################
  def readCInput(self, kernel, ss, addrCalc, vc0, data, gwvw, addr, tmpS01, tc='C'):
    kStr = ""
    useBuffer = kernel["BufferStore"]
    isWorkspace = tc == 'WS'
    dataType = kernel["ProblemType"]["DestDataType"]
    if isWorkspace:
      dataType = kernel["ProblemType"]["ComputeDataType"]
    bps = dataType.numBytes() * gwvw

    if kernel["BufferStore"]:
      addr0 = vgpr(addr)
      addr1 = sgpr("Srd%s"%(tc), 4)
    else:
      addr0 = vgpr(addr,2)
      addr1 = ""

    extraStr = ""
    if (kernel["NonTemporalC"] & 1) != 0:
      extraStr += " " + getGlcBitName(kernel["MemoryModifierFormat"])
    if (kernel["NonTemporalC"] & 2) != 0:
      extraStr += " " + getSlcBitName(kernel["MemoryModifierFormat"])
    if (kernel["NonTemporalC"] & 4) != 0:
      extraStr += " nt"

    if ss.optSrdIncForRow and addrCalc.rowInc and not isWorkspace:
      kStr += addrCalc.incrementToNextRow(kernel, tc, ss, tmpS01)

    soffset = 0
    offset = addrCalc.globalOffset
    comment = "load C for beta calc"
    if isWorkspace:
      soffset = tmpS01
      offset = 0
      comment = "load partials"

    if dataType.isHalf():
      hi16 = 0 if self.HHH_WMMA else (vc0 % 2)
      kStr += self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=soffset, offset=offset, \
                extraFields=extraStr, dtlNoDestVgpr=False, hi16=hi16, \
                comment=comment).toStr()

    elif dataType.isFloat8() or \
         dataType.isBFloat8():
      kStr += self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=soffset, offset=offset, \
                extraFields=extraStr, dtlNoDestVgpr=False, hi16=0, ubyteLoad=1, \
                comment=comment).toStr()

    elif dataType.isBFloat16() or \
         dataType.isInt32() or \
         dataType.isSingle() or \
         dataType.isDouble() or \
         dataType.isSingleComplex() or \
         dataType.isDoubleComplex():
      kStr += self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=soffset, offset=offset, \
                extraFields=extraStr, dtlNoDestVgpr=False, \
                comment=comment).toStr()

    return kStr

  ##############################################################################
  # Global Write Batch
  ##############################################################################
  def globalWriteBatch(self, kernel, ss, batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
      batchElements, coord0, coord1, addrD, addrC, \
      tmpVgpr, tmpCVTVgpr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, isOptNLL):
    kStr = ""

    kStr += self.comment1("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
              (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

    if kernel["StoreSyncOpt"]:
      kStr += "s_sleep %d // optimization: sync and wait\n" %(kernel["StoreSyncOpt"]-1)
      kStr += "s_barrier\n"

    if atomic:
      # all kinds of code relies on this assumption:
      assert(atomicW <= gwvw)
      # atomicW * numBytes should be >= self.bpr
      assert((atomicW * kernel["ProblemType"]["ComputeDataType"].numBytes()) >= self.bpr)

    # comment tt1, tt0, vc1, vc0
    # tt = thread tile, vc=vector component
    commentStr = "Global Write%s%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Alpha" if applyAlpha else "", " Beta" if beta else "", " Edge" if edge else "", batchIdx)
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      commentStr += "(%u,%u,%u,%u:vw%u%s)" % \
        (element[0], element[1], element[2], element[3], gwvw,
         ":vaw:%u"%atomicW if atomic else "")
      if elementIdx < len(batchElements)-1:
        commentStr += "; "
    kStr += self.comment3(commentStr)
    # print(self.kernelName)
    # print(commentStr)

    # allow expanding vgpr pool for OptNLL
    preventOverflow = (not isOptNLL)
    ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, preventOverflow=preventOverflow)

    loadsIssued = 0
    storesIssued = 0
    tmpS01 = tmpSgpr # scratch sgprs
    tmpS23 = tmpS01+self.laneSGPRCount

    wavelen = self.kernel["WavefrontSize"]
    laneSGPRC = self.laneSGPRCount
    # always use gwvw for buffer load C for atomic_cmpswap
    # bpm = self.bpeCexternal * atomicW
    bpm = self.bpeCexternal * gwvw
    vgprLoadDW = 1*(bpm//4)
    # atomic oparation width. 1 for b32, 2 for b64
    atomicOpW = (atomicW * self.bpeCexternal) // 4
    if atomicOpW > 2:
      # should not exceeding 2.
      atomicOpW = 2


    ########################################
    # calculate addr and masks
    kStr += self.comment("calc coords, apply mask, and issue loads (if necessary)")
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.db["ConservativeWaitCnt"] & 0x10:
      kStr += "s_barrier // debug\n"
      kStr += inst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      kStr += "s_barrier // debug\n"
    if not edge and self.db["ForceEdgeStores"]>=2:
      kStr += self.bomb() # should not get here
    if edge and self.db["AssertNoEdge"]:
      kStr += self.bomb() # should not get here

    ########################################
    # rC *= alpha
    if not kernel["InterleaveAlpha"] and applyAlpha and self.alphaBeforeLoadC:
      kStr += self.generateAlphaCodeForGlobalWriteBatch(kernel, ss, codeMulAlpha, batchElements, gwvw, tmpS01, tmpVgpr)

    atomicAddC = kernel["AtomicAddC"] and not edge

    loadCInputCode = ""
    LoadCCodeStr = ""
    ## create code Module to push mov vgpr,acc instructions
    if kernel["StoreCInUnroll"] and not edge:
      LoadCCodeMod = Code.Module("LoadC")
      AlphaCodeMod = Code.Module("Alpha")
      BetaCodeMod = Code.Module("Beta")
      accVgprRead = Code.Module("movaccVgpr")
      StoreCCodeMod = Code.Module("StoreC")
      self.StoreCUnrollLoadCWaitComment = "waitcnt for LoadC" # this will be used later to identify waitcnt for loadC
      if not atomicAddC and kernel["ProblemType"]["UseBeta"]:
        # put waitcnt for loadC before beta code
        BetaCodeMod.addText("s_waitcnt vmcnt(__placeholder__) // %s\n"%self.StoreCUnrollLoadCWaitComment)

    # add persistent kernel loopend code for StoreCInUnroll here after global offset calculation (only for the first batch)
    if isOptNLL and kernel["StoreCInUnroll"] and batchIdx == 0:
      # generate global offset before starting end process for StoreInUnroll
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
        addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
        addrCalc = ss.elementAddr[elementIdx]
        mask = ss.elementMask[elementIdx]

        if elementIdx == 0 and beta and not atomicAddC:
          kStr += addrCalc.emitAddressSetupCode(kernel, ss, 'C', tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrDVgpr)
          kStr += addrCalc.emitLdChange(kernel, ss, 'C', edge, beta, mask, (elementIdx == 0), tmpVgpr, addrCVgpr, addrC)

        if elementIdx == len(batchElements)-1:
          kStr += addrCalc.emitAddressSetupCode(kernel, ss, 'C', tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrDVgpr)
          kStr += addrCalc.emitLdChange(kernel, ss, 'D', edge, beta, mask, (elementIdx == len(batchElements)-1), tmpVgpr, addrDVgpr, addrD)

      kStr += self.endProcessPersistentLoopforStoreCInUnrollOptNLL(kernel)

    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
      addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
      addrCalc = ss.elementAddr[elementIdx]
      data = ss.elementData[elementIdx]
      mask = ss.elementMask[elementIdx]
      sumIdx = ss.elementSumIdx[elementIdx]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]

      kStr += addrCalc.emitAddressSetupCode(kernel, ss, 'C', tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrDVgpr)
      if edge and (kernel["BufferStore"] or beta):
        kStr += addrCalc.edgeProtectCode(kernel, edge, beta, atomic, mask, tmpSgpr)

      # create code Module to push mov vgpr,acc instructions

      if beta and not atomicAddC:
        kStr += addrCalc.emitLdChange(kernel, ss, 'C', edge, beta, mask, (elementIdx == 0), tmpVgpr, addrCVgpr, addrC)
        if kernel["GroupLoadStore"]:
          loadCInputCode += self.readCInput(kernel, ss, addrCalc, vc0, data, gwvw, addrCVgpr, tmpS01)
        else:
          kStr += self.readCInput(kernel, ss, addrCalc, vc0, data, gwvw, addrCVgpr, tmpS01)
        loadsIssued += 1
        if kernel["StoreCInUnroll"] and not edge:
          regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
          data = "G2LC+%s"%(elementIdx*gwvw*regsPerScalar)
          tmpS01 = "StoreCOffsetAddr"
          LoadCCodeStr = self.readCInput(kernel, ss, addrCalc, vc0, data, gwvw, addrCVgpr, tmpS01)
          if kernel["AssertCEqualsD"]:
            # CEqualsD case, use SrdD instead of SrdC
            LoadCCodeStr = LoadCCodeStr.replace('SrdC', 'SrdD')
          LoadCCodeMod.addCode(LoadCCodeStr)

      if not kernel["BufferStore"]:
        kStr += addrCalc.emitAddressSetupCode(kernel, ss, 'D', tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrDVgpr)
        if edge:
          kStr += addrCalc.edgeProtectCode(kernel, edge, beta, atomic, mask, tmpSgpr)

      kStr += addrCalc.emitLdChange(kernel, ss, 'D', edge, beta, mask, (elementIdx == len(batchElements)-1), tmpVgpr, addrDVgpr, addrD)

      if atomic and (not self.useAtomicAdd):
        # load c into data+1 because of CAS structure
        # TODO - Fix for double here, would need bigger load
        # FIME
        bps = kernel["ProblemType"]["DestDataType"].numBytes()
        # gwvw is the number of elements in the batch
        # load C always with gwvw
        dataV = ss.elementData[elementIdx]
        useBuffer = kernel["BufferStore"]
        if kernel["BufferStore"]: # yes, BufferStore here - use same addressing regs for this load
          addr0 = vgpr(addrDVgpr)
          addr1 = sgpr("SrdD", 4)
        else:
          addr0 = vgpr(addrDVgpr,2)
          addr1 = ""
        # Calculate vgpr Index for 32-bit/64-bit instruction
        # DGEMM use SRCS[2] register
        kStr += self.chooseGlobalRead(useBuffer, bpm, dataV+vgprLoadDW, \
                  addr0, addr1, soffset=0, offset=addrCalc.globalOffset, extraFields="",
                  dtlNoDestVgpr=False, \
                  comment="load D (atomic) bpm=%u vaw=%u"%(bpm,atomicW)).toStr()

      if kernel["InterleaveAlpha"] and applyAlpha:
        kStr += self.applyAlpha(kernel, gwvw, ss.elementSumIdx, elementIdx, tmpS01)

      if (kernel["StoreCInUnroll"] and applyAlpha) and (not edge):
        regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
        for vi in range(0, gwvw):
           L2GCidx = elementIdx*gwvw*regsPerScalar + vi*regsPerScalar
           if kernel["ProblemType"]["ComputeDataType"].isDouble():
             AlphaCodeMod.addCode(inst("v_mul_f64", vgpr("L2GC+%u"%(L2GCidx),2), sgpr("Alpha",2), vgpr("L2GC+%u"%(L2GCidx),2), "*= alpha"))
           elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
             # cannot use tmp vgpr for StoreCInUnroll, use allocated vgpr instead
             vtmp1 = self.startVgprAlphaTmp
             vtmp2 = vtmp1 + 2
             # tmp1 = a.real * b.real
             AlphaCodeMod.addCode(inst("v_mul_f64", vgpr(vtmp1,2), sgpr("Alpha+0",2), vgpr("L2GC+%u"%(L2GCidx),2), ""))
             # tmp2 = a.imag * b.real
             AlphaCodeMod.addCode(inst("v_mul_f64", vgpr(vtmp2,2), sgpr("Alpha+2",2), vgpr("L2GC+%u"%(L2GCidx),2), ""))
             # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
             AlphaCodeMod.addCode("v_fma_f64 %s, %s, -%s, %s%s" % (vgpr("L2GC+%u"%(L2GCidx),2), sgpr("Alpha+2",2), vgpr("L2GC+%u"%(L2GCidx+2),2), vgpr(vtmp1,2), self.endLine))
             # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
             AlphaCodeMod.addCode("v_fma_f64 %s, %s, %s, %s%s" % (vgpr("L2GC+%u"%(L2GCidx+2),2), sgpr("Alpha+0",2), vgpr("L2GC+%u"%(L2GCidx+2),2), vgpr(vtmp2,2), self.endLine))
           #TODO need fix for other precisions

      if not kernel["BufferStore"]:
        # restore full exec mask for calculating addr of next element
        if edge and (beta or atomic):
          kStr += inst("s_mov_b{}".format(kernel["WavefrontSize"]), self.exec, -1, "full mask -1 -> exec" )

    kStr += loadCInputCode

    if beta and kernel["StoreSyncOpt"]:
      kStr += "s_sleep %d // optimization: sync and wait\n" %(kernel["StoreSyncOpt"]-1)
      kStr += "s_barrier\n"

    ########################################
    # AccVgpr read
    if kernel.enabledSetPrioSplitLDS:
      kStr += inst("s_setprio", "0", "")
    if codeAccVgprRead is not None:
      regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(0, len(batchElements)):
        # loop over scalars within one store instruction
        for vi in range(0, gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            tempStr = str(codeAccVgprRead.items().pop(0))
            kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
            if kernel["StoreCInUnroll"] and not edge:
              tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
              accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

      if not kernel["MIArchVgpr"]:
        kStr += inst("s_nop 1", "2 wait states required before reading vgpr")

    ########################################
    # rC *= alpha
    if not kernel["InterleaveAlpha"] and applyAlpha and not self.alphaBeforeLoadC:
      kStr += self.generateAlphaCodeForGlobalWriteBatch(kernel, ss, codeMulAlpha, batchElements, gwvw, tmpS01, tmpVgpr)

    ########################################
    # Atomic
    ########################################
    # global_atomic_cmpswap tmp addr data:
    #   tmp = mem[addr]
    #   src = data[vi*numVgprsPerDataPerVI][0] new C
    #   cmp = data[vi*numVgprsPerDataPerVI][1] original C
    #   mem[addr] = (tmp==cmp) ? src : tmp
    #   addr = vgpr(addr,2)
    #   data = vgpr(tmpVgpr,2)
    #   tmp = vgpr(tmpVgpr+4)

    # buffer_atomic_cmpswap:
    #   dest is 64 bits, two consec VGPR:
    #     - lower is desired swap value (computed new value) "src"
    #       src = data[vi*numVgprsPerDataPerVI][0] new C
    #     - upper is expected value in memory (from prev load).  "cmp".
    #       cmp = data[vi*numVgprsPerDataPerVI][1] original C
    #   src0 is address offset from SRD
    #
    # After buffer_atomic_cmpswap:
    #   dest =
    #       - data[vi*numVgprsPerDataPerVI][0] C loaded from memory, overwrites src
    if atomic:
      del tmpVgpr # catch bugs
      # TODO for atomic GWVW:
      #  - Use vi to compute addresses, sumIdx.
      #  - Need a solution for the mask.  Can move to all buffer or can fix?

      element = batchElements[0]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]
      labelString = "Global_Write%s%s%s_vc=%u,%u_d=%u,%u" \
        % (" Beta" if beta else "", " Edge" if edge else "", " Opt" if isOptNLL else "", vc0, vc1, d0, d1 )
      label = self.getLabelNum(labelString)
      labelString += "EarlyExit"
      labelAfterAtomicLoop = self.getLabelNum(labelString)
      memoryBit = getGlcBitName(kernel["MemoryModifierFormat"])

      if self.useAtomicAdd:
        ########################################
        # first attempt write
        kStr += self.comment("issue first atomic writes")
        for elementIdx in range(0, len(batchElements)):
          addrCalc = ss.elementAddr[elementIdx]
          mask     = ss.elementMask[elementIdx]

          # apply in-bounds exec mask
          if edge:
            kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec (before atomic)" )

          for compIdx in range(vgprLoadDW // atomicOpW):
            addrOffset = compIdx * atomicOpW * self.bpr
            compOffset = compIdx * atomicOpW
            compDstRegOffset = compOffset * 2 # 2 sets (before and after atomic op) per atomic operation
            sumIdxV = ss.elementSumIdx[elementIdx]
            sumIdxV = int(sumIdxV * kernel["ProblemType"]["ComputeDataType"].numRegisters())
            sumIdxV += compOffset

            if self.do["GlobalWrite"]:
              if kernel["BufferStore"]:
                kStr += "buffer_atomic_add_f%u %s, %s, %s, %s    // %s%s" % \
                    (atomicOpW * 32, \
                     vgpr("ValuC+%u"%sumIdxV,atomicOpW), \
                     vgpr(addrCalc.addrDVgpr,1), \
                     sgpr("SrdD", 4), \
                     "0 offen offset:%u" % (addrCalc.globalOffset + addrOffset), \
                     "attempt write", self.endLine )
              else:
                pass # TODO:

        if edge:
          kStr += inst("s_mov_b{}".format(wavelen), self.exec, -1, "full mask -> exec" )
      else:
        ########################################
        # wait for batched load
        # TODO - we are always atomic here?
        kStr += inst("s_waitcnt", "vmcnt(0)", "wait C (atomic)" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

        ########################################
        # first attempt write
        kStr += self.comment("issue first atomic writes")
        for elementIdx in range(0, len(batchElements)):
          addrCalc = ss.elementAddr[elementIdx]
          mask = ss.elementMask[elementIdx]

          # apply in-bounds exec mask
          if edge:
            kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec (before atomic)" )

          dataV = ss.elementData[elementIdx]
          # vgprLoadDW > atomicOpW case, we use wider load
          # In this case, we need to move loaded data to use multiple cmpswap instructions
          #  ex.) vgprLoadDW=4, atomicOpW=2 case
          #   before move (after load b128) :
          #         dataV + 0-3: empty
          #         dataV + 4-7: loaded data
          #   after move (4-5 to 2-3):
          #         dataV + 0-1: empty (for add data)
          #         dataV + 2-3: loaded data (first 8Byte)  -> need to move
          #         dataV + 4-5: empty (for add data)
          #         dataV + 6-7: loaded data (second 8Byte) -> no move
          # adjust cmpswapWidth (no b128 instruction)
          if vgprLoadDW > atomicOpW:
            for idx1 in range((vgprLoadDW//atomicOpW) - 1):
              for idx0 in range(atomicOpW):
                dst = dataV + (2*atomicOpW) * idx1 + atomicOpW + idx0
                src = dataV + vgprLoadDW + atomicOpW * idx1 + idx0
                kStr += inst("v_mov_b32", vgpr(dst), vgpr(src), "v_mov for reordering loaded data" )

          for compIdx in range(vgprLoadDW // atomicOpW):
            addrOffset = compIdx * atomicOpW * self.bpr
            compOffset = compIdx * atomicOpW
            compDstRegOffset = compOffset * 2 # 2 sets (before and after atomic op) per atomic operation
            sumIdxV = ss.elementSumIdx[elementIdx]
            ## number of src[s]/dst[s] register for DGEMM / SGEMM HGEMM
            sumIdxV = int(sumIdxV * kernel["ProblemType"]["ComputeDataType"].numRegisters())
            sumIdxV += compOffset
            # Calculate vgpr Index for 32-bit/64-bit instruction
            # DGEMM use SRCS[2] register
            addDst = dataV + compDstRegOffset
            addSrc0 = addDst + atomicOpW
            comment = "desired value"
            # for atomic, data[1] = original c, data[0] = new c
            kStr += self.chooseAddForAtomic(kernel, addDst, addSrc0, sumIdxV, comment, atomicOpW)
            # attempt write
            kStr += self.chooseAtomicCmpswap(kernel, addrCalc, addDst, addrOffset, atomicOpW, memoryBit)

        ########################################
        # edge case only, copy out of range mask to all masks for each atomic instruction
        if edge:
          for elementIdx in range(0, len(batchElements)):
            mask = ss.elementMask[elementIdx]
            # calculate new masks
            for compIdx in range(vgprLoadDW // atomicOpW):
              maskIdx = mask + compIdx * laneSGPRC
              if compIdx > 0:
                # copy out of range mask from the first one
                kStr += inst("s_mov_b{}".format(wavelen),  sgpr(maskIdx,laneSGPRC), sgpr(mask,laneSGPRC), "copy out of range mask from the first one" )

        ########################################
        # wait for first attempt write
        kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

        ########################################
        # check first attempt
        kStr += self.comment("check success of writes, update masks")
        for elementIdx in range(0, len(batchElements)):
          mask = ss.elementMask[elementIdx]
          for compIdx in range(vgprLoadDW // atomicOpW):
            maskIdx = mask + compIdx * laneSGPRC
            compOffset = compIdx * atomicOpW
            compDstRegOffset = compOffset * 2 # 2 sets (before and after atomic op) per atomic operation
            dataV = ss.elementData[elementIdx]
            atomicDestVgpr = dataV + compDstRegOffset
            dstSgpr = tmpS01 if edge else maskIdx
            comment = "c read during atomic != c read during prior load"

            if edge:
              # edge case only, set exec flag
              kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(maskIdx,laneSGPRC), "sgprs -> exec" )
            kStr += inst("v_cmp_ne_u%u"%(32 * atomicOpW), sgpr(dstSgpr,laneSGPRC), vgpr(atomicDestVgpr,atomicOpW), \
                vgpr(atomicDestVgpr+atomicOpW,atomicOpW), comment )

            if kernel["DisableAtomicFail"]:
              kStr += inst("s_mov_b{}".format(wavelen),  sgpr(maskIdx,laneSGPRC), 0, "DisableAtomicFail, force 0" )
            elif edge:
              kStr += inst("s_and_b{}".format(wavelen),  sgpr(maskIdx,laneSGPRC), sgpr(tmpS01,laneSGPRC), sgpr(maskIdx,laneSGPRC), "inBounds & must try again" )

        # or masks together to check early exit
        kStr += self.comment("or masks to check for exit")
        kStr += inst("s_mov_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), hex(0), "empty mask" )
        for elementIdx in range(0, len(batchElements)):
          mask = ss.elementMask[elementIdx]
          for compIdx in range(vgprLoadDW // atomicOpW):
            maskIdx = mask + compIdx * laneSGPRC
            kStr += inst("s_or_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), sgpr(maskIdx,laneSGPRC), sgpr(tmpS01,laneSGPRC), "or to add threads" )
        kStr += inst("s_or_saveexec_b{}".format(wavelen), sgpr(tmpS23,laneSGPRC), sgpr(tmpS01,laneSGPRC), "apply combined mask" )
        kStr += inst("s_cbranch_execz", "label_%04u" % labelAfterAtomicLoop, "if exec is zero skip loop" )

        # begin atomic loop
        kStr += self.comment("atomic CAS loop")
        kStr += "label_%04u:%s" % (label, self.endLine)

        kStr += self.comment("apply updated masks and issue writes again")
        for elementIdx in range(0, len(batchElements)):
          addrCalc = ss.elementAddr[elementIdx]
          mask = ss.elementMask[elementIdx]

          for compIdx in range(vgprLoadDW // atomicOpW):
            maskIdx = mask + compIdx * laneSGPRC
            addrOffset = compIdx * atomicOpW * self.bpr
            compOffset = compIdx * atomicOpW
            compDstRegOffset = compOffset * 2 # 2 sets (before and after atomic op) per atomic operation
            dataV = ss.elementData[elementIdx]
            sumIdxV = ss.elementSumIdx[elementIdx]
            sumIdxV = int(sumIdxV * kernel["ProblemType"]["ComputeDataType"].numRegisters())
            sumIdxV += compOffset
            addDst = dataV + compDstRegOffset
            addSrc0 = addDst + atomicOpW
            comment = "newC = rC + originalC"

            # apply mask for element
            kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(maskIdx,laneSGPRC), "must try again" )
            # copy previous load data
            for idx in range(atomicOpW):
              movDst = addDst+atomicOpW
              kStr += inst("v_mov_b32", vgpr(movDst+idx), vgpr(addDst+idx), "dataV+%u = tmp (new original C)"%(atomicOpW+idx) )
            kStr += self.chooseAddForAtomic(kernel, addDst, addSrc0, sumIdxV, comment, atomicOpW)
            # attempt write
            kStr += self.chooseAtomicCmpswap(kernel, addrCalc, addDst, addrOffset, atomicOpW, memoryBit)

        # wait for batched write
        kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

        # check batched write success
        kStr += self.comment("apply masks and check for success")
        for elementIdx in range(0, len(batchElements)):
          data = ss.elementData[elementIdx]
          mask = ss.elementMask[elementIdx]
          for compIdx in range(vgprLoadDW // atomicOpW):
            maskIdx = mask + compIdx * laneSGPRC
            compOffset = compIdx * atomicOpW
            compDstRegOffset = compOffset * 2 # 2 sets (before and after atomic op) per atomic operation
            dataV = ss.elementData[elementIdx]
            atomicDestVgpr = dataV + compDstRegOffset
            comment = "c read during atomic != c read during prior load"
            # apply mask for element
            kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(maskIdx,laneSGPRC), "must try again" )
            # compare success
            kStr += inst("v_cmp_ne_u%u"%(32 * atomicOpW), sgpr(tmpS01,laneSGPRC), vgpr(atomicDestVgpr,atomicOpW), \
                vgpr(atomicDestVgpr+atomicOpW,atomicOpW), comment )
            # update element mask
            kStr += inst("s_and_b{}".format(wavelen),  sgpr(maskIdx,laneSGPRC), sgpr(tmpS01,laneSGPRC), sgpr(maskIdx,laneSGPRC), "inBounds & must try again" )

        # or masks together
        kStr += self.comment("or masks to check for exit")
        kStr += inst("s_mov_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), hex(0), "empty mask" )
        for elementIdx in range(0, len(batchElements)):
          mask = ss.elementMask[elementIdx]
          for compIdx in range(vgprLoadDW // atomicOpW):
            maskIdx = mask + compIdx * laneSGPRC
            kStr += inst("s_or_b{}".format(wavelen), sgpr(tmpS01,laneSGPRC), sgpr(maskIdx,laneSGPRC), sgpr(tmpS01,laneSGPRC), "or to add threads" )

        # apply combined masks and exit
        kStr += inst("s_or_saveexec_b{}".format(wavelen), sgpr(tmpS23,laneSGPRC), sgpr(tmpS01,laneSGPRC), "apply combined mask" )
        kStr += inst("s_cbranch_execnz", "label_%04u" % label, "try again if not complete" )
        kStr += "label_%04u:%s" % (labelAfterAtomicLoop, self.endLine)
        kStr += inst("s_mov_b{}".format(wavelen), self.exec, -1, "full mask -> exec" )

    ########################################
    # Not Atomic
    ########################################
    else:
      # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
      interleaveStoreVmcnt = self.interleaveStoreVmcnt and not edge
      for elementIdx in range(0, len(batchElements)):
        for vi in range(0, gwvw):
          sumIdxV = ss.elementSumIdx[elementIdx] + vi
          # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
          if kernel["ProblemType"]["ComputeDataType"].isInt32() or \
             kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
              if self.db["ForceExpectedValue"]:
                kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
              if self.db["ForceVSerial"]:
                kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
              if self.db["CheckValueC"]:
                kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
                kStr += self.assert_eq(vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01))

      ########################################
      # wait for batched load
      if beta and not interleaveStoreVmcnt:
        kStr += inst("s_waitcnt", "vmcnt(0)", "wait C")
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

        # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
        # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
        # And this only occurs when beta=true, so case must not be 2 or 3
        assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, \
          "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

      kStr += self.comment("apply mask, calc new C and issue writes")
      #kStr += self.bomb() # can see store addresses just before the store inst

      if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        vgprBf16Temp = tmpCVTVgpr
        vgprBf16Mask = vgprBf16Temp + 1
        vgprFp32Nan = vgprBf16Temp + 2
        vgprBf16Inc = vgprBf16Temp + 3
        kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
        kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
        kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )

      # DestDataType for 8bit Float can only be F8 or B8
      if kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8(): # F8 is always HPA
        # make vgprF8Temp0 always even to use pk instruction later
        if tmpCVTVgpr % 2 == 0:
          vgprF8Temp0 = tmpCVTVgpr
          vgprF8Temp1 = vgprF8Temp0 + 1
          vgprF8Max = vgprF8Temp0 + 2
          vgprF8Min = vgprF8Temp0 + 3
        else:
          vgprF8Max = tmpCVTVgpr
          vgprF8Temp0 = vgprF8Max + 1
          vgprF8Temp1 = vgprF8Max + 2
          vgprF8Min = vgprF8Max + 3

        lastVgprUsed = vgprF8Min
        if kernel["ProblemType"]["Fp32toFp8SWClip"]:
          # set flag of f32 NaN and +/- INF for v_cmp_class
          vgprFp32NanInfFlag = vgprF8Min + 1
          lastVgprUsed = vgprFp32NanInfFlag
          kStr += inst("v_mov_b32", vgpr(vgprFp32NanInfFlag), "0x207", "flag for Nan and +/- inf" )
          # set max/min values for clipping
          if kernel["ProblemType"]["DestDataType"].isFloat8():
            kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x43700000", "save 240.0f as max for clipping" )
            kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC3700000", "save -240.0f as min for clipping" )
          else: #BFloat8
            kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x47600000", "save 57344.0f as max for clipping" )
            kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC7600000", "save -57344`.0f as min for clipping" )

        # need extra two temp registers for PRNG func
        if kernel["ProblemType"]["StochasticRounding"]:  # in-device RND
          vgprSRTemp0 = lastVgprUsed + 1
          vgprSRTemp1 = lastVgprUsed + 2

      storeCode = ""
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = ss.elementAddr[elementIdx].addrDVgpr
        mask = ss.elementMask[elementIdx]
        addrCalc = ss.elementAddr[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]
        sumIdx = ss.elementSumIdx[elementIdx]

        # print(str(element)+" rowInc="+str(addrCalc.rowInc))
        # Already write wave column block into LDS
        # Now read lds data back to registers and write to global memroy
        if ss.optSrdIncForRow and addrCalc.rowInc and kernel["StoreRemapVectorWidth"] > 0:
          kStr += self.comment("StoreRemap: shift coord1 address")
          kStr += addrCalc.incrementToNextRow(kernel, "D", ss, tmpS01)
          kStr += inst("v_mov_b32", vgpr(tmpVgpr), addrCalc.rowInc, "set shift rows")
          kStr += inst("_v_add_u32", vgpr(self.storeRemapCoord1), vgpr(self.storeRemapCoord1), vgpr(tmpVgpr), "shift storeRemap coord1")

        # apply in-bounds exec mask
        if edge and not kernel["BufferStore"]:
          kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec" )

        if beta:
          # if GWVW=1 the half path still assumes we have
          # at least two stores so does some combining across VI -
          # for example assuming we can have two elements and can use pk_mul
          # here:
          if beta and interleaveStoreVmcnt:
            if self.archCaps["SeparateVscnt"]:
              vmcnt = loadsIssued - elementIdx - 1
              vmComment = "{} = {} - {} - 1".format(vmcnt, loadsIssued, elementIdx)
            else:
              waitStoreCnt = storesIssued if not kernel["GroupLoadStore"] else 0
              vmcnt = loadsIssued - elementIdx + waitStoreCnt - 1
              vmComment = "{} = {} - {} + {} - 1".format(vmcnt, loadsIssued, elementIdx, waitStoreCnt)

            maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
            vmcnt = min(vmcnt, maxVmcnt)
            #print "wmvcnt=", vmcnt
            kStr += "\n"
            if not atomicAddC:
              kStr += inst("s_waitcnt", "vmcnt(%u)"%vmcnt, "wait C (interleaved) " + vmComment)

            # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
            # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
            # And this only occurs when beta=true, so case must not be 2 or 3
            assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, \
              "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

          for vi in range(0, gwvw):
            dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
            sumIdxV = ss.elementSumIdx[elementIdx] + vi
            if kernel["ProblemType"]["DestDataType"].isHalf():
              if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
                if self.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"]:
                  dataV = ss.elementData[elementIdx] + int(vi / 2 * ss.cfg.numVgprsPerDataPerVI)
                  if (vi % 2) == 0:
                      kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                          "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                  else:
                      kStr += inst("v_lshrrev_b32", vgpr(dataV), 16, vgpr(dataV), \
                          "shift 16bit to get next half of packed ValueC")
                  # dataV+0 = new c = old c*beta + rC
                  kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%(sumIdxV)), vgpr(dataV), vgpr("ValuC+%u"%(sumIdxV)), \
                      "sum*alpha + C*beta")
                elif sumIdxV%2==0 or (not self.ss.cfg.halfDataRegPerVI and gwvw==1):
                  # dataV+0 = new c = old c*beta
                  kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                      "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                  # dataV+0 = new c = old c*beta + rC
                  kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%(sumIdxV//2)), vgpr(dataV), vgpr("ValuC+%u"%(sumIdxV//2)), \
                      "sum*alpha + C*beta")
                else:
                  pass # add will have been done previously
              else: # HPA
                # dataV+0 = new c = old c*beta + rC
                # src0 = beta = f32 = opsel 00
                # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                # src2 = sumIdxV = f32 = opsel 00
                dataCExternal = ss.elementData[elementIdx] + vi//2
                hi16 = (vi + gwvw*vc0) % 2
                kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), sgpr("Beta"), \
                    vgpr(dataCExternal), vgpr("ValuC+%u"%sumIdxV), \
                    "op_sel:[0,%u,0] op_sel_hi:[0,1,0]" % (hi16), \
                    "//C*=beta")

            elif kernel["ProblemType"]["DestDataType"].isBFloat16():
              if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                # dataV+0 = new c = old c*beta + rC
                # src0 = beta = f32 = opsel 00
                # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                # src2 = sumIdxV = f32 = opsel 00
                dataCExternal = ss.elementData[elementIdx] + vi//2
                if (vi%2) == 1:
                  kStr += inst("v_and_b32", vgpr(tmpVgpr), vgpr(dataCExternal), vgpr(vgprBf16Mask), "convert bf16 to fp32")
                else:
                  kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), "16", vgpr(dataCExternal), "convert bf16 to fp32" )
                kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(tmpVgpr), sgpr("Beta"), \
                    "finalSum = sum*alpha + C*beta")

            # 8bit-Float: dest can only be either F8 or B8
            elif kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8(): # F8 is always HPA
              # dataV+0 = new c = old c*beta + rC
              dataCExternal = ss.elementData[elementIdx] + vi//4

              #restructuring the code to handle edge case
              if kernel["ProblemType"]["Fp8NoPackUpConversion"] or gwvw == 1:
                byteSel = "src0_sel:BYTE_0" if vi%4==0 else "src0_sel:BYTE_1" if vi%4==1 else "src0_sel:BYTE_2" if vi%4==2 else "src0_sel:BYTE_3"
                if kernel["ProblemType"]["DestDataType"].isFloat8():
                  kStr += "v_cvt_f32_fp8 v%u, %s %s   // convert fp8 in lo_byte[0] to f32%s"%(vgprF8Temp0, vgpr(dataCExternal), byteSel, self.endLine)
                else:
                  kStr += "v_cvt_f32_bf8 v%u, %s %s   // convert bf8 in lo_byte[0] to f32%s"%(vgprF8Temp0, vgpr(dataCExternal), byteSel, self.endLine)
                kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(vgprF8Temp0), sgpr("Beta"), \
                  "finalSum = sum*alpha + C*beta")
              else: ## use packing inst
                if kernel["ProblemType"]["DestDataType"].isFloat8():
                  f32Tof8PkInst="V_cvt_pk_f32_fp8"
                else: #bf8
                  f32Tof8PkInst="V_cvt_pk_f32_bf8"

                if (vi%2) == 1: ## vi%4 == 1 or vi%4 == 3
                  wordSel = "" if vi%4 == 1 else "src0_sel:WORD_1"
                  wordTxt = "lo_16" if vi%4==1 else "hi_16"
                  kStr += f32Tof8PkInst + " v[%u:%u], %s  %s  // convert two f8 in %s to f32%s"%(vgprF8Temp0, vgprF8Temp1, vgpr(dataCExternal), wordSel, wordTxt, self.endLine)

                  kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV-1)), vgpr(vgprF8Temp0), sgpr("Beta"), \
                    "finalSum = sum*alpha + C*beta")
                  kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(vgprF8Temp1), sgpr("Beta"), \
                    "finalSum = sum*alpha + C*beta")

            elif kernel["ProblemType"]["DestDataType"].isSingle():
              kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), sgpr("Beta"), \
                  "finalSum = sum*alpha + C*beta")

            elif kernel["ProblemType"]["DestDataType"].isInt32():
              # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
              # v_mad_i32_i24
              # kStr += inst("v_mad_i32_i24", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), sgpr("Beta"), vgpr("ValuC+%u"%sumIdxV), \
              #     "finalSum = sum*alpha + C*beta")
              kStr += inst("v_mul_lo_u32", vgpr(dataV+0), sgpr("Beta"), vgpr(dataV+0), \
                  "C = C*beta")
              kStr += inst("_v_add_u32", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), vgpr("ValuC+%u"%sumIdxV), \
                  "finalSum = sum*alpha + C*beta")

            elif kernel["ProblemType"]["DestDataType"].isDouble():
              # dataV+0 = new c = old c*beta
              if not atomicAddC:
                kStr += inst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*2),2), vgpr(dataV+0,2), sgpr("Beta",2), vgpr("ValuC+%u"%(sumIdxV*2),2), \
                    "finalSum = sum*alpha + C*beta")
              if kernel["StoreCInUnroll"] and not atomicAddC and not edge:
                # generate beta code
                vregIdx = vi*regsPerScalar + elementIdx*gwvw*regsPerScalar
                if kernel["AssertBetaValue"] == 1:
                  if kernel["AssertAlphaValue"] == 1 or kernel["AssertAlphaValue"] == -1:
                    # beta == 1 and alpha == 1 or -1 case. Use add instead of fma
                    minusStr = ""
                    if kernel["AssertAlphaValue"] == -1:
                      # special case for alpha == -1. Add"-" before src0
                      minusStr = "-"
                    BetaCodeMod.addCode(inst("v_add_f64", vgpr("L2GC+%u"%(vregIdx),2), minusStr + vgpr("L2GC+%u"%(vregIdx),2), vgpr("G2LC+%u"%(vregIdx),2),"finalSum = sum*alpha + C*beta"))
                  else:
                    # beta == 1 and alpha != (1 or -1) case. Use fma for alpha.
                    BetaCodeMod.addCode(inst("v_fma_f64", vgpr("L2GC+%u"%(vregIdx),2), vgpr("L2GC+%u"%(vregIdx),2), sgpr("Alpha",2), vgpr("G2LC+%u"%(vregIdx),2),"finalSum = sum*alpha + C*beta"))
                else:
                  # beta != 1 case. Use fma.
                  BetaCodeMod.addCode(inst("v_fma_f64", vgpr("L2GC+%u"%(vregIdx),2), vgpr("G2LC+%u"%(vregIdx),2), sgpr("Beta",2), vgpr("L2GC+%u"%(vregIdx),2),"finalSum = sum*alpha + C*beta"))


            # single precision complex
            elif kernel["ProblemType"]["DestDataType"].isSingleComplex():
              kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), vgpr(dataV+0), sgpr("Beta"), "finalSum Cr += old Cr * Br")
              kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), vgpr(dataV+1), "-"+sgpr("Beta+1"), "finalSum Cr += old Ci * -Bi")
              kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr(dataV+1), sgpr("Beta"), "finalSum Ci += old Ci * Br")
              kStr += inst("_v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr(dataV+0), sgpr("Beta+1"), "finalSum Ci += old Cr * Bi")

            # double precision complex
            elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
              # c.real += a.real * b.real
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr(dataV+0,2), sgpr("Beta+0",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), self.endLine)
              # c.real -= a.imag * b.imag
              kStr += "v_fma_f64 %s, %s, -%s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr(dataV+2,2), sgpr("Beta+2",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), self.endLine)
              # c.imag += a.real * b.imag
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(dataV+0,2), sgpr("Beta+2",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), self.endLine)
              # c.imag += a.imag * b.real
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(dataV+2,2), sgpr("Beta+0",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), self.endLine)

              if kernel["StoreCInUnroll"] and not atomicAddC and not edge:
                # generate beta code for StoreCInUnroll
                vregIdx = vi*regsPerScalar + elementIdx*gwvw*regsPerScalar
                # c.real += a.real * b.real
                BetaCodeMod.addCode("v_fma_f64 %s, %s, %s, %s%s" % (vgpr("L2GC+%u"%(vregIdx),2), vgpr("G2LC+%u"%(vregIdx),2), sgpr("Beta+0",2), vgpr("L2GC+%u"%(vregIdx),2), self.endLine))
                # c.real -= a.imag * b.imag
                BetaCodeMod.addCode("v_fma_f64 %s, %s, -%s, %s%s" % (vgpr("L2GC+%u"%(vregIdx),2), vgpr("G2LC+%u"%(vregIdx+2),2), sgpr("Beta+2",2), vgpr("L2GC+%u"%(vregIdx),2), self.endLine))
                # c.imag += a.real * b.imag
                BetaCodeMod.addCode("v_fma_f64 %s, %s, %s, %s%s" % (vgpr("L2GC+%u"%(vregIdx+2),2), vgpr("G2LC+%u"%(vregIdx),2), sgpr("Beta+2",2), vgpr("L2GC+%u"%(vregIdx+2),2), self.endLine))
                # c.imag += a.imag * b.real
                BetaCodeMod.addCode("v_fma_f64 %s, %s, %s, %s%s" % (vgpr("L2GC+%u"%(vregIdx+2),2), vgpr("G2LC+%u"%(vregIdx+2),2), sgpr("Beta+0",2), vgpr("L2GC+%u"%(vregIdx+2),2), self.endLine))

        # pack stores, beta and non-beta reach here:
        if kernel["ProblemType"]["HighPrecisionAccumulate"] and (kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
          for vi in range(0, gwvw):
            sumIdxV = ss.elementSumIdx[elementIdx] + vi
            if kernel["ProblemType"]["DestDataType"].isHalf():
              kStr += inst("v_cvt_f16_f32", vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), "convert C to fp16" )
              if vi%2 == 1:
                assert (gwvw % 2 == 0)
                d = ss.elementSumIdx[elementIdx] + vi//2
                kStr += inst("v_pack_b32_f16", vgpr(d), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%sumIdxV), "Pack with neighbor" )

            elif kernel["ProblemType"]["DestDataType"].isBFloat16():
              kStr += inst("v_cmp_u_f32", sgpr(tmpS01,laneSGPRC), vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), "check Nan" )
              kStr += inst("v_bfe_u32", vgpr(vgprBf16Temp), vgpr("ValuC+%u"%sumIdxV), "16", "1", "Non-Nan case: store lsb of bf16" )
              kStr += inst("v_add3_u32", vgpr(vgprBf16Temp), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprBf16Inc), "Non-Nan case: add lsb and the increment for rounding" )
              kStr += inst("v_cndmask_b32", vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprFp32Nan), sgpr(tmpS01,laneSGPRC), "" )
              if vi%2 == 0:
                kStr += inst("v_lshrrev_b32", vgpr("ValuC+%u"%sumIdxV), "16", vgpr("ValuC+%u"%sumIdxV), "convert C to bf16" )
              elif vi%2 == 1:
                d = ss.elementSumIdx[elementIdx] + vi//2
                kStr += inst("v_and_or_b32", vgpr(d), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Mask), vgpr("ValuC+%u"%(sumIdxV-1)), "pack two bf16 to dword")

            elif kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8():
              #assert (gwvw % 4 == 0)
              d = ss.elementSumIdx[elementIdx] + vi//4
              #NON-SR(RNE CVT) case. NOTE: no non-pack CVT version for RNE rounding
              if not kernel["ProblemType"]["StochasticRounding"]:
                #handle edge case...
                #NOTE: no non-pack inst, so same f32 values are converted and packed in one vgpr.
                if gwvw == 1:
                  if kernel["ProblemType"]["Fp32toFp8SWClip"]:
                    kStr += inst("v_cmp_class_f32", sgpr(tmpS01,laneSGPRC), vgpr("ValuC+%u"%(sumIdxV)), vgpr(vgprFp32NanInfFlag), "check NaN and +/-INF" )
                    kStr += inst("v_med3_f32", vgpr(vgprF8Temp0), vgpr("ValuC+%u"%(sumIdxV)), vgpr(vgprF8Max), vgpr(vgprF8Min), "Clipping f32 value if exceeds the limit")
                    kStr += inst("v_cndmask_b32", vgpr(vgprF8Temp0), vgpr(vgprF8Temp0), vgpr("ValuC+%u"%(sumIdxV)), sgpr(tmpS01,laneSGPRC), "" )

                    if kernel["ProblemType"]["DestDataType"].isFloat8():
                      f32Tof8PkInst = "v_cvt_pk_fp8_f32"
                    else:
                      f32Tof8PkInst = "v_cvt_pk_bf8_f32"

                    src0 = vgpr(vgprF8Temp0) if kernel["ProblemType"]["Fp32toFp8SWClip"] else vgpr("ValuC+%u"%(sumIdxV))
                    instStr = "%s  %s, %s, %s " \
                            %(f32Tof8PkInst, vgpr(d), src0, src0)
                    comment = "convert f32 accumulated values to fp8"
                    kStr += "%-50s // %s\n" %(instStr, comment)

                #non-edge case and even iteration
                elif vi%2 == 1:  # vi%4 == 1 or vi%4 == 3:
                  # S/W clipping code for down conversion
                  if kernel["ProblemType"]["Fp32toFp8SWClip"]:
                    kStr += inst("v_cmp_class_f32", sgpr(tmpS01,laneSGPRC), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr(vgprFp32NanInfFlag), "check NaN and +/-INF" )
                    kStr += inst("v_med3_f32", vgpr(vgprF8Temp0), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr(vgprF8Max), vgpr(vgprF8Min), "Clipping f32 value if exceeds the limit")
                    kStr += inst("v_cndmask_b32", vgpr(vgprF8Temp0), vgpr(vgprF8Temp0), vgpr("ValuC+%u"%(sumIdxV-1)), sgpr(tmpS01,laneSGPRC), "" )

                    kStr += inst("v_cmp_class_f32", sgpr(tmpS01,laneSGPRC), vgpr("ValuC+%u"%(sumIdxV)), vgpr(vgprFp32NanInfFlag), "check NaN and +/-INF" )
                    kStr += inst("v_med3_f32", vgpr(vgprF8Temp1), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprF8Max), vgpr(vgprF8Min), "Clipping f32 value if exceeds the limit")
                    kStr += inst("v_cndmask_b32", vgpr(vgprF8Temp1), vgpr(vgprF8Temp1), vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01,laneSGPRC), "" )
                  if kernel["ProblemType"]["DestDataType"].isFloat8():
                    f32Tof8PkInst = "v_cvt_pk_fp8_f32"
                  else:
                    f32Tof8PkInst = "v_cvt_pk_bf8_f32"

                  opSelFlag = "[0,0,0]"
                  opSelComment = "lo_16[0:15]"
                  if vi%4 == 3:
                    opSelFlag = "[0,0,1]"
                    opSelComment = "hi_16[16:31]"

                  src0 = vgpr(vgprF8Temp0) if kernel["ProblemType"]["Fp32toFp8SWClip"] else vgpr("ValuC+%u"%(sumIdxV-1))
                  src1 = vgpr(vgprF8Temp1) if kernel["ProblemType"]["Fp32toFp8SWClip"] else vgpr("ValuC+%u"%sumIdxV)
                  # NOTE: inst() always adds ',' in-between parameters and doesn't work with op_sel
                  instStr = "%s  %s, %s, %s op_sel:%s" \
                            %(f32Tof8PkInst, vgpr(d), src0, src1, opSelFlag)
                  comment = "convert two f32 accumulated values to fp8 and save it to %s" %(opSelComment)
                  kStr += "%-50s // %s\n" %(instStr, comment)
                  # When S/W clipping is NOT set, we need nop in between CVT insts
                  # NOTE: no 4 clk forwarding on these operations, must insert a NOP or instruction writing some other destination VREG
                  # between the conversions writing the low/high half or bytes of the same destination register.
                  if not kernel["ProblemType"]["Fp32toFp8SWClip"]:
                    kStr += "s_nop 0 %s"%self.endLine

                  # TODO: do we need nop between last cvt inst and buffer store?
                  #if kernel["ProblemType"]["Fp32toFp8SWClip"]:
                  #  kStr += "s_nop 0 %s"%self.endLine

              ## StochasticRounding with in-device RND
              elif kernel["ProblemType"]["StochasticRounding"]:
                # SR version down conversion inst doesn't support packing
                # Note: no need of any special code for edge case; when gwvw=1, vi=0.
                if kernel["ProblemType"]["Fp32toFp8SWClip"]:
                  kStr += inst("v_cmp_class_f32", sgpr(tmpS01,laneSGPRC), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprFp32NanInfFlag), "check Nan and +/-INF" )
                  kStr += inst("v_med3_f32", vgpr(vgprF8Temp0), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprF8Max), vgpr(vgprF8Min), "Clipping f32 value if exceeds the limit")
                  kStr += inst("v_cndmask_b32", vgpr(vgprF8Temp0), vgpr(vgprF8Temp0), vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01,laneSGPRC), "" )

                src0 = vgpr(vgprF8Temp0) if kernel["ProblemType"]["Fp32toFp8SWClip"] else vgpr("ValuC+%u"%sumIdxV)
                #calling PRND_GENERATOR
                # How to get vgpr # from src
                kStr += "PRND_GENERATOR %s %s %s %s %s" \
                     %(vgprF8Temp1, src0, vgprSRTemp0, vgprSRTemp1, self.endLine)

                if kernel["ProblemType"]["DestDataType"].isFloat8():
                  f32Tof8SrInst = "v_cvt_sr_fp8_f32"
                else:
                  f32Tof8SrInst = "v_cvt_sr_bf8_f32"
                opSel = "[0,0,0,0]" if vi%4==0 else "[0,0,1,0]" if vi%4==1 else "[0,0,0,1]" if vi%4==2 else "[0,0,1,1]"
                dPos = "dest[7:0]" if vi%4==0 else "dest[15:8]" if vi%4==1 else "dest[23:16]" if vi%4==2 else "dest[31:24]"
                kStr += "%s  %s, %s, %s op_sel:%s  // convert f32 accumulated values to fp8 and save it to %s %s" \
                        %(f32Tof8SrInst, vgpr(d), src0, vgpr(vgprF8Temp1), opSel, dPos, self.endLine)

                #TODO: need to check if we need this nop... there can be a buffer store inst right after the CVT
                #kStr += "s_nop 0 %s"%self.endLine

        if self.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"] and kernel["ProblemType"]["DestDataType"].isHalf() and (not kernel["ProblemType"]["HighPrecisionAccumulate"]):
          for vi in range(0, gwvw):
            sumIdxV = ss.elementSumIdx[elementIdx] + vi
            if vi%2 == 1:
              d = ss.elementSumIdx[elementIdx] + vi//2
              kStr += inst("v_pack_b32_f16", vgpr(d), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%sumIdxV), "Pack with neighbor" )

        if not kernel["StoreRemapVectorWidth"]:
          tmpStoreCode = self.addStore(kernel, ss, addrCalc, sumIdx, tmpS01, edge)
          if kernel["GroupLoadStore"]:
            storeCode += tmpStoreCode
          else:
            kStr += tmpStoreCode
          if kernel["StoreCInUnroll"] and not edge:
            if kernel["AtomicAddC"]:
              StoreInst = "buffer_atomic_add_f64"
              StoreComment = "AtomicAddC"
              numDstReg = 2
            else:
              StoreComment = "store D"
              if kernel["StoreVectorWidth"] == 1 and not kernel["ProblemType"]["DestDataType"].isDoubleComplex():
                StoreInst = "_buffer_store_b64"
                numDstReg = 2
              else : # kernel["StoreVectorWidth"] == 2 or DoubleComplex
                StoreInst = "_buffer_store_b128"
                numDstReg = 4

            if ss.optSrdIncForRow and addrCalc.rowInc:
              tempStr = addrCalc.incrementToNextRow(kernel, "D", ss, "StoreCOffsetAddr")
              StoreCCodeMod.addCode(tempStr)
            tempStr = inst(StoreInst, vgpr("L2GC+%s"%(elementIdx * numDstReg), numDstReg), vgpr(addrCalc.addrDVgpr), sgpr("SrdD", 4), "0", "offen offset:{}".format(addrCalc.globalOffset), StoreComment)
            StoreCCodeMod.addCode(tempStr)
          storesIssued += 1

        else:
          rpe = self.bpeCinternal//self.bpr
          kStr += self.storeRemapAddLocalWrite(kernel, ss, addrCalc, sumIdx*rpe)
          # Column Block Shape has been written to LDS
          # Now read back and write out to global memory

      kStr += storeCode

      #kStr += self.bomb(5)
      if self.db["CheckStoreC"]>=0:
        useBuffer = kernel["BufferStore"]
        # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
        # Note - TODO- CheckStoreC also won't work for StoreRemap
        kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
        for elementIdx in range(0, len(batchElements)):
          addr = ss.elementAddr[elementIdx].addrDVgpr
          sumIdx = ss.elementSumIdx[elementIdx]

          bps = kernel["ProblemType"]["DestDataType"].numBytes() * gwvw
          if kernel["BufferStore"]:
            addr0 = vgpr(addr)
            addr1 = sgpr("SrdC", 4)
          else:
            addr0 = vgpr(addr,2)
            addr1 = ""

          if kernel["ProblemType"]["DestDataType"].isHalf() or kernel["ProblemType"]["DestDataType"].isBFloat16():
            if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
              kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
                        addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=sumIdx%2).toStr()
            else:
              kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                        addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=0).toStr()
          elif kernel["ProblemType"]["DestDataType"].isInt32() or kernel["ProblemType"]["DestDataType"].isSingle():
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                      addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
          elif kernel["ProblemType"]["DestDataType"].isDouble() or kernel["ProblemType"]["DestDataType"].isSingleComplex() :
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
                      addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
          elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
                      addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

        # Add checks for expected values:
        kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["CheckStoreC"], "expected value")
        for elementIdx in range(0, len(batchElements)):
          sumIdx = ss.elementSumIdx[elementIdx]
          # Need to fix for other types:
          assert (kernel["ProblemType"]["DestDataType"].isSingle() or kernel["ProblemType"]["DestDataType"].isInt32())
          kStr += self.assert_eq(vgpr(sumIdx), sgpr(tmpS01))


      if edge and (atomic or not kernel["BufferStore"]):
        # subsequent batch must start with full exec mask
        # BufferStore doesn't need exec since it used buffer range checking when
        # possible
        kStr += inst("s_mov_b{}".format(wavelen), self.exec, -1, "full mask -> exec" )

      if self.db["ConservativeWaitCnt"] & 0x40:
        kStr += "s_barrier // debug\n"
        kStr += inst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
        kStr += "s_barrier // debug\n"

    # return registers to pool:
    lastData = -1
    for elementIdx in range(0, len(batchElements)):
      if not ss.sharedColDVgprs:
        addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
        addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
        self.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.vgprPool.checkIn(addrCVgpr)

      data = ss.elementData[elementIdx]
      if data != 0:
        if data != lastData:
          self.vgprPool.checkIn(data)
        lastData = data

    self.ss.firstBatch = False
    self.ss.checkInTempVgprC()
    if kernel["StoreRemapVectorWidth"]:
      if self.StoreRemapLastBatch == 1:
        kStr += self.comment("Handle local read and global write")
        # this seems buggy? it's possible to issue more than one stores for SR
        # kStr += self.storeRemapAddStore(kernel, ss, addrCalc, tmpVgpr, tmpS01, edge)
        # storesIssued += 1
        storeStr, numNewStores = self.storeRemapAddStore(kernel, ss, addrCalc, tmpVgpr, tmpS01, edge)
        kStr += storeStr
        storesIssued += numNewStores

    if self.serializedStore:
      kStr += inst("s_nop 0", "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst")

    # Update the store cnt to preLoopVmcntDict for Case2/3
    # (No need to update for Case0:'Undefined' or Case4:'OrdNLL_B1_Store')
    if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
      if not self.archCaps["SeparateVscnt"]:
        self.preLoopVmcntDict[self.currPreLoopVmcntCase] += storesIssued

    ##update main Code sections
    if kernel["StoreCInUnroll"] and not edge:
      self.AlphaOpTemplate.addModule(AlphaCodeMod)
      if not kernel["AtomicAddC"]:
        self.BetaOpTemplate.addModule(BetaCodeMod)
        self.LoadCTemplate.addModule(LoadCCodeMod)
      self.StoreCTemplate.addModule(StoreCCodeMod)
      self.accVgprTemplate.addModule(accVgprRead)

    return kStr

  ##############################################################################
  # generate alpha code for Global Write Batch
  ##############################################################################
  def generateAlphaCodeForGlobalWriteBatch(self, kernel, ss, codeMulAlpha, batchElements, gwvw, tmpS01, tmpVgpr):
    kStr = ""
    kStr += self.comment("rC *= alpha batchElements=%s"%batchElements)
    if codeMulAlpha is None:
      for elementIdx in range(0, len(batchElements)):
        kStr += self.applyAlpha(kernel, gwvw, ss.elementSumIdx, elementIdx, tmpS01)
    else:
        regsPerScalar = self.bpeCinternal//self.bpr # register per scalar
        for elementIdx in range(0, len(batchElements)):
          for vi in range(0, gwvw):
            alphaCode = str(codeMulAlpha.items().pop(0))
            if "__placeholdervtmp" in alphaCode:
              # allocate vtmp regs for alpha calculation
              num = kernel["ProblemType"]["DataType"].numRegisters()
              num2 = num//2
              vtmp1 = tmpVgpr
              vtmp2 = vtmp1 + num2
            alphaCode = alphaCode.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi ))
            if "__placeholdervtmp1__" in alphaCode:
              alphaCode = alphaCode.replace("__placeholdervtmp1__", vgpr(vtmp1,num2))
            if "__placeholdervtmp2__" in alphaCode:
              alphaCode = alphaCode.replace("__placeholdervtmp2__", vgpr(vtmp2,num2))
            kStr += alphaCode
    return kStr

  ##############################################################################
  def openPrefetchAcrossPersistent(self, kernel, isOptNLL, useBufferOOB=False):
    label = "SkipTo_PureOptNLL_LastTile" if isOptNLL else "SkipPrefetchAcrossPersistent"
    imod = Code.Module()
    #if useBufferOOB:
    #  maskTmp = self.getTmpSgpr(3,2).idx()
    #  stmp = maskTmp + 2
    #else:
    stmp = self.getTmpSgpr(1).idx()
    imod.addCode(self.comment3("PrefetchAcrossPersistent - Open"))
    imod.addInst("s_mul_i32", sgpr(stmp), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), "Total WG-0x1")
    if kernel["PersistentKernelAlongBatch"]:
      imod.addInst("s_mul_i32", sgpr(stmp), sgpr(stmp), sgpr("NumWorkGroups2"), "Total WG-0 x 1 x 2")
    imod.addInst("s_cmp_ge_u32", sgpr("SerialWorkGroupIter"), sgpr(stmp), "outside legal WG?")
    if useBufferOOB:
      #vtmp = self.vgprPool.checkOut(1)
      #imod.addInst("s_cselect_b32", sgpr(maskTmp), 0, hex(0xffffffff), "mask 1")
      #imod.addInst("s_mov_b32", sgpr(maskTmp+1), sgpr(maskTmp), "mask 2")
      #imod.addInst("s_mov_b32", sgpr(stmp), "BufferOOB", "OOB")
      #imod.addInst("v_mov_b32", vgpr(vtmp), sgpr(stmp), "OOB V")
      #numOffset = 1 if kernel["_UseSgprForGRO"] else self.numGlobalReadOffsetsA
      #for i in range(numOffset):
      #  imod.addInst("v_cndmask_b32", "v[vgprGlobalReadOffsetA+%u]"%i, vgpr(vtmp), "v[vgprGlobalReadOffsetA]", sgpr(maskTmp, 2), "mask 3")
      #numOffset = 1 if kernel["_UseSgprForGRO"] else self.numGlobalReadOffsetsB
      #for i in range(numOffset):
      #  imod.addInst("v_cndmask_b32", "v[vgprGlobalReadOffsetB+%u]"%i, vgpr(vtmp), "v[vgprGlobalReadOffsetB]", sgpr(maskTmp, 2), "mask 4")
      #self.vgprPool.checkIn(vtmp)
      #
      # reseting SrdA/B, ShadowLimitA/B, GlobalReadIncsA/B is more efficiently way than using BufferOOB
      imod.addInst("s_cmov_b32", sgpr("SrdA+2"), 0, "Set SrdA+2 to 0 for outside legal WG")
      imod.addInst("s_cmov_b32", sgpr("SrdB+2"), 0, "Set SrdB+2 to 0 for outside legal WG")
      imod.addInst("s_cmov_b64", sgpr("ShadowLimitA", 2), 0, "Set ShadowLimitA to 0 for outside legal WG")
      imod.addInst("s_cmov_b64", sgpr("ShadowLimitB", 2), 0, "Set ShadowLimitB to 0 for outside legal WG")
      imod.addInst("s_cmov_b32", sgpr("GlobalReadIncsA"), 0, "Stop decrementing ShadowLimitA and incrementing SrdA for outside legal WG")
      imod.addInst("s_cmov_b32", sgpr("GlobalReadIncsB"), 0, "Stop decrementing ShadowLimitB and incrementing SrdB for outside legal WG")
    else:
      imod.addInst("s_cbranch_scc1", self.getNamedLabel(label), "skip pf if OOB - last tile no PAP, go to pure OptNLL")

    #imod.addInst("s_branch", self.getLabelTarget("SkipPrefetchAcrossPersistent"), "skip pf if OOB")
    return imod

  ##############################################################################
  def closePrefetchAcrossPersistent(self, kernel, isOptNLL, useBufferOOB=False):
    imod = Code.Module()
    if not isOptNLL:
      label =  "SkipPrefetchAcrossPersistent"
      # imod.addCode(Code.WaitCnt(self.version, 0,0, "bozo, conservative wait"))
      if not useBufferOOB:
        imod.addCode("%s: //%s"%(self.getNamedLabel(label), "SkipPrefetchAcrossPersistent"))
      imod.addCode(self.comment3("PrefetchAcrossPersistent - Close"))
      #imod.addText(self.bomb())
    return imod

  ##############################################################################
  def openPrefetchGlobalRead2(self, kernel):
    imod = Code.Module()
    loopCounter = self.loopCounter(kernel, self.unrollIdx)
    imod.addInst("s_cmp_eq_u32 %s %s" %(loopCounter, hex(1)),"PGR=2 but only 1 loop")
    skipPGR2 = self.getLabelNum("skipPGR2")
    imod.addInst("s_cbranch_scc1 label_%04u" %(skipPGR2),"PGR=2 but only 1 loop")
    return imod

  def closePrefetchGlobalRead2(self, kernel):
    imod = Code.Module()
    skipPGR2 = self.getLabelNum("skipPGR2")
    imod.addInst("label_%04u:" % (skipPGR2),"")
    return imod

  ##############################################################################
  # Persistent Loop End long jump
  ##############################################################################
  def persistentLoopendLongjump(self, kernel):
    kStr = ""
    if kernel["StreamK"]:
      # endIter = "StreamKIterEnd" if kernel["StreamK"] == 1 else "TotalIters"
      endIter = "TotalIters" if kernel["StreamK"] == 2 else "StreamKIterEnd"
      kStr += inst("s_cmp_ge_u32", sgpr("StreamKIter"), sgpr(endIter), "Check if done all StreamK iterations")
      kStr += self.longBranchScc0(self.getLabelTarget("PersistentLoopStart"), negativeOnly=True)

    if kernel["PersistentKernel"]:
      # Persistent may generate a SerialWorkGroupIter which is OOB, only loop back if we are in a valid WG:
      stmp = self.getTmpSgpr(1).idx()
      kStr += inst("s_mul_i32", sgpr(stmp), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), "Total WG-0x1")
      if kernel["PersistentKernelAlongBatch"]:
        kStr += inst("s_mul_i32", sgpr(stmp), sgpr(stmp), sgpr("NumWorkGroups2"), "Total WG-0 x 1 x 2")
      kStr += inst("s_cmp_ge_u32", sgpr("SerialWorkGroupIter"), sgpr(stmp), "outside legal WG?")
      # use negative offset only long jump
      kStr += self.longBranchScc0(self.getLabelTarget("PersistentLoopStart"), negativeOnly=True)
    return kStr

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel, addLabel=True):
    imod = Code.Module()
    if kernel["PersistentKernel"] or kernel["StreamK"]:
      if kernel["StoreCInUnroll"]:
        # StoreCInUnroll case, reset StoreCAvail here for the next persistent loop (StoreCInUnroll disabled)
        imod.addCode(self.resetStoreCsyncObject(kernel))
      # Persistent may generate a SerialWorkGroupIter which is OOB, only loop back if we are in a valid WG:
      imod.addCode(self.persistentLoopendLongjump(kernel))
    if addLabel:
      imod.addCode(Code.Label(self.getLabelNum("KernelEnd"), "KernelEnd"))
    imod.addInst("s_endpgm", "Kernel End")
    return imod

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    kStr = ""
    if self.vgprPool.size() > self.maxVgprs:
      self.overflowedResources = 1
    elif self.sgprPool.size() > self.maxSgprs:
      self.overflowedResources = 2
    elif kernel["EnableMatrixInstruction"] and self.agprPool.size() > self.maxAgprs:
      self.overflowedResources = 7

    if kernel["ScheduleIterAlg"] == 2 and \
        self.getOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
        self.getLdsSize(kernel), self.agprPool.size(), self.doubleVgpr) < 2:
      self.overflowedResources = 6

    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU // kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy // max(self.vgprPool.size(), self.agprPool.size())
    if numWorkGroupsPerCU < 1:
      self.overflowedResources = 4

    if self.overflowedResources:
      kStr += ".endif // overflowed resources \n"

    self.vgprPool.checkFinalState()
    return kStr

  ##############################################################################
  # Code to populate code segments for storeC
  ##############################################################################
  def PopulateStoreCCode(self, kernel, tPA, tPB):
    ## generate store Code segments for storeInUnroll option
    ## generate only for edge=False and beta !=0 case
    ## code segments are populated in code templates later used in unroll Loop

    betas = [True] if not kernel["AtomicAddC"] and kernel["ProblemType"]["UseBeta"] else [False]
    edges = [False]
    applyAlpha = True
    # disable alpha for the following scenarios
    #   isDouble
    #   alpha = 1 or beta == 1 and not Atomic
    if kernel["ProblemType"]["DestDataType"].isDouble() and \
       (kernel["AssertAlphaValue"] == 1 or \
       (kernel["AssertBetaValue"] == 1 and not kernel["AtomicAddC"])):
      applyAlpha = False
    self.notLocalSplitUGlobalWriteIndices(kernel)
    (fullVw, elements) = self.notLocalFullTileElements(kernel, False)
    vectorWidths = [fullVw]

    self.MapAcctoArchRegs(kernel,option=0,isOptNLL=True)
    partElements = []
    numElements = 1
    if kernel["StoreVectorWidth"] == 1 and (not kernel["ProblemType"]["ComputeDataType"].isDoubleComplex()):
      # StoreVectorWidth==1 and not ZGEMM case, add elements[1] to cover 2 x2 load
      numElements = 2
    for i in range(numElements):
      partElements.append(elements[i])
    self.globalWriteElements(kernel, vectorWidths, [partElements], applyAlpha, betas, edges)

    self.cleanupGlobalWrite(kernel)

    return ""

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
    kStr = ""
    if kernel["StoreCInUnroll"]:
      self.PopulateStoreCCode(kernel,tPA, tPB)
    return kStr

  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  def kernelBodySuffix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # init for StoreCInUnroll
  ##############################################################################
  def initStoreCInUnroll(self, kernel):
    kStr = ""
    needAddrC = not kernel["AssertCEqualsD"] and kernel["ProblemType"]["UseBeta"]
    kStr += self.comment("Initialization for StoreCInUnroll")

    # reset StoreC sync object
    kStr += self.resetStoreCsyncObject(kernel)
    if needAddrC:
      # generate C address inc for first/second vertical offset
      kStr += self.generateInitialCorDaddrIncrement(kernel, "C")
    # generate D address inc for first/second vertical offset
    kStr += self.generateInitialCorDaddrIncrement(kernel, "D")

    # generate enable count init code
    kStr += self.generateStoreCEnableCountInitValue(kernel)
    if kernel["BufferStore"] or kernel["BufferLoad"]:
      # generate BufferOOB vreg value
      tmpSgpr  = self.getTmpSgpr(1).idx()
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), "BufferOOB", "" )
      kStr += inst("v_mov_b32", vgpr("GlobalBufferOOB"), sgpr(tmpSgpr), "" )
      # initialize srdC/DBackup to avoid accessing memory beyond the largest legal address
      if kernel["BufferStore"]:
        if needAddrC:
          kStr += inst("s_mov_b64", sgpr("SrdCBackup", 2), sgpr("SrdC", 2), "Initialize SrcCBackup")
        kStr += inst("s_mov_b64", sgpr("SrdDBackup", 2), sgpr("SrdD", 2), "Initialize SrcDBackup")

    return kStr

  ##############################################################################
  # init for StoreCInUnroll per Persistent Loop
  ##############################################################################
  def initStoreCInUnrollPerPersistentLoop(self, kernel):
    kStr = ""
    needLoadC = not kernel["AtomicAddC"] and kernel["ProblemType"]["UseBeta"]
    kStr += self.comment("Initialization for StoreCInUnroll per Persistent Loop")
    # reset StoreCIndex0
    kStr += inst("s_mov_b32", sgpr("StoreCIndex0"), hex(0), "Reset StoreC index")
    # set init value to StoreCEnableCount
    kStr += inst("s_mov_b32", sgpr("StoreCEnableCount"), sgpr("StoreCEnableCountInit"), "Set init value for StoreCEnableCount")
    if not kernel["StoreCInUnrollExact"]:
      # backup global read/write offset
      if kernel["BufferStore"]:
        # mask load/storeC based on StoreCAvail
        if needLoadC:
          kStr += inst("v_mov_b32", vgpr("GlobalReadOffsetCBackup"), vgpr("GlobalReadOffsetC"), "backup GlobalReadOffsetC")
        kStr += inst("v_mov_b32", vgpr("GlobalWriteOffsetDBackup"), vgpr("GlobalWriteOffsetD"), "backup GlobalWriteOffsetD")
    else:
      # StoreCInUnroll exact case, initialize offset C/D at the beginning of persistent loop
      # (no need to use backup of offset C/D)
      if kernel["BufferStore"]:
        # mask load/storeC based on StoreCAvail
        if needLoadC:
          addrDstVgpr = "GlobalReadOffsetC"
          addrSrcVgpr = "GlobalReadOffsetC"
          kStr += inst("v_cndmask_b32", \
              vgpr(addrDstVgpr), \
              vgpr("GlobalBufferOOB"), \
              vgpr(addrSrcVgpr), \
              sgpr("StoreCAvail",2), \
              "Mask load so OOB will return 0")
        addrDstVgpr = "GlobalWriteOffsetD"
        addrSrcVgpr = "GlobalWriteOffsetD"
        kStr += inst("v_cndmask_b32", \
            vgpr(addrDstVgpr), \
            vgpr("GlobalBufferOOB"), \
            vgpr(addrSrcVgpr), \
            sgpr("StoreCAvail",2), \
            "Mask store so OOB will return 0")

    return kStr

  ##############################################################################
  # init for StoreCInUnroll per Unroll Loop
  ##############################################################################
  def initStoreCInUnrollPerUnrollLoop(self, kernel, needInit):
    kStr = ""
    # init code for StoreCInUnroll per Unroll Loop is only for not odd case
    if not needInit:
      return kStr

    needLoadC = not kernel["AtomicAddC"] and kernel["ProblemType"]["UseBeta"]
    kStr += self.comment("Initialization for StoreCInUnroll per Unroll Loop")
    # decrement StoreCEnableCount
    kStr += inst("s_sub_u32",  sgpr("StoreCEnableCount"), sgpr("StoreCEnableCount"), hex(1), \
                 "decrement StoreCEnableCount. Set scc when StoreCEnableCount is -1")
    if not kernel["StoreCInUnrollExact"]:
      # Set StoreCAvalEach = 0 when StoreCEnableCount==-1 (scc=1 by s_sub_u32)
      #  if StoreCEnableCount >= 0, StoreCAvalEachLoop = StoreCAvail (means Load/StoreC enabled)
      #  else (if StoreCEnableCount <0, StoreCAvalEachLoop = 0 (means Load/StoreC notenabled)
      kStr += inst("s_cmov_b64", sgpr("StoreCAvail", 2), hex(0),  "Set UseStoreCAvail = 0 only when StoreCEnableCount==-1")

      if kernel["BufferStore"]:
        # mask load/storeC based on StoreCAvail
        if needLoadC:
          addrDstVgpr = "GlobalReadOffsetC"
          addrSrcVgpr = "GlobalReadOffsetCBackup"
          kStr += inst("v_cndmask_b32", \
              vgpr(addrDstVgpr), \
              vgpr("GlobalBufferOOB"), \
              vgpr(addrSrcVgpr), \
              sgpr("StoreCAvail",2), \
              "Mask load so OOB will return 0")
        addrDstVgpr = "GlobalWriteOffsetD"
        addrSrcVgpr = "GlobalWriteOffsetDBackup"
        kStr += inst("v_cndmask_b32", \
            vgpr(addrDstVgpr), \
            vgpr("GlobalBufferOOB"), \
            vgpr(addrSrcVgpr), \
            sgpr("StoreCAvail",2), \
            "Mask store so OOB will return 0")

    return kStr

  ##############################################################################
  # swap SrdC and SrdCbackup, SrdD and SrdDbackup
  ##############################################################################
  def swapSrdCDandBackup(self, kernel):
    kStr = ""
    needAddrC = not kernel["AssertCEqualsD"] and kernel["ProblemType"]["UseBeta"]
    if kernel["BufferStore"]:
      tmpSgpr  = self.getTmpSgpr(2,2).idx()
      # swap SrcC/D and SrcC/DBackup
      if needAddrC:
        kStr += inst("s_mov_b64", sgpr(tmpSgpr, 2), sgpr("SrdC", 2), "Swap SrcC and SrcCBackup")
        kStr += inst("s_mov_b64", sgpr("SrdC", 2), sgpr("SrdCBackup", 2), "Swap SrcC and SrcCBackup")
        kStr += inst("s_mov_b64", sgpr("SrdCBackup", 2), sgpr(tmpSgpr, 2), "Swap SrcC and SrcCBackup")
      kStr += inst("s_mov_b64", sgpr(tmpSgpr, 2), sgpr("SrdD", 2), "Swap SrcD and SrcDBackup")
      kStr += inst("s_mov_b64", sgpr("SrdD", 2), sgpr("SrdDBackup", 2), "Swap SrcD and SrcDBackup")
      kStr += inst("s_mov_b64", sgpr("SrdDBackup", 2), sgpr(tmpSgpr, 2), "Swap SrcD and SrcDBackup")

    return kStr

  ##############################################################################
  # restore SrdCbackup and SrdDbackup
  ##############################################################################
  def restoreSrdCandDBackup(self, kernel):
    kStr = ""
    needAddrC = not kernel["AssertCEqualsD"] and kernel["ProblemType"]["UseBeta"]
    if kernel["BufferStore"]:
      if needAddrC:
        kStr += inst("s_mov_b64", sgpr("SrdC", 2), sgpr("SrdCBackup", 2), "Restore SrcC and SrcCBackup")
      kStr += inst("s_mov_b64", sgpr("SrdD", 2), sgpr("SrdDBackup", 2), "Restore SrcD and SrcDBackup")

    return kStr

  ##############################################################################
  # initialization of StoreCInUnroll C/D addr inc
  ##############################################################################
  def initializeStoreCInUnrollAddrIncValues(self, kernel):
    # reference: NotLocalFullTileElementsMFMA
    # From here
    # handle mfma 4x4 instruction
    matrixInstM  = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
    matrixInstN  = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
    matrixInstBM = 1                                              if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
    matrixInstBN = 1                                              if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

    outputsPerThread = matrixInstM * matrixInstN // kernel["WavefrontSize"]

    # handle SourceSwap
    totalTT0     = matrixInstBM * kernel["MIWaveTile"][0]
    totalTT1     = matrixInstBN * kernel["MIWaveTile"][1]

    totalTT0     = totalTT0                      if kernel["SourceSwap"] else (totalTT0 * outputsPerThread)
    totalTT1     = (totalTT1 * outputsPerThread) if kernel["SourceSwap"] else totalTT1
    vectorWidth0 = kernel["VectorWidthA"]        if kernel["SourceSwap"] else kernel["VectorWidthA"] * kernel["MIOutputVectorWidth"]
    vectorWidth1 = kernel["VectorWidthB"] * kernel["MIOutputVectorWidth"] if kernel["SourceSwap"] else kernel["VectorWidthB"]

    # To here

    # Tile allocation patterns (MT=128x128 DGEMM SourceSwap)
    #  case 1: NumHorizontalTiles == 1 (MIWaveGroup=[4,1])
    #   case 1-1: NumInterleaveV = 1
    #                            0
    #     v1addrInc   gprInc1    1      (v1addrInc = 4 line,    gprInc1  = 2)
    #     v1addrInc   gprInc1    2
    #     v1addrInc   gprInc1    3
    #     v1addrInc   gprIncB1   4      (                       gprIncB1 = 16-6=10)
    #      ...    ...
    #     v1addrInc   gprInc1   31
    #   case 1-2: NumInterleaveV = 2
    #                             0
    #     v1addrInc   gprInc1     1      (v1addrInc = 1 line,    gprInc1  = 16)
    #     v2addrInc   gprIncB1    2      (v2addrInc = 7 line,    gprIncB1 = 2-16=-14)
    #     v1addrInc   gprInc1     3
    #     v2addrInc   gprIncB1    4
    #     v1addrInc   gprInc1     5
    #     v2addrInc   gprIncB1    6
    #     v1addrInc   gprInc1     7
    #     v2addrInc   gprIncB2    8      (v2addrInc = 7 line,    gprIncB2 = 32-22=10)
    #      ...    ...
    #     v2addrInc   gprInc1    30
    #     v1addrInc   gprIncB1   31
    #
    #  case 2: NumHorizontalTiles == 4 (MIWaveGroup=[1,4])
    #   case 2-1: NumInterleaveV = 1
    #                 gprInc1     0   1   2   3                        gprInc1  = 16)
    #     v1addrInc   gprIncB1    4   5   6   7  (v1addrInc =  4 line, gprIncB1 = 2-48=-46)
    #     v1addrInc   gprIncB1    8   9  10  11
    #     v1addrInc   gprIncB1   12  13  14  15
    #     v2addrInc   gprIncB2   16  17  18  19  (v2addrInc = 52 line, gprIncB2 = 64-54=10)
    #     v1addrInc   gprIncB1   20  21  22  23
    #     v1addrInc   gprIncB1   24  25  26  27
    #     v1addrInc   gprIncB1   28  29  30  31
    #   case 2-2: NumInterleaveV = 2
    #                 gprInc1     0   1   2   3                        gprInc1  = 16)
    #     v1addrInc   gprIncB1    4   5   6   7  (v1addrInc =  1 line, gprIncB1 = 64-48=16)
    #     v2addrInc   gprIncB2    8   9  10  11  (v2addrInc =  7 line, gprIncB2 = 2-112=-110)
    #     v1addrInc   gprIncB1   12  13  14  15
    #     v2addrInc   gprIncB2   16  17  18  19
    #     v1addrInc   gprIncB1   20  21  22  23
    #     v2addrInc   gprIncB2   24  25  26  27
    #     v1addrInc   gprIncB1   28  29  30  31
    #
    #  case 3: NumHorizontalTiles == 2 (MIWaveGroup=[1,4], MIWaveTile=[2,4], VW=2)
    #   case 3-2: NumInterleaveV = 2
    #                 gprInc1     0   1                        gprInc1  = 16)
    #     v1addrInc   gprIncB1    2   3  (v1addrInc =  1 line, gprIncB1 = 32-16=16)
    #     v2addrInc   gprIncB2    4   5  (v2addrInc =  7 line, gprIncB2 = 2-48=-46)
    #     v1addrInc   gprIncB1    6   7
    #     v2addrInc   gprIncB2    8   9
    #     v1addrInc   gprIncB1   10  11
    #     v2addrInc   gprIncB2   12  13
    #     v1addrInc   gprIncB1   14  15
    #     v3addrInc   gprIncB3   16  17  (v3addrInc =103 line, gprIncB3 = 64-54=10)
    #      ...    ...
    #     v1addrInc   gprIncB1   30  31
    #
    # TODO:
    #   1. MIWaveGroup = [2,2]
    #   2. numTiles > 32
    #   3. no SourceSwap

    bpe = self.bpeAB
    self.StoreCInUnrollNumReg = 2 # for double/double complex(TODO: support other types)
    self.StoreCInUnrollnumRows = outputsPerThread
    self.StoreCInUnrollAddrIncV1Iterations = 0 # how many iterations to use first vertical offset
    self.StoreCInUnrollGprIncB1Iterations = 0 # how many iterations to use second block offset for Gpr (can be different from addr inc)
    self.StoreCInUnrollAddrIncV2Iterations = 0 # how many iterations to use second vertical offset
    self.StoreCInUnrollGprIncB2Iterations = 0 # how many iterations to use second block offset for Gpr (can be different from addr inc)
    self.StoreCInUnrollAddrIncV3Iterations = 0 # how many iterations to use second vertical offset
    self.StoreCInUnrollGprIncB3Iterations = 0 # how many iterations to use second block offset for Gpr (can be different from addr inc)
    self.StoreCInUnrollAddrIncHoffset = 0 # horizontal address offset
    self.StoreCInUnrollAddrIncVlineOffset1 = 0 # vertical address line offset (for V1)
    self.StoreCInUnrollAddrIncVlineOffset2 = 0 # vertical address line offset (for V2)
    self.StoreCInUnrollAddrIncVlineOffset3 = 0 # vertical address line offset (for V3)
    self.StoreCInUnrollNumHorizontalTiles = totalTT0 // vectorWidth0
    self.StoreCInUnrollNumInterleaveV = vectorWidth1 # LocalReadVectorWidth for B (if 2, store lines are interleaved)
    if self.StoreCInUnrollNumHorizontalTiles == 1:
      if self.StoreCInUnrollNumInterleaveV == 1:
        # case 1-1:
        self.StoreCInUnrollAddrIncV1Iterations = 1
        self.StoreCInUnrollAddrIncVlineOffset1 = self.StoreCInUnrollnumRows * self.StoreCInUnrollAddrIncV1Iterations
        self.StoreCInUnrollAddrIncV2Iterations = 0 # will not be used
        self.StoreCInUnrollAddrIncVlineOffset2 = 0 # will not be used
        self.StoreCInUnrollGprIncB1Iterations = self.StoreCInUnrollnumRows
        self.StoreCInUnrollGprIncB2Iterations = 0 # will not be used
      else:
        # case 1-2:
        self.StoreCInUnrollAddrIncV1Iterations = 1
        self.StoreCInUnrollAddrIncVlineOffset1 = 1
        self.StoreCInUnrollAddrIncV2Iterations = self.StoreCInUnrollNumInterleaveV * self.StoreCInUnrollAddrIncV1Iterations
        self.StoreCInUnrollAddrIncVlineOffset2 = self.StoreCInUnrollNumInterleaveV * self.StoreCInUnrollnumRows
        self.StoreCInUnrollGprIncB1Iterations = self.StoreCInUnrollNumInterleaveV
        self.StoreCInUnrollGprIncB2Iterations = self.StoreCInUnrollGprIncB1Iterations * self.StoreCInUnrollnumRows
    else:
      # need horizontal offset
      self.StoreCInUnrollAddrIncHoffset = kernel["MIWaveGroup"][0] * vectorWidth0 * kernel["MatrixInstM"] * bpe
      if self.StoreCInUnrollNumInterleaveV == 1:
        # case 2-1:
        self.StoreCInUnrollAddrIncV1Iterations = self.StoreCInUnrollNumHorizontalTiles
        self.StoreCInUnrollAddrIncVlineOffset1 = self.StoreCInUnrollnumRows
        self.StoreCInUnrollAddrIncV2Iterations = 4 * self.StoreCInUnrollAddrIncV1Iterations
        self.StoreCInUnrollAddrIncVlineOffset2 = matrixInstN * kernel["MIWaveGroup"][1]
        self.StoreCInUnrollGprIncB1Iterations = self.StoreCInUnrollNumHorizontalTiles
        self.StoreCInUnrollGprIncB2Iterations = 4 * self.StoreCInUnrollNumHorizontalTiles
      else:
        # case 2-2:
        self.StoreCInUnrollAddrIncV1Iterations = self.StoreCInUnrollNumHorizontalTiles
        self.StoreCInUnrollAddrIncVlineOffset1 = 1
        self.StoreCInUnrollAddrIncV2Iterations = self.StoreCInUnrollNumInterleaveV * self.StoreCInUnrollAddrIncV1Iterations
        self.StoreCInUnrollAddrIncVlineOffset2 = self.StoreCInUnrollNumInterleaveV * self.StoreCInUnrollnumRows
        self.StoreCInUnrollGprIncB1Iterations = self.StoreCInUnrollNumHorizontalTiles
        self.StoreCInUnrollGprIncB2Iterations = self.StoreCInUnrollNumInterleaveV * self.StoreCInUnrollNumHorizontalTiles
        VlineOffset3 = kernel["MIWaveGroup"][1] * self.StoreCInUnrollNumInterleaveV * matrixInstN
        if kernel["MacroTile1"] > VlineOffset3:
          # v3 address is necessary
         self.StoreCInUnrollAddrIncVlineOffset3 = VlineOffset3
         self.StoreCInUnrollAddrIncV3Iterations = self.StoreCInUnrollAddrIncV2Iterations * self.StoreCInUnrollnumRows
         self.StoreCInUnrollGprIncB3Iterations = self.StoreCInUnrollnumRows * self.StoreCInUnrollGprIncB2Iterations

    if self.getStoreCLoopIterTimes(kernel) <= self.StoreCInUnrollAddrIncV2Iterations:
      # if loop iteration not larger than V2 iterations, second vertical offset will not be used
      self.StoreCInUnrollAddrIncV2Iterations = 0
    if self.getStoreCLoopIterTimes(kernel) <= self.StoreCInUnrollAddrIncV3Iterations:
      # if loop iteration not larger than V3 iterations, second vertical offset will not be used
      self.StoreCInUnrollAddrIncV3Iterations = 0

  ##############################################################################
  # initial C/D address increment values
  ##############################################################################
  def generateInitialCorDaddrIncrement(self, kernel, CorD):
    kStr = ""
    sgprStride = "Stride{}1J".format(CorD)
    bpe = self.bpeAB
    numLinesV1 = self.StoreCInUnrollAddrIncVlineOffset1
    numLinesV2 = self.StoreCInUnrollAddrIncVlineOffset2
    mulValue1 = numLinesV1 * bpe
    hOffset = self.StoreCInUnrollAddrIncHoffset
    if hOffset == 0 and self.StoreCInUnrollNumInterleaveV == 1:
      # 1. first vertical offset code only
      sgprDst = CorD + "AddrInc"
      kStr += inst("s_mul_i32", sgpr(sgprDst), sgpr(sgprStride), mulValue1, "scale Stride{} *= numRows({}) * bpe".format(CorD, numLinesV1))
    else:
      # 1. first vertical offset code
      sgprDst = CorD + "AddrIncV1"
      kStr += inst("s_mul_i32", sgpr(sgprDst), sgpr(sgprStride), mulValue1, "scale Stride{} *= numRows({}) * bpe".format(CorD, numLinesV1))

      # horizontal offset > 0 case, need to subtract horizontal offset from vertical offset
      backToLeft = (self.StoreCInUnrollNumHorizontalTiles - 1) * hOffset
      kStr += inst("s_addk_i32", sgpr(sgprDst), hex(-backToLeft), "")

      # 2. second vertical offset code
      V1Iterations = self.StoreCInUnrollAddrIncV1Iterations
      V2Iterations = self.StoreCInUnrollAddrIncV2Iterations
      if V2Iterations > 0:
        numV2LineOffset = numLinesV2
        # subtract V1 offsets (which is already added)
        numV2LineOffset -= ((V2Iterations//V1Iterations - 1) * numLinesV1)
        mulValue2 = numV2LineOffset * bpe
        sgprDst = CorD + "AddrIncV2"
        # 416 is too large. Need to split into 2 instructions (movk and mul)
        kStr += inst("s_movk_i32", sgpr(sgprDst), mulValue2, "")
        kStr += inst("s_mul_i32", sgpr(sgprDst), sgpr(sgprStride), sgpr(sgprDst), "scale Stride{} *= numRows({}) * bpe".format(CorD, numV2LineOffset))
        # subtract H offsets (which is already added)
        kStr += inst("s_addk_i32", sgpr(sgprDst), hex(-backToLeft), "")

      # 3. third vertical offset code
      V3Iterations = self.StoreCInUnrollAddrIncV3Iterations
      if V3Iterations > 0:
        numV3LineOffset = self.StoreCInUnrollAddrIncVlineOffset3
        # subtract V1,V2 offsets (which is already added)
        numV3LineOffset -= ((V2Iterations//V1Iterations - 1) * numLinesV1) + ((V3Iterations//V2Iterations - 1) * numLinesV2)
        mulValue3 = numV3LineOffset * bpe
        sgprDst = CorD + "AddrIncV3"
        # Need to split into 2 instructions (movk and mul)
        kStr += inst("s_movk_i32", sgpr(sgprDst), mulValue3, "")
        kStr += inst("s_mul_i32", sgpr(sgprDst), sgpr(sgprStride), sgpr(sgprDst), "scale Stride{} *= numRows({}) * bpe".format(CorD, numV3LineOffset))
        # subtract H offsets (which is already added)
        kStr += inst("s_addk_i32", sgpr(sgprDst), hex(-backToLeft), "")

    return kStr

  ##############################################################################
  # get StoreC loop iteration times
  ##############################################################################
  def getStoreCLoopIterTimes(self, kernel):
    StoreCLoopIterTimes = kernel["ThreadTile0"]*kernel["ThreadTile1"] // kernel["VectorWidthA"] # 128x128 case, 32 times. 128x64 case, 16 times for double
    return StoreCLoopIterTimes

  ##############################################################################
  # get supported K value (log 2) for StoreCInUnroll
  ##############################################################################
  def getSupportedKvalueLog2ForStoreCInUnroll(self, kernel):
    frequency = self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel)
    # minimum of supported K. So far, DepthU * storeC loop iter times
    minK = self.getStoreCLoopIterTimes(kernel) * kernel["DepthU"]
    # minimum of multiple support StoreCInUnroll. So far, DepthU * frequency
    supportMinKmask = (kernel["DepthU"] * frequency) - 1
    return minK, supportMinKmask

  ##############################################################################
  # StoreC enable count init value code
  ##############################################################################
  def generateStoreCEnableCountInitValue(self, kernel):
    kStr = ""
    frequency = self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel)
    storeIterTimes = self.getStoreCLoopIterTimes(kernel)
    initValue = storeIterTimes // frequency
    minK, supportMinKmask = self.getSupportedKvalueLog2ForStoreCInUnroll(kernel)
    kStr += inst("s_mov_b32", sgpr("StoreCEnableCountInit"), hex(initValue), \
                  "init StoreC enable counter for Tile %u x %u"%(kernel["MacroTile0"], kernel["MacroTile1"]))
    return kStr

  ##############################################################################
  # gpr index increment StoreCInUnroll
  ##############################################################################
  def generateSgprGprIdxIncrementForStoreCInUnroll(self, kernel,tmpSgprGprIdxInc, tmpSgprWork):
    kStr = ""
    sgprCounter = "StoreCEnableCount"
    tmpSgpr1 = tmpSgprWork
    frequency = self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel)
    HIterations = self.StoreCInUnrollNumHorizontalTiles
    B2Iterations = self.StoreCInUnrollGprIncB2Iterations
    B3Iterations = self.StoreCInUnrollGprIncB3Iterations
    inc2 = self.getAccVgprInc1(kernel, frequency)
    offsetB1 = self.getAccVgprOffsetB1(kernel)
    if HIterations == 1:
      # need adjustment for HIterations==1 case.
      # use self.StoreCInUnrollGprIncB1Iterations as HIterations
      HIterations = self.StoreCInUnrollGprIncB1Iterations
    numInc2 = HIterations // frequency
    incB1 = offsetB1 - inc2 * (numInc2 - 1)
    # first block increment (B1) check
    if incB1 == inc2:
      # this case, always use same inc value for gp
      kStr += inst("s_mov_b32", sgpr(tmpSgprGprIdxInc), hex(inc2),  "Use %u"%(inc2))
    else:
      maskB1 = numInc2 - 1 # StoreCEnableCount decremented at every 2 iterations
      #increment index
      kStr += inst("s_and_b32", sgpr(tmpSgpr1), sgpr(sgprCounter), hex(maskB1), "")
      kStr += inst("s_cselect_b32", sgpr(tmpSgprGprIdxInc), hex(inc2), hex(incB1),  "Use %u if %s & %u == 0 else %u"%(incB1, sgprCounter, maskB1, inc2))

    # second block increment (B2) check
    maskB2 = (B2Iterations//frequency) - 1 # StoreCEnableCount decremented at every 2 iterations
    offsetB2 = self.getAccVgprOffsetB2(kernel)
    numB1 = B2Iterations // HIterations
    #incB2 = offsetB2 - (incB1 * (numB1 - 1) + inc2 * numB1 * (numInc2 - 1))
    incB2 = offsetB2 - (offsetB1 * (numB1 - 1) + inc2 * (numInc2 - 1))
    if (B2Iterations > 0):
      if (incB2 != incB1):
        kStr += inst("s_and_b32", sgpr(tmpSgpr1), sgpr(sgprCounter), hex(maskB2), "")
        kStr += inst("s_cselect_b32", sgpr(tmpSgprGprIdxInc), sgpr(tmpSgprGprIdxInc), hex(incB2),  "Use %u if %s & %u == 0"%(incB2, sgprCounter, maskB2))

      # third block increment (B3) check
      maskB3 = (B3Iterations//frequency) - 1 # StoreCEnableCount decremented at every 2 iterations
      offsetB3 = self.getAccVgprOffsetB3(kernel)
      numB2 = B3Iterations // B2Iterations
      incB3 = offsetB3 - (offsetB2 * (numB2 - 1) + offsetB1 * (numB1 - 1) + inc2 * (numInc2 - 1))
      if (B3Iterations > 0) and (incB3 != incB2):
        kStr += inst("s_and_b32", sgpr(tmpSgpr1), sgpr(sgprCounter), hex(maskB3), "")
        kStr += inst("s_cselect_b32", sgpr(tmpSgprGprIdxInc), sgpr(tmpSgprGprIdxInc), hex(incB3),  "Use %u if %s & %u == 0"%(incB3, sgprCounter, maskB3))

    return kStr

  ##############################################################################
  # C/D address increment value for StoreCInUnroll
  ##############################################################################
  def generateCorDaddrIncrementForStoreCInUnroll(self, kernel, CorD, odd, tmpSgprWork):
    kStr = ""
    sgprInc = CorD + "AddrInc"
    sgprIncV1 = CorD + "AddrIncV1"
    sgprIncV2 = CorD + "AddrIncV2"
    sgprIncV3 = CorD + "AddrIncV3"
    tmpSgpr1 = tmpSgprWork
    sgprCounter = "StoreCEnableCount"
    frequency = self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel)
    hOffset = self.StoreCInUnrollAddrIncHoffset
    V2Iterations = self.StoreCInUnrollAddrIncV2Iterations
    V3Iterations = self.StoreCInUnrollAddrIncV3Iterations
    # v1 or v2 select code
    v1OrV2SelectCode = ""
    # if V2Iterations == frequency, always pick IncV2, otherwise, check if it is V2 or V1
    if V2Iterations == frequency:
      v1OrV2SelectCode += inst("s_mov_b32", sgpr(sgprInc), sgpr(sgprIncV2),  "Use AddrIncV2")
    else:
      src1 = sgpr(sgprInc)
      if (odd or frequency == 1) and V2Iterations > 0:
        maskV1 = (self.StoreCInUnrollNumHorizontalTiles//frequency) - 1 # StoreCEnableCount decremented at every frequency iterations
        if maskV1 == 0:
          # odd (or frequency == 1) + maskV1 == 0 case, directly use IncV1 as src1 of s_cselect instruction (to remove s_mov for IncV1)
          src1 = sgpr(sgprIncV1)
      maskV2 = (V2Iterations//frequency) - 1 # StoreCEnableCount decremented at every frequency iterations
      v1OrV2SelectCode += inst("s_and_b32", sgpr(tmpSgpr1), sgpr(sgprCounter), hex(maskV2), "")
      v1OrV2SelectCode += inst("s_cselect_b32", sgpr(sgprInc), src1, sgpr(sgprIncV2),  "Use AddrIncV2 if %s & %u == 0"%(sgprCounter, maskV2))
    if V3Iterations > 0:
      maskV3 = (V3Iterations//frequency) - 1 # StoreCEnableCount decremented at every frequency iterations
      v1OrV2SelectCode += inst("s_and_b32", sgpr(tmpSgpr1), sgpr(sgprCounter), hex(maskV3), "")
      v1OrV2SelectCode += inst("s_cselect_b32", sgpr(sgprInc), sgpr(sgprInc), sgpr(sgprIncV3),  "Use AddrIncV3 if %s & %u == 0"%(sgprCounter, maskV3))

    if hOffset == 0:
      if V2Iterations == 0:
        # no code here (v1 offset only. no other choice)
        # offset already in sgprInc = CorD + "AddrInc" at initialization stage
        kStr = kStr
      else:
        if not (odd or frequency == 1):
          # even case, always v1
          kStr += inst("s_mov_b32", sgpr(sgprInc), sgpr(sgprIncV1),  "Use AddrIncV1")
        else:
          # odd case
          kStr += v1OrV2SelectCode
    else:
      AddrInc1 = hOffset
      if odd or frequency == 1:
        maskV1 = (self.StoreCInUnrollNumHorizontalTiles//frequency) - 1 # StoreCEnableCount decremented at every frequency iterations
        if maskV1 > 0:
          # mask>0 case, select values1
          kStr += inst("s_and_b32", sgpr(tmpSgpr1), sgpr(sgprCounter), hex(maskV1), "")
          kStr += inst("s_cselect_b32", sgpr(sgprInc), hex(AddrInc1), sgpr(sgprIncV1),  "Use AddrIncV1 if %s & %u == 0 else %u"%(sgprCounter, maskV1, AddrInc1))
        elif V2Iterations == 0:
          # mask==0 case, use IncV1
          kStr += inst("s_mov_b32", sgpr(sgprInc), sgpr(sgprIncV1),  "Use AddrIncV1")
        if V2Iterations > 0:
          kStr += v1OrV2SelectCode
      #else:
        # even case, iteration will not be multiple of 4 or 16, just use AddrInc1
        # no need to copy AddrInc1 to sgpr(sgprInc). Hex value AddrInc1 will be directly used at the next instruction (StoreCAvail==0 check)
        #kStr += inst("s_mov_b32", sgpr(sgprInc), hex(AddrInc1), "Use AddrInc1 = %u"%(AddrInc1))
    # TODO: support other MIWaveGroup
    # else:

    return kStr

  ##############################################################################
  # get address/gpr index increment frequency for StoreCInUnroll
  ##############################################################################
  def getAddrGprIdxIncrementFrequencyForStoreCInUnroll(self, kernel):
    frequency = 2
    # use 1 if one of the following cases is True
    #  ExpandPointerSwap is False
    #  (ASEM/GSU)%(DepthU*2) != 0 (odd exit case)
    asem = kernel["AssertSummationElementMultiple"]
    gsu = kernel["GlobalSplitU"]
    if kernel["ExpandPointerSwap"] == False or \
       ((asem % gsu != 0) or ((asem//gsu) % (kernel["DepthU"] * 2) != 0)):
      frequency = 1
    return frequency

  ##############################################################################
  # generate post process for StoreCInUnroll loop
  # 1) increment gpr indexing (new value in tmp). Put this as separate item in StoreCUnrollCode
  # 2-1) increment StoreC address  (new value in tmp)
  # 2-2) check enable count and apply new values when necessary
  ##############################################################################
  def generatePostProcessForStoreCInUnrollLoop(self, kernel, needPost):
    needAddrC = (not kernel["AssertCEqualsD"]) and kernel["ProblemType"]["UseBeta"]

    postProcessList = []
    finalAddrIncList = []

    # tmp sgpr allocation
    # StaggerU case, 2 tmp sgpr regs can be used and it can be same as the following allocation.
    # We allocate 2 more sgpr regs if StaggerU is used to avoid StoreCInUnroll code overwritting tmp regs for StaggerU
    tmpSregOffset = 2 if kernel["StaggerU"] else 0
    tmpSgprGprIdxInc = self.getTmpSgpr(4 + tmpSregOffset).idx()  + tmpSregOffset# 1 reg
    tmpSgprDinc = tmpSgprGprIdxInc + 1 # 1 reg
    tmpSgprCinc = tmpSgprGprIdxInc + 2 # 1 reg
    tmpSgprWork = tmpSgprGprIdxInc + 3 # 1 reg

    # add increment value generation code for gpr indexing
    # for the last iteration or not odd iteration, no increment code for gpr indexing
    if needPost:
      postProcessList.append(self.generateSgprGprIdxIncrementForStoreCInUnroll(kernel, tmpSgprGprIdxInc, tmpSgprWork))

    # Addr D increment value generation code
    postProcessList.append(self.generateCorDaddrIncrementForStoreCInUnroll(kernel, "D", needPost, tmpSgprWork))

    # generate final increment code
    if needPost:
      # odd lc case (Unroll Loop 2/2)
      # timing to disable increment is 1 iteration ahead of setting StoreCAvail=0 (StoreCEnableCount<0)
      # here, we check if sgprStoreCEnableCount<=0

      if not kernel["StoreCInUnrollExact"]:
        conditionComment = "if StoreCEnableCount<=0"
        # select increment value or 0 depending on sgprStoreCEnableCount value
        postProcessList.append(inst("s_cmp_le_i32", sgpr("StoreCEnableCount"), hex(0), "set scc if StoreCEnableCount<=0"))
        # generate final gpr index increment value
        postProcessList.append(inst("s_cselect_b32", sgpr(tmpSgprGprIdxInc), hex(0), sgpr(tmpSgprGprIdxInc),  \
                                    "set gpr index increment value to 0 %s"%conditionComment))
        # generate final SrdD increment value
        postProcessList.append(inst("s_cselect_b32", sgpr(tmpSgprDinc), hex(0), sgpr("DAddrInc"),  \
                                    "set SrdD increment value to 0 when StoreCAvail==0"))

        if needAddrC:
          if not kernel["StoreCInUnrollExact"]:
            # generate final SrdC increment value
            postProcessList.append(inst("s_cselect_b32", sgpr(tmpSgprCinc), hex(0), sgpr("CAddrInc"),
                                        "set SrdC increment value to 0 when StoreCAvail==0"))

    else:
      # even lc case (Unroll Loop 1/2)

      if not kernel["StoreCInUnrollExact"]:
        # generate final SrdD increment value
        DAddrInc = sgpr("DAddrInc")
        if self.StoreCInUnrollAddrIncHoffset > 0:
          DAddrInc = hex(self.StoreCInUnrollAddrIncHoffset)
        postProcessList.append(inst("s_and_b32", sgpr(tmpSgprDinc), sgpr("StoreCAvail"), DAddrInc,  \
                                    "set SrdD increment value to 0 when StoreCAvail==0"))
        if needAddrC:
          if not kernel["StoreCInUnrollExact"]:
            # generate final SrdC increment value
            CAddrInc = sgpr("CAddrInc")
            if self.StoreCInUnrollAddrIncHoffset > 0:
              CAddrInc = hex(self.StoreCInUnrollAddrIncHoffset)
            postProcessList.append(inst("s_and_b32", sgpr(tmpSgprCinc), sgpr("StoreCAvail"), CAddrInc,
                                        "set SrdC increment value to 0 when StoreCAvail==0"))

    # increment gpr index
    if needPost:
      postProcessList.append(inst("s_add_u32", sgpr("StoreCIndex0"), sgpr("StoreCIndex0"), sgpr(tmpSgprGprIdxInc),  ""))

    # increment SrdD
    sgprSrd = "SrdD"
    sgprSrd1 = sgprSrd + "+1"
    if not kernel["StoreCInUnrollExact"]:
      src2 = sgpr(tmpSgprDinc)
    else:
      src2 = sgpr("DAddrInc")
      if not needPost and self.StoreCInUnrollAddrIncHoffset > 0:
        src2 = hex(self.StoreCInUnrollAddrIncHoffset)
    finalAddrIncList.append(inst("s_add_u32", sgpr(sgprSrd), sgpr(sgprSrd), src2,  ""))
    finalAddrIncList.append(inst("s_addc_u32", sgpr(sgprSrd1), sgpr(sgprSrd1), 0,  ""))
    # increment SrdC
    if needAddrC:
      sgprSrd = "SrdC"
      sgprSrd1 = sgprSrd + "+1"
      if not kernel["StoreCInUnrollExact"]:
        src2 = sgpr(tmpSgprCinc)
      else:
        src2 = sgpr("CAddrInc")
        if not needPost and self.StoreCInUnrollAddrIncHoffset > 0:
          src2 = hex(self.StoreCInUnrollAddrIncHoffset)
      finalAddrIncList.append(inst("s_add_u32", sgpr(sgprSrd), sgpr(sgprSrd), src2,  ""))
      finalAddrIncList.append(inst("s_addc_u32", sgpr(sgprSrd1), sgpr(sgprSrd1), 0,  ""))

    # combine multiple instructions in one item
    numCombine = 2
    tmpList = []
    for index in range(0,len(postProcessList),numCombine):
      kStr = ""
      for n in range(numCombine):
        if index + n <= len(postProcessList) - 1:
          # combine this instruction if index + n is valid range
          kStr += postProcessList[index + n]
      if kStr != "":
        tmpList.append(kStr)
    # overwrite original list with combined one
    postProcessList = tmpList

    return postProcessList, finalAddrIncList

  ##############################################################################
  # Reset StoreC sync object
  ##############################################################################
  def resetStoreCsyncObject(self, kernel):
    kStr = ""
    kStr += inst("s_mov_b64", sgpr("StoreCAvail",2), hex(0), "Reset StoreC syncObject")
    return kStr

  ##############################################################################
  # Set StoreC sync object
  ##############################################################################
  def setStoreCsyncObject(self, kernel):
     kStr = ""
     kStr += inst("s_mov_b64", sgpr("StoreCAvail",2), -1, "set StoreC syncObject")
     return kStr

  ##############################################################################
  # end process for StoreCInUnroll per PersistentLoop (OptNLL)
  ##############################################################################
  def endProcessPersistentLoopforStoreCInUnrollOptNLL(self, kernel):
    kStr = ""
    kStr += self.comment1("end process for StoreCInUnroll per PersistentLoop (OptNLL)")
    # restore srcC/D backup for next PersistentLoop (restore current address from backup)
    kStr += self.restoreSrdCandDBackup(kernel)
    # set PreLoopLWVmcntCase = 5 for StoreCInUnroll (no store executed at the end of PK loop)
    kStr += inst("s_movk_i32", sgpr("PreLoopLWVmcntCase"), hex(5), "Use case 5 for next PK Loop (only the last storeCInUnroll executed at the end of PK Loop)")

    # if K is not multiple of DepthU*loopCopies or K<512, skip long jump to the top of persistent loop
    endLabel = self.getNamedLabelUnique("PersistentKernel_End_For_StoreCInUnroll")
    # no StoreCInUnroll path for not supported K
    tmpSgpr = self.getTmpSgpr(1).idx()
    minK, supportMinKmask = self.getSupportedKvalueLog2ForStoreCInUnroll(kernel)
    supportedK = supportMinKmask + 1
    minK = minK//kernel["DepthU"] # divided by DepthU to use OrigLoopCounter (= K / DepthU)
    # skip multiple of DepthU * frequency check if AssertSummationElementMultiple is multiple of DepthU * frequency
    frequency = self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel)
    asem = kernel["AssertSummationElementMultiple"]
    gsu = kernel["GlobalSplitU"]
    if (asem % gsu != 0) or ((asem//gsu) % (kernel["DepthU"] * frequency) != 0):
      kStr += inst("s_and_b32", sgpr(tmpSgpr), sgpr("SizesSum"), hex(supportMinKmask), "if K is not multiple of DepthU * %u (%u)"%(frequency, supportedK) )
      kStr += inst("s_cbranch_scc1", "label_%s"%endLabel, "Skip long jump to the top of persistent loop if K is not multiple of %u"%(supportedK) )
    # if PostLoop is enabled, minK check is unnecessary.
    if not kernel["StoreCInUnrollPostLoop"]:
      if not kernel["StoreCInUnrollExact"]:
        kStr += inst("s_cmp_lt_u32", sgpr("OrigLoopCounter"), hex(minK), "if OrigLoopCounter(=K/DepthU) < minK / DepthU (= %u)"%(minK) )
        kStr += inst("s_cbranch_scc1", "label_%s"%endLabel, "Skip long jump to the top of persistent loop")
      else:
        # exact mode case, only == minK supported
        kStr += inst("s_cmp_eq_u32", sgpr("OrigLoopCounter"), hex(minK), "if OrigLoopCounter(=K/DepthU) != minK / DepthU (= %u)"%(minK) )
        kStr += inst("s_cbranch_scc0", "label_%s"%endLabel, "Skip long jump to the top of persistent loop")

    # set StoreCAvail for the next persistent loop
    kStr += self.setStoreCsyncObject(kernel)
    #  branch code to the top of Persistent loop here
    kStr += self.persistentLoopendLongjump(kernel)

    # add label
    kStr += "label_%s:%s"%(endLabel, self.endLine)

    # add new line at the end
    kStr += self.endLine

    return kStr

  ##############################################################################
  # end process for StoreCInUnroll per PersistentLoop (NoOptNLL)
  ##############################################################################
  def endProcessPersistentLoopforStoreCInUnrollNoOptNLL(self, kernel):
    kStr = ""
    kStr += self.comment("end process for StoreCInUnroll per PersistentLoop (NoOptNLL)")
    # restore srcC/D backup for next PersistentLoop (restore current address from backup)
    kStr += self.restoreSrdCandDBackup(kernel)

    # add new line at the end
    kStr += self.endLine

    return kStr

  ##############################################################################
  # number of storeC code in template for StoreCInUnroll
  ##############################################################################
  def getNumberOfStoreCInTemplate(self, kernel):
    numGlobalStoreCinTemplate = 0
    if kernel["StoreCInUnroll"]:
      # count number of StoreC in template
      tmpStr = ' '.join([str(x) for x in self.StoreCTemplate.items()])
      numGlobalStoreCinTemplate  = tmpStr.count("_buffer_store")  # count buffer_store
      numGlobalStoreCinTemplate += tmpStr.count("buffer_atomic_add")   # count buffer_atomic_add
    return numGlobalStoreCinTemplate

  ##############################################################################
  # number of LoadC code in template for StoreCInUnroll
  ##############################################################################
  def getNumberOfLoadCInForLoadC(self, kernel):
    numGlobalReadC = 0
    if kernel["StoreCInUnroll"] and not kernel["AtomicAddC"]:
      tmpStr = ' '.join([str(x) for x in self.LoadCTemplate.items()])
      numGlobalReadC = tmpStr.count("_buffer_load")
    return numGlobalReadC

  ##############################################################################
  # generate storeCInUnroll post loop code
  ##############################################################################
  def generateStoreInUnrollPostLoop(self, kernel, isOptNLL, isDTVodd):
    kStr = ""
    if kernel["StoreCInUnroll"] and kernel["StoreCInUnrollPostLoop"]:
      OptName = "Opt" if isOptNLL else "Ord"
      if isDTVodd:
        # add "Odd" to make odd case label name different from even label name
        OptName += "Odd"
      StartLabelName = "StoreCInUnrollPostLoopStart" + OptName
      EndLabelName = "StoreCInUnrollPostLoopEnd" + OptName
      kStr += self.comment1("StoreCInUnroll PostLoop (%s NLL)"%OptName)

      # StorePriorityOpt
      if kernel["StorePriorityOpt"]:
        kStr += inst("s_setprio 0","store optimization")

      # generate start label
      kStr += self.getNamedLabelDef(StartLabelName)

      # check if StoreCAvail == 0
      kStr += inst("s_cmp_eq_i32", sgpr("StoreCAvail"), hex(0), "if StoreCAvail==0")
      # jump to end
      kStr += inst("s_cbranch_scc1 %s"%(self.getNamedLabel(EndLabelName)), "jump to StoreCInUnrollPostLoopEnd")
      # check if StoreCEnableCount <= 0
      kStr += inst("s_cmp_le_i32", sgpr("StoreCEnableCount"), hex(0), "if StoreCEnableCount<=0")
      # jump to end
      kStr += inst("s_cbranch_scc1 %s"%(self.getNamedLabel(EndLabelName)), "jump to StoreCInUnrollPostLoopEnd")

      # generate PostLoop code (non OptNLL (means Ord NLL) case are merged into OptNLL)
      kStrPL = ""

      # decrement StoreCEnableCount
      kStrPL += inst("s_sub_u32",  sgpr("StoreCEnableCount"), sgpr("StoreCEnableCount"), hex(1), \
                     "decrement StoreCEnableCount.")

      backupSgpr = self.getTmpSgpr(2).idx()  # allocate all tmp register here
      tmpSgprWork = backupSgpr + 1
      loopCount = self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel)
      for lc in range(loopCount):
        # generate LoadC code
        needAddrC = (not kernel["AssertCEqualsD"]) and kernel["ProblemType"]["UseBeta"]
        for x in self.LoadCTemplate.items():
          kStrPL += str(x)
        # Addr C increment code
        if needAddrC:
          kStrPL += self.generateCorDaddrIncrementForStoreCInUnroll(kernel, "C", (lc % 2) == 0, tmpSgprWork)

        # these 3 items need to be in the same set
        #  open gpr indexing
        #  accVgpr (need gpr indexing)
        #  close gpr indexing
        kStrPL += self.openmovaccVgpr(kernel, backupSgpr)
        # odd case, use + (1 iteration) for gpr index, but not necessary if index frequency is 1
        odd = (lc % 2) != 0 and (self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel) > 1)
        kStrPL += self.getAccVgprCode(kernel, odd)
        first, second = self.closemovaccVgpr(kernel, backupSgpr)
        kStrPL += first
        kStrPL += second
        # Alpha
        for x in self.AlphaOpTemplate.items():
          kStrPL += str(x)
        # Beta
        for x in self.BetaOpTemplate.items():
          kStrPL += str(x)

        # StoreC

        # generate post process for StoreCInUnroll loop
        # 1) increment gpr indexing (new value in tmp). Put this as separate item in StoreCUnrollCode
        # 2-1) increment StoreC address  (new value in tmp)
        # 2-2) check enable count and apply new values when necessary
        needPost = (lc % 2) != 0 or (self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel) == 1)
        postProcessList, finalAddrIncList = self.generatePostProcessForStoreCInUnrollLoop(kernel, needPost)

        for x in self.StoreCTemplate.items():
          kStrPL += str(x)
        # StorSyncOpt
        if kernel["StoreSyncOpt"]:
          kStrPL += "s_sleep %d // optimization: sync and wait\n" %(kernel["StoreSyncOpt"]-1)
          kStrPL += "s_barrier\n"
        # add all finalAddrInc code after the last StoreC (in the same item)
        for item in (postProcessList + finalAddrIncList):
          kStrPL += item

      # add PostLoop code
      kStr += kStrPL

      # jump to start
      kStr += inst("s_branch %s"%(self.getNamedLabel(StartLabelName)), "restart StoreCInUnrollPostLoop")

    # generate end label
    kStr += self.getNamedLabelDef(EndLabelName)

    # replace __placeholder__ for wait loadC (to 0)
    kStr = kStr.replace("__placeholder__", "0")

    return kStr

  ##############################################################################
  # openOddNoLoadLoopForDTV
  # generate open code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def openOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    kStr = ""
    evenStartLabelName = "EvenStart" + name
    # odd exit check code
    # use OrigLoopCounter & 1
    tmpSgpr = self.getTmpSgpr(1).idx()
    #scc0or1 = 0 if isNGLL else 1
    #oddOrEven = "Even" if isNGLL else "Odd"
    kStr += inst("s_and_b32",sgpr(tmpSgpr), sgpr("OrigLoopCounter"), 1, "test if OrigLoopCounter is Odd ?")
    kStr += inst("s_cbranch_scc0", self.getLabelTarget(evenStartLabelName), "Skip odd code if OrigLoopCounter is Even")

    return kStr

  ##############################################################################
  # closeOddNoLoadLoopForDTV
  # generate close code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def closeOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    kStr = ""
    evenStartLabelName = "EvenStart" + name
    evenEndLabelName = "EvenEnd" + name
    # odd exit code
    kStr += inst("s_branch", self.getLabelTarget(evenEndLabelName), "Skip even code")
    # generate even start label
    kStr += self.getLabelDef(evenStartLabelName)
    return kStr

  ##############################################################################
  # generateEvenEndLabeNoLoadLoopForDTV
  # generate even end label for DirectToVgpr
  ##############################################################################
  def generateEvenEndLabeNoLoadLoopForDTV(self, kernel, isNGLL, name):
    kStr = ""
    evenEndLabelName = "EvenEnd" + name
    # generate even end label
    kStr += self.getLabelDef(evenEndLabelName)
    return kStr

  ##############################################################################
  # generateOddEndVgprCopyForDTV
  # generate odd end vgpr copy for DirectToVgpr
  ##############################################################################
  def generateOddEndVgprCopyForDTV(self, kernel):
    kStr = ""
    vregNameBase = "G2LA" if kernel["DirectToVgprA"] else "G2LB"
    numVreg = self.numVgprG2LA//2 if kernel["DirectToVgprA"] else self.numVgprG2LB//2
    vregSet0 = vregNameBase + "0+"
    vregSet1 = vregNameBase + "1+"
    self.comment("copy Vreg set1 to Vreg set0 for DirectToVgpr + PrefetchAcrossPersistent")
    for index in range(numVreg):
      kStr += inst("v_mov_b32", vgpr(vregSet0+str(index)), vgpr(vregSet1+str(index)), "")
    return kStr

  ##############################################################################
  # isSwapGlobalReadOrderForDtvOrDtl
  ##############################################################################
  def isSwapGlobalReadOrderForDtvOrDtl(self, kernel):
    # swap global read order (from A, B to B, A) if the following condition is true
    #  - DirectToVgprA and B are true and number of global read B >= number of global read A
    #  - if DirectToVgprA is true and DirectToVgprB is false
    #  - if DirectToLdsA=False and DirectToLdsB=True (need to put DTLB first)
    if kernel["DirectToVgprA"] and kernel["DirectToVgprB"]:
      if kernel["NumLoadsB"] >= kernel["NumLoadsA"]:
        return True
    elif kernel["DirectToVgprA"]:
      return True
    elif (not kernel["DirectToLdsA"]) and kernel["DirectToLdsB"]:
      return True
    return False

  ##############################################################################
  # WaitCnt- DONE
  # 3 components can contribute to the waitcnt:
  #   - Pending global reads.  (skipGlobalRead)
  #   - Pending local write.  (skipLocalWrite)
  #   - Pending local reads (skipLocalRead)
  # If a skip* arg is -1, the associated component does not contribute to
  # the expected lgkmcnt or vmcnt
  ##############################################################################
  def wait(self, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, \
      skipLocalRead, comment):
    if not self.do["Wait"]: return ""
    # skip = -1 -> ignore
    # skip =  n -> waitcnt(n*num)

    lgkmcnt = 0 if skipLocalWrite > -1 or skipLocalRead > -1 else -1

    if skipLocalWrite > -1 or skipLocalRead > -1:
      if skipLocalWrite > -1:
        numA = 0 if (kernel["DirectToLdsA"] or  kernel["DirectToVgprA"])\
               else tPA["nrp"]*tPA["nrc"]*max(tPA["nwcv"],tPA["nwpv"])//tPA["nwcvpi"]
        numB = 0 if (kernel["DirectToLdsB"] or  kernel["DirectToVgprB"])\
               else tPB["nrp"]*tPB["nrc"]*max(tPB["nwcv"],tPB["nwpv"])//tPB["nwcvpi"]
        lgkmcnt += skipLocalWrite * (numA + numB)
      if skipLocalRead > -1:
        numReadsPerIterA = 0 if kernel["DirectToVgprA"] else self.numReadsPerIterA
        numReadsPerIterB = 0 if kernel["DirectToVgprB"] else self.numReadsPerIterB
        readsPerIter = numReadsPerIterA + numReadsPerIterB
        lgkmcnt += skipLocalRead * readsPerIter

    vmcnt = 0 if skipGlobalRead > -1 else -1
    if skipGlobalRead > -1:
      numA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] \
          * self.numReadVectorComponentsA
      numB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] \
          * self.numReadVectorComponentsB
      vmcnt += skipGlobalRead * (numA + numB)

    if (self.db["ConservativeWaitCnt"] & 0x2) and skipGlobalRead != -1 or \
       (self.db["ConservativeWaitCnt"] & 0x4) and skipLocalWrite != -1 or \
       (self.db["ConservativeWaitCnt"] & 0x8) and skipLocalRead  != -1:
        imod = Code.Module("ConservativeWaitCnt")
        imod.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "debug %s"%comment )
        if self.archCaps["SeparateVscnt"]:
          imod.addInst("s_waitcnt_vscnt", "null", "0", "writes")
        imod.addInst("s_barrier", "debug" )
        return imod

    maxLgkmcnt = globalParameters["AsmCaps"][self.version]["MaxLgkmcnt"]
    lgkmcnt = min(lgkmcnt, maxLgkmcnt)
    if lgkmcnt >= 0 and vmcnt >= 0:
      vmcnt = -1 # preserve prior behavior of removing vmcnt here?
    maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
    vmcnt = min(vmcnt, maxVmcnt)

    waitcnt = Code.WaitCnt(self.version, lgkmcnt,vmcnt,comment)
    if 0 and lgkmcnt == 0:
      imod = Code.Module("DebugWait")
      imod.addCode(waitcnt)
      imod.addText(self.bomb())
      return imod
    return waitcnt

  ##############################################################################
  # WaitCnt code
  ##############################################################################
  def waitcntCode(self, kernel,lgkmcnt, vmcnt, comment):
    if not self.do["Wait"]: return ""
    return Code.WaitCnt(self.version, lgkmcnt,vmcnt,comment)

  ##############################################################################
  # return number of store instructions
  ##############################################################################
  def getNumStoreInst(self, str):
    ret = 0
    ret += str.count("_buffer_store")  # count _buffer_store
    ret += str.count("_global_store")  # count _global_store
    ret += str.count("buffer_atomic_add")   # count buffer_atomic_add
    ret += str.count("global_atomic_add")   # count global_atomic_add
    return ret

  ##############################################################################
  # return number of load instructions
  ##############################################################################
  def getNumLoadInst(self, str):
    ret = 0
    ret += str.count("_buffer_load")  # count _buffer_load
    ret += str.count("_global_load")  # count _global_load
    return ret

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  def getWaitcntCodeForDirectToVgpr(self, kernel, localWriteEndIter, u, firstIter, isPap=True, beforeBarrier=False, NLLlast=False, oddLast=False):
    retStr = ""
    # generate wait
    if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
      if self.enable["Wait"]:
        pgr2 = kernel["PrefetchGlobalRead"] == 2
        numGlobalReadA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] * self.numReadVectorComponentsA
        numGlobalReadB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] * self.numReadVectorComponentsB
        numGlobalRead = numGlobalReadA if self.isSwapGlobalReadOrderForDtvOrDtl(kernel) else numGlobalReadB
        numGlobalReadAll = numGlobalReadA + numGlobalReadB
        numGlobalStoreC = 0
        numReadsIterCoalesced = self.numReadsIterCoalescedA if self.isSwapGlobalReadOrderForDtvOrDtl(kernel) else self.numReadsIterCoalescedB
        waitComment = "global read wait for DirectToVgpr"
        # delay DirectToVgpr global read (from previous iteration) which is not referred yet (do not delay in beforeBarrier case)
        numRegsIn1set = (numGlobalRead * numReadsIterCoalesced) // kernel["LoopIters"]
        numSet = (u + numReadsIterCoalesced) // numReadsIterCoalesced
        numSetMod = (u + numReadsIterCoalesced) % numReadsIterCoalesced
        if (not beforeBarrier) and numSetMod > 0:
          # if mod > 0, wait is already done by mod == 0 case and no need to wait for same set of global read
          return ""
        needToWait = numGlobalRead - numSet * numRegsIn1set
        if not isPap:
          # not isPap case, no global load A, B in no load loop. Reset numGlobalReadAll and numGlobalRead
          numGlobalReadAll = 0
          numGlobalRead = 0
        if pgr2:
          # PGR=2 case, add numGlobalReadAll for second set of prefetch
          needToWait += numGlobalReadAll
        if u > 0:
          # count number of global read for i < u
          count = 0
          for i in range(u):
            globalReadStr = ' '.join([str(x) for x in self.perIterGlobalReadCode[i].flatitems()])
            count += self.getNumLoadInst(globalReadStr)
            # PGR=2 case, global read is in LocalWriteCode
            localWriteStr = ' '.join([str(x) for x in self.perIterLocalWriteCode[i].flatitems()])
            count += self.getNumLoadInst(localWriteStr)
          needToWait += count
          if u == localWriteEndIter + 1 and beforeBarrier:
            # beforeBarrier case, reduce the amount of non-Vgpr global read
            needToWait -= (numGlobalReadAll - numGlobalRead)
        # adjustment for oddLast
        # oddLast case or ScheduleIterAlg < 3 case, ignore all of above and set 0
        if oddLast or kernel["ScheduleIterAlg"] < 3:
          needToWait = 0
        if kernel["StoreCInUnroll"]:
          # In StoreCInUnroll case,
          # 1) last iteration case (u == localWriteEndIter + 1)
          #  1-1) if StoreC is already executed in the previous u, add number of executed buffer_store/atomic_add
          #      (global read C wait is already done in this case)
          #  1-2) else, add number of global read C to numGlobalReadAll

          # count number of StoreC in template
          tmpStr = ' '.join([str(x) for x in self.StoreCUnrollCode.flatitems()])
          numGlobalStoreCinTemplate  = self.getNumStoreInst(tmpStr) # count store instructions
          numGlobalStoreC = 0

          if u == localWriteEndIter + 1:
            if beforeBarrier:
              # before barrier case (DirectToLds+DirectToVgpr), put waitcnt vmcnt just before barrier (before ds_read)
              # In that case, StoreC is already done. Add number of store C from template to vmcnt.
              numGlobalStoreC += numGlobalStoreCinTemplate
              # It means LoadC wait is already done. Deduct the number of load C in template
              # count number of Load in template
              tmpStr = ' '.join([str(x) for x in self.LoadCUnrollCode.flatitems()])
              numGlobalLoadCinTemplate  = self.getNumLoadInst(tmpStr)  # count load instructions
              needToWait -= numGlobalLoadCinTemplate
            else:
              # check if store C is already in perIterLocalWriteCode
              for i in range(u):
                # scheduled storeC in unroll is in LocalWriteCode
                localWriteStr = ' '.join([str(x) for x in self.perIterLocalWriteCode[i].flatitems()])
                numGlobalStoreC += self.getNumStoreInst(localWriteStr)
              # no LDS write (DirectToLds+DirectToVgpr) and not beforeBarrier and not firstIter case,
              # no need to wait for StoreC in previous iteration
              # Then, add the number of storeC in template
              #if kernel["NoLdsWriteCode"] and not firstIter:
              #  numGlobalStoreC += numGlobalStoreCinTemplate
          # 2) add number of store C from previous iter to needToWait
          #   2-1) not firstIter and u < localWriteEndIter + 1 case
          #   2-2) noLoadC and last NoLoadLoop
          needLoadC = (not kernel["AtomicAddC"]) and kernel["ProblemType"]["UseBeta"]
          if not firstIter and (u < localWriteEndIter + 1 or ((not needLoadC) and NLLlast)):
            numGlobalStoreC += numGlobalStoreCinTemplate

          # oddLast case, ignore all of above and set numGlobalStoreCinTemplate
          if oddLast:
            numGlobalStoreC = numGlobalStoreCinTemplate

          # add numGlobalStoreC to needToWait
          needToWait += numGlobalStoreC
          waitComment = "global read/store wait for DirectToVgpr with StoreCInUnroll (StoreC=%u)"%(numGlobalStoreC)

        # generate waitcnt code
        retStr = self.getWaitcntCodeForDTV(kernel, needToWait, waitComment)
    return retStr

  ##############################################################################
  # waitcnt code for PrefetchGlobalRead
  ##############################################################################
  def getWaitcntCodeForPGR(self, kernel):
    # generate wait
    if (kernel["DirectToVgprA"] and kernel["DirectToVgprB"]):
      # DTVA and DTVB case, wait code for prefetch is unnecessary
      return ""
    count = 0
    if kernel["DirectToVgprA"]:
      # increase vmcnt for DTVA (no need to wait for DTVA global load)
      numGlobalReadA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] * self.numReadVectorComponentsA
      count += numGlobalReadA
    if kernel["DirectToVgprB"]:
      # increase vmcnt for DTVB (no need to wait for DTVB global load)
      numGlobalReadB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] * self.numReadVectorComponentsB
      count += numGlobalReadB

    return self.getWaitcntCodeForDTV(kernel, count, " 8wait for global read")

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  def getWaitcntCodeForDTV(self, kernel, count, waitComment):
    # vmcnt should not go over MaxVmcnt
    maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
    count = min(count, maxVmcnt)
    return self.waitcntCode(kernel, -1, count, waitComment)

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel, comment=""):

    if kernel["NumThreads"] > self.kernel["WavefrontSize"] and self.do["Sync"]:
      kStr = ""
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_lgkmcnt", "null", "0", "extra navi wait")
      elif kernel.enabledSplitLDS or kernel["ScheduleIterAlg"] == 2 \
        or kernel["PrefetchGlobalRead"] == 2 or self.prefetchAcrossPersistent:
        kStr += "// Skip force waitcnt0" + self.endLine
      elif self.archCaps["Waitcnt0Disabled"]:
        kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "force waitcnt0" )

      kStr += self.indent + self.syncStr + " //" + comment + self.endLine
      return kStr
    else:
      return "// Skip barrier: NumThreads=%s"%(kernel["NumThreads"]) + \
              comment + self.endLine

  ########################################
  # dump lds state
  ########################################
  def dumpLds(self, kernel, startU, numU):
    kStr = ""
    if globalParameters["DebugKernel"]:
      kStr += self.comment("dump lds state")
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      kStr += inst("s_barrier", "dump LDS" )
      tmp = self.vgprPool.checkOut(1)
      tmpAddr = self.vgprPool.checkOut(1)
      kStr += inst("v_lshlrev_b32", \
          vgpr(tmpAddr), \
          hex(log2(self.bpeAB)), \
          vgpr("Serial"), \
          "dump lds")
      for i in range(startU, startU+numU):
        kStr += inst("_ds_load_b32", vgpr(tmp), \
            vgpr(tmpAddr) + " offset:%u"%(i*kernel["NumThreads"]*4), "dump lds")
        kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "dump" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
        kStr += self.dump(vgpr(tmp))
      self.vgprPool.checkIn(tmp)
      self.vgprPool.checkIn(tmpAddr)
    return kStr

  ########################################
  # init lds state
  ########################################
  def initLds(self, kernel, value):
    kStr = ""
    kStr += self.comment("init lds state")
    kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
    kStr += inst("s_barrier", "init LDS" )
    tmp = self.vgprPool.checkOut(1)
    tmpAddr = self.vgprPool.checkOut(1)
    kStr += inst("v_mov_b32", vgpr(tmp), hex(value), "Init value")
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    writesPerThread = ((kernel["LdsNumElements"]*numBytesPerElement-1)//kernel["NumThreads"]//4) + 1
    kStr += inst("v_lshlrev_b32", \
        vgpr(tmpAddr), \
        2,
        vgpr("Serial"), \
        "set per-thread address to init LDS")
    for i in range(0, writesPerThread):
      kStr += "_ds_store_b32 %s, %s offset:%u %s" \
          %( vgpr(tmpAddr), vgpr(tmp), (i*kernel["NumThreads"]*4), \
          "//init lds" + self.endLine)

    kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "wait for LDS init to complete" )
    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
    kStr += inst("s_barrier", "init LDS exit" )
    self.vgprPool.checkIn(tmp)
    self.vgprPool.checkIn(tmpAddr)
    return kStr

  def AccVgprImagNumOffset(self, kernel):
    acc2arch, _ = self.AccToArchMapper(kernel)
    return len(acc2arch) * kernel["MIRegPerOut"]

  ##############################################################################
  # AccToArchMapper
  # Provides forward (acc2arch) and backward (arch2acc) index transformation
  #  - Forward transformation is currently used for acc->vgpr copying
  #  - Backward transformation is used in ShiftVectorComponent() to map logical
  #    C-tile index back to original acc index
  ##############################################################################
  def AccToArchMapper(self, kernel):
    acc2arch = dict()
    arch2acc = dict()

    matrixInstM  = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
    matrixInstN  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
    matrixInstBM = 1                                                if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
    matrixInstBN = 1                                                if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

    OutputsPerMFMA1B = matrixInstM * matrixInstN // self.kernel["WavefrontSize"]
    VectorWidth0     = kernel["VectorWidthA"]
    outerTT0         = kernel["MIWaveTile"][0] // VectorWidth0
    VectorWidth1     = kernel["VectorWidthB"]
    outerTT1         = kernel["MIWaveTile"][1] // VectorWidth1

    for wgIdx1 in range(0, outerTT1):
      for wgIdx0 in range(0, outerTT0):
        for bIdx1 in range(0, matrixInstBN):
          for bIdx0 in range(0, matrixInstBM):
            for tIdx in range(0, OutputsPerMFMA1B):
              for vw1 in range(0, VectorWidth1):
                for vw0 in range(0, VectorWidth0):
                  src, dst = 0, 0
                  if kernel["SourceSwap"]:
                    src = tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (bIdx1 + matrixInstBN * (vw0 + VectorWidth0 * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * wgIdx1)))))
                    dst = vw0 + VectorWidth0 * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * (tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * wgIdx1)))))
                  else:
                    src = tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * (bIdx0 + matrixInstBM * (vw0 + VectorWidth0 * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * wgIdx1)))))
                    dst = vw0 + VectorWidth0 * (tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * (bIdx1 + matrixInstBN * wgIdx1)))))
                  acc2arch[src] = dst
                  arch2acc[dst] = src

    return acc2arch, arch2acc

  ##############################################################################
  # MapAcctoArch
  # function to map MFMA Acc  Registers to Arch VGPR register
  # option :
  #         0 - one-to-one mapping of ACC -> VGPR  using VW
  #         1 - using ds swizzle map strided lanes output of MFMA to  coalescing
  #             lanes of v_mac
  ##############################################################################
  def MapAcctoArchRegs(self, kernel, option, isOptNLL=False):
    kStr = ""
    kStr += self.comment("Mapping of Acc register -> C Vgpr register")

    acc2arch, _ = self.AccToArchMapper(kernel)

    complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1
    streamK = (kernel["StreamK"] > 0 and kernel["StreamKAtomic"] == 0)
    self.codeAccVgprRead = Code.Module("AccVgprRead")
    self.codeAccVgprRead.itemList = [None] * kernel["MIRegPerOut"] * complexMultiplier * len(acc2arch)
    if streamK:
      self.codeAccVgprWrite = Code.Module("AccVgprWrite")
      self.codeAccVgprWrite.itemList = [None] * kernel["MIRegPerOut"] * complexMultiplier * len(acc2arch)
    # srcIdxList for MFMA+LSU+MIArchVgpr
    # This is used to store accumulated values in vgpr to LDS without v_mov instructions
    # Exception for complex case and StoreVectorWidth>1 case. Need to reorder vgpr with v_mov instructions
    noMove = kernel["MIArchVgpr"] and (not kernel["ProblemType"]["ComputeDataType"].isComplex()) and \
             (not kernel["StoreVectorWidth"]>1)
    self.srcIdxList = None
    if noMove:
      self.srcIdxList = [None] * kernel["MIRegPerOut"] * complexMultiplier * len(acc2arch)
    accImOffset = self.AccVgprImagNumOffset(kernel)
    for i in range(len(acc2arch)):
      for cm in range(complexMultiplier):
        for r in range(kernel["MIRegPerOut"]):
          destIdx = (acc2arch[i]*complexMultiplier + cm) * kernel["MIRegPerOut"] + r
          srcIdx = ((i * kernel["MIRegPerOut"] + r) + (cm*accImOffset))
          if not kernel["MIArchVgpr"]:
            accStr = "acc%u"%srcIdx
            if kernel["StoreCInUnroll"]:
              if isOptNLL or self.enableSingleNLLOpt:
                # OptNLL case, use StoreCBuf1
                accStr = accvgpr("StoreCBuf1+%u"%srcIdx)
              else:
                # NoOptNLL case, use StoreCBuf0
                accStr = accvgpr("StoreCBuf0+%u"%srcIdx)
            self.codeAccVgprRead.itemList[destIdx] = Code.Inst("v_accvgpr_read_b32",
                                                            vgpr("ValuC+__placeholder__"),
                                                            accStr, "copy acc to vreg[%u]" % destIdx)
            if streamK:
              self.codeAccVgprWrite.itemList[destIdx] = Code.Inst("v_accvgpr_write_b32",
                                                            accStr,
                                                            vgpr("ValuC+__placeholder__"),
                                                            "copy vreg[%u] to acc" % destIdx)
          else:
            self.codeAccVgprRead.itemList[destIdx] = Code.Inst("v_mov_b32",
                                                              vgpr("ValuC+__placeholder__"),
                                                              vgpr("ValuC+%u"%srcIdx), "copy MI out reg to vreg[%u]" % destIdx)
            if streamK:
              self.codeAccVgprWrite.itemList[destIdx] = Code.Inst("v_mov_b32",
                                                              vgpr("ValuC+%u"%srcIdx),
                                                              vgpr("ValuC+__placeholder__"), "copy vreg[%u] to MI out reg" % destIdx)

          if noMove:
            # keep srcIdx for MFMA+LSU+MIArchVgpr
            self.srcIdxList[destIdx] = srcIdx

    return kStr

  ##############################################################################
  # openmovaccVgpr
  # code segment to mov acc -> VGPR
  # using VGPR index addressing
  ##############################################################################
  def openmovaccVgpr(self,kernel, backupSgpr):
    kStr = ""
    kStr += self.comment("open: mov acc[C]- vgpr using vgpr indexing")
    #backupEnable = kernel["DirectToLdsA"] or kernel["DirectToLdsB"]
    backupEnable = False
    if  backupEnable:
      # m0 backup for DirectToLds
      kStr += inst("s_mov_b32", sgpr(backupSgpr), "m0", "SgprStoreCtmp <- m0")
    kStr += inst("s_set_gpr_idx_on",  sgpr("StoreCIndex0"), "0x1", "Enable GPR indexing mode: VSRC0_rel")
    return kStr

  ##############################################################################
  # getAccVgprInc1
  ##############################################################################
  def getAccVgprInc1(self,kernel,frequency=1):
    # return acc register increment for accvgpr_read instructions for each iteration
    # frequency = 1 or 2
    inc = 0
    numReg = self.StoreCInUnrollNumReg
    numRows = self.StoreCInUnrollnumRows
    numReg1Block = numReg * numRows * kernel["VectorWidthA"]
    if self.StoreCInUnrollNumHorizontalTiles == 1:
      if self.StoreCInUnrollNumInterleaveV == 1:
        # no line interleave case
        inc = numReg * frequency
      else:
        # line interleave case
        if frequency == 1:
          inc = numReg1Block
        else:
          inc = numReg
    else:
      # multiple Horizontal tile case (supporting only number of horizontal tiles > 2)
      inc = numReg1Block * frequency
    return inc

  ##############################################################################
  # getAccVgprOffsetB1
  ##############################################################################
  def getAccVgprOffsetB1(self,kernel):
    numReg = self.StoreCInUnrollNumReg
    numRows = self.StoreCInUnrollnumRows
    numReg1Block = numReg * numRows * kernel["VectorWidthA"]
    if self.StoreCInUnrollNumHorizontalTiles == 1 and self.StoreCInUnrollNumInterleaveV == 1:
      offsetV1 = numReg1Block
    elif self.StoreCInUnrollNumHorizontalTiles > 1 and self.StoreCInUnrollNumInterleaveV > 1:
      offsetV1 = numReg1Block * self.StoreCInUnrollNumHorizontalTiles
    else:
      offsetV1 = numReg
    return offsetV1

  ##############################################################################
  # getAccVgprOffsetB2
  ##############################################################################
  def getAccVgprOffsetB2(self,kernel):
    numReg = self.StoreCInUnrollNumReg
    B2Iterations = self.StoreCInUnrollGprIncB2Iterations
    if self.StoreCInUnrollNumHorizontalTiles > 1 and self.StoreCInUnrollNumInterleaveV > 1:
      inc = numReg
    else:
      inc = numReg * kernel["VectorWidthA"] * B2Iterations
    return inc

  ##############################################################################
  # getAccVgprOffsetB3
  ##############################################################################
  def getAccVgprOffsetB3(self,kernel):
    numReg = self.StoreCInUnrollNumReg
    B3Iterations = self.StoreCInUnrollGprIncB3Iterations
    inc = numReg * kernel["VectorWidthA"] * B3Iterations
    return inc

  ##############################################################################
  # getAccVgprCode
  ##############################################################################
  def getAccVgprCode(self,kernel,odd):
    kStr = ""
    inc1 = self.getAccVgprInc1(kernel)
    for x in self.accVgprTemplate.items():
      kStr += str(x)
    if odd:
      # odd case, add acc register index by 1 iteration
      kStr = kStr.replace("accgprStoreCBuf1", "accgprStoreCBuf1+%u"%inc1)
    return kStr

  ##############################################################################
  # closemovaccVgpr
  # code segment to mov acc -> VGPR
  # using VGPR index addressing
  ##############################################################################
  def closemovaccVgpr(self,kernel, backupSgpr):
    kStr = ""
    kStr += self.comment("close: mov acc[C]- vgpr using vgpr indexing")
    kStr += inst("s_set_gpr_idx_off", "Disable GPR indexing mode: VSRC0_rel")
    # split instruction set here
    first = kStr
    kStr = ""
    #backupEnable = kernel["DirectToLdsA"] or kernel["DirectToLdsB"]
    backupEnable = False
    if  backupEnable:
      # m0 restore for DirectToLds
      kStr += inst("s_mov_b32", "m0", sgpr(backupSgpr)," m0 <- SgprStoreCtmp")
    else:
      # set m0 for LDS clamp
      clampSize = kernel["LdsNumElements"] * self.bpeAB
      kStr += inst("s_mov_b32", "m0", hex(clampSize), "LDS clamp at %u bytes"%(clampSize) )
    second = kStr
    return first, second

  ##############################################################################
  # MulMIoutAlphaToArch
  # function to handle MFMA alpha*MIout to Arch VGPR register
  ##############################################################################
  def MulMIoutAlphaToArch(self, kernel):
    kStr = ""
    kStr += self.comment("Multiply MI out register with Alpha -> C Vgpr register")

    acc2arch, _ = self.AccToArchMapper(kernel)

    self.codeMulAlpha = Code.Module("MulAlpha")
    self.codeMulAlpha.itemList = [None] * len(acc2arch)
    for i in range(len(acc2arch)):
      destIdx = acc2arch[i]
      srcIdx  = i * kernel["MIRegPerOut"]
      if kernel["ProblemType"]["ComputeDataType"].isDouble():
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mul_f64", vgpr("ValuC+__placeholder__",2),
                                                       sgpr("Alpha",2),
                                                       vgpr("ValuC+%u"%srcIdx,2), "Multiply MI out reg with alpha")
      elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
          (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mul_f32", vgpr("ValuC+__placeholder__"),
                                                       sgpr("Alpha"),
                                                       vgpr("ValuC+%u"%srcIdx), "Multiply MI out reg with alpha")
      elif (kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]):
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_pk_mul_f16", vgpr("ValuC+__placeholder__"),
                                                       sgpr("Alpha"),
                                                       vgpr("ValuC+%u"%srcIdx), "Multiply MI out reg with alpha")
      elif kernel["ProblemType"]["ComputeDataType"].isInt32():
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mul_lo_u32", vgpr("ValuC+__placeholder__"),
                                                       sgpr("Alpha"),
                                                       vgpr("ValuC+%u"%srcIdx), "Multiply MI out reg with alpha")
      elif kernel["ProblemType"]["ComputeDataType"].isComplex():
        numReg = kernel["ProblemType"]["DataType"].numRegisters() // 2 # number of registers for real or img only (numRegisters() includes both)
        numBits = numReg * 32
        accImOffset = self.AccVgprImagNumOffset(kernel)
        imod = Code.Module()
        # Allocate vgpr for alpha later (when actually generating the alpha code)
        vtmp1 = "__placeholdervtmp1__"
        vtmp2 = "__placeholdervtmp2__"
        # tmp1 = a.real * b.real
        imod.addInst("v_mul_f%u"%numBits, vtmp1, sgpr("Alpha+0", numReg), vgpr("ValuC+%u"%srcIdx, numReg), "")
        # tmp2 = a.imag * b.real
        imod.addInst("v_mul_f%u"%numBits, vtmp2, sgpr("Alpha+%u"%numReg, numReg), vgpr("ValuC+%u"%srcIdx, numReg), "")
        # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
        imod.addText("v_fma_f%u %s, %s, -%s, %s%s" % (numBits, vgpr("ValuC+__placeholder__", numReg), sgpr("Alpha+%u"%numReg, numReg), \
                     vgpr("ValuC+%u"%(srcIdx+accImOffset), numReg), vtmp1, self.endLine))
        # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
        imod.addText("v_fma_f%u %s, %s, %s, %s%s" % (numBits, vgpr("ValuC+__placeholder__ +%u"%numReg, numReg), sgpr("Alpha+0", numReg), \
                     vgpr("ValuC+%u"%(srcIdx+accImOffset), numReg), vtmp2, self.endLine))
        self.codeMulAlpha.itemList[destIdx] = imod

    return kStr


  ##############################################################################
  # MoveMIoutToArch
  # function to handle MFMA MIout to Arch VGPR register
  ##############################################################################
  def MoveMIoutToArch(self, kernel):
    kStr = ""
    kStr += self.comment("Rearrange MI out register -> C Vgpr register")

    acc2arch, _ = self.AccToArchMapper(kernel)

    self.codeMulAlpha = Code.Module("MulAlpha")
    self.codeMulAlpha.itemList = [None] * len(acc2arch)
    for i in range(len(acc2arch)):
      destIdx = acc2arch[i]
      srcIdx  = i * kernel["MIRegPerOut"]
      if kernel["ProblemType"]["ComputeDataType"].isDouble():
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_lshlrev_b64", vgpr("ValuC+__placeholder__",2),
                                                       0,
                                                       vgpr("ValuC+%u"%srcIdx,2), "Rearrange MI out reg")
      elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
          (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mov_b32", vgpr("ValuC+__placeholder__"),
                                                       vgpr("ValuC+%u"%srcIdx), "Rearrange MI out reg")
      elif (kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]):
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mov_b32", vgpr("ValuC+__placeholder__"),
                                                       vgpr("ValuC+%u"%srcIdx), "Rearrange MI out reg")
      elif kernel["ProblemType"]["ComputeDataType"].isInt32():
        self.codeMulAlpha.itemList[destIdx] = Code.Inst("v_mov_b32", vgpr("ValuC+__placeholder__"),
                                                       vgpr("ValuC+%u"%srcIdx), "Rearrange MI out reg")
      elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
        accImOffset = self.AccVgprImagNumOffset(kernel)
        imod = Code.Module()
        imod.addInst("v_mov_b32", vgpr("ValuC+__placeholder__"), vgpr("ValuC+%u"%srcIdx), "Rearrange MI out reg")
        imod.addInst("v_mov_b32", vgpr("ValuC+__placeholder__ +1"), vgpr("ValuC+%u"%(srcIdx+accImOffset)), "Rearrange MI out reg")
        self.codeMulAlpha.itemList[destIdx] = imod
      elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
        accImOffset = self.AccVgprImagNumOffset(kernel)
        imod = Code.Module()
        imod.addInst("v_lshlrev_b64", vgpr("ValuC+__placeholder__",2), 0, vgpr("ValuC+%u"%srcIdx,2), "Rearrange MI out reg")
        imod.addInst("v_lshlrev_b64", vgpr("ValuC+__placeholder__ +2",2), 0, vgpr("ValuC+%u"%(srcIdx+accImOffset),2), "Rearrange MI out reg")
        self.codeMulAlpha.itemList[destIdx] = imod

    return kStr

  # Perform 32-bit scalar mul and save 64-bit result in two SGPR
  # src0 and src1 are 32-bit ints in scalar sgpr or small int constants (<64?))
  # signed indicates if input and output data is signed
  # return returns in dst0:dest (lower 32-bit in dst0, high 64-bit in dst1))
  def s_mul_int_64_32(self, dst0, dst1, src0, src1, signed, comment):
    kStr = ""
    sign = "i" if signed else "u"
    assert(dst1 != src0) # no worky since dst1 overwritten by first mul operations
    assert(dst1 != src1) # no worky since dst1 overwritten by first mul operations
    # the else path below has less restrictions but prefer consistency
    if globalParameters["AsmCaps"][self.version]["HasSMulHi"]:
      kStr += inst("s_mul_hi_{}32".format(sign), dst1, src0, src1, comment)
      kStr += inst("s_mul_i32", dst0, src0, src1, comment)
    else:
      if type(src1) != 'str' or not src1.startswith("s"):
        # Swap operands, need a scalar sgpr in src1 (not a constant)
        t = src0
        src0 = src1
        src1 = t
      vtmp0 = self.vgprPool.checkOut(2)
      vtmp1 = vtmp0+1
      kStr += inst("v_mov_b32", vgpr(vtmp0), src0, comment)
      kStr += inst("v_mul_hi_{}32".format(sign), vgpr(vtmp1), vgpr(vtmp0), src1, comment)
      kStr += inst("v_readfirstlane_b32", dst1, vgpr(vtmp1), comment)
      kStr += inst("v_mul_lo_u32", vgpr(vtmp1), vgpr(vtmp0), src1, comment)
      kStr += inst("v_readfirstlane_b32", dst0, vgpr(vtmp1), comment)
      self.vgprPool.checkIn(vtmp0)
    return kStr

  def s_mul_u64_u32 (self, dst0, dst1,  src0, src1, comment):
    return self.s_mul_int_64_32(dst0, dst1, src0, src1, False, comment)

  def s_mul_i64_i32 (self, dst0, dst1,  src0, src1, comment):
    return self.s_mul_int_64_32(dst0, dst1, src0, src1, True, comment)

  # dividend is a symbol (constant or sgpr).  Used directly not inside automatic sgpr(..)
  # dst is 2 consecutive SGPR
  #   result returned in dst0. dst1 is used as a temp,
  # dst[1] cannot be same as divident, dst[0] can be same as dividend and this can be useful
  def scalarMagicDivExplicit(self, dst, dividend, magicNumber, magicAbit, magicShift):
    kStr = ""
    kStr = self.comment("dst1:0 = dividend(%s) / magicTag(%s)" % (dividend, magicNumber))
    kStr += inst("s_mul_hi_u32", sgpr(dst+1), dividend, sgpr(magicNumber), "scalar magic div (magicnum)")
    kStr += inst("s_mul_i32", sgpr(dst+0), dividend, sgpr(magicAbit), "scalar magic div (abit)")
    kStr += inst("s_add_u32", sgpr(dst+0), sgpr(dst+0), sgpr(dst+1), "scalar magic div (combine)")
    kStr += inst("s_lshr_b32", sgpr(dst+0), sgpr(dst+0), sgpr(magicShift), \
                "scalar magic div (shift), quotient in s%s"%dst)
    return kStr

  def scalarMagicDiv(self, dst, dividend, magicTag):
    return self.scalarMagicDivExplicit(dst, dividend,
                                        magicNumber="MagicNumberSize"+magicTag,
                                        magicAbit="MagicAbitSize"+magicTag,
                                        magicShift="MagicShiftSize"+magicTag)

  def bomb(self,cookie=None,scratchVgpr=-1):
      """
      Cause a GPUVM fault.
      Instruction after the bomb will write the cookie to SGPR0, so you can see the cookie in the
      backtrace. Useful for locating which spot in code generated the bomb
      vgprAddr controls which vgpr to overwrite with the null pointer address
      """

      kStr =""
      if scratchVgpr==-1:
        vgprAddr = self.vgprPool.checkOut(2)
      else:
        vgprAddr = scratchVgpr
      if cookie != None:
        if cookie < 0:
          kStr += "bomb_neg%u:\n" % abs(cookie)
        else:
          kStr += "bomb_%u:\n" % abs(cookie)
      kStr += inst("v_mov_b32", vgpr(vgprAddr+0), 0, "")
      kStr += inst("v_mov_b32", vgpr(vgprAddr+1), 0, "")
      #kStr += inst("s_trap",1,  "")
      kStr += inst("_global_load_b32", vgpr(vgprAddr), vgpr(vgprAddr,2), "off", "bomb - force fault" )

      # This move does not execute but appears in the instruction stream immediately following
      # the faulting load:
      if cookie != None:
        kStr += inst("s_mov_b32", sgpr(0), cookie, "bomb cookie=%d(0x%x)"%(cookie,cookie&0xffffffff))

      if scratchVgpr == -1:
        self.vgprPool.checkIn(vgprAddr)
      return kStr

  ##############################################################################
  # assertCommon : Common routine for all assert functions.
  # On entry, we have already set the exec-mask so any enabled lanes should bomb
  ##############################################################################
  def assertCommon(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      self.printedAssertCnt += 1
      # Default cookie for asserts is negative of printed #asserts
      # Can be used to roughly identify which assert in the code is firing
      kStr += self.bomb(cookie if cookie != -1 else -self.printedAssertCnt)
    return kStr

  ##############################################################################
  # assertCmpCommon : Common routine for all assert comparison functions
  ##############################################################################
  def assertCmpCommon(self, cond, val0, val1, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), 0, \
          "assert: saved execmask")
      kStr += inst("_v_cmpx_%s"%cond, self.vcc, val0, val1, "v_cmp" )
      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: restore execmask")
    return kStr

  ##############################################################################
  # Handle different conditions for the asserts:
  # These support uin32 compare, float could be added later
  # Asserts currently modify vcc
  ##############################################################################
  def assert_eq(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("ne_u32", val0, val1, cookie)

  def assert_eq_u16(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("ne_u16", val0, val1, cookie)

  def assert_ne(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("eq_u32", val0, val1, cookie)

  def assert_lt_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("ge_u32", val0, val1, cookie)

  def assert_gt_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("le_u32", val0, val1, cookie)

  def assert_le_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("gt_u32", val0, val1, cookie)

  def assert_ge_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("lt_u32", val0, val1, cookie)

  def assert_ge_i32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("lt_i32", val0, val1, cookie)

  # can left shift w/o losing non-zero bits:
  def assert_no_shift_of(self, val0, shift, stmp, cookie=-1):
    kStr = ""
    # TODO - use BFE here:
    kStr += inst ("s_mov_b32", stmp, hex((shift-1) << (32-log2(shift))), "assert_no_shift_of - compute mask")
    kStr += inst ("s_and_b32", stmp, stmp, val0, "assert_no_shift_of")
    kStr += self.assert_eq(stmp, 0, cookie)
    return kStr

  # asserts if val0 is not an integer multiple of multiple2
  # multiple2 must be a constant and power of 2
  # for example assert_multiple(A, 8) will assert if A is not multiple of 8
  def assert_multiple_b32(self, sval, multiple2, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:

      stmp = sgpr("SaveExecMask") # repurpose to get a tmp sgpr

      kStr += inst("s_and_b{}".format(self.kernel["WavefrontSize"]), stmp, sval, multiple2-1, "mask" )
      kStr += inst("s_cmp_eq_u32", stmp, 0, "if maskedBits==0 then SCC=1 == no fault" )
      kStr += inst("s_mov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), -1, "")
      kStr += inst("s_cmov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask", self.laneSGPRCount),  0, "Clear exec mask")

      kStr += inst("s_and_saveexec_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: saved execmask")

      kStr += self.assertCommon(cookie)

      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: restore execmask")

    return kStr

  def assert_s_eq(self, sval0, sval1, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_and_saveexec_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: saved execmask")

      kStr += inst("s_mov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask", self.laneSGPRCount), -1, "")
      kStr += inst("s_cmp_eq_u32", sval0, sval1, "cmp")
      kStr += inst("s_cmov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask", self.laneSGPRCount),  0, "No assert if SCC=1")

      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: restore execmask")

      return kStr

  def assert_scc_is_1(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_and_saveexec_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: saved execmask")

      kStr += inst("s_mov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), -1, "")
      kStr += inst("s_cmov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount),  0, "No assert if SCC=1")

      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: restore execmask")

      return kStr

  def assert_scc_is_0(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_and_saveexec_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: saved execmask")

      kStr += inst("s_mov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), -1, "")
      kStr += inst("s_cmov_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask", self.laneSGPRCount),  0, "")
      kStr += inst("s_not_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), sgpr("SaveExecMask", self.laneSGPRCount), "Assert if SCC==1")

      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: restore execmask")

      return kStr

  # Assert that all bits in vcc are true, or assert/bomb otherwise
  def assert_vcc_all_true(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), 0, \
          "assert: saved execmask")
      kStr += inst("s_mov_b{}".format(self.kernel["WavefrontSize"]), self.exec, self.vcc, "Predicate based on VCC")
      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: restore execmask")
    return kStr

  # Assert that all bits in vcc are false, or assert/bomb otherwise
  def assert_vcc_all_false(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), sgpr("SaveExecMask",self.laneSGPRCount), 0, \
          "assert: saved execmask")
      kStr += inst("s_not_b{}".format(self.kernel["WavefrontSize"]), self.exec, self.vcc, "Predicate based on !VCC")
      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b{}".format(self.kernel["WavefrontSize"]), self.vcc, sgpr("SaveExecMask",self.laneSGPRCount), \
          "assert: restore execmask")
    return kStr

  # assert v0 + expectedScalarDiff == v1
  # Verify that each element in v1 is scalar offset from v0
  def assert_vector_diff(self, v0, v1, expectedScalarDiff, cookie=-1):
    kStr = ""
    cmpVgpr = self.vgprPool.checkOut(1)
    kStr += inst("_v_add_co_u32", \
                 vgpr(cmpVgpr), self.vcc, \
                 expectedScalarDiff, \
                 v0, \
                 "assert_vector_diff add expectedScalarDiff")
    kStr += self.assert_eq(vgpr(cmpVgpr), v1, cookie)
    self.vgprPool.checkIn(cmpVgpr)
    return kStr

  ########################################
  # Store to Debug Buffer
  ########################################
  def dump(self, vgprStore):
    kStr = ""
    if globalParameters["DebugKernel"]:
      afterDump = -1
      if self.db["DebugKernelMaxItems"] != -1:
        afterDump = self.getUniqLabel()
        kStr += inst("s_cmp_lt_u32", sgpr("DebugKernelItems"), 16,  "")
        kStr += inst("s_cbranch_scc0", "label_%04u"%afterDump, \
                     "skip if already wrote enough work-items" )
        kStr += inst("s_add_u32", sgpr("DebugKernelItems"), \
                     sgpr("DebugKernelItems"), \
                     hex(1), "inc items written" )

      kStr += inst("_global_store_b32", vgpr("AddressDbg", 2), \
          vgprStore, "off", "debug dump store" )
      kStr += inst("_v_add_co_u32", vgpr("AddressDbg"), self.vcc, vgpr("AddressDbg"), \
          hex(4), "debug dump inc" )

      if self.db["DebugKernelMaxItems"] != -1:
        kStr += "label_%04u:%s  %s" % (afterDump, "// skip debug target", self.endLine)

    return kStr

  def dumpSgpr(self, sgprStore):
    kStr = ""
    if globalParameters["DebugKernel"]:
      tmp = self.vgprPool.checkOut(1,"tmp")
      kStr += inst("v_mov_b32", vgpr(tmp), sgprStore, "debug dump sgpr store")
      kStr += self.dump(tmp)
      self.vgprPool.checkIn(vgpr(tmp))
    return kStr
