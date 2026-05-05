import math
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

from ..Common import printWarning, roundUp, print2, DebugConfig, DataDirection, \
  INDEX_CHARS, IsaVersion


from rocisa.code import Module, TextBlock, StructuredModule, KernelBody, Label
from rocisa.label import LabelManager

from rocisa.container import MUBUFModifiers, vgpr, sgpr, accvgpr, mgpr
from rocisa.enum import InstType, SelectBit, CacheScope
from rocisa.instruction import MFMAInstruction

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Type
from contextlib import contextmanager
from collections import deque
from rocisa import rocIsa, countInstruction, countGlobalRead, \
  countLocalRead, countLocalWrite, countDSStoreB256, getMFMAs
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import KernelBody, Label, Module, StructuredModule, TextBlock
from rocisa.container import (
  DPPModifiers, DSModifiers, EXEC, HWRegContainer, MUBUFModifiers,
  RegisterContainer, VCC, VOP3PModifiers,
  accvgpr, mgpr, replaceHolder, sgpr, vgpr,
)
from rocisa.enum import CacheScope, DataTypeEnum, InstType, RegisterType, SelectBit
from rocisa.instruction import (
  BufferLoadB128, BufferLoadB32, BufferLoadB64,
  BufferLoadD16B16, BufferLoadD16U8,
  DSLoad2B32, DSLoad2B64, DSLoadB128, DSLoadB32, DSLoadB64,
  DSLoadB64TrB16, DSLoadInstruction, DSLoadU16, DSLoadU8,
  DSStore2B32, DSStore2B64, DSStoreB128, DSStoreB16, DSStoreB256,
  DSStoreB32, DSStoreB64, DSStoreB8, DSStoreInstruction,
  FlatLoadB128, FlatLoadB32, FlatLoadB64,
  FlatStoreB128, FlatStoreB32, FlatStoreB64,
  Instruction, MacroInstruction,
  MFMAInstruction, MXMFMAInstruction, SMFMAInstruction,
  SAddCU32, SAddU32, SBarrier, SBranch,
  SCBranchSCC0, SCBranchSCC1, SCBranchVCCNZ,
  SCmpEQU32, SCmpLeU32, SLShiftLeftB32, SLongBranchPositive,
  SMovB32, SMovB64, SMulI32, SNop,
  SSetPrior, SSetRegIMM32B32, SSubBU32, SSubU32, SWaitAlu, SWaitCnt, SXorB32,
  VAccvgprWrite, VAddCCOU32, VAddCOU32, VAddU32, VAndB32,
  VCmpXEqU32, VCndMaskB32, VFmaMixF32, VMadMixF32,
  VLShiftLeftB32, VLShiftRightB32, VMovB32, VMovB64,
  VMulLOU32, VPermlane16SwapB32, VReadfirstlaneB32, VSubU32, VXorB32,
)
from rocisa.label import LabelManager
from rocisa.register import RegisterPool
# Store various scheduling info
class ScheduleInfo:

  availableVgprATiles = field(init=False)
  availableVgprBTiles = field(init=False)

  usedVgprATiles = field(init=False)
  usedVgprBTiles = field(init=False)

  def __init__(self, aTileInfo, bTileInfo):
    # TODOBS: check that vgpr tiles are init first before calling these
    self.availableVgprATiles = deque(list(range(len(aTileInfo.vgprTiles))))
    self.availableVgprBTiles = deque(list(range(len(bTileInfo.vgprTiles))))

    self.usedVgprATiles = dict()
    self.usedVgprBTiles = dict()


# Tile info
class TileInfo:

  class RegisterList:
    regPool = field(init=False)
    regValues : List[int] = field(init=False)

    def __init__(self, pool):
      self.regPool = pool
      self.regValues = []

    def append(self, val):
      self.regValues.append(val)

    def index(self, val):
      return self.regValues.index(val)

    def __iter__(self):
      for vals in self.regValues:
        yield vals

    def __len__(self):
      return len(self.regValues)

    def __str__(self):
      return str(self.regValues)

  class RegisterTileInfo:
    tileSize: int = 0
    regList = field(init=False)

    def __init__(self, pool):
      self.regList = TileInfo.RegisterList(pool)

    def append(self, val):
      self.regList.append(val)

    def index(self, val):
      return self.regList.index(val)

    def __iter__(self):
      # The generator automatically handles the iteration logic
      for vals in self.regList:
        yield vals

    def __str__(self):
      return str(self.regList)

  class SubtileInfo:
    tc: str = field(init=False)
    subtileId: List[int] = field(init=False)

    # List of GR that loads this subtile
    globalReadMap: List[int] = field(init=False)
    # List of LR that loads this subtile
    localReadMap: List[int] = field(init=False)

    # Store registers used for constant offsets
    useSgpr = field(init=False)
    regListId: int = -1

    def __init__(self, tc, subtileId):
      self.tc = tc
      self.subtileId = subtileId
      self.globalReadMap = []
      self.localReadMap = []

  tc: str = field(init=False)
  bpe: float = 0
  depthUBytes: int = 0 # Num bytes in K dim for all subtiles
  subIterKBytes: int = 0 # Num bytes in K dim for one subtile
  loadWidthGR: int = 16  # Always assume widest load width for global reads
  loadWidthLR: int = 0  # load width in bytes for local reads
  isSwizzled: bool = False

  # MMA Shape is w.r.t to data element (not size in bytes)
  #
  mmaTileShape: List[int] = field(init=False)
  mmaTileSize: int = 0 # subtile size in bytes
  mmaTileLocalTotalCount: int = 0 # total number of mmaTiles
  mmaTileRegCount: float = 0 # number of registers needed for per mma tile for specific A/B matrix

  subtileShape: List[int] = field(init=False)
  subtileSize: int = 0 # subtile size in bytes
  subtileLocalTotalCount: int = 0

  globalMMATileGrid: List[int]  = field(init=False)
  globalSubtileGrid: List[int]  = field(init=False)

  localMMATileGrid: List[int]  = field(init=False)
  localSubtileGrid: List[int]  = field(init=False)

  localSubtiles: List[SubtileInfo] = field(init=False)
  localSubtilesRegister: List[RegisterList] = field(init=False)

  loadRatioGR: int = 0
  numGRPerSubtile: int = 0 # may not be needed
  numGRTotal: int = 0

  loadRatioLR: int = 0
  numLRPerSubtile: int = 0 # may not be needed
  numLRTotal: int = 0

  sharedVgprGROffset: List[int] = field(init=False)
  sharedVgprLROffset: List[int] = field(init=False)
  sharedVgprLROffsetSwap: List[int] = field(init=False)

  vgprTileFactor: float = 1.0
  # VGPR buffers available for this tile
  vgprTiles: List[RegisterTileInfo] = field(init=False)

  # MX scale fields (set for A/B when mxBlock > 0, else 0)
  mxBlock: int = 0

  def __init__(self, tc, kernel):
    isAB = tc in ['A', 'B']
    isMXSAB = tc in ['MXSA', 'MXSB']

    self.subtileShape = [1, 2]

    self.tc = tc
    self.isSwizzled = isMXSAB

    isA = tc in ['A', 'MXSA']
    _tc = 'A' if isA else 'B'

    if isAB or isMXSAB:

      # TODO query vgpr factors from kernel
      self.vgprTileFactor = 1.0 if tc == 'A' else 1.0
      miWaveGroupSize0 = kernel["MIWaveGroup"][0 if isA else 1]
      miWaveGroupSize1 = 1

      macroTile = kernel["MacroTileA"] if isA else kernel["MacroTileB"]
      depthU = kernel["_DepthU%s"%tc]
      # TODO: Need to update ProblemType to query scale size?
      bpe = kernel["ProblemType"]["DataType%s"%tc].numBytes() if isAB else 1
      self.bpe = bpe
      self.depthUBytes = int(depthU * bpe)

      numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]

      # Always assumes widest load is used
      loadWidth = 128
      # Number of bytes loaded per wave with widest load width, global
      numBytesGRPerWave = (loadWidth // 8) * kernel["WavefrontSize"]
      # Number of bytes loaded per wave with widest load width, local
      numBytesLRPerWave = (loadWidth // 8) * kernel["WavefrontSize"]

      # MMA Tile Shape is based on matrix instruction
      mmaTileShape0 = kernel["MatrixInstM"]
      mmaTileShape1 = kernel["MatrixInstK"]
      if isMXSAB:
        mmaTileShape1 //= kernel["ProblemType"].get("MXBlock%s"%_tc)
      self.mmaTileShape = [mmaTileShape0, mmaTileShape1]
      mmaTileGrid0 = macroTile // mmaTileShape0
      mmaTileGrid1 = depthU // mmaTileShape1

      subtileShape0 = self.subtileShape[0]
      subtileShape1 = self.subtileShape[1]
    else: # Tile info for C matrix
      # TODOBS: check if 'C' or 'D'.. decide which to use
      self.vgprTileFactor = 1.0
      miWaveGroupSize0 = kernel["MIWaveGroup"][0]
      miWaveGroupSize1 = kernel["MIWaveGroup"][1]

      macroTile = kernel["MacroTile0"]
      depthU = kernel["MacroTile1"]
      bpe = kernel["ProblemType"]["ComputeDataType"].numBytes()
      self.bpe = bpe
      self.depthUBytes = int(depthU * bpe)

      numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]

      # Always assumes widest load is used
      loadWidth = 128
      # Number of bytes loaded per wave with widest load width, global
      numBytesGRPerWave = (loadWidth // 8) * kernel["WavefrontSize"]
      # Number of bytes loaded per wave with widest load width, local
      numBytesLRPerWave = (loadWidth // 8) * kernel["WavefrontSize"]

      # MMA Tile Shape is based on matrix instruction
      mmaTileShape0 = kernel["MatrixInstM"]
      mmaTileShape1 = kernel["MatrixInstM"]
      self.mmaTileShape = [mmaTileShape0, mmaTileShape1]
      mmaTileGrid0 = macroTile // mmaTileShape0
      mmaTileGrid1 = depthU // mmaTileShape1

      subtileShape0 = self.subtileShape[0]
      subtileShape1 = self.subtileShape[0]

    assert kernel["MatrixInstM"] == 16, \
      "SubtileBasedKernel only supports MatrixInstM=16, got %u" % kernel["MatrixInstM"]
    assert kernel["MatrixInstK"] in (32, 128), \
      "SubtileBasedKernel only supports MatrixInstK=32 (bf16) or MatrixInstK=128 (mxfp4), got %u" % kernel["MatrixInstK"]

    self.mmaTileSize = int(mmaTileShape0 * mmaTileShape1 * bpe)
    self.loadWidthLR = self.mmaTileSize // kernel["WavefrontSize"]
    # Number of registers needed for one tile, count w.r.t dword
    self.mmaTileRegCount = (self.mmaTileSize // kernel["WavefrontSize"]) / 4
    # Number of mma tiles for each wave
    self.mmaTileLocalTotalCount = (mmaTileGrid0 // miWaveGroupSize0) * (mmaTileGrid1 // miWaveGroupSize1)

    # Subtile Shape is w.r.t of units of mma tile

    self.subtileSize = subtileShape0 * subtileShape1 * self.mmaTileSize

    # TODO: This won't be needed if we assume all loads are split
    # Compute number of mfma tiles globally in MT0 dim

    self.globalMMATileGrid = [mmaTileGrid0, mmaTileGrid1]
    self.globalSubtileGrid = [mmaTileGrid0 // subtileShape0, mmaTileGrid1 // subtileShape1]

    # Compute number of mfma tiles locally (wave pov)
    self.localMMATileGrid = deepcopy(self.globalMMATileGrid)
    self.localSubtileGrid = deepcopy(self.globalSubtileGrid)
    self.localMMATileGrid[0] //= miWaveGroupSize0
    self.localSubtileGrid[0] //= miWaveGroupSize0
    self.localMMATileGrid[1] //= miWaveGroupSize1
    self.localSubtileGrid[1] //= miWaveGroupSize1

    self.subIterKBytes = self.depthUBytes // self.localSubtileGrid[1]

    self.subtileLocalTotalCount = self.localSubtileGrid[0] * self.localSubtileGrid[1]

    # Allocate subtileInfo structs
    self.localSubtiles = []
    self.localSubtilesRegister = []
    for sId0 in range(self.localSubtileGrid[0]):
      for sId1 in range(self.localSubtileGrid[1]):
        self.localSubtiles.append(TileInfo.SubtileInfo(tc, [sId0, sId1]))

    if isAB or isMXSAB:
      # Compute load ratio
      # Represents the amount of subtiles fetched by a single global load across all waves
      # < 1 means a global load fetches multiple subtiles
      self.loadRatioGR = (numBytesGRPerWave * numWaves) / self.subtileSize / miWaveGroupSize0
      self.numGRPerSubtile = int(math.ceil(1/self.loadRatioGR))
      self.numGRTotal = int((self.localSubtileGrid[0] * self.localSubtileGrid[1]) / self.loadRatioGR)

      # Compute load ratio
      # Represents the amount of subtiles fetched by a single ds_read
      # < 1 means a global load fetches multiple subtiles
      self.loadRatioLR = (numBytesLRPerWave) / self.subtileSize
      self.numLRPerSubtile = int(math.ceil(1/self.loadRatioLR))
      self.numLRTotal = int((self.localSubtileGrid[0] * self.localSubtileGrid[1]) / self.loadRatioLR)

      # Scale tensor geometry (MX block scaling)
      mxBlockKey = "MXBlock%s"%_tc
      self.mxBlock = kernel["ProblemType"].get(mxBlockKey, 0)

      # Map subtiles to GR
      for sId0 in range(self.localSubtileGrid[0]):
        for sId1 in range(self.localSubtileGrid[1]):
          linearId = self.getLocalSubtileLinearId(sId0, sId1)
          subtileInfo = self.localSubtiles[linearId]
          baseGR = math.floor(linearId / self.loadRatioGR)
          for nGL in range(self.numGRPerSubtile):
            subtileInfo.globalReadMap.append(baseGR + nGL)
          baseLR = math.floor(linearId / self.loadRatioLR)
          for nLL in range(self.numLRPerSubtile):
            subtileInfo.localReadMap.append(baseLR + nLL)
          # print("GR map", sId0, sId1, subtileInfo.globalReadMap)
          # print("LR map", sId0, sId1, subtileInfo.localReadMap)



  def __str__(self):
    lines = [
      f"TileInfo(tc={self.tc})",
      f"  mmaTileShape:           {self.mmaTileShape if isinstance(getattr(self, 'mmaTileShape', None), list) else 'not set'}",
      f"  mmaTileSize:            {self.mmaTileSize} bytes",
      f"  mmaTileRegCount:        {self.mmaTileRegCount}",
      f"  mmaTileLocalTotalCount: {self.mmaTileLocalTotalCount}",
      f"  subtileShape:           {self.subtileShape}",
      f"  subtileSize:            {self.subtileSize} bytes",
      f"  subtileLocalTotalCount: {self.subtileLocalTotalCount}",
      f"  globalMMATileGrid:      {self.globalMMATileGrid}",
      f"  globalSubtileGrid:      {self.globalSubtileGrid}",
      f"  localMMATileGrid:       {self.localMMATileGrid}",
      f"  localSubtileGrid:       {self.localSubtileGrid}",
      f"  mxBlock:                {self.mxBlock}",
      f"  loadRatioGR:            {self.loadRatioGR}",
      f"  numGRPerSubtile:        {self.numGRPerSubtile}",
      f"  numGRTotal:             {self.numGRTotal}",
      f"  loadRatioLR:            {self.loadRatioLR}",
      f"  numLRPerSubtile:        {self.numLRPerSubtile}",
      f"  numLRTotal:             {self.numLRTotal}",
      f"  vgprTileFactor:         {self.vgprTileFactor}",
      f"  vgprTiles:              {[str(t) for t in self.vgprTiles] if isinstance(getattr(self, 'vgprTiles', None), list) else 'not allocated'}",
      f"  sharedVgprGROffset:     {self.sharedVgprGROffset if isinstance(getattr(self, 'sharedVgprGROffset', None), list) else 'not allocated'}",
      f"  sharedVgprLROffset:     {self.sharedVgprLROffset if isinstance(getattr(self, 'sharedVgprLROffset', None), list) else 'not allocated'}",
    ]
    return "\n".join(lines)

  ####################################
  # Given 2d local mma tile Id, return 2d id for local subtile containing that tile
  def getLocalSubtileIdFromMMATile(self, mmaId0, mmaId1):
    return [mmaId0 // self.subtileShape[0], mmaId1 // self.subtileShape[1]]

  def getLocalSubtileLinearId(self, sId0, sId1):
    # Returns linear id for subtiles assumes block col major format
    return sId1 * self.localSubtileGrid[0] + sId0

  def getLocalSubtileIdFromLinearId(self, linearId):
    sId0 = linearId % self.localSubtileGrid[0]
    sId1 = linearId // self.localSubtileGrid[0]
    return [sId0, sId1]

  def getSubtileShapeLinearId(self, k0, k1):
    # Returns linear id within a subtile, col major
    return k1 * self.subtileShape[0] + k0

  def getLocalMMATileLinearId(self, mmaId0, mmaId1):
    # Returns linear id for subtiles assumes block col major format
    return mmaId1 * self.localMMATileGrid[0] + mmaId0

  def allocOffsetRegisters(self, writer, kernel):
    self.sharedVgprGROffset = []
    self.sharedVgprLROffset = []
    self.sharedVgprLROffsetSwap = []

    isSwizzledScales = self.isSwizzled and self.tc in ['MXSA', 'MXSB']

    # Allocate share vgprs for GR
    for i in range(self.numGRPerSubtile):
      self.sharedVgprGROffset.append(writer.vgprPool.checkOut(1))

    # Allocate shared vgprs for LR
    for i in range(self.numLRPerSubtile):
      self.sharedVgprLROffset.append(writer.vgprPool.checkOut(1))
      self.sharedVgprLROffsetSwap.append(writer.vgprPool.checkOut(1))

    # For swizzled scale layout, we assume we can stream.
    # So only need shared vgprs for GR
    if isSwizzledScales:
      return

    # Allocate registers for each subtile
    # TODOBS: Check TLU instead of hardcoding False
    perpDimSize = (self.localSubtileGrid[1] if False else self.localSubtileGrid[0])
    if self.loadRatioGR == 2.0:
      perpDimSize = math.ceil(perpDimSize / self.loadRatioGR)

    # TOBODS: Can this be done better
    for reg in range(perpDimSize):
      tmpSgprBuffer = 3 # Hardcoded for now, the amount of sgprs to use for temps
      sgprLimit = writer.states.regCaps["MaxSgpr"] - tmpSgprBuffer
      regPool = writer.sgprPool if writer.sgprPool.size() < sgprLimit else writer.vgprPool
      self.localSubtilesRegister.append(TileInfo.RegisterList(regPool))
      # No registers needed for perp 0
      if reg == 0:
        continue
      if regPool == writer.sgprPool:
        # TODOBS: Need to prevent overflow here, better way to do it?
        self.localSubtilesRegister[-1].append(regPool.checkOut(1, preventOverflow=False))
      else:
        for i in range(self.numGRPerSubtile):
          self.localSubtilesRegister[-1].append(regPool.checkOut(1, preventOverflow=False))
    # Iterate through subtiles and allocate vgpr/sgpr if needed
    linearId = 0
    for st in self.localSubtiles:

      # Get 2D Id for subtile
      sId0, sId1 = self.getLocalSubtileIdFromLinearId(linearId)
      linearId += 1

      # TODOBS: Check TLU instead of hardcoding
      slowId = sId1 if False else sId0
      # Only associate a SGPR to 1 other subtile when loadRatioGR == 2.0
      if self.loadRatioGR == 2.0:
        slowId = int(slowId // self.loadRatioGR)
      st.regListId = slowId
      st.useSgpr = self.localSubtilesRegister[slowId].regPool == writer.sgprPool

  def allocVgprTileRegisters(self, writer, kernel, schedulerManaged=False):
    self.vgprTiles = []

    numMMATiles = self.localMMATileGrid[0] * self.localMMATileGrid[1]
    numMMATilesPerReg = max(1, int(1//self.mmaTileRegCount))

    for i in range(int(self.vgprTileFactor * numMMATiles)):
      # Determine which pool to allocate registers from
      if self.tc in ['A', 'B', 'MXSA', 'MXSB']:
        self.vgprTiles.append(TileInfo.RegisterTileInfo(writer.vgprPool))
      else:
        useAgpr = True
        if useAgpr:
          maxAgpr = writer.states.regCaps["PhysicalMaxVgpr"] - writer.states.regCaps["MaxVgpr"]
          # TODOBS: agpr limit is hardcoded here.. fix
          if writer.agprPool.size() < maxAgpr:
            self.vgprTiles.append(TileInfo.RegisterTileInfo(writer.agprPool))
          else:
            self.vgprTiles.append(TileInfo.RegisterTileInfo(writer.vgprPool))

      if i % numMMATilesPerReg != 0:
        continue
      # TODOBS: Hard code this block for now?
      numDword = int(math.ceil(self.mmaTileRegCount));
      for j in range(0, numDword, numDword):
        pool = self.vgprTiles[-1].regList.regPool
        vstart = pool.checkOutAligned(numDword,numDword)
        for k in range(numDword):
          self.vgprTiles[-1].append(vstart + k)

  def allocScaleVgprTiles(self, writer, kernel):
    if self.mxBlock == 0:
      return
    numScaleVgprs = math.ceil(self.localMMATileGrid[0] / 2)
    self.scaleVgprTiles = []
    for i in range(numScaleVgprs):
      self.scaleVgprTiles.append(writer.vgprPool.checkOut(1))

  def deallocOffsetRegisters(self, writer, kernel):
    # checkin GR registers
    for voff in self.sharedVgprGROffset:
      writer.vgprPool.checkIn(voff)
    # checkin LR registers
    for voff in self.sharedVgprLROffset:
      writer.vgprPool.checkIn(voff)
    # checkin LR registers
    for voff in self.sharedVgprLROffsetSwap:
      writer.vgprPool.checkIn(voff)

    for reg in self.localSubtilesRegister:
      regPool = reg.regPool
      for val in reg.regValues:
        regPool.checkIn(val)

  def deallocScaleVgprTiles(self, writer, kernel):
    for sv in self.scaleVgprTiles:
      writer.vgprPool.checkIn(sv)
    self.scaleVgprTiles = []

  def deallocVgprTileRegisters(self, writer, kernel):
    numMMATilesPerReg = max(1, int(1 // self.mmaTileRegCount))
    for i, vtiles in enumerate(self.vgprTiles):
      if i % numMMATilesPerReg != 0:
        continue
      pool = vtiles.regList.regPool
      if vtiles.regList.regValues:
        pool.checkIn(vtiles.regList.regValues[0])

def _computeLROffset(module, kernel, tileInfo, colOffset, rowOffset):
  tc = tileInfo.tc
  wavesize = kernel["WavefrontSize"]
  subIterKBytes = tileInfo.subIterKBytes
  loadWidth = tileInfo.loadWidthLR
  numMFMACols = int(tileInfo.mmaTileShape[1] * tileInfo.bpe) // loadWidth  # TN case only
  blockSize = subIterKBytes // loadWidth

  module.add(VMovB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]), src=vgpr(colOffset), comment="%s: laneId"%tc))
  for vgprId in range(1, len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId-1]), src1=hex(numMFMACols), comment="%s: colOffset for MFMA %u of subtile"%(tc, vgprId)))
    module.add(VAndB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=hex(blockSize-1), comment="%s: colOffset = colOffset %% block_size"%tc))

  for vgprId in range(0, len(tileInfo.sharedVgprLROffset)):
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(tileInfo.sharedVgprLROffset[vgprId]), comment="%s: colOffset*loadWidth"%tc))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(rowOffset), comment="%s: row + col"%tc))

def _applyWavePartitionLROffset(module, writer, kernel, tileInfo):
  """Apply wave-based partition offset to LR offsets.

  loadRatioGR >= 2.0: no partition needed, contiguous subtiles (1x4 for A , 4x1 for B)
  loadRatioGR == 1.0: 2x2 config, each wave loads half of the subtile
  loadRatioGR == 0.5: 4x1 for A , 1x4 for B. Split in 4 subtiles groups
  """
  tc = tileInfo.tc

  if tileInfo.loadRatioGR >= 2.0:
    return

  wavesize = kernel["WavefrontSize"]
  subIterKBytes = tileInfo.subIterKBytes
  MT = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]

  waveId = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="waveId"))

  if tileInfo.loadRatioGR == 1.0:
    # W0 W2
    # W1 W3
    # W1-3 : A / W2-3 : B
    if tc == 'A':
      module.add(VAndB32(dst=vgpr(waveId), src0=hex(1), src1=vgpr(waveId), comment="%s: waveId %% 2"%tc))
    else:
      module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(1), src=vgpr(waveId), comment="%s: waveId / 2"%tc))

    sInterval = MT * subIterKBytes // 2
  elif tileInfo.loadRatioGR == 0.5:
    sInterval = MT * subIterKBytes // 4
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))

  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(sInterval), comment="%s: interleave stride"%tc))
  module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr), comment=""))
  for vgprId in range(len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(waveId), comment="%s: wave partition LR offset"%tc))

  writer.vgprPool.checkIn(waveId)
  writer.sgprPool.checkIn(tmpSgpr)

def _lraWavePartitioning(module, writer, kernel):
  """Compute waveId and apply per-matrix wave partition offsets."""
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  _applyWavePartitionLROffset(module, writer, kernel, tileInfoA)
  _applyWavePartitionLROffset(module, writer, kernel, tileInfoB)


def setExecMask(module, writer, maskLo, maskHi):
  tmpSgpr = writer.sgprPool.checkOutAligned(2, 2, "setExecMask tmpSgpr", False)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(maskLo), comment="exec mask lo"))
  module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(maskHi), comment="exec mask hi"))
  module.add(SMovB64(dst=EXEC(), src=sgpr(tmpSgpr, 2), comment="Set exec mask"))
  writer.sgprPool.checkIn(tmpSgpr)

##################################################
# Subroutine to generate LR offset calculation code
#
def lraTileAssignment(writer, kernel):
  module = Module()
  module.addComment0("LR Offset Calculation for Subtile Based Tiling")

  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo

  # Input Parameters.
  subIterKBytes = tileInfoA.subIterKBytes
  wavesize = kernel["WavefrontSize"]

  mi_m = tileInfoA.mmaTileShape[0]
  loadWidth = tileInfoA.loadWidthLR
  ldsRowBankSize = 64*4 # 64 banks, 4 bytes per bank
  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes
  assert tileInfoA.mmaTileShape == tileInfoB.mmaTileShape, "Expect same MMA tile shape for A and B"

  blockSize = subIterKBytes // loadWidth

  tmpVgpr = writer.vgprPool.checkOut(6)
  lane16, lane16Group, rotation, rowOffset, colOffset = range(tmpVgpr, tmpVgpr + 5)

  # Calculate lane16 and lane16Group for current wave (used by MFMA layout)
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize-1, comment="laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="lane16Group"))
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="laneId % 16"))

  swizzling = True
  if swizzling:
    # Get lds row id
    module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(lane16), comment="lds_row_id"))
    module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="(lds_row_id //2 )"))
    # Calculate rotation
    module.add(VLShiftLeftB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="rotation=(lds_row_id //2) * 2"))
    # Apply rotation on Col
    module.add(VAddU32(dst=vgpr(colOffset), src0=vgpr(rotation), src1=vgpr(lane16Group), comment="colOffset = rotation + lane16Group"))
    module.add(VAndB32(dst=vgpr(colOffset), src0=vgpr(colOffset), src1=hex(blockSize-1), comment="colOffset = colOffset % blockSize"))
    # Swizzle col
    setExecMask(module, writer, 0x33333333, 0x33333333)
    module.add(VPermlane16SwapB32(dst=vgpr(colOffset), src=vgpr(colOffset), comment="apply swizzling"))
    setExecMask(module, writer, -1, -1)
  else:
    module.add(VMovB32(dst=vgpr(colOffset), src=vgpr(lane16Group), comment="colOffset = lane16Group"))

  # Row
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(subIterKBytes.bit_length()-1), src=vgpr(lane16), comment="offsetRow = subIterKBytes*lane16"))

  # Calculate LR offset for A and B
  _computeLROffset(module, kernel, tileInfoA, colOffset, rowOffset)
  _computeLROffset(module, kernel, tileInfoB, colOffset, rowOffset)

  writer.vgprPool.checkIn(tmpVgpr)

  # Wave partitioning (e.g. 2x2 or 4x1/1x4)
  _lraWavePartitioning(module, writer, kernel)

  # Apply global offset on B
  MT0A = tileInfoA.globalMMATileGrid[0] * tileInfoA.mmaTileShape[0]
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), src0=writer.ldsStartOffsetB, src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), comment="B matrix offset in LDS"))

  return module


def _zeroRegRange(module, writer, tileInfo, firstReg, totalRegs, isAgpr):
  """Zero a contiguous register range using MFMA for blocks of 16, scalar writes for remainder."""
  tileAlias = accvgpr if isAgpr else vgpr
  tileCopyInst = VAccvgprWrite if isAgpr else VMovB32
  regsPerMfma = 16
  numMfma = totalRegs // regsPerMfma

  if numMfma > 0:
    tmpVgpr = writer.vgprPool.checkOutAligned(2, 2)
    module.add(VMovB64(dst=vgpr(tmpVgpr, 2), src=0, comment=""))
    module.add(SNop(waitState=1, comment="wait for vgpr to be ready before MFMA"))
    for i in range(numMfma):
      r = firstReg + i * regsPerMfma
      module.add(MFMAInstruction(instType=InstType.INST_I8, accType=InstType.INST_I32,
                                 variant=[32, 32, 16, 1], mfma1k=False,
                                 acc=tileAlias(r, regsPerMfma),
                                 a=vgpr(tmpVgpr, 2), b=vgpr(tmpVgpr, 2),
                                 acc2=0,
                                 comment="init%s: [%u:%u]"%(tileInfo.tc, r, r + regsPerMfma - 1)))
    writer.vgprPool.checkIn(tmpVgpr)

  for i in range(numMfma * regsPerMfma, totalRegs):
    module.add(tileCopyInst(dst=tileAlias(firstReg + i), src=0, comment="init%s"%(tileInfo.tc)))

def initVgprTilesToZero(writer, kernel, tileInfo):
  """Initialize vgprTiles to zero using MFMA for blocks of 16, scalar writes for remainder."""
  module = Module()
  module.addComment0("Init %s vgprTiles to zero"%(tileInfo.tc))

  if not tileInfo.vgprTiles:
    return module

  # Group contiguous tiles by pool type (agpr vs vgpr) since D tiles can use both
  firstReg = tileInfo.vgprTiles[0].regList.regValues[0]
  totalRegs = 0
  curPool = tileInfo.vgprTiles[0].regList.regPool

  for tile in tileInfo.vgprTiles:
    pool = tile.regList.regPool
    numRegs = len(tile.regList.regValues)
    if pool != curPool:
      _zeroRegRange(module, writer, tileInfo, firstReg, totalRegs, curPool == writer.agprPool)
      firstReg = tile.regList.regValues[0]
      totalRegs = numRegs
      curPool = pool
    else:
      totalRegs += numRegs

  _zeroRegRange(module, writer, tileInfo, firstReg, totalRegs, curPool == writer.agprPool)

  return module


def localReadResetOffsetsSubtile(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based LR offset reset code")
  for i in range(8):
    module.addComment("")

  return module


##################################################
# Subroutine to generate GR offset calculation code
#
def graInitPointer(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for GR base pointer init")
  for i in range(8):
    module.addComment("")

  return module


##################################################
# Compute GR offset for a single matrix (A or B)
#
def _grComputeOffset(module, writer, tileInfo, colId, rowId, output):
  tc = tileInfo.tc
  bpeBits = int(8*tileInfo.bpe)

  tmpVgpr = writer.vgprPool.checkOut(2)
  colBytes = tmpVgpr + 1
  loadWidth = tileInfo.loadWidthGR

  module.add(VLShiftLeftB32(dst=vgpr(colBytes), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colId), comment="scale col_id by load_width"))
  MT0 = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]
  subtileSize = tileInfo.subtileShape[0]*tileInfo.mmaTileShape[0]
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(strideRef), src1=vgpr(rowId), comment="%s: rowId * stride"%tc))
  module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(bpeBits.bit_length()-1), src=vgpr(tmpVgpr), comment="%s: rowId*stride*bpe"%tc))
  module.add(VLShiftRightB32(dst=vgpr(tmpVgpr), shiftHex=hex(3), src=vgpr(tmpVgpr), comment="to bytes"))
  module.add(VAddU32(dst=vgpr(output), src0=vgpr(colBytes), src1=vgpr(tmpVgpr), comment="%s: GR row_offset"%tc))
  writer.vgprPool.checkIn(tmpVgpr)

##################################################
# Compute subtile perpendicular offsets for a single matrix
#
# TODO: need to generalize this to support TLU=1
def _grComputeSubtileOffsets(writer, module, tileInfo):
  tc = tileInfo.tc
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  subtile_size = tileInfo.subtileShape[0]*tileInfo.mmaTileShape[0]
  # rowOffset between 2 subtiles offset, ie how many consecutive subtile covered by a single subtileOffset.
  # rowOffset = numGRPerSubtile * (local load ratio * subtile size)
  rowOffset = math.ceil(tileInfo.numGRPerSubtile*tileInfo.loadRatioGR*subtile_size)
  s_stride = int(rowOffset * tileInfo.bpe)

  for regId in range(len(tileInfo.localSubtilesRegister)):
    regPool = tileInfo.localSubtilesRegister[regId].regPool
    for reg in tileInfo.localSubtilesRegister[regId]:
      if regPool == writer.sgprPool:
        module.add(SMulI32(dst=sgpr(reg), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
      else:
        stmp = writer.sgprPool.checkOut(1)
        idx = tileInfo.localSubtilesRegister[regId].index(reg)
        module.add(SMulI32(dst=sgpr(stmp), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
        module.add(VAddU32(dst=vgpr(reg), src0=vgpr(tileInfo.sharedVgprGROffset[idx]), src1=sgpr(stmp)))
        writer.sgprPool.checkIn(stmp)

# Compute wave partition offset for a single tile (A or B)
#
def _grComputeRowPartition(module, kernel, writer, tileInfo, waveId, rowOffset):
  subIterKBytes = tileInfo.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  loadWidth = tileInfo.loadWidthGR
  numRowsPerWave = wavesize // (subIterKBytes // loadWidth)
  tc = tileInfo.tc
  tmpVgpr = writer.vgprPool.checkOut(2)
  tmpSgpr = writer.sgprPool.checkOut(1, preventOverflow=False)
  localRow = tmpVgpr
  partitionRow = tmpVgpr+1
  partitionOffset = tileInfo.mmaTileShape[0]*tileInfo.localSubtileGrid[0]
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionOffset, comment="%s: row offset"%tc))

  if tileInfo.loadRatioGR == 1.0:
    module.add(VAndB32(dst=vgpr(localRow), src0=hex(1), src1=vgpr(waveId), comment="%s: waveId %% 2"%tc))
    module.add(VLShiftRightB32(dst=vgpr(partitionRow), shiftHex=hex(1), src=vgpr(waveId), comment="%s: waveId / 2"%tc))
  elif tileInfo.loadRatioGR == 0.5:
    module.add(VMovB32(dst=vgpr(localRow), src=0, comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=vgpr(waveId), comment="%s"%tc))
  elif tileInfo.loadRatioGR == 2.0:
    module.add(VMovB32(dst=vgpr(localRow), src=vgpr(waveId), comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=0, comment="%s"%tc))
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))

  module.add(VLShiftLeftB32(dst=vgpr(localRow), shiftHex=hex(numRowsPerWave.bit_length()-1), src=vgpr(localRow), comment="%s: local row offset"%tc))
  module.add(VMulLOU32(dst=vgpr(partitionRow), src0=sgpr(tmpSgpr), src1=vgpr(partitionRow), comment="%s: wave row offset"%tc))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(localRow), src1=vgpr(partitionRow), comment="%s: row offset"%tc))


  writer.vgprPool.checkIn(tmpVgpr)
  writer.sgprPool.checkIn(tmpSgpr)

##################################################
# Compute GR offsets for all subtiles of a single matrix (A or B)
#
def _grComputeAllOffsets(module, writer, tileInfo, colId, rowId, rowOffset):
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(rowId), src1=vgpr(rowOffset), comment="%s: row offset"%tileInfo.tc))
  _grComputeOffset(module, writer, tileInfo, colId, rowOffset, tileInfo.sharedVgprGROffset[0])
  for i in range(1, len(tileInfo.sharedVgprGROffset)):
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    offset = math.ceil(subtileSize * tileInfo.loadRatioGR)
    module.add(VAddU32(dst=vgpr(rowOffset), src0=offset, src1=vgpr(rowOffset), comment="%s: advance row for GR offset %u"%(tileInfo.tc, i)))

    # Apply Rotation on entire wave. Only applies to 4x case as a subtile is loaded by a single wave in 2 steps. (waveId rotation not applied)
    rotatedcolId = writer.vgprPool.checkOut(1)
    loadWidth = tileInfo.loadWidthGR
    if tileInfo.loadRatioGR == 0.5:
      blockSize = tileInfo.subIterKBytes // loadWidth
      module.add(VAddU32(dst=vgpr(rotatedcolId), src0=4, src1=vgpr(colId), comment="%s: advance row for GR offset %u"%(tileInfo.tc, i)))
      module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(blockSize-1), comment="(col + offset) % block_size"))
    else:
      module.add(VMovB32(dst=vgpr(rotatedcolId), src=vgpr(colId), comment=""))

    _grComputeOffset(module, writer, tileInfo, rotatedcolId, rowOffset, tileInfo.sharedVgprGROffset[i])
    writer.vgprPool.checkIn(rotatedcolId)

##################################################
# Apply swizzling and rotation to col IDs for GR offset calculation.
#
# Swizzling reorders column indices to avoid LDS bank conflicts.
# Two levels of rotation are applied to the column IDs:
#   1. Intra-wave rotation: rotates colId based on the LDS row id within
#      a single wave. The rotation offset is: blockSize - (ldsRowId // 2) * 2.
#      This ensures consecutive rows access different LDS banks.
#   2. Inter-wave rotation: an additional per-wave offset derived from waveId
#      shifts the column further so that different waves also avoid bank
#      conflicts with each other. Only applied when loadRatioGR != 0.5
#      (i.e. when multiple waves share the same subtile region).
#
def _grSwizzleColIds(module, writer, tileInfoA, tileInfoB, blockSize, numRowsPerLDSBanks,
                     laneId, colIdA, colIdB, waveId):
  tmpVgpr = writer.vgprPool.checkOut(3)
  ldsRowId = tmpVgpr
  tmp = tmpVgpr + 1
  waveRotation = tmpVgpr + 2

  module.addComment0("Swizzling")
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(ldsRowId), comment="lds row id"))
  module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(ldsRowId), src1=hex(1), comment="lds row id % 2"))
  module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(tmp), comment="lds row id % 2 == 0 ?"))
  module.add(VMovB32(dst=vgpr(colIdA), src=vgpr(colIdA), dpp=DPPModifiers(quad_perm=[1,0,3,2]), comment="swap colId pairs for swizzling"))
  module.add(SMovB64(dst=EXEC(), src=-1))
  module.add(VMovB32(dst=vgpr(colIdB), src=vgpr(colIdA), comment=""))
  module.addComment0("Rotation within a single wave")
  # wave rotation
  module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(ldsRowId), comment=""))
  module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="(ldsRowId //2) * 2"))
  module.add(VSubU32(dst=vgpr(tmp), src0=hex(blockSize), src1=vgpr(tmp), comment="rotation offset : blockSize - (ldsRowId//2)*2"))

  for tInfo, cId in [(tileInfoA, colIdA), (tileInfoB, colIdB)]:
    if tInfo.loadRatioGR != 0.5:
      module.addComment0("Rotation per wave")
      module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(waveId), src1=hex(1), comment=""))
      module.add(VLShiftLeftB32(dst=vgpr(waveRotation), shiftHex=hex((2*numRowsPerLDSBanks).bit_length() - 1), src=vgpr(waveRotation), comment=""))
      module.add(VSubU32(dst=vgpr(waveRotation), src0=vgpr(tmp), src1=vgpr(waveRotation), comment=""))
      module.add(VAddU32(dst=vgpr(cId), src0=vgpr(waveRotation), src1=vgpr(cId), comment=""))
    else:
      module.add(VAddU32(dst=vgpr(cId), src0=vgpr(tmp), src1=vgpr(cId), comment=""))

  module.add(VAndB32(dst=vgpr(colIdA), src0=vgpr(colIdA), src1=hex(blockSize-1), comment="(col + offset) % block_size"))
  module.add(VAndB32(dst=vgpr(colIdB), src0=vgpr(colIdB), src1=hex(blockSize-1), comment="(col + offset) % block_size"))

  writer.vgprPool.checkIn(tmpVgpr)

##################################################
# Subroutine to generate GR offset calculation code
#
def graTileAssignment(writer, kernel, useSwizzling=True):
  module = Module()
  module.addComment0("GR Offset Calculation for Subtile Based Tiling")

  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo

  # Input Parameters.
  subIterKBytes = tileInfoA.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  ldsRowBankSize = 64 * 4 # 64 banks, 4 bytes per bank.

  loadWidth = tileInfoA.loadWidthGR # Assumes loadwidth for A/B tiles are the same
  assert subIterKBytes % loadWidth == 0, "subIterKBytes (%u) must be a multiple of loadWidth (%u)" % (subIterKBytes, loadWidth)
  assert subIterKBytes <= ldsRowBankSize, "Only support subIterKBytes smaller than %u (lds row bank size) for now"%ldsRowBankSize
  blockSize = subIterKBytes // loadWidth

  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes

  tmpVgpr = writer.vgprPool.checkOut(7)
  colIdA = tmpVgpr
  colIdB = tmpVgpr + 1
  rowId = tmpVgpr + 2
  rowOffsetA = tmpVgpr + 3
  rowOffsetB = tmpVgpr + 4
  waveId = tmpVgpr + 5
  laneId = tmpVgpr + 6

  # Compute waveId and laneId
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=wavesize-1, comment=""))
  # Common code for both A & B
  # Calculate col and row id within a wave for 128b loads
  module.add(VAndB32(dst=vgpr(colIdA), src0=vgpr("Serial"), src1=(blockSize-1), comment="get col_id in wave for %uB load"%loadWidth))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))

  # Apply swizzling and rotation to colId for A and B
  _grSwizzleColIds(module, writer, tileInfoA, tileInfoB, blockSize, numRowsPerLDSBanks,
                   laneId, colIdA, colIdB, waveId)

  # Compute rowOffsetA and rowOffsetB row offset based on wave partitioning (e.g. 2x2, 4x1/1x4)
  _grComputeRowPartition(module, kernel, writer, tileInfoA, waveId, rowOffsetA)
  _grComputeRowPartition(module, kernel, writer, tileInfoB, waveId, rowOffsetB)

  # Compute GR offset for A and B
  _grComputeAllOffsets(module, writer, tileInfoA, colIdA, rowId, rowOffsetA)
  _grComputeAllOffsets(module, writer, tileInfoB, colIdB, rowId, rowOffsetB)

  writer.vgprPool.checkIn(tmpVgpr)

  # Compute subtile offsets for A and B
  _grComputeSubtileOffsets(writer, module, tileInfoA)
  _grComputeSubtileOffsets(writer, module, tileInfoB)

  return module

def _getScaleTileInfo(tc, writer, kernel):
  """Get MXSA/MXSB tileInfo for matrix tc, or None if MX scaling is inactive."""
  key = "MXBlock%s" % tc
  if not kernel["ProblemType"].get(key, 0):
    return None
  return (writer.states.mxsa.tileInfo if tc == 'A' else writer.states.mxsb.tileInfo)

##################################################
# Compute the per-thread global-read (DTL) vaddr for scale tensor tc.
#
# With DTL (buffer_load lds=True) the same vaddr serves as:
#   - global byte offset from the SRD base  (where to read from global memory)
#   - LDS byte offset from M0               (where to write in LDS)
#
# Threads within a wave are split into groups of numThreadsPerGroup.
# Each group loads one contiguous subtile-column worth of scale bytes:
#
#   groupId  = serial / numThreadsPerGroup          (which scale column)
#   threadId = serial % numThreadsPerGroup           (position within group)
#
#   grOffset = groupId  * stride_bpe                (column byte offset via tensor stride)
#            + threadId * loadWidth                  (byte offset within column)
#
# Output: sharedVgprGROffset[0] = grOffset (used as vaddr in DTL load)
#
def _graTileAssignmentScaleSwizzledCommon(tc, writer, kernel):
  module = Module()

  module.addComment("Computing GR Offset for %s"%tc)

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  loadWidth = tileInfo.loadWidthGR
  loadWidthShift = loadWidth.bit_length() - 1

  # TODO: this logic assumes scales are in block TLU=0 format.
  # Scale groups span 2 M-adjacent subtiles (matching the physical 32-row scale blocks),
  # so multiply subtileSize by 2 to get the actual bytes per group.
  scaleGroupSize = 2 * tileInfo.subtileSize # bytes per scale group (2 subtiles in dim0)
  # number of consecutive threads needed to load all subtiles in contiguous dim
  numThreadsPerGroup = (scaleGroupSize * tileInfo.localSubtileGrid[1]) // loadWidth

  vtmp = writer.vgprPool.checkOut(1)

  stmp = writer.sgprPool.checkOut(1)

  module.add(VLShiftRightB32(dst=vgpr(vtmp),
                            shiftHex=hex(int(math.log2(numThreadsPerGroup))), src=vgpr("Serial"),
                            comment="%s: grOffset = serial / %d" % (tc, loadWidth)))
  module.add(SLShiftLeftB32(sgpr(stmp), int(math.log2(tileInfo.bpe)), sgpr("Strides%s"%tc), comment="*= bpe (%d)"%(tileInfo.bpe)))

  module.add(VMulLOU32(dst=vgpr(vtmp), src1=vgpr(vtmp), src0=sgpr(stmp), comment="Apply scale%s stride to each group"%tc))
  module.add(VAndB32(dst=vgpr(tileInfo.sharedVgprGROffset[0]),
                     src0=hex(numThreadsPerGroup - 1), src1=vgpr("Serial"),
                     comment="%s: grOffset = serial %% %d" % (tc, loadWidth)))
  module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprGROffset[0]),
                            shiftHex=hex(loadWidthShift), src=vgpr(tileInfo.sharedVgprGROffset[0]),
                            comment="Scale by load width for each thread in group"))
  module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprGROffset[0]), src0=vgpr(tileInfo.sharedVgprGROffset[0]), src1=vgpr(vtmp), comment="Final offset calc"))
  writer.vgprPool.checkIn(vtmp)
  writer.sgprPool.checkIn(stmp)

  return module

##################################################
# Generate GR offset calculation for scaleA/B (DTL).
#
# With DTL, vaddr serves as both the global read offset (from SRD)
# and the LDS write offset (from M0). Simple linear access:
#   grOffset = serial * scaleLoadWidth
#
def graTileAssignmentScaleSwizzled(writer, kernel):
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    module.addComment0("Scale GR tile assignment: skipped (no MX block scaling)")
    return module

  # DTL linear offset: vaddr = serial * scaleLoadWidth (= serial << log2(loadWidth))
  module.add(_graTileAssignmentScaleSwizzledCommon('MXSA', writer, kernel))
  module.add(_graTileAssignmentScaleSwizzledCommon('MXSB', writer, kernel))

  return module


##################################################
# Apply wave partition offset for scale LR.
#
# Each wave reads from its assigned LDS partition for scale A or B.
#
#   MXSA: partition index = waveId % MIWaveGroup[0]  (M-direction wave index)
#   MXSB: partition index = waveId / MIWaveGroup[0]  (N-direction wave index)
#         Using MIWaveGroup[0] (not [1]) correctly handles asymmetric configs
#         (e.g. 4x1: all 4 M-waves share the same N partition → index = 0).
#
# Output: sharedVgprLROffset[0] = partitionIndex * totalScaleBytes
#
def _applyScaleWavePartitionLROffset(module, writer, kernel, tileInfo, waveId):
  tc = tileInfo.tc

  # Partition stride is based on actual scale data size, not GR load capacity.
  # Mirrors data tile partition (_applyWavePartitionLROffset) which uses
  # MT * depthUBytes // numPartitions.
  # TODO: Calculate num of rows in subtile instead of hardcoding
  scaleSubtileBytes = tileInfo.subtileSize * tileInfo.bpe
  # Note MMATile format is always [NonK dim, K dim]
  MT = tileInfo.globalMMATileGrid[0] // tileInfo.subtileShape[0]
  index = 0 if tc == 'MXSA' else 1
  totalScaleBytes = (MT // kernel["MIWaveGroup"][index]) * (tileInfo.localSubtileGrid[1]) * scaleSubtileBytes

  tmpSgpr = writer.sgprPool.checkOut(1)
  tmp = writer.vgprPool.checkOut(2)

  if tc == 'MXSA':
    module.add(VAndB32(dst=vgpr(tmp), src0=kernel["MIWaveGroup"][0]-1, src1=vgpr(waveId), comment="scale%s: waveId %% 2"%tc))
  else:
    # N-direction wave index = waveId / numWavesInM (MIWaveGroup[0])
    # Using MIWaveGroup[0] (not [1]) correctly handles asymmetric configs like 4x1
    # where log2(MIWaveGroup[1])=0 would give waveId unchanged instead of waveId/4.
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=int(math.log2(kernel["MIWaveGroup"][0])), src=vgpr(waveId), comment="scale%s: waveId / numWavesM"%tc))

  module.add(SMovB32(dst=sgpr(tmpSgpr), src=totalScaleBytes, comment="scale%s: scale region"%tc))
  module.add(VMulLOU32(dst=vgpr(tileInfo.sharedVgprLROffset[0]), src0=sgpr(tmpSgpr), src1=vgpr(tmp), comment="scale%s: partition offset"%tc))

  writer.vgprPool.checkIn(tmp)
  writer.sgprPool.checkIn(tmpSgpr)


##################################################
# Generate LR offset calculation for scaleA/B.
#
# Computes the per-lane LDS read offset for scale tensors. Called once
# during kernel setup; the resulting VGPRs are used every loop iteration.
#
# Final LR offset per lane:
#   lrOffset[lane] = wavePartitionOffset + laneId * 4 + ldsStartOffset
#
# where:
#   wavePartitionOffset  = partitionIndex * totalScaleBytes
#     MXSA partitionIndex = waveId % MIWaveGroup[0]   (M-direction)
#     MXSB partitionIndex = waveId / MIWaveGroup[0]   (N-direction)
#   laneId               = serial & (wavesize - 1)
#   ldsStartOffset       = writer.ldsStartOffsetMXSA/B
#
# LDS layout (double-buffered, one buffer shown):
#   [ DataA | DataB | ScaleA | ScaleB ]
#   ScaleA starts at ldsStartOffsetMXSA, ScaleB at ldsStartOffsetMXSB.
#
# After the LR offset is fully computed, the double-buffer swap VGPR is
# initialised here (not in localReadDTLInitCommonSwapVgpr, which runs
# before this function and would use uninitialised values):
#   swapVgpr = lrOffset XOR (lrOffset + ldsTotalSize)
# This lets localReadLDSBufferSwap toggle between buffer 0 and buffer 1.
#
def lraTileAssignmentScaleSwizzled(writer, kernel):
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    module.addComment0("Scale LR tile assignment: skipped (no MX block scaling)")
    return module

  mxsaTileInfo = writer.states.mxsa.tileInfo
  mxsbTileInfo = writer.states.mxsb.tileInfo

  module.addComment0("LR Offset Calculation for Scale Tensors")

  wavesize = kernel["WavefrontSize"]

  # Wave partitioning
  waveIdVgpr = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="scale: waveId"))

  _applyScaleWavePartitionLROffset(module, writer, kernel, mxsaTileInfo, waveIdVgpr)
  _applyScaleWavePartitionLROffset(module, writer, kernel, mxsbTileInfo, waveIdVgpr)
  writer.vgprPool.checkIn(waveIdVgpr)

  # Per-lane offset: laneId * sizeof(dword) = (serial & (wavesize-1)) << 2
  laneOffset = writer.vgprPool.checkOut(1)
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1, comment="scale: laneId"))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset), comment="scale: laneId * 4"))

  module.add(VAddU32(dst=vgpr(mxsaTileInfo.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(mxsaTileInfo.sharedVgprLROffset[0]), comment="scaleA: lrOffset = laneId * 4"))
  module.add(VAddU32(dst=vgpr(mxsbTileInfo.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(mxsbTileInfo.sharedVgprLROffset[0]), comment="scaleB: lrOffset = laneId * 4"))
  writer.vgprPool.checkIn(laneOffset)


  # Apply global LDS offset for A scale (scale A follows data A+B in LDS)
  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSA), comment="scale: LDS offset for A scale"))
  module.add(VAddU32(dst=vgpr(mxsaTileInfo.sharedVgprLROffset[0]), src0=vgpr(mxsaTileInfo.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleA: +=LDS offset"))

  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSB), comment="scale: LDS offset for B scale"))
  module.add(VAddU32(dst=vgpr(mxsbTileInfo.sharedVgprLROffset[0]), src0=vgpr(mxsbTileInfo.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleB: +=LDS offset"))

  # Init scale LR swap VGPRs here, after the LR offsets are fully computed.
  # (Must NOT be done in localReadDTLInitCommonSwapVgpr, which runs before this function.)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=writer.ldsTotalSize, comment="scale: total LDS size for swap"))
  for tileInfo in [mxsaTileInfo, mxsbTileInfo]:
    for i in range(len(tileInfo.sharedVgprLROffset)):
      vgprId     = tileInfo.sharedVgprLROffset[i]
      vgprSwapId = tileInfo.sharedVgprLROffsetSwap[i]
      module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(tmpSgpr), comment="scale%s: LR swap"%tileInfo.tc))
      module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId),  comment="scale%s: LR swap"%tileInfo.tc))

  writer.sgprPool.checkIn(tmpSgpr)

  return module

##################################################
# Scale GR: Load scale bytes from global memory directly to LDS (DTL).
#
# Uses BufferLoadB128 with lds=True. M0 is set to scaleLdsBase, and
# sharedVgprGROffset[0] = serial * scaleLoadWidth serves as both the
# global read offset (from SRD) and the LDS write offset (from M0).
def globalReadDoScaleSubtile(tc, writer, kernel):
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  assert len(tileInfo.sharedVgprGROffset) > 0, "Scale GR requires at least 1 GR offset VGPR"

  module.addComment0("Scale GR: %s (DTL: BufferLoadB128 -> LDS)" % tc)

  # Set M0 to scale LDS base address for DTL write destination
  module.add(SMovB32(dst=mgpr(0), src=sgpr("LocalWriteBaseAddr%s"%tc),
                     comment="scale%s: M0 = scaleLdsBase" % tc))

  # DTL load: data goes directly from global memory to LDS (no intermediate VGPR)
  mubuf = MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT, lds=True)
  module.add(BufferLoadB128(dst=None, vaddr=vgpr(tileInfo.sharedVgprGROffset[0]),
                            saddr=sgpr("Srd%s" % tc, 4), soffset=0, mubuf=mubuf,
                            comment="scale%s: DTL b128 load" % tc))

  return module

##################################################
# Scale LR: Read scale data from LDS into scale VGPRs (DSLoadB32).
#
# Each lane reads 4 bytes from LDS using ds_read_b32. The base address
# is sharedVgprLROffset[0] (computed by lraTileAssignmentScaleSwizzled).
# MMA tile and subtile selection is done via constant ds_offset at emit time.
#
# Each 32-bit VGPR holds 4 E8M0 scale bytes; opsel/opsel_hi selects
# the correct byte per MFMA invocation.
#
def emitSubtileScaleDsRead(tc, writer, kernel, scaleGroupIdx):
  """Emit a single DSLoadB32 for a scale group (2 M-adjacent [1,2] subtiles).
  Each ds_read_b32 loads 4 bytes = 4 E8M0 scale values into one VGPR."""
  module = Module()
  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  if tileInfo.mxBlock == 0:
    return module

  # Each scale group covers 2 M-adjacent subtiles, stride = 2 * subtileSize
  groupStride = 2 * tileInfo.subtileSize
  dsOffset = groupStride * scaleGroupIdx
  vdst = tileInfo.vgprTiles[4 * scaleGroupIdx].regList.regValues[0]
  module.add(DSLoadB32(dst=vgpr(vdst),
                       src=vgpr(tileInfo.sharedVgprLROffset[0]),
                       ds=DSModifiers(offset=dsOffset),
                       comment="scale%s[group%u]: load 4B from LDS" % (tc, scaleGroupIdx)))
  return module

def localReadDoScaleSubtile(tc, writer, kernel):
  """Emit scale ds_reads for all scale groups (PGR=0 path)."""
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  # Iterate over scale groups: one ds_read per 2 M-adjacent subtiles
  numScaleGroups = math.ceil(tileInfo.localSubtileGrid[0] / 2) * tileInfo.localSubtileGrid[1]
  for gid in range(numScaleGroups):
    module.add(emitSubtileScaleDsRead(tc, writer, kernel, gid))

  return module

##################################################
# Scale SRD pointer update: advance scale SRD by scaleDepthU * scaleBpe bytes.
#
def globalReadScalePtrUpdates(tc, writer, kernel):
  module = Module()
  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  if tileInfo.mxBlock == 0:
    return module

  inc = 2 * tileInfo.subtileSize * tileInfo.localSubtileGrid[1]
  module.addComment0("Scale SRD update: %s += %u" % (tc, inc))
  module.add(SAddU32(dst=sgpr("Srd%s" % tc), src0=sgpr("Srd%s" % tc), src1=inc))
  module.add(SAddCU32(dst=sgpr("Srd%s+1" % tc), src0=sgpr("Srd%s+1" % tc), src1=0))

  # No need to decrement Srd+2
  # We use fixed value for Srd+2

  return module

##################################################
# Subroutine to generate GR load code
#
def emitSingleBufferLoad(tileInfo, kernel, sId0, sId1):
  """Emit buffer_load instructions for a single subtile (sId0, sId1).

  When loadRatioGR > 1, multiple local subtiles share the same global read.
  Only the first subtile in each group emits the load; others return empty.

  Args:
      tileInfo: TileInfo for the tensor component
      sId0:     Subtile row index
      sId1:     Subtile column index (K-dimension)
  """
  module = Module()

  subtileInfo = tileInfo.localSubtiles[tileInfo.getLocalSubtileLinearId(sId0, sId1)]
  grBaseId = subtileInfo.globalReadMap[0]

  # TODO: Still needed for PGR=0 path but not needed by scheduler
  # When loadRatioGR > 1, multiple subtiles share one global read.
  # Only emit the load for the first subtile of each group.
  if tileInfo.loadRatioGR > 1:
    linearId = tileInfo.getLocalSubtileLinearId(sId0, sId1)
    firstInGroup = int(grBaseId * tileInfo.loadRatioGR)
    if linearId != firstInGroup:
      return module

  tc = tileInfo.tc
  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  regList = tileInfo.localSubtilesRegister[subtileInfo.regListId]

  offsetK = sId1 * int(tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1] * tileInfo.bpe)

  subtileOffset = math.ceil(tileInfo.loadRatioGR*tileInfo.subtileSize)
  WriteBaseAddr = "LocalWriteBaseAddr%s"%tc
  # Emit number of buffer loads equal to number of loads needed to load a subtile
  for i in range(tileInfo.numGRPerSubtile):
    m0Offset = i * subtileOffset + (sId0 + sId1 * tileInfo.globalSubtileGrid[0]) * tileInfo.subtileSize
    module.add(SAddU32(dst=mgpr(0), src0=sgpr(WriteBaseAddr), src1=(m0Offset - offsetK)))
    mubuf = MUBUFModifiers(offen=True, offset12=offsetK, glc=isGlc, slc=isSlc, nt=isNT, lds=True)

    # Check if the subtile specific registers is SGPR or VGPR
    # For SGPR we can keep the same shared vgpr offset and use the soffset field for the subtile specific SGPR
    # For VGPR we need to update the apply the subtile-specific constant offset to the VGPR
    #   the shared VGPR offset is not used for that specific tile, soffset is also set to zero.
    useSgpr = subtileInfo.useSgpr
    soffset = sgpr(regList.regValues[0]) if len(regList) > 0 and useSgpr else 0
    voff = tileInfo.sharedVgprGROffset[i] if useSgpr or len(regList) == 0 else regList.regValues[i]
    module.add(BufferLoadB128(dst=None, vaddr=vgpr(voff), saddr=sgpr("Srd%s"%tc, 4), soffset=soffset, mubuf=mubuf, comment="grBaseId = %u, i= %u"%(grBaseId , i)))

  return module


def emitSubtileBufferLoad(tc, writer, kernel, subtileId):
  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return emitSingleBufferLoad(tileInfo, kernel, subtileId[0], subtileId[1])

##################################################
# Subroutine to generate GR load code
# Initial idea: maybe store asm in modules in a separate obj?
#
def globalReadDoSubtile(tc, writer, kernel):
  module = Module()

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo

  for j in range(tileInfo.localSubtileGrid[1]):
    for i in range(tileInfo.localSubtileGrid[0]):
      module.addComment0("Emit load for %s subtile: [%u, %u]"%(tc, i, j))
      module.add(emitSubtileBufferLoad(tc, writer, kernel, [i, j]))

  return module

def emitSingleDsRead(tileInfo, sId0, sId1, subIterK, dstTile):
  """Emit a single DSLoadB128 for one MMA tile within a subtile.

  Args:
      tileInfo:  TileInfo (for subtileSize, loadRatioGR, sharedVgprLROffset, tc)
      sId0:      Subtile row index (used for offset computation)
      subIterK:  subIterK index within the subtile (maps to mfmaC; subtileShape[0]=1 so mfmaR=0)
      dstTile:   RegisterTileInfo — destination vgpr tile for the load
  """

  # du maps to mfmaC, mfmaR is always 0 (subtileShape[0]=1)
  mfmaId = tileInfo.getSubtileShapeLinearId(subIterK, 0)
  addrVgpr = tileInfo.sharedVgprLROffset[mfmaId]

  offsetStride = tileInfo.subtileSize
  offset = sId0*offsetStride

  offset = offset + sId1 * tileInfo.globalSubtileGrid[0] * offsetStride

  dstVgpr = dstTile.regList.regValues[0]
  numRegs = len(dstTile.regList.regValues)
  return DSLoadB128(
      dst=vgpr(dstVgpr, numRegs),
      src=vgpr(addrVgpr),
      ds=DSModifiers(offset=offset),
      comment="Subtile%s[%u, %u] subIterK=%u" % (tileInfo.tc, sId0, sId1, subIterK))


def emitSubtileDsRead(writer, kernel, tileInfo, subtileId):

  module = Module()
  sId0 = subtileId[0]
  sId1 = subtileId[1]

  linearId = tileInfo.getLocalSubtileLinearId(sId0, sId1)
  subtileInfo = tileInfo.localSubtiles[linearId]

  for du in range(tileInfo.subtileShape[1]):
    mfmaId = tileInfo.getSubtileShapeLinearId(du, 0)
    dstTile = tileInfo.vgprTiles[subtileInfo.localReadMap[mfmaId]]
    module.add(emitSingleDsRead(tileInfo, sId0, sId1, du, dstTile))

  return module

##################################################
# Subroutine to generate LR load code
# Initial idea: maybe store asm in modules in a separate obj?
#
def localReadDoSubtile(tc, writer, kernel):
  module = Module()

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo

  for i in range(tileInfo.localSubtileGrid[0]):
    for j in range(tileInfo.localSubtileGrid[1]):
        module.add(emitSubtileDsRead(writer, kernel, tileInfo, [i, j]))

  return module

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def globalReadDTLInitCommonSgpr(writer, kernel):
  module = Module()

  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1)
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))

  atile = writer.states.a.tileInfo
  btile = writer.states.b.tileInfo

  tmpVgpr = writer.vgprPool.checkOut(2)
  rowOffsetA = tmpVgpr
  rowOffsetB = tmpVgpr + 1

  _grComputeRowPartition(module, kernel, writer, atile, vgprWaveId, rowOffsetA)
  _grComputeRowPartition(module, kernel, writer, btile, vgprWaveId, rowOffsetB)

  subIterKBytes = atile.subIterKBytes

  module.add(VLShiftLeftB32(dst=vgpr(rowOffsetA), shiftHex=hex((subIterKBytes).bit_length()-1), src=vgpr(rowOffsetA), comment="Apply wave-specific offset for A"))
  module.add(VLShiftLeftB32(dst=vgpr(rowOffsetB), shiftHex=hex((subIterKBytes).bit_length()-1), src=vgpr(rowOffsetB), comment="Apply wave-specific offset for B"))

  module.add(SNop(waitState=0, comment="Wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrA"), src=vgpr(rowOffsetA), comment="Store base LDS offset, will be modified"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrB"), src=vgpr(rowOffsetB), comment="Store base LDS offset, will be modified"))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrB"), src0=sgpr("LocalWriteBaseAddrB"), src1=hex(writer.ldsStartOffsetB), comment=""))

  module.add(SAddU32(dst=sgpr("SwapA"), src0=sgpr("LocalWriteBaseAddrA"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapA"), src0=sgpr("LocalWriteBaseAddrA"), src1=sgpr("SwapA"), comment=""))
  module.add(SAddU32(dst=sgpr("SwapB"), src0=sgpr("LocalWriteBaseAddrB"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapB"), src0=sgpr("LocalWriteBaseAddrB"), src1=sgpr("SwapB"), comment=""))

  writer.vgprPool.checkIn(vgprWaveId)
  writer.vgprPool.checkIn(tmpVgpr)


  return module

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
# For Swizzled Scales each wave will collectively stream
# the scale values
#
def globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel):
  module = Module()


  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1)
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))

  mxsatile = writer.states.mxsa.tileInfo
  mxsbtile = writer.states.mxsb.tileInfo

  loadWidth = mxsatile.loadWidthGR # Assumes load width for scaleA/B are the same

  bytesPerLoad = loadWidth * wavesize
  module.add(VLShiftLeftB32(dst=vgpr(vgprWaveId), shiftHex=hex((bytesPerLoad).bit_length()-1), src=vgpr(vgprWaveId), comment="Apply wave-specific common offset (%u) for A/B"%bytesPerLoad))

  module.add(SNop(waitState=0, comment="Wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrMXSA"), src=vgpr(vgprWaveId), comment="Store base LDS offset, will be modified"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrMXSB"), src=vgpr(vgprWaveId), comment="Store base LDS offset, will be modified"))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=hex(writer.ldsStartOffsetMXSA), comment=""))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=hex(writer.ldsStartOffsetMXSB), comment=""))

  module.add(SAddU32(dst=sgpr("SwapMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=sgpr("SwapMXSA"), comment=""))
  module.add(SAddU32(dst=sgpr("SwapMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=sgpr("SwapMXSB"), comment=""))

  writer.vgprPool.checkIn(vgprWaveId)
  return module




def localReadDTLInitCommonSwapVgpr(writer, kernel):
  module = Module()

  atile = writer.states.a.tileInfo
  btile = writer.states.b.tileInfo

  stmp = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsTotalSize, comment="Store Total Lds Size for one buffer"))
  for i in range(len(atile.sharedVgprLROffset)):
    vgprId = atile.sharedVgprLROffset[i]
    vgprSwapId = atile.sharedVgprLROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(stmp), comment=""))
    module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment=""))

  for i in range(len(btile.sharedVgprLROffset)):
    vgprId = btile.sharedVgprLROffset[i]
    vgprSwapId = btile.sharedVgprLROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(stmp), comment=""))
    module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment=""))

  writer.sgprPool.checkIn(stmp)
  return module



##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def globalReadLDSBufferSwap(tc, writer, kernel):
  module = Module()
  module.addComment0("Emit code to swap %s GR m0 offsets"%tc)
  module.add(SXorB32(dst=sgpr("LocalWriteBaseAddr%s"%tc), src0=sgpr("LocalWriteBaseAddr%s"%tc), src1=sgpr("Swap%s"%tc), comment=""))
  return module

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def localReadLDSBufferSwap(tc, writer, kernel):
  module = Module()

  if tc in ['A', 'B']:
    tile = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  else:
    tile = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  module.addComment0("Emit code to swap %s LR vgpr offsets"%tc)

  for i in range(len(tile.sharedVgprLROffset)):
    vgprId = tile.sharedVgprLROffset[i]
    vgprSwapId = tile.sharedVgprLROffsetSwap[i]
    module.add(VXorB32(dst=vgpr(vgprId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment=""))

  return module

##################################################
# Subroutine to update ptrs
#
def globalReadPtrUpdates(tc, writer, kernel):
  module = Module()
  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  inc = int(tileInfo.localSubtileGrid[1] * tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1] * tileInfo.bpe)
  module.add(SAddU32(dst=sgpr("Srd%s"%tc), src0=sgpr("Srd%s"%tc), src1=inc))
  module.add(SAddCU32(dst=sgpr("Srd%s+1"%tc), src0=sgpr("Srd%s+1"%tc), src1=0))

  return module

##################################################
# Subroutine to generate MMA Instruction
# Given RegisterTileInfo inputs for A,B,C,D operands
# emit corresponding mfma instruction
#
def emitMfmaInstruction(writer, kernel, vgprTileA, vgprTileB, vgprTileC, vgprTileD, scaleAVgpr=-1, scaleBVgpr=-1, scaleAsel=-1, scaleBsel=-1, comment = ""):
  module = Module()

  vgprAStart = vgprTileA.regList.regValues[0]
  vgprBStart = vgprTileB.regList.regValues[0]
  vgprCStart = vgprTileC.regList.regValues[0]
  vgprDStart = vgprTileD.regList.regValues[0]

  opASize = len(vgprTileA.regList.regValues)
  opBSize = len(vgprTileB.regList.regValues)
  opCSize = len(vgprTileC.regList.regValues)
  opDSize = len(vgprTileD.regList.regValues)

  # For subtile kernels with agpr overflow, D/C tiles that spilled to the vgpr
  # pool must use vgpr() in the MFMA operands, not accvgpr().
  dIsVgpr = (vgprTileD.regList.regPool == writer.vgprPool)
  cIsVgpr = (vgprTileC.regList.regPool == writer.vgprPool)
  dAccAlias = vgpr if (dIsVgpr or kernel["MIArchVgpr"]) else accvgpr
  cAccAlias = vgpr if (cIsVgpr or kernel["MIArchVgpr"]) else accvgpr

  aOperand = vgpr(vgprBStart,opBSize) if kernel["SourceSwap"] else vgpr(vgprAStart,opASize)
  bOperand = vgpr(vgprAStart,opASize) if kernel["SourceSwap"] else vgpr(vgprBStart,opBSize)

  miK = kernel["MatrixInstK"]

  if miK == 128:
    # MX FP4: 16x16x128
    if scaleAVgpr >= 0 and scaleBVgpr >= 0:
      # Use actual loaded scale VGPRs
      module.add(MXMFMAInstruction(instType=InstType.INST_F4, accType=InstType.INST_F32, variant=[16,16,miK,1], \
                                   acc=dAccAlias(vgprDStart,opDSize), \
                                   a=aOperand, \
                                   b=bOperand, \
                                   acc2=cAccAlias(vgprCStart,opCSize), \
                                   mxsa=vgpr(scaleAVgpr), mxsb=vgpr(scaleBVgpr), \
                                   vop3=VOP3PModifiers(op_sel=[scaleAsel%2, scaleBsel%2], op_sel_hi=[(scaleAsel>>1)%2, (scaleBsel>>1)%2]), \
                                   comment=comment))
    else:
      # Fallback: hardcoded scale 0x7f (scale=1.0 for all elements)
      tmpVgprScale = writer.vgprPool.checkOut(1)
      module.add(VMovB32(dst=vgpr(tmpVgprScale), src=hex(0x7f7f7f7f), comment="hardcoded scale 0x7f (E8M0)"))
      module.add(MXMFMAInstruction(instType=InstType.INST_F4, accType=InstType.INST_F32, variant=[16,16,miK,1], \
                                   acc=dAccAlias(vgprDStart,opDSize), \
                                   a=aOperand, \
                                   b=bOperand, \
                                   acc2=cAccAlias(vgprCStart,opCSize), \
                                   mxsa=vgpr(tmpVgprScale), mxsb=vgpr(tmpVgprScale), \
                                   comment=comment))
      writer.vgprPool.checkIn(tmpVgprScale)
  else:
    # BF16: 16x16x32
    module.add(MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16,16,miK,1], mfma1k=False, \
                               acc=dAccAlias(vgprDStart,opDSize), \
                               a=aOperand, \
                               b=bOperand, \
                               acc2=cAccAlias(vgprCStart,opCSize), \
                               comment=comment))

  return module


##################################################
# Subroutine to generate MMA code
# Initial idea: maybe store asm in modules in a separate obj?
#
def emitMfmaCode(writer, kernel):
  module = Module()

  atileInfo = writer.states.a.tileInfo
  btileInfo = writer.states.b.tileInfo
  dtileInfo = writer.states.d.tileInfo

  mxsatileInfo = writer.states.mxsa.tileInfo if kernel["ProblemType"].get("MXBlockA", 0) > 0 else None
  mxsbtileInfo = writer.states.mxsb.tileInfo if kernel["ProblemType"].get("MXBlockB", 0) > 0 else None

  # Use loaded scale VGPRs when MX block scaling is active.
  # Note: scaleVgprTiles is only populated by the scheduler path;
  # in the non-scheduler path we use vgprTiles (populated by localReadDoScaleSubtile).
  hasScaleA = mxsatileInfo is not None and mxsatileInfo.mxBlock > 0
  hasScaleB = mxsbtileInfo is not None and mxsbtileInfo.mxBlock > 0

  for mmak in range(atileInfo.localMMATileGrid[1]):
    for mma1 in range(btileInfo.localMMATileGrid[0]):
      for mma0 in range(atileInfo.localMMATileGrid[0]):

        aSId0, aSId1 = atileInfo.getLocalSubtileIdFromMMATile(mma0, mmak)
        bSId0, bSId1 = btileInfo.getLocalSubtileIdFromMMATile(mma1, mmak)
        _mma0 = mma0 % atileInfo.subtileShape[0]
        _mma1 = mma1 % btileInfo.subtileShape[0]
        _mmak = mmak % atileInfo.subtileShape[1]

        numMmaTilePerSubtileA = atileInfo.subtileShape[0] * atileInfo.subtileShape[1]
        numMmaTilePerSubtileB = btileInfo.subtileShape[0] * btileInfo.subtileShape[1]

        # TODO: Fix mma index calc for larger subtile shapes
        atileId = atileInfo.getLocalSubtileLinearId(aSId0, aSId1) * numMmaTilePerSubtileA + (_mmak)
        btileId = btileInfo.getLocalSubtileLinearId(bSId0, bSId1) * numMmaTilePerSubtileB + (_mmak)

        atiles = atileInfo.vgprTiles[atileId]
        btiles = btileInfo.vgprTiles[btileId]
        dtiles = dtileInfo.vgprTiles[mma0 + mma1 * dtileInfo.localMMATileGrid[0]]


        if hasScaleA:
          mxsatileInfo = writer.states.mxsa.tileInfo
          mxsbtileInfo = writer.states.mxsb.tileInfo
          # Scale group index: one VGPR per 2 M-adjacent subtiles (ds_read_b32 loads 4 bytes)
          subtileKShape = atileInfo.subtileShape[1]
          subtileKGrid = atileInfo.localSubtileGrid[1]
          scaleGroupA = (mma0 // 2) * subtileKGrid + mmak // subtileKShape
          scaleGroupB = (mma1 // 2) * subtileKGrid + mmak // subtileKShape

          scaleAVgpr = mxsatileInfo.vgprTiles[4 * scaleGroupA].regList.regValues[0] if mxsatileInfo.mxBlock else -1
          scaleBVgpr = mxsbtileInfo.vgprTiles[4 * scaleGroupB].regList.regValues[0] if mxsbtileInfo.mxBlock else -1

          sAsel = (mma0 % 2) + 2 * (mmak % 2)
          sBsel = (mma1 % 2) + 2 * (mmak % 2)
        else:
          scaleAVgpr = -1
          scaleBVgpr = -1
          sAsel = sBsel = -1

        module.add(emitMfmaInstruction(writer, kernel, atiles, btiles, dtiles, dtiles,
                                       scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr, scaleAsel=sAsel, scaleBsel=sBsel,
                                       comment="Emit MMFA code for MMA tiles C[%u, %u] += A[%u, %u] * B[%u, %u] sA = %u, sB = %u"%(mma0, mma1, mma0, mmak, mmak, mma1, sAsel, sBsel)))

  return module


##################################################
# Subroutine entry point for main loop impl
#
# This should be shared logic for both main loop and nnl loops
# It would be nice to have this support generic loop unroll
# and possibly SIMD spec paths
#
# Scheduling logic would be introduced here
#
def mainLoopImplPGR0(writer, kernel, isNLL = False):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based main loop impl")

  hasMXScale = kernel["ProblemType"].get("MXBlockA", 0) and kernel["ProblemType"].get("MXBlockB", 0)

  label = Label("start", comment="")
  module.add(label)

  if not isNLL:
    module.add(globalReadDoSubtile('A', writer, kernel))
    module.add(globalReadDoSubtile('B', writer, kernel))
    if hasMXScale:
      # Scale GR: load scale data from global to LDS (non-DTL)
      module.add(globalReadDoScaleSubtile('MXSA', writer, kernel))
      module.add(globalReadDoScaleSubtile('MXSB', writer, kernel))
    module.add(SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1, comment="Wait for all subtile GRs to complete"))
    module.add(SBarrier(comment=""))

  module.add(localReadDoSubtile('A', writer, kernel))
  module.add(localReadDoSubtile('B', writer, kernel))
  if hasMXScale:
    # Scale LR: load scale data from LDS to VGPRs
    module.add(localReadDoScaleSubtile('MXSA', writer, kernel))
    module.add(localReadDoScaleSubtile('MXSB', writer, kernel))
  module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all subtile LRs to complete"))

  module.add(emitMfmaCode(writer, kernel))
  module.add(globalReadLDSBufferSwap('A', writer, kernel))
  module.add(globalReadLDSBufferSwap('B', writer, kernel))

  if hasMXScale:
    module.add(globalReadLDSBufferSwap('MXSA', writer, kernel))
    module.add(globalReadLDSBufferSwap('MXSB', writer, kernel))

  module.add(localReadLDSBufferSwap('A', writer, kernel))
  module.add(localReadLDSBufferSwap('B', writer, kernel))


  if hasMXScale:
    module.add(localReadLDSBufferSwap('MXSA', writer, kernel))
    module.add(localReadLDSBufferSwap('MXSB', writer, kernel))

  module.add(globalReadPtrUpdates('A', writer, kernel))
  module.add(globalReadPtrUpdates('B', writer, kernel))
  if hasMXScale:
    # Scale SRD pointer updates
    module.add(globalReadScalePtrUpdates('MXSA', writer, kernel))
    module.add(globalReadScalePtrUpdates('MXSB', writer, kernel))

  module.add(SSubU32(dst=sgpr("LoopCounterL"), src0=sgpr("LoopCounterL"), src1=1))
  module.add(SCmpEQU32(src0=sgpr("LoopCounterL"), src1=0))
  module.add(SCBranchSCC0(labelName=label.getLabelName()))

  return module


##################################################
# Subroutine entry point for preloop
#
# We will need to support different PGR values
# We will need to support different PLR values
#
def preLoop(writer, kernel):
  module = Module()
  module.addComment("")
  module.addComment("")
  pgr = kernel["PrefetchGlobalRead"]
  plr = kernel["PrefetchLocalRead"]
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based Preloop code with PGR=%u"%pgr)

  # Just sample impl, we can also interleave A/B loads
  for i in range(pgr):
    module.addComment0("Emitting %u-th set of GRs"%i)
    module.add(globalReadDoSubtile('A', writer, kernel))
    module.add(globalReadDoSubtile('B', writer, kernel))
    # Scale GR in preloop
    module.add(globalReadDoScaleSubtile('A', writer, kernel))
    module.add(globalReadDoScaleSubtile('B', writer, kernel))
    module.addComment("Add appropriate GR offset swap logic")
  module.addComment("")

  for i in range(plr):
    module.addComment("Add correct waits..")
    module.addComment0("Emitting LR to read data loaded by %u-th set of GRs"%(i))
    module.add(localReadDoSubtile('A', writer, kernel))
    module.add(localReadDoSubtile('B', writer, kernel))
    # Scale LR in preloop
    module.add(localReadDoScaleSubtile('A', writer, kernel))
    module.add(localReadDoScaleSubtile('B', writer, kernel))
    module.addComment("Add appropriate LR offset swap logic")

  module.addComment("")
  return module

##################################################
# Subroutine entry point for main loop
#
#
def mainLoop(writer, kernel):
  module = Module()
  pgr = kernel["PrefetchGlobalRead"]
  assert pgr in (0, 2), "SubtileBasedKernel only supports PGR=0 and PGR=2, got PGR=%d" % pgr

  # PGR=2 pipelining with LogicalScheduler
  if pgr == 2:
    from Tensile.Components.SubtileBasedLogicalScheduler import (
        LogicalScheduler, SchedulerConfig as MFMASchedulerConfig,
        ReadGranularity)
    tiA = writer.states.a.tileInfo
    tiB = writer.states.b.tileInfo
    scaleTiA = writer.states.mxsa.tileInfo if kernel["ProblemType"].get("MXBlockA", 0) else None
    scaleTiB = writer.states.mxsb.tileInfo if kernel["ProblemType"].get("MXBlockB", 0) else None

    lrAGran = ReadGranularity(mn=1, k=1)
    lrBGran = ReadGranularity(mn=1, k=1)
    # Based on current subtile shape. loadRatioGR == 2.0 has 2x2 granularity.
    grAGran = ReadGranularity(mn=1, k=2) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    grBGran = ReadGranularity(mn=1, k=2) if tiB.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    lrSAGran = ReadGranularity(mn=2, k=2) if scaleTiA else None
    lrSBGran = ReadGranularity(mn=2, k=2) if scaleTiB else None
    grSAGran = ReadGranularity(mn=scaleTiA.localMMATileGrid[0], k=scaleTiA.localMMATileGrid[1]) if scaleTiA else None
    grSBGran = ReadGranularity(mn=scaleTiB.localMMATileGrid[0], k=scaleTiB.localMMATileGrid[1]) if scaleTiB else None

    vgprBudget = writer.states.regCaps["MaxVgpr"]
    vgprUsed = writer.vgprPool.size() - writer.vgprPool.available()

    for numPartM, numPartN in MFMASchedulerConfig.get_partition_candidates(tiA, tiB):
        cfg = MFMASchedulerConfig(
            numMFMATilesM=tiA.localMMATileGrid[0],
            numMFMATilesN=tiB.localMMATileGrid[0],
            numSubIterK=tiA.localMMATileGrid[1],
            lrA=lrAGran,
            lrB=lrBGran,
            grA=grAGran,
            grB=grBGran,
            lrSA=lrSAGran,
            lrSB=lrSBGran,
            grSA=grSAGran,
            grSB=grSBGran,
            numPartitionsM=numPartM,
            numPartitionsN=numPartN,
        )
        scheduler = LogicalScheduler(cfg)
        scheduler.build()

        numVgpr = scheduler.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)
        if vgprUsed + numVgpr <= vgprBudget:
            break


    # Allocation and instruction emit
    scheduler.allocVgprTiles(writer, tiA, tiB,
                             scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)
    dtileInfo = writer.states.d.tileInfo
    scheduler.populate_instructions(
        writer, kernel,
        tileInfoA=tiA, tileInfoB=tiB, dtileInfo=dtileInfo,
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    module.add(scheduler.emitAllLoops(writer, kernel))
    scheduler.deallocVgprTiles(writer)

  else:
    # PGR=0: non-pipelined
    module.addComment0("MAINLOOP")
    module.add(mainLoopImplPGR0(writer, kernel))
    module.addComment("")

  return module
