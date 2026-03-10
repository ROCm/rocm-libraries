

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
from typing import Dict, List, NamedTuple, Optional,Tuple, Type
from contextlib import contextmanager
from collections import deque
from rocisa import rocIsa, countInstruction, countGlobalRead, \
            countLocalRead, countLocalWrite, countDSStoreB256, getMFMAs
from rocisa.code import Module, TextBlock, StructuredModule, KernelBody
from rocisa.container import RegisterContainer, replaceHolder, HWRegContainer, VCC, vgpr, sgpr, DPPModifiers, DSModifiers, EXEC
from rocisa.label import LabelManager
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.instruction import BufferLoadB128, BufferLoadB32, BufferLoadB64, \
  BufferLoadD16B16, BufferLoadD16U8, DSLoad2B32, DSLoad2B64, DSLoadB128, \
  DSLoadB32, DSLoadB64, DSLoadB64TrB16, DSLoadInstruction, DSLoadU16, \
  DSLoadU8, DSStore2B32, DSStore2B64, DSStoreB128, DSStoreB16, DSStoreB256, \
  DSStoreB32, DSStoreB64, DSStoreB8, DSStoreInstruction, FlatLoadB128, FlatLoadB32, \
  FlatLoadB64, FlatStoreB128, FlatStoreB32, FlatStoreB64, Instruction, MacroInstruction, \
  MFMAInstruction, SAddU32, SBarrier, SBranch, SCBranchSCC0, SCBranchSCC1, SCBranchVCCNZ, SCmpEQU32, SCmpLeU32, \
  SMFMAInstruction, SNop, SSetPrior, SSetRegIMM32B32, SSubU32, SWaitCnt, SWaitAlu, \
  SLongBranchPositive, VAccvgprWrite, VFmaMixF32, VMadMixF32, VMovB32, VAndB32, VCmpXEqU32, VCndMaskB32, VReadfirstlaneB32, \
  VMovB64, VLShiftRightB32, VLShiftLeftB32, VMulLOU32, VAddU32, VAddCOU32, VAddCCOU32, SMovB32, SMulI32, FlatStoreB32, SWaitCnt, SMovB64, VSubU32, VPermlane16SwapB32, MFMAInstruction
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType, DataTypeEnum
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

  # MMA Shape is w.r.t to data element (not size in bytes)
  #
  mmaTileShape: List[int] = field(init=False)
  mmaTileSize: int = 0 # subtile size in bytes
  mmaTileLocalTotalCount: int = 0 # total number of mmaTiles
  mmaTileRegCount: int = 0 # number of registers needed for per mma tile for specific A/B matrix

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

  vgprTileFactor: float = 1.0
  # VGPR buffers available for this tile
  vgprTiles: List[RegisterTileInfo] = field(init=False)

  def __init__(self, tc, kernel):
    self.subtileShape = [1, 2]

    isAB = tc in ['A', 'B']

    self.tc = tc

    if isAB:
      self.vgprTileFactor = 1.0 if tc == 'A' else 1.0
      miWaveGroupSize0 = kernel["MIWaveGroup"][0 if tc == 'A' else 1]
      miWaveGroupSize1 = 1

      macroTile = kernel["MacroTile%s"%tc]
      depthU = kernel["DepthU"]
      bpe = kernel["ProblemType"]["DataType%s"%tc].numBytes()
      self.bpe = bpe

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

    self.mmaTileSize = mmaTileShape0 * mmaTileShape1 * bpe
    # Number of registers needed for one tile, count w.r.t dword
    self.mmaTileRegCount = (self.mmaTileSize // kernel["WavefrontSize"]) // 4
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

    self.subtileLocalTotalCount = self.localSubtileGrid[0] * self.localSubtileGrid[1]

    # Allocate subtileInfo structs
    self.localSubtiles = []
    self.localSubtilesRegister = []
    for sId0 in range(self.localSubtileGrid[0]):
      for sId1 in range(self.localSubtileGrid[1]):
        self.localSubtiles.append(TileInfo.SubtileInfo(tc, [sId0, sId1]))

    if isAB:
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

    for i in range(self.numGRPerSubtile):
      self.sharedVgprGROffset.append(writer.vgprPool.checkOut(1))
    for i in range(self.numLRPerSubtile):
      self.sharedVgprLROffset.append(writer.vgprPool.checkOut(1))

    # Allocate registers for each subtile
    # TODOBS: Check TLU instead of hardcoding False
    perpDimSize = (self.localSubtileGrid[1] if False else self.localSubtileGrid[0])
    if self.loadRatioGR == 2.0:
      perpDimSize = math.ceil(perpDimSize / self.loadRatioGR)

    # TOBODS: Can this be done better
    for reg in range(perpDimSize):
      tmpSgprBuffer = 10 # Hardcoded for now, the amount of sgprs to use for temps
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

  def allocVgprTileRegisters(self, writer, kernel):
    self.vgprTiles = []

    numMMATiles = self.localMMATileGrid[0] * self.localMMATileGrid[1]
    for i in range(int(self.vgprTileFactor * numMMATiles)):
      if self.tc in ['A', 'B']:
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

      # TODOBS: Hard code this block for now?
      for j in range(0, self.mmaTileRegCount, 4):
        pool = self.vgprTiles[-1].regList.regPool
        vstart = pool.checkOutAligned(4,4)
        for k in range(4):
          self.vgprTiles[-1].append(vstart + k)



  def deallocOffsetRegisters(self, writer, kernel):
    # checkin GR registers
    for voff in self.sharedVgprGROffset:
      writer.vgprPool.checkIn(voff)
    # checkin LR registers
    for voff in self.sharedVgprLROffset:
      writer.vgprPool.checkIn(voff)


    for reg in self.localSubtilesRegister:
      regPool = reg.regPool
      for val in reg.regValues:
        regPool.checkIn(val)

  def deallocVgprTileRegisters(self, writer, kernel):
    # checkin GR registers
    for vtiles in self.vgprTiles:
      pool = vtiles.regList.regPool
      for vval in vtiles:
        # TODOBS: kinda hacky but checks in every 4 vgprs
        # to avoid double checking in vgprs
        if vtiles.index(vval) % 4 == 0:
          pool.checkIn(vval)




def _applySplitOffset(module, writer, kernel, tileInfo, lane16):
  tc = tileInfo.tc
  if tileInfo.loadRatioGR <= 1.0:
    wavesize = kernel["WavefrontSize"]
    depthUBytes = kernel["DepthU"] * tileInfo.bpe
    loadWidth = tileInfo.mmaTileShape[0] * tileInfo.mmaTileShape[1] * tileInfo.bpe // wavesize
    blockSize = depthUBytes // loadWidth
    numRowsPerWave = wavesize // blockSize
    offset = wavesize * loadWidth // 2  # bytes_loaded // 2

    splitOffset = writer.vgprPool.checkOut(1)
    module.add(VLShiftRightB32(dst=vgpr(splitOffset), shiftHex=hex((numRowsPerWave//2).bit_length()-1), src=vgpr(lane16), comment="%s: check 2nd half wave"%tc))
    module.add(VLShiftLeftB32(dst=vgpr(splitOffset), shiftHex=hex(offset.bit_length()-1), src=vgpr(splitOffset), comment="%s: x splitOffset"%tc))
    for vgprId in range(0, len(tileInfo.sharedVgprLROffset)):
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(splitOffset), comment="%s: +=splitOffset"%tc))
    writer.vgprPool.checkIn(splitOffset)

def _computeLROffset(module, kernel, tileInfo, colOffset, rowOffset):
  tc = tileInfo.tc
  wavesize = kernel["WavefrontSize"]
  depthUBytes = kernel["DepthU"] * tileInfo.bpe
  loadWidth = tileInfo.mmaTileShape[0] * tileInfo.mmaTileShape[1] * tileInfo.bpe // wavesize
  numMFMACols = tileInfo.mmaTileShape[1] * tileInfo.bpe // loadWidth  # TN case only
  blockSize = depthUBytes // loadWidth

  module.add(VMovB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]), src=vgpr(colOffset), comment="%s: laneId"%tc))
  for vgprId in range(1, len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId-1]), src1=hex(numMFMACols), comment="%s: colOffset for MFMA %u of subtile"%(tc, vgprId)))
    module.add(VAndB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=hex(blockSize-1), comment="%s: colOffset = colOffset %% block_size"%tc))

  for vgprId in range(0, len(tileInfo.sharedVgprLROffset)):
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(tileInfo.sharedVgprLROffset[vgprId]), comment="%s: colOffset*loadWidth"%tc))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(rowOffset), comment="%s: row + col"%tc))

def _applyWavePartitionLROffset(module, writer, kernel, tileInfo, waveId):
  """Apply wave-based partition offset to LR offsets.

  loadRatioGR >= 2.0: no partition needed, contiguous subtiles (1x4 for A , 4x1 for B)
  loadRatioGR == 1.0: 2x2 config, each wave loads half of the subtile (using interleaved blocks from split wave loads)
  loadRatioGR == 0.5: 4x1 for A , 1x4 for B. Split in 4 subtiles (interleaving + offset of MT/4)
  """
  tc = tileInfo.tc

  if tileInfo.loadRatioGR >= 2.0:
    return

  wavesize = kernel["WavefrontSize"]
  depthUBytes = kernel["DepthU"] * tileInfo.bpe
  loadWidth = tileInfo.mmaTileShape[0] * tileInfo.mmaTileShape[1] * tileInfo.bpe // wavesize
  bytes_loaded = wavesize * loadWidth

  tmpSgpr = writer.sgprPool.checkOut(1)
  tmp = writer.vgprPool.checkOut(2)
  tmp1 = tmp + 1

  if tileInfo.loadRatioGR == 1.0:
    # W0 W2
    # W1 W3
    # W1-3 : A / W2-3 : B
    if tc == 'A':
      module.add(VAndB32(dst=vgpr(tmp), src0=hex(1), src1=vgpr(waveId), comment="%s: waveId %% 2"%tc))
    else:
      module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(waveId), comment="%s: waveId / 2"%tc))
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=bytes_loaded // 2, comment="%s: bytes loaded per wave / 2"%tc))
    module.add(VMulLOU32(dst=vgpr(tmp), src0=sgpr(tmpSgpr), src1=vgpr(tmp), comment="%s: wave partition offset"%tc))

    for vgprId in range(len(tileInfo.sharedVgprLROffset)):
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(tmp), comment="%s: wave partition LR offset"%tc))

  elif tileInfo.loadRatioGR == 0.5:
    MT0 = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(MT0 * depthUBytes // 4), comment="%s: interleave stride"%tc))
    module.add(VAndB32(dst=vgpr(tmp1), src0=hex(1), src1=vgpr(waveId), comment="%s: waveId & 1"%tc))
    module.add(VMulLOU32(dst=vgpr(tmp1), src1=vgpr(tmp1), src0=sgpr(tmpSgpr), comment="%s: interleave offset"%tc))

    module.add(SMovB32(dst=sgpr(tmpSgpr), src=bytes_loaded // 2, comment="%s: bytes loaded per wave / 2"%tc))
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(waveId), comment="%s: waveId / 2"%tc))
    module.add(VMulLOU32(dst=vgpr(tmp), src1=vgpr(tmp), src0=sgpr(tmpSgpr), comment="%s: wave pair offset"%tc))
    module.add(VAddU32(dst=vgpr(tmp), src0=vgpr(tmp), src1=vgpr(tmp1), comment="%s: total partition offset"%tc))

    for vgprId in range(len(tileInfo.sharedVgprLROffset)):
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(tmp), comment="%s: wave partition LR offset"%tc))

  else:
    writer.vgprPool.checkIn(tmp)
    writer.sgprPool.checkIn(tmpSgpr)
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))

  writer.vgprPool.checkIn(tmp)
  writer.sgprPool.checkIn(tmpSgpr)

def _lraWavePartitioning(module, writer, kernel):
  """Compute waveId and apply per-matrix wave partition offsets."""
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  wavesize = kernel["WavefrontSize"]

  waveId = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="waveId"))

  _applyWavePartitionLROffset(module, writer, kernel, tileInfoA, waveId)
  _applyWavePartitionLROffset(module, writer, kernel, tileInfoB, waveId)

  writer.vgprPool.checkIn(waveId)

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
  depthU = kernel["DepthU"]
  bpeA = kernel["ProblemType"]["DataTypeA"].numBytes()
  bpeB = kernel["ProblemType"]["DataTypeB"].numBytes()
  depthUBytes = depthU * bpeA
  wavesize = kernel["WavefrontSize"]

  mi_m = tileInfoA.mmaTileShape[0]
  loadWidth = tileInfoA.mmaTileShape[0]*tileInfoA.mmaTileShape[1]*tileInfoA.bpe//wavesize
  ldsRowBankSize = 64*4 # 64 banks, 4 bytes per bank
  numRowsPerLDSBanks = ldsRowBankSize // depthUBytes
  assert tileInfoA.mmaTileShape == tileInfoB.mmaTileShape, "Expect same MMA tile shape for A and B"

  blockSize = depthUBytes // loadWidth

  tmpVgpr = writer.vgprPool.checkOut(8)
  lane16, lane16Group, rotation, rowOffset, colOffset, waveId, tmp, tmp1 = range(tmpVgpr, tmpVgpr + 8)

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
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(depthUBytes.bit_length()-1), src=vgpr(lane16), comment="offsetRow = depthUBytes*lane16"))

  # Calculate LR offset for A and B
  _computeLROffset(module, kernel, tileInfoA, colOffset, rowOffset)
  _computeLROffset(module, kernel, tileInfoB, colOffset, rowOffset)

  # Apply wavesplit offset separately on A & B as they are different for 1x4 and 4x1
  _applySplitOffset(module, writer, kernel, tileInfoA, lane16)
  _applySplitOffset(module, writer, kernel, tileInfoB, lane16)

  writer.vgprPool.checkIn(tmpVgpr)

  # Wave partitioning (e.g. 2x2 or 4x1/1x4)
  _lraWavePartitioning(module, writer, kernel)

  # Apply global offset on B (B data follows A in LDS).
  MT0A = tileInfoA.globalMMATileGrid[0] * tileInfoA.mmaTileShape[0]
  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(MT0A*depthUBytes), comment="LDS offset for B matrix"))
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), src1=sgpr(tmpSgpr), comment="B matrix offset : mt0*depthUBytes"))
  writer.sgprPool.checkIn(tmpSgpr)

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
def _grComputeOffset(module, writer, tileInfo, col_id, row_id, split_id):
  tc = tileInfo.tc
  bpe = tileInfo.bpe

  assert len(tileInfo.sharedVgprGROffset)<=2, "Only support 2 GR offset vgpr for now, found %u"%(len(tileInfo.sharedVgprGROffset))

  MT0 = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]
  subtile_size = tileInfo.subtileShape[0]*tileInfo.mmaTileShape[0]
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"

  tmpVgpr = writer.vgprPool.checkOut(2)
  sHalfOffset = writer.sgprPool.checkOut(1, preventOverflow=False)

  module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(strideRef), src1=vgpr(row_id), comment="%s: row_id * stride"%tc))
  # TODO : handle FP4 (sub byte type once available)
  module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(bpe.bit_length()-1), src=vgpr(tmpVgpr), comment="%s: row_id*stride*bpe"%tc))
  module.add(VAddU32(dst=vgpr(tmpVgpr), src0=vgpr(col_id), src1=vgpr(tmpVgpr), comment="%s: GR row_offset"%tc))

  # # apply top-half / bottom half offset according to wave split id
  if tileInfo.loadRatioGR == 2.0:
    module.add(SMovB32(dst=sgpr(sHalfOffset), src=(subtile_size * bpe), comment="%s: subtile row offset x bytes"%tc))
  else:
    module.add(SMovB32(dst=sgpr(sHalfOffset), src=(MT0 * bpe) // 2, comment="%s: Half Tile row offset x bytes"%tc))
  module.add(VMulLOU32(dst=vgpr(tmpVgpr+1), src0=sgpr(sHalfOffset), src1=vgpr(split_id), comment="%s: Apply offset for 2nd half wave"%tc))
  module.add(VMulLOU32(dst=vgpr(tmpVgpr+1), src0=sgpr(strideRef), src1=vgpr(tmpVgpr+1), comment="%s: Multiply by stride"%tc))

  module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprGROffset[0]), src0=vgpr(tmpVgpr), src1=vgpr(tmpVgpr+1), comment="%s: GR offset = row_offset + split_wave_offset"%tc))

  if len(tileInfo.sharedVgprGROffset)>1:
    module.add(SMovB32(dst=sgpr(sHalfOffset), src=(MT0 * bpe) // 4, comment="%s: 2nd GR offset calc : + %u rows"%(tc,MT0 // 4)))
    module.add(SMulI32(dst=sgpr(sHalfOffset), src0=sgpr(strideRef), src1=(MT0 * bpe) // 4, comment="%s: 2nd GR offset calc : + %u rows"%(tc,MT0 // 4)))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprGROffset[1]), src0=vgpr(tileInfo.sharedVgprGROffset[0]), src1=sgpr(sHalfOffset), comment="%s: GR offset for 2nd subtile = GR offset + subtile row offset"%tc))

  writer.sgprPool.checkIn(sHalfOffset)
  writer.vgprPool.checkIn(tmpVgpr)

##################################################
# Compute subtile perpendicular offsets for a single matrix
#
# TODO: need to generalize this to support TLU=1
def _grComputeSubtileOffsets(writer, module, tileInfo):
  tc = tileInfo.tc
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  subtile_size = tileInfo.subtileShape[0]*tileInfo.mmaTileShape[0]
  if tileInfo.loadRatioGR == 2.0:
    rowOffset = 2*subtile_size
  else:
    rowOffset = subtile_size

  s_stride = rowOffset * tileInfo.bpe

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

##################################################
# Subroutine to generate GR offset calculation code
#
def graTileAssignment(writer, kernel, useSwizzling=True):
  module = Module()
  module.addComment0("GR Offset Calculation for Subtile Based Tiling")

  # Input Parameters.
  depthU = kernel["DepthU"]
  bpeA = kernel["ProblemType"]["DataTypeA"].numBytes()
  bpeB = kernel["ProblemType"]["DataTypeB"].numBytes()
  depthUBytes = depthU * bpeA
  wavesize = kernel["WavefrontSize"]
  ldsRowBankSize = 64 * 4 # 64 banks, 4 bytes per bank.

  assert bpeA == 2 and bpeB == 2, "Only support fp16 for now"
  assert depthUBytes % 128 == 0, "Only support depthUBytes multiple of 128 for now"
  assert depthUBytes <= ldsRowBankSize, "Only support depthUBytes smaller than %u (lds row bank size) for now"%ldsRowBankSize

  loadWidth = 16 # dwordx4 loads only
  block_size = depthUBytes // loadWidth
  numRowsPerLDSBanks = ldsRowBankSize // depthUBytes

  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo

  tmpVgpr = writer.vgprPool.checkOut(7)
  col_id     = tmpVgpr
  row_id     = tmpVgpr + 1
  lds_row_id = tmpVgpr + 2
  split_id   = tmpVgpr + 3
  new_serial = tmpVgpr + 4
  wave_id    = tmpVgpr + 5
  tmp = tmpVgpr + 6

  # Compute newSerial
  module.add(VLShiftRightB32(dst=vgpr(wave_id), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  module.add(VAndB32(dst=vgpr(new_serial), src0=vgpr("Serial"), src1=31, comment=""))
  module.add(VLShiftLeftB32(dst=vgpr(wave_id), shiftHex=hex(5), src=vgpr(wave_id), comment=""))
  module.add(VAddU32(dst=vgpr(new_serial), src0=vgpr(wave_id), src1=vgpr(new_serial), comment="New Serial"))

  # Common code for both A & B
  # Calculate col and row id within a wave for 128b loads
  module.add(VAndB32(dst=vgpr(col_id), src0=vgpr(new_serial), src1=(block_size-1), comment="get col_id in wave for %uB load"%loadWidth))
  module.add(VLShiftRightB32(dst=vgpr(row_id), shiftHex=hex(block_size.bit_length()-1), src=vgpr(new_serial), comment="row id within wave"))

  if useSwizzling:
    module.addComment0("Swizzling")
    module.add(VLShiftRightB32(dst=vgpr(lds_row_id), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(row_id), comment="lds row id"))
    module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(lds_row_id), src1=hex(1), comment="lds row id % 2"))
    module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(tmp), comment="lds row id % 2 == 0 ?"))
    module.add(VMovB32(dst=vgpr(col_id), src=vgpr(col_id), dpp=DPPModifiers(quad_perm=[1,0,3,2]), comment="swap col_id pairs for swizzling"))
    module.add(SMovB64(dst=EXEC(), src=-1))
    module.addComment0("Rotation")
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(lds_row_id), comment=""))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="(lds_row_id //2) * 2"))
    module.add(VSubU32(dst=vgpr(tmp), src0=hex(block_size), src1=vgpr(tmp), comment="rotation offset : block_size - (lds_row_id//2)*2"))
    module.add(VAddU32(dst=vgpr(col_id), src0=vgpr(tmp), src1=vgpr(col_id), comment=""))
    module.add(VAndB32(dst=vgpr(col_id), src0=vgpr(col_id), src1=hex(block_size-1), comment="(col + offset) % block_size"))


  module.add(VLShiftLeftB32(dst=vgpr(col_id), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(col_id), comment="scale col_id by load_width"))

  # Get split Wave Id
  module.add(VLShiftRightB32(dst=vgpr(split_id), shiftHex=hex((wavesize//2).bit_length()-1), src=vgpr("Serial"), comment=""))
  module.add(VAndB32(dst=vgpr(split_id), src0=vgpr(split_id), src1=1, comment="wave split id [0-1]"))

  # Compute GR offset for A
  _grComputeOffset(module, writer, tileInfoA, col_id, row_id, split_id)

  # Compute GR offset for B
  _grComputeOffset(module, writer, tileInfoB, col_id, row_id, split_id)

  writer.vgprPool.checkIn(tmpVgpr)



  # Compute subtile offsets for A and B
  _grComputeSubtileOffsets(writer, module, tileInfoA)
  _grComputeSubtileOffsets(writer, module, tileInfoB)

  return module


##################################################
# Subroutine to generate GR offset calculation code for scaleA/B
#
def graTileAssignmentScaleSwizzled(writer, kernel):
  module = Module()
  module.addComment0("Placeholder code to generate GR offset calculations for scaleA/B")
  return module


##################################################
# Subroutine to generate LR offset calculation code for scaleA/B
#
def lraTileAssignmentScaleSwizzled(writer, kernel):
  module = Module()
  module.addComment0("Placeholder code to generate LR offset calculations for scaleA/B")
  return module

##################################################
# Subroutine to generate GR load code
#
def emitSubtileBufferLoad(tc, writer, kernel, subtileId):
  module = Module()
  sId0 = subtileId[0]
  sId1 = subtileId[1]

  loadWidth = 16
  numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
  numBytesPerLoad = loadWidth * kernel["WavefrontSize"]
  numBytesPerLoadWG = numBytesPerLoad * numWaves

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  subtileInfo = tileInfo.localSubtiles[tileInfo.getLocalSubtileLinearId(sId0, sId1)]
  regList = tileInfo.localSubtilesRegister[subtileInfo.regListId]

  offset = sId1 * tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1] * tileInfo.bpe
  ldsStartOffset = writer.ldsStartOffsetA if tc == 'A' else writer.ldsStartOffsetB
  m0Offset = ldsStartOffset

  grBaseId = tileInfo.localSubtiles[tileInfo.getLocalSubtileLinearId(sId0, sId1)].globalReadMap[0]

  # Emit number of buffer loads equal to number of loads needed to load a subtile
  for i in range(tileInfo.numGRPerSubtile):
    module.add(SAddU32(dst=mgpr(0), src0=sgpr("LocalWriteBaseAddr"), src1=(m0Offset + (grBaseId + i) * numBytesPerLoadWG - offset)))
    mubuf = MUBUFModifiers(offen=True, offset12=offset, glc=False, slc=False, nt=False, lds=True)

    # Check if the subtile specific registers is SGPR or VGPR
    # For SGPR we can keep the same shared vgpr offset and use the soffset field for the subtile specific SGPR
    # For VGPR we need to update the apply the subtile-specific constant offset to the VGPR
    #   the shared VGPR offset is not used for that specific tile, soffset is also set to zero.
    useSgpr = subtileInfo.useSgpr
    soffset = sgpr(regList.regValues[0]) if len(regList) > 0 and useSgpr else 0
    voff = tileInfo.sharedVgprGROffset[i] if useSgpr or len(regList) == 0 else regList.regValues[i]
    module.add(BufferLoadB128(dst=None, vaddr=vgpr(voff), saddr=sgpr("Srd%s"%tc, 4), soffset=soffset, mubuf=mubuf, comment=""))

  return module

##################################################
# Subroutine to generate GR load code
# Initial idea: maybe store asm in modules in a separate obj?
#
def globalReadDoSubtile(tc, writer, kernel):
  module = Module()

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo

  grTracker = set()
  for i in range(tileInfo.localSubtileGrid[0]):
    for j in range(tileInfo.localSubtileGrid[1]):
      grIds = tileInfo.localSubtiles[tileInfo.getLocalSubtileLinearId(i ,j)].globalReadMap
      if not set(grIds).issubset(grTracker):
        for grId in grIds:
          grTracker.add(grId)
        module.addComment0("Emit load for %s subtile: [%u, %u]"%(tc, i, j))
        module.add(emitSubtileBufferLoad(tc, writer, kernel, [i, j]))
      else:
        module.addComment0("Emit load for %s subtile: [%u, %u] - already covered"%(tc, i, j))

  return module

def emitSubtileDsRead(writer, kernel, tileInfo, subtileId):
  
  module = Module()
  sId0 = subtileId[0]
  sId1 = subtileId[1]

  linearId = tileInfo.getLocalSubtileLinearId(sId0, sId1)
  subtileInfo = tileInfo.localSubtiles[linearId]

  for mfmaC in range(tileInfo.subtileShape[1]):
    for mfmaR in range(tileInfo.subtileShape[0]):
      mfmaId = tileInfo.getSubtileShapeLinearId(mfmaC, mfmaR)
      addrVgpr = tileInfo.sharedVgprLROffset[mfmaId]
      dstTile = tileInfo.vgprTiles[subtileInfo.localReadMap[mfmaId]]
      dstVgpr = dstTile.regList.regValues[0]
      numRegs = len(dstTile.regList.regValues)
      offset = sId0*2*tileInfo.subtileSize
      module.add(DSLoadB128(dst=vgpr(dstVgpr, numRegs), src=vgpr(addrVgpr), ds=DSModifiers(offset=offset),
                            comment="Subtile%s[%u,%u] mfmaId=[%u,%u]"%(tileInfo.tc, sId0, sId1, mfmaR, mfmaC)))
     
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

  loadWidth = 16 # dwordx4 loads only
  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1)
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  module.add(VLShiftLeftB32(dst=vgpr(vgprWaveId), shiftHex=hex((loadWidth * wavesize).bit_length()-1), src=vgpr(vgprWaveId), comment="Apply wave-specific offset of %u"%(loadWidth * wavesize)))
  module.add(SNop(waitState=0, comment="Wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddr"), src=vgpr(vgprWaveId), comment="Store base LDS offset, will be modified"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteDTLOffset"), src=vgpr(vgprWaveId), comment="Store DTL wave-specific offset, this will not be modified"))
  writer.vgprPool.checkIn(vgprWaveId)
  return module



##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def globalReadLDSBufferSwap(tc, writer, kernel):
  module = Module()
  module.addComment0("Emit code to swap %s GR vgpr offsets"%tc)
  return module

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def localReadLDSBufferSwap(tc, writer, kernel):
  module = Module()
  module.addComment0("Emit code to swap %s LR vgpr offsets"%tc)
  return module

##################################################
# Subroutine to update ptrs
#
def globalReadPtrUpdates(tc, writer, kernel):
  module = Module()
  module.addComment0("Emit code to update %s pointers"%tc)
  return module

##################################################
# Subroutine to generate MMA Instruction
# Given RegisterTileInfo inputs for A,B,C,D operands
# emit corresponding mfma instruction
#
def emitMfmaInstruction(writer, kernel, vgprTileA, vgprTileB, vgprTileC, vgprTileD, comment = ""):
  module = Module()

  vgprAStart = vgprTileA.regList.regValues[0]
  vgprBStart = vgprTileB.regList.regValues[0]
  vgprCStart = vgprTileC.regList.regValues[0]
  vgprDStart = vgprTileD.regList.regValues[0]

  opASize = len(vgprTileA.regList.regValues)
  opBSize = len(vgprTileB.regList.regValues)
  opCSize = len(vgprTileC.regList.regValues)
  opDSize = len(vgprTileD.regList.regValues)

  accvgprAlias = vgpr if kernel["MIArchVgpr"] else accvgpr
  module.add(MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[16,16,32,1], mfma1k=False, \
                             acc=accvgprAlias(vgprDStart,opDSize), \
                             a=vgpr(vgprAStart,opBSize), \
                             b=vgpr(vgprBStart,opBSize), \
                             acc2=accvgprAlias(vgprCStart,opCSize), \
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

  for mmak in range(atileInfo.localMMATileGrid[1]):
    for mma1 in range(btileInfo.localMMATileGrid[0]):
      for mma0 in range(atileInfo.localMMATileGrid[0]):
        atiles = atileInfo.vgprTiles[mma0 + mmak * atileInfo.localMMATileGrid[0]]
        btiles = btileInfo.vgprTiles[mma1 + mmak * btileInfo.localMMATileGrid[0]]
        dtiles = dtileInfo.vgprTiles[mma0 + mma1 * dtileInfo.localMMATileGrid[0]]
        module.add(emitMfmaInstruction(writer, kernel, atiles, btiles, dtiles, dtiles, "Emit MMFA code for MMA tiles C[%u, %u] += A[%u, %u] * B[%u, %u]"%(mma0, mma1, mma0, mmak, mmak, mma1)))

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
def mainLoopImpl(writer, kernel, isNLL = False):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based main loop impl")



  if not isNLL:
    #module.add(Label("testL", comment=""))
    module.add(globalReadDoSubtile('A', writer, kernel))
    module.add(globalReadDoSubtile('B', writer, kernel))
    module.add(SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1, comment="Wait for all subtile GRs to complete"))
    module.add(SBarrier(comment=""))

  module.add(localReadDoSubtile('A', writer, kernel))
  module.add(localReadDoSubtile('B', writer, kernel))
  module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all subtile LRs to complete"))
  

  #module.add(globalReadLDSBufferSwap('A', writer, kernel))
  #module.add(globalReadLDSBufferSwap('B', writer, kernel))

  #module.add(localReadLDSBufferSwap('A', writer, kernel))
  #module.add(localReadLDSBufferSwap('B', writer, kernel))

  #module.add(globalReadPtrUpdates('A', writer, kernel))
  #module.add(globalReadPtrUpdates('B', writer, kernel))

  module.add(emitMfmaCode(writer, kernel))

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
    module.addComment("Add appropriate GR offset swap logic")
  module.addComment("")

  for i in range(plr):
    module.addComment("Add correct waits..")
    module.addComment0("Emitting LR to read data loaded by %u-th set of GRs"%(i))
    module.add(localReadDoSubtile('A', writer, kernel))
    module.add(localReadDoSubtile('B', writer, kernel))
    module.addComment("Add appropriate LR offset swap logic")

  module.addComment("")
  return module

##################################################
# Subroutine entry point for main loop
#
#
def mainLoop(writer, kernel):
  module = Module()
  module.addComment0("MAINLOOP")
  module.add(mainLoopImpl(writer, kernel))
  module.addComment("")

  #module.addComment0("MAINLOOP-NLL")
  #isNLL = True
  #module.add(mainLoopImpl(writer, kernel, isNLL))
  #module.addComment("")

  return module
