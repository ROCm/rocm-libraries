

from ..Common import printWarning, roundUp, print2, DebugConfig, DataDirection, \
  INDEX_CHARS, IsaVersion


from rocisa.code import Module, TextBlock, StructuredModule, KernelBody, Label
from rocisa.label import LabelManager

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional,Tuple, Type
from contextlib import contextmanager
from collections import deque

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

    # Store registers used for constant offsets
    useSgpr = field(init=False)
    regListId: int = -1

    def __init__(self, tc, subtileId):
      self.tc = tc
      self.subtileId = subtileId

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
  localSubtileRegisters: List[RegisterList] = field(init=False)

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

    # Allocate subtileInfo structs
    self.localSubtiles = []
    self.localSubtilesRegister = []
    for sId0 in range(self.localSubtileGrid[0]):
      for sId1 in range(self.localSubtileGrid[1]):
        self.localSubtiles.append(TileInfo.SubtileInfo(tc, [sId0, sId1]))

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
    for reg in range(perpDimSize):
      tmpSgprBuffer = 10 # Hardcoded for now, the amount of sgprs to use for temps
      sgprLimit = writer.states.regCaps["MaxSgpr"] - tmpSgprBuffer
      regPool = writer.sgprPool if writer.sgprPool.size() < sgprLimit else writer.vgprPool
      self.localSubtilesRegister.append(TileInfo.RegisterList(regPool))
      # No registers needed for perp 0
      if reg == 0:
        continue
      for i in range(self.numGRPerSubtile):
        # TODOBS: Need to prevent overflow here, better way to do it?
        self.localSubtilesRegister[-1].append(regPool.checkOut(1, preventOverflow=False))

    # Iterate through subtiles and allocate vgpr/sgpr if needed
    linearId = 0
    for st in self.localSubtiles:

      sId0, sId1 = self.getLocalSubtileIdFromLinearId(linearId)
      linearId += 1

      # TODOBS: Check TLU instead of hardcoding
      slowId = sId1 if False else sId0
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




##################################################
# Subroutine to generate LR offset calculation code
#
def lraTileAssignment(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based LR Offset computation")
  for i in range(8):
    module.addComment("")

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
def graTileAssignment(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based GR Offset computation")
  for i in range(8):
    module.addComment("")

  return module


##################################################
# Subroutine to generate GR load code
# Initial idea: maybe store asm in modules in a separate obj?
#
def globalReadDoSubtile(tc, writer, kernel):
  module = Module()

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo

  for i in range(tileInfo.localSubtileGrid[0]):
    for j in range(tileInfo.localSubtileGrid[1]):
      for k in range(tileInfo.numGRPerSubtile):
        module.addComment("Emit GR code for subtile %s(%u, %u) - %u"%(tc, i,j,k))

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
      for k in range(tileInfo.numLRPerSubtile):
        module.addComment("Emit LR code for subtile %s(%u, %u) - %u"%(tc, i,j,k))

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
        atiles = atileInfo.vgprTiles
        btiles = btileInfo.vgprTiles
        dtiles = dtileInfo.vgprTiles
        module.addComment("Emit MMFA code for MMA tiles C[%u, %u] += A[%u, %u] * B[%u, %u]"%(mma0, mma1, mma0, mmak, mmak, mma1))

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
  module.add(emitMfmaCode(writer, kernel))
  if not isNLL:
    module.add(globalReadDoSubtile('A', writer, kernel))
    module.add(globalReadDoSubtile('B', writer, kernel))
  module.add(localReadDoSubtile('A', writer, kernel))
  module.add(localReadDoSubtile('B', writer, kernel))

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

  module.addComment0("MAINLOOP-NLL")
  isNLL = True
  module.add(mainLoopImpl(writer, kernel, isNLL))
  module.addComment("")

  return module
