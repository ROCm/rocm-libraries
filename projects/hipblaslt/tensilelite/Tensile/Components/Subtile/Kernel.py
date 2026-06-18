# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Dict, List, NamedTuple, Optional, Tuple, Type
from Tensile.Components.Subtile.LogicalScheduler import (
      LogicalScheduler, SchedulerConfig as MFMASchedulerConfig,
      ReadGranularity, GRPlacementStrategy)

from ...Common import printWarning, roundUp, print2, DebugConfig, DataDirection, \
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

from .SubtileGeometry import (
  RegList,
  LoadShape,
  MMALayout,
  MMAScaleLayout,
  MFMA_16x16_1B_4K_4V,
  MFMA_16x16_1B_4K_8V,
  MFMA_16x16_1B_4N_4V,
  MFMA_SCALE_16x16_1B_MX32_8V,
  TileGeometry,
  ABInputGeometry,
  ABGRGeometry,
  ABLRGeometry,
  GRTag_1x1, GRTag_1x2, GRTag_2x2, GRTag_TLU1,
  LRTag_1x1, LRTag_1x2, LRTag_TLU1,
  ABTilePair,
  CDTileGeometry,
  MXScaleInputGeometry,
  MXScaleGRGeometry,
  MXScaleLRGeometry,
  MXScaleTilePair,
)

################################################################################
# Concrete tile classes and pre-defined config instances
#
# Config objects (frozen): ABGRGeometry / ABLRGeometry instances, each with a
#   tag field (GRTag_1x2 | GRTag_TLU1 / LRTag_1x2 | LRTag_TLU1) imported from
#   SubtileGeometry. The tag selects the emit implementation via singledispatch.
#
# Runtime tile classes (mutable): ABGRTile, ABLRTile — hold any frozen
#   ABGRGeometry / ABLRGeometry config + mutable register state + emit logic.
#   Created by TileInfo.__init__; emit dispatch is via self.config.tag.
#
# Pre-defined pair instances (e.g. AB_B16): ABTilePair of frozen configs.
#   The code generator passes these to TileInfo, which creates the mutable
#   tile instances automatically.
################################################################################

from .SubtileGREmit import (
    _emitGlobalReadOffset, _emitGlobalRead, _emitLocalWrite,
    _allocGROffsetRegisters, _deallocGROffsetRegisters,
    _emitDTLInit, _emitGRLDSBufferSwap, _emitGRPtrUpdate,
    graInitPointer, graTileAssignment,
    emitSingleBufferLoad, emitSubtileBufferLoad, globalReadDoSubtile,
    globalReadDTLInitCommonSgpr, globalReadLDSBufferSwap, globalReadPtrUpdates,
    tdmGlobalOffsetSubtile, initTDMDescriptorSubtile, tdmApplyStreamKOffsetSubtile,
)
from .SubtileLREmit import (
    _emitLocalReadOffset, _emitLocalRead,
    _allocLROffsetRegisters, _deallocLROffsetRegisters,
    _emitLRDTLInit, _emitLRLDSBufferSwap,
    lraTileAssignment, localReadDoSubtile, localReadDTLInitCommonSwapVgpr,
    localReadLDSBufferSwap, localReadResetOffsetsSubtile,
    emitSingleDsRead, emitSubtileDsRead, setExecMask,
)
from .SubtileScaleEmit import (
    emitScaleGROffset, emitScaleLROffset,
    emitScaleGRLoad, emitScaleLRLoad,
    emitScaleGRPtrUpdate, emitScaleGRLDSSwap, emitScaleLRLDSSwap,
    graTileAssignmentScaleSwizzled, lraTileAssignmentScaleSwizzled,
    globalReadDoScaleSubtile, localReadDoScaleSubtile,
    globalReadScalePtrUpdates, globalReadScaleSwizzledDTLInitCommonSgpr,
    emitSubtileScaleDsRead,
)

class ABGRTile:
  """Mutable GR tile for A/B global reads.

  Holds any frozen ABGRGeometry config. Shape-dependent parameters are
  computed once in __init__ from the config; emit methods read those
  parameters directly with no isinstance branching.

  Migration path: as offset formulas are auto-computed from config fields,
  the isinstance block in __init__ collapses into direct field reads and
  eventually disappears.
  """

  def __init__(self, config: ABGRGeometry):
    self.config = config
    self.sharedVgprGROffset: List[int] = []

    # Shape descriptor — computed once, read by emit methods generically.
    # contiguousDim:      dimension that is stride-1 in memory ('K' or 'M').
    # contiguousElements: elements per load in the contiguous dimension.
    # Memory stride is supplied by LDA/LDB kernel parameters, not stored here.
    if isinstance(config.tag, GRTag_TLU1):
      self.contiguousDim      = 'M'
      self.contiguousElements = config.loadShape.m
    else:  # row-major (GRTag_1x2 and future row-major shapes)
      self.contiguousDim      = 'K'
      self.contiguousElements = config.loadShape.k

  @property
  def subtileShape(self) -> Tuple[int, int]:
    return self.config.subtileShape

  @property
  def subtileCount(self) -> int:
    return self.config.subtileCount

  @property
  def subtileStride(self) -> int:
    return self.config.subtileStride

  def localGRGranularity(self, numWaves: int) -> Tuple[int, int]:
    return self.config.localGRGranularity(numWaves)


  @property
  def loadShape(self):
    return self.config.loadShape

  # --- Register allocation ---

  def allocOffsetRegisters(self, ti, writer, kernel):
    return _allocGROffsetRegisters(self.config.tag, self, ti, writer, kernel)

  def deallocOffsetRegisters(self, ti, writer, kernel):
    return _deallocGROffsetRegisters(self.config.tag, self, ti, writer, kernel)

  # --- Emit ---

  def emitGlobalReadOffset(self, ti, writer, kernel):
    return _emitGlobalReadOffset(self.config.tag, self, ti, writer, kernel)

  def emitGlobalRead(self, ti, writer, kernel):
    return _emitGlobalRead(self.config.tag, self, ti, writer, kernel)

  def emitLocalWrite(self, ti, writer, kernel):
    return _emitLocalWrite(self.config.tag, self, ti, writer, kernel)

  def emitDTLInit(self, ti, writer, kernel):
    return _emitDTLInit(self.config.tag, self, ti, writer, kernel)

  def emitLDSBufferSwap(self, ti, writer, kernel):
    return _emitGRLDSBufferSwap(self.config.tag, self, ti, writer, kernel)

  def emitPtrUpdate(self, ti, writer, kernel):
    return _emitGRPtrUpdate(self.config.tag, self, ti, writer, kernel)


class ABLRTile:
  """Mutable LR tile for A/B local reads.

  Holds any frozen ABLRGeometry config. Shape-dependent parameters are
  computed once in __init__ from the config; emit methods read those
  parameters directly with no isinstance branching.
  """

  def __init__(self, config: ABLRGeometry):
    self.config = config
    self.sharedVgprLROffset: List[int] = []
    self.sharedVgprLROffsetSwap: List[int] = []
    self.localSubtiles: List = []

    # Shape descriptor — same convention as ABGRTile.
    if isinstance(config.tag, LRTag_TLU1):
      self.contiguousDim      = 'M'
      self.contiguousElements = config.loadShape.m
    else:  # row-major
      self.contiguousDim      = 'K'
      self.contiguousElements = config.loadShape.k

  @property
  def subtileShape(self) -> Tuple[int, int]:
    return self.config.subtileShape

  @property
  def loadShape(self):
    return self.config.loadShape

  # --- Register allocation ---

  def allocOffsetRegisters(self, ti, writer, kernel):
    return _allocLROffsetRegisters(self.config.tag, self, ti, writer, kernel)

  def deallocOffsetRegisters(self, ti, writer, kernel):
    return _deallocLROffsetRegisters(self.config.tag, self, ti, writer, kernel)

  # --- Emit ---

  def emitLocalReadOffset(self, ti, writer, kernel):
    return _emitLocalReadOffset(self.config.tag, self, ti, writer, kernel)

  def emitLocalRead(self, ti, writer, kernel):
    return _emitLocalRead(self.config.tag, self, ti, writer, kernel)

  def emitDTLInit(self, ti, writer, kernel):
    return _emitLRDTLInit(self.config.tag, self, ti, writer, kernel)

  def emitLDSBufferSwap(self, ti, writer, kernel):
    return _emitLRLDSBufferSwap(self.config.tag, self, ti, writer, kernel)



# MX scale tile and C/D tile (still frozen — no register state yet)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CDTile_1x1(CDTileGeometry):
  """C/D tile with subtileShape (1, 1) — 1 MMA tile in both M and N."""
  subtileShape: Tuple[int, int] = (1, 1)

  def emitStoreD(self, ti: 'TileInfo', writer, kernel): pass
  def emitLoadC(self, ti: 'TileInfo', writer, kernel): pass


# ---------------------------------------------------------------------------
# Pre-defined instances (frozen config pairs)
# TODO: rename configs to make the geometry explicit (subtileShape, subtileCount
#       derivation policy, TLU) — e.g. AB_B16_1x2_bcN, AB_B16_2x2_bc1
# ---------------------------------------------------------------------------

_B16 = dict(mmaLayout=MFMA_16x16_1B_4K_4V, instK=32,  bpe=2,   supportedTypes=('bf16', 'fp16'))
_B4  = dict(mmaLayout=MFMA_16x16_1B_4K_4V, instK=128, bpe=0.5, supportedTypes=('fp4',))
_B8  = dict(mmaLayout=MFMA_16x16_1B_4K_8V, instK=128, bpe=1,   supportedTypes=('fp8', 'bf8'))

# Row-major A/B: GR and LR both contiguous along K
AB_B16 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_1x2(), **_B16, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=8)),   # 128-bit GR: 8 bf16 along K
    lr=ABLRGeometry(tag=LRTag_1x2(), **_B16, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=8)), # 128-bit LR: 8 bf16 along K
)

# Wave32 bf16: 8 VGPRs per operand (WMMA V3 gfx1250)
_B16_W32 = dict(mmaLayout=MMALayout(instM=16, blocks=1, vgprs=8, waveSize=32), instK=32, bpe=2, supportedTypes=('bf16', 'fp16'))
AB_B16_W32 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_1x2(), **_B16_W32, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=8)),
    lr=ABLRGeometry(tag=LRTag_1x2(), **_B16_W32, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=8)),
)

AB_B4 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_1x2(), **_B4, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=32)),   # 128-bit GR: 32 fp4 along K
    lr=ABLRGeometry(tag=LRTag_1x2(), **_B4, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=32)), # 128-bit LR: 32 fp4 along K
)
AB_B8 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_1x1(), **_B8, subtileShape=(1, 1), loadShape=LoadShape(m=1, k=16)),  # 128-bit GR: 16 fp8 along K
    lr=ABLRGeometry(tag=LRTag_1x1(), **_B8, subtileShape=(1, 1), loadShape=LoadShape(m=1, k=16)), # 128-bit LR: 16 fp8 along K
)

AB_B4_2x2 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_2x2(), **_B4, subtileShape=(2, 2), subtileCount=1, subtileStride=0, loadShape=LoadShape(m=1, k=32)),
    lr=ABLRGeometry(tag=LRTag_1x2(), **_B4, subtileShape=(2, 2), loadShape=LoadShape(m=1, k=32)),
)
AB_B16_2x2 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_2x2(), **_B16, subtileShape=(2, 2), subtileCount=1, subtileStride=0, loadShape=LoadShape(m=1, k=8)),
    lr=ABLRGeometry(tag=LRTag_1x2(), **_B16, subtileShape=(2, 2), loadShape=LoadShape(m=1, k=8)),
)

# Column-major A/B (TLU=1): GR and LR contiguous along M
AB_B16_TLU1 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_TLU1(), **_B16, tlu=True, subtileShape=(8, 1), subtileCount=1, subtileStride=0, loadShape=LoadShape(m=8, k=1)),   # 128-bit GR: 8 bf16 along M
    lr=ABLRGeometry(tag=LRTag_TLU1(), **_B16, tlu=True, subtileShape=(8, 1), loadShape=LoadShape(m=8, k=1)),                              # 128-bit LR: 8 bf16 along M
)
AB_B16_TLU1_16x1 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_TLU1(), **_B16, tlu=True, subtileShape=(16, 1), subtileCount=1, subtileStride=0, loadShape=LoadShape(m=16, k=1), loadWidth=32), # 256-bit GR: 16 bf16 along M
    lr=ABLRGeometry(tag=LRTag_TLU1(), **_B16, tlu=True, subtileShape=(16, 1), loadShape=LoadShape(m=16, k=1), loadWidth=32),                            # 256-bit LR: 16 bf16 along M
)

# MX scale factor inputs (one scale per mxBlock data elements)
_MXS_B4 = dict(scaleLayout=MFMA_SCALE_16x16_1B_MX32_8V, instK=128, bpe=1, supportedTypes=('fp4',))
_MXS_B8 = dict(scaleLayout=MFMA_SCALE_16x16_1B_MX32_8V, instK=128, bpe=1, supportedTypes=('fp8', 'bf8'))

# GR: subtileShape=None -> derived from kernel as (mt_mma, du_scale) to span entire macro tile
# LR: subtileShape=(2,2) -> 2 scale MMA tiles in M x 2 in K per local read
MXSA_B4 = MXScaleTilePair(gr=MXScaleGRGeometry(**_MXS_B4, loadWidth=16), lr=MXScaleLRGeometry(**_MXS_B4, loadWidth=4))
MXSB_B4 = MXScaleTilePair(gr=MXScaleGRGeometry(**_MXS_B4, loadWidth=16), lr=MXScaleLRGeometry(**_MXS_B4, loadWidth=4))
MXSA_B8 = MXScaleTilePair(gr=MXScaleGRGeometry(**_MXS_B8, loadWidth=16), lr=MXScaleLRGeometry(**_MXS_B8, loadWidth=4))
MXSB_B8 = MXScaleTilePair(gr=MXScaleGRGeometry(**_MXS_B8, loadWidth=16), lr=MXScaleLRGeometry(**_MXS_B8, loadWidth=4))

# C/D output: 128-bit store = 4 f32 elements along N
CD_F32 = CDTile_1x1(mmaLayout=MFMA_16x16_1B_4N_4V, bpe=4, supportedTypes=('f32',), storeShape=LoadShape(m=1, k=4))
# Wave32 f32 output: 8 VGPRs per lane (WMMA V3 gfx1250)
CD_F32_W32 = CDTile_1x1(mmaLayout=MMALayout(instM=16, blocks=1, vgprs=8, waveSize=32), bpe=4, supportedTypes=('f32',), storeShape=LoadShape(m=1, k=8))

def selectMXScaleGeometry(kernel: dict, tc: str) -> MXScaleTilePair:
  """Return the MXScaleTilePair for scale tensor tc ('MXSA' or 'MXSB')."""
  data_tc = 'A' if tc == 'MXSA' else 'B'
  dtype = kernel["ProblemType"][f"DataType{data_tc}"]
  if dtype.is6bitFloat() or dtype.isFloat4():
    return MXSA_B4 if tc == 'MXSA' else MXSB_B4
  if dtype.is8bitFloat():
    return MXSA_B8 if tc == 'MXSA' else MXSB_B8
  raise NotImplementedError(f"selectMXScaleGeometry: unsupported dtype {dtype} for tc={tc}")


AB_GEOMETRY_MAP = {
  "AB_B16":      AB_B16,
  "AB_B16_2x2":  AB_B16_2x2,
  "AB_B4":       AB_B4,
  "AB_B4_2x2":   AB_B4_2x2,
  "AB_B8":       AB_B8,
  "AB_B16_TLU1": AB_B16_TLU1,
  "AB_B16_TLU1_16x1": AB_B16_TLU1_16x1,
  "AB_B16_W32":  AB_B16_W32,
}

def selectABGeometry(kernel: dict, tc: str) -> ABTilePair:
  """Return the ABTilePair selected by Solution.py for tc ('A' or 'B')."""
  key = kernel[f"_ABTilePair{tc}"]
  return AB_GEOMETRY_MAP[key]


def selectDGeometry(kernel: dict) -> CDTileGeometry:
  """Return the CDTileGeometry for the D (output/accumulator) tile."""
  if kernel["WavefrontSize"] == 32:
    return CD_F32_W32
  return CD_F32


################################################################################
# TileInfo — runtime tile state
################################################################################

class TileInfo:
  """Runtime tile state combining frozen geometry with kernel/writer config.

  Takes an immutable TileGeometry (defines the MMA/subtile structure) and
  binds it to a specific kernel configuration (macro tile, wave group,
  depthU) and writer (register pools). This is the mutable working object
  used during code generation.

  Args:
    geometry: A concrete TileGeometry instance (e.g. AB_B16, CD_F32).
    tc:       Tensor component ('A', 'B', 'MXSA', 'MXSB', 'C').
    writer:   KernelWriter with register pools (vgprPool, sgprPool, agprPool).
    kernel:   Kernel configuration dictionary.
  """

  def __init__(self, geometry: TileGeometry, tc: str, writer, kernel):
    self.geometry = geometry
    self.tc = tc

    isA = tc in ['A', 'MXSA']
    isAB = tc in ['A', 'B']
    isMXSAB = tc in ['MXSA', 'MXSB']
    _tc = 'A' if isA else 'B'

    # --- Extract kernel config ---
    if isinstance(geometry, (ABTilePair, MXScaleTilePair)):
      self.macroTile = kernel["MacroTileA"] if isA else kernel["MacroTileB"]
      # MXScaleTilePair geometry expects data DepthU (not scale DepthU = _DepthUMXSA/MXSB).
      # ABTilePair uses the per-TC DepthU key directly.
      if isinstance(geometry, MXScaleTilePair):
        self.depthU = kernel["_DepthU%s" % _tc]  # data DepthU for A or B (needed by globalMMATileGrid)
        self.scaleDepthU = kernel["_DepthU%s" % tc]  # scale DepthU (e.g. _DepthUMXSA = 8)
      else:
        self.depthU = kernel["_DepthU%s" % tc]
        self.scaleDepthU = self.depthU
      self.waveGroupSize = kernel["MIWaveGroup"][0 if isA else 1]
      self.isSwizzled = isinstance(geometry, MXScaleTilePair)
    elif isinstance(geometry, CDTileGeometry):
      self.macroTile = None  # C/D uses macroTile0/1
      self.macroTile0 = kernel["MacroTile0"]
      self.macroTile1 = kernel["MacroTile1"]
      self.waveGroup = (kernel["MIWaveGroup"][0], kernel["MIWaveGroup"][1])
      self.depthU = None
      self.isSwizzled = False

    self.waveSize = kernel["WavefrontSize"]
    self.numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]

    # --- Compute instantiated grids (geometry + kernel config) ---
    # Subtile grid is global (waves cooperate on subtiles).
    # MMA tile grid has both global and local (per-wave) views.
    if isinstance(geometry, ABTilePair):
      gr_cfg = geometry.gr.for_kernel(kernel, tc)
      lr_cfg = geometry.lr

      # Create mutable runtime tile instances (hold frozen config + register state)
      self.gr = ABGRTile(gr_cfg)
      self.lr = ABLRTile(lr_cfg)

      # MMA tile grids (shared — GR and LR use the same MMA instruction)
      self.globalMMATileGrid = list(gr_cfg.globalMMATileGrid(self.macroTile, self.depthU))
      self.localMMATileGrid  = list(gr_cfg.localMMATileGrid(self.macroTile, self.depthU, self.waveGroupSize))

      # GR strip grid — primary scheduler-facing grid
      self.subtileShape        = list(gr_cfg.subtileShape)
      self.subtileCount        = gr_cfg.subtileCount
      self.subtileStride       = gr_cfg.subtileStride
      self.globalSubtileGrid = list(gr_cfg.globalSubtileGrid(self.macroTile, self.depthU))
      self.localSubtileGrid  = [int(self.localMMATileGrid[0] / self.subtileShape[0]),
                                 int(self.localMMATileGrid[1] / self.subtileShape[1])]
      self.subtileSize       = gr_cfg.subtileSizeBytes()

      # Cooperative GR load counts (scheduler: vmcnt, loop trip count).
      # loadRatioGR uses the global GR tile size (subtileShape * subtileCount),
      # which is the full hardware granularity for one cooperative load round.
      _grBytesPerLoad      = gr_cfg.bytesPerLoad(self.numWaves)
      _globalGRTileSize    = self.subtileSize * (int(self.subtileCount) if self.subtileCount else 1)
      self.loadRatioGR     = _grBytesPerLoad / _globalGRTileSize if _globalGRTileSize else 0
      self.numGRPerSubtile = int(math.ceil(1.0 / self.loadRatioGR)) if self.loadRatioGR else 0
      self.numGRTotal      = int(self.localSubtileGrid[0] * self.localSubtileGrid[1] / self.loadRatioGR) if self.loadRatioGR else 0

      # LR subtile grid — used by LR emit dispatch (may differ from GR)
      self.lrGlobalSubtileGrid = list(lr_cfg.globalSubtileGrid(self.macroTile, self.depthU))
      self.lrSubtileSize       = lr_cfg.subtileSizeBytes()
      self.lrSubtileShape      = list(lr_cfg.subtileShape)
      self.lrLocalSubtileGrid  = list(self.localSubtileGrid)  # AB: LR iterates over GR subtile grid
      # LR load counts: one ds_read per wave covers loadWidthLR * waveSize bytes
      _lrBytesPerLoad      = geometry.lr.loadWidth * self.waveSize
      self.loadRatioLR     = _lrBytesPerLoad / self.lrSubtileSize if self.lrSubtileSize else 0
      self.numLRPerSubtile = int(math.ceil(1.0 / self.loadRatioLR)) if self.loadRatioLR else 0

      # Derived byte-counts for emit logic
      self.depthUBytes   = int(self.depthU * geometry.bpe)
      self.subIterKBytes = self.depthUBytes // self.localSubtileGrid[1]
      # TDM path. We apply 16 Bytes padding to each row.
      # TDM only exists on gfx1250, which is never swizzled (gfx950-only).
      isTDM = kernel.get("enableTDM%s" % tc, False)
      self.ldsRowPadBytes = 16 if isTDM else 0

      # Convenience counts for scheduler / diagram
      self.mmaTileLocalTotalCount = self.localMMATileGrid[0] * self.localMMATileGrid[1]
      self.grGlobalSubtileGrid  = self.globalSubtileGrid           # GR subtile grid alias
      self.grSubtileTotalCount  = int(self.globalSubtileGrid[0] * self.globalSubtileGrid[1])
      self.grSubtileSizeBytes   = self.subtileSize
      self.grBytesPerLoad       = int(gr_cfg.bytesPerLoad(self.numWaves))
      self.grLoadsPerSubtile    = self.numGRPerSubtile
      self.lrSubtileTotalCount  = int(self.lrGlobalSubtileGrid[0] * self.lrGlobalSubtileGrid[1])
      self.lrSubtileSizeBytes   = self.lrSubtileSize
      self.subtileTotalCount    = self.grSubtileTotalCount
      self.subtileSizeBytes     = self.subtileSize
      self.bytesPerLoad         = self.grBytesPerLoad
      self.loadsPerSubtile      = self.numGRPerSubtile

    elif isinstance(geometry, MXScaleTilePair):
      gr_cfg = geometry.gr.for_kernel(kernel, _tc)
      lr_cfg = geometry.lr
      self.gr = None
      self.lr = None
      self.globalMMATileGrid   = list(gr_cfg.globalMMATileGrid(self.macroTile, self.depthU))
      self.localMMATileGrid    = [self.globalMMATileGrid[0] // self.waveGroupSize, self.globalMMATileGrid[1]]
      self.subtileShape          = list(gr_cfg.subtileShape)
      self.subtileShape        = self.subtileShape
      self.globalSubtileGrid   = [1, 1]  # all waves load the full scale tile in one round
      self.localSubtileGrid    = [1, 1]
      self.subtileSize         = lr_cfg.subtileSizeBytes() // lr_cfg.subtileShape[0]
      self.lrGlobalSubtileGrid = list(lr_cfg.globalSubtileGrid(self.macroTile, self.depthU))
      self.lrSubtileSize       = lr_cfg.subtileSizeBytes()
      self.lrSubtileShape      = list(lr_cfg.subtileShape)
      self.lrLocalSubtileGrid  = [int(self.localMMATileGrid[0] / lr_cfg.subtileShape[0]),
                                   int(self.localMMATileGrid[1] / lr_cfg.subtileShape[1])]
      self.loadRatioGR         = 0
      self.numGRPerSubtile     = 1
      self.numGRTotal          = 1  # one buffer_load covers the full scale tile grid
      self.grBytesPerLoad      = gr_cfg.subtileShape[0] * gr_cfg.subtileShape[1] * gr_cfg.mmaTileSize
      self._mxBlock            = geometry.gr.scaleLayout.mxBlock

      _lrBytesPerLoad          = geometry.lr.loadWidth * self.waveSize
      self.loadRatioLR         = _lrBytesPerLoad / self.lrSubtileSize if self.lrSubtileSize else 0

    elif isinstance(geometry, CDTileGeometry):
      self.gr = None
      self.lr = None
      self.globalMMATileGrid = list(geometry.globalMMATileGrid(self.macroTile0, self.macroTile1))
      self.localMMATileGrid  = list(geometry.localMMATileGrid(self.macroTile0, self.macroTile1, self.waveGroup))
      self.subtileShape      = list(geometry.subtileShape)
      self.globalSubtileGrid = list(geometry.globalSubtileGrid(self.macroTile0, self.macroTile1, geometry.subtileShape))
      self.localSubtileGrid  = list(geometry.localSubtileGrid(self.macroTile0, self.macroTile1, self.waveGroup, geometry.subtileShape))
      self.subtileSize       = geometry.subtileShape[0] * geometry.subtileShape[1] * geometry.mmaTileSize
      self.subtileLocalTotalCount = self.localSubtileGrid[0] * self.localSubtileGrid[1]

    # --- Convenience accessors delegated to geometry ---
    self.bpe = geometry.bpe
    self.mmaTileShape = list(geometry.mmaTileShape)
    self.mmaTileSize = geometry.mmaTileSize
    self.mmaTileRegCount = geometry.mmaTileRegCount
    if isinstance(geometry, ABTilePair):
      self.loadWidthGR = geometry.gr.loadWidth
      self.loadWidthLR = geometry.lr.loadWidth
    elif isinstance(geometry, MXScaleTilePair):
      self.loadWidthGR = geometry.gr.loadWidth
      self.loadWidthLR = geometry.lr.loadWidth
    elif isinstance(geometry, CDTileGeometry):
      self.loadWidthGR = int(geometry.storeShape.m * geometry.storeShape.k * geometry.bpe)
      self.loadWidthLR = self.loadWidthGR

    # --- Mutable register state (filled by allocOffsetRegisters) ---
    self.localSubtilesRegister: List = []

    # --- Consistency checks ---
    if isinstance(geometry, ABTilePair):
      gr_cfg = self.gr.config
      lr_cfg = self.lr.config
      mmaM, mmaK = geometry.mmaTileShape
      self._check_dim(self.macroTile, gr_cfg.subtileShape[0] * mmaM, self.globalSubtileGrid[0], self.waveGroupSize, 'macroTile[GR]')
      self._check_dim(self.depthU,    gr_cfg.subtileShape[1] * mmaK, self.globalSubtileGrid[1], 1,                 'depthU[GR]')
      self._check_dim(self.macroTile, lr_cfg.subtileShape[0] * mmaM, self.lrGlobalSubtileGrid[0], self.waveGroupSize, 'macroTile[LR]')
      self._check_dim(self.depthU,    lr_cfg.subtileShape[1] * mmaK, self.lrGlobalSubtileGrid[1], 1,                 'depthU[LR]')
    elif isinstance(geometry, MXScaleTilePair):
      # GR covers the full scale MMA tile grid (subtileShape = entire grid, globalSubtileGrid=[1,1])
      # LR uses subtileShape; check coverage in scale MMA tile units.
      mmaM, mmaK = geometry.mmaTileShape
      lr_st = geometry.lr.subtileShape
      scale_K_tiles = self.depthU // geometry.instK  # data depthU → scale MMA K tile count
      self._check_dim(self.macroTile // mmaM, lr_st[0], self.lrGlobalSubtileGrid[0], self.waveGroupSize, 'macroTile[LR]')
      self._check_dim(scale_K_tiles,          lr_st[1], self.lrGlobalSubtileGrid[1], 1,                 'depthU[LR]')
    elif isinstance(geometry, CDTileGeometry):
      st = geometry.subtileShape
      mmaM, mmaN = geometry.mmaTileShape
      wg0, wg1 = self.waveGroup
      self._check_dim(self.macroTile0, st[0] * mmaM, self.globalSubtileGrid[0], wg0, 'macroTile0')
      self._check_dim(self.macroTile1, st[1] * mmaN, self.globalSubtileGrid[1], wg1, 'macroTile1')

  # --- Consistency validation ---

  def _check_dim(self, mt, subtile_span, num_subtiles, wg_size, label):
    """Verify subtile_span * num_subtiles * wg_size == mt for one tile dimension.

    subtile_span : elements covered by one subtile in this dim
    num_subtiles : globalSubtileGrid value for this dim (may be float)
    wg_size      : wave group count partitioning this dim (1 = no partitioning)
    """
    if subtile_span * num_subtiles != mt:
      raise ValueError(
        f"TileInfo({self.tc}): {label}={mt} not covered exactly. "
        f"subtile_span({subtile_span}) x globalSubtileGrid({num_subtiles}) "
        f"= {subtile_span * num_subtiles} (expected {mt}). "
        f"Minimum {label} = subtile_span({subtile_span}) x waveGroupSize({wg_size}) = {subtile_span * wg_size}."
      )

  # --- Grid utility methods ---

  def getLocalSubtileLinearId(self, sId0, sId1):
    return sId1 * self.localSubtileGrid[0] + sId0

  def getLocalSubtileIdFromLinearId(self, linearId):
    sId0 = linearId % self.localSubtileGrid[0]
    sId1 = linearId // self.localSubtileGrid[0]
    return [sId0, sId1]

  def getLocalMMATileLinearId(self, mmaId0, mmaId1):
    return mmaId1 * self.localMMATileGrid[0] + mmaId0

  def getLocalSubtileIdFromMMATile(self, mmaId0, mmaId1):
    st = self.subtileShape
    return [mmaId0 // st[0], mmaId1 // st[1]]

  def getSubtileShapeLinearId(self, k0, k1):
    st = self.subtileShape
    return k1 * st[0] + k0

  # --- Tile index mappings ---

  def grLoadIndexForSubtile(self, sId0, sId1, loadIdx=0):
    """Return the global-read load index for subtile (sId0, sId1).

    Each GR load is a buffer_load that fetches one tile-shaped region from
    global memory.  When loadRatioGR > 1, multiple subtiles share a single
    GR load (they are serviced by the same buffer_load instruction).
    loadIdx selects among the numGRPerSubtile loads within that subtile.
    """
    linearId = self.getLocalSubtileLinearId(sId0, sId1)
    baseGR = int(linearId // self.loadRatioGR) if self.loadRatioGR else 0
    return baseGR + loadIdx

  def lrTileIndexForSubtile(self, sId0, sId1, mfmaId=0):
    """Return the local-read vgprTile index for subtile (sId0, sId1).

    Each vgprTile is a ds_read destination register group that feeds one
    MFMA instruction.  mfmaId selects among the MMA tiles within the
    subtile (0 .. lrSubtileShape[0]*lrSubtileShape[1]-1).

    The base index is linearId * tilesPerSubtile, where tilesPerSubtile
    is the total number of MMA tiles in the LR subtile (M * K).
    Uses lrLocalSubtileGrid for linearization — for AB this equals
    localSubtileGrid; for scale it is derived from lrSubtileShape.
    """
    linearId = sId1 * self.lrLocalSubtileGrid[0] + sId0
    tilesPerSubtile = int(self.lrSubtileShape[0]) * int(self.lrSubtileShape[1])
    return linearId * tilesPerSubtile + mfmaId

  def globalMmaTilesForSubtile(self, sId0, sId1):
    """Return all global MMA tile coordinates belonging to subtile (sId0, sId1).

    Uses gr_cfg.subtileForMmaTile to account for subtileCount/subtileStride.
    The returned list is in geometric order (M-outer, K-inner).
    """
    st = self.subtileShape
    baseRow = sId0 * int(st[0])
    baseCol = sId1 * int(st[1])
    _, _, mma_tiles = self.gr.config.subtileForMmaTile(baseRow, baseCol)
    return mma_tiles

  def waveMmaTilesForSubtile(self, sId0, sId1):
    """Return the local MMA tile coordinates this wave uses from subtile (sId0, sId1).

    Each wave covers localMMATileGrid rows; within that, subtile (sId0, sId1)
    spans subtileShape[0] rows x subtileShape[1] columns of MMA tiles.
    """
    st = self.subtileShape
    baseRow = sId0 * int(st[0])
    baseCol = sId1 * int(st[1])
    return [(baseRow + m, baseCol + k)
            for m in range(int(st[0]))
            for k in range(int(st[1]))]

  def grRegGroupForSubtileRow(self, sId0):
    """Return the GR offset-register group index for subtile row sId0.

    Offset VGPRs for GR buffer_loads are stored in localSubtilesRegister,
    grouped by load.  When loadRatioGR >= 2, multiple subtile rows share
    the same buffer_load and therefore the same register group.
    """
    if self.loadRatioGR >= 2.0:
      return int(sId0 // self.loadRatioGR)
    return sId0

  # --- Register allocation ---

  def allocOffsetRegisters(self, writer, kernel):
    if self.gr is not None:
      self.gr.allocOffsetRegisters(self, writer, kernel)
    if self.lr is not None:
      self.lr.allocOffsetRegisters(self, writer, kernel)
    # MXScaleTilePair offset registers
    # should be managed by scale-specific alloc in SubtileScaleEmit.py
    if isinstance(self.geometry, MXScaleTilePair):
      self._sharedVgprGROffset = [writer.vgprPool.checkOut(1, tag="allocOffsetRegisters_sharedVgprGROffset")]
      self._sharedVgprLROffset = [writer.vgprPool.checkOut(1, tag="allocOffsetRegisters_sharedVgprLROffset")]
      self._sharedVgprLROffsetSwap = [writer.vgprPool.checkOut(1, tag="allocOffsetRegisters_sharedVgprLROffsetSwap")]

  def allocVgprTileRegisters_legacy(self, writer, kernel):
    """Allocate data tile registers for A/B/D MMA operands.
    """
    self.vgprTiles = []
    numMMATiles = int(self.localMMATileGrid[0] * self.localMMATileGrid[1])
    numMMATilesPerReg = max(1, int(1 // self.mmaTileRegCount))
    # Scale tiles: legacy MXSA/MXSB used bpe=1 (scale byte) which gives mmaTileRegCount=0.25
    # and numMMATilesPerReg=4. TileInfo uses data bpe (0.5 for f4), halving mmaTileRegCount.
    # The scale emit code (SubtileScaleEmit.py) uses stride 4 to index vgprTiles, so override.
    if isinstance(self.geometry, MXScaleTilePair):
      numMMATilesPerReg = 4
    numDword = int(math.ceil(self.mmaTileRegCount))

    isDTile = isinstance(self.geometry, CDTileGeometry)
    maxAgpr = writer.states.regCaps["PhysicalMaxVgpr"] - writer.states.regCaps["MaxVgpr"] if isDTile else 0

    for i in range(numMMATiles):
      if isDTile and writer.agprPool.size() < maxAgpr:
        pool = writer.agprPool
        regType = RegisterType.Accvgpr
      else:
        pool = writer.vgprPool
        regType = RegisterType.Vgpr
      self.vgprTiles.append(RegisterTileInfo(pool, regType))
      if i % numMMATilesPerReg != 0:
        continue
      vstart = pool.checkOutAligned(numDword, numDword, tag="allocVgprTileRegisters_legacy_vstart")
      for k in range(numDword):
        self.vgprTiles[-1].append(vstart + k)

  def deallocOffsetRegisters(self, writer, kernel):
    if self.gr is not None:
      self.gr.deallocOffsetRegisters(self, writer, kernel)
    if self.lr is not None:
      self.lr.deallocOffsetRegisters(self, writer, kernel)
    # MXScaleTilePair dealloc
    for attr in ('_sharedVgprGROffset', '_sharedVgprLROffset', '_sharedVgprLROffsetSwap'):
      for v in getattr(self, attr, []):
        writer.vgprPool.checkIn(v)
      if hasattr(self, attr):
        setattr(self, attr, [])

  # --- Emit dispatchers ---
  # For ABTilePair: route through self.gr / self.lr (mutable runtime tiles).
  # For MXScale/CD: route directly to geometry (still frozen with emit stubs).

  def emitGlobalReadOffset(self, writer, kernel):
    if self.gr is not None:
      return self.gr.emitGlobalReadOffset(self, writer, kernel)
    return self.geometry.emitGlobalReadOffset(self, writer, kernel)

  def emitGlobalRead(self, writer, kernel):
    if self.gr is not None:
      return self.gr.emitGlobalRead(self, writer, kernel)
    return self.geometry.emitGlobalRead(self, writer, kernel)

  def emitLocalWrite(self, writer, kernel):
    if self.gr is not None:
      return self.gr.emitLocalWrite(self, writer, kernel)
    return self.geometry.emitLocalWrite(self, writer, kernel)

  def emitLocalReadOffset(self, writer, kernel):
    if self.lr is not None:
      return self.lr.emitLocalReadOffset(self, writer, kernel)
    return self.geometry.emitLocalReadOffset(self, writer, kernel)

  def emitLocalRead(self, writer, kernel):
    if self.lr is not None:
      return self.lr.emitLocalRead(self, writer, kernel)
    return self.geometry.emitLocalRead(self, writer, kernel)

  def emitLRDTLInit(self, writer, kernel):
    if self.lr is not None:
      return self.lr.emitDTLInit(self, writer, kernel)
    return Module()

  def emitLRLDSBufferSwap(self, writer, kernel):
    if self.lr is not None:
      return self.lr.emitLDSBufferSwap(self, writer, kernel)
    return Module()

  def emitGRDTLInit(self, writer, kernel):
    if self.gr is not None:
      return self.gr.emitDTLInit(self, writer, kernel)
    return Module()

  def emitGRLDSBufferSwap(self, writer, kernel):
    if self.gr is not None:
      return self.gr.emitLDSBufferSwap(self, writer, kernel)
    return Module()

  def emitGRPtrUpdate(self, writer, kernel):
    if self.gr is not None:
      return self.gr.emitPtrUpdate(self, writer, kernel)
    return Module()

  def emitStoreD(self, writer, kernel):
    return self.geometry.emitStoreD(self, writer, kernel)

  def emitLoadC(self, writer, kernel):
    return self.geometry.emitLoadC(self, writer, kernel)

  # --- Register accessors ---
  # Uniform interface for emit code to access offset registers.
  # GR registers: sharedVgprGROffset (per-lane byte offsets) live on ABGRTile;
  #               localSubtilesRegister (per-subtile-row soffsets) live on TileInfo.
  # LR registers: sharedVgprLROffset/Swap live on ABLRTile.
  # MXScale registers: _sharedVgpr* live directly on TileInfo (no tile object).

  @property
  def sharedVgprGROffset(self):
    if self.gr: return self.gr.sharedVgprGROffset
    return getattr(self, '_sharedVgprGROffset', [])

  @property
  def sharedVgprLROffset(self):
    if self.lr: return self.lr.sharedVgprLROffset
    return getattr(self, '_sharedVgprLROffset', [])

  @property
  def sharedVgprLROffsetSwap(self):
    if self.lr: return self.lr.sharedVgprLROffsetSwap
    return getattr(self, '_sharedVgprLROffsetSwap', [])

  def grOffsetVgpr(self, idx: int) -> int:
    """VGPR holding per-lane GR byte offset for load `idx` within a subtile."""
    return self.sharedVgprGROffset[idx]

  def grSubtileRegList(self, rowIdx: int):
    """RegList for subtile row `rowIdx` (soffset registers in M dimension).
    Row 0 returns an empty RegList (soffset=0)."""
    return self.localSubtilesRegister[rowIdx]

  def lrOffsetVgpr(self, idx: int) -> int:
    """VGPR holding per-lane LR byte offset for ds_read `idx` within a subtile."""
    return self.sharedVgprLROffset[idx]

  def lrSwapVgpr(self, idx: int) -> int:
    """VGPR holding the double-buffer swap offset for LR load `idx`."""
    return self.sharedVgprLROffsetSwap[idx]

  @property
  def numGROffsetVgprs(self) -> int:
    return len(self.sharedVgprGROffset)

  @property
  def numSubtileRows(self) -> int:
    """Number of perpendicular (M) subtile rows with distinct soffsets."""
    return len(self.localSubtilesRegister)

  @property
  def numLROffsetVgprs(self) -> int:
    return len(self.sharedVgprLROffset)

  @property
  def localSubtiles(self):
    """Empty — SubtileInfo objects not used by TileInfo emit."""
    return []

  @property
  def mxBlock(self):
    """Scale mxBlock from geometry."""
    return getattr(self, '_mxBlock', 0)

  @property
  def numLRTotal(self):
    return int((self.localSubtileGrid[0] * self.localSubtileGrid[1]) / self.loadRatioLR) if self.loadRatioLR else 0

  @property
  def vgprTileFactor(self):
    return 1.0

  def deallocVgprTileRegisters_legacy(self, writer, kernel):
    """Deallocate vgprTiles.
    TODO: Remove after full migration — temporary port from legacy TileInfo."""
    numMMATilesPerReg = max(1, int(1 // self.mmaTileRegCount))
    if isinstance(self.geometry, MXScaleTilePair):
      numMMATilesPerReg = 4  # mirror allocVgprTileRegisters_legacy override for scale tiles
    for i, vtiles in enumerate(self.vgprTiles):
      if i % numMMATilesPerReg != 0:
        continue
      if vtiles.regList.indices:
        vtiles.regList.pool.checkIn(vtiles.regList.indices[0])
    self.vgprTiles = []

  def __str__(self):
    return (f"TileInfo(tc={self.tc}, geometry={type(self.geometry).__name__}, "
            f"mmaTileShape={self.mmaTileShape}, "
            f"localMMATileGrid={self.localMMATileGrid}, "
            f"localSubtileGrid={self.localSubtileGrid})")

################################################################################
# Legacy subtile-based kernel classes (incremental migration in progress)
################################################################################

class RegisterTileInfo:
  """Holds a list of register indices for a single MMA tile slot."""
  tileSize: int = 0

  def __init__(self, pool, regType=RegisterType.Vgpr):
    self.regList = RegList(pool, regType)

  def append(self, val):
    self.regList.append(val)

  def index(self, val):
    return self.regList.index(val)

  def __iter__(self):
    for vals in self.regList:
      yield vals

  def __str__(self):
    return str(self.regList)


def _zeroRegRange(module, writer, tileInfo, firstReg, totalRegs, isAgpr):
  """Zero a contiguous register range using MFMA (16/inst) or WMMA (8/inst)."""
  useWmma = writer.states.asmCaps.get("HasWMMA_AccImmZero", False)
  tileAlias = vgpr if useWmma else (accvgpr if isAgpr else vgpr)
  tileCopyInst = VMovB32 if useWmma else (VAccvgprWrite if isAgpr else VMovB32)

  if useWmma:
    regsPerInst = 8
    instType, accType = InstType.INST_F32, InstType.INST_F32
    variant = [16, 16, 4, 1]
    acc2_kwargs = {"acc2_imm": 0}
  else:
    regsPerInst = 16
    instType, accType = InstType.INST_I8, InstType.INST_I32
    variant = [32, 32, 16, 1]
    acc2_kwargs = {"acc2": 0}

  numInst = totalRegs // regsPerInst

  if numInst > 0:
    tmpVgpr = writer.vgprPool.checkOutAligned(2, 2, tag="zeroRegRange_tmpVgpr")
    module.add(VMovB64(dst=vgpr(tmpVgpr, 2), src=0, comment="zero A/B"))
    module.add(SNop(waitState=1, comment="wait for vgpr before matrix inst"))
    for i in range(numInst):
      r = firstReg + i * regsPerInst
      module.add(MFMAInstruction(instType=instType, accType=accType,
                                 variant=variant, mfma1k=False,
                                 acc=tileAlias(r, regsPerInst),
                                 a=vgpr(tmpVgpr, 2), b=vgpr(tmpVgpr, 2),
                                 **acc2_kwargs,
                                 comment="init%s: [%u:%u]"%(tileInfo.tc, r, r + regsPerInst - 1)))
    writer.vgprPool.checkIn(tmpVgpr)

  for i in range(numInst * regsPerInst, totalRegs):
    module.add(tileCopyInst(dst=tileAlias(firstReg + i), src=0, comment="init%s"%(tileInfo.tc)))

def initVgprTilesToZero(writer, kernel, tileInfo):
  """Initialize vgprTiles to zero using MFMA for blocks of 16, scalar writes for remainder."""
  module = Module()
  module.addComment0("Init %s vgprTiles to zero"%(tileInfo.tc))

  if not tileInfo.vgprTiles:
    return module

  # Group contiguous tiles by pool type (agpr vs vgpr) since D tiles can use both
  firstReg = tileInfo.vgprTiles[0].regList.indices[0]
  totalRegs = 0
  curPool = tileInfo.vgprTiles[0].regList.pool

  for tile in tileInfo.vgprTiles:
    pool = tile.regList.pool
    numRegs = len(tile.regList.indices)
    if pool != curPool:
      _zeroRegRange(module, writer, tileInfo, firstReg, totalRegs, curPool == writer.agprPool)
      firstReg = tile.regList.indices[0]
      totalRegs = numRegs
      curPool = pool
    else:
      totalRegs += numRegs

  _zeroRegRange(module, writer, tileInfo, firstReg, totalRegs, curPool == writer.agprPool)

  return module

# ---------------------------------------------------------------------------
# Pick the MXMFMAInstruction instType for the V_MFMA_SCALE_F32_<MxNxK>_F8F6F4
# family from kernel data types.
#
# The CBSZ/BLGP fields:
#       000 E4M3 (FP8)        010 E2M3 (FP6)        100 E2M1 (FP4)
#       001 E5M2 (BF8)        011 E3M2 (BF6)
#
# Returns None when DataType{A,B} aren't populated
# ---------------------------------------------------------------------------
def _selectF8F6F4InstType(kernel):
  pt = kernel.get("ProblemType")
  if pt is None:
    return None

  aType = pt.get("DataTypeA")
  bType = pt.get("DataTypeB")
  if aType is None or bType is None:
    raise RuntimeError(f"Unsupported data types for MFMA instruction: A = {aType}, B = {bType}\n")

  sourceSwap = bool(kernel.get("SourceSwap", False))
  if sourceSwap:
    aType, bType = bType, aType

  # Defensive: support MagicMock / minimal stubs that don't define predicates.
  def _pred(t, name):
    fn = getattr(t, name, None)
    return bool(fn()) if callable(fn) else False

  # Pure types
  aIsF8  = _pred(aType, "isFloat8")
  bIsF8  = _pred(bType, "isFloat8")
  if aIsF8 and bIsF8:
    return InstType.INST_F8

  aIsBF8 = _pred(aType, "isBFloat8")
  bIsBF8 = _pred(bType, "isBFloat8")
  if aIsBF8 and bIsBF8:
    return InstType.INST_BF8

  aIsF4  = _pred(aType, "isFloat4")
  bIsF4  = _pred(bType, "isFloat4")
  if aIsF4 and bIsF4:
    return InstType.INST_F4

  # Mixed FP8/BF8 (8-bit only)
  if aIsF8 and bIsBF8:
    return InstType.INST_F8_BF8

  if aIsBF8 and bIsF8:
    return InstType.INST_BF8_F8

  # Mixed F8 and F4
  if aIsF8 and bIsF4:
    return InstType.INST_F8_F4

  if aIsF4 and bIsF8:
    return InstType.INST_F4_F8

  # Mixed BF8 and F4
  if aIsBF8 and bIsF4:
    return InstType.INST_B8_F4

  if aIsF4 and bIsBF8:
    return InstType.INST_F4_B8

  raise RuntimeError(f"Unsupported data types for MFMA instruction: A = {aType}, B = {bType}\n")


##################################################
# Subroutine to generate MMA Instruction
# Given RegisterTileInfo inputs for A,B,C,D operands
# emit corresponding mfma instruction
#
def emitMfmaInstruction(writer, kernel, vgprTileA, vgprTileB, vgprTileC, vgprTileD, scaleAVgpr=-1, scaleBVgpr=-1, scaleAsel=-1, scaleBsel=-1, comment = ""):
  module = Module()

  vgprAStart = vgprTileA.regList.indices[0]
  vgprBStart = vgprTileB.regList.indices[0]
  vgprCStart = vgprTileC.regList.indices[0]
  vgprDStart = vgprTileD.regList.indices[0]

  opASize = len(vgprTileA.regList.indices)
  opBSize = len(vgprTileB.regList.indices)
  opCSize = len(vgprTileC.regList.indices)
  opDSize = len(vgprTileD.regList.indices)

  # For subtile kernels with agpr overflow, D/C tiles that spilled to the vgpr
  # pool must use vgpr() in the MFMA operands, not accvgpr().
  dIsVgpr = (vgprTileD.regList.pool == writer.vgprPool)
  cIsVgpr = (vgprTileC.regList.pool == writer.vgprPool)
  dAccAlias = vgpr if (dIsVgpr or kernel["MIArchVgpr"]) else accvgpr
  cAccAlias = vgpr if (cIsVgpr or kernel["MIArchVgpr"]) else accvgpr

  aOperand = vgpr(vgprBStart,opBSize) if kernel["SourceSwap"] else vgpr(vgprAStart,opASize)
  bOperand = vgpr(vgprAStart,opASize) if kernel["SourceSwap"] else vgpr(vgprBStart,opBSize)

  miK = kernel["MatrixInstK"]

  if miK == 128:
    # MX FP4: 16x16x128
    mxInstType = _selectF8F6F4InstType(kernel)
    if scaleAVgpr >= 0 and scaleBVgpr >= 0:
      # Use actual loaded scale VGPRs
      module.add(MXMFMAInstruction(instType=mxInstType, accType=InstType.INST_F32, variant=[16,16,miK,1], \
                                   acc=dAccAlias(vgprDStart,opDSize), \
                                   a=aOperand, \
                                   b=bOperand, \
                                   acc2=cAccAlias(vgprCStart,opCSize), \
                                   mxsa=vgpr(scaleAVgpr), mxsb=vgpr(scaleBVgpr), \
                                   vop3=VOP3PModifiers(op_sel=[scaleAsel%2, scaleBsel%2], op_sel_hi=[(scaleAsel>>1)%2, (scaleBsel>>1)%2]), \
                                   comment=comment))
    else:
      # Fallback: use unit scale VGPR pre-initialized to 0x7f7f7f7f (scale=1.0 E8M0).
      # Initialized once in mainLoop() before emitMainAndExitLoops() — VMovB32 cannot live here
      # because InstructionScheduler drops non-MFMA instructions from the MFMA module.
      unitScaleVgpr = kernel.get("_subtileUnitScaleVgpr", -1)
      assert unitScaleVgpr >= 0, \
          "emitMfmaInstruction: plain FP8 fallback requires _subtileUnitScaleVgpr in kernel dict"
      module.add(MXMFMAInstruction(instType=mxInstType, accType=InstType.INST_F32, variant=[16,16,miK,1], \
                                   acc=dAccAlias(vgprDStart,opDSize), \
                                   a=aOperand, \
                                   b=bOperand, \
                                   acc2=cAccAlias(vgprCStart,opCSize), \
                                   mxsa=vgpr(unitScaleVgpr), mxsb=vgpr(unitScaleVgpr), \
                                   comment=comment))
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

  # Legacy path (commented out):
  # atileInfo = writer.states.a.tileInfo
  # btileInfo = writer.states.b.tileInfo
  # dtileInfo = writer.states.d.tileInfo
  # mxsatileInfo = writer.states.mxsa.tileInfo if kernel["ProblemType"].get("MXBlockA", 0) > 0 else None
  # mxsbtileInfo = writer.states.mxsb.tileInfo if kernel["ProblemType"].get("MXBlockB", 0) > 0 else None
  # hasScaleA = mxsatileInfo is not None and mxsatileInfo.mxBlock > 0
  # hasScaleB = mxsbtileInfo is not None and mxsbtileInfo.mxBlock > 0

  tiA = writer.states.a.tileInfo
  tiB = writer.states.b.tileInfo
  dtileInfo = writer.states.d.tileInfo  # D has no TileInfo yet
  tiMXSA = writer.states.mxsa.tileInfo if kernel["ProblemType"].get("MXBlockA", 0) > 0 else None
  tiMXSB = writer.states.mxsb.tileInfo if kernel["ProblemType"].get("MXBlockB", 0) > 0 else None

  # Use loaded scale VGPRs when MX block scaling is active.
  # Note: scaleVgprTiles is only populated by the scheduler path;
  # in the non-scheduler path we use vgprTiles (populated by localReadDoScaleSubtile).
  hasScaleA = tiMXSA is not None and tiMXSA.mxBlock > 0
  hasScaleB = tiMXSB is not None and tiMXSB.mxBlock > 0

  # LR subtile shape governs the MFMA register layout (always (1,2) for current geometries).
  # Use ti.lr.subtileShape rather than ti.subtileShape (= GR subtileShape, which differs for
  # asymmetric WGs where waves_coop >= 4 expands subtileShape to (2,2)).
  lrSubtileShapeA = tiA.lr.subtileShape
  lrSubtileShapeB = tiB.lr.subtileShape

  for mmak in range(tiA.localMMATileGrid[1]):
    for mma1 in range(tiB.localMMATileGrid[0]):
      for mma0 in range(tiA.localMMATileGrid[0]):

        aSId0, aSId1 = mma0 // lrSubtileShapeA[0], mmak // lrSubtileShapeA[1]
        bSId0, bSId1 = mma1 // lrSubtileShapeB[0], mmak // lrSubtileShapeB[1]
        _mma0 = mma0 % lrSubtileShapeA[0]
        _mma1 = mma1 % lrSubtileShapeB[0]
        _mmak = mmak % lrSubtileShapeA[1]

        numMmaTilePerSubtileA = lrSubtileShapeA[0] * lrSubtileShapeA[1]
        numMmaTilePerSubtileB = lrSubtileShapeB[0] * lrSubtileShapeB[1]

        lrLocalGridA0 = tiA.localMMATileGrid[0] // lrSubtileShapeA[0]
        lrLocalGridB0 = tiB.localMMATileGrid[0] // lrSubtileShapeB[0]
        atileId = (aSId1 * lrLocalGridA0 + aSId0) * numMmaTilePerSubtileA + (_mmak)
        btileId = (bSId1 * lrLocalGridB0 + bSId0) * numMmaTilePerSubtileB + (_mmak)

        atiles = tiA.vgprTiles[atileId]
        btiles = tiB.vgprTiles[btileId]
        dtiles = dtileInfo.vgprTiles[mma0 + mma1 * dtileInfo.localMMATileGrid[0]]

        if hasScaleA:
          # Scale group index: one VGPR per lrSubtileShape[0] M-tiles x lrSubtileShape[1] K-tiles
          scaleMShapeA = tiMXSA.lrSubtileShape[0]
          scaleMShapeB = tiMXSB.lrSubtileShape[0]
          scaleKShapeA = tiMXSA.lrSubtileShape[1]
          scaleKShapeB = tiMXSB.lrSubtileShape[1]
          # Use the scale's own K LR subtile grid (not the data's K subtile grid).
          scaleKGridA = tiMXSA.lrLocalSubtileGrid[1]
          scaleKGridB = tiMXSB.lrLocalSubtileGrid[1]
          scaleGroupA = (mma0 // scaleMShapeA) * scaleKGridA + mmak // scaleKShapeA
          scaleGroupB = (mma1 // scaleMShapeB) * scaleKGridB + mmak // scaleKShapeB

          scaleAVgpr = tiMXSA.vgprTiles[4 * scaleGroupA].regList.indices[0] if tiMXSA.mxBlock else -1
          scaleBVgpr = tiMXSB.vgprTiles[4 * scaleGroupB].regList.indices[0] if tiMXSB.mxBlock else -1

          sAsel = (mma0 % scaleMShapeA) + scaleMShapeA * (mmak % scaleKShapeA)
          sBsel = (mma1 % scaleMShapeB) + scaleMShapeB * (mmak % scaleKShapeB)
        else:
          scaleAVgpr = -1
          scaleBVgpr = -1
          sAsel = sBsel = -1

        module.add(emitMfmaInstruction(writer, kernel, atiles, btiles, dtiles, dtiles,
                                       scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr, scaleAsel=sAsel, scaleBsel=sBsel,
                                       comment="Emit MMFA code for MMA tiles C[%u, %u] += A[%u, %u] * B[%u, %u] sA = %u, sB = %u"%(mma0, mma1, mma0, mmak, mmak, mma1, sAsel, sBsel)))

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
  tensorParametersA = writer.tPA
  tensorParametersB = writer.tPB

  pgr = kernel["PrefetchGlobalRead"]
  assert pgr in (0, 1, 2), "SubtileBasedKernel only supports PGR=0, PGR=1, and PGR=2, got PGR=%d" % pgr

  tiA = writer.states.a.tileInfo
  tiB = writer.states.b.tileInfo
  scaleTiA = writer.states.mxsa.tileInfo if kernel["ProblemType"].get("MXBlockA", 0) else None
  scaleTiB = writer.states.mxsb.tileInfo if kernel["ProblemType"].get("MXBlockB", 0) else None

  lrAGran = ReadGranularity(mn=1, k=1)
  lrBGran = ReadGranularity(mn=1, k=1)
  grMNA, grKA = tiA.subtileShape[0], tiA.subtileShape[1]
  grMNB, grKB = tiB.subtileShape[0], tiB.subtileShape[1]
  # TDM: one tensor_load_to_lds covers the full localMMATileGrid.
  if kernel.get("enableTDMA", False):
    grAGran = ReadGranularity(mn=tiA.localMMATileGrid[0], k=tiA.localMMATileGrid[1])
  else:
    grAGran = ReadGranularity(mn=grMNA, k=grKA) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2*grMNA, k=grKA)
  if kernel.get("enableTDMB", False):
    grBGran = ReadGranularity(mn=tiB.localMMATileGrid[0], k=tiB.localMMATileGrid[1])
  else:
    grBGran = ReadGranularity(mn=grMNB, k=grKB) if tiB.loadRatioGR <= 1.0 else ReadGranularity(mn=2*grMNB, k=grKB)
  lrSAGran = ReadGranularity(mn=scaleTiA.lrSubtileShape[0], k=scaleTiA.lrSubtileShape[1]) if scaleTiA else None
  lrSBGran = ReadGranularity(mn=scaleTiB.lrSubtileShape[0], k=scaleTiB.lrSubtileShape[1]) if scaleTiB else None
  grSAGran = ReadGranularity(mn=scaleTiA.localMMATileGrid[0], k=scaleTiA.localMMATileGrid[1]) if scaleTiA else None
  grSBGran = ReadGranularity(mn=scaleTiB.localMMATileGrid[0], k=scaleTiB.localMMATileGrid[1]) if scaleTiB else None

  schedulerPgr = pgr

  vgprBudget = writer.states.regCaps["MaxVgpr"]
  vgprUsed = writer.vgprPool.size() - writer.vgprPool.available()

  M = tiA.localMMATileGrid[0]
  N = tiB.localMMATileGrid[0]
  candidates = [(M, N)] if pgr == 0 else MFMASchedulerConfig.get_partition_candidates(tiA, tiB)
  for partSizeM, partSizeN in candidates:
      hasTDM = bool(kernel.get("enableTDMA")) and bool(kernel.get("enableTDMB"))
      grPlacement = (GRPlacementStrategy.BUNCHED if hasTDM
                     else GRPlacementStrategy.SPREAD)
      cfg = MFMASchedulerConfig(
          numMFMATilesM=M,
          numMFMATilesN=N,
          numSubIterK=tiA.localMMATileGrid[1],
          lrA=lrAGran,
          lrB=lrBGran,
          grA=grAGran,
          grB=grBGran,
          lrSA=lrSAGran,
          lrSB=lrSBGran,
          grSA=grSAGran,
          grSB=grSBGran,
          partitionSizeM=partSizeM,
          partitionSizeN=partSizeN,
          pgr=schedulerPgr,
          grPlacement=grPlacement,
      )

      scheduler = LogicalScheduler(cfg)
      scheduler.build()

      numVgpr = scheduler.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)
      if vgprUsed + numVgpr <= vgprBudget:
          break
  scheduler.allocVgprTiles(writer, tiA, tiB,
                           scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

  dtileInfo = writer.states.d.tileInfo

  # For plain FP8 (miK=128, no MX scale): allocate a unit scale VGPR and initialize
  # it once here, before the loop. emitMfmaInstruction will reference it via kernel dict.
  miK = kernel["MatrixInstK"]
  unitScaleVgpr = -1
  if miK == 128 and scaleTiA is None:
      unitScaleVgpr = writer.vgprPool.checkOut(1)
      module.add(VMovB32(dst=vgpr(unitScaleVgpr), src=hex(0x7f7f7f7f),
                         comment="unit scale=1.0 (E8M0) for plain FP8 MFMA"))
      kernel["_subtileUnitScaleVgpr"] = unitScaleVgpr

  scheduler.populate_instructions(
      writer, kernel,
      tileInfoA=tiA, tileInfoB=tiB, dtileInfo=dtileInfo,
      scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
      tensorParametersA=tensorParametersA,
      tensorParametersB=tensorParametersB)

  # gfx1250: enable expert scheduling mode and disable WMMA arb stall
  # before entering the mainloop / any wmma issue.
  if writer.states.archCaps.get("HasWmmaArbStallBit", False):
    module.add(SNop(waitState=0, comment="nop before SSetReg"))
    module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0, 2]),
                               src=2,
                               comment="enable expert scheduling mode"))
    module.add(SNop(waitState=0, comment="nop after SSetReg"))
    module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[4, 1]),
                               src=1,
                               comment="Disable WMMA arb stall"))
    module.add(SNop(waitState=0, comment="nop after SSetReg"))

  module.add(scheduler.emitMainAndExitLoops(writer, kernel, tensorParametersA, tensorParametersB))

  # gfx1250: disable expert scheduling mode after NLL.
  if writer.states.archCaps.get("HasWmmaArbStallBit", False):
    module.add(SNop(waitState=0, comment="nop before SSetReg"))
    module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0, 2]),
                               src=0,
                               comment="disable expert scheduling mode"))
    module.add(SNop(waitState=0, comment="nop after SSetReg"))

  # Wrap the tail loop with the runtime K%DU counter setup and skip branch,
  # mirroring the legacy KernelWriter pattern (KernelWriter.py:5237 / 5618).
  if not kernel["NoTailLoop"]:
    module.add(writer.calculateLoopNumIter(
        kernel, tensorParametersA, tensorParametersB, -1))
    # Tighten Srd{A,B}+2 OOB limit using the K remainder just computed
    # (no-op outside UseSubtileImpl A/B). Needed for bf16 (boundary DTL
    # load) and fp4 (regular tail-loop dwordx4 must see the actual K_rem
    # to avoid pulling stale OOB-zeroed dwords into LDS).
    module.add(writer.computeTailLoopSrdLimit(
        kernel, [tensorParametersA, tensorParametersB]))
    # MX scale operands: SrdMXS{A,B}+2 tightened with K_pad=256 (host scale
    # re-scatter granularity from DataInitialization.cpp::rearrangePaddedMXScaleLayout).
    # No-op when DepthU<=256 since host padding alone already covers K_rem.
    mxScaleTPs = []
    if kernel["ProblemType"].get("MXBlockA", 0) > 0 and "MX" in tensorParametersA:
      mxScaleTPs.append(tensorParametersA["MX"])
    if kernel["ProblemType"].get("MXBlockB", 0) > 0 and "MX" in tensorParametersB:
      mxScaleTPs.append(tensorParametersB["MX"])
    if mxScaleTPs:
      module.add(writer.computeTailLoopSrdLimit(kernel, mxScaleTPs))
    module.add(scheduler.emitTailLoop(writer, kernel))
    module.add(writer.closeLoop(
        kernel, tensorParametersA, tensorParametersB,
        -1, None, emitEndLabelOnly=True))

  scheduler.deallocVgprTiles(writer)

  if unitScaleVgpr >= 0:
      writer.vgprPool.checkIn(unitScaleVgpr)

  return module
