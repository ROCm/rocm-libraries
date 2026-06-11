# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import singledispatch, cached_property
from typing import Dict, List, NamedTuple, Optional, Tuple, Type
from Tensile.Components.Subtile.LogicalScheduler import (
      LogicalScheduler, SchedulerConfig as MFMASchedulerConfig,
      ReadGranularity)

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

################################################################################
# Geometry value/query layer — inlined from SubtileGeometry.py
# (hipblaslt_incremental_refactor-grc.137)
#
# The *pure geometry math* query methods below are serviced unconditionally by
# the compiled ``tensile_writer`` nanobind extension
# (tensile_writer.subtile.geometry). The Python dataclasses in this section are
# the public facade: they own the API objects (tags, ``replace()``,
# singledispatch dispatch) while every ported value/query method forwards to
# the matching C++ object built on demand. There is no pure-Python fallback —
# the geometry formulas live in C++ only.
#
# Only value/query math crosses into C++. No writer state, register allocation,
# rocisa emission, or main-loop logic is delegated.
################################################################################

from tensile_writer.subtile import geometry as _cppgeo


################################################################################
# C++-backed geometry value/query layer
#
# The *pure geometry math* query methods below are serviced unconditionally by
# the compiled ``tensile_writer`` nanobind extension
# (tensile_writer.subtile.geometry). The Python dataclasses in this module are
# the public facade: they own the API objects (tags, ``replace()``,
# singledispatch dispatch) while every ported value/query method forwards to
# the matching C++ object built on demand. There is no pure-Python fallback —
# the geometry formulas live in C++ only.
#
# Only value/query math crosses into C++. No writer state, register allocation,
# rocisa emission, or main-loop logic is delegated.
################################################################################

def _cpp_mma(layout: 'MMALayout'):
  return _cppgeo.MMALayout(layout.instM, layout.blocks, layout.vgprs, layout.waveSize)

def _cpp_scale(layout: 'MMAScaleLayout'):
  return _cppgeo.MMAScaleLayout(layout.instM, layout.blocks, layout.vgprs,
                                layout.mxBlock, layout.waveSize)

def _cpp_loadshape(ls: 'LoadShape'):
  return _cppgeo.LoadShape(ls.m, ls.k)


################################################################################
# RegList — typed register list
################################################################################

_REF_FN = {
    RegisterType.Sgpr:    sgpr,
    RegisterType.Vgpr:    vgpr,
    RegisterType.Accvgpr: accvgpr,
}

class RegList:
  """Typed register list — knows its register kind and how to reference them.

  Usage:
    rl = RegList(pool, RegisterType.Sgpr)
    rl.alloc()                     # checkout and track one register
    rl.append(pool.checkOut(1))    # track an externally checked-out register
    soffset = rl.ref(0)            # -> sgpr(idx), vgpr(idx), or accvgpr(idx)
    if rl.is_sgpr: ...             # type check
  """
  def __init__(self, pool, regType):
    self.pool = pool
    self.regType = regType
    self.indices = []
    self._ref = _REF_FN[regType]

  def alloc(self, count=1, **kwargs):
    """Checkout from pool and track the index."""
    idx = self.pool.checkOut(count, **kwargs)
    self.indices.append(idx)
    return idx

  def append(self, idx):
    """Track an externally checked-out register index."""
    self.indices.append(idx)

  def index(self, val):
    return self.indices.index(val)

  def ref(self, i=0):
    """Return the rocisa register container (sgpr/vgpr/accvgpr) for slot i."""
    return self._ref(self.indices[i])

  @property
  def is_sgpr(self):
    return self.regType == RegisterType.Sgpr

  @property
  def is_vgpr(self):
    return self.regType == RegisterType.Vgpr

  def dealloc(self):
    """Check all tracked registers back into the pool."""
    for idx in self.indices:
      self.pool.checkIn(idx)
    self.indices.clear()

  def __len__(self):
    return len(self.indices)

  def __iter__(self):
    return iter(self.indices)

  def __str__(self):
    return str(self.indices)


################################################################################
# MMA Layout — data-type independent MFMA/WMMA lane geometry
################################################################################

@dataclass(frozen=True)
class LoadShape:
  """Shape of a single load or store instruction in (non-K, K) element dimensions.

  Separates the 2D access pattern from the total width, making the contiguous
  direction explicit. Used for global read (GR) and local read/write (LR/LW).

    m: elements per instruction in the non-K dimension (M for A, N for B/C/D).
    k: elements per instruction in the K dimension.

  The instruction width in bits is derived as m * k * bpe * 8 (bpe from the
  owning geometry). For row-major data (K contiguous), k > 1 and m == 1.
  For column-major data (M contiguous), m > 1 and k == 1.
  """
  m: int
  k: int


@dataclass(frozen=True)
class MMALayout:
  """Data-type independent MMA (MFMA/WMMA) lane layout.

  Captures how lanes map to the non-K dimension of the MMA tile and
  how input data is packed into VGPRs. Instructions that share the
  same lane mapping but differ in VGPR packing (e.g. fp4 vs fp8 for
  16x16x128) have separate layout objects.

  Structural parameters from the ISA:
    instM:      Non-K output dimension (e.g. 16).
    blocks:     Independent MxN outputs per instruction (ISA "Blocks").
    vgprs:      VGPRs per lane for the operand.

  Derived:
    contiguousLanes:    Lanes on contiguous non-K elements (= instM).
    kGroups:            Lane groups for different K ranges within each
                        block (= waveSize / instM / blocks).
    elementsPerLaneNonK: Non-K elements per lane (= instM / kGroups).
  """
  instM: int
  blocks: int = -1
  vgprs: int = -1
  waveSize: int = -1

  # Derived attributes (computed in __post_init__)
  contiguousLanes: int = field(init=False)
  kGroups: int = field(init=False)
  elementsPerLaneNonK: int = field(init=False)

  def __post_init__(self):
    object.__setattr__(self, 'contiguousLanes', self.instM)
    object.__setattr__(self, 'kGroups', self.waveSize // (self.contiguousLanes * self.blocks))
    object.__setattr__(self, 'elementsPerLaneNonK', self.instM // self.kGroups)

  def inputBytesPerLane(self) -> int:
    """Input bytes per lane = vgprs * 4."""
    return _cpp_mma(self).inputBytesPerLane()

  def tileSizeBytes(self, instK: int, elementBytes: float) -> int:
    """Total tile size in bytes."""
    return _cpp_mma(self).tileSizeBytes(instK, elementBytes)

  def regsPerTile(self, instK: int, elementBytes: float) -> float:
    """VGPRs per lane (as float), matching TileInfo.mmaTileRegCount."""
    return _cpp_mma(self).regsPerTile(instK, elementBytes)

# Pre-defined immutable MMA layouts for gfx950.
# Name: MFMA_{M}x{N}_{Blocks}B_{Groups}[K|N]_{vgprs}V
#
# A/B input layouts (lane groups handle K ranges):
#   bf16 (16x16x32) and fp4 (16x16x128) both use 4 VGPRs.
#   fp8  (16x16x128) uses 8 VGPRs — different packing layout.
MFMA_16x16_1B_4K_4V = MMALayout(instM=16, blocks=1, vgprs=4, waveSize=64)  # bf16 / fp4
MFMA_16x16_1B_4K_8V = MMALayout(instM=16, blocks=1, vgprs=8, waveSize=64)  # fp8
#
# C/D output layout (same lane mapping, groups handle M ranges instead of K):
#   MFMA always accumulates in f32 (or i32) — 4 VGPRs per lane.
#   Conversion to bf16 happens in the store path, not the MFMA output.
MFMA_16x16_1B_4N_4V = MMALayout(instM=16, blocks=1, vgprs=4, waveSize=64)  # f32/i32 C/D


@dataclass(frozen=True)
class MMAScaleLayout:
  """Data-type independent MX scale factor lane layout.

  Parallel to MMALayout but for MX scale factor operands (MXSA/MXSB).
  Scale factors compress the K dimension by mxBlock: one scale element
  covers mxBlock data elements, so the effective K tile is instK // mxBlock.

  Structural parameters from the ISA:
    instM:    Non-K output dimension (e.g. 16) — shared with the MMA instruction.
    blocks:   Independent MxN outputs per instruction (ISA "Blocks").
    vgprs:    VGPRs per lane for the scale operand (instruction-specific).
    mxBlock:  Scaling block size (e.g. 32 for mxfp4).
    waveSize: Wavefront size (default 64).

  Derived:
    contiguousLanes: Lanes on contiguous non-K elements (= instM).
  """
  instM: int
  blocks: int = -1
  vgprs: float = -1
  mxBlock: int = -1
  waveSize: int = -1

  # Derived attributes (computed in __post_init__)
  contiguousLanes: int = field(init=False)

  def __post_init__(self):
    object.__setattr__(self, 'contiguousLanes', self.instM)


# Pre-defined immutable MX scale layouts for gfx950.
# Name: MFMA_SCALE_{M}x{N}_{Blocks}B_MX{mxBlock}_{bits}b
#
# mxfp4 scale (16x16x128, mxBlock=32): 4 scale elements per MMA tile in K (instK//mxBlock=4).
#   Scale tile = instM(16) x 4 x 1B = 64B / 64 lanes = 1B per lane = 0.25 VGPRs.
#   The 2x2 subtile shape covers 4 MMA scale tiles → 4 x 0.25 = 1 full VGPR per subtile.
MFMA_SCALE_16x16_1B_MX32_8V = MMAScaleLayout(instM=16, blocks=1, vgprs=0.25, mxBlock=32, waveSize=64)


################################################################################
# Tile Geometry — subtile partitioning and constraints
################################################################################

class TileGeometry(ABC):
  """Abstract base for tile geometries.

  Defines the interface that all tile geometries must implement.
  Subclasses represent different matrix roles (A/B input, C/D output)
  and can implement their own emit logic for code generation.

  Common properties (must be available on all subclasses):
    mmaLayout:        MMALayout for this operand.
    bpe:              Bytes per element.
    supportedTypes:   Tuple of supported data type names.
    mmaTileShape:     MMA tile dimensions as (dim0, dim1).
    mmaTileSize:      MMA tile size in bytes.
    mmaTileRegCount:  VGPRs per lane for one MMA tile.

  Grid queries are methods that take macro tile dimensions as parameters,
  keeping the geometry independent of any specific macro tile config.
  """


@dataclass(frozen=True)
class ABInputGeometry(TileGeometry):
  """Intermediate base for A/B input tile geometries.

  Holds data-type parameters (MMA instruction shape, element size, supported
  types) that are shared by both the GR and LR access geometries for the same
  dtype. loadShape is declared here but GR and LR instances are initialized
  with different values — subtile shape lives on ABGRGeometry/ABLRGeometry.
  """
  mmaLayout: MMALayout                # MMA instruction layout (instM, instK, vgprs, waveSize)
  instK: int                          # K-dimension of the MMA instruction (elements per inst)
  bpe: float                          # bytes per element (e.g. 2 for bf16, 0.5 for fp4)
  tlu: bool = False                   # True = column-major (contiguous along M); False = row-major (contiguous along K)
  supportedTypes: Tuple[str, ...] = ()                                     # dtype names this geometry supports
  loadShape: LoadShape = field(default_factory=lambda: LoadShape(m=1, k=1)) # elements loaded per lane (m, k)
  loadWidth: int = 16                 # load instruction width in bytes per lane (e.g. 16 = 128-bit, 32 = 256-bit)

  # Derived (computed in __post_init__, independent of macro tile and subtile shape)
  mmaTileShape: Tuple[int, int] = field(init=False)
  mmaTileSize: int = field(init=False)
  mmaTileRegCount: float = field(init=False)

  def __post_init__(self):
    instM = self.mmaLayout.instM
    mmaTileSize = int(instM * self.instK * self.bpe)
    object.__setattr__(self, 'mmaTileShape', (instM, self.instK))
    object.__setattr__(self, 'mmaTileSize', mmaTileSize)
    object.__setattr__(self, 'mmaTileRegCount', float(self.mmaLayout.vgprs))

  @cached_property
  def _cpp(self):
    # Concrete subclasses (ABGRGeometry, ABLRGeometry) build the matching
    # tensile_writer C++ object that services every ported query method. The
    # abstract base is never instantiated directly; surface the contract
    # explicitly so a missing override fails loudly rather than with an opaque
    # AttributeError.
    raise NotImplementedError(
        f"{type(self).__name__} must override the _cpp property")

  # --- MMA tile grid queries (no subtile shape dependency) ---

  def globalMMATileGrid(self, macroTile: int, depthU: int) -> Tuple[int, int]:
    return tuple(self._cpp.globalMMATileGrid(macroTile, depthU))

  def localMMATileGrid(self, macroTile: int, depthU: int, waveGroupSize: int) -> Tuple[int, int]:
    return tuple(self._cpp.localMMATileGrid(macroTile, depthU, waveGroupSize))


@dataclass(frozen=True)
class ABGRGeometry(ABInputGeometry):
  """A/B tile geometry for global reads.

  The GR footprint is described as N discontiguous strips in M, each of shape
  (subtileShape[0], subtileShape[1]) MMA tiles, separated by subtileStride MMA tiles
  in M. This matches the CuTe layout ((subtileShape[0], subtileCount), subtileShape[1])
  with stride ((1, subtileStride), ldA).

  subtileCount/subtileStride can be pinned explicitly (set to a non-None value) or left
  as None to be derived from the kernel config via for_kernel():
    None  -> derived: subtileCount=wg_m, subtileStride=MT0_mma/wg_m
    set   -> pinned:  for_kernel() is a no-op, values are used as-is

  For the contiguous TLU=1 case: subtileCount=1, subtileStride=0 (pinned).
  For TLU=0 with wg_m=4: subtileCount=None -> derived as 4, subtileStride=MT0_mma/4.
  """
  tag:           object              = field(default=None) # emit strategy tag (GRTag_1x2 | GRTag_2x2 | GRTag_TLU1) — dispatches to singledispatch emit impl
  subtileShape:  Tuple[int, int]     = (1, 1)             # MMA tiles per contiguous GR block: (rows_M, cols_K)
  subtileCount:  Optional[int]       = None               # number of blocks per wave group; None = derived from wg_m in for_kernel()
  subtileStride: Optional[int]       = None               # stride between blocks in MMA tiles (M-dim); None = derived from MT0_mma/wg_m in for_kernel()

  @cached_property
  def _cpp(self):
    return _cppgeo.ABGRGeometry(_cpp_mma(self.mmaLayout), self.instK,
                                float(self.bpe), _cpp_loadshape(self.loadShape),
                                tuple(self.subtileShape), self.subtileCount,
                                self.subtileStride, self.tlu, self.loadWidth)

  def localGRGranularity(self, numWaves: int) -> Tuple[int, int]:
    """Number of localSubtile rows covered by one GR load, as (M, K).

    Used as a divisor of localSubtileGrid[0] to obtain the number of distinct
    soffset positions in M (perpDimSize):
        perpDimSize = ceil(localSubtileGrid[0] / localGRGranularity(numWaves)[0])

    For contiguous or strided multi-block shapes (bc > 1): localSubtileGrid[0]
    already folds subtileCount in (it equals localMMATileGrid[0] / subtileShape[0]),
    so each soffset position maps to exactly one localSubtile row — granularity
    is (1, 1) and perpDimSize == localSubtileGrid[0].

    For bc == 1 with wave-cooperative expansion (loadRatioGR > 1): one
    buffer_load covers multiple consecutive localSubtile rows in M.  The
    expansion factor is bytesPerLoad(numWaves) / subtileSizeBytes.

    subtileCount/subtileStride must be materialized via for_kernel() before use.
    """
    return tuple(self._cpp.localGRGranularity(numWaves))

  def globalSubtileGrid(self, macroTile: int, depthU: int) -> Tuple[float, float]:
    return tuple(self._cpp.globalSubtileGrid(macroTile, depthU))

  def subtileSizeBytes(self) -> float:
    """Bytes in one contiguous strip."""
    return self._cpp.subtileSizeBytes()

  def bytesPerLoad(self, numWaves: int) -> int:
    """Total bytes loaded cooperatively per load round (all waves, all lanes)."""
    return self._cpp.bytesPerLoad(numWaves)

  def loadsPerStrip(self, numWaves: int) -> float:
    return self._cpp.loadsPerStrip(numWaves)

  def for_kernel(self, kernel: dict, tc: str) -> 'ABGRGeometry':
    """Return a new frozen instance with subtileCount/subtileStride from kernel config.

    The pre-defined instances (AB_B16.gr etc.) are dtype-only templates; this
    method materializes them for a specific wave-group and macro tile size.
    tc: 'A' or 'B' — selects the correct wave-group axis and macro tile key.

    subtileShape is expanded when cooperating waves (numWaves // wg_m) can load
    more MMA tiles than the base subtileShape covers.  This eliminates loadRatio > 1
    cases: the effective per-load coverage IS the subtileShape.
    """
    cpp_fk = self._cpp.for_kernel(kernel, tc)
    return replace(self, subtileCount=cpp_fk.subtileCount,
                   subtileStride=cpp_fk.subtileStride)

  # --- Subtile query ---

  def subtileForMmaTile(self, r: int, c: int):
    """Return the global subtile containing MMA tile (r, c), the block shape,
    and every MMA tile that belongs to that same subtile.

    This method is geometry-only: it groups MMA tiles into subtiles based on
    subtileShape/subtileCount/subtileStride without regard to TLU or load ordering.
    The returned mma_tiles list is in a fixed geometric order (M-outer, K-inner)
    and is not suitable as a position index for wave/GR assignment — callers
    that need load ordering must apply TLU-aware sorting on top.

    Args:
        r: global MMA tile row index (0-based, M dimension)
        c: global MMA tile column index (0-based, K dimension)

    Returns:
        subtile_id  : (subtile_m, subtile_k) — global subtile coordinate
        block_shape : (bM, bK) — self.subtileShape as a tuple of ints
        mma_tiles   : list[(row, col)] for every MMA tile in the subtile,
                      in geometric order (M-outer, K-inner); not TLU-ordered

    Requires subtileCount/subtileStride to be materialized via for_kernel().
    """
    if self.subtileCount is None or self.subtileStride is None:
      raise RuntimeError("subtileForMmaTile requires for_kernel() to be called first")

    sid, bshape, tiles = self._cpp.subtileForMmaTile(r, c)
    return (tuple(sid), tuple(bshape), [tuple(t) for t in tiles])

  # --- Emit stubs: GR offset, GR instruction, LW to LDS ---

  def emitGlobalReadOffset(self, writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitGlobalReadOffset not implemented")

  def emitGlobalRead(self, writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitGlobalRead not implemented")

  def emitLocalWrite(self, writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitLocalWrite not implemented")


@dataclass(frozen=True)
class ABLRGeometry(ABInputGeometry):
  """A/B tile geometry for local reads (LDS).

  Owns the LR subtile shape, which may differ from the GR subtile shape.
  Concrete subclasses implement LR offset and LR instruction emit.
  """
  tag:          object           = field(default=None) # emit strategy tag (LRTag_1x2 | LRTag_TLU1) — dispatches to singledispatch emit impl
  subtileShape: Tuple[int, int]  = (1, 1)             # MMA tiles per LR subtile: (rows_M, cols_K)

  @cached_property
  def _cpp(self):
    return _cppgeo.ABLRGeometry(_cpp_mma(self.mmaLayout), self.instK,
                                float(self.bpe), _cpp_loadshape(self.loadShape),
                                tuple(self.subtileShape), self.tlu, self.loadWidth)

  def globalSubtileGrid(self, macroTile: int, depthU: int) -> Tuple[float, float]:
    return tuple(self._cpp.globalSubtileGrid(macroTile, depthU))

  def subtileSizeBytes(self) -> float:
    return self._cpp.subtileSizeBytes()

  # --- Emit stubs: LR offset, LR instruction ---

  def emitLocalReadOffset(self, writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitLocalReadOffset not implemented")

  def emitLocalRead(self, writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitLocalRead not implemented")


@dataclass(frozen=True)
class ABTilePair(TileGeometry):
  """Bundles GR and LR geometries for one A/B matrix operand.

  GR and LR are allowed to have different subtile shapes — this is the primary
  motivation for the split. Common dtype properties (mmaLayout, instK, bpe,
  supportedTypes, mmaTileShape, mmaTileSize, mmaTileRegCount) are delegated
  to the GR geometry (both must share the same dtype params).

  This is the object passed to TileInfo; the scheduler never reaches into
  gr or lr directly.
  """
  gr: ABGRGeometry
  lr: ABLRGeometry

  def for_kernel(self, kernel: dict, tc: str) -> 'ABTilePair':
    """Return a new ABTilePair with gr materialized for the given kernel config.
    tc: 'A' or 'B' — passed through to ABGRGeometry.for_kernel.
    """
    return replace(self, gr=self.gr.for_kernel(kernel, tc))

  # Delegate TileGeometry common properties to gr.
  @property
  def mmaLayout(self): return self.gr.mmaLayout
  @property
  def instK(self): return self.gr.instK
  @property
  def bpe(self): return self.gr.bpe
  @property
  def supportedTypes(self): return self.gr.supportedTypes
  @property
  def mmaTileShape(self): return self.gr.mmaTileShape
  @property
  def mmaTileSize(self): return self.gr.mmaTileSize
  @property
  def mmaTileRegCount(self): return self.gr.mmaTileRegCount


@dataclass(frozen=True)
class CDTileGeometry(TileGeometry):
  """Abstract geometry for C/D output tiles.

  Describes how the output macro tile is partitioned into MMA tiles and
  subtiles, with wave-group partitioning in both M and N dimensions.

  This class is abstract — concrete subclasses (e.g. for f32, bf16 store)
  implement the emit methods for their specific store/conversion patterns.
  """
  mmaLayout: MMALayout
  bpe: float
  supportedTypes: Tuple[str, ...] = ()

  storeShape: LoadShape = field(default_factory=lambda: LoadShape(m=1, k=1))

  # Derived (computed in __post_init__, independent of macro tile and subtile shape)
  mmaTileShape: Tuple[int, int] = field(init=False)
  mmaTileSize: int = field(init=False)
  mmaTileRegCount: float = field(init=False)

  def __post_init__(self):
    instM = self.mmaLayout.instM
    mmaTileSize = int(instM * instM * self.bpe)
    object.__setattr__(self, 'mmaTileShape', (instM, instM))
    object.__setattr__(self, 'mmaTileSize', mmaTileSize)
    object.__setattr__(self, 'mmaTileRegCount',
                       mmaTileSize / self.mmaLayout.waveSize / 4)

  # --- Grid queries (depend on macro tile config, computed on demand) ---

  @cached_property
  def _cpp(self):
    return _cppgeo.CDTileGeometry(_cpp_mma(self.mmaLayout), float(self.bpe),
                                  _cpp_loadshape(self.storeShape))

  def globalMMATileGrid(self, macroTile0: int, macroTile1: int) -> Tuple[int, int]:
    return tuple(self._cpp.globalMMATileGrid(macroTile0, macroTile1))

  def localMMATileGrid(self, macroTile0: int, macroTile1: int,
                       waveGroup: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(self._cpp.localMMATileGrid(macroTile0, macroTile1,
                                                 tuple(waveGroup)))

  def globalSubtileGrid(self, macroTile0: int, macroTile1: int,
                        subtileShape: Tuple[float, float]) -> Tuple[float, float]:
    """Subtile grid over the full macro tile."""
    return tuple(self._cpp.globalSubtileGrid(macroTile0, macroTile1,
                                                  tuple(subtileShape)))

  def localSubtileGrid(self, macroTile0: int, macroTile1: int,
                       waveGroup: Tuple[int, int],
                       subtileShape: Tuple[float, float]) -> Tuple[float, float]:
    """Subtile grid per wave (each wave stores its own chunk)."""
    return tuple(self._cpp.localSubtileGrid(macroTile0, macroTile1,
                                                 tuple(waveGroup),
                                                 tuple(subtileShape)))

  # --- Emit stubs (to be implemented by concrete subclasses) ---

  @abstractmethod
  def emitStoreD(self, ti: 'TileInfo', writer, kernel) -> 'Module':
    """Emit store-D instructions for this tile."""
    ...

  @abstractmethod
  def emitLoadC(self, ti: 'TileInfo', writer, kernel) -> 'Module':
    """Emit load-C instructions for this tile."""
    ...


@dataclass(frozen=True)
class MXScaleInputGeometry(TileGeometry):
  """Common base for MX scale GR and LR geometries.

  MX scale factors use a compressed K dimension: one scale element covers
  mxBlock data elements, so the effective K size is instK // mxBlock.

  Structural parameters:
    scaleLayout: MMAScaleLayout (lane/VGPR geometry for the scale operand).
    instK:       Full data K dimension of the MMA instruction (e.g. 128 for mxfp4).
    bpe:         Bytes per scale element (typically 1.0).

  Derived:
    mmaTileShape:    (instM, instK // mxBlock) — scale tile in elements.
    mmaTileSize:     instM * (instK // mxBlock) * bpe bytes.
    mmaTileRegCount: VGPRs per lane for one scale MMA tile.
  """
  scaleLayout: MMAScaleLayout
  instK: int
  bpe: float
  supportedTypes: Tuple[str, ...] = ()
  loadWidth: int = 16  # load instruction width in bytes per lane (e.g. 16 = 128-bit, 32 = 256-bit) (GR=16, LR=4)

  mmaTileShape: Tuple[int, int] = field(init=False)
  mmaTileSize: int = field(init=False)
  mmaTileRegCount: float = field(init=False)

  def __post_init__(self):
    instM      = self.scaleLayout.instM
    instKScale = self.instK // self.scaleLayout.mxBlock
    mmaTileSize = int(instM * instKScale * self.bpe)
    object.__setattr__(self, 'mmaTileShape',    (instM, instKScale))
    object.__setattr__(self, 'mmaTileSize',     mmaTileSize)
    object.__setattr__(self, 'mmaTileRegCount', mmaTileSize / self.scaleLayout.waveSize / 4)

  @cached_property
  def _cpp(self):
    # Concrete subclasses (MXScaleGRGeometry, MXScaleLRGeometry) build the
    # matching tensile_writer C++ object that services every ported query
    # method. The abstract base is never instantiated directly; surface the
    # contract explicitly so a missing override fails loudly rather than with an
    # opaque AttributeError.
    raise NotImplementedError(
        f"{type(self).__name__} must override the _cpp property")

  def globalMMATileGrid(self, macroTile: int, depthU: int) -> Tuple[int, int]:
    # depthU is in data elements; divide by instK (not instKScale) to get scale MMA K tiles.
    return tuple(self._cpp.globalMMATileGrid(macroTile, depthU))


@dataclass(frozen=True)
class MXScaleGRGeometry(MXScaleInputGeometry):
  """GR geometry for MX scale factors.

  subtileShape covers the entire global scale MMA tile grid so that all waves
  together can load all scale factors in a single buffer_load round.
  subtileShape is derived from the kernel (None = not yet materialized).

  for_kernel sets subtileShape = (mt // instM, depthU // (instK // mxBlock)).
  """
  subtileShape: Optional[Tuple[int, int]] = None  # None = derive from kernel; set explicitly to pin

  @cached_property
  def _cpp(self):
    shape = tuple(self.subtileShape) if self.subtileShape is not None else None
    return _cppgeo.MXScaleGRGeometry(_cpp_scale(self.scaleLayout), self.instK,
                                     float(self.bpe), self.loadWidth, shape)

  def for_kernel(self, kernel: dict, tc: str) -> 'MXScaleGRGeometry':
    if self.subtileShape is not None:
      return self
    cpp_fk = self._cpp.for_kernel(kernel, tc)
    return replace(self, subtileShape=tuple(cpp_fk.subtileShape))

  def emitGlobalReadOffset(self, ti: 'TileInfo', writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitGlobalReadOffset not implemented")

  def emitGlobalRead(self, ti: 'TileInfo', writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitGlobalRead not implemented")

  def emitLocalWrite(self, ti: 'TileInfo', writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitLocalWrite not implemented")


@dataclass(frozen=True)
class MXScaleLRGeometry(MXScaleInputGeometry):
  """LR geometry for MX scale factors.

  subtileShape is fixed at (2, 2) — 2 scale MMA tiles in M × 2 in K.
  This matches the 2×2 VGPR packing used by the MX scale LDS layout.
  """
  subtileShape: Tuple[int, int] = (2, 2)

  @cached_property
  def _cpp(self):
    return _cppgeo.MXScaleLRGeometry(_cpp_scale(self.scaleLayout), self.instK,
                                     float(self.bpe), self.loadWidth,
                                     tuple(self.subtileShape))

  def globalSubtileGrid(self, macroTile: int, depthU: int) -> Tuple[float, float]:
    return tuple(self._cpp.globalSubtileGrid(macroTile, depthU))

  def subtileSizeBytes(self) -> float:
    return self._cpp.subtileSizeBytes()

  def emitLocalReadOffset(self, ti: 'TileInfo', writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitLocalReadOffset not implemented")

  def emitLocalRead(self, ti: 'TileInfo', writer, kernel) -> 'Module':
    raise NotImplementedError(f"{type(self).__name__}.emitLocalRead not implemented")


@dataclass(frozen=True)
class MXScaleTilePair(TileGeometry):
  """Bundles GR and LR geometries for one MX scale operand (MXSA or MXSB).

  Mirrors ABTilePair: GR owns the global-read layout (subtileShape derived from
  the kernel macro tile and depthU); LR owns the local-read subtile shape.
  Common properties are delegated to gr.
  """
  gr: MXScaleGRGeometry
  lr: MXScaleLRGeometry

  def for_kernel(self, kernel: dict, tc: str) -> 'MXScaleTilePair':
    return replace(self, gr=self.gr.for_kernel(kernel, tc))

  @property
  def scaleLayout(self):    return self.gr.scaleLayout
  @property
  def instK(self):          return self.gr.instK
  @property
  def bpe(self):            return self.gr.bpe
  @property
  def supportedTypes(self): return self.gr.supportedTypes
  @property
  def mmaTileShape(self):   return self.gr.mmaTileShape
  @property
  def mmaTileSize(self):    return self.gr.mmaTileSize
  @property
  def mmaTileRegCount(self): return self.gr.mmaTileRegCount


################################################################################
# Tag sentinels — pure marker types for singledispatch in Kernel.py
#
# Each tag selects an emit strategy. Tags carry no data — they are analogous
# to C++ tag-dispatch types (e.g. std::random_access_iterator_tag).
# The ABGRGeometry / ABLRGeometry classes store one tag instance as `self.tag`.
################################################################################

@dataclass(frozen=True)
class GRTag_1x1:
  """GR emit strategy: row-major (TLU=0), 1×1 block shape."""

@dataclass(frozen=True)
class GRTag_1x2:
  """GR emit strategy: row-major (TLU=0), 1×2 block shape."""

@dataclass(frozen=True)
class GRTag_2x2:
  """GR emit strategy: row-major (TLU=0), 2×2 block shape."""

@dataclass(frozen=True)
class GRTag_TLU1:
  """GR emit strategy: column-major (TLU=1), 8×1 block shape."""

@dataclass(frozen=True)
class LRTag_1x1:
  """LR emit strategy: row-major (TLU=0), 1×1 subtile shape."""

@dataclass(frozen=True)
class LRTag_1x2:
  """LR emit strategy: row-major (TLU=0), 1×2 subtile shape."""

@dataclass(frozen=True)
class LRTag_TLU1:
  """LR emit strategy: column-major (TLU=1), 8×1 subtile shape."""


################################################################################
# C++-backed TileInfo query layer and emit-leaf decisions.
#
# Like the geometry value/query layer above (already C++-only), the read-only
# TileInfo grid/index query methods and the single buffer-load / ds-read
# emit-leaf plans for the ABTilePair case are serviced unconditionally by the
# compiled ``tensile_writer.subtile`` nanobind extension (the ``tile_info`` and
# ``emit`` submodules). There is no parallel Python formula for these ported AB
# cases. The Python TileInfo remains the canonical object: register allocation,
# rocisa emission, scale/CD paths, and main-loop orchestration never cross into
# C++, and the MX scale / C/D geometries keep their pure-Python out-of-scope
# paths.
#
# The GR/LR offset-assignment plans (graTileAssignment / lraTileAssignment) are
# also C++-only for the ported row-major BF16 (B16/TLU0) AB path: there is no
# env switch and no Python scalar-math twin for that case. FP8 / FP4 / TLU1 and
# the native (non-subtile) paths stay on the Python legacy emit as explicitly
# unported behavior.
################################################################################

# The query + emit-leaf layers are C++-only (no Python fallback): import the
# compiled submodules at module load.
from tensile_writer.subtile import tile_info as _CPP_TI
from tensile_writer.subtile import emit as _CPP_EMIT
from tensile_writer.subtile.module_builder import ModuleBuilder as _ModuleBuilder

# Global singleton — cheap: imports and caches rocisa handles once on first call.
_MFMA_BUILDER = None

def _mfma_builder():
    global _MFMA_BUILDER
    if _MFMA_BUILDER is None:
        _MFMA_BUILDER = _ModuleBuilder()
    return _MFMA_BUILDER

################################################################################
# Concrete tile classes and pre-defined config instances
#
# Config objects (frozen): ABGRGeometry / ABLRGeometry instances, each with a
#   tag field (GRTag_1x2 | GRTag_TLU1 / LRTag_1x2 | LRTag_TLU1) defined above.
#
# Runtime tile classes (mutable): ABGRTile, ABLRTile — hold any frozen
#   ABGRGeometry / ABLRGeometry config + mutable register state + emit logic.
#   Created by TileInfo.__init__; emit dispatch is via self.config.tag.
#
# Pre-defined pair instances (e.g. AB_B16): ABTilePair of frozen configs.
#   The code generator passes these to TileInfo, which creates the mutable
#   tile instances automatically.
################################################################################

################################################################################
# GR (global read) emit and alloc dispatch.
# (Consolidated from SubtileGREmit.py - hipblaslt_incremental_refactor-grc.138)
################################################################################
################################################################################
# 1. Dispatch bases
################################################################################

@singledispatch
def _emitGlobalReadOffset(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGlobalReadOffset not implemented for {type(tag).__name__}")

@singledispatch
def _emitGlobalRead(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGlobalRead not implemented for {type(tag).__name__}")

@singledispatch
def _emitLocalWrite(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLocalWrite not implemented for {type(tag).__name__}")

@singledispatch
def _allocGROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"allocGROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _deallocGROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"deallocGROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _emitDTLInit(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitDTLInit not implemented for {type(tag).__name__}")

@singledispatch
def _emitGRLDSBufferSwap(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGRLDSBufferSwap not implemented for {type(tag).__name__}")

@singledispatch
def _emitGRPtrUpdate(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGRPtrUpdate not implemented for {type(tag).__name__}")

# Stubs for tags not yet implemented.
_stub = lambda tag, tile, ti, writer, kernel: None
_emitGlobalReadOffset.register(GRTag_TLU1)(_stub)
_allocGROffsetRegisters.register(GRTag_TLU1)(_stub)
_deallocGROffsetRegisters.register(GRTag_TLU1)(_stub)
_emitGlobalRead.register(GRTag_TLU1)(_stub)
_emitDTLInit.register(GRTag_TLU1)(_stub)
_emitGRLDSBufferSwap.register(GRTag_TLU1)(_stub)
_emitGRPtrUpdate.register(GRTag_TLU1)(_stub)
for _tag in (GRTag_1x1, GRTag_1x2, GRTag_2x2, GRTag_TLU1):
  _emitLocalWrite.register(_tag)(_stub)


################################################################################
# 2. Implementations — TLU=0 (shared by GRTag_1x2 and GRTag_2x2)
################################################################################

@_emitGlobalReadOffset.register(GRTag_1x1)
@_emitGlobalReadOffset.register(GRTag_1x2)
@_emitGlobalReadOffset.register(GRTag_2x2)
def _emitGROffset_TLU0(tag, tile, ti, writer, kernel):
  return Module(f"GR Offset TLU0 ({ti.tc})")  # STUB — legacy path in graTileAssignment
  """GR offset for row-major (TLU=0) geometry with swizzling and rotation.

  Ported from legacy graTileAssignment. Operates on a single tensor component.

  1. Compute waveId, laneId, colId, rowId from Serial (v0).
  2. Swizzle colId via DPP quad_perm to avoid LDS bank conflicts.
  3. Intra-wave rotation: shift colId based on LDS row parity.
  4. Inter-wave rotation: additional shift from waveId (when waves_coop > 1).
  5. Unified wave partition: localRow + partitionRow from waveId.
  6. Compute byte offsets for each GR load into sharedVgprGROffset[].
  7. Compute subtile perpendicular soffsets.
  """
  module = Module(f"GR Offset TLU0 ({ti.tc})")
  tc = ti.tc
  loadWidth = ti.loadWidthGR
  subIterKBytes = ti.subIterKBytes
  blockSize = subIterKBytes // loadWidth
  wavesize = kernel["WavefrontSize"]
  bpe = ti.bpe
  bpeBits = int(8 * bpe)
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]

  wg_m       = ti.waveGroupSize
  numWaves   = ti.numWaves
  waves_coop = numWaves // wg_m
  numRowsPerWave    = wavesize // blockSize
  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes

  tmpVgpr = writer.vgprPool.checkOut(4)
  colId     = tmpVgpr
  rowId     = tmpVgpr + 1
  waveId    = tmpVgpr + 2
  localRow  = tmpVgpr + 3
  tmpSgpr   = writer.sgprPool.checkOut(1, preventOverflow=False)

  # --- 1. waveId, laneId, colId, rowId ---
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment=f"{tc}: waveId"))
  module.add(VAndB32(dst=vgpr(localRow), src0=vgpr("Serial"), src1=wavesize-1,
             comment=f"{tc}: laneId"))
  module.add(VAndB32(dst=vgpr(colId), src0=vgpr("Serial"), src1=blockSize-1,
             comment=f"{tc}: colId for {loadWidth}B load"))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1),
             src=vgpr(localRow), comment=f"{tc}: rowId within wave"))

  # --- 2. Swizzle: DPP quad_perm swap colId pairs on even LDS rows ---
  tmpSwz = writer.vgprPool.checkOut(2)
  ldsRowId     = tmpSwz
  swzTmp       = tmpSwz + 1

  module.addComment0(f"{tc}: Swizzling")
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(blockSize.bit_length()-1),
             src=vgpr(localRow), comment=f"{tc}: row id within wave"))
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1),
             src=vgpr(ldsRowId), comment=f"{tc}: lds row id"))
  module.add(VAndB32(dst=vgpr(swzTmp), src0=vgpr(ldsRowId), src1=hex(1),
             comment=f"{tc}: lds row id %% 2"))
  module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(swzTmp),
             comment=f"{tc}: lds row id %% 2 == 0?"))
  module.add(VMovB32(dst=vgpr(colId), src=vgpr(colId), dpp=DPPModifiers(quad_perm=[1,0,3,2]),
             comment=f"{tc}: swap colId pairs"))
  module.add(SMovB64(dst=EXEC(), src=-1))

  # --- 3. Intra-wave rotation: blockSize - (ldsRowId // 2) * 2 ---
  module.addComment0(f"{tc}: Intra-wave rotation")
  module.add(VLShiftRightB32(dst=vgpr(swzTmp), shiftHex=hex(1), src=vgpr(ldsRowId)))
  module.add(VLShiftLeftB32(dst=vgpr(swzTmp), shiftHex=hex(1), src=vgpr(swzTmp),
             comment=f"{tc}: (ldsRowId // 2) * 2"))
  module.add(VSubU32(dst=vgpr(swzTmp), src0=hex(blockSize), src1=vgpr(swzTmp),
             comment=f"{tc}: rotation = blockSize - (ldsRowId//2)*2"))

  # --- 4. Inter-wave rotation (when waves cooperate on a subtile) ---
  if waves_coop > 1:
    waveRotation = writer.vgprPool.checkOut(1)
    module.addComment0(f"{tc}: Inter-wave rotation")
    module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(waveId), src1=hex(1)))
    module.add(VLShiftLeftB32(dst=vgpr(waveRotation),
               shiftHex=hex((2*numRowsPerLDSBanks).bit_length() - 1), src=vgpr(waveRotation)))
    module.add(VSubU32(dst=vgpr(waveRotation), src0=vgpr(swzTmp), src1=vgpr(waveRotation)))
    module.add(VAddU32(dst=vgpr(colId), src0=vgpr(waveRotation), src1=vgpr(colId)))
    writer.vgprPool.checkIn(waveRotation)
  else:
    module.add(VAddU32(dst=vgpr(colId), src0=vgpr(swzTmp), src1=vgpr(colId)))

  module.add(VAndB32(dst=vgpr(colId), src0=vgpr(colId), src1=hex(blockSize-1),
             comment=f"{tc}: (col + rotation) %% blockSize"))
  writer.vgprPool.checkIn(tmpSwz)

  # --- 5. Unified wave partition ---
  rowOffset = writer.vgprPool.checkOut(1)
  partitionStride = ti.mmaTileShape[0] * int(ti.localSubtileGrid[0])
  waves_coop_shift = max(0, waves_coop.bit_length() - 1) if waves_coop > 0 else 0
  module.add(VAndB32(dst=vgpr(localRow), src0=hex(waves_coop - 1), src1=vgpr(waveId),
             comment=f"{tc}: waveId %% {waves_coop}"))
  module.add(VLShiftRightB32(dst=vgpr(rowOffset), shiftHex=hex(waves_coop_shift),
             src=vgpr(waveId), comment=f"{tc}: waveId // {waves_coop}"))
  module.add(VLShiftLeftB32(dst=vgpr(localRow), shiftHex=hex(numRowsPerWave.bit_length()-1),
             src=vgpr(localRow), comment=f"{tc}: local row * {numRowsPerWave}"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionStride,
             comment=f"{tc}: partition stride"))
  module.add(VMulLOU32(dst=vgpr(rowOffset), src0=sgpr(tmpSgpr), src1=vgpr(rowOffset),
             comment=f"{tc}: partition row offset"))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(localRow), src1=vgpr(rowOffset),
             comment=f"{tc}: + local row"))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(rowId), src1=vgpr(rowOffset),
             comment=f"{tc}: + lane rowId"))

  # --- 6. Compute byte offsets for each GR load ---
  tmpVgpr2 = writer.vgprPool.checkOut(2)
  colBytes = tmpVgpr2 + 1
  for i in range(ti.numGRPerSubtile):
    useColId = colId
    # For numGRPerSubtile > 1 with single-wave subtiles: rotate colId between loads
    if i > 0 and waves_coop == 1 and ti.numGRPerSubtile > 1:
      rotatedCol = writer.vgprPool.checkOut(1)
      colRotation = blockSize // 2
      module.add(VAddU32(dst=vgpr(rotatedCol), src0=colRotation, src1=vgpr(colId),
                 comment=f"{tc}: rotate col for GR {i}"))
      module.add(VAndB32(dst=vgpr(rotatedCol), src0=vgpr(rotatedCol), src1=hex(blockSize-1),
                 comment=f"{tc}: (col + {colRotation}) %% blockSize"))
      useColId = rotatedCol

    module.add(VLShiftLeftB32(dst=vgpr(colBytes), shiftHex=hex(loadWidth.bit_length()-1),
               src=vgpr(useColId), comment=f"{tc}: colId * {loadWidth}"))
    module.add(VMulLOU32(dst=vgpr(tmpVgpr2), src0=sgpr(strideRef), src1=vgpr(rowOffset),
               comment=f"{tc}: rowOffset * stride"))
    module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr2), shiftHex=hex(bpeBits.bit_length()-1),
               src=vgpr(tmpVgpr2), comment=f"{tc}: * bpe"))
    module.add(VLShiftRightB32(dst=vgpr(tmpVgpr2), shiftHex=hex(3), src=vgpr(tmpVgpr2),
               comment=f"{tc}: bits to bytes"))
    module.add(VAddU32(dst=vgpr(tile.sharedVgprGROffset[i]), src0=vgpr(colBytes), src1=vgpr(tmpVgpr2),
               comment=f"{tc}: GR offset {i}"))

    if i > 0 and waves_coop == 1 and ti.numGRPerSubtile > 1:
      writer.vgprPool.checkIn(rotatedCol)

    if i + 1 < ti.numGRPerSubtile:
      advance = ti.subtileShape[0] * ti.mmaTileShape[0] // ti.numGRPerSubtile
      module.add(VAddU32(dst=vgpr(rowOffset), src0=advance, src1=vgpr(rowOffset),
                 comment=f"{tc}: advance row for GR {i+1}"))
  writer.vgprPool.checkIn(tmpVgpr2)

  # --- 7. Subtile perpendicular soffsets ---
  subtileRowElements = ti.subtileShape[0] * ti.mmaTileShape[0]
  s_stride_bpe = int(subtileRowElements * bpe)
  for reg_idx in range(len(ti.localSubtilesRegister)):
    rl = ti.localSubtilesRegister[reg_idx]
    if len(rl) == 0:
      continue
    if rl.is_sgpr:
      module.add(SMulI32(dst=rl.ref(0), src0=hex(s_stride_bpe * reg_idx),
                 src1=sgpr(strideRef), comment=f"{tc}: subtile row {reg_idx} soffset"))
    else:
      stmp = writer.sgprPool.checkOut(1)
      for i, reg in enumerate(rl):
        module.add(SMulI32(dst=sgpr(stmp), src0=hex(s_stride_bpe * reg_idx),
                   src1=sgpr(strideRef), comment=f"{tc}: subtile row {reg_idx} soffset"))
        module.add(VAddU32(dst=vgpr(reg), src0=vgpr(tile.sharedVgprGROffset[i]), src1=sgpr(stmp),
                   comment=f"{tc}: bake soffset into vgpr"))
      writer.sgprPool.checkIn(stmp)

  writer.vgprPool.checkIn(rowOffset)
  writer.vgprPool.checkIn(tmpVgpr)
  writer.sgprPool.checkIn(tmpSgpr)
  return module


@_allocGROffsetRegisters.register(GRTag_1x1)
@_allocGROffsetRegisters.register(GRTag_1x2)
@_allocGROffsetRegisters.register(GRTag_2x2)
def _allocGROffsetRegs_TLU0(tag, tile, ti, writer, kernel):
  """Allocate GR offset registers for TLU=0 shapes.

  Two register groups are allocated:

  1. sharedVgprGROffset[]: one VGPR per GR load within a subtile.
     These hold per-lane byte offsets for buffer_load (colId * loadWidth +
     rowOffset * stride * bpe).  Shared across all subtile rows — only the
     soffset changes between rows.

  2. localSubtilesRegister[]: one RegList per perpendicular subtile row.
     Each entry holds the constant M-direction offset (soffset) that shifts
     the shared VGPR offset to the correct subtile row.

     Row 0 needs no offset (soffset=0), so its RegList is left empty.
     Row 1+ gets either:
       - 1 SGPR (preferred): used as the soffset field in buffer_load.
         The shared VGPR offset is reused as-is across rows.
       - numGRPerSubtile VGPRs (fallback when SGPRs exhausted): each VGPR
         has the shared offset + row offset baked in, replacing soffset.
  """
  # Per-lane byte offsets: one VGPR per GR load within a subtile
  tile.sharedVgprGROffset = []
  for i in range(ti.numGRPerSubtile):
    tile.sharedVgprGROffset.append(writer.vgprPool.checkOut(1))

  # Per-subtile-row soffset registers.
  # perpDimSize = how many GR subtile shapes tile the perpendicular (M) dimension
  # per wave. Each position needs its own soffset register.
  ti.localSubtilesRegister = []
  # perpDimSize: distinct soffset positions in M = how many localSubtile rows
  # need their own soffset register.  localGRGranularity[0] tells how many
  # consecutive localSubtile rows one GR load covers (>1 only for bc==1 with
  # wave-cooperative expansion, i.e. loadRatioGR > 1).
  localSubtileRowCount = int(ti.localSubtileGrid[0])
  gran = tile.localGRGranularity(ti.numWaves)
  perpDimSize = math.ceil(localSubtileRowCount / gran[0])
  tmpSgprBuffer = 3
  sgprLimit = writer.states.regCaps["MaxSgpr"] - tmpSgprBuffer

  for reg_idx in range(perpDimSize):
    useSgpr = writer.sgprPool.size() < sgprLimit
    if useSgpr:
      rl = RegList(writer.sgprPool, RegisterType.Sgpr)
    else:
      rl = RegList(writer.vgprPool, RegisterType.Vgpr)
    ti.localSubtilesRegister.append(rl)
    # Row 0 is the base position — no soffset needed, RegList stays empty.
    if reg_idx == 0:
      continue
    if useSgpr:
      # SGPR path: 1 register for soffset, shared VGPR offset reused.
      rl.alloc(preventOverflow=False)
    else:
      # VGPR fallback: one VGPR per GR load, each with soffset baked in.
      for i in range(ti.numGRPerSubtile):
        rl.alloc(preventOverflow=False)


@_deallocGROffsetRegisters.register(GRTag_1x1)
@_deallocGROffsetRegisters.register(GRTag_1x2)
@_deallocGROffsetRegisters.register(GRTag_2x2)
def _deallocGROffsetRegs_TLU0(tag, tile, ti, writer, kernel):
  """Deallocate GR offset registers for TLU=0 shapes."""
  if isinstance(tile.sharedVgprGROffset, list):
    for voff in tile.sharedVgprGROffset:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprGROffset = []
  if isinstance(ti.localSubtilesRegister, list):
    for rl in ti.localSubtilesRegister:
      rl.dealloc()
    ti.localSubtilesRegister = []


# --- GR load emit (TLU=0) ---------------------------------------------------

@_emitGlobalRead.register(GRTag_1x1)
@_emitGlobalRead.register(GRTag_1x2)
@_emitGlobalRead.register(GRTag_2x2)
def _emitGR_TLU0(tag, tile, ti, writer, kernel):
  """Emit buffer_load_dwordx4 (DTL) for all subtiles in the local grid.

  For each subtile (sId0, sId1):
    - Computes LDS write address (m0) from LocalWriteBaseAddr + subtile offset.
    - Emits buffer_load_b128 with lds=True (direct-to-LDS).
    - Uses soffset (SGPR path) or baked VGPR offset for the subtile row.

  When loadRatioGR > 1, multiple subtiles share one GR load; only the first
  subtile in each group emits the load.
  """
  module = Module(f"GR Load TLU0 ({ti.tc})")
  tc = ti.tc
  isGlc = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x1)
  isSlc = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x2)
  isNT  = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x4)

  perpDimSize = len(ti.localSubtilesRegister)

  # TODO: Remove legacy TileInfo dependency after full migration.
  # Currently uses legacy's grid/sizes because subtileShape expansion in for_kernel
  # changes subtileSize/localSubtileGrid/loadRatioGR, which must match the LDS
  # layout computed from legacy values.
  legacyTi = getattr(writer.states, tc.lower()).tileInfo
  localGrid0 = int(legacyTi.localSubtileGrid[0])
  localGrid1 = int(legacyTi.localSubtileGrid[1])
  legacyLoadRatio = legacyTi.loadRatioGR
  legacySubtileSize = int(legacyTi.subtileSize)

  for j in range(localGrid1):
    for i in range(localGrid0):
      slowId = i
      if legacyLoadRatio == 2.0:
        slowId = int(i // legacyLoadRatio)
      reg_idx = slowId

      # Skip duplicate loads when loadRatio > 1
      if legacyLoadRatio > 1:
        linearId = j * localGrid0 + i
        grBaseId = int(linearId // legacyLoadRatio)
        firstInGroup = int(grBaseId * legacyLoadRatio)
        if linearId != firstInGroup:
          continue

      rl = ti.localSubtilesRegister[min(reg_idx, perpDimSize - 1)]
      offsetK = j * int(ti.mmaTileShape[1] * ti.subtileShape[1] * ti.bpe)

      module.addComment0(f"GR load {tc} subtile [{i},{j}]")

      subtileOffset = int(math.ceil(legacyLoadRatio * legacySubtileSize)) if legacyLoadRatio else legacySubtileSize
      WriteBaseAddr = f"LocalWriteBaseAddr{tc}"

      for gr_idx in range(legacyTi.numGRPerSubtile):
        m0Offset = gr_idx * subtileOffset + (i + j * int(legacyTi.globalSubtileGrid[0])) * legacySubtileSize
        module.add(SAddU32(dst=mgpr(0), src0=sgpr(WriteBaseAddr), src1=(m0Offset - offsetK)))
        mubuf = MUBUFModifiers(offen=True, offset12=offsetK, glc=isGlc, slc=isSlc, nt=isNT, lds=True)

        use_sgpr = rl.is_sgpr if len(rl) > 0 else True
        soffset = rl.ref(0) if len(rl) > 0 and use_sgpr else 0
        voff = tile.sharedVgprGROffset[gr_idx] if use_sgpr or len(rl) == 0 else rl.indices[gr_idx]
        module.add(BufferLoadB128(dst=None, vaddr=vgpr(voff), saddr=sgpr(f"Srd{tc}", 4),
                   soffset=soffset, mubuf=mubuf, comment=f"GR{gr_idx} [{i},{j}]"))

  return module


# --- DTL init (TLU=0) -------------------------------------------------------

@_emitDTLInit.register(GRTag_1x1)
@_emitDTLInit.register(GRTag_1x2)
@_emitDTLInit.register(GRTag_2x2)
def _emitDTLInit_TLU0(tag, tile, ti, writer, kernel):
  return Module(f"DTL Init ({ti.tc})")  # STUB — legacy path in globalReadDTLInitCommonSgpr
  """Compute LocalWriteBaseAddr and Swap SGPR for one tensor component.

  The DTL (direct-to-LDS) buffer_load writes data at m0 = LocalWriteBaseAddr + subtile offset.
  LocalWriteBaseAddr is the wave's base LDS position, derived from the wave partition.
  Swap holds the XOR mask to toggle between double-buffer halves.

  For double-buffering: LocalWriteBaseAddr XOR Swap flips to the other buffer.

  Requires sgprs: LocalWriteBaseAddr{tc}, Swap{tc} (must be pre-allocated by caller).
  """
  module = Module(f"DTL Init ({ti.tc})")
  tc = ti.tc
  wavesize = kernel["WavefrontSize"]
  wg_m     = ti.waveGroupSize
  numWaves = ti.numWaves
  waves_coop = numWaves // wg_m

  vgprWaveId = writer.vgprPool.checkOut(1)
  rowOffset  = writer.vgprPool.checkOut(1)

  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment=f"{tc}: waveId"))

  # Wave partition: same unified formula as GR offset step 5
  numRowsPerWave  = wavesize // (ti.subIterKBytes // ti.loadWidthGR)
  partitionStride = ti.mmaTileShape[0] * int(ti.localSubtileGrid[0])
  waves_coop_shift = max(0, waves_coop.bit_length() - 1) if waves_coop > 0 else 0

  module.add(VLShiftRightB32(dst=vgpr(rowOffset), shiftHex=hex(waves_coop_shift),
             src=vgpr(vgprWaveId), comment=f"{tc}: partitionRow = waveId // {waves_coop}"))
  tmpSgpr = writer.sgprPool.checkOut(1, preventOverflow=False)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionStride))
  module.add(VMulLOU32(dst=vgpr(rowOffset), src0=sgpr(tmpSgpr), src1=vgpr(rowOffset),
             comment=f"{tc}: partition row offset"))
  writer.sgprPool.checkIn(tmpSgpr)

  # Scale by subIterKBytes to get LDS byte offset
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset),
             shiftHex=hex(ti.subIterKBytes.bit_length()-1), src=vgpr(rowOffset),
             comment=f"{tc}: * subIterKBytes"))

  # Move to SGPR via readfirstlane (uniform across wave)
  module.add(SNop(waitState=0, comment="wait for VGPR"))
  WriteBaseAddr = f"LocalWriteBaseAddr{tc}"
  Swap = f"Swap{tc}"
  module.add(VReadfirstlaneB32(dst=sgpr(WriteBaseAddr), src=vgpr(rowOffset),
             comment=f"{tc}: base LDS offset"))

  # Add global LDS start offset for B (B data follows A in LDS)
  ldsStartOffset = getattr(writer, f'ldsStartOffset{tc}', 0)
  if ldsStartOffset:
    module.add(SAddU32(dst=sgpr(WriteBaseAddr), src0=sgpr(WriteBaseAddr),
               src1=hex(ldsStartOffset), comment=f"{tc}: + ldsStartOffset"))

  # Swap mask: XOR(base, base + ldsTotalSize) toggles between buffer halves
  module.add(SAddU32(dst=sgpr(Swap), src0=sgpr(WriteBaseAddr), src1=writer.ldsTotalSize))
  module.add(SXorB32(dst=sgpr(Swap), src0=sgpr(WriteBaseAddr), src1=sgpr(Swap)))

  writer.vgprPool.checkIn(vgprWaveId)
  writer.vgprPool.checkIn(rowOffset)
  return module


# --- GR LDS buffer swap (TLU=0) ---------------------------------------------

@_emitGRLDSBufferSwap.register(GRTag_1x1)
@_emitGRLDSBufferSwap.register(GRTag_1x2)
@_emitGRLDSBufferSwap.register(GRTag_2x2)
def _emitGRLDSSwap_TLU0(tag, tile, ti, writer, kernel):
  """Toggle GR DTL write target between double-buffer halves.

  XOR LocalWriteBaseAddr with Swap to flip to the other LDS buffer. Boundary
  call: the rocisa construction lives in the C++ ModuleBuilder.
  """
  return _mfma_builder().gr_lds_buffer_swap(ti.tc)


# --- GR pointer update (TLU=0) ----------------------------------------------

@_emitGRPtrUpdate.register(GRTag_1x1)
@_emitGRPtrUpdate.register(GRTag_1x2)
@_emitGRPtrUpdate.register(GRTag_2x2)
def _emitGRPtrUpdate_TLU0(tag, tile, ti, writer, kernel):
  """Advance SRD base pointer by one depthU iteration (depthU * bpe bytes).

  Boundary call: the depthU byte increment is a writer-resolved scalar; the
  rocisa SAddU32/SAddCU32 construction lives in the C++ ModuleBuilder.
  """
  return _mfma_builder().gr_ptr_update(ti.tc, int(ti.depthUBytes))


################################################################################
# Legacy GR emit functions (moved from SubtileBasedKernel.py)
################################################################################

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
##################################################
# Subroutine to generate GR offset calculation code
#
def graTileAssignment(writer, kernel, useSwizzling=True):
  # The AB GR offset-assignment scalar math (block/row/partition sizes,
  # advance/rotation strides, subtile soffset stride, and the FP8 swizzle
  # selector) is computed by the C++ ABTileInfoQuery.grOffsetAssignPlan for
  # every AB geometry — BF16/B16, FP4/B4, FP8/B8, and the TLU1 BF16 variants.
  # The rocisa emission stays here.
  return _graTileAssignment_cpp(writer, kernel, useSwizzling)

# --- C++-plan-driven GR offset assignment (all AB geometries) ---------------
#
# These source every derived scalar (blockSize, numRowsPerLDSBanks,
# numRowsPerWave, partition stride/mode, advance/rotation offsets, subtile
# soffset stride, FP8 swizzle selector) from the C++
# ABTileInfoQuery.grOffsetAssignPlan. The register state (sharedVgprGROffset,
# localSubtilesRegister) and all rocisa construction stay in Python.

def _grComputeOffset_cpp(module, writer, tileInfo, plan, colId, rowId, output):
  tc = tileInfo.tc
  bpeBits = plan.bpeBits
  tmpVgpr = writer.vgprPool.checkOut(2)
  colBytes = tmpVgpr + 1
  loadWidth = plan.loadWidth
  module.add(VLShiftLeftB32(dst=vgpr(colBytes), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colId), comment="scale col_id by load_width"))
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(strideRef), src1=vgpr(rowId), comment="%s: rowId * stride"%tc))
  module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(bpeBits.bit_length()-1), src=vgpr(tmpVgpr), comment="%s: rowId*stride*bpe"%tc))
  module.add(VLShiftRightB32(dst=vgpr(tmpVgpr), shiftHex=hex(3), src=vgpr(tmpVgpr), comment="to bytes"))
  module.add(VAddU32(dst=vgpr(output), src0=vgpr(colBytes), src1=vgpr(tmpVgpr), comment="%s: GR row_offset"%tc))
  writer.vgprPool.checkIn(tmpVgpr)

def _grComputeSubtileOffsets_cpp(writer, module, tileInfo, plan):
  tc = tileInfo.tc
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  rowOffset = plan.grSubtileRowOffset
  s_stride = plan.sStride
  for regId in range(len(tileInfo.localSubtilesRegister)):
    rl = tileInfo.localSubtilesRegister[regId]
    for i, reg in enumerate(rl):
      if rl.is_sgpr:
        module.add(SMulI32(dst=sgpr(reg), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
      else:
        stmp = writer.sgprPool.checkOut(1)
        module.add(SMulI32(dst=sgpr(stmp), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
        module.add(VAddU32(dst=vgpr(reg), src0=vgpr(tileInfo.sharedVgprGROffset[i]), src1=sgpr(stmp)))
        writer.sgprPool.checkIn(stmp)

def _grComputeRowPartition_cpp(module, writer, tileInfo, plan, waveId, rowOffset):
  numRowsPerWave = plan.numRowsPerWave
  tc = tileInfo.tc
  tmpVgpr = writer.vgprPool.checkOut(2)
  tmpSgpr = writer.sgprPool.checkOut(1, preventOverflow=False)
  localRow = tmpVgpr
  partitionRow = tmpVgpr+1
  partitionOffset = plan.partitionOffset
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionOffset, comment="%s: row offset"%tc))
  if plan.partitionMode == 1:
    module.add(VAndB32(dst=vgpr(localRow), src0=hex(1), src1=vgpr(waveId), comment="%s: waveId %% 2"%tc))
    module.add(VLShiftRightB32(dst=vgpr(partitionRow), shiftHex=hex(1), src=vgpr(waveId), comment="%s: waveId / 2"%tc))
  elif plan.partitionMode == 0:
    module.add(VMovB32(dst=vgpr(localRow), src=0, comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=vgpr(waveId), comment="%s"%tc))
  elif plan.partitionMode == 2:
    module.add(VMovB32(dst=vgpr(localRow), src=vgpr(waveId), comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=0, comment="%s"%tc))
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(plan.loadRatioGR))
  module.add(VLShiftLeftB32(dst=vgpr(localRow), shiftHex=hex(numRowsPerWave.bit_length()-1), src=vgpr(localRow), comment="%s: local row offset"%tc))
  module.add(VMulLOU32(dst=vgpr(partitionRow), src0=sgpr(tmpSgpr), src1=vgpr(partitionRow), comment="%s: wave row offset"%tc))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(localRow), src1=vgpr(partitionRow), comment="%s: row offset"%tc))
  writer.vgprPool.checkIn(tmpVgpr)
  writer.sgprPool.checkIn(tmpSgpr)

def _grComputeAllOffsets_cpp(module, writer, tileInfo, plan, colId, rowId, rowOffset):
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(rowId), src1=vgpr(rowOffset), comment="%s: row offset"%tileInfo.tc))
  _grComputeOffset_cpp(module, writer, tileInfo, plan, colId, rowOffset, tileInfo.sharedVgprGROffset[0])
  for i in range(1, len(tileInfo.sharedVgprGROffset)):
    offset = plan.grAdvanceOffset
    module.add(VAddU32(dst=vgpr(rowOffset), src0=offset, src1=vgpr(rowOffset), comment="%s: advance row for GR offset %u"%(tileInfo.tc, i)))
    rotatedcolId = writer.vgprPool.checkOut(1)
    if plan.loadRatioGR == 0.5:
      if plan.isFp8:  # FP8: intra-block K_group +2 rotation, preserving block bit
        tmpBlock = writer.vgprPool.checkOut(1)
        module.add(VAndB32(dst=vgpr(tmpBlock), src0=vgpr(colId), src1=hex(4), comment="%s: block_bit = colId & 4"%tileInfo.tc))
        module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(colId), src1=hex(3), comment="%s: K_group = colId & 3"%tileInfo.tc))
        module.add(VAddU32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(2), comment="%s: K_group + 2"%tileInfo.tc))
        module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(3), comment="%s: (K_group+2) %% 4"%tileInfo.tc))
        module.add(VAddU32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=vgpr(tmpBlock), comment="%s: K_group_rot + block_bit"%tileInfo.tc))
        writer.vgprPool.checkIn(tmpBlock)
      else:  # FP4/FP16: half-block rotation
        blockSize = plan.blockSize
        colRotation = blockSize // 2
        module.add(VAddU32(dst=vgpr(rotatedcolId), src0=colRotation, src1=vgpr(colId), comment="%s: rotate col for GR offset %u"%(tileInfo.tc, i)))
        module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(blockSize-1), comment="(col + %d) %% block_size"%colRotation))
    else:
      module.add(VMovB32(dst=vgpr(rotatedcolId), src=vgpr(colId), comment=""))
    _grComputeOffset_cpp(module, writer, tileInfo, plan, rotatedcolId, rowOffset, tileInfo.sharedVgprGROffset[i])
    writer.vgprPool.checkIn(rotatedcolId)

def _grSwizzleColIds_cpp(module, writer, planA, planB, blockSize, numRowsPerLDSBanks,
                         laneId, colIdA, colIdB, waveId):
  tmpVgpr = writer.vgprPool.checkOut(3)
  ldsRowId = tmpVgpr
  tmp = tmpVgpr + 1
  waveRotation = tmpVgpr + 2
  half = blockSize // 2
  module.addComment0("Swizzling")
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(ldsRowId), comment="lds row id"))
  module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(ldsRowId), src1=hex(1), comment="swap_bit = ldsRowId & 1"))
  if planA.isFp8:  # FP8: step1=block-swap, step2=wave K_group rotation
    # Step 1: block-swap (XOR blockSize//2 for odd ldsRowId)
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(int(math.log2(half))), src=vgpr(tmp),
               comment=f"swap_bit * {half}"))
    module.add(VXorB32(dst=vgpr(colIdA), src0=vgpr(colIdA), src1=vgpr(tmp),
               comment="FP8 step1: block-swap colIdA"))
    module.add(VMovB32(dst=vgpr(colIdB), src=vgpr(colIdA), comment="colIdB = colIdA"))
    # Step 2: K_group rotation = (waveId & 1) * 2 (only for loadRatioGR != 0.5)
    module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(waveId), src1=hex(1), comment="wave_half = waveId & 1"))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="rotation = wave_half * 2"))
    for plan, cId in [(planA, colIdA), (planB, colIdB)]:
      if plan.loadRatioGR != 0.5:
        module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(cId), src1=hex(4), comment="FP8 step2: block_bit = colId & 4"))
        module.add(VAndB32(dst=vgpr(cId), src0=vgpr(cId), src1=hex(3), comment="K_group = colId & 3"))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(cId), src1=vgpr(tmp), comment="K_group + rotation"))
        module.add(VAndB32(dst=vgpr(cId), src0=vgpr(cId), src1=hex(3), comment="(K_group+rotation) % 4"))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(cId), src1=vgpr(waveRotation), comment="K_group_rot + block_bit"))
  else:  # FP4/FP16/BF16: pair-swap (even ldsRowId) + intra/inter-wave rotation
    module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(tmp), comment="lds row id % 2 == 0 ?"))
    module.add(VMovB32(dst=vgpr(colIdA), src=vgpr(colIdA), dpp=DPPModifiers(quad_perm=[1,0,3,2]), comment="swap colId pairs for swizzling"))
    module.add(SMovB64(dst=EXEC(), src=-1))
    module.add(VMovB32(dst=vgpr(colIdB), src=vgpr(colIdA), comment=""))
    module.addComment0("Rotation within a single wave")
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(ldsRowId), comment=""))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="(ldsRowId //2) * 2"))
    module.add(VSubU32(dst=vgpr(tmp), src0=hex(blockSize), src1=vgpr(tmp), comment="rotation offset : blockSize - (ldsRowId//2)*2"))
    for plan, cId in [(planA, colIdA), (planB, colIdB)]:
      if plan.loadRatioGR != 0.5:
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

def _graTileAssignment_cpp(writer, kernel, useSwizzling=True):
  module = Module()
  module.addComment0("GR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  planA = tileInfoA.grOffsetAssignPlan(writer)
  planB = tileInfoB.grOffsetAssignPlan(writer)
  wavesize = kernel["WavefrontSize"]
  loadWidth = planA.loadWidth
  blockSize = planA.blockSize
  numRowsPerLDSBanks = planA.numRowsPerLDSBanks
  tmpVgpr = writer.vgprPool.checkOut(7)
  colIdA = tmpVgpr
  colIdB = tmpVgpr + 1
  rowId = tmpVgpr + 2
  rowOffsetA = tmpVgpr + 3
  rowOffsetB = tmpVgpr + 4
  waveId = tmpVgpr + 5
  laneId = tmpVgpr + 6
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=wavesize-1, comment=""))
  module.add(VAndB32(dst=vgpr(colIdA), src0=vgpr("Serial"), src1=(blockSize-1), comment="get col_id in wave for %uB load"%loadWidth))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  _grSwizzleColIds_cpp(module, writer, planA, planB, blockSize, numRowsPerLDSBanks,
                       laneId, colIdA, colIdB, waveId)
  _grComputeRowPartition_cpp(module, writer, tileInfoA, planA, waveId, rowOffsetA)
  _grComputeRowPartition_cpp(module, writer, tileInfoB, planB, waveId, rowOffsetB)
  _grComputeAllOffsets_cpp(module, writer, tileInfoA, planA, colIdA, rowId, rowOffsetA)
  _grComputeAllOffsets_cpp(module, writer, tileInfoB, planB, colIdB, rowId, rowOffsetB)
  writer.vgprPool.checkIn(tmpVgpr)
  _grComputeSubtileOffsets_cpp(writer, module, tileInfoA, planA)
  _grComputeSubtileOffsets_cpp(writer, module, tileInfoB, planB)
  return module

##################################################
# Subroutine to generate GR load code
#
def emitSingleBufferLoad(tileInfo, kernel, sId0, sId1):
  """Emit buffer_load instructions for a single subtile (sId0, sId1).

  When loadRatioGR > 1, multiple local subtiles share the same global read.
  Only the first subtile in each group emits the load; others return empty.

  Args:
      tileInfo: TileInfo or TileInfo for the tensor component
      sId0:     Subtile row index
      sId1:     Subtile column index (K-dimension)
  """
  # Instruction-shape plan (skip predicate, MUBUF offsetK, per-load m0 offsets)
  # computed by the C++ ABTileInfoQuery via TileInfo — pure data. Register
  # state (soffset/voff) is resolved here (writer-owned) and the rocisa
  # construction is done by the C++ ModuleBuilder.
  plan = tileInfo.singleBufferLoadPlan(sId0, sId1)
  if plan.skip:
    return Module()

  tc = tileInfo.tc
  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  regListIdx = tileInfo.grRegGroupForSubtileRow(sId0)
  regList = tileInfo.localSubtilesRegister[regListIdx]
  useSgpr = regList.is_sgpr

  # soffset is constant across loads; voff is per-load. Both are resolved from
  # the writer's register state and passed to the builder as operand objects.
  soffset = regList.ref(0) if len(regList) > 0 and useSgpr else 0
  voffs = [
      (tileInfo.sharedVgprGROffset[i] if useSgpr or len(regList) == 0
       else regList.indices[i])
      for i in range(len(plan.m0Offsets))
  ]
  return _mfma_builder().single_buffer_load(
      tc, isGlc, isSlc, isNT, plan.offsetK, plan.grBaseId,
      list(plan.m0Offsets), soffset, voffs)


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

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def globalReadDTLInitCommonSgpr(writer, kernel):
  return _globalReadDTLInitCommonSgpr_legacy(writer, kernel)

def _globalReadDTLInitCommonSgpr_legacy(writer, kernel):
  module = Module()
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1)
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  tmpVgpr = writer.vgprPool.checkOut(2)
  rowOffsetA = tmpVgpr
  rowOffsetB = tmpVgpr + 1
  planA = tileInfoA.grOffsetAssignPlan(writer)
  planB = tileInfoB.grOffsetAssignPlan(writer)
  _grComputeRowPartition_cpp(module, writer, tileInfoA, planA, vgprWaveId, rowOffsetA)
  _grComputeRowPartition_cpp(module, writer, tileInfoB, planB, vgprWaveId, rowOffsetB)
  subIterKBytes = tileInfoA.subIterKBytes
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
def globalReadLDSBufferSwap(tc, writer, kernel):
  if tc in ['A', 'B']:
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    return ti_.emitGRLDSBufferSwap(writer, kernel)
  else:
    ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
    return emitScaleGRLDSSwap(ti_, writer, kernel)

##################################################
# Subroutine to update ptrs
#
def globalReadPtrUpdates(tc, writer, kernel):
  ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return ti_.emitGRPtrUpdate(writer, kernel)


# ---------------------------------------------------------------------------
# Scale GR emit
# ---------------------------------------------------------------------------

def emitScaleGRLDSSwap(ti, writer, kernel):
  """Toggle scale GR DTL write target between double-buffer halves."""
  return _mfma_builder().gr_lds_buffer_swap(ti.tc)


def globalReadDoScaleSubtile(tc, writer, kernel):
  """Scale GR: load scale bytes global -> LDS via DTL BufferLoadB128."""
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return Module()

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  assert len(tileInfo.sharedVgprGROffset) > 0, "Scale GR requires at least 1 GR offset VGPR"

  return _mfma_builder().scale_gr_load(tc, isGlc, isSlc, isNT, tileInfo.sharedVgprGROffset[0])


def globalReadScalePtrUpdates(tc, writer, kernel):
  """Advance scale SRD base pointer by one depthU iteration."""
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  inc = int(ti_.lrSubtileSize * ti_.lrGlobalSubtileGrid[1])
  return _mfma_builder().scale_gr_ptr_update(tc, inc)

# emitSingleBufferLoad, emitScaleGRLDSSwap, globalReadPtrUpdates,
# globalReadLDSBufferSwap, globalReadDoScaleSubtile, globalReadScalePtrUpdates
# live in _gr_emit_leaves to break the Kernel <-> LogicalScheduler import cycle.
from ._gr_emit_leaves import (
    emitSingleBufferLoad, emitScaleGRLDSSwap, globalReadPtrUpdates,
    globalReadLDSBufferSwap, globalReadDoScaleSubtile, globalReadScalePtrUpdates,
)

def emitSubtileBufferLoad(tc, writer, kernel, subtileId):
  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return emitSingleBufferLoad(tileInfo, kernel, subtileId[0], subtileId[1])

################################################################################
# LR (local read) emit and alloc dispatch.
# (Inlined from SubtileLREmit.py — hipblaslt_incremental_refactor-grc.139)
#
# singledispatch over LR tag sentinels (LRTag_1x2, LRTag_TLU1, etc.).
# ABLRTile calls these via self.config.tag as the dispatch key.
################################################################################

_LR_MODULE_BUILDER = None


def _lr_builder():
  global _LR_MODULE_BUILDER
  if _LR_MODULE_BUILDER is None:
    _LR_MODULE_BUILDER = _ModuleBuilder()
  return _LR_MODULE_BUILDER


################################################################################
# LR Dispatch bases
################################################################################

@singledispatch
def _emitLocalReadOffset(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLocalReadOffset not implemented for {type(tag).__name__}")

@singledispatch
def _emitLocalRead(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLocalRead not implemented for {type(tag).__name__}")

@singledispatch
def _allocLROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"allocLROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _deallocLROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"deallocLROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _emitLRDTLInit(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLRDTLInit not implemented for {type(tag).__name__}")

@singledispatch
def _emitLRLDSBufferSwap(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLRLDSBufferSwap not implemented for {type(tag).__name__}")

# Stubs for tags not yet implemented.
_lr_stub = lambda tag, tile, ti, writer, kernel: None
_emitLocalReadOffset.register(LRTag_TLU1)(_lr_stub)
_emitLocalRead.register(LRTag_TLU1)(_lr_stub)
_allocLROffsetRegisters.register(LRTag_TLU1)(_lr_stub)
_deallocLROffsetRegisters.register(LRTag_TLU1)(_lr_stub)
_emitLRDTLInit.register(LRTag_TLU1)(_lr_stub)
_emitLRLDSBufferSwap.register(LRTag_TLU1)(_lr_stub)


################################################################################
# LR Helpers
################################################################################

def _setExecMask(module, writer, maskLo, maskHi):
  """Set EXEC mask to a 64-bit immediate value."""
  tmpSgpr = writer.sgprPool.checkOutAligned(2, 2, "setExecMask tmpSgpr", False)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(maskLo), comment="exec mask lo"))
  module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(maskHi), comment="exec mask hi"))
  module.add(SMovB64(dst=EXEC(), src=sgpr(tmpSgpr, 2), comment="Set exec mask"))
  writer.sgprPool.checkIn(tmpSgpr)

setExecMask = _setExecMask


################################################################################
# LR Implementations
################################################################################

# --- LR offset emit (TLU=0) --------------------------------------------------

@_emitLocalReadOffset.register(LRTag_1x1)
@_emitLocalReadOffset.register(LRTag_1x2)
def _emitLROffset_TLU0(tag, tile, ti, writer, kernel):
  return Module(f"LR Offset 1x2 ({ti.tc})")  # STUB


# --- LR alloc/dealloc (LRTag_1x2) -------------------------------------------

@_allocLROffsetRegisters.register(LRTag_1x1)
@_allocLROffsetRegisters.register(LRTag_1x2)
def _allocLROffsetRegs_1x2(tag, tile, ti, writer, kernel):
  """Allocate LR offset VGPRs (offset + swap) for row-major (TLU=0) subtile."""
  tile.sharedVgprLROffset = []
  tile.sharedVgprLROffsetSwap = []
  for i in range(ti.numLRPerSubtile):
    tile.sharedVgprLROffset.append(writer.vgprPool.checkOut(1))
    tile.sharedVgprLROffsetSwap.append(writer.vgprPool.checkOut(1))


@_deallocLROffsetRegisters.register(LRTag_1x1)
@_deallocLROffsetRegisters.register(LRTag_1x2)
def _deallocLROffsetRegs_1x2(tag, tile, ti, writer, kernel):
  """Deallocate LR offset registers."""
  if isinstance(tile.sharedVgprLROffset, list):
    for voff in tile.sharedVgprLROffset:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprLROffset = []
  if isinstance(tile.sharedVgprLROffsetSwap, list):
    for voff in tile.sharedVgprLROffsetSwap:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprLROffsetSwap = []


# --- LR load emit (LRTag_1x2) -----------------------------------------------

@_emitLocalRead.register(LRTag_1x1)
@_emitLocalRead.register(LRTag_1x2)
def _emitLR_1x2(tag, tile, ti, writer, kernel):
  return Module(f"LR Load 1x2 ({ti.tc})")  # STUB


# --- LR DTL init (LRTag_1x2) ------------------------------------------------

@_emitLRDTLInit.register(LRTag_1x1)
@_emitLRDTLInit.register(LRTag_1x2)
def _emitLRDTLInit_1x2(tag, tile, ti, writer, kernel):
  return Module(f"LR DTL Init ({ti.tc})")  # STUB


# --- LR LDS buffer swap (LRTag_1x2) -----------------------------------------

@_emitLRLDSBufferSwap.register(LRTag_1x1)
@_emitLRLDSBufferSwap.register(LRTag_1x2)
def _emitLRLDSSwap_1x2(tag, tile, ti, writer, kernel):
  """Toggle LR read offsets between double-buffer halves."""
  return _lr_builder().lr_lds_buffer_swap(
      ti.tc, list(tile.sharedVgprLROffset), list(tile.sharedVgprLROffsetSwap))


##################################################
# LR offset assignment
##################################################

def _computeLROffset_cpp(module, tileInfo, plan, colOffset, rowOffset):
  tc = tileInfo.tc
  loadWidth = plan.loadWidthLR
  numMFMACols = plan.numMFMACols
  blockSize = plan.blockSize
  module.add(VMovB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]), src=vgpr(colOffset), comment="%s: laneId"%tc))
  for vgprId in range(1, len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId-1]), src1=hex(numMFMACols), comment="%s: colOffset for MFMA %u of subtile"%(tc, vgprId)))
    module.add(VAndB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=hex(blockSize-1), comment="%s: colOffset = colOffset %% block_size"%tc))
  for vgprId in range(0, len(tileInfo.sharedVgprLROffset)):
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(tileInfo.sharedVgprLROffset[vgprId]), comment="%s: colOffset*loadWidth"%tc))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(rowOffset), comment="%s: row + col"%tc))

def _applyWavePartitionLROffset_cpp(module, writer, kernel, tileInfo, plan):
  tc = tileInfo.tc
  if plan.wavePartMode == -1:
    return
  wavesize = kernel["WavefrontSize"]
  waveId = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="waveId"))
  if plan.wavePartMode == 1:
    mWaves = plan.mWavesM
    if tc == 'A':
      module.add(VAndB32(dst=vgpr(waveId), src0=hex(mWaves - 1), src1=vgpr(waveId), comment="%s: waveId %% %d"%(tc, mWaves)))
    else:
      module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(mWaves.bit_length()-1), src=vgpr(waveId), comment="%s: waveId / %d"%(tc, mWaves)))
    sInterval = plan.sInterval
  elif plan.wavePartMode == 0:
    sInterval = plan.sInterval
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(plan.loadRatioGR))
  if sInterval == 0:
    writer.vgprPool.checkIn(waveId)
    return
  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(sInterval), comment="%s: interleave stride"%tc))
  module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr), comment=""))
  for vgprId in range(len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(waveId), comment="%s: wave partition LR offset"%tc))
  writer.vgprPool.checkIn(waveId)
  writer.sgprPool.checkIn(tmpSgpr)

def _lraWavePartitioning_cpp(module, writer, kernel, planA, planB):
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  _applyWavePartitionLROffset_cpp(module, writer, kernel, tileInfoA, planA)
  _applyWavePartitionLROffset_cpp(module, writer, kernel, tileInfoB, planB)

def _lraTileAssignment_fp8_cpp(writer, kernel, module, planA, planB):
  """FP8 LR offset: block-swap + wave de-rotation for MFMA 16x16x128."""
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  subIterKBytes = planA.subIterKBytes
  mi_m = planA.miM
  loadWidth = planA.loadWidthLR
  tmpVgpr = writer.vgprPool.checkOut(6)
  lane16, lane16Group, scratch, rowOffset, colOffset0, colOffset1 = range(tmpVgpr, tmpVgpr + 6)
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="lane16 = laneId % 16"))
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=kernel["WavefrontSize"]-1, comment="laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="lane16Group = laneId // 16"))
  module.add(VLShiftRightB32(dst=vgpr(scratch), shiftHex=hex(3), src=vgpr(lane16), comment="lane16 >> 3 (1 if M-row >= 8)"))
  module.add(VLShiftLeftB32(dst=vgpr(scratch), shiftHex=hex(1), src=vgpr(scratch), comment="rotation = 2 * (lane16 >> 3)"))
  module.add(VAddU32(dst=vgpr(colOffset0), src0=vgpr(lane16Group), src1=vgpr(scratch), comment="lane16Group + rotation"))
  module.add(VAndB32(dst=vgpr(colOffset0), src0=vgpr(colOffset0), src1=hex(3), comment="finalColId = (lane16Group + rotation) % 4"))
  module.add(VLShiftRightB32(dst=vgpr(scratch), shiftHex=hex(1), src=vgpr(lane16), comment="lane16 >> 1"))
  module.add(VAndB32(dst=vgpr(scratch), src0=vgpr(scratch), src1=hex(1), comment="swap_bit"))
  module.add(VLShiftLeftB32(dst=vgpr(scratch), shiftHex=hex(2), src=vgpr(scratch), comment="swap_val = swap_bit * 4"))
  module.add(VAddU32(dst=vgpr(colOffset0), src0=vgpr(colOffset0), src1=vgpr(scratch), comment="colOffset_0 = finalColId + swap_val"))
  module.add(VXorB32(dst=vgpr(colOffset1), src0=vgpr(colOffset0), src1=hex(4), comment="colOffset_1 = colOffset_0 ^ 4"))
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(subIterKBytes.bit_length()-1), src=vgpr(lane16), comment=f"rowOffset = lane16 * {subIterKBytes}"))
  for tileInfo in [tileInfoA, tileInfoB]:
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]),
               shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colOffset0),
               comment=f"{tileInfo.tc}: col0 * {loadWidth}"))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[0]),
               src0=vgpr(tileInfo.sharedVgprLROffset[0]), src1=vgpr(rowOffset),
               comment=f"{tileInfo.tc}: offset[0]"))
    if len(tileInfo.sharedVgprLROffset) > 1:
      module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[1]),
                 shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colOffset1),
                 comment=f"{tileInfo.tc}: col1 * {loadWidth}"))
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[1]),
                 src0=vgpr(tileInfo.sharedVgprLROffset[1]), src1=vgpr(rowOffset),
                 comment=f"{tileInfo.tc}: offset[1]"))
  writer.vgprPool.checkIn(tmpVgpr)
  _lraWavePartitioning_cpp(module, writer, kernel, planA, planB)
  stmp = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsStartOffsetB, comment="ldsStartOffsetB"))
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               src0=sgpr(stmp),
               src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               comment="B matrix offset in LDS"))
  writer.sgprPool.checkIn(stmp)
  return module


def _lraTileAssignment_cpp(writer, kernel):
  module = Module()
  module.addComment0("LR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  planA = tileInfoA.lrOffsetAssignPlan(writer, kernel)
  planB = tileInfoB.lrOffsetAssignPlan(writer, kernel)
  if planA.isFp8:
    return _lraTileAssignment_fp8_cpp(writer, kernel, module, planA, planB)
  subIterKBytes = planA.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  mi_m = planA.miM
  numRowsPerLDSBanks = planA.numRowsPerLDSBanks
  blockSize = planA.blockSize
  tmpVgpr = writer.vgprPool.checkOut(6)
  lane16, lane16Group, rotation, rowOffset, colOffset = range(tmpVgpr, tmpVgpr + 5)
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize-1, comment="laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="lane16Group"))
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="laneId %% 16"))
  module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(lane16), comment="lds_row_id"))
  module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="(lds_row_id //2 )"))
  module.add(VLShiftLeftB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="rotation=(lds_row_id //2) * 2"))
  module.add(VAddU32(dst=vgpr(colOffset), src0=vgpr(rotation), src1=vgpr(lane16Group), comment="colOffset = rotation + lane16Group"))
  module.add(VAndB32(dst=vgpr(colOffset), src0=vgpr(colOffset), src1=hex(blockSize-1), comment="colOffset = colOffset %% blockSize"))
  setExecMask(module, writer, 0x33333333, 0x33333333)
  module.add(VPermlane16SwapB32(dst=vgpr(colOffset), src=vgpr(colOffset), comment="apply swizzling"))
  setExecMask(module, writer, -1, -1)
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(subIterKBytes.bit_length()-1), src=vgpr(lane16), comment="offsetRow = subIterKBytes*lane16"))
  _computeLROffset_cpp(module, tileInfoA, planA, colOffset, rowOffset)
  _computeLROffset_cpp(module, tileInfoB, planB, colOffset, rowOffset)
  writer.vgprPool.checkIn(tmpVgpr)
  _lraWavePartitioning_cpp(module, writer, kernel, planA, planB)
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), src0=writer.ldsStartOffsetB, src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), comment="B matrix offset in LDS"))
  return module


def lraTileAssignment(writer, kernel):
  return _lraTileAssignment_cpp(writer, kernel)


def localReadResetOffsetsSubtile(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based LR offset reset code")
  for i in range(8):
    module.addComment("")
  return module


def emitSubtileDsRead(writer, kernel, tileInfo, subtileId):
  module = Module()
  sId0 = subtileId[0]
  sId1 = subtileId[1]
  for du in range(tileInfo.subtileShape[1]):
    mfmaId = tileInfo.getSubtileShapeLinearId(du, 0)
    tileIdx = tileInfo.lrTileIndexForSubtile(sId0, sId1, mfmaId)
    dstTile = tileInfo.vgprTiles[tileIdx]
    module.add(emitSingleDsRead(tileInfo, sId0, sId1, du, dstTile))
  return module


def localReadDoSubtile(tc, writer, kernel):
  module = Module()
  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  for i in range(tileInfo.localSubtileGrid[0]):
    for j in range(tileInfo.localSubtileGrid[1]):
        module.add(emitSubtileDsRead(writer, kernel, tileInfo, [i, j]))
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


# emitSingleDsRead, localReadLDSBufferSwap, emitScaleLRLDSSwap, emitScaleDsRead
# live in _lr_emit_leaves to break the Kernel ↔ LogicalScheduler import cycle.
from ._lr_emit_leaves import (
    emitSingleDsRead, localReadLDSBufferSwap,
    emitScaleLRLDSSwap, emitScaleDsRead,
)



# ---------------------------------------------------------------------------
# LR subroutines (inlined from SubtileLREmit.py — hipblaslt_incremental_refactor-grc.139)
# ---------------------------------------------------------------------------


##################################################
# Subroutine to generate LR offset calculation code
#
def lraTileAssignment(writer, kernel):
  # The AB LR offset-assignment scalar math is computed by the C++
  # ABTileInfoQuery.lrOffsetAssignPlan for every AB geometry — BF16/B16,
  # FP4/B4, FP8/B8 (distinct block-swap routine), and the TLU1 BF16 variants.
  # The rocisa emission stays here.
  return _lraTileAssignment_cpp(writer, kernel)


# --- C++-plan-driven LR offset assignment (all AB geometries) ---------------
#
# Source every derived scalar (blockSize, numRowsPerLDSBanks, MFMA column
# stride, wave-partition stride/selector, FP8 routine selector) from the C++
# ABTileInfoQuery.lrOffsetAssignPlan. Register state (sharedVgprLROffset) and
# all rocisa construction stay in Python.

def _computeLROffset_cpp(module, tileInfo, plan, colOffset, rowOffset):
  tc = tileInfo.tc
  loadWidth = plan.loadWidthLR
  numMFMACols = plan.numMFMACols
  blockSize = plan.blockSize
  module.add(VMovB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]), src=vgpr(colOffset), comment="%s: laneId"%tc))
  for vgprId in range(1, len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId-1]), src1=hex(numMFMACols), comment="%s: colOffset for MFMA %u of subtile"%(tc, vgprId)))
    module.add(VAndB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=hex(blockSize-1), comment="%s: colOffset = colOffset %% block_size"%tc))
  for vgprId in range(0, len(tileInfo.sharedVgprLROffset)):
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(tileInfo.sharedVgprLROffset[vgprId]), comment="%s: colOffset*loadWidth"%tc))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(rowOffset), comment="%s: row + col"%tc))

def _applyWavePartitionLROffset_cpp(module, writer, kernel, tileInfo, plan):
  tc = tileInfo.tc
  if plan.wavePartMode == -1:   # loadRatioGR >= 2.0: no partition
    return
  wavesize = kernel["WavefrontSize"]
  waveId = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="waveId"))
  if plan.wavePartMode == 1:    # loadRatioGR == 1.0
    mWaves = plan.mWavesM
    if tc == 'A':
      module.add(VAndB32(dst=vgpr(waveId), src0=hex(mWaves - 1), src1=vgpr(waveId), comment="%s: waveId %% %d"%(tc, mWaves)))
    else:
      module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(mWaves.bit_length()-1), src=vgpr(waveId), comment="%s: waveId / %d"%(tc, mWaves)))
    sInterval = plan.sInterval
  elif plan.wavePartMode == 0:  # loadRatioGR == 0.5
    sInterval = plan.sInterval
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(plan.loadRatioGR))
  if sInterval == 0:
    writer.vgprPool.checkIn(waveId)
    return
  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(sInterval), comment="%s: interleave stride"%tc))
  module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr), comment=""))
  for vgprId in range(len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(waveId), comment="%s: wave partition LR offset"%tc))
  writer.vgprPool.checkIn(waveId)
  writer.sgprPool.checkIn(tmpSgpr)

def _lraWavePartitioning_cpp(module, writer, kernel, planA, planB):
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  _applyWavePartitionLROffset_cpp(module, writer, kernel, tileInfoA, planA)
  _applyWavePartitionLROffset_cpp(module, writer, kernel, tileInfoB, planB)

def _lraTileAssignment_fp8_cpp(writer, kernel, module, planA, planB):
  """FP8 LR offset: block-swap + wave de-rotation for MFMA 16x16x128.

  Two ds_read_b128 per MFMA (numLRPerSubtile=2), using complementary block
  assignments to achieve zero LDS bank conflicts:
    finalColId  = (lane16Group + 2*(lane16 >> 3)) % 4  [undo GR wave rotation]
    colOffset_0 = finalColId + swap_bit * 4
    colOffset_1 = colOffset_0 ^ 4
  where:
    swap_bit = (lane16 >> 1) & 1

  The rotation 2*(lane16>>3) undoes the GR step 2 wave K_group rotation:
  waves with waveId&1==1 (M-rows 8..15) wrote with rotation=2; lane16>=8
  reads them back with de-rotation=2. Together they achieve zero bank conflicts.

  Scalar math (subIterKBytes, miM, loadWidthLR, wave-partition stride/selector)
  is sourced from the C++ lrOffsetAssignPlan; register state and rocisa
  construction stay in Python.
  """
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  subIterKBytes = planA.subIterKBytes
  mi_m = planA.miM
  loadWidth = planA.loadWidthLR
  tmpVgpr = writer.vgprPool.checkOut(6)
  lane16, lane16Group, scratch, rowOffset, colOffset0, colOffset1 = range(tmpVgpr, tmpVgpr + 6)
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="lane16 = laneId % 16"))
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=kernel["WavefrontSize"]-1, comment="laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="lane16Group = laneId // 16"))
  module.add(VLShiftRightB32(dst=vgpr(scratch), shiftHex=hex(3), src=vgpr(lane16), comment="lane16 >> 3 (1 if M-row >= 8)"))
  module.add(VLShiftLeftB32(dst=vgpr(scratch), shiftHex=hex(1), src=vgpr(scratch), comment="rotation = 2 * (lane16 >> 3)"))
  module.add(VAddU32(dst=vgpr(colOffset0), src0=vgpr(lane16Group), src1=vgpr(scratch), comment="lane16Group + rotation"))
  module.add(VAndB32(dst=vgpr(colOffset0), src0=vgpr(colOffset0), src1=hex(3), comment="finalColId = (lane16Group + rotation) % 4"))
  module.add(VLShiftRightB32(dst=vgpr(scratch), shiftHex=hex(1), src=vgpr(lane16), comment="lane16 >> 1"))
  module.add(VAndB32(dst=vgpr(scratch), src0=vgpr(scratch), src1=hex(1), comment="swap_bit"))
  module.add(VLShiftLeftB32(dst=vgpr(scratch), shiftHex=hex(2), src=vgpr(scratch), comment="swap_val = swap_bit * 4"))
  module.add(VAddU32(dst=vgpr(colOffset0), src0=vgpr(colOffset0), src1=vgpr(scratch), comment="colOffset_0 = finalColId + swap_val"))
  module.add(VXorB32(dst=vgpr(colOffset1), src0=vgpr(colOffset0), src1=hex(4), comment="colOffset_1 = colOffset_0 ^ 4"))
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(subIterKBytes.bit_length()-1), src=vgpr(lane16), comment=f"rowOffset = lane16 * {subIterKBytes}"))
  for tileInfo in [tileInfoA, tileInfoB]:
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]),
               shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colOffset0),
               comment=f"{tileInfo.tc}: col0 * {loadWidth}"))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[0]),
               src0=vgpr(tileInfo.sharedVgprLROffset[0]), src1=vgpr(rowOffset),
               comment=f"{tileInfo.tc}: offset[0]"))
    if len(tileInfo.sharedVgprLROffset) > 1:
      module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[1]),
                 shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colOffset1),
                 comment=f"{tileInfo.tc}: col1 * {loadWidth}"))
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[1]),
                 src0=vgpr(tileInfo.sharedVgprLROffset[1]), src1=vgpr(rowOffset),
                 comment=f"{tileInfo.tc}: offset[1]"))
  writer.vgprPool.checkIn(tmpVgpr)
  _lraWavePartitioning_cpp(module, writer, kernel, planA, planB)
  stmp = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsStartOffsetB, comment="ldsStartOffsetB"))
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               src0=sgpr(stmp),
               src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               comment="B matrix offset in LDS"))
  writer.sgprPool.checkIn(stmp)
  return module


def _lraTileAssignment_cpp(writer, kernel):
  module = Module()
  module.addComment0("LR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  planA = tileInfoA.lrOffsetAssignPlan(writer, kernel)
  planB = tileInfoB.lrOffsetAssignPlan(writer, kernel)
  if planA.isFp8:  # FP8: block-swap swizzle, no VPermlane16Swap
    return _lraTileAssignment_fp8_cpp(writer, kernel, module, planA, planB)
  subIterKBytes = planA.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  mi_m = planA.miM
  numRowsPerLDSBanks = planA.numRowsPerLDSBanks
  blockSize = planA.blockSize
  tmpVgpr = writer.vgprPool.checkOut(6)
  lane16, lane16Group, rotation, rowOffset, colOffset = range(tmpVgpr, tmpVgpr + 5)
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize-1, comment="laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="lane16Group"))
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="laneId %% 16"))
  module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(lane16), comment="lds_row_id"))
  module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="(lds_row_id //2 )"))
  module.add(VLShiftLeftB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="rotation=(lds_row_id //2) * 2"))
  module.add(VAddU32(dst=vgpr(colOffset), src0=vgpr(rotation), src1=vgpr(lane16Group), comment="colOffset = rotation + lane16Group"))
  module.add(VAndB32(dst=vgpr(colOffset), src0=vgpr(colOffset), src1=hex(blockSize-1), comment="colOffset = colOffset %% blockSize"))
  setExecMask(module, writer, 0x33333333, 0x33333333)
  module.add(VPermlane16SwapB32(dst=vgpr(colOffset), src=vgpr(colOffset), comment="apply swizzling"))
  setExecMask(module, writer, -1, -1)
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(subIterKBytes.bit_length()-1), src=vgpr(lane16), comment="offsetRow = subIterKBytes*lane16"))
  _computeLROffset_cpp(module, tileInfoA, planA, colOffset, rowOffset)
  _computeLROffset_cpp(module, tileInfoB, planB, colOffset, rowOffset)
  writer.vgprPool.checkIn(tmpVgpr)
  _lraWavePartitioning_cpp(module, writer, kernel, planA, planB)
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), src0=writer.ldsStartOffsetB, src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), comment="B matrix offset in LDS"))
  return module


def localReadResetOffsetsSubtile(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based LR offset reset code")
  for i in range(8):
    module.addComment("")

  return module


def emitSingleDsRead(tileInfo, sId0, sId1, subIterK, dstTile):
  """Emit DSLoadB128 instruction(s) for one MMA tile within a subtile.

  Args:
      tileInfo:  TileInfo (for subtileSize, loadRatioGR, sharedVgprLROffset, tc)
      sId0:      Subtile row index (used for offset computation)
      subIterK:  subIterK index within the subtile (maps to mfmaC; subtileShape[0]=1 so mfmaR=0)
      dstTile:   RegisterTileInfo — destination vgpr tile for the load

  Returns a Module. For tiles with numRegs > 4 (e.g. FP8 8-VGPR tiles), emits
  multiple ds_read_b128 instructions (one per 4 VGPRs), each using the next
  sharedVgprLROffset entry.
  """
  dstVgpr = dstTile.regList.indices[0]
  numRegs = len(dstTile.regList.indices)

  # Instruction-shape plan (DS offset, register stride, per-read map) computed
  # by the C++ ABTileInfoQuery via TileInfo — pure data. The destination VGPR
  # base and the sharedVgprLROffset registers are writer-owned register state,
  # resolved here and passed to the C++ ModuleBuilder which does the rocisa
  # construction.
  plan = tileInfo.singleDsReadPlan(sId0, sId1, subIterK, numRegs)
  dstRegOffsets = [rd.dstRegOffset for rd in plan.reads]
  addrVgprs = [tileInfo.sharedVgprLROffset[rd.addrIdx] for rd in plan.reads]
  return _lr_builder().single_ds_read(
      tileInfo.tc, sId0, sId1, subIterK, dstVgpr, plan.regsPerDsRead,
      plan.offset, dstRegOffsets, addrVgprs)


def emitSubtileDsRead(writer, kernel, tileInfo, subtileId):

  module = Module()
  sId0 = subtileId[0]
  sId1 = subtileId[1]

  # Emit one ds_read group per K-direction MMA tile in the subtile. Each group
  # is built by emitSingleDsRead from the C++ ABTileInfoQuery plan; with
  # subtileShape[0]==1 the per-du address indices are contiguous, matching the
  # previous flat lrOffsetIdx walk byte-for-byte.
  for du in range(tileInfo.subtileShape[1]):
    mfmaId = tileInfo.getSubtileShapeLinearId(du, 0)
    tileIdx = tileInfo.lrTileIndexForSubtile(sId0, sId1, mfmaId)
    dstTile = tileInfo.vgprTiles[tileIdx]
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
def localReadLDSBufferSwap(tc, writer, kernel):
  if tc in ['A', 'B']:
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    return ti_.emitLRLDSBufferSwap(writer, kernel)
  else:
    ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
    return emitScaleLRLDSSwap(ti_, writer, kernel)


# ---------------------------------------------------------------------------
# Scale LR emit
# ---------------------------------------------------------------------------

def emitScaleLRLDSSwap(ti, writer, kernel):
  """Toggle scale LR read offsets between double-buffer halves."""
  return _lr_builder().lr_lds_buffer_swap(
      ti.tc, list(ti.sharedVgprLROffset), list(ti.sharedVgprLROffsetSwap))


def emitScaleDsRead(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k=-1):
  """Scale LR: read 4 scale bytes (one E8M0 group) from LDS via ds_read_b32."""
  return _lr_builder().scale_ds_read(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k)


# ---------------------------------------------------------------------------
# Scale offset-assignment and LR emit
# ---------------------------------------------------------------------------

def localReadDoScaleSubtile(tc, writer, kernel):
  """Emit scale ds_reads for all scale groups (PGR=0 path)."""
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  if tileInfo.mxBlock == 0:
    return module

  if hasattr(tileInfo, 'lrSubtileSize'):
    groupStride = int(tileInfo.lrSubtileSize)
  else:
    groupStride = 2 * tileInfo.subtileSize

  numScaleGroups = math.ceil(tileInfo.localSubtileGrid[0] / 2) * tileInfo.localSubtileGrid[1]
  for gid in range(numScaleGroups):
    dsOffset = groupStride * gid
    vdst = tileInfo.vgprTiles[4 * gid].regList.indices[0]
    module.add(emitScaleDsRead(tc, vdst, tileInfo.sharedVgprLROffset[0], dsOffset, gid))

  return module


def _graScaleOffset_cpp(tc, writer, kernel):
  module = Module()
  module.addComment("Computing GR Offset for %s"%tc)
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  plan = ti_.scaleGrOffsetAssignPlan()
  loadWidth = plan.loadWidth
  loadWidthShift = loadWidth.bit_length() - 1
  numThreadsPerGroup = plan.numThreadsPerGroup
  bpe = plan.bpe
  vtmp = writer.vgprPool.checkOut(1)
  stmp = writer.sgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(vtmp),
                            shiftHex=hex(int(math.log2(numThreadsPerGroup))), src=vgpr("Serial"),
                            comment="%s: grOffset = serial / %d" % (tc, loadWidth)))
  module.add(SLShiftLeftB32(sgpr(stmp), int(math.log2(bpe)), sgpr("Strides%s"%tc), comment="*= bpe (%d)"%bpe))
  module.add(VMulLOU32(dst=vgpr(vtmp), src1=vgpr(vtmp), src0=sgpr(stmp), comment="Apply scale%s stride to each group"%tc))
  module.add(VAndB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                     src0=hex(numThreadsPerGroup - 1), src1=vgpr("Serial"),
                     comment="%s: grOffset = serial %% %d" % (tc, loadWidth)))
  module.add(VLShiftLeftB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                            shiftHex=hex(loadWidthShift), src=vgpr(ti_.sharedVgprGROffset[0]),
                            comment="Scale by load width for each thread in group"))
  module.add(VAddU32(dst=vgpr(ti_.sharedVgprGROffset[0]), src0=vgpr(ti_.sharedVgprGROffset[0]), src1=vgpr(vtmp), comment="Final offset calc"))
  writer.vgprPool.checkIn(vtmp)
  writer.sgprPool.checkIn(stmp)
  return module


def graTileAssignmentScaleSwizzled(writer, kernel):
  """Generate GR offset calculation for scaleA/B (DTL)."""
  module = Module()
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module
  module.add(_graScaleOffset_cpp('MXSA', writer, kernel))
  module.add(_graScaleOffset_cpp('MXSB', writer, kernel))
  return module


def _applyScaleWavePartitionLROffset_cpp(module, writer, ti_, plan, waveId):
  tc = ti_.tc
  tmpSgpr = writer.sgprPool.checkOut(1)
  tmp = writer.vgprPool.checkOut(2)
  if plan.isA:
    module.add(VAndB32(dst=vgpr(tmp), src0=plan.mWavesM-1, src1=vgpr(waveId), comment="scale%s: waveId %% %d"%(tc, plan.mWavesM)))
  else:
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=int(math.log2(plan.mWavesM)), src=vgpr(waveId), comment="scale%s: waveId / numWavesM"%tc))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=plan.totalScaleBytes, comment="scale%s: scale region"%tc))
  module.add(VMulLOU32(dst=vgpr(ti_.sharedVgprLROffset[0]), src0=sgpr(tmpSgpr), src1=vgpr(tmp), comment="scale%s: partition offset"%tc))
  writer.vgprPool.checkIn(tmp)
  writer.sgprPool.checkIn(tmpSgpr)


def lraTileAssignmentScaleSwizzled(writer, kernel):
  """Generate LR offset calculation for scaleA/B."""
  module = Module()
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module
  tiA_ = writer.states.mxsa.tileInfo
  tiB_ = writer.states.mxsb.tileInfo
  planA = tiA_.scaleLrOffsetAssignPlan(kernel)
  planB = tiB_.scaleLrOffsetAssignPlan(kernel)
  module.addComment0("LR Offset Calculation for Scale Tensors")
  wavesize = kernel["WavefrontSize"]
  waveIdVgpr = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="scale: waveId"))
  _applyScaleWavePartitionLROffset_cpp(module, writer, tiA_, planA, waveIdVgpr)
  _applyScaleWavePartitionLROffset_cpp(module, writer, tiB_, planB, waveIdVgpr)
  writer.vgprPool.checkIn(waveIdVgpr)
  laneOffset = writer.vgprPool.checkOut(1)
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1, comment="scale: laneId"))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset), comment="scale: laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiA_.sharedVgprLROffset[0]), comment="scaleA: lrOffset = laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiB_.sharedVgprLROffset[0]), comment="scaleB: lrOffset = laneId * 4"))
  writer.vgprPool.checkIn(laneOffset)
  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSA), comment="scale: LDS offset for A scale"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(tiA_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleA: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSB), comment="scale: LDS offset for B scale"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(tiB_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleB: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=writer.ldsTotalSize, comment="scale: total LDS size for swap"))
  for ti_ in [tiA_, tiB_]:
    for i in range(len(ti_.sharedVgprLROffset)):
      vgprId     = ti_.sharedVgprLROffset[i]
      vgprSwapId = ti_.sharedVgprLROffsetSwap[i]
      module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(tmpSgpr), comment="scale%s: LR swap"%ti_.tc))
      module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment="scale%s: LR swap"%ti_.tc))
  writer.sgprPool.checkIn(tmpSgpr)
  return module


def globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel):
  """Compute shared offsets used by m0 in DTL loads."""
  module = Module()
  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1)
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  tiMXSA_ = writer.states.mxsa.tileInfo
  tiMXSB_ = writer.states.mxsb.tileInfo
  loadWidth = tiMXSA_.loadWidthGR
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
}

def selectABGeometry(kernel: dict, tc: str) -> ABTilePair:
  """Return the ABTilePair selected by Solution.py for tc ('A' or 'B')."""
  key = kernel[f"_ABTilePair{tc}"]
  return AB_GEOMETRY_MAP[key]


def selectDGeometry(kernel: dict) -> CDTileGeometry:
  """Return the CDTileGeometry for the D (output/accumulator) tile."""
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
      # Materialized GR/LR configs retained for the C++ scale offset-assignment
      # query layer (_cppScaleQuery); GR is kernel-materialized (subtileShape).
      self._scaleGrCfg = gr_cfg
      self._scaleLrCfg = lr_cfg
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

  # --- C++-backed read-only query layer (AB case only) ---

  def _cppQuery(self):
    """Build (and cache) the C++ ABTileInfoQuery twin for this TileInfo.

    The AB (ABTilePair) query layer is C++-only: the twin is constructed from
    the already-materialized GR/LR configs plus the kernel-derived scalar
    fields, reusing the C++ geometry surface rather than duplicating any
    geometry formula. MX scale and C/D geometries are out of scope for the C++
    query layer and must not call this.
    """
    if not isinstance(self.geometry, ABTilePair):
      raise NotImplementedError(
          "TileInfo C++ query layer covers the ABTilePair case only; "
          f"got {type(self.geometry).__name__} (tc={self.tc})")
    q = getattr(self, '_cppQueryCache', None)
    if q is None:
      cpp_gr = self.gr.config._cpp
      cpp_lr = self.lr.config._cpp
      q = _CPP_TI.ABTileInfoQuery(cpp_gr, cpp_lr, self.macroTile, self.depthU,
                                  self.waveGroupSize, self.waveSize, self.numWaves)
      self._cppQueryCache = q
    return q

  # --- Grid utility methods ---

  def getLocalSubtileLinearId(self, sId0, sId1):
    return self._cppQuery().getLocalSubtileLinearId(sId0, sId1)

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
    return self._cppQuery().grLoadIndexForSubtile(sId0, sId1, loadIdx)

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
    return self._cppQuery().lrTileIndexForSubtile(sId0, sId1, mfmaId)

  def globalMmaTilesForSubtile(self, sId0, sId1):
    """Return all global MMA tile coordinates belonging to subtile (sId0, sId1).

    Uses gr_cfg.subtileForMmaTile to account for subtileCount/subtileStride.
    The returned list is in geometric order (M-outer, K-inner).
    """
    return [tuple(t) for t in self._cppQuery().globalMmaTilesForSubtile(sId0, sId1)]

  def waveMmaTilesForSubtile(self, sId0, sId1):
    """Return the local MMA tile coordinates this wave uses from subtile (sId0, sId1).

    Each wave covers localMMATileGrid rows; within that, subtile (sId0, sId1)
    spans subtileShape[0] rows x subtileShape[1] columns of MMA tiles.
    """
    return [tuple(t) for t in self._cppQuery().waveMmaTilesForSubtile(sId0, sId1)]

  def grRegGroupForSubtileRow(self, sId0):
    """Return the GR offset-register group index for subtile row sId0.

    Offset VGPRs for GR buffer_loads are stored in localSubtilesRegister,
    grouped by load.  When loadRatioGR >= 2, multiple subtile rows share
    the same buffer_load and therefore the same register group.
    """
    return self._cppQuery().grRegGroupForSubtileRow(sId0)

  # --- Emit-leaf plans (instruction shape only) ---
  # These compute the pure-data decisions for the buffer-load / ds-read emit
  # leaves (skip predicate, m0/DS offsets, per-instruction loop structure).
  # The emit functions in Kernel.py build the rocisa Module from the returned
  # plan. For the AB case the plans are computed by the C++ ABTileInfoQuery
  # (no parallel Python formula).

  def singleBufferLoadPlan(self, sId0, sId1):
    """Plan for emitSingleBufferLoad: skip flag, MUBUF offsetK, m0 offsets."""
    return self._cppQuery().singleBufferLoadPlan(sId0, sId1)

  def singleDsReadPlan(self, sId0, sId1, subIterK, numRegs):
    """Plan for emitSingleDsRead: DS offset, register stride, per-read map."""
    return self._cppQuery().singleDsReadPlan(sId0, sId1, subIterK, numRegs)

  # --- GR/LR offset-assignment math (C++-only for all AB geometries) ---
  # The scalar offset-assignment math for graTileAssignment / lraTileAssignment
  # is computed by the C++ ABTileInfoQuery for every AB (ABTilePair) geometry —
  # BF16/B16, FP4/B4, FP8/B8 (FP8 swizzle selected by the plan's isFp8 flag),
  # and the column-major TLU1 BF16 variants. There is no Python scalar-math twin
  # and no env switch; the rocisa emission stays in Kernel.py.

  def grOffsetAssignPlan(self, writer):
    """C++ GR offset-assignment scalar plan for this tensor (all AB geometries)."""
    ldsRowBankSize = (writer.states.archCaps["LDSBankCount"]
                      * writer.states.archCaps["LDSBankWidth"])
    return self._cppQuery().grOffsetAssignPlan(ldsRowBankSize)

  def lrOffsetAssignPlan(self, writer, kernel):
    """C++ LR offset-assignment scalar plan for this tensor (all AB geometries)."""
    ldsRowBankSize = (writer.states.archCaps["LDSBankCount"]
                      * writer.states.archCaps["LDSBankWidth"])
    mWavesM = kernel["MIWaveGroup"][0]
    return self._cppQuery().lrOffsetAssignPlan(ldsRowBankSize, mWavesM)

  # --- MX scale offset-assignment math (C++-only for MXScaleTilePair) ---
  # The swizzled-scale GR/LR offset-assignment scalar math for
  # graTileAssignmentScaleSwizzled / lraTileAssignmentScaleSwizzled is computed
  # by the C++ MXScaleTileInfoQuery for the gfx950 scale geometries (MXFP4 /
  # MXFP8). There is no Python scalar-math twin; the rocisa emission stays in
  # Kernel.py (scale offset-assignment section).

  def _cppScaleQuery(self):
    """Build (and cache) the C++ MXScaleTileInfoQuery twin for this scale TileInfo.

    The MX scale (MXScaleTilePair) offset-assignment query layer is C++-only:
    the twin is constructed from the already-materialized GR/LR scale configs
    plus the kernel-derived scalar fields, reusing the C++ geometry surface.
    AB and C/D geometries are out of scope and must not call this.
    """
    if not isinstance(self.geometry, MXScaleTilePair):
      raise NotImplementedError(
          "TileInfo C++ scale query layer covers the MXScaleTilePair case "
          f"only; got {type(self.geometry).__name__} (tc={self.tc})")
    q = getattr(self, '_cppScaleQueryCache', None)
    if q is None:
      cpp_gr = self._scaleGrCfg._cpp
      cpp_lr = self._scaleLrCfg._cpp
      q = _CPP_TI.MXScaleTileInfoQuery(cpp_gr, cpp_lr, self.macroTile,
                                       self.depthU, self.waveGroupSize,
                                       self.waveSize, self.numWaves)
      self._cppScaleQueryCache = q
    return q

  def scaleGrOffsetAssignPlan(self):
    """C++ swizzled-scale GR offset-assignment scalar plan for this tensor."""
    return self._cppScaleQuery().scaleGrOffsetAssignPlan()

  def scaleLrOffsetAssignPlan(self, kernel):
    """C++ swizzled-scale LR offset-assignment scalar plan for this tensor."""
    mWavesM = kernel["MIWaveGroup"][0]
    return self._cppScaleQuery().scaleLrOffsetAssignPlan(mWavesM,
                                                         self.tc == 'MXSA')

  # --- Register allocation ---

  def allocOffsetRegisters(self, writer, kernel):
    if self.gr is not None:
      self.gr.allocOffsetRegisters(self, writer, kernel)
    if self.lr is not None:
      self.lr.allocOffsetRegisters(self, writer, kernel)
    # MXScaleTilePair offset registers
    # managed by scale-specific alloc below (MXScaleTilePair)
    if isinstance(self.geometry, MXScaleTilePair):
      self._sharedVgprGROffset = [writer.vgprPool.checkOut(1)]
      self._sharedVgprLROffset = [writer.vgprPool.checkOut(1)]
      self._sharedVgprLROffsetSwap = [writer.vgprPool.checkOut(1)]

  def allocVgprTileRegisters_legacy(self, writer, kernel):
    """Allocate data tile registers for A/B/D MMA operands.
    """
    self.vgprTiles = []
    numMMATiles = int(self.localMMATileGrid[0] * self.localMMATileGrid[1])
    numMMATilesPerReg = max(1, int(1 // self.mmaTileRegCount))
    # Scale tiles: legacy MXSA/MXSB used bpe=1 (scale byte) which gives mmaTileRegCount=0.25
    # and numMMATilesPerReg=4. TileInfo uses data bpe (0.5 for f4), halving mmaTileRegCount.
    # The scale emit code (localReadDoScaleSubtile) uses stride 4 to index vgprTiles, so override.
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
      vstart = pool.checkOutAligned(numDword, numDword)
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


def initVgprTilesToZero(writer, kernel, tileInfo):
  """Initialize vgprTiles to zero using MFMA for blocks of 16, scalar writes for remainder.

  Delegates rocisa instruction construction to the C++ loop_orchestrator.
  Python resolves pool identity (AGPR vs VGPR) and allocates a single tmpVgpr
  pair shared across all MFMA-eligible groups; C++ uses the index but does not
  call checkOut/checkIn.  Returns the rocisa Module from C++ directly.
  """
  from tensile_writer.subtile.loop_orchestrator import init_vgpr_tiles_to_zero
  from tensile_writer.subtile.module_builder import ModuleBuilder

  builder = ModuleBuilder()

  if not tileInfo.vgprTiles:
    return init_vgpr_tiles_to_zero(builder, tileInfo.tc, [])

  # Group contiguous vgprTiles by pool type (agpr vs vgpr); D tiles can use both.
  # Python resolves pool identity here; C++ receives plain
  # (firstReg, totalRegs, isAgpr, tmpVgpr) tuples.
  groups = []
  firstReg = tileInfo.vgprTiles[0].regList.indices[0]
  totalRegs = 0
  curPool = tileInfo.vgprTiles[0].regList.pool

  for tile in tileInfo.vgprTiles:
    pool = tile.regList.pool
    numRegs = len(tile.regList.indices)
    if pool != curPool:
      groups.append((firstReg, totalRegs, curPool == writer.agprPool))
      firstReg = tile.regList.indices[0]
      totalRegs = numRegs
      curPool = pool
    else:
      totalRegs += numRegs
  groups.append((firstReg, totalRegs, curPool == writer.agprPool))

  # Allocate a single tmpVgpr pair shared across all MFMA-eligible groups.
  # zero_reg_range re-zeroes tmpVgpr at the start of each group (vmov_b64),
  # so one pair suffices regardless of the number of groups.  Allocating one
  # pair (vs one per group) keeps peak VGPR liveness equivalent to the old
  # per-group checkOut/checkIn pattern.
  needs_mfma = any(totalRegs >= 16 for _, totalRegs, _ in groups)
  tmpVgpr = writer.vgprPool.checkOutAligned(2, 2) if needs_mfma else -1

  reg_groups = [
      (firstReg, totalRegs, isAgpr, tmpVgpr if totalRegs >= 16 else -1)
      for firstReg, totalRegs, isAgpr in groups
  ]

  result = init_vgpr_tiles_to_zero(builder, tileInfo.tc, reg_groups)

  if tmpVgpr >= 0:
    writer.vgprPool.checkIn(tmpVgpr)

  return result

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
##################################################
# Subroutine to generate MMA Instruction
# Given RegisterTileInfo inputs for A,B,C,D operands
# emit corresponding mfma instruction.
#
# This is a thin Python resolver: it extracts primitive register indices and
# boolean flags from the tile / kernel / writer objects and delegates all rocisa
# Module construction to the C++ ModuleBuilder (emit_mfma). The instType
# selection for miK==128 is also done via C++ (mfma_f8f6f4_inst_type).
#
def emitMfmaInstruction(writer, kernel, vgprTileA, vgprTileB, vgprTileC, vgprTileD,
                        scaleAVgpr=-1, scaleBVgpr=-1, scaleAsel=-1, scaleBsel=-1,
                        comment=""):
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

  miK = kernel["MatrixInstK"]

  # Select instType for the MX FP4/FP8 family (C++ mapping, no Python fallback).
  instTypeName = ""
  if miK == 128:
    pt = kernel.get("ProblemType")
    aType = pt.get("DataTypeA") if pt else None
    bType = pt.get("DataTypeB") if pt else None
    if aType is None or bType is None:
      raise RuntimeError(
          f"emitMfmaInstruction: unsupported data types for miK=128: "
          f"A={aType}, B={bType}")
    def _pred(t, name):
      fn = getattr(t, name, None)
      return bool(fn()) if callable(fn) else False
    instTypeName = _CPP_EMIT.mfma_f8f6f4_inst_type(
        _pred(aType, "isFloat8"), _pred(aType, "isBFloat8"), _pred(aType, "isFloat4"),
        _pred(bType, "isFloat8"), _pred(bType, "isBFloat8"), _pred(bType, "isFloat4"),
        bool(kernel.get("SourceSwap", False)))

  unitScaleVgpr = kernel.get("_subtileUnitScaleVgpr", -1)
  if miK == 128 and scaleAVgpr < 0:
    assert unitScaleVgpr >= 0, \
        "emitMfmaInstruction: plain FP8/FP4 fallback requires _subtileUnitScaleVgpr in kernel dict"

  return _mfma_builder().emit_mfma(
      vgprAStart, opASize, vgprBStart, opBSize,
      vgprCStart, opCSize, vgprDStart, opDSize,
      bool(dIsVgpr), bool(cIsVgpr), bool(kernel["MIArchVgpr"]),
      bool(kernel.get("SourceSwap", False)),
      miK, instTypeName,
      scaleAVgpr, scaleBVgpr, unitScaleVgpr,
      scaleAsel if scaleAsel >= 0 else 0,
      scaleBsel if scaleBsel >= 0 else 0,
      comment)


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
def mainLoop(writer, kernel, tensorParametersA, tensorParametersB):
  module = Module()
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
  grAGran = ReadGranularity(mn=grMNA, k=grKA) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2*grMNA, k=grKA)
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
          pgr=schedulerPgr
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

  module.add(scheduler.emitMainAndExitLoops(writer, kernel))

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
