# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SubtileGeometry — abstract tile geometry definitions for the subtile-based kernel.

Contains layout classes, abstract geometry base classes, and pre-defined instances.
No emit logic lives here — concrete shape classes with emit implementations are in
Kernel.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Optional, Tuple

from rocisa.container import vgpr, sgpr, accvgpr
from rocisa.enum import RegisterType

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
  def _cpp_twin(self):
    # Concrete subclasses (ABGRGeometry, ABLRGeometry) build the matching
    # tensile_writer C++ object that services every ported query method. The
    # abstract base is never instantiated directly; surface the contract
    # explicitly so a missing override fails loudly rather than with an opaque
    # AttributeError.
    raise NotImplementedError(
        f"{type(self).__name__} must override the _cpp_twin property")

  # --- MMA tile grid queries (no subtile shape dependency) ---

  def globalMMATileGrid(self, macroTile: int, depthU: int) -> Tuple[int, int]:
    return tuple(self._cpp_twin.globalMMATileGrid(macroTile, depthU))

  def localMMATileGrid(self, macroTile: int, depthU: int, waveGroupSize: int) -> Tuple[int, int]:
    return tuple(self._cpp_twin.localMMATileGrid(macroTile, depthU, waveGroupSize))


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
  def _cpp_twin(self):
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
    return tuple(self._cpp_twin.localGRGranularity(numWaves))

  def globalSubtileGrid(self, macroTile: int, depthU: int) -> Tuple[float, float]:
    return tuple(self._cpp_twin.globalSubtileGrid(macroTile, depthU))

  def subtileSizeBytes(self) -> float:
    """Bytes in one contiguous strip."""
    return self._cpp_twin.subtileSizeBytes()

  def bytesPerLoad(self, numWaves: int) -> int:
    """Total bytes loaded cooperatively per load round (all waves, all lanes)."""
    return self._cpp_twin.bytesPerLoad(numWaves)

  def loadsPerStrip(self, numWaves: int) -> float:
    return self._cpp_twin.loadsPerStrip(numWaves)

  def for_kernel(self, kernel: dict, tc: str) -> 'ABGRGeometry':
    """Return a new frozen instance with subtileCount/subtileStride from kernel config.

    The pre-defined instances (AB_B16.gr etc.) are dtype-only templates; this
    method materializes them for a specific wave-group and macro tile size.
    tc: 'A' or 'B' — selects the correct wave-group axis and macro tile key.

    subtileShape is expanded when cooperating waves (numWaves // wg_m) can load
    more MMA tiles than the base subtileShape covers.  This eliminates loadRatio > 1
    cases: the effective per-load coverage IS the subtileShape.
    """
    cpp_fk = self._cpp_twin.for_kernel(kernel, tc)
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

    sid, bshape, tiles = self._cpp_twin.subtileForMmaTile(r, c)
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
  def _cpp_twin(self):
    return _cppgeo.ABLRGeometry(_cpp_mma(self.mmaLayout), self.instK,
                                float(self.bpe), _cpp_loadshape(self.loadShape),
                                tuple(self.subtileShape), self.tlu, self.loadWidth)

  def globalSubtileGrid(self, macroTile: int, depthU: int) -> Tuple[float, float]:
    return tuple(self._cpp_twin.globalSubtileGrid(macroTile, depthU))

  def subtileSizeBytes(self) -> float:
    return self._cpp_twin.subtileSizeBytes()

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
  def _cpp_twin(self):
    return _cppgeo.CDTileGeometry(_cpp_mma(self.mmaLayout), float(self.bpe),
                                  _cpp_loadshape(self.storeShape))

  def globalMMATileGrid(self, macroTile0: int, macroTile1: int) -> Tuple[int, int]:
    return tuple(self._cpp_twin.globalMMATileGrid(macroTile0, macroTile1))

  def localMMATileGrid(self, macroTile0: int, macroTile1: int,
                       waveGroup: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(self._cpp_twin.localMMATileGrid(macroTile0, macroTile1,
                                                 tuple(waveGroup)))

  def globalSubtileGrid(self, macroTile0: int, macroTile1: int,
                        subtileShape: Tuple[float, float]) -> Tuple[float, float]:
    """Subtile grid over the full macro tile."""
    return tuple(self._cpp_twin.globalSubtileGrid(macroTile0, macroTile1,
                                                  tuple(subtileShape)))

  def localSubtileGrid(self, macroTile0: int, macroTile1: int,
                       waveGroup: Tuple[int, int],
                       subtileShape: Tuple[float, float]) -> Tuple[float, float]:
    """Subtile grid per wave (each wave stores its own chunk)."""
    return tuple(self._cpp_twin.localSubtileGrid(macroTile0, macroTile1,
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
  def _cpp_twin(self):
    # Concrete subclasses (MXScaleGRGeometry, MXScaleLRGeometry) build the
    # matching tensile_writer C++ object that services every ported query
    # method. The abstract base is never instantiated directly; surface the
    # contract explicitly so a missing override fails loudly rather than with an
    # opaque AttributeError.
    raise NotImplementedError(
        f"{type(self).__name__} must override the _cpp_twin property")

  def globalMMATileGrid(self, macroTile: int, depthU: int) -> Tuple[int, int]:
    # depthU is in data elements; divide by instK (not instKScale) to get scale MMA K tiles.
    return tuple(self._cpp_twin.globalMMATileGrid(macroTile, depthU))


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
  def _cpp_twin(self):
    shape = tuple(self.subtileShape) if self.subtileShape is not None else None
    return _cppgeo.MXScaleGRGeometry(_cpp_scale(self.scaleLayout), self.instK,
                                     float(self.bpe), self.loadWidth, shape)

  def for_kernel(self, kernel: dict, tc: str) -> 'MXScaleGRGeometry':
    if self.subtileShape is not None:
      return self
    cpp_fk = self._cpp_twin.for_kernel(kernel, tc)
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
  def _cpp_twin(self):
    return _cppgeo.MXScaleLRGeometry(_cpp_scale(self.scaleLayout), self.instK,
                                     float(self.bpe), self.loadWidth,
                                     tuple(self.subtileShape))

  def globalSubtileGrid(self, macroTile: int, depthU: int) -> Tuple[float, float]:
    return tuple(self._cpp_twin.globalSubtileGrid(macroTile, depthU))

  def subtileSizeBytes(self) -> float:
    return self._cpp_twin.subtileSizeBytes()

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
