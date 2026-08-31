################################################################################
#
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
################################################################################
"""LDS layout for the subtile path.

The subtile kernel packs LDS as a flat sequence of per-operand regions::

    | A strips | B strips | MX scale A | MX scale B |
    ^          ^          ^            ^
    0          offsetB    offsetMXSA   offsetMXSB

`computeLdsLayout` sizes those regions and returns their start offsets, which
the emitters read back off the writer as ``ldsStartOffset<tc>`` /
``ldsTotalSize``.  Keeping the arithmetic here rather than in `KernelWriter`
makes it testable in isolation and keeps it next to the geometry
(`Kernel.TileInfo`) and swizzle (`SubtileTLUSwizzle`) code it depends on.
"""

import math
from dataclasses import dataclass
from typing import Tuple

from .Kernel import TileInfo, selectABGeometry, selectDGeometry, selectMXScaleGeometry
from .SubtileTLUSwizzle import swizzlePadPerStrip


def selectGRSubtileShape(kernel: dict, tc: str) -> Tuple[int, int]:
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


def initSubtileInfo(writer, kernel: dict, tc: str):
  """Build the TileInfo for one operand and attach it to the writer's state."""
  tileMap = {
    'A'    : writer.states.a,
    'B'    : writer.states.b,
    'D'    : writer.states.d,
    'MXSA' : writer.states.mxsa,
    'MXSB' : writer.states.mxsb,
  }
  matrixInfo = tileMap[tc]
  if tc == 'D':
    geometry = selectDGeometry(kernel)
  elif tc in ('A', 'B'):
    matrixInfo.grSubtileShape = selectGRSubtileShape(kernel, tc)
    geometry = selectABGeometry(kernel, tc)
  elif tc in ('MXSA', 'MXSB'):
    geometry = selectMXScaleGeometry(kernel, tc)
  else:
    raise ValueError("unknown subtile operand %r" % tc)
  matrixInfo.tileInfo = TileInfo(geometry, tc, writer, kernel)


def initSubtileInfos(writer, kernel: dict):
  """Build TileInfos for every operand this kernel uses."""
  for tc in ('A', 'B', 'D'):
    initSubtileInfo(writer, kernel, tc)
  if kernel["ProblemType"].get("MXBlockA", 0) > 0:
    initSubtileInfo(writer, kernel, 'MXSA')
  if kernel["ProblemType"].get("MXBlockB", 0) > 0:
    initSubtileInfo(writer, kernel, 'MXSB')


def subtileRegionSize(tileInfo, macroTile: int, ldsRowBankSize: int) -> int:
  """Bytes reserved for one operand's subtile strips, including alignment.

  The payload is the operand's strips plus the padding the emitters insert into
  the LDS image: the TDM per-row pad, and the TLU=1 bank-conflict swizzle's
  per-strip pad (without which adjacent strips overlap).

  The alignment granularity is derived from *this* operand's subtileSize rather
  than always from A's.  With an asymmetric tile -- e.g. the NT fp4 16x1 stack,
  where A stacks 16 MFMA-M tiles per strip and B only 2 -- charging B the A
  granularity rounds B's 2KB of strips up to A's 16KB.
  """
  numSubtiles = int(tileInfo.globalSubtileGrid[0] * tileInfo.globalSubtileGrid[1])
  rowPad = int(tileInfo.ldsRowPadBytes) * macroTile
  swzPad = swizzlePadPerStrip(tileInfo) * numSubtiles
  raw = int(numSubtiles * tileInfo.subtileSize + rowPad + swzPad)

  # The region is measured in subtiles but filled by DTL writes covering
  # loadRatioGR of them, so a subtile count that is not a multiple of that
  # leaves a trailing partial group writing into the next operand.  Round to
  # the write, not to a constant: a fixed 2 over-pads the common
  # loadRatioGR <= 1 case and under-pads the wide wave groups, where the ratio
  # reaches 4 or 8.  Floor at an LDS bank row so the bank-conflict swizzle maps
  # the same way in B's region as in A's.
  # A ratio <= 1 means one write covers at most one subtile, so the subtile
  # boundary is already a write boundary and only the bank-row floor applies.
  ratio = tileInfo.loadRatioGR
  writeUnit = int(math.ceil(ratio) * tileInfo.subtileSize) if ratio > 1 else 0
  align = max(writeUnit, int(ldsRowBankSize))
  return int(((raw + align - 1) // align) * align)


@dataclass
class SubtileLdsLayout:
  """Start offsets and total size of the subtile LDS regions, in bytes."""
  offsetA: int    = 0
  offsetB: int    = 0
  offsetMXSA: int = -1
  offsetMXSB: int = -1
  totalSize: int  = 0


def computeLdsLayout(writer, kernel: dict) -> SubtileLdsLayout:
  """Lay out A, B and (when present) the MX scale regions in LDS."""
  archCaps = writer.states.archCaps
  ldsRowBankSize = archCaps["LDSBankCount"] * archCaps["LDSBankWidth"]

  sizeA = subtileRegionSize(writer.states.a.tileInfo, kernel["MacroTile0"], ldsRowBankSize)
  sizeB = subtileRegionSize(writer.states.b.tileInfo, kernel["MacroTile1"], ldsRowBankSize)

  layout = SubtileLdsLayout(offsetA=0, offsetB=sizeA)
  sizeMXSA = sizeMXSB = 0
  if kernel["ProblemType"].get("MXBlockA", 0) > 0 and kernel["ProblemType"].get("MXBlockB", 0) > 0:
    # For swizzled scale we use extra LDS space for now to allow wider DTL loads.
    numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
    sizeMXSA = writer.states.mxsa.tileInfo.loadWidthGR * kernel["WavefrontSize"] * numWaves
    sizeMXSB = writer.states.mxsb.tileInfo.loadWidthGR * kernel["WavefrontSize"] * numWaves
    layout.offsetMXSA = sizeA + sizeB
    layout.offsetMXSB = sizeA + sizeB + sizeMXSA

  layout.totalSize = sizeA + sizeB + sizeMXSA + sizeMXSB
  return layout


def applyLdsLayout(writer, kernel: dict):
  """Build the subtile TileInfos, lay out LDS, and publish the result.

  Sets ``kernel["LdsNumBytes"]`` and flags an LDS overflow on the writer when
  the layout does not fit the device.
  """
  initSubtileInfos(writer, kernel)

  layout = computeLdsLayout(writer, kernel)
  writer.ldsStartOffsetA    = layout.offsetA
  writer.ldsStartOffsetB    = layout.offsetB
  if layout.offsetMXSA >= 0:
    writer.ldsStartOffsetMXSA = layout.offsetMXSA
    writer.ldsStartOffsetMXSB = layout.offsetMXSB
  writer.ldsTotalSize       = layout.totalSize

  kernel["LdsNumBytes"] = max(1, int(layout.totalSize * kernel["NumLdsBlk"]))
  if kernel["LdsNumBytes"] > writer.states.archCaps["DeviceLDS"]:
    writer.states.overflowedResources = 8
  return layout
