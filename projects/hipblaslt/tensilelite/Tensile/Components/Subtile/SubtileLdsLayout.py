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
from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...Common import roundUpToNearestMultiple
from .Kernel import TileInfo, selectABGeometry, selectDGeometry, selectMXScaleGeometry
from .SubtileTLUSwizzle import swizzlePadPerStrip

# A GR load round covers one MMA tile in M and two in K; when four or more waves
# cooperate on that round it widens to two tiles in M.  LR is always the base
# shape regardless of wave config.
_GR_SUBTILE_SHAPE_BASE = (1, 2)
_GR_SUBTILE_SHAPE_WIDE = (2, 2)
_GR_WAVES_TO_WIDEN = 4

# states.overflowedResources code meaning "the LDS layout exceeds the device".
_LDS_OVERFLOW_RESOURCE_CODE = 8

_AB_OPERANDS = ('A', 'B')
_MX_SCALE_OPERANDS = ('MXSA', 'MXSB')


def mxBlockOf(kernel: dict, tc: str) -> int:
  """MXBlock size for operand 'A'/'B' (0 when that operand carries no scales)."""
  return kernel["ProblemType"].get("MXBlock%s" % tc, 0)


def selectGRSubtileShape(kernel: dict, tc: str) -> Tuple[int, int]:
  """GR subtile shape for one operand, from how many waves share its load.

  The waves cooperating on an operand are the ones spanning the *other* free
  dimension: a 1x4 wave group has 4 waves sharing A's load and 1 sharing B's.
  """
  waveGroup = kernel["MIWaveGroup"]
  numWaves = waveGroup[0] * waveGroup[1]
  wavesInFreeDim = waveGroup[0 if tc == 'A' else 1]
  wavesCooperating = numWaves // wavesInFreeDim
  return (_GR_SUBTILE_SHAPE_WIDE if wavesCooperating >= _GR_WAVES_TO_WIDEN
          else _GR_SUBTILE_SHAPE_BASE)


def matrixInfoOf(writer, tc: str):
  """The writer's per-operand state slot ('A' -> states.a, 'MXSA' -> states.mxsa)."""
  slots = {
    'A'    : writer.states.a,
    'B'    : writer.states.b,
    'D'    : writer.states.d,
    'MXSA' : writer.states.mxsa,
    'MXSB' : writer.states.mxsb,
  }
  if tc not in slots:
    raise ValueError("unknown subtile operand %r" % tc)
  return slots[tc]


def tileInfoOf(writer, tc: str):
  """The TileInfo built for one operand by `initSubtileInfo`."""
  return matrixInfoOf(writer, tc).tileInfo


def initSubtileInfo(writer, kernel: dict, tc: str):
  """Build the TileInfo for one operand and attach it to the writer's state."""
  matrixInfo = matrixInfoOf(writer, tc)
  if tc == 'D':
    geometry = selectDGeometry(kernel)
  elif tc in _AB_OPERANDS:
    matrixInfo.grSubtileShape = selectGRSubtileShape(kernel, tc)
    geometry = selectABGeometry(kernel, tc)
  else:
    geometry = selectMXScaleGeometry(kernel, tc)
  matrixInfo.tileInfo = TileInfo(geometry, tc, writer, kernel)


def initSubtileInfos(writer, kernel: dict):
  """Build TileInfos for every operand this kernel uses."""
  for tc in ('A', 'B', 'D'):
    initSubtileInfo(writer, kernel, tc)
  for tc in _AB_OPERANDS:
    if mxBlockOf(kernel, tc) > 0:
      initSubtileInfo(writer, kernel, 'MXS%s' % tc)


def subtileRegionSize(tileInfo, macroTile: int, ldsRowBankSize: int) -> int:
  """Bytes reserved for one operand's subtile strips, including alignment.

  Sized from *this* operand's subtileSize: with an asymmetric tile -- the NT fp4
  16x1 stack, where A stacks 16 MFMA-M tiles per strip and B only 2 -- charging
  B the A granularity would round B's 2KB of strips up to A's 16KB.
  """
  numStrips = int(tileInfo.globalSubtileGrid[0] * tileInfo.globalSubtileGrid[1])
  payload = (int(numStrips * tileInfo.subtileSize)
             + int(tileInfo.ldsRowPadBytes) * macroTile
             + swizzlePadPerStrip(tileInfo) * numStrips)

  # A region is measured in subtiles but filled by DTL write groups, so a
  # trailing partial group would spill into the next operand unless the region
  # is a whole number of groups.  At most one subtile per write, nothing to
  # round.  Floor at an LDS bank row either way, so the bank-conflict swizzle
  # maps the same way in B's region as it does in A's.
  subtilesPerWrite = tileInfo.loadRatioGR
  writeGroupBytes = (int(math.ceil(subtilesPerWrite) * tileInfo.subtileSize)
                     if subtilesPerWrite > 1 else 0)
  return roundUpToNearestMultiple(payload, max(writeGroupBytes, int(ldsRowBankSize)))


def _mxScaleRegionSize(tileInfo, kernel: dict) -> int:
  """Bytes reserved for one MX scale operand.

  Swizzled scale takes more LDS than the scales strictly need, so that the DTL
  loads feeding it can stay wide.
  """
  numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
  return tileInfo.loadWidthGR * kernel["WavefrontSize"] * numWaves


@dataclass
class SubtileLdsLayout:
  """Start offset of each operand's LDS region, and the total they occupy."""
  offsets: Dict[str, int] = field(default_factory=dict)
  totalSize: int = 0

  def offsetOf(self, tc: str, default: int = -1) -> int:
    return self.offsets.get(tc, default)


def computeLdsLayout(writer, kernel: dict) -> SubtileLdsLayout:
  """Lay out A, B and (when present) the MX scale regions in LDS."""
  archCaps = writer.states.archCaps
  ldsRowBankSize = archCaps["LDSBankCount"] * archCaps["LDSBankWidth"]
  macroTiles = {'A': kernel["MacroTile0"], 'B': kernel["MacroTile1"]}

  sizes = [(tc, subtileRegionSize(tileInfoOf(writer, tc), macroTiles[tc], ldsRowBankSize))
           for tc in _AB_OPERANDS]
  # Scale regions are sized as a pair, so scales on one side only get no region
  # here -- initSubtileInfos builds a TileInfo per scaled operand regardless.
  if all(mxBlockOf(kernel, tc) > 0 for tc in _AB_OPERANDS):
    sizes += [(tc, _mxScaleRegionSize(tileInfoOf(writer, tc), kernel))
              for tc in _MX_SCALE_OPERANDS]

  # Each region starts at the next free byte, in order.
  layout = SubtileLdsLayout()
  for tc, size in sizes:
    layout.offsets[tc] = layout.totalSize
    layout.totalSize += size
  return layout


def applyLdsLayout(writer, kernel: dict) -> SubtileLdsLayout:
  """Build the subtile TileInfos, lay out LDS, and publish the result.

  Sets ``kernel["LdsNumBytes"]`` and flags an LDS overflow on the writer when
  the layout does not fit the device.
  """
  initSubtileInfos(writer, kernel)

  layout = computeLdsLayout(writer, kernel)
  for tc, offset in layout.offsets.items():
    setattr(writer, "ldsStartOffset%s" % tc, offset)
  writer.ldsTotalSize = layout.totalSize

  kernel["LdsNumBytes"] = max(1, int(layout.totalSize * kernel["NumLdsBlk"]))
  if kernel["LdsNumBytes"] > writer.states.archCaps["DeviceLDS"]:
    writer.states.overflowedResources = _LDS_OVERFLOW_RESOURCE_CODE
  return layout
