################################################################################
#
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
################################################################################
"""LDS layout for the subtile path.

LDS is packed as a flat sequence of per-operand regions::

    | A strips | B strips | MX scale A | MX scale B |
    0          offsetB    offsetMXSA   offsetMXSB

`computeLdsLayout` sizes them and returns the start offsets, which emitters read
back off the writer as ``ldsStartOffset<tc>`` / ``ldsTotalSize``.
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

  Sized from *this* operand's subtileSize: on an asymmetric tile, charging B the
  A granularity would round B's strips up to A's much larger stack.
  """
  numStrips = int(tileInfo.globalSubtileGrid[0] * tileInfo.globalSubtileGrid[1])
  payload = (int(numStrips * tileInfo.subtileSize)
             + int(tileInfo.ldsRowPadBytes) * macroTile
             + swizzlePadPerStrip(tileInfo) * numStrips)

  # Regions are measured in subtiles but filled by DTL write groups, so round to
  # a whole group or a trailing partial spills into the next operand; at most one
  # subtile per write there is nothing to round.  Floor at an LDS bank row either
  # way, so the swizzle maps the same way in every region.
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
  dtlSize = tileInfo.loadWidthGR * kernel["WavefrontSize"] * numWaves
  # The region must also hold the whole scale tile, (DepthU / MXBlock) * MT * bpe
  # bytes (4 KiB per tile at MT 256 and DepthU 512).
  side = tileInfo.tc[-1]
  macroTile = kernel["MacroTile0"] if side == 'A' else kernel["MacroTile1"]
  tileSize = int((kernel["DepthU"] // mxBlockOf(kernel, side)) * macroTile * tileInfo.bpe)
  return max(dtlSize, tileSize)


@dataclass
class SubtileLdsLayout:
  """Start offset of each operand's LDS region, and the total they occupy."""
  offsets: Dict[str, int] = field(default_factory=dict)
  totalSize: int = 0


# Writer attribute each region publishes.  Spelled out rather than built from a
# format string so the names the emit paths read are greppable to their source.
_LDS_START_ATTR = {
    'A':    "ldsStartOffsetA",
    'B':    "ldsStartOffsetB",
    'MXSA': "ldsStartOffsetMXSA",
    'MXSB': "ldsStartOffsetMXSB",
}


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

  segMode = kernel.get("LDSSegmentInterleave")
  if segMode == 1:
    return _segmentInterleaveCompLayout(writer, kernel, dict(sizes))
  if segMode == 2:
    return _segmentInterleaveSideLayout(dict(sizes))

  # Each region starts at the next free byte, in order.
  layout = SubtileLdsLayout()
  for tc, size in sizes:
    layout.offsets[tc] = layout.totalSize
    layout.totalSize += size
  return layout


def _ldsSegmentBytes() -> int:
  from ...SolutionStructs.segment_interleave import SEG
  return SEG


def _roundUpToSegment(size: int) -> int:
  seg = _ldsSegmentBytes()
  return -(-size // seg) * seg


def _segmentInterleaveCompLayout(writer, kernel: dict, sizes: Dict[str, int]) -> SubtileLdsLayout:
  """LDSSegmentInterleave=1: [A0][B0] and [A1][B1] each start on a 64 KiB segment.

  Wave bit 0 picks both the LDS read port and the A half, so the two ports never
  read A from the same segment.  Publishes the per-buffer stride as
  ``writer.ldsSegCompStride``.
  """
  from .SubtileLREmit import subtileTdmLdsSpanBytes
  compA = subtileTdmLdsSpanBytes(kernel, tileInfoOf(writer, 'A'), "A", 2)
  compB = subtileTdmLdsSpanBytes(kernel, tileInfoOf(writer, 'B'), "B", 2)
  writer.ldsSegCompStride = _roundUpToSegment(compA + compB)

  layout = SubtileLdsLayout()
  layout.offsets['A'] = 0
  layout.offsets['B'] = compA
  layout.totalSize = writer.ldsSegCompStride + compA + compB
  for tc in _MX_SCALE_OPERANDS:
    if tc in sizes:
      layout.offsets[tc] = layout.totalSize
      layout.totalSize += sizes[tc]
  # Keep the second LDS buffer's components segment-aligned too.
  layout.totalSize = _roundUpToSegment(layout.totalSize)
  return layout


def _segmentInterleaveSideLayout(sizes: Dict[str, int]) -> SubtileLdsLayout:
  """LDSSegmentInterleave=2: [A][MXSA] fill one 64 KiB segment and [B][MXSB] the next.

  A wave reading A-side data and a wave reading B-side data never share a
  segment; the main loop has the two LDS read ports walk them in opposite order.
  """
  seg = _ldsSegmentBytes()
  sizeA, sizeB = sizes['A'], sizes['B']
  sizeMXSA, sizeMXSB = sizes.get('MXSA', 0), sizes.get('MXSB', 0)
  assert sizeA + sizeMXSA <= seg and sizeB + sizeMXSB <= seg, \
    "LDSSegmentInterleave=2 needs A+MXSA and B+MXSB to each fit one LDS segment"
  layout = SubtileLdsLayout()
  layout.offsets['A'] = 0
  layout.offsets['B'] = seg
  if 'MXSA' in sizes:
    layout.offsets['MXSA'] = sizeA
    layout.offsets['MXSB'] = seg + sizeB
  layout.totalSize = 2 * seg
  return layout


def _applyTdmSingleIssuerLayout(writer, kernel: dict) -> None:
  """TDMSingleIssuer: gemm_wt A0/A1/B0/B1 in distinct 64 KB segments.

  A/B xor-swap by buffer, so ``ldsTotalSize`` is the double-buffer xor stride
  rather than the footprint; the scales use a ring after the data groups.
  """
  from .LdsV7Layout import make_lds_v7_layout
  lay = make_lds_v7_layout()
  writer.ldsV7 = lay
  writer.ldsStartOffsetA = lay.a0
  writer.ldsStartOffsetB = lay.b0
  writer.ldsStartOffsetMXSA = lay.sc_mxsa
  writer.ldsStartOffsetMXSB = lay.sc_mxsb
  writer.ldsTotalSize = lay.buf
  kernel["LdsNumBytes"] = lay.end


def applyLdsLayout(writer, kernel: dict) -> SubtileLdsLayout:
  """Build the subtile TileInfos, lay out LDS, and publish the result.

  Sets ``kernel["LdsNumBytes"]`` and flags an LDS overflow on the writer when
  the layout does not fit the device.
  """
  initSubtileInfos(writer, kernel)
  writer.ldsV7 = None
  writer.ldsSegCompStride = 0

  if kernel.get("TDMSingleIssuer"):
    _applyTdmSingleIssuerLayout(writer, kernel)
    if kernel["LdsNumBytes"] > writer.states.archCaps["DeviceLDS"]:
      writer.states.overflowedResources = _LDS_OVERFLOW_RESOURCE_CODE
    return None

  layout = computeLdsLayout(writer, kernel)
  for tc, offset in layout.offsets.items():
    setattr(writer, _LDS_START_ATTR[tc], offset)
  writer.ldsTotalSize = layout.totalSize

  kernel["LdsNumBytes"] = max(1, int(layout.totalSize * kernel["NumLdsBlk"]))
  if kernel["LdsNumBytes"] > writer.states.archCaps["DeviceLDS"]:
    writer.states.overflowedResources = _LDS_OVERFLOW_RESOURCE_CODE
  return layout
