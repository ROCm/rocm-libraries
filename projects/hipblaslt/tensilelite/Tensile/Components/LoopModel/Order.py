# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Calculate the LoopModel loop order and axes."""

from math import gcd

from ..TDMSplit import (
    AXIS_DU,
    AXIS_MT,
    factor_axis,
    split_factors,
    split_of,
)


_INNER = ("KMN", "KNM", "MNK", "MKN", "NMK", "NKM")
_OUTER = (None, "M", "N", "K")


def loopOrder(inner, outer) -> str:
  """Loop-order word named by WmmaInnerOrder/WmmaOuterOrder."""
  try:
    inner_index = int(inner) - 1
  except (TypeError, ValueError):
    raise ValueError("WmmaInnerOrder=%r must be 1..6" % inner)
  if not 0 <= inner_index < len(_INNER):
    raise ValueError("WmmaInnerOrder=%r must be 1..6" % inner)

  try:
    outer_index = int(outer)
  except (TypeError, ValueError):
    raise ValueError("WmmaOuterOrder=%r must be 0..3" % outer)
  if not 0 <= outer_index < len(_OUTER):
    raise ValueError("WmmaOuterOrder=%r must be 0..3" % outer)

  inner = _INNER[inner_index]
  outer = _OUTER[outer_index]
  return inner if outer is None \
      else outer + "".join(axis for axis in inner if axis != outer) + inner


def _splitAxis(kernel, side):
  _factor, axis = split_of(kernel, side)
  if axis == AXIS_MT:
    return "M" if side == "A" else "N"
  return "K" if axis == AXIS_DU else None


def _autoOuter(kernel):
  tiles = kernel["MIWaveTile"]
  k_tiles = max(
      1,
      int(kernel["DepthU"]) // int(kernel["MatrixInstruction"][2]),
  )
  weights = {}
  for side, tile in (("A", tiles[0]), ("B", tiles[1])):
    axis = _splitAxis(kernel, side)
    if axis is None:
      continue
    data_type = kernel["ProblemType"].get("DataType" + side) \
        or kernel["ProblemType"]["DataType"]
    extent = k_tiles if axis == "K" else max(1, int(tile))
    weights[axis] = max(weights.get(axis, 0), extent * data_type.numBytes())
  if not weights:
    return 0
  axes = ("M", "N", "K")
  outer = max(weights, key=lambda axis: (weights[axis], -axes.index(axis)))
  return _OUTER.index(outer)


def calculate(kernel):
  """Resolve WmmaOuterOrder and store the LoopOrder word once."""
  if "WmmaInnerOrder" not in kernel and "WmmaOuterOrder" not in kernel \
      and kernel.get("LoopOrder"):
    return

  outer = int(kernel.get("WmmaOuterOrder", -1))
  if outer == -1:
    outer = _autoOuter(kernel) if kernel.get("UseLoopModel") else 0
  elif not 0 <= outer < len(_OUTER):
    raise ValueError("WmmaOuterOrder=%r must be 0..3" % outer)

  if outer:
    axis = _OUTER[outer]
    split_axes = (_splitAxis(kernel, "A"), _splitAxis(kernel, "B"))
    if axis not in split_axes:
      raise RuntimeError(
          "WmmaOuterOrder=%d hoists the %s split outermost, but "
          "TDMSplitA=%s/TDMSplitB=%s do not split %s "
          "(A splits %s, B splits %s)"
          % (
              outer,
              axis,
              kernel.get("TDMSplitA"),
              kernel.get("TDMSplitB"),
              axis,
              split_axes[0],
              split_axes[1],
          )
      )

  kernel["WmmaOuterOrder"] = outer
  kernel["LoopOrder"] = loopOrder(kernel.get("WmmaInnerOrder", 1), outer)


def axes(kernel):
  """Return the ordered axes, regions, spans, and split factors."""
  word = str(kernel["LoopOrder"]).upper()
  if len(word) == 3 and sorted(word) == ["K", "M", "N"]:
    word = "".join(axis * 2 for axis in word)
  if len(word) != 6 or sorted(word) != ["K", "K", "M", "M", "N", "N"]:
    raise ValueError("LoopOrder=%r is not a K/M/N loop-order word" % word)

  splits = split_factors(kernel) or (1, 1, 1, 1)
  a_mt, b_mt, a_du, b_du = (max(1, int(value)) for value in splits)
  wave_tile_pair = kernel.get("MIWaveTile") or (1, 1)
  wave_tiles = (
      kernel.get("MIWaveTileA", wave_tile_pair[0]),
      kernel.get("MIWaveTileB", wave_tile_pair[1]),
  )
  m_inner = factor_axis(wave_tiles[0], a_mt, "MIWaveTileA")
  n_inner = factor_axis(wave_tiles[1], b_mt, "MIWaveTileB")
  k_tiles = max(1, int(kernel["LoopIters"]))
  k_inner = gcd(
      max(1, k_tiles // a_du),
      max(1, k_tiles // b_du),
  )
  extents = {
      "M_split": a_mt,
      "M_inner": m_inner,
      "N_split": b_mt,
      "N_inner": n_inner,
      "K_split": k_tiles // max(1, k_inner),
      "K_inner": k_inner,
  }

  seen = set()
  names = []
  for axis in word:
    name = axis + ("_split" if axis not in seen else "_inner")
    seen.add(axis)
    if name.endswith("_inner") or extents[name] > 1:
      names.append(name)

  regions_a = tuple(
      name for name in ("M_split", "K_split")
      if extents[name] > 1 and (name != "K_split" or a_du > 1)
  )
  regions_b = tuple(
      name for name in ("N_split", "K_split")
      if extents[name] > 1 and (name != "K_split" or b_du > 1)
  )
  k_split = max(1, extents["K_split"])
  span_a = {"K_split": max(1, k_split // a_du)} \
      if "K_split" in regions_a else {}
  span_b = {"K_split": max(1, k_split // b_du)} \
      if "K_split" in regions_b else {}
  return {
      "axes": tuple((name, extents[name]) for name in names),
      "regions": {"A": regions_a, "B": regions_b},
      "span": {"A": span_a, "B": span_b},
      "splits": (a_mt, b_mt, a_du, b_du),
  }
