# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Calculate the LoopModel loop order and axes."""

from ..TDMSplit import (
    factor_axis,
    split_factors,
)


_INNER = ("KMN", "KNM", "MNK", "MKN", "NMK", "NKM")
_OUTER = (None, "M", "N", "K")

#: The parameter that states each axis role's extra outer factor.
FACTOR_PARAM = {"M": "WmmaFactorM", "N": "WmmaFactorN", "K": "WmmaFactorK"}


def forcedFactors(kernel) -> dict:
  """`{role: extent}` that TDMSplit PINS, 1 where it leaves the factor free.

  A storage region IS one value of the extra axis -- the region index is the axis value --
  so a split operand cannot carry a factor the split does not also make.  Everything else is
  a loop axis with no storage behind it, which is what WmmaFactor adds on its own."""
  a_mt, b_mt, a_du, b_du = split_factors(kernel) or (1, 1, 1, 1)
  return {"M": max(1, int(a_mt)), "N": max(1, int(b_mt)),
          "K": max(1, int(a_du), int(b_du))}


def factors(kernel) -> dict:
  """`{role: extent}` for the three WmmaFactor axes -- the ONE reading of the parameters.

  `-1` derives: the TDMSplit-pinned extent where that operand splits, else 1."""
  forced = forcedFactors(kernel)
  resolved = {}
  for role, name in FACTOR_PARAM.items():
    value = kernel.get(name)
    value = -1 if value is None else int(value)
    if value == -1:
      value = forced[role]
    elif value < 1:
      raise ValueError("%s=%d must be -1 (derive) or a positive extent" % (name, value))
    elif forced[role] > 1 and value != forced[role]:
      raise ValueError(
          "%s=%d contradicts the TDMSplit cutting the same axis into %d: a storage region is "
          "one value of that axis, so the two extents cannot differ"
          % (name, value, forced[role]))
    resolved[role] = value
  return resolved


def loopOrder(inner, outer, extra) -> str:
  """Lower WIO/WOO plus the three WmmaFactor extents to one LoopOrder word."""
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
  if outer is None:
    return inner
  split_axes = {role for role, extent in dict(extra).items() if int(extent) > 1}
  remaining = "".join(axis for axis in inner if axis != outer)
  return (
      outer + outer + remaining + remaining
      if outer not in split_axes else
      outer + remaining + inner
  )


def _autoOuter(kernel):
  tiles = kernel["MIWaveTile"]
  k_tiles = max(
      1,
      int(kernel["DepthU"]) // int(kernel["MatrixInstruction"][2]),
  )
  extra = factors(kernel)
  weights = {}
  for side, tile in (("A", tiles[0]), ("B", tiles[1])):
    # ONE AXIS PER SIDE: its own free axis when that is factored, else the shared K.  An
    # operand already paying for its free factor does not also price the reduction.
    free = "M" if side == "A" else "N"
    axis, extent = (free, max(1, int(tile))) if extra[free] > 1 else ("K", k_tiles)
    if extra[axis] <= 1:
      continue
    data_type = kernel["ProblemType"].get("DataType" + side) \
        or kernel["ProblemType"]["DataType"]
    weights[axis] = max(weights.get(axis, 0), extent * data_type.numBytes())
  if not weights:
    return 0
  # THE FACTOR LEADS, the footprint breaks its ties: promoting an axis saves `1 - 1/factor` of
  # what it holds, so the two terms price one saving.  TDMSplit only ever makes a factor of 2,
  # so on a split kernel the lead term always ties and the footprint still decides.
  axes = ("M", "N", "K")
  outer = max(weights, key=lambda axis: (extra[axis], weights[axis], -axes.index(axis)))
  return _OUTER.index(outer)


def calculate(kernel):
  """Resolve WmmaFactorM/N/K and WmmaOuterOrder, and store the LoopOrder word once."""
  if "WmmaInnerOrder" not in kernel and "WmmaOuterOrder" not in kernel \
      and kernel.get("LoopOrder"):
    return

  extra = factors(kernel)
  for role, name in FACTOR_PARAM.items():
    kernel[name] = extra[role]

  outer = int(kernel.get("WmmaOuterOrder", -1))
  if outer == -1:
    outer = _autoOuter(kernel) if kernel.get("UseLoopModel") else 0
  elif not 0 <= outer < len(_OUTER):
    raise ValueError("WmmaOuterOrder=%r must be 0..3" % outer)

  kernel["WmmaOuterOrder"] = outer
  kernel["LoopOrder"] = loopOrder(kernel.get("WmmaInnerOrder", 1), outer, extra)


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
  # THE AXIS EXTENTS COME FROM WmmaFactor, the storage regions from TDMSplit.  They are the
  # same number wherever an operand splits, because a region is one value of the axis; where
  # it does not, the factor still buys the extra loop level and the copy stays whole.
  extra = factors(kernel)
  k_tiles = max(1, int(kernel["LoopIters"]))
  extents = {
      "M_split": extra["M"],
      "M_inner": factor_axis(wave_tiles[0], extra["M"], "MIWaveTileA", "WmmaFactorM"),
      "N_split": extra["N"],
      "N_inner": factor_axis(wave_tiles[1], extra["N"], "MIWaveTileB", "WmmaFactorN"),
      "K_split": extra["K"],
      "K_inner": factor_axis(k_tiles, extra["K"], "LoopIters", "WmmaFactorK"),
  }

  seen = set()
  names = []
  for axis in word:
    name = axis + ("_split" if axis not in seen else "_inner")
    seen.add(axis)
    if name.endswith("_inner") or extents[name] > 1:
      names.append(name)

  # A REGION AXIS IS ONE THIS OPERAND SPLITS STORAGE ON.  An axis the factor alone makes is
  # walked like any other; its copy writes one region and owes no region term.
  a_cuts = (("M_split", a_mt), ("K_split", a_du))
  b_cuts = (("N_split", b_mt), ("K_split", b_du))
  regions_a = tuple(name for name, split in a_cuts if split > 1 and extents[name] > 1)
  regions_b = tuple(name for name, split in b_cuts if split > 1 and extents[name] > 1)
  # The span is how many axis values ONE region covers: 1 when the factor is the split.
  span_a = {name: max(1, extents[name] // split)
            for name, split in a_cuts if name in regions_a}
  span_b = {name: max(1, extents[name] // split)
            for name, split in b_cuts if name in regions_b}
  # THE FACTOR AXES ARE WHAT ORDER SEES: the prefetch unit, the rotation and the grouping are
  # all "per one value of the extra axis", and that is true of a factor with no storage behind
  # it too.  A split operand's two sets coincide, which is why registers never had to tell them
  # apart before.
  factors_a = tuple(name for name, _split in a_cuts if extents[name] > 1)
  factors_b = tuple(name for name, _split in b_cuts if extents[name] > 1)
  return {
      "axes": tuple((name, extents[name]) for name in names),
      "regions": {"A": regions_a, "B": regions_b},
      "factor_axes": {"A": factors_a, "B": factors_b},
      "span": {"A": span_a, "B": span_b},
      "splits": (a_mt, b_mt, a_du, b_du),
  }
