# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""TDMSplitA/B validation.

Each rule reads state Solution resolves at a different point -- the wave tile, the TDMFuse
groups, DepthU, LDSSegmentInterleave -- so Solution calls each one where its inputs are final,
and every rule answers with the rejection reason or ``None``.  :func:`tdmSplitRejection` asks
them all at once for a caller that already holds a resolved state: the LoopModel parameter
search, which picks TDMSplitA/B.
"""

from ...Components import TDMSplit
from ...Components.TDMFuse import tdmFusedGroups


def waveTileSplitRejection(state):
  """A split factor the wave tile or its physical wave span cannot carry (UseLoopModel only)."""
  # TDMSplit factors either the operand's free axis or its shared-K axis.
  split = TDMSplit.split_factors(state)
  if split is None:
    return None
  aMT, bMT, _aDU, _bDU = split
  for tc, wt, n in (("A", state["MIWaveTile"][0], aMT), ("B", state["MIWaveTile"][1], bMT)):
    try:
      TDMSplit.factor_axis(wt, n, "MIWaveTile%s" % tc)
    except ValueError as exc:
      return "UseLoopModel requires %s" % exc

    # Wave placement is a separate physical fact.  It controls which region aliases a read
    # across waves, but never changes the split or inner axis extents.
    vw = state.get("VectorWidth%s" % tc) or 1
    try:
      span = max(1, min(n, int(TDMSplit.wave_region_span(state, tc)) or n))
    except Exception:
      span = n
    per = wt // span if span else wt
    if n > 1 and (wt % span or vw > per):
      return ("UseLoopModel: VectorWidth%s=%d does not fit the physical TDMSplit wave span "
              "(MIWaveTile=%d / wave-region-span=%d = %d tile(s))"
              % (tc, vw, wt, span, per))
  return None


def fusedRegionRejection(state):
  """A TDMFuse group whose members are split into different region counts."""
  if not tdmFusedGroups(state):
    return None
  regions = {"A": TDMSplit.split_of(state, "A")[0], "B": TDMSplit.split_of(state, "B")[0]}
  if state["ProblemType"]["MXBlockA"] and state["ProblemType"]["MXBlockB"]:
    regions["MXSA"] = 1        # scales are never split — `TDMSplitA/B` name the data only
    regions["MXSB"] = 1
  for group in tdmFusedGroups(state):
    counts = {m: regions[m] for m in group if m in regions}
    # An unsplit member rides region 0 and is nulled for the rest of the walk, so {1, R} is
    # supported; two different splits in one group still are not.
    if len({c for c in counts.values() if c > 1}) > 1:
      return ("TDMFuse=%d groups %s into ONE cooperative tensor_load, but its members have "
              "DIFFERENT REGION COUNTS (%s) under TDMSplitA=%s/TDMSplitB=%s.  A fused movement "
              "walks one region sequence, so two members split DIFFERENT ways cannot pair "
              "by index (an unsplit member is fine -- it rides region 0).  Give the split "
              "members the same number of regions -- the two AXES may differ, TDMSplitA=1 with "
              "TDMSplitB=2 is two regions each and is allowed."
              % (state["TDMFuse"], "+".join(group),
                 ", ".join("%s=%d" % kv for kv in sorted(counts.items())),
                 state.get("TDMSplitA"), state.get("TDMSplitB")))
  return None


def duSplitQuantumRejection(state):
  """A DU split whose per-region unroll slab is not a whole number of MatrixInstK."""
  split = TDMSplit.split_factors(state)
  if split is None or not (split[2] > 1 or split[3] > 1):
    return None
  mik = state["MatrixInstK"]
  for tc, n in (("A", split[2]), ("B", split[3])):
    if n > 1 and (mik <= 0 or (state["DepthU"] // n) % mik != 0):
      return ("TDMSplit%s=2 (DU split by %d) leaves %d unroll elements per storage region, "
              "which is not a multiple of MatrixInstK=%d.  A region is a separately packed LDS "
              "block, so a K slab straddling the boundary would need a per-lane region term "
              "that neither read path derives."
              % (tc, n, state["DepthU"] // n, mik))
  return None


def segmentInterleaveSplitRejection(state):
  """A DU split under LDSSegmentInterleave=1, which only partitions the free axis."""
  split = TDMSplit.split_factors(state)
  if state["LDSSegmentInterleave"] == 1 and split is not None \
     and (split[2] > 1 or split[3] > 1):
    return ("LDSSegmentInterleave=1 with a DU split (TDMSplitA=%d/TDMSplitB=%d) is not supported: "
            "the interleave lays components out along the FREE axis, so its port/component "
            "placement cannot express a reduction-axis partition and `tdmSplitLdsBoundary` would "
            "return the component jump where the region step is the packed half-tile."
            % (state.get("TDMSplitA", 0), state.get("TDMSplitB", 0)))
  return None


def tdmSplitRejection(state):
  """The first TDMSplit rule ``state`` breaks, in Solution's order, or ``None``.

  ``state`` must already carry every input the rules read, the latest being
  LDSSegmentInterleave.
  """
  for rule in (waveTileSplitRejection, fusedRegionRejection, duSplitQuantumRejection,
               segmentInterleaveSplitRejection):
    reason = rule(state)
    if reason is not None:
      return reason
  return None
