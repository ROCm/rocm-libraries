# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`LoopModelAutoSelect=1`: derive, search the free LoopModel parameters, derive the winner.

The search needs a derived Solution -- the resolved tile, LRVW, MIInputPerThread and the rest
-- so it runs between two derivations: a quiet one with every free parameter at its usual
default, which is the search's starting state, and the real one, with the winner's values
stated as if the config had named them.  The final Solution therefore carries no trace of the
search: `LoopModelAutoSelect` is cleared, and a config naming the same values builds the same
kernel.

A free TDMSplitA/B changes what Solution derives (the LDS layout, the factor it pins, two
SGPRs), so each split pair gets its own starting derivation and its own search, and the pairs'
rankings are merged.  A pair Solution refuses is skipped.
"""

import copy
import itertools

from ....Common import print1
from . import search, space
from .report import AutoResult, Ranked

#: Ranked candidates re-derived, best first, when Solution refuses the winner.
MAX_FINALISTS = 4
#: The splits a free TDMSplitA/B is searched over.
SPLIT_VALUES = (0, 1, 2)


def _valid(state):
  return state.get("Valid", True) is not False


def _splitPairs(pristine, free):
  domains = [SPLIT_VALUES if name in free else (int(pristine.get(name, 0) or 0),)
             for name in space.SPLIT_PARAMS]
  return [dict(zip(space.SPLIT_PARAMS, pair)) for pair in itertools.product(*domains)]


def rankSplits(pristine, free, deriveProbe, asmCaps, tier2=True, topK=search.TIER2_COUNT):
  """Search every split pair `free` allows, from its own starting derivation; one ranking.

  `deriveProbe(state)` derives a copy of `pristine` with the pair stated and returns the
  derived state, or None when Solution refuses it.
  """
  searched = set(free) - set(space.SPLIT_PARAMS)
  merged = AutoResult(None)
  for split in _splitPairs(pristine, free):
    probe = copy.deepcopy(pristine)
    probe.update(split)
    probe = deriveProbe(probe)
    if probe is None:
      merged.rejected["split"] += 1
      merged.examples.setdefault("split", "Solution refuses %s" % split)
      continue
    result = search.search(probe, searched, asmCaps, tier2=tier2, topK=topK)
    for item in result.ranked + result.refused:
      item.params.update(split)
    merged.ranked.extend(result.ranked)
    merged.refused.extend(result.refused)
    merged.rejected.update(result.rejected)
    for stage, reason in result.examples.items():
      merged.examples.setdefault(stage, reason)
    merged.searched += result.searched
    merged.seconds += result.seconds
  merged.ranked.sort(key=Ranked.key)
  merged.winner = merged.ranked[0] if merged.ranked else None
  return merged


def autoSelect(state, derive, asmCaps, printRejectionReason, tier2=True):
  """Resolve `state`'s free LoopModel parameters in place.

  `derive(state, printRejectionReason)` is Solution's own derivation.  Returns None when
  `state` ends derived (valid or not, with Solution's reason printed), or the reason to reject
  it when the search finds nothing buildable.
  """
  pristine = copy.deepcopy(state)
  pristine["LoopModelAutoSelect"] = 0
  if not pristine.get("UseLoopModel", False):
    return "LoopModelAutoSelect requires UseLoopModel"
  free = set(space.freeParams(pristine))

  def restart(extra=None):
    state.clear()
    state.update(copy.deepcopy(pristine))
    state.update(extra or {})

  if not free:
    restart()
    derive(state, printRejectionReason)
    return None

  def deriveProbe(probe):
    derive(probe, False)
    return probe if _valid(probe) else None

  # The winner alone first: each lowering costs about as much as the rest of the search, and
  # Solution seldom refuses a candidate tier 1 passed.  The wider ranking is the fallback.
  merged = rankSplits(pristine, free, deriveProbe, asmCaps, tier2=tier2, topK=1)
  tried = []
  if merged.ranked and _deriveFirstValid(merged.ranked[:MAX_FINALISTS], tried, free, state,
                                         restart, derive, False):
    print1(merged.summary())
    return None
  merged = rankSplits(pristine, free, deriveProbe, asmCaps, tier2=tier2)
  if not merged.ranked:
    if merged.rejected.get("split") == len(_splitPairs(pristine, free)):
      # Every starting derivation is refused: Solution's own reason is the answer.
      restart()
      derive(state, printRejectionReason)
      return None
    return merged.reason()

  print1(merged.summary())
  finalists = [item for item in merged.ranked if item.params not in tried][:MAX_FINALISTS]
  if not finalists:
    finalists = merged.ranked[:1]
  _deriveFirstValid(finalists, tried, free, state, restart, derive, printRejectionReason)
  return None


def _deriveFirstValid(finalists, tried, free, state, restart, derive, printRejectionReason):
  """Derive `finalists` in turn, stated, until one is valid; True when one is."""
  for index, item in enumerate(finalists):
    tried.append(item.params)
    restart({name: value for name, value in item.params.items() if name in free})
    derive(state, printRejectionReason and index == len(finalists) - 1)
    if _valid(state):
      return True
  return False
