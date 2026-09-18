# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile geometry validation for ``UseSubtileImpl`` solutions.

Lives under ``Tensile/SolutionStructs/Validators/`` alongside the other
Solution-level validators, so the subtile layout rules are not interleaved with
the rest of Solution.py.

These decide whether a wave group can lay out a subtile: how tall an LDS strip
may be, whether waves share one, and whether the global-read K partition hits a
known bug. They answer in rejection-reason strings -- ``None`` means the
geometry is fine -- because the same predicates drive the stack-height backoff
ladder and produce the message the caller rejects with.

Public entry points: :func:`subtileStackForTLU1`, :func:`subtileTLU1StackReason`
and :func:`validateSubtileGRKPartition`.
"""

from ..Utilities import reject


def _subtileGRKPartitionIsBuggy(loadRatioGR, localSubtileGrid):
  # TODO: TEMPORARY FIX. Encodes the trigger for the subtile global-read
  # cooperative-group + K-partition LDS bug (see fix_subtile_gr_missing_k_partition):
  # when one buffer_load covers multiple consecutive M-subtiles (loadRatioGR > 1)
  # and the per-wave M-subtile count (localSubtileGrid[0]) is not a multiple of
  # loadRatioGR, the last partial M-group writes a "ghost" slot that overlaps the
  # next K-partition's data, and the K>=1 representative subtile can be silently
  # dropped. This only manifests when there is more than one K-partition
  # (localSubtileGrid[1] > 1). Remove once the emit-side fix lands.
  return (loadRatioGR > 1
          and localSubtileGrid[1] > 1
          and localSubtileGrid[0] % int(loadRatioGR) != 0)


def _subtileWaveStraddlesStrip(stack, perWaveMTiles):
  # A wave must own whole strips or share one with other waves; a fraction of a
  # strip has no soffset register, and the GR emit indexes past the end of
  # localSubtilesRegister.  Only reachable on a non-power-of-two free dim.
  if stack <= 0:
    return False
  return perWaveMTiles % stack != 0 and stack % perWaveMTiles != 0


def _subtilePerWaveMTiles(mtTiles, stack, wgSize):
  # MFMA-M tiles per wave, measured against the strip: a strip is padded out to
  # a whole stack and the wave owns that padding too.  Shared so the straddle
  # check and the fetch-group count cannot drift apart.
  if stack <= 0 or wgSize <= 0:
    return max(1, int(mtTiles))
  padded = -(-int(mtTiles) // int(stack)) * int(stack)
  return max(1, padded // int(wgSize))


# A TLU=1 fp4 strip is stackM * MatrixInstM * 0.5 bytes wide, so a 16-tile stack
# fills one 128B cache line and a 2-tile stack uses only 16B of each line it
# touches.  Taller is therefore better, up to a full line.
_SUBTILE_STACK_SIZES = (16, 8, 4, 2)
_SUBTILE_STACK_MIN = 2
_SUBTILE_STACK_FULL_LINE = 16


def _subtileStackForTile(mtTiles):
  """Free-dim MFMA-M tiles per LDS strip for one TLU=1 fp4 operand.

  Tallest power-of-two stack that still holds the tile in one strip, else the
  tallest exact divisor.  Pad tiles cost LDS footprint but no traffic, since the
  pad lanes go to BufferOOB.  Rounding past the tile would need a partial
  trailing strip that the subtile grids do not count.
  """
  mtTiles = int(mtTiles)
  exact = next((s for s in _SUBTILE_STACK_SIZES if mtTiles % s == 0),
               _SUBTILE_STACK_MIN)
  if mtTiles <= 1:
    return exact
  roundedUp = min(_SUBTILE_STACK_FULL_LINE, 1 << (mtTiles - 1).bit_length())
  if roundedUp > exact and roundedUp >= mtTiles:
    return roundedUp
  return exact


# Strip sharing is only policed on gfx950; validateSubtileGRKPartition returns
# early elsewhere, and the stack chooser has to agree with it.
_SUBTILE_STRIP_SHARING_ISA = (9, 5, 0)


def _subtileStripSharingReason(state, tc, mtTiles, stack):
  """Why tensor tc's waves cannot share a strip of `stack`, or None when they can.

  These two rules hold for every subtile geometry, not just TLU=1 fp4, so they
  stay separate from the fp4-only layout rules below.
  """
  wgSize = state["MIWaveGroup"][0 if tc == 'A' else 1]
  perWaveMTiles = _subtilePerWaveMTiles(mtTiles, stack, wgSize)
  if _subtileWaveStraddlesStrip(stack, perWaveMTiles):
    return ("UseSubtileImpl=1 leaves a wave straddling an LDS strip on tensor %s: "
            "%d MMA tiles per wave against a strip of %d, so the wave neither owns "
            "whole strips nor shares one"
            % (tc, perWaveMTiles, stack))
  # A shared strip is addressed as a whole number of per-wave MFMA windows, so
  # the strip height has to divide by the wave's MIWaveTile.  perWaveMTiles
  # above is the padded share, which can hide the misalignment: a 24-tile dim
  # over 4 waves pads to 8 and looks clean against a strip of 16, while the
  # wave actually owns 6 and straddles the strip boundary.
  miWaveTile = int(state["MIWaveTile"][0 if tc == 'A' else 1])
  wavesPerStrip = max(1, stack // perWaveMTiles)
  if wavesPerStrip > 1 and miWaveTile and stack % miWaveTile != 0:
    return ("UseSubtileImpl=1 shares an LDS strip on tensor %s between waves whose "
            "MIWaveTile %d does not divide the strip of %d, so a wave's MFMA tiles "
            "cross the strip boundary"
            % (tc, miWaveTile, stack))
  return None


def subtileTLU1StackReason(state, tc, mtTiles, stack):
  """Why `stack` cannot lay out the TLU=1 fp4 operand tc, or None when it can."""
  mtFree = state["MacroTile0"] if tc == 'A' else state["MacroTile1"]
  strips = -(-mtTiles // stack)
  # A partial tail strip has no register list of its own, so the GR emit indexes
  # past the end of localSubtilesRegister.  Padding is only emittable while the
  # operand is a single strip.
  if mtTiles % stack != 0 and strips > 1:
    return ("UseSubtileImpl=1 TLU=1 fp4 pads tensor %s across more than one "
            "LDS strip: %d MMA tiles on a stack of %d is %d strips with a "
            "partial tail, which the GR emit cannot address (MacroTile=%d)"
            % (tc, mtTiles, stack, strips, mtFree))
  # The strip-sharing rules are enforced on gfx950 only, in
  # validateSubtileGRKPartition.  Apply the same gate here so the chooser never
  # rejects a height the validator would have let through on another ISA.
  if tuple(state["ISA"]) == _SUBTILE_STRIP_SHARING_ISA:
    sharing = _subtileStripSharingReason(state, tc, mtTiles, stack)
    if sharing:
      return sharing
  # A strip offers (blocks per strip) x (K windows) fetch slots.  With fewer
  # slots than waves in the group the surplus waves reissue a load someone else
  # made, so the operand comes off memory more than once.  Test the slots, not
  # the tile shape -- a wide wavefront or a shallow DepthU reaches the same
  # shortage.
  wgSize     = state["MIWaveGroup"][0 if tc == 'A' else 1]
  numWaves   = state["MIWaveGroup"][0] * state["MIWaveGroup"][1]
  otherWaves = max(1, numWaves // wgSize)
  perWave    = _subtilePerWaveMTiles(mtTiles, stack, wgSize)
  fetchGroup = max(1, stack // perWave) * otherWaves
  stripBytes = stack * state["MatrixInstM"] * state["MatrixInstK"] * 0.5
  slots      = int(stripBytes // (state["WavefrontSize"] * 16)) \
               * (state["DepthU"] // state["MatrixInstK"])
  if slots < fetchGroup:
    return ("UseSubtileImpl=1 TLU=1 fp4 leaves the LDS strip on tensor %s with "
            "%d (block x K window) slots for a fetch group of %d, so the surplus "
            "waves refetch it (MacroTile=%d, DepthU=%d, stack=%d)"
            % (tc, slots, fetchGroup, mtFree, state["DepthU"], stack))
  return None


def subtileStackForTLU1(state, tc, mtTiles):
  """Stack height for a TLU=1 fp4 operand, backing off when the geometry refuses it.

  _subtileStackForTile picks purely on cache-line utilization, and a height it
  likes can still be unlayoutable for this wave group.  Walk down the ladder
  from the preferred height rather than rejecting the solution outright.
  """
  preferred = _subtileStackForTile(mtTiles)
  for stack in [preferred] + [s for s in _SUBTILE_STACK_SIZES if s < preferred]:
    if subtileTLU1StackReason(state, tc, mtTiles, stack) is None:
      return stack
  return preferred


def validateSubtileGRKPartition(state, printRejectionReason):
  # TODO: TEMPORARY FIX. Reject gfx950 subtile solutions that hit the GR
  # K-partition bug (see _subtileGRKPartitionIsBuggy). Remove once
  # fix_subtile_gr_missing_k_partition is merged.
  if not state["UseSubtileImpl"]:
    return True
  if tuple(state["ISA"]) != (9, 5, 0):
    return True
  # Lazy import: Components/Subtile pulls the Components package and would
  # deadlock at module-load time if imported from Solution.py's top level.
  from Tensile.Components.Subtile.Kernel import selectABGeometry, TileInfo
  for tc in ("A", "B"):
    tileInfo = TileInfo(selectABGeometry(state, tc), tc, None, state)
    stack = int(tileInfo.subtileShape[0])
    mtTiles = int(tileInfo.macroTile // state["MatrixInstM"])
    sharingReason = _subtileStripSharingReason(state, tc, mtTiles, stack)
    if sharingReason:
      reject(state, printRejectionReason, sharingReason)
      return False
    loadRatioGR = tileInfo.loadRatioGR
    localSubtileGrid = tileInfo.localSubtileGrid
    if _subtileGRKPartitionIsBuggy(loadRatioGR, localSubtileGrid):
      reject(state, printRejectionReason,
             "UseSubtileImpl=1 hits the subtile GR K-partition bug on tensor %s: "
             "loadRatioGR=%s with localSubtileGrid=%s (M-subtile count %d is not a "
             "multiple of loadRatioGR and there is more than one K-partition)"
             % (tc, loadRatioGR, localSubtileGrid, localSubtileGrid[0]))
      return False
  return True
