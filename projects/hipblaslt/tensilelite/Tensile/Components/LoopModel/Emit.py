# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit the LoopModel prologue, steady body, and drain stages."""

from rocisa.code import Module

from ...Lowering import steady_chain
from ...Lowering.gir_to_rocisa import GirToRocisa

from .Program import loopModelGirProgram


def girEmitStage(writer, kernel, tPA, tPB, phase, *, internalPointerSwap=False, kinds=None):
  """Emit one GIR stage from the cached program."""
  prog = loopModelGirProgram(writer, kernel)
  return GirToRocisa(writer, kernel, tPA, tPB, prog).emit_block(
      prog, phase, internalPointerSwap=internalPointerSwap, kinds=kinds)


def loopModelSteadyBlocks(writer, kernel) -> list:
  """GIR's steady chain for this kernel, in program order."""
  return steady_chain(loopModelGirProgram(writer, kernel))


def loopModelSteadyPhase(writer, kernel, loopCopy) -> str:
  """The chain block this unrolled copy emits.

  ONE block serves every scaffold copy: the copies then differ only in scaffold-owned state,
  which is what ExpandPointerSwap's pair has always done.  A longer chain is GIR's own unroll --
  the copies hold different buffers -- so there the copy index selects the block.
  """
  chain = loopModelSteadyBlocks(writer, kernel)
  if len(chain) <= 1:
    return chain[0] if chain else "steady"
  if not 0 <= int(loopCopy) < len(chain):
    raise RuntimeError(
        "UseLoopModel: unrolled loop copy %s has no GIR steady block -- the chain is %r.  The "
        "scaffold copy count must match LoopIR's derived steady ring period."
        % (loopCopy, chain))
  return chain[int(loopCopy)]


#: The prologue's GLOBAL->SHARED movement, which legacy issues from `setupNewTile`, and the
#: REGISTER fill, which legacy issues at the prefetch-local point.  One GIR block, two sites.
PROLOGUE_MOVEMENT = ("copy", "gr_inc", "desc_enable", "region_inc", "gl2_prefetch")
PROLOGUE_FILL = ("read", "wmma", "swap", "fence", "gsu_guard")


def _prologueGuardLadder(writer, kernel, tensorParametersA, tensorParametersB, module):
  """The single-trip skipPGR ladder; it brackets the COPIES, so it follows them to their site."""
  guard = loopModelGirProgram(writer, kernel).meta.get("prefetch_guard")
  if not guard:
    return
  module.add(writer.openPrefetchGlobalRead2orMore(kernel, guard["gen"]))
  module.add(girEmitStage(
      writer, kernel, tensorParametersA, tensorParametersB, guard["block"]))
  for idxPgr in range(0, kernel["PrefetchGlobalRead"] + 1):
    module.add(writer.closePrefetchGlobalRead2orMore(
        kernel, tensorParametersA, tensorParametersB, idxPgr))
    if idxPgr == 1 and guard.get("skip"):
      module.add(girEmitStage(
          writer, kernel, tensorParametersA, tensorParametersB, guard["skip"]))
  module.add(girEmitStage(
      writer, kernel, tensorParametersA, tensorParametersB, guard["join"]))


def loopModelPrologueCopies(writer, kernel, tensorParametersA, tensorParametersB, module):
  """HalfPLR's global prefetch, emitted where the legacy scaffold emits its own.

  HalfPLR is diffed instruction-for-instruction against its legacy twin, and legacy issues this
  from `setupNewTile` -- ~230 lines before the prefetch-local point the rest of the prologue comes
  from, which is a different basic block.  Every other UseLoopModel shape keeps the single site,
  so nothing but HalfPLR moves.
  """
  if kernel["UseLoopModel"] and kernel["HalfPLR"]:
    module.add(girEmitStage(writer, kernel, tensorParametersA, tensorParametersB, "prologue",
                            kinds=PROLOGUE_MOVEMENT))
    _prologueGuardLadder(writer, kernel, tensorParametersA, tensorParametersB, module)


def loopModelPrologue(writer, kernel, tensorParametersA, tensorParametersB, module, pack, packPre):
  """The prologue fork: the PLR fill, plus the copies unless HalfPLR already took them early."""
  # Keep empty pack modules for the steady loop; the drain no longer consumes
  # localReadDo's offset state.
  for plrIdx in range(0, writer.states.numItersPLR):
    packPre[plrIdx] = Module()
    pack[plrIdx] = Module()
  early = bool(kernel["HalfPLR"])
  module.add(girEmitStage(writer, kernel, tensorParametersA, tensorParametersB, "prologue",
                          kinds=PROLOGUE_FILL if early else None))
  if not early:
    _prologueGuardLadder(writer, kernel, tensorParametersA, tensorParametersB, module)
  writer.states.SubTileIdx = (writer.states.SubTileIdx + 1) % kernel["numSubTiles"]


def loopModelSteadyIter(writer, kernel, tensorParametersA, tensorParametersB, module,
                        LoopModelScaffold, u, waitLWCode, syncCode, loopCopy=0):
  """One steady substep.

  The walker unrolls the whole inner loop, so the order-independent scaffold accumulates across
  u and is emitted once, after the unroll, at the last substep.  `loopCopy` selects this copy's
  block in GIR's steady chain: with one copy that is the only block, and nothing changes.
  """
  LoopModelScaffold.add(waitLWCode)
  LoopModelScaffold.add(syncCode)
  _perLW = writer.codes.perIterLocalWrite[u][1] if u < len(writer.codes.perIterLocalWrite) else None
  if _perLW is not None and _perLW.count():
    LoopModelScaffold.add(_perLW)
  if u == kernel["LoopIters"] - 1:
    # Copies, swaps, and GR increments are GIR leaves, just like reads and
    # WMMA operations.
    module.add(girEmitStage(
        writer, kernel, tensorParametersA, tensorParametersB,
        loopModelSteadyPhase(writer, kernel, loopCopy),
        internalPointerSwap=kernel["ExpandPointerSwap"]))
    module.add(LoopModelScaffold)


def loopModelDrainIter(writer, kernel, tensorParametersA, tensorParametersB, module,
                       LoopModelDrainScaffold, u, waitLWCode, syncCode, remainPgr):
  """Emit one drain substep."""
  _lmDrainStep = (kernel["PrefetchGlobalRead"] - 1) - remainPgr
  LoopModelDrainScaffold.add(waitLWCode)
  LoopModelDrainScaffold.add(syncCode)
  # The drain block already carries the copies issued by the peel.
  _perLW = writer.codes.perIterLocalWrite[u][1] \
      if (writer.codes.perIterLocalWrite and u < len(writer.codes.perIterLocalWrite)) else None
  if _perLW is not None and _perLW.count():
    LoopModelDrainScaffold.add(_perLW)
  if u == kernel["LoopIters"] - 1:
    # GIR places the swap and global-read increment for the drain.
    module.add(girEmitStage(
        writer, kernel, tensorParametersA, tensorParametersB, "drain%d" % _lmDrainStep,
        internalPointerSwap=False))
    module.add(LoopModelDrainScaffold)
