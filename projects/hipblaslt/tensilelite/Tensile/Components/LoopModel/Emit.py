# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit the LoopModel prologue, steady body, and drain stages."""

from rocisa.code import Module

from ...Lowering.gir_to_rocisa import GirToRocisa

from .Program import loopModelGirProgram


def girEmitStage(writer, kernel, tPA, tPB, phase, *, internalPointerSwap=False):
  """Emit one GIR stage from the cached program."""
  prog = loopModelGirProgram(writer, kernel)
  return GirToRocisa(writer, kernel, tPA, tPB, prog).emit_block(
      prog, phase, internalPointerSwap=internalPointerSwap)


def loopModelPrologue(writer, kernel, tensorParametersA, tensorParametersB, module, pack, packPre):
  """Emit the prologue fork, including the PLR fill and global prefetch."""
  # Keep empty pack modules for the steady loop; the drain no longer consumes
  # localReadDo's offset state.
  for plrIdx in range(0, writer.states.numItersPLR):
    packPre[plrIdx] = Module()
    pack[plrIdx] = Module()
  module.add(girEmitStage(
      writer, kernel, tensorParametersA, tensorParametersB, "prologue"))
  # A single-iteration loop has its own GIR block, so the skipPGR ladder
  # brackets the copies without duplicating advances.
  guard = loopModelGirProgram(writer, kernel).meta.get("prefetch_guard")
  if guard:
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
  writer.states.SubTileIdx = (writer.states.SubTileIdx + 1) % kernel["numSubTiles"]


def loopModelSteadyIter(writer, kernel, tensorParametersA, tensorParametersB, module,
                        LoopModelScaffold, u, waitLWCode, syncCode):
  """One steady substep.

  The walker unrolls the whole inner loop, so the order-independent scaffold accumulates across
  u and is emitted once, after the unroll, at the last substep.
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
        writer, kernel, tensorParametersA, tensorParametersB, "steady",
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
