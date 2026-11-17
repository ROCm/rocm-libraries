# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Local-read prefetch validation against ``LoopIters``.

These rules judge the kernel-wide ``PrefetchLocalRead``/``ClusterLocalRead`` scalars, which under
UseLoopModel are the deepest and the least clustered of the two operands.  They answer in
``(reason, stop)`` pairs so a caller that only asks -- the LoopModel parameter search -- gets the
same verdict Solution rejects with.  ``stop`` says Solution returns at that rejection rather than
carrying on to the next rule.

Public entry point: :func:`localReadPrefetchRejections`.
"""


def _wideLocalReadRejections(state, tc, hasWmmaV3):
  # Multiple = WLR-size / input-size = how many iters could be covered by one WLR ?
  wlrMultiple = state["LocalReadVectorWidth%s" % tc] // state["MIInputPerThread%s" % tc]
  # NOTE: wlrmultiple can be 0 for new MFMA
  if not state["ProblemType"]["Sparse"] and not state["UseF32XEmulation"] \
      and not ((state["ProblemType"]["DataType"].is8bitFloat() or state["ProblemType"]["DataType%s" % tc].is8bitFloat())
               and (state["MatrixInstK"] in [64, 128,])) \
      and not hasWmmaV3:
    if wlrMultiple == 0:
      yield ("LocalReadVectorWidth%s %u is less than MIInput%s"
             % (tc, state["LocalReadVectorWidth%s" % tc], tc)), True
      return
  # for example, if the original ds_read is b32...
  #   1. if LoopIters = 5 (b32 x 5 times), WLR-Multiple = 2 (b64), then we can fit the WLR
  #   2. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 4 (b128), this is not allowed
  #   3. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 2 (b64), this is allowed
  if wlrMultiple and state["LoopIters"] % wlrMultiple != 0:
    yield ("LocalReadVectorWidth%s %u cannot be distributed evenly, LoopIters %u should be divisible by WLR-Multiple %u"
           % (tc, state["LocalReadVectorWidth%s" % tc], state["LoopIters"], wlrMultiple)), False

  if state["LoopIters"] - (state["PrefetchLocalRead"] * wlrMultiple) < 0:
    yield ("with PrefetchLocalRead %u LoopIters %u LocalReadVectorWidth%s %u, not enough LoopIters to prefetch %ux%u iterations, "
           % (state["PrefetchLocalRead"], state["LoopIters"], tc, state["LocalReadVectorWidth%s" % tc],
              state["PrefetchLocalRead"], wlrMultiple)), False


def localReadPrefetchRejections(state, hasWmmaV3):
  """Every rejection the local-read prefetch rules raise for ``state``, in Solution's order.

  Yields ``(reason, stop)``; a ``stop`` rejection is the last one Solution reaches.
  """
  if (state["EnableMatrixInstruction"] and not state["ClusterLocalRead"] and state["PrefetchLocalRead"] == 2
      and state["LoopIters"] >= 4 and state["ProblemType"]["MacDataTypeA"].numBytes() < 4
      and state["ProblemType"]["MacDataTypeB"].numBytes() < 4):
    yield "PrefetchLocalRead=2 with ClusterLocalRead=0 and numByte<4 does not support LoopIters>=4", True
    return

  # reject iterations are not enough to use wider local read
  if state["EnableMatrixInstruction"] and state["PrefetchLocalRead"] > 0:
    for tc in ("A", "B"):
      for reason, stop in _wideLocalReadRejections(state, tc, hasWmmaV3):
        yield reason, stop
        if stop:
          return
