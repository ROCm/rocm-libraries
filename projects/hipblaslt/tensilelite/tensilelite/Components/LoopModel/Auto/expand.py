# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`LoopModelAutoSelect: N` with N > 1: a fork becomes the search's N best candidates.

Each candidate is written as stated parameters with `LoopModelAutoSelect` cleared, so the
benchmark builds and times it like any other fork and the GPU picks among them.  N = 1 stays
Derive mode (`derive.autoSelect`), which picks the winner inside Solution derivation.
"""

import copy

from . import space
from .derive import rankSplits

KEY = "LoopModelAutoSelect"


def count(params):
  return int(params.get(KEY, 0) or 0)


def wanted(constantParams, forkPermutations):
  """Whether any fork of `forkPermutations` asks for more than one candidate."""
  if count(constantParams) > 1:
    return True
  forkParams = getattr(forkPermutations, "forkParams", None)
  if forkParams is None:
    return any(count(perm) > 1 for perm in forkPermutations)
  values = list(forkParams.get(KEY, []))
  for groups in getattr(forkPermutations, "paramGroups", []):
    values.extend(group.get(KEY, 0) for group in groups)
  return any(int(value or 0) > 1 for value in values)


def expandFork(pristine, n, build, asmCaps, emits=None):
  """`(choices, result)`: the free parameters of `pristine`'s `n` best candidates that build
  (and, with `emits`, emit), best first, and the search's `AutoResult` (None when nothing is
  free).

  `build(state)` returns the Solution for a state, or None when Solution refuses it;
  `emits(solution)` says whether the kernel writer emits it.  The search asks both of each
  candidate as it reaches it, in ranking order, and stops once `n` pass; one either refuses is
  counted in `result.rejected` and the next one in the ranking takes its place.  Near the VGPR
  budget most of a tied band can fail to emit.
  """
  pristine = copy.deepcopy(pristine)
  pristine[KEY] = 0
  free = space.freeParams(pristine)
  if not free:
    return [{}], None

  def deriveProbe(state):
    built = build(state)
    return None if built is None else dict(built._state)

  def accept(params):
    state = copy.deepcopy(pristine)
    state.update(params)
    solution = build(state)
    if solution is None:
      return "build"
    if emits is not None and not emits(solution):
      return "emit"
    return None

  result = rankSplits(pristine, set(free), deriveProbe, asmCaps, topK=n, accept=accept)
  choices = []
  for item in result.ranked:
    if len(choices) >= n:
      break
    chosen = {name: value for name, value in item.params.items() if name in free}
    if chosen not in choices:
      choices.append(chosen)
  return choices, result


def emittedLoop(solution, assembler):
  """`(body, reason)`: the steady loop StinkyTofu emits for `solution`'s kernel, or None and
  why it emits none."""
  import shutil
  import rocisa
  from ....Common.Types import DebugConfig
  from .asmsim import loopBody
  from ....KernelWriterAssembly import KernelWriterAssembly
  from ....SolutionStructs.Naming import getKernelFileBase
  from ....create_library.run import processKernelSource

  kernel = solution.getKernels()[0]
  isa = tuple(kernel["ISA"])
  instance = rocisa.rocIsa.getInstance()
  instance.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
  instance.setKernel(isa, kernel["WavefrontSize"])
  kernel["BaseName"] = getKernelFileBase(False, kernel)
  writer = KernelWriterAssembly(assembler, DebugConfig())
  result = processKernelSource(writer, instance.getData(), instance.getOutputOptions(), False,
                               kernel)
  source = result.src
  if isinstance(source, (bytes, bytearray)):
    source = source.decode(errors="replace")
  if result.err or not source:
    return None, "the kernel writer failed (err %s); its warning above says why" % result.err
  body = loopBody(source)
  return (body, None) if body is not None else (None, "the emitted kernel has no steady loop")
