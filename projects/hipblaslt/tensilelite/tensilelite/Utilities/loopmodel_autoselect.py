# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Expand LoopModelAutoSelect into a config that benchmarks the search's top candidates.

    python -m tensilelite.Utilities.loopmodel_autoselect CONFIG --out OUT.yaml [--top-k 8]
        [--problem 0] [--size-group 0] [--arch gfx1250] [--no-tier2] [--tier3 16]
        [--include '{"WmmaInnerOrder": 2, ...}'] ...

Each fork of the selected problem group is derived with its free LoopModel parameters (-1, as
`LoopModelAutoSelect` reads them) at their defaults, searched, and its top-k candidates written
as one `Groups:` fork in OUT.yaml in place of the free parameters, so a Tensile run of OUT.yaml
ranks them on the GPU.  `--include` adds a group of its own -- a hand-tuned reference, say;
the free parameters it leaves out are filled with what Solution derives for them, so it is not
listed twice when the search ranks the same kernel.
With `--tier3 N` the first N candidates are emitted (CPU-only, a few seconds each) and re-ranked
by the steady loop StinkyTofu produced for them (`LoopModel.Auto.asmsim`).  The ranking, with
each tier's cycles, is printed.
"""

import argparse
import copy
import json
import sys

import yaml


def _toolchain(arch):
  from ..Common.Architectures import gfxToIsa
  from ..Common.Capabilities import makeIsaInfoMap
  from ..Toolchain.Assembly import makeAssemblyToolchain
  from ..Toolchain.Validators import ToolchainDefaults, validateToolchain

  isa = gfxToIsa(arch)
  if isa is None:
    raise SystemExit("unknown architecture %r" % arch)
  cxx = validateToolchain(ToolchainDefaults.CXX_COMPILER)
  bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
  assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
  return assembler, makeIsaInfoMap([isa], cxx)


def _forks(config, problem, sizeGroup):
  from ..BenchmarkStructs import BenchmarkProcess, constructForkPermutations

  entry = config["BenchmarkProblems"][problem]
  process = BenchmarkProcess(entry[0], entry[1 + sizeGroup], False)
  step = process[0]
  perms = list(constructForkPermutations(step.forkParams, step.paramGroups)) \
      if entry[1 + sizeGroup].get("ForkParameters") else [{}]
  return process.problemType, step.constantParams, perms


def tier3(pristine, result, build, assembler, count, costs):
  """Re-rank `result`'s first `count` candidates by their emitted loop (`asmsim`), best first.

  `build(state)` returns the Solution for a state, or None.  A candidate Solution refuses or
  the writer cannot emit is rejected at stage `emit`; candidates past `count` keep their tier-0
  place after the timed ones.
  """
  from ..Components.LoopModel.Auto.asmsim import loopCycles
  from ..Components.LoopModel.Auto.expand import emittedLoop

  head, tail = result.ranked[:count], result.ranked[count:]
  timed = []
  for item in head:
    state = copy.deepcopy(pristine)
    state.update(item.params)
    solution = build(state)
    body, reason = (None, "Solution refuses it") if solution is None \
        else emittedLoop(solution, assembler)
    if body is None:
      item.note = "tier3: %s" % reason
      result.refused.append(item)
      result.rejected["emit"] += 1
      result.examples.setdefault("emit", reason)
      continue
    mi = solution["MatrixInstruction"]
    wmmasPerTrip = solution["MIWaveTile"][0] * solution["MIWaveTile"][1] \
        * (solution["DepthU"] // mi[2])
    item.tier3 = loopCycles(body, costs, costs.wmma(mi[0], mi[1], mi[2],
                                                    bool(solution["ProblemType"].get("MX"))),
                            wmmasPerTrip)
    timed.append(item)
  result.ranked = sorted(timed, key=lambda i: i.tier3Key()) + tail
  result.winner = result.ranked[0] if result.ranked else None


def resolveInclude(pristine, free, extra, deriveProbe):
  """`extra` with each free parameter it leaves out set to what Solution derives for it.

  A parameter left out derives (`WmmaFactorK: -1` is the factor a TDM split pins, else 1),
  so the same kernel written with and without it must compare equal.  When Solution refuses
  `extra`, it is returned as given.
  """
  state = copy.deepcopy(pristine)
  state.update(extra)
  derived = deriveProbe(state)
  if derived is None:
    print("--include %s: Solution refuses it; kept as given" % json.dumps(extra), file=sys.stderr)
    return dict(extra)
  resolved = {name: int(derived[name]) for name in free if derived.get(name) is not None}
  resolved.update(extra)
  return resolved


def rank(configPath, topK=8, problem=0, sizeGroup=0, arch="gfx1250", tier2=True, tier3Count=0,
         include=()):
  """`(config, [(fork, free, AutoResult)], includes)`: the search of every fork of the problem
  group that leaves a parameter free, and the `include` groups resolved against those forks
  (`resolveInclude`)."""
  from .. import LibraryIO
  from ..BenchmarkProblems import _build_and_validate_solution
  from ..Common.GlobalParameters import assignGlobalParameters
  from ..Common.Types import makeDebugConfig
  from ..Components.LoopModel.Auto import space
  from ..Components.LoopModel.Auto.derive import rankSplits
  from ..Components.LoopModel.Auto.hw import costsFor

  config = LibraryIO.read(configPath)
  assembler, isaInfoMap = _toolchain(arch)
  assignGlobalParameters(config.get("GlobalParameters", {}), isaInfoMap)
  debugConfig = makeDebugConfig(config.get("GlobalParameters", {}))
  asmCaps = next(iter(isaInfoMap.values())).asmCaps
  problemType, constantParams, perms = _forks(config, problem, sizeGroup)

  def build(state):
    return _build_and_validate_solution(state, assembler, debugConfig, isaInfoMap, silent=True)

  def deriveProbe(state):
    built = build(state)
    return None if built is None else dict(built._state)

  out, includes = [], []
  for perm in perms:
    pristine = {"ProblemType": copy.deepcopy(problemType.state),
                "ISA": next(iter(isaInfoMap.keys()))}
    pristine.update(constantParams)
    pristine.update(perm)
    pristine["LoopModelAutoSelect"] = 0
    free = space.freeParams(pristine)
    if not free:
      continue
    for extra in include:
      resolved = resolveInclude(pristine, free, extra, deriveProbe)
      if resolved not in includes:
        includes.append(resolved)
    result = rankSplits(pristine, set(free), deriveProbe, asmCaps, tier2=tier2, topK=topK)
    costs = costsFor(pristine["ISA"])
    if tier3Count and costs is not None and result.ranked:
      tier3(pristine, result, build, assembler, tier3Count, costs)
    if not result.ranked:
      print("fork %s: nothing buildable -- %s" % (perm, result.reason()), file=sys.stderr)
    out.append((perm, free, result))
  return config, out, includes or [dict(extra) for extra in include]


def expand(config, ranked, problem=0, sizeGroup=0, topK=8, include=()):
  """`config` with the problem group's free parameters replaced by one `Groups:` fork."""
  config = copy.deepcopy(config)
  group = config["BenchmarkProblems"][problem][1 + sizeGroup]
  forks = group.setdefault("ForkParameters", [])
  names, choices = set(), []
  for _perm, free, result in ranked:
    for item in result.ranked[:topK]:
      chosen = {name: [value] for name, value in sorted(item.params.items()) if name in free}
      if chosen not in choices:
        choices.append(chosen)
      names |= set(chosen)
  for extra in include:
    chosen = {name: [value] for name, value in sorted(extra.items())}
    names |= set(chosen)
    if chosen not in choices:
      choices.insert(0, chosen)
  forks[:] = [entry for entry in forks
              if not (isinstance(entry, dict) and len(entry) == 1
                      and next(iter(entry)) in names | {"LoopModelAutoSelect"})]
  forks.append({"Groups": [choices]})
  return config


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("config")
  parser.add_argument("--out", required=True)
  parser.add_argument("--top-k", type=int, default=8)
  parser.add_argument("--problem", type=int, default=0)
  parser.add_argument("--size-group", type=int, default=0)
  parser.add_argument("--arch", default="gfx1250")
  parser.add_argument("--no-tier2", action="store_true")
  parser.add_argument("--tier3", type=int, default=16, metavar="N",
                      help="emit the first N tier-0 candidates and re-rank them by their "
                           "emitted loop (0: off)")
  parser.add_argument("--include", action="append", default=[], type=json.loads)
  args = parser.parse_args(argv)

  config, ranked, includes = rank(args.config, args.top_k, args.problem, args.size_group,
                                  args.arch, not args.no_tier2, args.tier3, args.include)
  for perm, _free, result in ranked:
    print("fork %s" % json.dumps(perm, default=str))
    print(result.summary(top=args.top_k))
  if not any(result.ranked for _perm, _free, result in ranked) and not args.include:
    raise SystemExit("no fork has a buildable candidate")
  with open(args.out, "w") as handle:
    yaml.safe_dump(expand(config, ranked, args.problem, args.size_group, args.top_k,
                          includes), handle, sort_keys=False, default_flow_style=None)
  print("wrote %s" % args.out)


if __name__ == "__main__":
  main()
