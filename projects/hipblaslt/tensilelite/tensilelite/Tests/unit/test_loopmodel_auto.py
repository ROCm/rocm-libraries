# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LoopModelAutoSelect on the hand-tuned MX-F8 kernel (`loopmodel_mxf8_autoselect.yaml`).

The tuned kernel's numbers are measured, not modelled: its emitted kernel uses 1023 VGPRs after
allocation (the count is 1024 with the fixed registers), a ring one buffer shorter than
`VgprAllocA=3` overwrites a tile a WMMA still reads, and GIR refuses exactly those rings.
"""

import copy
import dataclasses
import os

import pytest

from tensilelite.Components.LoopModel.Auto import asmsim, evaluate, hw, perf, search, space

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_CONFIG = os.path.join(
    os.path.dirname(__file__), "characterization", "_codegen", "data", "test_data",
    "_designed", "gfx1250", "loopmodel_mxf8_autoselect.yaml")
_TUNED_ORDER = space.OrderChoice(inner=2, outer=3, factorM=2, factorN=2, factorK=1)
_TUNED = space.RegisterChoice(space.SideChoice(2, 0, 1, 3), space.SideChoice(1, 0, 1, 2))
_SEARCHED_FREE = set(space.ORDER_PARAMS) | set(space.SIDE_REGISTER_PARAMS)


def _derive(path):
  from config_harness import derive_states
  return derive_states(path, arch=_ARCH, limit_solutions=None)


def _asm_caps():
  from toolchain_test_helpers import toolchain_for
  return next(iter(toolchain_for(_ARCH)[1].values())).asmCaps


@pytest.fixture(scope="module")
def tuned():
  states = _derive(_CONFIG)
  assert len(states) == 1
  return dict(states[0])


def _walk(state):
  kernel, reason = evaluate.applyOrder(state, _TUNED_ORDER)
  assert kernel is not None, reason
  return kernel, perf.NestWalk(kernel, space.registerShapes(kernel))


def test_the_nest_walk_finds_where_each_unit_is_first_and_last_used(tuned):
  kernel, walk = _walk(tuned)
  assert kernel["LoopOrder"] == "KKNMNM"
  assert walk.wmmas == 160
  # A's unit (Mi) comes back for the second No; B's (Ni) is used once per Mo pair.
  assert walk.uses("A", 1) == [(0, 59), (20, 79), (80, 139), (100, 159)]
  assert walk.uses("B", 1) == [(0, 39), (40, 79), (80, 119), (120, 159)]


def test_a_ring_too_short_for_the_walk_is_refused_as_gir_refuses_it(tuned):
  _kernel, walk = _walk(tuned)
  assert walk.distances("A", 1, 2, 3) == [80, 40, 80, 40]
  # PLR 1 of two buffers: each refill waits for its slot's last reader, 20 WMMAs ahead of use.
  assert walk.distances("A", 1, 1, 2) == [20, 20, 20, 20]
  # One buffer, or 1-tile sub-units in 5: a slot is refilled before its tile's second use.
  assert walk.distances("A", 1, 0, 1) is None
  assert walk.distances("A", 4, 4, 5) is None
  assert walk.distances("B", 1, 1, 2) == [40, 40, 40, 40]


def test_the_tuned_kernel_takes_exactly_the_vgpr_budget(tuned):
  ev = evaluate.tier1(tuned, _TUNED_ORDER, _TUNED, hw.GFX1250, _asm_caps())
  assert ev.feasible, ev.reason
  assert ev.gpr.operands == {"A": 192, "B": 160, "MXSA": 12, "MXSB": 10}
  assert ev.gpr.accumulator == 640
  assert ev.gpr.vgpr == 1024
  # One more A buffer is over.
  wider = space.RegisterChoice(space.SideChoice(2, 0, 1, 4), _TUNED.b)
  over = evaluate.tier1(tuned, _TUNED_ORDER, wider, hw.GFX1250, _asm_caps())
  assert over.stage == "budget" and over.gpr.vgpr > 1024
  # WmmaFactorN=1 holds every B tile: 1193 VGPRs after allocation on the real kernel.
  single = dataclasses.replace(_TUNED_ORDER, factorN=1)
  over = evaluate.tier1(tuned, single, _TUNED, hw.GFX1250, _asm_caps())
  assert over.stage == "budget" and over.gpr.vgpr == 1184


def test_a_search_with_nothing_free_returns_the_stated_kernel(tuned):
  result = search.search(tuned, set(), _asm_caps(), tier2=False)
  assert result.winner is not None
  stated = dict(_TUNED_ORDER.params(), **_TUNED.params())
  assert {name: result.winner.params[name] for name in stated} == stated


def test_the_search_is_deterministic(tuned):
  first = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps(), tier2=False)
  second = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps(), tier2=False)
  assert first.ranked
  assert [r.params for r in first.ranked] == [r.params for r in second.ranked]
  assert all(r.vgpr <= hw.GFX1250.maxVgpr for r in first.ranked)


def test_the_free_search_picks_the_tuned_kernel(tuned):
  result = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps(), tier2=False)
  stated = dict(_TUNED_ORDER.params(), **_TUNED.params())
  assert {name: result.winner.params[name] for name in stated} == stated


def test_a_search_for_the_winner_alone_picks_the_same_winner(tuned):
  wide = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps())
  alone = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps(), topK=1)
  assert alone.winner.params == wide.winner.params
  assert sum(1 for r in alone.ranked if r.lowered) < sum(1 for r in wide.ranked if r.lowered)


def test_tier3_charges_a_load_where_the_schedule_waits_for_it():
  costs = hw.GFX1250
  body = ["ds_load_b128 v[0:3], v100", "s_wait_dscnt 0x0",
          "v_wmma_f32_16x16x128 v[8:15], v[0:15], v[16:31], v[8:15]",
          "s_cbranch_scc0 label_LoopBeginL"]
  waited = asmsim.loopCycles(body, costs, 8, 1)
  # Waiting for nothing outstanding leaves only issue and the WMMA.
  free = asmsim.loopCycles(body[:1] + ["s_wait_dscnt 0x1"] + body[2:], costs, 8, 1)
  assert waited >= costs.asmDsLatency
  assert free < waited and free <= 8 + len(body)
  source = "x:\nlabel_LoopBeginL:\n  " + "\n  ".join(body) + "  // back edge\nlabel_End:\n"
  assert asmsim.loopBody(source) == body


def test_tier3_times_the_tuned_kernel_ahead_of_a_shorter_a_ring(tmp_path):
  from config_harness import emit_kernels_from_config
  from tensilelite import LibraryIO
  import yaml

  def cycles(path):
    (_base, source, err), = emit_kernels_from_config(path, limit=None, arch=_ARCH,
                                                     canonical=False)
    assert not err
    return asmsim.loopCycles(asmsim.loopBody(source), hw.GFX1250, 8, 160)

  config = LibraryIO.read(_CONFIG)
  for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
    for name, value in (("PrefetchLocalReadA", 1), ("VgprAllocA", 2)):
      if name in entry:
        entry[name] = [value]
  shorter = tmp_path / "shorter.yaml"
  shorter.write_text(yaml.safe_dump(config, default_flow_style=None))
  # Measured on one gfx1250 card, 4096x5120x65536: 9.33e6 vs 8.67e6 GFLOPS.
  assert cycles(_CONFIG) < cycles(str(shorter))


def test_every_lowered_candidate_passes_gir(tuned):
  result = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps(), topK=4)
  assert result.winner is not None and result.winner.lowered
  assert all(r.lowered and r.tier2 is not None for r in result.ranked)
  # The ring check leaves GIR little to refuse.
  assert result.rejected["gir"] <= 2, result.refused


def test_auto_select_states_its_choice_in_the_solution(tmp_path):
  from tensilelite import LibraryIO
  import yaml

  config = LibraryIO.read(_CONFIG)
  forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
  for entry in forks:
    for name in list(entry):
      if name in _SEARCHED_FREE:
        entry[name] = [-1]
  forks.append({"LoopModelAutoSelect": [1]})
  path = tmp_path / "auto.yaml"
  path.write_text(yaml.safe_dump(config, default_flow_style=None))

  states = _derive(str(path))
  assert len(states) == 1
  state = states[0]
  assert state["LoopModelAutoSelect"] == 0
  for name in _SEARCHED_FREE:
    assert int(state[name]) != -1, name
  # Stated values: the no-load loop is dropped, as for a config that names them.
  assert state["SuppressNoLoadLoop"] is True
