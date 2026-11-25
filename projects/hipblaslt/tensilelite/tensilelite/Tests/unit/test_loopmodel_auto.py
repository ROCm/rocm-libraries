# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LoopModelAutoSelect on the hand-tuned MX-F8 kernel (`loopmodel_mxf8_autoselect.yaml`).

The tuned kernel's numbers are measured, not modelled: its emitted kernel uses 1023 VGPRs after
allocation (the search's lower bound is 1020), a ring one buffer shorter than
`VgprAllocA=3` overwrites a tile a WMMA still reads, and GIR refuses exactly those rings.
"""

import copy
import dataclasses
import os

import pytest

from tensilelite.Components.LoopModel.Auto import asmsim, evaluate, hw, perf, resources, search, space

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


def _walk(state, order=_TUNED_ORDER):
  kernel, reason = evaluate.applyOrder(state, order)
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


def test_a_burst_of_late_refills_waits_about_one_latency(tuned):
  # KNNMM sweeps A's four Mo units every N step, so a ring of four has no slot to read the
  # next K step ahead into: all four refills land 6 WMMAs before use, issued 2 WMMAs apart.
  kernel, walk = _walk(tuned, space.OrderChoice(inner=2, outer=0, factorM=4, factorN=2,
                                                factorK=1))
  assert kernel["LoopOrder"] == "KNM"
  assert walk.distances("A", 1, 3, 4) == [6] * 8
  latency, perWmma = 300.0, 8.0
  # The first refill of each K step waits; the next three drain through that wait.
  assert walk.exposure("A", 1, 3, 4, latency, perWmma) == 2 * (latency - 6 * perWmma)
  # Refills far enough ahead expose nothing, the same as their distances say.
  _kernel, tunedWalk = _walk(tuned)
  assert tunedWalk.exposure("A", 1, 2, 3, latency, perWmma) == 0.0


def test_a_rotating_ring_is_offered_only_longer_than_its_read_ahead(tuned):
  kernel, walk = _walk(tuned)
  for side in "AB":
    choices = space.sideChoices(walk.shapes[side], side, _SEARCHED_FREE, kernel)
    assert choices
    assert all(c.va > c.plr for c in choices if c.clr == 0), side
  # B at PLR 2 / VA 2 reads a tile its slot no longer holds (GIR: WRONG SOURCE).
  assert space.SideChoice(2, 0, 1, 2) not in space.sideChoices(
      walk.shapes["B"], "B", _SEARCHED_FREE, kernel)
  # Stated by the config, PLR = VA is kept for Solution and GIR to judge.
  pinned = dict(kernel, PrefetchLocalReadB=2, VgprAllocB=2, ClusterLocalReadB=0, VgprGroupB=1)
  free = _SEARCHED_FREE - {"PrefetchLocalReadB", "VgprAllocB", "ClusterLocalReadB", "VgprGroupB"}
  assert space.sideChoices(walk.shapes["B"], "B", free, pinned) == [space.SideChoice(2, 0, 1, 2)]


def test_a_nest_is_represented_by_a_spelling_without_an_extent_1_axis(tuned):
  from collections import Counter
  walks = search._walks(tuned, _SEARCHED_FREE, Counter(), {})
  nests = [tuple((name[0], extent) for name, extent in walk.axes if extent > 1)
           for _order, _kernel, walk in walks]
  assert len(nests) == len(set(nests)) == 150
  degenerate = [kernel["LoopOrder"] for _order, kernel, walk in walks
                if any(extent == 1 for _name, extent in walk.axes)]
  # Only the nests no clean spelling reaches stay degenerate: with LoopIters 2, any word with
  # two K letters splits K into 2 x 1.
  assert len(degenerate) == 16
  assert all(word.count("K") == 2 for word in degenerate)
  # MMKNKN (M1, N10, K1) walks Mi8 Ko2 No2 Ki1 Ni5; MKN walks the same tiles cleanly.
  axes = [(kernel["LoopOrder"], walk.axes) for _order, kernel, walk in walks]
  assert ("MMKNKN", [("Mi", 8), ("Ko", 2), ("No", 2), ("Ki", 1), ("Ni", 5)]) not in axes
  assert ("MKN", [("Mi", 8), ("Ki", 2), ("No", 2), ("Ni", 5)]) in axes


def test_tier3_rejects_a_candidate_it_cannot_emit():
  from tensilelite.Components.LoopModel.Auto.report import AutoResult, Ranked
  from tensilelite.Utilities.loopmodel_autoselect import tier3

  ranked = [Ranked({"VgprAllocA": va}, 1297.0) for va in (3, 4, 5)]
  result = AutoResult(ranked[0], ranked=list(ranked))
  tier3({}, result, lambda _state: None, None, 2, hw.GFX1250)
  # The two it tried are rejected, not kept after the timed ones; the untried one stays.
  assert result.ranked == [ranked[2]] and result.winner is ranked[2]
  assert result.refused == ranked[:2]
  assert result.rejected["emit"] == 2
  assert result.examples["emit"] == "Solution refuses it"


def test_an_include_leaving_out_a_derived_parameter_matches_the_ranked_kernel():
  from tensilelite.Components.LoopModel.Auto.report import AutoResult, Ranked
  from tensilelite.Utilities.loopmodel_autoselect import expand, resolveInclude

  free = ("WmmaFactorK", "VgprAllocA")
  derived = lambda state: dict(state, WmmaFactorK=1)
  tuned = resolveInclude({"WmmaFactorK": -1}, free, {"VgprAllocA": 3}, derived)
  assert tuned == {"WmmaFactorK": 1, "VgprAllocA": 3}
  # Refused by Solution, a group is kept as given.
  assert resolveInclude({}, free, {"VgprAllocA": 3}, lambda _state: None) == {"VgprAllocA": 3}

  config = {"BenchmarkProblems": [[{}, {"ForkParameters": [{"VgprAllocA": [-1]}]}]]}
  ranked = [Ranked({"WmmaFactorK": 1, "VgprAllocA": va}, 1297.0) for va in (4, 3)]
  out = expand(config, [({}, free, AutoResult(ranked[0], ranked=ranked))], include=[tuned])
  groups = out["BenchmarkProblems"][0][1]["ForkParameters"][-1]["Groups"][0]
  assert groups == [{"VgprAllocA": [4], "WmmaFactorK": [1]}, {"VgprAllocA": [3], "WmmaFactorK": [1]}]


def test_the_tuned_kernel_is_counted_at_or_below_its_emitted_vgprs(tuned):
  ev = evaluate.tier1(tuned, _TUNED_ORDER, _TUNED, hw.GFX1250, _asm_caps())
  assert ev.feasible, ev.reason
  # The rings as the writer sizes them: MXSB's unit is 5 VGPRs, not its 10-VGPR K-step.
  assert ev.gpr.operands == {"A": 192, "B": 160, "MXSA": 12, "MXSB": 5}
  # A+B already hold the tail loop's 128 + 160 slabs.
  assert ev.gpr.tailReserve == 0
  assert ev.gpr.accumulator == 640
  # Under StinkyTofu allocation the B ring's top in the writer's layout, 3 below the emitted
  # 1023: the scale region's 12 + 5 aligned to 20, and the 6 addresses below A and 2-VGPR pad
  # before B.
  assert ev.gpr.scaleAlignment == 3
  assert ev.gpr.fixedVgpr == 6 + 2
  assert ev.gpr.vgpr == 1020
  # One more A buffer is over even as a lower bound.
  wider = space.RegisterChoice(space.SideChoice(2, 0, 1, 4), _TUNED.b)
  over = evaluate.tier1(tuned, _TUNED_ORDER, wider, hw.GFX1250, _asm_caps())
  assert over.stage == "budget" and over.gpr.vgpr > 1024
  # WmmaFactorN=1 holds every B tile: the real kernel needs 1193 VGPRs and is refused.
  single = dataclasses.replace(_TUNED_ORDER, factorN=1)
  over = evaluate.tier1(tuned, single, _TUNED, hw.GFX1250, _asm_caps())
  assert over.stage == "budget" and over.gpr.vgpr == 1184


def test_tier0_reserves_what_the_tail_slabs_outgrow_the_rings_by(tuned):
  kernel, walk = _walk(tuned)
  cWmma, cLds, rho, _loads = perf.orderCore(kernel, walk, hw.GFX1250)
  # A's 8 tiles and B's 10, 16 VGPRs each.
  assert perf.tailSlabs(kernel) == 128 + 160
  single = space.SideChoice(0, 0, 1, 1)
  for b, reserve in ((_TUNED.b, 0), (single, 16)):
    costA = perf.sideCost(kernel, walk, "A", _TUNED.a, hw.GFX1250, rho, True)
    costB = perf.sideCost(kernel, walk, "B", b, hw.GFX1250, rho, True)
    assert (costA.ring, costB.ring) == ((192, 160) if b == _TUNED.b else (192, 80))
    assert max(0, perf.tailSlabs(kernel) - costA.ring - costB.ring) == reserve
    ev = evaluate.tier1(tuned, _TUNED_ORDER, space.RegisterChoice(_TUNED.a, b), hw.GFX1250,
                        _asm_caps())
    assert ev.gpr.tailReserve == reserve
  # Tier 0 counts the reserve into the pair's VGPRs.
  pinned = dict(tuned, **_TUNED.params(), **_TUNED_ORDER.params())
  pinned["PrefetchLocalReadB"], pinned["VgprAllocB"] = 0, 1
  (t, *_), _rejected, _examples = search._tier0(pinned, set(), hw.GFX1250, hw.GFX1250.maxVgpr)
  ringA = perf.sideCost(kernel, walk, "A", _TUNED.a, hw.GFX1250, rho, True)
  ringB = perf.sideCost(kernel, walk, "B", single, hw.GFX1250, rho, True)
  # Plus the pad before B and MXSA 12 + MXSB 5 aligned to 20.
  assert t.vgpr == ringA.regs + ringB.regs + 16 + 2 + 3


def test_with_the_target_tier0_counts_the_scale_rings_theta_sizes(tuned):
  pinned = dict(tuned, **_TUNED.params(), **_TUNED_ORDER.params())
  budget = hw.GFX1250.maxVgpr
  (estimated, *_), _rejected, _examples = search._tier0(pinned, set(), hw.GFX1250, budget)
  (exact, *_), _rejected, _examples = search._tier0(pinned, set(), hw.GFX1250, budget,
                                                    asmCaps=_asm_caps())
  # One VGPR per tile per buffer is 4 x 3 for MXSA and 5 x 2 for MXSB; θ's MXSB follows the
  # tiles B's ring covers, which one 5-VGPR unit spans.
  # Each aligned as the writer lays the scales out (24 and 20), with the pad before B.
  assert estimated.vgpr == 192 + 160 + 24 + 2
  assert exact.vgpr == 192 + 160 + 20 + 2
  # Over the free register choices, every kept candidate is counted as tier 1 counts it.
  ranked, _rejected, _examples = search._tier0(tuned, set(space.SIDE_REGISTER_PARAMS),
                                               hw.GFX1250, budget, asmCaps=_asm_caps())
  assert ranked
  for t in ranked:
    ev = evaluate.tier1(tuned, t.order, t.registers, hw.GFX1250, _asm_caps())
    assert t.vgpr + resources.fixedVgpr(tuned, hw.GFX1250) + ev.gpr.accumulator == ev.gpr.vgpr, \
        t.registers


def test_without_vgpr_allocation_the_writer_blocks_outside_the_rings_are_counted(tuned):
  state = dict(tuned, StinkyTofuRegisterAllocation=0)
  ev = evaluate.tier1(state, _TUNED_ORDER, _TUNED, hw.GFX1250, _asm_caps())
  assert resources.nonRingVgprs(ev.kernel, hw.GFX1250, ev.gpr.operands) == {
      "Serial": 1, "BankPadB": 2, "GL2PrefetchAddr": 4,
      "LocalReadAddrA": 2, "LocalReadAddrB": 2, "LocalReadAddrMXSA": 1, "LocalReadAddrMXSB": 1,
      "SwapAddrA": 1, "SwapAddrB": 1, "SwapAddrMXSA": 1, "SwapAddrMXSB": 1}
  # Nothing is folded into the rings, so the tuned kernel no longer fits.
  assert ev.gpr.fixedVgpr == 17 and ev.gpr.vgpr == 1029
  assert ev.stage == "budget"


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


def test_a_candidate_accept_refuses_gives_its_place_to_the_next(tuned):
  wide = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps(), tier2=False, topK=3)
  refused = wide.ranked[0].params
  result = search.search(copy.deepcopy(tuned), _SEARCHED_FREE, _asm_caps(), tier2=False, topK=2,
                         accept=lambda params: "emit" if params == refused else None)
  assert result.rejected["emit"] == 1
  assert [r.params for r in result.ranked[:2]] == [r.params for r in wide.ranked[1:3]]


def test_each_storage_region_is_one_tensor_load_per_trip(tuned):
  # The MX scales ride A's and B's descriptors (TDMFuse), so they add no load of their own.
  counts = {}
  for split in ((0, 0), (1, 0), (0, 1), (1, 1)):
    kernel = dict(tuned, TDMSplitA=split[0], TDMSplitB=split[1])
    counts[split] = perf.tensorLoads(kernel)
  assert counts == {(0, 0): 2, (1, 0): 3, (0, 1): 3, (1, 1): 4}


def test_tier0_charges_every_tensor_load_of_the_trip(tuned):
  budget = hw.GFX1250.maxVgpr
  free = dataclasses.replace(hw.GFX1250, tdmIssueCycles=0)
  (priced, *_), _rejected, _examples = search._tier0(tuned, set(), hw.GFX1250, budget)
  (bare, *_), _rejected, _examples = search._tier0(tuned, set(), free, budget)
  assert priced.score - bare.score == 2 * hw.GFX1250.tdmIssueCycles


def test_tier0_exposes_the_tensor_load_latency_of_a_one_copy_body(tuned):
  budget = hw.GFX1250.maxVgpr
  free = set(space.SIDE_REGISTER_PARAMS)
  ranked, _rejected, _examples = search._tier0(tuned, free, hw.GFX1250, budget, keep=None)
  hidden, _rejected, _examples = search._tier0(
      tuned, free, dataclasses.replace(hw.GFX1250, tdmLatency=0), budget, keep=None)
  scores = {t.registers: t.score for t in ranked}
  both = [t for t in hidden if t.registers in scores]
  assert {t.copies == 1 for t in both} == {True, False}
  # A longer body's trip (1280 WMMA cycles) covers the latency; a one-copy body drains it.
  for t in both:
    assert scores[t.registers] - t.score == (hw.GFX1250.tdmLatency if t.copies == 1 else 0)
  assert hidden[0].copies == 1
  assert ranked[0].registers == _TUNED


def test_the_tensor_load_window_runs_from_the_buffer_free_to_the_first_read(tuned):
  kernel, walk = _walk(tuned)
  assert perf.ldsDepth(kernel, "A") == perf.ldsDepth(kernel, "B") == 2
  # Chunk c overwrites chunk c-2.  From the start of c's trip: A's last read of c-2 is 260
  # WMMAs earlier, and PLR 2 reads c's first unit 80 WMMAs early.  B (PLR 1): 240 and 40.
  assert walk.tdmWindow("A", 1, 2, 3, 2) == (-260, -80)
  assert walk.tdmWindow("B", 1, 1, 2, 2) == (-240, -40)
  # PLR moves the buffer's last read and the chunk's first read alike: 220 WMMAs at any
  # depth.  PLR 2 in a three-buffer ring is shorter, its refills held for their slots.
  assert [walk.tdmWindow("A", 1, plr, None, 2) for plr in (0, 4, 8)] == [
      (-220, 0), (-380, -160), (-540, -320)]
  # A third buffer adds a trip.
  assert walk.tdmWindow("A", 1, 2, 3, 3) == (-420, -80)
  # A ring too short to be correct has no window either.
  assert walk.tdmWindow("A", 1, 0, 1, 2) is None


def test_tensor_loads_short_of_their_window_wait_in_order():
  latency, perPosition = 800.0, 8.0
  assert perf.tdmWindowExposed([None], latency, perPosition) == 0.0
  # The load issues once the buffer's last read has returned.
  assert perf.tdmWindowExposed([(0, 100, 0)], latency, perPosition) == 0.0
  assert perf.tdmWindowExposed([(0, 100, 300)], latency, perPosition) == 300
  # B's load lands during A's wait, so only A's shortfall shows.
  assert perf.tdmWindowExposed([(0, 32, 0), (10, 48, 0)], latency, perPosition) == \
      latency - 32 * 8
  # Issued after A's wait, B's load waits its own shortfall too.
  assert perf.tdmWindowExposed([(0, 32, 0), (40, 48, 0)], latency, perPosition) == \
      2 * latency - 40 * 8


def test_tier0_charges_a_tensor_load_the_window_cannot_cover(tuned):
  kernel, walk = _walk(tuned)
  cWmma, cLds, rho, _loads = perf.orderCore(kernel, walk, hw.GFX1250)
  perPosition = max(cWmma, cLds) / walk.wmmas
  windows = [perf.sideCost(kernel, walk, side, choice, hw.GFX1250, rho, True).tdm
             for side, choice in (("A", _TUNED.a), ("B", _TUNED.b))]
  # The read return is the last unit's ds_load_b128 drain past the barrier: A's 16 loads
  # 56 + 15 x 4 + 11, B's 20 loads 4 more.
  assert windows == [(-260, -80, 127), (-240, -40, 135)]
  # 256 + 320 rows land in 1267 cycles.
  landing = perf.tdmLandingLatency(kernel, hw.GFX1250)
  assert landing == pytest.approx(1267.2)
  # A's window, 180 WMMAs (1440 cycles) less its read return, covers it by 46.
  assert perf.tdmExposed(cWmma, cLds, 3, hw.GFX1250, windows, perPosition, landing) == 0.0
  # Landing in 1500, A's load is 187 short; B's lands within A's wait.
  assert perf.tdmExposed(cWmma, cLds, 3, hw.GFX1250, windows, perPosition, 1500) == 187


def test_the_ds_read_return_drains_like_stinkytofu():
  costs = hw.GFX1250
  # Up to the queue depth each load ahead costs a wave's issue slot.
  assert perf.drainLatency(costs, 56, 16, 4) == 56 + 15 * 4
  # Past it ds_load_b128 retires two a cycle, and its drain stops at 255.
  assert perf.drainLatency(costs, 56, 40, 4) == 56 + 15 * 4 + 24 * 4 // 2
  assert perf.drainLatency(costs, 56, 100, 4) == 255
  # ds_load_b64 retires four a cycle and stops at 131.
  assert perf.drainLatency(costs, 52, 40, 4, 8) == 131
  # Waves sharing the queue clamp to 1..4.
  assert perf.drainLatency(costs, 56, 16, 8) == perf.drainLatency(costs, 56, 16, 4)
  assert perf.drainLatency(costs, 56, 16, 0) == 56 + 15


def test_tier2_waits_for_the_tensor_load_filling_an_lds_generation(tuned):
  ev = evaluate.tier1(tuned, _TUNED_ORDER, _TUNED, hw.GFX1250, _asm_caps())
  assert ev.feasible, ev.reason
  program = perf.lower(ev.theta)
  landed = dataclasses.replace(hw.GFX1250, tdmLatency=0)
  late = dataclasses.replace(hw.GFX1250, tdmLatency=4000)
  timeline = perf.simulate(ev.kernel, program, landed)
  assert timeline.tdmWait == 0
  assert timeline.tdmIssue == perf.tensorLoads(ev.kernel) * hw.GFX1250.tdmIssueCycles
  # A load landing after its trip's ds_loads read it holds issue up.
  slow = perf.simulate(ev.kernel, program, late)
  assert slow.tdmWait > 0 and slow.cycles > timeline.cycles
  assert perf.simulate(ev.kernel, program, hw.GFX1250).tdmWait == 0


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


def _free_config(tmp_path, autoSelect):
  from tensilelite import LibraryIO
  import yaml

  config = LibraryIO.read(_CONFIG)
  forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
  for entry in forks:
    for name in list(entry):
      if name in _SEARCHED_FREE:
        entry[name] = [-1]
  forks.append({"LoopModelAutoSelect": [autoSelect]})
  path = tmp_path / ("auto%d.yaml" % autoSelect)
  path.write_text(yaml.safe_dump(config, default_flow_style=None))
  return str(path)


def test_auto_select_states_its_choice_in_the_solution(tmp_path):
  states = _derive(_free_config(tmp_path, 1))
  assert len(states) == 1
  state = states[0]
  assert state["LoopModelAutoSelect"] == 0
  for name in _SEARCHED_FREE:
    assert int(state[name]) != -1, name
  # Stated values: the no-load loop is dropped, as for a config that names them.
  assert state["SuppressNoLoadLoop"] is True


def test_auto_select_n_benchmarks_the_n_best_candidates(tmp_path):
  states = _derive(_free_config(tmp_path, 3))
  assert len(states) == 3
  chosen = [tuple(int(state[name]) for name in sorted(_SEARCHED_FREE)) for state in states]
  assert len(set(chosen)) == 3
  for state in states:
    assert state["LoopModelAutoSelect"] == 0
    assert state["SuppressNoLoadLoop"] is True
  # The best of the N is the kernel Derive mode picks alone.
  single = _derive(_free_config(tmp_path, 1))[0]
  assert chosen[0] == tuple(int(single[name]) for name in sorted(_SEARCHED_FREE))
