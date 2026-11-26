# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""KernelWriter._mxsInterleavePlan / _mxsDefaultBlocks: when the gfx1250 MX scale interleave
applies and the reason it gives when it does not.

CPU-only; the writer and kernel are minimal stand-ins, so no codegen runs.
"""

from types import SimpleNamespace

import pytest

from tensilelite.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit


def _writer(numVgprBuffer=2, lrvwTile=1, valuPerBlock=(4, 4)):
  states = SimpleNamespace(
    numVgprBuffer=numVgprBuffer,
    archCaps={"VgprBank": True},
    lrvwTileMXSA=lrvwTile,
    lrvwTileMXSB=lrvwTile,
    kernelName="kernel",
    mxsa=SimpleNamespace(numVgprValu=16, numVgprValuPerBlock=valuPerBlock[0]),
    mxsb=SimpleNamespace(numVgprValu=16, numVgprValuPerBlock=valuPerBlock[1]),
  )
  writer = SimpleNamespace(states=states)
  writer._mxsDefaultBlocks = lambda kernel, tc: KernelWriter._mxsDefaultBlocks(writer, kernel, tc)
  writer._alignMXSB = lambda kernel, idx: KernelWriter._alignMXSB(writer, kernel, idx)
  writer._mxsInterleavePlan = lambda kernel, tPA, tPB, **kw: KernelWriter._mxsInterleavePlan(writer, kernel, tPA, tPB, **kw)
  writer._mxsBudgetPlan = lambda *args: KernelWriter._mxsBudgetPlan(writer, *args)
  writer._mxsLayoutConflicts = lambda *args, **kw: KernelWriter._mxsLayoutConflicts(writer, *args, **kw)
  return writer


def _kernel(**over):
  kernel = {
    "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    "ReuseAcrossPersistent": 0,
    "DirectToVgprA": 0, "DirectToVgprB": 0, "DirectToVgprMXSA": 0, "DirectToVgprMXSB": 0,
    "InnerUnroll": 1,
    "UseSubtileImpl": 0,
    "PrefetchGlobalRead": 2,
    "ISA": (12, 5, 0),
    "UnrollMajorLDSA": 1, "UnrollMajorLDSB": 1,
    "SourceSwap": 1,
    "MXScaleVgprInterleaveExtraVgpr": False,
  }
  kernel.update(over)
  return kernel


def _tensor(blockWidth=1):
  return {"localReadInstruction": SimpleNamespace(blockWidth=blockWidth)}


def _plan(writer, kernel, widthA=1, widthB=1, **kw):
  return KernelWriter._mxsInterleavePlan(writer, kernel, _tensor(widthA), _tensor(widthB), **kw)


def test_plan_packs_blocks_on_disjoint_banks():
  plan, reason = _plan(_writer(), _kernel())
  assert reason is None
  assert plan["piece"] == {"A": 4, "B": 4}
  for grp in plan["groups"]:
    banks = {tc: {c % 8 for pc, p in grp["cells"].items() if pc[0] == tc for c in range(p, p + 4)} for tc in "AB"}
    assert not banks["A"] & banks["B"]


def test_plan_splits_a_large_block_into_pieces():
  plan, reason = _plan(_writer(valuPerBlock=(8, 2)), _kernel())
  assert reason is None
  assert plan["piece"] == {"A": 4, "B": 2}
  assert {pc[2] for pc in plan["groups"][0]["cells"] if pc[0] == "A"} == {0, 1}


def test_plan_aliased_buffers_share_a_block():
  # With an MXS tile read wider than a scale and a tile-major LDS layout every buffer
  # restarts at offset 0, so the buffers alias one block per (bi, iui).
  writer = _writer(numVgprBuffer=2, lrvwTile=2)
  kernel = _kernel(UnrollMajorLDSA=0, UnrollMajorLDSB=0)
  assert KernelWriter._mxsDefaultBlocks(writer, kernel, "A") == {(0, 0): 0, (1, 0): 0}
  plan, reason = _plan(writer, kernel)
  assert reason is None
  assert len(plan["groups"]) == 1
  assert len(plan["groups"][0]["A"]) == 1


def test_default_blocks_advance_per_buffer_and_inner_unroll():
  assert KernelWriter._mxsDefaultBlocks(_writer(), _kernel(InnerUnroll=2), "A") == \
    {(0, 0): 0, (0, 1): 4, (1, 0): 8, (1, 1): 12}


@pytest.mark.parametrize("problemType", [{"MXBlockA": 0, "MXBlockB": 32}, {"MXBlockA": 32, "MXBlockB": 0}])
def test_plan_needs_mx_scales_on_both_operands(problemType):
  plan, reason = _plan(_writer(), _kernel(ProblemType=problemType))
  assert plan is None and "needs MX scales" in reason


@pytest.mark.parametrize("side", ["mxsa", "mxsb"])
def test_plan_needs_scale_vgprs_on_both_operands(side):
  writer = _writer()
  getattr(writer.states, side).numVgprValu = 0
  plan, reason = _plan(writer, _kernel())
  assert plan is None and "needs scale VGPRs" in reason


def test_plan_not_applied_with_reuse_across_persistent():
  plan, reason = _plan(_writer(), _kernel(ReuseAcrossPersistent=1))
  assert plan is None and "ReuseAcrossPersistent" in reason


def test_plan_not_applied_on_the_subtile_path():
  plan, reason = _plan(_writer(), _kernel(UseSubtileImpl=1))
  assert plan is None and "UseSubtileImpl" in reason


@pytest.mark.parametrize("tc", ["A", "B", "MXSA", "MXSB"])
def test_plan_not_applied_with_direct_to_vgpr(tc):
  plan, reason = _plan(_writer(), _kernel(**{"DirectToVgpr" + tc: 1}))
  assert plan is None and "DirectToVgpr" in reason


def test_plan_not_applied_when_a_block_cannot_be_split():
  # 6 VGPRs is neither <= 4 nor a multiple of 4, and a 4-VGPR local read does not divide it.
  plan, reason = _plan(_writer(valuPerBlock=(6, 4)), _kernel(), widthA=4)
  assert plan is None and "cannot be split" in reason


def test_plan_not_applied_when_the_local_read_is_wider_than_a_piece():
  plan, reason = _plan(_writer(valuPerBlock=(2, 4)), _kernel(), widthA=4)
  assert plan is None and "does not fit" in reason


@pytest.mark.parametrize("isa, vgprBank, idx, expected", [
  (12, True, 100, 102),   # gfx1250: the next even VGPR above idx
  (12, True, 101, 102),
  (12, False, 101, 102),  # no VGPR banks: round up to an even VGPR
  (12, False, 100, 100),
  (9, True, 100, 101),    # other ISAs with banks: even bank 0/2 moves to bank 1/3
  (9, True, 101, 101),
])
def test_align_mxsb(isa, vgprBank, idx, expected):
  writer = _writer()
  writer.states.archCaps["VgprBank"] = vgprBank
  assert KernelWriter._alignMXSB(writer, _kernel(ISA=(isa, 5, 0)), idx) == expected


def _region(writer, kernel, vgprIdx=100, papEnabled=False):
  return KernelWriter._mxsInterleaveRegion(writer, kernel, _tensor(), _tensor(), vgprIdx, papEnabled)


def test_region_is_the_default_length():
  plan, regionLen, reason = _region(_writer(), _kernel())
  assert plan is not None and reason is None
  # align(100 + 16) = 118, plus 16 MXSB VGPRs, from v100
  assert regionLen == 34


def test_region_falls_back_when_the_plan_does_not_apply():
  plan, regionLen, reason = _region(_writer(), _kernel(ReuseAcrossPersistent=1))
  assert (plan, regionLen) == (None, 0) and "ReuseAcrossPersistent" in reason


def _unbalanced_writer():
  # MIWaveTile 4x13 on one side: a 4-VGPR MXSA block (one piece) against a 13-VGPR MXSB block
  # (13 one-VGPR pieces), two buffers, so two pairing groups.
  writer = _writer(valuPerBlock=(4, 13))
  writer.states.mxsa.numVgprValu = 4 * 2
  writer.states.mxsb.numVgprValu = 13 * 2
  return writer


def test_shared_groups_pack_into_each_others_holes():
  writer = _unbalanced_writer()
  stacked, _ = _plan(writer, _kernel())
  shared, reason = _plan(writer, _kernel(), shareGroups=True)
  assert reason is None
  assert len(stacked["groups"]) == 2 and len(shared["groups"]) == 1
  stackedSpan = sum(g["span"] for g in stacked["groups"])
  sharedSpan = shared["groups"][0]["span"]
  assert stackedSpan == 60
  assert sharedSpan < stackedSpan
  # the scales of the two groups are 2 * (4 + 13) VGPRs; none shares a VGPR
  cells = shared["groups"][0]["cells"]
  used = [c for pc, p in cells.items() for c in range(p, p + shared["piece"][pc[0]])]
  assert len(used) == len(set(used)) == 2 * (4 + 13)


def test_shared_groups_keep_each_pairing_bank_disjoint():
  writer = _unbalanced_writer()
  shared, _ = _plan(writer, _kernel(), shareGroups=True)
  cells = shared["groups"][0]["cells"]
  piece = shared["piece"]
  # A scale block and the B scale block a WMMA reads with it come from the same (bi, iui).
  for key, offA in shared["blocks"]["A"].items():
    offB = shared["blocks"]["B"][key]
    banksA = {c % 8 for pc, p in cells.items() if pc[0] == "A" and pc[1] == offA for c in range(p, p + piece["A"])}
    banksB = {c % 8 for pc, p in cells.items() if pc[0] == "B" and pc[1] == offB for c in range(p, p + piece["B"])}
    assert banksA and banksB and not banksA & banksB


def test_shared_groups_leave_a_single_group_alone():
  writer = _writer(numVgprBuffer=2, lrvwTile=2)  # buffers alias: one pairing group
  kernel = _kernel(UnrollMajorLDSA=0, UnrollMajorLDSB=0)
  stacked, _ = _plan(writer, kernel)
  shared, _ = _plan(writer, kernel, shareGroups=True)
  assert shared["groups"][0]["cells"] == stacked["groups"][0]["cells"]
  assert shared["groups"][0]["span"] == stacked["groups"][0]["span"]


def _conflicts(plan, pairing):
  """sum over pairing groups and banks of (#MXSA VGPRs on the bank) * (#MXSB VGPRs on the bank)"""
  cells = plan["groups"][0]["cells"] if len(plan["groups"]) == 1 else None
  total = 0
  for grp in pairing:
    count = {"A": [0] * 8, "B": [0] * 8}
    for tc in "AB":
      for off in grp[tc]:
        for pc, p in cells.items():
          if pc[0] == tc and pc[1] == off:
            for c in range(p, p + plan["piece"][tc]):
              count[tc][c % 8] += 1
    total += sum(a * b for a, b in zip(count["A"], count["B"]))
  return total


def test_region_shares_groups_when_extra_vgprs_are_allowed():
  writer = _unbalanced_writer()
  # default: align(0 + 8) = 10, plus 26 MXSB VGPRs, from v0 -> 36; the stacked layout needs 60
  plan, regionLen, reason = _region(writer, _kernel(MXScaleVgprInterleaveExtraVgpr=True), vgprIdx=0)
  assert reason is None and len(plan["groups"]) == 1 and "conflicts" not in plan
  assert 36 <= regionLen < 60
  # a layout that already fits keeps its per-group stacking
  plan, regionLen, reason = _region(_writer(), _kernel())
  assert reason is None and len(plan["groups"]) == 2 and regionLen == 34


def test_region_stays_in_the_default_length_with_the_fewest_conflicts():
  writer = _unbalanced_writer()
  stacked, _ = _plan(writer, _kernel())
  plan, regionLen, reason = _region(writer, _kernel(), vgprIdx=0)
  assert reason is None
  assert regionLen == 36  # the default length: no extra VGPRs
  assert plan["groups"][0]["span"] <= regionLen
  # the pieces do not overlap
  cells = plan["groups"][0]["cells"]
  used = [c for pc, p in cells.items() for c in range(p, p + plan["piece"][pc[0]])]
  assert len(used) == len(set(used)) == 2 * (4 + 13)
  # never worse than keeping the default layout (MXSA from 0, MXSB from 10 = align(8))
  default = {pc: (0 if pc[0] == "A" else 10) + pc[1] + pc[2] * stacked["piece"][pc[0]] for g in stacked["groups"]
             for pc in g["cells"]}
  defaultPlan = {"piece": stacked["piece"], "groups": [{"cells": default}]}
  assert plan["conflicts"] == _conflicts(plan, stacked["groups"])
  assert plan["conflicts"] <= _conflicts(defaultPlan, stacked["groups"])
  # and it does remove conflicts the default layout has
  assert plan["conflicts"] < _conflicts(defaultPlan, stacked["groups"])


def test_region_without_a_fitting_conflict_free_layout_is_conflict_free_when_one_fits():
  # a balanced kernel: the conflict-free layout fits the default length and is used as is
  plan, regionLen, reason = _region(_writer(), _kernel())
  assert reason is None and "conflicts" not in plan


def test_layout_conflicts_count_shared_banks_of_the_default_layout():
  writer = _unbalanced_writer()
  stacked, _ = _plan(writer, _kernel())
  # default layout: MXSA at 0..7 (two 4-VGPR blocks), MXSB from 10; per group 4 MXSA x 13 MXSB pairs
  conflicts, pairs = KernelWriter._mxsLayoutConflicts(writer, stacked, 10)
  assert pairs == 2 * 4 * 13
  default = {pc: (0 if pc[0] == "A" else 10) + pc[1] + pc[2] * stacked["piece"][pc[0]] for g in stacked["groups"]
             for pc in g["cells"]}
  assert KernelWriter._mxsLayoutConflicts(writer, stacked, 10, default) == (conflicts, pairs)
  assert 0 < conflicts < pairs
  # the budget layout lowers it and reports the same number it gets from the same model
  plan, _regionLen, _reason = _region(writer, _kernel(), vgprIdx=0)
  assert KernelWriter._mxsLayoutConflicts(writer, stacked, 10, plan["groups"][0]["cells"])[0] == plan["conflicts"]
  assert plan["conflicts"] < conflicts
