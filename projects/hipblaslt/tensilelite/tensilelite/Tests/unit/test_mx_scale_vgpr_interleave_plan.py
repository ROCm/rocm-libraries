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
    mxsa=SimpleNamespace(numVgprValu=16, numVgprValuPerBlock=valuPerBlock[0]),
    mxsb=SimpleNamespace(numVgprValu=16, numVgprValuPerBlock=valuPerBlock[1]),
  )
  writer = SimpleNamespace(states=states)
  writer._mxsDefaultBlocks = lambda kernel, tc: KernelWriter._mxsDefaultBlocks(writer, kernel, tc)
  writer._alignMXSB = lambda kernel, idx: KernelWriter._alignMXSB(writer, kernel, idx)
  writer._mxsInterleavePlan = lambda kernel, tPA, tPB: KernelWriter._mxsInterleavePlan(writer, kernel, tPA, tPB)
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
  }
  kernel.update(over)
  return kernel


def _tensor(blockWidth=1):
  return {"localReadInstruction": SimpleNamespace(blockWidth=blockWidth)}


def _plan(writer, kernel, widthA=1, widthB=1):
  return KernelWriter._mxsInterleavePlan(writer, kernel, _tensor(widthA), _tensor(widthB))


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


def test_region_falls_back_when_the_layout_needs_more_vgprs_than_the_default():
  writer = _writer()
  writer.states.mxsa.numVgprValu = 1
  writer.states.mxsb.numVgprValu = 1
  plan, regionLen, reason = _region(writer, _kernel())
  assert (plan, regionLen) == (None, 0) and "more than the default" in reason


@pytest.mark.parametrize("pap, applies", [(False, False), (True, True)])
def test_region_needs_prefetch_global_read_or_pap(pap, applies):
  plan, regionLen, reason = _region(_writer(), _kernel(PrefetchGlobalRead=0), papEnabled=pap)
  assert (plan is not None) == applies
  if not applies:
    assert regionLen == 0 and "G2L" in reason


def test_region_falls_back_past_v255():
  plan, regionLen, reason = _region(_writer(), _kernel(), vgprIdx=250)
  assert (plan, regionLen) == (None, 0) and "v255" in reason
