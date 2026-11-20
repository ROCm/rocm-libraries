# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""KernelWriter._mxsInterleavePlan / _mxsDefaultBlocks / remapMXScaleInterleave: when the gfx1250
MX scale interleave applies, the reason it gives when it does not, and where it moves a reference.

CPU-only; the writer and kernel are minimal stand-ins, so no codegen runs.
"""

from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import vgpr
from rocisa.instruction import DSLoadB32, DSLoadB128

from tensilelite.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit


def _writer(numVgprBuffer=2, lrvwTile=1, valuPerBlock=(4, 4), bufferMXSA=0, bufferMXSB=0):
  states = SimpleNamespace(
    numVgprBuffer=numVgprBuffer,
    # 0 means allocation has not run, so the per-operand depth falls back to the shared count.
    numVgprBufferA=0,
    numVgprBufferB=0,
    numVgprBufferMXSA=bufferMXSA,
    numVgprBufferMXSB=bufferMXSB,
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
  writer._mxsPackGroup = KernelWriter._mxsPackGroup
  writer._mxsPackGroupFirstFit = KernelWriter._mxsPackGroupFirstFit
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


def _loopModelWriter():
  """An MX-F8 UseLoopModel shape: two 4-VGPR MXSA buffers and one 5-VGPR MXSB buffer under a
  three-deep scaffold, the counts that once sized the region for three of each."""
  writer = _writer(numVgprBuffer=3, valuPerBlock=(4, 5), bufferMXSA=2, bufferMXSB=1)
  writer.states.lrvwTileMXSA, writer.states.lrvwTileMXSB = 8, 1
  writer.states.mxsa.numVgprValu, writer.states.mxsb.numVgprValu = 8, 5
  return writer


def _raggedWriter():
  """The same kernel once the scale rings follow their tile coverage: two 4-VGPR MXSA buffers,
  and an MXSB ring of two buffers holding 3 and 2 VGPRs (5 in all), which the writer addresses
  as 3-VGPR blocks."""
  writer = _writer(numVgprBuffer=3, valuPerBlock=(4, 3), bufferMXSA=2, bufferMXSB=2)
  writer.states.lrvwTileMXSA, writer.states.lrvwTileMXSB = 8, 1
  writer.states.mxsa.numVgprValu, writer.states.mxsb.numVgprValu = 8, 5
  return writer


def test_a_ragged_rings_last_block_ends_with_the_side():
  writer = _raggedWriter()
  assert KernelWriter._mxsDefaultBlocks(writer, _kernel(), "B") == {(0, 0): 0, (1, 0): 3}
  plan, reason = _plan(writer, _kernel(UseLoopModel=True), widthA=4)
  assert reason is None
  assert plan["sizes"] == {"A": {0: 4, 4: 4}, "B": {0: 3, 3: 2}}
  # Only the local read's own width divides both 3 and 2.
  assert plan["piece"] == {"A": 4, "B": 1}


def test_the_loop_model_pairs_every_buffer_with_every_buffer():
  # Its rings split by free tile within one K step, so MXSA's first buffer also meets MXSB's
  # second. Local-read stages rotate together.
  writer = _raggedWriter()
  groups = lambda kernel: [(grp["A"], grp["B"]) for grp in _plan(writer, kernel, widthA=4)[0]["groups"]]
  assert groups(_kernel(UseLoopModel=True)) == [([0, 4], [0, 3])]
  assert groups(_kernel()) == [([0], [0]), ([4], [3])]


def test_ragged_loop_model_shape_fits_the_default_span_on_disjoint_banks():
  plan, regionLen, reason = KernelWriter._mxsInterleaveRegion(
    _raggedWriter(), _kernel(UseLoopModel=True), _tensor(4), _tensor(), 0, False)
  assert reason is None and regionLen == 15
  (grp,) = plan["groups"]
  regs = {tc: {c for pc, p in grp["cells"].items() if pc[0] == tc for c in range(p, p + plan["piece"][tc])}
          for tc in "AB"}
  assert grp["span"] == 14
  assert regs == {"A": {0, 1, 2, 3, 8, 9, 10, 11}, "B": {4, 5, 6, 7, 12}}


def _threeBufferWriter():
  """The ragged kernel at VgprAllocA 3, once a merged scale ring keeps the depth its coverage asks
  for: three 4-VGPR MXSA buffers beside MXSB's 3 and 2. The default lays the 17 VGPRs out in 19,
  MXSB from v14, and a bank-disjoint layout needs 20."""
  writer = _writer(numVgprBuffer=3, valuPerBlock=(4, 3), bufferMXSA=3, bufferMXSB=2)
  writer.states.lrvwTileMXSA, writer.states.lrvwTileMXSB = 8, 1
  writer.states.mxsa.numVgprValu, writer.states.mxsb.numVgprValu = 12, 5
  return writer


def _accStart(writer, kernel, between=0, vgprIdx=0):
  """A stand-in for _layoutMXScaleVgprs when `between` VGPRs of packs or G2L follow the scales:
  the accumulators start on the even VGPR after them."""
  def start(regionLen):
    if regionLen is None:
      regionLen = KernelWriter._alignMXSB(writer, kernel, vgprIdx + writer.states.mxsa.numVgprValu) \
        + writer.states.mxsb.numVgprValu - vgprIdx
    return (vgprIdx + regionLen + between + 1) // 2 * 2
  return start


def test_three_mxsa_buffers_need_a_vgpr_past_the_default_span():
  plan, regionLen, reason = KernelWriter._mxsInterleaveRegion(
    _threeBufferWriter(), _kernel(UseLoopModel=True), _tensor(4), _tensor(), 0, False)
  assert (plan, regionLen) == (None, 0) and "20 VGPRs, more than the default's 19" in reason


def test_a_longer_region_is_taken_when_the_accumulators_keep_their_place():
  # The default's 19 VGPRs end on v18, so the accumulators start at v20 either way.
  writer, kernel = _threeBufferWriter(), _kernel(UseLoopModel=True)
  plan, regionLen, reason = KernelWriter._mxsInterleaveRegion(
    writer, kernel, _tensor(4), _tensor(), 0, False, _accStart(writer, kernel))
  assert reason is None and regionLen == 20
  (grp,) = plan["groups"]
  regs = {tc: {c for pc, p in grp["cells"].items() if pc[0] == tc for c in range(p, p + plan["piece"][tc])}
          for tc in "AB"}
  assert regs == {"A": {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19}, "B": {4, 5, 6, 7, 12}}


def test_a_longer_region_is_refused_when_it_moves_the_accumulators():
  # One more VGPR after the scales ends the default on v19, so the longer region would push the
  # accumulators from v20 to v22.
  writer, kernel = _threeBufferWriter(), _kernel(UseLoopModel=True)
  plan, regionLen, reason = KernelWriter._mxsInterleaveRegion(
    writer, kernel, _tensor(4), _tensor(), 0, False, _accStart(writer, kernel, between=1))
  assert (plan, regionLen) == (None, 0)
  assert "20 VGPRs, more than the default's 19, and would move the accumulators from v20 to v22" in reason


def test_default_blocks_follow_each_operands_own_buffers():
  writer = _loopModelWriter()
  assert KernelWriter._mxsDefaultBlocks(writer, _kernel(), "A") == {(0, 0): 0, (1, 0): 4}
  assert KernelWriter._mxsDefaultBlocks(writer, _kernel(), "B") == {(0, 0): 0}


def test_unequal_buffer_counts_put_every_block_in_one_group():
  plan, reason = _plan(_loopModelWriter(), _kernel(UseLoopModel=True), widthA=4)
  assert reason is None
  assert [(grp["A"], grp["B"]) for grp in plan["groups"]] == [([0, 4], [0])]


def test_loop_model_shape_fits_the_default_span_on_disjoint_banks():
  # UseLoopModel names the path: the bank split is taken only where the layout is being changed,
  # and the scaffold keeps the one-at-a-time placement its kernels are laid out with.
  plan, regionLen, reason = KernelWriter._mxsInterleaveRegion(
    _loopModelWriter(), _kernel(UseLoopModel=True), _tensor(4), _tensor(), 0, False)
  assert reason is None and regionLen == 15
  (grp,) = plan["groups"]
  regs = {tc: {c for pc, p in grp["cells"].items() if pc[0] == tc for c in range(p, p + plan["piece"][tc])}
          for tc in "AB"}
  assert grp["span"] == 14
  assert regs == {"A": {0, 1, 2, 3, 8, 9, 10, 11}, "B": {4, 5, 6, 7, 12}}


def _remapped(*dsts, writer=None, kernel=None):
  """The destination text of one ds_load per `dsts` entry, after remapping the interleave of
  `writer`, by default _loopModelWriter: MXSA blocks X0 at +0 and X1 at +8, MXSB pieces at
  +4..+7 and +12."""
  writer = writer or _loopModelWriter()
  writer.states.mxsInterleave, _reason = _plan(writer, kernel or _kernel(), widthA=4)
  module = Module("remap")
  loads = [DSLoadB128(vgpr(name, 4), vgpr("LocalReadAddrMXSA")) if width == 4 else
           DSLoadB32(vgpr(name), vgpr("LocalReadAddrMXSB")) for name, width in dsts]
  for load in loads:
    module.add(load)
  KernelWriter.remapMXScaleInterleave(writer, module)
  return [str(load).split()[1].rstrip(",") for load in loads]


def test_remap_moves_a_reference_to_the_block_the_default_layout_puts_it_in():
  # Loop-model drain code names MXSA's second buffer through the first one's symbol. X1 no longer
  # follows X0, so reading +4 as piece 1 of X0 named a piece that does not exist.
  assert _remapped(("ValuMXSA_X0_I0+4", 4), ("ValuMXSA_X1_I0", 4), ("ValuMXSA_X0_I0", 4)) == [
    "v[vgprValuMXSA_X1_I0+0:vgprValuMXSA_X1_I0+0+3]",
    "v[vgprValuMXSA_X1_I0:vgprValuMXSA_X1_I0+3]",
    "v[vgprValuMXSA_X0_I0:vgprValuMXSA_X0_I0+3]",
  ]


def test_remap_points_a_reference_inside_its_block_at_the_piece():
  assert _remapped(("ValuMXSB_X0_I0", 1), ("ValuMXSB_X0_I0+2", 1), ("ValuMXSB_X0_I0+4", 1)) == [
    "v[vgprValuMXSB_X0_I0]", "v[vgprValuMXSB_X0_I0_P2+0]", "v[vgprValuMXSB_X0_I0_P4+0]"]


def test_remap_moves_a_reference_into_a_ragged_rings_short_block():
  # The WMMAs also reach MXSB's second buffer as X0_I0+3 and +4, its only two VGPRs.
  assert _remapped(("ValuMXSB_X0_I0+2", 1), ("ValuMXSB_X0_I0+3", 1), ("ValuMXSB_X0_I0+4", 1),
                   ("ValuMXSB_X1_I0+1", 1), writer=_raggedWriter(), kernel=_kernel(UseLoopModel=True)) == [
    "v[vgprValuMXSB_X0_I0_P2+0]", "v[vgprValuMXSB_X1_I0+0]", "v[vgprValuMXSB_X1_I0_P1+0]",
    "v[vgprValuMXSB_X1_I0_P1+0]"]


def test_remap_refuses_a_reference_past_every_block():
  with pytest.raises(AssertionError, match="outside every MXSA block"):
    _remapped(("ValuMXSA_X1_I0+4", 4))


def _firstFitSpan(lists, piece):
  """The packing the bank split replaced: pieces placed one by one, MXSA and MXSB alternating,
  each at the first free start that takes no bank the other side holds."""
  order = [lst[i] for i in range(max(map(len, lists.values()))) for lst in (lists["A"], lists["B"]) if i < len(lst)]
  used, banks = set(), {"A": set(), "B": set()}
  for tc, _off, _k in order:
    size, other = piece[tc], banks["B" if tc == "A" else "A"]
    p = 0
    while any(c in used for c in range(p, p + size)) or {c % 8 for c in range(p, p + size)} & other:
      p += 2 if size >= 2 else 1
    used.update(range(p, p + size))
    banks[tc].update(c % 8 for c in range(p, p + size))
  return (max(used) + 2) // 2 * 2


@pytest.mark.parametrize("sizeA, sizeB", [(a, b) for a in (1, 2, 3, 4) for b in (1, 2, 3, 4)])
def test_bank_split_packs_no_looser_than_first_fit(sizeA, sizeB):
  piece = {"A": sizeA, "B": sizeB}
  for countA in range(1, 7):
    for countB in range(1, 7):
      lists = {"A": [("A", 0, k) for k in range(countA)], "B": [("B", 0, k) for k in range(countB)]}
      cells, span = KernelWriter._mxsPackGroup(lists, piece)
      regs = {tc: [c for pc, p in cells.items() if pc[0] == tc for c in range(p, p + piece[tc])] for tc in "AB"}
      assert span <= _firstFitSpan(lists, piece), (countA, countB)
      assert not {c % 8 for c in regs["A"]} & {c % 8 for c in regs["B"]}
      allRegs = regs["A"] + regs["B"]
      assert len(allRegs) == len(set(allRegs)) and max(allRegs) < span
      assert all(p % 2 == 0 for pc, p in cells.items() if piece[pc[0]] >= 2)


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
  # Each group's two 3-VGPR pieces start on even VGPRs, banks apart: 8 a group, 16 in all,
  # where the default lays the same 12 VGPRs out in 14.
  writer = _writer(valuPerBlock=(3, 3))
  writer.states.mxsa.numVgprValu = writer.states.mxsb.numVgprValu = 6
  plan, regionLen, reason = _region(writer, _kernel())
  assert (plan, regionLen) == (None, 0) and "16 VGPRs, more than the default's 14" in reason


def test_plan_refuses_a_block_past_what_its_side_holds():
  writer = _writer()
  writer.states.mxsb.numVgprValu = 4
  plan, reason = _plan(writer, _kernel())
  assert plan is None and "an MXSB block starts past the 4 VGPRs the side holds" in reason


@pytest.mark.parametrize("pap, applies", [(False, False), (True, True)])
def test_region_needs_prefetch_global_read_or_pap(pap, applies):
  plan, regionLen, reason = _region(_writer(), _kernel(PrefetchGlobalRead=0), papEnabled=pap)
  assert (plan is not None) == applies
  if not applies:
    assert regionLen == 0 and "G2L" in reason


def test_region_falls_back_past_v255():
  plan, regionLen, reason = _region(_writer(), _kernel(), vgprIdx=250)
  assert (plan, regionLen) == (None, 0) and "v255" in reason
