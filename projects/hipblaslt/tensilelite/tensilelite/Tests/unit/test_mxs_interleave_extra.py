# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""extraVgprMatters: does MXScaleVgprInterleaveExtraVgpr change the scale layout?

Target code: tensilelite/SolutionStructs/mxs_interleave_extra.py. The cross-check against the real
KernelWriter (every gfx1250 MX logic kernel) lives in the codegen characterization tests.
"""
from types import SimpleNamespace

from tensilelite.KernelWriter import KernelWriter

import pytest

from tensilelite.SolutionStructs import mxs_interleave_extra as X
from tensilelite.Components.LocalRead import LocalReadMFMA

pytestmark = pytest.mark.unit

_ISA = SimpleNamespace(asmCaps={"HasWMMA_V3": True}, archCaps={"VgprBank": True})


@pytest.fixture(autouse=True)
def _no_tile_span(monkeypatch):
  # TileSpan only halves the scale block size; it is covered by the codegen cross-check.
  monkeypatch.setattr(LocalReadMFMA, "getMxsTileSpanInfo", staticmethod(lambda *a, **k: None))


def _state(miWaveTileA, miWaveTileB, plr, **over):
  state = {
    "ProblemType": {"MXBlockA": 32, "MXBlockB": 32, "Index01A": 0, "Index01B": 1},
    "ISA": (12, 5, 0), "MXScaleFormat": "InMemorySwizzle",
    "ReuseAcrossPersistent": 0, "UseSubtileImpl": 0, "ForceUnrollSubIter": 0,
    "DirectToVgprA": 0, "DirectToVgprB": 0, "DirectToVgprMXSA": 0, "DirectToVgprMXSB": 0,
    "InnerUnroll": 1, "PrefetchGlobalRead": 2, "SourceSwap": 1,
    "PrefetchLocalRead": plr, "ClusterLocalRead": 0, "LoopIters": 8,
    "MIWaveTileMXSA": miWaveTileA, "MIWaveTileMXSB": miWaveTileB,
    "MIInputPerThreadMXSA": 4, "MIInputPerThreadMXSB": 4,
    "VectorWidthA": 1, "VectorWidthB": 1, "VectorWidthMXSA": 1, "VectorWidthMXSB": 1,
    "UnrollMajorLDSA": 0, "UnrollMajorLDSB": 0, "UnrollMajorLDSMXSA": 0, "UnrollMajorLDSMXSB": 0,
    "LocalReadVectorWidthMXS": 1,
  }
  state.update(over)
  return state


def test_select_block_width_follows_the_local_read_table():
  assert X._selectBlockWidth(16) == 4      # widest single-address read that divides the width
  assert X._selectBlockWidth(2) == 2       # b64; the two-offset load2_b64 is skipped
  assert X._selectBlockWidth(1) == 1
  assert X._selectBlockWidth(0.25) == 0.25
  assert X._selectBlockWidth(0.1) is None  # narrower than every instruction


def test_two_buffers_share_holes_so_the_default_span_is_enough():
  # MIWaveTile [4,11], two buffers: the stacked layout needs 48 VGPRs, the shared one 32 = default.
  assert X.extraVgprMatters(_state(4, 11, plr=1), _ISA) is False


def test_single_group_with_a_tiny_a_side_needs_more_than_the_default_span():
  # [4,19], one buffer, one group: nobody to share holes with (cluster_padding / [4,19] case).
  assert X.extraVgprMatters(_state(4, 19, plr=0), _ISA) is True


def test_balanced_blocks_fit():
  assert X.extraVgprMatters(_state(8, 8, plr=1), _ISA) is False


@pytest.mark.parametrize("over", [
  {"ISA": (9, 4, 2)},
  {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0, "Index01A": 0, "Index01B": 1}},
  {"UseSubtileImpl": 1},
  {"ReuseAcrossPersistent": 1},
  {"DirectToVgprA": 1},
  {"DirectToVgprMXSB": 1},
  {"ForceUnrollSubIter": 1},
  {"MXScaleFormat": "NoSwizzle"},
])
def test_cases_it_cannot_reproduce_are_undecidable(over):
  assert X.extraVgprMatters(_state(4, 19, plr=0, **over), _ISA) is None


def test_no_wmma_v3_is_undecidable():
  isa = SimpleNamespace(asmCaps={"HasWMMA_V3": False}, archCaps={"VgprBank": True})
  assert X.extraVgprMatters(_state(4, 19, plr=0), isa) is None



def test_no_interleave_without_prefetch_global_read_is_the_same_kernel():
  # PGR=0 without PAP: the region cannot hold the MXSA G2L, so True and False both keep the default layout.
  assert X.extraVgprMatters(_state(4, 19, plr=0, PrefetchGlobalRead=0), _ISA) is False
  # With PAP enabled the interleave applies again and the tight layout still needs the extra VGPRs.
  assert X.extraVgprMatters(_state(4, 19, plr=0, PrefetchGlobalRead=0, PrefetchAcrossPersistent=1), _ISA) is True


def test_region_past_v255_keeps_the_default_layout():
  # Both layouts are too long for v0-v255: True and False keep the default layout, the same kernel.
  assert X.extraVgprMatters(_state(160, 160, plr=0), _ISA) is False


def test_conflict_free_layout_past_v255_differs_from_the_budget_layout():
  # The budget layout (False) fits v255 but the conflict-free one (True) does not and falls back to the
  # default layout: the two kernels differ, so True stays.
  assert X.extraVgprMatters(_state(4, 160, plr=0), _ISA) is True


def _real_region(state, numBuffer, perBlockA, perBlockB, extra):
  """KernelWriter._mxsInterleaveRegion on the same geometry (stub writer as in the plan tests)."""
  kernel = dict(state, MXScaleVgprInterleaveExtraVgpr=extra)
  states = SimpleNamespace(
    numVgprBuffer=numBuffer, lrvwTileMXSA=1, lrvwTileMXSB=1, archCaps=_ISA.archCaps,
    kernelName="stub",
    mxsa=SimpleNamespace(numVgprValu=perBlockA * numBuffer, numVgprValuPerBlock=perBlockA),
    mxsb=SimpleNamespace(numVgprValu=perBlockB * numBuffer, numVgprValuPerBlock=perBlockB))
  w = SimpleNamespace(states=states)
  w._mxsDefaultBlocks = lambda k, tc: KernelWriter._mxsDefaultBlocks(w, k, tc)
  w._alignMXSB = lambda k, idx: KernelWriter._alignMXSB(w, k, idx)
  w._mxsInterleavePlan = lambda k, a, b, **kw: KernelWriter._mxsInterleavePlan(w, k, a, b, **kw)
  w._mxsBudgetPlan = lambda *a: KernelWriter._mxsBudgetPlan(w, *a)
  w._mxsLayoutConflicts = lambda *a, **kw: KernelWriter._mxsLayoutConflicts(w, *a, **kw)
  w.do = {}
  tensor = {"localReadInstruction": SimpleNamespace(blockWidth=1)}
  pap = bool(kernel.get("PrefetchAcrossPersistent", 0))
  plan, regionLen, _ = KernelWriter._mxsInterleaveRegion(w, kernel, tensor, tensor, 0, pap)
  return None if plan is None else [(g["A"], g["B"], g["span"], sorted(g["cells"].items())) for g in plan["groups"]], regionLen


@pytest.mark.parametrize("a, b, plr, over", [
  (4, 11, 1, {}), (4, 19, 0, {}), (8, 8, 1, {}), (3, 3, 1, {}), (12, 12, 1, {}), (1, 14, 1, {}),
  (4, 19, 0, {"PrefetchGlobalRead": 0}), (4, 19, 0, {"PrefetchGlobalRead": 0, "PrefetchAcrossPersistent": 1}),
  (160, 160, 0, {}), (4, 160, 0, {}),
])
def test_it_agrees_with_the_real_region_decision(a, b, plr, over):
  # True matters exactly when KernelWriter lays the scales out differently for True and False.
  state = _state(a, b, plr, **over)
  numBuffer = plr + 1
  differs = _real_region(state, numBuffer, a, b, False) != _real_region(state, numBuffer, a, b, True)
  assert X.extraVgprMatters(state, _ISA) is differs
