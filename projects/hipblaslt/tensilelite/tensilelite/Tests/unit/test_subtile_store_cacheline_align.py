# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Store cacheline align (StoreCachelineAlign) unit tests.

A paired 16-bit subtile store fills half of each 128B line. The cacheline align blends two paired
stores that hold the two halves of the same lines so that each store writes full
lines. The blend is done in place, which makes its instruction order load-bearing, and
the staged store keeps one pair alive across scheduler units, which makes its hold
state load-bearing. Neither shows up as a build failure when broken -- only as wrong
output on a GPU -- so these tests pin them without one.
"""

import re
from types import SimpleNamespace

import pytest

from rocisa.code import Module
from rocisa.container import vgpr
from rocisa.instruction import BufferStoreB128, VPermlane16SwapB32

from tensilelite.Common.GlobalParameters import defaultBenchmarkCommonParameters
from tensilelite.Common.RequiredParameters import getRequiredParametersMin
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.Components import GlobalWriteBatch as gwb
from tensilelite.Components.GlobalWriteBatch import GlobalWriteBatchWriter, storeCachelineAlignEnabled
from tensilelite.Components.Subtile.LogicalScheduler import LogicalScheduler
from tensilelite.SolutionStructs import Naming

pytestmark = pytest.mark.unit

H_EVEN, H_ODD, ADDR, TMP = 20, 24, 28, 30


def _writer(stage=0, fused=True, stages=8, cachelineAlignTmp=TMP, ring=H_ODD, align=True):
    """A GlobalWriteBatchWriter carrying only the state the cacheline align emitters read."""
    w = object.__new__(GlobalWriteBatchWriter)
    w.kernel = {"NonTemporalD": 0, "StoreCachelineAlign": align}
    w.cvtVgprStruct = SimpleNamespace(vgprBf16Temp=H_EVEN, vgprPairPackRing=ring,
                                      vgprAddrScratch=ADDR, vgprCachelineAlignTmp=cachelineAlignTmp)
    w._permlane16Active = True
    w._subtileAbsRowAddr = False
    w.ss = SimpleNamespace(optSrdIncForRow=False)
    states = SimpleNamespace(bpeCexternal=2, bpeCexternalGSU1=2,
                             subtileStoreStageHighWater=stage, subtileStoreStages=stages,
                             subtileStoreTt0PerStage=4, subtileStagedHold=None,
                             subtileFusedFullTileStore=fused)
    w.parentWriter = SimpleNamespace(states=states, _plsinFusedNoGuards=lambda: fused)

    def phase1(addrCalc, s0, s1, prefixOffset, **kw):
        return Module("phase1")

    def phase2(addrCalc, **kw):
        m = Module("phase2")
        q = kw["vPackOverride"]
        assert kw.get("emitStore") is False, "the cacheline align must prepare without storing"
        m.add(VPermlane16SwapB32(dst=w._vPackReg(q, 0), src=w._vPackReg(q, 2), comment="swap"))
        return m

    w._emit16bitSubtilePairedStorePhase1 = phase1
    w._emit16bitSubtilePairedStorePhase2 = phase2
    return w


def _addr(offset, row=0, addrD=40):
    return SimpleNamespace(addrDVgpr=addrD, globalOffset=offset, coordOffset1=row)


def _insts(module):
    return [i for i in module.flatitems() if not type(i).__name__ == "TextBlock"]


def _text(inst):
    return str(inst).split("//")[0].strip()


def _dst(inst):
    return _text(inst).split(None, 1)[1].split(",")[0].strip()


# ---------------------------------------------------------------- parameter

def test_parameter_is_registered_and_off_by_default():
    assert validParameters["StoreCachelineAlign"] == [False, True]
    assert {"StoreCachelineAlign": [False]} in defaultBenchmarkCommonParameters
    assert "StoreCachelineAlign" in getRequiredParametersMin()
    assert storeCachelineAlignEnabled({}) is False
    assert storeCachelineAlignEnabled({"StoreCachelineAlign": True}) is True


def test_name_token_only_on_the_eligible_path(monkeypatch):
    base = {
        "ProblemType": {"DataType": 0, "GroupedGemm": False},
        "GlobalSplitU": 1,
        "WorkGroupMappingXCC": 1,
        "UseCustomMainLoopSchedule": False,
        "SpaceFillingAlgo": [],
    }
    monkeypatch.setattr(Naming, "isMxf4SubtilePath", lambda state: True)
    plain = Naming.getKernelNameMin(dict(base, ISA=(9, 5, 0)), splitGSU=True)
    off = Naming.getKernelNameMin(dict(base, ISA=(9, 5, 0), StoreCachelineAlign=False), splitGSU=True)
    on = Naming.getKernelNameMin(dict(base, ISA=(9, 5, 0), StoreCachelineAlign=True), splitGSU=True)
    assert off == plain
    assert on != plain and "_SCA1" in on
    # another ISA has its own name, and the parameter does not change it
    otherIsa = Naming.getKernelNameMin(dict(base, ISA=(9, 4, 0)), splitGSU=True)
    otherIsaOn = Naming.getKernelNameMin(dict(base, ISA=(9, 4, 0), StoreCachelineAlign=True), splitGSU=True)
    assert otherIsaOn == otherIsa and "_SCA1" not in otherIsaOn

    monkeypatch.setattr(Naming, "isMxf4SubtilePath", lambda state: False)
    notSubtile = Naming.getKernelNameMin(dict(base, ISA=(9, 5, 0), StoreCachelineAlign=True), splitGSU=True)
    assert notSubtile == plain


# ---------------------------------------------------------------- pack register helper

def test_vpackreg_physical_and_valuc():
    w = _writer()
    assert str(w._vPackReg(20, 3)) == str(vgpr(23))
    assert str(w._vPackReg(20, 0, 4)) == str(vgpr(20, 4))
    assert str(w._vPackReg(("c", 8), 2)) == str(vgpr("ValuC+10"))


# ---------------------------------------------------------------- blend

@pytest.mark.parametrize("rebuildMask", [False, True])
def test_blend_is_in_place_and_ordered(rebuildMask):
    w = _writer()
    m = Module("align")
    X = ("c", 8)
    w._emitCachelineAlignBlendAndStores(m, _addr(128), 4, False, H_EVEN, X, "ValuC+12", marker=14,
                              rebuildMask=rebuildMask)
    insts = _insts(m)
    texts = [_text(i) for i in insts]

    # Two stores: the odd n-columns from X first, then the even n-columns from H, both
    # at pair A's offset.
    stores = [i for i in insts if isinstance(i, BufferStoreB128)]
    assert len(stores) == 2
    assert _text(stores[0]).split()[1].startswith("v[vgprValuC+8")
    assert _text(stores[1]).split()[1].startswith("v[%d:" % H_EVEN)
    assert all("offset:128" in _text(s) for s in stores)

    # Per dword: A's odd columns are saved before A is overwritten, and B's even lanes are
    # read (store-1 blend) before B is overwritten (store-2 blend).
    firstStore = insts.index(stores[0])
    for k in range(4):
        a, b = str(vgpr(H_EVEN + k)), str(vgpr("ValuC+%d" % (8 + k)))
        save = next(i for i, t in enumerate(texts) if t.startswith("v_mov_b32") and ", %s " % a in t + " ")
        wrA = next(i for i, t in enumerate(texts) if t.startswith("v_cndmask_b32 %s," % a))
        wrB = next(i for i, t in enumerate(texts) if t.startswith("v_cndmask_b32 %s," % b))
        assert save < wrA < wrB < firstStore
        assert "quad_perm:[1,1,3,3]" in texts[save]
        assert "quad_perm:[0,0,2,2]" in texts[wrA]      # DPP on src0 = B
        assert texts[wrA].split(",")[1].strip() == b
        assert "quad_perm" not in texts[wrB]

    # The shared address register is only ever read.
    assert all(_dst(i) != str(vgpr(ADDR)) for i in insts if not isinstance(i, BufferStoreB128))

    # VCC is written only by the mask compare: once, or twice when the second store is
    # its own scheduler unit.
    assert sum(t.startswith("v_cmp_eq_u32 vcc") for t in texts) == (2 if rebuildMask else 1)
    secondCmp = [i for i, t in enumerate(texts) if t.startswith("v_cmp_eq_u32 vcc")][-1]
    assert (secondCmp > firstStore) == rebuildMask


# ---------------------------------------------------------------- staged store

def test_staged_line_start_is_held_without_a_store():
    w = _writer(stage=0)
    m = w._emitStagedCachelineAlignPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    insts = _insts(m)
    assert not any(isinstance(i, BufferStoreB128) for i in insts)
    assert GlobalWriteBatchWriter.STAGED_UNIT_CUT in insts[-1].comment
    hold = w.parentWriter.states.subtileStagedHold
    assert hold["tt0"] == 0 and hold["H"] == H_EVEN and hold["stage"] == 0


def test_staged_partner_cacheline_aligns_and_clears_the_hold():
    w = _writer(stage=0)
    w._emitStagedCachelineAlignPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    m = w._emitStagedCachelineAlignPair(_addr(64), 24, 28, 0, 2, 2, 3, False)
    stores = [i for i in _insts(m) if isinstance(i, BufferStoreB128)]
    assert len(stores) == 2
    assert all("offset:0" in _text(s) or "offset" not in _text(s) for s in stores)
    assert w.parentWriter.states.subtileStagedHold is None
    # pair B was packed into its own ValuC registers, not into a pack quad
    assert _text(stores[0]).split()[1].startswith("v[vgprValuC+24")


@pytest.mark.parametrize("stage, expected", [(0, H_EVEN), (1, H_ODD), (2, H_EVEN), (5, H_ODD)])
def test_staged_hold_quad_is_private_to_the_stage_parity(stage, expected):
    w = _writer(stage=stage)
    w._emitStagedCachelineAlignPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    assert w.parentWriter.states.subtileStagedHold["H"] == expected


@pytest.mark.parametrize("kw", [
    dict(offset=128),            # not 64B above the held pair
    dict(blockIdxN=4),           # another N-group
    dict(addrD=41),              # another address register
    dict(row=32),                # another D row
])
def test_staged_non_partner_flushes_the_held_pair(kw):
    w = _writer(stage=0)
    w._emitStagedCachelineAlignPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    addr = _addr(kw.get("offset", 64), row=kw.get("row", 0), addrD=kw.get("addrD", 40))
    m = w._emitStagedCachelineAlignPair(addr, 24, 28, 0, 2, 2, kw.get("blockIdxN", 3), False)
    texts = [_text(i) for i in _insts(m)]
    assert not any("quad_perm" in t for t in texts), "a non-partner must not be blended"
    stores = [i for i in _insts(m) if isinstance(i, BufferStoreB128)]
    # the held pair goes out unblended, then this pair on its own
    assert len(stores) == 2
    assert _text(stores[0]).split()[1].startswith("v[%d:" % H_EVEN)
    assert w.parentWriter.states.subtileStagedHold is None


def test_staged_pair_without_a_partner_in_its_stage_is_stored_directly():
    w = _writer(stage=0)
    # tt0 = 2 is the second pair of a line; with nothing held it cannot start one
    m = w._emitStagedCachelineAlignPair(_addr(64), 24, 28, 0, 2, 2, 3, False)
    stores = [i for i in _insts(m) if isinstance(i, BufferStoreB128)]
    assert len(stores) == 1
    assert w.parentWriter.states.subtileStagedHold is None


@pytest.mark.parametrize("change, active", [
    ({}, True),
    ({"align": False}, False),
    ({"stages": 0}, False),
    ({"fused": False}, False),
    ({"cachelineAlignTmp": -1}, False),
    ({"ring": -1}, False),
])
def test_staged_cacheline_align_gate(change, active):
    assert _writer(**change)._stagedCachelineAlignActive() is active


# ---------------------------------------------------------------- plain store

@pytest.mark.parametrize("change, active", [
    ({"fused": False}, True),
    ({"fused": True}, False),                 # the fused store never uses the interior body
    ({"fused": False, "align": False}, False),
    ({"fused": False, "cachelineAlignTmp": -1}, False),
])
def test_plain_cacheline_align_gate(change, active):
    w = _writer(**change)
    assert w._plainCachelineAlignActive() is active
    w._permlane16Active = False               # e.g. MT320x256's plain path
    assert w._plainCachelineAlignActive() is False


def test_plain_cacheline_align_emits_two_full_line_stores_and_a_two_wait_state_fence():
    w = _writer(fused=False)
    m = w._emitPlainCachelineAlignPair(_addr(0), 16, 20, _addr(64), 24, 28, 0, 0, 3, False)
    insts = _insts(m)
    stores = [i for i in insts if isinstance(i, BufferStoreB128)]
    assert len(stores) == 2
    assert _text(stores[1]).split()[1].startswith("v[%d:" % H_EVEN)
    # the next line's first convert overwrites H: gfx940+ needs 2 wait states (s_nop 1)
    assert re.fullmatch(r"s_nop\s+1", _text(insts[-1]))


# ---------------------------------------------------------------- per-wave line start

@pytest.mark.parametrize("miwt0, miwg0, selector", [
    (4, 2, None), (8, 2, None), (12, 2, None), (16, 2, None),   # whole lines: every wave starts one
    (2, 2, None),                                               # one pair per wave: nothing to align
    (6, 2, "wave"), (10, 2, "wave"), (6, 4, "wave"), (14, 2, "wave"),
    (6, 1, "tile"), (10, 1, "tile"), (18, 1, "tile"),
    (6, 3, None),                                               # not derivable from one bit
])
def test_line_start_selector(miwt0, miwg0, selector):
    w = _writer(fused=False)
    w.kernel.update(MIWaveTile=[miwt0, 4], MIWaveGroup=[miwg0, 4])
    assert w._cachelineAlignLineStartSelector() == selector


def _interiorWriter(miwt0, miwg0, align=True):
    w = _writer(fused=False, align=align)
    w.kernel.update(MIWaveTile=[miwt0, 4], MIWaveGroup=[miwg0, 4], WavefrontSize=64)
    w.parentWriter.states.subtilePairPackSlot = 5
    names = iter(range(100))
    w.parentWriter.labels = SimpleNamespace(getNameInc=lambda n: "%s_%d" % (n, next(names)))
    w._epilogScratchSgpr = lambda n: 40
    w._epilogScratchFree = lambda r: None
    built = []

    def body(forceSlc, cachelineAlignParity):
        built.append((cachelineAlignParity, w.parentWriter.states.subtilePairPackSlot))
        w.parentWriter.states.subtilePairPackSlot += 3      # a body advances the ring
        return Module("body%d" % cachelineAlignParity)

    w._buildSubtileInteriorStoreBody = body
    return w, built


@pytest.mark.parametrize("miwt0, miwg0, test", [
    (10, 2, r"s_bitcmp1_b32\s+s40,\s*6"),               # wave position in M = Serial bit 6
    (18, 1, r"s_bitcmp1_b32\s+s\[sgprWorkGroup0\],\s*0"),
])
def test_interior_body_is_emitted_per_line_start(miwt0, miwg0, test):
    w, built = _interiorWriter(miwt0, miwg0)
    m = w._buildSubtileInteriorStores()
    # both alignments, each built from the same pack-ring state
    assert built == [(0, 5), (1, 5)]
    text = [_text(i) for i in m.flatitems()]
    assert any(re.fullmatch(test, t) for t in text), text
    order = [t for t in text if t.startswith(("s_cbranch", "s_branch", "label_"))]
    assert len(order) == 4
    assert order[0].startswith("s_cbranch_scc1") and order[0].split()[1] == order[2].rstrip(":")
    assert order[1].startswith("s_branch") and order[1].split()[1] == order[3].rstrip(":")
    names = [x.name for x in m.items() if isinstance(x, Module)]
    assert names == ["body0", "body1"]


@pytest.mark.parametrize("miwt0, align", [(8, True), (10, False)])
def test_interior_body_is_emitted_once_when_every_wave_starts_a_line(miwt0, align):
    w, built = _interiorWriter(miwt0, 2, align=align)
    m = w._buildSubtileInteriorStores()
    assert built == [(0, 5)] and m.name == "body0"


# ---------------------------------------------------------------- MT320x256 serial store

def test_cacheline_align_puts_the_mt320x256_serial_store_on_permlane16_with_a_wider_fence(monkeypatch):
    monkeypatch.setattr(gwb, "isMxf4SubtilePath", lambda kernel: True)
    k = {"ISA": (9, 5, 0), "MatrixInstM": 16, "MacroTile0": 320, "MacroTile1": 256}
    on = dict(k, StoreCachelineAlign=True)
    other = dict(on, MacroTile0=256, MacroTile1=320)
    # develop: serial store keeps ds_bpermute, woven store uses permlane16, fence is 1
    assert gwb.plsinStorePermlane16Active(k, None) is False
    assert gwb.plsinStorePermlane16Active(k, [1]) is True
    assert gwb.serialStoreFenceWaitStates(k, None) == 1
    # cacheline align on: serial store too, behind two wait states; the woven store is unchanged
    assert gwb.plsinStorePermlane16Active(on, None) is True
    assert gwb.serialStoreFenceWaitStates(on, None) == 2
    assert gwb.serialStoreFenceWaitStates(on, [1]) == 1
    assert gwb.serialStoreFenceWaitStates(other, None) == 1


# ---------------------------------------------------------------- scheduler units

def test_stage_drain_units_end_at_a_store_or_at_the_cut_tag():
    stage = Module("stage")
    held = VPermlane16SwapB32(dst=vgpr(20), src=vgpr(22),
                              comment="swap " + GlobalWriteBatchWriter.STAGED_UNIT_CUT)
    plain = VPermlane16SwapB32(dst=vgpr(24), src=vgpr(26), comment="swap")
    w = _writer()
    stage.add(held)
    stage.add(plain)
    w._cachelineAlignStoreNow(stage, _addr(0), 0, False, 24, ADDR, "a")
    w._cachelineAlignStoreNow(stage, _addr(0), 0, False, 20, ADDR, "b")
    units = LogicalScheduler._plsinStageDrainUnits(stage)
    assert [len(u) for u in units] == [1, 2, 1]
    assert units[0][0] is held
