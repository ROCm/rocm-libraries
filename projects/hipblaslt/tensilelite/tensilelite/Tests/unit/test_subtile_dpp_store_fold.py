# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DPP store fold (DPPStoreFold) unit tests.

A paired 16-bit subtile store fills half of each 128B line. The fold blends two paired
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
from rocisa.instruction import BufferStoreB128, SNop, VPermlane16SwapB32

from tensilelite.Common.GlobalParameters import defaultBenchmarkCommonParameters
from tensilelite.Common.RequiredParameters import getRequiredParametersMin
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.Components import GlobalWriteBatch as gwb
from tensilelite.Components.GlobalWriteBatch import GlobalWriteBatchWriter, storeFoldEnabled
from tensilelite.Components.Subtile.LogicalScheduler import LogicalScheduler
from tensilelite.SolutionStructs import Naming

pytestmark = pytest.mark.unit

H_EVEN, H_ODD, ADDR, TMP = 20, 24, 28, 30


def _writer(stage=0, fused=True, stages=8, foldTmp=TMP, ring=H_ODD, fold=True):
    """A GlobalWriteBatchWriter carrying only the state the fold emitters read."""
    w = object.__new__(GlobalWriteBatchWriter)
    w.kernel = {"NonTemporalD": 0, "DPPStoreFold": fold}
    w.cvtVgprStruct = SimpleNamespace(vgprBf16Temp=H_EVEN, vgprPairPackRing=ring,
                                      vgprAddrScratch=ADDR, vgprFoldTmp=foldTmp)
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
        assert kw.get("emitStore") is False, "the fold must prepare without storing"
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
    assert validParameters["DPPStoreFold"] == [False, True]
    assert {"DPPStoreFold": [False]} in defaultBenchmarkCommonParameters
    assert "DPPStoreFold" in getRequiredParametersMin()
    assert storeFoldEnabled({}) is False
    assert storeFoldEnabled({"DPPStoreFold": True}) is True


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
    off = Naming.getKernelNameMin(dict(base, ISA=(9, 5, 0), DPPStoreFold=False), splitGSU=True)
    on = Naming.getKernelNameMin(dict(base, ISA=(9, 5, 0), DPPStoreFold=True), splitGSU=True)
    assert off == plain
    assert on != plain and "DPPSF" in on
    # another ISA has its own name, and the parameter does not change it
    otherIsa = Naming.getKernelNameMin(dict(base, ISA=(9, 4, 0)), splitGSU=True)
    otherIsaOn = Naming.getKernelNameMin(dict(base, ISA=(9, 4, 0), DPPStoreFold=True), splitGSU=True)
    assert otherIsaOn == otherIsa and "DPPSF" not in otherIsaOn

    monkeypatch.setattr(Naming, "isMxf4SubtilePath", lambda state: False)
    notSubtile = Naming.getKernelNameMin(dict(base, ISA=(9, 5, 0), DPPStoreFold=True), splitGSU=True)
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
    m = Module("fold")
    X = ("c", 8)
    w._emitFoldBlendAndStores(m, _addr(128), 4, False, H_EVEN, X, "ValuC+12", marker=14,
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
    m = w._emitStagedFoldPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    insts = _insts(m)
    assert not any(isinstance(i, BufferStoreB128) for i in insts)
    assert GlobalWriteBatchWriter.STAGED_UNIT_CUT in insts[-1].comment
    hold = w.parentWriter.states.subtileStagedHold
    assert hold["tt0"] == 0 and hold["H"] == H_EVEN and hold["stage"] == 0


def test_staged_partner_folds_and_clears_the_hold():
    w = _writer(stage=0)
    w._emitStagedFoldPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    m = w._emitStagedFoldPair(_addr(64), 24, 28, 0, 2, 2, 3, False)
    stores = [i for i in _insts(m) if isinstance(i, BufferStoreB128)]
    assert len(stores) == 2
    assert all("offset:0" in _text(s) or "offset" not in _text(s) for s in stores)
    assert w.parentWriter.states.subtileStagedHold is None
    # pair B was packed into its own ValuC registers, not into a pack quad
    assert _text(stores[0]).split()[1].startswith("v[vgprValuC+24")


@pytest.mark.parametrize("stage, expected", [(0, H_EVEN), (1, H_ODD), (2, H_EVEN), (5, H_ODD)])
def test_staged_hold_quad_is_private_to_the_stage_parity(stage, expected):
    w = _writer(stage=stage)
    w._emitStagedFoldPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    assert w.parentWriter.states.subtileStagedHold["H"] == expected


@pytest.mark.parametrize("kw", [
    dict(offset=128),            # not 64B above the held pair
    dict(blockIdxN=4),           # another N-group
    dict(addrD=41),              # another address register
    dict(row=32),                # another D row
])
def test_staged_non_partner_flushes_the_held_pair(kw):
    w = _writer(stage=0)
    w._emitStagedFoldPair(_addr(0), 16, 20, 0, 0, 0, 3, False)
    addr = _addr(kw.get("offset", 64), row=kw.get("row", 0), addrD=kw.get("addrD", 40))
    m = w._emitStagedFoldPair(addr, 24, 28, 0, 2, 2, kw.get("blockIdxN", 3), False)
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
    m = w._emitStagedFoldPair(_addr(64), 24, 28, 0, 2, 2, 3, False)
    stores = [i for i in _insts(m) if isinstance(i, BufferStoreB128)]
    assert len(stores) == 1
    assert w.parentWriter.states.subtileStagedHold is None


@pytest.mark.parametrize("change, active", [
    ({}, True),
    ({"fold": False}, False),
    ({"stages": 0}, False),
    ({"fused": False}, False),
    ({"foldTmp": -1}, False),
    ({"ring": -1}, False),
])
def test_staged_fold_gate(change, active):
    assert _writer(**change)._stagedFoldActive() is active


# ---------------------------------------------------------------- plain store

@pytest.mark.parametrize("change, active", [
    ({"fused": False}, True),
    ({"fused": True}, False),                 # the fused store never uses the interior body
    ({"fused": False, "fold": False}, False),
    ({"fused": False, "foldTmp": -1}, False),
])
def test_plain_fold_gate(change, active):
    w = _writer(**change)
    assert w._plainFoldActive() is active
    w._permlane16Active = False               # e.g. MT320x256's plain path
    assert w._plainFoldActive() is False


def test_plain_fold_emits_two_full_line_stores_and_a_two_wait_state_fence():
    w = _writer(fused=False)
    m = w._emitPlainFoldPair(_addr(0), 16, 20, _addr(64), 24, 28, 0, 0, 3, False)
    insts = _insts(m)
    stores = [i for i in insts if isinstance(i, BufferStoreB128)]
    assert len(stores) == 2
    assert _text(stores[1]).split()[1].startswith("v[%d:" % H_EVEN)
    # the next line's first convert overwrites H: gfx940+ needs 2 wait states (s_nop 1)
    assert re.fullmatch(r"s_nop\s+1", _text(insts[-1]))


# ---------------------------------------------------------------- per-wave line start

@pytest.mark.parametrize("miwt0, miwg0, selector", [
    (4, 2, None), (8, 2, None), (12, 2, None), (16, 2, None),   # whole lines: every wave starts one
    (2, 2, None),                                               # one pair per wave: nothing to fold
    (6, 2, "wave"), (10, 2, "wave"), (6, 4, "wave"), (14, 2, "wave"),
    (6, 1, "tile"), (10, 1, "tile"), (18, 1, "tile"),
    (6, 3, None),                                               # not derivable from one bit
])
def test_line_start_selector(miwt0, miwg0, selector):
    w = _writer(fused=False)
    w.kernel.update(MIWaveTile=[miwt0, 4], MIWaveGroup=[miwg0, 4])
    assert w._foldLineStartSelector() == selector


def _interiorWriter(miwt0, miwg0, fold=True):
    w = _writer(fused=False, fold=fold)
    w.kernel.update(MIWaveTile=[miwt0, 4], MIWaveGroup=[miwg0, 4], WavefrontSize=64)
    w.parentWriter.states.subtilePairPackSlot = 5
    names = iter(range(100))
    w.parentWriter.labels = SimpleNamespace(getNameInc=lambda n: "%s_%d" % (n, next(names)))
    w._epilogScratchSgpr = lambda n: 40
    w._epilogScratchFree = lambda r: None
    built = []

    def body(forceSlc, foldParity):
        built.append((foldParity, w.parentWriter.states.subtilePairPackSlot))
        w.parentWriter.states.subtilePairPackSlot += 3      # a body advances the ring
        return Module("body%d" % foldParity)

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


@pytest.mark.parametrize("miwt0, fold", [(8, True), (10, False)])
def test_interior_body_is_emitted_once_when_every_wave_starts_a_line(miwt0, fold):
    w, built = _interiorWriter(miwt0, 2, fold=fold)
    m = w._buildSubtileInteriorStores()
    assert built == [(0, 5)] and m.name == "body0"


# ---------------------------------------------------------------- MT320x256 serial store

def test_fold_puts_the_mt320x256_serial_store_on_permlane16_with_a_wider_fence(monkeypatch):
    monkeypatch.setattr(gwb, "isMxf4SubtilePath", lambda kernel: True)
    k = {"ISA": (9, 5, 0), "MatrixInstM": 16, "MacroTile0": 320, "MacroTile1": 256}
    on = dict(k, DPPStoreFold=True)
    other = dict(on, MacroTile0=256, MacroTile1=320)
    # develop: serial store keeps ds_bpermute, woven store uses permlane16, fence is 1
    assert gwb.plsinStorePermlane16Active(k, None) is False
    assert gwb.plsinStorePermlane16Active(k, [1]) is True
    assert gwb.serialStoreFenceWaitStates(k, None) == 1
    # fold on: serial store too, behind two wait states; the woven store is unchanged
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
    w._foldStoreNow(stage, _addr(0), 0, False, 24, ADDR, "a")
    w._foldStoreNow(stage, _addr(0), 0, False, 20, ADDR, "b")
    units = LogicalScheduler._plsinStageDrainUnits(stage)
    assert [len(u) for u in units] == [1, 2, 1]
    assert units[0][0] is held


# ================================================================ woven store fold
#
# The woven fused store defers each pair's buffer_store by one pair so the held pack
# quad is still live when the next pair is prepared, then DPP-blends the two into two
# full-line stores. None of this shows up as a build failure -- a mis-fold writes the
# wrong bytes, a dropped deferral drops a store -- so these pin the pairing predicate,
# the blend/store structure, the deferred-store commit, and the address rotation.

WA, WB, QA, QB, SA, SB = 20, 24, 40, 44, 28, 34   # bf16 temp, ring, pack quad A/B, addr slot A/B


def _wovenWriter(fold=True, foldTmp=TMP, weave=True, fulltile=True, perm16=True,
                 pairPackSlot=0, ring=WB, addrRing=SB):
    """A GlobalWriteBatchWriter carrying only the state the woven-fold emitters read."""
    w = object.__new__(GlobalWriteBatchWriter)
    w.kernel = {"NonTemporalD": 0, "DPPStoreFold": fold, "WavefrontSize": 64}
    w.cvtVgprStruct = SimpleNamespace(vgprBf16Temp=WA, vgprPairPackRing=ring,
                                      vgprAddrScratch=SA, vgprPairAddrRing=addrRing,
                                      vgprFoldTmp=foldTmp, numPairPackQuads=2)
    w._permlane16Active = perm16
    w._weaveMode = weave
    states = SimpleNamespace(bpeCexternal=2, bpeCexternalGSU1=2, laneSGPRCount=2,
                             subtilePendingPhase2b=None, subtileDeferredSrdDIncs=[],
                             subtilePairPackSlot=pairPackSlot, subtileWovenPackPlan=None)
    w.parentWriter = SimpleNamespace(states=states)
    w._fusedFullTileNoGuards = lambda: fulltile
    w._epilogScratchSgpr = lambda n=1: 60
    w._epilogScratchFree = lambda r: None
    w._epilogueNtd = lambda: 0
    w._subtileStoreSoffset = lambda module, addrCalc: (0, None)
    return w


def _pend(tt0=0, idx=0, q=QA, a=SA, offset=0, blockIdxN=3, addrD=50, folded=False):
    return (_addr(offset, addrD=addrD), tt0, False, q, a, idx,
            {"blockIdxN": blockIdxN, "folded": folded})


def _partners():
    # A holds line-start (tt0 0,1 -> pair-in-column 0 = line start), B is the next pair
    # (tt0 2,3) of the same N-group, its 128B line 64 bytes along, distinct quad + slot.
    return _pend(tt0=0, idx=0, q=QA, a=SA, offset=0), _pend(tt0=2, idx=1, q=QB, a=SB, offset=64)


# ---------------------------------------------------------------- pairing predicate

def test_fold_partners_accepts_the_two_halves_of_a_line():
    w = _wovenWriter()
    assert w._foldPartners(*_partners()) is True


@pytest.mark.parametrize("mutate", [
    "foldOff",                                                    # fold off
    "noFoldTmp",                                                   # no fold temp register
    "aFolded",                                                     # A already folded
    "wrongParity",                                                 # A is not a line start
    "notAdjacentTt0",                                              # B is not tt0A+2
    "notAdjacentIdx",                                              # B is not the next pair
    "otherNColumn",                                               # different N block
    "otherAddrD",                                                  # different row-address reg
    "pendingSrdInc",                                               # an SrdD advance is queued
    "overlapQuad",                                                 # pack quads overlap
    "sameAddrSlot",                                                # same address slot
    "offsetNot64",                                                 # lines are not 64B apart
    "leasedQuad",                                                  # a non-int (ValuC) lease
])
def test_fold_partners_rejects_non_partners(mutate):
    w = _wovenWriter()
    a, b = _partners()
    if mutate == "foldOff":
        w.kernel["DPPStoreFold"] = False
    elif mutate == "noFoldTmp":
        w.cvtVgprStruct.vgprFoldTmp = -1
    elif mutate == "aFolded":
        a[6]["folded"] = True
    elif mutate == "wrongParity":
        a = _pend(tt0=2, idx=0, q=QA, a=SA, offset=0); b = _pend(tt0=4, idx=1, q=QB, a=SB, offset=64)
    elif mutate == "notAdjacentTt0":
        b = _pend(tt0=4, idx=1, q=QB, a=SB, offset=64)
    elif mutate == "notAdjacentIdx":
        b = _pend(tt0=2, idx=2, q=QB, a=SB, offset=64)
    elif mutate == "otherNColumn":
        b[6]["blockIdxN"] = 4
    elif mutate == "otherAddrD":
        b = _pend(tt0=2, idx=1, q=QB, a=SB, offset=64, addrD=51)
    elif mutate == "pendingSrdInc":
        w._deferredSrdDIncs().append(SNop(waitState=0))
    elif mutate == "overlapQuad":
        b = _pend(tt0=2, idx=1, q=QA + 2, a=SB, offset=64)
    elif mutate == "sameAddrSlot":
        b = _pend(tt0=2, idx=1, q=QB, a=SA, offset=64)
    elif mutate == "offsetNot64":
        b = _pend(tt0=2, idx=1, q=QB, a=SB, offset=96)
    elif mutate == "leasedQuad":
        a = _pend(tt0=0, idx=0, q=("c", 8), a=SA, offset=0)
    assert w._foldPartners(a, b) is False


def test_fold_partners_needs_both_pairs():
    w = _wovenWriter()
    a, b = _partners()
    assert w._foldPartners(None, b) is False
    assert w._foldPartners(a, None) is False


# ---------------------------------------------------------------- blend + stores

def test_stagger_fold_emits_one_store_and_leaves_the_partner_pending():
    w = _wovenWriter()
    a, b = _partners()
    m = w._emitStaggerFold(a, b)
    insts = _insts(m)
    texts = [_text(i) for i in insts]

    # Cadence stays one buffer_store per pair: store 1 now (A's quad), store 2 pending.
    stores = [i for i in insts if isinstance(i, BufferStoreB128)]
    assert len(stores) == 1
    assert _text(stores[0]).split()[1].startswith("v[%d:" % QA)        # even n-columns from A

    # The partner is the still-pending store: B's quad and B's address slot, at A's
    # address/offset, now flagged folded so it cannot be folded again.
    pend = w._pendingPhase2b
    assert pend is not None
    _ac, _tt0, _f, _q, _aslot, _idx, _meta = pend
    assert (_q, _aslot, _idx) == (QB, SB, 1)
    assert _ac is a[0] and _tt0 == 0 and _meta["folded"] is True

    # In-place optimised blend form: store-1 reads B through a DPP quad_perm, store-2
    # reads A through the other.
    assert any("quad_perm:[0,0,2,2]" in t for t in texts)
    assert any("quad_perm:[1,1,3,3]" in t for t in texts)


def test_stagger_fold_blend_order_is_safe_per_dword():
    w = _wovenWriter()
    a, b = _partners()
    insts = _insts(w._emitStaggerFold(a, b))
    texts = [_text(i) for i in insts]
    firstStore = next(i for i, x in enumerate(insts) if isinstance(x, BufferStoreB128))
    for k in range(4):
        ra, rb = str(vgpr(QA + k)), str(vgpr(QB + k))
        save = next(i for i, t in enumerate(texts) if t.startswith("v_mov_b32") and ", %s " % ra in t + " ")
        wrA = next(i for i, t in enumerate(texts) if t.startswith("v_cndmask_b32 %s," % ra))
        wrB = next(i for i, t in enumerate(texts) if t.startswith("v_cndmask_b32 %s," % rb))
        # A's odd columns are saved before A is overwritten; B is read (store-1) before
        # it is overwritten (store-2); both blends precede the store that reads them.
        assert save < wrA < wrB < firstStore


# ---------------------------------------------------------------- deferred-store commit

def test_commit_pending_flushes_the_store_then_the_held_row_increments():
    w = _wovenWriter()
    w._pendingPhase2b = _pend(tt0=4, idx=2, q=QB, a=SB, offset=128)
    inc = SNop(waitState=5, comment="held SrdD advance")
    w._deferredSrdDIncs().append(inc)
    insts = _insts(w._commitPendingPhase2b())
    stores = [i for i, x in enumerate(insts) if isinstance(x, BufferStoreB128)]
    assert len(stores) == 1
    # the row advance is emitted AFTER the store that still used the old SrdD cursor
    assert insts.index(inc) > stores[0]
    assert w._pendingPhase2b is None
    assert w._deferredSrdDIncs() == []


def test_commit_with_explicit_pending_leaves_the_writers_pending_alone():
    w = _wovenWriter()
    held = _pend(tt0=0, idx=0, q=QA, a=SA)
    w._pendingPhase2b = _pend(tt0=2, idx=1, q=QB, a=SB, offset=64)   # the current pair
    m = w._commitPendingPhase2b(held)                                # flush-last of the previous
    assert sum(isinstance(i, BufferStoreB128) for i in _insts(m)) == 1
    assert w._pendingPhase2b is not None and w._pendingPhase2b[5] == 1   # untouched


def test_commit_with_nothing_pending_is_empty():
    w = _wovenWriter()
    assert not any(isinstance(i, BufferStoreB128) for i in _insts(w._commitPendingPhase2b()))


# ---------------------------------------------------------------- address rotation

@pytest.mark.parametrize("slot, expected", [(0, SA), (1, SB), (2, SA), (3, SB)])
def test_pair_address_slot_rotates_two_deep(slot, expected):
    w = _wovenWriter(pairPackSlot=slot)
    assert w._pairAddrSlot(slot) == expected


def test_pair_address_slot_is_scratch_without_the_ring():
    w = _wovenWriter()
    w.cvtVgprStruct.vgprPairAddrRing = -1
    assert w._pairAddrSlot(1) == SA


# ---------------------------------------------------------------- conflict + gate

def test_stagger_conflicts_on_shared_pack_or_address():
    w = _wovenWriter(pairPackSlot=0)                 # next pair: pack=WA(20), addr slot0=SA(28)
    assert w._staggerConflicts(_pend(q=QA, a=QB), 1) is False       # distinct pack + distinct addr
    assert w._staggerConflicts(_pend(q=WA + 1, a=QB), 1) is True    # pack quads overlap
    assert w._staggerConflicts(_pend(q=QA, a=SA), 1) is True        # same address slot
    assert w._staggerConflicts(_pend(q=("c", 8), a=QB), 1) is True  # non-int (leased) held quad


@pytest.mark.parametrize("change, active", [
    ({}, True),
    ({"fold": False}, False),
    ({"weave": False}, False),
    ({"fulltile": False}, False),
    ({"perm16": False}, False),
])
def test_stagger_store_gate(change, active):
    assert _wovenWriter(**change)._staggerStoreActive() is active
