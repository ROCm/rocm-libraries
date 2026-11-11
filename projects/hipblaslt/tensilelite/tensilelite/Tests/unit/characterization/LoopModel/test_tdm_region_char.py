# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Characterization tests for ``Components.LoopModel.TdmRegion``.

These are the TDM descriptor primitives with the region walk taken out, so GIR
can place each step itself. They emit rocisa instructions off writer state, so
the tests drive them through a stub writer carrying only the fields each one
reads -- the shape ``test_custom_kernel_staggeru_metadata`` established for TDM
emitters -- and assert the emitted text. The text is the contract: which SGPR
moves, in which direction, under which wave predicate.
"""

import pytest

from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SMovB32

from tensilelite.Components.LoopModel import TdmRegion as T


class _Tmp:
    def __init__(self, idx, size):
        self.idx = idx
        self.size = size


class _TmpCtx:
    """Stand-in for allocTmpSgpr, handing out a fixed high base."""

    def __init__(self, idx, size):
        self._t = _Tmp(idx, size)

    def __enter__(self):
        return self._t

    def __exit__(self, *exc):
        return False


class _States:
    def __init__(self, inTailLoop=False):
        self.staggerUCode = True
        self.unrollIdx = 0
        self.inTailLoop = inTailLoop
        self.ldsTensorTokenIdx = 7


class _Geometry:
    def __init__(self, span):
        self.dim1SpanPerRegion = span


class _Writer:
    """Enough of KernelWriterAssembly for the region primitives."""

    def __init__(self, members=("A", "B"), ranges=None, waveSeparated=True,
                 waveIdxLive=True, issuesOwnLoad=None, owners=None,
                 spans=None, inTailLoop=False):
        self.states = _States(inTailLoop)
        self._members = list(members) if members else []
        self._ranges = ranges
        self._waveSeparated = waveSeparated
        self._waveIdxLive = waveIdxLive
        self._issuesOwnLoad = issuesOwnLoad if issuesOwnLoad is not None else {}
        self._owners = owners or {}
        self._spans = spans or {}
        self.tPA = {"tensorChar": "A"}
        self.tPB = {"tensorChar": "B"}
        self.emitted = []

    def allocTmpSgpr(self, num, align=None, tag=None):
        self.emitted.append(("alloc", tag, num))
        return _TmpCtx(90, num)

    def tdmFuseGroupOf(self, kernel, tc):
        return self._members, self._ranges

    def tdmDescriptorOwners(self, kernel):
        return self._owners

    def tdmIssuesOwnLoad(self, kernel, m):
        return self._issuesOwnLoad.get(m, False)

    def isTdmWaveSeparated(self, kernel):
        return self._waveSeparated

    def isTdmWaveIdxLive(self, kernel):
        return self._waveIdxLive

    def tdmSplitGeometry(self, kernel, tP):
        return _Geometry(self._spans.get(tP["tensorChar"], 4))

    def tdmTpByChar(self, tc):
        return {"tensorChar": tc}

    @staticmethod
    def _marker(mod, text):
        """A rendered stand-in: an empty Module prints nothing, an instruction does."""
        mod.add(SMovB32(sgpr(99), 0, text))

    def _emitTdmWaveParitySCC(self, mod, kernel, idx=None, comment="parity"):
        self._marker(mod, f"parity({comment})")

    def _emitTdmWaveParitySCCAuto(self, mod, kernel, comment="parity", tmpTag=None):
        self._marker(mod, f"parityAuto({comment})")

    def _emitTdmWaveIdx(self, mod, kernel, idx=None):
        self._marker(mod, "waveIdx")
        return idx if idx is not None else "WaveIdx"

    def _tdmSplitGroupInc(self, mod, kernel, gIncIdx, scratchIdx, tc):
        self._marker(mod, f"splitGroupInc({tc})")
        return sgpr(scratchIdx)


def _kernel(**over):
    kernel = {
        "NumWaves": 1,
        "1LDSBuffer": 1,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {"Sparse": 0},
    }
    kernel.update(over)
    return kernel


def _tp(tc="A"):
    return {"tensorChar": tc}


# --- tdmGirIsIter ---------------------------------------------------------


def test_is_iter_is_false_when_no_member_asks_for_it():
    assert T.tdmGirIsIter(_Writer(), _kernel(), "A") is False


def test_is_iter_follows_the_operands_own_flag():
    assert T.tdmGirIsIter(_Writer(), _kernel(_TDMIterateModeA=True), "A") is True


def test_is_iter_is_inherited_from_an_aliased_peer():
    """One register set: either member asking puts the pair in iterate mode."""
    writer = _Writer(members=("A", "B"))
    assert T.tdmGirIsIter(writer, _kernel(_TDMIterateModeB=True), "A") is True


def test_is_iter_ignores_a_peer_outside_the_group():
    """B is unfused here, so its flag does not reach A."""
    writer = _Writer(members=("A",))
    assert T.tdmGirIsIter(writer, _kernel(_TDMIterateModeB=True), "A") is False


# --- tdmRegionIncrementGir ------------------------------------------------


@pytest.mark.parametrize("tc, sparse", [("MXSA", 0), ("A", 1)])
def test_region_increment_refuses_mx_scale_and_sparse(tc, sparse):
    kernel = _kernel(ProblemType={"Sparse": sparse})
    with pytest.raises(NotImplementedError, match="out of scope"):
        T.tdmRegionIncrementGir(_Writer(), kernel, _tp(tc))


def test_region_increment_single_wave_adds_the_persistent_split_sgprs():
    text = str(T.tdmRegionIncrementGir(_Writer(), _kernel(), _tp("A")))
    assert "s_add_u32" in text and "tdmALdsSplitIncs" in text
    assert "tdmAGlobalSplitIncs" in text
    assert "s_addc_u32" in text, "the 64-bit global address carries"
    assert "s_sub" not in text


def test_region_increment_single_wave_back_subtracts_the_same_pair():
    text = str(T.tdmRegionIncrementGir(_Writer(), _kernel(), _tp("A"), back=True))
    assert "s_sub_u32" in text and "s_subb_u32" in text
    assert "tdmALdsSplitIncs" in text and "tdmAGlobalSplitIncs" in text
    assert "s_add_u32" not in text


def test_region_increment_multi_wave_recomputes_the_step_per_parity():
    """The multi-wave step is not a persistent SGPR; it is recomputed each time."""
    writer = _Writer()
    text = str(T.tdmRegionIncrementGir(writer, _kernel(NumWaves=4), _tp("A")))
    assert "splitGroupInc(A)" in text
    assert "tdmALdsSplitIncs" not in text, "the persistent SGPRs are not used here"
    assert ("alloc", "tdmSplitIncGir", 2) in writer.emitted
    assert "s_add_u32" in text and "s_addc_u32" in text


def test_region_increment_multi_wave_back_subtracts_after_recomputing():
    writer = _Writer()
    text = str(T.tdmRegionIncrementGir(writer, _kernel(NumWaves=4), _tp("A"), back=True))
    assert "splitGroupInc(A)" in text
    assert "s_sub_u32" in text and "s_subb_u32" in text


@pytest.mark.parametrize("back, expected", [(False, "fwd"), (True, "back")])
def test_region_increment_names_its_module_by_direction(back, expected):
    mod = T.tdmRegionIncrementGir(_Writer(), _kernel(), _tp("A"), back=back)
    assert mod.name == f"GIR region {expected} A"


# --- tdmDescriptorEnableGir -----------------------------------------------


def test_descriptor_enable_is_inert_when_the_waves_are_not_separated():
    writer = _Writer(waveSeparated=False)
    assert str(T.tdmDescriptorEnableGir(writer, _kernel(), _tp("A"), True)).strip() == ""


def test_descriptor_enable_is_inert_without_a_fuse_group():
    writer = _Writer(members=())
    assert str(T.tdmDescriptorEnableGir(writer, _kernel(), _tp("A"), True)).strip() == ""


@pytest.mark.parametrize("tc, enabled, word", [("A", True, "valid"), ("B", False, "NULL")])
def test_descriptor_enable_parity_form_selects_on_the_wave_parity(tc, enabled, word):
    """Two members and no ranges: SCC carries "odd" and one cselect does it."""
    writer = _Writer(members=("A", "B"), ranges=None, owners={"A": "A", "B": "A"})
    text = str(T.tdmDescriptorEnableGir(writer, _kernel(), _tp(tc), enabled))
    assert "parityAuto" in text
    assert "s_cselect_b32" in text
    assert f"{tc} descriptor {word}" in text
    assert "s_cmp" not in text, "parity needs no compare of its own"


def test_descriptor_enable_nulls_the_register_set_the_load_reads():
    """The issuer owns the registers, which is not always the named owner."""
    writer = _Writer(members=("A", "B"), ranges=None,
                     issuesOwnLoad={"B": True}, owners={"A": "A"})
    text = str(T.tdmDescriptorEnableGir(writer, _kernel(), _tp("A"), False))
    assert "tdmBGroup0+0" in text, "B issues the load, so B's descriptor is the one nulled"


def test_descriptor_enable_range_form_at_wave_zero_compares_without_a_temp():
    writer = _Writer(members=("A", "B"), ranges=[(0, 0, 2), (1, 2, 2)], waveIdxLive=True)
    text = str(T.tdmDescriptorEnableGir(writer, _kernel(), _tp("A"), True))
    assert "s_cmp_lt_u32" in text and "WaveIdx" in text
    assert "A owns waves 0..1?" in text
    assert "s_sub_u32" not in text, "first == 0 needs no bias"


def test_descriptor_enable_range_form_biases_a_nonzero_first_wave():
    """`first <= wId < first+count` is one unsigned compare of `wId - first`."""
    writer = _Writer(members=("A", "B"), ranges=[(0, 0, 2), (1, 2, 2)], waveIdxLive=True)
    text = str(T.tdmDescriptorEnableGir(writer, _kernel(), _tp("B"), True))
    assert "wId - 2" in text
    assert "B owns waves 2..3?" in text
    assert "s_cmp_lt_u32" in text


def test_descriptor_enable_range_form_materialises_a_dead_wave_index():
    writer = _Writer(members=("A", "B"), ranges=[(0, 0, 2), (1, 2, 2)], waveIdxLive=False)
    text = str(T.tdmDescriptorEnableGir(writer, _kernel(), _tp("A"), True))
    assert "waveIdx" in text, "the index is recomputed into a temp"
    assert ("alloc", "tdmDescriptorEnableGir_waveIdTmp", 1) in writer.emitted


# --- tdmMaskShortMembers --------------------------------------------------


def test_mask_short_members_nulls_only_members_that_cannot_reach_the_region(monkeypatch):
    """A is split 2 ways and reaches region 1; B is unsplit and does not."""
    monkeypatch.setattr(T._tdm_split, "split_of",
                        lambda kernel, m: ((2, "DU") if m == "A" else (1, "DU")))
    writer = _Writer(members=("A", "B"), ranges=None, owners={"A": "A", "B": "A"})
    mod = Module("t")
    T.tdmMaskShortMembers(writer, mod, _kernel(), "A", region=1, enabled=False)
    text = str(mod)
    assert "B descriptor NULL" in text
    assert "A descriptor" not in text


def test_mask_short_members_can_name_the_issuing_member_itself(monkeypatch):
    """The walk length is the set's, so the arm issuing it can be the short one."""
    monkeypatch.setattr(T._tdm_split, "split_of", lambda kernel, m: (1, "DU"))
    writer = _Writer(members=("A",), ranges=None, owners={"A": "A"})
    mod = Module("t")
    T.tdmMaskShortMembers(writer, mod, _kernel(), "A", region=1, enabled=True)
    assert "A descriptor valid" in str(mod)


def test_mask_short_members_is_inert_at_region_zero(monkeypatch):
    """Every member reaches region 0, so nothing is masked."""
    monkeypatch.setattr(T._tdm_split, "split_of", lambda kernel, m: (2, "DU"))
    writer = _Writer(members=("A", "B"), ranges=None)
    mod = Module("t")
    T.tdmMaskShortMembers(writer, mod, _kernel(), "A", region=0, enabled=False)
    assert str(mod).strip() == ""


# --- tdmSplitDim1ForRegion ------------------------------------------------


def test_split_dim1_uses_one_span_when_the_group_is_not_shared():
    """Without A and B on one descriptor, every wave shortens by its own span."""
    writer = _Writer(members=("A",), spans={"A": 4})
    mod = Module("t")
    group1 = T.tdmSplitDim1ForRegion(writer, mod, _kernel(), _tp("A"), 90, 91, 92)
    text = str(mod)
    assert group1 == "tdmAGroup1"
    assert "halfRows" in text
    assert "parity" not in text, "one span needs no parity select"


def test_split_dim1_parity_selects_when_a_and_b_share_the_descriptor():
    writer = _Writer(members=("A", "B"), spans={"A": 4, "B": 6})
    mod = Module("t")
    T.tdmSplitDim1ForRegion(writer, mod, _kernel(), _tp("A"), 90, 91, 92)
    text = str(mod)
    assert "parity" in text
    assert "halfRows = A" in text and "halfRows = parity ? B : A" in text


def test_split_dim1_skips_the_parity_select_when_both_spans_agree():
    writer = _Writer(members=("A", "B"), spans={"A": 4, "B": 4})
    mod = Module("t")
    T.tdmSplitDim1ForRegion(writer, mod, _kernel(), _tp("A"), 90, 91, 92)
    assert "parity" not in str(mod)


def test_split_dim1_rebuilds_the_extents_and_clamps_the_second_half():
    writer = _Writer(members=("A",), spans={"A": 4})
    mod = Module("t")
    T.tdmSplitDim1ForRegion(writer, mod, _kernel(), _tp("A"), 90, 91, 92)
    text = str(mod)
    assert "H0 = dim1 lo" in text and "H0 hi << 16" in text
    assert "H0 = full dim1" in text
    assert "H1 = H0 - halfRows" in text
    assert "clamp H1 to 0" in text


def test_split_dim1_allocates_a_temp_when_the_wave_index_is_dead():
    writer = _Writer(members=("A", "B"), spans={"A": 4, "B": 6}, waveIdxLive=False)
    mod = Module("t")
    T.tdmSplitDim1ForRegion(writer, mod, _kernel(), _tp("A"), 90, 91, 92)
    assert ("alloc", "tdmSplitDim1ParityGir", 1) in writer.emitted


# --- tdmLoadRegionGir -----------------------------------------------------


def test_load_region_refuses_an_operand_whose_tdm_arm_is_off():
    kernel = _kernel(enableTDMA=False)
    with pytest.raises(NotImplementedError, match="enableTDMA is off"):
        T.tdmLoadRegionGir(_Writer(), kernel, _tp("A"), None)


def test_load_region_refuses_the_persistent_tail_bank_select(monkeypatch):
    monkeypatch.setattr(T, "isPersistent", lambda kernel: True)
    monkeypatch.setattr(T, "isStreamK", lambda kernel: False)
    writer = _Writer(inTailLoop=True)
    with pytest.raises(NotImplementedError, match="PAP bank select"):
        T.tdmLoadRegionGir(writer, _kernel(**{"1LDSBuffer": 0}), _tp("A"), None)
