# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the TLU=1 global-read offset helpers in SubtileGREmit.

These helpers decide how a fetch group is spread across waves: which axis a wave
indexes on, whether it shares a strip, and how a strip column is cut along K.
The geometries that turn each split on are selected by Solution.py from the tile
shape, so a curated logic file only ever reaches one combination at a time.
Driving the helpers directly lets every branch be stated as a case.

Assertions are on the emitted instruction stream and the returned flags, not on
numerical results: correctness of the addresses is established by the gfx950
suites, and what these pin is which branch a geometry takes and that the
register bookkeeping balances.
"""

from types import SimpleNamespace

import pytest

from rocisa.code import Module
from rocisa.instruction import VAddU32, VAndB32, VLShiftRightB32, VMovB32, VMulLOU32

from Tensile.Components.Subtile.SubtileGREmit import (
    _tluWaveAxisId,
    _tluOtherAxisId,
    _tluCoopWaveId,
    _tluStripIdx,
    _tluKWaveSlots,
    _tluKSliceTerms,
    _tluKSliceGlobalOffset,
    _tluWaveAxisGlobalOffset,
    _grDTLAddKSlice,
)

pytestmark = pytest.mark.unit


class _Pool:
    """Register pool stub that hands out distinct indices and tracks balance."""

    def __init__(self, base=0):
        self._next = base
        self.outstanding = set()

    def checkOut(self, n, tag=None, preventOverflow=None, align=None):
        idx = self._next
        self._next += n
        self.outstanding.add(idx)
        return idx

    def checkIn(self, v):
        self.outstanding.discard(v)


class _Writer:
    def __init__(self):
        self.vgprPool = _Pool(0)
        self.sgprPool = _Pool(100)

    def strideRef(self, tc, idx):
        from rocisa.container import sgpr
        return sgpr("Stride%s%s" % (tc, idx))


def _kernel(miWaveGroup=(2, 2), wavefrontSize=64):
    return {
        "WavefrontSize": wavefrontSize,
        "MIWaveGroup": list(miWaveGroup),
        "ProblemType": {"IndexUnroll": 2},
    }


def _tile(tc="A", **kw):
    """TileInfo stand-in carrying only the fields these helpers read."""
    base = dict(
        tc=tc,
        bpe=0.5,
        waveSize=64,
        mmaTileShape=(16, 128),
        subtileShape=(16, 1),
        numGRPerSubtile=4,
        globalSubtileGrid=(1, 1),
        localSubtileGrid=(1, 1),
        grWavesPerStrip=1,
        grCoopWaves=1,
        grKSplit=1,
        grKWindowSplit=1,
        grOtherAxisWaves=1,
        grWindowsPerWave=1,
        subtileSize=2048,
        gr=SimpleNamespace(config=SimpleNamespace(loadWidth=16)),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _count(module, cls):
    """How many of one logical instruction a module emitted.

    Assert on the instruction type, not the rendered text: the spelling is
    backend- and ISA-specific (native rocisa renders VAddU32 as v_add_u32 on
    gfx9, the stinkytofu adaptor always writes v_add_nc_u32), while the logical
    instruction is what these helpers are contracted to emit.
    """
    return sum(1 for i in module.items() if isinstance(i, cls))


def _has(module, cls):
    return _count(module, cls) > 0


def _text(module):
    """Rendered text, for checks on operands rather than mnemonics.

    Register names are stable across backends, so matching them is safe in a
    way that matching an instruction's spelling is not.
    """
    return str(module)


# --- axis id ---------------------------------------------------------------

def test_wave_axis_id_single_wave_axis_is_a_constant_zero():
    """One wave on the axis: no id to compute, and the caller is told so."""
    module = Module()
    ok = _tluWaveAxisId(_Writer(), _kernel(miWaveGroup=(1, 1)), module, "A", 0)
    assert ok is False
    assert _has(module, VMovB32)


def test_wave_axis_id_for_a_masks_and_for_b_shifts():
    """A splits along M (waveId % mWaves); B along N (waveId / mWaves)."""
    kernel = _kernel(miWaveGroup=(2, 2))
    modA, modB = Module(), Module()
    assert _tluWaveAxisId(_Writer(), kernel, modA, "A", 0) is True
    assert _tluWaveAxisId(_Writer(), kernel, modB, "B", 0) is True
    assert _has(modA, VAndB32)
    # B divides instead of masking, so it must not reach for the mask.
    assert not _has(modB, VAndB32)
    assert _count(modB, VLShiftRightB32) == 2


def test_other_axis_id_is_the_mirror_of_the_axis_id():
    """The 'other' axis is whichever one the operand does not depend on."""
    kernel = _kernel(miWaveGroup=(2, 2))
    modA, modB = Module(), Module()
    _tluOtherAxisId(_Writer(), kernel, modA, "A", 0)
    _tluOtherAxisId(_Writer(), kernel, modB, "B", 0)
    assert not _has(modA, VAndB32)   # A divides here
    assert _has(modB, VAndB32)       # B masks here


# --- cooperative fetch index ----------------------------------------------

def test_coop_wave_id_is_zero_when_one_wave_fetches():
    module = Module()
    ok = _tluCoopWaveId(_Writer(), _kernel(), module, _tile(grCoopWaves=1), 0)
    assert ok is False
    assert _has(module, VMovB32)


def test_coop_wave_id_wraps_at_the_strip_when_waves_share_one():
    """With no K split the index is the axis id, wrapped to the strip width."""
    ti = _tile(grCoopWaves=2, grWavesPerStrip=2)
    module = Module()
    ok = _tluCoopWaveId(_Writer(), _kernel(miWaveGroup=(2, 2)), module, ti, 0)
    assert ok is True
    assert _has(module, VAndB32)


def test_coop_wave_id_combines_axis_and_other_when_k_is_split():
    """coop > perStrip means a K split, so the index mixes both axes."""
    ti = _tile(grCoopWaves=4, grWavesPerStrip=2)
    writer = _Writer()
    module = Module()
    ok = _tluCoopWaveId(writer, _kernel(miWaveGroup=(2, 2)), module, ti, 0)
    assert ok is True
    assert not writer.vgprPool.outstanding, "temporaries must be checked back in"


# --- strip index -----------------------------------------------------------

@pytest.mark.parametrize("grid,perStrip", [((1, 1), 2), ((4, 1), 1)])
def test_strip_idx_is_skipped_for_a_single_strip_or_unshared_strip(grid, perStrip):
    ti = _tile(globalSubtileGrid=grid, grWavesPerStrip=perStrip)
    assert _tluStripIdx(_Writer(), _kernel(), Module(), "A", ti, 0) is False


def test_strip_idx_divides_the_axis_id_by_the_sharing_waves():
    ti = _tile(globalSubtileGrid=(4, 1), grWavesPerStrip=2)
    module = Module()
    assert _tluStripIdx(_Writer(), _kernel(miWaveGroup=(2, 2)), module, "A", ti, 0) is True
    assert _has(module, VLShiftRightB32)


def test_strip_idx_rejects_a_non_power_of_two_sharing_count():
    """The divide is a shift, so a non-power-of-two would silently mis-address."""
    ti = _tile(globalSubtileGrid=(4, 1), grWavesPerStrip=3)
    with pytest.raises(AssertionError, match="power of two"):
        _tluStripIdx(_Writer(), _kernel(miWaveGroup=(2, 2)), Module(), "A", ti, 0)


# --- K slot decomposition --------------------------------------------------

def test_k_wave_slots_without_a_split_is_one_slice_per_window():
    """Short unshared stack: no col_scatter, so a slice is the whole window."""
    kSplit, winSplit, rowsPerSlice, rowsPerRun = _tluKWaveSlots(_tile(subtileShape=(2, 1)))
    assert (kSplit, winSplit) == (1, 1)
    assert rowsPerSlice == rowsPerRun == 128


def test_k_wave_slots_splits_the_window_when_waves_cooperate():
    """coop waves with an unshared strip cut the window kSplit ways."""
    ti = _tile(grCoopWaves=4, grWavesPerStrip=1, subtileShape=(2, 1))
    kSplit, winSplit, rowsPerSlice, rowsPerRun = _tluKWaveSlots(ti)
    assert kSplit == 4
    assert rowsPerSlice == rowsPerRun // kSplit


def test_k_wave_slots_uses_the_load_count_on_a_col_scatter_stack():
    """col_scatter makes the load index the K column, so rows come from it."""
    ti = _tile(grCoopWaves=4, grWavesPerStrip=2, subtileShape=(16, 1), numGRPerSubtile=7)
    _, _, rowsPerSlice, _ = _tluKWaveSlots(ti)
    assert rowsPerSlice == 7


# --- K slice terms and offsets ---------------------------------------------

def test_k_slice_terms_emit_nothing_but_a_zero_without_a_split():
    module = Module()
    writer = _Writer()
    _tluKSliceTerms(writer, _kernel(), module, _tile(), 0, 1, 8, 16, "t")
    assert _has(module, VMovB32)
    assert not writer.sgprPool.outstanding


def test_k_slice_terms_scale_both_the_slice_and_the_window_run():
    ti = _tile(grCoopWaves=4, grWavesPerStrip=2, grKWindowSplit=2,
               grOtherAxisWaves=4, subtileShape=(2, 1))
    module = Module()
    writer = _Writer()
    _tluKSliceTerms(writer, _kernel(), module, ti, 0, 1, 8, 16, "t")
    assert _has(module, VMulLOU32)
    assert _has(module, VAddU32), "the window-run term must be folded in"
    assert not writer.vgprPool.outstanding and not writer.sgprPool.outstanding


def test_k_slice_global_offset_is_absent_without_any_split():
    assert _tluKSliceGlobalOffset(_Writer(), _kernel(), Module(), _tile()) is None


def test_k_slice_global_offset_converts_rows_to_bytes():
    ti = _tile(grCoopWaves=4, grWavesPerStrip=2, grOtherAxisWaves=4, subtileShape=(2, 1))
    module = Module()
    writer = _Writer()
    dst = _tluKSliceGlobalOffset(writer, _kernel(), module, ti)
    assert dst is not None
    assert "StrideA2" in _text(module), "the K stride must be applied"


# --- wave free-dim offset --------------------------------------------------

def test_wave_axis_global_offset_steps_by_whole_strips_per_wave():
    ti = _tile(localSubtileGrid=(2, 1), grWavesPerStrip=1, grCoopWaves=1)
    module = Module()
    writer = _Writer()
    dst = _tluWaveAxisGlobalOffset(writer, _kernel(miWaveGroup=(2, 2)), module, ti)
    assert dst is not None
    assert _has(module, VMulLOU32)
    assert not writer.sgprPool.outstanding


# --- DTL write-side K slice ------------------------------------------------

def test_dtl_k_slice_is_skipped_when_the_column_is_not_split():
    module = Module()
    _grDTLAddKSlice(_Writer(), _kernel(), module, "A", _tile(subtileShape=(2, 1)), 0)
    assert module.items() == [], "nothing to add without a K split"


def test_dtl_k_slice_mirrors_the_global_k_term():
    """The DTL write must step by the same slice the global read took."""
    ti = _tile(grCoopWaves=4, grWavesPerStrip=1, grOtherAxisWaves=4,
               subtileShape=(2, 1))
    module = Module()
    writer = _Writer()
    _grDTLAddKSlice(writer, _kernel(), module, "A", ti, 0)
    assert _has(module, VAddU32)
    assert not writer.vgprPool.outstanding and not writer.sgprPool.outstanding
