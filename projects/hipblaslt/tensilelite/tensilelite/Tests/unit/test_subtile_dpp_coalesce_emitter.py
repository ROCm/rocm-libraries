# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DPP store-coalesce emitter-invariant unit tests.

`_emit16bitSubtilePairedStoreCoalesce` reuses batchA's two dead ValuC accumulator
windows as scratch for the batchB pack and the blended store source.  That reuse is
only correct if the two windows are dword-aligned and mutually DISJOINT, and if
batchB sits a positive half-cache-line above batchA.  Both facts are load-bearing --
a violation silently corrupts the stored D -- so the emitter asserts them, and these
tests drive those asserts directly (no GPU) with a mock writer carrying just the
state the emitter reads before each check.
"""

from types import SimpleNamespace

import pytest

from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter

pytestmark = pytest.mark.unit


class _DestType:
    # bf16 path (isHalf() False selects VCvtPkF32toBF16); the value is irrelevant to the asserts.
    def isHalf(self):
        return False


def _writer():
    """A GlobalWriteBatchWriter carrying only what the coalesce emitter touches before its asserts."""
    w = object.__new__(GlobalWriteBatchWriter)
    w.kernel = {"ProblemType": {"DestDataType": _DestType()}, "NonTemporalD": 0}
    w.cvtVgprStruct = SimpleNamespace(
        vgprBf16Temp=20, vgprVoff=30, vgprBlendTmp=31,
        vgprPermAddr=32, vgprLaneGroupDelta=33, vgprAddrScratch=34,
    )
    w._permlane16Active = True
    w.parentWriter = SimpleNamespace(
        states=SimpleNamespace(bpeCexternal=2, bpeCexternalGSU1=2, subtileHoistedAddrArm=-1))
    return w


def _emit(w, sumIdx0, sumIdx1, this_off=0, partner_off=64):
    addrCalc = SimpleNamespace(addrDVgpr=40, globalOffset=this_off)
    partnerAddrCalc = SimpleNamespace(addrDVgpr=41, globalOffset=partner_off)
    # partner sum indices are disjoint from batchA and irrelevant to the two asserts under test.
    return w._emit16bitSubtilePairedStoreCoalesce(
        addrCalc, sumIdx0, sumIdx1, prefixOffset=0,
        partnerSumIdx0=40, partnerSumIdx1=44, partnerAddrCalc=partnerAddrCalc,
        tt0=3, blockIdxM=0, blockIdxN=0)


def test_valuc_windows_must_be_disjoint():
    # sumIdx1 == sumIdx0 + 2 is 2-aligned but ALIASES at regs sumIdx0+2..+3 -- the silent-corruption
    # case the assert exists to reject.
    w = _writer()
    with pytest.raises(AssertionError, match="ValuC reuse needs"):
        _emit(w, sumIdx0=0, sumIdx1=2)


def test_valuc_windows_must_be_dword_aligned():
    # An odd base breaks the dwordx4 store alignment.
    w = _writer()
    with pytest.raises(AssertionError, match="ValuC reuse needs"):
        _emit(w, sumIdx0=1, sumIdx1=8)


def test_batchb_must_sit_above_batcha_in_the_cache_line():
    # halfLine = partner.globalOffset - this.globalOffset must be a positive half-cache-line; a
    # flipped element order makes it <= 0, which would emit a negative immediate silently.
    w = _writer()
    with pytest.raises(AssertionError, match="batchB above batchA"):
        _emit(w, sumIdx0=0, sumIdx1=4, this_off=64, partner_off=0)
