#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the subtile epilogue pre-store bias/SAV drain+barrier emit fork
# (Components/GlobalWriteBatch.py).
#
# The subtile bias/scaleAlphaVec epilogue stages a per-column vector into LDS and
# every wave reads it while computing the paired stores. A drain (s_waitcnt
# lgkmcnt(0)) + workgroup s_barrier is emitted for multi-DU only, and which side of
# _emitAdd it lands on is what decides whether it does anything. Multi-DU emits it
# first, so it sits between those ds_reads and the v_pk_mul/v_pk_add that consume
# them and is the only thing retiring them (UseSubtileImpl drops bias/SAV from the
# interleaved per-element waitcnt). Single-DU emits _emitAdd first, so it would land
# after its own consumers and retire nothing, and is elided. These tests pin that
# emit fork so a future refactor cannot silently add/drop the barrier, and pin the
# derivation of the two predicates (isSubtileMultiDU, needs-bias/SAV-drain) that
# drive it -- including PGR-invariance and both the useBias and UseScaleAlphaVec
# triggers. No GPU required: the emitted rocisa module is the contract.
#
# Usage:
#   pytest test_subtile_prestore_barrier.py -v
################################################################################

import os
import shutil
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from gpu_test_helpers import init_rocisa  # noqa: E402

GFX950 = "gfx950"
WAVESIZE_64 = 64

pytestmark = pytest.mark.skipif(
    shutil.which("amdclang++") is None and not os.path.exists("/usr/bin/amdclang++"),
    reason="amdclang++ not found; cannot init rocisa",
)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    init_rocisa(target=GFX950, wavesize=WAVESIZE_64)


def _kernel(*, multi_du, use_subtile=True, sav=0, depth_u=64):
    """Minimal kernel dict driving isSubtileMultiDU + _needsBiasSavDrain.

    Multi-DU is expressed exactly as isSubtileMultiDU reads it: a per-uid DepthU
    (_DepthUA) smaller than the loop DepthU.
    """
    return {
        "DepthU": depth_u,
        "_DepthUA": (depth_u // 2) if multi_du else depth_u,
        "_DepthUB": depth_u,
        "UseSubtileImpl": use_subtile,
        "ProblemType": {"UseScaleAlphaVec": sav},
    }


def _barrier_counts(module):
    """(#SBarrier, #SWaitCnt with dscnt==0) in the emitted module."""
    from rocisa.instruction import SBarrier, SWaitCnt
    items = list(module.flatitems())
    nbar = sum(isinstance(i, SBarrier) for i in items)
    ndrain = sum(isinstance(i, SWaitCnt) and getattr(i, "dscnt", None) == 0 for i in items)
    return nbar, ndrain


def _emit_barrier(is_multi_du, needs_drain):
    from rocisa.code import Module
    from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
    mod = Module("test")
    emitted = GlobalWriteBatchWriter._emitBiasSavDrainBarrier(mod, is_multi_du, needs_drain)
    return mod, emitted


# ---------------------------------------------------------------------------
# The emit fork: barrier present iff (multi-DU AND needs-drain).
# ---------------------------------------------------------------------------
class TestBarrierEmitFork:
    def test_multidu_with_drain_emits_drain_and_barrier(self):
        mod, emitted = _emit_barrier(is_multi_du=True, needs_drain=True)
        nbar, ndrain = _barrier_counts(mod)
        assert emitted is True
        assert nbar == 1, "multi-DU bias/SAV epilogue must emit exactly one s_barrier"
        assert ndrain == 1, "multi-DU must emit the s_waitcnt lgkmcnt(0) LDS-read drain"

    def test_singledu_with_drain_elides_barrier(self):
        # The whole point of the change: single-DU must NOT emit the barrier.
        mod, emitted = _emit_barrier(is_multi_du=False, needs_drain=True)
        nbar, ndrain = _barrier_counts(mod)
        assert emitted is False
        assert nbar == 0, "single-DU epilogue must elide the pre-store s_barrier"
        assert ndrain == 0, "single-DU epilogue must elide the s_waitcnt lgkmcnt(0) drain"

    def test_no_drain_never_emits_barrier(self):
        # No bias/SAV staging -> no ordering needed, in either DU mode.
        for multi in (True, False):
            mod, emitted = _emit_barrier(is_multi_du=multi, needs_drain=False)
            nbar, ndrain = _barrier_counts(mod)
            assert emitted is False
            assert (nbar, ndrain) == (0, 0)


# ---------------------------------------------------------------------------
# The invariant the fork exists to produce: WHERE the pair lands relative to
# _emitAdd. Counting barriers (above) is not sufficient on its own -- a refactor
# that moved the multi-DU pair to after _emitAdd would keep every test above
# green while leaving the bias/SAV reads with nothing to retire them, because
# UseSubtileImpl drops bias/SAV from the interleaved per-element waitcnt.
# ---------------------------------------------------------------------------
def _bpermuteStoreRegion(module):
    """Stand-in for _emitAdd: the ds_bpermute crossbar the real store region emits."""
    from rocisa.container import vgpr
    from rocisa.instruction import DSBPermuteB32
    module.add(DSBPermuteB32(vgpr(0), vgpr(1), vgpr(2)))


def _dsWriteStoreRegion(module):
    """Stand-in for _emitAdd that writes real LDS -- must trip the single-DU guard."""
    from rocisa.container import vgpr
    from rocisa.instruction import DSStoreB32
    module.add(DSStoreB32(vgpr(0), vgpr(1)))


def _writer(*, multi_du, use_bias=True, sav=0, store_region=None):
    """A GlobalWriteBatchWriter carrying only what emit() touches.

    __init__ takes ~30 collaborators none of which the barrier fork reads, so the
    instance is built directly and the three sub-emitters are stubbed. The fork
    itself, both predicates and the LDS-write guard are the real ones.
    """
    from Tensile.Common import DataDirection
    from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter

    w = object.__new__(GlobalWriteBatchWriter)
    k = _kernel(multi_du=multi_du, sav=sav)
    k["CompactLoopStore"] = False
    w.kernel = k
    w.atomic = False  # drives the moduleName property
    w.parentWriter = SimpleNamespace(states=SimpleNamespace(
        useBias=DataDirection.WRITE if use_bias else DataDirection.NONE))
    w._checkAtomicPreconditions = lambda: True
    w._prolog = lambda module: None
    w._epilog = lambda module: None
    w._emitAdd = store_region or _bpermuteStoreRegion
    return w


def _positions(module):
    """First index of (drain, barrier, store-region marker); None if absent."""
    from rocisa.instruction import SBarrier, SWaitCnt
    items = list(module.flatitems())

    def first(pred):
        return next((i for i, x in enumerate(items) if pred(x)), None)

    return (first(lambda x: isinstance(x, SWaitCnt) and getattr(x, "dscnt", None) == 0),
            first(lambda x: isinstance(x, SBarrier)),
            first(lambda x: "permute" in str(x) or "ds_write" in str(x)))


class TestEmitOrderingFork:
    def test_multidu_pair_precedes_the_store_region(self):
        drain, barrier, store = _positions(_writer(multi_du=True).emit())
        assert None not in (drain, barrier, store), \
            "multi-DU must emit the drain, the barrier and the store region"
        assert drain < barrier < store, (
            "multi-DU must emit the drain+barrier BEFORE _emitAdd -- that is what puts "
            "the s_waitcnt lgkmcnt(0) between the bias/SAV ds_reads and the "
            "v_pk_mul/v_pk_add that consume them")

    def test_singledu_emits_neither_around_the_store_region(self):
        drain, barrier, store = _positions(_writer(multi_du=False).emit())
        assert store is not None, "the store region must still be emitted"
        assert (drain, barrier) == (None, None), (
            "single-DU must elide the pair: _emitAdd comes first, so the pair would "
            "land past its own consumers and retire nothing")

    def test_singledu_lds_write_in_store_region_trips_the_guard(self):
        w = _writer(multi_du=False, store_region=_dsWriteStoreRegion)
        with pytest.raises(AssertionError, match="LDS-write-free"):
            w.emit()

    def test_singledu_bpermute_in_store_region_does_not_trip_the_guard(self):
        # ds_bpermute is a DSStoreInstruction subclass that writes no LDS memory, so
        # the real store region must pass the guard.
        _writer(multi_du=False).emit()

    def test_no_drain_trigger_skips_the_guard_entirely(self):
        # Without bias or SAV there is no staged vector, so even a real ds_write in
        # the store region is not this guard's business.
        w = _writer(multi_du=False, use_bias=False, sav=0,
                    store_region=_dsWriteStoreRegion)
        drain, barrier, _ = _positions(w.emit())
        assert (drain, barrier) == (None, None)


# ---------------------------------------------------------------------------
# isSubtileMultiDU: single- vs multi-DU, invariant to PGR level.
# ---------------------------------------------------------------------------
class TestIsSubtileMultiDU:
    @pytest.mark.parametrize("pgr", [0, 1, 2])
    def test_single_vs_multi_du(self, pgr):
        from Tensile.Common import isSubtileMultiDU
        single = _kernel(multi_du=False)
        multi = _kernel(multi_du=True)
        single["PrefetchGlobalRead"] = pgr
        multi["PrefetchGlobalRead"] = pgr
        assert isSubtileMultiDU(single) is False
        assert isSubtileMultiDU(multi) is True


# ---------------------------------------------------------------------------
# _needsBiasSavDrain: both triggers (useBias, UseScaleAlphaVec) + gating.
# ---------------------------------------------------------------------------
class TestNeedsBiasSavDrain:
    def test_bias_trigger(self):
        from Tensile.Common import DataDirection
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        k = _kernel(multi_du=False, sav=0)
        assert GlobalWriteBatchWriter._needsBiasSavDrain(k, DataDirection.WRITE) is True
        assert GlobalWriteBatchWriter._needsBiasSavDrain(k, DataDirection.NONE) is False

    def test_scalealphavec_trigger(self):
        from Tensile.Common import DataDirection
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        k = _kernel(multi_du=False, sav=1)
        # SAV alone (no bias) still needs the drain.
        assert GlobalWriteBatchWriter._needsBiasSavDrain(k, DataDirection.NONE) is True

    def test_requires_subtile(self):
        from Tensile.Common import DataDirection
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        k = _kernel(multi_du=False, use_subtile=False, sav=1)
        # Non-subtile kernels never take the LDS-staging path.
        assert GlobalWriteBatchWriter._needsBiasSavDrain(k, DataDirection.WRITE) is False


# ---------------------------------------------------------------------------
# End-to-end fork over the full matrix: (DU) x (bias/SAV trigger) x (PGR).
# Barrier present iff multi-DU and a trigger is set; single-DU always elides.
# ---------------------------------------------------------------------------
class TestForkMatrix:
    @pytest.mark.parametrize("pgr", [0, 1, 2])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("sav", [0, 1])
    def test_matrix(self, pgr, use_bias, sav):
        from Tensile.Common import DataDirection, isSubtileMultiDU
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        useBias = DataDirection.WRITE if use_bias else DataDirection.NONE
        trigger = use_bias or bool(sav)
        for multi in (True, False):
            k = _kernel(multi_du=multi, sav=sav)
            k["PrefetchGlobalRead"] = pgr
            is_multi = isSubtileMultiDU(k)
            needs = GlobalWriteBatchWriter._needsBiasSavDrain(k, useBias)
            assert needs is trigger
            mod, emitted = _emit_barrier(is_multi, needs)
            nbar, ndrain = _barrier_counts(mod)
            expect = multi and trigger
            assert emitted is expect
            assert (nbar, ndrain) == ((1, 1) if expect else (0, 0)), \
                f"multi={multi} bias={use_bias} sav={sav} pgr={pgr}: barrier emit mismatch"


# ---------------------------------------------------------------------------
# _isLdsMemoryWrite: the single-DU elision invariant guard must flag real LDS
# writes but NOT the ds_permute/ds_bpermute crossbar (which rocisa models under
# DSStoreInstruction but writes no LDS memory).
# ---------------------------------------------------------------------------
class TestIsLdsMemoryWrite:
    def test_real_ds_write_is_flagged(self):
        from rocisa.container import vgpr
        from rocisa.instruction import DSStoreB32
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        store = DSStoreB32(vgpr(0), vgpr(1))
        assert str(store).split()[0].startswith("ds_write")  # sanity: it is a write
        assert GlobalWriteBatchWriter._isLdsMemoryWrite(store) is True

    def test_bpermute_crossbar_is_not_flagged(self):
        # ds_bpermute is a DSStoreInstruction subclass but writes no LDS memory.
        from rocisa.container import vgpr
        from rocisa.instruction import DSBPermuteB32
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        bp = DSBPermuteB32(vgpr(0), vgpr(1), vgpr(2))
        assert "bpermute" in str(bp)
        assert GlobalWriteBatchWriter._isLdsMemoryWrite(bp) is False

    def test_non_ds_instruction_is_not_flagged(self):
        from rocisa.instruction import SBarrier
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        assert GlobalWriteBatchWriter._isLdsMemoryWrite(SBarrier(comment="x")) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
