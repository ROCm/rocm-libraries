#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the subtile epilogue pre-store bias/SAV drain+barrier emit fork
# (Components/GlobalWriteBatch.py).
#
# The subtile bias/scaleAlphaVec epilogue stages a per-column vector into LDS and
# every wave reads it while computing the paired stores. A drain (s_waitcnt
# lgkmcnt(0)) + workgroup s_barrier orders those LDS reads ahead of the stores.
# It is required only for multi-DU (the store loop re-stages the LDS vector, so an
# LDS write can race in-flight reads); single-DU stages the vector once and the
# store region is LDS-write-free, so the barrier is elided. These tests pin that
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

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

WAVESIZE_64 = 64


def _init_rocisa_gfx950():
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_64)


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
        _init_rocisa_gfx950()
        mod, emitted = _emit_barrier(is_multi_du=True, needs_drain=True)
        nbar, ndrain = _barrier_counts(mod)
        assert emitted is True
        assert nbar == 1, "multi-DU bias/SAV epilogue must emit exactly one s_barrier"
        assert ndrain == 1, "multi-DU must emit the s_waitcnt lgkmcnt(0) LDS-read drain"

    def test_singledu_with_drain_elides_barrier(self):
        # The whole point of the change: single-DU must NOT emit the barrier.
        _init_rocisa_gfx950()
        mod, emitted = _emit_barrier(is_multi_du=False, needs_drain=True)
        nbar, ndrain = _barrier_counts(mod)
        assert emitted is False
        assert nbar == 0, "single-DU epilogue must elide the pre-store s_barrier"
        assert ndrain == 0, "single-DU epilogue must elide the s_waitcnt lgkmcnt(0) drain"

    def test_no_drain_never_emits_barrier(self):
        # No bias/SAV staging -> no ordering needed, in either DU mode.
        _init_rocisa_gfx950()
        for multi in (True, False):
            mod, emitted = _emit_barrier(is_multi_du=multi, needs_drain=False)
            nbar, ndrain = _barrier_counts(mod)
            assert emitted is False
            assert (nbar, ndrain) == (0, 0)


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
        _init_rocisa_gfx950()
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
        _init_rocisa_gfx950()
        store = DSStoreB32(vgpr(0), vgpr(1))
        assert str(store).split()[0].startswith("ds_write")  # sanity: it is a write
        assert GlobalWriteBatchWriter._isLdsMemoryWrite(store) is True

    def test_bpermute_crossbar_is_not_flagged(self):
        # ds_bpermute is a DSStoreInstruction subclass but writes no LDS memory.
        from rocisa.container import vgpr
        from rocisa.instruction import DSBPermuteB32
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        _init_rocisa_gfx950()
        bp = DSBPermuteB32(vgpr(0), vgpr(1), vgpr(2))
        assert "bpermute" in str(bp)
        assert GlobalWriteBatchWriter._isLdsMemoryWrite(bp) is False

    def test_non_ds_instruction_is_not_flagged(self):
        from rocisa.instruction import SBarrier
        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter
        _init_rocisa_gfx950()
        assert GlobalWriteBatchWriter._isLdsMemoryWrite(SBarrier(comment="x")) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
