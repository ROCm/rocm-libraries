# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# GR emit leaf functions (no-cycle internal module).
#
# Contains emitSingleBufferLoad, emitScaleGRLDSSwap, globalReadPtrUpdates,
# globalReadLDSBufferSwap, globalReadDoScaleSubtile, and
# globalReadScalePtrUpdates — the GR-emit symbols that LogicalScheduler.py
# needs. Keeping them here avoids the Kernel.py ↔ LogicalScheduler.py cycle:
#
#   Kernel.py         imports LogicalScheduler
#   LogicalScheduler  imports _gr_emit_leaves   (this file — no back-edge)
#   Kernel.py         re-exports from here for external callers
################################################################################

from rocisa.code import Module

from tensile_writer.subtile.module_builder import ModuleBuilder as _ModuleBuilder

_GR_LEAVES_BUILDER = None


def _gr_leaves_builder():
    global _GR_LEAVES_BUILDER
    if _GR_LEAVES_BUILDER is None:
        _GR_LEAVES_BUILDER = _ModuleBuilder()
    return _GR_LEAVES_BUILDER


def emitSingleBufferLoad(tileInfo, kernel, sId0, sId1):
    """Emit buffer_load instructions for a single subtile (sId0, sId1)."""
    plan = tileInfo.singleBufferLoadPlan(sId0, sId1)
    if plan.skip:
        return Module()

    tc = tileInfo.tc
    isGlc = bool(kernel["NonTemporal%s" % tc] & 0x1)
    isSlc = bool(kernel["NonTemporal%s" % tc] & 0x2)
    isNT  = bool(kernel["NonTemporal%s" % tc] & 0x4)

    regListIdx = tileInfo.grRegGroupForSubtileRow(sId0)
    regList = tileInfo.localSubtilesRegister[regListIdx]
    useSgpr = regList.is_sgpr

    soffset = regList.ref(0) if len(regList) > 0 and useSgpr else 0
    voffs = [
        (tileInfo.sharedVgprGROffset[i] if useSgpr or len(regList) == 0
         else regList.indices[i])
        for i in range(len(plan.m0Offsets))
    ]
    return _gr_leaves_builder().single_buffer_load(
        tc, isGlc, isSlc, isNT, plan.offsetK, plan.grBaseId,
        list(plan.m0Offsets), soffset, voffs)


def emitScaleGRLDSSwap(ti, writer, kernel):
    """Toggle scale GR DTL write target between double-buffer halves."""
    return _gr_leaves_builder().gr_lds_buffer_swap(ti.tc)


def globalReadPtrUpdates(tc, writer, kernel):
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    return ti_.emitGRPtrUpdate(writer, kernel)


def globalReadLDSBufferSwap(tc, writer, kernel):
    if tc in ['A', 'B']:
        ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
        return ti_.emitGRLDSBufferSwap(writer, kernel)
    else:
        ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
        return emitScaleGRLDSSwap(ti_, writer, kernel)


def globalReadDoScaleSubtile(tc, writer, kernel):
    """Scale GR: load scale bytes global -> LDS via DTL BufferLoadB128."""
    if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
        return Module()

    tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

    isGlc = bool(kernel["NonTemporal%s" % tc] & 0x1)
    isSlc = bool(kernel["NonTemporal%s" % tc] & 0x2)
    isNT  = bool(kernel["NonTemporal%s" % tc] & 0x4)

    assert len(tileInfo.sharedVgprGROffset) > 0, "Scale GR requires at least 1 GR offset VGPR"

    return _gr_leaves_builder().scale_gr_load(tc, isGlc, isSlc, isNT, tileInfo.sharedVgprGROffset[0])


def globalReadScalePtrUpdates(tc, writer, kernel):
    """Advance scale SRD base pointer by one depthU iteration."""
    ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
    inc = int(ti_.lrSubtileSize * ti_.lrGlobalSubtileGrid[1])
    return _gr_leaves_builder().scale_gr_ptr_update(tc, inc)
