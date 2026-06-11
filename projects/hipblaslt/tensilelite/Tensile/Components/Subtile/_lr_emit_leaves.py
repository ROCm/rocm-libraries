# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# LR emit leaf functions (no-cycle internal module).
#
# Contains emitSingleDsRead, emitScaleDsRead, emitScaleLRLDSSwap, and
# localReadLDSBufferSwap — the only LR-emit symbols that LogicalScheduler.py
# needs.  Keeping them here avoids the Kernel.py ↔ LogicalScheduler.py cycle:
#
#   Kernel.py         imports LogicalScheduler
#   LogicalScheduler  imports _lr_emit_leaves   (this file — no back-edge)
#   Kernel.py         re-exports from here for external callers
################################################################################

from rocisa.code import Module

from tensile_writer.subtile.module_builder import ModuleBuilder as _ModuleBuilder

_LR_LEAVES_BUILDER = None


def _lr_leaves_builder():
    global _LR_LEAVES_BUILDER
    if _LR_LEAVES_BUILDER is None:
        _LR_LEAVES_BUILDER = _ModuleBuilder()
    return _LR_LEAVES_BUILDER


def emitSingleDsRead(tileInfo, sId0, sId1, subIterK, dstTile):
    """Emit DSLoadB128 instruction(s) for one MMA tile within a subtile.

    Args:
        tileInfo:  TileInfo (for subtileSize, loadRatioGR, sharedVgprLROffset, tc)
        sId0:      Subtile row index (used for offset computation)
        subIterK:  subIterK index within the subtile
        dstTile:   RegisterTileInfo — destination vgpr tile for the load

    Returns a Module.
    """
    dstVgpr = dstTile.regList.indices[0]
    numRegs = len(dstTile.regList.indices)
    plan = tileInfo.singleDsReadPlan(sId0, sId1, subIterK, numRegs)
    dstRegOffsets = [rd.dstRegOffset for rd in plan.reads]
    addrVgprs = [tileInfo.sharedVgprLROffset[rd.addrIdx] for rd in plan.reads]
    return _lr_leaves_builder().single_ds_read(
        tileInfo.tc, sId0, sId1, subIterK, dstVgpr, plan.regsPerDsRead,
        plan.offset, dstRegOffsets, addrVgprs)


def emitScaleDsRead(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k=-1):
    """Scale LR: read 4 scale bytes (one E8M0 group) from LDS via ds_read_b32."""
    return _lr_leaves_builder().scale_ds_read(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k)


def emitScaleLRLDSSwap(ti, writer, kernel):
    """Toggle scale LR read offsets between double-buffer halves."""
    return _lr_leaves_builder().lr_lds_buffer_swap(
        ti.tc, list(ti.sharedVgprLROffset), list(ti.sharedVgprLROffsetSwap))


def localReadLDSBufferSwap(tc, writer, kernel):
    if tc in ['A', 'B']:
        ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
        return ti_.emitLRLDSBufferSwap(writer, kernel)
    else:
        ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
        return emitScaleLRLDSSwap(ti_, writer, kernel)
