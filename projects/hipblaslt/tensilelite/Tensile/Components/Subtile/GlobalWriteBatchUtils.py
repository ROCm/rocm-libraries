# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lightweight utilities for the subtile GlobalWriteBatch integration.

Kept in a separate module so that unit tests can import these helpers without
pulling in the rocisa C++ bindings.
"""

from ...Common import DataDirection


# ---------------------------------------------------------------------------
# Low-level helpers shared between GlobalWriteBatch and StreamK
# ---------------------------------------------------------------------------

def _extract_direct_vgpr_from_acc_read(accReadInst):
    """Return the physical VGPR index if accReadInst is a VGPR→VGPR move, else None.

    For VGPR-first FP4 subtile accumulators the acc→ValuC code list contains
    ``v_mov_b32 v[ValuC+N], v[physReg]`` moves (src is a plain arch-VGPR).
    For AGPR-backed accumulators the src is an accvgpr (regType="a"), which
    returns None here so the normal replaceHolder path is used instead.
    """
    if len(accReadInst.srcs) != 1:
        return None
    src = accReadInst.srcs[0]
    if getattr(src, "regType", None) != "v":
        return None
    if getattr(src, "regName", None) is not None:
        return None
    regIdx = getattr(src, "regIdx", None)
    return regIdx if isinstance(regIdx, int) else None


def _is_legal_valuC_offset(startVgprValu, maxVgpr, valuCOffset, width=1):
    """Return True when v[startVgprValu + valuCOffset + width - 1] is within bounds."""
    return startVgprValu + valuCOffset + width <= maxVgpr


def _has_any_vgpr_backed_accumulator(tileInfo) -> bool:
    """Return True when at least one D-tile accumulator was actually allocated as a VGPR.

    This inspects the actual register allocation recorded in tileInfo.vgprTiles
    (rather than just the kernel data types), so it correctly returns False when
    the VGPR budget was exhausted and all D-tile registers fell back to AGPR.
    """
    if tileInfo is None:
        return False
    return any(vtile.regList.is_vgpr for vtile in tileInfo.vgprTiles)


def _can_bypass_valu_c(kernel, edge: bool, atomic: bool, use_bias,
                       beta: bool = False) -> bool:
    """Return True when acc->ValuC v_mov moves can be skipped (subtile bypass).

    NOTE: `edge` is intentionally NOT a gate. Both edge and non-edge stores take
    the bypass; edge correctness is handled by the epilogue source-map remap
    (every edge ValuC read/write resolves through _valuCVgpr / _storeSumIdx), so
    there is no need to disable the bypass for edge. The parameter is kept for
    call-site clarity and is covered by test_edge_store.

    The bypass is only valid when every ValuC read in the global-write epilogue
    uses _valuCVgpr / _storeSumIdx. Any epilogue feature whose ValuC reads were
    NOT updated to those helpers must disable the bypass to avoid reading
    uninitialized staging registers:

    * Bias read           - the VAddF32/VAddPKF32 bias add now reads src1 via
      _valuCVgpr, so it is bypass-safe. The Int8/Int32 convertData path that runs
      before the bias add still uses inputPrefix="ValuC+", so those dest/data-type
      combos keep the bypass disabled.
    * Bias write (BiasSrc=D, not WorkGroupReduction)
                          - biasReductionModule addStore uses raw "ValuC+%d".
    * UseScaleCD          - scaleDModule uses raw "ValuC+%d".
    * UseE (non-gradient) - E-output pack path uses raw ValuC prefix.
    * HPA with non-16bit dest - packdata/convertData use inputPrefix="ValuC+";
      16-bit (Half/BF16) is already protected by the is16bitSubtile packdata skip.
    * Non-HPA Half dest + beta - _addSumAlphaWithCBeta VAddPKF16 path is not
      updated. HPA Half/BF16 beta paths use _valuCVgpr and are bypass-safe.
    * Non-HPA Half compute  - _applyAlpha VMulPKF16 path is not updated.
    * Int32 compute         - _applyAlpha VMulLOU32 path is not updated.

    UseScaleAB=Vector and UseScaleAlphaVec: applyScaleVec was updated to use
    _valuCVgpr for all src/dst reads, so these no longer block the bypass.
    ActivationFuncCall uses _copyActivationData, which also resolves ValuC
    through _valuCVgpr, so activation function calls are bypass-safe.
    However, the bypass for these scale-vec paths is only validated for the
    beta=0 (C = A*B, no C-accumulation) case. When beta is non-zero the
    interaction with _addSumAlphaWithCBeta has not been tested and bypass
    is conservatively disabled.
    """
    if not kernel.get("UseSubtileImpl"):
        return False
    if kernel["LocalSplitU"] != 1:
        return False
    if atomic:
        return False
    pt = kernel["ProblemType"]
    if pt.get("Gradient", False):
        return False
    # Complex / F64 epilogues (alpha, beta, pack/convert) still address the raw
    # "ValuC+N" staging slots and were not routed through _valuCVgpr, so the
    # bypass would read uninitialized registers. FP4 subtile is never complex or
    # F64, so this only guards hypothetical subtile configs from silent corruption.
    computeDt = pt.get("ComputeDataType")
    destDt = pt.get("DestDataType")
    if (computeDt is not None and (computeDt.isComplex() or computeDt.isDouble())) \
            or (destDt is not None and (destDt.isComplex() or destDt.isDouble())):
        return False
    if use_bias == DataDirection.READ:
        # The bias-add epilogue (VAddF32 / VAddPKF32, GlobalWriteBatch._epilogue)
        # now resolves its ValuC src1 reads through _valuCVgpr, so bias read is
        # bypass-safe for the common Single-compute path. Two sub-paths still read
        # the raw "ValuC+%d" staging slot and must keep the bypass disabled:
        #   * compute != f32 -> the bias branch raises today, but guard anyway.
        #   * the Int8/Int32 convertData(CVT_I32_to_F32, inputPrefix="ValuC+") path
        #     that runs before the bias add for integer dest / int8-in dest combos.
        compute = pt.get("ComputeDataType")
        if compute is None or not compute.isSingle():
            return False
        dest = pt.get("DestDataType")
        dtype = pt.get("DataType")
        biasConvertRawValuC = (
            (dest is not None and (dest.isInt8() or dest.isInt32()))
            or (dtype is not None and dtype.isInt8()
                and dest is not None and (dest.isHalf() or dest.isBFloat16()))
        )
        if biasConvertRawValuC:
            return False
    # biasReductionModule: stores "ValuC+%d" directly when BiasSrc=D and
    # WorkGroupReduction is off - that code was not updated for bypass.
    if (use_bias == DataDirection.WRITE
            and not kernel.get("WorkGroupReduction", False)
            and pt.get("BiasSrc") == "D"):
        return False
    if pt.get("UseScaleCD", False):
        return False
    if pt.get("UseE", False):
        return False
    # applyScaleVec bypass is only validated for beta=0 (C = A*B).
    # Non-zero beta interacts with _addSumAlphaWithCBeta in untested ways.
    if beta and (pt.get("UseScaleAlphaVec", False) or pt.get("UseScaleAB", "None") == "Vector"):
        return False
    # GroupLoadStore + beta adds an extra C-load/addressing path in the store
    # (see _preciseStorePerElem) that was not audited against the source-map
    # remap; disable the bypass conservatively when both are active.
    if beta and kernel.get("GroupLoadStore", False):
        return False
    dest = pt.get("DestDataType")
    # _addSumAlphaWithCBeta: non-HPA Half path (VAddPKF16) is not updated.
    if beta and dest is not None and dest.isHalf() and not pt.get("HighPrecisionAccumulate", False):
        return False
    # _applyAlpha: non-HPA Half compute (VMulPKF16) and Int32 compute
    # (VMulLOU32) paths are not updated for bypass.
    compute = pt.get("ComputeDataType")
    if compute is not None:
        if compute.isHalf() and not pt.get("HighPrecisionAccumulate", False):
            return False
        if compute.isInt32():
            return False
    # HPA pack/convert calls use inputPrefix="ValuC+" for FP8/BF8/Int8/Int32 dest types.
    # F32 (Single) has no pack step at all; its epilogue paths already use _valuCVgpr().
    # Half and BFloat16 are protected by the is16bitSubtile packdata skip.
    if pt.get("HighPrecisionAccumulate", False):
        if dest is not None and not (dest.isHalf() or dest.isBFloat16() or dest.isSingle()):
            return False
    return True


def _derive_use_bias_direction(kernel):
    """Reproduce KernelWriter's states.useBias purely from `kernel`.

    The register allocator (estimateVgprAccumulatorSplit) runs before
    states.useBias is assigned, so it must re-derive the bias direction the same
    way KernelWriter.py does to predict whether any store section can bypass.
    """
    pt = kernel["ProblemType"]
    if not pt.get("UseBias", False):
        return DataDirection.NONE
    if pt.get("Gradient", False):
        if pt.get("BiasSrc") in ("D", "A", "B"):
            return DataDirection.WRITE
        return DataDirection.NONE
    return DataDirection.READ


def _derive_atomic(kernel):
    """Reproduce the store-site `atomic` decision purely from `kernel`.

    Mirrors KernelWriterAssembly's atomic gate: GlobalSplitU accumulation that is
    not buffered to a separate workspace stores atomically into D.
    """
    gsu = kernel.get("GlobalSplitU", 1)
    if not (gsu > 1 or gsu == -1):
        return False
    accum = kernel.get("_GlobalAccumulation", None)
    return accum not in ("MultipleBuffer", "MultipleBufferSingleKernel")


def _can_bypass_any_store_section(kernel) -> bool:
    """Return True if at least one emitted store section can bypass ValuC staging.

    The epilogue emits a store section per (beta, factorDim) combination. A kernel
    that can never bypass any section pays the full VGPR-first cost (constrained
    accumulator placement, reserved valuCStage window, every acc->ValuC v_mov)
    with zero benefit, so the allocator should fall back to AGPR-first for it.

    Computed purely from `kernel` (no AsmStoreState / states.useBias dependency)
    so it is safe to call from estimateVgprAccumulatorSplit at allocation time.
    factorDim does not affect _can_bypass_valu_c, so only beta is varied here.
    """
    betas = [False, True] if kernel["ProblemType"].get("UseBeta", False) else [False]
    use_bias = _derive_use_bias_direction(kernel)
    atomic = _derive_atomic(kernel)
    return any(
        _can_bypass_valu_c(kernel, edge=False, atomic=atomic, use_bias=use_bias, beta=b)
        for b in betas
    )
