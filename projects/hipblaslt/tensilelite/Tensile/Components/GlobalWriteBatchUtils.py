################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Lightweight utilities for GlobalWriteBatch — no rocisa dependency.

Kept in a separate module so that unit tests can import these helpers without
pulling in the rocisa C++ bindings.
"""

from ..Common import DataDirection


def _can_bypass_valu_c(kernel, edge: bool, atomic: bool, use_bias,
                       beta: bool = False) -> bool:
    """Return True when acc→ValuC v_mov moves can be skipped (subtile bypass).

    The bypass is only valid when every ValuC read in the global-write epilogue
    uses _valuCVgpr / _storeSumIdx.  Any epilogue feature whose ValuC reads were
    NOT updated to those helpers must disable the bypass to avoid reading
    uninitialized staging registers:

    * ActivationFuncCall  – copyData reads raw elementSumIdx (ValuC).
    * Bias read           – VAddF32/VAddPKF32 src1 uses raw "ValuC+%d".
    * Bias write (BiasSrc=D, not WorkGroupReduction)
                          – biasReductionModule addStore uses raw "ValuC+%d".
    * UseScaleCD          – scaleDModule uses raw "ValuC+%d".
    * UseE (non-gradient) – E-output pack path uses raw ValuC prefix.
    * HPA with non-16bit dest – packdata/convertData use inputPrefix="ValuC+";
      16-bit (Half/BF16) is already protected by the is16bitSubtile packdata skip.
    * Half/BF16 dest + beta – _addSumAlphaWithCBeta Half/BF16 paths (VAddPKF16,
      mixinst, VMacF32) are not updated; only the F32-single beta path is.
    * Non-HPA Half compute  – _applyAlpha VMulPKF16 path is not updated.
    * Int32 compute         – _applyAlpha VMulLOU32 path is not updated.

    UseScaleAB=Vector and UseScaleAlphaVec: applyScaleVec was updated to use
    _valuCVgpr for all src/dst reads, so these no longer block the bypass.
    However, the bypass for these scale-vec paths is only validated for the
    beta=0 (C = A*B, no C-accumulation) case.  When beta is non-zero the
    interaction with _addSumAlphaWithCBeta has not been tested and bypass
    is conservatively disabled.
    """
    if not kernel.get("UseSubtileImpl"):
        return False
    if kernel["LocalSplitU"] != 1:
        return False
    if edge or atomic:
        return False
    pt = kernel["ProblemType"]
    if pt.get("Gradient", False):
        return False
    if kernel.get("ActivationFuncCall", False):
        return False
    if use_bias == DataDirection.READ:
        return False
    # biasReductionModule: stores "ValuC+%d" directly when BiasSrc=D and
    # WorkGroupReduction is off — that code was not updated for bypass.
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
    dest = pt.get("DestDataType")
    # _addSumAlphaWithCBeta: Half/BF16 paths (VAddPKF16, mixinst, VMacF32)
    # are not updated; only DestDataType.isSingle() beta was updated.
    if beta and dest is not None and (dest.isHalf() or dest.isBFloat16()):
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
