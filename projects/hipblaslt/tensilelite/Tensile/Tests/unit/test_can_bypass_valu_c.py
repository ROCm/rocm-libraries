################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Tests for _can_bypass_valu_c().

The bypass skips acc→ValuC v_mov copies in the subtile non-edge epilogue.
It is only safe when every ValuC read in the epilogue goes through
_valuCVgpr / _storeSumIdx.  Features whose reads were NOT updated must
disable the bypass so they do not access uninitialized staging registers.

The common plain-GEMM case (C = A * B, UseSubtileImpl=1, LSU=1, non-edge,
non-atomic, no activation/bias/scale) SHOULD bypass — confirming that no
extra v_mov instructions are emitted for those kernels.
"""

import pytest
from unittest.mock import MagicMock

from Tensile.Components.Subtile.GlobalWriteBatchUtils import _can_bypass_valu_c
from Tensile.Common import DataDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dest_type(is_half=False, is_bf16=False, is_single=False,
                    is_int8=False, is_int32=False):
    """Return a mock DestDataType with controllable type predicates.

    All flags default to False so that unspecified combinations do not
    accidentally pass the HPA allow-list guard added for F32 support, nor
    accidentally trip the bias-read Int8/Int32 convert guard (those predicates
    would otherwise return a truthy MagicMock).
    """
    dt = MagicMock()
    dt.isHalf.return_value = is_half
    dt.isBFloat16.return_value = is_bf16
    dt.isSingle.return_value = is_single
    dt.isInt8.return_value = is_int8
    dt.isInt32.return_value = is_int32
    return dt


def _make_compute_type(is_half=False, is_int32=False, is_single=False):
    """Return a mock ComputeDataType."""
    ct = MagicMock()
    ct.isHalf.return_value = is_half
    ct.isInt32.return_value = is_int32
    ct.isSingle.return_value = is_single
    return ct


def _base_kernel(*, dest_type=None, compute_type=None):
    """Minimal kernel dict for a plain subtile FP32 GEMM (C = A * B)."""
    if dest_type is None:
        dest_type = _make_dest_type()          # non-Half, non-BF16 (e.g. FP4/FP32)
    pt = {
        "Gradient": False,
        "UseScaleAlphaVec": False,
        "UseScaleCD": False,
        "UseE": False,
        "UseScaleAB": "None",
        "HighPrecisionAccumulate": False,
        "DestDataType": dest_type,
    }
    if compute_type is not None:
        pt["ComputeDataType"] = compute_type
    return {
        "UseSubtileImpl": True,
        "LocalSplitU": 1,
        "ActivationFuncCall": False,
        "WorkGroupReduction": False,
        "ProblemType": pt,
    }


# ---------------------------------------------------------------------------
# The common case: plain GEMM should bypass (no v_mov for acc→ValuC)
# ---------------------------------------------------------------------------

class TestPlainGemmBypass:
    """C = A * B with subtile impl enables the bypass — no v_mov overhead."""

    def test_plain_gemm_non_edge_non_atomic(self):
        """UseSubtileImpl=1, LSU=1, non-edge, non-atomic → bypass enabled."""
        assert _can_bypass_valu_c(_base_kernel(), edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_plain_gemm_beta_zero_no_activation(self):
        """Beta=0 path (no bias, no activation) is just as plain — still bypasses."""
        k = _base_kernel()
        k["ActivationFuncCall"] = False
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True


# ---------------------------------------------------------------------------
# Guards that MUST disable the bypass
# ---------------------------------------------------------------------------

class TestBypassDisabledForStructuralReasons:
    """Basic structural conditions that must prevent the bypass."""

    def test_no_subtile_impl(self):
        k = _base_kernel()
        k["UseSubtileImpl"] = False
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_subtile_impl_missing(self):
        k = _base_kernel()
        del k["UseSubtileImpl"]
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_local_split_u_greater_than_1(self):
        k = _base_kernel()
        k["LocalSplitU"] = 2
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_edge_store(self):
        """Subtile edge stores route ValuC through _valuCVgpr/_storeSumIdx."""
        assert _can_bypass_valu_c(_base_kernel(), edge=True, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_atomic_store(self):
        assert _can_bypass_valu_c(_base_kernel(), edge=False, atomic=True,
                                  use_bias=DataDirection.NONE) is False

    def test_gradient(self):
        k = _base_kernel()
        k["ProblemType"]["Gradient"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False


class TestBypassDisabledForEpilogueFeatures:
    """Epilogue features with un-updated raw ValuC reads must disable bypass.

    Each of these paths reads vgpr("ValuC+%d"%idx) or uses inputPrefix="ValuC+"
    directly, bypassing _valuCVgpr, so they would read uninitialized registers
    if the acc→ValuC moves are skipped.

    Note: UseScaleAlphaVec and UseScaleAB=Vector previously blocked bypass but
    applyScaleVec was updated to use _valuCVgpr for all src/dst, so they are now
    bypass-safe.
    """

    def test_activation_func_call(self):
        """ActivationFuncCall copies through _valuCVgpr, so it is bypass-safe."""
        k = _base_kernel()
        k["ActivationFuncCall"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_bias_read_no_compute_type_disabled(self):
        """Bias read without a known compute type stays conservatively disabled."""
        assert _can_bypass_valu_c(_base_kernel(), edge=False, atomic=False,
                                  use_bias=DataDirection.READ) is False

    def test_bias_read_f32_compute_bf16_dest_allowed(self):
        """FP4->BF16 bias read: the VAddF32/VAddPKF32 add reads src1 via _valuCVgpr.

        ComputeDataType=Single, DestDataType=BF16 (with HPA) -> bypass-safe.
        This is the target subtile FP4->BF16 + bias kernel.
        """
        ct = _make_compute_type(is_single=True)
        dest = _make_dest_type(is_bf16=True)
        k = _base_kernel(dest_type=dest, compute_type=ct)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.READ) is True

    def test_bias_read_f32_compute_f32_dest_allowed(self):
        """FP4->F32 bias read: F32 dest has no pack step; bias add uses _valuCVgpr."""
        ct = _make_compute_type(is_single=True)
        dest = _make_dest_type(is_single=True)
        k = _base_kernel(dest_type=dest, compute_type=ct)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.READ) is True

    def test_bias_read_int32_dest_disabled(self):
        """Int32 dest bias read: convertData(CVT_I32_to_F32) reads raw "ValuC+"."""
        ct = _make_compute_type(is_single=True)
        dest = _make_dest_type(is_int32=True)
        k = _base_kernel(dest_type=dest, compute_type=ct)
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.READ) is False

    def test_bias_read_int8_dest_disabled(self):
        """Int8 dest bias read: convertData reads raw "ValuC+" before the add."""
        ct = _make_compute_type(is_single=True)
        dest = _make_dest_type(is_int8=True)
        k = _base_kernel(dest_type=dest, compute_type=ct)
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.READ) is False

    def test_bias_read_int8_in_bf16_dest_disabled(self):
        """Int8 input + BF16 dest bias read: convertData path reads raw "ValuC+"."""
        ct = _make_compute_type(is_single=True)
        dest = _make_dest_type(is_bf16=True)
        dtype = MagicMock()
        dtype.isInt8.return_value = True
        k = _base_kernel(dest_type=dest, compute_type=ct)
        k["ProblemType"]["DataType"] = dtype
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.READ) is False

    def test_bias_write_is_allowed(self):
        """Writing bias (e.g. BiasSrc=D) does not read ValuC via raw path."""
        assert _can_bypass_valu_c(_base_kernel(), edge=False, atomic=False,
                                  use_bias=DataDirection.WRITE) is True

    def test_use_scale_alpha_vec_bypass_allowed(self):
        """applyScaleVec was updated to use _valuCVgpr — bypass is now safe."""
        k = _base_kernel()
        k["ProblemType"]["UseScaleAlphaVec"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_use_scale_cd(self):
        """scaleDModule uses raw vgpr("ValuC+%d") for ScaleD multiply."""
        k = _base_kernel()
        k["ProblemType"]["UseScaleCD"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_use_e_output(self):
        """E-output pack path uses raw ValuC prefix (non-gradient UseE)."""
        k = _base_kernel()
        k["ProblemType"]["UseE"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_use_scale_ab_vector_bypass_allowed(self):
        """applyScaleVec was updated to use _valuCVgpr — ScaleAB Vector bypass is safe."""
        k = _base_kernel()
        k["ProblemType"]["UseScaleAB"] = "Vector"
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_use_scale_ab_scalar_is_allowed(self):
        """Scalar ScaleAB does not use the applyScaleVec path."""
        k = _base_kernel()
        k["ProblemType"]["UseScaleAB"] = "Scalar"
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True


class TestBypassDisabledForHpaPackData:
    """HPA pack/convert paths use inputPrefix="ValuC+" for FP8/BF8/Int8/Int32 dest.

    F32 (Single) has NO pack step in the HPA epilogue, so all its code paths
    already use _valuCVgpr / _storeSumIdx and the bypass is safe.
    Half and BFloat16 are protected by the is16bitSubtile packdata skip
    (packdata is not called for subtile non-edge 16-bit stores).
    """

    def test_hpa_fp8_disabled(self):
        """FP8 + HPA: packdata uses raw inputPrefix="ValuC+" → bypass blocked."""
        dest = _make_dest_type(is_half=False, is_bf16=False, is_single=False)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_hpa_int8_disabled(self):
        """Int8 + HPA: convertData+packdata use raw inputPrefix="ValuC+" → blocked."""
        dest = _make_dest_type(is_half=False, is_bf16=False, is_single=False)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_hpa_half_allowed(self):
        """Half + HPA: packdata is skipped by is16bitSubtile, so bypass is safe."""
        dest = _make_dest_type(is_half=True, is_bf16=False)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_hpa_bf16_allowed(self):
        """BFloat16 + HPA: same is16bitSubtile protection applies."""
        dest = _make_dest_type(is_half=False, is_bf16=True)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_hpa_f32_dest_beta0_allowed(self):
        """F32 dest + HPA + beta=0: no pack step exists; all paths use _valuCVgpr.

        This is the FP4→F32 kernel case: DataType=FP4, DestDataType=Single,
        ComputeDataType=Single, HPA=True, UseSubtileImpl=1, beta=0.
        Bypass MUST be active so that no v_mov acc→ValuC instructions appear.
        """
        dest = _make_dest_type(is_half=False, is_bf16=False, is_single=True)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE, beta=False) is True

    def test_hpa_f32_dest_beta_nonzero_allowed(self):
        """F32 dest + HPA + beta≠0: _addSumAlphaWithCBeta isSingle() path uses _valuCVgpr.

        The VMacF32 branch for isSingle() was updated to use self._valuCVgpr()
        so both beta=0 and beta≠0 are safe for F32 dest.
        """
        dest = _make_dest_type(is_half=False, is_bf16=False, is_single=True)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE, beta=True) is True

    def test_no_hpa_non16bit_allowed(self):
        """Without HPA the packdata block is not entered at all — safe."""
        dest = _make_dest_type(is_half=False, is_bf16=False, is_single=False)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = False
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True


# ---------------------------------------------------------------------------
# Combinations / interaction checks
# ---------------------------------------------------------------------------

class TestBypassCombinations:
    """Any single disqualifying condition is enough to prevent the bypass."""

    @pytest.mark.parametrize("flag,value", [
        ("UseScaleCD",  True),
        ("UseE",        True),
        ("Gradient",    True),
    ])
    def test_single_problem_type_flag_disables(self, flag, value):
        k = _base_kernel()
        k["ProblemType"][flag] = value
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    @pytest.mark.parametrize("flag,value", [
        ("UseScaleAlphaVec", True),
        ("UseScaleAB",       "Vector"),
    ])
    def test_scale_vec_flags_allow_bypass(self, flag, value):
        """applyScaleVec uses _valuCVgpr — these flags no longer block bypass."""
        k = _base_kernel()
        k["ProblemType"][flag] = value
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_all_clear_is_bypass(self):
        """Explicitly verify the all-clear case matches the plain GEMM expectation."""
        k = {
            "UseSubtileImpl": True,
            "LocalSplitU": 1,
            "ActivationFuncCall": False,
            "WorkGroupReduction": False,
            "ProblemType": {
                "Gradient": False,
                "UseScaleAlphaVec": False,
                "UseScaleCD": False,
                "UseE": False,
                "UseScaleAB": "None",
                "HighPrecisionAccumulate": False,
                "DestDataType": _make_dest_type(),
            },
        }
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True


# ---------------------------------------------------------------------------
# storeBiasD path (biasReductionModule)
# ---------------------------------------------------------------------------

class TestBypassDisabledForBiasStore:
    """biasReductionModule uses raw "ValuC+%d" in addStore when storeBiasD==1.

    storeBiasD==1 fires when: useBias==WRITE and BiasSrc=="D" and not
    WorkGroupReduction.  The other write combinations (BiasSrc!=D, or
    WorkGroupReduction==True) do NOT take that code path and remain safe.
    """

    def test_bias_write_biasSrc_D_no_wgr_disabled(self):
        """storeBiasD==1 path: biasReductionModule reads raw "ValuC+%d"."""
        k = _base_kernel()
        k["ProblemType"]["BiasSrc"] = "D"
        k["WorkGroupReduction"] = False
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.WRITE) is False

    def test_bias_write_biasSrc_D_with_wgr_allowed(self):
        """WorkGroupReduction==True prevents storeBiasD==1; path is safe."""
        k = _base_kernel()
        k["ProblemType"]["BiasSrc"] = "D"
        k["WorkGroupReduction"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.WRITE) is True

    def test_bias_write_biasSrc_A_allowed(self):
        """BiasSrc=A does not trigger storeBiasD; path is safe."""
        k = _base_kernel()
        k["ProblemType"]["BiasSrc"] = "A"
        k["WorkGroupReduction"] = False
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.WRITE) is True

    def test_bias_write_no_biasSrc_key_allowed(self):
        """Missing BiasSrc key (not set) does not trigger storeBiasD."""
        k = _base_kernel()
        k["WorkGroupReduction"] = False
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.WRITE) is True


# ---------------------------------------------------------------------------
# Half/BF16 dest + beta (_addSumAlphaWithCBeta coverage)
# ---------------------------------------------------------------------------

class TestBypassDisabledForHalfBf16Beta:
    """Non-HPA Half beta remains unsafe; HPA Half/BF16 beta is bypass-safe."""

    def test_half_dest_beta_disabled(self):
        """Non-HPA Half + beta=True: VAddPKF16 still reads raw "ValuC+%u"."""
        dest = _make_dest_type(is_half=True)
        assert _can_bypass_valu_c(_base_kernel(dest_type=dest),
                                  edge=False, atomic=False,
                                  use_bias=DataDirection.NONE,
                                  beta=True) is False

    def test_bf16_dest_beta_allowed(self):
        """BFloat16 + beta=True: VMacF32 path uses _valuCVgpr."""
        dest = _make_dest_type(is_bf16=True)
        assert _can_bypass_valu_c(_base_kernel(dest_type=dest),
                                  edge=False, atomic=False,
                                  use_bias=DataDirection.NONE,
                                  beta=True) is True

    def test_hpa_half_dest_beta_allowed(self):
        """HPA Half + beta=True: mixinst path uses _valuCVgpr."""
        dest = _make_dest_type(is_half=True)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE,
                                  beta=True) is True

    def test_half_dest_beta_false_allowed(self):
        """Half + beta=False: beta path not entered; only alpha+D-store used.

        For HPA Half, those paths are updated; bypass is safe.
        """
        dest = _make_dest_type(is_half=True)
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE,
                                  beta=False) is True

    def test_fp32_dest_beta_allowed(self):
        """F32 dest + beta=True: VMacF32 path IS updated via _valuCVgpr."""
        dest = _make_dest_type(is_half=False, is_bf16=False, is_single=True)
        assert _can_bypass_valu_c(_base_kernel(dest_type=dest),
                                  edge=False, atomic=False,
                                  use_bias=DataDirection.NONE,
                                  beta=True) is True

    def test_beta_default_is_false(self):
        """Callers that omit beta= get beta=False (backward-compatible)."""
        dest = _make_dest_type(is_half=True)
        # Without explicit beta=True the Half dest does NOT disable bypass
        # (beta defaults to False, so the beta-path guard is inactive).
        k = _base_kernel(dest_type=dest)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True


# ---------------------------------------------------------------------------
# Non-HPA Half and Int32 compute (_applyAlpha paths not updated)
# ---------------------------------------------------------------------------

class TestBypassDisabledForAlphaNonF32:
    """_applyAlpha VMulPKF16 (non-HPA Half compute) and VMulLOU32 (Int32 compute)
    paths are not updated; they still read raw "ValuC+%u".
    """

    def test_non_hpa_half_compute_disabled(self):
        """Non-HPA Half compute: _applyAlpha VMulPKF16 reads raw ValuC."""
        ct = _make_compute_type(is_half=True)
        k = _base_kernel(compute_type=ct)
        k["ProblemType"]["HighPrecisionAccumulate"] = False
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_hpa_half_compute_allowed(self):
        """HPA Half compute: _applyAlpha takes the F32-equivalent path (updated)."""
        ct = _make_compute_type(is_half=True)
        dest = _make_dest_type(is_half=True)
        k = _base_kernel(dest_type=dest, compute_type=ct)
        k["ProblemType"]["HighPrecisionAccumulate"] = True
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True

    def test_int32_compute_disabled(self):
        """Int32 compute: _applyAlpha VMulLOU32 reads raw "ValuC+%u"."""
        ct = _make_compute_type(is_int32=True)
        assert _can_bypass_valu_c(_base_kernel(compute_type=ct),
                                  edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is False

    def test_no_compute_type_key_allowed(self):
        """Missing ComputeDataType key: guard is skipped, bypass permitted."""
        k = _base_kernel()   # no ComputeDataType in ProblemType
        assert _can_bypass_valu_c(k, edge=False, atomic=False,
                                  use_bias=DataDirection.NONE) is True
