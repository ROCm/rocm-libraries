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

"""Characterization tests for ``Tensile.SolutionStructs.Problem.ProblemType`` —
its construction (`__init__` dtype-defaulting + HPA conversions + GEMM type
check + `initGEMM` + `assignDerivedParameters`), the very branchy `__str__`
naming, the ``Mapping`` interface, equality/hash, ``FromDefaultConfig``, and the
constructor's raise paths.

Each feature config overlays a small override on the default ProblemType; the
snapshot is ``{str, normalized state}`` (state rendered object-free by the
``norm`` fixture). The configs are chosen to walk the naming/index branches.
"""

import pytest

from Tensile.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit


# Each entry: name -> config overrides on top of _defaultProblemType. All are
# constructed with a valid GEMM type so __init__ completes.
_CONFIGS = {
    "default": {},
    "batched": {"Batched": True},
    "transpose_nn": {"TransposeA": False, "TransposeB": False},
    "transpose_tt": {"TransposeA": True, "TransposeB": True},
    "hhs_bh": {"DataType": "H", "DestDataType": "H", "ComputeDataType": "S",
               "HighPrecisionAccumulate": True},
    "hhh_hpa_converted": {"DataType": "H", "DestDataType": "H", "ComputeDataType": "H",
                          "HighPrecisionAccumulate": True},
    "bbb_hpa_converted": {"DataType": "B", "DestDataType": "B", "ComputeDataType": "B",
                          "HighPrecisionAccumulate": True},
    "i8_hpa_converted": {"DataType": "I8", "DestDataType": "I8", "ComputeDataType": "I8",
                         "HighPrecisionAccumulate": True},
    "silent_hpa": {"DataType": "H", "DestDataType": "H", "ComputeDataType": "S",
                   "HighPrecisionAccumulate": True, "SilentHighPrecisionAccumulate": True},
    "complex_single": {"DataType": "C", "DestDataType": "C", "ComputeDataType": "C"},
    "complex_conj": {"DataType": "C", "DestDataType": "C", "ComputeDataType": "C",
                     "ComplexConjugateA": True, "ComplexConjugateB": True},
    "double": {"DataType": "D", "DestDataType": "D", "ComputeDataType": "D"},
    "bias_m": {"UseBias": 1},
    "bias_n": {"UseBias": 2},
    "bias_mn": {"UseBias": 3},
    "bias_with_list": {"UseBias": 1, "BiasDataTypeList": ["s", "h"]},
    "activation_all": {"Activation": True, "ActivationType": "all"},
    "activation_hipblaslt": {"Activation": True, "ActivationType": "hipblaslt_all"},
    "activation_unknown_to_none": {"Activation": True, "ActivationType": "relu"},
    "use_e": {"Activation": True, "ActivationType": "all", "UseE": True},
    "gradient": {"Activation": True, "ActivationType": "all", "Gradient": True,
                 "UseBias": 1, "BiasSrc": "A"},
    "activation_noguard": {"Activation": True, "ActivationType": "all", "Gradient": True,
                           "UseBias": 1, "ActivationNoGuard": True},
    "hhs_bias": {"DataType": "H", "DestDataType": "H", "ComputeDataType": "S",
                 "HighPrecisionAccumulate": True, "UseBias": 1},
    "bias_src_a_no_gradient": {"UseBias": 1, "BiasSrc": "A"},
    "activation_compute_dtype": {"Activation": True, "ActivationType": "all",
                                 "ActivationComputeDataType": "s"},
    "use_e_no_activation": {"UseE": True},
    "gradient_disabled_no_bias": {"Gradient": True},
    "gradient_use_e_no_activation": {"Gradient": True, "UseE": True},
    "activation_compute_mismatch": {"Activation": True, "ActivationType": "all",
                                    "ActivationComputeDataType": "d"},
    "activation_noguard_off": {"ActivationNoGuard": True},
    "activation_noguard_grad_off": {"Activation": True, "ActivationType": "all",
                                    "ActivationNoGuard": True},
    "sparse_a": {"Sparse": 1},
    "sparse_a_metalayout": {"Sparse": 1, "MetadataLayout": 1},
    "sparse_b_no_metalayout": {"Sparse": 2},
    "sparse_b_metalayout": {"Sparse": 2, "MetadataLayout": 1},
    "mxblock": {"MXBlockA": 32, "MXBlockB": 32},
    "scale_ab_scalar": {"UseScaleAB": "Scalar"},
    "scale_ab_vector": {"UseScaleAB": "Vector"},
    "scale_cd": {"UseScaleCD": True},
    "scale_alpha_vec": {"UseScaleAlphaVec": 1},
    "allow_no_free_dims": {"AllowNoFreeDims": True},
    "activation_compute_regsize": {"DataType": "F8", "DestDataType": "H",
                                   "ComputeDataType": "S", "Activation": True,
                                   "ActivationType": "all", "ActivationComputeDataType": "h"},
    "f32xdl_mathop": {"F32XdlMathOp": "x"},
    "output_amaxd": {"OutputAmaxD": True},
    "grouped_gemm": {"GroupedGemm": True},
    "general_batched": {"StridedBatched": False},
    "swizzle": {"SwizzleTensorA": True, "SwizzleTensorB": True},
    "initial_strides": {"UseInitialStridesAB": True, "UseInitialStridesCD": True},
}


@pytest.mark.parametrize("name", list(_CONFIGS), ids=list(_CONFIGS))
def test_problem_type_construction(name, make_pt, norm, snapshot):
    pt = make_pt(**_CONFIGS[name])
    assert {"str": str(pt), "state": norm(pt)} == snapshot


# ===========================================================================
# printIndexAssignmentInfo=True — exercise the print branches in
# assignDerivedParameters / initGEMM index reporting
# ===========================================================================

def test_construction_with_index_print(make_pt, norm, snapshot):
    pt = make_pt(printInfo=True)
    assert norm(pt) == snapshot


def test_construction_sparse_with_index_print(make_pt, norm, snapshot):
    pt = make_pt(printInfo=True, Sparse=1)
    assert norm(pt) == snapshot


def test_assign_derived_parameters_early_return(make_pt):
    # A constructed ProblemType already has AssignedDerivedParameters=True;
    # calling the static method again returns immediately (the guard).
    pt = make_pt()
    before = dict(pt.state)
    ProblemType.assignDerivedParameters(pt.state, False)
    assert pt.state == before


# ===========================================================================
# Mapping interface + identity dunders
# ===========================================================================

def test_mapping_interface(make_pt, snapshot):
    pt = make_pt()
    summary = {
        "len": len(pt),
        "is_mapping_iter_sorted": sorted(iter(pt)) == sorted(pt.keys()),
        "getitem_datatype": pt["DataType"].toName(),
        "get_existing": pt.get("OperationType"),
        "get_missing_default": pt.get("NoSuchKey", "DEFAULT"),
        "keys_is_list": isinstance(pt.keys(), list),
        "getattributes_is_state": pt.getAttributes() is pt.state,
    }
    assert summary == snapshot


def test_setitem(make_pt):
    pt = make_pt()
    pt["OperationType"] = "GEMM"  # round-trips through __setitem__/__getitem__
    assert pt["OperationType"] == "GEMM"


def test_equality_and_hash(make_pt):
    a = make_pt()
    b = make_pt()
    c = make_pt(Batched=True)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    # __eq__ against a non-ProblemType returns False (isinstance guard).
    assert (a == "not a problem type") is False
    assert (a != "not a problem type") is True
    # hash is hash(str(self)).
    assert hash(a) == hash(str(a))


def test_str_repr_agree(make_pt):
    pt = make_pt()
    assert repr(pt) == str(pt)


def test_from_default_config_returns_problem_type():
    # NOTE: FromDefaultConfig is a @classmethod whose only parameter receives
    # `cls` (a latent signature quirk), so it is called with no explicit args.
    pt = ProblemType.FromDefaultConfig()
    assert isinstance(pt, ProblemType)
    assert pt["OperationType"] == "GEMM"


# ===========================================================================
# __init__ raise paths
# ===========================================================================

def test_no_data_type_raises(snapshot):
    # A config with no "DataType" key at all -> the first raise in __init__.
    with pytest.raises(Exception) as excinfo:
        ProblemType({}, False)
    assert str(excinfo.value) == snapshot


def test_unsupported_gemm_type_raises(snapshot):
    cfg = {"DataType": "S", "DestDataType": "D", "ComputeDataType": "S"}  # (S,S,D,S)
    with pytest.raises(Exception) as excinfo:
        ProblemType(cfg, False)
    assert str(excinfo.value) == snapshot


def test_unsupported_operation_type_raises(snapshot):
    cfg = {"DataType": 0, "OperationType": "ConvolutionForward"}
    with pytest.raises(Exception) as excinfo:
        ProblemType(cfg, False)
    assert str(excinfo.value) == snapshot


def test_set_const_stride_bad_anchor_raises(snapshot):
    cfg = {"DataType": 0, "SetConstStrideA": [[9, 1]]}  # 9 not in IndexAssignmentsA
    with pytest.raises(Exception) as excinfo:
        ProblemType(cfg, False)
    assert str(excinfo.value) == snapshot


def test_bad_bias_src_with_gradient_raises(snapshot):
    cfg = {"DataType": 0, "Activation": True, "ActivationType": "all",
           "Gradient": True, "UseBias": 1, "BiasSrc": "Z"}
    with pytest.raises(Exception) as excinfo:
        ProblemType(cfg, False)
    assert str(excinfo.value) == snapshot
