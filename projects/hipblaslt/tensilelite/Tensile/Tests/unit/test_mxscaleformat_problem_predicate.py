################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""MXScaleFormat matching predicates must stay API-layout only (0/1).

gfx950 split Origami libs name ProblemType.MXScaleFormat as NoSwizzle or
HostPreSwizzle so host 0/1 selects the right library. gfx1250 (and most
libraries) omit ProblemType.MXScaleFormat and keep solution-level
InMemorySwizzle for kernels — matching must not require host==2.
"""

import copy

import pytest

import Tensile.Contractions as C

pytestmark = pytest.mark.unit

# Minimal GEMM ProblemType dict accepted by ProblemType.FromOriginalState.
_BASE_PT = {
    "TotalIndices": 4,
    "NumIndicesC": 3,
    "IndicesBatch": [2],
    "IndicesFree": [0, 1],
    "IndicesSummation": [3],
    "IndexAssignmentsA": [3, 0, 2],
    "IndexAssignmentsB": [1, 3, 2],
    "ComplexConjugateA": False,
    "ComplexConjugateB": False,
    "TransposeA": 1,
    "TransposeB": 1,
    "DataType": 0,
    "DestDataType": 0,
    "ComputeDataType": 0,
    "ActivationComputeDataType": 0,
    "Batched": True,
    "StridedBatched": True,
    "UseBeta": True,
    "F32XdlMathOp": 0,
    "UseScaleAlphaVec": 0,
    "UseBias": 0,
    "BiasDataTypeList": [],
    "BiasSrc": "D",
}


def _pt(**overrides):
    d = copy.deepcopy(_BASE_PT)
    d.update(overrides)
    return C.ProblemType.FromOriginalState(d)


def _mx_preds(problem_type):
    return [
        p
        for p in problem_type.predicates(includeBatch=True, includeOperation=True, includeType=True)
        if p.tag == "MXScaleFormat"
    ]


def test_omitted_problemtype_mxscaleformat_emits_no_predicate():
    """gfx1250-style: no ProblemType.MXScaleFormat → no matching key."""
    assert _mx_preds(_pt()) == []


def test_noswizzle_problemtype_emits_predicate_0():
    preds = _mx_preds(_pt(MXScaleFormat="NoSwizzle"))
    assert len(preds) == 1 and preds[0].value == 0


def test_hostpreswizzle_problemtype_emits_predicate_1():
    preds = _mx_preds(_pt(MXScaleFormat="HostPreSwizzle"))
    assert len(preds) == 1 and preds[0].value == 1


@pytest.mark.parametrize("fmt", ["NoSwizzle", 0, "HostPreSwizzle", 1])
def test_api_layout_int_or_name_is_discriminator(fmt):
    preds = _mx_preds(_pt(MXScaleFormat=fmt))
    expected = fmt if isinstance(fmt, int) else {"NoSwizzle": 0, "HostPreSwizzle": 1}[fmt]
    assert len(preds) == 1 and preds[0].value == expected


def test_auto_on_problemtype_maps_to_noswizzle_discriminator():
    """Auto is host-layout 0 (NoSwizzle); still an API matching key."""
    pt = _pt(MXScaleFormat="Auto")
    assert pt.mxScaleFormat == 0
    assert pt._mxScaleFormatApiDiscriminator is True
    preds = _mx_preds(pt)
    assert len(preds) == 1 and preds[0].value == 0


@pytest.mark.parametrize("fmt", ["InMemorySwizzle", 2])
def test_in_memory_swizzle_on_problemtype_is_not_a_matching_key(fmt):
    """Even if YAML wrongly put IMS on ProblemType, never bake host==2."""
    assert _mx_preds(_pt(MXScaleFormat=fmt)) == []

def test_solution_ims_must_not_poison_predicates_before_build():
    """Ordering contract for Solution.FromOriginalState.

    Solution-level InMemorySwizzle must not be copied onto problemType before
    predicates are built. After predicates, mirroring IMS for DataInit is fine
    and must still leave matching unaffected.
    """
    pt_dict = copy.deepcopy(_BASE_PT)
    assert "MXScaleFormat" not in pt_dict

    problem_type = C.ProblemType.FromOriginalState(pt_dict)
    assert problem_type._mxScaleFormatApiDiscriminator is False
    assert _mx_preds(problem_type) == []

    # Post-predicate DataInit mirror (Solution.FromOriginalState does this).
    problem_type.mxScaleFormat = 2
    assert problem_type._mxScaleFormatApiDiscriminator is False
    assert _mx_preds(problem_type) == []


def test_inmemory_swizzle_problemtype_sets_no_api_discriminator():
    pt = _pt(MXScaleFormat="InMemorySwizzle")
    assert pt.mxScaleFormat == 2
    assert pt._mxScaleFormatApiDiscriminator is False
    assert _mx_preds(pt) == []
