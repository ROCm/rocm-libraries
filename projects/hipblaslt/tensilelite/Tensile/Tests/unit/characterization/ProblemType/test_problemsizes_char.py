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

"""Characterization tests for the problem-size holders in
``Tensile.SolutionStructs.Problem``: ``ProblemSizeRange``, ``Problem``,
``ExactList`` / ``ExactDict``, ``ProblemSizesMock`` / ``ProblemSizesMockDummy``,
and ``ProblemSizes``. Driven against a default ``ProblemType`` with small
configs; ``printExit`` paths (``sys.exit(-1)``) are pinned via
``pytest.raises(SystemExit)``.
"""

import pytest

from Tensile.SolutionStructs.Problem import (
    ProblemType,
    _defaultProblemType,
    ProblemSizeRange,
    Problem,
    ExactList,
    ExactDict,
    ProblemSizes,
    ProblemSizesMock,
    ProblemSizesMockDummy,
)

pytestmark = pytest.mark.unit

# A default GEMM ProblemType: TotalIndices=3, NumIndicesLD=4, IndexAssignmentsA
# =[0,2], IndexAssignmentsB=[1,2]. These holders only read it.
_PT = ProblemType(_defaultProblemType, False)


def _range_summary(psr):
    return {
        "str": str(psr),
        "totalIndices": psr.totalIndices,
        "totalProblemSizes": psr.totalProblemSizes,
        "numProblemSizes": psr.numProblemSizes,
        "indexIsSized": psr.indexIsSized,
        "problemSizes": [list(s) for s in psr.problemSizes],
    }


# ===========================================================================
# ProblemSizeRange
# ===========================================================================

def test_problem_size_range_sized(snapshot):
    cfg = [[128, 128, 128], [128, 128, 128], [64, 64, 64]]  # padded to 7 internally
    psr = ProblemSizeRange(_PT, cfg)
    assert _range_summary(psr) == snapshot


def test_problem_size_range_with_mapped_index(snapshot):
    # An int index is a "mapped" index (references an earlier sized index).
    cfg = [[256, 256, 256], 0, [64, 64, 64]]
    psr = ProblemSizeRange(_PT, cfg)
    assert _range_summary(psr) == snapshot


def test_problem_size_range_multi_size(snapshot):
    # A [min, step, max] descriptor that yields several sizes.
    cfg = [[64, 64, 256], [128, 128, 128], [32, 32, 32]]
    psr = ProblemSizeRange(_PT, cfg)
    assert _range_summary(psr) == snapshot


def test_problem_size_range_descriptor_lengths(snapshot):
    # len-2 [min,max] and len-4 [min,step,stepInc,max] descriptors.
    cfg = [[128, 512], [64, 64, 0, 256], [32, 32, 32]]
    psr = ProblemSizeRange(_PT, cfg)
    assert _range_summary(psr) == snapshot


def test_problem_size_range_too_many_descriptors_exits():
    # A descriptor with >4 entries -> printExit.
    with pytest.raises(SystemExit):
        ProblemSizeRange(_PT, [[1, 2, 3, 4, 5], [64], [32]])


def test_problem_size_range_short_config_padded(snapshot):
    # A config shorter than the 3 size indices -> the int-padding branch
    # (append 0) for the missing leading size indices.
    psr = ProblemSizeRange(_PT, [[128]])
    assert _range_summary(psr) == snapshot


# ===========================================================================
# Problem
# ===========================================================================

def test_problem_sizes_only(snapshot):
    assert str(Problem(sizes=[128, 256, 1, 64])) == snapshot


def test_problem_with_strides(snapshot):
    p = Problem(sizes=[128, 256, 1, 64], stridesA=[1, 128], stridesB=[1, 64],
                stridesC=[1, 128], stridesD=[1, 128], count=3)
    assert str(p) == snapshot


# ===========================================================================
# ExactList
# ===========================================================================

def test_exact_list_total_indices(snapshot):
    # len(e) == TotalIndices (3) -> padded with [-1,-1,-1,-1] + convertLeadingDims.
    el = ExactList([128, 128, 64], _PT)
    assert {"str": str(el), "sizes": list(el.sizes)} == snapshot


def test_exact_list_with_ld(snapshot):
    # len(e) == TotalIndices + NumIndicesLD (7) -> convertLeadingDims directly.
    el = ExactList([128, 128, 64, 1, 2, 3, 4], _PT)
    assert {"str": str(el), "sizes": list(el.sizes)} == snapshot


def test_exact_list_contains_minus_one_exits():
    with pytest.raises(SystemExit):
        ExactList([128, -1, 64], _PT)


def test_exact_list_wrong_length_exits():
    with pytest.raises(SystemExit):
        ExactList([128, 128], _PT)


def test_convert_leading_dims_with_explicit_strides(snapshot):
    # Drives the predStrides* True branches of the static helper.
    out = ExactList.convertLeadingDims(
        _PT, (128, 128, 64, -1, -1, -1, -1),
        stridesA=[0, 999], stridesB=[0, 888], stridesC=[0, 777], stridesD=[0, 666],
    )
    assert list(out) == snapshot


# ===========================================================================
# ExactDict
# ===========================================================================

def test_exact_dict_gemm(snapshot):
    ed = ExactDict({"sizes": [128, 128, 64]}, _PT)
    assert {"str": str(ed), "sizes": list(ed.sizes)} == snapshot


def test_exact_dict_no_problem_type(snapshot):
    # problemType=None -> skips the GEMM convert + size-count check branches.
    ed = ExactDict({"sizes": [128, 128, 64]}, None)
    assert list(ed.sizes) == snapshot


def test_exact_dict_bad_field_raises(snapshot):
    with pytest.raises(RuntimeError) as excinfo:
        ExactDict({"not_a_field": 1}, _PT)
    assert str(excinfo.value) == snapshot


def test_exact_dict_non_gemm_wrong_count_raises(snapshot):
    # A non-GEMM problemType (plain dict) with too few size indices -> the
    # non-GEMM size-count RuntimeError (the GEMM branch always normalises to 7
    # via convertLeadingDims, so this is the reachable count-mismatch path).
    fake_pt = {"OperationType": "TensorContraction", "TotalIndices": 5}
    with pytest.raises(RuntimeError) as excinfo:
        ExactDict({"sizes": [128, 128, 64]}, fake_pt)
    assert str(excinfo.value) == snapshot


# ===========================================================================
# ProblemSizesMock / ProblemSizesMockDummy
# ===========================================================================

def test_problem_sizes_mock(snapshot):
    exactLogic = [([128, 128, 1, 64], None), ([256, 256, 1, 128], None)]
    mock = ProblemSizesMock(exactLogic)
    assert [str(p) for p in mock.problems] == snapshot


def test_problem_sizes_mock_dummy(snapshot):
    assert [str(p) for p in ProblemSizesMockDummy().problems] == snapshot


# ===========================================================================
# ProblemSizes
# ===========================================================================

def _ps_summary(ps):
    return {
        "str": str(ps),
        "totalProblemSizes": ps.totalProblemSizes,
        "maxD": ps.maxD, "maxC": ps.maxC, "maxA": ps.maxA, "maxB": ps.maxB,
        "minStrides": list(ps.minStrides),
        "problems": [str(p) for p in ps.problems],
    }


def test_problem_sizes_range_exact_minstride(snapshot):
    cfg = [
        {"Range": [[128, 128, 128], [128, 128, 128], [64, 64, 64]]},
        {"Exact": [256, 256, 128]},
        {"MinStride": [0, 0, 0]},
    ]
    ps = ProblemSizes(_PT, cfg)
    assert _ps_summary(ps) == snapshot


def test_problem_sizes_exact_dict(snapshot):
    cfg = [{"Exact": {"sizes": [128, 128, 64]}}]
    ps = ProblemSizes(_PT, cfg)
    assert _ps_summary(ps) == snapshot


def test_problem_sizes_empty_config(snapshot):
    ps = ProblemSizes(_PT, None)
    assert _ps_summary(ps) == snapshot


def test_problem_sizes_unsupported_size_type_exits():
    with pytest.raises(SystemExit):
        ProblemSizes(_PT, [{"Bogus": []}])


def test_problem_sizes_exact_bad_type_exits():
    with pytest.raises(SystemExit):
        ProblemSizes(_PT, [{"Exact": 5}])


def test_problem_sizes_minstride_wrong_length_exits():
    with pytest.raises(SystemExit):
        ProblemSizes(_PT, [{"MinStride": [0, 0]}])


def test_problem_sizes_duplicate_minstride_exits():
    with pytest.raises(SystemExit):
        ProblemSizes(_PT, [{"MinStride": [0, 0, 0]}, {"MinStride": [0, 0, 0]}])
