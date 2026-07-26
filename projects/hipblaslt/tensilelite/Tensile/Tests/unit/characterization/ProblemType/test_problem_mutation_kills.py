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

"""Mutation-kill tests for ``Tensile.SolutionStructs.Problem``.

These pin behaviour the snapshot characterization tests leave undistinguished:
they use descriptors whose four fields (min/step/stepIncrement/max) are all
distinct and whose step-increment is non-zero, plus multiple mapped indices with
distinct referents, so that argument swaps, index off-by-ones, accumulator seeds,
and increment-sign flips inside ``ProblemSizeRange.__init__`` change an asserted
value. Expected values are ground truth captured from the unmutated module.
"""

import pytest

from Tensile.Activation import ActivationType
from Tensile.SolutionStructs.Problem import (
    ProblemType,
    _defaultProblemType,
    ProblemSizeRange,
    ExactDict,
)

pytestmark = pytest.mark.unit

_PT = ProblemType(_defaultProblemType, False)


def _full_state(psr):
    return {
        "indicesSized": psr.indicesSized,
        "indexIsSized": psr.indexIsSized,
        "indicesMapped": psr.indicesMapped,
        "indexMax": psr.indexMax,
        "maxNumElements": psr.maxNumElements,
        "numProblemSizes": psr.numProblemSizes,
        "totalProblemSizes": psr.totalProblemSizes,
        "problemSizeToIndex": psr.problemSizeToIndex,
        "problemIndexToSize": psr.problemIndexToSize,
        "str": str(psr),
        "problemSizes": [tuple(s) for s in psr.problemSizes],
    }


def test_psr_all_sized_distinct_full_state():
    psr = ProblemSizeRange(_PT, [[64, 128, 16, 512], [32, 96, 8, 256], [16, 48, 80]])
    assert _full_state(psr) == {
        "indicesSized": [
            [64, 128, 16, 512],
            [32, 96, 8, 256],
            [16, 48, 0, 80],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        "indexIsSized": [True, True, True, True, True, True, True],
        "indicesMapped": [],
        "indexMax": [512, 256, 80, 0, 0, 0, 0],
        "maxNumElements": [131072, 40960, 20480],
        "numProblemSizes": [4, 3, 2, 1, 1, 1, 1],
        "totalProblemSizes": 24,
        "problemSizeToIndex": [{}, {}, {}, {}, {}, {}, {}],
        "problemIndexToSize": [{}, {}, {}, {}, {}, {}, {}],
        "str": "[ [ 64, 128, 16, 512 ], [ 32, 96, 8, 256 ], [ 16, 48, 0, 80 ], "
               "[ 0, 1, 0, 0 ], [ 0, 1, 0, 0 ], [ 0, 1, 0, 0 ], [ 0, 1, 0, 0 ] ]",
        "problemSizes": [
            (64, 32, 16, 0, 0, 0, 0), (192, 32, 16, 0, 0, 0, 0),
            (336, 32, 16, 0, 0, 0, 0), (496, 32, 16, 0, 0, 0, 0),
            (64, 128, 16, 0, 0, 0, 0), (192, 128, 16, 0, 0, 0, 0),
            (336, 128, 16, 0, 0, 0, 0), (496, 128, 16, 0, 0, 0, 0),
            (64, 232, 16, 0, 0, 0, 0), (192, 232, 16, 0, 0, 0, 0),
            (336, 232, 16, 0, 0, 0, 0), (496, 232, 16, 0, 0, 0, 0),
            (64, 32, 64, 0, 0, 0, 0), (192, 32, 64, 0, 0, 0, 0),
            (336, 32, 64, 0, 0, 0, 0), (496, 32, 64, 0, 0, 0, 0),
            (64, 128, 64, 0, 0, 0, 0), (192, 128, 64, 0, 0, 0, 0),
            (336, 128, 64, 0, 0, 0, 0), (496, 128, 64, 0, 0, 0, 0),
            (64, 232, 64, 0, 0, 0, 0), (192, 232, 64, 0, 0, 0, 0),
            (336, 232, 64, 0, 0, 0, 0), (496, 232, 64, 0, 0, 0, 0),
        ],
    }


def test_psr_mapped_indices_full_state():
    psr = ProblemSizeRange(_PT, [[256, 300, 256], [64, 80, 96], [16, 40, 24], 0, 1, 2])
    assert _full_state(psr) == {
        "indicesSized": [
            [256, 300, 0, 256],
            [64, 80, 0, 96],
            [16, 40, 0, 24],
            [0, 1, 0, 0],
        ],
        "indexIsSized": [True, True, True, False, False, False, True],
        "indicesMapped": [0, 1, 2],
        "indexMax": [256, 96, 24, 256, 96, 24, 0],
        "maxNumElements": [24576, 6144, 2304],
        "numProblemSizes": [1, 1, 1, 1, 1, 1, 1],
        "totalProblemSizes": 1,
        "problemSizeToIndex": [{}, {}, {}, {}, {}, {}, {}],
        "problemIndexToSize": [{}, {}, {}, {}, {}, {}, {}],
        "str": "[ [ 256, 300, 0, 256 ], [ 64, 80, 0, 96 ], [ 16, 40, 0, 24 ], "
               "0, 1, 2, [ 0, 1, 0, 0 ] ]",
        "problemSizes": [(256, 64, 16, 256, 64, 16, 0)],
    }


def test_psr_too_many_descriptors_fatal_message(capsys):
    with pytest.raises(SystemExit):
        ProblemSizeRange(_PT, [[1, 2, 3, 4, 5], [64], [32]])
    out = capsys.readouterr().out.strip()
    assert out == (
        "Tensile::FATAL: dimension[0] config ([1, 2, 3, 4, 5]) "
        "has 5 descriptors rather than 1-4."
    )


def test_exactdict_explicit_strides_used_in_leading_dims():
    ed = ExactDict(
        {
            "sizes": [128, 128, 64],
            "stridesA": [0, 111],
            "stridesB": [0, 222],
            "stridesC": [0, 333],
            "stridesD": [0, 444],
        },
        _PT,
    )
    assert list(ed.sizes) == [128, 128, 64, 444, 333, 111, 222]


def test_exactdict_zero_size_leading_dim_placeholders():
    ed = ExactDict({"sizes": [0, 0, 0]}, _PT)
    assert list(ed.sizes) == [0, 0, 0, 0, 0, 0, 0]


def test_exactdict_leading_dim_placeholder_sentinel_value():
    ed = ExactDict({"sizes": [-5, -5, 0]}, _PT)
    assert list(ed.sizes) == [-5, -5, 0, -1, -1, -1, -1]


def test_exactdict_gemm_size_count_mismatch_message():
    fake_gemm = {
        "OperationType": "GEMM",
        "TotalIndices": 99,
        "NumIndicesLD": 4,
        "NumIndicesC": 2,
        "IndexAssignmentsLD": [3, 4, 5, 6],
        "IndexAssignmentsA": [0, 2],
        "IndexAssignmentsB": [1, 2],
    }
    with pytest.raises(RuntimeError) as excinfo:
        ExactDict({"sizes": [1, 2, 3]}, fake_gemm)
    assert str(excinfo.value) == (
        "specified size=(1, 2, 3, 1, 1, 1, 2) does not have enough indices "
        "for problem (expected 103, got 7)"
    )


def test_fromdefaultconfig_prints_index_assignment_info(capsys):
    ProblemType.FromDefaultConfig()
    out = capsys.readouterr().out
    assert "IndicesFree:" in out
    assert "IndexAssignmentsA:" in out


def test_problemtype_str_default_kernel_name():
    pt = ProblemType(_defaultProblemType, False)
    assert str(pt) == "Cij_Aik_Bjk_S_B_UserArgs"


def test_problemtype_str_named_activation_uppercased():
    pt = ProblemType(_defaultProblemType, False)
    pt["ActivationType"] = ActivationType("relu")
    assert str(pt) == "Cij_Aik_Bjk_S_B_RELU_S_UserArgs"
