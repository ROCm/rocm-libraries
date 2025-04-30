################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import logging,pytest
from Tensile.SolutionStructs import ExactList
from Tensile.SolutionStructs import ExactDict
log =logging.getLogger("testlog")

@pytest.mark.parametrize("test_input, expected", [
    ([1,2,3,4], (1,2,3,4, 1,1,1,)),
    ([4,5,6,7], (4,5,6,7, 4,4,4)),
    ])
def test_exact_list(test_input, expected):
    z={
        "TotalIndices" : len(test_input),
        "OperationType" : 'GEMM',
        "NumIndicesC" : 2,
        "IndexAssignmentsLD" : [3,4,5,6],
        "NumIndicesLD" : 4,
        "IndexAssignmentsA" : [0,1,3],
        "IndexAssignmentsB" : [0,2,3],
        "ZeroPadA" : [],
        "ZeroPadB" : []
    }
    e = ExactList(test_input, z)
    assert(e.sizes == expected)

@pytest.mark.parametrize("test_input, expected", [
    ([1,2,3,4], (1,2,3,4, 1,1,1)),
    ([4,5,6,7], (4,5,6,7 ,4,4,4)),
    ([43,56,4,8], (43,56,4, 43,43,43,43)),
    ])
def test_exact_list_zp(test_input, expected):
    z={
        "TotalIndices" : len(test_input),
        "OperationType" : 'GEMM',
        "NumIndicesC" : 2,
        "IndexAssignmentsLD" : [3,4,5,6],
        "NumIndicesLD" : 4,
        "IndexAssignmentsA" : [0,1,3],
        "IndexAssignmentsB" : [0,2,3],
        "ZeroPadA" : [[1,4,5,7]],
        "ZeroPadB" : [[1,2,44,55]],
    }

    e = ExactList(test_input, z)
    assert(e.sizes == expected)
    assert(e.zeroPadA == [[1,4,5,7]])
    assert(e.zeroPadB == [[1,2,44,55]])


def test_exact_dict():
    z={
        "TotalIndices" : 3,
        "ZeroPadA" : [[1,2,  3,4]],
        "ZeroPadB" : []
    }

    e = ExactDict({'sizes': [100,200,300]}, z)
    assert(e.sizes == [100,200,300])
    assert(e.stridesA == None)
    assert(e.zeroPadA == [[1,2,3,4]])
    assert(e.zeroPadB == [])

def test_exact_dict2():
    e = ExactDict({'sizes': [100,200,300], 'stridesA' : [-1,100,1000]}, None)
    assert(e.sizes == [100,200,300])
    assert(e.stridesA == [-1,100,1000])
    assert(e.zeroPadA == [])

def test_exact_padstart_left_tbd():
    z={
        "TotalIndices" : 3,
       "ZeroPadA" : [[1,2,  -1,-1]], # TBD pads should be specified at end:
       "ZeroPadB" : []
    }
    #with pytest.raises(RuntimeError, match="RuntimeError:.*"):
    with pytest.raises(RuntimeError):
        ExactDict({'sizes': [100,200,300], 'stridesA' : [-1,100,1000] }, z)

def test_exact_padstart_mismatch():
    z={
        "TotalIndices" : 3,
        "ZeroPadA" : [[1,2, 3,4]],
        "ZeroPadB" : []
    }

    with pytest.raises(RuntimeError, match="problem-specified padStartA==6 does not match problem-type==3"):
        ExactDict({'sizes': [100,200,300], 'padStartA':[6]}, z)

def test_exact_padend_mismatch():
    z={
        "TotalIndices" : 3,
        "ZeroPadA" : [[1,2, 3,4]],
        "ZeroPadB" : []
    }

    with pytest.raises(RuntimeError, match="problem-specified padEndA==7 does not match problem-type==4"):
        ExactDict({'sizes': [100,200,300], 'padEndA':[7]}, z)

def test_exact_dict3():
    z={
        "TotalIndices" : 3,
        "ZeroPadA" : [[1,2, 6,4]],
        "ZeroPadB" : []
    }

    e = ExactDict({'sizes': [100,200,300], 'padStartA':[6]}, z)
    assert(e.sizes == [100,200,300])
    assert(e.zeroPadA == [[1,2, 6,4]])
    assert(e.stridesA == None)

def test_exact_override_pad_tbd():
    z={
        "TotalIndices" : 3,
        "ZeroPadA" : [[1,2, -1,-1]],
        "ZeroPadB" : []
    }

    e = ExactDict({'sizes': [100,200,300], 'padStartA':[6], 'padEndA':[7]}, z)
    assert(e.sizes == [100,200,300])
    assert(e.zeroPadA == [[1,2, 6,7]])
    assert(e.stridesA == None)

def test_exact_override_pad_tbd_multi_pad():
    z={
        "TotalIndices" : 3,
        "ZeroPadA" : [[1,2, -1,-1], [4,5, -1, -1]],
        "ZeroPadB" : []
    }

    e = ExactDict({'sizes': [100,200,300], 'padStartA':[6,66], 'padEndA':[7,77]}, z)
    assert(e.sizes == [100,200,300])
    assert(e.zeroPadA == [[1,2, 6,7], [4,5, 66,77]])
    assert(e.stridesA == None)

def test_exact_override_pad_tbd_multi_pad_b():
    z={
        "TotalIndices" : 3,
        "ZeroPadA" : [],
        "ZeroPadB" : [[1,2, -1,-1], [4,5, -1, -1]],
    }

    e = ExactDict({'sizes': [100,200,300], 'padStartB':[6,66], 'padEndB':[7,77]}, z)
    assert(e.sizes == [100,200,300])
    assert(e.zeroPadB == [[1,2, 6,7], [4,5, 66,77]])
    assert(e.stridesA == None)
