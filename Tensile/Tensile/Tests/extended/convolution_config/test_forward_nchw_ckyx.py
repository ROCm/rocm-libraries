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
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import defaultSizes, resnetSizes, inceptionSizes

log =logging.getLogger("testlog")

@pytest.mark.parametrize("problemSizes", [defaultSizes, resnetSizes, inceptionSizes])
def test_ckyx_1x1(tensile_state, run_convolution_level,problemSizes):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'CKYX',
                      'Filter': '1x1',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[0, 3, 2])
        assert(z['IndexAssignmentsB']==[1, 3, 2])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})

    run_convolution_level.func(conv, z, run_convolution_level.solution, problemSizes[0], problemSizes[1])

def test_ckyx_1x1_nopack(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'CKYX',
                      'PackedSpatialDims': 0,
                      'Filter': '1x1',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
        assert(z['IndexAssignmentsB']==[2, 4, 3])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})

    run_convolution_level.func(conv, z, run_convolution_level.solution)


def test_ckyx_2x2(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'CKYX',
                      'Filter': '2x3',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [4,3] if conv.unrollOnChannel else [5,4]
        cdim = 5 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [1, cdim, 2])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1,2:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:3, filterDims[1]:2})

    run_convolution_level.func(conv, z, run_convolution_level.solution)
