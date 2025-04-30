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

import logging
from Tensile.SolutionStructs import Convolution
log =logging.getLogger("testlog")

def test_nhwc_defaults(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'TensorBFormat': 'KYXC',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[3, 1, 2])
        assert(z['IndexAssignmentsB']==[3, 0, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nhwc_filter3x2(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'TensorBFormat': 'KYXC',
                      'Filter': '3x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [4,3] if conv.unrollOnChannel else [5,4]
        cdim = 5 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[cdim] + filterDims + [1, 2])
        assert(z['IndexAssignmentsB']==[cdim] + filterDims + [0, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:3})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nhwc_filter2x2_dilation(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'TensorBFormat': 'KYXC',
                      'Filter':   '2x2',
                      'Dilation': '2x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [4,3] if conv.unrollOnChannel else [5,4]
        cdim = 5 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[cdim] + filterDims + [1, 2])
        assert(z['IndexAssignmentsB']==[cdim] + filterDims + [0, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nhwc_filter2x2_stride(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'TensorBFormat': 'KYXC',
                      'Filter': '2x2',
                      'Stride': '2x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [4,3] if conv.unrollOnChannel else [5,4]
        cdim = 5 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[cdim] + filterDims + [1, 2])
        assert(z['IndexAssignmentsB']==[cdim] + filterDims + [0, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nhwc_filter3x3_pad(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NHWC',
                      'TensorBFormat': 'KYXC',
                      'Filter':   '3x3',
                      'PadStart': '1x1',
                      'PadEnd':   '1x1',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [4,3] if conv.unrollOnChannel else [5,4]
        cdim = 5 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[cdim] + filterDims + [1, 2])
        assert(z['IndexAssignmentsB']==[cdim] + filterDims + [0, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)
