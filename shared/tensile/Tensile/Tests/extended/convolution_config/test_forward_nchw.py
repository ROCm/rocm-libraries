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

def test_nchw_defaults(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={})
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[0, 3, 2])
        assert(z['IndexAssignmentsB']==[3, 1, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.parametrize("problemSizes", [defaultSizes, resnetSizes, inceptionSizes])
def test_nchw_filter1x1(tensile_state, run_convolution_level, problemSizes):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Filter': '1x1',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==[0, 3, 2])
        assert(z['IndexAssignmentsB']==[3, 1, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution, problemSizes[0], problemSizes[1])

def test_nchw_packed_spatial0(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'PackedSpatialDims': 0
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
        assert(z['IndexAssignmentsB']==[4, 2, 3])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})

    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_tbd_strides(tensile_state, run_convolution_level):

    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': 'NxN',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
        assert(z['IndexAssignmentsB']==[4, 2, 3])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_const_strides(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '2x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
        assert(z['IndexAssignmentsB']==[4, 2, 3])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:2})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_const_use_initial_strides(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '2x3',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==[0, 1, 4, 3])
        assert(z['IndexAssignmentsB']==[4, 2, 3])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:3})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.parametrize("unrollOnChannel", [0, 1], ids=["unrollOnChannel0", "unrollOnChannel1"])
def test_nchw_filter2x2(tensile_state, run_convolution_level, unrollOnChannel):

    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'UnrollOnChannel': unrollOnChannel,
                      'Filter': '2x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==3)
        filterDims = [4, 3] if conv.unrollOnChannel else [5,4]
        cdim = 5 if conv.unrollOnChannel else 3
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1,2:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:2})

    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.parametrize("problemSizes", [defaultSizes, resnetSizes, inceptionSizes])
def test_nchw_filter2x1(tensile_state, run_convolution_level, problemSizes):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '2x1',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution, problemSizes[0], problemSizes[1])

@pytest.mark.parametrize("problemSizes", [defaultSizes, resnetSizes, inceptionSizes])
def test_nchw_filter7x1(tensile_state, run_convolution_level, problemSizes):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '7x1',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:7})
    run_convolution_level.func(conv, z, run_convolution_level.solution, problemSizes[0], problemSizes[1])

def test_nchw_filter2x1_dilation(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '2x1',
                      'Dilation': '1x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter1x2(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '1x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1,1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_filter1x2_dilation(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'TensorBFormat': 'KCYX',
                      'Filter': '1x2',
                      'Dilation': '1x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [3] if conv.unrollOnChannel else [4]
        cdim = 4 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:2,1:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,3:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_dilation2x2(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Dilation': '2x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        (cdim, filterDims) = (5,[4,3]) if conv.unrollOnChannel else (5,[4,3])
        assert(z['NumIndicesC']==3)
        assert(z['IndexAssignmentsA']==filterDims + [0, cdim, 2])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 1, 2])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:2,2:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,filterDims[0]:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:1, filterDims[1]:1})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_nchw_stride_filter(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': 'NxN',
                      'Filter': '2x2',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        filterDims = [5,4] if conv.unrollOnChannel else [6,5]
        cdim = 6 if conv.unrollOnChannel else 3
        assert(z['NumIndicesC']==4)
        assert(z['IndexAssignmentsA']==filterDims + [0, 1, cdim, 3])
        assert(z['IndexAssignmentsB']==filterDims + [cdim, 2, 3])
        assert(not z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {0:1})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,4:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {filterDims[0]:2, filterDims[1]:2})

    run_convolution_level.func(conv, z, run_convolution_level.solution)

def test_ncdhw_packed_strides3d_defaults(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'Stride': 'NxNxN',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==5)
        assert(z['IndexAssignmentsA']==[0, 1, 2, 5, 4])
        assert(z['IndexAssignmentsB']==[5, 3, 4])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)

@pytest.mark.skip(reason="out of registers in asm runs")
def test_ncdhw_packed_strides_filter3d(tensile_state, run_convolution_level):
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'TensorBFormat': 'KCZYX',
                      'TensorDFormat': 'NCDHW',
                      'Stride': 'NxNxN',
                      })
    log.debug(conv.printUsage(z))
    if not tensile_state.args["no_conv_assertions"]:
        assert(z['NumIndicesC']==5)
        assert(z['IndexAssignmentsA']==[0, 1, 2, 5, 4])
        assert(z['IndexAssignmentsB']==[5, 3, 4])
        assert(z['UseInitialStridesAB'])
        assert(conv.solutionParms["AssertStrideAEqual"] == {})
        assert(conv.solutionParms["AssertStrideBEqual"] == {0:1,2:0})
        assert(conv.solutionParms["AssertSizeEqual"] == {})
    run_convolution_level.func(conv, z, run_convolution_level.solution)
