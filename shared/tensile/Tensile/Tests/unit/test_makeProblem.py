################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest,logging
from Tensile.SolutionStructs import Convolution,ConvolutionConfig
log =logging.getLogger("testlog")

def test_spatial_in():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                     })

    log.debug(conv.printUsage(z))
    pcc = ConvolutionConfig(spatial=[14,15])
    p = conv.makeProblem(n=64, c=1024, k=256, pcc=pcc)
    assert(p[0] == [210, 256, 64, 1024])
    assert(p[1] == [1, 210, 215040])
    assert(p[2] == [-1, -1, 0])

def test_spatial_parm():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                     })

    log.debug(conv.printUsage(z))
    pcc = ConvolutionConfig()
    p = conv.makeProblem(n=64, c=1024, k=256, pcc=pcc)
    assert(p[0] == [182, 256, 64, 1024])
    assert(p[1] == [1, 182, 186368])
    assert(p[2] == [-1, -1, 0])


def test_stride():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                      'Stride' : '2x3',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(n=64, c=1024, k=256, pcc=conv.cc)
    assert(p[0] == [5, 7, 256, 64, 1024])
    assert(p[1] == [3, 28, 182, 186368])
    assert(p[2] == [-1, -1, 0])


def test_stride_filter():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                      'Stride' : '2x3',
                      'Filter' : '3x4',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(n=64, c=1024, k=256, pcc=conv.cc)
    assert(p[0] == [4, 6, 256, 64, 3, 4, 1024])
    assert(p[1] == [1, 14, 3, 28, 182, 186368])
    assert(p[2] == [-1, -1, -1, -1, 0])

def test_stride_filter_dilated():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '13x14',
                      'Stride' : '2x3',
                      'Filter' : '3x4',
                      'Dilation': '2x3',
                     })

    log.debug(conv.printUsage(z))
    p = conv.makeProblem(n=64, c=1024, k=256, pcc=conv.cc)
    assert(p[0] == [2, 5, 256, 64, 3, 4, 1024])
    assert(p[1] == [3, 28, 3, 28, 182, 186368])
    assert(p[2] == [-1, -1, -1, -1, 0])

def test_spatial_unspecified():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                     })

    log.debug(conv.printUsage(z))
    with pytest.raises(RuntimeError, match="ConvolutionConfig field 'spatial' == None"):
        conv.makeProblem(n=64, c=1024, k=256, pcc=conv.cc)

def test_mismatch():
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Spatial' : '34x99',
                     })

    log.debug(conv.printUsage(z))
    pcc = ConvolutionConfig(spatial=[14,15])
    with pytest.raises(RuntimeError, match="Mismatch between ConvolutionConfig value"):
        conv.makeProblem(n=64, c=1024, k=256, pcc=pcc)
