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

import pytest
from Tensile.SolutionStructs import Convolution
from YamlBuilder.YamlBuilder import YamlBuilder
from YamlBuilder.YamlBuilder import defaultSizes, resnetSizes

@pytest.mark.parametrize(
        "problemSizes",
        [pytest.param((YamlBuilder.ProblemSizes,level), id="default-lvl=%d"%level) for level in [1,2,3,4]] +
        [resnetSizes]
        )
def test_2d_stride1(problemSizes):
    "Test number of problems generated"
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '1x1',
                      'Filter': '1x1',
                      })

    exacts = problemSizes[0](conv, z, problemSizes[1])
    if exacts:
        None


@pytest.mark.parametrize("problemSizes", [defaultSizes, resnetSizes])
@pytest.mark.parametrize("problemLevel", [1,2,3,4])
def test_2d_stride2(problemSizes, problemLevel):
    "Test number of problems generated"
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCHW',
                      'Stride': '2x2',
                      'Filter': '3x3',
                      })

    exacts = problemSizes[0](conv, z, problemLevel)
    if exacts:
        None


@pytest.mark.parametrize("problem_level", [1,2,3,4])
def test_3d(problem_level):
    "Test number of problems generated"
    z={} # problemType definition
    conv = Convolution(z, 'ConvolutionForward',
              config={'TensorAFormat': 'NCDHW',
                      'Stride': '2x2x2',
                      'Filter': '3x3x3',
                      })

    exacts = YamlBuilder.ProblemSizes(conv, z, problem_level)
    if exacts:
        None
