#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
"""
Implements ONNX's backend API.
"""
import sys
if sys.version_info < (3, 0):
    sys.exit()

import migraphx
from onnx.backend.base import BackendRep
import numpy as np
from typing import Any, Tuple


class MIGraphXBackendRep(BackendRep):
    """
    Computes the prediction for a pipeline converted into
    an :class:`onnxruntime.InferenceSession` node.
    """
    def __init__(self, prog, input_names):
        """
        :param session: :class:`migraphx.program`
        """
        self._program = prog
        self._input_names = input_names

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        """
        Computes the prediction.
        See :meth:`migraphx.program.run`.
        """

        if isinstance(inputs, list):
            inps = {}
            for i, name in enumerate(self._input_names):
                inps[name] = migraphx.argument(inputs[i])
            mgx_outputs = self._program.run(inps)
            outs = []
            for out in mgx_outputs:
                outs.append(np.array(out))
            return outs
        else:
            inp = self._program.get_parameter_shapes().keys()
            if len(inp) != 1:
                raise RuntimeError("Model expect {0} inputs".format(len(inp)))
            inps = {inp[0]: migraphx.argument(inputs)}
            mgx_outputs = self._program.run(inps)
            outs = []
            for out in mgx_outputs:
                outs.append(np.array(out))
            return self._program.run(inps)
