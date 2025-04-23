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
# -------------------------------------------------------------------------
# Copyright (c) Advanced Micro Devices. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Implements ONNX's backend API.
"""
import sys
if sys.version_info < (3, 0):
    sys.exit()

from onnx import ModelProto
from onnx.checker import check_model
from onnx.backend.base import Backend
import migraphx
from onnx_migraphx.backend_rep import MIGraphXBackendRep


def get_device():
    return ("CPU", "GPU")


class MIGraphXBackend(Backend):
    _device = "GPU"
    _input_names = []
    _prog_string = ""

    @classmethod
    def set_device(cls, device):
        cls._device = device
    """
    Implements
    `ONNX's backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
    with *ONNX Runtime*.
    The backend is mostly used when you need to switch between
    multiple runtimes with the same API.
    `Importing models from ONNX to Caffe2 <https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCaffe2Import.ipynb>`_
    shows how to use *caffe2* as a backend for a converted model.
    Note: This is not the official Python API.
    """  # noqa: E501

    @classmethod
    def get_program(cls):
        return cls._prog_string

    @classmethod
    def is_compatible(cls, model, device=None, **kwargs):
        """
        Return whether the model is compatible with the backend.

        :param model: unused
        :param device: None to use the default device or a string (ex: `'CPU'`)
        :return: boolean
        """
        device = cls._device
        return cls.supports_device(device)

    @classmethod
    def supports_device(cls, device):
        """
        Check whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return device in get_device()

    @classmethod
    def prepare(cls, model, device=None, **kwargs):
        """
        Load the model and creates a :class:`migraphx.program`
        ready to be used as a backend.

        :param model: ModelProto (returned by `onnx.load`),
            string for a filename or bytes for a serialized model
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see :class:`onnxruntime.SessionOptions`
        :return: :class:`migraphx.program`
        """
        if isinstance(model, MIGraphXBackendRep):
            return model
        elif isinstance(model, migraphx.program):
            return MIGraphXBackendRep(model, cls._input_names)
        elif isinstance(model, (str, bytes)):
            if device is not None and not cls.supports_device(device):
                raise RuntimeError(
                    "Incompatible device expected '{0}', got '{1}'".format(
                        device, get_device()))
            inf = migraphx.parse_onnx_buffer(model)
            cls._prog_string = str("\nPython =\n{}\nProgram =\n{}".format(
                inf.to_py(), inf))
            device = cls._device
            cls._input_names = inf.get_parameter_names()
            inf.compile(migraphx.get_target(device.lower()))
            cls._prog_string = cls._prog_string + str(
                "\nCompiled program =\n{}".format(inf))

            return cls.prepare(inf, device, **kwargs)
        else:
            # type: ModelProto
            check_model(model)
            bin = model.SerializeToString()
            return cls.prepare(bin, device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        """
        Compute the prediction.

        :param model: :class:`migraphx.program` returned
            by function *prepare*
        :param inputs: inputs
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see :class:`migraphx.program`
        :return: predictions
        """
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        '''
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        '''
        raise NotImplementedError(
            "It is much more efficient to run a whole model than every node independently."
        )


is_compatible = MIGraphXBackend.is_compatible
prepare = MIGraphXBackend.prepare
run = MIGraphXBackend.run_model
supports_device = MIGraphXBackend.supports_device
