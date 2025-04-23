#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# This script generates onnx files for MIGraphX onnx operator tests.
# To generate an individual onnx file, you can use the following
# command: python3 -c "import gen_onnx; gen_onnx.{test_name}_test()"
import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
from onnx.numpy_helper import from_array


def onnx_test(external_data=False):
    def create_onnx_test(op_test):
        def run_test():
            op_info = op_test()
            if len(op_info) > 3:
                graph_def = helper.make_graph(op_info[0],
                                              op_test.__name__,
                                              op_info[1],
                                              op_info[2],
                                              initializer=op_info[3])
            else:
                graph_def = helper.make_graph(op_info[0], op_test.__name__,
                                              op_info[1], op_info[2])
            model_def = helper.make_model(graph_def,
                                          producer_name=op_test.__name__)
            onnx.save_model(model_def,
                            '{}.onnx'.format(op_test.__name__),
                            save_as_external_data=external_data,
                            location='{}.weight'.format(op_test.__name__),
                            size_threshold=0,
                            convert_attribute=True)

        return run_test

    return create_onnx_test


@onnx_test()
def acos_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Acos',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def acosh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Acosh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def add_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Add',
                                 inputs=['0', '1'],
                                 broadcast=1,
                                 axis=1,
                                 outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test()
def add_fp16_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [1])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [1])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [1])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    return (
        [node],
        [x, y],
        [z],
        # '0' -> 1.5, '1' -> 2.5
        [
            onnx.helper.make_tensor('0', TensorProto.FLOAT16, [1], [15872]),
            onnx.helper.make_tensor('1', TensorProto.FLOAT16, [1], [16640])
        ])


@onnx_test()
def add_fp8_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT8E4M3FNUZ, [1])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT8E4M3FNUZ, [1])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT8E4M3FNUZ, [1])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test()
def add_bf16_test():
    x = helper.make_tensor_value_info('0', TensorProto.BFLOAT16, [1])
    y = helper.make_tensor_value_info('1', TensorProto.BFLOAT16, [1])
    z = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [1])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test()
def add_scalar_test():
    x = helper.make_tensor_value_info('0', TensorProto.UINT8, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.UINT8, [])
    z = helper.make_tensor_value_info('2', TensorProto.UINT8, [2, 3, 4, 5])

    node = onnx.helper.make_node('Add', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test()
def argmax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 6])

    node = onnx.helper.make_node('ArgMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=2,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def argmax_select_last_index_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 6])

    node = onnx.helper.make_node('ArgMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=2,
                                 keepdims=0,
                                 select_last_index=1)

    return ([node], [x], [y])


@onnx_test()
def argmax_dyn_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 4, 6])

    node = onnx.helper.make_node('ArgMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=2,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def argmin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5])

    node = onnx.helper.make_node('ArgMin',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=3,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def argmin_select_last_index_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5])

    node = onnx.helper.make_node('ArgMin',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=3,
                                 keepdims=0,
                                 select_last_index=1)

    return ([node], [x], [y])


@onnx_test()
def asin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Asin',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def asinh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Asinh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def atan_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Atan',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def atanh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Atanh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def averagepool_1d_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 5])
    out = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['0'],
                                 outputs=['1'],
                                 kernel_shape=[3])

    return ([node], [x], [out])


@onnx_test()
def averagepool_dilate_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 2])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2],
                                 strides=[1],
                                 pads=[1, 1],
                                 dilations=[3])

    return ([node], [x], [y])


@onnx_test()
def averagepool_3d_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 5, 5, 5])
    out = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                        [1, 3, 3, 3, 3])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['0'],
                                 outputs=['1'],
                                 kernel_shape=[3, 3, 3])

    return ([node], [x], [out])


@onnx_test()
def averagepool_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [None, 3, 5, 5, 5])
    out = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                        [None, 3, 3, 3, 3])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['0'],
                                 outputs=['1'],
                                 kernel_shape=[3, 3, 3],
                                 strides=[2, 2, 2],
                                 pads=[1, 1, 1, 1, 1, 1])
    return ([node], [x], [out])


@onnx_test()
def averagepool_dyn_autopad_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [None, 3, 5, 5, 5])
    out = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                        [None, 3, 3, 3, 3])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['0'],
                                 outputs=['1'],
                                 kernel_shape=[3, 3, 3],
                                 strides=[2, 2, 2],
                                 auto_pad='SAME_UPPER')
    return ([node], [x], [out])


@onnx_test()
def averagepool_dyn_asym_padding_error_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 1, 3, 3])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2, 2],
                                 strides=[2, 2],
                                 pads=[0, 0, 1, 1])

    return ([node], [x], [y])


@onnx_test()
def averagepool_dyn_cip_error_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 1, 1, 1])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2, 2],
                                 count_include_pad=1)

    return ([node], [x], [y])


@onnx_test()
def averagepool_notset_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 1, 1])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[6, 6],
                                 strides=[2, 2],
                                 pads=[0, 0, 1, 1],
                                 auto_pad='NOTSET')

    return ([node], [x], [y])


@onnx_test()
def averagepool_nt_cip_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 1, 1])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[6, 6],
                                 strides=[2, 2],
                                 pads=[0, 0, 1, 1],
                                 auto_pad='NOTSET',
                                 count_include_pad=1)

    return ([node], [x], [y])


@onnx_test()
def averagepool_same_lower_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2, 2],
                                 auto_pad='SAME_LOWER')

    return ([node], [x], [y])


@onnx_test()
def averagepool_sl_cip_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2, 2],
                                 auto_pad='SAME_LOWER',
                                 count_include_pad=1)

    return ([node], [x], [y])


@onnx_test()
def averagepool_same_upper_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('AveragePool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2, 2],
                                 auto_pad='SAME_UPPER')

    return ([node], [x], [y])


@onnx_test()
def batch_norm_flat_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [1])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [1])
    var = helper.make_tensor_value_info('variance', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'variance'],
        outputs=['y'],
        epsilon=1e-6)

    return ([node], [x, scale, bias, mean, var], [out])


@onnx_test()
def batch_norm_rank_2_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 5])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [5])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [5])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [5])
    var = helper.make_tensor_value_info('variance', TensorProto.FLOAT, [5])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 5])

    node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'variance'],
        outputs=['y'],
        epsilon=1e-6)

    return ([node], [x, scale, bias, mean, var], [out])


@onnx_test()
def batch_norm_1d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 3, 4])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [3])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [3])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [3])
    var = helper.make_tensor_value_info('variance', TensorProto.FLOAT, [3])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 3, 4])

    node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'variance'],
        outputs=['y'])

    return ([node], [x, scale, bias, mean, var], [out])


@onnx_test()
def batch_norm_2d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4, 4])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [3])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [3])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [3])
    var = helper.make_tensor_value_info('variance', TensorProto.FLOAT, [3])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 4])

    node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'variance'],
        outputs=['y'])

    return ([node], [x, scale, bias, mean, var], [out])


@onnx_test()
def batch_norm_3d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16,
                                      [2, 2, 2, 2, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT16, [2])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT16, [2])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT16, [2])
    var = helper.make_tensor_value_info('variance', TensorProto.FLOAT16, [2])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT16,
                                        [2, 2, 2, 2, 2])

    node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'variance'],
        outputs=['y'],
        epsilon=1e-6)

    return ([node], [x, scale, bias, mean, var], [out])


@onnx_test()
def batch_norm_invalid_bias_rank_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4, 4])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [3])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [3, 1])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [3])
    var = helper.make_tensor_value_info('variance', TensorProto.FLOAT, [3])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 4])

    node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'variance'],
        outputs=['y'])

    return ([node], [x, scale, bias, mean, var], [out])


@onnx_test()
def binary_dyn_brcst_prelu_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                         [None, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [None, 3, 4, 5])

    node = onnx.helper.make_node(
        'PRelu',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def binary_dyn_brcst_add_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                         [None, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [None, 3, 4, 5])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def binary_dyn_brcst_attr_error_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                         [None, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [None, 3, 4, 5])

    node = onnx.helper.make_node('Add',
                                 inputs=['0', '1'],
                                 outputs=['out'],
                                 broadcast=1,
                                 axis=1)

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def binary_dyn_brcst_mul_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                         [None, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [None, 3, 4, 5])

    node = onnx.helper.make_node(
        'Mul',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def binary_dyn_brcst_mul_fp8_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT8E4M3FNUZ,
                                         [None, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT8E4M3FNUZ,
                                         [4, 1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT8E4M3FNUZ,
                                            [None, 3, 4, 5])

    node = onnx.helper.make_node(
        'Mul',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def bitwise_and_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT32, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 5])
    z = helper.make_tensor_value_info('2', TensorProto.INT32, [2, 3, 4, 5])

    node = onnx.helper.make_node('BitwiseAnd',
                                 inputs=['0', '1'],
                                 outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test()
def div_fp8_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT8E4M3FNUZ,
                                         [2, 3])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT8E4M3FNUZ,
                                         [2, 3])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT8E4M3FNUZ,
                                            [2, 3])

    node = onnx.helper.make_node(
        'Div',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def cast_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node('Cast', inputs=['x'], outputs=['y'], to=1)

    return ([node], [x], [y])


@onnx_test()
def castlike_test():
    input = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [10])
    target_type = helper.make_tensor_value_info('1', TensorProto.FLOAT, [10])
    output = helper.make_tensor_value_info('out', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node('CastLike',
                                 inputs=['0', '1'],
                                 outputs=['out'])

    return ([node], [input, target_type], [output])


@onnx_test()
def castlike_error_test():
    input = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [10])
    output = helper.make_tensor_value_info('out', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node('CastLike', inputs=['0'], outputs=['out'])

    return ([node], [input], [output])


@onnx_test()
def ceil_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Ceil',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def celu_alpha_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Celu',
                                 inputs=['x'],
                                 outputs=['y'],
                                 alpha=0.8)

    return ([node], [x], [y])


@onnx_test()
def celu_default_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('Celu', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def celu_verify_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('Celu',
                                 inputs=['x'],
                                 outputs=['y'],
                                 alpha=0.5)

    return ([node], [x], [y])


@onnx_test()
def celu_wrong_type_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 3])

    node = onnx.helper.make_node('Celu', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def celu_zero_alpha_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('Celu',
                                 inputs=['x'],
                                 outputs=['y'],
                                 alpha=0.0)

    return ([node], [x], [y])


@onnx_test()
def clip_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Clip',
                                 inputs=['0'],
                                 outputs=['1'],
                                 max=6.0,
                                 min=0.0)

    return ([node], [x], [y])


@onnx_test()
def clip_test_op11():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    min_val = helper.make_tensor('min', TensorProto.FLOAT, [], [0.0])
    max_val = helper.make_tensor('max', TensorProto.FLOAT, [], [6.0])

    node = onnx.helper.make_node('Clip',
                                 inputs=['0', 'min', 'max'],
                                 outputs=['1'])

    return ([node], [x], [y], [min_val, max_val])


@onnx_test()
def clip_test_op11_max_only():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    max_val = helper.make_tensor('max', TensorProto.FLOAT, [], [0.0])

    node = onnx.helper.make_node('Clip',
                                 inputs=['0', '', 'max'],
                                 outputs=['1'])

    return ([node], [x], [y], [max_val])


@onnx_test()
def clip_test_op11_min_only():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    min_val = helper.make_tensor('min', TensorProto.FLOAT, [], [0.0])

    node = onnx.helper.make_node('Clip', inputs=['0', 'min'], outputs=['1'])

    return ([node], [x], [y], [min_val])


@onnx_test()
def clip_test_op11_no_args():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Clip', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def clip_test_op11_no_args1():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Clip', inputs=['0', '', ''], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def clip_test_args_type_mismatch():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 3])

    min_val = helper.make_tensor('min', TensorProto.FLOAT, [1, 3],
                                 [1.5, 2.5, 3.5])
    max_val = helper.make_tensor('max', TensorProto.INT64, [3, 1], [2, 3, 4])

    node = onnx.helper.make_node('Clip',
                                 inputs=['0', 'min', 'max'],
                                 outputs=['1'])

    return ([node], [x], [y], [min_val, max_val])


@onnx_test()
def clip_dyn_min_max_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None])

    min_val = helper.make_tensor('min', TensorProto.FLOAT, [], [0.0])
    max_val = helper.make_tensor('max', TensorProto.FLOAT, [], [6.0])

    node = onnx.helper.make_node('Clip',
                                 inputs=['0', 'min', 'max'],
                                 outputs=['1'])

    return ([node], [x], [y], [min_val, max_val])


@onnx_test()
def clip_dyn_min_only_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None])

    min_val = helper.make_tensor('min', TensorProto.FLOAT, [], [0.0])

    node = onnx.helper.make_node('Clip', inputs=['0', 'min'], outputs=['1'])

    return ([node], [x], [y], [min_val])


@onnx_test()
def concat_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 4, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7, 4, 3])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [9, 4, 3])

    node = onnx.helper.make_node(
        'Concat',
        inputs=['0', '1'],
        axis=0,
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test()
def concat_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, None, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, None, 3])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [None, None, 3])

    node = onnx.helper.make_node(
        'Concat',
        inputs=['0', '1'],
        axis=0,
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test()
def constant_test():
    x = np.array([0, 1, 2])
    y = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['0'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        ),
    )

    return ([node], [], [y])


@onnx_test()
def constant_value_float_test():

    node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=[],
                                 value_float=[1.0])

    return ([node], [], [])


@onnx_test()
def constant_value_floats_test():

    node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=[],
                                 value_floats=[1.0, 2.0, 3.0])

    return ([node], [], [])


@onnx_test()
def constant_value_int_test():

    node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=[],
                                 value_int=[1])

    return ([node], [], [])


@onnx_test()
def constant_value_ints_test():

    node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=[],
                                 value_ints=[1, 2, 3])

    return ([node], [], [])


@onnx_test()
def constant_no_attributes_test():

    node = onnx.helper.make_node('Constant', inputs=[], outputs=[])

    return ([node], [], [])


@onnx_test()
def constant_multiple_attributes_test():
    x = np.array([0, 1, 2])

    node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=[],
                                 value_floats=[1.0, 2.0],
                                 value_ints=[1, 2],
                                 value=onnx.helper.make_tensor(
                                     name='const_tensor',
                                     data_type=TensorProto.FLOAT,
                                     dims=x.shape,
                                     vals=x.flatten().astype(float)))

    return ([node], [], [])


@onnx_test()
def constant_fill_test():
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'ConstantFill',
        inputs=[],
        outputs=['value'],
        dtype=1,
        value=1.0,
        shape=[2, 3],
        input_as_shape=0,
    )

    return ([node], [], [value])


@onnx_test()
def constant_fill_input_as_shape_test():
    np_shape = np.array([2, 3])
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [2, 3])

    ts_shape = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=np_shape.shape,
                                  vals=np_shape.flatten().astype(int))

    const_shape_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=ts_shape,
    )

    node = onnx.helper.make_node(
        'ConstantFill',
        inputs=['shape'],
        outputs=['value'],
        dtype=1,
        value=1.0,
        input_as_shape=1,
    )

    return ([const_shape_node, node], [], [value])


@onnx_test()
def constant_scalar_test():
    x = np.array([1])
    y = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['0'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=TensorProto.INT32,
            dims=x.shape,
            vals=x.flatten().astype(int),
        ),
    )

    return ([node], [], [y])


@onnx_test()
def constant_empty_scalar_int64_test():
    x = np.array([]).astype(np.int64)
    y = helper.make_tensor_value_info('0', TensorProto.INT64, [0])

    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['0'],
        value=onnx.helper.make_tensor(
            name='one_element_tensor',
            data_type=TensorProto.INT64,
            dims=x.shape,
            vals=x.flatten().astype(np.int64),
        ),
    )

    return ([node], [], [y])


@onnx_test()
def constant_one_val_int64_test():
    x = np.array([1]).astype(np.int64)
    y = helper.make_tensor_value_info('0', TensorProto.INT64, [0])

    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['0'],
        value=onnx.helper.make_tensor(
            name='empty_tensor',
            data_type=TensorProto.INT64,
            dims=x.shape,
            vals=x.flatten().astype(np.int64),
        ),
    )

    return ([node], [], [y])


@onnx_test()
def const_of_shape_empty_input_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.INT64, [1],
                                         [10])
    empty_val = np.array([]).astype(np.int64)
    empty_ts = helper.make_tensor(name='empty_tensor',
                                  data_type=TensorProto.INT64,
                                  dims=empty_val.shape,
                                  vals=empty_val.flatten().astype(np.int64))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=empty_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['shape'],
        outputs=['y'],
        value=tensor_val,
    )

    return ([shape_const, node], [], [y])


@onnx_test()
def const_of_shape_float_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [1],
                                         [10])

    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT64,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(np.int64))

    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node('ConstantOfShape',
                                 inputs=['shape'],
                                 outputs=['y'],
                                 value=tensor_val)

    return ([shape_const, node], [], [y])


@onnx_test()
def const_of_shape_default_test():
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT64,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(np.int64))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2, 3, 4])

    node = onnx.helper.make_node('ConstantOfShape',
                                 inputs=['shape'],
                                 outputs=['y'])

    return ([shape_const, node], [], [y])


@onnx_test()
def const_of_shape_int64_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.INT64, [1],
                                         [10])
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT64,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(np.int64))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2, 3, 4])

    node = onnx.helper.make_node('ConstantOfShape',
                                 inputs=['shape'],
                                 outputs=['y'],
                                 value=tensor_val)

    return ([shape_const, node], [], [y])


@onnx_test()
def const_of_shape_no_value_attr_test():
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT64,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(np.int64))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['shape'],
        outputs=['y'],
    )

    return ([shape_const, node], [], [y])


@onnx_test()
def const_of_shape_dyn_float_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [1],
                                         [10])

    output_dims = helper.make_tensor_value_info('output_dims',
                                                TensorProto.INT64, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node('ConstantOfShape',
                                 inputs=['output_dims'],
                                 outputs=['y'],
                                 value=tensor_val)

    return ([node], [output_dims], [y])


@onnx_test()
def const_of_shape_dyn_int64_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.INT64, [1],
                                         [10])

    output_dims = helper.make_tensor_value_info('output_dims',
                                                TensorProto.INT64, [3])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2, 3, 4])

    node = onnx.helper.make_node('ConstantOfShape',
                                 inputs=['output_dims'],
                                 outputs=['y'],
                                 value=tensor_val)

    return ([node], [output_dims], [y])


@onnx_test()
def conv_1d_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 3])

    node = onnx.helper.make_node('Conv', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [out])


@onnx_test()
def conv_3d_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 5, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                        [1, 1, 3, 3, 3])

    node = onnx.helper.make_node('Conv', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [out])


@onnx_test()
def conv_1d_fp8_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT8E4M3FNUZ,
                                      [1, 3, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT8E4M3FNUZ,
                                      [1, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT8E4M3FNUZ,
                                        [1, 1, 3])

    node = onnx.helper.make_node('Conv', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [out])


@onnx_test()
def conv_attr_fail_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 3])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 strides=[1, 1],
                                 outputs=['2'])

    return ([node], [x, y], [out])


@onnx_test()
def conv_autopad_fail_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 34, 34])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 dilations=[1, 1],
                                 strides=[1, 1],
                                 auto_pad='SAME',
                                 pads=[0, 0, 1, 1, 0, 0, 1, 1])

    return ([node], [x, y], [out])


@onnx_test()
def conv_autopad_same_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 32, 32])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 dilations=[1, 1],
                                 strides=[1, 1],
                                 auto_pad='SAME')

    return ([node], [x, y], [out])


@onnx_test()
def conv_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 2, 28, 28])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z], [out])


@onnx_test()
def conv_bad_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.INT32, [1])
    out = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 2, 28, 28])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z], [out])


@onnx_test()
def conv_bn_relu_maxpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1])
    m = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1])
    n = helper.make_tensor_value_info('4', TensorProto.FLOAT, [1])
    k = helper.make_tensor_value_info('5', TensorProto.FLOAT, [1])
    l = helper.make_tensor_value_info('6', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('10', TensorProto.FLOAT,
                                        [1, 1, 14, 14])

    node0 = onnx.helper.make_node('Conv',
                                  inputs=['0', '1', '2'],
                                  outputs=['7'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node1 = onnx.helper.make_node('BatchNormalization',
                                  inputs=['7', '3', '4', '5', '6'],
                                  outputs=['8'],
                                  epsilon=9.99999974737875e-06,
                                  momentum=0.899999976158142)

    node2 = onnx.helper.make_node('Relu', inputs=['8'], outputs=['9'])
    node3 = onnx.helper.make_node('MaxPool',
                                  inputs=['9'],
                                  outputs=['10'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    return ([node0, node1, node2, node3], [x, y, z, m, n, k, l], [out])


@onnx_test()
def conv_dynamic_batch_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 3, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                        [None, 1, 3, 3])

    node = onnx.helper.make_node('Conv', inputs=['0', '1'], outputs=['2'])
    return ([node], [x, y], [out])


@onnx_test()
def conv_dynamic_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [None, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('3', TensorProto.FLOAT,
                                        [None, 2, 28, 28])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z], [out])


@onnx_test()
def conv_dynamic_img_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [1, 3, None, None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                        [1, 1, None, None])

    node = onnx.helper.make_node('Conv', inputs=['0', '1'], outputs=['2'])
    return ([node], [x, y], [out])


@onnx_test()
def conv_dynamic_weights_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                      [1, 3, None, None])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                        [1, 1, None, None])

    node = onnx.helper.make_node('Conv', inputs=['0', '1'], outputs=['2'])
    return ([node], [x, y], [out])


@onnx_test()
def conv_dynamic_img_and_weights_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [1, 3, None, None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                      [1, 3, None, None])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                        [1, 1, None, None])

    node = onnx.helper.make_node('Conv', inputs=['0', '1'], outputs=['2'])
    return ([node], [x, y], [out])


@onnx_test()
def conv_dynamic_batch_same_upper_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 3, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 auto_pad='SAME_UPPER')
    return ([node], [x, y], [out])


@onnx_test()
def conv_dynamic_img_same_upper_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [1, 3, None, None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 3, 3])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                        [1, 1, None, None])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 auto_pad='SAME_UPPER')
    return ([node], [x, y], [out])


@onnx_test()
def conv_dynamic_kernel_same_lower_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                      [1, 3, None, None])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 auto_pad='SAME_LOWER')
    return ([node], [x, y], [out])


@onnx_test()
def conv_relu_maxpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('5', TensorProto.FLOAT, [1, 1, 14, 14])

    node1 = onnx.helper.make_node('Conv',
                                  inputs=['0', '1', '2'],
                                  outputs=['3'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node2 = onnx.helper.make_node('Relu', inputs=['3'], outputs=['4'])

    node3 = onnx.helper.make_node('MaxPool',
                                  inputs=['4'],
                                  outputs=['5'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    return ([node1, node2, node3], [x, y, z], [out])


@onnx_test()
def conv_relu_maxpool_x2_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [5, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5])
    m = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 5, 5, 5])
    n = helper.make_tensor_value_info('4', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('10', TensorProto.FLOAT, [1, 1, 5, 5])

    node1 = onnx.helper.make_node('Conv',
                                  inputs=['0', '1', '2'],
                                  outputs=['5'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node2 = onnx.helper.make_node('Relu', inputs=['5'], outputs=['6'])

    node3 = onnx.helper.make_node('MaxPool',
                                  inputs=['6'],
                                  outputs=['7'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    node4 = onnx.helper.make_node('Conv',
                                  inputs=['7', '3', '4'],
                                  outputs=['8'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node5 = onnx.helper.make_node('Relu', inputs=['8'], outputs=['9'])

    node6 = onnx.helper.make_node('MaxPool',
                                  inputs=['9'],
                                  outputs=['10'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    return ([node1, node2, node3, node4, node5, node6], [x, y, z, m, n], [out])


@onnx_test()
def convinteger_no_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT8, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.INT8, [1, 3, 2, 2])
    out = helper.make_tensor_value_info('3', TensorProto.INT32, [1, 1, 4, 4])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1'],
                                 outputs=['3'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y], [out])


@onnx_test()
def convinteger_no_bias_uint8_test():
    x = helper.make_tensor_value_info('0', TensorProto.UINT8, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.UINT8, [1, 3, 5, 5])
    out = helper.make_tensor_value_info('3', TensorProto.INT32, [1, 2, 28, 28])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1'],
                                 outputs=['3'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y], [out])


@onnx_test()
def convinteger_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT8, [2, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.INT8, [1])
    out = helper.make_tensor_value_info('3', TensorProto.INT32, [2, 4, 28, 28])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z], [out])


@onnx_test()
def convinteger_dual_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT8, [2, 3, 10, 10])
    y = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3, 3, 3])
    z = helper.make_tensor_value_info('2', TensorProto.INT8, [1])
    w = helper.make_tensor_value_info('3', TensorProto.INT8, [1])
    out = helper.make_tensor_value_info('4', TensorProto.INT32, [2, 4, 8, 8])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1', '2', '3'],
                                 outputs=['4'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z, w], [out])


@onnx_test()
def convinteger_dual_bias_simple_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT8, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.INT8, [1, 3, 2, 2])
    z = helper.make_tensor_value_info('2', TensorProto.INT8, [1])
    w = helper.make_tensor_value_info('3', TensorProto.INT8, [1])
    out = helper.make_tensor_value_info('4', TensorProto.INT32, [1, 1, 4, 4])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1', '2', '3'],
                                 outputs=['4'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z, w], [out])


@onnx_test()
def convinteger_mismatched_input_types_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT8, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.UINT8, [1, 3, 5, 5])
    out = helper.make_tensor_value_info('4', TensorProto.INT32, [1, 2, 28, 28])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1'],
                                 outputs=['4'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y], [out])


@onnx_test()
def convinteger_mismatched_inputs_dual_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.UINT8, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.INT8, [1, 3, 2, 2])
    z = helper.make_tensor_value_info('2', TensorProto.UINT8, [1])
    w = helper.make_tensor_value_info('3', TensorProto.INT8, [1])
    out = helper.make_tensor_value_info('4', TensorProto.INT32, [1, 1, 4, 4])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1', '2', '3'],
                                 outputs=['4'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z, w], [out])


@onnx_test()
def convinteger_mismatched_data_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT8, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.INT8, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.UINT8, [1])
    w = helper.make_tensor_value_info('3', TensorProto.INT8, [1])
    out = helper.make_tensor_value_info('4', TensorProto.INT32, [1, 2, 28, 28])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1', '2', '3'],
                                 outputs=['4'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z, w], [out])


@onnx_test()
def convinteger_mismatched_weight_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT8, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.INT8, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.INT8, [1])
    w = helper.make_tensor_value_info('3', TensorProto.UINT8, [1])
    out = helper.make_tensor_value_info('4', TensorProto.INT32, [1, 2, 28, 28])

    node = onnx.helper.make_node('ConvInteger',
                                 inputs=['0', '1', '2', '3'],
                                 outputs=['4'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z, w], [out])


@onnx_test()
def cos_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Cos',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def cos_fp8_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT8E4M3FNUZ, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT8E4M3FNUZ, [10])

    node = onnx.helper.make_node(
        'Cos',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def cosh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'Cosh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def conv_transpose_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('ConvTranspose',
                                 name='conv1',
                                 inputs=['x', 'w'],
                                 outputs=['y'])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 1, 3, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('ConvTranspose',
                                 name='conv1',
                                 inputs=['x', 'w', 'b'],
                                 outputs=['y'])

    return ([node], [x, w, b], [y])


@onnx_test()
def conv_transpose_input_pads_strides_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 7, 5])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2],
                                 pads=[1, 1, 1, 1])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_input_pads_asymm_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 8, 6])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2],
                                 pads=[0, 0, 1, 1])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_input_pads_asymm_1d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 6])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[2],
                                 pads=[0, 1],
                                 dilations=[1])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_output_padding_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 10, 8])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2],
                                 output_padding=[1, 1])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_output_padding_3d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 10, 8, 8])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2, 2],
                                 output_padding=[1, 1, 1])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_output_shape_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 10, 8])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2],
                                 output_shape=[10, 8])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_output_shape_3d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 10, 8, 8])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2, 2],
                                 output_shape=[10, 8, 8])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_stride_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 7, 3])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_auto_pad_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 3, 3])

    node = onnx.helper.make_node('ConvTranspose',
                                 name='conv1',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 auto_pad='SAME_UPPER')

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_dyn_asym_padding_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 8, 6])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2],
                                 pads=[0, 0, 1, 1])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_dyn_output_shape_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 2, 10, 8])

    node = onnx.helper.make_node('ConvTranspose',
                                 inputs=['x', 'w'],
                                 outputs=['y'],
                                 strides=[3, 2],
                                 output_shape=[10, 8])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_dyn_batch_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 1, 5, 5])

    node = onnx.helper.make_node('ConvTranspose',
                                 name='conv1',
                                 inputs=['x', 'w'],
                                 outputs=['y'])

    return ([node], [x, w], [y])


@onnx_test()
def conv_transpose_dyn_img_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [1, 1, None, None])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT,
                                      [1, 1, None, None])

    node = onnx.helper.make_node('ConvTranspose',
                                 name='conv1',
                                 inputs=['x', 'w'],
                                 outputs=['y'])

    return ([node], [x, w], [y])


@onnx_test()
def depthtospace_test():

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 8, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 10, 10])

    node = onnx.helper.make_node('DepthToSpace',
                                 inputs=['x'],
                                 outputs=['y'],
                                 blocksize=2,
                                 mode='DCR')

    return ([node], [x], [y])


@onnx_test()
def depthtospace_simple_test():

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 8, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 4, 6])

    node = onnx.helper.make_node('DepthToSpace',
                                 inputs=['x'],
                                 outputs=['y'],
                                 blocksize=2,
                                 mode='DCR')

    return ([node], [x], [y])


@onnx_test()
def depthtospace_crd_test():

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 8, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 10, 10])

    node = onnx.helper.make_node('DepthToSpace',
                                 inputs=['x'],
                                 outputs=['y'],
                                 blocksize=2,
                                 mode='CRD')

    return ([node], [x], [y])


@onnx_test()
def spacetodepth_test():

    x = helper.make_tensor_value_info('x', TensorProto.float, [2, 2, 10, 10])
    y = helper.make_tensor_value_info('y', TensorProto.float, [2, 8, 5, 5])

    node = onnx.helper.make_node('spacetodepth',
                                 inputs=['x'],
                                 outputs=['y'],
                                 blocksize=2)

    return ([node], [x], [y])


@onnx_test()
def spacetodepth_simple_test():

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 4, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 8, 2, 3])

    node = onnx.helper.make_node('SpaceToDepth',
                                 inputs=['x'],
                                 outputs=['y'],
                                 blocksize=2)

    return ([node], [x], [y])


@onnx_test()
def spacetodepth_invalid_blocksize_test():

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 4, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 8, 2, 3])

    node = onnx.helper.make_node('SpaceToDepth',
                                 inputs=['x'],
                                 outputs=['y'],
                                 blocksize=0.3)

    return ([node], [x], [y])


@onnx_test()
def spacetodepth_nondivisibility_test():

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 8, 2, 2])

    node = onnx.helper.make_node('SpaceToDepth',
                                 inputs=['x'],
                                 outputs=['y'],
                                 blocksize=2)

    return ([node], [x], [y])


@onnx_test()
def dequantizelinear_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.INT8, [5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [5])

    node = onnx.helper.make_node(
        'DequantizeLinear',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def dequantizelinear_zero_point_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.INT8, [5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1])
    arg2 = helper.make_tensor_value_info('2', TensorProto.INT8, [1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [5])

    node = onnx.helper.make_node(
        'DequantizeLinear',
        inputs=['0', '1', '2'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1, arg2], [arg_out])


def make_dequantizelinear_axis_graph(axis):
    arg0 = helper.make_tensor_value_info('0', TensorProto.INT8, [1, 1, 5, 1])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [5])
    arg2 = helper.make_tensor_value_info('2', TensorProto.INT8, [5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [1, 1, 5, 1])

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['0', '1', '2'],
                                 outputs=['out'],
                                 axis=axis)

    return ([node], [arg0, arg1, arg2], [arg_out])


@onnx_test()
def dequantizelinear_axis_test():
    return make_dequantizelinear_axis_graph(2)


@onnx_test()
def dequantizelinear_neg_axis_test():
    return make_dequantizelinear_axis_graph(-2)


@onnx_test()
def dequantizelinear_2d_blocked_with_zp_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [2, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=1)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def dequantizelinear_3d_blocked_with_zp_runt_block_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [2, 5, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT,
                                          [2, 2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 5, 2])

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=1,
                                 block_size=3)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def dequantizelinear_too_few_inputs_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=1)

    return ([node], [x], [y])


@onnx_test()
def dequantizelinear_too_many_inputs_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [2, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 2])
    zp2 = helper.make_tensor_value_info('zp2', TensorProto.INT8, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x', 'scale', 'zp', 'zp2'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=1)

    return ([node], [x, scale, zp, zp2], [y])


@onnx_test()
def dequantizelinear_x_and_zp_type_mismatch_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [2, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.UINT8, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=1)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def dequantizelinear_scale_and_zp_shape_mismatch_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [2, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=1)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def quantizelinear_2d_blocked_with_zp_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [6, 2])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=3)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def quantizelinear_2d_blocked_runt_block_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 5])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT,
                                            [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 5])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale'],
                                 outputs=['y'],
                                 axis=1,
                                 block_size=2)

    return ([node], [x, y_scale], [y])


@onnx_test()
def quantizelinear_3d_blocked_with_zp_runt_block_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 5, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT,
                                          [2, 2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [2, 5, 2])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=1,
                                 block_size=3)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def quantizelinear_too_few_inputs_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [6, 2])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=3)

    return ([node], [x], [y])


@onnx_test()
def quantizelinear_too_many_inputs_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 2])
    zp2 = helper.make_tensor_value_info('zp2', TensorProto.INT8, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [6, 2])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'scale', 'zp', 'zp2'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=3)

    return ([node], [x, scale, zp, zp2], [y])


@onnx_test()
def quantizelinear_scales_and_zp_shape_mismatch_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [1, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [6, 2])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=3)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def quantizelinear_output_dtype_and_zp_type_mismatch_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2, 2])
    zp = helper.make_tensor_value_info('zp', TensorProto.INT8, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [6, 2])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'scale', 'zp'],
                                 outputs=['y'],
                                 axis=0,
                                 block_size=3,
                                 output_dtype=2)

    return ([node], [x, scale, zp], [y])


@onnx_test()
def quantizelinear_per_axis_shape_mismatch_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 5])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 5])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale'],
                                 outputs=['y'],
                                 axis=1)

    return ([node], [x, y_scale], [y])


@onnx_test()
def quantizelinear_blocked_zero_block_size_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 5])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT,
                                            [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 5])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale'],
                                 outputs=['y'],
                                 axis=1,
                                 block_size=0)

    return ([node], [x, y_scale], [y])


@onnx_test()
def quantizelinear_blocked_x_and_scales_rank_mismatch_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 5])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT,
                                            [3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 5])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale'],
                                 outputs=['y'],
                                 axis=1,
                                 block_size=2)

    return ([node], [x, y_scale], [y])


@onnx_test()
def quantizelinear_blocked_non_bc_axis_size_mismatch_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 5])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT,
                                            [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 5])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale'],
                                 outputs=['y'],
                                 axis=1,
                                 block_size=2)

    return ([node], [x, y_scale], [y])


@onnx_test()
def quantizelinear_blocked_invalid_block_size_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 5])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT,
                                            [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 5])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale'],
                                 outputs=['y'],
                                 axis=1,
                                 block_size=3)

    return ([node], [x, y_scale], [y])


@onnx_test()
def dim_param_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, ["dim0", "dim1"])

    return ([], [x], [x])


@onnx_test()
def dropout_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 2, 2])

    node = onnx.helper.make_node(
        'Dropout',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def dynamicquantizelinear_1d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [6])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT, [1])
    y_zero_point = helper.make_tensor_value_info('y_zero_point',
                                                 TensorProto.UINT8, [1])

    node = onnx.helper.make_node(
        'DynamicQuantizeLinear',
        inputs=['x'],
        outputs=['y', 'y_scale', 'y_zero_point'],
    )

    return ([node], [x], [y, y_scale, y_zero_point])


@onnx_test()
def dynamicquantizelinear_2d_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 4])
    y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT, [1])
    y_zero_point = helper.make_tensor_value_info('y_zero_point',
                                                 TensorProto.UINT8, [1])

    node = onnx.helper.make_node(
        'DynamicQuantizeLinear',
        inputs=['x'],
        outputs=['y', 'y_scale', 'y_zero_point'],
    )

    return ([node], [x], [y, y_scale, y_zero_point])


@onnx_test()
def einsum_permute_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ij->ji')

    return ([node], [x], [y])


@onnx_test()
def einsum_summation_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ij->')

    return ([node], [x], [y])


@onnx_test()
def einsum_column_sum_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ij->j')

    return ([node], [x], [y])


@onnx_test()
def einsum_row_sum_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ij->i')

    return ([node], [x], [y])


@onnx_test()
def einsum_matrix_vector_multiplication_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    v = helper.make_tensor_value_info('v', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 1])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x', 'v'],
                                 outputs=['y'],
                                 equation='ij,j->i')

    return ([node], [x, v], [y])


@onnx_test()
def einsum_matrix_matrix_multiplication_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ij,kj->ik')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_vector_dot_product_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='i,i->')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_matrix_dot_product_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ij,ij->')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_hadamard_product_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ij,ij->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_vector_outer_product_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='i,j->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_matrix_outer_product_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 2, 5])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ij,kl->ijkl')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_batch_matrix_multiplication_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 2, 5])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 5, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 2, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ijk,ikl->ijl')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_tensor_contraction_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3, 5, 7])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT,
                                       [1, 3, 3, 7, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 7, 1, 3, 7])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='pqrs,tuqvr->pstuv')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_matrix_diagonal_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ii->i')

    return ([node], [x], [y])


@onnx_test()
def einsum_batch_matrix_diagonal_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='...ii->...i')

    return ([node], [x], [y])


@onnx_test()
def einsum_3d_diagonal_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='iii->i')

    return ([node], [x], [y])


@onnx_test()
def einsum_diag_vector_multiply_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,i->i')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_matrix_trace_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ii->')

    return ([node], [x], [y])


@onnx_test()
def einsum_matrix_trace_implicit_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ii')

    return ([node], [x], [y])


@onnx_test()
def einsum_2d_3d_multiplication_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ij,jkl')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_element_wise_multiplication_and_row_sum_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='i,ij->i')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_broadcast_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 1])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ij, jk -> ik')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_3d_broadcast_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [1, 3, 1])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='bik,bkj->bij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_3d_opposite_broadcast_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [1, 3, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 1, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='bik,bkj->bij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_3_inputs_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 2, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2])
    x3 = helper.make_tensor_value_info('x3', TensorProto.FLOAT, [2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2', 'x3'],
                                 outputs=['y'],
                                 equation='bac,cd,def->ebc')

    return ([node], [x1, x2, x3], [y])


@onnx_test()
def einsum_bilinear_transformation_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [5, 3, 7])
    x3 = helper.make_tensor_value_info('x3', TensorProto.FLOAT, [2, 7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 5])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2', 'x3'],
                                 outputs=['y'],
                                 equation='ik,jkl,il->ij')

    return ([node], [x1, x2, x3], [y])


@onnx_test()
def einsum_ellipsis_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='...ik,kj...->ij...')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_ellipsis_multidim_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 2, 3, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 4, 3, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 3, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='...ik,kj...->ij...')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_ellipsis_zero_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [4, 3, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 2, 4])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='...qhd,...khd->...hqk')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_ellipsis_implicit_form_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 2, 3, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 4, 3, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='...qhd,...khd')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_ellipsis_scalar_multiplication_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='..., ...')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_common_1_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 2, 2, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 2, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='bsnh,btnh->bnts')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_common_2_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 2, 2, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='bsnh,ctnh->nts')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_common_3_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 2, 2, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 2])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='bnst,chst->shn')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_common_4_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 2, 3, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 3, 4])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='bcxd,bcyd->bcxy')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_common_5_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 2, 3, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 4, 3, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3, 2, 4])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='...qhd,...khd->...hqk')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_common_6_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 2, 2])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 2, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='i...k,k...j->i...j')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_common_7_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='...j->...')

    return ([node], [x], [y])


@onnx_test()
def einsum_common_8_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,jj->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_missing_equation_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum', inputs=['x1', 'x2'], outputs=['y'])

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_multiple_arrows_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,jj->->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_empty_term_before_arrow_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_multiple_ellipses_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='......ii,...jj->...ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_comma_in_output_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,jj->i,j')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_empty_term_before_comma_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,,jj->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_last_input_missing_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,jj,')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_term_input_mismatch_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,jj,kk->ijk')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_ellipsis_mismatch_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='...ii,...jj->...ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_rank_mismatch_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='iik,jj->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_output_surplus_label_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='ii,jj->ijk')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_output_missing_ellipsis_negative_test():
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, 3, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x1', 'x2'],
                                 outputs=['y'],
                                 equation='...ii,...jj->ij')

    return ([node], [x1, x2], [y])


@onnx_test()
def einsum_multiple_diagonals_negative_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='iijj->ij')

    return ([node], [x], [y])


@onnx_test()
def einsum_diagonal_dim_mismatch_negative_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ii->i')

    return ([node], [x], [y])


@onnx_test()
def einsum_right_batch_diagonal_negative_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node('Einsum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 equation='ii...->i...')

    return ([node], [x], [y])


@onnx_test()
def elu_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Elu',
                                 inputs=['0'],
                                 outputs=['1'],
                                 alpha=0.01)

    return ([node], [x], [y])


@onnx_test()
def embedding_bag_test():

    index_val = np.array([1, 0, 2])
    offset_val = np.array([0])

    index_tensor = helper.make_tensor(name='index_val',
                                      data_type=TensorProto.INT32,
                                      dims=index_val.shape,
                                      vals=index_val.astype(np.int32))

    index = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['index'],
                                  value=index_tensor)

    offset_tensor = helper.make_tensor(name='offset_val',
                                       data_type=TensorProto.INT32,
                                       dims=offset_val.reshape(()).shape,
                                       vals=offset_val.astype(np.int32))

    offset = onnx.helper.make_node('Constant',
                                   inputs=[],
                                   outputs=['offset'],
                                   value=offset_tensor)

    weight = helper.make_tensor_value_info('weight', TensorProto.FLOAT, [4, 2])

    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [1, 2])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [1, 2])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [1, 2])

    node1 = onnx.helper.make_node('ATen',
                                  inputs=['weight', 'index', 'offset'],
                                  outputs=['y1'],
                                  mode=0,
                                  operator='embedding_bag')

    node2 = onnx.helper.make_node('ATen',
                                  inputs=['weight', 'index', 'offset'],
                                  outputs=['y2'],
                                  mode=1,
                                  operator='embedding_bag')

    node3 = onnx.helper.make_node('ATen',
                                  inputs=['weight', 'index', 'offset'],
                                  outputs=['y3'],
                                  mode=2,
                                  operator='embedding_bag')

    return ([index, offset, node1, node2, node3], [weight], [y1, y2, y3])


@onnx_test()
def embedding_bag_offset_test():

    index_val = np.array([1, 0])
    offset_val = np.array([0, 1])

    index_tensor = helper.make_tensor(name='index_val',
                                      data_type=TensorProto.INT32,
                                      dims=index_val.shape,
                                      vals=index_val.astype(np.int32))

    index = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['index'],
                                  value=index_tensor)

    offset_tensor = helper.make_tensor(name='offset_val',
                                       data_type=TensorProto.INT32,
                                       dims=offset_val.shape,
                                       vals=offset_val.astype(np.int32))

    offset = onnx.helper.make_node('Constant',
                                   inputs=[],
                                   outputs=['offset'],
                                   value=offset_tensor)

    weight = helper.make_tensor_value_info('weight', TensorProto.FLOAT, [2, 3])

    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('ATen',
                                 inputs=['weight', 'index', 'offset'],
                                 outputs=['y'],
                                 mode=0,
                                 operator='embedding_bag')

    return ([index, offset, node], [weight], [y])


@onnx_test()
def equal_test():
    ax1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x1 = helper.make_tensor("x1",
                            data_type=TensorProto.FLOAT,
                            dims=(2, 3),
                            vals=ax1.astype(np.float32))

    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'Equal',
        inputs=['x1', 'x2'],
        outputs=['y'],
    )

    return ([node], [x2], [y], [x1])


@onnx_test()
def equal_bool_test():

    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.BOOL, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node1 = onnx.helper.make_node('Cast', inputs=['x1'], outputs=['bx1'], to=9)

    node2 = onnx.helper.make_node(
        'Equal',
        inputs=['bx1', 'x2'],
        outputs=['y'],
    )

    return ([node1, node2], [x1, x2], [y])


@onnx_test()
def erf_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 15])

    node = onnx.helper.make_node(
        'Erf',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def exp_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Exp',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def expand_test():
    shape_val = np.array([2, 3, 4, 5]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(int))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Expand',
                                 inputs=['x', 'shape'],
                                 outputs=['y'])

    return ([shape_const, node], [x], [y])


@onnx_test()
def expand_static_input_dyn_output_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1, 1])
    dims_in = helper.make_tensor_value_info('dims', TensorProto.INT64, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Expand', inputs=['x', 'dims'], outputs=['y'])

    return ([node], [x, dims_in], [y])


@onnx_test()
def expand_dyn_input_dyn_output_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 1, 1])
    dims_in = helper.make_tensor_value_info('dims', TensorProto.INT64, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Expand', inputs=['x', 'dims'], outputs=['y'])

    return ([node], [x, dims_in], [y])


@onnx_test()
def expand_dyn_input_static_dims_throw():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 4])

    shape_val = np.array([3, 4, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(int))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )

    node = onnx.helper.make_node('Expand',
                                 inputs=['x', 'shape'],
                                 outputs=['y'])

    return ([shape_const, node], [x], [y])


@onnx_test(True)
def external_constant_test():
    x = np.array([0, 1, 2])
    y = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])

    tensor = from_array(x)
    tensor.name = 'const_tensor'

    node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['0'],
                                 value=tensor)

    return ([node], [], [y])


@onnx_test()
def eyelike_default_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [3, 4])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT, [3, 4])

    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['T1'],
        outputs=['T2'],
    )
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_double_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.DOUBLE, [6, 15])
    T2 = helper.make_tensor_value_info('T2', TensorProto.DOUBLE, [6, 15])

    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['T1'],
        outputs=['T2'],
    )
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_half_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT16, [8, 8])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT16, [8, 8])

    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['T1'],
        outputs=['T2'],
    )
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_bf16_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.BFLOAT16, [8, 8])
    T2 = helper.make_tensor_value_info('T2', TensorProto.BFLOAT16, [8, 8])

    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['T1'],
        outputs=['T2'],
    )
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_k_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [3, 4])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT, [3, 4])
    node = onnx.helper.make_node('EyeLike', inputs=['T1'], outputs=['T2'], k=1)
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_k_outofbounds_neg_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [2, 4])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT, [2, 4])
    node = onnx.helper.make_node('EyeLike',
                                 inputs=['T1'],
                                 outputs=['T2'],
                                 k=-2)
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_k_outofbounds_pos_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [3, 4])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT, [3, 4])
    node = onnx.helper.make_node('EyeLike', inputs=['T1'], outputs=['T2'], k=4)
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_not_rank2_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [3, 4, 2])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT, [3, 4])
    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['T1'],
        outputs=['T2'],
    )
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_verify_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [3, 4])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT, [3, 4])
    node = onnx.helper.make_node('EyeLike', inputs=['T1'], outputs=['T2'], k=1)
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_verify_negk_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [3, 4])
    T2 = helper.make_tensor_value_info('T2', TensorProto.FLOAT, [3, 4])
    node = onnx.helper.make_node('EyeLike',
                                 inputs=['T1'],
                                 outputs=['T2'],
                                 k=-2)
    return ([node], [T1], [T2])


@onnx_test()
def eyelike_set_dtype_test():
    T1 = helper.make_tensor_value_info('T1', TensorProto.FLOAT, [3, 4])
    T2 = helper.make_tensor_value_info('T2', TensorProto.DOUBLE, [3, 4])
    node = onnx.helper.make_node('EyeLike',
                                 inputs=['T1'],
                                 outputs=['T2'],
                                 dtype=TensorProto.DOUBLE)
    return ([node], [T1], [T2])


@onnx_test()
def flatten_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [6, 20])
    y2 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 60])

    node = onnx.helper.make_node('Flatten',
                                 inputs=['0'],
                                 axis=2,
                                 outputs=['2'])

    node2 = onnx.helper.make_node('Flatten', inputs=['0'], outputs=['3'])

    return ([node, node2], [x], [y, y2])


@onnx_test()
def flatten_nonstd_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 5, 4])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [6, 20])
    y2 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 60])

    trans = helper.make_node(
        'Transpose',
        inputs=['0'],
        outputs=['tx'],
        perm=[0, 1, 3, 2],
    )

    node = onnx.helper.make_node('Flatten',
                                 inputs=['tx'],
                                 axis=2,
                                 outputs=['2'])

    node2 = onnx.helper.make_node('Flatten', inputs=['tx'], outputs=['3'])

    return ([trans, node, node2], [x], [y, y2])


@onnx_test()
def flatten_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 3, 4, 5])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [None, 20])

    node = onnx.helper.make_node('Flatten',
                                 inputs=['0'],
                                 axis=2,
                                 outputs=['2'])

    return ([node], [x], [y])


@onnx_test()
def floor_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Floor',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def gather_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32,
                                      [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Gather',
        inputs=['data', 'indices'],
        outputs=['y'],
        axis=1,
    )

    return ([node], [x, i], [y])


@onnx_test()
def gather_scalar_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32, [])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 5, 6])

    node = onnx.helper.make_node(
        'Gather',
        inputs=['data', 'indices'],
        outputs=['y'],
        axis=1,
    )

    return ([node], [x, i], [y])


@onnx_test()
def gather_dyn_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT,
                                      [None, 4, 5, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32,
                                      [None, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Gather',
        inputs=['data', 'indices'],
        outputs=['y'],
        axis=1,
    )

    return ([node], [x, i], [y])


@onnx_test()
def gather_elements_axis0_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'GatherElements',
        inputs=['data', 'indices'],
        outputs=['y'],
        axis=0,
    )

    return ([node], [x, i], [y])


@onnx_test()
def gather_elements_axis1_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'GatherElements',
        inputs=['data', 'indices'],
        outputs=['y'],
        axis=1,
    )

    return ([node], [x, i], [y])


@onnx_test()
def gathernd_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 2])
    i = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])

    node = onnx.helper.make_node('GatherND',
                                 inputs=['data', 'indices'],
                                 outputs=['y'])

    return ([node], [x, i], [y])


@onnx_test()
def gathernd_dyn_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [None, 2])
    i = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])

    node = onnx.helper.make_node('GatherND',
                                 inputs=['data', 'indices'],
                                 outputs=['y'])

    return ([node], [x, i], [y])


@onnx_test()
def gathernd_batch_dims_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 2, 2])
    i = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node(
        'GatherND',
        inputs=['data', 'indices'],
        outputs=['y'],
        batch_dims=1,
    )

    return ([node], [x, i], [y])


@onnx_test()
def gelu_default_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("Gelu", inputs=["x"], outputs=["y"])

    return ([node], [x], [y])


@onnx_test()
def gelu_default_half_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [3, 3])

    node = onnx.helper.make_node("Gelu", inputs=["x"], outputs=["y"])

    return ([node], [x], [y])


# @onnx_test()
# def gelu_default_bf16_test():
#     x = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [3, 3])
#     y = helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [3, 3])

#     node = onnx.helper.make_node("Gelu", inputs=["x"], outputs=["y"])

#     return ([node], [x], [y])


@onnx_test()
def gelu_tanh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("Gelu",
                                 inputs=["x"],
                                 outputs=["y"],
                                 approximate="tanh")

    return ([node], [x], [y])


@onnx_test()
def gelu_tanh_double_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [3, 3])

    node = onnx.helper.make_node("Gelu",
                                 inputs=["x"],
                                 outputs=["y"],
                                 approximate="tanh")

    return ([node], [x], [y])


@onnx_test()
def gelu_invalid_input_type_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [3])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [3])

    node = onnx.helper.make_node("Gelu", inputs=["x"], outputs=["y"])

    return ([node], [x], [y])


@onnx_test()
def gelu_add_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("BiasGelu",
                                 inputs=["x", "y"],
                                 outputs=["z"],
                                 domain="com.microsoft")

    return ([node], [x, y], [z])


@onnx_test()
def gelu_bias_invalid_type_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [3])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("BiasGelu",
                                 inputs=["x", "y"],
                                 outputs=["z"],
                                 domain="com.microsoft")

    return ([node], [x, y], [z])


@onnx_test()
def gelu_fast_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("FastGelu",
                                 inputs=["x"],
                                 outputs=["y"],
                                 domain="com.microsoft")

    return ([node], [x], [y])


@onnx_test()
def gelu_fast_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [3, 3])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT16, [3, 3])

    node = onnx.helper.make_node("FastGelu",
                                 inputs=["x", "y"],
                                 outputs=["z"],
                                 domain="com.microsoft")

    return ([node], [x, y], [z])


@onnx_test()
def gelu_fast_invalid_x_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("FastGelu",
                                 inputs=["x", "y"],
                                 outputs=["z"],
                                 domain="com.microsoft")

    return ([node], [x, y], [z])


@onnx_test()
def gelu_fast_invalid_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [3, 3])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("FastGelu",
                                 inputs=["x", "y"],
                                 outputs=["z"],
                                 domain="com.microsoft")

    return ([node], [x, y], [z])


@onnx_test()
def gelu_quick_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])

    node = onnx.helper.make_node("QuickGelu",
                                 inputs=["x"],
                                 outputs=["y"],
                                 alpha=0.5,
                                 domain="com.microsoft")

    return ([node], [x], [y])


@onnx_test()
def gemm_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [8, 6])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [8, 7])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [6, 7])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def gemm_no_C_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [5, 7])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [11, 5])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [7, 11])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=2.0,
                                 beta=2.0,
                                 transA=1,
                                 transB=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def gemm_brcst_C_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [5, 6])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [5, 7])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [6, 1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def gemm_half_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT16, [8, 6])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT16, [8, 7])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT16, [6, 1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT16, [6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def gemm_bf16_test():
    A = helper.make_tensor_value_info('A', TensorProto.BFLOAT16, [8, 6])
    B = helper.make_tensor_value_info('B', TensorProto.BFLOAT16, [8, 7])
    C = helper.make_tensor_value_info('C', TensorProto.BFLOAT16, [6, 1])
    Y = helper.make_tensor_value_info('Y', TensorProto.BFLOAT16, [6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def gemm_fp8_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT8E4M3FNUZ, [8, 6])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT8E4M3FNUZ, [8, 7])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT8E4M3FNUZ, [6, 1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT8E4M3FNUZ, [6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def gemm_dyn_inner_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [None, 6])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [None, 7])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B'],
                                 outputs=['Y'],
                                 alpha=0.5,
                                 transA=1)

    return ([node], [A, B], [Y])


@onnx_test()
def gemm_dyn_outer_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [5, None])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [11, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 11])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B'],
                                 outputs=['Y'],
                                 alpha=2.0,
                                 transA=1,
                                 transB=1)

    return ([node], [A, B], [Y])


@onnx_test()
def gemm_dyn_bias_test():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [8, None])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [8, 7])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 7])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=1.0,
                                 beta=1.0,
                                 transA=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def gemm_rank_error():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [4, 1, 8, 6])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4, 1, 8, 7])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [6, 7])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4, 1, 6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['A', 'B', 'C'],
                                 outputs=['Y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [A, B, C], [Y])


@onnx_test()
def globalavgpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def globalavgpool_fp8_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT8E4M3FNUZ,
                                      [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT8E4M3FNUZ,
                                      [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def globalavgpool_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [None, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def globallppool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalLpPool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def globallppool_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [1, 3, None, None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalLpPool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def globalmaxpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def globalmaxpool_fp8_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT8E4M3FNUZ,
                                      [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT8E4M3FNUZ,
                                      [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def globalmaxpool_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [None, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def greater_test():
    ax1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x1 = helper.make_tensor("x1",
                            data_type=TensorProto.FLOAT,
                            dims=(2, 3),
                            vals=ax1.astype(np.float32))

    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'Greater',
        inputs=['x1', 'x2'],
        outputs=['y'],
    )

    return ([node], [x2], [y], [x1])


@onnx_test()
def greater_bool_test():

    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.BOOL, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node1 = onnx.helper.make_node('Cast', inputs=['x1'], outputs=['bx1'], to=9)

    node2 = onnx.helper.make_node(
        'Greater',
        inputs=['bx1', 'x2'],
        outputs=['y'],
    )

    return ([node1, node2], [x1, x2], [y])


@onnx_test()
def greaterorequal_test():

    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'GreaterOrEqual',
        inputs=['x1', 'x2'],
        outputs=['y'],
    )

    return ([node], [x1, x2], [y])


@onnx_test()
def gridsample_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 4, 4])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 6, 6, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 6, 6])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_half_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 1, 4, 4])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 6, 6, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 1, 6, 6])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_bf16_test():
    x = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [1, 1, 4, 4])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 6, 6, 2])
    y = helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [1, 1, 6, 6])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_int_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [1, 1, 4, 4])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 6, 6, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [1, 1, 6, 6])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_simple_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 1, 1, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 1, 1])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_bilinear_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        align_corners=0,
        mode="linear",
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_aligncorners_true_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
        align_corners=1,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_nearest_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="nearest",
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_bicubic_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="cubic",
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_bicubic_align_corners_0_additional_1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="cubic",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_bicubic_align_corners_1_additional_1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="cubic",
        align_corners=1,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_nearest_align_corners_0_additional_1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="nearest",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_nearest_align_corners_1_additional_1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="nearest",
        align_corners=1,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_bilinear_align_corners_0_additional_1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_bilinear_align_corners_1_additional_1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="linear",
        align_corners=1,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_zeros_padding_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        padding_mode="zeros",
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_border_padding_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        padding_mode="border",
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_reflection_padding_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        padding_mode="reflection",
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_volumetric_nearest_align_corners_0_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4, 2])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
        mode="nearest",
        align_corners=0,
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_wrong_grid_type_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.INT32,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
    )

    return ([node], [x, grid], [y])


@onnx_test()
def gridsample_mismatching_dims_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 2, 2])
    grid = helper.make_tensor_value_info('grid', TensorProto.FLOAT,
                                         [1, 2, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 4])

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=["y"],
    )

    return ([node], [x, grid], [y])


@onnx_test()
def group_conv_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 4, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 1, 3, 3])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 4, 14, 14])

    node = onnx.helper.make_node(
        'Conv',
        inputs=['0', '1'],
        group=4,
        outputs=['2'],
    )

    return ([node], [x, y], [z])


def group_norm_test(x_dims,
                    scale_dims,
                    bias_dims,
                    y_dims,
                    num_groups,
                    eps_value=1e-5,
                    dtype=TensorProto.FLOAT):
    x = helper.make_tensor_value_info('x', dtype, x_dims)
    scale = helper.make_tensor_value_info('scale', dtype, scale_dims)
    bias = helper.make_tensor_value_info('bias', dtype, bias_dims)
    y = helper.make_tensor_value_info('y', dtype, y_dims)

    node = onnx.helper.make_node('GroupNormalization',
                                 inputs=['x', 'scale', 'bias'],
                                 outputs=['y'],
                                 num_groups=num_groups,
                                 epsilon=eps_value)

    return ([node], [x, scale, bias], [y])


@onnx_test()
def group_norm_3d_test():
    return group_norm_test([1, 4, 2], [2], [2], [1, 4, 2], 2)


@onnx_test()
def group_norm_3d_half_test():
    return group_norm_test([1, 4, 2], [2], [2], [1, 4, 2],
                           2,
                           dtype=TensorProto.FLOAT16)


@onnx_test()
def group_norm_3d_bf16_test():
    return group_norm_test([1, 4, 2], [2], [2], [1, 4, 2],
                           2,
                           dtype=TensorProto.BFLOAT16)


@onnx_test()
def group_norm_4d_test():
    return group_norm_test([1, 4, 3, 3], [2], [2], [1, 4, 3, 3], 2)


@onnx_test()
def group_norm_4d_half_test():
    return group_norm_test([1, 4, 3, 3], [2], [2], [1, 4, 3, 3],
                           2,
                           dtype=TensorProto.FLOAT16)


@onnx_test()
def group_norm_4d_bf16_test():
    return group_norm_test([1, 4, 3, 3], [2], [2], [1, 4, 3, 3],
                           2,
                           dtype=TensorProto.BFLOAT16)


@onnx_test()
def group_norm_5d_test():
    return group_norm_test([3, 3, 3, 3, 3], [1], [1], [3, 3, 3, 3, 3], 1)


@onnx_test()
def group_norm_5d_half_test():
    return group_norm_test([3, 3, 3, 3, 3], [1], [1], [3, 3, 3, 3, 3],
                           1,
                           dtype=TensorProto.FLOAT16)


@onnx_test()
def group_norm_5d_bf16_test():
    return group_norm_test([3, 3, 3, 3, 3], [1], [1], [3, 3, 3, 3, 3],
                           1,
                           dtype=TensorProto.BFLOAT16)


@onnx_test()
def group_norm_small_eps_half_test():
    return group_norm_test([1, 4, 2], [2], [2], [1, 4, 2],
                           2,
                           eps_value=1e-12,
                           dtype=TensorProto.FLOAT16)


@onnx_test()
def group_norm_small_eps_bf16_test():
    return group_norm_test([1, 4, 2], [2], [2], [1, 4, 2],
                           2,
                           eps_value=1e-7,
                           dtype=TensorProto.BFLOAT16)


@onnx_test()
def group_norm_invalid_num_groups_error_test():
    return group_norm_test([1, 4, 3, 3], [2], [2], [1, 4, 3, 3], 3)


@onnx_test()
def group_norm_missing_attribute_error_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])

    node = onnx.helper.make_node('GroupNormalization',
                                 inputs=['x', 'scale', 'bias'],
                                 outputs=['y'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def group_norm_invalid_input_count_error_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 3, 3])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 3, 3])

    node = onnx.helper.make_node('GroupNormalization',
                                 inputs=['x', 'scale'],
                                 outputs=['y'],
                                 num_groups=2)

    return ([node], [x, scale], [y])


@onnx_test()
def group_norm_invalid_input_shape_error_test():
    return group_norm_test([1, 4], [2], [2], [1, 4], 2)


@onnx_test()
def group_norm_invalid_scale_shape_test():
    return group_norm_test([1, 4, 3, 3], [1], [2], [1, 4, 3, 3], 2)


@onnx_test()
def group_norm_invalid_bias_shape_test():
    return group_norm_test([1, 4, 3, 3], [2], [3], [1, 4, 3, 3], 2)


def group_norm_contrib_test(x_dims,
                            gamma_dims,
                            beta_dims,
                            y_dims,
                            num_groups,
                            activation,
                            channels_last,
                            eps_value=1e-5,
                            dtype=TensorProto.FLOAT):
    x = helper.make_tensor_value_info('x', dtype, x_dims)
    gamma = helper.make_tensor_value_info('gamma', dtype, gamma_dims)
    beta = helper.make_tensor_value_info('beta', dtype, beta_dims)
    y = helper.make_tensor_value_info('y', dtype, y_dims)

    node = onnx.helper.make_node('GroupNorm',
                                 inputs=['x', 'gamma', 'beta'],
                                 outputs=['y'],
                                 activation=activation,
                                 channels_last=channels_last,
                                 groups=num_groups,
                                 epsilon=eps_value)

    return ([node], [x, gamma, beta], [y])


@onnx_test()
def group_norm_contrib_3d_test():
    return group_norm_contrib_test([1, 4, 2], [2], [2], [1, 4, 2], 2, 0, 0)


@onnx_test()
def group_norm_contrib_3d_channel_last_test():
    return group_norm_contrib_test([1, 4, 2], [2], [2], [1, 4, 2], 2, 0, 1)


@onnx_test()
def group_norm_contrib_3d_channel_last_half_test():
    return group_norm_contrib_test([1, 4, 2], [2], [2], [1, 4, 2],
                                   2,
                                   0,
                                   1,
                                   dtype=TensorProto.FLOAT16)


@onnx_test()
def group_norm_contrib_3d_channel_last_bf16_test():
    return group_norm_contrib_test([1, 4, 2], [2], [2], [1, 4, 2],
                                   2,
                                   0,
                                   1,
                                   dtype=TensorProto.BFLOAT16)


@onnx_test()
def group_norm_contrib_silu_3d_test():
    return group_norm_contrib_test([1, 4, 2], [2], [2], [1, 4, 2], 2, 1, 0)


@onnx_test()
def group_norm_contrib_channels_last_3d_test():
    return group_norm_contrib_test([1, 4, 2], [2], [2], [1, 4, 2], 2, 0, 1)


@onnx_test()
def group_norm_contrib_channels_last_4d_test():
    return group_norm_contrib_test([1, 3, 3, 4], [2], [2], [1, 3, 3, 4], 2, 0,
                                   1)


@onnx_test()
def group_norm_contrib_channels_last_and_silu_3d_test():
    return group_norm_contrib_test([1, 4, 2], [2], [2], [1, 4, 2], 2, 1, 1)


@onnx_test()
def group_norm_contrib_no_activation_attr_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 2])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT, [2])
    beta = helper.make_tensor_value_info('beta', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 2])

    node = onnx.helper.make_node('GroupNorm',
                                 inputs=['x', 'gamma', 'Beta'],
                                 outputs=['y'],
                                 channels_last=0,
                                 groups=2)

    return ([node], [x, gamma, beta], [y])


@onnx_test()
def group_norm_contrib_no_num_groups_attr_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 2])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT, [2])
    beta = helper.make_tensor_value_info('beta', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 2])

    node = onnx.helper.make_node('GroupNorm',
                                 inputs=['x', 'gamma', 'Beta'],
                                 outputs=['y'],
                                 activation=0,
                                 channels_last=0)

    return ([node], [x, gamma, beta], [y])


@onnx_test()
def group_query_attention_test():
    qkv = helper.make_tensor_value_info('qkv', TensorProto.FLOAT16,
                                        [1, 1, 12288])
    key = helper.make_tensor_value_info('key', TensorProto.FLOAT, [1])
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [1])
    past_key_values_key = helper.make_tensor_value_info(
        'past_key_values_key', TensorProto.FLOAT16, [1, 32, 4096, 128])
    past_key_values_value = helper.make_tensor_value_info(
        'past_key_values_value', TensorProto.FLOAT16, [1, 32, 4096, 128])
    slk_val = np.array([1])
    seqlens_k = helper.make_tensor(name="seqlens_k",
                                   data_type=TensorProto.INT32,
                                   dims=slk_val.shape,
                                   vals=slk_val.astype(int))
    tsl_val = np.array([2])
    total_sequence_length = helper.make_tensor(name="total_sequence_length",
                                               data_type=TensorProto.INT32,
                                               dims=tsl_val.shape,
                                               vals=tsl_val.astype(int))
    cc_val = np.ones([4096, 64], dtype=np.float16)
    cos_cache = helper.make_tensor(name="cos_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    sin_cache = helper.make_tensor(name="sin_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 1, 4096])
    present_key = helper.make_tensor_value_info('present_key',
                                                TensorProto.FLOAT16,
                                                [1, 32, 4096, 128])
    present_value = helper.make_tensor_value_info('present_value',
                                                  TensorProto.FLOAT16,
                                                  [1, 32, 4096, 128])

    node = onnx.helper.make_node(
        'GroupQueryAttention',
        inputs=[
            'qkv', 'key', 'value', 'past_key_values_key',
            'past_key_values_value', 'seqlens_k', 'total_sequence_length',
            'cos_cache', 'sin_cache'
        ],
        outputs=['output', 'present_key', 'present_value'],
        do_rotary=1,
        kv_num_heads=32,
        local_window_size=-1,
        num_heads=32,
        rotary_interleaved=0,
        scale=1.0,
        domain="com.microsoft")

    return ([node
             ], [qkv, key, value, past_key_values_key,
                 past_key_values_value], [output, present_key, present_value],
            [seqlens_k, total_sequence_length, cos_cache, sin_cache])


@onnx_test()
def group_query_attention_non_packed_qkv_test():
    query = helper.make_tensor_value_info('query', TensorProto.FLOAT16,
                                          [1, 1, 4096])
    key = helper.make_tensor_value_info('key', TensorProto.FLOAT16,
                                        [1, 1, 4096])
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT16,
                                          [1, 1, 4096])
    past_key_values_key = helper.make_tensor_value_info(
        'past_key_values_key', TensorProto.FLOAT16, [1, 32, 4096, 128])
    past_key_values_value = helper.make_tensor_value_info(
        'past_key_values_value', TensorProto.FLOAT16, [1, 32, 4096, 128])
    slk_val = np.array([1])
    seqlens_k = helper.make_tensor(name="seqlens_k",
                                   data_type=TensorProto.INT32,
                                   dims=slk_val.shape,
                                   vals=slk_val.astype(int))
    tsl_val = np.array([2])
    total_sequence_length = helper.make_tensor(name="total_sequence_length",
                                               data_type=TensorProto.INT32,
                                               dims=tsl_val.shape,
                                               vals=tsl_val.astype(int))
    cc_val = np.ones([4096, 64], dtype=np.float16)
    cos_cache = helper.make_tensor(name="cos_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    sin_cache = helper.make_tensor(name="sin_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 1, 4096])
    present_key = helper.make_tensor_value_info('present_key',
                                                TensorProto.FLOAT16,
                                                [1, 32, 4096, 128])
    present_value = helper.make_tensor_value_info('present_value',
                                                  TensorProto.FLOAT16,
                                                  [1, 32, 4096, 128])

    node = onnx.helper.make_node(
        'GroupQueryAttention',
        inputs=[
            'query', 'key', 'value', 'past_key_values_key',
            'past_key_values_value', 'seqlens_k', 'total_sequence_length',
            'cos_cache', 'sin_cache'
        ],
        outputs=['output', 'present_key', 'present_value'],
        do_rotary=1,
        kv_num_heads=32,
        local_window_size=-1,
        num_heads=32,
        rotary_interleaved=0,
        scale=1.0,
        domain="com.microsoft")

    return ([node],
            [query, key, value, past_key_values_key,
             past_key_values_value], [output, present_key, present_value],
            [seqlens_k, total_sequence_length, cos_cache, sin_cache])


@onnx_test()
def group_query_attention_defaults_test():
    qkv = helper.make_tensor_value_info('qkv', TensorProto.FLOAT16,
                                        [1, 1, 12288])
    key = helper.make_tensor_value_info('key', TensorProto.FLOAT, [1])
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [1])
    past_key_values_key = helper.make_tensor_value_info(
        'past_key_values_key', TensorProto.FLOAT16, [1, 32, 4096, 128])
    past_key_values_value = helper.make_tensor_value_info(
        'past_key_values_value', TensorProto.FLOAT16, [1, 32, 4096, 128])
    slk_val = np.array([1])
    seqlens_k = helper.make_tensor(name="seqlens_k",
                                   data_type=TensorProto.INT32,
                                   dims=slk_val.shape,
                                   vals=slk_val.astype(int))
    tsl_val = np.array([2])
    total_sequence_length = helper.make_tensor(name="total_sequence_length",
                                               data_type=TensorProto.INT32,
                                               dims=tsl_val.shape,
                                               vals=tsl_val.astype(int))
    cc_val = np.ones([4096, 64], dtype=np.float16)
    cos_cache = helper.make_tensor(name="cos_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    sin_cache = helper.make_tensor(name="sin_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 1, 4096])
    present_key = helper.make_tensor_value_info('present_key',
                                                TensorProto.FLOAT16,
                                                [1, 32, 4096, 128])
    present_value = helper.make_tensor_value_info('present_value',
                                                  TensorProto.FLOAT16,
                                                  [1, 32, 4096, 128])

    node = onnx.helper.make_node(
        'GroupQueryAttention',
        inputs=[
            'qkv', 'key', 'value', 'past_key_values_key',
            'past_key_values_value', 'seqlens_k', 'total_sequence_length',
            'cos_cache', 'sin_cache'
        ],
        outputs=['output', 'present_key', 'present_value'],
        domain="com.microsoft")

    return ([node
             ], [qkv, key, value, past_key_values_key,
                 past_key_values_value], [output, present_key, present_value],
            [seqlens_k, total_sequence_length, cos_cache, sin_cache])


@onnx_test()
def group_query_attention_invalid_test():
    qkv = helper.make_tensor_value_info('qkv', TensorProto.FLOAT16,
                                        [1, 1, 12288])
    past_key_values_key = helper.make_tensor_value_info(
        'past_key_values_key', TensorProto.FLOAT16, [1, 32, 4096, 128])
    past_key_values_value = helper.make_tensor_value_info(
        'past_key_values_value', TensorProto.FLOAT16, [1, 32, 4096, 128])
    slk_val = np.array([1])
    seqlens_k = helper.make_tensor(name="seqlens_k",
                                   data_type=TensorProto.INT32,
                                   dims=slk_val.shape,
                                   vals=slk_val.astype(int))
    tsl_val = np.array([2])
    total_sequence_length = helper.make_tensor(name="total_sequence_length",
                                               data_type=TensorProto.INT32,
                                               dims=tsl_val.shape,
                                               vals=tsl_val.astype(int))
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 1, 4096])
    present_key = helper.make_tensor_value_info('present_key',
                                                TensorProto.FLOAT16,
                                                [1, 32, 4096, 128])
    present_value = helper.make_tensor_value_info('present_value',
                                                  TensorProto.FLOAT16,
                                                  [1, 32, 4096, 128])

    node = onnx.helper.make_node(
        'GroupQueryAttention',
        inputs=[
            'qkv', 'past_key_values_key', 'past_key_values_value', 'seqlens_k',
            'total_sequence_length', 'cos_cache', 'sin_cache'
        ],
        outputs=['output', 'present_key', 'present_value'],
        do_rotary=1,
        kv_num_heads=32,
        local_window_size=-1,
        num_heads=32,
        rotary_interleaved=0,
        scale=1.0,
        domain="com.microsoft")

    return ([node], [qkv, past_key_values_key, past_key_values_value],
            [output, present_key,
             present_value], [seqlens_k, total_sequence_length])


@onnx_test()
def group_query_attention_softcap_test():
    qkv = helper.make_tensor_value_info('qkv', TensorProto.FLOAT16,
                                        [1, 1, 12288])
    key = helper.make_tensor_value_info('key', TensorProto.FLOAT, [1])
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [1])
    past_key_values_key = helper.make_tensor_value_info(
        'past_key_values_key', TensorProto.FLOAT16, [1, 32, 4096, 128])
    past_key_values_value = helper.make_tensor_value_info(
        'past_key_values_value', TensorProto.FLOAT16, [1, 32, 4096, 128])
    slk_val = np.array([1])
    seqlens_k = helper.make_tensor(name="seqlens_k",
                                   data_type=TensorProto.INT32,
                                   dims=slk_val.shape,
                                   vals=slk_val.astype(int))
    tsl_val = np.array([2])
    total_sequence_length = helper.make_tensor(name="total_sequence_length",
                                               data_type=TensorProto.INT32,
                                               dims=tsl_val.shape,
                                               vals=tsl_val.astype(int))
    cc_val = np.ones([4096, 64], dtype=np.float16)
    cos_cache = helper.make_tensor(name="cos_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    sin_cache = helper.make_tensor(name="sin_cache",
                                   data_type=TensorProto.FLOAT16,
                                   dims=cc_val.shape,
                                   vals=cc_val)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 1, 4096])
    present_key = helper.make_tensor_value_info('present_key',
                                                TensorProto.FLOAT16,
                                                [1, 32, 4096, 128])
    present_value = helper.make_tensor_value_info('present_value',
                                                  TensorProto.FLOAT16,
                                                  [1, 32, 4096, 128])

    node = onnx.helper.make_node(
        'GroupQueryAttention',
        inputs=[
            'qkv', 'key', 'value', 'past_key_values_key',
            'past_key_values_value', 'seqlens_k', 'total_sequence_length',
            'cos_cache', 'sin_cache'
        ],
        outputs=['output', 'present_key', 'present_value'],
        softcap=1.0,
        domain="com.microsoft")

    return ([node
             ], [qkv, key, value, past_key_values_key,
                 past_key_values_value], [output, present_key, present_value],
            [seqlens_k, total_sequence_length, cos_cache, sin_cache])


@onnx_test()
def gru_bi_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [2, 60, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [2, 60, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, 120])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 2, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 2, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 2, 20])

    node = onnx.helper.make_node(
        'GRU',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0'],
        outputs=['hs', 'output'],
        activations=['tanh', 'sigmoid', 'relu', 'tanh'],
        clip=0,
        direction='bidirectional',
        hidden_size=20,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0], [hs, output])


@onnx_test()
def gru_bi_5arg_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [2, 60, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [2, 60, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, 120])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 2, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 2, 20])

    node = onnx.helper.make_node(
        'GRU',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len'],
        outputs=['hs', 'output'],
        activations=['tanh', 'sigmoid', 'relu', 'tanh'],
        clip=0,
        direction='bidirectional',
        hidden_size=20,
        linear_before_reset=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len], [hs, output])


@onnx_test()
def gru_f_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 60, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 60, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 120])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node(
        'GRU',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0'],
        outputs=['hs', 'output'],
        activations=['tanh', 'sigmoid'],
        clip=0,
        direction='forward',
        hidden_size=20,
        linear_before_reset=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0], [hs, output])


@onnx_test()
def gru_f_3arg_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 60, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 60, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node('GRU',
                                 inputs=['seq', 'w', 'r'],
                                 outputs=['hs', 'output'],
                                 activations=['tanh', 'sigmoid'],
                                 clip=0,
                                 direction='forward',
                                 hidden_size=20,
                                 layout=1)

    return ([node], [seq, w, r], [hs, output])


@onnx_test()
def gru_f_1af_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [5, 3, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 60, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 60, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [5, 1, 3, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [1, 3, 20])

    node = onnx.helper.make_node('GRU',
                                 inputs=['seq', 'w', 'r'],
                                 outputs=['hs', 'output'],
                                 activations=['tanh'],
                                 clip=0,
                                 direction='forward',
                                 hidden_size=20)

    return ([node], [seq, w, r], [hs, output])


@onnx_test()
def gru_r_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 60, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 60, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 120])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node(
        'GRU',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0'],
        outputs=['hs', 'output'],
        activations=['tanh', 'sigmoid'],
        clip=0,
        direction='reverse',
        hidden_size=20,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0], [hs, output])


@onnx_test()
def gru_r_4arg_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 60, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 60, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 120])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node('GRU',
                                 inputs=['seq', 'w', 'r', 'bias'],
                                 outputs=['hs', 'output'],
                                 activations=['relu', 'tanh'],
                                 clip=0,
                                 direction='reverse',
                                 hidden_size=20,
                                 linear_before_reset=1,
                                 layout=1)

    return ([node], [seq, w, r, bias], [hs, output])


@onnx_test()
def hardmax_default_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 3, 4])

    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def hardmax_axis_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [1, 2, 3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [1, 2, 3, 4])

    node = onnx.helper.make_node('Hardmax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=2)

    return ([node], [x], [y])


@onnx_test()
def hardmax_axis_neg_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 2, 3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 2, 3, 4])

    node = onnx.helper.make_node('Hardmax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=-3)

    return ([node], [x], [y])


@onnx_test()
def hardsigmoid_default_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 4, 5])

    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def hardsigmoid_double_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [1, 3, 4, 5])

    node = onnx.helper.make_node('HardSigmoid',
                                 inputs=['x'],
                                 outputs=['y'],
                                 alpha=0.3,
                                 beta=0.7)

    return ([node], [x], [y])


@onnx_test()
def hardsigmoid_half_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 3, 4, 5])

    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def hardsigmoid_bf16_test():
    x = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [1, 3, 4, 5])

    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def hardsigmoid_verify_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 5])

    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def hardswish_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 5])

    node = onnx.helper.make_node('HardSwish', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def if_else_test():
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])

    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    xt = np.ones((2, 3)).astype(np.float)
    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt = np.random.randn(2, 3).astype(np.float)
    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    then_body = onnx.helper.make_graph([then_add_node], 'then_body', [],
                                       [then_out])

    else_body = onnx.helper.make_graph([else_mul_node], 'else_body', [],
                                       [else_out])

    cond_tensor = onnx.helper.make_tensor_value_info("cond",
                                                     onnx.TensorProto.BOOL,
                                                     [1])
    res = onnx.helper.make_tensor_value_info('res', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['res'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [x, y, cond_tensor], [res], [xt_tensor, yt_tensor])


@onnx_test()
def if_else_test_inlined():
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])

    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    xt = np.ones((2, 3)).astype(np.float)
    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt = np.random.randn(2, 3).astype(np.float)
    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    then_body = onnx.helper.make_graph([then_add_node], 'then_body', [],
                                       [then_out])

    else_body = onnx.helper.make_graph([else_mul_node], 'else_body', [],
                                       [else_out])

    cond = np.array([0]).astype(bool)
    cond_tensor = helper.make_tensor(name="cond",
                                     data_type=TensorProto.BOOL,
                                     dims=cond.shape,
                                     vals=cond.astype(bool))
    res = onnx.helper.make_tensor_value_info('res', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['res'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [x, y], [res], [cond_tensor, xt_tensor, yt_tensor])


@onnx_test()
def if_then_else_multi_output_shapes_inlined_test():
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT,
                                           [2, 3, 1])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])

    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3, 1])
    then_out2 = onnx.helper.make_tensor_value_info('then_out2',
                                                   onnx.TensorProto.FLOAT,
                                                   [2, 3, 1])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    else_out2 = onnx.helper.make_tensor_value_info('else_out2',
                                                   onnx.TensorProto.FLOAT,
                                                   [2, 3])

    xt = np.ones((2, 3, 1)).astype(np.float)
    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt = np.random.randn(2, 3).astype(np.float)
    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    then_add_node2 = onnx.helper.make_node('Add',
                                           inputs=['x', 'x'],
                                           outputs=['then_out2'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    else_sub_node = onnx.helper.make_node('Sub',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out2'])

    then_body = onnx.helper.make_graph([then_add_node, then_add_node2],
                                       'then_body', [], [then_out, then_out2])

    else_body = onnx.helper.make_graph([else_mul_node, else_sub_node],
                                       'else_body', [], [else_out, else_out2])

    cond = np.array([1]).astype(bool)
    cond_tensor = helper.make_tensor(name="cond",
                                     data_type=TensorProto.BOOL,
                                     dims=cond.shape,
                                     vals=cond.astype(bool))

    res1 = onnx.helper.make_tensor_value_info('res1', TensorProto.FLOAT, [])
    res2 = onnx.helper.make_tensor_value_info('res2', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['res1', 'res2'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [x, y], [res1, res2], [cond_tensor, xt_tensor, yt_tensor])


@onnx_test()
def if_then_else_multi_output_shapes_test():
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT,
                                           [2, 3, 1])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT,
                                           [2, 3, 1])

    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3, 1])
    then_out2 = onnx.helper.make_tensor_value_info('then_out2',
                                                   onnx.TensorProto.FLOAT,
                                                   [2, 3, 1])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3, 1])

    else_out2 = onnx.helper.make_tensor_value_info('else_out2',
                                                   onnx.TensorProto.FLOAT,
                                                   [2, 3, 1])

    xt = np.ones((2, 3, 1)).astype(np.float)
    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt = np.random.randn(2, 3, 1).astype(np.float)
    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    then_add_node2 = onnx.helper.make_node('Add',
                                           inputs=['x', 'x'],
                                           outputs=['then_out2'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    else_sub_node = onnx.helper.make_node('Sub',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out2'])

    then_body = onnx.helper.make_graph([then_add_node, then_add_node2],
                                       'then_body', [], [then_out, then_out2])

    else_body = onnx.helper.make_graph([else_mul_node, else_sub_node],
                                       'else_body', [], [else_out, else_out2])

    cond_tensor = onnx.helper.make_tensor_value_info("cond",
                                                     onnx.TensorProto.BOOL,
                                                     [1])

    res1 = onnx.helper.make_tensor_value_info('res1', TensorProto.FLOAT, [])
    res2 = onnx.helper.make_tensor_value_info('res2', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['res1', 'res2'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [x, y, cond_tensor], [res1, res2], [xt_tensor, yt_tensor])


@onnx_test()
def if_literal_test():
    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT, [5])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT, [5])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)
    z = np.array([]).astype(np.float32)

    then_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['then_out'],
        value=onnx.numpy_helper.from_array(x))

    else_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['else_out'],
        value=onnx.numpy_helper.from_array(y))

    empty_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['empty_out'],
        value=onnx.numpy_helper.from_array(z))

    then_body = onnx.helper.make_graph([then_const_node, empty_const_node],
                                       'then_body', [], [then_out])

    else_body = onnx.helper.make_graph([else_const_node, empty_const_node],
                                       'else_body', [], [else_out])

    cond_input = onnx.helper.make_tensor_value_info('cond',
                                                    onnx.TensorProto.BOOL, [])
    ret = onnx.helper.make_tensor_value_info('ret', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['ret'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [cond_input], [ret])


@onnx_test()
def if_param_excp_test():
    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 4])

    yt = np.random.randn(2, 4).astype(np.float)
    xt = np.random.randn(2, 3).astype(np.float)

    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    then_body = onnx.helper.make_graph([then_add_node], 'then_body', [],
                                       [then_out], [xt_tensor])

    else_body = onnx.helper.make_graph([else_mul_node], 'else_body', [],
                                       [else_out], [yt_tensor])

    cond_input = onnx.helper.make_tensor_value_info('cond',
                                                    onnx.TensorProto.BOOL, [])
    ret = onnx.helper.make_tensor_value_info('ret', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['ret'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [cond_input, x, y], [ret])


@onnx_test()
def if_param_excp1_test():
    then_out = onnx.helper.make_tensor_value_info('sub_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])

    xt = np.random.randn(2, 3).astype(np.float)

    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['sub_out'])

    sub_body = onnx.helper.make_graph([then_add_node], 'sub_body', [],
                                      [then_out], [xt_tensor])

    cond_input = onnx.helper.make_tensor_value_info('cond',
                                                    onnx.TensorProto.BOOL, [2])
    ret = onnx.helper.make_tensor_value_info('ret', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['ret'],
                                 then_branch=sub_body,
                                 else_branch=sub_body)

    return ([node], [cond_input, x], [ret])


@onnx_test()
def if_param_test():
    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])

    yt = np.random.randn(2, 3).astype(np.float)
    xt = np.random.randn(2, 3).astype(np.float)

    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    then_body = onnx.helper.make_graph([then_add_node], 'then_body', [],
                                       [then_out], [xt_tensor])

    else_body = onnx.helper.make_graph([else_mul_node], 'else_body', [],
                                       [else_out], [yt_tensor])

    cond_input = onnx.helper.make_tensor_value_info('cond',
                                                    onnx.TensorProto.BOOL, [])
    ret = onnx.helper.make_tensor_value_info('ret', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['ret'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [cond_input, x, y], [ret])


@onnx_test()
def if_pl_test():
    out_x = onnx.helper.make_tensor_value_info('out_x', onnx.TensorProto.FLOAT,
                                               [2, 3])
    out_l_x = onnx.helper.make_tensor_value_info('out_l_x',
                                                 onnx.TensorProto.FLOAT,
                                                 [2, 3])
    out_y = onnx.helper.make_tensor_value_info('out_y', onnx.TensorProto.FLOAT,
                                               [3, 3])
    out_l_y = onnx.helper.make_tensor_value_info('out_l_y',
                                                 onnx.TensorProto.FLOAT,
                                                 [3, 3])

    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [3, 3])

    xt = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    yt = np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]]).astype(np.float32)

    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['out_x'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['out_y'])

    then_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['out_l_y'],
        value=onnx.numpy_helper.from_array(yt))

    else_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['out_l_x'],
        value=onnx.numpy_helper.from_array(xt))

    then_body = onnx.helper.make_graph([then_add_node, then_const_node],
                                       'then_body', [], [out_x, out_l_y])

    else_body = onnx.helper.make_graph([else_mul_node, else_const_node],
                                       'else_body', [], [out_l_x, out_y])

    cond_input = onnx.helper.make_tensor_value_info('cond',
                                                    onnx.TensorProto.BOOL, [])
    ret = onnx.helper.make_tensor_value_info('ret', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['ret'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [cond_input, x, y], [ret], [xt_tensor, yt_tensor])


@onnx_test()
def if_then_test():
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])

    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    xt = np.ones((2, 3)).astype(np.float)
    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt = np.random.randn(2, 3).astype(np.float)
    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    then_body = onnx.helper.make_graph([then_add_node], 'then_body', [],
                                       [then_out])

    else_body = onnx.helper.make_graph([else_mul_node], 'else_body', [],
                                       [else_out])

    cond_tensor = onnx.helper.make_tensor_value_info("cond",
                                                     onnx.TensorProto.BOOL,
                                                     [1])

    res = onnx.helper.make_tensor_value_info('res', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['res'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [x, y, cond_tensor], [res], [xt_tensor, yt_tensor])


@onnx_test()
def if_then_test_inlined():
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])

    then_out = onnx.helper.make_tensor_value_info('then_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])
    else_out = onnx.helper.make_tensor_value_info('else_out',
                                                  onnx.TensorProto.FLOAT,
                                                  [2, 3])

    xt = np.ones((2, 3)).astype(np.float)
    xt_tensor = helper.make_tensor(name='xt',
                                   data_type=TensorProto.FLOAT,
                                   dims=xt.shape,
                                   vals=xt.flatten().astype(np.float32))

    yt = np.random.randn(2, 3).astype(np.float)
    yt_tensor = helper.make_tensor(name='yt',
                                   data_type=TensorProto.FLOAT,
                                   dims=yt.shape,
                                   vals=yt.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'xt'],
                                          outputs=['then_out'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'yt'],
                                          outputs=['else_out'])

    then_body = onnx.helper.make_graph([then_add_node], 'then_body', [],
                                       [then_out])

    else_body = onnx.helper.make_graph([else_mul_node], 'else_body', [],
                                       [else_out])

    cond = np.array([1]).astype(bool)
    cond_tensor = helper.make_tensor(name="cond",
                                     data_type=TensorProto.BOOL,
                                     dims=cond.shape,
                                     vals=cond.astype(bool))
    res = onnx.helper.make_tensor_value_info('res', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['res'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [x, y], [res], [cond_tensor, xt_tensor, yt_tensor])


@onnx_test()
def if_tuple_test():
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 4])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [3, 4])
    cond_input = onnx.helper.make_tensor_value_info('cond',
                                                    onnx.TensorProto.BOOL, [])

    then_out0 = onnx.helper.make_tensor_value_info('then_out0',
                                                   onnx.TensorProto.FLOAT,
                                                   [1, 4])
    then_out1 = onnx.helper.make_tensor_value_info('then_out1',
                                                   onnx.TensorProto.FLOAT,
                                                   [3, 4])
    else_out0 = onnx.helper.make_tensor_value_info('else_out0',
                                                   onnx.TensorProto.FLOAT,
                                                   [1, 4])
    else_out1 = onnx.helper.make_tensor_value_info('else_out1',
                                                   onnx.TensorProto.FLOAT,
                                                   [3, 4])

    one = np.ones([1]).astype(np.float)
    one_tensor = helper.make_tensor(name='one',
                                    data_type=TensorProto.FLOAT,
                                    dims=one.shape,
                                    vals=one.flatten().astype(np.float32))

    two = np.array([2]).astype(np.float)
    two_tensor = helper.make_tensor(name='two',
                                    data_type=TensorProto.FLOAT,
                                    dims=two.shape,
                                    vals=two.flatten().astype(np.float32))

    three = np.array([3]).astype(np.float)
    three_tensor = helper.make_tensor(name='three',
                                      data_type=TensorProto.FLOAT,
                                      dims=three.shape,
                                      vals=three.flatten().astype(np.float32))

    then_add_node = onnx.helper.make_node('Add',
                                          inputs=['x', 'one'],
                                          outputs=['then_out0'])
    then_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['y', 'two'],
                                          outputs=['then_out1'])

    else_mul_node = onnx.helper.make_node('Mul',
                                          inputs=['x', 'three'],
                                          outputs=['else_out0'])
    else_add_node = onnx.helper.make_node('Add',
                                          inputs=['y', 'three'],
                                          outputs=['else_out1'])

    then_body = onnx.helper.make_graph([then_add_node, then_mul_node],
                                       'then_body', [], [then_out0, then_out1])

    else_body = onnx.helper.make_graph([else_mul_node, else_add_node],
                                       'else_body', [], [else_out0, else_out1])

    res0 = onnx.helper.make_tensor_value_info('res0', TensorProto.FLOAT, [])
    res1 = onnx.helper.make_tensor_value_info('res1', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('If',
                                 inputs=['cond'],
                                 outputs=['res0', 'res1'],
                                 then_branch=then_body,
                                 else_branch=else_body)

    return ([node], [cond_input, x,
                     y], [res0, res1], [one_tensor, two_tensor, three_tensor])


@onnx_test()
def imagescaler_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 16, 16])

    node = onnx.helper.make_node('ImageScaler',
                                 inputs=['0'],
                                 outputs=['1'],
                                 bias=[0.01, 0.02, 0.03],
                                 scale=0.5)

    return ([node], [x], [y])


@onnx_test()
def imagescaler_half_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [1, 3, 16, 16])

    node = onnx.helper.make_node('ImageScaler',
                                 inputs=['0'],
                                 outputs=['1'],
                                 bias=[0.01, 0.02, 0.03],
                                 scale=0.5)

    return ([node], [x], [y])


@onnx_test()
def imagescaler_bf16_test():
    x = helper.make_tensor_value_info('0', TensorProto.BFLOAT16,
                                      [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.BFLOAT16,
                                      [1, 3, 16, 16])

    node = onnx.helper.make_node('ImageScaler',
                                 inputs=['0'],
                                 outputs=['1'],
                                 bias=[0.01, 0.02, 0.03],
                                 scale=0.5)

    return ([node], [x], [y])


@onnx_test()
def implicit_add_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4, 1])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test()
def implicit_pow_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4, 1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def implicit_sub_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.UINT64, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.UINT64, [4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.UINT64,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def initializer_not_an_input():
    values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    w = helper.make_tensor(name='w',
                           data_type=TensorProto.FLOAT,
                           dims=values.shape,
                           vals=values.flatten().astype(np.float))

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 4])

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['x', 'w'],
        outputs=['y'],
    )

    return ([node], [x], [y], [w])


@onnx_test()
def instance_norm_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_half_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [1, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [2])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [1, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_bf16_test():
    x = helper.make_tensor_value_info('0', TensorProto.BFLOAT16, [1, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.BFLOAT16, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [2])
    y = helper.make_tensor_value_info('3', TensorProto.BFLOAT16, [1, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_type_mismatch_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [2])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_dyn_batch_test():
    # the batch size is a dynamic dimension
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [None, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_dyn_batch_half_test():
    # the batch size is a dynamic dimension
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT16,
                                      [None, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [2])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT16,
                                      [None, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_dyn_batch_bf16_test():
    # the batch size is a dynamic dimension
    x = helper.make_tensor_value_info('0', TensorProto.BFLOAT16,
                                      [None, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.BFLOAT16, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [2])
    y = helper.make_tensor_value_info('3', TensorProto.BFLOAT16,
                                      [None, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_invalid_type_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT32, [1, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2])
    bias = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_nonbroadcastable_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 3, 3])
    scale = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4])
    bias = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 2, 3, 3])

    node = onnx.helper.make_node('InstanceNormalization',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'])

    return ([node], [x, scale, bias], [y])


@onnx_test()
def instance_norm_val_test():
    x = np.array([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                   [[0, 1, 2], [3, 4, 5], [6, 7, 8]]]])
    scale = np.array([1, 2])
    bias = np.array([0, 1])

    x_tensor = helper.make_tensor(name='x_tensor',
                                  data_type=TensorProto.FLOAT,
                                  dims=x.shape,
                                  vals=x.flatten().astype(np.float))
    scale_tensor = helper.make_tensor(name='scale_tensor',
                                      data_type=TensorProto.FLOAT,
                                      dims=scale.shape,
                                      vals=scale.flatten().astype(np.float))
    bias_tensor = helper.make_tensor(name='bias_tensor',
                                     data_type=TensorProto.FLOAT,
                                     dims=bias.shape,
                                     vals=bias.flatten().astype(np.float))

    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 3, 3])

    node = onnx.helper.make_node(
        'InstanceNormalization',
        inputs=['x_tensor', 'scale_tensor', 'bias_tensor'],
        outputs=['y'])

    return ([node], [], [y], [x_tensor, scale_tensor, bias_tensor])


@onnx_test()
def instance_norm_val_3d_test():
    x = np.array([[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                   [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]]])
    scale = np.array([1, 2])
    bias = np.array([0, 1])

    x_tensor = helper.make_tensor(name='x_tensor',
                                  data_type=TensorProto.FLOAT,
                                  dims=x.shape,
                                  vals=x.flatten().astype(np.float))
    scale_tensor = helper.make_tensor(name='scale_tensor',
                                      data_type=TensorProto.FLOAT,
                                      dims=scale.shape,
                                      vals=scale.flatten().astype(np.float))
    bias_tensor = helper.make_tensor(name='bias_tensor',
                                     data_type=TensorProto.FLOAT,
                                     dims=bias.shape,
                                     vals=bias.flatten().astype(np.float))

    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 2, 2, 2])

    node = onnx.helper.make_node(
        'InstanceNormalization',
        inputs=['x_tensor', 'scale_tensor', 'bias_tensor'],
        outputs=['y'])

    return ([node], [], [y], [x_tensor, scale_tensor, bias_tensor])


@onnx_test()
def int4_const_identity_qdq_test():
    # Graph for int4, with an identity opr + QDQ
    zp_values = np.array([0, 0, 0, 0])
    x_t = helper.make_tensor(name='i_x',
                             data_type=TensorProto.INT4,
                             dims=zp_values.shape,
                             vals=zp_values.flatten().astype(np.int32))

    i_node = onnx.helper.make_node(
        'Identity',
        inputs=['i_x'],
        outputs=['i_y_zp'],
    )

    data_values = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    data_t = helper.make_tensor(name='data',
                                data_type=TensorProto.FLOAT16,
                                dims=data_values.shape,
                                vals=data_values.flatten().astype(np.float16))

    sc_values = np.array([1.0, 0.5, 1.0, 0.25])
    sc_t = helper.make_tensor(name='sc_q',
                              data_type=TensorProto.FLOAT16,
                              dims=sc_values.shape,
                              vals=sc_values.flatten().astype(np.float16))

    q_node = onnx.helper.make_node(
        'QuantizeLinear',
        inputs=['data', 'sc_q', 'i_y_zp'],
        outputs=['q_y'],
    )

    #dequantizer uses same scale values as the quantizer:
    dq_node = onnx.helper.make_node(
        'DequantizeLinear',
        inputs=['q_y', 'sc_q', 'i_y_zp'],
        outputs=['dq_y'],
    )

    t_node = helper.make_node(
        'Transpose',
        inputs=['dq_y'],
        outputs=['t_y'],
        perm=[1, 0],
    )

    x2_t = helper.make_tensor_value_info('x2', TensorProto.FLOAT16, [4, 4])

    dot_node = helper.make_node(
        'MatMul',
        inputs=['t_y', 'x2'],
        outputs=['y'],
    )

    y_t = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [4, 4])

    return ([i_node, q_node, dq_node, t_node,
             dot_node], [x2_t], [y_t], [x_t, data_t, sc_t])


@onnx_test()
def int4_const_identity_block_sz_1_qdq_test():
    # Graph for int4, with an identity opr + QDQ. Quantization Block size = 1
    zp_values = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    x_t = helper.make_tensor(name='i_x',
                             data_type=TensorProto.INT4,
                             dims=zp_values.shape,
                             vals=zp_values.flatten().astype(np.int32))

    i_node = onnx.helper.make_node(
        'Identity',
        inputs=['i_x'],
        outputs=['i_y_zp'],
    )

    data_values = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    data_t = helper.make_tensor(name='data',
                                data_type=TensorProto.FLOAT16,
                                dims=data_values.shape,
                                vals=data_values.flatten().astype(np.float16))

    sc_values = np.array([[0.5, 0.25, 0.5, 0.125], [0.25, 0.5, 0.5, 0.25]])
    sc_t = helper.make_tensor(name='sc_q',
                              data_type=TensorProto.FLOAT16,
                              dims=sc_values.shape,
                              vals=sc_values.flatten().astype(np.float16))

    q_node = onnx.helper.make_node(
        'QuantizeLinear',
        inputs=['data', 'sc_q', 'i_y_zp'],
        outputs=['q_y'],
    )

    # dequantizer uses same scale values as the quantizer:
    dq_node = onnx.helper.make_node(
        'DequantizeLinear',
        inputs=['q_y', 'sc_q', 'i_y_zp'],
        outputs=['dq_y'],
    )

    t_node = helper.make_node(
        'Transpose',
        inputs=['dq_y'],
        outputs=['t_y'],
        perm=[1, 0],
    )

    x2_t = helper.make_tensor_value_info('x2', TensorProto.FLOAT16, [2, 4])

    dot_node = helper.make_node(
        'MatMul',
        inputs=['t_y', 'x2'],
        outputs=['y'],
    )

    y_t = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [4, 4])

    return ([i_node, q_node, dq_node, t_node,
             dot_node], [x2_t], [y_t], [x_t, data_t, sc_t])


@onnx_test()
def int4_const_identity_block_sz_2_qdq_test():
    # Graph for int4, with an identity opr + QDQ. Quantization Block size = 2
    zp_values = np.array([[0, 0], [0, 0]])
    x_t = helper.make_tensor(name='i_x',
                             data_type=TensorProto.INT4,
                             dims=zp_values.shape,
                             vals=zp_values.flatten().astype(np.int32))

    i_node = onnx.helper.make_node(
        'Identity',
        inputs=['i_x'],
        outputs=['i_y_zp'],
    )

    data_values = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    data_t = helper.make_tensor(name='data',
                                data_type=TensorProto.FLOAT16,
                                dims=data_values.shape,
                                vals=data_values.flatten().astype(np.float16))

    sc_values = np.array([[0.5, 0.125], [0.5, 0.25]])
    sc_t = helper.make_tensor(name='sc_q',
                              data_type=TensorProto.FLOAT16,
                              dims=sc_values.shape,
                              vals=sc_values.flatten().astype(np.float16))

    q_node = onnx.helper.make_node(
        'QuantizeLinear',
        inputs=['data', 'sc_q', 'i_y_zp'],
        outputs=['q_y'],
    )

    # dequantizer uses same scale values as the quantizer:
    dq_node = onnx.helper.make_node(
        'DequantizeLinear',
        inputs=['q_y', 'sc_q', 'i_y_zp'],
        outputs=['dq_y'],
    )

    t_node = helper.make_node(
        'Transpose',
        inputs=['dq_y'],
        outputs=['t_y'],
        perm=[1, 0],
    )

    x2_t = helper.make_tensor_value_info('x2', TensorProto.FLOAT16, [2, 4])

    dot_node = helper.make_node(
        'MatMul',
        inputs=['t_y', 'x2'],
        outputs=['y'],
    )

    y_t = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [4, 4])

    return ([i_node, q_node, dq_node, t_node,
             dot_node], [x2_t], [y_t], [x_t, data_t, sc_t])


@onnx_test()
def isinf_half_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.FLOAT16, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.BOOL, [2, 3])

    node = onnx.helper.make_node(
        'IsInf',
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def isinf_bf16_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.BFLOAT16, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.BOOL, [2, 3])

    node = onnx.helper.make_node(
        'IsInf',
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def isinf_neg_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.FLOAT, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.BOOL, [2, 3])

    node = onnx.helper.make_node(
        'IsInf',
        detect_negative=[1],
        detect_positive=[0],
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def isinf_double_pos_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.DOUBLE, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.BOOL, [2, 3])

    node = onnx.helper.make_node(
        'IsInf',
        detect_negative=[0],
        detect_positive=[1],
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def isinf_no_detect_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.FLOAT, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.BOOL, [2, 3])

    node = onnx.helper.make_node(
        'IsInf',
        detect_negative=[0],
        detect_positive=[0],
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def isnan_float_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.FLOAT, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'IsNaN',
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def isnan_half_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.FLOAT16, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.FLOAT16, [2, 3])

    node = onnx.helper.make_node(
        'IsNaN',
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def isnan_bf16_test():
    t1 = helper.make_tensor_value_info('t1', TensorProto.BFLOAT16, [2, 3])
    t2 = helper.make_tensor_value_info('t2', TensorProto.BFLOAT16, [2, 3])

    node = onnx.helper.make_node(
        'IsNaN',
        inputs=['t1'],
        outputs=['t2'],
    )
    return ([node], [t1], [t2])


@onnx_test()
def layernorm_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 1, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 1, 5])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [5])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [5])
    axes = [2]
    pow_2 = np.array([[[2, 2, 2, 2, 2]]])
    epsilon = np.array([1e-12])

    pow_tensor = helper.make_tensor(name='pow',
                                    data_type=TensorProto.FLOAT,
                                    dims=pow_2.shape,
                                    vals=pow_2.flatten().astype(np.float))

    epsilon_tensor = helper.make_tensor(name='epsilon',
                                        data_type=TensorProto.FLOAT,
                                        dims=epsilon.shape,
                                        vals=epsilon.flatten().astype(
                                            np.float))

    mean = onnx.helper.make_node('ReduceMean',
                                 inputs=['0'],
                                 outputs=['mean_out'],
                                 axes=axes)

    sub_mean = onnx.helper.make_node('Sub',
                                     inputs=['0', 'mean_out'],
                                     outputs=['sub_out'])

    sub_pow = onnx.helper.make_node('Pow',
                                    inputs=['sub_out', 'pow'],
                                    outputs=['pow_out'])

    var = onnx.helper.make_node('ReduceMean',
                                inputs=['pow_out'],
                                outputs=['var_out'],
                                axes=axes)

    add = onnx.helper.make_node('Add',
                                inputs=['var_out', 'epsilon'],
                                outputs=['add_out'])

    sqrt = onnx.helper.make_node('Sqrt',
                                 inputs=['add_out'],
                                 outputs=['sqrt_out'])

    div = onnx.helper.make_node('Div',
                                inputs=['sub_out', 'sqrt_out'],
                                outputs=['div_out'])

    mul = onnx.helper.make_node('Mul',
                                inputs=['scale', 'div_out'],
                                outputs=['mul_out'])

    bias_add = onnx.helper.make_node('Add',
                                     inputs=['mul_out', 'bias'],
                                     outputs=['1'])

    return ([mean, sub_mean, sub_pow, var, add, sqrt, div, mul,
             bias_add], [x, scale, bias], [y], [pow_tensor, epsilon_tensor])


def make_layer_norm(shape, axis, dtype=TensorProto.FLOAT):
    norm_axis = axis + len(shape) if axis < 0 else axis
    x = helper.make_tensor_value_info('x', dtype, shape)
    scale = helper.make_tensor_value_info('scale', dtype, shape[norm_axis:])
    bias = helper.make_tensor_value_info('bias', dtype, shape[norm_axis:])
    y = helper.make_tensor_value_info('y', dtype, shape)

    node = onnx.helper.make_node('LayerNormalization',
                                 inputs=['x', 'scale', 'bias'],
                                 outputs=['y'],
                                 axis=axis)

    return ([node], [x, scale, bias], [y])


@onnx_test()
def layer_norm_invalid_shape_error_test():
    return make_layer_norm([3], 0)


@onnx_test()
def layer_norm_2d_axis_zero_test():
    return make_layer_norm([3, 4], 0)


@onnx_test()
def layer_norm_2d_axis_one_test():
    return make_layer_norm([3, 4], 1)


@onnx_test()
def layer_norm_2d_axis_minus_one_test():
    return make_layer_norm([3, 4], -1)


@onnx_test()
def layer_norm_3d_test():
    return make_layer_norm([1, 4, 2], -1)


@onnx_test()
def layer_norm_3d_half_test():
    return make_layer_norm([1, 4, 2], -1, TensorProto.FLOAT16)


@onnx_test()
def layer_norm_3d_bf16_test():
    return make_layer_norm([1, 4, 2], -1, TensorProto.BFLOAT16)


@onnx_test()
def layer_norm_4d_test():
    return make_layer_norm([3, 3, 3, 3], -1)


@onnx_test()
def layer_norm_4d_half_test():
    return make_layer_norm([3, 3, 3, 3], -1, TensorProto.FLOAT16)


@onnx_test()
def layer_norm_4d_bf16_test():
    return make_layer_norm([3, 3, 3, 3], -1, TensorProto.BFLOAT16)


@onnx_test()
def layer_norm_invalid_axis_error_test():
    return make_layer_norm([1, 4, 2], 1000)


@onnx_test()
def layer_norm_invalid_minus_axis_error_test():
    return make_layer_norm([1, 4, 2], -1000)


@onnx_test()
def layer_norm_invalid_input_count_error_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('LayerNormalization',
                                 inputs=['x'],
                                 outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def layer_norm_without_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('LayerNormalization',
                                 inputs=['x', 'scale'],
                                 outputs=['y'])

    return ([node], [x, scale], [y])


@onnx_test()
def layer_norm_small_eps_half_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT16, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 2])

    node = onnx.helper.make_node('LayerNormalization',
                                 inputs=['x', 'scale'],
                                 outputs=['y'],
                                 epsilon=1e-12)

    return ([node], [x, scale], [y])


@onnx_test()
def layer_norm_small_eps_bf16_test():
    x = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [1, 2])
    scale = helper.make_tensor_value_info('scale', TensorProto.BFLOAT16, [2])
    y = helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [1, 2])

    node = onnx.helper.make_node('LayerNormalization',
                                 inputs=['x', 'scale'],
                                 outputs=['y'],
                                 epsilon=1e-7)

    return ([node], [x, scale], [y])


@onnx_test()
def leaky_relu_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('LeakyRelu',
                                 inputs=['0'],
                                 outputs=['1'],
                                 alpha=0.01)

    return ([node], [x], [y])


@onnx_test()
def less_test():
    ax1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x1 = helper.make_tensor("x1",
                            data_type=TensorProto.FLOAT,
                            dims=(2, 3),
                            vals=ax1.astype(np.float32))

    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'Less',
        inputs=['x1', 'x2'],
        outputs=['y'],
    )

    return ([node], [x2], [y], [x1])


@onnx_test()
def less_bool_test():

    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.BOOL, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node1 = onnx.helper.make_node('Cast', inputs=['x1'], outputs=['bx1'], to=9)

    node2 = onnx.helper.make_node(
        'Less',
        inputs=['bx1', 'x2'],
        outputs=['y'],
    )

    return ([node1, node2], [x1, x2], [y])


@onnx_test()
def lessorequal_test():

    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'LessOrEqual',
        inputs=['x1', 'x2'],
        outputs=['y'],
    )

    return ([node], [x1, x2], [y])


@onnx_test()
def log_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Log',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def logical_and_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.BOOL, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.BOOL, [4, 5])
    z = helper.make_tensor_value_info('2', TensorProto.BOOL, [2, 3, 4, 5])

    node = onnx.helper.make_node('And', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test()
def logical_or_test():
    x = helper.make_tensor_value_info('0', TensorProto.BOOL, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.BOOL, [2, 3, 4, 5])
    z = helper.make_tensor_value_info('2', TensorProto.BOOL, [2, 3, 4, 5])

    node = onnx.helper.make_node('Or', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test()
def logical_xor_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.BOOL, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.BOOL, [4, 1])
    z = helper.make_tensor_value_info('2', TensorProto.BOOL, [2, 3, 4, 5])

    node = onnx.helper.make_node('Xor', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test()
def logsoftmax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5, 6])

    node = onnx.helper.make_node('LogSoftmax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=1)

    return ([node], [x], [y])


@onnx_test()
def logsoftmax_nonstd_input_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [6, 9])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 4])

    node0 = onnx.helper.make_node('Slice',
                                  inputs=['0'],
                                  axes=[0, 1],
                                  starts=[1, 0],
                                  ends=[4, 4],
                                  outputs=['1'])

    node1 = onnx.helper.make_node('LogSoftmax',
                                  inputs=['1'],
                                  outputs=['2'],
                                  axis=-1)

    return ([node0, node1], [x], [y])


@onnx_test()
def loop_default_test():
    body = helper.make_graph([
        helper.make_node("Add", ["a", "b_in"], ["my_local"]),
        helper.make_node("Sub", ["a", "b_in"], ["a_sub_b_in"]),
        helper.make_node("Greater", ["my_local", "a_sub_b_in"],
                         ["keep_going"]),
        helper.make_node("Add", ["a_sub_b_in", "a_sub_b_in"],
                         ["user_defined_vals"]),
    ], "body", [
        helper.make_tensor_value_info('iteration_num', TensorProto.INT64, []),
        helper.make_tensor_value_info('keep_going_inp', TensorProto.BOOL, []),
        helper.make_tensor_value_info('b_in', TensorProto.FLOAT, [])
    ], [
        helper.make_tensor_value_info('keep_going', TensorProto.BOOL, []),
        helper.make_tensor_value_info('a_sub_b_in', TensorProto.FLOAT, []),
        helper.make_tensor_value_info('my_local', TensorProto.FLOAT, []),
        helper.make_tensor_value_info('user_defined_vals', TensorProto.FLOAT,
                                      []),
    ])

    node = helper.make_node(
        "Loop",
        inputs=["", "", "b"],
        outputs=["b_loop", "my_local_loop", "user_defined_vals_loop"],
        body=body)

    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [])

    b_loop = helper.make_tensor_value_info('b_loop', TensorProto.FLOAT, [])
    uout = helper.make_tensor_value_info('user_defined_vals_loop',
                                         TensorProto.FLOAT, [2, 1])

    return ([node], [a, b], [b_loop, uout])


@onnx_test()
def loop_test():
    body = helper.make_graph([
        helper.make_node("Add", ["a", "b_in"], ["my_local"]),
        helper.make_node("Sub", ["a", "b_in"], ["a_sub_b_in"]),
        helper.make_node("Greater", ["my_local", "a_sub_b_in"],
                         ["keep_going"]),
        helper.make_node("Add", ["a_sub_b_in", "a_sub_b_in"],
                         ["user_defined_vals"]),
    ], "body", [
        helper.make_tensor_value_info('iteration_num', TensorProto.INT64, [1]),
        helper.make_tensor_value_info('keep_going_inp', TensorProto.BOOL, [1]),
        helper.make_tensor_value_info('b_in', TensorProto.FLOAT, [1])
    ], [
        helper.make_tensor_value_info('keep_going', TensorProto.BOOL, [1]),
        helper.make_tensor_value_info('a_sub_b_in', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('my_local', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('user_defined_vals', TensorProto.FLOAT,
                                      [1]),
    ])

    node = helper.make_node(
        "Loop",
        inputs=["max_trip_count", "keep_going_cond", "b"],
        outputs=["b_loop", "my_local_loop", "user_defined_vals_loop"],
        body=body)

    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [1])
    cond = helper.make_tensor_value_info('keep_going_cond', TensorProto.BOOL,
                                         [1])
    iter = helper.make_tensor_value_info('max_trip_count', TensorProto.INT64,
                                         [1])

    b_loop = helper.make_tensor_value_info('b_loop', TensorProto.FLOAT, [1])
    uout = helper.make_tensor_value_info('user_defined_vals_loop',
                                         TensorProto.FLOAT, [2, 1])

    return ([node], [iter, cond, a, b], [b_loop, uout])


@onnx_test()
def loop_test_implicit_tripcnt():
    body = helper.make_graph([
        helper.make_node("Add", ["a", "b_in"], ["my_local"]),
        helper.make_node("Sub", ["a", "b_in"], ["a_sub_b_in"]),
        helper.make_node("Greater", ["my_local", "a_sub_b_in"],
                         ["keep_going"]),
        helper.make_node("Add", ["a_sub_b_in", "a_sub_b_in"],
                         ["user_defined_vals"]),
    ], "body", [
        helper.make_tensor_value_info('iteration_num', TensorProto.INT64, [1]),
        helper.make_tensor_value_info('keep_going_inp', TensorProto.BOOL, [1]),
        helper.make_tensor_value_info('b_in', TensorProto.FLOAT, [1])
    ], [
        helper.make_tensor_value_info('keep_going', TensorProto.BOOL, [1]),
        helper.make_tensor_value_info('a_sub_b_in', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('my_local', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('user_defined_vals', TensorProto.FLOAT,
                                      [1]),
    ])

    iter = helper.make_tensor(name='max_trip_count',
                              data_type=TensorProto.INT64,
                              dims=[1],
                              vals=[15])

    node = helper.make_node(
        "Loop",
        inputs=["max_trip_count", "keep_going_cond", "b"],
        outputs=["b_loop", "my_local_loop", "user_defined_vals_loop"],
        body=body)

    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [1])
    cond = helper.make_tensor_value_info('keep_going_cond', TensorProto.BOOL,
                                         [1])

    b_loop = helper.make_tensor_value_info('b_loop', TensorProto.FLOAT, [1])
    uout = helper.make_tensor_value_info('user_defined_vals_loop',
                                         TensorProto.FLOAT, [2, 1])

    return ([node], [cond, a, b], [b_loop, uout], [iter])


@onnx_test()
def lpnormalization_axis_error_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('LpNormalization',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=2)
    return ([node], [x], [y])


@onnx_test()
def lpnormalization_default_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])

    node = onnx.helper.make_node(
        'LpNormalization',
        inputs=['x'],
        outputs=['y'],
        axis=0,
    )
    return ([node], [x], [y])


@onnx_test()
def lpnormalization_l1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])

    node = onnx.helper.make_node(
        'LpNormalization',
        inputs=['x'],
        outputs=['y'],
        p=1,
    )
    return ([node], [x], [y])


@onnx_test()
def lpnormalization_l2_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])

    node = onnx.helper.make_node('LpNormalization',
                                 inputs=['x'],
                                 outputs=['y'],
                                 p=2)
    return ([node], [x], [y])


@onnx_test()
def lpnormalization_p_error_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node('LpNormalization',
                                 inputs=['x'],
                                 outputs=['y'],
                                 p=3)
    return ([node], [x], [y])


@onnx_test()
def lppool_l1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 3])

    node = onnx.helper.make_node('LpPool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[3],
                                 p=1)
    return ([node], [x], [y])


@onnx_test()
def lppool_l2_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 3])

    node = onnx.helper.make_node('LpPool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[3],
                                 p=2)
    return ([node], [x], [y])


@onnx_test()
def lrn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 28, 24, 24])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 28, 24, 24])

    node = onnx.helper.make_node('LRN',
                                 inputs=['0'],
                                 size=5,
                                 alpha=0.0001,
                                 beta=0.75,
                                 bias=1.0,
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def lstm_bi_layout_cell_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [2, 80, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [2, 80, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, 160])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 2, 20])
    c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [3, 2, 20])
    pph = helper.make_tensor_value_info('pph', TensorProto.FLOAT, [2, 60])

    cellout = helper.make_tensor_value_info('cellout', TensorProto.FLOAT,
                                            [3, 2, 20])

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0', 'c0', 'pph'],
        outputs=['', '', 'cellout'],
        activations=['sigmoid', 'tanh', 'tanh', 'sigmoid', 'tanh', 'tanh'],
        clip=0,
        direction='bidirectional',
        hidden_size=20,
        input_forget=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0, c0, pph], [cellout])


@onnx_test()
def lstm_bi_layout_last_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [2, 80, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [2, 80, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, 160])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 2, 20])
    c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [3, 2, 20])
    pph = helper.make_tensor_value_info('pph', TensorProto.FLOAT, [2, 60])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 2, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 2, 20])

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0', 'c0', 'pph'],
        outputs=['hs', 'output'],
        activations=['sigmoid', 'tanh', 'tanh', 'sigmoid', 'tanh', 'tanh'],
        clip=0,
        direction='bidirectional',
        hidden_size=20,
        input_forget=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0, c0, pph], [hs, output])


@onnx_test()
def lstm_f_layout_hs_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 80, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 80, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 160])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])
    c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [3, 1, 20])
    pph = helper.make_tensor_value_info('pph', TensorProto.FLOAT, [1, 60])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0', 'c0', 'pph'],
        outputs=['hs', 'output'],
        activations=['sigmoid', 'tanh', 'tanh'],
        clip=0,
        direction='forward',
        hidden_size=20,
        input_forget=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0, c0, pph], [hs, output])


@onnx_test()
def lstm_f_layout_cell_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 80, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 80, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 160])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])
    c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [3, 1, 20])
    pph = helper.make_tensor_value_info('pph', TensorProto.FLOAT, [1, 60])

    cellout = helper.make_tensor_value_info('cellout', TensorProto.FLOAT,
                                            [3, 1, 20])

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0', 'c0', 'pph'],
        outputs=['', '', 'cellout'],
        activations=['sigmoid', 'tanh', 'tanh'],
        clip=0,
        direction='forward',
        hidden_size=20,
        input_forget=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0, c0, pph], [cellout])


@onnx_test()
def lstm_f_1af_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [5, 3, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 80, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 80, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [5, 1, 3, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [1, 3, 20])

    node = onnx.helper.make_node('LSTM',
                                 inputs=['seq', 'w', 'r'],
                                 outputs=['hs', 'output'],
                                 activations=['sigmoid'],
                                 clip=0,
                                 direction='forward',
                                 hidden_size=20,
                                 input_forget=1)

    return ([node], [seq, w, r], [hs, output])


@onnx_test()
def lstm_r_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 80, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 80, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 160])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])
    c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [3, 1, 20])
    pph = helper.make_tensor_value_info('pph', TensorProto.FLOAT, [1, 60])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0', 'c0', 'pph'],
        outputs=['hs'],
        activations=['sigmoid', 'tanh', 'tanh'],
        clip=0,
        direction='reverse',
        hidden_size=20,
        input_forget=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0, c0, pph], [hs])


@onnx_test()
def lstm_r_layout_hs_cell_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 80, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 80, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 160])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])
    c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [3, 1, 20])
    pph = helper.make_tensor_value_info('pph', TensorProto.FLOAT, [1, 60])

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])
    cellout = helper.make_tensor_value_info('cellout', TensorProto.FLOAT,
                                            [3, 1, 20])

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0', 'c0', 'pph'],
        outputs=['', 'output', 'cellout'],
        activations=['sigmoid', 'tanh', 'tanh'],
        clip=0,
        direction='reverse',
        hidden_size=20,
        input_forget=1,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0, c0, pph], [output, cellout])


@onnx_test()
def matmul_bmbm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5, 2, 1, 7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 2, 3, 6, 8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_bmv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 6])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_mv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [6])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_vbm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5, 7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_vm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_vv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_dyn_mm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7, None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, None])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_dyn_mv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 1])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_dyn_vm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7, None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, None])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_dyn_vv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmul_dyn_broadcast_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5, 7, None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 1, None])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmulinteger_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [3, 6, 16])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 16, 8])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [3, 6, 8])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmulinteger_dyn_error():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [None, 6, 16])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [None, 16, 8])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [None, 6, 8])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmulinteger_invalid_type_error():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [None, 6, 16])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT16, [None, 16, 8])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [None, 6, 8])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmulinteger_uns_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.UINT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmulinteger_int8_uint8_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test()
def matmulinteger_uns_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.UINT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    zp1 = helper.make_tensor('3', TensorProto.UINT8, [], [0])
    zp2 = helper.make_tensor('4', TensorProto.UINT8, [], [1])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2', '3', '4'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y], [zp1, zp2])


@onnx_test()
def matmulinteger_int8_uint8_one_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    zp1 = helper.make_tensor('3', TensorProto.INT8, [], [5])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2', '3'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y], [zp1])


@onnx_test()
def matmulinteger_int8_uint8_one_zp_zero_vec_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    zp1 = helper.make_tensor('3', TensorProto.INT8, [4, 3],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2', '3'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y], [zp1])


@onnx_test()
def matmulinteger_int8_uint8_one_zp_zero_vec_test2():
    m1 = helper.make_tensor_value_info('1', TensorProto.UINT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 2])
    zp1 = helper.make_tensor(
        '3', TensorProto.UINT8, [4, 3],
        [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2', '3'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y], [zp1])


@onnx_test()
def matmulinteger_int8_uint8_one_zp_error_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.UINT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    zp1 = helper.make_tensor('3', TensorProto.INT8, [], [5])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2', '3'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y], [zp1])


@onnx_test()
def matmulinteger_int8_uint8_dual_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    zp1 = helper.make_tensor('3', TensorProto.INT8, [], [1])
    zp2 = helper.make_tensor('4', TensorProto.UINT8, [], [1])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2', '3', '4'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y], [zp1, zp2])


@onnx_test()
def matmulinteger_int8_uint8_dual_zero_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    zp1 = helper.make_tensor('3', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('4', TensorProto.UINT8, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulInteger',
        inputs=['1', '2', '3', '4'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y], [zp1, zp2])


@onnx_test()
def matmulintegertofloat_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [])


@onnx_test()
def matmulintegertofloat_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [2])
    zp1 = helper.make_tensor_value_info('5', TensorProto.INT8, [3])
    zp2 = helper.make_tensor_value_info('6', TensorProto.INT8, [2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2, zp1, zp2], [y], [])


@onnx_test()
def matmulintegertofloat_scalar_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [2])
    zp1 = helper.make_tensor_value_info('5', TensorProto.INT8, [3])
    zp2 = helper.make_tensor('6', TensorProto.INT8, [], [129])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2, zp1], [y], [zp2])


@onnx_test()
def matmulintegertofloat_scalar_scale_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])
    s2 = helper.make_tensor('4', TensorProto.FLOAT, [], [10])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1], [y], [s2])


@onnx_test()
def matmulintegertofloat_zp_bias_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [2])
    zp1 = helper.make_tensor_value_info('5', TensorProto.INT8, [3])
    zp2 = helper.make_tensor_value_info('6', TensorProto.UINT8, [2])
    b1 = helper.make_tensor_value_info('7', TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6', '7'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2, zp1, zp2, b1], [y], [])


@onnx_test()
def matmulintegertofloat_bad_scale_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.INT8, [4, 3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT16, [3, 2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2])


@onnx_test()
def matmulintegertofloat_bad_scale2_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.INT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4, 3])
    s2 = helper.make_tensor_value_info('4', TensorProto.INT8, [3, 2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2])


@onnx_test()
def matmulintegertofloat_bad_scale3_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4, 3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [3, 2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2])


@onnx_test()
def matmulintegertofloat_bad_scale4_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.INT8, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.UINT8, [2, 2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2])


@onnx_test()
def matmulintegertofloat_bad_scale5_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.INT8, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.UINT8, [7])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2])


@onnx_test()
def matmulintegertofloat_bad_bias_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4, 3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [3, 2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    b1 = helper.make_tensor('7', TensorProto.UINT8, [2], [128, 128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6', '7'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2, b1])


@onnx_test()
def matmulintegertofloat_bad_bias_test2():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4, 3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [3, 2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    b1 = helper.make_tensor('7', TensorProto.FLOAT16, [2], [128, -128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6', '7'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2, b1])


@onnx_test()
def matmulintegertofloat_bad_bias_test3():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4, 3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [3, 2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    b1 = helper.make_tensor('7', TensorProto.FLOAT16, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6', '7'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2, b1])


@onnx_test()
def matmulintegertofloat_half_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT16, [2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [])


@onnx_test()
def matmulintegertofloat_half_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT16, [2])
    zp1 = helper.make_tensor_value_info('5', TensorProto.INT8, [3])
    zp2 = helper.make_tensor_value_info('6', TensorProto.UINT8, [2])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2, zp1, zp2], [y], [])


@onnx_test()
def matmulintegertofloat_half_scalar_zp_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT16, [2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2])


@onnx_test()
def matmulintegertofloat_half_zp_bias_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [3, 2])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [3])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT16, [2])
    zp1 = helper.make_tensor('5', TensorProto.INT8, [], [0])
    zp2 = helper.make_tensor('6', TensorProto.UINT8, [], [128])
    b1 = helper.make_tensor('7', TensorProto.FLOAT16, [2], [128, -128])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [4, 2])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6', '7'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2], [y], [zp1, zp2, b1])


@onnx_test()
def matmulintegertofloat_zp_bias_3d_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.INT8, [4, 3, 2])
    m2 = helper.make_tensor_value_info('2', TensorProto.UINT8, [4, 2, 3])
    s1 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2])
    s2 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [3])
    zp1 = helper.make_tensor_value_info('5', TensorProto.INT8, [2])
    zp2 = helper.make_tensor_value_info('6', TensorProto.UINT8, [3])
    b1 = helper.make_tensor_value_info('7', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 3, 3])

    node = onnx.helper.make_node(
        'MatMulIntegerToFloat',
        inputs=['1', '2', '3', '4', '5', '6', '7'],
        outputs=['y'],
    )

    return ([node], [m1, m2, s1, s2, zp1, zp2, b1], [y])


@onnx_test()
def max_test():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Max',
        inputs=['0', '1', '2'],
        outputs=['3'],
    )

    return ([node], [a, b, c], [y])


@onnx_test()
def maxpool_notset_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 1, 1])

    node = onnx.helper.make_node('MaxPool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[6, 6],
                                 strides=[2, 2],
                                 pads=[0, 0, 1, 1],
                                 auto_pad='NOTSET')

    return ([node], [x], [y])


@onnx_test()
def maxpool_dilate_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 2])

    node = onnx.helper.make_node('MaxPool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2],
                                 strides=[1],
                                 pads=[1, 1],
                                 dilations=[3])

    return ([node], [x], [y])


@onnx_test()
def maxpool_same_upper_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

    node = onnx.helper.make_node('MaxPool',
                                 inputs=['x'],
                                 outputs=['y'],
                                 kernel_shape=[2, 2],
                                 auto_pad='SAME_UPPER')

    return ([node], [x], [y])


@onnx_test()
def mean_broadcast_test():
    data_0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 4])
    data_1 = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                           [1, 2, 3, 4])
    data_2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    data_3 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1])
    data_4 = helper.make_tensor_value_info('4', TensorProto.FLOAT, [2, 3, 1])

    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT,
                                         [1, 2, 3, 4])

    node = onnx.helper.make_node("Mean",
                                 inputs=["0", "1", "2", "3", "4"],
                                 outputs=["mean"])

    return ([node], [data_0, data_1, data_2, data_3, data_4], [mean])


@onnx_test()
def mean_fp16_test():
    data_0 = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [1, 2, 3])
    data_1 = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [1, 2, 3])
    data_2 = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [1, 2, 3])

    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT16,
                                         [1, 2, 3])

    node = onnx.helper.make_node("Mean",
                                 inputs=["0", "1", "2"],
                                 outputs=["mean"])

    return ([node], [data_0, data_1, data_2], [mean])


@onnx_test()
def mean_bf16_test():
    data_0 = helper.make_tensor_value_info('0', TensorProto.BFLOAT16,
                                           [1, 2, 3])
    data_1 = helper.make_tensor_value_info('1', TensorProto.BFLOAT16,
                                           [1, 2, 3])
    data_2 = helper.make_tensor_value_info('2', TensorProto.BFLOAT16,
                                           [1, 2, 3])

    mean = helper.make_tensor_value_info('mean', TensorProto.BFLOAT16,
                                         [1, 2, 3])

    node = onnx.helper.make_node("Mean",
                                 inputs=["0", "1", "2"],
                                 outputs=["mean"])

    return ([node], [data_0, data_1, data_2], [mean])


@onnx_test()
def mean_invalid_broadcast_test():
    data_0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 3])
    data_1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 2, 3])
    data_2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 2, 4])

    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [1, 2, 3])

    node = onnx.helper.make_node("Mean",
                                 inputs=["0", "1", "2"],
                                 outputs=["mean"])

    return ([node], [data_0, data_1, data_2], [mean])


@onnx_test()
def mean_single_input_test():
    data_0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 3])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [1, 2, 3])

    node = onnx.helper.make_node("Mean", inputs=["0"], outputs=["mean"])

    return ([node], [data_0], [mean])


@onnx_test()
def mean_test():
    data = [
        helper.make_tensor_value_info(str(i), TensorProto.DOUBLE, [2, 2, 2])
        for i in range(10)
    ]
    data_names = [str(i) for i in range(10)]
    mean = helper.make_tensor_value_info('mean', TensorProto.DOUBLE, [2, 2, 2])

    node = onnx.helper.make_node("Mean", inputs=data_names, outputs=["mean"])

    return ([node], data, [mean])


@onnx_test()
def mean_integral_test():
    data = [
        helper.make_tensor_value_info(str(i), TensorProto.INT32, [2, 2, 2])
        for i in range(10)
    ]
    data_names = [str(i) for i in range(10)]
    mean = helper.make_tensor_value_info('mean', TensorProto.INT32, [2, 2, 2])

    node = onnx.helper.make_node("Mean", inputs=data_names, outputs=["mean"])

    return ([node], data, [mean])


def mvn_default_axes_test_base(dims, type=TensorProto.FLOAT):
    data = helper.make_tensor_value_info("data", type, dims)
    out = helper.make_tensor_value_info("out", type, dims)
    node = helper.make_node("MeanVarianceNormalization",
                            inputs=["data"],
                            outputs=["out"])

    return ([node], [data], [out])


@onnx_test()
def mvn_default_axes_test():
    return mvn_default_axes_test_base([2, 2, 2, 2])


@onnx_test()
def mvn_default_axes_fp16_test():
    return mvn_default_axes_test_base([2, 2, 2, 2], TensorProto.FLOAT16)


@onnx_test()
def mvn_default_axes_bf16_test():
    return mvn_default_axes_test_base([2, 2, 2, 2], TensorProto.BFLOAT16)


@onnx_test()
def mvn_default_axes_rank_too_small_test():
    return mvn_default_axes_test_base([2, 2, 2])


@onnx_test()
def mvn_default_axes_rank_too_big_test():
    return mvn_default_axes_test_base([2, 2, 2, 2, 2])


def mvn_n_rank_test_base(axes, dims, type=TensorProto.FLOAT):
    data = helper.make_tensor_value_info("data", type, dims)
    out = helper.make_tensor_value_info("out", type, dims)
    node = helper.make_node("MeanVarianceNormalization",
                            inputs=["data"],
                            outputs=["out"],
                            axes=axes)

    return ([node], [data], [out])


@onnx_test()
def mvn_rank_2_test():
    return mvn_n_rank_test_base([1], [2, 2])


@onnx_test()
def mvn_rank_2_fp16_test():
    return mvn_n_rank_test_base([1], [2, 2], TensorProto.FLOAT16)


@onnx_test()
def mvn_rank_2_bf16_test():
    return mvn_n_rank_test_base([1], [2, 2], TensorProto.BFLOAT16)


@onnx_test()
def mvn_rank_3_test():
    return mvn_n_rank_test_base([0, 1], [2, 2, 2])


@onnx_test()
def mvn_rank_3_fp16_test():
    return mvn_n_rank_test_base([0, 1], [2, 2, 2], TensorProto.FLOAT16)


@onnx_test()
def mvn_rank_3_bf16_test():
    return mvn_n_rank_test_base([0, 1], [2, 2, 2], TensorProto.BFLOAT16)


@onnx_test()
def mvn_axes_rank_too_small_test():
    return mvn_n_rank_test_base([0, 1, 2], [2, 2, 2])


@onnx_test()
def mvn_axes_rank_too_big_test():
    return mvn_n_rank_test_base([0], [2, 2, 2])


@onnx_test()
def min_test():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Min',
        inputs=['0', '1', '2'],
        outputs=['3'],
    )

    return ([node], [a, b, c], [y])


@onnx_test()
def mod_test():
    a = helper.make_tensor_value_info('0', TensorProto.INT32, [3, 3, 3])
    b = helper.make_tensor_value_info('1', TensorProto.INT32, [3, 3, 3])
    y = helper.make_tensor_value_info('2', TensorProto.INT32, [3, 3, 3])

    node = onnx.helper.make_node('Mod', inputs=['0', '1'], outputs=['2'])

    return ([node], [a, b], [y])


@onnx_test()
def mod_test_half():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [3, 3, 3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [3, 3, 3])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [3, 3, 3])

    node = onnx.helper.make_node('Mod', inputs=['0', '1'], outputs=['2'])

    return ([node], [a, b], [y])


# @onnx_test()
# def mod_test_bf16():
#     a = helper.make_tensor_value_info('0', TensorProto.BFLOAT16, [3, 3, 3])
#     b = helper.make_tensor_value_info('1', TensorProto.BFLOAT16, [3, 3, 3])
#     y = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [3, 3, 3])

#     node = onnx.helper.make_node('Mod', inputs=['0', '1'], outputs=['2'])

#     return ([node], [a, b], [y])


@onnx_test()
def mod_test_different_dtypes():
    a = helper.make_tensor_value_info('0', TensorProto.INT16, [3, 3, 3])
    b = helper.make_tensor_value_info('1', TensorProto.INT32, [3, 3, 3])
    y = helper.make_tensor_value_info('2', TensorProto.INT32, [3, 3, 3])

    node = onnx.helper.make_node(
        'Mod',
        inputs=['0', '1'],
        outputs=['2'],
    )

    return ([node], [a, b], [y])


@onnx_test()
def mod_test_fmod():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 3, 3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 3, 3])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 3, 3])

    node = onnx.helper.make_node(
        'Mod',
        inputs=['0', '1'],
        outputs=['2'],
        fmod=1  #fmod flag = 1
    )

    return ([node], [a, b], [y])


@onnx_test()
def mod_test_fmod_half():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [3, 3, 3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [3, 3, 3])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [3, 3, 3])

    node = onnx.helper.make_node('Mod',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 fmod=1)

    return ([node], [a, b], [y])


@onnx_test()
def mod_test_fmod_bf16():
    a = helper.make_tensor_value_info('0', TensorProto.BFLOAT16, [3, 3, 3])
    b = helper.make_tensor_value_info('1', TensorProto.BFLOAT16, [3, 3, 3])
    y = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [3, 3, 3])

    node = onnx.helper.make_node('Mod',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 fmod=1)

    return ([node], [a, b], [y])


@onnx_test()
def mod_test_fmod_different_dtypes():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 3, 3])
    b = helper.make_tensor_value_info('1', TensorProto.INT32, [3, 3, 3])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 3, 3])

    node = onnx.helper.make_node(
        'Mod',
        inputs=['0', '1'],
        outputs=['2'],
        fmod=1  #fmod flag = 1
    )

    return ([node], [a, b], [y])


@onnx_test()
def mha_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 4])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 4])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=['out'],
                            num_heads=2,
                            domain='com.microsoft')

    return ([node], [query, key, value], [out])


@onnx_test()
def mha_cross_attention_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 4])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 2, 2])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 2, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=['out'],
                            num_heads=2,
                            domain='com.microsoft')

    return ([node], [query, key, value], [out])


@onnx_test()
def mha_kv_packed_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 4])
    kv = helper.make_tensor_value_info("kv", TensorProto.FLOAT,
                                       [1, 2, 2, 2, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'kv'],
                            outputs=['out'],
                            num_heads=2,
                            domain='com.microsoft')

    return ([node], [query, kv], [out])


@onnx_test()
def mha_qkv_packed_test():
    qkv = helper.make_tensor_value_info("qkv", TensorProto.FLOAT,
                                        [1, 2, 2, 3, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['qkv'],
                            outputs=['out'],
                            num_heads=2,
                            domain='com.microsoft')

    return ([node], [qkv], [out])


def mha_scale_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 4])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 4])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=['out'],
                            num_heads=2,
                            scale=0.1,
                            domain='com.microsoft')

    return ([node], [query, key, value], [out])


@onnx_test()
def mha_invalid_attribute_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 4])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 4])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=['out'],
                            domain='com.microsoft')

    return ([node], [query, key, value], [out])


@onnx_test()
def mha_invalid_input_test():
    node = helper.make_node('MultiHeadAttention',
                            inputs=[],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [], [])


@onnx_test()
def mha_invalid_query_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query], [])


@onnx_test()
def mha_invalid_qkv_test():
    qkv = helper.make_tensor_value_info("qkv", TensorProto.FLOAT,
                                        [1, 1, 1, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['qkv'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [qkv], [])


@onnx_test()
def mha_invalid_key_missing_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query], [])


@onnx_test()
def mha_invalid_key_ndim_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, key], [])


@onnx_test()
def mha_invalid_kv_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    kv = helper.make_tensor_value_info("kv", TensorProto.FLOAT,
                                       [1, 1, 1, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'kv'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, kv], [])


@onnx_test()
def mha_invalid_key_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, 2])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, key, value], [])


@onnx_test()
def mha_invalid_value_missing_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, key], [])


@onnx_test()
def mha_invalid_value_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, 1])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [2, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, key, value], [])


@onnx_test()
def mha_invalid_value_ndim_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, 1])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 1, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, key, value], [])


@onnx_test()
def mha_invalid_cross_key_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 1, 1])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 1, 1, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, key, value], [])


@onnx_test()
def mha_invalid_cross_value_test():
    query = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 1])
    key = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, 1, 1])
    value = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 1, 2, 1])

    node = helper.make_node('MultiHeadAttention',
                            inputs=['q', 'k', 'v'],
                            outputs=[],
                            num_heads=1,
                            domain='com.microsoft')

    return ([node], [query, key, value], [])


@onnx_test()
def multinomial_test():
    sample_size = 13
    seed = 0.
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 10])
    output = helper.make_tensor_value_info("output", TensorProto.INT32,
                                           [1, 10])

    node = onnx.helper.make_node('Multinomial',
                                 inputs=['input'],
                                 sample_size=sample_size,
                                 seed=seed,
                                 outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def multinomial_dyn_test():
    sample_size = 100000
    seed = 1.3
    categories = 5
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                          [None, categories])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                           [None, categories])

    node = onnx.helper.make_node(
        'Multinomial',
        inputs=['input'],
        sample_size=sample_size,
        dtype=1,  # shape::float_type
        seed=seed,
        outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def multinomial_autoseed_dyn_test():
    # If seed attribute is not given, device should auto generate one at runtime
    sample_size = 12
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                          [None, 10])
    output = helper.make_tensor_value_info("output", TensorProto.INT32,
                                           [None, 10])

    node = onnx.helper.make_node('Multinomial',
                                 inputs=['input'],
                                 sample_size=sample_size,
                                 outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def multinomial_generated_seed_test():
    sample_size = 10
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output = helper.make_tensor_value_info("output", TensorProto.INT32,
                                           [1, 10])

    node = onnx.helper.make_node('Multinomial',
                                 inputs=['input'],
                                 sample_size=sample_size,
                                 outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def multinomial_dtype_error_test():
    sample_size = 10
    dtype = 0
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output = helper.make_tensor_value_info("output", TensorProto.INT64,
                                           [1, 10])

    node = onnx.helper.make_node('Multinomial',
                                 inputs=['input'],
                                 sample_size=sample_size,
                                 dtype=dtype,
                                 outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def multinomial_int64_test():
    sample_size = 10
    dtype = 7
    seed = 1.0
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output = helper.make_tensor_value_info("output", TensorProto.INT64,
                                           [1, 10])

    node = onnx.helper.make_node('Multinomial',
                                 inputs=['input'],
                                 sample_size=sample_size,
                                 dtype=dtype,
                                 seed=seed,
                                 outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def neg_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT64, [2, 3])
    y = helper.make_tensor_value_info('1', TensorProto.INT64, [2, 3])

    node = onnx.helper.make_node('Neg', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def neg_dynamic_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT64, [None, 3])
    y = helper.make_tensor_value_info('1', TensorProto.INT64, [None, 3])

    node = onnx.helper.make_node('Neg', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def nhwcconv_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 7, 7, 1])
    w = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 1, 1, 1])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 7, 7, 1])

    node = onnx.helper.make_node('NhwcConv', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, w], [out])


@onnx_test()
def nms_test():
    b = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 6, 4])
    s = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 1, 6])
    mo = helper.make_tensor_value_info('max_output_boxes_per_class',
                                       TensorProto.INT64, [1])
    iou = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT,
                                        [1])
    st = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT,
                                       [1])
    out = helper.make_tensor_value_info('selected_indices', TensorProto.INT64,
                                        [None, 3])

    node = onnx.helper.make_node('NonMaxSuppression',
                                 inputs=[
                                     'boxes', 'scores',
                                     'max_output_boxes_per_class',
                                     'iou_threshold', 'score_threshold'
                                 ],
                                 outputs=['selected_indices'],
                                 center_point_box=1)

    return ([node], [b, s, mo, iou, st], [out])


@onnx_test()
def nms_use_dyn_output_false_test():
    b = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 6, 4])
    s = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 1, 6])
    mo = helper.make_tensor_value_info('max_output_boxes_per_class',
                                       TensorProto.INT64, [1])
    iou = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT,
                                        [1])
    st = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT,
                                       [1])
    out = helper.make_tensor_value_info('selected_indices', TensorProto.INT64,
                                        [None, 3])

    node = onnx.helper.make_node('NonMaxSuppression',
                                 inputs=[
                                     'boxes', 'scores',
                                     'max_output_boxes_per_class',
                                     'iou_threshold', 'score_threshold'
                                 ],
                                 outputs=['selected_indices'],
                                 use_dyn_output=0)

    return ([node], [b, s, mo, iou, st], [out])


@onnx_test()
def nms_dynamic_batch_test():
    b = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [None, 6, 4])
    s = helper.make_tensor_value_info('scores', TensorProto.FLOAT,
                                      [None, 1, 6])
    mo = helper.make_tensor_value_info('max_output_boxes_per_class',
                                       TensorProto.INT64, [1])
    iou = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT,
                                        [1])
    st = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT,
                                       [1])
    out = helper.make_tensor_value_info('selected_indices', TensorProto.INT64,
                                        [None, 3])

    node = onnx.helper.make_node('NonMaxSuppression',
                                 inputs=[
                                     'boxes', 'scores',
                                     'max_output_boxes_per_class',
                                     'iou_threshold', 'score_threshold'
                                 ],
                                 outputs=['selected_indices'],
                                 center_point_box=1,
                                 use_dyn_output=1)

    return ([node], [b, s, mo, iou, st], [out])


@onnx_test()
def nms_dynamic_boxes_test():
    b = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, None, 4])
    s = helper.make_tensor_value_info('scores', TensorProto.FLOAT,
                                      [1, 1, None])
    mo = helper.make_tensor_value_info('max_output_boxes_per_class',
                                       TensorProto.INT64, [1])
    iou = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT,
                                        [1])
    st = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT,
                                       [1])
    out = helper.make_tensor_value_info('selected_indices', TensorProto.INT64,
                                        [None, 3])

    node = onnx.helper.make_node('NonMaxSuppression',
                                 inputs=[
                                     'boxes', 'scores',
                                     'max_output_boxes_per_class',
                                     'iou_threshold', 'score_threshold'
                                 ],
                                 outputs=['selected_indices'])

    return ([node], [b, s, mo, iou, st], [out])


@onnx_test()
def nms_dynamic_classes_test():
    b = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 6, 4])
    s = helper.make_tensor_value_info('scores', TensorProto.FLOAT,
                                      [1, None, 6])
    mo = helper.make_tensor_value_info('max_output_boxes_per_class',
                                       TensorProto.INT64, [1])
    iou = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT,
                                        [1])
    st = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT,
                                       [1])
    out = helper.make_tensor_value_info('selected_indices', TensorProto.INT64,
                                        [None, 3])

    node = onnx.helper.make_node('NonMaxSuppression',
                                 inputs=[
                                     'boxes', 'scores',
                                     'max_output_boxes_per_class',
                                     'iou_threshold', 'score_threshold'
                                 ],
                                 outputs=['selected_indices'])

    return ([node], [b, s, mo, iou, st], [out])


@onnx_test()
def not_test():
    x = helper.make_tensor_value_info('0', TensorProto.INT32, [4])
    y = helper.make_tensor_value_info('1', TensorProto.INT32, [4])

    node = onnx.helper.make_node('Not', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def not_bool_test():
    x = helper.make_tensor_value_info('0', TensorProto.BOOL, [4])
    y = helper.make_tensor_value_info('1', TensorProto.BOOL, [4])

    node = onnx.helper.make_node('Not', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def no_pad_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0'],
                                 pads=[0, 0, 0, 0],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def nonzero_dynamic_test():
    x = helper.make_tensor_value_info('data', TensorProto.BOOL, [2, 2])
    y = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 3])

    node = onnx.helper.make_node('NonZero',
                                 inputs=['data'],
                                 outputs=['indices'])

    return ([node], [x], [y])


@onnx_test()
def nonzero_test():
    data1 = np.array([[1., 0.], [1., 1.]])
    data = helper.make_tensor(name='data',
                              data_type=TensorProto.FLOAT,
                              dims=data1.shape,
                              vals=data1.flatten().astype(np.float))
    y = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 3])

    node = onnx.helper.make_node('NonZero',
                                 inputs=['data'],
                                 outputs=['indices'])

    return ([node], [], [y], [data])


@onnx_test()
def nonzero_int_test():
    data1 = np.array([[1, 1, 0], [1, 0, 1]])
    data = helper.make_tensor(name='data',
                              data_type=TensorProto.INT16,
                              dims=data1.shape,
                              vals=data1.flatten().astype(np.int16))
    y = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 4])

    node = onnx.helper.make_node('NonZero',
                                 inputs=['data'],
                                 outputs=['indices'])

    return ([node], [], [y], [data])


@onnx_test()
def onehot_static_test():
    axis_value = 0
    depth = np.array([3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT32,
                                            [5, 2])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT16, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [3, 5, 2])

    depth_tensor = helper.make_tensor(name="depth",
                                      data_type=TensorProto.INT32,
                                      dims=depth.shape,
                                      vals=depth.astype(int))

    node = onnx.helper.make_node('OneHot',
                                 inputs=['indices', 'depth', 'values'],
                                 outputs=['y'],
                                 axis=axis_value)

    return ([node], [indices, values], [y], [depth_tensor])


@onnx_test()
def onehot_dyn_test0():
    axis_value = -1
    depth = np.array([3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT32,
                                            [None, 2])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT16, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [3, 5, 2])

    depth_tensor = helper.make_tensor(name="depth",
                                      data_type=TensorProto.INT32,
                                      dims=depth.shape,
                                      vals=depth.astype(int))

    node = onnx.helper.make_node('OneHot',
                                 inputs=['indices', 'depth', 'values'],
                                 outputs=['y'],
                                 axis=axis_value)

    return ([node], [indices, values], [y], [depth_tensor])


@onnx_test()
def onehot_dyn_test1():
    axis_value = 1
    indices = helper.make_tensor_value_info("indices", TensorProto.INT32,
                                            [None, 2])
    depth = helper.make_tensor_value_info("depth", TensorProto.INT64, [1])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5, 2])

    node = onnx.helper.make_node('OneHot',
                                 inputs=['indices', 'depth', 'values'],
                                 outputs=['y'],
                                 axis=axis_value)

    return ([node], [indices, values, depth], [y])


@onnx_test()
def pad_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0'],
                                 pads=[1, 1, 1, 1],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def pad_asym_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 6, 4, 12])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0'],
                                 pads=[0, 1, 0, 3, 0, 2, 0, 4],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def pad_asym_invalid_pads_error_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 6, 4, 12])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0'],
                                 pads=[0, 1, 0, 3, 0, 2],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def pad_3arg_test():
    values = np.array([1])
    val_tensor = helper.make_tensor(name='val',
                                    data_type=TensorProto.FLOAT,
                                    dims=values.reshape(()).shape,
                                    vals=values.astype(float))
    arg_val = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_val'],
                                    value=val_tensor)

    sizes = np.array([1, 1, 2, 2])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [5, 5])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0', 'arg_pad', 'arg_val'],
                                 outputs=['1'])

    return ([arg_val, arg_pad, node], [x], [y])


@onnx_test()
def pad_undef_const_val_test():
    sizes = np.array([1, 1, 1, 1])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0', 'arg_pad', ''],
                                 outputs=['1'])

    return ([arg_pad, node], [x], [y])


@onnx_test()
def pad_4arg_axes_test():
    values = np.array([1])
    val_tensor = helper.make_tensor(name='val',
                                    data_type=TensorProto.FLOAT,
                                    dims=values.reshape(()).shape,
                                    vals=values.astype(float))
    arg_val = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_val'],
                                    value=val_tensor)

    sizes = np.array([1, 3, 2, 4])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    axes = np.array([1, 3])
    axes_tensor = helper.make_tensor(name='pad_axes',
                                     data_type=TensorProto.INT32,
                                     dims=axes.shape,
                                     vals=axes.astype(int))
    arg_axes = onnx.helper.make_node('Constant',
                                     inputs=[],
                                     outputs=['arg_axes'],
                                     value=axes_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 6, 4, 12])

    node = onnx.helper.make_node(
        'Pad', inputs=['0', 'arg_pad', 'arg_val', 'arg_axes'], outputs=['1'])

    return ([arg_axes, arg_val, arg_pad, node], [x], [y])


@onnx_test()
def pad_4arg_invalid_axes_error_test():
    values = np.array([1])
    val_tensor = helper.make_tensor(name='val',
                                    data_type=TensorProto.FLOAT,
                                    dims=values.reshape(()).shape,
                                    vals=values.astype(float))
    arg_val = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_val'],
                                    value=val_tensor)

    sizes = np.array([1, 3, 2, 4])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    axes = np.array([1, 2, 3])
    axes_tensor = helper.make_tensor(name='pad_axes',
                                     data_type=TensorProto.INT32,
                                     dims=axes.shape,
                                     vals=axes.astype(int))
    arg_axes = onnx.helper.make_node('Constant',
                                     inputs=[],
                                     outputs=['arg_axes'],
                                     value=axes_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 6, 4, 12])

    node = onnx.helper.make_node(
        'Pad', inputs=['0', 'arg_pad', 'arg_val', 'arg_axes'], outputs=['1'])

    return ([arg_axes, arg_val, arg_pad, node], [x], [y])


@onnx_test()
def pad_4arg_neg_axes_test():
    values = np.array([1])
    val_tensor = helper.make_tensor(name='val',
                                    data_type=TensorProto.FLOAT,
                                    dims=values.reshape(()).shape,
                                    vals=values.astype(float))
    arg_val = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_val'],
                                    value=val_tensor)

    sizes = np.array([1, 3, 2, 4])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    axes = np.array([-3, -1])
    axes_tensor = helper.make_tensor(name='pad_axes',
                                     data_type=TensorProto.INT32,
                                     dims=axes.shape,
                                     vals=axes.astype(int))
    arg_axes = onnx.helper.make_node('Constant',
                                     inputs=[],
                                     outputs=['arg_axes'],
                                     value=axes_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 6, 4, 12])

    node = onnx.helper.make_node(
        'Pad', inputs=['0', 'arg_pad', 'arg_val', 'arg_axes'], outputs=['1'])

    return ([arg_axes, arg_val, arg_pad, node], [x], [y])


@onnx_test()
def pad_reflect_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 5])

    sizes = np.array([0, 2, 0, 1])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    node = onnx.helper.make_node('Pad',
                                 mode='reflect',
                                 inputs=['0', 'arg_pad'],
                                 outputs=['1'])

    return ([arg_pad, node], [x], [y])


@onnx_test()
def pad_reflect_with_axes_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 5])

    sizes = np.array([2, 1])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    axes = np.array([1])
    axes_tensor = helper.make_tensor(name='pad_axes',
                                     data_type=TensorProto.INT32,
                                     dims=axes.shape,
                                     vals=axes.astype(int))
    arg_axes = onnx.helper.make_node('Constant',
                                     inputs=[],
                                     outputs=['arg_axes'],
                                     value=axes_tensor)

    node = onnx.helper.make_node('Pad',
                                 mode='reflect',
                                 inputs=['0', 'arg_pad', 'arg_axes'],
                                 outputs=['1'])

    return ([arg_axes, arg_pad, node], [x], [y])


@onnx_test()
def pad_reflect_multiaxis_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 5])

    sizes = np.array([0, 2, 2, 0])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    node = onnx.helper.make_node('Pad',
                                 mode='reflect',
                                 inputs=['0', 'arg_pad'],
                                 outputs=['1'])

    return ([arg_pad, node], [x], [y])


@onnx_test()
def pad_attr_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, None])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0'],
                                 pads=[1, 1, 1, 1],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def pad_cnst_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, None])

    sizes = np.array([0, 2, 0, 1])
    pad_tensor = helper.make_tensor(name='pad_size',
                                    data_type=TensorProto.INT32,
                                    dims=sizes.shape,
                                    vals=sizes.astype(int))
    arg_pad = onnx.helper.make_node('Constant',
                                    inputs=[],
                                    outputs=['arg_pad'],
                                    value=pad_tensor)

    node = onnx.helper.make_node('Pad', inputs=['0', 'arg_pad'], outputs=['1'])

    return ([arg_pad, node], [x], [y])


@onnx_test()
def pad_dyn_reflect_error():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, None])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, None])

    node = onnx.helper.make_node('Pad',
                                 mode='reflect',
                                 inputs=['0'],
                                 pads=[0, 2, 0, 1],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def pow_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def pow_fp32_i64_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.INT64, [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def pow_i64_fp32_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.INT64, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.INT64,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def prefix_scan_sum_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 2])
    axis_val = np.array([0])
    axis_tensor = helper.make_tensor(name="axis",
                                     data_type=TensorProto.INT32,
                                     dims=axis_val.shape,
                                     vals=axis_val.astype(int))
    node = onnx.helper.make_node('CumSum',
                                 inputs=['x', 'axis'],
                                 outputs=['y'],
                                 exclusive=1,
                                 reverse=1)
    return ([node], [x], [y], [axis_tensor])


@onnx_test()
def prelu_brcst_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'PRelu',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def qlinearadd_test():
    a = helper.make_tensor_value_info('A', TensorProto.UINT8, [64])
    sc_a = helper.make_tensor('A_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_a = helper.make_tensor('A_zero_point', TensorProto.UINT8, [], [0])

    b = helper.make_tensor_value_info('B', TensorProto.UINT8, [64])
    sc_b = helper.make_tensor('B_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_b = helper.make_tensor('B_zero_point', TensorProto.UINT8, [],
                                   [128])

    sc_c = helper.make_tensor('C_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_c = helper.make_tensor('C_zero_point', TensorProto.UINT8, [], [64])

    c = helper.make_tensor_value_info('C', TensorProto.UINT8, [64])

    node = onnx.helper.make_node(
        'QLinearAdd',
        inputs=[
            'A', 'A_scale', 'A_zero_point', 'B', 'B_scale', 'B_zero_point',
            'C_scale', 'C_zero_point'
        ],
        outputs=['C'],
    )
    return ([node], [a, b], [c],
            [sc_a, zero_pt_a, sc_b, zero_pt_b, sc_c, zero_pt_c])


@onnx_test()
def qlinearadd_bcast_test():
    a = helper.make_tensor_value_info('A', TensorProto.INT8, [64])
    sc_a = helper.make_tensor('A_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_a = helper.make_tensor('A_zero_point', TensorProto.INT8, [], [0])

    b = helper.make_tensor_value_info('B', TensorProto.INT8, [1, 1, 64])
    sc_b = helper.make_tensor('B_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_b = helper.make_tensor('B_zero_point', TensorProto.INT8, [], [32])

    sc_c = helper.make_tensor('C_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_c = helper.make_tensor('C_zero_point', TensorProto.INT8, [], [-64])

    c = helper.make_tensor_value_info('C', TensorProto.INT8, [1, 1, 64])

    node = onnx.helper.make_node(
        'QLinearAdd',
        inputs=[
            'A', 'A_scale', 'A_zero_point', 'B', 'B_scale', 'B_zero_point',
            'C_scale', 'C_zero_point'
        ],
        outputs=['C'],
    )
    return ([node], [a, b], [c],
            [sc_a, zero_pt_a, sc_b, zero_pt_b, sc_c, zero_pt_c])


@onnx_test()
def qlinearaveragepool_1d_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 3, 32])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.05])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 31])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.05])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [16])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[2],
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_2d_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 3, 4, 4])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.05])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.INT8, [1, 3, 3, 3])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.015])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [16])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[2, 2],
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_2d_ceil_test():
    x = helper.make_tensor_value_info('x', TensorProto.UINT8, [1, 1, 4, 4])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.5])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.UINT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [1, 1, 2, 2])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.05])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.UINT8, [],
                                      [0])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        ceil_mode=True,
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_2d_dilations_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 1, 4, 4])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.5])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.INT8, [1, 1, 2, 2])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.25])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [84])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[2, 2],
        strides=[1, 1],
        dilations=[2, 2],
        ceil_mode=True,
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_2d_pads_count_include_pad_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 3, 4, 4])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.05])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.INT8, [1, 3, 6, 6])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.01])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [32])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2],
        count_include_pad=1,
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_2d_same_lower_test():
    x = helper.make_tensor_value_info('x', TensorProto.UINT8, [1, 3, 4, 4])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.5])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.UINT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [1, 3, 4, 4])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.5])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.UINT8, [],
                                      [0])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[2, 2],
        auto_pad="SAME_LOWER",
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_2d_same_upper_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 3, 4, 4])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.5])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [32])

    y = helper.make_tensor_value_info('y', TensorProto.INT8, [1, 3, 4, 4])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.25])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [0])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[2, 2],
        auto_pad="SAME_UPPER",
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_2d_strides_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 3, 8, 8])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.05])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.INT8, [1, 3, 2, 2])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.05])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [8])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[5, 5],
        strides=[2, 2],
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_3d_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 3, 3, 3, 3])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.05])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [0])

    y = helper.make_tensor_value_info('y', TensorProto.INT8, [1, 3, 2, 2, 2])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.02])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [0])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[2, 2, 2],
    )

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_notset_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [1, 1, 5, 5])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.5])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.INT8, [],
                                      [0])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [1, 1, 1, 1])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.5])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.INT8, [],
                                      [10])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[6, 6],
        strides=[2, 2],
        pads=[0, 0, 1, 1],
        channels_last=0,
        auto_pad='NOTSET')

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearaveragepool_nt_cip_test():
    x = helper.make_tensor_value_info('x', TensorProto.UINT8, [1, 1, 5, 5])
    x_scale = helper.make_tensor('x_scale', TensorProto.FLOAT, [], [0.5])
    x_zero_point = helper.make_tensor('x_zero_point', TensorProto.UINT8, [],
                                      [0])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [1, 1, 1, 1])
    y_scale = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [0.5])
    y_zero_point = helper.make_tensor('y_zero_point', TensorProto.UINT8, [],
                                      [10])

    node = onnx.helper.make_node(
        'QLinearAveragePool',
        inputs=['x', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[6, 6],
        strides=[2, 2],
        pads=[0, 0, 1, 1],
        channels_last=0,
        auto_pad='NOTSET',
        count_include_pad=1)

    return ([node], [x], [y], [x_scale, x_zero_point, y_scale, y_zero_point])


@onnx_test()
def qlinearconcat_test():
    y_scale = helper.make_tensor('1', TensorProto.FLOAT, [], [0.5])
    y_zero_point = helper.make_tensor('2', TensorProto.INT8, [], [2])

    t0 = helper.make_tensor_value_info('t0', TensorProto.INT8, [2])
    s0 = helper.make_tensor('3', TensorProto.FLOAT, [], [0.5])
    zp0 = helper.make_tensor('4', TensorProto.INT8, [], [1])

    t1 = helper.make_tensor_value_info('t1', TensorProto.INT8, [3])
    s1 = helper.make_tensor('5', TensorProto.FLOAT, [], [0.25])
    zp1 = helper.make_tensor('6', TensorProto.INT8, [], [0])

    y = helper.make_tensor_value_info('out', TensorProto.INT8, [5])

    node = onnx.helper.make_node(
        'QLinearConcat',
        inputs=['1', '2', 't0', '3', '4', 't1', '5', '6'],
        axis=0,
        outputs=['out'],
    )

    return ([node], [t0, t1], [y], [y_scale, y_zero_point, s0, zp0, s1, zp1])


@onnx_test()
def qlinearconcat_3d_test():
    y_scale = helper.make_tensor('1', TensorProto.FLOAT, [], [0.5])
    y_zero_point = helper.make_tensor('2', TensorProto.INT8, [], [2])

    t0 = helper.make_tensor_value_info('t0', TensorProto.INT8, [3, 4, 2])
    s0 = helper.make_tensor('3', TensorProto.FLOAT, [], [0.5])
    zp0 = helper.make_tensor('4', TensorProto.INT8, [], [10])

    t1 = helper.make_tensor_value_info('t1', TensorProto.INT8, [3, 2, 2])
    s1 = helper.make_tensor('5', TensorProto.FLOAT, [], [0.4])
    zp1 = helper.make_tensor('6', TensorProto.INT8, [], [20])

    y = helper.make_tensor_value_info('out', TensorProto.UINT8, [3, 6, 2])

    node = onnx.helper.make_node(
        'QLinearConcat',
        inputs=['1', '2', 't0', '3', '4', 't1', '5', '6'],
        axis=1,
        outputs=['out'],
    )

    return ([node], [t0, t1], [y], [y_scale, y_zero_point, s0, zp0, s1, zp1])


@onnx_test()
def qlinearconv_test():
    # https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__QLinearConv.html
    x = helper.make_tensor_value_info('X', TensorProto.UINT8, [1, 1, 7, 7])
    sc_x = helper.make_tensor('1', TensorProto.FLOAT, [], [0.00369204697])
    zero_pt_x = helper.make_tensor('2', TensorProto.UINT8, [], [132])

    wt = helper.make_tensor('3', TensorProto.UINT8, [1, 1, 1, 1], [0])
    sc_wt = helper.make_tensor('4', TensorProto.FLOAT, [], [0.00172794575])
    zero_pt_wt = helper.make_tensor('5', TensorProto.UINT8, [], [255])

    sc_y = helper.make_tensor('6', TensorProto.FLOAT, [], [0.00162681262])
    zero_pt_y = helper.make_tensor('7', TensorProto.UINT8, [], [123])

    out = helper.make_tensor_value_info('out', TensorProto.UINT8, [1, 1, 7, 7])

    node = onnx.helper.make_node(
        'QLinearConv',
        inputs=['X', '1', '2', '3', '4', '5', '6', '7'],
        outputs=['out'],
    )
    return ([node], [x], [out],
            [sc_x, zero_pt_x, wt, sc_wt, zero_pt_wt, sc_y, zero_pt_y])


@onnx_test()
def qlinearconv_pad_1_test():
    # https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html
    x = helper.make_tensor_value_info('X', TensorProto.UINT8, [1, 1, 5, 5])
    sc_x = helper.make_tensor('1', TensorProto.FLOAT, [],
                              [0.09411764705882353])
    zero_pt_x = helper.make_tensor('2', TensorProto.UINT8, [], [0])

    wt = helper.make_tensor('3', TensorProto.UINT8, [1, 1, 3, 3],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1])
    sc_wt = helper.make_tensor('4', TensorProto.FLOAT, [], [1.0])
    zero_pt_wt = helper.make_tensor('5', TensorProto.UINT8, [], [0])

    sc_y = helper.make_tensor('6', TensorProto.FLOAT, [], [0.6352941176470588])
    zero_pt_y = helper.make_tensor('7', TensorProto.UINT8, [], [0])

    out = helper.make_tensor_value_info('out', TensorProto.UINT8, [1, 1, 5, 5])

    node = onnx.helper.make_node(
        'QLinearConv',
        inputs=['X', '1', '2', '3', '4', '5', '6', '7'],
        outputs=['out'],
        pads=[1, 1, 1, 1],
    )
    return ([node], [x], [out],
            [sc_x, zero_pt_x, wt, sc_wt, zero_pt_wt, sc_y, zero_pt_y])


@onnx_test()
def qlinearconv_pad_0_test():
    # https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html
    x = helper.make_tensor_value_info('X', TensorProto.UINT8, [1, 1, 5, 5])
    sc_x = helper.make_tensor('1', TensorProto.FLOAT, [],
                              [0.09411764705882353])
    zero_pt_x = helper.make_tensor('2', TensorProto.UINT8, [], [0])

    wt = helper.make_tensor('3', TensorProto.UINT8, [1, 1, 3, 3],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1])
    sc_wt = helper.make_tensor('4', TensorProto.FLOAT, [], [1.0])
    zero_pt_wt = helper.make_tensor('5', TensorProto.UINT8, [], [0])

    sc_y = helper.make_tensor('6', TensorProto.FLOAT, [], [0.6352941176470588])
    zero_pt_y = helper.make_tensor('7', TensorProto.INT8, [], [-128])

    out = helper.make_tensor_value_info('out', TensorProto.INT8, [1, 1, 3, 3])

    node = onnx.helper.make_node(
        'QLinearConv',
        inputs=['X', '1', '2', '3', '4', '5', '6', '7'],
        outputs=['out'],
        pads=[0, 0, 0, 0],
    )
    return ([node], [x], [out],
            [sc_x, zero_pt_x, wt, sc_wt, zero_pt_wt, sc_y, zero_pt_y])


@onnx_test()
def qlinearconv_scale_1D_test():
    # https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html
    x = helper.make_tensor_value_info('X', TensorProto.UINT8, [1, 1, 5, 5])
    sc_x = helper.make_tensor('1', TensorProto.FLOAT, [],
                              [0.09411764705882353])
    zero_pt_x = helper.make_tensor('2', TensorProto.UINT8, [], [0])

    wt = helper.make_tensor(
        '3', TensorProto.UINT8, [2, 1, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    sc_wt = helper.make_tensor('4', TensorProto.FLOAT, [2], [1.0, 0.5])
    zero_pt_wt = helper.make_tensor('5', TensorProto.UINT8, [2], [0, 0])

    sc_y = helper.make_tensor('6', TensorProto.FLOAT, [], [0.6352941176470588])
    zero_pt_y = helper.make_tensor('7', TensorProto.INT8, [], [-128])

    out = helper.make_tensor_value_info('out', TensorProto.INT8, [1, 2, 3, 3])

    node = onnx.helper.make_node(
        'QLinearConv',
        inputs=['X', '1', '2', '3', '4', '5', '6', '7'],
        outputs=['out'],
        pads=[0, 0, 0, 0],
    )
    return ([node], [x], [out],
            [sc_x, zero_pt_x, wt, sc_wt, zero_pt_wt, sc_y, zero_pt_y])


@onnx_test()
def qlinearglobalavgpool_test():
    x = helper.make_tensor_value_info('X', TensorProto.UINT8, [1, 3, 4, 4])

    sc_x = helper.make_tensor('X_scale', TensorProto.FLOAT, [], [0.05])
    z_pt_x = helper.make_tensor('X_zero_point', TensorProto.UINT8, [], [128])

    y = helper.make_tensor_value_info('Y', TensorProto.UINT8, [1, 3, 1, 1])

    sc_y = helper.make_tensor('Y_scale', TensorProto.FLOAT, [], [0.025])
    z_pt_y = helper.make_tensor('Y_zero_point', TensorProto.UINT8, [], [64])

    n = onnx.helper.make_node(
        'QLinearGlobalAveragePool',
        inputs=['X', 'X_scale', 'X_zero_point', 'Y_scale', 'Y_zero_point'],
        outputs=['Y'],
        channels_last=0,
    )

    return ([n], [x], [y], [sc_x, z_pt_x, sc_y, z_pt_y])


@onnx_test()
def qlinearleakyrelu_test():
    x = helper.make_tensor_value_info('X', TensorProto.INT8, [64])
    sc_x = helper.make_tensor('X_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_x = helper.make_tensor('X_zero_point', TensorProto.INT8, [], [0])

    sc_y = helper.make_tensor('Y_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_y = helper.make_tensor('Y_zero_point', TensorProto.INT8, [], [10])

    y = helper.make_tensor_value_info('Y', TensorProto.INT8, [64])

    node = onnx.helper.make_node(
        'QLinearLeakyRelu',
        inputs=['X', 'X_scale', 'X_zero_point', 'Y_scale', 'Y_zero_point'],
        outputs=['Y'],
        alpha=1.1,
    )
    return ([node], [x], [y], [sc_x, zero_pt_x, sc_y, zero_pt_y])


def qlinearmatmul_1D_test():
    a = helper.make_tensor_value_info('A', TensorProto.UINT8, [8])
    sc_a = helper.make_tensor('A_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_a = helper.make_tensor('A_zero_point', TensorProto.UINT8, [], [0])

    b = helper.make_tensor_value_info('B', TensorProto.UINT8, [8])
    sc_b = helper.make_tensor('B_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_b = helper.make_tensor('B_zero_point', TensorProto.UINT8, [],
                                   [128])

    sc_c = helper.make_tensor('C_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_c = helper.make_tensor('C_zero_point', TensorProto.UINT8, [], [64])

    c = helper.make_tensor_value_info('C', TensorProto.UINT8, [1])

    node = onnx.helper.make_node(
        'QLinearMatMul',
        inputs=[
            'A', 'A_scale', 'A_zero_point', 'B', 'B_scale', 'B_zero_point',
            'C_scale', 'C_zero_point'
        ],
        outputs=['C'],
    )
    return ([node], [a, b], [c],
            [sc_a, zero_pt_a, sc_b, zero_pt_b, sc_c, zero_pt_c])


@onnx_test()
def qlinearmatmul_2D_test():
    a = helper.make_tensor_value_info('A', TensorProto.UINT8, [1, 8])
    sc_a = helper.make_tensor('A_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_a = helper.make_tensor('A_zero_point', TensorProto.UINT8, [], [0])

    b = helper.make_tensor_value_info('B', TensorProto.UINT8, [8, 1])
    sc_b = helper.make_tensor('B_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_b = helper.make_tensor('B_zero_point', TensorProto.UINT8, [],
                                   [128])

    sc_c = helper.make_tensor('C_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_c = helper.make_tensor('C_zero_point', TensorProto.UINT8, [], [64])

    c = helper.make_tensor_value_info('C', TensorProto.UINT8, [1, 1])

    node = onnx.helper.make_node(
        'QLinearMatMul',
        inputs=[
            'A', 'A_scale', 'A_zero_point', 'B', 'B_scale', 'B_zero_point',
            'C_scale', 'C_zero_point'
        ],
        outputs=['C'],
    )
    return ([node], [a, b], [c],
            [sc_a, zero_pt_a, sc_b, zero_pt_b, sc_c, zero_pt_c])


@onnx_test()
def qlinearmatmul_3D_test():
    a = helper.make_tensor_value_info('A', TensorProto.UINT8, [2, 2, 4])
    sc_a = helper.make_tensor('A_scale', TensorProto.FLOAT, [], [0.0066])
    zero_pt_a = helper.make_tensor('A_zero_point', TensorProto.UINT8, [],
                                   [113])

    b = helper.make_tensor_value_info('B', TensorProto.UINT8, [2, 4, 3])
    sc_b = helper.make_tensor('B_scale', TensorProto.FLOAT, [], [0.00705])
    zero_pt_b = helper.make_tensor('B_zero_point', TensorProto.UINT8, [],
                                   [114])

    sc_c = helper.make_tensor('C_scale', TensorProto.FLOAT, [], [0.0107])
    zero_pt_c = helper.make_tensor('C_zero_point', TensorProto.UINT8, [],
                                   [118])

    c = helper.make_tensor_value_info('C', TensorProto.UINT8, [2, 2, 3])

    node = onnx.helper.make_node(
        'QLinearMatMul',
        inputs=[
            'A', 'A_scale', 'A_zero_point', 'B', 'B_scale', 'B_zero_point',
            'C_scale', 'C_zero_point'
        ],
        outputs=['C'],
    )
    return ([node], [a, b], [c],
            [sc_a, zero_pt_a, sc_b, zero_pt_b, sc_c, zero_pt_c])


@onnx_test()
def matmulnbits_mm_test():
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT,
                                           [2, 16])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.UINT8,
                                           [4, 1, 8])
    scales = onnx.helper.make_tensor_value_info("scales",
                                                onnx.TensorProto.FLOAT, [4])
    zp = onnx.helper.make_tensor_value_info("zp", onnx.TensorProto.UINT8, [4])
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [2, 4])

    node = onnx.helper.make_node("MatMulNBits",
                                 inputs=["a", "b", "scales", "zp"],
                                 outputs=["c"],
                                 bits=4,
                                 block_size=16,
                                 K=16,
                                 N=4,
                                 domain='com.microsoft')
    return ([node], [a, b, scales, zp], [c])


@onnx_test()
def matmulnbits_mm2_test():
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT,
                                           [2, 33])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.UINT8,
                                           [2, 3, 8])
    scales = onnx.helper.make_tensor_value_info("scales",
                                                onnx.TensorProto.FLOAT, [6])
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node("MatMulNBits",
                                 inputs=["a", "b", "scales"],
                                 outputs=["c"],
                                 bits=4,
                                 block_size=16,
                                 K=33,
                                 N=2,
                                 domain='com.microsoft')
    return ([node], [a, b, scales], [c])


@onnx_test()
def matmulnbits_vm_test():
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [20])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.UINT8,
                                           [3, 2, 8])
    scales = onnx.helper.make_tensor_value_info("scales",
                                                onnx.TensorProto.FLOAT, [6])
    zp = onnx.helper.make_tensor_value_info("zp", onnx.TensorProto.UINT8, [3])
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [3])

    node = onnx.helper.make_node("MatMulNBits",
                                 inputs=["a", "b", "scales", "zp"],
                                 outputs=["c"],
                                 bits=4,
                                 block_size=16,
                                 K=20,
                                 N=3,
                                 domain='com.microsoft')
    return ([node], [a, b, scales, zp], [c])


@onnx_test()
def matmulnbits_bmm_test():
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT,
                                           [2, 3, 8])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.UINT8,
                                           [2, 1, 8])
    scales = onnx.helper.make_tensor_value_info("scales",
                                                onnx.TensorProto.FLOAT, [2])
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT,
                                           [2, 3, 2])

    node = onnx.helper.make_node("MatMulNBits",
                                 inputs=["a", "b", "scales"],
                                 outputs=["c"],
                                 bits=4,
                                 block_size=16,
                                 K=8,
                                 N=2,
                                 domain='com.microsoft')
    return ([node], [a, b, scales], [c])


def matmulnbits_negative_test(bits=4,
                              block_size=16,
                              a_dims=[2, 16],
                              b_dims=[4, 1, 8],
                              scales_dims=[4],
                              zp_dims=[4],
                              out_dims=[2, 4]):
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, a_dims)
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.UINT8, b_dims)
    scales = onnx.helper.make_tensor_value_info("scales",
                                                onnx.TensorProto.FLOAT,
                                                scales_dims)
    zp = onnx.helper.make_tensor_value_info("zp", onnx.TensorProto.UINT8,
                                            zp_dims)
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT,
                                           out_dims)

    node = onnx.helper.make_node("MatMulNBits",
                                 inputs=["a", "b", "scales", "zp"],
                                 outputs=["c"],
                                 bits=bits,
                                 block_size=block_size,
                                 K=16,
                                 N=4,
                                 domain='com.microsoft')
    return ([node], [a, b, scales, zp], [c])


@onnx_test()
def matmulnbits_invalid_bits_value_test():
    return matmulnbits_negative_test(bits=5)


@onnx_test()
def matmulnbits_block_size_too_small_test():
    return matmulnbits_negative_test(block_size=8)


@onnx_test()
def matmulnbits_block_size_not_power_of_two_test():
    return matmulnbits_negative_test(block_size=20)


@onnx_test()
def matmulnbits_invalid_b_dims_test():
    return matmulnbits_negative_test(b_dims=[4, 2, 8])


@onnx_test()
def matmulnbits_invalid_scales_dims_test():
    return matmulnbits_negative_test(scales_dims=[3])


@onnx_test()
def matmulnbits_invalid_zp_dims_test():
    return matmulnbits_negative_test(zp_dims=[5])


@onnx_test()
def qlinearmul_test():
    a = helper.make_tensor_value_info('A', TensorProto.UINT8, [64])
    sc_a = helper.make_tensor('A_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_a = helper.make_tensor('A_zero_point', TensorProto.UINT8, [], [0])

    b = helper.make_tensor_value_info('B', TensorProto.UINT8, [64])
    sc_b = helper.make_tensor('B_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_b = helper.make_tensor('B_zero_point', TensorProto.UINT8, [], [16])

    sc_c = helper.make_tensor('C_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_c = helper.make_tensor('C_zero_point', TensorProto.UINT8, [],
                                   [100])

    c = helper.make_tensor_value_info('C', TensorProto.UINT8, [64])

    node = onnx.helper.make_node(
        'QLinearMul',
        inputs=[
            'A', 'A_scale', 'A_zero_point', 'B', 'B_scale', 'B_zero_point',
            'C_scale', 'C_zero_point'
        ],
        outputs=['C'],
    )
    return ([node], [a, b], [c],
            [sc_a, zero_pt_a, sc_b, zero_pt_b, sc_c, zero_pt_c])


@onnx_test()
def qlinearmul_bcast_test():
    a = helper.make_tensor_value_info('A', TensorProto.INT8, [64])
    sc_a = helper.make_tensor('A_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_a = helper.make_tensor('A_zero_point', TensorProto.INT8, [], [0])

    b = helper.make_tensor_value_info('B', TensorProto.INT8, [1, 1, 64])
    sc_b = helper.make_tensor('B_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_b = helper.make_tensor('B_zero_point', TensorProto.INT8, [], [128])

    sc_c = helper.make_tensor('C_scale', TensorProto.FLOAT, [], [0.15])
    zero_pt_c = helper.make_tensor('C_zero_point', TensorProto.INT8, [], [32])

    c = helper.make_tensor_value_info('C', TensorProto.INT8, [1, 1, 64])

    node = onnx.helper.make_node(
        'QLinearMul',
        inputs=[
            'A', 'A_scale', 'A_zero_point', 'B', 'B_scale', 'B_zero_point',
            'C_scale', 'C_zero_point'
        ],
        outputs=['C'],
    )
    return ([node], [a, b], [c],
            [sc_a, zero_pt_a, sc_b, zero_pt_b, sc_c, zero_pt_c])


@onnx_test()
def qlinearsigmoid_test():
    x = helper.make_tensor_value_info('X', TensorProto.INT8, [64])
    sc_x = helper.make_tensor('X_scale', TensorProto.FLOAT, [], [0.05])
    zero_pt_x = helper.make_tensor('X_zero_point', TensorProto.INT8, [], [0])

    sc_y = helper.make_tensor('Y_scale', TensorProto.FLOAT, [], [0.0035])
    zero_pt_y = helper.make_tensor('Y_zero_point', TensorProto.INT8, [],
                                   [-128])

    y = helper.make_tensor_value_info('Y', TensorProto.INT8, [64])

    node = onnx.helper.make_node(
        'QLinearSigmoid',
        inputs=['X', 'X_scale', 'X_zero_point', 'Y_scale', 'Y_zero_point'],
        outputs=['Y'],
    )
    return ([node], [x], [y], [sc_x, zero_pt_x, sc_y, zero_pt_y])


@onnx_test()
def quantizelinear_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.INT8, [5])

    node = onnx.helper.make_node(
        'QuantizeLinear',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def quantizelinear_int32_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.INT32, [5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.INT8, [5])

    node = onnx.helper.make_node(
        'QuantizeLinear',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def quantizelinear_zero_point_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1])
    arg2 = helper.make_tensor_value_info('2', TensorProto.INT8, [1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.INT8, [5])

    node = onnx.helper.make_node(
        'QuantizeLinear',
        inputs=['0', '1', '2'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1, arg2], [arg_out])


def make_quantizelinear_axis_graph(axis):
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 1, 5, 1])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [5])
    arg2 = helper.make_tensor_value_info('2', TensorProto.INT8, [5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.INT8,
                                            [1, 1, 5, 1])

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['0', '1', '2'],
                                 outputs=['out'],
                                 axis=axis)

    return ([node], [arg0, arg1, arg2], [arg_out])


@onnx_test()
def quantizelinear_axis_test():
    return make_quantizelinear_axis_graph(2)


@onnx_test()
def quantizelinear_neg_axis_test():
    return make_quantizelinear_axis_graph(-2)


@onnx_test()
def randomnormal_test():
    dtype = 11
    mean = 10.0
    scale = 1.5
    seed = 0.0
    shape = [2, 3, 4]
    output = helper.make_tensor_value_info('output', TensorProto.DOUBLE,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomNormal',
                                 inputs=[],
                                 outputs=['output'],
                                 dtype=dtype,
                                 mean=mean,
                                 scale=scale,
                                 seed=seed,
                                 shape=shape)

    return ([node], [], [output])


@onnx_test()
def randomnormal_dtype_error_test():
    dtype = 6
    shape = [2, 3, 4]
    output = helper.make_tensor_value_info('output', TensorProto.INT32,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomNormal',
                                 inputs=[],
                                 outputs=['output'],
                                 dtype=dtype,
                                 shape=shape)

    return ([node], [], [output])


@onnx_test()
def randomnormal_generated_seed_test():
    sample_size = 10
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output = helper.make_tensor_value_info("output", TensorProto.INT32,
                                           [1, 10])

    node = onnx.helper.make_node('RandomNormal',
                                 inputs=['input'],
                                 sample_size=sample_size,
                                 outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def randomnormal_shape_error_test():
    dtype = 1
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomNormal',
                                 inputs=[],
                                 outputs=['output'],
                                 dtype=dtype)

    return ([node], [], [output])


@onnx_test()
def randomnormallike_test():
    dtype = 10
    mean = 10.0
    scale = 1.5
    seed = 0.0
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [2, 3, 4])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomNormalLike',
                                 inputs=['input'],
                                 outputs=['output'],
                                 dtype=dtype,
                                 mean=mean,
                                 scale=scale,
                                 seed=seed)

    return ([node], [input], [output])


@onnx_test()
def randomnormallike_type_error_test():
    seed = 0
    input = helper.make_tensor_value_info('input', TensorProto.INT32,
                                          [2, 3, 4])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomNormalLike',
                                 inputs=['input'],
                                 outputs=['output'],
                                 seed=seed)

    return ([node], [input], [output])


@onnx_test()
def randomuniform_test():
    dtype = 11
    high = 1.0
    low = 0.0
    seed = 0.0
    shape = [2, 3, 4]
    output = helper.make_tensor_value_info('output', TensorProto.DOUBLE,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomUniform',
                                 inputs=[],
                                 outputs=['output'],
                                 dtype=dtype,
                                 high=high,
                                 low=low,
                                 seed=seed,
                                 shape=shape)

    return ([node], [], [output])


@onnx_test()
def randomuniform_dtype_error_test():
    dtype = 6
    shape = [2, 3, 4]
    output = helper.make_tensor_value_info('output', TensorProto.INT32,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomUniform',
                                 inputs=[],
                                 outputs=['output'],
                                 dtype=dtype,
                                 shape=shape)

    return ([node], [], [output])


@onnx_test()
def randomuniform_generated_seed_test():
    sample_size = 10
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output = helper.make_tensor_value_info("output", TensorProto.INT32,
                                           [1, 10])

    node = onnx.helper.make_node('RandomUniform',
                                 inputs=['input'],
                                 sample_size=sample_size,
                                 outputs=['output'])

    return ([node], [input], [output])


@onnx_test()
def randomuniform_shape_error_test():
    dtype = 1
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomUniform',
                                 inputs=[],
                                 outputs=['output'],
                                 dtype=dtype)

    return ([node], [], [output])


@onnx_test()
def randomuniformlike_test():
    dtype = 10
    high = 10.0
    low = 1.0
    seed = 0.0
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [2, 3, 4])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomUniformLike',
                                 inputs=['input'],
                                 outputs=['output'],
                                 dtype=dtype,
                                 high=high,
                                 low=low,
                                 seed=seed)

    return ([node], [input], [output])


@onnx_test()
def randomuniformlike_type_error_test():
    seed = 0
    input = helper.make_tensor_value_info('input', TensorProto.INT32,
                                          [2, 3, 4])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [2, 3, 4])

    node = onnx.helper.make_node('RandomUniformLike',
                                 inputs=['input'],
                                 outputs=['output'],
                                 seed=seed)

    return ([node], [input], [output])


@onnx_test()
def range_test():

    start_val = np.array([10])
    limit_val = np.array([6])
    delta_val = np.array([-3])

    start_tensor = helper.make_tensor(name='start_val',
                                      data_type=TensorProto.INT64,
                                      dims=start_val.reshape(()).shape,
                                      vals=start_val.astype(np.int64))
    start = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['start'],
                                  value=start_tensor)

    limit_tensor = helper.make_tensor(name='limit_val',
                                      data_type=TensorProto.INT64,
                                      dims=limit_val.reshape(()).shape,
                                      vals=limit_val.astype(np.int64))
    limit = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['limit'],
                                  value=limit_tensor)

    delta_tensor = helper.make_tensor(name='delta_val',
                                      data_type=TensorProto.INT64,
                                      dims=delta_val.reshape(()).shape,
                                      vals=delta_val.astype(np.int64))
    delta = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['delta'],
                                  value=delta_tensor)

    node = onnx.helper.make_node('Range',
                                 inputs=['start', 'limit', 'delta'],
                                 outputs=['1'])

    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    return ([start, limit, delta, node], [], [y])


@onnx_test()
def range_float_test():

    start_val = np.array([2])
    limit_val = np.array([11])
    delta_val = np.array([2])

    start_tensor = helper.make_tensor(name='start_val',
                                      data_type=TensorProto.FLOAT,
                                      dims=start_val.reshape(()).shape,
                                      vals=start_val.astype(np.float))
    start = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['start'],
                                  value=start_tensor)

    limit_tensor = helper.make_tensor(name='limit_val',
                                      data_type=TensorProto.FLOAT,
                                      dims=limit_val.reshape(()).shape,
                                      vals=limit_val.astype(np.float))
    limit = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['limit'],
                                  value=limit_tensor)

    delta_tensor = helper.make_tensor(name='delta_val',
                                      data_type=TensorProto.FLOAT,
                                      dims=delta_val.reshape(()).shape,
                                      vals=delta_val.astype(np.float))
    delta = onnx.helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['delta'],
                                  value=delta_tensor)

    node = onnx.helper.make_node('Range',
                                 inputs=['start', 'limit', 'delta'],
                                 outputs=['1'])

    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    return ([start, limit, delta, node], [], [y])


@onnx_test()
def recip_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Reciprocal',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


def reduceop_variable_axes_test(op_name,
                                axes_len=1,
                                keepdims=1,
                                noop_with_empty_axes=0):
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [axes_len])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])

    node = onnx.helper.make_node(op_name,
                                 inputs=['x', 'axes'],
                                 outputs=['y'],
                                 keepdims=keepdims,
                                 noop_with_empty_axes=noop_with_empty_axes)

    return ([node], [x, axes], [y])


@onnx_test()
def reducel1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 6])
    axes = [-2]

    node = onnx.helper.make_node('ReduceL1',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def reducel1_dyn_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None])
    axes = [-2]

    node = onnx.helper.make_node('ReduceL1',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def reducel1_dyn_noaxes_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None])

    node = onnx.helper.make_node('ReduceL1',
                                 inputs=['x'],
                                 outputs=['y'],
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reducel2_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5])
    axes = [-1]

    node = onnx.helper.make_node('ReduceL2',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reduce_log_sum_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 1, 5, 6])
    axes = [-3]

    node = onnx.helper.make_node('ReduceLogSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test()
def reduce_log_sum_exp_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 5, 6])
    axes = [-4]

    node = onnx.helper.make_node('ReduceLogSumExp',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test()
def reducemax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 6])

    axes = [2]

    node = onnx.helper.make_node('ReduceMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reducemax_fp8_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT8E4M3FNUZ,
                                      [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT8E4M3FNUZ,
                                      [3, 4, 6])

    axes = [2]

    node = onnx.helper.make_node('ReduceMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def reducemax_dyn_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 4, 6])
    axes = [2]

    node = onnx.helper.make_node('ReduceMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reducemean_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    axes = [2, 3]

    node = onnx.helper.make_node('ReduceMean',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reducemean_keepdims_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])
    axes = [2]

    node = onnx.helper.make_node('ReduceMean',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test()
def reducemin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 1, 5, 1])
    axes = [1, 3]

    node = onnx.helper.make_node('ReduceMin',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test()
def reduceprod_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])
    axes = [2]

    node = onnx.helper.make_node('ReduceProd',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test()
def reducesum_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=[2],
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reducesum_fp8_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT8E4M3FNUZ,
                                      [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT8E4M3FNUZ,
                                      [3, 4, 1, 6])

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=[2],
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reducesum_empty_axes_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])
    axes = np.array([], dtype=np.int64)
    axes_tensor = helper.make_tensor(name="axes",
                                     data_type=TensorProto.INT64,
                                     dims=axes.shape,
                                     vals=axes.astype(np.int64))

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x', 'axes'],
                                 outputs=['y'],
                                 keepdims=0,
                                 noop_with_empty_axes=False)

    return ([node], [x], [y], [axes_tensor])


@onnx_test()
def reducesum_noop_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])
    axes = np.array([], dtype=np.int64)
    axes_tensor = helper.make_tensor(name="axes",
                                     data_type=TensorProto.INT64,
                                     dims=axes.shape,
                                     vals=axes.astype(np.int64))

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x', 'axes'],
                                 outputs=['y'],
                                 keepdims=0,
                                 noop_with_empty_axes=True)

    return ([node], [x], [y], [axes_tensor])


@onnx_test()
def reducesum_keepdims_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes = [2, 3]

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test()
def reducesum_multiaxis_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes = [2, 3]

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reducesum_variable_axes_test():
    return reduceop_variable_axes_test('ReduceSum')


@onnx_test()
def reducesum_variable_axes_noop_test():
    return reduceop_variable_axes_test('ReduceSum', noop_with_empty_axes=1)


@onnx_test()
def reducesum_variable_axes_keepdims_clear_test():
    return reduceop_variable_axes_test('ReduceSum', keepdims=0)


@onnx_test()
def reducesum_variable_dynamic_axes_test():
    return reduceop_variable_axes_test('ReduceSum', None)


@onnx_test()
def reducesum_variable_dynamic_axes_verify_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 2])
    axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None])

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x', 'axes'],
                                 outputs=['y'])

    return ([node], [x, axes], [y])


@onnx_test()
def reducesum_variable_dynamic_axes_noop_set_verify_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 2])
    axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None])

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x', 'axes'],
                                 outputs=['y'],
                                 noop_with_empty_axes=1)

    return ([node], [x, axes], [y])


@onnx_test()
def reducesum_square_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 6])
    axes = [-2]

    node = onnx.helper.make_node('ReduceSumSquare',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test()
def reshape_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 2, 3])
    x_shape = helper.make_tensor_value_info('1', TensorProto.INT64, [2])
    x_shape_list = [3, 8]
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 8])
    y2 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3, 8])

    node = onnx.helper.make_node('Reshape', inputs=['0', '1'], outputs=['2'])

    node2 = onnx.helper.make_node('Reshape',
                                  inputs=['0'],
                                  shape=x_shape_list,
                                  outputs=['3'])

    return ([node, node2], [x, x_shape], [y, y2],
            [helper.make_tensor('1', TensorProto.INT64, [2], [3, 8])])


@onnx_test()
def reshape_non_standard_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 3, 2])

    trans = helper.make_node(
        'Transpose',
        inputs=['x'],
        outputs=['trans_x'],
        perm=[0, 2, 1],
    )

    res = onnx.helper.make_node('Reshape',
                                inputs=['trans_x'],
                                outputs=['y'],
                                shape=[4, 3, 2])

    return ([trans, res], [x], [y])


@onnx_test()
def reshape_variable_input_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 2, 3])
    x_shape = helper.make_tensor_value_info('1', TensorProto.INT64, [2])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 8])
    node = onnx.helper.make_node('Reshape', inputs=['0', '1'], outputs=['2'])
    return ([node], [x, x_shape], [y])


@onnx_test()
def reshape_variable_input_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 2, 3])
    x_shape = helper.make_tensor_value_info('1', TensorProto.INT64, [2])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [None, 6])
    node = onnx.helper.make_node('Reshape', inputs=['0', '1'], outputs=['2'])
    return ([node], [x, x_shape], [y])


@onnx_test()
def resize_downsample_c_test():
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 1, 2])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='asymmetric',
                                 mode='nearest',
                                 nearest_mode='ceil')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_downsample_f_test():
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        coordinate_transformation_mode='align_corners',
        mode='nearest',
        nearest_mode='floor')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_downsample_f_dyn_test():
    # scales is a compile-time input
    scales = np.array([1.0, 1.0, 0.601, 0.601], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 1, 5, 9])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='asymmetric',
                                 mode='nearest',
                                 nearest_mode='floor')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_downsample_f_dyn2_test():
    # output shape is an input
    sizes = np.array([2, 1, 3, 5], dtype=np.int64)
    sizes_tensor = helper.make_tensor(name='sizes',
                                      data_type=TensorProto.INT64,
                                      dims=sizes.shape,
                                      vals=sizes.flatten().astype(np.int64))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 1, 5, 9])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', 'sizes', ''],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='asymmetric',
                                 mode='nearest',
                                 nearest_mode='floor')

    return ([node], [X], [Y], [sizes_tensor])


@onnx_test()
def resize_downsample_f_dyn3_test():
    # scales is a runtime input
    scalesX = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 1, 5, 9])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='asymmetric',
                                 mode='nearest',
                                 nearest_mode='floor')

    return ([node], [X, scalesX], [Y])


@onnx_test()
def resize_downsample_f_ref_test():
    #  Same as resize_downsample_f_dyn_test but with static input
    scales = np.array([1.0, 1.0, 0.601, 0.601], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 1, 5, 9])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='asymmetric',
                                 mode='nearest',
                                 nearest_mode='floor')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_downsample_f_ref2_test():
    # output shape is an input
    sizes = np.array([2, 1, 3, 5], dtype=np.int64)
    sizes_tensor = helper.make_tensor(name='sizes',
                                      data_type=TensorProto.INT64,
                                      dims=sizes.shape,
                                      vals=sizes.flatten().astype(np.int64))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 1, 5, 9])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', 'sizes', ''],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='asymmetric',
                                 mode='nearest',
                                 nearest_mode='floor')

    return ([node], [X], [Y], [sizes_tensor])


@onnx_test()
def resize_dyn_err1_test():
    scales = np.array([1.601], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='half_pixel',
                                 mode='nearest',
                                 nearest_mode='round_prefer_ceil')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_upsample_f_dyn_test():
    scales = np.array([1.0, 1.0, 1.601, 1.601], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 1, 3, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='half_pixel',
                                 mode='nearest',
                                 nearest_mode='round_prefer_ceil')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_no_scale_test():
    # node has no scales or shapes input
    scales = np.array([1.0, 1.0, 1.601, 1.601], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 1, 3, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', ''],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='half_pixel',
                                 mode='nearest',
                                 nearest_mode='round_prefer_ceil')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_downsample_linear_test():
    scales = np.array([1.0, 1.0, 0.6, 0.5], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 mode='linear')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_linear_non_const_test():
    # scales is a runtime input
    scalesX = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 mode='linear')

    return ([node], [X, scalesX], [Y])


@onnx_test()
def resize_nonstd_input_test():
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 4, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 1, 2])

    trn = onnx.helper.make_node('Transpose',
                                inputs=['X'],
                                outputs=['TX'],
                                perm=[0, 1, 3, 2])

    node = onnx.helper.make_node('Resize',
                                 inputs=['TX', '', 'scales'],
                                 outputs=['Y'],
                                 coordinate_transformation_mode='asymmetric',
                                 mode='nearest',
                                 nearest_mode='ceil')

    return ([trn, node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_outsize_test():
    out_lens = np.array([1, 1, 4, 6], dtype=np.int64)
    out_lens_tensor = helper.make_tensor(name='out_lens',
                                         data_type=TensorProto.INT64,
                                         dims=out_lens.shape,
                                         vals=out_lens.flatten().astype(
                                             np.int64))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 4, 6])

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'out_lens'],
        outputs=['Y'],
        coordinate_transformation_mode='tf_half_pixel_for_nn',
        mode='nearest',
        nearest_mode='round_prefer_floor')

    return ([node], [X], [Y], [out_lens_tensor])


@onnx_test()
def resize_upsample_linear_ac_test():
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    scales_tensor = helper.make_tensor(name='scales',
                                       data_type=TensorProto.FLOAT,
                                       dims=scales.shape,
                                       vals=scales.flatten().astype(
                                           np.float32))
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='align_corners')

    return ([node], [X], [Y], [scales_tensor])


@onnx_test()
def resize_upsample_linear_test():
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    scales_tensor = helper.make_tensor(name='scales',
                                       data_type=TensorProto.FLOAT,
                                       dims=scales.shape,
                                       vals=scales.flatten().astype(
                                           np.float32))
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 mode='linear')

    return ([node], [X], [Y], [scales_tensor])


@onnx_test()
def resize_upsample_linear_large_test():
    x = helper.make_tensor_value_info('X', TensorProto.FLOAT,
                                      [1, 1, 1024, 1024])
    s = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 2, 2])
    y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                      [1, 1, 2048, 2048])
    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 mode='linear')
    return ([node], [x], [y], [s])


@onnx_test()
def resize_upsample_pf_test():
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 4, 6])

    node = onnx.helper.make_node('Resize',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 mode='nearest')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def resize_upsample_pc_test():
    scales = np.array([1.0, 1.0, 2.0, 1.5], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 4, 6])

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode='nearest',
        exclude_outside=0,
        nearest_mode='round_prefer_ceil')

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def reversesequence_4D_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 2, 2])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x'],
        outputs=['y'],
        time_axis=0,
        batch_axis=1,
        sequence_lens=[2, 1],
    )
    return ([node], [x], [y])


@onnx_test()
def reversesequence_batch_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 4])
    seq_lens = np.array([1, 2, 3, 4])
    seq_lens_tensor = helper.make_tensor(
        name="sequence_lens",
        data_type=TensorProto.INT64,
        dims=seq_lens.shape,
        vals=seq_lens.astype(np.int64),
    )
    arg_seq_lens = helper.make_node(
        "Constant",
        inputs=[],
        outputs=['arg_seq_lens'],
        value=seq_lens_tensor,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x', 'arg_seq_lens'],
        outputs=['y'],
        time_axis=1,
        batch_axis=0,
    )
    return ([arg_seq_lens, node], [x], [y])


@onnx_test()
def reversesequence_batch_axis_err_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 4, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 4, 2])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x'],
        outputs=['y'],
        time_axis=0,
        batch_axis=2,
        sequence_lens=[4, 3, 2, 1],
    )
    return ([node], [x], [y])


@onnx_test()
def reversesequence_rank_err_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x'],
        outputs=['y'],
        sequence_lens=[4, 3, 2, 1],
    )
    return ([node], [x], [y])


@onnx_test()
def reversesequence_sequence_lens_shape_err_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x'],
        outputs=['y'],
        sequence_lens=[4, 3, 2],
    )
    return ([node], [x], [y])


@onnx_test()
def reversesequence_same_axis_err_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x'],
        outputs=['y'],
        time_axis=1,
        batch_axis=1,
        sequence_lens=[4, 3, 2, 1],
    )
    return ([node], [x], [y])


@onnx_test()
def reversesequence_time_axis_err_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 4, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 4, 2, 3])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x'],
        outputs=['y'],
        time_axis=3,
        batch_axis=0,
        sequence_lens=[4, 3, 2, 1],
    )
    return ([node], [x], [y])


@onnx_test()
def reversesequence_time_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x'],
        outputs=['y'],
        time_axis=0,
        batch_axis=1,
        sequence_lens=[4, 3, 2, 1],
    )
    return ([node], [x], [y])


@onnx_test()
def rnn_bi_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [2, 20, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [2, 20, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, 40])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 2, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 2, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 2, 20])

    node = onnx.helper.make_node(
        'RNN',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0'],
        outputs=['hs', 'output'],
        activations=['tanh', 'sigmoid'],
        clip=0,
        direction='bidirectional',
        hidden_size=20,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0], [hs, output])


@onnx_test()
def rnn_bi_1af_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [5, 3, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [2, 20, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [2, 20, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [5, 2, 3, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [2, 3, 20])

    node = onnx.helper.make_node('RNN',
                                 inputs=['seq', 'w', 'r'],
                                 outputs=['hs', 'output'],
                                 activations=['tanh'],
                                 clip=0,
                                 direction='bidirectional',
                                 hidden_size=20)

    return ([node], [seq, w, r], [hs, output])


@onnx_test()
def rnn_f_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 20, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 20, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 40])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node(
        'RNN',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0'],
        outputs=['hs', 'output'],
        activations=['tanh'],
        clip=0,
        direction='forward',
        hidden_size=20,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0], [hs, output])


@onnx_test()
def rnn_f_5arg_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 20, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 20, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 40])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node('RNN',
                                 inputs=['seq', 'w', 'r', 'bias', 'seq_len'],
                                 outputs=['hs', 'output'],
                                 activations=['tanh'],
                                 clip=0,
                                 direction='forward',
                                 hidden_size=20,
                                 layout=1)

    return ([node], [seq, w, r, bias, seq_len], [hs, output])


@onnx_test()
def rnn_f_default_af_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [5, 3, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 20, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 20, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [5, 1, 3, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [1, 3, 20])

    node = onnx.helper.make_node('RNN',
                                 inputs=['seq', 'w', 'r'],
                                 outputs=['hs', 'output'],
                                 clip=0,
                                 direction='forward',
                                 hidden_size=20)

    return ([node], [seq, w, r], [hs, output])


@onnx_test()
def rnn_r_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 20, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 20, 20])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 40])
    seq_len = helper.make_tensor_value_info('seq_len', TensorProto.INT32, [3])
    h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [3, 1, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node(
        'RNN',
        inputs=['seq', 'w', 'r', 'bias', 'seq_len', 'h0'],
        outputs=['hs', 'output'],
        activations=['tanh'],
        clip=0,
        direction='reverse',
        hidden_size=20,
        layout=1)

    return ([node], [seq, w, r, bias, seq_len, h0], [hs, output])


@onnx_test()
def rnn_r_3arg_layout_test():
    seq = helper.make_tensor_value_info('seq', TensorProto.FLOAT, [3, 5, 10])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 20, 10])
    r = helper.make_tensor_value_info('r', TensorProto.FLOAT, [1, 20, 20])

    hs = helper.make_tensor_value_info('hs', TensorProto.FLOAT, [3, 5, 1, 20])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [3, 1, 20])

    node = onnx.helper.make_node('RNN',
                                 inputs=['seq', 'w', 'r'],
                                 outputs=['hs', 'output'],
                                 activations=['tanh'],
                                 clip=0,
                                 direction='reverse',
                                 hidden_size=20,
                                 layout=1)

    return ([node], [seq, w, r], [hs, output])


@onnx_test()
def roialign_default_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 4, 7, 8])
    roi = helper.make_tensor_value_info('rois', TensorProto.FLOAT, [8, 4])
    bi = helper.make_tensor_value_info('batch_ind', TensorProto.INT64, [8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [8, 4, 1, 1])

    node = onnx.helper.make_node('RoiAlign',
                                 inputs=['x', 'rois', 'batch_ind'],
                                 outputs=['y'])

    return ([node], [x, roi, bi], [y])


@onnx_test()
def roialign_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 5, 4, 7])
    roi = helper.make_tensor_value_info('rois', TensorProto.FLOAT, [8, 4])
    bi = helper.make_tensor_value_info('batch_ind', TensorProto.INT64, [8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [8, 4, 5, 5])

    node = onnx.helper.make_node(
        'RoiAlign',
        inputs=['x', 'rois', 'batch_ind'],
        outputs=['y'],
        spatial_scale=2.0,
        output_height=5,
        output_width=5,
        sampling_ratio=3,
        mode="avg",
        coordinate_transformation_mode="output_half_pixel")

    return ([node], [x, roi, bi], [y])


@onnx_test()
def rotary_embedding_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_interleaved_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 3, 8])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32, [1])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [8, 2])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [8, 2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 3, 8])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=1,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_interleaved_large_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [2, 8, 24])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32, [1])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [16, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [16, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [2, 8, 24])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=1,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_dim_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 6])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32, [1])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [2, 2])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [2, 2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 6])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        rotary_embedding_dim=4,
        num_heads=1,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_packed_batching_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 3, 6])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 3])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [2, 2])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [2, 2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 3, 6])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        rotary_embedding_dim=4,
        num_heads=1,
        is_packed_batching=1,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_float_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                          [1, 3, 2, 6])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [1, 3, 2, 6])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_scale_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        scale=0.0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_num_heads_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        num_heads=0,
        rotary_embedding_dim=2,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_input_dims_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 1, 1, 1, 1])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 1])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [1, 1])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [1, 1])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 1, 1, 1])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_pos_ids_1_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 1, 1])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_pos_ids_2_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32, [2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_pos_ids_3_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [2, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 1])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_dim_size_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        num_heads=3,
        rotary_embedding_dim=8,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_cache_1_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 18])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [2, 3])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 18])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_cache_2_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 8])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    cos_cache = helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT16,
                                              [4, 4])
    sin_cache = helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT16,
                                              [4, 6])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 8])

    node = onnx.helper.make_node(
        'RotaryEmbedding',
        inputs=['input', 'pos_ids', 'cos_cache', 'sin_cache'],
        outputs=['output'],
        interleaved=0,
        domain="com.microsoft")

    return ([node], [input, pos_ids, cos_cache, sin_cache], [output])


@onnx_test()
def rotary_embedding_wrong_n_inputs_test():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16,
                                          [1, 2, 8])
    pos_ids = helper.make_tensor_value_info('pos_ids', TensorProto.INT32,
                                            [1, 2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16,
                                           [1, 2, 8])

    node = onnx.helper.make_node('RotaryEmbedding',
                                 inputs=['input', 'pos_ids'],
                                 outputs=['output'],
                                 interleaved=0,
                                 domain="com.microsoft")

    return ([node], [input, pos_ids], [output])


@onnx_test()
def round_half_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [4, 4])

    node = onnx.helper.make_node('Round', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def round_bf16_test():
    x = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [4, 4])

    node = onnx.helper.make_node('Round', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


def make_scatter_elements_test(reduction="none"):
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32,
                                      [2, 3, 4, 5])
    u = helper.make_tensor_value_info('update', TensorProto.FLOAT,
                                      [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5, 6])

    node = onnx.helper.make_node(
        'ScatterElements',
        reduction=reduction,
        inputs=['data', 'indices', 'update'],
        outputs=['y'],
        axis=-2,
    )

    return ([node], [x, i, u], [y])


@onnx_test()
def scatter_add_test():
    return make_scatter_elements_test("add")


@onnx_test()
def scatter_mul_test():
    return make_scatter_elements_test("mul")


@onnx_test()
def scatter_min_test():
    return make_scatter_elements_test("min")


@onnx_test()
def scatter_max_test():
    return make_scatter_elements_test("max")


@onnx_test()
def scatter_none_test():
    return make_scatter_elements_test()


@onnx_test()
def scatter_elements_invalid_reduction_test():
    return make_scatter_elements_test("invalid")


def make_scatternd_test(reduction="none"):
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 2, 2])
    indices = helper.make_tensor_value_info('indices', TensorProto.INT64,
                                            [2, 1, 2])
    updates = helper.make_tensor_value_info('updates', TensorProto.FLOAT,
                                            [2, 1, 2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [2, 2, 2])

    node = onnx.helper.make_node('ScatterND',
                                 inputs=['data', 'indices', 'updates'],
                                 outputs=['output'],
                                 reduction=reduction)

    return ([node], [data, indices, updates], [output])


@onnx_test()
def scatternd_add_test():
    return make_scatternd_test("add")


@onnx_test()
def scatternd_mul_test():
    return make_scatternd_test("mul")


@onnx_test()
def scatternd_max_test():
    return make_scatternd_test("max")


@onnx_test()
def scatternd_min_test():
    return make_scatternd_test("min")


@onnx_test()
def scatternd_test():
    return make_scatternd_test()


@onnx_test()
def scatternd_invalid_reduction_test():
    return make_scatternd_test("invalid")


@onnx_test()
def scatternd_dyn_test():
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT,
                                         [None, 2, 2])
    indices = helper.make_tensor_value_info('indices', TensorProto.INT64,
                                            [None, 1, 2])
    updates = helper.make_tensor_value_info('updates', TensorProto.FLOAT,
                                            [None, 1, 2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                           [None, 2, 2])

    node = onnx.helper.make_node('ScatterND',
                                 inputs=['data', 'indices', 'updates'],
                                 outputs=['output'])

    return ([node], [data, indices, updates], [output])


@onnx_test()
def selu_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [2, 3])

    node = onnx.helper.make_node('Selu',
                                 inputs=['x'],
                                 outputs=['y'],
                                 alpha=0.3,
                                 gamma=0.5)

    return ([node], [x], [y])


@onnx_test()
def shape_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def shape_dyn_test0():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [None, 4, None, None])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def shape_dyn_test1():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [None, 4, None, None])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])

    node = onnx.helper.make_node('Shape', inputs=['x'], outputs=['y'], start=2)

    return ([node], [x], [y])


@onnx_test()
def shape_dyn_test2():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [None, 4, None, None])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])

    node = onnx.helper.make_node('Shape',
                                 inputs=['x'],
                                 outputs=['y'],
                                 start=-2)

    return ([node], [x], [y])


@onnx_test()
def shape_dyn_test3():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [None, 4, None, None])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])

    node = onnx.helper.make_node('Shape',
                                 inputs=['x'],
                                 outputs=['y'],
                                 start=1,
                                 end=2)

    return ([node], [x], [y])


@onnx_test()
def shape_end_oob_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [None, 4, None, None])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])

    node = onnx.helper.make_node('Shape', inputs=['x'], outputs=['y'], end=5)

    return ([node], [x], [y])


@onnx_test()
def shape_start_oob_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [None, 4, None, None])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])

    node = onnx.helper.make_node('Shape',
                                 inputs=['x'],
                                 outputs=['y'],
                                 start=-6)

    return ([node], [x], [y])


@onnx_test()
def shape_end_less_start_error():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                      [None, 4, None, None])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])

    node = onnx.helper.make_node('Shape',
                                 inputs=['x'],
                                 outputs=['y'],
                                 start=3,
                                 end=1)

    return ([node], [x], [y])


@onnx_test()
def shape_gather_test():
    values = np.array([1])
    # value = helper.make_tensor_value_info('value', TensorProto.INT32, [1])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [7, 3, 10])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1])

    value_tensor = helper.make_tensor(name='const_tensor',
                                      data_type=TensorProto.INT32,
                                      dims=values.shape,
                                      vals=values.flatten().astype(int))

    node_const = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['value'],
        value=value_tensor,
    )

    node_shape = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'],
    )

    node_gather = helper.make_node(
        'Gather',
        inputs=['y', 'value'],
        outputs=['z'],
        axis=0,
    )

    return ([node_const, node_shape, node_gather], [x], [z])


@onnx_test()
def shrink_hard_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5])

    node = onnx.helper.make_node(
        "Shrink",
        inputs=["x"],
        outputs=["y"],
        lambd=1.5,
    )

    return ([node], [x], [y])


@onnx_test()
def shrink_soft_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5])

    node = onnx.helper.make_node(
        "Shrink",
        inputs=["x"],
        outputs=["y"],
        lambd=1.5,
        bias=1.5,
    )

    return ([node], [x], [y])


@onnx_test()
def shrink_verify_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [5])

    node = onnx.helper.make_node(
        "Shrink",
        inputs=["x"],
        outputs=["y"],
        lambd=-5.0,
        bias=1.0,
    )

    return ([node], [x], [y])


@onnx_test()
def shrink_verify2_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [5])

    node = onnx.helper.make_node(
        "Shrink",
        inputs=["x"],
        outputs=["y"],
        lambd=-6.0,
        bias=5.0,
    )

    return ([node], [x], [y])


@onnx_test()
def shrink_int8_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [3, 3])

    node = onnx.helper.make_node(
        "Shrink",
        inputs=["x"],
        outputs=["y"],
        lambd=1.5,
        bias=1.5,
    )

    return ([node], [x], [y])


@onnx_test()
def shrink_fp8_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT8E4M3FNUZ, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT8E4M3FNUZ, [3, 3])

    node = onnx.helper.make_node(
        "Shrink",
        inputs=["x"],
        outputs=["y"],
        lambd=1.5,
        bias=1.5,
    )

    return ([node], [x], [y])


@onnx_test()
def shrink_uint8_test():
    x = helper.make_tensor_value_info('x', TensorProto.UINT8, [3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, [3, 3])

    node = onnx.helper.make_node(
        "Shrink",
        inputs=["x"],
        outputs=["y"],
        lambd=5.0,
        bias=-4.5,
    )

    return ([node], [x], [y])


@onnx_test()
def sign_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [10, 5])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [10, 5])

    node = onnx.helper.make_node(
        'Sign',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def simplified_layer_normalization_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SimplifiedLayerNormalization',
        inputs=['x', 'scale'],
        outputs=['y'],
        axis=-1,
        epsilon=1e-5,
        stash_type=1,
    )

    return ([node], [x, scale], [y])


@onnx_test()
def simplified_layer_normalization_invalid_input_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 2, 2, 4])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 2, 2, 4])

    node = onnx.helper.make_node(
        'SimplifiedLayerNormalization',
        inputs=['x', 'scale'],
        outputs=['y'],
    )

    return ([node], [x, scale], [y])


@onnx_test()
def simplified_layer_normalization_invalid_n_args_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT16, [4])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT16,
                                         [1, 2, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SimplifiedLayerNormalization',
        inputs=['x', 'scale', 'bias'],
        outputs=['y'],
    )

    return ([node], [x, scale, bias], [y])


@onnx_test()
def sin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Sin',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def sin_fp8_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT8E4M3FNUZ, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT8E4M3FNUZ, [10])

    node = onnx.helper.make_node(
        'Sin',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def sinh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Sinh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def sinh_dynamic_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None])

    node = onnx.helper.make_node(
        'Sinh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def size_float_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1])
    node = onnx.helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )
    return ([node], [x], [y])


@onnx_test()
def size_half_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [3, 1])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1])
    node = onnx.helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )
    return ([node], [x], [y])


@onnx_test()
def size_bf16_test():
    x = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [3, 1])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1])
    node = onnx.helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )
    return ([node], [x], [y])


@onnx_test()
def size_int_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [8, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1])
    node = onnx.helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )
    return ([node], [x], [y])


@onnx_test()
def size_fp8_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT8E4M3FNUZ,
                                      [2, 5, 3])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1])
    node = onnx.helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )
    return ([node], [x], [y])


@onnx_test()
def size_verify_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 5, 3])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1])
    node = onnx.helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )
    return ([node], [x], [y])


@onnx_test()
def skip_simplified_layer_normalization_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipSimplifiedLayerNormalization',
        inputs=['x', 'skip', 'gamma'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip,
                     gamma], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_simplified_layer_normalization_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipSimplifiedLayerNormalization',
        inputs=['x', 'skip', 'gamma', 'bias'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip, gamma,
                     bias], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_simplified_layer_normalization_invalid_n_args_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node('SkipSimplifiedLayerNormalization',
                                 inputs=['x', 'skip'],
                                 outputs=['y'],
                                 epsilon=1e-5,
                                 domain="com.microsoft")

    return ([node], [x, skip], [y])


@onnx_test()
def skip_simplified_layer_normalization_invalid_input_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [2, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 2, 4])

    node = onnx.helper.make_node('SkipSimplifiedLayerNormalization',
                                 inputs=['x', 'skip', 'gamma'],
                                 outputs=['y'],
                                 epsilon=1e-5,
                                 domain="com.microsoft")

    return ([node], [x, skip, gamma], [y])


@onnx_test()
def skip_layer_normalization_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipLayerNormalization',
        inputs=['x', 'skip', 'gamma'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip,
                     gamma], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_layer_normalization_2d_skip_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16, [2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipLayerNormalization',
        inputs=['x', 'skip', 'gamma'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip,
                     gamma], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_layer_normalization_skip_batch_size_1_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [1, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipLayerNormalization',
        inputs=['x', 'skip', 'gamma'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip,
                     gamma], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_layer_normalization_beta_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    beta = helper.make_tensor_value_info('beta', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipLayerNormalization',
        inputs=['x', 'skip', 'gamma', 'beta'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip, gamma,
                     beta], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_layer_normalization_invalid_beta_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    beta = helper.make_tensor_value_info('beta', TensorProto.FLOAT16, [4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipLayerNormalization',
        inputs=['x', 'skip', 'gamma', 'beta'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip, gamma,
                     beta], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_layer_normalization_beta_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    beta = helper.make_tensor_value_info('beta', TensorProto.FLOAT16, [4])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT16, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipLayerNormalization',
        inputs=['x', 'skip', 'gamma', 'beta', 'bias'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip, gamma, beta,
                     bias], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_layer_normalization_invalid_bias_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [4])
    beta = helper.make_tensor_value_info('beta', TensorProto.FLOAT16, [4])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, 2, 1])
    inv_std_var = helper.make_tensor_value_info('inv_std_var',
                                                TensorProto.FLOAT, [2, 2, 1])
    input_skip_bias_sum = helper.make_tensor_value_info(
        'input_skip_bias_sum', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node(
        'SkipLayerNormalization',
        inputs=['x', 'skip', 'gamma', 'beta', 'bias'],
        outputs=['y', 'mean', 'inv_std_var', 'input_skip_bias_sum'],
        epsilon=1e-5,
        domain="com.microsoft")

    return ([node], [x, skip, gamma, beta,
                     bias], [y, mean, inv_std_var, input_skip_bias_sum])


@onnx_test()
def skip_layer_normalization_invalid_n_args_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 4])

    node = onnx.helper.make_node('SkipLayerNormalization',
                                 inputs=['x', 'skip'],
                                 outputs=['y'],
                                 epsilon=1e-5,
                                 domain="com.microsoft")

    return ([node], [x, skip], [y])


@onnx_test()
def skip_layer_normalization_invalid_input_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 2, 2, 4])
    skip = helper.make_tensor_value_info('skip', TensorProto.FLOAT16,
                                         [2, 2, 4])
    gamma = helper.make_tensor_value_info('gamma', TensorProto.FLOAT16, [2, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2, 2, 4])

    node = onnx.helper.make_node('SkipLayerNormalization',
                                 inputs=['x', 'skip', 'gamma'],
                                 outputs=['y'],
                                 epsilon=1e-5,
                                 domain="com.microsoft")

    return ([node], [x, skip, gamma], [y])


@onnx_test()
def slice_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('Slice',
                                 inputs=['0'],
                                 axes=[0, 1],
                                 starts=[1, 0],
                                 ends=[2, 2],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def slice_constant_test():
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 2])

    x_tensor = helper.make_tensor(name='x_tensor',
                                  data_type=TensorProto.FLOAT,
                                  dims=[3, 2],
                                  vals=[0, 1, 2, 3, 4, 5])

    x = onnx.helper.make_node('Constant',
                              inputs=[],
                              outputs=['x'],
                              value=x_tensor)

    node = onnx.helper.make_node('Slice',
                                 inputs=['x'],
                                 axes=[0, 1],
                                 starts=[1, 0],
                                 ends=[2, 2],
                                 outputs=['1'])

    return ([x, node], [], [y])


@onnx_test()
def slice_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, None, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, None, 2])

    node = onnx.helper.make_node('Slice',
                                 inputs=['0'],
                                 axes=[0],
                                 starts=[1],
                                 ends=[2],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test
def slice_step_dyn_test():
    # A slice command with non - default steps will have a "Step"
    # instruction added in parsing.
    step = np.array([2, 1])
    step_tensor = helper.make_tensor(name="step",
                                     data_type=TensorProto.INT32,
                                     dims=step.shape,
                                     vals=step.astype(int))
    arg_step = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_step'],
                                value=step_tensor)

    axis = np.array([-1, -2])
    axis_tensor = helper.make_tensor(name="axis",
                                     data_type=TensorProto.INT32,
                                     dims=axis.shape,
                                     vals=axis.astype(int))
    arg_axis = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_axis'],
                                value=axis_tensor)

    end = np.array([-1, -1])
    end_tensor = helper.make_tensor(name="end",
                                    data_type=TensorProto.INT32,
                                    dims=end.shape,
                                    vals=end.astype(int))
    arg_end = helper.make_node("Constant",
                               inputs=[],
                               outputs=['arg_end'],
                               value=end_tensor)

    start = np.array([-5, -3])
    start_tensor = helper.make_tensor(name="start",
                                      data_type=TensorProto.INT32,
                                      dims=start.shape,
                                      vals=start.astype(int))
    arg_start = helper.make_node("Constant",
                                 inputs=[],
                                 outputs=['arg_start'],
                                 value=start_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['0', 'arg_start', 'arg_end', 'arg_axis', 'arg_step'],
        outputs=['1'])

    return ([arg_step, arg_axis, arg_end, arg_start, node], [x], [y])


@onnx_test
def slice_reverse_dyn_test():
    # A slice command with negative step on any axis will have
    # a "Reverse" instruction added in parsing.

    step = np.array([-1, 1])
    step_tensor = helper.make_tensor(name="step",
                                     data_type=TensorProto.INT32,
                                     dims=step.shape,
                                     vals=step.astype(int))
    arg_step = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_step'],
                                value=step_tensor)

    axis = np.array([-1, -2])
    axis_tensor = helper.make_tensor(name="axis",
                                     data_type=TensorProto.INT32,
                                     dims=axis.shape,
                                     vals=axis.astype(int))
    arg_axis = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_axis'],
                                value=axis_tensor)

    end = np.array([-1, -1])
    end_tensor = helper.make_tensor(name="end",
                                    data_type=TensorProto.INT32,
                                    dims=end.shape,
                                    vals=end.astype(int))
    arg_end = helper.make_node("Constant",
                               inputs=[],
                               outputs=['arg_end'],
                               value=end_tensor)

    start = np.array([-5, -3])
    start_tensor = helper.make_tensor(name="start",
                                      data_type=TensorProto.INT32,
                                      dims=start.shape,
                                      vals=start.astype(int))
    arg_start = helper.make_node("Constant",
                                 inputs=[],
                                 outputs=['arg_start'],
                                 value=start_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['0', 'arg_start', 'arg_end', 'arg_axis', 'arg_step'],
        outputs=['1'])

    return ([arg_step, arg_axis, arg_end, arg_start, node], [x], [y])


@onnx_test()
def slice_3arg_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 5])
    start = np.array([0, 0])
    start_tensor = helper.make_tensor(name="start",
                                      data_type=TensorProto.INT32,
                                      dims=start.shape,
                                      vals=start.astype(int))

    arg_start = helper.make_node("Constant",
                                 inputs=[],
                                 outputs=['arg_start'],
                                 value=start_tensor)

    end = np.array([2, 5])
    end_tensor = helper.make_tensor(name="end",
                                    data_type=TensorProto.INT32,
                                    dims=end.shape,
                                    vals=end.astype(int))
    arg_end = helper.make_node("Constant",
                               inputs=[],
                               outputs=['arg_end'],
                               value=end_tensor)

    node = onnx.helper.make_node('Slice',
                                 inputs=['0', 'arg_start', 'arg_end'],
                                 outputs=['1'])

    return ([arg_start, arg_end, node], [x], [y])


@onnx_test()
def slice_5arg_test():
    step = np.array([1, 1])
    step_tensor = helper.make_tensor(name="step",
                                     data_type=TensorProto.INT32,
                                     dims=step.shape,
                                     vals=step.astype(int))
    arg_step = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_step'],
                                value=step_tensor)

    axis = np.array([-1, -2])
    axis_tensor = helper.make_tensor(name="axis",
                                     data_type=TensorProto.INT32,
                                     dims=axis.shape,
                                     vals=axis.astype(int))
    arg_axis = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_axis'],
                                value=axis_tensor)

    end = np.array([-1, -1])
    end_tensor = helper.make_tensor(name="end",
                                    data_type=TensorProto.INT32,
                                    dims=end.shape,
                                    vals=end.astype(int))
    arg_end = helper.make_node("Constant",
                               inputs=[],
                               outputs=['arg_end'],
                               value=end_tensor)

    start = np.array([-5, -3])
    start_tensor = helper.make_tensor(name="start",
                                      data_type=TensorProto.INT32,
                                      dims=start.shape,
                                      vals=start.astype(int))
    arg_start = helper.make_node("Constant",
                                 inputs=[],
                                 outputs=['arg_start'],
                                 value=start_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['0', 'arg_start', 'arg_end', 'arg_axis', 'arg_step'],
        outputs=['1'])

    return ([arg_step, arg_axis, arg_end, arg_start, node], [x], [y])


@onnx_test()
def slice_5arg_reverse_test():
    step = np.array([-1, 1])
    step_tensor = helper.make_tensor(name="step",
                                     data_type=TensorProto.INT32,
                                     dims=step.shape,
                                     vals=step.astype(int))
    arg_step = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_step'],
                                value=step_tensor)

    axis = np.array([-1, -2])
    axis_tensor = helper.make_tensor(name="axis",
                                     data_type=TensorProto.INT32,
                                     dims=axis.shape,
                                     vals=axis.astype(int))
    arg_axis = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_axis'],
                                value=axis_tensor)

    end = np.array([-5, -1])
    end_tensor = helper.make_tensor(name="end",
                                    data_type=TensorProto.INT32,
                                    dims=end.shape,
                                    vals=end.astype(int))
    arg_end = helper.make_node("Constant",
                               inputs=[],
                               outputs=['arg_end'],
                               value=end_tensor)

    start = np.array([-1, -3])
    start_tensor = helper.make_tensor(name="start",
                                      data_type=TensorProto.INT32,
                                      dims=start.shape,
                                      vals=start.astype(int))
    arg_start = helper.make_node("Constant",
                                 inputs=[],
                                 outputs=['arg_start'],
                                 value=start_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['0', 'arg_start', 'arg_end', 'arg_axis', 'arg_step'],
        outputs=['1'])

    return ([arg_step, arg_axis, arg_end, arg_start, node], [x], [y])


@onnx_test()
def slice_5arg_step_test():
    step = np.array([-2, 2])
    step_tensor = helper.make_tensor(name="step",
                                     data_type=TensorProto.INT32,
                                     dims=step.shape,
                                     vals=step.astype(int))
    arg_step = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_step'],
                                value=step_tensor)

    axis = np.array([-1, -2])
    axis_tensor = helper.make_tensor(name="axis",
                                     data_type=TensorProto.INT32,
                                     dims=axis.shape,
                                     vals=axis.astype(int))
    arg_axis = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_axis'],
                                value=axis_tensor)

    end = np.array([-5, -1])
    end_tensor = helper.make_tensor(name="end",
                                    data_type=TensorProto.INT32,
                                    dims=end.shape,
                                    vals=end.astype(int))
    arg_end = helper.make_node("Constant",
                               inputs=[],
                               outputs=['arg_end'],
                               value=end_tensor)

    start = np.array([-1, -3])
    start_tensor = helper.make_tensor(name="start",
                                      data_type=TensorProto.INT32,
                                      dims=start.shape,
                                      vals=start.astype(int))
    arg_start = helper.make_node("Constant",
                                 inputs=[],
                                 outputs=['arg_start'],
                                 value=start_tensor)

    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['0', 'arg_start', 'arg_end', 'arg_axis', 'arg_step'],
        outputs=['1'])

    return ([arg_step, arg_axis, arg_end, arg_start, node], [x], [y])


@onnx_test()
def slice_max_end_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [10, 20])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [9, 17])

    node = onnx.helper.make_node('Slice',
                                 inputs=['0'],
                                 axes=[0, 1],
                                 starts=[1, 2],
                                 ends=[3000000000, -1],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def slice_var_input_static0():
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2])
    starts = helper.make_tensor_value_info('starts', TensorProto.INT32, [2])
    ends = helper.make_tensor_value_info('ends', TensorProto.INT32, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('Slice',
                                 inputs=['data', 'starts', 'ends'],
                                 axes=[0, 1],
                                 outputs=['output'])

    return ([node], [data, starts, ends], [output])


@onnx_test()
def slice_var_input_static1():
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2])
    starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2])
    ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2])
    axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('Slice',
                                 inputs=['data', 'starts', 'ends', 'axes'],
                                 outputs=['output'])

    return ([node], [data, starts, ends, axes], [output])


@onnx_test()
def slice_var_input_dyn0():
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [None, 2])
    starts = helper.make_tensor_value_info('starts', TensorProto.INT32, [2])
    ends = helper.make_tensor_value_info('ends', TensorProto.INT32, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('Slice',
                                 inputs=['data', 'starts', 'ends'],
                                 axes=[0, 1],
                                 outputs=['output'])

    return ([node], [data, starts, ends], [output])


@onnx_test()
def slice_var_input_dyn1():
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [None, 2])
    starts = helper.make_tensor_value_info('starts', TensorProto.INT32, [2])
    ends = helper.make_tensor_value_info('ends', TensorProto.INT32, [2])
    axes = helper.make_tensor_value_info('axes', TensorProto.INT32, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('Slice',
                                 inputs=['data', 'starts', 'ends', 'axes'],
                                 outputs=['output'])

    return ([node], [data, starts, ends, axes], [output])


@onnx_test()
def slice_var_input_default_steps():
    step = np.array([1, 1])
    step_tensor = helper.make_tensor(name="step",
                                     data_type=TensorProto.INT64,
                                     dims=step.shape,
                                     vals=step.astype(int))
    arg_step = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_step'],
                                value=step_tensor)

    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [None, 2])
    starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2])
    ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2])
    axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['data', 'starts', 'ends', 'axes', 'arg_step'],
        outputs=['output'])

    return ([arg_step, node], [data, starts, ends, axes], [output])


@onnx_test()
def slice_var_input_steps_error():
    step = np.array([2, 1])
    step_tensor = helper.make_tensor(name="step",
                                     data_type=TensorProto.INT32,
                                     dims=step.shape,
                                     vals=step.astype(int))
    arg_step = helper.make_node("Constant",
                                inputs=[],
                                outputs=['arg_step'],
                                value=step_tensor)

    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2])
    starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2])
    ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2])
    axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['data', 'starts', 'ends', 'axes', 'arg_step'],
        outputs=['output'])

    return ([arg_step, node], [data, starts, ends, axes], [output])


@onnx_test()
def softmax_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3])

    node = onnx.helper.make_node('Softmax', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def softmax_nonstd_input_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [6, 8])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 4])

    node0 = onnx.helper.make_node('Slice',
                                  inputs=['0'],
                                  axes=[0, 1],
                                  starts=[1, 0],
                                  ends=[4, 4],
                                  outputs=['1'])

    node1 = onnx.helper.make_node('Softmax', inputs=['1'], outputs=['2'])

    return ([node0, node1], [x], [y])


@onnx_test()
def softmax_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 3, 4, 4])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 3, 4, 4])

    node = onnx.helper.make_node('Softmax', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def softmaxcrossentropyloss_score_dim_err_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [2])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_score_label_mismatch_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [1])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_score_label_wrong_k_dims_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 4, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [2])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_weight_wrong_dims_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    weight = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1", "3"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels, weight], [loss])


@onnx_test()
def softmaxcrossentropyloss_label_wrong_type_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT8, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    weight = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1", "3"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels, weight], [loss])


@onnx_test()
def softmaxcrossentropyloss_scores_wrong_type_test():
    scores = helper.make_tensor_value_info('0', TensorProto.INT32, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_weight_wrong_type_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    weight = helper.make_tensor_value_info('3', TensorProto.INT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1", "3"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels, weight], [loss])


@onnx_test()
def softmaxcrossentropyloss_weight_score_mismatch_valid_type_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    weight = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1", "3"],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels, weight], [loss])


@onnx_test()
def softmaxcrossentropyloss_invalid_reduction_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    weight = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["0", "1", "3"],
        outputs=["2"],
        reduction="BadReductionName",
    )

    return ([node], [scores, labels, weight], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_asym_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [3])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_double_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_half_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_bf16_test():
    scores = helper.make_tensor_value_info('0', TensorProto.BFLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="none",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_sum_reduction_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="sum",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_sum_reduction_double_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="sum",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_sum_reduction_half_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="sum",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_sum_reduction_bf16_test():
    scores = helper.make_tensor_value_info('0', TensorProto.BFLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="sum",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_mean_reduction_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="mean",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_mean_reduction_double_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="mean",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_mean_reduction_half_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="mean",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_mean_reduction_bf16_test():
    scores = helper.make_tensor_value_info('0', TensorProto.BFLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    loss = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
        ],
        outputs=["2"],
        reduction="mean",
    )

    return ([node], [scores, labels], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_out_bounds_ignore_idx_test(
):
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
        ignore_index=5,
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_neg_out_bounds_ignore_idx_test(
):
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
        ignore_index=-5,
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_asym_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [3])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_asym_class_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 3])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_ignore_idx_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
        ignore_index=2,
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_idx_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
        ignore_index=-2,
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_idx_test2():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
        ignore_index=-1,
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_double_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_no_reduction_half_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_sum_reduction_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="sum",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_sum_reduction_double_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="sum",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_sum_reduction_bf16_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="sum",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_mean_reduction_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_mean_reduction_double_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_2d_mean_reduction_half_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [4, 4])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_kdim_not_equal_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16,
                                           [4, 4, 2, 1])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 1, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_kd_mean_reduction_half_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_kd_mean_reduction_bf16_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.BFLOAT16,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.BFLOAT16, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_kd_sum_reduction_double_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="sum",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softmaxcrossentropyloss_kd_no_reduction_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_mean_reduction_half_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [4])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_mean_reduction_bf16_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.BFLOAT16,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.BFLOAT16, [4])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_mean_reduction_half_weighted_test2():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [2, 3, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [3])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT16, [2])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_mean_reduction_bf16_weighted_test2():
    scores = helper.make_tensor_value_info('0', TensorProto.BFLOAT16,
                                           [2, 3, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.BFLOAT16, [3])
    loss = helper.make_tensor_value_info('3', TensorProto.BFLOAT16, [2])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="mean",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_sum_reduction_double_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [4])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="sum",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_sum_reduction_double_weighted_test2():
    scores = helper.make_tensor_value_info('0', TensorProto.DOUBLE, [2, 3, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.DOUBLE, [3])
    loss = helper.make_tensor_value_info('3', TensorProto.DOUBLE, [1])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="sum",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_no_reduction_weighted_test():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                           [4, 4, 2, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [4, 2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [4])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def negativeloglikelihoodloss_kd_no_reduction_weighted_test2():
    scores = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 2])
    labels = helper.make_tensor_value_info('1', TensorProto.INT32, [2, 2])
    weights = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    loss = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=[
            "0",
            "1",
            "2",
        ],
        outputs=["3"],
        reduction="none",
    )

    return ([node], [scores, labels, weights], [loss])


@onnx_test()
def softsign_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5])

    node = onnx.helper.make_node('Softsign', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


def softplus_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5])

    node = onnx.helper.make_node('Softplus', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def softsign_nd_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [3, 4, 5])

    node = onnx.helper.make_node('Softsign', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


def softplus_nd_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [3, 4, 5])

    node = onnx.helper.make_node('Softplus', inputs=['x'], outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def split_minus_axis_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [10, 5])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [10, 5])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [10, 5])

    node = onnx.helper.make_node(
        'Split',
        inputs=['x'],
        outputs=['y1', 'y2', 'y3'],
        axis=-1,
    )

    return ([node], [x], [y1, y2, y3])


@onnx_test()
def split_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [10, 7])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [10, 4])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [10, 4])

    node = onnx.helper.make_node('Split',
                                 inputs=['x'],
                                 outputs=['y1', 'y2', 'y3'],
                                 axis=1,
                                 split=[7, 4, 4])

    return ([node], [x], [y1, y2, y3])


@onnx_test()
def split_test_default():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [5, 15])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [5, 15])

    node = onnx.helper.make_node(
        'Split',
        inputs=['x'],
        outputs=['y1', 'y2'],
    )

    return ([node], [x], [y1, y2])


@onnx_test()
def split_test_no_attribute():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [300, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [75, 15])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [75, 15])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [75, 15])
    y4 = helper.make_tensor_value_info('y4', TensorProto.FLOAT, [75, 15])

    split = np.ones(4) * 75
    split_tensor = helper.make_tensor(name="split",
                                      data_type=TensorProto.INT64,
                                      dims=split.shape,
                                      vals=split.astype(np.int64))
    const_node = helper.make_node("Constant",
                                  inputs=[],
                                  outputs=['split'],
                                  value=split_tensor)

    node = onnx.helper.make_node(
        'Split',
        inputs=['x', 'split'],
        outputs=['y1', 'y2', 'y3', 'y4'],
    )

    return ([const_node, node], [x], [y1, y2, y3, y4])


@onnx_test()
def split_test_uneven():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [12, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [3, 15])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [3, 15])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [3, 15])
    y4 = helper.make_tensor_value_info('y4', TensorProto.FLOAT, [3, 15])
    y5 = helper.make_tensor_value_info('y5', TensorProto.FLOAT, [0, 15])

    node = onnx.helper.make_node(
        'Split',
        inputs=['x'],
        outputs=['y1', 'y2', 'y3', 'y4', 'y5'],
    )

    return ([node], [x], [y1, y2, y3, y4, y5])


@onnx_test()
def split_test_uneven_num_outputs():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [11, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [3, 15])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [3, 15])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [3, 15])
    y4 = helper.make_tensor_value_info('y4', TensorProto.FLOAT, [2, 15])

    node = onnx.helper.make_node(
        'Split',
        inputs=['x'],
        outputs=['y1', 'y2', 'y3', 'y4'],
        num_outputs=4,
    )

    return ([node], [x], [y1, y2, y3, y4])


@onnx_test()
def split_test_no_attribute_invalid_split():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [300, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [75, 15])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [75, 15])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [75, 15])
    y4 = helper.make_tensor_value_info('y4', TensorProto.FLOAT, [75, 15])

    split = np.ones(4)
    split_tensor = helper.make_tensor(name="split",
                                      data_type=TensorProto.INT64,
                                      dims=split.shape,
                                      vals=split.astype(np.int64))
    const_node = helper.make_node("Constant",
                                  inputs=[],
                                  outputs=['split'],
                                  value=split_tensor)

    node = onnx.helper.make_node(
        'Split',
        inputs=['x', 'split'],
        outputs=['y1', 'y2', 'y3', 'y4'],
    )

    return ([const_node, node], [x], [y1, y2, y3, y4])


@onnx_test()
def split_test_invalid_split():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [10, 7])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [10, 4])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [10, 4])

    node = onnx.helper.make_node('Split',
                                 inputs=['x'],
                                 outputs=['y1', 'y2', 'y3'],
                                 axis=1,
                                 split=[1, 1, 1])

    return ([node], [x], [y1, y2, y3])


@onnx_test()
def split_test_no_attribute_invalid_input_split():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [10, 7])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [10, 4])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [10, 4])

    node = onnx.helper.make_node('Split',
                                 inputs=['x'],
                                 outputs=['y1', 'y2', 'y3'],
                                 axis=1,
                                 split=[])

    return ([node], [x], [y1, y2, y3])


@onnx_test()
def split_test_invalid_num_outputs():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [11, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [3, 15])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [3, 15])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [3, 15])
    y4 = helper.make_tensor_value_info('y4', TensorProto.FLOAT, [2, 15])

    node = onnx.helper.make_node(
        'Split',
        inputs=['x'],
        outputs=['y1', 'y2', 'y3', 'y4'],
        num_outputs=5,
    )

    return ([node], [x], [y1, y2, y3, y4])


@onnx_test()
def split_dyn_input_fixed_split_axis_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [None, 5])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [None, 5])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [None, 5])

    node = onnx.helper.make_node('Split',
                                 inputs=['x'],
                                 outputs=['y1', 'y2', 'y3'],
                                 axis=1)

    return ([node], [x], [y1, y2, y3])


@onnx_test()
def split_dyn_input_dyn_split_axis_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [None, 5])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [None, 5])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [None, 5])

    node = onnx.helper.make_node('Split',
                                 inputs=['x'],
                                 outputs=['y1', 'y2', 'y3'],
                                 axis=0)

    return ([node], [x], [y1, y2, y3])


@onnx_test()
def split_dyn_input_split_attr_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [None, 5])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [None, 5])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [None, 5])

    node = onnx.helper.make_node('Split',
                                 inputs=['x'],
                                 outputs=['y1', 'y2', 'y3'],
                                 axis=0,
                                 split=[7, 4, 4])

    return ([node], [x], [y1, y2, y3])


@onnx_test()
def split_dyn_input_split_input_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 15])
    y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [None, 5])
    y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [None, 5])
    y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [None, 5])

    split = np.ones(3) * 5
    split_tensor = helper.make_tensor(name="split",
                                      data_type=TensorProto.INT64,
                                      dims=split.shape,
                                      vals=split.astype(np.int64))
    const_node = helper.make_node("Constant",
                                  inputs=[],
                                  outputs=['split'],
                                  value=split_tensor)

    node = onnx.helper.make_node('Split',
                                 inputs=['x', 'split'],
                                 outputs=['y1', 'y2', 'y3'],
                                 axis=0)

    return ([const_node, node], [x], [y1, y2, y3])


@onnx_test()
def sqrt_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 15])

    node = onnx.helper.make_node(
        'Sqrt',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def sqrt_fp8_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT8E4M3FNUZ,
                                      [10, 15])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT8E4M3FNUZ,
                                      [10, 15])

    node = onnx.helper.make_node(
        'Sqrt',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def squeeze_axes_input_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1, 5, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])
    axes = np.array([1, 3], dtype=np.int64)
    axes_tensor = helper.make_tensor(name="axes",
                                     data_type=TensorProto.INT64,
                                     dims=axes.shape,
                                     vals=axes.astype(np.int64))

    node = onnx.helper.make_node('Squeeze',
                                 inputs=['x', 'axes'],
                                 outputs=['y'])

    return ([node], [x], [y], [axes_tensor])


@onnx_test()
def squeeze_empty_axes_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1, 5, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])
    axes = np.array([], dtype=np.int64)
    axes_tensor = helper.make_tensor(name="axes",
                                     data_type=TensorProto.INT64,
                                     dims=axes.shape,
                                     vals=axes.astype(np.int64))

    node = onnx.helper.make_node('Squeeze',
                                 inputs=['x', 'axes'],
                                 outputs=['y'])

    return ([node], [x], [y], [axes_tensor])


@onnx_test()
def squeeze_unsqueeze_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [1, 3, 1, 1, 2, 1])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                      [1, 1, 3, 1, 2, 1])

    node = onnx.helper.make_node('Squeeze',
                                 inputs=['0'],
                                 axes=[0, 2, 3, 5],
                                 outputs=['1'])

    node2 = onnx.helper.make_node('Unsqueeze',
                                  inputs=['1'],
                                  axes=[0, 1, 3, 5],
                                  outputs=['2'])

    return ([node, node2], [x], [y])


@onnx_test()
def squeeze_unsqueeze_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [1, None, 1, 1, None, 1])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                      [1, 1, None, 1, None, 1])

    node = onnx.helper.make_node('Squeeze',
                                 inputs=['0'],
                                 axes=[0, 2, 3, 5],
                                 outputs=['1'])

    node2 = onnx.helper.make_node('Unsqueeze',
                                  inputs=['1'],
                                  axes=[0, 1, 3, 5],
                                  outputs=['2'])

    return ([node, node2], [x], [y])


@onnx_test()
def sub_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', '1'],
        outputs=['out'],
        broadcast=1,
        axis=1,
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test()
def sub_scalar_test():
    values = np.array([1])
    arg_node = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                             [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    values_tensor = helper.make_tensor(name='const',
                                       data_type=TensorProto.FLOAT,
                                       dims=values.reshape(()).shape,
                                       vals=values.flatten().astype(float))

    arg_const = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['arg_const'],
        value=values_tensor,
    )

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', 'arg_const'],
        outputs=['out'],
    )

    return ([arg_const, node], [arg_node], [arg_out])


@onnx_test()
def sum_int_test():
    a = helper.make_tensor_value_info('0', TensorProto.INT16, [3])
    b = helper.make_tensor_value_info('1', TensorProto.UINT16, [3])
    c = helper.make_tensor_value_info('2', TensorProto.UINT32, [3])
    y = helper.make_tensor_value_info('3', TensorProto.UINT32, [3])

    cnode1 = onnx.helper.make_node('Cast', inputs=['0'], outputs=['c0'], to=12)

    cnode2 = onnx.helper.make_node('Cast', inputs=['1'], outputs=['c1'], to=12)

    node = onnx.helper.make_node(
        'Sum',
        inputs=['c0', 'c1', '2'],
        outputs=['3'],
    )

    return ([cnode1, cnode2, node], [a, b, c], [y])


@onnx_test()
def sum_test():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Sum',
        inputs=['0', '1', '2'],
        outputs=['3'],
    )

    return ([node], [a, b, c], [y])


@onnx_test()
def sum_type_test():
    valb = np.array([1, 0])
    t_bool = helper.make_tensor(name="bool",
                                data_type=TensorProto.BOOL,
                                dims=valb.shape,
                                vals=valb.astype(bool))

    val = np.array([1, 1])
    t_int8 = helper.make_tensor(name="int8",
                                data_type=TensorProto.INT8,
                                dims=val.shape,
                                vals=val.astype(np.int8))

    t_uint8 = helper.make_tensor(name="uint8",
                                 data_type=TensorProto.UINT8,
                                 dims=val.shape,
                                 vals=val.astype(np.uint8))

    t_uint16 = helper.make_tensor(name="uint16",
                                  data_type=TensorProto.UINT16,
                                  dims=val.shape,
                                  vals=val.astype(np.uint16))

    t_uint32 = helper.make_tensor(name="uint32",
                                  data_type=TensorProto.UINT32,
                                  dims=val.shape,
                                  vals=val.astype(np.uint32))

    t_uint64 = helper.make_tensor(name="uint64",
                                  data_type=TensorProto.UINT64,
                                  dims=val.shape,
                                  vals=val.astype(np.uint64))

    t_double = helper.make_tensor(name="double",
                                  data_type=TensorProto.DOUBLE,
                                  dims=val.shape,
                                  vals=val.astype(np.float64))

    valr = np.array([1.5, 2.0])
    t_raw = helper.make_tensor(name="raw",
                               data_type=TensorProto.DOUBLE,
                               dims=valr.shape,
                               vals=valr.tobytes(),
                               raw=True)

    n_bool = onnx.helper.make_node('Cast',
                                   inputs=['bool'],
                                   outputs=['o_bool'],
                                   to=11)

    n_int8 = onnx.helper.make_node('Cast',
                                   inputs=['int8'],
                                   outputs=['o_int8'],
                                   to=11)

    n_uint8 = onnx.helper.make_node('Cast',
                                    inputs=['uint8'],
                                    outputs=['o_uint8'],
                                    to=11)

    n_uint16 = onnx.helper.make_node('Cast',
                                     inputs=['uint16'],
                                     outputs=['o_uint16'],
                                     to=11)

    n_uint32 = onnx.helper.make_node('Cast',
                                     inputs=['uint32'],
                                     outputs=['o_uint32'],
                                     to=11)

    n_uint64 = onnx.helper.make_node('Cast',
                                     inputs=['uint64'],
                                     outputs=['o_uint64'],
                                     to=11)

    node = onnx.helper.make_node(
        'Sum',
        inputs=[
            'o_bool', 'o_int8', 'o_uint8', 'o_uint16', 'o_uint32', 'o_uint64',
            'double', 'raw'
        ],
        outputs=['out'],
    )

    y = helper.make_tensor_value_info('out', TensorProto.DOUBLE, [2])

    return ([n_bool, n_int8, n_uint8, n_uint16, n_uint32, n_uint64,
             node], [], [y], [
                 t_bool, t_int8, t_uint8, t_uint16, t_uint32, t_uint64,
                 t_double, t_raw
             ])


@onnx_test()
def tan_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Tan',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def tanh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'Tanh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test()
def thresholdedrelu_default_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 3])

    node = onnx.helper.make_node('ThresholdedRelu',
                                 inputs=['x'],
                                 outputs=['y'])

    return ([node], [x], [y])


@onnx_test()
def thresholdedrelu_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 3])
    alpha = 3.0

    node = onnx.helper.make_node('ThresholdedRelu',
                                 inputs=['x'],
                                 outputs=['y'],
                                 alpha=alpha)

    return ([node], [x], [y])


@onnx_test()
def thresholdedrelu_int_test():
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [2, 2, 3])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [2, 2, 3])
    alpha = 3.0

    node = onnx.helper.make_node('ThresholdedRelu',
                                 inputs=['x'],
                                 outputs=['y'],
                                 alpha=alpha)

    return ([node], [x], [y])


@onnx_test()
def tile_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 4])

    node = onnx.helper.make_node('Tile', inputs=['x', 'y'], outputs=['z'])

    return ([node], [x, y], [z],
            [helper.make_tensor('y', TensorProto.INT64, [2], [1, 2])])


@onnx_test()
def tile_test_3x2():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [6, 4])

    node = onnx.helper.make_node('Tile', inputs=['x', 'y'], outputs=['z'])

    return ([node], [x, y], [z],
            [helper.make_tensor('y', TensorProto.INT64, [2], [3, 2])])


@onnx_test()
def topk_attrk_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 5, 3, 2])
    val = helper.make_tensor_value_info('val', TensorProto.FLOAT, [2, 2, 3, 2])
    ind = helper.make_tensor_value_info('indices', TensorProto.INT64,
                                        [2, 2, 3, 2])

    node = onnx.helper.make_node('TopK',
                                 inputs=['data'],
                                 outputs=['val', 'indices'],
                                 k=2)
    return ([node], [x], [val, ind])


@onnx_test()
def topk_neg_axis_test():
    k = np.array([3])
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5, 6])
    val = helper.make_tensor_value_info('val', TensorProto.FLOAT, [3, 3, 5, 6])
    ind = helper.make_tensor_value_info('indices', TensorProto.INT64,
                                        [3, 3, 5, 6])

    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node('TopK',
                                 inputs=['data', 'k'],
                                 outputs=['val', 'indices'],
                                 axis=-2,
                                 sorted=0)
    return ([node], [x], [val, ind], [k_tensor])


@onnx_test()
def topk_test():
    k = np.array([4])
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 5, 3, 2])
    val = helper.make_tensor_value_info('val', TensorProto.FLOAT, [2, 4, 3, 2])
    ind = helper.make_tensor_value_info('indices', TensorProto.INT64,
                                        [2, 4, 3, 2])

    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node('TopK',
                                 inputs=['data', 'k'],
                                 outputs=['val', 'indices'],
                                 largest=0,
                                 axis=1)
    return ([node], [x], [val, ind], [k_tensor])


def transpose_default_perm_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 5, 2, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 2, 5, 1])

    node = onnx.helper.make_node(
        'Transpose',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def transpose_invalid_perm_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 4, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 2, 2])

    node = onnx.helper.make_node(
        'Transpose',
        perm=[0, 2, 1],
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def transpose_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 2, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 2, 2])

    node = onnx.helper.make_node(
        'Transpose',
        perm=[0, 3, 1, 2],
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test()
def transpose_dyn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [None, 2, 2, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [None, 3, 2, 2])

    node = onnx.helper.make_node(
        'Transpose',
        perm=[0, 3, 1, 2],
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test
def transpose_gather_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 5, 4, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32,
                                      [2, 4, 3, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT,
                                      [3, 2, 3, 4, 5, 4, 5, 6])

    td = onnx.helper.make_node(
        'Transpose',
        inputs=['data'],
        outputs=['tdata'],
        perm=[0, 2, 1, 3],
    )

    ti = onnx.helper.make_node('Transpose',
                               inputs=['indices'],
                               outputs=['tindices'],
                               perm=[0, 2, 1, 3])

    node = onnx.helper.make_node(
        'Gather',
        inputs=['tdata', 'tindices'],
        outputs=['y'],
        axis=1,
    )

    return ([td, ti, node], [x, i], [y])


@onnx_test()
def triu_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x'],
        outputs=['y'],
    )
    return ([node], [x], [y])


@onnx_test()
def triu_batch_diff_k_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 3])
    k = np.array([2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 3])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def tril_batch_diff_k_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 3])
    k = np.array([2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 3])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node('Trilu',
                                 inputs=['x', 'k'],
                                 outputs=['y'],
                                 upper=0)
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def tril_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])

    node = onnx.helper.make_node('Trilu', inputs=['x'], outputs=['y'], upper=0)
    return ([node], [x], [y])


@onnx_test()
def triu_neg_k_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    k = np.array([-1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node('Trilu', inputs=['x', 'k'], outputs=['y'])
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def tril_neg_k_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    k = np.array([-1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))
    node = onnx.helper.make_node('Trilu',
                                 inputs=['x', 'k'],
                                 outputs=['y'],
                                 upper=0)
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def triu_out_k_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    k = np.array([5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node('Trilu', inputs=['x', 'k'], outputs=['y'])
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def tril_out_k_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    k = np.array([5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))
    node = onnx.helper.make_node('Trilu',
                                 inputs=['x', 'k'],
                                 outputs=['y'],
                                 upper=0)
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def triu_row_one_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
    k = np.array([1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def tril_row_one_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
    k = np.array([1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])
    k_tensor = helper.make_tensor(name='k',
                                  data_type=TensorProto.INT64,
                                  dims=k.shape,
                                  vals=k.astype(np.int64))

    node = onnx.helper.make_node('Trilu',
                                 inputs=['x', 'k'],
                                 outputs=['y'],
                                 upper=0)
    return ([node], [x], [y], [k_tensor])


@onnx_test()
def undefined_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Identity', inputs=[''], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def unique_dynamic_sorted_test():
    x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [6])
    y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4])
    y_ind = helper.make_tensor_value_info('indices', TensorProto.INT64, [4])
    x_ind = helper.make_tensor_value_info('inverse_indices', TensorProto.INT64,
                                          [6])
    count = helper.make_tensor_value_info('counts', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        axis=0,
        sorted=1)
    return ([node], [x], [y, y_ind, x_ind, count])


@onnx_test()
def unique_dynamic_sorted_3D_test():
    x = helper.make_tensor_value_info('X', TensorProto.INT64, [4, 4, 4])
    y = helper.make_tensor_value_info('Y', TensorProto.INT64, [16])
    y_ind = helper.make_tensor_value_info('indices', TensorProto.INT64, [16])
    x_ind = helper.make_tensor_value_info('inverse_indices', TensorProto.INT64,
                                          [64])
    count = helper.make_tensor_value_info('counts', TensorProto.INT64, [16])

    node = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        sorted=1)
    return ([node], [x], [y, y_ind, x_ind, count])


@onnx_test()
def unique_dynamic_unsorted_test():
    x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [6])
    y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4])
    y_ind = helper.make_tensor_value_info('indices', TensorProto.INT64, [4])
    x_ind = helper.make_tensor_value_info('inverse_indices', TensorProto.INT64,
                                          [6])
    count = helper.make_tensor_value_info('counts', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        axis=0,
        sorted=0)
    return ([node], [x], [y, y_ind, x_ind, count])


@onnx_test()
def unique_sorted_test():
    x = helper.make_tensor('X', TensorProto.FLOAT, [6], [2, 1, 1, 3, 4, 3])

    y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4])
    y_ind = helper.make_tensor_value_info('indices', TensorProto.INT64, [4])
    x_ind = helper.make_tensor_value_info('inverse_indices', TensorProto.INT64,
                                          [6])
    count = helper.make_tensor_value_info('counts', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        axis=0,
        sorted=1)
    return ([node], [], [y, y_ind, x_ind, count], [x])


@onnx_test()
def unique_unsorted_test():
    x = helper.make_tensor('X', TensorProto.FLOAT, [6], [2, 1, 1, 3, 4, 3])

    y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4])
    y_ind = helper.make_tensor_value_info('indices', TensorProto.INT64, [4])
    x_ind = helper.make_tensor_value_info('inverse_indices', TensorProto.INT64,
                                          [6])
    count = helper.make_tensor_value_info('counts', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        axis=0,
        sorted=0)
    return ([node], [], [y, y_ind, x_ind, count], [x])


@onnx_test()
def unknown_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])

    helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    a = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Unknown', inputs=['0', '1'], outputs=['2'])

    node2 = onnx.helper.make_node('Unknown', inputs=['2'], outputs=['3'])

    return ([node, node2], [x, y], [a])


@onnx_test()
def unknown_aten_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])

    helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    a = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('ATen',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 operator='unknown')

    return ([node], [x, y], [a])


@onnx_test()
def upsample_linear_test():
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    scales_tensor = helper.make_tensor(name='scales',
                                       data_type=TensorProto.FLOAT,
                                       dims=scales.shape,
                                       vals=scales.flatten().astype(
                                           np.float32))
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node = onnx.helper.make_node('Upsample',
                                 inputs=['X', '', 'scales'],
                                 outputs=['Y'],
                                 mode='linear')

    return ([node], [X], [Y], [scales_tensor])


@onnx_test()
def upsample_test():
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scale_tensor = helper.make_tensor(name='scales',
                                      data_type=TensorProto.FLOAT,
                                      dims=scales.shape,
                                      vals=scales.flatten().astype(np.float32))

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 4, 6])

    node = onnx.helper.make_node(
        'Upsample',
        inputs=['X', 'scales'],
        outputs=['Y'],
        mode='nearest',
    )

    return ([node], [X], [Y], [scale_tensor])


@onnx_test()
def upsample_ver7_test():
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 4, 6])

    node = onnx.helper.make_node('Upsample',
                                 inputs=['X'],
                                 outputs=['Y'],
                                 mode='nearest',
                                 scales=[1.0, 1.0, 2.0, 3.0])

    return ([node], [X], [Y])


@onnx_test()
def variable_batch_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [None, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT,
                                      [None, 3, 16, 16])

    node = onnx.helper.make_node('Identity', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test()
def variable_batch_leq_zero_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [0, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [-1, 3, 16, 16])

    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [-1, 3, 16, 16])
    node = onnx.helper.make_node('Add', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test()
def where_test():
    c = helper.make_tensor_value_info('c', TensorProto.BOOL, [2])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 1, 2, 2])

    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 2, 2, 2])
    node = onnx.helper.make_node('Where',
                                 inputs=['c', 'x', 'y'],
                                 outputs=['z'])

    return ([node], [c, x, y], [z])


@onnx_test()
def where_dyn_test():
    c = helper.make_tensor_value_info('c', TensorProto.BOOL, [None, 2, 2])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 2, 2])

    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [None, 2, 2])
    node = onnx.helper.make_node('Where',
                                 inputs=['c', 'x', 'y'],
                                 outputs=['z'])

    return ([node], [c, x, y], [z])


@onnx_test()
def where_mixed_test():
    # mixture of static and dynamic input shapes is not supported
    c = helper.make_tensor_value_info('c', TensorProto.BOOL, [None, 2, 2])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 2, 2])

    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [None, 2, 2])
    node = onnx.helper.make_node('Where',
                                 inputs=['c', 'x', 'y'],
                                 outputs=['z'])

    return ([node], [c, x, y], [z])


def scan_test(scan_input_axes=[0, 0],
              scan_input_directions=[0, 0],
              scan_output_axes=[0, 0],
              scan_output_directions=[0, 0]):
    sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2, 2])
    scan_in1 = helper.make_tensor_value_info("scan_in1", TensorProto.FLOAT,
                                             [2, 2])
    scan_in2 = helper.make_tensor_value_info("scan_in2", TensorProto.FLOAT,
                                             [1])
    sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT,
                                            [2, 2])
    scan_out1 = helper.make_tensor_value_info("scan_out1", TensorProto.FLOAT,
                                              [2, 2])
    scan_out2 = helper.make_tensor_value_info("scan_out2", TensorProto.FLOAT,
                                              [2])
    add1 = helper.make_node("Add",
                            inputs=["sum_in", "scan_in1"],
                            outputs=["add1_out"])
    add2 = helper.make_node("Add",
                            inputs=["add1_out", "scan_in2"],
                            outputs=["sum_out"])
    id = helper.make_node("Identity",
                          inputs=["sum_out"],
                          outputs=["scan_out1"])
    reduce_sum = helper.make_node("ReduceSum",
                                  axes=[0],
                                  keepdims=0,
                                  inputs=["sum_out"],
                                  outputs=["scan_out2"])
    scan_body = helper.make_graph([add1, add2, id, reduce_sum], "scan_body",
                                  [sum_in, scan_in1, scan_in2],
                                  [sum_out, scan_out1, scan_out2])

    init_state = helper.make_tensor_value_info("init_state", TensorProto.FLOAT,
                                               [2, 2])
    scan_ins1_sh = [2, 2, 2]
    scan_ins1_sh[scan_input_axes[0]] = 3
    scan_ins1 = helper.make_tensor_value_info("scan_ins1", TensorProto.FLOAT,
                                              scan_ins1_sh)
    scan_ins2_sh = [1, 1]
    scan_ins2_sh[scan_input_axes[1]] = 3
    scan_ins2 = helper.make_tensor_value_info("scan_ins2", TensorProto.FLOAT,
                                              scan_ins2_sh)

    final_state = helper.make_tensor_value_info("final_state",
                                                TensorProto.FLOAT, [2, 2])
    scan_outs1_sh = [2, 2, 2]
    scan_outs1_sh[scan_output_axes[0]] = 3
    scan_outs1 = helper.make_tensor_value_info("scan_outs1", TensorProto.FLOAT,
                                               scan_outs1_sh)
    scan_outs2_sh = [2, 2]
    scan_outs2_sh[scan_output_axes[1]] = 3
    scan_outs2 = helper.make_tensor_value_info("scan_outs2", TensorProto.FLOAT,
                                               scan_outs2_sh)
    node = helper.make_node(
        "Scan",
        inputs=["init_state", "scan_ins1", "scan_ins2"],
        outputs=["final_state", "scan_outs1", "scan_outs2"],
        num_scan_inputs=2,
        scan_input_axes=scan_input_axes,
        scan_input_directions=scan_input_directions,
        scan_output_axes=scan_output_axes,
        scan_output_directions=scan_output_directions,
        body=scan_body,
    )

    return ([node], [init_state, scan_ins1,
                     scan_ins2], [final_state, scan_outs1, scan_outs2])


@onnx_test()
def scan_test1():
    return scan_test()


@onnx_test()
def scan_test2():
    return scan_test(scan_output_directions=[1, 0])


@onnx_test()
def scan_test3():
    return scan_test(scan_output_axes=[1, -1])


@onnx_test()
def scan_test4():
    return scan_test(scan_input_directions=[1, 0])


@onnx_test()
def scan_test5():
    return scan_test(scan_input_axes=[2, -1])


@onnx_test()
def scan_test6():
    return scan_test(scan_input_axes=[-2, 0],
                     scan_input_directions=[0, 1],
                     scan_output_directions=[1, 1],
                     scan_output_axes=[2, 1])


@onnx_test()
def scan_test7():
    sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2, 2])
    scan_in = helper.make_tensor_value_info("scan_in", TensorProto.FLOAT,
                                            [2, 2])
    sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT,
                                            [2, 2])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT,
                                             [2, 2])
    add1 = helper.make_node("Add",
                            inputs=["sum_in", "scan_in"],
                            outputs=["add1_out"])
    add2 = helper.make_node("Add",
                            inputs=["add1_out", "scan_in"],
                            outputs=["sum_out"])
    id = helper.make_node("Identity", inputs=["scan_in"], outputs=["scan_out"])

    scan_body = helper.make_graph([add1, add2, id], "scan_body",
                                  [sum_in, scan_in], [sum_out, scan_out])

    init_state = helper.make_tensor_value_info("init_state", TensorProto.FLOAT,
                                               [2, 2])
    scan_ins_sh = [3, 2, 2]
    scan_ins = helper.make_tensor_value_info("scan_ins", TensorProto.FLOAT,
                                             scan_ins_sh)

    final_state = helper.make_tensor_value_info("final_state",
                                                TensorProto.FLOAT, [2, 2])
    scan_outs_sh = [3, 2, 2]
    scan_outs = helper.make_tensor_value_info("scan_outs", TensorProto.FLOAT,
                                              scan_outs_sh)
    node = helper.make_node(
        "Scan",
        inputs=["init_state", "scan_ins"],
        outputs=["final_state", "scan_outs"],
        num_scan_inputs=1,
        scan_input_axes=[0],
        scan_input_directions=[0],
        scan_output_axes=[0],
        scan_output_directions=[0],
        body=scan_body,
    )

    return ([node], [init_state, scan_ins], [final_state, scan_outs])


def scan_negative_test(scan_input_axes=[0],
                       scan_input_directions=[0],
                       scan_output_axes=[0],
                       scan_output_directions=[0]):
    sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2, 2])
    scan_in = helper.make_tensor_value_info("scan_in", TensorProto.FLOAT,
                                            [2, 2])
    sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT,
                                            [2, 2])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT,
                                             [2, 2])
    add = helper.make_node("Add",
                           inputs=["sum_in", "scan_in"],
                           outputs=["sum_out"])
    id = helper.make_node("Identity", inputs=["sum_out"], outputs=["scan_out"])
    scan_body = helper.make_graph([add, id], "scan_body", [sum_in, scan_in],
                                  [sum_out, scan_out])

    init_state = helper.make_tensor_value_info("init_state", TensorProto.FLOAT,
                                               [2, 2])
    scan_ins = helper.make_tensor_value_info("scan_ins", TensorProto.FLOAT,
                                             [3, 2, 2])

    final_state = helper.make_tensor_value_info("final_state",
                                                TensorProto.FLOAT, [2, 2])
    scan_outs = helper.make_tensor_value_info("scan_outs", TensorProto.FLOAT,
                                              [3, 2, 2])
    node = helper.make_node(
        "Scan",
        inputs=["init_state", "scan_ins"],
        outputs=["final_state", "scan_outs"],
        num_scan_inputs=1,
        scan_input_axes=scan_input_axes,
        scan_input_directions=scan_input_directions,
        scan_output_axes=scan_output_axes,
        scan_output_directions=scan_output_directions,
        body=scan_body,
    )

    return ([node], [init_state, scan_ins], [final_state, scan_outs])


@onnx_test()
def scan_invalid_input_axes_len_test():
    return scan_negative_test(scan_input_axes=[0, 0])


@onnx_test()
def scan_invalid_input_dirs_len_test():
    return scan_negative_test(scan_input_directions=[0, 0])


@onnx_test()
def scan_invalid_output_axes_len_test():
    return scan_negative_test(scan_output_axes=[0, 0])


@onnx_test()
def scan_invalid_output_dirs_len_test():
    return scan_negative_test(scan_output_directions=[0, 0])


@onnx_test()
def scan_invalid_input_axes_vals_test():
    return scan_negative_test(scan_input_axes=[3])


@onnx_test()
def scan_invalid_input_dirs_vals_test():
    return scan_negative_test(scan_input_directions=[2])


@onnx_test()
def scan_invalid_output_axes_vals_test():
    return scan_negative_test(scan_output_axes=[-4])


@onnx_test()
def scan_invalid_output_dirs_vals_test():
    return scan_negative_test(scan_output_directions=[-1])


@onnx_test()
def scan_arg_count_mismatch_test():
    sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2, 2])
    scan_in1 = helper.make_tensor_value_info("scan_in1", TensorProto.FLOAT,
                                             [2, 2])
    sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT,
                                            [2, 2])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT,
                                             [2, 2])
    add = helper.make_node("Add",
                           inputs=["sum_in", "scan_in1"],
                           outputs=["sum_out"])
    id = helper.make_node("Identity", inputs=["sum_out"], outputs=["scan_out"])
    scan_body = helper.make_graph([add, id], "scan_body", [sum_in, scan_in1],
                                  [sum_out, scan_out])

    init_state = helper.make_tensor_value_info("init_state", TensorProto.FLOAT,
                                               [2, 2])
    scan_ins1 = helper.make_tensor_value_info("scan_ins1", TensorProto.FLOAT,
                                              [3, 2, 2])
    scan_ins2 = helper.make_tensor_value_info("scan_ins2", TensorProto.FLOAT,
                                              [2, 3, 2])

    final_state = helper.make_tensor_value_info("final_state",
                                                TensorProto.FLOAT, [2, 2])
    scan_outs = helper.make_tensor_value_info("scan_outs", TensorProto.FLOAT,
                                              [3, 2, 2])
    node = helper.make_node(
        "Scan",
        inputs=["init_state", "scan_ins1", "scan_ins2"],
        outputs=["final_state", "scan_outs"],
        num_scan_inputs=2,
        scan_input_axes=[0, 0],
        scan_input_directions=[0, 0],
        scan_output_axes=[0],
        scan_output_directions=[0],
        body=scan_body,
    )
    return ([node], [init_state, scan_ins1,
                     scan_ins2], [final_state, scan_outs])


@onnx_test()
def scan_input_axes_lens_mismatch_test():
    sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2, 2])
    scan_in1 = helper.make_tensor_value_info("scan_in1", TensorProto.FLOAT,
                                             [2, 2])
    scan_in2 = helper.make_tensor_value_info("scan_in2", TensorProto.FLOAT,
                                             [2, 2])
    sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT,
                                            [2, 2])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT,
                                             [2, 2])
    add = helper.make_node("Add",
                           inputs=["sum_in", "scan_in1"],
                           outputs=["sum_out"])
    id = helper.make_node("Identity", inputs=["sum_out"], outputs=["scan_out"])
    scan_body = helper.make_graph([add, id], "scan_body",
                                  [sum_in, scan_in1, scan_in2],
                                  [sum_out, scan_out])

    init_state = helper.make_tensor_value_info("init_state", TensorProto.FLOAT,
                                               [2, 2])
    scan_ins1 = helper.make_tensor_value_info("scan_ins1", TensorProto.FLOAT,
                                              [3, 2, 2])
    scan_ins2 = helper.make_tensor_value_info("scan_ins2", TensorProto.FLOAT,
                                              [2, 3, 2])

    final_state = helper.make_tensor_value_info("final_state",
                                                TensorProto.FLOAT, [2, 2])
    scan_outs = helper.make_tensor_value_info("scan_outs", TensorProto.FLOAT,
                                              [3, 2, 2])
    node = helper.make_node(
        "Scan",
        inputs=["init_state", "scan_ins1", "scan_ins2"],
        outputs=["final_state", "scan_outs"],
        num_scan_inputs=2,
        scan_input_axes=[0, 0],
        scan_input_directions=[0, 0],
        scan_output_axes=[0],
        scan_output_directions=[0],
        body=scan_body,
    )
    return ([node], [init_state, scan_ins1,
                     scan_ins2], [final_state, scan_outs])


@onnx_test()
def scan_arg_shapes_mismatch_test():
    sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2, 2])
    scan_in1 = helper.make_tensor_value_info("scan_in1", TensorProto.FLOAT,
                                             [2, 2])
    scan_in2 = helper.make_tensor_value_info("scan_in2", TensorProto.FLOAT,
                                             [2, 2])
    sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT,
                                            [2, 2])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT,
                                             [2, 2])
    add = helper.make_node("Add",
                           inputs=["sum_in", "scan_in1"],
                           outputs=["sum_out"])
    id = helper.make_node("Identity", inputs=["sum_out"], outputs=["scan_out"])
    scan_body = helper.make_graph([add, id], "scan_body",
                                  [sum_in, scan_in1, scan_in2],
                                  [sum_out, scan_out])

    init_state = helper.make_tensor_value_info("init_state", TensorProto.FLOAT,
                                               [2, 2])
    scan_ins1 = helper.make_tensor_value_info("scan_ins1", TensorProto.FLOAT,
                                              [3, 2, 2])
    scan_ins2 = helper.make_tensor_value_info("scan_ins2", TensorProto.FLOAT,
                                              [3, 2])

    final_state = helper.make_tensor_value_info("final_state",
                                                TensorProto.FLOAT, [2, 2])
    scan_outs = helper.make_tensor_value_info("scan_outs", TensorProto.FLOAT,
                                              [3, 2, 2])
    node = helper.make_node(
        "Scan",
        inputs=["init_state", "scan_ins1", "scan_ins2"],
        outputs=["final_state", "scan_outs"],
        num_scan_inputs=2,
        scan_input_axes=[0, 0],
        scan_input_directions=[0, 0],
        scan_output_axes=[0],
        scan_output_directions=[0],
        body=scan_body,
    )
    return ([node], [init_state, scan_ins1,
                     scan_ins2], [final_state, scan_outs])
