#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
import migraphx


def test_conv_relu():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"), offload_copy=False)
    print(p)
    params = {}

    for key, value in p.get_parameter_shapes().items():
        print("Parameter {} -> {}".format(key, value))
        params[key] = migraphx.to_gpu(migraphx.generate_argument(value))

    r = migraphx.from_gpu(p.run(params)[-1])
    print(r)


# TODO: placeholder until tuple shapes and arguments exposed
#def test_dyn_batch():
#    a = migraphx.shape.dynamic_dimension(1, 4, {2, 4})
#    b = migraphx.shape.dynamic_dimension(3, 3)
#    c = migraphx.shape.dynamic_dimension(32, 32)
#    dd_map = {"0": [a, b, c, c]}
#    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx",
#                            map_dyn_input_dims=dd_map)
#    print(p)
#    print("Compiling ...")
#    p.compile(migraphx.get_target("gpu"), offload_copy=False)
#
#    print(p)
#
#    def run_prog(batch_size):
#        params = {}
#        for key, value in p.get_parameter_shapes().items():
#            print("Parameter {} -> {}".format(key, value))
#            params[key] = migraphx.to_gpu(
#                migraphx.generate_argument(value.to_static(batch_size)))
#
#        print("before_output")
#        outputs = p.run(params)
#        print(outputs)
#        r = migraphx.from_gpu(p.run(params)[-1])
#        print(r)
#
#    run_prog(1)
#    run_prog(2)
#    run_prog(3)
#    run_prog(4)

test_conv_relu()
