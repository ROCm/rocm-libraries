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
    # set offload_copy, fast_match to true
    p.compile(migraphx.get_target("gpu"), True, True)
    print(p)
    params = {}

    for key, value in p.get_parameter_shapes().items():
        print("Parameter {} -> {}".format(key, value))
        params[key] = migraphx.generate_argument(value)

    r = p.run(params)
    print(r)


def test_sub_uint64():
    p = migraphx.parse_onnx("implicit_sub_bcast_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    params = {}

    shapes = p.get_parameter_shapes()
    params["0"] = migraphx.create_argument(
        migraphx.shape(type='uint64_type', lens=shapes["0"].lens()),
        list(range(120)))
    params["1"] = migraphx.create_argument(
        migraphx.shape(type='uint64_type', lens=shapes["1"].lens()),
        list(range(20)))

    r = p.run(params)
    print(r)


def test_neg_int64():
    p = migraphx.parse_onnx("neg_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    params = {}

    shapes = p.get_parameter_shapes()
    params["0"] = migraphx.create_argument(
        migraphx.shape(type='int64_type', lens=shapes["0"].lens()),
        list(range(6)))

    r = p.run(params)
    print(r)


def test_nonzero():
    p = migraphx.parse_onnx("nonzero_dynamic_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    params = {}

    shapes = p.get_parameter_shapes()
    params["data"] = migraphx.create_argument(
        migraphx.shape(type='bool_type', lens=shapes["data"].lens()),
        [1, 1, 0, 1])

    r = p.run(params)
    print(r)


def test_fp16_imagescaler():
    p = migraphx.parse_onnx("imagescaler_half_test.onnx")
    print(p)
    s1 = p.get_output_shapes()[-1]
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2

    params = {}
    shapes = p.get_parameter_shapes()
    params["0"] = migraphx.generate_argument(
        migraphx.shape(type='half_type', lens=shapes["0"].lens()), 768)

    r = p.run(params)[-1]
    print(r)


def test_if_pl():
    p = migraphx.parse_onnx("if_pl_test.onnx")
    print(p)
    s1 = p.get_output_shapes()[-1]
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2

    params = {}
    shapes = p.get_parameter_shapes()
    params["x"] = migraphx.fill_argument(
        migraphx.shape(type='float_type', lens=shapes["x"].lens()), 1)
    params["y"] = migraphx.fill_argument(
        migraphx.shape(type='float_type', lens=shapes["y"].lens()), 2.0)
    params["cond"] = migraphx.fill_argument(
        migraphx.shape(type="bool", lens=[1], strides=[0]), 1)

    r = p.run(params)[-1]
    print(r)


def test_dyn_batch():
    a = migraphx.shape.dynamic_dimension(1, 4, {2, 4})
    b = migraphx.shape.dynamic_dimension(3, 3)
    c = migraphx.shape.dynamic_dimension(32, 32)
    dd_map = {"0": [a, b, c, c]}
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx",
                            map_dyn_input_dims=dd_map)
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)

    def run_prog(batch_size):
        params = {}
        for key, value in p.get_parameter_shapes().items():
            # convert to a static shape
            if value.dynamic():
                dds = value.dyn_dims()
                new_lens = []
                for dd in dds:
                    if dd.is_fixed():
                        new_lens.append(dd.min)
                    else:
                        new_lens.append(batch_size)
                s = migraphx.shape(type=value.type_string(), lens=new_lens)
            else:
                s = value
            print("Parameter {} -> {}".format(key, s))
            params[key] = migraphx.generate_argument(s)
        r = p.run(params)
        print(r)

    run_prog(1)
    run_prog(2)
    run_prog(3)
    run_prog(4)


test_conv_relu()
test_sub_uint64()
test_neg_int64()
test_fp16_imagescaler()
test_if_pl()
test_nonzero()
test_dyn_batch()
