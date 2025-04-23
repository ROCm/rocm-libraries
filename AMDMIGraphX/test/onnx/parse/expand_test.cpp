/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <onnx_test.hpp>

TEST_CASE(expand_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {3, 1, 1});
    auto param = mm->add_parameter("x", s);
    migraphx::shape ss(migraphx::shape::int32_type, {4});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4, 5}));
    mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5}}}), param);

    auto prog = optimize_onnx("expand_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(expand_static_input_dyn_output_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {3, 1, 1});
    auto param = mm->add_parameter("x", s);
    migraphx::shape ss(migraphx::shape::int64_type, {4});
    auto dims = mm->add_parameter("dims", ss);
    mm->add_instruction(migraphx::make_op("broadcast_with_dims"), param, dims);

    auto prog = optimize_onnx("expand_static_input_dyn_output_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(expand_dyn_input_dyn_output_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {{3, 8}, {1, 1}, {1, 1}});
    auto param = mm->add_parameter("x", s);
    migraphx::shape ss(migraphx::shape::int64_type, {4});
    auto dims = mm->add_parameter("dims", ss);
    auto ret  = mm->add_instruction(migraphx::make_op("broadcast_with_dims"), param, dims);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {3, 8};
    auto prog                     = read_onnx("expand_dyn_input_dyn_output_test.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(expand_dyn_input_static_dims_throw)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {3, 8};
    EXPECT(test::throws([&] { read_onnx("expand_dyn_input_static_dims_throw.onnx", options); }));
}
