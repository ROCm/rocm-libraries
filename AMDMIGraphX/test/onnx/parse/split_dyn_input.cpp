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

TEST_CASE(split_dyn_input_fixed_split_axis_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {{10, 30}, {15, 15}}});
    auto r1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {5}}, {"ends", {10}}}), input);
    auto r3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {10}}, {"ends", {15}}}), input);
    mm->add_return({r1, r2, r3});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {10, 30};
    auto prog = read_onnx("split_dyn_input_fixed_split_axis_test.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(split_dyn_input_dyn_split_axis_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {{10, 30}, {15, 15}}});
    auto split_dim =
        mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), input);
    migraphx::shape int64_scalar_shape{migraphx::shape::int64_type, {1}, {0}};
    auto num_outputs_lit         = mm->add_literal(migraphx::literal{int64_scalar_shape, {3}});
    auto num_outputs_minus_1_lit = mm->add_literal(migraphx::literal{int64_scalar_shape, {2}});
    auto chunk_size              = mm->add_instruction(
        migraphx::make_op("div"),
        mm->add_instruction(migraphx::make_op("add"), split_dim, num_outputs_minus_1_lit),
        num_outputs_lit);
    auto r1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}}),
        input,
        mm->add_instruction(migraphx::make_op("mul"),
                            chunk_size,
                            mm->add_literal(migraphx::literal{int64_scalar_shape, {0}})),
        mm->add_instruction(migraphx::make_op("mul"),
                            chunk_size,
                            mm->add_literal(migraphx::literal{int64_scalar_shape, {1}})));
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}}),
        input,
        mm->add_instruction(migraphx::make_op("mul"),
                            chunk_size,
                            mm->add_literal(migraphx::literal{int64_scalar_shape, {1}})),
        mm->add_instruction(migraphx::make_op("mul"),
                            chunk_size,
                            mm->add_literal(migraphx::literal{int64_scalar_shape, {2}})));
    auto r3 = mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0}}, {"ends", {std::numeric_limits<int64_t>::max()}}}),
        input,
        mm->add_instruction(migraphx::make_op("mul"),
                            chunk_size,
                            mm->add_literal(migraphx::literal{int64_scalar_shape, {2}})));
    mm->add_return({r1, r2, r3});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {10, 30};
    auto prog                     = read_onnx("split_dyn_input_dyn_split_axis_test.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(split_dyn_input_split_attr_error)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {10, 30};
    EXPECT(test::throws([&] { read_onnx("split_dyn_input_split_attr_test.onnx", options); }));
}

TEST_CASE(split_dyn_input_split_input_error)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {10, 30};
    EXPECT(test::throws([&] { read_onnx("split_dyn_input_split_input_test.onnx", options); }));
}
