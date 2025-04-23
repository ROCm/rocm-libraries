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

TEST_CASE(convinteger_no_bias_uint8)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("0", {migraphx::shape::uint8_type, {1, 3, 32, 32}});
    auto weights = mm->add_parameter("1", {migraphx::shape::uint8_type, {1, 3, 5, 5}});

    mm->add_literal(migraphx::literal{migraphx::shape{data->get_shape().type(), {1}, {0}}, {128}});
    mm->add_literal(migraphx::literal{migraphx::shape{data->get_shape().type(), {1}, {0}}, {128}});

    // Shift uint8 input
    auto int8_shift2 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {-128}});

    // Shift uint8 input
    auto unshifted_input_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), data);

    auto mbr2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 3, 32, 32}}}), int8_shift2);

    auto input_shifted_half =
        mm->add_instruction(migraphx::make_op("add"), unshifted_input_half, mbr2);

    data = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
        input_shifted_half);

    // Shift uint8 weights
    auto unshifted_weights_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), weights);

    auto mbr3 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 3, 5, 5}}}), int8_shift2);

    auto weights_shifted_half =
        mm->add_instruction(migraphx::make_op("add"), unshifted_weights_half, mbr3);

    weights = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
        weights_shifted_half);

    mm->add_instruction(migraphx::make_op("quant_convolution"), data, weights);

    auto prog = optimize_onnx("convinteger_no_bias_uint8_test.onnx");
    EXPECT(p == prog);
}
