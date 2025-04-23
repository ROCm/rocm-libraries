/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(convinteger_bias_test)
{
    migraphx::program p;
    auto* mm       = p.get_main_module();
    auto data      = mm->add_parameter("0", {migraphx::shape::int8_type, {2, 3, 32, 32}});
    auto weights   = mm->add_parameter("1", {migraphx::shape::int8_type, {4, 3, 5, 5}});
    auto data_bias = mm->add_parameter("2", {migraphx::shape::int8_type, {1}, {1}});

    mm->add_literal(migraphx::literal{migraphx::shape{data->get_shape().type(), {1}, {0}}, {0}});
    auto quant = mm->add_instruction(migraphx::make_op("quant_convolution"), data, weights);

    auto bcast_data_bias = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", data->get_shape().lens()}}), data_bias);

    auto quant2 =
        mm->add_instruction(migraphx::make_op("quant_convolution"), bcast_data_bias, weights);

    mm->add_instruction(migraphx::make_op("sub"), quant, quant2);

    auto prog = optimize_onnx("convinteger_bias_test.onnx");
    EXPECT(p == prog);
}
