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

TEST_CASE(matmul_dyn_broadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto p0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto p1  = mm->add_parameter(
        "2", migraphx::shape{migraphx::shape::float_type, {{5, 5}, {7, 7}, {4, 8, {6}}}});
    auto usp0         = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), p0);
    auto broadcast_p0 = mm->add_instruction(migraphx::make_op("broadcast_for_dot"), usp0, p1);
    auto broadcast_p1 = mm->add_instruction(migraphx::make_op("broadcast_for_dot"), p1, usp0);
    auto dot_ins      = mm->add_instruction(migraphx::make_op("dot"), broadcast_p0, broadcast_p1);
    auto ret          = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), dot_ins);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["2"] = {{5, 5}, {7, 7}, {4, 8, {6}}};
    auto prog                       = read_onnx("matmul_dyn_broadcast_test.onnx", options);

    EXPECT(p == prog);
}
