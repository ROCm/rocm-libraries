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

TEST_CASE(gather_elements_axis1_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", {migraphx::shape::float_type, {3, 4}});
    auto indices = mm->add_parameter("indices", {migraphx::shape::int32_type, {2, 3}});
    std::vector<int> ind_indices{0, 1, 2, 4, 5, 6};
    std::vector<int> ind_axis_indices{0, 1, 2, 0, 1, 2};
    migraphx::shape ind_s{migraphx::shape::int32_type, {2, 3}};
    auto l_data_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_indices.begin(), ind_indices.end()});
    auto l_ind_axis_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_axis_indices.begin(), ind_axis_indices.end()});
    auto l_stride = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {1}}, {1}});

    auto rsp_data    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
    auto lbst_stride = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", ind_s.lens()}}), l_stride);
    auto axis_delta = mm->add_instruction(migraphx::make_op("sub"), indices, l_ind_axis_indices);
    auto mul_delta  = mm->add_instruction(migraphx::make_op("mul"), axis_delta, lbst_stride);
    auto ind        = mm->add_instruction(migraphx::make_op("add"), l_data_indices, mul_delta);
    auto ret = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp_data, ind);
    mm->add_return({ret});

    auto prog = read_onnx("gather_elements_axis1_test.onnx");

    EXPECT(p == prog);
}
