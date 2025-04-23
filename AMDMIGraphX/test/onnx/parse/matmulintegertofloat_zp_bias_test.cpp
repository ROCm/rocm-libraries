/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(matmulintegertofloat_zp_bias_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto x0       = mm->add_parameter("1", migraphx::shape{migraphx::shape::int8_type, {4, 3}});
    auto x1       = mm->add_parameter("2", migraphx::shape{migraphx::shape::uint8_type, {3, 2}});
    auto scale_x0 = mm->add_parameter("3", migraphx::shape{migraphx::shape::float_type, {3}});
    auto scale_x1 = mm->add_parameter("4", migraphx::shape{migraphx::shape::float_type, {2}});
    auto zp_x0    = mm->add_parameter("5", migraphx::shape{migraphx::shape::int8_type, {3}});
    auto zp_x1    = mm->add_parameter("6", migraphx::shape{migraphx::shape::uint8_type, {2}});
    auto bias     = mm->add_parameter("7", migraphx::shape{migraphx::shape::float_type, {2}});

    auto sq_scale_x0 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), scale_x0);
    auto sq_scale_x1 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), scale_x1);

    auto sq_zp_x0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), zp_x0);
    auto sq_zp_x1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), zp_x1);

    auto bc_scale_x0 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x0->get_shape().lens()}}), sq_scale_x0);
    auto bc_zp_x0 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x0->get_shape().lens()}}), sq_zp_x0);

    auto r0 = mm->add_instruction(migraphx::make_op("dequantizelinear"), x0, bc_scale_x0, bc_zp_x0);

    auto bc_scale_x1 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x1->get_shape().lens()}}), sq_scale_x1);

    auto bc_zp_x1 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x1->get_shape().lens()}}), sq_zp_x1);

    auto r1 = mm->add_instruction(migraphx::make_op("dequantizelinear"), x1, bc_scale_x1, bc_zp_x1);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), r0, r1);

    auto mb_bias =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2}}}), bias);

    mm->add_instruction(migraphx::make_op("sub"), dot, mb_bias);

    auto prog = optimize_onnx("matmulintegertofloat_zp_bias_test.onnx");

    EXPECT(p == prog);
}
