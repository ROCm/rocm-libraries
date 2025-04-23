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
#include <migraphx/op/pooling.hpp>

TEST_CASE(conv_bn_relu_maxpool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2  = mm->add_parameter("2", {migraphx::shape::float_type, {1}});

    auto p3 = mm->add_parameter("3", {migraphx::shape::float_type, {1}});
    auto p4 = mm->add_parameter("4", {migraphx::shape::float_type, {1}});
    auto p5 = mm->add_parameter("5", {migraphx::shape::float_type, {1}});
    auto p6 = mm->add_parameter("6", {migraphx::shape::float_type, {1}});

    auto eps = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {1e-5f}});

    uint64_t axis = 1;
    auto l3 =
        mm->add_instruction(migraphx::make_op("convolution", {{"padding", {0, 0, 0, 0}}}), l0, l1);
    auto l4 = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l3->get_shape().lens()}}), l2);
    auto l5 = mm->add_instruction(migraphx::make_op("add"), l3, l4);

    auto usq_scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), p3);
    auto usq_bias  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), p4);
    auto usq_mean  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), p5);
    auto usq_var   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), p6);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {l5, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    auto l6         = add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    auto l7 = mm->add_instruction(migraphx::make_op("relu"), l6);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"padding", {0, 0, 0, 0}},
                                           {"stride", {2, 2}},
                                           {"lengths", {2, 2}},
                                           {"dilations", {1, 1}}}),
                        l7);

    auto prog = optimize_onnx("conv_bn_relu_maxpool_test.onnx");
    EXPECT(p == prog);
}
