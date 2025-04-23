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
 *
 */

#include <tf_test.hpp>

TEST_CASE(batchnorm_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x    = mm->add_parameter("x", {migraphx::shape::half_type, {1, 32, 16, 16}});
    auto bias = mm->add_parameter("bias", {migraphx::shape::float_type, {32}});
    auto mean = mm->add_parameter("mean", {migraphx::shape::float_type, {32}});
    auto var  = mm->add_parameter("variance", {migraphx::shape::float_type, {32}});

    std::vector<float> scale_data(32, 1.0);
    auto scale = mm->add_literal(migraphx::shape{migraphx::shape::float_type, {32}}, scale_data);
    auto eps   = mm->add_literal(migraphx::literal{migraphx::shape::half_type, {1e-4f}});

    auto usq_scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
    auto usq_bias  = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", x->get_shape().lens()}}), bias);
    auto usq_mean = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", x->get_shape().lens()}}), mean);
    auto usq_var   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), var);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    auto prog = optimize_tf("batchnorm_half_test.pb", true);
    EXPECT(p.sort() == prog.sort());
}
