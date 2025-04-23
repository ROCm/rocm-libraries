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
#include <migraphx/apply_alpha_beta.hpp>

TEST_CASE(gemm_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto l0    = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {8, 6}});
    auto l1    = mm->add_parameter("B", migraphx::shape{migraphx::shape::float_type, {8, 7}});
    auto l2    = mm->add_parameter("C", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto alpha = 0.5f;
    auto beta  = 0.8f;
    auto a_l   = mm->add_literal(alpha);
    auto t_a   = add_common_op(*mm, migraphx::make_op("mul"), {a_l, l0});
    t_a      = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t_a);
    auto dot = migraphx::add_apply_alpha_beta(*mm, {t_a, l1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto b_l = mm->add_literal(beta);
    auto b_b = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", l2->get_shape().lens()}}), b_l);
    auto l2_b = mm->add_instruction(migraphx::make_op("mul"), l2, b_b);
    mm->add_instruction(migraphx::make_op("add"), dot, l2_b);

    auto prog = optimize_onnx("gemm_test.onnx");
    EXPECT(p == prog);
}
