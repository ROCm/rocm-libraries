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
#include <migraphx/apply_alpha_beta.hpp>

TEST_CASE(gemm_fp8_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("A", migraphx::shape{migraphx::shape::fp8e4m3fnuz_type, {8, 6}});
    auto l1  = mm->add_parameter("B", migraphx::shape{migraphx::shape::fp8e4m3fnuz_type, {8, 7}});
    auto l2  = mm->add_parameter("C", migraphx::shape{migraphx::shape::fp8e4m3fnuz_type, {6, 1}});

    auto alpha = 0.5f;
    auto beta  = 0.8f;
    auto a_l   = mm->add_literal(alpha);
    auto t_a   = add_common_op(*mm, migraphx::make_op("mul"), {a_l, l0});
    t_a        = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::fp8e4m3fnuz_type}}), t_a);
    t_a = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t_a);
    std::vector<std::size_t> lens = {6, 7};
    auto dot                      = mm->add_instruction(migraphx::make_op("dot"), t_a, l1);
    l2        = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), l2);
    auto b_l  = mm->add_literal(beta);
    auto l2_b = add_common_op(*mm, migraphx::make_op("mul"), {l2, b_l});
    l2_b      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::fp8e4m3fnuz_type}}), l2_b);
    mm->add_instruction(migraphx::make_op("add"), dot, l2_b);

    auto prog = optimize_onnx("gemm_fp8_test.onnx");
    EXPECT(p == prog);
}
