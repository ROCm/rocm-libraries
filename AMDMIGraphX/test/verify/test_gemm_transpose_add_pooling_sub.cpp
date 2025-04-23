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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/apply_alpha_beta.hpp>

template <migraphx::shape::type_t DType>
struct test_gemm_transpose_add_pooling_sub
    : verify_program<test_gemm_transpose_add_pooling_sub<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::shape s1{migraphx::shape::float_type, {1, 1, 2, 5}};
        migraphx::shape s2{migraphx::shape::float_type, {1, 1, 5, 10}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto dot       = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot_trans = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), dot);
        auto dot_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
        auto add = mm->add_instruction(migraphx::make_op("add"), {dot_rsp, x});
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {1, 0, 0, 0}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                add);
        auto sub = mm->add_instruction(migraphx::make_op("sub"), dot_rsp, pooling);
        mm->add_return({sub});
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_gemm_transpose_add_pooling_sub<migraphx::shape::float_type>;
template struct test_gemm_transpose_add_pooling_sub<migraphx::shape::half_type>;
template struct test_gemm_transpose_add_pooling_sub<migraphx::shape::bf16_type>;
template struct test_gemm_transpose_add_pooling_sub<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_gemm_transpose_add_pooling_sub<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_gemm_transpose_add_pooling_sub<migraphx::shape::fp8e4m3fn_type>;
template struct test_gemm_transpose_add_pooling_sub<migraphx::shape::fp8e5m2_type>;
