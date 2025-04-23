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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/float8.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <typename DType, typename CType>
struct test_batch_quant_dot_1 : verify_program<test_batch_quant_dot_1<DType, CType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::get_type<DType>{};
        auto ctype = migraphx::shape::get_type<CType>{};
        migraphx::shape m1_shape{dtype, {3, 2, 8, 2}};
        migraphx::shape m2_shape{dtype, {3, 2, 7, 8}};
        migraphx::shape m3_shape{ctype, {3, 2, 2, 7}};

        auto l1  = mm->add_parameter("a", m1_shape);
        auto tl1 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);
        auto l2  = mm->add_parameter("b", m2_shape);
        auto tl2 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l2);
        auto l3 = mm->add_parameter("c", m3_shape);
        migraphx::add_apply_alpha_beta(
            *mm, {tl1, tl2, l3}, migraphx::make_op("quant_dot"), CType{3}, CType{2});
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_batch_quant_dot_1<int8_t, int32_t>;
template struct test_batch_quant_dot_1<migraphx::fp8::fp8e4m3fnuz, float>;
template struct test_batch_quant_dot_1<migraphx::fp8::fp8e5m2fnuz, float>;
template struct test_batch_quant_dot_1<migraphx::fp8::fp8e4m3fn, float>;
template struct test_batch_quant_dot_1<migraphx::fp8::fp8e5m2, float>;
