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
struct test_transpose_reshape_add_sub_mul
    : verify_program<test_transpose_reshape_add_sub_mul<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::shape s1{migraphx::shape::float_type, {2, 10}};
        migraphx::shape s2{migraphx::shape::float_type, {5, 4}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x   = mm->add_parameter("x", s2);
        auto a_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), a);
        auto a_rsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {5, 4}}}), a_trans);
        auto add   = mm->add_instruction(migraphx::make_op("add"), {a_rsp, b});
        auto sub   = mm->add_instruction(migraphx::make_op("sub"), {b, x});
        auto mul   = mm->add_instruction(migraphx::make_op("mul"), add, sub);
        mm->add_return({mul});
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_transpose_reshape_add_sub_mul<migraphx::shape::float_type>;
template struct test_transpose_reshape_add_sub_mul<migraphx::shape::half_type>;
template struct test_transpose_reshape_add_sub_mul<migraphx::shape::bf16_type>;
template struct test_transpose_reshape_add_sub_mul<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_transpose_reshape_add_sub_mul<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_transpose_reshape_add_sub_mul<migraphx::shape::fp8e4m3fn_type>;
template struct test_transpose_reshape_add_sub_mul<migraphx::shape::fp8e5m2_type>;
