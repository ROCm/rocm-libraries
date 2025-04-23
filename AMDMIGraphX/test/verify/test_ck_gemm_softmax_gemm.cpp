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
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_ck_gemm_softmax_gemm : verify_program<test_ck_gemm_softmax_gemm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::half_type, {1, 12, 256, 256}};
        migraphx::shape m2_shape{migraphx::shape::half_type, {1, 12, 256, 256}};
        auto m2_elements = m2_shape.elements();
        auto a           = mm->add_parameter("1", m1_shape);
        auto b           = mm->add_parameter("2", m1_shape);
        auto b1          = mm->add_parameter("3", m1_shape);
        std::vector<float> eights(m2_elements, 0.125);
        auto eight = mm->add_literal(migraphx::literal{m2_shape, eights});
        std::vector<float> zeros(m2_elements, 0);
        auto zero = mm->add_literal(migraphx::literal{m2_shape, zeros});

        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        auto gemm1   = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto scale   = mm->add_instruction(migraphx::make_op("mul"), gemm1, eight);
        auto bias    = mm->add_instruction(migraphx::make_op("add"), scale, zero);
        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), bias);
        mm->add_instruction(migraphx::make_op("dot"), softmax, b1);

        return p;
    }
    std::string section() const { return "gemm"; }
};
