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
#include <migraphx/instruction.hpp>

struct test_dot_mul_b : verify_program<test_dot_mul_b>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape as{migraphx::shape::float_type, {2, 256, 32}};
        migraphx::shape bs{migraphx::shape::float_type, {2, 32, 128}};
        auto a   = mm->add_literal(migraphx::generate_literal(as));
        auto b   = mm->add_parameter("input", bs);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto lit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 256, 1}}));
        auto litb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot->get_shape().lens()}}), lit);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), dot, litb);
        mm->add_return({mul});
        return p;
    }
    std::string section() const { return "gemm"; }
};
