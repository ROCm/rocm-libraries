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

struct test_reduce_mean_reduce_sum : verify_program<test_reduce_mean_reduce_sum>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {5, 784, 768}};
        auto x     = mm->add_parameter("x", s);
        auto n     = mm->add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::float_type, {1}}, {s.lens().back()}});
        auto mean  = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sub = mm->add_instruction(migraphx::make_op("sub"), x, meanb);
        auto nb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), n);
        auto div  = mm->add_instruction(migraphx::make_op("div"), sub, nb);
        auto div2 = mm->add_instruction(migraphx::make_op("mul"), div, div);
        auto mean_div2 =
            mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), div2);
        mm->add_return({mean_div2});
        return p;
    };

    std::string section() const { return "reduce"; }
};
