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

struct test_pointwise_broadcast_reduce : verify_program<test_pointwise_broadcast_reduce>
{
    migraphx::program create_program() const
    {
        migraphx::shape s{migraphx::shape::half_type, {2, 32, 96}};
        migraphx::shape rs{migraphx::shape::half_type, {2, 1, 1}};
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto x     = mm->add_parameter("x", rs);
        auto y     = mm->add_parameter("y", s);
        auto abs   = mm->add_instruction(migraphx::make_op("abs"), x);
        auto sqrt  = mm->add_instruction(migraphx::make_op("sqrt"), abs);
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto add   = mm->add_instruction(migraphx::make_op("add"), y, sqrtb);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), add);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto sub = mm->add_instruction(migraphx::make_op("sub"), rsumb, add);
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {s.elements()}}}), sub);
        mm->add_return({reshape});
        return p;
    };

    std::string section() const { return "reduce"; }
};
