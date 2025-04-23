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

struct test_pointwise_multi_out : verify_program<test_pointwise_multi_out>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto z1  = mm->add_parameter("z1", s);
        auto z2  = mm->add_parameter("z2", s);
        auto* pm = p.create_module("pointwise");
        {
            auto x1   = pm->add_parameter("x1", {migraphx::shape::float_type});
            auto x2   = pm->add_parameter("x2", {migraphx::shape::float_type});
            auto add  = pm->add_instruction(migraphx::make_op("add"), x1, x2);
            auto abs  = pm->add_instruction(migraphx::make_op("abs"), add);
            auto sqrt = pm->add_instruction(migraphx::make_op("sqrt"), abs);
            pm->add_return({add, sqrt});
        }
        pm->set_bypass();
        auto pw  = mm->add_instruction(migraphx::make_op("pointwise"), {z1, z2}, {pm});
        auto e0  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pw);
        auto e1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), pw);
        auto sub = mm->add_instruction(migraphx::make_op("sub"), e0, e1);
        mm->add_return({sub});
        return p;
    }
};
