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

struct test_concat_multi_out : verify_program<test_concat_multi_out>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        int axis = 0;
        migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::float_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::float_type, {5, 2}};
        auto x      = mm->add_parameter("x", s0);
        auto y      = mm->add_parameter("y", s1);
        auto z      = mm->add_parameter("z", s2);
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), x, y);
        auto relu   = mm->add_instruction(migraphx::make_op("relu"), concat);
        auto add    = mm->add_instruction(migraphx::make_op("add"), relu, z);
        mm->add_return({relu, add});
        return p;
    }
};
