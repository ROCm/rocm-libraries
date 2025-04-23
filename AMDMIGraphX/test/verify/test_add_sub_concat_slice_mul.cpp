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
#include <migraphx/op/pooling.hpp>

// test for fuse_concat pass where pointwise input to concat has multiple outputs
struct test_add_sub_concat_slice_mul : verify_program<test_add_sub_concat_slice_mul>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{migraphx::shape::float_type, {1, 4, 8, 8}};
        auto x      = mm->add_parameter("x", s1);
        auto y      = mm->add_parameter("y", s1);
        auto add    = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto sub    = mm->add_instruction(migraphx::make_op("sub"), x, y);
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), add, sub);
        auto relu   = mm->add_instruction(migraphx::make_op("relu"), concat);
        auto slice  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {6}}}), relu);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), sub, slice);
        mm->add_return({mul});
        return p;
    }
};
