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

struct test_concat_pooling : verify_program<test_concat_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 256, 8, 8}});
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), input);
        auto concat   = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), transpose);
        auto concat_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), concat);

        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::average},
                                                   {"padding", {0, 0}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {8, 8}},
                                                   {"dilations", {1, 1}}}),
                                concat_t);
        mm->add_instruction(migraphx::make_op("relu"), pooling);
        return p;
    }
};
