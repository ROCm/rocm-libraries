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
#include <migraphx/op/reduce_mean.hpp>

/**
 * @brief test_shape_alloc sets up a situation that could lead to an exception "convolution: Shapes
 * are not in standard layout" if a "replace_allocate" compiler pass is not followed with
 *   "adjust_allocation".  The last transpose instruction generates a shape with a stride of 1 in
 *   the 2nd index, a non-standard layout that should be reallocated by adjust_allocation.
 */
struct test_shape_alloc : verify_program<test_shape_alloc>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto weights = mm->add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {11, 8, 1, 1}, {8, 1, 1, 1}}));

        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 8, 7, 7}});
        auto transpose1 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}),
                                x); //  -> float_type, {1, 7, 7, 8}, {392, 7, 1, 49}
        auto reduce_ins =
            mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}),
                                transpose1); //  -> float_type, {1, 1, 1, 8}, {8, 8, 8, 1}
        auto transpose2 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}),
                                reduce_ins); //  -> float_type, {1, 8, 1, 1}, {8, 1, 8, 8}
        auto conv_op = migraphx::make_op("convolution");
        mm->add_instruction(conv_op, transpose2, weights);

        return p;
    }
    std::string section() const { return "conv"; }
};
