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
#include <migraphx/op/argmax.hpp>
#include <migraphx/op/argmin.hpp>

struct test_hardmax_default : verify_program<test_hardmax_default>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm        = p.get_main_module();
        auto input_type = migraphx::shape::float_type;
        std::vector<std::size_t> input_lens{2, 1, 4, 1025};
        migraphx::shape data_shape{input_type, input_lens};
        auto input = mm->add_parameter("x", data_shape);

        auto indices   = mm->add_instruction(migraphx::make_op("argmax", {{"axis", -1}}), input);
        auto zero_data = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
        auto updates = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 1, 4, 1}}}),
            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));
        mm->add_instruction(
            migraphx::make_op("scatter_none", {{"axis", -1}}), zero_data, indices, updates);

        return p;
    }
};
