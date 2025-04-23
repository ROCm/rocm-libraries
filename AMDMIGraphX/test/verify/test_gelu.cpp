/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

struct test_gelu : verify_program<test_gelu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<size_t> input_lens{1, 1, 5};
        auto x           = mm->add_parameter("x", {migraphx::shape::float_type, input_lens});
        auto half        = mm->add_literal(0.5f);
        auto one         = mm->add_literal(1.0f);
        auto sqrt2       = mm->add_literal(static_cast<float>(M_SQRT2));
        auto half_mbcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), half);
        auto mul_half     = mm->add_instruction(migraphx::make_op("mul"), x, half_mbcast);
        auto sqrt2_mbcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), sqrt2);
        auto div        = mm->add_instruction(migraphx::make_op("div"), x, sqrt2_mbcast);
        auto erf        = mm->add_instruction(migraphx::make_op("erf"), div);
        auto one_mbcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), one);
        auto add_one = mm->add_instruction(migraphx::make_op("add"), erf, one_mbcast);
        mm->add_instruction(migraphx::make_op("mul"), mul_half, add_one);
        return p;
    }
};
