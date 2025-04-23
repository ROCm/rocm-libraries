/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>

TEST_CASE(lpnormalization_default_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<std::size_t> input_lens{3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape s{input_type, input_lens};
    auto x = mm->add_parameter("x", s);

    std::ptrdiff_t axis = 0;
    auto p_val          = mm->add_instruction(migraphx::make_op("mul"), x, x);
    auto norms = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {axis}}}), p_val);
    norms      = mm->add_instruction(migraphx::make_op("sqrt"), norms);
    norms =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), norms);
    auto zero_mb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0.}}));
    auto one_mb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1.}}));
    auto is_zero = mm->add_instruction(migraphx::make_op("equal"), norms, zero_mb);
    auto norms_zeros_to_one =
        mm->add_instruction(migraphx::make_op("where"), is_zero, one_mb, norms);
    mm->add_instruction(migraphx::make_op("div"), x, norms_zeros_to_one);

    auto prog = optimize_onnx("lpnormalization_default_test.onnx");
    EXPECT(p == prog);
}
