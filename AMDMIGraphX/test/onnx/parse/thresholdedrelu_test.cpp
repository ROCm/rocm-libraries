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

TEST_CASE(thresholdedrelu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2, 3}});
    auto lz  = mm->add_literal(migraphx::literal{migraphx::shape{x->get_shape().type()}, {0}});
    auto la  = mm->add_literal(migraphx::literal{migraphx::shape{x->get_shape().type()}, {3.0f}});
    auto mbz = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), lz);
    auto mba = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), la);
    auto condition = mm->add_instruction(migraphx::make_op("greater"), x, mba);
    mm->add_instruction(migraphx::make_op("where"), condition, x, mbz);

    auto prog = optimize_onnx("thresholdedrelu_test.onnx");

    EXPECT(p == prog);
}
