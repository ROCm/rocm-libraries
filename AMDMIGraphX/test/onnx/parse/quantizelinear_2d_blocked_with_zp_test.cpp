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

#include "migraphx/make_op.hpp"
#include <onnx_test.hpp>

TEST_CASE(quantizelinear_2d_blocked_with_zp_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x     = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {6, 2}});
    auto scale = mm->add_parameter("scale", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    auto zp    = mm->add_parameter("zp", migraphx::shape{migraphx::shape::int8_type, {2, 2}});

    scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), scale);
    scale =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), scale);
    scale = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {6, 2}}}), scale);

    zp = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), zp);
    zp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {6, 2}}}), zp);

    mm->add_instruction(migraphx::make_op("quantizelinear"), x, scale, zp);

    auto prog = optimize_onnx("quantizelinear_2d_blocked_with_zp_test.onnx");
    EXPECT(p == prog);
}
