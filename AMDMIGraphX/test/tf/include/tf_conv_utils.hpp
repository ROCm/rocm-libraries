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
 *
 */

#ifndef MIGRAPHX_GUARD_TEST_TF_TF_CONV_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_TF_TF_CONV_UTILS_HPP

inline migraphx::program create_conv()
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 32);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 32}}, weight_data);

    migraphx::op::convolution op;
    op.padding  = {1, 1, 1, 1};
    op.stride   = {1, 1};
    op.dilation = {1, 1};
    auto l2 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 0, 1}}}), l1);
    mm->add_instruction(op, l0, l2);
    return p;
}

#endif
