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

#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(group_norm_contrib_silu_3d_test)
{
    migraphx::program p = make_group_norm({1, 4, 2},
                                          {2},
                                          {2},
                                          {1, 2, 2, 2},
                                          {2, 3},
                                          1e-5f,
                                          migraphx::shape::float_type,
                                          "gamma",
                                          "beta");

    // Add sigmoid at the end of the program to represent the added SILU block
    auto* mm     = p.get_main_module();
    auto output  = std::prev(mm->end());
    auto sigmoid = mm->add_instruction(migraphx::make_op("sigmoid"), output);
    output       = mm->add_instruction(migraphx::make_op("mul"), output, sigmoid);

    auto prog = optimize_onnx("group_norm_contrib_silu_3d_test.onnx");
    EXPECT(p == prog);
}
