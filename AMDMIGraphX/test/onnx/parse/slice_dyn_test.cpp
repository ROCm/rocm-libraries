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

#include <onnx_test.hpp>

TEST_CASE(slice_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{3, 3}, {1, 3}, {2, 2}}});
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), l0);
    mm->add_return({ret});

    migraphx::onnx_options options;
    // Parser converts the dynamic input shape to static unless there is at least one non-fixed
    // dynamic dimension. Slicing is not allowed along the non-fixed axis 1.
    options.map_dyn_input_dims["0"] = {{3, 3}, {1, 3}, {2, 2}};
    auto prog                       = read_onnx("slice_dyn_test.onnx", options);

    EXPECT(p == prog);
}
