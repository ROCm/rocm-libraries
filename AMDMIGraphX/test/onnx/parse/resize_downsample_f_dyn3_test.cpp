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

TEST_CASE(resize_downsample_f_dyn3_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_instruction(migraphx::make_op("undefined"));
    migraphx::shape sx{migraphx::shape::float_type, {{1, 4, {1, 4}}, {1, 1}, {5, 5}, {9, 9}}};
    auto inx = mm->add_parameter("X", sx);
    auto li  = mm->add_parameter("scales", ss);
    auto r =
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"mode", "nearest"},
                                               {"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            inx,
                            li);

    mm->add_return({r});
    migraphx::onnx_options options;
    options.map_dyn_input_dims["X"] = {{1, 4, {1, 4}}, {1, 1}, {5, 5}, {9, 9}};

    auto prog = read_onnx("resize_downsample_f_dyn3_test.onnx", options);
    EXPECT(p == prog);
}
