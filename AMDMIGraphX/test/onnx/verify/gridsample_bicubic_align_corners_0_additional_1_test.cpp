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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(gridsample_bicubic_align_corners_0_additional_1_test)
{
    migraphx::program p = read_onnx("gridsample_bicubic_align_corners_0_additional_1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {1, 1, 3, 2}};
    migraphx::shape grid_shape{input_type, {1, 2, 4, 2}};
    std::vector<float> data = {0., 1., 2., 3., 4., 5.};
    std::vector<float> grid = {
        -1., -0.8, -0.6, -0.5, -0.1, -0.2, 0.7, 0., 0., 0.4, 0.2, -0.2, -0.3, 0.5, -1., 1.};

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(data_shape, data.data());
    pp["grid"] = migraphx::argument(grid_shape, grid.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -0.173250, 0.284265, 1.923106, 2.568000, 5.170375, 2.284414, 4.744844, 1.046875};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
