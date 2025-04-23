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

TEST_CASE(gridsample_half_test)
{
    migraphx::program p = read_onnx("gridsample_half_test.onnx");
    migraphx::compile_options options;
    p.compile(migraphx::make_target("ref"));

    migraphx::shape data_shape{migraphx::shape::half_type, {1, 1, 4, 4}};
    migraphx::shape grid_shape{migraphx::shape::float_type, {1, 6, 6, 2}};
    std::vector<float> tmp = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};
    std::vector<migraphx::half> data = {tmp.begin(), tmp.end()};
    std::vector<float> grid = {-1., -1.,  -0.6, -1.,  -0.2, -1.,  0.2, -1.,  0.6, -1.,  1., -1.,
                               -1., -0.6, -0.6, -0.6, -0.2, -0.6, 0.2, -0.6, 0.6, -0.6, 1., -0.6,
                               -1., -0.2, -0.6, -0.2, -0.2, -0.2, 0.2, -0.2, 0.6, -0.2, 1., -0.2,
                               -1., 0.2,  -0.6, 0.2,  -0.2, 0.2,  0.2, 0.2,  0.6, 0.2,  1., 0.2,
                               -1., 0.6,  -0.6, 0.6,  -0.2, 0.6,  0.2, 0.6,  0.6, 0.6,  1., 0.6,
                               -1., 1.,   -0.6, 1.,   -0.2, 1.,   0.2, 1.,   0.6, 1.,   1., 1.};

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(data_shape, data.data());
    pp["grid"] = migraphx::argument(grid_shape, grid.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {0.,  0.15, 0.55, 0.95, 1.35, 0.75, 0.6, 1.5,  2.3,  3.1,  3.9,  2.1,
           2.2, 4.7,  5.5,  6.3,  7.1,  3.7,  3.8, 7.9,  8.7,  9.5,  10.3, 5.3,
           5.4, 11.1, 11.9, 12.7, 13.5, 6.9,  3.,  6.15, 6.55, 6.95, 7.35, 3.75};

    std::vector<migraphx::half> gold = {tmp.begin(), tmp.end()};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
