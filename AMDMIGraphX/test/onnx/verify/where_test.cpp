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

TEST_CASE(where_test)
{
    migraphx::program p = read_onnx("where_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape c_shape{migraphx::shape::bool_type, {2}};
    std::vector<int8_t> c_data = {1, 0};

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 2, 2}};
    std::vector<float> x_data(8, 1.0f);

    migraphx::shape y_shape{migraphx::shape::float_type, {2, 1, 2, 2}};
    std::vector<float> y_data(8, 2.0f);

    migraphx::parameter_map pp;
    pp["c"] = migraphx::argument(c_shape, c_data.data());
    pp["x"] = migraphx::argument(x_shape, x_data.data());
    pp["y"] = migraphx::argument(y_shape, y_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
