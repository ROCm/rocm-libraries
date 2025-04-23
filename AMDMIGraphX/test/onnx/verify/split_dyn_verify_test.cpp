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

TEST_CASE(split_dyn_input_fixed_split_axis_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {10, 30};
    auto p = read_onnx("split_dyn_input_fixed_split_axis_test.onnx", options);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape data_shape{migraphx::shape::float_type, {10, 15}};
    std::vector<float> data(150, 1.23);
    migraphx::parameter_map pm;
    pm["x"]      = migraphx::argument(data_shape, data.data());
    auto results = p.eval(pm);
    std::vector<float> result_vector;
    std::vector<float> gold(50, 1.23);

    results.at(0).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));

    results.at(1).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));

    results.at(2).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(split_dyn_input_dyn_split_axis_test0)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {10, 30};
    auto p                        = read_onnx("split_dyn_input_dyn_split_axis_test.onnx", options);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape data_shape{migraphx::shape::float_type, {12, 15}};
    std::vector<float> data(180, 1.23);
    migraphx::parameter_map pm;
    pm["x"]      = migraphx::argument(data_shape, data.data());
    auto results = p.eval(pm);
    std::vector<float> result_vector;
    std::vector<float> gold(60, 1.23);

    results.at(0).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));

    results.at(1).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));

    results.at(2).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

// different static shape that doesn't split evenly
TEST_CASE(split_dyn_input_dyn_split_axis_test1)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {10, 30};
    auto p                        = read_onnx("split_dyn_input_dyn_split_axis_test.onnx", options);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape data_shape{migraphx::shape::float_type, {20, 15}};
    std::vector<float> data(300, 1.23);
    migraphx::parameter_map pm;
    pm["x"]      = migraphx::argument(data_shape, data.data());
    auto results = p.eval(pm);
    std::vector<float> result_vector;
    std::vector<float> gold_1(105, 1.23);

    results.at(0).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold_1));

    results.at(1).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold_1));

    std::vector<float> gold_2(90, 1.23);
    results.at(2).visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold_2));
}
