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

TEST_CASE(gelu_add_bias_test)
{
    migraphx::program p = read_onnx("gelu_add_bias_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {3, 3}};
    migraphx::shape bias_shape{input_type, {3}};
    std::vector<float> data = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    std::vector<float> bias = {-1, 0, 1};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(data_shape, data.data());
    pp["y"] = migraphx::argument(bias_shape, bias.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> import math
    // >>> x = np.array([[-1, -1, -1],[0, 0, 0], [1, 1, 1]]).astype(np.float32)
    // >>> y = np.array([-1, 0, 1]).astype(np.float32)
    // >>> sum = x + y
    // >>> 0.5 * sum * (1 + np.vectorize(math.erf)(sum / np.sqrt(2)))
    std::vector<float> gold = {
        -0.04550027, -0.15865526, 0.0, -0.15865526, 0.0, 0.8413447, 0.0, 0.8413447, 1.9544997};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
