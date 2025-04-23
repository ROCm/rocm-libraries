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

TEST_CASE(gelu_fast_bias_test)
{
    migraphx::program p = read_onnx("gelu_fast_bias_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape shape{migraphx::shape::half_type, {3, 3}};
    std::vector<float> tmp           = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    std::vector<migraphx::half> x    = {tmp.begin(), tmp.end()};
    tmp                              = {-10, 0.5, 10, -2, 0.5, 2, -1, 0.5, 1};
    std::vector<migraphx::half> bias = {tmp.begin(), tmp.end()};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(shape, x.data());
    pp["y"] = migraphx::argument(shape, bias.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // gold values generated using numpy:
    // >>> import numpy as np
    // >>> import math
    // >>> x = np.array([[-1, -1, -1],[0, 0, 0], [1, 1, 1]]).astype(np.float16)
    // >>> bias = np.array([[-10, 0.5, 10], [-2, 0.5, 2], [-1, 0.5, 1]]).astype(np.float16)
    // >>> x = x + bias
    // >>> 0.5 * x * (1 + np.tanh(0.797885 * x + 0.035677 * np.power(x, 3)))
    tmp = {0.0, -0.1543, 9.0, -0.0454, 0.3457, 1.955, 0.0, 1.399, 1.955};

    std::vector<migraphx::half> gold = {tmp.begin(), tmp.end()};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
